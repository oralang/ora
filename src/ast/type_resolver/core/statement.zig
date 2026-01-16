// ============================================================================
// Statement Type Resolution
// ============================================================================
// Resolves types for all statement types (VariableDecl, Return, If, etc.)
// ============================================================================

const std = @import("std");
const ast = @import("../../../ast.zig");
const type_info_mod = @import("../../type_info.zig");
const TypeInfo = type_info_mod.TypeInfo;
const OraType = type_info_mod.OraType;
const CommonTypes = type_info_mod.CommonTypes;
const deinitOraType = type_info_mod.deinitOraType;
const state_mod = @import("../../../semantics/state.zig");
const SymbolTable = state_mod.SymbolTable;
const Scope = state_mod.Scope;
const Symbol = state_mod.Symbol;
const semantics = @import("../../../semantics.zig");
const copyOraTypeOwned = @import("../../../semantics/function_analyzer.zig").copyOraTypeOwned;
const TypeResolutionError = @import("../mod.zig").TypeResolutionError;
const Typed = @import("mod.zig").Typed;
const Effect = @import("mod.zig").Effect;
const SlotSet = @import("mod.zig").SlotSet;
const mergeEffects = @import("mod.zig").mergeEffects;
const takeEffect = @import("mod.zig").takeEffect;
const TypeContext = @import("mod.zig").TypeContext;
const validation = @import("../validation/mod.zig");
const refinements = @import("../refinements/mod.zig");
const utils = @import("../utils/mod.zig");
const log = @import("log");

const CoreResolver = @import("mod.zig").CoreResolver;
const expression = @import("expression.zig");

/// Compute block scope key from statement pointer.
/// Must match locals_binder.getBlockScopeKey for consistency.
/// block_id: 0 = primary block (then/body/try), 1 = secondary block (else/catch)
fn getBlockScopeKey(stmt: *const ast.Statements.StmtNode, block_id: usize) usize {
    return @intFromPtr(stmt) * 4 + block_id;
}

fn prependAssumeToBlock(
    self: *CoreResolver,
    block: *ast.Statements.BlockNode,
    condition: ast.Expressions.ExprNode,
    span: ast.SourceSpan,
) !void {
    // Use AST arena allocator - same as parser, freed together at end
    var new_statements = try self.type_storage_allocator.alloc(ast.Statements.StmtNode, block.statements.len + 1);
    new_statements[0] = ast.Statements.StmtNode{
        .Assume = .{
            .condition = condition,
            .span = span,
        },
    };
    std.mem.copyForwards(ast.Statements.StmtNode, new_statements[1..], block.statements);
    block.statements = new_statements;
    // No need to track ownership - arena frees everything at once
}

fn buildNotExpr(
    _: *CoreResolver,
    expr: *ast.Expressions.ExprNode,
    span: ast.SourceSpan,
) ast.Expressions.ExprNode {
    // Build UnaryExpr inline without heap allocation
    // The operand pointer is stored in the returned value
    return ast.Expressions.ExprNode{ .Unary = ast.Expressions.UnaryExpr{
        .operator = .Bang,
        .operand = expr,
        .type_info = TypeInfo.unknown(),
        .span = span,
    } };
}

fn buildSwitchAssumeExpr(
    self: *CoreResolver,
    condition: *ast.Expressions.ExprNode,
    pattern: *const ast.Expressions.SwitchPattern,
    span: ast.SourceSpan,
) !?*ast.Expressions.ExprNode {
    switch (pattern.*) {
        .Else => return null,
        .Literal => |lit| {
            const lit_node = try self.allocator.create(ast.Expressions.ExprNode);
            lit_node.* = .{ .Literal = lit.value };
            return try ast.Expressions.createBinaryExpr(self.allocator, condition, .EqualEqual, lit_node, span);
        },
        .EnumValue => |ev| {
            if (ev.enum_name.len == 0) return null;
            const enum_node = try self.allocator.create(ast.Expressions.ExprNode);
            enum_node.* = .{ .EnumLiteral = .{
                .enum_name = ev.enum_name,
                .variant_name = ev.variant_name,
                .span = span,
            } };
            return try ast.Expressions.createBinaryExpr(self.allocator, condition, .EqualEqual, enum_node, span);
        },
        .Range => |range| {
            const start_cmp = try ast.Expressions.createBinaryExpr(self.allocator, condition, .GreaterEqual, range.start, span);
            const end_op: ast.Expressions.BinaryOp = if (range.inclusive) .LessEqual else .Less;
            const end_cmp = try ast.Expressions.createBinaryExpr(self.allocator, condition, end_op, range.end, span);
            return try ast.Expressions.createBinaryExpr(self.allocator, start_cmp, .And, end_cmp, span);
        },
    }
}

/// Resolve types for a statement
pub fn resolveStatement(
    self: *CoreResolver,
    stmt: *ast.Statements.StmtNode,
    context: TypeContext,
) TypeResolutionError!Typed {
    return switch (stmt.*) {
        .VariableDecl => |*var_decl| resolveVariableDecl(self, var_decl, context),
        .Return => |*ret| resolveReturn(self, ret, context),
        .If => |*if_stmt| resolveIf(self, stmt, if_stmt, context),
        .While => |*while_stmt| resolveWhile(self, stmt, while_stmt, context),
        .ForLoop => |*for_stmt| resolveFor(self, stmt, for_stmt, context),
        .TryBlock => |*try_block| resolveTryBlock(self, stmt, try_block, context),
        .Expr => |*expr_stmt| resolveExpressionStmt(self, expr_stmt, context),
        .Break => |*break_stmt| resolveBreak(self, break_stmt, context),
        .Continue => |*continue_stmt| resolveContinue(self, continue_stmt, context),
        .Switch => |*switch_stmt| resolveSwitch(self, switch_stmt, context),
        .Log => |*log_stmt| resolveLog(self, log_stmt, context),
        else => {
            // unhandled statement types - return unknown
            return Typed.init(TypeInfo.unknown(), Effect.pure(), self.allocator);
        },
    };
}

// ============================================================================
// Statement Resolvers
// ============================================================================

fn resolveVariableDecl(
    self: *CoreResolver,
    var_decl: *ast.Statements.VariableDeclNode,
    _: TypeContext,
) TypeResolutionError!Typed {
    // fix custom type names: if parser assumed struct_type but it's actually enum_type
    if (var_decl.type_info.ora_type) |ot| {
        if (ot == .struct_type) {
            const type_name = ot.struct_type;
            // look up symbol to see if it's actually an enum
            // search from root scope to find type declarations
            const symbol = SymbolTable.findUp(@as(?*const Scope, @ptrCast(self.symbol_table.root)), type_name);
            if (symbol) |sym| {
                if (sym.kind == .Enum) {
                    // fix: change struct_type to enum_type
                    var_decl.type_info.ora_type = OraType{ .enum_type = type_name };
                    var_decl.type_info.category = .Enum;
                }
            }
        }
        // ensure category matches ora_type if present (fixes refinement types)
        var derived_category = ot.getCategory();
        if (ot == ._union and ot._union.len > 0 and ot._union[0] == .error_union) {
            derived_category = .ErrorUnion;
        }
        var_decl.type_info.category = derived_category;
    }

    var init_effect = Effect.pure();
    if (var_decl.value) |value_expr| {
        // ensure category matches ora_type if present (fixes refinement types)
        if (var_decl.type_info.ora_type) |ot| {
            var derived_category = ot.getCategory();
            if (ot == ._union and ot._union.len > 0 and ot._union[0] == .error_union) {
                derived_category = .ErrorUnion;
            }
            var_decl.type_info.category = derived_category;
        }
        // if type is unknown, infer it from the initializer expression
        if (!var_decl.type_info.isResolved()) {
            // synthesize type from expression
            var typed = try expression.synthExpr(self, value_expr);
            defer typed.deinit(self.allocator);
            init_effect = takeEffect(&typed);

            // assign inferred type to variable
            var_decl.type_info = typed.ty;

            // If inside a try block and the initializer is an error union, record it
            if (self.in_try_block) {
                const is_error_union = typed.ty.category == .ErrorUnion or
                    (typed.ty.category == .Union and typed.ty.ora_type != null and
                    typed.ty.ora_type.? == ._union and typed.ty.ora_type.?._union.len > 0 and
                    typed.ty.ora_type.?._union[0] == .error_union);
                if (is_error_union) {
                    self.last_try_error_union = typed.ty;
                }
            }
        } else {
            // type is explicit, check expression against it
            var checked = try expression.checkExpr(self, value_expr, var_decl.type_info);
            defer checked.deinit(self.allocator);
            init_effect = takeEffect(&checked);

            // If inside a try block and the initializer is an error union, record it
            if (self.in_try_block) {
                const is_error_union = checked.ty.category == .ErrorUnion or
                    (checked.ty.category == .Union and checked.ty.ora_type != null and
                    checked.ty.ora_type.? == ._union and checked.ty.ora_type.?._union.len > 0 and
                    checked.ty.ora_type.?._union[0] == .error_union);
                if (is_error_union) {
                    self.last_try_error_union = checked.ty;
                }
            }

            // validate refinement types if present
            if (var_decl.type_info.ora_type) |target_ora_type| {
                // validate refinement type structure
                try self.refinement_system.validate(target_ora_type);

                // validate constant literal values against refinement constraints
                try validateLiteralAgainstRefinement(self, value_expr, target_ora_type);

                // check guard optimizations: skip guard if optimization applies
                var_decl.skip_guard = shouldSkipGuard(self, value_expr, target_ora_type);
            }
        }

        const target_region = var_decl.region;
        const source_region = expression.inferExprRegion(self, value_expr);
        var target_expr = ast.Expressions.ExprNode{ .Identifier = .{
            .name = var_decl.name,
            .type_info = var_decl.type_info,
            .span = var_decl.span,
        } };
        if (!expression.isRegionAssignmentAllowed(target_region, source_region, &target_expr)) {
            log.debug(
                "[resolveVariableDecl] RegionMismatch: '{s}' target={s} source={s}\n",
                .{ var_decl.name, @tagName(target_region), @tagName(source_region) },
            );
            return TypeResolutionError.RegionMismatch;
        }

        // record implicit region effects for the initializer assignment
        const region_eff = try expression.regionEffectForAssignment(
            self,
            target_region,
            source_region,
            &target_expr,
            value_expr,
        );
        mergeEffects(self.allocator, &init_effect, region_eff);
    } else {
        // no initializer - type must be explicit
        if (!var_decl.type_info.isResolved()) {
            return TypeResolutionError.UnresolvedType;
        }
    }

    // add variable to symbol table AFTER type is resolved
    // ensure type is resolved - if it has ora_type, derive category if needed
    var final_type = var_decl.type_info;
    if (final_type.ora_type) |ot| {
        // ensure category matches the ora_type (refinement types inherit base category)
        const derived_category = ot.getCategory();
        // always update category to match ora_type to ensure isResolved() works
        final_type.category = derived_category;
        // also update the original type_info to ensure it's resolved
        var_decl.type_info.category = derived_category;
    }
    var_decl.type_info.region = var_decl.region;
    final_type.region = var_decl.region;

    // type should be resolved now - if not, there's a bug
    if (!final_type.isResolved()) {
        return TypeResolutionError.UnresolvedType;
    }

    // copy the type if it contains pointers (refinement types, slices, etc.)
    // to ensure it's properly owned by the symbol table
    // ensure final_type has ora_type before proceeding
    if (final_type.ora_type == null) {
        return TypeResolutionError.UnresolvedType;
    }
    var stored_type = final_type;
    var typ_owned_flag = false;
    const ot = final_type.ora_type.?;
    // check if this type needs copying (has pointers)
    const needs_copy = switch (ot) {
        .min_value, .max_value, .in_range, .scaled, .exact, .slice, .error_union, .array, .map, .tuple, .anonymous_struct, ._union, .function => true,
        else => false,
    };

    // only set typ_owned = true if we're storing in a function scope, not a block scope
    // block scopes are temporary and shouldn't own types (they point to arena memory)
    if (needs_copy) {
        if (copyOraTypeOwned(self.type_storage_allocator, ot)) |copied_ora_type| {
            // derive category from the copied type to ensure it's correct
            const derived_category = copied_ora_type.getCategory();
            stored_type = TypeInfo{
                .category = derived_category,
                .ora_type = copied_ora_type,
                .source = final_type.source,
                .span = final_type.span,
                .region = var_decl.region,
            };
            // keep copied refinement types alive for MLIR lowering; symbol table
            // deinit would otherwise free them before MLIR runs.
            typ_owned_flag = false;
        } else |_| {
            // if copying fails, we cannot safely store this type as it may contain pointers to temporary memory
            // this should be rare (usually only out of memory), but we need to handle it
            // we cannot proceed without copying refinement types safely
            return TypeResolutionError.UnresolvedType;
        }
    } else {
        // no copying needed, or we're in a block scope (don't own types in block scopes)
        // ensure category is correct
        stored_type.category = ot.getCategory();
        stored_type.region = var_decl.region;
    }

    // final check: ensure stored type is resolved
    // double-check that ora_type is set and category matches
    if (stored_type.ora_type) |stored_ot| {
        stored_type.category = stored_ot.getCategory();
    }
    if (!stored_type.isResolved()) {
        return TypeResolutionError.UnresolvedType;
    }
    // update the AST node to point at the stored type so later passes have stable pointers
    var_decl.type_info = stored_type;
    try self.validateErrorUnionType(stored_type);

    // final verification before storing
    if (stored_type.ora_type == null) {
        log.debug("[resolveVariableDecl] CRITICAL: Variable '{s}' has null ora_type before storing!\n", .{var_decl.name});
        return TypeResolutionError.UnresolvedType;
    }
    const symbol = semantics.state.Symbol{
        .name = var_decl.name,
        .kind = .Var, // SymbolKind only has .Var
        .typ = stored_type,
        .span = var_decl.span,
        .mutable = var_decl.kind == .Var,
        .region = var_decl.region,
        .typ_owned = typ_owned_flag,
    };

    // track if we successfully stored the type to avoid leaks
    var type_stored = false;
    errdefer {
        // if we allocated a type but failed to store it, deallocate it
        if (typ_owned_flag and !type_stored) {
            if (stored_type.ora_type) |*ora_type| {
                const type_info = @import("../../type_info.zig");
                type_info.deinitOraType(self.allocator, @constCast(ora_type));
            }
        }
    }

    if (self.current_scope) |scope| {
        // find the actual scope where the symbol is stored (could be in a block scope)
        // this ensures we update it in the correct scope
        if (self.symbol_table.findScopeContainingSafe(scope, var_decl.name)) |found| {
            // found the symbol - update it in the scope where it's actually stored
            const old_symbol = &found.scope.symbols.items[found.idx];
            const type_info = @import("../../type_info.zig");
            // only deallocate if the old type was owned AND has a valid ora_type
            if (old_symbol.typ_owned) {
                if (old_symbol.typ) |*old_typ| {
                    if (old_typ.ora_type != null) {
                        type_info.deinitTypeInfo(self.allocator, old_typ);
                    }
                }
            }
            // only set typ_owned=true if this scope is a function scope
            // block scopes should never own types (they point to arena memory)
            old_symbol.typ = stored_type;
            old_symbol.typ_owned = typ_owned_flag and self.symbol_table.isFunctionScope(found.scope);
            type_stored = true;
        } else {
            // symbol doesn't exist yet - declare it in the current scope
            // only set typ_owned if we're in a function scope
            var new_symbol = symbol;
            new_symbol.typ_owned = typ_owned_flag and self.symbol_table.isFunctionScope(scope);
            if (self.symbol_table.declare(scope, new_symbol)) |_| {
                // duplicate declaration - this shouldn't happen if findScopeContaining worked
                // but handle it gracefully
                type_stored = true;
            } else |_| {
                // failed to declare - this is an error
                return TypeResolutionError.UnresolvedType;
            }
        }
    } else {
        // no current scope - this shouldn't happen during normal resolution
        // but handle it gracefully by not storing the type
        log.debug("[resolveVariableDecl] WARNING: No current scope for variable '{s}'\n", .{var_decl.name});
        // deallocate the copied type since we can't store it
        if (typ_owned_flag) {
            if (stored_type.ora_type) |*ora_type| {
                const type_info = @import("../../type_info.zig");
                type_info.deinitOraType(self.allocator, @constCast(ora_type));
                type_stored = true; // Mark as "handled" to prevent errdefer from double-freeing
            }
        }
    }

    // critical: If we copied a type but didn't store it, we have a leak
    // this should never happen, but if it does, we need to clean up
    if (typ_owned_flag and !type_stored) {
        log.debug("[resolveVariableDecl] CRITICAL: Variable '{s}' - copied type was not stored! Deallocating to prevent leak.\n", .{var_decl.name});
        if (stored_type.ora_type) |*ora_type| {
            const type_info = @import("../../type_info.zig");
            type_info.deinitOraType(self.allocator, @constCast(ora_type));
        }
    }

    const combined_eff = init_effect;

    return Typed.init(var_decl.type_info, combined_eff, self.allocator);
}

fn resolveReturn(
    self: *CoreResolver,
    ret: *ast.Statements.ReturnNode,
    context: TypeContext,
) TypeResolutionError!Typed {
    var ret_effect = Effect.pure();
    if (ret.value) |*value_expr| {
        // check expression against expected return type
        if (context.function_return_type) |return_type| {
            var checked = try expression.checkExpr(self, value_expr, return_type);
            defer checked.deinit(self.allocator);
            ret_effect = takeEffect(&checked);

            // check guard optimizations: skip guard if optimization applies
            if (return_type.ora_type) |target_ora_type| {
                ret.skip_guard = shouldSkipGuard(self, value_expr, target_ora_type);
            }
        } else {
            // no return type expected - synthesize
            var typed = try expression.synthExpr(self, value_expr);
            defer typed.deinit(self.allocator);
            ret_effect = takeEffect(&typed);
        }
    }

    return Typed.init(TypeInfo.unknown(), ret_effect, self.allocator);
}

fn resolveIf(
    self: *CoreResolver,
    stmt: *const ast.Statements.StmtNode,
    if_stmt: *ast.Statements.IfNode,
    context: TypeContext,
) TypeResolutionError!Typed {
    // resolve condition (should be bool)
    var condition_typed = try expression.synthExpr(self, &if_stmt.condition);
    defer condition_typed.deinit(self.allocator);
    try validateBoolCondition(condition_typed.ty);
    var combined_eff = takeEffect(&condition_typed);

    var then_refinements = try deriveBranchRefinements(self, &if_stmt.condition, .Then);
    defer then_refinements.deinit(self.allocator);

    const prev_scope = self.current_scope;

    // resolve then branch statements - use stmt pointer for consistent key
    const then_scope_key = getBlockScopeKey(stmt, 0);
    const then_scope = self.symbol_table.block_scopes.get(then_scope_key) orelse self.current_scope;
    self.current_scope = then_scope;
    try prependAssumeToBlock(self, &if_stmt.then_branch, if_stmt.condition, if_stmt.span);
    for (then_refinements.items) |*ref| {
        applyRefinementToScope(self, then_scope, ref) catch |err| switch (err) {
            error.SymbolNotFound => {},
            else => return TypeResolutionError.TypeMismatch,
        };
    }
    for (if_stmt.then_branch.statements) |*then_stmt| {
        var stmt_typed = try resolveStatement(self, then_stmt, context);
        defer stmt_typed.deinit(self.allocator);
        const eff = takeEffect(&stmt_typed);
        mergeEffects(self.allocator, &combined_eff, eff);
    }

    // resolve else branch if present
    if (if_stmt.else_branch) |*else_branch| {
        // Only create else refinements if there's an else branch to apply them to
        var else_refinements = try deriveBranchRefinements(self, &if_stmt.condition, .Else);
        defer else_refinements.deinit(self.allocator);

        const else_scope_key = getBlockScopeKey(stmt, 1);
        const else_scope = self.symbol_table.block_scopes.get(else_scope_key) orelse self.current_scope;
        self.current_scope = else_scope;
        const negated = buildNotExpr(self, &if_stmt.condition, if_stmt.span);
        try prependAssumeToBlock(self, else_branch, negated, if_stmt.span);
        for (else_refinements.items) |*ref| {
            applyRefinementToScope(self, else_scope, ref) catch |err| switch (err) {
                error.SymbolNotFound => {},
                else => return TypeResolutionError.TypeMismatch,
            };
        }
        for (else_branch.statements) |*else_stmt| {
            var stmt_typed = try resolveStatement(self, else_stmt, context);
            defer stmt_typed.deinit(self.allocator);
            const eff = takeEffect(&stmt_typed);
            mergeEffects(self.allocator, &combined_eff, eff);
        }
    }

    self.current_scope = prev_scope;

    return Typed.init(TypeInfo.unknown(), combined_eff, self.allocator);
}

fn resolveWhile(
    self: *CoreResolver,
    stmt: *const ast.Statements.StmtNode,
    while_stmt: *ast.Statements.WhileNode,
    context: TypeContext,
) TypeResolutionError!Typed {
    // resolve condition
    var condition_typed = try expression.synthExpr(self, &while_stmt.condition);
    defer condition_typed.deinit(self.allocator);
    try validateBoolCondition(condition_typed.ty);
    var combined_eff = takeEffect(&condition_typed);

    var then_refinements = try deriveBranchRefinements(self, &while_stmt.condition, .Then);
    defer then_refinements.deinit(self.allocator);

    const prev_scope = self.current_scope;
    const body_scope_key = getBlockScopeKey(stmt, 0);
    const body_scope = self.symbol_table.block_scopes.get(body_scope_key) orelse self.current_scope;
    self.current_scope = body_scope;
    defer {
        self.current_scope = prev_scope;
    }

    try prependAssumeToBlock(self, &while_stmt.body, while_stmt.condition, while_stmt.span);
    for (then_refinements.items) |*ref| {
        applyRefinementToScope(self, body_scope, ref) catch |err| switch (err) {
            error.SymbolNotFound => {},
            else => return TypeResolutionError.TypeMismatch,
        };
    }
    for (while_stmt.invariants) |*inv| {
        var inv_typed = try expression.synthExpr(self, inv);
        defer inv_typed.deinit(self.allocator);
        try validateBoolCondition(inv_typed.ty);
        try prependAssumeToBlock(self, &while_stmt.body, inv.*, while_stmt.span);
        var inv_refinements = try deriveBranchRefinements(self, inv, .Then);
        defer inv_refinements.deinit(self.allocator);
        for (inv_refinements.items) |*ref| {
            applyRefinementToScope(self, body_scope, ref) catch |err| switch (err) {
                error.SymbolNotFound => {},
                else => return TypeResolutionError.TypeMismatch,
            };
        }
    }

    // resolve body
    for (while_stmt.body.statements) |*body_stmt| {
        var stmt_typed = try resolveStatement(self, body_stmt, context);
        defer stmt_typed.deinit(self.allocator);
        const eff = takeEffect(&stmt_typed);
        mergeEffects(self.allocator, &combined_eff, eff);
    }

    return Typed.init(TypeInfo.unknown(), combined_eff, self.allocator);
}

fn resolveFor(
    self: *CoreResolver,
    stmt: *const ast.Statements.StmtNode,
    for_stmt: *ast.Statements.ForLoopNode,
    context: TypeContext,
) TypeResolutionError!Typed {
    // resolve iterable expression to get its type
    var iterable_typed = try expression.synthExpr(self, &for_stmt.iterable);
    defer iterable_typed.deinit(self.allocator);
    var combined_eff = takeEffect(&iterable_typed);

    // determine element type from iterable type
    const element_type: TypeInfo = if (iterable_typed.ty.ora_type) |iterable_ora_type| blk: {
        const elem_type = switch (iterable_ora_type) {
            .array => |arr| arr.elem.*,
            .slice => |elem_ptr| elem_ptr.*,
            .map => |m| m.value.*,
            // for range-style loops (integer), element type is the same as iterable
            else => iterable_ora_type,
        };
        const elem_category = elem_type.getCategory();
        break :blk TypeInfo{
            .category = elem_category,
            .ora_type = elem_type,
            .source = .inferred,
            .span = for_stmt.span,
        };
    } else if (iterable_typed.ty.category == .Integer) blk: {
        // range expressions with integer literals (e.g., 0..20) resolve to Integer category
        // but have null ora_type. Infer u256 as the element type.
        break :blk CommonTypes.u256_type();
    } else TypeInfo.unknown();

    // resolve loop pattern variables and update their types in symbol table
    // loop variables are declared in the body scope, not the function scope
    const body_scope_key = getBlockScopeKey(stmt, 0);
    const body_scope = self.symbol_table.block_scopes.get(body_scope_key) orelse self.current_scope;

    if (body_scope) |scope| {
        switch (for_stmt.pattern) {
            .Single => |s| {
                // update the symbol type for the loop variable
                // loop variables are typically simple types, not owned
                if (element_type.ora_type != null) {
                    _ = self.symbol_table.updateSymbolType(scope, s.name, element_type, false) catch {
                        // symbol might not exist yet, that's okay - it will be created during lowering
                    };
                }
            },
            .IndexPair => |p| {
                // update item variable type
                // loop variables are typically simple types, not owned
                if (element_type.ora_type != null) {
                    _ = self.symbol_table.updateSymbolType(scope, p.item, element_type, false) catch {};
                }
                // index variable is always u256 (or mlir index type) - simple type, not owned
                const index_type = CommonTypes.u256_type();
                _ = self.symbol_table.updateSymbolType(scope, p.index, index_type, false) catch {};
            },
            .Destructured => |d| {
                // for destructuring, we need to extract field types from the element type
                // this is complex and depends on the destructuring pattern
                // for now, use element type for all fields (simplified)
                // loop variables are typically simple types, not owned
                if (element_type.ora_type != null) {
                    switch (d.pattern) {
                        .Struct => |fields| {
                            for (fields) |field| {
                                _ = self.symbol_table.updateSymbolType(scope, field.variable, element_type, false) catch {};
                            }
                        },
                        .Tuple => |names| {
                            for (names) |name| {
                                _ = self.symbol_table.updateSymbolType(scope, name, element_type, false) catch {};
                            }
                        },
                        .Array => |names| {
                            for (names) |name| {
                                _ = self.symbol_table.updateSymbolType(scope, name, element_type, false) catch {};
                            }
                        },
                    }
                }
            },
        }
    }

    // resolve loop body
    // set current_scope to body_scope so loop variables can be found
    const prev_scope = self.current_scope;
    if (body_scope) |bs| {
        self.current_scope = bs;
    }
    defer {
        self.current_scope = prev_scope;
    }

    for (for_stmt.invariants) |*inv| {
        var inv_typed = try expression.synthExpr(self, inv);
        defer inv_typed.deinit(self.allocator);
        try validateBoolCondition(inv_typed.ty);
        try prependAssumeToBlock(self, &for_stmt.body, inv.*, for_stmt.span);
        var inv_refinements = try deriveBranchRefinements(self, inv, .Then);
        defer inv_refinements.deinit(self.allocator);
        for (inv_refinements.items) |*ref| {
            applyRefinementToScope(self, body_scope, ref) catch |err| switch (err) {
                error.SymbolNotFound => {},
                else => return TypeResolutionError.TypeMismatch,
            };
        }
    }

    for (for_stmt.body.statements) |*body_stmt| {
        var stmt_typed = try resolveStatement(self, body_stmt, context);
        defer stmt_typed.deinit(self.allocator);
        const eff = takeEffect(&stmt_typed);
        mergeEffects(self.allocator, &combined_eff, eff);
    }

    return Typed.init(TypeInfo.unknown(), combined_eff, self.allocator);
}

fn validateBoolCondition(condition_type: TypeInfo) TypeResolutionError!void {
    if (!condition_type.isResolved()) {
        return TypeResolutionError.UnresolvedType;
    }
    if (condition_type.ora_type) |ot| {
        if (ot != .bool) {
            return TypeResolutionError.TypeMismatch;
        }
    } else if (condition_type.category != .Bool) {
        return TypeResolutionError.TypeMismatch;
    }
}

fn resolveTryBlock(
    self: *CoreResolver,
    stmt: *const ast.Statements.StmtNode,
    try_block: *ast.Statements.TryBlockNode,
    context: TypeContext,
) TypeResolutionError!Typed {
    const prev_try = self.in_try_block;
    const prev_error_union = self.last_try_error_union;

    // resolve try block statements - use stmt pointer for consistent key
    const prev_scope = self.current_scope;
    const try_block_key = getBlockScopeKey(stmt, 0);
    if (self.symbol_table.block_scopes.get(try_block_key)) |try_scope| {
        self.current_scope = try_scope;
    }
    defer {
        self.current_scope = prev_scope;
    }

    var combined_eff = Effect.pure();
    self.last_try_error_union = null;
    self.in_try_block = true;
    defer {
        self.in_try_block = prev_try;
    }
    for (try_block.try_block.statements) |*inner_stmt| {
        var stmt_typed = try resolveStatement(self, inner_stmt, context);
        defer stmt_typed.deinit(self.allocator);
        const eff = takeEffect(&stmt_typed);
        mergeEffects(self.allocator, &combined_eff, eff);
    }

    // resolve catch block if present
    if (try_block.catch_block) |*catch_block| {
        // set scope for catch block so error variable is accessible
        const catch_block_key = getBlockScopeKey(stmt, 1);
        if (self.symbol_table.block_scopes.get(catch_block_key)) |catch_scope| {
            self.current_scope = catch_scope;
        }
        defer {
            self.current_scope = prev_scope;
        }

        self.in_try_block = false;
        if (catch_block.error_variable) |ename| {
            const inferred_error_union = self.last_try_error_union orelse return TypeResolutionError.InvalidErrorUsage;
            // update symbol type in catch scope
            if (self.current_scope) |scope| {
                _ = self.symbol_table.updateSymbolType(scope, ename, inferred_error_union, false) catch {
                    return TypeResolutionError.UnresolvedType;
                };
            }
        }
        for (catch_block.block.statements) |*catch_stmt| {
            var stmt_typed = try resolveStatement(self, catch_stmt, context);
            defer stmt_typed.deinit(self.allocator);
            const eff = takeEffect(&stmt_typed);
            mergeEffects(self.allocator, &combined_eff, eff);
        }
    }

    self.last_try_error_union = prev_error_union;
    return Typed.init(TypeInfo.unknown(), combined_eff, self.allocator);
}

fn resolveExpressionStmt(
    self: *CoreResolver,
    expr_stmt: *ast.Expressions.ExprNode,
    context: TypeContext,
) TypeResolutionError!Typed {
    _ = context;
    var typed = try expression.synthExpr(self, expr_stmt);
    const ty = typed.ty;
    const is_error_union = ty.category == .ErrorUnion or (ty.category == .Union and ty.ora_type != null and ty.ora_type.? == ._union and ty.ora_type.?._union.len > 0 and ty.ora_type.?._union[0] == .error_union);
    if (is_error_union) {
        // Inside a try block, error union expressions are allowed - the catch block handles errors
        if (self.in_try_block) {
            // Record the error union type for the catch block's error variable
            self.last_try_error_union = ty;
            return typed;
        }
        typed.deinit(self.allocator);
        return TypeResolutionError.ErrorUnionOutsideTry;
    }
    return typed;
}

fn resolveBreak(
    self: *CoreResolver,
    break_stmt: *ast.Statements.BreakNode,
    context: TypeContext,
) TypeResolutionError!Typed {
    // if break has a value expression, synthesize its type
    if (break_stmt.value) |value_expr| {
        var typed = try expression.synthExpr(self, value_expr);
        defer typed.deinit(self.allocator);
        const eff = takeEffect(&typed);
        // todo: Validate that break value type matches expected type from labeled block/switch
        // this requires tracking the expected type from the enclosing labeled block or switch expression
        return Typed.init(TypeInfo.unknown(), eff, self.allocator);
    }
    _ = context; // Context may contain expected break value type in the future
    return Typed.init(TypeInfo.unknown(), Effect.pure(), self.allocator);
}

fn resolveContinue(
    self: *CoreResolver,
    continue_stmt: *ast.Statements.ContinueNode,
    context: TypeContext,
) TypeResolutionError!Typed {
    // if continue has a value expression (for labeled switch continue), synthesize its type
    if (continue_stmt.value) |value_expr| {
        var typed = try expression.synthExpr(self, value_expr);
        defer typed.deinit(self.allocator);
        const eff = takeEffect(&typed);
        // todo: Validate that continue value type matches expected type from labeled switch
        return Typed.init(TypeInfo.unknown(), eff, self.allocator);
    }
    _ = context; // Context may contain expected continue value type in the future
    return Typed.init(TypeInfo.unknown(), Effect.pure(), self.allocator);
}

fn resolveSwitch(
    self: *CoreResolver,
    switch_stmt: *ast.Statements.SwitchNode,
    context: TypeContext,
) TypeResolutionError!Typed {
    // synthesize type of the switch condition expression
    var condition_typed = try expression.synthExpr(self, &switch_stmt.condition);
    defer condition_typed.deinit(self.allocator);
    var combined_eff = takeEffect(&condition_typed);

    const condition_type = condition_typed.ty;

    // validate that case patterns match the condition type
    for (switch_stmt.cases) |*case| {
        try validateSwitchPattern(self, &case.pattern, condition_type, case.span);
    }

    const prev_scope = self.current_scope;
    const condition_ident = extractIdentifierName(&switch_stmt.condition);

    // resolve types for each case body
    for (switch_stmt.cases) |*case| {
        switch (case.body) {
            .Expression => |case_expr| {
                var case_typed = try expression.synthExpr(self, case_expr);
                defer case_typed.deinit(self.allocator);
                const eff = takeEffect(&case_typed);
                mergeEffects(self.allocator, &combined_eff, eff);
            },
            .Block => |*case_block| {
                const case_scope_key: usize = @intFromPtr(case_block);
                const case_scope = self.symbol_table.block_scopes.get(case_scope_key) orelse self.current_scope;
                self.current_scope = case_scope;
                if (try buildSwitchAssumeExpr(self, &switch_stmt.condition, &case.pattern, case.span)) |assume_expr| {
                    try prependAssumeToBlock(self, case_block, assume_expr.*, case.span);
                }
                if (condition_ident) |name| {
                    if (try deriveSwitchRefinement(self, name, &case.pattern)) |r| {
                        var ref = r;
                        applyRefinementToScope(self, case_scope, &ref) catch |err| switch (err) {
                            error.SymbolNotFound => {},
                            else => return TypeResolutionError.TypeMismatch,
                        };
                    }
                }
                for (case_block.statements) |*case_stmt| {
                    var stmt_typed = try resolveStatement(self, case_stmt, context);
                    defer stmt_typed.deinit(self.allocator);
                    const eff = takeEffect(&stmt_typed);
                    mergeEffects(self.allocator, &combined_eff, eff);
                }
            },
            .LabeledBlock => |*labeled| {
                const labeled_scope_key: usize = @intFromPtr(&labeled.block);
                const labeled_scope = self.symbol_table.block_scopes.get(labeled_scope_key) orelse self.current_scope;
                self.current_scope = labeled_scope;
                if (try buildSwitchAssumeExpr(self, &switch_stmt.condition, &case.pattern, case.span)) |assume_expr| {
                    try prependAssumeToBlock(self, &labeled.block, assume_expr.*, case.span);
                }
                if (condition_ident) |name| {
                    if (try deriveSwitchRefinement(self, name, &case.pattern)) |r| {
                        var ref = r;
                        applyRefinementToScope(self, labeled_scope, &ref) catch |err| switch (err) {
                            error.SymbolNotFound => {},
                            else => return TypeResolutionError.TypeMismatch,
                        };
                    }
                }
                for (labeled.block.statements) |*case_stmt| {
                    var stmt_typed = try resolveStatement(self, case_stmt, context);
                    defer stmt_typed.deinit(self.allocator);
                    const eff = takeEffect(&stmt_typed);
                    mergeEffects(self.allocator, &combined_eff, eff);
                }
            },
        }
        self.current_scope = prev_scope;
    }
    self.current_scope = prev_scope;

    // resolve types for default case if present
    if (switch_stmt.default_case) |*default_block| {
        const default_scope_key: usize = @intFromPtr(default_block);
        const default_scope = self.symbol_table.block_scopes.get(default_scope_key) orelse self.current_scope;
        self.current_scope = default_scope;
        for (default_block.statements) |*default_stmt| {
            var stmt_typed = try resolveStatement(self, default_stmt, context);
            defer stmt_typed.deinit(self.allocator);
            const eff = takeEffect(&stmt_typed);
            mergeEffects(self.allocator, &combined_eff, eff);
        }
    }
    self.current_scope = prev_scope;

    return Typed.init(TypeInfo.unknown(), combined_eff, self.allocator);
}

/// Validate that a switch pattern matches the condition type
fn validateSwitchPattern(
    self: *CoreResolver,
    pattern: *const ast.Expressions.SwitchPattern,
    condition_type: TypeInfo,
    pattern_span: ast.SourceSpan,
) TypeResolutionError!void {
    if (!condition_type.isResolved()) {
        // condition type not resolved - skip validation
        return;
    }

    switch (pattern.*) {
        .Else => {
            // else patterns are always valid
        },
        .Literal => |lit| {
            // validate literal type matches condition type category
            const lit_type_info = getLiteralTypeInfo(lit.value);
            if (!lit_type_info.isResolved()) {
                return TypeResolutionError.TypeMismatch;
            }

            // pattern literal must be assignable to condition type
            if (!self.validation.isAssignable(lit_type_info, condition_type)) {
                log.debug(
                    "[type_resolver] Switch pattern type mismatch: pattern category {s}, condition category {s}\n",
                    .{ @tagName(lit_type_info.category), @tagName(condition_type.category) },
                );
                return TypeResolutionError.TypeMismatch;
            }
        },
        .Range => |range| {
            // validate range endpoints are compatible with condition type
            var start_typed = try expression.synthExpr(self, range.start);
            defer start_typed.deinit(self.allocator);
            var end_typed = try expression.synthExpr(self, range.end);
            defer end_typed.deinit(self.allocator);

            // both endpoints must be assignable to condition type
            if (!self.validation.isAssignable(start_typed.ty, condition_type)) {
                log.debug(
                    "[type_resolver] Switch range start type mismatch\n",
                    .{},
                );
                return TypeResolutionError.TypeMismatch;
            }
            if (!self.validation.isAssignable(end_typed.ty, condition_type)) {
                log.debug(
                    "[type_resolver] Switch range end type mismatch\n",
                    .{},
                );
                return TypeResolutionError.TypeMismatch;
            }
        },
        .EnumValue => |ev| {
            // validate enum value matches condition enum type
            if (condition_type.category == .Error or condition_type.category == .ErrorUnion) {
                // error switch: allow bare error tags (no enum name) that resolve to declared errors
                if (ev.enum_name.len == 0) {
                    if (self.symbol_table.safeFindUpOpt(self.current_scope, ev.variant_name)) |sym| {
                        if (sym.kind == .Error) return;
                    }
                    if (self.symbol_table.safeFindUpOpt(self.symbol_table.root, ev.variant_name)) |sym| {
                        if (sym.kind == .Error) return;
                    }
                }
                log.debug(
                    "[type_resolver] Switch error pattern did not match declared error: {s}\n",
                    .{ev.variant_name},
                );
                return TypeResolutionError.TypeMismatch;
            }
            if (condition_type.ora_type) |cond_ora_type| {
                switch (cond_ora_type) {
                    .enum_type => |enum_name| {
                        // if pattern provided a qualified enum name, it must match
                        if (ev.enum_name.len != 0 and !std.mem.eql(u8, ev.enum_name, enum_name)) {
                            log.debug(
                                "[type_resolver] Switch enum pattern enum name mismatch: got {s}, expected {s}\n",
                                .{ ev.enum_name, enum_name },
                            );
                            return TypeResolutionError.TypeMismatch;
                        }
                    },
                    else => {
                        log.debug(
                            "[type_resolver] Switch enum pattern used with non-enum condition\n",
                            .{},
                        );
                        return TypeResolutionError.TypeMismatch;
                    },
                }
            } else {
                log.debug(
                    "[type_resolver] Switch enum pattern used with unresolved condition type\n",
                    .{},
                );
                return TypeResolutionError.TypeMismatch;
            }
        },
    }
    _ = pattern_span; // Span available for future error reporting improvements
}

const BranchKind = enum { Then, Else };

const RefinementOverride = struct {
    name: []const u8,
    type_info: TypeInfo,
    typ_owned: bool,
};

const RefinementList = struct {
    items: []RefinementOverride,

    pub fn initEmpty() RefinementList {
        return .{ .items = &.{} };
    }

    pub fn deinit(self: *RefinementList, allocator: std.mem.Allocator) void {
        // Note: Types inside RefinementOverride are transferred to symbols via
        // applyRefinementToScope, so we don't free them here - the symbol table
        // owns them and will free them during its cleanup.
        if (self.items.len > 0) allocator.free(self.items);
        self.items = &.{};
    }
};

fn extractIdentifierName(expr: *ast.Expressions.ExprNode) ?[]const u8 {
    return switch (expr.*) {
        .Identifier => |id| id.name,
        else => null,
    };
}

fn getBaseOraType(self: *CoreResolver, ty: TypeInfo) ?OraType {
    if (ty.ora_type) |ot| {
        return self.utils.extractBaseType(ot) orelse ot;
    }
    return null;
}

fn buildRefinedType(
    self: *CoreResolver,
    base_type: OraType,
    min_val: ?u256,
    max_val: ?u256,
    span: ast.SourceSpan,
) !TypeInfo {
    // Use type_storage_allocator for types stored in symbols (survives past type resolution)
    const base_ptr = try self.type_storage_allocator.create(OraType);
    base_ptr.* = try copyOraTypeOwned(self.type_storage_allocator, base_type);

    const refined: OraType = if (min_val != null and max_val != null) blk: {
        break :blk OraType{ .in_range = .{ .base = base_ptr, .min = min_val.?, .max = max_val.? } };
    } else if (min_val != null) blk: {
        break :blk OraType{ .min_value = .{ .base = base_ptr, .min = min_val.? } };
    } else if (max_val != null) blk: {
        break :blk OraType{ .max_value = .{ .base = base_ptr, .max = max_val.? } };
    } else {
        return TypeInfo.unknown();
    };

    return TypeInfo{
        .category = base_type.getCategory(),
        .ora_type = refined,
        .source = .inferred,
        .span = span,
    };
}

fn applyRefinementToScope(
    self: *CoreResolver,
    scope: ?*Scope,
    refinement: *const RefinementOverride,
) !void {
    // Helper to free the refinement type if we own it and don't apply it
    // Note: Flow refinement types are allocated with type_storage_allocator
    const freeIfOwned = struct {
        fn call(allocator: std.mem.Allocator, ref: *const RefinementOverride) void {
            if (ref.typ_owned) {
                // Need mutable access to free the type's inner pointers
                var ti = ref.type_info;
                if (ti.ora_type) |*ot| {
                    deinitOraType(allocator, ot);
                }
            }
        }
    }.call;

    if (scope == null) {
        freeIfOwned(self.type_storage_allocator, refinement);
        return;
    }
    if (scope.?.findInCurrent(refinement.name)) |idx| {
        const existing = scope.?.symbols.items[idx];
        if (existing.typ) |existing_ty| {
            if (mergeRefinementTypes(self, existing_ty, refinement.type_info)) |merged| {
                try self.symbol_table.updateSymbolType(scope, refinement.name, merged, true);
            }
        }
        // Type was merged or couldn't merge - original not stored, free it
        freeIfOwned(self.type_storage_allocator, refinement);
        return;
    }
    const base_sym = SymbolTable.findUp(scope, refinement.name) orelse {
        // Symbol not found - type not applied, free it
        freeIfOwned(self.type_storage_allocator, refinement);
        return;
    };

    const shadow = Symbol{
        .name = refinement.name,
        .kind = base_sym.kind,
        .typ = refinement.type_info,
        .span = base_sym.span,
        .mutable = base_sym.mutable,
        .region = base_sym.region,
        .typ_owned = refinement.typ_owned,
        .is_flow_refinement = true, // Mark as flow-derived, not declared
    };
    _ = try self.symbol_table.declare(scope.?, shadow);
    // Type ownership transferred to symbol - don't free
}

fn deriveBranchRefinements(
    self: *CoreResolver,
    condition: *ast.Expressions.ExprNode,
    branch: BranchKind,
) !RefinementList {
    const bin = switch (condition.*) {
        .Binary => |*b| b,
        .Unary => |*u| {
            if (u.operator != .Bang) return RefinementList.initEmpty();
            const flipped = if (branch == .Then) BranchKind.Else else BranchKind.Then;
            return deriveBranchRefinements(self, u.operand, flipped);
        },
        else => return RefinementList.initEmpty(),
    };

    if (bin.operator == .And or bin.operator == .Or) {
        if (bin.operator == .And and branch == .Then) {
            return mergeRefinementLists(
                self,
                try deriveBranchRefinements(self, bin.lhs, .Then),
                try deriveBranchRefinements(self, bin.rhs, .Then),
            );
        }
        if (bin.operator == .Or and branch == .Else) {
            return mergeRefinementLists(
                self,
                try deriveBranchRefinements(self, bin.lhs, .Else),
                try deriveBranchRefinements(self, bin.rhs, .Else),
            );
        }
        // Conservative for (A && B) else or (A || B) then
        return RefinementList.initEmpty();
    }

    var ident = extractIdentifierName(bin.lhs);
    var value_expr = bin.rhs;
    var op = bin.operator;

    if (ident == null) {
        ident = extractIdentifierName(bin.rhs);
        value_expr = bin.lhs;
        op = invertComparisonOperator(op) orelse return RefinementList.initEmpty();
    }
    if (ident == null) return RefinementList.initEmpty();

    const const_val = try self.utils.evaluateConstantExpression(value_expr);
    const value = const_val orelse return RefinementList.initEmpty();

    const sym = SymbolTable.findUp(self.current_scope, ident.?) orelse return RefinementList.initEmpty();
    const sym_type = sym.typ orelse return RefinementList.initEmpty();
    const base_ora = getBaseOraType(self, sym_type) orelse return RefinementList.initEmpty();
    if (!base_ora.isInteger()) return RefinementList.initEmpty();

    const max_u256 = std.math.maxInt(u256);
    var min_val: ?u256 = null;
    var max_val: ?u256 = null;

    switch (op) {
        .GreaterEqual => if (branch == .Then) {
            min_val = value;
        } else if (value > 0) {
            max_val = value - 1;
        },
        .Greater => if (branch == .Then) {
            if (value < max_u256) min_val = value + 1;
        } else {
            max_val = value;
        },
        .LessEqual => if (branch == .Then) {
            max_val = value;
        } else if (value < max_u256) {
            min_val = value + 1;
        },
        .Less => if (branch == .Then) {
            if (value > 0) max_val = value - 1;
        } else {
            min_val = value;
        },
        .EqualEqual => if (branch == .Then) {
            min_val = value;
            max_val = value;
        },
        else => return RefinementList.initEmpty(),
    }

    if (min_val == null and max_val == null) return RefinementList.initEmpty();
    const refined_type = try buildRefinedType(self, base_ora, min_val, max_val, bin.span);
    var list = std.ArrayList(RefinementOverride){};
    errdefer list.deinit(self.allocator);
    // typ_owned = false: flow refinement types are allocated with type_storage_allocator (arena)
    // which is freed by the caller, so symbol table should not try to free them
    try list.append(self.allocator, RefinementOverride{ .name = ident.?, .type_info = refined_type, .typ_owned = false });
    return RefinementList{ .items = try list.toOwnedSlice(self.allocator) };
}

fn invertComparisonOperator(op: ast.Expressions.BinaryOp) ?ast.Expressions.BinaryOp {
    return switch (op) {
        .Less => .Greater,
        .LessEqual => .GreaterEqual,
        .Greater => .Less,
        .GreaterEqual => .LessEqual,
        .EqualEqual => .EqualEqual,
        .BangEqual => .BangEqual,
        else => null,
    };
}

fn deriveSwitchRefinement(
    self: *CoreResolver,
    name: []const u8,
    pattern: *const ast.Expressions.SwitchPattern,
) !?RefinementOverride {
    const sym = SymbolTable.findUp(self.current_scope, name) orelse return null;
    const base_ora = getBaseOraType(self, sym.typ orelse return null) orelse return null;
    if (!base_ora.isInteger()) return null;

    return switch (pattern.*) {
        .Literal => |lit| blk: {
            var lit_expr = ast.Expressions.ExprNode{ .Literal = lit.value };
            const const_val = try self.utils.evaluateConstantExpression(&lit_expr);
            const value = const_val orelse break :blk null;
            const refined_type = try buildRefinedType(self, base_ora, value, value, lit.span);
            // typ_owned = false: allocated with arena, not symbol table's allocator
            break :blk RefinementOverride{ .name = name, .type_info = refined_type, .typ_owned = false };
        },
        .Range => |range| blk: {
            const start_val = try self.utils.evaluateConstantExpression(@constCast(range.start));
            const end_val = try self.utils.evaluateConstantExpression(@constCast(range.end));
            const start_num = start_val orelse break :blk null;
            const end_num = end_val orelse break :blk null;
            const min_val: u256 = start_num;
            var max_val: u256 = end_num;
            if (!range.inclusive and max_val > 0) {
                max_val -= 1;
            }
            const refined_type = try buildRefinedType(self, base_ora, min_val, max_val, range.span);
            // typ_owned = false: allocated with arena, not symbol table's allocator
            break :blk RefinementOverride{ .name = name, .type_info = refined_type, .typ_owned = false };
        },
        else => null,
    };
}

fn mergeRefinementLists(
    self: *CoreResolver,
    left: RefinementList,
    right: RefinementList,
) !RefinementList {
    var list = std.ArrayList(RefinementOverride){};
    errdefer list.deinit(self.allocator);
    defer {
        var l = left;
        var r = right;
        l.deinit(self.allocator);
        r.deinit(self.allocator);
    }

    for (left.items) |item| {
        try list.append(self.allocator, item);
    }
    for (right.items) |item| {
        try list.append(self.allocator, item);
    }
    return RefinementList{ .items = try list.toOwnedSlice(self.allocator) };
}

fn mergeRefinementTypes(
    self: *CoreResolver,
    existing: TypeInfo,
    incoming: TypeInfo,
) ?TypeInfo {
    const existing_bounds = getBoundsFromType(self, existing) orelse return incoming;
    const incoming_bounds = getBoundsFromType(self, incoming) orelse return existing;

    if (!std.meta.eql(existing_bounds.base, incoming_bounds.base)) {
        return incoming;
    }

    var min_val = existing_bounds.min;
    if (incoming_bounds.min) |v| {
        if (min_val) |m| {
            min_val = if (v > m) v else m;
        } else {
            min_val = v;
        }
    }

    var max_val = existing_bounds.max;
    if (incoming_bounds.max) |v| {
        if (max_val) |m| {
            max_val = if (v < m) v else m;
        } else {
            max_val = v;
        }
    }

    if (min_val != null and max_val != null and min_val.? > max_val.?) {
        return null;
    }

    const span = incoming.span orelse existing.span orelse return null;
    const refined = buildRefinedType(self, existing_bounds.base, min_val, max_val, span) catch return null;
    return refined;
}

const BoundsInfo = struct {
    base: OraType,
    min: ?u256,
    max: ?u256,
};

fn getBoundsFromType(self: *CoreResolver, ty: TypeInfo) ?BoundsInfo {
    const ora = ty.ora_type orelse return null;
    const base = self.utils.extractBaseType(ora) orelse ora;
    return switch (ora) {
        .min_value => |mv| .{ .base = base, .min = mv.min, .max = null },
        .max_value => |mv| .{ .base = base, .min = null, .max = mv.max },
        .in_range => |ir| .{ .base = base, .min = ir.min, .max = ir.max },
        else => null,
    };
}

/// Get TypeInfo from a literal expression
fn getLiteralTypeInfo(lit: ast.Expressions.LiteralExpr) TypeInfo {
    return switch (lit) {
        .Integer => |int_lit| int_lit.type_info,
        .String => |str_lit| str_lit.type_info,
        .Bool => |bool_lit| bool_lit.type_info,
        .Address => |addr_lit| addr_lit.type_info,
        .Hex => |hex_lit| hex_lit.type_info,
        .Binary => |bin_lit| bin_lit.type_info,
        .Character => |char_lit| char_lit.type_info,
        .Bytes => |bytes_lit| bytes_lit.type_info,
    };
}

/// Format TypeInfo for error messages
fn formatTypeInfo(type_info: TypeInfo, allocator: std.mem.Allocator) ![]const u8 {
    if (type_info.ora_type) |ora_type| {
        return formatOraType(ora_type, allocator);
    }
    return std.fmt.allocPrint(allocator, "{s}", .{@tagName(type_info.category)});
}

/// Format OraType for error messages
fn formatOraType(ora_type: OraType, allocator: std.mem.Allocator) ![]const u8 {
    return switch (ora_type) {
        .min_value => |mv| {
            const base_str = try formatOraType(mv.base.*, allocator);
            defer allocator.free(base_str);
            return std.fmt.allocPrint(allocator, "MinValue<{s}, {d}>", .{ base_str, mv.min });
        },
        .max_value => |mv| {
            const base_str = try formatOraType(mv.base.*, allocator);
            defer allocator.free(base_str);
            return std.fmt.allocPrint(allocator, "MaxValue<{s}, {d}>", .{ base_str, mv.max });
        },
        .in_range => |ir| {
            const base_str = try formatOraType(ir.base.*, allocator);
            defer allocator.free(base_str);
            return std.fmt.allocPrint(allocator, "InRange<{s}, {d}, {d}>", .{ base_str, ir.min, ir.max });
        },
        .scaled => |s| {
            const base_str = try formatOraType(s.base.*, allocator);
            defer allocator.free(base_str);
            return std.fmt.allocPrint(allocator, "Scaled<{s}, {d}>", .{ base_str, s.decimals });
        },
        .exact => |e| {
            const base_str = try formatOraType(e.*, allocator);
            defer allocator.free(base_str);
            return std.fmt.allocPrint(allocator, "Exact<{s}>", .{base_str});
        },
        .non_zero_address => return std.fmt.allocPrint(allocator, "NonZeroAddress", .{}),
        .u8 => return std.fmt.allocPrint(allocator, "u8", .{}),
        .u16 => return std.fmt.allocPrint(allocator, "u16", .{}),
        .u32 => return std.fmt.allocPrint(allocator, "u32", .{}),
        .u64 => return std.fmt.allocPrint(allocator, "u64", .{}),
        .u128 => return std.fmt.allocPrint(allocator, "u128", .{}),
        .u256 => return std.fmt.allocPrint(allocator, "u256", .{}),
        .i8 => return std.fmt.allocPrint(allocator, "i8", .{}),
        .i16 => return std.fmt.allocPrint(allocator, "i16", .{}),
        .i32 => return std.fmt.allocPrint(allocator, "i32", .{}),
        .i64 => return std.fmt.allocPrint(allocator, "i64", .{}),
        .i128 => return std.fmt.allocPrint(allocator, "i128", .{}),
        .i256 => return std.fmt.allocPrint(allocator, "i256", .{}),
        .bool => return std.fmt.allocPrint(allocator, "bool", .{}),
        .address => return std.fmt.allocPrint(allocator, "address", .{}),
        .string => return std.fmt.allocPrint(allocator, "string", .{}),
        else => return std.fmt.allocPrint(allocator, "{s}", .{@tagName(ora_type)}),
    };
}

fn resolveLog(
    self: *CoreResolver,
    log_stmt: *ast.Statements.LogNode,
    context: TypeContext,
) TypeResolutionError!Typed {
    _ = context;

    // synthesize types for all log arguments
    var arg_types = try self.allocator.alloc(TypeInfo, log_stmt.args.len);
    defer self.allocator.free(arg_types);

    var combined_eff = Effect.pure();
    for (log_stmt.args, 0..) |*arg, i| {
        var arg_typed = try expression.synthExpr(self, arg);
        defer arg_typed.deinit(self.allocator);
        arg_types[i] = arg_typed.ty;
        const eff = takeEffect(&arg_typed);
        mergeEffects(self.allocator, &combined_eff, eff);
    }

    // validate log signature (event name exists, argument count matches, types match)
    var sig_fields_opt: ?[]const ast.LogField = null;
    if (self.symbol_table.getContractLogSignatures(self.current_scope)) |log_map| {
        sig_fields_opt = log_map.get(log_stmt.event_name);
    }
    if (sig_fields_opt == null) {
        return TypeResolutionError.UndefinedIdentifier;
    }
    const sig_fields = sig_fields_opt.?;

    // arity check: argument count must match field count
    if (sig_fields.len != log_stmt.args.len) {
        log.debug(
            "[type_resolver] Log argument count mismatch: got {d}, expected {d} for log {s}\n",
            .{ log_stmt.args.len, sig_fields.len, log_stmt.event_name },
        );
        return TypeResolutionError.TypeMismatch;
    }

    // type check each argument against corresponding field type
    for (sig_fields, arg_types, 0..) |field, arg_type, i| {
        // check if argument type is assignable to field type
        if (!self.validation.isAssignable(arg_type, field.type_info)) {
            const got_str_opt = formatTypeInfo(arg_type, self.allocator) catch null;
            const expected_str_opt = formatTypeInfo(field.type_info, self.allocator) catch null;
            defer if (got_str_opt) |s| self.allocator.free(s);
            defer if (expected_str_opt) |s| self.allocator.free(s);
            const got_str = got_str_opt orelse "unknown";
            const expected_str = expected_str_opt orelse "unknown";
            log.debug(
                "[type_resolver] Log argument {d} type mismatch for {s}: got {s}, expected {s}\n",
                .{ i, log_stmt.event_name, got_str, expected_str },
            );
            return TypeResolutionError.TypeMismatch;
        }
    }

    return Typed.init(TypeInfo.unknown(), combined_eff, self.allocator);
}

// Helpers
// ============================================================================

/// Check if guard should be skipped based on optimizations
/// Returns true if guard can be skipped (constant satisfies constraint, subtyping, or trusted builtin)
pub fn shouldSkipGuard(
    self: *CoreResolver,
    value_expr: *ast.Expressions.ExprNode,
    target_ora_type: ast.type_info.OraType,
) bool {
    // optimization 1: Check if value is a compile-time constant that satisfies the constraint
    const constant_result = self.utils.evaluateConstantExpression(value_expr) catch return false;
    if (constant_result) |constant_value| {
        if (constantSatisfiesRefinement(constant_value, target_ora_type)) {
            return true; // Constant satisfies constraint - skip guard
        }
    }

    // optimization 2: Check if source type is a subtype of target type (source <: target)
    const value_type_info = getExpressionTypeInfo(value_expr);
    if (value_type_info.ora_type) |source_ora_type| {
        const compat = @import("../validation/compatibility.zig");
        if (self.refinement_system.checkSubtype(
            source_ora_type,
            target_ora_type,
            compat.isBaseTypeCompatible,
        )) {
            return true; // Subtyping guarantees safety - skip guard
        }
    }

    // optimization 3: Check if value comes from trusted builtin (std.transaction.sender, std.msg.sender)
    // these are guaranteed by EVM semantics to be non-zero addresses
    if (target_ora_type == .non_zero_address) {
        if (isTrustedBuiltin(value_expr)) {
            return true; // Trusted builtin - skip guard
        }
    }

    return false; // No optimization applies - generate guard
}

/// Check if a constant value satisfies a refinement constraint
fn constantSatisfiesRefinement(constant_value: u256, ora_type: ast.type_info.OraType) bool {
    return switch (ora_type) {
        .min_value => |mv| constant_value >= mv.min,
        .max_value => |mv| constant_value <= mv.max,
        .in_range => |ir| constant_value >= ir.min and constant_value <= ir.max,
        .non_zero_address => constant_value != 0,
        else => true, // Other refinements don't have simple value constraints
    };
}

/// Get TypeInfo from an expression node
fn getExpressionTypeInfo(expr: *ast.Expressions.ExprNode) TypeInfo {
    return switch (expr.*) {
        .Identifier => |id| id.type_info,
        .Literal => |lit| switch (lit) {
            .Integer => |int_lit| int_lit.type_info,
            .String => |str_lit| str_lit.type_info,
            .Bool => |bool_lit| bool_lit.type_info,
            .Address => |addr_lit| addr_lit.type_info,
            .Hex => |hex_lit| hex_lit.type_info,
            .Binary => |bin_lit| bin_lit.type_info,
            .Character => |char_lit| char_lit.type_info,
            .Bytes => |bytes_lit| bytes_lit.type_info,
        },
        .Binary => |bin| bin.type_info,
        .Unary => |unary| unary.type_info,
        .Call => |call| call.type_info,
        .FieldAccess => |fa| fa.type_info,
        else => TypeInfo.unknown(),
    };
}

/// Check if expression is a trusted builtin call (std.transaction.sender, std.msg.sender)
/// These are guaranteed by EVM semantics to be non-zero addresses
fn isTrustedBuiltin(expr: *ast.Expressions.ExprNode) bool {
    return switch (expr.*) {
        .Call => |call| {
            if (call.callee.* == .FieldAccess) {
                const fa = &call.callee.FieldAccess;
                // check if field is "sender"
                if (!std.mem.eql(u8, fa.field, "sender")) {
                    return false;
                }
                // check if target is std.transaction or std.msg
                if (fa.target.* == .FieldAccess) {
                    const inner_fa = &fa.target.FieldAccess;
                    // check if inner field is "transaction" or "msg"
                    const is_transaction = std.mem.eql(u8, inner_fa.field, "transaction");
                    const is_msg = std.mem.eql(u8, inner_fa.field, "msg");
                    if (!is_transaction and !is_msg) {
                        return false;
                    }
                    // check if base is "std"
                    if (inner_fa.target.* == .Identifier) {
                        return std.mem.eql(u8, inner_fa.target.Identifier.name, "std");
                    }
                }
            }
            return false;
        },
        else => false,
    };
}

/// Validate that a literal value satisfies refinement type constraints.
/// Uses constant evaluation to check compile-time constants against refinement constraints.
fn validateLiteralAgainstRefinement(
    self: *CoreResolver,
    expr: *ast.Expressions.ExprNode,
    target_ora_type: ast.type_info.OraType,
) TypeResolutionError!void {
    // evaluate the constant expression
    const constant_result = self.utils.evaluateConstantExpression(expr) catch {
        // evaluation error (e.g., overflow) - let runtime guard handle it
        return;
    };

    const constant_value = constant_result orelse {
        // not a compile-time constant - let runtime guard handle it
        return;
    };

    // validate against refinement constraints
    switch (target_ora_type) {
        .min_value => |mv| {
            if (constant_value < mv.min) {
                return TypeResolutionError.TypeMismatch;
            }
        },
        .max_value => |mv| {
            if (constant_value > mv.max) {
                return TypeResolutionError.TypeMismatch;
            }
        },
        .in_range => |ir| {
            if (constant_value < ir.min or constant_value > ir.max) {
                return TypeResolutionError.TypeMismatch;
            }
        },
        .non_zero_address => {
            // for addresses, check if value is zero
            // note: This assumes the constant is an address value
            // in practice, address literals are hex strings, so this might not apply
            // but if we have a numeric constant representing an address, check it
            if (constant_value == 0) {
                return TypeResolutionError.TypeMismatch;
            }
        },
        .scaled, .exact => {
            // scaled and Exact don't have simple value constraints
            // they're validated at operation time (division, arithmetic)
            // no compile-time validation needed
        },
        else => {
            // not a refinement type, nothing to validate
        },
    }
}
