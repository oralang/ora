// ============================================================================
// Statement Type Resolution
// ============================================================================
// Resolves types for all statement types (VariableDecl, Return, If, etc.)
// ============================================================================

const std = @import("std");
const ast = @import("../../../ast.zig");
const TypeInfo = @import("../../type_info.zig").TypeInfo;
const OraType = @import("../../type_info.zig").OraType;
const CommonTypes = @import("../../type_info.zig").CommonTypes;
const state_mod = @import("../../../semantics/state.zig");
const SymbolTable = state_mod.SymbolTable;
const Scope = state_mod.Scope;
const semantics = @import("../../../semantics.zig");
const TypeResolutionError = @import("../mod.zig").TypeResolutionError;
const Typed = @import("mod.zig").Typed;
const Effect = @import("mod.zig").Effect;
const TypeContext = @import("mod.zig").TypeContext;
const validation = @import("../validation/mod.zig");
const refinements = @import("../refinements/mod.zig");
const utils = @import("../utils/mod.zig");

const CoreResolver = @import("mod.zig").CoreResolver;
const expression = @import("expression.zig");

/// Resolve types for a statement
pub fn resolveStatement(
    self: *CoreResolver,
    stmt: *ast.Statements.StmtNode,
    context: TypeContext,
) TypeResolutionError!Typed {
    return switch (stmt.*) {
        .VariableDecl => |*var_decl| resolveVariableDecl(self, var_decl, context),
        .Return => |*ret| resolveReturn(self, ret, context),
        .If => |*if_stmt| resolveIf(self, if_stmt, context),
        .While => |*while_stmt| resolveWhile(self, while_stmt, context),
        .ForLoop => |*for_stmt| resolveFor(self, for_stmt, context),
        .TryBlock => |*try_block| resolveTryBlock(self, try_block, context),
        .Expr => |*expr_stmt| resolveExpressionStmt(self, expr_stmt, context),
        .Break => |*break_stmt| resolveBreak(self, break_stmt, context),
        .Continue => |*continue_stmt| resolveContinue(self, continue_stmt, context),
        .Switch => |*switch_stmt| resolveSwitch(self, switch_stmt, context),
        .Log => |*log_stmt| resolveLog(self, log_stmt, context),
        else => {
            // Unhandled statement types - return unknown
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
    // Fix custom type names: if parser assumed struct_type but it's actually enum_type
    if (var_decl.type_info.ora_type) |ot| {
        if (ot == .struct_type) {
            const type_name = ot.struct_type;
            // Look up symbol to see if it's actually an enum
            // Search from root scope to find type declarations
            const symbol = SymbolTable.findUp(@as(?*const Scope, @ptrCast(&self.symbol_table.root)), type_name);
            if (symbol) |sym| {
                if (sym.kind == .Enum) {
                    // Fix: change struct_type to enum_type
                    var_decl.type_info.ora_type = OraType{ .enum_type = type_name };
                    var_decl.type_info.category = .Enum;
                }
            }
        }
        // Ensure category matches ora_type if present (fixes refinement types)
        var_decl.type_info.category = ot.getCategory();
    }

    if (var_decl.value) |value_expr| {
        // Ensure category matches ora_type if present (fixes refinement types)
        if (var_decl.type_info.ora_type) |ot| {
            var_decl.type_info.category = ot.getCategory();
        }
        // If type is unknown, infer it from the initializer expression
        if (!var_decl.type_info.isResolved()) {
            // Synthesize type from expression
            const typed = try expression.synthExpr(self, value_expr);

            // Assign inferred type to variable
            var_decl.type_info = typed.ty;
        } else {
            // Type is explicit, check expression against it
            _ = try expression.checkExpr(self, value_expr, var_decl.type_info);

            // Validate refinement types if present
            if (var_decl.type_info.ora_type) |target_ora_type| {
                // Validate refinement type structure
                try self.refinement_system.validate(target_ora_type);

                // Validate constant literal values against refinement constraints
                try validateLiteralAgainstRefinement(self, value_expr, target_ora_type);

                // Check guard optimizations: skip guard if optimization applies
                var_decl.skip_guard = shouldSkipGuard(self, value_expr, target_ora_type);
            }
        }
    } else {
        // No initializer - type must be explicit
        if (!var_decl.type_info.isResolved()) {
            return TypeResolutionError.UnresolvedType;
        }
    }

    // Add variable to symbol table AFTER type is resolved
    // Ensure type is resolved - if it has ora_type, derive category if needed
    var final_type = var_decl.type_info;
    if (final_type.ora_type) |ot| {
        // Ensure category matches the ora_type (refinement types inherit base category)
        const derived_category = ot.getCategory();
        // Always update category to match ora_type to ensure isResolved() works
        final_type.category = derived_category;
        // Also update the original type_info to ensure it's resolved
        var_decl.type_info.category = derived_category;
    }

    // Type should be resolved now - if not, there's a bug
    if (!final_type.isResolved()) {
        return TypeResolutionError.UnresolvedType;
    }

    // Copy the type if it contains pointers (refinement types, slices, etc.)
    // to ensure it's properly owned by the symbol table
    // Ensure final_type has ora_type before proceeding
    if (final_type.ora_type == null) {
        return TypeResolutionError.UnresolvedType;
    }
    var stored_type = final_type;
    var typ_owned_flag = false;
    const ot = final_type.ora_type.?;
    // Check if this type needs copying (has pointers)
    const needs_copy = switch (ot) {
        .min_value, .max_value, .in_range, .scaled, .exact, .slice, .error_union, .array, .map, .tuple, .anonymous_struct, ._union, .function => true,
        else => false,
    };

    // Only set typ_owned = true if we're storing in a function scope, not a block scope
    // Block scopes are temporary and shouldn't own types (they point to arena memory)
    const is_function_scope = if (self.current_scope) |scope|
        self.symbol_table.isFunctionScope(scope)
    else
        false;

    if (needs_copy and is_function_scope) {
        // Import copyOraTypeOwned from function_analyzer
        const copyOraTypeOwned = @import("../../../semantics/function_analyzer.zig").copyOraTypeOwned;
        if (copyOraTypeOwned(self.allocator, ot)) |copied_ora_type| {
            // Derive category from the copied type to ensure it's correct
            const derived_category = copied_ora_type.getCategory();
            stored_type = TypeInfo{
                .category = derived_category,
                .ora_type = copied_ora_type,
                .source = final_type.source,
                .span = final_type.span,
            };
            typ_owned_flag = true;
        } else |_| {
            // If copying fails, we cannot safely store this type as it may contain pointers to temporary memory
            // This should be rare (usually only out of memory), but we need to handle it
            // We cannot proceed without copying refinement types safely
            return TypeResolutionError.UnresolvedType;
        }
    } else {
        // No copying needed, or we're in a block scope (don't own types in block scopes)
        // Ensure category is correct
        stored_type.category = ot.getCategory();
    }

    // Final check: ensure stored type is resolved
    // Double-check that ora_type is set and category matches
    if (stored_type.ora_type) |stored_ot| {
        stored_type.category = stored_ot.getCategory();
    }
    if (!stored_type.isResolved()) {
        return TypeResolutionError.UnresolvedType;
    }
    // Final verification before storing
    if (stored_type.ora_type == null) {
        std.debug.print("[resolveVariableDecl] CRITICAL: Variable '{s}' has null ora_type before storing!\n", .{var_decl.name});
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

    // Track if we successfully stored the type to avoid leaks
    var type_stored = false;
    errdefer {
        // If we allocated a type but failed to store it, deallocate it
        if (typ_owned_flag and !type_stored) {
            if (stored_type.ora_type) |*ora_type| {
                const type_info = @import("../../type_info.zig");
                type_info.deinitOraType(self.allocator, @constCast(ora_type));
            }
        }
    }

    if (self.current_scope) |scope| {
        // Find the actual scope where the symbol is stored (could be in a block scope)
        // This ensures we update it in the correct scope
        const semantics_state = @import("../../../semantics/state.zig");
        if (semantics_state.SymbolTable.findScopeContaining(scope, var_decl.name)) |found| {
            // Found the symbol - update it in the scope where it's actually stored
            const old_symbol = &found.scope.symbols.items[found.idx];
            const type_info = @import("../../type_info.zig");
            // Only deallocate if the old type was owned AND has a valid ora_type
            if (old_symbol.typ_owned) {
                if (old_symbol.typ) |*old_typ| {
                    if (old_typ.ora_type != null) {
                        type_info.deinitTypeInfo(self.allocator, old_typ);
                    }
                }
            }
            // Only set typ_owned=true if this scope is a function scope
            // Block scopes should never own types (they point to arena memory)
            old_symbol.typ = stored_type;
            old_symbol.typ_owned = typ_owned_flag and self.symbol_table.isFunctionScope(found.scope);
            type_stored = true;
        } else {
            // Symbol doesn't exist yet - declare it in the current scope
            // Only set typ_owned if we're in a function scope
            var new_symbol = symbol;
            new_symbol.typ_owned = typ_owned_flag and self.symbol_table.isFunctionScope(scope);
            if (self.symbol_table.declare(scope, new_symbol)) |_| {
                // Duplicate declaration - this shouldn't happen if findScopeContaining worked
                // But handle it gracefully
                type_stored = true;
            } else |_| {
                // Failed to declare - this is an error
                return TypeResolutionError.UnresolvedType;
            }
        }
    } else {
        // No current scope - this shouldn't happen during normal resolution
        // But handle it gracefully by not storing the type
        std.debug.print("[resolveVariableDecl] WARNING: No current scope for variable '{s}'\n", .{var_decl.name});
        // Deallocate the copied type since we can't store it
        if (typ_owned_flag) {
            if (stored_type.ora_type) |*ora_type| {
                const type_info = @import("../../type_info.zig");
                type_info.deinitOraType(self.allocator, @constCast(ora_type));
                type_stored = true; // Mark as "handled" to prevent errdefer from double-freeing
            }
        }
    }

    // CRITICAL: If we copied a type but didn't store it, we have a leak
    // This should never happen, but if it does, we need to clean up
    if (typ_owned_flag and !type_stored) {
        std.debug.print("[resolveVariableDecl] CRITICAL: Variable '{s}' - copied type was not stored! Deallocating to prevent leak.\n", .{var_decl.name});
        if (stored_type.ora_type) |*ora_type| {
            const type_info = @import("../../type_info.zig");
            type_info.deinitOraType(self.allocator, @constCast(ora_type));
        }
    }

    return Typed.init(var_decl.type_info, Effect.pure(), self.allocator);
}

fn resolveReturn(
    self: *CoreResolver,
    ret: *ast.Statements.ReturnNode,
    context: TypeContext,
) TypeResolutionError!Typed {
    if (ret.value) |*value_expr| {
        // Check expression against expected return type
        if (context.function_return_type) |return_type| {
            _ = try expression.checkExpr(self, value_expr, return_type);

            // Check guard optimizations: skip guard if optimization applies
            if (return_type.ora_type) |target_ora_type| {
                ret.skip_guard = shouldSkipGuard(self, value_expr, target_ora_type);
            }
        } else {
            // No return type expected - synthesize
            _ = try expression.synthExpr(self, value_expr);
        }
    }

    return Typed.init(TypeInfo.unknown(), Effect.pure(), self.allocator);
}

fn resolveIf(
    self: *CoreResolver,
    if_stmt: *ast.Statements.IfNode,
    context: TypeContext,
) TypeResolutionError!Typed {
    // Resolve condition (should be bool)
    _ = try expression.synthExpr(self, &if_stmt.condition);
    // TODO: Validate condition is bool type

    // Resolve then branch statements
    // NOTE: We don't set block scopes here to avoid double-frees during deinit
    // Variables declared in blocks will be found via findUp from the function scope
    for (if_stmt.then_branch.statements) |*stmt| {
        _ = try resolveStatement(self, stmt, context);
    }

    // Resolve else branch if present
    if (if_stmt.else_branch) |*else_branch| {
        for (else_branch.statements) |*stmt| {
            _ = try resolveStatement(self, stmt, context);
        }
    }

    return Typed.init(TypeInfo.unknown(), Effect.pure(), self.allocator);
}

fn resolveWhile(
    self: *CoreResolver,
    while_stmt: *ast.Statements.WhileNode,
    context: TypeContext,
) TypeResolutionError!Typed {
    // Resolve condition
    _ = try expression.synthExpr(self, &while_stmt.condition);

    // Resolve body
    // NOTE: We don't set block scopes here to avoid double-frees during deinit
    // Variables declared in blocks will be found via findUp from the function scope
    for (while_stmt.body.statements) |*stmt| {
        _ = try resolveStatement(self, stmt, context);
    }

    return Typed.init(TypeInfo.unknown(), Effect.pure(), self.allocator);
}

fn resolveFor(
    self: *CoreResolver,
    for_stmt: *ast.Statements.ForLoopNode,
    context: TypeContext,
) TypeResolutionError!Typed {
    // Resolve iterable expression to get its type
    var iterable_typed = try expression.synthExpr(self, &for_stmt.iterable);
    defer iterable_typed.deinit(self.allocator);

    // Determine element type from iterable type
    const element_type: TypeInfo = if (iterable_typed.ty.ora_type) |iterable_ora_type| blk: {
        const elem_type = switch (iterable_ora_type) {
            .array => |arr| arr.elem.*,
            .slice => |elem_ptr| elem_ptr.*,
            .map => |m| m.value.*,
            // For range-style loops (integer), element type is the same as iterable
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
        // Range expressions with integer literals (e.g., 0..20) resolve to Integer category
        // but have null ora_type. Infer u256 as the element type.
        break :blk CommonTypes.u256_type();
    } else TypeInfo.unknown();

    // Resolve loop pattern variables and update their types in symbol table
    // Loop variables are declared in the body scope, not the function scope
    const body_scope_key: usize = @intFromPtr(&for_stmt.body);
    const body_scope = self.symbol_table.block_scopes.get(body_scope_key) orelse self.current_scope;

    if (body_scope) |scope| {
        switch (for_stmt.pattern) {
            .Single => |s| {
                // Update the symbol type for the loop variable
                // Loop variables are typically simple types, not owned
                if (element_type.ora_type != null) {
                    _ = self.symbol_table.updateSymbolType(scope, s.name, element_type, false) catch {
                        // Symbol might not exist yet, that's okay - it will be created during lowering
                    };
                }
            },
            .IndexPair => |p| {
                // Update item variable type
                // Loop variables are typically simple types, not owned
                if (element_type.ora_type != null) {
                    _ = self.symbol_table.updateSymbolType(scope, p.item, element_type, false) catch {};
                }
                // Index variable is always u256 (or mlir index type) - simple type, not owned
                const index_type = CommonTypes.u256_type();
                _ = self.symbol_table.updateSymbolType(scope, p.index, index_type, false) catch {};
            },
            .Destructured => |d| {
                // For destructuring, we need to extract field types from the element type
                // This is complex and depends on the destructuring pattern
                // For now, use element type for all fields (simplified)
                // Loop variables are typically simple types, not owned
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

    // Resolve loop body
    // Set current_scope to body_scope so loop variables can be found
    const prev_scope = self.current_scope;
    if (body_scope) |bs| {
        self.current_scope = bs;
    }
    defer {
        self.current_scope = prev_scope;
    }

    for (for_stmt.body.statements) |*stmt| {
        _ = try resolveStatement(self, stmt, context);
    }

    return Typed.init(TypeInfo.unknown(), Effect.pure(), self.allocator);
}

fn resolveTryBlock(
    self: *CoreResolver,
    try_block: *ast.Statements.TryBlockNode,
    context: TypeContext,
) TypeResolutionError!Typed {
    // Resolve try block statements
    const prev_scope = self.current_scope;
    const try_block_key: usize = @intFromPtr(&try_block.try_block);
    if (self.symbol_table.block_scopes.get(try_block_key)) |try_scope| {
        self.current_scope = try_scope;
    }
    defer {
        self.current_scope = prev_scope;
    }

    for (try_block.try_block.statements) |*stmt| {
        _ = try resolveStatement(self, stmt, context);
    }

    // Resolve catch block if present
    if (try_block.catch_block) |*catch_block| {
        // Set scope for catch block so error variable is accessible
        const catch_block_key: usize = @intFromPtr(&catch_block.block);
        if (self.symbol_table.block_scopes.get(catch_block_key)) |catch_scope| {
            self.current_scope = catch_scope;
        }
        defer {
            self.current_scope = prev_scope;
        }

        for (catch_block.block.statements) |*stmt| {
            _ = try resolveStatement(self, stmt, context);
        }
    }

    return Typed.init(TypeInfo.unknown(), Effect.pure(), self.allocator);
}

fn resolveExpressionStmt(
    self: *CoreResolver,
    expr_stmt: *ast.Expressions.ExprNode,
    context: TypeContext,
) TypeResolutionError!Typed {
    _ = context;
    _ = try expression.synthExpr(self, expr_stmt);
    return Typed.init(TypeInfo.unknown(), Effect.pure(), self.allocator);
}

fn resolveBreak(
    self: *CoreResolver,
    break_stmt: *ast.Statements.BreakNode,
    context: TypeContext,
) TypeResolutionError!Typed {
    // If break has a value expression, synthesize its type
    if (break_stmt.value) |value_expr| {
        _ = try expression.synthExpr(self, value_expr);
        // TODO: Validate that break value type matches expected type from labeled block/switch
        // This requires tracking the expected type from the enclosing labeled block or switch expression
    }
    _ = context; // Context may contain expected break value type in the future
    return Typed.init(TypeInfo.unknown(), Effect.pure(), self.allocator);
}

fn resolveContinue(
    self: *CoreResolver,
    continue_stmt: *ast.Statements.ContinueNode,
    context: TypeContext,
) TypeResolutionError!Typed {
    // If continue has a value expression (for labeled switch continue), synthesize its type
    if (continue_stmt.value) |value_expr| {
        _ = try expression.synthExpr(self, value_expr);
        // TODO: Validate that continue value type matches expected type from labeled switch
    }
    _ = context; // Context may contain expected continue value type in the future
    return Typed.init(TypeInfo.unknown(), Effect.pure(), self.allocator);
}

fn resolveSwitch(
    self: *CoreResolver,
    switch_stmt: *ast.Statements.SwitchNode,
    context: TypeContext,
) TypeResolutionError!Typed {
    // Synthesize type of the switch condition expression
    var condition_typed = try expression.synthExpr(self, &switch_stmt.condition);
    defer condition_typed.deinit(self.allocator);

    const condition_type = condition_typed.ty;

    // Validate that case patterns match the condition type
    for (switch_stmt.cases) |*case| {
        try validateSwitchPattern(self, &case.pattern, condition_type, case.span);
    }

    // Resolve types for each case body
    for (switch_stmt.cases) |*case| {
        switch (case.body) {
            .Expression => |case_expr| {
                _ = try expression.synthExpr(self, case_expr);
            },
            .Block => |*case_block| {
                for (case_block.statements) |*case_stmt| {
                    _ = try resolveStatement(self, case_stmt, context);
                }
            },
            .LabeledBlock => |*labeled| {
                for (labeled.block.statements) |*case_stmt| {
                    _ = try resolveStatement(self, case_stmt, context);
                }
            },
        }
    }

    // Resolve types for default case if present
    if (switch_stmt.default_case) |*default_block| {
        for (default_block.statements) |*default_stmt| {
            _ = try resolveStatement(self, default_stmt, context);
        }
    }

    return Typed.init(TypeInfo.unknown(), Effect.pure(), self.allocator);
}

/// Validate that a switch pattern matches the condition type
fn validateSwitchPattern(
    self: *CoreResolver,
    pattern: *const ast.Expressions.SwitchPattern,
    condition_type: TypeInfo,
    pattern_span: ast.SourceSpan,
) TypeResolutionError!void {
    if (!condition_type.isResolved()) {
        // Condition type not resolved - skip validation
        return;
    }

    switch (pattern.*) {
        .Else => {
            // Else patterns are always valid
        },
        .Literal => |lit| {
            // Validate literal type matches condition type category
            const lit_type_info = getLiteralTypeInfo(lit.value);
            if (!lit_type_info.isResolved()) {
                return TypeResolutionError.TypeMismatch;
            }

            // Check category compatibility
            if (lit_type_info.category != condition_type.category) {
                std.debug.print(
                    "[type_resolver] Switch pattern type mismatch: pattern category {s}, condition category {s}\n",
                    .{ @tagName(lit_type_info.category), @tagName(condition_type.category) },
                );
                return TypeResolutionError.TypeMismatch;
            }

            // For integer types, check base type compatibility
            if (lit_type_info.category == .Integer and condition_type.category == .Integer) {
                if (lit_type_info.ora_type != null and condition_type.ora_type != null) {
                    const compat = @import("../validation/compatibility.zig");
                    // Pattern literal type should be assignable to condition type
                    if (!compat.isBaseTypeCompatible(lit_type_info.ora_type.?, condition_type.ora_type.?)) {
                        std.debug.print(
                            "[type_resolver] Switch pattern integer type mismatch\n",
                            .{},
                        );
                        return TypeResolutionError.TypeMismatch;
                    }
                }
            }
        },
        .Range => |range| {
            // Validate range endpoints are compatible with condition type
            var start_typed = try expression.synthExpr(self, range.start);
            defer start_typed.deinit(self.allocator);
            var end_typed = try expression.synthExpr(self, range.end);
            defer end_typed.deinit(self.allocator);

            // Both endpoints must be assignable to condition type
            if (!self.validation.isAssignable(condition_type, start_typed.ty)) {
                std.debug.print(
                    "[type_resolver] Switch range start type mismatch\n",
                    .{},
                );
                return TypeResolutionError.TypeMismatch;
            }
            if (!self.validation.isAssignable(condition_type, end_typed.ty)) {
                std.debug.print(
                    "[type_resolver] Switch range end type mismatch\n",
                    .{},
                );
                return TypeResolutionError.TypeMismatch;
            }
        },
        .EnumValue => |ev| {
            // Validate enum value matches condition enum type
            if (condition_type.ora_type) |cond_ora_type| {
                switch (cond_ora_type) {
                    .enum_type => |enum_name| {
                        // If pattern provided a qualified enum name, it must match
                        if (ev.enum_name.len != 0 and !std.mem.eql(u8, ev.enum_name, enum_name)) {
                            std.debug.print(
                                "[type_resolver] Switch enum pattern enum name mismatch: got {s}, expected {s}\n",
                                .{ ev.enum_name, enum_name },
                            );
                            return TypeResolutionError.TypeMismatch;
                        }
                    },
                    else => {
                        std.debug.print(
                            "[type_resolver] Switch enum pattern used with non-enum condition\n",
                            .{},
                        );
                        return TypeResolutionError.TypeMismatch;
                    },
                }
            } else {
                std.debug.print(
                    "[type_resolver] Switch enum pattern used with unresolved condition type\n",
                    .{},
                );
                return TypeResolutionError.TypeMismatch;
            }
        },
    }
    _ = pattern_span; // Span available for future error reporting improvements
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

    // Synthesize types for all log arguments
    var arg_types = try self.allocator.alloc(TypeInfo, log_stmt.args.len);
    defer self.allocator.free(arg_types);

    for (log_stmt.args, 0..) |*arg, i| {
        var arg_typed = try expression.synthExpr(self, arg);
        defer arg_typed.deinit(self.allocator);
        arg_types[i] = arg_typed.ty;
    }

    // Validate log signature (event name exists, argument count matches, types match)
    const sig_fields_opt = self.symbol_table.log_signatures.get(log_stmt.event_name);
    if (sig_fields_opt == null) {
        std.debug.print(
            "[type_resolver] Unknown log event: {s}\n",
            .{log_stmt.event_name},
        );
        return TypeResolutionError.TypeMismatch;
    }
    const sig_fields = sig_fields_opt.?;

    // Arity check: argument count must match field count
    if (sig_fields.len != log_stmt.args.len) {
        std.debug.print(
            "[type_resolver] Log argument count mismatch: got {d}, expected {d} for log {s}\n",
            .{ log_stmt.args.len, sig_fields.len, log_stmt.event_name },
        );
        return TypeResolutionError.TypeMismatch;
    }

    // Type check each argument against corresponding field type
    for (sig_fields, arg_types, 0..) |field, arg_type, i| {
        // Check if argument type is assignable to field type
        if (!self.validation.isAssignable(field.type_info, arg_type)) {
            const got_str = formatTypeInfo(arg_type, self.allocator) catch "unknown";
            const expected_str = formatTypeInfo(field.type_info, self.allocator) catch "unknown";
            std.debug.print(
                "[type_resolver] Log argument {d} type mismatch for {s}: got {s}, expected {s}\n",
                .{ i, log_stmt.event_name, got_str, expected_str },
            );
            return TypeResolutionError.TypeMismatch;
        }
    }

    return Typed.init(TypeInfo.unknown(), Effect.pure(), self.allocator);
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
    // Optimization 1: Check if value is a compile-time constant that satisfies the constraint
    const constant_result = self.utils.evaluateConstantExpression(value_expr) catch return false;
    if (constant_result) |constant_value| {
        if (constantSatisfiesRefinement(constant_value, target_ora_type)) {
            return true; // Constant satisfies constraint - skip guard
        }
    }

    // Optimization 2: Check if source type is a subtype of target type (source <: target)
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

    // Optimization 3: Check if value comes from trusted builtin (std.transaction.sender, std.msg.sender)
    // These are guaranteed by EVM semantics to be non-zero addresses
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
                // Check if field is "sender"
                if (!std.mem.eql(u8, fa.field, "sender")) {
                    return false;
                }
                // Check if target is std.transaction or std.msg
                if (fa.target.* == .FieldAccess) {
                    const inner_fa = &fa.target.FieldAccess;
                    // Check if inner field is "transaction" or "msg"
                    const is_transaction = std.mem.eql(u8, inner_fa.field, "transaction");
                    const is_msg = std.mem.eql(u8, inner_fa.field, "msg");
                    if (!is_transaction and !is_msg) {
                        return false;
                    }
                    // Check if base is "std"
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
    // Evaluate the constant expression
    const constant_result = self.utils.evaluateConstantExpression(expr) catch {
        // Evaluation error (e.g., overflow) - let runtime guard handle it
        return;
    };

    const constant_value = constant_result orelse {
        // Not a compile-time constant - let runtime guard handle it
        return;
    };

    // Validate against refinement constraints
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
            // For addresses, check if value is zero
            // Note: This assumes the constant is an address value
            // In practice, address literals are hex strings, so this might not apply
            // But if we have a numeric constant representing an address, check it
            if (constant_value == 0) {
                return TypeResolutionError.TypeMismatch;
            }
        },
        .scaled, .exact => {
            // Scaled and Exact don't have simple value constraints
            // They're validated at operation time (division, arithmetic)
            // No compile-time validation needed
        },
        else => {
            // Not a refinement type, nothing to validate
        },
    }
}
