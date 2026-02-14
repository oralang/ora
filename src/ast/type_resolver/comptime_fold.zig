// ============================================================================
// Comptime Folding Pass
// ============================================================================
// Replace constant expressions with integer literals after type resolution.
// Uses CoreResolver's const table for identifier resolution.
// ============================================================================

const std = @import("std");
const ast = @import("../../ast.zig");
const state = @import("../../semantics/state.zig");
const core = @import("core/mod.zig");
const stmt_resolver = @import("core/statement.zig");
const type_info_mod = @import("../type_info.zig");
const comptime_eval = @import("../../comptime/mod.zig");
const log = @import("log");

const CtValue = comptime_eval.CtValue;

const TypeInfo = type_info_mod.TypeInfo;
const CommonTypes = type_info_mod.CommonTypes;
const SymbolTable = state.SymbolTable;
const Scope = state.Scope;

const FoldError = error{OutOfMemory};

/// Helper to fold a CtValue into a literal expression
/// Returns true if folding succeeded, false otherwise
fn tryFoldValue(
    ctx: *FoldContext,
    expr: *ast.Expressions.ExprNode,
    ct_val: CtValue,
    span: ast.SourceSpan,
    type_info: TypeInfo,
) FoldError!bool {
    switch (ct_val) {
        .integer => |int_val| {
            var ty = type_info;
            if (!ty.isResolved()) {
                ty = CommonTypes.unknown_integer();
            } else if (ty.ora_type) |ot| {
                // Unwrap refinement types to base type for literal
                switch (ot) {
                    .min_value => |mv| ty = TypeInfo.fromOraType(mv.base.*),
                    .max_value => |mv| ty = TypeInfo.fromOraType(mv.base.*),
                    .in_range => |ir| ty = TypeInfo.fromOraType(ir.base.*),
                    .scaled => |s| ty = TypeInfo.fromOraType(s.base.*),
                    .exact => |e| ty = TypeInfo.fromOraType(e.*),
                    else => {},
                }
            }
            // Check that the value fits within the declared type width
            if (ty.ora_type) |ot| {
                if (!fitsInType(int_val, ot)) {
                    log.warn("comptime value {d} overflows type '{s}' at line {d}\n", .{
                        int_val, @tagName(ot), span.line,
                    });
                    return false; // don't fold — leave as runtime expression
                }
            }
            const value_str = try std.fmt.allocPrint(ctx.type_storage_allocator, "{}", .{int_val});
            expr.* = .{ .Literal = .{ .Integer = .{
                .value = value_str,
                .type_info = ty,
                .span = span,
            } } };
            return true;
        },
        .boolean => |bool_val| {
            var ty = type_info;
            if (!ty.isResolved()) {
                ty = CommonTypes.bool_type();
            }
            expr.* = .{ .Literal = .{ .Bool = .{
                .value = bool_val,
                .type_info = ty,
                .span = span,
            } } };
            return true;
        },
        .address => |addr_val| {
            const value_str = try std.fmt.allocPrint(ctx.type_storage_allocator, "0x{x:0>40}", .{addr_val});
            var ty = type_info;
            if (!ty.isResolved()) {
                ty = CommonTypes.address_type();
            }
            expr.* = .{ .Literal = .{ .Address = .{
                .value = value_str,
                .type_info = ty,
                .span = span,
            } } };
            return true;
        },
        else => return false,
    }
}

pub const FoldContext = struct {
    allocator: std.mem.Allocator,
    type_storage_allocator: std.mem.Allocator,
    symbol_table: *SymbolTable,
    core_resolver: *core.CoreResolver,
    current_scope: ?*Scope = null,
};

pub fn foldConstants(ctx: *FoldContext, nodes: []ast.AstNode) FoldError!void {
    const prev_scope = ctx.current_scope;
    defer ctx.current_scope = prev_scope;

    for (nodes) |*node| {
        try foldNode(ctx, node);
    }
}

fn foldNode(ctx: *FoldContext, node: *ast.AstNode) FoldError!void {
    switch (node.*) {
        .Contract => |*contract| try foldContract(ctx, contract),
        .Function => |*function| try foldFunction(ctx, function),
        .VariableDecl => |*var_decl| {
            if (var_decl.value) |value| {
                try foldExpr(ctx, value, true);
                if (var_decl.type_info.ora_type) |target_ora_type| {
                    var_decl.skip_guard = stmt_resolver.shouldSkipGuard(ctx.core_resolver, value, target_ora_type);
                }
                if (var_decl.kind != .Var) {
                    const const_result = ctx.core_resolver.evaluateConstantExpression(value);
                    if (const_result.getValue()) |ct_value| {
                        if (ctx.current_scope) |scope| {
                            ctx.core_resolver.setComptimeValue(scope, var_decl.name, ct_value) catch {};
                        }
                    }
                }
            }
        },
        .Constant => |*constant| {
            try foldExpr(ctx, constant.value, true);
        },
        .ContractInvariant => |*invariant| {
            try foldExpr(ctx, invariant.condition, true);
        },
        .Block => |*block| {
            try foldBlock(ctx, block, null);
        },
        .Statement => |stmt| {
            try foldStmt(ctx, @constCast(stmt));
        },
        .TryBlock => |*try_block| {
            try foldBlock(ctx, &try_block.try_block, null);
            if (try_block.catch_block) |*catch_block| {
                try foldBlock(ctx, &catch_block.block, null);
            }
        },
        else => {},
    }
}

fn foldContract(ctx: *FoldContext, contract: *ast.ContractNode) FoldError!void {
    const prev_scope = ctx.current_scope;
    if (ctx.symbol_table.contract_scopes.get(contract.name)) |scope| {
        ctx.current_scope = scope;
        ctx.core_resolver.current_scope = scope;
    }
    defer {
        ctx.current_scope = prev_scope;
        ctx.core_resolver.current_scope = prev_scope;
    }

    for (contract.body) |*child| {
        try foldNode(ctx, child);
    }

    // DCE: mark private functions whose call sites were all folded away
    markDeadPrivateFunctions(contract);
}

fn foldFunction(ctx: *FoldContext, function: *ast.FunctionNode) FoldError!void {
    const prev_scope = ctx.current_scope;
    const prev_return_type = ctx.core_resolver.current_function_return_type;
    if (ctx.symbol_table.function_scopes.get(function.name)) |scope| {
        ctx.current_scope = scope;
        ctx.core_resolver.current_scope = scope;
    }
    ctx.core_resolver.current_function_return_type = function.return_type_info;

    // Push a comptime env scope so function-local consts don't leak to other functions
    ctx.core_resolver.comptime_env.pushScope(false) catch {};
    defer {
        ctx.core_resolver.comptime_env.popScope();
        ctx.current_scope = prev_scope;
        ctx.core_resolver.current_scope = prev_scope;
        ctx.core_resolver.current_function_return_type = prev_return_type;
    }

    for (function.parameters) |*param| {
        if (param.default_value) |default_val| {
            try foldExpr(ctx, default_val, true);
        }
    }
    for (function.requires_clauses) |clause| {
        try foldExpr(ctx, clause, true);
    }
    for (function.ensures_clauses) |clause| {
        try foldExpr(ctx, clause, true);
    }
    const body_key: usize = @intFromPtr(&function.body);
    try foldBlock(ctx, &function.body, body_key);
}

fn foldBlock(ctx: *FoldContext, block: *ast.Statements.BlockNode, scope_key: ?usize) FoldError!void {
    const prev_scope = ctx.current_scope;
    if (scope_key) |key| {
        if (ctx.symbol_table.block_scopes.get(key)) |scope| {
            ctx.current_scope = scope;
            ctx.core_resolver.current_scope = scope;
        }
    }
    defer {
        ctx.current_scope = prev_scope;
        ctx.core_resolver.current_scope = prev_scope;
    }

    for (block.statements) |*stmt| {
        try foldStmt(ctx, stmt);
    }
}

fn foldStmt(ctx: *FoldContext, stmt: *ast.Statements.StmtNode) FoldError!void {
    switch (stmt.*) {
        .Expr => |*expr| try foldExpr(ctx, expr, true),
        .VariableDecl => |*var_decl| {
            if (var_decl.value) |value| {
                try foldExpr(ctx, value, true);
                if (var_decl.type_info.ora_type) |target_ora_type| {
                    var_decl.skip_guard = stmt_resolver.shouldSkipGuard(ctx.core_resolver, value, target_ora_type);
                }
                // Track both const and var in comptime_env. Var values may be
                // updated by comptime while/for and folded into later expressions.
                const const_result = ctx.core_resolver.evaluateConstantExpression(value);
                if (const_result.getValue()) |ct_value| {
                    if (ctx.current_scope) |scope| {
                        ctx.core_resolver.setComptimeValue(scope, var_decl.name, ct_value) catch {};
                    }
                }
            }
        },
        .DestructuringAssignment => |*dest| {
            try foldExpr(ctx, dest.value, true);
        },
        .Return => |*ret| {
            if (ret.value) |*value| {
                try foldExpr(ctx, value, true);
                if (ctx.core_resolver.current_function_return_type) |return_type| {
                    if (return_type.ora_type) |target_ora_type| {
                        ret.skip_guard = stmt_resolver.shouldSkipGuard(ctx.core_resolver, value, target_ora_type);
                    }
                }
            }
        },
        .If => |*if_stmt| {
            try foldExpr(ctx, &if_stmt.condition, true);
            const then_key = getBlockScopeKey(stmt, 0);
            try foldBlock(ctx, &if_stmt.then_branch, then_key);
            if (if_stmt.else_branch) |*else_block| {
                const else_key = getBlockScopeKey(stmt, 1);
                try foldBlock(ctx, else_block, else_key);
            }
        },
        .While => |*while_stmt| {
            if (while_stmt.is_comptime) {
                // Comptime while: evaluate entirely at compile time
                tryEvalComptimeWhile(ctx, while_stmt);
            }
            try foldExpr(ctx, &while_stmt.condition, true);
            for (while_stmt.invariants) |*inv| {
                try foldExpr(ctx, inv, true);
            }
            if (while_stmt.decreases) |dec| try foldExpr(ctx, dec, true);
            if (while_stmt.increases) |inc| try foldExpr(ctx, inc, true);
            const body_key = getBlockScopeKey(stmt, 0);
            try foldBlock(ctx, &while_stmt.body, body_key);
        },
        .ForLoop => |*for_stmt| {
            if (for_stmt.is_comptime) {
                // Comptime for: evaluate entirely at compile time
                tryEvalComptimeFor(ctx, for_stmt);
            }
            try foldExpr(ctx, &for_stmt.iterable, true);
            for (for_stmt.invariants) |*inv| {
                try foldExpr(ctx, inv, true);
            }
            if (for_stmt.decreases) |dec| try foldExpr(ctx, dec, true);
            if (for_stmt.increases) |inc| try foldExpr(ctx, inc, true);
            const body_key = getBlockScopeKey(stmt, 0);
            try foldBlock(ctx, &for_stmt.body, body_key);
        },
        .Log => |*log_stmt| {
            for (log_stmt.args) |*arg| {
                try foldExpr(ctx, arg, true);
            }
        },
        .Assert => |*assert_stmt| try foldExpr(ctx, &assert_stmt.condition, true),
        .Invariant => |*inv| try foldExpr(ctx, &inv.condition, true),
        .Requires => |*req| try foldExpr(ctx, &req.condition, true),
        .Ensures => |*ens| try foldExpr(ctx, &ens.condition, true),
        .Assume => |*assume| try foldExpr(ctx, &assume.condition, true),
        .Switch => |*switch_stmt| {
            try foldExpr(ctx, &switch_stmt.condition, true);
            for (switch_stmt.cases) |*case| {
                try foldSwitchPattern(ctx, &case.pattern);
                switch (case.body) {
                    .Expression => |expr| try foldExpr(ctx, expr, true),
                    .Block => |*case_block| {
                        const key: usize = @intFromPtr(case_block);
                        try foldBlock(ctx, case_block, key);
                    },
                    .LabeledBlock => |*labeled| {
                        const key: usize = @intFromPtr(&labeled.block);
                        try foldBlock(ctx, &labeled.block, key);
                    },
                }
            }
            if (switch_stmt.default_case) |*default_block| {
                const key: usize = @intFromPtr(default_block);
                try foldBlock(ctx, default_block, key);
            }
        },
        .TryBlock => |*try_block| {
            const try_key = getBlockScopeKey(stmt, 0);
            try foldBlock(ctx, &try_block.try_block, try_key);
            if (try_block.catch_block) |*catch_block| {
                const catch_key = getBlockScopeKey(stmt, 1);
                try foldBlock(ctx, &catch_block.block, catch_key);
            }
        },
        .CompoundAssignment => |*compound| {
            try foldExpr(ctx, compound.value, true);
        },
        .Lock => |*lock_stmt| try foldExpr(ctx, &lock_stmt.path, true),
        .Unlock => |*unlock_stmt| try foldExpr(ctx, &unlock_stmt.path, true),
        .Break => |*break_stmt| {
            if (break_stmt.value) |value| try foldExpr(ctx, value, true);
        },
        .Continue => |*continue_stmt| {
            if (continue_stmt.value) |value| try foldExpr(ctx, value, true);
        },
        .LabeledBlock => |*labeled| {
            const key: usize = @intFromPtr(&labeled.block);
            try foldBlock(ctx, &labeled.block, key);
        },
        else => {},
    }
}

fn foldSwitchPattern(ctx: *FoldContext, pattern: *ast.Expressions.SwitchPattern) FoldError!void {
    switch (pattern.*) {
        .Literal => |*lit| {
            var expr = ast.Expressions.ExprNode{ .Literal = lit.value };
            try foldExpr(ctx, &expr, true);
            if (expr == .Literal) {
                lit.value = expr.Literal;
            }
        },
        .Range => |*range| {
            try foldExpr(ctx, range.start, true);
            try foldExpr(ctx, range.end, true);
        },
        else => {},
    }
}

fn foldExpr(ctx: *FoldContext, expr: *ast.Expressions.ExprNode, allow_fold: bool) FoldError!void {
    switch (expr.*) {
        .Identifier => |*id| {
            if (allow_fold) {
                // Don't fold function parameters — they are runtime values
                const is_param = if (ctx.current_scope) |scope|
                    if (SymbolTable.findUp(scope, id.name)) |sym| sym.kind == .Param else false
                else
                    false;
                if (is_param) return; // early return prevents catch-all fold below
                if (ctx.core_resolver.lookupComptimeValue(id.name)) |val| {
                    _ = try tryFoldValue(ctx, expr, val, getExpressionSpan(expr), id.type_info);
                }
            }
        },
        .Binary => |*bin| {
            try foldExpr(ctx, bin.lhs, true);
            try foldExpr(ctx, bin.rhs, true);
            if (allow_fold) {
                const eval_result = ctx.core_resolver.evaluateConstantExpression(expr);
                if (eval_result.getValue()) |ct_val| {
                    if (try tryFoldValue(ctx, expr, ct_val, getExpressionSpan(expr), getExpressionTypeInfo(expr))) return;
                } else if (bin.lhs.* == .Literal and bin.rhs.* == .Literal) {
                    // Both operands are compile-time known but the result failed —
                    // overflow/underflow/div-by-zero. Warn but keep emitting MLIR.
                    emitComptimeArithError(ctx, expr);
                }
            }
        },
        .Unary => |*unary| {
            try foldExpr(ctx, unary.operand, true);
            if (allow_fold) {
                if (ctx.core_resolver.evaluateConstantExpression(expr).getValue()) |ct_val| {
                    if (try tryFoldValue(ctx, expr, ct_val, getExpressionSpan(expr), getExpressionTypeInfo(expr))) return;
                }
            }
        },
        .Assignment => |*assign| {
            try foldExpr(ctx, assign.value, true);
            const target_type = getExpressionTypeInfo(assign.target);
            if (target_type.ora_type) |target_ora_type| {
                assign.skip_guard = stmt_resolver.shouldSkipGuard(ctx.core_resolver, assign.value, target_ora_type);
            }
        },
        .CompoundAssignment => |*compound| {
            try foldExpr(ctx, compound.value, true);
        },
        .Call => |*call| {
            for (call.arguments) |arg| try foldExpr(ctx, arg, true);
            // Try to evaluate the entire call at comptime (pure fn + all args known)
            if (allow_fold) {
                if (ctx.core_resolver.evaluateConstantExpression(expr).getValue()) |ct_val| {
                    if (try tryFoldValue(ctx, expr, ct_val, getExpressionSpan(expr), getExpressionTypeInfo(expr))) return;
                }
            }
        },
        .Index => |*idx| {
            try foldExpr(ctx, idx.target, true);
            try foldExpr(ctx, idx.index, true);
        },
        .FieldAccess => |*fa| {
            // fold target first so identifier -> tuple literal can be applied
            try foldExpr(ctx, fa.target, true);
            if (allow_fold) {
                if (ctx.core_resolver.evaluateConstantExpression(expr).getValue()) |ct_val| {
                    if (try tryFoldValue(ctx, expr, ct_val, getExpressionSpan(expr), getExpressionTypeInfo(expr))) return;
                }
            }
            // TODO: Field access on array_ref once heap-based aggregates are supported
            if (fa.target.* == .AnonymousStruct) {
                const anon = &fa.target.AnonymousStruct;
                for (anon.fields) |*field| {
                    if (!std.mem.eql(u8, field.name, fa.field)) continue;
                    if (field.value.* == .Literal) {
                        const span = getExpressionSpan(expr);
                        expr.* = .{ .Literal = field.value.Literal };
                        setLiteralSpan(&expr.Literal, span);
                    }
                    break;
                }
            } else if (fa.target.* == .StructInstantiation) {
                const inst = &fa.target.StructInstantiation;
                for (inst.fields) |*field| {
                    if (!std.mem.eql(u8, field.name, fa.field)) continue;
                    if (field.value.* == .Literal) {
                        const span = getExpressionSpan(expr);
                        expr.* = .{ .Literal = field.value.Literal };
                        setLiteralSpan(&expr.Literal, span);
                    }
                    break;
                }
            } else if (fa.target.* == .Tuple) {
                const tuple = &fa.target.Tuple;
                const field_str = if (fa.field.len > 0 and fa.field[0] == '_') fa.field[1..] else fa.field;
                const idx = std.fmt.parseInt(usize, field_str, 10) catch return;
                if (idx < tuple.elements.len) {
                    const element = tuple.elements[idx];
                    if (element.* == .Literal) {
                        const span = getExpressionSpan(expr);
                        expr.* = .{ .Literal = element.Literal };
                        setLiteralSpan(&expr.Literal, span);
                    }
                }
            }
            // TODO: Handle array_ref field access once heap-based aggregates are supported
        },
        .Cast => |*cast_expr| {
            try foldExpr(ctx, cast_expr.operand, true);
        },
        .Comptime => |*ct| {
            for (ct.block.statements) |*stmt| try foldStmt(ctx, stmt);
        },
        .Old => |*old_expr| {
            try foldExpr(ctx, old_expr.expr, true);
        },
        .Quantified => |*q| {
            if (q.condition) |cond| try foldExpr(ctx, cond, true);
            try foldExpr(ctx, q.body, true);
        },
        .Tuple => |*t| {
            for (t.elements) |el| try foldExpr(ctx, el, true);
        },
        .SwitchExpression => |*sw| {
            try foldExpr(ctx, sw.condition, true);
            for (sw.cases) |*case| {
                try foldSwitchPattern(ctx, &case.pattern);
                switch (case.body) {
                    .Expression => |body_expr| try foldExpr(ctx, body_expr, true),
                    .Block => |*case_block| {
                        const key: usize = @intFromPtr(case_block);
                        try foldBlock(ctx, case_block, key);
                    },
                    .LabeledBlock => |*labeled| {
                        const key: usize = @intFromPtr(&labeled.block);
                        try foldBlock(ctx, &labeled.block, key);
                    },
                }
            }
            if (sw.default_case) |*default_block| {
                const key: usize = @intFromPtr(default_block);
                try foldBlock(ctx, default_block, key);
            }
        },
        .Try => |*try_expr| {
            try foldExpr(ctx, try_expr.expr, true);
        },
        .ErrorCast => |*err_cast| {
            try foldExpr(ctx, err_cast.operand, true);
        },
        .Shift => |*shift| {
            try foldExpr(ctx, shift.mapping, true);
            try foldExpr(ctx, shift.source, true);
            try foldExpr(ctx, shift.dest, true);
            try foldExpr(ctx, shift.amount, true);
        },
        .StructInstantiation => |*si| {
            for (si.fields) |*field| {
                try foldExpr(ctx, field.value, true);
            }
        },
        .AnonymousStruct => |*anon| {
            for (anon.fields) |*field| {
                try foldExpr(ctx, field.value, true);
            }
        },
        .Range => |*range| {
            try foldExpr(ctx, range.start, true);
            try foldExpr(ctx, range.end, true);
        },
        .LabeledBlock => |*labeled| {
            const key: usize = @intFromPtr(&labeled.block);
            try foldBlock(ctx, &labeled.block, key);
        },
        .Destructuring => |*destr| {
            try foldExpr(ctx, destr.value, true);
        },
        .ArrayLiteral => |*arr| {
            for (arr.elements) |el| try foldExpr(ctx, el, true);
        },
        else => {},
    }

    if (!allow_fold) return;

    // Don't fold identifiers that are function parameters (runtime values)
    if (expr.* == .Identifier) {
        const is_param = if (ctx.current_scope) |scope|
            if (SymbolTable.findUp(scope, expr.Identifier.name)) |sym| sym.kind == .Param else false
        else
            false;
        if (is_param) return;
    }

    const eval_result = ctx.core_resolver.evaluateConstantExpression(expr);
    const ct_val = eval_result.getValue() orelse return;
    const span = getExpressionSpan(expr);
    switch (ct_val) {
        .integer => |int_val| {
            var ty = getExpressionTypeInfo(expr);
            if (!ty.isResolved()) {
                ty = CommonTypes.unknown_integer();
            } else if (ty.ora_type) |ot| {
                switch (ot) {
                    .min_value => |mv| ty = TypeInfo.fromOraType(mv.base.*),
                    .max_value => |mv| ty = TypeInfo.fromOraType(mv.base.*),
                    .in_range => |ir| ty = TypeInfo.fromOraType(ir.base.*),
                    .scaled => |s| ty = TypeInfo.fromOraType(s.base.*),
                    .exact => |e| ty = TypeInfo.fromOraType(e.*),
                    else => {},
                }
            }
            // Don't fold if value overflows the target type
            if (ty.ora_type) |ot| {
                if (!fitsInType(int_val, ot)) return;
            }
            const value_str = try std.fmt.allocPrint(ctx.type_storage_allocator, "{}", .{int_val});
            expr.* = .{ .Literal = .{ .Integer = .{
                .value = value_str,
                .type_info = ty,
                .span = span,
            } } };
        },
        .boolean => |bool_val| {
            var ty = getExpressionTypeInfo(expr);
            if (!ty.isResolved()) {
                ty = CommonTypes.bool_type();
            }
            expr.* = .{ .Literal = .{ .Bool = .{
                .value = bool_val,
                .type_info = ty,
                .span = span,
            } } };
        },
        else => return,
    }
}

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
        .Cast => |cast_expr| cast_expr.target_type,
        .Comptime => |ct| ct.type_info,
        else => TypeInfo.unknown(),
    };
}

fn getExpressionSpan(expr: *ast.Expressions.ExprNode) ast.SourceSpan {
    return switch (expr.*) {
        .Identifier => |id| id.span,
        .Literal => |lit| switch (lit) {
            .Integer => |int_lit| int_lit.span,
            .String => |str_lit| str_lit.span,
            .Bool => |bool_lit| bool_lit.span,
            .Address => |addr_lit| addr_lit.span,
            .Hex => |hex_lit| hex_lit.span,
            .Binary => |bin_lit| bin_lit.span,
            .Character => |char_lit| char_lit.span,
            .Bytes => |bytes_lit| bytes_lit.span,
        },
        .Binary => |bin| bin.span,
        .Unary => |unary| unary.span,
        .Assignment => |assign| assign.span,
        .CompoundAssignment => |compound| compound.span,
        .Call => |call| call.span,
        .Index => |idx| idx.span,
        .FieldAccess => |fa| fa.span,
        .Cast => |cast_expr| cast_expr.span,
        .Comptime => |ct| ct.span,
        .Old => |old_expr| old_expr.span,
        .Quantified => |q| q.span,
        .Tuple => |t| t.span,
        .SwitchExpression => |sw| sw.span,
        .Try => |tr| tr.span,
        .ErrorReturn => |err| err.span,
        .ErrorCast => |err| err.span,
        .Shift => |shift| shift.span,
        .StructInstantiation => |si| si.span,
        .AnonymousStruct => |anon| anon.span,
        .Range => |range| range.span,
        .LabeledBlock => |lb| lb.span,
        .Destructuring => |destr| destr.span,
        .EnumLiteral => |el| el.span,
        .ArrayLiteral => |arr| arr.span,
    };
}

/// When both operands of a binary expression are compile-time known but the
/// evaluator fails (overflow, underflow, division by zero), re-evaluate in
/// strict mode to extract the error and emit a compile-time diagnostic.
/// Check if a u256 value fits within the range of the given OraType.
fn fitsInType(val: u256, ot: type_info_mod.OraType) bool {
    return switch (ot) {
        .u8 => val <= std.math.maxInt(u8),
        .u16 => val <= std.math.maxInt(u16),
        .u32 => val <= std.math.maxInt(u32),
        .u64 => val <= std.math.maxInt(u64),
        .u128 => val <= std.math.maxInt(u128),
        .u256 => true,
        .i8 => val <= @as(u256, @intCast(std.math.maxInt(i8))),
        .i16 => val <= @as(u256, @intCast(std.math.maxInt(i16))),
        .i32 => val <= @as(u256, @intCast(std.math.maxInt(i32))),
        .i64 => val <= @as(u256, @intCast(std.math.maxInt(i64))),
        .i128 => val <= @as(u256, @intCast(std.math.maxInt(i128))),
        .i256 => val <= @as(u256, @intCast(std.math.maxInt(i256))),
        else => true, // non-integer types: don't constrain
    };
}

/// Both operands are compile-time known but the result failed — re-evaluate
/// in strict mode to extract the precise error and emit a compile-time warning.
/// The MLIR is still emitted (with the runtime overflow guard), so pub functions
/// remain in the ABI.
fn emitComptimeArithError(ctx: *FoldContext, expr: *ast.Expressions.ExprNode) void {
    const lookup = comptime_eval.IdentifierLookup{
        .ctx = ctx.core_resolver,
        .lookupFn = core.lookupComptimeValueThunk,
        .enumLookupFn = core.lookupEnumValueThunk,
    };
    var eval = comptime_eval.AstEvaluator.init(
        &ctx.core_resolver.comptime_env,
        .must_eval,
        .strict,
        lookup,
    );
    const strict_result = eval.evalExpr(expr);
    switch (strict_result) {
        .err => |ct_err| {
            const span = getExpressionSpan(expr);
            log.warn("comptime arithmetic error at line {d}: {s}\n", .{ span.line, ct_err.message });
        },
        else => {},
    }
}

fn setLiteralSpan(lit: *ast.Expressions.LiteralExpr, span: ast.SourceSpan) void {
    switch (lit.*) {
        .Integer => |*int_lit| int_lit.span = span,
        .String => |*str_lit| str_lit.span = span,
        .Bool => |*bool_lit| bool_lit.span = span,
        .Address => |*addr_lit| addr_lit.span = span,
        .Hex => |*hex_lit| hex_lit.span = span,
        .Binary => |*bin_lit| bin_lit.span = span,
        .Character => |*char_lit| char_lit.span = span,
        .Bytes => |*bytes_lit| bytes_lit.span = span,
    }
}

/// Evaluate a comptime while loop at compile time.
/// On success: bindings are updated in comptime_env, is_comptime stays true (loop skipped in MLIR).
/// On failure: is_comptime is cleared so the loop falls back to runtime emission.
fn tryEvalComptimeWhile(ctx: *FoldContext, while_stmt: *ast.Statements.WhileNode) void {
    const lookup = comptime_eval.IdentifierLookup{
        .ctx = ctx.core_resolver,
        .lookupFn = @import("core/mod.zig").lookupComptimeValueThunk,
        .enumLookupFn = @import("core/mod.zig").lookupEnumValueThunk,
    };
    var stmt_eval = comptime_eval.StmtEvaluator.init(
        ctx.allocator,
        &ctx.core_resolver.comptime_env,
        .try_eval,
        .forgiving,
        lookup,
    );
    const while_as_stmt = ast.Statements.StmtNode{ .While = while_stmt.* };
    const result = stmt_eval.evalStatement(&while_as_stmt);
    switch (result) {
        .ok => {}, // success — env is updated, loop will be skipped in MLIR
        else => while_stmt.is_comptime = false, // failed — fall back to runtime loop
    }
}

/// Evaluate a comptime for loop at compile time.
/// On success: bindings are updated in comptime_env, is_comptime stays true (loop skipped in MLIR).
/// On failure: is_comptime is cleared so the loop falls back to runtime emission.
fn tryEvalComptimeFor(ctx: *FoldContext, for_stmt: *ast.Statements.ForLoopNode) void {
    const lookup = comptime_eval.IdentifierLookup{
        .ctx = ctx.core_resolver,
        .lookupFn = @import("core/mod.zig").lookupComptimeValueThunk,
        .enumLookupFn = @import("core/mod.zig").lookupEnumValueThunk,
    };
    var stmt_eval = comptime_eval.StmtEvaluator.init(
        ctx.allocator,
        &ctx.core_resolver.comptime_env,
        .try_eval,
        .forgiving,
        lookup,
    );
    const for_as_stmt = ast.Statements.StmtNode{ .ForLoop = for_stmt.* };
    const result = stmt_eval.evalStatement(&for_as_stmt);
    switch (result) {
        .ok => {}, // success — loop will be skipped in MLIR
        else => for_stmt.is_comptime = false, // failed — fall back to runtime loop
    }
}

/// Must match locals_binder.getBlockScopeKey for consistency.
fn getBlockScopeKey(stmt: *const ast.Statements.StmtNode, block_id: usize) usize {
    return @intFromPtr(stmt) * 4 + block_id;
}

// ============================================================================
// Dead Code Elimination for comptime-only private functions
// ============================================================================

/// After folding, scan the contract AST for remaining call targets.
/// Private functions with zero live callers are marked `is_comptime_only = true`
/// so the MLIR emitter can skip them.
/// Uses iterative fixpoint: dead functions are excluded from subsequent scans
/// so their callees can also be collected as dead.
fn markDeadPrivateFunctions(contract: *ast.ContractNode) void {
    var changed = true;
    while (changed) {
        changed = false;
        var live_calls = std.StringHashMap(void).init(std.heap.page_allocator);
        defer live_calls.deinit();

        // Only collect call targets from live (non-dead) functions.
        // Self-calls are excluded: a recursive fn calling itself doesn't keep it live.
        for (contract.body) |child| {
            switch (child) {
                .Function => |func| {
                    if (func.is_comptime_only) continue;
                    collectCallTargetsBlock(&func.body, &live_calls, func.name);
                    // Also scan requires/ensures — they contain runtime calls too
                    for (func.requires_clauses) |clause| {
                        collectCallTargetsExprVal(clause.*, &live_calls, func.name);
                    }
                    for (func.ensures_clauses) |clause| {
                        collectCallTargetsExprVal(clause.*, &live_calls, func.name);
                    }
                },
                else => {},
            }
        }

        for (contract.body) |*child| {
            switch (child.*) {
                .Function => |*func| {
                    if (func.is_comptime_only) continue;
                    if (func.visibility != .Private) continue;
                    if (func.is_ghost) continue;
                    if (!live_calls.contains(func.name)) {
                        func.is_comptime_only = true;
                        changed = true;
                    }
                },
                else => {},
            }
        }
    }
}

fn collectCallTargetsBlock(block: *const ast.Statements.BlockNode, live: *std.StringHashMap(void), exclude: ?[]const u8) void {
    for (block.statements) |stmt| {
        collectCallTargetsStmt(&stmt, live, exclude);
    }
}

fn collectCallTargetsStmt(stmt: *const ast.Statements.StmtNode, live: *std.StringHashMap(void), exclude: ?[]const u8) void {
    switch (stmt.*) {
        .Expr => |expr| collectCallTargetsExprVal(expr, live, exclude),
        .Return => |ret| {
            if (ret.value) |v| collectCallTargetsExprVal(v, live, exclude);
        },
        .VariableDecl => |vd| {
            if (vd.value) |v| collectCallTargetsExprVal(v.*, live, exclude);
        },
        .If => |if_s| {
            collectCallTargetsExprVal(if_s.condition, live, exclude);
            collectCallTargetsBlock(&if_s.then_branch, live, exclude);
            if (if_s.else_branch) |eb| collectCallTargetsBlock(&eb, live, exclude);
        },
        .While => |w| {
            collectCallTargetsExprVal(w.condition, live, exclude);
            collectCallTargetsBlock(&w.body, live, exclude);
        },
        .ForLoop => |f| {
            collectCallTargetsExprVal(f.iterable, live, exclude);
            collectCallTargetsBlock(&f.body, live, exclude);
        },
        .Switch => |sw| {
            collectCallTargetsExprVal(sw.condition, live, exclude);
            for (sw.cases) |case| {
                switch (case.body) {
                    .Expression => |e| collectCallTargetsExprVal(e.*, live, exclude),
                    .Block => |*b| collectCallTargetsBlock(b, live, exclude),
                    .LabeledBlock => |*lb| collectCallTargetsBlock(&lb.block, live, exclude),
                }
            }
            if (sw.default_case) |*dc| collectCallTargetsBlock(dc, live, exclude);
        },
        .Assert => |a| collectCallTargetsExprVal(a.condition, live, exclude),
        .Log => |lg| {
            for (lg.args) |arg| collectCallTargetsExprVal(arg, live, exclude);
        },
        .CompoundAssignment => |ca| collectCallTargetsExprVal(ca.value.*, live, exclude),
        .TryBlock => |*tb| {
            collectCallTargetsBlock(&tb.try_block, live, exclude);
            if (tb.catch_block) |*cb| collectCallTargetsBlock(&cb.block, live, exclude);
        },
        .LabeledBlock => |*lb| collectCallTargetsBlock(&lb.block, live, exclude),
        .Break => |br| {
            if (br.value) |v| collectCallTargetsExprVal(v.*, live, exclude);
        },
        .Continue => |cont| {
            if (cont.value) |v| collectCallTargetsExprVal(v.*, live, exclude);
        },
        .DestructuringAssignment => |da| collectCallTargetsExprVal(da.value.*, live, exclude),
        else => {},
    }
}

fn collectCallTargetsExprVal(expr: ast.Expressions.ExprNode, live: *std.StringHashMap(void), exclude: ?[]const u8) void {
    switch (expr) {
        .Call => |call| {
            if (call.callee.* == .Identifier) {
                const name = call.callee.Identifier.name;
                // Skip self-calls — a recursive fn calling itself doesn't keep it live
                const is_self = if (exclude) |ex| std.mem.eql(u8, name, ex) else false;
                if (!is_self) live.put(name, {}) catch {};
            }
            for (call.arguments) |arg| collectCallTargetsExprVal(arg.*, live, exclude);
        },
        .Binary => |bin| {
            collectCallTargetsExprVal(bin.lhs.*, live, exclude);
            collectCallTargetsExprVal(bin.rhs.*, live, exclude);
        },
        .Unary => |un| collectCallTargetsExprVal(un.operand.*, live, exclude),
        .Assignment => |asgn| {
            collectCallTargetsExprVal(asgn.target.*, live, exclude);
            collectCallTargetsExprVal(asgn.value.*, live, exclude);
        },
        .CompoundAssignment => |ca| {
            collectCallTargetsExprVal(ca.target.*, live, exclude);
            collectCallTargetsExprVal(ca.value.*, live, exclude);
        },
        .Index => |idx| {
            collectCallTargetsExprVal(idx.target.*, live, exclude);
            collectCallTargetsExprVal(idx.index.*, live, exclude);
        },
        .FieldAccess => |fa| collectCallTargetsExprVal(fa.target.*, live, exclude),
        .Cast => |c_expr| collectCallTargetsExprVal(c_expr.operand.*, live, exclude),
        .Tuple => |t| {
            for (t.elements) |el| collectCallTargetsExprVal(el.*, live, exclude);
        },
        .SwitchExpression => |sw| {
            collectCallTargetsExprVal(sw.condition.*, live, exclude);
            for (sw.cases) |case| {
                switch (case.body) {
                    .Expression => |e| collectCallTargetsExprVal(e.*, live, exclude),
                    .Block => |*b| collectCallTargetsBlock(b, live, exclude),
                    .LabeledBlock => |*lb| collectCallTargetsBlock(&lb.block, live, exclude),
                }
            }
            if (sw.default_case) |*dc| collectCallTargetsBlock(dc, live, exclude);
        },
        .Try => |tr| collectCallTargetsExprVal(tr.expr.*, live, exclude),
        .ErrorCast => |ec| collectCallTargetsExprVal(ec.operand.*, live, exclude),
        .Comptime => |ct| collectCallTargetsBlock(&ct.block, live, exclude),
        .Old => |old| collectCallTargetsExprVal(old.expr.*, live, exclude),
        .Quantified => |q| {
            if (q.condition) |cond| collectCallTargetsExprVal(cond.*, live, exclude);
            collectCallTargetsExprVal(q.body.*, live, exclude);
        },
        .StructInstantiation => |si| {
            for (si.fields) |f| collectCallTargetsExprVal(f.value.*, live, exclude);
        },
        .AnonymousStruct => |anon| {
            for (anon.fields) |f| collectCallTargetsExprVal(f.value.*, live, exclude);
        },
        .ArrayLiteral => |arr| {
            for (arr.elements) |el| collectCallTargetsExprVal(el.*, live, exclude);
        },
        .Destructuring => |d| collectCallTargetsExprVal(d.value.*, live, exclude),
        .Range => |r| {
            collectCallTargetsExprVal(r.start.*, live, exclude);
            collectCallTargetsExprVal(r.end.*, live, exclude);
        },
        .LabeledBlock => |lb| collectCallTargetsBlock(&lb.block, live, exclude),
        .Shift => |sh| {
            collectCallTargetsExprVal(sh.mapping.*, live, exclude);
            collectCallTargetsExprVal(sh.source.*, live, exclude);
            collectCallTargetsExprVal(sh.dest.*, live, exclude);
            collectCallTargetsExprVal(sh.amount.*, live, exclude);
        },
        else => {},
    }
}
