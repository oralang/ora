// ============================================================================
// Comptime Folding Pass
// ============================================================================
// Replace constant expressions with integer literals after type resolution.
// Uses CoreResolver's const table for identifier resolution.
// ============================================================================

const std = @import("std");
const builtin = @import("builtin");
const ast = @import("../../ast.zig");
const state = @import("../../semantics/state.zig");
const core = @import("core/mod.zig");
const stmt_resolver = @import("core/statement.zig");
const type_info_mod = @import("../type_info.zig");
const comptime_eval = @import("../../comptime/mod.zig");
const mono = @import("monomorphize.zig");
const log = @import("log");

const CtValue = comptime_eval.CtValue;

const TypeInfo = type_info_mod.TypeInfo;
const CommonTypes = type_info_mod.CommonTypes;
const SymbolTable = state.SymbolTable;
const Scope = state.Scope;

const FoldError = error{
    OutOfMemory,
    ComptimeArithmeticError,
    ComptimeEvaluationError,
};

const FoldMode = enum {
    /// Preserve historical behavior: compile-time-known checked arithmetic can hard-fail.
    strict,
    /// Optimization-only folding: never escalate arithmetic not_constant into hard errors.
    relaxed,
    /// Proof-only context (requires/ensures/ghost assumptions).
    proof,
};

fn modeAllowsHardComptimeError(mode: FoldMode) bool {
    return mode == .strict;
}

fn branchFoldMode(mode: FoldMode) FoldMode {
    return switch (mode) {
        .strict => .relaxed,
        .relaxed, .proof => mode,
    };
}

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
                    if (!builtin.is_test) {
                        log.err("comptime value {d} overflows type '{s}' at line {d}\n", .{
                            int_val, @tagName(ot), span.line,
                        });
                    }
                    return error.ComptimeArithmeticError;
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
    monomorphizer: ?*mono.Monomorphizer = null,
    /// Function registry for looking up generic function templates
    function_registry: ?*std.StringHashMap(*ast.FunctionNode) = null,
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
                const had_wrapping_op = exprContainsWrappingOperator(value);
                try foldExpr(ctx, value, true, .strict);
                if (var_decl.type_info.ora_type) |target_ora_type| {
                    var_decl.skip_guard = stmt_resolver.shouldSkipGuard(ctx.core_resolver, value, target_ora_type);
                }
                if (var_decl.kind != .Var) {
                    const const_result = ctx.core_resolver.evaluateConstantExpression(value);
                    if (const_result.getValue()) |ct_value| {
                        const normalized = normalizeWrappingConstForDeclaredType(had_wrapping_op, var_decl.type_info, ct_value);
                        _ = try tryFoldValue(ctx, value, normalized, getExpressionSpan(value), var_decl.type_info);
                        if (ctx.current_scope) |scope| {
                            try ctx.core_resolver.setComptimeValue(scope, var_decl.name, normalized);
                        }
                    }
                }
            }
        },
        .Constant => |*constant| {
            try foldExpr(ctx, constant.value, true, .strict);
        },
        .ContractInvariant => |*invariant| {
            try foldExpr(ctx, invariant.condition, true, .proof);
        },
        .Block => |*block| {
            try foldBlock(ctx, block, null, .strict);
        },
        .Statement => |stmt| {
            try foldStmt(ctx, @constCast(stmt), .strict);
        },
        .TryBlock => |*try_block| {
            try foldBlock(ctx, &try_block.try_block, null, .strict);
            if (try_block.catch_block) |*catch_block| {
                try foldBlock(ctx, &catch_block.block, null, .strict);
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
    try ctx.core_resolver.comptime_env.pushScope(false);
    defer {
        ctx.core_resolver.comptime_env.popScope();
        ctx.current_scope = prev_scope;
        ctx.core_resolver.current_scope = prev_scope;
        ctx.core_resolver.current_function_return_type = prev_return_type;
    }

    for (function.parameters) |*param| {
        if (param.default_value) |default_val| {
            try foldExpr(ctx, default_val, true, .strict);
        }
    }
    for (function.requires_clauses) |clause| {
        try foldExpr(ctx, clause, true, .proof);
    }
    for (function.ensures_clauses) |clause| {
        try foldExpr(ctx, clause, true, .proof);
    }
    const body_key: usize = @intFromPtr(&function.body);
    try foldBlock(ctx, &function.body, body_key, .strict);
}

fn foldBlock(ctx: *FoldContext, block: *ast.Statements.BlockNode, scope_key: ?usize, mode: FoldMode) FoldError!void {
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
        try foldStmt(ctx, stmt, mode);
    }
}

fn foldStmt(ctx: *FoldContext, stmt: *ast.Statements.StmtNode, mode: FoldMode) FoldError!void {
    switch (stmt.*) {
        .Expr => |*expr| try foldExpr(ctx, expr, true, mode),
        .VariableDecl => |*var_decl| {
            if (var_decl.value) |value| {
                const had_wrapping_op = exprContainsWrappingOperator(value);
                try foldExpr(ctx, value, true, mode);
                if (var_decl.type_info.ora_type) |target_ora_type| {
                    var_decl.skip_guard = stmt_resolver.shouldSkipGuard(ctx.core_resolver, value, target_ora_type);
                }
                // Keep initializer values in the comptime environment for local
                // statement evaluation (notably explicit comptime while/for).
                // Knownness classification below explicitly treats mutable vars
                // as runtime so this does not force strict comptime errors.
                const const_result = ctx.core_resolver.evaluateConstantExpression(value);
                if (const_result.getValue()) |ct_value| {
                    const normalized = normalizeWrappingConstForDeclaredType(had_wrapping_op, var_decl.type_info, ct_value);
                    _ = try tryFoldValue(ctx, value, normalized, getExpressionSpan(value), var_decl.type_info);
                    if (ctx.current_scope) |scope| {
                        try ctx.core_resolver.setComptimeValue(scope, var_decl.name, normalized);
                    }
                }
            }
        },
        .DestructuringAssignment => |*dest| {
            try foldExpr(ctx, dest.value, true, mode);
        },
        .Return => |*ret| {
            if (ret.value) |*value| {
                try foldExpr(ctx, value, true, mode);
                if (ctx.core_resolver.current_function_return_type) |return_type| {
                    if (return_type.ora_type) |target_ora_type| {
                        ret.skip_guard = stmt_resolver.shouldSkipGuard(ctx.core_resolver, value, target_ora_type);
                    }
                }
            }
        },
        .If => |*if_stmt| {
            try foldExpr(ctx, &if_stmt.condition, true, mode);
            const then_key = getBlockScopeKey(stmt, 0);
            const cond_known = foldConditionKnownBool(ctx, &if_stmt.condition);
            if (cond_known) |is_true| {
                if (is_true) {
                    try foldBlock(ctx, &if_stmt.then_branch, then_key, mode);
                } else if (if_stmt.else_branch) |*else_block| {
                    const else_key = getBlockScopeKey(stmt, 1);
                    try foldBlock(ctx, else_block, else_key, mode);
                }
            } else {
                const bm = branchFoldMode(mode);
                try foldBlock(ctx, &if_stmt.then_branch, then_key, bm);
                if (if_stmt.else_branch) |*else_block| {
                    const else_key = getBlockScopeKey(stmt, 1);
                    try foldBlock(ctx, else_block, else_key, bm);
                }
            }
        },
        .While => |*while_stmt| {
            if (while_stmt.is_comptime) {
                // Comptime while: evaluate entirely at compile time
                try tryEvalComptimeWhile(ctx, while_stmt);
            }
            try foldExpr(ctx, &while_stmt.condition, true, mode);
            for (while_stmt.invariants) |*inv| {
                try foldExpr(ctx, inv, true, .proof);
            }
            if (while_stmt.decreases) |dec| try foldExpr(ctx, dec, true, .proof);
            if (while_stmt.increases) |inc| try foldExpr(ctx, inc, true, .proof);
            const body_key = getBlockScopeKey(stmt, 0);
            try foldBlock(ctx, &while_stmt.body, body_key, mode);
        },
        .ForLoop => |*for_stmt| {
            if (for_stmt.is_comptime) {
                // Comptime for: evaluate entirely at compile time
                try tryEvalComptimeFor(ctx, for_stmt);
            }
            try foldExpr(ctx, &for_stmt.iterable, true, mode);
            for (for_stmt.invariants) |*inv| {
                try foldExpr(ctx, inv, true, .proof);
            }
            if (for_stmt.decreases) |dec| try foldExpr(ctx, dec, true, .proof);
            if (for_stmt.increases) |inc| try foldExpr(ctx, inc, true, .proof);
            const body_key = getBlockScopeKey(stmt, 0);
            try foldBlock(ctx, &for_stmt.body, body_key, mode);
        },
        .Log => |*log_stmt| {
            for (log_stmt.args) |*arg| {
                try foldExpr(ctx, arg, true, mode);
            }
        },
        .Assert => |*assert_stmt| try foldExpr(ctx, &assert_stmt.condition, true, mode),
        .Invariant => |*inv| try foldExpr(ctx, &inv.condition, true, .proof),
        .Requires => |*req| try foldExpr(ctx, &req.condition, true, .proof),
        .Ensures => |*ens| try foldExpr(ctx, &ens.condition, true, .proof),
        .Assume => |*assume| try foldExpr(ctx, &assume.condition, true, .proof),
        .Switch => |*switch_stmt| {
            try foldExpr(ctx, &switch_stmt.condition, true, mode);
            for (switch_stmt.cases) |*case| {
                try foldSwitchPattern(ctx, &case.pattern, mode);
                switch (case.body) {
                    .Expression => |expr| try foldExpr(ctx, expr, true, mode),
                    .Block => |*case_block| {
                        const key: usize = @intFromPtr(case_block);
                        try foldBlock(ctx, case_block, key, mode);
                    },
                    .LabeledBlock => |*labeled| {
                        const key: usize = @intFromPtr(&labeled.block);
                        try foldBlock(ctx, &labeled.block, key, mode);
                    },
                }
            }
            if (switch_stmt.default_case) |*default_block| {
                const key: usize = @intFromPtr(default_block);
                try foldBlock(ctx, default_block, key, mode);
            }
        },
        .TryBlock => |*try_block| {
            const try_key = getBlockScopeKey(stmt, 0);
            try foldBlock(ctx, &try_block.try_block, try_key, mode);
            if (try_block.catch_block) |*catch_block| {
                const catch_key = getBlockScopeKey(stmt, 1);
                try foldBlock(ctx, &catch_block.block, catch_key, mode);
            }
        },
        .CompoundAssignment => |*compound| {
            try foldExpr(ctx, compound.value, true, mode);
        },
        .Lock => |*lock_stmt| try foldExpr(ctx, &lock_stmt.path, true, mode),
        .Unlock => |*unlock_stmt| try foldExpr(ctx, &unlock_stmt.path, true, mode),
        .Break => |*break_stmt| {
            if (break_stmt.value) |value| try foldExpr(ctx, value, true, mode);
        },
        .Continue => |*continue_stmt| {
            if (continue_stmt.value) |value| try foldExpr(ctx, value, true, mode);
        },
        .LabeledBlock => |*labeled| {
            const key: usize = @intFromPtr(&labeled.block);
            try foldBlock(ctx, &labeled.block, key, mode);
        },
        else => {},
    }
}

fn foldConditionKnownBool(ctx: *FoldContext, expr: *ast.Expressions.ExprNode) ?bool {
    const result = ctx.core_resolver.evaluateConstantExpression(expr);
    return switch (result) {
        .value => |v| switch (v) {
            .boolean => |b| b,
            .integer => |n| n != 0,
            else => null,
        },
        else => null,
    };
}

fn foldSwitchPattern(ctx: *FoldContext, pattern: *ast.Expressions.SwitchPattern, mode: FoldMode) FoldError!void {
    switch (pattern.*) {
        .Literal => |*lit| {
            var expr = ast.Expressions.ExprNode{ .Literal = lit.value };
            try foldExpr(ctx, &expr, true, mode);
            if (expr == .Literal) {
                lit.value = expr.Literal;
            }
        },
        .Range => |*range| {
            try foldExpr(ctx, range.start, true, mode);
            try foldExpr(ctx, range.end, true, mode);
        },
        else => {},
    }
}

fn foldExpr(ctx: *FoldContext, expr: *ast.Expressions.ExprNode, allow_fold: bool, mode: FoldMode) FoldError!void {
    switch (expr.*) {
        .Identifier => |*id| {
            if (allow_fold) {
                if (!isDefinitelyComptimeKnown(ctx, expr)) return;
                if (ctx.core_resolver.lookupComptimeValue(id.name)) |val| {
                    _ = try tryFoldValue(ctx, expr, val, getExpressionSpan(expr), id.type_info);
                }
            }
        },
        .Binary => |*bin| {
            try foldExpr(ctx, bin.lhs, true, mode);
            try foldExpr(ctx, bin.rhs, true, mode);
            if (allow_fold and isDefinitelyComptimeKnown(ctx, expr)) {
                const eval_result = ctx.core_resolver.evaluateConstantExpression(expr);
                if (eval_result.getValue()) |ct_val| {
                    if (try tryFoldValue(ctx, expr, ct_val, getExpressionSpan(expr), getExpressionTypeInfo(expr))) return;
                } else if ((bin.lhs.* == .Literal and bin.rhs.* == .Literal) and modeAllowsHardComptimeError(mode)) {
                    // Both operands are compile-time known but checked arithmetic failed.
                    // This is a hard compile-time error: do not lower guaranteed-invalid code.
                    try emitComptimeArithError(ctx, expr);
                }
            }
        },
        .Unary => |*unary| {
            try foldExpr(ctx, unary.operand, true, mode);
            if (allow_fold and isDefinitelyComptimeKnown(ctx, expr)) {
                if (ctx.core_resolver.evaluateConstantExpression(expr).getValue()) |ct_val| {
                    if (try tryFoldValue(ctx, expr, ct_val, getExpressionSpan(expr), getExpressionTypeInfo(expr))) return;
                }
            }
        },
        .Assignment => |*assign| {
            try foldExpr(ctx, assign.value, true, mode);
            const target_type = getExpressionTypeInfo(assign.target);
            if (target_type.ora_type) |target_ora_type| {
                assign.skip_guard = stmt_resolver.shouldSkipGuard(ctx.core_resolver, assign.value, target_ora_type);
            }
        },
        .CompoundAssignment => |*compound| {
            try foldExpr(ctx, compound.value, true, mode);
        },
        .Call => |*call| {
            for (call.arguments) |arg| try foldExpr(ctx, arg, true, mode);
            // Try to evaluate the entire call at comptime (pure fn + all args known)
            if (allow_fold and isDefinitelyComptimeKnown(ctx, expr)) {
                if (ctx.core_resolver.evaluateConstantExpression(expr).getValue()) |ct_val| {
                    if (try tryFoldValue(ctx, expr, ct_val, getExpressionSpan(expr), getExpressionTypeInfo(expr))) return;
                }
            }
            // If call targets a generic function, monomorphize and rewrite the call
            try tryMonomorphizeCall(ctx, call);
        },
        .Index => |*idx| {
            try foldExpr(ctx, idx.target, true, mode);
            try foldExpr(ctx, idx.index, true, mode);
        },
        .FieldAccess => |*fa| {
            // fold target first so identifier -> tuple literal can be applied
            try foldExpr(ctx, fa.target, true, mode);
            if (allow_fold and isDefinitelyComptimeKnown(ctx, expr)) {
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
            try foldExpr(ctx, cast_expr.operand, true, mode);
        },
        .Comptime => |*ct| {
            for (ct.block.statements) |*stmt| try foldStmt(ctx, stmt, .strict);
        },
        .Old => |*old_expr| {
            try foldExpr(ctx, old_expr.expr, true, mode);
        },
        .Quantified => |*q| {
            if (q.condition) |cond| try foldExpr(ctx, cond, true, .proof);
            try foldExpr(ctx, q.body, true, .proof);
        },
        .Tuple => |*t| {
            for (t.elements) |el| try foldExpr(ctx, el, true, mode);
        },
        .SwitchExpression => |*sw| {
            try foldExpr(ctx, sw.condition, true, mode);
            for (sw.cases) |*case| {
                try foldSwitchPattern(ctx, &case.pattern, mode);
                switch (case.body) {
                    .Expression => |body_expr| try foldExpr(ctx, body_expr, true, mode),
                    .Block => |*case_block| {
                        const key: usize = @intFromPtr(case_block);
                        try foldBlock(ctx, case_block, key, mode);
                    },
                    .LabeledBlock => |*labeled| {
                        const key: usize = @intFromPtr(&labeled.block);
                        try foldBlock(ctx, &labeled.block, key, mode);
                    },
                }
            }
            if (sw.default_case) |*default_block| {
                const key: usize = @intFromPtr(default_block);
                try foldBlock(ctx, default_block, key, mode);
            }
        },
        .Try => |*try_expr| {
            try foldExpr(ctx, try_expr.expr, true, mode);
        },
        .ErrorCast => |*err_cast| {
            try foldExpr(ctx, err_cast.operand, true, mode);
        },
        .Shift => |*shift| {
            try foldExpr(ctx, shift.mapping, true, mode);
            try foldExpr(ctx, shift.source, true, mode);
            try foldExpr(ctx, shift.dest, true, mode);
            try foldExpr(ctx, shift.amount, true, mode);
        },
        .StructInstantiation => |*si| {
            for (si.fields) |*field| {
                try foldExpr(ctx, field.value, true, mode);
            }
        },
        .AnonymousStruct => |*anon| {
            for (anon.fields) |*field| {
                try foldExpr(ctx, field.value, true, mode);
            }
        },
        .Range => |*range| {
            try foldExpr(ctx, range.start, true, mode);
            try foldExpr(ctx, range.end, true, mode);
        },
        .LabeledBlock => |*labeled| {
            const key: usize = @intFromPtr(&labeled.block);
            try foldBlock(ctx, &labeled.block, key, mode);
        },
        .Destructuring => |*destr| {
            try foldExpr(ctx, destr.value, true, mode);
        },
        .ArrayLiteral => |*arr| {
            for (arr.elements) |el| try foldExpr(ctx, el, true, mode);
        },
        else => {},
    }

    if (!allow_fold) return;

    // Only fold expressions that are provably compile-time known.
    // This prevents freezing mutable/runtime bindings into stale literals.
    if (!isDefinitelyComptimeKnown(ctx, expr)) return;

    const span = getExpressionSpan(expr);
    const eval_result = ctx.core_resolver.evaluateConstantExpression(expr);
    const ct_val = switch (eval_result) {
        .value => |v| v,
        .err => |ct_err| {
            if (modeAllowsHardComptimeError(mode)) {
                logComptimeStrictFailure(span, ct_err.message);
                return foldErrorFromCtError(ct_err.kind);
            }
            return;
        },
        .not_constant => {
            if (modeAllowsHardComptimeError(mode) and isDefinitelyComptimeKnown(ctx, expr)) {
                try emitComptimeArithError(ctx, expr);
            }
            return;
        },
    };
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

/// If the call targets a generic function, monomorphize it and rewrite the
/// call site to target the mangled concrete name, stripping type args.
fn tryMonomorphizeCall(ctx: *FoldContext, call: *ast.Expressions.CallExpr) FoldError!void {
    const monomorphizer = ctx.monomorphizer orelse return;
    const registry = ctx.function_registry orelse return;

    // Only handle direct calls (Identifier callee)
    const fn_name = switch (call.callee.*) {
        .Identifier => |id| id.name,
        else => return,
    };

    const generic_fn = registry.get(fn_name) orelse return;
    if (!generic_fn.is_generic) return;

    // Extract concrete type args from the call (heap-allocated, temporary)
    const type_args = mono.extractTypeArgs(generic_fn, call, ctx.allocator) orelse return;
    defer ctx.allocator.free(type_args);

    // Monomorphize
    const owner_contract = ctx.symbol_table.findEnclosingContractName(ctx.current_scope);
    const mangled_name = try monomorphizer.monomorphize(generic_fn, type_args, owner_contract);

    // Rewrite call: change callee name and strip type arguments
    call.callee.Identifier.name = mangled_name;

    // Strip the comptime type arguments from the argument list
    // Use type_storage_allocator (arena) since the result persists in the AST
    var new_args = std.ArrayList(*ast.Expressions.ExprNode){};
    var arg_idx: usize = 0;
    for (generic_fn.parameters) |param| {
        if (arg_idx >= call.arguments.len) break;
        if (param.is_comptime) {
            if (param.type_info.ora_type) |ot| {
                if (ot == .type) {
                    arg_idx += 1;
                    continue; // skip type argument
                }
            }
        }
        try new_args.append(ctx.type_storage_allocator, call.arguments[arg_idx]);
        arg_idx += 1;
    }
    call.arguments = try new_args.toOwnedSlice(ctx.type_storage_allocator);

    // Update call type_info to reflect the concrete return type
    if (generic_fn.return_type_info) |ret_ti| {
        if (ret_ti.ora_type) |ot| {
            if (ot == .type_parameter) {
                // Find the matching type arg
                var ti: usize = 0;
                for (generic_fn.parameters) |param| {
                    if (param.is_comptime) {
                        if (param.type_info.ora_type) |pot| {
                            if (pot == .type) {
                                if (std.mem.eql(u8, param.name, ot.type_parameter)) {
                                    if (ti < type_args.len) {
                                        const concrete = type_args[ti];
                                        call.type_info = TypeInfo.inferred(concrete.getCategory(), concrete, call.span);
                                    }
                                    break;
                                }
                                ti += 1;
                            }
                        }
                    }
                }
            }
        }
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

fn isDefinitelyComptimeKnown(ctx: *FoldContext, expr: *const ast.Expressions.ExprNode) bool {
    return switch (expr.*) {
        .Literal, .EnumLiteral => true,
        .Identifier => |id| blk: {
            // Function parameters are runtime values by definition.
            if (ctx.current_scope) |scope| {
                if (SymbolTable.findUp(scope, id.name)) |sym| {
                    if (sym.kind == .Param) break :blk false;
                    // Mutable vars are runtime bindings even if they currently
                    // have a value in the comptime environment.
                    if (sym.kind == .Var and sym.mutable) break :blk false;
                }
            }
            if (ctx.core_resolver.lookupComptimeValue(id.name)) |val| {
                break :blk isStrictlyFoldableCtValue(val);
            }
            break :blk false;
        },
        .Unary => |unary| isDefinitelyComptimeKnown(ctx, unary.operand),
        .Binary => |bin| isDefinitelyComptimeKnown(ctx, bin.lhs) and
            isDefinitelyComptimeKnown(ctx, bin.rhs),
        .Cast => |cast_expr| isDefinitelyComptimeKnown(ctx, cast_expr.operand),
        .Index => |idx| isDefinitelyComptimeKnown(ctx, idx.target) and
            isDefinitelyComptimeKnown(ctx, idx.index),
        .FieldAccess => |fa| isFieldAccessDefinitelyComptimeKnown(ctx, &fa),
        .Tuple => |t| blk: {
            for (t.elements) |el| {
                if (!isDefinitelyComptimeKnown(ctx, el)) break :blk false;
            }
            break :blk true;
        },
        .ArrayLiteral => |arr| blk: {
            for (arr.elements) |el| {
                if (!isDefinitelyComptimeKnown(ctx, el)) break :blk false;
            }
            break :blk true;
        },
        .StructInstantiation => |si| blk: {
            for (si.fields) |field| {
                if (!isDefinitelyComptimeKnown(ctx, field.value)) break :blk false;
            }
            break :blk true;
        },
        .AnonymousStruct => |anon| blk: {
            for (anon.fields) |field| {
                if (!isDefinitelyComptimeKnown(ctx, field.value)) break :blk false;
            }
            break :blk true;
        },
        .Range => |range| isDefinitelyComptimeKnown(ctx, range.start) and
            isDefinitelyComptimeKnown(ctx, range.end),
        .Comptime => |*ct| isBlockDefinitelyComptimeKnown(ctx, &ct.block),
        .Old => |old_expr| isDefinitelyComptimeKnown(ctx, old_expr.expr),
        .Quantified => |q| (if (q.condition) |cond| isDefinitelyComptimeKnown(ctx, cond) else true) and
            isDefinitelyComptimeKnown(ctx, q.body),
        .SwitchExpression => |sw| blk: {
            if (!isDefinitelyComptimeKnown(ctx, sw.condition)) break :blk false;
            for (sw.cases) |*case| {
                if (!isSwitchPatternDefinitelyComptimeKnown(ctx, &case.pattern)) break :blk false;
                if (!isSwitchBodyDefinitelyComptimeKnown(ctx, &case.body)) break :blk false;
            }
            if (sw.default_case) |*default_case| {
                if (!isBlockDefinitelyComptimeKnown(ctx, default_case)) break :blk false;
            }
            break :blk true;
        },
        .Try => |try_expr| isDefinitelyComptimeKnown(ctx, try_expr.expr),
        .ErrorCast => |err_cast| isDefinitelyComptimeKnown(ctx, err_cast.operand),
        .Shift => |shift| isDefinitelyComptimeKnown(ctx, shift.mapping) and
            isDefinitelyComptimeKnown(ctx, shift.source) and
            isDefinitelyComptimeKnown(ctx, shift.dest) and
            isDefinitelyComptimeKnown(ctx, shift.amount),
        .LabeledBlock => |*lb| isBlockDefinitelyComptimeKnown(ctx, &lb.block),
        .Destructuring => |destr| isDefinitelyComptimeKnown(ctx, destr.value),
        .Call => |call| blk: {
            // Error-propagating calls are semantically runtime-flow values (`try`/`catch`),
            // not guaranteed foldable constants.
            if (isErrorPropagatingType(call.type_info)) break :blk false;

            // Only direct calls can be judged here; member/runtime calls are not forced.
            if (call.callee.* != .Identifier) break :blk false;
            const fn_name = call.callee.Identifier.name;
            if (fn_name.len == 0) break :blk false;

            // Non-builtin function calls are only "definitely known" when purity is known.
            if (fn_name[0] != '@') {
                const effect = ctx.symbol_table.function_effects.get(fn_name) orelse break :blk false;
                if (effect != .Pure) break :blk false;
            }

            for (call.arguments) |arg| {
                if (!isDefinitelyComptimeKnown(ctx, arg)) break :blk false;
            }
            break :blk true;
        },
        else => false,
    };
}

fn isSwitchPatternDefinitelyComptimeKnown(ctx: *FoldContext, pattern: *const ast.Expressions.SwitchPattern) bool {
    return switch (pattern.*) {
        .Literal => true,
        .Range => |range| isDefinitelyComptimeKnown(ctx, range.start) and
            isDefinitelyComptimeKnown(ctx, range.end),
        .EnumValue, .Else => true,
    };
}

fn isFieldAccessDefinitelyComptimeKnown(
    ctx: *FoldContext,
    fa: *const ast.Expressions.FieldAccessExpr,
) bool {
    if (!isDefinitelyComptimeKnown(ctx, fa.target)) return false;

    // Be conservative: only force strict comptime for field accesses that the
    // AST evaluator can deterministically resolve without extra type metadata.
    return switch (fa.target.*) {
        .Tuple => |tuple| blk: {
            const field_str = if (fa.field.len > 0 and fa.field[0] == '_') fa.field[1..] else fa.field;
            const idx = std.fmt.parseInt(usize, field_str, 10) catch break :blk false;
            if (idx >= tuple.elements.len) break :blk false;
            break :blk isDefinitelyComptimeKnown(ctx, tuple.elements[idx]);
        },
        .AnonymousStruct => |anon| blk: {
            for (anon.fields) |field| {
                if (std.mem.eql(u8, field.name, fa.field)) {
                    break :blk isDefinitelyComptimeKnown(ctx, field.value);
                }
            }
            break :blk false;
        },
        .StructInstantiation => |si| blk: {
            for (si.fields) |field| {
                if (std.mem.eql(u8, field.name, fa.field)) {
                    break :blk isDefinitelyComptimeKnown(ctx, field.value);
                }
            }
            break :blk false;
        },
        .Call => |call| blk: {
            if (call.callee.* != .Identifier) break :blk false;
            const fn_name = call.callee.Identifier.name;
            if (fn_name.len == 0 or fn_name[0] != '@') break :blk false;
            if (!std.mem.eql(u8, fa.field, "value") and !std.mem.eql(u8, fa.field, "overflow")) {
                break :blk false;
            }
            for (call.arguments) |arg| {
                if (!isDefinitelyComptimeKnown(ctx, arg)) break :blk false;
            }
            break :blk true;
        },
        else => false,
    };
}

fn isSwitchBodyDefinitelyComptimeKnown(ctx: *FoldContext, body: *const ast.Expressions.SwitchBody) bool {
    return switch (body.*) {
        .Expression => |expr| isDefinitelyComptimeKnown(ctx, expr),
        .Block => |*block| isBlockDefinitelyComptimeKnown(ctx, block),
        .LabeledBlock => |*labeled| isBlockDefinitelyComptimeKnown(ctx, &labeled.block),
    };
}

fn isBlockDefinitelyComptimeKnown(ctx: *FoldContext, block: *const ast.Statements.BlockNode) bool {
    for (block.statements) |*stmt| {
        if (!isStmtDefinitelyComptimeKnown(ctx, stmt)) return false;
    }
    return true;
}

fn isStmtDefinitelyComptimeKnown(ctx: *FoldContext, stmt: *const ast.Statements.StmtNode) bool {
    return switch (stmt.*) {
        .Expr => |*expr| isDefinitelyComptimeKnown(ctx, expr),
        .VariableDecl => |*var_decl| if (var_decl.value) |value|
            isDefinitelyComptimeKnown(ctx, value)
        else
            false,
        .DestructuringAssignment => |*dest| isDefinitelyComptimeKnown(ctx, dest.value),
        .Return => |*ret| if (ret.value) |*value|
            isDefinitelyComptimeKnown(ctx, value)
        else
            true,
        .If => |*if_stmt| isDefinitelyComptimeKnown(ctx, &if_stmt.condition) and
            isBlockDefinitelyComptimeKnown(ctx, &if_stmt.then_branch) and
            (if (if_stmt.else_branch) |*else_block| isBlockDefinitelyComptimeKnown(ctx, else_block) else true),
        .Switch => |*sw| blk: {
            if (!isDefinitelyComptimeKnown(ctx, &sw.condition)) break :blk false;
            for (sw.cases) |*case| {
                if (!isSwitchPatternDefinitelyComptimeKnown(ctx, &case.pattern)) break :blk false;
                if (!isSwitchBodyDefinitelyComptimeKnown(ctx, &case.body)) break :blk false;
            }
            if (sw.default_case) |*default_case| {
                if (!isBlockDefinitelyComptimeKnown(ctx, default_case)) break :blk false;
            }
            break :blk true;
        },
        .Break => |*brk| if (brk.value) |value|
            isDefinitelyComptimeKnown(ctx, value)
        else
            true,
        .Continue => |*cont| if (cont.value) |value|
            isDefinitelyComptimeKnown(ctx, value)
        else
            true,
        .CompoundAssignment => |*compound| blk: {
            if (compound.target.* != .Identifier) break :blk false;
            if (!isDefinitelyComptimeKnown(ctx, compound.value)) break :blk false;

            const target_name = compound.target.Identifier.name;
            if (ctx.current_scope) |scope| {
                if (SymbolTable.findUp(scope, target_name)) |sym| {
                    if (sym.kind == .Param) break :blk false;
                    if (sym.kind == .Var and sym.mutable) break :blk false;
                }
            }

            if (ctx.core_resolver.lookupComptimeValue(target_name)) |val| {
                break :blk isStrictlyFoldableCtValue(val);
            }
            break :blk false;
        },
        .LabeledBlock => |*labeled| isBlockDefinitelyComptimeKnown(ctx, &labeled.block),
        .Assert => |*assert_stmt| isDefinitelyComptimeKnown(ctx, &assert_stmt.condition),
        .Invariant => |*inv| isDefinitelyComptimeKnown(ctx, &inv.condition),
        .Requires => |*req| isDefinitelyComptimeKnown(ctx, &req.condition),
        .Ensures => |*ens| isDefinitelyComptimeKnown(ctx, &ens.condition),
        .Assume => |*assume| isDefinitelyComptimeKnown(ctx, &assume.condition),
        // Conservative false for statements that are runtime-only or require
        // richer control-flow/data-flow analysis to guarantee comptime knownness.
        .While, .ForLoop, .Log, .Lock, .Unlock, .ErrorDecl, .TryBlock, .Havoc => false,
    };
}

fn isStrictlyFoldableCtValue(val: CtValue) bool {
    return switch (val) {
        .integer, .boolean, .address => true,
        else => false,
    };
}

fn isErrorPropagatingType(ty: TypeInfo) bool {
    if (ty.category == .ErrorUnion or ty.category == .Result or ty.category == .Error) return true;

    const ora_ty = ty.ora_type orelse return false;
    return switch (ora_ty) {
        .error_union => true,
        ._union => |members| members.len > 0 and members[0] == .error_union,
        else => false,
    };
}

fn normalizeWrappingConstForDeclaredType(
    had_wrapping_op: bool,
    declared_type: TypeInfo,
    val: CtValue,
) CtValue {
    if (!had_wrapping_op) return val;
    if (val != .integer) return val;

    const ot = declared_type.ora_type orelse return val;
    const base = unwrapIntegerBaseOraType(ot);
    if (!base.isUnsignedInteger()) return val;

    const width = base.bitWidth() orelse return val;
    if (width >= 256) return val;

    const shift_amt: u8 = @intCast(width);
    const mask: u256 = (@as(u256, 1) << shift_amt) - 1;
    return .{ .integer = val.integer & mask };
}

fn unwrapIntegerBaseOraType(ot: type_info_mod.OraType) type_info_mod.OraType {
    var current = ot;
    while (true) {
        switch (current) {
            .min_value => |mv| current = mv.base.*,
            .max_value => |mv| current = mv.base.*,
            .in_range => |ir| current = ir.base.*,
            .scaled => |s| current = s.base.*,
            .exact => |e| current = e.*,
            else => break,
        }
    }
    return current;
}

fn exprContainsWrappingOperator(expr: *const ast.Expressions.ExprNode) bool {
    return switch (expr.*) {
        .Binary => |bin| isWrappingBinaryOp(bin.operator) or
            exprContainsWrappingOperator(bin.lhs) or
            exprContainsWrappingOperator(bin.rhs),
        .Unary => |unary| exprContainsWrappingOperator(unary.operand),
        .Call => |call| blk: {
            if (exprContainsWrappingOperator(call.callee)) break :blk true;
            for (call.arguments) |arg| {
                if (exprContainsWrappingOperator(arg)) break :blk true;
            }
            break :blk false;
        },
        .Index => |idx| exprContainsWrappingOperator(idx.target) or
            exprContainsWrappingOperator(idx.index),
        .FieldAccess => |fa| exprContainsWrappingOperator(fa.target),
        .Cast => |cast_expr| exprContainsWrappingOperator(cast_expr.operand),
        .Tuple => |tuple_expr| blk: {
            for (tuple_expr.elements) |el| {
                if (exprContainsWrappingOperator(el)) break :blk true;
            }
            break :blk false;
        },
        .ArrayLiteral => |arr| blk: {
            for (arr.elements) |el| {
                if (exprContainsWrappingOperator(el)) break :blk true;
            }
            break :blk false;
        },
        .StructInstantiation => |si| blk: {
            for (si.fields) |field| {
                if (exprContainsWrappingOperator(field.value)) break :blk true;
            }
            break :blk false;
        },
        .AnonymousStruct => |anon| blk: {
            for (anon.fields) |field| {
                if (exprContainsWrappingOperator(field.value)) break :blk true;
            }
            break :blk false;
        },
        .Range => |range| exprContainsWrappingOperator(range.start) or
            exprContainsWrappingOperator(range.end),
        .Comptime => |ct| blockContainsWrappingOperator(&ct.block),
        .Old => |old_expr| exprContainsWrappingOperator(old_expr.expr),
        .Quantified => |q| (if (q.condition) |cond| exprContainsWrappingOperator(cond) else false) or
            exprContainsWrappingOperator(q.body),
        .SwitchExpression => |sw| blk: {
            if (exprContainsWrappingOperator(sw.condition)) break :blk true;
            for (sw.cases) |case| {
                switch (case.body) {
                    .Expression => |body_expr| if (exprContainsWrappingOperator(body_expr)) break :blk true,
                    .Block => |block| if (blockContainsWrappingOperator(&block)) break :blk true,
                    .LabeledBlock => |lb| if (blockContainsWrappingOperator(&lb.block)) break :blk true,
                }
            }
            if (sw.default_case) |default_block| {
                if (blockContainsWrappingOperator(&default_block)) break :blk true;
            }
            break :blk false;
        },
        .Try => |try_expr| exprContainsWrappingOperator(try_expr.expr),
        .ErrorCast => |err_cast| exprContainsWrappingOperator(err_cast.operand),
        .Shift => |shift| exprContainsWrappingOperator(shift.mapping) or
            exprContainsWrappingOperator(shift.source) or
            exprContainsWrappingOperator(shift.dest) or
            exprContainsWrappingOperator(shift.amount),
        .LabeledBlock => |lb| blockContainsWrappingOperator(&lb.block),
        .Destructuring => |destr| exprContainsWrappingOperator(destr.value),
        else => false,
    };
}

fn blockContainsWrappingOperator(block: *const ast.Statements.BlockNode) bool {
    for (block.statements) |*stmt| {
        if (stmtContainsWrappingOperator(stmt)) return true;
    }
    return false;
}

fn stmtContainsWrappingOperator(stmt: *const ast.Statements.StmtNode) bool {
    return switch (stmt.*) {
        .Expr => |expr| exprContainsWrappingOperator(&expr),
        .VariableDecl => |vd| if (vd.value) |value| exprContainsWrappingOperator(value) else false,
        .DestructuringAssignment => |dest| exprContainsWrappingOperator(dest.value),
        .Return => |ret| if (ret.value) |value| exprContainsWrappingOperator(&value) else false,
        .If => |if_stmt| exprContainsWrappingOperator(&if_stmt.condition) or
            blockContainsWrappingOperator(&if_stmt.then_branch) or
            (if (if_stmt.else_branch) |else_block| blockContainsWrappingOperator(&else_block) else false),
        .While => |while_stmt| exprContainsWrappingOperator(&while_stmt.condition) or
            blockContainsWrappingOperator(&while_stmt.body),
        .ForLoop => |for_stmt| exprContainsWrappingOperator(&for_stmt.iterable) or
            blockContainsWrappingOperator(&for_stmt.body),
        .Switch => |sw| blk: {
            if (exprContainsWrappingOperator(&sw.condition)) break :blk true;
            for (sw.cases) |case| {
                switch (case.body) {
                    .Expression => |body_expr| if (exprContainsWrappingOperator(body_expr)) break :blk true,
                    .Block => |block| if (blockContainsWrappingOperator(&block)) break :blk true,
                    .LabeledBlock => |lb| if (blockContainsWrappingOperator(&lb.block)) break :blk true,
                }
            }
            if (sw.default_case) |default_block| {
                if (blockContainsWrappingOperator(&default_block)) break :blk true;
            }
            break :blk false;
        },
        .TryBlock => |tb| blockContainsWrappingOperator(&tb.try_block) or
            (if (tb.catch_block) |cb| blockContainsWrappingOperator(&cb.block) else false),
        .CompoundAssignment => |ca| exprContainsWrappingOperator(ca.value),
        .Lock => |lock_stmt| exprContainsWrappingOperator(&lock_stmt.path),
        .Unlock => |unlock_stmt| exprContainsWrappingOperator(&unlock_stmt.path),
        .Break => |brk| if (brk.value) |value| exprContainsWrappingOperator(value) else false,
        .Continue => |cont| if (cont.value) |value| exprContainsWrappingOperator(value) else false,
        .LabeledBlock => |lb| blockContainsWrappingOperator(&lb.block),
        .Assert => |assert_stmt| exprContainsWrappingOperator(&assert_stmt.condition),
        .Invariant => |inv| exprContainsWrappingOperator(&inv.condition),
        .Requires => |req| exprContainsWrappingOperator(&req.condition),
        .Ensures => |ens| exprContainsWrappingOperator(&ens.condition),
        .Assume => |assume| exprContainsWrappingOperator(&assume.condition),
        else => false,
    };
}

fn isWrappingBinaryOp(op: ast.Expressions.BinaryOp) bool {
    return switch (op) {
        .WrappingAdd, .WrappingSub, .WrappingMul, .WrappingPow, .WrappingShl, .WrappingShr => true,
        else => false,
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
        .i8 => fitsInSignedWidth(val, 8),
        .i16 => fitsInSignedWidth(val, 16),
        .i32 => fitsInSignedWidth(val, 32),
        .i64 => fitsInSignedWidth(val, 64),
        .i128 => fitsInSignedWidth(val, 128),
        .i256 => fitsInSignedWidth(val, 256),
        else => true, // non-integer types: don't constrain
    };
}

/// Signed integers are carried as u256 two's-complement bit patterns.
/// A value fits iN when it is either:
/// - a non-negative value <= max(iN), or
/// - a sign-extended negative pattern in [2^256 - 2^(N-1), 2^256 - 1].
fn fitsInSignedWidth(val: u256, bits: u16) bool {
    if (bits == 256) return true;
    if (bits == 0) return false;

    const sign_bit_shift: u8 = @intCast(bits - 1);
    const positive_max: u256 = (@as(u256, 1) << sign_bit_shift) - 1;
    if (val <= positive_max) return true;

    const negative_min_encoded: u256 = @as(u256, 0) -% (@as(u256, 1) << sign_bit_shift);
    return val >= negative_min_encoded;
}

fn foldErrorFromCtError(kind: comptime_eval.CtErrorKind) FoldError {
    return if (kind.isArithmeticError())
        error.ComptimeArithmeticError
    else
        error.ComptimeEvaluationError;
}

fn logComptimeStrictFailure(span: ast.SourceSpan, message: []const u8) void {
    if (!builtin.is_test) {
        log.err("comptime evaluation error at line {d}: {s}\n", .{ span.line, message });
    }
}

/// Expression is definitely compile-time-known but failed to evaluate.
/// Re-evaluate in strict mode and convert the error into a hard compiler error.
fn emitComptimeArithError(ctx: *FoldContext, expr: *ast.Expressions.ExprNode) FoldError!void {
    const lookup = comptime_eval.IdentifierLookup{
        .ctx = ctx.core_resolver,
        .lookupFn = core.lookupComptimeValueThunk,
        .enumLookupFn = core.lookupEnumValueThunk,
    };
    const fn_lookup = comptime_eval.FunctionLookup{
        .ctx = ctx.core_resolver,
        .lookupFn = core.lookupComptimeFnThunk,
    };
    var eval = comptime_eval.AstEvaluator.initWithFnLookup(
        &ctx.core_resolver.comptime_env,
        .must_eval,
        .strict,
        lookup,
        fn_lookup,
    );
    const strict_result = eval.evalExpr(expr);
    switch (strict_result) {
        .err => |ct_err| {
            logComptimeStrictFailure(getExpressionSpan(expr), ct_err.message);
            return foldErrorFromCtError(ct_err.kind);
        },
        .not_constant => {
            // Knownness checks are intentionally conservative, but may still
            // occasionally classify a runtime expression as "definitely known".
            // If strict re-eval is also not constant, do not hard-fail here:
            // keep runtime lowering and reserve hard errors for concrete
            // arithmetic/stage evaluator failures above.
            return;
        },
        else => return,
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
/// Explicit `comptime` loops are strict: failure is a hard compile-time error.
fn tryEvalComptimeWhile(ctx: *FoldContext, while_stmt: *ast.Statements.WhileNode) FoldError!void {
    const lookup = comptime_eval.IdentifierLookup{
        .ctx = ctx.core_resolver,
        .lookupFn = @import("core/mod.zig").lookupComptimeValueThunk,
        .enumLookupFn = @import("core/mod.zig").lookupEnumValueThunk,
    };
    var stmt_eval = comptime_eval.StmtEvaluator.init(
        ctx.allocator,
        &ctx.core_resolver.comptime_env,
        .must_eval,
        .strict,
        lookup,
    );
    const while_as_stmt = ast.Statements.StmtNode{ .While = while_stmt.* };
    const result = stmt_eval.evalStatement(&while_as_stmt);
    switch (result) {
        .ok => {}, // success — env is updated, loop will be skipped in MLIR
        .err => |ct_err| {
            logComptimeStrictFailure(while_stmt.span, ct_err.message);
            return foldErrorFromCtError(ct_err.kind);
        },
        else => {
            logComptimeStrictFailure(while_stmt.span, "explicit comptime while could not be fully evaluated");
            return error.ComptimeEvaluationError;
        },
    }
}

/// Evaluate a comptime for loop at compile time.
/// Explicit `comptime` loops are strict: failure is a hard compile-time error.
fn tryEvalComptimeFor(ctx: *FoldContext, for_stmt: *ast.Statements.ForLoopNode) FoldError!void {
    const lookup = comptime_eval.IdentifierLookup{
        .ctx = ctx.core_resolver,
        .lookupFn = @import("core/mod.zig").lookupComptimeValueThunk,
        .enumLookupFn = @import("core/mod.zig").lookupEnumValueThunk,
    };
    var stmt_eval = comptime_eval.StmtEvaluator.init(
        ctx.allocator,
        &ctx.core_resolver.comptime_env,
        .must_eval,
        .strict,
        lookup,
    );
    const for_as_stmt = ast.Statements.StmtNode{ .ForLoop = for_stmt.* };
    const result = stmt_eval.evalStatement(&for_as_stmt);
    switch (result) {
        .ok => {}, // success — loop will be skipped in MLIR
        .err => |ct_err| {
            logComptimeStrictFailure(for_stmt.span, ct_err.message);
            return foldErrorFromCtError(ct_err.kind);
        },
        else => {
            logComptimeStrictFailure(for_stmt.span, "explicit comptime for could not be fully evaluated");
            return error.ComptimeEvaluationError;
        },
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
                if (!is_self) live.put(name, {}) catch @panic("failed to record live call target");
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
