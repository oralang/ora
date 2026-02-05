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
const const_eval = @import("../../const_eval.zig");

const TypeInfo = type_info_mod.TypeInfo;
const CommonTypes = type_info_mod.CommonTypes;
const SymbolTable = state.SymbolTable;
const Scope = state.Scope;

const FoldError = error{OutOfMemory};

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
                    const const_result = ctx.core_resolver.evaluateConstantExpressionWithLookup(value) catch const_eval.ConstantValue.NotConstant;
                    if (const_result != .NotConstant) {
                        if (ctx.current_scope) |scope| {
                            ctx.core_resolver.setComptimeValue(scope, var_decl.name, const_result) catch {};
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
}

fn foldFunction(ctx: *FoldContext, function: *ast.FunctionNode) FoldError!void {
    const prev_scope = ctx.current_scope;
    const prev_return_type = ctx.core_resolver.current_function_return_type;
    if (ctx.symbol_table.function_scopes.get(function.name)) |scope| {
        ctx.current_scope = scope;
        ctx.core_resolver.current_scope = scope;
    }
    ctx.core_resolver.current_function_return_type = function.return_type_info;
    defer {
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
                if (var_decl.kind != .Var) {
                    const const_result = ctx.core_resolver.evaluateConstantExpressionWithLookup(value) catch const_eval.ConstantValue.NotConstant;
                    if (const_result != .NotConstant) {
                        if (ctx.current_scope) |scope| {
                            ctx.core_resolver.setComptimeValue(scope, var_decl.name, const_result) catch {};
                        }
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
                if (ctx.core_resolver.lookupComptimeValue(id.name)) |val| {
                    const span = getExpressionSpan(expr);
                    switch (val) {
                        .Integer => {
                            const value_str = try std.fmt.allocPrint(ctx.type_storage_allocator, "{}", .{val.Integer});
                            var ty = id.type_info;
                            if (!ty.isResolved()) {
                                ty = CommonTypes.unknown_integer();
                            }
                            expr.* = .{ .Literal = .{ .Integer = .{
                                .value = value_str,
                                .type_info = ty,
                                .span = span,
                            } } };
                        },
                        .Bool => {
                            var ty = id.type_info;
                            if (!ty.isResolved()) {
                                ty = CommonTypes.bool_type();
                            }
                            expr.* = .{ .Literal = .{ .Bool = .{
                                .value = val.Bool,
                                .type_info = ty,
                                .span = span,
                            } } };
                        },
                        .Array => {
                            // Only fold arrays into tuple literals when the identifier is typed as a tuple.
                            // For arrays/slices, keep the identifier to preserve iterable semantics.
                            if (id.type_info.ora_type) |ora_ty| {
                                switch (ora_ty) {
                                    .tuple => {},
                                    .array, .slice => return,
                                    else => {},
                                }
                            } else {
                                return;
                            }

                            var elements = std.ArrayListUnmanaged(*ast.Expressions.ExprNode){};
                            defer elements.deinit(ctx.type_storage_allocator);

                            for (val.Array) |elem_val| {
                                const elem_node = try ctx.type_storage_allocator.create(ast.Expressions.ExprNode);
                                switch (elem_val) {
                                    .Integer => {
                                        const value_str = try std.fmt.allocPrint(ctx.type_storage_allocator, "{}", .{elem_val.Integer});
                                        elem_node.* = .{ .Literal = .{ .Integer = .{
                                            .value = value_str,
                                            .type_info = CommonTypes.unknown_integer(),
                                            .span = span,
                                        } } };
                                    },
                                    .Bool => {
                                        elem_node.* = .{ .Literal = .{ .Bool = .{
                                            .value = elem_val.Bool,
                                            .type_info = CommonTypes.bool_type(),
                                            .span = span,
                                        } } };
                                    },
                                    else => {
                                        // non-scalar element: don't fold identifier into tuple
                                        ctx.type_storage_allocator.destroy(elem_node);
                                        return;
                                    },
                                }
                                elements.append(ctx.type_storage_allocator, elem_node) catch return;
                            }

                            const owned = elements.toOwnedSlice(ctx.type_storage_allocator) catch return;
                            expr.* = .{ .Tuple = .{
                                .elements = owned,
                                .span = span,
                            } };
                        },
                        else => {},
                    }
                }
            }
        },
        .Binary => |*bin| {
            try foldExpr(ctx, bin.lhs, true);
            try foldExpr(ctx, bin.rhs, true);
            if (allow_fold) {
                const folded = ctx.core_resolver.evaluateConstantExpressionWithLookup(expr) catch const_eval.ConstantValue.NotConstant;
                if (folded != .NotConstant) {
                    const span = getExpressionSpan(expr);
                    switch (folded) {
                        .Integer => {
                            const value_str = try std.fmt.allocPrint(ctx.type_storage_allocator, "{}", .{folded.Integer});
                            var ty = getExpressionTypeInfo(expr);
                            if (!ty.isResolved()) {
                                ty = CommonTypes.unknown_integer();
                            }
                            expr.* = .{ .Literal = .{ .Integer = .{
                                .value = value_str,
                                .type_info = ty,
                                .span = span,
                            } } };
                            return;
                        },
                        .Bool => {
                            var ty = getExpressionTypeInfo(expr);
                            if (!ty.isResolved()) {
                                ty = CommonTypes.bool_type();
                            }
                            expr.* = .{ .Literal = .{ .Bool = .{
                                .value = folded.Bool,
                                .type_info = ty,
                                .span = span,
                            } } };
                            return;
                        },
                        else => {},
                    }
                }
            }
        },
        .Unary => |*unary| {
            try foldExpr(ctx, unary.operand, true);
            if (allow_fold) {
                const folded = ctx.core_resolver.evaluateConstantExpressionWithLookup(expr) catch const_eval.ConstantValue.NotConstant;
                if (folded != .NotConstant) {
                    const span = getExpressionSpan(expr);
                    switch (folded) {
                        .Integer => {
                            const value_str = try std.fmt.allocPrint(ctx.type_storage_allocator, "{}", .{folded.Integer});
                            var ty = getExpressionTypeInfo(expr);
                            if (!ty.isResolved()) {
                                ty = CommonTypes.unknown_integer();
                            }
                            expr.* = .{ .Literal = .{ .Integer = .{
                                .value = value_str,
                                .type_info = ty,
                                .span = span,
                            } } };
                            return;
                        },
                        .Bool => {
                            var ty = getExpressionTypeInfo(expr);
                            if (!ty.isResolved()) {
                                ty = CommonTypes.bool_type();
                            }
                            expr.* = .{ .Literal = .{ .Bool = .{
                                .value = folded.Bool,
                                .type_info = ty,
                                .span = span,
                            } } };
                            return;
                        },
                        else => {},
                    }
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
        },
        .Index => |*idx| {
            try foldExpr(ctx, idx.target, true);
            try foldExpr(ctx, idx.index, true);
        },
        .FieldAccess => |*fa| {
            // fold target first so identifier -> tuple literal can be applied
            try foldExpr(ctx, fa.target, true);
            if (allow_fold) {
            }
            if (allow_fold) {
                const folded = ctx.core_resolver.evaluateConstantExpressionWithLookup(expr) catch const_eval.ConstantValue.NotConstant;
                if (folded != .NotConstant) {
                    const span = getExpressionSpan(expr);
                    switch (folded) {
                        .Integer => {
                            const value_str = try std.fmt.allocPrint(ctx.type_storage_allocator, "{}", .{folded.Integer});
                            var ty = getExpressionTypeInfo(expr);
                            if (!ty.isResolved()) {
                                ty = CommonTypes.unknown_integer();
                            }
                            expr.* = .{ .Literal = .{ .Integer = .{
                                .value = value_str,
                                .type_info = ty,
                                .span = span,
                            } } };
                            return;
                        },
                        .Bool => {
                            var ty = getExpressionTypeInfo(expr);
                            if (!ty.isResolved()) {
                                ty = CommonTypes.bool_type();
                            }
                            expr.* = .{ .Literal = .{ .Bool = .{
                                .value = folded.Bool,
                                .type_info = ty,
                                .span = span,
                            } } };
                            return;
                        },
                        else => {},
                    }
                }
            }
            if (allow_fold and fa.target.* == .Identifier) {
                const id = &fa.target.Identifier;
                if (ctx.core_resolver.lookupComptimeValue(id.name)) |val| {
                    if (val == .Array) {
                        const field_str = if (fa.field.len > 0 and fa.field[0] == '_') fa.field[1..] else fa.field;
                        const idx = std.fmt.parseInt(usize, field_str, 10) catch return;
                        if (idx < val.Array.len) {
                            const elem_val = val.Array[idx];
                            const span = getExpressionSpan(expr);
                            switch (elem_val) {
                                .Integer => {
                                    const value_str = try std.fmt.allocPrint(ctx.type_storage_allocator, "{}", .{elem_val.Integer});
                                    var ty = getExpressionTypeInfo(expr);
                                    if (!ty.isResolved()) {
                                        ty = CommonTypes.unknown_integer();
                                    }
                                    expr.* = .{ .Literal = .{ .Integer = .{
                                        .value = value_str,
                                        .type_info = ty,
                                        .span = span,
                                    } } };
                                    return;
                                },
                                .Bool => {
                                    var ty = getExpressionTypeInfo(expr);
                                    if (!ty.isResolved()) {
                                        ty = CommonTypes.bool_type();
                                    }
                                    expr.* = .{ .Literal = .{ .Bool = .{
                                        .value = elem_val.Bool,
                                        .type_info = ty,
                                        .span = span,
                                    } } };
                                    return;
                                },
                                else => {},
                            }
                        }
                    }
                }
            }
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
            } else {
                const target_val = ctx.core_resolver.evaluateConstantExpressionWithLookup(fa.target) catch const_eval.ConstantValue.NotConstant;
                if (target_val == .Array) {
                    const field_str = if (fa.field.len > 0 and fa.field[0] == '_') fa.field[1..] else fa.field;
                    const idx = std.fmt.parseInt(usize, field_str, 10) catch return;
                    if (idx < target_val.Array.len) {
                        const elem_val = target_val.Array[idx];
                        const span = getExpressionSpan(expr);
                        switch (elem_val) {
                            .Integer => {
                                const value_str = try std.fmt.allocPrint(ctx.type_storage_allocator, "{}", .{elem_val.Integer});
                                var ty = getExpressionTypeInfo(expr);
                                if (!ty.isResolved()) {
                                    ty = CommonTypes.unknown_integer();
                                }
                                expr.* = .{ .Literal = .{ .Integer = .{
                                    .value = value_str,
                                    .type_info = ty,
                                    .span = span,
                                } } };
                            },
                            .Bool => {
                                var ty = getExpressionTypeInfo(expr);
                                if (!ty.isResolved()) {
                                    ty = CommonTypes.bool_type();
                                }
                                expr.* = .{ .Literal = .{ .Bool = .{
                                    .value = elem_val.Bool,
                                    .type_info = ty,
                                    .span = span,
                                } } };
                            },
                            else => {},
                        }
                    }
                }
            }
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

    const result = ctx.core_resolver.evaluateConstantExpressionWithLookup(expr) catch return;
    const span = getExpressionSpan(expr);
    switch (result) {
        .Integer => {
            const value_str = try std.fmt.allocPrint(ctx.type_storage_allocator, "{}", .{result.Integer});
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
            expr.* = .{ .Literal = .{ .Integer = .{
                .value = value_str,
                .type_info = ty,
                .span = span,
            } } };
        },
        .Bool => {
            var ty = getExpressionTypeInfo(expr);
            if (!ty.isResolved()) {
                ty = CommonTypes.bool_type();
            }
            expr.* = .{ .Literal = .{ .Bool = .{
                .value = result.Bool,
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

/// Must match locals_binder.getBlockScopeKey for consistency.
fn getBlockScopeKey(stmt: *const ast.Statements.StmtNode, block_id: usize) usize {
    return @intFromPtr(stmt) * 4 + block_id;
}
