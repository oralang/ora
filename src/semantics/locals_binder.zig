// ============================================================================
// Local Variable Binder
// ============================================================================
//
// Binds local variables and creates nested block scopes within functions.
// Part of Phase 1 (after function scopes are created).
//
// RESPONSIBILITIES:
//   • Create nested block scopes (if/while/for/try/switch)
//   • Bind local variable declarations
//   • Handle loop patterns (single, index-pair, destructured)
//   • Infer types for variables with initializers
//
// SCOPE HIERARCHY:
//   Root → Function → Blocks (if/while/for/try/switch)
//
// ============================================================================

const std = @import("std");
const ast = @import("../ast.zig");
const state = @import("state.zig");
const expr = @import("expression_analyzer.zig");

fn mapBlockScope(table: *state.SymbolTable, block: *const ast.Statements.BlockNode, scope: *state.Scope) !void {
    const key: usize = @intFromPtr(block);
    try table.block_scopes.put(key, scope);
}

fn createChildScope(table: *state.SymbolTable, parent: *state.Scope, name: ?[]const u8) !*state.Scope {
    const sc = try table.allocator.create(state.Scope);
    sc.* = state.Scope.init(table.allocator, parent, name);
    try table.scopes.append(table.allocator, sc);
    return sc;
}

fn isErrorUnionType(ti: ast.Types.TypeInfo) bool {
    if (ti.category == .ErrorUnion) return true;
    if (ti.ora_type) |ot| {
        return switch (ot) {
            .error_union => true,
            ._union => |members| blk: {
                if (members.len == 0) break :blk false;
                break :blk members[0] == .error_union;
            },
            else => false,
        };
    }
    return false;
}

fn findErrorUnionInExprPtr(table: *state.SymbolTable, scope: *state.Scope, expr_ptr: *ast.Expressions.ExprNode) ?ast.Types.TypeInfo {
    return findErrorUnionInExpr(table, scope, expr_ptr.*);
}

fn findErrorUnionInExpr(table: *state.SymbolTable, scope: *state.Scope, expr_node: ast.Expressions.ExprNode) ?ast.Types.TypeInfo {
    switch (expr_node) {
        .Try => |t| {
            const inner = expr.inferExprType(table, scope, t.expr.*);
            if (isErrorUnionType(inner)) return inner;
            return findErrorUnionInExprPtr(table, scope, t.expr);
        },
        .Call => |c| {
            const ti = expr.inferExprType(table, scope, expr_node);
            if (isErrorUnionType(ti)) return ti;
            if (findErrorUnionInExprPtr(table, scope, c.callee)) |found| return found;
            for (c.arguments) |arg| {
                if (findErrorUnionInExprPtr(table, scope, arg)) |found| return found;
            }
            return null;
        },
        .Binary => |b| {
            if (findErrorUnionInExprPtr(table, scope, b.lhs)) |found| return found;
            return findErrorUnionInExprPtr(table, scope, b.rhs);
        },
        .Unary => |u| return findErrorUnionInExprPtr(table, scope, u.operand),
        .Assignment => |a| {
            if (findErrorUnionInExprPtr(table, scope, a.target)) |found| return found;
            return findErrorUnionInExprPtr(table, scope, a.value);
        },
        .CompoundAssignment => |ca| {
            if (findErrorUnionInExprPtr(table, scope, ca.target)) |found| return found;
            return findErrorUnionInExprPtr(table, scope, ca.value);
        },
        .Index => |ix| {
            if (findErrorUnionInExprPtr(table, scope, ix.target)) |found| return found;
            return findErrorUnionInExprPtr(table, scope, ix.index);
        },
        .FieldAccess => |fa| return findErrorUnionInExprPtr(table, scope, fa.target),
        .Cast => |c| return findErrorUnionInExprPtr(table, scope, c.operand),
        .Tuple => |t| {
            for (t.elements) |el| {
                if (findErrorUnionInExprPtr(table, scope, el)) |found| return found;
            }
            return null;
        },
        .SwitchExpression => |sw| {
            if (findErrorUnionInExprPtr(table, scope, sw.condition)) |found| return found;
            for (sw.cases) |case| {
                switch (case.body) {
                    .Expression => |expr_ptr| {
                        if (findErrorUnionInExprPtr(table, scope, expr_ptr)) |found| return found;
                    },
                    .Block => |*blk| if (findErrorUnionInBlock(table, scope, blk)) |found| return found,
                    .LabeledBlock => |lb| if (findErrorUnionInBlock(table, scope, &lb.block)) |found| return found,
                }
            }
            if (sw.default_case) |*defb| return findErrorUnionInBlock(table, scope, defb);
            return null;
        },
        .AnonymousStruct => |as| {
            for (as.fields) |f| {
                if (findErrorUnionInExprPtr(table, scope, f.value)) |found| return found;
            }
            return null;
        },
        .ArrayLiteral => |al| {
            for (al.elements) |el| {
                if (findErrorUnionInExprPtr(table, scope, el)) |found| return found;
            }
            return null;
        },
        .StructInstantiation => |si| {
            if (findErrorUnionInExprPtr(table, scope, si.struct_name)) |found| return found;
            for (si.fields) |f| {
                if (findErrorUnionInExprPtr(table, scope, f.value)) |found| return found;
            }
            return null;
        },
        .Destructuring => |d| return findErrorUnionInExprPtr(table, scope, d.value),
        .Range => |r| {
            if (findErrorUnionInExprPtr(table, scope, r.start)) |found| return found;
            return findErrorUnionInExprPtr(table, scope, r.end);
        },
        .Shift => |sh| {
            if (findErrorUnionInExprPtr(table, scope, sh.mapping)) |found| return found;
            if (findErrorUnionInExprPtr(table, scope, sh.source)) |found| return found;
            if (findErrorUnionInExprPtr(table, scope, sh.dest)) |found| return found;
            return findErrorUnionInExprPtr(table, scope, sh.amount);
        },
        .Quantified => |q| {
            if (q.condition) |cond| {
                if (findErrorUnionInExprPtr(table, scope, cond)) |found| return found;
            }
            return findErrorUnionInExprPtr(table, scope, q.body);
        },
        else => return null,
    }
}

fn findErrorUnionInBlock(table: *state.SymbolTable, scope: *state.Scope, block: *const ast.Statements.BlockNode) ?ast.Types.TypeInfo {
    for (block.statements) |stmt| {
        switch (stmt) {
            .VariableDecl => |v| {
                if (v.value) |val| {
                    if (findErrorUnionInExprPtr(table, scope, val)) |found| return found;
                }
            },
            .Expr => |e| if (findErrorUnionInExpr(table, scope, e)) |found| return found,
            .Return => |r| if (r.value) |v| {
                if (findErrorUnionInExpr(table, scope, v)) |found| return found;
            },
            .CompoundAssignment => |ca| {
                if (findErrorUnionInExprPtr(table, scope, ca.target)) |found| return found;
                if (findErrorUnionInExprPtr(table, scope, ca.value)) |found| return found;
            },
            .If => |iff| {
                if (findErrorUnionInExpr(table, scope, iff.condition)) |found| return found;
                if (findErrorUnionInBlock(table, scope, &iff.then_branch)) |found| return found;
                if (iff.else_branch) |*eb| if (findErrorUnionInBlock(table, scope, eb)) |found| return found;
            },
            .While => |wh| {
                if (findErrorUnionInExpr(table, scope, wh.condition)) |found| return found;
                if (findErrorUnionInBlock(table, scope, &wh.body)) |found| return found;
            },
            .ForLoop => |fl| {
                if (findErrorUnionInExpr(table, scope, fl.iterable)) |found| return found;
                if (findErrorUnionInBlock(table, scope, &fl.body)) |found| return found;
            },
            .TryBlock => |tb| {
                if (findErrorUnionInBlock(table, scope, &tb.try_block)) |found| return found;
                if (tb.catch_block) |cb| if (findErrorUnionInBlock(table, scope, &cb.block)) |found| return found;
            },
            .Switch => |sw| {
                if (findErrorUnionInExpr(table, scope, sw.condition)) |found| return found;
                for (sw.cases) |*case| {
                    switch (case.body) {
                        .Block => |*blk| if (findErrorUnionInBlock(table, scope, blk)) |found| return found,
                        .Expression => |expr_ptr| if (findErrorUnionInExprPtr(table, scope, expr_ptr)) |found| return found,
                        .LabeledBlock => |*lb| if (findErrorUnionInBlock(table, scope, &lb.block)) |found| return found,
                    }
                }
                if (sw.default_case) |*defb| if (findErrorUnionInBlock(table, scope, defb)) |found| return found;
            },
            else => {},
        }
    }

    return null;
}

pub fn bindFunctionLocals(table: *state.SymbolTable, fn_scope: *state.Scope, f: *const ast.FunctionNode) !void {
    // associate the function body block with the function scope
    try mapBlockScope(table, &f.body, fn_scope);
    // walk the body and create nested scopes for blocks; declare locals into their owning scope
    try bindBlock(table, fn_scope, &f.body);
}

fn declareVar(table: *state.SymbolTable, scope: *state.Scope, v: ast.statements.VariableDeclNode) !void {
    var resolved_type = v.type_info;
    if (resolved_type.ora_type == null) {
        if (v.value) |val_ptr| {
            const inferred = expr.inferExprType(table, scope, val_ptr.*);
            if (inferred.ora_type != null) {
                resolved_type = inferred;
            }
        }
    }
    const sym = state.Symbol{
        .name = v.name,
        .kind = .Var,
        .typ = resolved_type,
        .span = v.span,
        .mutable = (v.kind == .Var),
        .region = v.region,
    };
    _ = try table.declare(scope, sym);
}

fn bindForPattern(table: *state.SymbolTable, scope: *state.Scope, pattern: ast.statements.LoopPattern) !void {
    switch (pattern) {
        .Single => |s| {
            const sym = state.Symbol{ .name = s.name, .kind = .Var, .typ = null, .span = s.span, .mutable = false };
            _ = try table.declare(scope, sym);
        },
        .IndexPair => |p| {
            const sym_item = state.Symbol{ .name = p.item, .kind = .Var, .typ = null, .span = p.span, .mutable = false };
            const sym_idx = state.Symbol{ .name = p.index, .kind = .Var, .typ = null, .span = p.span, .mutable = false };
            _ = try table.declare(scope, sym_item);
            _ = try table.declare(scope, sym_idx);
        },
        .Destructured => |d| {
            switch (d.pattern) {
                .Struct => |fields| {
                    for (fields) |fld| {
                        const sym = state.Symbol{ .name = fld.variable, .kind = .Var, .typ = null, .span = d.span, .mutable = false };
                        _ = try table.declare(scope, sym);
                    }
                },
                .Tuple => |names| {
                    for (names) |nm| {
                        const sym = state.Symbol{ .name = nm, .kind = .Var, .typ = null, .span = d.span, .mutable = false };
                        _ = try table.declare(scope, sym);
                    }
                },
                .Array => |names| {
                    for (names) |nm| {
                        const sym = state.Symbol{ .name = nm, .kind = .Var, .typ = null, .span = d.span, .mutable = false };
                        _ = try table.declare(scope, sym);
                    }
                },
            }
        },
    }
}

fn bindBlock(table: *state.SymbolTable, scope: *state.Scope, block: *const ast.Statements.BlockNode) !void {
    // declare direct local variable declarations in this block
    for (block.statements) |stmt| {
        switch (stmt) {
            .VariableDecl => |v| try declareVar(table, scope, v),
            .DestructuringAssignment => |_| {},
            .Return => |_| {},
            .Expr => |_| {},
            .CompoundAssignment => |_| {},
            .If => |iff| {
                // then branch
                const then_scope = try createChildScope(table, scope, scope.name);
                try mapBlockScope(table, &iff.then_branch, then_scope);
                try bindBlock(table, then_scope, &iff.then_branch);
                // else branch
                if (iff.else_branch) |*eb| {
                    const else_scope = try createChildScope(table, scope, scope.name);
                    try mapBlockScope(table, eb, else_scope);
                    try bindBlock(table, else_scope, eb);
                }
            },
            .While => |wh| {
                const body_scope = try createChildScope(table, scope, scope.name);
                try mapBlockScope(table, &wh.body, body_scope);
                try bindBlock(table, body_scope, &wh.body);
            },
            .ForLoop => |fl| {
                const body_scope = try createChildScope(table, scope, scope.name);
                try mapBlockScope(table, &fl.body, body_scope);
                // bind the loop variables in the body scope
                try bindForPattern(table, body_scope, fl.pattern);
                try bindBlock(table, body_scope, &fl.body);
            },
            .TryBlock => |tb| {
                const try_scope = try createChildScope(table, scope, scope.name);
                try mapBlockScope(table, &tb.try_block, try_scope);
                try bindBlock(table, try_scope, &tb.try_block);
                if (tb.catch_block) |cb| {
                    const catch_scope = try createChildScope(table, scope, scope.name);
                    try mapBlockScope(table, &cb.block, catch_scope);
                    if (cb.error_variable) |ename| {
                        const inferred_error_union = findErrorUnionInBlock(table, try_scope, &tb.try_block);
                        const error_type_info = inferred_error_union orelse @import("../ast/type_info.zig").TypeInfo{
                            .category = .Error,
                            .ora_type = null,
                            .source = .inferred,
                            .span = cb.span,
                        };
                        const sym = state.Symbol{ .name = ename, .kind = .Var, .typ = error_type_info, .span = cb.span, .mutable = false };
                        _ = try table.declare(catch_scope, sym);
                    }
                    try bindBlock(table, catch_scope, &cb.block);
                }
            },
            .LabeledBlock => |lb| {
                const inner_scope = try createChildScope(table, scope, scope.name);
                try mapBlockScope(table, &lb.block, inner_scope);
                try bindBlock(table, inner_scope, &lb.block);
            },
            .Switch => |sw| {
                // each case with a block gets its own scope
                for (sw.cases) |*case| {
                    switch (case.body) {
                        .Block => |*blk| {
                            const cs = try createChildScope(table, scope, scope.name);
                            try mapBlockScope(table, blk, cs);
                            try bindBlock(table, cs, blk);
                        },
                        else => {},
                    }
                }
                if (sw.default_case) |*defb| {
                    const ds = try createChildScope(table, scope, scope.name);
                    try mapBlockScope(table, defb, ds);
                    try bindBlock(table, ds, defb);
                }
            },
            else => {},
        }
    }
}
