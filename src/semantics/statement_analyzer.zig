// ============================================================================
// Statement Analyzer - Phase 2 Diagnostics (Lightweight)
// ============================================================================
//
// Provides lightweight diagnostics that are not part of the core type
// correctness rules. Type validation and return typing are handled by the
// type resolver.
//
// SECTIONS:
//   • Unknown identifier checking (optional)
//
// NOTE: Unknown identifier checking is enabled in semantics/core.zig.
//
// ============================================================================

const std = @import("std");
const ast = @import("../ast.zig");
const state = @import("state.zig");
const builtins = @import("../semantics.zig").builtins;
const ManagedArrayList = std.array_list.Managed;

pub fn collectUnknownIdentifierSpans(
    allocator: std.mem.Allocator,
    table: *state.SymbolTable,
    scope: *state.Scope,
    f: *const ast.FunctionNode,
) ![]const ast.SourceSpan {
    var issues = ManagedArrayList(ast.SourceSpan).init(allocator);
    try walkBlockForUnknowns(&issues, table, scope, &f.body);
    return try issues.toOwnedSlice();
}

fn resolveBlockScope(table: *state.SymbolTable, default_scope: *state.Scope, block: *const ast.Statements.BlockNode) *state.Scope {
    const key: usize = @intFromPtr(block);
    if (table.block_scopes.get(key)) |sc| return sc;
    return default_scope;
}

fn walkBlockForUnknowns(issues: *ManagedArrayList(ast.SourceSpan), table: *state.SymbolTable, scope: *state.Scope, block: *const ast.Statements.BlockNode) !void {
    for (block.statements) |stmt| {
        switch (stmt) {
            .Expr => |e| try visitExprForUnknowns(issues, table, scope, e),
            .VariableDecl => |v| {
                if (v.value) |val| try visitExprForUnknowns(issues, table, scope, val.*);
            },
            .DestructuringAssignment => |da| try visitExprForUnknowns(issues, table, scope, da.value.*),
            .Return => |r| if (r.value) |v| try visitExprForUnknowns(issues, table, scope, v),
            .CompoundAssignment => |ca| {
                try visitExprForUnknowns(issues, table, scope, ca.target.*);
                try visitExprForUnknowns(issues, table, scope, ca.value.*);
            },
            .If => |iff| {
                try visitExprForUnknowns(issues, table, scope, iff.condition);
                const then_scope = resolveBlockScope(table, scope, &iff.then_branch);
                try walkBlockForUnknowns(issues, table, then_scope, &iff.then_branch);
                if (iff.else_branch) |*eb| {
                    const else_scope = resolveBlockScope(table, scope, eb);
                    try walkBlockForUnknowns(issues, table, else_scope, eb);
                }
            },
            .While => |wh| {
                try visitExprForUnknowns(issues, table, scope, wh.condition);
                const body_scope = resolveBlockScope(table, scope, &wh.body);
                try walkBlockForUnknowns(issues, table, body_scope, &wh.body);
            },
            .ForLoop => |fl| {
                try visitExprForUnknowns(issues, table, scope, fl.iterable);
                const body_scope = resolveBlockScope(table, scope, &fl.body);
                try walkBlockForUnknowns(issues, table, body_scope, &fl.body);
            },
            .TryBlock => |tb| {
                const try_scope = resolveBlockScope(table, scope, &tb.try_block);
                try walkBlockForUnknowns(issues, table, try_scope, &tb.try_block);
                if (tb.catch_block) |cb| {
                    const catch_scope = resolveBlockScope(table, scope, &cb.block);
                    try walkBlockForUnknowns(issues, table, catch_scope, &cb.block);
                }
            },
            .Log => |log_stmt| {
                for (log_stmt.args) |arg| try visitExprForUnknowns(issues, table, scope, arg);
            },
            .Lock => |lock| try visitExprForUnknowns(issues, table, scope, lock.path),
            .Unlock => |unlock| try visitExprForUnknowns(issues, table, scope, unlock.path),
            .Assert => |assert_stmt| try visitExprForUnknowns(issues, table, scope, assert_stmt.condition),
            .Invariant => |inv| try visitExprForUnknowns(issues, table, scope, inv.condition),
            .Requires => |req| try visitExprForUnknowns(issues, table, scope, req.condition),
            .Ensures => |ens| try visitExprForUnknowns(issues, table, scope, ens.condition),
            .Assume => |assume| try visitExprForUnknowns(issues, table, scope, assume.condition),
            .Havoc => |_| {},
            .Break => |br| if (br.value) |val| try visitExprForUnknowns(issues, table, scope, val.*),
            .Continue => |cont| if (cont.value) |val| try visitExprForUnknowns(issues, table, scope, val.*),
            .Switch => |sw| {
                try visitExprForUnknowns(issues, table, scope, sw.condition);
                for (sw.cases) |*case| {
                    switch (case.pattern) {
                        .Range => |r| {
                            try visitExprForUnknowns(issues, table, scope, r.start.*);
                            try visitExprForUnknowns(issues, table, scope, r.end.*);
                        },
                        else => {},
                    }
                    switch (case.body) {
                        .Expression => |expr_ptr| try visitExprForUnknowns(issues, table, scope, expr_ptr.*),
                        .Block => |*blk| {
                            const case_scope = resolveBlockScope(table, scope, blk);
                            try walkBlockForUnknowns(issues, table, case_scope, blk);
                        },
                        .LabeledBlock => |*lb| {
                            const case_scope = resolveBlockScope(table, scope, &lb.block);
                            try walkBlockForUnknowns(issues, table, case_scope, &lb.block);
                        },
                    }
                }
                if (sw.default_case) |*defb| {
                    const def_scope = resolveBlockScope(table, scope, defb);
                    try walkBlockForUnknowns(issues, table, def_scope, defb);
                }
            },
            .LabeledBlock => |lb| {
                const inner_scope = resolveBlockScope(table, scope, &lb.block);
                try walkBlockForUnknowns(issues, table, inner_scope, &lb.block);
            },
            else => {},
        }
    }
}

fn visitExprForUnknowns(issues: *ManagedArrayList(ast.SourceSpan), table: *state.SymbolTable, scope: *state.Scope, expr_node: ast.Expressions.ExprNode) !void {
    if (!table.isScopeKnown(scope)) return;
    switch (expr_node) {
        .Identifier => |id| {
            if (table.isScopeKnown(scope) and table.safeFindUpOpt(scope, id.name) != null) {
                // ok
            } else {
                try issues.append(id.span);
            }
        },
        .Binary => |b| {
            try visitExprForUnknowns(issues, table, scope, b.lhs.*);
            try visitExprForUnknowns(issues, table, scope, b.rhs.*);
        },
        .Unary => |u| try visitExprForUnknowns(issues, table, scope, u.operand.*),
        .Assignment => |a| {
            try visitExprForUnknowns(issues, table, scope, a.target.*);
            try visitExprForUnknowns(issues, table, scope, a.value.*);
        },
        .CompoundAssignment => |ca| {
            try visitExprForUnknowns(issues, table, scope, ca.target.*);
            try visitExprForUnknowns(issues, table, scope, ca.value.*);
        },
        .Call => |c| {
            if (builtins.isMemberAccessChain(c.callee)) {
                const path = builtins.getMemberAccessPath(table.allocator, c.callee) catch "";
                defer if (path.len > 0) table.allocator.free(path);
                if (path.len > 0 and table.builtin_registry.lookup(path) != null) {
                    for (c.arguments) |arg| try visitExprForUnknowns(issues, table, scope, arg.*);
                    return;
                }
            }
            try visitExprForUnknowns(issues, table, scope, c.callee.*);
            for (c.arguments) |arg| try visitExprForUnknowns(issues, table, scope, arg.*);
        },
        .Index => |ix| {
            try visitExprForUnknowns(issues, table, scope, ix.target.*);
            try visitExprForUnknowns(issues, table, scope, ix.index.*);
        },
        .FieldAccess => |fa| {
            if (builtins.isMemberAccessChain(&expr_node)) {
                const path = builtins.getMemberAccessPath(table.allocator, &expr_node) catch "";
                defer if (path.len > 0) table.allocator.free(path);
                if (path.len > 0 and table.builtin_registry.lookup(path) != null) return;
            }
            try visitExprForUnknowns(issues, table, scope, fa.target.*);
        },
        .Cast => |c| try visitExprForUnknowns(issues, table, scope, c.operand.*),
        .Comptime => |_| {},
        .Tuple => |t| for (t.elements) |el| try visitExprForUnknowns(issues, table, scope, el.*),
        .SwitchExpression => {
            if (expr_node == .SwitchExpression) {
                try visitExprForUnknowns(issues, table, scope, expr_node.SwitchExpression.condition.*);
            }
        },
        .AnonymousStruct => |as| for (as.fields) |f| try visitExprForUnknowns(issues, table, scope, f.value.*),
        .ArrayLiteral => |al| for (al.elements) |el| try visitExprForUnknowns(issues, table, scope, el.*),
        .Range => |r| {
            try visitExprForUnknowns(issues, table, scope, r.start.*);
            try visitExprForUnknowns(issues, table, scope, r.end.*);
        },
        .StructInstantiation => |si| {
            try visitExprForUnknowns(issues, table, scope, si.struct_name.*);
            for (si.fields) |f| try visitExprForUnknowns(issues, table, scope, f.value.*);
        },
        .Destructuring => |d| try visitExprForUnknowns(issues, table, scope, d.value.*),
        .Quantified => |q| {
            if (q.condition) |cond| try visitExprForUnknowns(issues, table, scope, cond.*);
            try visitExprForUnknowns(issues, table, scope, q.body.*);
        },
        .Try => |t| try visitExprForUnknowns(issues, table, scope, t.expr.*),
        .ErrorCast => |ec| try visitExprForUnknowns(issues, table, scope, ec.operand.*),
        .Shift => |sh| {
            try visitExprForUnknowns(issues, table, scope, sh.mapping.*);
            try visitExprForUnknowns(issues, table, scope, sh.source.*);
            try visitExprForUnknowns(issues, table, scope, sh.dest.*);
            try visitExprForUnknowns(issues, table, scope, sh.amount.*);
        },
        else => {},
    }
}
