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
const type_info = @import("../ast/type_info.zig");
fn mapBlockScope(table: *state.SymbolTable, block: *const ast.Statements.BlockNode, scope: *state.Scope) !void {
    const key: usize = @intFromPtr(block);
    try table.block_scopes.put(key, scope);
}

/// Map a block scope using the statement pointer as the key base.
/// This avoids issues with stack copies when switching on union values.
/// block_id: 0 = primary block (then/body/try), 1 = secondary block (else/catch)
fn mapBlockScopeFromStmt(table: *state.SymbolTable, stmt: *const ast.Statements.StmtNode, block_id: usize, scope: *state.Scope) !void {
    const key: usize = @intFromPtr(stmt) * 4 + block_id;
    try table.block_scopes.put(key, scope);
}

fn getBlockScopeKey(stmt: *const ast.Statements.StmtNode, block_id: usize) usize {
    return @intFromPtr(stmt) * 4 + block_id;
}

fn createChildScope(table: *state.SymbolTable, parent: *state.Scope, name: ?[]const u8) !*state.Scope {
    const sc = try table.allocator.create(state.Scope);
    sc.* = state.Scope.init(table.allocator, parent, name);
    try table.scopes.append(table.allocator, sc);
    return sc;
}

pub fn bindFunctionLocals(table: *state.SymbolTable, fn_scope: *state.Scope, f: *const ast.FunctionNode) !void {
    // associate the function body block with the function scope
    try mapBlockScope(table, &f.body, fn_scope);
    // walk the body and create nested scopes for blocks; declare locals into their owning scope
    try bindBlock(table, fn_scope, &f.body);
}

fn declareVar(table: *state.SymbolTable, scope: *state.Scope, v: ast.statements.VariableDeclNode) !void {
    if (v.tuple_names) |names| {
        for (names) |name| {
            const sym = state.Symbol{
                .name = name,
                .kind = .Var,
                .typ = v.type_info,
                .span = v.span,
                .mutable = (v.kind == .Var),
                .region = v.region,
            };
            _ = try table.declare(scope, sym);
        }
        return;
    }
    const resolved_type = v.type_info;
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
    for (block.statements) |*stmt| {
        switch (stmt.*) {
            .VariableDecl => |v| try declareVar(table, scope, v),
            .DestructuringAssignment => |_| {},
            .Return => |_| {},
            .Expr => |_| {},
            .CompoundAssignment => |_| {},
            .If => |*iff| {
                // then branch - use stmt pointer for consistent key
                const then_scope = try createChildScope(table, scope, scope.name);
                try mapBlockScopeFromStmt(table, stmt, 0, then_scope);
                try bindBlock(table, then_scope, &iff.then_branch);
                // else branch
                if (iff.else_branch) |*eb| {
                    const else_scope = try createChildScope(table, scope, scope.name);
                    try mapBlockScopeFromStmt(table, stmt, 1, else_scope);
                    try bindBlock(table, else_scope, eb);
                }
            },
            .While => |*wh| {
                const body_scope = try createChildScope(table, scope, scope.name);
                try mapBlockScopeFromStmt(table, stmt, 0, body_scope);
                try bindBlock(table, body_scope, &wh.body);
            },
            .ForLoop => |*fl| {
                const body_scope = try createChildScope(table, scope, scope.name);
                try mapBlockScopeFromStmt(table, stmt, 0, body_scope);
                // bind the loop variables in the body scope
                try bindForPattern(table, body_scope, fl.pattern);
                try bindBlock(table, body_scope, &fl.body);
            },
            .TryBlock => |*tb| {
                const try_scope = try createChildScope(table, scope, scope.name);
                try mapBlockScopeFromStmt(table, stmt, 0, try_scope);
                try bindBlock(table, try_scope, &tb.try_block);
                if (tb.catch_block) |*cb| {
                    const catch_scope = try createChildScope(table, scope, scope.name);
                    try mapBlockScopeFromStmt(table, stmt, 1, catch_scope);
                    if (cb.error_variable) |ename| {
                        var error_type = type_info.CommonTypes.u256_type();
                        error_type.span = cb.span;
                        const sym = state.Symbol{ .name = ename, .kind = .Var, .typ = error_type, .span = cb.span, .mutable = false };
                        _ = try table.declare(catch_scope, sym);
                    }
                    try bindBlock(table, catch_scope, &cb.block);
                }
            },
            .LabeledBlock => |*lb| {
                const inner_scope = try createChildScope(table, scope, scope.name);
                try mapBlockScopeFromStmt(table, stmt, 0, inner_scope);
                try bindBlock(table, inner_scope, &lb.block);
            },
            .Switch => |*sw| {
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
