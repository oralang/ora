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

pub fn bindFunctionLocals(table: *state.SymbolTable, fn_scope: *state.Scope, f: *const ast.FunctionNode) !void {
    // Associate the function body block with the function scope
    try mapBlockScope(table, &f.body, fn_scope);
    // Walk the body and create nested scopes for blocks; declare locals into their owning scope
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
    // Declare direct local variable declarations in this block
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
                // Bind the loop variables in the body scope
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
                        // Create Error type for catch block error variable
                        // TODO: Properly infer error type from the try expression's error union
                        const error_type_info = @import("../ast/type_info.zig").TypeInfo{
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
                // Each case with a block gets its own scope
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
