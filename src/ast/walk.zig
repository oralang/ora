//! Generic walkers over AST expressions.
//!
//! Today: a recursive `collectNamesInExpr` that flattens every
//! `NameExpr.name` referenced under an expression subtree. Used by
//! the debug-info emitter to compute statement-level liveness for
//! local bindings (B3 in the debugger plan): a binding's last-use
//! position is the largest statement index whose expression tree
//! mentions the binding's name.
//!
//! Naming-only — does not resolve scopes or distinguish a local
//! reference from a type name with the same spelling. The caller
//! filters against the binding names actually in scope.

const std = @import("std");
const file_mod = @import("file.zig");
const nodes = @import("nodes.zig");
const ids = @import("ids.zig");

const AstFile = file_mod.AstFile;
const ExprId = ids.ExprId;

/// Append every `Name` reference reachable from `expr_id` to `out`.
/// Recursive; bounded by the AST depth, which is bounded by source
/// nesting depth (no cycles in a tree). Names are borrowed from the
/// AST — caller must not free them.
///
/// Stops at expression-body boundaries: `ComptimeExpr.body` and
/// `QuantifiedExpr.body` are NOT recursed into. Both are
/// statement-bearing structures; a future caller that wants to walk
/// them needs to also walk their statements (see `walkStmt`-shaped
/// helpers, not yet implemented).
pub fn collectNamesInExpr(
    allocator: std.mem.Allocator,
    ast_file: *const AstFile,
    expr_id: ExprId,
    out: *std.ArrayList([]const u8),
) !void {
    const expr = ast_file.expression(expr_id).*;
    switch (expr) {
        // Leaf literals — nothing to walk.
        .IntegerLiteral, .StringLiteral, .BoolLiteral, .AddressLiteral, .BytesLiteral, .TypeValue, .Result, .Error => {},

        .Name => |name| try out.append(allocator, name.name),

        .Tuple => |tuple| {
            for (tuple.elements) |sub| try collectNamesInExpr(allocator, ast_file, sub, out);
        },
        .ArrayLiteral => |array| {
            for (array.elements) |sub| try collectNamesInExpr(allocator, ast_file, sub, out);
        },
        .StructLiteral => |sl| {
            for (sl.fields) |field| try collectNamesInExpr(allocator, ast_file, field.value, out);
        },
        .Switch => |sw| {
            try collectNamesInExpr(allocator, ast_file, sw.condition, out);
            for (sw.arms) |arm| try collectNamesInExpr(allocator, ast_file, arm.value, out);
            if (sw.else_expr) |e| try collectNamesInExpr(allocator, ast_file, e, out);
        },
        .ExternalProxy => |ep| {
            try collectNamesInExpr(allocator, ast_file, ep.address_expr, out);
            try collectNamesInExpr(allocator, ast_file, ep.gas_expr, out);
        },
        .Comptime => {
            // body is a BodyId, not an ExprId — skip. A future
            // statement-walker can pick this up.
        },
        .ErrorReturn => |er| {
            for (er.args) |sub| try collectNamesInExpr(allocator, ast_file, sub, out);
        },
        .Unary => |u| try collectNamesInExpr(allocator, ast_file, u.operand, out),
        .Binary => |b| {
            try collectNamesInExpr(allocator, ast_file, b.lhs, out);
            try collectNamesInExpr(allocator, ast_file, b.rhs, out);
        },
        .Call => |call| {
            try collectNamesInExpr(allocator, ast_file, call.callee, out);
            for (call.args) |sub| try collectNamesInExpr(allocator, ast_file, sub, out);
        },
        .Builtin => |b| {
            // type_arg is a TypeExprId — not walked (no value uses).
            for (b.args) |sub| try collectNamesInExpr(allocator, ast_file, sub, out);
        },
        .Field => |f| try collectNamesInExpr(allocator, ast_file, f.base, out),
        .Index => |idx| {
            try collectNamesInExpr(allocator, ast_file, idx.base, out);
            try collectNamesInExpr(allocator, ast_file, idx.index, out);
        },
        .Group => |g| try collectNamesInExpr(allocator, ast_file, g.expr, out),
        .Old => |o| try collectNamesInExpr(allocator, ast_file, o.expr, out),
        .Quantified => |_| {
            // body is an ExprId, but the pattern introduces a
            // fresh binding that shadows outer names. Walking the
            // body without scope-awareness would conflate the
            // bound variable with outer references. Skip until a
            // scope-aware walker exists.
        },
    }
}

// =============================================================================
// Tests
// =============================================================================

const testing = std.testing;

test "collectNamesInExpr type check" {
    // No-op type-level assertion: ensures the file compiles and
    // the function is referenced from the test target. AST-driven
    // coverage lives in compiler tests that already build full
    // ASTs; reproducing that machinery here would be wasteful.
    const fn_ptr: *const fn (
        std.mem.Allocator,
        *const AstFile,
        ExprId,
        *std.ArrayList([]const u8),
    ) anyerror!void = collectNamesInExpr;
    _ = fn_ptr;
    try testing.expect(true);
}
