//! Shared prelude for the split encoder.test.* files: common imports and
//! test helper functions, all `pub` so each category file can alias them.
// ============================================================================
// Z3 Encoder Tests
// ============================================================================
//
// unit tests for MLIR-to-Z3 encoding behavior.
//
// ============================================================================

pub const std = @import("std");
pub const testing = std.testing;
pub const z3 = @import("c.zig");
pub const mlir = @import("mlir_c_api").c;
pub const Context = @import("context.zig").Context;
pub const Encoder = @import("encoder.zig").Encoder;
pub const Solver = @import("solver.zig").Solver;

pub fn stringRef(comptime s: []const u8) mlir.MlirStringRef {
    return mlir.oraStringRefCreate(s.ptr, s.len);
}

pub fn namedAttr(ctx: mlir.MlirContext, comptime name: []const u8, attr: mlir.MlirAttribute) mlir.MlirNamedAttribute {
    const id = mlir.oraIdentifierGet(ctx, mlir.oraStringRefCreate(name.ptr, name.len));
    return mlir.oraNamedAttributeGet(id, attr);
}

pub fn loadAllDialects(ctx: mlir.MlirContext) void {
    const registry = mlir.oraDialectRegistryCreate();
    defer mlir.oraDialectRegistryDestroy(registry);
    mlir.oraRegisterAllDialects(registry);
    mlir.oraContextAppendDialectRegistry(ctx, registry);
    mlir.oraContextLoadAllAvailableDialects(ctx);
}

pub fn expectAstEquivalent(ctx: *Context, lhs: z3.Z3_ast, rhs: z3.Z3_ast) !void {
    var solver = try Solver.init(ctx, testing.allocator);
    defer solver.deinit();
    solver.assert(z3.Z3_mk_not(ctx.ctx, z3.Z3_mk_eq(ctx.ctx, lhs, rhs)));
    try testing.expectEqual(@as(z3.Z3_lbool, z3.Z3_L_FALSE), solver.check());
}

pub fn expectSingleSelectTrigger(ctx: *Context, quantifier: z3.Z3_ast) !void {
    try testing.expectEqual(@as(c_uint, 1), z3.Z3_get_quantifier_num_patterns(ctx.ctx, quantifier));
    const pattern = z3.Z3_get_quantifier_pattern_ast(ctx.ctx, quantifier, 0);
    try testing.expectEqual(@as(c_uint, 1), z3.Z3_get_pattern_num_terms(ctx.ctx, pattern));
    const trigger = z3.Z3_get_pattern(ctx.ctx, pattern, 0);
    try testing.expectEqual(@as(c_uint, z3.Z3_APP_AST), z3.Z3_get_ast_kind(ctx.ctx, trigger));
    const app = z3.Z3_to_app(ctx.ctx, trigger);
    const decl = z3.Z3_get_app_decl(ctx.ctx, app);
    try testing.expectEqual(@as(c_uint, z3.Z3_OP_SELECT), @as(c_uint, @intCast(z3.Z3_get_decl_kind(ctx.ctx, decl))));
}

pub fn expectNoQuantifiedConstraints(ctx: *Context, constraints: []const z3.Z3_ast) !void {
    for (constraints) |constraint| {
        try testing.expect(!astContainsQuantifier(ctx, constraint));
    }
}

pub fn astContainsQuantifier(ctx: *Context, ast: z3.Z3_ast) bool {
    const kind = z3.Z3_get_ast_kind(ctx.ctx, ast);
    if (kind == z3.Z3_QUANTIFIER_AST) return true;
    if (kind != z3.Z3_APP_AST) return false;

    const app = z3.Z3_to_app(ctx.ctx, ast);
    const arg_count: usize = @intCast(z3.Z3_get_app_num_args(ctx.ctx, app));
    for (0..arg_count) |idx| {
        if (astContainsQuantifier(ctx, z3.Z3_get_app_arg(ctx.ctx, app, @intCast(idx)))) return true;
    }
    return false;
}
