// ============================================================================
// Z3 Encoder Tests
// ============================================================================
//
// unit tests for MLIR-to-Z3 encoding behavior.
//
// ============================================================================

const std = @import("std");
const testing = std.testing;
const z3 = @import("c.zig");
const mlir = @import("mlir_c_api").c;
const Context = @import("context.zig").Context;
const Encoder = @import("encoder.zig").Encoder;

test "encodeMLIRType maps bool and i32" {
    var z3_ctx = try Context.init(testing.allocator);
    defer z3_ctx.deinit();

    var encoder = Encoder.init(&z3_ctx, testing.allocator);
    defer encoder.deinit();

    const mlir_ctx = mlir.oraContextCreate();
    defer mlir.oraContextDestroy(mlir_ctx);

    const ty_i1 = mlir.oraIntegerTypeCreate(mlir_ctx, 1);
    const bool_sort = try encoder.encodeMLIRType(ty_i1);
    try testing.expectEqual(@as(u32, z3.Z3_BOOL_SORT), @as(u32, @intCast(z3.Z3_get_sort_kind(z3_ctx.ctx, bool_sort))));

    const ty_i32 = mlir.oraIntegerTypeCreate(mlir_ctx, 32);
    const i32_sort = try encoder.encodeMLIRType(ty_i32);
    try testing.expectEqual(@as(u32, z3.Z3_BV_SORT), @as(u32, @intCast(z3.Z3_get_sort_kind(z3_ctx.ctx, i32_sort))));
    try testing.expectEqual(@as(u32, 32), @as(u32, @intCast(z3.Z3_get_bv_sort_size(z3_ctx.ctx, i32_sort))));
}

test "encodeIntegerConstant encodes large numerals" {
    var z3_ctx = try Context.init(testing.allocator);
    defer z3_ctx.deinit();

    var encoder = Encoder.init(&z3_ctx, testing.allocator);
    defer encoder.deinit();

    const value: u256 = (@as(u256, 1) << 128) + 123;
    const ast = try encoder.encodeIntegerConstant(value, 256);
    const numeral = z3.Z3_get_numeral_string(z3_ctx.ctx, ast);
    const expected = try std.fmt.allocPrint(testing.allocator, "{d}", .{value});
    defer testing.allocator.free(expected);

    try testing.expect(std.mem.eql(u8, std.mem.span(numeral), expected));
}

test "encodeConstantOp uses bool sort for i1" {
    var z3_ctx = try Context.init(testing.allocator);
    defer z3_ctx.deinit();

    var encoder = Encoder.init(&z3_ctx, testing.allocator);
    defer encoder.deinit();

    const ast = try encoder.encodeConstantOp(1, 1);
    const sort_kind = z3.Z3_get_sort_kind(z3_ctx.ctx, z3.Z3_get_sort(z3_ctx.ctx, ast));
    try testing.expectEqual(@as(u32, z3.Z3_BOOL_SORT), @as(u32, @intCast(sort_kind)));
}
