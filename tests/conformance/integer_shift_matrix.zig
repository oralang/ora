const std = @import("std");
const testing = std.testing;

const ora_types = @import("ora_types");
const runner = @import("runner.zig");
const types = @import("types.zig");

const source = @embedFile("integer_shift_matrix.ora");
const shift_revert_data = [_]u8{ 0xb3, 0x21, 0x5f, 0x31 };

const Mode = enum(u8) {
    checked_left = 0,
    checked_right = 1,
    wrapping_left = 2,
    wrapping_right = 3,
};

fn modeText(mode: Mode) []const u8 {
    return switch (mode) {
        .checked_left => "0",
        .checked_right => "1",
        .wrapping_left => "2",
        .wrapping_right => "3",
    };
}

fn signature(
    allocator: std.mem.Allocator,
    spec: ora_types.BuiltinTypeSpec,
) ![]const u8 {
    const width = spec.bit_width orelse return error.IntegerShiftCorpusMissingWidth;
    const signed = spec.signed orelse return error.IntegerShiftCorpusMissingSignedness;
    const abi_prefix = if (signed) "int" else "uint";
    return std.fmt.allocPrint(
        allocator,
        "shift_{s}({s}{d},{s}{d},uint8)",
        .{ spec.source_name, abi_prefix, width, abi_prefix, width },
    );
}

fn call(
    runtime: *runner.PropertyRuntime,
    spec: ora_types.BuiltinTypeSpec,
    mode: Mode,
    value: []const u8,
    amount: []const u8,
) !types.Evm.CallResult {
    const function_signature = try signature(runtime.allocator, spec);
    const args = [_]types.ArgValue{
        .{ .literal = value },
        .{ .literal = amount },
        .{ .literal = modeText(mode) },
    };
    return runtime.call(function_signature, &args);
}

fn expectWord(result: types.Evm.CallResult, expected: u256) !void {
    try testing.expect(result.success);
    try testing.expectEqual(@as(usize, 32), result.output.len);
    try testing.expectEqual(expected, std.mem.readInt(u256, result.output[0..32], .big));
}

fn expectRevert(result: types.Evm.CallResult) !void {
    try testing.expect(!result.success);
    try testing.expectEqualSlices(u8, &shift_revert_data, result.output);
}

fn checkIntegerType(
    runtime: *runner.PropertyRuntime,
    spec: ora_types.BuiltinTypeSpec,
) !void {
    const width = spec.bit_width orelse return error.IntegerShiftCorpusMissingWidth;
    const signed = spec.signed orelse return error.IntegerShiftCorpusMissingSignedness;
    const edge_shift: u8 = @intCast(width - 1);
    const top_bit = @as(u256, 1) << edge_shift;
    const signed_min_word = @as(u256, 0) -% top_bit;

    const width_text = try std.fmt.allocPrint(runtime.allocator, "{d}", .{width});
    const above_width_text = try std.fmt.allocPrint(runtime.allocator, "{d}", .{width + 1});
    const edge_text = try std.fmt.allocPrint(runtime.allocator, "{d}", .{width - 1});
    const top_bit_text = try std.fmt.allocPrint(runtime.allocator, "{d}", .{top_bit});
    const signed_min_text = try std.fmt.allocPrint(runtime.allocator, "-{d}", .{top_bit});

    const left_edge_word = if (signed) signed_min_word else top_bit;
    const right_value = if (signed) signed_min_text else top_bit_text;
    const right_value_word = if (signed) signed_min_word else top_bit;
    const right_edge_word: u256 = if (signed) std.math.maxInt(u256) else 1;
    const right_out_of_range_word: u256 = if (signed) std.math.maxInt(u256) else 0;

    try expectWord(try call(runtime, spec, .checked_left, "1", "0"), 1);
    try expectWord(try call(runtime, spec, .checked_left, "1", edge_text), left_edge_word);
    try expectRevert(try call(runtime, spec, .checked_left, "1", width_text));
    try expectRevert(try call(runtime, spec, .checked_left, "1", above_width_text));

    try expectWord(try call(runtime, spec, .checked_right, right_value, "0"), right_value_word);
    try expectWord(try call(runtime, spec, .checked_right, right_value, edge_text), right_edge_word);
    try expectRevert(try call(runtime, spec, .checked_right, right_value, width_text));
    try expectRevert(try call(runtime, spec, .checked_right, right_value, above_width_text));

    try expectWord(try call(runtime, spec, .wrapping_left, "1", "0"), 1);
    try expectWord(try call(runtime, spec, .wrapping_left, "1", edge_text), left_edge_word);
    try expectWord(try call(runtime, spec, .wrapping_left, "1", width_text), 0);
    try expectWord(try call(runtime, spec, .wrapping_left, "1", above_width_text), 0);

    try expectWord(try call(runtime, spec, .wrapping_right, right_value, "0"), right_value_word);
    try expectWord(try call(runtime, spec, .wrapping_right, right_value, edge_text), right_edge_word);
    try expectWord(try call(runtime, spec, .wrapping_right, right_value, width_text), right_out_of_range_word);
    try expectWord(try call(runtime, spec, .wrapping_right, right_value, above_width_text), right_out_of_range_word);

    if (signed) {
        try expectRevert(try call(runtime, spec, .checked_left, "1", "-1"));
        try expectRevert(try call(runtime, spec, .checked_right, right_value, "-1"));
        try expectWord(try call(runtime, spec, .wrapping_left, "1", "-1"), 0);
        try expectWord(
            try call(runtime, spec, .wrapping_right, right_value, "-1"),
            std.math.maxInt(u256),
        );
    }
}

fn checkMatrix(runtime: *runner.PropertyRuntime) !void {
    var integer_type_count: usize = 0;
    for (ora_types.builtin.builtin_types) |spec| {
        if (spec.category != .Integer) continue;
        integer_type_count += 1;
        try checkIntegerType(runtime, spec);
    }
    try testing.expectEqual(@as(usize, 13), integer_type_count);
}

pub fn run(allocator: std.mem.Allocator) !void {
    try runner.compileAndRunPropertySource(allocator, "integer_shift_matrix", source, checkMatrix);
}
