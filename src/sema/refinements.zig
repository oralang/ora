const std = @import("std");
const ast = @import("../ast/mod.zig");
const model = @import("model.zig");
const registry = @import("ora_refinements");

// Re-export the neutral registry, then add sema-specific helpers that work with
// model.RefinementType values.
pub const Kind = registry.Kind;
pub const Entry = registry.Entry;
pub const entries = registry.entries;
pub const entryForName = registry.entryForName;
pub const entryForKind = registry.entryForKind;
pub const kindForName = registry.kindForName;
pub const nameForKind = registry.nameForKind;
pub const isKnownName = registry.isKnownName;
pub const isPathFormName = registry.isPathFormName;
pub const hasNativeMlirTypeName = registry.hasNativeMlirTypeName;
pub const isBoundsBackedKind = registry.isBoundsBackedKind;

pub fn supportsRuntimeGuard(refinement: model.RefinementType) bool {
    return if (entryForName(refinement.name)) |entry| entry.has_runtime_guard else false;
}

pub fn isCompileTimeOnly(refinement: model.RefinementType) bool {
    return if (entryForName(refinement.name)) |entry| entry.compile_time_only else false;
}

pub const Bounds = struct {
    base_type: model.Type,
    min_text: ?[]const u8 = null,
    max_text: ?[]const u8 = null,
};

pub fn bounds(refinement: model.RefinementType) ?Bounds {
    const kind = kindForName(refinement.name) orelse return null;
    return switch (kind) {
        .min_value => {
            if (refinement.args.len < 2 or refinement.args[1] != .Integer) return null;
            return .{
                .base_type = refinement.base_type.*,
                .min_text = refinement.args[1].Integer.text,
            };
        },
        .max_value => {
            if (refinement.args.len < 2 or refinement.args[1] != .Integer) return null;
            return .{
                .base_type = refinement.base_type.*,
                .max_text = refinement.args[1].Integer.text,
            };
        },
        .in_range => {
            if (refinement.args.len < 3 or refinement.args[1] != .Integer or refinement.args[2] != .Integer) return null;
            return .{
                .base_type = refinement.base_type.*,
                .min_text = refinement.args[1].Integer.text,
                .max_text = refinement.args[2].Integer.text,
            };
        },
        .basis_points => .{
            .base_type = refinement.base_type.*,
            .min_text = "0",
            .max_text = "10000",
        },
        // Current numeric comptime validation models NonZero integer literals as
        // unsigned values, so "not zero" is represented as the lower bound 1.
        .non_zero => .{
            .base_type = refinement.base_type.*,
            .min_text = "1",
        },
        .exact => null,
        .scaled => null,
        .non_zero_address => null,
    };
}

pub fn expectationText(allocator: std.mem.Allocator, refinement: model.RefinementType) ![]const u8 {
    const kind = kindForName(refinement.name) orelse
        return std.fmt.allocPrint(allocator, "expected {s}", .{refinement.name});
    return switch (kind) {
        .min_value => blk: {
            const info = bounds(refinement) orelse break :blk try allocator.dupe(u8, "expected MinValue");
            break :blk try std.fmt.allocPrint(allocator, "expected MinValue value >= {s}", .{info.min_text.?});
        },
        .max_value => blk: {
            const info = bounds(refinement) orelse break :blk try allocator.dupe(u8, "expected MaxValue");
            break :blk try std.fmt.allocPrint(allocator, "expected MaxValue value <= {s}", .{info.max_text.?});
        },
        .in_range => blk: {
            const info = bounds(refinement) orelse break :blk try allocator.dupe(u8, "expected InRange");
            break :blk try std.fmt.allocPrint(allocator, "expected InRange value between {s} and {s}", .{ info.min_text.?, info.max_text.? });
        },
        .basis_points => try allocator.dupe(u8, "expected BasisPoints value between 0 and 10000"),
        .non_zero => try allocator.dupe(u8, "expected NonZero value != 0"),
        .non_zero_address => try allocator.dupe(u8, "expected NonZeroAddress value != 0"),
        .exact => try allocator.dupe(u8, "expected Exact"),
        .scaled => try allocator.dupe(u8, "expected Scaled"),
    };
}

pub fn refinementType(
    allocator: std.mem.Allocator,
    name: []const u8,
    base_type: model.Type,
    args: []const ast.TypeArg,
) !model.Type {
    return .{ .refinement = .{
        .name = name,
        .base_type = try storeType(allocator, base_type),
        .args = args,
    } };
}

fn storeType(allocator: std.mem.Allocator, ty: model.Type) !*model.Type {
    const ptr = try allocator.create(model.Type);
    ptr.* = ty;
    return ptr;
}

test "refinement registry exposes bounds for bounds-backed refinements" {
    const base: model.Type = .{ .integer = .{ .bits = 256, .signed = false, .spelling = "u256" } };
    const type_arg: ast.TypeArg = .{ .Type = ast.TypeExprId.fromIndex(0) };
    const ten_arg: ast.TypeArg = .{ .Integer = .{ .range = .{ .start = 0, .end = 2 }, .text = "10" } };
    const twenty_arg: ast.TypeArg = .{ .Integer = .{ .range = .{ .start = 0, .end = 2 }, .text = "20" } };

    for (entries) |entry| {
        if (!isBoundsBackedKind(entry.kind)) continue;
        const refinement: model.RefinementType = switch (entry.kind) {
            .min_value, .max_value => .{ .name = entry.name, .base_type = &base, .args = &.{ type_arg, ten_arg } },
            .in_range => .{ .name = entry.name, .base_type = &base, .args = &.{ type_arg, ten_arg, twenty_arg } },
            .basis_points, .non_zero => .{ .name = entry.name, .base_type = &base, .args = &.{type_arg} },
            else => unreachable,
        };
        const info = bounds(refinement) orelse return error.TestUnexpectedResult;
        try std.testing.expect(info.min_text != null or info.max_text != null);
    }

    const basis_points = bounds(.{ .name = nameForKind(.basis_points), .base_type = &base, .args = &.{type_arg} }) orelse return error.TestUnexpectedResult;
    try std.testing.expectEqualStrings("0", basis_points.min_text.?);
    try std.testing.expectEqualStrings("10000", basis_points.max_text.?);

    const non_zero = bounds(.{ .name = nameForKind(.non_zero), .base_type = &base, .args = &.{type_arg} }) orelse return error.TestUnexpectedResult;
    try std.testing.expectEqualStrings("1", non_zero.min_text.?);
}
