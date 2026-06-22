const std = @import("std");
const semantic = @import("semantic.zig");
const registry = @import("ora_refinements");

pub const Kind = registry.Kind;
pub const Entry = registry.Entry;
pub const entries = registry.entries;
pub const entryForName = registry.entryForName;
pub const entryForKind = registry.entryForKind;
pub const kindForName = registry.kindForName;
pub const kindForGuardText = registry.kindForGuardText;
pub const nameForKind = registry.nameForKind;
pub const guardFailureSubtype = registry.guardFailureSubtype;
pub const isKnownName = registry.isKnownName;
pub const isPathFormName = registry.isPathFormName;
pub const hasNativeMlirTypeName = registry.hasNativeMlirTypeName;
pub const isBoundsBackedKind = registry.isBoundsBackedKind;

pub fn supportsRuntimeGuard(refinement: semantic.RefinementType) bool {
    return if (entryForName(refinement.name)) |entry| entry.has_runtime_guard else false;
}

pub fn isCompileTimeOnly(refinement: semantic.RefinementType) bool {
    return if (entryForName(refinement.name)) |entry| entry.compile_time_only else false;
}

pub const Bounds = struct {
    base_type: semantic.Type,
    min_text: ?[]const u8 = null,
    max_text: ?[]const u8 = null,
};

pub fn bounds(refinement: semantic.RefinementType) ?Bounds {
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

pub fn expectationText(allocator: std.mem.Allocator, refinement: semantic.RefinementType) ![]const u8 {
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

pub fn writeGuardViolationExplanation(writer: anytype, refinement_name: []const u8, variable_name: []const u8) !bool {
    const kind = kindForName(refinement_name) orelse return false;
    switch (kind) {
        .non_zero_address => try writer.print("`{s}` can be the zero address", .{variable_name}),
        .non_zero => try writer.print("`{s}` can be zero", .{variable_name}),
        .min_value => try writer.print("`{s}` can be below the minimum value", .{variable_name}),
        .max_value => try writer.print("`{s}` can exceed the maximum value", .{variable_name}),
        .in_range, .basis_points => try writer.print("`{s}` can be out of range", .{variable_name}),
        // Compile-time-only refinements never produce runtime guard violations.
        .exact, .scaled => return false,
    }
    return true;
}

pub fn hasGuardViolationExplanation(refinement_name: []const u8) bool {
    const kind = kindForName(refinement_name) orelse return false;
    return switch (kind) {
        .exact, .scaled => false,
        else => true,
    };
}

pub fn refinementType(
    allocator: std.mem.Allocator,
    name: []const u8,
    base_type: semantic.Type,
    args: []const semantic.RefinementArg,
) !semantic.Type {
    return .{ .refinement = .{
        .name = name,
        .base_type = try storeType(allocator, base_type),
        .args = args,
    } };
}

fn storeType(allocator: std.mem.Allocator, ty: semantic.Type) !*semantic.Type {
    const ptr = try allocator.create(semantic.Type);
    ptr.* = ty;
    return ptr;
}

test "refinement semantics expose bounds for bounds-backed refinements" {
    const base: semantic.Type = .{ .integer = .{ .bits = 256, .signed = false, .spelling = "u256" } };
    const type_arg: semantic.RefinementArg = .{ .Type = {} };
    const ten_arg: semantic.RefinementArg = .{ .Integer = .{ .text = "10" } };
    const twenty_arg: semantic.RefinementArg = .{ .Integer = .{ .text = "20" } };

    for (entries) |entry| {
        if (!isBoundsBackedKind(entry.kind)) continue;
        const refinement: semantic.RefinementType = switch (entry.kind) {
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

test "refinement semantics own runtime guard explanations" {
    var buffer: std.ArrayList(u8) = .empty;
    defer buffer.deinit(std.testing.allocator);

    try std.testing.expect(try writeGuardViolationExplanation(buffer.writer(std.testing.allocator), nameForKind(.min_value), "amount"));
    try std.testing.expectEqualStrings("`amount` can be below the minimum value", buffer.items);

    buffer.clearRetainingCapacity();
    try std.testing.expect(try writeGuardViolationExplanation(buffer.writer(std.testing.allocator), nameForKind(.max_value), "amount"));
    try std.testing.expectEqualStrings("`amount` can exceed the maximum value", buffer.items);

    buffer.clearRetainingCapacity();
    try std.testing.expect(try writeGuardViolationExplanation(buffer.writer(std.testing.allocator), nameForKind(.in_range), "amount"));
    try std.testing.expectEqualStrings("`amount` can be out of range", buffer.items);

    buffer.clearRetainingCapacity();
    try std.testing.expect(try writeGuardViolationExplanation(buffer.writer(std.testing.allocator), nameForKind(.non_zero_address), "owner"));
    try std.testing.expectEqualStrings("`owner` can be the zero address", buffer.items);

    buffer.clearRetainingCapacity();
    try std.testing.expect(try writeGuardViolationExplanation(buffer.writer(std.testing.allocator), nameForKind(.non_zero), "amount"));
    try std.testing.expectEqualStrings("`amount` can be zero", buffer.items);

    buffer.clearRetainingCapacity();
    try std.testing.expect(try writeGuardViolationExplanation(buffer.writer(std.testing.allocator), nameForKind(.basis_points), "rate"));
    try std.testing.expectEqualStrings("`rate` can be out of range", buffer.items);

    buffer.clearRetainingCapacity();
    const exact_has_explanation = try writeGuardViolationExplanation(buffer.writer(std.testing.allocator), nameForKind(.exact), "value");
    try std.testing.expect(!exact_has_explanation);
    try std.testing.expectEqual(@as(usize, 0), buffer.items.len);

    buffer.clearRetainingCapacity();
    const scaled_has_explanation = try writeGuardViolationExplanation(buffer.writer(std.testing.allocator), nameForKind(.scaled), "value");
    try std.testing.expect(!scaled_has_explanation);
    try std.testing.expectEqual(@as(usize, 0), buffer.items.len);

    buffer.clearRetainingCapacity();
    const unknown_has_explanation = try writeGuardViolationExplanation(buffer.writer(std.testing.allocator), "UnknownRefinement", "value");
    try std.testing.expect(!unknown_has_explanation);
    try std.testing.expectEqual(@as(usize, 0), buffer.items.len);
}
