const std = @import("std");

pub fn clone(comptime T: type, allocator: std.mem.Allocator, items: []const T) !std.ArrayList(T) {
    var result: std.ArrayList(T) = .{};
    errdefer result.deinit(allocator);
    try result.appendSlice(allocator, items);
    return result;
}

pub fn appendUnique(
    comptime T: type,
    allocator: std.mem.Allocator,
    items: *std.ArrayList(T),
    item: T,
    comptime eql: fn (T, T) bool,
) !void {
    for (items.items) |existing| {
        if (eql(existing, item)) return;
    }
    try items.append(allocator, item);
}

pub fn mergeUnique(
    comptime T: type,
    allocator: std.mem.Allocator,
    dst: *std.ArrayList(T),
    src: []const T,
    comptime eql: fn (T, T) bool,
    comptime include: ?fn (T) bool,
) !void {
    for (src) |item| {
        if (include) |predicate| {
            if (!predicate(item)) continue;
        }
        try appendUnique(T, allocator, dst, item, eql);
    }
}

pub fn intersect(
    comptime T: type,
    allocator: std.mem.Allocator,
    lhs: []const T,
    rhs: []const T,
    comptime eql: fn (T, T) bool,
    comptime include: ?fn (T) bool,
) !std.ArrayList(T) {
    var result: std.ArrayList(T) = .{};
    errdefer result.deinit(allocator);
    for (lhs) |lhs_item| {
        if (include) |predicate| {
            if (!predicate(lhs_item)) continue;
        }
        for (rhs) |rhs_item| {
            if (include) |predicate| {
                if (!predicate(rhs_item)) continue;
            }
            if (!eql(lhs_item, rhs_item)) continue;
            try result.append(allocator, lhs_item);
            break;
        }
    }
    return result;
}

fn u8Eql(lhs: u8, rhs: u8) bool {
    return lhs == rhs;
}

fn even(value: u8) bool {
    return value % 2 == 0;
}

test "appendUnique keeps first occurrence order" {
    var items: std.ArrayList(u8) = .{};
    defer items.deinit(std.testing.allocator);

    try appendUnique(u8, std.testing.allocator, &items, 1, u8Eql);
    try appendUnique(u8, std.testing.allocator, &items, 2, u8Eql);
    try appendUnique(u8, std.testing.allocator, &items, 1, u8Eql);

    try std.testing.expectEqualSlices(u8, &.{ 1, 2 }, items.items);
}

test "mergeUnique applies optional filter" {
    var items: std.ArrayList(u8) = .{};
    defer items.deinit(std.testing.allocator);

    try mergeUnique(u8, std.testing.allocator, &items, &.{ 1, 2, 3, 4, 2 }, u8Eql, even);

    try std.testing.expectEqualSlices(u8, &.{ 2, 4 }, items.items);
}

test "intersect preserves lhs order" {
    var items = try intersect(u8, std.testing.allocator, &.{ 4, 3, 2, 1 }, &.{ 1, 2, 4 }, u8Eql, null);
    defer items.deinit(std.testing.allocator);

    try std.testing.expectEqualSlices(u8, &.{ 4, 2, 1 }, items.items);
}
