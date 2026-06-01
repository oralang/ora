const std = @import("std");

pub const NamedEntry = struct {
    name: []const u8,
    index: usize,
};

pub const PairEntry = struct {
    first: []const u8,
    second: []const u8,
    index: usize,
};

pub fn buildNamed(
    comptime T: type,
    allocator: std.mem.Allocator,
    items: []const T,
    comptime name_field: []const u8,
) ![]NamedEntry {
    const entries = try allocator.alloc(NamedEntry, items.len);
    for (items, 0..) |item, index| {
        entries[index] = .{
            .name = @field(item, name_field),
            .index = index,
        };
    }
    std.sort.heap(NamedEntry, entries, {}, namedLessThan);
    return entries;
}

pub fn buildPair(
    comptime T: type,
    allocator: std.mem.Allocator,
    items: []const T,
    comptime first_field: []const u8,
    comptime second_field: []const u8,
) ![]PairEntry {
    const entries = try allocator.alloc(PairEntry, items.len);
    for (items, 0..) |item, index| {
        entries[index] = .{
            .first = @field(item, first_field),
            .second = @field(item, second_field),
            .index = index,
        };
    }
    std.sort.heap(PairEntry, entries, {}, pairLessThan);
    return entries;
}

pub fn findNamed(entries: []const NamedEntry, name: []const u8) ?usize {
    var left: usize = 0;
    var right: usize = entries.len;
    while (left < right) {
        const mid = left + (right - left) / 2;
        switch (std.mem.order(u8, entries[mid].name, name)) {
            .lt => left = mid + 1,
            .eq, .gt => right = mid,
        }
    }
    if (left >= entries.len) return null;
    if (!std.mem.eql(u8, entries[left].name, name)) return null;
    return entries[left].index;
}

pub fn findNamedItem(comptime T: type, items: []const T, entries: []const NamedEntry, name: []const u8) ?T {
    const index = findNamed(entries, name) orelse return null;
    return items[index];
}

pub fn findPair(entries: []const PairEntry, first: []const u8, second: []const u8) ?usize {
    var left: usize = 0;
    var right: usize = entries.len;
    while (left < right) {
        const mid = left + (right - left) / 2;
        switch (comparePairToKey(entries[mid], first, second)) {
            .lt => left = mid + 1,
            .eq, .gt => right = mid,
        }
    }
    if (left >= entries.len) return null;
    if (!std.mem.eql(u8, entries[left].first, first)) return null;
    if (!std.mem.eql(u8, entries[left].second, second)) return null;
    return entries[left].index;
}

fn namedLessThan(_: void, lhs: NamedEntry, rhs: NamedEntry) bool {
    return switch (std.mem.order(u8, lhs.name, rhs.name)) {
        .lt => true,
        .gt => false,
        .eq => lhs.index < rhs.index,
    };
}

fn pairLessThan(_: void, lhs: PairEntry, rhs: PairEntry) bool {
    return switch (std.mem.order(u8, lhs.first, rhs.first)) {
        .lt => true,
        .gt => false,
        .eq => switch (std.mem.order(u8, lhs.second, rhs.second)) {
            .lt => true,
            .gt => false,
            .eq => lhs.index < rhs.index,
        },
    };
}

fn comparePairToKey(entry: PairEntry, first: []const u8, second: []const u8) std.math.Order {
    return switch (std.mem.order(u8, entry.first, first)) {
        .lt => .lt,
        .gt => .gt,
        .eq => std.mem.order(u8, entry.second, second),
    };
}

test "named lookup returns original first duplicate" {
    const Item = struct {
        name: []const u8,
    };
    const items = [_]Item{
        .{ .name = "Beta" },
        .{ .name = "Alpha" },
        .{ .name = "Beta" },
    };
    const entries = try buildNamed(Item, std.testing.allocator, &items, "name");
    defer std.testing.allocator.free(entries);

    try std.testing.expectEqual(@as(?usize, 1), findNamed(entries, "Alpha"));
    try std.testing.expectEqual(@as(?usize, 0), findNamed(entries, "Beta"));
    try std.testing.expectEqual(@as(?usize, null), findNamed(entries, "Missing"));
    try std.testing.expectEqualStrings("Beta", findNamedItem(Item, &items, entries, "Beta").?.name);
}

test "pair lookup returns original first duplicate" {
    const Item = struct {
        trait_name: []const u8,
        target_name: []const u8,
    };
    const items = [_]Item{
        .{ .trait_name = "Eq", .target_name = "Wallet" },
        .{ .trait_name = "Display", .target_name = "Token" },
        .{ .trait_name = "Eq", .target_name = "Wallet" },
    };
    const entries = try buildPair(Item, std.testing.allocator, &items, "trait_name", "target_name");
    defer std.testing.allocator.free(entries);

    try std.testing.expectEqual(@as(?usize, 1), findPair(entries, "Display", "Token"));
    try std.testing.expectEqual(@as(?usize, 0), findPair(entries, "Eq", "Wallet"));
    try std.testing.expectEqual(@as(?usize, null), findPair(entries, "Eq", "Token"));
}
