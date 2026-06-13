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

pub const MemberEntry = struct {
    owner_index: usize,
    name: []const u8,
    index: usize,
};

pub const IndexEntry = struct {
    key_index: usize,
    index: usize,
};

pub const EntryRange = struct {
    start: usize,
    end: usize,
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
    const range = findNamedRange(entries, name) orelse return null;
    return entries[range.start].index;
}

pub fn findNamedRange(entries: []const NamedEntry, name: []const u8) ?EntryRange {
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

    var end = left + 1;
    while (end < entries.len and std.mem.eql(u8, entries[end].name, name)) {
        end += 1;
    }
    return .{ .start = left, .end = end };
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

pub fn sortMembers(entries: []MemberEntry) void {
    std.sort.heap(MemberEntry, entries, {}, memberLessThan);
}

pub fn sortIndexes(entries: []IndexEntry) void {
    std.sort.heap(IndexEntry, entries, {}, indexLessThan);
}

pub fn findIndex(entries: []const IndexEntry, key_index: usize) ?usize {
    var left: usize = 0;
    var right: usize = entries.len;
    while (left < right) {
        const mid = left + (right - left) / 2;
        switch (compareIndexToKey(entries[mid], key_index)) {
            .lt => left = mid + 1,
            .eq, .gt => right = mid,
        }
    }
    if (left >= entries.len) return null;
    if (entries[left].key_index != key_index) return null;
    return entries[left].index;
}

pub fn findMember(entries: []const MemberEntry, owner_index: usize, name: []const u8) ?usize {
    const range = findMemberRange(entries, owner_index, name) orelse return null;
    return entries[range.start].index;
}

pub fn findMemberRange(entries: []const MemberEntry, owner_index: usize, name: []const u8) ?EntryRange {
    var left: usize = 0;
    var right: usize = entries.len;
    while (left < right) {
        const mid = left + (right - left) / 2;
        switch (compareMemberToKey(entries[mid], owner_index, name)) {
            .lt => left = mid + 1,
            .eq, .gt => right = mid,
        }
    }
    if (left >= entries.len) return null;
    if (entries[left].owner_index != owner_index) return null;
    if (!std.mem.eql(u8, entries[left].name, name)) return null;

    var end = left + 1;
    while (end < entries.len and
        entries[end].owner_index == owner_index and
        std.mem.eql(u8, entries[end].name, name))
    {
        end += 1;
    }
    return .{ .start = left, .end = end };
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

fn memberLessThan(_: void, lhs: MemberEntry, rhs: MemberEntry) bool {
    if (lhs.owner_index < rhs.owner_index) return true;
    if (lhs.owner_index > rhs.owner_index) return false;
    return switch (std.mem.order(u8, lhs.name, rhs.name)) {
        .lt => true,
        .gt => false,
        .eq => lhs.index < rhs.index,
    };
}

fn indexLessThan(_: void, lhs: IndexEntry, rhs: IndexEntry) bool {
    if (lhs.key_index < rhs.key_index) return true;
    if (lhs.key_index > rhs.key_index) return false;
    return lhs.index < rhs.index;
}

fn comparePairToKey(entry: PairEntry, first: []const u8, second: []const u8) std.math.Order {
    return switch (std.mem.order(u8, entry.first, first)) {
        .lt => .lt,
        .gt => .gt,
        .eq => std.mem.order(u8, entry.second, second),
    };
}

fn compareMemberToKey(entry: MemberEntry, owner_index: usize, name: []const u8) std.math.Order {
    if (entry.owner_index < owner_index) return .lt;
    if (entry.owner_index > owner_index) return .gt;
    return std.mem.order(u8, entry.name, name);
}

fn compareIndexToKey(entry: IndexEntry, key_index: usize) std.math.Order {
    if (entry.key_index < key_index) return .lt;
    if (entry.key_index > key_index) return .gt;
    return .eq;
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
    const range = findNamedRange(entries, "Beta").?;
    try std.testing.expectEqual(@as(usize, 2), range.end - range.start);
    try std.testing.expectEqual(@as(usize, 0), entries[range.start].index);
    try std.testing.expectEqual(@as(usize, 2), entries[range.start + 1].index);
}

test "member lookup returns original first duplicate within owner" {
    var entries = [_]MemberEntry{
        .{ .owner_index = 2, .name = "Beta", .index = 0 },
        .{ .owner_index = 1, .name = "Beta", .index = 1 },
        .{ .owner_index = 2, .name = "Alpha", .index = 2 },
        .{ .owner_index = 2, .name = "Beta", .index = 3 },
    };
    sortMembers(&entries);

    try std.testing.expectEqual(@as(?usize, 2), findMember(&entries, 2, "Alpha"));
    try std.testing.expectEqual(@as(?usize, 0), findMember(&entries, 2, "Beta"));
    try std.testing.expectEqual(@as(?usize, 1), findMember(&entries, 1, "Beta"));
    try std.testing.expectEqual(@as(?usize, null), findMember(&entries, 1, "Alpha"));
    const range = findMemberRange(&entries, 2, "Beta").?;
    try std.testing.expectEqual(@as(usize, 2), range.end - range.start);
    try std.testing.expectEqual(@as(usize, 0), entries[range.start].index);
    try std.testing.expectEqual(@as(usize, 3), entries[range.start + 1].index);
}

test "index lookup returns original first duplicate" {
    var entries = [_]IndexEntry{
        .{ .key_index = 8, .index = 3 },
        .{ .key_index = 2, .index = 1 },
        .{ .key_index = 8, .index = 0 },
    };
    sortIndexes(&entries);

    try std.testing.expectEqual(@as(?usize, 1), findIndex(&entries, 2));
    try std.testing.expectEqual(@as(?usize, 0), findIndex(&entries, 8));
    try std.testing.expectEqual(@as(?usize, null), findIndex(&entries, 4));
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
