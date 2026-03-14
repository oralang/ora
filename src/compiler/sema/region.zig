const std = @import("std");
const model = @import("model.zig");
const type_descriptors = @import("type_descriptors.zig");

const LocatedType = model.LocatedType;
const Region = model.Region;

pub fn regionDisplayName(region: Region) []const u8 {
    return switch (region) {
        .none => "none",
        .storage => "storage",
        .memory => "memory",
        .transient => "transient",
        .calldata => "calldata",
    };
}

pub fn locatedTypeEql(lhs: LocatedType, rhs: LocatedType) bool {
    return lhs.region == rhs.region and type_descriptors.typeEql(lhs.type, rhs.type);
}

pub fn isAssignable(from: LocatedType, to: LocatedType) bool {
    return type_descriptors.typeEql(from.type, to.type) and regionAssignable(from.region, to.region);
}

pub fn regionAssignable(from: Region, to: Region) bool {
    if (from == to) return true;

    return switch (from) {
        .none => true,
        .memory => switch (to) {
            .none, .storage, .transient => true,
            .memory, .calldata => false,
        },
        .storage => switch (to) {
            .none, .memory => true,
            .storage, .transient, .calldata => false,
        },
        .transient => switch (to) {
            .none, .memory => true,
            .storage, .transient, .calldata => false,
        },
        .calldata => switch (to) {
            .none, .memory => true,
            .storage, .transient, .calldata => false,
        },
    };
}

test "regionDisplayName formats all regions" {
    try std.testing.expectEqualStrings("none", regionDisplayName(.none));
    try std.testing.expectEqualStrings("storage", regionDisplayName(.storage));
    try std.testing.expectEqualStrings("memory", regionDisplayName(.memory));
    try std.testing.expectEqualStrings("transient", regionDisplayName(.transient));
    try std.testing.expectEqualStrings("calldata", regionDisplayName(.calldata));
}

test "locatedTypeEql compares value type and region" {
    const lhs = LocatedType.withRegion(.{ .bool = {} }, .storage);
    const rhs_same = LocatedType.withRegion(.{ .bool = {} }, .storage);
    const rhs_other_region = LocatedType.withRegion(.{ .bool = {} }, .memory);
    const rhs_other_type = LocatedType.withRegion(.{ .address = {} }, .storage);

    try std.testing.expect(locatedTypeEql(lhs, rhs_same));
    try std.testing.expect(!locatedTypeEql(lhs, rhs_other_region));
    try std.testing.expect(!locatedTypeEql(lhs, rhs_other_type));
}

test "regionAssignable follows implicit coercion rules" {
    try std.testing.expect(regionAssignable(.none, .storage));
    try std.testing.expect(regionAssignable(.none, .calldata));
    try std.testing.expect(regionAssignable(.memory, .storage));
    try std.testing.expect(regionAssignable(.memory, .transient));
    try std.testing.expect(regionAssignable(.storage, .none));
    try std.testing.expect(regionAssignable(.storage, .memory));
    try std.testing.expect(regionAssignable(.transient, .none));
    try std.testing.expect(regionAssignable(.transient, .memory));
    try std.testing.expect(regionAssignable(.calldata, .none));
    try std.testing.expect(regionAssignable(.calldata, .memory));

    try std.testing.expect(!regionAssignable(.memory, .calldata));
    try std.testing.expect(!regionAssignable(.storage, .transient));
    try std.testing.expect(!regionAssignable(.transient, .storage));
    try std.testing.expect(!regionAssignable(.storage, .calldata));
    try std.testing.expect(!regionAssignable(.transient, .calldata));
}

test "isAssignable combines type equality and region coercion" {
    const storage_bool = LocatedType.withRegion(.{ .bool = {} }, .storage);
    const memory_bool = LocatedType.withRegion(.{ .bool = {} }, .memory);
    const calldata_bool = LocatedType.withRegion(.{ .bool = {} }, .calldata);
    const memory_address = LocatedType.withRegion(.{ .address = {} }, .memory);

    try std.testing.expect(isAssignable(storage_bool, memory_bool));
    try std.testing.expect(isAssignable(calldata_bool, memory_bool));
    try std.testing.expect(!isAssignable(memory_bool, calldata_bool));
    try std.testing.expect(!isAssignable(storage_bool, memory_address));
}
