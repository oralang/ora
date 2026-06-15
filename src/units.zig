const std = @import("std");

pub const EtherUnit = struct {
    name: []const u8,
    factor: u64,
};

/// Canonical Ethereum unit table. Source of truth for both lexical recognition
/// and comptime scaling.
pub const ether_units = [_]EtherUnit{
    .{ .name = "wei", .factor = 1 },
    .{ .name = "kwei", .factor = 1_000 },
    .{ .name = "mwei", .factor = 1_000_000 },
    .{ .name = "gwei", .factor = 1_000_000_000 },
    .{ .name = "microether", .factor = 1_000_000_000_000 },
    .{ .name = "milliether", .factor = 1_000_000_000_000_000 },
    .{ .name = "ether", .factor = 1_000_000_000_000_000_000 },
};

pub fn etherUnitFactor(unit: []const u8) ?u64 {
    for (ether_units) |entry| {
        if (std.mem.eql(u8, entry.name, unit)) return entry.factor;
    }
    return null;
}

pub fn isEtherUnit(unit: []const u8) bool {
    return etherUnitFactor(unit) != null;
}
