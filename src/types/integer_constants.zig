const std = @import("std");

const Entry = struct {
    name: []const u8,
    value: []const u8,
};

const entries = [_]Entry{
    .{ .name = "U8_MIN", .value = "0" },
    .{ .name = "U8_MAX", .value = "255" },
    .{ .name = "U16_MIN", .value = "0" },
    .{ .name = "U16_MAX", .value = "65535" },
    .{ .name = "U32_MIN", .value = "0" },
    .{ .name = "U32_MAX", .value = "4294967295" },
    .{ .name = "U64_MIN", .value = "0" },
    .{ .name = "U64_MAX", .value = "18446744073709551615" },
    .{ .name = "U128_MIN", .value = "0" },
    .{ .name = "U128_MAX", .value = "340282366920938463463374607431768211455" },
    .{ .name = "U160_MIN", .value = "0" },
    .{ .name = "U160_MAX", .value = "1461501637330902918203684832716283019655932542975" },
    .{ .name = "U256_MIN", .value = "0" },
    .{ .name = "U256_MAX", .value = "115792089237316195423570985008687907853269984665640564039457584007913129639935" },
    .{ .name = "I8_MIN", .value = "-128" },
    .{ .name = "I8_MAX", .value = "127" },
    .{ .name = "I16_MIN", .value = "-32768" },
    .{ .name = "I16_MAX", .value = "32767" },
    .{ .name = "I32_MIN", .value = "-2147483648" },
    .{ .name = "I32_MAX", .value = "2147483647" },
    .{ .name = "I64_MIN", .value = "-9223372036854775808" },
    .{ .name = "I64_MAX", .value = "9223372036854775807" },
    .{ .name = "I128_MIN", .value = "-170141183460469231731687303715884105728" },
    .{ .name = "I128_MAX", .value = "170141183460469231731687303715884105727" },
    .{ .name = "I256_MIN", .value = "-57896044618658097711785492504343953926634992332820282019728792003956564819968" },
    .{ .name = "I256_MAX", .value = "57896044618658097711785492504343953926634992332820282019728792003956564819967" },
};

const map = blk: {
    var pairs: [entries.len]struct { []const u8, []const u8 } = undefined;
    for (entries, 0..) |entry, index| {
        pairs[index] = .{ entry.name, entry.value };
    }
    break :blk std.StaticStringMap([]const u8).initComptime(pairs);
};

pub fn lookup(name: []const u8) ?[]const u8 {
    const trimmed = std.mem.trim(u8, name, " \t\n\r");
    if (!std.mem.startsWith(u8, trimmed, "std.constants.")) return null;
    const short = trimmed["std.constants.".len..];
    return map.get(short);
}
