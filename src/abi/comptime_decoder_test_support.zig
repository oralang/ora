const std = @import("std");
const abi_layout = @import("layout.zig");
const abi_comptime_decoder = @import("comptime_decoder.zig");
const comptime_mod = @import("../comptime/mod.zig");
const Type = @import("ora_types").SemanticType;

pub fn decodeHexBytes(allocator: std.mem.Allocator, hex_with_optional_prefix: []const u8) ![]u8 {
    const hex = if (std.mem.startsWith(u8, hex_with_optional_prefix, "0x")) hex_with_optional_prefix[2..] else hex_with_optional_prefix;
    if (hex.len % 2 != 0) return error.InvalidHexByteLength;
    const bytes = try allocator.alloc(u8, hex.len / 2);
    errdefer allocator.free(bytes);
    for (bytes, 0..) |*byte, index| {
        const hi = try std.fmt.charToDigit(hex[index * 2], 16);
        const lo = try std.fmt.charToDigit(hex[index * 2 + 1], 16);
        byte.* = @intCast((hi << 4) | lo);
    }
    return bytes;
}

pub fn expectDecodeErrorForType(
    allocator: std.mem.Allocator,
    target_type: Type,
    payload_hex: []const u8,
    resolver: abi_comptime_decoder.TypeResolver,
    expected: abi_comptime_decoder.DecodeError,
) !void {
    var arena = std.heap.ArenaAllocator.init(allocator);
    defer arena.deinit();
    const arena_allocator = arena.allocator();

    var layout = try abi_layout.fromType(arena_allocator, target_type);
    defer layout.deinit(arena_allocator);

    const bytes = try decodeHexBytes(allocator, payload_hex);
    defer allocator.free(bytes);

    var heap = comptime_mod.CtHeap.init(arena_allocator);
    defer heap.deinit();

    const decoded = try abi_comptime_decoder.decodeComptimeValue(
        arena_allocator,
        &heap,
        resolver,
        layout,
        target_type,
        bytes,
    );
    if (decoded != .err or decoded.err != expected) return error.UnexpectedDecodeError;
}

pub fn expectDecodeErrorBytesForType(
    allocator: std.mem.Allocator,
    target_type: Type,
    bytes: []const u8,
    resolver: abi_comptime_decoder.TypeResolver,
    expected: abi_comptime_decoder.DecodeError,
) !void {
    var arena = std.heap.ArenaAllocator.init(allocator);
    defer arena.deinit();
    const arena_allocator = arena.allocator();

    var layout = try abi_layout.fromType(arena_allocator, target_type);
    defer layout.deinit(arena_allocator);

    var heap = comptime_mod.CtHeap.init(arena_allocator);
    defer heap.deinit();

    const decoded = try abi_comptime_decoder.decodeComptimeValue(
        arena_allocator,
        &heap,
        resolver,
        layout,
        target_type,
        bytes,
    );
    if (decoded != .err or decoded.err != expected) return error.UnexpectedDecodeError;
}
