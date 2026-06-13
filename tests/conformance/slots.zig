const std = @import("std");
const abi = @import("abi.zig");

pub fn parseSlotExpressionValue(value: []const u8) !u256 {
    const token = if (value.len > 0 and value[0] == '"') try abi.parseString(value) else value;
    return try parseSlotExpression(std.mem.trim(u8, token, " \t"));
}

pub fn parseSlotExpression(text: []const u8) !u256 {
    const trimmed = std.mem.trim(u8, text, " \t");
    if (trimmed.len == 0) return error.InvalidSlotExpression;

    if (callInner(trimmed, "map")) |inner| {
        var parts: [3][]const u8 = undefined;
        if (try abi.splitTopLevelArgs(inner, &parts) != 3) return error.InvalidSlotExpression;
        const root = try parseSlotExpression(parts[2]);
        return try mappingSlot(parts[0], parts[1], root);
    }

    if (callInner(trimmed, "add")) |inner| {
        var parts: [2][]const u8 = undefined;
        if (try abi.splitTopLevelArgs(inner, &parts) != 2) return error.InvalidSlotExpression;
        return (try parseSlotExpression(parts[0])) +% (try abi.parseU256(parts[1]));
    }

    if (callInner(trimmed, "keccak")) |inner| {
        var parts: [1][]const u8 = undefined;
        if (try abi.splitTopLevelArgs(inner, &parts) != 1) return error.InvalidSlotExpression;
        return keccakSlot(try parseSlotExpression(parts[0]));
    }

    return try abi.parseU256(trimmed);
}

fn callInner(text: []const u8, name: []const u8) ?[]const u8 {
    if (!std.mem.startsWith(u8, text, name)) return null;
    if (text.len < name.len + 2 or text[name.len] != '(' or text[text.len - 1] != ')') return null;
    return text[name.len + 1 .. text.len - 1];
}

pub fn mappingSlot(wire_type: []const u8, key_text: []const u8, root_slot: u256) !u256 {
    var input: [64]u8 = undefined;
    if (std.mem.eql(u8, wire_type, "string")) {
        std.mem.writeInt(u256, input[0..32], dynamicKeyHash(std.mem.trim(u8, key_text, " \t")), .big);
    } else if (std.mem.eql(u8, wire_type, "bytes")) {
        const bytes = try abi.parseHexBytes(std.heap.page_allocator, std.mem.trim(u8, key_text, " \t"));
        defer std.heap.page_allocator.free(bytes);
        std.mem.writeInt(u256, input[0..32], dynamicKeyHash(bytes), .big);
    } else {
        try abi.encodeStaticAbiWord(input[0..32], wire_type, try abi.parseArgValue(key_text));
    }
    std.mem.writeInt(u256, input[32..64], root_slot, .big);

    var hash: [32]u8 = undefined;
    std.crypto.hash.sha3.Keccak256.hash(&input, &hash, .{});
    return std.mem.readInt(u256, &hash, .big);
}

fn dynamicKeyHash(bytes: []const u8) u256 {
    var hash: [32]u8 = undefined;
    std.crypto.hash.sha3.Keccak256.hash(bytes, &hash, .{});
    return std.mem.readInt(u256, &hash, .big);
}

pub fn keccakSlot(slot: u256) u256 {
    var input: [32]u8 = undefined;
    std.mem.writeInt(u256, &input, slot, .big);
    var hash: [32]u8 = undefined;
    std.crypto.hash.sha3.Keccak256.hash(&input, &hash, .{});
    return std.mem.readInt(u256, &hash, .big);
}
