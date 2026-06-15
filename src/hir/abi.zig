const std = @import("std");
const crypto = std.crypto;
const ast = @import("../ast/mod.zig");

pub fn metadataStringLiteral(file: *const ast.AstFile, metadata: []const ast.ItemId, name: []const u8) ?[]const u8 {
    for (metadata) |metadata_id| {
        const metadata_item = file.item(metadata_id).*;
        if (metadata_item != .Constant) continue;
        const constant = metadata_item.Constant;
        if (!std.mem.eql(u8, constant.name, name)) continue;
        return switch (file.expression(constant.value).*) {
            .StringLiteral => |literal| literal.text,
            else => null,
        };
    }
    return null;
}

pub fn eventWireNameFromLogDecl(file: *const ast.AstFile, log_decl: ast.LogDeclItem) ?[]const u8 {
    return metadataStringLiteral(file, log_decl.metadata, "event_name") orelse log_decl.name;
}

pub fn signatureForAbiTypes(allocator: std.mem.Allocator, name: []const u8, abi_types: []const []const u8) ![]const u8 {
    const joined = try std.mem.join(allocator, ",", abi_types);
    defer allocator.free(joined);
    return std.fmt.allocPrint(allocator, "{s}({s})", .{ name, joined });
}

pub fn keccakSelectorHex(allocator: std.mem.Allocator, signature: []const u8) ![]const u8 {
    const value = keccakSelectorValue(signature);
    var hex: [8]u8 = undefined;
    const bytes = [_]u8{
        @intCast((value >> 24) & 0xff),
        @intCast((value >> 16) & 0xff),
        @intCast((value >> 8) & 0xff),
        @intCast(value & 0xff),
    };
    for (bytes, 0..) |byte, index| {
        hex[index * 2] = std.fmt.hex_charset[byte >> 4];
        hex[index * 2 + 1] = std.fmt.hex_charset[byte & 0x0f];
    }
    return std.fmt.allocPrint(allocator, "0x{s}", .{hex[0..]});
}

pub fn keccakTopicHex(allocator: std.mem.Allocator, signature: []const u8) ![]const u8 {
    const hash = keccak256(signature);
    var hex: [64]u8 = undefined;
    for (hash, 0..) |byte, index| {
        hex[index * 2] = std.fmt.hex_charset[byte >> 4];
        hex[index * 2 + 1] = std.fmt.hex_charset[byte & 0x0f];
    }
    return std.fmt.allocPrint(allocator, "0x{s}", .{hex[0..]});
}

/// EVM keccak-256 (original Keccak, 0x01 padding) - single source of truth.
pub fn keccak256(bytes: []const u8) [32]u8 {
    var hash: [32]u8 = undefined;
    crypto.hash.sha3.Keccak256.hash(bytes, &hash, .{});
    return hash;
}

pub fn keccakSelectorValue(signature: []const u8) u32 {
    const hash = keccak256(signature);
    const selector = hash[0..4];
    return (@as(u32, selector[0]) << 24) |
        (@as(u32, selector[1]) << 16) |
        (@as(u32, selector[2]) << 8) |
        @as(u32, selector[3]);
}
