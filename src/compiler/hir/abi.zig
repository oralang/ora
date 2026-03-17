const std = @import("std");
const crypto = std.crypto;
const sema = @import("../sema/mod.zig");

pub fn canonicalAbiType(allocator: std.mem.Allocator, ty: sema.Type) ![]const u8 {
    return switch (ty) {
        .bool => allocator.dupe(u8, "bool"),
        .address => allocator.dupe(u8, "address"),
        .string => allocator.dupe(u8, "string"),
        .bytes => allocator.dupe(u8, "bytes"),
        .integer => |integer| blk: {
            const spelling = integer.spelling orelse "u256";
            if (std.mem.eql(u8, spelling, "u256")) break :blk allocator.dupe(u8, "uint256");
            if (std.mem.eql(u8, spelling, "i256")) break :blk allocator.dupe(u8, "int256");
            if (std.mem.startsWith(u8, spelling, "u")) break :blk std.fmt.allocPrint(allocator, "uint{s}", .{spelling[1..]});
            if (std.mem.startsWith(u8, spelling, "i")) break :blk std.fmt.allocPrint(allocator, "int{s}", .{spelling[1..]});
            break :blk allocator.dupe(u8, spelling);
        },
        .refinement => |refinement| canonicalAbiType(allocator, refinement.base_type.*),
        .array => |array| blk: {
            const element = try canonicalAbiType(allocator, array.element_type.*);
            defer allocator.free(element);
            if (array.len) |len| break :blk std.fmt.allocPrint(allocator, "{s}[{d}]", .{ element, len });
            break :blk std.fmt.allocPrint(allocator, "{s}[]", .{element});
        },
        else => error.UnsupportedAbiType,
    };
}

pub fn externReturnAbiType(allocator: std.mem.Allocator, ty: sema.Type) ![]const u8 {
    return switch (ty) {
        .tuple => allocator.dupe(u8, "tuple"),
        .struct_, .contract => allocator.dupe(u8, "tuple"),
        else => canonicalAbiType(allocator, ty),
    };
}

pub fn signatureForMethod(allocator: std.mem.Allocator, name: []const u8, has_self: bool, param_types: []const sema.Type) ![]const u8 {
    var signature_parts: std.ArrayList([]const u8) = .{};
    defer {
        for (signature_parts.items) |part| allocator.free(part);
        signature_parts.deinit(allocator);
    }

    _ = has_self;
    for (param_types) |param_type| {
        try signature_parts.append(allocator, try canonicalAbiType(allocator, param_type));
    }

    const joined = try std.mem.join(allocator, ",", signature_parts.items);
    defer allocator.free(joined);
    return std.fmt.allocPrint(allocator, "{s}({s})", .{ name, joined });
}

pub fn keccakSelectorHex(allocator: std.mem.Allocator, signature: []const u8) ![]const u8 {
    var hash: [32]u8 = undefined;
    crypto.hash.sha3.Keccak256.hash(signature, &hash, .{});
    const selector = hash[0..4];

    var hex: [8]u8 = undefined;
    for (selector, 0..) |byte, index| {
        hex[index * 2] = std.fmt.hex_charset[byte >> 4];
        hex[index * 2 + 1] = std.fmt.hex_charset[byte & 0x0f];
    }
    return std.fmt.allocPrint(allocator, "0x{s}", .{hex[0..]});
}
