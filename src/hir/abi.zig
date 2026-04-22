const std = @import("std");
const crypto = std.crypto;
const sema = @import("../sema/mod.zig");

pub fn canonicalAbiType(allocator: std.mem.Allocator, ty: sema.Type) ![]const u8 {
    return switch (ty) {
        .bool => allocator.dupe(u8, "bool"),
        .address => allocator.dupe(u8, "address"),
        .string => allocator.dupe(u8, "string"),
        .bytes => allocator.dupe(u8, "bytes"),
        .enum_ => error.UnsupportedAbiType,
        .bitfield => allocator.dupe(u8, "uint256"),
        .integer => |integer| blk: {
            const spelling = integer.spelling orelse "u256";
            if (std.mem.eql(u8, spelling, "u256")) break :blk allocator.dupe(u8, "uint256");
            if (std.mem.eql(u8, spelling, "i256")) break :blk allocator.dupe(u8, "int256");
            if (std.mem.startsWith(u8, spelling, "u")) break :blk std.fmt.allocPrint(allocator, "uint{s}", .{spelling[1..]});
            if (std.mem.startsWith(u8, spelling, "i")) break :blk std.fmt.allocPrint(allocator, "int{s}", .{spelling[1..]});
            break :blk allocator.dupe(u8, spelling);
        },
        .refinement => |refinement| canonicalAbiType(allocator, refinement.base_type.*),
        .anonymous_struct => |struct_type| blk: {
            var parts: std.ArrayList([]const u8) = .{};
            defer {
                for (parts.items) |part| allocator.free(part);
                parts.deinit(allocator);
            }
            for (struct_type.fields) |field| {
                try parts.append(allocator, try canonicalAbiType(allocator, field.ty));
            }
            const joined = try std.mem.join(allocator, ",", parts.items);
            defer allocator.free(joined);
            break :blk std.fmt.allocPrint(allocator, "({s})", .{joined});
        },
        .array => |array| blk: {
            const element = try canonicalAbiType(allocator, array.element_type.*);
            defer allocator.free(element);
            if (array.len) |len| break :blk std.fmt.allocPrint(allocator, "{s}[{d}]", .{ element, len });
            break :blk std.fmt.allocPrint(allocator, "{s}[]", .{element});
        },
        .slice => |slice| blk: {
            const element = try canonicalAbiType(allocator, slice.element_type.*);
            defer allocator.free(element);
            break :blk std.fmt.allocPrint(allocator, "{s}[]", .{element});
        },
        .error_union => |error_union| blk: {
            if (error_union.error_types.len != 1) break :blk error.UnsupportedAbiType;
            const payload = try canonicalAbiType(allocator, error_union.payload_type.*);
            defer allocator.free(payload);
            if (!errorTypeHasPayload(error_union.error_types[0])) {
                break :blk std.fmt.allocPrint(allocator, "(bool,{s})", .{payload});
            }
            if (!resultCarrierShapeSupported(error_union.error_types[0])) break :blk error.UnsupportedAbiType;
            if (staticAbiWordCount(error_union.payload_type.*) == 1) {
                const err_words = staticAbiWordCount(error_union.error_types[0]) orelse break :blk error.UnsupportedAbiType;
                if (err_words > 1) break :blk error.UnsupportedAbiType;
            }
            const err_payload = try canonicalAbiType(allocator, error_union.error_types[0]);
            defer allocator.free(err_payload);
            break :blk std.fmt.allocPrint(allocator, "(bool,{s},{s})", .{ payload, err_payload });
        },
        else => error.UnsupportedAbiType,
    };
}

fn errorTypeHasPayload(ty: sema.Type) bool {
    return switch (ty) {
        .named, .anonymous_struct, .tuple, .bytes, .string, .slice, .array, .struct_, .contract => true,
        .bool, .address, .integer, .enum_, .bitfield => true,
        .refinement => |refinement| errorTypeHasPayload(refinement.base_type.*),
        else => false,
    };
}

fn resultDynamicArrayElementSupported(ty: sema.Type) bool {
    return switch (ty) {
        .bool, .address, .integer, .enum_, .bitfield => true,
        .refinement => |refinement| resultDynamicArrayElementSupported(refinement.base_type.*),
        else => false,
    };
}

fn resultCarrierShapeSupported(ty: sema.Type) bool {
    if (staticAbiWordCount(ty) != null) return true;
    return switch (ty) {
        .bytes, .string => true,
        .slice => |slice| resultDynamicArrayElementSupported(slice.element_type.*),
        .array => |array| array.len == null and resultDynamicArrayElementSupported(array.element_type.*),
        .refinement => |refinement| resultCarrierShapeSupported(refinement.base_type.*),
        else => false,
    };
}

pub fn externReturnAbiType(allocator: std.mem.Allocator, ty: sema.Type) ![]const u8 {
    return switch (ty) {
        .tuple => allocator.dupe(u8, "tuple"),
        .anonymous_struct => allocator.dupe(u8, "struct"),
        .bitfield => allocator.dupe(u8, "uint256"),
        .struct_, .contract => allocator.dupe(u8, "tuple"),
        else => canonicalAbiType(allocator, ty),
    };
}

pub fn staticAbiWordCount(ty: sema.Type) ?usize {
    return switch (ty) {
        .bool, .address, .integer, .enum_, .bitfield => 1,
        .refinement => |refinement| staticAbiWordCount(refinement.base_type.*),
        .tuple => |elements| blk: {
            var total: usize = 0;
            for (elements) |element| {
                const words = staticAbiWordCount(element) orelse return null;
                total += words;
            }
            break :blk total;
        },
        .anonymous_struct => |struct_type| blk: {
            var total: usize = 0;
            for (struct_type.fields) |field| {
                const words = staticAbiWordCount(field.ty) orelse return null;
                total += words;
            }
            break :blk total;
        },
        .array => |array| blk: {
            const len = array.len orelse return null;
            const element_words = staticAbiWordCount(array.element_type.*) orelse return null;
            break :blk element_words * len;
        },
        .slice => null,
        else => null,
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
