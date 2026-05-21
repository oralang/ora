const std = @import("std");
const crypto = std.crypto;
const ast = @import("../ast/mod.zig");
const sema = @import("../sema/mod.zig");

pub fn canonicalAbiType(allocator: std.mem.Allocator, ty: sema.Type) ![]const u8 {
    return switch (ty) {
        .bool => allocator.dupe(u8, "bool"),
        .address => allocator.dupe(u8, "address"),
        .string => allocator.dupe(u8, "string"),
        .bytes => allocator.dupe(u8, "bytes"),
        .fixed_bytes => |fixed_bytes| fixedBytesAbiType(allocator, fixed_bytes),
        .named => |named| if (parseFixedBytesSpelling(named.name)) |len|
            fixedBytesAbiType(allocator, .{ .len = len, .spelling = named.name })
        else
            error.UnsupportedAbiType,
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

pub fn parseFixedBytesSpelling(name: []const u8) ?u8 {
    if (!std.mem.startsWith(u8, name, "bytes")) return null;
    if (name.len <= "bytes".len) return null;
    const digits = name["bytes".len..];
    if (digits.len > 1 and digits[0] == '0') return null;
    const len = std.fmt.parseUnsigned(u8, digits, 10) catch return null;
    if (len < 1 or len > 32) return null;
    return len;
}

pub fn fixedBytesAbiType(allocator: std.mem.Allocator, fixed_bytes: sema.FixedBytesType) ![]const u8 {
    if (fixed_bytes.spelling) |spelling| return allocator.dupe(u8, spelling);
    return std.fmt.allocPrint(allocator, "bytes{d}", .{fixed_bytes.len});
}

fn errorTypeHasPayload(ty: sema.Type) bool {
    return switch (ty) {
        .named, .anonymous_struct, .tuple, .bytes, .string, .slice, .array, .struct_, .contract => true,
        .bool, .address, .fixed_bytes, .integer, .enum_, .bitfield => true,
        .refinement => |refinement| errorTypeHasPayload(refinement.base_type.*),
        else => false,
    };
}

fn resultDynamicArrayElementSupported(ty: sema.Type) bool {
    return switch (ty) {
        .bool, .address, .fixed_bytes, .integer, .enum_, .bitfield => true,
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
        .bool, .address, .fixed_bytes, .integer, .enum_, .bitfield => 1,
        .named => |named| if (parseFixedBytesSpelling(named.name) != null) 1 else null,
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

pub fn eip712WireNameFromStructItem(file: *const ast.AstFile, struct_item: ast.StructItem) ?[]const u8 {
    return metadataStringLiteral(file, struct_item.metadata, "eip712_name") orelse struct_item.name;
}

pub fn signatureForAbiTypes(allocator: std.mem.Allocator, name: []const u8, abi_types: []const []const u8) ![]const u8 {
    const joined = try std.mem.join(allocator, ",", abi_types);
    defer allocator.free(joined);
    return std.fmt.allocPrint(allocator, "{s}({s})", .{ name, joined });
}

pub const Eip712Field = struct {
    name: []const u8,
    abi_type: []const u8,
};

pub fn signatureForEip712Type(allocator: std.mem.Allocator, type_name: []const u8, fields: []const Eip712Field) ![]const u8 {
    var buffer: std.ArrayList(u8) = .{};
    defer buffer.deinit(allocator);

    try buffer.appendSlice(allocator, type_name);
    try buffer.append(allocator, '(');
    for (fields, 0..) |field, index| {
        if (index > 0) try buffer.append(allocator, ',');
        try buffer.appendSlice(allocator, field.abi_type);
        try buffer.append(allocator, ' ');
        try buffer.appendSlice(allocator, field.name);
    }
    try buffer.append(allocator, ')');

    return buffer.toOwnedSlice(allocator);
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

pub fn keccakSelectorValue(signature: []const u8) u32 {
    var hash: [32]u8 = undefined;
    crypto.hash.sha3.Keccak256.hash(signature, &hash, .{});
    const selector = hash[0..4];
    return (@as(u32, selector[0]) << 24) |
        (@as(u32, selector[1]) << 16) |
        (@as(u32, selector[2]) << 8) |
        @as(u32, selector[3]);
}
