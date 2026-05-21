const std = @import("std");
const crypto = std.crypto;
const ast = @import("../ast/mod.zig");
const sema = @import("../sema/mod.zig");
const abi_layout = @import("../abi/layout.zig");

pub fn canonicalAbiType(allocator: std.mem.Allocator, ty: sema.Type) ![]const u8 {
    return abi_layout.canonicalAbiTypeFromType(allocator, ty);
}

pub fn parseFixedBytesSpelling(name: []const u8) ?u8 {
    return abi_layout.parseFixedBytesSpelling(name);
}

pub fn fixedBytesAbiType(allocator: std.mem.Allocator, fixed_bytes: sema.FixedBytesType) ![]const u8 {
    if (fixed_bytes.spelling) |spelling| return allocator.dupe(u8, spelling);
    return std.fmt.allocPrint(allocator, "bytes{d}", .{fixed_bytes.len});
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
    return abi_layout.staticWordCountForType(ty);
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
