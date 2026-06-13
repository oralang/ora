const std = @import("std");
const builtin = @import("ora_types").builtin;

const builtin_abi_names = blk: {
    var names: [builtin.builtin_types.len][]const u8 = undefined;
    for (builtin.builtin_types) |spec| {
        names[@intFromEnum(spec.id)] = comptimeAbiName(spec);
    }
    break :blk names;
};

fn comptimeAbiName(comptime spec: builtin.BuiltinTypeSpec) []const u8 {
    return switch (spec.category) {
        .Integer => std.fmt.comptimePrint("{s}{d}", .{
            if (spec.signed orelse @compileError("integer builtin missing signedness")) "int" else "uint",
            spec.bit_width orelse @compileError("integer builtin missing bit width"),
        }),
        .Bool => "bool",
        .Address => "address",
        .String => "string",
        .Bytes => "bytes",
        .Void => "void",
        else => @compileError("unsupported builtin category for ABI name"),
    };
}

pub fn builtinAbiName(id: builtin.BuiltinTypeId) []const u8 {
    return builtin_abi_names[@intFromEnum(id)];
}

pub fn builtinSpecAbiName(spec: builtin.BuiltinTypeSpec) []const u8 {
    return builtinAbiName(spec.id);
}

pub fn fixedBytesAbiName(len: u8) ?[]const u8 {
    return builtin.fixedBytesName(len);
}

pub fn integerAbiName(signed: bool, bits: u16) ?[]const u8 {
    const spec = builtin.lookupIntegerBuiltin(signed, bits) orelse return null;
    return builtinSpecAbiName(spec);
}

pub fn publicTupleReturnAbiName() []const u8 {
    return "tuple";
}

pub fn publicStructReturnAbiName() []const u8 {
    return "struct";
}

test "ABI builtin names are derived from builtin type metadata" {
    try std.testing.expectEqualStrings("uint8", builtinAbiName(.u8));
    try std.testing.expectEqualStrings("uint160", builtinAbiName(.u160));
    try std.testing.expectEqualStrings("int256", builtinAbiName(.i256));
    try std.testing.expectEqualStrings("address", builtinAbiName(.address));
    try std.testing.expectEqualStrings("void", builtinAbiName(.void));
    try std.testing.expectEqualStrings("bytes20", fixedBytesAbiName(20) orelse return error.TestUnexpectedResult);
    try std.testing.expectEqualStrings("uint256", integerAbiName(false, 256) orelse return error.TestUnexpectedResult);
    try std.testing.expectEqualStrings("int128", integerAbiName(true, 128) orelse return error.TestUnexpectedResult);
}
