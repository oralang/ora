const std = @import("std");

const TypeCategory = @import("type_info.zig").TypeCategory;

pub const BuiltinTypeId = enum(u16) {
    u8,
    u16,
    u32,
    u64,
    u128,
    u160,
    u256,
    i8,
    i16,
    i32,
    i64,
    i128,
    i256,
    bool,
    address,
    string,
    bytes,
    void,
};

pub const BuiltinTypeSpec = struct {
    id: BuiltinTypeId,
    source_name: []const u8,
    category: TypeCategory,
    bit_width: ?u16 = null,
    byte_width: ?u16 = null,
    signed: ?bool = null,
    comptime_type_id: u32,
};

pub const primitive_comptime_type_id_min: u32 = 1;
pub const primitive_comptime_type_id_max: u32 = 18;

pub const fixed_bytes_min_len: u8 = 1;
pub const fixed_bytes_max_len: u8 = 32;
pub const fixed_bytes_prefix = "bytes";
pub const fixed_bytes_type_id_base: u32 = 500_000;
pub const fixed_bytes_type_id_min: u32 = fixed_bytes_type_id_base + fixed_bytes_min_len;
pub const fixed_bytes_type_id_max: u32 = fixed_bytes_type_id_base + fixed_bytes_max_len;

pub const abi_decode_error_type_id: u32 = 900_000;
pub const named_type_id_base: u32 = 1_000_000;

// Audited test-only sentinels from comptime heap/value/pool unit tests. Keep
// them out of reserved bands so tests cannot accidentally alias production IDs.
const known_test_sentinel_type_ids = [_]u32{ 42, 100 };

pub const builtin_types = [_]BuiltinTypeSpec{
    .{ .id = .u8, .source_name = "u8", .category = .Integer, .bit_width = 8, .byte_width = 1, .signed = false, .comptime_type_id = 1 },
    .{ .id = .u16, .source_name = "u16", .category = .Integer, .bit_width = 16, .byte_width = 2, .signed = false, .comptime_type_id = 2 },
    .{ .id = .u32, .source_name = "u32", .category = .Integer, .bit_width = 32, .byte_width = 4, .signed = false, .comptime_type_id = 3 },
    .{ .id = .u64, .source_name = "u64", .category = .Integer, .bit_width = 64, .byte_width = 8, .signed = false, .comptime_type_id = 4 },
    .{ .id = .u128, .source_name = "u128", .category = .Integer, .bit_width = 128, .byte_width = 16, .signed = false, .comptime_type_id = 5 },
    .{ .id = .u160, .source_name = "u160", .category = .Integer, .bit_width = 160, .byte_width = 20, .signed = false, .comptime_type_id = 18 },
    .{ .id = .u256, .source_name = "u256", .category = .Integer, .bit_width = 256, .byte_width = 32, .signed = false, .comptime_type_id = 6 },
    .{ .id = .i8, .source_name = "i8", .category = .Integer, .bit_width = 8, .byte_width = 1, .signed = true, .comptime_type_id = 7 },
    .{ .id = .i16, .source_name = "i16", .category = .Integer, .bit_width = 16, .byte_width = 2, .signed = true, .comptime_type_id = 8 },
    .{ .id = .i32, .source_name = "i32", .category = .Integer, .bit_width = 32, .byte_width = 4, .signed = true, .comptime_type_id = 9 },
    .{ .id = .i64, .source_name = "i64", .category = .Integer, .bit_width = 64, .byte_width = 8, .signed = true, .comptime_type_id = 10 },
    .{ .id = .i128, .source_name = "i128", .category = .Integer, .bit_width = 128, .byte_width = 16, .signed = true, .comptime_type_id = 11 },
    .{ .id = .i256, .source_name = "i256", .category = .Integer, .bit_width = 256, .byte_width = 32, .signed = true, .comptime_type_id = 12 },
    .{ .id = .bool, .source_name = "bool", .category = .Bool, .bit_width = 1, .byte_width = 1, .comptime_type_id = 13 },
    .{ .id = .address, .source_name = "address", .category = .Address, .bit_width = 160, .byte_width = 20, .comptime_type_id = 14 },
    .{ .id = .string, .source_name = "string", .category = .String, .comptime_type_id = 15 },
    .{ .id = .bytes, .source_name = "bytes", .category = .Bytes, .comptime_type_id = 16 },
    .{ .id = .void, .source_name = "void", .category = .Void, .byte_width = 0, .comptime_type_id = 17 },
};

const BuiltinNameEntry = struct { []const u8, BuiltinTypeId };

const builtin_name_entries = blk: {
    var entries: [builtin_types.len]BuiltinNameEntry = undefined;
    for (builtin_types, 0..) |spec, index| {
        entries[index] = .{ spec.source_name, spec.id };
    }
    break :blk entries;
};

const builtin_name_map = std.StaticStringMap(BuiltinTypeId).initComptime(builtin_name_entries);

const builtin_by_comptime_type_id = blk: {
    var specs = [_]?BuiltinTypeSpec{null} ** (primitive_comptime_type_id_max + 1);
    for (builtin_types) |spec| {
        specs[spec.comptime_type_id] = spec;
    }
    break :blk specs;
};

const supported_integer_type_names_buffer = blk: {
    var count: usize = 0;
    for (builtin_types) |spec| {
        if (spec.category == .Integer) count += 1;
    }

    var len: usize = 0;
    var seen: usize = 0;
    for (builtin_types) |spec| {
        if (spec.category != .Integer) continue;
        if (seen > 0) len += if (seen + 1 == count) ", and ".len else ", ".len;
        len += spec.source_name.len;
        seen += 1;
    }

    var text: [len]u8 = undefined;
    var offset: usize = 0;
    seen = 0;
    for (builtin_types) |spec| {
        if (spec.category != .Integer) continue;
        if (seen > 0) {
            const separator = if (seen + 1 == count) ", and " else ", ";
            @memcpy(text[offset..][0..separator.len], separator);
            offset += separator.len;
        }
        @memcpy(text[offset..][0..spec.source_name.len], spec.source_name);
        offset += spec.source_name.len;
        seen += 1;
    }
    break :blk text;
};

pub const supported_integer_type_names_text: []const u8 = supported_integer_type_names_buffer[0..];

pub const fixed_bytes_names = blk: {
    var names: [fixed_bytes_max_len][]const u8 = undefined;
    for (&names, 0..) |*name, index| {
        name.* = std.fmt.comptimePrint("bytes{d}", .{index + 1});
    }
    break :blk names;
};

comptime {
    assertBuiltinTable();
}

pub fn lookupBuiltinByName(name: []const u8) ?BuiltinTypeSpec {
    const id = builtin_name_map.get(name) orelse return null;
    return lookupBuiltinById(id);
}

pub fn lookupBuiltinById(id: BuiltinTypeId) BuiltinTypeSpec {
    return builtin_types[@intFromEnum(id)];
}

pub fn lookupBuiltinByComptimeTypeId(type_id: u32) ?BuiltinTypeSpec {
    if (type_id >= builtin_by_comptime_type_id.len) return null;
    return builtin_by_comptime_type_id[type_id];
}

pub fn builtinName(id: BuiltinTypeId) []const u8 {
    return lookupBuiltinById(id).source_name;
}

pub fn builtinByteWidth(id: BuiltinTypeId) ?u16 {
    return lookupBuiltinById(id).byte_width;
}

pub fn builtinBitWidth(id: BuiltinTypeId) ?u16 {
    return lookupBuiltinById(id).bit_width;
}

pub fn builtinSignedness(id: BuiltinTypeId) ?bool {
    return lookupBuiltinById(id).signed;
}

pub fn parseIntegerBuiltin(name: []const u8) ?BuiltinTypeSpec {
    const spec = lookupBuiltinByName(name) orelse return null;
    return if (spec.category == .Integer) spec else null;
}

/// Parses a fixed-bytes family name like "bytes20" into its length, or null.
///
/// The suffix must be a plain canonical decimal: no leading zero ("bytes01"),
/// no sign ("bytes+5"), no digit separators ("bytes1_6"). The legacy comptime
/// parser leaned on std.fmt.parseInt and silently accepted all of those, aliasing
/// e.g. "bytes01" to bytes1 — a latent bug. This is the canonical definition;
/// such spellings are not fixed-bytes type names.
pub fn parseFixedBytesName(name: []const u8) ?u8 {
    if (!std.mem.startsWith(u8, name, fixed_bytes_prefix)) return null;

    const suffix = name[fixed_bytes_prefix.len..];
    if (suffix.len == 0) return null;
    if (suffix[0] == '0') return null; // reject leading zero, including "bytes0"
    for (suffix) |ch| {
        if (ch < '0' or ch > '9') return null;
    }

    const len = std.fmt.parseInt(u8, suffix, 10) catch return null;
    return if (fixedBytesTypeId(len) != null) len else null;
}

pub fn fixedBytesTypeId(len: u8) ?u32 {
    if (len < fixed_bytes_min_len or len > fixed_bytes_max_len) return null;
    return fixed_bytes_type_id_base + len;
}

pub fn fixedBytesLenForTypeId(type_id: u32) ?u8 {
    if (type_id < fixed_bytes_type_id_min or type_id > fixed_bytes_type_id_max) return null;
    return @intCast(type_id - fixed_bytes_type_id_base);
}

pub fn fixedBytesName(len: u8) ?[]const u8 {
    if (len < fixed_bytes_min_len or len > fixed_bytes_max_len) return null;
    return fixed_bytes_names[len - fixed_bytes_min_len];
}

pub fn isBuiltinTypeName(name: []const u8) bool {
    return lookupBuiltinByName(name) != null;
}

fn assertBuiltinTable() void {
    if (builtin_types.len != @typeInfo(BuiltinTypeId).@"enum".fields.len) {
        @compileError("builtin_types must have one row per BuiltinTypeId");
    }
    if (fixed_bytes_names.len != fixed_bytes_max_len) {
        @compileError("fixed_bytes_names must cover bytes1 through bytes32");
    }

    for (builtin_types, 0..) |spec, index| {
        if (@intFromEnum(spec.id) != index) {
            @compileError("builtin_types order must match BuiltinTypeId order");
        }
        if (spec.comptime_type_id < primitive_comptime_type_id_min or spec.comptime_type_id > primitive_comptime_type_id_max) {
            @compileError("builtin comptime TypeId is outside the primitive band");
        }
        for (builtin_types[index + 1 ..]) |other| {
            if (spec.comptime_type_id == other.comptime_type_id) {
                @compileError("duplicate builtin comptime TypeId");
            }
            if (std.mem.eql(u8, spec.source_name, other.source_name)) {
                @compileError("duplicate builtin source name");
            }
        }
    }

    if (rangesOverlap(primitive_comptime_type_id_min, primitive_comptime_type_id_max, fixed_bytes_type_id_min, fixed_bytes_type_id_max)) {
        @compileError("primitive and fixed-bytes TypeId bands overlap");
    }
    if (rangesOverlap(fixed_bytes_type_id_min, fixed_bytes_type_id_max, abi_decode_error_type_id, abi_decode_error_type_id)) {
        @compileError("fixed-bytes and ABI decode error TypeId bands overlap");
    }
    if (rangesOverlap(fixed_bytes_type_id_min, fixed_bytes_type_id_max, named_type_id_base, std.math.maxInt(u32))) {
        @compileError("fixed-bytes and named TypeId bands overlap");
    }
    if (abi_decode_error_type_id >= named_type_id_base) {
        @compileError("ABI decode error TypeId must remain below named TypeId band");
    }
    for (known_test_sentinel_type_ids) |sentinel_type_id| {
        if (lookupBuiltinByComptimeTypeId(sentinel_type_id) != null or
            fixedBytesLenForTypeId(sentinel_type_id) != null or
            sentinel_type_id == abi_decode_error_type_id or
            sentinel_type_id >= named_type_id_base)
        {
            @compileError("known test-only TypeId collides with a reserved band");
        }
    }
}

fn rangesOverlap(a_min: u32, a_max: u32, b_min: u32, b_max: u32) bool {
    return a_min <= b_max and b_min <= a_max;
}

test "builtin lookup by source name and frozen comptime id" {
    try std.testing.expectEqual(@as(usize, 18), builtin_types.len);

    const u8_spec = lookupBuiltinByName("u8") orelse return error.TestUnexpectedResult;
    try std.testing.expectEqual(BuiltinTypeId.u8, u8_spec.id);
    try std.testing.expectEqual(@as(u32, 1), u8_spec.comptime_type_id);
    try std.testing.expectEqual(@as(?u16, 8), u8_spec.bit_width);
    try std.testing.expectEqual(@as(?u16, 1), u8_spec.byte_width);
    try std.testing.expectEqual(@as(?bool, false), u8_spec.signed);

    const void_spec = lookupBuiltinByComptimeTypeId(17) orelse return error.TestUnexpectedResult;
    try std.testing.expectEqual(BuiltinTypeId.void, void_spec.id);
    try std.testing.expectEqualStrings("void", void_spec.source_name);

    try std.testing.expect(lookupBuiltinByName("bytes32") == null);
    const u160_spec = lookupBuiltinByComptimeTypeId(18) orelse return error.TestUnexpectedResult;
    try std.testing.expectEqual(BuiltinTypeId.u160, u160_spec.id);
    try std.testing.expectEqualStrings("u160", u160_spec.source_name);
    try std.testing.expect(lookupBuiltinByComptimeTypeId(19) == null);
}

test "integer builtin metadata is table-driven" {
    const u256_spec = parseIntegerBuiltin("u256") orelse return error.TestUnexpectedResult;
    try std.testing.expectEqual(BuiltinTypeId.u256, u256_spec.id);
    try std.testing.expectEqual(@as(?u16, 256), u256_spec.bit_width);
    try std.testing.expectEqual(@as(?u16, 32), u256_spec.byte_width);
    try std.testing.expectEqual(@as(?bool, false), u256_spec.signed);

    const u160_spec = parseIntegerBuiltin("u160") orelse return error.TestUnexpectedResult;
    try std.testing.expectEqual(BuiltinTypeId.u160, u160_spec.id);
    try std.testing.expectEqual(@as(?u16, 160), u160_spec.bit_width);
    try std.testing.expectEqual(@as(?u16, 20), u160_spec.byte_width);
    try std.testing.expectEqual(@as(?bool, false), u160_spec.signed);

    const i128_spec = parseIntegerBuiltin("i128") orelse return error.TestUnexpectedResult;
    try std.testing.expectEqual(BuiltinTypeId.i128, i128_spec.id);
    try std.testing.expectEqual(@as(?u16, 128), i128_spec.bit_width);
    try std.testing.expectEqual(@as(?bool, true), i128_spec.signed);

    try std.testing.expect(parseIntegerBuiltin("bool") == null);
    try std.testing.expect(parseIntegerBuiltin("bytes32") == null);
    try std.testing.expect(parseIntegerBuiltin("u24") == null);
    try std.testing.expect(parseIntegerBuiltin("i96") == null);
}

test "supported integer type names text is table-driven" {
    try std.testing.expectEqualStrings("u8, u16, u32, u64, u128, u160, u256, i8, i16, i32, i64, i128, and i256", supported_integer_type_names_text);
}

test "fixed bytes names and TypeIds use frozen band" {
    try std.testing.expectEqual(@as(?u8, 1), parseFixedBytesName("bytes1"));
    try std.testing.expectEqual(@as(?u8, 32), parseFixedBytesName("bytes32"));
    try std.testing.expectEqual(@as(?u32, 500_001), fixedBytesTypeId(1));
    try std.testing.expectEqual(@as(?u32, 500_032), fixedBytesTypeId(32));
    try std.testing.expectEqual(@as(?u8, 1), fixedBytesLenForTypeId(500_001));
    try std.testing.expectEqual(@as(?u8, 32), fixedBytesLenForTypeId(500_032));
    try std.testing.expectEqualStrings("bytes20", fixedBytesName(20) orelse return error.TestUnexpectedResult);
}

test "fixed bytes parsing rejects non-family names" {
    try std.testing.expect(parseFixedBytesName("bytes") == null);
    try std.testing.expect(parseFixedBytesName("bytes0") == null);
    try std.testing.expect(parseFixedBytesName("bytes33") == null);
    try std.testing.expect(parseFixedBytesName("bytes01") == null);
    try std.testing.expect(parseFixedBytesName("bytes+5") == null);
    try std.testing.expect(parseFixedBytesName("bytes1_6") == null);
    try std.testing.expect(parseFixedBytesName("bytesx") == null);
    try std.testing.expect(parseFixedBytesName("u256") == null);
    try std.testing.expect(fixedBytesTypeId(0) == null);
    try std.testing.expect(fixedBytesTypeId(33) == null);
    try std.testing.expect(fixedBytesLenForTypeId(500_000) == null);
    try std.testing.expect(fixedBytesLenForTypeId(500_033) == null);
}
