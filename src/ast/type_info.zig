const std = @import("std");
const SourceSpan = @import("../ast.zig").SourceSpan;

/// Unified type information system for all AST nodes
/// This replaces the various type enums and provides consistent type representation
pub const TypeInfo = struct {
    category: TypeCategory,
    ora_type: ?OraType,
    source: TypeSource,
    span: ?SourceSpan, // Where the type was determined/declared

    /// Create unknown type info (used during parsing)
    pub fn unknown() TypeInfo {
        return TypeInfo{
            .category = .Unknown,
            .ora_type = null,
            .source = .unknown,
            .span = null,
        };
    }

    /// Create explicit type info (user declared)
    pub fn explicit(category: TypeCategory, ora_type: OraType, span: SourceSpan) TypeInfo {
        return TypeInfo{
            .category = category,
            .ora_type = ora_type,
            .source = .explicit,
            .span = span,
        };
    }

    /// Create inferred type info (from context)
    pub fn inferred(category: TypeCategory, ora_type: OraType, span: ?SourceSpan) TypeInfo {
        return TypeInfo{
            .category = category,
            .ora_type = ora_type,
            .source = .inferred,
            .span = span,
        };
    }

    /// Check if type is fully resolved
    pub fn isResolved(self: TypeInfo) bool {
        return self.ora_type != null and self.category != .Unknown;
    }

    /// Ensure that a type is represented as TypeInfo
    /// This helps with the transition from various type representations to TypeInfo
    pub fn ensureTypeInfo(type_value: anytype) TypeInfo {
        const T = @TypeOf(type_value);
        if (T == TypeInfo) {
            return type_value;
        } else if (T == OraType or T == *const OraType) {
            return fromOraType(type_value);
        } else {
            return unknown(); // Default to unknown type if can't convert
        }
    }

    /// Check if type is compatible with another type
    pub fn isCompatibleWith(self: TypeInfo, other: TypeInfo) bool {
        if (!self.isResolved() or !other.isResolved()) return false;

        // Same types are always compatible
        if (std.meta.eql(self.ora_type.?, other.ora_type.?)) return true;

        // Integer types can be compatible based on size and signedness
        return switch (self.ora_type.?) {
            .u8, .u16, .u32, .u64, .u128, .u256 => switch (other.ora_type.?) {
                .u8, .u16, .u32, .u64, .u128, .u256 => true, // Unsigned integers
                else => false,
            },
            .i8, .i16, .i32, .i64, .i128, .i256 => switch (other.ora_type.?) {
                .i8, .i16, .i32, .i64, .i128, .i256 => true, // Signed integers
                else => false,
            },
            else => false,
        };
    }

    /// Get the size in bits for integer types
    pub fn getBitSize(self: TypeInfo) ?u32 {
        return switch (self.ora_type orelse return null) {
            .u8, .i8 => 8,
            .u16, .i16 => 16,
            .u32, .i32 => 32,
            .u64, .i64 => 64,
            .u128, .i128 => 128,
            .u256, .i256 => 256,
            .bool => 1,
            else => null,
        };
    }

    /// Create TypeInfo from OraType
    pub fn fromOraType(ora_type: OraType) TypeInfo {
        const category = ora_type.getCategory();
        return TypeInfo{
            .category = category,
            .ora_type = ora_type,
            .source = .inferred,
            .span = null,
        };
    }

    pub fn equals(a: TypeInfo, b: TypeInfo) bool {
        const ao = a.ora_type;
        const bo = b.ora_type;
        // If both have concrete OraTypes, compare structurally regardless of category
        if (ao != null and bo != null) {
            return OraType.equals(ao.?, bo.?);
        }
        // Fallback to category comparison when OraType is missing
        if (a.category != b.category) return false;
        if ((ao == null) != (bo == null)) return false;
        if (ao == null) return true;
        return OraType.equals(ao.?, bo.?);
    }

    pub fn hash(self: TypeInfo) u64 {
        var h = std.hash.Wyhash.init(0);
        var cat: u64 = @intCast(@intFromEnum(self.category));
        h.update(std.mem.asBytes(&cat));
        if (self.ora_type) |ot| {
            const sub = OraType.hash(ot);
            h.update(std.mem.asBytes(&sub));
        }
        return h.final();
    }
};

/// Generic type categories for high-level classification
pub const TypeCategory = enum {
    // Primitive categories
    Integer,
    String,
    Bool,
    Address,
    Hex,
    Bytes,

    // Complex type categories
    Struct,
    Enum,
    Contract, // Contract type category
    Function,
    Array,
    Slice,
    Mapping,
    DoubleMap,
    Tuple,
    ErrorUnion,
    Result,
    Union,

    // Special categories
    Void,
    Error,
    Module,
    Unknown,

    pub fn format(self: TypeCategory, comptime fmt: []const u8, options: std.fmt.FormatOptions, writer: anytype) !void {
        _ = fmt;
        _ = options;
        try writer.writeAll(@tagName(self));
    }
};

/// Specific Ora language types
pub const OraType = union(enum) {
    // Unsigned integer types
    u8: void,
    u16: void,
    u32: void,
    u64: void,
    u128: void,
    u256: void,

    // Signed integer types
    i8: void,
    i16: void,
    i32: void,
    i64: void,
    i128: void,
    i256: void,

    // Other primitive types
    bool: void,
    string: void,
    address: void,
    bytes: void,
    void: void,

    // Complex types with additional data
    struct_type: []const u8, // Struct name
    enum_type: []const u8, // Enum name
    contract_type: []const u8, // Contract name
    array: struct { elem: *const OraType, len: u64 }, // Fixed-size array [T; N]
    slice: *const OraType, // Element type
    mapping: MappingType, // Key and value types
    double_map: DoubleMapType, // Two keys and value type
    tuple: []const OraType, // Element types
    function: FunctionType, // Parameter and return types
    error_union: *const OraType, // Success type (!T)
    _union: []const OraType, // Union of types T1 | T2 | ... (underscore to avoid Zig keyword conflict)
    anonymous_struct: []const AnonymousStructFieldType, // struct { field: T, ... }
    module: ?[]const u8, // Optional module name

    /// Get the category for this Ora type
    pub fn getCategory(self: OraType) TypeCategory {
        return switch (self) {
            .u8, .u16, .u32, .u64, .u128, .u256, .i8, .i16, .i32, .i64, .i128, .i256 => .Integer,
            .bool => .Bool,
            .string => .String,
            .address => .Address,
            .bytes => .Bytes,
            .void => .Void,
            .struct_type => .Struct,
            .enum_type => .Enum,
            .contract_type => .Contract,
            .array => .Array,
            .slice => .Slice,
            .mapping => .Mapping,
            .double_map => .DoubleMap,
            .tuple => .Tuple,
            .function => .Function,
            .error_union => .Error,
            ._union => .Union,
            .anonymous_struct => .Struct,
            .module => .Module,
        };
    }

    /// Check if this is an integer type
    pub fn isInteger(self: OraType) bool {
        return switch (self) {
            .u8, .u16, .u32, .u64, .u128, .u256, .i8, .i16, .i32, .i64, .i128, .i256 => true,
            else => false,
        };
    }

    /// Check if this is an unsigned integer type
    pub fn isUnsignedInteger(self: OraType) bool {
        return switch (self) {
            .u8, .u16, .u32, .u64, .u128, .u256 => true,
            else => false,
        };
    }

    /// Check if this is a signed integer type
    pub fn isSignedInteger(self: OraType) bool {
        return switch (self) {
            .i8, .i16, .i32, .i64, .i128, .i256 => true,
            else => false,
        };
    }

    /// Get string representation for serialization
    pub fn toString(self: OraType) []const u8 {
        return switch (self) {
            .u8 => "u8",
            .u16 => "u16",
            .u32 => "u32",
            .u64 => "u64",
            .u128 => "u128",
            .u256 => "u256",
            .i8 => "i8",
            .i16 => "i16",
            .i32 => "i32",
            .i64 => "i64",
            .i128 => "i128",
            .i256 => "i256",
            .bool => "bool",
            .string => "string",
            .address => "address",
            .bytes => "bytes",
            .void => "void",
            .struct_type => |name| name,
            .enum_type => |name| name,
            .contract_type => |name| name,
            .array => "array",
            .slice => "slice",
            .mapping => "mapping",
            .double_map => "double_map",
            .tuple => "tuple",
            .function => "function",
            .error_union => "error_union",
            ._union => "union",
            .anonymous_struct => "struct",
            .module => "module",
        };
    }

    pub fn format(self: OraType, comptime fmt: []const u8, options: std.fmt.FormatOptions, writer: anytype) !void {
        _ = fmt;
        _ = options;
        try writer.writeAll(self.toString());
    }

    pub fn equals(a: OraType, b: OraType) bool {
        if (std.meta.activeTag(a) != std.meta.activeTag(b)) return false;
        return switch (a) {
            .u8, .u16, .u32, .u64, .u128, .u256, .i8, .i16, .i32, .i64, .i128, .i256, .bool, .string, .address, .bytes, .void => true,
            .struct_type => |an| switch (b) {
                .struct_type => |bn| std.mem.eql(u8, an, bn),
                else => unreachable,
            },
            .enum_type => |an| switch (b) {
                .enum_type => |bn| std.mem.eql(u8, an, bn),
                else => unreachable,
            },
            .contract_type => |an| switch (b) {
                .contract_type => |bn| std.mem.eql(u8, an, bn),
                else => unreachable,
            },
            .array => |aa| switch (b) {
                .array => |bb| equals(@constCast(aa.elem).*, @constCast(bb.elem).*) and (aa.len == bb.len),
                else => unreachable,
            },
            .slice => |ap| switch (b) {
                .slice => |bp| equals(@constCast(ap).*, @constCast(bp).*),
                else => unreachable,
            },
            .mapping => |am| switch (b) {
                .mapping => |bm| equals(@constCast(am.key).*, @constCast(bm.key).*) and equals(@constCast(am.value).*, @constCast(bm.value).*),
                else => unreachable,
            },
            .double_map => |am| switch (b) {
                .double_map => |bm| equals(@constCast(am.key1).*, @constCast(bm.key1).*) and
                    equals(@constCast(am.key2).*, @constCast(bm.key2).*) and
                    equals(@constCast(am.value).*, @constCast(bm.value).*),
                else => unreachable,
            },
            .tuple => |ats| switch (b) {
                .tuple => |bts| blk: {
                    if (ats.len != bts.len) break :blk false;
                    var i: usize = 0;
                    while (i < ats.len) : (i += 1) {
                        if (!equals(@constCast(&ats[i]).*, @constCast(&bts[i]).*)) break :blk false;
                    }
                    break :blk true;
                },
                else => unreachable,
            },
            .function => |af| switch (b) {
                .function => |bf| blk: {
                    if (af.params.len != bf.params.len) break :blk false;
                    var i: usize = 0;
                    while (i < af.params.len) : (i += 1) {
                        if (!equals(@constCast(&af.params[i]).*, @constCast(&bf.params[i]).*)) break :blk false;
                    }
                    const ar = if (af.return_type) |r| @constCast(r).* else OraType{ .void = {} };
                    const br = if (bf.return_type) |r| @constCast(r).* else OraType{ .void = {} };
                    break :blk equals(ar, br);
                },
                else => unreachable,
            },
            .error_union => |ap| switch (b) {
                .error_union => |bp| equals(@constCast(ap).*, @constCast(bp).*),
                else => unreachable,
            },
            ._union => |as| switch (b) {
                ._union => |bs| blk: {
                    if (as.len != bs.len) break :blk false;
                    var i: usize = 0;
                    while (i < as.len) : (i += 1) {
                        if (!equals(@constCast(&as[i]).*, @constCast(&bs[i]).*)) break :blk false;
                    }
                    break :blk true;
                },
                else => unreachable,
            },
            .anonymous_struct => |af| switch (b) {
                .anonymous_struct => |bf| blk: {
                    if (af.len != bf.len) break :blk false;
                    var i: usize = 0;
                    while (i < af.len) : (i += 1) {
                        if (!std.mem.eql(u8, af[i].name, bf[i].name)) break :blk false;
                        if (!equals(@constCast(af[i].typ).*, @constCast(bf[i].typ).*)) break :blk false;
                    }
                    break :blk true;
                },
                else => unreachable,
            },
            .module => |am| switch (b) {
                .module => |bm| blk: {
                    if ((am == null) != (bm == null)) break :blk false;
                    if (am == null) break :blk true;
                    break :blk std.mem.eql(u8, am.?, bm.?);
                },
                else => unreachable,
            },
        };
    }

    pub fn hash(self: OraType) u64 {
        var h = std.hash.Wyhash.init(0);
        const tag_val: u64 = @intCast(@intFromEnum(std.meta.activeTag(self)));
        h.update(std.mem.asBytes(&tag_val));
        switch (self) {
            .u8, .u16, .u32, .u64, .u128, .u256, .i8, .i16, .i32, .i64, .i128, .i256, .bool, .string, .address, .bytes, .void => {},
            .struct_type => |name| h.update(name),
            .enum_type => |name| h.update(name),
            .contract_type => |name| h.update(name),
            .array => |arr| {
                const sub = OraType.hash(@constCast(arr.elem).*);
                h.update(std.mem.asBytes(&sub));
                h.update(std.mem.asBytes(&arr.len));
            },
            .slice => |elem| {
                const sub = OraType.hash(@constCast(elem).*);
                h.update(std.mem.asBytes(&sub));
            },
            .mapping => |m| {
                const k = OraType.hash(@constCast(m.key).*);
                const v = OraType.hash(@constCast(m.value).*);
                h.update(std.mem.asBytes(&k));
                h.update(std.mem.asBytes(&v));
            },
            .double_map => |dm| {
                const k1 = OraType.hash(@constCast(dm.key1).*);
                const k2 = OraType.hash(@constCast(dm.key2).*);
                const v = OraType.hash(@constCast(dm.value).*);
                h.update(std.mem.asBytes(&k1));
                h.update(std.mem.asBytes(&k2));
                h.update(std.mem.asBytes(&v));
            },
            .tuple => |ts| {
                for (ts) |t| {
                    const sub = OraType.hash(@constCast(&t).*);
                    h.update(std.mem.asBytes(&sub));
                }
            },
            .function => |f| {
                for (f.params) |p| {
                    const sub = OraType.hash(@constCast(&p).*);
                    h.update(std.mem.asBytes(&sub));
                }
                const ret = if (f.return_type) |r| OraType.hash(@constCast(r).*) else 0;
                h.update(std.mem.asBytes(&ret));
            },
            .error_union => |t| {
                const sub = OraType.hash(@constCast(t).*);
                h.update(std.mem.asBytes(&sub));
            },
            ._union => |us| {
                for (us) |u| {
                    const sub = OraType.hash(@constCast(&u).*);
                    h.update(std.mem.asBytes(&sub));
                }
            },
            .anonymous_struct => |fs| {
                for (fs) |f| {
                    h.update(f.name);
                    const sub = OraType.hash(@constCast(f.typ).*);
                    h.update(std.mem.asBytes(&sub));
                }
            },
            .module => |m| {
                if (m) |name| h.update(name);
            },
        }
        return h.final();
    }

    /// Render this type as source-like syntax into writer (structural, not pointer-based)
    pub fn render(self: OraType, writer: anytype) !void {
        switch (self) {
            .u8 => try writer.writeAll("u8"),
            .u16 => try writer.writeAll("u16"),
            .u32 => try writer.writeAll("u32"),
            .u64 => try writer.writeAll("u64"),
            .u128 => try writer.writeAll("u128"),
            .u256 => try writer.writeAll("u256"),
            .i8 => try writer.writeAll("i8"),
            .i16 => try writer.writeAll("i16"),
            .i32 => try writer.writeAll("i32"),
            .i64 => try writer.writeAll("i64"),
            .i128 => try writer.writeAll("i128"),
            .i256 => try writer.writeAll("i256"),
            .bool => try writer.writeAll("bool"),
            .string => try writer.writeAll("string"),
            .address => try writer.writeAll("address"),
            .bytes => try writer.writeAll("bytes"),
            .void => try writer.writeAll("void"),
            .struct_type => |name| try writer.writeAll(name),
            .enum_type => |name| try writer.writeAll(name),
            .contract_type => |name| try writer.writeAll(name),
            .array => |arr| {
                try writer.writeByte('[');
                try (@constCast(arr.elem).*).render(writer);
                try writer.writeAll("; ");
                var buf: [32]u8 = undefined;
                const s = std.fmt.bufPrint(&buf, "{d}", .{arr.len}) catch "?";
                try writer.writeAll(s);
                try writer.writeByte(']');
            },
            .slice => |elem| {
                try writer.writeAll("slice[");
                try (@constCast(elem).*).render(writer);
                try writer.writeByte(']');
            },
            .mapping => |m| {
                try writer.writeAll("map[");
                try (@constCast(m.key).*).render(writer);
                try writer.writeAll(", ");
                try (@constCast(m.value).*).render(writer);
                try writer.writeByte(']');
            },
            .double_map => |dm| {
                try writer.writeAll("doublemap[");
                try (@constCast(dm.key1).*).render(writer);
                try writer.writeAll(", ");
                try (@constCast(dm.key2).*).render(writer);
                try writer.writeAll(", ");
                try (@constCast(dm.value).*).render(writer);
                try writer.writeByte(']');
            },
            .tuple => |elems| {
                try writer.writeByte('(');
                var first = true;
                for (elems) |e| {
                    if (!first) try writer.writeAll(", ");
                    first = false;
                    try (@constCast(&e).*).render(writer);
                }
                try writer.writeByte(')');
            },
            .function => |f| {
                try writer.writeAll("fn(");
                var first = true;
                for (f.params) |p| {
                    if (!first) try writer.writeAll(", ");
                    first = false;
                    try (@constCast(&p).*).render(writer);
                }
                try writer.writeByte(')');
                if (f.return_type) |ret| {
                    try writer.writeAll(" -> ");
                    try (@constCast(ret).*).render(writer);
                }
            },
            .error_union => |succ| {
                try writer.writeByte('!');
                try (@constCast(succ).*).render(writer);
            },
            ._union => |members| {
                var first = true;
                for (members) |m| {
                    if (!first) try writer.writeAll(" | ");
                    first = false;
                    try (@constCast(&m).*).render(writer);
                }
            },
            .anonymous_struct => |fields| {
                try writer.writeAll("struct {");
                if (fields.len > 0) try writer.writeByte(' ');
                var first = true;
                for (fields) |f| {
                    if (!first) try writer.writeAll(", ");
                    first = false;
                    try writer.writeAll(f.name);
                    try writer.writeAll(": ");
                    try (@constCast(f.typ).*).render(writer);
                }
                if (fields.len > 0) try writer.writeByte(' ');
                try writer.writeByte('}');
            },
            .module => |_| try writer.writeAll("module"),
        }
    }
};

/// Complex type definitions
pub const MappingType = struct {
    key: *const OraType,
    value: *const OraType,
};

pub const DoubleMapType = struct {
    key1: *const OraType,
    key2: *const OraType,
    value: *const OraType,
};

pub const FunctionType = struct {
    params: []const OraType,
    return_type: ?*const OraType,
};

pub const ResultType = struct {
    ok_type: *const OraType,
    error_type: *const OraType,
};

pub const AnonymousStructFieldType = struct {
    name: []const u8,
    typ: *const OraType,
};

/// Source of type information
pub const TypeSource = enum {
    explicit, // User explicitly declared the type
    inferred, // Inferred from context (e.g., enum underlying type)
    default, // Language default (e.g., integer literals default to u256)
    unknown, // Not yet determined

    pub fn format(self: TypeSource, comptime fmt: []const u8, options: std.fmt.FormatOptions, writer: anytype) !void {
        _ = fmt;
        _ = options;
        try writer.writeAll(@tagName(self));
    }
};

/// Helper functions for creating common types
pub const CommonTypes = struct {
    pub fn u8_type() TypeInfo {
        return TypeInfo{
            .category = .Integer,
            .ora_type = .u8,
            .source = .explicit,
            .span = null,
        };
    }

    pub fn u16_type() TypeInfo {
        return TypeInfo{
            .category = .Integer,
            .ora_type = .u16,
            .source = .explicit,
            .span = null,
        };
    }

    pub fn u32_type() TypeInfo {
        return TypeInfo{
            .category = .Integer,
            .ora_type = .u32,
            .source = .explicit,
            .span = null,
        };
    }

    pub fn u64_type() TypeInfo {
        return TypeInfo{
            .category = .Integer,
            .ora_type = .u64,
            .source = .explicit,
            .span = null,
        };
    }

    pub fn u128_type() TypeInfo {
        return TypeInfo{
            .category = .Integer,
            .ora_type = .u128,
            .source = .explicit,
            .span = null,
        };
    }

    pub fn u256_type() TypeInfo {
        return TypeInfo{
            .category = .Integer,
            .ora_type = .u256,
            .source = .explicit,
            .span = null,
        };
    }

    pub fn bool_type() TypeInfo {
        return TypeInfo{
            .category = .Bool,
            .ora_type = .bool,
            .source = .explicit,
            .span = null,
        };
    }

    pub fn string_type() TypeInfo {
        return TypeInfo{
            .category = .String,
            .ora_type = .string,
            .source = .explicit,
            .span = null,
        };
    }

    pub fn address_type() TypeInfo {
        return TypeInfo{
            .category = .Address,
            .ora_type = .address,
            .source = .explicit,
            .span = null,
        };
    }

    pub fn void_type() TypeInfo {
        return TypeInfo{
            .category = .Void,
            .ora_type = .void,
            .source = .explicit,
            .span = null,
        };
    }

    pub fn contract_type(name: []const u8) TypeInfo {
        return TypeInfo{
            .category = .Contract,
            .ora_type = OraType{ .contract_type = name },
            .ast_type = null,
            .source = .explicit,
            .span = null,
        };
    }

    pub fn unknown_integer() TypeInfo {
        return TypeInfo{
            .category = .Integer,
            .ora_type = null,
            .source = .unknown,
            .span = null,
        };
    }

    pub fn unknown_string() TypeInfo {
        return TypeInfo{
            .category = .String,
            .ora_type = null,
            .source = .unknown,
            .span = null,
        };
    }
};

/// Memory management for complex types
pub fn deinitTypeInfo(allocator: std.mem.Allocator, type_info: *TypeInfo) void {
    if (type_info.ora_type) |*ora_type| {
        switch (ora_type.*) {
            .array => |arr| {
                deinitOraType(allocator, @constCast(arr.elem));
                allocator.destroy(arr.elem);
            },
            .slice, .error_union => |ptr| {
                deinitOraType(allocator, @constCast(ptr));
                allocator.destroy(ptr);
            },
            ._union => |types| {
                // Free each union member type (by value in slice)
                for (types) |member| {
                    deinitOraType(allocator, @constCast(&member));
                }
                allocator.free(types);
            },
            .anonymous_struct => |fields| {
                // Free each field's allocated type pointer
                for (fields) |field| {
                    deinitOraType(allocator, @constCast(field.typ));
                    allocator.destroy(field.typ);
                }
                allocator.free(fields);
            },
            .mapping => |mapping| {
                // Properly handle mapping's key and value
                deinitOraType(allocator, @constCast(mapping.key));
                deinitOraType(allocator, @constCast(mapping.value));
                allocator.destroy(mapping.key);
                allocator.destroy(mapping.value);
            },
            .double_map => |double_map| {
                // Properly handle double_map's keys and value
                deinitOraType(allocator, @constCast(double_map.key1));
                deinitOraType(allocator, @constCast(double_map.key2));
                deinitOraType(allocator, @constCast(double_map.value));
                allocator.destroy(double_map.key1);
                allocator.destroy(double_map.key2);
                allocator.destroy(double_map.value);
            },
            .tuple => |types| {
                // Handle tuple elements
                for (types) |element_type| {
                    deinitOraType(allocator, @constCast(&element_type));
                }
                allocator.free(types);
            },
            .function => |func| {
                // Handle function parameters
                for (func.params) |param| {
                    deinitOraType(allocator, @constCast(&param));
                }
                allocator.free(func.params);

                // Handle optional return type safely
                if (func.return_type) |ret_type| {
                    deinitOraType(allocator, @constCast(ret_type));
                    allocator.destroy(ret_type);
                }
            },
            else => {
                // Primitive types don't need cleanup
            },
        }
    }
}

fn deinitOraType(allocator: std.mem.Allocator, ora_type: *OraType) void {
    // Handle nested type cleanup based on OraType variant
    switch (ora_type.*) {
        .array => |arr| {
            deinitOraType(allocator, @constCast(arr.elem));
            allocator.destroy(arr.elem);
        },
        .slice => |elem_type_ptr| {
            deinitOraType(allocator, @constCast(elem_type_ptr));
            allocator.destroy(elem_type_ptr);
        },
        ._union => |types| {
            for (types) |member| {
                deinitOraType(allocator, @constCast(&member));
            }
            allocator.free(types);
        },
        .anonymous_struct => |fields| {
            for (fields) |field| {
                deinitOraType(allocator, @constCast(field.typ));
                allocator.destroy(field.typ);
            }
            allocator.free(fields);
        },
        .mapping => |mapping| {
            // MappingType's key and value are defined as *const OraType (not optional)
            deinitOraType(allocator, @constCast(mapping.key));
            allocator.destroy(mapping.key);

            deinitOraType(allocator, @constCast(mapping.value));
            allocator.destroy(mapping.value);
        },
        .double_map => |double_map| {
            // DoubleMapType's keys and value are defined as *const OraType (not optional)
            deinitOraType(allocator, @constCast(double_map.key1));
            allocator.destroy(double_map.key1);

            deinitOraType(allocator, @constCast(double_map.key2));
            allocator.destroy(double_map.key2);

            deinitOraType(allocator, @constCast(double_map.value));
            allocator.destroy(double_map.value);
        },
        .function => |function| {
            // FunctionType has params as []const OraType and return_type as ?*const OraType
            // Only return_type is optional and needs null check
            if (function.return_type) |return_type| {
                deinitOraType(allocator, @constCast(return_type));
                allocator.destroy(return_type);
            }

            // We don't need to deallocate params here as they're handled by the caller
        },
        // String references in structs, enums, and contracts are typically owned by the parser
        .struct_type, .enum_type, .contract_type => {
            // No additional cleanup for these string references
            // They're typically owned by the parser, not by OraType
        },

        // Other primitive types don't need cleanup
        else => {},
    }
}
