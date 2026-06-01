// ============================================================================
// Type Information System
// ============================================================================
//
// Unified type system providing comprehensive type representation,
// validation, and operations for all AST nodes.
//
// KEY CONCEPTS:
//   • TypeInfo: Unified type representation (category + ora_type + source)
//   • TypeCategory: High-level classification (Integer, String, Array, etc.)
//   • OraType: Specific type details (u8, u256, arrays, maps, 80+ variants)
//   • TypeSource: How type was determined (explicit, inferred, unknown)
//
// SECTIONS:
//   • TypeInfo core & common types
//   • TypeCategory & OraType definitions
//   • Type rendering & display
//
// ============================================================================

const std = @import("std");
const refinements = @import("ora_refinements");
const SourceSpan = @import("source_span.zig").SourceSpan;
const MemoryRegion = @import("region.zig").MemoryRegion;

/// Unified type information system for all AST nodes
/// This replaces the various type enums and provides consistent type representation
pub const TypeInfo = struct {
    category: TypeCategory,
    ora_type: ?OraType,
    source: TypeSource,
    span: ?SourceSpan, // Where the type was determined/declared
    region: ?MemoryRegion = null, // Located type region (if known)
    generic_type_args: ?[]const OraType = null, // For generic instantiations like Pair(u256)

    /// Create unknown type info (used during parsing)
    pub fn unknown() TypeInfo {
        return TypeInfo{
            .category = .Unknown,
            .ora_type = null,
            .source = .unknown,
            .span = null,
            .region = null,
        };
    }

    /// Create explicit type info (user declared)
    pub fn explicit(category: TypeCategory, ora_type: OraType, span: SourceSpan) TypeInfo {
        return TypeInfo{
            .category = category,
            .ora_type = ora_type,
            .source = .explicit,
            .span = span,
            .region = null,
        };
    }

    /// Create inferred type info (from context)
    pub fn inferred(category: TypeCategory, ora_type: OraType, span: ?SourceSpan) TypeInfo {
        return TypeInfo{
            .category = category,
            .ora_type = ora_type,
            .source = .inferred,
            .span = span,
            .region = null,
        };
    }

    /// Check if type is fully resolved
    pub fn isResolved(self: TypeInfo) bool {
        if (self.category == .Error) return true;
        if (self.ora_type) |ot| {
            // type_parameter is a valid placeholder in generic signatures
            if (ot == .type_parameter) return true;
        }
        return self.ora_type != null and self.category != .Unknown;
    }

    /// Create TypeInfo from OraType
    pub fn fromOraType(ora_type: OraType) TypeInfo {
        return TypeInfo{
            .category = ora_type.getCategory(),
            .ora_type = ora_type,
            .source = .inferred,
            .span = null,
            .region = null,
        };
    }
};

/// Generic type categories for high-level classification
pub const TypeCategory = enum {
    // primitive categories
    Integer,
    String,
    Bool,
    Address,
    Hex,
    Bytes,

    // complex type categories
    Struct,
    Bitfield,
    Enum,
    Contract, // Contract type category
    Function,
    Array,
    Slice,
    Map,
    Tuple,
    ErrorUnion,
    Result,

    // special categories
    Void,
    Error,
    Module,
    Type, // comptime type — the metatype
    Unknown,

    pub fn format(self: TypeCategory, comptime fmt: []const u8, options: std.fmt.FormatOptions, writer: anytype) !void {
        _ = fmt;
        _ = options;
        try writer.writeAll(@tagName(self));
    }
};

/// Specific Ora language types
pub const OraType = union(enum) {
    // unsigned integer types
    u8: void,
    u16: void,
    u32: void,
    u64: void,
    u128: void,
    u160: void,
    u256: void,

    // signed integer types
    i8: void,
    i16: void,
    i32: void,
    i64: void,
    i128: void,
    i256: void,

    // other primitive types
    bool: void,
    string: void,
    address: void,
    bytes: void,
    void: void,

    // complex types with additional data
    struct_type: []const u8, // Struct name
    bitfield_type: []const u8, // Bitfield name
    enum_type: []const u8, // Enum name
    contract_type: []const u8, // Contract name
    array: struct { elem: *const OraType, len: u64 }, // Fixed-size array [T; N]
    slice: *const OraType, // Element type
    map: MapType, // Key and value types
    tuple: []const OraType, // Element types
    function: FunctionType, // Parameter and return types
    error_union: *const OraType, // Success type (!T)
    anonymous_struct: []const AnonymousStructFieldType, // struct { field: T, ... }
    module: ?[]const u8, // Optional module name

    // generics / comptime type system
    type: void, // the metatype — `comptime T: type`
    type_parameter: []const u8, // placeholder for a generic type param, e.g. "T"

    // refinement types
    min_value: struct {
        base: *const OraType, // Base integer type
        min: u256, // Minimum value (compile-time constant)
    },
    max_value: struct {
        base: *const OraType, // Base integer type
        max: u256, // Maximum value (compile-time constant)
    },
    in_range: struct {
        base: *const OraType, // Base integer type
        min: u256, // Minimum value
        max: u256, // Maximum value
    },
    scaled: struct {
        base: *const OraType, // Base integer type
        decimals: u32, // Scale factor (10^decimals)
    },
    exact: *const OraType, // Base integer type (must participate in exact division)
    non_zero_address: void, // Address type that cannot be zero address

    /// Get the category for this Ora type
    pub fn getCategory(self: OraType) TypeCategory {
        return switch (self) {
            .u8, .u16, .u32, .u64, .u128, .u160, .u256, .i8, .i16, .i32, .i64, .i128, .i256 => .Integer,
            .bool => .Bool,
            .string => .String,
            .address => .Address,
            .bytes => .Bytes,
            .void => .Void,
            .struct_type => .Struct,
            .bitfield_type => .Bitfield,
            .enum_type => .Enum,
            .contract_type => .Contract,
            .array => .Array,
            .slice => .Slice,
            .map => .Map,
            .tuple => .Tuple,
            .function => .Function,
            .error_union => .ErrorUnion,
            .anonymous_struct => .Struct,
            .module => .Module,
            .type => .Type,
            .type_parameter => .Unknown, // resolved during monomorphization
            // refinement types inherit the category of their base type
            .min_value => |mv| mv.base.*.getCategory(),
            .max_value => |mv| mv.base.*.getCategory(),
            .in_range => |ir| ir.base.*.getCategory(),
            .scaled => |s| s.base.*.getCategory(),
            .exact => |e| e.*.getCategory(),
            .non_zero_address => .Address, // NonZeroAddress is an Address type
        };
    }

    /// Check if this is an integer type
    pub fn isInteger(self: OraType) bool {
        return switch (self) {
            .u8, .u16, .u32, .u64, .u128, .u160, .u256, .i8, .i16, .i32, .i64, .i128, .i256 => true,
            // refinement types: check the base type
            .min_value => |mv| mv.base.isInteger(),
            .max_value => |mv| mv.base.isInteger(),
            .in_range => |ir| ir.base.isInteger(),
            .scaled => |s| s.base.isInteger(),
            .exact => |e| e.isInteger(),
            else => false,
        };
    }

    /// Check if this is an unsigned integer type
    pub fn isUnsignedInteger(self: OraType) bool {
        return switch (self) {
            .u8, .u16, .u32, .u64, .u128, .u160, .u256 => true,
            // refinement types: check the base type
            .min_value => |mv| mv.base.isUnsignedInteger(),
            .max_value => |mv| mv.base.isUnsignedInteger(),
            .in_range => |ir| ir.base.isUnsignedInteger(),
            .scaled => |s| s.base.isUnsignedInteger(),
            .exact => |e| e.isUnsignedInteger(),
            else => false,
        };
    }

    /// Return the bit width of a primitive/integer type, or null for complex types.
    pub fn bitWidth(self: OraType) ?u32 {
        return switch (self) {
            .bool => 1,
            .u8, .i8 => 8,
            .u16, .i16 => 16,
            .u32, .i32 => 32,
            .u64, .i64 => 64,
            .u128, .i128 => 128,
            .u160 => 160,
            .u256, .i256 => 256,
            .address => 160,
            else => null,
        };
    }

    /// Check if this is a signed integer type
    pub fn isSignedInteger(self: OraType) bool {
        return switch (self) {
            .i8, .i16, .i32, .i64, .i128, .i256 => true,
            // refinement types: check the base type
            .min_value => |mv| mv.base.isSignedInteger(),
            .max_value => |mv| mv.base.isSignedInteger(),
            .in_range => |ir| ir.base.isSignedInteger(),
            .scaled => |s| s.base.isSignedInteger(),
            .exact => |e| e.isSignedInteger(),
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
            .u160 => "u160",
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
            .bitfield_type => |name| name,
            .contract_type => |name| name,
            .array => "array",
            .slice => "slice",
            .map => "map",
            .tuple => "tuple",
            .function => "function",
            .error_union => "error_union",
            .anonymous_struct => "struct",
            .module => "module",
            .type => "type",
            .type_parameter => |name| name,
            // refinement types - use render() for proper formatting
            .min_value, .max_value, .in_range, .scaled, .exact, .non_zero_address => "refinement",
        };
    }

    pub fn format(self: OraType, comptime fmt: []const u8, options: std.fmt.FormatOptions, writer: anytype) !void {
        _ = fmt;
        _ = options;
        try writer.writeAll(self.toString());
    }

    /// Render this type as source-like syntax into writer (structural, not pointer-based)
    pub fn render(self: OraType, writer: anytype) !void {
        switch (self) {
            .u8 => try writer.writeAll("u8"),
            .u16 => try writer.writeAll("u16"),
            .u32 => try writer.writeAll("u32"),
            .u64 => try writer.writeAll("u64"),
            .u128 => try writer.writeAll("u128"),
            .u160 => try writer.writeAll("u160"),
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
            .bitfield_type => |name| try writer.writeAll(name),
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
            .map => |m| {
                try writer.writeAll("map<");
                try (@constCast(m.key).*).render(writer);
                try writer.writeAll(", ");
                try (@constCast(m.value).*).render(writer);
                try writer.writeByte('>');
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
            .type => try writer.writeAll("type"),
            .type_parameter => |name| try writer.writeAll(name),
            .min_value => |mv| {
                try writer.writeAll(refinements.nameForKind(.min_value));
                try writer.writeByte('<');
                try (@constCast(mv.base).*).render(writer);
                try writer.writeAll(", ");
                var buf: [32]u8 = undefined;
                const s = std.fmt.bufPrint(&buf, "{d}", .{mv.min}) catch "?";
                try writer.writeAll(s);
                try writer.writeByte('>');
            },
            .max_value => |mv| {
                try writer.writeAll(refinements.nameForKind(.max_value));
                try writer.writeByte('<');
                try (@constCast(mv.base).*).render(writer);
                try writer.writeAll(", ");
                var buf: [32]u8 = undefined;
                const s = std.fmt.bufPrint(&buf, "{d}", .{mv.max}) catch "?";
                try writer.writeAll(s);
                try writer.writeByte('>');
            },
            .in_range => |ir| {
                try writer.writeAll(refinements.nameForKind(.in_range));
                try writer.writeByte('<');
                try (@constCast(ir.base).*).render(writer);
                try writer.writeAll(", ");
                var buf_min: [32]u8 = undefined;
                var buf_max: [32]u8 = undefined;
                const s_min = std.fmt.bufPrint(&buf_min, "{d}", .{ir.min}) catch "?";
                const s_max = std.fmt.bufPrint(&buf_max, "{d}", .{ir.max}) catch "?";
                try writer.writeAll(s_min);
                try writer.writeAll(", ");
                try writer.writeAll(s_max);
                try writer.writeByte('>');
            },
            .scaled => |s| {
                try writer.writeAll(refinements.nameForKind(.scaled));
                try writer.writeByte('<');
                try (@constCast(s.base).*).render(writer);
                try writer.writeAll(", ");
                var buf: [32]u8 = undefined;
                const s_dec = std.fmt.bufPrint(&buf, "{d}", .{s.decimals}) catch "?";
                try writer.writeAll(s_dec);
                try writer.writeByte('>');
            },
            .exact => |e| {
                try writer.writeAll(refinements.nameForKind(.exact));
                try writer.writeByte('<');
                try (@constCast(e).*).render(writer);
                try writer.writeByte('>');
            },
            .non_zero_address => {
                try writer.writeAll(refinements.nameForKind(.non_zero_address));
            },
        }
    }
};

/// Complex type definitions
pub const MapType = struct {
    key: *const OraType,
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
