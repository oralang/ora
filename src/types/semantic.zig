const std = @import("std");

const type_builtin = @import("builtin.zig");

pub const TypeKind = enum {
    unknown,
    never,
    void,
    bool,
    integer,
    comptime_integer,
    string,
    address,
    bytes,
    fixed_bytes,
    external_proxy,
    named,
    function,
    contract,
    struct_,
    bitfield,
    enum_,
    tuple,
    anonymous_struct,
    array,
    slice,
    map,
    error_union,
    refinement,
};

pub const Region = enum {
    none,
    storage,
    memory,
    transient,
    calldata,
};

pub const Provenance = enum {
    local,
    calldata,
    storage,
    external,
};

pub const NamedType = struct {
    name: []const u8,
};

pub const ExternalProxyType = struct {
    trait_name: []const u8,
};

pub const IntegerType = struct {
    bits: u16,
    signed: bool,
    spelling: ?[]const u8 = null,

    pub fn builtinSpec(self: IntegerType) ?type_builtin.BuiltinTypeSpec {
        return type_builtin.lookupIntegerBuiltin(self.signed, self.bits);
    }

    pub fn isUnsignedBits(self: IntegerType, bits: u16) bool {
        return !self.signed and self.bits == bits;
    }
};

pub const ComptimeIntegerType = struct {
    spelling: ?[]const u8 = null,
};

pub const FixedBytesType = struct {
    len: u8,
    spelling: ?[]const u8 = null,
};

pub const FunctionType = struct {
    name: ?[]const u8 = null,
    param_types: []const Type = &.{},
    return_types: []const Type = &.{},
};

pub const ArrayType = struct {
    element_type: *const Type,
    len: ?u32 = null,
};

pub const SliceType = struct {
    element_type: *const Type,
};

pub const MapType = struct {
    key_type: ?*const Type = null,
    value_type: ?*const Type = null,
};

pub const AnonymousStructField = struct {
    name: []const u8,
    ty: Type,
};

pub const AnonymousStructType = struct {
    fields: []const AnonymousStructField,

    pub fn fieldIndex(self: AnonymousStructType, name: []const u8) ?usize {
        return anonymousStructFieldIndex(self.fields, name);
    }

    pub fn fieldByName(self: AnonymousStructType, name: []const u8) ?AnonymousStructField {
        const index = self.fieldIndex(name) orelse return null;
        return self.fields[index];
    }
};

pub fn anonymousStructFieldIndex(fields: []const AnonymousStructField, name: []const u8) ?usize {
    for (fields, 0..) |field, index| {
        if (std.mem.eql(u8, field.name, name)) return index;
    }
    return null;
}

pub fn anonymousStructFieldByName(fields: []const AnonymousStructField, name: []const u8) ?AnonymousStructField {
    const index = anonymousStructFieldIndex(fields, name) orelse return null;
    return fields[index];
}

pub const ErrorUnionType = struct {
    payload_type: *const Type,
    error_types: []const Type = &.{},
};

pub const RefinementIntegerArg = struct {
    text: []const u8,
};

pub const RefinementArg = union(enum) {
    Type,
    Integer: RefinementIntegerArg,
};

pub const RefinementType = struct {
    name: []const u8,
    base_type: *const Type,
    args: []const RefinementArg = &.{},
};

pub const Type = union(TypeKind) {
    unknown: void,
    never: void,
    void: void,
    bool: void,
    integer: IntegerType,
    comptime_integer: ComptimeIntegerType,
    string: void,
    address: void,
    bytes: void,
    fixed_bytes: FixedBytesType,
    external_proxy: ExternalProxyType,
    named: NamedType,
    function: FunctionType,
    contract: NamedType,
    struct_: NamedType,
    bitfield: NamedType,
    enum_: NamedType,
    tuple: []const Type,
    anonymous_struct: AnonymousStructType,
    array: ArrayType,
    slice: SliceType,
    map: MapType,
    error_union: ErrorUnionType,
    refinement: RefinementType,

    pub fn kind(self: Type) TypeKind {
        return std.meta.activeTag(self);
    }

    pub fn name(self: *const Type) ?[]const u8 {
        return switch (self.*) {
            .integer => |integer| integer.spelling,
            .comptime_integer => |integer| integer.spelling,
            .fixed_bytes => |fixed_bytes| fixed_bytes.spelling,
            .external_proxy => |proxy| proxy.trait_name,
            .named => |named| named.name,
            .function => |function| function.name,
            .contract => |named| named.name,
            .struct_ => |named| named.name,
            .bitfield => |named| named.name,
            .enum_ => |named| named.name,
            .refinement => |refinement| refinement.name,
            else => null,
        };
    }

    pub fn elementType(self: *const Type) ?*const Type {
        return switch (self.*) {
            .array => |array| array.element_type,
            .slice => |slice| slice.element_type,
            else => null,
        };
    }

    pub fn keyType(self: *const Type) ?*const Type {
        return switch (self.*) {
            .map => |map| map.key_type,
            else => null,
        };
    }

    pub fn valueType(self: *const Type) ?*const Type {
        return switch (self.*) {
            .map => |map| map.value_type,
            else => null,
        };
    }

    pub fn payloadType(self: *const Type) ?*const Type {
        return switch (self.*) {
            .error_union => |error_union| error_union.payload_type,
            else => null,
        };
    }

    pub fn refinementBaseType(self: *const Type) ?*const Type {
        return switch (self.*) {
            .refinement => |refinement| refinement.base_type,
            else => null,
        };
    }

    pub fn arrayLen(self: *const Type) ?u32 {
        return switch (self.*) {
            .array => |array| array.len,
            else => null,
        };
    }

    pub fn tupleTypes(self: *const Type) []const Type {
        return switch (self.*) {
            .tuple => |tuple| tuple,
            else => &.{},
        };
    }

    pub fn anonymousStructFields(self: *const Type) []const AnonymousStructField {
        return switch (self.*) {
            .anonymous_struct => |struct_type| struct_type.fields,
            else => &.{},
        };
    }

    pub fn errorTypes(self: *const Type) []const Type {
        return switch (self.*) {
            .error_union => |error_union| error_union.error_types,
            else => &.{},
        };
    }

    pub fn paramTypes(self: *const Type) []const Type {
        return switch (self.*) {
            .function => |function| function.param_types,
            else => &.{},
        };
    }

    pub fn returnTypes(self: *const Type) []const Type {
        return switch (self.*) {
            .function => |function| function.return_types,
            else => &.{},
        };
    }
};

pub const GenericBindingValue = union(enum) {
    ty: Type,
    integer: []const u8,
};

pub const GenericTypeBinding = struct {
    name: []const u8,
    value: GenericBindingValue,
};

pub fn appendTypeMangleName(allocator: std.mem.Allocator, buffer: *std.ArrayList(u8), ty: Type) !void {
    switch (ty) {
        .never => try buffer.appendSlice(allocator, "never"),
        .bool => try buffer.appendSlice(allocator, "bool"),
        .address => try buffer.appendSlice(allocator, "address"),
        .string => try buffer.appendSlice(allocator, "string"),
        .bytes => try buffer.appendSlice(allocator, "bytes"),
        .fixed_bytes => |fixed_bytes| try buffer.writer(allocator).print("bytes{d}", .{fixed_bytes.len}),
        .external_proxy => |proxy| try buffer.writer(allocator).print("external_{s}", .{proxy.trait_name}),
        .void => try buffer.appendSlice(allocator, "void"),
        .integer => |integer| try appendIntegerTypeMangleName(allocator, buffer, integer),
        .comptime_integer => |integer| try buffer.appendSlice(allocator, integer.spelling orelse "comptime_int"),
        .named => |named| try buffer.appendSlice(allocator, named.name),
        .struct_ => |named| try buffer.appendSlice(allocator, named.name),
        .contract => |named| try buffer.appendSlice(allocator, named.name),
        .bitfield => |named| try buffer.appendSlice(allocator, named.name),
        .enum_ => |named| try buffer.appendSlice(allocator, named.name),
        .refinement => |refinement| try buffer.appendSlice(allocator, refinement.name),
        .anonymous_struct => |struct_type| {
            try buffer.appendSlice(allocator, "anon_struct");
            for (struct_type.fields) |field| {
                try buffer.append(allocator, '_');
                try buffer.appendSlice(allocator, field.name);
                try buffer.append(allocator, '_');
                try appendTypeMangleName(allocator, buffer, field.ty);
            }
        },
        .slice => |slice| {
            try buffer.appendSlice(allocator, "slice_");
            try appendTypeMangleName(allocator, buffer, slice.element_type.*);
        },
        .array => |array| {
            try buffer.appendSlice(allocator, "array_");
            try appendTypeMangleName(allocator, buffer, array.element_type.*);
            if (array.len) |len| {
                try buffer.append(allocator, '_');
                try buffer.writer(allocator).print("{d}", .{len});
            }
        },
        .map => |map| {
            try buffer.appendSlice(allocator, "map");
            if (map.key_type) |key| {
                try buffer.append(allocator, '_');
                try appendTypeMangleName(allocator, buffer, key.*);
            }
            if (map.value_type) |value| {
                try buffer.append(allocator, '_');
                try appendTypeMangleName(allocator, buffer, value.*);
            }
        },
        .tuple => |elements| {
            try buffer.appendSlice(allocator, "tuple");
            for (elements) |element| {
                try buffer.append(allocator, '_');
                try appendTypeMangleName(allocator, buffer, element);
            }
        },
        .error_union => |error_union| {
            try buffer.appendSlice(allocator, "error_union_");
            try appendTypeMangleName(allocator, buffer, error_union.payload_type.*);
        },
        .function => |function| try buffer.appendSlice(allocator, function.name orelse "fn"),
        .unknown => try buffer.appendSlice(allocator, "type"),
    }
}

fn appendIntegerTypeMangleName(allocator: std.mem.Allocator, buffer: *std.ArrayList(u8), integer: IntegerType) !void {
    if (integer.spelling) |spelling| {
        try buffer.appendSlice(allocator, spelling);
        return;
    }
    try buffer.writer(allocator).print("{c}{d}", .{ if (integer.signed) @as(u8, 'i') else @as(u8, 'u'), integer.bits });
}

pub const LocatedType = struct {
    type: Type,
    region: Region = .none,
    provenance: Provenance = .local,

    pub fn unlocated(ty: Type) LocatedType {
        return .{
            .type = ty,
            .region = .none,
            .provenance = .local,
        };
    }

    pub fn withRegion(ty: Type, region: Region) LocatedType {
        return .{
            .type = ty,
            .region = region,
            .provenance = .local,
        };
    }

    pub fn withRegionAndProvenance(ty: Type, region: Region, provenance: Provenance) LocatedType {
        return .{
            .type = ty,
            .region = region,
            .provenance = provenance,
        };
    }

    pub fn kind(self: LocatedType) TypeKind {
        return self.type.kind();
    }

    pub fn name(self: *const LocatedType) ?[]const u8 {
        return self.type.name();
    }

    pub fn elementType(self: *const LocatedType) ?*const Type {
        return self.type.elementType();
    }

    pub fn keyType(self: *const LocatedType) ?*const Type {
        return self.type.keyType();
    }

    pub fn valueType(self: *const LocatedType) ?*const Type {
        return self.type.valueType();
    }

    pub fn payloadType(self: *const LocatedType) ?*const Type {
        return self.type.payloadType();
    }

    pub fn arrayLen(self: *const LocatedType) ?u32 {
        return self.type.arrayLen();
    }

    pub fn tupleTypes(self: *const LocatedType) []const Type {
        return self.type.tupleTypes();
    }

    pub fn errorTypes(self: *const LocatedType) []const Type {
        return self.type.errorTypes();
    }

    pub fn paramTypes(self: *const LocatedType) []const Type {
        return self.type.paramTypes();
    }

    pub fn returnTypes(self: *const LocatedType) []const Type {
        return self.type.returnTypes();
    }
};

test "semantic type helpers expose refinement base type" {
    const base: Type = .{ .integer = .{ .bits = 256, .signed = false, .spelling = "u256" } };
    const refinement: Type = .{ .refinement = .{
        .name = "MinValue",
        .base_type = &base,
        .args = &.{},
    } };

    try std.testing.expectEqual(TypeKind.refinement, refinement.kind());
    try std.testing.expectEqualStrings("MinValue", refinement.name().?);
    try std.testing.expect(refinement.refinementBaseType() != null);
    try std.testing.expectEqual(TypeKind.integer, refinement.refinementBaseType().?.kind());
}
