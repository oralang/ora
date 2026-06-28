const std = @import("std");
const ora_types = @import("ora_types");
const refinement_semantics = ora_types.refinement_semantics;
const type_builtin = ora_types.builtin;
const abi_type_names = @import("type_names.zig");

const Type = ora_types.SemanticType;

pub const Position = enum {
    input,
    output,
};

pub const ResultInputMode = enum {
    none,
    narrow_payloadless,
    wide_payloadless,
    wide_single_error,
};

pub const DecodeMode = enum {
    strict,
    permissive,
};

pub const ResultCarrierPlan = struct {
    mode: ResultInputMode,
    payload: Type,
    err: ?Type = null,
};

pub const NamedTypeKind = enum {
    none,
    enum_,
    bitfield,
    struct_,
    contract,
    error_decl,
    type_alias,
};

pub fn publicReturnAbiTypeName(ty: Type) ?[]const u8 {
    return switch (ty) {
        .void => abi_type_names.builtinAbiName(.void),
        .tuple => abi_type_names.publicTupleReturnAbiName(),
        .anonymous_struct => abi_type_names.publicStructReturnAbiName(),
        .bitfield => abi_type_names.builtinAbiName(.u256),
        .struct_, .contract => abi_type_names.publicTupleReturnAbiName(),
        else => null,
    };
}

pub fn Policy(comptime Provider: type) type {
    return struct {
        provider: Provider,

        const Self = @This();

        pub fn supportsType(self: *const Self, ty: Type, position: Position) bool {
            return switch (ty) {
                .unknown => true,
                .void => position == .output,
                .bool, .address, .string, .bytes, .fixed_bytes, .integer, .bitfield => true,
                .resource_domain => |resource| self.supportsType(resource.carrier_type.*, position),
                .enum_ => |named| !self.provider.enumHasPayload(named.name),
                .refinement => |refinement| self.supportsType(refinement.base_type.*, position),
                .array => |array| self.supportsType(array.element_type.*, position),
                .slice => |slice| self.supportsType(slice.element_type.*, position),
                .tuple => |elements| blk: {
                    for (elements) |element| {
                        if (!self.supportsType(element, position)) break :blk false;
                    }
                    break :blk true;
                },
                .anonymous_struct => |struct_type| blk: {
                    for (struct_type.fields) |field| {
                        if (!self.supportsType(field.ty, position)) break :blk false;
                    }
                    break :blk true;
                },
                .struct_, .contract, .named => true,
                .error_union => |error_union| switch (position) {
                    .input => self.planResultCarrier(ty) != null,
                    .output => self.supportsType(error_union.payload_type.*, .output),
                },
                else => false,
            };
        }

        pub fn planResultCarrier(self: *const Self, ty: Type) ?ResultCarrierPlan {
            const error_union = switch (ty) {
                .error_union => |error_union| error_union,
                else => return null,
            };
            if (error_union.error_types.len != 1) return null;

            const payload = error_union.payload_type.*;
            if (!self.resultInputCarrierShapeSupported(payload)) return null;

            const err_ty = error_union.error_types[0];
            const payload_words = self.staticWordCount(payload);
            if (!self.errorTypeHasPayload(err_ty)) {
                const mode: ResultInputMode = if (payload_words != null and payload_words.? == 1 and self.resultInputPayloadFitsNarrowCarrier(payload) and self.payloadlessErrorHasRuntimeId(err_ty))
                    .narrow_payloadless
                else
                    .wide_payloadless;
                return .{ .mode = mode, .payload = payload };
            }

            if (!self.resultInputCarrierShapeSupported(err_ty)) return null;
            if (payload_words != null and payload_words.? == 1) {
                if (self.staticWordCount(err_ty)) |error_words| {
                    if (error_words > 1) return null;
                }
            }
            return .{ .mode = .wide_single_error, .payload = payload, .err = err_ty };
        }

        pub fn staticWordCount(self: *const Self, ty: Type) ?usize {
            if (@hasDecl(Provider, "staticWordCount")) return self.provider.staticWordCount(ty);
            return self.defaultStaticWordCount(ty);
        }

        pub fn errorTypeHasPayload(self: *const Self, ty: Type) bool {
            if (@hasDecl(Provider, "errorTypeHasPayload")) return self.provider.errorTypeHasPayload(ty);
            return self.defaultErrorTypeHasPayload(ty);
        }

        pub fn supportsAbiEncode(self: *const Self, ty: Type) bool {
            return switch (ty) {
                .bool, .address, .fixed_bytes, .bitfield, .void => true,
                .resource_domain => |resource| self.supportsAbiEncode(resource.carrier_type.*),
                .integer => |integer| integerSpec(integer) != null,
                .enum_ => |named| !self.provider.enumHasPayload(named.name),
                .string, .bytes => true,
                .slice => |slice| self.supportsAbiEncode(slice.element_type.*),
                .array => |array| self.supportsAbiEncode(array.element_type.*),
                .tuple => |elements| blk: {
                    for (elements) |element| {
                        if (!self.supportsAbiEncode(element)) break :blk false;
                    }
                    break :blk true;
                },
                .anonymous_struct => |struct_type| blk: {
                    for (struct_type.fields) |field| {
                        if (!self.supportsAbiEncode(field.ty)) break :blk false;
                    }
                    break :blk true;
                },
                .struct_ => |named| self.supportsAbiEncodeStruct(named.name),
                .named => |named| self.supportsAbiEncodeNamed(named.name),
                .refinement => |refinement| self.supportsAbiEncode(refinement.base_type.*),
                else => false,
            };
        }

        pub fn supportsAbiDecode(self: *const Self, ty: Type) bool {
            return switch (ty) {
                .refinement => |refinement| !refinement_semantics.isCompileTimeOnly(refinement) and self.supportsAbiDecode(refinement.base_type.*),
                .resource_domain => |resource| self.supportsAbiDecode(resource.carrier_type.*),
                .slice => |slice| self.supportsAbiDecode(slice.element_type.*),
                .array => |array| self.supportsAbiDecode(array.element_type.*),
                .tuple => |elements| blk: {
                    for (elements) |element| {
                        if (!self.supportsAbiDecode(element)) break :blk false;
                    }
                    break :blk true;
                },
                .anonymous_struct => |struct_type| blk: {
                    for (struct_type.fields) |field| {
                        if (!self.supportsAbiDecode(field.ty)) break :blk false;
                    }
                    break :blk true;
                },
                .struct_ => |named| self.supportsAbiDecodeStruct(named.name),
                .named => |named| self.supportsAbiDecodeNamed(named.name),
                else => self.supportsAbiEncode(ty),
            };
        }

        pub fn supportsRuntimeAbiDecode(self: *const Self, ty: Type, mode: DecodeMode) bool {
            return self.supportsRuntimeAbiDecodeInContext(ty, mode, true);
        }

        fn defaultStaticWordCount(self: *const Self, ty: Type) ?usize {
            return switch (ty) {
                .bool, .address, .fixed_bytes, .bitfield => 1,
                .integer => |integer| if (integerSpec(integer) != null) 1 else null,
                .enum_ => |named| if (self.provider.enumHasPayload(named.name)) null else 1,
                .refinement => |refinement| self.staticWordCount(refinement.base_type.*),
                .resource_domain => |resource| self.staticWordCount(resource.carrier_type.*),
                .tuple => |elements| blk: {
                    var total: usize = 0;
                    for (elements) |element| {
                        total += self.staticWordCount(element) orelse break :blk null;
                    }
                    break :blk total;
                },
                .array => |array| blk: {
                    const len = array.len orelse break :blk null;
                    const element_words = self.staticWordCount(array.element_type.*) orelse break :blk null;
                    break :blk element_words * len;
                },
                .anonymous_struct => |struct_type| blk: {
                    var total: usize = 0;
                    for (struct_type.fields) |field| {
                        total += self.staticWordCount(field.ty) orelse break :blk null;
                    }
                    break :blk total;
                },
                .struct_ => |named| self.staticWordCountForNamedStruct(named.name),
                .contract => |named| self.staticWordCountForNamedStruct(named.name),
                .named => |named| blk: {
                    if (staticWordCountForBuiltinName(named.name)) |words| break :blk words;
                    break :blk switch (self.namedTypeKind(named.name)) {
                        .enum_ => if (self.provider.enumHasPayload(named.name)) null else 1,
                        .bitfield => 1,
                        .struct_ => self.staticWordCountForStructDecl(named.name),
                        .contract => self.staticWordCountForContractDecl(named.name),
                        .error_decl => blk2: {
                            const payloads = self.errorPayloadTypes(named.name) orelse break :blk2 null;
                            var total: usize = 0;
                            for (payloads) |payload| {
                                total += self.staticWordCount(payload) orelse break :blk2 null;
                            }
                            break :blk2 total;
                        },
                        else => null,
                    };
                },
                else => null,
            };
        }

        fn defaultErrorTypeHasPayload(self: *const Self, ty: Type) bool {
            const name = ty.name() orelse return true;
            const payloads = self.errorPayloadTypes(name) orelse return true;
            return payloads.len != 0;
        }

        fn supportsAbiEncodeNamed(self: *const Self, name: []const u8) bool {
            if (abiNamedScalarSupported(name)) return true;
            return switch (self.namedTypeKind(name)) {
                .enum_ => !self.provider.enumHasPayload(name),
                .bitfield => true,
                .struct_ => self.supportsAbiEncodeStruct(name),
                .type_alias => blk: {
                    const target = self.typeAliasTarget(name) orelse break :blk false;
                    break :blk self.supportsAbiEncode(target);
                },
                else => false,
            };
        }

        fn supportsAbiEncodeStruct(self: *const Self, name: []const u8) bool {
            const fields = self.structFieldTypes(name) orelse return false;
            for (fields) |field_type| {
                if (!self.supportsAbiEncode(field_type)) return false;
            }
            return true;
        }

        fn supportsAbiDecodeNamed(self: *const Self, name: []const u8) bool {
            if (abiNamedScalarSupported(name)) return true;
            return switch (self.namedTypeKind(name)) {
                .enum_ => !self.provider.enumHasPayload(name),
                .bitfield => true,
                .struct_ => self.supportsAbiDecodeStruct(name),
                .type_alias => blk: {
                    const target = self.typeAliasTarget(name) orelse break :blk false;
                    break :blk self.supportsAbiDecode(target);
                },
                else => false,
            };
        }

        fn supportsAbiDecodeStruct(self: *const Self, name: []const u8) bool {
            const fields = self.structFieldTypes(name) orelse return false;
            for (fields) |field_type| {
                if (!self.supportsAbiDecode(field_type)) return false;
            }
            return true;
        }

        fn supportsRuntimeAbiDecodeInContext(self: *const Self, ty: Type, mode: DecodeMode, allow_top_level_dynamic: bool) bool {
            return switch (ty) {
                .bool, .address, .fixed_bytes, .enum_, .bitfield, .void => true,
                .integer => |integer| integerSpec(integer) != null,
                .string, .bytes => allow_top_level_dynamic,
                .slice => |slice| allow_top_level_dynamic and self.supportsRuntimeAbiDecodeSliceElement(slice.element_type.*, mode),
                .refinement => |refinement| refinement_semantics.supportsRuntimeGuard(refinement) and
                    refinement_semantics.hasNativeMlirTypeName(refinement.name) and
                    self.supportsRuntimeAbiDecodeInContext(refinement.base_type.*, mode, allow_top_level_dynamic),
                .tuple => |elements| blk: {
                    if (elements.len <= 1) break :blk false;
                    if (allow_top_level_dynamic and self.supportsRuntimeAbiDecodeTopLevelMixedDynamicTuple(elements, mode)) break :blk true;
                    for (elements) |element| {
                        if (element == .void or !self.supportsRuntimeAbiDecodeInContext(element, mode, false)) break :blk false;
                    }
                    break :blk true;
                },
                .named => |named| self.supportsRuntimeAbiDecodeNamed(named.name, mode, allow_top_level_dynamic),
                else => false,
            };
        }

        fn supportsRuntimeAbiDecodeNamed(self: *const Self, name: []const u8, mode: DecodeMode, allow_top_level_dynamic: bool) bool {
            if (runtimeDecodeBuiltinNameSupported(name, allow_top_level_dynamic)) return true;
            return switch (self.namedTypeKind(name)) {
                .enum_ => !self.provider.enumHasPayload(name),
                .bitfield => true,
                .type_alias => blk: {
                    const target = self.typeAliasTarget(name) orelse break :blk false;
                    break :blk self.supportsRuntimeAbiDecodeInContext(target, mode, allow_top_level_dynamic);
                },
                else => false,
            };
        }

        fn supportsRuntimeAbiDecodeTopLevelMixedDynamicTuple(self: *const Self, elements: []const Type, mode: DecodeMode) bool {
            if (elements.len != 2) return false;
            if (!self.isRuntimeAbiDecodeU256(elements[0])) return false;
            if (self.isRuntimeAbiDecodeDynamicBytesLike(elements[1])) return true;
            if (self.isRuntimeAbiDecodeU256Slice(elements[1])) return true;
            return switch (mode) {
                .permissive => false,
                .strict => self.isRuntimeAbiDecodeAddressSlice(elements[1]) or
                    self.isRuntimeAbiDecodeBoolSlice(elements[1]) or
                    self.isRuntimeAbiDecodeFixedBytesSlice(elements[1]),
            };
        }

        fn supportsRuntimeAbiDecodeSliceElement(self: *const Self, ty: Type, mode: DecodeMode) bool {
            if (self.isRuntimeAbiDecodeU256(ty)) return true;
            return switch (mode) {
                .permissive => false,
                .strict => self.isRuntimeAbiDecodeAddress(ty) or
                    self.isRuntimeAbiDecodeBool(ty) or
                    self.isRuntimeAbiDecodeFixedBytes(ty),
            };
        }

        fn isRuntimeAbiDecodeDynamicBytesLike(self: *const Self, ty: Type) bool {
            return switch (ty) {
                .string, .bytes => true,
                .named => |named| blk: {
                    if (dynamicBytesBuiltinName(named.name)) break :blk true;
                    const target = self.typeAliasTarget(named.name) orelse break :blk false;
                    break :blk self.isRuntimeAbiDecodeDynamicBytesLike(target);
                },
                else => false,
            };
        }

        fn isRuntimeAbiDecodeU256Slice(self: *const Self, ty: Type) bool {
            return switch (ty) {
                .slice => |slice| self.isRuntimeAbiDecodeU256(slice.element_type.*),
                .named => |named| blk: {
                    const target = self.typeAliasTarget(named.name) orelse break :blk false;
                    break :blk self.isRuntimeAbiDecodeU256Slice(target);
                },
                else => false,
            };
        }

        fn isRuntimeAbiDecodeAddressSlice(self: *const Self, ty: Type) bool {
            return switch (ty) {
                .slice => |slice| self.isRuntimeAbiDecodeAddress(slice.element_type.*),
                .named => |named| blk: {
                    const target = self.typeAliasTarget(named.name) orelse break :blk false;
                    break :blk self.isRuntimeAbiDecodeAddressSlice(target);
                },
                else => false,
            };
        }

        fn isRuntimeAbiDecodeBoolSlice(self: *const Self, ty: Type) bool {
            return switch (ty) {
                .slice => |slice| self.isRuntimeAbiDecodeBool(slice.element_type.*),
                .named => |named| blk: {
                    const target = self.typeAliasTarget(named.name) orelse break :blk false;
                    break :blk self.isRuntimeAbiDecodeBoolSlice(target);
                },
                else => false,
            };
        }

        fn isRuntimeAbiDecodeFixedBytesSlice(self: *const Self, ty: Type) bool {
            return switch (ty) {
                .slice => |slice| self.isRuntimeAbiDecodeFixedBytes(slice.element_type.*),
                .named => |named| blk: {
                    const target = self.typeAliasTarget(named.name) orelse break :blk false;
                    break :blk self.isRuntimeAbiDecodeFixedBytesSlice(target);
                },
                else => false,
            };
        }

        fn isRuntimeAbiDecodeFixedBytes(self: *const Self, ty: Type) bool {
            return switch (ty) {
                .fixed_bytes => |fixed_bytes| fixed_bytes.len >= 1 and fixed_bytes.len <= 32,
                .named => |named| blk: {
                    if (type_builtin.parseFixedBytesName(named.name) != null) break :blk true;
                    const target = self.typeAliasTarget(named.name) orelse break :blk false;
                    break :blk self.isRuntimeAbiDecodeFixedBytes(target);
                },
                else => false,
            };
        }

        fn isRuntimeAbiDecodeU256(_: *const Self, ty: Type) bool {
            return switch (ty) {
                .integer => |integer| blk: {
                    const spec = integerSpec(integer) orelse break :blk false;
                    break :blk spec.id == .u256;
                },
                .named => |named| blk: {
                    const spec = type_builtin.lookupBuiltinByName(named.name) orelse break :blk false;
                    break :blk spec.id == .u256;
                },
                else => false,
            };
        }

        fn isRuntimeAbiDecodeBool(_: *const Self, ty: Type) bool {
            return switch (ty) {
                .bool => true,
                .named => |named| blk: {
                    const spec = type_builtin.lookupBuiltinByName(named.name) orelse break :blk false;
                    break :blk spec.id == .bool;
                },
                else => false,
            };
        }

        fn isRuntimeAbiDecodeAddress(_: *const Self, ty: Type) bool {
            return switch (ty) {
                .address => true,
                .named => |named| blk: {
                    const spec = type_builtin.lookupBuiltinByName(named.name) orelse break :blk false;
                    break :blk spec.id == .address;
                },
                else => false,
            };
        }

        fn namedTypeKind(self: *const Self, name: []const u8) NamedTypeKind {
            if (@hasDecl(Provider, "namedTypeKind")) return self.provider.namedTypeKind(name);
            return .none;
        }

        fn typeAliasTarget(self: *const Self, name: []const u8) ?Type {
            if (@hasDecl(Provider, "typeAliasTarget")) return self.provider.typeAliasTarget(name);
            return null;
        }

        fn structFieldTypes(self: *const Self, name: []const u8) ?[]const Type {
            if (@hasDecl(Provider, "structFieldTypes")) return self.provider.structFieldTypes(name);
            return null;
        }

        fn contractFieldTypes(self: *const Self, name: []const u8) ?[]const Type {
            if (@hasDecl(Provider, "contractFieldTypes")) return self.provider.contractFieldTypes(name);
            return null;
        }

        fn errorPayloadTypes(self: *const Self, name: []const u8) ?[]const Type {
            if (@hasDecl(Provider, "errorPayloadTypes")) return self.provider.errorPayloadTypes(name);
            return null;
        }

        fn resultInputCarrierShapeSupported(self: *const Self, ty: Type) bool {
            if (self.staticWordCount(ty) != null) return true;
            return switch (ty) {
                .bytes, .string => true,
                .slice => |slice| self.resultInputDynamicArrayElementSupported(slice.element_type.*),
                .array => |array| array.len == null and self.resultInputDynamicArrayElementSupported(array.element_type.*),
                .anonymous_struct => |struct_type| {
                    for (struct_type.fields) |field| {
                        if (!self.resultInputCarrierShapeSupported(field.ty)) return false;
                    }
                    return true;
                },
                .tuple => |elements| {
                    for (elements) |element| {
                        if (!self.resultInputCarrierShapeSupported(element)) return false;
                    }
                    return true;
                },
                .refinement => |refinement| self.resultInputCarrierShapeSupported(refinement.base_type.*),
                .named => |named| blk: {
                    const payloads = self.errorPayloadTypes(named.name) orelse break :blk false;
                    for (payloads) |payload| {
                        if (!self.resultInputCarrierShapeSupported(payload)) break :blk false;
                    }
                    break :blk true;
                },
                else => false,
            };
        }

        fn resultInputDynamicArrayElementSupported(self: *const Self, ty: Type) bool {
            return switch (ty) {
                .bool, .address, .fixed_bytes => true,
                .integer => |integer| blk: {
                    const spec = integerSpec(integer) orelse break :blk false;
                    break :blk spec.id == .u256;
                },
                .refinement => |refinement| self.resultInputDynamicArrayElementSupported(refinement.base_type.*),
                .named => |named| blk: {
                    if (resultDynamicArrayBuiltinNameSupported(named.name)) break :blk true;
                    const target = self.typeAliasTarget(named.name) orelse break :blk false;
                    break :blk self.resultInputDynamicArrayElementSupported(target);
                },
                else => false,
            };
        }

        fn resultInputPayloadFitsNarrowCarrier(self: *const Self, ty: Type) bool {
            return switch (ty) {
                .bool, .address => true,
                .integer => |integer| {
                    const spec = integerSpec(integer) orelse return false;
                    const bits = spec.bit_width orelse return false;
                    return bits <= 255;
                },
                .refinement => |refinement| self.resultInputPayloadFitsNarrowCarrier(refinement.base_type.*),
                else => false,
            };
        }

        fn payloadlessErrorHasRuntimeId(self: *const Self, ty: Type) bool {
            const error_name = ty.name() orelse return false;
            return self.errorPayloadTypes(error_name) != null;
        }

        fn staticWordCountForNamedStruct(self: *const Self, name: []const u8) ?usize {
            return switch (self.namedTypeKind(name)) {
                .struct_ => self.staticWordCountForStructDecl(name),
                .contract => self.staticWordCountForContractDecl(name),
                else => null,
            };
        }

        fn staticWordCountForStructDecl(self: *const Self, name: []const u8) ?usize {
            const fields = self.structFieldTypes(name) orelse return null;
            var total: usize = 0;
            for (fields) |field_type| {
                total += self.staticWordCount(field_type) orelse return null;
            }
            return total;
        }

        fn staticWordCountForContractDecl(self: *const Self, name: []const u8) ?usize {
            const fields = self.contractFieldTypes(name) orelse return null;
            var total: usize = 0;
            for (fields) |field_type| {
                total += self.staticWordCount(field_type) orelse return null;
            }
            return total;
        }
    };
}

fn staticWordCountForBuiltinName(name: []const u8) ?usize {
    if (type_builtin.lookupBuiltinByName(name)) |spec| {
        return switch (spec.category) {
            .Bool, .Address, .Integer => 1,
            else => null,
        };
    }
    return if (type_builtin.parseFixedBytesName(name) != null) 1 else null;
}

fn resultDynamicArrayBuiltinNameSupported(name: []const u8) bool {
    if (type_builtin.lookupBuiltinByName(name)) |spec| {
        return spec.id == .u256 or spec.id == .bool or spec.id == .address;
    }
    return type_builtin.parseFixedBytesName(name) != null;
}

fn abiNamedScalarSupported(name: []const u8) bool {
    if (type_builtin.lookupBuiltinByName(name)) |spec| {
        return switch (spec.category) {
            .Bool, .Address, .Integer => true,
            else => false,
        };
    }
    return type_builtin.parseFixedBytesName(name) != null;
}

fn runtimeDecodeBuiltinNameSupported(name: []const u8, allow_top_level_dynamic: bool) bool {
    if (type_builtin.lookupBuiltinByName(name)) |spec| {
        return switch (spec.category) {
            .Bool, .Address, .Integer => true,
            .String, .Bytes => allow_top_level_dynamic,
            else => false,
        };
    }
    return type_builtin.parseFixedBytesName(name) != null;
}

fn dynamicBytesBuiltinName(name: []const u8) bool {
    const spec = type_builtin.lookupBuiltinByName(name) orelse return false;
    return spec.id == .string or spec.id == .bytes;
}

fn integerSpec(integer: ora_types.IntegerType) ?type_builtin.BuiltinTypeSpec {
    return integer.builtinSpec();
}

test "ABI policy owns public return ABI marker names" {
    const u256_ty: Type = .{ .integer = .{ .bits = 256, .signed = false, .spelling = "u256" } };
    const tuple_elems = [_]Type{u256_ty};
    const tuple_ty: Type = .{ .tuple = &tuple_elems };
    const empty_struct_fields = [_]ora_types.semantic.AnonymousStructField{};
    const anonymous_struct_ty: Type = .{ .anonymous_struct = .{ .fields = &empty_struct_fields } };

    try std.testing.expectEqualStrings("void", publicReturnAbiTypeName(.void) orelse return error.TestUnexpectedResult);
    try std.testing.expectEqualStrings("tuple", publicReturnAbiTypeName(tuple_ty) orelse return error.TestUnexpectedResult);
    try std.testing.expectEqualStrings("struct", publicReturnAbiTypeName(anonymous_struct_ty) orelse return error.TestUnexpectedResult);
    try std.testing.expectEqualStrings("uint256", publicReturnAbiTypeName(.{ .bitfield = .{ .name = "Flags" } }) orelse return error.TestUnexpectedResult);
    try std.testing.expectEqualStrings("tuple", publicReturnAbiTypeName(.{ .struct_ = .{ .name = "S" } }) orelse return error.TestUnexpectedResult);
    try std.testing.expect(publicReturnAbiTypeName(u256_ty) == null);
}
