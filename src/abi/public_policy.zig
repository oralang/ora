const std = @import("std");
const ast = @import("../ast/mod.zig");
const sema_model = @import("../sema/model.zig");
const type_descriptors = @import("../sema/type_descriptors.zig");
const type_builtin = @import("ora_types").builtin;

const Type = sema_model.Type;

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

pub const ResultCarrierPlan = struct {
    mode: ResultInputMode,
    payload: Type,
    err: ?Type = null,
};

pub fn Policy(comptime Provider: type) type {
    return struct {
        allocator: std.mem.Allocator,
        file: *const ast.AstFile,
        item_index: *const sema_model.ItemIndexResult,
        provider: Provider,

        const Self = @This();

        pub fn supportsType(self: *const Self, ty: Type, position: Position) bool {
            return switch (ty) {
                .unknown => true,
                .void => position == .output,
                .bool, .address, .string, .bytes, .fixed_bytes, .integer, .bitfield => true,
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

        fn defaultStaticWordCount(self: *const Self, ty: Type) ?usize {
            return switch (ty) {
                .bool, .address, .fixed_bytes, .bitfield => 1,
                .integer => |integer| if (integerSpec(integer) != null) 1 else null,
                .enum_ => |named| if (self.provider.enumHasPayload(named.name)) null else 1,
                .refinement => |refinement| self.staticWordCount(refinement.base_type.*),
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
                    const item_id = self.item_index.lookup(named.name) orelse break :blk null;
                    break :blk switch (self.file.item(item_id).*) {
                        .Enum => if (self.provider.enumHasPayload(named.name)) null else 1,
                        .Bitfield => 1,
                        .Struct => self.staticWordCountForStructDecl(named.name),
                        .Contract => self.staticWordCountForContractDecl(named.name),
                        .ErrorDecl => |error_decl| blk2: {
                            var total: usize = 0;
                            for (error_decl.parameters) |parameter| {
                                total += self.staticWordCount(self.provider.patternType(parameter.pattern)) orelse break :blk2 null;
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
            const item_id = self.item_index.lookup(name) orelse return true;
            return switch (self.file.item(item_id).*) {
                .ErrorDecl => |error_decl| error_decl.parameters.len != 0,
                else => true,
            };
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
                    const item_id = self.item_index.lookup(named.name) orelse break :blk false;
                    break :blk switch (self.file.item(item_id).*) {
                        .ErrorDecl => |error_decl| error_blk: {
                            for (error_decl.parameters) |parameter| {
                                if (!self.resultInputCarrierShapeSupported(self.provider.patternType(parameter.pattern))) break :error_blk false;
                            }
                            break :error_blk true;
                        },
                        else => false,
                    };
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
                    const item_id = self.item_index.lookup(named.name) orelse break :blk false;
                    break :blk switch (self.file.item(item_id).*) {
                        .TypeAlias => |type_alias| {
                            const target = type_descriptors.descriptorFromTypeExpr(self.allocator, self.file, self.item_index, type_alias.target_type) catch break :blk false;
                            break :blk self.resultInputDynamicArrayElementSupported(target);
                        },
                        else => false,
                    };
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
            if (self.item_index.lookup(error_name)) |item_id| {
                return self.file.item(item_id).* == .ErrorDecl;
            }
            return false;
        }

        fn staticWordCountForNamedStruct(self: *const Self, name: []const u8) ?usize {
            const item_id = self.item_index.lookup(name) orelse return null;
            return switch (self.file.item(item_id).*) {
                .Struct => self.staticWordCountForStructDecl(name),
                .Contract => self.staticWordCountForContractDecl(name),
                else => null,
            };
        }

        fn staticWordCountForStructDecl(self: *const Self, name: []const u8) ?usize {
            const item_id = self.item_index.lookup(name) orelse return null;
            const struct_item = switch (self.file.item(item_id).*) {
                .Struct => |struct_item| struct_item,
                else => return null,
            };
            var total: usize = 0;
            for (struct_item.fields) |field| {
                const field_type = type_descriptors.descriptorFromTypeExpr(self.allocator, self.file, self.item_index, field.type_expr) catch return null;
                total += self.staticWordCount(field_type) orelse return null;
            }
            return total;
        }

        fn staticWordCountForContractDecl(self: *const Self, name: []const u8) ?usize {
            const item_id = self.item_index.lookup(name) orelse return null;
            const contract_item = switch (self.file.item(item_id).*) {
                .Contract => |contract_item| contract_item,
                else => return null,
            };
            var total: usize = 0;
            for (contract_item.members) |member_id| {
                switch (self.file.item(member_id).*) {
                    .Field => |field| {
                        const type_expr = field.type_expr orelse return null;
                        const field_type = type_descriptors.descriptorFromTypeExpr(self.allocator, self.file, self.item_index, type_expr) catch return null;
                        total += self.staticWordCount(field_type) orelse return null;
                    },
                    else => {},
                }
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

fn integerSpec(integer: sema_model.IntegerType) ?type_builtin.BuiltinTypeSpec {
    if (integer.signed) |signed| {
        if (integer.bits) |bits| {
            return type_builtin.lookupIntegerBuiltin(signed, bits);
        }
    }
    const spelling = integer.spelling orelse return null;
    return type_builtin.parseIntegerBuiltin(spelling);
}
