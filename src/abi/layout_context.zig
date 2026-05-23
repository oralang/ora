const std = @import("std");
const ast = @import("../ast/mod.zig");
const sema = @import("../sema/mod.zig");
const sema_model = @import("../sema/model.zig");
const type_descriptors = @import("../sema/type_descriptors.zig");
const abi_layout = @import("layout.zig");

pub const ResultInputMode = enum {
    none,
    narrow_payloadless,
    wide_payloadless,
    wide_single_error,
};

pub const ResultCarrierPlan = struct {
    mode: ResultInputMode,
    payload: sema.Type,
    err: ?sema.Type = null,
};

pub const LayoutContext = struct {
    allocator: std.mem.Allocator,
    file: *const ast.AstFile,
    item_index: *const sema.ItemIndexResult,
    typecheck: *const sema.TypeCheckResult,

    pub fn canonicalAbiTypeForType(self: *const LayoutContext, ty: sema.Type) anyerror![]const u8 {
        var layout = try self.layoutForType(ty);
        defer layout.deinit(self.allocator);
        return abi_layout.canonicalAbiType(self.allocator, layout);
    }

    pub fn canonicalAbiTypeForTypeExpr(self: *const LayoutContext, type_expr_id: ast.TypeExprId) anyerror![]const u8 {
        var arena = std.heap.ArenaAllocator.init(self.allocator);
        defer arena.deinit();
        const ty = try type_descriptors.descriptorFromTypeExpr(arena.allocator(), self.file, self.item_index, type_expr_id);
        return self.canonicalAbiTypeForType(ty);
    }

    pub fn staticWordCountForType(self: *const LayoutContext, ty: sema.Type) ?usize {
        var layout = self.layoutForType(ty) catch return null;
        defer layout.deinit(self.allocator);
        return layout.staticWordCount();
    }

    pub fn staticWordCountForTypeExpr(self: *const LayoutContext, type_expr_id: ast.TypeExprId) ?usize {
        var arena = std.heap.ArenaAllocator.init(self.allocator);
        defer arena.deinit();
        const ty = type_descriptors.descriptorFromTypeExpr(arena.allocator(), self.file, self.item_index, type_expr_id) catch return null;
        return self.staticWordCountForType(ty);
    }

    pub fn publicResultInputMode(self: *const LayoutContext, ty: sema.Type) ResultInputMode {
        const plan = self.planResultCarrier(ty) orelse return .none;
        return plan.mode;
    }

    pub fn planResultCarrier(self: *const LayoutContext, ty: sema.Type) ?ResultCarrierPlan {
        const error_union = switch (ty) {
            .error_union => |error_union| error_union,
            else => return null,
        };
        if (error_union.error_types.len != 1) return null;
        const payload = error_union.payload_type.*;
        if (!self.resultInputCarrierShapeSupported(payload)) return null;

        const err_ty = error_union.error_types[0];
        const payload_words = self.staticWordCountForType(payload);
        if (!self.errorTypeHasPayload(err_ty)) {
            const mode: ResultInputMode = if (payload_words != null and payload_words.? == 1 and self.resultInputPayloadFitsNarrowCarrier(payload) and self.payloadlessErrorHasRuntimeId(err_ty))
                .narrow_payloadless
            else
                .wide_payloadless;
            return .{ .mode = mode, .payload = payload };
        }

        if (!self.resultInputCarrierShapeSupported(err_ty)) return null;
        if (payload_words != null and payload_words.? == 1) {
            const error_words = self.staticWordCountForType(err_ty) orelse return null;
            if (error_words > 1) return null;
        }
        return .{ .mode = .wide_single_error, .payload = payload, .err = err_ty };
    }

    pub fn errorTypeHasPayload(self: *const LayoutContext, ty: sema.Type) bool {
        return switch (ty) {
            .named => |named| blk: {
                const item_id = self.item_index.lookup(named.name) orelse break :blk true;
                break :blk switch (self.file.item(item_id).*) {
                    .ErrorDecl => |error_decl| error_decl.parameters.len != 0,
                    .Struct => |struct_item| struct_item.fields.len != 0,
                    else => true,
                };
            },
            .anonymous_struct => |struct_type| struct_type.fields.len != 0,
            .tuple => |elements| elements.len != 0,
            .struct_ => |named| blk: {
                const count = self.structFieldCount(named.name) orelse break :blk true;
                break :blk count != 0;
            },
            .contract => true,
            .refinement => |refinement| self.errorTypeHasPayload(refinement.base_type.*),
            .unknown, .never, .void, .function, .map, .external_proxy => false,
            else => true,
        };
    }

    pub fn layoutForType(self: *const LayoutContext, ty: sema.Type) anyerror!abi_layout.LayoutNode {
        var arena = std.heap.ArenaAllocator.init(self.allocator);
        defer arena.deinit();
        // The normalized type tree is temporary context-resolved scaffolding.
        // abi_layout.fromType copies it into an owned LayoutNode before the arena dies.
        const normalized = try self.normalizeType(arena.allocator(), ty);
        return abi_layout.fromType(self.allocator, normalized);
    }

    fn normalizeType(self: *const LayoutContext, arena: std.mem.Allocator, ty: sema.Type) anyerror!sema.Type {
        return switch (ty) {
            .void, .bool, .address, .string, .bytes, .fixed_bytes, .integer => ty,
            .enum_ => |named| self.normalizeEnum(arena, named.name),
            .bitfield => |named| self.normalizeBitfield(arena, named.name),
            .named => |named| self.normalizeNamed(arena, named.name),
            .struct_ => |named| self.normalizeStruct(arena, named.name),
            .contract => error.UnsupportedAbiType,
            .refinement => |refinement| self.normalizeType(arena, refinement.base_type.*),
            .error_union => |error_union| self.normalizeErrorUnion(arena, error_union),
            .tuple => |elements| blk: {
                const normalized = try arena.alloc(sema.Type, elements.len);
                for (elements, 0..) |element, index| {
                    normalized[index] = try self.normalizeType(arena, element);
                }
                break :blk .{ .tuple = normalized };
            },
            .anonymous_struct => |struct_type| blk: {
                const fields = try arena.alloc(sema.AnonymousStructField, struct_type.fields.len);
                for (struct_type.fields, 0..) |field, index| {
                    fields[index] = .{
                        .name = field.name,
                        .ty = try self.normalizeType(arena, field.ty),
                    };
                }
                break :blk .{ .anonymous_struct = .{ .fields = fields } };
            },
            .array => |array| .{ .array = .{
                .element_type = try storeType(arena, try self.normalizeType(arena, array.element_type.*)),
                .len = array.len,
            } },
            .slice => |slice| .{ .slice = .{
                .element_type = try storeType(arena, try self.normalizeType(arena, slice.element_type.*)),
            } },
            else => error.UnsupportedAbiType,
        };
    }

    fn normalizeNamed(self: *const LayoutContext, arena: std.mem.Allocator, name: []const u8) anyerror!sema.Type {
        if (abi_layout.parseFixedBytesSpelling(name)) |len| return .{ .fixed_bytes = .{ .len = len, .spelling = name } };
        if (self.typecheck.instantiatedEnumByName(name)) |_| return self.normalizeEnum(arena, name);
        if (self.typecheck.instantiatedBitfieldByName(name)) |_| return self.normalizeBitfield(arena, name);
        if (self.typecheck.instantiatedStructByName(name)) |_| return self.normalizeStruct(arena, name);

        const item_id = self.item_index.lookup(name) orelse return error.UnsupportedAbiType;
        return switch (self.file.item(item_id).*) {
            .Enum => self.normalizeEnum(arena, name),
            .Bitfield => self.normalizeBitfield(arena, name),
            .Struct => self.normalizeStruct(arena, name),
            .ErrorDecl => |error_decl| self.normalizeErrorPayloadTypeFromDecl(arena, error_decl),
            .TypeAlias => |type_alias| self.normalizeType(arena, try type_descriptors.descriptorFromTypeExpr(arena, self.file, self.item_index, type_alias.target_type)),
            else => error.UnsupportedAbiType,
        };
    }

    fn normalizeEnum(self: *const LayoutContext, arena: std.mem.Allocator, name: []const u8) anyerror!sema.Type {
        if (self.typecheck.instantiatedEnumByName(name)) |instantiated| {
            if (instantiatedEnumHasPayload(instantiated)) return error.UnsupportedAbiType;
            if (instantiated.repr_type) |repr| return self.normalizeType(arena, repr);
            // Unlike bitfields, scalar enums have an established default
            // representation in Ora's ABI and HIR lowering: uint256/i256.
            return uintType(256, "uint256");
        }

        const item_id = self.item_index.lookup(name) orelse return error.UnsupportedAbiType;
        return switch (self.file.item(item_id).*) {
            .Enum => |enum_item| blk: {
                if (enumItemHasPayload(enum_item)) break :blk error.UnsupportedAbiType;
                if (enum_item.base_type) |base_type| {
                    const base = try type_descriptors.descriptorFromTypeExpr(arena, self.file, self.item_index, base_type);
                    break :blk try self.normalizeType(arena, base);
                }
                // Unlike bitfields, scalar enums have an established default
                // representation in Ora's ABI and HIR lowering: uint256/i256.
                break :blk uintType(256, "uint256");
            },
            else => error.UnsupportedAbiType,
        };
    }

    fn normalizeBitfield(self: *const LayoutContext, arena: std.mem.Allocator, name: []const u8) anyerror!sema.Type {
        if (self.typecheck.instantiatedBitfieldByName(name)) |bitfield| {
            if (bitfield.base_type) |base_type| return self.normalizeType(arena, base_type);
            // Context-aware bitfield layout must know the declared base type.
            // Falling back to uint256 would silently change the wire ABI.
            return error.UnsupportedAbiType;
        }

        const item_id = self.item_index.lookup(name) orelse return error.UnsupportedAbiType;
        return switch (self.file.item(item_id).*) {
            .Bitfield => |bitfield| blk: {
                if (bitfield.base_type) |base_type| {
                    const base = try type_descriptors.descriptorFromTypeExpr(arena, self.file, self.item_index, base_type);
                    break :blk try self.normalizeType(arena, base);
                }
                break :blk error.UnsupportedAbiType;
            },
            else => error.UnsupportedAbiType,
        };
    }

    fn normalizeStruct(self: *const LayoutContext, arena: std.mem.Allocator, name: []const u8) anyerror!sema.Type {
        if (self.typecheck.instantiatedStructByName(name)) |instantiated| {
            const fields = try arena.alloc(sema.AnonymousStructField, instantiated.fields.len);
            for (instantiated.fields, 0..) |field, index| {
                fields[index] = .{
                    .name = field.name,
                    .ty = try self.normalizeType(arena, field.ty),
                };
            }
            return .{ .anonymous_struct = .{ .fields = fields } };
        }

        const item_id = self.item_index.lookup(name) orelse return error.UnsupportedAbiType;
        return switch (self.file.item(item_id).*) {
            .Struct => |struct_item| blk: {
                const fields = try arena.alloc(sema.AnonymousStructField, struct_item.fields.len);
                for (struct_item.fields, 0..) |field, index| {
                    const field_type = try type_descriptors.descriptorFromTypeExpr(arena, self.file, self.item_index, field.type_expr);
                    fields[index] = .{
                        .name = field.name,
                        .ty = try self.normalizeType(arena, field_type),
                    };
                }
                break :blk .{ .anonymous_struct = .{ .fields = fields } };
            },
            else => error.UnsupportedAbiType,
        };
    }

    fn normalizeErrorUnion(self: *const LayoutContext, arena: std.mem.Allocator, error_union: sema_model.ErrorUnionType) anyerror!sema.Type {
        const plan = self.planResultCarrier(.{ .error_union = error_union }) orelse return error.UnsupportedAbiType;

        const element_count: usize = if (plan.err != null) 3 else 2;
        const elements = try arena.alloc(sema.Type, element_count);
        elements[0] = .bool;
        elements[1] = try self.normalizeType(arena, plan.payload);
        if (plan.err) |err| {
            elements[2] = try self.normalizeType(arena, err);
        }
        return .{ .tuple = elements };
    }

    fn normalizeErrorPayloadTypeFromDecl(self: *const LayoutContext, arena: std.mem.Allocator, error_decl: ast.ErrorDeclItem) anyerror!sema.Type {
        if (error_decl.parameters.len == 0) return uintType(256, "uint256");
        if (error_decl.parameters.len == 1) {
            const param_ty = self.typecheck.pattern_types[error_decl.parameters[0].pattern.index()].type;
            return self.normalizeType(arena, param_ty);
        }

        const elements = try arena.alloc(sema.Type, error_decl.parameters.len);
        for (error_decl.parameters, 0..) |parameter, index| {
            elements[index] = try self.normalizeType(arena, self.typecheck.pattern_types[parameter.pattern.index()].type);
        }
        return .{ .tuple = elements };
    }

    fn resultInputCarrierShapeSupported(self: *const LayoutContext, ty: sema.Type) bool {
        if (self.staticWordCountForType(ty) != null) return true;
        return switch (ty) {
            .bytes, .string => true,
            .slice => |slice| self.resultInputDynamicArrayElementSupported(slice.element_type.*),
            .array => |array| array.len == null and self.resultInputDynamicArrayElementSupported(array.element_type.*),
            .refinement => |refinement| self.resultInputCarrierShapeSupported(refinement.base_type.*),
            else => false,
        };
    }

    fn resultInputDynamicArrayElementSupported(self: *const LayoutContext, ty: sema.Type) bool {
        return switch (ty) {
            .bool, .address, .fixed_bytes, .integer => true,
            .enum_, .bitfield => self.staticWordCountForType(ty) != null,
            .named => |named| blk: {
                if (abi_layout.parseFixedBytesSpelling(named.name) != null) break :blk true;
                const item_id = self.item_index.lookup(named.name) orelse break :blk false;
                break :blk switch (self.file.item(item_id).*) {
                    .Enum, .Bitfield => self.staticWordCountForType(ty) != null,
                    else => false,
                };
            },
            .refinement => |refinement| self.resultInputDynamicArrayElementSupported(refinement.base_type.*),
            else => false,
        };
    }

    fn resultInputPayloadFitsNarrowCarrier(self: *const LayoutContext, ty: sema.Type) bool {
        return switch (ty) {
            .bool, .address => true,
            .integer => |integer| (integer.bits orelse bitsFromIntegerSpelling(integer.spelling) orelse 256) <= 255,
            .refinement => |refinement| self.resultInputPayloadFitsNarrowCarrier(refinement.base_type.*),
            else => false,
        };
    }

    fn payloadlessErrorHasRuntimeId(self: *const LayoutContext, ty: sema.Type) bool {
        const error_name = ty.name() orelse return false;
        if (self.item_index.lookup(error_name)) |item_id| {
            return self.file.item(item_id).* == .ErrorDecl;
        }
        return false;
    }

    fn structFieldCount(self: *const LayoutContext, name: []const u8) ?usize {
        if (self.typecheck.instantiatedStructByName(name)) |instantiated| return instantiated.fields.len;
        const item_id = self.item_index.lookup(name) orelse return null;
        return switch (self.file.item(item_id).*) {
            .Struct => |struct_item| struct_item.fields.len,
            else => null,
        };
    }
};

fn storeType(allocator: std.mem.Allocator, ty: sema.Type) !*const sema.Type {
    const ptr = try allocator.create(sema.Type);
    ptr.* = ty;
    return ptr;
}

fn uintType(bits: u16, spelling: []const u8) sema.Type {
    return .{ .integer = .{ .bits = bits, .signed = false, .spelling = spelling } };
}

fn bitsFromIntegerSpelling(spelling: ?[]const u8) ?u16 {
    const text = spelling orelse return null;
    if (text.len < 2) return null;
    if (text[0] != 'u' and text[0] != 'i') return null;
    return std.fmt.parseUnsigned(u16, text[1..], 10) catch null;
}

fn enumItemHasPayload(enum_item: ast.EnumItem) bool {
    for (enum_item.variants) |variant| {
        switch (variant.payload) {
            .none => {},
            else => return true,
        }
    }
    return false;
}

fn instantiatedEnumHasPayload(instantiated: sema.InstantiatedEnum) bool {
    for (instantiated.variants) |variant| {
        if (variant.payload_type != null) return true;
    }
    return false;
}
