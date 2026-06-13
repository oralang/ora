const std = @import("std");
const ast = @import("../ast/mod.zig");
const ora_types = @import("ora_types");
const abi_layout = @import("layout.zig");
const abi_policy = @import("policy.zig");
const abi_type_names = @import("type_names.zig");

const Type = ora_types.SemanticType;
const AnonymousStructField = ora_types.semantic.AnonymousStructField;

pub const ResultInputMode = abi_policy.ResultInputMode;
pub const ResultCarrierPlan = abi_policy.ResultCarrierPlan;

pub const EnumInfo = struct {
    has_payload: bool,
    repr_type: ?Type = null,
};

pub const Provider = struct {
    file: *const ast.AstFile,
    index_context: *const anyopaque,
    facts_context: *const anyopaque,
    vtable: *const VTable,

    pub const VTable = struct {
        resolve_type_expr: *const fn (*const Provider, std.mem.Allocator, ast.TypeExprId) anyerror!Type,
        named_type_kind: *const fn (*const Provider, []const u8) abi_policy.NamedTypeKind,
        enum_info: *const fn (*const Provider, std.mem.Allocator, []const u8) anyerror!?EnumInfo,
        enum_variant_count: *const fn (*const Provider, []const u8) ?usize,
        bitfield_base_type: *const fn (*const Provider, std.mem.Allocator, []const u8) anyerror!?Type,
        type_alias_target: *const fn (*const Provider, std.mem.Allocator, []const u8) anyerror!?Type,
        struct_fields: *const fn (*const Provider, std.mem.Allocator, []const u8) anyerror!?[]const AnonymousStructField,
        contract_field_types: *const fn (*const Provider, std.mem.Allocator, []const u8) anyerror!?[]const Type,
        error_payload_types: *const fn (*const Provider, std.mem.Allocator, []const u8) anyerror!?[]const Type,
    };

    pub fn resolveTypeExpr(self: *const Provider, allocator: std.mem.Allocator, type_expr_id: ast.TypeExprId) anyerror!Type {
        return self.vtable.resolve_type_expr(self, allocator, type_expr_id);
    }

    pub fn namedTypeKind(self: *const Provider, name: []const u8) abi_policy.NamedTypeKind {
        return self.vtable.named_type_kind(self, name);
    }

    pub fn enumInfo(self: *const Provider, allocator: std.mem.Allocator, name: []const u8) anyerror!?EnumInfo {
        return self.vtable.enum_info(self, allocator, name);
    }

    pub fn enumVariantCount(self: *const Provider, name: []const u8) ?usize {
        return self.vtable.enum_variant_count(self, name);
    }

    pub fn bitfieldBaseType(self: *const Provider, allocator: std.mem.Allocator, name: []const u8) anyerror!?Type {
        return self.vtable.bitfield_base_type(self, allocator, name);
    }

    pub fn typeAliasTarget(self: *const Provider, allocator: std.mem.Allocator, name: []const u8) anyerror!?Type {
        return self.vtable.type_alias_target(self, allocator, name);
    }

    pub fn structFields(self: *const Provider, allocator: std.mem.Allocator, name: []const u8) anyerror!?[]const AnonymousStructField {
        return self.vtable.struct_fields(self, allocator, name);
    }

    pub fn contractFieldTypes(self: *const Provider, allocator: std.mem.Allocator, name: []const u8) anyerror!?[]const Type {
        return self.vtable.contract_field_types(self, allocator, name);
    }

    pub fn errorPayloadTypes(self: *const Provider, allocator: std.mem.Allocator, name: []const u8) anyerror!?[]const Type {
        return self.vtable.error_payload_types(self, allocator, name);
    }
};

pub const LayoutContext = struct {
    allocator: std.mem.Allocator,
    provider: Provider,

    pub fn canonicalAbiTypeForType(self: *const LayoutContext, ty: Type) anyerror![]const u8 {
        var layout = try self.layoutForType(ty);
        defer layout.deinit(self.allocator);
        return abi_layout.canonicalAbiType(self.allocator, layout);
    }

    pub fn canonicalAbiTypeForTypeExpr(self: *const LayoutContext, type_expr_id: ast.TypeExprId) anyerror![]const u8 {
        var arena = std.heap.ArenaAllocator.init(self.allocator);
        defer arena.deinit();
        const ty = try self.provider.resolveTypeExpr(arena.allocator(), type_expr_id);
        return self.canonicalAbiTypeForType(ty);
    }

    pub fn publicReturnAbiTypeForType(self: *const LayoutContext, ty: Type) anyerror![]const u8 {
        if (abi_policy.publicReturnAbiTypeName(ty)) |name| return self.allocator.dupe(u8, name);
        return self.canonicalAbiTypeForType(ty);
    }

    pub fn signatureForMethod(self: *const LayoutContext, name: []const u8, has_self: bool, param_types: []const Type) anyerror![]const u8 {
        var parts: std.ArrayList([]const u8) = .{};
        defer {
            for (parts.items) |part| self.allocator.free(part);
            parts.deinit(self.allocator);
        }

        _ = has_self;
        for (param_types) |param_type| {
            try parts.append(self.allocator, try self.canonicalAbiTypeForType(param_type));
        }

        const joined = try std.mem.join(self.allocator, ",", parts.items);
        defer self.allocator.free(joined);
        return std.fmt.allocPrint(self.allocator, "{s}({s})", .{ name, joined });
    }

    pub fn staticWordCountForType(self: *const LayoutContext, ty: Type) ?usize {
        var layout = self.layoutForType(ty) catch return null;
        defer layout.deinit(self.allocator);
        return layout.staticWordCount();
    }

    pub fn staticWordCountForTypeExpr(self: *const LayoutContext, type_expr_id: ast.TypeExprId) ?usize {
        var arena = std.heap.ArenaAllocator.init(self.allocator);
        defer arena.deinit();
        const ty = self.provider.resolveTypeExpr(arena.allocator(), type_expr_id) catch return null;
        return self.staticWordCountForType(ty);
    }

    pub fn publicResultInputMode(self: *const LayoutContext, ty: Type) ResultInputMode {
        const plan = self.planResultCarrier(ty) orelse return .none;
        return plan.mode;
    }

    pub fn planResultCarrier(self: *const LayoutContext, ty: Type) ?ResultCarrierPlan {
        var arena = std.heap.ArenaAllocator.init(self.allocator);
        defer arena.deinit();
        const policy = self.abiPolicy(arena.allocator());
        return policy.planResultCarrier(ty);
    }

    pub fn errorTypeHasPayload(self: *const LayoutContext, ty: Type) bool {
        var arena = std.heap.ArenaAllocator.init(self.allocator);
        defer arena.deinit();
        return self.errorTypeHasPayloadWithAllocator(arena.allocator(), ty);
    }

    fn errorTypeHasPayloadWithAllocator(self: *const LayoutContext, allocator: std.mem.Allocator, ty: Type) bool {
        return switch (ty) {
            .named => |named| blk: {
                break :blk switch (self.provider.namedTypeKind(named.name)) {
                    .error_decl => blk2: {
                        const payloads = self.provider.errorPayloadTypes(allocator, named.name) catch break :blk2 true;
                        break :blk2 (payloads orelse break :blk2 true).len != 0;
                    },
                    .struct_ => blk2: {
                        const fields = self.provider.structFields(allocator, named.name) catch break :blk2 true;
                        break :blk2 (fields orelse break :blk2 true).len != 0;
                    },
                    else => true,
                };
            },
            .anonymous_struct => |struct_type| struct_type.fields.len != 0,
            .tuple => |elements| elements.len != 0,
            .struct_ => |named| blk: {
                const fields = self.provider.structFields(allocator, named.name) catch break :blk true;
                break :blk (fields orelse break :blk true).len != 0;
            },
            .contract => true,
            .refinement => |refinement| self.errorTypeHasPayloadWithAllocator(allocator, refinement.base_type.*),
            .unknown, .never, .void, .function, .map, .external_proxy => false,
            else => true,
        };
    }

    pub fn enumVariantCount(self: *const LayoutContext, ty: Type) ?usize {
        const unwrapped = unwrapRefinement(ty);
        const name = switch (unwrapped) {
            .enum_ => |named| named.name,
            .named => |named| named.name,
            else => return null,
        };
        return self.provider.enumVariantCount(name);
    }

    pub fn layoutForType(self: *const LayoutContext, ty: Type) anyerror!abi_layout.LayoutNode {
        var arena = std.heap.ArenaAllocator.init(self.allocator);
        defer arena.deinit();
        // The normalized type tree is temporary context-resolved scaffolding.
        // abi_layout.fromType copies it into an owned LayoutNode before the arena dies.
        const normalized = try self.normalizeType(arena.allocator(), ty);
        return abi_layout.fromType(self.allocator, normalized);
    }

    fn normalizeType(self: *const LayoutContext, arena: std.mem.Allocator, ty: Type) anyerror!Type {
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
                const normalized = try arena.alloc(Type, elements.len);
                for (elements, 0..) |element, index| {
                    normalized[index] = try self.normalizeType(arena, element);
                }
                break :blk .{ .tuple = normalized };
            },
            .anonymous_struct => |struct_type| blk: {
                const fields = try arena.alloc(AnonymousStructField, struct_type.fields.len);
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

    fn normalizeNamed(self: *const LayoutContext, arena: std.mem.Allocator, name: []const u8) anyerror!Type {
        if (abi_layout.parseFixedBytesSpelling(name)) |len| return .{ .fixed_bytes = .{ .len = len, .spelling = name } };
        return switch (self.provider.namedTypeKind(name)) {
            .enum_ => self.normalizeEnum(arena, name),
            .bitfield => self.normalizeBitfield(arena, name),
            .struct_ => self.normalizeStruct(arena, name),
            .error_decl => self.normalizeErrorPayloadTypeFromName(arena, name),
            .type_alias => blk: {
                const target = (try self.provider.typeAliasTarget(arena, name)) orelse break :blk error.UnsupportedAbiType;
                break :blk self.normalizeType(arena, target);
            },
            else => error.UnsupportedAbiType,
        };
    }

    fn normalizeEnum(self: *const LayoutContext, arena: std.mem.Allocator, name: []const u8) anyerror!Type {
        const info = (try self.provider.enumInfo(arena, name)) orelse return error.UnsupportedAbiType;
        if (info.has_payload) return error.UnsupportedAbiType;
        if (info.repr_type) |repr| return self.normalizeType(arena, repr);
        // Unlike bitfields, scalar enums have an established default
        // representation in Ora's ABI and HIR lowering: uint256/i256.
        return uintType(256, abi_type_names.builtinAbiName(.u256));
    }

    fn normalizeBitfield(self: *const LayoutContext, arena: std.mem.Allocator, name: []const u8) anyerror!Type {
        const base = (try self.provider.bitfieldBaseType(arena, name)) orelse return error.UnsupportedAbiType;
        return self.normalizeType(arena, base);
    }

    fn normalizeStruct(self: *const LayoutContext, arena: std.mem.Allocator, name: []const u8) anyerror!Type {
        const raw_fields = (try self.provider.structFields(arena, name)) orelse return error.UnsupportedAbiType;
        const fields = try arena.alloc(AnonymousStructField, raw_fields.len);
        for (raw_fields, 0..) |field, index| {
            fields[index] = .{
                .name = field.name,
                .ty = try self.normalizeType(arena, field.ty),
            };
        }
        return .{ .anonymous_struct = .{ .fields = fields } };
    }

    fn normalizeErrorUnion(self: *const LayoutContext, arena: std.mem.Allocator, error_union: anytype) anyerror!Type {
        const plan = self.planResultCarrier(.{ .error_union = error_union }) orelse return error.UnsupportedAbiType;

        const element_count: usize = if (plan.err != null) 3 else 2;
        const elements = try arena.alloc(Type, element_count);
        elements[0] = .bool;
        elements[1] = try self.normalizeType(arena, plan.payload);
        if (plan.err) |err| {
            elements[2] = try self.normalizeType(arena, err);
        }
        return .{ .tuple = elements };
    }

    fn normalizeErrorPayloadTypeFromName(self: *const LayoutContext, arena: std.mem.Allocator, name: []const u8) anyerror!Type {
        const payloads = (try self.provider.errorPayloadTypes(arena, name)) orelse return error.UnsupportedAbiType;
        if (payloads.len == 0) return uintType(256, abi_type_names.builtinAbiName(.u256));
        if (payloads.len == 1) return self.normalizeType(arena, payloads[0]);

        const elements = try arena.alloc(Type, payloads.len);
        for (payloads, 0..) |payload, index| {
            elements[index] = try self.normalizeType(arena, payload);
        }
        return .{ .tuple = elements };
    }

    const AbiPolicyProvider = struct {
        context: *const LayoutContext,
        allocator: std.mem.Allocator,

        pub fn enumHasPayload(self: @This(), name: []const u8) bool {
            const info = (self.context.provider.enumInfo(self.context.allocator, name) catch return false) orelse return false;
            return info.has_payload;
        }

        pub fn staticWordCount(self: @This(), ty: Type) ?usize {
            return self.context.staticWordCountForType(ty);
        }

        pub fn errorTypeHasPayload(self: @This(), ty: Type) bool {
            return self.context.errorTypeHasPayloadWithAllocator(self.allocator, ty);
        }

        pub fn namedTypeKind(self: @This(), name: []const u8) abi_policy.NamedTypeKind {
            return self.context.provider.namedTypeKind(name);
        }

        pub fn typeAliasTarget(self: @This(), name: []const u8) ?Type {
            return self.context.provider.typeAliasTarget(self.allocator, name) catch null;
        }

        pub fn structFieldTypes(self: @This(), name: []const u8) ?[]const Type {
            return self.context.fieldTypesForStruct(self.allocator, name) catch null;
        }

        pub fn contractFieldTypes(self: @This(), name: []const u8) ?[]const Type {
            return self.context.provider.contractFieldTypes(self.allocator, name) catch null;
        }

        pub fn errorPayloadTypes(self: @This(), name: []const u8) ?[]const Type {
            return self.context.provider.errorPayloadTypes(self.allocator, name) catch null;
        }
    };

    fn fieldTypesForStruct(self: *const LayoutContext, allocator: std.mem.Allocator, name: []const u8) !?[]const Type {
        const fields = (try self.provider.structFields(allocator, name)) orelse return null;
        const types = try allocator.alloc(Type, fields.len);
        for (fields, 0..) |field, index| types[index] = field.ty;
        return types;
    }

    fn abiPolicy(self: *const LayoutContext, allocator: std.mem.Allocator) abi_policy.Policy(AbiPolicyProvider) {
        return .{
            .provider = .{ .context = self, .allocator = allocator },
        };
    }
};

fn storeType(allocator: std.mem.Allocator, ty: Type) !*const Type {
    const ptr = try allocator.create(Type);
    ptr.* = ty;
    return ptr;
}

fn uintType(bits: u16, spelling: []const u8) Type {
    return .{ .integer = .{ .bits = bits, .signed = false, .spelling = spelling } };
}

fn unwrapRefinement(ty: Type) Type {
    return switch (ty) {
        .refinement => |refinement| unwrapRefinement(refinement.base_type.*),
        else => ty,
    };
}
