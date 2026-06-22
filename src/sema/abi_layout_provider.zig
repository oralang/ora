const std = @import("std");
const ast = @import("../ast/mod.zig");
const abi_layout_context = @import("../abi/layout_context.zig");
const abi_policy = @import("../abi/policy.zig");
const model = @import("model.zig");
const type_descriptors = @import("type_descriptors.zig");

const Type = model.Type;

pub fn abiLayoutProvider(
    file: *const ast.AstFile,
    item_index: *const model.ItemIndexResult,
    typecheck_result: *const model.TypeCheckResult,
) abi_layout_context.Provider {
    return .{
        .file = file,
        .index_context = item_index,
        .facts_context = typecheck_result,
        .vtable = &vtable,
    };
}

const vtable = abi_layout_context.Provider.VTable{
    .resolve_type_expr = resolveTypeExpr,
    .named_type_kind = namedTypeKind,
    .enum_info = enumInfo,
    .enum_variant_count = enumVariantCount,
    .bitfield_base_type = bitfieldBaseType,
    .type_alias_target = typeAliasTarget,
    .struct_fields = structFields,
    .contract_field_types = contractFieldTypes,
    .error_payload_types = errorPayloadTypes,
};

fn itemIndex(provider: *const abi_layout_context.Provider) *const model.ItemIndexResult {
    return @ptrCast(@alignCast(provider.index_context));
}

fn typecheck(provider: *const abi_layout_context.Provider) *const model.TypeCheckResult {
    return @ptrCast(@alignCast(provider.facts_context));
}

fn resolveTypeExpr(provider: *const abi_layout_context.Provider, allocator: std.mem.Allocator, type_expr_id: ast.TypeExprId) anyerror!Type {
    return type_descriptors.descriptorFromTypeExpr(allocator, provider.file, itemIndex(provider), type_expr_id);
}

fn namedTypeKind(provider: *const abi_layout_context.Provider, name: []const u8) abi_policy.NamedTypeKind {
    const tc = typecheck(provider);
    if (tc.instantiatedEnumByName(name) != null) return .enum_;
    if (tc.instantiatedBitfieldByName(name) != null) return .bitfield;
    if (tc.instantiatedStructByName(name) != null) return .struct_;
    const item_id = itemIndex(provider).lookup(name) orelse return .none;
    return switch (provider.file.item(item_id).*) {
        .Enum => .enum_,
        .Bitfield => .bitfield,
        .Struct => .struct_,
        .Contract => .contract,
        .ErrorDecl => .error_decl,
        .TypeAlias => .type_alias,
        else => .none,
    };
}

fn enumInfo(provider: *const abi_layout_context.Provider, allocator: std.mem.Allocator, name: []const u8) anyerror!?abi_layout_context.EnumInfo {
    if (typecheck(provider).instantiatedEnumByName(name)) |instantiated| {
        return .{
            .has_payload = instantiatedEnumHasPayload(instantiated),
            .repr_type = instantiated.repr_type,
        };
    }
    const item_id = itemIndex(provider).lookup(name) orelse return null;
    return switch (provider.file.item(item_id).*) {
        .Enum => |enum_item| .{
            .has_payload = enumItemHasPayload(enum_item),
            .repr_type = if (enum_item.base_type) |base_type|
                try resolveTypeExpr(provider, allocator, base_type)
            else
                null,
        },
        else => null,
    };
}

fn enumVariantCount(provider: *const abi_layout_context.Provider, name: []const u8) ?usize {
    if (typecheck(provider).instantiatedEnumByName(name)) |instantiated| return instantiated.variants.len;
    const item_id = itemIndex(provider).lookup(name) orelse return null;
    return switch (provider.file.item(item_id).*) {
        .Enum => |enum_item| enum_item.variants.len,
        else => null,
    };
}

fn bitfieldBaseType(provider: *const abi_layout_context.Provider, allocator: std.mem.Allocator, name: []const u8) anyerror!?Type {
    if (typecheck(provider).instantiatedBitfieldByName(name)) |bitfield| {
        return bitfield.base_type;
    }
    const item_id = itemIndex(provider).lookup(name) orelse return null;
    return switch (provider.file.item(item_id).*) {
        .Bitfield => |bitfield| if (bitfield.base_type) |base_type|
            try resolveTypeExpr(provider, allocator, base_type)
        else
            null,
        else => null,
    };
}

fn typeAliasTarget(provider: *const abi_layout_context.Provider, allocator: std.mem.Allocator, name: []const u8) anyerror!?Type {
    const item_id = itemIndex(provider).lookup(name) orelse return null;
    return switch (provider.file.item(item_id).*) {
        .TypeAlias => |type_alias| try resolveTypeExpr(provider, allocator, type_alias.target_type),
        else => null,
    };
}

fn structFields(provider: *const abi_layout_context.Provider, allocator: std.mem.Allocator, name: []const u8) anyerror!?[]const model.AnonymousStructField {
    if (typecheck(provider).instantiatedStructByName(name)) |instantiated| {
        const fields = try allocator.alloc(model.AnonymousStructField, instantiated.fields.len);
        for (instantiated.fields, 0..) |field, index| {
            fields[index] = .{
                .name = field.name,
                .ty = field.ty,
            };
        }
        return fields;
    }
    const item_id = itemIndex(provider).lookup(name) orelse return null;
    const struct_item = switch (provider.file.item(item_id).*) {
        .Struct => |struct_item| struct_item,
        else => return null,
    };
    const fields = try allocator.alloc(model.AnonymousStructField, struct_item.fields.len);
    for (struct_item.fields, 0..) |field, index| {
        fields[index] = .{
            .name = field.name,
            .ty = try resolveTypeExpr(provider, allocator, field.type_expr),
        };
    }
    return fields;
}

fn contractFieldTypes(provider: *const abi_layout_context.Provider, allocator: std.mem.Allocator, name: []const u8) anyerror!?[]const Type {
    const item_id = itemIndex(provider).lookup(name) orelse return null;
    const contract_item = switch (provider.file.item(item_id).*) {
        .Contract => |contract_item| contract_item,
        else => return null,
    };
    var fields: std.ArrayList(Type) = .empty;
    for (contract_item.members) |member_id| {
        switch (provider.file.item(member_id).*) {
            .Field => |field| {
                const type_expr = field.type_expr orelse return null;
                try fields.append(allocator, try resolveTypeExpr(provider, allocator, type_expr));
            },
            else => {},
        }
    }
    return try fields.toOwnedSlice(allocator);
}

fn errorPayloadTypes(provider: *const abi_layout_context.Provider, allocator: std.mem.Allocator, name: []const u8) anyerror!?[]const Type {
    const item_id = itemIndex(provider).lookup(name) orelse return null;
    const error_decl = switch (provider.file.item(item_id).*) {
        .ErrorDecl => |error_decl| error_decl,
        else => return null,
    };
    const payloads = try allocator.alloc(Type, error_decl.parameters.len);
    const tc = typecheck(provider);
    for (error_decl.parameters, 0..) |parameter, index| {
        payloads[index] = tc.pattern_types[parameter.pattern.index()].type;
    }
    return payloads;
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

fn instantiatedEnumHasPayload(instantiated: model.InstantiatedEnum) bool {
    for (instantiated.variants) |variant| {
        if (variant.payload_type != null) return true;
    }
    return false;
}
