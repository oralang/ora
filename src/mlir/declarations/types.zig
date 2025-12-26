// ============================================================================
// Declaration Lowering - Types (Struct/Enum)
// ============================================================================

const std = @import("std");
const c = @import("mlir_c_api").c;
const lib = @import("ora_lib");
const h = @import("../helpers.zig");
const constants = @import("../lower.zig");
const DeclarationLowerer = @import("mod.zig").DeclarationLowerer;
const helpers = @import("helpers.zig");
const log = @import("log");

/// Lower struct declarations with type definitions and field information (Requirements 7.1)
pub fn lowerStruct(self: *const DeclarationLowerer, struct_decl: *const lib.ast.StructDeclNode) c.MlirOperation {
    // create ora.struct.decl operation
    var state = h.opState("ora.struct.decl", helpers.createFileLocation(self, struct_decl.span));

    // collect struct attributes
    var attributes = std.ArrayList(c.MlirNamedAttribute){};
    defer attributes.deinit(std.heap.page_allocator);

    // add struct name
    const name_ref = c.mlirStringRefCreate(struct_decl.name.ptr, struct_decl.name.len);
    const name_attr = c.mlirStringAttrGet(self.ctx, name_ref);
    const name_id = h.identifier(self.ctx, "sym_name");
    attributes.append(std.heap.page_allocator, c.mlirNamedAttributeGet(name_id, name_attr)) catch {};

    // create field information as attributes
    var field_names = std.ArrayList(c.MlirAttribute){};
    defer field_names.deinit(std.heap.page_allocator);
    var field_types = std.ArrayList(c.MlirAttribute){};
    defer field_types.deinit(std.heap.page_allocator);

    for (struct_decl.fields) |field| {
        // add field name
        const field_name_ref = c.mlirStringRefCreate(field.name.ptr, field.name.len);
        const field_name_attr = c.mlirStringAttrGet(self.ctx, field_name_ref);
        field_names.append(std.heap.page_allocator, field_name_attr) catch {};

        // add field type
        const field_type = self.type_mapper.toMlirType(field.type_info);
        const field_type_attr = c.mlirTypeAttrGet(field_type);
        field_types.append(std.heap.page_allocator, field_type_attr) catch {};
    }

    // add field names array attribute
    if (field_names.items.len > 0) {
        const field_names_array = c.mlirArrayAttrGet(self.ctx, @intCast(field_names.items.len), field_names.items.ptr);
        const field_names_id = h.identifier(self.ctx, "ora.field_names");
        attributes.append(std.heap.page_allocator, c.mlirNamedAttributeGet(field_names_id, field_names_array)) catch {};

        // add field types array attribute
        const field_types_array = c.mlirArrayAttrGet(self.ctx, @intCast(field_types.items.len), field_types.items.ptr);
        const field_types_id = h.identifier(self.ctx, "ora.field_types");
        attributes.append(std.heap.page_allocator, c.mlirNamedAttributeGet(field_types_id, field_types_array)) catch {};
    }

    // add struct declaration marker
    const struct_decl_attr = c.mlirBoolAttrGet(self.ctx, 1);
    const struct_decl_id = h.identifier(self.ctx, "ora.struct_decl");
    attributes.append(std.heap.page_allocator, c.mlirNamedAttributeGet(struct_decl_id, struct_decl_attr)) catch {};

    // apply all attributes
    c.mlirOperationStateAddAttributes(&state, @intCast(attributes.items.len), attributes.items.ptr);

    // create the struct type and add it as a result
    const struct_type = self.createStructType(struct_decl);
    c.mlirOperationStateAddResults(&state, 1, @ptrCast(&struct_type));

    return c.mlirOperationCreate(&state);
}

/// Lower enum declarations with enum type definitions and variant information (Requirements 7.2)
pub fn lowerEnum(self: *const DeclarationLowerer, enum_decl: *const lib.ast.EnumDeclNode) c.MlirOperation {
    // create ora.enum.decl operation
    var state = h.opState("ora.enum.decl", helpers.createFileLocation(self, enum_decl.span));

    // collect enum attributes
    var attributes = std.ArrayList(c.MlirNamedAttribute){};
    defer attributes.deinit(std.heap.page_allocator);

    // add enum name
    const name_ref = c.mlirStringRefCreate(enum_decl.name.ptr, enum_decl.name.len);
    const name_attr = c.mlirStringAttrGet(self.ctx, name_ref);
    const name_id = h.identifier(self.ctx, "sym_name");
    attributes.append(std.heap.page_allocator, c.mlirNamedAttributeGet(name_id, name_attr)) catch {};

    // add underlying type information
    const underlying_type = if (enum_decl.underlying_type_info) |type_info|
        self.type_mapper.toMlirType(type_info)
    else
        c.mlirIntegerTypeGet(self.ctx, 32); // Default to i32
    const underlying_type_attr = c.mlirTypeAttrGet(underlying_type);
    const underlying_type_id = h.identifier(self.ctx, "ora.underlying_type");
    attributes.append(std.heap.page_allocator, c.mlirNamedAttributeGet(underlying_type_id, underlying_type_attr)) catch {};

    // create variant information as attributes
    var variant_names = std.ArrayList(c.MlirAttribute){};
    defer variant_names.deinit(std.heap.page_allocator);
    var variant_values = std.ArrayList(c.MlirAttribute){};
    defer variant_values.deinit(std.heap.page_allocator);

    for (enum_decl.variants, 0..) |variant, i| {
        // add variant name
        const variant_name_ref = c.mlirStringRefCreate(variant.name.ptr, variant.name.len);
        const variant_name_attr = c.mlirStringAttrGet(self.ctx, variant_name_ref);
        variant_names.append(std.heap.page_allocator, variant_name_attr) catch {};

        // add variant value (use resolved value if available, otherwise use index)
        const variant_value = if (variant.resolved_value) |resolved|
            @as(i64, @intCast(resolved))
        else
            @as(i64, @intCast(i));
        const variant_value_attr = c.mlirIntegerAttrGet(underlying_type, variant_value);
        variant_values.append(std.heap.page_allocator, variant_value_attr) catch {};
    }

    // add variant names array attribute
    if (variant_names.items.len > 0) {
        const variant_names_array = c.mlirArrayAttrGet(self.ctx, @intCast(variant_names.items.len), variant_names.items.ptr);
        const variant_names_id = h.identifier(self.ctx, "ora.variant_names");
        attributes.append(std.heap.page_allocator, c.mlirNamedAttributeGet(variant_names_id, variant_names_array)) catch {};

        // add variant values array attribute
        const variant_values_array = c.mlirArrayAttrGet(self.ctx, @intCast(variant_values.items.len), variant_values.items.ptr);
        const variant_values_id = h.identifier(self.ctx, "ora.variant_values");
        attributes.append(std.heap.page_allocator, c.mlirNamedAttributeGet(variant_values_id, variant_values_array)) catch {};
    }

    // add explicit values flag
    const has_explicit_values_attr = c.mlirBoolAttrGet(self.ctx, if (enum_decl.has_explicit_values) 1 else 0);
    const has_explicit_values_id = h.identifier(self.ctx, "ora.has_explicit_values");
    attributes.append(std.heap.page_allocator, c.mlirNamedAttributeGet(has_explicit_values_id, has_explicit_values_attr)) catch {};

    // add enum declaration marker
    const enum_decl_attr = c.mlirBoolAttrGet(self.ctx, 1);
    const enum_decl_id = h.identifier(self.ctx, "ora.enum_decl");
    attributes.append(std.heap.page_allocator, c.mlirNamedAttributeGet(enum_decl_id, enum_decl_attr)) catch {};

    // apply all attributes
    c.mlirOperationStateAddAttributes(&state, @intCast(attributes.items.len), attributes.items.ptr);

    // create the enum type and add it as a result
    const enum_type = self.createEnumType(enum_decl);
    c.mlirOperationStateAddResults(&state, 1, @ptrCast(&enum_type));

    return c.mlirOperationCreate(&state);
}

/// Create struct type from struct declaration
pub fn createStructType(self: *const DeclarationLowerer, struct_decl: *const lib.ast.StructDeclNode) c.MlirType {
    // create the actual !ora.struct<struct_name> type
    // this is the correct way to create struct types - using the struct name, not the first field type
    const struct_name_ref = h.strRef(struct_decl.name);
    const struct_type = c.oraStructTypeGet(self.ctx, struct_name_ref);

    if (struct_type.ptr != null) {
        return struct_type;
    }

    // fallback: should not happen if struct type creation works correctly
    log.warn("Failed to create struct type '{s}', using i256 fallback\n", .{struct_decl.name});
    return c.mlirIntegerTypeGet(self.ctx, constants.DEFAULT_INTEGER_BITS);
}

/// Create enum type from enum declaration
pub fn createEnumType(self: *const DeclarationLowerer, enum_decl: *const lib.ast.EnumDeclNode) c.MlirType {
    // enum type uses underlying integer representation
    // migration to !ora.enum<name, repr> planned for TableGen integration
    return if (enum_decl.underlying_type_info) |type_info|
        self.type_mapper.toMlirType(type_info)
    else
        c.mlirIntegerTypeGet(self.ctx, 32); // Default to i32
}

/// Create error type from error declaration
pub fn createErrorType(self: *const DeclarationLowerer, error_decl: *const lib.ast.Statements.ErrorDeclNode) c.MlirType {
    // error type uses i32 representation for error codes
    // migration to !ora.error<T> planned for TableGen integration
    _ = error_decl;
    return c.mlirIntegerTypeGet(self.ctx, 32); // Placeholder error type
}
