// ============================================================================
// Declaration Lowering - Logs and Errors
// ============================================================================

const std = @import("std");
const c = @import("mlir_c_api").c;
const lib = @import("ora_lib");
const h = @import("../helpers.zig");
const DeclarationLowerer = @import("mod.zig").DeclarationLowerer;
const helpers = @import("helpers.zig");

/// Lower log declarations with event type definitions and indexed field information (Requirements 7.3)
pub fn lowerLogDecl(self: *const DeclarationLowerer, log_decl: *const lib.ast.LogDeclNode) c.MlirOperation {
    // Create ora.log.decl operation
    var state = h.opState("ora.log.decl", helpers.createFileLocation(self, log_decl.span));

    // Collect log attributes
    var attributes = std.ArrayList(c.MlirNamedAttribute){};
    defer attributes.deinit(std.heap.page_allocator);

    // Add log name
    const name_ref = c.mlirStringRefCreate(log_decl.name.ptr, log_decl.name.len);
    const name_attr = c.mlirStringAttrGet(self.ctx, name_ref);
    const name_id = h.identifier(self.ctx, "sym_name");
    attributes.append(std.heap.page_allocator, c.mlirNamedAttributeGet(name_id, name_attr)) catch {};

    // Create field information as attributes
    var field_names = std.ArrayList(c.MlirAttribute){};
    defer field_names.deinit(std.heap.page_allocator);
    var field_types = std.ArrayList(c.MlirAttribute){};
    defer field_types.deinit(std.heap.page_allocator);
    var field_indexed = std.ArrayList(c.MlirAttribute){};
    defer field_indexed.deinit(std.heap.page_allocator);

    for (log_decl.fields) |field| {
        // Add field name
        const field_name_ref = c.mlirStringRefCreate(field.name.ptr, field.name.len);
        const field_name_attr = c.mlirStringAttrGet(self.ctx, field_name_ref);
        field_names.append(std.heap.page_allocator, field_name_attr) catch {};

        // Add field type
        const field_type = self.type_mapper.toMlirType(field.type_info);
        const field_type_attr = c.mlirTypeAttrGet(field_type);
        field_types.append(std.heap.page_allocator, field_type_attr) catch {};

        // Add indexed flag
        const indexed_attr = c.mlirBoolAttrGet(self.ctx, if (field.indexed) 1 else 0);
        field_indexed.append(std.heap.page_allocator, indexed_attr) catch {};
    }

    // Add field arrays as attributes
    if (field_names.items.len > 0) {
        const field_names_array = c.mlirArrayAttrGet(self.ctx, @intCast(field_names.items.len), field_names.items.ptr);
        const field_names_id = h.identifier(self.ctx, "ora.field_names");
        attributes.append(std.heap.page_allocator, c.mlirNamedAttributeGet(field_names_id, field_names_array)) catch {};

        const field_types_array = c.mlirArrayAttrGet(self.ctx, @intCast(field_types.items.len), field_types.items.ptr);
        const field_types_id = h.identifier(self.ctx, "ora.field_types");
        attributes.append(std.heap.page_allocator, c.mlirNamedAttributeGet(field_types_id, field_types_array)) catch {};

        const field_indexed_array = c.mlirArrayAttrGet(self.ctx, @intCast(field_indexed.items.len), field_indexed.items.ptr);
        const field_indexed_id = h.identifier(self.ctx, "ora.field_indexed");
        attributes.append(std.heap.page_allocator, c.mlirNamedAttributeGet(field_indexed_id, field_indexed_array)) catch {};
    }

    // Add log declaration marker
    const log_decl_attr = c.mlirBoolAttrGet(self.ctx, 1);
    const log_decl_id = h.identifier(self.ctx, "ora.log_decl");
    attributes.append(std.heap.page_allocator, c.mlirNamedAttributeGet(log_decl_id, log_decl_attr)) catch {};

    // Apply all attributes
    c.mlirOperationStateAddAttributes(&state, @intCast(attributes.items.len), attributes.items.ptr);

    return c.mlirOperationCreate(&state);
}

/// Lower error declarations with error type definitions (Requirements 7.4)
pub fn lowerErrorDecl(self: *const DeclarationLowerer, error_decl: *const lib.ast.Statements.ErrorDeclNode) c.MlirOperation {
    // Create ora.error.decl operation
    var state = h.opState("ora.error.decl", helpers.createFileLocation(self, error_decl.span));

    // Collect error attributes
    var attributes = std.ArrayList(c.MlirNamedAttribute){};
    defer attributes.deinit(std.heap.page_allocator);

    // Add error name
    const name_ref = c.mlirStringRefCreate(error_decl.name.ptr, error_decl.name.len);
    const name_attr = c.mlirStringAttrGet(self.ctx, name_ref);
    const name_id = h.identifier(self.ctx, "sym_name");
    attributes.append(std.heap.page_allocator, c.mlirNamedAttributeGet(name_id, name_attr)) catch {};

    // Add error parameters if present
    if (error_decl.parameters) |params| {
        var param_names = std.ArrayList(c.MlirAttribute){};
        defer param_names.deinit(std.heap.page_allocator);
        var param_types = std.ArrayList(c.MlirAttribute){};
        defer param_types.deinit(std.heap.page_allocator);

        for (params) |param| {
            // Add parameter name
            const param_name_ref = c.mlirStringRefCreate(param.name.ptr, param.name.len);
            const param_name_attr = c.mlirStringAttrGet(self.ctx, param_name_ref);
            param_names.append(std.heap.page_allocator, param_name_attr) catch {};

            // Add parameter type
            const param_type = self.type_mapper.toMlirType(param.type_info);
            const param_type_attr = c.mlirTypeAttrGet(param_type);
            param_types.append(std.heap.page_allocator, param_type_attr) catch {};
        }

        // Add parameter arrays as attributes
        if (param_names.items.len > 0) {
            const param_names_array = c.mlirArrayAttrGet(self.ctx, @intCast(param_names.items.len), param_names.items.ptr);
            const param_names_id = h.identifier(self.ctx, "ora.param_names");
            attributes.append(std.heap.page_allocator, c.mlirNamedAttributeGet(param_names_id, param_names_array)) catch {};

            const param_types_array = c.mlirArrayAttrGet(self.ctx, @intCast(param_types.items.len), param_types.items.ptr);
            const param_types_id = h.identifier(self.ctx, "ora.param_types");
            attributes.append(std.heap.page_allocator, c.mlirNamedAttributeGet(param_types_id, param_types_array)) catch {};
        }
    }

    // Add error declaration marker
    const error_decl_attr = c.mlirBoolAttrGet(self.ctx, 1);
    const error_decl_id = h.identifier(self.ctx, "ora.error_decl");
    attributes.append(std.heap.page_allocator, c.mlirNamedAttributeGet(error_decl_id, error_decl_attr)) catch {};

    // Apply all attributes
    c.mlirOperationStateAddAttributes(&state, @intCast(attributes.items.len), attributes.items.ptr);

    // Create the error type and add it as a result
    const error_type = self.createErrorType(error_decl);
    c.mlirOperationStateAddResults(&state, 1, @ptrCast(&error_type));

    return c.mlirOperationCreate(&state);
}
