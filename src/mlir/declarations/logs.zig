// ============================================================================
// Declaration Lowering - Logs and Errors
// ============================================================================

const std = @import("std");
const c = @import("mlir_c_api").c;
const lib = @import("ora_lib");
const h = @import("../helpers.zig");
const DeclarationLowerer = @import("mod.zig").DeclarationLowerer;
const helpers = @import("helpers.zig");
const log = @import("log");

/// Lower log declarations with event type definitions and indexed field information (Requirements 7.3)
pub fn lowerLogDecl(self: *const DeclarationLowerer, log_decl: *const lib.ast.LogDeclNode) c.MlirOperation {
    const loc = helpers.createFileLocation(self, log_decl.span);

    if (self.symbol_table) |st| {
        st.log_signatures.put(log_decl.name, log_decl.fields) catch {
            log.warn("Failed to register log signature: {s}\n", .{log_decl.name});
        };
    }

    // collect log attributes
    var attributes = std.ArrayList(c.MlirNamedAttribute){};
    defer attributes.deinit(std.heap.page_allocator);

    // add log name
    const name_ref = c.oraStringRefCreate(log_decl.name.ptr, log_decl.name.len);
    const name_attr = c.oraStringAttrCreate(self.ctx, name_ref);
    const name_id = h.identifier(self.ctx, "sym_name");
    attributes.append(std.heap.page_allocator, c.oraNamedAttributeGet(name_id, name_attr)) catch {};

    // create field information as attributes
    var field_names = std.ArrayList(c.MlirAttribute){};
    defer field_names.deinit(std.heap.page_allocator);
    var field_types = std.ArrayList(c.MlirAttribute){};
    defer field_types.deinit(std.heap.page_allocator);
    var field_indexed = std.ArrayList(c.MlirAttribute){};
    defer field_indexed.deinit(std.heap.page_allocator);

    for (log_decl.fields) |field| {
        // add field name
        const field_name_ref = c.oraStringRefCreate(field.name.ptr, field.name.len);
        const field_name_attr = c.oraStringAttrCreate(self.ctx, field_name_ref);
        field_names.append(std.heap.page_allocator, field_name_attr) catch {};

        // add field type
        const field_type = self.type_mapper.toMlirType(field.type_info);
        const field_type_attr = c.oraTypeAttrCreateFromType(field_type);
        field_types.append(std.heap.page_allocator, field_type_attr) catch {};

        // add indexed flag
        const indexed_attr = h.boolAttr(self.ctx, @as(i32, @intFromBool(field.indexed)));
        field_indexed.append(std.heap.page_allocator, indexed_attr) catch {};
    }

    // add field arrays as attributes
    if (field_names.items.len > 0) {
        const field_names_array = c.oraArrayAttrCreate(self.ctx, @intCast(field_names.items.len), field_names.items.ptr);
        const field_names_id = h.identifier(self.ctx, "ora.field_names");
        attributes.append(std.heap.page_allocator, c.oraNamedAttributeGet(field_names_id, field_names_array)) catch {};

        const field_types_array = c.oraArrayAttrCreate(self.ctx, @intCast(field_types.items.len), field_types.items.ptr);
        const field_types_id = h.identifier(self.ctx, "ora.field_types");
        attributes.append(std.heap.page_allocator, c.oraNamedAttributeGet(field_types_id, field_types_array)) catch {};

        const field_indexed_array = c.oraArrayAttrCreate(self.ctx, @intCast(field_indexed.items.len), field_indexed.items.ptr);
        const field_indexed_id = h.identifier(self.ctx, "ora.field_indexed");
        attributes.append(std.heap.page_allocator, c.oraNamedAttributeGet(field_indexed_id, field_indexed_array)) catch {};
    }

    // add log declaration marker
    const log_decl_attr = h.boolAttr(self.ctx, 1);
    const log_decl_id = h.identifier(self.ctx, "ora.log_decl");
    attributes.append(std.heap.page_allocator, c.oraNamedAttributeGet(log_decl_id, log_decl_attr)) catch {};

    return c.oraLogDeclOpCreate(
        self.ctx,
        loc,
        if (attributes.items.len == 0) null else attributes.items.ptr,
        attributes.items.len,
    );
}

/// Lower error declarations with error type definitions (Requirements 7.4)
pub fn lowerErrorDecl(self: *const DeclarationLowerer, error_decl: *const lib.ast.Statements.ErrorDeclNode) c.MlirOperation {
    const loc = helpers.createFileLocation(self, error_decl.span);

    // collect error attributes
    var attributes = std.ArrayList(c.MlirNamedAttribute){};
    defer attributes.deinit(std.heap.page_allocator);

    // add error name
    const name_ref = c.oraStringRefCreate(error_decl.name.ptr, error_decl.name.len);
    const name_attr = c.oraStringAttrCreate(self.ctx, name_ref);
    const name_id = h.identifier(self.ctx, "sym_name");
    attributes.append(std.heap.page_allocator, c.oraNamedAttributeGet(name_id, name_attr)) catch {};

    // add error parameters if present
    if (error_decl.parameters) |params| {
        var param_names = std.ArrayList(c.MlirAttribute){};
        defer param_names.deinit(std.heap.page_allocator);
        var param_types = std.ArrayList(c.MlirAttribute){};
        defer param_types.deinit(std.heap.page_allocator);

        for (params) |param| {
            // add parameter name
            const param_name_ref = c.oraStringRefCreate(param.name.ptr, param.name.len);
            const param_name_attr = c.oraStringAttrCreate(self.ctx, param_name_ref);
            param_names.append(std.heap.page_allocator, param_name_attr) catch {};

            // add parameter type
            const param_type = self.type_mapper.toMlirType(param.type_info);
            const param_type_attr = c.oraTypeAttrCreateFromType(param_type);
            param_types.append(std.heap.page_allocator, param_type_attr) catch {};
        }

        // add parameter arrays as attributes
        if (param_names.items.len > 0) {
            const param_names_array = c.oraArrayAttrCreate(self.ctx, @intCast(param_names.items.len), param_names.items.ptr);
            const param_names_id = h.identifier(self.ctx, "ora.param_names");
            attributes.append(std.heap.page_allocator, c.oraNamedAttributeGet(param_names_id, param_names_array)) catch {};

            const param_types_array = c.oraArrayAttrCreate(self.ctx, @intCast(param_types.items.len), param_types.items.ptr);
            const param_types_id = h.identifier(self.ctx, "ora.param_types");
            attributes.append(std.heap.page_allocator, c.oraNamedAttributeGet(param_types_id, param_types_array)) catch {};
        }
    }

    // add error declaration marker
    const error_decl_attr = h.boolAttr(self.ctx, 1);
    const error_decl_id = h.identifier(self.ctx, "ora.error_decl");
    attributes.append(std.heap.page_allocator, c.oraNamedAttributeGet(error_decl_id, error_decl_attr)) catch {};

    if (self.symbol_table) |st| {
        const err_id = st.getOrCreateErrorId(error_decl.name) catch 0;
        const id_type = c.oraIntegerTypeCreate(self.ctx, 32);
        const id_attr = h.intAttrCast(self.ctx, id_type, err_id);
        const id_name = h.identifier(self.ctx, "ora.error_id");
        attributes.append(std.heap.page_allocator, c.oraNamedAttributeGet(id_name, id_attr)) catch {};
    }

    // create the error type and add it as a result
    const error_type = self.createErrorType(error_decl);
    return c.oraErrorDeclOpCreate(
        self.ctx,
        loc,
        &[_]c.MlirType{error_type},
        1,
        if (attributes.items.len == 0) null else attributes.items.ptr,
        attributes.items.len,
    );
}
