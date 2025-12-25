// ============================================================================
// Declaration Lowering - Module/Block/Import
// ============================================================================

const std = @import("std");
const c = @import("mlir_c_api").c;
const lib = @import("ora_lib");
const h = @import("../helpers.zig");
const DeclarationLowerer = @import("mod.zig").DeclarationLowerer;
const helpers = @import("helpers.zig");

/// Lower module declarations for top-level program structure
pub fn lowerModule(self: *const DeclarationLowerer, module: *const lib.ast.ModuleNode) c.MlirOperation {
    // create ora.module operation
    var state = h.opState("ora.module", helpers.createFileLocation(self, module.span));

    // collect module attributes
    var attributes = std.ArrayList(c.MlirNamedAttribute){};
    defer attributes.deinit(std.heap.page_allocator);

    // add module name if present
    if (module.name) |name| {
        const name_ref = c.mlirStringRefCreate(name.ptr, name.len);
        const name_attr = c.mlirStringAttrGet(self.ctx, name_ref);
        const name_id = h.identifier(self.ctx, "sym_name");
        attributes.append(std.heap.page_allocator, c.mlirNamedAttributeGet(name_id, name_attr)) catch {};
    }

    // add module declaration marker
    const module_decl_attr = c.mlirBoolAttrGet(self.ctx, 1);
    const module_decl_id = h.identifier(self.ctx, "ora.module_decl");
    attributes.append(std.heap.page_allocator, c.mlirNamedAttributeGet(module_decl_id, module_decl_attr)) catch {};

    // add import count attribute
    const import_count_attr = c.mlirIntegerAttrGet(c.mlirIntegerTypeGet(self.ctx, 32), @intCast(module.imports.len));
    const import_count_id = h.identifier(self.ctx, "ora.import_count");
    attributes.append(std.heap.page_allocator, c.mlirNamedAttributeGet(import_count_id, import_count_attr)) catch {};

    // add declaration count attribute
    const decl_count_attr = c.mlirIntegerAttrGet(c.mlirIntegerTypeGet(self.ctx, 32), @intCast(module.declarations.len));
    const decl_count_id = h.identifier(self.ctx, "ora.declaration_count");
    attributes.append(std.heap.page_allocator, c.mlirNamedAttributeGet(decl_count_id, decl_count_attr)) catch {};

    // apply all attributes
    c.mlirOperationStateAddAttributes(&state, @intCast(attributes.items.len), attributes.items.ptr);

    // create a region for the module body
    const region = c.mlirRegionCreate();
    c.mlirOperationStateAddOwnedRegions(&state, 1, @ptrCast(&region));

    return c.mlirOperationCreate(&state);
}

/// Lower block declarations for block constructs
pub fn lowerBlock(self: *const DeclarationLowerer, block_decl: *const lib.ast.Statements.BlockNode) c.MlirOperation {
    // create ora.block operation
    var state = h.opState("ora.block", helpers.createFileLocation(self, block_decl.span));

    // collect block attributes
    var attributes = std.ArrayList(c.MlirNamedAttribute){};
    defer attributes.deinit(std.heap.page_allocator);

    // add block declaration marker
    const block_decl_attr = c.mlirBoolAttrGet(self.ctx, 1);
    const block_decl_id = h.identifier(self.ctx, "ora.block_decl");
    attributes.append(std.heap.page_allocator, c.mlirNamedAttributeGet(block_decl_id, block_decl_attr)) catch {};

    // add statement count attribute
    const stmt_count_attr = c.mlirIntegerAttrGet(c.mlirIntegerTypeGet(self.ctx, 32), @intCast(block_decl.statements.len));
    const stmt_count_id = h.identifier(self.ctx, "ora.statement_count");
    attributes.append(std.heap.page_allocator, c.mlirNamedAttributeGet(stmt_count_id, stmt_count_attr)) catch {};

    // apply all attributes
    c.mlirOperationStateAddAttributes(&state, @intCast(attributes.items.len), attributes.items.ptr);

    // create a region for the block body
    const region = c.mlirRegionCreate();
    const block = c.mlirBlockCreate(0, null, null);
    c.mlirRegionInsertOwnedBlock(region, 0, block);
    c.mlirOperationStateAddOwnedRegions(&state, 1, @ptrCast(&region));

    return c.mlirOperationCreate(&state);
}

/// Lower try-block declarations for try-catch blocks
pub fn lowerTryBlock(self: *const DeclarationLowerer, try_block: *const lib.ast.Statements.TryBlockNode) c.MlirOperation {
    // create ora.try_block operation
    var state = h.opState("ora.try_block", helpers.createFileLocation(self, try_block.span));

    // collect try-block attributes
    var attributes = std.ArrayList(c.MlirNamedAttribute){};
    defer attributes.deinit(std.heap.page_allocator);

    // add try-block declaration marker
    const try_block_decl_attr = c.mlirBoolAttrGet(self.ctx, 1);
    const try_block_decl_id = h.identifier(self.ctx, "ora.try_block_decl");
    attributes.append(std.heap.page_allocator, c.mlirNamedAttributeGet(try_block_decl_id, try_block_decl_attr)) catch {};

    // add error handling marker
    const error_handling_attr = c.mlirBoolAttrGet(self.ctx, 1);
    const error_handling_id = h.identifier(self.ctx, "ora.error_handling");
    attributes.append(std.heap.page_allocator, c.mlirNamedAttributeGet(error_handling_id, error_handling_attr)) catch {};

    // add catch block presence attribute
    const has_catch_attr = c.mlirBoolAttrGet(self.ctx, if (try_block.catch_block != null) 1 else 0);
    const has_catch_id = h.identifier(self.ctx, "ora.has_catch");
    attributes.append(std.heap.page_allocator, c.mlirNamedAttributeGet(has_catch_id, has_catch_attr)) catch {};

    // apply all attributes
    c.mlirOperationStateAddAttributes(&state, @intCast(attributes.items.len), attributes.items.ptr);

    // create regions for try and catch blocks
    const try_region = c.mlirRegionCreate();
    const try_block_mlir = c.mlirBlockCreate(0, null, null);
    c.mlirRegionInsertOwnedBlock(try_region, 0, try_block_mlir);
    c.mlirOperationStateAddOwnedRegions(&state, 1, @ptrCast(&try_region));

    // add catch region if present
    if (try_block.catch_block != null) {
        const catch_region = c.mlirRegionCreate();
        const catch_block_mlir = c.mlirBlockCreate(0, null, null);
        c.mlirRegionInsertOwnedBlock(catch_region, 0, catch_block_mlir);
        c.mlirOperationStateAddOwnedRegions(&state, 1, @ptrCast(&catch_region));
    }

    return c.mlirOperationCreate(&state);
}

/// Lower import declarations with module import constructs (Requirements 7.5)
pub fn lowerImport(self: *const DeclarationLowerer, import_decl: *const lib.ast.ImportNode) c.MlirOperation {
    // create ora.import operation
    var state = h.opState("ora.import", helpers.createFileLocation(self, import_decl.span));

    // collect import attributes
    var attributes = std.ArrayList(c.MlirNamedAttribute){};
    defer attributes.deinit(std.heap.page_allocator);

    // add import path
    const path_ref = c.mlirStringRefCreate(import_decl.path.ptr, import_decl.path.len);
    const path_attr = c.mlirStringAttrGet(self.ctx, path_ref);
    const path_id = h.identifier(self.ctx, "ora.import_path");
    attributes.append(std.heap.page_allocator, c.mlirNamedAttributeGet(path_id, path_attr)) catch {};

    // add alias if present
    if (import_decl.alias) |alias| {
        const alias_ref = c.mlirStringRefCreate(alias.ptr, alias.len);
        const alias_attr = c.mlirStringAttrGet(self.ctx, alias_ref);
        const alias_id = h.identifier(self.ctx, "ora.import_alias");
        attributes.append(std.heap.page_allocator, c.mlirNamedAttributeGet(alias_id, alias_attr)) catch {};
    }

    // add import declaration marker
    const import_decl_attr = c.mlirBoolAttrGet(self.ctx, 1);
    const import_decl_id = h.identifier(self.ctx, "ora.import_decl");
    attributes.append(std.heap.page_allocator, c.mlirNamedAttributeGet(import_decl_id, import_decl_attr)) catch {};

    // apply all attributes
    c.mlirOperationStateAddAttributes(&state, @intCast(attributes.items.len), attributes.items.ptr);

    return c.mlirOperationCreate(&state);
}
