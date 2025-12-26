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
    const loc = helpers.createFileLocation(self, module.span);

    // collect module attributes
    var attributes = std.ArrayList(c.MlirNamedAttribute){};
    defer attributes.deinit(std.heap.page_allocator);

    // add module name if present
    if (module.name) |name| {
        const name_ref = c.oraStringRefCreate(name.ptr, name.len);
        const name_attr = c.oraStringAttrCreate(self.ctx, name_ref);
        const name_id = h.identifier(self.ctx, "sym_name");
        attributes.append(std.heap.page_allocator, c.oraNamedAttributeGet(name_id, name_attr)) catch {};
    }

    // add module declaration marker
    const module_decl_attr = h.boolAttr(self.ctx, 1);
    const module_decl_id = h.identifier(self.ctx, "ora.module_decl");
    attributes.append(std.heap.page_allocator, c.oraNamedAttributeGet(module_decl_id, module_decl_attr)) catch {};

    // add import count attribute
    const import_count_attr = c.oraIntegerAttrCreateI64FromType(c.oraIntegerTypeCreate(self.ctx, 32), @intCast(module.imports.len));
    const import_count_id = h.identifier(self.ctx, "ora.import_count");
    attributes.append(std.heap.page_allocator, c.oraNamedAttributeGet(import_count_id, import_count_attr)) catch {};

    // add declaration count attribute
    const decl_count_attr = c.oraIntegerAttrCreateI64FromType(c.oraIntegerTypeCreate(self.ctx, 32), @intCast(module.declarations.len));
    const decl_count_id = h.identifier(self.ctx, "ora.declaration_count");
    attributes.append(std.heap.page_allocator, c.oraNamedAttributeGet(decl_count_id, decl_count_attr)) catch {};

    return c.oraModuleOpCreate(
        self.ctx,
        loc,
        if (attributes.items.len == 0) null else attributes.items.ptr,
        attributes.items.len,
        1,
        true,
    );
}

/// Lower block declarations for block constructs
pub fn lowerBlock(self: *const DeclarationLowerer, block_decl: *const lib.ast.Statements.BlockNode) c.MlirOperation {
    const loc = helpers.createFileLocation(self, block_decl.span);

    // collect block attributes
    var attributes = std.ArrayList(c.MlirNamedAttribute){};
    defer attributes.deinit(std.heap.page_allocator);

    // add block declaration marker
    const block_decl_attr = h.boolAttr(self.ctx, 1);
    const block_decl_id = h.identifier(self.ctx, "ora.block_decl");
    attributes.append(std.heap.page_allocator, c.oraNamedAttributeGet(block_decl_id, block_decl_attr)) catch {};

    // add statement count attribute
    const stmt_count_attr = c.oraIntegerAttrCreateI64FromType(c.oraIntegerTypeCreate(self.ctx, 32), @intCast(block_decl.statements.len));
    const stmt_count_id = h.identifier(self.ctx, "ora.statement_count");
    attributes.append(std.heap.page_allocator, c.oraNamedAttributeGet(stmt_count_id, stmt_count_attr)) catch {};

    return c.oraBlockOpCreate(
        self.ctx,
        loc,
        if (attributes.items.len == 0) null else attributes.items.ptr,
        attributes.items.len,
        1,
        true,
    );
}

/// Lower try-block declarations for try-catch blocks
pub fn lowerTryBlock(self: *const DeclarationLowerer, try_block: *const lib.ast.Statements.TryBlockNode) c.MlirOperation {
    const loc = helpers.createFileLocation(self, try_block.span);

    // collect try-block attributes
    var attributes = std.ArrayList(c.MlirNamedAttribute){};
    defer attributes.deinit(std.heap.page_allocator);

    // add try-block declaration marker
    const try_block_decl_attr = h.boolAttr(self.ctx, 1);
    const try_block_decl_id = h.identifier(self.ctx, "ora.try_block_decl");
    attributes.append(std.heap.page_allocator, c.oraNamedAttributeGet(try_block_decl_id, try_block_decl_attr)) catch {};

    // add error handling marker
    const error_handling_attr = h.boolAttr(self.ctx, 1);
    const error_handling_id = h.identifier(self.ctx, "ora.error_handling");
    attributes.append(std.heap.page_allocator, c.oraNamedAttributeGet(error_handling_id, error_handling_attr)) catch {};

    // add catch block presence attribute
    const has_catch_attr = h.boolAttr(self.ctx, @as(i32, @intFromBool(try_block.catch_block != null)));
    const has_catch_id = h.identifier(self.ctx, "ora.has_catch");
    attributes.append(std.heap.page_allocator, c.oraNamedAttributeGet(has_catch_id, has_catch_attr)) catch {};

    const region_count: usize = if (try_block.catch_block != null) 2 else 1;
    return c.oraTryBlockOpCreate(
        self.ctx,
        loc,
        if (attributes.items.len == 0) null else attributes.items.ptr,
        attributes.items.len,
        region_count,
        true,
    );
}

/// Lower import declarations with module import constructs (Requirements 7.5)
pub fn lowerImport(self: *const DeclarationLowerer, import_decl: *const lib.ast.ImportNode) c.MlirOperation {
    const loc = helpers.createFileLocation(self, import_decl.span);

    // collect import attributes
    var attributes = std.ArrayList(c.MlirNamedAttribute){};
    defer attributes.deinit(std.heap.page_allocator);

    // add import path
    const path_ref = c.oraStringRefCreate(import_decl.path.ptr, import_decl.path.len);
    const path_attr = c.oraStringAttrCreate(self.ctx, path_ref);
    const path_id = h.identifier(self.ctx, "ora.import_path");
    attributes.append(std.heap.page_allocator, c.oraNamedAttributeGet(path_id, path_attr)) catch {};

    // add alias if present
    if (import_decl.alias) |alias| {
        const alias_ref = c.oraStringRefCreate(alias.ptr, alias.len);
        const alias_attr = c.oraStringAttrCreate(self.ctx, alias_ref);
        const alias_id = h.identifier(self.ctx, "ora.import_alias");
        attributes.append(std.heap.page_allocator, c.oraNamedAttributeGet(alias_id, alias_attr)) catch {};
    }

    // add import declaration marker
    const import_decl_attr = h.boolAttr(self.ctx, 1);
    const import_decl_id = h.identifier(self.ctx, "ora.import_decl");
    attributes.append(std.heap.page_allocator, c.oraNamedAttributeGet(import_decl_id, import_decl_attr)) catch {};

    return c.oraImportOpCreate(
        self.ctx,
        loc,
        if (attributes.items.len == 0) null else attributes.items.ptr,
        attributes.items.len,
    );
}
