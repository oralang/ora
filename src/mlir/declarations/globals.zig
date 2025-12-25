// ============================================================================
// Declaration Lowering - Globals and Constants
// ============================================================================

const std = @import("std");
const c = @import("mlir_c_api").c;
const lib = @import("ora_lib");
const h = @import("../helpers.zig");
const constants = @import("../lower.zig");
const ExpressionLowerer = @import("../expressions.zig").ExpressionLowerer;
const DeclarationLowerer = @import("mod.zig").DeclarationLowerer;
const helpers = @import("helpers.zig");

/// Lower const declarations with global constant definitions (Requirements 7.6)
pub fn lowerConstDecl(self: *const DeclarationLowerer, const_decl: *const lib.ast.ConstantNode) c.MlirOperation {
    // create ora.const operation
    var state = h.opState("ora.const", helpers.createFileLocation(self, const_decl.span));

    // collect const attributes
    var attributes = std.ArrayList(c.MlirNamedAttribute){};
    defer attributes.deinit(std.heap.page_allocator);

    // add constant name
    const name_ref = c.mlirStringRefCreate(const_decl.name.ptr, const_decl.name.len);
    const name_attr = c.mlirStringAttrGet(self.ctx, name_ref);
    const name_id = h.identifier(self.ctx, "sym_name");
    attributes.append(std.heap.page_allocator, c.mlirNamedAttributeGet(name_id, name_attr)) catch {};

    // add constant type
    const const_type = self.type_mapper.toMlirType(const_decl.typ);
    const type_attr = c.mlirTypeAttrGet(const_type);
    const type_id = h.identifier(self.ctx, "type");
    attributes.append(std.heap.page_allocator, c.mlirNamedAttributeGet(type_id, type_attr)) catch {};

    // add visibility modifier
    const visibility_attr = switch (const_decl.visibility) {
        .Public => c.mlirStringAttrGet(self.ctx, h.strRef("pub")),
        .Private => c.mlirStringAttrGet(self.ctx, h.strRef("private")),
    };
    const visibility_id = h.identifier(self.ctx, "ora.visibility");
    attributes.append(std.heap.page_allocator, c.mlirNamedAttributeGet(visibility_id, visibility_attr)) catch {};

    // add constant declaration marker
    const const_decl_attr = c.mlirBoolAttrGet(self.ctx, 1);
    const const_decl_id = h.identifier(self.ctx, "ora.const_decl");
    attributes.append(std.heap.page_allocator, c.mlirNamedAttributeGet(const_decl_id, const_decl_attr)) catch {};

    // apply all attributes
    c.mlirOperationStateAddAttributes(&state, @intCast(attributes.items.len), attributes.items.ptr);

    // add the constant type as a result
    c.mlirOperationStateAddResults(&state, 1, @ptrCast(&const_type));

    // create a region for the constant value initialization
    const region = c.mlirRegionCreate();
    const block = c.mlirBlockCreate(0, null, null);
    c.mlirRegionInsertOwnedBlock(region, 0, block);
    c.mlirOperationStateAddOwnedRegions(&state, 1, @ptrCast(&region));

    // lower the constant value expression
    // create a temporary expression lowerer to lower the constant value
    const expr_lowerer = ExpressionLowerer.init(self.ctx, block, self.type_mapper, null, null, null, self.symbol_table, self.builtin_registry, self.error_handler, self.locations, self.ora_dialect);
    const const_value = expr_lowerer.lowerExpression(const_decl.value);

    // add a yield to terminate the region (required for regions)
    const yield_op = self.ora_dialect.createYield(&[_]c.MlirValue{const_value}, helpers.createFileLocation(self, const_decl.span));
    h.appendOp(block, yield_op);

    const const_op = c.mlirOperationCreate(&state);
    return const_op;
}

/// Lower immutable declarations with immutable global definitions and initialization constraints (Requirements 7.7)
pub fn lowerImmutableDecl(self: *const DeclarationLowerer, var_decl: *const lib.ast.Statements.VariableDeclNode) c.MlirOperation {
    // create ora.immutable operation for immutable global variables
    var state = h.opState("ora.immutable", helpers.createFileLocation(self, var_decl.span));

    // collect immutable attributes
    var attributes = std.ArrayList(c.MlirNamedAttribute){};
    defer attributes.deinit(std.heap.page_allocator);

    // add variable name
    const name_ref = c.mlirStringRefCreate(var_decl.name.ptr, var_decl.name.len);
    const name_attr = c.mlirStringAttrGet(self.ctx, name_ref);
    const name_id = h.identifier(self.ctx, "sym_name");
    attributes.append(std.heap.page_allocator, c.mlirNamedAttributeGet(name_id, name_attr)) catch {};

    // add variable type
    const var_type = self.type_mapper.toMlirType(var_decl.type_info);
    const type_attr = c.mlirTypeAttrGet(var_type);
    const type_id = h.identifier(self.ctx, "type");
    attributes.append(std.heap.page_allocator, c.mlirNamedAttributeGet(type_id, type_attr)) catch {};

    // add immutable constraint marker
    const immutable_attr = c.mlirBoolAttrGet(self.ctx, 1);
    const immutable_id = h.identifier(self.ctx, "ora.immutable");
    attributes.append(std.heap.page_allocator, c.mlirNamedAttributeGet(immutable_id, immutable_attr)) catch {};

    // add initialization constraint - immutable variables must be initialized
    if (var_decl.value == null) {
        const requires_init_attr = c.mlirBoolAttrGet(self.ctx, 1);
        const requires_init_id = h.identifier(self.ctx, "ora.requires_init");
        attributes.append(std.heap.page_allocator, c.mlirNamedAttributeGet(requires_init_id, requires_init_attr)) catch {};
    }

    // apply all attributes
    c.mlirOperationStateAddAttributes(&state, @intCast(attributes.items.len), attributes.items.ptr);

    // add the variable type as a result
    c.mlirOperationStateAddResults(&state, 1, @ptrCast(&var_type));

    // create a region for initialization if there's an initial value
    if (var_decl.value != null) {
        const region = c.mlirRegionCreate();
        const block = c.mlirBlockCreate(0, null, null);
        c.mlirRegionInsertOwnedBlock(region, 0, block);
        c.mlirOperationStateAddOwnedRegions(&state, 1, @ptrCast(&region));

        // initialization expression lowering handled by expression lowerer
        // body block created for variable init code
    }

    return c.mlirOperationCreate(&state);
}

/// Create global storage variable declaration
pub fn createGlobalDeclaration(self: *const DeclarationLowerer, var_decl: *const lib.ast.Statements.VariableDeclNode) c.MlirOperation {
    // use the type mapper to get the correct MLIR type (preserves signedness: u256 -> ui256, i256 -> si256)
    const var_type = self.type_mapper.toMlirType(var_decl.type_info);

    // the symbol table should already be populated in the first pass with the correct type
    // (using toMlirType which now returns i1 for booleans)
    // if it's not correct, that's a separate issue that needs to be fixed at the source

    // check if this is a map type - maps don't have scalar initializers
    const is_map_type = if (var_decl.type_info.ora_type) |ora_type| blk: {
        break :blk ora_type == .map;
    } else false;

    // determine the initial value based on the actual type
    // maps don't have initializers - use null attribute
    const init_attr = if (is_map_type) blk: {
        // maps are first-class types, no scalar initializer
        break :blk c.mlirAttributeGetNull();
    } else if (var_decl.value) |_| blk: {
        // if there's an initial value expression, we need to lower it
        // for now, use a placeholder - the actual value should come from lowering the expression
        const zero_attr = c.mlirIntegerAttrGet(var_type, 0);
        break :blk zero_attr;
    } else blk: {
        // no initial value - use zero of the correct type
        const zero_attr = c.mlirIntegerAttrGet(var_type, 0);
        break :blk zero_attr;
    };

    // use the dialect helper function to create the global operation
    return self.ora_dialect.createGlobal(var_decl.name, var_type, init_attr, helpers.createFileLocation(self, var_decl.span));
}

/// Create memory global variable declaration
pub fn createMemoryGlobalDeclaration(self: *const DeclarationLowerer, var_decl: *const lib.ast.Statements.VariableDeclNode) c.MlirOperation {
    // create ora.memory.global operation
    var state = h.opState("ora.memory.global", helpers.createFileLocation(self, var_decl.span));

    // add the global name as a symbol attribute
    const name_ref = c.mlirStringRefCreate(var_decl.name.ptr, var_decl.name.len);
    const name_attr = c.mlirStringAttrGet(self.ctx, name_ref);
    const name_id = h.identifier(self.ctx, "sym_name");
    var attrs = [_]c.MlirNamedAttribute{
        c.mlirNamedAttributeGet(name_id, name_attr),
    };
    c.mlirOperationStateAddAttributes(&state, @intCast(attrs.len), &attrs);

    // add the type attribute
    const var_type = c.mlirIntegerTypeGet(self.ctx, constants.DEFAULT_INTEGER_BITS); // default to i256
    const type_attr = c.mlirTypeAttrGet(var_type);
    const type_id = h.identifier(self.ctx, "type");
    var type_attrs = [_]c.MlirNamedAttribute{
        c.mlirNamedAttributeGet(type_id, type_attr),
    };
    c.mlirOperationStateAddAttributes(&state, @intCast(type_attrs.len), &type_attrs);

    return c.mlirOperationCreate(&state);
}

/// Create transient storage global variable declaration
pub fn createTStoreGlobalDeclaration(self: *const DeclarationLowerer, var_decl: *const lib.ast.Statements.VariableDeclNode) c.MlirOperation {
    // create ora.tstore.global operation
    var state = h.opState("ora.tstore.global", helpers.createFileLocation(self, var_decl.span));

    // add the global name as a symbol attribute
    const name_ref = c.mlirStringRefCreate(var_decl.name.ptr, var_decl.name.len);
    const name_attr = c.mlirStringAttrGet(self.ctx, name_ref);
    const name_id = h.identifier(self.ctx, "sym_name");
    var attrs = [_]c.MlirNamedAttribute{
        c.mlirNamedAttributeGet(name_id, name_attr),
    };
    c.mlirOperationStateAddAttributes(&state, @intCast(attrs.len), &attrs);

    // add the type attribute
    const var_type = c.mlirIntegerTypeGet(self.ctx, constants.DEFAULT_INTEGER_BITS); // default to i256
    const type_attr = c.mlirTypeAttrGet(var_type);
    const type_id = h.identifier(self.ctx, "type");
    var type_attrs = [_]c.MlirNamedAttribute{
        c.mlirNamedAttributeGet(type_id, type_attr),
    };
    c.mlirOperationStateAddAttributes(&state, @intCast(type_attrs.len), &type_attrs);

    return c.mlirOperationCreate(&state);
}
