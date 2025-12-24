// ============================================================================
// Declaration Lowering - Verification
// ============================================================================

const std = @import("std");
const c = @import("mlir_c_api").c;
const lib = @import("ora_lib");
const h = @import("../helpers.zig");
const LocalVarMap = @import("../symbols.zig").LocalVarMap;
const ParamMap = @import("../symbols.zig").ParamMap;
const StorageMap = @import("../memory.zig").StorageMap;
const ExpressionLowerer = @import("../expressions.zig").ExpressionLowerer;
const DeclarationLowerer = @import("mod.zig").DeclarationLowerer;
const helpers = @import("helpers.zig");

/// Lower quantified expressions (forall, exists) with verification constructs and ora.quantified attributes (Requirements 6.6)
pub fn lowerQuantifiedExpression(self: *const DeclarationLowerer, quantified: *const lib.ast.Expressions.QuantifiedExpr, block: c.MlirBlock, param_map: *const ParamMap, storage_map: ?*const StorageMap, local_var_map: ?*LocalVarMap) !c.MlirValue {
    // Create ora.quantified operation
    var state = h.opState("ora.quantified", helpers.createFileLocation(self, quantified.span));

    // Collect quantified attributes
    var attributes = std.ArrayList(c.MlirNamedAttribute){};
    defer attributes.deinit(std.heap.page_allocator);

    // Add quantifier type (forall or exists)
    const quantifier_str = switch (quantified.quantifier) {
        .Forall => "forall",
        .Exists => "exists",
    };
    const quantifier_attr = h.stringAttr(self.ctx, quantifier_str);
    const quantifier_id = h.identifier(self.ctx, "ora.quantifier");
    attributes.append(std.heap.page_allocator, c.mlirNamedAttributeGet(quantifier_id, quantifier_attr)) catch {};

    // Add bound variable name
    const var_name_ref = c.mlirStringRefCreate(quantified.variable.ptr, quantified.variable.len);
    const var_name_attr = c.mlirStringAttrGet(self.ctx, var_name_ref);
    const var_name_id = h.identifier(self.ctx, "ora.bound_variable");
    attributes.append(std.heap.page_allocator, c.mlirNamedAttributeGet(var_name_id, var_name_attr)) catch {};

    // Add bound variable type
    const var_type = self.type_mapper.toMlirType(quantified.variable_type);
    const var_type_attr = c.mlirTypeAttrGet(var_type);
    const var_type_id = h.identifier(self.ctx, "ora.bound_variable_type");
    attributes.append(std.heap.page_allocator, c.mlirNamedAttributeGet(var_type_id, var_type_attr)) catch {};

    // Add quantified expression marker
    const quantified_attr = c.mlirBoolAttrGet(self.ctx, 1);
    const quantified_id = h.identifier(self.ctx, "ora.quantified");
    attributes.append(std.heap.page_allocator, c.mlirNamedAttributeGet(quantified_id, quantified_attr)) catch {};

    // Apply all attributes
    c.mlirOperationStateAddAttributes(&state, @intCast(attributes.items.len), attributes.items.ptr);

    // Create result type (quantified expressions return boolean)
    const result_type = h.boolType(self.ctx); // i1 for boolean
    c.mlirOperationStateAddResults(&state, 1, @ptrCast(&result_type));

    // Create regions for condition and body
    const condition_region = c.mlirRegionCreate();
    const body_region = c.mlirRegionCreate();

    // Create blocks for condition and body
    const condition_block = c.mlirBlockCreate(0, null, null);
    const body_block = c.mlirBlockCreate(0, null, null);

    c.mlirRegionInsertOwnedBlock(condition_region, 0, condition_block);
    c.mlirRegionInsertOwnedBlock(body_region, 0, body_block);

    // Add regions to the operation
    var regions = [_]c.MlirRegion{ condition_region, body_region };
    c.mlirOperationStateAddOwnedRegions(&state, regions.len, &regions);

    // Lower the condition if present
    if (quantified.condition) |condition| {
        const const_local_var_map = if (local_var_map) |lvm| @as(*const LocalVarMap, lvm) else null;
        const expr_lowerer = ExpressionLowerer.init(self.ctx, condition_block, self.type_mapper, param_map, storage_map, const_local_var_map, self.symbol_table, self.builtin_registry, self.error_handler, self.locations, self.ora_dialect);
        _ = expr_lowerer.lowerExpression(condition);
    }

    // Lower the body expression
    const const_local_var_map = if (local_var_map) |lvm| @as(*const LocalVarMap, lvm) else null;
    const expr_lowerer = ExpressionLowerer.init(self.ctx, body_block, self.type_mapper, param_map, storage_map, const_local_var_map, self.symbol_table, self.builtin_registry, self.error_handler, self.locations, self.ora_dialect);
    _ = expr_lowerer.lowerExpression(quantified.body);

    // Create the quantified operation
    const quantified_op = c.mlirOperationCreate(&state);
    h.appendOp(block, quantified_op);

    return c.mlirValueFromOpResult(h.getResult(quantified_op, 0));
}

/// Add verification-related attributes and metadata support
pub fn addVerificationAttributes(self: *const DeclarationLowerer, operation: c.MlirOperation, verification_type: []const u8, metadata: ?[]const u8) void {
    // Add verification type attribute
    const verification_attr = h.stringAttr(self.ctx, verification_type);
    const verification_id = h.identifier(self.ctx, "ora.verification_type");
    c.mlirOperationSetAttribute(operation, verification_id, verification_attr);

    // Add metadata if provided
    if (metadata) |meta| {
        const metadata_attr = h.stringAttr(self.ctx, meta);
        const metadata_id = h.identifier(self.ctx, "ora.verification_metadata");
        c.mlirOperationSetAttribute(operation, metadata_id, metadata_attr);
    }

    // Add verification marker
    const verification_marker = c.mlirBoolAttrGet(self.ctx, 1);
    const verification_marker_id = h.identifier(self.ctx, "ora.formal_verification");
    c.mlirOperationSetAttribute(operation, verification_marker_id, verification_marker);
}

/// Handle formal verification constructs in function contracts
pub fn lowerFormalVerificationConstructs(self: *const DeclarationLowerer, func: *const lib.FunctionNode, func_op: c.MlirOperation) void {
    // Add verification attributes for functions with requires/ensures clauses
    if (func.requires_clauses.len > 0 or func.ensures_clauses.len > 0) {
        self.addVerificationAttributes(func_op, "function_contract", null);
    }

    // Add specific attributes for preconditions
    if (func.requires_clauses.len > 0) {
        const precondition_attr = c.mlirIntegerAttrGet(c.mlirIntegerTypeGet(self.ctx, 32), @intCast(func.requires_clauses.len));
        const precondition_id = h.identifier(self.ctx, "ora.precondition_count");
        c.mlirOperationSetAttribute(func_op, precondition_id, precondition_attr);
    }

    // Add specific attributes for postconditions
    if (func.ensures_clauses.len > 0) {
        const postcondition_attr = c.mlirIntegerAttrGet(c.mlirIntegerTypeGet(self.ctx, 32), @intCast(func.ensures_clauses.len));
        const postcondition_id = h.identifier(self.ctx, "ora.postcondition_count");
        c.mlirOperationSetAttribute(func_op, postcondition_id, postcondition_attr);
    }
}
