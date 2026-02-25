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
    const loc = helpers.createFileLocation(self, quantified.span);

    // collect quantified attributes
    var attributes = std.ArrayList(c.MlirNamedAttribute){};
    defer attributes.deinit(std.heap.page_allocator);

    // add quantifier type (forall or exists)
    const quantifier_str = switch (quantified.quantifier) {
        .Forall => "forall",
        .Exists => "exists",
    };
    const quantifier_attr = h.stringAttr(self.ctx, quantifier_str);
    const quantifier_id = h.identifier(self.ctx, "ora.quantifier");
    attributes.append(std.heap.page_allocator, c.oraNamedAttributeGet(quantifier_id, quantifier_attr)) catch {};

    // add bound variable name
    const var_name_ref = c.oraStringRefCreate(quantified.variable.ptr, quantified.variable.len);
    const var_name_attr = c.oraStringAttrCreate(self.ctx, var_name_ref);
    const var_name_id = h.identifier(self.ctx, "ora.bound_variable");
    attributes.append(std.heap.page_allocator, c.oraNamedAttributeGet(var_name_id, var_name_attr)) catch {};

    // add bound variable type
    const var_type = self.type_mapper.toMlirType(quantified.variable_type);
    const var_type_attr = c.oraTypeAttrCreateFromType(var_type);
    const var_type_id = h.identifier(self.ctx, "ora.bound_variable_type");
    attributes.append(std.heap.page_allocator, c.oraNamedAttributeGet(var_type_id, var_type_attr)) catch {};

    // add quantified expression marker
    const quantified_attr = h.boolAttr(self.ctx, 1);
    const quantified_id = h.identifier(self.ctx, "ora.quantified");
    attributes.append(std.heap.page_allocator, c.oraNamedAttributeGet(quantified_id, quantified_attr)) catch {};

    const quantified_op = c.oraQuantifiedDeclOpCreate(
        self.ctx,
        loc,
        if (attributes.items.len == 0) null else attributes.items.ptr,
        attributes.items.len,
        2,
        false,
    );

    // create blocks for condition and body
    const condition_block = c.oraOperationGetRegionBlock(quantified_op, 0);
    const body_block = c.oraOperationGetRegionBlock(quantified_op, 1);
    if (c.oraBlockIsNull(condition_block) or c.oraBlockIsNull(body_block)) {
        @panic("ora.quantified missing region blocks");
    }

    // lower the condition if present
    if (quantified.condition) |condition| {
        const const_local_var_map = if (local_var_map) |lvm| @as(*const LocalVarMap, lvm) else null;
        var expr_lowerer = ExpressionLowerer.init(self.ctx, condition_block, self.type_mapper, param_map, storage_map, const_local_var_map, self.symbol_table, self.builtin_registry, self.error_handler, self.locations, self.ora_dialect);
        expr_lowerer.module_exports = self.module_exports;
        _ = expr_lowerer.lowerExpression(condition);
    }

    // lower the body expression
    const const_local_var_map = if (local_var_map) |lvm| @as(*const LocalVarMap, lvm) else null;
    var expr_lowerer = ExpressionLowerer.init(self.ctx, body_block, self.type_mapper, param_map, storage_map, const_local_var_map, self.symbol_table, self.builtin_registry, self.error_handler, self.locations, self.ora_dialect);
    expr_lowerer.module_exports = self.module_exports;
    _ = expr_lowerer.lowerExpression(quantified.body);

    h.appendOp(block, quantified_op);

    return h.getResult(quantified_op, 0);
}

/// Add verification-related attributes and metadata support
pub fn addVerificationAttributes(self: *const DeclarationLowerer, operation: c.MlirOperation, verification_type: []const u8, metadata: ?[]const u8) void {
    // add verification type attribute
    const verification_attr = h.stringAttr(self.ctx, verification_type);
    c.oraOperationSetAttributeByName(operation, h.strRef("ora.verification_type"), verification_attr);

    // add metadata if provided
    if (metadata) |meta| {
        const metadata_attr = h.stringAttr(self.ctx, meta);
        c.oraOperationSetAttributeByName(operation, h.strRef("ora.verification_metadata"), metadata_attr);
    }

    // add verification marker
    const verification_marker = h.boolAttr(self.ctx, 1);
    c.oraOperationSetAttributeByName(operation, h.strRef("ora.formal_verification"), verification_marker);
}

/// Handle formal verification constructs in function contracts
pub fn lowerFormalVerificationConstructs(self: *const DeclarationLowerer, func: *const lib.FunctionNode, func_op: c.MlirOperation) void {
    // add verification attributes for functions with requires/ensures clauses
    if (func.requires_clauses.len > 0 or func.ensures_clauses.len > 0) {
        self.addVerificationAttributes(func_op, "function_contract", null);
    }

    // add specific attributes for preconditions
    if (func.requires_clauses.len > 0) {
        const precondition_attr = c.oraIntegerAttrCreateI64FromType(c.oraIntegerTypeCreate(self.ctx, 32), @intCast(func.requires_clauses.len));
        c.oraOperationSetAttributeByName(func_op, h.strRef("ora.precondition_count"), precondition_attr);
    }

    // add specific attributes for postconditions
    if (func.ensures_clauses.len > 0) {
        const postcondition_attr = c.oraIntegerAttrCreateI64FromType(c.oraIntegerTypeCreate(self.ctx, 32), @intCast(func.ensures_clauses.len));
        c.oraOperationSetAttributeByName(func_op, h.strRef("ora.postcondition_count"), postcondition_attr);
    }
}
