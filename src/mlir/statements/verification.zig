// ============================================================================
// Verification Statement Lowering
// ============================================================================
// Formal verification operations: assert, invariant, requires, ensures, assume, havoc

const std = @import("std");
const c = @import("mlir_c_api").c;
const lib = @import("ora_lib");
const h = @import("../helpers.zig");
const h_helpers = @import("../helpers.zig");
const StatementLowerer = @import("statement_lowerer.zig").StatementLowerer;
const LoweringError = StatementLowerer.LoweringError;

/// Add verification attributes to an operation state
pub fn addVerificationAttributes(
    self: *const StatementLowerer,
    state: *c.MlirOperationState,
    verification_type: []const u8,
    context: []const u8,
) void {
    var attributes = std.ArrayList(c.MlirNamedAttribute){};
    defer attributes.deinit(self.allocator);

    // add verification marker
    const verification_attr = c.mlirBoolAttrGet(self.ctx, 1);
    const verification_id = h.identifier(self.ctx, "ora.verification");
    attributes.append(self.allocator, c.mlirNamedAttributeGet(verification_id, verification_attr)) catch {};

    // add verification type
    const type_id = h.identifier(self.ctx, "ora.verification_type");
    const type_attr = h.stringAttr(self.ctx, verification_type);
    attributes.append(self.allocator, c.mlirNamedAttributeGet(type_id, type_attr)) catch {};

    // add verification context
    const context_id = h.identifier(self.ctx, "ora.verification_context");
    const context_attr = h.stringAttr(self.ctx, context);
    attributes.append(self.allocator, c.mlirNamedAttributeGet(context_id, context_attr)) catch {};

    // add formal verification marker
    const formal_id = h.identifier(self.ctx, "ora.formal");
    const formal_attr = c.mlirBoolAttrGet(self.ctx, 1);
    attributes.append(self.allocator, c.mlirNamedAttributeGet(formal_id, formal_attr)) catch {};

    // apply all attributes
    if (attributes.items.len > 0) {
        c.mlirOperationStateAddAttributes(state, @intCast(attributes.items.len), attributes.items.ptr);
    }
}

/// Add verification attributes to an existing operation
pub fn addVerificationAttributesToOp(
    self: *const StatementLowerer,
    op: c.MlirOperation,
    verification_type: []const u8,
    context: []const u8,
) void {
    // add verification marker
    const verification_attr = c.mlirBoolAttrGet(self.ctx, 1);
    const verification_name = h.strRef("ora.verification");
    c.mlirOperationSetAttributeByName(op, verification_name, verification_attr);

    // add verification type
    const type_name = h.strRef("ora.verification_type");
    const type_attr = h.stringAttr(self.ctx, verification_type);
    c.mlirOperationSetAttributeByName(op, type_name, type_attr);

    // add verification context
    const context_name = h.strRef("ora.verification_context");
    const context_attr = h.stringAttr(self.ctx, context);
    c.mlirOperationSetAttributeByName(op, context_name, context_attr);

    // add formal verification marker
    const formal_attr = c.mlirBoolAttrGet(self.ctx, 1);
    const formal_name = h.strRef("ora.formal");
    c.mlirOperationSetAttributeByName(op, formal_name, formal_attr);
}

/// Lower assert statements (runtime or ghost assertions)
pub fn lowerAssert(self: *const StatementLowerer, assert_stmt: *const lib.ast.Statements.AssertNode) LoweringError!void {
    const loc = self.fileLoc(assert_stmt.span);
    const condition = self.expr_lowerer.lowerExpression(&assert_stmt.condition);

    var state = h.opState("ora.assert", loc);
    c.mlirOperationStateAddOperands(&state, 1, @ptrCast(&condition));

    // add optional message attribute
    if (assert_stmt.message) |msg| {
        const msg_attr = h.stringAttr(self.ctx, msg);
        const msg_id = h.identifier(self.ctx, "message");
        var attrs = [_]c.MlirNamedAttribute{c.mlirNamedAttributeGet(msg_id, msg_attr)};
        c.mlirOperationStateAddAttributes(&state, attrs.len, &attrs);
    }

    // add verification attributes
    const context_str = if (assert_stmt.is_ghost) "ghost_assertion" else "runtime_assertion";
    addVerificationAttributes(self, &state, "assert", context_str);

    const op = c.mlirOperationCreate(&state);
    h.appendOp(self.block, op);
}

/// Lower invariant statements for loop invariants
pub fn lowerInvariant(self: *const StatementLowerer, invariant: *const lib.ast.Statements.InvariantNode) LoweringError!void {
    const loc = self.fileLoc(invariant.span);
    const condition = self.expr_lowerer.lowerExpression(&invariant.condition);

    var state = h.opState("ora.invariant", loc);
    c.mlirOperationStateAddOperands(&state, 1, @ptrCast(&condition));
    addVerificationAttributes(self, &state, "invariant", "loop_invariant");

    const op = c.mlirOperationCreate(&state);
    h.appendOp(self.block, op);
}

/// Lower requires statements for function preconditions
pub fn lowerRequires(self: *const StatementLowerer, requires: *const lib.ast.Statements.RequiresNode) LoweringError!void {
    const loc = self.fileLoc(requires.span);
    const condition = self.expr_lowerer.lowerExpression(&requires.condition);

    var state = h.opState("ora.requires", loc);
    c.mlirOperationStateAddOperands(&state, 1, @ptrCast(&condition));
    addVerificationAttributes(self, &state, "requires", "function_precondition");

    const op = c.mlirOperationCreate(&state);
    h.appendOp(self.block, op);
}

/// Lower ensures statements for function postconditions
pub fn lowerEnsures(self: *const StatementLowerer, ensures: *const lib.ast.Statements.EnsuresNode) LoweringError!void {
    const loc = self.fileLoc(ensures.span);
    const condition = self.expr_lowerer.lowerExpression(&ensures.condition);

    var state = h.opState("ora.ensures", loc);
    c.mlirOperationStateAddOperands(&state, 1, @ptrCast(&condition));
    addVerificationAttributes(self, &state, "ensures", "function_postcondition");

    const op = c.mlirOperationCreate(&state);
    h.appendOp(self.block, op);
}

/// Lower assume statements for formal verification assumptions
pub fn lowerAssume(self: *const StatementLowerer, assume: *const lib.ast.Statements.AssumeNode) LoweringError!void {
    const loc = self.fileLoc(assume.span);
    const condition = self.expr_lowerer.lowerExpression(&assume.condition);

    const op = self.ora_dialect.createAssume(condition, loc);
    addVerificationAttributesToOp(self, op, "assume", "verification_assumption");
    h.appendOp(self.block, op);
}

/// Lower havoc statements for formal verification
pub fn lowerHavoc(self: *const StatementLowerer, havoc: *const lib.ast.Statements.HavocNode) LoweringError!void {
    const loc = self.fileLoc(havoc.span);
    const op = self.ora_dialect.createHavoc(havoc.variable_name, loc);
    addVerificationAttributesToOp(self, op, "havoc", "verification_havoc");
    h.appendOp(self.block, op);
}
