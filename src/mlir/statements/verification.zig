// ============================================================================
// Verification Statement Lowering
// ============================================================================
// Formal verification operations: assert, invariant, requires, ensures, assume, havoc

const std = @import("std");
const c = @import("mlir_c_api").c;
const lib = @import("ora_lib");
const h = @import("../helpers.zig");
const stmt_helpers = @import("helpers.zig");
const StatementLowerer = @import("statement_lowerer.zig").StatementLowerer;
const LoweringError = StatementLowerer.LoweringError;

/// Add verification attributes to an existing operation
pub fn addVerificationAttributesToOp(
    self: *const StatementLowerer,
    op: c.MlirOperation,
    verification_type: []const u8,
    context: []const u8,
) void {
    // add verification marker
    const verification_attr = h.boolAttr(self.ctx, 1);
    const verification_name = h.strRef("ora.verification");
    c.oraOperationSetAttributeByName(op, verification_name, verification_attr);

    // add verification type
    const type_name = h.strRef("ora.verification_type");
    const type_attr = h.stringAttr(self.ctx, verification_type);
    c.oraOperationSetAttributeByName(op, type_name, type_attr);

    // add verification context
    const context_name = h.strRef("ora.verification_context");
    const context_attr = h.stringAttr(self.ctx, context);
    c.oraOperationSetAttributeByName(op, context_name, context_attr);

    // add formal verification marker
    const formal_attr = h.boolAttr(self.ctx, 1);
    const formal_name = h.strRef("ora.formal");
    c.oraOperationSetAttributeByName(op, formal_name, formal_attr);
}

/// Lower assert statements (runtime or ghost assertions)
pub fn lowerAssert(self: *const StatementLowerer, assert_stmt: *const lib.ast.Statements.AssertNode) LoweringError!void {
    const loc = self.fileLoc(assert_stmt.span);
    const condition = self.expr_lowerer.lowerExpression(&assert_stmt.condition);

    const context_str = if (assert_stmt.is_ghost) "ghost_assertion" else "runtime_assertion";
    const op = self.ora_dialect.createAssert(condition, loc, assert_stmt.message);
    addVerificationAttributesToOp(self, op, "assert", context_str);
    h.appendOp(self.block, op);
}

/// Lower invariant statements for loop invariants
pub fn lowerInvariant(self: *const StatementLowerer, invariant: *const lib.ast.Statements.InvariantNode) LoweringError!void {
    const loc = self.fileLoc(invariant.span);
    const condition = self.expr_lowerer.lowerExpression(&invariant.condition);

    const op = self.ora_dialect.createInvariant(condition, loc);
    addVerificationAttributesToOp(self, op, "invariant", "loop_invariant");
    h.appendOp(self.block, op);
}

/// Lower requires statements for function preconditions
pub fn lowerRequires(self: *const StatementLowerer, requires: *const lib.ast.Statements.RequiresNode) LoweringError!void {
    const loc = self.fileLoc(requires.span);
    const condition = self.expr_lowerer.lowerExpression(&requires.condition);

    const op = self.ora_dialect.createRequires(condition, loc);
    addVerificationAttributesToOp(self, op, "requires", "function_precondition");
    h.appendOp(self.block, op);
}

/// Lower ensures statements for function postconditions
pub fn lowerEnsures(self: *const StatementLowerer, ensures: *const lib.ast.Statements.EnsuresNode) LoweringError!void {
    const loc = self.fileLoc(ensures.span);
    const condition = self.expr_lowerer.lowerExpression(&ensures.condition);

    const op = self.ora_dialect.createEnsures(condition, loc);
    addVerificationAttributesToOp(self, op, "ensures", "function_postcondition");
    h.appendOp(self.block, op);
}

/// Lower assume statements for formal verification assumptions
pub fn lowerAssume(self: *const StatementLowerer, assume: *const lib.ast.Statements.AssumeNode) LoweringError!void {
    const loc = self.fileLoc(assume.span);
    const condition = blk: {
        if (self.active_condition_safe) {
            if (self.active_condition_value) |val| {
                if (self.active_condition_expr) |expr_ptr| {
                    if (expr_ptr == &assume.condition) {
                        break :blk val;
                    }
                }
            }
        }
        break :blk self.expr_lowerer.lowerExpression(&assume.condition);
    };

    const op = self.ora_dialect.createAssume(condition, loc);
    const origin_name = h.strRef("ora.assume_origin");
    const origin_str = switch (assume.origin) {
        .User => "user",
        .CompilerPath => "path",
    };
    const origin_attr = h.stringAttr(self.ctx, origin_str);
    c.oraOperationSetAttributeByName(op, origin_name, origin_attr);

    const assume_context = switch (assume.origin) {
        .User => "verification_assumption",
        .CompilerPath => "path_assumption",
    };
    addVerificationAttributesToOp(self, op, "assume", assume_context);
    h.appendOp(self.block, op);
}

/// Lower havoc statements for formal verification
pub fn lowerHavoc(self: *const StatementLowerer, havoc: *const lib.ast.Statements.HavocNode) LoweringError!void {
    const loc = self.fileLoc(havoc.span);
    const op = self.ora_dialect.createHavoc(havoc.variable_name, loc);
    addVerificationAttributesToOp(self, op, "havoc", "verification_havoc");
    h.appendOp(self.block, op);
}
