// ============================================================================
// Return Statement Lowering
// ============================================================================

const std = @import("std");
const c = @import("mlir_c_api").c;
const lib = @import("ora_lib");
const h = @import("../helpers.zig");
const StatementLowerer = @import("statement_lowerer.zig").StatementLowerer;
const LoweringError = StatementLowerer.LoweringError;
const helpers = @import("helpers.zig");

/// Lower return statements using ora.return with proper value handling
pub fn lowerReturn(self: *const StatementLowerer, ret: *const lib.ast.Statements.ReturnNode) LoweringError!void {
    // If we're inside a try block, we can't use ora.return (it must be last in function block)
    // Instead, store the return value in memrefs and set the return flag
    if (self.in_try_block and self.try_return_flag_memref != null) {
        const loc = self.fileLoc(ret.span);
        const return_flag_memref = self.try_return_flag_memref.?;

        // Set return flag to true
        const true_val = helpers.createBoolConstant(self, true, loc);
        helpers.storeToMemref(self, true_val, return_flag_memref, loc);

        // Store return value if present
        if (ret.value) |value_expr| {
            if (self.try_return_value_memref) |return_value_memref| {
                var v = self.expr_lowerer.lowerExpression(&value_expr);

                // Insert refinement guard if return type is a refinement type
                if (self.current_function_return_type_info) |return_type_info| {
                    if (return_type_info.ora_type) |ora_type| {
                        v = try helpers.insertRefinementGuard(self, v, ora_type, ret.span, ret.skip_guard);
                    }
                }

                // Get target type from memref element type
                const memref_type = c.mlirValueGetType(return_value_memref);
                const element_type = c.mlirShapedTypeGetElementType(memref_type);

                // Convert value to match memref element type
                const final_value = helpers.convertValueToType(self, v, element_type, ret.span, loc);

                // Store return value
                helpers.storeToMemref(self, final_value, return_value_memref, loc);
            }
        }

        // Return early - the actual ora.return will be handled after the try block
        return;
    }

    const loc = self.fileLoc(ret.span);

    // Insert ensures clause checks before return (postconditions must hold at every return point)
    if (self.ensures_clauses.len > 0) {
        try lowerEnsuresBeforeReturn(self, self.block, ret.span);
    }

    if (ret.value) |e| {
        var v = self.expr_lowerer.lowerExpression(&e);

        // Insert refinement guard if return type is a refinement type
        if (self.current_function_return_type_info) |return_type_info| {
            if (return_type_info.ora_type) |ora_type| {
                v = try helpers.insertRefinementGuard(self, v, ora_type, ret.span, ret.skip_guard);
            }
        }

        // Convert return value to match function return type if available
        const final_value = if (self.current_function_return_type) |ret_type| blk: {
            const value_type = c.mlirValueGetType(v);
            if (!c.mlirTypeEqual(value_type, ret_type)) {
                // Convert to match return type (e.g., i1 -> i256 for bool -> u256)
                break :blk helpers.convertValueToType(self, v, ret_type, ret.span, loc);
            }
            break :blk v;
        } else v;

        const op = self.ora_dialect.createFuncReturnWithValue(final_value, loc);
        h.appendOp(self.block, op);
    } else {
        const op = self.ora_dialect.createFuncReturn(loc);
        h.appendOp(self.block, op);
    }
}

/// Get the span from an expression node
fn getExpressionSpan(expr: *const lib.ast.Expressions.ExprNode) lib.ast.SourceSpan {
    return switch (expr.*) {
        .Identifier => |ident| ident.span,
        .Literal => |lit| switch (lit) {
            .Integer => |int| int.span,
            .String => |str| str.span,
            .Bool => |bool_lit| bool_lit.span,
            .Address => |addr| addr.span,
            .Hex => |hex| hex.span,
            .Binary => |bin| bin.span,
            .Character => |char| char.span,
            .Bytes => |bytes| bytes.span,
        },
        .Binary => |bin| bin.span,
        .Unary => |unary| unary.span,
        .Assignment => |assign| assign.span,
        .CompoundAssignment => |comp_assign| comp_assign.span,
        .Call => |call| call.span,
        .Index => |index| index.span,
        .FieldAccess => |field| field.span,
        .Cast => |cast| cast.span,
        .Comptime => |comptime_expr| comptime_expr.span,
        .Old => |old| old.span,
        .Tuple => |tuple| tuple.span,
        .SwitchExpression => |switch_expr| switch_expr.span,
        .Quantified => |quantified| quantified.span,
        .Try => |try_expr| try_expr.span,
        .ErrorReturn => |error_ret| error_ret.span,
        .ErrorCast => |error_cast| error_cast.span,
        .Shift => |shift| shift.span,
        .StructInstantiation => |struct_inst| struct_inst.span,
        .AnonymousStruct => |anon_struct| anon_struct.span,
        .Range => |range| range.span,
        .LabeledBlock => |labeled_block| labeled_block.span,
        .Destructuring => |destructuring| destructuring.span,
        .EnumLiteral => |enum_lit| enum_lit.span,
        .ArrayLiteral => |array_lit| array_lit.span,
    };
}

/// Lower ensures clauses before a return statement
pub fn lowerEnsuresBeforeReturn(self: *const StatementLowerer, block: c.MlirBlock, span: lib.ast.SourceSpan) LoweringError!void {
    _ = span; // Unused parameter

    for (self.ensures_clauses, 0..) |clause, i| {
        // Lower the ensures expression
        const condition_value = self.expr_lowerer.lowerExpression(clause);

        // Create an assertion operation with comprehensive verification attributes
        // Get the clause's span by switching on the expression type
        const clause_span = getExpressionSpan(clause);
        var assert_state = h.opState("cf.assert", self.fileLoc(clause_span));

        // Add the condition as an operand
        c.mlirOperationStateAddOperands(&assert_state, 1, @ptrCast(&condition_value));

        // Collect verification attributes
        var attributes = std.ArrayList(c.MlirNamedAttribute){};
        defer attributes.deinit(self.allocator);

        // Add required 'msg' attribute first (cf.assert requires this)
        const msg_text = try std.fmt.allocPrint(self.allocator, "Postcondition {d} failed", .{i});
        defer self.allocator.free(msg_text);
        const msg_attr = h.stringAttr(self.ctx, msg_text);
        const msg_id = h.identifier(self.ctx, "msg");
        try attributes.append(self.allocator, c.mlirNamedAttributeGet(msg_id, msg_attr));

        // Add ora.ensures attribute to mark this as a postcondition
        const ensures_attr = c.mlirBoolAttrGet(self.ctx, 1);
        const ensures_id = h.identifier(self.ctx, "ora.ensures");
        attributes.append(self.allocator, c.mlirNamedAttributeGet(ensures_id, ensures_attr)) catch {};

        // Add verification context attribute
        const context_attr = c.mlirStringAttrGet(self.ctx, h.strRef("function_postcondition"));
        const context_id = h.identifier(self.ctx, "ora.verification_context");
        attributes.append(self.allocator, c.mlirNamedAttributeGet(context_id, context_attr)) catch {};

        // Add verification marker for formal verification tools
        const verification_attr = c.mlirBoolAttrGet(self.ctx, 1);
        const verification_id = h.identifier(self.ctx, "ora.verification");
        attributes.append(self.allocator, c.mlirNamedAttributeGet(verification_id, verification_attr)) catch {};

        // Add postcondition index for multiple ensures clauses
        const index_attr = c.mlirIntegerAttrGet(c.mlirIntegerTypeGet(self.ctx, 32), @intCast(i));
        const index_id = h.identifier(self.ctx, "ora.postcondition_index");
        attributes.append(self.allocator, c.mlirNamedAttributeGet(index_id, index_attr)) catch {};

        // Apply all attributes
        c.mlirOperationStateAddAttributes(&assert_state, @intCast(attributes.items.len), attributes.items.ptr);

        const assert_op = c.mlirOperationCreate(&assert_state);
        h.appendOp(block, assert_op);
    }
}

/// Lower return statements in control flow context using scf.yield
pub fn lowerReturnInControlFlow(self: *const StatementLowerer, ret: *const lib.ast.Statements.ReturnNode) LoweringError!void {
    const loc = self.fileLoc(ret.span);

    if (ret.value) |e| {
        var v = self.expr_lowerer.lowerExpression(&e);

        // Insert refinement guard if return type is a refinement type
        if (self.current_function_return_type_info) |return_type_info| {
            if (return_type_info.ora_type) |ora_type| {
                v = try helpers.insertRefinementGuard(self, v, ora_type, ret.span, ret.skip_guard);
            }
        }

        const op = self.ora_dialect.createScfYieldWithValues(&[_]c.MlirValue{v}, loc);
        h.appendOp(self.block, op);
    } else {
        const op = self.ora_dialect.createScfYield(loc);
        h.appendOp(self.block, op);
    }
}
