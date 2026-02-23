// ============================================================================
// Return Statement Lowering
// ============================================================================

const std = @import("std");
const c = @import("mlir_c_api").c;
const lib = @import("ora_lib");
const h = @import("../helpers.zig");
const log = @import("log");
const StatementLowerer = @import("statement_lowerer.zig").StatementLowerer;
const LoweringError = StatementLowerer.LoweringError;
const helpers = @import("helpers.zig");
const dedup = @import("../refinement_guard_helpers.zig");

/// Lower return statements using ora.return with proper value handling
pub fn lowerReturn(self: *const StatementLowerer, ret: *const lib.ast.Statements.ReturnNode) LoweringError!void {
    // if we're inside a try block, we can't use ora.return (it must be last in function block)
    // instead, store the return value in memrefs and set the return flag
    if (self.in_try_block and self.try_return_flag_memref != null) {
        const loc = self.fileLoc(ret.span);
        const return_flag_memref = self.try_return_flag_memref.?;
        var final_value: ?c.MlirValue = null;

        if (ret.value) |value_expr| {
            if (self.try_return_value_memref) |return_value_memref| {
                var v = self.expr_lowerer.lowerExpression(&value_expr);

                const memref_type = c.oraValueGetType(return_value_memref);
                const element_type = c.oraShapedTypeGetElementType(memref_type);

                const is_error_union = if (self.current_function_return_type_info) |ti|
                    helpers.isErrorUnionTypeInfo(ti)
                else
                    false;

                if (is_error_union) {
                    const err_info = helpers.getErrorUnionPayload(self, &value_expr, v, element_type, self.block, loc);
                    v = helpers.encodeErrorUnionValue(self, err_info.payload, err_info.is_error, element_type, self.block, ret.span, loc);
                } else {
                    if (self.current_function_return_type_info) |ti| {
                        if (ti.ora_type) |ora_type| {
                            v = try helpers.insertRefinementGuard(self, v, ora_type, ret.span, null, ret.skip_guard);
                        }
                    }
                    v = helpers.convertValueToType(self, v, element_type, ret.span, loc);
                }
                final_value = v;
            }
        }

        // insert ensures clause checks before recording the try-return
        if (self.ensures_clauses.len > 0) {
            try lowerEnsuresBeforeReturn(self, self.block, ret.span, final_value);
        }

        const true_val = helpers.createBoolConstant(self, true, loc);
        helpers.storeToMemref(self, true_val, return_flag_memref, loc);

        if (self.try_return_value_memref) |return_value_memref| {
            if (final_value) |v| {
                helpers.storeToMemref(self, v, return_value_memref, loc);
            }
        }

        return;
    }

    const loc = self.fileLoc(ret.span);
    log.debug("[lowerReturn] self.block ptr = {*}, self.expr_lowerer.block ptr = {*}\n", .{ self.block.ptr, self.expr_lowerer.block.ptr });

    var final_value: ?c.MlirValue = null;

    if (ret.value) |e| {
        var v = self.expr_lowerer.lowerExpression(&e);

        // convert/wrap return value to match function return type if available
        const converted_value = if (self.current_function_return_type) |ret_type| blk: {
            const is_error_union = if (self.current_function_return_type_info) |ti|
                helpers.isErrorUnionTypeInfo(ti)
            else
                false;
            if (is_error_union) {
                const err_info = helpers.getErrorUnionPayload(self, &e, v, ret_type, self.block, loc);
                break :blk helpers.encodeErrorUnionValue(self, err_info.payload, err_info.is_error, ret_type, self.block, ret.span, loc);
            }
            if (self.current_function_return_type_info) |ti| {
                if (ti.ora_type) |ora_type| {
                    v = try helpers.insertRefinementGuard(self, v, ora_type, ret.span, null, ret.skip_guard);
                }
            }

            const value_type = c.oraValueGetType(v);
            if (!c.oraTypeEqual(value_type, ret_type)) {
                // convert to match return type (e.g., i1 -> i256 for bool -> u256)
                break :blk helpers.convertValueToType(self, v, ret_type, ret.span, loc);
            }
            break :blk v;
        } else v;
        final_value = converted_value;
    }

    // insert ensures clause checks before return (postconditions must hold at every return point)
    if (self.ensures_clauses.len > 0) {
        try lowerEnsuresBeforeReturn(self, self.block, ret.span, final_value);
    }

    if (final_value) |v| {
        const op = self.ora_dialect.createFuncReturnWithValue(v, loc);
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
pub fn lowerEnsuresBeforeReturn(self: *const StatementLowerer, block: c.MlirBlock, span: lib.ast.SourceSpan, return_value: ?c.MlirValue) LoweringError!void {
    _ = span;
    const function_name_for_dedup = self.expr_lowerer.current_function_name orelse resolveCurrentFunctionNameFromBlock(block);

    const return_refinement: ?lib.ast.Types.OraType = if (self.current_function_return_type_info) |ti|
        if (ti.ora_type) |ot| switch (ot) {
            .min_value, .max_value, .in_range, .non_zero_address => ot,
            else => null,
        } else null
    else
        null;

    for (self.ensures_clauses, 0..) |clause, i| {
        var ensures_expr_lowerer = self.expr_lowerer.*;
        ensures_expr_lowerer.block = block;
        ensures_expr_lowerer.postcondition_return_value = return_value;

        const condition_value = ensures_expr_lowerer.lowerExpression(clause);

        const clause_span = getExpressionSpan(clause);
        const clause_loc = self.fileLoc(clause_span);
        const final_condition = gateEnsuresForErrorUnion(self, block, condition_value, return_value, clause_loc);

        // If the postcondition is already guaranteed by the return type refinement,
        // keep it for SMT reasoning but avoid emitting a duplicate runtime check.
        if (return_refinement) |ret_ref| {
            if (dedup.isEnsuresCoveredByReturnRefinement(
                clause,
                ret_ref,
                function_name_for_dedup,
                self.param_map,
            )) {
                appendEnsuresAssume(self, block, final_condition, clause_loc, i);
                continue;
            }
        }

        var attributes = std.ArrayList(c.MlirNamedAttribute){};
        defer attributes.deinit(self.allocator);

        const msg_text = try std.fmt.allocPrint(self.allocator, "Postcondition {d} failed", .{i});
        defer self.allocator.free(msg_text);
        const msg_attr = h.stringAttr(self.ctx, msg_text);
        const msg_id = h.identifier(self.ctx, "msg");
        try attributes.append(self.allocator, c.oraNamedAttributeGet(msg_id, msg_attr));

        const ensures_attr = h.boolAttr(self.ctx, 1);
        const ensures_id = h.identifier(self.ctx, "ora.ensures");
        attributes.append(self.allocator, c.oraNamedAttributeGet(ensures_id, ensures_attr)) catch {};

        const context_attr = c.oraStringAttrCreate(self.ctx, h.strRef("function_postcondition"));
        const context_id = h.identifier(self.ctx, "ora.verification_context");
        attributes.append(self.allocator, c.oraNamedAttributeGet(context_id, context_attr)) catch {};

        const verification_attr = h.boolAttr(self.ctx, 1);
        const verification_id = h.identifier(self.ctx, "ora.verification");
        attributes.append(self.allocator, c.oraNamedAttributeGet(verification_id, verification_attr)) catch {};

        const index_attr = c.oraIntegerAttrCreateI64FromType(c.oraIntegerTypeCreate(self.ctx, 32), @intCast(i));
        const index_id = h.identifier(self.ctx, "ora.postcondition_index");
        attributes.append(self.allocator, c.oraNamedAttributeGet(index_id, index_attr)) catch {};

        const assert_op = self.ora_dialect.createCfAssertWithAttrs(final_condition, attributes.items, clause_loc);
        h.appendOp(block, assert_op);
    }
}

fn resolveCurrentFunctionNameFromBlock(block: c.MlirBlock) ?[]const u8 {
    var current_block = block;
    while (!c.mlirBlockIsNull(current_block)) {
        const parent_op = c.mlirBlockGetParentOperation(current_block);
        if (c.oraOperationIsNull(parent_op)) return null;

        const parent_name_ref = c.oraOperationGetName(parent_op);
        if (parent_name_ref.data != null and std.mem.eql(u8, parent_name_ref.data[0..parent_name_ref.length], "func.func")) {
            const sym_name_attr = c.oraOperationGetAttributeByName(parent_op, h.strRef("sym_name"));
            if (c.oraAttributeIsNull(sym_name_attr)) return null;
            const sym_name_ref = c.oraStringAttrGetValue(sym_name_attr);
            if (sym_name_ref.data == null or sym_name_ref.length == 0) return null;
            return sym_name_ref.data[0..sym_name_ref.length];
        }

        current_block = c.mlirOperationGetBlock(parent_op);
    }
    return null;
}

fn appendEnsuresAssume(
    self: *const StatementLowerer,
    block: c.MlirBlock,
    condition_value: c.MlirValue,
    clause_loc: c.MlirLocation,
    index: usize,
) void {
    const assume_op = self.ora_dialect.createAssume(condition_value, clause_loc);
    c.oraOperationSetAttributeByName(
        assume_op,
        h.strRef("ora.assume_origin"),
        c.oraStringAttrCreate(self.ctx, h.strRef("user")),
    );
    c.oraOperationSetAttributeByName(assume_op, h.strRef("ora.verification"), h.boolAttr(self.ctx, 1));
    c.oraOperationSetAttributeByName(assume_op, h.strRef("ora.formal"), h.boolAttr(self.ctx, 1));
    c.oraOperationSetAttributeByName(
        assume_op,
        h.strRef("ora.verification_type"),
        c.oraStringAttrCreate(self.ctx, h.strRef("assume")),
    );
    c.oraOperationSetAttributeByName(
        assume_op,
        h.strRef("ora.verification_context"),
        c.oraStringAttrCreate(self.ctx, h.strRef("function_postcondition")),
    );
    c.oraOperationSetAttributeByName(
        assume_op,
        h.strRef("ora.postcondition_index"),
        c.oraIntegerAttrCreateI64FromType(c.oraIntegerTypeCreate(self.ctx, 32), @intCast(index)),
    );
    h.appendOp(block, assume_op);
}

/// For error-union returns, enforce ensures only on the success path:
///   (!is_error(return_value)) => ensures_condition
/// encoded as:
///   is_error(return_value) || ensures_condition
fn gateEnsuresForErrorUnion(
    self: *const StatementLowerer,
    block: c.MlirBlock,
    condition_value: c.MlirValue,
    return_value: ?c.MlirValue,
    loc: c.MlirLocation,
) c.MlirValue {
    const rv = return_value orelse return condition_value;
    const ret_ti = self.current_function_return_type_info orelse return condition_value;
    if (!helpers.isErrorUnionTypeInfo(ret_ti)) return condition_value;

    const is_error_op = self.ora_dialect.createErrorIsError(rv, loc);
    h.appendOp(block, is_error_op);
    const is_error_val = h.getResult(is_error_op, 0);

    const gated_op = c.oraArithOrIOpCreate(self.ctx, loc, is_error_val, condition_value);
    h.appendOp(block, gated_op);
    return h.getResult(gated_op, 0);
}

/// Lower return statements in control flow context using scf.yield
pub fn lowerReturnInControlFlow(self: *const StatementLowerer, ret: *const lib.ast.Statements.ReturnNode) LoweringError!void {
    const loc = self.fileLoc(ret.span);

    if (ret.value) |e| {
        var v = self.expr_lowerer.lowerExpression(&e);

        // convert/wrap return value to match function return type if available
        // this ensures literals like `return 1` coerce to refinement types like MinValue<u256, 1>
        const final_value = if (self.current_function_return_type) |ret_type| blk: {
            const is_error_union = if (self.current_function_return_type_info) |ti|
                helpers.isErrorUnionTypeInfo(ti)
            else
                false;
            if (is_error_union) {
                const err_info = helpers.getErrorUnionPayload(self, &e, v, ret_type, self.block, loc);
                break :blk helpers.encodeErrorUnionValue(self, err_info.payload, err_info.is_error, ret_type, self.block, ret.span, loc);
            }
            if (self.current_function_return_type_info) |ti| {
                if (ti.ora_type) |ora_type| {
                    v = try helpers.insertRefinementGuard(self, v, ora_type, ret.span, null, ret.skip_guard);
                }
            }
            const value_type = c.oraValueGetType(v);
            if (!c.oraTypeEqual(value_type, ret_type)) {
                break :blk helpers.convertValueToType(self, v, ret_type, ret.span, loc);
            }
            break :blk v;
        } else v;

        const op = self.ora_dialect.createScfYieldWithValues(&[_]c.MlirValue{final_value}, loc);
        h.appendOp(self.block, op);
    } else {
        const op = self.ora_dialect.createScfYield(loc);
        h.appendOp(self.block, op);
    }
}
