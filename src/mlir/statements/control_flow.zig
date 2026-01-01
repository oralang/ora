// ============================================================================
// Control Flow Statement Lowering
// ============================================================================
// If, while, for, and switch statement lowering

const std = @import("std");
const c = @import("mlir_c_api").c;
const lib = @import("ora_lib");
const h = @import("../helpers.zig");
const constants = @import("../lower.zig");
const StatementLowerer = @import("statement_lowerer.zig").StatementLowerer;
const LabelContext = @import("statement_lowerer.zig").LabelContext;
const LoweringError = StatementLowerer.LoweringError;
const ExpressionLowerer = @import("../expressions.zig").ExpressionLowerer;
const helpers = @import("helpers.zig");
const verification = @import("verification.zig");
const log = @import("log");

/// Lower if statements using ora.if with then/else regions
pub fn lowerIf(self: *const StatementLowerer, if_stmt: *const lib.ast.Statements.IfNode) LoweringError!void {
    const loc = self.fileLoc(if_stmt.span);

    // lower the condition expression
    const condition = self.expr_lowerer.lowerExpression(&if_stmt.condition);

    // check if this if statement contains return statements
    if (helpers.ifStatementHasReturns(self, if_stmt)) {
        // for if statements with returns, use scf.if with scf.yield
        try lowerIfWithReturns(self, if_stmt, condition, loc);
        return;
    }

    // create the ora.if operation using C++ API (enables custom assembly formats)
    const if_op = self.ora_dialect.createIf(condition, loc);

    // get then and else regions
    const then_block = c.oraIfOpGetThenBlock(if_op);
    const else_block = c.oraIfOpGetElseBlock(if_op);
    if (c.oraBlockIsNull(then_block) or c.oraBlockIsNull(else_block)) {
        @panic("ora.if missing then/else blocks");
    }

    // lower else branch if present, otherwise add ora.yield to empty region
    if (if_stmt.else_branch) |else_branch| {
        _ = try self.lowerBlockBody(else_branch, else_block);
    } else {
        // add ora.yield to empty else region to satisfy MLIR requirements
        const yield_op = self.ora_dialect.createYield(&[_]c.MlirValue{}, loc);
        h.appendOp(else_block, yield_op);
    }

    // lower then branch FIRST (before creating the ora.if operation)
    _ = try self.lowerBlockBody(if_stmt.then_branch, then_block);

    // add ora.yield to then region if it doesn't end with one
    if (!helpers.blockEndsWithYield(self, then_block)) {
        const yield_op = self.ora_dialect.createYield(&[_]c.MlirValue{}, loc);
        h.appendOp(then_block, yield_op);
    }

    // add ora.yield to else region if it doesn't end with one (for non-empty else branches)
    if (if_stmt.else_branch != null and !helpers.blockEndsWithYield(self, else_block)) {
        const yield_op = self.ora_dialect.createYield(&[_]c.MlirValue{}, loc);
        h.appendOp(else_block, yield_op);
    }

    // now append the scf.if operation to the block
    h.appendOp(self.block, if_op);
}

/// Lower if statements with returns by using scf.if with scf.yield and single return
fn lowerIfWithReturns(self: *const StatementLowerer, if_stmt: *const lib.ast.Statements.IfNode, condition: c.MlirValue, loc: c.MlirLocation) LoweringError!void {
    // for if statements with returns, we need to restructure the logic:
    // 1. Use scf.if with scf.yield to pass values out of regions
    // 2. Have a single func.return at the end that uses the result from scf.if

    // determine the result type from the return statements
    const result_type = helpers.getReturnTypeFromIfStatement(self, if_stmt);
    var result_types: [1]c.MlirType = undefined;
    var result_slice: []c.MlirType = &[_]c.MlirType{};
    if (result_type) |ret_type| {
        result_types[0] = ret_type;
        result_slice = result_types[0..1];
    }

    // create the scf.if operation with proper then/else regions
    const op = self.ora_dialect.createScfIf(condition, result_slice, loc);
    const then_block = c.oraScfIfOpGetThenBlock(op);
    const else_block = c.oraScfIfOpGetElseBlock(op);
    if (c.oraBlockIsNull(then_block) or c.oraBlockIsNull(else_block)) {
        @panic("scf.if missing then/else blocks");
    }

    // lower then branch - replace return statements with scf.yield
    const then_has_return = helpers.blockHasReturn(self, if_stmt.then_branch);
    try lowerBlockBodyWithYield(self, if_stmt.then_branch, then_block);

    // if then branch doesn't end with a yield, add one with a default value
    if (!then_has_return and result_type != null) {
        if (result_type) |ret_type| {
            const default_val = try helpers.createDefaultValueForType(self, ret_type, loc);
            const yield_op = self.ora_dialect.createScfYieldWithValues(&[_]c.MlirValue{default_val}, loc);
            h.appendOp(then_block, yield_op);
        }
    }

    // lower else branch if present, otherwise add scf.yield with default value
    if (if_stmt.else_branch) |else_branch| {
        const else_has_return = helpers.blockHasReturn(self, else_branch);
        try lowerBlockBodyWithYield(self, else_branch, else_block);

        // if else branch doesn't end with a yield, add one with a default value
        if (!else_has_return and result_type != null) {
            if (result_type) |ret_type| {
                const default_val = try helpers.createDefaultValueForType(self, ret_type, loc);
                const yield_op = self.ora_dialect.createScfYieldWithValues(&[_]c.MlirValue{default_val}, loc);
                h.appendOp(else_block, yield_op);
            }
        }
    } else {
        // no else branch - add scf.yield with default value if needed
        if (result_type) |ret_type| {
            const default_val = try helpers.createDefaultValueForType(self, ret_type, loc);
            const yield_op = self.ora_dialect.createScfYieldWithValues(&[_]c.MlirValue{default_val}, loc);
            h.appendOp(else_block, yield_op);
        } else {
            const yield_op = self.ora_dialect.createScfYield(loc);
            h.appendOp(else_block, yield_op);
        }
    }

    // append the scf.if operation to the block
    h.appendOp(self.block, op);

    // if both branches return (no subsequent statements), add func.return
    // this handles the case where the entire function is just an if-else with returns
    if (result_type) |_| {
        const then_returns = helpers.blockHasReturn(self, if_stmt.then_branch);
        const else_returns = if (if_stmt.else_branch) |else_branch| helpers.blockHasReturn(self, else_branch) else false;

        // if both branches return, this is the function's final statement
        if (then_returns and else_returns) {
            const result_value = h.getResult(op, 0);
            const return_op = self.ora_dialect.createFuncReturnWithValue(result_value, loc);
            h.appendOp(self.block, return_op);
        }
    }
}

/// Lower block body with yield - replaces return statements with scf.yield
fn lowerBlockBodyWithYield(self: *const StatementLowerer, block_body: lib.ast.Statements.BlockNode, target_block: c.MlirBlock) LoweringError!void {
    log.debug("[lowerBlockBodyWithYield] Starting, block has {} statements\n", .{block_body.statements.len});
    // create a temporary lowerer for this block by copying the current one and changing the block
    var temp_lowerer = self.*;
    temp_lowerer.block = target_block;

    // create a new expression lowerer with the target block to ensure constants are created in the correct block
    var expr_lowerer = ExpressionLowerer.init(
        self.ctx,
        target_block,
        self.expr_lowerer.type_mapper,
        self.expr_lowerer.param_map,
        self.expr_lowerer.storage_map,
        self.expr_lowerer.local_var_map,
        self.expr_lowerer.symbol_table,
        self.expr_lowerer.builtin_registry,
        self.expr_lowerer.error_handler,
        self.expr_lowerer.locations,
        self.ora_dialect,
    );
    expr_lowerer.current_function_return_type = self.current_function_return_type;
    expr_lowerer.current_function_return_type_info = self.current_function_return_type_info;
    expr_lowerer.in_try_block = self.in_try_block;

    // track if we've added a terminator to this block
    var has_terminator = false;

    // lower each statement, replacing returns with yields
    for (block_body.statements) |stmt| {
        // if we've already added a terminator, skip remaining statements
        if (has_terminator) break;

        switch (stmt) {
            .Return => |ret| {
                // if we're inside a try block, use memref-based approach instead of scf.yield
                if (self.in_try_block and self.try_return_flag_memref != null) {
                    log.debug("[lowerBlockBodyWithYield] Return inside try block - using memref approach\n", .{});
                    const loc = temp_lowerer.fileLoc(ret.span);
                    const return_flag_memref = self.try_return_flag_memref.?;

                    // set return flag to true
                    const true_val = helpers.createBoolConstant(&temp_lowerer, true, loc);
                    helpers.storeToMemref(&temp_lowerer, true_val, return_flag_memref, loc);

                    // store return value if present
                    if (ret.value) |value_expr| {
                        if (self.try_return_value_memref) |return_value_memref| {
                            var v = expr_lowerer.lowerExpression(&value_expr);

                            // insert refinement guard if return type is a refinement type
                            if (temp_lowerer.current_function_return_type_info) |return_type_info| {
                                if (return_type_info.ora_type) |ora_type| {
                                    v = try helpers.insertRefinementGuard(&temp_lowerer, v, ora_type, ret.span, ret.skip_guard);
                                }
                            }

                            // get target type from memref element type
                            const memref_type = c.oraValueGetType(return_value_memref);
                            const element_type = c.oraShapedTypeGetElementType(memref_type);

                            // convert value to match memref element type
                            const final_value = helpers.convertValueToType(&temp_lowerer, v, element_type, ret.span, loc);

                            // store return value
                            helpers.storeToMemref(&temp_lowerer, final_value, return_value_memref, loc);
                        }
                    }

                    // use empty scf.yield to terminate the block (the actual return happens after try/catch)
                    const yield_op = temp_lowerer.ora_dialect.createScfYield(loc);
                    h.appendOp(target_block, yield_op);
                    has_terminator = true;
                    continue;
                }

                // replace return with scf.yield (normal case, not in try block)
                log.debug("[lowerBlockBodyWithYield] Converting return to scf.yield\n", .{});
                const loc = temp_lowerer.fileLoc(ret.span);

                if (ret.value) |e| {
                    const v = expr_lowerer.lowerExpression(&e);
                    // convert return value to match function return type if available
                    const final_value = if (temp_lowerer.current_function_return_type) |ret_type| blk: {
                        const value_type = c.oraValueGetType(v);
                        if (!c.oraTypeEqual(value_type, ret_type)) {
                            // convert to match return type (e.g., i256 -> i8 for u8 return)
                            break :blk helpers.convertValueToType(&temp_lowerer, v, ret_type, ret.span, loc);
                        }
                        break :blk v;
                    } else v;
                    const yield_op = temp_lowerer.ora_dialect.createScfYieldWithValues(&[_]c.MlirValue{final_value}, loc);
                    h.appendOp(target_block, yield_op);
                    log.debug("[lowerBlockBodyWithYield] Added scf.yield with value\n", .{});
                } else {
                    const yield_op = temp_lowerer.ora_dialect.createScfYield(loc);
                    h.appendOp(target_block, yield_op);
                    log.debug("[lowerBlockBodyWithYield] Added scf.yield without value\n", .{});
                }
                has_terminator = true;
            },
            .If => |if_stmt| {
                // handle nested if statements with returns
                const loc = temp_lowerer.fileLoc(if_stmt.span);
                const condition = expr_lowerer.lowerExpression(&if_stmt.condition);

                // if the nested if has returns, handle it specially
                if (helpers.ifStatementHasReturns(&temp_lowerer, &if_stmt)) {
                    const result_type = helpers.getReturnTypeFromIfStatement(&temp_lowerer, &if_stmt);
                    var result_types: [1]c.MlirType = undefined;
                    var result_slice: []c.MlirType = &[_]c.MlirType{};
                    if (result_type) |ret_type| {
                        result_types[0] = ret_type;
                        result_slice = result_types[0..1];
                    }

                    const if_op = temp_lowerer.ora_dialect.createScfIf(condition, result_slice, loc);
                    const then_block = c.oraScfIfOpGetThenBlock(if_op);
                    const else_block = c.oraScfIfOpGetElseBlock(if_op);
                    if (c.oraBlockIsNull(then_block) or c.oraBlockIsNull(else_block)) {
                        @panic("scf.if missing then/else blocks");
                    }

                    // lower then branch
                    const then_has_return = helpers.blockHasReturn(&temp_lowerer, if_stmt.then_branch);
                    try lowerBlockBodyWithYield(&temp_lowerer, if_stmt.then_branch, then_block);

                    if (!then_has_return and result_type != null) {
                        if (result_type) |ret_type| {
                            const default_val = try helpers.createDefaultValueForType(&temp_lowerer, ret_type, loc);
                            const yield_op = temp_lowerer.ora_dialect.createScfYieldWithValues(&[_]c.MlirValue{default_val}, loc);
                            h.appendOp(then_block, yield_op);
                        }
                    }

                    // lower else branch
                    if (if_stmt.else_branch) |else_branch| {
                        const else_has_return = helpers.blockHasReturn(&temp_lowerer, else_branch);
                        try lowerBlockBodyWithYield(&temp_lowerer, else_branch, else_block);

                        if (!else_has_return and result_type != null) {
                            if (result_type) |ret_type| {
                                const default_val = try helpers.createDefaultValueForType(&temp_lowerer, ret_type, loc);
                                const yield_op = temp_lowerer.ora_dialect.createScfYieldWithValues(&[_]c.MlirValue{default_val}, loc);
                                h.appendOp(else_block, yield_op);
                            }
                        }
                    } else {
                        if (result_type) |ret_type| {
                            const default_val = try helpers.createDefaultValueForType(&temp_lowerer, ret_type, loc);
                            const yield_op = temp_lowerer.ora_dialect.createScfYieldWithValues(&[_]c.MlirValue{default_val}, loc);
                            h.appendOp(else_block, yield_op);
                        } else {
                            const yield_op = temp_lowerer.ora_dialect.createScfYield(loc);
                            h.appendOp(else_block, yield_op);
                        }
                    }

                    h.appendOp(target_block, if_op);

                    // if the nested scf.if has a result, we need to yield it
                    // this is because we're inside an outer scf.if's else block, and we need to yield the result
                    if (result_type) |_| {
                        const then_returns = helpers.blockHasReturn(&temp_lowerer, if_stmt.then_branch);
                        const else_returns = if (if_stmt.else_branch) |else_branch| helpers.blockHasReturn(&temp_lowerer, else_branch) else false;

                        if (then_returns and else_returns) {
                            // both branches return - yield the result from nested scf.if
                            const nested_result = h.getResult(if_op, 0);
                            const yield_op = temp_lowerer.ora_dialect.createScfYieldWithValues(&[_]c.MlirValue{nested_result}, loc);
                            h.appendOp(target_block, yield_op);
                            has_terminator = true;
                        } else {
                            // not all branches return - still need to yield if we have a result type
                            // the nested scf.if will handle its own yields, but we need to yield its result here
                            const nested_result = h.getResult(if_op, 0);
                            const yield_op = temp_lowerer.ora_dialect.createScfYieldWithValues(&[_]c.MlirValue{nested_result}, loc);
                            h.appendOp(target_block, yield_op);
                            has_terminator = true;
                        }
                    }
                } else {
                    // regular if statement without returns - lower normally
                    var temp_lowerer2 = temp_lowerer;
                    temp_lowerer2.expr_lowerer = &expr_lowerer;
                    try lowerIf(&temp_lowerer2, &if_stmt);
                }
            },
            else => {
                // lower other statements normally
                var temp_lowerer2 = temp_lowerer;
                temp_lowerer2.expr_lowerer = &expr_lowerer;
                try temp_lowerer2.lowerStatement(&stmt);
            },
        }
    }
}

/// Lower while loop statements using ora.while
pub fn lowerWhile(self: *const StatementLowerer, while_stmt: *const lib.ast.Statements.WhileNode) LoweringError!void {
    const loc = self.fileLoc(while_stmt.span);

    // lower condition first (before creating the while operation)
    const condition_raw = self.expr_lowerer.lowerExpression(&while_stmt.condition);

    // ensure condition is boolean (i1) - ora.while requires I1 type
    const condition_type = c.oraValueGetType(condition_raw);
    const bool_ty = h.boolType(self.ctx);
    const condition = if (c.oraTypeEqual(condition_type, bool_ty))
        condition_raw
    else
        helpers.convertValueToType(self, condition_raw, bool_ty, while_stmt.span, loc);

    // create ora.while operation using C++ API (enables custom assembly formats)
    // note: ora.while has a simpler structure - condition is an operand, body is a region
    const op = self.ora_dialect.createWhile(condition, loc);
    h.appendOp(self.block, op);

    const body_block = c.oraWhileOpGetBodyBlock(op);
    if (c.oraBlockIsNull(body_block)) {
        @panic("ora.while missing body block");
    }

    // create expression lowerer for loop body (uses body_block, shares local_var_map for memref access)
    var body_expr_lowerer = ExpressionLowerer.init(
        self.ctx,
        body_block,
        self.type_mapper,
        self.expr_lowerer.param_map,
        self.expr_lowerer.storage_map,
        self.expr_lowerer.local_var_map,
        self.expr_lowerer.symbol_table,
        self.expr_lowerer.builtin_registry,
        self.expr_lowerer.error_handler,
        self.expr_lowerer.locations,
        self.ora_dialect,
    );
    body_expr_lowerer.current_function_return_type = self.current_function_return_type;
    body_expr_lowerer.current_function_return_type_info = self.current_function_return_type_info;
    body_expr_lowerer.in_try_block = self.in_try_block;

    // lower loop invariants if present (in body block, using body expression lowerer)
    for (while_stmt.invariants) |*invariant| {
        const invariant_value = body_expr_lowerer.lowerExpression(invariant);

        const inv_op = self.ora_dialect.createInvariant(invariant_value, loc);
        verification.addVerificationAttributesToOp(self, inv_op, "invariant", "loop_invariant");
        h.appendOp(body_block, inv_op);
    }

    // lower decreases clause if present (using body expression lowerer)
    if (while_stmt.decreases) |decreases_expr| {
        const decreases_value = body_expr_lowerer.lowerExpression(decreases_expr);
        const dec_op = self.ora_dialect.createDecreases(decreases_value, loc);
        // add verification attributes using C API (after creation)
        verification.addVerificationAttributesToOp(self, dec_op, "decreases", "loop_termination_measure");
        h.appendOp(body_block, dec_op);
    }

    // lower increases clause if present (using body expression lowerer)
    if (while_stmt.increases) |increases_expr| {
        const increases_value = body_expr_lowerer.lowerExpression(increases_expr);
        const inc_op = self.ora_dialect.createIncreases(increases_value, loc);
        // add verification attributes using C API (after creation)
        verification.addVerificationAttributesToOp(self, inc_op, "increases", "loop_progress_measure");
        h.appendOp(body_block, inc_op);
    }

    // lower body in body region with loop label context
    const loop_label = while_stmt.label orelse "";
    const loop_ctx = LabelContext{
        .label = loop_label,
        .label_type = .While,
        .parent = self.label_context,
    };
    var body_lowerer = self.*;
    body_lowerer.label_context = &loop_ctx;
    _ = try body_lowerer.lowerBlockBody(while_stmt.body, body_block);

    // add ora.yield at end of body to continue loop
    const yield_op = self.ora_dialect.createYield(&[_]c.MlirValue{}, loc);
    h.appendOp(body_block, yield_op);
}

/// Lower for loop statements using scf.for with proper iteration variables
pub fn lowerFor(self: *const StatementLowerer, for_stmt: *const lib.ast.Statements.ForLoopNode) LoweringError!void {
    const loc = self.fileLoc(for_stmt.span);

    // lower the iterable expression
    const iterable = self.expr_lowerer.lowerExpression(&for_stmt.iterable);

    // handle different loop patterns
    switch (for_stmt.pattern) {
        .Single => |single| {
            try lowerSimpleForLoop(self, single.name, iterable, for_stmt.body, for_stmt.invariants, for_stmt.decreases, for_stmt.increases, for_stmt.label, loc);
        },
        .IndexPair => |pair| {
            try lowerIndexedForLoop(self, pair.item, pair.index, iterable, for_stmt.body, for_stmt.invariants, for_stmt.decreases, for_stmt.increases, for_stmt.label, loc);
        },
        .Destructured => |destructured| {
            try lowerDestructuredForLoop(self, destructured.pattern, iterable, for_stmt.body, for_stmt.invariants, for_stmt.decreases, for_stmt.increases, for_stmt.label, loc);
        },
    }
}

/// Lower simple for loop (for (iterable) |item| body)
fn lowerSimpleForLoop(
    self: *const StatementLowerer,
    item_name: []const u8,
    iterable: c.MlirValue,
    body: lib.ast.Statements.BlockNode,
    invariants: []lib.ast.Expressions.ExprNode,
    decreases: ?*lib.ast.Expressions.ExprNode,
    increases: ?*lib.ast.Expressions.ExprNode,
    label: ?[]const u8,
    loc: c.MlirLocation,
) LoweringError!void {
    // get the iterable type to determine proper iteration strategy
    const iterable_ty = c.oraValueGetType(iterable);
    const index_ty = c.oraIndexTypeCreate(self.ctx);

    // determine iteration strategy based on type
    var lower_bound: c.MlirValue = undefined;
    var upper_bound: c.MlirValue = undefined;
    var step: c.MlirValue = undefined;

    const span = body.span;

    // decide how to obtain an index-typed upper bound:
    // - If the iterable is already an integer, cast once to index (range-style loop)
    // - If it's a memref or other shaped/collection type (tensor, slice, map),
    //   use createLengthAccess/ora.length which returns index.
    if (c.oraTypeIsAInteger(iterable_ty)) {
        const upper_raw = iterable;
        upper_bound = self.expr_lowerer.convertIndexToIndexType(upper_raw, span);
    } else if (c.oraTypeIsAMemRef(iterable_ty) or c.oraTypeIsAShaped(iterable_ty)) {
        const len_index = self.expr_lowerer.createLengthAccess(iterable, span);
        upper_bound = len_index;
    } else {
        // fallback: treat as range upper bound and cast to index
        upper_bound = self.expr_lowerer.convertIndexToIndexType(iterable, span);
    }

    // create constants for loop bounds (index-typed)
    const zero_op = self.ora_dialect.createArithConstant(0, index_ty, loc);
    h.appendOp(self.block, zero_op);
    lower_bound = h.getResult(zero_op, 0);

    // create step constant (index-typed)
    const step_op = self.ora_dialect.createArithConstant(1, index_ty, loc);
    h.appendOp(self.block, step_op);
    step = h.getResult(step_op, 0);

    // create scf.for operation
    const for_op = self.ora_dialect.createScfFor(lower_bound, upper_bound, step, &[_]c.MlirValue{}, &[_]c.MlirType{}, loc);
    h.appendOp(self.block, for_op);

    const body_block = c.oraScfForOpGetBodyBlock(for_op);
    if (c.oraBlockIsNull(body_block)) {
        @panic("scf.for missing body block");
    }

    // get the induction variable
    const induction_var = c.oraBlockGetArgument(body_block, 0);

    // set up a body-scoped expression lowerer so that any index→element
    // conversions and bounds checks land inside the loop body.
    var body_expr_lowerer = ExpressionLowerer.init(
        self.ctx,
        body_block,
        self.type_mapper,
        self.expr_lowerer.param_map,
        self.expr_lowerer.storage_map,
        self.expr_lowerer.local_var_map,
        self.expr_lowerer.symbol_table,
        self.expr_lowerer.builtin_registry,
        self.expr_lowerer.error_handler,
        self.expr_lowerer.locations,
        self.ora_dialect,
    );
    body_expr_lowerer.current_function_return_type = self.current_function_return_type;
    body_expr_lowerer.current_function_return_type_info = self.current_function_return_type_info;
    body_expr_lowerer.in_try_block = self.in_try_block;

    // bind the loop variable:
    // - For range-style loops (integer iterable), item is the index itself.
    // - For collection-style loops, item is the element at the current index.
    if (self.local_var_map) |lvm| {
        if (c.oraTypeIsAInteger(iterable_ty)) {
            // range-based: item is the index
            lvm.addLocalVar(item_name, induction_var) catch {
                log.warn("Failed to add loop variable to map: {s}\n", .{item_name});
            };
        } else if (c.oraTypeIsAMemRef(iterable_ty) or c.oraTypeIsAShaped(iterable_ty)) {
            const elem_value = body_expr_lowerer.createArrayIndexLoad(iterable, induction_var, span);
            lvm.addLocalVar(item_name, elem_value) catch {
                log.warn("Failed to add element variable to map: {s}\n", .{item_name});
            };
        } else {
            // fallback: expose the index directly
            lvm.addLocalVar(item_name, induction_var) catch {
                log.warn("Failed to add loop variable to map: {s}\n", .{item_name});
            };
        }
    }

    // lower loop invariants if present
    for (invariants) |*invariant| {
        const invariant_value = self.expr_lowerer.lowerExpression(invariant);
        const inv_op = self.ora_dialect.createInvariant(invariant_value, loc);
        verification.addVerificationAttributesToOp(self, inv_op, "invariant", "loop_invariant");
        h.appendOp(body_block, inv_op);
    }

    // lower decreases clause if present
    if (decreases) |decreases_expr| {
        const decreases_value = self.expr_lowerer.lowerExpression(decreases_expr);
        const dec_op = self.ora_dialect.createDecreases(decreases_value, loc);
        verification.addVerificationAttributesToOp(self, dec_op, "decreases", "loop_termination_measure");
        h.appendOp(body_block, dec_op);
    }

    // lower increases clause if present
    if (increases) |increases_expr| {
        const increases_value = self.expr_lowerer.lowerExpression(increases_expr);
        const inc_op = self.ora_dialect.createIncreases(increases_value, loc);
        verification.addVerificationAttributesToOp(self, inc_op, "increases", "loop_progress_measure");
        h.appendOp(body_block, inc_op);
    }

    // lower the loop body with loop label context
    const loop_label = label orelse "";
    const loop_ctx = LabelContext{
        .label = loop_label,
        .label_type = .For,
        .parent = self.label_context,
    };
    var body_lowerer = self.*;
    body_lowerer.label_context = &loop_ctx;
    const ended_with_terminator = try body_lowerer.lowerBlockBody(body, body_block);

    // add scf.yield at end of body if no terminator
    if (!ended_with_terminator) {
        const yield_op = self.ora_dialect.createScfYield(loc);
        h.appendOp(body_block, yield_op);
    }
}

/// Lower indexed for loop (for (iterable) |item, index| body)
fn lowerIndexedForLoop(
    self: *const StatementLowerer,
    item_name: []const u8,
    index_name: []const u8,
    iterable: c.MlirValue,
    body: lib.ast.Statements.BlockNode,
    invariants: []lib.ast.Expressions.ExprNode,
    decreases: ?*lib.ast.Expressions.ExprNode,
    increases: ?*lib.ast.Expressions.ExprNode,
    label: ?[]const u8,
    loc: c.MlirLocation,
) LoweringError!void {
    // use MLIR index type for loop bounds and induction variable
    const index_ty = c.oraIndexTypeCreate(self.ctx);

    // create constants for loop bounds
    const zero_op = self.ora_dialect.createArithConstant(0, index_ty, loc);
    h.appendOp(self.block, zero_op);
    const lower_bound = h.getResult(zero_op, 0);

    // determine upper bound: integer iterables are ranges; shaped collections use length().
    const iterable_ty = c.oraValueGetType(iterable);
    const span = body.span;
    const upper_bound = blk: {
        if (c.oraTypeIsAInteger(iterable_ty)) {
            const upper_raw = iterable;
            break :blk self.expr_lowerer.convertIndexToIndexType(upper_raw, span);
        } else if (c.oraTypeIsAMemRef(iterable_ty) or c.oraTypeIsAShaped(iterable_ty)) {
            const len_index = self.expr_lowerer.createLengthAccess(iterable, span);
            break :blk len_index;
        } else {
            break :blk self.expr_lowerer.convertIndexToIndexType(iterable, span);
        }
    };

    // create step constant
    const step_op = self.ora_dialect.createArithConstant(1, index_ty, loc);
    h.appendOp(self.block, step_op);
    const step = h.getResult(step_op, 0);

    // create scf.for operation
    const for_op = self.ora_dialect.createScfFor(lower_bound, upper_bound, step, &[_]c.MlirValue{}, &[_]c.MlirType{}, loc);
    h.appendOp(self.block, for_op);

    const op_body_block = c.oraScfForOpGetBodyBlock(for_op);
    if (c.oraBlockIsNull(op_body_block)) {
        @panic("scf.for missing body block");
    }

    // body-scoped expression lowerer to keep index/element IR in the loop region
    var body_expr_lowerer = ExpressionLowerer.init(
        self.ctx,
        op_body_block,
        self.type_mapper,
        self.expr_lowerer.param_map,
        self.expr_lowerer.storage_map,
        self.expr_lowerer.local_var_map,
        self.expr_lowerer.symbol_table,
        self.expr_lowerer.builtin_registry,
        self.expr_lowerer.error_handler,
        self.expr_lowerer.locations,
        self.ora_dialect,
    );
    body_expr_lowerer.current_function_return_type = self.current_function_return_type;
    body_expr_lowerer.current_function_return_type_info = self.current_function_return_type_info;
    body_expr_lowerer.in_try_block = self.in_try_block;

    // get the induction variable (index)
    const index_var = c.oraBlockGetArgument(op_body_block, 0);
    // for range-based loops, item is the same as index.
    // for collection-based loops, item is the element at iterable[index].
    if (self.local_var_map) |lvm| {
        lvm.addLocalVar(index_name, index_var) catch {
            log.warn("Failed to add index variable to map: {s}\n", .{index_name});
        };

        if (c.oraTypeIsAInteger(iterable_ty)) {
            // range-based: item is also the index
            lvm.addLocalVar(item_name, index_var) catch {
                log.warn("Failed to add item variable to map: {s}\n", .{item_name});
            };
        } else if (c.oraTypeIsAMemRef(iterable_ty) or c.oraTypeIsAShaped(iterable_ty)) {
            const elem_value = body_expr_lowerer.createArrayIndexLoad(iterable, index_var, span);
            lvm.addLocalVar(item_name, elem_value) catch {
                log.warn("Failed to add element variable to map: {s}\n", .{item_name});
            };
        } else {
            lvm.addLocalVar(item_name, index_var) catch {
                log.warn("Failed to add item variable to map: {s}\n", .{item_name});
            };
        }
    }

    // lower loop invariants if present
    for (invariants) |*invariant| {
        const invariant_value = self.expr_lowerer.lowerExpression(invariant);
        const inv_op = self.ora_dialect.createInvariant(invariant_value, loc);
        verification.addVerificationAttributesToOp(self, inv_op, "invariant", "loop_invariant");
        h.appendOp(op_body_block, inv_op);
    }

    // lower decreases clause if present
    if (decreases) |decreases_expr| {
        const decreases_value = self.expr_lowerer.lowerExpression(decreases_expr);
        const dec_op = self.ora_dialect.createDecreases(decreases_value, loc);
        verification.addVerificationAttributesToOp(self, dec_op, "decreases", "loop_termination_measure");
        h.appendOp(op_body_block, dec_op);
    }

    // lower increases clause if present
    if (increases) |increases_expr| {
        const increases_value = self.expr_lowerer.lowerExpression(increases_expr);
        const inc_op = self.ora_dialect.createIncreases(increases_value, loc);
        verification.addVerificationAttributesToOp(self, inc_op, "increases", "loop_progress_measure");
        h.appendOp(op_body_block, inc_op);
    }

    // lower the loop body with loop label context
    const loop_label = label orelse "";
    const loop_ctx = LabelContext{
        .label = loop_label,
        .label_type = .For,
        .parent = self.label_context,
    };
    var body_lowerer = self.*;
    body_lowerer.label_context = &loop_ctx;
    const ended_with_terminator = try body_lowerer.lowerBlockBody(body, op_body_block);

    // add scf.yield at end of body if no terminator
    if (!ended_with_terminator) {
        const yield_op = self.ora_dialect.createScfYield(loc);
        h.appendOp(op_body_block, yield_op);
    }
}

/// Lower destructured for loop (for (iterable) |.{field1, field2}| body)
fn lowerDestructuredForLoop(self: *const StatementLowerer, pattern: lib.ast.Expressions.DestructuringPattern, iterable: c.MlirValue, body: lib.ast.Statements.BlockNode, invariants: []lib.ast.Expressions.ExprNode, decreases: ?*lib.ast.Expressions.ExprNode, increases: ?*lib.ast.Expressions.ExprNode, label: ?[]const u8, loc: c.MlirLocation) LoweringError!void {
    // create integer type for loop bounds
    const zero_ty = c.oraIntegerTypeCreate(self.ctx, constants.DEFAULT_INTEGER_BITS);

    // create constants for loop bounds
    const zero_op = self.ora_dialect.createArithConstant(0, zero_ty, loc);
    h.appendOp(self.block, zero_op);
    const lower_bound = h.getResult(zero_op, 0);

    // use iterable as upper bound (simplified)
    const upper_bound = iterable;

    // create step constant
    const step_op = self.ora_dialect.createArithConstant(1, zero_ty, loc);
    h.appendOp(self.block, step_op);
    const step = h.getResult(step_op, 0);

    // create scf.for operation
    const for_op = self.ora_dialect.createScfFor(lower_bound, upper_bound, step, &[_]c.MlirValue{}, &[_]c.MlirType{}, loc);
    h.appendOp(self.block, for_op);

    const body_block = c.oraScfForOpGetBodyBlock(for_op);
    if (c.oraBlockIsNull(body_block)) {
        @panic("scf.for missing body block");
    }

    // get the item variable
    const item_var = c.oraBlockGetArgument(body_block, 0);

    // add destructured fields to local variable map
    if (self.local_var_map) |lvm| {
        switch (pattern) {
            .Struct => |struct_pattern| {
                for (struct_pattern, 0..) |field, i| {
                    // create field access for each destructured field
                    const result_ty = c.oraIntegerTypeCreate(self.ctx, constants.DEFAULT_INTEGER_BITS);
                    const indices = [_]u32{@intCast(i)};
                    const field_access_op = self.ora_dialect.createLlvmExtractvalue(item_var, &indices, result_ty, loc);
                    h.appendOp(body_block, field_access_op);
                    const field_value = h.getResult(field_access_op, 0);

                    // add to variable map
                    lvm.addLocalVar(field.variable, field_value) catch {
                        log.warn("Failed to add destructured field to map: {s}\n", .{field.variable});
                    };
                }
            },
            else => {
                log.warn("Unsupported destructuring pattern type\n", .{});
            },
        }
    }

    // lower loop invariants if present
    for (invariants) |*invariant| {
        const invariant_value = self.expr_lowerer.lowerExpression(invariant);
        const inv_op = self.ora_dialect.createInvariant(invariant_value, loc);
        verification.addVerificationAttributesToOp(self, inv_op, "invariant", "loop_invariant");
        h.appendOp(body_block, inv_op);
    }

    // lower decreases clause if present
    if (decreases) |decreases_expr| {
        const decreases_value = self.expr_lowerer.lowerExpression(decreases_expr);
        const dec_op = self.ora_dialect.createDecreases(decreases_value, loc);
        verification.addVerificationAttributesToOp(self, dec_op, "decreases", "loop_termination_measure");
        h.appendOp(body_block, dec_op);
    }

    // lower increases clause if present
    if (increases) |increases_expr| {
        const increases_value = self.expr_lowerer.lowerExpression(increases_expr);
        const inc_op = self.ora_dialect.createIncreases(increases_value, loc);
        verification.addVerificationAttributesToOp(self, inc_op, "increases", "loop_progress_measure");
        h.appendOp(body_block, inc_op);
    }

    // lower the loop body with loop label context
    const loop_label = label orelse "";
    const loop_ctx = LabelContext{
        .label = loop_label,
        .label_type = .For,
        .parent = self.label_context,
    };
    var body_lowerer = self.*;
    body_lowerer.label_context = &loop_ctx;
    const ended_with_terminator = try body_lowerer.lowerBlockBody(body, body_block);

    // add scf.yield at end of body if no terminator
    if (!ended_with_terminator) {
        const yield_op = self.ora_dialect.createScfYield(loc);
        h.appendOp(body_block, yield_op);
    }
}

/// Extract integer value from a literal expression (for switch case patterns)
fn extractIntegerFromLiteral(_: *const StatementLowerer, literal: *const lib.ast.Expressions.LiteralExpr) ?i64 {
    return switch (literal.*) {
        .Integer => |int| blk: {
            const parsed = std.fmt.parseInt(i64, int.value, 0) catch return null;
            break :blk parsed;
        },
        else => null,
    };
}

/// Extract integer value from an expression (for switch case patterns)
fn extractIntegerFromExpr(self: *const StatementLowerer, expr: *const lib.ast.Expressions.ExprNode) ?i64 {
    return switch (expr.*) {
        .Literal => |lit| extractIntegerFromLiteral(self, &lit),
        else => null,
    };
}

/// Lower switch statements using ora.switch operation
/// Switch statements never produce values - they are control flow only
pub fn lowerSwitch(self: *const StatementLowerer, switch_stmt: *const lib.ast.Statements.SwitchNode) LoweringError!void {
    const loc = self.fileLoc(switch_stmt.span);

    const condition_raw = self.expr_lowerer.lowerExpression(&switch_stmt.condition);
    const condition = helpers.ensureValue(self, condition_raw, loc);

    // check if all cases have returns - if so, use scf.if pattern (like lowerIfWithReturns)
    var all_cases_return = true;
    for (switch_stmt.cases) |case| {
        const case_has_return = switch (case.body) {
            .Block => |block| helpers.blockHasReturn(self, block),
            .LabeledBlock => |labeled| helpers.blockHasReturn(self, labeled.block),
            .Expression => false, // Expressions don't return
        };
        if (!case_has_return) {
            all_cases_return = false;
            break;
        }
    }
    if (switch_stmt.default_case) |default_block| {
        if (!helpers.blockHasReturn(self, default_block)) {
            all_cases_return = false;
        }
    } else {
        // no default case means not all paths return
        all_cases_return = false;
    }

    // if all cases return, use scf.if pattern (lowerSwitchCases) instead of ora.switch
    if (all_cases_return) {
        // use lowerSwitchCases which creates nested scf.if with scf.yield
        // this will create a chain of scf.if operations that yield values
        const result_value = try lowerSwitchCases(self, switch_stmt.cases, condition, 0, self.block, loc, switch_stmt.default_case);

        // after the switch, return the result from the scf.if chain
        if (result_value) |result| {
            const return_op = self.ora_dialect.createFuncReturnWithValue(result, loc);
            h.appendOp(self.block, return_op);
        }
        return;
    }

    const total_cases = switch_stmt.cases.len + if (switch_stmt.default_case != null) @as(usize, 1) else 0;
    const switch_op = c.oraSwitchOpCreateWithCases(
        self.ctx,
        loc,
        condition,
        null,
        0,
        total_cases,
    );
    if (c.oraOperationIsNull(switch_op)) {
        @panic("Failed to create ora.switch operation");
    }
    h.appendOp(self.block, switch_op);

    var case_values = std.ArrayList(i64){};
    defer case_values.deinit(self.allocator);
    var range_starts = std.ArrayList(i64){};
    defer range_starts.deinit(self.allocator);
    var range_ends = std.ArrayList(i64){};
    defer range_ends.deinit(self.allocator);
    var case_kinds = std.ArrayList(i64){};
    defer case_kinds.deinit(self.allocator);
    var default_case_index: i64 = -1;

    var case_idx: usize = 0;
    for (switch_stmt.cases) |case| {
        const case_block = c.oraSwitchOpGetCaseBlock(switch_op, case_idx);
        if (c.oraBlockIsNull(case_block)) {
            @panic("ora.switch missing case block");
        }

        var case_expr_lowerer = ExpressionLowerer.init(self.ctx, case_block, self.type_mapper, self.expr_lowerer.param_map, self.expr_lowerer.storage_map, self.expr_lowerer.local_var_map, self.expr_lowerer.symbol_table, self.expr_lowerer.builtin_registry, self.expr_lowerer.error_handler, self.expr_lowerer.locations, self.ora_dialect);
        case_expr_lowerer.current_function_return_type = self.current_function_return_type;
        case_expr_lowerer.current_function_return_type_info = self.current_function_return_type_info;
        case_expr_lowerer.in_try_block = self.in_try_block;
        const switch_ctx = LabelContext{
            .label = "",
            .label_type = .Switch,
            .parent = self.label_context,
        };

        switch (case.pattern) {
            .Literal => |lit| {
                _ = self.expr_lowerer.lowerLiteral(&lit.value);
                if (extractIntegerFromLiteral(self, &lit.value)) |val| {
                    case_values.append(self.allocator, val) catch {};
                    range_starts.append(self.allocator, 0) catch {};
                    range_ends.append(self.allocator, 0) catch {};
                    case_kinds.append(self.allocator, 0) catch {}; // 0 = literal
                } else {
                    case_values.append(self.allocator, 0) catch {};
                    range_starts.append(self.allocator, 0) catch {};
                    range_ends.append(self.allocator, 0) catch {};
                    case_kinds.append(self.allocator, 0) catch {};
                }
            },
            .Range => |range| {
                _ = self.expr_lowerer.lowerExpression(range.start);
                _ = self.expr_lowerer.lowerExpression(range.end);
                const start_val = extractIntegerFromExpr(self, range.start) orelse 0;
                const end_val = extractIntegerFromExpr(self, range.end) orelse 0;
                case_values.append(self.allocator, 0) catch {};
                range_starts.append(self.allocator, start_val) catch {};
                range_ends.append(self.allocator, end_val) catch {};
                case_kinds.append(self.allocator, 1) catch {}; // 1 = range
            },
            .EnumValue => {
                case_values.append(self.allocator, 0) catch {};
                range_starts.append(self.allocator, 0) catch {};
                range_ends.append(self.allocator, 0) catch {};
                case_kinds.append(self.allocator, 0) catch {}; // Treat enum as literal for now
            },
            .Else => {
                default_case_index = @intCast(case_idx);
                case_values.append(self.allocator, 0) catch {};
                range_starts.append(self.allocator, 0) catch {};
                range_ends.append(self.allocator, 0) catch {};
                case_kinds.append(self.allocator, 2) catch {}; // 2 = else
            },
        }

        switch (case.body) {
            .Expression => |expr| {
                // for switch statements, evaluate expression but don't yield its value
                _ = case_expr_lowerer.lowerExpression(expr);
                // switch statements terminate with empty yield (no value)
                const yield_op = self.ora_dialect.createYield(&[_]c.MlirValue{}, loc);
                h.appendOp(case_block, yield_op);
            },
            .Block => |block| {
                const has_return = helpers.blockHasReturn(self, block);
                if (has_return) {
                    stmt_loop: for (block.statements) |stmt| {
                        switch (stmt) {
                            .Return => |ret| {
                                // return statements in switch cases should use ora.return (terminator)
                                // no yield needed - ora.return is a proper terminator
                                if (ret.value) |e| {
                                    const v = case_expr_lowerer.lowerExpression(&e);
                                    const return_op = self.ora_dialect.createFuncReturnWithValue(v, loc);
                                    h.appendOp(case_block, return_op);
                                } else {
                                    const return_op = self.ora_dialect.createFuncReturn(loc);
                                    h.appendOp(case_block, return_op);
                                }
                                // break out of the for loop - return is a terminator, nothing should follow
                                break :stmt_loop;
                            },
                            else => {
                                var temp_lowerer = self.*;
                                temp_lowerer.block = case_block;
                                temp_lowerer.label_context = &switch_ctx;
                                try temp_lowerer.lowerStatement(&stmt);
                            },
                        }
                    }
                } else {
                    var body_lowerer = self.*;
                    body_lowerer.label_context = &switch_ctx;
                    const ended_with_terminator = try body_lowerer.lowerBlockBody(block, case_block);
                    if (!ended_with_terminator) {
                        // block doesn't have a terminator, add one
                        // switch statements use empty yield (no value)
                        const yield_op = self.ora_dialect.createYield(&[_]c.MlirValue{}, loc);
                        h.appendOp(case_block, yield_op);
                    }
                }
            },
            .LabeledBlock => |labeled| {
                const has_return = helpers.blockHasReturn(self, labeled.block);
                if (has_return) {
                    stmt_loop: for (labeled.block.statements) |stmt| {
                        switch (stmt) {
                            .Return => |ret| {
                                // return statements in switch cases should use ora.return (terminator)
                                // no yield needed - ora.return is a proper terminator
                                if (ret.value) |e| {
                                    const v = case_expr_lowerer.lowerExpression(&e);
                                    const return_op = self.ora_dialect.createFuncReturnWithValue(v, loc);
                                    h.appendOp(case_block, return_op);
                                } else {
                                    const return_op = self.ora_dialect.createFuncReturn(loc);
                                    h.appendOp(case_block, return_op);
                                }
                                // break out of the for loop - return is a terminator, nothing should follow
                                break :stmt_loop;
                            },
                            else => {
                                var temp_lowerer = self.*;
                                temp_lowerer.block = case_block;
                                temp_lowerer.label_context = &switch_ctx;
                                try temp_lowerer.lowerStatement(&stmt);
                            },
                        }
                    }
                } else {
                    var body_lowerer = self.*;
                    body_lowerer.label_context = &switch_ctx;
                    const ended_with_terminator = try body_lowerer.lowerBlockBody(labeled.block, case_block);
                    if (!ended_with_terminator) {
                        // block doesn't have a terminator, add one
                        // switch statements use empty yield (no value)
                        const yield_op = self.ora_dialect.createYield(&[_]c.MlirValue{}, loc);
                        h.appendOp(case_block, yield_op);
                    }
                }
            },
        }

        case_idx += 1;
    }

    if (switch_stmt.default_case) |default_block| {
        const default_block_mlir = c.oraSwitchOpGetCaseBlock(switch_op, case_idx);
        if (c.oraBlockIsNull(default_block_mlir)) {
            @panic("ora.switch missing default block");
        }

        const switch_ctx = LabelContext{
            .label = "",
            .label_type = .Switch,
            .parent = self.label_context,
        };
        const default_has_return = helpers.blockHasReturn(self, default_block);
        if (default_has_return) {
            var default_expr_lowerer = ExpressionLowerer.init(self.ctx, default_block_mlir, self.type_mapper, self.expr_lowerer.param_map, self.expr_lowerer.storage_map, self.expr_lowerer.local_var_map, self.expr_lowerer.symbol_table, self.expr_lowerer.builtin_registry, self.expr_lowerer.error_handler, self.expr_lowerer.locations, self.ora_dialect);
            default_expr_lowerer.current_function_return_type = self.current_function_return_type;
            default_expr_lowerer.current_function_return_type_info = self.current_function_return_type_info;
            default_expr_lowerer.in_try_block = self.in_try_block;
            stmt_loop: for (default_block.statements) |stmt| {
                switch (stmt) {
                    .Return => |ret| {
                        // return statements in switch default case should use ora.return (terminator)
                        // no yield needed - ora.return is a proper terminator
                        if (ret.value) |e| {
                            const v = default_expr_lowerer.lowerExpression(&e);
                            const return_op = self.ora_dialect.createFuncReturnWithValue(v, loc);
                            h.appendOp(default_block_mlir, return_op);
                        } else {
                            const return_op = self.ora_dialect.createFuncReturn(loc);
                            h.appendOp(default_block_mlir, return_op);
                        }
                        // break out of the for loop - return is a terminator, nothing should follow
                        break :stmt_loop;
                    },
                    else => {
                        var temp_lowerer = self.*;
                        temp_lowerer.block = default_block_mlir;
                        temp_lowerer.label_context = &switch_ctx;
                        try temp_lowerer.lowerStatement(&stmt);
                    },
                }
            }
        } else {
            var body_lowerer = self.*;
            body_lowerer.label_context = &switch_ctx;
            const ended_with_terminator = try body_lowerer.lowerBlockBody(default_block, default_block_mlir);
            if (!ended_with_terminator) {
                // block doesn't have a terminator, add one
                // switch statements use empty yield (no value)
                const yield_op = self.ora_dialect.createYield(&[_]c.MlirValue{}, loc);
                h.appendOp(default_block_mlir, yield_op);
            }
        }

        if (default_case_index < 0) {
            default_case_index = @intCast(case_idx);
        }
        case_values.append(self.allocator, 0) catch {};
        range_starts.append(self.allocator, 0) catch {};
        range_ends.append(self.allocator, 0) catch {};
        case_kinds.append(self.allocator, 2) catch {}; // 2 = else
        case_idx += 1;
    }

    if (case_values.items.len > 0) {
        c.oraSwitchOpSetCasePatterns(
            switch_op,
            case_values.items.ptr,
            range_starts.items.ptr,
            range_ends.items.ptr,
            case_kinds.items.ptr,
            default_case_index,
            case_values.items.len,
        );
    }

    // switch statements don't produce values - no return needed here
    // the switch just executes and control flow continues to the next statement
}

/// Recursively lower switch cases as nested if-else-if chain
/// For cases with returns, use scf.yield (like lowerIfWithReturns) instead of ora.return
/// However, if we're inside a labeled switch (scf.while), we can't use result types
pub fn lowerSwitchCases(self: *const StatementLowerer, cases: []const lib.ast.Expressions.SwitchCase, condition: c.MlirValue, case_idx: usize, target_block: c.MlirBlock, loc: c.MlirLocation, default_case: ?lib.ast.Statements.BlockNode) LoweringError!?c.MlirValue {
    log.debug("[lowerSwitchCases] case_idx={}, total_cases={}, has_default={}\n", .{ case_idx, cases.len, default_case != null });

    // determine if we need result type (if any case or default has return)
    // but NOT if we're inside a labeled switch (scf.while) - those can't yield values
    const is_labeled_switch = if (self.label_context) |ctx| ctx.label_type == .Switch else false;
    const result_type = if (!is_labeled_switch) blk: {
        break :blk if (self.current_function_return_type) |ret_type| ret_type else null;
    } else null;

    log.debug("[lowerSwitchCases] is_labeled_switch={}, result_type={any}\n", .{ is_labeled_switch, result_type != null });

    if (case_idx >= cases.len) {
        log.debug("[lowerSwitchCases] Reached end of cases, handling default\n", .{});
        if (default_case) |default_block| {
            const has_return = helpers.blockHasReturn(self, default_block);
            log.debug("[lowerSwitchCases] Default case has_return={}, is_labeled_switch={}\n", .{ has_return, is_labeled_switch });
            if (has_return and is_labeled_switch) {
                // for labeled switches, store return value/flag and then scf.yield
                log.debug("[lowerSwitchCases] Labeled switch default with return - storing return value and flag\n", .{});
                var temp_lowerer = self.*;
                temp_lowerer.block = target_block;
                var expr_lowerer = ExpressionLowerer.init(self.ctx, target_block, self.type_mapper, self.expr_lowerer.param_map, self.expr_lowerer.storage_map, self.expr_lowerer.local_var_map, self.expr_lowerer.symbol_table, self.expr_lowerer.builtin_registry, self.expr_lowerer.error_handler, self.expr_lowerer.locations, self.ora_dialect);
                expr_lowerer.current_function_return_type = self.current_function_return_type;
                expr_lowerer.current_function_return_type_info = self.current_function_return_type_info;
                expr_lowerer.in_try_block = self.in_try_block;
                var has_terminator = false;
                for (default_block.statements) |stmt| {
                    if (has_terminator) break;
                    switch (stmt) {
                        .Return => |ret| {
                            if (self.label_context) |label_ctx| {
                                if (label_ctx.return_flag_memref) |return_flag_memref| {
                                    if (label_ctx.return_value_memref) |return_value_memref| {
                                        const ret_loc = temp_lowerer.fileLoc(ret.span);
                                        if (ret.value) |value_expr| {
                                            const return_value = expr_lowerer.lowerExpression(&value_expr);
                                            const value_to_store = helpers.ensureValue(&temp_lowerer, return_value, ret_loc);
                                            const memref_type = c.oraValueGetType(return_value_memref);
                                            const element_type = c.oraShapedTypeGetElementType(memref_type);
                                            const final_value = helpers.convertValueToType(&temp_lowerer, value_to_store, element_type, ret.span, ret_loc);
                                            helpers.storeToMemref(&temp_lowerer, final_value, return_value_memref, ret_loc);
                                        } else if (self.current_function_return_type) |ret_type| {
                                            const default_val = try helpers.createDefaultValueForType(&temp_lowerer, ret_type, ret_loc);
                                            helpers.storeToMemref(&temp_lowerer, default_val, return_value_memref, ret_loc);
                                        }
                                        const true_val = helpers.createBoolConstant(&temp_lowerer, true, ret_loc);
                                        helpers.storeToMemref(&temp_lowerer, true_val, return_flag_memref, ret_loc);
                                    }
                                }
                            }
                            const ret_loc = temp_lowerer.fileLoc(ret.span);
                            const yield_op = self.ora_dialect.createScfYield(ret_loc);
                            h.appendOp(target_block, yield_op);
                            has_terminator = true;
                        },
                        else => {
                            try temp_lowerer.lowerStatement(&stmt);
                            const is_terminator = switch (stmt) {
                                .Break, .Continue, .Return => true,
                                else => false,
                            };
                            if (is_terminator) has_terminator = true;
                        },
                    }
                }
                if (!has_terminator) {
                    const yield_op = self.ora_dialect.createScfYield(loc);
                    h.appendOp(target_block, yield_op);
                }
            } else if (has_return) {
                log.debug("[lowerSwitchCases] Non-labeled switch default with return - using lowerBlockBodyWithYield\n", .{});
                // for non-labeled switches with returns, use lowerBlockBodyWithYield (converts to scf.yield)
                try lowerBlockBodyWithYield(self, default_block, target_block);

                // ensure block has a terminator (lowerBlockBodyWithYield already adds scf.yield for returns)
                const has_yield = helpers.blockEndsWithYield(self, target_block);
                if (!has_yield) {
                    // only add yield if lowerBlockBodyWithYield didn't add one (no return statement)
                    if (result_type) |ret_type| {
                        const default_val = try helpers.createDefaultValueForType(self, ret_type, loc);
                        const yield_op = self.ora_dialect.createScfYieldWithValues(&[_]c.MlirValue{default_val}, loc);
                        h.appendOp(target_block, yield_op);
                    } else {
                        const yield_op = self.ora_dialect.createScfYield(loc);
                        h.appendOp(target_block, yield_op);
                    }
                }
                if (!has_yield) {
                    if (is_labeled_switch) {
                        const yield_op = self.ora_dialect.createYield(&[_]c.MlirValue{}, loc);
                        h.appendOp(target_block, yield_op);
                    } else {
                        if (result_type) |ret_type| {
                            const default_val = try helpers.createDefaultValueForType(self, ret_type, loc);
                            const yield_op = self.ora_dialect.createScfYieldWithValues(&[_]c.MlirValue{default_val}, loc);
                            h.appendOp(target_block, yield_op);
                        } else {
                            const yield_op = self.ora_dialect.createScfYield(loc);
                            h.appendOp(target_block, yield_op);
                        }
                    }
                }
            } else {
                // no return - use lowerBlockBody which handles break/continue properly
                const ended_with_terminator = try self.lowerBlockBody(default_block, target_block);
                if (!ended_with_terminator) {
                    // block doesn't have a terminator, add scf.yield for scf.if block
                    const yield_op = self.ora_dialect.createScfYield(loc);
                    h.appendOp(target_block, yield_op);
                }
            }
        } else {
            // no default case - add scf.yield
            const yield_op = self.ora_dialect.createScfYield(loc);
            h.appendOp(target_block, yield_op);
        }
        // for default case handling, return null (no result from this path)
        return null;
    }

    const case = cases[case_idx];
    log.debug("[lowerSwitchCases] Processing case {}\n", .{case_idx});

    const case_condition = switch (case.pattern) {
        .Literal => |lit| blk: {
            // create case value constant in target_block (where it will be used)
            var case_expr_lowerer = ExpressionLowerer.init(self.ctx, target_block, self.type_mapper, self.expr_lowerer.param_map, self.expr_lowerer.storage_map, self.expr_lowerer.local_var_map, self.expr_lowerer.symbol_table, self.expr_lowerer.builtin_registry, self.expr_lowerer.error_handler, self.expr_lowerer.locations, self.ora_dialect);
            case_expr_lowerer.current_function_return_type = self.current_function_return_type;
            case_expr_lowerer.current_function_return_type_info = self.current_function_return_type_info;
            case_expr_lowerer.in_try_block = self.in_try_block;
            const case_value = case_expr_lowerer.lowerLiteral(&lit.value);
            const cmp_op = c.oraCmpOpCreate(
                self.ctx,
                loc,
                h.strRef("eq"),
                condition,
                case_value,
                h.boolType(self.ctx),
            );
            h.appendOp(target_block, cmp_op);
            break :blk h.getResult(cmp_op, 0);
        },
        .Range => |range| blk: {
            // create range values in target_block (where they will be used)
            var case_expr_lowerer = ExpressionLowerer.init(self.ctx, target_block, self.type_mapper, self.expr_lowerer.param_map, self.expr_lowerer.storage_map, self.expr_lowerer.local_var_map, self.expr_lowerer.symbol_table, self.expr_lowerer.builtin_registry, self.expr_lowerer.error_handler, self.expr_lowerer.locations, self.ora_dialect);
            case_expr_lowerer.current_function_return_type = self.current_function_return_type;
            case_expr_lowerer.current_function_return_type_info = self.current_function_return_type_info;
            case_expr_lowerer.in_try_block = self.in_try_block;
            const start_val = case_expr_lowerer.lowerExpression(range.start);
            const end_val = case_expr_lowerer.lowerExpression(range.end);

            const lower_cmp_op = c.oraCmpOpCreate(
                self.ctx,
                loc,
                h.strRef("uge"),
                condition,
                start_val,
                h.boolType(self.ctx),
            );
            h.appendOp(target_block, lower_cmp_op);
            const lower_bound = h.getResult(lower_cmp_op, 0);

            const upper_cmp_op = c.oraCmpOpCreate(
                self.ctx,
                loc,
                h.strRef("ule"),
                condition,
                end_val,
                h.boolType(self.ctx),
            );
            h.appendOp(target_block, upper_cmp_op);
            const upper_bound = h.getResult(upper_cmp_op, 0);

            const and_op = c.oraArithAndIOpCreate(self.ctx, loc, lower_bound, upper_bound);
            h.appendOp(target_block, and_op);
            break :blk h.getResult(and_op, 0);
        },
        .EnumValue => |enum_val| blk: {
            if (self.symbol_table) |st| {
                if (st.lookupType(enum_val.enum_name)) |enum_type| {
                    if (enum_type.getVariantIndex(enum_val.variant_name)) |variant_idx| {
                        // use the enum's underlying type (stored in mlir_type) instead of hardcoded i256
                        const enum_underlying_type = enum_type.mlir_type;
                        const const_op = self.ora_dialect.createArithConstant(@intCast(variant_idx), enum_underlying_type, loc);
                        h.appendOp(self.block, const_op);
                        const variant_const = h.getResult(const_op, 0);

                        const cmp_op = c.oraArithCmpIOpCreate(self.ctx, loc, 0, condition, variant_const);
                        h.appendOp(target_block, cmp_op);
                        break :blk h.getResult(cmp_op, 0);
                    }
                }
            }
            const const_op = self.ora_dialect.createArithConstant(0, h.boolType(self.ctx), loc);
            h.appendOp(self.block, const_op);
            break :blk h.getResult(const_op, 0);
        },
        .Else => blk: {
            // else case always matches
            const const_op = self.ora_dialect.createArithConstant(1, h.boolType(self.ctx), loc);
            h.appendOp(self.block, const_op);
            break :blk h.getResult(const_op, 0);
        },
    };

    // for labeled switches with returns, use scf.if without result types and ora.return inside
    // for non-labeled switches with returns, use scf.if with result types and scf.yield
    // note: Regular if statements with returns also use scf.if, not ora.if
    const if_op = if (is_labeled_switch) blk: {
        log.debug("[lowerSwitchCases] Creating scf.if for labeled switch (no result type, ora.return inside)\n", .{});
        // use scf.if without result type - allows ora.return inside regions
        const result_types = [_]c.MlirType{};
        const op = self.ora_dialect.createScfIf(case_condition, result_types[0..], loc);
        break :blk op;
    } else blk: {
        log.debug("[lowerSwitchCases] Creating scf.if for non-labeled switch\n", .{});
        // use scf.if - allows scf.yield with values
        var result_types: [1]c.MlirType = undefined;
        var result_slice: []c.MlirType = &[_]c.MlirType{};
        if (result_type) |ret_type| {
            result_types[0] = ret_type;
            result_slice = result_types[0..1];
        }
        const op = self.ora_dialect.createScfIf(case_condition, result_slice, loc);
        break :blk op;
    };

    // get regions (works for both ora.if and scf.if)
    // for labeled switches, we already created the regions above
    // for non-labeled switches, we need to get them from the operation
    const then_block = c.oraScfIfOpGetThenBlock(if_op);
    const else_block = c.oraScfIfOpGetElseBlock(if_op);
    if (c.oraBlockIsNull(then_block) or c.oraBlockIsNull(else_block)) {
        @panic("scf.if missing then/else blocks");
    }

    log.debug("[lowerSwitchCases] if operation created, appending to self.block\n", .{});
    h.appendOp(target_block, if_op);
    log.debug("[lowerSwitchCases] if operation appended\n", .{});

    // get result from scf.if if it has one (for non-labeled switches with returns)
    const result_value = if (!is_labeled_switch and result_type != null) blk: {
        const result = h.getResult(if_op, 0);
        break :blk result;
    } else null;

    // lower case body in then block
    // use lowerBlockBodyWithYield which converts returns to scf.yield (like lowerIfWithReturns)
    log.debug("[lowerSwitchCases] Lowering case body\n", .{});
    switch (case.body) {
        .Expression => |expr| {
            log.debug("[lowerSwitchCases] Case body is Expression\n", .{});
            var case_expr_lowerer = ExpressionLowerer.init(self.ctx, then_block, self.type_mapper, self.expr_lowerer.param_map, self.expr_lowerer.storage_map, self.expr_lowerer.local_var_map, self.expr_lowerer.symbol_table, self.expr_lowerer.builtin_registry, self.expr_lowerer.error_handler, self.expr_lowerer.locations, self.ora_dialect);
            case_expr_lowerer.current_function_return_type = self.current_function_return_type;
            case_expr_lowerer.current_function_return_type_info = self.current_function_return_type_info;
            case_expr_lowerer.in_try_block = self.in_try_block;
            _ = case_expr_lowerer.lowerExpression(expr);
            // add appropriate yield - scf.if always uses scf.yield (even for labeled switches)
            if (result_type) |ret_type| {
                const default_val = try helpers.createDefaultValueForType(self, ret_type, loc);
                const yield_op = self.ora_dialect.createScfYieldWithValues(&[_]c.MlirValue{default_val}, loc);
                h.appendOp(then_block, yield_op);
            } else {
                // no result type - use empty scf.yield
                const yield_op = self.ora_dialect.createScfYield(loc);
                h.appendOp(then_block, yield_op);
            }
        },
        .Block => |block| {
            const has_return = helpers.blockHasReturn(self, block);
            log.debug("[lowerSwitchCases] Case body is Block, has_return={}, is_labeled_switch={}\n", .{ has_return, is_labeled_switch });
            if (has_return and is_labeled_switch) {
                // for labeled switches, store return value and set return flag, then use scf.yield with value
                log.debug("[lowerSwitchCases] Labeled switch with return - storing return value and flag\n", .{});
                var temp_lowerer = self.*;
                temp_lowerer.block = then_block;
                var expr_lowerer = ExpressionLowerer.init(self.ctx, then_block, self.type_mapper, self.expr_lowerer.param_map, self.expr_lowerer.storage_map, self.expr_lowerer.local_var_map, self.expr_lowerer.symbol_table, self.expr_lowerer.builtin_registry, self.expr_lowerer.error_handler, self.expr_lowerer.locations, self.ora_dialect);
                expr_lowerer.current_function_return_type = self.current_function_return_type;
                expr_lowerer.current_function_return_type_info = self.current_function_return_type_info;
                expr_lowerer.in_try_block = self.in_try_block;
                var has_terminator = false;
                for (block.statements) |stmt| {
                    if (has_terminator) break;
                    switch (stmt) {
                        .Return => |ret| {
                            // store return value and set return flag
                            if (self.label_context) |label_ctx| {
                                if (label_ctx.return_flag_memref) |return_flag_memref| {
                                    if (label_ctx.return_value_memref) |return_value_memref| {
                                        const ret_loc = temp_lowerer.fileLoc(ret.span);
                                        if (ret.value) |value_expr| {
                                            const return_value = expr_lowerer.lowerExpression(&value_expr);
                                            const value_to_store = helpers.ensureValue(&temp_lowerer, return_value, ret_loc);
                                            // get target type from memref element type
                                            const memref_type = c.oraValueGetType(return_value_memref);
                                            const element_type = c.oraShapedTypeGetElementType(memref_type);
                                            // convert value to match memref element type
                                            const final_value = helpers.convertValueToType(&temp_lowerer, value_to_store, element_type, ret.span, ret_loc);
                                            // store return value
                                            helpers.storeToMemref(&temp_lowerer, final_value, return_value_memref, ret_loc);
                                        } else {
                                            // no return value - use default value for type
                                            if (self.current_function_return_type) |ret_type| {
                                                const default_val = try helpers.createDefaultValueForType(&temp_lowerer, ret_type, ret_loc);
                                                helpers.storeToMemref(&temp_lowerer, default_val, return_value_memref, ret_loc);
                                            }
                                        }
                                        // set return flag to true
                                        const true_val = helpers.createBoolConstant(&temp_lowerer, true, ret_loc);
                                        helpers.storeToMemref(&temp_lowerer, true_val, return_flag_memref, ret_loc);
                                    }
                                }
                            }
                            // use scf.yield to exit scf.if region (return handled after scf.while)
                            const ret_loc = temp_lowerer.fileLoc(ret.span);
                            const yield_op = self.ora_dialect.createScfYield(ret_loc);
                            h.appendOp(then_block, yield_op);
                            has_terminator = true;
                        },
                        else => {
                            try temp_lowerer.lowerStatement(&stmt);
                            const is_terminator = switch (stmt) {
                                .Break, .Continue, .Return => true,
                                else => false,
                            };
                            if (is_terminator) has_terminator = true;
                        },
                    }
                }
                // ensure block has a terminator
                if (!has_terminator) {
                    const yield_op = self.ora_dialect.createScfYield(loc);
                    h.appendOp(then_block, yield_op);
                }
            } else if (has_return) {
                log.debug("[lowerSwitchCases] Non-labeled switch with return - using lowerBlockBodyWithYield\n", .{});
                // for non-labeled switches with returns, use lowerBlockBodyWithYield (converts to scf.yield)
                try lowerBlockBodyWithYield(self, block, then_block);

                // ensure block has a terminator (lowerBlockBodyWithYield already adds scf.yield for returns)
                const has_yield = helpers.blockEndsWithYield(self, then_block);
                if (!has_yield) {
                    // only add yield if lowerBlockBodyWithYield didn't add one (no return statement)
                    if (result_type) |ret_type| {
                        const default_val = try helpers.createDefaultValueForType(self, ret_type, loc);
                        const yield_op = self.ora_dialect.createScfYieldWithValues(&[_]c.MlirValue{default_val}, loc);
                        h.appendOp(then_block, yield_op);
                    } else {
                        const yield_op = self.ora_dialect.createScfYield(loc);
                        h.appendOp(then_block, yield_op);
                    }
                }
            } else {
                // no return - use lowerBlockBody and ensure the scf.if region terminates
                const ended_with_terminator = try self.lowerBlockBody(block, then_block);
                if (!ended_with_terminator) {
                    const yield_op = self.ora_dialect.createScfYield(loc);
                    h.appendOp(then_block, yield_op);
                }
            }
        },
        .LabeledBlock => |labeled| {
            const has_return = helpers.blockHasReturn(self, labeled.block);
            log.debug("[lowerSwitchCases] Case body is LabeledBlock, has_return={}, is_labeled_switch={}\n", .{ has_return, is_labeled_switch });
            if (has_return and is_labeled_switch) {
                // for labeled switches, store return value and set return flag, then use scf.yield
                log.debug("[lowerSwitchCases] Labeled switch with return - storing return value and flag\n", .{});
                var temp_lowerer = self.*;
                temp_lowerer.block = then_block;
                var expr_lowerer = ExpressionLowerer.init(self.ctx, then_block, self.type_mapper, self.expr_lowerer.param_map, self.expr_lowerer.storage_map, self.expr_lowerer.local_var_map, self.expr_lowerer.symbol_table, self.expr_lowerer.builtin_registry, self.expr_lowerer.error_handler, self.expr_lowerer.locations, self.ora_dialect);
                expr_lowerer.current_function_return_type = self.current_function_return_type;
                expr_lowerer.current_function_return_type_info = self.current_function_return_type_info;
                expr_lowerer.in_try_block = self.in_try_block;
                var has_terminator = false;
                for (labeled.block.statements) |stmt| {
                    if (has_terminator) break;
                    switch (stmt) {
                        .Return => |ret| {
                            // store return value and set return flag
                            if (self.label_context) |label_ctx| {
                                if (label_ctx.return_flag_memref) |return_flag_memref| {
                                    if (label_ctx.return_value_memref) |return_value_memref| {
                                        const ret_loc = temp_lowerer.fileLoc(ret.span);
                                        if (ret.value) |value_expr| {
                                            const return_value = expr_lowerer.lowerExpression(&value_expr);
                                            const value_to_store = helpers.ensureValue(&temp_lowerer, return_value, ret_loc);
                                            // get target type from memref element type
                                            const memref_type = c.oraValueGetType(return_value_memref);
                                            const element_type = c.oraShapedTypeGetElementType(memref_type);
                                            // convert value to match memref element type
                                            const final_value = helpers.convertValueToType(&temp_lowerer, value_to_store, element_type, ret.span, ret_loc);
                                            // store return value
                                            helpers.storeToMemref(&temp_lowerer, final_value, return_value_memref, ret_loc);
                                        } else {
                                            // no return value - use default value for type
                                            if (self.current_function_return_type) |ret_type| {
                                                const default_val = try helpers.createDefaultValueForType(&temp_lowerer, ret_type, ret_loc);
                                                helpers.storeToMemref(&temp_lowerer, default_val, return_value_memref, ret_loc);
                                            }
                                        }
                                        // set return flag to true
                                        const true_val = helpers.createBoolConstant(&temp_lowerer, true, ret_loc);
                                        helpers.storeToMemref(&temp_lowerer, true_val, return_flag_memref, ret_loc);
                                    }
                                }
                            }
                            // use scf.yield to exit scf.if region (return handled after scf.while)
                            const ret_loc = temp_lowerer.fileLoc(ret.span);
                            const yield_op = self.ora_dialect.createScfYield(ret_loc);
                            h.appendOp(then_block, yield_op);
                            has_terminator = true;
                        },
                        else => {
                            try temp_lowerer.lowerStatement(&stmt);
                            const is_terminator = switch (stmt) {
                                .Break, .Continue, .Return => true,
                                else => false,
                            };
                            if (is_terminator) has_terminator = true;
                        },
                    }
                }
                // ensure block has a terminator
                if (!has_terminator) {
                    const yield_op = self.ora_dialect.createScfYield(loc);
                    h.appendOp(then_block, yield_op);
                }
            } else if (has_return) {
                log.debug("[lowerSwitchCases] Non-labeled switch with return - using lowerBlockBodyWithYield\n", .{});
                // for non-labeled switches with returns, use lowerBlockBodyWithYield (converts to scf.yield)
                try lowerBlockBodyWithYield(self, labeled.block, then_block);

                // if no return and we have result_type, add default yield
                if (result_type != null) {
                    if (result_type) |ret_type| {
                        const default_val = try helpers.createDefaultValueForType(self, ret_type, loc);
                        const yield_op = self.ora_dialect.createScfYieldWithValues(&[_]c.MlirValue{default_val}, loc);
                        h.appendOp(then_block, yield_op);
                    }
                }

                // ensure block has a terminator
                const has_yield = helpers.blockEndsWithYield(self, then_block);
                if (!has_yield) {
                    if (result_type) |ret_type| {
                        const default_val = try helpers.createDefaultValueForType(self, ret_type, loc);
                        const yield_op = self.ora_dialect.createScfYieldWithValues(&[_]c.MlirValue{default_val}, loc);
                        h.appendOp(then_block, yield_op);
                    } else {
                        const yield_op = self.ora_dialect.createScfYield(loc);
                        h.appendOp(then_block, yield_op);
                    }
                }
            } else {
                // no return - use lowerBlockBody and ensure the scf.if region terminates
                const ended_with_terminator = try self.lowerBlockBody(labeled.block, then_block);
                if (!ended_with_terminator) {
                    const yield_op = self.ora_dialect.createScfYield(loc);
                    h.appendOp(then_block, yield_op);
                }
            }
        },
    }

    // recursively lower remaining cases in else block
    // the recursive call will handle adding terminators to else_block (including default case)
    log.debug("[lowerSwitchCases] Recursively lowering remaining cases in else_block\n", .{});
    const recursive_result = try lowerSwitchCases(self, cases, condition, case_idx + 1, else_block, loc, default_case);

    // if the recursive call returned a result (nested scf.if), we need to yield it
    // this is the same pattern as nested if statements - yield the nested scf.if result
    if (!is_labeled_switch and result_type != null) {
        if (recursive_result) |nested_result| {
            // ensure nested result matches the expected result type
            if (result_type) |ret_type| {
                const nested_type = c.oraValueGetType(nested_result);
                const final_result = if (!c.oraTypeEqual(nested_type, ret_type))
                    helpers.convertValueToType(self, nested_result, ret_type, cases[0].span, loc)
                else
                    nested_result;
                const yield_op = self.ora_dialect.createScfYieldWithValues(&[_]c.MlirValue{final_result}, loc);
                h.appendOp(else_block, yield_op);
            } else {
                // yield the result from the nested scf.if chain
                const yield_op = self.ora_dialect.createScfYieldWithValues(&[_]c.MlirValue{nested_result}, loc);
                h.appendOp(else_block, yield_op);
            }
        } else {
            // no result from recursive call - ensure else_block has a terminator
            const has_yield = helpers.blockEndsWithYield(self, else_block);
            if (!has_yield) {
                if (result_type) |ret_type| {
                    const default_val = try helpers.createDefaultValueForType(self, ret_type, loc);
                    const yield_op = self.ora_dialect.createScfYieldWithValues(&[_]c.MlirValue{default_val}, loc);
                    h.appendOp(else_block, yield_op);
                } else {
                    const yield_op = self.ora_dialect.createScfYield(loc);
                    h.appendOp(else_block, yield_op);
                }
            }
        }
    } else {
        // for labeled switches or switches without result types, just ensure terminator
        const has_yield = helpers.blockEndsWithYield(self, else_block);
        if (!has_yield) {
            const yield_op = self.ora_dialect.createScfYield(loc);
            h.appendOp(else_block, yield_op);
        }
    }

    // return the result value from the outermost scf.if (for non-labeled switches with returns)
    // for labeled switches, return null (returns are handled inside scf.if regions)
    if (!is_labeled_switch and result_type != null) {
        return result_value;
    }
    return null;
}
