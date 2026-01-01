// ============================================================================
// Labeled Blocks and Control Flow Lowering
// ============================================================================

const std = @import("std");
const c = @import("mlir_c_api").c;
const lib = @import("ora_lib");
const StatementLowerer = @import("statement_lowerer.zig").StatementLowerer;
const LabelContext = @import("statement_lowerer.zig").LabelContext;
const LoweringError = StatementLowerer.LoweringError;
const helpers = @import("helpers.zig");
const control_flow = @import("control_flow.zig");
const log = @import("log");
const h = @import("../helpers.zig");

/// Lower break statements with label support
pub fn lowerBreak(self: *const StatementLowerer, break_stmt: *const lib.ast.Statements.BreakNode) LoweringError!void {
    const loc = self.fileLoc(break_stmt.span);

    // collect break value if present
    var operands = std.ArrayList(c.MlirValue){};
    defer operands.deinit(self.allocator);

    if (break_stmt.value) |value_expr| {
        const value = self.expr_lowerer.lowerExpression(value_expr);
        operands.append(self.allocator, value) catch unreachable;
    }

    // labeled breaks are handled via label context
    if (break_stmt.label) |label| {
        var matched_label = false;
        var ctx_opt = self.label_context;
        while (ctx_opt) |ctx| : (ctx_opt = ctx.parent) {
            if (!std.mem.eql(u8, label, ctx.label)) continue;
            matched_label = true;
            switch (ctx.label_type) {
                .Block => {
                    if (ctx.break_flag_memref) |break_flag_memref| {
                        const true_val = helpers.createBoolConstant(self, true, loc);
                        helpers.storeToMemref(self, true_val, break_flag_memref, loc);
                        return;
                    }
                },
                .Switch => {
                    if (ctx.continue_flag_memref) |continue_flag_memref| {
                        const false_val = helpers.createBoolConstant(self, false, loc);
                        helpers.storeToMemref(self, false_val, continue_flag_memref, loc);
                        const yield_op = self.ora_dialect.createScfYield(loc);
                        h.appendOp(self.block, yield_op);
                        return;
                    }
                },
                .While, .For => {
                    const break_op = self.ora_dialect.createBreak(label, operands.items, loc);
                    h.appendOp(self.block, break_op);
                    return;
                },
            }
        }
        if (!matched_label) {
            return LoweringError.InvalidControlFlow;
        }
    }

    // unlabeled break: prefer switch (breaks switch), else nearest loop
    var ctx_opt = self.label_context;
    while (ctx_opt) |ctx| : (ctx_opt = ctx.parent) {
        switch (ctx.label_type) {
            .Switch => {
                const yield_op = if (ctx.continue_flag_memref != null)
                    self.ora_dialect.createScfYield(loc)
                else
                    self.ora_dialect.createYield(&[_]c.MlirValue{}, loc);
                h.appendOp(self.block, yield_op);
                return;
            },
            .While, .For => {
                const break_op = self.ora_dialect.createBreak(null, operands.items, loc);
                h.appendOp(self.block, break_op);
                return;
            },
            .Block => {},
        }
    }

    // No label-handling path matched, report invalid control flow without ora.break.
    return LoweringError.InvalidControlFlow;
}

/// Lower continue statements with label support
pub fn lowerContinue(self: *const StatementLowerer, continue_stmt: *const lib.ast.Statements.ContinueNode) LoweringError!void {
    const loc = self.fileLoc(continue_stmt.span);

    if (continue_stmt.label) |label| {
        var matched_label = false;
        var ctx_opt = self.label_context;
        while (ctx_opt) |ctx| : (ctx_opt = ctx.parent) {
            if (!std.mem.eql(u8, label, ctx.label)) continue;
            matched_label = true;
            switch (ctx.label_type) {
                .Switch => {
                    if (ctx.continue_flag_memref != null) {
                        try handleLabeledSwitchContinue(self, continue_stmt, ctx, loc);
                        return;
                    }
                    return LoweringError.InvalidControlFlow;
                },
                .While, .For => {
                    const cont_op = self.ora_dialect.createContinue(label, loc);
                    h.appendOp(self.block, cont_op);
                    return;
                },
                .Block => {
                    if (ctx.break_flag_memref) |break_flag_memref| {
                        const true_val = helpers.createBoolConstant(self, true, loc);
                        helpers.storeToMemref(self, true_val, break_flag_memref, loc);
                        return;
                    }
                },
            }
        }
        if (!matched_label) {
            return LoweringError.InvalidControlFlow;
        }
    }

    var ctx_opt = self.label_context;
    while (ctx_opt) |ctx| : (ctx_opt = ctx.parent) {
        if (ctx.label_type == .While or ctx.label_type == .For) {
            const cont_op = self.ora_dialect.createContinue(null, loc);
            h.appendOp(self.block, cont_op);
            return;
        }
    }

    return LoweringError.InvalidControlFlow;
}

/// Handle continue to a labeled switch (stores value and sets continue flag)
fn handleLabeledSwitchContinue(
    self: *const StatementLowerer,
    continue_stmt: *const lib.ast.Statements.ContinueNode,
    label_ctx: *const LabelContext,
    loc: c.MlirLocation,
) LoweringError!void {
    // this only works for labeled switches with continue flag memref
    if (label_ctx.continue_flag_memref == null or label_ctx.value_memref == null) {
        if (self.expr_lowerer.error_handler) |handler| {
            handler.reportError(
                .InternalError,
                continue_stmt.span,
                "labeled switch continue without state",
                "internal error: continue state missing for labeled switch",
            ) catch {};
        }
        return LoweringError.InvalidControlFlow;
    }

    // unwrap the memrefs (we know they're not null from the check above)
    const value_memref = label_ctx.value_memref.?;
    const continue_flag = label_ctx.continue_flag_memref.?;

    // store new value if provided
    if (continue_stmt.value) |value_expr| {
        const value = self.expr_lowerer.lowerExpression(value_expr);
        const value_to_store = helpers.ensureValue(self, value, loc);

        // get target type from memref element type
        const memref_type = c.oraValueGetType(value_memref);
        const element_type = c.oraShapedTypeGetElementType(memref_type);

        // convert value to match memref element type
        const final_value = helpers.convertValueToType(self, value_to_store, element_type, continue_stmt.span, loc);

        // store converted value
        helpers.storeToMemref(self, final_value, value_memref, loc);
    }

    // set continue_flag to true
    const true_val = helpers.createBoolConstant(self, true, loc);
    helpers.storeToMemref(self, true_val, continue_flag, loc);

    // End the current scf.if region so the while after-region can continue.
    const yield_op = self.ora_dialect.createScfYield(loc);
    h.appendOp(self.block, yield_op);
}

/// Lower labeled blocks (including labeled switch, while, for)
pub fn lowerLabeledBlock(self: *const StatementLowerer, labeled_block: *const lib.ast.Statements.LabeledBlockNode) LoweringError!void {
    // check if this is a labeled switch (special handling for continue with value replacement)
    if (labeled_block.block.statements.len > 0) {
        const first_stmt = labeled_block.block.statements[0];
        if (first_stmt == .Switch) {
            try lowerLabeledSwitch(self, labeled_block);
            return;
        }
    }

    const loc = self.fileLoc(labeled_block.span);

    // regular labeled block - use break flag to gate statement execution
    const i1_type = h.boolType(self.ctx);
    const empty_attr = c.oraNullAttrCreate();
    const break_flag_memref_type = h.memRefType(self.ctx, i1_type, 0, null, empty_attr, empty_attr);
    const break_flag_alloca = self.ora_dialect.createMemrefAlloca(break_flag_memref_type, loc);
    h.appendOp(self.block, break_flag_alloca);
    const break_flag_memref = h.getResult(break_flag_alloca, 0);

    const false_val = helpers.createBoolConstant(self, false, loc);
    helpers.storeToMemref(self, false_val, break_flag_memref, loc);

    const label_ctx = LabelContext{
        .label = labeled_block.label,
        .break_flag_memref = break_flag_memref,
        .label_type = .Block,
        .parent = self.label_context,
    };

    for (labeled_block.block.statements) |stmt| {
        const load_break = self.ora_dialect.createMemrefLoad(break_flag_memref, &[_]c.MlirValue{}, i1_type, loc);
        h.appendOp(self.block, load_break);
        const break_flag_val = h.getResult(load_break, 0);

        const cond = c.oraArithCmpIOpCreate(self.ctx, loc, 0, break_flag_val, false_val);
        if (c.oraOperationIsNull(cond)) {
            @panic("Failed to create arith.cmpi for labeled block break guard");
        }
        h.appendOp(self.block, cond);
        const cond_val = h.getResult(cond, 0);

        const if_op = self.ora_dialect.createScfIf(cond_val, &[_]c.MlirType{}, loc);
        h.appendOp(self.block, if_op);

        const then_block = c.oraScfIfOpGetThenBlock(if_op);
        const else_block = c.oraScfIfOpGetElseBlock(if_op);
        if (c.oraBlockIsNull(then_block) or c.oraBlockIsNull(else_block)) {
            @panic("scf.if missing then/else blocks");
        }

        var lowerer_with_label = StatementLowerer.init(
            self.ctx,
            then_block,
            self.type_mapper,
            self.expr_lowerer,
            self.param_map,
            self.storage_map,
            self.local_var_map,
            self.locations,
            self.symbol_table,
            self.builtin_registry,
            self.allocator,
            self.current_function_return_type,
            self.current_function_return_type_info,
            self.ora_dialect,
            self.ensures_clauses,
        );
        lowerer_with_label.label_context = &label_ctx;
        lowerer_with_label.force_stack_memref = true;

        try lowerer_with_label.lowerStatement(&stmt);

        if (!helpers.blockEndsWithTerminator(&lowerer_with_label, then_block)) {
            const yield_op = self.ora_dialect.createScfYield(loc);
            h.appendOp(then_block, yield_op);
        }

        const else_yield = self.ora_dialect.createScfYield(loc);
        h.appendOp(else_block, else_yield);
    }
}

/// Lower labeled switch with continue support using scf.while
fn lowerLabeledSwitch(self: *const StatementLowerer, labeled_block: *const lib.ast.Statements.LabeledBlockNode) LoweringError!void {
    log.debug("[lowerLabeledSwitch] Starting labeled switch lowering\n", .{});
    const loc = self.fileLoc(labeled_block.span);

    // find the switch statement
    const switch_stmt = blk: {
        for (labeled_block.block.statements) |stmt| {
            if (stmt == .Switch) break :blk &stmt.Switch;
        }
        return LoweringError.InvalidSwitch;
    };

    // create memrefs for continue flag and switch value
    const i1_type = h.boolType(self.ctx);
    const empty_attr = c.oraNullAttrCreate();

    // continue flag memref
    const continue_flag_memref_type = h.memRefType(self.ctx, i1_type, 0, null, empty_attr, empty_attr);
    const continue_flag_alloca = self.ora_dialect.createMemrefAlloca(continue_flag_memref_type, loc);
    h.appendOp(self.block, continue_flag_alloca);
    const continue_flag_memref = h.getResult(continue_flag_alloca, 0);

    // switch value memref (use Ora types consistently)
    const condition_raw = self.expr_lowerer.lowerExpression(&switch_stmt.condition);
    const initial_value = helpers.ensureValue(self, condition_raw, loc);
    const value_type = c.oraValueGetType(initial_value);
    const value_memref_type = h.memRefType(self.ctx, value_type, 0, null, empty_attr, empty_attr);
    const value_alloca = self.ora_dialect.createMemrefAlloca(value_memref_type, loc);
    h.appendOp(self.block, value_alloca);
    const value_memref = h.getResult(value_alloca, 0);

    // return flag memref (for returns in labeled switches)
    const return_flag_memref_type = h.memRefType(self.ctx, i1_type, 0, null, empty_attr, empty_attr);
    const return_flag_alloca = self.ora_dialect.createMemrefAlloca(return_flag_memref_type, loc);
    h.appendOp(self.block, return_flag_alloca);
    const return_flag_memref = h.getResult(return_flag_alloca, 0);

    // return value memref (for return values in labeled switches)
    const return_value_type = if (self.current_function_return_type) |ret_type| ret_type else value_type;
    const return_value_memref_type = h.memRefType(self.ctx, return_value_type, 0, null, empty_attr, empty_attr);
    const return_value_alloca = self.ora_dialect.createMemrefAlloca(return_value_memref_type, loc);
    h.appendOp(self.block, return_value_alloca);
    const return_value_memref = h.getResult(return_value_alloca, 0);

    // initialize continue_flag to true, return_flag to false, and store initial value
    const true_val = helpers.createBoolConstant(self, true, loc);
    const false_val = helpers.createBoolConstant(self, false, loc);
    helpers.storeToMemref(self, true_val, continue_flag_memref, loc);
    helpers.storeToMemref(self, false_val, return_flag_memref, loc);
    helpers.storeToMemref(self, initial_value, value_memref, loc);

    // create scf.while operation
    const while_op = self.ora_dialect.createScfWhile(&[_]c.MlirValue{}, &[_]c.MlirType{}, loc);
    h.appendOp(self.block, while_op);

    // before region: load continue_flag and check condition
    const before_block = c.oraScfWhileOpGetBeforeBlock(while_op);
    if (c.oraBlockIsNull(before_block)) {
        @panic("scf.while missing before block");
    }

    const load_flag = self.ora_dialect.createMemrefLoad(continue_flag_memref, &[_]c.MlirValue{}, i1_type, loc);
    h.appendOp(before_block, load_flag);
    const should_continue = h.getResult(load_flag, 0);

    const condition_op = self.ora_dialect.createScfCondition(should_continue, &[_]c.MlirValue{}, loc);
    h.appendOp(before_block, condition_op);

    // after region: reset flag, load value, execute switch
    const after_block = c.oraScfWhileOpGetAfterBlock(while_op);
    if (c.oraBlockIsNull(after_block)) {
        @panic("scf.while missing after block");
    }

    // reset continue_flag to false
    const false_val_continue = helpers.createBoolConstant(self, false, loc);
    helpers.storeToMemref(self, false_val_continue, continue_flag_memref, loc);

    // load switch value
    const load_value = self.ora_dialect.createMemrefLoad(value_memref, &[_]c.MlirValue{}, value_type, loc);
    h.appendOp(after_block, load_value);
    const switch_value = h.getResult(load_value, 0);

    // lower switch cases with label context
    log.debug("[lowerLabeledSwitch] Lowering switch cases with label context\n", .{});
    try lowerSwitchCasesWithLabel(self, switch_stmt.cases, switch_value, 0, after_block, loc, switch_stmt.default_case, labeled_block.label, continue_flag_memref, value_memref, return_flag_memref, return_value_memref);

    // check if after_block already has a terminator (e.g., ora.return from a case)
    // only add yield if there's no terminator
    // note: scf.while's after region must end with scf.yield (not ora.yield!)
    const has_terminator = helpers.blockEndsWithTerminator(self, after_block);
    log.debug("[lowerLabeledSwitch] after_block has_terminator={}\n", .{has_terminator});
    if (!has_terminator) {
        log.debug("[lowerLabeledSwitch] Adding scf.yield to after_block (scf.while requires scf.yield)\n", .{});
        // scf.while's after region must end with scf.yield to continue the loop
        const yield_op = self.ora_dialect.createScfYield(loc);
        h.appendOp(after_block, yield_op);
    } else {
        log.debug("[lowerLabeledSwitch] after_block already has terminator, skipping yield\n", .{});
    }

    // after scf.while, check return flag and return if needed
    if (self.current_function_return_type) |ret_type| {
        // load return flag
        const load_return_flag = self.ora_dialect.createMemrefLoad(return_flag_memref, &[_]c.MlirValue{}, i1_type, loc);
        h.appendOp(self.block, load_return_flag);
        const should_return = h.getResult(load_return_flag, 0);

        // use ora.if so the then-region can legally contain ora.return
        const return_if_op = self.ora_dialect.createIf(should_return, loc);
        h.appendOp(self.block, return_if_op);

        // get the then and else blocks from ora.if
        const return_if_then_block = c.oraIfOpGetThenBlock(return_if_op);
        const return_if_else_block = c.oraIfOpGetElseBlock(return_if_op);
        if (c.oraBlockIsNull(return_if_then_block) or c.oraBlockIsNull(return_if_else_block)) {
            @panic("ora.if missing then/else blocks");
        }

        // then block: load return value and return directly
        const load_return_value = self.ora_dialect.createMemrefLoad(return_value_memref, &[_]c.MlirValue{}, ret_type, loc);
        h.appendOp(return_if_then_block, load_return_value);
        const return_val = h.getResult(load_return_value, 0);
        const return_op = self.ora_dialect.createFuncReturnWithValue(return_val, loc);
        h.appendOp(return_if_then_block, return_op);

        // else block: empty yield (no return, function continues to next statement)
        const else_yield_op = self.ora_dialect.createYield(&[_]c.MlirValue{}, loc);
        h.appendOp(return_if_else_block, else_yield_op);
    }
}

/// Lower switch cases with label context for continue support
fn lowerSwitchCasesWithLabel(
    self: *const StatementLowerer,
    cases: []const lib.ast.Expressions.SwitchCase,
    condition: c.MlirValue,
    case_idx: usize,
    target_block: c.MlirBlock,
    loc: c.MlirLocation,
    default_case: ?lib.ast.Statements.BlockNode,
    label: []const u8,
    continue_flag_memref: c.MlirValue,
    value_memref: c.MlirValue,
    return_flag_memref: c.MlirValue,
    return_value_memref: c.MlirValue,
) LoweringError!void {
    log.debug("[lowerSwitchCasesWithLabel] Starting, label={s}, case_idx={}, total_cases={}\n", .{ label, case_idx, cases.len });
    // create label context for labeled switch
    const label_ctx = LabelContext{
        .label = label,
        .continue_flag_memref = continue_flag_memref,
        .value_memref = value_memref,
        .return_flag_memref = return_flag_memref,
        .return_value_memref = return_value_memref,
        .label_type = .Switch,
        .parent = self.label_context,
    };

    // create statement lowerer with label context
    var lowerer_with_label = StatementLowerer.init(
        self.ctx,
        target_block,
        self.type_mapper,
        self.expr_lowerer,
        self.param_map,
        self.storage_map,
        self.local_var_map,
        self.locations,
        self.symbol_table,
        self.builtin_registry,
        self.allocator,
        self.current_function_return_type,
        self.current_function_return_type_info,
        self.ora_dialect,
        self.ensures_clauses,
    );
    lowerer_with_label.label_context = &label_ctx;

    // lower switch cases - now in control_flow.zig
    // for labeled switches, we don't need the result (returns are handled inside scf.if regions)
    _ = try control_flow.lowerSwitchCases(&lowerer_with_label, cases, condition, case_idx, target_block, loc, default_case);
}
