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

/// Lower break statements with label support
pub fn lowerBreak(self: *const StatementLowerer, break_stmt: *const lib.ast.Statements.BreakNode) LoweringError!void {
    const loc = self.fileLoc(break_stmt.span);
    const h = @import("../helpers.zig");

    // Collect break value if present
    var operands = std.ArrayList(c.MlirValue){};
    defer operands.deinit(self.allocator);

    if (break_stmt.value) |value_expr| {
        const value = self.expr_lowerer.lowerExpression(value_expr);
        operands.append(self.allocator, value) catch unreachable;
    }

    // Use ora.break operation with optional label and values
    const label = if (break_stmt.label) |l| l else null;
    const break_op = self.ora_dialect.createBreak(label, operands.items, loc);
    h.appendOp(self.block, break_op);
}

/// Lower continue statements with label support
pub fn lowerContinue(self: *const StatementLowerer, continue_stmt: *const lib.ast.Statements.ContinueNode) LoweringError!void {
    const loc = self.fileLoc(continue_stmt.span);
    const h = @import("../helpers.zig");

    if (continue_stmt.label) |label| {
        // Check if this continue targets a labeled switch (special handling for value replacement)
        if (self.label_context) |label_ctx| {
            if (std.mem.eql(u8, label, label_ctx.label)) {
                // Handle switch-specific continue (with value replacement)
                if (label_ctx.label_type == .Switch and label_ctx.continue_flag_memref != null) {
                    try handleLabeledSwitchContinue(self, continue_stmt, label_ctx, loc);
                    return;
                }
                // For other labeled contexts (while, for, block), use ora.continue with label
            }
        }

        // Labeled continue - use ora.continue with label
        const continue_op = self.ora_dialect.createContinue(label, loc);
        h.appendOp(self.block, continue_op);
    } else {
        // Unlabeled continue - use ora.continue without label
        const continue_op = self.ora_dialect.createContinue(null, loc);
        h.appendOp(self.block, continue_op);
    }
}

/// Handle continue to a labeled switch (stores value and sets continue flag)
fn handleLabeledSwitchContinue(
    self: *const StatementLowerer,
    continue_stmt: *const lib.ast.Statements.ContinueNode,
    label_ctx: *const LabelContext,
    loc: c.MlirLocation,
) LoweringError!void {
    // This only works for labeled switches with continue flag memref
    if (label_ctx.continue_flag_memref == null or label_ctx.value_memref == null) {
        // Fallback to regular continue
        const continue_op = self.ora_dialect.createContinue(label_ctx.label, loc);
        const h = @import("../helpers.zig");
        h.appendOp(self.block, continue_op);
        return;
    }

    // Unwrap the memrefs (we know they're not null from the check above)
    const value_memref = label_ctx.value_memref.?;
    const continue_flag = label_ctx.continue_flag_memref.?;

    // Store new value if provided
    if (continue_stmt.value) |value_expr| {
        const value = self.expr_lowerer.lowerExpression(value_expr);
        const value_to_store = helpers.ensureValue(self, value, loc);

        // Get target type from memref element type
        const memref_type = c.mlirValueGetType(value_memref);
        const element_type = c.mlirShapedTypeGetElementType(memref_type);

        // Convert value to match memref element type
        const final_value = helpers.convertValueToType(self, value_to_store, element_type, continue_stmt.span, loc);

        // Store converted value
        helpers.storeToMemref(self, final_value, value_memref, loc);
    }

    // Set continue_flag to true
    const true_val = helpers.createBoolConstant(self, true, loc);
    helpers.storeToMemref(self, true_val, continue_flag, loc);

    // Use ora.continue to exit current case (switch-specific handling)
    const continue_op = self.ora_dialect.createContinue(label_ctx.label, loc);
    const h = @import("../helpers.zig");
    h.appendOp(self.block, continue_op);
}

/// Lower labeled blocks (including labeled switch, while, for)
pub fn lowerLabeledBlock(self: *const StatementLowerer, labeled_block: *const lib.ast.Statements.LabeledBlockNode) LoweringError!void {
    // Check if this is a labeled switch (special handling for continue with value replacement)
    if (labeled_block.block.statements.len > 0) {
        const first_stmt = labeled_block.block.statements[0];
        if (first_stmt == .Switch) {
            try lowerLabeledSwitch(self, labeled_block);
            return;
        }
    }

    // Regular labeled block - create label context and lower the block
    const label_ctx = LabelContext{
        .label = labeled_block.label,
        .label_type = .Block,
    };

    // Create statement lowerer with label context for break/continue
    var lowerer_with_label = StatementLowerer.init(
        self.ctx,
        self.block,
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

    _ = try lowerer_with_label.lowerBlockBody(labeled_block.block, self.block);
}

/// Lower labeled switch with continue support using scf.while
fn lowerLabeledSwitch(self: *const StatementLowerer, labeled_block: *const lib.ast.Statements.LabeledBlockNode) LoweringError!void {
    std.debug.print("[lowerLabeledSwitch] Starting labeled switch lowering\n", .{});
    const loc = self.fileLoc(labeled_block.span);
    const h = @import("../helpers.zig");

    // Find the switch statement
    const switch_stmt = blk: {
        for (labeled_block.block.statements) |stmt| {
            if (stmt == .Switch) break :blk &stmt.Switch;
        }
        return LoweringError.InvalidSwitch;
    };

    // Create memrefs for continue flag and switch value
    const i1_type = h.boolType(self.ctx);
    const empty_attr = c.mlirAttributeGetNull();

    // Continue flag memref
    const continue_flag_memref_type = c.mlirMemRefTypeGet(i1_type, 0, null, empty_attr, empty_attr);
    var continue_flag_alloca_state = h.opState("memref.alloca", loc);
    c.mlirOperationStateAddResults(&continue_flag_alloca_state, 1, @ptrCast(&continue_flag_memref_type));
    const continue_flag_alloca = c.mlirOperationCreate(&continue_flag_alloca_state);
    h.appendOp(self.block, continue_flag_alloca);
    const continue_flag_memref = h.getResult(continue_flag_alloca, 0);

    // Switch value memref (use Ora types consistently)
    const condition_raw = self.expr_lowerer.lowerExpression(&switch_stmt.condition);
    const initial_value = helpers.ensureValue(self, condition_raw, loc);
    const value_type = c.mlirValueGetType(initial_value);
    const value_memref_type = c.mlirMemRefTypeGet(value_type, 0, null, empty_attr, empty_attr);
    var value_alloca_state = h.opState("memref.alloca", loc);
    c.mlirOperationStateAddResults(&value_alloca_state, 1, @ptrCast(&value_memref_type));
    const value_alloca = c.mlirOperationCreate(&value_alloca_state);
    h.appendOp(self.block, value_alloca);
    const value_memref = h.getResult(value_alloca, 0);

    // Return flag memref (for returns in labeled switches)
    const return_flag_memref_type = c.mlirMemRefTypeGet(i1_type, 0, null, empty_attr, empty_attr);
    var return_flag_alloca_state = h.opState("memref.alloca", loc);
    c.mlirOperationStateAddResults(&return_flag_alloca_state, 1, @ptrCast(&return_flag_memref_type));
    const return_flag_alloca = c.mlirOperationCreate(&return_flag_alloca_state);
    h.appendOp(self.block, return_flag_alloca);
    const return_flag_memref = h.getResult(return_flag_alloca, 0);

    // Return value memref (for return values in labeled switches)
    const return_value_type = if (self.current_function_return_type) |ret_type| ret_type else value_type;
    const return_value_memref_type = c.mlirMemRefTypeGet(return_value_type, 0, null, empty_attr, empty_attr);
    var return_value_alloca_state = h.opState("memref.alloca", loc);
    c.mlirOperationStateAddResults(&return_value_alloca_state, 1, @ptrCast(&return_value_memref_type));
    const return_value_alloca = c.mlirOperationCreate(&return_value_alloca_state);
    h.appendOp(self.block, return_value_alloca);
    const return_value_memref = h.getResult(return_value_alloca, 0);

    // Initialize continue_flag to true, return_flag to false, and store initial value
    const true_val = helpers.createBoolConstant(self, true, loc);
    const false_val = helpers.createBoolConstant(self, false, loc);
    helpers.storeToMemref(self, true_val, continue_flag_memref, loc);
    helpers.storeToMemref(self, false_val, return_flag_memref, loc);
    helpers.storeToMemref(self, initial_value, value_memref, loc);

    // Create scf.while operation
    var while_state = h.opState("scf.while", loc);

    // Before region: load continue_flag and check condition
    const before_region = c.mlirRegionCreate();
    const before_block = c.mlirBlockCreate(0, null, null);
    c.mlirRegionInsertOwnedBlock(before_region, 0, before_block);

    var load_flag_state = h.opState("memref.load", loc);
    c.mlirOperationStateAddOperands(&load_flag_state, 1, @ptrCast(&continue_flag_memref));
    c.mlirOperationStateAddResults(&load_flag_state, 1, @ptrCast(&i1_type));
    const load_flag = c.mlirOperationCreate(&load_flag_state);
    h.appendOp(before_block, load_flag);
    const should_continue = h.getResult(load_flag, 0);

    var condition_state = h.opState("scf.condition", loc);
    c.mlirOperationStateAddOperands(&condition_state, 1, @ptrCast(&should_continue));
    const condition_op = c.mlirOperationCreate(&condition_state);
    h.appendOp(before_block, condition_op);

    // After region: reset flag, load value, execute switch
    const after_region = c.mlirRegionCreate();
    const after_block = c.mlirBlockCreate(0, null, null);
    c.mlirRegionInsertOwnedBlock(after_region, 0, after_block);

    // Reset continue_flag to false
    const false_val_continue = helpers.createBoolConstant(self, false, loc);
    helpers.storeToMemref(self, false_val_continue, continue_flag_memref, loc);

    // Load switch value
    var load_value_state = h.opState("memref.load", loc);
    c.mlirOperationStateAddOperands(&load_value_state, 1, @ptrCast(&value_memref));
    c.mlirOperationStateAddResults(&load_value_state, 1, @ptrCast(&value_type));
    const load_value = c.mlirOperationCreate(&load_value_state);
    h.appendOp(after_block, load_value);
    const switch_value = h.getResult(load_value, 0);

    // Lower switch cases with label context
    std.debug.print("[lowerLabeledSwitch] Lowering switch cases with label context\n", .{});
    try lowerSwitchCasesWithLabel(self, switch_stmt.cases, switch_value, 0, after_block, loc, switch_stmt.default_case, labeled_block.label, continue_flag_memref, value_memref, return_flag_memref, return_value_memref);

    // Check if after_block already has a terminator (e.g., ora.return from a case)
    // Only add yield if there's no terminator
    // Note: scf.while's after region must end with scf.yield (not ora.yield!)
    const has_terminator = helpers.blockEndsWithTerminator(self, after_block);
    std.debug.print("[lowerLabeledSwitch] after_block has_terminator={}\n", .{has_terminator});
    if (!has_terminator) {
        std.debug.print("[lowerLabeledSwitch] Adding scf.yield to after_block (scf.while requires scf.yield)\n", .{});
        // scf.while's after region must end with scf.yield to continue the loop
        var yield_state = h.opState("scf.yield", loc);
        const yield_op = c.mlirOperationCreate(&yield_state);
        h.appendOp(after_block, yield_op);
    } else {
        std.debug.print("[lowerLabeledSwitch] after_block already has terminator, skipping yield\n", .{});
    }

    // Add regions and create while operation
    c.mlirOperationStateAddOwnedRegions(&while_state, 1, @ptrCast(&before_region));
    c.mlirOperationStateAddOwnedRegions(&while_state, 1, @ptrCast(&after_region));
    const while_op = c.mlirOperationCreate(&while_state);
    h.appendOp(self.block, while_op);

    // After scf.while, check return flag and return if needed
    if (self.current_function_return_type) |ret_type| {
        // Load return flag
        var load_return_flag_state = h.opState("memref.load", loc);
        c.mlirOperationStateAddOperands(&load_return_flag_state, 1, @ptrCast(&return_flag_memref));
        c.mlirOperationStateAddResults(&load_return_flag_state, 1, @ptrCast(&i1_type));
        const load_return_flag = c.mlirOperationCreate(&load_return_flag_state);
        h.appendOp(self.block, load_return_flag);
        const should_return = h.getResult(load_return_flag, 0);

        // Use ora.if to check return flag (ora.if allows ora.return inside its regions)
        const return_if_op = self.ora_dialect.createIf(should_return, loc);
        h.appendOp(self.block, return_if_op);

        // Get the then and else blocks from ora.if
        const then_region = c.mlirOperationGetRegion(return_if_op, 0);
        const else_region = c.mlirOperationGetRegion(return_if_op, 1);
        const return_if_then_block = c.mlirRegionGetFirstBlock(then_region);
        const return_if_else_block = c.mlirRegionGetFirstBlock(else_region);

        // Then block: load return value and return directly
        var load_return_value_state = h.opState("memref.load", loc);
        c.mlirOperationStateAddOperands(&load_return_value_state, 1, @ptrCast(&return_value_memref));
        c.mlirOperationStateAddResults(&load_return_value_state, 1, @ptrCast(&ret_type));
        const load_return_value = c.mlirOperationCreate(&load_return_value_state);
        h.appendOp(return_if_then_block, load_return_value);
        const return_val = h.getResult(load_return_value, 0);
        const return_op = self.ora_dialect.createFuncReturnWithValue(return_val, loc);
        h.appendOp(return_if_then_block, return_op);

        // Else block: empty yield (no return, function continues to next statement)
        var else_yield_state = h.opState("ora.yield", loc);
        const else_yield_op = c.mlirOperationCreate(&else_yield_state);
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
    std.debug.print("[lowerSwitchCasesWithLabel] Starting, label={s}, case_idx={}, total_cases={}\n", .{ label, case_idx, cases.len });
    // Create label context for labeled switch
    const label_ctx = LabelContext{
        .label = label,
        .continue_flag_memref = continue_flag_memref,
        .value_memref = value_memref,
        .return_flag_memref = return_flag_memref,
        .return_value_memref = return_value_memref,
        .label_type = .Switch,
    };

    // Create statement lowerer with label context
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

    // Lower switch cases - now in control_flow.zig
    // For labeled switches, we don't need the result (returns are handled inside scf.if regions)
    _ = try control_flow.lowerSwitchCases(&lowerer_with_label, cases, condition, case_idx, target_block, loc, default_case);
}
