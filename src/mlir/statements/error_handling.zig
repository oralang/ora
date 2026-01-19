// ============================================================================
// Error Handling Statement Lowering
// ============================================================================
// Error handling operations: try/catch, error declarations

const c = @import("mlir_c_api").c;
const lib = @import("ora_lib");
const h = @import("../helpers.zig");
const StatementLowerer = @import("statement_lowerer.zig").StatementLowerer;
const LabelContext = @import("statement_lowerer.zig").LabelContext;
const LoweringError = StatementLowerer.LoweringError;
const helpers = @import("helpers.zig");
const ExpressionLowerer = @import("../expressions/mod.zig").ExpressionLowerer;
const log = @import("log");
const constants = @import("../lower.zig");

/// Check if we're inside a for/while loop context where we can't use ora.return
fn isInsideLoopContext(label_context: ?*const LabelContext) bool {
    var ctx_opt = label_context;
    while (ctx_opt) |ctx| : (ctx_opt = ctx.parent) {
        switch (ctx.label_type) {
            .For, .While => return true,
            .Block, .Switch => {},
        }
    }
    return false;
}

// Lower block body with ora.yield for try/catch regions (no memref return shims).
fn lowerBlockBodyWithYield(
    self: *const StatementLowerer,
    block_body: lib.ast.Statements.BlockNode,
    target_block: c.MlirBlock,
    expected_result_type: ?c.MlirType,
) LoweringError!void {
    var temp_lowerer = self.*;
    temp_lowerer.block = target_block;

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
    expr_lowerer.refinement_base_cache = self.expr_lowerer.refinement_base_cache;
    expr_lowerer.refinement_guard_cache = self.expr_lowerer.refinement_guard_cache;
    expr_lowerer.current_function_return_type = self.current_function_return_type;
    expr_lowerer.current_function_return_type_info = self.current_function_return_type_info;
    expr_lowerer.in_try_block = self.in_try_block;
    temp_lowerer.expr_lowerer = &expr_lowerer;

    var has_terminator = false;

    for (block_body.statements) |stmt| {
        if (has_terminator) break;

        switch (stmt) {
            .Return => |ret| {
                const loc = temp_lowerer.fileLoc(ret.span);
                if (expected_result_type) |ret_type| {
                    var v: c.MlirValue = undefined;
                    if (ret.value) |value_expr| {
                        v = expr_lowerer.lowerExpression(&value_expr);

                        const is_error_union = if (temp_lowerer.current_function_return_type_info) |ti|
                            helpers.isErrorUnionTypeInfo(ti)
                        else
                            false;

                        if (is_error_union) {
                            const err_info = helpers.getErrorUnionPayload(&temp_lowerer, &value_expr, v, ret_type, target_block, loc);
                            v = helpers.encodeErrorUnionValue(&temp_lowerer, err_info.payload, err_info.is_error, ret_type, target_block, ret.span, loc);
                        } else {
                            const value_type = c.oraValueGetType(v);
                            if (!c.oraTypeEqual(value_type, ret_type)) {
                                v = helpers.convertValueToType(&temp_lowerer, v, ret_type, ret.span, loc);
                            }
                        }
                    } else {
                        v = try helpers.createDefaultValueForType(&temp_lowerer, ret_type, loc);
                    }

                    const yield_op = temp_lowerer.ora_dialect.createYield(&[_]c.MlirValue{v}, loc);
                    h.appendOp(target_block, yield_op);
                } else {
                    const yield_op = temp_lowerer.ora_dialect.createYield(&[_]c.MlirValue{}, loc);
                    h.appendOp(target_block, yield_op);
                }
                has_terminator = true;
            },
            else => {
                expr_lowerer.block = temp_lowerer.block;
                const maybe_new_block = try temp_lowerer.lowerStatement(&stmt);
                if (maybe_new_block) |new_block| {
                    temp_lowerer.block = new_block;
                    expr_lowerer.block = new_block;
                }
                if (helpers.blockEndsWithTerminator(&temp_lowerer, temp_lowerer.block)) {
                    has_terminator = true;
                }
            },
        }
    }

    if (!has_terminator and !helpers.blockEndsWithTerminator(&temp_lowerer, temp_lowerer.block)) {
        const yield_op = temp_lowerer.ora_dialect.createYield(&[_]c.MlirValue{}, self.fileLoc(block_body.span));
        h.appendOp(temp_lowerer.block, yield_op);
    }
}

// legacy helpers removed (try/catch now uses ora.try_stmt regions directly)

/// Lower try-catch statements with exception handling
pub fn lowerTryBlock(self: *const StatementLowerer, try_stmt: *const lib.ast.Statements.TryBlockNode) LoweringError!void {
    const loc = self.fileLoc(try_stmt.span);
    const try_has_return = helpers.blockHasReturn(self, try_stmt.try_block);
    const catch_has_return = if (try_stmt.catch_block) |cb|
        helpers.blockHasReturn(self, cb.block)
    else
        false;

    const result_type_opt: ?c.MlirType = if ((try_has_return or catch_has_return) and self.current_function_return_type != null)
        self.current_function_return_type.?
    else
        null;

    const result_types = if (result_type_opt) |rt| &[_]c.MlirType{rt} else &[_]c.MlirType{};
    const try_stmt_op = self.ora_dialect.createTryStmt(result_types, loc);
    h.appendOp(self.block, try_stmt_op);

    const try_block = c.oraTryStmtOpGetTryBlock(try_stmt_op);
    const catch_block_mlir = c.oraTryStmtOpGetCatchBlock(try_stmt_op);
    if (c.oraBlockIsNull(try_block) or c.oraBlockIsNull(catch_block_mlir)) {
        @panic("ora.try_stmt missing try/catch blocks");
    }

    var try_lowerer = self.*;
    try_lowerer.block = try_block;
    try_lowerer.in_try_block = true;
    try_lowerer.try_return_flag_memref = null;
    try_lowerer.try_return_value_memref = null;
    try lowerBlockBodyWithYield(&try_lowerer, try_stmt.try_block, try_block, result_type_opt);
    if (!helpers.blockEndsWithTerminator(&try_lowerer, try_block)) {
        const yield_op = self.ora_dialect.createYield(&[_]c.MlirValue{}, loc);
        h.appendOp(try_block, yield_op);
    }

    if (try_stmt.catch_block) |catch_block| {
        var catch_lowerer = self.*;
        catch_lowerer.block = catch_block_mlir;
        catch_lowerer.in_try_block = true;
        catch_lowerer.try_return_flag_memref = null;
        catch_lowerer.try_return_value_memref = null;

        if (catch_block.error_variable) |error_var_name| {
            const err_type = c.oraIntegerTypeCreate(self.ctx, constants.DEFAULT_INTEGER_BITS);
            const err_val = c.mlirBlockAddArgument(catch_block_mlir, err_type, loc);
            if (catch_lowerer.local_var_map) |lvm| {
                lvm.addLocalVar(error_var_name, err_val) catch {
                    log.warn("Failed to add error variable '{s}' to local var map\n", .{error_var_name});
                };
            }
        }

        try lowerBlockBodyWithYield(&catch_lowerer, catch_block.block, catch_block_mlir, result_type_opt);
        if (!helpers.blockEndsWithTerminator(&catch_lowerer, catch_block_mlir)) {
            const yield_op = self.ora_dialect.createYield(&[_]c.MlirValue{}, loc);
            h.appendOp(catch_block_mlir, yield_op);
        }
    } else {
        const yield_op = self.ora_dialect.createYield(&[_]c.MlirValue{}, loc);
        h.appendOp(catch_block_mlir, yield_op);
    }

    if (result_type_opt != null and try_has_return and catch_has_return and !isInsideLoopContext(self.label_context)) {
        const result_val = h.getResult(try_stmt_op, 0);
        const ret_op = self.ora_dialect.createFuncReturnWithValue(result_val, loc);
        h.appendOp(self.block, ret_op);
    }
}

// Error declarations are lowered at the declaration pass, not here.
