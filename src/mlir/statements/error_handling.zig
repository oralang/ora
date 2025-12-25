// ============================================================================
// Error Handling Statement Lowering
// ============================================================================
// Error handling operations: try/catch, error declarations

const std = @import("std");
const c = @import("mlir_c_api").c;
const lib = @import("ora_lib");
const h = @import("../helpers.zig");
const StatementLowerer = @import("statement_lowerer.zig").StatementLowerer;
const LoweringError = StatementLowerer.LoweringError;
const helpers = @import("helpers.zig");
const return_stmt = @import("return.zig");

fn getLastOperation(block: c.MlirBlock) c.MlirOperation {
    var op = c.mlirBlockGetFirstOperation(block);
    var last: c.MlirOperation = c.MlirOperation{};
    while (!c.mlirOperationIsNull(op)) {
        last = op;
        op = c.mlirOperationGetNextInBlock(op);
    }
    return last;
}

fn findTryCatchOperandAfter(block: c.MlirBlock, last_before: c.MlirOperation) ?c.MlirValue {
    var op = if (c.mlirOperationIsNull(last_before))
        c.mlirBlockGetFirstOperation(block)
    else
        c.mlirOperationGetNextInBlock(last_before);

    var last_try_operand: ?c.MlirValue = null;
    while (!c.mlirOperationIsNull(op)) {
        const name_id = c.mlirOperationGetName(op);
        const name_ref = c.mlirIdentifierStr(name_id);
        if (name_ref.length > 0 and std.mem.eql(u8, name_ref.data[0..name_ref.length], "ora.try_catch")) {
            if (c.mlirOperationGetNumOperands(op) > 0) {
                last_try_operand = c.mlirOperationGetOperand(op, 0);
            }
        }
        op = c.mlirOperationGetNextInBlock(op);
    }

    return last_try_operand;
}

/// Lower try-catch statements with exception handling
pub fn lowerTryBlock(self: *const StatementLowerer, try_stmt: *const lib.ast.Statements.TryBlockNode) LoweringError!void {
    const loc = self.fileLoc(try_stmt.span);
    const i1_type = h.boolType(self.ctx);
    const empty_attr = c.mlirAttributeGetNull();

    // create memrefs for return flag and return value (similar to labeled blocks)
    // this allows returns inside try blocks to store their values instead of using ora.return
    const return_flag_memref_type = c.mlirMemRefTypeGet(i1_type, 0, null, empty_attr, empty_attr);
    var return_flag_alloca_state = h.opState("memref.alloca", loc);
    c.mlirOperationStateAddResults(&return_flag_alloca_state, 1, @ptrCast(&return_flag_memref_type));
    const return_flag_alloca = c.mlirOperationCreate(&return_flag_alloca_state);
    h.appendOp(self.block, return_flag_alloca);
    const return_flag_memref = h.getResult(return_flag_alloca, 0);

    // return value memref (only if function has return type)
    const return_value_memref = if (self.current_function_return_type) |ret_type| blk: {
        const return_value_memref_type = c.mlirMemRefTypeGet(ret_type, 0, null, empty_attr, empty_attr);
        var return_value_alloca_state = h.opState("memref.alloca", loc);
        c.mlirOperationStateAddResults(&return_value_alloca_state, 1, @ptrCast(&return_value_memref_type));
        const return_value_alloca = c.mlirOperationCreate(&return_value_alloca_state);
        h.appendOp(self.block, return_value_alloca);
        break :blk h.getResult(return_value_alloca, 0);
    } else null;

    // initialize return flag to false
    const false_val = helpers.createBoolConstant(self, false, loc);
    helpers.storeToMemref(self, false_val, return_flag_memref, loc);

    // create a new StatementLowerer with in_try_block flag and memrefs set
    var try_lowerer = self.*;
    try_lowerer.in_try_block = true;
    try_lowerer.try_return_flag_memref = return_flag_memref;
    try_lowerer.try_return_value_memref = return_value_memref;

    const last_before_try = getLastOperation(self.block);

    // lower the try block body with the flag and memrefs set
    _ = try try_lowerer.lowerBlockBody(try_stmt.try_block, self.block);

    // if there's a catch block, lower it after the try block
    if (try_stmt.catch_block) |catch_block| {
        // if there's an error variable, create a placeholder value and add it to LocalVarMap
        if (catch_block.error_variable) |error_var_name| {
            const error_value = findTryCatchOperandAfter(self.block, last_before_try) orelse blk: {
                const error_type = c.mlirIntegerTypeGet(self.ctx, 256);
                var error_const_state = h.opState("arith.constant", loc);
                const error_const_attr = c.mlirIntegerAttrGet(error_type, 0);
                const error_value_id = h.identifier(self.ctx, "value");
                var error_attrs = [_]c.MlirNamedAttribute{c.mlirNamedAttributeGet(error_value_id, error_const_attr)};
                c.mlirOperationStateAddAttributes(&error_const_state, error_attrs.len, &error_attrs);
                c.mlirOperationStateAddResults(&error_const_state, 1, @ptrCast(&error_type));
                const error_const_op = c.mlirOperationCreate(&error_const_state);
                h.appendOp(self.block, error_const_op);
                break :blk h.getResult(error_const_op, 0);
            };

            // add error variable to LocalVarMap so it can be referenced in the catch block
            if (self.local_var_map) |lvm| {
                lvm.addLocalVar(error_var_name, error_value) catch {
                    std.debug.print("WARNING: Failed to add error variable '{s}' to local var map\n", .{error_var_name});
                };
            }
        }

        // catch block is also inside try context, so set the flag and memrefs
        var catch_lowerer = self.*;
        catch_lowerer.in_try_block = true;
        catch_lowerer.try_return_flag_memref = return_flag_memref;
        catch_lowerer.try_return_value_memref = return_value_memref;
        _ = try catch_lowerer.lowerBlockBody(catch_block.block, self.block);
    }

    // after try/catch, check return flag and return if needed
    if (self.current_function_return_type) |ret_type| {
        // load return flag
        var load_return_flag_state = h.opState("memref.load", loc);
        c.mlirOperationStateAddOperands(&load_return_flag_state, 1, @ptrCast(&return_flag_memref));
        c.mlirOperationStateAddResults(&load_return_flag_state, 1, @ptrCast(&i1_type));
        const load_return_flag = c.mlirOperationCreate(&load_return_flag_state);
        h.appendOp(self.block, load_return_flag);
        const should_return = h.getResult(load_return_flag, 0);

        // use ora.if to check return flag (ora.if allows ora.return inside its regions)
        const return_if_op = self.ora_dialect.createIf(should_return, loc);
        h.appendOp(self.block, return_if_op);

        // get the then and else blocks from ora.if
        const then_region = c.mlirOperationGetRegion(return_if_op, 0);
        const else_region = c.mlirOperationGetRegion(return_if_op, 1);
        const return_if_then_block = c.mlirRegionGetFirstBlock(then_region);
        const return_if_else_block = c.mlirRegionGetFirstBlock(else_region);

        // then block: load return value and return directly
        if (return_value_memref) |ret_val_memref| {
            var load_return_value_state = h.opState("memref.load", loc);
            c.mlirOperationStateAddOperands(&load_return_value_state, 1, @ptrCast(&ret_val_memref));
            c.mlirOperationStateAddResults(&load_return_value_state, 1, @ptrCast(&ret_type));
            const load_return_value = c.mlirOperationCreate(&load_return_value_state);
            h.appendOp(return_if_then_block, load_return_value);
            const return_val = h.getResult(load_return_value, 0);

            // insert ensures clause checks before return
            if (self.ensures_clauses.len > 0) {
                try return_stmt.lowerEnsuresBeforeReturn(self, return_if_then_block, try_stmt.span);
            }

            const return_op = self.ora_dialect.createFuncReturnWithValue(return_val, loc);
            h.appendOp(return_if_then_block, return_op);
        } else {
            // no return value
            if (self.ensures_clauses.len > 0) {
                try return_stmt.lowerEnsuresBeforeReturn(self, return_if_then_block, try_stmt.span);
            }
            const return_op = self.ora_dialect.createFuncReturn(loc);
            h.appendOp(return_if_then_block, return_op);
        }

        // else block: empty yield (no return, function continues to next statement)
        var else_yield_state = h.opState("ora.yield", loc);
        const else_yield_op = c.mlirOperationCreate(&else_yield_state);
        h.appendOp(return_if_else_block, else_yield_op);
    }
}

/// Lower error declarations
pub fn lowerErrorDecl(self: *const StatementLowerer, error_decl: *const lib.ast.Statements.ErrorDeclNode) LoweringError!void {
    const loc = self.fileLoc(error_decl.span);

    var state = h.opState("ora.error.decl", loc);

    // add error name as attribute
    const name_ref = c.mlirStringRefCreate(error_decl.name.ptr, error_decl.name.len);
    const name_attr = c.mlirStringAttrGet(self.ctx, name_ref);
    const name_id = h.identifier(self.ctx, "name");
    var attrs = [_]c.MlirNamedAttribute{c.mlirNamedAttributeGet(name_id, name_attr)};
    c.mlirOperationStateAddAttributes(&state, attrs.len, &attrs);

    // handle error parameters if present
    if (error_decl.parameters) |parameters| {
        for (parameters) |param| {
            const param_ref = c.mlirStringRefCreate(param.name.ptr, param.name.len);
            const param_attr = c.mlirStringAttrGet(self.ctx, param_ref);
            const param_id = h.identifier(self.ctx, "param");
            var param_attrs = [_]c.MlirNamedAttribute{c.mlirNamedAttributeGet(param_id, param_attr)};
            c.mlirOperationStateAddAttributes(&state, param_attrs.len, &param_attrs);
        }
    }

    const op = c.mlirOperationCreate(&state);
    h.appendOp(self.block, op);
}
