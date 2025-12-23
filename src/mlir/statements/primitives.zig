// ============================================================================
// Primitive Statement Lowering
// ============================================================================
// Primitive language operations: log, lock/unlock

const c = @import("../c.zig").c;
const lib = @import("ora_lib");
const h = @import("../helpers.zig");
const StatementLowerer = @import("statement_lowerer.zig").StatementLowerer;
const LoweringError = StatementLowerer.LoweringError;

/// Lower log statements (event emission)
pub fn lowerLog(self: *const StatementLowerer, log_stmt: *const lib.ast.Statements.LogNode) LoweringError!void {
    const loc = self.fileLoc(log_stmt.span);

    var state = h.opState("ora.log", loc);

    // Add event name as attribute
    const event_ref = c.mlirStringRefCreate(log_stmt.event_name.ptr, log_stmt.event_name.len);
    const event_attr = c.mlirStringAttrGet(self.ctx, event_ref);
    const event_id = h.identifier(self.ctx, "event");
    var attrs = [_]c.MlirNamedAttribute{c.mlirNamedAttributeGet(event_id, event_attr)};
    c.mlirOperationStateAddAttributes(&state, attrs.len, &attrs);

    // Lower and add log arguments as operands
    if (log_stmt.args.len > 0) {
        var operands = try self.allocator.alloc(c.MlirValue, log_stmt.args.len);
        defer self.allocator.free(operands);

        for (log_stmt.args, 0..) |*arg, i| {
            operands[i] = self.expr_lowerer.lowerExpression(arg);
        }

        c.mlirOperationStateAddOperands(&state, @intCast(operands.len), operands.ptr);
    }

    const op = c.mlirOperationCreate(&state);
    h.appendOp(self.block, op);
}

/// Lower lock statements
pub fn lowerLock(self: *const StatementLowerer, lock_stmt: *const lib.ast.Statements.LockNode) LoweringError!void {
    const loc = self.fileLoc(lock_stmt.span);
    const path_value = self.expr_lowerer.lowerExpression(&lock_stmt.path);
    const op = self.ora_dialect.createLock(path_value, loc);
    h.appendOp(self.block, op);
}

/// Lower unlock statements
pub fn lowerUnlock(self: *const StatementLowerer, unlock_stmt: *const lib.ast.Statements.UnlockNode) LoweringError!void {
    const loc = self.fileLoc(unlock_stmt.span);
    const path_value = self.expr_lowerer.lowerExpression(&unlock_stmt.path);
    const op = self.ora_dialect.createUnlock(path_value, loc);
    h.appendOp(self.block, op);
}
