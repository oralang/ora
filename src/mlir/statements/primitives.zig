// ============================================================================
// Primitive Statement Lowering
// ============================================================================
// Primitive language operations: log, lock/unlock

const std = @import("std");
const c = @import("mlir_c_api").c;
const lib = @import("ora_lib");
const h = @import("../helpers.zig");
const StatementLowerer = @import("statement_lowerer.zig").StatementLowerer;
const LoweringError = StatementLowerer.LoweringError;
const helpers = @import("helpers.zig");
const error_handling = @import("../error_handling.zig");
const log = @import("log");

/// Lower log statements (event emission)
pub fn lowerLog(self: *const StatementLowerer, log_stmt: *const lib.ast.Statements.LogNode) LoweringError!void {
    const loc = self.fileLoc(log_stmt.span);
    const log_debug = std.process.hasEnvVar(self.allocator, "ORA_LOG_DEBUG") catch false;

    const sig_fields_opt = if (self.symbol_table) |st|
        st.log_signatures.get(log_stmt.event_name)
    else
        null;
    if (sig_fields_opt == null) {
        if (self.expr_lowerer.error_handler) |handler| {
            handler.reportError(
                .UndefinedSymbol,
                log_stmt.span,
                "log event not declared in contract",
                "add a log declaration or check the event name",
            ) catch {};
        }
        return;
    }
    if (sig_fields_opt.?.len != log_stmt.args.len) {
        if (self.expr_lowerer.error_handler) |handler| {
            handler.reportError(
                .TypeMismatch,
                log_stmt.span,
                "log argument count does not match signature",
                "check event parameter count",
            ) catch {};
        }
        return;
    }

    // lower and add log arguments as operands
    var operands: []c.MlirValue = &[_]c.MlirValue{};
    var operands_allocated = false;
    defer if (operands_allocated) self.allocator.free(operands);

    if (log_stmt.args.len > 0) {
        operands = try self.allocator.alloc(c.MlirValue, log_stmt.args.len);
        operands_allocated = true;
        for (log_stmt.args, 0..) |*arg, i| {
            const expected_type = if (sig_fields_opt != null and i < sig_fields_opt.?.len)
                sig_fields_opt.?[i].type_info
            else
                null;
            var value = helpers.lowerValueWithImplicitTry(@constCast(self), arg, expected_type);
            if (c.oraValueIsNull(value)) {
                if (self.expr_lowerer.error_handler) |handler| {
                    handler.reportError(
                        .MlirOperationFailed,
                        log_stmt.span,
                        "failed to lower log argument",
                        "check log argument expressions for unsupported constructs",
                    ) catch {};
                }
                return;
            }
            if (c.oraTypeIsNull(c.oraValueGetType(value))) {
                if (self.expr_lowerer.error_handler) |handler| {
                    handler.reportError(
                        .MlirOperationFailed,
                        log_stmt.span,
                        "log argument has null type",
                        "check log argument expressions for unsupported constructs",
                    ) catch {};
                }
                return;
            }
            if (expected_type) |expected_ti| {
                const expected_mlir = self.expr_lowerer.type_mapper.toMlirType(expected_ti);
                value = self.expr_lowerer.convertToType(
                    value,
                    expected_mlir,
                    error_handling.getSpanFromExpression(arg),
                );
                if (c.oraValueIsNull(value)) {
                    if (self.expr_lowerer.error_handler) |handler| {
                        handler.reportError(
                            .MlirOperationFailed,
                            log_stmt.span,
                            "failed to convert log argument to expected type",
                            "check log argument types against the event signature",
                        ) catch {};
                    }
                    return;
                }
                if (c.oraTypeIsNull(c.oraValueGetType(value))) {
                    if (self.expr_lowerer.error_handler) |handler| {
                        handler.reportError(
                            .MlirOperationFailed,
                            log_stmt.span,
                            "log argument conversion produced null type",
                            "check log argument types against the event signature",
                        ) catch {};
                    }
                    return;
                }
            }

            const error_union_success = c.oraErrorUnionTypeGetSuccessType(c.oraValueGetType(value));
            if (!c.oraTypeIsNull(error_union_success)) {
                if (self.expr_lowerer.error_handler) |handler| {
                    handler.reportError(
                        .TypeMismatch,
                        log_stmt.span,
                        "log argument is still an error union",
                        "use try to unwrap or adjust the log signature",
                    ) catch {};
                }
                return;
            }

            if (log_debug) {
                log.debug(
                    "[lowerLog] arg {d}: value_ptr=0x{x} type_ptr=0x{x}\n",
                    .{ i, @intFromPtr(value.ptr), @intFromPtr(c.oraValueGetType(value).ptr) },
                );
            }
            operands[i] = value;
        }
    }

    const op = self.ora_dialect.createLog(log_stmt.event_name, operands, loc);
    h.appendOp(self.block, op);
}

/// Lower lock statements
pub fn lowerLock(self: *const StatementLowerer, lock_stmt: *const lib.ast.Statements.LockNode) LoweringError!void {
    const loc = self.fileLoc(lock_stmt.span);
    const key = (try helpers.buildRuntimeLockKeyForPath(self, &lock_stmt.path)) orelse {
        if (self.expr_lowerer.error_handler) |handler| {
            handler.reportError(
                .UnsupportedFeature,
                lock_stmt.span,
                "@lock supports only storage identifier or storage identifier[index] paths",
                null,
            ) catch {};
        }
        return LoweringError.InvalidLValue;
    };
    defer self.allocator.free(key);
    const resource_expr = helpers.lockRuntimeResourceExpr(&lock_stmt.path);
    const resource = self.expr_lowerer.lowerExpression(resource_expr);
    const op = self.ora_dialect.createLockWithKey(resource, key, loc);
    h.appendOp(self.block, op);
}

/// Lower unlock statements
pub fn lowerUnlock(self: *const StatementLowerer, unlock_stmt: *const lib.ast.Statements.UnlockNode) LoweringError!void {
    const loc = self.fileLoc(unlock_stmt.span);
    const key = (try helpers.buildRuntimeLockKeyForPath(self, &unlock_stmt.path)) orelse {
        if (self.expr_lowerer.error_handler) |handler| {
            handler.reportError(
                .UnsupportedFeature,
                unlock_stmt.span,
                "@unlock supports only storage identifier or storage identifier[index] paths",
                null,
            ) catch {};
        }
        return LoweringError.InvalidLValue;
    };
    defer self.allocator.free(key);
    const resource_expr = helpers.lockRuntimeResourceExpr(&unlock_stmt.path);
    const resource = self.expr_lowerer.lowerExpression(resource_expr);
    const op = self.ora_dialect.createUnlockWithKey(resource, key, loc);
    h.appendOp(self.block, op);
}
