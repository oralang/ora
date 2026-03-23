const std = @import("std");
const mlir = @import("mlir_c_api").c;
const ast = @import("../ast/mod.zig");
const hir_locals = @import("locals.zig");
const source = @import("../source/mod.zig");
const analysis = @import("analysis.zig");
const support = @import("support.zig");

const LoopContext = support.LoopContext;
const BlockContext = support.BlockContext;
const SwitchContext = support.SwitchContext;
const SwitchPatternData = support.SwitchPatternData;
const LocalEnv = hir_locals.LocalEnv;
const LocalId = hir_locals.LocalId;
const LocalIdList = hir_locals.LocalIdList;
const LocalIdSet = hir_locals.LocalIdSet;
const appendEmptyYield = support.appendEmptyYield;
const appendOraYieldValues = support.appendOraYieldValues;
const appendOp = support.appendOp;
const appendValueOp = support.appendValueOp;
const blockEndsWithTerminator = support.blockEndsWithTerminator;
const boolType = support.boolType;
const createIntegerConstant = support.createIntegerConstant;
const defaultIntegerType = support.defaultIntegerType;
const exprRange = support.exprRange;
const memRefType = support.memRefType;
const namedBoolAttr = support.namedBoolAttr;
const bodyContainsStructuredLoopControl = analysis.bodyContainsStructuredLoopControl;
const bodyContainsSwitchBreak = analysis.bodyContainsSwitchBreak;
const bodyMayReturn = analysis.bodyMayReturn;
const collectIfCarriedLocals = analysis.collectIfCarriedLocals;
const collectLoopCarriedLocals = analysis.collectLoopCarriedLocals;
const collectSwitchCarriedLocals = analysis.collectSwitchCarriedLocals;
const collectTryCarriedLocals = analysis.collectTryCarriedLocals;
const switchMayReturn = analysis.switchMayReturn;

pub fn mixin(FunctionLowerer: type, Lowerer: type) type {
    _ = Lowerer;
    return struct {
        pub fn lowerIfStmt(self: *FunctionLowerer, if_stmt: ast.IfStmt, locals: *LocalEnv) anyerror!bool {
            const has_return =
                bodyMayReturn(self.parent.file, if_stmt.then_body) or
                (if_stmt.else_body != null and bodyMayReturn(self.parent.file, if_stmt.else_body.?));

            if (if_stmt.else_body == null and self.deferred_return_flag == null) {
                const then_body = self.parent.file.body(if_stmt.then_body).*;
                if (then_body.statements.len == 1 and self.parent.file.statement(then_body.statements[0]).* == .Return) {
                    const condition = try self.lowerExpr(if_stmt.condition, locals);
                    const loc = self.parent.location(if_stmt.range);
                    const branch = mlir.oraConditionalReturnOpCreate(self.parent.context, loc, condition);
                    if (mlir.oraOperationIsNull(branch)) return error.MlirOperationCreationFailed;
                    appendOp(self.block, branch);

                    const then_block = mlir.oraConditionalReturnOpGetThenBlock(branch);
                    const else_block = mlir.oraConditionalReturnOpGetElseBlock(branch);
                    if (mlir.oraBlockIsNull(then_block) or mlir.oraBlockIsNull(else_block)) {
                        return error.MlirOperationCreationFailed;
                    }

                    var then_lowerer = self.*;
                    then_lowerer.block = then_block;
                    var then_locals = try self.cloneLocals(locals);
                    const then_terminated = try then_lowerer.lowerBody(if_stmt.then_body, &then_locals);
                    if (!then_terminated) {
                        try self.appendUnsupportedControlPlaceholder("ora.if_placeholder", if_stmt.range);
                        return false;
                    }
                    try appendEmptyYield(self.parent.context, else_block, loc);
                    return false;
                }
            }

            const condition = try self.lowerExpr(if_stmt.condition, locals);
            const loc = self.parent.location(if_stmt.range);
            const created_deferred_return = has_return and self.deferred_return_flag == null;
            if (created_deferred_return) {
                try self.ensureDeferredReturnSlots(if_stmt.range);
            }
            var carried_locals: LocalIdList = .{};
            var carried_seen = LocalIdSet.init(self.parent.allocator);
            if (!try collectIfCarriedLocals(self.parent.allocator, self.parent.file, if_stmt.then_body, locals, &carried_locals, &carried_seen)) {
                try self.appendUnsupportedControlPlaceholder("ora.if_placeholder", if_stmt.range);
                return false;
            }
            if (if_stmt.else_body) |else_body| {
                if (!try collectIfCarriedLocals(self.parent.allocator, self.parent.file, else_body, locals, &carried_locals, &carried_seen)) {
                    try self.appendUnsupportedControlPlaceholder("ora.if_placeholder", if_stmt.range);
                    return false;
                }
            }
            carried_locals = try self.filterCarriedLocals(locals, carried_locals.items);

            const result_types = if (carried_locals.items.len == 0)
                std.ArrayList(mlir.MlirType){}
            else
                (try self.buildCarriedResultTypes(locals, carried_locals.items)) orelse {
                    try self.appendUnsupportedControlPlaceholder("ora.if_placeholder", if_stmt.range);
                    return false;
                };

            const op = mlir.oraScfIfOpCreate(
                self.parent.context,
                loc,
                condition,
                result_types.items.ptr,
                result_types.items.len,
                true,
            );
            if (mlir.oraOperationIsNull(op)) return error.MlirOperationCreationFailed;
            appendOp(self.block, op);

            const then_block = mlir.oraScfIfOpGetThenBlock(op);
            const else_block = mlir.oraScfIfOpGetElseBlock(op);
            if (mlir.oraBlockIsNull(then_block) or mlir.oraBlockIsNull(else_block)) {
                return error.MlirOperationCreationFailed;
            }

            var then_lowerer = self.*;
            then_lowerer.block = then_block;
            then_lowerer.current_scf_carried_locals = carried_locals.items;
            if (has_return) {
                then_lowerer.deferred_return_kind = .scf_yield;
                then_lowerer.deferred_return_carried_locals = carried_locals.items;
            }
            var then_locals = try self.cloneLocals(locals);
            const then_terminated = try then_lowerer.lowerBody(if_stmt.then_body, &then_locals);
            if (!blockEndsWithTerminator(then_block)) {
                try then_lowerer.appendScfYieldFromLocals(then_block, if_stmt.range, &then_locals, carried_locals.items);
            }

            var else_locals = try self.cloneLocals(locals);
            var else_terminated = false;
            if (if_stmt.else_body) |else_body| {
                var else_lowerer = self.*;
                else_lowerer.block = else_block;
                else_lowerer.current_scf_carried_locals = carried_locals.items;
                if (has_return) {
                    else_lowerer.deferred_return_kind = .scf_yield;
                    else_lowerer.deferred_return_carried_locals = carried_locals.items;
                }
                else_terminated = try else_lowerer.lowerBody(else_body, &else_locals);
            }
            if (!blockEndsWithTerminator(else_block)) {
                try self.appendScfYieldFromLocals(else_block, if_stmt.range, &else_locals, carried_locals.items);
            }

            try FunctionLowerer.writeBackCarriedLocals(locals, carried_locals.items, op);
            if (created_deferred_return) {
                try self.appendDeferredReturnCheck(if_stmt.range);
                if (then_terminated and else_terminated and !blockEndsWithTerminator(self.block)) {
                    try self.appendDeferredReturnTerminator(if_stmt.range, locals);
                }
            }
            return then_terminated and else_terminated;
        }

        pub fn lowerTryStmt(self: *FunctionLowerer, try_stmt: ast.TryStmt, locals: *LocalEnv) anyerror!bool {
            const catch_has_return = if (try_stmt.catch_clause) |catch_clause|
                bodyMayReturn(self.parent.file, catch_clause.body)
            else
                false;
            const has_return = bodyMayReturn(self.parent.file, try_stmt.try_body) or catch_has_return;
            const loc = self.parent.location(try_stmt.range);
            const created_deferred_return = has_return and self.deferred_return_flag == null;
            if (created_deferred_return) {
                try self.ensureDeferredReturnSlots(try_stmt.range);
            }
            var carried_locals: LocalIdList = .{};
            var carried_seen = LocalIdSet.init(self.parent.allocator);
            if (!try collectTryCarriedLocals(self.parent.allocator, self.parent.file, try_stmt, locals, &carried_locals, &carried_seen)) {
                try self.appendUnsupportedControlPlaceholder("ora.try_placeholder", try_stmt.range);
                return false;
            }
            carried_locals = try self.filterCarriedLocals(locals, carried_locals.items);

            const result_types = if (carried_locals.items.len == 0)
                null
            else
                (try self.buildCarriedResultTypes(locals, carried_locals.items)) orelse {
                    try self.appendUnsupportedControlPlaceholder("ora.try_placeholder", try_stmt.range);
                    return false;
                };

            const op = mlir.oraTryStmtOpCreate(
                self.parent.context,
                loc,
                if (result_types) |types| types.items.ptr else null,
                if (result_types) |types| types.items.len else 0,
            );
            if (mlir.oraOperationIsNull(op)) return error.MlirOperationCreationFailed;
            appendOp(self.block, op);

            const try_block = mlir.oraTryStmtOpGetTryBlock(op);
            const catch_block = mlir.oraTryStmtOpGetCatchBlock(op);
            if (mlir.oraBlockIsNull(try_block) or mlir.oraBlockIsNull(catch_block)) {
                return error.MlirOperationCreationFailed;
            }

            var try_lowerer = self.*;
            try_lowerer.block = try_block;
            try_lowerer.in_try_block = true;
            if (has_return) {
                try_lowerer.deferred_return_kind = .ora_yield;
                try_lowerer.deferred_return_carried_locals = carried_locals.items;
            }
            var try_locals = try self.cloneLocals(locals);
            const try_terminated = try try_lowerer.lowerBody(try_stmt.try_body, &try_locals);
            if (!blockEndsWithTerminator(try_block)) {
                try try_lowerer.appendOraYieldFromLocals(try_block, try_stmt.range, &try_locals, carried_locals.items);
            }

            var catch_locals = try self.cloneLocals(locals);
            var catch_terminated = false;
            if (try_stmt.catch_clause) |catch_clause| {
                var catch_lowerer = self.*;
                catch_lowerer.block = catch_block;
                catch_lowerer.in_try_block = true;
                if (has_return) {
                    catch_lowerer.deferred_return_kind = .ora_yield;
                    catch_lowerer.deferred_return_carried_locals = carried_locals.items;
                }
                if (catch_clause.error_pattern) |pattern_id| {
                    const catch_type = self.parent.typecheck.pattern_types[pattern_id.index()].type;
                    const lowered_type = if (catch_type.kind() == .unknown)
                        defaultIntegerType(self.parent.context)
                    else
                        self.parent.lowerSemaType(catch_type, catch_clause.range);
                    const error_arg = mlir.mlirBlockAddArgument(catch_block, lowered_type, self.parent.location(catch_clause.range));
                    try catch_lowerer.bindPatternValue(pattern_id, error_arg, &catch_locals);
                }
                catch_terminated = try catch_lowerer.lowerBody(catch_clause.body, &catch_locals);
                if (!blockEndsWithTerminator(catch_block)) {
                    try catch_lowerer.appendOraYieldFromLocals(catch_block, catch_clause.range, &catch_locals, carried_locals.items);
                }
            } else if (!blockEndsWithTerminator(catch_block)) {
                try self.appendOraYieldFromLocals(catch_block, try_stmt.range, &catch_locals, carried_locals.items);
            }

            if (carried_locals.items.len > 0) {
                try FunctionLowerer.writeBackCarriedLocals(locals, carried_locals.items, op);
            }
            if (created_deferred_return) {
                try self.appendDeferredReturnCheck(try_stmt.range);
                if (try_terminated and catch_terminated and !blockEndsWithTerminator(self.block)) {
                    try self.appendDeferredReturnTerminator(try_stmt.range, locals);
                }
            }
            return try_terminated and catch_terminated;
        }

        pub fn lowerLabeledBlockStmt(self: *FunctionLowerer, block_stmt: ast.LabeledBlockStmt, locals: *LocalEnv) anyerror!bool {
            const loc = self.parent.location(block_stmt.range);
            const has_return = bodyMayReturn(self.parent.file, block_stmt.body);
            const created_deferred_return = has_return and self.deferred_return_flag == null;
            if (created_deferred_return) {
                try self.ensureDeferredReturnSlots(block_stmt.range);
            }

            const continue_flag_alloc = mlir.oraMemrefAllocaOpCreate(self.parent.context, loc, memRefType(self.parent.context, boolType(self.parent.context)));
            if (mlir.oraOperationIsNull(continue_flag_alloc)) return error.MlirOperationCreationFailed;
            const continue_flag = appendValueOp(self.block, continue_flag_alloc);
            const continue_true = appendValueOp(self.block, createIntegerConstant(self.parent.context, loc, boolType(self.parent.context), 1));
            const init_continue = mlir.oraMemrefStoreOpCreate(self.parent.context, loc, continue_true, continue_flag, null, 0);
            if (mlir.oraOperationIsNull(init_continue)) return error.MlirOperationCreationFailed;
            appendOp(self.block, init_continue);

            var carried_locals: LocalIdList = .{};
            var carried_seen = LocalIdSet.init(self.parent.allocator);
            if (!try collectLoopCarriedLocals(self.parent.allocator, self.parent.file, block_stmt.body, locals, &carried_locals, &carried_seen)) {
                try self.appendUnsupportedControlPlaceholder("ora.labeled_block_placeholder", block_stmt.range);
                return false;
            }
            carried_locals = try self.filterCarriedLocals(locals, carried_locals.items);

            var init_operands: std.ArrayList(mlir.MlirValue) = .{};
            for (carried_locals.items) |local_id| {
                const value = try self.materializeCarriedLocalValue(locals, local_id);
                try init_operands.append(self.parent.allocator, value);
            }
            const result_types = (try self.buildCarriedResultTypes(locals, carried_locals.items)) orelse {
                try self.appendUnsupportedControlPlaceholder("ora.labeled_block_placeholder", block_stmt.range);
                return false;
            };

            const while_op = mlir.oraScfWhileOpCreate(
                self.parent.context,
                loc,
                if (init_operands.items.len == 0) null else init_operands.items.ptr,
                init_operands.items.len,
                if (result_types.items.len == 0) null else result_types.items.ptr,
                result_types.items.len,
            );
            if (mlir.oraOperationIsNull(while_op)) return error.MlirOperationCreationFailed;
            appendOp(self.block, while_op);

            const before_block = mlir.oraScfWhileOpGetBeforeBlock(while_op);
            const after_block = mlir.oraScfWhileOpGetAfterBlock(while_op);
            if (mlir.oraBlockIsNull(before_block) or mlir.oraBlockIsNull(after_block)) {
                return error.MlirOperationCreationFailed;
            }
            if (carried_locals.items.len > 0) {
                var before_args = mlir.oraBlockGetNumArguments(before_block);
                while (before_args < carried_locals.items.len) : (before_args += 1) {
                    _ = mlir.mlirBlockAddArgument(before_block, result_types.items[before_args], loc);
                }
                var after_args = mlir.oraBlockGetNumArguments(after_block);
                while (after_args < carried_locals.items.len) : (after_args += 1) {
                    _ = mlir.mlirBlockAddArgument(after_block, result_types.items[after_args], loc);
                }
            }

            var before_lowerer = self.*;
            before_lowerer.block = before_block;
            var before_locals = try self.cloneLocals(locals);
            for (carried_locals.items, 0..) |local_id, index| {
                try before_locals.setValue(local_id, mlir.oraBlockGetArgument(before_block, index));
            }
            var continue_value = appendValueOp(before_block, blk: {
                const load = mlir.oraMemrefLoadOpCreate(self.parent.context, loc, continue_flag, null, 0, boolType(self.parent.context));
                if (mlir.oraOperationIsNull(load)) return error.MlirOperationCreationFailed;
                break :blk load;
            });
            if (has_return) {
                const return_flag = self.deferred_return_flag.?;
                const return_flag_value = appendValueOp(before_block, blk: {
                    const load = mlir.oraMemrefLoadOpCreate(self.parent.context, loc, return_flag, null, 0, boolType(self.parent.context));
                    if (mlir.oraOperationIsNull(load)) return error.MlirOperationCreationFailed;
                    break :blk load;
                });
                const return_flag_clear = appendValueOp(before_block, createIntegerConstant(self.parent.context, loc, boolType(self.parent.context), 0));
                const return_enabled = before_lowerer.createCompareOp(loc, "eq", return_flag_value, return_flag_clear);
                if (mlir.oraOperationIsNull(return_enabled)) return error.MlirOperationCreationFailed;
                continue_value = appendValueOp(before_block, mlir.oraArithAndIOpCreate(self.parent.context, loc, continue_value, appendValueOp(before_block, return_enabled)));
            }
            var condition_values: std.ArrayList(mlir.MlirValue) = .{};
            for (carried_locals.items) |local_id| {
                const value = try before_lowerer.materializeCarriedLocalValue(&before_locals, local_id);
                try condition_values.append(self.parent.allocator, value);
            }
            const condition_op = mlir.oraScfConditionOpCreate(
                self.parent.context,
                loc,
                continue_value,
                if (condition_values.items.len == 0) null else condition_values.items.ptr,
                condition_values.items.len,
            );
            if (mlir.oraOperationIsNull(condition_op)) return error.MlirOperationCreationFailed;
            appendOp(before_block, condition_op);

            var body_lowerer = self.*;
            body_lowerer.block = after_block;
            body_lowerer.current_scf_carried_locals = carried_locals.items;
            if (has_return) {
                body_lowerer.deferred_return_kind = .scf_yield;
                body_lowerer.deferred_return_carried_locals = carried_locals.items;
            }
            var block_context = BlockContext{
                .parent = self.block_context,
                .label = block_stmt.label,
                .continue_flag = continue_flag,
                .carried_locals = carried_locals.items,
            };
            body_lowerer.block_context = &block_context;
            var body_locals = try self.cloneLocals(locals);
            for (carried_locals.items, 0..) |local_id, index| {
                try body_locals.setValue(local_id, mlir.oraBlockGetArgument(after_block, index));
            }
            const clear_continue = appendValueOp(after_block, createIntegerConstant(self.parent.context, loc, boolType(self.parent.context), 0));
            const clear_continue_store = mlir.oraMemrefStoreOpCreate(self.parent.context, loc, clear_continue, continue_flag, null, 0);
            if (mlir.oraOperationIsNull(clear_continue_store)) return error.MlirOperationCreationFailed;
            appendOp(after_block, clear_continue_store);
            const terminated = try body_lowerer.lowerBody(block_stmt.body, &body_locals);
            if (!blockEndsWithTerminator(after_block)) {
                try body_lowerer.appendScfYieldFromLocals(after_block, block_stmt.range, &body_locals, carried_locals.items);
            }

            if (carried_locals.items.len > 0) {
                try FunctionLowerer.writeBackCarriedLocals(locals, carried_locals.items, while_op);
            }
            if (created_deferred_return) {
                try self.appendDeferredReturnCheck(block_stmt.range);
                if (terminated and !blockEndsWithTerminator(self.block)) {
                    try self.appendDeferredReturnTerminator(block_stmt.range, locals);
                }
            }
            return false;
        }

        pub fn lowerWhileStmt(self: *FunctionLowerer, while_stmt: ast.WhileStmt, locals: *LocalEnv) anyerror!bool {
            const has_return = bodyMayReturn(self.parent.file, while_stmt.body);
            if (bodyContainsStructuredLoopControl(self.parent.file, while_stmt.body)) {
                try self.appendUnsupportedControlPlaceholder("ora.while_placeholder", while_stmt.range);
                return false;
            }

            const loc = self.parent.location(while_stmt.range);
            const created_deferred_return = has_return and self.deferred_return_flag == null;
            if (created_deferred_return) {
                try self.ensureDeferredReturnSlots(while_stmt.range);
            }
            const break_flag_alloc = mlir.oraMemrefAllocaOpCreate(self.parent.context, loc, memRefType(self.parent.context, boolType(self.parent.context)));
            if (mlir.oraOperationIsNull(break_flag_alloc)) return error.MlirOperationCreationFailed;
            const break_flag = appendValueOp(self.block, break_flag_alloc);

            const break_flag_zero = appendValueOp(self.block, createIntegerConstant(self.parent.context, loc, boolType(self.parent.context), 0));
            const clear_break = mlir.oraMemrefStoreOpCreate(self.parent.context, loc, break_flag_zero, break_flag, null, 0);
            if (mlir.oraOperationIsNull(clear_break)) return error.MlirOperationCreationFailed;
            appendOp(self.block, clear_break);

            const continue_flag_alloc = mlir.oraMemrefAllocaOpCreate(self.parent.context, loc, memRefType(self.parent.context, boolType(self.parent.context)));
            if (mlir.oraOperationIsNull(continue_flag_alloc)) return error.MlirOperationCreationFailed;
            const continue_flag = appendValueOp(self.block, continue_flag_alloc);
            const continue_flag_zero = appendValueOp(self.block, createIntegerConstant(self.parent.context, loc, boolType(self.parent.context), 0));
            const clear_continue = mlir.oraMemrefStoreOpCreate(self.parent.context, loc, continue_flag_zero, continue_flag, null, 0);
            if (mlir.oraOperationIsNull(clear_continue)) return error.MlirOperationCreationFailed;
            appendOp(self.block, clear_continue);

            var carried_locals: LocalIdList = .{};
            var carried_seen = LocalIdSet.init(self.parent.allocator);
            const carried_supported = try collectLoopCarriedLocals(self.parent.allocator, self.parent.file, while_stmt.body, locals, &carried_locals, &carried_seen);
            if (!carried_supported) {
                try self.appendUnsupportedControlPlaceholder("ora.while_placeholder", while_stmt.range);
                return false;
            }
            carried_locals = try self.filterCarriedLocals(locals, carried_locals.items);

            var init_operands: std.ArrayList(mlir.MlirValue) = .{};
            for (carried_locals.items) |local_id| {
                const value = try self.materializeCarriedLocalValue(locals, local_id);
                try init_operands.append(self.parent.allocator, value);
            }
            const result_types = (try self.buildCarriedResultTypes(locals, carried_locals.items)) orelse {
                try self.appendUnsupportedControlPlaceholder("ora.while_placeholder", while_stmt.range);
                return false;
            };

            const while_op = mlir.oraScfWhileOpCreate(
                self.parent.context,
                loc,
                if (init_operands.items.len == 0) null else init_operands.items.ptr,
                init_operands.items.len,
                if (result_types.items.len == 0) null else result_types.items.ptr,
                result_types.items.len,
            );
            if (mlir.oraOperationIsNull(while_op)) return error.MlirOperationCreationFailed;
            appendOp(self.block, while_op);

            const before_block = mlir.oraScfWhileOpGetBeforeBlock(while_op);
            const after_block = mlir.oraScfWhileOpGetAfterBlock(while_op);
            if (mlir.oraBlockIsNull(before_block) or mlir.oraBlockIsNull(after_block)) {
                return error.MlirOperationCreationFailed;
            }
            if (carried_locals.items.len > 0) {
                var before_args = mlir.oraBlockGetNumArguments(before_block);
                while (before_args < carried_locals.items.len) : (before_args += 1) {
                    _ = mlir.mlirBlockAddArgument(before_block, result_types.items[before_args], loc);
                }

                var after_args = mlir.oraBlockGetNumArguments(after_block);
                while (after_args < carried_locals.items.len) : (after_args += 1) {
                    _ = mlir.mlirBlockAddArgument(after_block, result_types.items[after_args], loc);
                }
            }

            var before_lowerer = self.*;
            before_lowerer.block = before_block;
            var before_locals = try self.cloneLocals(locals);
            for (carried_locals.items, 0..) |local_id, index| {
                try before_locals.setValue(local_id, mlir.oraBlockGetArgument(before_block, index));
            }

            var condition = try before_lowerer.lowerExpr(while_stmt.condition, &before_locals);
            if (!mlir.oraTypeEqual(mlir.oraValueGetType(condition), boolType(self.parent.context))) {
                const zero = try before_lowerer.defaultValue(mlir.oraValueGetType(condition), while_stmt.range);
                const cmp = before_lowerer.createCompareOp(loc, "ne", condition, zero);
                if (mlir.oraOperationIsNull(cmp)) return error.MlirOperationCreationFailed;
                condition = appendValueOp(before_block, cmp);
            }

            const break_flag_value = appendValueOp(before_block, blk: {
                const load = mlir.oraMemrefLoadOpCreate(self.parent.context, loc, break_flag, null, 0, boolType(self.parent.context));
                if (mlir.oraOperationIsNull(load)) return error.MlirOperationCreationFailed;
                break :blk load;
            });
            const break_flag_clear = appendValueOp(before_block, createIntegerConstant(self.parent.context, loc, boolType(self.parent.context), 0));
            const loop_enabled = before_lowerer.createCompareOp(loc, "eq", break_flag_value, break_flag_clear);
            if (mlir.oraOperationIsNull(loop_enabled)) return error.MlirOperationCreationFailed;
            condition = appendValueOp(before_block, mlir.oraArithAndIOpCreate(self.parent.context, loc, appendValueOp(before_block, loop_enabled), condition));
            if (has_return) {
                const return_flag = self.deferred_return_flag.?;
                const return_flag_value = appendValueOp(before_block, blk: {
                    const load = mlir.oraMemrefLoadOpCreate(self.parent.context, loc, return_flag, null, 0, boolType(self.parent.context));
                    if (mlir.oraOperationIsNull(load)) return error.MlirOperationCreationFailed;
                    break :blk load;
                });
                const return_flag_clear = appendValueOp(before_block, createIntegerConstant(self.parent.context, loc, boolType(self.parent.context), 0));
                const return_enabled = before_lowerer.createCompareOp(loc, "eq", return_flag_value, return_flag_clear);
                if (mlir.oraOperationIsNull(return_enabled)) return error.MlirOperationCreationFailed;
                condition = appendValueOp(before_block, mlir.oraArithAndIOpCreate(self.parent.context, loc, appendValueOp(before_block, return_enabled), condition));
            }

            var condition_values: std.ArrayList(mlir.MlirValue) = .{};
            for (carried_locals.items) |local_id| {
                const value = try before_lowerer.materializeCarriedLocalValue(&before_locals, local_id);
                try condition_values.append(self.parent.allocator, value);
            }

            const condition_op = mlir.oraScfConditionOpCreate(
                self.parent.context,
                loc,
                condition,
                if (condition_values.items.len == 0) null else condition_values.items.ptr,
                condition_values.items.len,
            );
            if (mlir.oraOperationIsNull(condition_op)) return error.MlirOperationCreationFailed;
            appendOp(before_block, condition_op);

            var body_lowerer = self.*;
            body_lowerer.block = after_block;
            if (has_return) {
                body_lowerer.deferred_return_kind = .scf_yield;
                body_lowerer.deferred_return_carried_locals = carried_locals.items;
            }
            var loop_context = LoopContext{
                .parent = self.loop_context,
                .label = while_stmt.label,
                .break_flag = break_flag,
                .continue_flag = continue_flag,
                .carried_locals = carried_locals.items,
            };
            body_lowerer.loop_context = &loop_context;
            var body_locals = try self.cloneLocals(locals);
            for (carried_locals.items, 0..) |local_id, index| {
                try body_locals.setValue(local_id, mlir.oraBlockGetArgument(after_block, index));
            }
            const clear_continue_body = appendValueOp(after_block, createIntegerConstant(self.parent.context, loc, boolType(self.parent.context), 0));
            const clear_continue_store = mlir.oraMemrefStoreOpCreate(self.parent.context, loc, clear_continue_body, continue_flag, null, 0);
            if (mlir.oraOperationIsNull(clear_continue_store)) return error.MlirOperationCreationFailed;
            appendOp(after_block, clear_continue_store);
            for (while_stmt.invariants) |expr_id| {
                const value = try body_lowerer.lowerExpr(expr_id, &body_locals);
                const op = mlir.oraInvariantOpCreate(
                    self.parent.context,
                    self.parent.location(exprRange(self.parent.file, expr_id)),
                    value,
                );
                if (mlir.oraOperationIsNull(op)) return error.MlirOperationCreationFailed;
                appendOp(after_block, op);
            }
            _ = try body_lowerer.lowerBody(while_stmt.body, &body_locals);
            if (!blockEndsWithTerminator(after_block)) {
                try body_lowerer.appendScfYieldFromLocals(after_block, while_stmt.range, &body_locals, carried_locals.items);
            }

            try FunctionLowerer.writeBackCarriedLocals(locals, carried_locals.items, while_op);
            if (created_deferred_return) {
                try self.appendDeferredReturnCheck(while_stmt.range);
            }
            return false;
        }

        pub fn lowerSwitchStmt(self: *FunctionLowerer, switch_stmt: ast.SwitchStmt, locals: *LocalEnv) anyerror!bool {
            if (switch_stmt.label != null) {
                return @This().lowerLabeledSwitchStmt(self, switch_stmt, locals);
            }
            return @This().lowerSwitchStmtWithCondition(self, switch_stmt, try self.lowerExpr(switch_stmt.condition, locals), locals);
        }

        fn lowerSwitchStmtWithCondition(
            self: *FunctionLowerer,
            switch_stmt: ast.SwitchStmt,
            condition: mlir.MlirValue,
            locals: *LocalEnv,
        ) anyerror!bool {
            const has_return = switchMayReturn(self.parent.file, switch_stmt);
            const created_deferred_return = has_return and self.deferred_return_flag == null;
            if (created_deferred_return) {
                try self.ensureDeferredReturnSlots(switch_stmt.range);
            }
            var carried_locals: LocalIdList = .{};
            var carried_seen = LocalIdSet.init(self.parent.allocator);
            if (!try collectSwitchCarriedLocals(self.parent.allocator, self.parent.file, switch_stmt, locals, &carried_locals, &carried_seen)) {
                try self.appendUnsupportedControlPlaceholder("ora.switch_placeholder", switch_stmt.range);
                return false;
            }
            carried_locals = try self.filterCarriedLocals(locals, carried_locals.items);

            const pattern_data = (try self.buildSwitchPatternData(switch_stmt, carried_locals.items.len > 0 and switch_stmt.else_body == null)) orelse {
                try self.appendUnsupportedControlPlaceholder("ora.switch_placeholder", switch_stmt.range);
                return false;
            };

            const result_types = if (carried_locals.items.len == 0)
                null
            else
                (try self.buildCarriedResultTypes(locals, carried_locals.items)) orelse {
                    try self.appendUnsupportedControlPlaceholder("ora.switch_placeholder", switch_stmt.range);
                    return false;
                };

            const op = mlir.oraSwitchOpCreateWithCases(
                self.parent.context,
                self.parent.location(switch_stmt.range),
                condition,
                if (result_types) |types| types.items.ptr else null,
                if (result_types) |types| types.items.len else 0,
                pattern_data.total_cases,
            );
            if (mlir.oraOperationIsNull(op)) return error.MlirOperationCreationFailed;

            var all_cases_terminate = @This().switchIsExhaustive(self, switch_stmt);
            for (switch_stmt.arms, 0..) |arm, case_index| {
                const arm_terminated = try self.lowerSwitchCaseBlock(op, case_index, arm.body, arm.range, locals, carried_locals.items, has_return);
                all_cases_terminate = all_cases_terminate and arm_terminated;
            }

            if (switch_stmt.else_body) |else_body| {
                const else_terminated = try self.lowerSwitchCaseBlock(op, switch_stmt.arms.len, else_body, switch_stmt.range, locals, carried_locals.items, has_return);
                all_cases_terminate = all_cases_terminate and else_terminated;
            } else if (carried_locals.items.len > 0) {
                _ = try self.lowerSwitchCaseBlock(op, switch_stmt.arms.len, null, switch_stmt.range, locals, carried_locals.items, has_return);
            }

            mlir.oraSwitchOpSetCasePatterns(
                op,
                pattern_data.case_values.items.ptr,
                pattern_data.range_starts.items.ptr,
                pattern_data.range_ends.items.ptr,
                pattern_data.case_kinds.items.ptr,
                pattern_data.default_case_index,
                pattern_data.case_values.items.len,
            );
            appendOp(self.block, op);
            if (carried_locals.items.len > 0) {
                try FunctionLowerer.writeBackCarriedLocals(locals, carried_locals.items, op);
            }
            if (created_deferred_return) {
                try self.appendDeferredReturnCheck(switch_stmt.range);
                if (all_cases_terminate and !blockEndsWithTerminator(self.block)) {
                    try self.appendDeferredReturnTerminator(switch_stmt.range, locals);
                }
            }
            return all_cases_terminate;
        }

        fn switchIsExhaustive(self: *FunctionLowerer, switch_stmt: ast.SwitchStmt) bool {
            if (switch_stmt.else_body != null) return true;

            const condition_type = self.parent.typecheck.exprType(switch_stmt.condition);
            if (condition_type.kind() == .bool) {
                var seen_true = false;
                var seen_false = false;
                for (switch_stmt.arms) |arm| {
                    switch (arm.pattern) {
                        .Expr => |pattern_expr| {
                            const value = self.switchPatternValue(pattern_expr) orelse continue;
                            if (value == 0) seen_false = true;
                            if (value == 1) seen_true = true;
                        },
                        else => return false,
                    }
                }
                return seen_true and seen_false;
            }

            const enum_name = condition_type.name() orelse return false;
            const item_id = self.parent.item_index.lookup(enum_name) orelse return false;
            const enum_item = switch (self.parent.file.item(item_id).*) {
                .Enum => |item| item,
                else => return false,
            };

            var seen = std.AutoHashMap(i64, void).init(self.parent.allocator);
            defer seen.deinit();

            for (switch_stmt.arms) |arm| {
                switch (arm.pattern) {
                    .Expr => |pattern_expr| {
                        const value = self.switchPatternValue(pattern_expr) orelse return false;
                        seen.put(value, {}) catch return false;
                    },
                    else => return false,
                }
            }

            return seen.count() == enum_item.variants.len;
        }

        fn lowerLabeledSwitchStmt(self: *FunctionLowerer, switch_stmt: ast.SwitchStmt, locals: *LocalEnv) anyerror!bool {
            const loc = self.parent.location(switch_stmt.range);
            const has_return = switchMayReturn(self.parent.file, switch_stmt);
            const created_deferred_return = has_return and self.deferred_return_flag == null;
            if (created_deferred_return) {
                try self.ensureDeferredReturnSlots(switch_stmt.range);
            }

            const initial_condition = try self.lowerExpr(switch_stmt.condition, locals);
            const condition_type = mlir.oraValueGetType(initial_condition);

            const continue_flag_alloc = mlir.oraMemrefAllocaOpCreate(self.parent.context, loc, memRefType(self.parent.context, boolType(self.parent.context)));
            if (mlir.oraOperationIsNull(continue_flag_alloc)) return error.MlirOperationCreationFailed;
            const continue_flag = appendValueOp(self.block, continue_flag_alloc);
            const continue_true = appendValueOp(self.block, createIntegerConstant(self.parent.context, loc, boolType(self.parent.context), 1));
            const init_continue = mlir.oraMemrefStoreOpCreate(self.parent.context, loc, continue_true, continue_flag, null, 0);
            if (mlir.oraOperationIsNull(init_continue)) return error.MlirOperationCreationFailed;
            appendOp(self.block, init_continue);

            const value_slot_alloc = mlir.oraMemrefAllocaOpCreate(self.parent.context, loc, memRefType(self.parent.context, condition_type));
            if (mlir.oraOperationIsNull(value_slot_alloc)) return error.MlirOperationCreationFailed;
            const value_slot = appendValueOp(self.block, value_slot_alloc);
            const init_value = mlir.oraMemrefStoreOpCreate(self.parent.context, loc, initial_condition, value_slot, null, 0);
            if (mlir.oraOperationIsNull(init_value)) return error.MlirOperationCreationFailed;
            appendOp(self.block, init_value);

            var carried_locals: LocalIdList = .{};
            var carried_seen = LocalIdSet.init(self.parent.allocator);
            if (!try collectSwitchCarriedLocals(self.parent.allocator, self.parent.file, switch_stmt, locals, &carried_locals, &carried_seen)) {
                try self.appendUnsupportedControlPlaceholder("ora.switch_placeholder", switch_stmt.range);
                return false;
            }
            carried_locals = try self.filterCarriedLocals(locals, carried_locals.items);

            var init_operands: std.ArrayList(mlir.MlirValue) = .{};
            for (carried_locals.items) |local_id| {
                const value = try self.materializeCarriedLocalValue(locals, local_id);
                try init_operands.append(self.parent.allocator, value);
            }
            const result_types = if (carried_locals.items.len == 0)
                null
            else
                (try self.buildCarriedResultTypes(locals, carried_locals.items)) orelse {
                    try self.appendUnsupportedControlPlaceholder("ora.switch_placeholder", switch_stmt.range);
                    return false;
                };

            const while_op = mlir.oraScfWhileOpCreate(
                self.parent.context,
                loc,
                if (init_operands.items.len == 0) null else init_operands.items.ptr,
                init_operands.items.len,
                if (result_types) |types| types.items.ptr else null,
                if (result_types) |types| types.items.len else 0,
            );
            if (mlir.oraOperationIsNull(while_op)) return error.MlirOperationCreationFailed;
            appendOp(self.block, while_op);

            const before_block = mlir.oraScfWhileOpGetBeforeBlock(while_op);
            const after_block = mlir.oraScfWhileOpGetAfterBlock(while_op);
            if (mlir.oraBlockIsNull(before_block) or mlir.oraBlockIsNull(after_block)) {
                return error.MlirOperationCreationFailed;
            }
            if (carried_locals.items.len > 0) {
                var before_args = mlir.oraBlockGetNumArguments(before_block);
                while (before_args < carried_locals.items.len) : (before_args += 1) {
                    _ = mlir.mlirBlockAddArgument(before_block, result_types.?.items[before_args], loc);
                }

                var after_args = mlir.oraBlockGetNumArguments(after_block);
                while (after_args < carried_locals.items.len) : (after_args += 1) {
                    _ = mlir.mlirBlockAddArgument(after_block, result_types.?.items[after_args], loc);
                }
            }

            var before_lowerer = self.*;
            before_lowerer.block = before_block;
            var before_locals = try self.cloneLocals(locals);
            for (carried_locals.items, 0..) |local_id, index| {
                try before_locals.setValue(local_id, mlir.oraBlockGetArgument(before_block, index));
            }
            const continue_value = appendValueOp(before_block, blk: {
                const load = mlir.oraMemrefLoadOpCreate(self.parent.context, loc, continue_flag, null, 0, boolType(self.parent.context));
                if (mlir.oraOperationIsNull(load)) return error.MlirOperationCreationFailed;
                break :blk load;
            });
            var condition_values: std.ArrayList(mlir.MlirValue) = .{};
            for (carried_locals.items) |local_id| {
                const value = try before_lowerer.materializeCarriedLocalValue(&before_locals, local_id);
                try condition_values.append(self.parent.allocator, value);
            }
            const condition_op = mlir.oraScfConditionOpCreate(
                self.parent.context,
                loc,
                continue_value,
                if (condition_values.items.len == 0) null else condition_values.items.ptr,
                condition_values.items.len,
            );
            if (mlir.oraOperationIsNull(condition_op)) return error.MlirOperationCreationFailed;
            appendOp(before_block, condition_op);

            var body_lowerer = self.*;
            body_lowerer.block = after_block;
            if (has_return) {
                body_lowerer.deferred_return_kind = .scf_yield;
                body_lowerer.deferred_return_carried_locals = carried_locals.items;
            }
            var labeled_switch_context = SwitchContext{
                .parent = self.switch_context,
                .label = switch_stmt.label,
                .continue_flag = continue_flag,
                .value_slot = value_slot,
                .value_type = condition_type,
                .carried_locals = carried_locals.items,
            };
            body_lowerer.switch_context = &labeled_switch_context;
            var body_locals = try self.cloneLocals(locals);
            for (carried_locals.items, 0..) |local_id, index| {
                try body_locals.setValue(local_id, mlir.oraBlockGetArgument(after_block, index));
            }
            const clear_continue = appendValueOp(after_block, createIntegerConstant(self.parent.context, loc, boolType(self.parent.context), 0));
            const clear_continue_store = mlir.oraMemrefStoreOpCreate(self.parent.context, loc, clear_continue, continue_flag, null, 0);
            if (mlir.oraOperationIsNull(clear_continue_store)) return error.MlirOperationCreationFailed;
            appendOp(after_block, clear_continue_store);
            const current_condition = appendValueOp(after_block, blk: {
                const load = mlir.oraMemrefLoadOpCreate(self.parent.context, loc, value_slot, null, 0, condition_type);
                if (mlir.oraOperationIsNull(load)) return error.MlirOperationCreationFailed;
                break :blk load;
            });
            var unlabeled_switch = switch_stmt;
            unlabeled_switch.label = null;
            _ = try @This().lowerSwitchStmtWithCondition(&body_lowerer, unlabeled_switch, current_condition, &body_locals);
            if (!blockEndsWithTerminator(after_block)) {
                try body_lowerer.appendScfYieldFromLocals(after_block, switch_stmt.range, &body_locals, carried_locals.items);
            }

            if (carried_locals.items.len > 0) {
                try FunctionLowerer.writeBackCarriedLocals(locals, carried_locals.items, while_op);
            }
            if (created_deferred_return) {
                try self.appendDeferredReturnCheck(switch_stmt.range);
            }
            return false;
        }

        pub fn buildSwitchPatternData(self: *FunctionLowerer, switch_stmt: ast.SwitchStmt, synthesize_default_case: bool) anyerror!?SwitchPatternData {
            const total_cases: usize = switch_stmt.arms.len + (if (switch_stmt.else_body != null or synthesize_default_case) @as(usize, 1) else 0);
            if (total_cases == 0) return null;

            var data = SwitchPatternData{
                .case_values = .{},
                .range_starts = .{},
                .range_ends = .{},
                .case_kinds = .{},
                .total_cases = total_cases,
            };

            for (switch_stmt.arms) |arm| {
                if (!try self.appendSwitchPatternData(arm.pattern, &data)) return null;
            }

            if (switch_stmt.else_body != null or synthesize_default_case) {
                data.default_case_index = @intCast(data.case_values.items.len);
                try data.case_values.append(self.parent.allocator, 0);
                try data.range_starts.append(self.parent.allocator, 0);
                try data.range_ends.append(self.parent.allocator, 0);
                try data.case_kinds.append(self.parent.allocator, 2);
            }
            return data;
        }

        pub fn buildSwitchExprPatternData(self: *FunctionLowerer, switch_expr: ast.SwitchExpr) anyerror!?SwitchPatternData {
            const total_cases: usize = switch_expr.arms.len + (if (switch_expr.else_expr != null) @as(usize, 1) else 0);
            if (total_cases == 0) return null;

            var data = SwitchPatternData{
                .case_values = .{},
                .range_starts = .{},
                .range_ends = .{},
                .case_kinds = .{},
                .total_cases = total_cases,
            };

            for (switch_expr.arms) |arm| {
                if (!try self.appendSwitchPatternData(arm.pattern, &data)) return null;
            }

            if (switch_expr.else_expr != null) {
                data.default_case_index = @intCast(data.case_values.items.len);
                try data.case_values.append(self.parent.allocator, 0);
                try data.range_starts.append(self.parent.allocator, 0);
                try data.range_ends.append(self.parent.allocator, 0);
                try data.case_kinds.append(self.parent.allocator, 2);
            }
            return data;
        }

        pub fn appendSwitchPatternData(self: *FunctionLowerer, pattern: ast.SwitchPattern, data: *SwitchPatternData) anyerror!bool {
            switch (pattern) {
                .Expr => |pattern_expr| {
                    const value = self.switchPatternValue(pattern_expr) orelse return false;
                    try data.case_values.append(self.parent.allocator, value);
                    try data.range_starts.append(self.parent.allocator, 0);
                    try data.range_ends.append(self.parent.allocator, 0);
                    try data.case_kinds.append(self.parent.allocator, 0);
                },
                .Range => |range_pattern| {
                    const start_value = self.switchPatternValue(range_pattern.start) orelse return false;
                    var end_value = self.switchPatternValue(range_pattern.end) orelse return false;
                    if (!range_pattern.inclusive) {
                        end_value = std.math.sub(i64, end_value, 1) catch return false;
                    }
                    try data.case_values.append(self.parent.allocator, 0);
                    try data.range_starts.append(self.parent.allocator, start_value);
                    try data.range_ends.append(self.parent.allocator, end_value);
                    try data.case_kinds.append(self.parent.allocator, 1);
                },
                .Else => unreachable,
            }
            return true;
        }

        pub fn lowerSwitchCaseBlock(
            self: *FunctionLowerer,
            op: mlir.MlirOperation,
            case_index: usize,
            body_id: ?ast.BodyId,
            range: source.TextRange,
            locals: *const LocalEnv,
            carried_locals: []const LocalId,
            has_return: bool,
        ) anyerror!bool {
            const case_block = mlir.oraSwitchOpGetCaseBlock(op, case_index);
            if (mlir.oraBlockIsNull(case_block)) return error.MlirOperationCreationFailed;

            var case_lowerer = self.*;
            case_lowerer.block = case_block;
            if (has_return) {
                case_lowerer.deferred_return_kind = .ora_yield;
                case_lowerer.deferred_return_carried_locals = carried_locals;
            }
            var switch_context = SwitchContext{
                .parent = self.switch_context,
                .carried_locals = carried_locals,
            };
            case_lowerer.switch_context = &switch_context;
            var case_locals = try self.cloneLocals(locals);
            var terminated = false;
            if (body_id) |body| {
                terminated = try case_lowerer.lowerBody(body, &case_locals);
            }
            if (!blockEndsWithTerminator(case_block)) {
                if (carried_locals.len == 0) {
                    try appendEmptyYield(self.parent.context, case_block, self.parent.location(range));
                } else {
                    try case_lowerer.appendOraYieldFromLocals(case_block, range, &case_locals, carried_locals);
                }
            }
            return terminated;
        }

        pub fn lowerSwitchExpr(self: *FunctionLowerer, expr_id: ast.ExprId, switch_expr: ast.SwitchExpr, locals: *LocalEnv) anyerror!mlir.MlirValue {
            const condition = try self.lowerExpr(switch_expr.condition, locals);
            const pattern_data = (try self.buildSwitchExprPatternData(switch_expr)) orelse {
                const op = try self.createAggregatePlaceholder("ora.switch_expr", switch_expr.range, &.{}, self.parent.lowerExprType(expr_id));
                return appendValueOp(self.block, op);
            };

            const result_type = self.parent.lowerExprType(expr_id);
            const op = mlir.oraSwitchExprOpCreateWithCases(
                self.parent.context,
                self.parent.location(switch_expr.range),
                condition,
                &[_]mlir.MlirType{result_type},
                1,
                pattern_data.total_cases,
            );
            if (mlir.oraOperationIsNull(op)) return error.MlirOperationCreationFailed;

            for (switch_expr.arms, 0..) |arm, case_index| {
                const case_block = mlir.oraSwitchExprOpGetCaseBlock(op, case_index);
                if (mlir.oraBlockIsNull(case_block)) return error.MlirOperationCreationFailed;

                var case_lowerer = self.*;
                case_lowerer.block = case_block;
                const value = try case_lowerer.lowerExpr(arm.value, locals);
                try appendOraYieldValues(self.parent.context, case_block, self.parent.location(arm.range), &[_]mlir.MlirValue{value});
            }

            if (switch_expr.else_expr) |else_expr| {
                const else_block = mlir.oraSwitchExprOpGetCaseBlock(op, switch_expr.arms.len);
                if (mlir.oraBlockIsNull(else_block)) return error.MlirOperationCreationFailed;

                var else_lowerer = self.*;
                else_lowerer.block = else_block;
                const value = try else_lowerer.lowerExpr(else_expr, locals);
                try appendOraYieldValues(self.parent.context, else_block, self.parent.location(switch_expr.range), &[_]mlir.MlirValue{value});
            }

            mlir.oraSwitchOpSetCasePatterns(
                op,
                pattern_data.case_values.items.ptr,
                pattern_data.range_starts.items.ptr,
                pattern_data.range_ends.items.ptr,
                pattern_data.case_kinds.items.ptr,
                pattern_data.default_case_index,
                pattern_data.case_values.items.len,
            );
            return appendValueOp(self.block, op);
        }

        pub fn switchPatternValue(self: *FunctionLowerer, expr_id: ast.ExprId) ?i64 {
            if (self.parent.const_eval.values[expr_id.index()]) |value| {
                return switch (value) {
                    .integer => |integer| integer.toInt(i64) catch null,
                    .boolean => |boolean| if (boolean) 1 else 0,
                    else => @This().enumPatternValue(self, expr_id),
                };
            }
            return @This().enumPatternValue(self, expr_id);
        }

        fn enumPatternValue(self: *FunctionLowerer, expr_id: ast.ExprId) ?i64 {
            const expr = self.parent.file.expression(expr_id).*;
            return switch (expr) {
                .Group => |group| @This().enumPatternValue(self, group.expr),
                .Field => |field| blk: {
                    const base_type = self.parent.typecheck.exprType(field.base);
                    const enum_name = base_type.name() orelse break :blk null;
                    const item_id = self.parent.item_index.lookup(enum_name) orelse break :blk null;
                    if (self.parent.file.item(item_id).* != .Enum) break :blk null;
                    const enum_item = self.parent.file.item(item_id).Enum;
                    for (enum_item.variants, 0..) |variant, index| {
                        if (std.mem.eql(u8, variant.name, field.name)) break :blk @intCast(index);
                    }
                    break :blk null;
                },
                else => null,
            };
        }
    };
}
