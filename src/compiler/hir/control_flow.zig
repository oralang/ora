const std = @import("std");
const mlir = @import("mlir_c_api").c;
const ast = @import("../ast/mod.zig");
const hir_locals = @import("locals.zig");
const source = @import("../source/mod.zig");
const analysis = @import("analysis.zig");
const support = @import("support.zig");

const LoopContext = support.LoopContext;
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
const memRefType = support.memRefType;
const namedBoolAttr = support.namedBoolAttr;
const parseIntLiteral = support.parseIntLiteral;
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
            if (bodyMayReturn(self.parent.file, if_stmt.then_body) or
                (if_stmt.else_body != null and bodyMayReturn(self.parent.file, if_stmt.else_body.?)))
            {
                try self.appendUnsupportedControlPlaceholder("ora.if_placeholder", if_stmt.range);
                return false;
            }

            const condition = try self.lowerExpr(if_stmt.condition, locals);
            const loc = self.parent.location(if_stmt.range);
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

            if (carried_locals.items.len == 0) {
                const op = mlir.oraConditionalReturnOpCreate(self.parent.context, loc, condition);
                if (mlir.oraOperationIsNull(op)) return error.MlirOperationCreationFailed;
                appendOp(self.block, op);

                const then_block = mlir.oraConditionalReturnOpGetThenBlock(op);
                const else_block = mlir.oraConditionalReturnOpGetElseBlock(op);
                if (mlir.oraBlockIsNull(then_block) or mlir.oraBlockIsNull(else_block)) {
                    return error.MlirOperationCreationFailed;
                }

                var then_lowerer = self.*;
                then_lowerer.block = then_block;
                var then_locals = try self.cloneLocals(locals);
                _ = try then_lowerer.lowerBody(if_stmt.then_body, &then_locals);
                if (!blockEndsWithTerminator(then_block)) {
                    try appendEmptyYield(self.parent.context, then_block, loc);
                }

                if (if_stmt.else_body) |else_body| {
                    var else_lowerer = self.*;
                    else_lowerer.block = else_block;
                    var else_locals = try self.cloneLocals(locals);
                    _ = try else_lowerer.lowerBody(else_body, &else_locals);
                    if (!blockEndsWithTerminator(else_block)) {
                        try appendEmptyYield(self.parent.context, else_block, loc);
                    }
                } else {
                    try appendEmptyYield(self.parent.context, else_block, loc);
                }
                return false;
            }

            const result_types = (try self.buildCarriedResultTypes(locals, carried_locals.items)) orelse {
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
            var then_locals = try self.cloneLocals(locals);
            _ = try then_lowerer.lowerBody(if_stmt.then_body, &then_locals);
            if (!blockEndsWithTerminator(then_block)) {
                try then_lowerer.appendScfYieldFromLocals(then_block, if_stmt.range, &then_locals, carried_locals.items);
            }

            var else_locals = try self.cloneLocals(locals);
            if (if_stmt.else_body) |else_body| {
                var else_lowerer = self.*;
                else_lowerer.block = else_block;
                _ = try else_lowerer.lowerBody(else_body, &else_locals);
            }
            if (!blockEndsWithTerminator(else_block)) {
                try self.appendScfYieldFromLocals(else_block, if_stmt.range, &else_locals, carried_locals.items);
            }

            try FunctionLowerer.writeBackCarriedLocals(locals, carried_locals.items, op);
            return false;
        }

        pub fn lowerTryStmt(self: *FunctionLowerer, try_stmt: ast.TryStmt, locals: *LocalEnv) anyerror!bool {
            const catch_has_return = if (try_stmt.catch_clause) |catch_clause|
                bodyMayReturn(self.parent.file, catch_clause.body)
            else
                false;
            if (bodyMayReturn(self.parent.file, try_stmt.try_body) or catch_has_return) {
                try self.appendUnsupportedControlPlaceholder("ora.try_placeholder", try_stmt.range);
                return false;
            }
            const loc = self.parent.location(try_stmt.range);
            var carried_locals: LocalIdList = .{};
            var carried_seen = LocalIdSet.init(self.parent.allocator);
            if (!try collectTryCarriedLocals(self.parent.allocator, self.parent.file, try_stmt, locals, &carried_locals, &carried_seen)) {
                try self.appendUnsupportedControlPlaceholder("ora.try_placeholder", try_stmt.range);
                return false;
            }

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
            var try_locals = try self.cloneLocals(locals);
            _ = try try_lowerer.lowerBody(try_stmt.try_body, &try_locals);
            if (!blockEndsWithTerminator(try_block)) {
                try try_lowerer.appendOraYieldFromLocals(try_block, try_stmt.range, &try_locals, carried_locals.items);
            }

            var catch_locals = try self.cloneLocals(locals);
            if (try_stmt.catch_clause) |catch_clause| {
                var catch_lowerer = self.*;
                catch_lowerer.block = catch_block;
                catch_lowerer.in_try_block = true;
                if (catch_clause.error_pattern) |pattern_id| {
                    const error_arg = mlir.mlirBlockAddArgument(catch_block, defaultIntegerType(self.parent.context), self.parent.location(catch_clause.range));
                    try catch_lowerer.bindPatternValue(pattern_id, error_arg, &catch_locals);
                }
                _ = try catch_lowerer.lowerBody(catch_clause.body, &catch_locals);
                if (!blockEndsWithTerminator(catch_block)) {
                    try catch_lowerer.appendOraYieldFromLocals(catch_block, catch_clause.range, &catch_locals, carried_locals.items);
                }
            } else if (!blockEndsWithTerminator(catch_block)) {
                try self.appendOraYieldFromLocals(catch_block, try_stmt.range, &catch_locals, carried_locals.items);
            }

            if (carried_locals.items.len > 0) {
                try FunctionLowerer.writeBackCarriedLocals(locals, carried_locals.items, op);
            }
            return false;
        }

        pub fn lowerWhileStmt(self: *FunctionLowerer, while_stmt: ast.WhileStmt, locals: *LocalEnv) anyerror!bool {
            if (bodyMayReturn(self.parent.file, while_stmt.body) or
                bodyContainsStructuredLoopControl(self.parent.file, while_stmt.body))
            {
                try self.appendUnsupportedControlPlaceholder("ora.while_placeholder", while_stmt.range);
                return false;
            }

            const loc = self.parent.location(while_stmt.range);
            const break_flag_alloc = mlir.oraMemrefAllocaOpCreate(self.parent.context, loc, memRefType(self.parent.context, boolType(self.parent.context)));
            if (mlir.oraOperationIsNull(break_flag_alloc)) return error.MlirOperationCreationFailed;
            const break_flag = appendValueOp(self.block, break_flag_alloc);

            const break_flag_zero = appendValueOp(self.block, createIntegerConstant(self.parent.context, loc, boolType(self.parent.context), 0));
            const clear_break = mlir.oraMemrefStoreOpCreate(self.parent.context, loc, break_flag_zero, break_flag, null, 0);
            if (mlir.oraOperationIsNull(clear_break)) return error.MlirOperationCreationFailed;
            appendOp(self.block, clear_break);

            var carried_locals: LocalIdList = .{};
            var carried_seen = LocalIdSet.init(self.parent.allocator);
            const carried_supported = try collectLoopCarriedLocals(self.parent.allocator, self.parent.file, while_stmt.body, locals, &carried_locals, &carried_seen);
            if (!carried_supported) {
                try self.appendUnsupportedControlPlaceholder("ora.while_placeholder", while_stmt.range);
                return false;
            }

            var init_operands: std.ArrayList(mlir.MlirValue) = .{};
            for (carried_locals.items) |local_id| {
                const value = locals.getValue(local_id) orelse {
                    try self.appendUnsupportedControlPlaceholder("ora.while_placeholder", while_stmt.range);
                    return false;
                };
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

            var condition_values: std.ArrayList(mlir.MlirValue) = .{};
            for (carried_locals.items) |local_id| {
                const value = before_locals.getValue(local_id) orelse return error.MlirOperationCreationFailed;
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
            var loop_context = LoopContext{
                .parent = self.loop_context,
                .break_flag = break_flag,
                .carried_locals = carried_locals.items,
            };
            body_lowerer.loop_context = &loop_context;
            var body_locals = try self.cloneLocals(locals);
            for (carried_locals.items, 0..) |local_id, index| {
                try body_locals.setValue(local_id, mlir.oraBlockGetArgument(after_block, index));
            }
            _ = try body_lowerer.lowerBody(while_stmt.body, &body_locals);
            if (!blockEndsWithTerminator(after_block)) {
                try body_lowerer.appendScfYieldFromLocals(after_block, while_stmt.range, &body_locals, carried_locals.items);
            }

            try FunctionLowerer.writeBackCarriedLocals(locals, carried_locals.items, while_op);
            return false;
        }

        pub fn lowerSwitchStmt(self: *FunctionLowerer, switch_stmt: ast.SwitchStmt, locals: *LocalEnv) anyerror!bool {
            if (switchMayReturn(self.parent.file, switch_stmt)) {
                try self.appendUnsupportedControlPlaceholder("ora.switch_placeholder", switch_stmt.range);
                return false;
            }

            const condition = try self.lowerExpr(switch_stmt.condition, locals);
            var carried_locals: LocalIdList = .{};
            var carried_seen = LocalIdSet.init(self.parent.allocator);
            if (!try collectSwitchCarriedLocals(self.parent.allocator, self.parent.file, switch_stmt, locals, &carried_locals, &carried_seen)) {
                try self.appendUnsupportedControlPlaceholder("ora.switch_placeholder", switch_stmt.range);
                return false;
            }

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

            for (switch_stmt.arms, 0..) |arm, case_index| {
                try self.lowerSwitchCaseBlock(op, case_index, arm.body, arm.range, locals, carried_locals.items);
            }

            if (switch_stmt.else_body) |else_body| {
                try self.lowerSwitchCaseBlock(op, switch_stmt.arms.len, else_body, switch_stmt.range, locals, carried_locals.items);
            } else if (carried_locals.items.len > 0) {
                try self.lowerSwitchCaseBlock(op, switch_stmt.arms.len, null, switch_stmt.range, locals, carried_locals.items);
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
        ) anyerror!void {
            const case_block = mlir.oraSwitchOpGetCaseBlock(op, case_index);
            if (mlir.oraBlockIsNull(case_block)) return error.MlirOperationCreationFailed;

            var case_lowerer = self.*;
            case_lowerer.block = case_block;
            var switch_context = SwitchContext{ .parent = self.switch_context };
            case_lowerer.switch_context = &switch_context;
            var case_locals = try self.cloneLocals(locals);
            if (body_id) |body| {
                _ = try case_lowerer.lowerBody(body, &case_locals);
            }
            if (!blockEndsWithTerminator(case_block)) {
                if (carried_locals.len == 0) {
                    try appendEmptyYield(self.parent.context, case_block, self.parent.location(range));
                } else {
                    try case_lowerer.appendOraYieldFromLocals(case_block, range, &case_locals, carried_locals);
                }
            }
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
            return switch (self.parent.file.expression(expr_id).*) {
                .IntegerLiteral => |literal| parseIntLiteral(literal.text),
                .BoolLiteral => |literal| if (literal.value) 1 else 0,
                .Group => |group| self.switchPatternValue(group.expr),
                .Unary => |unary| switch (unary.op) {
                    .neg => if (self.switchPatternValue(unary.operand)) |value|
                        std.math.negate(value) catch null
                    else
                        null,
                    else => null,
                },
                .Name => |name| blk: {
                    if (self.parent.resolution.expr_bindings[expr_id.index()]) |binding| {
                        switch (binding) {
                            .item => |item_id| {
                                const item = self.parent.file.item(item_id).*;
                                if (item == .Constant) break :blk self.switchPatternValue(item.Constant.value);
                            },
                            else => {},
                        }
                    }
                    if (self.parent.item_index.lookup(name.name)) |item_id| {
                        const item = self.parent.file.item(item_id).*;
                        if (item == .Constant) break :blk self.switchPatternValue(item.Constant.value);
                    }
                    break :blk null;
                },
                else => null,
            };
        }
    };
}
