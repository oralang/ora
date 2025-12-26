// ============================================================================
// Advanced Expression Lowering
// ============================================================================
// Lowering for advanced expressions: casts, comptime, tuples, switch, etc.

const std = @import("std");
const c = @import("mlir_c_api").c;
const lib = @import("ora_lib");
const constants = @import("../lower.zig");
const h = @import("../helpers.zig");
const TypeMapper = @import("../types.zig").TypeMapper;
const LocationTracker = @import("../locations.zig").LocationTracker;
const OraDialect = @import("../dialect.zig").OraDialect;
const expr_helpers = @import("helpers.zig");
const expr_access = @import("access.zig");
const expr_literals = @import("literals.zig");
const log = @import("log");

/// ExpressionLowerer type (forward declaration)
const ExpressionLowerer = @import("mod.zig").ExpressionLowerer;

/// Lower cast expressions
pub fn lowerCast(
    self: *const ExpressionLowerer,
    cast: *const lib.ast.Expressions.CastExpr,
) c.MlirValue {
    const operand = self.lowerExpression(cast.operand);
    const target_mlir_type = self.type_mapper.toMlirType(cast.target_type);

    switch (cast.cast_type) {
        .Unsafe, .Safe, .Forced => {
            const op = self.ora_dialect.createArithBitcast(operand, target_mlir_type, self.fileLoc(cast.span));
            h.appendOp(self.block, op);
            return h.getResult(op, 0);
        },
    }
}

/// Lower comptime expressions
pub fn lowerComptime(
    self: *const ExpressionLowerer,
    comptime_expr: *const lib.ast.Expressions.ComptimeExpr,
) c.MlirValue {
    const loc = self.fileLoc(comptime_expr.span);
    var result_value: ?c.MlirValue = null;
    var result_type: ?c.MlirType = null;

    for (comptime_expr.block.statements) |*stmt| {
        switch (stmt.*) {
            .Return => |ret| {
                if (ret.value) |*expr| {
                    result_value = self.lowerExpression(expr);
                    result_type = c.oraValueGetType(result_value.?);
                    break;
                }
            },
            .Expr => |expr_node| {
                result_value = self.lowerExpression(&expr_node);
                result_type = c.oraValueGetType(result_value.?);
                break;
            },
            else => {},
        }
    }

    const ty = result_type orelse c.oraIntegerTypeCreate(self.ctx, constants.DEFAULT_INTEGER_BITS);
    const comptime_id = h.identifier(self.ctx, "ora.comptime");
    const comptime_attr = h.boolAttr(self.ctx, 1);

    const attrs = [_]c.MlirNamedAttribute{
        c.oraNamedAttributeGet(comptime_id, comptime_attr),
    };
    const op = self.ora_dialect.createArithConstantWithAttrs(0, ty, &attrs, loc);
    h.appendOp(self.block, op);
    return h.getResult(op, 0);
}

/// Lower old expressions (for verification)
pub fn lowerOld(
    self: *const ExpressionLowerer,
    old: *const lib.ast.Expressions.OldExpr,
) c.MlirValue {
    const expr_value = self.lowerExpression(old.expr);
    const loc = self.fileLoc(old.span);
    const old_op = self.ora_dialect.createOld(expr_value, loc);
    h.appendOp(self.block, old_op);
    return h.getResult(old_op, 0);
}

/// Lower tuple expressions
pub fn lowerTuple(
    self: *const ExpressionLowerer,
    tuple: *const lib.ast.Expressions.TupleExpr,
) c.MlirValue {
    if (tuple.elements.len == 0) {
        return self.createConstant(0, tuple.span);
    }

    var element_values = std.ArrayList(c.MlirValue){};
    defer element_values.deinit(std.heap.page_allocator);

    for (tuple.elements) |element| {
        const value = self.lowerExpression(element);
        element_values.append(std.heap.page_allocator, value) catch {};
    }

    var element_types = std.ArrayList(c.MlirType){};
    defer element_types.deinit(std.heap.page_allocator);

    for (element_values.items) |value| {
        const ty = c.oraValueGetType(value);
        element_types.append(std.heap.page_allocator, ty) catch {};
    }

    const tuple_ty = createTupleType(self, element_types.items);
    const undef_op = c.oraLlvmUndefOpCreate(self.ctx, self.fileLoc(tuple.span), tuple_ty);
    if (c.oraOperationIsNull(undef_op)) {
        @panic("Failed to create llvm.mlir.undef operation");
    }
    h.appendOp(self.block, undef_op);
    var current_tuple = h.getResult(undef_op, 0);

    for (element_values.items, 0..) |element_value, i| {
        const position = [_]i64{@intCast(i)};
        const insert_op = c.oraLlvmInsertValueOpCreate(
            self.ctx,
            self.fileLoc(tuple.span),
            tuple_ty,
            current_tuple,
            element_value,
            &position,
            position.len,
        );
        if (c.oraOperationIsNull(insert_op)) {
            @panic("Failed to create llvm.insertvalue operation");
        }
        h.appendOp(self.block, insert_op);
        current_tuple = h.getResult(insert_op, 0);
    }

    return current_tuple;
}

/// Lower switch expressions
pub fn lowerSwitchExpression(
    self: *const ExpressionLowerer,
    switch_expr: *const lib.ast.Expressions.SwitchExprNode,
) c.MlirValue {
    const loc = self.fileLoc(switch_expr.span);
    const condition_raw = self.lowerExpression(switch_expr.condition);
    const condition = if (c.oraTypeIsAMemRef(c.oraValueGetType(condition_raw))) blk: {
        const result_type = c.oraShapedTypeGetElementType(c.oraValueGetType(condition_raw));
        const load_op = c.oraMemrefLoadOpCreate(self.ctx, loc, condition_raw, null, 0, result_type);
        h.appendOp(self.block, load_op);
        break :blk h.getResult(load_op, 0);
    } else condition_raw;

    const result_type = blk: {
        if (switch_expr.cases.len > 0) {
            switch (switch_expr.cases[0].body) {
                .Expression => |expr| {
                    const expr_type = c.oraValueGetType(self.lowerExpression(expr));
                    break :blk expr_type;
                },
                .Block, .LabeledBlock => {
                    break :blk h.i256Type(self.ctx);
                },
            }
        } else if (switch_expr.default_case) |default_block| {
            if (default_block.statements.len > 0) {
                switch (default_block.statements[0]) {
                    .Return => |ret| {
                        if (ret.value) |e| {
                            const expr_type = c.oraValueGetType(self.lowerExpression(&e));
                            break :blk expr_type;
                        }
                    },
                    .Expr => |e| {
                        const expr_type = c.oraValueGetType(self.lowerExpression(&e));
                        break :blk expr_type;
                    },
                    else => {},
                }
            }
        }
        break :blk h.i256Type(self.ctx);
    };

    const total_cases = switch_expr.cases.len + if (switch_expr.default_case != null) @as(usize, 1) else 0;
    const switch_op = c.oraSwitchExprOpCreateWithCases(
        self.ctx,
        loc,
        condition,
        &[_]c.MlirType{result_type},
        1,
        total_cases,
    );
    if (c.oraOperationIsNull(switch_op)) {
        @panic("Failed to create ora.switch_expr operation");
    }
    h.appendOp(self.block, switch_op);

    const StatementLowerer = @import("../statements.zig").StatementLowerer;

    var case_values = std.ArrayList(i64){};
    defer case_values.deinit(std.heap.page_allocator);
    var range_starts = std.ArrayList(i64){};
    defer range_starts.deinit(std.heap.page_allocator);
    var range_ends = std.ArrayList(i64){};
    defer range_ends.deinit(std.heap.page_allocator);
    var case_kinds = std.ArrayList(i64){};
    defer case_kinds.deinit(std.heap.page_allocator);
    var default_case_index: i64 = -1;

    var case_idx: usize = 0;
    for (switch_expr.cases) |case| {
        const case_block = c.oraSwitchExprOpGetCaseBlock(switch_op, case_idx);
        if (c.oraBlockIsNull(case_block)) {
            @panic("ora.switch_expr missing case block");
        }

        var case_expr_lowerer = ExpressionLowerer.init(self.ctx, case_block, self.type_mapper, self.param_map, self.storage_map, self.local_var_map, self.symbol_table, self.builtin_registry, self.error_handler, self.locations, self.ora_dialect);
        case_expr_lowerer.current_function_return_type = self.current_function_return_type;
        case_expr_lowerer.current_function_return_type_info = self.current_function_return_type_info;
        case_expr_lowerer.in_try_block = self.in_try_block;

        switch (case.pattern) {
            .Literal => |lit| {
                _ = case_expr_lowerer.lowerLiteral(&lit.value);
                if (expr_literals.extractIntegerFromLiteral(&lit.value)) |val| {
                    case_values.append(std.heap.page_allocator, val) catch {};
                    range_starts.append(std.heap.page_allocator, 0) catch {};
                    range_ends.append(std.heap.page_allocator, 0) catch {};
                    case_kinds.append(std.heap.page_allocator, 0) catch {};
                } else {
                    case_values.append(std.heap.page_allocator, 0) catch {};
                    range_starts.append(std.heap.page_allocator, 0) catch {};
                    range_ends.append(std.heap.page_allocator, 0) catch {};
                    case_kinds.append(std.heap.page_allocator, 0) catch {};
                }
            },
            .Range => |range| {
                _ = case_expr_lowerer.lowerExpression(range.start);
                _ = case_expr_lowerer.lowerExpression(range.end);
                const start_val = expr_literals.extractIntegerFromExpr(range.start) orelse 0;
                const end_val = expr_literals.extractIntegerFromExpr(range.end) orelse 0;
                case_values.append(std.heap.page_allocator, 0) catch {};
                range_starts.append(std.heap.page_allocator, start_val) catch {};
                range_ends.append(std.heap.page_allocator, end_val) catch {};
                case_kinds.append(std.heap.page_allocator, 1) catch {};
            },
            .EnumValue => {
                case_values.append(std.heap.page_allocator, 0) catch {};
                range_starts.append(std.heap.page_allocator, 0) catch {};
                range_ends.append(std.heap.page_allocator, 0) catch {};
                case_kinds.append(std.heap.page_allocator, 0) catch {};
            },
            .Else => {
                default_case_index = @intCast(case_idx);
                case_values.append(std.heap.page_allocator, 0) catch {};
                range_starts.append(std.heap.page_allocator, 0) catch {};
                range_ends.append(std.heap.page_allocator, 0) catch {};
                case_kinds.append(std.heap.page_allocator, 2) catch {};
            },
        }

        switch (case.body) {
            .Expression => |expr| {
                const expr_result = case_expr_lowerer.lowerExpression(expr);
                const yield_op = self.ora_dialect.createYield(&[_]c.MlirValue{expr_result}, loc);
                h.appendOp(case_block, yield_op);
            },
            .Block => |block| {
                var has_terminator = false;
                for (block.statements) |stmt| {
                    if (stmt == .Return or stmt == .Break or stmt == .Continue) {
                        has_terminator = true;
                        break;
                    }
                }

                if (has_terminator) {
                    var temp_lowerer = StatementLowerer.init(self.ctx, case_block, self.type_mapper, &case_expr_lowerer, self.param_map, self.storage_map, @constCast(self.local_var_map), self.locations, @constCast(self.symbol_table), self.builtin_registry, std.heap.page_allocator, result_type, null, self.ora_dialect, &[_]*lib.ast.Expressions.ExprNode{});
                    for (block.statements) |stmt| {
                        switch (stmt) {
                            .Return => |ret| {
                                if (ret.value) |e| {
                                    const v = case_expr_lowerer.lowerExpression(&e);
                                    const yield_op = self.ora_dialect.createYield(&[_]c.MlirValue{v}, loc);
                                    h.appendOp(case_block, yield_op);
                                } else {
                                    const default_val = createDefaultValueForType(self, result_type, loc) catch return condition;
                                    const yield_op = self.ora_dialect.createYield(&[_]c.MlirValue{default_val}, loc);
                                    h.appendOp(case_block, yield_op);
                                }
                                break;
                            },
                            .Break => |break_stmt| {
                                if (break_stmt.value) |value_expr| {
                                    const v = case_expr_lowerer.lowerExpression(value_expr);
                                    const yield_op = self.ora_dialect.createYield(&[_]c.MlirValue{v}, loc);
                                    h.appendOp(case_block, yield_op);
                                } else {
                                    const default_val = createDefaultValueForType(self, result_type, loc) catch return condition;
                                    const yield_op = self.ora_dialect.createYield(&[_]c.MlirValue{default_val}, loc);
                                    h.appendOp(case_block, yield_op);
                                }
                                break;
                            },
                            .Continue => |continue_stmt| {
                                if (continue_stmt.value) |value_expr| {
                                    const v = case_expr_lowerer.lowerExpression(value_expr);
                                    const yield_op = self.ora_dialect.createYield(&[_]c.MlirValue{v}, loc);
                                    h.appendOp(case_block, yield_op);
                                } else {
                                    const default_val = createDefaultValueForType(self, result_type, loc) catch return condition;
                                    const yield_op = self.ora_dialect.createYield(&[_]c.MlirValue{default_val}, loc);
                                    h.appendOp(case_block, yield_op);
                                }
                                break;
                            },
                            else => {
                                temp_lowerer.lowerStatement(&stmt) catch {};
                            },
                        }
                    }
                } else {
                    var temp_lowerer = StatementLowerer.init(self.ctx, case_block, self.type_mapper, &case_expr_lowerer, self.param_map, self.storage_map, @constCast(self.local_var_map), self.locations, @constCast(self.symbol_table), self.builtin_registry, std.heap.page_allocator, result_type, null, self.ora_dialect, &[_]*lib.ast.Expressions.ExprNode{});
                    _ = temp_lowerer.lowerBlockBody(block, case_block) catch false;
                    const default_val = createDefaultValueForType(self, result_type, loc) catch return condition;
                    const yield_op = self.ora_dialect.createYield(&[_]c.MlirValue{default_val}, loc);
                    h.appendOp(case_block, yield_op);
                }
            },
            .LabeledBlock => |labeled| {
                var has_terminator = false;
                for (labeled.block.statements) |stmt| {
                    if (stmt == .Return or stmt == .Break or stmt == .Continue) {
                        has_terminator = true;
                        break;
                    }
                }

                if (has_terminator) {
                    var temp_lowerer = StatementLowerer.init(self.ctx, case_block, self.type_mapper, &case_expr_lowerer, self.param_map, self.storage_map, @constCast(self.local_var_map), self.locations, @constCast(self.symbol_table), self.builtin_registry, std.heap.page_allocator, result_type, null, self.ora_dialect, &[_]*lib.ast.Expressions.ExprNode{});
                    for (labeled.block.statements) |stmt| {
                        switch (stmt) {
                            .Return => |ret| {
                                if (ret.value) |e| {
                                    const v = case_expr_lowerer.lowerExpression(&e);
                                    const yield_op = self.ora_dialect.createYield(&[_]c.MlirValue{v}, loc);
                                    h.appendOp(case_block, yield_op);
                                } else {
                                    const default_val = createDefaultValueForType(self, result_type, loc) catch return condition;
                                    const yield_op = self.ora_dialect.createYield(&[_]c.MlirValue{default_val}, loc);
                                    h.appendOp(case_block, yield_op);
                                }
                                break;
                            },
                            .Break => |break_stmt| {
                                if (break_stmt.value) |value_expr| {
                                    const v = case_expr_lowerer.lowerExpression(value_expr);
                                    const yield_op = self.ora_dialect.createYield(&[_]c.MlirValue{v}, loc);
                                    h.appendOp(case_block, yield_op);
                                } else {
                                    const default_val = createDefaultValueForType(self, result_type, loc) catch return condition;
                                    const yield_op = self.ora_dialect.createYield(&[_]c.MlirValue{default_val}, loc);
                                    h.appendOp(case_block, yield_op);
                                }
                                break;
                            },
                            .Continue => |continue_stmt| {
                                if (continue_stmt.value) |value_expr| {
                                    const v = case_expr_lowerer.lowerExpression(value_expr);
                                    const yield_op = self.ora_dialect.createYield(&[_]c.MlirValue{v}, loc);
                                    h.appendOp(case_block, yield_op);
                                } else {
                                    const default_val = createDefaultValueForType(self, result_type, loc) catch return condition;
                                    const yield_op = self.ora_dialect.createYield(&[_]c.MlirValue{default_val}, loc);
                                    h.appendOp(case_block, yield_op);
                                }
                                break;
                            },
                            else => {
                                temp_lowerer.lowerStatement(&stmt) catch {};
                            },
                        }
                    }
                } else {
                    var temp_lowerer = StatementLowerer.init(self.ctx, case_block, self.type_mapper, &case_expr_lowerer, self.param_map, self.storage_map, @constCast(self.local_var_map), self.locations, @constCast(self.symbol_table), self.builtin_registry, std.heap.page_allocator, result_type, null, self.ora_dialect, &[_]*lib.ast.Expressions.ExprNode{});
                    _ = temp_lowerer.lowerBlockBody(labeled.block, case_block) catch false;
                    const default_val = createDefaultValueForType(self, result_type, loc) catch return condition;
                    const yield_op = self.ora_dialect.createYield(&[_]c.MlirValue{default_val}, loc);
                    h.appendOp(case_block, yield_op);
                }
            },
        }

        case_idx += 1;
    }

    if (switch_expr.default_case) |default_block| {
        const default_block_mlir = c.oraSwitchExprOpGetCaseBlock(switch_op, case_idx);
        if (c.oraBlockIsNull(default_block_mlir)) {
            @panic("ora.switch_expr missing default block");
        }

        var default_expr_lowerer = ExpressionLowerer.init(self.ctx, default_block_mlir, self.type_mapper, self.param_map, self.storage_map, self.local_var_map, self.symbol_table, self.builtin_registry, self.error_handler, self.locations, self.ora_dialect);
        default_expr_lowerer.current_function_return_type = self.current_function_return_type;
        default_expr_lowerer.current_function_return_type_info = self.current_function_return_type_info;
        default_expr_lowerer.in_try_block = self.in_try_block;

        var default_has_return = false;
        for (default_block.statements) |stmt| {
            if (stmt == .Return) {
                default_has_return = true;
                break;
            }
        }
        if (default_has_return) {
            var temp_lowerer = StatementLowerer.init(self.ctx, default_block_mlir, self.type_mapper, &default_expr_lowerer, self.param_map, self.storage_map, @constCast(self.local_var_map), self.locations, @constCast(self.symbol_table), self.builtin_registry, std.heap.page_allocator, result_type, null, self.ora_dialect, &[_]*lib.ast.Expressions.ExprNode{});
            for (default_block.statements) |stmt| {
                switch (stmt) {
                    .Return => |ret| {
                        if (ret.value) |e| {
                            const v = default_expr_lowerer.lowerExpression(&e);
                            const yield_op = self.ora_dialect.createYield(&[_]c.MlirValue{v}, loc);
                            h.appendOp(default_block_mlir, yield_op);
                        } else {
                            const default_val = createDefaultValueForType(self, result_type, loc) catch return condition;
                            const yield_op = self.ora_dialect.createYield(&[_]c.MlirValue{default_val}, loc);
                            h.appendOp(default_block_mlir, yield_op);
                        }
                        break;
                    },
                    else => {
                        temp_lowerer.lowerStatement(&stmt) catch {};
                    },
                }
            }
        } else {
            var temp_lowerer = StatementLowerer.init(self.ctx, default_block_mlir, self.type_mapper, &default_expr_lowerer, self.param_map, self.storage_map, @constCast(self.local_var_map), self.locations, @constCast(self.symbol_table), self.builtin_registry, std.heap.page_allocator, result_type, null, self.ora_dialect, &[_]*lib.ast.Expressions.ExprNode{});
            _ = temp_lowerer.lowerBlockBody(default_block, default_block_mlir) catch false;
            const default_val = createDefaultValueForType(self, result_type, loc) catch return condition;
            const yield_op = self.ora_dialect.createYield(&[_]c.MlirValue{default_val}, loc);
            h.appendOp(default_block_mlir, yield_op);
        }

        if (default_case_index < 0) {
            default_case_index = @intCast(case_idx);
        }
        case_values.append(std.heap.page_allocator, 0) catch {};
        range_starts.append(std.heap.page_allocator, 0) catch {};
        range_ends.append(std.heap.page_allocator, 0) catch {};
        case_kinds.append(std.heap.page_allocator, 2) catch {};
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

    return h.getResult(switch_op, 0);
}

/// Lower quantified expressions
pub fn lowerQuantified(
    self: *const ExpressionLowerer,
    quantified: *const lib.ast.Expressions.QuantifiedExpr,
) c.MlirValue {
    const result_ty = h.boolType(self.ctx);
    const body_value = self.lowerExpression(quantified.body);

    var condition_value: ?c.MlirValue = null;
    if (quantified.condition) |condition| {
        condition_value = self.lowerExpression(condition);
    }

    var attributes = std.ArrayList(c.MlirNamedAttribute){};
    defer attributes.deinit(std.heap.page_allocator);

    const quantifier_type_str = if (quantified.verification_metadata) |metadata|
        switch (metadata.quantifier_type) {
            .Forall => "forall",
            .Exists => "exists",
        }
    else switch (quantified.quantifier) {
        .Forall => "forall",
        .Exists => "exists",
    };
    const variable_name = if (quantified.verification_metadata) |metadata|
        metadata.variable_name
    else
        quantified.variable;
    const var_type_str = if (quantified.verification_metadata) |metadata|
        getTypeString(self, metadata.variable_type)
    else
        getTypeString(self, quantified.variable_type);

    if (quantified.verification_metadata) |metadata| {
        const quantifier_id = h.identifier(self.ctx, "quantifier");
        const quantifier_attr = h.stringAttr(self.ctx, quantifier_type_str);
        attributes.append(std.heap.page_allocator, c.oraNamedAttributeGet(quantifier_id, quantifier_attr)) catch {};

        const var_name_id = h.identifier(self.ctx, "variable");
        const var_name_attr = h.stringAttr(self.ctx, metadata.variable_name);
        attributes.append(std.heap.page_allocator, c.oraNamedAttributeGet(var_name_id, var_name_attr)) catch {};

        const var_type_id = h.identifier(self.ctx, "variable_type");
        const var_type_attr = h.stringAttr(self.ctx, var_type_str);
        attributes.append(std.heap.page_allocator, c.oraNamedAttributeGet(var_type_id, var_type_attr)) catch {};

        const has_condition_id = h.identifier(self.ctx, "ora.has_condition");
        const has_condition_attr = h.boolAttr(self.ctx, @as(i32, @intFromBool(metadata.has_condition)));
        attributes.append(std.heap.page_allocator, c.oraNamedAttributeGet(has_condition_id, has_condition_attr)) catch {};

        const span_id = h.identifier(self.ctx, "ora.span");
        const span_str = std.fmt.allocPrint(std.heap.page_allocator, "{}:{}", .{ metadata.span.line, metadata.span.column }) catch "0:0";
        defer std.heap.page_allocator.free(span_str);
        const span_attr = h.stringAttr(self.ctx, span_str);
        attributes.append(std.heap.page_allocator, c.oraNamedAttributeGet(span_id, span_attr)) catch {};
    } else {
        const quantifier_id = h.identifier(self.ctx, "quantifier");
        const quantifier_attr = h.stringAttr(self.ctx, quantifier_type_str);
        attributes.append(std.heap.page_allocator, c.oraNamedAttributeGet(quantifier_id, quantifier_attr)) catch {};

        const var_name_id = h.identifier(self.ctx, "variable");
        const var_name_attr = h.stringAttr(self.ctx, quantified.variable);
        attributes.append(std.heap.page_allocator, c.oraNamedAttributeGet(var_name_id, var_name_attr)) catch {};

        const var_type_id = h.identifier(self.ctx, "variable_type");
        const var_type_attr = h.stringAttr(self.ctx, var_type_str);
        attributes.append(std.heap.page_allocator, c.oraNamedAttributeGet(var_type_id, var_type_attr)) catch {};

        const has_condition_id = h.identifier(self.ctx, "ora.has_condition");
        const has_condition_attr = h.boolAttr(self.ctx, @as(i32, @intFromBool(quantified.condition != null)));
        attributes.append(std.heap.page_allocator, c.oraNamedAttributeGet(has_condition_id, has_condition_attr)) catch {};
    }

    if (quantified.verification_attributes.len > 0) {
        for (quantified.verification_attributes) |attr| {
            if (attr.name) |name| {
                const attr_name_id = c.oraIdentifierGet(self.ctx, h.strRef(name));
                const attr_value = if (attr.value) |value|
                    h.stringAttr(self.ctx, value)
                else
                    c.oraStringAttrCreate(self.ctx, c.oraStringRefCreateFromCString(""));
                attributes.append(std.heap.page_allocator, c.oraNamedAttributeGet(attr_name_id, attr_value)) catch {};
            }
        }
    }

    addVerificationAttributes(self, &attributes, "quantified", "formal_verification");

    const domain_id = h.identifier(self.ctx, "ora.domain");
    const domain_str = switch (quantified.quantifier) {
        .Forall => "universal",
        .Exists => "existential",
    };
    const domain_attr = h.stringAttr(self.ctx, domain_str);
    attributes.append(std.heap.page_allocator, c.oraNamedAttributeGet(domain_id, domain_attr)) catch {};

    const condition_val = if (condition_value) |cond_val| cond_val else c.MlirValue{ .ptr = null };
    const quantified_op = c.oraQuantifiedOpCreate(
        self.ctx,
        self.fileLoc(quantified.span),
        h.strRef(quantifier_type_str),
        h.strRef(variable_name),
        h.strRef(var_type_str),
        condition_val,
        condition_value != null,
        body_value,
        result_ty,
    );
    for (attributes.items) |attr| {
        c.oraOperationSetAttributeByName(quantified_op, c.oraIdentifierStr(attr.name), attr.attribute);
    }
    h.appendOp(self.block, quantified_op);

    return h.getResult(quantified_op, 0);
}

/// Lower try expressions
pub fn lowerTry(
    self: *const ExpressionLowerer,
    try_expr: *const lib.ast.Expressions.TryExpr,
) c.MlirValue {
    const expr_value = self.lowerExpression(try_expr.expr);
    if (c.oraValueIsNull(expr_value)) {
        if (self.error_handler) |handler| {
            handler.reportError(.InternalError, try_expr.span, "failed to lower try operand", "check the called expression inside the try block") catch {};
        }
        return self.createErrorPlaceholder(try_expr.span, "failed to lower try operand");
    }
    const expr_ty = c.oraValueGetType(expr_value);
    var result_type = expr_ty;
    const success_ty = c.oraErrorUnionTypeGetSuccessType(expr_ty);
    if (!c.oraTypeIsNull(success_ty)) {
        result_type = success_ty;
    }
    const loc = self.fileLoc(try_expr.span);
    const should_propagate = if (self.in_try_block)
        false
    else if (self.current_function_return_type_info) |ret_info|
        isErrorUnionType(ret_info)
    else
        false;

    // use the C++ API which automatically creates the catch region
    const op = self.ora_dialect.createTry(expr_value, result_type, loc);
    h.appendOp(self.block, op);

    // ensure the catch region has a terminator to avoid empty block errors
    const catch_block = c.oraTryOpGetCatchBlock(op);
    if (!c.oraBlockIsNull(catch_block)) {
        const first_op = c.oraBlockGetFirstOperation(catch_block);
        if (c.oraOperationIsNull(first_op)) {
            if (should_propagate) {
                const return_op = self.ora_dialect.createFuncReturnWithValue(expr_value, loc);
                h.appendOp(catch_block, return_op);
            } else {
                if (!self.in_try_block) {
                    if (self.error_handler) |handler| {
                        handler.reportError(.TypeMismatch, try_expr.span, "try expression requires an error union return type", "change the function return type to '!T' or use try/catch block") catch {};
                    }
                }
                var catch_lowerer = self.*;
                catch_lowerer.block = catch_block;
                if (c.oraTypeEqual(result_type, c.oraNoneTypeCreate(self.ctx))) {
                    const yield_op = self.ora_dialect.createYield(&[_]c.MlirValue{}, loc);
                    h.appendOp(catch_block, yield_op);
                } else {
                    const default_val = catch_lowerer.createDefaultValueForType(result_type, loc) catch {
                        const yield_op = self.ora_dialect.createYield(&[_]c.MlirValue{}, loc);
                        h.appendOp(catch_block, yield_op);
                        return h.getResult(op, 0);
                    };
                    const yield_op = self.ora_dialect.createYield(&[_]c.MlirValue{default_val}, loc);
                    h.appendOp(catch_block, yield_op);
                }
            }
        }
    }
    return h.getResult(op, 0);
}

fn isErrorUnionType(ti: lib.ast.Types.TypeInfo) bool {
    if (ti.category == .ErrorUnion) return true;
    if (ti.ora_type) |ot| switch (ot) {
        .error_union => return true,
        ._union => |members| return members.len > 0 and members[0] == .error_union,
        else => {},
    };
    return false;
}

/// Lower error return expressions
pub fn lowerErrorReturn(
    self: *const ExpressionLowerer,
    error_ret: *const lib.ast.Expressions.ErrorReturnExpr,
) c.MlirValue {
    const ty = c.oraIntegerTypeCreate(self.ctx, constants.DEFAULT_INTEGER_BITS);
    const error_id = h.identifier(self.ctx, "ora.error");
    const error_name_attr = h.stringAttr(self.ctx, error_ret.error_name);

    const attrs = [_]c.MlirNamedAttribute{
        c.oraNamedAttributeGet(error_id, error_name_attr),
    };
    const op = self.ora_dialect.createArithConstantWithAttrs(1, ty, &attrs, self.fileLoc(error_ret.span));
    h.appendOp(self.block, op);
    return h.getResult(op, 0);
}

/// Lower error cast expressions
pub fn lowerErrorCast(
    self: *const ExpressionLowerer,
    error_cast: *const lib.ast.Expressions.ErrorCastExpr,
) c.MlirValue {
    const operand = self.lowerExpression(error_cast.operand);
    const target_type = self.type_mapper.toMlirType(error_cast.target_type);
    const op = self.ora_dialect.createArithBitcast(operand, target_type, self.fileLoc(error_cast.span));
    h.appendOp(self.block, op);
    return h.getResult(op, 0);
}

/// Lower shift expressions
pub fn lowerShift(
    self: *const ExpressionLowerer,
    shift: *const lib.ast.Expressions.ShiftExpr,
) c.MlirValue {
    const mapping = self.lowerExpression(shift.mapping);
    const source = self.lowerExpression(shift.source);
    const dest = self.lowerExpression(shift.dest);
    const amount = self.lowerExpression(shift.amount);

    const ty = c.oraIntegerTypeCreate(self.ctx, constants.DEFAULT_INTEGER_BITS);
    const op = c.oraMoveOpCreateWithMapping(
        self.ctx,
        self.fileLoc(shift.span),
        mapping,
        source,
        dest,
        amount,
        ty,
    );
    h.appendOp(self.block, op);
    return h.getResult(op, 0);
}

/// Lower struct instantiation expressions
pub fn lowerStructInstantiation(
    self: *const ExpressionLowerer,
    struct_inst: *const lib.ast.Expressions.StructInstantiationExpr,
) c.MlirValue {
    const struct_name_str = switch (struct_inst.struct_name.*) {
        .Identifier => |*id| id.name,
        else => {
            log.err("Struct instantiation struct_name must be an identifier\n", .{});
            return self.reportLoweringError(
                struct_inst.span,
                "invalid struct instantiation target",
                "use a struct identifier when instantiating a struct",
            );
        },
    };

    const result_ty = self.type_mapper.mapStructType(struct_name_str);

    if (struct_inst.fields.len == 0) {
        const op = self.ora_dialect.createStructInstantiate(struct_name_str, &[_]c.MlirValue{}, result_ty, self.fileLoc(struct_inst.span));
        h.appendOp(self.block, op);
        return h.getResult(op, 0);
    }

    var field_values = std.ArrayList(c.MlirValue){};
    defer field_values.deinit(std.heap.page_allocator);

    for (struct_inst.fields) |field| {
        const field_value = self.lowerExpression(field.value);
        field_values.append(std.heap.page_allocator, field_value) catch {};
    }

    const op = self.ora_dialect.createStructInstantiate(struct_name_str, field_values.items, result_ty, self.fileLoc(struct_inst.span));
    h.appendOp(self.block, op);
    return h.getResult(op, 0);
}

/// Lower anonymous struct expressions
pub fn lowerAnonymousStruct(
    self: *const ExpressionLowerer,
    anon_struct: *const lib.ast.Expressions.AnonymousStructExpr,
) c.MlirValue {
    if (anon_struct.fields.len == 0) {
        return createEmptyStruct(self, anon_struct.span);
    }

    return createInitializedStruct(self, anon_struct.fields, anon_struct.span);
}

/// Lower range expressions
pub fn lowerRange(
    self: *const ExpressionLowerer,
    range: *const lib.ast.Expressions.RangeExpr,
) c.MlirValue {
    const start = self.lowerExpression(range.start);
    const end = self.lowerExpression(range.end);

    const result_ty = c.oraIntegerTypeCreate(self.ctx, constants.DEFAULT_INTEGER_BITS);
    const op = c.oraRangeOpCreate(
        self.ctx,
        self.fileLoc(range.span),
        start,
        end,
        result_ty,
        range.inclusive,
    );
    h.appendOp(self.block, op);
    return h.getResult(op, 0);
}

/// Lower labeled block expressions
pub fn lowerLabeledBlock(
    self: *const ExpressionLowerer,
    labeled_block: *const lib.ast.Expressions.LabeledBlockExpr,
) c.MlirValue {
    const ty = c.oraIntegerTypeCreate(self.ctx, constants.DEFAULT_INTEGER_BITS);
    const op = c.oraScfExecuteRegionOpCreate(self.ctx, self.fileLoc(labeled_block.span), &ty, 1, false);
    if (c.oraOperationIsNull(op)) {
        @panic("Failed to create scf.execute_region operation");
    }
    h.appendOp(self.block, op);

    const label_attr = h.stringAttr(self.ctx, labeled_block.label);
    c.oraOperationSetAttributeByName(op, h.strRef("ora.label"), label_attr);

    const StatementLowerer = @import("../statements.zig").StatementLowerer;
    const block = c.oraScfExecuteRegionOpGetBodyBlock(op);
    if (c.oraBlockIsNull(block)) {
        @panic("scf.execute_region missing body block");
    }

    const stmt_lowerer = StatementLowerer.init(self.ctx, block, self.type_mapper, self, self.param_map, self.storage_map, @constCast(self.local_var_map), self.locations, null, self.builtin_registry, std.heap.page_allocator, null, null, self.ora_dialect, &[_]*lib.ast.Expressions.ExprNode{});

    for (labeled_block.block.statements) |stmt| {
        stmt_lowerer.lowerStatement(&stmt) catch |err| {
            log.err("lowering statement in labeled block: {s}\n", .{@errorName(err)});
            return self.createConstant(0, labeled_block.span);
        };
    }

    return h.getResult(op, 0);
}

/// Lower destructuring expressions
pub fn lowerDestructuring(
    self: *const ExpressionLowerer,
    destructuring: *const lib.ast.Expressions.DestructuringExpr,
) c.MlirValue {
    const value = self.lowerExpression(destructuring.value);

    const result_ty = c.oraIntegerTypeCreate(self.ctx, constants.DEFAULT_INTEGER_BITS);
    const pattern_type = switch (destructuring.pattern) {
        .Struct => "struct",
        .Tuple => "tuple",
        .Array => "array",
    };
    const op = self.ora_dialect.createDestructure(
        value,
        pattern_type,
        result_ty,
        self.fileLoc(destructuring.span),
    );
    h.appendOp(self.block, op);
    return h.getResult(op, 0);
}

/// Lower enum literal expressions
pub fn lowerEnumLiteral(
    self: *const ExpressionLowerer,
    enum_lit: *const lib.ast.Expressions.EnumLiteralExpr,
) c.MlirValue {
    var enum_ty = c.oraIntegerTypeCreate(self.ctx, constants.DEFAULT_INTEGER_BITS);
    var enum_value: i64 = 0;

    if (self.symbol_table) |st| {
        if (st.lookupType(enum_lit.enum_name)) |type_sym| {
            if (type_sym.type_kind == .Enum) {
                enum_ty = type_sym.mlir_type;

                if (type_sym.variants) |variants| {
                    for (variants) |variant| {
                        if (std.mem.eql(u8, variant.name, enum_lit.variant_name)) {
                            if (variant.value) |val| {
                                enum_value = val;
                            }
                            break;
                        }
                    }
                }
            }
        }
    }

    const enum_id = h.identifier(self.ctx, "ora.enum");
    const enum_name_attr = h.stringAttr(self.ctx, enum_lit.enum_name);

    const attrs = [_]c.MlirNamedAttribute{
        c.oraNamedAttributeGet(enum_id, enum_name_attr),
    };
    const op = self.ora_dialect.createArithConstantWithAttrs(enum_value, enum_ty, &attrs, self.fileLoc(enum_lit.span));
    h.appendOp(self.block, op);
    return h.getResult(op, 0);
}

/// Lower array literal expressions
pub fn lowerArrayLiteral(
    self: *const ExpressionLowerer,
    array_lit: *const lib.ast.Expressions.ArrayLiteralExpr,
) c.MlirValue {
    if (array_lit.elements.len == 0) {
        return createEmptyArray(self, array_lit.span);
    }

    return createInitializedArray(self, array_lit.elements, array_lit.span);
}

/// Create default value for a given MLIR type
pub fn createDefaultValueForType(
    self: *const ExpressionLowerer,
    mlir_type: c.MlirType,
    loc: c.MlirLocation,
) !c.MlirValue {
    if (c.oraTypeIsAInteger(mlir_type)) {
        const const_op = self.ora_dialect.createArithConstant(0, mlir_type, loc);
        h.appendOp(self.block, const_op);
        return h.getResult(const_op, 0);
    }
    const zero_type = c.oraIntegerTypeCreate(self.ctx, 256);
    const const_op = self.ora_dialect.createArithConstant(0, zero_type, loc);
    h.appendOp(self.block, const_op);
    return h.getResult(const_op, 0);
}

/// Create tuple type from element types
pub fn createTupleType(
    self: *const ExpressionLowerer,
    element_types: []c.MlirType,
) c.MlirType {
    if (element_types.len == 0) {
        return c.oraIntegerTypeCreate(self.ctx, constants.DEFAULT_INTEGER_BITS);
    }

    return element_types[0];
}

/// Create empty array memref
pub fn createEmptyArray(
    self: *const ExpressionLowerer,
    span: lib.ast.SourceSpan,
) c.MlirValue {
    const element_ty = c.oraIntegerTypeCreate(self.ctx, constants.DEFAULT_INTEGER_BITS);
    const memref_ty = h.memRefType(self.ctx, element_ty, 1, @ptrCast(&@as(i64, 0)), h.nullAttr(), h.nullAttr());

    const op = self.ora_dialect.createMemrefAlloca(memref_ty, self.fileLoc(span));
    h.appendOp(self.block, op);
    return h.getResult(op, 0);
}

/// Create initialized array with elements
pub fn createInitializedArray(
    self: *const ExpressionLowerer,
    elements: []*lib.ast.Expressions.ExprNode,
    span: lib.ast.SourceSpan,
) c.MlirValue {
    const first_element_val = self.lowerExpression(elements[0]);
    const element_ty = c.oraValueGetType(first_element_val);

    const array_size = @as(i64, @intCast(elements.len));
    const memref_ty = h.memRefType(self.ctx, element_ty, 1, @ptrCast(&array_size), h.nullAttr(), h.nullAttr());

    const alloca_op = self.ora_dialect.createMemrefAlloca(memref_ty, self.fileLoc(span));
    h.appendOp(self.block, alloca_op);
    const array_ref = h.getResult(alloca_op, 0);

    const array_type = c.oraValueGetType(array_ref);
    const element_type = c.oraShapedTypeGetElementType(array_type);
    for (elements, 0..) |element, i| {
        var element_val = if (i == 0) first_element_val else self.lowerExpression(element);
        element_val = self.convertToType(element_val, element_type, span);
        const index_val = self.createConstant(@intCast(i), span);
        const index_index = expr_access.convertIndexToIndexType(self, index_val, span);

        const store_op = c.oraMemrefStoreOpCreate(
            self.ctx,
            self.fileLoc(span),
            element_val,
            array_ref,
            &[_]c.MlirValue{index_index},
            1,
        );
        h.appendOp(self.block, store_op);
    }

    return array_ref;
}

/// Create empty struct
pub fn createEmptyStruct(
    self: *const ExpressionLowerer,
    span: lib.ast.SourceSpan,
) c.MlirValue {
    const struct_ty = c.oraIntegerTypeCreate(self.ctx, constants.DEFAULT_INTEGER_BITS);
    const struct_id = h.identifier(self.ctx, "ora.empty_struct");
    const struct_attr = h.boolAttr(self.ctx, 1);

    const attrs = [_]c.MlirNamedAttribute{
        c.oraNamedAttributeGet(struct_id, struct_attr),
    };
    const op = self.ora_dialect.createArithConstantWithAttrs(0, struct_ty, &attrs, self.fileLoc(span));
    h.appendOp(self.block, op);
    return h.getResult(op, 0);
}

/// Create initialized struct with fields
pub fn createInitializedStruct(
    self: *const ExpressionLowerer,
    fields: []lib.ast.Expressions.AnonymousStructField,
    span: lib.ast.SourceSpan,
) c.MlirValue {
    const struct_ty = c.oraIntegerTypeCreate(self.ctx, constants.DEFAULT_INTEGER_BITS);

    var field_values = std.ArrayList(c.MlirValue){};
    defer field_values.deinit(std.heap.page_allocator);

    for (fields) |field| {
        const field_val = self.lowerExpression(field.value);
        field_values.append(std.heap.page_allocator, field_val) catch {
            log.warn("Failed to append field value to struct initialization\n", .{});
            return self.createErrorPlaceholder(span, "Failed to append field value");
        };
    }

    const struct_op = self.ora_dialect.createStructInit(field_values.items, struct_ty, self.fileLoc(span));
    h.appendOp(self.block, struct_op);
    return h.getResult(struct_op, 0);
}

/// Create expression capture operation
pub fn createExpressionCapture(
    self: *const ExpressionLowerer,
    expr_value: c.MlirValue,
    span: lib.ast.SourceSpan,
) c.MlirOperation {
    const result_ty = c.oraValueGetType(expr_value);

    const op = c.oraExpressionCaptureOpCreate(self.ctx, self.fileLoc(span), expr_value, result_ty);
    h.appendOp(self.block, op);
    return op;
}

/// Convert TypeInfo to string representation
pub fn getTypeString(
    self: *const ExpressionLowerer,
    type_info: lib.ast.Types.TypeInfo,
) []const u8 {
    _ = self;

    if (type_info.ora_type) |ora_type| {
        return switch (ora_type) {
            .u8 => "u8",
            .u16 => "u16",
            .u32 => "u32",
            .u64 => "u64",
            .u128 => "u128",
            .u256 => "u256",
            .i8 => "i8",
            .i16 => "i16",
            .i32 => "i32",
            .i64 => "i64",
            .i128 => "i128",
            .i256 => "i256",
            .bool => "bool",
            .address => "address",
            .string => "string",
            .bytes => "bytes",
            .void => "void",
            .array => "array",
            .slice => "slice",
            .map => "map",
            .struct_type => "struct",
            .enum_type => "enum",
            .error_union => "error_union",
            .function => "function",
            .contract_type => "contract",
            .tuple => "tuple",
            ._union => "union",
            .anonymous_struct => "anonymous_struct",
            .module => "module",
            .min_value => "MinValue",
            .max_value => "MaxValue",
            .in_range => "InRange",
            .scaled => "Scaled",
            .exact => "Exact",
            .non_zero_address => "NonZeroAddress",
        };
    }

    return "unknown";
}

/// Add verification-related attributes
pub fn addVerificationAttributes(
    self: *const ExpressionLowerer,
    attributes: *std.ArrayList(c.MlirNamedAttribute),
    verification_type: []const u8,
    context: []const u8,
) void {
    const verification_id = h.identifier(self.ctx, "ora.verification");
    const verification_attr = h.boolAttr(self.ctx, 1);
    attributes.append(std.heap.page_allocator, c.oraNamedAttributeGet(verification_id, verification_attr)) catch {};

    const type_id = h.identifier(self.ctx, "ora.verification_type");
    const type_attr = h.stringAttr(self.ctx, verification_type);
    attributes.append(std.heap.page_allocator, c.oraNamedAttributeGet(type_id, type_attr)) catch {};

    const context_id = h.identifier(self.ctx, "ora.verification_context");
    const context_attr = h.stringAttr(self.ctx, context);
    attributes.append(std.heap.page_allocator, c.oraNamedAttributeGet(context_id, context_attr)) catch {};

    const formal_id = h.identifier(self.ctx, "ora.formal");
    const formal_attr = h.boolAttr(self.ctx, 1);
    attributes.append(std.heap.page_allocator, c.oraNamedAttributeGet(formal_id, formal_attr)) catch {};
}

/// Create verification metadata
pub fn createVerificationMetadata(
    self: *const ExpressionLowerer,
    quantifier_type: lib.ast.Expressions.QuantifierType,
    variable_name: []const u8,
    variable_type: lib.ast.Types.TypeInfo,
) std.ArrayList(c.MlirNamedAttribute) {
    var metadata = std.ArrayList(c.MlirNamedAttribute){};

    const quantifier_str = switch (quantifier_type) {
        .Forall => "forall",
        .Exists => "exists",
    };
    const quantifier_id = h.identifier(self.ctx, "quantifier");
    const quantifier_attr = h.stringAttr(self.ctx, quantifier_str);
    metadata.append(std.heap.page_allocator, c.oraNamedAttributeGet(quantifier_id, quantifier_attr)) catch {};

    const var_name_id = h.identifier(self.ctx, "variable");
    const var_name_attr = h.stringAttr(self.ctx, variable_name);
    metadata.append(std.heap.page_allocator, c.oraNamedAttributeGet(var_name_id, var_name_attr)) catch {};

    const var_type_id = h.identifier(self.ctx, "variable_type");
    const var_type_str = getTypeString(self, variable_type);
    const var_type_attr = h.stringAttr(self.ctx, var_type_str);
    metadata.append(std.heap.page_allocator, c.oraNamedAttributeGet(var_type_id, var_type_attr)) catch {};

    addVerificationAttributes(self, &metadata, "quantified", "formal_verification");

    return metadata;
}

/// Create switch if chain (alternative implementation)
pub fn createSwitchIfChain(
    self: *const ExpressionLowerer,
    condition: c.MlirValue,
    cases: []lib.ast.Expressions.SwitchCase,
    span: lib.ast.SourceSpan,
) c.MlirValue {
    _ = self;
    _ = cases;
    _ = span;
    log.warn("createSwitchIfChain called but not fully implemented - using condition as fallback\n", .{});
    return condition;
}
