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
                    result_type = c.mlirValueGetType(result_value.?);
                    break;
                }
            },
            .Expr => |expr_node| {
                result_value = self.lowerExpression(&expr_node);
                result_type = c.mlirValueGetType(result_value.?);
                break;
            },
            else => {},
        }
    }

    const ty = result_type orelse c.mlirIntegerTypeGet(self.ctx, constants.DEFAULT_INTEGER_BITS);
    var state = h.opState("arith.constant", loc);
    c.mlirOperationStateAddResults(&state, 1, @ptrCast(&ty));

    const attr = c.mlirIntegerAttrGet(ty, 0);
    const value_id = h.identifier(self.ctx, "value");
    const comptime_id = h.identifier(self.ctx, "ora.comptime");
    const comptime_attr = c.mlirBoolAttrGet(self.ctx, 1);

    var attrs = [_]c.MlirNamedAttribute{
        c.mlirNamedAttributeGet(value_id, attr),
        c.mlirNamedAttributeGet(comptime_id, comptime_attr),
    };
    c.mlirOperationStateAddAttributes(&state, attrs.len, &attrs);

    const op = c.mlirOperationCreate(&state);
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
        const ty = c.mlirValueGetType(value);
        element_types.append(std.heap.page_allocator, ty) catch {};
    }

    const tuple_ty = createTupleType(self, element_types.items);
    var undef_state = h.opState("llvm.mlir.undef", self.fileLoc(tuple.span));
    c.mlirOperationStateAddResults(&undef_state, 1, @ptrCast(&tuple_ty));
    const undef_op = c.mlirOperationCreate(&undef_state);
    h.appendOp(self.block, undef_op);
    var current_tuple = h.getResult(undef_op, 0);

    for (element_values.items, 0..) |element_value, i| {
        var insert_state = h.opState("llvm.insertvalue", self.fileLoc(tuple.span));
        c.mlirOperationStateAddOperands(&insert_state, 2, @ptrCast(&[_]c.MlirValue{ current_tuple, element_value }));
        c.mlirOperationStateAddResults(&insert_state, 1, @ptrCast(&tuple_ty));

        const position_attr = c.mlirIntegerAttrGet(c.mlirIntegerTypeGet(self.ctx, 32), @intCast(i));
        const position_id = h.identifier(self.ctx, "position");
        var attrs = [_]c.MlirNamedAttribute{
            c.mlirNamedAttributeGet(position_id, position_attr),
        };
        c.mlirOperationStateAddAttributes(&insert_state, attrs.len, &attrs);

        const insert_op = c.mlirOperationCreate(&insert_state);
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
    const condition = if (c.mlirTypeIsAMemRef(c.mlirValueGetType(condition_raw))) blk: {
        var load_state = h.opState("memref.load", loc);
        c.mlirOperationStateAddOperands(&load_state, 1, @ptrCast(&condition_raw));
        const result_type = c.mlirShapedTypeGetElementType(c.mlirValueGetType(condition_raw));
        c.mlirOperationStateAddResults(&load_state, 1, @ptrCast(&result_type));
        const load_op = c.mlirOperationCreate(&load_state);
        h.appendOp(self.block, load_op);
        break :blk h.getResult(load_op, 0);
    } else condition_raw;

    const result_type = blk: {
        if (switch_expr.cases.len > 0) {
            switch (switch_expr.cases[0].body) {
                .Expression => |expr| {
                    const expr_type = c.mlirValueGetType(self.lowerExpression(expr));
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
                            const expr_type = c.mlirValueGetType(self.lowerExpression(&e));
                            break :blk expr_type;
                        }
                    },
                    .Expr => |e| {
                        const expr_type = c.mlirValueGetType(self.lowerExpression(&e));
                        break :blk expr_type;
                    },
                    else => {},
                }
            }
        }
        break :blk h.i256Type(self.ctx);
    };

    var switch_state = h.opState("ora.switch_expr", loc);
    c.mlirOperationStateAddOperands(&switch_state, 1, @ptrCast(&condition));
    c.mlirOperationStateAddResults(&switch_state, 1, @ptrCast(&result_type));

    const total_cases = switch_expr.cases.len + if (switch_expr.default_case != null) @as(usize, 1) else 0;
    var case_regions_buf: [16]c.MlirRegion = undefined;
    const case_regions = if (total_cases <= 16) case_regions_buf[0..total_cases] else blk: {
        const regions = std.heap.page_allocator.alloc(c.MlirRegion, total_cases) catch return condition;
        break :blk regions;
    };
    defer if (total_cases > 16) std.heap.page_allocator.free(case_regions);

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
        const case_region = c.mlirRegionCreate();
        const case_block = c.mlirBlockCreate(0, null, null);
        c.mlirRegionInsertOwnedBlock(case_region, 0, case_block);

        const case_expr_lowerer = ExpressionLowerer.init(self.ctx, case_block, self.type_mapper, self.param_map, self.storage_map, self.local_var_map, self.symbol_table, self.builtin_registry, self.error_handler, self.locations, self.ora_dialect);

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

        case_regions[case_idx] = case_region;
        case_idx += 1;
    }

    if (switch_expr.default_case) |default_block| {
        const default_region = c.mlirRegionCreate();
        const default_block_mlir = c.mlirBlockCreate(0, null, null);
        c.mlirRegionInsertOwnedBlock(default_region, 0, default_block_mlir);

        const default_expr_lowerer = ExpressionLowerer.init(self.ctx, default_block_mlir, self.type_mapper, self.param_map, self.storage_map, self.local_var_map, self.symbol_table, self.builtin_registry, self.error_handler, self.locations, self.ora_dialect);

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

        case_regions[case_idx] = default_region;
        if (default_case_index < 0) {
            default_case_index = @intCast(case_idx);
        }
        case_values.append(std.heap.page_allocator, 0) catch {};
        range_starts.append(std.heap.page_allocator, 0) catch {};
        range_ends.append(std.heap.page_allocator, 0) catch {};
        case_kinds.append(std.heap.page_allocator, 2) catch {};
        case_idx += 1;
    }

    c.mlirOperationStateAddOwnedRegions(&switch_state, @intCast(case_regions.len), case_regions.ptr);

    const switch_op = c.mlirOperationCreate(&switch_state);
    h.appendOp(self.block, switch_op);

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

    var state = h.opState("ora.quantified", self.fileLoc(quantified.span));
    c.mlirOperationStateAddResults(&state, 1, @ptrCast(&result_ty));

    var attributes = std.ArrayList(c.MlirNamedAttribute){};
    defer attributes.deinit(std.heap.page_allocator);

    if (quantified.verification_metadata) |metadata| {
        const quantifier_type_str = switch (metadata.quantifier_type) {
            .Forall => "forall",
            .Exists => "exists",
        };
        const quantifier_id = h.identifier(self.ctx, "quantifier");
        const quantifier_attr = h.stringAttr(self.ctx, quantifier_type_str);
        attributes.append(std.heap.page_allocator, c.mlirNamedAttributeGet(quantifier_id, quantifier_attr)) catch {};

        const var_name_id = h.identifier(self.ctx, "variable");
        const var_name_attr = h.stringAttr(self.ctx, metadata.variable_name);
        attributes.append(std.heap.page_allocator, c.mlirNamedAttributeGet(var_name_id, var_name_attr)) catch {};

        const var_type_id = h.identifier(self.ctx, "variable_type");
        const var_type_str = getTypeString(self, metadata.variable_type);
        const var_type_attr = h.stringAttr(self.ctx, var_type_str);
        attributes.append(std.heap.page_allocator, c.mlirNamedAttributeGet(var_type_id, var_type_attr)) catch {};

        const has_condition_id = h.identifier(self.ctx, "ora.has_condition");
        const has_condition_attr = c.mlirBoolAttrGet(self.ctx, if (metadata.has_condition) 1 else 0);
        attributes.append(std.heap.page_allocator, c.mlirNamedAttributeGet(has_condition_id, has_condition_attr)) catch {};

        const span_id = h.identifier(self.ctx, "ora.span");
        const span_str = std.fmt.allocPrint(std.heap.page_allocator, "{}:{}", .{ metadata.span.line, metadata.span.column }) catch "0:0";
        defer std.heap.page_allocator.free(span_str);
        const span_attr = h.stringAttr(self.ctx, span_str);
        attributes.append(std.heap.page_allocator, c.mlirNamedAttributeGet(span_id, span_attr)) catch {};
    } else {
        const quantifier_type_str = switch (quantified.quantifier) {
            .Forall => "forall",
            .Exists => "exists",
        };
        const quantifier_id = h.identifier(self.ctx, "quantifier");
        const quantifier_attr = h.stringAttr(self.ctx, quantifier_type_str);
        attributes.append(std.heap.page_allocator, c.mlirNamedAttributeGet(quantifier_id, quantifier_attr)) catch {};

        const var_name_id = h.identifier(self.ctx, "variable");
        const var_name_attr = h.stringAttr(self.ctx, quantified.variable);
        attributes.append(std.heap.page_allocator, c.mlirNamedAttributeGet(var_name_id, var_name_attr)) catch {};

        const var_type_id = h.identifier(self.ctx, "variable_type");
        const var_type_str = getTypeString(self, quantified.variable_type);
        const var_type_attr = h.stringAttr(self.ctx, var_type_str);
        attributes.append(std.heap.page_allocator, c.mlirNamedAttributeGet(var_type_id, var_type_attr)) catch {};

        const has_condition_id = h.identifier(self.ctx, "ora.has_condition");
        const has_condition_attr = c.mlirBoolAttrGet(self.ctx, if (quantified.condition != null) 1 else 0);
        attributes.append(std.heap.page_allocator, c.mlirNamedAttributeGet(has_condition_id, has_condition_attr)) catch {};
    }

    if (quantified.verification_attributes.len > 0) {
        for (quantified.verification_attributes) |attr| {
            if (attr.name) |name| {
                const attr_name_id = c.mlirIdentifierGet(self.ctx, h.strRef(name));
                const attr_value = if (attr.value) |value|
                    h.stringAttr(self.ctx, value)
                else
                    c.mlirStringAttrGet(self.ctx, c.mlirStringRefCreateFromCString(""));
                attributes.append(std.heap.page_allocator, c.mlirNamedAttributeGet(attr_name_id, attr_value)) catch {};
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
    attributes.append(std.heap.page_allocator, c.mlirNamedAttributeGet(domain_id, domain_attr)) catch {};

    c.mlirOperationStateAddAttributes(&state, @intCast(attributes.items.len), attributes.items.ptr);

    var operands = std.ArrayList(c.MlirValue){};
    defer operands.deinit(std.heap.page_allocator);

    if (condition_value) |cond_val| {
        operands.append(std.heap.page_allocator, cond_val) catch {};
    }
    operands.append(std.heap.page_allocator, body_value) catch {};

    c.mlirOperationStateAddOperands(&state, @intCast(operands.items.len), operands.items.ptr);

    const quantified_op = c.mlirOperationCreate(&state);
    h.appendOp(self.block, quantified_op);

    return h.getResult(quantified_op, 0);
}

/// Lower try expressions
pub fn lowerTry(
    self: *const ExpressionLowerer,
    try_expr: *const lib.ast.Expressions.TryExpr,
) c.MlirValue {
    const expr_value = self.lowerExpression(try_expr.expr);
    const expr_ty = c.mlirValueGetType(expr_value);
    const loc = self.fileLoc(try_expr.span);

    // use the C++ API which automatically creates the catch region
    const op = self.ora_dialect.createTry(expr_value, expr_ty, loc);
    h.appendOp(self.block, op);
    return h.getResult(op, 0);
}

/// Lower error return expressions
pub fn lowerErrorReturn(
    self: *const ExpressionLowerer,
    error_ret: *const lib.ast.Expressions.ErrorReturnExpr,
) c.MlirValue {
    const ty = c.mlirIntegerTypeGet(self.ctx, constants.DEFAULT_INTEGER_BITS);
    var state = h.opState("arith.constant", self.fileLoc(error_ret.span));
    c.mlirOperationStateAddResults(&state, 1, @ptrCast(&ty));

    const attr = c.mlirIntegerAttrGet(ty, 1);
    const value_id = h.identifier(self.ctx, "value");
    const error_id = h.identifier(self.ctx, "ora.error");
    const error_name_attr = h.stringAttr(self.ctx, error_ret.error_name);

    var attrs = [_]c.MlirNamedAttribute{
        c.mlirNamedAttributeGet(value_id, attr),
        c.mlirNamedAttributeGet(error_id, error_name_attr),
    };
    c.mlirOperationStateAddAttributes(&state, attrs.len, &attrs);

    const op = c.mlirOperationCreate(&state);
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

    const ty = c.mlirIntegerTypeGet(self.ctx, constants.DEFAULT_INTEGER_BITS);
    var state = h.opState("ora.move", self.fileLoc(shift.span));
    c.mlirOperationStateAddOperands(&state, 4, @ptrCast(&[_]c.MlirValue{ mapping, source, dest, amount }));
    c.mlirOperationStateAddResults(&state, 1, @ptrCast(&ty));

    const op = c.mlirOperationCreate(&state);
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
            std.debug.print("ERROR: Struct instantiation struct_name must be an identifier\n", .{});
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

    const result_ty = c.mlirIntegerTypeGet(self.ctx, constants.DEFAULT_INTEGER_BITS);
    var state = h.opState("ora.range", self.fileLoc(range.span));
    c.mlirOperationStateAddOperands(&state, 2, @ptrCast(&[_]c.MlirValue{ start, end }));
    c.mlirOperationStateAddResults(&state, 1, @ptrCast(&result_ty));

    const inclusive_attr = c.mlirBoolAttrGet(self.ctx, if (range.inclusive) 1 else 0);
    const inclusive_id = h.identifier(self.ctx, "inclusive");
    var attrs = [_]c.MlirNamedAttribute{
        c.mlirNamedAttributeGet(inclusive_id, inclusive_attr),
    };
    c.mlirOperationStateAddAttributes(&state, attrs.len, &attrs);

    const op = c.mlirOperationCreate(&state);
    h.appendOp(self.block, op);
    return h.getResult(op, 0);
}

/// Lower labeled block expressions
pub fn lowerLabeledBlock(
    self: *const ExpressionLowerer,
    labeled_block: *const lib.ast.Expressions.LabeledBlockExpr,
) c.MlirValue {
    const ty = c.mlirIntegerTypeGet(self.ctx, constants.DEFAULT_INTEGER_BITS);
    var state = h.opState("scf.execute_region", self.fileLoc(labeled_block.span));
    c.mlirOperationStateAddResults(&state, 1, @ptrCast(&ty));

    const op = c.mlirOperationCreate(&state);
    h.appendOp(self.block, op);

    const label_attr = h.stringAttr(self.ctx, labeled_block.label);
    const label_id = h.identifier(self.ctx, "ora.label");
    var attrs = [_]c.MlirNamedAttribute{
        c.mlirNamedAttributeGet(label_id, label_attr),
    };
    c.mlirOperationStateAddAttributes(&state, attrs.len, &attrs);

    const StatementLowerer = @import("../statements.zig").StatementLowerer;
    const region = c.mlirOperationGetRegion(op, 0);
    const block = c.mlirRegionGetFirstBlock(region);

    const stmt_lowerer = StatementLowerer.init(self.ctx, block, self.type_mapper, self, self.param_map, self.storage_map, @constCast(self.local_var_map), self.locations, null, self.builtin_registry, std.heap.page_allocator, null, null, self.ora_dialect, &[_]*lib.ast.Expressions.ExprNode{});

    for (labeled_block.block.statements) |stmt| {
        stmt_lowerer.lowerStatement(&stmt) catch |err| {
            std.debug.print("Error lowering statement in labeled block: {s}\n", .{@errorName(err)});
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

    const result_ty = c.mlirIntegerTypeGet(self.ctx, constants.DEFAULT_INTEGER_BITS);
    var state = h.opState("ora.destructure", self.fileLoc(destructuring.span));
    c.mlirOperationStateAddOperands(&state, 1, @ptrCast(&value));
    c.mlirOperationStateAddResults(&state, 1, @ptrCast(&result_ty));

    const pattern_type = switch (destructuring.pattern) {
        .Struct => "struct",
        .Tuple => "tuple",
        .Array => "array",
    };
    const pattern_attr = c.mlirStringAttrGet(self.ctx, c.mlirStringRefCreateFromCString(pattern_type));
    const pattern_id = h.identifier(self.ctx, "pattern_type");

    var attrs = [_]c.MlirNamedAttribute{
        c.mlirNamedAttributeGet(pattern_id, pattern_attr),
    };
    c.mlirOperationStateAddAttributes(&state, attrs.len, &attrs);

    const op = c.mlirOperationCreate(&state);
    h.appendOp(self.block, op);
    return h.getResult(op, 0);
}

/// Lower enum literal expressions
pub fn lowerEnumLiteral(
    self: *const ExpressionLowerer,
    enum_lit: *const lib.ast.Expressions.EnumLiteralExpr,
) c.MlirValue {
    var enum_ty = c.mlirIntegerTypeGet(self.ctx, constants.DEFAULT_INTEGER_BITS);
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

    var state = h.opState("arith.constant", self.fileLoc(enum_lit.span));
    c.mlirOperationStateAddResults(&state, 1, @ptrCast(&enum_ty));

    const attr = c.mlirIntegerAttrGet(enum_ty, enum_value);
    const value_id = h.identifier(self.ctx, "value");
    const enum_id = h.identifier(self.ctx, "ora.enum");
    const enum_name_attr = h.stringAttr(self.ctx, enum_lit.enum_name);

    var attrs = [_]c.MlirNamedAttribute{
        c.mlirNamedAttributeGet(value_id, attr),
        c.mlirNamedAttributeGet(enum_id, enum_name_attr),
    };
    c.mlirOperationStateAddAttributes(&state, attrs.len, &attrs);

    const op = c.mlirOperationCreate(&state);
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
    if (c.mlirTypeIsAInteger(mlir_type)) {
        var const_state = h.opState("arith.constant", loc);
        const value_attr = c.mlirIntegerAttrGet(mlir_type, 0);
        const value_id = h.identifier(self.ctx, "value");
        var attrs = [_]c.MlirNamedAttribute{c.mlirNamedAttributeGet(value_id, value_attr)};
        c.mlirOperationStateAddAttributes(&const_state, attrs.len, &attrs);
        c.mlirOperationStateAddResults(&const_state, 1, @ptrCast(&mlir_type));
        const const_op = c.mlirOperationCreate(&const_state);
        h.appendOp(self.block, const_op);
        return h.getResult(const_op, 0);
    }
    var const_state = h.opState("arith.constant", loc);
    const zero_type = c.mlirIntegerTypeGet(self.ctx, 256);
    const value_attr = c.mlirIntegerAttrGet(zero_type, 0);
    const value_id = h.identifier(self.ctx, "value");
    var attrs = [_]c.MlirNamedAttribute{c.mlirNamedAttributeGet(value_id, value_attr)};
    c.mlirOperationStateAddAttributes(&const_state, attrs.len, &attrs);
    c.mlirOperationStateAddResults(&const_state, 1, @ptrCast(&zero_type));
    const const_op = c.mlirOperationCreate(&const_state);
    h.appendOp(self.block, const_op);
    return h.getResult(const_op, 0);
}

/// Create tuple type from element types
pub fn createTupleType(
    self: *const ExpressionLowerer,
    element_types: []c.MlirType,
) c.MlirType {
    if (element_types.len == 0) {
        return c.mlirIntegerTypeGet(self.ctx, constants.DEFAULT_INTEGER_BITS);
    }

    return element_types[0];
}

/// Create empty array memref
pub fn createEmptyArray(
    self: *const ExpressionLowerer,
    span: lib.ast.SourceSpan,
) c.MlirValue {
    const element_ty = c.mlirIntegerTypeGet(self.ctx, constants.DEFAULT_INTEGER_BITS);
    const memref_ty = c.mlirMemRefTypeGet(element_ty, 1, @ptrCast(&@as(i64, 0)), c.mlirAttributeGetNull(), c.mlirAttributeGetNull());

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
    const element_ty = c.mlirValueGetType(first_element_val);

    const array_size = @as(i64, @intCast(elements.len));
    const memref_ty = c.mlirMemRefTypeGet(element_ty, 1, @ptrCast(&array_size), c.mlirAttributeGetNull(), c.mlirAttributeGetNull());

    const alloca_op = self.ora_dialect.createMemrefAlloca(memref_ty, self.fileLoc(span));
    h.appendOp(self.block, alloca_op);
    const array_ref = h.getResult(alloca_op, 0);

    const array_type = c.mlirValueGetType(array_ref);
    const element_type = c.mlirShapedTypeGetElementType(array_type);
    for (elements, 0..) |element, i| {
        var element_val = if (i == 0) first_element_val else self.lowerExpression(element);
        element_val = self.convertToType(element_val, element_type, span);
        const index_val = self.createConstant(@intCast(i), span);
        const index_index = expr_access.convertIndexToIndexType(self, index_val, span);

        var store_state = h.opState("memref.store", self.fileLoc(span));
        c.mlirOperationStateAddOperands(&store_state, 3, @ptrCast(&[_]c.MlirValue{ element_val, array_ref, index_index }));
        const store_op = c.mlirOperationCreate(&store_state);
        h.appendOp(self.block, store_op);
    }

    return array_ref;
}

/// Create empty struct
pub fn createEmptyStruct(
    self: *const ExpressionLowerer,
    span: lib.ast.SourceSpan,
) c.MlirValue {
    const struct_ty = c.mlirIntegerTypeGet(self.ctx, constants.DEFAULT_INTEGER_BITS);
    var state = h.opState("arith.constant", self.fileLoc(span));
    c.mlirOperationStateAddResults(&state, 1, @ptrCast(&struct_ty));

    const attr = c.mlirIntegerAttrGet(struct_ty, 0);
    const value_id = h.identifier(self.ctx, "value");
    const struct_id = h.identifier(self.ctx, "ora.empty_struct");
    const struct_attr = c.mlirBoolAttrGet(self.ctx, 1);

    var attrs = [_]c.MlirNamedAttribute{
        c.mlirNamedAttributeGet(value_id, attr),
        c.mlirNamedAttributeGet(struct_id, struct_attr),
    };
    c.mlirOperationStateAddAttributes(&state, attrs.len, &attrs);

    const op = c.mlirOperationCreate(&state);
    h.appendOp(self.block, op);
    return h.getResult(op, 0);
}

/// Create initialized struct with fields
pub fn createInitializedStruct(
    self: *const ExpressionLowerer,
    fields: []lib.ast.Expressions.AnonymousStructField,
    span: lib.ast.SourceSpan,
) c.MlirValue {
    const struct_ty = c.mlirIntegerTypeGet(self.ctx, constants.DEFAULT_INTEGER_BITS);

    var field_values = std.ArrayList(c.MlirValue){};
    defer field_values.deinit(std.heap.page_allocator);

    for (fields) |field| {
        const field_val = self.lowerExpression(field.value);
        field_values.append(std.heap.page_allocator, field_val) catch {
            std.debug.print("WARNING: Failed to append field value to struct initialization\n", .{});
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
    const result_ty = c.mlirValueGetType(expr_value);

    var state = h.opState("ora.expression_capture", self.fileLoc(span));
    c.mlirOperationStateAddOperands(&state, 1, @ptrCast(&expr_value));
    c.mlirOperationStateAddResults(&state, 1, @ptrCast(&result_ty));

    const capture_id = h.identifier(self.ctx, "ora.top_level_expression");
    const capture_attr = c.mlirBoolAttrGet(self.ctx, 1);
    var attrs = [_]c.MlirNamedAttribute{c.mlirNamedAttributeGet(capture_id, capture_attr)};
    c.mlirOperationStateAddAttributes(&state, attrs.len, &attrs);

    const op = c.mlirOperationCreate(&state);
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
    const verification_attr = c.mlirBoolAttrGet(self.ctx, 1);
    attributes.append(std.heap.page_allocator, c.mlirNamedAttributeGet(verification_id, verification_attr)) catch {};

    const type_id = h.identifier(self.ctx, "ora.verification_type");
    const type_attr = h.stringAttr(self.ctx, verification_type);
    attributes.append(std.heap.page_allocator, c.mlirNamedAttributeGet(type_id, type_attr)) catch {};

    const context_id = h.identifier(self.ctx, "ora.verification_context");
    const context_attr = h.stringAttr(self.ctx, context);
    attributes.append(std.heap.page_allocator, c.mlirNamedAttributeGet(context_id, context_attr)) catch {};

    const formal_id = h.identifier(self.ctx, "ora.formal");
    const formal_attr = c.mlirBoolAttrGet(self.ctx, 1);
    attributes.append(std.heap.page_allocator, c.mlirNamedAttributeGet(formal_id, formal_attr)) catch {};
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
    metadata.append(std.heap.page_allocator, c.mlirNamedAttributeGet(quantifier_id, quantifier_attr)) catch {};

    const var_name_id = h.identifier(self.ctx, "variable");
    const var_name_attr = h.stringAttr(self.ctx, variable_name);
    metadata.append(std.heap.page_allocator, c.mlirNamedAttributeGet(var_name_id, var_name_attr)) catch {};

    const var_type_id = h.identifier(self.ctx, "variable_type");
    const var_type_str = getTypeString(self, variable_type);
    const var_type_attr = h.stringAttr(self.ctx, var_type_str);
    metadata.append(std.heap.page_allocator, c.mlirNamedAttributeGet(var_type_id, var_type_attr)) catch {};

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
    std.debug.print("WARNING: createSwitchIfChain called but not fully implemented - using condition as fallback\n", .{});
    return condition;
}
