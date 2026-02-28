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
const LocalVarMap = @import("../symbols.zig").LocalVarMap;
const expr_helpers = @import("helpers.zig");
const expr_access = @import("access.zig");
const expr_literals = @import("literals.zig");
const log = @import("log");
const comptime_eval = lib.comptime_eval;

/// ExpressionLowerer type (forward declaration)
const ExpressionLowerer = @import("mod.zig").ExpressionLowerer;

fn reportSwitchPatternLoweringError(
    self: *const ExpressionLowerer,
    span: lib.ast.SourceSpan,
    message: []const u8,
    suggestion: ?[]const u8,
) c.MlirValue {
    if (self.error_handler) |handler| {
        handler.reportError(.TypeMismatch, span, message, suggestion) catch {};
        return self.createErrorPlaceholder(span, message);
    }
    @panic(message);
}

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
    var result_expr: ?*const lib.ast.Expressions.ExprNode = null;

    for (comptime_expr.block.statements) |*stmt| {
        switch (stmt.*) {
            .Return => |ret| {
                if (ret.value) |*expr| {
                    result_expr = expr;
                    break;
                }
            },
            .Expr => |expr_node| {
                result_expr = &expr_node;
                break;
            },
            else => {},
        }
    }

    const ty = blk: {
        if (comptime_expr.type_info.ora_type != null) {
            break :blk self.type_mapper.toMlirType(comptime_expr.type_info);
        }
        if (result_expr) |expr| {
            if (getExpressionTypeInfo(expr)) |expr_type| {
                if (expr_type.ora_type != null) {
                    break :blk self.type_mapper.toMlirType(expr_type);
                }
            }
        }
        if (self.error_handler) |handler| {
            handler.reportError(.TypeMismatch, comptime_expr.span, "Missing Ora type for comptime expression", "ensure comptime expression is typed during type resolution") catch {};
        }
        break :blk c.oraNoneTypeCreate(self.ctx);
    };
    const comptime_id = h.identifier(self.ctx, "ora.comptime");
    const comptime_attr = h.boolAttr(self.ctx, 1);

    const attrs = [_]c.MlirNamedAttribute{
        c.oraNamedAttributeGet(comptime_id, comptime_attr),
    };

    if (result_expr) |expr| {
        const eval_result = comptime_eval.evaluateExpr(std.heap.page_allocator, @constCast(expr), null);

        switch (eval_result) {
            .value => |ct_val| {
                switch (ct_val) {
                    .integer => |value| {
                        if (value > @as(u256, @intCast(std.math.maxInt(i64)))) {
                            log.err("comptime constant too large for MLIR i64 attribute\n", .{});
                            if (self.error_handler) |handler| {
                                handler.reportError(.CompilationLimit, comptime_expr.span, "comptime constant too large for MLIR i64 attribute", null) catch {};
                            }
                            const op = self.ora_dialect.createArithConstantWithAttrs(0, ty, &attrs, loc);
                            h.appendOp(self.block, op);
                            return h.getResult(op, 0);
                        }
                        const op = self.ora_dialect.createArithConstantWithAttrs(@intCast(value), ty, &attrs, loc);
                        h.appendOp(self.block, op);
                        return h.getResult(op, 0);
                    },
                    .boolean => |value| {
                        const op = self.ora_dialect.createArithConstantWithAttrs(if (value) 1 else 0, ty, &attrs, loc);
                        h.appendOp(self.block, op);
                        return h.getResult(op, 0);
                    },
                    else => {
                        log.err("comptime expression is not a scalar constant yet\n", .{});
                        if (self.error_handler) |handler| {
                            handler.reportError(.InternalError, comptime_expr.span, "comptime expression is not a scalar constant", null) catch {};
                        }
                        const op = self.ora_dialect.createArithConstantWithAttrs(0, ty, &attrs, loc);
                        h.appendOp(self.block, op);
                        return h.getResult(op, 0);
                    },
                }
            },
            .not_constant => {
                log.err("comptime evaluation produced non-constant\n", .{});
                if (self.error_handler) |handler| {
                    handler.reportError(.InternalError, comptime_expr.span, "comptime evaluation produced non-constant", null) catch {};
                }
                return self.createErrorPlaceholder(comptime_expr.span, "comptime evaluation produced non-constant");
            },
            .err => |err| {
                log.err("comptime evaluation failed: {s}\n", .{@tagName(err.kind)});
                if (self.error_handler) |handler| {
                    handler.reportError(.InternalError, comptime_expr.span, "comptime evaluation error", null) catch {};
                }
                return self.createErrorPlaceholder(comptime_expr.span, "comptime evaluation error");
            },
        }
    }

    if (self.error_handler) |handler| {
        handler.reportError(
            .InternalError,
            comptime_expr.span,
            "Unhandled comptime evaluation result during MLIR lowering",
            "Fix comptime evaluator result handling instead of emitting a placeholder constant.",
        ) catch {};
    }
    return self.createErrorPlaceholder(comptime_expr.span, "unhandled comptime evaluation result");
}

fn getExpressionTypeInfo(expr: *const lib.ast.Expressions.ExprNode) ?lib.ast.type_info.TypeInfo {
    return switch (expr.*) {
        .Identifier => |id| id.type_info,
        .Literal => |lit| switch (lit) {
            .Integer => |int_lit| int_lit.type_info,
            .String => |str_lit| str_lit.type_info,
            .Bool => |bool_lit| bool_lit.type_info,
            .Address => |addr_lit| addr_lit.type_info,
            .Hex => |hex_lit| hex_lit.type_info,
            .Binary => |bin_lit| bin_lit.type_info,
            .Character => |char_lit| char_lit.type_info,
            .Bytes => |bytes_lit| bytes_lit.type_info,
        },
        .Binary => |bin| bin.type_info,
        .Unary => |unary| unary.type_info,
        .Call => |call| call.type_info,
        .FieldAccess => |fa| fa.type_info,
        .Cast => |cast| cast.target_type,
        .ErrorCast => |ecast| ecast.target_type,
        .Index, .Assignment, .CompoundAssignment, .Comptime, .Old, .Tuple, .SwitchExpression, .Quantified, .Try, .ErrorReturn, .Shift, .StructInstantiation, .AnonymousStruct, .Range, .LabeledBlock, .Destructuring, .EnumLiteral, .ArrayLiteral => null,
    };
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
        return createEmptyStruct(self, tuple.span);
    }

    var fields = std.ArrayList(lib.ast.Expressions.AnonymousStructField){};
    defer fields.deinit(std.heap.page_allocator);
    var field_names = std.ArrayList([]const u8){};
    defer {
        for (field_names.items) |name| std.heap.page_allocator.free(name);
        field_names.deinit(std.heap.page_allocator);
    }

    for (tuple.elements, 0..) |element, i| {
        // Tuple fields use numeric names ("0", "1", ...) to align with t.0 syntax.
        const field_name = std.fmt.allocPrint(std.heap.page_allocator, "{d}", .{i}) catch {
            return self.createErrorPlaceholder(tuple.span, "Failed to allocate tuple field name");
        };
        field_names.append(std.heap.page_allocator, field_name) catch {};
        fields.append(std.heap.page_allocator, .{
            .name = field_name,
            .value = element,
            .span = tuple.span,
        }) catch {};
    }

    return createInitializedStruct(self, fields.items, tuple.span);
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
        case_expr_lowerer.refinement_base_cache = self.refinement_base_cache;
        case_expr_lowerer.refinement_guard_cache = self.refinement_guard_cache;
        case_expr_lowerer.current_function_return_type = self.current_function_return_type;
        case_expr_lowerer.current_function_return_type_info = self.current_function_return_type_info;
        case_expr_lowerer.in_try_block = self.in_try_block;
        case_expr_lowerer.module_exports = self.module_exports;

        switch (case.pattern) {
            .Literal => |lit| {
                _ = case_expr_lowerer.lowerLiteral(&lit.value);
                if (expr_literals.extractIntegerFromLiteral(&lit.value)) |val| {
                    case_values.append(std.heap.page_allocator, val) catch {};
                    range_starts.append(std.heap.page_allocator, 0) catch {};
                    range_ends.append(std.heap.page_allocator, 0) catch {};
                    case_kinds.append(std.heap.page_allocator, 0) catch {};
                } else {
                    return reportSwitchPatternLoweringError(
                        self,
                        switch_expr.span,
                        "Switch literal case must be an integer constant",
                        "Use an integer literal (or enum value) as a switch case pattern.",
                    );
                }
            },
            .Range => |range| {
                _ = case_expr_lowerer.lowerExpression(range.start);
                _ = case_expr_lowerer.lowerExpression(range.end);
                const start_val = expr_literals.extractIntegerFromExpr(range.start) orelse {
                    return reportSwitchPatternLoweringError(
                        self,
                        switch_expr.span,
                        "Switch range start must be an integer constant",
                        "Use constant integer bounds in switch range patterns.",
                    );
                };
                const end_val = expr_literals.extractIntegerFromExpr(range.end) orelse {
                    return reportSwitchPatternLoweringError(
                        self,
                        switch_expr.span,
                        "Switch range end must be an integer constant",
                        "Use constant integer bounds in switch range patterns.",
                    );
                };
                case_values.append(std.heap.page_allocator, 0) catch {};
                range_starts.append(std.heap.page_allocator, start_val) catch {};
                range_ends.append(std.heap.page_allocator, end_val) catch {};
                case_kinds.append(std.heap.page_allocator, 1) catch {};
            },
            .EnumValue => |enum_val| {
                var case_value: i64 = 0;
                var resolved = false;
                if (enum_val.enum_name.len == 0) {
                    if (self.symbol_table) |st| {
                        if (st.getErrorId(enum_val.variant_name)) |err_id| {
                            case_value = @intCast(err_id);
                            resolved = true;
                        }
                    }
                } else if (self.symbol_table) |st| {
                    if (st.lookupType(enum_val.enum_name)) |enum_type| {
                        if (enum_type.getVariantIndex(enum_val.variant_name)) |variant_idx| {
                            case_value = @intCast(variant_idx);
                            resolved = true;
                        }
                    }
                }
                if (!resolved) {
                    return reportSwitchPatternLoweringError(
                        self,
                        switch_expr.span,
                        "Unable to resolve enum switch case to an integer tag",
                        "Ensure the enum/value exists and is type-resolved before lowering.",
                    );
                }
                case_values.append(std.heap.page_allocator, case_value) catch {};
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
                    var temp_lowerer = StatementLowerer.init(self.ctx, case_block, self.type_mapper, &case_expr_lowerer, self.param_map, self.storage_map, @constCast(self.local_var_map), self.locations, @constCast(self.symbol_table), self.builtin_registry, std.heap.page_allocator, self.refinement_guard_cache, result_type, null, self.ora_dialect, &[_]*lib.ast.Expressions.ExprNode{});
                    for (block.statements) |stmt| {
                        switch (stmt) {
                            .Return => |ret| {
                                if (ret.value) |e| {
                                    const v = case_expr_lowerer.lowerExpression(&e);
                                    const yield_op = self.ora_dialect.createYield(&[_]c.MlirValue{v}, loc);
                                    h.appendOp(case_block, yield_op);
                                } else {
                                    const default_val = case_expr_lowerer.createDefaultValueForType(result_type, loc) catch return condition;
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
                                    const default_val = case_expr_lowerer.createDefaultValueForType(result_type, loc) catch return condition;
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
                                    const default_val = case_expr_lowerer.createDefaultValueForType(result_type, loc) catch return condition;
                                    const yield_op = self.ora_dialect.createYield(&[_]c.MlirValue{default_val}, loc);
                                    h.appendOp(case_block, yield_op);
                                }
                                break;
                            },
                            else => {
                                _ = temp_lowerer.lowerStatement(&stmt) catch null;
                            },
                        }
                    }
                } else {
                    var temp_lowerer = StatementLowerer.init(self.ctx, case_block, self.type_mapper, &case_expr_lowerer, self.param_map, self.storage_map, @constCast(self.local_var_map), self.locations, @constCast(self.symbol_table), self.builtin_registry, std.heap.page_allocator, self.refinement_guard_cache, result_type, null, self.ora_dialect, &[_]*lib.ast.Expressions.ExprNode{});
                    _ = temp_lowerer.lowerBlockBody(block, case_block) catch false;
                    const default_val = case_expr_lowerer.createDefaultValueForType(result_type, loc) catch return condition;
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
                    var temp_lowerer = StatementLowerer.init(self.ctx, case_block, self.type_mapper, &case_expr_lowerer, self.param_map, self.storage_map, @constCast(self.local_var_map), self.locations, @constCast(self.symbol_table), self.builtin_registry, std.heap.page_allocator, self.refinement_guard_cache, result_type, null, self.ora_dialect, &[_]*lib.ast.Expressions.ExprNode{});
                    for (labeled.block.statements) |stmt| {
                        switch (stmt) {
                            .Return => |ret| {
                                if (ret.value) |e| {
                                    const v = case_expr_lowerer.lowerExpression(&e);
                                    const yield_op = self.ora_dialect.createYield(&[_]c.MlirValue{v}, loc);
                                    h.appendOp(case_block, yield_op);
                                } else {
                                    const default_val = case_expr_lowerer.createDefaultValueForType(result_type, loc) catch return condition;
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
                                    const default_val = case_expr_lowerer.createDefaultValueForType(result_type, loc) catch return condition;
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
                                    const default_val = case_expr_lowerer.createDefaultValueForType(result_type, loc) catch return condition;
                                    const yield_op = self.ora_dialect.createYield(&[_]c.MlirValue{default_val}, loc);
                                    h.appendOp(case_block, yield_op);
                                }
                                break;
                            },
                            else => {
                                _ = temp_lowerer.lowerStatement(&stmt) catch null;
                            },
                        }
                    }
                } else {
                    var temp_lowerer = StatementLowerer.init(self.ctx, case_block, self.type_mapper, &case_expr_lowerer, self.param_map, self.storage_map, @constCast(self.local_var_map), self.locations, @constCast(self.symbol_table), self.builtin_registry, std.heap.page_allocator, self.refinement_guard_cache, result_type, null, self.ora_dialect, &[_]*lib.ast.Expressions.ExprNode{});
                    _ = temp_lowerer.lowerBlockBody(labeled.block, case_block) catch false;
                    const default_val = case_expr_lowerer.createDefaultValueForType(result_type, loc) catch return condition;
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
        default_expr_lowerer.refinement_base_cache = self.refinement_base_cache;
        default_expr_lowerer.refinement_guard_cache = self.refinement_guard_cache;
        default_expr_lowerer.current_function_return_type = self.current_function_return_type;
        default_expr_lowerer.current_function_return_type_info = self.current_function_return_type_info;
        default_expr_lowerer.in_try_block = self.in_try_block;
        default_expr_lowerer.module_exports = self.module_exports;

        var default_has_return = false;
        for (default_block.statements) |stmt| {
            if (stmt == .Return) {
                default_has_return = true;
                break;
            }
        }
        if (default_has_return) {
            var temp_lowerer = StatementLowerer.init(self.ctx, default_block_mlir, self.type_mapper, &default_expr_lowerer, self.param_map, self.storage_map, @constCast(self.local_var_map), self.locations, @constCast(self.symbol_table), self.builtin_registry, std.heap.page_allocator, self.refinement_guard_cache, result_type, null, self.ora_dialect, &[_]*lib.ast.Expressions.ExprNode{});
            for (default_block.statements) |stmt| {
                switch (stmt) {
                    .Return => |ret| {
                        if (ret.value) |e| {
                            const v = default_expr_lowerer.lowerExpression(&e);
                            const yield_op = self.ora_dialect.createYield(&[_]c.MlirValue{v}, loc);
                            h.appendOp(default_block_mlir, yield_op);
                        } else {
                            const default_val = default_expr_lowerer.createDefaultValueForType(result_type, loc) catch return condition;
                            const yield_op = self.ora_dialect.createYield(&[_]c.MlirValue{default_val}, loc);
                            h.appendOp(default_block_mlir, yield_op);
                        }
                        break;
                    },
                    else => {
                        _ = temp_lowerer.lowerStatement(&stmt) catch null;
                    },
                }
            }
        } else {
            var temp_lowerer = StatementLowerer.init(self.ctx, default_block_mlir, self.type_mapper, &default_expr_lowerer, self.param_map, self.storage_map, @constCast(self.local_var_map), self.locations, @constCast(self.symbol_table), self.builtin_registry, std.heap.page_allocator, self.refinement_guard_cache, result_type, null, self.ora_dialect, &[_]*lib.ast.Expressions.ExprNode{});
            _ = temp_lowerer.lowerBlockBody(default_block, default_block_mlir) catch false;
            const default_val = default_expr_lowerer.createDefaultValueForType(result_type, loc) catch return condition;
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

    h.appendOp(self.block, switch_op);

    return h.getResult(switch_op, 0);
}

/// Lower quantified expressions
pub fn lowerQuantified(
    self: *const ExpressionLowerer,
    quantified: *const lib.ast.Expressions.QuantifiedExpr,
) c.MlirValue {
    const result_ty = h.boolType(self.ctx);
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
    const variable_type_info = if (quantified.verification_metadata) |metadata|
        metadata.variable_type
    else
        quantified.variable_type;
    const var_type_str = if (quantified.verification_metadata) |metadata|
        getTypeString(self, metadata.variable_type)
    else
        getTypeString(self, quantified.variable_type);

    // Bind the quantified variable as a scoped symbolic constant so references
    // in the condition/body resolve to the quantified value.
    const variable_mlir_type = self.type_mapper.toMlirType(variable_type_info);
    const bound_var_name_id = h.identifier(self.ctx, "ora.bound_variable");
    const bound_var_name_attr = h.stringAttr(self.ctx, variable_name);
    const placeholder_attrs = [_]c.MlirNamedAttribute{
        c.oraNamedAttributeGet(bound_var_name_id, bound_var_name_attr),
    };
    const placeholder_op = self.ora_dialect.createArithConstantWithAttrs(
        0,
        variable_mlir_type,
        &placeholder_attrs,
        self.fileLoc(quantified.span),
    );
    h.appendOp(self.block, placeholder_op);
    const placeholder_value = h.getResult(placeholder_op, 0);

    var scoped_local_vars = LocalVarMap.init(std.heap.page_allocator);
    defer scoped_local_vars.deinit();
    if (self.local_var_map) |existing_locals| {
        var vars_it = existing_locals.variables.iterator();
        while (vars_it.next()) |entry| {
            scoped_local_vars.addLocalVar(entry.key_ptr.*, entry.value_ptr.*) catch {};
        }
        var kinds_it = existing_locals.kinds.iterator();
        while (kinds_it.next()) |entry| {
            scoped_local_vars.setLocalVarKind(entry.key_ptr.*, entry.value_ptr.*) catch {};
        }
    }
    scoped_local_vars.addLocalVar(variable_name, placeholder_value) catch {};

    var quantified_lowerer = self.*;
    quantified_lowerer.local_var_map = &scoped_local_vars;

    const body_value = quantified_lowerer.lowerExpression(quantified.body);
    var condition_value: ?c.MlirValue = null;
    if (quantified.condition) |condition| {
        condition_value = quantified_lowerer.lowerExpression(condition);
    }

    var attributes = std.ArrayList(c.MlirNamedAttribute){};
    defer attributes.deinit(std.heap.page_allocator);

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

    if (should_propagate and !c.oraTypeIsNull(success_ty)) {
        const is_err_op = self.ora_dialect.createErrorIsError(expr_value, loc);
        h.appendOp(self.block, is_err_op);
        const is_err_val = h.getResult(is_err_op, 0);

        const if_op = self.ora_dialect.createIf(is_err_val, loc);
        h.appendOp(self.block, if_op);

        const then_block = c.oraIfOpGetThenBlock(if_op);
        const else_block = c.oraIfOpGetElseBlock(if_op);
        if (c.oraBlockIsNull(then_block) or c.oraBlockIsNull(else_block)) {
            @panic("ora.if missing then/else blocks");
        }

        const err_ty = c.oraIntegerTypeCreate(self.ctx, constants.DEFAULT_INTEGER_BITS);
        const get_err_op = self.ora_dialect.createErrorGetError(expr_value, err_ty, loc);
        h.appendOp(then_block, get_err_op);
        const err_val = h.getResult(get_err_op, 0);

        const err_union_op = self.ora_dialect.createErrorErr(err_val, expr_ty, loc);
        h.appendOp(then_block, err_union_op);
        const err_union_val = h.getResult(err_union_op, 0);

        const return_op = self.ora_dialect.createFuncReturnWithValue(err_union_val, loc);
        h.appendOp(then_block, return_op);

        const else_yield_op = self.ora_dialect.createYield(&[_]c.MlirValue{}, loc);
        h.appendOp(else_block, else_yield_op);

        const unwrap_op = self.ora_dialect.createErrorUnwrap(expr_value, result_type, loc);
        h.appendOp(self.block, unwrap_op);
        return h.getResult(unwrap_op, 0);
    }

    // use ora.try_stmt with explicit try/catch regions
    const op = self.ora_dialect.createTryStmt(&[_]c.MlirType{result_type}, loc);
    h.appendOp(self.block, op);

    // populate try region with the expression value
    const try_block = c.oraTryStmtOpGetTryBlock(op);
    if (!c.oraBlockIsNull(try_block)) {
        const first_op = c.oraBlockGetFirstOperation(try_block);
        if (c.oraOperationIsNull(first_op)) {
            const unwrap_op = self.ora_dialect.createErrorUnwrap(expr_value, result_type, loc);
            h.appendOp(try_block, unwrap_op);
            const unwrapped = h.getResult(unwrap_op, 0);
            const yield_op = self.ora_dialect.createYield(&[_]c.MlirValue{unwrapped}, loc);
            h.appendOp(try_block, yield_op);
        }
    }

    // ensure the catch region has a terminator to avoid empty block errors
    const catch_block = c.oraTryStmtOpGetCatchBlock(op);
    if (!c.oraBlockIsNull(catch_block)) {
        const first_op = c.oraBlockGetFirstOperation(catch_block);
        if (c.oraOperationIsNull(first_op)) {
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
    const error_code: i64 = if (self.symbol_table) |st|
        @intCast(st.getErrorId(error_ret.error_name) orelse 1)
    else
        1;
    const error_id = h.identifier(self.ctx, "ora.error");
    const error_name_attr = h.stringAttr(self.ctx, error_ret.error_name);

    const attrs = [_]c.MlirNamedAttribute{
        c.oraNamedAttributeGet(error_id, error_name_attr),
    };
    const op = self.ora_dialect.createArithConstantWithAttrs(error_code, ty, &attrs, self.fileLoc(error_ret.span));
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

    // Check if this is a bitfield construction → emit shift/OR chain
    if (self.symbol_table) |st| {
        if (st.lookupType(struct_name_str)) |type_sym| {
            if (type_sym.type_kind == .Bitfield) {
                return lowerBitfieldConstruction(self, struct_inst, type_sym);
            }
        }
    }

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

/// Lower bitfield construction: fold field values into a packed integer via OR/SHL.
/// Result starts at 0 and each specified field is masked, shifted, and OR'd in.
fn lowerBitfieldConstruction(
    self: *const ExpressionLowerer,
    struct_inst: *const lib.ast.Expressions.StructInstantiationExpr,
    type_sym: *const constants.TypeSymbol,
) c.MlirValue {
    const loc = self.fileLoc(struct_inst.span);
    const int_ty = c.oraIntegerTypeCreate(self.ctx, constants.DEFAULT_INTEGER_BITS);

    // Start with zero
    const zero_op = self.ora_dialect.createArithConstant(0, int_ty, loc);
    h.appendOp(self.block, zero_op);
    var word = h.getResult(zero_op, 0);

    const fields_info = type_sym.fields orelse return word;

    for (struct_inst.fields) |init_field| {
        const field_val = self.lowerExpression(init_field.value);
        const field_uw = expr_helpers.unwrapRefinementValue(self.ctx, self.block, self.locations, self.refinement_base_cache, field_val, init_field.span);
        const field_i256 = coerceBitfieldFieldToWord(self, field_uw, int_ty, init_field.span);

        // Find matching field layout
        for (fields_info) |info| {
            if (!std.mem.eql(u8, info.name, init_field.name)) continue;

            const offset: i64 = if (info.offset) |o| @intCast(o) else 0;
            const width: u32 = info.bit_width orelse 256;
            const mask: i64 = if (width >= 64) -1 else (@as(i64, 1) << @intCast(width)) - 1;

            // masked = field_val & mask
            const mask_op = self.ora_dialect.createArithConstant(mask, int_ty, loc);
            h.appendOp(self.block, mask_op);
            const masked = c.oraArithAndIOpCreate(self.ctx, loc, field_i256, h.getResult(mask_op, 0));
            h.appendOp(self.block, masked);

            // shifted = masked << offset
            const off_op = self.ora_dialect.createArithConstant(offset, int_ty, loc);
            h.appendOp(self.block, off_op);
            const shifted = c.oraArithShlIOpCreate(self.ctx, loc, h.getResult(masked, 0), h.getResult(off_op, 0));
            h.appendOp(self.block, shifted);

            // word = word | shifted
            const or_op = c.oraArithOrIOpCreate(self.ctx, loc, word, h.getResult(shifted, 0));
            h.appendOp(self.block, or_op);
            word = h.getResult(or_op, 0);
            break;
        }
    }

    return word;
}

fn lowerAnonymousBitfieldConstruction(
    self: *const ExpressionLowerer,
    fields: []const lib.ast.Expressions.AnonymousStructField,
    span: lib.ast.SourceSpan,
    type_sym: *const constants.TypeSymbol,
) c.MlirValue {
    const loc = self.fileLoc(span);
    const int_ty = c.oraIntegerTypeCreate(self.ctx, constants.DEFAULT_INTEGER_BITS);

    const zero_op = self.ora_dialect.createArithConstant(0, int_ty, loc);
    h.appendOp(self.block, zero_op);
    var word = h.getResult(zero_op, 0);

    const fields_info = type_sym.fields orelse return word;

    for (fields) |init_field| {
        const field_val = self.lowerExpression(init_field.value);
        const field_uw = expr_helpers.unwrapRefinementValue(self.ctx, self.block, self.locations, self.refinement_base_cache, field_val, init_field.span);
        const field_i256 = coerceBitfieldFieldToWord(self, field_uw, int_ty, init_field.span);

        for (fields_info) |info| {
            if (!std.mem.eql(u8, info.name, init_field.name)) continue;

            const offset: i64 = if (info.offset) |o| @intCast(o) else 0;
            const width: u32 = info.bit_width orelse 256;
            const mask: i64 = if (width >= 64) -1 else (@as(i64, 1) << @intCast(width)) - 1;

            const mask_op = self.ora_dialect.createArithConstant(mask, int_ty, loc);
            h.appendOp(self.block, mask_op);
            const masked = c.oraArithAndIOpCreate(self.ctx, loc, field_i256, h.getResult(mask_op, 0));
            h.appendOp(self.block, masked);

            const off_op = self.ora_dialect.createArithConstant(offset, int_ty, loc);
            h.appendOp(self.block, off_op);
            const shifted = c.oraArithShlIOpCreate(self.ctx, loc, h.getResult(masked, 0), h.getResult(off_op, 0));
            h.appendOp(self.block, shifted);

            const or_op = c.oraArithOrIOpCreate(self.ctx, loc, word, h.getResult(shifted, 0));
            h.appendOp(self.block, or_op);
            word = h.getResult(or_op, 0);
            break;
        }
    }

    return word;
}

fn coerceBitfieldFieldToWord(
    self: *const ExpressionLowerer,
    value: c.MlirValue,
    word_ty: c.MlirType,
    span: lib.ast.SourceSpan,
) c.MlirValue {
    const value_ty = c.oraValueGetType(value);
    if (c.oraTypeEqual(value_ty, word_ty)) return value;

    const loc = self.fileLoc(span);

    // Bool fields lower as i1 and must be widened before mask/shift in i256 domain.
    if (c.oraTypeEqual(value_ty, h.boolType(self.ctx))) {
        const ext_bool = c.oraArithExtUIOpCreate(self.ctx, loc, value, word_ty);
        h.appendOp(self.block, ext_bool);
        return h.getResult(ext_bool, 0);
    }

    if (c.oraTypeIsAInteger(value_ty)) {
        const value_width = c.oraIntegerTypeGetWidth(value_ty);
        const word_width = c.oraIntegerTypeGetWidth(word_ty);

        if (value_width < word_width) {
            const ext_op = c.oraArithExtUIOpCreate(self.ctx, loc, value, word_ty);
            h.appendOp(self.block, ext_op);
            return h.getResult(ext_op, 0);
        }
        if (value_width > word_width) {
            const trunc_op = c.oraArithTruncIOpCreate(self.ctx, loc, value, word_ty);
            h.appendOp(self.block, trunc_op);
            return h.getResult(trunc_op, 0);
        }

        const cast_op = c.oraArithBitcastOpCreate(self.ctx, loc, value, word_ty);
        h.appendOp(self.block, cast_op);
        return h.getResult(cast_op, 0);
    }

    return self.convertToType(value, word_ty, span);
}

fn matchesBitfieldFields(
    init_fields: []const lib.ast.Expressions.AnonymousStructField,
    bitfield_fields: []const constants.TypeSymbol.FieldInfo,
) bool {
    if (init_fields.len != bitfield_fields.len) return false;

    for (init_fields) |init_field| {
        var found = false;
        for (bitfield_fields) |bf_field| {
            if (std.mem.eql(u8, init_field.name, bf_field.name)) {
                found = true;
                break;
            }
        }
        if (!found) return false;
    }

    return true;
}

fn inferBitfieldFromAnonymousFields(
    self: *const ExpressionLowerer,
    fields: []const lib.ast.Expressions.AnonymousStructField,
) ?*const constants.TypeSymbol {
    if (self.symbol_table == null or fields.len == 0) return null;

    var match: ?*const constants.TypeSymbol = null;
    var type_iter = self.symbol_table.?.types.iterator();
    while (type_iter.next()) |entry| {
        const type_symbols = entry.value_ptr.*;
        for (type_symbols) |*type_sym| {
            if (type_sym.type_kind != .Bitfield) continue;
            const bitfield_fields = type_sym.fields orelse continue;
            if (!matchesBitfieldFields(fields, bitfield_fields)) continue;

            // Ambiguous shape: fall back to regular anonymous struct lowering.
            if (match != null) return null;
            match = type_sym;
        }
    }

    return match;
}

/// Lower anonymous struct expressions
pub fn lowerAnonymousStruct(
    self: *const ExpressionLowerer,
    anon_struct: *const lib.ast.Expressions.AnonymousStructExpr,
) c.MlirValue {
    if (self.expected_type_info) |expected| {
        if (expected.ora_type) |ora_type| {
            if (self.symbol_table) |st| switch (ora_type) {
                .bitfield_type => |bf_name| {
                    if (st.lookupType(bf_name)) |type_sym| {
                        if (type_sym.type_kind == .Bitfield) {
                            return lowerAnonymousBitfieldConstruction(self, anon_struct.fields, anon_struct.span, type_sym);
                        }
                    }
                },
                .struct_type => |type_name| {
                    if (st.lookupType(type_name)) |type_sym| {
                        if (type_sym.type_kind == .Bitfield) {
                            return lowerAnonymousBitfieldConstruction(self, anon_struct.fields, anon_struct.span, type_sym);
                        }
                    }
                },
                else => {},
            };
        }
    }

    if (inferBitfieldFromAnonymousFields(self, anon_struct.fields)) |type_sym| {
        if (type_sym.type_kind == .Bitfield) {
            return lowerAnonymousBitfieldConstruction(self, anon_struct.fields, anon_struct.span, type_sym);
        }
    }

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

    const stmt_lowerer = StatementLowerer.init(self.ctx, block, self.type_mapper, self, self.param_map, self.storage_map, @constCast(self.local_var_map), self.locations, null, self.builtin_registry, std.heap.page_allocator, self.refinement_guard_cache, null, null, self.ora_dialect, &[_]*lib.ast.Expressions.ExprNode{});

    for (labeled_block.block.statements) |stmt| {
        _ = stmt_lowerer.lowerStatement(&stmt) catch |err| {
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

                // Check if enum_ty is an !ora.enum<...> type and extract repr type
                if (c.oraTypeIsAEnum(enum_ty)) {
                    const repr_type = c.oraEnumTypeGetReprType(enum_ty);
                    if (repr_type.ptr == null) {
                        @panic("Failed to extract repr type from enum");
                    }
                    enum_ty = repr_type;
                }

                // Handle integer enum types
                if (c.oraTypeIsAInteger(enum_ty) or c.oraTypeIsAOraInteger(enum_ty)) {
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

                    const enum_id = h.identifier(self.ctx, "ora.enum");
                    const enum_name_attr = h.stringAttr(self.ctx, enum_lit.enum_name);

                    const attrs = [_]c.MlirNamedAttribute{
                        c.oraNamedAttributeGet(enum_id, enum_name_attr),
                    };
                    const op = self.ora_dialect.createArithConstantWithAttrs(enum_value, enum_ty, &attrs, self.fileLoc(enum_lit.span));
                    h.appendOp(self.block, op);
                    return h.getResult(op, 0);
                }

                // Handle string enum types
                const string_ty = c.oraStringTypeGet(self.ctx);
                if (c.oraTypeEqual(enum_ty, string_ty)) {
                    var string_value: ?[]const u8 = null;

                    if (type_sym.variants) |variants| {
                        for (variants) |variant| {
                            if (std.mem.eql(u8, variant.name, enum_lit.variant_name)) {
                                if (variant.string_value) |str_val| {
                                    string_value = str_val;
                                }
                                break;
                            }
                        }
                    }

                    if (string_value == null) {
                        @panic("String enum variant value not found");
                    }

                    const enum_id = h.identifier(self.ctx, "ora.enum");
                    const enum_name_attr = h.stringAttr(self.ctx, enum_lit.enum_name);

                    const attrs = [_]c.MlirNamedAttribute{
                        c.oraNamedAttributeGet(enum_id, enum_name_attr),
                    };

                    const loc = self.fileLoc(enum_lit.span);
                    const op = self.ora_dialect.createStringConstant(string_value.?, enum_ty, loc);
                    for (attrs) |custom_attr| {
                        c.oraOperationSetAttributeByName(op, c.oraIdentifierStr(custom_attr.name), custom_attr.attribute);
                    }
                    h.appendOp(self.block, op);
                    return h.getResult(op, 0);
                }

                // Other non-integer enum types are not yet supported
                @panic("Enum with non-integer underlying type is not yet supported");
            }
        }
    }

    // Fallback: reinterpret as field access (e.g., f.mode on a bitfield/struct variable)
    {
        var ident_expr = lib.ast.Expressions.ExprNode{ .Identifier = lib.ast.Expressions.IdentifierExpr{
            .name = enum_lit.enum_name,
            .type_info = lib.ast.Types.TypeInfo.unknown(),
            .span = enum_lit.span,
        } };
        var field_expr = lib.ast.Expressions.FieldAccessExpr{
            .target = &ident_expr,
            .field = enum_lit.variant_name,
            .type_info = lib.ast.Types.TypeInfo.unknown(),
            .span = enum_lit.span,
        };
        return @import("access.zig").lowerFieldAccess(self, &field_expr);
    }
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
    if (self.error_handler) |handler| {
        handler.reportError(
            .TypeMismatch,
            null,
            "Cannot synthesize expression default value for non-integer MLIR type",
            "Add explicit default-value lowering for this MLIR type.",
        ) catch {};
    } else {
        log.err("Cannot synthesize expression default value for non-integer MLIR type\n", .{});
    }
    return error.TypeMismatch;
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
    const struct_ty = blk: {
        if (self.current_function_return_type_info) |ret_info| {
            if (ret_info.ora_type) |ora_type| {
                switch (ora_type) {
                    .anonymous_struct => |ret_fields| {
                        if (matchesAnonymousStructFields(fields, ret_fields)) {
                            break :blk self.type_mapper.mapAnonymousStructType(ret_fields);
                        }
                    },
                    else => {},
                }
            }
        }
        if (deriveAnonymousStructFields(fields)) |derived_fields| {
            defer std.heap.page_allocator.free(derived_fields.types);
            defer std.heap.page_allocator.free(derived_fields.fields);
            break :blk self.type_mapper.mapAnonymousStructType(derived_fields.fields);
        }
        if (self.error_handler) |handler| {
            handler.reportError(
                .TypeMismatch,
                span,
                "Failed to infer anonymous struct type during MLIR lowering",
                "Provide explicit type information for anonymous struct fields.",
            ) catch {};
        }
        return self.createErrorPlaceholder(span, "failed to infer anonymous struct type");
    };

    if (c.oraTypeIsNull(struct_ty)) {
        return self.createErrorPlaceholder(span, "invalid anonymous struct type");
    }

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

fn matchesAnonymousStructFields(
    fields: []lib.ast.Expressions.AnonymousStructField,
    type_fields: []const lib.ast.type_info.AnonymousStructFieldType,
) bool {
    if (fields.len != type_fields.len) return false;
    for (fields, 0..) |field, i| {
        if (!std.mem.eql(u8, field.name, type_fields[i].name)) return false;
    }
    return true;
}

const DerivedAnonymousFields = struct {
    fields: []lib.ast.type_info.AnonymousStructFieldType,
    types: []lib.ast.type_info.OraType,
};

fn deriveAnonymousStructFields(
    fields: []lib.ast.Expressions.AnonymousStructField,
) ?DerivedAnonymousFields {
    if (fields.len == 0) return null;
    const types = std.heap.page_allocator.alloc(lib.ast.type_info.OraType, fields.len) catch return null;
    const anon_fields = std.heap.page_allocator.alloc(lib.ast.type_info.AnonymousStructFieldType, fields.len) catch {
        std.heap.page_allocator.free(types);
        return null;
    };

    for (fields, 0..) |field, i| {
        const type_info = getExprTypeInfo(field.value) orelse {
            std.heap.page_allocator.free(types);
            std.heap.page_allocator.free(anon_fields);
            return null;
        };
        if (type_info.ora_type) |ora_type| {
            types[i] = ora_type;
            anon_fields[i] = .{ .name = field.name, .typ = &types[i] };
            continue;
        }
        std.heap.page_allocator.free(types);
        std.heap.page_allocator.free(anon_fields);
        return null;
    }

    return .{ .fields = anon_fields, .types = types };
}

fn inferTupleTypeInfo(tuple_expr: *const lib.ast.Expressions.TupleExpr) ?lib.ast.Types.TypeInfo {
    const tuple_elems = std.heap.page_allocator.alloc(lib.ast.type_info.OraType, tuple_expr.elements.len) catch return null;

    for (tuple_expr.elements, 0..) |elem, i| {
        const elem_ti = getExprTypeInfo(elem) orelse {
            std.heap.page_allocator.free(tuple_elems);
            return null;
        };

        if (elem_ti.ora_type) |elem_ora| {
            tuple_elems[i] = elem_ora;
            continue;
        }

        std.heap.page_allocator.free(tuple_elems);
        return null;
    }

    return lib.ast.Types.TypeInfo.fromOraType(.{ .tuple = tuple_elems });
}

fn getExprTypeInfo(expr: *const lib.ast.Expressions.ExprNode) ?lib.ast.Types.TypeInfo {
    return switch (expr.*) {
        .Identifier => |id| id.type_info,
        .Literal => |lit| switch (lit) {
            .Integer => |int_lit| int_lit.type_info,
            .String => |str_lit| str_lit.type_info,
            .Bool => |bool_lit| bool_lit.type_info,
            .Address => |addr_lit| addr_lit.type_info,
            .Hex => |hex_lit| hex_lit.type_info,
            .Binary => |bin_lit| bin_lit.type_info,
            .Character => |char_lit| char_lit.type_info,
            .Bytes => |bytes_lit| bytes_lit.type_info,
        },
        .Binary => |bin| bin.type_info,
        .Unary => |unary| unary.type_info,
        .Call => |call| call.type_info,
        .FieldAccess => |fa| fa.type_info,
        .Range => |range| range.type_info,
        .Tuple => |tuple_expr| inferTupleTypeInfo(&tuple_expr),
        else => null,
    };
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
            .bitfield_type => "bitfield",
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
            .type => "type",
            .type_parameter => "type_parameter",
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
    _ = condition;
    _ = cases;
    return reportSwitchPatternLoweringError(
        self,
        span,
        "createSwitchIfChain is not implemented",
        "Use switch expression lowering via ora.switch_expr until this path is implemented.",
    );
}
