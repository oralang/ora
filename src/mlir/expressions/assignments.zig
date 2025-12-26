// ============================================================================
// Assignment Expression Lowering
// ============================================================================
// Lowering for assignments and compound assignments

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
const expr_operators = @import("operators.zig");
const log = @import("log");

/// ExpressionLowerer type (forward declaration)
const ExpressionLowerer = @import("mod.zig").ExpressionLowerer;

/// LValue mode for load/store operations
pub const LValueMode = enum {
    Load,
    Store,
};

/// Lower assignment expressions
pub fn lowerAssignment(
    self: *const ExpressionLowerer,
    assign: *const lib.ast.Expressions.AssignmentExpr,
) c.MlirValue {
    const value = self.lowerExpression(assign.value);

    switch (assign.target.*) {
        .Identifier => |ident| {
            if (self.local_var_map) |lvm| {
                if (lvm.getLocalVar(ident.name)) |local_var_ref| {
                    const var_type = c.mlirValueGetType(local_var_ref);
                    var store_value = value;
                    if (c.mlirTypeIsAMemRef(var_type)) {
                        const element_type = c.mlirShapedTypeGetElementType(var_type);
                        const value_type = c.mlirValueGetType(value);
                        log.debug("[ASSIGN Expression] Variable: {s}, value_type != element_type: {}, value_is_ora: {}, element_is_ora: {}\n", .{ ident.name, !c.mlirTypeEqual(value_type, element_type), c.oraTypeIsIntegerType(value_type), c.oraTypeIsIntegerType(element_type) });
                        if (!c.mlirTypeEqual(value_type, element_type)) {
                            log.debug("[ASSIGN Expression] Converting (types not equal)\n", .{});
                            store_value = self.convertToType(value, element_type, assign.span);
                        } else if (c.oraTypeIsIntegerType(value_type) and c.oraTypeIsIntegerType(element_type)) {
                            log.debug("[ASSIGN Expression] Converting (Ora types, explicit cast)\n", .{});
                            store_value = self.convertToType(value, element_type, assign.span);
                        } else {
                            log.debug("[ASSIGN Expression] No conversion needed\n", .{});
                        }
                        const store_value_type = c.mlirValueGetType(store_value);
                        log.debug("[ASSIGN Expression] After conversion: types_equal: {}\n", .{c.mlirTypeEqual(store_value_type, element_type)});
                    }
                    var store_state = h.opState("memref.store", self.fileLoc(assign.span));
                    c.mlirOperationStateAddOperands(&store_state, 2, @ptrCast(&[_]c.MlirValue{ store_value, local_var_ref }));
                    const store_op = c.mlirOperationCreate(&store_state);
                    h.appendOp(self.block, store_op);
                    return value;
                }
            }

            if (self.storage_map) |sm| {
                if (sm.hasStorageVariable(ident.name)) {
                    const memory_manager = @import("../memory.zig").MemoryManager.init(self.ctx, self.ora_dialect);
                    const store_op = memory_manager.createStorageStore(value, ident.name, self.fileLoc(assign.span));
                    h.appendOp(self.block, store_op);
                    return value;
                }
            }

            const var_type = c.mlirValueGetType(value);
            const memref_type = c.mlirMemRefTypeGet(var_type, 0, null, c.mlirAttributeGetNull(), c.mlirAttributeGetNull());
            const alloca_op = self.ora_dialect.createMemrefAlloca(memref_type, self.fileLoc(assign.span));
            h.appendOp(self.block, alloca_op);
            const alloca_result = h.getResult(alloca_op, 0);

            var store_state = h.opState("memref.store", self.fileLoc(assign.span));
            c.mlirOperationStateAddOperands(&store_state, 2, @ptrCast(&[_]c.MlirValue{ value, alloca_result }));
            const store_op = c.mlirOperationCreate(&store_state);
            h.appendOp(self.block, store_op);

            return value;
        },
        .FieldAccess => |field_access| {
            const target_value = self.lowerExpression(field_access.target);
            const target_type = c.mlirValueGetType(target_value);
            const field_name = field_access.field;

            if (field_access.target.* == .Identifier) {
                const ident = field_access.target.Identifier;
                if (ident.type_info.ora_type) |ora_type| {
                    if (ora_type == .struct_type) {
                        const expected_struct_type = self.type_mapper.toMlirType(ident.type_info);

                        if (c.oraTypeIsAddressType(target_type)) {
                            log.debug(
                                "ERROR [lowerAssignment]: Variable '{s}' is address type but should be struct type for field access\n",
                                .{ident.name},
                            );
                            log.debug(
                                "  This likely means a map load returned !ora.address instead of the struct type\n",
                                .{},
                            );
                            log.debug("  Expected struct type: {any}\n", .{expected_struct_type});
                            return self.reportLoweringError(
                                assign.span,
                                "cannot update field on address type - map load returned wrong type",
                                "check map value type and struct layout for field access",
                            );
                        }

                        if (!c.mlirTypeEqual(target_type, expected_struct_type)) {
                            log.debug(
                                "ERROR [lowerAssignment]: Variable '{s}' should be struct type but got: {any}\n",
                                .{ ident.name, target_type },
                            );
                            log.debug("  Expected struct type: {any}\n", .{expected_struct_type});
                            log.debug(
                                "  This likely means a map load returned wrong type instead of the struct type\n",
                                .{},
                            );
                            return self.reportLoweringError(
                                assign.span,
                                "cannot update field on non-struct type - map load likely returned wrong type",
                                "check map value type and struct layout for field access",
                            );
                        }
                    }
                }
            } else {
                if (c.oraTypeIsAddressType(target_type)) {
                    log.debug(
                        "ERROR [lowerAssignment]: Field access target is address type but should be struct type\n",
                        .{},
                    );
                    log.debug(
                        "  This likely means a map load returned !ora.address instead of the struct type\n",
                        .{},
                    );
                    return self.reportLoweringError(
                        assign.span,
                        "cannot update field on address type - map load returned wrong type",
                        "check map value type and struct layout for field access",
                    );
                }
            }

            const update_op = self.ora_dialect.createStructFieldUpdate(target_value, field_name, value, self.fileLoc(assign.span));
            h.appendOp(self.block, update_op);
            const updated_struct = h.getResult(update_op, 0);

            return updated_struct;
        },
        .Index => |index_expr| {
            const target_value = self.lowerExpression(index_expr.target);
            const index_value = self.lowerExpression(index_expr.index);
            const target_type = c.mlirValueGetType(target_value);

            if (c.mlirTypeIsAMemRef(target_type) or c.mlirTypeIsAShaped(target_type)) {
                const index_index = expr_access.convertIndexToIndexType(self, index_value, index_expr.span);
                var store_state = h.opState("memref.store", self.fileLoc(assign.span));
                c.mlirOperationStateAddOperands(&store_state, 1, @ptrCast(&value));
                c.mlirOperationStateAddOperands(&store_state, 1, @ptrCast(&target_value));
                c.mlirOperationStateAddOperands(&store_state, 1, @ptrCast(&index_index));
                const store_op = c.mlirOperationCreate(&store_state);
                h.appendOp(self.block, store_op);
            } else {
                const op = self.ora_dialect.createMapStore(target_value, index_value, value, self.fileLoc(assign.span));
                h.appendOp(self.block, op);
            }
            return value;
        },
        else => {
            log.err("Invalid assignment target\n", .{});
            return value;
        },
    }
}

/// Lower compound assignment expressions
pub fn lowerCompoundAssignment(
    self: *const ExpressionLowerer,
    comp_assign: *const lib.ast.Expressions.CompoundAssignmentExpr,
) c.MlirValue {
    const current_value = lowerLValue(self, comp_assign.target, .Load);
    const rhs_value = self.lowerExpression(comp_assign.value);

    const current_ty = c.mlirValueGetType(current_value);
    const rhs_ty = c.mlirValueGetType(rhs_value);
    const common_ty = self.getCommonType(current_ty, rhs_ty);

    const current_converted = self.convertToType(current_value, common_ty, comp_assign.span);
    const rhs_converted = self.convertToType(rhs_value, common_ty, comp_assign.span);

    if (comp_assign.operator == .SlashEqual or comp_assign.operator == .PercentEqual) {
        const current_type_info = expr_operators.extractTypeInfo(comp_assign.target);
        const rhs_type_info = expr_operators.extractTypeInfo(comp_assign.value);

        const current_is_exact = if (current_type_info.ora_type) |ora_type|
            ora_type == .exact
        else
            false;

        const rhs_is_exact = if (rhs_type_info.ora_type) |ora_type|
            ora_type == .exact
        else
            false;

        if (current_is_exact or rhs_is_exact) {
            expr_operators.insertExactDivisionGuard(self, current_converted, rhs_converted, comp_assign.span);
        }
    }

    const result_value = switch (comp_assign.operator) {
        .PlusEqual => self.createArithmeticOp("arith.addi", current_converted, rhs_converted, common_ty, comp_assign.span),
        .MinusEqual => self.createArithmeticOp("arith.subi", current_converted, rhs_converted, common_ty, comp_assign.span),
        .StarEqual => self.createArithmeticOp("arith.muli", current_converted, rhs_converted, common_ty, comp_assign.span),
        .SlashEqual => self.createArithmeticOp("arith.divsi", current_converted, rhs_converted, common_ty, comp_assign.span),
        .PercentEqual => self.createArithmeticOp("arith.remsi", current_converted, rhs_converted, common_ty, comp_assign.span),
    };

    storeLValue(self, comp_assign.target, result_value, comp_assign.span);

    return result_value;
}

/// Lower lvalue expression (load or get address)
pub fn lowerLValue(
    self: *const ExpressionLowerer,
    lvalue: *const lib.ast.Expressions.ExprNode,
    mode: LValueMode,
) c.MlirValue {
    return switch (lvalue.*) {
        .Identifier => |ident| blk: {
            if (mode == .Load) {
                break :blk self.lowerIdentifier(&ident);
            } else {
                break :blk self.createErrorPlaceholder(ident.span, "Store mode not supported in lowerLValue");
            }
        },
        .FieldAccess => |field| blk: {
            if (mode == .Load) {
                break :blk self.lowerFieldAccess(&field);
            } else {
                break :blk self.createErrorPlaceholder(field.span, "Field store should use storeLValue, not lowerLValue with Store mode");
            }
        },
        .Index => |index| blk: {
            if (mode == .Load) {
                break :blk self.lowerIndex(&index);
            } else {
                break :blk self.createErrorPlaceholder(index.span, "Index store should use storeLValue, not lowerLValue with Store mode");
            }
        },
        else => blk: {
            log.err("Invalid lvalue expression type\n", .{});
            break :blk self.createErrorPlaceholder(lib.ast.SourceSpan{ .line = 0, .column = 0, .length = 0, .byte_offset = 0 }, "Invalid lvalue");
        },
    };
}

/// Store value to lvalue target
pub fn storeLValue(
    self: *const ExpressionLowerer,
    lvalue: *const lib.ast.Expressions.ExprNode,
    value: c.MlirValue,
    span: lib.ast.SourceSpan,
) void {
    switch (lvalue.*) {
        .Identifier => |ident| {
            if (self.local_var_map) |lvm| {
                if (lvm.getLocalVar(ident.name)) |local_var_ref| {
                    const var_type = c.mlirValueGetType(local_var_ref);
                    var store_value = value;
                    if (c.mlirTypeIsAMemRef(var_type)) {
                        const element_type = c.mlirShapedTypeGetElementType(var_type);
                        const value_type = c.mlirValueGetType(value);
                        log.debug("[storeLValue] Variable: {s}, value_type != element_type: {}, value_is_ora: {}, element_is_ora: {}\n", .{ ident.name, !c.mlirTypeEqual(value_type, element_type), c.oraTypeIsIntegerType(value_type), c.oraTypeIsIntegerType(element_type) });
                        if (!c.mlirTypeEqual(value_type, element_type)) {
                            log.debug("[storeLValue] Converting (types not equal)\n", .{});
                            store_value = self.convertToType(value, element_type, span);
                        } else if (c.oraTypeIsIntegerType(value_type) and c.oraTypeIsIntegerType(element_type)) {
                            log.debug("[storeLValue] Converting (Ora types, explicit cast)\n", .{});
                            store_value = self.convertToType(value, element_type, span);
                        } else {
                            log.debug("[storeLValue] No conversion needed\n", .{});
                        }
                        const store_value_type = c.mlirValueGetType(store_value);
                        log.debug("[storeLValue] After conversion: types_equal: {}\n", .{c.mlirTypeEqual(store_value_type, element_type)});
                    }
                    var store_state = h.opState("memref.store", self.fileLoc(span));
                    c.mlirOperationStateAddOperands(&store_state, 2, @ptrCast(&[_]c.MlirValue{ store_value, local_var_ref }));
                    const store_op = c.mlirOperationCreate(&store_state);
                    h.appendOp(self.block, store_op);
                    return;
                }
            }

            if (self.storage_map) |sm| {
                if (sm.hasStorageVariable(ident.name)) {
                    const memory_manager = @import("../memory.zig").MemoryManager.init(self.ctx, self.ora_dialect);
                    const store_op = memory_manager.createStorageStore(value, ident.name, self.fileLoc(span));
                    h.appendOp(self.block, store_op);
                    return;
                }
            }

            log.err("Cannot store to undefined variable: {s}\n", .{ident.name});
        },
        .FieldAccess => |field| {
            const target_val = self.lowerExpression(field.target);
            const target_type = c.mlirValueGetType(target_val);

            if (field.target.* == .Identifier) {
                const ident = field.target.Identifier;
                if (ident.type_info.ora_type) |ora_type| {
                    if (ora_type == .struct_type) {
                        const expected_struct_type = self.type_mapper.toMlirType(ident.type_info);

                        if (c.oraTypeIsAddressType(target_type)) {
                            log.debug(
                                "ERROR [storeLValue]: Variable '{s}' is address type but should be struct type for field access\n",
                                .{ident.name},
                            );
                            log.debug(
                                "  This likely means a map load returned !ora.address instead of the struct type\n",
                                .{},
                            );
                            log.debug("  Expected struct type: {any}\n", .{expected_struct_type});
                            _ = self.reportLoweringError(
                                span,
                                "cannot update field on address type - map load returned wrong type",
                                "check map value type and struct layout for field access",
                            );
                            return;
                        }

                        if (!c.mlirTypeEqual(target_type, expected_struct_type)) {
                            log.debug(
                                "ERROR [storeLValue]: Variable '{s}' should be struct type but got: {any}\n",
                                .{ ident.name, target_type },
                            );
                            log.debug("  Expected struct type: {any}\n", .{expected_struct_type});
                            log.debug(
                                "  This likely means a map load returned wrong type instead of the struct type\n",
                                .{},
                            );
                            _ = self.reportLoweringError(
                                span,
                                "cannot update field on non-struct type - map load likely returned wrong type",
                                "check map value type and struct layout for field access",
                            );
                            return;
                        }
                    }
                }
            } else {
                if (c.oraTypeIsAddressType(target_type)) {
                    log.debug(
                        "ERROR [storeLValue]: Field access target is address type but should be struct type\n",
                        .{},
                    );
                    log.debug(
                        "  This likely means a map load returned !ora.address instead of the struct type\n",
                        .{},
                    );
                    _ = self.reportLoweringError(
                        span,
                        "cannot update field on address type - map load returned wrong type",
                        "check map value type and struct layout for field access",
                    );
                    return;
                }
            }

            const update_op = self.ora_dialect.createStructFieldUpdate(target_val, field.field, value, self.fileLoc(span));
            h.appendOp(self.block, update_op);
            _ = h.getResult(update_op, 0);
        },
        .Index => |index| {
            const target_val = self.lowerExpression(index.target);
            const index_val = self.lowerExpression(index.index);
            const target_type = c.mlirValueGetType(target_val);

            if (c.mlirTypeIsAMemRef(target_type) or c.mlirTypeIsAShaped(target_type)) {
                const index_index = expr_access.convertIndexToIndexType(self, index_val, index.span);
                var store_state = h.opState("memref.store", self.fileLoc(span));
                c.mlirOperationStateAddOperands(&store_state, 1, @ptrCast(&value));
                c.mlirOperationStateAddOperands(&store_state, 1, @ptrCast(&target_val));
                c.mlirOperationStateAddOperands(&store_state, 1, @ptrCast(&index_index));
                const store_op = c.mlirOperationCreate(&store_state);
                h.appendOp(self.block, store_op);
            } else {
                const op = self.ora_dialect.createMapStore(target_val, index_val, value, self.fileLoc(span));
                h.appendOp(self.block, op);
            }
        },
        else => {
            log.err("Invalid lvalue for assignment\n", .{});
        },
    }
}
