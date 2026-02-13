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

fn createBitfieldFieldUpdate(
    self: *const ExpressionLowerer,
    word: c.MlirValue,
    field_name: []const u8,
    new_val: c.MlirValue,
    type_sym: *const constants.TypeSymbol,
    span: lib.ast.SourceSpan,
) c.MlirValue {
    return createBitfieldFieldUpdateImpl(self, word, field_name, new_val, type_sym, span);
}

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
                    const var_type = c.oraValueGetType(local_var_ref);
                    var store_value = value;
                    if (c.oraTypeIsAMemRef(var_type)) {
                        const element_type = c.oraShapedTypeGetElementType(var_type);
                        const value_type = c.oraValueGetType(value);
                        log.debug("[ASSIGN Expression] Variable: {s}, value_type != element_type: {}, value_is_ora: {}, element_is_ora: {}\n", .{ ident.name, !c.oraTypeEqual(value_type, element_type), c.oraTypeIsIntegerType(value_type), c.oraTypeIsIntegerType(element_type) });
                        if (!c.oraTypeEqual(value_type, element_type)) {
                            log.debug("[ASSIGN Expression] Converting (types not equal)\n", .{});
                            store_value = self.convertToType(value, element_type, assign.span);
                        } else if (c.oraTypeIsIntegerType(value_type) and c.oraTypeIsIntegerType(element_type)) {
                            log.debug("[ASSIGN Expression] Converting (Ora types, explicit cast)\n", .{});
                            store_value = self.convertToType(value, element_type, assign.span);
                        } else {
                            log.debug("[ASSIGN Expression] No conversion needed\n", .{});
                        }
                        const store_value_type = c.oraValueGetType(store_value);
                        log.debug("[ASSIGN Expression] After conversion: types_equal: {}\n", .{c.oraTypeEqual(store_value_type, element_type)});
                    }
                    const store_op = c.oraMemrefStoreOpCreate(
                        self.ctx,
                        self.fileLoc(assign.span),
                        store_value,
                        local_var_ref,
                        null,
                        0,
                    );
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

            const var_type = c.oraValueGetType(value);
            const memref_type = h.memRefType(self.ctx, var_type, 0, null, h.nullAttr(), h.nullAttr());
            const alloca_op = self.ora_dialect.createMemrefAlloca(memref_type, self.fileLoc(assign.span));
            h.appendOp(self.block, alloca_op);
            const alloca_result = h.getResult(alloca_op, 0);

            const store_op = c.oraMemrefStoreOpCreate(
                self.ctx,
                self.fileLoc(assign.span),
                value,
                alloca_result,
                null,
                0,
            );
            h.appendOp(self.block, store_op);

            return value;
        },
        .FieldAccess => |field_access| {
            const target_value = self.lowerExpression(field_access.target);
            const target_type = c.oraValueGetType(target_value);
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

                        if (!c.oraTypeEqual(target_type, expected_struct_type)) {
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

            // Check if target is a bitfield → emit clear+set bit manipulation
            if (self.symbol_table) |st| {
                const target_type_info = expr_operators.extractTypeInfo(field_access.target);
                if (target_type_info.ora_type) |ora_type| {
                    if (ora_type == .bitfield_type) {
                        if (st.lookupType(ora_type.bitfield_type)) |type_sym| {
                            if (type_sym.type_kind == .Bitfield) {
                                const updated = createBitfieldFieldUpdate(self, target_value, field_name, value, type_sym, assign.span);
                                // Store back to storage if target is a storage variable
                                storeBitfieldBackToStorage(self, field_access.target, updated, assign.span);
                                return updated;
                            }
                        }
                    }
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
            const target_type = c.oraValueGetType(target_value);

            if (c.oraTypeIsAMemRef(target_type) or c.oraTypeIsAShaped(target_type)) {
                const index_index = expr_access.convertIndexToIndexType(self, index_value, index_expr.span);
                const store_op = c.oraMemrefStoreOpCreate(
                    self.ctx,
                    self.fileLoc(assign.span),
                    value,
                    target_value,
                    &[_]c.MlirValue{index_index},
                    1,
                );
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

    const current_ty = c.oraValueGetType(current_value);
    const rhs_ty = c.oraValueGetType(rhs_value);
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

    const lhs_type_info = expr_operators.extractTypeInfo(comp_assign.target);
    const rhs_type_info = expr_operators.extractTypeInfo(comp_assign.value);
    const uses_signed_integer_semantics = expr_operators.usesSignedIntegerSemantics(lhs_type_info, rhs_type_info);
    const div_op_name = if (uses_signed_integer_semantics) "arith.divsi" else "arith.divui";
    const rem_op_name = if (uses_signed_integer_semantics) "arith.remsi" else "arith.remui";

    const result_value = switch (comp_assign.operator) {
        .PlusEqual => self.createArithmeticOp("arith.addi", current_converted, rhs_converted, common_ty, comp_assign.span),
        .MinusEqual => self.createArithmeticOp("arith.subi", current_converted, rhs_converted, common_ty, comp_assign.span),
        .StarEqual => self.createArithmeticOp("arith.muli", current_converted, rhs_converted, common_ty, comp_assign.span),
        .SlashEqual => self.createArithmeticOp(div_op_name, current_converted, rhs_converted, common_ty, comp_assign.span),
        .PercentEqual => self.createArithmeticOp(rem_op_name, current_converted, rhs_converted, common_ty, comp_assign.span),
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
                    const var_type = c.oraValueGetType(local_var_ref);
                    var store_value = value;
                    if (c.oraTypeIsAMemRef(var_type)) {
                        const element_type = c.oraShapedTypeGetElementType(var_type);
                        const value_type = c.oraValueGetType(value);
                        log.debug("[storeLValue] Variable: {s}, value_type != element_type: {}, value_is_ora: {}, element_is_ora: {}\n", .{ ident.name, !c.oraTypeEqual(value_type, element_type), c.oraTypeIsIntegerType(value_type), c.oraTypeIsIntegerType(element_type) });
                        if (!c.oraTypeEqual(value_type, element_type)) {
                            log.debug("[storeLValue] Converting (types not equal)\n", .{});
                            store_value = self.convertToType(value, element_type, span);
                        } else if (c.oraTypeIsIntegerType(value_type) and c.oraTypeIsIntegerType(element_type)) {
                            log.debug("[storeLValue] Converting (Ora types, explicit cast)\n", .{});
                            store_value = self.convertToType(value, element_type, span);
                        } else {
                            log.debug("[storeLValue] No conversion needed\n", .{});
                        }
                        const store_value_type = c.oraValueGetType(store_value);
                        log.debug("[storeLValue] After conversion: types_equal: {}\n", .{c.oraTypeEqual(store_value_type, element_type)});
                    }
                    const store_op = c.oraMemrefStoreOpCreate(
                        self.ctx,
                        self.fileLoc(span),
                        store_value,
                        local_var_ref,
                        null,
                        0,
                    );
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

            if (self.symbol_table) |st| {
                if (st.lookupSymbol(ident.name)) |symbol| {
                    if (std.mem.eql(u8, symbol.region, "tstore")) {
                        const memory_manager = @import("../memory.zig").MemoryManager.init(self.ctx, self.ora_dialect);
                        const store_op = memory_manager.createTStoreStore(value, ident.name, self.fileLoc(span));
                        h.appendOp(self.block, store_op);
                        return;
                    }
                }
            }

            log.err("Cannot store to undefined variable: {s}\n", .{ident.name});
        },
        .FieldAccess => |field| {
            const target_val = self.lowerExpression(field.target);
            const target_type = c.oraValueGetType(target_val);

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

                        if (!c.oraTypeEqual(target_type, expected_struct_type)) {
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

            // Check if target is a bitfield → emit clear+set bit manipulation
            if (self.symbol_table) |st| {
                const target_type_info = expr_operators.extractTypeInfo(field.target);
                if (target_type_info.ora_type) |ora_type| {
                    if (ora_type == .bitfield_type) {
                        if (st.lookupType(ora_type.bitfield_type)) |type_sym| {
                            if (type_sym.type_kind == .Bitfield) {
                                const updated = createBitfieldFieldUpdate(self, target_val, field.field, value, type_sym, span);
                                storeBitfieldBackToStorage(self, field.target, updated, span);
                                return;
                            }
                        }
                    }
                }
            }

            const update_op = self.ora_dialect.createStructFieldUpdate(target_val, field.field, value, self.fileLoc(span));
            h.appendOp(self.block, update_op);
            _ = h.getResult(update_op, 0);
        },
        .Index => |index| {
            const target_val = self.lowerExpression(index.target);
            const index_val = self.lowerExpression(index.index);
            const target_type = c.oraValueGetType(target_val);

            if (c.oraTypeIsAMemRef(target_type) or c.oraTypeIsAShaped(target_type)) {
                const index_index = expr_access.convertIndexToIndexType(self, index_val, index.span);
                const store_op = c.oraMemrefStoreOpCreate(
                    self.ctx,
                    self.fileLoc(span),
                    value,
                    target_val,
                    &[_]c.MlirValue{index_index},
                    1,
                );
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

/// Bitfield field write: cleared = word & ~(mask << offset), updated = cleared | ((val & mask) << offset)
pub fn createBitfieldFieldUpdateImpl(
    self: *const ExpressionLowerer,
    word: c.MlirValue,
    field_name: []const u8,
    new_val: c.MlirValue,
    type_sym: *const constants.TypeSymbol,
    span: lib.ast.SourceSpan,
) c.MlirValue {
    const loc = self.fileLoc(span);
    const int_ty = c.oraIntegerTypeCreate(self.ctx, constants.DEFAULT_INTEGER_BITS);

    if (type_sym.fields) |fields| {
        for (fields) |field| {
            if (!std.mem.eql(u8, field.name, field_name)) continue;

            const offset: i64 = if (field.offset) |o| @intCast(o) else 0;
            const width: u32 = field.bit_width orelse 256;
            const mask: i64 = if (width >= 64) -1 else (@as(i64, 1) << @intCast(width)) - 1;

            // mask_val = (1 << width) - 1
            const mask_op = self.ora_dialect.createArithConstant(mask, int_ty, loc);
            h.appendOp(self.block, mask_op);
            const mask_val = h.getResult(mask_op, 0);

            // offset_val = offset
            const offset_op = self.ora_dialect.createArithConstant(offset, int_ty, loc);
            h.appendOp(self.block, offset_op);
            const offset_val = h.getResult(offset_op, 0);

            // shifted_mask = mask << offset
            const shl_mask_op = c.oraArithShlIOpCreate(self.ctx, loc, mask_val, offset_val);
            h.appendOp(self.block, shl_mask_op);
            const shifted_mask = h.getResult(shl_mask_op, 0);

            // inv_mask = ~(mask << offset) = shifted_mask XOR -1
            const all_ones_op = self.ora_dialect.createArithConstant(-1, int_ty, loc);
            h.appendOp(self.block, all_ones_op);
            const all_ones = h.getResult(all_ones_op, 0);

            const inv_mask_op = c.oraArithXorIOpCreate(self.ctx, loc, shifted_mask, all_ones);
            h.appendOp(self.block, inv_mask_op);
            const inv_mask = h.getResult(inv_mask_op, 0);

            // cleared = word & inv_mask
            const cleared_op = c.oraArithAndIOpCreate(self.ctx, loc, word, inv_mask);
            h.appendOp(self.block, cleared_op);
            const cleared = h.getResult(cleared_op, 0);

            // Widen new_val to i256 if it's a narrower integer (e.g. u8 field value)
            var widened_val = new_val;
            const val_type = c.oraValueGetType(new_val);
            if (!c.oraTypeEqual(val_type, int_ty)) {
                const ext_op = c.oraArithExtUIOpCreate(self.ctx, loc, new_val, int_ty);
                h.appendOp(self.block, ext_op);
                widened_val = h.getResult(ext_op, 0);
            }

            // masked_val = new_val & mask
            const masked_val_op = c.oraArithAndIOpCreate(self.ctx, loc, widened_val, mask_val);
            h.appendOp(self.block, masked_val_op);
            const masked_val = h.getResult(masked_val_op, 0);

            // shifted_val = masked_val << offset
            const shl_val_op = c.oraArithShlIOpCreate(self.ctx, loc, masked_val, offset_val);
            h.appendOp(self.block, shl_val_op);
            const shifted_val = h.getResult(shl_val_op, 0);

            // updated = cleared | shifted_val
            const or_op = c.oraArithOrIOpCreate(self.ctx, loc, cleared, shifted_val);
            h.appendOp(self.block, or_op);
            return h.getResult(or_op, 0);
        }
    }

    log.debug("ERROR: Bitfield field '{s}' not found in type '{s}'\n", .{ field_name, type_sym.name });
    return self.createErrorPlaceholder(span, "bitfield field not found");
}

/// If the field access target is a storage-backed variable, emit SSTORE with the updated value.
fn storeBitfieldBackToStorage(
    self: *const ExpressionLowerer,
    target_expr: *const lib.ast.Expressions.ExprNode,
    updated_value: c.MlirValue,
    span: lib.ast.SourceSpan,
) void {
    if (target_expr.* != .Identifier) return;
    const ident = target_expr.Identifier;
    if (self.storage_map) |sm| {
        if (sm.hasStorageVariable(ident.name)) {
            const memory_manager = @import("../memory.zig").MemoryManager.init(self.ctx, self.ora_dialect);
            const store_op = memory_manager.createStorageStore(updated_value, ident.name, self.fileLoc(span));
            h.appendOp(self.block, store_op);
        }
    }
}
