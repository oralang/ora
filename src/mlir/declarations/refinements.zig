// ============================================================================
// Declaration Lowering - Refinement Guards
// ============================================================================

const std = @import("std");
const c = @import("mlir_c_api").c;
const lib = @import("ora_lib");
const h = @import("../helpers.zig");
const StatementLowerer = @import("../statements.zig").StatementLowerer;
const LoweringError = StatementLowerer.LoweringError;
const DeclarationLowerer = @import("mod.zig").DeclarationLowerer;
const helpers = @import("helpers.zig");

fn reportRefinementGuardError(self: *const DeclarationLowerer, span: lib.ast.SourceSpan, message: []const u8, suggestion: ?[]const u8) void {
    if (self.error_handler) |handler| {
        handler.reportError(.MlirOperationFailed, span, message, suggestion) catch {};
    } else {
        std.log.warn("{s}", .{message});
    }
}

/// Insert refinement guard for function parameters and other declaration-level values
pub fn insertRefinementGuard(
    self: *const DeclarationLowerer,
    block: c.MlirBlock,
    value: c.MlirValue,
    ora_type: lib.ast.Types.OraType,
    span: lib.ast.SourceSpan,
) LoweringError!void {
    if (c.mlirValueIsNull(value)) {
        reportRefinementGuardError(self, span, "Refinement guard creation failed: value is null", "Ensure the value is produced before guard insertion.");
        return;
    }

    const loc = helpers.createFileLocation(self, span);

    switch (ora_type) {
        .min_value => |mv| {
            // generate: require(value >= min)
            const value_type = c.mlirValueGetType(value);
            // extract base type from refinement type for operations
            const base_type = c.oraRefinementTypeGetBaseType(value_type);
            const min_type = if (base_type.ptr != null) base_type else value_type;
            // convert value from refinement type to base type if needed
            const actual_value = if (base_type.ptr != null) blk: {
                const convert_op = c.oraRefinementToBaseOpCreate(self.ctx, loc, value, block);
                if (c.mlirOperationIsNull(convert_op)) {
                    reportRefinementGuardError(self, span, "Refinement guard creation failed: ora.refinement_to_base returned null", "Ensure the Ora dialect is registered before lowering.");
                    return;
                }
                // operation is already inserted by oraRefinementToBaseOpCreate, no need to append
                break :blk h.getResult(convert_op, 0);
            } else value;
            // for u256 values, create attribute from string to support full precision
            const min_attr = if (mv.min > std.math.maxInt(i64)) blk: {
                var min_buf: [100]u8 = undefined;
                const min_str = std.fmt.bufPrint(&min_buf, "{d}", .{mv.min}) catch {
                    reportRefinementGuardError(self, span, "Refinement guard creation failed: could not format min_value", "Reduce the refinement bound or report a compiler bug.");
                    return;
                };
                const min_str_ref = h.strRef(min_str);
                break :blk c.oraIntegerAttrGetFromString(min_type, min_str_ref);
            } else blk: {
                break :blk c.mlirIntegerAttrGet(min_type, @intCast(mv.min));
            };
            const min_const = helpers.createConstant(self, block, min_attr, min_type, loc);

            var cmp_state = h.opState("arith.cmpi", loc);
            const pred_id = h.identifier(self.ctx, "predicate");
            const pred_uge_attr = c.mlirIntegerAttrGet(c.mlirIntegerTypeGet(self.ctx, 64), 9); // uge
            var attrs = [_]c.MlirNamedAttribute{c.mlirNamedAttributeGet(pred_id, pred_uge_attr)};
            c.mlirOperationStateAddAttributes(&cmp_state, attrs.len, &attrs);
            c.mlirOperationStateAddOperands(&cmp_state, 2, @ptrCast(&[_]c.MlirValue{ actual_value, min_const }));
            c.mlirOperationStateAddResults(&cmp_state, 1, @ptrCast(&h.boolType(self.ctx)));
            const cmp_op = c.mlirOperationCreate(&cmp_state);
            h.appendOp(block, cmp_op);
            const condition = h.getResult(cmp_op, 0);

            var assert_state = h.opState("cf.assert", loc);
            c.mlirOperationStateAddOperands(&assert_state, 1, @ptrCast(&condition));
            const msg = try std.fmt.allocPrint(std.heap.page_allocator, "Refinement violation: expected MinValue<u256, {d}>", .{mv.min});
            defer std.heap.page_allocator.free(msg);
            const msg_attr = h.stringAttr(self.ctx, msg);
            const msg_id = h.identifier(self.ctx, "msg");
            var assert_attrs = [_]c.MlirNamedAttribute{c.mlirNamedAttributeGet(msg_id, msg_attr)};
            c.mlirOperationStateAddAttributes(&assert_state, assert_attrs.len, &assert_attrs);
            const assert_op = c.mlirOperationCreate(&assert_state);
            h.appendOp(block, assert_op);
        },
        .max_value => |mv| {
            // generate: require(value <= max)
            const value_type = c.mlirValueGetType(value);
            // extract base type from refinement type for operations
            const base_type = c.oraRefinementTypeGetBaseType(value_type);
            const max_type = if (base_type.ptr != null) base_type else value_type;
            // convert value from refinement type to base type if needed
            const actual_value = if (base_type.ptr != null) blk: {
                const convert_op = c.oraRefinementToBaseOpCreate(self.ctx, loc, value, block);
                if (c.mlirOperationIsNull(convert_op)) {
                    reportRefinementGuardError(self, span, "Refinement guard creation failed: ora.refinement_to_base returned null", "Ensure the Ora dialect is registered before lowering.");
                    return;
                }
                // operation is already inserted by oraRefinementToBaseOpCreate, no need to append
                break :blk h.getResult(convert_op, 0);
            } else value;
            // for u256 values, create attribute from string to support full precision
            const max_attr = if (mv.max > std.math.maxInt(i64)) blk: {
                var max_buf: [100]u8 = undefined;
                const max_str = std.fmt.bufPrint(&max_buf, "{d}", .{mv.max}) catch {
                    reportRefinementGuardError(self, span, "Refinement guard creation failed: could not format max_value", "Reduce the refinement bound or report a compiler bug.");
                    return;
                };
                const max_str_ref = h.strRef(max_str);
                break :blk c.oraIntegerAttrGetFromString(max_type, max_str_ref);
            } else blk: {
                break :blk c.mlirIntegerAttrGet(max_type, @intCast(mv.max));
            };
            const max_const = helpers.createConstant(self, block, max_attr, max_type, loc);

            var cmp_state = h.opState("arith.cmpi", loc);
            const pred_id = h.identifier(self.ctx, "predicate");
            const pred_ule_attr = c.mlirIntegerAttrGet(c.mlirIntegerTypeGet(self.ctx, 64), 7); // ule
            var attrs = [_]c.MlirNamedAttribute{c.mlirNamedAttributeGet(pred_id, pred_ule_attr)};
            c.mlirOperationStateAddAttributes(&cmp_state, attrs.len, &attrs);
            c.mlirOperationStateAddOperands(&cmp_state, 2, @ptrCast(&[_]c.MlirValue{ actual_value, max_const }));
            c.mlirOperationStateAddResults(&cmp_state, 1, @ptrCast(&h.boolType(self.ctx)));
            const cmp_op = c.mlirOperationCreate(&cmp_state);
            h.appendOp(block, cmp_op);
            const condition = h.getResult(cmp_op, 0);

            var assert_state = h.opState("cf.assert", loc);
            c.mlirOperationStateAddOperands(&assert_state, 1, @ptrCast(&condition));
            const msg = try std.fmt.allocPrint(std.heap.page_allocator, "Refinement violation: expected MaxValue<u256, {d}>", .{mv.max});
            defer std.heap.page_allocator.free(msg);
            const msg_attr = h.stringAttr(self.ctx, msg);
            const msg_id = h.identifier(self.ctx, "msg");
            var assert_attrs = [_]c.MlirNamedAttribute{c.mlirNamedAttributeGet(msg_id, msg_attr)};
            c.mlirOperationStateAddAttributes(&assert_state, assert_attrs.len, &assert_attrs);
            const assert_op = c.mlirOperationCreate(&assert_state);
            h.appendOp(block, assert_op);
        },
        .in_range => |ir| {
            // generate: require(min <= value <= max)
            const value_type = c.mlirValueGetType(value);
            // extract base type from refinement type for operations
            const base_type = c.oraRefinementTypeGetBaseType(value_type);
            const op_type = if (base_type.ptr != null) base_type else value_type;
            // convert value from refinement type to base type if needed
            const actual_value = if (base_type.ptr != null) blk: {
                const convert_op = c.oraRefinementToBaseOpCreate(self.ctx, loc, value, block);
                if (c.mlirOperationIsNull(convert_op)) {
                    reportRefinementGuardError(self, span, "Refinement guard creation failed: ora.refinement_to_base returned null", "Ensure the Ora dialect is registered before lowering.");
                    return;
                }
                // operation is already inserted by oraRefinementToBaseOpCreate, no need to append
                break :blk h.getResult(convert_op, 0);
            } else value;
            // for u256 values, create attributes from strings to support full precision
            const min_attr = if (ir.min > std.math.maxInt(i64)) blk: {
                var min_buf: [100]u8 = undefined;
                const min_str = std.fmt.bufPrint(&min_buf, "{d}", .{ir.min}) catch {
                    reportRefinementGuardError(self, span, "Refinement guard creation failed: could not format in_range min", "Reduce the refinement bound or report a compiler bug.");
                    return;
                };
                const min_str_ref = h.strRef(min_str);
                break :blk c.oraIntegerAttrGetFromString(op_type, min_str_ref);
            } else blk: {
                break :blk c.mlirIntegerAttrGet(op_type, @intCast(ir.min));
            };
            const max_attr = if (ir.max > std.math.maxInt(i64)) blk: {
                var max_buf: [100]u8 = undefined;
                const max_str = std.fmt.bufPrint(&max_buf, "{d}", .{ir.max}) catch {
                    reportRefinementGuardError(self, span, "Refinement guard creation failed: could not format in_range max", "Reduce the refinement bound or report a compiler bug.");
                    return;
                };
                const max_str_ref = h.strRef(max_str);
                break :blk c.oraIntegerAttrGetFromString(op_type, max_str_ref);
            } else blk: {
                break :blk c.mlirIntegerAttrGet(op_type, @intCast(ir.max));
            };
            const min_const = helpers.createConstant(self, block, min_attr, op_type, loc);
            const max_const = helpers.createConstant(self, block, max_attr, op_type, loc);

            // check: value >= min
            var min_cmp_state = h.opState("arith.cmpi", loc);
            const pred_id = h.identifier(self.ctx, "predicate");
            const pred_uge_attr = c.mlirIntegerAttrGet(c.mlirIntegerTypeGet(self.ctx, 64), 9); // uge
            var min_attrs = [_]c.MlirNamedAttribute{c.mlirNamedAttributeGet(pred_id, pred_uge_attr)};
            c.mlirOperationStateAddAttributes(&min_cmp_state, min_attrs.len, &min_attrs);
            c.mlirOperationStateAddOperands(&min_cmp_state, 2, @ptrCast(&[_]c.MlirValue{ actual_value, min_const }));
            c.mlirOperationStateAddResults(&min_cmp_state, 1, @ptrCast(&h.boolType(self.ctx)));
            const min_cmp_op = c.mlirOperationCreate(&min_cmp_state);
            h.appendOp(block, min_cmp_op);
            const min_check = h.getResult(min_cmp_op, 0);

            // check: value <= max
            var max_cmp_state = h.opState("arith.cmpi", loc);
            const pred_ule_attr = c.mlirIntegerAttrGet(c.mlirIntegerTypeGet(self.ctx, 64), 7); // ule
            var max_attrs = [_]c.MlirNamedAttribute{c.mlirNamedAttributeGet(pred_id, pred_ule_attr)};
            c.mlirOperationStateAddAttributes(&max_cmp_state, max_attrs.len, &max_attrs);
            c.mlirOperationStateAddOperands(&max_cmp_state, 2, @ptrCast(&[_]c.MlirValue{ actual_value, max_const }));
            c.mlirOperationStateAddResults(&max_cmp_state, 1, @ptrCast(&h.boolType(self.ctx)));
            const max_cmp_op = c.mlirOperationCreate(&max_cmp_state);
            h.appendOp(block, max_cmp_op);
            const max_check = h.getResult(max_cmp_op, 0);

            // combine: min_check && max_check
            var and_state = h.opState("arith.andi", loc);
            c.mlirOperationStateAddOperands(&and_state, 2, @ptrCast(&[_]c.MlirValue{ min_check, max_check }));
            c.mlirOperationStateAddResults(&and_state, 1, @ptrCast(&h.boolType(self.ctx)));
            const and_op = c.mlirOperationCreate(&and_state);
            h.appendOp(block, and_op);
            const condition = h.getResult(and_op, 0);

            var assert_state = h.opState("cf.assert", loc);
            c.mlirOperationStateAddOperands(&assert_state, 1, @ptrCast(&condition));
            const msg = try std.fmt.allocPrint(std.heap.page_allocator, "Refinement violation: expected InRange<u256, {d}, {d}>", .{ ir.min, ir.max });
            defer std.heap.page_allocator.free(msg);
            const msg_attr = h.stringAttr(self.ctx, msg);
            const msg_id = h.identifier(self.ctx, "msg");
            var assert_attrs = [_]c.MlirNamedAttribute{c.mlirNamedAttributeGet(msg_id, msg_attr)};
            c.mlirOperationStateAddAttributes(&assert_state, assert_attrs.len, &assert_attrs);
            const assert_op = c.mlirOperationCreate(&assert_state);
            h.appendOp(block, assert_op);
        },
        .exact, .scaled => {
            // exact and Scaled guards are inserted at specific operations (division, arithmetic)
            // no parameter guard needed
        },
        else => {
            // not a refinement type, no guard needed
        },
    }
}
