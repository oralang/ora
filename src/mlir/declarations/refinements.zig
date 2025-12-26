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
    if (c.oraValueIsNull(value)) {
        reportRefinementGuardError(self, span, "Refinement guard creation failed: value is null", "Ensure the value is produced before guard insertion.");
        return;
    }

    const loc = helpers.createFileLocation(self, span);

    switch (ora_type) {
        .min_value => |mv| {
            // generate: require(value >= min)
            const value_type = c.oraValueGetType(value);
            // extract base type from refinement type for operations
            const base_type = c.oraRefinementTypeGetBaseType(value_type);
            const min_type = if (base_type.ptr != null) base_type else value_type;
            // convert value from refinement type to base type if needed
            const actual_value = if (base_type.ptr != null) blk: {
                const convert_op = c.oraRefinementToBaseOpCreate(self.ctx, loc, value, block);
                if (c.oraOperationIsNull(convert_op)) {
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
                break :blk c.oraIntegerAttrCreateI64FromType(min_type, @intCast(mv.min));
            };
            const min_const = helpers.createConstant(self, block, min_attr, min_type, loc);

            const cmp_op = c.oraArithCmpIOpCreate(self.ctx, loc, 9, actual_value, min_const);
            h.appendOp(block, cmp_op);
            const condition = h.getResult(cmp_op, 0);

            const msg = try std.fmt.allocPrint(std.heap.page_allocator, "Refinement violation: expected MinValue<u256, {d}>", .{mv.min});
            defer std.heap.page_allocator.free(msg);
            const assert_op = self.ora_dialect.createCfAssert(condition, msg, loc);
            h.appendOp(block, assert_op);
        },
        .max_value => |mv| {
            // generate: require(value <= max)
            const value_type = c.oraValueGetType(value);
            // extract base type from refinement type for operations
            const base_type = c.oraRefinementTypeGetBaseType(value_type);
            const max_type = if (base_type.ptr != null) base_type else value_type;
            // convert value from refinement type to base type if needed
            const actual_value = if (base_type.ptr != null) blk: {
                const convert_op = c.oraRefinementToBaseOpCreate(self.ctx, loc, value, block);
                if (c.oraOperationIsNull(convert_op)) {
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
                break :blk c.oraIntegerAttrCreateI64FromType(max_type, @intCast(mv.max));
            };
            const max_const = helpers.createConstant(self, block, max_attr, max_type, loc);

            const cmp_op = c.oraArithCmpIOpCreate(self.ctx, loc, 7, actual_value, max_const);
            h.appendOp(block, cmp_op);
            const condition = h.getResult(cmp_op, 0);

            const msg = try std.fmt.allocPrint(std.heap.page_allocator, "Refinement violation: expected MaxValue<u256, {d}>", .{mv.max});
            defer std.heap.page_allocator.free(msg);
            const assert_op = self.ora_dialect.createCfAssert(condition, msg, loc);
            h.appendOp(block, assert_op);
        },
        .in_range => |ir| {
            // generate: require(min <= value <= max)
            const value_type = c.oraValueGetType(value);
            // extract base type from refinement type for operations
            const base_type = c.oraRefinementTypeGetBaseType(value_type);
            const op_type = if (base_type.ptr != null) base_type else value_type;
            // convert value from refinement type to base type if needed
            const actual_value = if (base_type.ptr != null) blk: {
                const convert_op = c.oraRefinementToBaseOpCreate(self.ctx, loc, value, block);
                if (c.oraOperationIsNull(convert_op)) {
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
                break :blk c.oraIntegerAttrCreateI64FromType(op_type, @intCast(ir.min));
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
                break :blk c.oraIntegerAttrCreateI64FromType(op_type, @intCast(ir.max));
            };
            const min_const = helpers.createConstant(self, block, min_attr, op_type, loc);
            const max_const = helpers.createConstant(self, block, max_attr, op_type, loc);

            // check: value >= min
            const min_cmp_op = c.oraArithCmpIOpCreate(self.ctx, loc, 9, actual_value, min_const);
            h.appendOp(block, min_cmp_op);
            const min_check = h.getResult(min_cmp_op, 0);

            // check: value <= max
            const max_cmp_op = c.oraArithCmpIOpCreate(self.ctx, loc, 7, actual_value, max_const);
            h.appendOp(block, max_cmp_op);
            const max_check = h.getResult(max_cmp_op, 0);

            // combine: min_check && max_check
            const and_op = c.oraArithAndIOpCreate(self.ctx, loc, min_check, max_check);
            h.appendOp(block, and_op);
            const condition = h.getResult(and_op, 0);

            const msg = try std.fmt.allocPrint(std.heap.page_allocator, "Refinement violation: expected InRange<u256, {d}, {d}>", .{ ir.min, ir.max });
            defer std.heap.page_allocator.free(msg);
            const assert_op = self.ora_dialect.createCfAssert(condition, msg, loc);
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
