// ============================================================================
// Refinement Type Runtime Guards
// ============================================================================
//
// Generates runtime checks for refinement type constraints when static
// inference cannot prove safety.
//
// GUARD INSERTION POINTS:
//   • Function entry: Check refined parameters
//   • Variable initialization: Check initial value
//   • Assignment: Check value against target refinement
//   • Return: Check return value against function return type
//   • Division: Check exactness for Exact<T> dividends
//
// ============================================================================

const std = @import("std");
const c = @import("mlir/c_api.zig");
const h = @import("mlir/helpers.zig");
const lib = @import("../ast.zig");
const OraType = lib.Types.OraType;
const TypeInfo = lib.Types.TypeInfo;

/// Context for generating refinement guards
pub const GuardContext = struct {
    ctx: c.MlirContext,
    block: c.MlirBlock,
    expr_lowerer: *const anyopaque, // ExpressionLowerer - avoid circular dependency
    ora_dialect: *const anyopaque, // OraDialect - avoid circular dependency

    /// Create a file location from a span
    fn fileLoc(self: *const GuardContext, span: lib.SourceSpan) c.MlirLocation {
        _ = span; // TODO: Use span to create proper location
        // This should delegate to the actual location manager
        // For now, return a null location
        return c.mlirLocationUnknownGet(self.ctx);
    }
};

/// Generate runtime guard for a value against a refinement type
/// Returns the guarded value (same as input, but with checks inserted)
pub fn generateRefinementGuard(
    self: *const GuardContext,
    value: c.MlirValue,
    target_type: TypeInfo,
    span: lib.SourceSpan,
) !c.MlirValue {
    // If no refinement type, no guard needed
    const ora_type = target_type.ora_type orelse return value;

    return switch (ora_type) {
        .min_value => |mv| try self.generateMinValueGuard(value, mv.min, span),
        .max_value => |mv| try self.generateMaxValueGuard(value, mv.max, span),
        .in_range => |ir| try self.generateInRangeGuard(value, ir.min, ir.max, span),
        .exact => |_| value, // Exact guards are inserted at division operations
        .scaled => |_| value, // Scaled types don't need runtime guards (compile-time only)
        .non_zero_address => try self.generateNonZeroAddressGuard(value, span),
        else => value, // Not a refinement type
    };
}

/// Generate guard for MinValue<T, N>: require(value >= N)
fn generateMinValueGuard(
    self: *const GuardContext,
    value: c.MlirValue,
    min: u256,
    span: lib.SourceSpan,
) !c.MlirValue {
    const loc = self.fileLoc(span);

    // Create constant for minimum value
    const min_type = c.mlirValueGetType(value);
    const min_attr = c.mlirIntegerAttrGet(min_type, @intCast(min));
    const min_const = self.createConstant(min_attr, loc);

    // Create comparison: value >= min
    var cmp_state = h.opState("arith.cmpi", loc);
    const pred_id = h.identifier(self.ctx, "predicate");
    // uge = unsigned greater than or equal (predicate 5 for uge)
    const pred_attr = c.mlirIntegerAttrGet(c.mlirIntegerTypeGet(self.ctx, 64), 5);
    var attrs = [_]c.MlirNamedAttribute{c.mlirNamedAttributeGet(pred_id, pred_attr)};
    c.mlirOperationStateAddAttributes(&cmp_state, attrs.len, &attrs);
    c.mlirOperationStateAddOperands(&cmp_state, 2, @ptrCast(&[_]c.MlirValue{ value, min_const }));
    c.mlirOperationStateAddResults(&cmp_state, 1, @ptrCast(&h.boolType(self.ctx)));
    const cmp_op = c.mlirOperationCreate(&cmp_state);
    h.appendOp(self.block, cmp_op);
    const condition = h.getResult(cmp_op, 0);

    // Create assertion: cf.assert condition
    var assert_state = h.opState("cf.assert", loc);
    c.mlirOperationStateAddOperands(&assert_state, 1, @ptrCast(&condition));

    // Add error message attribute
    const msg = try std.fmt.allocPrint(std.heap.page_allocator, "Refinement violation: expected MinValue<u256, {d}>", .{min});
    defer std.heap.page_allocator.free(msg);
    const msg_attr = h.stringAttr(self.ctx, msg);
    const msg_id = h.identifier(self.ctx, "msg");
    var assert_attrs = [_]c.MlirNamedAttribute{c.mlirNamedAttributeGet(msg_id, msg_attr)};
    c.mlirOperationStateAddAttributes(&assert_state, assert_attrs.len, &assert_attrs);

    const assert_op = c.mlirOperationCreate(&assert_state);
    h.appendOp(self.block, assert_op);

    return value;
}

/// Generate guard for MaxValue<T, N>: require(value <= N)
fn generateMaxValueGuard(
    self: *const GuardContext,
    value: c.MlirValue,
    max: u256,
    span: lib.SourceSpan,
) !c.MlirValue {
    const loc = self.fileLoc(span);

    // Create constant for maximum value
    const max_type = c.mlirValueGetType(value);
    const max_attr = c.mlirIntegerAttrGet(max_type, @intCast(max));
    const max_const = self.createConstant(max_attr, loc);

    // Create comparison: value <= max
    var cmp_state = h.opState("arith.cmpi", loc);
    const pred_id = h.identifier(self.ctx, "predicate");
    // ule = unsigned less than or equal (predicate 3 for ule)
    const pred_attr = c.mlirIntegerAttrGet(c.mlirIntegerTypeGet(self.ctx, 64), 3);
    var attrs = [_]c.MlirNamedAttribute{c.mlirNamedAttributeGet(pred_id, pred_attr)};
    c.mlirOperationStateAddAttributes(&cmp_state, attrs.len, &attrs);
    c.mlirOperationStateAddOperands(&cmp_state, 2, @ptrCast(&[_]c.MlirValue{ value, max_const }));
    c.mlirOperationStateAddResults(&cmp_state, 1, @ptrCast(&h.boolType(self.ctx)));
    const cmp_op = c.mlirOperationCreate(&cmp_state);
    h.appendOp(self.block, cmp_op);
    const condition = h.getResult(cmp_op, 0);

    // Create assertion
    var assert_state = h.opState("cf.assert", loc);
    c.mlirOperationStateAddOperands(&assert_state, 1, @ptrCast(&condition));

    const msg = try std.fmt.allocPrint(std.heap.page_allocator, "Refinement violation: expected MaxValue<u256, {d}>", .{max});
    defer std.heap.page_allocator.free(msg);
    const msg_attr = h.stringAttr(self.ctx, msg);
    const msg_id = h.identifier(self.ctx, "msg");
    var assert_attrs = [_]c.MlirNamedAttribute{c.mlirNamedAttributeGet(msg_id, msg_attr)};
    c.mlirOperationStateAddAttributes(&assert_state, assert_attrs.len, &assert_attrs);

    const assert_op = c.mlirOperationCreate(&assert_state);
    h.appendOp(self.block, assert_op);

    return value;
}

/// Generate guard for InRange<T, MIN, MAX>: require(MIN <= value <= MAX)
fn generateInRangeGuard(
    self: *const GuardContext,
    value: c.MlirValue,
    min: u256,
    max: u256,
    span: lib.SourceSpan,
) !c.MlirValue {
    const loc = self.fileLoc(span);

    // Create constants
    const value_type = c.mlirValueGetType(value);
    const min_attr = c.mlirIntegerAttrGet(value_type, @intCast(min));
    const max_attr = c.mlirIntegerAttrGet(value_type, @intCast(max));
    const min_const = self.createConstant(min_attr, loc);
    const max_const = self.createConstant(max_attr, loc);

    // Check: value >= min
    var min_cmp_state = h.opState("arith.cmpi", loc);
    const pred_id = h.identifier(self.ctx, "predicate");
    const pred_uge_attr = c.mlirIntegerAttrGet(c.mlirIntegerTypeGet(self.ctx, 64), 5);
    var min_attrs = [_]c.MlirNamedAttribute{c.mlirNamedAttributeGet(pred_id, pred_uge_attr)};
    c.mlirOperationStateAddAttributes(&min_cmp_state, min_attrs.len, &min_attrs);
    c.mlirOperationStateAddOperands(&min_cmp_state, 2, @ptrCast(&[_]c.MlirValue{ value, min_const }));
    c.mlirOperationStateAddResults(&min_cmp_state, 1, @ptrCast(&h.boolType(self.ctx)));
    const min_cmp_op = c.mlirOperationCreate(&min_cmp_state);
    h.appendOp(self.block, min_cmp_op);
    const min_check = h.getResult(min_cmp_op, 0);

    // Check: value <= max
    var max_cmp_state = h.opState("arith.cmpi", loc);
    const pred_ule_attr = c.mlirIntegerAttrGet(c.mlirIntegerTypeGet(self.ctx, 64), 3);
    var max_attrs = [_]c.MlirNamedAttribute{c.mlirNamedAttributeGet(pred_id, pred_ule_attr)};
    c.mlirOperationStateAddAttributes(&max_cmp_state, max_attrs.len, &max_attrs);
    c.mlirOperationStateAddOperands(&max_cmp_state, 2, @ptrCast(&[_]c.MlirValue{ value, max_const }));
    c.mlirOperationStateAddResults(&max_cmp_state, 1, @ptrCast(&h.boolType(self.ctx)));
    const max_cmp_op = c.mlirOperationCreate(&max_cmp_state);
    h.appendOp(self.block, max_cmp_op);
    const max_check = h.getResult(max_cmp_op, 0);

    // Combine: min_check && max_check
    var and_state = h.opState("arith.andi", loc);
    c.mlirOperationStateAddOperands(&and_state, 2, @ptrCast(&[_]c.MlirValue{ min_check, max_check }));
    c.mlirOperationStateAddResults(&and_state, 1, @ptrCast(&h.boolType(self.ctx)));
    const and_op = c.mlirOperationCreate(&and_state);
    h.appendOp(self.block, and_op);
    const condition = h.getResult(and_op, 0);

    // Create assertion
    var assert_state = h.opState("cf.assert", loc);
    c.mlirOperationStateAddOperands(&assert_state, 1, @ptrCast(&condition));

    const msg = try std.fmt.allocPrint(std.heap.page_allocator, "Refinement violation: expected InRange<u256, {d}, {d}>", .{ min, max });
    defer std.heap.page_allocator.free(msg);
    const msg_attr = h.stringAttr(self.ctx, msg);
    const msg_id = h.identifier(self.ctx, "msg");
    var assert_attrs = [_]c.MlirNamedAttribute{c.mlirNamedAttributeGet(msg_id, msg_attr)};
    c.mlirOperationStateAddAttributes(&assert_state, assert_attrs.len, &assert_attrs);

    const assert_op = c.mlirOperationCreate(&assert_state);
    h.appendOp(self.block, assert_op);

    return value;
}

/// Generate exactness guard for division: require(dividend % divisor == 0)
/// This is called when dividing an Exact<T> value
pub fn generateExactDivisionGuard(
    self: *const GuardContext,
    dividend: c.MlirValue,
    divisor: c.MlirValue,
    span: lib.SourceSpan,
) !void {
    const loc = self.fileLoc(span);

    // Compute: dividend % divisor
    var mod_state = h.opState("arith.remui", loc);
    c.mlirOperationStateAddOperands(&mod_state, 2, @ptrCast(&[_]c.MlirValue{ dividend, divisor }));
    c.mlirOperationStateAddResults(&mod_state, 1, @ptrCast(&c.mlirValueGetType(dividend)));
    const mod_op = c.mlirOperationCreate(&mod_state);
    h.appendOp(self.block, mod_op);
    const remainder = h.getResult(mod_op, 0);

    // Create constant 0
    const zero_type = c.mlirValueGetType(dividend);
    const zero_attr = c.mlirIntegerAttrGet(zero_type, 0);
    const zero_const = self.createConstant(zero_attr, loc);

    // Check: remainder == 0
    var cmp_state = h.opState("arith.cmpi", loc);
    const pred_id = h.identifier(self.ctx, "predicate");
    const pred_eq_attr = c.mlirIntegerAttrGet(c.mlirIntegerTypeGet(self.ctx, 64), 0); // eq
    var attrs = [_]c.MlirNamedAttribute{c.mlirNamedAttributeGet(pred_id, pred_eq_attr)};
    c.mlirOperationStateAddAttributes(&cmp_state, attrs.len, &attrs);
    c.mlirOperationStateAddOperands(&cmp_state, 2, @ptrCast(&[_]c.MlirValue{ remainder, zero_const }));
    c.mlirOperationStateAddResults(&cmp_state, 1, @ptrCast(&h.boolType(self.ctx)));
    const cmp_op = c.mlirOperationCreate(&cmp_state);
    h.appendOp(self.block, cmp_op);
    const condition = h.getResult(cmp_op, 0);

    // Create assertion
    var assert_state = h.opState("cf.assert", loc);
    c.mlirOperationStateAddOperands(&assert_state, 1, @ptrCast(&condition));

    const msg = "Refinement violation: Exact<T> division must have no remainder";
    const msg_attr = h.stringAttr(self.ctx, msg);
    const msg_id = h.identifier(self.ctx, "msg");
    var assert_attrs = [_]c.MlirNamedAttribute{c.mlirNamedAttributeGet(msg_id, msg_attr)};
    c.mlirOperationStateAddAttributes(&assert_state, assert_attrs.len, &assert_attrs);

    const assert_op = c.mlirOperationCreate(&assert_state);
    h.appendOp(self.block, assert_op);
}

/// Generate guard for NonZeroAddress: require(address != 0)
fn generateNonZeroAddressGuard(
    self: *const GuardContext,
    value: c.MlirValue,
    span: lib.SourceSpan,
) !c.MlirValue {
    const loc = self.fileLoc(span);

    // Create constant 0
    const value_type = c.mlirValueGetType(value);
    const zero_attr = c.mlirIntegerAttrGet(value_type, 0);
    const zero_const = self.createConstant(zero_attr, loc);

    // Create comparison: value != 0
    var cmp_state = h.opState("arith.cmpi", loc);
    const pred_id = h.identifier(self.ctx, "predicate");
    // ne = not equal (predicate 1 for ne)
    const pred_attr = c.mlirIntegerAttrGet(c.mlirIntegerTypeGet(self.ctx, 64), 1);
    var attrs = [_]c.MlirNamedAttribute{c.mlirNamedAttributeGet(pred_id, pred_attr)};
    c.mlirOperationStateAddAttributes(&cmp_state, attrs.len, &attrs);
    c.mlirOperationStateAddOperands(&cmp_state, 2, @ptrCast(&[_]c.MlirValue{ value, zero_const }));
    c.mlirOperationStateAddResults(&cmp_state, 1, @ptrCast(&h.boolType(self.ctx)));
    const cmp_op = c.mlirOperationCreate(&cmp_state);
    h.appendOp(self.block, cmp_op);
    const condition = h.getResult(cmp_op, 0);

    // Create assertion
    var assert_state = h.opState("cf.assert", loc);
    c.mlirOperationStateAddOperands(&assert_state, 1, @ptrCast(&condition));

    const msg = "Refinement violation: expected NonZeroAddress (address cannot be zero)";
    const msg_attr = h.stringAttr(self.ctx, msg);
    const msg_id = h.identifier(self.ctx, "msg");
    var assert_attrs = [_]c.MlirNamedAttribute{c.mlirNamedAttributeGet(msg_id, msg_attr)};
    c.mlirOperationStateAddAttributes(&assert_state, assert_attrs.len, &assert_attrs);

    const assert_op = c.mlirOperationCreate(&assert_state);
    h.appendOp(self.block, assert_op);

    return value;
}

/// Helper to create a constant value from an attribute
fn createConstant(self: *const GuardContext, attr: c.MlirAttribute, loc: c.MlirLocation) c.MlirValue {
    const attr_type = c.mlirAttributeGetType(attr);
    var const_state = h.opState("arith.constant", loc);
    const value_id = h.identifier(self.ctx, "value");
    var attrs = [_]c.MlirNamedAttribute{c.mlirNamedAttributeGet(value_id, attr)};
    c.mlirOperationStateAddAttributes(&const_state, attrs.len, &attrs);
    c.mlirOperationStateAddResults(&const_state, 1, @ptrCast(&attr_type));
    const const_op = c.mlirOperationCreate(&const_state);
    h.appendOp(self.block, const_op);
    return h.getResult(const_op, 0);
}
