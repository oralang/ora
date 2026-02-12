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
const c = @import("mlir_c_api").c;
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
    filename: ?[]const u8 = null,

    /// Create a file location from a span
    fn fileLoc(self: *const GuardContext, span: lib.SourceSpan) c.MlirLocation {
        const fname = if (self.filename) |name| name else "input.ora";
        const fname_ref = c.mlirStringRefCreate(fname.ptr, fname.len);
        return c.mlirLocationFileLineColGet(self.ctx, fname_ref, span.line, span.column);
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
    // if no refinement type, no guard needed
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

    // create constant for minimum value
    const min_type = c.mlirValueGetType(value);
    const min_attr = c.mlirIntegerAttrGet(min_type, @intCast(min));
    const min_const = self.createConstant(min_attr, loc);

    // create comparison: value >= min
    const pred_id = h.identifier(self.ctx, "predicate");
    // uge = unsigned greater than or equal (predicate 5 for uge)
    const pred_attr = c.mlirIntegerAttrGet(c.mlirIntegerTypeGet(self.ctx, 64), 5);
    var attrs = [_]c.MlirNamedAttribute{c.mlirNamedAttributeGet(pred_id, pred_attr)};
    const cmp_op = h.createOp(
        self.ctx,
        loc,
        "arith.cmpi",
        &[_]c.MlirValue{ value, min_const },
        &[_]c.MlirType{h.boolType(self.ctx)},
        &attrs,
        0,
        false,
    );
    h.appendOp(self.block, cmp_op);
    const condition = h.getResult(cmp_op, 0);

    // create assertion: cf.assert condition
    // add error message attribute
    const msg = try std.fmt.allocPrint(std.heap.page_allocator, "Refinement violation: expected MinValue<u256, {d}>", .{min});
    defer std.heap.page_allocator.free(msg);
    const msg_attr = h.stringAttr(self.ctx, msg);
    const msg_id = h.identifier(self.ctx, "msg");
    var assert_attrs = [_]c.MlirNamedAttribute{c.mlirNamedAttributeGet(msg_id, msg_attr)};
    const assert_op = h.createOp(
        self.ctx,
        loc,
        "cf.assert",
        &[_]c.MlirValue{condition},
        &[_]c.MlirType{},
        &assert_attrs,
        0,
        false,
    );
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

    // create constant for maximum value
    const max_type = c.mlirValueGetType(value);
    const max_attr = c.mlirIntegerAttrGet(max_type, @intCast(max));
    const max_const = self.createConstant(max_attr, loc);

    // create comparison: value <= max
    const pred_id = h.identifier(self.ctx, "predicate");
    // ule = unsigned less than or equal (predicate 3 for ule)
    const pred_attr = c.mlirIntegerAttrGet(c.mlirIntegerTypeGet(self.ctx, 64), 3);
    var attrs = [_]c.MlirNamedAttribute{c.mlirNamedAttributeGet(pred_id, pred_attr)};
    const cmp_op = h.createOp(
        self.ctx,
        loc,
        "arith.cmpi",
        &[_]c.MlirValue{ value, max_const },
        &[_]c.MlirType{h.boolType(self.ctx)},
        &attrs,
        0,
        false,
    );
    h.appendOp(self.block, cmp_op);
    const condition = h.getResult(cmp_op, 0);

    // create assertion
    const msg = try std.fmt.allocPrint(std.heap.page_allocator, "Refinement violation: expected MaxValue<u256, {d}>", .{max});
    defer std.heap.page_allocator.free(msg);
    const msg_attr = h.stringAttr(self.ctx, msg);
    const msg_id = h.identifier(self.ctx, "msg");
    var assert_attrs = [_]c.MlirNamedAttribute{c.mlirNamedAttributeGet(msg_id, msg_attr)};
    const assert_op = h.createOp(
        self.ctx,
        loc,
        "cf.assert",
        &[_]c.MlirValue{condition},
        &[_]c.MlirType{},
        &assert_attrs,
        0,
        false,
    );
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

    // create constants
    const value_type = c.mlirValueGetType(value);
    const min_attr = c.mlirIntegerAttrGet(value_type, @intCast(min));
    const max_attr = c.mlirIntegerAttrGet(value_type, @intCast(max));
    const min_const = self.createConstant(min_attr, loc);
    const max_const = self.createConstant(max_attr, loc);

    // check: value >= min
    const pred_id = h.identifier(self.ctx, "predicate");
    const pred_uge_attr = c.mlirIntegerAttrGet(c.mlirIntegerTypeGet(self.ctx, 64), 5);
    var min_attrs = [_]c.MlirNamedAttribute{c.mlirNamedAttributeGet(pred_id, pred_uge_attr)};
    const min_cmp_op = h.createOp(
        self.ctx,
        loc,
        "arith.cmpi",
        &[_]c.MlirValue{ value, min_const },
        &[_]c.MlirType{h.boolType(self.ctx)},
        &min_attrs,
        0,
        false,
    );
    h.appendOp(self.block, min_cmp_op);
    const min_check = h.getResult(min_cmp_op, 0);

    // check: value <= max
    const pred_ule_attr = c.mlirIntegerAttrGet(c.mlirIntegerTypeGet(self.ctx, 64), 3);
    var max_attrs = [_]c.MlirNamedAttribute{c.mlirNamedAttributeGet(pred_id, pred_ule_attr)};
    const max_cmp_op = h.createOp(
        self.ctx,
        loc,
        "arith.cmpi",
        &[_]c.MlirValue{ value, max_const },
        &[_]c.MlirType{h.boolType(self.ctx)},
        &max_attrs,
        0,
        false,
    );
    h.appendOp(self.block, max_cmp_op);
    const max_check = h.getResult(max_cmp_op, 0);

    // combine: min_check && max_check
    const and_op = h.createOp(
        self.ctx,
        loc,
        "arith.andi",
        &[_]c.MlirValue{ min_check, max_check },
        &[_]c.MlirType{h.boolType(self.ctx)},
        &[_]c.MlirNamedAttribute{},
        0,
        false,
    );
    h.appendOp(self.block, and_op);
    const condition = h.getResult(and_op, 0);

    // create assertion
    const msg = try std.fmt.allocPrint(std.heap.page_allocator, "Refinement violation: expected InRange<u256, {d}, {d}>", .{ min, max });
    defer std.heap.page_allocator.free(msg);
    const msg_attr = h.stringAttr(self.ctx, msg);
    const msg_id = h.identifier(self.ctx, "msg");
    var assert_attrs = [_]c.MlirNamedAttribute{c.mlirNamedAttributeGet(msg_id, msg_attr)};
    const assert_op = h.createOp(
        self.ctx,
        loc,
        "cf.assert",
        &[_]c.MlirValue{condition},
        &[_]c.MlirType{},
        &assert_attrs,
        0,
        false,
    );
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
    const divisor_type = c.mlirValueGetType(divisor);
    const zero_divisor_attr = c.mlirIntegerAttrGet(divisor_type, 0);
    const zero_divisor = self.createConstant(zero_divisor_attr, loc);

    // check: divisor != 0
    const ne_pred_id = h.identifier(self.ctx, "predicate");
    const ne_pred_attr = c.mlirIntegerAttrGet(c.mlirIntegerTypeGet(self.ctx, 64), 1); // ne
    var ne_attrs = [_]c.MlirNamedAttribute{c.mlirNamedAttributeGet(ne_pred_id, ne_pred_attr)};
    const non_zero_cmp = h.createOp(
        self.ctx,
        loc,
        "arith.cmpi",
        &[_]c.MlirValue{ divisor, zero_divisor },
        &[_]c.MlirType{h.boolType(self.ctx)},
        &ne_attrs,
        0,
        false,
    );
    h.appendOp(self.block, non_zero_cmp);
    const non_zero = h.getResult(non_zero_cmp, 0);

    // compute: dividend % divisor
    const mod_op = h.createOp(
        self.ctx,
        loc,
        "arith.remui",
        &[_]c.MlirValue{ dividend, divisor },
        &[_]c.MlirType{c.mlirValueGetType(dividend)},
        &[_]c.MlirNamedAttribute{},
        0,
        false,
    );
    h.appendOp(self.block, mod_op);
    const remainder = h.getResult(mod_op, 0);

    // create constant 0
    const zero_type = c.mlirValueGetType(dividend);
    const zero_attr = c.mlirIntegerAttrGet(zero_type, 0);
    const zero_const = self.createConstant(zero_attr, loc);

    // check: remainder == 0
    const pred_id = h.identifier(self.ctx, "predicate");
    const pred_eq_attr = c.mlirIntegerAttrGet(c.mlirIntegerTypeGet(self.ctx, 64), 0); // eq
    var attrs = [_]c.MlirNamedAttribute{c.mlirNamedAttributeGet(pred_id, pred_eq_attr)};
    const cmp_op = h.createOp(
        self.ctx,
        loc,
        "arith.cmpi",
        &[_]c.MlirValue{ remainder, zero_const },
        &[_]c.MlirType{h.boolType(self.ctx)},
        &attrs,
        0,
        false,
    );
    h.appendOp(self.block, cmp_op);
    const exact = h.getResult(cmp_op, 0);

    const and_op = h.createOp(
        self.ctx,
        loc,
        "arith.andi",
        &[_]c.MlirValue{ non_zero, exact },
        &[_]c.MlirType{h.boolType(self.ctx)},
        &[_]c.MlirNamedAttribute{},
        0,
        false,
    );
    h.appendOp(self.block, and_op);
    const condition = h.getResult(and_op, 0);

    // create assertion
    const msg = "Refinement violation: Exact<T> division must have no remainder";
    const msg_attr = h.stringAttr(self.ctx, msg);
    const msg_id = h.identifier(self.ctx, "msg");
    var assert_attrs = [_]c.MlirNamedAttribute{c.mlirNamedAttributeGet(msg_id, msg_attr)};
    const assert_op = h.createOp(
        self.ctx,
        loc,
        "cf.assert",
        &[_]c.MlirValue{condition},
        &[_]c.MlirType{},
        &assert_attrs,
        0,
        false,
    );
    h.appendOp(self.block, assert_op);
}

/// Generate guard for NonZeroAddress: require(address != 0)
fn generateNonZeroAddressGuard(
    self: *const GuardContext,
    value: c.MlirValue,
    span: lib.SourceSpan,
) !c.MlirValue {
    const loc = self.fileLoc(span);

    // create constant 0
    const value_type = c.mlirValueGetType(value);
    const zero_attr = c.mlirIntegerAttrGet(value_type, 0);
    const zero_const = self.createConstant(zero_attr, loc);

    // create comparison: value != 0
    const pred_id = h.identifier(self.ctx, "predicate");
    // ne = not equal (predicate 1 for ne)
    const pred_attr = c.mlirIntegerAttrGet(c.mlirIntegerTypeGet(self.ctx, 64), 1);
    var attrs = [_]c.MlirNamedAttribute{c.mlirNamedAttributeGet(pred_id, pred_attr)};
    const cmp_op = h.createOp(
        self.ctx,
        loc,
        "arith.cmpi",
        &[_]c.MlirValue{ value, zero_const },
        &[_]c.MlirType{h.boolType(self.ctx)},
        &attrs,
        0,
        false,
    );
    h.appendOp(self.block, cmp_op);
    const condition = h.getResult(cmp_op, 0);

    // create assertion
    const msg = "Refinement violation: expected NonZeroAddress (address cannot be zero)";
    const msg_attr = h.stringAttr(self.ctx, msg);
    const msg_id = h.identifier(self.ctx, "msg");
    var assert_attrs = [_]c.MlirNamedAttribute{c.mlirNamedAttributeGet(msg_id, msg_attr)};
    const assert_op = h.createOp(
        self.ctx,
        loc,
        "cf.assert",
        &[_]c.MlirValue{condition},
        &[_]c.MlirType{},
        &assert_attrs,
        0,
        false,
    );
    h.appendOp(self.block, assert_op);

    return value;
}

/// Helper to create a constant value from an attribute
fn createConstant(self: *const GuardContext, attr: c.MlirAttribute, loc: c.MlirLocation) c.MlirValue {
    const attr_type = c.mlirAttributeGetType(attr);
    const const_op = c.oraArithConstantOpCreate(self.ctx, loc, attr_type, attr);
    if (const_op.ptr == null) {
        @panic("Failed to create arith.constant operation");
    }
    h.appendOp(self.block, const_op);
    return h.getResult(const_op, 0);
}
