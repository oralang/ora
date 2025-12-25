// ============================================================================
// Expression Lowering Helpers
// ============================================================================
// Standalone helper functions for expression lowering operations
// These are called by ExpressionLowerer methods to avoid circular dependencies

const std = @import("std");
const c = @import("mlir_c_api").c;
const lib = @import("ora_lib");
const constants = @import("../lower.zig");
const h = @import("../helpers.zig");
const TypeMapper = @import("../types.zig").TypeMapper;
const LocationTracker = @import("../locations.zig").LocationTracker;
const OraDialect = @import("../dialect.zig").OraDialect;

/// Create a constant value
pub fn createConstant(ctx: c.MlirContext, block: c.MlirBlock, ora_dialect: *OraDialect, locations: LocationTracker, value: i64, span: lib.ast.SourceSpan) c.MlirValue {
    const ty = c.mlirIntegerTypeGet(ctx, constants.DEFAULT_INTEGER_BITS);
    const loc = locations.createLocation(span);
    const op = ora_dialect.createArithConstant(value, ty, loc);
    h.appendOp(block, op);
    return h.getResult(op, 0);
}

/// Create an error placeholder value
pub fn createErrorPlaceholder(ctx: c.MlirContext, block: c.MlirBlock, locations: LocationTracker, span: lib.ast.SourceSpan, error_msg: []const u8) c.MlirValue {
    const ty = c.mlirIntegerTypeGet(ctx, constants.DEFAULT_INTEGER_BITS);
    const loc = locations.createLocation(span);
    var state = h.opState("arith.constant", loc);
    c.mlirOperationStateAddResults(&state, 1, @ptrCast(&ty));

    const attr = c.mlirIntegerAttrGet(ty, 0);
    const value_id = h.identifier(ctx, "value");
    const error_id = h.identifier(ctx, "ora.error_placeholder");
    const error_attr = h.stringAttr(ctx, error_msg);

    var attrs = [_]c.MlirNamedAttribute{
        c.mlirNamedAttributeGet(value_id, attr),
        c.mlirNamedAttributeGet(error_id, error_attr),
    };
    c.mlirOperationStateAddAttributes(&state, attrs.len, &attrs);
    const op = c.mlirOperationCreate(&state);
    h.appendOp(block, op);
    return h.getResult(op, 0);
}

/// Create arithmetic operation
pub fn createArithmeticOp(_: c.MlirContext, block: c.MlirBlock, type_mapper: *const TypeMapper, _: *OraDialect, locations: LocationTracker, op_name: []const u8, lhs: c.MlirValue, rhs: c.MlirValue, span: lib.ast.SourceSpan) c.MlirValue {
    const arith_op_name = if (std.mem.eql(u8, op_name, "ora.add"))
        "arith.addi"
    else if (std.mem.eql(u8, op_name, "ora.sub"))
        "arith.subi"
    else if (std.mem.eql(u8, op_name, "ora.mul"))
        "arith.muli"
    else if (std.mem.eql(u8, op_name, "ora.div"))
        "arith.divui"
    else if (std.mem.eql(u8, op_name, "ora.rem"))
        "arith.remui"
    else
        op_name;

    const lhs_type = c.mlirValueGetType(lhs);
    const rhs_type = c.mlirValueGetType(rhs);

    var rhs_converted = rhs;
    if (!c.mlirTypeEqual(lhs_type, rhs_type)) {
        rhs_converted = type_mapper.createConversionOp(block, rhs, lhs_type, span);
    }

    const loc = locations.createLocation(span);
    var state = c.mlirOperationStateGet(h.strRef(arith_op_name), loc);
    c.mlirOperationStateAddOperands(&state, 2, @ptrCast(&[_]c.MlirValue{ lhs, rhs_converted }));
    c.mlirOperationStateAddResults(&state, 1, @ptrCast(&lhs_type));
    const op = c.mlirOperationCreate(&state);
    h.appendOp(block, op);
    return h.getResult(op, 0);
}

/// Create comparison operation
pub fn createComparisonOp(ctx: c.MlirContext, block: c.MlirBlock, locations: LocationTracker, predicate: []const u8, lhs: c.MlirValue, rhs: c.MlirValue, span: lib.ast.SourceSpan) c.MlirValue {
    const lhs_ty = c.mlirValueGetType(lhs);
    const rhs_ty = c.mlirValueGetType(rhs);
    const is_lhs_address = c.oraTypeIsAddressType(lhs_ty);
    const is_rhs_address = c.oraTypeIsAddressType(rhs_ty);
    const loc = locations.createLocation(span);

    const lhs_converted = if (is_lhs_address) blk: {
        const addr_to_i160_op = c.oraAddrToI160OpCreate(ctx, loc, lhs);
        h.appendOp(block, addr_to_i160_op);
        break :blk h.getResult(addr_to_i160_op, 0);
    } else lhs;
    const rhs_converted = if (is_rhs_address) blk: {
        const addr_to_i160_op = c.oraAddrToI160OpCreate(ctx, loc, rhs);
        h.appendOp(block, addr_to_i160_op);
        break :blk h.getResult(addr_to_i160_op, 0);
    } else rhs;

    if (is_lhs_address or is_rhs_address) {
        // both operands should be converted to i160 for comparison
        // ensure both are the same type (i160)
        const i160_ty = c.mlirIntegerTypeGet(ctx, 160);
        const lhs_final = if (is_lhs_address) lhs_converted else blk: {
            // convert non-address to i160 if needed
            const lhs_ty_inner = c.mlirValueGetType(lhs);
            if (c.mlirTypeEqual(lhs_ty_inner, i160_ty)) {
                break :blk lhs;
            } else {
                // convert to i160 using arith.extui or arith.trunci
                const lhs_width = c.mlirIntegerTypeGetWidth(lhs_ty_inner);
                if (lhs_width < 160) {
                    var ext_state = h.opState("arith.extui", loc);
                    c.mlirOperationStateAddOperands(&ext_state, 1, @ptrCast(&[_]c.MlirValue{lhs}));
                    c.mlirOperationStateAddResults(&ext_state, 1, @ptrCast(&i160_ty));
                    const ext_op = c.mlirOperationCreate(&ext_state);
                    h.appendOp(block, ext_op);
                    break :blk h.getResult(ext_op, 0);
                } else {
                    var trunc_state = h.opState("arith.trunci", loc);
                    c.mlirOperationStateAddOperands(&trunc_state, 1, @ptrCast(&[_]c.MlirValue{lhs}));
                    c.mlirOperationStateAddResults(&trunc_state, 1, @ptrCast(&i160_ty));
                    const trunc_op = c.mlirOperationCreate(&trunc_state);
                    h.appendOp(block, trunc_op);
                    break :blk h.getResult(trunc_op, 0);
                }
            }
        };
        const rhs_final = if (is_rhs_address) rhs_converted else blk: {
            // convert non-address to i160 if needed
            const rhs_ty_inner = c.mlirValueGetType(rhs);
            if (c.mlirTypeEqual(rhs_ty_inner, i160_ty)) {
                break :blk rhs;
            } else {
                // convert to i160 using arith.extui or arith.trunci
                const rhs_width = c.mlirIntegerTypeGetWidth(rhs_ty_inner);
                if (rhs_width < 160) {
                    var ext_state = h.opState("arith.extui", loc);
                    c.mlirOperationStateAddOperands(&ext_state, 1, @ptrCast(&[_]c.MlirValue{rhs}));
                    c.mlirOperationStateAddResults(&ext_state, 1, @ptrCast(&i160_ty));
                    const ext_op = c.mlirOperationCreate(&ext_state);
                    h.appendOp(block, ext_op);
                    break :blk h.getResult(ext_op, 0);
                } else {
                    var trunc_state = h.opState("arith.trunci", loc);
                    c.mlirOperationStateAddOperands(&trunc_state, 1, @ptrCast(&[_]c.MlirValue{rhs}));
                    c.mlirOperationStateAddResults(&trunc_state, 1, @ptrCast(&i160_ty));
                    const trunc_op = c.mlirOperationCreate(&trunc_state);
                    h.appendOp(block, trunc_op);
                    break :blk h.getResult(trunc_op, 0);
                }
            }
        };

        var state = h.opState("arith.cmpi", loc);
        c.mlirOperationStateAddOperands(&state, 2, @ptrCast(&[_]c.MlirValue{ lhs_final, rhs_final }));
        const bool_ty = h.boolType(ctx);
        c.mlirOperationStateAddResults(&state, 1, @ptrCast(&bool_ty));

        const predicate_value = predicateStringToInt(predicate);
        const predicate_attr = c.mlirIntegerAttrGet(c.mlirIntegerTypeGet(ctx, 64), predicate_value);
        const predicate_id = h.identifier(ctx, "predicate");
        var attrs = [_]c.MlirNamedAttribute{
            c.mlirNamedAttributeGet(predicate_id, predicate_attr),
        };
        c.mlirOperationStateAddAttributes(&state, attrs.len, &attrs);

        const op = c.mlirOperationCreate(&state);
        h.appendOp(block, op);
        return h.getResult(op, 0);
    } else {
        var state = h.opState("ora.cmp", loc);
        const predicate_attr = h.stringAttr(ctx, predicate);
        const predicate_id = h.identifier(ctx, "predicate");
        var attrs = [_]c.MlirNamedAttribute{
            c.mlirNamedAttributeGet(predicate_id, predicate_attr),
        };
        c.mlirOperationStateAddAttributes(&state, attrs.len, &attrs);

        c.mlirOperationStateAddOperands(&state, 2, @ptrCast(&[_]c.MlirValue{ lhs_converted, rhs_converted }));
        const bool_ty = h.boolType(ctx);
        c.mlirOperationStateAddResults(&state, 1, @ptrCast(&bool_ty));

        const op = c.mlirOperationCreate(&state);
        h.appendOp(block, op);
        return h.getResult(op, 0);
    }
}

/// Convert predicate string to integer
pub fn predicateStringToInt(predicate: []const u8) i64 {
    if (std.mem.eql(u8, predicate, "eq")) return 0;
    if (std.mem.eql(u8, predicate, "ne")) return 1;
    if (std.mem.eql(u8, predicate, "slt")) return 2;
    if (std.mem.eql(u8, predicate, "sle")) return 3;
    if (std.mem.eql(u8, predicate, "sgt")) return 4;
    if (std.mem.eql(u8, predicate, "sge")) return 5;
    if (std.mem.eql(u8, predicate, "ult")) return 6;
    if (std.mem.eql(u8, predicate, "ule")) return 7;
    if (std.mem.eql(u8, predicate, "ugt")) return 8;
    if (std.mem.eql(u8, predicate, "uge")) return 9;
    std.debug.print("WARNING: Unknown predicate '{s}', defaulting to 'eq' (0)\n", .{predicate});
    return 0;
}

/// Get common type for binary operations
pub fn getCommonType(ctx: c.MlirContext, lhs_ty: c.MlirType, rhs_ty: c.MlirType) c.MlirType {
    if (c.mlirTypeEqual(lhs_ty, rhs_ty)) {
        return lhs_ty;
    }

    if (c.mlirTypeIsAInteger(lhs_ty) and c.mlirTypeIsAInteger(rhs_ty)) {
        const lhs_width = c.mlirIntegerTypeGetWidth(lhs_ty);
        const rhs_width = c.mlirIntegerTypeGetWidth(rhs_ty);

        if (lhs_width == 1 and rhs_width > 1) return rhs_ty;
        if (rhs_width == 1 and lhs_width > 1) return lhs_ty;
        if (lhs_width == rhs_width) return lhs_ty;
        return if (lhs_width >= rhs_width) lhs_ty else rhs_ty;
    }

    return c.mlirIntegerTypeGet(ctx, constants.DEFAULT_INTEGER_BITS);
}

/// Create boolean constant
pub fn createBoolConstant(_: c.MlirContext, block: c.MlirBlock, ora_dialect: *OraDialect, locations: LocationTracker, value: bool, span: lib.ast.SourceSpan) c.MlirValue {
    const loc = locations.createLocation(span);
    const op = ora_dialect.createArithConstantBool(value, loc);
    h.appendOp(block, op);
    return h.getResult(op, 0);
}

/// Create typed constant
pub fn createTypedConstant(_: c.MlirContext, block: c.MlirBlock, ora_dialect: *OraDialect, locations: LocationTracker, value: i64, ty: c.MlirType, span: lib.ast.SourceSpan) c.MlirValue {
    const loc = locations.createLocation(span);
    const op = ora_dialect.createArithConstant(value, ty, loc);
    h.appendOp(block, op);
    return h.getResult(op, 0);
}
