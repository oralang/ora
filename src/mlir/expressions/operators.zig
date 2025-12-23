// ============================================================================
// Operator Expression Lowering
// ============================================================================
// Lowering for binary and unary operations

const std = @import("std");
const c = @import("../c.zig").c;
const lib = @import("ora_lib");
const constants = @import("../lower.zig");
const h = @import("../helpers.zig");
const TypeMapper = @import("../types.zig").TypeMapper;
const LocationTracker = @import("../locations.zig").LocationTracker;
const OraDialect = @import("../dialect.zig").OraDialect;
const expr_helpers = @import("helpers.zig");

/// ExpressionLowerer type (forward declaration)
const ExpressionLowerer = @import("mod.zig").ExpressionLowerer;

/// Lower binary expressions
pub fn lowerBinary(
    self: *const ExpressionLowerer,
    bin: *const lib.ast.Expressions.BinaryExpr,
) c.MlirValue {
    const lhs = self.lowerExpression(bin.lhs);
    const rhs = self.lowerExpression(bin.rhs);

    const lhs_ty = c.mlirValueGetType(lhs);
    const rhs_ty = c.mlirValueGetType(rhs);
    const result_ty = self.getCommonType(lhs_ty, rhs_ty);

    const lhs_converted = self.convertToType(lhs, result_ty, bin.span);
    const rhs_converted = self.convertToType(rhs, result_ty, bin.span);

    // Check for Exact<T> division/modulo guard
    if (bin.operator == .Slash or bin.operator == .Percent) {
        const lhs_type_info = extractTypeInfo(bin.lhs);
        const rhs_type_info = extractTypeInfo(bin.rhs);

        const needs_exact_guard = if (lhs_type_info.ora_type) |lhs_ora_type|
            lhs_ora_type == .exact
        else
            false;

        const rhs_is_exact = if (rhs_type_info.ora_type) |rhs_ora_type|
            rhs_ora_type == .exact
        else
            false;

        if (needs_exact_guard or rhs_is_exact) {
            insertExactDivisionGuard(self, lhs_converted, rhs_converted, bin.span);
        }
    }

    return switch (bin.operator) {
        .Plus => self.createArithmeticOp("ora.add", lhs_converted, rhs_converted, result_ty, bin.span),
        .Minus => self.createArithmeticOp("ora.sub", lhs_converted, rhs_converted, result_ty, bin.span),
        .Star => self.createArithmeticOp("ora.mul", lhs_converted, rhs_converted, result_ty, bin.span),
        .Slash => self.createArithmeticOp("ora.div", lhs_converted, rhs_converted, result_ty, bin.span),
        .Percent => self.createArithmeticOp("ora.rem", lhs_converted, rhs_converted, result_ty, bin.span),
        .StarStar => lowerPowerOp(self, lhs_converted, rhs_converted, result_ty, bin.span),
        .EqualEqual => self.createComparisonOp("eq", lhs_converted, rhs_converted, bin.span),
        .BangEqual => self.createComparisonOp("ne", lhs_converted, rhs_converted, bin.span),
        .Less => self.createComparisonOp("ult", lhs_converted, rhs_converted, bin.span),
        .LessEqual => self.createComparisonOp("ule", lhs_converted, rhs_converted, bin.span),
        .Greater => self.createComparisonOp("ugt", lhs_converted, rhs_converted, bin.span),
        .GreaterEqual => self.createComparisonOp("uge", lhs_converted, rhs_converted, bin.span),
        .And => lowerLogicalAnd(self, bin),
        .Or => lowerLogicalOr(self, bin),
        .BitwiseAnd => self.createArithmeticOp("arith.andi", lhs_converted, rhs_converted, result_ty, bin.span),
        .BitwiseOr => self.createArithmeticOp("arith.ori", lhs_converted, rhs_converted, result_ty, bin.span),
        .BitwiseXor => self.createArithmeticOp("arith.xori", lhs_converted, rhs_converted, result_ty, bin.span),
        .LeftShift => self.createArithmeticOp("arith.shli", lhs_converted, rhs_converted, result_ty, bin.span),
        .RightShift => self.createArithmeticOp("arith.shrsi", lhs_converted, rhs_converted, result_ty, bin.span),
        .Comma => lowerCommaOp(self, lhs_converted, rhs_converted, result_ty, bin.span),
    };
}

/// Lower unary expressions
pub fn lowerUnary(
    self: *const ExpressionLowerer,
    unary: *const lib.ast.Expressions.UnaryExpr,
) c.MlirValue {
    const operand = self.lowerExpression(unary.operand);
    const operand_ty = c.mlirValueGetType(operand);

    return switch (unary.operator) {
        .Bang => lowerLogicalNot(self, operand, operand_ty, unary.span),
        .Minus => lowerUnaryMinus(self, operand, operand_ty, unary.span),
        .BitNot => lowerBitwiseNot(self, operand, operand_ty, unary.span),
    };
}

fn lowerPowerOp(
    self: *const ExpressionLowerer,
    lhs: c.MlirValue,
    rhs: c.MlirValue,
    result_ty: c.MlirType,
    span: lib.ast.SourceSpan,
) c.MlirValue {
    const power_op = self.ora_dialect.createPower(lhs, rhs, result_ty, self.fileLoc(span));
    h.appendOp(self.block, power_op);
    return h.getResult(power_op, 0);
}

fn lowerLogicalAnd(
    self: *const ExpressionLowerer,
    bin: *const lib.ast.Expressions.BinaryExpr,
) c.MlirValue {
    const lhs_val_raw = self.lowerExpression(bin.lhs);
    const bool_ty = h.boolType(self.ctx);
    const lhs_type = c.mlirValueGetType(lhs_val_raw);
    const lhs_val = if (c.mlirTypeEqual(lhs_type, bool_ty))
        lhs_val_raw
    else blk: {
        const zero_val = self.createConstant(0, bin.span);
        break :blk self.createComparisonOp("ne", lhs_val_raw, zero_val, bin.span);
    };

    const result_types = [_]c.MlirType{bool_ty};
    var state = h.opState("scf.if", self.fileLoc(bin.span));
    c.mlirOperationStateAddOperands(&state, 1, @ptrCast(&lhs_val));
    c.mlirOperationStateAddResults(&state, 1, &result_types);

    const then_region = c.mlirRegionCreate();
    const then_block = c.mlirBlockCreate(0, null, null);
    c.mlirRegionInsertOwnedBlock(then_region, 0, then_block);
    c.mlirOperationStateAddOwnedRegions(&state, 1, @ptrCast(&then_region));

    const else_region = c.mlirRegionCreate();
    const else_block = c.mlirBlockCreate(0, null, null);
    c.mlirRegionInsertOwnedBlock(else_region, 0, else_block);
    c.mlirOperationStateAddOwnedRegions(&state, 1, @ptrCast(&else_region));

    const if_op = c.mlirOperationCreate(&state);
    h.appendOp(self.block, if_op);

    const then_lowerer = ExpressionLowerer.init(self.ctx, then_block, self.type_mapper, self.param_map, self.storage_map, self.local_var_map, self.symbol_table, self.builtin_registry, self.locations, self.ora_dialect);
    const rhs_val_raw = then_lowerer.lowerExpression(bin.rhs);

    const rhs_type = c.mlirValueGetType(rhs_val_raw);
    const rhs_val = if (c.mlirTypeEqual(rhs_type, bool_ty))
        rhs_val_raw
    else blk: {
        const zero_val = then_lowerer.createConstant(0, bin.span);
        break :blk then_lowerer.createComparisonOp("ne", rhs_val_raw, zero_val, bin.span);
    };

    const then_yield_op = self.ora_dialect.createScfYieldWithValues(&[_]c.MlirValue{rhs_val}, self.fileLoc(bin.span));
    h.appendOp(then_block, then_yield_op);

    const false_val = then_lowerer.createBoolConstant(false, bin.span);
    const else_yield_op = self.ora_dialect.createScfYieldWithValues(&[_]c.MlirValue{false_val}, self.fileLoc(bin.span));
    h.appendOp(else_block, else_yield_op);

    return h.getResult(if_op, 0);
}

fn lowerLogicalOr(
    self: *const ExpressionLowerer,
    bin: *const lib.ast.Expressions.BinaryExpr,
) c.MlirValue {
    const lhs_val_raw = self.lowerExpression(bin.lhs);
    const bool_ty = h.boolType(self.ctx);
    const lhs_type = c.mlirValueGetType(lhs_val_raw);
    const lhs_val = if (c.mlirTypeEqual(lhs_type, bool_ty))
        lhs_val_raw
    else blk: {
        const zero_val = self.createConstant(0, bin.span);
        break :blk self.createComparisonOp("ne", lhs_val_raw, zero_val, bin.span);
    };

    const result_types = [_]c.MlirType{bool_ty};
    var state = h.opState("scf.if", self.fileLoc(bin.span));
    c.mlirOperationStateAddOperands(&state, 1, @ptrCast(&lhs_val));
    c.mlirOperationStateAddResults(&state, 1, &result_types);

    const then_region = c.mlirRegionCreate();
    const then_block = c.mlirBlockCreate(0, null, null);
    c.mlirRegionInsertOwnedBlock(then_region, 0, then_block);
    c.mlirOperationStateAddOwnedRegions(&state, 1, @ptrCast(&then_region));

    const else_region = c.mlirRegionCreate();
    const else_block = c.mlirBlockCreate(0, null, null);
    c.mlirRegionInsertOwnedBlock(else_region, 0, else_block);
    c.mlirOperationStateAddOwnedRegions(&state, 1, @ptrCast(&else_region));

    const if_op = c.mlirOperationCreate(&state);
    h.appendOp(self.block, if_op);

    const then_lowerer = ExpressionLowerer.init(self.ctx, then_block, self.type_mapper, self.param_map, self.storage_map, self.local_var_map, self.symbol_table, self.builtin_registry, self.locations, self.ora_dialect);
    const true_val = then_lowerer.createBoolConstant(true, bin.span);

    const then_yield_op = self.ora_dialect.createScfYieldWithValues(&[_]c.MlirValue{true_val}, self.fileLoc(bin.span));
    h.appendOp(then_block, then_yield_op);

    const else_lowerer = ExpressionLowerer.init(self.ctx, else_block, self.type_mapper, self.param_map, self.storage_map, self.local_var_map, self.symbol_table, self.builtin_registry, self.locations, self.ora_dialect);
    const rhs_val_raw = else_lowerer.lowerExpression(bin.rhs);

    const rhs_type = c.mlirValueGetType(rhs_val_raw);
    const rhs_val = if (c.mlirTypeEqual(rhs_type, bool_ty))
        rhs_val_raw
    else blk: {
        const zero_val = else_lowerer.createConstant(0, bin.span);
        break :blk else_lowerer.createComparisonOp("ne", rhs_val_raw, zero_val, bin.span);
    };

    const else_yield_op = self.ora_dialect.createScfYieldWithValues(&[_]c.MlirValue{rhs_val}, self.fileLoc(bin.span));
    h.appendOp(else_block, else_yield_op);

    return h.getResult(if_op, 0);
}

fn lowerCommaOp(
    _: *const ExpressionLowerer,
    _: c.MlirValue,
    rhs: c.MlirValue,
    _: c.MlirType,
    _: lib.ast.SourceSpan,
) c.MlirValue {
    // Comma expression: evaluate both sides, return the right-hand side
    // The left-hand side is evaluated for side effects but its value is discarded
    // We just need to ensure both are evaluated, then return the right-hand side
    // No need for ora.sequence - just return rhs (lhs is already evaluated)
    return rhs;
}

fn lowerLogicalNot(
    self: *const ExpressionLowerer,
    operand: c.MlirValue,
    operand_ty: c.MlirType,
    span: lib.ast.SourceSpan,
) c.MlirValue {
    if (c.mlirTypeIsAInteger(operand_ty) and c.mlirIntegerTypeGetWidth(operand_ty) == 1) {
        const one_val = self.createBoolConstant(true, span);
        return self.createArithmeticOp("arith.xori", operand, one_val, operand_ty, span);
    } else {
        const zero_val = self.createConstant(0, span);
        return self.createComparisonOp("eq", operand, zero_val, span);
    }
}

fn lowerUnaryMinus(
    self: *const ExpressionLowerer,
    operand: c.MlirValue,
    operand_ty: c.MlirType,
    span: lib.ast.SourceSpan,
) c.MlirValue {
    const zero_val = self.createTypedConstant(0, operand_ty, span);
    return self.createArithmeticOp("arith.subi", zero_val, operand, operand_ty, span);
}

fn lowerBitwiseNot(
    self: *const ExpressionLowerer,
    operand: c.MlirValue,
    operand_ty: c.MlirType,
    span: lib.ast.SourceSpan,
) c.MlirValue {
    const bit_width = if (c.mlirTypeIsAInteger(operand_ty))
        c.mlirIntegerTypeGetWidth(operand_ty)
    else
        constants.DEFAULT_INTEGER_BITS;

    const all_ones = if (bit_width >= 64)
        -1
    else
        (@as(i64, 1) << @intCast(bit_width)) - 1;
    const all_ones_val = self.createTypedConstant(all_ones, operand_ty, span);

    return self.createArithmeticOp("arith.xori", operand, all_ones_val, operand_ty, span);
}

/// Extract TypeInfo from an expression node
pub fn extractTypeInfo(expr: *const lib.ast.Expressions.ExprNode) lib.ast.Types.TypeInfo {
    return switch (expr.*) {
        .Binary => |b| b.type_info,
        .Unary => |u| u.type_info,
        .Identifier => |i| i.type_info,
        .Literal => |l| switch (l) {
            .Integer => |int_lit| int_lit.type_info,
            .String => |str_lit| str_lit.type_info,
            .Bool => |bool_lit| bool_lit.type_info,
            .Address => |addr_lit| addr_lit.type_info,
            .Hex => |hex_lit| hex_lit.type_info,
            .Binary => |bin_lit| bin_lit.type_info,
            .Character => |char_lit| char_lit.type_info,
            .Bytes => |bytes_lit| bytes_lit.type_info,
        },
        .Call => |call_expr| call_expr.type_info,
        .FieldAccess => |f| f.type_info,
        .Index => lib.ast.Types.TypeInfo.unknown(),
        .Cast => |cast_expr| cast_expr.target_type,
        .Assignment => |assign| if (assign.target.* == .Identifier)
            assign.target.Identifier.type_info
        else
            lib.ast.Types.TypeInfo.unknown(),
        .CompoundAssignment => lib.ast.Types.TypeInfo.unknown(),
        else => lib.ast.Types.TypeInfo.unknown(),
    };
}

/// Insert exactness guard for Exact<T> division
pub fn insertExactDivisionGuard(
    self: *const ExpressionLowerer,
    dividend: c.MlirValue,
    divisor: c.MlirValue,
    span: lib.ast.SourceSpan,
) void {
    const loc = self.fileLoc(span);
    const dividend_type = c.mlirValueGetType(dividend);

    var mod_state = h.opState("arith.remui", loc);
    c.mlirOperationStateAddOperands(&mod_state, 2, @ptrCast(&[_]c.MlirValue{ dividend, divisor }));
    c.mlirOperationStateAddResults(&mod_state, 1, @ptrCast(&dividend_type));
    const mod_op = c.mlirOperationCreate(&mod_state);
    h.appendOp(self.block, mod_op);
    const remainder = h.getResult(mod_op, 0);

    const zero_attr = c.mlirIntegerAttrGet(dividend_type, 0);
    var zero_state = h.opState("arith.constant", loc);
    const value_id = h.identifier(self.ctx, "value");
    var zero_attrs = [_]c.MlirNamedAttribute{c.mlirNamedAttributeGet(value_id, zero_attr)};
    c.mlirOperationStateAddAttributes(&zero_state, zero_attrs.len, &zero_attrs);
    c.mlirOperationStateAddResults(&zero_state, 1, @ptrCast(&dividend_type));
    const zero_op = c.mlirOperationCreate(&zero_state);
    h.appendOp(self.block, zero_op);
    const zero_const = h.getResult(zero_op, 0);

    var cmp_state = h.opState("arith.cmpi", loc);
    const pred_id = h.identifier(self.ctx, "predicate");
    const pred_eq_attr = c.mlirIntegerAttrGet(c.mlirIntegerTypeGet(self.ctx, 64), 0);
    var attrs = [_]c.MlirNamedAttribute{c.mlirNamedAttributeGet(pred_id, pred_eq_attr)};
    c.mlirOperationStateAddAttributes(&cmp_state, attrs.len, &attrs);
    c.mlirOperationStateAddOperands(&cmp_state, 2, @ptrCast(&[_]c.MlirValue{ remainder, zero_const }));
    c.mlirOperationStateAddResults(&cmp_state, 1, @ptrCast(&h.boolType(self.ctx)));
    const cmp_op = c.mlirOperationCreate(&cmp_state);
    h.appendOp(self.block, cmp_op);
    const condition = h.getResult(cmp_op, 0);

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
