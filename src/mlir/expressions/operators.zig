// ============================================================================
// Operator Expression Lowering
// ============================================================================
// Lowering for binary and unary operations

const std = @import("std");
const c = @import("mlir_c_api").c;
const lib = @import("ora_lib");
const constants = @import("../lower.zig");
const h = @import("../helpers.zig");
const TypeMapper = @import("../types.zig").TypeMapper;
const LocationTracker = @import("../locations.zig").LocationTracker;
const OraDialect = @import("../dialect.zig").OraDialect;
const expr_helpers = @import("helpers.zig");
const guard_helpers = @import("../refinement_guard_helpers.zig");

/// ExpressionLowerer type (forward declaration)
const ExpressionLowerer = @import("mod.zig").ExpressionLowerer;

/// Lower binary expressions
pub fn lowerBinary(
    self: *const ExpressionLowerer,
    bin: *const lib.ast.Expressions.BinaryExpr,
) c.MlirValue {
    const lhs = self.lowerExpression(bin.lhs);
    const rhs = self.lowerExpression(bin.rhs);

    const lhs_ty = c.oraValueGetType(lhs);
    const rhs_ty = c.oraValueGetType(rhs);
    const result_ty = self.getCommonType(lhs_ty, rhs_ty);

    const lhs_converted = self.convertToType(lhs, result_ty, bin.span);
    const rhs_converted = self.convertToType(rhs, result_ty, bin.span);

    // check for Exact<T> division/modulo guard
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
        .Plus => self.createArithmeticOp("arith.addi", lhs_converted, rhs_converted, result_ty, bin.span),
        .Minus => self.createArithmeticOp("arith.subi", lhs_converted, rhs_converted, result_ty, bin.span),
        .Star => self.createArithmeticOp("arith.muli", lhs_converted, rhs_converted, result_ty, bin.span),
        .Slash => self.createArithmeticOp("arith.divui", lhs_converted, rhs_converted, result_ty, bin.span),
        .Percent => self.createArithmeticOp("arith.remui", lhs_converted, rhs_converted, result_ty, bin.span),
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
    const operand_ty = c.oraValueGetType(operand);

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
    const lhs_type = c.oraValueGetType(lhs_val_raw);
    const lhs_val = if (c.oraTypeEqual(lhs_type, bool_ty))
        lhs_val_raw
    else blk: {
        const zero_val = self.createConstant(0, bin.span);
        break :blk self.createComparisonOp("ne", lhs_val_raw, zero_val, bin.span);
    };

    const result_types = [_]c.MlirType{bool_ty};
    const if_op = self.ora_dialect.createScfIf(lhs_val, result_types[0..], self.fileLoc(bin.span));
    h.appendOp(self.block, if_op);
    const then_block = c.oraScfIfOpGetThenBlock(if_op);
    const else_block = c.oraScfIfOpGetElseBlock(if_op);
    if (c.oraBlockIsNull(then_block) or c.oraBlockIsNull(else_block)) {
        @panic("scf.if missing then/else blocks");
    }

    var then_lowerer = ExpressionLowerer.init(self.ctx, then_block, self.type_mapper, self.param_map, self.storage_map, self.local_var_map, self.symbol_table, self.builtin_registry, self.error_handler, self.locations, self.ora_dialect);
    then_lowerer.refinement_base_cache = self.refinement_base_cache;
    then_lowerer.refinement_guard_cache = self.refinement_guard_cache;
    then_lowerer.current_function_return_type = self.current_function_return_type;
    then_lowerer.current_function_return_type_info = self.current_function_return_type_info;
    then_lowerer.in_try_block = self.in_try_block;
    const rhs_val_raw = then_lowerer.lowerExpression(bin.rhs);

    const rhs_type = c.oraValueGetType(rhs_val_raw);
    const rhs_val = if (c.oraTypeEqual(rhs_type, bool_ty))
        rhs_val_raw
    else blk: {
        const zero_val = then_lowerer.createConstant(0, bin.span);
        break :blk then_lowerer.createComparisonOp("ne", rhs_val_raw, zero_val, bin.span);
    };

    const then_yield_op = self.ora_dialect.createScfYieldWithValues(&[_]c.MlirValue{rhs_val}, self.fileLoc(bin.span));
    h.appendOp(then_block, then_yield_op);

    var else_lowerer = ExpressionLowerer.init(self.ctx, else_block, self.type_mapper, self.param_map, self.storage_map, self.local_var_map, self.symbol_table, self.builtin_registry, self.error_handler, self.locations, self.ora_dialect);
    else_lowerer.refinement_base_cache = self.refinement_base_cache;
    else_lowerer.refinement_guard_cache = self.refinement_guard_cache;
    else_lowerer.current_function_return_type = self.current_function_return_type;
    else_lowerer.current_function_return_type_info = self.current_function_return_type_info;
    else_lowerer.in_try_block = self.in_try_block;
    const false_val = else_lowerer.createBoolConstant(false, bin.span);
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
    const lhs_type = c.oraValueGetType(lhs_val_raw);
    const lhs_val = if (c.oraTypeEqual(lhs_type, bool_ty))
        lhs_val_raw
    else blk: {
        const zero_val = self.createConstant(0, bin.span);
        break :blk self.createComparisonOp("ne", lhs_val_raw, zero_val, bin.span);
    };

    const result_types = [_]c.MlirType{bool_ty};
    const if_op = self.ora_dialect.createScfIf(lhs_val, result_types[0..], self.fileLoc(bin.span));
    h.appendOp(self.block, if_op);
    const then_block = c.oraScfIfOpGetThenBlock(if_op);
    const else_block = c.oraScfIfOpGetElseBlock(if_op);
    if (c.oraBlockIsNull(then_block) or c.oraBlockIsNull(else_block)) {
        @panic("scf.if missing then/else blocks");
    }

    var then_lowerer = ExpressionLowerer.init(self.ctx, then_block, self.type_mapper, self.param_map, self.storage_map, self.local_var_map, self.symbol_table, self.builtin_registry, self.error_handler, self.locations, self.ora_dialect);
    then_lowerer.refinement_base_cache = self.refinement_base_cache;
    then_lowerer.refinement_guard_cache = self.refinement_guard_cache;
    then_lowerer.current_function_return_type = self.current_function_return_type;
    then_lowerer.current_function_return_type_info = self.current_function_return_type_info;
    then_lowerer.in_try_block = self.in_try_block;
    const true_val = then_lowerer.createBoolConstant(true, bin.span);

    const then_yield_op = self.ora_dialect.createScfYieldWithValues(&[_]c.MlirValue{true_val}, self.fileLoc(bin.span));
    h.appendOp(then_block, then_yield_op);

    var else_lowerer = ExpressionLowerer.init(self.ctx, else_block, self.type_mapper, self.param_map, self.storage_map, self.local_var_map, self.symbol_table, self.builtin_registry, self.error_handler, self.locations, self.ora_dialect);
    else_lowerer.refinement_base_cache = self.refinement_base_cache;
    else_lowerer.refinement_guard_cache = self.refinement_guard_cache;
    else_lowerer.current_function_return_type = self.current_function_return_type;
    else_lowerer.current_function_return_type_info = self.current_function_return_type_info;
    else_lowerer.in_try_block = self.in_try_block;
    const rhs_val_raw = else_lowerer.lowerExpression(bin.rhs);

    const rhs_type = c.oraValueGetType(rhs_val_raw);
    const rhs_val = if (c.oraTypeEqual(rhs_type, bool_ty))
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
    // comma expression: evaluate both sides, return the right-hand side
    // the left-hand side is evaluated for side effects but its value is discarded
    // we just need to ensure both are evaluated, then return the right-hand side
    // no need for ora.sequence - just return rhs (lhs is already evaluated)
    return rhs;
}

fn lowerLogicalNot(
    self: *const ExpressionLowerer,
    operand: c.MlirValue,
    operand_ty: c.MlirType,
    span: lib.ast.SourceSpan,
) c.MlirValue {
    if (c.oraTypeIsAInteger(operand_ty) and c.oraIntegerTypeGetWidth(operand_ty) == 1) {
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
    const bit_width = if (c.oraTypeIsAInteger(operand_ty))
        c.oraIntegerTypeGetWidth(operand_ty)
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
    const dividend_unwrapped = expr_helpers.unwrapRefinementValue(self.ctx, self.block, self.locations, self.refinement_base_cache, dividend, span);
    const divisor_unwrapped = expr_helpers.unwrapRefinementValue(self.ctx, self.block, self.locations, self.refinement_base_cache, divisor, span);
    const dividend_type = c.oraValueGetType(dividend_unwrapped);

    const mod_op = c.oraArithRemUIOpCreate(self.ctx, loc, dividend_unwrapped, divisor_unwrapped);
    h.appendOp(self.block, mod_op);
    const remainder = h.getResult(mod_op, 0);

    const zero_op = self.ora_dialect.createArithConstant(0, dividend_type, loc);
    h.appendOp(self.block, zero_op);
    const zero_const = h.getResult(zero_op, 0);

    const cmp_op = c.oraArithCmpIOpCreate(self.ctx, loc, 0, remainder, zero_const);
    h.appendOp(self.block, cmp_op);
    const condition = h.getResult(cmp_op, 0);

    const msg = "Refinement violation: Exact<T> division must have no remainder";
    guard_helpers.emitRefinementGuard(
        self.ctx,
        self.block,
        self.ora_dialect,
        self.locations,
        self.refinement_guard_cache,
        span,
        condition,
        msg,
        "exact_division",
        null,
        std.heap.page_allocator,
    );
}
