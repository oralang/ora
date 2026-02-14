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
    const lhs_type_info = extractTypeInfo(bin.lhs);
    const rhs_type_info = extractTypeInfo(bin.rhs);
    const uses_signed_integer_semantics = usesSignedIntegerSemantics(lhs_type_info, rhs_type_info);
    const div_op_name = if (uses_signed_integer_semantics) "arith.divsi" else "arith.divui";
    const rem_op_name = if (uses_signed_integer_semantics) "arith.remsi" else "arith.remui";
    const lt_predicate = if (uses_signed_integer_semantics) "slt" else "ult";
    const le_predicate = if (uses_signed_integer_semantics) "sle" else "ule";
    const gt_predicate = if (uses_signed_integer_semantics) "sgt" else "ugt";
    const ge_predicate = if (uses_signed_integer_semantics) "sge" else "uge";
    const shr_op_name = if (uses_signed_integer_semantics) "arith.shrsi" else "arith.shrui";

    // check for Exact<T> division/modulo guard
    if (bin.operator == .Slash or bin.operator == .Percent) {
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
        // Checked arithmetic: compute result then assert no overflow
        .Plus => blk: {
            const result = self.createArithmeticOp("arith.addi", lhs_converted, rhs_converted, result_ty, bin.span);
            insertAddOverflowCheck(self, result, lhs_converted, rhs_converted, uses_signed_integer_semantics, bin.span);
            insertNarrowingOverflowCheck(self, result, bin.type_info, uses_signed_integer_semantics, bin.span);
            break :blk result;
        },
        .Minus => blk: {
            const result = self.createArithmeticOp("arith.subi", lhs_converted, rhs_converted, result_ty, bin.span);
            insertSubOverflowCheck(self, result, lhs_converted, rhs_converted, uses_signed_integer_semantics, bin.span);
            insertNarrowingOverflowCheck(self, result, bin.type_info, uses_signed_integer_semantics, bin.span);
            break :blk result;
        },
        .Star => blk: {
            const result = self.createArithmeticOp("arith.muli", lhs_converted, rhs_converted, result_ty, bin.span);
            insertMulOverflowCheck(self, result, lhs_converted, rhs_converted, uses_signed_integer_semantics, bin.span);
            insertNarrowingOverflowCheck(self, result, bin.type_info, uses_signed_integer_semantics, bin.span);
            break :blk result;
        },
        .Slash => blk: {
            const result = self.createArithmeticOp(div_op_name, lhs_converted, rhs_converted, result_ty, bin.span);
            insertDivChecks(self, lhs_converted, rhs_converted, uses_signed_integer_semantics, bin.span);
            break :blk result;
        },
        .Percent => blk: {
            const result = self.createArithmeticOp(rem_op_name, lhs_converted, rhs_converted, result_ty, bin.span);
            insertDivChecks(self, lhs_converted, rhs_converted, uses_signed_integer_semantics, bin.span);
            break :blk result;
        },
        .StarStar => lowerPowerOp(self, lhs_converted, rhs_converted, result_ty, bin.span),
        // Wrapping operators lower to raw modular arithmetic (no overflow trap)
        .WrappingAdd => self.createArithmeticOp("arith.addi", lhs_converted, rhs_converted, result_ty, bin.span),
        .WrappingSub => self.createArithmeticOp("arith.subi", lhs_converted, rhs_converted, result_ty, bin.span),
        .WrappingMul => self.createArithmeticOp("arith.muli", lhs_converted, rhs_converted, result_ty, bin.span),
        .WrappingShl => self.createArithmeticOp("arith.shli", lhs_converted, rhs_converted, result_ty, bin.span),
        .WrappingShr => self.createArithmeticOp(shr_op_name, lhs_converted, rhs_converted, result_ty, bin.span),
        .EqualEqual => self.createComparisonOp("eq", lhs_converted, rhs_converted, bin.span),
        .BangEqual => self.createComparisonOp("ne", lhs_converted, rhs_converted, bin.span),
        .Less => self.createComparisonOp(lt_predicate, lhs_converted, rhs_converted, bin.span),
        .LessEqual => self.createComparisonOp(le_predicate, lhs_converted, rhs_converted, bin.span),
        .Greater => self.createComparisonOp(gt_predicate, lhs_converted, rhs_converted, bin.span),
        .GreaterEqual => self.createComparisonOp(ge_predicate, lhs_converted, rhs_converted, bin.span),
        .And => lowerLogicalAnd(self, bin),
        .Or => lowerLogicalOr(self, bin),
        .BitwiseAnd => self.createArithmeticOp("arith.andi", lhs_converted, rhs_converted, result_ty, bin.span),
        .BitwiseOr => self.createArithmeticOp("arith.ori", lhs_converted, rhs_converted, result_ty, bin.span),
        .BitwiseXor => self.createArithmeticOp("arith.xori", lhs_converted, rhs_converted, result_ty, bin.span),
        .LeftShift => blk: {
            const result = self.createArithmeticOp("arith.shli", lhs_converted, rhs_converted, result_ty, bin.span);
            insertShiftBoundsCheck(self, rhs_converted, bin.span);
            break :blk result;
        },
        .RightShift => blk: {
            const result = self.createArithmeticOp(shr_op_name, lhs_converted, rhs_converted, result_ty, bin.span);
            insertShiftBoundsCheck(self, rhs_converted, bin.span);
            break :blk result;
        },
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
    const result = self.createArithmeticOp("arith.subi", zero_val, operand, operand_ty, span);
    // Guard: -INT_MIN overflows for signed types (MIN_INT = 1 << 255)
    insertNegateOverflowCheck(self, operand, span);
    return result;
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

fn isTopLevelSignedIntegerOraType(ora_type: lib.ast.Types.OraType) bool {
    return switch (ora_type) {
        .i8, .i16, .i32, .i64, .i128, .i256 => true,
        else => false,
    };
}

pub fn isSignedIntegerTypeInfo(type_info: lib.ast.Types.TypeInfo) bool {
    if (type_info.ora_type) |ora_type| {
        // Avoid dereferencing refinement base pointers here; some paths can
        // carry transient refinement wrappers that are not stable to recurse.
        return isTopLevelSignedIntegerOraType(ora_type);
    }
    return false;
}

pub fn usesSignedIntegerSemantics(
    lhs_type_info: lib.ast.Types.TypeInfo,
    rhs_type_info: lib.ast.Types.TypeInfo,
) bool {
    return isSignedIntegerTypeInfo(lhs_type_info) or isSignedIntegerTypeInfo(rhs_type_info);
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
    const divisor_type = c.oraValueGetType(divisor_unwrapped);

    const zero_divisor_op = self.ora_dialect.createArithConstant(0, divisor_type, loc);
    h.appendOp(self.block, zero_divisor_op);
    const zero_divisor = h.getResult(zero_divisor_op, 0);

    // divisor != 0
    const non_zero_op = c.oraArithCmpIOpCreate(self.ctx, loc, 1, divisor_unwrapped, zero_divisor);
    h.appendOp(self.block, non_zero_op);
    const non_zero = h.getResult(non_zero_op, 0);

    const mod_op = c.oraArithRemUIOpCreate(self.ctx, loc, dividend_unwrapped, divisor_unwrapped);
    c.oraOperationSetAttributeByName(mod_op, h.strRef("ora.guard_internal"), h.stringAttr(self.ctx, "true"));
    h.appendOp(self.block, mod_op);
    const remainder = h.getResult(mod_op, 0);

    const zero_op = self.ora_dialect.createArithConstant(0, dividend_type, loc);
    h.appendOp(self.block, zero_op);
    const zero_const = h.getResult(zero_op, 0);

    const cmp_op = c.oraArithCmpIOpCreate(self.ctx, loc, 0, remainder, zero_const);
    h.appendOp(self.block, cmp_op);
    const exact = h.getResult(cmp_op, 0);

    // divisor != 0 && dividend % divisor == 0
    const and_op = c.oraArithAndIOpCreate(self.ctx, loc, non_zero, exact);
    h.appendOp(self.block, and_op);
    const condition = h.getResult(and_op, 0);

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

// ============================================================================
// Checked Arithmetic Overflow Guards
// ============================================================================
// For checked (default) operators, emit overflow detection + assert.
// Unsigned addition overflow:  result < a
// Unsigned subtraction overflow: a < b
// Unsigned mul overflow: b != 0 && result / b != a
// Signed variants use signed comparison predicates.

/// Emit assert(!overflow) for checked addition.
/// Unsigned: overflow iff result < a.
/// Signed: overflow iff (b > 0 && result < a) || (b < 0 && result > a).
fn insertAddOverflowCheck(
    self: *const ExpressionLowerer,
    result: c.MlirValue,
    lhs: c.MlirValue,
    rhs: c.MlirValue,
    is_signed: bool,
    span: lib.ast.SourceSpan,
) void {
    const loc = self.fileLoc(span);
    const result_uw = expr_helpers.unwrapRefinementValue(self.ctx, self.block, self.locations, self.refinement_base_cache, result, span);
    const lhs_uw = expr_helpers.unwrapRefinementValue(self.ctx, self.block, self.locations, self.refinement_base_cache, lhs, span);
    if (!is_signed) {
        // Unsigned: overflow iff result < a (ult)
        const cmp = c.oraArithCmpIOpCreate(self.ctx, loc, expr_helpers.predicateStringToInt("ult"), result_uw, lhs_uw);
        h.appendOp(self.block, cmp);
        emitOverflowAssert(self, h.getResult(cmp, 0), "checked addition overflow", span);
    } else {
        // Signed: overflow iff ((result ^ a) & (result ^ b)) < 0 (sign bit set)
        const rhs_uw = expr_helpers.unwrapRefinementValue(self.ctx, self.block, self.locations, self.refinement_base_cache, rhs, span);
        const xor_ra = c.oraArithXorIOpCreate(self.ctx, loc, result_uw, lhs_uw);
        h.appendOp(self.block, xor_ra);
        const xor_rb = c.oraArithXorIOpCreate(self.ctx, loc, result_uw, rhs_uw);
        h.appendOp(self.block, xor_rb);
        const and_op = c.oraArithAndIOpCreate(self.ctx, loc, h.getResult(xor_ra, 0), h.getResult(xor_rb, 0));
        h.appendOp(self.block, and_op);
        const zero_op = self.ora_dialect.createArithConstant(0, c.oraValueGetType(result_uw), loc);
        h.appendOp(self.block, zero_op);
        const cmp = c.oraArithCmpIOpCreate(self.ctx, loc, expr_helpers.predicateStringToInt("slt"), h.getResult(and_op, 0), h.getResult(zero_op, 0));
        h.appendOp(self.block, cmp);
        emitOverflowAssert(self, h.getResult(cmp, 0), "checked signed addition overflow", span);
    }
}

/// Emit assert(!overflow) for checked subtraction.
/// Unsigned: overflow iff a < b.
fn insertSubOverflowCheck(
    self: *const ExpressionLowerer,
    result: c.MlirValue,
    lhs: c.MlirValue,
    rhs: c.MlirValue,
    is_signed: bool,
    span: lib.ast.SourceSpan,
) void {
    const loc = self.fileLoc(span);
    const lhs_uw = expr_helpers.unwrapRefinementValue(self.ctx, self.block, self.locations, self.refinement_base_cache, lhs, span);
    const rhs_uw = expr_helpers.unwrapRefinementValue(self.ctx, self.block, self.locations, self.refinement_base_cache, rhs, span);
    if (!is_signed) {
        // Unsigned: overflow iff a < b (ult)
        const cmp = c.oraArithCmpIOpCreate(self.ctx, loc, expr_helpers.predicateStringToInt("ult"), lhs_uw, rhs_uw);
        h.appendOp(self.block, cmp);
        emitOverflowAssert(self, h.getResult(cmp, 0), "checked subtraction overflow", span);
    } else {
        // Signed: overflow iff ((a ^ b) & (result ^ a)) < 0 (sign bit set)
        const result_uw = expr_helpers.unwrapRefinementValue(self.ctx, self.block, self.locations, self.refinement_base_cache, result, span);
        const xor_ab = c.oraArithXorIOpCreate(self.ctx, loc, lhs_uw, rhs_uw);
        h.appendOp(self.block, xor_ab);
        const xor_ra = c.oraArithXorIOpCreate(self.ctx, loc, result_uw, lhs_uw);
        h.appendOp(self.block, xor_ra);
        const and_op = c.oraArithAndIOpCreate(self.ctx, loc, h.getResult(xor_ab, 0), h.getResult(xor_ra, 0));
        h.appendOp(self.block, and_op);
        const zero_op = self.ora_dialect.createArithConstant(0, c.oraValueGetType(result_uw), loc);
        h.appendOp(self.block, zero_op);
        const cmp = c.oraArithCmpIOpCreate(self.ctx, loc, expr_helpers.predicateStringToInt("slt"), h.getResult(and_op, 0), h.getResult(zero_op, 0));
        h.appendOp(self.block, cmp);
        emitOverflowAssert(self, h.getResult(cmp, 0), "checked signed subtraction overflow", span);
    }
}

/// Emit assert(!overflow) for checked multiplication.
/// Unsigned: overflow iff (b != 0 && result / b != a).
fn insertMulOverflowCheck(
    self: *const ExpressionLowerer,
    result: c.MlirValue,
    lhs: c.MlirValue,
    rhs: c.MlirValue,
    is_signed: bool,
    span: lib.ast.SourceSpan,
) void {
    const loc = self.fileLoc(span);
    const result_uw = expr_helpers.unwrapRefinementValue(self.ctx, self.block, self.locations, self.refinement_base_cache, result, span);
    const lhs_uw = expr_helpers.unwrapRefinementValue(self.ctx, self.block, self.locations, self.refinement_base_cache, lhs, span);
    const rhs_uw = expr_helpers.unwrapRefinementValue(self.ctx, self.block, self.locations, self.refinement_base_cache, rhs, span);
    const val_ty = c.oraValueGetType(rhs_uw);
    const pred_eq = expr_helpers.predicateStringToInt("eq");
    const pred_ne = expr_helpers.predicateStringToInt("ne");

    const zero_op = self.ora_dialect.createArithConstant(0, val_ty, loc);
    h.appendOp(self.block, zero_op);
    const zero = h.getResult(zero_op, 0);

    // b_nonzero = b != 0
    const b_nz_cmp = c.oraArithCmpIOpCreate(self.ctx, loc, pred_ne, rhs_uw, zero);
    h.appendOp(self.block, b_nz_cmp);
    const b_nonzero = h.getResult(b_nz_cmp, 0);

    if (!is_signed) {
        // Unsigned: overflow iff b != 0 && result / b != a
        const quot_op = c.oraArithDivUIOpCreate(self.ctx, loc, result_uw, rhs_uw);
        c.oraOperationSetAttributeByName(quot_op, h.strRef("ora.guard_internal"), h.stringAttr(self.ctx, "true"));
        h.appendOp(self.block, quot_op);
        const mismatch_cmp = c.oraArithCmpIOpCreate(self.ctx, loc, pred_ne, h.getResult(quot_op, 0), lhs_uw);
        h.appendOp(self.block, mismatch_cmp);
        const and_op = c.oraArithAndIOpCreate(self.ctx, loc, h.getResult(mismatch_cmp, 0), b_nonzero);
        h.appendOp(self.block, and_op);
        emitOverflowAssert(self, h.getResult(and_op, 0), "checked multiplication overflow", span);
    } else {
        // Signed: special case a == MIN_INT && b == -1 (sdiv(MIN,-1) = MIN, masking the bug)
        // MIN_INT = 1 << 255
        const one_op = self.ora_dialect.createArithConstant(1, val_ty, loc);
        h.appendOp(self.block, one_op);
        const shift_op = self.ora_dialect.createArithConstant(255, val_ty, loc);
        h.appendOp(self.block, shift_op);
        const min_int_op = c.oraArithShlIOpCreate(self.ctx, loc, h.getResult(one_op, 0), h.getResult(shift_op, 0));
        h.appendOp(self.block, min_int_op);
        const min_int = h.getResult(min_int_op, 0);

        const neg1_op = self.ora_dialect.createArithConstant(-1, val_ty, loc);
        h.appendOp(self.block, neg1_op);
        const neg1 = h.getResult(neg1_op, 0);

        // special = (a == MIN_INT) && (b == -1)
        const a_min_cmp = c.oraArithCmpIOpCreate(self.ctx, loc, pred_eq, lhs_uw, min_int);
        h.appendOp(self.block, a_min_cmp);
        const b_neg1_cmp = c.oraArithCmpIOpCreate(self.ctx, loc, pred_eq, rhs_uw, neg1);
        h.appendOp(self.block, b_neg1_cmp);
        const special = c.oraArithAndIOpCreate(self.ctx, loc, h.getResult(a_min_cmp, 0), h.getResult(b_neg1_cmp, 0));
        h.appendOp(self.block, special);

        // general = b != 0 && sdiv(result, b) != a
        const quot_op = c.oraArithDivSIOpCreate(self.ctx, loc, result_uw, rhs_uw);
        c.oraOperationSetAttributeByName(quot_op, h.strRef("ora.guard_internal"), h.stringAttr(self.ctx, "true"));
        h.appendOp(self.block, quot_op);
        const mismatch_cmp = c.oraArithCmpIOpCreate(self.ctx, loc, pred_ne, h.getResult(quot_op, 0), lhs_uw);
        h.appendOp(self.block, mismatch_cmp);
        const general = c.oraArithAndIOpCreate(self.ctx, loc, h.getResult(mismatch_cmp, 0), b_nonzero);
        h.appendOp(self.block, general);

        // overflow = special || general
        const or_op = c.oraArithOrIOpCreate(self.ctx, loc, h.getResult(special, 0), h.getResult(general, 0));
        h.appendOp(self.block, or_op);
        emitOverflowAssert(self, h.getResult(or_op, 0), "checked signed multiplication overflow", span);
    }
}

/// Emit div-by-zero and signed-division-overflow guards for / and %.
/// - Always: assert(divisor != 0)
/// - Signed: assert(!(dividend == MIN_INT && divisor == -1))
fn insertDivChecks(
    self: *const ExpressionLowerer,
    lhs: c.MlirValue,
    rhs: c.MlirValue,
    is_signed: bool,
    span: lib.ast.SourceSpan,
) void {
    const loc = self.fileLoc(span);
    const rhs_uw = expr_helpers.unwrapRefinementValue(self.ctx, self.block, self.locations, self.refinement_base_cache, rhs, span);
    const val_ty = c.oraValueGetType(rhs_uw);

    // assert(divisor != 0)
    const zero_op = self.ora_dialect.createArithConstant(0, val_ty, loc);
    h.appendOp(self.block, zero_op);
    const ne_cmp = c.oraArithCmpIOpCreate(self.ctx, loc, expr_helpers.predicateStringToInt("ne"), rhs_uw, h.getResult(zero_op, 0));
    h.appendOp(self.block, ne_cmp);
    const assert_nz = self.ora_dialect.createAssert(h.getResult(ne_cmp, 0), loc, "division by zero");
    h.appendOp(self.block, assert_nz);

    if (is_signed) {
        // assert(!(a == MIN_INT && b == -1))
        const lhs_uw = expr_helpers.unwrapRefinementValue(self.ctx, self.block, self.locations, self.refinement_base_cache, lhs, span);
        const one_op = self.ora_dialect.createArithConstant(1, val_ty, loc);
        h.appendOp(self.block, one_op);
        const shift_op = self.ora_dialect.createArithConstant(255, val_ty, loc);
        h.appendOp(self.block, shift_op);
        const min_int_op = c.oraArithShlIOpCreate(self.ctx, loc, h.getResult(one_op, 0), h.getResult(shift_op, 0));
        h.appendOp(self.block, min_int_op);
        const neg1_op = self.ora_dialect.createArithConstant(-1, val_ty, loc);
        h.appendOp(self.block, neg1_op);

        const a_min = c.oraArithCmpIOpCreate(self.ctx, loc, expr_helpers.predicateStringToInt("eq"), lhs_uw, h.getResult(min_int_op, 0));
        h.appendOp(self.block, a_min);
        const b_neg1 = c.oraArithCmpIOpCreate(self.ctx, loc, expr_helpers.predicateStringToInt("eq"), rhs_uw, h.getResult(neg1_op, 0));
        h.appendOp(self.block, b_neg1);
        const both = c.oraArithAndIOpCreate(self.ctx, loc, h.getResult(a_min, 0), h.getResult(b_neg1, 0));
        h.appendOp(self.block, both);
        emitOverflowAssert(self, h.getResult(both, 0), "signed division overflow (MIN_INT / -1)", span);
    }
}

/// Emit overflow guard for unary negation: -MIN_INT overflows.
fn insertNegateOverflowCheck(
    self: *const ExpressionLowerer,
    operand: c.MlirValue,
    span: lib.ast.SourceSpan,
) void {
    const loc = self.fileLoc(span);
    const operand_uw = expr_helpers.unwrapRefinementValue(self.ctx, self.block, self.locations, self.refinement_base_cache, operand, span);
    const val_ty = c.oraValueGetType(operand_uw);

    // MIN_INT = 1 << 255
    const one_op = self.ora_dialect.createArithConstant(1, val_ty, loc);
    h.appendOp(self.block, one_op);
    const shift_op = self.ora_dialect.createArithConstant(255, val_ty, loc);
    h.appendOp(self.block, shift_op);
    const min_int_op = c.oraArithShlIOpCreate(self.ctx, loc, h.getResult(one_op, 0), h.getResult(shift_op, 0));
    h.appendOp(self.block, min_int_op);

    // assert(operand != MIN_INT)
    const ne_cmp = c.oraArithCmpIOpCreate(self.ctx, loc, expr_helpers.predicateStringToInt("ne"), operand_uw, h.getResult(min_int_op, 0));
    h.appendOp(self.block, ne_cmp);
    const assert_op = self.ora_dialect.createAssert(h.getResult(ne_cmp, 0), loc, "signed negation overflow (negating MIN_INT)");
    h.appendOp(self.block, assert_op);
}

/// Emit assert(shift_amount < 256) for checked shifts.
fn insertShiftBoundsCheck(
    self: *const ExpressionLowerer,
    shift_amt: c.MlirValue,
    span: lib.ast.SourceSpan,
) void {
    const loc = self.fileLoc(span);
    const amt_uw = expr_helpers.unwrapRefinementValue(self.ctx, self.block, self.locations, self.refinement_base_cache, shift_amt, span);
    const val_ty = c.oraValueGetType(amt_uw);
    const limit_op = self.ora_dialect.createArithConstant(256, val_ty, loc);
    h.appendOp(self.block, limit_op);
    const cmp = c.oraArithCmpIOpCreate(self.ctx, loc, expr_helpers.predicateStringToInt("ult"), amt_uw, h.getResult(limit_op, 0));
    h.appendOp(self.block, cmp);
    const assert_op = self.ora_dialect.createAssert(h.getResult(cmp, 0), loc, "shift amount exceeds bit width");
    h.appendOp(self.block, assert_op);
}

/// Emit narrowing overflow check when the Ora-level type is narrower than the
/// MLIR computation type (i256). Without this, u8: 255+1 passes the i256
/// overflow check but silently wraps to 0 after truncation.
fn insertNarrowingOverflowCheck(
    self: *const ExpressionLowerer,
    result: c.MlirValue,
    type_info: lib.ast.Types.TypeInfo,
    is_signed: bool,
    span: lib.ast.SourceSpan,
) void {
    const ora_type = type_info.ora_type orelse return;
    const bit_width = ora_type.bitWidth() orelse return;
    if (bit_width >= 256) return; // native width — already covered

    const result_uw = expr_helpers.unwrapRefinementValue(
        self.ctx, self.block, self.locations, self.refinement_base_cache, result, span,
    );
    const loc = self.fileLoc(span);

    if (!is_signed) {
        // Unsigned: truncate to iN then zero-extend back — if different, overflow.
        const narrow_ty = c.oraIntegerTypeCreate(self.ctx, bit_width);
        const wide_ty = c.oraValueGetType(result_uw);
        const trunc_op = c.oraArithTruncIOpCreate(self.ctx, loc, result_uw, narrow_ty);
        h.appendOp(self.block, trunc_op);
        const ext_op = c.oraArithExtUIOpCreate(self.ctx, loc, h.getResult(trunc_op, 0), wide_ty);
        h.appendOp(self.block, ext_op);
        const cmp = c.oraArithCmpIOpCreate(
            self.ctx, loc, expr_helpers.predicateStringToInt("ne"),
            result_uw, h.getResult(ext_op, 0),
        );
        h.appendOp(self.block, cmp);
        emitOverflowAssert(self, h.getResult(cmp, 0), "checked narrowing overflow", span);
    } else {
        // Signed: result must be in [min_signed, max_signed].
        // Use string-based constants to avoid Zig integer overflow for large widths.
        const wide_ty = c.oraValueGetType(result_uw);
        const max_u256 = (@as(u256, 1) << @intCast(bit_width - 1)) - 1;
        const min_u256_abs = @as(u256, 1) << @intCast(bit_width - 1);
        var max_buf: [80]u8 = undefined;
        var min_buf: [80]u8 = undefined;
        const max_str = std.fmt.bufPrint(&max_buf, "{}", .{max_u256}) catch unreachable;
        const min_str = std.fmt.bufPrint(&min_buf, "-{}", .{min_u256_abs}) catch unreachable;
        const max_attr = c.oraIntegerAttrGetFromString(wide_ty, h.strRef(max_str));
        const max_op = c.oraArithConstantOpCreate(self.ctx, loc, wide_ty, max_attr);
        h.appendOp(self.block, max_op);
        const min_attr = c.oraIntegerAttrGetFromString(wide_ty, h.strRef(min_str));
        const min_op = c.oraArithConstantOpCreate(self.ctx, loc, wide_ty, min_attr);
        h.appendOp(self.block, min_op);
        // overflow if result > max OR result < min
        const cmp_gt = c.oraArithCmpIOpCreate(
            self.ctx, loc, expr_helpers.predicateStringToInt("sgt"),
            result_uw, h.getResult(max_op, 0),
        );
        h.appendOp(self.block, cmp_gt);
        const cmp_lt = c.oraArithCmpIOpCreate(
            self.ctx, loc, expr_helpers.predicateStringToInt("slt"),
            result_uw, h.getResult(min_op, 0),
        );
        h.appendOp(self.block, cmp_lt);
        const or_op = c.oraArithOrIOpCreate(self.ctx, loc, h.getResult(cmp_gt, 0), h.getResult(cmp_lt, 0));
        h.appendOp(self.block, or_op);
        emitOverflowAssert(self, h.getResult(or_op, 0), "checked signed narrowing overflow", span);
    }
}

/// Emit an ora.assert(!flag) for overflow. The flag is true when overflow occurred.
fn emitOverflowAssert(
    self: *const ExpressionLowerer,
    overflow_flag: c.MlirValue,
    message: []const u8,
    span: lib.ast.SourceSpan,
) void {
    const loc = self.fileLoc(span);
    // Invert: condition for assert is "no overflow"
    const true_op = self.ora_dialect.createArithConstantBool(true, loc);
    h.appendOp(self.block, true_op);
    const true_val = h.getResult(true_op, 0);

    const not_op = c.oraArithXorIOpCreate(self.ctx, loc, overflow_flag, true_val);
    h.appendOp(self.block, not_op);
    const no_overflow = h.getResult(not_op, 0);

    const assert_op = self.ora_dialect.createAssert(no_overflow, loc, message);
    h.appendOp(self.block, assert_op);
}
