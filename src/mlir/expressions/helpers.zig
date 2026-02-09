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
const log = @import("log");

/// Create a constant value
pub fn createConstant(ctx: c.MlirContext, block: c.MlirBlock, ora_dialect: *OraDialect, locations: LocationTracker, value: i64, span: lib.ast.SourceSpan) c.MlirValue {
    const ty = c.oraIntegerTypeCreate(ctx, constants.DEFAULT_INTEGER_BITS);
    const loc = locations.createLocation(span);
    const op = ora_dialect.createArithConstant(value, ty, loc);
    h.appendOp(block, op);
    return h.getResult(op, 0);
}

/// Create an error placeholder value
pub fn createErrorPlaceholder(ctx: c.MlirContext, block: c.MlirBlock, locations: LocationTracker, span: lib.ast.SourceSpan, error_msg: []const u8) c.MlirValue {
    const ty = c.oraIntegerTypeCreate(ctx, constants.DEFAULT_INTEGER_BITS);
    const loc = locations.createLocation(span);
    const attr = c.oraIntegerAttrCreateI64FromType(ty, 0);
    const error_attr = h.stringAttr(ctx, error_msg);

    const op = c.oraArithConstantOpCreate(ctx, loc, ty, attr);
    if (op.ptr == null) {
        @panic("Failed to create error placeholder constant");
    }
    c.oraOperationSetAttributeByName(op, h.strRef("ora.error_placeholder"), error_attr);
    h.appendOp(block, op);
    return h.getResult(op, 0);
}

fn refinementCacheKey(value: c.MlirValue) usize {
    if (c.mlirValueIsABlockArgument(value)) {
        const owner = c.mlirBlockArgumentGetOwner(value);
        const arg_no = c.mlirBlockArgumentGetArgNumber(value);
        const block_key = @intFromPtr(owner.ptr);
        const arg_key: usize = @intCast(arg_no);
        return block_key ^ (arg_key *% 0x9e3779b97f4a7c15);
    }
    return @intFromPtr(value.ptr);
}

pub fn unwrapRefinementValue(
    ctx: c.MlirContext,
    block: c.MlirBlock,
    locations: LocationTracker,
    refinement_base_cache: ?*std.AutoHashMap(usize, c.MlirValue),
    value: c.MlirValue,
    span: lib.ast.SourceSpan,
) c.MlirValue {
    const value_type = c.oraValueGetType(value);
    const base_type = c.oraRefinementTypeGetBaseType(value_type);
    if (base_type.ptr != null) {
        if (refinement_base_cache) |cache| {
            const key = refinementCacheKey(value);
            if (cache.get(key)) |cached| {
                return cached;
            }
        }
        const loc = locations.createLocation(span);
        const op = c.oraRefinementToBaseOpCreate(ctx, loc, value, block);
        if (op.ptr != null) {
            const base_value = h.getResult(op, 0);
            if (refinement_base_cache) |cache| {
                cache.put(refinementCacheKey(value), base_value) catch {};
            }
            return base_value;
        }
    }
    return value;
}

/// Create arithmetic operation using standard arith dialect ops.
/// op_name must be a standard arith op name (e.g., "arith.addi", "arith.divui").
pub fn createArithmeticOp(ctx: c.MlirContext, block: c.MlirBlock, type_mapper: *const TypeMapper, _: *OraDialect, locations: LocationTracker, refinement_base_cache: ?*std.AutoHashMap(usize, c.MlirValue), op_name: []const u8, lhs: c.MlirValue, rhs: c.MlirValue, span: lib.ast.SourceSpan) c.MlirValue {
    const lhs_unwrapped = unwrapRefinementValue(ctx, block, locations, refinement_base_cache, lhs, span);
    const rhs_unwrapped = unwrapRefinementValue(ctx, block, locations, refinement_base_cache, rhs, span);

    const lhs_type = c.oraValueGetType(lhs_unwrapped);
    const rhs_type = c.oraValueGetType(rhs_unwrapped);

    var rhs_converted = rhs_unwrapped;
    if (!c.oraTypeEqual(lhs_type, rhs_type)) {
        rhs_converted = type_mapper.createConversionOp(block, rhs_unwrapped, lhs_type, span);
    }

    const loc = locations.createLocation(span);

    const op = if (std.mem.eql(u8, op_name, "arith.addi"))
        c.oraArithAddIOpCreate(ctx, loc, lhs_unwrapped, rhs_converted)
    else if (std.mem.eql(u8, op_name, "arith.subi"))
        c.oraArithSubIOpCreate(ctx, loc, lhs_unwrapped, rhs_converted)
    else if (std.mem.eql(u8, op_name, "arith.muli"))
        c.oraArithMulIOpCreate(ctx, loc, lhs_unwrapped, rhs_converted)
    else if (std.mem.eql(u8, op_name, "arith.divui"))
        c.oraArithDivUIOpCreate(ctx, loc, lhs_unwrapped, rhs_converted)
    else if (std.mem.eql(u8, op_name, "arith.divsi"))
        c.oraArithDivSIOpCreate(ctx, loc, lhs_unwrapped, rhs_converted)
    else if (std.mem.eql(u8, op_name, "arith.remui"))
        c.oraArithRemUIOpCreate(ctx, loc, lhs_unwrapped, rhs_converted)
    else if (std.mem.eql(u8, op_name, "arith.remsi"))
        c.oraArithRemSIOpCreate(ctx, loc, lhs_unwrapped, rhs_converted)
    else if (std.mem.eql(u8, op_name, "arith.andi"))
        c.oraArithAndIOpCreate(ctx, loc, lhs_unwrapped, rhs_converted)
    else if (std.mem.eql(u8, op_name, "arith.ori"))
        c.oraArithOrIOpCreate(ctx, loc, lhs_unwrapped, rhs_converted)
    else if (std.mem.eql(u8, op_name, "arith.xori"))
        c.oraArithXorIOpCreate(ctx, loc, lhs_unwrapped, rhs_converted)
    else if (std.mem.eql(u8, op_name, "arith.shli"))
        c.oraArithShlIOpCreate(ctx, loc, lhs_unwrapped, rhs_converted)
    else if (std.mem.eql(u8, op_name, "arith.shrui"))
        if (@hasDecl(c, "oraArithShrUIOpCreate"))
            c.oraArithShrUIOpCreate(ctx, loc, lhs_unwrapped, rhs_converted)
        else
            // Compatibility fallback for environments using older Ora C API headers.
            c.oraArithShrSIOpCreate(ctx, loc, lhs_unwrapped, rhs_converted)
    else if (std.mem.eql(u8, op_name, "arith.shrsi"))
        c.oraArithShrSIOpCreate(ctx, loc, lhs_unwrapped, rhs_converted)
    else {
        log.err("Unsupported arithmetic op: {s}\n", .{op_name});
        @panic("Unsupported arithmetic op");
    };

    h.appendOp(block, op);
    return h.getResult(op, 0);
}

/// Create comparison operation using arith.cmpi for integer types
/// and ora.cmp for address types (which need special lowering).
pub fn createComparisonOp(ctx: c.MlirContext, block: c.MlirBlock, locations: LocationTracker, refinement_base_cache: ?*std.AutoHashMap(usize, c.MlirValue), predicate: []const u8, lhs: c.MlirValue, rhs: c.MlirValue, span: lib.ast.SourceSpan) c.MlirValue {
    const lhs_unwrapped = unwrapRefinementValue(ctx, block, locations, refinement_base_cache, lhs, span);
    const rhs_unwrapped = unwrapRefinementValue(ctx, block, locations, refinement_base_cache, rhs, span);

    const lhs_ty = c.oraValueGetType(lhs_unwrapped);
    const rhs_ty = c.oraValueGetType(rhs_unwrapped);
    const is_lhs_address = c.oraTypeIsAddressType(lhs_ty);
    const is_rhs_address = c.oraTypeIsAddressType(rhs_ty);
    const loc = locations.createLocation(span);

    // Address comparisons still use ora.cmp (needs address masking in lowering)
    if (is_lhs_address or is_rhs_address) {
        const bool_ty = h.boolType(ctx);
        if (is_lhs_address and is_rhs_address) {
            const op = c.oraCmpOpCreate(ctx, loc, h.strRef(predicate), lhs_unwrapped, rhs_unwrapped, bool_ty);
            h.appendOp(block, op);
            return h.getResult(op, 0);
        }

        const i160_ty = c.oraIntegerTypeCreate(ctx, 160);
        const addr_operand = if (is_lhs_address) lhs_unwrapped else rhs_unwrapped;
        const int_operand = if (is_lhs_address) rhs_unwrapped else lhs_unwrapped;
        const int_ty = c.oraValueGetType(int_operand);
        const int_i160 = if (c.oraTypeIsAInteger(int_ty) and !c.oraTypeEqual(int_ty, i160_ty)) blk: {
            const int_width = c.oraIntegerTypeGetWidth(int_ty);
            if (int_width < 160) {
                const ext_op = c.oraArithExtUIOpCreate(ctx, loc, int_operand, i160_ty);
                h.appendOp(block, ext_op);
                break :blk h.getResult(ext_op, 0);
            }
            const trunc_op = c.oraArithTruncIOpCreate(ctx, loc, int_operand, i160_ty);
            h.appendOp(block, trunc_op);
            break :blk h.getResult(trunc_op, 0);
        } else int_operand;
        const i160_to_addr = c.oraI160ToAddrOpCreate(ctx, loc, int_i160);
        h.appendOp(block, i160_to_addr);
        const int_addr = h.getResult(i160_to_addr, 0);

        const lhs_final = if (is_lhs_address) addr_operand else int_addr;
        const rhs_final = if (is_lhs_address) int_addr else addr_operand;
        const op = c.oraCmpOpCreate(ctx, loc, h.strRef(predicate), lhs_final, rhs_final, bool_ty);
        h.appendOp(block, op);
        return h.getResult(op, 0);
    }

    // Integer comparisons use standard arith.cmpi
    const pred_int = predicateStringToInt(predicate);
    const op = c.oraArithCmpIOpCreate(ctx, loc, pred_int, lhs_unwrapped, rhs_unwrapped);
    h.appendOp(block, op);
    return h.getResult(op, 0);
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
    log.warn("Unknown predicate '{s}', defaulting to 'eq' (0)\n", .{predicate});
    return 0;
}

/// Get common type for binary operations
pub fn getCommonType(ctx: c.MlirContext, lhs_ty: c.MlirType, rhs_ty: c.MlirType) c.MlirType {
    if (c.oraTypeEqual(lhs_ty, rhs_ty)) {
        return lhs_ty;
    }

    if (c.oraTypeIsAInteger(lhs_ty) and c.oraTypeIsAInteger(rhs_ty)) {
        const lhs_width = c.oraIntegerTypeGetWidth(lhs_ty);
        const rhs_width = c.oraIntegerTypeGetWidth(rhs_ty);

        if (lhs_width == 1 and rhs_width > 1) return rhs_ty;
        if (rhs_width == 1 and lhs_width > 1) return lhs_ty;
        if (lhs_width == rhs_width) return lhs_ty;
        return if (lhs_width >= rhs_width) lhs_ty else rhs_ty;
    }

    return c.oraIntegerTypeCreate(ctx, constants.DEFAULT_INTEGER_BITS);
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
