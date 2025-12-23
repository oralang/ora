// ============================================================================
// MLIR Helper Utilities
// ============================================================================
//
// Common helper functions to reduce boilerplate and duplication across
// the MLIR lowering code.
//
// KEY HELPERS:
//   • String reference creation
//   • Operation state builders
//   • Location creation shortcuts
//   • Type creation shortcuts
//
// ============================================================================

const std = @import("std");
const c = @import("c.zig").c;
const lib = @import("ora_lib");

/// Create MLIR string reference from Zig slice
/// Replaces the verbose: c.mlirStringRefCreate(str.ptr, str.len)
pub inline fn strRef(str: []const u8) c.MlirStringRef {
    return c.mlirStringRefCreate(str.ptr, str.len);
}

/// Create MLIR string reference from string literal at comptime
pub inline fn strRefLit(comptime str: []const u8) c.MlirStringRef {
    return c.mlirStringRefCreateFromCString(str.ptr);
}

/// Create MLIR identifier from string
pub inline fn identifier(ctx: c.MlirContext, name: []const u8) c.MlirIdentifier {
    return c.mlirIdentifierGet(ctx, strRef(name));
}

/// Create MLIR string attribute from string
pub inline fn stringAttr(ctx: c.MlirContext, value: []const u8) c.MlirAttribute {
    return c.mlirStringAttrGet(ctx, strRef(value));
}

/// Create MLIR integer attribute
pub inline fn intAttr(_: c.MlirContext, ty: c.MlirType, value: i64) c.MlirAttribute {
    return c.mlirIntegerAttrGet(ty, value);
}

/// Create MLIR named attribute
pub inline fn namedAttr(ctx: c.MlirContext, name: []const u8, attr: c.MlirAttribute) c.MlirNamedAttribute {
    return c.mlirNamedAttributeGet(identifier(ctx, name), attr);
}

/// Create operation state with name and location
pub inline fn opState(name: []const u8, loc: c.MlirLocation) c.MlirOperationState {
    return c.mlirOperationStateGet(strRef(name), loc);
}

/// Create unknown location (fallback)
pub inline fn unknownLoc(ctx: c.MlirContext) c.MlirLocation {
    return c.mlirLocationUnknownGet(ctx);
}

/// Create file location from SourceSpan
pub inline fn fileLoc(ctx: c.MlirContext, span: ?lib.ast.SourceSpan) c.MlirLocation {
    if (span) |s| {
        const filename_ref = strRefLit("source.ora");
        return c.mlirLocationFileLineColGet(ctx, filename_ref, s.line, s.column);
    }
    return unknownLoc(ctx);
}

/// Create integer type
pub inline fn intType(ctx: c.MlirContext, bits: u32) c.MlirType {
    return c.mlirIntegerTypeGet(ctx, bits);
}

/// Create i256 type (default for Ora)
pub inline fn i256Type(ctx: c.MlirContext) c.MlirType {
    return intType(ctx, 256);
}

/// Create i1 type (boolean) - builtin MLIR type for ora.cmp and other operations
pub inline fn boolType(ctx: c.MlirContext) c.MlirType {
    // Use MLIR C API directly to create builtin i1 type (not Ora type)
    // ora.cmp requires builtin i1, not !ora.int<1, false>
    // mlirIntegerTypeGet is from mlir-c/BuiltinTypes.h
    return c.mlirIntegerTypeGet(ctx, 1);
}

/// Create and append operation to block in one step
pub inline fn appendOp(block: c.MlirBlock, op: c.MlirOperation) void {
    c.mlirBlockAppendOwnedOperation(block, op);
}

/// Get first block from region
pub inline fn firstBlock(region: c.MlirRegion) c.MlirBlock {
    return c.mlirRegionGetFirstBlock(region);
}

/// Get region from operation
pub inline fn getRegion(op: c.MlirOperation, index: usize) c.MlirRegion {
    return c.mlirOperationGetRegion(op, index);
}

/// Get result from operation
pub inline fn getResult(op: c.MlirOperation, index: usize) c.MlirValue {
    return c.mlirOperationGetResult(op, index);
}

/// Common builder for simple operations (opcode, operands[], results[])
pub fn buildSimpleOp(
    _: c.MlirContext,
    block: c.MlirBlock,
    opcode: []const u8,
    operands: []const c.MlirValue,
    result_types: []const c.MlirType,
    loc: c.MlirLocation,
) c.MlirValue {
    var state = opState(opcode, loc);

    if (operands.len > 0) {
        c.mlirOperationStateAddOperands(&state, operands.len, @ptrCast(operands.ptr));
    }

    if (result_types.len > 0) {
        c.mlirOperationStateAddResults(&state, result_types.len, @ptrCast(result_types.ptr));
    }

    const op = c.mlirOperationCreate(&state);
    appendOp(block, op);

    return if (result_types.len > 0) getResult(op, 0) else c.MlirValue{};
}
