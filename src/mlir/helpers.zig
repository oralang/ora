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
const c = @import("mlir_c_api").c;
const lib = @import("ora_lib");

/// Create MLIR string reference from Zig slice
/// Replaces the verbose: c.oraStringRefCreate(str.ptr, str.len)
pub inline fn strRef(str: []const u8) c.MlirStringRef {
    return c.oraStringRefCreate(str.ptr, str.len);
}

/// Create MLIR string reference from string literal at comptime
pub inline fn strRefLit(comptime str: []const u8) c.MlirStringRef {
    return c.oraStringRefCreateFromCString(str.ptr);
}

/// Create MLIR identifier from string
pub inline fn identifier(ctx: c.MlirContext, name: []const u8) c.MlirIdentifier {
    return c.oraIdentifierGet(ctx, strRef(name));
}

/// Create MLIR string attribute from string
pub inline fn stringAttr(ctx: c.MlirContext, value: []const u8) c.MlirAttribute {
    return c.oraStringAttrCreate(ctx, strRef(value));
}

/// Create MLIR string attribute from string ref
pub inline fn stringAttrRef(ctx: c.MlirContext, value: c.MlirStringRef) c.MlirAttribute {
    return c.oraStringAttrCreate(ctx, value);
}

/// Create MLIR bool attribute
pub inline fn boolAttr(ctx: c.MlirContext, value: anytype) c.MlirAttribute {
    return c.oraBoolAttrCreate(ctx, value != 0);
}

/// Create MLIR integer attribute
pub inline fn intAttr(ctx: c.MlirContext, ty: c.MlirType, value: i64) c.MlirAttribute {
    return c.oraIntegerAttrCreateI64(ctx, ty, value);
}

/// Create MLIR integer attribute with implicit i64 cast
pub inline fn intAttrCast(ctx: c.MlirContext, ty: c.MlirType, value: anytype) c.MlirAttribute {
    return c.oraIntegerAttrCreateI64(ctx, ty, @as(i64, @intCast(value)));
}

/// Create MLIR type attribute
pub inline fn typeAttr(ctx: c.MlirContext, ty: c.MlirType) c.MlirAttribute {
    return c.oraTypeAttrCreate(ctx, ty);
}

/// Create MLIR array attribute
pub inline fn arrayAttr(ctx: c.MlirContext, attrs: []const c.MlirAttribute) c.MlirAttribute {
    return c.oraArrayAttrCreate(ctx, attrs.len, if (attrs.len == 0) null else attrs.ptr);
}

/// Create MLIR null attribute
pub inline fn nullAttr() c.MlirAttribute {
    return c.oraNullAttrCreate();
}

/// Create MLIR named attribute
pub inline fn namedAttr(ctx: c.MlirContext, name: []const u8, attr: c.MlirAttribute) c.MlirNamedAttribute {
    return c.oraNamedAttributeGet(identifier(ctx, name), attr);
}

/// Create operation using C++ API shim
pub inline fn createOp(
    ctx: c.MlirContext,
    loc: c.MlirLocation,
    name: []const u8,
    operands: []const c.MlirValue,
    result_types: []const c.MlirType,
    attrs: []const c.MlirNamedAttribute,
    num_regions: usize,
    add_empty_blocks: bool,
) c.MlirOperation {
    const op = c.oraOperationCreate(
        ctx,
        loc,
        strRef(name),
        if (operands.len == 0) null else operands.ptr,
        operands.len,
        if (result_types.len == 0) null else result_types.ptr,
        result_types.len,
        if (attrs.len == 0) null else attrs.ptr,
        attrs.len,
        num_regions,
        add_empty_blocks,
    );
    if (op.ptr == null) {
        @panic("Failed to create operation");
    }
    return op;
}

/// Create unknown location (fallback)
pub inline fn unknownLoc(ctx: c.MlirContext) c.MlirLocation {
    return c.oraLocationUnknownGet(ctx);
}

/// Create file location from SourceSpan
pub inline fn fileLoc(ctx: c.MlirContext, span: ?lib.ast.SourceSpan) c.MlirLocation {
    if (span) |s| {
        const filename_ref = strRefLit("source.ora");
        return c.oraLocationFileLineColGet(ctx, filename_ref, s.line, s.column);
    }
    return unknownLoc(ctx);
}

/// Create integer type
pub inline fn intType(ctx: c.MlirContext, bits: u32) c.MlirType {
    return c.oraIntegerTypeCreate(ctx, bits);
}

/// Create i256 type (default for Ora)
pub inline fn i256Type(ctx: c.MlirContext) c.MlirType {
    return intType(ctx, 256);
}

/// Create i1 type (boolean) - builtin MLIR type for ora.cmp and other operations
pub inline fn boolType(ctx: c.MlirContext) c.MlirType {
    return c.oraIntegerTypeCreate(ctx, 1);
}

/// Create index type
pub inline fn indexType(ctx: c.MlirContext) c.MlirType {
    return c.oraIndexTypeCreate(ctx);
}

/// Create none type
pub inline fn noneType(ctx: c.MlirContext) c.MlirType {
    return c.oraNoneTypeCreate(ctx);
}

/// Create ranked tensor type
pub inline fn rankedTensorType(ctx: c.MlirContext, rank: i64, shape: *const i64, element_type: c.MlirType, encoding: c.MlirAttribute) c.MlirType {
    return c.oraRankedTensorTypeCreate(ctx, rank, shape, element_type, encoding);
}

/// Create memref type
pub inline fn memRefType(ctx: c.MlirContext, element_type: c.MlirType, rank: i64, shape: ?*const i64, layout: c.MlirAttribute, memory_space: c.MlirAttribute) c.MlirType {
    return c.oraMemRefTypeCreate(ctx, element_type, rank, shape, layout, memory_space);
}

/// Shaped dynamic size sentinel
pub inline fn dynamicSize() i64 {
    return c.oraShapedTypeDynamicSize();
}

/// Create and append operation to block in one step
pub inline fn appendOp(block: c.MlirBlock, op: c.MlirOperation) void {
    c.oraBlockAppendOwnedOperation(block, op);
}

/// Insert operation before another operation in the same block
pub inline fn insertOpBefore(block: c.MlirBlock, op: c.MlirOperation, before: c.MlirOperation) void {
    c.oraBlockInsertOwnedOperationBefore(block, op, before);
}

/// Get result from operation
pub inline fn getResult(op: c.MlirOperation, index: usize) c.MlirValue {
    return c.oraOperationGetResult(op, index);
}

/// Common builder for simple operations (opcode, operands[], results[])
pub fn buildSimpleOp(
    ctx: c.MlirContext,
    block: c.MlirBlock,
    opcode: []const u8,
    operands: []const c.MlirValue,
    result_types: []const c.MlirType,
    loc: c.MlirLocation,
) c.MlirValue {
    const op = createOp(
        ctx,
        loc,
        opcode,
        operands,
        result_types,
        &[_]c.MlirNamedAttribute{},
        0,
        false,
    );
    appendOp(block, op);

    return if (result_types.len > 0) getResult(op, 0) else c.MlirValue{};
}
