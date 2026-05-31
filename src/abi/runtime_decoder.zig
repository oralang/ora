const std = @import("std");
const mlir = @import("mlir_c_api").c;
const mlir_helpers = @import("mlir_helpers");
const sema = @import("../sema/mod.zig");
const abi_layout = @import("layout.zig");
const abi_layout_context = @import("layout_context.zig");
const hir_abi = @import("../hir/abi.zig");

const strRef = mlir_helpers.strRef;

pub fn createAbiDecodeOp(
    allocator: std.mem.Allocator,
    mlir_context: mlir.MlirContext,
    loc: mlir.MlirLocation,
    layout_context: abi_layout_context.LayoutContext,
    return_type: sema.Type,
    returndata: mlir.MlirValue,
    result_type: mlir.MlirType,
) !mlir.MlirOperation {
    return createAbiDecodeOpWithMode(
        allocator,
        mlir_context,
        loc,
        layout_context,
        return_type,
        returndata,
        result_type,
        "returndata",
        "error_union",
    );
}

pub fn createExternalReturnAbiDecodeOp(
    allocator: std.mem.Allocator,
    mlir_context: mlir.MlirContext,
    loc: mlir.MlirLocation,
    layout_context: abi_layout_context.LayoutContext,
    return_type: sema.Type,
    returndata: mlir.MlirValue,
    result_type: mlir.MlirType,
    failure_error: []const u8,
) !mlir.MlirOperation {
    return createAbiDecodeOpWithModeAndFailureError(
        allocator,
        mlir_context,
        loc,
        layout_context,
        return_type,
        returndata,
        result_type,
        "returndata",
        "error_union",
        failure_error,
    );
}

pub fn createMemoryResultAbiDecodeOp(
    allocator: std.mem.Allocator,
    mlir_context: mlir.MlirContext,
    loc: mlir.MlirLocation,
    layout_context: abi_layout_context.LayoutContext,
    target_type: sema.Type,
    bytes: mlir.MlirValue,
    result_type: mlir.MlirType,
) !mlir.MlirOperation {
    return createAbiDecodeOpWithMode(
        allocator,
        mlir_context,
        loc,
        layout_context,
        target_type,
        bytes,
        result_type,
        "memory",
        "result",
    );
}

fn createAbiDecodeOpWithMode(
    allocator: std.mem.Allocator,
    mlir_context: mlir.MlirContext,
    loc: mlir.MlirLocation,
    layout_context: abi_layout_context.LayoutContext,
    return_type: sema.Type,
    bytes: mlir.MlirValue,
    result_type: mlir.MlirType,
    source: []const u8,
    failure_mode: []const u8,
) !mlir.MlirOperation {
    return createAbiDecodeOpWithModeAndFailureError(
        allocator,
        mlir_context,
        loc,
        layout_context,
        return_type,
        bytes,
        result_type,
        source,
        failure_mode,
        null,
    );
}

fn createAbiDecodeOpWithModeAndFailureError(
    allocator: std.mem.Allocator,
    mlir_context: mlir.MlirContext,
    loc: mlir.MlirLocation,
    layout_context: abi_layout_context.LayoutContext,
    return_type: sema.Type,
    bytes: mlir.MlirValue,
    result_type: mlir.MlirType,
    source: []const u8,
    failure_mode: []const u8,
    failure_error: ?[]const u8,
) !mlir.MlirOperation {
    const attrs = try attrsForReturnType(allocator, mlir_context, layout_context, return_type);
    const enum_variant_count = enumVariantCount(layout_context, return_type);
    var op_attrs: std.ArrayList(mlir.MlirNamedAttribute) = .{};
    defer op_attrs.deinit(allocator);
    try op_attrs.append(allocator, mlir_helpers.namedAttr(mlir_context, "return_types", attrs.return_types));
    try op_attrs.append(allocator, mlir_helpers.namedAttr(mlir_context, "layout", attrs.layout));
    try op_attrs.append(allocator, mlir_helpers.namedAttr(
        mlir_context,
        "source",
        mlir.oraStringAttrCreate(mlir_context, strRef(source)),
    ));
    try op_attrs.append(allocator, mlir_helpers.namedAttr(
        mlir_context,
        "failure_mode",
        mlir.oraStringAttrCreate(mlir_context, strRef(failure_mode)),
    ));
    if (enum_variant_count) |count| {
        try op_attrs.append(allocator, mlir_helpers.namedAttr(
            mlir_context,
            "enum_variant_count",
            mlir.oraIntegerAttrCreateI64(mlir_context, mlir.oraIntegerTypeCreate(mlir_context, 64), @intCast(count)),
        ));
    }
    if (failure_error) |name| {
        try op_attrs.append(allocator, mlir_helpers.namedAttr(
            mlir_context,
            "returndata_failure_error",
            mlir.oraStringAttrCreate(mlir_context, strRef(name)),
        ));
    }
    const operands = [_]mlir.MlirValue{bytes};
    const result_types = [_]mlir.MlirType{result_type};
    return mlir_helpers.createOp(
        mlir_context,
        loc,
        "ora.abi_decode",
        &operands,
        &result_types,
        op_attrs.items,
        0,
        false,
    );
}

fn attrsForReturnType(
    allocator: std.mem.Allocator,
    mlir_context: mlir.MlirContext,
    layout_context: abi_layout_context.LayoutContext,
    return_type: sema.Type,
) !struct { return_types: mlir.MlirAttribute, layout: mlir.MlirAttribute } {
    const return_abi_type = try decodeReturnAbiType(allocator, layout_context, return_type);
    defer allocator.free(return_abi_type);
    var return_type_attrs = [_]mlir.MlirAttribute{
        mlir.oraStringAttrCreate(mlir_context, strRef(return_abi_type)),
    };
    const return_types_attr = mlir.oraArrayAttrCreate(mlir_context, return_type_attrs.len, &return_type_attrs);

    var single_return_storage: [1]sema.Type = undefined;
    const layout_type = layoutTypeForReturn(return_type, &single_return_storage);
    var layout = try layout_context.layoutForType(layout_type);
    defer layout.deinit(allocator);
    const serialized = try abi_layout.serializeForMlirAttr(allocator, layout);
    defer allocator.free(serialized);
    return .{
        .return_types = return_types_attr,
        .layout = mlir.oraStringAttrCreate(mlir_context, strRef(serialized)),
    };
}

fn enumVariantCount(layout_context: abi_layout_context.LayoutContext, ty: sema.Type) ?usize {
    const unwrapped = unwrapRefinement(ty);
    const name = switch (unwrapped) {
        .enum_ => |named| named.name,
        .named => |named| named.name,
        else => return null,
    };
    if (layout_context.typecheck.instantiatedEnumByName(name)) |instantiated| {
        return instantiated.variants.len;
    }
    const item_id = layout_context.item_index.lookup(name) orelse return null;
    return switch (layout_context.file.item(item_id).*) {
        .Enum => |enum_item| enum_item.variants.len,
        else => null,
    };
}

fn unwrapRefinement(ty: sema.Type) sema.Type {
    return switch (ty) {
        .refinement => |refinement| unwrapRefinement(refinement.base_type.*),
        else => ty,
    };
}

// Returns the tuple-shaped sema.Type to feed into layoutForType. A tuple
// return uses its argument-list shape directly; any other return is wrapped
// in a 1-tuple so the layout matches Solidity's single-arg convention.
// `single_storage` backs the wrap case; the returned slice must outlive
// the caller's use of the result (layoutForType copies before returning).
fn layoutTypeForReturn(return_type: sema.Type, single_storage: *[1]sema.Type) sema.Type {
    return switch (unwrapRefinement(return_type)) {
        .tuple, .void => return_type,
        else => blk: {
            single_storage[0] = return_type;
            break :blk .{ .tuple = single_storage };
        },
    };
}

fn decodeReturnAbiType(
    allocator: std.mem.Allocator,
    layout_context: abi_layout_context.LayoutContext,
    return_type: sema.Type,
) ![]const u8 {
    const unwrapped = unwrapRefinement(return_type);
    if (try immediateDecodeReturnAbiType(allocator, unwrapped)) |abi_type| {
        return abi_type;
    }
    return layout_context.canonicalAbiTypeForType(return_type) catch hir_abi.externReturnAbiType(allocator, return_type);
}

fn immediateDecodeReturnAbiType(allocator: std.mem.Allocator, unwrapped: sema.Type) !?[]const u8 {
    return switch (unwrapped) {
        .void => try allocator.dupe(u8, "void"),
        .tuple => try allocator.dupe(u8, "tuple"),
        .anonymous_struct => try allocator.dupe(u8, "struct"),
        // Named structs and contract references decode through tuple-shaped
        // return metadata; the layout attribute carries the concrete ABI shape.
        .struct_, .contract => try allocator.dupe(u8, "tuple"),
        else => null,
    };
}

test "runtime decoder recursively unwraps refined tuple returns" {
    const u256_ty: sema.Type = .{ .integer = .{ .bits = 256, .signed = false, .spelling = "u256" } };
    const tuple_elems = [_]sema.Type{ u256_ty, .bool };
    const tuple_ty: sema.Type = .{ .tuple = &tuple_elems };
    const inner_refined: sema.Type = .{ .refinement = .{
        .name = "InnerTupleRefinement",
        .base_type = &tuple_ty,
    } };
    const outer_refined: sema.Type = .{ .refinement = .{
        .name = "OuterTupleRefinement",
        .base_type = &inner_refined,
    } };

    try std.testing.expectEqual(sema.TypeKind.tuple, unwrapRefinement(outer_refined).kind());

    const return_abi_type = (try immediateDecodeReturnAbiType(
        std.testing.allocator,
        unwrapRefinement(outer_refined),
    )) orelse return error.TestUnexpectedResult;
    defer std.testing.allocator.free(return_abi_type);
    try std.testing.expectEqualStrings("tuple", return_abi_type);
}

test "runtime decoder doubly-refined tuple serializes flat layout (no double wrap)" {
    const u256_ty: sema.Type = .{ .integer = .{ .bits = 256, .signed = false, .spelling = "u256" } };
    const tuple_elems = [_]sema.Type{ u256_ty, .bool };
    const tuple_ty: sema.Type = .{ .tuple = &tuple_elems };
    const inner_refined: sema.Type = .{ .refinement = .{
        .name = "InnerTupleRefinement",
        .base_type = &tuple_ty,
    } };
    const outer_refined: sema.Type = .{ .refinement = .{
        .name = "OuterTupleRefinement",
        .base_type = &inner_refined,
    } };

    // Drive the same dispatch helper production uses, then run the layout
    // and serialization chain directly. Payload is primitive-only so we
    // can bypass LayoutContext (fromType recurses through refinements
    // without needing typecheck/file context).
    var single_storage: [1]sema.Type = undefined;
    const layout_type = layoutTypeForReturn(outer_refined, &single_storage);

    var layout = try abi_layout.fromType(std.testing.allocator, layout_type);
    defer layout.deinit(std.testing.allocator);
    const serialized = try abi_layout.serializeForMlirAttr(std.testing.allocator, layout);
    defer std.testing.allocator.free(serialized);

    try std.testing.expectEqualStrings("tuple(static(uint256),static(bool))", serialized);
    try std.testing.expect(!std.mem.containsAtLeast(u8, serialized, 1, "tuple(tuple("));
}
