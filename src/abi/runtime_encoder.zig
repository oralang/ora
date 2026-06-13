const std = @import("std");
const mlir = @import("mlir_c_api").c;
const mlir_helpers = @import("mlir_helpers");
const Type = @import("ora_types").SemanticType;
const abi_layout = @import("layout.zig");
const abi_layout_context = @import("layout_context.zig");
const hir_abi = @import("../hir/abi.zig");

const strRef = mlir_helpers.strRef;

pub fn createAbiEncodeWithSelectorOp(
    allocator: std.mem.Allocator,
    mlir_context: mlir.MlirContext,
    loc: mlir.MlirLocation,
    layout_context: abi_layout_context.LayoutContext,
    method_name: []const u8,
    param_types: []const Type,
    operands: []const mlir.MlirValue,
    result_type: mlir.MlirType,
) !mlir.MlirOperation {
    const attrs = try externalCallAbiAttrs(
        allocator,
        mlir_context,
        layout_context,
        method_name,
        param_types,
    );
    const layout_attr = try layoutAttrForParamTypes(allocator, mlir_context, layout_context, param_types);
    const op_attrs = [_]mlir.MlirNamedAttribute{
        mlir_helpers.namedAttr(mlir_context, "selector", attrs[1]),
        mlir_helpers.namedAttr(mlir_context, "arg_types", attrs[0]),
        mlir_helpers.namedAttr(mlir_context, "layout", layout_attr),
    };
    const result_types = [_]mlir.MlirType{result_type};
    return mlir_helpers.createOp(
        mlir_context,
        loc,
        "ora.abi_encode_with_selector",
        operands,
        &result_types,
        &op_attrs,
        0,
        false,
    );
}

pub fn createAbiEncodeOp(
    allocator: std.mem.Allocator,
    mlir_context: mlir.MlirContext,
    loc: mlir.MlirLocation,
    layout_context: abi_layout_context.LayoutContext,
    param_types: []const Type,
    operands: []const mlir.MlirValue,
    result_type: mlir.MlirType,
) !mlir.MlirOperation {
    const layout_attr = try layoutAttrForParamTypes(allocator, mlir_context, layout_context, param_types);
    const op_attrs = [_]mlir.MlirNamedAttribute{
        mlir_helpers.namedAttr(mlir_context, "layout", layout_attr),
    };
    const result_types = [_]mlir.MlirType{result_type};
    return mlir_helpers.createOp(
        mlir_context,
        loc,
        "ora.abi_encode",
        operands,
        &result_types,
        &op_attrs,
        0,
        false,
    );
}

fn layoutAttrForParamTypes(
    allocator: std.mem.Allocator,
    mlir_context: mlir.MlirContext,
    layout_context: abi_layout_context.LayoutContext,
    param_types: []const Type,
) !mlir.MlirAttribute {
    const tuple_ty: Type = .{ .tuple = param_types };
    var layout = try layout_context.layoutForType(tuple_ty);
    defer layout.deinit(allocator);
    const serialized = try abi_layout.serializeForMlirAttr(allocator, layout);
    defer allocator.free(serialized);
    return mlir.oraStringAttrCreate(mlir_context, strRef(serialized));
}

fn externalCallAbiAttrs(
    allocator: std.mem.Allocator,
    mlir_context: mlir.MlirContext,
    layout_context: abi_layout_context.LayoutContext,
    method_name: []const u8,
    param_types: []const Type,
) !struct { mlir.MlirAttribute, mlir.MlirAttribute } {
    var arg_type_attrs: std.ArrayList(mlir.MlirAttribute) = .{};
    defer arg_type_attrs.deinit(allocator);
    var signature_parts: std.ArrayList([]const u8) = .{};
    defer {
        for (signature_parts.items) |part| allocator.free(part);
        signature_parts.deinit(allocator);
    }

    for (param_types) |param_type| {
        const abi_type = try layout_context.canonicalAbiTypeForType(param_type);
        try arg_type_attrs.append(allocator, mlir.oraStringAttrCreate(mlir_context, strRef(abi_type)));
        try signature_parts.append(allocator, abi_type);
    }

    const arg_types_attr = mlir.oraArrayAttrCreate(
        mlir_context,
        @intCast(arg_type_attrs.items.len),
        if (arg_type_attrs.items.len == 0) null else arg_type_attrs.items.ptr,
    );
    const signature = try hir_abi.signatureForAbiTypes(allocator, method_name, signature_parts.items);
    defer allocator.free(signature);
    const selector_text = try hir_abi.keccakSelectorHex(allocator, signature);
    defer allocator.free(selector_text);
    const selector_value = try std.fmt.parseUnsigned(u32, selector_text[2..], 16);
    const selector_type = mlir.oraIntegerTypeCreate(mlir_context, 32);
    const selector_attr = mlir.oraIntegerAttrCreateI64FromType(selector_type, selector_value);
    return .{ arg_types_attr, selector_attr };
}
