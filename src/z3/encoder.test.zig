// ============================================================================
// Z3 Encoder Tests
// ============================================================================
//
// unit tests for MLIR-to-Z3 encoding behavior.
//
// ============================================================================

const std = @import("std");
const testing = std.testing;
const z3 = @import("c.zig");
const mlir = @import("mlir_c_api").c;
const Context = @import("context.zig").Context;
const Encoder = @import("encoder.zig").Encoder;
const Solver = @import("solver.zig").Solver;

fn stringRef(comptime s: []const u8) mlir.MlirStringRef {
    return mlir.oraStringRefCreate(s.ptr, s.len);
}

fn namedAttr(ctx: mlir.MlirContext, comptime name: []const u8, attr: mlir.MlirAttribute) mlir.MlirNamedAttribute {
    const id = mlir.oraIdentifierGet(ctx, mlir.oraStringRefCreate(name.ptr, name.len));
    return mlir.oraNamedAttributeGet(id, attr);
}

fn loadAllDialects(ctx: mlir.MlirContext) void {
    const registry = mlir.oraDialectRegistryCreate();
    defer mlir.oraDialectRegistryDestroy(registry);
    mlir.oraRegisterAllDialects(registry);
    mlir.oraContextAppendDialectRegistry(ctx, registry);
    mlir.oraContextLoadAllAvailableDialects(ctx);
}

test "encodeMLIRType maps bool and i32" {
    var z3_ctx = try Context.init(testing.allocator);
    defer z3_ctx.deinit();

    var encoder = Encoder.init(&z3_ctx, testing.allocator);
    defer encoder.deinit();

    const mlir_ctx = mlir.oraContextCreate();
    defer mlir.oraContextDestroy(mlir_ctx);
    loadAllDialects(mlir_ctx);
    _ = mlir.oraDialectRegister(mlir_ctx);

    const ty_i1 = mlir.oraIntegerTypeCreate(mlir_ctx, 1);
    const bool_sort = try encoder.encodeMLIRType(ty_i1);
    try testing.expectEqual(@as(u32, z3.Z3_BOOL_SORT), @as(u32, @intCast(z3.Z3_get_sort_kind(z3_ctx.ctx, bool_sort))));

    const ty_i32 = mlir.oraIntegerTypeCreate(mlir_ctx, 32);
    const i32_sort = try encoder.encodeMLIRType(ty_i32);
    try testing.expectEqual(@as(u32, z3.Z3_BV_SORT), @as(u32, @intCast(z3.Z3_get_sort_kind(z3_ctx.ctx, i32_sort))));
    try testing.expectEqual(@as(u32, 32), @as(u32, @intCast(z3.Z3_get_bv_sort_size(z3_ctx.ctx, i32_sort))));

    const ty_bytes = mlir.oraBytesTypeGet(mlir_ctx);
    const bytes_sort = try encoder.encodeMLIRType(ty_bytes);
    try testing.expect(z3.Z3_is_string_sort(z3_ctx.ctx, bytes_sort));
}

test "ora.bytes.constant encodes canonical hex string" {
    var z3_ctx = try Context.init(testing.allocator);
    defer z3_ctx.deinit();

    var encoder = Encoder.init(&z3_ctx, testing.allocator);
    defer encoder.deinit();

    const mlir_ctx = mlir.oraContextCreate();
    defer mlir.oraContextDestroy(mlir_ctx);
    loadAllDialects(mlir_ctx);
    _ = mlir.oraDialectRegister(mlir_ctx);

    const loc = mlir.oraLocationUnknownGet(mlir_ctx);
    const bytes_ty = mlir.oraBytesTypeGet(mlir_ctx);

    const with_prefix = mlir.oraBytesConstantOpCreate(mlir_ctx, loc, stringRef("0xDEADbeef"), bytes_ty);
    const without_prefix = mlir.oraBytesConstantOpCreate(mlir_ctx, loc, stringRef("deadBEEF"), bytes_ty);

    const lhs = try encoder.encodeOperation(with_prefix);
    const rhs = try encoder.encodeOperation(without_prefix);

    var solver = try Solver.init(&z3_ctx, testing.allocator);
    defer solver.deinit();
    solver.assert(z3.Z3_mk_not(z3_ctx.ctx, z3.Z3_mk_eq(z3_ctx.ctx, lhs, rhs)));
    try testing.expectEqual(@as(z3.Z3_lbool, z3.Z3_L_FALSE), solver.check());
}

test "ora.bytes.constant rejects invalid hex" {
    var z3_ctx = try Context.init(testing.allocator);
    defer z3_ctx.deinit();

    var encoder = Encoder.init(&z3_ctx, testing.allocator);
    defer encoder.deinit();

    const mlir_ctx = mlir.oraContextCreate();
    defer mlir.oraContextDestroy(mlir_ctx);
    loadAllDialects(mlir_ctx);
    _ = mlir.oraDialectRegister(mlir_ctx);

    const loc = mlir.oraLocationUnknownGet(mlir_ctx);
    const bytes_ty = mlir.oraBytesTypeGet(mlir_ctx);
    const invalid = mlir.oraBytesConstantOpCreate(mlir_ctx, loc, stringRef("0xabc"), bytes_ty);

    try testing.expectError(error.UnsupportedOperation, encoder.encodeOperation(invalid));
}

test "encodeIntegerConstant encodes large numerals" {
    var z3_ctx = try Context.init(testing.allocator);
    defer z3_ctx.deinit();

    var encoder = Encoder.init(&z3_ctx, testing.allocator);
    defer encoder.deinit();

    const value: u256 = (@as(u256, 1) << 128) + 123;
    const ast = try encoder.encodeIntegerConstant(value, 256);
    const numeral = z3.Z3_get_numeral_string(z3_ctx.ctx, ast);
    const expected = try std.fmt.allocPrint(testing.allocator, "{d}", .{value});
    defer testing.allocator.free(expected);

    try testing.expect(std.mem.eql(u8, std.mem.span(numeral), expected));
}

test "encodeConstantOp uses bool sort for i1" {
    var z3_ctx = try Context.init(testing.allocator);
    defer z3_ctx.deinit();

    var encoder = Encoder.init(&z3_ctx, testing.allocator);
    defer encoder.deinit();

    const ast = try encoder.encodeConstantOp(1, 1);
    const sort_kind = z3.Z3_get_sort_kind(z3_ctx.ctx, z3.Z3_get_sort(z3_ctx.ctx, ast));
    try testing.expectEqual(@as(u32, z3.Z3_BOOL_SORT), @as(u32, @intCast(sort_kind)));
}

test "encodeNot coerces bitvector condition to bool" {
    var z3_ctx = try Context.init(testing.allocator);
    defer z3_ctx.deinit();

    var encoder = Encoder.init(&z3_ctx, testing.allocator);
    defer encoder.deinit();

    const bv256 = encoder.mkBitVectorSort(256);
    const cond = try encoder.mkVariable("cond_bv", bv256);
    const not_cond = encoder.encodeNot(cond);
    const not_sort = z3.Z3_get_sort(z3_ctx.ctx, not_cond);
    const not_kind = z3.Z3_get_sort_kind(z3_ctx.ctx, not_sort);
    try testing.expectEqual(@as(u32, z3.Z3_BOOL_SORT), @as(u32, @intCast(not_kind)));
}

test "encodeSLoad returns global value with correct sort" {
    var z3_ctx = try Context.init(testing.allocator);
    defer z3_ctx.deinit();

    var encoder = Encoder.init(&z3_ctx, testing.allocator);
    defer encoder.deinit();

    const mlir_ctx = mlir.oraContextCreate();
    defer mlir.oraContextDestroy(mlir_ctx);

    loadAllDialects(mlir_ctx);
    _ = mlir.oraDialectRegister(mlir_ctx);

    const loc = mlir.oraLocationUnknownGet(mlir_ctx);
    const ty_i256 = mlir.oraIntegerTypeCreate(mlir_ctx, 256);
    const name_ref = mlir.oraStringRefCreate("counter".ptr, "counter".len);
    const op = mlir.oraSLoadOpCreate(mlir_ctx, loc, name_ref, ty_i256);
    const ast = try encoder.encodeOperation(op);
    const sort_kind = z3.Z3_get_sort_kind(z3_ctx.ctx, z3.Z3_get_sort(z3_ctx.ctx, ast));
    try testing.expectEqual(@as(u32, z3.Z3_BV_SORT), @as(u32, @intCast(sort_kind)));
}

test "encodeSLoad for map returns array sort" {
    var z3_ctx = try Context.init(testing.allocator);
    defer z3_ctx.deinit();

    var encoder = Encoder.init(&z3_ctx, testing.allocator);
    defer encoder.deinit();

    const mlir_ctx = mlir.oraContextCreate();
    defer mlir.oraContextDestroy(mlir_ctx);

    loadAllDialects(mlir_ctx);
    _ = mlir.oraDialectRegister(mlir_ctx);

    const loc = mlir.oraLocationUnknownGet(mlir_ctx);
    const addr_ty = mlir.oraAddressTypeGet(mlir_ctx);
    const value_ty = mlir.oraIntegerTypeCreate(mlir_ctx, 256);
    const map_ty = mlir.oraMapTypeGet(mlir_ctx, addr_ty, value_ty);
    const name_ref = mlir.oraStringRefCreate("balances".ptr, "balances".len);
    const op = mlir.oraSLoadOpCreate(mlir_ctx, loc, name_ref, map_ty);
    const ast = try encoder.encodeOperation(op);
    const sort_kind = z3.Z3_get_sort_kind(z3_ctx.ctx, z3.Z3_get_sort(z3_ctx.ctx, ast));
    try testing.expectEqual(@as(u32, z3.Z3_ARRAY_SORT), @as(u32, @intCast(sort_kind)));
}

test "struct_field_update preserves untouched fields exactly" {
    var z3_ctx = try Context.init(testing.allocator);
    defer z3_ctx.deinit();

    var encoder = Encoder.init(&z3_ctx, testing.allocator);
    defer encoder.deinit();

    const mlir_ctx = mlir.oraContextCreate();
    defer mlir.oraContextDestroy(mlir_ctx);
    loadAllDialects(mlir_ctx);
    _ = mlir.oraDialectRegister(mlir_ctx);

    const loc = mlir.oraLocationUnknownGet(mlir_ctx);
    const i256_ty = mlir.oraIntegerTypeCreate(mlir_ctx, 256);
    const struct_ty = mlir.oraStructTypeGet(mlir_ctx, stringRef("Pair__u256"));

    const struct_decl = mlir.oraStructDeclOpCreate(mlir_ctx, loc, stringRef("Pair__u256"));
    const field_name_attrs = [_]mlir.MlirAttribute{
        mlir.oraStringAttrCreate(mlir_ctx, stringRef("left")),
        mlir.oraStringAttrCreate(mlir_ctx, stringRef("right")),
    };
    const field_type_attrs = [_]mlir.MlirAttribute{
        mlir.oraTypeAttrCreateFromType(i256_ty),
        mlir.oraTypeAttrCreateFromType(i256_ty),
    };
    mlir.oraOperationSetAttributeByName(struct_decl, stringRef("ora.field_names"), mlir.oraArrayAttrCreate(mlir_ctx, field_name_attrs.len, &field_name_attrs));
    mlir.oraOperationSetAttributeByName(struct_decl, stringRef("ora.field_types"), mlir.oraArrayAttrCreate(mlir_ctx, field_type_attrs.len, &field_type_attrs));
    try encoder.registerStructDeclOperation(struct_decl);

    const one = mlir.oraOperationGetResult(mlir.oraArithConstantOpCreate(
        mlir_ctx,
        loc,
        i256_ty,
        mlir.oraIntegerAttrCreateI64FromType(i256_ty, 1),
    ), 0);
    const two = mlir.oraOperationGetResult(mlir.oraArithConstantOpCreate(
        mlir_ctx,
        loc,
        i256_ty,
        mlir.oraIntegerAttrCreateI64FromType(i256_ty, 2),
    ), 0);
    const seven = mlir.oraOperationGetResult(mlir.oraArithConstantOpCreate(
        mlir_ctx,
        loc,
        i256_ty,
        mlir.oraIntegerAttrCreateI64FromType(i256_ty, 7),
    ), 0);

    const fields = [_]mlir.MlirValue{ one, two };
    const init_op = mlir.oraStructInstantiateOpCreate(mlir_ctx, loc, stringRef("Pair__u256"), &fields, fields.len, struct_ty);
    const pair = mlir.oraOperationGetResult(init_op, 0);
    const update_op = mlir.oraStructFieldUpdateOpCreate(mlir_ctx, loc, pair, stringRef("left"), seven);
    const updated_pair = mlir.oraOperationGetResult(update_op, 0);

    const extract_left = mlir.oraStructFieldExtractOpCreate(mlir_ctx, loc, updated_pair, stringRef("left"), i256_ty);
    const extract_right = mlir.oraStructFieldExtractOpCreate(mlir_ctx, loc, updated_pair, stringRef("right"), i256_ty);
    const left_ast = try encoder.encodeOperation(extract_left);
    const right_ast = try encoder.encodeOperation(extract_right);

    if (encoder.isDegraded()) {
        std.debug.print("struct update degradation: {s}\n", .{encoder.degradationReason().?});
    }
    try testing.expect(!encoder.isDegraded());

    const constraints = try encoder.takeConstraints(testing.allocator);
    defer if (constraints.len > 0) testing.allocator.free(constraints);

    var solver = try Solver.init(&z3_ctx, testing.allocator);
    defer solver.deinit();
    for (constraints) |cst| solver.assert(cst);

    solver.assert(z3.Z3_mk_not(z3_ctx.ctx, z3.Z3_mk_eq(z3_ctx.ctx, left_ast, try encoder.encodeValue(seven))));
    try testing.expectEqual(@as(z3.Z3_lbool, z3.Z3_L_FALSE), solver.check());

    solver.reset();
    for (constraints) |cst| solver.assert(cst);
    solver.assert(z3.Z3_mk_not(z3_ctx.ctx, z3.Z3_mk_eq(z3_ctx.ctx, right_ast, try encoder.encodeValue(two))));
    try testing.expectEqual(@as(z3.Z3_lbool, z3.Z3_L_FALSE), solver.check());
}

test "known pure callee returning updated struct preserves untouched fields exactly" {
    var z3_ctx = try Context.init(testing.allocator);
    defer z3_ctx.deinit();

    var encoder = Encoder.init(&z3_ctx, testing.allocator);
    defer encoder.deinit();

    const mlir_ctx = mlir.oraContextCreate();
    defer mlir.oraContextDestroy(mlir_ctx);
    loadAllDialects(mlir_ctx);
    _ = mlir.oraDialectRegister(mlir_ctx);

    const loc = mlir.oraLocationUnknownGet(mlir_ctx);
    const i256_ty = mlir.oraIntegerTypeCreate(mlir_ctx, 256);
    const struct_ty = mlir.oraStructTypeGet(mlir_ctx, stringRef("Pair__u256"));

    const struct_decl = mlir.oraStructDeclOpCreate(mlir_ctx, loc, stringRef("Pair__u256"));
    const field_name_attrs = [_]mlir.MlirAttribute{
        mlir.oraStringAttrCreate(mlir_ctx, stringRef("left")),
        mlir.oraStringAttrCreate(mlir_ctx, stringRef("right")),
    };
    const field_type_attrs = [_]mlir.MlirAttribute{
        mlir.oraTypeAttrCreateFromType(i256_ty),
        mlir.oraTypeAttrCreateFromType(i256_ty),
    };
    mlir.oraOperationSetAttributeByName(struct_decl, stringRef("ora.field_names"), mlir.oraArrayAttrCreate(mlir_ctx, field_name_attrs.len, &field_name_attrs));
    mlir.oraOperationSetAttributeByName(struct_decl, stringRef("ora.field_types"), mlir.oraArrayAttrCreate(mlir_ctx, field_type_attrs.len, &field_type_attrs));
    try encoder.registerStructDeclOperation(struct_decl);

    const helper_attrs = [_]mlir.MlirNamedAttribute{
        namedAttr(mlir_ctx, "sym_name", mlir.oraStringAttrCreate(mlir_ctx, stringRef("makePair"))),
    };
    const helper = mlir.oraFuncFuncOpCreate(mlir_ctx, loc, &helper_attrs, helper_attrs.len, &[_]mlir.MlirType{}, &[_]mlir.MlirLocation{}, 0);
    const helper_body = mlir.oraFuncOpGetBodyBlock(helper);

    const one = mlir.oraOperationGetResult(mlir.oraArithConstantOpCreate(
        mlir_ctx,
        loc,
        i256_ty,
        mlir.oraIntegerAttrCreateI64FromType(i256_ty, 1),
    ), 0);
    const two = mlir.oraOperationGetResult(mlir.oraArithConstantOpCreate(
        mlir_ctx,
        loc,
        i256_ty,
        mlir.oraIntegerAttrCreateI64FromType(i256_ty, 2),
    ), 0);
    const seven = mlir.oraOperationGetResult(mlir.oraArithConstantOpCreate(
        mlir_ctx,
        loc,
        i256_ty,
        mlir.oraIntegerAttrCreateI64FromType(i256_ty, 7),
    ), 0);
    const fields = [_]mlir.MlirValue{ one, two };
    const init_op = mlir.oraStructInstantiateOpCreate(mlir_ctx, loc, stringRef("Pair__u256"), &fields, fields.len, struct_ty);
    const pair = mlir.oraOperationGetResult(init_op, 0);
    const update_op = mlir.oraStructFieldUpdateOpCreate(mlir_ctx, loc, pair, stringRef("left"), seven);
    const updated_pair = mlir.oraOperationGetResult(update_op, 0);
    const ret_vals = [_]mlir.MlirValue{updated_pair};
    const ret_op = mlir.oraReturnOpCreate(mlir_ctx, loc, &ret_vals, ret_vals.len);
    mlir.oraBlockAppendOwnedOperation(helper_body, init_op);
    mlir.oraBlockAppendOwnedOperation(helper_body, update_op);
    mlir.oraBlockAppendOwnedOperation(helper_body, ret_op);
    try encoder.registerFunctionOperation(helper);

    const call = mlir.oraFuncCallOpCreate(mlir_ctx, loc, stringRef("makePair"), &[_]mlir.MlirValue{}, 0, &[_]mlir.MlirType{struct_ty}, 1);
    const call_result = mlir.oraOperationGetResult(call, 0);
    const extract_right = mlir.oraStructFieldExtractOpCreate(mlir_ctx, loc, call_result, stringRef("right"), i256_ty);
    const right_ast = try encoder.encodeOperation(extract_right);

    if (encoder.isDegraded()) {
        std.debug.print("pure callee struct update degradation: {s}\n", .{encoder.degradationReason().?});
    }
    try testing.expect(!encoder.isDegraded());

    const constraints = try encoder.takeConstraints(testing.allocator);
    defer if (constraints.len > 0) testing.allocator.free(constraints);

    var solver = try Solver.init(&z3_ctx, testing.allocator);
    defer solver.deinit();
    for (constraints) |cst| solver.assert(cst);
    solver.assert(z3.Z3_mk_not(z3_ctx.ctx, z3.Z3_mk_eq(z3_ctx.ctx, right_ast, try encoder.encodeValue(two))));
    try testing.expectEqual(@as(z3.Z3_lbool, z3.Z3_L_FALSE), solver.check());
}

test "struct_field_update recovers untouched fields from source struct_init metadata" {
    var z3_ctx = try Context.init(testing.allocator);
    defer z3_ctx.deinit();

    var encoder = Encoder.init(&z3_ctx, testing.allocator);
    defer encoder.deinit();

    const mlir_ctx = mlir.oraContextCreate();
    defer mlir.oraContextDestroy(mlir_ctx);
    loadAllDialects(mlir_ctx);
    _ = mlir.oraDialectRegister(mlir_ctx);

    const loc = mlir.oraLocationUnknownGet(mlir_ctx);
    const i256_ty = mlir.oraIntegerTypeCreate(mlir_ctx, 256);
    const struct_ty = mlir.oraStructTypeGet(mlir_ctx, stringRef("Pair__u256"));

    const one = mlir.oraOperationGetResult(mlir.oraArithConstantOpCreate(
        mlir_ctx,
        loc,
        i256_ty,
        mlir.oraIntegerAttrCreateI64FromType(i256_ty, 1),
    ), 0);
    const two = mlir.oraOperationGetResult(mlir.oraArithConstantOpCreate(
        mlir_ctx,
        loc,
        i256_ty,
        mlir.oraIntegerAttrCreateI64FromType(i256_ty, 2),
    ), 0);
    const seven = mlir.oraOperationGetResult(mlir.oraArithConstantOpCreate(
        mlir_ctx,
        loc,
        i256_ty,
        mlir.oraIntegerAttrCreateI64FromType(i256_ty, 7),
    ), 0);

    const fields = [_]mlir.MlirValue{ one, two };
    const init_op = mlir.oraStructInitOpCreate(mlir_ctx, loc, &fields, fields.len, struct_ty);
    const field_name_attrs = [_]mlir.MlirAttribute{
        mlir.oraStringAttrCreate(mlir_ctx, stringRef("left")),
        mlir.oraStringAttrCreate(mlir_ctx, stringRef("right")),
    };
    mlir.oraOperationSetAttributeByName(init_op, stringRef("ora.field_names"), mlir.oraArrayAttrCreate(mlir_ctx, field_name_attrs.len, &field_name_attrs));
    const pair = mlir.oraOperationGetResult(init_op, 0);
    const update_op = mlir.oraStructFieldUpdateOpCreate(mlir_ctx, loc, pair, stringRef("left"), seven);
    const updated_pair = mlir.oraOperationGetResult(update_op, 0);

    const extract_right = mlir.oraStructFieldExtractOpCreate(mlir_ctx, loc, updated_pair, stringRef("right"), i256_ty);
    const right_ast = try encoder.encodeOperation(extract_right);

    try testing.expect(!encoder.isDegraded());

    const constraints = try encoder.takeConstraints(testing.allocator);
    defer if (constraints.len > 0) testing.allocator.free(constraints);

    var solver = try Solver.init(&z3_ctx, testing.allocator);
    defer solver.deinit();
    for (constraints) |cst| solver.assert(cst);
    solver.assert(z3.Z3_mk_not(z3_ctx.ctx, z3.Z3_mk_eq(z3_ctx.ctx, right_ast, try encoder.encodeValue(two))));
    try testing.expectEqual(@as(z3.Z3_lbool, z3.Z3_L_FALSE), solver.check());
}

test "known pure callee returning source-metadata struct update preserves untouched fields exactly" {
    var z3_ctx = try Context.init(testing.allocator);
    defer z3_ctx.deinit();

    var encoder = Encoder.init(&z3_ctx, testing.allocator);
    defer encoder.deinit();

    const mlir_ctx = mlir.oraContextCreate();
    defer mlir.oraContextDestroy(mlir_ctx);
    loadAllDialects(mlir_ctx);
    _ = mlir.oraDialectRegister(mlir_ctx);

    const loc = mlir.oraLocationUnknownGet(mlir_ctx);
    const i256_ty = mlir.oraIntegerTypeCreate(mlir_ctx, 256);
    const struct_ty = mlir.oraStructTypeGet(mlir_ctx, stringRef("Pair__u256"));

    const helper_attrs = [_]mlir.MlirNamedAttribute{
        namedAttr(mlir_ctx, "sym_name", mlir.oraStringAttrCreate(mlir_ctx, stringRef("makePairViaInit"))),
    };
    const helper = mlir.oraFuncFuncOpCreate(mlir_ctx, loc, &helper_attrs, helper_attrs.len, &[_]mlir.MlirType{}, &[_]mlir.MlirLocation{}, 0);
    const helper_body = mlir.oraFuncOpGetBodyBlock(helper);

    const one = mlir.oraOperationGetResult(mlir.oraArithConstantOpCreate(
        mlir_ctx,
        loc,
        i256_ty,
        mlir.oraIntegerAttrCreateI64FromType(i256_ty, 1),
    ), 0);
    const two = mlir.oraOperationGetResult(mlir.oraArithConstantOpCreate(
        mlir_ctx,
        loc,
        i256_ty,
        mlir.oraIntegerAttrCreateI64FromType(i256_ty, 2),
    ), 0);
    const seven = mlir.oraOperationGetResult(mlir.oraArithConstantOpCreate(
        mlir_ctx,
        loc,
        i256_ty,
        mlir.oraIntegerAttrCreateI64FromType(i256_ty, 7),
    ), 0);
    const fields = [_]mlir.MlirValue{ one, two };
    const init_op = mlir.oraStructInitOpCreate(mlir_ctx, loc, &fields, fields.len, struct_ty);
    const field_name_attrs = [_]mlir.MlirAttribute{
        mlir.oraStringAttrCreate(mlir_ctx, stringRef("left")),
        mlir.oraStringAttrCreate(mlir_ctx, stringRef("right")),
    };
    mlir.oraOperationSetAttributeByName(init_op, stringRef("ora.field_names"), mlir.oraArrayAttrCreate(mlir_ctx, field_name_attrs.len, &field_name_attrs));
    const pair = mlir.oraOperationGetResult(init_op, 0);
    const update_op = mlir.oraStructFieldUpdateOpCreate(mlir_ctx, loc, pair, stringRef("left"), seven);
    const updated_pair = mlir.oraOperationGetResult(update_op, 0);
    const ret_vals = [_]mlir.MlirValue{updated_pair};
    const ret_op = mlir.oraReturnOpCreate(mlir_ctx, loc, &ret_vals, ret_vals.len);
    mlir.oraBlockAppendOwnedOperation(helper_body, init_op);
    mlir.oraBlockAppendOwnedOperation(helper_body, update_op);
    mlir.oraBlockAppendOwnedOperation(helper_body, ret_op);
    try encoder.registerFunctionOperation(helper);

    const call = mlir.oraFuncCallOpCreate(mlir_ctx, loc, stringRef("makePairViaInit"), &[_]mlir.MlirValue{}, 0, &[_]mlir.MlirType{struct_ty}, 1);
    const call_result = mlir.oraOperationGetResult(call, 0);
    const extract_right = mlir.oraStructFieldExtractOpCreate(mlir_ctx, loc, call_result, stringRef("right"), i256_ty);
    const right_ast = try encoder.encodeOperation(extract_right);

    try testing.expect(!encoder.isDegraded());

    const constraints = try encoder.takeConstraints(testing.allocator);
    defer if (constraints.len > 0) testing.allocator.free(constraints);

    var solver = try Solver.init(&z3_ctx, testing.allocator);
    defer solver.deinit();
    for (constraints) |cst| solver.assert(cst);
    solver.assert(z3.Z3_mk_not(z3_ctx.ctx, z3.Z3_mk_eq(z3_ctx.ctx, right_ast, try encoder.encodeValue(two))));
    try testing.expectEqual(@as(z3.Z3_lbool, z3.Z3_L_FALSE), solver.check());
}

test "caller struct_field_update recovers untouched fields from known callee source metadata" {
    var z3_ctx = try Context.init(testing.allocator);
    defer z3_ctx.deinit();

    var encoder = Encoder.init(&z3_ctx, testing.allocator);
    defer encoder.deinit();

    const mlir_ctx = mlir.oraContextCreate();
    defer mlir.oraContextDestroy(mlir_ctx);
    loadAllDialects(mlir_ctx);
    _ = mlir.oraDialectRegister(mlir_ctx);

    const loc = mlir.oraLocationUnknownGet(mlir_ctx);
    const i256_ty = mlir.oraIntegerTypeCreate(mlir_ctx, 256);
    const struct_ty = mlir.oraStructTypeGet(mlir_ctx, stringRef("Pair__u256"));

    const helper_attrs = [_]mlir.MlirNamedAttribute{
        namedAttr(mlir_ctx, "sym_name", mlir.oraStringAttrCreate(mlir_ctx, stringRef("buildPairViaInit"))),
    };
    const helper = mlir.oraFuncFuncOpCreate(mlir_ctx, loc, &helper_attrs, helper_attrs.len, &[_]mlir.MlirType{}, &[_]mlir.MlirLocation{}, 0);
    const helper_body = mlir.oraFuncOpGetBodyBlock(helper);

    const one = mlir.oraOperationGetResult(mlir.oraArithConstantOpCreate(
        mlir_ctx,
        loc,
        i256_ty,
        mlir.oraIntegerAttrCreateI64FromType(i256_ty, 1),
    ), 0);
    const two = mlir.oraOperationGetResult(mlir.oraArithConstantOpCreate(
        mlir_ctx,
        loc,
        i256_ty,
        mlir.oraIntegerAttrCreateI64FromType(i256_ty, 2),
    ), 0);
    const seven = mlir.oraOperationGetResult(mlir.oraArithConstantOpCreate(
        mlir_ctx,
        loc,
        i256_ty,
        mlir.oraIntegerAttrCreateI64FromType(i256_ty, 7),
    ), 0);

    const init_op = mlir.oraStructInitOpCreate(mlir_ctx, loc, &[_]mlir.MlirValue{ one, two }, 2, struct_ty);
    const field_name_attrs = [_]mlir.MlirAttribute{
        mlir.oraStringAttrCreate(mlir_ctx, stringRef("left")),
        mlir.oraStringAttrCreate(mlir_ctx, stringRef("right")),
    };
    mlir.oraOperationSetAttributeByName(init_op, stringRef("ora.field_names"), mlir.oraArrayAttrCreate(mlir_ctx, field_name_attrs.len, &field_name_attrs));
    mlir.oraBlockAppendOwnedOperation(helper_body, init_op);
    mlir.oraBlockAppendOwnedOperation(helper_body, mlir.oraReturnOpCreate(
        mlir_ctx,
        loc,
        &[_]mlir.MlirValue{mlir.oraOperationGetResult(init_op, 0)},
        1,
    ));
    try encoder.registerFunctionOperation(helper);

    const call = mlir.oraFuncCallOpCreate(mlir_ctx, loc, stringRef("buildPairViaInit"), &[_]mlir.MlirValue{}, 0, &[_]mlir.MlirType{struct_ty}, 1);
    const updated = mlir.oraStructFieldUpdateOpCreate(mlir_ctx, loc, mlir.oraOperationGetResult(call, 0), stringRef("left"), seven);
    const extract_right = mlir.oraStructFieldExtractOpCreate(mlir_ctx, loc, mlir.oraOperationGetResult(updated, 0), stringRef("right"), i256_ty);
    const right_ast = try encoder.encodeOperation(extract_right);

    try testing.expect(!encoder.isDegraded());

    const constraints = try encoder.takeConstraints(testing.allocator);
    defer if (constraints.len > 0) testing.allocator.free(constraints);

    var solver = try Solver.init(&z3_ctx, testing.allocator);
    defer solver.deinit();
    for (constraints) |cst| solver.assert(cst);
    solver.assert(z3.Z3_mk_not(z3_ctx.ctx, z3.Z3_mk_eq(z3_ctx.ctx, right_ast, try encoder.encodeValue(two))));
    try testing.expectEqual(@as(z3.Z3_lbool, z3.Z3_L_FALSE), solver.check());
}

test "caller struct_field_update recovers untouched fields from known callee structured source metadata" {
    var z3_ctx = try Context.init(testing.allocator);
    defer z3_ctx.deinit();

    var encoder = Encoder.init(&z3_ctx, testing.allocator);
    defer encoder.deinit();

    const mlir_ctx = mlir.oraContextCreate();
    defer mlir.oraContextDestroy(mlir_ctx);
    loadAllDialects(mlir_ctx);
    _ = mlir.oraDialectRegister(mlir_ctx);

    const loc = mlir.oraLocationUnknownGet(mlir_ctx);
    const i1_ty = mlir.oraIntegerTypeCreate(mlir_ctx, 1);
    const i256_ty = mlir.oraIntegerTypeCreate(mlir_ctx, 256);
    const struct_ty = mlir.oraStructTypeGet(mlir_ctx, stringRef("Pair__u256"));

    const helper_attrs = [_]mlir.MlirNamedAttribute{
        namedAttr(mlir_ctx, "sym_name", mlir.oraStringAttrCreate(mlir_ctx, stringRef("buildPairViaIf"))),
    };
    const helper = mlir.oraFuncFuncOpCreate(mlir_ctx, loc, &helper_attrs, helper_attrs.len, &[_]mlir.MlirType{}, &[_]mlir.MlirLocation{}, 0);
    const helper_body = mlir.oraFuncOpGetBodyBlock(helper);

    const cond = mlir.oraOperationGetResult(mlir.oraArithConstantOpCreate(
        mlir_ctx,
        loc,
        i1_ty,
        mlir.oraIntegerAttrCreateI64FromType(i1_ty, 0),
    ), 0);
    const one = mlir.oraOperationGetResult(mlir.oraArithConstantOpCreate(
        mlir_ctx,
        loc,
        i256_ty,
        mlir.oraIntegerAttrCreateI64FromType(i256_ty, 1),
    ), 0);
    const three = mlir.oraOperationGetResult(mlir.oraArithConstantOpCreate(
        mlir_ctx,
        loc,
        i256_ty,
        mlir.oraIntegerAttrCreateI64FromType(i256_ty, 3),
    ), 0);
    const two = mlir.oraOperationGetResult(mlir.oraArithConstantOpCreate(
        mlir_ctx,
        loc,
        i256_ty,
        mlir.oraIntegerAttrCreateI64FromType(i256_ty, 2),
    ), 0);
    const seven = mlir.oraOperationGetResult(mlir.oraArithConstantOpCreate(
        mlir_ctx,
        loc,
        i256_ty,
        mlir.oraIntegerAttrCreateI64FromType(i256_ty, 7),
    ), 0);

    const if_op = mlir.oraScfIfOpCreate(mlir_ctx, loc, cond, &[_]mlir.MlirType{struct_ty}, 1, true);
    const then_block = mlir.oraScfIfOpGetThenBlock(if_op);
    const else_block = mlir.oraScfIfOpGetElseBlock(if_op);
    const field_name_attrs = [_]mlir.MlirAttribute{
        mlir.oraStringAttrCreate(mlir_ctx, stringRef("left")),
        mlir.oraStringAttrCreate(mlir_ctx, stringRef("right")),
    };

    const then_init = mlir.oraStructInitOpCreate(mlir_ctx, loc, &[_]mlir.MlirValue{ one, two }, 2, struct_ty);
    mlir.oraOperationSetAttributeByName(then_init, stringRef("ora.field_names"), mlir.oraArrayAttrCreate(mlir_ctx, field_name_attrs.len, &field_name_attrs));
    mlir.oraBlockAppendOwnedOperation(then_block, then_init);
    mlir.oraBlockAppendOwnedOperation(then_block, mlir.oraScfYieldOpCreate(
        mlir_ctx,
        loc,
        &[_]mlir.MlirValue{mlir.oraOperationGetResult(then_init, 0)},
        1,
    ));

    const else_init = mlir.oraStructInitOpCreate(mlir_ctx, loc, &[_]mlir.MlirValue{ three, two }, 2, struct_ty);
    mlir.oraOperationSetAttributeByName(else_init, stringRef("ora.field_names"), mlir.oraArrayAttrCreate(mlir_ctx, field_name_attrs.len, &field_name_attrs));
    mlir.oraBlockAppendOwnedOperation(else_block, else_init);
    mlir.oraBlockAppendOwnedOperation(else_block, mlir.oraScfYieldOpCreate(
        mlir_ctx,
        loc,
        &[_]mlir.MlirValue{mlir.oraOperationGetResult(else_init, 0)},
        1,
    ));

    mlir.oraBlockAppendOwnedOperation(helper_body, if_op);
    mlir.oraBlockAppendOwnedOperation(helper_body, mlir.oraReturnOpCreate(
        mlir_ctx,
        loc,
        &[_]mlir.MlirValue{mlir.oraOperationGetResult(if_op, 0)},
        1,
    ));
    try encoder.registerFunctionOperation(helper);

    const call = mlir.oraFuncCallOpCreate(mlir_ctx, loc, stringRef("buildPairViaIf"), &[_]mlir.MlirValue{}, 0, &[_]mlir.MlirType{struct_ty}, 1);
    const updated = mlir.oraStructFieldUpdateOpCreate(mlir_ctx, loc, mlir.oraOperationGetResult(call, 0), stringRef("left"), seven);
    const extract_right = mlir.oraStructFieldExtractOpCreate(mlir_ctx, loc, mlir.oraOperationGetResult(updated, 0), stringRef("right"), i256_ty);
    const right_ast = try encoder.encodeOperation(extract_right);

    try testing.expect(!encoder.isDegraded());

    const constraints = try encoder.takeConstraints(testing.allocator);
    defer if (constraints.len > 0) testing.allocator.free(constraints);

    var solver = try Solver.init(&z3_ctx, testing.allocator);
    defer solver.deinit();
    for (constraints) |cst| solver.assert(cst);
    solver.assert(z3.Z3_mk_not(z3_ctx.ctx, z3.Z3_mk_eq(z3_ctx.ctx, right_ast, try encoder.encodeValue(two))));
    try testing.expectEqual(@as(z3.Z3_lbool, z3.Z3_L_FALSE), solver.check());
}

test "caller struct_field_update recovers untouched fields from known callee branch returns" {
    var z3_ctx = try Context.init(testing.allocator);
    defer z3_ctx.deinit();

    var encoder = Encoder.init(&z3_ctx, testing.allocator);
    defer encoder.deinit();

    const mlir_ctx = mlir.oraContextCreate();
    defer mlir.oraContextDestroy(mlir_ctx);
    loadAllDialects(mlir_ctx);
    _ = mlir.oraDialectRegister(mlir_ctx);

    const loc = mlir.oraLocationUnknownGet(mlir_ctx);
    const i1_ty = mlir.oraIntegerTypeCreate(mlir_ctx, 1);
    const i256_ty = mlir.oraIntegerTypeCreate(mlir_ctx, 256);
    const struct_ty = mlir.oraStructTypeGet(mlir_ctx, stringRef("Pair__u256"));

    const helper_attrs = [_]mlir.MlirNamedAttribute{
        namedAttr(mlir_ctx, "sym_name", mlir.oraStringAttrCreate(mlir_ctx, stringRef("buildPairViaConditionalReturn"))),
    };
    const helper = mlir.oraFuncFuncOpCreate(mlir_ctx, loc, &helper_attrs, helper_attrs.len, &[_]mlir.MlirType{i1_ty}, &[_]mlir.MlirLocation{loc}, 1);
    const helper_body = mlir.oraFuncOpGetBodyBlock(helper);
    const helper_flag = mlir.oraBlockGetArgument(helper_body, 0);

    const one = mlir.oraOperationGetResult(mlir.oraArithConstantOpCreate(
        mlir_ctx,
        loc,
        i256_ty,
        mlir.oraIntegerAttrCreateI64FromType(i256_ty, 1),
    ), 0);
    const three = mlir.oraOperationGetResult(mlir.oraArithConstantOpCreate(
        mlir_ctx,
        loc,
        i256_ty,
        mlir.oraIntegerAttrCreateI64FromType(i256_ty, 3),
    ), 0);
    const two = mlir.oraOperationGetResult(mlir.oraArithConstantOpCreate(
        mlir_ctx,
        loc,
        i256_ty,
        mlir.oraIntegerAttrCreateI64FromType(i256_ty, 2),
    ), 0);
    const seven = mlir.oraOperationGetResult(mlir.oraArithConstantOpCreate(
        mlir_ctx,
        loc,
        i256_ty,
        mlir.oraIntegerAttrCreateI64FromType(i256_ty, 7),
    ), 0);
    const field_name_attrs = [_]mlir.MlirAttribute{
        mlir.oraStringAttrCreate(mlir_ctx, stringRef("left")),
        mlir.oraStringAttrCreate(mlir_ctx, stringRef("right")),
    };

    const conditional_ret = mlir.oraConditionalReturnOpCreate(mlir_ctx, loc, helper_flag);
    const then_block = mlir.oraConditionalReturnOpGetThenBlock(conditional_ret);
    const else_block = mlir.oraConditionalReturnOpGetElseBlock(conditional_ret);

    const then_init = mlir.oraStructInitOpCreate(mlir_ctx, loc, &[_]mlir.MlirValue{ one, two }, 2, struct_ty);
    mlir.oraOperationSetAttributeByName(then_init, stringRef("ora.field_names"), mlir.oraArrayAttrCreate(mlir_ctx, field_name_attrs.len, &field_name_attrs));
    mlir.oraBlockAppendOwnedOperation(then_block, then_init);
    mlir.oraBlockAppendOwnedOperation(then_block, mlir.oraReturnOpCreate(
        mlir_ctx,
        loc,
        &[_]mlir.MlirValue{mlir.oraOperationGetResult(then_init, 0)},
        1,
    ));
    mlir.oraBlockAppendOwnedOperation(else_block, mlir.oraYieldOpCreate(mlir_ctx, loc, &[_]mlir.MlirValue{}, 0));

    const else_init = mlir.oraStructInitOpCreate(mlir_ctx, loc, &[_]mlir.MlirValue{ three, two }, 2, struct_ty);
    mlir.oraOperationSetAttributeByName(else_init, stringRef("ora.field_names"), mlir.oraArrayAttrCreate(mlir_ctx, field_name_attrs.len, &field_name_attrs));
    mlir.oraBlockAppendOwnedOperation(helper_body, conditional_ret);
    mlir.oraBlockAppendOwnedOperation(helper_body, else_init);
    mlir.oraBlockAppendOwnedOperation(helper_body, mlir.oraReturnOpCreate(
        mlir_ctx,
        loc,
        &[_]mlir.MlirValue{mlir.oraOperationGetResult(else_init, 0)},
        1,
    ));
    try encoder.registerFunctionOperation(helper);

    const flag = mlir.oraOperationGetResult(mlir.oraArithConstantOpCreate(
        mlir_ctx,
        loc,
        i1_ty,
        mlir.oraIntegerAttrCreateI64FromType(i1_ty, 0),
    ), 0);
    const call = mlir.oraFuncCallOpCreate(mlir_ctx, loc, stringRef("buildPairViaConditionalReturn"), &[_]mlir.MlirValue{flag}, 1, &[_]mlir.MlirType{struct_ty}, 1);
    const updated = mlir.oraStructFieldUpdateOpCreate(mlir_ctx, loc, mlir.oraOperationGetResult(call, 0), stringRef("left"), seven);
    const extract_right = mlir.oraStructFieldExtractOpCreate(mlir_ctx, loc, mlir.oraOperationGetResult(updated, 0), stringRef("right"), i256_ty);
    const right_ast = try encoder.encodeOperation(extract_right);

    try testing.expect(!encoder.isDegraded());

    const constraints = try encoder.takeConstraints(testing.allocator);
    defer if (constraints.len > 0) testing.allocator.free(constraints);

    var solver = try Solver.init(&z3_ctx, testing.allocator);
    defer solver.deinit();
    for (constraints) |cst| solver.assert(cst);
    solver.assert(z3.Z3_mk_not(z3_ctx.ctx, z3.Z3_mk_eq(z3_ctx.ctx, right_ast, try encoder.encodeValue(two))));
    try testing.expectEqual(@as(z3.Z3_lbool, z3.Z3_L_FALSE), solver.check());
}

test "caller struct_field_update recovers untouched fields from known callee switch returns" {
    var z3_ctx = try Context.init(testing.allocator);
    defer z3_ctx.deinit();

    var encoder = Encoder.init(&z3_ctx, testing.allocator);
    defer encoder.deinit();

    const mlir_ctx = mlir.oraContextCreate();
    defer mlir.oraContextDestroy(mlir_ctx);
    loadAllDialects(mlir_ctx);
    _ = mlir.oraDialectRegister(mlir_ctx);

    const loc = mlir.oraLocationUnknownGet(mlir_ctx);
    const i256_ty = mlir.oraIntegerTypeCreate(mlir_ctx, 256);
    const struct_ty = mlir.oraStructTypeGet(mlir_ctx, stringRef("Pair__u256"));

    const helper_attrs = [_]mlir.MlirNamedAttribute{
        namedAttr(mlir_ctx, "sym_name", mlir.oraStringAttrCreate(mlir_ctx, stringRef("buildPairViaSwitchReturn"))),
    };
    const helper = mlir.oraFuncFuncOpCreate(mlir_ctx, loc, &helper_attrs, helper_attrs.len, &[_]mlir.MlirType{i256_ty}, &[_]mlir.MlirLocation{loc}, 1);
    const helper_body = mlir.oraFuncOpGetBodyBlock(helper);
    const helper_scrutinee = mlir.oraBlockGetArgument(helper_body, 0);

    const one = mlir.oraOperationGetResult(mlir.oraArithConstantOpCreate(
        mlir_ctx,
        loc,
        i256_ty,
        mlir.oraIntegerAttrCreateI64FromType(i256_ty, 1),
    ), 0);
    const three = mlir.oraOperationGetResult(mlir.oraArithConstantOpCreate(
        mlir_ctx,
        loc,
        i256_ty,
        mlir.oraIntegerAttrCreateI64FromType(i256_ty, 3),
    ), 0);
    const two = mlir.oraOperationGetResult(mlir.oraArithConstantOpCreate(
        mlir_ctx,
        loc,
        i256_ty,
        mlir.oraIntegerAttrCreateI64FromType(i256_ty, 2),
    ), 0);
    const seven = mlir.oraOperationGetResult(mlir.oraArithConstantOpCreate(
        mlir_ctx,
        loc,
        i256_ty,
        mlir.oraIntegerAttrCreateI64FromType(i256_ty, 7),
    ), 0);
    const field_name_attrs = [_]mlir.MlirAttribute{
        mlir.oraStringAttrCreate(mlir_ctx, stringRef("left")),
        mlir.oraStringAttrCreate(mlir_ctx, stringRef("right")),
    };

    const switch_op = mlir.oraSwitchOpCreateWithCases(
        mlir_ctx,
        loc,
        helper_scrutinee,
        &[_]mlir.MlirType{},
        0,
        2,
    );
    const case_values = [_]i64{1};
    const range_starts = [_]i64{0};
    const range_ends = [_]i64{0};
    const case_kinds = [_]i64{0};
    mlir.oraSwitchOpSetCasePatterns(
        switch_op,
        &case_values,
        &range_starts,
        &range_ends,
        &case_kinds,
        1,
        case_values.len,
    );
    const case_block = mlir.oraSwitchOpGetCaseBlock(switch_op, 0);
    const default_block = mlir.oraSwitchOpGetCaseBlock(switch_op, 1);

    const case_init = mlir.oraStructInitOpCreate(mlir_ctx, loc, &[_]mlir.MlirValue{ one, two }, 2, struct_ty);
    mlir.oraOperationSetAttributeByName(case_init, stringRef("ora.field_names"), mlir.oraArrayAttrCreate(mlir_ctx, field_name_attrs.len, &field_name_attrs));
    mlir.oraBlockAppendOwnedOperation(case_block, case_init);
    mlir.oraBlockAppendOwnedOperation(case_block, mlir.oraReturnOpCreate(
        mlir_ctx,
        loc,
        &[_]mlir.MlirValue{mlir.oraOperationGetResult(case_init, 0)},
        1,
    ));

    const default_init = mlir.oraStructInitOpCreate(mlir_ctx, loc, &[_]mlir.MlirValue{ three, two }, 2, struct_ty);
    mlir.oraOperationSetAttributeByName(default_init, stringRef("ora.field_names"), mlir.oraArrayAttrCreate(mlir_ctx, field_name_attrs.len, &field_name_attrs));
    mlir.oraBlockAppendOwnedOperation(default_block, default_init);
    mlir.oraBlockAppendOwnedOperation(default_block, mlir.oraReturnOpCreate(
        mlir_ctx,
        loc,
        &[_]mlir.MlirValue{mlir.oraOperationGetResult(default_init, 0)},
        1,
    ));

    mlir.oraBlockAppendOwnedOperation(helper_body, switch_op);
    try encoder.registerFunctionOperation(helper);

    const scrutinee = mlir.oraOperationGetResult(mlir.oraArithConstantOpCreate(
        mlir_ctx,
        loc,
        i256_ty,
        mlir.oraIntegerAttrCreateI64FromType(i256_ty, 0),
    ), 0);
    const call = mlir.oraFuncCallOpCreate(mlir_ctx, loc, stringRef("buildPairViaSwitchReturn"), &[_]mlir.MlirValue{scrutinee}, 1, &[_]mlir.MlirType{struct_ty}, 1);
    const updated = mlir.oraStructFieldUpdateOpCreate(mlir_ctx, loc, mlir.oraOperationGetResult(call, 0), stringRef("left"), seven);
    const extract_right = mlir.oraStructFieldExtractOpCreate(mlir_ctx, loc, mlir.oraOperationGetResult(updated, 0), stringRef("right"), i256_ty);
    const right_ast = try encoder.encodeOperation(extract_right);

    try testing.expect(!encoder.isDegraded());

    const constraints = try encoder.takeConstraints(testing.allocator);
    defer if (constraints.len > 0) testing.allocator.free(constraints);

    var solver = try Solver.init(&z3_ctx, testing.allocator);
    defer solver.deinit();
    for (constraints) |cst| solver.assert(cst);
    solver.assert(z3.Z3_mk_not(z3_ctx.ctx, z3.Z3_mk_eq(z3_ctx.ctx, right_ast, try encoder.encodeValue(two))));
    try testing.expectEqual(@as(z3.Z3_lbool, z3.Z3_L_FALSE), solver.check());
}

test "caller struct_field_update recovers untouched fields from known callee execute_region returns" {
    var z3_ctx = try Context.init(testing.allocator);
    defer z3_ctx.deinit();

    var encoder = Encoder.init(&z3_ctx, testing.allocator);
    defer encoder.deinit();

    const mlir_ctx = mlir.oraContextCreate();
    defer mlir.oraContextDestroy(mlir_ctx);
    loadAllDialects(mlir_ctx);
    _ = mlir.oraDialectRegister(mlir_ctx);

    const loc = mlir.oraLocationUnknownGet(mlir_ctx);
    const i256_ty = mlir.oraIntegerTypeCreate(mlir_ctx, 256);
    const struct_ty = mlir.oraStructTypeGet(mlir_ctx, stringRef("Pair__u256"));

    const helper_attrs = [_]mlir.MlirNamedAttribute{
        namedAttr(mlir_ctx, "sym_name", mlir.oraStringAttrCreate(mlir_ctx, stringRef("buildPairViaExecuteRegionReturn"))),
    };
    const helper = mlir.oraFuncFuncOpCreate(mlir_ctx, loc, &helper_attrs, helper_attrs.len, &[_]mlir.MlirType{}, &[_]mlir.MlirLocation{}, 0);
    const helper_body = mlir.oraFuncOpGetBodyBlock(helper);

    const one = mlir.oraOperationGetResult(mlir.oraArithConstantOpCreate(
        mlir_ctx,
        loc,
        i256_ty,
        mlir.oraIntegerAttrCreateI64FromType(i256_ty, 1),
    ), 0);
    const two = mlir.oraOperationGetResult(mlir.oraArithConstantOpCreate(
        mlir_ctx,
        loc,
        i256_ty,
        mlir.oraIntegerAttrCreateI64FromType(i256_ty, 2),
    ), 0);
    const seven = mlir.oraOperationGetResult(mlir.oraArithConstantOpCreate(
        mlir_ctx,
        loc,
        i256_ty,
        mlir.oraIntegerAttrCreateI64FromType(i256_ty, 7),
    ), 0);
    const field_name_attrs = [_]mlir.MlirAttribute{
        mlir.oraStringAttrCreate(mlir_ctx, stringRef("left")),
        mlir.oraStringAttrCreate(mlir_ctx, stringRef("right")),
    };

    const exec = mlir.oraScfExecuteRegionOpCreate(mlir_ctx, loc, &[_]mlir.MlirType{}, 0, false);
    const exec_block = mlir.oraScfExecuteRegionOpGetBodyBlock(exec);
    const init_op = mlir.oraStructInitOpCreate(mlir_ctx, loc, &[_]mlir.MlirValue{ one, two }, 2, struct_ty);
    mlir.oraOperationSetAttributeByName(init_op, stringRef("ora.field_names"), mlir.oraArrayAttrCreate(mlir_ctx, field_name_attrs.len, &field_name_attrs));
    mlir.oraBlockAppendOwnedOperation(exec_block, init_op);
    mlir.oraBlockAppendOwnedOperation(exec_block, mlir.oraReturnOpCreate(
        mlir_ctx,
        loc,
        &[_]mlir.MlirValue{mlir.oraOperationGetResult(init_op, 0)},
        1,
    ));

    mlir.oraBlockAppendOwnedOperation(helper_body, exec);
    try encoder.registerFunctionOperation(helper);

    const call = mlir.oraFuncCallOpCreate(mlir_ctx, loc, stringRef("buildPairViaExecuteRegionReturn"), &[_]mlir.MlirValue{}, 0, &[_]mlir.MlirType{struct_ty}, 1);
    const updated = mlir.oraStructFieldUpdateOpCreate(mlir_ctx, loc, mlir.oraOperationGetResult(call, 0), stringRef("left"), seven);
    const extract_right = mlir.oraStructFieldExtractOpCreate(mlir_ctx, loc, mlir.oraOperationGetResult(updated, 0), stringRef("right"), i256_ty);
    const right_ast = try encoder.encodeOperation(extract_right);

    try testing.expect(!encoder.isDegraded());

    const constraints = try encoder.takeConstraints(testing.allocator);
    defer if (constraints.len > 0) testing.allocator.free(constraints);

    var solver = try Solver.init(&z3_ctx, testing.allocator);
    defer solver.deinit();
    for (constraints) |cst| solver.assert(cst);
    solver.assert(z3.Z3_mk_not(z3_ctx.ctx, z3.Z3_mk_eq(z3_ctx.ctx, right_ast, try encoder.encodeValue(two))));
    try testing.expectEqual(@as(z3.Z3_lbool, z3.Z3_L_FALSE), solver.check());
}

test "struct_field_update recovers untouched fields through scf.if source metadata" {
    var z3_ctx = try Context.init(testing.allocator);
    defer z3_ctx.deinit();

    var encoder = Encoder.init(&z3_ctx, testing.allocator);
    defer encoder.deinit();

    const mlir_ctx = mlir.oraContextCreate();
    defer mlir.oraContextDestroy(mlir_ctx);
    loadAllDialects(mlir_ctx);
    _ = mlir.oraDialectRegister(mlir_ctx);

    const loc = mlir.oraLocationUnknownGet(mlir_ctx);
    const i1_ty = mlir.oraIntegerTypeCreate(mlir_ctx, 1);
    const i256_ty = mlir.oraIntegerTypeCreate(mlir_ctx, 256);
    const struct_ty = mlir.oraStructTypeGet(mlir_ctx, stringRef("Pair__u256"));

    const zero = mlir.oraOperationGetResult(mlir.oraArithConstantOpCreate(
        mlir_ctx,
        loc,
        i1_ty,
        mlir.oraIntegerAttrCreateI64FromType(i1_ty, 0),
    ), 0);
    const one = mlir.oraOperationGetResult(mlir.oraArithConstantOpCreate(
        mlir_ctx,
        loc,
        i256_ty,
        mlir.oraIntegerAttrCreateI64FromType(i256_ty, 1),
    ), 0);
    const three = mlir.oraOperationGetResult(mlir.oraArithConstantOpCreate(
        mlir_ctx,
        loc,
        i256_ty,
        mlir.oraIntegerAttrCreateI64FromType(i256_ty, 3),
    ), 0);
    const two = mlir.oraOperationGetResult(mlir.oraArithConstantOpCreate(
        mlir_ctx,
        loc,
        i256_ty,
        mlir.oraIntegerAttrCreateI64FromType(i256_ty, 2),
    ), 0);
    const seven = mlir.oraOperationGetResult(mlir.oraArithConstantOpCreate(
        mlir_ctx,
        loc,
        i256_ty,
        mlir.oraIntegerAttrCreateI64FromType(i256_ty, 7),
    ), 0);

    const if_op = mlir.oraScfIfOpCreate(mlir_ctx, loc, zero, &[_]mlir.MlirType{struct_ty}, 1, true);
    const then_block = mlir.oraScfIfOpGetThenBlock(if_op);
    const else_block = mlir.oraScfIfOpGetElseBlock(if_op);

    const then_init = mlir.oraStructInitOpCreate(mlir_ctx, loc, &[_]mlir.MlirValue{ one, two }, 2, struct_ty);
    const else_init = mlir.oraStructInitOpCreate(mlir_ctx, loc, &[_]mlir.MlirValue{ three, two }, 2, struct_ty);
    const field_name_attrs = [_]mlir.MlirAttribute{
        mlir.oraStringAttrCreate(mlir_ctx, stringRef("left")),
        mlir.oraStringAttrCreate(mlir_ctx, stringRef("right")),
    };
    mlir.oraOperationSetAttributeByName(then_init, stringRef("ora.field_names"), mlir.oraArrayAttrCreate(mlir_ctx, field_name_attrs.len, &field_name_attrs));
    mlir.oraOperationSetAttributeByName(else_init, stringRef("ora.field_names"), mlir.oraArrayAttrCreate(mlir_ctx, field_name_attrs.len, &field_name_attrs));
    mlir.oraBlockAppendOwnedOperation(then_block, then_init);
    mlir.oraBlockAppendOwnedOperation(then_block, mlir.oraScfYieldOpCreate(mlir_ctx, loc, &[_]mlir.MlirValue{mlir.oraOperationGetResult(then_init, 0)}, 1));
    mlir.oraBlockAppendOwnedOperation(else_block, else_init);
    mlir.oraBlockAppendOwnedOperation(else_block, mlir.oraScfYieldOpCreate(mlir_ctx, loc, &[_]mlir.MlirValue{mlir.oraOperationGetResult(else_init, 0)}, 1));

    const pair = mlir.oraOperationGetResult(if_op, 0);
    const update_op = mlir.oraStructFieldUpdateOpCreate(mlir_ctx, loc, pair, stringRef("left"), seven);
    const updated_pair = mlir.oraOperationGetResult(update_op, 0);
    const extract_right = mlir.oraStructFieldExtractOpCreate(mlir_ctx, loc, updated_pair, stringRef("right"), i256_ty);
    const right_ast = try encoder.encodeOperation(extract_right);

    try testing.expect(!encoder.isDegraded());

    const constraints = try encoder.takeConstraints(testing.allocator);
    defer if (constraints.len > 0) testing.allocator.free(constraints);

    var solver = try Solver.init(&z3_ctx, testing.allocator);
    defer solver.deinit();
    for (constraints) |cst| solver.assert(cst);
    solver.assert(z3.Z3_mk_not(z3_ctx.ctx, z3.Z3_mk_eq(z3_ctx.ctx, right_ast, try encoder.encodeValue(two))));
    try testing.expectEqual(@as(z3.Z3_lbool, z3.Z3_L_FALSE), solver.check());
}

test "struct_field_update recovers untouched fields through ora.switch_expr source metadata" {
    var z3_ctx = try Context.init(testing.allocator);
    defer z3_ctx.deinit();

    var encoder = Encoder.init(&z3_ctx, testing.allocator);
    defer encoder.deinit();

    const mlir_ctx = mlir.oraContextCreate();
    defer mlir.oraContextDestroy(mlir_ctx);
    loadAllDialects(mlir_ctx);
    _ = mlir.oraDialectRegister(mlir_ctx);

    const loc = mlir.oraLocationUnknownGet(mlir_ctx);
    const i256_ty = mlir.oraIntegerTypeCreate(mlir_ctx, 256);
    const struct_ty = mlir.oraStructTypeGet(mlir_ctx, stringRef("Pair__u256"));

    const scrutinee = mlir.oraOperationGetResult(mlir.oraArithConstantOpCreate(
        mlir_ctx,
        loc,
        i256_ty,
        mlir.oraIntegerAttrCreateI64FromType(i256_ty, 1),
    ), 0);
    const one = mlir.oraOperationGetResult(mlir.oraArithConstantOpCreate(
        mlir_ctx,
        loc,
        i256_ty,
        mlir.oraIntegerAttrCreateI64FromType(i256_ty, 1),
    ), 0);
    const three = mlir.oraOperationGetResult(mlir.oraArithConstantOpCreate(
        mlir_ctx,
        loc,
        i256_ty,
        mlir.oraIntegerAttrCreateI64FromType(i256_ty, 3),
    ), 0);
    const two = mlir.oraOperationGetResult(mlir.oraArithConstantOpCreate(
        mlir_ctx,
        loc,
        i256_ty,
        mlir.oraIntegerAttrCreateI64FromType(i256_ty, 2),
    ), 0);
    const seven = mlir.oraOperationGetResult(mlir.oraArithConstantOpCreate(
        mlir_ctx,
        loc,
        i256_ty,
        mlir.oraIntegerAttrCreateI64FromType(i256_ty, 7),
    ), 0);

    const switch_expr = mlir.oraSwitchExprOpCreateWithCases(
        mlir_ctx,
        loc,
        scrutinee,
        &[_]mlir.MlirType{struct_ty},
        1,
        2,
    );
    const case_values = [_]i64{1};
    const range_starts = [_]i64{0};
    const range_ends = [_]i64{0};
    const case_kinds = [_]i64{0};
    mlir.oraSwitchOpSetCasePatterns(
        switch_expr,
        &case_values,
        &range_starts,
        &range_ends,
        &case_kinds,
        1,
        case_values.len,
    );

    const case_block = mlir.oraSwitchExprOpGetCaseBlock(switch_expr, 0);
    const default_block = mlir.oraSwitchExprOpGetCaseBlock(switch_expr, 1);
    const field_name_attrs = [_]mlir.MlirAttribute{
        mlir.oraStringAttrCreate(mlir_ctx, stringRef("left")),
        mlir.oraStringAttrCreate(mlir_ctx, stringRef("right")),
    };

    const case_init = mlir.oraStructInitOpCreate(mlir_ctx, loc, &[_]mlir.MlirValue{ one, two }, 2, struct_ty);
    mlir.oraOperationSetAttributeByName(case_init, stringRef("ora.field_names"), mlir.oraArrayAttrCreate(mlir_ctx, field_name_attrs.len, &field_name_attrs));
    mlir.oraBlockAppendOwnedOperation(case_block, case_init);
    mlir.oraBlockAppendOwnedOperation(case_block, mlir.oraYieldOpCreate(mlir_ctx, loc, &[_]mlir.MlirValue{mlir.oraOperationGetResult(case_init, 0)}, 1));

    const default_init = mlir.oraStructInitOpCreate(mlir_ctx, loc, &[_]mlir.MlirValue{ three, two }, 2, struct_ty);
    mlir.oraOperationSetAttributeByName(default_init, stringRef("ora.field_names"), mlir.oraArrayAttrCreate(mlir_ctx, field_name_attrs.len, &field_name_attrs));
    mlir.oraBlockAppendOwnedOperation(default_block, default_init);
    mlir.oraBlockAppendOwnedOperation(default_block, mlir.oraYieldOpCreate(mlir_ctx, loc, &[_]mlir.MlirValue{mlir.oraOperationGetResult(default_init, 0)}, 1));

    const pair = mlir.oraOperationGetResult(switch_expr, 0);
    const update_op = mlir.oraStructFieldUpdateOpCreate(mlir_ctx, loc, pair, stringRef("left"), seven);
    const updated_pair = mlir.oraOperationGetResult(update_op, 0);
    const extract_right = mlir.oraStructFieldExtractOpCreate(mlir_ctx, loc, updated_pair, stringRef("right"), i256_ty);
    const right_ast = try encoder.encodeOperation(extract_right);

    try testing.expect(!encoder.isDegraded());

    const constraints = try encoder.takeConstraints(testing.allocator);
    defer if (constraints.len > 0) testing.allocator.free(constraints);

    var solver = try Solver.init(&z3_ctx, testing.allocator);
    defer solver.deinit();
    for (constraints) |cst| solver.assert(cst);
    solver.assert(z3.Z3_mk_not(z3_ctx.ctx, z3.Z3_mk_eq(z3_ctx.ctx, right_ast, try encoder.encodeValue(two))));
    try testing.expectEqual(@as(z3.Z3_lbool, z3.Z3_L_FALSE), solver.check());
}

test "struct_field_update recovers untouched fields through scf.execute_region source metadata" {
    var z3_ctx = try Context.init(testing.allocator);
    defer z3_ctx.deinit();

    var encoder = Encoder.init(&z3_ctx, testing.allocator);
    defer encoder.deinit();

    const mlir_ctx = mlir.oraContextCreate();
    defer mlir.oraContextDestroy(mlir_ctx);
    loadAllDialects(mlir_ctx);
    _ = mlir.oraDialectRegister(mlir_ctx);

    const loc = mlir.oraLocationUnknownGet(mlir_ctx);
    const i256_ty = mlir.oraIntegerTypeCreate(mlir_ctx, 256);
    const struct_ty = mlir.oraStructTypeGet(mlir_ctx, stringRef("Pair__u256"));

    const one = mlir.oraOperationGetResult(mlir.oraArithConstantOpCreate(
        mlir_ctx,
        loc,
        i256_ty,
        mlir.oraIntegerAttrCreateI64FromType(i256_ty, 1),
    ), 0);
    const two = mlir.oraOperationGetResult(mlir.oraArithConstantOpCreate(
        mlir_ctx,
        loc,
        i256_ty,
        mlir.oraIntegerAttrCreateI64FromType(i256_ty, 2),
    ), 0);
    const seven = mlir.oraOperationGetResult(mlir.oraArithConstantOpCreate(
        mlir_ctx,
        loc,
        i256_ty,
        mlir.oraIntegerAttrCreateI64FromType(i256_ty, 7),
    ), 0);

    const exec = mlir.oraScfExecuteRegionOpCreate(mlir_ctx, loc, &[_]mlir.MlirType{struct_ty}, 1, false);
    const exec_block = mlir.oraScfExecuteRegionOpGetBodyBlock(exec);
    const field_name_attrs = [_]mlir.MlirAttribute{
        mlir.oraStringAttrCreate(mlir_ctx, stringRef("left")),
        mlir.oraStringAttrCreate(mlir_ctx, stringRef("right")),
    };
    const init_op = mlir.oraStructInitOpCreate(mlir_ctx, loc, &[_]mlir.MlirValue{ one, two }, 2, struct_ty);
    mlir.oraOperationSetAttributeByName(init_op, stringRef("ora.field_names"), mlir.oraArrayAttrCreate(mlir_ctx, field_name_attrs.len, &field_name_attrs));
    mlir.oraBlockAppendOwnedOperation(exec_block, init_op);
    mlir.oraBlockAppendOwnedOperation(exec_block, mlir.oraScfYieldOpCreate(
        mlir_ctx,
        loc,
        &[_]mlir.MlirValue{mlir.oraOperationGetResult(init_op, 0)},
        1,
    ));

    const pair = mlir.oraOperationGetResult(exec, 0);
    const update_op = mlir.oraStructFieldUpdateOpCreate(mlir_ctx, loc, pair, stringRef("left"), seven);
    const updated_pair = mlir.oraOperationGetResult(update_op, 0);
    const extract_right = mlir.oraStructFieldExtractOpCreate(mlir_ctx, loc, updated_pair, stringRef("right"), i256_ty);
    const right_ast = try encoder.encodeOperation(extract_right);

    try testing.expect(!encoder.isDegraded());

    const constraints = try encoder.takeConstraints(testing.allocator);
    defer if (constraints.len > 0) testing.allocator.free(constraints);

    var solver = try Solver.init(&z3_ctx, testing.allocator);
    defer solver.deinit();
    for (constraints) |cst| solver.assert(cst);
    solver.assert(z3.Z3_mk_not(z3_ctx.ctx, z3.Z3_mk_eq(z3_ctx.ctx, right_ast, try encoder.encodeValue(two))));
    try testing.expectEqual(@as(z3.Z3_lbool, z3.Z3_L_FALSE), solver.check());
}

test "chained source-metadata struct updates preserve untouched fields exactly" {
    var z3_ctx = try Context.init(testing.allocator);
    defer z3_ctx.deinit();

    var encoder = Encoder.init(&z3_ctx, testing.allocator);
    defer encoder.deinit();

    const mlir_ctx = mlir.oraContextCreate();
    defer mlir.oraContextDestroy(mlir_ctx);
    loadAllDialects(mlir_ctx);
    _ = mlir.oraDialectRegister(mlir_ctx);

    const loc = mlir.oraLocationUnknownGet(mlir_ctx);
    const i256_ty = mlir.oraIntegerTypeCreate(mlir_ctx, 256);
    const struct_ty = mlir.oraStructTypeGet(mlir_ctx, stringRef("Pair__u256"));

    const one = mlir.oraOperationGetResult(mlir.oraArithConstantOpCreate(
        mlir_ctx, loc, i256_ty, mlir.oraIntegerAttrCreateI64FromType(i256_ty, 1),
    ), 0);
    const two = mlir.oraOperationGetResult(mlir.oraArithConstantOpCreate(
        mlir_ctx, loc, i256_ty, mlir.oraIntegerAttrCreateI64FromType(i256_ty, 2),
    ), 0);
    const seven = mlir.oraOperationGetResult(mlir.oraArithConstantOpCreate(
        mlir_ctx, loc, i256_ty, mlir.oraIntegerAttrCreateI64FromType(i256_ty, 7),
    ), 0);
    const nine = mlir.oraOperationGetResult(mlir.oraArithConstantOpCreate(
        mlir_ctx, loc, i256_ty, mlir.oraIntegerAttrCreateI64FromType(i256_ty, 9),
    ), 0);

    const fields = [_]mlir.MlirValue{ one, two };
    const init_op = mlir.oraStructInitOpCreate(mlir_ctx, loc, &fields, fields.len, struct_ty);
    const field_name_attrs = [_]mlir.MlirAttribute{
        mlir.oraStringAttrCreate(mlir_ctx, stringRef("left")),
        mlir.oraStringAttrCreate(mlir_ctx, stringRef("right")),
    };
    mlir.oraOperationSetAttributeByName(init_op, stringRef("ora.field_names"), mlir.oraArrayAttrCreate(mlir_ctx, field_name_attrs.len, &field_name_attrs));
    const pair = mlir.oraOperationGetResult(init_op, 0);
    const update_left = mlir.oraStructFieldUpdateOpCreate(mlir_ctx, loc, pair, stringRef("left"), seven);
    const update_right = mlir.oraStructFieldUpdateOpCreate(mlir_ctx, loc, mlir.oraOperationGetResult(update_left, 0), stringRef("right"), nine);

    const extract_left = mlir.oraStructFieldExtractOpCreate(mlir_ctx, loc, mlir.oraOperationGetResult(update_right, 0), stringRef("left"), i256_ty);
    const left_ast = try encoder.encodeOperation(extract_left);
    try testing.expect(!encoder.isDegraded());

    const constraints = try encoder.takeConstraints(testing.allocator);
    defer if (constraints.len > 0) testing.allocator.free(constraints);

    var solver = try Solver.init(&z3_ctx, testing.allocator);
    defer solver.deinit();
    for (constraints) |cst| solver.assert(cst);
    solver.assert(z3.Z3_mk_not(z3_ctx.ctx, z3.Z3_mk_eq(z3_ctx.ctx, left_ast, try encoder.encodeValue(seven))));
    try testing.expectEqual(@as(z3.Z3_lbool, z3.Z3_L_FALSE), solver.check());
}

test "known pure callee chained source-metadata struct updates preserve untouched fields exactly" {
    var z3_ctx = try Context.init(testing.allocator);
    defer z3_ctx.deinit();

    var encoder = Encoder.init(&z3_ctx, testing.allocator);
    defer encoder.deinit();

    const mlir_ctx = mlir.oraContextCreate();
    defer mlir.oraContextDestroy(mlir_ctx);
    loadAllDialects(mlir_ctx);
    _ = mlir.oraDialectRegister(mlir_ctx);

    const loc = mlir.oraLocationUnknownGet(mlir_ctx);
    const i256_ty = mlir.oraIntegerTypeCreate(mlir_ctx, 256);
    const struct_ty = mlir.oraStructTypeGet(mlir_ctx, stringRef("Pair__u256"));

    const helper_attrs = [_]mlir.MlirNamedAttribute{
        namedAttr(mlir_ctx, "sym_name", mlir.oraStringAttrCreate(mlir_ctx, stringRef("makePairViaChainedInit"))),
    };
    const helper = mlir.oraFuncFuncOpCreate(mlir_ctx, loc, &helper_attrs, helper_attrs.len, &[_]mlir.MlirType{}, &[_]mlir.MlirLocation{}, 0);
    const helper_body = mlir.oraFuncOpGetBodyBlock(helper);

    const one = mlir.oraOperationGetResult(mlir.oraArithConstantOpCreate(
        mlir_ctx, loc, i256_ty, mlir.oraIntegerAttrCreateI64FromType(i256_ty, 1),
    ), 0);
    const two = mlir.oraOperationGetResult(mlir.oraArithConstantOpCreate(
        mlir_ctx, loc, i256_ty, mlir.oraIntegerAttrCreateI64FromType(i256_ty, 2),
    ), 0);
    const seven = mlir.oraOperationGetResult(mlir.oraArithConstantOpCreate(
        mlir_ctx, loc, i256_ty, mlir.oraIntegerAttrCreateI64FromType(i256_ty, 7),
    ), 0);
    const nine = mlir.oraOperationGetResult(mlir.oraArithConstantOpCreate(
        mlir_ctx, loc, i256_ty, mlir.oraIntegerAttrCreateI64FromType(i256_ty, 9),
    ), 0);
    const fields = [_]mlir.MlirValue{ one, two };
    const init_op = mlir.oraStructInitOpCreate(mlir_ctx, loc, &fields, fields.len, struct_ty);
    const field_name_attrs = [_]mlir.MlirAttribute{
        mlir.oraStringAttrCreate(mlir_ctx, stringRef("left")),
        mlir.oraStringAttrCreate(mlir_ctx, stringRef("right")),
    };
    mlir.oraOperationSetAttributeByName(init_op, stringRef("ora.field_names"), mlir.oraArrayAttrCreate(mlir_ctx, field_name_attrs.len, &field_name_attrs));
    const pair = mlir.oraOperationGetResult(init_op, 0);
    const update_left = mlir.oraStructFieldUpdateOpCreate(mlir_ctx, loc, pair, stringRef("left"), seven);
    const update_right = mlir.oraStructFieldUpdateOpCreate(mlir_ctx, loc, mlir.oraOperationGetResult(update_left, 0), stringRef("right"), nine);
    mlir.oraBlockAppendOwnedOperation(helper_body, init_op);
    mlir.oraBlockAppendOwnedOperation(helper_body, update_left);
    mlir.oraBlockAppendOwnedOperation(helper_body, update_right);
    mlir.oraBlockAppendOwnedOperation(helper_body, mlir.oraReturnOpCreate(mlir_ctx, loc, &[_]mlir.MlirValue{mlir.oraOperationGetResult(update_right, 0)}, 1));
    try encoder.registerFunctionOperation(helper);

    const call = mlir.oraFuncCallOpCreate(mlir_ctx, loc, stringRef("makePairViaChainedInit"), &[_]mlir.MlirValue{}, 0, &[_]mlir.MlirType{struct_ty}, 1);
    const extract_left = mlir.oraStructFieldExtractOpCreate(mlir_ctx, loc, mlir.oraOperationGetResult(call, 0), stringRef("left"), i256_ty);
    const left_ast = try encoder.encodeOperation(extract_left);
    try testing.expect(!encoder.isDegraded());

    const constraints = try encoder.takeConstraints(testing.allocator);
    defer if (constraints.len > 0) testing.allocator.free(constraints);

    var solver = try Solver.init(&z3_ctx, testing.allocator);
    defer solver.deinit();
    for (constraints) |cst| solver.assert(cst);
    solver.assert(z3.Z3_mk_not(z3_ctx.ctx, z3.Z3_mk_eq(z3_ctx.ctx, left_ast, try encoder.encodeValue(seven))));
    try testing.expectEqual(@as(z3.Z3_lbool, z3.Z3_L_FALSE), solver.check());
}

test "memref store threads into later scalar load" {
    var z3_ctx = try Context.init(testing.allocator);
    defer z3_ctx.deinit();

    var encoder = Encoder.init(&z3_ctx, testing.allocator);
    defer encoder.deinit();

    const mlir_ctx = mlir.oraContextCreate();
    defer mlir.oraContextDestroy(mlir_ctx);

    loadAllDialects(mlir_ctx);
    _ = mlir.oraDialectRegister(mlir_ctx);

    const loc = mlir.oraLocationUnknownGet(mlir_ctx);
    const i256_ty = mlir.oraIntegerTypeCreate(mlir_ctx, 256);
    const null_attr = mlir.MlirAttribute{ .ptr = null };
    const memref_ty = mlir.oraMemRefTypeCreate(mlir_ctx, i256_ty, 0, null, null_attr, null_attr);

    const alloca = mlir.oraMemrefAllocaOpCreate(mlir_ctx, loc, memref_ty);
    const slot = mlir.oraOperationGetResult(alloca, 0);
    _ = try encoder.encodeOperation(alloca);

    const value_attr = mlir.oraIntegerAttrCreateI64FromType(i256_ty, 42);
    const value_op = mlir.oraArithConstantOpCreate(mlir_ctx, loc, i256_ty, value_attr);
    const value = mlir.oraOperationGetResult(value_op, 0);

    const store = mlir.oraMemrefStoreOpCreate(mlir_ctx, loc, value, slot, null, 0);
    _ = try encoder.encodeOperation(store);

    const load = mlir.oraMemrefLoadOpCreate(mlir_ctx, loc, slot, null, 0, i256_ty);
    const loaded = try encoder.encodeOperation(load);
    const expected = try encoder.encodeValue(value);

    var solver = try Solver.init(&z3_ctx, testing.allocator);
    defer solver.deinit();
    solver.assert(z3.Z3_mk_not(z3_ctx.ctx, z3.Z3_mk_eq(z3_ctx.ctx, loaded, expected)));
    try testing.expectEqual(@as(z3.Z3_lbool, z3.Z3_L_FALSE), solver.check());
}

test "uninitialized memref load degrades encoding" {
    var z3_ctx = try Context.init(testing.allocator);
    defer z3_ctx.deinit();

    var encoder = Encoder.init(&z3_ctx, testing.allocator);
    defer encoder.deinit();

    const mlir_ctx = mlir.oraContextCreate();
    defer mlir.oraContextDestroy(mlir_ctx);

    loadAllDialects(mlir_ctx);
    _ = mlir.oraDialectRegister(mlir_ctx);

    const loc = mlir.oraLocationUnknownGet(mlir_ctx);
    const i256_ty = mlir.oraIntegerTypeCreate(mlir_ctx, 256);
    const null_attr = mlir.MlirAttribute{ .ptr = null };
    const memref_ty = mlir.oraMemRefTypeCreate(mlir_ctx, i256_ty, 0, null, null_attr, null_attr);

    const alloca = mlir.oraMemrefAllocaOpCreate(mlir_ctx, loc, memref_ty);
    const slot = mlir.oraOperationGetResult(alloca, 0);
    _ = try encoder.encodeOperation(alloca);
    try testing.expect(!encoder.isDegraded());

    const load = mlir.oraMemrefLoadOpCreate(mlir_ctx, loc, slot, null, 0, i256_ty);
    _ = try encoder.encodeOperation(load);

    try testing.expect(encoder.isDegraded());
    try testing.expect(std.mem.startsWith(u8, encoder.degradationReason().?, "memref.load read from uninitialized tracked local state"));
}

test "scalar memref load recovers dominating store before scf.while" {
    var z3_ctx = try Context.init(testing.allocator);
    defer z3_ctx.deinit();

    var encoder = Encoder.init(&z3_ctx, testing.allocator);
    defer encoder.deinit();

    const mlir_ctx = mlir.oraContextCreate();
    defer mlir.oraContextDestroy(mlir_ctx);

    loadAllDialects(mlir_ctx);
    _ = mlir.oraDialectRegister(mlir_ctx);

    const loc = mlir.oraLocationUnknownGet(mlir_ctx);
    const i1_ty = mlir.oraIntegerTypeCreate(mlir_ctx, 1);
    const null_attr = mlir.MlirAttribute{ .ptr = null };
    const memref_ty = mlir.oraMemRefTypeCreate(mlir_ctx, i1_ty, 0, null, null_attr, null_attr);

    const empty_attrs = [_]mlir.MlirNamedAttribute{};
    const func = mlir.oraFuncFuncOpCreate(mlir_ctx, loc, &empty_attrs, empty_attrs.len, &[_]mlir.MlirType{}, &[_]mlir.MlirLocation{}, 0);
    const body = mlir.oraFuncOpGetBodyBlock(func);

    const alloca = mlir.oraMemrefAllocaOpCreate(mlir_ctx, loc, memref_ty);
    const slot = mlir.oraOperationGetResult(alloca, 0);
    mlir.oraBlockAppendOwnedOperation(body, alloca);

    const false_attr = mlir.oraIntegerAttrCreateI64FromType(i1_ty, 0);
    const false_op = mlir.oraArithConstantOpCreate(mlir_ctx, loc, i1_ty, false_attr);
    const false_val = mlir.oraOperationGetResult(false_op, 0);
    mlir.oraBlockAppendOwnedOperation(body, false_op);

    const init_store = mlir.oraMemrefStoreOpCreate(mlir_ctx, loc, false_val, slot, null, 0);
    mlir.oraBlockAppendOwnedOperation(body, init_store);

    const init_vals = [_]mlir.MlirValue{};
    const result_types = [_]mlir.MlirType{};
    const while_op = mlir.oraScfWhileOpCreate(mlir_ctx, loc, &init_vals, init_vals.len, &result_types, result_types.len);
    const before_block = mlir.oraScfWhileOpGetBeforeBlock(while_op);
    const load_op = mlir.oraMemrefLoadOpCreate(mlir_ctx, loc, slot, null, 0, i1_ty);
    const loaded = mlir.oraOperationGetResult(load_op, 0);
    mlir.oraBlockAppendOwnedOperation(before_block, load_op);
    mlir.oraBlockAppendOwnedOperation(before_block, mlir.oraScfConditionOpCreate(
        mlir_ctx,
        loc,
        loaded,
        &[_]mlir.MlirValue{},
        0,
    ));
    const after_block = mlir.oraScfWhileOpGetAfterBlock(while_op);
    mlir.oraBlockAppendOwnedOperation(after_block, mlir.oraScfYieldOpCreate(mlir_ctx, loc, &[_]mlir.MlirValue{}, 0));
    mlir.oraBlockAppendOwnedOperation(body, while_op);

    _ = try encoder.encodeOperation(alloca);
    _ = try encoder.encodeOperation(false_op);
    _ = try encoder.encodeOperation(init_store);

    encoder.memref_map.clearRetainingCapacity();
    const recovered = try encoder.encodeOperation(load_op);

    try testing.expect(!encoder.isDegraded());
    const expected_false = try encoder.encodeValue(false_val);

    var solver = try Solver.init(&z3_ctx, testing.allocator);
    defer solver.deinit();
    solver.assert(z3.Z3_mk_not(z3_ctx.ctx, z3.Z3_mk_eq(z3_ctx.ctx, recovered, expected_false)));
    try testing.expectEqual(@as(z3.Z3_lbool, z3.Z3_L_FALSE), solver.check());
}

test "indexed memref store threads into later indexed load" {
    var z3_ctx = try Context.init(testing.allocator);
    defer z3_ctx.deinit();

    var encoder = Encoder.init(&z3_ctx, testing.allocator);
    defer encoder.deinit();

    const mlir_ctx = mlir.oraContextCreate();
    defer mlir.oraContextDestroy(mlir_ctx);

    loadAllDialects(mlir_ctx);
    _ = mlir.oraDialectRegister(mlir_ctx);

    const loc = mlir.oraLocationUnknownGet(mlir_ctx);
    const i256_ty = mlir.oraIntegerTypeCreate(mlir_ctx, 256);
    const index_ty = mlir.oraIndexTypeCreate(mlir_ctx);
    const null_attr = mlir.MlirAttribute{ .ptr = null };
    const shape: [1]i64 = .{5};
    const memref_ty = mlir.oraMemRefTypeCreate(mlir_ctx, i256_ty, 1, &shape, null_attr, null_attr);

    const alloca = mlir.oraMemrefAllocaOpCreate(mlir_ctx, loc, memref_ty);
    const slot = mlir.oraOperationGetResult(alloca, 0);
    _ = try encoder.encodeOperation(alloca);

    const idx_attr = mlir.oraIntegerAttrCreateI64FromType(index_ty, 2);
    const idx_op = mlir.oraArithConstantOpCreate(mlir_ctx, loc, index_ty, idx_attr);
    const idx = mlir.oraOperationGetResult(idx_op, 0);
    _ = try encoder.encodeOperation(idx_op);

    const value_attr = mlir.oraIntegerAttrCreateI64FromType(i256_ty, 42);
    const value_op = mlir.oraArithConstantOpCreate(mlir_ctx, loc, i256_ty, value_attr);
    const value = mlir.oraOperationGetResult(value_op, 0);
    _ = try encoder.encodeOperation(value_op);

    const store = mlir.oraMemrefStoreOpCreate(mlir_ctx, loc, value, slot, &[_]mlir.MlirValue{idx}, 1);
    _ = try encoder.encodeOperation(store);

    const load = mlir.oraMemrefLoadOpCreate(mlir_ctx, loc, slot, &[_]mlir.MlirValue{idx}, 1, i256_ty);
    const loaded = try encoder.encodeOperation(load);
    const expected = try encoder.encodeValue(value);

    try testing.expect(!encoder.isDegraded());
    var solver = try Solver.init(&z3_ctx, testing.allocator);
    defer solver.deinit();
    solver.assert(z3.Z3_mk_not(z3_ctx.ctx, z3.Z3_mk_eq(z3_ctx.ctx, loaded, expected)));
    try testing.expectEqual(@as(z3.Z3_lbool, z3.Z3_L_FALSE), solver.check());
}

test "tload encodes exact transient slot value" {
    var z3_ctx = try Context.init(testing.allocator);
    defer z3_ctx.deinit();

    var encoder = Encoder.init(&z3_ctx, testing.allocator);
    defer encoder.deinit();

    const mlir_ctx = mlir.oraContextCreate();
    defer mlir.oraContextDestroy(mlir_ctx);

    loadAllDialects(mlir_ctx);
    _ = mlir.oraDialectRegister(mlir_ctx);

    const loc = mlir.oraLocationUnknownGet(mlir_ctx);
    const i256_ty = mlir.oraIntegerTypeCreate(mlir_ctx, 256);
    const tload = mlir.oraTLoadOpCreate(mlir_ctx, loc, stringRef("pending"), i256_ty);

    const loaded = try encoder.encodeOperation(tload);
    const expected = encoder.global_map.get("transient:pending").?;

    try testing.expect(!encoder.isDegraded());
    try testing.expectEqualStrings(
        std.mem.span(z3.Z3_ast_to_string(z3_ctx.ctx, expected)),
        std.mem.span(z3.Z3_ast_to_string(z3_ctx.ctx, loaded)),
    );
}

test "tensor.dim encodes dynamic shape dims consistently and folds static dims" {
    var z3_ctx = try Context.init(testing.allocator);
    defer z3_ctx.deinit();

    var encoder = Encoder.init(&z3_ctx, testing.allocator);
    defer encoder.deinit();

    const mlir_ctx = mlir.oraContextCreate();
    defer mlir.oraContextDestroy(mlir_ctx);
    loadAllDialects(mlir_ctx);
    _ = mlir.oraDialectRegister(mlir_ctx);

    const loc = mlir.oraLocationUnknownGet(mlir_ctx);
    const index_ty = mlir.oraIndexTypeCreate(mlir_ctx);
    const addr_ty = mlir.oraAddressTypeGet(mlir_ctx);
    const null_attr = mlir.MlirAttribute{ .ptr = null };

    const c0_attr = mlir.oraIntegerAttrCreateI64FromType(index_ty, 0);
    const c0_op = mlir.oraArithConstantOpCreate(mlir_ctx, loc, index_ty, c0_attr);
    const c0 = mlir.oraOperationGetResult(c0_op, 0);
    const empty_attrs = [_]mlir.MlirNamedAttribute{};

    const dyn_shape: [1]i64 = .{mlir.oraShapedTypeDynamicSize()};
    const dyn_tensor_ty = mlir.oraRankedTensorTypeCreate(mlir_ctx, 1, &dyn_shape, addr_ty, null_attr);
    const dyn_param_types = [_]mlir.MlirType{dyn_tensor_ty};
    const dyn_param_locs = [_]mlir.MlirLocation{loc};
    const dyn_func = mlir.oraFuncFuncOpCreate(mlir_ctx, loc, &empty_attrs, empty_attrs.len, &dyn_param_types, &dyn_param_locs, dyn_param_types.len);
    const dyn_body = mlir.oraFuncOpGetBodyBlock(dyn_func);
    const dyn_arg = mlir.oraBlockGetArgument(dyn_body, 0);

    const dyn_dim_a_op = mlir.oraTensorDimOpCreate(mlir_ctx, loc, dyn_arg, c0);
    const dyn_dim_b_op = mlir.oraTensorDimOpCreate(mlir_ctx, loc, dyn_arg, c0);
    const dyn_dim_a = try encoder.encodeOperation(dyn_dim_a_op);
    const dyn_dim_b = try encoder.encodeOperation(dyn_dim_b_op);

    var solver_dyn = try Solver.init(&z3_ctx, testing.allocator);
    defer solver_dyn.deinit();
    solver_dyn.assert(z3.Z3_mk_not(z3_ctx.ctx, z3.Z3_mk_eq(z3_ctx.ctx, dyn_dim_a, dyn_dim_b)));
    try testing.expectEqual(@as(z3.Z3_lbool, z3.Z3_L_FALSE), solver_dyn.check());

    const static_shape: [1]i64 = .{7};
    const static_tensor_ty = mlir.oraRankedTensorTypeCreate(mlir_ctx, 1, &static_shape, addr_ty, null_attr);
    const static_param_types = [_]mlir.MlirType{static_tensor_ty};
    const static_param_locs = [_]mlir.MlirLocation{loc};
    const static_func = mlir.oraFuncFuncOpCreate(mlir_ctx, loc, &empty_attrs, empty_attrs.len, &static_param_types, &static_param_locs, static_param_types.len);
    const static_body = mlir.oraFuncOpGetBodyBlock(static_func);
    const static_arg = mlir.oraBlockGetArgument(static_body, 0);

    const static_dim_op = mlir.oraTensorDimOpCreate(mlir_ctx, loc, static_arg, c0);
    const static_dim = try encoder.encodeOperation(static_dim_op);
    const static_sort = z3.Z3_get_sort(z3_ctx.ctx, static_dim);
    const expected = z3.Z3_mk_unsigned_int64(z3_ctx.ctx, 7, static_sort);

    var solver_static = try Solver.init(&z3_ctx, testing.allocator);
    defer solver_static.deinit();
    solver_static.assert(z3.Z3_mk_not(z3_ctx.ctx, z3.Z3_mk_eq(z3_ctx.ctx, static_dim, expected)));
    try testing.expectEqual(@as(z3.Z3_lbool, z3.Z3_L_FALSE), solver_static.check());
}

test "tensor.insert followed by tensor.extract returns inserted value" {
    var z3_ctx = try Context.init(testing.allocator);
    defer z3_ctx.deinit();

    var encoder = Encoder.init(&z3_ctx, testing.allocator);
    defer encoder.deinit();

    const mlir_ctx = mlir.oraContextCreate();
    defer mlir.oraContextDestroy(mlir_ctx);
    loadAllDialects(mlir_ctx);
    _ = mlir.oraDialectRegister(mlir_ctx);

    const loc = mlir.oraLocationUnknownGet(mlir_ctx);
    const index_ty = mlir.oraIndexTypeCreate(mlir_ctx);
    const i256_ty = mlir.oraIntegerTypeCreate(mlir_ctx, 256);
    const null_attr = mlir.MlirAttribute{ .ptr = null };
    const shape: [1]i64 = .{4};
    const tensor_ty = mlir.oraRankedTensorTypeCreate(mlir_ctx, 1, &shape, i256_ty, null_attr);

    const empty_attrs = [_]mlir.MlirNamedAttribute{};
    const param_types = [_]mlir.MlirType{tensor_ty};
    const param_locs = [_]mlir.MlirLocation{loc};
    const func = mlir.oraFuncFuncOpCreate(mlir_ctx, loc, &empty_attrs, empty_attrs.len, &param_types, &param_locs, param_types.len);
    const body = mlir.oraFuncOpGetBodyBlock(func);
    const tensor_arg = mlir.oraBlockGetArgument(body, 0);

    const c0_attr = mlir.oraIntegerAttrCreateI64FromType(index_ty, 0);
    const c0_op = mlir.oraArithConstantOpCreate(mlir_ctx, loc, index_ty, c0_attr);
    const c0 = mlir.oraOperationGetResult(c0_op, 0);

    const value_attr = mlir.oraIntegerAttrCreateI64FromType(i256_ty, 42);
    const value_op = mlir.oraArithConstantOpCreate(mlir_ctx, loc, i256_ty, value_attr);
    const value = mlir.oraOperationGetResult(value_op, 0);

    const insert_operands = [_]mlir.MlirValue{ value, tensor_arg, c0 };
    const insert_results = [_]mlir.MlirType{tensor_ty};
    const insert_op = mlir.oraOperationCreate(
        mlir_ctx,
        loc,
        stringRef("tensor.insert"),
        &insert_operands,
        insert_operands.len,
        &insert_results,
        insert_results.len,
        &empty_attrs,
        empty_attrs.len,
        0,
        false,
    );
    const inserted_tensor = mlir.oraOperationGetResult(insert_op, 0);

    const extract_indices = [_]mlir.MlirValue{c0};
    const extract_op = mlir.oraTensorExtractOpCreate(mlir_ctx, loc, inserted_tensor, &extract_indices, extract_indices.len, i256_ty);
    const extracted = try encoder.encodeOperation(extract_op);
    const expected = try encoder.encodeValue(value);

    var solver = try Solver.init(&z3_ctx, testing.allocator);
    defer solver.deinit();
    solver.assert(z3.Z3_mk_not(z3_ctx.ctx, z3.Z3_mk_eq(z3_ctx.ctx, extracted, expected)));
    try testing.expectEqual(@as(z3.Z3_lbool, z3.Z3_L_FALSE), solver.check());
}

test "old global is linked to entry current state" {
    var z3_ctx = try Context.init(testing.allocator);
    defer z3_ctx.deinit();

    var encoder = Encoder.init(&z3_ctx, testing.allocator);
    defer encoder.deinit();

    const mlir_ctx = mlir.oraContextCreate();
    defer mlir.oraContextDestroy(mlir_ctx);
    loadAllDialects(mlir_ctx);
    _ = mlir.oraDialectRegister(mlir_ctx);

    const loc = mlir.oraLocationUnknownGet(mlir_ctx);
    const i256_ty = mlir.oraIntegerTypeCreate(mlir_ctx, 256);

    const sload = mlir.oraSLoadOpCreate(mlir_ctx, loc, stringRef("total"), i256_ty);
    const old_op = mlir.oraOldOpCreate(mlir_ctx, loc, mlir.oraOperationGetResult(sload, 0), i256_ty);

    // Encode old(...) first to exercise old-before-current flow.
    const old_value = try encoder.encodeOperation(old_op);
    const current_value = try encoder.encodeOperation(sload);

    const constraints = try encoder.takeConstraints(testing.allocator);
    defer if (constraints.len > 0) testing.allocator.free(constraints);
    try testing.expect(constraints.len > 0);

    var solver = try Solver.init(&z3_ctx, testing.allocator);
    defer solver.deinit();
    for (constraints) |cst| solver.assert(cst);
    solver.assert(z3.Z3_mk_not(z3_ctx.ctx, z3.Z3_mk_eq(z3_ctx.ctx, old_value, current_value)));
    try testing.expectEqual(@as(z3.Z3_lbool, z3.Z3_L_FALSE), solver.check());
}

test "encodeValue errors on unsupported operation" {
    var z3_ctx = try Context.init(testing.allocator);
    defer z3_ctx.deinit();

    var encoder = Encoder.init(&z3_ctx, testing.allocator);
    defer encoder.deinit();

    const mlir_ctx = mlir.oraContextCreate();
    defer mlir.oraContextDestroy(mlir_ctx);

    loadAllDialects(mlir_ctx);
    _ = mlir.oraDialectRegister(mlir_ctx);

    const loc = mlir.oraLocationUnknownGet(mlir_ctx);
    const string_ty = mlir.oraStringTypeGet(mlir_ctx);
    const value_ref = mlir.oraStringRefCreate("hello".ptr, 5);
    const op = mlir.oraStringConstantOpCreate(mlir_ctx, loc, value_ref, string_ty);
    const result = mlir.oraOperationGetResult(op, 0);
    _ = try encoder.encodeValue(result);
}

test "ora.evm.caller shares symbol and is constrained non-zero" {
    var z3_ctx = try Context.init(testing.allocator);
    defer z3_ctx.deinit();

    var encoder = Encoder.init(&z3_ctx, testing.allocator);
    defer encoder.deinit();

    const mlir_ctx = mlir.oraContextCreate();
    defer mlir.oraContextDestroy(mlir_ctx);
    loadAllDialects(mlir_ctx);
    _ = mlir.oraDialectRegister(mlir_ctx);

    const loc = mlir.oraLocationUnknownGet(mlir_ctx);
    const addr_ty = mlir.oraAddressTypeGet(mlir_ctx);
    const caller_a_op = mlir.oraEvmOpCreate(mlir_ctx, loc, stringRef("ora.evm.caller"), null, 0, addr_ty);
    const caller_b_op = mlir.oraEvmOpCreate(mlir_ctx, loc, stringRef("ora.evm.caller"), null, 0, addr_ty);

    const caller_a = try encoder.encodeOperation(caller_a_op);
    const caller_b = try encoder.encodeOperation(caller_b_op);
    const constraints = try encoder.takeConstraints(testing.allocator);
    defer if (constraints.len > 0) testing.allocator.free(constraints);
    try testing.expect(constraints.len > 0);

    // Repeated caller reads within one function should resolve to the same symbol.
    var solver_same = try Solver.init(&z3_ctx, testing.allocator);
    defer solver_same.deinit();
    for (constraints) |cst| solver_same.assert(cst);
    const neq = z3.Z3_mk_not(z3_ctx.ctx, z3.Z3_mk_eq(z3_ctx.ctx, caller_a, caller_b));
    solver_same.assert(neq);
    try testing.expectEqual(@as(z3.Z3_lbool, z3.Z3_L_FALSE), solver_same.check());

    // Runtime semantics guarantee msg.sender/caller is non-zero.
    var solver_non_zero = try Solver.init(&z3_ctx, testing.allocator);
    defer solver_non_zero.deinit();
    for (constraints) |cst| solver_non_zero.assert(cst);
    const caller_sort = z3.Z3_get_sort(z3_ctx.ctx, caller_a);
    const zero = z3.Z3_mk_unsigned_int64(z3_ctx.ctx, 0, caller_sort);
    solver_non_zero.assert(z3.Z3_mk_eq(z3_ctx.ctx, caller_a, zero));
    try testing.expectEqual(@as(z3.Z3_lbool, z3.Z3_L_FALSE), solver_non_zero.check());
}

test "scf.for induction variable is constrained by loop bounds" {
    var z3_ctx = try Context.init(testing.allocator);
    defer z3_ctx.deinit();

    var encoder = Encoder.init(&z3_ctx, testing.allocator);
    defer encoder.deinit();

    const mlir_ctx = mlir.oraContextCreate();
    defer mlir.oraContextDestroy(mlir_ctx);
    loadAllDialects(mlir_ctx);
    _ = mlir.oraDialectRegister(mlir_ctx);

    const loc = mlir.oraLocationUnknownGet(mlir_ctx);
    const index_ty = mlir.oraIndexTypeCreate(mlir_ctx);

    const c0_attr = mlir.oraIntegerAttrCreateI64FromType(index_ty, 0);
    const c5_attr = mlir.oraIntegerAttrCreateI64FromType(index_ty, 5);
    const c1_attr = mlir.oraIntegerAttrCreateI64FromType(index_ty, 1);
    const c0_op = mlir.oraArithConstantOpCreate(mlir_ctx, loc, index_ty, c0_attr);
    const c5_op = mlir.oraArithConstantOpCreate(mlir_ctx, loc, index_ty, c5_attr);
    const c1_op = mlir.oraArithConstantOpCreate(mlir_ctx, loc, index_ty, c1_attr);

    const lb = mlir.oraOperationGetResult(c0_op, 0);
    const ub = mlir.oraOperationGetResult(c5_op, 0);
    const step = mlir.oraOperationGetResult(c1_op, 0);
    const empty_init_args = [_]mlir.MlirValue{};
    const for_op = mlir.oraScfForOpCreate(mlir_ctx, loc, lb, ub, step, &empty_init_args, empty_init_args.len, false);
    const body = mlir.oraScfForOpGetBodyBlock(for_op);
    const induction_var = mlir.oraBlockGetArgument(body, 0);

    const iv_ast = try encoder.encodeValue(induction_var);
    const lb_ast = try encoder.encodeValue(lb);
    const ub_ast = try encoder.encodeValue(ub);

    const constraints = try encoder.takeConstraints(testing.allocator);
    defer if (constraints.len > 0) testing.allocator.free(constraints);
    try testing.expect(constraints.len > 0);

    var solver = try Solver.init(&z3_ctx, testing.allocator);
    defer solver.deinit();
    for (constraints) |cst| solver.assert(cst);

    const below_lb = z3.Z3_mk_bvslt(z3_ctx.ctx, iv_ast, lb_ast);
    const at_or_above_ub = z3.Z3_mk_bvsge(z3_ctx.ctx, iv_ast, ub_ast);
    var range_violation = [_]z3.Z3_ast{ below_lb, at_or_above_ub };
    solver.assert(z3.Z3_mk_or(z3_ctx.ctx, range_violation.len, &range_violation));
    try testing.expectEqual(@as(z3.Z3_lbool, z3.Z3_L_FALSE), solver.check());
}

test "func.call summary with single-iteration scf.for state effects encodes exactly" {
    var z3_ctx = try Context.init(testing.allocator);
    defer z3_ctx.deinit();

    var encoder = Encoder.init(&z3_ctx, testing.allocator);
    defer encoder.deinit();

    const mlir_ctx = mlir.oraContextCreate();
    defer mlir.oraContextDestroy(mlir_ctx);
    loadAllDialects(mlir_ctx);
    _ = mlir.oraDialectRegister(mlir_ctx);

    const loc = mlir.oraLocationUnknownGet(mlir_ctx);
    const index_ty = mlir.oraIndexTypeCreate(mlir_ctx);
    const i256_ty = mlir.oraIntegerTypeCreate(mlir_ctx, 256);

    const helper_attrs = [_]mlir.MlirNamedAttribute{
        namedAttr(mlir_ctx, "sym_name", mlir.oraStringAttrCreate(mlir_ctx, stringRef("loopWriter"))),
        namedAttr(mlir_ctx, "ora.effect", mlir.oraStringAttrCreate(mlir_ctx, stringRef("writes"))),
        namedAttr(mlir_ctx, "ora.write_slots", mlir.oraArrayAttrCreate(mlir_ctx, 1, &[_]mlir.MlirAttribute{
            mlir.oraStringAttrCreate(mlir_ctx, stringRef("counter")),
        })),
    };
    const helper = mlir.oraFuncFuncOpCreate(mlir_ctx, loc, &helper_attrs, helper_attrs.len, &[_]mlir.MlirType{}, &[_]mlir.MlirLocation{}, 0);
    const body = mlir.oraFuncOpGetBodyBlock(helper);

    const c0_attr = mlir.oraIntegerAttrCreateI64FromType(index_ty, 0);
    const c1_attr = mlir.oraIntegerAttrCreateI64FromType(index_ty, 1);
    const c0_op = mlir.oraArithConstantOpCreate(mlir_ctx, loc, index_ty, c0_attr);
    const c1_op = mlir.oraArithConstantOpCreate(mlir_ctx, loc, index_ty, c1_attr);
    mlir.oraBlockAppendOwnedOperation(body, c0_op);
    mlir.oraBlockAppendOwnedOperation(body, c1_op);
    const lb = mlir.oraOperationGetResult(c0_op, 0);
    const ub = mlir.oraOperationGetResult(c1_op, 0);
    const step = mlir.oraOperationGetResult(c1_op, 0);
    const loop = mlir.oraScfForOpCreate(mlir_ctx, loc, lb, ub, step, &[_]mlir.MlirValue{}, 0, false);
    const loop_body = mlir.oraScfForOpGetBodyBlock(loop);
    const store_val_attr = mlir.oraIntegerAttrCreateI64FromType(i256_ty, 7);
    const store_val_op = mlir.oraArithConstantOpCreate(mlir_ctx, loc, i256_ty, store_val_attr);
    mlir.oraBlockAppendOwnedOperation(loop_body, store_val_op);
    mlir.oraBlockAppendOwnedOperation(loop_body, mlir.oraSStoreOpCreate(mlir_ctx, loc, mlir.oraOperationGetResult(store_val_op, 0), stringRef("counter")));
    mlir.oraBlockAppendOwnedOperation(loop_body, mlir.oraScfYieldOpCreate(mlir_ctx, loc, &[_]mlir.MlirValue{}, 0));
    mlir.oraBlockAppendOwnedOperation(body, loop);
    mlir.oraBlockAppendOwnedOperation(body, mlir.oraReturnOpCreate(mlir_ctx, loc, &[_]mlir.MlirValue{}, 0));

    try encoder.registerFunctionOperation(helper);

    const call = mlir.oraFuncCallOpCreate(mlir_ctx, loc, stringRef("loopWriter"), &[_]mlir.MlirValue{}, 0, &[_]mlir.MlirType{}, 0);
    _ = try encoder.encodeOperation(call);

    try testing.expect(!encoder.isDegraded());
    const counter = encoder.global_map.get("counter").?;
    const expected = z3.Z3_mk_numeral(z3_ctx.ctx, "7", z3.Z3_mk_bv_sort(z3_ctx.ctx, 256));

    var solver = try Solver.init(&z3_ctx, testing.allocator);
    defer solver.deinit();
    solver.assert(z3.Z3_mk_not(z3_ctx.ctx, z3.Z3_mk_eq(z3_ctx.ctx, counter, expected)));
    try testing.expectEqual(@as(z3.Z3_lbool, z3.Z3_L_FALSE), solver.check());
}

test "func.call summary with finite two-iteration scf.for state effects encodes exactly" {
    var z3_ctx = try Context.init(testing.allocator);
    defer z3_ctx.deinit();

    var encoder = Encoder.init(&z3_ctx, testing.allocator);
    defer encoder.deinit();

    const mlir_ctx = mlir.oraContextCreate();
    defer mlir.oraContextDestroy(mlir_ctx);
    loadAllDialects(mlir_ctx);
    _ = mlir.oraDialectRegister(mlir_ctx);

    const loc = mlir.oraLocationUnknownGet(mlir_ctx);
    const index_ty = mlir.oraIndexTypeCreate(mlir_ctx);
    const i256_ty = mlir.oraIntegerTypeCreate(mlir_ctx, 256);

    const helper_attrs = [_]mlir.MlirNamedAttribute{
        namedAttr(mlir_ctx, "sym_name", mlir.oraStringAttrCreate(mlir_ctx, stringRef("finiteForWriter"))),
        namedAttr(mlir_ctx, "ora.effect", mlir.oraStringAttrCreate(mlir_ctx, stringRef("writes"))),
        namedAttr(mlir_ctx, "ora.write_slots", mlir.oraArrayAttrCreate(mlir_ctx, 1, &[_]mlir.MlirAttribute{
            mlir.oraStringAttrCreate(mlir_ctx, stringRef("counter")),
        })),
    };
    const helper = mlir.oraFuncFuncOpCreate(mlir_ctx, loc, &helper_attrs, helper_attrs.len, &[_]mlir.MlirType{}, &[_]mlir.MlirLocation{}, 0);
    const body = mlir.oraFuncOpGetBodyBlock(helper);

    const c0_attr = mlir.oraIntegerAttrCreateI64FromType(index_ty, 0);
    const c2_attr = mlir.oraIntegerAttrCreateI64FromType(index_ty, 2);
    const c1_attr = mlir.oraIntegerAttrCreateI64FromType(index_ty, 1);
    const c0_op = mlir.oraArithConstantOpCreate(mlir_ctx, loc, index_ty, c0_attr);
    const c2_op = mlir.oraArithConstantOpCreate(mlir_ctx, loc, index_ty, c2_attr);
    const c1_op = mlir.oraArithConstantOpCreate(mlir_ctx, loc, index_ty, c1_attr);
    mlir.oraBlockAppendOwnedOperation(body, c0_op);
    mlir.oraBlockAppendOwnedOperation(body, c2_op);
    mlir.oraBlockAppendOwnedOperation(body, c1_op);

    const init_attr = mlir.oraIntegerAttrCreateI64FromType(i256_ty, 0);
    const init_op = mlir.oraArithConstantOpCreate(mlir_ctx, loc, i256_ty, init_attr);
    mlir.oraBlockAppendOwnedOperation(body, init_op);

    const loop = mlir.oraScfForOpCreate(
        mlir_ctx,
        loc,
        mlir.oraOperationGetResult(c0_op, 0),
        mlir.oraOperationGetResult(c2_op, 0),
        mlir.oraOperationGetResult(c1_op, 0),
        &[_]mlir.MlirValue{mlir.oraOperationGetResult(init_op, 0)},
        1,
        false,
    );
    const loop_body = mlir.oraScfForOpGetBodyBlock(loop);
    const iv = mlir.oraBlockGetArgument(loop_body, 0);
    const carried = mlir.oraBlockGetArgument(loop_body, 1);
    const iv_i256 = mlir.oraArithIndexCastUIOpCreate(mlir_ctx, loc, iv, i256_ty);
    mlir.oraBlockAppendOwnedOperation(loop_body, iv_i256);
    const next = mlir.oraArithAddIOpCreate(mlir_ctx, loc, carried, mlir.oraOperationGetResult(iv_i256, 0));
    mlir.oraBlockAppendOwnedOperation(loop_body, next);
    mlir.oraBlockAppendOwnedOperation(loop_body, mlir.oraSStoreOpCreate(mlir_ctx, loc, mlir.oraOperationGetResult(next, 0), stringRef("counter")));
    mlir.oraBlockAppendOwnedOperation(loop_body, mlir.oraScfYieldOpCreate(
        mlir_ctx,
        loc,
        &[_]mlir.MlirValue{mlir.oraOperationGetResult(next, 0)},
        1,
    ));
    mlir.oraBlockAppendOwnedOperation(body, loop);
    mlir.oraBlockAppendOwnedOperation(body, mlir.oraReturnOpCreate(mlir_ctx, loc, &[_]mlir.MlirValue{}, 0));

    try encoder.registerFunctionOperation(helper);

    const call = mlir.oraFuncCallOpCreate(mlir_ctx, loc, stringRef("finiteForWriter"), &[_]mlir.MlirValue{}, 0, &[_]mlir.MlirType{}, 0);
    _ = try encoder.encodeOperation(call);

    try testing.expect(!encoder.isDegraded());
    const counter = encoder.global_map.get("counter").?;
    const expected = try encoder.encodeIntegerConstant(1, 256);

    var solver = try Solver.init(&z3_ctx, testing.allocator);
    defer solver.deinit();
    solver.assert(z3.Z3_mk_not(z3_ctx.ctx, z3.Z3_mk_eq(z3_ctx.ctx, counter, expected)));
    try testing.expectEqual(@as(z3.Z3_lbool, z3.Z3_L_FALSE), solver.check());
}

test "func.call summary with zero-iteration scf.for preserves state exactly" {
    var z3_ctx = try Context.init(testing.allocator);
    defer z3_ctx.deinit();

    var encoder = Encoder.init(&z3_ctx, testing.allocator);
    defer encoder.deinit();

    const mlir_ctx = mlir.oraContextCreate();
    defer mlir.oraContextDestroy(mlir_ctx);
    loadAllDialects(mlir_ctx);
    _ = mlir.oraDialectRegister(mlir_ctx);

    const loc = mlir.oraLocationUnknownGet(mlir_ctx);
    const index_ty = mlir.oraIndexTypeCreate(mlir_ctx);
    const i256_ty = mlir.oraIntegerTypeCreate(mlir_ctx, 256);

    const helper_attrs = [_]mlir.MlirNamedAttribute{
        namedAttr(mlir_ctx, "sym_name", mlir.oraStringAttrCreate(mlir_ctx, stringRef("zeroIterForWriter"))),
        namedAttr(mlir_ctx, "ora.effect", mlir.oraStringAttrCreate(mlir_ctx, stringRef("writes"))),
        namedAttr(mlir_ctx, "ora.write_slots", mlir.oraArrayAttrCreate(mlir_ctx, 1, &[_]mlir.MlirAttribute{
            mlir.oraStringAttrCreate(mlir_ctx, stringRef("counter")),
        })),
    };
    const helper = mlir.oraFuncFuncOpCreate(mlir_ctx, loc, &helper_attrs, helper_attrs.len, &[_]mlir.MlirType{}, &[_]mlir.MlirLocation{}, 0);
    const body = mlir.oraFuncOpGetBodyBlock(helper);

    const c5_attr = mlir.oraIntegerAttrCreateI64FromType(index_ty, 5);
    const c1_attr = mlir.oraIntegerAttrCreateI64FromType(index_ty, 1);
    const c5_op = mlir.oraArithConstantOpCreate(mlir_ctx, loc, index_ty, c5_attr);
    const c1_op = mlir.oraArithConstantOpCreate(mlir_ctx, loc, index_ty, c1_attr);
    mlir.oraBlockAppendOwnedOperation(body, c5_op);
    mlir.oraBlockAppendOwnedOperation(body, c1_op);

    const loop = mlir.oraScfForOpCreate(
        mlir_ctx,
        loc,
        mlir.oraOperationGetResult(c5_op, 0),
        mlir.oraOperationGetResult(c5_op, 0),
        mlir.oraOperationGetResult(c1_op, 0),
        &[_]mlir.MlirValue{},
        0,
        false,
    );
    const loop_body = mlir.oraScfForOpGetBodyBlock(loop);
    const store_val_attr = mlir.oraIntegerAttrCreateI64FromType(i256_ty, 7);
    const store_val_op = mlir.oraArithConstantOpCreate(mlir_ctx, loc, i256_ty, store_val_attr);
    mlir.oraBlockAppendOwnedOperation(loop_body, store_val_op);
    mlir.oraBlockAppendOwnedOperation(loop_body, mlir.oraSStoreOpCreate(mlir_ctx, loc, mlir.oraOperationGetResult(store_val_op, 0), stringRef("counter")));
    mlir.oraBlockAppendOwnedOperation(loop_body, mlir.oraScfYieldOpCreate(mlir_ctx, loc, &[_]mlir.MlirValue{}, 0));
    mlir.oraBlockAppendOwnedOperation(body, loop);
    mlir.oraBlockAppendOwnedOperation(body, mlir.oraReturnOpCreate(mlir_ctx, loc, &[_]mlir.MlirValue{}, 0));

    try encoder.registerFunctionOperation(helper);

    const counter_sort = z3.Z3_mk_bv_sort(z3_ctx.ctx, 256);
    const pre_counter = z3.Z3_mk_const(z3_ctx.ctx, z3.Z3_mk_string_symbol(z3_ctx.ctx, "pre_counter_zero_for"), counter_sort);
    try encoder.global_map.put(try testing.allocator.dupe(u8, "counter"), pre_counter);

    const call = mlir.oraFuncCallOpCreate(mlir_ctx, loc, stringRef("zeroIterForWriter"), &[_]mlir.MlirValue{}, 0, &[_]mlir.MlirType{}, 0);
    _ = try encoder.encodeOperation(call);

    try testing.expect(!encoder.isDegraded());
    const post_counter = encoder.global_map.get("counter").?;

    var solver = try Solver.init(&z3_ctx, testing.allocator);
    defer solver.deinit();
    solver.assert(z3.Z3_mk_not(z3_ctx.ctx, z3.Z3_mk_eq(z3_ctx.ctx, post_counter, pre_counter)));
    try testing.expectEqual(@as(z3.Z3_lbool, z3.Z3_L_FALSE), solver.check());
}

test "func.call summary with symbolic no-write scf.for preserves state exactly" {
    var z3_ctx = try Context.init(testing.allocator);
    defer z3_ctx.deinit();

    var encoder = Encoder.init(&z3_ctx, testing.allocator);
    defer encoder.deinit();

    const mlir_ctx = mlir.oraContextCreate();
    defer mlir.oraContextDestroy(mlir_ctx);
    loadAllDialects(mlir_ctx);
    _ = mlir.oraDialectRegister(mlir_ctx);

    const loc = mlir.oraLocationUnknownGet(mlir_ctx);
    const index_ty = mlir.oraIndexTypeCreate(mlir_ctx);

    const helper_attrs = [_]mlir.MlirNamedAttribute{
        namedAttr(mlir_ctx, "sym_name", mlir.oraStringAttrCreate(mlir_ctx, stringRef("symbolicNoWriteFor"))),
    };
    const helper_param_types = [_]mlir.MlirType{index_ty};
    const helper_param_locs = [_]mlir.MlirLocation{loc};
    const helper = mlir.oraFuncFuncOpCreate(mlir_ctx, loc, &helper_attrs, helper_attrs.len, &helper_param_types, &helper_param_locs, helper_param_types.len);
    const body = mlir.oraFuncOpGetBodyBlock(helper);

    const c0_attr = mlir.oraIntegerAttrCreateI64FromType(index_ty, 0);
    const c1_attr = mlir.oraIntegerAttrCreateI64FromType(index_ty, 1);
    const c0_op = mlir.oraArithConstantOpCreate(mlir_ctx, loc, index_ty, c0_attr);
    const c1_op = mlir.oraArithConstantOpCreate(mlir_ctx, loc, index_ty, c1_attr);
    mlir.oraBlockAppendOwnedOperation(body, c0_op);
    mlir.oraBlockAppendOwnedOperation(body, c1_op);

    const ub = mlir.oraBlockGetArgument(body, 0);
    const loop = mlir.oraScfForOpCreate(
        mlir_ctx,
        loc,
        mlir.oraOperationGetResult(c0_op, 0),
        ub,
        mlir.oraOperationGetResult(c1_op, 0),
        &[_]mlir.MlirValue{},
        0,
        false,
    );
    const loop_body = mlir.oraScfForOpGetBodyBlock(loop);
    mlir.oraBlockAppendOwnedOperation(loop_body, mlir.oraScfYieldOpCreate(mlir_ctx, loc, &[_]mlir.MlirValue{}, 0));
    mlir.oraBlockAppendOwnedOperation(body, loop);
    mlir.oraBlockAppendOwnedOperation(body, mlir.oraReturnOpCreate(mlir_ctx, loc, &[_]mlir.MlirValue{}, 0));

    try encoder.registerFunctionOperation(helper);

    const counter_sort = z3.Z3_mk_bv_sort(z3_ctx.ctx, 256);
    const pre_counter = z3.Z3_mk_const(z3_ctx.ctx, z3.Z3_mk_string_symbol(z3_ctx.ctx, "pre_counter_symbolic_for"), counter_sort);
    try encoder.global_map.put(try testing.allocator.dupe(u8, "counter"), pre_counter);

    const symbolic_ub = mlir.oraVariablePlaceholderOpCreate(mlir_ctx, loc, stringRef("ubValue"), index_ty);
    const call = mlir.oraFuncCallOpCreate(
        mlir_ctx,
        loc,
        stringRef("symbolicNoWriteFor"),
        &[_]mlir.MlirValue{mlir.oraOperationGetResult(symbolic_ub, 0)},
        1,
        &[_]mlir.MlirType{},
        0,
    );
    _ = try encoder.encodeOperation(call);

    try testing.expect(!encoder.isDegraded());
    const post_counter = encoder.global_map.get("counter").?;

    var solver = try Solver.init(&z3_ctx, testing.allocator);
    defer solver.deinit();
    solver.assert(z3.Z3_mk_not(z3_ctx.ctx, z3.Z3_mk_eq(z3_ctx.ctx, post_counter, pre_counter)));
    try testing.expectEqual(@as(z3.Z3_lbool, z3.Z3_L_FALSE), solver.check());
}

test "func.call summary with canonical symbolic no-write scf.while preserves state exactly" {
    var z3_ctx = try Context.init(testing.allocator);
    defer z3_ctx.deinit();

    var encoder = Encoder.init(&z3_ctx, testing.allocator);
    defer encoder.deinit();

    const mlir_ctx = mlir.oraContextCreate();
    defer mlir.oraContextDestroy(mlir_ctx);
    loadAllDialects(mlir_ctx);
    _ = mlir.oraDialectRegister(mlir_ctx);

    const loc = mlir.oraLocationUnknownGet(mlir_ctx);
    const index_ty = mlir.oraIndexTypeCreate(mlir_ctx);

    const helper_attrs = [_]mlir.MlirNamedAttribute{
        namedAttr(mlir_ctx, "sym_name", mlir.oraStringAttrCreate(mlir_ctx, stringRef("symbolicNoWriteWhile"))),
    };
    const helper_param_types = [_]mlir.MlirType{index_ty};
    const helper_param_locs = [_]mlir.MlirLocation{loc};
    const helper = mlir.oraFuncFuncOpCreate(mlir_ctx, loc, &helper_attrs, helper_attrs.len, &helper_param_types, &helper_param_locs, helper_param_types.len);
    const body = mlir.oraFuncOpGetBodyBlock(helper);

    const c0_attr = mlir.oraIntegerAttrCreateI64FromType(index_ty, 0);
    const c1_attr = mlir.oraIntegerAttrCreateI64FromType(index_ty, 1);
    const c0_op = mlir.oraArithConstantOpCreate(mlir_ctx, loc, index_ty, c0_attr);
    const c1_op = mlir.oraArithConstantOpCreate(mlir_ctx, loc, index_ty, c1_attr);
    mlir.oraBlockAppendOwnedOperation(body, c0_op);
    mlir.oraBlockAppendOwnedOperation(body, c1_op);

    const ub = mlir.oraBlockGetArgument(body, 0);
    const init_vals = [_]mlir.MlirValue{mlir.oraOperationGetResult(c0_op, 0)};
    const result_types = [_]mlir.MlirType{index_ty};
    const while_op = mlir.oraScfWhileOpCreate(mlir_ctx, loc, &init_vals, init_vals.len, &result_types, result_types.len);
    const before_block = mlir.oraScfWhileOpGetBeforeBlock(while_op);
    const after_block = mlir.oraScfWhileOpGetAfterBlock(while_op);
    _ = mlir.mlirBlockAddArgument(before_block, index_ty, loc);
    _ = mlir.mlirBlockAddArgument(after_block, index_ty, loc);
    const before_arg = mlir.oraBlockGetArgument(before_block, 0);
    const after_arg = mlir.oraBlockGetArgument(after_block, 0);

    const cmp_op = mlir.oraArithCmpIOpCreate(mlir_ctx, loc, 6, before_arg, ub); // ult
    mlir.oraBlockAppendOwnedOperation(before_block, cmp_op);
    mlir.oraBlockAppendOwnedOperation(before_block, mlir.oraScfConditionOpCreate(
        mlir_ctx,
        loc,
        mlir.oraOperationGetResult(cmp_op, 0),
        &[_]mlir.MlirValue{before_arg},
        1,
    ));

    const next_op = mlir.oraArithAddIOpCreate(mlir_ctx, loc, after_arg, mlir.oraOperationGetResult(c1_op, 0));
    mlir.oraBlockAppendOwnedOperation(after_block, next_op);
    mlir.oraBlockAppendOwnedOperation(after_block, mlir.oraScfYieldOpCreate(
        mlir_ctx,
        loc,
        &[_]mlir.MlirValue{mlir.oraOperationGetResult(next_op, 0)},
        1,
    ));

    mlir.oraBlockAppendOwnedOperation(body, while_op);
    mlir.oraBlockAppendOwnedOperation(body, mlir.oraReturnOpCreate(mlir_ctx, loc, &[_]mlir.MlirValue{}, 0));

    try encoder.registerFunctionOperation(helper);

    const counter_sort = z3.Z3_mk_bv_sort(z3_ctx.ctx, 256);
    const pre_counter = z3.Z3_mk_const(z3_ctx.ctx, z3.Z3_mk_string_symbol(z3_ctx.ctx, "pre_counter_symbolic_while"), counter_sort);
    try encoder.global_map.put(try testing.allocator.dupe(u8, "counter"), pre_counter);

    const symbolic_ub = mlir.oraVariablePlaceholderOpCreate(mlir_ctx, loc, stringRef("ubWhileValue"), index_ty);
    const call = mlir.oraFuncCallOpCreate(
        mlir_ctx,
        loc,
        stringRef("symbolicNoWriteWhile"),
        &[_]mlir.MlirValue{mlir.oraOperationGetResult(symbolic_ub, 0)},
        1,
        &[_]mlir.MlirType{},
        0,
    );
    _ = try encoder.encodeOperation(call);

    try testing.expect(!encoder.isDegraded());
    const post_counter = encoder.global_map.get("counter").?;

    var solver = try Solver.init(&z3_ctx, testing.allocator);
    defer solver.deinit();
    solver.assert(z3.Z3_mk_not(z3_ctx.ctx, z3.Z3_mk_eq(z3_ctx.ctx, post_counter, pre_counter)));
    try testing.expectEqual(@as(z3.Z3_lbool, z3.Z3_L_FALSE), solver.check());
}

test "func.call summary with canonical symbolic decrement no-write scf.while preserves state exactly" {
    var z3_ctx = try Context.init(testing.allocator);
    defer z3_ctx.deinit();

    var encoder = Encoder.init(&z3_ctx, testing.allocator);
    defer encoder.deinit();

    const mlir_ctx = mlir.oraContextCreate();
    defer mlir.oraContextDestroy(mlir_ctx);
    loadAllDialects(mlir_ctx);
    _ = mlir.oraDialectRegister(mlir_ctx);

    const loc = mlir.oraLocationUnknownGet(mlir_ctx);
    const index_ty = mlir.oraIndexTypeCreate(mlir_ctx);

    const helper_attrs = [_]mlir.MlirNamedAttribute{
        namedAttr(mlir_ctx, "sym_name", mlir.oraStringAttrCreate(mlir_ctx, stringRef("symbolicNoWriteWhileDec"))),
    };
    const helper_param_types = [_]mlir.MlirType{index_ty};
    const helper_param_locs = [_]mlir.MlirLocation{loc};
    const helper = mlir.oraFuncFuncOpCreate(mlir_ctx, loc, &helper_attrs, helper_attrs.len, &helper_param_types, &helper_param_locs, helper_param_types.len);
    const body = mlir.oraFuncOpGetBodyBlock(helper);

    const c0_attr = mlir.oraIntegerAttrCreateI64FromType(index_ty, 0);
    const c1_attr = mlir.oraIntegerAttrCreateI64FromType(index_ty, 1);
    const c0_op = mlir.oraArithConstantOpCreate(mlir_ctx, loc, index_ty, c0_attr);
    const c1_op = mlir.oraArithConstantOpCreate(mlir_ctx, loc, index_ty, c1_attr);
    mlir.oraBlockAppendOwnedOperation(body, c0_op);
    mlir.oraBlockAppendOwnedOperation(body, c1_op);

    const init = mlir.oraBlockGetArgument(body, 0);
    const init_vals = [_]mlir.MlirValue{init};
    const result_types = [_]mlir.MlirType{index_ty};
    const while_op = mlir.oraScfWhileOpCreate(mlir_ctx, loc, &init_vals, init_vals.len, &result_types, result_types.len);
    const before_block = mlir.oraScfWhileOpGetBeforeBlock(while_op);
    const after_block = mlir.oraScfWhileOpGetAfterBlock(while_op);
    _ = mlir.mlirBlockAddArgument(before_block, index_ty, loc);
    _ = mlir.mlirBlockAddArgument(after_block, index_ty, loc);
    const before_arg = mlir.oraBlockGetArgument(before_block, 0);
    const after_arg = mlir.oraBlockGetArgument(after_block, 0);

    const cmp_op = mlir.oraArithCmpIOpCreate(mlir_ctx, loc, 8, before_arg, mlir.oraOperationGetResult(c0_op, 0)); // ugt
    mlir.oraBlockAppendOwnedOperation(before_block, cmp_op);
    mlir.oraBlockAppendOwnedOperation(before_block, mlir.oraScfConditionOpCreate(
        mlir_ctx,
        loc,
        mlir.oraOperationGetResult(cmp_op, 0),
        &[_]mlir.MlirValue{before_arg},
        1,
    ));

    const next_op = mlir.oraArithSubIOpCreate(mlir_ctx, loc, after_arg, mlir.oraOperationGetResult(c1_op, 0));
    mlir.oraBlockAppendOwnedOperation(after_block, next_op);
    mlir.oraBlockAppendOwnedOperation(after_block, mlir.oraScfYieldOpCreate(
        mlir_ctx,
        loc,
        &[_]mlir.MlirValue{mlir.oraOperationGetResult(next_op, 0)},
        1,
    ));

    mlir.oraBlockAppendOwnedOperation(body, while_op);
    mlir.oraBlockAppendOwnedOperation(body, mlir.oraReturnOpCreate(mlir_ctx, loc, &[_]mlir.MlirValue{}, 0));

    try encoder.registerFunctionOperation(helper);

    const counter_sort = z3.Z3_mk_bv_sort(z3_ctx.ctx, 256);
    const pre_counter = z3.Z3_mk_const(z3_ctx.ctx, z3.Z3_mk_string_symbol(z3_ctx.ctx, "pre_counter_symbolic_while_dec"), counter_sort);
    try encoder.global_map.put(try testing.allocator.dupe(u8, "counter"), pre_counter);

    const symbolic_init = mlir.oraVariablePlaceholderOpCreate(mlir_ctx, loc, stringRef("initWhileValue"), index_ty);
    const call = mlir.oraFuncCallOpCreate(
        mlir_ctx,
        loc,
        stringRef("symbolicNoWriteWhileDec"),
        &[_]mlir.MlirValue{mlir.oraOperationGetResult(symbolic_init, 0)},
        1,
        &[_]mlir.MlirType{},
        0,
    );
    _ = try encoder.encodeOperation(call);

    try testing.expect(!encoder.isDegraded());
    const post_counter = encoder.global_map.get("counter").?;

    var solver = try Solver.init(&z3_ctx, testing.allocator);
    defer solver.deinit();
    solver.assert(z3.Z3_mk_not(z3_ctx.ctx, z3.Z3_mk_eq(z3_ctx.ctx, post_counter, pre_counter)));
    try testing.expectEqual(@as(z3.Z3_lbool, z3.Z3_L_FALSE), solver.check());
}

test "func.call summary with canonical signed symbolic no-write scf.while preserves state exactly" {
    var z3_ctx = try Context.init(testing.allocator);
    defer z3_ctx.deinit();

    var encoder = Encoder.init(&z3_ctx, testing.allocator);
    defer encoder.deinit();

    const mlir_ctx = mlir.oraContextCreate();
    defer mlir.oraContextDestroy(mlir_ctx);
    loadAllDialects(mlir_ctx);
    _ = mlir.oraDialectRegister(mlir_ctx);

    const loc = mlir.oraLocationUnknownGet(mlir_ctx);
    const i256_ty = mlir.oraIntegerTypeCreate(mlir_ctx, 256);

    const helper_attrs = [_]mlir.MlirNamedAttribute{
        namedAttr(mlir_ctx, "sym_name", mlir.oraStringAttrCreate(mlir_ctx, stringRef("symbolicNoWriteSignedWhile"))),
    };
    const helper_param_types = [_]mlir.MlirType{i256_ty};
    const helper_param_locs = [_]mlir.MlirLocation{loc};
    const helper = mlir.oraFuncFuncOpCreate(mlir_ctx, loc, &helper_attrs, helper_attrs.len, &helper_param_types, &helper_param_locs, helper_param_types.len);
    const body = mlir.oraFuncOpGetBodyBlock(helper);

    const c0_attr = mlir.oraIntegerAttrCreateI64FromType(i256_ty, 0);
    const c1_attr = mlir.oraIntegerAttrCreateI64FromType(i256_ty, 1);
    const c0_op = mlir.oraArithConstantOpCreate(mlir_ctx, loc, i256_ty, c0_attr);
    const c1_op = mlir.oraArithConstantOpCreate(mlir_ctx, loc, i256_ty, c1_attr);
    mlir.oraBlockAppendOwnedOperation(body, c0_op);
    mlir.oraBlockAppendOwnedOperation(body, c1_op);

    const ub = mlir.oraBlockGetArgument(body, 0);
    const init_vals = [_]mlir.MlirValue{mlir.oraOperationGetResult(c0_op, 0)};
    const result_types = [_]mlir.MlirType{i256_ty};
    const while_op = mlir.oraScfWhileOpCreate(mlir_ctx, loc, &init_vals, init_vals.len, &result_types, result_types.len);
    const before_block = mlir.oraScfWhileOpGetBeforeBlock(while_op);
    const after_block = mlir.oraScfWhileOpGetAfterBlock(while_op);
    _ = mlir.mlirBlockAddArgument(before_block, i256_ty, loc);
    _ = mlir.mlirBlockAddArgument(after_block, i256_ty, loc);
    const before_arg = mlir.oraBlockGetArgument(before_block, 0);
    const after_arg = mlir.oraBlockGetArgument(after_block, 0);

    const cmp_op = mlir.oraArithCmpIOpCreate(mlir_ctx, loc, 2, before_arg, ub); // slt
    mlir.oraBlockAppendOwnedOperation(before_block, cmp_op);
    mlir.oraBlockAppendOwnedOperation(before_block, mlir.oraScfConditionOpCreate(
        mlir_ctx,
        loc,
        mlir.oraOperationGetResult(cmp_op, 0),
        &[_]mlir.MlirValue{before_arg},
        1,
    ));

    const next_op = mlir.oraArithAddIOpCreate(mlir_ctx, loc, after_arg, mlir.oraOperationGetResult(c1_op, 0));
    mlir.oraBlockAppendOwnedOperation(after_block, next_op);
    mlir.oraBlockAppendOwnedOperation(after_block, mlir.oraScfYieldOpCreate(
        mlir_ctx,
        loc,
        &[_]mlir.MlirValue{mlir.oraOperationGetResult(next_op, 0)},
        1,
    ));

    mlir.oraBlockAppendOwnedOperation(body, while_op);
    mlir.oraBlockAppendOwnedOperation(body, mlir.oraReturnOpCreate(mlir_ctx, loc, &[_]mlir.MlirValue{}, 0));

    try encoder.registerFunctionOperation(helper);

    const counter_sort = z3.Z3_mk_bv_sort(z3_ctx.ctx, 256);
    const pre_counter = z3.Z3_mk_const(z3_ctx.ctx, z3.Z3_mk_string_symbol(z3_ctx.ctx, "pre_counter_symbolic_signed_while"), counter_sort);
    try encoder.global_map.put(try testing.allocator.dupe(u8, "counter"), pre_counter);

    const symbolic_ub = mlir.oraVariablePlaceholderOpCreate(mlir_ctx, loc, stringRef("signedUbWhileValue"), i256_ty);
    const call = mlir.oraFuncCallOpCreate(
        mlir_ctx,
        loc,
        stringRef("symbolicNoWriteSignedWhile"),
        &[_]mlir.MlirValue{mlir.oraOperationGetResult(symbolic_ub, 0)},
        1,
        &[_]mlir.MlirType{},
        0,
    );
    _ = try encoder.encodeOperation(call);

    try testing.expect(!encoder.isDegraded());
    const post_counter = encoder.global_map.get("counter").?;

    var solver = try Solver.init(&z3_ctx, testing.allocator);
    defer solver.deinit();
    solver.assert(z3.Z3_mk_not(z3_ctx.ctx, z3.Z3_mk_eq(z3_ctx.ctx, post_counter, pre_counter)));
    try testing.expectEqual(@as(z3.Z3_lbool, z3.Z3_L_FALSE), solver.check());
}

test "func.call summary with canonical unsigned multi-result nonzero-control no-write scf.while preserves state exactly" {
    var z3_ctx = try Context.init(testing.allocator);
    defer z3_ctx.deinit();

    var encoder = Encoder.init(&z3_ctx, testing.allocator);
    defer encoder.deinit();

    const mlir_ctx = mlir.oraContextCreate();
    defer mlir.oraContextDestroy(mlir_ctx);
    loadAllDialects(mlir_ctx);
    _ = mlir.oraDialectRegister(mlir_ctx);

    const loc = mlir.oraLocationUnknownGet(mlir_ctx);
    const i256_ty = mlir.oraIntegerTypeCreate(mlir_ctx, 256);

    const helper_attrs = [_]mlir.MlirNamedAttribute{
        namedAttr(mlir_ctx, "sym_name", mlir.oraStringAttrCreate(mlir_ctx, stringRef("symbolicNoWriteWhileMultiState"))),
    };
    const helper_param_types = [_]mlir.MlirType{i256_ty};
    const helper_param_locs = [_]mlir.MlirLocation{loc};
    const helper = mlir.oraFuncFuncOpCreate(mlir_ctx, loc, &helper_attrs, helper_attrs.len, &helper_param_types, &helper_param_locs, helper_param_types.len);
    const body = mlir.oraFuncOpGetBodyBlock(helper);

    const init_sum_attr = mlir.oraIntegerAttrCreateI64FromType(i256_ty, 10);
    const init_ctrl_attr = mlir.oraIntegerAttrCreateI64FromType(i256_ty, 1);
    const ctrl_delta_attr = mlir.oraIntegerAttrCreateI64FromType(i256_ty, 3);
    const sum_delta_attr = mlir.oraIntegerAttrCreateI64FromType(i256_ty, 5);
    const init_sum_op = mlir.oraArithConstantOpCreate(mlir_ctx, loc, i256_ty, init_sum_attr);
    const init_ctrl_op = mlir.oraArithConstantOpCreate(mlir_ctx, loc, i256_ty, init_ctrl_attr);
    const ctrl_delta_op = mlir.oraArithConstantOpCreate(mlir_ctx, loc, i256_ty, ctrl_delta_attr);
    const sum_delta_op = mlir.oraArithConstantOpCreate(mlir_ctx, loc, i256_ty, sum_delta_attr);
    mlir.oraBlockAppendOwnedOperation(body, init_sum_op);
    mlir.oraBlockAppendOwnedOperation(body, init_ctrl_op);
    mlir.oraBlockAppendOwnedOperation(body, ctrl_delta_op);
    mlir.oraBlockAppendOwnedOperation(body, sum_delta_op);

    const ub = mlir.oraBlockGetArgument(body, 0);
    const init_vals = [_]mlir.MlirValue{
        mlir.oraOperationGetResult(init_sum_op, 0),
        mlir.oraOperationGetResult(init_ctrl_op, 0),
    };
    const result_types = [_]mlir.MlirType{ i256_ty, i256_ty };
    const while_op = mlir.oraScfWhileOpCreate(mlir_ctx, loc, &init_vals, init_vals.len, &result_types, result_types.len);
    const before_block = mlir.oraScfWhileOpGetBeforeBlock(while_op);
    const after_block = mlir.oraScfWhileOpGetAfterBlock(while_op);
    _ = mlir.mlirBlockAddArgument(before_block, i256_ty, loc);
    _ = mlir.mlirBlockAddArgument(before_block, i256_ty, loc);
    _ = mlir.mlirBlockAddArgument(after_block, i256_ty, loc);
    _ = mlir.mlirBlockAddArgument(after_block, i256_ty, loc);
    const before_sum = mlir.oraBlockGetArgument(before_block, 0);
    const before_ctrl = mlir.oraBlockGetArgument(before_block, 1);
    const after_sum = mlir.oraBlockGetArgument(after_block, 0);
    const after_ctrl = mlir.oraBlockGetArgument(after_block, 1);

    const cmp_op = mlir.oraArithCmpIOpCreate(mlir_ctx, loc, 6, before_ctrl, ub); // ult
    mlir.oraBlockAppendOwnedOperation(before_block, cmp_op);
    mlir.oraBlockAppendOwnedOperation(before_block, mlir.oraScfConditionOpCreate(
        mlir_ctx,
        loc,
        mlir.oraOperationGetResult(cmp_op, 0),
        &[_]mlir.MlirValue{ before_sum, before_ctrl },
        2,
    ));

    const next_sum_op = mlir.oraArithAddIOpCreate(mlir_ctx, loc, after_sum, mlir.oraOperationGetResult(sum_delta_op, 0));
    const next_ctrl_op = mlir.oraArithAddIOpCreate(mlir_ctx, loc, after_ctrl, mlir.oraOperationGetResult(ctrl_delta_op, 0));
    mlir.oraBlockAppendOwnedOperation(after_block, next_sum_op);
    mlir.oraBlockAppendOwnedOperation(after_block, next_ctrl_op);
    mlir.oraBlockAppendOwnedOperation(after_block, mlir.oraScfYieldOpCreate(
        mlir_ctx,
        loc,
        &[_]mlir.MlirValue{
            mlir.oraOperationGetResult(next_sum_op, 0),
            mlir.oraOperationGetResult(next_ctrl_op, 0),
        },
        2,
    ));

    mlir.oraBlockAppendOwnedOperation(body, while_op);
    mlir.oraBlockAppendOwnedOperation(body, mlir.oraReturnOpCreate(mlir_ctx, loc, &[_]mlir.MlirValue{}, 0));

    try encoder.registerFunctionOperation(helper);

    const counter_sort = z3.Z3_mk_bv_sort(z3_ctx.ctx, 256);
    const pre_counter = z3.Z3_mk_const(z3_ctx.ctx, z3.Z3_mk_string_symbol(z3_ctx.ctx, "pre_counter_symbolic_while_multi_state"), counter_sort);
    try encoder.global_map.put(try testing.allocator.dupe(u8, "counter"), pre_counter);

    const symbolic_ub = mlir.oraVariablePlaceholderOpCreate(mlir_ctx, loc, stringRef("whileMultiStateUb"), i256_ty);
    const call = mlir.oraFuncCallOpCreate(
        mlir_ctx,
        loc,
        stringRef("symbolicNoWriteWhileMultiState"),
        &[_]mlir.MlirValue{mlir.oraOperationGetResult(symbolic_ub, 0)},
        1,
        &[_]mlir.MlirType{},
        0,
    );
    _ = try encoder.encodeOperation(call);

    try testing.expect(!encoder.isDegraded());
    const post_counter = encoder.global_map.get("counter").?;

    var solver = try Solver.init(&z3_ctx, testing.allocator);
    defer solver.deinit();
    solver.assert(z3.Z3_mk_not(z3_ctx.ctx, z3.Z3_mk_eq(z3_ctx.ctx, post_counter, pre_counter)));
    try testing.expectEqual(@as(z3.Z3_lbool, z3.Z3_L_FALSE), solver.check());
}

test "func.call summary with zero-result symbolic no-write scf.while degrades exact state modeling" {
    var z3_ctx = try Context.init(testing.allocator);
    defer z3_ctx.deinit();

    var encoder = Encoder.init(&z3_ctx, testing.allocator);
    defer encoder.deinit();

    const mlir_ctx = mlir.oraContextCreate();
    defer mlir.oraContextDestroy(mlir_ctx);
    loadAllDialects(mlir_ctx);
    _ = mlir.oraDialectRegister(mlir_ctx);

    const loc = mlir.oraLocationUnknownGet(mlir_ctx);
    const i1_ty = mlir.oraIntegerTypeCreate(mlir_ctx, 1);

    const helper_attrs = [_]mlir.MlirNamedAttribute{
        namedAttr(mlir_ctx, "sym_name", mlir.oraStringAttrCreate(mlir_ctx, stringRef("symbolicNoWriteZeroResultWhile"))),
    };
    const helper_param_types = [_]mlir.MlirType{i1_ty};
    const helper_param_locs = [_]mlir.MlirLocation{loc};
    const helper = mlir.oraFuncFuncOpCreate(mlir_ctx, loc, &helper_attrs, helper_attrs.len, &helper_param_types, &helper_param_locs, helper_param_types.len);
    const body = mlir.oraFuncOpGetBodyBlock(helper);

    const cond = mlir.oraBlockGetArgument(body, 0);
    const while_op = mlir.oraScfWhileOpCreate(mlir_ctx, loc, &[_]mlir.MlirValue{}, 0, &[_]mlir.MlirType{}, 0);
    const before_block = mlir.oraScfWhileOpGetBeforeBlock(while_op);
    const after_block = mlir.oraScfWhileOpGetAfterBlock(while_op);
    mlir.oraBlockAppendOwnedOperation(before_block, mlir.oraScfConditionOpCreate(
        mlir_ctx,
        loc,
        cond,
        &[_]mlir.MlirValue{},
        0,
    ));
    mlir.oraBlockAppendOwnedOperation(after_block, mlir.oraScfYieldOpCreate(
        mlir_ctx,
        loc,
        &[_]mlir.MlirValue{},
        0,
    ));

    mlir.oraBlockAppendOwnedOperation(body, while_op);
    mlir.oraBlockAppendOwnedOperation(body, mlir.oraReturnOpCreate(mlir_ctx, loc, &[_]mlir.MlirValue{}, 0));

    try encoder.registerFunctionOperation(helper);

    const call_cond = mlir.oraVariablePlaceholderOpCreate(mlir_ctx, loc, stringRef("zeroResultWhileCond"), i1_ty);
    const call = mlir.oraFuncCallOpCreate(
        mlir_ctx,
        loc,
        stringRef("symbolicNoWriteZeroResultWhile"),
        &[_]mlir.MlirValue{mlir.oraOperationGetResult(call_cond, 0)},
        1,
        &[_]mlir.MlirType{},
        0,
    );
    _ = try encoder.encodeOperation(call);

    try testing.expect(encoder.isDegraded());
}

test "func.call summary with non-throwing ora.try_stmt state effects encodes exactly" {
    var z3_ctx = try Context.init(testing.allocator);
    defer z3_ctx.deinit();

    var encoder = Encoder.init(&z3_ctx, testing.allocator);
    defer encoder.deinit();

    const mlir_ctx = mlir.oraContextCreate();
    defer mlir.oraContextDestroy(mlir_ctx);
    loadAllDialects(mlir_ctx);
    _ = mlir.oraDialectRegister(mlir_ctx);

    const loc = mlir.oraLocationUnknownGet(mlir_ctx);
    const i256_ty = mlir.oraIntegerTypeCreate(mlir_ctx, 256);

    const helper_attrs = [_]mlir.MlirNamedAttribute{
        namedAttr(mlir_ctx, "sym_name", mlir.oraStringAttrCreate(mlir_ctx, stringRef("tryWriter"))),
        namedAttr(mlir_ctx, "ora.effect", mlir.oraStringAttrCreate(mlir_ctx, stringRef("writes"))),
        namedAttr(mlir_ctx, "ora.write_slots", mlir.oraArrayAttrCreate(mlir_ctx, 1, &[_]mlir.MlirAttribute{
            mlir.oraStringAttrCreate(mlir_ctx, stringRef("counter")),
        })),
    };
    const helper = mlir.oraFuncFuncOpCreate(mlir_ctx, loc, &helper_attrs, helper_attrs.len, &[_]mlir.MlirType{}, &[_]mlir.MlirLocation{}, 0);
    const body = mlir.oraFuncOpGetBodyBlock(helper);

    const try_stmt = mlir.oraTryStmtOpCreate(mlir_ctx, loc, &[_]mlir.MlirType{}, 0);
    const try_block = mlir.oraTryStmtOpGetTryBlock(try_stmt);
    const catch_block = mlir.oraTryStmtOpGetCatchBlock(try_stmt);
    const store_val_attr = mlir.oraIntegerAttrCreateI64FromType(i256_ty, 7);
    const store_val_op = mlir.oraArithConstantOpCreate(mlir_ctx, loc, i256_ty, store_val_attr);
    mlir.oraBlockAppendOwnedOperation(try_block, store_val_op);
    mlir.oraBlockAppendOwnedOperation(try_block, mlir.oraSStoreOpCreate(mlir_ctx, loc, mlir.oraOperationGetResult(store_val_op, 0), stringRef("counter")));
    mlir.oraBlockAppendOwnedOperation(try_block, mlir.oraYieldOpCreate(mlir_ctx, loc, &[_]mlir.MlirValue{}, 0));
    mlir.oraBlockAppendOwnedOperation(catch_block, mlir.oraYieldOpCreate(mlir_ctx, loc, &[_]mlir.MlirValue{}, 0));

    mlir.oraBlockAppendOwnedOperation(body, try_stmt);
    mlir.oraBlockAppendOwnedOperation(body, mlir.oraReturnOpCreate(mlir_ctx, loc, &[_]mlir.MlirValue{}, 0));

    try encoder.registerFunctionOperation(helper);

    const call = mlir.oraFuncCallOpCreate(mlir_ctx, loc, stringRef("tryWriter"), &[_]mlir.MlirValue{}, 0, &[_]mlir.MlirType{}, 0);
    _ = try encoder.encodeOperation(call);

    try testing.expect(!encoder.isDegraded());
    const counter = encoder.global_map.get("counter").?;
    const expected = z3.Z3_mk_numeral(z3_ctx.ctx, "7", z3.Z3_mk_bv_sort(z3_ctx.ctx, 256));

    var solver = try Solver.init(&z3_ctx, testing.allocator);
    defer solver.deinit();
    solver.assert(z3.Z3_mk_not(z3_ctx.ctx, z3.Z3_mk_eq(z3_ctx.ctx, counter, expected)));
    try testing.expectEqual(@as(z3.Z3_lbool, z3.Z3_L_FALSE), solver.check());
}

test "func.call summary with always-catching ora.try_stmt state effects encodes exactly" {
    var z3_ctx = try Context.init(testing.allocator);
    defer z3_ctx.deinit();

    var encoder = Encoder.init(&z3_ctx, testing.allocator);
    defer encoder.deinit();

    const mlir_ctx = mlir.oraContextCreate();
    defer mlir.oraContextDestroy(mlir_ctx);
    loadAllDialects(mlir_ctx);
    _ = mlir.oraDialectRegister(mlir_ctx);

    const loc = mlir.oraLocationUnknownGet(mlir_ctx);
    const i256_ty = mlir.oraIntegerTypeCreate(mlir_ctx, 256);
    const eu_ty = mlir.oraErrorUnionTypeGet(mlir_ctx, i256_ty);

    const helper_attrs = [_]mlir.MlirNamedAttribute{
        namedAttr(mlir_ctx, "sym_name", mlir.oraStringAttrCreate(mlir_ctx, stringRef("alwaysCatchTryWriter"))),
        namedAttr(mlir_ctx, "ora.effect", mlir.oraStringAttrCreate(mlir_ctx, stringRef("writes"))),
        namedAttr(mlir_ctx, "ora.write_slots", mlir.oraArrayAttrCreate(mlir_ctx, 1, &[_]mlir.MlirAttribute{
            mlir.oraStringAttrCreate(mlir_ctx, stringRef("counter")),
        })),
    };
    const helper = mlir.oraFuncFuncOpCreate(mlir_ctx, loc, &helper_attrs, helper_attrs.len, &[_]mlir.MlirType{}, &[_]mlir.MlirLocation{}, 0);
    const body = mlir.oraFuncOpGetBodyBlock(helper);

    const try_stmt = mlir.oraTryStmtOpCreate(mlir_ctx, loc, &[_]mlir.MlirType{}, 0);
    const try_block = mlir.oraTryStmtOpGetTryBlock(try_stmt);
    const catch_block = mlir.oraTryStmtOpGetCatchBlock(try_stmt);

    const err_attr = mlir.oraIntegerAttrCreateI64FromType(i256_ty, 1);
    const err_val_op = mlir.oraArithConstantOpCreate(mlir_ctx, loc, i256_ty, err_attr);
    mlir.oraBlockAppendOwnedOperation(try_block, err_val_op);
    const err_op = mlir.oraErrorErrOpCreate(mlir_ctx, loc, mlir.oraOperationGetResult(err_val_op, 0), eu_ty);
    mlir.oraBlockAppendOwnedOperation(try_block, err_op);
    const unwrap_op = mlir.oraErrorUnwrapOpCreate(mlir_ctx, loc, mlir.oraOperationGetResult(err_op, 0), i256_ty);
    mlir.oraBlockAppendOwnedOperation(try_block, unwrap_op);

    const store_val_attr = mlir.oraIntegerAttrCreateI64FromType(i256_ty, 9);
    const store_val_op = mlir.oraArithConstantOpCreate(mlir_ctx, loc, i256_ty, store_val_attr);
    mlir.oraBlockAppendOwnedOperation(catch_block, store_val_op);
    mlir.oraBlockAppendOwnedOperation(catch_block, mlir.oraSStoreOpCreate(mlir_ctx, loc, mlir.oraOperationGetResult(store_val_op, 0), stringRef("counter")));
    mlir.oraBlockAppendOwnedOperation(catch_block, mlir.oraYieldOpCreate(mlir_ctx, loc, &[_]mlir.MlirValue{}, 0));

    mlir.oraBlockAppendOwnedOperation(body, try_stmt);
    mlir.oraBlockAppendOwnedOperation(body, mlir.oraReturnOpCreate(mlir_ctx, loc, &[_]mlir.MlirValue{}, 0));

    try encoder.registerFunctionOperation(helper);

    const call = mlir.oraFuncCallOpCreate(mlir_ctx, loc, stringRef("alwaysCatchTryWriter"), &[_]mlir.MlirValue{}, 0, &[_]mlir.MlirType{}, 0);
    _ = try encoder.encodeOperation(call);

    try testing.expect(!encoder.isDegraded());
    const counter = encoder.global_map.get("counter").?;
    const expected = z3.Z3_mk_numeral(z3_ctx.ctx, "9", z3.Z3_mk_bv_sort(z3_ctx.ctx, 256));

    var solver = try Solver.init(&z3_ctx, testing.allocator);
    defer solver.deinit();
    solver.assert(z3.Z3_mk_not(z3_ctx.ctx, z3.Z3_mk_eq(z3_ctx.ctx, counter, expected)));
    try testing.expectEqual(@as(z3.Z3_lbool, z3.Z3_L_FALSE), solver.check());
}

test "func.call summary with direct symbolic ora.try_stmt state effects encodes exactly" {
    var z3_ctx = try Context.init(testing.allocator);
    defer z3_ctx.deinit();

    var encoder = Encoder.init(&z3_ctx, testing.allocator);
    defer encoder.deinit();

    const mlir_ctx = mlir.oraContextCreate();
    defer mlir.oraContextDestroy(mlir_ctx);
    loadAllDialects(mlir_ctx);
    _ = mlir.oraDialectRegister(mlir_ctx);

    const loc = mlir.oraLocationUnknownGet(mlir_ctx);
    const i256_ty = mlir.oraIntegerTypeCreate(mlir_ctx, 256);
    const eu_ty = mlir.oraErrorUnionTypeGet(mlir_ctx, i256_ty);

    const helper_attrs = [_]mlir.MlirNamedAttribute{
        namedAttr(mlir_ctx, "sym_name", mlir.oraStringAttrCreate(mlir_ctx, stringRef("symbolicTryWriter"))),
        namedAttr(mlir_ctx, "ora.effect", mlir.oraStringAttrCreate(mlir_ctx, stringRef("writes"))),
        namedAttr(mlir_ctx, "ora.write_slots", mlir.oraArrayAttrCreate(mlir_ctx, 1, &[_]mlir.MlirAttribute{
            mlir.oraStringAttrCreate(mlir_ctx, stringRef("counter")),
        })),
    };
    const helper_param_types = [_]mlir.MlirType{eu_ty};
    const helper_param_locs = [_]mlir.MlirLocation{loc};
    const helper = mlir.oraFuncFuncOpCreate(mlir_ctx, loc, &helper_attrs, helper_attrs.len, &helper_param_types, &helper_param_locs, helper_param_types.len);
    const body = mlir.oraFuncOpGetBodyBlock(helper);

    const try_stmt = mlir.oraTryStmtOpCreate(mlir_ctx, loc, &[_]mlir.MlirType{i256_ty}, 1);
    const try_block = mlir.oraTryStmtOpGetTryBlock(try_stmt);
    const catch_block = mlir.oraTryStmtOpGetCatchBlock(try_stmt);

    const maybe_arg = mlir.oraBlockGetArgument(body, 0);
    const unwrap_op = mlir.oraErrorUnwrapOpCreate(mlir_ctx, loc, maybe_arg, i256_ty);
    mlir.oraBlockAppendOwnedOperation(try_block, unwrap_op);
    mlir.oraBlockAppendOwnedOperation(try_block, mlir.oraYieldOpCreate(
        mlir_ctx,
        loc,
        &[_]mlir.MlirValue{mlir.oraOperationGetResult(unwrap_op, 0)},
        1,
    ));

    const store_val_attr = mlir.oraIntegerAttrCreateI64FromType(i256_ty, 33);
    const store_val_op = mlir.oraArithConstantOpCreate(mlir_ctx, loc, i256_ty, store_val_attr);
    mlir.oraBlockAppendOwnedOperation(catch_block, store_val_op);
    mlir.oraBlockAppendOwnedOperation(catch_block, mlir.oraSStoreOpCreate(mlir_ctx, loc, mlir.oraOperationGetResult(store_val_op, 0), stringRef("counter")));
    mlir.oraBlockAppendOwnedOperation(catch_block, mlir.oraYieldOpCreate(
        mlir_ctx,
        loc,
        &[_]mlir.MlirValue{mlir.oraOperationGetResult(store_val_op, 0)},
        1,
    ));

    mlir.oraBlockAppendOwnedOperation(body, try_stmt);
    mlir.oraBlockAppendOwnedOperation(body, mlir.oraReturnOpCreate(mlir_ctx, loc, &[_]mlir.MlirValue{}, 0));

    try encoder.registerFunctionOperation(helper);

    const outer_maybe = mlir.oraFuncCallOpCreate(mlir_ctx, loc, stringRef("maybeValue"), &[_]mlir.MlirValue{}, 0, &[_]mlir.MlirType{eu_ty}, 1);
    const outer_maybe_result = mlir.oraOperationGetResult(outer_maybe, 0);
    const call_helper = mlir.oraFuncCallOpCreate(mlir_ctx, loc, stringRef("symbolicTryWriter"), &[_]mlir.MlirValue{outer_maybe_result}, 1, &[_]mlir.MlirType{}, 0);
    _ = try encoder.encodeOperation(call_helper);

    try testing.expect(!encoder.isDegraded());
    const counter = encoder.global_map.get("counter").?;
    const encoded_text = std.mem.span(z3.Z3_ast_to_string(z3_ctx.ctx, counter));
    try testing.expect(std.mem.indexOf(u8, encoded_text, "ite") != null);
    try testing.expect(std.mem.indexOf(u8, encoded_text, "maybeValue") != null);
}

test "func.call summary with direct symbolic ora.try_stmt result and state stay aligned" {
    var z3_ctx = try Context.init(testing.allocator);
    defer z3_ctx.deinit();

    var encoder = Encoder.init(&z3_ctx, testing.allocator);
    defer encoder.deinit();

    const mlir_ctx = mlir.oraContextCreate();
    defer mlir.oraContextDestroy(mlir_ctx);
    loadAllDialects(mlir_ctx);
    _ = mlir.oraDialectRegister(mlir_ctx);

    const loc = mlir.oraLocationUnknownGet(mlir_ctx);
    const i256_ty = mlir.oraIntegerTypeCreate(mlir_ctx, 256);
    const eu_ty = mlir.oraErrorUnionTypeGet(mlir_ctx, i256_ty);

    const helper_attrs = [_]mlir.MlirNamedAttribute{
        namedAttr(mlir_ctx, "sym_name", mlir.oraStringAttrCreate(mlir_ctx, stringRef("symbolicTryResultWriter"))),
        namedAttr(mlir_ctx, "ora.effect", mlir.oraStringAttrCreate(mlir_ctx, stringRef("writes"))),
        namedAttr(mlir_ctx, "ora.write_slots", mlir.oraArrayAttrCreate(mlir_ctx, 1, &[_]mlir.MlirAttribute{
            mlir.oraStringAttrCreate(mlir_ctx, stringRef("counter")),
        })),
    };
    const helper_param_types = [_]mlir.MlirType{eu_ty};
    const helper_param_locs = [_]mlir.MlirLocation{loc};
    const helper = mlir.oraFuncFuncOpCreate(mlir_ctx, loc, &helper_attrs, helper_attrs.len, &helper_param_types, &helper_param_locs, helper_param_types.len);
    const body = mlir.oraFuncOpGetBodyBlock(helper);

    const try_stmt = mlir.oraTryStmtOpCreate(mlir_ctx, loc, &[_]mlir.MlirType{i256_ty}, 1);
    const try_block = mlir.oraTryStmtOpGetTryBlock(try_stmt);
    const catch_block = mlir.oraTryStmtOpGetCatchBlock(try_stmt);

    const maybe_arg = mlir.oraBlockGetArgument(body, 0);
    const unwrap_op = mlir.oraErrorUnwrapOpCreate(mlir_ctx, loc, maybe_arg, i256_ty);
    const unwrapped = mlir.oraOperationGetResult(unwrap_op, 0);
    mlir.oraBlockAppendOwnedOperation(try_block, unwrap_op);
    mlir.oraBlockAppendOwnedOperation(try_block, mlir.oraSStoreOpCreate(mlir_ctx, loc, unwrapped, stringRef("counter")));
    mlir.oraBlockAppendOwnedOperation(try_block, mlir.oraYieldOpCreate(
        mlir_ctx,
        loc,
        &[_]mlir.MlirValue{unwrapped},
        1,
    ));

    const catch_attr = mlir.oraIntegerAttrCreateI64FromType(i256_ty, 33);
    const catch_op = mlir.oraArithConstantOpCreate(mlir_ctx, loc, i256_ty, catch_attr);
    const catch_value = mlir.oraOperationGetResult(catch_op, 0);
    mlir.oraBlockAppendOwnedOperation(catch_block, catch_op);
    mlir.oraBlockAppendOwnedOperation(catch_block, mlir.oraSStoreOpCreate(mlir_ctx, loc, catch_value, stringRef("counter")));
    mlir.oraBlockAppendOwnedOperation(catch_block, mlir.oraYieldOpCreate(
        mlir_ctx,
        loc,
        &[_]mlir.MlirValue{catch_value},
        1,
    ));

    mlir.oraBlockAppendOwnedOperation(body, try_stmt);
    mlir.oraBlockAppendOwnedOperation(body, mlir.oraReturnOpCreate(
        mlir_ctx,
        loc,
        &[_]mlir.MlirValue{mlir.oraOperationGetResult(try_stmt, 0)},
        1,
    ));

    try encoder.registerFunctionOperation(helper);

    const outer_maybe = mlir.oraFuncCallOpCreate(mlir_ctx, loc, stringRef("maybeValue"), &[_]mlir.MlirValue{}, 0, &[_]mlir.MlirType{eu_ty}, 1);
    const outer_maybe_result = mlir.oraOperationGetResult(outer_maybe, 0);
    const call_helper = mlir.oraFuncCallOpCreate(mlir_ctx, loc, stringRef("symbolicTryResultWriter"), &[_]mlir.MlirValue{outer_maybe_result}, 1, &[_]mlir.MlirType{i256_ty}, 1);
    const encoded_result = try encoder.encodeOperation(call_helper);
    const catch_ast = try encoder.encodeValue(catch_value);

    try testing.expect(!encoder.isDegraded());
    const counter = encoder.global_map.get("counter").?;

    const is_error_op = mlir.oraErrorIsErrorOpCreate(mlir_ctx, loc, outer_maybe_result);
    const is_error = try encoder.encodeOperation(is_error_op);
    const outer_unwrap_op = mlir.oraErrorUnwrapOpCreate(mlir_ctx, loc, outer_maybe_result, i256_ty);
    const ok_value = try encoder.encodeOperation(outer_unwrap_op);

    var solver = try Solver.init(&z3_ctx, testing.allocator);
    defer solver.deinit();

    solver.assert(z3.Z3_mk_not(z3_ctx.ctx, z3.Z3_mk_eq(z3_ctx.ctx, encoded_result, counter)));
    try testing.expectEqual(@as(z3.Z3_lbool, z3.Z3_L_FALSE), solver.check());

    solver.reset();
    solver.assert(z3.Z3_mk_not(z3_ctx.ctx, encoder.coerceBoolean(is_error)));
    solver.assert(z3.Z3_mk_not(z3_ctx.ctx, z3.Z3_mk_eq(z3_ctx.ctx, encoded_result, ok_value)));
    try testing.expectEqual(@as(z3.Z3_lbool, z3.Z3_L_FALSE), solver.check());

    solver.reset();
    solver.assert(z3.Z3_mk_not(z3_ctx.ctx, encoder.coerceBoolean(is_error)));
    solver.assert(z3.Z3_mk_not(z3_ctx.ctx, z3.Z3_mk_eq(z3_ctx.ctx, counter, ok_value)));
    try testing.expectEqual(@as(z3.Z3_lbool, z3.Z3_L_FALSE), solver.check());

    solver.reset();
    solver.assert(encoder.coerceBoolean(is_error));
    solver.assert(z3.Z3_mk_not(z3_ctx.ctx, z3.Z3_mk_eq(z3_ctx.ctx, encoded_result, catch_ast)));
    try testing.expectEqual(@as(z3.Z3_lbool, z3.Z3_L_FALSE), solver.check());

    solver.reset();
    solver.assert(encoder.coerceBoolean(is_error));
    solver.assert(z3.Z3_mk_not(z3_ctx.ctx, z3.Z3_mk_eq(z3_ctx.ctx, counter, catch_ast)));
    try testing.expectEqual(@as(z3.Z3_lbool, z3.Z3_L_FALSE), solver.check());
}

test "func.call summary with direct symbolic no-result ora.try_stmt state effects encodes exactly" {
    var z3_ctx = try Context.init(testing.allocator);
    defer z3_ctx.deinit();

    var encoder = Encoder.init(&z3_ctx, testing.allocator);
    defer encoder.deinit();

    const mlir_ctx = mlir.oraContextCreate();
    defer mlir.oraContextDestroy(mlir_ctx);
    loadAllDialects(mlir_ctx);
    _ = mlir.oraDialectRegister(mlir_ctx);

    const loc = mlir.oraLocationUnknownGet(mlir_ctx);
    const i256_ty = mlir.oraIntegerTypeCreate(mlir_ctx, 256);
    const eu_ty = mlir.oraErrorUnionTypeGet(mlir_ctx, i256_ty);

    const helper_attrs = [_]mlir.MlirNamedAttribute{
        namedAttr(mlir_ctx, "sym_name", mlir.oraStringAttrCreate(mlir_ctx, stringRef("symbolicNoResultTryWriter"))),
        namedAttr(mlir_ctx, "ora.effect", mlir.oraStringAttrCreate(mlir_ctx, stringRef("writes"))),
        namedAttr(mlir_ctx, "ora.write_slots", mlir.oraArrayAttrCreate(mlir_ctx, 1, &[_]mlir.MlirAttribute{
            mlir.oraStringAttrCreate(mlir_ctx, stringRef("counter")),
        })),
    };
    const helper_param_types = [_]mlir.MlirType{eu_ty};
    const helper_param_locs = [_]mlir.MlirLocation{loc};
    const helper = mlir.oraFuncFuncOpCreate(mlir_ctx, loc, &helper_attrs, helper_attrs.len, &helper_param_types, &helper_param_locs, helper_param_types.len);
    const body = mlir.oraFuncOpGetBodyBlock(helper);

    const try_stmt = mlir.oraTryStmtOpCreate(mlir_ctx, loc, &[_]mlir.MlirType{}, 0);
    const try_block = mlir.oraTryStmtOpGetTryBlock(try_stmt);
    const catch_block = mlir.oraTryStmtOpGetCatchBlock(try_stmt);

    const maybe_arg = mlir.oraBlockGetArgument(body, 0);
    const unwrap_op = mlir.oraErrorUnwrapOpCreate(mlir_ctx, loc, maybe_arg, i256_ty);
    mlir.oraBlockAppendOwnedOperation(try_block, unwrap_op);
    const try_store_attr = mlir.oraIntegerAttrCreateI64FromType(i256_ty, 11);
    const try_store_op = mlir.oraArithConstantOpCreate(mlir_ctx, loc, i256_ty, try_store_attr);
    mlir.oraBlockAppendOwnedOperation(try_block, try_store_op);
    mlir.oraBlockAppendOwnedOperation(try_block, mlir.oraSStoreOpCreate(
        mlir_ctx,
        loc,
        mlir.oraOperationGetResult(try_store_op, 0),
        stringRef("counter"),
    ));
    mlir.oraBlockAppendOwnedOperation(try_block, mlir.oraYieldOpCreate(mlir_ctx, loc, &[_]mlir.MlirValue{}, 0));

    const catch_store_attr = mlir.oraIntegerAttrCreateI64FromType(i256_ty, 33);
    const catch_store_op = mlir.oraArithConstantOpCreate(mlir_ctx, loc, i256_ty, catch_store_attr);
    mlir.oraBlockAppendOwnedOperation(catch_block, catch_store_op);
    mlir.oraBlockAppendOwnedOperation(catch_block, mlir.oraSStoreOpCreate(
        mlir_ctx,
        loc,
        mlir.oraOperationGetResult(catch_store_op, 0),
        stringRef("counter"),
    ));
    mlir.oraBlockAppendOwnedOperation(catch_block, mlir.oraYieldOpCreate(mlir_ctx, loc, &[_]mlir.MlirValue{}, 0));

    mlir.oraBlockAppendOwnedOperation(body, try_stmt);
    mlir.oraBlockAppendOwnedOperation(body, mlir.oraReturnOpCreate(mlir_ctx, loc, &[_]mlir.MlirValue{}, 0));

    try encoder.registerFunctionOperation(helper);

    const outer_maybe = mlir.oraFuncCallOpCreate(mlir_ctx, loc, stringRef("maybeValue"), &[_]mlir.MlirValue{}, 0, &[_]mlir.MlirType{eu_ty}, 1);
    const outer_maybe_result = mlir.oraOperationGetResult(outer_maybe, 0);
    const call_helper = mlir.oraFuncCallOpCreate(mlir_ctx, loc, stringRef("symbolicNoResultTryWriter"), &[_]mlir.MlirValue{outer_maybe_result}, 1, &[_]mlir.MlirType{}, 0);
    _ = try encoder.encodeOperation(call_helper);

    try testing.expect(!encoder.isDegraded());
    const counter = encoder.global_map.get("counter").?;
    const encoded_text = std.mem.span(z3.Z3_ast_to_string(z3_ctx.ctx, counter));
    try testing.expect(std.mem.indexOf(u8, encoded_text, "ite") != null);
    try testing.expect(std.mem.indexOf(u8, encoded_text, "maybeValue") != null);
}

test "func.call summary with branch-conditioned ora.try_stmt state effects encodes exactly" {
    var z3_ctx = try Context.init(testing.allocator);
    defer z3_ctx.deinit();

    var encoder = Encoder.init(&z3_ctx, testing.allocator);
    defer encoder.deinit();

    const mlir_ctx = mlir.oraContextCreate();
    defer mlir.oraContextDestroy(mlir_ctx);
    loadAllDialects(mlir_ctx);
    _ = mlir.oraDialectRegister(mlir_ctx);

    const loc = mlir.oraLocationUnknownGet(mlir_ctx);
    const i1_ty = mlir.oraIntegerTypeCreate(mlir_ctx, 1);
    const i256_ty = mlir.oraIntegerTypeCreate(mlir_ctx, 256);
    const eu_ty = mlir.oraErrorUnionTypeGet(mlir_ctx, i256_ty);

    const helper_attrs = [_]mlir.MlirNamedAttribute{
        namedAttr(mlir_ctx, "sym_name", mlir.oraStringAttrCreate(mlir_ctx, stringRef("branchConditionedTryWriter"))),
        namedAttr(mlir_ctx, "ora.effect", mlir.oraStringAttrCreate(mlir_ctx, stringRef("writes"))),
        namedAttr(mlir_ctx, "ora.write_slots", mlir.oraArrayAttrCreate(mlir_ctx, 1, &[_]mlir.MlirAttribute{
            mlir.oraStringAttrCreate(mlir_ctx, stringRef("counter")),
        })),
    };
    const helper_param_types = [_]mlir.MlirType{ i1_ty, eu_ty };
    const helper_param_locs = [_]mlir.MlirLocation{ loc, loc };
    const helper = mlir.oraFuncFuncOpCreate(mlir_ctx, loc, &helper_attrs, helper_attrs.len, &helper_param_types, &helper_param_locs, helper_param_types.len);
    const body = mlir.oraFuncOpGetBodyBlock(helper);

    const try_stmt = mlir.oraTryStmtOpCreate(mlir_ctx, loc, &[_]mlir.MlirType{}, 0);
    const try_block = mlir.oraTryStmtOpGetTryBlock(try_stmt);
    const catch_block = mlir.oraTryStmtOpGetCatchBlock(try_stmt);

    const helper_flag = mlir.oraBlockGetArgument(body, 0);
    const helper_maybe = mlir.oraBlockGetArgument(body, 1);
    const no_results = [_]mlir.MlirType{};
    const if_op = mlir.oraScfIfOpCreate(mlir_ctx, loc, helper_flag, &no_results, no_results.len, true);
    const then_block = mlir.oraScfIfOpGetThenBlock(if_op);
    const else_block = mlir.oraScfIfOpGetElseBlock(if_op);

    const unwrap_op = mlir.oraErrorUnwrapOpCreate(mlir_ctx, loc, helper_maybe, i256_ty);
    mlir.oraBlockAppendOwnedOperation(then_block, unwrap_op);
    const then_store_attr = mlir.oraIntegerAttrCreateI64FromType(i256_ty, 11);
    const then_store_op = mlir.oraArithConstantOpCreate(mlir_ctx, loc, i256_ty, then_store_attr);
    mlir.oraBlockAppendOwnedOperation(then_block, then_store_op);
    mlir.oraBlockAppendOwnedOperation(then_block, mlir.oraSStoreOpCreate(
        mlir_ctx,
        loc,
        mlir.oraOperationGetResult(then_store_op, 0),
        stringRef("counter"),
    ));
    mlir.oraBlockAppendOwnedOperation(then_block, mlir.oraScfYieldOpCreate(mlir_ctx, loc, &[_]mlir.MlirValue{}, 0));

    const else_store_attr = mlir.oraIntegerAttrCreateI64FromType(i256_ty, 7);
    const else_store_op = mlir.oraArithConstantOpCreate(mlir_ctx, loc, i256_ty, else_store_attr);
    mlir.oraBlockAppendOwnedOperation(else_block, else_store_op);
    mlir.oraBlockAppendOwnedOperation(else_block, mlir.oraSStoreOpCreate(
        mlir_ctx,
        loc,
        mlir.oraOperationGetResult(else_store_op, 0),
        stringRef("counter"),
    ));
    mlir.oraBlockAppendOwnedOperation(else_block, mlir.oraScfYieldOpCreate(mlir_ctx, loc, &[_]mlir.MlirValue{}, 0));

    mlir.oraBlockAppendOwnedOperation(try_block, if_op);
    mlir.oraBlockAppendOwnedOperation(try_block, mlir.oraYieldOpCreate(mlir_ctx, loc, &[_]mlir.MlirValue{}, 0));

    const catch_store_attr = mlir.oraIntegerAttrCreateI64FromType(i256_ty, 33);
    const catch_store_op = mlir.oraArithConstantOpCreate(mlir_ctx, loc, i256_ty, catch_store_attr);
    mlir.oraBlockAppendOwnedOperation(catch_block, catch_store_op);
    mlir.oraBlockAppendOwnedOperation(catch_block, mlir.oraSStoreOpCreate(
        mlir_ctx,
        loc,
        mlir.oraOperationGetResult(catch_store_op, 0),
        stringRef("counter"),
    ));
    mlir.oraBlockAppendOwnedOperation(catch_block, mlir.oraYieldOpCreate(mlir_ctx, loc, &[_]mlir.MlirValue{}, 0));

    mlir.oraBlockAppendOwnedOperation(body, try_stmt);
    mlir.oraBlockAppendOwnedOperation(body, mlir.oraReturnOpCreate(mlir_ctx, loc, &[_]mlir.MlirValue{}, 0));

    try encoder.registerFunctionOperation(helper);

    const outer_flag = mlir.oraVariablePlaceholderOpCreate(mlir_ctx, loc, stringRef("condValue"), i1_ty);
    const outer_maybe = mlir.oraFuncCallOpCreate(mlir_ctx, loc, stringRef("maybeValue"), &[_]mlir.MlirValue{}, 0, &[_]mlir.MlirType{eu_ty}, 1);
    const helper_call = mlir.oraFuncCallOpCreate(
        mlir_ctx,
        loc,
        stringRef("branchConditionedTryWriter"),
        &[_]mlir.MlirValue{
            mlir.oraOperationGetResult(outer_flag, 0),
            mlir.oraOperationGetResult(outer_maybe, 0),
        },
        2,
        &[_]mlir.MlirType{},
        0,
    );
    _ = try encoder.encodeOperation(helper_call);

    try testing.expect(!encoder.isDegraded());
    const counter = encoder.global_map.get("counter").?;
    const encoded_text = std.mem.span(z3.Z3_ast_to_string(z3_ctx.ctx, counter));
    try testing.expect(std.mem.indexOf(u8, encoded_text, "ite") != null);
    try testing.expect(std.mem.indexOf(u8, encoded_text, "condValue") != null);
    try testing.expect(std.mem.indexOf(u8, encoded_text, "maybeValue") != null);
}

test "func.call summary with conditional-return ora.try_stmt state effects encodes exactly" {
    var z3_ctx = try Context.init(testing.allocator);
    defer z3_ctx.deinit();

    var encoder = Encoder.init(&z3_ctx, testing.allocator);
    defer encoder.deinit();

    const mlir_ctx = mlir.oraContextCreate();
    defer mlir.oraContextDestroy(mlir_ctx);
    loadAllDialects(mlir_ctx);
    _ = mlir.oraDialectRegister(mlir_ctx);

    const loc = mlir.oraLocationUnknownGet(mlir_ctx);
    const i1_ty = mlir.oraIntegerTypeCreate(mlir_ctx, 1);
    const i256_ty = mlir.oraIntegerTypeCreate(mlir_ctx, 256);
    const eu_ty = mlir.oraErrorUnionTypeGet(mlir_ctx, i256_ty);

    const helper_attrs = [_]mlir.MlirNamedAttribute{
        namedAttr(mlir_ctx, "sym_name", mlir.oraStringAttrCreate(mlir_ctx, stringRef("conditionalReturnTryWriter"))),
        namedAttr(mlir_ctx, "ora.effect", mlir.oraStringAttrCreate(mlir_ctx, stringRef("writes"))),
        namedAttr(mlir_ctx, "ora.write_slots", mlir.oraArrayAttrCreate(mlir_ctx, 1, &[_]mlir.MlirAttribute{
            mlir.oraStringAttrCreate(mlir_ctx, stringRef("counter")),
        })),
    };
    const helper_param_types = [_]mlir.MlirType{ i1_ty, eu_ty };
    const helper_param_locs = [_]mlir.MlirLocation{ loc, loc };
    const helper = mlir.oraFuncFuncOpCreate(mlir_ctx, loc, &helper_attrs, helper_attrs.len, &helper_param_types, &helper_param_locs, helper_param_types.len);
    const body = mlir.oraFuncOpGetBodyBlock(helper);

    const helper_flag = mlir.oraBlockGetArgument(body, 0);
    const helper_maybe = mlir.oraBlockGetArgument(body, 1);

    const try_stmt = mlir.oraTryStmtOpCreate(mlir_ctx, loc, &[_]mlir.MlirType{}, 0);
    const try_block = mlir.oraTryStmtOpGetTryBlock(try_stmt);
    const catch_block = mlir.oraTryStmtOpGetCatchBlock(try_stmt);

    const conditional_ret = mlir.oraConditionalReturnOpCreate(mlir_ctx, loc, helper_flag);
    const then_block = mlir.oraConditionalReturnOpGetThenBlock(conditional_ret);
    const else_block = mlir.oraConditionalReturnOpGetElseBlock(conditional_ret);

    const unwrap_op = mlir.oraErrorUnwrapOpCreate(mlir_ctx, loc, helper_maybe, i256_ty);
    mlir.oraBlockAppendOwnedOperation(then_block, unwrap_op);
    mlir.oraBlockAppendOwnedOperation(then_block, mlir.oraReturnOpCreate(mlir_ctx, loc, &[_]mlir.MlirValue{}, 0));
    mlir.oraBlockAppendOwnedOperation(else_block, mlir.oraYieldOpCreate(mlir_ctx, loc, &[_]mlir.MlirValue{}, 0));

    const try_store_attr = mlir.oraIntegerAttrCreateI64FromType(i256_ty, 7);
    const try_store_op = mlir.oraArithConstantOpCreate(mlir_ctx, loc, i256_ty, try_store_attr);
    mlir.oraBlockAppendOwnedOperation(try_block, conditional_ret);
    mlir.oraBlockAppendOwnedOperation(try_block, try_store_op);
    mlir.oraBlockAppendOwnedOperation(try_block, mlir.oraSStoreOpCreate(
        mlir_ctx,
        loc,
        mlir.oraOperationGetResult(try_store_op, 0),
        stringRef("counter"),
    ));
    mlir.oraBlockAppendOwnedOperation(try_block, mlir.oraYieldOpCreate(mlir_ctx, loc, &[_]mlir.MlirValue{}, 0));

    const catch_store_attr = mlir.oraIntegerAttrCreateI64FromType(i256_ty, 33);
    const catch_store_op = mlir.oraArithConstantOpCreate(mlir_ctx, loc, i256_ty, catch_store_attr);
    mlir.oraBlockAppendOwnedOperation(catch_block, catch_store_op);
    mlir.oraBlockAppendOwnedOperation(catch_block, mlir.oraSStoreOpCreate(
        mlir_ctx,
        loc,
        mlir.oraOperationGetResult(catch_store_op, 0),
        stringRef("counter"),
    ));
    mlir.oraBlockAppendOwnedOperation(catch_block, mlir.oraYieldOpCreate(mlir_ctx, loc, &[_]mlir.MlirValue{}, 0));

    mlir.oraBlockAppendOwnedOperation(body, try_stmt);
    mlir.oraBlockAppendOwnedOperation(body, mlir.oraReturnOpCreate(mlir_ctx, loc, &[_]mlir.MlirValue{}, 0));

    try encoder.registerFunctionOperation(helper);

    const outer_flag = mlir.oraVariablePlaceholderOpCreate(mlir_ctx, loc, stringRef("condValue"), i1_ty);
    const outer_maybe = mlir.oraFuncCallOpCreate(mlir_ctx, loc, stringRef("maybeValue"), &[_]mlir.MlirValue{}, 0, &[_]mlir.MlirType{eu_ty}, 1);
    const helper_call = mlir.oraFuncCallOpCreate(
        mlir_ctx,
        loc,
        stringRef("conditionalReturnTryWriter"),
        &[_]mlir.MlirValue{
            mlir.oraOperationGetResult(outer_flag, 0),
            mlir.oraOperationGetResult(outer_maybe, 0),
        },
        2,
        &[_]mlir.MlirType{},
        0,
    );
    _ = try encoder.encodeOperation(helper_call);

    try testing.expect(!encoder.isDegraded());
    const counter = encoder.global_map.get("counter").?;
    const encoded_text = std.mem.span(z3.Z3_ast_to_string(z3_ctx.ctx, counter));
    try testing.expect(std.mem.indexOf(u8, encoded_text, "ite") != null);
    try testing.expect(std.mem.indexOf(u8, encoded_text, "condValue") != null);
    try testing.expect(std.mem.indexOf(u8, encoded_text, "maybeValue") != null);
}

test "func.call summary with switch_expr ora.try_stmt state effects encodes exactly" {
    var z3_ctx = try Context.init(testing.allocator);
    defer z3_ctx.deinit();

    var encoder = Encoder.init(&z3_ctx, testing.allocator);
    defer encoder.deinit();

    const mlir_ctx = mlir.oraContextCreate();
    defer mlir.oraContextDestroy(mlir_ctx);
    loadAllDialects(mlir_ctx);
    _ = mlir.oraDialectRegister(mlir_ctx);

    const loc = mlir.oraLocationUnknownGet(mlir_ctx);
    const i1_ty = mlir.oraIntegerTypeCreate(mlir_ctx, 1);
    const i256_ty = mlir.oraIntegerTypeCreate(mlir_ctx, 256);
    const eu_ty = mlir.oraErrorUnionTypeGet(mlir_ctx, i256_ty);

    const helper_attrs = [_]mlir.MlirNamedAttribute{
        namedAttr(mlir_ctx, "sym_name", mlir.oraStringAttrCreate(mlir_ctx, stringRef("switchExprTryWriter"))),
        namedAttr(mlir_ctx, "ora.effect", mlir.oraStringAttrCreate(mlir_ctx, stringRef("writes"))),
        namedAttr(mlir_ctx, "ora.write_slots", mlir.oraArrayAttrCreate(mlir_ctx, 1, &[_]mlir.MlirAttribute{
            mlir.oraStringAttrCreate(mlir_ctx, stringRef("counter")),
        })),
    };
    const helper_param_types = [_]mlir.MlirType{ i1_ty, eu_ty };
    const helper_param_locs = [_]mlir.MlirLocation{ loc, loc };
    const helper = mlir.oraFuncFuncOpCreate(mlir_ctx, loc, &helper_attrs, helper_attrs.len, &helper_param_types, &helper_param_locs, helper_param_types.len);
    const body = mlir.oraFuncOpGetBodyBlock(helper);

    const helper_flag = mlir.oraBlockGetArgument(body, 0);
    const helper_maybe = mlir.oraBlockGetArgument(body, 1);

    const try_stmt = mlir.oraTryStmtOpCreate(mlir_ctx, loc, &[_]mlir.MlirType{}, 0);
    const try_block = mlir.oraTryStmtOpGetTryBlock(try_stmt);
    const catch_block = mlir.oraTryStmtOpGetCatchBlock(try_stmt);

    const switch_expr = mlir.oraSwitchExprOpCreateWithCases(
        mlir_ctx,
        loc,
        helper_flag,
        &[_]mlir.MlirType{i256_ty},
        1,
        2,
    );
    const case_values = [_]i64{ 0, 1 };
    const range_starts = [_]i64{ 0, 0 };
    const range_ends = [_]i64{ 0, 0 };
    const case_kinds = [_]i64{ 0, 0 };
    mlir.oraSwitchOpSetCasePatterns(
        switch_expr,
        &case_values,
        &range_starts,
        &range_ends,
        &case_kinds,
        -1,
        case_values.len,
    );

    const false_block = mlir.oraSwitchExprOpGetCaseBlock(switch_expr, 0);
    const true_block = mlir.oraSwitchExprOpGetCaseBlock(switch_expr, 1);

    const false_attr = mlir.oraIntegerAttrCreateI64FromType(i256_ty, 7);
    const false_op = mlir.oraArithConstantOpCreate(mlir_ctx, loc, i256_ty, false_attr);
    mlir.oraBlockAppendOwnedOperation(false_block, false_op);
    mlir.oraBlockAppendOwnedOperation(false_block, mlir.oraYieldOpCreate(
        mlir_ctx,
        loc,
        &[_]mlir.MlirValue{mlir.oraOperationGetResult(false_op, 0)},
        1,
    ));

    const unwrap_op = mlir.oraErrorUnwrapOpCreate(mlir_ctx, loc, helper_maybe, i256_ty);
    mlir.oraBlockAppendOwnedOperation(true_block, unwrap_op);
    mlir.oraBlockAppendOwnedOperation(true_block, mlir.oraYieldOpCreate(
        mlir_ctx,
        loc,
        &[_]mlir.MlirValue{mlir.oraOperationGetResult(unwrap_op, 0)},
        1,
    ));

    const try_store_attr = mlir.oraIntegerAttrCreateI64FromType(i256_ty, 11);
    const try_store_op = mlir.oraArithConstantOpCreate(mlir_ctx, loc, i256_ty, try_store_attr);
    mlir.oraBlockAppendOwnedOperation(try_block, switch_expr);
    mlir.oraBlockAppendOwnedOperation(try_block, try_store_op);
    mlir.oraBlockAppendOwnedOperation(try_block, mlir.oraSStoreOpCreate(
        mlir_ctx,
        loc,
        mlir.oraOperationGetResult(try_store_op, 0),
        stringRef("counter"),
    ));
    mlir.oraBlockAppendOwnedOperation(try_block, mlir.oraYieldOpCreate(mlir_ctx, loc, &[_]mlir.MlirValue{}, 0));

    const catch_store_attr = mlir.oraIntegerAttrCreateI64FromType(i256_ty, 33);
    const catch_store_op = mlir.oraArithConstantOpCreate(mlir_ctx, loc, i256_ty, catch_store_attr);
    mlir.oraBlockAppendOwnedOperation(catch_block, catch_store_op);
    mlir.oraBlockAppendOwnedOperation(catch_block, mlir.oraSStoreOpCreate(
        mlir_ctx,
        loc,
        mlir.oraOperationGetResult(catch_store_op, 0),
        stringRef("counter"),
    ));
    mlir.oraBlockAppendOwnedOperation(catch_block, mlir.oraYieldOpCreate(mlir_ctx, loc, &[_]mlir.MlirValue{}, 0));

    mlir.oraBlockAppendOwnedOperation(body, try_stmt);
    mlir.oraBlockAppendOwnedOperation(body, mlir.oraReturnOpCreate(mlir_ctx, loc, &[_]mlir.MlirValue{}, 0));

    try encoder.registerFunctionOperation(helper);

    const outer_flag = mlir.oraVariablePlaceholderOpCreate(mlir_ctx, loc, stringRef("condValue"), i1_ty);
    const outer_maybe = mlir.oraFuncCallOpCreate(mlir_ctx, loc, stringRef("maybeValue"), &[_]mlir.MlirValue{}, 0, &[_]mlir.MlirType{eu_ty}, 1);
    const helper_call = mlir.oraFuncCallOpCreate(
        mlir_ctx,
        loc,
        stringRef("switchExprTryWriter"),
        &[_]mlir.MlirValue{
            mlir.oraOperationGetResult(outer_flag, 0),
            mlir.oraOperationGetResult(outer_maybe, 0),
        },
        2,
        &[_]mlir.MlirType{},
        0,
    );
    _ = try encoder.encodeOperation(helper_call);

    try testing.expect(!encoder.isDegraded());
    const counter = encoder.global_map.get("counter").?;
    const encoded_text = std.mem.span(z3.Z3_ast_to_string(z3_ctx.ctx, counter));
    try testing.expect(std.mem.indexOf(u8, encoded_text, "ite") != null);
    try testing.expect(std.mem.indexOf(u8, encoded_text, "condValue") != null);
    try testing.expect(std.mem.indexOf(u8, encoded_text, "maybeValue") != null);
}

test "func.call summary with switch_expr state effects encodes exactly" {
    var z3_ctx = try Context.init(testing.allocator);
    defer z3_ctx.deinit();

    var encoder = Encoder.init(&z3_ctx, testing.allocator);
    defer encoder.deinit();

    const mlir_ctx = mlir.oraContextCreate();
    defer mlir.oraContextDestroy(mlir_ctx);
    loadAllDialects(mlir_ctx);
    _ = mlir.oraDialectRegister(mlir_ctx);

    const loc = mlir.oraLocationUnknownGet(mlir_ctx);
    const i1_ty = mlir.oraIntegerTypeCreate(mlir_ctx, 1);
    const i256_ty = mlir.oraIntegerTypeCreate(mlir_ctx, 256);

    const helper_attrs = [_]mlir.MlirNamedAttribute{
        namedAttr(mlir_ctx, "sym_name", mlir.oraStringAttrCreate(mlir_ctx, stringRef("switchExprWriter"))),
        namedAttr(mlir_ctx, "ora.effect", mlir.oraStringAttrCreate(mlir_ctx, stringRef("writes"))),
        namedAttr(mlir_ctx, "ora.write_slots", mlir.oraArrayAttrCreate(mlir_ctx, 1, &[_]mlir.MlirAttribute{
            mlir.oraStringAttrCreate(mlir_ctx, stringRef("counter")),
        })),
    };
    const helper_param_types = [_]mlir.MlirType{i1_ty};
    const helper_param_locs = [_]mlir.MlirLocation{loc};
    const helper = mlir.oraFuncFuncOpCreate(mlir_ctx, loc, &helper_attrs, helper_attrs.len, &helper_param_types, &helper_param_locs, helper_param_types.len);
    const body = mlir.oraFuncOpGetBodyBlock(helper);
    const helper_flag = mlir.oraBlockGetArgument(body, 0);

    const switch_expr = mlir.oraSwitchExprOpCreateWithCases(
        mlir_ctx,
        loc,
        helper_flag,
        &[_]mlir.MlirType{i256_ty},
        1,
        2,
    );
    const case_values = [_]i64{ 0, 1 };
    const range_starts = [_]i64{ 0, 0 };
    const range_ends = [_]i64{ 0, 0 };
    const case_kinds = [_]i64{ 0, 0 };
    mlir.oraSwitchOpSetCasePatterns(
        switch_expr,
        &case_values,
        &range_starts,
        &range_ends,
        &case_kinds,
        -1,
        case_values.len,
    );

    const false_block = mlir.oraSwitchExprOpGetCaseBlock(switch_expr, 0);
    const true_block = mlir.oraSwitchExprOpGetCaseBlock(switch_expr, 1);

    const false_store_attr = mlir.oraIntegerAttrCreateI64FromType(i256_ty, 7);
    const false_store_op = mlir.oraArithConstantOpCreate(mlir_ctx, loc, i256_ty, false_store_attr);
    mlir.oraBlockAppendOwnedOperation(false_block, false_store_op);
    mlir.oraBlockAppendOwnedOperation(false_block, mlir.oraSStoreOpCreate(
        mlir_ctx,
        loc,
        mlir.oraOperationGetResult(false_store_op, 0),
        stringRef("counter"),
    ));
    mlir.oraBlockAppendOwnedOperation(false_block, mlir.oraYieldOpCreate(
        mlir_ctx,
        loc,
        &[_]mlir.MlirValue{mlir.oraOperationGetResult(false_store_op, 0)},
        1,
    ));

    const true_store_attr = mlir.oraIntegerAttrCreateI64FromType(i256_ty, 11);
    const true_store_op = mlir.oraArithConstantOpCreate(mlir_ctx, loc, i256_ty, true_store_attr);
    mlir.oraBlockAppendOwnedOperation(true_block, true_store_op);
    mlir.oraBlockAppendOwnedOperation(true_block, mlir.oraSStoreOpCreate(
        mlir_ctx,
        loc,
        mlir.oraOperationGetResult(true_store_op, 0),
        stringRef("counter"),
    ));
    mlir.oraBlockAppendOwnedOperation(true_block, mlir.oraYieldOpCreate(
        mlir_ctx,
        loc,
        &[_]mlir.MlirValue{mlir.oraOperationGetResult(true_store_op, 0)},
        1,
    ));

    mlir.oraBlockAppendOwnedOperation(body, switch_expr);
    mlir.oraBlockAppendOwnedOperation(body, mlir.oraReturnOpCreate(mlir_ctx, loc, &[_]mlir.MlirValue{}, 0));

    try encoder.registerFunctionOperation(helper);

    const outer_flag = mlir.oraVariablePlaceholderOpCreate(mlir_ctx, loc, stringRef("condValue"), i1_ty);
    const helper_call = mlir.oraFuncCallOpCreate(
        mlir_ctx,
        loc,
        stringRef("switchExprWriter"),
        &[_]mlir.MlirValue{mlir.oraOperationGetResult(outer_flag, 0)},
        1,
        &[_]mlir.MlirType{},
        0,
    );
    _ = try encoder.encodeOperation(helper_call);

    try testing.expect(!encoder.isDegraded());
    const counter = encoder.global_map.get("counter").?;
    const encoded_text = std.mem.span(z3.Z3_ast_to_string(z3_ctx.ctx, counter));
    try testing.expect(std.mem.indexOf(u8, encoded_text, "ite") != null);
    try testing.expect(std.mem.indexOf(u8, encoded_text, "condValue") != null);
}

test "func.call summary with equivalent ora.try_stmt branches preserves state exactly" {
    var z3_ctx = try Context.init(testing.allocator);
    defer z3_ctx.deinit();

    var encoder = Encoder.init(&z3_ctx, testing.allocator);
    defer encoder.deinit();

    const mlir_ctx = mlir.oraContextCreate();
    defer mlir.oraContextDestroy(mlir_ctx);
    loadAllDialects(mlir_ctx);
    _ = mlir.oraDialectRegister(mlir_ctx);

    const loc = mlir.oraLocationUnknownGet(mlir_ctx);
    const i256_ty = mlir.oraIntegerTypeCreate(mlir_ctx, 256);

    const helper_attrs = [_]mlir.MlirNamedAttribute{
        namedAttr(mlir_ctx, "sym_name", mlir.oraStringAttrCreate(mlir_ctx, stringRef("equivalentTryWriter"))),
        namedAttr(mlir_ctx, "ora.effect", mlir.oraStringAttrCreate(mlir_ctx, stringRef("writes"))),
        namedAttr(mlir_ctx, "ora.write_slots", mlir.oraArrayAttrCreate(mlir_ctx, 1, &[_]mlir.MlirAttribute{
            mlir.oraStringAttrCreate(mlir_ctx, stringRef("counter")),
        })),
    };
    const helper = mlir.oraFuncFuncOpCreate(mlir_ctx, loc, &helper_attrs, helper_attrs.len, &[_]mlir.MlirType{}, &[_]mlir.MlirLocation{}, 0);
    const body = mlir.oraFuncOpGetBodyBlock(helper);

    const try_stmt = mlir.oraTryStmtOpCreate(mlir_ctx, loc, &[_]mlir.MlirType{}, 0);
    const try_block = mlir.oraTryStmtOpGetTryBlock(try_stmt);
    const catch_block = mlir.oraTryStmtOpGetCatchBlock(try_stmt);

    const nested_try = mlir.oraTryStmtOpCreate(mlir_ctx, loc, &[_]mlir.MlirType{}, 0);
    const nested_try_block = mlir.oraTryStmtOpGetTryBlock(nested_try);
    const nested_catch_block = mlir.oraTryStmtOpGetCatchBlock(nested_try);
    mlir.oraBlockAppendOwnedOperation(nested_try_block, mlir.oraYieldOpCreate(mlir_ctx, loc, &[_]mlir.MlirValue{}, 0));
    mlir.oraBlockAppendOwnedOperation(nested_catch_block, mlir.oraYieldOpCreate(mlir_ctx, loc, &[_]mlir.MlirValue{}, 0));
    mlir.oraBlockAppendOwnedOperation(try_block, nested_try);

    const store_attr = mlir.oraIntegerAttrCreateI64FromType(i256_ty, 7);
    const try_store_op = mlir.oraArithConstantOpCreate(mlir_ctx, loc, i256_ty, store_attr);
    const catch_store_op = mlir.oraArithConstantOpCreate(mlir_ctx, loc, i256_ty, store_attr);
    mlir.oraBlockAppendOwnedOperation(try_block, try_store_op);
    mlir.oraBlockAppendOwnedOperation(try_block, mlir.oraSStoreOpCreate(mlir_ctx, loc, mlir.oraOperationGetResult(try_store_op, 0), stringRef("counter")));
    mlir.oraBlockAppendOwnedOperation(try_block, mlir.oraYieldOpCreate(mlir_ctx, loc, &[_]mlir.MlirValue{}, 0));
    mlir.oraBlockAppendOwnedOperation(catch_block, catch_store_op);
    mlir.oraBlockAppendOwnedOperation(catch_block, mlir.oraSStoreOpCreate(mlir_ctx, loc, mlir.oraOperationGetResult(catch_store_op, 0), stringRef("counter")));
    mlir.oraBlockAppendOwnedOperation(catch_block, mlir.oraYieldOpCreate(mlir_ctx, loc, &[_]mlir.MlirValue{}, 0));

    mlir.oraBlockAppendOwnedOperation(body, try_stmt);
    mlir.oraBlockAppendOwnedOperation(body, mlir.oraReturnOpCreate(mlir_ctx, loc, &[_]mlir.MlirValue{}, 0));

    try encoder.registerFunctionOperation(helper);

    const call = mlir.oraFuncCallOpCreate(mlir_ctx, loc, stringRef("equivalentTryWriter"), &[_]mlir.MlirValue{}, 0, &[_]mlir.MlirType{}, 0);
    _ = try encoder.encodeOperation(call);

    try testing.expect(!encoder.isDegraded());
    const counter = encoder.global_map.get("counter").?;
    const expected = z3.Z3_mk_numeral(z3_ctx.ctx, "7", z3.Z3_mk_bv_sort(z3_ctx.ctx, 256));

    var solver = try Solver.init(&z3_ctx, testing.allocator);
    defer solver.deinit();
    solver.assert(z3.Z3_mk_not(z3_ctx.ctx, z3.Z3_mk_eq(z3_ctx.ctx, counter, expected)));
    try testing.expectEqual(@as(z3.Z3_lbool, z3.Z3_L_FALSE), solver.check());
}

test "func.call summary with dead zero-iteration scf.for in ora.try_stmt preserves state exactly" {
    var z3_ctx = try Context.init(testing.allocator);
    defer z3_ctx.deinit();

    var encoder = Encoder.init(&z3_ctx, testing.allocator);
    defer encoder.deinit();

    const mlir_ctx = mlir.oraContextCreate();
    defer mlir.oraContextDestroy(mlir_ctx);
    loadAllDialects(mlir_ctx);
    _ = mlir.oraDialectRegister(mlir_ctx);

    const loc = mlir.oraLocationUnknownGet(mlir_ctx);
    const index_ty = mlir.oraIndexTypeCreate(mlir_ctx);
    const i256_ty = mlir.oraIntegerTypeCreate(mlir_ctx, 256);
    const eu_ty = mlir.oraErrorUnionTypeGet(mlir_ctx, i256_ty);

    const helper_attrs = [_]mlir.MlirNamedAttribute{
        namedAttr(mlir_ctx, "sym_name", mlir.oraStringAttrCreate(mlir_ctx, stringRef("deadTryForWriter"))),
        namedAttr(mlir_ctx, "ora.effect", mlir.oraStringAttrCreate(mlir_ctx, stringRef("writes"))),
        namedAttr(mlir_ctx, "ora.write_slots", mlir.oraArrayAttrCreate(mlir_ctx, 1, &[_]mlir.MlirAttribute{
            mlir.oraStringAttrCreate(mlir_ctx, stringRef("counter")),
        })),
    };
    const helper = mlir.oraFuncFuncOpCreate(mlir_ctx, loc, &helper_attrs, helper_attrs.len, &[_]mlir.MlirType{}, &[_]mlir.MlirLocation{}, 0);
    const body = mlir.oraFuncOpGetBodyBlock(helper);

    const try_stmt = mlir.oraTryStmtOpCreate(mlir_ctx, loc, &[_]mlir.MlirType{}, 0);
    const try_block = mlir.oraTryStmtOpGetTryBlock(try_stmt);
    const catch_block = mlir.oraTryStmtOpGetCatchBlock(try_stmt);

    const zero_attr = mlir.oraIntegerAttrCreateI64FromType(index_ty, 0);
    const one_attr = mlir.oraIntegerAttrCreateI64FromType(index_ty, 1);
    const zero_op = mlir.oraArithConstantOpCreate(mlir_ctx, loc, index_ty, zero_attr);
    const one_op = mlir.oraArithConstantOpCreate(mlir_ctx, loc, index_ty, one_attr);
    mlir.oraBlockAppendOwnedOperation(try_block, one_op);
    mlir.oraBlockAppendOwnedOperation(try_block, zero_op);

    const loop = mlir.oraScfForOpCreate(
        mlir_ctx,
        loc,
        mlir.oraOperationGetResult(one_op, 0),
        mlir.oraOperationGetResult(zero_op, 0),
        mlir.oraOperationGetResult(one_op, 0),
        &[_]mlir.MlirValue{},
        0,
        false,
    );
    const loop_body = mlir.oraScfForOpGetBodyBlock(loop);

    const err_attr = mlir.oraIntegerAttrCreateI64FromType(i256_ty, 1);
    const err_val_op = mlir.oraArithConstantOpCreate(mlir_ctx, loc, i256_ty, err_attr);
    mlir.oraBlockAppendOwnedOperation(loop_body, err_val_op);
    const err_op = mlir.oraErrorErrOpCreate(mlir_ctx, loc, mlir.oraOperationGetResult(err_val_op, 0), eu_ty);
    mlir.oraBlockAppendOwnedOperation(loop_body, err_op);
    const unwrap_op = mlir.oraErrorUnwrapOpCreate(mlir_ctx, loc, mlir.oraOperationGetResult(err_op, 0), i256_ty);
    mlir.oraBlockAppendOwnedOperation(loop_body, unwrap_op);
    mlir.oraBlockAppendOwnedOperation(loop_body, mlir.oraScfYieldOpCreate(mlir_ctx, loc, &[_]mlir.MlirValue{}, 0));
    mlir.oraBlockAppendOwnedOperation(try_block, loop);

    const store_val_attr = mlir.oraIntegerAttrCreateI64FromType(i256_ty, 7);
    const store_val_op = mlir.oraArithConstantOpCreate(mlir_ctx, loc, i256_ty, store_val_attr);
    mlir.oraBlockAppendOwnedOperation(try_block, store_val_op);
    mlir.oraBlockAppendOwnedOperation(try_block, mlir.oraSStoreOpCreate(mlir_ctx, loc, mlir.oraOperationGetResult(store_val_op, 0), stringRef("counter")));
    mlir.oraBlockAppendOwnedOperation(try_block, mlir.oraYieldOpCreate(mlir_ctx, loc, &[_]mlir.MlirValue{}, 0));

    mlir.oraBlockAppendOwnedOperation(catch_block, mlir.oraYieldOpCreate(mlir_ctx, loc, &[_]mlir.MlirValue{}, 0));

    mlir.oraBlockAppendOwnedOperation(body, try_stmt);
    mlir.oraBlockAppendOwnedOperation(body, mlir.oraReturnOpCreate(mlir_ctx, loc, &[_]mlir.MlirValue{}, 0));

    try encoder.registerFunctionOperation(helper);

    const call = mlir.oraFuncCallOpCreate(mlir_ctx, loc, stringRef("deadTryForWriter"), &[_]mlir.MlirValue{}, 0, &[_]mlir.MlirType{}, 0);
    _ = try encoder.encodeOperation(call);

    try testing.expect(!encoder.isDegraded());
    const counter = encoder.global_map.get("counter").?;
    const expected = z3.Z3_mk_numeral(z3_ctx.ctx, "7", z3.Z3_mk_bv_sort(z3_ctx.ctx, 256));

    var solver = try Solver.init(&z3_ctx, testing.allocator);
    defer solver.deinit();
    solver.assert(z3.Z3_mk_not(z3_ctx.ctx, z3.Z3_mk_eq(z3_ctx.ctx, counter, expected)));
    try testing.expectEqual(@as(z3.Z3_lbool, z3.Z3_L_FALSE), solver.check());
}

test "func.call summary with dead zero-iteration scf.while in ora.try_stmt preserves state exactly" {
    var z3_ctx = try Context.init(testing.allocator);
    defer z3_ctx.deinit();

    var encoder = Encoder.init(&z3_ctx, testing.allocator);
    defer encoder.deinit();

    const mlir_ctx = mlir.oraContextCreate();
    defer mlir.oraContextDestroy(mlir_ctx);
    loadAllDialects(mlir_ctx);
    _ = mlir.oraDialectRegister(mlir_ctx);

    const loc = mlir.oraLocationUnknownGet(mlir_ctx);
    const i1_ty = mlir.oraIntegerTypeCreate(mlir_ctx, 1);
    const i256_ty = mlir.oraIntegerTypeCreate(mlir_ctx, 256);
    const eu_ty = mlir.oraErrorUnionTypeGet(mlir_ctx, i256_ty);

    const helper_attrs = [_]mlir.MlirNamedAttribute{
        namedAttr(mlir_ctx, "sym_name", mlir.oraStringAttrCreate(mlir_ctx, stringRef("deadTryWhileWriter"))),
        namedAttr(mlir_ctx, "ora.effect", mlir.oraStringAttrCreate(mlir_ctx, stringRef("writes"))),
        namedAttr(mlir_ctx, "ora.write_slots", mlir.oraArrayAttrCreate(mlir_ctx, 1, &[_]mlir.MlirAttribute{
            mlir.oraStringAttrCreate(mlir_ctx, stringRef("counter")),
        })),
    };
    const helper = mlir.oraFuncFuncOpCreate(mlir_ctx, loc, &helper_attrs, helper_attrs.len, &[_]mlir.MlirType{}, &[_]mlir.MlirLocation{}, 0);
    const body = mlir.oraFuncOpGetBodyBlock(helper);

    const try_stmt = mlir.oraTryStmtOpCreate(mlir_ctx, loc, &[_]mlir.MlirType{}, 0);
    const try_block = mlir.oraTryStmtOpGetTryBlock(try_stmt);
    const catch_block = mlir.oraTryStmtOpGetCatchBlock(try_stmt);

    const false_attr = mlir.oraIntegerAttrCreateI64FromType(i1_ty, 0);
    const false_op = mlir.oraArithConstantOpCreate(mlir_ctx, loc, i1_ty, false_attr);
    mlir.oraBlockAppendOwnedOperation(try_block, false_op);

    const while_op = mlir.oraScfWhileOpCreate(mlir_ctx, loc, &[_]mlir.MlirValue{}, 0, &[_]mlir.MlirType{}, 0);
    const before = mlir.oraScfWhileOpGetBeforeBlock(while_op);
    const after = mlir.oraScfWhileOpGetAfterBlock(while_op);
    mlir.oraBlockAppendOwnedOperation(before, mlir.oraScfConditionOpCreate(
        mlir_ctx,
        loc,
        mlir.oraOperationGetResult(false_op, 0),
        &[_]mlir.MlirValue{},
        0,
    ));

    const err_attr = mlir.oraIntegerAttrCreateI64FromType(i256_ty, 1);
    const err_val_op = mlir.oraArithConstantOpCreate(mlir_ctx, loc, i256_ty, err_attr);
    mlir.oraBlockAppendOwnedOperation(after, err_val_op);
    const err_op = mlir.oraErrorErrOpCreate(mlir_ctx, loc, mlir.oraOperationGetResult(err_val_op, 0), eu_ty);
    mlir.oraBlockAppendOwnedOperation(after, err_op);
    const unwrap_op = mlir.oraErrorUnwrapOpCreate(mlir_ctx, loc, mlir.oraOperationGetResult(err_op, 0), i256_ty);
    mlir.oraBlockAppendOwnedOperation(after, unwrap_op);
    mlir.oraBlockAppendOwnedOperation(after, mlir.oraScfYieldOpCreate(mlir_ctx, loc, &[_]mlir.MlirValue{}, 0));
    mlir.oraBlockAppendOwnedOperation(try_block, while_op);

    const store_val_attr = mlir.oraIntegerAttrCreateI64FromType(i256_ty, 7);
    const store_val_op = mlir.oraArithConstantOpCreate(mlir_ctx, loc, i256_ty, store_val_attr);
    mlir.oraBlockAppendOwnedOperation(try_block, store_val_op);
    mlir.oraBlockAppendOwnedOperation(try_block, mlir.oraSStoreOpCreate(mlir_ctx, loc, mlir.oraOperationGetResult(store_val_op, 0), stringRef("counter")));
    mlir.oraBlockAppendOwnedOperation(try_block, mlir.oraYieldOpCreate(mlir_ctx, loc, &[_]mlir.MlirValue{}, 0));

    mlir.oraBlockAppendOwnedOperation(catch_block, mlir.oraYieldOpCreate(mlir_ctx, loc, &[_]mlir.MlirValue{}, 0));

    mlir.oraBlockAppendOwnedOperation(body, try_stmt);
    mlir.oraBlockAppendOwnedOperation(body, mlir.oraReturnOpCreate(mlir_ctx, loc, &[_]mlir.MlirValue{}, 0));

    try encoder.registerFunctionOperation(helper);

    const call = mlir.oraFuncCallOpCreate(mlir_ctx, loc, stringRef("deadTryWhileWriter"), &[_]mlir.MlirValue{}, 0, &[_]mlir.MlirType{}, 0);
    _ = try encoder.encodeOperation(call);

    try testing.expect(!encoder.isDegraded());
    const counter = encoder.global_map.get("counter").?;
    const expected = z3.Z3_mk_numeral(z3_ctx.ctx, "7", z3.Z3_mk_bv_sort(z3_ctx.ctx, 256));

    var solver = try Solver.init(&z3_ctx, testing.allocator);
    defer solver.deinit();
    solver.assert(z3.Z3_mk_not(z3_ctx.ctx, z3.Z3_mk_eq(z3_ctx.ctx, counter, expected)));
    try testing.expectEqual(@as(z3.Z3_lbool, z3.Z3_L_FALSE), solver.check());
}

test "func.call summary with single-iteration scf.for in ora.try_stmt preserves state exactly" {
    var z3_ctx = try Context.init(testing.allocator);
    defer z3_ctx.deinit();

    var encoder = Encoder.init(&z3_ctx, testing.allocator);
    defer encoder.deinit();

    const mlir_ctx = mlir.oraContextCreate();
    defer mlir.oraContextDestroy(mlir_ctx);
    loadAllDialects(mlir_ctx);
    _ = mlir.oraDialectRegister(mlir_ctx);

    const loc = mlir.oraLocationUnknownGet(mlir_ctx);
    const index_ty = mlir.oraIndexTypeCreate(mlir_ctx);
    const i256_ty = mlir.oraIntegerTypeCreate(mlir_ctx, 256);
    const eu_ty = mlir.oraErrorUnionTypeGet(mlir_ctx, i256_ty);

    const helper_attrs = [_]mlir.MlirNamedAttribute{
        namedAttr(mlir_ctx, "sym_name", mlir.oraStringAttrCreate(mlir_ctx, stringRef("singleIterTryForWriter"))),
        namedAttr(mlir_ctx, "ora.effect", mlir.oraStringAttrCreate(mlir_ctx, stringRef("writes"))),
        namedAttr(mlir_ctx, "ora.write_slots", mlir.oraArrayAttrCreate(mlir_ctx, 1, &[_]mlir.MlirAttribute{
            mlir.oraStringAttrCreate(mlir_ctx, stringRef("counter")),
        })),
    };
    const helper = mlir.oraFuncFuncOpCreate(mlir_ctx, loc, &helper_attrs, helper_attrs.len, &[_]mlir.MlirType{}, &[_]mlir.MlirLocation{}, 0);
    const body = mlir.oraFuncOpGetBodyBlock(helper);

    const try_stmt = mlir.oraTryStmtOpCreate(mlir_ctx, loc, &[_]mlir.MlirType{}, 0);
    const try_block = mlir.oraTryStmtOpGetTryBlock(try_stmt);
    const catch_block = mlir.oraTryStmtOpGetCatchBlock(try_stmt);

    const zero_attr = mlir.oraIntegerAttrCreateI64FromType(index_ty, 0);
    const one_attr = mlir.oraIntegerAttrCreateI64FromType(index_ty, 1);
    const zero_op = mlir.oraArithConstantOpCreate(mlir_ctx, loc, index_ty, zero_attr);
    const one_op = mlir.oraArithConstantOpCreate(mlir_ctx, loc, index_ty, one_attr);
    mlir.oraBlockAppendOwnedOperation(try_block, zero_op);
    mlir.oraBlockAppendOwnedOperation(try_block, one_op);

    const loop = mlir.oraScfForOpCreate(
        mlir_ctx,
        loc,
        mlir.oraOperationGetResult(zero_op, 0),
        mlir.oraOperationGetResult(one_op, 0),
        mlir.oraOperationGetResult(one_op, 0),
        &[_]mlir.MlirValue{},
        0,
        false,
    );
    const loop_body = mlir.oraScfForOpGetBodyBlock(loop);

    const err_attr = mlir.oraIntegerAttrCreateI64FromType(i256_ty, 1);
    const err_val_op = mlir.oraArithConstantOpCreate(mlir_ctx, loc, i256_ty, err_attr);
    mlir.oraBlockAppendOwnedOperation(loop_body, err_val_op);
    const err_op = mlir.oraErrorErrOpCreate(mlir_ctx, loc, mlir.oraOperationGetResult(err_val_op, 0), eu_ty);
    mlir.oraBlockAppendOwnedOperation(loop_body, err_op);
    const unwrap_op = mlir.oraErrorUnwrapOpCreate(mlir_ctx, loc, mlir.oraOperationGetResult(err_op, 0), i256_ty);
    mlir.oraBlockAppendOwnedOperation(loop_body, unwrap_op);
    mlir.oraBlockAppendOwnedOperation(loop_body, mlir.oraScfYieldOpCreate(mlir_ctx, loc, &[_]mlir.MlirValue{}, 0));
    mlir.oraBlockAppendOwnedOperation(try_block, loop);

    const try_store_attr = mlir.oraIntegerAttrCreateI64FromType(i256_ty, 7);
    const try_store_op = mlir.oraArithConstantOpCreate(mlir_ctx, loc, i256_ty, try_store_attr);
    mlir.oraBlockAppendOwnedOperation(try_block, try_store_op);
    mlir.oraBlockAppendOwnedOperation(try_block, mlir.oraSStoreOpCreate(
        mlir_ctx,
        loc,
        mlir.oraOperationGetResult(try_store_op, 0),
        stringRef("counter"),
    ));
    mlir.oraBlockAppendOwnedOperation(try_block, mlir.oraYieldOpCreate(mlir_ctx, loc, &[_]mlir.MlirValue{}, 0));

    const catch_store_attr = mlir.oraIntegerAttrCreateI64FromType(i256_ty, 33);
    const catch_store_op = mlir.oraArithConstantOpCreate(mlir_ctx, loc, i256_ty, catch_store_attr);
    mlir.oraBlockAppendOwnedOperation(catch_block, catch_store_op);
    mlir.oraBlockAppendOwnedOperation(catch_block, mlir.oraSStoreOpCreate(
        mlir_ctx,
        loc,
        mlir.oraOperationGetResult(catch_store_op, 0),
        stringRef("counter"),
    ));
    mlir.oraBlockAppendOwnedOperation(catch_block, mlir.oraYieldOpCreate(mlir_ctx, loc, &[_]mlir.MlirValue{}, 0));

    mlir.oraBlockAppendOwnedOperation(body, try_stmt);
    mlir.oraBlockAppendOwnedOperation(body, mlir.oraReturnOpCreate(mlir_ctx, loc, &[_]mlir.MlirValue{}, 0));

    try encoder.registerFunctionOperation(helper);

    const call = mlir.oraFuncCallOpCreate(mlir_ctx, loc, stringRef("singleIterTryForWriter"), &[_]mlir.MlirValue{}, 0, &[_]mlir.MlirType{}, 0);
    _ = try encoder.encodeOperation(call);

    try testing.expect(!encoder.isDegraded());
    const counter = encoder.global_map.get("counter").?;
    const expected = z3.Z3_mk_numeral(z3_ctx.ctx, "33", z3.Z3_mk_bv_sort(z3_ctx.ctx, 256));

    var solver = try Solver.init(&z3_ctx, testing.allocator);
    defer solver.deinit();
    solver.assert(z3.Z3_mk_not(z3_ctx.ctx, z3.Z3_mk_eq(z3_ctx.ctx, counter, expected)));
    try testing.expectEqual(@as(z3.Z3_lbool, z3.Z3_L_FALSE), solver.check());
}

test "func.call summary with single-iteration scf.while in ora.try_stmt preserves state exactly" {
    var z3_ctx = try Context.init(testing.allocator);
    defer z3_ctx.deinit();

    var encoder = Encoder.init(&z3_ctx, testing.allocator);
    defer encoder.deinit();

    const mlir_ctx = mlir.oraContextCreate();
    defer mlir.oraContextDestroy(mlir_ctx);
    loadAllDialects(mlir_ctx);
    _ = mlir.oraDialectRegister(mlir_ctx);

    const loc = mlir.oraLocationUnknownGet(mlir_ctx);
    const i1_ty = mlir.oraIntegerTypeCreate(mlir_ctx, 1);
    const i256_ty = mlir.oraIntegerTypeCreate(mlir_ctx, 256);
    const eu_ty = mlir.oraErrorUnionTypeGet(mlir_ctx, i256_ty);

    const helper_attrs = [_]mlir.MlirNamedAttribute{
        namedAttr(mlir_ctx, "sym_name", mlir.oraStringAttrCreate(mlir_ctx, stringRef("singleIterTryWhileWriter"))),
        namedAttr(mlir_ctx, "ora.effect", mlir.oraStringAttrCreate(mlir_ctx, stringRef("writes"))),
        namedAttr(mlir_ctx, "ora.write_slots", mlir.oraArrayAttrCreate(mlir_ctx, 1, &[_]mlir.MlirAttribute{
            mlir.oraStringAttrCreate(mlir_ctx, stringRef("counter")),
        })),
    };
    const helper = mlir.oraFuncFuncOpCreate(mlir_ctx, loc, &helper_attrs, helper_attrs.len, &[_]mlir.MlirType{}, &[_]mlir.MlirLocation{}, 0);
    const body = mlir.oraFuncOpGetBodyBlock(helper);

    const true_attr = mlir.oraIntegerAttrCreateI64FromType(i1_ty, 1);
    const false_attr = mlir.oraIntegerAttrCreateI64FromType(i1_ty, 0);
    const true_op = mlir.oraArithConstantOpCreate(mlir_ctx, loc, i1_ty, true_attr);
    const false_op = mlir.oraArithConstantOpCreate(mlir_ctx, loc, i1_ty, false_attr);
    mlir.oraBlockAppendOwnedOperation(body, true_op);
    mlir.oraBlockAppendOwnedOperation(body, false_op);

    const try_stmt = mlir.oraTryStmtOpCreate(mlir_ctx, loc, &[_]mlir.MlirType{}, 0);
    const try_block = mlir.oraTryStmtOpGetTryBlock(try_stmt);
    const catch_block = mlir.oraTryStmtOpGetCatchBlock(try_stmt);

    const init_vals = [_]mlir.MlirValue{mlir.oraOperationGetResult(true_op, 0)};
    const result_types = [_]mlir.MlirType{i1_ty};
    const while_op = mlir.oraScfWhileOpCreate(mlir_ctx, loc, &init_vals, init_vals.len, &result_types, result_types.len);
    const before_block = mlir.oraScfWhileOpGetBeforeBlock(while_op);
    const after_block = mlir.oraScfWhileOpGetAfterBlock(while_op);
    _ = mlir.mlirBlockAddArgument(before_block, i1_ty, loc);
    _ = mlir.mlirBlockAddArgument(after_block, i1_ty, loc);
    const before_arg = mlir.oraBlockGetArgument(before_block, 0);
    mlir.oraBlockAppendOwnedOperation(before_block, mlir.oraScfConditionOpCreate(
        mlir_ctx,
        loc,
        before_arg,
        &[_]mlir.MlirValue{before_arg},
        1,
    ));

    const err_attr = mlir.oraIntegerAttrCreateI64FromType(i256_ty, 1);
    const err_val_op = mlir.oraArithConstantOpCreate(mlir_ctx, loc, i256_ty, err_attr);
    mlir.oraBlockAppendOwnedOperation(after_block, err_val_op);
    const err_op = mlir.oraErrorErrOpCreate(mlir_ctx, loc, mlir.oraOperationGetResult(err_val_op, 0), eu_ty);
    mlir.oraBlockAppendOwnedOperation(after_block, err_op);
    const unwrap_op = mlir.oraErrorUnwrapOpCreate(mlir_ctx, loc, mlir.oraOperationGetResult(err_op, 0), i256_ty);
    mlir.oraBlockAppendOwnedOperation(after_block, unwrap_op);
    mlir.oraBlockAppendOwnedOperation(after_block, mlir.oraScfYieldOpCreate(
        mlir_ctx,
        loc,
        &[_]mlir.MlirValue{mlir.oraOperationGetResult(false_op, 0)},
        1,
    ));

    mlir.oraBlockAppendOwnedOperation(try_block, while_op);
    const try_store_attr = mlir.oraIntegerAttrCreateI64FromType(i256_ty, 7);
    const try_store_op = mlir.oraArithConstantOpCreate(mlir_ctx, loc, i256_ty, try_store_attr);
    mlir.oraBlockAppendOwnedOperation(try_block, try_store_op);
    mlir.oraBlockAppendOwnedOperation(try_block, mlir.oraSStoreOpCreate(
        mlir_ctx,
        loc,
        mlir.oraOperationGetResult(try_store_op, 0),
        stringRef("counter"),
    ));
    mlir.oraBlockAppendOwnedOperation(try_block, mlir.oraYieldOpCreate(mlir_ctx, loc, &[_]mlir.MlirValue{}, 0));

    const catch_store_attr = mlir.oraIntegerAttrCreateI64FromType(i256_ty, 33);
    const catch_store_op = mlir.oraArithConstantOpCreate(mlir_ctx, loc, i256_ty, catch_store_attr);
    mlir.oraBlockAppendOwnedOperation(catch_block, catch_store_op);
    mlir.oraBlockAppendOwnedOperation(catch_block, mlir.oraSStoreOpCreate(
        mlir_ctx,
        loc,
        mlir.oraOperationGetResult(catch_store_op, 0),
        stringRef("counter"),
    ));
    mlir.oraBlockAppendOwnedOperation(catch_block, mlir.oraYieldOpCreate(mlir_ctx, loc, &[_]mlir.MlirValue{}, 0));

    mlir.oraBlockAppendOwnedOperation(body, try_stmt);
    mlir.oraBlockAppendOwnedOperation(body, mlir.oraReturnOpCreate(mlir_ctx, loc, &[_]mlir.MlirValue{}, 0));

    try encoder.registerFunctionOperation(helper);

    const call = mlir.oraFuncCallOpCreate(mlir_ctx, loc, stringRef("singleIterTryWhileWriter"), &[_]mlir.MlirValue{}, 0, &[_]mlir.MlirType{}, 0);
    _ = try encoder.encodeOperation(call);

    try testing.expect(!encoder.isDegraded());
    const counter = encoder.global_map.get("counter").?;
    const expected = z3.Z3_mk_numeral(z3_ctx.ctx, "33", z3.Z3_mk_bv_sort(z3_ctx.ctx, 256));

    var solver = try Solver.init(&z3_ctx, testing.allocator);
    defer solver.deinit();
    solver.assert(z3.Z3_mk_not(z3_ctx.ctx, z3.Z3_mk_eq(z3_ctx.ctx, counter, expected)));
    try testing.expectEqual(@as(z3.Z3_lbool, z3.Z3_L_FALSE), solver.check());
}

test "func.call summary with finite two-iteration scf.for in ora.try_stmt preserves state exactly" {
    var z3_ctx = try Context.init(testing.allocator);
    defer z3_ctx.deinit();

    var encoder = Encoder.init(&z3_ctx, testing.allocator);
    defer encoder.deinit();

    const mlir_ctx = mlir.oraContextCreate();
    defer mlir.oraContextDestroy(mlir_ctx);
    loadAllDialects(mlir_ctx);
    _ = mlir.oraDialectRegister(mlir_ctx);

    const loc = mlir.oraLocationUnknownGet(mlir_ctx);
    const index_ty = mlir.oraIndexTypeCreate(mlir_ctx);
    const i256_ty = mlir.oraIntegerTypeCreate(mlir_ctx, 256);
    const eu_ty = mlir.oraErrorUnionTypeGet(mlir_ctx, i256_ty);

    const helper_attrs = [_]mlir.MlirNamedAttribute{
        namedAttr(mlir_ctx, "sym_name", mlir.oraStringAttrCreate(mlir_ctx, stringRef("twoIterTryForWriter"))),
        namedAttr(mlir_ctx, "ora.effect", mlir.oraStringAttrCreate(mlir_ctx, stringRef("writes"))),
        namedAttr(mlir_ctx, "ora.write_slots", mlir.oraArrayAttrCreate(mlir_ctx, 1, &[_]mlir.MlirAttribute{
            mlir.oraStringAttrCreate(mlir_ctx, stringRef("counter")),
        })),
    };
    const helper = mlir.oraFuncFuncOpCreate(mlir_ctx, loc, &helper_attrs, helper_attrs.len, &[_]mlir.MlirType{}, &[_]mlir.MlirLocation{}, 0);
    const body = mlir.oraFuncOpGetBodyBlock(helper);

    const try_stmt = mlir.oraTryStmtOpCreate(mlir_ctx, loc, &[_]mlir.MlirType{}, 0);
    const try_block = mlir.oraTryStmtOpGetTryBlock(try_stmt);
    const catch_block = mlir.oraTryStmtOpGetCatchBlock(try_stmt);

    const zero_attr = mlir.oraIntegerAttrCreateI64FromType(index_ty, 0);
    const one_attr = mlir.oraIntegerAttrCreateI64FromType(index_ty, 1);
    const two_attr = mlir.oraIntegerAttrCreateI64FromType(index_ty, 2);
    const zero_op = mlir.oraArithConstantOpCreate(mlir_ctx, loc, index_ty, zero_attr);
    const one_op = mlir.oraArithConstantOpCreate(mlir_ctx, loc, index_ty, one_attr);
    const two_op = mlir.oraArithConstantOpCreate(mlir_ctx, loc, index_ty, two_attr);
    mlir.oraBlockAppendOwnedOperation(try_block, zero_op);
    mlir.oraBlockAppendOwnedOperation(try_block, one_op);
    mlir.oraBlockAppendOwnedOperation(try_block, two_op);

    const loop = mlir.oraScfForOpCreate(
        mlir_ctx,
        loc,
        mlir.oraOperationGetResult(zero_op, 0),
        mlir.oraOperationGetResult(two_op, 0),
        mlir.oraOperationGetResult(one_op, 0),
        &[_]mlir.MlirValue{},
        0,
        false,
    );
    const loop_body = mlir.oraScfForOpGetBodyBlock(loop);
    const iv = mlir.oraBlockGetArgument(loop_body, 0);
    const cmp_op = mlir.oraArithCmpIOpCreate(mlir_ctx, loc, 0, iv, mlir.oraOperationGetResult(one_op, 0)); // eq
    mlir.oraBlockAppendOwnedOperation(loop_body, cmp_op);
    const no_results = [_]mlir.MlirType{};
    const if_op = mlir.oraScfIfOpCreate(mlir_ctx, loc, mlir.oraOperationGetResult(cmp_op, 0), &no_results, no_results.len, true);
    const then_block = mlir.oraScfIfOpGetThenBlock(if_op);
    const else_block = mlir.oraScfIfOpGetElseBlock(if_op);
    const err_attr = mlir.oraIntegerAttrCreateI64FromType(i256_ty, 1);
    const err_val_op = mlir.oraArithConstantOpCreate(mlir_ctx, loc, i256_ty, err_attr);
    mlir.oraBlockAppendOwnedOperation(then_block, err_val_op);
    const err_op = mlir.oraErrorErrOpCreate(mlir_ctx, loc, mlir.oraOperationGetResult(err_val_op, 0), eu_ty);
    mlir.oraBlockAppendOwnedOperation(then_block, err_op);
    const unwrap_op = mlir.oraErrorUnwrapOpCreate(mlir_ctx, loc, mlir.oraOperationGetResult(err_op, 0), i256_ty);
    mlir.oraBlockAppendOwnedOperation(then_block, unwrap_op);
    mlir.oraBlockAppendOwnedOperation(then_block, mlir.oraScfYieldOpCreate(mlir_ctx, loc, &[_]mlir.MlirValue{}, 0));
    mlir.oraBlockAppendOwnedOperation(else_block, mlir.oraScfYieldOpCreate(mlir_ctx, loc, &[_]mlir.MlirValue{}, 0));
    mlir.oraBlockAppendOwnedOperation(loop_body, if_op);
    mlir.oraBlockAppendOwnedOperation(loop_body, mlir.oraScfYieldOpCreate(mlir_ctx, loc, &[_]mlir.MlirValue{}, 0));
    mlir.oraBlockAppendOwnedOperation(try_block, loop);

    const try_store_attr = mlir.oraIntegerAttrCreateI64FromType(i256_ty, 7);
    const try_store_op = mlir.oraArithConstantOpCreate(mlir_ctx, loc, i256_ty, try_store_attr);
    mlir.oraBlockAppendOwnedOperation(try_block, try_store_op);
    mlir.oraBlockAppendOwnedOperation(try_block, mlir.oraSStoreOpCreate(
        mlir_ctx,
        loc,
        mlir.oraOperationGetResult(try_store_op, 0),
        stringRef("counter"),
    ));
    mlir.oraBlockAppendOwnedOperation(try_block, mlir.oraYieldOpCreate(mlir_ctx, loc, &[_]mlir.MlirValue{}, 0));

    const catch_store_attr = mlir.oraIntegerAttrCreateI64FromType(i256_ty, 33);
    const catch_store_op = mlir.oraArithConstantOpCreate(mlir_ctx, loc, i256_ty, catch_store_attr);
    mlir.oraBlockAppendOwnedOperation(catch_block, catch_store_op);
    mlir.oraBlockAppendOwnedOperation(catch_block, mlir.oraSStoreOpCreate(
        mlir_ctx,
        loc,
        mlir.oraOperationGetResult(catch_store_op, 0),
        stringRef("counter"),
    ));
    mlir.oraBlockAppendOwnedOperation(catch_block, mlir.oraYieldOpCreate(mlir_ctx, loc, &[_]mlir.MlirValue{}, 0));

    mlir.oraBlockAppendOwnedOperation(body, try_stmt);
    mlir.oraBlockAppendOwnedOperation(body, mlir.oraReturnOpCreate(mlir_ctx, loc, &[_]mlir.MlirValue{}, 0));

    try encoder.registerFunctionOperation(helper);

    const call = mlir.oraFuncCallOpCreate(mlir_ctx, loc, stringRef("twoIterTryForWriter"), &[_]mlir.MlirValue{}, 0, &[_]mlir.MlirType{}, 0);
    _ = try encoder.encodeOperation(call);

    try testing.expect(!encoder.isDegraded());
    const counter = encoder.global_map.get("counter").?;
    const expected = z3.Z3_mk_numeral(z3_ctx.ctx, "33", z3.Z3_mk_bv_sort(z3_ctx.ctx, 256));

    var solver = try Solver.init(&z3_ctx, testing.allocator);
    defer solver.deinit();
    solver.assert(z3.Z3_mk_not(z3_ctx.ctx, z3.Z3_mk_eq(z3_ctx.ctx, counter, expected)));
    try testing.expectEqual(@as(z3.Z3_lbool, z3.Z3_L_FALSE), solver.check());
}

test "func.call summary with nested caught ora.try_stmt preserves outer state exactly" {
    var z3_ctx = try Context.init(testing.allocator);
    defer z3_ctx.deinit();

    var encoder = Encoder.init(&z3_ctx, testing.allocator);
    defer encoder.deinit();

    const mlir_ctx = mlir.oraContextCreate();
    defer mlir.oraContextDestroy(mlir_ctx);
    loadAllDialects(mlir_ctx);
    _ = mlir.oraDialectRegister(mlir_ctx);

    const loc = mlir.oraLocationUnknownGet(mlir_ctx);
    const i256_ty = mlir.oraIntegerTypeCreate(mlir_ctx, 256);
    const eu_ty = mlir.oraErrorUnionTypeGet(mlir_ctx, i256_ty);

    const helper_attrs = [_]mlir.MlirNamedAttribute{
        namedAttr(mlir_ctx, "sym_name", mlir.oraStringAttrCreate(mlir_ctx, stringRef("nestedCaughtTryWriter"))),
        namedAttr(mlir_ctx, "ora.effect", mlir.oraStringAttrCreate(mlir_ctx, stringRef("writes"))),
        namedAttr(mlir_ctx, "ora.write_slots", mlir.oraArrayAttrCreate(mlir_ctx, 1, &[_]mlir.MlirAttribute{
            mlir.oraStringAttrCreate(mlir_ctx, stringRef("counter")),
        })),
    };
    const helper = mlir.oraFuncFuncOpCreate(mlir_ctx, loc, &helper_attrs, helper_attrs.len, &[_]mlir.MlirType{}, &[_]mlir.MlirLocation{}, 0);
    const body = mlir.oraFuncOpGetBodyBlock(helper);

    const outer_try = mlir.oraTryStmtOpCreate(mlir_ctx, loc, &[_]mlir.MlirType{}, 0);
    const outer_try_block = mlir.oraTryStmtOpGetTryBlock(outer_try);
    const outer_catch_block = mlir.oraTryStmtOpGetCatchBlock(outer_try);

    const inner_try = mlir.oraTryStmtOpCreate(mlir_ctx, loc, &[_]mlir.MlirType{i256_ty}, 1);
    const inner_try_block = mlir.oraTryStmtOpGetTryBlock(inner_try);
    const inner_catch_block = mlir.oraTryStmtOpGetCatchBlock(inner_try);

    const err_attr = mlir.oraIntegerAttrCreateI64FromType(i256_ty, 1);
    const err_val_op = mlir.oraArithConstantOpCreate(mlir_ctx, loc, i256_ty, err_attr);
    mlir.oraBlockAppendOwnedOperation(inner_try_block, err_val_op);
    const err_op = mlir.oraErrorErrOpCreate(mlir_ctx, loc, mlir.oraOperationGetResult(err_val_op, 0), eu_ty);
    mlir.oraBlockAppendOwnedOperation(inner_try_block, err_op);
    const unwrap_op = mlir.oraErrorUnwrapOpCreate(mlir_ctx, loc, mlir.oraOperationGetResult(err_op, 0), i256_ty);
    mlir.oraBlockAppendOwnedOperation(inner_try_block, unwrap_op);

    const inner_catch_attr = mlir.oraIntegerAttrCreateI64FromType(i256_ty, 5);
    const inner_catch_op = mlir.oraArithConstantOpCreate(mlir_ctx, loc, i256_ty, inner_catch_attr);
    mlir.oraBlockAppendOwnedOperation(inner_catch_block, inner_catch_op);
    mlir.oraBlockAppendOwnedOperation(inner_catch_block, mlir.oraYieldOpCreate(
        mlir_ctx,
        loc,
        &[_]mlir.MlirValue{mlir.oraOperationGetResult(inner_catch_op, 0)},
        1,
    ));

    mlir.oraBlockAppendOwnedOperation(outer_try_block, inner_try);
    const try_store_attr = mlir.oraIntegerAttrCreateI64FromType(i256_ty, 7);
    const try_store_op = mlir.oraArithConstantOpCreate(mlir_ctx, loc, i256_ty, try_store_attr);
    mlir.oraBlockAppendOwnedOperation(outer_try_block, try_store_op);
    mlir.oraBlockAppendOwnedOperation(outer_try_block, mlir.oraSStoreOpCreate(
        mlir_ctx,
        loc,
        mlir.oraOperationGetResult(try_store_op, 0),
        stringRef("counter"),
    ));
    mlir.oraBlockAppendOwnedOperation(outer_try_block, mlir.oraYieldOpCreate(mlir_ctx, loc, &[_]mlir.MlirValue{}, 0));

    const outer_catch_attr = mlir.oraIntegerAttrCreateI64FromType(i256_ty, 33);
    const outer_catch_op = mlir.oraArithConstantOpCreate(mlir_ctx, loc, i256_ty, outer_catch_attr);
    mlir.oraBlockAppendOwnedOperation(outer_catch_block, outer_catch_op);
    mlir.oraBlockAppendOwnedOperation(outer_catch_block, mlir.oraSStoreOpCreate(
        mlir_ctx,
        loc,
        mlir.oraOperationGetResult(outer_catch_op, 0),
        stringRef("counter"),
    ));
    mlir.oraBlockAppendOwnedOperation(outer_catch_block, mlir.oraYieldOpCreate(mlir_ctx, loc, &[_]mlir.MlirValue{}, 0));

    mlir.oraBlockAppendOwnedOperation(body, outer_try);
    mlir.oraBlockAppendOwnedOperation(body, mlir.oraReturnOpCreate(mlir_ctx, loc, &[_]mlir.MlirValue{}, 0));

    try encoder.registerFunctionOperation(helper);

    const call = mlir.oraFuncCallOpCreate(mlir_ctx, loc, stringRef("nestedCaughtTryWriter"), &[_]mlir.MlirValue{}, 0, &[_]mlir.MlirType{}, 0);
    _ = try encoder.encodeOperation(call);

    try testing.expect(!encoder.isDegraded());
    const counter = encoder.global_map.get("counter").?;
    const expected = z3.Z3_mk_numeral(z3_ctx.ctx, "7", z3.Z3_mk_bv_sort(z3_ctx.ctx, 256));

    var solver = try Solver.init(&z3_ctx, testing.allocator);
    defer solver.deinit();
    solver.assert(z3.Z3_mk_not(z3_ctx.ctx, z3.Z3_mk_eq(z3_ctx.ctx, counter, expected)));
    try testing.expectEqual(@as(z3.Z3_lbool, z3.Z3_L_FALSE), solver.check());
}

test "func.call summary with dead catch-capable try branch preserves state exactly" {
    var z3_ctx = try Context.init(testing.allocator);
    defer z3_ctx.deinit();

    var encoder = Encoder.init(&z3_ctx, testing.allocator);
    defer encoder.deinit();

    const mlir_ctx = mlir.oraContextCreate();
    defer mlir.oraContextDestroy(mlir_ctx);
    loadAllDialects(mlir_ctx);
    _ = mlir.oraDialectRegister(mlir_ctx);

    const loc = mlir.oraLocationUnknownGet(mlir_ctx);
    const i1_ty = mlir.oraIntegerTypeCreate(mlir_ctx, 1);
    const i256_ty = mlir.oraIntegerTypeCreate(mlir_ctx, 256);
    const eu_ty = mlir.oraErrorUnionTypeGet(mlir_ctx, i256_ty);

    const helper_attrs = [_]mlir.MlirNamedAttribute{
        namedAttr(mlir_ctx, "sym_name", mlir.oraStringAttrCreate(mlir_ctx, stringRef("deadCatchTryWriter"))),
        namedAttr(mlir_ctx, "ora.effect", mlir.oraStringAttrCreate(mlir_ctx, stringRef("writes"))),
        namedAttr(mlir_ctx, "ora.write_slots", mlir.oraArrayAttrCreate(mlir_ctx, 1, &[_]mlir.MlirAttribute{
            mlir.oraStringAttrCreate(mlir_ctx, stringRef("counter")),
        })),
    };
    const helper = mlir.oraFuncFuncOpCreate(mlir_ctx, loc, &helper_attrs, helper_attrs.len, &[_]mlir.MlirType{}, &[_]mlir.MlirLocation{}, 0);
    const body = mlir.oraFuncOpGetBodyBlock(helper);

    const try_stmt = mlir.oraTryStmtOpCreate(mlir_ctx, loc, &[_]mlir.MlirType{}, 0);
    const try_block = mlir.oraTryStmtOpGetTryBlock(try_stmt);
    const catch_block = mlir.oraTryStmtOpGetCatchBlock(try_stmt);

    const false_attr = mlir.oraIntegerAttrCreateI64FromType(i1_ty, 0);
    const false_op = mlir.oraArithConstantOpCreate(mlir_ctx, loc, i1_ty, false_attr);
    mlir.oraBlockAppendOwnedOperation(try_block, false_op);

    const if_op = mlir.oraScfIfOpCreate(
        mlir_ctx,
        loc,
        mlir.oraOperationGetResult(false_op, 0),
        &[_]mlir.MlirType{},
        0,
        true,
    );
    const then_block = mlir.oraScfIfOpGetThenBlock(if_op);
    const else_block = mlir.oraScfIfOpGetElseBlock(if_op);

    const err_attr = mlir.oraIntegerAttrCreateI64FromType(i256_ty, 1);
    const err_val_op = mlir.oraArithConstantOpCreate(mlir_ctx, loc, i256_ty, err_attr);
    mlir.oraBlockAppendOwnedOperation(then_block, err_val_op);
    const err_op = mlir.oraErrorErrOpCreate(mlir_ctx, loc, mlir.oraOperationGetResult(err_val_op, 0), eu_ty);
    mlir.oraBlockAppendOwnedOperation(then_block, err_op);
    const unwrap_op = mlir.oraErrorUnwrapOpCreate(mlir_ctx, loc, mlir.oraOperationGetResult(err_op, 0), i256_ty);
    mlir.oraBlockAppendOwnedOperation(then_block, unwrap_op);
    mlir.oraBlockAppendOwnedOperation(then_block, mlir.oraScfYieldOpCreate(mlir_ctx, loc, &[_]mlir.MlirValue{}, 0));

    const seven_attr = mlir.oraIntegerAttrCreateI64FromType(i256_ty, 7);
    const seven_op = mlir.oraArithConstantOpCreate(mlir_ctx, loc, i256_ty, seven_attr);
    mlir.oraBlockAppendOwnedOperation(else_block, seven_op);
    mlir.oraBlockAppendOwnedOperation(else_block, mlir.oraSStoreOpCreate(
        mlir_ctx,
        loc,
        mlir.oraOperationGetResult(seven_op, 0),
        stringRef("counter"),
    ));
    mlir.oraBlockAppendOwnedOperation(else_block, mlir.oraScfYieldOpCreate(mlir_ctx, loc, &[_]mlir.MlirValue{}, 0));

    mlir.oraBlockAppendOwnedOperation(try_block, if_op);
    mlir.oraBlockAppendOwnedOperation(try_block, mlir.oraYieldOpCreate(mlir_ctx, loc, &[_]mlir.MlirValue{}, 0));

    const catch_store_attr = mlir.oraIntegerAttrCreateI64FromType(i256_ty, 9);
    const catch_store_op = mlir.oraArithConstantOpCreate(mlir_ctx, loc, i256_ty, catch_store_attr);
    mlir.oraBlockAppendOwnedOperation(catch_block, catch_store_op);
    mlir.oraBlockAppendOwnedOperation(catch_block, mlir.oraSStoreOpCreate(
        mlir_ctx,
        loc,
        mlir.oraOperationGetResult(catch_store_op, 0),
        stringRef("counter"),
    ));
    mlir.oraBlockAppendOwnedOperation(catch_block, mlir.oraYieldOpCreate(mlir_ctx, loc, &[_]mlir.MlirValue{}, 0));

    mlir.oraBlockAppendOwnedOperation(body, try_stmt);
    mlir.oraBlockAppendOwnedOperation(body, mlir.oraReturnOpCreate(mlir_ctx, loc, &[_]mlir.MlirValue{}, 0));

    try encoder.registerFunctionOperation(helper);

    const call = mlir.oraFuncCallOpCreate(mlir_ctx, loc, stringRef("deadCatchTryWriter"), &[_]mlir.MlirValue{}, 0, &[_]mlir.MlirType{}, 0);
    _ = try encoder.encodeOperation(call);

    try testing.expect(!encoder.isDegraded());
    const counter = encoder.global_map.get("counter").?;
    const expected = z3.Z3_mk_numeral(z3_ctx.ctx, "7", z3.Z3_mk_bv_sort(z3_ctx.ctx, 256));

    var solver = try Solver.init(&z3_ctx, testing.allocator);
    defer solver.deinit();
    solver.assert(z3.Z3_mk_not(z3_ctx.ctx, z3.Z3_mk_eq(z3_ctx.ctx, counter, expected)));
    try testing.expectEqual(@as(z3.Z3_lbool, z3.Z3_L_FALSE), solver.check());
}

test "func.call summary with dead catch-capable switch branch preserves state exactly" {
    var z3_ctx = try Context.init(testing.allocator);
    defer z3_ctx.deinit();

    var encoder = Encoder.init(&z3_ctx, testing.allocator);
    defer encoder.deinit();

    const mlir_ctx = mlir.oraContextCreate();
    defer mlir.oraContextDestroy(mlir_ctx);
    loadAllDialects(mlir_ctx);
    _ = mlir.oraDialectRegister(mlir_ctx);

    const loc = mlir.oraLocationUnknownGet(mlir_ctx);
    const i256_ty = mlir.oraIntegerTypeCreate(mlir_ctx, 256);
    const eu_ty = mlir.oraErrorUnionTypeGet(mlir_ctx, i256_ty);

    const helper_attrs = [_]mlir.MlirNamedAttribute{
        namedAttr(mlir_ctx, "sym_name", mlir.oraStringAttrCreate(mlir_ctx, stringRef("deadCatchSwitchWriter"))),
        namedAttr(mlir_ctx, "ora.effect", mlir.oraStringAttrCreate(mlir_ctx, stringRef("writes"))),
        namedAttr(mlir_ctx, "ora.write_slots", mlir.oraArrayAttrCreate(mlir_ctx, 1, &[_]mlir.MlirAttribute{
            mlir.oraStringAttrCreate(mlir_ctx, stringRef("counter")),
        })),
    };
    const helper = mlir.oraFuncFuncOpCreate(mlir_ctx, loc, &helper_attrs, helper_attrs.len, &[_]mlir.MlirType{}, &[_]mlir.MlirLocation{}, 0);
    const body = mlir.oraFuncOpGetBodyBlock(helper);

    const tag_attr = mlir.oraIntegerAttrCreateI64FromType(i256_ty, 2);
    const tag_op = mlir.oraArithConstantOpCreate(mlir_ctx, loc, i256_ty, tag_attr);
    mlir.oraBlockAppendOwnedOperation(body, tag_op);

    const switch_op = mlir.oraSwitchOpCreateWithCases(
        mlir_ctx,
        loc,
        mlir.oraOperationGetResult(tag_op, 0),
        &[_]mlir.MlirType{},
        0,
        2,
    );
    const case0_block = mlir.oraSwitchOpGetCaseBlock(switch_op, 0);
    const default_block = mlir.oraSwitchOpGetCaseBlock(switch_op, 1);

    const err_attr = mlir.oraIntegerAttrCreateI64FromType(i256_ty, 1);
    const err_val_op = mlir.oraArithConstantOpCreate(mlir_ctx, loc, i256_ty, err_attr);
    mlir.oraBlockAppendOwnedOperation(case0_block, err_val_op);
    const err_op = mlir.oraErrorErrOpCreate(mlir_ctx, loc, mlir.oraOperationGetResult(err_val_op, 0), eu_ty);
    mlir.oraBlockAppendOwnedOperation(case0_block, err_op);
    const unwrap_op = mlir.oraErrorUnwrapOpCreate(mlir_ctx, loc, mlir.oraOperationGetResult(err_op, 0), i256_ty);
    mlir.oraBlockAppendOwnedOperation(case0_block, unwrap_op);
    mlir.oraBlockAppendOwnedOperation(case0_block, mlir.oraYieldOpCreate(mlir_ctx, loc, &[_]mlir.MlirValue{}, 0));

    const seven_attr = mlir.oraIntegerAttrCreateI64FromType(i256_ty, 7);
    const seven_op = mlir.oraArithConstantOpCreate(mlir_ctx, loc, i256_ty, seven_attr);
    mlir.oraBlockAppendOwnedOperation(default_block, seven_op);
    mlir.oraBlockAppendOwnedOperation(default_block, mlir.oraSStoreOpCreate(
        mlir_ctx,
        loc,
        mlir.oraOperationGetResult(seven_op, 0),
        stringRef("counter"),
    ));
    mlir.oraBlockAppendOwnedOperation(default_block, mlir.oraYieldOpCreate(mlir_ctx, loc, &[_]mlir.MlirValue{}, 0));

    var case_values = [_]i64{ 1, 0 };
    var range_starts = [_]i64{ 0, 0 };
    var range_ends = [_]i64{ 0, 0 };
    var case_kinds = [_]i64{ 0, 2 };
    mlir.oraSwitchOpSetCasePatterns(switch_op, &case_values, &range_starts, &range_ends, &case_kinds, 1, 2);

    mlir.oraBlockAppendOwnedOperation(body, switch_op);
    mlir.oraBlockAppendOwnedOperation(body, mlir.oraReturnOpCreate(mlir_ctx, loc, &[_]mlir.MlirValue{}, 0));

    try encoder.registerFunctionOperation(helper);

    const call = mlir.oraFuncCallOpCreate(mlir_ctx, loc, stringRef("deadCatchSwitchWriter"), &[_]mlir.MlirValue{}, 0, &[_]mlir.MlirType{}, 0);
    _ = try encoder.encodeOperation(call);

    try testing.expect(!encoder.isDegraded());
    const counter = encoder.global_map.get("counter").?;
    const expected = z3.Z3_mk_numeral(z3_ctx.ctx, "7", z3.Z3_mk_bv_sort(z3_ctx.ctx, 256));

    var solver = try Solver.init(&z3_ctx, testing.allocator);
    defer solver.deinit();
    solver.assert(z3.Z3_mk_not(z3_ctx.ctx, z3.Z3_mk_eq(z3_ctx.ctx, counter, expected)));
    try testing.expectEqual(@as(z3.Z3_lbool, z3.Z3_L_FALSE), solver.check());
}

test "func.call summary with dead catch-capable conditional return branch preserves state exactly" {
    var z3_ctx = try Context.init(testing.allocator);
    defer z3_ctx.deinit();

    var encoder = Encoder.init(&z3_ctx, testing.allocator);
    defer encoder.deinit();

    const mlir_ctx = mlir.oraContextCreate();
    defer mlir.oraContextDestroy(mlir_ctx);
    loadAllDialects(mlir_ctx);
    _ = mlir.oraDialectRegister(mlir_ctx);

    const loc = mlir.oraLocationUnknownGet(mlir_ctx);
    const i1_ty = mlir.oraIntegerTypeCreate(mlir_ctx, 1);
    const i256_ty = mlir.oraIntegerTypeCreate(mlir_ctx, 256);
    const eu_ty = mlir.oraErrorUnionTypeGet(mlir_ctx, i256_ty);

    const helper_attrs = [_]mlir.MlirNamedAttribute{
        namedAttr(mlir_ctx, "sym_name", mlir.oraStringAttrCreate(mlir_ctx, stringRef("deadCatchConditionalWriter"))),
        namedAttr(mlir_ctx, "ora.effect", mlir.oraStringAttrCreate(mlir_ctx, stringRef("writes"))),
        namedAttr(mlir_ctx, "ora.write_slots", mlir.oraArrayAttrCreate(mlir_ctx, 1, &[_]mlir.MlirAttribute{
            mlir.oraStringAttrCreate(mlir_ctx, stringRef("counter")),
        })),
    };
    const helper = mlir.oraFuncFuncOpCreate(mlir_ctx, loc, &helper_attrs, helper_attrs.len, &[_]mlir.MlirType{}, &[_]mlir.MlirLocation{}, 0);
    const body = mlir.oraFuncOpGetBodyBlock(helper);

    const false_attr = mlir.oraIntegerAttrCreateI64FromType(i1_ty, 0);
    const false_op = mlir.oraArithConstantOpCreate(mlir_ctx, loc, i1_ty, false_attr);
    mlir.oraBlockAppendOwnedOperation(body, false_op);

    const conditional_ret = mlir.oraConditionalReturnOpCreate(mlir_ctx, loc, mlir.oraOperationGetResult(false_op, 0));
    const then_block = mlir.oraConditionalReturnOpGetThenBlock(conditional_ret);
    const else_block = mlir.oraConditionalReturnOpGetElseBlock(conditional_ret);

    const err_attr = mlir.oraIntegerAttrCreateI64FromType(i256_ty, 1);
    const err_val_op = mlir.oraArithConstantOpCreate(mlir_ctx, loc, i256_ty, err_attr);
    mlir.oraBlockAppendOwnedOperation(then_block, err_val_op);
    const err_op = mlir.oraErrorErrOpCreate(mlir_ctx, loc, mlir.oraOperationGetResult(err_val_op, 0), eu_ty);
    mlir.oraBlockAppendOwnedOperation(then_block, err_op);
    const unwrap_op = mlir.oraErrorUnwrapOpCreate(mlir_ctx, loc, mlir.oraOperationGetResult(err_op, 0), i256_ty);
    mlir.oraBlockAppendOwnedOperation(then_block, unwrap_op);
    mlir.oraBlockAppendOwnedOperation(then_block, mlir.oraReturnOpCreate(mlir_ctx, loc, &[_]mlir.MlirValue{}, 0));

    mlir.oraBlockAppendOwnedOperation(else_block, mlir.oraYieldOpCreate(mlir_ctx, loc, &[_]mlir.MlirValue{}, 0));

    const seven_attr = mlir.oraIntegerAttrCreateI64FromType(i256_ty, 7);
    const seven_op = mlir.oraArithConstantOpCreate(mlir_ctx, loc, i256_ty, seven_attr);
    mlir.oraBlockAppendOwnedOperation(body, conditional_ret);
    mlir.oraBlockAppendOwnedOperation(body, seven_op);
    mlir.oraBlockAppendOwnedOperation(body, mlir.oraSStoreOpCreate(
        mlir_ctx,
        loc,
        mlir.oraOperationGetResult(seven_op, 0),
        stringRef("counter"),
    ));
    mlir.oraBlockAppendOwnedOperation(body, mlir.oraReturnOpCreate(mlir_ctx, loc, &[_]mlir.MlirValue{}, 0));

    try encoder.registerFunctionOperation(helper);

    const call = mlir.oraFuncCallOpCreate(mlir_ctx, loc, stringRef("deadCatchConditionalWriter"), &[_]mlir.MlirValue{}, 0, &[_]mlir.MlirType{}, 0);
    _ = try encoder.encodeOperation(call);

    try testing.expect(!encoder.isDegraded());
    const counter = encoder.global_map.get("counter").?;
    const expected = z3.Z3_mk_numeral(z3_ctx.ctx, "7", z3.Z3_mk_bv_sort(z3_ctx.ctx, 256));

    var solver = try Solver.init(&z3_ctx, testing.allocator);
    defer solver.deinit();
    solver.assert(z3.Z3_mk_not(z3_ctx.ctx, z3.Z3_mk_eq(z3_ctx.ctx, counter, expected)));
    try testing.expectEqual(@as(z3.Z3_lbool, z3.Z3_L_FALSE), solver.check());
}

test "func.call summary with zero-iteration scf.while preserves state exactly" {
    var z3_ctx = try Context.init(testing.allocator);
    defer z3_ctx.deinit();

    var encoder = Encoder.init(&z3_ctx, testing.allocator);
    defer encoder.deinit();

    const mlir_ctx = mlir.oraContextCreate();
    defer mlir.oraContextDestroy(mlir_ctx);
    loadAllDialects(mlir_ctx);
    _ = mlir.oraDialectRegister(mlir_ctx);

    const loc = mlir.oraLocationUnknownGet(mlir_ctx);
    const i1_ty = mlir.oraIntegerTypeCreate(mlir_ctx, 1);
    const i256_ty = mlir.oraIntegerTypeCreate(mlir_ctx, 256);

    const helper_attrs = [_]mlir.MlirNamedAttribute{
        namedAttr(mlir_ctx, "sym_name", mlir.oraStringAttrCreate(mlir_ctx, stringRef("zeroIterWhileWriter"))),
        namedAttr(mlir_ctx, "ora.effect", mlir.oraStringAttrCreate(mlir_ctx, stringRef("writes"))),
        namedAttr(mlir_ctx, "ora.write_slots", mlir.oraArrayAttrCreate(mlir_ctx, 1, &[_]mlir.MlirAttribute{
            mlir.oraStringAttrCreate(mlir_ctx, stringRef("counter")),
        })),
    };
    const helper = mlir.oraFuncFuncOpCreate(mlir_ctx, loc, &helper_attrs, helper_attrs.len, &[_]mlir.MlirType{}, &[_]mlir.MlirLocation{}, 0);
    const body = mlir.oraFuncOpGetBodyBlock(helper);

    const while_op = mlir.oraScfWhileOpCreate(mlir_ctx, loc, &[_]mlir.MlirValue{}, 0, &[_]mlir.MlirType{}, 0);
    const before_block = mlir.oraScfWhileOpGetBeforeBlock(while_op);
    const after_block = mlir.oraScfWhileOpGetAfterBlock(while_op);

    const false_attr = mlir.oraIntegerAttrCreateI64FromType(i1_ty, 0);
    const false_op = mlir.oraArithConstantOpCreate(mlir_ctx, loc, i1_ty, false_attr);
    const false_val = mlir.oraOperationGetResult(false_op, 0);
    mlir.oraBlockAppendOwnedOperation(before_block, false_op);
    mlir.oraBlockAppendOwnedOperation(before_block, mlir.oraScfConditionOpCreate(
        mlir_ctx,
        loc,
        false_val,
        &[_]mlir.MlirValue{},
        0,
    ));

    const store_attr = mlir.oraIntegerAttrCreateI64FromType(i256_ty, 9);
    const store_op = mlir.oraArithConstantOpCreate(mlir_ctx, loc, i256_ty, store_attr);
    mlir.oraBlockAppendOwnedOperation(after_block, store_op);
    mlir.oraBlockAppendOwnedOperation(after_block, mlir.oraSStoreOpCreate(
        mlir_ctx,
        loc,
        mlir.oraOperationGetResult(store_op, 0),
        stringRef("counter"),
    ));
    mlir.oraBlockAppendOwnedOperation(after_block, mlir.oraScfYieldOpCreate(
        mlir_ctx,
        loc,
        &[_]mlir.MlirValue{},
        0,
    ));

    mlir.oraBlockAppendOwnedOperation(body, while_op);
    mlir.oraBlockAppendOwnedOperation(body, mlir.oraReturnOpCreate(mlir_ctx, loc, &[_]mlir.MlirValue{}, 0));

    try encoder.registerFunctionOperation(helper);

    const counter_sort = z3.Z3_mk_bv_sort(z3_ctx.ctx, 256);
    const pre_counter = z3.Z3_mk_const(z3_ctx.ctx, z3.Z3_mk_string_symbol(z3_ctx.ctx, "pre_counter"), counter_sort);
    try encoder.global_map.put(try testing.allocator.dupe(u8, "counter"), pre_counter);

    const call = mlir.oraFuncCallOpCreate(mlir_ctx, loc, stringRef("zeroIterWhileWriter"), &[_]mlir.MlirValue{}, 0, &[_]mlir.MlirType{}, 0);
    _ = try encoder.encodeOperation(call);

    try testing.expect(!encoder.isDegraded());
    const post_counter = encoder.global_map.get("counter").?;

    var solver = try Solver.init(&z3_ctx, testing.allocator);
    defer solver.deinit();
    solver.assert(z3.Z3_mk_not(z3_ctx.ctx, z3.Z3_mk_eq(z3_ctx.ctx, post_counter, pre_counter)));
    try testing.expectEqual(@as(z3.Z3_lbool, z3.Z3_L_FALSE), solver.check());
}

test "func.call summary with single-iteration scf.while preserves state exactly" {
    var z3_ctx = try Context.init(testing.allocator);
    defer z3_ctx.deinit();

    var encoder = Encoder.init(&z3_ctx, testing.allocator);
    defer encoder.deinit();

    const mlir_ctx = mlir.oraContextCreate();
    defer mlir.oraContextDestroy(mlir_ctx);
    loadAllDialects(mlir_ctx);
    _ = mlir.oraDialectRegister(mlir_ctx);

    const loc = mlir.oraLocationUnknownGet(mlir_ctx);
    const i1_ty = mlir.oraIntegerTypeCreate(mlir_ctx, 1);
    const i256_ty = mlir.oraIntegerTypeCreate(mlir_ctx, 256);

    const helper_attrs = [_]mlir.MlirNamedAttribute{
        namedAttr(mlir_ctx, "sym_name", mlir.oraStringAttrCreate(mlir_ctx, stringRef("singleIterWhileWriter"))),
        namedAttr(mlir_ctx, "ora.effect", mlir.oraStringAttrCreate(mlir_ctx, stringRef("writes"))),
        namedAttr(mlir_ctx, "ora.write_slots", mlir.oraArrayAttrCreate(mlir_ctx, 1, &[_]mlir.MlirAttribute{
            mlir.oraStringAttrCreate(mlir_ctx, stringRef("counter")),
        })),
    };
    const helper = mlir.oraFuncFuncOpCreate(mlir_ctx, loc, &helper_attrs, helper_attrs.len, &[_]mlir.MlirType{}, &[_]mlir.MlirLocation{}, 0);
    const body = mlir.oraFuncOpGetBodyBlock(helper);

    const true_attr = mlir.oraIntegerAttrCreateI64FromType(i1_ty, 1);
    const false_attr = mlir.oraIntegerAttrCreateI64FromType(i1_ty, 0);
    const true_op = mlir.oraArithConstantOpCreate(mlir_ctx, loc, i1_ty, true_attr);
    const false_op = mlir.oraArithConstantOpCreate(mlir_ctx, loc, i1_ty, false_attr);
    mlir.oraBlockAppendOwnedOperation(body, true_op);
    mlir.oraBlockAppendOwnedOperation(body, false_op);

    const init_vals = [_]mlir.MlirValue{mlir.oraOperationGetResult(true_op, 0)};
    const result_types = [_]mlir.MlirType{i1_ty};
    const while_op = mlir.oraScfWhileOpCreate(mlir_ctx, loc, &init_vals, init_vals.len, &result_types, result_types.len);
    const before_block = mlir.oraScfWhileOpGetBeforeBlock(while_op);
    const after_block = mlir.oraScfWhileOpGetAfterBlock(while_op);
    _ = mlir.mlirBlockAddArgument(before_block, i1_ty, loc);
    _ = mlir.mlirBlockAddArgument(after_block, i1_ty, loc);
    const before_arg = mlir.oraBlockGetArgument(before_block, 0);
    mlir.oraBlockAppendOwnedOperation(before_block, mlir.oraScfConditionOpCreate(
        mlir_ctx,
        loc,
        before_arg,
        &[_]mlir.MlirValue{before_arg},
        1,
    ));

    const store_attr = mlir.oraIntegerAttrCreateI64FromType(i256_ty, 9);
    const store_op = mlir.oraArithConstantOpCreate(mlir_ctx, loc, i256_ty, store_attr);
    mlir.oraBlockAppendOwnedOperation(after_block, store_op);
    mlir.oraBlockAppendOwnedOperation(after_block, mlir.oraSStoreOpCreate(
        mlir_ctx,
        loc,
        mlir.oraOperationGetResult(store_op, 0),
        stringRef("counter"),
    ));
    mlir.oraBlockAppendOwnedOperation(after_block, mlir.oraScfYieldOpCreate(
        mlir_ctx,
        loc,
        &[_]mlir.MlirValue{mlir.oraOperationGetResult(false_op, 0)},
        1,
    ));

    mlir.oraBlockAppendOwnedOperation(body, while_op);
    mlir.oraBlockAppendOwnedOperation(body, mlir.oraReturnOpCreate(mlir_ctx, loc, &[_]mlir.MlirValue{}, 0));

    try encoder.registerFunctionOperation(helper);

    const call = mlir.oraFuncCallOpCreate(mlir_ctx, loc, stringRef("singleIterWhileWriter"), &[_]mlir.MlirValue{}, 0, &[_]mlir.MlirType{}, 0);
    _ = try encoder.encodeOperation(call);

    try testing.expect(!encoder.isDegraded());
    const counter = encoder.global_map.get("counter").?;
    const expected = z3.Z3_mk_numeral(z3_ctx.ctx, "9", z3.Z3_mk_bv_sort(z3_ctx.ctx, 256));

    var solver = try Solver.init(&z3_ctx, testing.allocator);
    defer solver.deinit();
    solver.assert(z3.Z3_mk_not(z3_ctx.ctx, z3.Z3_mk_eq(z3_ctx.ctx, counter, expected)));
    try testing.expectEqual(@as(z3.Z3_lbool, z3.Z3_L_FALSE), solver.check());
}

test "func.call summary with finite two-iteration scf.while preserves state exactly" {
    var z3_ctx = try Context.init(testing.allocator);
    defer z3_ctx.deinit();

    var encoder = Encoder.init(&z3_ctx, testing.allocator);
    defer encoder.deinit();

    const mlir_ctx = mlir.oraContextCreate();
    defer mlir.oraContextDestroy(mlir_ctx);
    loadAllDialects(mlir_ctx);
    _ = mlir.oraDialectRegister(mlir_ctx);

    const loc = mlir.oraLocationUnknownGet(mlir_ctx);
    const i1_ty = mlir.oraIntegerTypeCreate(mlir_ctx, 1);
    const i256_ty = mlir.oraIntegerTypeCreate(mlir_ctx, 256);

    const helper_attrs = [_]mlir.MlirNamedAttribute{
        namedAttr(mlir_ctx, "sym_name", mlir.oraStringAttrCreate(mlir_ctx, stringRef("twoIterWhileWriter"))),
        namedAttr(mlir_ctx, "ora.effect", mlir.oraStringAttrCreate(mlir_ctx, stringRef("writes"))),
        namedAttr(mlir_ctx, "ora.write_slots", mlir.oraArrayAttrCreate(mlir_ctx, 1, &[_]mlir.MlirAttribute{
            mlir.oraStringAttrCreate(mlir_ctx, stringRef("counter")),
        })),
    };
    const helper = mlir.oraFuncFuncOpCreate(mlir_ctx, loc, &helper_attrs, helper_attrs.len, &[_]mlir.MlirType{}, &[_]mlir.MlirLocation{}, 0);
    const body = mlir.oraFuncOpGetBodyBlock(helper);

    const true_attr = mlir.oraIntegerAttrCreateI64FromType(i1_ty, 1);
    const false_attr = mlir.oraIntegerAttrCreateI64FromType(i1_ty, 0);
    const true_op = mlir.oraArithConstantOpCreate(mlir_ctx, loc, i1_ty, true_attr);
    const false_op = mlir.oraArithConstantOpCreate(mlir_ctx, loc, i1_ty, false_attr);
    mlir.oraBlockAppendOwnedOperation(body, true_op);
    mlir.oraBlockAppendOwnedOperation(body, false_op);

    const init_vals = [_]mlir.MlirValue{
        mlir.oraOperationGetResult(true_op, 0),
        mlir.oraOperationGetResult(true_op, 0),
    };
    const result_types = [_]mlir.MlirType{ i1_ty, i1_ty };
    const while_op = mlir.oraScfWhileOpCreate(mlir_ctx, loc, &init_vals, init_vals.len, &result_types, result_types.len);
    const before_block = mlir.oraScfWhileOpGetBeforeBlock(while_op);
    const after_block = mlir.oraScfWhileOpGetAfterBlock(while_op);
    _ = mlir.mlirBlockAddArgument(before_block, i1_ty, loc);
    _ = mlir.mlirBlockAddArgument(before_block, i1_ty, loc);
    _ = mlir.mlirBlockAddArgument(after_block, i1_ty, loc);
    _ = mlir.mlirBlockAddArgument(after_block, i1_ty, loc);
    const before_flag = mlir.oraBlockGetArgument(before_block, 0);
    const before_next = mlir.oraBlockGetArgument(before_block, 1);
    mlir.oraBlockAppendOwnedOperation(before_block, mlir.oraScfConditionOpCreate(
        mlir_ctx,
        loc,
        before_flag,
        &[_]mlir.MlirValue{ before_flag, before_next },
        2,
    ));

    const store_attr = mlir.oraIntegerAttrCreateI64FromType(i256_ty, 9);
    const store_op = mlir.oraArithConstantOpCreate(mlir_ctx, loc, i256_ty, store_attr);
    mlir.oraBlockAppendOwnedOperation(after_block, store_op);
    mlir.oraBlockAppendOwnedOperation(after_block, mlir.oraSStoreOpCreate(
        mlir_ctx,
        loc,
        mlir.oraOperationGetResult(store_op, 0),
        stringRef("counter"),
    ));
    const after_next = mlir.oraBlockGetArgument(after_block, 1);
    mlir.oraBlockAppendOwnedOperation(after_block, mlir.oraScfYieldOpCreate(
        mlir_ctx,
        loc,
        &[_]mlir.MlirValue{ after_next, mlir.oraOperationGetResult(false_op, 0) },
        2,
    ));

    mlir.oraBlockAppendOwnedOperation(body, while_op);
    mlir.oraBlockAppendOwnedOperation(body, mlir.oraReturnOpCreate(mlir_ctx, loc, &[_]mlir.MlirValue{}, 0));

    try encoder.registerFunctionOperation(helper);

    const call = mlir.oraFuncCallOpCreate(mlir_ctx, loc, stringRef("twoIterWhileWriter"), &[_]mlir.MlirValue{}, 0, &[_]mlir.MlirType{}, 0);
    _ = try encoder.encodeOperation(call);

    try testing.expect(!encoder.isDegraded());
    const counter = encoder.global_map.get("counter").?;
    const expected = z3.Z3_mk_numeral(z3_ctx.ctx, "9", z3.Z3_mk_bv_sort(z3_ctx.ctx, 256));

    var solver = try Solver.init(&z3_ctx, testing.allocator);
    defer solver.deinit();
    solver.assert(z3.Z3_mk_not(z3_ctx.ctx, z3.Z3_mk_eq(z3_ctx.ctx, counter, expected)));
    try testing.expectEqual(@as(z3.Z3_lbool, z3.Z3_L_FALSE), solver.check());
}

test "direct finite two-iteration scf.while result encodes exactly" {
    var z3_ctx = try Context.init(testing.allocator);
    defer z3_ctx.deinit();

    var encoder = Encoder.init(&z3_ctx, testing.allocator);
    defer encoder.deinit();

    const mlir_ctx = mlir.oraContextCreate();
    defer mlir.oraContextDestroy(mlir_ctx);
    loadAllDialects(mlir_ctx);
    _ = mlir.oraDialectRegister(mlir_ctx);

    const loc = mlir.oraLocationUnknownGet(mlir_ctx);
    const i1_ty = mlir.oraIntegerTypeCreate(mlir_ctx, 1);

    const true_attr = mlir.oraIntegerAttrCreateI64FromType(i1_ty, 1);
    const false_attr = mlir.oraIntegerAttrCreateI64FromType(i1_ty, 0);
    const true_op = mlir.oraArithConstantOpCreate(mlir_ctx, loc, i1_ty, true_attr);
    const false_op = mlir.oraArithConstantOpCreate(mlir_ctx, loc, i1_ty, false_attr);
    const true_val = mlir.oraOperationGetResult(true_op, 0);
    const false_val = mlir.oraOperationGetResult(false_op, 0);

    const init_vals = [_]mlir.MlirValue{ true_val, true_val };
    const result_types = [_]mlir.MlirType{ i1_ty, i1_ty };
    const while_op = mlir.oraScfWhileOpCreate(mlir_ctx, loc, &init_vals, init_vals.len, &result_types, result_types.len);
    const before_block = mlir.oraScfWhileOpGetBeforeBlock(while_op);
    const after_block = mlir.oraScfWhileOpGetAfterBlock(while_op);
    _ = mlir.mlirBlockAddArgument(before_block, i1_ty, loc);
    _ = mlir.mlirBlockAddArgument(before_block, i1_ty, loc);
    _ = mlir.mlirBlockAddArgument(after_block, i1_ty, loc);
    _ = mlir.mlirBlockAddArgument(after_block, i1_ty, loc);
    const before_flag = mlir.oraBlockGetArgument(before_block, 0);
    const before_next = mlir.oraBlockGetArgument(before_block, 1);
    mlir.oraBlockAppendOwnedOperation(before_block, mlir.oraScfConditionOpCreate(
        mlir_ctx,
        loc,
        before_flag,
        &[_]mlir.MlirValue{ before_flag, before_next },
        2,
    ));

    const after_next = mlir.oraBlockGetArgument(after_block, 1);
    mlir.oraBlockAppendOwnedOperation(after_block, false_op);
    mlir.oraBlockAppendOwnedOperation(after_block, mlir.oraScfYieldOpCreate(
        mlir_ctx,
        loc,
        &[_]mlir.MlirValue{ after_next, false_val },
        2,
    ));

    const encoded = try encoder.encodeValue(mlir.oraOperationGetResult(while_op, 0));
    const expected = encoder.encodeBoolConstant(false);
    try testing.expect(!encoder.isDegraded());

    var solver = try Solver.init(&z3_ctx, testing.allocator);
    defer solver.deinit();
    solver.assert(z3.Z3_mk_not(z3_ctx.ctx, z3.Z3_mk_eq(z3_ctx.ctx, encoded, expected)));
    try testing.expectEqual(@as(z3.Z3_lbool, z3.Z3_L_FALSE), solver.check());
}

test "direct finite five-iteration scf.while result encodes exactly" {
    var z3_ctx = try Context.init(testing.allocator);
    defer z3_ctx.deinit();

    var encoder = Encoder.init(&z3_ctx, testing.allocator);
    defer encoder.deinit();

    const mlir_ctx = mlir.oraContextCreate();
    defer mlir.oraContextDestroy(mlir_ctx);
    loadAllDialects(mlir_ctx);
    _ = mlir.oraDialectRegister(mlir_ctx);

    const loc = mlir.oraLocationUnknownGet(mlir_ctx);
    const i1_ty = mlir.oraIntegerTypeCreate(mlir_ctx, 1);
    const i256_ty = mlir.oraIntegerTypeCreate(mlir_ctx, 256);

    const false_attr = mlir.oraIntegerAttrCreateI64FromType(i1_ty, 0);
    const true_attr = mlir.oraIntegerAttrCreateI64FromType(i1_ty, 1);
    const zero_attr = mlir.oraIntegerAttrCreateI64FromType(i256_ty, 0);
    const false_op = mlir.oraArithConstantOpCreate(mlir_ctx, loc, i1_ty, false_attr);
    const true_op = mlir.oraArithConstantOpCreate(mlir_ctx, loc, i1_ty, true_attr);
    const zero_op = mlir.oraArithConstantOpCreate(mlir_ctx, loc, i256_ty, zero_attr);
    const false_val = mlir.oraOperationGetResult(false_op, 0);
    const true_val = mlir.oraOperationGetResult(true_op, 0);
    const zero_val = mlir.oraOperationGetResult(zero_op, 0);

    const init_vals = [_]mlir.MlirValue{
        true_val,
        true_val,
        true_val,
        true_val,
        true_val,
        zero_val,
    };
    const result_types = [_]mlir.MlirType{ i1_ty, i1_ty, i1_ty, i1_ty, i1_ty, i256_ty };
    const while_op = mlir.oraScfWhileOpCreate(mlir_ctx, loc, &init_vals, init_vals.len, &result_types, result_types.len);

    const before_block = mlir.oraScfWhileOpGetBeforeBlock(while_op);
    const after_block = mlir.oraScfWhileOpGetAfterBlock(while_op);
    _ = mlir.mlirBlockAddArgument(before_block, i1_ty, loc);
    _ = mlir.mlirBlockAddArgument(before_block, i1_ty, loc);
    _ = mlir.mlirBlockAddArgument(before_block, i1_ty, loc);
    _ = mlir.mlirBlockAddArgument(before_block, i1_ty, loc);
    _ = mlir.mlirBlockAddArgument(before_block, i1_ty, loc);
    _ = mlir.mlirBlockAddArgument(before_block, i256_ty, loc);
    _ = mlir.mlirBlockAddArgument(after_block, i1_ty, loc);
    _ = mlir.mlirBlockAddArgument(after_block, i1_ty, loc);
    _ = mlir.mlirBlockAddArgument(after_block, i1_ty, loc);
    _ = mlir.mlirBlockAddArgument(after_block, i1_ty, loc);
    _ = mlir.mlirBlockAddArgument(after_block, i1_ty, loc);
    _ = mlir.mlirBlockAddArgument(after_block, i256_ty, loc);

    const before_f0 = mlir.oraBlockGetArgument(before_block, 0);
    const before_f1 = mlir.oraBlockGetArgument(before_block, 1);
    const before_f2 = mlir.oraBlockGetArgument(before_block, 2);
    const before_f3 = mlir.oraBlockGetArgument(before_block, 3);
    const before_f4 = mlir.oraBlockGetArgument(before_block, 4);
    const before_sum = mlir.oraBlockGetArgument(before_block, 5);
    mlir.oraBlockAppendOwnedOperation(before_block, mlir.oraScfConditionOpCreate(
        mlir_ctx,
        loc,
        before_f0,
        &[_]mlir.MlirValue{ before_f0, before_f1, before_f2, before_f3, before_f4, before_sum },
        6,
    ));

    const after_f1 = mlir.oraBlockGetArgument(after_block, 1);
    const after_f2 = mlir.oraBlockGetArgument(after_block, 2);
    const after_f3 = mlir.oraBlockGetArgument(after_block, 3);
    const after_f4 = mlir.oraBlockGetArgument(after_block, 4);
    const after_sum = mlir.oraBlockGetArgument(after_block, 5);
    const one_attr = mlir.oraIntegerAttrCreateI64FromType(i256_ty, 1);
    const one_op = mlir.oraArithConstantOpCreate(mlir_ctx, loc, i256_ty, one_attr);
    mlir.oraBlockAppendOwnedOperation(after_block, one_op);
    const next_sum = mlir.oraArithAddIOpCreate(
        mlir_ctx,
        loc,
        after_sum,
        mlir.oraOperationGetResult(one_op, 0),
    );
    mlir.oraBlockAppendOwnedOperation(after_block, next_sum);
    mlir.oraBlockAppendOwnedOperation(after_block, false_op);
    mlir.oraBlockAppendOwnedOperation(after_block, mlir.oraScfYieldOpCreate(
        mlir_ctx,
        loc,
        &[_]mlir.MlirValue{
            after_f1,
            after_f2,
            after_f3,
            after_f4,
            false_val,
            mlir.oraOperationGetResult(next_sum, 0),
        },
        6,
    ));

    const encoded = try encoder.encodeValue(mlir.oraOperationGetResult(while_op, 5));
    const expected = try encoder.encodeIntegerConstant(5, 256);
    try testing.expect(!encoder.isDegraded());

    var solver = try Solver.init(&z3_ctx, testing.allocator);
    defer solver.deinit();
    solver.assert(z3.Z3_mk_not(z3_ctx.ctx, z3.Z3_mk_eq(z3_ctx.ctx, encoded, expected)));
    try testing.expectEqual(@as(z3.Z3_lbool, z3.Z3_L_FALSE), solver.check());
}

test "direct finite ten-iteration scf.while result encodes exactly" {
    var z3_ctx = try Context.init(testing.allocator);
    defer z3_ctx.deinit();

    var encoder = Encoder.init(&z3_ctx, testing.allocator);
    defer encoder.deinit();

    const mlir_ctx = mlir.oraContextCreate();
    defer mlir.oraContextDestroy(mlir_ctx);
    loadAllDialects(mlir_ctx);
    _ = mlir.oraDialectRegister(mlir_ctx);

    const loc = mlir.oraLocationUnknownGet(mlir_ctx);
    const i256_ty = mlir.oraIntegerTypeCreate(mlir_ctx, 256);

    const zero_attr = mlir.oraIntegerAttrCreateI64FromType(i256_ty, 0);
    const one_attr = mlir.oraIntegerAttrCreateI64FromType(i256_ty, 1);
    const ten_attr = mlir.oraIntegerAttrCreateI64FromType(i256_ty, 10);

    const zero_op = mlir.oraArithConstantOpCreate(mlir_ctx, loc, i256_ty, zero_attr);
    const one_op = mlir.oraArithConstantOpCreate(mlir_ctx, loc, i256_ty, one_attr);
    const ten_op = mlir.oraArithConstantOpCreate(mlir_ctx, loc, i256_ty, ten_attr);

    const init_vals = [_]mlir.MlirValue{mlir.oraOperationGetResult(zero_op, 0)};
    const result_types = [_]mlir.MlirType{i256_ty};
    const while_op = mlir.oraScfWhileOpCreate(mlir_ctx, loc, &init_vals, init_vals.len, &result_types, result_types.len);
    const before_block = mlir.oraScfWhileOpGetBeforeBlock(while_op);
    const after_block = mlir.oraScfWhileOpGetAfterBlock(while_op);
    _ = mlir.mlirBlockAddArgument(before_block, i256_ty, loc);
    _ = mlir.mlirBlockAddArgument(after_block, i256_ty, loc);

    const before_count = mlir.oraBlockGetArgument(before_block, 0);
    const cmp_op = mlir.oraArithCmpIOpCreate(mlir_ctx, loc, 6, before_count, mlir.oraOperationGetResult(ten_op, 0)); // ult
    mlir.oraBlockAppendOwnedOperation(before_block, cmp_op);
    mlir.oraBlockAppendOwnedOperation(before_block, mlir.oraScfConditionOpCreate(
        mlir_ctx,
        loc,
        mlir.oraOperationGetResult(cmp_op, 0),
        &[_]mlir.MlirValue{before_count},
        1,
    ));

    const after_count = mlir.oraBlockGetArgument(after_block, 0);
    const next_count = mlir.oraArithAddIOpCreate(mlir_ctx, loc, after_count, mlir.oraOperationGetResult(one_op, 0));
    mlir.oraBlockAppendOwnedOperation(after_block, next_count);
    mlir.oraBlockAppendOwnedOperation(after_block, mlir.oraScfYieldOpCreate(
        mlir_ctx,
        loc,
        &[_]mlir.MlirValue{mlir.oraOperationGetResult(next_count, 0)},
        1,
    ));

    _ = try encoder.encodeOperation(zero_op);
    _ = try encoder.encodeOperation(one_op);
    _ = try encoder.encodeOperation(ten_op);

    const encoded = try encoder.encodeValue(mlir.oraOperationGetResult(while_op, 0));
    try testing.expect(!encoder.isDegraded());

    var solver = try Solver.init(&z3_ctx, testing.allocator);
    defer solver.deinit();
    const expected = try encoder.encodeIntegerConstant(10, 256);
    solver.assert(z3.Z3_mk_not(z3_ctx.ctx, z3.Z3_mk_eq(z3_ctx.ctx, encoded, expected)));
    try testing.expectEqual(@as(z3.Z3_lbool, z3.Z3_L_FALSE), solver.check());
}

test "direct canonical unsigned positive-delta scf.while result encodes exactly" {
    var z3_ctx = try Context.init(testing.allocator);
    defer z3_ctx.deinit();

    var encoder = Encoder.init(&z3_ctx, testing.allocator);
    defer encoder.deinit();

    const mlir_ctx = mlir.oraContextCreate();
    defer mlir.oraContextDestroy(mlir_ctx);
    loadAllDialects(mlir_ctx);
    _ = mlir.oraDialectRegister(mlir_ctx);

    const loc = mlir.oraLocationUnknownGet(mlir_ctx);
    const i256_ty = mlir.oraIntegerTypeCreate(mlir_ctx, 256);

    const init_attr = mlir.oraIntegerAttrCreateI64FromType(i256_ty, 1);
    const delta_attr = mlir.oraIntegerAttrCreateI64FromType(i256_ty, 3);
    const init_op = mlir.oraArithConstantOpCreate(mlir_ctx, loc, i256_ty, init_attr);
    const delta_op = mlir.oraArithConstantOpCreate(mlir_ctx, loc, i256_ty, delta_attr);
    const ub_op = mlir.oraVariablePlaceholderOpCreate(mlir_ctx, loc, stringRef("whileIncUb"), i256_ty);

    const init_vals = [_]mlir.MlirValue{mlir.oraOperationGetResult(init_op, 0)};
    const result_types = [_]mlir.MlirType{i256_ty};
    const while_op = mlir.oraScfWhileOpCreate(mlir_ctx, loc, &init_vals, init_vals.len, &result_types, result_types.len);
    const before_block = mlir.oraScfWhileOpGetBeforeBlock(while_op);
    const after_block = mlir.oraScfWhileOpGetAfterBlock(while_op);
    _ = mlir.mlirBlockAddArgument(before_block, i256_ty, loc);
    _ = mlir.mlirBlockAddArgument(after_block, i256_ty, loc);

    const before_arg = mlir.oraBlockGetArgument(before_block, 0);
    const cmp_op = mlir.oraArithCmpIOpCreate(mlir_ctx, loc, 6, before_arg, mlir.oraOperationGetResult(ub_op, 0)); // ult
    mlir.oraBlockAppendOwnedOperation(before_block, cmp_op);
    mlir.oraBlockAppendOwnedOperation(before_block, mlir.oraScfConditionOpCreate(
        mlir_ctx,
        loc,
        mlir.oraOperationGetResult(cmp_op, 0),
        &[_]mlir.MlirValue{before_arg},
        1,
    ));

    const after_arg = mlir.oraBlockGetArgument(after_block, 0);
    const next_op = mlir.oraArithAddIOpCreate(mlir_ctx, loc, after_arg, mlir.oraOperationGetResult(delta_op, 0));
    mlir.oraBlockAppendOwnedOperation(after_block, next_op);
    mlir.oraBlockAppendOwnedOperation(after_block, mlir.oraScfYieldOpCreate(
        mlir_ctx,
        loc,
        &[_]mlir.MlirValue{mlir.oraOperationGetResult(next_op, 0)},
        1,
    ));

    const encoded = try encoder.encodeValue(mlir.oraOperationGetResult(while_op, 0));
    try testing.expect(!encoder.isDegraded());

    const init_ast = try encoder.encodeOperation(init_op);
    const ub_ast = try encoder.encodeOperation(ub_op);
    const delta_ast = try encoder.encodeOperation(delta_op);
    const zero = try encoder.encodeIntegerConstant(0, 256);
    const one = try encoder.encodeIntegerConstant(1, 256);
    const ub_le_init = z3.Z3_mk_bvule(z3_ctx.ctx, ub_ast, init_ast);
    const distance = z3.Z3_mk_ite(
        z3_ctx.ctx,
        ub_le_init,
        zero,
        z3.Z3_mk_bv_sub(z3_ctx.ctx, ub_ast, init_ast),
    );
    const distance_is_zero = z3.Z3_mk_eq(z3_ctx.ctx, distance, zero);
    const step_count = z3.Z3_mk_ite(
        z3_ctx.ctx,
        distance_is_zero,
        zero,
        z3.Z3_mk_bv_add(
            z3_ctx.ctx,
            z3.Z3_mk_bv_udiv(z3_ctx.ctx, z3.Z3_mk_bv_sub(z3_ctx.ctx, distance, one), delta_ast),
            one,
        ),
    );
    const total_delta = try encoder.encodeArithmeticOp(.Mul, step_count, delta_ast);
    const expected = try encoder.encodeArithmeticOp(.Add, init_ast, total_delta);

    var solver = try Solver.init(&z3_ctx, testing.allocator);
    defer solver.deinit();
    solver.assert(z3.Z3_mk_not(z3_ctx.ctx, z3.Z3_mk_eq(z3_ctx.ctx, encoded, expected)));
    try testing.expectEqual(@as(z3.Z3_lbool, z3.Z3_L_FALSE), solver.check());
}

test "direct canonical unsigned swapped-compare scf.while result encodes exactly" {
    var z3_ctx = try Context.init(testing.allocator);
    defer z3_ctx.deinit();

    var encoder = Encoder.init(&z3_ctx, testing.allocator);
    defer encoder.deinit();

    const mlir_ctx = mlir.oraContextCreate();
    defer mlir.oraContextDestroy(mlir_ctx);
    loadAllDialects(mlir_ctx);
    _ = mlir.oraDialectRegister(mlir_ctx);

    const loc = mlir.oraLocationUnknownGet(mlir_ctx);
    const i256_ty = mlir.oraIntegerTypeCreate(mlir_ctx, 256);

    const init_attr = mlir.oraIntegerAttrCreateI64FromType(i256_ty, 1);
    const delta_attr = mlir.oraIntegerAttrCreateI64FromType(i256_ty, 3);
    const init_op = mlir.oraArithConstantOpCreate(mlir_ctx, loc, i256_ty, init_attr);
    const delta_op = mlir.oraArithConstantOpCreate(mlir_ctx, loc, i256_ty, delta_attr);
    const ub_op = mlir.oraVariablePlaceholderOpCreate(mlir_ctx, loc, stringRef("whileIncUbSwapped"), i256_ty);

    const init_vals = [_]mlir.MlirValue{mlir.oraOperationGetResult(init_op, 0)};
    const result_types = [_]mlir.MlirType{i256_ty};
    const while_op = mlir.oraScfWhileOpCreate(mlir_ctx, loc, &init_vals, init_vals.len, &result_types, result_types.len);
    const before_block = mlir.oraScfWhileOpGetBeforeBlock(while_op);
    const after_block = mlir.oraScfWhileOpGetAfterBlock(while_op);
    _ = mlir.mlirBlockAddArgument(before_block, i256_ty, loc);
    _ = mlir.mlirBlockAddArgument(after_block, i256_ty, loc);

    const before_arg = mlir.oraBlockGetArgument(before_block, 0);
    const cmp_op = mlir.oraArithCmpIOpCreate(mlir_ctx, loc, 8, mlir.oraOperationGetResult(ub_op, 0), before_arg); // bound > current
    mlir.oraBlockAppendOwnedOperation(before_block, cmp_op);
    mlir.oraBlockAppendOwnedOperation(before_block, mlir.oraScfConditionOpCreate(
        mlir_ctx,
        loc,
        mlir.oraOperationGetResult(cmp_op, 0),
        &[_]mlir.MlirValue{before_arg},
        1,
    ));

    const after_arg = mlir.oraBlockGetArgument(after_block, 0);
    const next_op = mlir.oraArithAddIOpCreate(mlir_ctx, loc, after_arg, mlir.oraOperationGetResult(delta_op, 0));
    mlir.oraBlockAppendOwnedOperation(after_block, next_op);
    mlir.oraBlockAppendOwnedOperation(after_block, mlir.oraScfYieldOpCreate(
        mlir_ctx,
        loc,
        &[_]mlir.MlirValue{mlir.oraOperationGetResult(next_op, 0)},
        1,
    ));

    _ = try encoder.encodeOperation(init_op);
    _ = try encoder.encodeOperation(delta_op);

    const encoded = try encoder.encodeValue(mlir.oraOperationGetResult(while_op, 0));
    try testing.expect(!encoder.isDegraded());

    const init_ast = try encoder.encodeValue(mlir.oraOperationGetResult(init_op, 0));
    const bound_ast = try encoder.encodeValue(mlir.oraOperationGetResult(ub_op, 0));
    const delta_ast = try encoder.encodeValue(mlir.oraOperationGetResult(delta_op, 0));
    const sort = z3.Z3_get_sort(z3_ctx.ctx, init_ast);
    const zero = z3.Z3_mk_unsigned_int64(z3_ctx.ctx, 0, sort);
    const upper_le_lower = z3.Z3_mk_bvule(z3_ctx.ctx, bound_ast, init_ast);
    const raw_distance = z3.Z3_mk_bv_sub(z3_ctx.ctx, bound_ast, init_ast);
    const distance = z3.Z3_mk_ite(z3_ctx.ctx, upper_le_lower, zero, raw_distance);
    const one = z3.Z3_mk_unsigned_int64(z3_ctx.ctx, 1, sort);
    const delta_const = z3.Z3_mk_unsigned_int64(z3_ctx.ctx, 3, sort);
    const distance_is_zero = z3.Z3_mk_eq(z3_ctx.ctx, distance, zero);
    const adjusted_distance = z3.Z3_mk_bv_sub(z3_ctx.ctx, distance, one);
    const quotient = z3.Z3_mk_bv_udiv(z3_ctx.ctx, adjusted_distance, delta_const);
    const step_count = z3.Z3_mk_ite(
        z3_ctx.ctx,
        distance_is_zero,
        zero,
        z3.Z3_mk_bv_add(z3_ctx.ctx, quotient, one),
    );
    const total_delta = try encoder.encodeArithmeticOp(.Mul, step_count, delta_ast);
    const expected = try encoder.encodeArithmeticOp(.Add, init_ast, total_delta);

    var solver = try Solver.init(&z3_ctx, testing.allocator);
    defer solver.deinit();
    solver.assert(z3.Z3_mk_not(z3_ctx.ctx, z3.Z3_mk_eq(z3_ctx.ctx, encoded, expected)));
    try testing.expectEqual(@as(z3.Z3_lbool, z3.Z3_L_FALSE), solver.check());
}

test "func.call summary with canonical unsigned positive-delta no-write scf.while preserves state exactly" {
    var z3_ctx = try Context.init(testing.allocator);
    defer z3_ctx.deinit();

    var encoder = Encoder.init(&z3_ctx, testing.allocator);
    defer encoder.deinit();

    const mlir_ctx = mlir.oraContextCreate();
    defer mlir.oraContextDestroy(mlir_ctx);
    loadAllDialects(mlir_ctx);
    _ = mlir.oraDialectRegister(mlir_ctx);

    const loc = mlir.oraLocationUnknownGet(mlir_ctx);
    const i256_ty = mlir.oraIntegerTypeCreate(mlir_ctx, 256);

    const helper_attrs = [_]mlir.MlirNamedAttribute{
        namedAttr(mlir_ctx, "sym_name", mlir.oraStringAttrCreate(mlir_ctx, stringRef("symbolicNoWriteWhileDelta"))),
    };
    const helper_param_types = [_]mlir.MlirType{i256_ty};
    const helper_param_locs = [_]mlir.MlirLocation{loc};
    const helper = mlir.oraFuncFuncOpCreate(mlir_ctx, loc, &helper_attrs, helper_attrs.len, &helper_param_types, &helper_param_locs, helper_param_types.len);
    const body = mlir.oraFuncOpGetBodyBlock(helper);

    const init_attr = mlir.oraIntegerAttrCreateI64FromType(i256_ty, 1);
    const delta_attr = mlir.oraIntegerAttrCreateI64FromType(i256_ty, 3);
    const init_op = mlir.oraArithConstantOpCreate(mlir_ctx, loc, i256_ty, init_attr);
    const delta_op = mlir.oraArithConstantOpCreate(mlir_ctx, loc, i256_ty, delta_attr);
    mlir.oraBlockAppendOwnedOperation(body, init_op);
    mlir.oraBlockAppendOwnedOperation(body, delta_op);

    const ub = mlir.oraBlockGetArgument(body, 0);
    const init_vals = [_]mlir.MlirValue{mlir.oraOperationGetResult(init_op, 0)};
    const result_types = [_]mlir.MlirType{i256_ty};
    const while_op = mlir.oraScfWhileOpCreate(mlir_ctx, loc, &init_vals, init_vals.len, &result_types, result_types.len);
    const before_block = mlir.oraScfWhileOpGetBeforeBlock(while_op);
    const after_block = mlir.oraScfWhileOpGetAfterBlock(while_op);
    _ = mlir.mlirBlockAddArgument(before_block, i256_ty, loc);
    _ = mlir.mlirBlockAddArgument(after_block, i256_ty, loc);
    const before_arg = mlir.oraBlockGetArgument(before_block, 0);
    const after_arg = mlir.oraBlockGetArgument(after_block, 0);

    const cmp_op = mlir.oraArithCmpIOpCreate(mlir_ctx, loc, 6, before_arg, ub); // ult
    mlir.oraBlockAppendOwnedOperation(before_block, cmp_op);
    mlir.oraBlockAppendOwnedOperation(before_block, mlir.oraScfConditionOpCreate(
        mlir_ctx,
        loc,
        mlir.oraOperationGetResult(cmp_op, 0),
        &[_]mlir.MlirValue{before_arg},
        1,
    ));

    const next_op = mlir.oraArithAddIOpCreate(mlir_ctx, loc, after_arg, mlir.oraOperationGetResult(delta_op, 0));
    mlir.oraBlockAppendOwnedOperation(after_block, next_op);
    mlir.oraBlockAppendOwnedOperation(after_block, mlir.oraScfYieldOpCreate(
        mlir_ctx,
        loc,
        &[_]mlir.MlirValue{mlir.oraOperationGetResult(next_op, 0)},
        1,
    ));

    mlir.oraBlockAppendOwnedOperation(body, while_op);
    mlir.oraBlockAppendOwnedOperation(body, mlir.oraReturnOpCreate(mlir_ctx, loc, &[_]mlir.MlirValue{}, 0));

    try encoder.registerFunctionOperation(helper);

    const counter_sort = z3.Z3_mk_bv_sort(z3_ctx.ctx, 256);
    const pre_counter = z3.Z3_mk_const(z3_ctx.ctx, z3.Z3_mk_string_symbol(z3_ctx.ctx, "pre_counter_symbolic_while_delta"), counter_sort);
    try encoder.global_map.put(try testing.allocator.dupe(u8, "counter"), pre_counter);

    const symbolic_ub = mlir.oraVariablePlaceholderOpCreate(mlir_ctx, loc, stringRef("ubWhileDeltaValue"), i256_ty);
    const call = mlir.oraFuncCallOpCreate(
        mlir_ctx,
        loc,
        stringRef("symbolicNoWriteWhileDelta"),
        &[_]mlir.MlirValue{mlir.oraOperationGetResult(symbolic_ub, 0)},
        1,
        &[_]mlir.MlirType{},
        0,
    );
    _ = try encoder.encodeOperation(call);

    try testing.expect(!encoder.isDegraded());
    const post_counter = encoder.global_map.get("counter").?;

    var solver = try Solver.init(&z3_ctx, testing.allocator);
    defer solver.deinit();
    solver.assert(z3.Z3_mk_not(z3_ctx.ctx, z3.Z3_mk_eq(z3_ctx.ctx, post_counter, pre_counter)));
    try testing.expectEqual(@as(z3.Z3_lbool, z3.Z3_L_FALSE), solver.check());
}

test "func.call summary with canonical unsigned swapped-compare no-write scf.while preserves state exactly" {
    var z3_ctx = try Context.init(testing.allocator);
    defer z3_ctx.deinit();

    var encoder = Encoder.init(&z3_ctx, testing.allocator);
    defer encoder.deinit();

    const mlir_ctx = mlir.oraContextCreate();
    defer mlir.oraContextDestroy(mlir_ctx);
    loadAllDialects(mlir_ctx);
    _ = mlir.oraDialectRegister(mlir_ctx);

    const loc = mlir.oraLocationUnknownGet(mlir_ctx);
    const i256_ty = mlir.oraIntegerTypeCreate(mlir_ctx, 256);

    const helper_attrs = [_]mlir.MlirNamedAttribute{
        namedAttr(mlir_ctx, "sym_name", mlir.oraStringAttrCreate(mlir_ctx, stringRef("symbolicSwappedNoWriteWhile"))),
    };
    const helper_param_types = [_]mlir.MlirType{i256_ty};
    const helper_param_locs = [_]mlir.MlirLocation{loc};
    const helper = mlir.oraFuncFuncOpCreate(mlir_ctx, loc, &helper_attrs, helper_attrs.len, &helper_param_types, &helper_param_locs, helper_param_types.len);
    const body = mlir.oraFuncOpGetBodyBlock(helper);

    const init_attr = mlir.oraIntegerAttrCreateI64FromType(i256_ty, 1);
    const delta_attr = mlir.oraIntegerAttrCreateI64FromType(i256_ty, 3);
    const init_op = mlir.oraArithConstantOpCreate(mlir_ctx, loc, i256_ty, init_attr);
    const delta_op = mlir.oraArithConstantOpCreate(mlir_ctx, loc, i256_ty, delta_attr);
    mlir.oraBlockAppendOwnedOperation(body, init_op);
    mlir.oraBlockAppendOwnedOperation(body, delta_op);

    const ub = mlir.oraBlockGetArgument(body, 0);
    const init_vals = [_]mlir.MlirValue{mlir.oraOperationGetResult(init_op, 0)};
    const result_types = [_]mlir.MlirType{i256_ty};
    const while_op = mlir.oraScfWhileOpCreate(mlir_ctx, loc, &init_vals, init_vals.len, &result_types, result_types.len);
    const before_block = mlir.oraScfWhileOpGetBeforeBlock(while_op);
    const after_block = mlir.oraScfWhileOpGetAfterBlock(while_op);
    _ = mlir.mlirBlockAddArgument(before_block, i256_ty, loc);
    _ = mlir.mlirBlockAddArgument(after_block, i256_ty, loc);

    const before_arg = mlir.oraBlockGetArgument(before_block, 0);
    const after_arg = mlir.oraBlockGetArgument(after_block, 0);
    const cmp_op = mlir.oraArithCmpIOpCreate(mlir_ctx, loc, 8, ub, before_arg); // bound > current
    mlir.oraBlockAppendOwnedOperation(before_block, cmp_op);
    mlir.oraBlockAppendOwnedOperation(before_block, mlir.oraScfConditionOpCreate(
        mlir_ctx,
        loc,
        mlir.oraOperationGetResult(cmp_op, 0),
        &[_]mlir.MlirValue{before_arg},
        1,
    ));

    const next_op = mlir.oraArithAddIOpCreate(mlir_ctx, loc, after_arg, mlir.oraOperationGetResult(delta_op, 0));
    mlir.oraBlockAppendOwnedOperation(after_block, next_op);
    mlir.oraBlockAppendOwnedOperation(after_block, mlir.oraScfYieldOpCreate(
        mlir_ctx,
        loc,
        &[_]mlir.MlirValue{mlir.oraOperationGetResult(next_op, 0)},
        1,
    ));

    mlir.oraBlockAppendOwnedOperation(body, while_op);
    mlir.oraBlockAppendOwnedOperation(body, mlir.oraReturnOpCreate(mlir_ctx, loc, &[_]mlir.MlirValue{}, 0));

    try encoder.registerFunctionOperation(helper);

    const counter_sort = z3.Z3_mk_bv_sort(z3_ctx.ctx, 256);
    const pre_counter = z3.Z3_mk_const(z3_ctx.ctx, z3.Z3_mk_string_symbol(z3_ctx.ctx, "pre_counter_swapped_while"), counter_sort);
    try encoder.global_map.put(try testing.allocator.dupe(u8, "counter"), pre_counter);

    const symbolic_ub = mlir.oraVariablePlaceholderOpCreate(mlir_ctx, loc, stringRef("swappedWhileUb"), i256_ty);
    const call = mlir.oraFuncCallOpCreate(
        mlir_ctx,
        loc,
        stringRef("symbolicSwappedNoWriteWhile"),
        &[_]mlir.MlirValue{mlir.oraOperationGetResult(symbolic_ub, 0)},
        1,
        &[_]mlir.MlirType{},
        0,
    );
    _ = try encoder.encodeOperation(call);

    try testing.expect(!encoder.isDegraded());
    const post_counter = encoder.global_map.get("counter").?;

    var solver = try Solver.init(&z3_ctx, testing.allocator);
    defer solver.deinit();
    solver.assert(z3.Z3_mk_not(z3_ctx.ctx, z3.Z3_mk_eq(z3_ctx.ctx, post_counter, pre_counter)));
    try testing.expectEqual(@as(z3.Z3_lbool, z3.Z3_L_FALSE), solver.check());
}

test "direct canonical unsigned positive-delta decrement scf.while result encodes exactly" {
    var z3_ctx = try Context.init(testing.allocator);
    defer z3_ctx.deinit();

    var encoder = Encoder.init(&z3_ctx, testing.allocator);
    defer encoder.deinit();

    const mlir_ctx = mlir.oraContextCreate();
    defer mlir.oraContextDestroy(mlir_ctx);
    loadAllDialects(mlir_ctx);
    _ = mlir.oraDialectRegister(mlir_ctx);

    const loc = mlir.oraLocationUnknownGet(mlir_ctx);
    const i256_ty = mlir.oraIntegerTypeCreate(mlir_ctx, 256);

    const init_attr = mlir.oraIntegerAttrCreateI64FromType(i256_ty, 11);
    const delta_attr = mlir.oraIntegerAttrCreateI64FromType(i256_ty, 3);
    const init_op = mlir.oraArithConstantOpCreate(mlir_ctx, loc, i256_ty, init_attr);
    const delta_op = mlir.oraArithConstantOpCreate(mlir_ctx, loc, i256_ty, delta_attr);
    const lb_op = mlir.oraVariablePlaceholderOpCreate(mlir_ctx, loc, stringRef("whileDecLb"), i256_ty);

    const init_vals = [_]mlir.MlirValue{mlir.oraOperationGetResult(init_op, 0)};
    const result_types = [_]mlir.MlirType{i256_ty};
    const while_op = mlir.oraScfWhileOpCreate(mlir_ctx, loc, &init_vals, init_vals.len, &result_types, result_types.len);
    const before_block = mlir.oraScfWhileOpGetBeforeBlock(while_op);
    const after_block = mlir.oraScfWhileOpGetAfterBlock(while_op);
    _ = mlir.mlirBlockAddArgument(before_block, i256_ty, loc);
    _ = mlir.mlirBlockAddArgument(after_block, i256_ty, loc);

    const before_arg = mlir.oraBlockGetArgument(before_block, 0);
    const cmp_op = mlir.oraArithCmpIOpCreate(mlir_ctx, loc, 8, before_arg, mlir.oraOperationGetResult(lb_op, 0)); // ugt
    mlir.oraBlockAppendOwnedOperation(before_block, cmp_op);
    mlir.oraBlockAppendOwnedOperation(before_block, mlir.oraScfConditionOpCreate(
        mlir_ctx,
        loc,
        mlir.oraOperationGetResult(cmp_op, 0),
        &[_]mlir.MlirValue{before_arg},
        1,
    ));

    const after_arg = mlir.oraBlockGetArgument(after_block, 0);
    const next_op = mlir.oraArithSubIOpCreate(mlir_ctx, loc, after_arg, mlir.oraOperationGetResult(delta_op, 0));
    mlir.oraBlockAppendOwnedOperation(after_block, next_op);
    mlir.oraBlockAppendOwnedOperation(after_block, mlir.oraScfYieldOpCreate(
        mlir_ctx,
        loc,
        &[_]mlir.MlirValue{mlir.oraOperationGetResult(next_op, 0)},
        1,
    ));

    const encoded = try encoder.encodeValue(mlir.oraOperationGetResult(while_op, 0));
    try testing.expect(!encoder.isDegraded());

    const init_ast = try encoder.encodeOperation(init_op);
    const lb_ast = try encoder.encodeOperation(lb_op);
    const delta_ast = try encoder.encodeOperation(delta_op);
    const zero = try encoder.encodeIntegerConstant(0, 256);
    const one = try encoder.encodeIntegerConstant(1, 256);
    const init_le_lb = z3.Z3_mk_bvule(z3_ctx.ctx, init_ast, lb_ast);
    const distance = z3.Z3_mk_ite(
        z3_ctx.ctx,
        init_le_lb,
        zero,
        z3.Z3_mk_bv_sub(z3_ctx.ctx, init_ast, lb_ast),
    );
    const distance_is_zero = z3.Z3_mk_eq(z3_ctx.ctx, distance, zero);
    const step_count = z3.Z3_mk_ite(
        z3_ctx.ctx,
        distance_is_zero,
        zero,
        z3.Z3_mk_bv_add(
            z3_ctx.ctx,
            z3.Z3_mk_bv_udiv(z3_ctx.ctx, z3.Z3_mk_bv_sub(z3_ctx.ctx, distance, one), delta_ast),
            one,
        ),
    );
    const total_delta = try encoder.encodeArithmeticOp(.Mul, step_count, delta_ast);
    const expected = try encoder.encodeArithmeticOp(.Sub, init_ast, total_delta);

    var solver = try Solver.init(&z3_ctx, testing.allocator);
    defer solver.deinit();
    solver.assert(z3.Z3_mk_not(z3_ctx.ctx, z3.Z3_mk_eq(z3_ctx.ctx, encoded, expected)));
    try testing.expectEqual(@as(z3.Z3_lbool, z3.Z3_L_FALSE), solver.check());
}

test "func.call summary with canonical unsigned positive-delta decrement no-write scf.while preserves state exactly" {
    var z3_ctx = try Context.init(testing.allocator);
    defer z3_ctx.deinit();

    var encoder = Encoder.init(&z3_ctx, testing.allocator);
    defer encoder.deinit();

    const mlir_ctx = mlir.oraContextCreate();
    defer mlir.oraContextDestroy(mlir_ctx);
    loadAllDialects(mlir_ctx);
    _ = mlir.oraDialectRegister(mlir_ctx);

    const loc = mlir.oraLocationUnknownGet(mlir_ctx);
    const i256_ty = mlir.oraIntegerTypeCreate(mlir_ctx, 256);

    const helper_attrs = [_]mlir.MlirNamedAttribute{
        namedAttr(mlir_ctx, "sym_name", mlir.oraStringAttrCreate(mlir_ctx, stringRef("symbolicNoWriteWhileDeltaDec"))),
    };
    const helper_param_types = [_]mlir.MlirType{i256_ty};
    const helper_param_locs = [_]mlir.MlirLocation{loc};
    const helper = mlir.oraFuncFuncOpCreate(mlir_ctx, loc, &helper_attrs, helper_attrs.len, &helper_param_types, &helper_param_locs, helper_param_types.len);
    const body = mlir.oraFuncOpGetBodyBlock(helper);

    const init_attr = mlir.oraIntegerAttrCreateI64FromType(i256_ty, 11);
    const delta_attr = mlir.oraIntegerAttrCreateI64FromType(i256_ty, 3);
    const init_op = mlir.oraArithConstantOpCreate(mlir_ctx, loc, i256_ty, init_attr);
    const delta_op = mlir.oraArithConstantOpCreate(mlir_ctx, loc, i256_ty, delta_attr);
    mlir.oraBlockAppendOwnedOperation(body, init_op);
    mlir.oraBlockAppendOwnedOperation(body, delta_op);

    const lb = mlir.oraBlockGetArgument(body, 0);
    const init_vals = [_]mlir.MlirValue{mlir.oraOperationGetResult(init_op, 0)};
    const result_types = [_]mlir.MlirType{i256_ty};
    const while_op = mlir.oraScfWhileOpCreate(mlir_ctx, loc, &init_vals, init_vals.len, &result_types, result_types.len);
    const before_block = mlir.oraScfWhileOpGetBeforeBlock(while_op);
    const after_block = mlir.oraScfWhileOpGetAfterBlock(while_op);
    _ = mlir.mlirBlockAddArgument(before_block, i256_ty, loc);
    _ = mlir.mlirBlockAddArgument(after_block, i256_ty, loc);
    const before_arg = mlir.oraBlockGetArgument(before_block, 0);
    const after_arg = mlir.oraBlockGetArgument(after_block, 0);

    const cmp_op = mlir.oraArithCmpIOpCreate(mlir_ctx, loc, 8, before_arg, lb); // ugt
    mlir.oraBlockAppendOwnedOperation(before_block, cmp_op);
    mlir.oraBlockAppendOwnedOperation(before_block, mlir.oraScfConditionOpCreate(
        mlir_ctx,
        loc,
        mlir.oraOperationGetResult(cmp_op, 0),
        &[_]mlir.MlirValue{before_arg},
        1,
    ));

    const next_op = mlir.oraArithSubIOpCreate(mlir_ctx, loc, after_arg, mlir.oraOperationGetResult(delta_op, 0));
    mlir.oraBlockAppendOwnedOperation(after_block, next_op);
    mlir.oraBlockAppendOwnedOperation(after_block, mlir.oraScfYieldOpCreate(
        mlir_ctx,
        loc,
        &[_]mlir.MlirValue{mlir.oraOperationGetResult(next_op, 0)},
        1,
    ));

    mlir.oraBlockAppendOwnedOperation(body, while_op);
    mlir.oraBlockAppendOwnedOperation(body, mlir.oraReturnOpCreate(mlir_ctx, loc, &[_]mlir.MlirValue{}, 0));

    try encoder.registerFunctionOperation(helper);

    const counter_sort = z3.Z3_mk_bv_sort(z3_ctx.ctx, 256);
    const pre_counter = z3.Z3_mk_const(z3_ctx.ctx, z3.Z3_mk_string_symbol(z3_ctx.ctx, "pre_counter_symbolic_while_delta_dec"), counter_sort);
    try encoder.global_map.put(try testing.allocator.dupe(u8, "counter"), pre_counter);

    const symbolic_lb = mlir.oraVariablePlaceholderOpCreate(mlir_ctx, loc, stringRef("lbWhileDeltaValue"), i256_ty);
    const call = mlir.oraFuncCallOpCreate(
        mlir_ctx,
        loc,
        stringRef("symbolicNoWriteWhileDeltaDec"),
        &[_]mlir.MlirValue{mlir.oraOperationGetResult(symbolic_lb, 0)},
        1,
        &[_]mlir.MlirType{},
        0,
    );
    _ = try encoder.encodeOperation(call);

    try testing.expect(!encoder.isDegraded());
    const post_counter = encoder.global_map.get("counter").?;

    var solver = try Solver.init(&z3_ctx, testing.allocator);
    defer solver.deinit();
    solver.assert(z3.Z3_mk_not(z3_ctx.ctx, z3.Z3_mk_eq(z3_ctx.ctx, post_counter, pre_counter)));
    try testing.expectEqual(@as(z3.Z3_lbool, z3.Z3_L_FALSE), solver.check());
}

test "known pure callee canonical unsigned positive-delta scf.while return encodes exactly" {
    var z3_ctx = try Context.init(testing.allocator);
    defer z3_ctx.deinit();

    var encoder = Encoder.init(&z3_ctx, testing.allocator);
    defer encoder.deinit();

    const mlir_ctx = mlir.oraContextCreate();
    defer mlir.oraContextDestroy(mlir_ctx);
    loadAllDialects(mlir_ctx);
    _ = mlir.oraDialectRegister(mlir_ctx);

    const loc = mlir.oraLocationUnknownGet(mlir_ctx);
    const i256_ty = mlir.oraIntegerTypeCreate(mlir_ctx, 256);

    const helper_attrs = [_]mlir.MlirNamedAttribute{
        namedAttr(mlir_ctx, "sym_name", mlir.oraStringAttrCreate(mlir_ctx, stringRef("symbolicWhileDeltaReturn"))),
    };
    const helper_param_types = [_]mlir.MlirType{i256_ty};
    const helper_param_locs = [_]mlir.MlirLocation{loc};
    const helper = mlir.oraFuncFuncOpCreate(mlir_ctx, loc, &helper_attrs, helper_attrs.len, &helper_param_types, &helper_param_locs, helper_param_types.len);
    const body = mlir.oraFuncOpGetBodyBlock(helper);

    const init_attr = mlir.oraIntegerAttrCreateI64FromType(i256_ty, 1);
    const delta_attr = mlir.oraIntegerAttrCreateI64FromType(i256_ty, 3);
    const init_op = mlir.oraArithConstantOpCreate(mlir_ctx, loc, i256_ty, init_attr);
    const delta_op = mlir.oraArithConstantOpCreate(mlir_ctx, loc, i256_ty, delta_attr);
    mlir.oraBlockAppendOwnedOperation(body, init_op);
    mlir.oraBlockAppendOwnedOperation(body, delta_op);

    const ub = mlir.oraBlockGetArgument(body, 0);
    const init_vals = [_]mlir.MlirValue{mlir.oraOperationGetResult(init_op, 0)};
    const result_types = [_]mlir.MlirType{i256_ty};
    const while_op = mlir.oraScfWhileOpCreate(mlir_ctx, loc, &init_vals, init_vals.len, &result_types, result_types.len);
    const before_block = mlir.oraScfWhileOpGetBeforeBlock(while_op);
    const after_block = mlir.oraScfWhileOpGetAfterBlock(while_op);
    _ = mlir.mlirBlockAddArgument(before_block, i256_ty, loc);
    _ = mlir.mlirBlockAddArgument(after_block, i256_ty, loc);
    const before_arg = mlir.oraBlockGetArgument(before_block, 0);
    const after_arg = mlir.oraBlockGetArgument(after_block, 0);

    const cmp_op = mlir.oraArithCmpIOpCreate(mlir_ctx, loc, 6, before_arg, ub); // ult
    mlir.oraBlockAppendOwnedOperation(before_block, cmp_op);
    mlir.oraBlockAppendOwnedOperation(before_block, mlir.oraScfConditionOpCreate(
        mlir_ctx,
        loc,
        mlir.oraOperationGetResult(cmp_op, 0),
        &[_]mlir.MlirValue{before_arg},
        1,
    ));

    const next_op = mlir.oraArithAddIOpCreate(mlir_ctx, loc, after_arg, mlir.oraOperationGetResult(delta_op, 0));
    mlir.oraBlockAppendOwnedOperation(after_block, next_op);
    mlir.oraBlockAppendOwnedOperation(after_block, mlir.oraScfYieldOpCreate(
        mlir_ctx,
        loc,
        &[_]mlir.MlirValue{mlir.oraOperationGetResult(next_op, 0)},
        1,
    ));
    mlir.oraBlockAppendOwnedOperation(body, while_op);
    mlir.oraBlockAppendOwnedOperation(body, mlir.oraReturnOpCreate(
        mlir_ctx,
        loc,
        &[_]mlir.MlirValue{mlir.oraOperationGetResult(while_op, 0)},
        1,
    ));

    try encoder.registerFunctionOperation(helper);

    const caller_ub = mlir.oraVariablePlaceholderOpCreate(mlir_ctx, loc, stringRef("callerWhileDeltaUb"), i256_ty);
    const call = mlir.oraFuncCallOpCreate(
        mlir_ctx,
        loc,
        stringRef("symbolicWhileDeltaReturn"),
        &[_]mlir.MlirValue{mlir.oraOperationGetResult(caller_ub, 0)},
        1,
        &[_]mlir.MlirType{i256_ty},
        1,
    );
    const encoded = try encoder.encodeOperation(call);
    try testing.expect(!encoder.isDegraded());

    const init_ast = try encoder.encodeOperation(init_op);
    const ub_ast = try encoder.encodeOperation(caller_ub);
    const delta_ast = try encoder.encodeOperation(delta_op);
    const zero = try encoder.encodeIntegerConstant(0, 256);
    const one = try encoder.encodeIntegerConstant(1, 256);
    const ub_le_init = z3.Z3_mk_bvule(z3_ctx.ctx, ub_ast, init_ast);
    const distance = z3.Z3_mk_ite(
        z3_ctx.ctx,
        ub_le_init,
        zero,
        z3.Z3_mk_bv_sub(z3_ctx.ctx, ub_ast, init_ast),
    );
    const distance_is_zero = z3.Z3_mk_eq(z3_ctx.ctx, distance, zero);
    const step_count = z3.Z3_mk_ite(
        z3_ctx.ctx,
        distance_is_zero,
        zero,
        z3.Z3_mk_bv_add(
            z3_ctx.ctx,
            z3.Z3_mk_bv_udiv(z3_ctx.ctx, z3.Z3_mk_bv_sub(z3_ctx.ctx, distance, one), delta_ast),
            one,
        ),
    );
    const total_delta = try encoder.encodeArithmeticOp(.Mul, step_count, delta_ast);
    const expected = try encoder.encodeArithmeticOp(.Add, init_ast, total_delta);

    var solver = try Solver.init(&z3_ctx, testing.allocator);
    defer solver.deinit();
    solver.assert(z3.Z3_mk_not(z3_ctx.ctx, z3.Z3_mk_eq(z3_ctx.ctx, encoded, expected)));
    try testing.expectEqual(@as(z3.Z3_lbool, z3.Z3_L_FALSE), solver.check());
}

test "known pure callee canonical unsigned positive-delta decrement scf.while return encodes exactly" {
    var z3_ctx = try Context.init(testing.allocator);
    defer z3_ctx.deinit();

    var encoder = Encoder.init(&z3_ctx, testing.allocator);
    defer encoder.deinit();

    const mlir_ctx = mlir.oraContextCreate();
    defer mlir.oraContextDestroy(mlir_ctx);
    loadAllDialects(mlir_ctx);
    _ = mlir.oraDialectRegister(mlir_ctx);

    const loc = mlir.oraLocationUnknownGet(mlir_ctx);
    const i256_ty = mlir.oraIntegerTypeCreate(mlir_ctx, 256);

    const helper_attrs = [_]mlir.MlirNamedAttribute{
        namedAttr(mlir_ctx, "sym_name", mlir.oraStringAttrCreate(mlir_ctx, stringRef("symbolicWhileDeltaDecReturn"))),
    };
    const helper_param_types = [_]mlir.MlirType{i256_ty};
    const helper_param_locs = [_]mlir.MlirLocation{loc};
    const helper = mlir.oraFuncFuncOpCreate(mlir_ctx, loc, &helper_attrs, helper_attrs.len, &helper_param_types, &helper_param_locs, helper_param_types.len);
    const body = mlir.oraFuncOpGetBodyBlock(helper);

    const init_attr = mlir.oraIntegerAttrCreateI64FromType(i256_ty, 11);
    const delta_attr = mlir.oraIntegerAttrCreateI64FromType(i256_ty, 3);
    const init_op = mlir.oraArithConstantOpCreate(mlir_ctx, loc, i256_ty, init_attr);
    const delta_op = mlir.oraArithConstantOpCreate(mlir_ctx, loc, i256_ty, delta_attr);
    mlir.oraBlockAppendOwnedOperation(body, init_op);
    mlir.oraBlockAppendOwnedOperation(body, delta_op);

    const lb = mlir.oraBlockGetArgument(body, 0);
    const init_vals = [_]mlir.MlirValue{mlir.oraOperationGetResult(init_op, 0)};
    const result_types = [_]mlir.MlirType{i256_ty};
    const while_op = mlir.oraScfWhileOpCreate(mlir_ctx, loc, &init_vals, init_vals.len, &result_types, result_types.len);
    const before_block = mlir.oraScfWhileOpGetBeforeBlock(while_op);
    const after_block = mlir.oraScfWhileOpGetAfterBlock(while_op);
    _ = mlir.mlirBlockAddArgument(before_block, i256_ty, loc);
    _ = mlir.mlirBlockAddArgument(after_block, i256_ty, loc);
    const before_arg = mlir.oraBlockGetArgument(before_block, 0);
    const after_arg = mlir.oraBlockGetArgument(after_block, 0);

    const cmp_op = mlir.oraArithCmpIOpCreate(mlir_ctx, loc, 8, before_arg, lb); // ugt
    mlir.oraBlockAppendOwnedOperation(before_block, cmp_op);
    mlir.oraBlockAppendOwnedOperation(before_block, mlir.oraScfConditionOpCreate(
        mlir_ctx,
        loc,
        mlir.oraOperationGetResult(cmp_op, 0),
        &[_]mlir.MlirValue{before_arg},
        1,
    ));

    const next_op = mlir.oraArithSubIOpCreate(mlir_ctx, loc, after_arg, mlir.oraOperationGetResult(delta_op, 0));
    mlir.oraBlockAppendOwnedOperation(after_block, next_op);
    mlir.oraBlockAppendOwnedOperation(after_block, mlir.oraScfYieldOpCreate(
        mlir_ctx,
        loc,
        &[_]mlir.MlirValue{mlir.oraOperationGetResult(next_op, 0)},
        1,
    ));
    mlir.oraBlockAppendOwnedOperation(body, while_op);
    mlir.oraBlockAppendOwnedOperation(body, mlir.oraReturnOpCreate(
        mlir_ctx,
        loc,
        &[_]mlir.MlirValue{mlir.oraOperationGetResult(while_op, 0)},
        1,
    ));

    try encoder.registerFunctionOperation(helper);

    const caller_lb = mlir.oraVariablePlaceholderOpCreate(mlir_ctx, loc, stringRef("callerWhileDeltaLb"), i256_ty);
    const call = mlir.oraFuncCallOpCreate(
        mlir_ctx,
        loc,
        stringRef("symbolicWhileDeltaDecReturn"),
        &[_]mlir.MlirValue{mlir.oraOperationGetResult(caller_lb, 0)},
        1,
        &[_]mlir.MlirType{i256_ty},
        1,
    );
    const encoded = try encoder.encodeOperation(call);
    try testing.expect(!encoder.isDegraded());

    const init_ast = try encoder.encodeOperation(init_op);
    const lb_ast = try encoder.encodeOperation(caller_lb);
    const delta_ast = try encoder.encodeOperation(delta_op);
    const zero = try encoder.encodeIntegerConstant(0, 256);
    const one = try encoder.encodeIntegerConstant(1, 256);
    const init_le_lb = z3.Z3_mk_bvule(z3_ctx.ctx, init_ast, lb_ast);
    const distance = z3.Z3_mk_ite(
        z3_ctx.ctx,
        init_le_lb,
        zero,
        z3.Z3_mk_bv_sub(z3_ctx.ctx, init_ast, lb_ast),
    );
    const distance_is_zero = z3.Z3_mk_eq(z3_ctx.ctx, distance, zero);
    const step_count = z3.Z3_mk_ite(
        z3_ctx.ctx,
        distance_is_zero,
        zero,
        z3.Z3_mk_bv_add(
            z3_ctx.ctx,
            z3.Z3_mk_bv_udiv(z3_ctx.ctx, z3.Z3_mk_bv_sub(z3_ctx.ctx, distance, one), delta_ast),
            one,
        ),
    );
    const total_delta = try encoder.encodeArithmeticOp(.Mul, step_count, delta_ast);
    const expected = try encoder.encodeArithmeticOp(.Sub, init_ast, total_delta);

    var solver = try Solver.init(&z3_ctx, testing.allocator);
    defer solver.deinit();
    solver.assert(z3.Z3_mk_not(z3_ctx.ctx, z3.Z3_mk_eq(z3_ctx.ctx, encoded, expected)));
    try testing.expectEqual(@as(z3.Z3_lbool, z3.Z3_L_FALSE), solver.check());
}

test "direct canonical unsigned multi-result scf.while encodes derived result exactly" {
    var z3_ctx = try Context.init(testing.allocator);
    defer z3_ctx.deinit();

    var encoder = Encoder.init(&z3_ctx, testing.allocator);
    defer encoder.deinit();

    const mlir_ctx = mlir.oraContextCreate();
    defer mlir.oraContextDestroy(mlir_ctx);
    loadAllDialects(mlir_ctx);
    _ = mlir.oraDialectRegister(mlir_ctx);

    const loc = mlir.oraLocationUnknownGet(mlir_ctx);
    const i256_ty = mlir.oraIntegerTypeCreate(mlir_ctx, 256);

    const init_ctrl_attr = mlir.oraIntegerAttrCreateI64FromType(i256_ty, 1);
    const init_sum_attr = mlir.oraIntegerAttrCreateI64FromType(i256_ty, 10);
    const ctrl_delta_attr = mlir.oraIntegerAttrCreateI64FromType(i256_ty, 3);
    const sum_delta_attr = mlir.oraIntegerAttrCreateI64FromType(i256_ty, 5);
    const init_ctrl_op = mlir.oraArithConstantOpCreate(mlir_ctx, loc, i256_ty, init_ctrl_attr);
    const init_sum_op = mlir.oraArithConstantOpCreate(mlir_ctx, loc, i256_ty, init_sum_attr);
    const ctrl_delta_op = mlir.oraArithConstantOpCreate(mlir_ctx, loc, i256_ty, ctrl_delta_attr);
    const sum_delta_op = mlir.oraArithConstantOpCreate(mlir_ctx, loc, i256_ty, sum_delta_attr);
    const ub_op = mlir.oraVariablePlaceholderOpCreate(mlir_ctx, loc, stringRef("whileMultiUb"), i256_ty);

    const init_vals = [_]mlir.MlirValue{
        mlir.oraOperationGetResult(init_ctrl_op, 0),
        mlir.oraOperationGetResult(init_sum_op, 0),
    };
    const result_types = [_]mlir.MlirType{ i256_ty, i256_ty };
    const while_op = mlir.oraScfWhileOpCreate(mlir_ctx, loc, &init_vals, init_vals.len, &result_types, result_types.len);
    const before_block = mlir.oraScfWhileOpGetBeforeBlock(while_op);
    const after_block = mlir.oraScfWhileOpGetAfterBlock(while_op);
    _ = mlir.mlirBlockAddArgument(before_block, i256_ty, loc);
    _ = mlir.mlirBlockAddArgument(before_block, i256_ty, loc);
    _ = mlir.mlirBlockAddArgument(after_block, i256_ty, loc);
    _ = mlir.mlirBlockAddArgument(after_block, i256_ty, loc);

    const before_ctrl = mlir.oraBlockGetArgument(before_block, 0);
    const before_sum = mlir.oraBlockGetArgument(before_block, 1);
    const cmp_op = mlir.oraArithCmpIOpCreate(mlir_ctx, loc, 6, before_ctrl, mlir.oraOperationGetResult(ub_op, 0)); // ult
    mlir.oraBlockAppendOwnedOperation(before_block, cmp_op);
    mlir.oraBlockAppendOwnedOperation(before_block, mlir.oraScfConditionOpCreate(
        mlir_ctx,
        loc,
        mlir.oraOperationGetResult(cmp_op, 0),
        &[_]mlir.MlirValue{ before_ctrl, before_sum },
        2,
    ));

    const after_ctrl = mlir.oraBlockGetArgument(after_block, 0);
    const after_sum = mlir.oraBlockGetArgument(after_block, 1);
    const next_ctrl = mlir.oraArithAddIOpCreate(mlir_ctx, loc, after_ctrl, mlir.oraOperationGetResult(ctrl_delta_op, 0));
    const next_sum = mlir.oraArithAddIOpCreate(mlir_ctx, loc, after_sum, mlir.oraOperationGetResult(sum_delta_op, 0));
    mlir.oraBlockAppendOwnedOperation(after_block, next_ctrl);
    mlir.oraBlockAppendOwnedOperation(after_block, next_sum);
    mlir.oraBlockAppendOwnedOperation(after_block, mlir.oraScfYieldOpCreate(
        mlir_ctx,
        loc,
        &[_]mlir.MlirValue{
            mlir.oraOperationGetResult(next_ctrl, 0),
            mlir.oraOperationGetResult(next_sum, 0),
        },
        2,
    ));

    const encoded = try encoder.encodeValue(mlir.oraOperationGetResult(while_op, 1));
    try testing.expect(!encoder.isDegraded());

    const init_sum_ast = try encoder.encodeOperation(init_sum_op);
    const init_ctrl_ast = try encoder.encodeOperation(init_ctrl_op);
    const ctrl_delta_ast = try encoder.encodeOperation(ctrl_delta_op);
    const sum_delta_ast = try encoder.encodeOperation(sum_delta_op);
    const ub_ast = try encoder.encodeOperation(ub_op);
    const zero = try encoder.encodeIntegerConstant(0, 256);
    const one = try encoder.encodeIntegerConstant(1, 256);
    const ub_le_init = z3.Z3_mk_bvule(z3_ctx.ctx, ub_ast, init_ctrl_ast);
    const distance = z3.Z3_mk_ite(
        z3_ctx.ctx,
        ub_le_init,
        zero,
        z3.Z3_mk_bv_sub(z3_ctx.ctx, ub_ast, init_ctrl_ast),
    );
    const distance_is_zero = z3.Z3_mk_eq(z3_ctx.ctx, distance, zero);
    const step_count = z3.Z3_mk_ite(
        z3_ctx.ctx,
        distance_is_zero,
        zero,
        z3.Z3_mk_bv_add(
            z3_ctx.ctx,
            z3.Z3_mk_bv_udiv(z3_ctx.ctx, z3.Z3_mk_bv_sub(z3_ctx.ctx, distance, one), ctrl_delta_ast),
            one,
        ),
    );
    const total_delta = try encoder.encodeArithmeticOp(.Mul, step_count, sum_delta_ast);
    const expected = try encoder.encodeArithmeticOp(.Add, init_sum_ast, total_delta);

    var solver = try Solver.init(&z3_ctx, testing.allocator);
    defer solver.deinit();
    solver.assert(z3.Z3_mk_not(z3_ctx.ctx, z3.Z3_mk_eq(z3_ctx.ctx, encoded, expected)));
    try testing.expectEqual(@as(z3.Z3_lbool, z3.Z3_L_FALSE), solver.check());
}

test "known pure callee canonical unsigned multi-result scf.while return encodes derived result exactly" {
    var z3_ctx = try Context.init(testing.allocator);
    defer z3_ctx.deinit();

    var encoder = Encoder.init(&z3_ctx, testing.allocator);
    defer encoder.deinit();

    const mlir_ctx = mlir.oraContextCreate();
    defer mlir.oraContextDestroy(mlir_ctx);
    loadAllDialects(mlir_ctx);
    _ = mlir.oraDialectRegister(mlir_ctx);

    const loc = mlir.oraLocationUnknownGet(mlir_ctx);
    const i256_ty = mlir.oraIntegerTypeCreate(mlir_ctx, 256);

    const helper_attrs = [_]mlir.MlirNamedAttribute{
        namedAttr(mlir_ctx, "sym_name", mlir.oraStringAttrCreate(mlir_ctx, stringRef("symbolicWhileMultiReturn"))),
    };
    const helper_param_types = [_]mlir.MlirType{i256_ty};
    const helper_param_locs = [_]mlir.MlirLocation{loc};
    const helper = mlir.oraFuncFuncOpCreate(mlir_ctx, loc, &helper_attrs, helper_attrs.len, &helper_param_types, &helper_param_locs, helper_param_types.len);
    const body = mlir.oraFuncOpGetBodyBlock(helper);

    const init_ctrl_attr = mlir.oraIntegerAttrCreateI64FromType(i256_ty, 1);
    const init_sum_attr = mlir.oraIntegerAttrCreateI64FromType(i256_ty, 10);
    const ctrl_delta_attr = mlir.oraIntegerAttrCreateI64FromType(i256_ty, 3);
    const sum_delta_attr = mlir.oraIntegerAttrCreateI64FromType(i256_ty, 5);
    const init_ctrl_op = mlir.oraArithConstantOpCreate(mlir_ctx, loc, i256_ty, init_ctrl_attr);
    const init_sum_op = mlir.oraArithConstantOpCreate(mlir_ctx, loc, i256_ty, init_sum_attr);
    const ctrl_delta_op = mlir.oraArithConstantOpCreate(mlir_ctx, loc, i256_ty, ctrl_delta_attr);
    const sum_delta_op = mlir.oraArithConstantOpCreate(mlir_ctx, loc, i256_ty, sum_delta_attr);
    mlir.oraBlockAppendOwnedOperation(body, init_ctrl_op);
    mlir.oraBlockAppendOwnedOperation(body, init_sum_op);
    mlir.oraBlockAppendOwnedOperation(body, ctrl_delta_op);
    mlir.oraBlockAppendOwnedOperation(body, sum_delta_op);

    const ub = mlir.oraBlockGetArgument(body, 0);
    const init_vals = [_]mlir.MlirValue{
        mlir.oraOperationGetResult(init_ctrl_op, 0),
        mlir.oraOperationGetResult(init_sum_op, 0),
    };
    const result_types = [_]mlir.MlirType{ i256_ty, i256_ty };
    const while_op = mlir.oraScfWhileOpCreate(mlir_ctx, loc, &init_vals, init_vals.len, &result_types, result_types.len);
    const before_block = mlir.oraScfWhileOpGetBeforeBlock(while_op);
    const after_block = mlir.oraScfWhileOpGetAfterBlock(while_op);
    _ = mlir.mlirBlockAddArgument(before_block, i256_ty, loc);
    _ = mlir.mlirBlockAddArgument(before_block, i256_ty, loc);
    _ = mlir.mlirBlockAddArgument(after_block, i256_ty, loc);
    _ = mlir.mlirBlockAddArgument(after_block, i256_ty, loc);
    const before_ctrl = mlir.oraBlockGetArgument(before_block, 0);
    const before_sum = mlir.oraBlockGetArgument(before_block, 1);
    const after_ctrl = mlir.oraBlockGetArgument(after_block, 0);
    const after_sum = mlir.oraBlockGetArgument(after_block, 1);

    const cmp_op = mlir.oraArithCmpIOpCreate(mlir_ctx, loc, 6, before_ctrl, ub); // ult
    mlir.oraBlockAppendOwnedOperation(before_block, cmp_op);
    mlir.oraBlockAppendOwnedOperation(before_block, mlir.oraScfConditionOpCreate(
        mlir_ctx,
        loc,
        mlir.oraOperationGetResult(cmp_op, 0),
        &[_]mlir.MlirValue{ before_ctrl, before_sum },
        2,
    ));

    const next_ctrl = mlir.oraArithAddIOpCreate(mlir_ctx, loc, after_ctrl, mlir.oraOperationGetResult(ctrl_delta_op, 0));
    const next_sum = mlir.oraArithAddIOpCreate(mlir_ctx, loc, after_sum, mlir.oraOperationGetResult(sum_delta_op, 0));
    mlir.oraBlockAppendOwnedOperation(after_block, next_ctrl);
    mlir.oraBlockAppendOwnedOperation(after_block, next_sum);
    mlir.oraBlockAppendOwnedOperation(after_block, mlir.oraScfYieldOpCreate(
        mlir_ctx,
        loc,
        &[_]mlir.MlirValue{
            mlir.oraOperationGetResult(next_ctrl, 0),
            mlir.oraOperationGetResult(next_sum, 0),
        },
        2,
    ));
    mlir.oraBlockAppendOwnedOperation(body, while_op);
    mlir.oraBlockAppendOwnedOperation(body, mlir.oraReturnOpCreate(
        mlir_ctx,
        loc,
        &[_]mlir.MlirValue{mlir.oraOperationGetResult(while_op, 1)},
        1,
    ));

    try encoder.registerFunctionOperation(helper);

    const caller_ub = mlir.oraVariablePlaceholderOpCreate(mlir_ctx, loc, stringRef("callerWhileMultiUb"), i256_ty);
    const call = mlir.oraFuncCallOpCreate(
        mlir_ctx,
        loc,
        stringRef("symbolicWhileMultiReturn"),
        &[_]mlir.MlirValue{mlir.oraOperationGetResult(caller_ub, 0)},
        1,
        &[_]mlir.MlirType{i256_ty},
        1,
    );
    const encoded = try encoder.encodeOperation(call);
    try testing.expect(!encoder.isDegraded());

    const init_ctrl_ast = try encoder.encodeOperation(init_ctrl_op);
    const init_sum_ast = try encoder.encodeOperation(init_sum_op);
    const ctrl_delta_ast = try encoder.encodeOperation(ctrl_delta_op);
    const sum_delta_ast = try encoder.encodeOperation(sum_delta_op);
    const ub_ast = try encoder.encodeOperation(caller_ub);
    const zero = try encoder.encodeIntegerConstant(0, 256);
    const one = try encoder.encodeIntegerConstant(1, 256);
    const ub_le_init = z3.Z3_mk_bvule(z3_ctx.ctx, ub_ast, init_ctrl_ast);
    const distance = z3.Z3_mk_ite(
        z3_ctx.ctx,
        ub_le_init,
        zero,
        z3.Z3_mk_bv_sub(z3_ctx.ctx, ub_ast, init_ctrl_ast),
    );
    const distance_is_zero = z3.Z3_mk_eq(z3_ctx.ctx, distance, zero);
    const step_count = z3.Z3_mk_ite(
        z3_ctx.ctx,
        distance_is_zero,
        zero,
        z3.Z3_mk_bv_add(
            z3_ctx.ctx,
            z3.Z3_mk_bv_udiv(z3_ctx.ctx, z3.Z3_mk_bv_sub(z3_ctx.ctx, distance, one), ctrl_delta_ast),
            one,
        ),
    );
    const total_delta = try encoder.encodeArithmeticOp(.Mul, step_count, sum_delta_ast);
    const expected = try encoder.encodeArithmeticOp(.Add, init_sum_ast, total_delta);

    var solver = try Solver.init(&z3_ctx, testing.allocator);
    defer solver.deinit();
    solver.assert(z3.Z3_mk_not(z3_ctx.ctx, z3.Z3_mk_eq(z3_ctx.ctx, encoded, expected)));
    try testing.expectEqual(@as(z3.Z3_lbool, z3.Z3_L_FALSE), solver.check());
}

test "direct canonical unsigned scf.while with nonzero control index encodes derived result exactly" {
    var z3_ctx = try Context.init(testing.allocator);
    defer z3_ctx.deinit();

    var encoder = Encoder.init(&z3_ctx, testing.allocator);
    defer encoder.deinit();

    const mlir_ctx = mlir.oraContextCreate();
    defer mlir.oraContextDestroy(mlir_ctx);
    loadAllDialects(mlir_ctx);
    _ = mlir.oraDialectRegister(mlir_ctx);

    const loc = mlir.oraLocationUnknownGet(mlir_ctx);
    const i256_ty = mlir.oraIntegerTypeCreate(mlir_ctx, 256);

    const init_sum_attr = mlir.oraIntegerAttrCreateI64FromType(i256_ty, 10);
    const init_ctrl_attr = mlir.oraIntegerAttrCreateI64FromType(i256_ty, 1);
    const sum_delta_attr = mlir.oraIntegerAttrCreateI64FromType(i256_ty, 5);
    const ctrl_delta_attr = mlir.oraIntegerAttrCreateI64FromType(i256_ty, 3);
    const init_sum_op = mlir.oraArithConstantOpCreate(mlir_ctx, loc, i256_ty, init_sum_attr);
    const init_ctrl_op = mlir.oraArithConstantOpCreate(mlir_ctx, loc, i256_ty, init_ctrl_attr);
    const sum_delta_op = mlir.oraArithConstantOpCreate(mlir_ctx, loc, i256_ty, sum_delta_attr);
    const ctrl_delta_op = mlir.oraArithConstantOpCreate(mlir_ctx, loc, i256_ty, ctrl_delta_attr);
    const ub_op = mlir.oraVariablePlaceholderOpCreate(mlir_ctx, loc, stringRef("whileControlSecondUb"), i256_ty);

    const init_vals = [_]mlir.MlirValue{
        mlir.oraOperationGetResult(init_sum_op, 0),
        mlir.oraOperationGetResult(init_ctrl_op, 0),
    };
    const result_types = [_]mlir.MlirType{ i256_ty, i256_ty };
    const while_op = mlir.oraScfWhileOpCreate(mlir_ctx, loc, &init_vals, init_vals.len, &result_types, result_types.len);
    const before_block = mlir.oraScfWhileOpGetBeforeBlock(while_op);
    const after_block = mlir.oraScfWhileOpGetAfterBlock(while_op);
    _ = mlir.mlirBlockAddArgument(before_block, i256_ty, loc);
    _ = mlir.mlirBlockAddArgument(before_block, i256_ty, loc);
    _ = mlir.mlirBlockAddArgument(after_block, i256_ty, loc);
    _ = mlir.mlirBlockAddArgument(after_block, i256_ty, loc);

    const before_ctrl = mlir.oraBlockGetArgument(before_block, 1);
    const cmp_op = mlir.oraArithCmpIOpCreate(mlir_ctx, loc, 6, before_ctrl, mlir.oraOperationGetResult(ub_op, 0)); // ult
    mlir.oraBlockAppendOwnedOperation(before_block, cmp_op);
    mlir.oraBlockAppendOwnedOperation(before_block, mlir.oraScfConditionOpCreate(
        mlir_ctx,
        loc,
        mlir.oraOperationGetResult(cmp_op, 0),
        &[_]mlir.MlirValue{
            mlir.oraBlockGetArgument(before_block, 0),
            before_ctrl,
        },
        2,
    ));

    const after_sum = mlir.oraBlockGetArgument(after_block, 0);
    const after_ctrl = mlir.oraBlockGetArgument(after_block, 1);
    const next_sum = mlir.oraArithAddIOpCreate(mlir_ctx, loc, after_sum, mlir.oraOperationGetResult(sum_delta_op, 0));
    const next_ctrl = mlir.oraArithAddIOpCreate(mlir_ctx, loc, after_ctrl, mlir.oraOperationGetResult(ctrl_delta_op, 0));
    mlir.oraBlockAppendOwnedOperation(after_block, next_sum);
    mlir.oraBlockAppendOwnedOperation(after_block, next_ctrl);
    mlir.oraBlockAppendOwnedOperation(after_block, mlir.oraScfYieldOpCreate(
        mlir_ctx,
        loc,
        &[_]mlir.MlirValue{
            mlir.oraOperationGetResult(next_sum, 0),
            mlir.oraOperationGetResult(next_ctrl, 0),
        },
        2,
    ));

    const encoded = try encoder.encodeValue(mlir.oraOperationGetResult(while_op, 0));
    try testing.expect(!encoder.isDegraded());

    const init_sum_ast = try encoder.encodeOperation(init_sum_op);
    const init_ctrl_ast = try encoder.encodeOperation(init_ctrl_op);
    const sum_delta_ast = try encoder.encodeOperation(sum_delta_op);
    const ctrl_delta_ast = try encoder.encodeOperation(ctrl_delta_op);
    const ub_ast = try encoder.encodeOperation(ub_op);
    const zero = try encoder.encodeIntegerConstant(0, 256);
    const one = try encoder.encodeIntegerConstant(1, 256);
    const ub_le_init = z3.Z3_mk_bvule(z3_ctx.ctx, ub_ast, init_ctrl_ast);
    const distance = z3.Z3_mk_ite(
        z3_ctx.ctx,
        ub_le_init,
        zero,
        z3.Z3_mk_bv_sub(z3_ctx.ctx, ub_ast, init_ctrl_ast),
    );
    const distance_is_zero = z3.Z3_mk_eq(z3_ctx.ctx, distance, zero);
    const step_count = z3.Z3_mk_ite(
        z3_ctx.ctx,
        distance_is_zero,
        zero,
        z3.Z3_mk_bv_add(
            z3_ctx.ctx,
            z3.Z3_mk_bv_udiv(z3_ctx.ctx, z3.Z3_mk_bv_sub(z3_ctx.ctx, distance, one), ctrl_delta_ast),
            one,
        ),
    );
    const total_delta = try encoder.encodeArithmeticOp(.Mul, step_count, sum_delta_ast);
    const expected = try encoder.encodeArithmeticOp(.Add, init_sum_ast, total_delta);

    var solver = try Solver.init(&z3_ctx, testing.allocator);
    defer solver.deinit();
    solver.assert(z3.Z3_mk_not(z3_ctx.ctx, z3.Z3_mk_eq(z3_ctx.ctx, encoded, expected)));
    try testing.expectEqual(@as(z3.Z3_lbool, z3.Z3_L_FALSE), solver.check());
}

test "known pure callee canonical unsigned scf.while with nonzero control index return encodes derived result exactly" {
    var z3_ctx = try Context.init(testing.allocator);
    defer z3_ctx.deinit();

    var encoder = Encoder.init(&z3_ctx, testing.allocator);
    defer encoder.deinit();

    const mlir_ctx = mlir.oraContextCreate();
    defer mlir.oraContextDestroy(mlir_ctx);
    loadAllDialects(mlir_ctx);
    _ = mlir.oraDialectRegister(mlir_ctx);

    const loc = mlir.oraLocationUnknownGet(mlir_ctx);
    const i256_ty = mlir.oraIntegerTypeCreate(mlir_ctx, 256);

    const helper_attrs = [_]mlir.MlirNamedAttribute{
        namedAttr(mlir_ctx, "sym_name", mlir.oraStringAttrCreate(mlir_ctx, stringRef("whileControlSecondReturn"))),
    };
    const helper_param_types = [_]mlir.MlirType{i256_ty};
    const helper_param_locs = [_]mlir.MlirLocation{loc};
    const helper = mlir.oraFuncFuncOpCreate(mlir_ctx, loc, &helper_attrs, helper_attrs.len, &helper_param_types, &helper_param_locs, helper_param_types.len);
    const body = mlir.oraFuncOpGetBodyBlock(helper);

    const init_sum_attr = mlir.oraIntegerAttrCreateI64FromType(i256_ty, 10);
    const init_ctrl_attr = mlir.oraIntegerAttrCreateI64FromType(i256_ty, 1);
    const sum_delta_attr = mlir.oraIntegerAttrCreateI64FromType(i256_ty, 5);
    const ctrl_delta_attr = mlir.oraIntegerAttrCreateI64FromType(i256_ty, 3);
    const init_sum_op = mlir.oraArithConstantOpCreate(mlir_ctx, loc, i256_ty, init_sum_attr);
    const init_ctrl_op = mlir.oraArithConstantOpCreate(mlir_ctx, loc, i256_ty, init_ctrl_attr);
    const sum_delta_op = mlir.oraArithConstantOpCreate(mlir_ctx, loc, i256_ty, sum_delta_attr);
    const ctrl_delta_op = mlir.oraArithConstantOpCreate(mlir_ctx, loc, i256_ty, ctrl_delta_attr);
    mlir.oraBlockAppendOwnedOperation(body, init_sum_op);
    mlir.oraBlockAppendOwnedOperation(body, init_ctrl_op);
    mlir.oraBlockAppendOwnedOperation(body, sum_delta_op);
    mlir.oraBlockAppendOwnedOperation(body, ctrl_delta_op);

    const ub = mlir.oraBlockGetArgument(body, 0);
    const init_vals = [_]mlir.MlirValue{
        mlir.oraOperationGetResult(init_sum_op, 0),
        mlir.oraOperationGetResult(init_ctrl_op, 0),
    };
    const result_types = [_]mlir.MlirType{ i256_ty, i256_ty };
    const while_op = mlir.oraScfWhileOpCreate(mlir_ctx, loc, &init_vals, init_vals.len, &result_types, result_types.len);
    const before_block = mlir.oraScfWhileOpGetBeforeBlock(while_op);
    const after_block = mlir.oraScfWhileOpGetAfterBlock(while_op);
    _ = mlir.mlirBlockAddArgument(before_block, i256_ty, loc);
    _ = mlir.mlirBlockAddArgument(before_block, i256_ty, loc);
    _ = mlir.mlirBlockAddArgument(after_block, i256_ty, loc);
    _ = mlir.mlirBlockAddArgument(after_block, i256_ty, loc);

    const before_ctrl = mlir.oraBlockGetArgument(before_block, 1);
    const cmp_op = mlir.oraArithCmpIOpCreate(mlir_ctx, loc, 6, before_ctrl, ub); // ult
    mlir.oraBlockAppendOwnedOperation(before_block, cmp_op);
    mlir.oraBlockAppendOwnedOperation(before_block, mlir.oraScfConditionOpCreate(
        mlir_ctx,
        loc,
        mlir.oraOperationGetResult(cmp_op, 0),
        &[_]mlir.MlirValue{
            mlir.oraBlockGetArgument(before_block, 0),
            before_ctrl,
        },
        2,
    ));

    const after_sum = mlir.oraBlockGetArgument(after_block, 0);
    const after_ctrl = mlir.oraBlockGetArgument(after_block, 1);
    const next_sum = mlir.oraArithAddIOpCreate(mlir_ctx, loc, after_sum, mlir.oraOperationGetResult(sum_delta_op, 0));
    const next_ctrl = mlir.oraArithAddIOpCreate(mlir_ctx, loc, after_ctrl, mlir.oraOperationGetResult(ctrl_delta_op, 0));
    mlir.oraBlockAppendOwnedOperation(after_block, next_sum);
    mlir.oraBlockAppendOwnedOperation(after_block, next_ctrl);
    mlir.oraBlockAppendOwnedOperation(after_block, mlir.oraScfYieldOpCreate(
        mlir_ctx,
        loc,
        &[_]mlir.MlirValue{
            mlir.oraOperationGetResult(next_sum, 0),
            mlir.oraOperationGetResult(next_ctrl, 0),
        },
        2,
    ));
    mlir.oraBlockAppendOwnedOperation(body, while_op);
    mlir.oraBlockAppendOwnedOperation(body, mlir.oraReturnOpCreate(
        mlir_ctx,
        loc,
        &[_]mlir.MlirValue{mlir.oraOperationGetResult(while_op, 0)},
        1,
    ));

    try encoder.registerFunctionOperation(helper);

    const caller_ub = mlir.oraVariablePlaceholderOpCreate(mlir_ctx, loc, stringRef("callerWhileControlSecondUb"), i256_ty);
    const call = mlir.oraFuncCallOpCreate(
        mlir_ctx,
        loc,
        stringRef("whileControlSecondReturn"),
        &[_]mlir.MlirValue{mlir.oraOperationGetResult(caller_ub, 0)},
        1,
        &[_]mlir.MlirType{i256_ty},
        1,
    );
    const encoded = try encoder.encodeOperation(call);
    try testing.expect(!encoder.isDegraded());

    const init_sum_ast = try encoder.encodeOperation(init_sum_op);
    const init_ctrl_ast = try encoder.encodeOperation(init_ctrl_op);
    const sum_delta_ast = try encoder.encodeOperation(sum_delta_op);
    const ctrl_delta_ast = try encoder.encodeOperation(ctrl_delta_op);
    const ub_ast = try encoder.encodeOperation(caller_ub);
    const zero = try encoder.encodeIntegerConstant(0, 256);
    const one = try encoder.encodeIntegerConstant(1, 256);
    const ub_le_init = z3.Z3_mk_bvule(z3_ctx.ctx, ub_ast, init_ctrl_ast);
    const distance = z3.Z3_mk_ite(
        z3_ctx.ctx,
        ub_le_init,
        zero,
        z3.Z3_mk_bv_sub(z3_ctx.ctx, ub_ast, init_ctrl_ast),
    );
    const distance_is_zero = z3.Z3_mk_eq(z3_ctx.ctx, distance, zero);
    const step_count = z3.Z3_mk_ite(
        z3_ctx.ctx,
        distance_is_zero,
        zero,
        z3.Z3_mk_bv_add(
            z3_ctx.ctx,
            z3.Z3_mk_bv_udiv(z3_ctx.ctx, z3.Z3_mk_bv_sub(z3_ctx.ctx, distance, one), ctrl_delta_ast),
            one,
        ),
    );
    const total_delta = try encoder.encodeArithmeticOp(.Mul, step_count, sum_delta_ast);
    const expected = try encoder.encodeArithmeticOp(.Add, init_sum_ast, total_delta);

    var solver = try Solver.init(&z3_ctx, testing.allocator);
    defer solver.deinit();
    solver.assert(z3.Z3_mk_not(z3_ctx.ctx, z3.Z3_mk_eq(z3_ctx.ctx, encoded, expected)));
    try testing.expectEqual(@as(z3.Z3_lbool, z3.Z3_L_FALSE), solver.check());
}

test "direct non-throwing ora.try_stmt result encodes exactly" {
    var z3_ctx = try Context.init(testing.allocator);
    defer z3_ctx.deinit();

    var encoder = Encoder.init(&z3_ctx, testing.allocator);
    defer encoder.deinit();

    const mlir_ctx = mlir.oraContextCreate();
    defer mlir.oraContextDestroy(mlir_ctx);
    loadAllDialects(mlir_ctx);
    _ = mlir.oraDialectRegister(mlir_ctx);

    const loc = mlir.oraLocationUnknownGet(mlir_ctx);
    const i256_ty = mlir.oraIntegerTypeCreate(mlir_ctx, 256);

    const try_stmt = mlir.oraTryStmtOpCreate(mlir_ctx, loc, &[_]mlir.MlirType{i256_ty}, 1);
    const try_block = mlir.oraTryStmtOpGetTryBlock(try_stmt);
    const catch_block = mlir.oraTryStmtOpGetCatchBlock(try_stmt);

    const try_attr = mlir.oraIntegerAttrCreateI64FromType(i256_ty, 7);
    const try_op = mlir.oraArithConstantOpCreate(mlir_ctx, loc, i256_ty, try_attr);
    const expected = try encoder.encodeOperation(try_op);
    mlir.oraBlockAppendOwnedOperation(try_block, try_op);
    mlir.oraBlockAppendOwnedOperation(try_block, mlir.oraYieldOpCreate(
        mlir_ctx,
        loc,
        &[_]mlir.MlirValue{mlir.oraOperationGetResult(try_op, 0)},
        1,
    ));

    const catch_attr = mlir.oraIntegerAttrCreateI64FromType(i256_ty, 99);
    const catch_op = mlir.oraArithConstantOpCreate(mlir_ctx, loc, i256_ty, catch_attr);
    mlir.oraBlockAppendOwnedOperation(catch_block, catch_op);
    mlir.oraBlockAppendOwnedOperation(catch_block, mlir.oraYieldOpCreate(
        mlir_ctx,
        loc,
        &[_]mlir.MlirValue{mlir.oraOperationGetResult(catch_op, 0)},
        1,
    ));

    const encoded = try encoder.encodeOperation(try_stmt);
    try testing.expect(!encoder.isDegraded());

    var solver = try Solver.init(&z3_ctx, testing.allocator);
    defer solver.deinit();
    solver.assert(z3.Z3_mk_not(z3_ctx.ctx, z3.Z3_mk_eq(z3_ctx.ctx, encoded, expected)));
    try testing.expectEqual(@as(z3.Z3_lbool, z3.Z3_L_FALSE), solver.check());
}

test "direct equivalent ora.try_stmt result encodes exactly" {
    var z3_ctx = try Context.init(testing.allocator);
    defer z3_ctx.deinit();

    var encoder = Encoder.init(&z3_ctx, testing.allocator);
    defer encoder.deinit();

    const mlir_ctx = mlir.oraContextCreate();
    defer mlir.oraContextDestroy(mlir_ctx);
    loadAllDialects(mlir_ctx);
    _ = mlir.oraDialectRegister(mlir_ctx);

    const loc = mlir.oraLocationUnknownGet(mlir_ctx);
    const i256_ty = mlir.oraIntegerTypeCreate(mlir_ctx, 256);

    const try_stmt = mlir.oraTryStmtOpCreate(mlir_ctx, loc, &[_]mlir.MlirType{i256_ty}, 1);
    const try_block = mlir.oraTryStmtOpGetTryBlock(try_stmt);
    const catch_block = mlir.oraTryStmtOpGetCatchBlock(try_stmt);

    const nested_try = mlir.oraTryStmtOpCreate(mlir_ctx, loc, &[_]mlir.MlirType{}, 0);
    const nested_try_block = mlir.oraTryStmtOpGetTryBlock(nested_try);
    const nested_catch_block = mlir.oraTryStmtOpGetCatchBlock(nested_try);
    mlir.oraBlockAppendOwnedOperation(nested_try_block, mlir.oraYieldOpCreate(mlir_ctx, loc, &[_]mlir.MlirValue{}, 0));
    mlir.oraBlockAppendOwnedOperation(nested_catch_block, mlir.oraYieldOpCreate(mlir_ctx, loc, &[_]mlir.MlirValue{}, 0));
    mlir.oraBlockAppendOwnedOperation(try_block, nested_try);

    const try_val_attr = mlir.oraIntegerAttrCreateI64FromType(i256_ty, 7);
    const catch_val_attr = mlir.oraIntegerAttrCreateI64FromType(i256_ty, 7);
    const try_val_op = mlir.oraArithConstantOpCreate(mlir_ctx, loc, i256_ty, try_val_attr);
    const catch_val_op = mlir.oraArithConstantOpCreate(mlir_ctx, loc, i256_ty, catch_val_attr);
    const expected = try encoder.encodeOperation(try_val_op);
    mlir.oraBlockAppendOwnedOperation(try_block, try_val_op);
    mlir.oraBlockAppendOwnedOperation(try_block, mlir.oraYieldOpCreate(
        mlir_ctx,
        loc,
        &[_]mlir.MlirValue{mlir.oraOperationGetResult(try_val_op, 0)},
        1,
    ));
    mlir.oraBlockAppendOwnedOperation(catch_block, catch_val_op);
    mlir.oraBlockAppendOwnedOperation(catch_block, mlir.oraYieldOpCreate(
        mlir_ctx,
        loc,
        &[_]mlir.MlirValue{mlir.oraOperationGetResult(catch_val_op, 0)},
        1,
    ));

    const encoded = try encoder.encodeValue(mlir.oraOperationGetResult(try_stmt, 0));
    try testing.expect(!encoder.isDegraded());

    var solver = try Solver.init(&z3_ctx, testing.allocator);
    defer solver.deinit();
    solver.assert(z3.Z3_mk_not(z3_ctx.ctx, z3.Z3_mk_eq(z3_ctx.ctx, encoded, expected)));
    try testing.expectEqual(@as(z3.Z3_lbool, z3.Z3_L_FALSE), solver.check());
}

test "direct always-catching ora.try_stmt result encodes exactly" {
    var z3_ctx = try Context.init(testing.allocator);
    defer z3_ctx.deinit();

    var encoder = Encoder.init(&z3_ctx, testing.allocator);
    defer encoder.deinit();

    const mlir_ctx = mlir.oraContextCreate();
    defer mlir.oraContextDestroy(mlir_ctx);
    loadAllDialects(mlir_ctx);
    _ = mlir.oraDialectRegister(mlir_ctx);

    const loc = mlir.oraLocationUnknownGet(mlir_ctx);
    const i256_ty = mlir.oraIntegerTypeCreate(mlir_ctx, 256);
    const eu_ty = mlir.oraErrorUnionTypeGet(mlir_ctx, i256_ty);

    const try_stmt = mlir.oraTryStmtOpCreate(mlir_ctx, loc, &[_]mlir.MlirType{i256_ty}, 1);
    const try_block = mlir.oraTryStmtOpGetTryBlock(try_stmt);
    const catch_block = mlir.oraTryStmtOpGetCatchBlock(try_stmt);

    const err_attr = mlir.oraIntegerAttrCreateI64FromType(i256_ty, 1);
    const err_val_op = mlir.oraArithConstantOpCreate(mlir_ctx, loc, i256_ty, err_attr);
    mlir.oraBlockAppendOwnedOperation(try_block, err_val_op);
    const err_op = mlir.oraErrorErrOpCreate(mlir_ctx, loc, mlir.oraOperationGetResult(err_val_op, 0), eu_ty);
    mlir.oraBlockAppendOwnedOperation(try_block, err_op);
    const unwrap_op = mlir.oraErrorUnwrapOpCreate(mlir_ctx, loc, mlir.oraOperationGetResult(err_op, 0), i256_ty);
    mlir.oraBlockAppendOwnedOperation(try_block, unwrap_op);

    const catch_attr = mlir.oraIntegerAttrCreateI64FromType(i256_ty, 13);
    const catch_op = mlir.oraArithConstantOpCreate(mlir_ctx, loc, i256_ty, catch_attr);
    const expected = try encoder.encodeOperation(catch_op);
    mlir.oraBlockAppendOwnedOperation(catch_block, catch_op);
    mlir.oraBlockAppendOwnedOperation(catch_block, mlir.oraYieldOpCreate(
        mlir_ctx,
        loc,
        &[_]mlir.MlirValue{mlir.oraOperationGetResult(catch_op, 0)},
        1,
    ));

    const encoded = try encoder.encodeValue(mlir.oraOperationGetResult(try_stmt, 0));
    try testing.expect(!encoder.isDegraded());

    var solver = try Solver.init(&z3_ctx, testing.allocator);
    defer solver.deinit();
    solver.assert(z3.Z3_mk_not(z3_ctx.ctx, z3.Z3_mk_eq(z3_ctx.ctx, encoded, expected)));
    try testing.expectEqual(@as(z3.Z3_lbool, z3.Z3_L_FALSE), solver.check());
}

test "known pure callee always-catching ora.try_stmt result encodes exactly" {
    var z3_ctx = try Context.init(testing.allocator);
    defer z3_ctx.deinit();

    var encoder = Encoder.init(&z3_ctx, testing.allocator);
    defer encoder.deinit();

    const mlir_ctx = mlir.oraContextCreate();
    defer mlir.oraContextDestroy(mlir_ctx);
    loadAllDialects(mlir_ctx);
    _ = mlir.oraDialectRegister(mlir_ctx);

    const loc = mlir.oraLocationUnknownGet(mlir_ctx);
    const i256_ty = mlir.oraIntegerTypeCreate(mlir_ctx, 256);
    const eu_ty = mlir.oraErrorUnionTypeGet(mlir_ctx, i256_ty);

    const helper_attrs = [_]mlir.MlirNamedAttribute{
        namedAttr(mlir_ctx, "sym_name", mlir.oraStringAttrCreate(mlir_ctx, stringRef("alwaysCatchTryHelper"))),
    };
    const helper = mlir.oraFuncFuncOpCreate(mlir_ctx, loc, &helper_attrs, helper_attrs.len, &[_]mlir.MlirType{}, &[_]mlir.MlirLocation{}, 0);
    const body = mlir.oraFuncOpGetBodyBlock(helper);

    const try_stmt = mlir.oraTryStmtOpCreate(mlir_ctx, loc, &[_]mlir.MlirType{i256_ty}, 1);
    const try_block = mlir.oraTryStmtOpGetTryBlock(try_stmt);
    const catch_block = mlir.oraTryStmtOpGetCatchBlock(try_stmt);

    const err_attr = mlir.oraIntegerAttrCreateI64FromType(i256_ty, 1);
    const err_val_op = mlir.oraArithConstantOpCreate(mlir_ctx, loc, i256_ty, err_attr);
    mlir.oraBlockAppendOwnedOperation(try_block, err_val_op);
    const err_op = mlir.oraErrorErrOpCreate(mlir_ctx, loc, mlir.oraOperationGetResult(err_val_op, 0), eu_ty);
    mlir.oraBlockAppendOwnedOperation(try_block, err_op);
    const unwrap_op = mlir.oraErrorUnwrapOpCreate(mlir_ctx, loc, mlir.oraOperationGetResult(err_op, 0), i256_ty);
    mlir.oraBlockAppendOwnedOperation(try_block, unwrap_op);

    const catch_attr = mlir.oraIntegerAttrCreateI64FromType(i256_ty, 21);
    const catch_op = mlir.oraArithConstantOpCreate(mlir_ctx, loc, i256_ty, catch_attr);
    mlir.oraBlockAppendOwnedOperation(catch_block, catch_op);
    mlir.oraBlockAppendOwnedOperation(catch_block, mlir.oraYieldOpCreate(
        mlir_ctx,
        loc,
        &[_]mlir.MlirValue{mlir.oraOperationGetResult(catch_op, 0)},
        1,
    ));

    mlir.oraBlockAppendOwnedOperation(body, try_stmt);
    mlir.oraBlockAppendOwnedOperation(body, mlir.oraReturnOpCreate(
        mlir_ctx,
        loc,
        &[_]mlir.MlirValue{mlir.oraOperationGetResult(try_stmt, 0)},
        1,
    ));

    try encoder.registerFunctionOperation(helper);

    const result_types = [_]mlir.MlirType{i256_ty};
    const call = mlir.oraFuncCallOpCreate(
        mlir_ctx,
        loc,
        stringRef("alwaysCatchTryHelper"),
        &[_]mlir.MlirValue{},
        0,
        &result_types,
        result_types.len,
    );

    const encoded = try encoder.encodeOperation(call);
    const expected = try encoder.encodeIntegerConstant(21, 256);
    try testing.expect(!encoder.isDegraded());

    var solver = try Solver.init(&z3_ctx, testing.allocator);
    defer solver.deinit();
    solver.assert(z3.Z3_mk_not(z3_ctx.ctx, z3.Z3_mk_eq(z3_ctx.ctx, encoded, expected)));
    try testing.expectEqual(@as(z3.Z3_lbool, z3.Z3_L_FALSE), solver.check());
}

test "known pure callee execute_region symbolic ora.try_stmt result encodes exactly" {
    var z3_ctx = try Context.init(testing.allocator);
    defer z3_ctx.deinit();

    var encoder = Encoder.init(&z3_ctx, testing.allocator);
    defer encoder.deinit();

    const mlir_ctx = mlir.oraContextCreate();
    defer mlir.oraContextDestroy(mlir_ctx);
    loadAllDialects(mlir_ctx);
    _ = mlir.oraDialectRegister(mlir_ctx);

    const loc = mlir.oraLocationUnknownGet(mlir_ctx);
    const i256_ty = mlir.oraIntegerTypeCreate(mlir_ctx, 256);
    const eu_ty = mlir.oraErrorUnionTypeGet(mlir_ctx, i256_ty);

    const maybe_value = mlir.oraVariablePlaceholderOpCreate(mlir_ctx, loc, stringRef("maybeValue"), eu_ty);

    const helper_attrs = [_]mlir.MlirNamedAttribute{
        namedAttr(mlir_ctx, "sym_name", mlir.oraStringAttrCreate(mlir_ctx, stringRef("executeRegionTryHelper"))),
    };
    const helper = mlir.oraFuncFuncOpCreate(mlir_ctx, loc, &helper_attrs, helper_attrs.len, &[_]mlir.MlirType{}, &[_]mlir.MlirLocation{}, 0);
    const body = mlir.oraFuncOpGetBodyBlock(helper);

    const try_stmt = mlir.oraTryStmtOpCreate(mlir_ctx, loc, &[_]mlir.MlirType{i256_ty}, 1);
    const try_block = mlir.oraTryStmtOpGetTryBlock(try_stmt);
    const catch_block = mlir.oraTryStmtOpGetCatchBlock(try_stmt);

    const exec = mlir.oraScfExecuteRegionOpCreate(mlir_ctx, loc, &[_]mlir.MlirType{i256_ty}, 1, false);
    const exec_block = mlir.oraScfExecuteRegionOpGetBodyBlock(exec);
    const unwrap_op = mlir.oraErrorUnwrapOpCreate(mlir_ctx, loc, mlir.oraOperationGetResult(maybe_value, 0), i256_ty);
    mlir.oraBlockAppendOwnedOperation(exec_block, unwrap_op);
    mlir.oraBlockAppendOwnedOperation(exec_block, mlir.oraScfYieldOpCreate(
        mlir_ctx,
        loc,
        &[_]mlir.MlirValue{mlir.oraOperationGetResult(unwrap_op, 0)},
        1,
    ));
    mlir.oraBlockAppendOwnedOperation(try_block, exec);
    mlir.oraBlockAppendOwnedOperation(try_block, mlir.oraYieldOpCreate(
        mlir_ctx,
        loc,
        &[_]mlir.MlirValue{mlir.oraOperationGetResult(exec, 0)},
        1,
    ));

    const catch_attr = mlir.oraIntegerAttrCreateI64FromType(i256_ty, 19);
    const catch_op = mlir.oraArithConstantOpCreate(mlir_ctx, loc, i256_ty, catch_attr);
    mlir.oraBlockAppendOwnedOperation(catch_block, catch_op);
    mlir.oraBlockAppendOwnedOperation(catch_block, mlir.oraYieldOpCreate(
        mlir_ctx,
        loc,
        &[_]mlir.MlirValue{mlir.oraOperationGetResult(catch_op, 0)},
        1,
    ));

    mlir.oraBlockAppendOwnedOperation(body, try_stmt);
    mlir.oraBlockAppendOwnedOperation(body, mlir.oraReturnOpCreate(
        mlir_ctx,
        loc,
        &[_]mlir.MlirValue{mlir.oraOperationGetResult(try_stmt, 0)},
        1,
    ));

    try encoder.registerFunctionOperation(helper);

    const result_types = [_]mlir.MlirType{i256_ty};
    const call = mlir.oraFuncCallOpCreate(
        mlir_ctx,
        loc,
        stringRef("executeRegionTryHelper"),
        &[_]mlir.MlirValue{},
        0,
        &result_types,
        result_types.len,
    );

    const encoded = try encoder.encodeOperation(call);
    try testing.expect(!encoder.isDegraded());
    const encoded_text = std.mem.span(z3.Z3_ast_to_string(z3_ctx.ctx, encoded));
    try testing.expect(std.mem.indexOf(u8, encoded_text, "ite") != null);
    try testing.expect(std.mem.indexOf(u8, encoded_text, "maybeValue") != null);
}

test "direct symbolic error-union ora.try_stmt result encodes exactly" {
    var z3_ctx = try Context.init(testing.allocator);
    defer z3_ctx.deinit();

    var encoder = Encoder.init(&z3_ctx, testing.allocator);
    defer encoder.deinit();

    const mlir_ctx = mlir.oraContextCreate();
    defer mlir.oraContextDestroy(mlir_ctx);
    loadAllDialects(mlir_ctx);
    _ = mlir.oraDialectRegister(mlir_ctx);

    const loc = mlir.oraLocationUnknownGet(mlir_ctx);
    const i256_ty = mlir.oraIntegerTypeCreate(mlir_ctx, 256);
    const eu_ty = mlir.oraErrorUnionTypeGet(mlir_ctx, i256_ty);

    const try_stmt = mlir.oraTryStmtOpCreate(mlir_ctx, loc, &[_]mlir.MlirType{i256_ty}, 1);
    const try_block = mlir.oraTryStmtOpGetTryBlock(try_stmt);
    const catch_block = mlir.oraTryStmtOpGetCatchBlock(try_stmt);

    const call = mlir.oraFuncCallOpCreate(mlir_ctx, loc, stringRef("maybeValue"), &[_]mlir.MlirValue{}, 0, &[_]mlir.MlirType{eu_ty}, 1);
    mlir.oraBlockAppendOwnedOperation(try_block, call);
    const call_result = mlir.oraOperationGetResult(call, 0);
    const unwrap_op = mlir.oraErrorUnwrapOpCreate(mlir_ctx, loc, call_result, i256_ty);
    mlir.oraBlockAppendOwnedOperation(try_block, unwrap_op);
    mlir.oraBlockAppendOwnedOperation(try_block, mlir.oraYieldOpCreate(
        mlir_ctx,
        loc,
        &[_]mlir.MlirValue{mlir.oraOperationGetResult(unwrap_op, 0)},
        1,
    ));

    const catch_attr = mlir.oraIntegerAttrCreateI64FromType(i256_ty, 21);
    const catch_op = mlir.oraArithConstantOpCreate(mlir_ctx, loc, i256_ty, catch_attr);
    mlir.oraBlockAppendOwnedOperation(catch_block, catch_op);
    mlir.oraBlockAppendOwnedOperation(catch_block, mlir.oraYieldOpCreate(
        mlir_ctx,
        loc,
        &[_]mlir.MlirValue{mlir.oraOperationGetResult(catch_op, 0)},
        1,
    ));

    const encoded = try encoder.encodeValue(mlir.oraOperationGetResult(try_stmt, 0));
    try testing.expect(!encoder.isDegraded());
    const encoded_text = std.mem.span(z3.Z3_ast_to_string(z3_ctx.ctx, encoded));
    try testing.expect(std.mem.indexOf(u8, encoded_text, "ite") != null);
    try testing.expect(std.mem.indexOf(u8, encoded_text, "maybeValue") != null);
}

test "direct symbolic error-union ora.try_stmt result with state effects encodes exactly" {
    var z3_ctx = try Context.init(testing.allocator);
    defer z3_ctx.deinit();

    var encoder = Encoder.init(&z3_ctx, testing.allocator);
    defer encoder.deinit();

    const mlir_ctx = mlir.oraContextCreate();
    defer mlir.oraContextDestroy(mlir_ctx);
    loadAllDialects(mlir_ctx);
    _ = mlir.oraDialectRegister(mlir_ctx);

    const loc = mlir.oraLocationUnknownGet(mlir_ctx);
    const i256_ty = mlir.oraIntegerTypeCreate(mlir_ctx, 256);
    const eu_ty = mlir.oraErrorUnionTypeGet(mlir_ctx, i256_ty);

    const maybe_value = mlir.oraVariablePlaceholderOpCreate(mlir_ctx, loc, stringRef("maybeValue"), eu_ty);
    const maybe_result = mlir.oraOperationGetResult(maybe_value, 0);

    const try_stmt = mlir.oraTryStmtOpCreate(mlir_ctx, loc, &[_]mlir.MlirType{i256_ty}, 1);
    const try_block = mlir.oraTryStmtOpGetTryBlock(try_stmt);
    const catch_block = mlir.oraTryStmtOpGetCatchBlock(try_stmt);

    const unwrap_op = mlir.oraErrorUnwrapOpCreate(mlir_ctx, loc, maybe_result, i256_ty);
    mlir.oraBlockAppendOwnedOperation(try_block, unwrap_op);
    const try_store_attr = mlir.oraIntegerAttrCreateI64FromType(i256_ty, 11);
    const try_store_op = mlir.oraArithConstantOpCreate(mlir_ctx, loc, i256_ty, try_store_attr);
    mlir.oraBlockAppendOwnedOperation(try_block, try_store_op);
    mlir.oraBlockAppendOwnedOperation(try_block, mlir.oraSStoreOpCreate(
        mlir_ctx,
        loc,
        mlir.oraOperationGetResult(try_store_op, 0),
        stringRef("counter"),
    ));
    mlir.oraBlockAppendOwnedOperation(try_block, mlir.oraYieldOpCreate(
        mlir_ctx,
        loc,
        &[_]mlir.MlirValue{mlir.oraOperationGetResult(unwrap_op, 0)},
        1,
    ));

    const catch_store_attr = mlir.oraIntegerAttrCreateI64FromType(i256_ty, 33);
    const catch_store_op = mlir.oraArithConstantOpCreate(mlir_ctx, loc, i256_ty, catch_store_attr);
    mlir.oraBlockAppendOwnedOperation(catch_block, catch_store_op);
    mlir.oraBlockAppendOwnedOperation(catch_block, mlir.oraSStoreOpCreate(
        mlir_ctx,
        loc,
        mlir.oraOperationGetResult(catch_store_op, 0),
        stringRef("counter"),
    ));
    mlir.oraBlockAppendOwnedOperation(catch_block, mlir.oraYieldOpCreate(
        mlir_ctx,
        loc,
        &[_]mlir.MlirValue{mlir.oraOperationGetResult(catch_store_op, 0)},
        1,
    ));

    const encoded = try encoder.encodeValue(mlir.oraOperationGetResult(try_stmt, 0));
    try testing.expect(!encoder.isDegraded());

    const encoded_text = std.mem.span(z3.Z3_ast_to_string(z3_ctx.ctx, encoded));
    try testing.expect(std.mem.indexOf(u8, encoded_text, "ite") != null);
    try testing.expect(std.mem.indexOf(u8, encoded_text, "maybeValue") != null);
}

test "direct branch-conditioned ora.try_stmt result encodes exactly" {
    var z3_ctx = try Context.init(testing.allocator);
    defer z3_ctx.deinit();

    var encoder = Encoder.init(&z3_ctx, testing.allocator);
    defer encoder.deinit();

    const mlir_ctx = mlir.oraContextCreate();
    defer mlir.oraContextDestroy(mlir_ctx);
    loadAllDialects(mlir_ctx);
    _ = mlir.oraDialectRegister(mlir_ctx);

    const loc = mlir.oraLocationUnknownGet(mlir_ctx);
    const i1_ty = mlir.oraIntegerTypeCreate(mlir_ctx, 1);
    const i256_ty = mlir.oraIntegerTypeCreate(mlir_ctx, 256);
    const eu_ty = mlir.oraErrorUnionTypeGet(mlir_ctx, i256_ty);

    const cond_op = mlir.oraVariablePlaceholderOpCreate(mlir_ctx, loc, stringRef("condValue"), i1_ty);
    const maybe_op = mlir.oraVariablePlaceholderOpCreate(mlir_ctx, loc, stringRef("maybeValue"), eu_ty);

    const try_stmt = mlir.oraTryStmtOpCreate(mlir_ctx, loc, &[_]mlir.MlirType{i256_ty}, 1);
    const try_block = mlir.oraTryStmtOpGetTryBlock(try_stmt);
    const catch_block = mlir.oraTryStmtOpGetCatchBlock(try_stmt);

    const if_op = mlir.oraScfIfOpCreate(
        mlir_ctx,
        loc,
        mlir.oraOperationGetResult(cond_op, 0),
        &[_]mlir.MlirType{i256_ty},
        1,
        true,
    );
    const then_block = mlir.oraScfIfOpGetThenBlock(if_op);
    const else_block = mlir.oraScfIfOpGetElseBlock(if_op);

    const unwrap_op = mlir.oraErrorUnwrapOpCreate(mlir_ctx, loc, mlir.oraOperationGetResult(maybe_op, 0), i256_ty);
    mlir.oraBlockAppendOwnedOperation(then_block, unwrap_op);
    mlir.oraBlockAppendOwnedOperation(then_block, mlir.oraScfYieldOpCreate(
        mlir_ctx,
        loc,
        &[_]mlir.MlirValue{mlir.oraOperationGetResult(unwrap_op, 0)},
        1,
    ));

    const else_attr = mlir.oraIntegerAttrCreateI64FromType(i256_ty, 7);
    const else_op = mlir.oraArithConstantOpCreate(mlir_ctx, loc, i256_ty, else_attr);
    mlir.oraBlockAppendOwnedOperation(else_block, else_op);
    mlir.oraBlockAppendOwnedOperation(else_block, mlir.oraScfYieldOpCreate(
        mlir_ctx,
        loc,
        &[_]mlir.MlirValue{mlir.oraOperationGetResult(else_op, 0)},
        1,
    ));

    mlir.oraBlockAppendOwnedOperation(try_block, if_op);
    mlir.oraBlockAppendOwnedOperation(try_block, mlir.oraYieldOpCreate(
        mlir_ctx,
        loc,
        &[_]mlir.MlirValue{mlir.oraOperationGetResult(if_op, 0)},
        1,
    ));

    const catch_attr = mlir.oraIntegerAttrCreateI64FromType(i256_ty, 21);
    const catch_op = mlir.oraArithConstantOpCreate(mlir_ctx, loc, i256_ty, catch_attr);
    mlir.oraBlockAppendOwnedOperation(catch_block, catch_op);
    mlir.oraBlockAppendOwnedOperation(catch_block, mlir.oraYieldOpCreate(
        mlir_ctx,
        loc,
        &[_]mlir.MlirValue{mlir.oraOperationGetResult(catch_op, 0)},
        1,
    ));

    const encoded = try encoder.encodeValue(mlir.oraOperationGetResult(try_stmt, 0));
    try testing.expect(!encoder.isDegraded());
    const encoded_text = std.mem.span(z3.Z3_ast_to_string(z3_ctx.ctx, encoded));
    try testing.expect(std.mem.indexOf(u8, encoded_text, "ite") != null);
    try testing.expect(std.mem.indexOf(u8, encoded_text, "condValue") != null);
    try testing.expect(std.mem.indexOf(u8, encoded_text, "maybeValue") != null);
}

test "direct execute_region symbolic ora.try_stmt result encodes exactly" {
    var z3_ctx = try Context.init(testing.allocator);
    defer z3_ctx.deinit();

    var encoder = Encoder.init(&z3_ctx, testing.allocator);
    defer encoder.deinit();

    const mlir_ctx = mlir.oraContextCreate();
    defer mlir.oraContextDestroy(mlir_ctx);
    loadAllDialects(mlir_ctx);
    _ = mlir.oraDialectRegister(mlir_ctx);

    const loc = mlir.oraLocationUnknownGet(mlir_ctx);
    const i256_ty = mlir.oraIntegerTypeCreate(mlir_ctx, 256);
    const eu_ty = mlir.oraErrorUnionTypeGet(mlir_ctx, i256_ty);

    const maybe_value = mlir.oraVariablePlaceholderOpCreate(mlir_ctx, loc, stringRef("maybeValue"), eu_ty);

    const try_stmt = mlir.oraTryStmtOpCreate(mlir_ctx, loc, &[_]mlir.MlirType{i256_ty}, 1);
    const try_block = mlir.oraTryStmtOpGetTryBlock(try_stmt);
    const catch_block = mlir.oraTryStmtOpGetCatchBlock(try_stmt);

    const exec = mlir.oraScfExecuteRegionOpCreate(mlir_ctx, loc, &[_]mlir.MlirType{i256_ty}, 1, false);
    const exec_block = mlir.oraScfExecuteRegionOpGetBodyBlock(exec);
    const unwrap_op = mlir.oraErrorUnwrapOpCreate(mlir_ctx, loc, mlir.oraOperationGetResult(maybe_value, 0), i256_ty);
    mlir.oraBlockAppendOwnedOperation(exec_block, unwrap_op);
    mlir.oraBlockAppendOwnedOperation(exec_block, mlir.oraScfYieldOpCreate(
        mlir_ctx,
        loc,
        &[_]mlir.MlirValue{mlir.oraOperationGetResult(unwrap_op, 0)},
        1,
    ));
    mlir.oraBlockAppendOwnedOperation(try_block, exec);
    mlir.oraBlockAppendOwnedOperation(try_block, mlir.oraYieldOpCreate(
        mlir_ctx,
        loc,
        &[_]mlir.MlirValue{mlir.oraOperationGetResult(exec, 0)},
        1,
    ));

    const catch_attr = mlir.oraIntegerAttrCreateI64FromType(i256_ty, 17);
    const catch_op = mlir.oraArithConstantOpCreate(mlir_ctx, loc, i256_ty, catch_attr);
    mlir.oraBlockAppendOwnedOperation(catch_block, catch_op);
    mlir.oraBlockAppendOwnedOperation(catch_block, mlir.oraYieldOpCreate(
        mlir_ctx,
        loc,
        &[_]mlir.MlirValue{mlir.oraOperationGetResult(catch_op, 0)},
        1,
    ));

    const encoded = try encoder.encodeValue(mlir.oraOperationGetResult(try_stmt, 0));
    try testing.expect(!encoder.isDegraded());
    const encoded_text = std.mem.span(z3.Z3_ast_to_string(z3_ctx.ctx, encoded));
    try testing.expect(std.mem.indexOf(u8, encoded_text, "ite") != null);
    try testing.expect(std.mem.indexOf(u8, encoded_text, "maybeValue") != null);
}

test "direct switch_expr symbolic ora.try_stmt result encodes exactly" {
    var z3_ctx = try Context.init(testing.allocator);
    defer z3_ctx.deinit();

    var encoder = Encoder.init(&z3_ctx, testing.allocator);
    defer encoder.deinit();

    const mlir_ctx = mlir.oraContextCreate();
    defer mlir.oraContextDestroy(mlir_ctx);
    loadAllDialects(mlir_ctx);
    _ = mlir.oraDialectRegister(mlir_ctx);

    const loc = mlir.oraLocationUnknownGet(mlir_ctx);
    const i1_ty = mlir.oraIntegerTypeCreate(mlir_ctx, 1);
    const i256_ty = mlir.oraIntegerTypeCreate(mlir_ctx, 256);
    const eu_ty = mlir.oraErrorUnionTypeGet(mlir_ctx, i256_ty);

    const cond = mlir.oraVariablePlaceholderOpCreate(mlir_ctx, loc, stringRef("condValue"), i1_ty);
    const maybe_value = mlir.oraVariablePlaceholderOpCreate(mlir_ctx, loc, stringRef("maybeValue"), eu_ty);

    const try_stmt = mlir.oraTryStmtOpCreate(mlir_ctx, loc, &[_]mlir.MlirType{i256_ty}, 1);
    const try_block = mlir.oraTryStmtOpGetTryBlock(try_stmt);
    const catch_block = mlir.oraTryStmtOpGetCatchBlock(try_stmt);

    const switch_expr = mlir.oraSwitchExprOpCreateWithCases(
        mlir_ctx,
        loc,
        mlir.oraOperationGetResult(cond, 0),
        &[_]mlir.MlirType{i256_ty},
        1,
        2,
    );
    const case_values = [_]i64{ 0, 1 };
    const range_starts = [_]i64{ 0, 0 };
    const range_ends = [_]i64{ 0, 0 };
    const case_kinds = [_]i64{ 0, 0 };
    mlir.oraSwitchOpSetCasePatterns(
        switch_expr,
        &case_values,
        &range_starts,
        &range_ends,
        &case_kinds,
        -1,
        case_values.len,
    );
    const false_block = mlir.oraSwitchExprOpGetCaseBlock(switch_expr, 0);
    const true_block = mlir.oraSwitchExprOpGetCaseBlock(switch_expr, 1);

    const false_attr = mlir.oraIntegerAttrCreateI64FromType(i256_ty, 7);
    const false_op = mlir.oraArithConstantOpCreate(mlir_ctx, loc, i256_ty, false_attr);
    mlir.oraBlockAppendOwnedOperation(false_block, false_op);
    mlir.oraBlockAppendOwnedOperation(false_block, mlir.oraYieldOpCreate(
        mlir_ctx,
        loc,
        &[_]mlir.MlirValue{mlir.oraOperationGetResult(false_op, 0)},
        1,
    ));

    const unwrap_op = mlir.oraErrorUnwrapOpCreate(mlir_ctx, loc, mlir.oraOperationGetResult(maybe_value, 0), i256_ty);
    mlir.oraBlockAppendOwnedOperation(true_block, unwrap_op);
    mlir.oraBlockAppendOwnedOperation(true_block, mlir.oraYieldOpCreate(
        mlir_ctx,
        loc,
        &[_]mlir.MlirValue{mlir.oraOperationGetResult(unwrap_op, 0)},
        1,
    ));

    mlir.oraBlockAppendOwnedOperation(try_block, switch_expr);
    mlir.oraBlockAppendOwnedOperation(try_block, mlir.oraYieldOpCreate(
        mlir_ctx,
        loc,
        &[_]mlir.MlirValue{mlir.oraOperationGetResult(switch_expr, 0)},
        1,
    ));

    const catch_attr = mlir.oraIntegerAttrCreateI64FromType(i256_ty, 23);
    const catch_op = mlir.oraArithConstantOpCreate(mlir_ctx, loc, i256_ty, catch_attr);
    mlir.oraBlockAppendOwnedOperation(catch_block, catch_op);
    mlir.oraBlockAppendOwnedOperation(catch_block, mlir.oraYieldOpCreate(
        mlir_ctx,
        loc,
        &[_]mlir.MlirValue{mlir.oraOperationGetResult(catch_op, 0)},
        1,
    ));

    const encoded = try encoder.encodeValue(mlir.oraOperationGetResult(try_stmt, 0));
    try testing.expect(!encoder.isDegraded());
    const encoded_text = std.mem.span(z3.Z3_ast_to_string(z3_ctx.ctx, encoded));
    try testing.expect(std.mem.indexOf(u8, encoded_text, "ite") != null);
    try testing.expect(std.mem.indexOf(u8, encoded_text, "condValue") != null);
    try testing.expect(std.mem.indexOf(u8, encoded_text, "maybeValue") != null);
}

test "direct ora.try_stmt ignores catch-capable dead branch" {
    var z3_ctx = try Context.init(testing.allocator);
    defer z3_ctx.deinit();

    var encoder = Encoder.init(&z3_ctx, testing.allocator);
    defer encoder.deinit();

    const mlir_ctx = mlir.oraContextCreate();
    defer mlir.oraContextDestroy(mlir_ctx);
    loadAllDialects(mlir_ctx);
    _ = mlir.oraDialectRegister(mlir_ctx);

    const loc = mlir.oraLocationUnknownGet(mlir_ctx);
    const i1_ty = mlir.oraIntegerTypeCreate(mlir_ctx, 1);
    const i256_ty = mlir.oraIntegerTypeCreate(mlir_ctx, 256);
    const eu_ty = mlir.oraErrorUnionTypeGet(mlir_ctx, i256_ty);

    const try_stmt = mlir.oraTryStmtOpCreate(mlir_ctx, loc, &[_]mlir.MlirType{i256_ty}, 1);
    const try_block = mlir.oraTryStmtOpGetTryBlock(try_stmt);
    const catch_block = mlir.oraTryStmtOpGetCatchBlock(try_stmt);

    const false_attr = mlir.oraIntegerAttrCreateI64FromType(i1_ty, 0);
    const false_op = mlir.oraArithConstantOpCreate(mlir_ctx, loc, i1_ty, false_attr);
    mlir.oraBlockAppendOwnedOperation(try_block, false_op);

    const if_op = mlir.oraScfIfOpCreate(
        mlir_ctx,
        loc,
        mlir.oraOperationGetResult(false_op, 0),
        &[_]mlir.MlirType{i256_ty},
        1,
        true,
    );
    const then_block = mlir.oraScfIfOpGetThenBlock(if_op);
    const else_block = mlir.oraScfIfOpGetElseBlock(if_op);

    const err_attr = mlir.oraIntegerAttrCreateI64FromType(i256_ty, 1);
    const err_val_op = mlir.oraArithConstantOpCreate(mlir_ctx, loc, i256_ty, err_attr);
    mlir.oraBlockAppendOwnedOperation(then_block, err_val_op);
    const err_op = mlir.oraErrorErrOpCreate(mlir_ctx, loc, mlir.oraOperationGetResult(err_val_op, 0), eu_ty);
    mlir.oraBlockAppendOwnedOperation(then_block, err_op);
    const unwrap_op = mlir.oraErrorUnwrapOpCreate(mlir_ctx, loc, mlir.oraOperationGetResult(err_op, 0), i256_ty);
    mlir.oraBlockAppendOwnedOperation(then_block, unwrap_op);
    mlir.oraBlockAppendOwnedOperation(then_block, mlir.oraScfYieldOpCreate(
        mlir_ctx,
        loc,
        &[_]mlir.MlirValue{mlir.oraOperationGetResult(unwrap_op, 0)},
        1,
    ));

    const seven_attr = mlir.oraIntegerAttrCreateI64FromType(i256_ty, 7);
    const seven_op = mlir.oraArithConstantOpCreate(mlir_ctx, loc, i256_ty, seven_attr);
    const expected = try encoder.encodeOperation(seven_op);
    mlir.oraBlockAppendOwnedOperation(else_block, seven_op);
    mlir.oraBlockAppendOwnedOperation(else_block, mlir.oraScfYieldOpCreate(
        mlir_ctx,
        loc,
        &[_]mlir.MlirValue{mlir.oraOperationGetResult(seven_op, 0)},
        1,
    ));

    mlir.oraBlockAppendOwnedOperation(try_block, if_op);
    mlir.oraBlockAppendOwnedOperation(try_block, mlir.oraYieldOpCreate(
        mlir_ctx,
        loc,
        &[_]mlir.MlirValue{mlir.oraOperationGetResult(if_op, 0)},
        1,
    ));

    const catch_val_attr = mlir.oraIntegerAttrCreateI64FromType(i256_ty, 9);
    const catch_val_op = mlir.oraArithConstantOpCreate(mlir_ctx, loc, i256_ty, catch_val_attr);
    mlir.oraBlockAppendOwnedOperation(catch_block, catch_val_op);
    mlir.oraBlockAppendOwnedOperation(catch_block, mlir.oraYieldOpCreate(
        mlir_ctx,
        loc,
        &[_]mlir.MlirValue{mlir.oraOperationGetResult(catch_val_op, 0)},
        1,
    ));

    const encoded = try encoder.encodeValue(mlir.oraOperationGetResult(try_stmt, 0));
    try testing.expect(!encoder.isDegraded());

    var solver = try Solver.init(&z3_ctx, testing.allocator);
    defer solver.deinit();
    solver.assert(z3.Z3_mk_not(z3_ctx.ctx, z3.Z3_mk_eq(z3_ctx.ctx, encoded, expected)));
    try testing.expectEqual(@as(z3.Z3_lbool, z3.Z3_L_FALSE), solver.check());
}

test "direct ora.try_stmt ignores catch-capable dead switch branch" {
    var z3_ctx = try Context.init(testing.allocator);
    defer z3_ctx.deinit();

    var encoder = Encoder.init(&z3_ctx, testing.allocator);
    defer encoder.deinit();

    const mlir_ctx = mlir.oraContextCreate();
    defer mlir.oraContextDestroy(mlir_ctx);
    loadAllDialects(mlir_ctx);
    _ = mlir.oraDialectRegister(mlir_ctx);

    const loc = mlir.oraLocationUnknownGet(mlir_ctx);
    const i256_ty = mlir.oraIntegerTypeCreate(mlir_ctx, 256);
    const eu_ty = mlir.oraErrorUnionTypeGet(mlir_ctx, i256_ty);

    const try_stmt = mlir.oraTryStmtOpCreate(mlir_ctx, loc, &[_]mlir.MlirType{i256_ty}, 1);
    const try_block = mlir.oraTryStmtOpGetTryBlock(try_stmt);
    const catch_block = mlir.oraTryStmtOpGetCatchBlock(try_stmt);

    const tag_attr = mlir.oraIntegerAttrCreateI64FromType(i256_ty, 2);
    const tag_op = mlir.oraArithConstantOpCreate(mlir_ctx, loc, i256_ty, tag_attr);
    mlir.oraBlockAppendOwnedOperation(try_block, tag_op);

    const switch_op = mlir.oraSwitchOpCreateWithCases(
        mlir_ctx,
        loc,
        mlir.oraOperationGetResult(tag_op, 0),
        &[_]mlir.MlirType{i256_ty},
        1,
        2,
    );
    const case0_block = mlir.oraSwitchOpGetCaseBlock(switch_op, 0);
    const default_block = mlir.oraSwitchOpGetCaseBlock(switch_op, 1);

    const err_attr = mlir.oraIntegerAttrCreateI64FromType(i256_ty, 1);
    const err_val_op = mlir.oraArithConstantOpCreate(mlir_ctx, loc, i256_ty, err_attr);
    mlir.oraBlockAppendOwnedOperation(case0_block, err_val_op);
    const err_op = mlir.oraErrorErrOpCreate(mlir_ctx, loc, mlir.oraOperationGetResult(err_val_op, 0), eu_ty);
    mlir.oraBlockAppendOwnedOperation(case0_block, err_op);
    const unwrap_op = mlir.oraErrorUnwrapOpCreate(mlir_ctx, loc, mlir.oraOperationGetResult(err_op, 0), i256_ty);
    mlir.oraBlockAppendOwnedOperation(case0_block, unwrap_op);
    mlir.oraBlockAppendOwnedOperation(case0_block, mlir.oraYieldOpCreate(
        mlir_ctx,
        loc,
        &[_]mlir.MlirValue{mlir.oraOperationGetResult(unwrap_op, 0)},
        1,
    ));

    const seven_attr = mlir.oraIntegerAttrCreateI64FromType(i256_ty, 7);
    const seven_op = mlir.oraArithConstantOpCreate(mlir_ctx, loc, i256_ty, seven_attr);
    const expected = try encoder.encodeOperation(seven_op);
    mlir.oraBlockAppendOwnedOperation(default_block, seven_op);
    mlir.oraBlockAppendOwnedOperation(default_block, mlir.oraYieldOpCreate(
        mlir_ctx,
        loc,
        &[_]mlir.MlirValue{mlir.oraOperationGetResult(seven_op, 0)},
        1,
    ));

    var case_values = [_]i64{ 1, 0 };
    var range_starts = [_]i64{ 0, 0 };
    var range_ends = [_]i64{ 0, 0 };
    var case_kinds = [_]i64{ 0, 2 };
    mlir.oraSwitchOpSetCasePatterns(switch_op, &case_values, &range_starts, &range_ends, &case_kinds, 1, 2);

    mlir.oraBlockAppendOwnedOperation(try_block, switch_op);
    mlir.oraBlockAppendOwnedOperation(try_block, mlir.oraYieldOpCreate(
        mlir_ctx,
        loc,
        &[_]mlir.MlirValue{mlir.oraOperationGetResult(switch_op, 0)},
        1,
    ));

    const catch_val_attr = mlir.oraIntegerAttrCreateI64FromType(i256_ty, 9);
    const catch_val_op = mlir.oraArithConstantOpCreate(mlir_ctx, loc, i256_ty, catch_val_attr);
    mlir.oraBlockAppendOwnedOperation(catch_block, catch_val_op);
    mlir.oraBlockAppendOwnedOperation(catch_block, mlir.oraYieldOpCreate(
        mlir_ctx,
        loc,
        &[_]mlir.MlirValue{mlir.oraOperationGetResult(catch_val_op, 0)},
        1,
    ));

    const encoded = try encoder.encodeValue(mlir.oraOperationGetResult(try_stmt, 0));
    try testing.expect(!encoder.isDegraded());

    var solver = try Solver.init(&z3_ctx, testing.allocator);
    defer solver.deinit();
    solver.assert(z3.Z3_mk_not(z3_ctx.ctx, z3.Z3_mk_eq(z3_ctx.ctx, encoded, expected)));
    try testing.expectEqual(@as(z3.Z3_lbool, z3.Z3_L_FALSE), solver.check());
}

test "direct ora.try_stmt ignores catch-capable dead conditional return branch" {
    var z3_ctx = try Context.init(testing.allocator);
    defer z3_ctx.deinit();

    var encoder = Encoder.init(&z3_ctx, testing.allocator);
    defer encoder.deinit();

    const mlir_ctx = mlir.oraContextCreate();
    defer mlir.oraContextDestroy(mlir_ctx);
    loadAllDialects(mlir_ctx);
    _ = mlir.oraDialectRegister(mlir_ctx);

    const loc = mlir.oraLocationUnknownGet(mlir_ctx);
    const i1_ty = mlir.oraIntegerTypeCreate(mlir_ctx, 1);
    const i256_ty = mlir.oraIntegerTypeCreate(mlir_ctx, 256);
    const eu_ty = mlir.oraErrorUnionTypeGet(mlir_ctx, i256_ty);

    const try_stmt = mlir.oraTryStmtOpCreate(mlir_ctx, loc, &[_]mlir.MlirType{i256_ty}, 1);
    const try_block = mlir.oraTryStmtOpGetTryBlock(try_stmt);
    const catch_block = mlir.oraTryStmtOpGetCatchBlock(try_stmt);

    const false_attr = mlir.oraIntegerAttrCreateI64FromType(i1_ty, 0);
    const false_op = mlir.oraArithConstantOpCreate(mlir_ctx, loc, i1_ty, false_attr);
    mlir.oraBlockAppendOwnedOperation(try_block, false_op);

    const conditional_ret = mlir.oraConditionalReturnOpCreate(mlir_ctx, loc, mlir.oraOperationGetResult(false_op, 0));
    const then_block = mlir.oraConditionalReturnOpGetThenBlock(conditional_ret);
    const else_block = mlir.oraConditionalReturnOpGetElseBlock(conditional_ret);

    const err_attr = mlir.oraIntegerAttrCreateI64FromType(i256_ty, 1);
    const err_val_op = mlir.oraArithConstantOpCreate(mlir_ctx, loc, i256_ty, err_attr);
    mlir.oraBlockAppendOwnedOperation(then_block, err_val_op);
    const err_op = mlir.oraErrorErrOpCreate(mlir_ctx, loc, mlir.oraOperationGetResult(err_val_op, 0), eu_ty);
    mlir.oraBlockAppendOwnedOperation(then_block, err_op);
    const unwrap_op = mlir.oraErrorUnwrapOpCreate(mlir_ctx, loc, mlir.oraOperationGetResult(err_op, 0), i256_ty);
    mlir.oraBlockAppendOwnedOperation(then_block, unwrap_op);
    mlir.oraBlockAppendOwnedOperation(then_block, mlir.oraReturnOpCreate(
        mlir_ctx,
        loc,
        &[_]mlir.MlirValue{mlir.oraOperationGetResult(unwrap_op, 0)},
        1,
    ));

    mlir.oraBlockAppendOwnedOperation(else_block, mlir.oraYieldOpCreate(mlir_ctx, loc, &[_]mlir.MlirValue{}, 0));

    const seven_attr = mlir.oraIntegerAttrCreateI64FromType(i256_ty, 7);
    const seven_op = mlir.oraArithConstantOpCreate(mlir_ctx, loc, i256_ty, seven_attr);
    const expected = try encoder.encodeOperation(seven_op);
    mlir.oraBlockAppendOwnedOperation(try_block, conditional_ret);
    mlir.oraBlockAppendOwnedOperation(try_block, seven_op);
    mlir.oraBlockAppendOwnedOperation(try_block, mlir.oraYieldOpCreate(
        mlir_ctx,
        loc,
        &[_]mlir.MlirValue{mlir.oraOperationGetResult(seven_op, 0)},
        1,
    ));

    const catch_val_attr = mlir.oraIntegerAttrCreateI64FromType(i256_ty, 9);
    const catch_val_op = mlir.oraArithConstantOpCreate(mlir_ctx, loc, i256_ty, catch_val_attr);
    mlir.oraBlockAppendOwnedOperation(catch_block, catch_val_op);
    mlir.oraBlockAppendOwnedOperation(catch_block, mlir.oraYieldOpCreate(
        mlir_ctx,
        loc,
        &[_]mlir.MlirValue{mlir.oraOperationGetResult(catch_val_op, 0)},
        1,
    ));

    const encoded = try encoder.encodeValue(mlir.oraOperationGetResult(try_stmt, 0));
    try testing.expect(!encoder.isDegraded());

    var solver = try Solver.init(&z3_ctx, testing.allocator);
    defer solver.deinit();
    solver.assert(z3.Z3_mk_not(z3_ctx.ctx, z3.Z3_mk_eq(z3_ctx.ctx, encoded, expected)));
    try testing.expectEqual(@as(z3.Z3_lbool, z3.Z3_L_FALSE), solver.check());
}

test "direct ora.try_stmt ignores catch-capable dead zero-iteration scf.for" {
    var z3_ctx = try Context.init(testing.allocator);
    defer z3_ctx.deinit();

    var encoder = Encoder.init(&z3_ctx, testing.allocator);
    defer encoder.deinit();

    const mlir_ctx = mlir.oraContextCreate();
    defer mlir.oraContextDestroy(mlir_ctx);
    loadAllDialects(mlir_ctx);
    _ = mlir.oraDialectRegister(mlir_ctx);

    const loc = mlir.oraLocationUnknownGet(mlir_ctx);
    const index_ty = mlir.oraIndexTypeCreate(mlir_ctx);
    const i256_ty = mlir.oraIntegerTypeCreate(mlir_ctx, 256);
    const eu_ty = mlir.oraErrorUnionTypeGet(mlir_ctx, i256_ty);

    const try_stmt = mlir.oraTryStmtOpCreate(mlir_ctx, loc, &[_]mlir.MlirType{i256_ty}, 1);
    const try_block = mlir.oraTryStmtOpGetTryBlock(try_stmt);
    const catch_block = mlir.oraTryStmtOpGetCatchBlock(try_stmt);

    const zero_attr = mlir.oraIntegerAttrCreateI64FromType(index_ty, 0);
    const one_attr = mlir.oraIntegerAttrCreateI64FromType(index_ty, 1);
    const zero_op = mlir.oraArithConstantOpCreate(mlir_ctx, loc, index_ty, zero_attr);
    const one_op = mlir.oraArithConstantOpCreate(mlir_ctx, loc, index_ty, one_attr);
    mlir.oraBlockAppendOwnedOperation(try_block, one_op);
    mlir.oraBlockAppendOwnedOperation(try_block, zero_op);

    const for_op = mlir.oraScfForOpCreate(
        mlir_ctx,
        loc,
        mlir.oraOperationGetResult(one_op, 0),
        mlir.oraOperationGetResult(zero_op, 0),
        mlir.oraOperationGetResult(one_op, 0),
        &[_]mlir.MlirValue{},
        0,
        false,
    );
    const body = mlir.oraScfForOpGetBodyBlock(for_op);

    const err_attr = mlir.oraIntegerAttrCreateI64FromType(i256_ty, 1);
    const err_val_op = mlir.oraArithConstantOpCreate(mlir_ctx, loc, i256_ty, err_attr);
    mlir.oraBlockAppendOwnedOperation(body, err_val_op);
    const err_op = mlir.oraErrorErrOpCreate(mlir_ctx, loc, mlir.oraOperationGetResult(err_val_op, 0), eu_ty);
    mlir.oraBlockAppendOwnedOperation(body, err_op);
    const unwrap_op = mlir.oraErrorUnwrapOpCreate(mlir_ctx, loc, mlir.oraOperationGetResult(err_op, 0), i256_ty);
    mlir.oraBlockAppendOwnedOperation(body, unwrap_op);
    mlir.oraBlockAppendOwnedOperation(body, mlir.oraScfYieldOpCreate(mlir_ctx, loc, &[_]mlir.MlirValue{}, 0));

    mlir.oraBlockAppendOwnedOperation(try_block, for_op);

    const seven_attr = mlir.oraIntegerAttrCreateI64FromType(i256_ty, 7);
    const seven_op = mlir.oraArithConstantOpCreate(mlir_ctx, loc, i256_ty, seven_attr);
    const expected = try encoder.encodeOperation(seven_op);
    mlir.oraBlockAppendOwnedOperation(try_block, seven_op);
    mlir.oraBlockAppendOwnedOperation(try_block, mlir.oraYieldOpCreate(
        mlir_ctx,
        loc,
        &[_]mlir.MlirValue{mlir.oraOperationGetResult(seven_op, 0)},
        1,
    ));

    const catch_val_attr = mlir.oraIntegerAttrCreateI64FromType(i256_ty, 9);
    const catch_val_op = mlir.oraArithConstantOpCreate(mlir_ctx, loc, i256_ty, catch_val_attr);
    mlir.oraBlockAppendOwnedOperation(catch_block, catch_val_op);
    mlir.oraBlockAppendOwnedOperation(catch_block, mlir.oraYieldOpCreate(
        mlir_ctx,
        loc,
        &[_]mlir.MlirValue{mlir.oraOperationGetResult(catch_val_op, 0)},
        1,
    ));

    const encoded = try encoder.encodeValue(mlir.oraOperationGetResult(try_stmt, 0));
    try testing.expect(!encoder.isDegraded());

    var solver = try Solver.init(&z3_ctx, testing.allocator);
    defer solver.deinit();
    solver.assert(z3.Z3_mk_not(z3_ctx.ctx, z3.Z3_mk_eq(z3_ctx.ctx, encoded, expected)));
    try testing.expectEqual(@as(z3.Z3_lbool, z3.Z3_L_FALSE), solver.check());
}

test "direct ora.try_stmt ignores catch-capable dead zero-iteration scf.while" {
    var z3_ctx = try Context.init(testing.allocator);
    defer z3_ctx.deinit();

    var encoder = Encoder.init(&z3_ctx, testing.allocator);
    defer encoder.deinit();

    const mlir_ctx = mlir.oraContextCreate();
    defer mlir.oraContextDestroy(mlir_ctx);
    loadAllDialects(mlir_ctx);
    _ = mlir.oraDialectRegister(mlir_ctx);

    const loc = mlir.oraLocationUnknownGet(mlir_ctx);
    const i1_ty = mlir.oraIntegerTypeCreate(mlir_ctx, 1);
    const i256_ty = mlir.oraIntegerTypeCreate(mlir_ctx, 256);
    const eu_ty = mlir.oraErrorUnionTypeGet(mlir_ctx, i256_ty);

    const try_stmt = mlir.oraTryStmtOpCreate(mlir_ctx, loc, &[_]mlir.MlirType{i256_ty}, 1);
    const try_block = mlir.oraTryStmtOpGetTryBlock(try_stmt);
    const catch_block = mlir.oraTryStmtOpGetCatchBlock(try_stmt);

    const false_attr = mlir.oraIntegerAttrCreateI64FromType(i1_ty, 0);
    const false_op = mlir.oraArithConstantOpCreate(mlir_ctx, loc, i1_ty, false_attr);
    mlir.oraBlockAppendOwnedOperation(try_block, false_op);

    const while_op = mlir.oraScfWhileOpCreate(
        mlir_ctx,
        loc,
        &[_]mlir.MlirValue{},
        0,
        &[_]mlir.MlirType{},
        0,
    );
    const before = mlir.oraScfWhileOpGetBeforeBlock(while_op);
    const after = mlir.oraScfWhileOpGetAfterBlock(while_op);

    mlir.oraBlockAppendOwnedOperation(before, mlir.oraScfConditionOpCreate(
        mlir_ctx,
        loc,
        mlir.oraOperationGetResult(false_op, 0),
        &[_]mlir.MlirValue{},
        0,
    ));

    const err_attr = mlir.oraIntegerAttrCreateI64FromType(i256_ty, 1);
    const err_val_op = mlir.oraArithConstantOpCreate(mlir_ctx, loc, i256_ty, err_attr);
    mlir.oraBlockAppendOwnedOperation(after, err_val_op);
    const err_op = mlir.oraErrorErrOpCreate(mlir_ctx, loc, mlir.oraOperationGetResult(err_val_op, 0), eu_ty);
    mlir.oraBlockAppendOwnedOperation(after, err_op);
    const unwrap_op = mlir.oraErrorUnwrapOpCreate(mlir_ctx, loc, mlir.oraOperationGetResult(err_op, 0), i256_ty);
    mlir.oraBlockAppendOwnedOperation(after, unwrap_op);
    mlir.oraBlockAppendOwnedOperation(after, mlir.oraScfYieldOpCreate(mlir_ctx, loc, &[_]mlir.MlirValue{}, 0));

    mlir.oraBlockAppendOwnedOperation(try_block, while_op);

    const seven_attr = mlir.oraIntegerAttrCreateI64FromType(i256_ty, 7);
    const seven_op = mlir.oraArithConstantOpCreate(mlir_ctx, loc, i256_ty, seven_attr);
    const expected = try encoder.encodeOperation(seven_op);
    mlir.oraBlockAppendOwnedOperation(try_block, seven_op);
    mlir.oraBlockAppendOwnedOperation(try_block, mlir.oraYieldOpCreate(
        mlir_ctx,
        loc,
        &[_]mlir.MlirValue{mlir.oraOperationGetResult(seven_op, 0)},
        1,
    ));

    const catch_val_attr = mlir.oraIntegerAttrCreateI64FromType(i256_ty, 9);
    const catch_val_op = mlir.oraArithConstantOpCreate(mlir_ctx, loc, i256_ty, catch_val_attr);
    mlir.oraBlockAppendOwnedOperation(catch_block, catch_val_op);
    mlir.oraBlockAppendOwnedOperation(catch_block, mlir.oraYieldOpCreate(
        mlir_ctx,
        loc,
        &[_]mlir.MlirValue{mlir.oraOperationGetResult(catch_val_op, 0)},
        1,
    ));

    const encoded = try encoder.encodeValue(mlir.oraOperationGetResult(try_stmt, 0));
    try testing.expect(!encoder.isDegraded());

    var solver = try Solver.init(&z3_ctx, testing.allocator);
    defer solver.deinit();
    solver.assert(z3.Z3_mk_not(z3_ctx.ctx, z3.Z3_mk_eq(z3_ctx.ctx, encoded, expected)));
    try testing.expectEqual(@as(z3.Z3_lbool, z3.Z3_L_FALSE), solver.check());
}

test "direct ora.try_stmt catches through single-iteration scf.for" {
    var z3_ctx = try Context.init(testing.allocator);
    defer z3_ctx.deinit();

    var encoder = Encoder.init(&z3_ctx, testing.allocator);
    defer encoder.deinit();

    const mlir_ctx = mlir.oraContextCreate();
    defer mlir.oraContextDestroy(mlir_ctx);
    loadAllDialects(mlir_ctx);
    _ = mlir.oraDialectRegister(mlir_ctx);

    const loc = mlir.oraLocationUnknownGet(mlir_ctx);
    const index_ty = mlir.oraIndexTypeCreate(mlir_ctx);
    const i256_ty = mlir.oraIntegerTypeCreate(mlir_ctx, 256);
    const eu_ty = mlir.oraErrorUnionTypeGet(mlir_ctx, i256_ty);

    const try_stmt = mlir.oraTryStmtOpCreate(mlir_ctx, loc, &[_]mlir.MlirType{i256_ty}, 1);
    const try_block = mlir.oraTryStmtOpGetTryBlock(try_stmt);
    const catch_block = mlir.oraTryStmtOpGetCatchBlock(try_stmt);

    const zero_attr = mlir.oraIntegerAttrCreateI64FromType(index_ty, 0);
    const one_attr = mlir.oraIntegerAttrCreateI64FromType(index_ty, 1);
    const zero_op = mlir.oraArithConstantOpCreate(mlir_ctx, loc, index_ty, zero_attr);
    const one_op = mlir.oraArithConstantOpCreate(mlir_ctx, loc, index_ty, one_attr);
    mlir.oraBlockAppendOwnedOperation(try_block, zero_op);
    mlir.oraBlockAppendOwnedOperation(try_block, one_op);

    const for_op = mlir.oraScfForOpCreate(
        mlir_ctx,
        loc,
        mlir.oraOperationGetResult(zero_op, 0),
        mlir.oraOperationGetResult(one_op, 0),
        mlir.oraOperationGetResult(one_op, 0),
        &[_]mlir.MlirValue{},
        0,
        false,
    );
    const body = mlir.oraScfForOpGetBodyBlock(for_op);

    const err_attr = mlir.oraIntegerAttrCreateI64FromType(i256_ty, 1);
    const err_val_op = mlir.oraArithConstantOpCreate(mlir_ctx, loc, i256_ty, err_attr);
    mlir.oraBlockAppendOwnedOperation(body, err_val_op);
    const err_op = mlir.oraErrorErrOpCreate(mlir_ctx, loc, mlir.oraOperationGetResult(err_val_op, 0), eu_ty);
    mlir.oraBlockAppendOwnedOperation(body, err_op);
    const unwrap_op = mlir.oraErrorUnwrapOpCreate(mlir_ctx, loc, mlir.oraOperationGetResult(err_op, 0), i256_ty);
    mlir.oraBlockAppendOwnedOperation(body, unwrap_op);
    mlir.oraBlockAppendOwnedOperation(body, mlir.oraScfYieldOpCreate(mlir_ctx, loc, &[_]mlir.MlirValue{}, 0));

    mlir.oraBlockAppendOwnedOperation(try_block, for_op);

    const seven_attr = mlir.oraIntegerAttrCreateI64FromType(i256_ty, 7);
    const seven_op = mlir.oraArithConstantOpCreate(mlir_ctx, loc, i256_ty, seven_attr);
    const expected = try encoder.encodeIntegerConstant(9, 256);
    mlir.oraBlockAppendOwnedOperation(try_block, seven_op);
    mlir.oraBlockAppendOwnedOperation(try_block, mlir.oraYieldOpCreate(
        mlir_ctx,
        loc,
        &[_]mlir.MlirValue{mlir.oraOperationGetResult(seven_op, 0)},
        1,
    ));

    const catch_val_attr = mlir.oraIntegerAttrCreateI64FromType(i256_ty, 9);
    const catch_val_op = mlir.oraArithConstantOpCreate(mlir_ctx, loc, i256_ty, catch_val_attr);
    mlir.oraBlockAppendOwnedOperation(catch_block, catch_val_op);
    mlir.oraBlockAppendOwnedOperation(catch_block, mlir.oraYieldOpCreate(
        mlir_ctx,
        loc,
        &[_]mlir.MlirValue{mlir.oraOperationGetResult(catch_val_op, 0)},
        1,
    ));

    const encoded = try encoder.encodeValue(mlir.oraOperationGetResult(try_stmt, 0));
    try testing.expect(!encoder.isDegraded());

    var solver = try Solver.init(&z3_ctx, testing.allocator);
    defer solver.deinit();
    solver.assert(z3.Z3_mk_not(z3_ctx.ctx, z3.Z3_mk_eq(z3_ctx.ctx, encoded, expected)));
    try testing.expectEqual(@as(z3.Z3_lbool, z3.Z3_L_FALSE), solver.check());
}

test "direct ora.try_stmt catches through single-iteration scf.while" {
    var z3_ctx = try Context.init(testing.allocator);
    defer z3_ctx.deinit();

    var encoder = Encoder.init(&z3_ctx, testing.allocator);
    defer encoder.deinit();

    const mlir_ctx = mlir.oraContextCreate();
    defer mlir.oraContextDestroy(mlir_ctx);
    loadAllDialects(mlir_ctx);
    _ = mlir.oraDialectRegister(mlir_ctx);

    const loc = mlir.oraLocationUnknownGet(mlir_ctx);
    const i1_ty = mlir.oraIntegerTypeCreate(mlir_ctx, 1);
    const i256_ty = mlir.oraIntegerTypeCreate(mlir_ctx, 256);
    const eu_ty = mlir.oraErrorUnionTypeGet(mlir_ctx, i256_ty);

    const try_stmt = mlir.oraTryStmtOpCreate(mlir_ctx, loc, &[_]mlir.MlirType{i256_ty}, 1);
    const try_block = mlir.oraTryStmtOpGetTryBlock(try_stmt);
    const catch_block = mlir.oraTryStmtOpGetCatchBlock(try_stmt);

    const true_attr = mlir.oraIntegerAttrCreateI64FromType(i1_ty, 1);
    const false_attr = mlir.oraIntegerAttrCreateI64FromType(i1_ty, 0);
    const true_op = mlir.oraArithConstantOpCreate(mlir_ctx, loc, i1_ty, true_attr);
    const false_op = mlir.oraArithConstantOpCreate(mlir_ctx, loc, i1_ty, false_attr);
    mlir.oraBlockAppendOwnedOperation(try_block, true_op);
    mlir.oraBlockAppendOwnedOperation(try_block, false_op);

    const init_vals = [_]mlir.MlirValue{mlir.oraOperationGetResult(true_op, 0)};
    const result_types = [_]mlir.MlirType{i1_ty};
    const while_op = mlir.oraScfWhileOpCreate(mlir_ctx, loc, &init_vals, init_vals.len, &result_types, result_types.len);
    const before_block = mlir.oraScfWhileOpGetBeforeBlock(while_op);
    const after_block = mlir.oraScfWhileOpGetAfterBlock(while_op);
    _ = mlir.mlirBlockAddArgument(before_block, i1_ty, loc);
    _ = mlir.mlirBlockAddArgument(after_block, i1_ty, loc);
    const before_arg = mlir.oraBlockGetArgument(before_block, 0);
    mlir.oraBlockAppendOwnedOperation(before_block, mlir.oraScfConditionOpCreate(
        mlir_ctx,
        loc,
        before_arg,
        &[_]mlir.MlirValue{before_arg},
        1,
    ));

    const err_attr = mlir.oraIntegerAttrCreateI64FromType(i256_ty, 1);
    const err_val_op = mlir.oraArithConstantOpCreate(mlir_ctx, loc, i256_ty, err_attr);
    mlir.oraBlockAppendOwnedOperation(after_block, err_val_op);
    const err_op = mlir.oraErrorErrOpCreate(mlir_ctx, loc, mlir.oraOperationGetResult(err_val_op, 0), eu_ty);
    mlir.oraBlockAppendOwnedOperation(after_block, err_op);
    const unwrap_op = mlir.oraErrorUnwrapOpCreate(mlir_ctx, loc, mlir.oraOperationGetResult(err_op, 0), i256_ty);
    mlir.oraBlockAppendOwnedOperation(after_block, unwrap_op);
    mlir.oraBlockAppendOwnedOperation(after_block, mlir.oraScfYieldOpCreate(
        mlir_ctx,
        loc,
        &[_]mlir.MlirValue{mlir.oraOperationGetResult(false_op, 0)},
        1,
    ));

    mlir.oraBlockAppendOwnedOperation(try_block, while_op);

    const seven_attr = mlir.oraIntegerAttrCreateI64FromType(i256_ty, 7);
    const seven_op = mlir.oraArithConstantOpCreate(mlir_ctx, loc, i256_ty, seven_attr);
    const expected = try encoder.encodeIntegerConstant(9, 256);
    mlir.oraBlockAppendOwnedOperation(try_block, seven_op);
    mlir.oraBlockAppendOwnedOperation(try_block, mlir.oraYieldOpCreate(
        mlir_ctx,
        loc,
        &[_]mlir.MlirValue{mlir.oraOperationGetResult(seven_op, 0)},
        1,
    ));

    const catch_val_attr = mlir.oraIntegerAttrCreateI64FromType(i256_ty, 9);
    const catch_val_op = mlir.oraArithConstantOpCreate(mlir_ctx, loc, i256_ty, catch_val_attr);
    mlir.oraBlockAppendOwnedOperation(catch_block, catch_val_op);
    mlir.oraBlockAppendOwnedOperation(catch_block, mlir.oraYieldOpCreate(
        mlir_ctx,
        loc,
        &[_]mlir.MlirValue{mlir.oraOperationGetResult(catch_val_op, 0)},
        1,
    ));

    const encoded = try encoder.encodeValue(mlir.oraOperationGetResult(try_stmt, 0));
    try testing.expect(!encoder.isDegraded());

    var solver = try Solver.init(&z3_ctx, testing.allocator);
    defer solver.deinit();
    solver.assert(z3.Z3_mk_not(z3_ctx.ctx, z3.Z3_mk_eq(z3_ctx.ctx, encoded, expected)));
    try testing.expectEqual(@as(z3.Z3_lbool, z3.Z3_L_FALSE), solver.check());
}

test "direct ora.try_stmt catches through finite two-iteration scf.for" {
    var z3_ctx = try Context.init(testing.allocator);
    defer z3_ctx.deinit();

    var encoder = Encoder.init(&z3_ctx, testing.allocator);
    defer encoder.deinit();

    const mlir_ctx = mlir.oraContextCreate();
    defer mlir.oraContextDestroy(mlir_ctx);
    loadAllDialects(mlir_ctx);
    _ = mlir.oraDialectRegister(mlir_ctx);

    const loc = mlir.oraLocationUnknownGet(mlir_ctx);
    const index_ty = mlir.oraIndexTypeCreate(mlir_ctx);
    const i256_ty = mlir.oraIntegerTypeCreate(mlir_ctx, 256);
    const eu_ty = mlir.oraErrorUnionTypeGet(mlir_ctx, i256_ty);

    const try_stmt = mlir.oraTryStmtOpCreate(mlir_ctx, loc, &[_]mlir.MlirType{i256_ty}, 1);
    const try_block = mlir.oraTryStmtOpGetTryBlock(try_stmt);
    const catch_block = mlir.oraTryStmtOpGetCatchBlock(try_stmt);

    const zero_attr = mlir.oraIntegerAttrCreateI64FromType(index_ty, 0);
    const one_attr = mlir.oraIntegerAttrCreateI64FromType(index_ty, 1);
    const two_attr = mlir.oraIntegerAttrCreateI64FromType(index_ty, 2);
    const zero_op = mlir.oraArithConstantOpCreate(mlir_ctx, loc, index_ty, zero_attr);
    const one_op = mlir.oraArithConstantOpCreate(mlir_ctx, loc, index_ty, one_attr);
    const two_op = mlir.oraArithConstantOpCreate(mlir_ctx, loc, index_ty, two_attr);
    mlir.oraBlockAppendOwnedOperation(try_block, zero_op);
    mlir.oraBlockAppendOwnedOperation(try_block, one_op);
    mlir.oraBlockAppendOwnedOperation(try_block, two_op);

    const for_op = mlir.oraScfForOpCreate(
        mlir_ctx,
        loc,
        mlir.oraOperationGetResult(zero_op, 0),
        mlir.oraOperationGetResult(two_op, 0),
        mlir.oraOperationGetResult(one_op, 0),
        &[_]mlir.MlirValue{},
        0,
        false,
    );
    const body = mlir.oraScfForOpGetBodyBlock(for_op);
    const iv = mlir.oraBlockGetArgument(body, 0);
    const cmp_op = mlir.oraArithCmpIOpCreate(mlir_ctx, loc, 0, iv, mlir.oraOperationGetResult(one_op, 0)); // eq
    mlir.oraBlockAppendOwnedOperation(body, cmp_op);
    const no_results = [_]mlir.MlirType{};
    const if_op = mlir.oraScfIfOpCreate(mlir_ctx, loc, mlir.oraOperationGetResult(cmp_op, 0), &no_results, no_results.len, true);
    const then_block = mlir.oraScfIfOpGetThenBlock(if_op);
    const else_block = mlir.oraScfIfOpGetElseBlock(if_op);
    const err_attr = mlir.oraIntegerAttrCreateI64FromType(i256_ty, 1);
    const err_val_op = mlir.oraArithConstantOpCreate(mlir_ctx, loc, i256_ty, err_attr);
    mlir.oraBlockAppendOwnedOperation(then_block, err_val_op);
    const err_op = mlir.oraErrorErrOpCreate(mlir_ctx, loc, mlir.oraOperationGetResult(err_val_op, 0), eu_ty);
    mlir.oraBlockAppendOwnedOperation(then_block, err_op);
    const unwrap_op = mlir.oraErrorUnwrapOpCreate(mlir_ctx, loc, mlir.oraOperationGetResult(err_op, 0), i256_ty);
    mlir.oraBlockAppendOwnedOperation(then_block, unwrap_op);
    mlir.oraBlockAppendOwnedOperation(then_block, mlir.oraScfYieldOpCreate(mlir_ctx, loc, &[_]mlir.MlirValue{}, 0));
    mlir.oraBlockAppendOwnedOperation(else_block, mlir.oraScfYieldOpCreate(mlir_ctx, loc, &[_]mlir.MlirValue{}, 0));
    mlir.oraBlockAppendOwnedOperation(body, if_op);
    mlir.oraBlockAppendOwnedOperation(body, mlir.oraScfYieldOpCreate(mlir_ctx, loc, &[_]mlir.MlirValue{}, 0));

    mlir.oraBlockAppendOwnedOperation(try_block, for_op);

    const seven_attr = mlir.oraIntegerAttrCreateI64FromType(i256_ty, 7);
    const seven_op = mlir.oraArithConstantOpCreate(mlir_ctx, loc, i256_ty, seven_attr);
    const expected = try encoder.encodeIntegerConstant(9, 256);
    mlir.oraBlockAppendOwnedOperation(try_block, seven_op);
    mlir.oraBlockAppendOwnedOperation(try_block, mlir.oraYieldOpCreate(
        mlir_ctx,
        loc,
        &[_]mlir.MlirValue{mlir.oraOperationGetResult(seven_op, 0)},
        1,
    ));

    const catch_val_attr = mlir.oraIntegerAttrCreateI64FromType(i256_ty, 9);
    const catch_val_op = mlir.oraArithConstantOpCreate(mlir_ctx, loc, i256_ty, catch_val_attr);
    mlir.oraBlockAppendOwnedOperation(catch_block, catch_val_op);
    mlir.oraBlockAppendOwnedOperation(catch_block, mlir.oraYieldOpCreate(
        mlir_ctx,
        loc,
        &[_]mlir.MlirValue{mlir.oraOperationGetResult(catch_val_op, 0)},
        1,
    ));

    const encoded = try encoder.encodeValue(mlir.oraOperationGetResult(try_stmt, 0));
    try testing.expect(!encoder.isDegraded());

    var solver = try Solver.init(&z3_ctx, testing.allocator);
    defer solver.deinit();
    solver.assert(z3.Z3_mk_not(z3_ctx.ctx, z3.Z3_mk_eq(z3_ctx.ctx, encoded, expected)));
    try testing.expectEqual(@as(z3.Z3_lbool, z3.Z3_L_FALSE), solver.check());
}

test "direct ora.try_stmt ignores nested caught ora.try_stmt" {
    var z3_ctx = try Context.init(testing.allocator);
    defer z3_ctx.deinit();

    var encoder = Encoder.init(&z3_ctx, testing.allocator);
    defer encoder.deinit();

    const mlir_ctx = mlir.oraContextCreate();
    defer mlir.oraContextDestroy(mlir_ctx);
    loadAllDialects(mlir_ctx);
    _ = mlir.oraDialectRegister(mlir_ctx);

    const loc = mlir.oraLocationUnknownGet(mlir_ctx);
    const i256_ty = mlir.oraIntegerTypeCreate(mlir_ctx, 256);
    const eu_ty = mlir.oraErrorUnionTypeGet(mlir_ctx, i256_ty);

    const outer_try = mlir.oraTryStmtOpCreate(mlir_ctx, loc, &[_]mlir.MlirType{i256_ty}, 1);
    const outer_try_block = mlir.oraTryStmtOpGetTryBlock(outer_try);
    const outer_catch_block = mlir.oraTryStmtOpGetCatchBlock(outer_try);

    const inner_try = mlir.oraTryStmtOpCreate(mlir_ctx, loc, &[_]mlir.MlirType{i256_ty}, 1);
    const inner_try_block = mlir.oraTryStmtOpGetTryBlock(inner_try);
    const inner_catch_block = mlir.oraTryStmtOpGetCatchBlock(inner_try);

    const err_attr = mlir.oraIntegerAttrCreateI64FromType(i256_ty, 1);
    const err_val_op = mlir.oraArithConstantOpCreate(mlir_ctx, loc, i256_ty, err_attr);
    mlir.oraBlockAppendOwnedOperation(inner_try_block, err_val_op);
    const err_op = mlir.oraErrorErrOpCreate(mlir_ctx, loc, mlir.oraOperationGetResult(err_val_op, 0), eu_ty);
    mlir.oraBlockAppendOwnedOperation(inner_try_block, err_op);
    const unwrap_op = mlir.oraErrorUnwrapOpCreate(mlir_ctx, loc, mlir.oraOperationGetResult(err_op, 0), i256_ty);
    mlir.oraBlockAppendOwnedOperation(inner_try_block, unwrap_op);

    const inner_catch_attr = mlir.oraIntegerAttrCreateI64FromType(i256_ty, 5);
    const inner_catch_op = mlir.oraArithConstantOpCreate(mlir_ctx, loc, i256_ty, inner_catch_attr);
    mlir.oraBlockAppendOwnedOperation(inner_catch_block, inner_catch_op);
    mlir.oraBlockAppendOwnedOperation(inner_catch_block, mlir.oraYieldOpCreate(
        mlir_ctx,
        loc,
        &[_]mlir.MlirValue{mlir.oraOperationGetResult(inner_catch_op, 0)},
        1,
    ));

    mlir.oraBlockAppendOwnedOperation(outer_try_block, inner_try);
    const seven_attr = mlir.oraIntegerAttrCreateI64FromType(i256_ty, 7);
    const seven_op = mlir.oraArithConstantOpCreate(mlir_ctx, loc, i256_ty, seven_attr);
    const expected = try encoder.encodeIntegerConstant(7, 256);
    mlir.oraBlockAppendOwnedOperation(outer_try_block, seven_op);
    mlir.oraBlockAppendOwnedOperation(outer_try_block, mlir.oraYieldOpCreate(
        mlir_ctx,
        loc,
        &[_]mlir.MlirValue{mlir.oraOperationGetResult(seven_op, 0)},
        1,
    ));

    const outer_catch_attr = mlir.oraIntegerAttrCreateI64FromType(i256_ty, 9);
    const outer_catch_op = mlir.oraArithConstantOpCreate(mlir_ctx, loc, i256_ty, outer_catch_attr);
    mlir.oraBlockAppendOwnedOperation(outer_catch_block, outer_catch_op);
    mlir.oraBlockAppendOwnedOperation(outer_catch_block, mlir.oraYieldOpCreate(
        mlir_ctx,
        loc,
        &[_]mlir.MlirValue{mlir.oraOperationGetResult(outer_catch_op, 0)},
        1,
    ));

    const encoded = try encoder.encodeValue(mlir.oraOperationGetResult(outer_try, 0));
    try testing.expect(!encoder.isDegraded());

    var solver = try Solver.init(&z3_ctx, testing.allocator);
    defer solver.deinit();
    solver.assert(z3.Z3_mk_not(z3_ctx.ctx, z3.Z3_mk_eq(z3_ctx.ctx, encoded, expected)));
    try testing.expectEqual(@as(z3.Z3_lbool, z3.Z3_L_FALSE), solver.check());
}

test "direct ora.try_stmt yielding nested caught ora.try_stmt result encodes exactly" {
    var z3_ctx = try Context.init(testing.allocator);
    defer z3_ctx.deinit();

    var encoder = Encoder.init(&z3_ctx, testing.allocator);
    defer encoder.deinit();

    const mlir_ctx = mlir.oraContextCreate();
    defer mlir.oraContextDestroy(mlir_ctx);
    loadAllDialects(mlir_ctx);
    _ = mlir.oraDialectRegister(mlir_ctx);

    const loc = mlir.oraLocationUnknownGet(mlir_ctx);
    const i256_ty = mlir.oraIntegerTypeCreate(mlir_ctx, 256);
    const eu_ty = mlir.oraErrorUnionTypeGet(mlir_ctx, i256_ty);

    const outer_try = mlir.oraTryStmtOpCreate(mlir_ctx, loc, &[_]mlir.MlirType{i256_ty}, 1);
    const outer_try_block = mlir.oraTryStmtOpGetTryBlock(outer_try);
    const outer_catch_block = mlir.oraTryStmtOpGetCatchBlock(outer_try);

    const inner_try = mlir.oraTryStmtOpCreate(mlir_ctx, loc, &[_]mlir.MlirType{i256_ty}, 1);
    const inner_try_block = mlir.oraTryStmtOpGetTryBlock(inner_try);
    const inner_catch_block = mlir.oraTryStmtOpGetCatchBlock(inner_try);

    const err_attr = mlir.oraIntegerAttrCreateI64FromType(i256_ty, 1);
    const err_val_op = mlir.oraArithConstantOpCreate(mlir_ctx, loc, i256_ty, err_attr);
    mlir.oraBlockAppendOwnedOperation(inner_try_block, err_val_op);
    const err_op = mlir.oraErrorErrOpCreate(mlir_ctx, loc, mlir.oraOperationGetResult(err_val_op, 0), eu_ty);
    mlir.oraBlockAppendOwnedOperation(inner_try_block, err_op);
    const unwrap_op = mlir.oraErrorUnwrapOpCreate(mlir_ctx, loc, mlir.oraOperationGetResult(err_op, 0), i256_ty);
    mlir.oraBlockAppendOwnedOperation(inner_try_block, unwrap_op);

    const inner_catch_attr = mlir.oraIntegerAttrCreateI64FromType(i256_ty, 5);
    const inner_catch_op = mlir.oraArithConstantOpCreate(mlir_ctx, loc, i256_ty, inner_catch_attr);
    mlir.oraBlockAppendOwnedOperation(inner_catch_block, inner_catch_op);
    mlir.oraBlockAppendOwnedOperation(inner_catch_block, mlir.oraYieldOpCreate(
        mlir_ctx,
        loc,
        &[_]mlir.MlirValue{mlir.oraOperationGetResult(inner_catch_op, 0)},
        1,
    ));

    mlir.oraBlockAppendOwnedOperation(outer_try_block, inner_try);
    mlir.oraBlockAppendOwnedOperation(outer_try_block, mlir.oraYieldOpCreate(
        mlir_ctx,
        loc,
        &[_]mlir.MlirValue{mlir.oraOperationGetResult(inner_try, 0)},
        1,
    ));

    const outer_catch_attr = mlir.oraIntegerAttrCreateI64FromType(i256_ty, 9);
    const outer_catch_op = mlir.oraArithConstantOpCreate(mlir_ctx, loc, i256_ty, outer_catch_attr);
    mlir.oraBlockAppendOwnedOperation(outer_catch_block, outer_catch_op);
    mlir.oraBlockAppendOwnedOperation(outer_catch_block, mlir.oraYieldOpCreate(
        mlir_ctx,
        loc,
        &[_]mlir.MlirValue{mlir.oraOperationGetResult(outer_catch_op, 0)},
        1,
    ));

    const encoded = try encoder.encodeValue(mlir.oraOperationGetResult(outer_try, 0));
    const expected = try encoder.encodeIntegerConstant(5, 256);
    try testing.expect(!encoder.isDegraded());

    var solver = try Solver.init(&z3_ctx, testing.allocator);
    defer solver.deinit();
    solver.assert(z3.Z3_mk_not(z3_ctx.ctx, z3.Z3_mk_eq(z3_ctx.ctx, encoded, expected)));
    try testing.expectEqual(@as(z3.Z3_lbool, z3.Z3_L_FALSE), solver.check());
}

test "direct ora.try_stmt yielding finite scf.for result encodes exactly" {
    var z3_ctx = try Context.init(testing.allocator);
    defer z3_ctx.deinit();

    var encoder = Encoder.init(&z3_ctx, testing.allocator);
    defer encoder.deinit();

    const mlir_ctx = mlir.oraContextCreate();
    defer mlir.oraContextDestroy(mlir_ctx);
    loadAllDialects(mlir_ctx);
    _ = mlir.oraDialectRegister(mlir_ctx);

    const loc = mlir.oraLocationUnknownGet(mlir_ctx);
    const index_ty = mlir.oraIndexTypeCreate(mlir_ctx);
    const i256_ty = mlir.oraIntegerTypeCreate(mlir_ctx, 256);
    const eu_ty = mlir.oraErrorUnionTypeGet(mlir_ctx, i256_ty);

    const outer_try = mlir.oraTryStmtOpCreate(mlir_ctx, loc, &[_]mlir.MlirType{i256_ty}, 1);
    const outer_try_block = mlir.oraTryStmtOpGetTryBlock(outer_try);
    const outer_catch_block = mlir.oraTryStmtOpGetCatchBlock(outer_try);

    const zero_idx_attr = mlir.oraIntegerAttrCreateI64FromType(index_ty, 0);
    const one_idx_attr = mlir.oraIntegerAttrCreateI64FromType(index_ty, 1);
    const two_idx_attr = mlir.oraIntegerAttrCreateI64FromType(index_ty, 2);
    const zero_idx_op = mlir.oraArithConstantOpCreate(mlir_ctx, loc, index_ty, zero_idx_attr);
    const one_idx_op = mlir.oraArithConstantOpCreate(mlir_ctx, loc, index_ty, one_idx_attr);
    const two_idx_op = mlir.oraArithConstantOpCreate(mlir_ctx, loc, index_ty, two_idx_attr);
    mlir.oraBlockAppendOwnedOperation(outer_try_block, zero_idx_op);
    mlir.oraBlockAppendOwnedOperation(outer_try_block, one_idx_op);
    mlir.oraBlockAppendOwnedOperation(outer_try_block, two_idx_op);

    const zero_i256_attr = mlir.oraIntegerAttrCreateI64FromType(i256_ty, 0);
    const one_i256_attr = mlir.oraIntegerAttrCreateI64FromType(i256_ty, 1);
    const zero_i256_op = mlir.oraArithConstantOpCreate(mlir_ctx, loc, i256_ty, zero_i256_attr);
    const one_i256_op = mlir.oraArithConstantOpCreate(mlir_ctx, loc, i256_ty, one_i256_attr);
    mlir.oraBlockAppendOwnedOperation(outer_try_block, zero_i256_op);
    mlir.oraBlockAppendOwnedOperation(outer_try_block, one_i256_op);

    const for_op = mlir.oraScfForOpCreate(
        mlir_ctx,
        loc,
        mlir.oraOperationGetResult(zero_idx_op, 0),
        mlir.oraOperationGetResult(two_idx_op, 0),
        mlir.oraOperationGetResult(one_idx_op, 0),
        &[_]mlir.MlirValue{mlir.oraOperationGetResult(zero_i256_op, 0)},
        1,
        false,
    );
    const body = mlir.oraScfForOpGetBodyBlock(for_op);
    const iv = mlir.oraBlockGetArgument(body, 0);
    const carried = mlir.oraBlockGetArgument(body, 1);
    const cmp_op = mlir.oraArithCmpIOpCreate(mlir_ctx, loc, 0, iv, mlir.oraOperationGetResult(one_idx_op, 0)); // eq
    mlir.oraBlockAppendOwnedOperation(body, cmp_op);
    const no_results = [_]mlir.MlirType{};
    const if_op = mlir.oraScfIfOpCreate(mlir_ctx, loc, mlir.oraOperationGetResult(cmp_op, 0), &no_results, no_results.len, true);
    const then_block = mlir.oraScfIfOpGetThenBlock(if_op);
    const else_block = mlir.oraScfIfOpGetElseBlock(if_op);
    const err_attr = mlir.oraIntegerAttrCreateI64FromType(i256_ty, 1);
    const err_val_op = mlir.oraArithConstantOpCreate(mlir_ctx, loc, i256_ty, err_attr);
    mlir.oraBlockAppendOwnedOperation(then_block, err_val_op);
    const err_op = mlir.oraErrorErrOpCreate(mlir_ctx, loc, mlir.oraOperationGetResult(err_val_op, 0), eu_ty);
    mlir.oraBlockAppendOwnedOperation(then_block, err_op);
    const unwrap_op = mlir.oraErrorUnwrapOpCreate(mlir_ctx, loc, mlir.oraOperationGetResult(err_op, 0), i256_ty);
    mlir.oraBlockAppendOwnedOperation(then_block, unwrap_op);
    mlir.oraBlockAppendOwnedOperation(then_block, mlir.oraScfYieldOpCreate(mlir_ctx, loc, &[_]mlir.MlirValue{}, 0));
    mlir.oraBlockAppendOwnedOperation(else_block, mlir.oraScfYieldOpCreate(mlir_ctx, loc, &[_]mlir.MlirValue{}, 0));
    mlir.oraBlockAppendOwnedOperation(body, if_op);
    const add_op = mlir.oraArithAddIOpCreate(mlir_ctx, loc, carried, mlir.oraOperationGetResult(one_i256_op, 0));
    mlir.oraBlockAppendOwnedOperation(body, add_op);
    mlir.oraBlockAppendOwnedOperation(body, mlir.oraScfYieldOpCreate(
        mlir_ctx,
        loc,
        &[_]mlir.MlirValue{mlir.oraOperationGetResult(add_op, 0)},
        1,
    ));

    mlir.oraBlockAppendOwnedOperation(outer_try_block, for_op);
    mlir.oraBlockAppendOwnedOperation(outer_try_block, mlir.oraYieldOpCreate(
        mlir_ctx,
        loc,
        &[_]mlir.MlirValue{mlir.oraOperationGetResult(for_op, 0)},
        1,
    ));

    const catch_attr = mlir.oraIntegerAttrCreateI64FromType(i256_ty, 9);
    const catch_op = mlir.oraArithConstantOpCreate(mlir_ctx, loc, i256_ty, catch_attr);
    mlir.oraBlockAppendOwnedOperation(outer_catch_block, catch_op);
    mlir.oraBlockAppendOwnedOperation(outer_catch_block, mlir.oraYieldOpCreate(
        mlir_ctx,
        loc,
        &[_]mlir.MlirValue{mlir.oraOperationGetResult(catch_op, 0)},
        1,
    ));

    const encoded = try encoder.encodeValue(mlir.oraOperationGetResult(outer_try, 0));
    const expected = try encoder.encodeIntegerConstant(9, 256);
    try testing.expect(!encoder.isDegraded());

    var solver = try Solver.init(&z3_ctx, testing.allocator);
    defer solver.deinit();
    solver.assert(z3.Z3_mk_not(z3_ctx.ctx, z3.Z3_mk_eq(z3_ctx.ctx, encoded, expected)));
    try testing.expectEqual(@as(z3.Z3_lbool, z3.Z3_L_FALSE), solver.check());
}

test "direct ora.try_stmt composes nested escaping catch predicate exactly" {
    var z3_ctx = try Context.init(testing.allocator);
    defer z3_ctx.deinit();

    var encoder = Encoder.init(&z3_ctx, testing.allocator);
    defer encoder.deinit();

    const mlir_ctx = mlir.oraContextCreate();
    defer mlir.oraContextDestroy(mlir_ctx);
    loadAllDialects(mlir_ctx);
    _ = mlir.oraDialectRegister(mlir_ctx);

    const loc = mlir.oraLocationUnknownGet(mlir_ctx);
    const i1_ty = mlir.oraIntegerTypeCreate(mlir_ctx, 1);
    const i256_ty = mlir.oraIntegerTypeCreate(mlir_ctx, 256);
    const eu_ty = mlir.oraErrorUnionTypeGet(mlir_ctx, i256_ty);

    const cond_op = mlir.oraVariablePlaceholderOpCreate(mlir_ctx, loc, stringRef("condValue"), i1_ty);

    const outer_try = mlir.oraTryStmtOpCreate(mlir_ctx, loc, &[_]mlir.MlirType{i256_ty}, 1);
    const outer_try_block = mlir.oraTryStmtOpGetTryBlock(outer_try);
    const outer_catch_block = mlir.oraTryStmtOpGetCatchBlock(outer_try);

    const inner_try = mlir.oraTryStmtOpCreate(mlir_ctx, loc, &[_]mlir.MlirType{i256_ty}, 1);
    const inner_try_block = mlir.oraTryStmtOpGetTryBlock(inner_try);
    const inner_catch_block = mlir.oraTryStmtOpGetCatchBlock(inner_try);

    const result_types = [_]mlir.MlirType{i256_ty};
    const if_op = mlir.oraScfIfOpCreate(
        mlir_ctx,
        loc,
        mlir.oraOperationGetResult(cond_op, 0),
        &result_types,
        result_types.len,
        true,
    );
    const then_block = mlir.oraScfIfOpGetThenBlock(if_op);
    const else_block = mlir.oraScfIfOpGetElseBlock(if_op);

    const err_attr = mlir.oraIntegerAttrCreateI64FromType(i256_ty, 1);
    const err_val_op = mlir.oraArithConstantOpCreate(mlir_ctx, loc, i256_ty, err_attr);
    mlir.oraBlockAppendOwnedOperation(then_block, err_val_op);
    const err_op = mlir.oraErrorErrOpCreate(mlir_ctx, loc, mlir.oraOperationGetResult(err_val_op, 0), eu_ty);
    mlir.oraBlockAppendOwnedOperation(then_block, err_op);
    const unwrap_op = mlir.oraErrorUnwrapOpCreate(mlir_ctx, loc, mlir.oraOperationGetResult(err_op, 0), i256_ty);
    mlir.oraBlockAppendOwnedOperation(then_block, unwrap_op);
    mlir.oraBlockAppendOwnedOperation(then_block, mlir.oraScfYieldOpCreate(
        mlir_ctx,
        loc,
        &[_]mlir.MlirValue{mlir.oraOperationGetResult(unwrap_op, 0)},
        1,
    ));

    const else_attr = mlir.oraIntegerAttrCreateI64FromType(i256_ty, 5);
    const else_op = mlir.oraArithConstantOpCreate(mlir_ctx, loc, i256_ty, else_attr);
    mlir.oraBlockAppendOwnedOperation(else_block, else_op);
    mlir.oraBlockAppendOwnedOperation(else_block, mlir.oraScfYieldOpCreate(
        mlir_ctx,
        loc,
        &[_]mlir.MlirValue{mlir.oraOperationGetResult(else_op, 0)},
        1,
    ));

    mlir.oraBlockAppendOwnedOperation(inner_try_block, if_op);
    mlir.oraBlockAppendOwnedOperation(inner_try_block, mlir.oraYieldOpCreate(
        mlir_ctx,
        loc,
        &[_]mlir.MlirValue{mlir.oraOperationGetResult(if_op, 0)},
        1,
    ));

    const catch_err_attr = mlir.oraIntegerAttrCreateI64FromType(i256_ty, 2);
    const catch_err_val_op = mlir.oraArithConstantOpCreate(mlir_ctx, loc, i256_ty, catch_err_attr);
    mlir.oraBlockAppendOwnedOperation(inner_catch_block, catch_err_val_op);
    const catch_err_op = mlir.oraErrorErrOpCreate(mlir_ctx, loc, mlir.oraOperationGetResult(catch_err_val_op, 0), eu_ty);
    mlir.oraBlockAppendOwnedOperation(inner_catch_block, catch_err_op);
    const catch_unwrap_op = mlir.oraErrorUnwrapOpCreate(mlir_ctx, loc, mlir.oraOperationGetResult(catch_err_op, 0), i256_ty);
    mlir.oraBlockAppendOwnedOperation(inner_catch_block, catch_unwrap_op);

    mlir.oraBlockAppendOwnedOperation(outer_try_block, inner_try);
    const seven_attr = mlir.oraIntegerAttrCreateI64FromType(i256_ty, 7);
    const seven_op = mlir.oraArithConstantOpCreate(mlir_ctx, loc, i256_ty, seven_attr);
    mlir.oraBlockAppendOwnedOperation(outer_try_block, seven_op);
    mlir.oraBlockAppendOwnedOperation(outer_try_block, mlir.oraYieldOpCreate(
        mlir_ctx,
        loc,
        &[_]mlir.MlirValue{mlir.oraOperationGetResult(seven_op, 0)},
        1,
    ));

    const outer_catch_attr = mlir.oraIntegerAttrCreateI64FromType(i256_ty, 9);
    const outer_catch_op = mlir.oraArithConstantOpCreate(mlir_ctx, loc, i256_ty, outer_catch_attr);
    mlir.oraBlockAppendOwnedOperation(outer_catch_block, outer_catch_op);
    mlir.oraBlockAppendOwnedOperation(outer_catch_block, mlir.oraYieldOpCreate(
        mlir_ctx,
        loc,
        &[_]mlir.MlirValue{mlir.oraOperationGetResult(outer_catch_op, 0)},
        1,
    ));

    const encoded = try encoder.encodeValue(mlir.oraOperationGetResult(outer_try, 0));
    try testing.expect(!encoder.isDegraded());
    const cond = try encoder.encodeOperation(cond_op);
    const seven = try encoder.encodeIntegerConstant(7, 256);
    const nine = try encoder.encodeIntegerConstant(9, 256);

    var solver = try Solver.init(&z3_ctx, testing.allocator);
    defer solver.deinit();

    solver.push();
    defer solver.pop();
    solver.assert(cond);
    solver.assert(z3.Z3_mk_not(z3_ctx.ctx, z3.Z3_mk_eq(z3_ctx.ctx, encoded, nine)));
    try testing.expectEqual(@as(z3.Z3_lbool, z3.Z3_L_FALSE), solver.check());

    solver.push();
    defer solver.pop();
    solver.assert(z3.Z3_mk_not(z3_ctx.ctx, cond));
    solver.assert(z3.Z3_mk_not(z3_ctx.ctx, z3.Z3_mk_eq(z3_ctx.ctx, encoded, seven)));
    try testing.expectEqual(@as(z3.Z3_lbool, z3.Z3_L_FALSE), solver.check());
}

test "direct ora.switch_expr with default encodes exactly" {
    var z3_ctx = try Context.init(testing.allocator);
    defer z3_ctx.deinit();

    var encoder = Encoder.init(&z3_ctx, testing.allocator);
    defer encoder.deinit();

    const mlir_ctx = mlir.oraContextCreate();
    defer mlir.oraContextDestroy(mlir_ctx);
    loadAllDialects(mlir_ctx);
    _ = mlir.oraDialectRegister(mlir_ctx);

    const loc = mlir.oraLocationUnknownGet(mlir_ctx);
    const i256_ty = mlir.oraIntegerTypeCreate(mlir_ctx, 256);

    const scrutinee_attr = mlir.oraIntegerAttrCreateI64FromType(i256_ty, 5);
    const scrutinee_op = mlir.oraArithConstantOpCreate(mlir_ctx, loc, i256_ty, scrutinee_attr);
    const scrutinee = mlir.oraOperationGetResult(scrutinee_op, 0);

    const switch_expr = mlir.oraSwitchExprOpCreateWithCases(
        mlir_ctx,
        loc,
        scrutinee,
        &[_]mlir.MlirType{i256_ty},
        1,
        2,
    );
    const case_values = [_]i64{5};
    const range_starts = [_]i64{0};
    const range_ends = [_]i64{0};
    const case_kinds = [_]i64{0};
    mlir.oraSwitchOpSetCasePatterns(
        switch_expr,
        &case_values,
        &range_starts,
        &range_ends,
        &case_kinds,
        1,
        case_values.len,
    );

    const case_block = mlir.oraSwitchExprOpGetCaseBlock(switch_expr, 0);
    const default_block = mlir.oraSwitchExprOpGetCaseBlock(switch_expr, 1);

    const case_attr = mlir.oraIntegerAttrCreateI64FromType(i256_ty, 7);
    const case_op = mlir.oraArithConstantOpCreate(mlir_ctx, loc, i256_ty, case_attr);
    const case_expected = try encoder.encodeOperation(case_op);
    mlir.oraBlockAppendOwnedOperation(case_block, case_op);
    mlir.oraBlockAppendOwnedOperation(case_block, mlir.oraYieldOpCreate(
        mlir_ctx,
        loc,
        &[_]mlir.MlirValue{mlir.oraOperationGetResult(case_op, 0)},
        1,
    ));

    const default_attr = mlir.oraIntegerAttrCreateI64FromType(i256_ty, 99);
    const default_op = mlir.oraArithConstantOpCreate(mlir_ctx, loc, i256_ty, default_attr);
    mlir.oraBlockAppendOwnedOperation(default_block, default_op);
    mlir.oraBlockAppendOwnedOperation(default_block, mlir.oraYieldOpCreate(
        mlir_ctx,
        loc,
        &[_]mlir.MlirValue{mlir.oraOperationGetResult(default_op, 0)},
        1,
    ));

    const encoded = try encoder.encodeValue(mlir.oraOperationGetResult(switch_expr, 0));
    try testing.expect(!encoder.isDegraded());

    var solver = try Solver.init(&z3_ctx, testing.allocator);
    defer solver.deinit();
    solver.assert(z3.Z3_mk_not(z3_ctx.ctx, z3.Z3_mk_eq(z3_ctx.ctx, encoded, case_expected)));
    try testing.expectEqual(@as(z3.Z3_lbool, z3.Z3_L_FALSE), solver.check());
}

test "direct exhaustive ora.switch_expr without default encodes exactly" {
    var z3_ctx = try Context.init(testing.allocator);
    defer z3_ctx.deinit();

    var encoder = Encoder.init(&z3_ctx, testing.allocator);
    defer encoder.deinit();

    const mlir_ctx = mlir.oraContextCreate();
    defer mlir.oraContextDestroy(mlir_ctx);
    loadAllDialects(mlir_ctx);
    _ = mlir.oraDialectRegister(mlir_ctx);

    const loc = mlir.oraLocationUnknownGet(mlir_ctx);
    const i1_ty = mlir.oraIntegerTypeCreate(mlir_ctx, 1);
    const i256_ty = mlir.oraIntegerTypeCreate(mlir_ctx, 256);

    const scrutinee_attr = mlir.oraIntegerAttrCreateI64FromType(i1_ty, 1);
    const scrutinee_op = mlir.oraArithConstantOpCreate(mlir_ctx, loc, i1_ty, scrutinee_attr);
    const scrutinee = mlir.oraOperationGetResult(scrutinee_op, 0);

    const switch_expr = mlir.oraSwitchExprOpCreateWithCases(
        mlir_ctx,
        loc,
        scrutinee,
        &[_]mlir.MlirType{i256_ty},
        1,
        2,
    );
    const case_values = [_]i64{ 0, 1 };
    const range_starts = [_]i64{ 0, 0 };
    const range_ends = [_]i64{ 0, 0 };
    const case_kinds = [_]i64{ 0, 0 };
    mlir.oraSwitchOpSetCasePatterns(
        switch_expr,
        &case_values,
        &range_starts,
        &range_ends,
        &case_kinds,
        2,
        case_values.len,
    );

    const false_block = mlir.oraSwitchExprOpGetCaseBlock(switch_expr, 0);
    const true_block = mlir.oraSwitchExprOpGetCaseBlock(switch_expr, 1);

    const false_attr = mlir.oraIntegerAttrCreateI64FromType(i256_ty, 7);
    const false_op = mlir.oraArithConstantOpCreate(mlir_ctx, loc, i256_ty, false_attr);
    mlir.oraBlockAppendOwnedOperation(false_block, false_op);
    mlir.oraBlockAppendOwnedOperation(false_block, mlir.oraYieldOpCreate(
        mlir_ctx,
        loc,
        &[_]mlir.MlirValue{mlir.oraOperationGetResult(false_op, 0)},
        1,
    ));

    const true_attr = mlir.oraIntegerAttrCreateI64FromType(i256_ty, 9);
    const true_op = mlir.oraArithConstantOpCreate(mlir_ctx, loc, i256_ty, true_attr);
    const expected = try encoder.encodeOperation(true_op);
    mlir.oraBlockAppendOwnedOperation(true_block, true_op);
    mlir.oraBlockAppendOwnedOperation(true_block, mlir.oraYieldOpCreate(
        mlir_ctx,
        loc,
        &[_]mlir.MlirValue{mlir.oraOperationGetResult(true_op, 0)},
        1,
    ));

    const encoded = try encoder.encodeValue(mlir.oraOperationGetResult(switch_expr, 0));
    try testing.expect(!encoder.isDegraded());

    var solver = try Solver.init(&z3_ctx, testing.allocator);
    defer solver.deinit();
    solver.assert(z3.Z3_mk_not(z3_ctx.ctx, z3.Z3_mk_eq(z3_ctx.ctx, encoded, expected)));
    try testing.expectEqual(@as(z3.Z3_lbool, z3.Z3_L_FALSE), solver.check());
}

test "func.call summary with scf.execute_region state effects encodes exactly" {
    var z3_ctx = try Context.init(testing.allocator);
    defer z3_ctx.deinit();

    var encoder = Encoder.init(&z3_ctx, testing.allocator);
    defer encoder.deinit();

    const mlir_ctx = mlir.oraContextCreate();
    defer mlir.oraContextDestroy(mlir_ctx);
    loadAllDialects(mlir_ctx);
    _ = mlir.oraDialectRegister(mlir_ctx);

    const loc = mlir.oraLocationUnknownGet(mlir_ctx);
    const i256_ty = mlir.oraIntegerTypeCreate(mlir_ctx, 256);

    const helper_attrs = [_]mlir.MlirNamedAttribute{
        namedAttr(mlir_ctx, "sym_name", mlir.oraStringAttrCreate(mlir_ctx, stringRef("executeRegionWriter"))),
        namedAttr(mlir_ctx, "ora.effect", mlir.oraStringAttrCreate(mlir_ctx, stringRef("writes"))),
        namedAttr(mlir_ctx, "ora.write_slots", mlir.oraArrayAttrCreate(mlir_ctx, 1, &[_]mlir.MlirAttribute{
            mlir.oraStringAttrCreate(mlir_ctx, stringRef("counter")),
        })),
    };
    const helper = mlir.oraFuncFuncOpCreate(mlir_ctx, loc, &helper_attrs, helper_attrs.len, &[_]mlir.MlirType{}, &[_]mlir.MlirLocation{}, 0);
    const body = mlir.oraFuncOpGetBodyBlock(helper);

    const exec = mlir.oraScfExecuteRegionOpCreate(mlir_ctx, loc, &[_]mlir.MlirType{}, 0, false);
    const exec_block = mlir.oraScfExecuteRegionOpGetBodyBlock(exec);
    const store_val_attr = mlir.oraIntegerAttrCreateI64FromType(i256_ty, 7);
    const store_val_op = mlir.oraArithConstantOpCreate(mlir_ctx, loc, i256_ty, store_val_attr);
    mlir.oraBlockAppendOwnedOperation(exec_block, store_val_op);
    mlir.oraBlockAppendOwnedOperation(exec_block, mlir.oraSStoreOpCreate(mlir_ctx, loc, mlir.oraOperationGetResult(store_val_op, 0), stringRef("counter")));
    mlir.oraBlockAppendOwnedOperation(exec_block, mlir.oraScfYieldOpCreate(mlir_ctx, loc, &[_]mlir.MlirValue{}, 0));

    mlir.oraBlockAppendOwnedOperation(body, exec);
    mlir.oraBlockAppendOwnedOperation(body, mlir.oraReturnOpCreate(mlir_ctx, loc, &[_]mlir.MlirValue{}, 0));

    try encoder.registerFunctionOperation(helper);

    const call = mlir.oraFuncCallOpCreate(mlir_ctx, loc, stringRef("executeRegionWriter"), &[_]mlir.MlirValue{}, 0, &[_]mlir.MlirType{}, 0);
    _ = try encoder.encodeOperation(call);

    try testing.expect(!encoder.isDegraded());
    const counter = encoder.global_map.get("counter").?;
    const expected = z3.Z3_mk_numeral(z3_ctx.ctx, "7", z3.Z3_mk_bv_sort(z3_ctx.ctx, 256));

    var solver = try Solver.init(&z3_ctx, testing.allocator);
    defer solver.deinit();
    solver.assert(z3.Z3_mk_not(z3_ctx.ctx, z3.Z3_mk_eq(z3_ctx.ctx, counter, expected)));
    try testing.expectEqual(@as(z3.Z3_lbool, z3.Z3_L_FALSE), solver.check());
}

test "known pure callee execute_region return encodes exactly" {
    var z3_ctx = try Context.init(testing.allocator);
    defer z3_ctx.deinit();

    var encoder = Encoder.init(&z3_ctx, testing.allocator);
    defer encoder.deinit();

    const mlir_ctx = mlir.oraContextCreate();
    defer mlir.oraContextDestroy(mlir_ctx);
    loadAllDialects(mlir_ctx);
    _ = mlir.oraDialectRegister(mlir_ctx);

    const loc = mlir.oraLocationUnknownGet(mlir_ctx);
    const i256_ty = mlir.oraIntegerTypeCreate(mlir_ctx, 256);

    const helper_attrs = [_]mlir.MlirNamedAttribute{
        namedAttr(mlir_ctx, "sym_name", mlir.oraStringAttrCreate(mlir_ctx, stringRef("executeRegionReturn"))),
    };
    const helper = mlir.oraFuncFuncOpCreate(mlir_ctx, loc, &helper_attrs, helper_attrs.len, &[_]mlir.MlirType{}, &[_]mlir.MlirLocation{}, 0);
    const body = mlir.oraFuncOpGetBodyBlock(helper);

    const exec = mlir.oraScfExecuteRegionOpCreate(mlir_ctx, loc, &[_]mlir.MlirType{}, 0, false);
    const exec_block = mlir.oraScfExecuteRegionOpGetBodyBlock(exec);
    const value_attr = mlir.oraIntegerAttrCreateI64FromType(i256_ty, 11);
    const value_op = mlir.oraArithConstantOpCreate(mlir_ctx, loc, i256_ty, value_attr);
    mlir.oraBlockAppendOwnedOperation(exec_block, value_op);
    mlir.oraBlockAppendOwnedOperation(exec_block, mlir.oraReturnOpCreate(
        mlir_ctx,
        loc,
        &[_]mlir.MlirValue{mlir.oraOperationGetResult(value_op, 0)},
        1,
    ));

    mlir.oraBlockAppendOwnedOperation(body, exec);

    try encoder.registerFunctionOperation(helper);

    const result_types = [_]mlir.MlirType{i256_ty};
    const call = mlir.oraFuncCallOpCreate(
        mlir_ctx,
        loc,
        stringRef("executeRegionReturn"),
        &[_]mlir.MlirValue{},
        0,
        &result_types,
        result_types.len,
    );

    const encoded = try encoder.encodeOperation(call);
    const expected = try encoder.encodeIntegerConstant(11, 256);
    try testing.expect(!encoder.isDegraded());

    var solver = try Solver.init(&z3_ctx, testing.allocator);
    defer solver.deinit();
    solver.assert(z3.Z3_mk_not(z3_ctx.ctx, z3.Z3_mk_eq(z3_ctx.ctx, encoded, expected)));
    try testing.expectEqual(@as(z3.Z3_lbool, z3.Z3_L_FALSE), solver.check());
}

test "direct scf.execute_region result encodes exactly" {
    var z3_ctx = try Context.init(testing.allocator);
    defer z3_ctx.deinit();

    var encoder = Encoder.init(&z3_ctx, testing.allocator);
    defer encoder.deinit();

    const mlir_ctx = mlir.oraContextCreate();
    defer mlir.oraContextDestroy(mlir_ctx);
    loadAllDialects(mlir_ctx);
    _ = mlir.oraDialectRegister(mlir_ctx);

    const loc = mlir.oraLocationUnknownGet(mlir_ctx);
    const i256_ty = mlir.oraIntegerTypeCreate(mlir_ctx, 256);

    const value_attr = mlir.oraIntegerAttrCreateI64FromType(i256_ty, 11);
    const value_op = mlir.oraArithConstantOpCreate(mlir_ctx, loc, i256_ty, value_attr);
    const value = mlir.oraOperationGetResult(value_op, 0);
    const expected = try encoder.encodeOperation(value_op);

    const exec = mlir.oraScfExecuteRegionOpCreate(mlir_ctx, loc, &[_]mlir.MlirType{i256_ty}, 1, false);
    const exec_block = mlir.oraScfExecuteRegionOpGetBodyBlock(exec);
    mlir.oraBlockAppendOwnedOperation(exec_block, value_op);
    mlir.oraBlockAppendOwnedOperation(exec_block, mlir.oraScfYieldOpCreate(
        mlir_ctx,
        loc,
        &[_]mlir.MlirValue{value},
        1,
    ));

    const encoded = try encoder.encodeOperation(exec);
    try testing.expect(!encoder.isDegraded());

    var solver = try Solver.init(&z3_ctx, testing.allocator);
    defer solver.deinit();
    solver.assert(z3.Z3_mk_not(z3_ctx.ctx, z3.Z3_mk_eq(z3_ctx.ctx, encoded, expected)));
    try testing.expectEqual(@as(z3.Z3_lbool, z3.Z3_L_FALSE), solver.check());
}

test "known pure callee scf.execute_region does not preempt later explicit return" {
    var z3_ctx = try Context.init(testing.allocator);
    defer z3_ctx.deinit();

    var encoder = Encoder.init(&z3_ctx, testing.allocator);
    defer encoder.deinit();

    const mlir_ctx = mlir.oraContextCreate();
    defer mlir.oraContextDestroy(mlir_ctx);
    loadAllDialects(mlir_ctx);
    _ = mlir.oraDialectRegister(mlir_ctx);

    const loc = mlir.oraLocationUnknownGet(mlir_ctx);
    const i256_ty = mlir.oraIntegerTypeCreate(mlir_ctx, 256);

    const helper_attrs = [_]mlir.MlirNamedAttribute{
        namedAttr(mlir_ctx, "sym_name", mlir.oraStringAttrCreate(mlir_ctx, stringRef("executeRegionThenReturn"))),
    };
    const helper = mlir.oraFuncFuncOpCreate(mlir_ctx, loc, &helper_attrs, helper_attrs.len, &[_]mlir.MlirType{}, &[_]mlir.MlirLocation{}, 0);
    const body = mlir.oraFuncOpGetBodyBlock(helper);

    const exec_attr = mlir.oraIntegerAttrCreateI64FromType(i256_ty, 11);
    const exec_value_op = mlir.oraArithConstantOpCreate(mlir_ctx, loc, i256_ty, exec_attr);
    const exec = mlir.oraScfExecuteRegionOpCreate(mlir_ctx, loc, &[_]mlir.MlirType{i256_ty}, 1, false);
    const exec_block = mlir.oraScfExecuteRegionOpGetBodyBlock(exec);
    mlir.oraBlockAppendOwnedOperation(exec_block, exec_value_op);
    mlir.oraBlockAppendOwnedOperation(exec_block, mlir.oraScfYieldOpCreate(
        mlir_ctx,
        loc,
        &[_]mlir.MlirValue{mlir.oraOperationGetResult(exec_value_op, 0)},
        1,
    ));
    mlir.oraBlockAppendOwnedOperation(body, exec);

    const ret_attr = mlir.oraIntegerAttrCreateI64FromType(i256_ty, 42);
    const ret_value_op = mlir.oraArithConstantOpCreate(mlir_ctx, loc, i256_ty, ret_attr);
    mlir.oraBlockAppendOwnedOperation(body, ret_value_op);
    mlir.oraBlockAppendOwnedOperation(body, mlir.oraReturnOpCreate(
        mlir_ctx,
        loc,
        &[_]mlir.MlirValue{mlir.oraOperationGetResult(ret_value_op, 0)},
        1,
    ));

    try encoder.registerFunctionOperation(helper);

    const call = mlir.oraFuncCallOpCreate(
        mlir_ctx,
        loc,
        stringRef("executeRegionThenReturn"),
        &[_]mlir.MlirValue{},
        0,
        &[_]mlir.MlirType{i256_ty},
        1,
    );
    const encoded = try encoder.encodeOperation(call);
    const expected = try encoder.encodeOperation(ret_value_op);
    try testing.expect(!encoder.isDegraded());

    var solver = try Solver.init(&z3_ctx, testing.allocator);
    defer solver.deinit();
    solver.assert(z3.Z3_mk_not(z3_ctx.ctx, z3.Z3_mk_eq(z3_ctx.ctx, encoded, expected)));
    try testing.expectEqual(@as(z3.Z3_lbool, z3.Z3_L_FALSE), solver.check());
}

test "known pure callee single-iteration scf.for return encodes exactly" {
    var z3_ctx = try Context.init(testing.allocator);
    defer z3_ctx.deinit();

    var encoder = Encoder.init(&z3_ctx, testing.allocator);
    defer encoder.deinit();

    const mlir_ctx = mlir.oraContextCreate();
    defer mlir.oraContextDestroy(mlir_ctx);
    loadAllDialects(mlir_ctx);
    _ = mlir.oraDialectRegister(mlir_ctx);

    const loc = mlir.oraLocationUnknownGet(mlir_ctx);
    const index_ty = mlir.oraIndexTypeCreate(mlir_ctx);
    const i256_ty = mlir.oraIntegerTypeCreate(mlir_ctx, 256);

    const helper_attrs = [_]mlir.MlirNamedAttribute{
        namedAttr(mlir_ctx, "sym_name", mlir.oraStringAttrCreate(mlir_ctx, stringRef("singleIterForReturn"))),
    };
    const helper = mlir.oraFuncFuncOpCreate(mlir_ctx, loc, &helper_attrs, helper_attrs.len, &[_]mlir.MlirType{}, &[_]mlir.MlirLocation{}, 0);
    const body = mlir.oraFuncOpGetBodyBlock(helper);

    const c0_attr = mlir.oraIntegerAttrCreateI64FromType(index_ty, 0);
    const c1_attr = mlir.oraIntegerAttrCreateI64FromType(index_ty, 1);
    const c0_op = mlir.oraArithConstantOpCreate(mlir_ctx, loc, index_ty, c0_attr);
    const c1_op = mlir.oraArithConstantOpCreate(mlir_ctx, loc, index_ty, c1_attr);
    mlir.oraBlockAppendOwnedOperation(body, c0_op);
    mlir.oraBlockAppendOwnedOperation(body, c1_op);

    const loop = mlir.oraScfForOpCreate(
        mlir_ctx,
        loc,
        mlir.oraOperationGetResult(c0_op, 0),
        mlir.oraOperationGetResult(c1_op, 0),
        mlir.oraOperationGetResult(c1_op, 0),
        &[_]mlir.MlirValue{},
        0,
        false,
    );
    const loop_body = mlir.oraScfForOpGetBodyBlock(loop);
    const val_attr = mlir.oraIntegerAttrCreateI64FromType(i256_ty, 13);
    const val_op = mlir.oraArithConstantOpCreate(mlir_ctx, loc, i256_ty, val_attr);
    mlir.oraBlockAppendOwnedOperation(loop_body, val_op);
    mlir.oraBlockAppendOwnedOperation(loop_body, mlir.oraReturnOpCreate(
        mlir_ctx,
        loc,
        &[_]mlir.MlirValue{mlir.oraOperationGetResult(val_op, 0)},
        1,
    ));
    mlir.oraBlockAppendOwnedOperation(body, loop);

    try encoder.registerFunctionOperation(helper);

    const result_types = [_]mlir.MlirType{i256_ty};
    const call = mlir.oraFuncCallOpCreate(
        mlir_ctx,
        loc,
        stringRef("singleIterForReturn"),
        &[_]mlir.MlirValue{},
        0,
        &result_types,
        result_types.len,
    );

    const encoded = try encoder.encodeOperation(call);
    const expected = try encoder.encodeIntegerConstant(13, 256);
    try testing.expect(!encoder.isDegraded());

    var solver = try Solver.init(&z3_ctx, testing.allocator);
    defer solver.deinit();
    solver.assert(z3.Z3_mk_not(z3_ctx.ctx, z3.Z3_mk_eq(z3_ctx.ctx, encoded, expected)));
    try testing.expectEqual(@as(z3.Z3_lbool, z3.Z3_L_FALSE), solver.check());
}

test "known pure callee finite two-iteration scf.for return encodes exactly" {
    var z3_ctx = try Context.init(testing.allocator);
    defer z3_ctx.deinit();

    var encoder = Encoder.init(&z3_ctx, testing.allocator);
    defer encoder.deinit();

    const mlir_ctx = mlir.oraContextCreate();
    defer mlir.oraContextDestroy(mlir_ctx);
    loadAllDialects(mlir_ctx);
    _ = mlir.oraDialectRegister(mlir_ctx);

    const loc = mlir.oraLocationUnknownGet(mlir_ctx);
    const index_ty = mlir.oraIndexTypeCreate(mlir_ctx);
    const i256_ty = mlir.oraIntegerTypeCreate(mlir_ctx, 256);

    const helper_attrs = [_]mlir.MlirNamedAttribute{
        namedAttr(mlir_ctx, "sym_name", mlir.oraStringAttrCreate(mlir_ctx, stringRef("finiteForReturn"))),
    };
    const helper = mlir.oraFuncFuncOpCreate(mlir_ctx, loc, &helper_attrs, helper_attrs.len, &[_]mlir.MlirType{i256_ty}, &[_]mlir.MlirLocation{loc}, 1);
    const body = mlir.oraFuncOpGetBodyBlock(helper);

    const c0_attr = mlir.oraIntegerAttrCreateI64FromType(index_ty, 0);
    const c2_attr = mlir.oraIntegerAttrCreateI64FromType(index_ty, 2);
    const c1_attr = mlir.oraIntegerAttrCreateI64FromType(index_ty, 1);
    const c0_op = mlir.oraArithConstantOpCreate(mlir_ctx, loc, index_ty, c0_attr);
    const c2_op = mlir.oraArithConstantOpCreate(mlir_ctx, loc, index_ty, c2_attr);
    const c1_op = mlir.oraArithConstantOpCreate(mlir_ctx, loc, index_ty, c1_attr);
    mlir.oraBlockAppendOwnedOperation(body, c0_op);
    mlir.oraBlockAppendOwnedOperation(body, c2_op);
    mlir.oraBlockAppendOwnedOperation(body, c1_op);

    const init_attr = mlir.oraIntegerAttrCreateI64FromType(i256_ty, 0);
    const init_op = mlir.oraArithConstantOpCreate(mlir_ctx, loc, i256_ty, init_attr);
    mlir.oraBlockAppendOwnedOperation(body, init_op);

    const loop = mlir.oraScfForOpCreate(
        mlir_ctx,
        loc,
        mlir.oraOperationGetResult(c0_op, 0),
        mlir.oraOperationGetResult(c2_op, 0),
        mlir.oraOperationGetResult(c1_op, 0),
        &[_]mlir.MlirValue{mlir.oraOperationGetResult(init_op, 0)},
        1,
        false,
    );
    const loop_body = mlir.oraScfForOpGetBodyBlock(loop);
    const iv = mlir.oraBlockGetArgument(loop_body, 0);
    const carried = mlir.oraBlockGetArgument(loop_body, 1);
    const iv_i256 = mlir.oraArithIndexCastUIOpCreate(mlir_ctx, loc, iv, i256_ty);
    mlir.oraBlockAppendOwnedOperation(loop_body, iv_i256);
    const next = mlir.oraArithAddIOpCreate(mlir_ctx, loc, carried, mlir.oraOperationGetResult(iv_i256, 0));
    mlir.oraBlockAppendOwnedOperation(loop_body, next);
    mlir.oraBlockAppendOwnedOperation(loop_body, mlir.oraScfYieldOpCreate(
        mlir_ctx,
        loc,
        &[_]mlir.MlirValue{mlir.oraOperationGetResult(next, 0)},
        1,
    ));
    mlir.oraBlockAppendOwnedOperation(body, loop);
    mlir.oraBlockAppendOwnedOperation(body, mlir.oraReturnOpCreate(
        mlir_ctx,
        loc,
        &[_]mlir.MlirValue{mlir.oraOperationGetResult(loop, 0)},
        1,
    ));

    try encoder.registerFunctionOperation(helper);

    const result_types = [_]mlir.MlirType{i256_ty};
    const call = mlir.oraFuncCallOpCreate(
        mlir_ctx,
        loc,
        stringRef("finiteForReturn"),
        &[_]mlir.MlirValue{},
        0,
        &result_types,
        result_types.len,
    );

    const encoded = try encoder.encodeOperation(call);
    try testing.expect(!encoder.isDegraded());

    const expected = try encoder.encodeIntegerConstant(1, 256);
    const eq = z3.Z3_mk_eq(z3_ctx.ctx, encoded, expected);
    const simplified = z3.Z3_simplify(z3_ctx.ctx, eq);
    const simplified_text = std.mem.span(z3.Z3_ast_to_string(z3_ctx.ctx, simplified));
    try testing.expect(std.mem.eql(u8, simplified_text, "true"));
}

test "direct zero-iteration scf.for iter-arg result encodes exactly" {
    var z3_ctx = try Context.init(testing.allocator);
    defer z3_ctx.deinit();

    var encoder = Encoder.init(&z3_ctx, testing.allocator);
    defer encoder.deinit();

    const mlir_ctx = mlir.oraContextCreate();
    defer mlir.oraContextDestroy(mlir_ctx);
    loadAllDialects(mlir_ctx);
    _ = mlir.oraDialectRegister(mlir_ctx);

    const loc = mlir.oraLocationUnknownGet(mlir_ctx);
    const index_ty = mlir.oraIndexTypeCreate(mlir_ctx);
    const i256_ty = mlir.oraIntegerTypeCreate(mlir_ctx, 256);

    const c5_attr = mlir.oraIntegerAttrCreateI64FromType(index_ty, 5);
    const c1_attr = mlir.oraIntegerAttrCreateI64FromType(index_ty, 1);
    const init_attr = mlir.oraIntegerAttrCreateI64FromType(i256_ty, 13);
    const c5_op = mlir.oraArithConstantOpCreate(mlir_ctx, loc, index_ty, c5_attr);
    const c1_op = mlir.oraArithConstantOpCreate(mlir_ctx, loc, index_ty, c1_attr);
    const init_op = mlir.oraArithConstantOpCreate(mlir_ctx, loc, i256_ty, init_attr);
    const expected = try encoder.encodeOperation(init_op);

    const loop = mlir.oraScfForOpCreate(
        mlir_ctx,
        loc,
        mlir.oraOperationGetResult(c5_op, 0),
        mlir.oraOperationGetResult(c5_op, 0),
        mlir.oraOperationGetResult(c1_op, 0),
        &[_]mlir.MlirValue{mlir.oraOperationGetResult(init_op, 0)},
        1,
        false,
    );

    const encoded = try encoder.encodeValue(mlir.oraOperationGetResult(loop, 0));
    try testing.expect(!encoder.isDegraded());

    var solver = try Solver.init(&z3_ctx, testing.allocator);
    defer solver.deinit();
    solver.assert(z3.Z3_mk_not(z3_ctx.ctx, z3.Z3_mk_eq(z3_ctx.ctx, encoded, expected)));
    try testing.expectEqual(@as(z3.Z3_lbool, z3.Z3_L_FALSE), solver.check());
}

test "direct symbolic scf.for identity iter-arg result encodes exactly" {
    var z3_ctx = try Context.init(testing.allocator);
    defer z3_ctx.deinit();

    var encoder = Encoder.init(&z3_ctx, testing.allocator);
    defer encoder.deinit();

    const mlir_ctx = mlir.oraContextCreate();
    defer mlir.oraContextDestroy(mlir_ctx);
    loadAllDialects(mlir_ctx);
    _ = mlir.oraDialectRegister(mlir_ctx);

    const loc = mlir.oraLocationUnknownGet(mlir_ctx);
    const index_ty = mlir.oraIndexTypeCreate(mlir_ctx);
    const i256_ty = mlir.oraIntegerTypeCreate(mlir_ctx, 256);

    const c0_attr = mlir.oraIntegerAttrCreateI64FromType(index_ty, 0);
    const c1_attr = mlir.oraIntegerAttrCreateI64FromType(index_ty, 1);
    const init_attr = mlir.oraIntegerAttrCreateI64FromType(i256_ty, 13);
    const c0_op = mlir.oraArithConstantOpCreate(mlir_ctx, loc, index_ty, c0_attr);
    const c1_op = mlir.oraArithConstantOpCreate(mlir_ctx, loc, index_ty, c1_attr);
    const init_op = mlir.oraArithConstantOpCreate(mlir_ctx, loc, i256_ty, init_attr);
    const ub_op = mlir.oraVariablePlaceholderOpCreate(mlir_ctx, loc, stringRef("forIdentityUb"), index_ty);
    const expected = try encoder.encodeOperation(init_op);

    const loop = mlir.oraScfForOpCreate(
        mlir_ctx,
        loc,
        mlir.oraOperationGetResult(c0_op, 0),
        mlir.oraOperationGetResult(ub_op, 0),
        mlir.oraOperationGetResult(c1_op, 0),
        &[_]mlir.MlirValue{mlir.oraOperationGetResult(init_op, 0)},
        1,
        false,
    );
    const body = mlir.oraScfForOpGetBodyBlock(loop);
    const carried_arg = mlir.oraBlockGetArgument(body, 1);
    mlir.oraBlockAppendOwnedOperation(body, mlir.oraScfYieldOpCreate(
        mlir_ctx,
        loc,
        &[_]mlir.MlirValue{carried_arg},
        1,
    ));

    const encoded = try encoder.encodeValue(mlir.oraOperationGetResult(loop, 0));
    try testing.expect(!encoder.isDegraded());

    var solver = try Solver.init(&z3_ctx, testing.allocator);
    defer solver.deinit();
    solver.assert(z3.Z3_mk_not(z3_ctx.ctx, z3.Z3_mk_eq(z3_ctx.ctx, encoded, expected)));
    try testing.expectEqual(@as(z3.Z3_lbool, z3.Z3_L_FALSE), solver.check());
}

test "direct symbolic canonical increment scf.for result encodes exactly" {
    var z3_ctx = try Context.init(testing.allocator);
    defer z3_ctx.deinit();

    var encoder = Encoder.init(&z3_ctx, testing.allocator);
    defer encoder.deinit();

    const mlir_ctx = mlir.oraContextCreate();
    defer mlir.oraContextDestroy(mlir_ctx);
    loadAllDialects(mlir_ctx);
    _ = mlir.oraDialectRegister(mlir_ctx);

    const loc = mlir.oraLocationUnknownGet(mlir_ctx);
    const index_ty = mlir.oraIndexTypeCreate(mlir_ctx);
    const i256_ty = mlir.oraIntegerTypeCreate(mlir_ctx, 256);

    const c0_attr = mlir.oraIntegerAttrCreateI64FromType(index_ty, 0);
    const c1_attr = mlir.oraIntegerAttrCreateI64FromType(index_ty, 1);
    const init_attr = mlir.oraIntegerAttrCreateI64FromType(i256_ty, 13);
    const c0_op = mlir.oraArithConstantOpCreate(mlir_ctx, loc, index_ty, c0_attr);
    const c1_op = mlir.oraArithConstantOpCreate(mlir_ctx, loc, index_ty, c1_attr);
    const init_op = mlir.oraArithConstantOpCreate(mlir_ctx, loc, i256_ty, init_attr);
    const ub_op = mlir.oraVariablePlaceholderOpCreate(mlir_ctx, loc, stringRef("forIncrementUb"), index_ty);
    const expected_init = try encoder.encodeOperation(init_op);

    const loop = mlir.oraScfForOpCreate(
        mlir_ctx,
        loc,
        mlir.oraOperationGetResult(c0_op, 0),
        mlir.oraOperationGetResult(ub_op, 0),
        mlir.oraOperationGetResult(c1_op, 0),
        &[_]mlir.MlirValue{mlir.oraOperationGetResult(init_op, 0)},
        1,
        false,
    );
    const body = mlir.oraScfForOpGetBodyBlock(loop);
    const carried_arg = mlir.oraBlockGetArgument(body, 1);
    const next_op = mlir.oraArithAddIOpCreate(mlir_ctx, loc, carried_arg, mlir.oraOperationGetResult(c1_op, 0));
    mlir.oraBlockAppendOwnedOperation(body, next_op);
    mlir.oraBlockAppendOwnedOperation(body, mlir.oraScfYieldOpCreate(
        mlir_ctx,
        loc,
        &[_]mlir.MlirValue{mlir.oraOperationGetResult(next_op, 0)},
        1,
    ));

    const encoded = try encoder.encodeValue(mlir.oraOperationGetResult(loop, 0));
    try testing.expect(!encoder.isDegraded());
    const ub_ast = try encoder.encodeOperation(ub_op);
    const expected = try encoder.encodeArithmeticOp(.Add, expected_init, ub_ast);

    var solver = try Solver.init(&z3_ctx, testing.allocator);
    defer solver.deinit();
    solver.assert(z3.Z3_mk_not(z3_ctx.ctx, z3.Z3_mk_eq(z3_ctx.ctx, encoded, expected)));
    try testing.expectEqual(@as(z3.Z3_lbool, z3.Z3_L_FALSE), solver.check());
}

test "direct symbolic canonical decrement scf.for result encodes exactly" {
    var z3_ctx = try Context.init(testing.allocator);
    defer z3_ctx.deinit();

    var encoder = Encoder.init(&z3_ctx, testing.allocator);
    defer encoder.deinit();

    const mlir_ctx = mlir.oraContextCreate();
    defer mlir.oraContextDestroy(mlir_ctx);
    loadAllDialects(mlir_ctx);
    _ = mlir.oraDialectRegister(mlir_ctx);

    const loc = mlir.oraLocationUnknownGet(mlir_ctx);
    const index_ty = mlir.oraIndexTypeCreate(mlir_ctx);
    const i256_ty = mlir.oraIntegerTypeCreate(mlir_ctx, 256);

    const c0_attr = mlir.oraIntegerAttrCreateI64FromType(index_ty, 0);
    const c1_attr = mlir.oraIntegerAttrCreateI64FromType(index_ty, 1);
    const init_attr = mlir.oraIntegerAttrCreateI64FromType(i256_ty, 13);
    const c0_op = mlir.oraArithConstantOpCreate(mlir_ctx, loc, index_ty, c0_attr);
    const c1_op = mlir.oraArithConstantOpCreate(mlir_ctx, loc, index_ty, c1_attr);
    const init_op = mlir.oraArithConstantOpCreate(mlir_ctx, loc, i256_ty, init_attr);
    const ub_op = mlir.oraVariablePlaceholderOpCreate(mlir_ctx, loc, stringRef("forDecrementUb"), index_ty);
    const expected_init = try encoder.encodeOperation(init_op);

    const loop = mlir.oraScfForOpCreate(
        mlir_ctx,
        loc,
        mlir.oraOperationGetResult(c0_op, 0),
        mlir.oraOperationGetResult(ub_op, 0),
        mlir.oraOperationGetResult(c1_op, 0),
        &[_]mlir.MlirValue{mlir.oraOperationGetResult(init_op, 0)},
        1,
        false,
    );
    const body = mlir.oraScfForOpGetBodyBlock(loop);
    const carried_arg = mlir.oraBlockGetArgument(body, 1);
    const next_op = mlir.oraArithSubIOpCreate(mlir_ctx, loc, carried_arg, mlir.oraOperationGetResult(c1_op, 0));
    mlir.oraBlockAppendOwnedOperation(body, next_op);
    mlir.oraBlockAppendOwnedOperation(body, mlir.oraScfYieldOpCreate(
        mlir_ctx,
        loc,
        &[_]mlir.MlirValue{mlir.oraOperationGetResult(next_op, 0)},
        1,
    ));

    const encoded = try encoder.encodeValue(mlir.oraOperationGetResult(loop, 0));
    try testing.expect(!encoder.isDegraded());
    const ub_ast = try encoder.encodeOperation(ub_op);
    const expected = try encoder.encodeArithmeticOp(.Sub, expected_init, ub_ast);

    var solver = try Solver.init(&z3_ctx, testing.allocator);
    defer solver.deinit();
    solver.assert(z3.Z3_mk_not(z3_ctx.ctx, z3.Z3_mk_eq(z3_ctx.ctx, encoded, expected)));
    try testing.expectEqual(@as(z3.Z3_lbool, z3.Z3_L_FALSE), solver.check());
}

test "direct symbolic canonical positive-step scf.for result encodes exactly" {
    var z3_ctx = try Context.init(testing.allocator);
    defer z3_ctx.deinit();

    var encoder = Encoder.init(&z3_ctx, testing.allocator);
    defer encoder.deinit();

    const mlir_ctx = mlir.oraContextCreate();
    defer mlir.oraContextDestroy(mlir_ctx);
    loadAllDialects(mlir_ctx);
    _ = mlir.oraDialectRegister(mlir_ctx);

    const loc = mlir.oraLocationUnknownGet(mlir_ctx);
    const index_ty = mlir.oraIndexTypeCreate(mlir_ctx);
    const i256_ty = mlir.oraIntegerTypeCreate(mlir_ctx, 256);

    const ub = mlir.oraVariablePlaceholderOpCreate(mlir_ctx, loc, stringRef("ub_step2"), index_ty);
    const init_attr = mlir.oraIntegerAttrCreateI64FromType(i256_ty, 5);
    const init_op = mlir.oraArithConstantOpCreate(mlir_ctx, loc, i256_ty, init_attr);
    const lb_attr = mlir.oraIntegerAttrCreateI64FromType(index_ty, 0);
    const step_attr = mlir.oraIntegerAttrCreateI64FromType(index_ty, 2);
    const lb_op = mlir.oraArithConstantOpCreate(mlir_ctx, loc, index_ty, lb_attr);
    const step_op = mlir.oraArithConstantOpCreate(mlir_ctx, loc, index_ty, step_attr);

    const loop = mlir.oraScfForOpCreate(
        mlir_ctx,
        loc,
        mlir.oraOperationGetResult(lb_op, 0),
        mlir.oraOperationGetResult(ub, 0),
        mlir.oraOperationGetResult(step_op, 0),
        &[_]mlir.MlirValue{mlir.oraOperationGetResult(init_op, 0)},
        1,
        true,
    );
    const body = mlir.oraScfForOpGetBodyBlock(loop);
    const carried = mlir.oraBlockGetArgument(body, 1);
    const one_attr = mlir.oraIntegerAttrCreateI64FromType(i256_ty, 1);
    const one_op = mlir.oraArithConstantOpCreate(mlir_ctx, loc, i256_ty, one_attr);
    const next_op = mlir.oraArithAddIOpCreate(mlir_ctx, loc, carried, mlir.oraOperationGetResult(one_op, 0));
    mlir.oraBlockAppendOwnedOperation(body, one_op);
    mlir.oraBlockAppendOwnedOperation(body, next_op);
    mlir.oraBlockAppendOwnedOperation(body, mlir.oraScfYieldOpCreate(
        mlir_ctx,
        loc,
        &[_]mlir.MlirValue{mlir.oraOperationGetResult(next_op, 0)},
        1,
    ));

    const encoded = try encoder.encodeValue(mlir.oraOperationGetResult(loop, 0));
    try testing.expect(!encoder.isDegraded());

    const init_ast = try encoder.encodeOperation(init_op);
    const ub_ast = try encoder.encodeOperation(ub);
    const sort = z3.Z3_get_sort(z3_ctx.ctx, ub_ast);
    const zero = z3.Z3_mk_unsigned_int64(z3_ctx.ctx, 0, sort);
    const one = z3.Z3_mk_unsigned_int64(z3_ctx.ctx, 1, sort);
    const two = z3.Z3_mk_unsigned_int64(z3_ctx.ctx, 2, sort);
    const trip_count = z3.Z3_mk_ite(
        z3_ctx.ctx,
        z3.Z3_mk_eq(z3_ctx.ctx, ub_ast, zero),
        zero,
        z3.Z3_mk_bv_add(
            z3_ctx.ctx,
            z3.Z3_mk_bv_udiv(z3_ctx.ctx, z3.Z3_mk_bv_sub(z3_ctx.ctx, ub_ast, one), two),
            one,
        ),
    );
    const expected = try encoder.encodeArithmeticOp(.Add, init_ast, trip_count);

    var solver = try Solver.init(&z3_ctx, testing.allocator);
    defer solver.deinit();
    solver.assert(z3.Z3_mk_not(z3_ctx.ctx, z3.Z3_mk_eq(z3_ctx.ctx, encoded, expected)));
    try testing.expectEqual(@as(z3.Z3_lbool, z3.Z3_L_FALSE), solver.check());
}

test "known pure callee finite five-iteration scf.for return encodes exactly" {
    var z3_ctx = try Context.init(testing.allocator);
    defer z3_ctx.deinit();

    var encoder = Encoder.init(&z3_ctx, testing.allocator);
    defer encoder.deinit();

    const mlir_ctx = mlir.oraContextCreate();
    defer mlir.oraContextDestroy(mlir_ctx);
    loadAllDialects(mlir_ctx);
    _ = mlir.oraDialectRegister(mlir_ctx);

    const loc = mlir.oraLocationUnknownGet(mlir_ctx);
    const index_ty = mlir.oraIndexTypeCreate(mlir_ctx);
    const i256_ty = mlir.oraIntegerTypeCreate(mlir_ctx, 256);

    const helper_attrs = [_]mlir.MlirNamedAttribute{
        namedAttr(mlir_ctx, "sym_name", mlir.oraStringAttrCreate(mlir_ctx, stringRef("finiteFiveForReturn"))),
    };
    const helper = mlir.oraFuncFuncOpCreate(mlir_ctx, loc, &helper_attrs, helper_attrs.len, &[_]mlir.MlirType{}, &[_]mlir.MlirLocation{}, 0);
    const body = mlir.oraFuncOpGetBodyBlock(helper);

    const c0_attr = mlir.oraIntegerAttrCreateI64FromType(index_ty, 0);
    const c5_attr = mlir.oraIntegerAttrCreateI64FromType(index_ty, 5);
    const c1_attr = mlir.oraIntegerAttrCreateI64FromType(index_ty, 1);
    const c0_op = mlir.oraArithConstantOpCreate(mlir_ctx, loc, index_ty, c0_attr);
    const c5_op = mlir.oraArithConstantOpCreate(mlir_ctx, loc, index_ty, c5_attr);
    const c1_op = mlir.oraArithConstantOpCreate(mlir_ctx, loc, index_ty, c1_attr);
    mlir.oraBlockAppendOwnedOperation(body, c0_op);
    mlir.oraBlockAppendOwnedOperation(body, c5_op);
    mlir.oraBlockAppendOwnedOperation(body, c1_op);

    const init_attr = mlir.oraIntegerAttrCreateI64FromType(i256_ty, 0);
    const init_op = mlir.oraArithConstantOpCreate(mlir_ctx, loc, i256_ty, init_attr);
    mlir.oraBlockAppendOwnedOperation(body, init_op);

    const loop = mlir.oraScfForOpCreate(
        mlir_ctx,
        loc,
        mlir.oraOperationGetResult(c0_op, 0),
        mlir.oraOperationGetResult(c5_op, 0),
        mlir.oraOperationGetResult(c1_op, 0),
        &[_]mlir.MlirValue{mlir.oraOperationGetResult(init_op, 0)},
        1,
        false,
    );
    const loop_body = mlir.oraScfForOpGetBodyBlock(loop);
    const iv = mlir.oraBlockGetArgument(loop_body, 0);
    const carried = mlir.oraBlockGetArgument(loop_body, 1);
    const iv_i256 = mlir.oraArithIndexCastUIOpCreate(mlir_ctx, loc, iv, i256_ty);
    mlir.oraBlockAppendOwnedOperation(loop_body, iv_i256);
    const next = mlir.oraArithAddIOpCreate(
        mlir_ctx,
        loc,
        carried,
        mlir.oraOperationGetResult(iv_i256, 0),
    );
    mlir.oraBlockAppendOwnedOperation(loop_body, next);
    mlir.oraBlockAppendOwnedOperation(loop_body, mlir.oraScfYieldOpCreate(
        mlir_ctx,
        loc,
        &[_]mlir.MlirValue{mlir.oraOperationGetResult(next, 0)},
        1,
    ));
    mlir.oraBlockAppendOwnedOperation(body, loop);
    mlir.oraBlockAppendOwnedOperation(body, mlir.oraReturnOpCreate(
        mlir_ctx,
        loc,
        &[_]mlir.MlirValue{mlir.oraOperationGetResult(loop, 0)},
        1,
    ));

    try encoder.registerFunctionOperation(helper);

    const result_types = [_]mlir.MlirType{i256_ty};
    const call = mlir.oraFuncCallOpCreate(
        mlir_ctx,
        loc,
        stringRef("finiteFiveForReturn"),
        &[_]mlir.MlirValue{},
        0,
        &result_types,
        result_types.len,
    );

    const encoded = try encoder.encodeOperation(call);
    try testing.expect(!encoder.isDegraded());

    const expected = try encoder.encodeIntegerConstant(10, 256);
    var solver = try Solver.init(&z3_ctx, testing.allocator);
    defer solver.deinit();
    solver.assert(z3.Z3_mk_not(z3_ctx.ctx, z3.Z3_mk_eq(z3_ctx.ctx, encoded, expected)));
    try testing.expectEqual(@as(z3.Z3_lbool, z3.Z3_L_FALSE), solver.check());
}

test "known pure callee symbolic identity scf.for return encodes exactly" {
    var z3_ctx = try Context.init(testing.allocator);
    defer z3_ctx.deinit();

    var encoder = Encoder.init(&z3_ctx, testing.allocator);
    defer encoder.deinit();

    const mlir_ctx = mlir.oraContextCreate();
    defer mlir.oraContextDestroy(mlir_ctx);
    loadAllDialects(mlir_ctx);
    _ = mlir.oraDialectRegister(mlir_ctx);

    const loc = mlir.oraLocationUnknownGet(mlir_ctx);
    const index_ty = mlir.oraIndexTypeCreate(mlir_ctx);
    const i256_ty = mlir.oraIntegerTypeCreate(mlir_ctx, 256);

    const helper_attrs = [_]mlir.MlirNamedAttribute{
        namedAttr(mlir_ctx, "sym_name", mlir.oraStringAttrCreate(mlir_ctx, stringRef("symbolicIdentityForReturn"))),
    };
    const helper_param_types = [_]mlir.MlirType{index_ty};
    const helper_param_locs = [_]mlir.MlirLocation{loc};
    const helper = mlir.oraFuncFuncOpCreate(mlir_ctx, loc, &helper_attrs, helper_attrs.len, &helper_param_types, &helper_param_locs, helper_param_types.len);
    const body = mlir.oraFuncOpGetBodyBlock(helper);

    const c0_attr = mlir.oraIntegerAttrCreateI64FromType(index_ty, 0);
    const c1_attr = mlir.oraIntegerAttrCreateI64FromType(index_ty, 1);
    const init_attr = mlir.oraIntegerAttrCreateI64FromType(i256_ty, 13);
    const c0_op = mlir.oraArithConstantOpCreate(mlir_ctx, loc, index_ty, c0_attr);
    const c1_op = mlir.oraArithConstantOpCreate(mlir_ctx, loc, index_ty, c1_attr);
    const init_op = mlir.oraArithConstantOpCreate(mlir_ctx, loc, i256_ty, init_attr);
    mlir.oraBlockAppendOwnedOperation(body, c0_op);
    mlir.oraBlockAppendOwnedOperation(body, c1_op);
    mlir.oraBlockAppendOwnedOperation(body, init_op);

    const ub = mlir.oraBlockGetArgument(body, 0);
    const loop = mlir.oraScfForOpCreate(
        mlir_ctx,
        loc,
        mlir.oraOperationGetResult(c0_op, 0),
        ub,
        mlir.oraOperationGetResult(c1_op, 0),
        &[_]mlir.MlirValue{mlir.oraOperationGetResult(init_op, 0)},
        1,
        false,
    );
    const loop_body = mlir.oraScfForOpGetBodyBlock(loop);
    const carried = mlir.oraBlockGetArgument(loop_body, 1);
    mlir.oraBlockAppendOwnedOperation(loop_body, mlir.oraScfYieldOpCreate(
        mlir_ctx,
        loc,
        &[_]mlir.MlirValue{carried},
        1,
    ));
    mlir.oraBlockAppendOwnedOperation(body, loop);
    mlir.oraBlockAppendOwnedOperation(body, mlir.oraReturnOpCreate(
        mlir_ctx,
        loc,
        &[_]mlir.MlirValue{mlir.oraOperationGetResult(loop, 0)},
        1,
    ));

    try encoder.registerFunctionOperation(helper);

    const ub_op = mlir.oraVariablePlaceholderOpCreate(mlir_ctx, loc, stringRef("pureIdentityForUb"), index_ty);
    const result_types = [_]mlir.MlirType{i256_ty};
    const call = mlir.oraFuncCallOpCreate(
        mlir_ctx,
        loc,
        stringRef("symbolicIdentityForReturn"),
        &[_]mlir.MlirValue{mlir.oraOperationGetResult(ub_op, 0)},
        1,
        &result_types,
        result_types.len,
    );

    const encoded = try encoder.encodeOperation(call);
    try testing.expect(!encoder.isDegraded());

    const expected = try encoder.encodeOperation(init_op);
    var solver = try Solver.init(&z3_ctx, testing.allocator);
    defer solver.deinit();
    solver.assert(z3.Z3_mk_not(z3_ctx.ctx, z3.Z3_mk_eq(z3_ctx.ctx, encoded, expected)));
    try testing.expectEqual(@as(z3.Z3_lbool, z3.Z3_L_FALSE), solver.check());
}

test "known pure callee symbolic canonical increment scf.for return encodes exactly" {
    var z3_ctx = try Context.init(testing.allocator);
    defer z3_ctx.deinit();

    var encoder = Encoder.init(&z3_ctx, testing.allocator);
    defer encoder.deinit();

    const mlir_ctx = mlir.oraContextCreate();
    defer mlir.oraContextDestroy(mlir_ctx);
    loadAllDialects(mlir_ctx);
    _ = mlir.oraDialectRegister(mlir_ctx);

    const loc = mlir.oraLocationUnknownGet(mlir_ctx);
    const index_ty = mlir.oraIndexTypeCreate(mlir_ctx);
    const i256_ty = mlir.oraIntegerTypeCreate(mlir_ctx, 256);

    const helper_attrs = [_]mlir.MlirNamedAttribute{
        namedAttr(mlir_ctx, "sym_name", mlir.oraStringAttrCreate(mlir_ctx, stringRef("symbolicIncrementForReturn"))),
    };
    const helper_param_types = [_]mlir.MlirType{index_ty};
    const helper_param_locs = [_]mlir.MlirLocation{loc};
    const helper = mlir.oraFuncFuncOpCreate(mlir_ctx, loc, &helper_attrs, helper_attrs.len, &helper_param_types, &helper_param_locs, helper_param_types.len);
    const body = mlir.oraFuncOpGetBodyBlock(helper);

    const c0_attr = mlir.oraIntegerAttrCreateI64FromType(index_ty, 0);
    const c1_attr = mlir.oraIntegerAttrCreateI64FromType(index_ty, 1);
    const init_attr = mlir.oraIntegerAttrCreateI64FromType(i256_ty, 13);
    const c0_op = mlir.oraArithConstantOpCreate(mlir_ctx, loc, index_ty, c0_attr);
    const c1_op = mlir.oraArithConstantOpCreate(mlir_ctx, loc, index_ty, c1_attr);
    const init_op = mlir.oraArithConstantOpCreate(mlir_ctx, loc, i256_ty, init_attr);
    mlir.oraBlockAppendOwnedOperation(body, c0_op);
    mlir.oraBlockAppendOwnedOperation(body, c1_op);
    mlir.oraBlockAppendOwnedOperation(body, init_op);

    const ub = mlir.oraBlockGetArgument(body, 0);
    const loop = mlir.oraScfForOpCreate(
        mlir_ctx,
        loc,
        mlir.oraOperationGetResult(c0_op, 0),
        ub,
        mlir.oraOperationGetResult(c1_op, 0),
        &[_]mlir.MlirValue{mlir.oraOperationGetResult(init_op, 0)},
        1,
        false,
    );
    const loop_body = mlir.oraScfForOpGetBodyBlock(loop);
    const carried = mlir.oraBlockGetArgument(loop_body, 1);
    const next_op = mlir.oraArithAddIOpCreate(mlir_ctx, loc, carried, mlir.oraOperationGetResult(c1_op, 0));
    mlir.oraBlockAppendOwnedOperation(loop_body, next_op);
    mlir.oraBlockAppendOwnedOperation(loop_body, mlir.oraScfYieldOpCreate(
        mlir_ctx,
        loc,
        &[_]mlir.MlirValue{mlir.oraOperationGetResult(next_op, 0)},
        1,
    ));
    mlir.oraBlockAppendOwnedOperation(body, loop);
    mlir.oraBlockAppendOwnedOperation(body, mlir.oraReturnOpCreate(
        mlir_ctx,
        loc,
        &[_]mlir.MlirValue{mlir.oraOperationGetResult(loop, 0)},
        1,
    ));

    try encoder.registerFunctionOperation(helper);

    const ub_op = mlir.oraVariablePlaceholderOpCreate(mlir_ctx, loc, stringRef("pureIncrementForUb"), index_ty);
    const result_types = [_]mlir.MlirType{i256_ty};
    const call = mlir.oraFuncCallOpCreate(
        mlir_ctx,
        loc,
        stringRef("symbolicIncrementForReturn"),
        &[_]mlir.MlirValue{mlir.oraOperationGetResult(ub_op, 0)},
        1,
        &result_types,
        result_types.len,
    );

    const encoded = try encoder.encodeOperation(call);
    try testing.expect(!encoder.isDegraded());

    const init_ast = try encoder.encodeOperation(init_op);
    const ub_ast = try encoder.encodeOperation(ub_op);
    const expected = try encoder.encodeArithmeticOp(.Add, init_ast, ub_ast);
    var solver = try Solver.init(&z3_ctx, testing.allocator);
    defer solver.deinit();
    solver.assert(z3.Z3_mk_not(z3_ctx.ctx, z3.Z3_mk_eq(z3_ctx.ctx, encoded, expected)));
    try testing.expectEqual(@as(z3.Z3_lbool, z3.Z3_L_FALSE), solver.check());
}

test "known pure callee symbolic canonical decrement scf.for return encodes exactly" {
    var z3_ctx = try Context.init(testing.allocator);
    defer z3_ctx.deinit();

    var encoder = Encoder.init(&z3_ctx, testing.allocator);
    defer encoder.deinit();

    const mlir_ctx = mlir.oraContextCreate();
    defer mlir.oraContextDestroy(mlir_ctx);
    loadAllDialects(mlir_ctx);
    _ = mlir.oraDialectRegister(mlir_ctx);

    const loc = mlir.oraLocationUnknownGet(mlir_ctx);
    const index_ty = mlir.oraIndexTypeCreate(mlir_ctx);
    const i256_ty = mlir.oraIntegerTypeCreate(mlir_ctx, 256);

    const helper_attrs = [_]mlir.MlirNamedAttribute{
        namedAttr(mlir_ctx, "sym_name", mlir.oraStringAttrCreate(mlir_ctx, stringRef("symbolicDecrementForReturn"))),
    };
    const helper_param_types = [_]mlir.MlirType{index_ty};
    const helper_param_locs = [_]mlir.MlirLocation{loc};
    const helper = mlir.oraFuncFuncOpCreate(mlir_ctx, loc, &helper_attrs, helper_attrs.len, &helper_param_types, &helper_param_locs, helper_param_types.len);
    const body = mlir.oraFuncOpGetBodyBlock(helper);

    const c0_attr = mlir.oraIntegerAttrCreateI64FromType(index_ty, 0);
    const c1_attr = mlir.oraIntegerAttrCreateI64FromType(index_ty, 1);
    const init_attr = mlir.oraIntegerAttrCreateI64FromType(i256_ty, 13);
    const c0_op = mlir.oraArithConstantOpCreate(mlir_ctx, loc, index_ty, c0_attr);
    const c1_op = mlir.oraArithConstantOpCreate(mlir_ctx, loc, index_ty, c1_attr);
    const init_op = mlir.oraArithConstantOpCreate(mlir_ctx, loc, i256_ty, init_attr);
    mlir.oraBlockAppendOwnedOperation(body, c0_op);
    mlir.oraBlockAppendOwnedOperation(body, c1_op);
    mlir.oraBlockAppendOwnedOperation(body, init_op);

    const ub = mlir.oraBlockGetArgument(body, 0);
    const loop = mlir.oraScfForOpCreate(
        mlir_ctx,
        loc,
        mlir.oraOperationGetResult(c0_op, 0),
        ub,
        mlir.oraOperationGetResult(c1_op, 0),
        &[_]mlir.MlirValue{mlir.oraOperationGetResult(init_op, 0)},
        1,
        false,
    );
    const loop_body = mlir.oraScfForOpGetBodyBlock(loop);
    const carried = mlir.oraBlockGetArgument(loop_body, 1);
    const next_op = mlir.oraArithSubIOpCreate(mlir_ctx, loc, carried, mlir.oraOperationGetResult(c1_op, 0));
    mlir.oraBlockAppendOwnedOperation(loop_body, next_op);
    mlir.oraBlockAppendOwnedOperation(loop_body, mlir.oraScfYieldOpCreate(
        mlir_ctx,
        loc,
        &[_]mlir.MlirValue{mlir.oraOperationGetResult(next_op, 0)},
        1,
    ));
    mlir.oraBlockAppendOwnedOperation(body, loop);
    mlir.oraBlockAppendOwnedOperation(body, mlir.oraReturnOpCreate(
        mlir_ctx,
        loc,
        &[_]mlir.MlirValue{mlir.oraOperationGetResult(loop, 0)},
        1,
    ));

    try encoder.registerFunctionOperation(helper);

    const ub_op = mlir.oraVariablePlaceholderOpCreate(mlir_ctx, loc, stringRef("pureDecrementForUb"), index_ty);
    const result_types = [_]mlir.MlirType{i256_ty};
    const call = mlir.oraFuncCallOpCreate(
        mlir_ctx,
        loc,
        stringRef("symbolicDecrementForReturn"),
        &[_]mlir.MlirValue{mlir.oraOperationGetResult(ub_op, 0)},
        1,
        &result_types,
        result_types.len,
    );

    const encoded = try encoder.encodeOperation(call);
    try testing.expect(!encoder.isDegraded());

    const init_ast = try encoder.encodeOperation(init_op);
    const ub_ast = try encoder.encodeOperation(ub_op);
    const expected = try encoder.encodeArithmeticOp(.Sub, init_ast, ub_ast);
    var solver = try Solver.init(&z3_ctx, testing.allocator);
    defer solver.deinit();
    solver.assert(z3.Z3_mk_not(z3_ctx.ctx, z3.Z3_mk_eq(z3_ctx.ctx, encoded, expected)));
    try testing.expectEqual(@as(z3.Z3_lbool, z3.Z3_L_FALSE), solver.check());
}

test "known pure callee symbolic canonical positive-step scf.for return encodes exactly" {
    var z3_ctx = try Context.init(testing.allocator);
    defer z3_ctx.deinit();

    var encoder = Encoder.init(&z3_ctx, testing.allocator);
    defer encoder.deinit();

    const mlir_ctx = mlir.oraContextCreate();
    defer mlir.oraContextDestroy(mlir_ctx);
    loadAllDialects(mlir_ctx);
    _ = mlir.oraDialectRegister(mlir_ctx);

    const loc = mlir.oraLocationUnknownGet(mlir_ctx);
    const index_ty = mlir.oraIndexTypeCreate(mlir_ctx);
    const i256_ty = mlir.oraIntegerTypeCreate(mlir_ctx, 256);

    const helper_attrs = [_]mlir.MlirNamedAttribute{
        namedAttr(mlir_ctx, "sym_name", mlir.oraStringAttrCreate(mlir_ctx, stringRef("symbolicStepTwoForReturn"))),
    };
    const helper_param_types = [_]mlir.MlirType{index_ty};
    const helper_param_locs = [_]mlir.MlirLocation{loc};
    const helper = mlir.oraFuncFuncOpCreate(mlir_ctx, loc, &helper_attrs, helper_attrs.len, &helper_param_types, &helper_param_locs, helper_param_types.len);
    const body = mlir.oraFuncOpGetBodyBlock(helper);

    const init_attr = mlir.oraIntegerAttrCreateI64FromType(i256_ty, 5);
    const init_op = mlir.oraArithConstantOpCreate(mlir_ctx, loc, i256_ty, init_attr);
    const lb_attr = mlir.oraIntegerAttrCreateI64FromType(index_ty, 0);
    const step_attr = mlir.oraIntegerAttrCreateI64FromType(index_ty, 2);
    const one_attr = mlir.oraIntegerAttrCreateI64FromType(i256_ty, 1);
    const lb_op = mlir.oraArithConstantOpCreate(mlir_ctx, loc, index_ty, lb_attr);
    const step_op = mlir.oraArithConstantOpCreate(mlir_ctx, loc, index_ty, step_attr);
    const one_op = mlir.oraArithConstantOpCreate(mlir_ctx, loc, i256_ty, one_attr);
    mlir.oraBlockAppendOwnedOperation(body, init_op);
    mlir.oraBlockAppendOwnedOperation(body, lb_op);
    mlir.oraBlockAppendOwnedOperation(body, step_op);
    mlir.oraBlockAppendOwnedOperation(body, one_op);

    const ub = mlir.oraBlockGetArgument(body, 0);
    const loop = mlir.oraScfForOpCreate(
        mlir_ctx,
        loc,
        mlir.oraOperationGetResult(lb_op, 0),
        ub,
        mlir.oraOperationGetResult(step_op, 0),
        &[_]mlir.MlirValue{mlir.oraOperationGetResult(init_op, 0)},
        1,
        true,
    );
    const loop_body = mlir.oraScfForOpGetBodyBlock(loop);
    const carried = mlir.oraBlockGetArgument(loop_body, 1);
    const next_op = mlir.oraArithAddIOpCreate(mlir_ctx, loc, carried, mlir.oraOperationGetResult(one_op, 0));
    mlir.oraBlockAppendOwnedOperation(loop_body, next_op);
    mlir.oraBlockAppendOwnedOperation(loop_body, mlir.oraScfYieldOpCreate(
        mlir_ctx,
        loc,
        &[_]mlir.MlirValue{mlir.oraOperationGetResult(next_op, 0)},
        1,
    ));
    mlir.oraBlockAppendOwnedOperation(body, loop);
    mlir.oraBlockAppendOwnedOperation(body, mlir.oraReturnOpCreate(
        mlir_ctx,
        loc,
        &[_]mlir.MlirValue{mlir.oraOperationGetResult(loop, 0)},
        1,
    ));

    try encoder.registerFunctionOperation(helper);

    const caller_ub = mlir.oraVariablePlaceholderOpCreate(mlir_ctx, loc, stringRef("caller_ub_step2"), index_ty);
    const call = mlir.oraFuncCallOpCreate(
        mlir_ctx,
        loc,
        stringRef("symbolicStepTwoForReturn"),
        &[_]mlir.MlirValue{mlir.oraOperationGetResult(caller_ub, 0)},
        1,
        &[_]mlir.MlirType{i256_ty},
        1,
    );
    const encoded = try encoder.encodeOperation(call);
    try testing.expect(!encoder.isDegraded());

    const init_ast = try encoder.encodeOperation(init_op);
    const ub_ast = try encoder.encodeOperation(caller_ub);
    const sort = z3.Z3_get_sort(z3_ctx.ctx, ub_ast);
    const zero = z3.Z3_mk_unsigned_int64(z3_ctx.ctx, 0, sort);
    const one = z3.Z3_mk_unsigned_int64(z3_ctx.ctx, 1, sort);
    const two = z3.Z3_mk_unsigned_int64(z3_ctx.ctx, 2, sort);
    const trip_count = z3.Z3_mk_ite(
        z3_ctx.ctx,
        z3.Z3_mk_eq(z3_ctx.ctx, ub_ast, zero),
        zero,
        z3.Z3_mk_bv_add(
            z3_ctx.ctx,
            z3.Z3_mk_bv_udiv(z3_ctx.ctx, z3.Z3_mk_bv_sub(z3_ctx.ctx, ub_ast, one), two),
            one,
        ),
    );
    const expected = try encoder.encodeArithmeticOp(.Add, init_ast, trip_count);

    var solver = try Solver.init(&z3_ctx, testing.allocator);
    defer solver.deinit();
    solver.assert(z3.Z3_mk_not(z3_ctx.ctx, z3.Z3_mk_eq(z3_ctx.ctx, encoded, expected)));
    try testing.expectEqual(@as(z3.Z3_lbool, z3.Z3_L_FALSE), solver.check());
}

test "direct symbolic canonical offset positive-step scf.for result encodes exactly" {
    var z3_ctx = try Context.init(testing.allocator);
    defer z3_ctx.deinit();

    var encoder = Encoder.init(&z3_ctx, testing.allocator);
    defer encoder.deinit();

    const mlir_ctx = mlir.oraContextCreate();
    defer mlir.oraContextDestroy(mlir_ctx);
    loadAllDialects(mlir_ctx);
    _ = mlir.oraDialectRegister(mlir_ctx);

    const loc = mlir.oraLocationUnknownGet(mlir_ctx);
    const index_ty = mlir.oraIndexTypeCreate(mlir_ctx);
    const i256_ty = mlir.oraIntegerTypeCreate(mlir_ctx, 256);

    const ub = mlir.oraVariablePlaceholderOpCreate(mlir_ctx, loc, stringRef("ub_step2_offset"), index_ty);
    const init_attr = mlir.oraIntegerAttrCreateI64FromType(i256_ty, 5);
    const init_op = mlir.oraArithConstantOpCreate(mlir_ctx, loc, i256_ty, init_attr);
    const lb_attr = mlir.oraIntegerAttrCreateI64FromType(index_ty, 3);
    const step_attr = mlir.oraIntegerAttrCreateI64FromType(index_ty, 2);
    const lb_op = mlir.oraArithConstantOpCreate(mlir_ctx, loc, index_ty, lb_attr);
    const step_op = mlir.oraArithConstantOpCreate(mlir_ctx, loc, index_ty, step_attr);

    const loop = mlir.oraScfForOpCreate(
        mlir_ctx,
        loc,
        mlir.oraOperationGetResult(lb_op, 0),
        mlir.oraOperationGetResult(ub, 0),
        mlir.oraOperationGetResult(step_op, 0),
        &[_]mlir.MlirValue{mlir.oraOperationGetResult(init_op, 0)},
        1,
        true,
    );
    const body = mlir.oraScfForOpGetBodyBlock(loop);
    const carried = mlir.oraBlockGetArgument(body, 1);
    const one_attr = mlir.oraIntegerAttrCreateI64FromType(i256_ty, 1);
    const one_op = mlir.oraArithConstantOpCreate(mlir_ctx, loc, i256_ty, one_attr);
    const next_op = mlir.oraArithAddIOpCreate(mlir_ctx, loc, carried, mlir.oraOperationGetResult(one_op, 0));
    mlir.oraBlockAppendOwnedOperation(body, one_op);
    mlir.oraBlockAppendOwnedOperation(body, next_op);
    mlir.oraBlockAppendOwnedOperation(body, mlir.oraScfYieldOpCreate(
        mlir_ctx,
        loc,
        &[_]mlir.MlirValue{mlir.oraOperationGetResult(next_op, 0)},
        1,
    ));

    const encoded = try encoder.encodeValue(mlir.oraOperationGetResult(loop, 0));
    try testing.expect(!encoder.isDegraded());

    const init_ast = try encoder.encodeOperation(init_op);
    const lb_ast = try encoder.encodeOperation(lb_op);
    const ub_ast = try encoder.encodeOperation(ub);
    const sort = z3.Z3_get_sort(z3_ctx.ctx, ub_ast);
    const zero = z3.Z3_mk_unsigned_int64(z3_ctx.ctx, 0, sort);
    const one = z3.Z3_mk_unsigned_int64(z3_ctx.ctx, 1, sort);
    const two = z3.Z3_mk_unsigned_int64(z3_ctx.ctx, 2, sort);
    const ub_le_lb = z3.Z3_mk_bvule(z3_ctx.ctx, ub_ast, lb_ast);
    const distance = z3.Z3_mk_bv_sub(z3_ctx.ctx, ub_ast, lb_ast);
    const trip_count = z3.Z3_mk_ite(
        z3_ctx.ctx,
        ub_le_lb,
        zero,
        z3.Z3_mk_bv_add(
            z3_ctx.ctx,
            z3.Z3_mk_bv_udiv(z3_ctx.ctx, z3.Z3_mk_bv_sub(z3_ctx.ctx, distance, one), two),
            one,
        ),
    );
    const expected = try encoder.encodeArithmeticOp(.Add, init_ast, trip_count);

    var solver = try Solver.init(&z3_ctx, testing.allocator);
    defer solver.deinit();
    solver.assert(z3.Z3_mk_not(z3_ctx.ctx, z3.Z3_mk_eq(z3_ctx.ctx, encoded, expected)));
    try testing.expectEqual(@as(z3.Z3_lbool, z3.Z3_L_FALSE), solver.check());
}

test "known pure callee symbolic canonical offset positive-step scf.for return encodes exactly" {
    var z3_ctx = try Context.init(testing.allocator);
    defer z3_ctx.deinit();

    var encoder = Encoder.init(&z3_ctx, testing.allocator);
    defer encoder.deinit();

    const mlir_ctx = mlir.oraContextCreate();
    defer mlir.oraContextDestroy(mlir_ctx);
    loadAllDialects(mlir_ctx);
    _ = mlir.oraDialectRegister(mlir_ctx);

    const loc = mlir.oraLocationUnknownGet(mlir_ctx);
    const index_ty = mlir.oraIndexTypeCreate(mlir_ctx);
    const i256_ty = mlir.oraIntegerTypeCreate(mlir_ctx, 256);

    const helper_attrs = [_]mlir.MlirNamedAttribute{
        namedAttr(mlir_ctx, "sym_name", mlir.oraStringAttrCreate(mlir_ctx, stringRef("symbolicOffsetStepTwoForReturn"))),
    };
    const helper_param_types = [_]mlir.MlirType{index_ty};
    const helper_param_locs = [_]mlir.MlirLocation{loc};
    const helper = mlir.oraFuncFuncOpCreate(mlir_ctx, loc, &helper_attrs, helper_attrs.len, &helper_param_types, &helper_param_locs, helper_param_types.len);
    const body = mlir.oraFuncOpGetBodyBlock(helper);

    const init_attr = mlir.oraIntegerAttrCreateI64FromType(i256_ty, 5);
    const init_op = mlir.oraArithConstantOpCreate(mlir_ctx, loc, i256_ty, init_attr);
    const lb_attr = mlir.oraIntegerAttrCreateI64FromType(index_ty, 3);
    const step_attr = mlir.oraIntegerAttrCreateI64FromType(index_ty, 2);
    const one_attr = mlir.oraIntegerAttrCreateI64FromType(i256_ty, 1);
    const lb_op = mlir.oraArithConstantOpCreate(mlir_ctx, loc, index_ty, lb_attr);
    const step_op = mlir.oraArithConstantOpCreate(mlir_ctx, loc, index_ty, step_attr);
    const one_op = mlir.oraArithConstantOpCreate(mlir_ctx, loc, i256_ty, one_attr);
    mlir.oraBlockAppendOwnedOperation(body, init_op);
    mlir.oraBlockAppendOwnedOperation(body, lb_op);
    mlir.oraBlockAppendOwnedOperation(body, step_op);
    mlir.oraBlockAppendOwnedOperation(body, one_op);

    const ub = mlir.oraBlockGetArgument(body, 0);
    const loop = mlir.oraScfForOpCreate(
        mlir_ctx,
        loc,
        mlir.oraOperationGetResult(lb_op, 0),
        ub,
        mlir.oraOperationGetResult(step_op, 0),
        &[_]mlir.MlirValue{mlir.oraOperationGetResult(init_op, 0)},
        1,
        true,
    );
    const loop_body = mlir.oraScfForOpGetBodyBlock(loop);
    const carried = mlir.oraBlockGetArgument(loop_body, 1);
    const next_op = mlir.oraArithAddIOpCreate(mlir_ctx, loc, carried, mlir.oraOperationGetResult(one_op, 0));
    mlir.oraBlockAppendOwnedOperation(loop_body, next_op);
    mlir.oraBlockAppendOwnedOperation(loop_body, mlir.oraScfYieldOpCreate(
        mlir_ctx,
        loc,
        &[_]mlir.MlirValue{mlir.oraOperationGetResult(next_op, 0)},
        1,
    ));
    mlir.oraBlockAppendOwnedOperation(body, loop);
    mlir.oraBlockAppendOwnedOperation(body, mlir.oraReturnOpCreate(
        mlir_ctx,
        loc,
        &[_]mlir.MlirValue{mlir.oraOperationGetResult(loop, 0)},
        1,
    ));

    try encoder.registerFunctionOperation(helper);

    const caller_ub = mlir.oraVariablePlaceholderOpCreate(mlir_ctx, loc, stringRef("caller_ub_step2_offset"), index_ty);
    const call = mlir.oraFuncCallOpCreate(
        mlir_ctx,
        loc,
        stringRef("symbolicOffsetStepTwoForReturn"),
        &[_]mlir.MlirValue{mlir.oraOperationGetResult(caller_ub, 0)},
        1,
        &[_]mlir.MlirType{i256_ty},
        1,
    );
    const encoded = try encoder.encodeOperation(call);
    try testing.expect(!encoder.isDegraded());

    const init_ast = try encoder.encodeOperation(init_op);
    const lb_ast = try encoder.encodeOperation(lb_op);
    const ub_ast = try encoder.encodeOperation(caller_ub);
    const sort = z3.Z3_get_sort(z3_ctx.ctx, ub_ast);
    const zero = z3.Z3_mk_unsigned_int64(z3_ctx.ctx, 0, sort);
    const one = z3.Z3_mk_unsigned_int64(z3_ctx.ctx, 1, sort);
    const two = z3.Z3_mk_unsigned_int64(z3_ctx.ctx, 2, sort);
    const ub_le_lb = z3.Z3_mk_bvule(z3_ctx.ctx, ub_ast, lb_ast);
    const distance = z3.Z3_mk_bv_sub(z3_ctx.ctx, ub_ast, lb_ast);
    const trip_count = z3.Z3_mk_ite(
        z3_ctx.ctx,
        ub_le_lb,
        zero,
        z3.Z3_mk_bv_add(
            z3_ctx.ctx,
            z3.Z3_mk_bv_udiv(z3_ctx.ctx, z3.Z3_mk_bv_sub(z3_ctx.ctx, distance, one), two),
            one,
        ),
    );
    const expected = try encoder.encodeArithmeticOp(.Add, init_ast, trip_count);

    var solver = try Solver.init(&z3_ctx, testing.allocator);
    defer solver.deinit();
    solver.assert(z3.Z3_mk_not(z3_ctx.ctx, z3.Z3_mk_eq(z3_ctx.ctx, encoded, expected)));
    try testing.expectEqual(@as(z3.Z3_lbool, z3.Z3_L_FALSE), solver.check());
}

test "direct symbolic constant-delta scf.for result encodes exactly" {
    var z3_ctx = try Context.init(testing.allocator);
    defer z3_ctx.deinit();

    var encoder = Encoder.init(&z3_ctx, testing.allocator);
    defer encoder.deinit();

    const mlir_ctx = mlir.oraContextCreate();
    defer mlir.oraContextDestroy(mlir_ctx);
    loadAllDialects(mlir_ctx);
    _ = mlir.oraDialectRegister(mlir_ctx);

    const loc = mlir.oraLocationUnknownGet(mlir_ctx);
    const index_ty = mlir.oraIndexTypeCreate(mlir_ctx);
    const i256_ty = mlir.oraIntegerTypeCreate(mlir_ctx, 256);

    const ub = mlir.oraVariablePlaceholderOpCreate(mlir_ctx, loc, stringRef("ub_delta_three"), index_ty);
    const init_attr = mlir.oraIntegerAttrCreateI64FromType(i256_ty, 5);
    const init_op = mlir.oraArithConstantOpCreate(mlir_ctx, loc, i256_ty, init_attr);
    const lb_attr = mlir.oraIntegerAttrCreateI64FromType(index_ty, 3);
    const step_attr = mlir.oraIntegerAttrCreateI64FromType(index_ty, 2);
    const delta_attr = mlir.oraIntegerAttrCreateI64FromType(i256_ty, 3);
    const lb_op = mlir.oraArithConstantOpCreate(mlir_ctx, loc, index_ty, lb_attr);
    const step_op = mlir.oraArithConstantOpCreate(mlir_ctx, loc, index_ty, step_attr);
    const delta_op = mlir.oraArithConstantOpCreate(mlir_ctx, loc, i256_ty, delta_attr);

    const loop = mlir.oraScfForOpCreate(
        mlir_ctx,
        loc,
        mlir.oraOperationGetResult(lb_op, 0),
        mlir.oraOperationGetResult(ub, 0),
        mlir.oraOperationGetResult(step_op, 0),
        &[_]mlir.MlirValue{mlir.oraOperationGetResult(init_op, 0)},
        1,
        true,
    );
    const body = mlir.oraScfForOpGetBodyBlock(loop);
    const carried = mlir.oraBlockGetArgument(body, 1);
    const next_op = mlir.oraArithAddIOpCreate(mlir_ctx, loc, carried, mlir.oraOperationGetResult(delta_op, 0));
    mlir.oraBlockAppendOwnedOperation(body, next_op);
    mlir.oraBlockAppendOwnedOperation(body, mlir.oraScfYieldOpCreate(
        mlir_ctx,
        loc,
        &[_]mlir.MlirValue{mlir.oraOperationGetResult(next_op, 0)},
        1,
    ));

    const encoded = try encoder.encodeValue(mlir.oraOperationGetResult(loop, 0));
    try testing.expect(!encoder.isDegraded());

    const init_ast = try encoder.encodeOperation(init_op);
    const lb_ast = try encoder.encodeOperation(lb_op);
    const ub_ast = try encoder.encodeOperation(ub);
    const delta_ast = try encoder.encodeOperation(delta_op);
    const sort = z3.Z3_get_sort(z3_ctx.ctx, ub_ast);
    const zero = z3.Z3_mk_unsigned_int64(z3_ctx.ctx, 0, sort);
    const one = z3.Z3_mk_unsigned_int64(z3_ctx.ctx, 1, sort);
    const two = z3.Z3_mk_unsigned_int64(z3_ctx.ctx, 2, sort);
    const ub_le_lb = z3.Z3_mk_bvule(z3_ctx.ctx, ub_ast, lb_ast);
    const distance = z3.Z3_mk_bv_sub(z3_ctx.ctx, ub_ast, lb_ast);
    const trip_count = z3.Z3_mk_ite(
        z3_ctx.ctx,
        ub_le_lb,
        zero,
        z3.Z3_mk_bv_add(
            z3_ctx.ctx,
            z3.Z3_mk_bv_udiv(z3_ctx.ctx, z3.Z3_mk_bv_sub(z3_ctx.ctx, distance, one), two),
            one,
        ),
    );
    const total_delta = try encoder.encodeArithmeticOp(.Mul, trip_count, delta_ast);
    const expected = try encoder.encodeArithmeticOp(.Add, init_ast, total_delta);

    var solver = try Solver.init(&z3_ctx, testing.allocator);
    defer solver.deinit();
    solver.assert(z3.Z3_mk_not(z3_ctx.ctx, z3.Z3_mk_eq(z3_ctx.ctx, encoded, expected)));
    try testing.expectEqual(@as(z3.Z3_lbool, z3.Z3_L_FALSE), solver.check());
}

test "known pure callee symbolic constant-delta scf.for return encodes exactly" {
    var z3_ctx = try Context.init(testing.allocator);
    defer z3_ctx.deinit();

    var encoder = Encoder.init(&z3_ctx, testing.allocator);
    defer encoder.deinit();

    const mlir_ctx = mlir.oraContextCreate();
    defer mlir.oraContextDestroy(mlir_ctx);
    loadAllDialects(mlir_ctx);
    _ = mlir.oraDialectRegister(mlir_ctx);

    const loc = mlir.oraLocationUnknownGet(mlir_ctx);
    const index_ty = mlir.oraIndexTypeCreate(mlir_ctx);
    const i256_ty = mlir.oraIntegerTypeCreate(mlir_ctx, 256);

    const helper_attrs = [_]mlir.MlirNamedAttribute{
        namedAttr(mlir_ctx, "sym_name", mlir.oraStringAttrCreate(mlir_ctx, stringRef("symbolicConstantDeltaForReturn"))),
    };
    const helper_param_types = [_]mlir.MlirType{index_ty};
    const helper_param_locs = [_]mlir.MlirLocation{loc};
    const helper = mlir.oraFuncFuncOpCreate(mlir_ctx, loc, &helper_attrs, helper_attrs.len, &helper_param_types, &helper_param_locs, helper_param_types.len);
    const body = mlir.oraFuncOpGetBodyBlock(helper);

    const init_attr = mlir.oraIntegerAttrCreateI64FromType(i256_ty, 5);
    const init_op = mlir.oraArithConstantOpCreate(mlir_ctx, loc, i256_ty, init_attr);
    const lb_attr = mlir.oraIntegerAttrCreateI64FromType(index_ty, 3);
    const step_attr = mlir.oraIntegerAttrCreateI64FromType(index_ty, 2);
    const delta_attr = mlir.oraIntegerAttrCreateI64FromType(i256_ty, 3);
    const lb_op = mlir.oraArithConstantOpCreate(mlir_ctx, loc, index_ty, lb_attr);
    const step_op = mlir.oraArithConstantOpCreate(mlir_ctx, loc, index_ty, step_attr);
    const delta_op = mlir.oraArithConstantOpCreate(mlir_ctx, loc, i256_ty, delta_attr);
    mlir.oraBlockAppendOwnedOperation(body, init_op);
    mlir.oraBlockAppendOwnedOperation(body, lb_op);
    mlir.oraBlockAppendOwnedOperation(body, step_op);
    mlir.oraBlockAppendOwnedOperation(body, delta_op);

    const ub = mlir.oraBlockGetArgument(body, 0);
    const loop = mlir.oraScfForOpCreate(
        mlir_ctx,
        loc,
        mlir.oraOperationGetResult(lb_op, 0),
        ub,
        mlir.oraOperationGetResult(step_op, 0),
        &[_]mlir.MlirValue{mlir.oraOperationGetResult(init_op, 0)},
        1,
        true,
    );
    const loop_body = mlir.oraScfForOpGetBodyBlock(loop);
    const carried = mlir.oraBlockGetArgument(loop_body, 1);
    const next_op = mlir.oraArithAddIOpCreate(mlir_ctx, loc, carried, mlir.oraOperationGetResult(delta_op, 0));
    mlir.oraBlockAppendOwnedOperation(loop_body, next_op);
    mlir.oraBlockAppendOwnedOperation(loop_body, mlir.oraScfYieldOpCreate(
        mlir_ctx,
        loc,
        &[_]mlir.MlirValue{mlir.oraOperationGetResult(next_op, 0)},
        1,
    ));
    mlir.oraBlockAppendOwnedOperation(body, loop);
    mlir.oraBlockAppendOwnedOperation(body, mlir.oraReturnOpCreate(
        mlir_ctx,
        loc,
        &[_]mlir.MlirValue{mlir.oraOperationGetResult(loop, 0)},
        1,
    ));

    try encoder.registerFunctionOperation(helper);

    const caller_ub = mlir.oraVariablePlaceholderOpCreate(mlir_ctx, loc, stringRef("caller_ub_delta_three"), index_ty);
    const call = mlir.oraFuncCallOpCreate(
        mlir_ctx,
        loc,
        stringRef("symbolicConstantDeltaForReturn"),
        &[_]mlir.MlirValue{mlir.oraOperationGetResult(caller_ub, 0)},
        1,
        &[_]mlir.MlirType{i256_ty},
        1,
    );
    const encoded = try encoder.encodeOperation(call);
    try testing.expect(!encoder.isDegraded());

    const init_ast = try encoder.encodeOperation(init_op);
    const lb_ast = try encoder.encodeOperation(lb_op);
    const ub_ast = try encoder.encodeOperation(caller_ub);
    const delta_ast = try encoder.encodeOperation(delta_op);
    const sort = z3.Z3_get_sort(z3_ctx.ctx, ub_ast);
    const zero = z3.Z3_mk_unsigned_int64(z3_ctx.ctx, 0, sort);
    const one = z3.Z3_mk_unsigned_int64(z3_ctx.ctx, 1, sort);
    const two = z3.Z3_mk_unsigned_int64(z3_ctx.ctx, 2, sort);
    const ub_le_lb = z3.Z3_mk_bvule(z3_ctx.ctx, ub_ast, lb_ast);
    const distance = z3.Z3_mk_bv_sub(z3_ctx.ctx, ub_ast, lb_ast);
    const trip_count = z3.Z3_mk_ite(
        z3_ctx.ctx,
        ub_le_lb,
        zero,
        z3.Z3_mk_bv_add(
            z3_ctx.ctx,
            z3.Z3_mk_bv_udiv(z3_ctx.ctx, z3.Z3_mk_bv_sub(z3_ctx.ctx, distance, one), two),
            one,
        ),
    );
    const total_delta = try encoder.encodeArithmeticOp(.Mul, trip_count, delta_ast);
    const expected = try encoder.encodeArithmeticOp(.Add, init_ast, total_delta);

    var solver = try Solver.init(&z3_ctx, testing.allocator);
    defer solver.deinit();
    solver.assert(z3.Z3_mk_not(z3_ctx.ctx, z3.Z3_mk_eq(z3_ctx.ctx, encoded, expected)));
    try testing.expectEqual(@as(z3.Z3_lbool, z3.Z3_L_FALSE), solver.check());
}

test "direct symbolic multi-result canonical scf.for derived result encodes exactly" {
    var z3_ctx = try Context.init(testing.allocator);
    defer z3_ctx.deinit();

    var encoder = Encoder.init(&z3_ctx, testing.allocator);
    defer encoder.deinit();

    const mlir_ctx = mlir.oraContextCreate();
    defer mlir.oraContextDestroy(mlir_ctx);
    loadAllDialects(mlir_ctx);
    _ = mlir.oraDialectRegister(mlir_ctx);

    const loc = mlir.oraLocationUnknownGet(mlir_ctx);
    const index_ty = mlir.oraIndexTypeCreate(mlir_ctx);
    const i256_ty = mlir.oraIntegerTypeCreate(mlir_ctx, 256);

    const lb_attr = mlir.oraIntegerAttrCreateI64FromType(index_ty, 1);
    const step_attr = mlir.oraIntegerAttrCreateI64FromType(index_ty, 3);
    const init_a_attr = mlir.oraIntegerAttrCreateI64FromType(i256_ty, 7);
    const init_b_attr = mlir.oraIntegerAttrCreateI64FromType(i256_ty, 10);
    const delta_a_attr = mlir.oraIntegerAttrCreateI64FromType(i256_ty, 2);
    const delta_b_attr = mlir.oraIntegerAttrCreateI64FromType(i256_ty, 5);
    const lb_op = mlir.oraArithConstantOpCreate(mlir_ctx, loc, index_ty, lb_attr);
    const step_op = mlir.oraArithConstantOpCreate(mlir_ctx, loc, index_ty, step_attr);
    const init_a_op = mlir.oraArithConstantOpCreate(mlir_ctx, loc, i256_ty, init_a_attr);
    const init_b_op = mlir.oraArithConstantOpCreate(mlir_ctx, loc, i256_ty, init_b_attr);
    const delta_a_op = mlir.oraArithConstantOpCreate(mlir_ctx, loc, i256_ty, delta_a_attr);
    const delta_b_op = mlir.oraArithConstantOpCreate(mlir_ctx, loc, i256_ty, delta_b_attr);
    const ub_op = mlir.oraVariablePlaceholderOpCreate(mlir_ctx, loc, stringRef("forMultiUb"), index_ty);

    const loop = mlir.oraScfForOpCreate(
        mlir_ctx,
        loc,
        mlir.oraOperationGetResult(lb_op, 0),
        mlir.oraOperationGetResult(ub_op, 0),
        mlir.oraOperationGetResult(step_op, 0),
        &[_]mlir.MlirValue{
            mlir.oraOperationGetResult(init_a_op, 0),
            mlir.oraOperationGetResult(init_b_op, 0),
        },
        2,
        true,
    );
    const body = mlir.oraScfForOpGetBodyBlock(loop);
    const carried_a = mlir.oraBlockGetArgument(body, 1);
    const carried_b = mlir.oraBlockGetArgument(body, 2);
    const next_a = mlir.oraArithAddIOpCreate(mlir_ctx, loc, carried_a, mlir.oraOperationGetResult(delta_a_op, 0));
    const next_b = mlir.oraArithAddIOpCreate(mlir_ctx, loc, carried_b, mlir.oraOperationGetResult(delta_b_op, 0));
    mlir.oraBlockAppendOwnedOperation(body, next_a);
    mlir.oraBlockAppendOwnedOperation(body, next_b);
    mlir.oraBlockAppendOwnedOperation(body, mlir.oraScfYieldOpCreate(
        mlir_ctx,
        loc,
        &[_]mlir.MlirValue{
            mlir.oraOperationGetResult(next_a, 0),
            mlir.oraOperationGetResult(next_b, 0),
        },
        2,
    ));

    const encoded = try encoder.encodeValue(mlir.oraOperationGetResult(loop, 1));
    try testing.expect(!encoder.isDegraded());

    const lb_ast = try encoder.encodeOperation(lb_op);
    const ub_ast = try encoder.encodeOperation(ub_op);
    const step_ast = try encoder.encodeOperation(step_op);
    const init_b_ast = try encoder.encodeOperation(init_b_op);
    const delta_b_ast = try encoder.encodeOperation(delta_b_op);
    const sort = z3.Z3_get_sort(z3_ctx.ctx, ub_ast);
    const zero = z3.Z3_mk_unsigned_int64(z3_ctx.ctx, 0, sort);
    const one = z3.Z3_mk_unsigned_int64(z3_ctx.ctx, 1, sort);
    const ub_le_lb = z3.Z3_mk_bvule(z3_ctx.ctx, ub_ast, lb_ast);
    const distance = z3.Z3_mk_bv_sub(z3_ctx.ctx, ub_ast, lb_ast);
    const trip_count = z3.Z3_mk_ite(
        z3_ctx.ctx,
        ub_le_lb,
        zero,
        z3.Z3_mk_bv_add(
            z3_ctx.ctx,
            z3.Z3_mk_bv_udiv(z3_ctx.ctx, z3.Z3_mk_bv_sub(z3_ctx.ctx, distance, one), step_ast),
            one,
        ),
    );
    const total_delta = try encoder.encodeArithmeticOp(.Mul, trip_count, delta_b_ast);
    const expected = try encoder.encodeArithmeticOp(.Add, init_b_ast, total_delta);

    var solver = try Solver.init(&z3_ctx, testing.allocator);
    defer solver.deinit();
    solver.assert(z3.Z3_mk_not(z3_ctx.ctx, z3.Z3_mk_eq(z3_ctx.ctx, encoded, expected)));
    try testing.expectEqual(@as(z3.Z3_lbool, z3.Z3_L_FALSE), solver.check());
}

test "known pure callee symbolic multi-result canonical scf.for return encodes derived result exactly" {
    var z3_ctx = try Context.init(testing.allocator);
    defer z3_ctx.deinit();

    var encoder = Encoder.init(&z3_ctx, testing.allocator);
    defer encoder.deinit();

    const mlir_ctx = mlir.oraContextCreate();
    defer mlir.oraContextDestroy(mlir_ctx);
    loadAllDialects(mlir_ctx);
    _ = mlir.oraDialectRegister(mlir_ctx);

    const loc = mlir.oraLocationUnknownGet(mlir_ctx);
    const index_ty = mlir.oraIndexTypeCreate(mlir_ctx);
    const i256_ty = mlir.oraIntegerTypeCreate(mlir_ctx, 256);

    const helper_attrs = [_]mlir.MlirNamedAttribute{
        namedAttr(mlir_ctx, "sym_name", mlir.oraStringAttrCreate(mlir_ctx, stringRef("symbolicForMultiReturn"))),
    };
    const helper_param_types = [_]mlir.MlirType{index_ty};
    const helper_param_locs = [_]mlir.MlirLocation{loc};
    const helper = mlir.oraFuncFuncOpCreate(mlir_ctx, loc, &helper_attrs, helper_attrs.len, &helper_param_types, &helper_param_locs, helper_param_types.len);
    const body = mlir.oraFuncOpGetBodyBlock(helper);

    const lb_attr = mlir.oraIntegerAttrCreateI64FromType(index_ty, 1);
    const step_attr = mlir.oraIntegerAttrCreateI64FromType(index_ty, 3);
    const init_a_attr = mlir.oraIntegerAttrCreateI64FromType(i256_ty, 7);
    const init_b_attr = mlir.oraIntegerAttrCreateI64FromType(i256_ty, 10);
    const delta_a_attr = mlir.oraIntegerAttrCreateI64FromType(i256_ty, 2);
    const delta_b_attr = mlir.oraIntegerAttrCreateI64FromType(i256_ty, 5);
    const lb_op = mlir.oraArithConstantOpCreate(mlir_ctx, loc, index_ty, lb_attr);
    const step_op = mlir.oraArithConstantOpCreate(mlir_ctx, loc, index_ty, step_attr);
    const init_a_op = mlir.oraArithConstantOpCreate(mlir_ctx, loc, i256_ty, init_a_attr);
    const init_b_op = mlir.oraArithConstantOpCreate(mlir_ctx, loc, i256_ty, init_b_attr);
    const delta_a_op = mlir.oraArithConstantOpCreate(mlir_ctx, loc, i256_ty, delta_a_attr);
    const delta_b_op = mlir.oraArithConstantOpCreate(mlir_ctx, loc, i256_ty, delta_b_attr);
    mlir.oraBlockAppendOwnedOperation(body, lb_op);
    mlir.oraBlockAppendOwnedOperation(body, step_op);
    mlir.oraBlockAppendOwnedOperation(body, init_a_op);
    mlir.oraBlockAppendOwnedOperation(body, init_b_op);
    mlir.oraBlockAppendOwnedOperation(body, delta_a_op);
    mlir.oraBlockAppendOwnedOperation(body, delta_b_op);

    const ub = mlir.oraBlockGetArgument(body, 0);
    const loop = mlir.oraScfForOpCreate(
        mlir_ctx,
        loc,
        mlir.oraOperationGetResult(lb_op, 0),
        ub,
        mlir.oraOperationGetResult(step_op, 0),
        &[_]mlir.MlirValue{
            mlir.oraOperationGetResult(init_a_op, 0),
            mlir.oraOperationGetResult(init_b_op, 0),
        },
        2,
        true,
    );
    const loop_body = mlir.oraScfForOpGetBodyBlock(loop);
    const carried_a = mlir.oraBlockGetArgument(loop_body, 1);
    const carried_b = mlir.oraBlockGetArgument(loop_body, 2);
    const next_a = mlir.oraArithAddIOpCreate(mlir_ctx, loc, carried_a, mlir.oraOperationGetResult(delta_a_op, 0));
    const next_b = mlir.oraArithAddIOpCreate(mlir_ctx, loc, carried_b, mlir.oraOperationGetResult(delta_b_op, 0));
    mlir.oraBlockAppendOwnedOperation(loop_body, next_a);
    mlir.oraBlockAppendOwnedOperation(loop_body, next_b);
    mlir.oraBlockAppendOwnedOperation(loop_body, mlir.oraScfYieldOpCreate(
        mlir_ctx,
        loc,
        &[_]mlir.MlirValue{
            mlir.oraOperationGetResult(next_a, 0),
            mlir.oraOperationGetResult(next_b, 0),
        },
        2,
    ));
    mlir.oraBlockAppendOwnedOperation(body, loop);
    mlir.oraBlockAppendOwnedOperation(body, mlir.oraReturnOpCreate(
        mlir_ctx,
        loc,
        &[_]mlir.MlirValue{mlir.oraOperationGetResult(loop, 1)},
        1,
    ));

    try encoder.registerFunctionOperation(helper);

    const caller_ub = mlir.oraVariablePlaceholderOpCreate(mlir_ctx, loc, stringRef("callerForMultiUb"), index_ty);
    const call = mlir.oraFuncCallOpCreate(
        mlir_ctx,
        loc,
        stringRef("symbolicForMultiReturn"),
        &[_]mlir.MlirValue{mlir.oraOperationGetResult(caller_ub, 0)},
        1,
        &[_]mlir.MlirType{i256_ty},
        1,
    );
    const encoded = try encoder.encodeOperation(call);
    try testing.expect(!encoder.isDegraded());

    const lb_ast = try encoder.encodeOperation(lb_op);
    const ub_ast = try encoder.encodeOperation(caller_ub);
    const step_ast = try encoder.encodeOperation(step_op);
    const init_b_ast = try encoder.encodeOperation(init_b_op);
    const delta_b_ast = try encoder.encodeOperation(delta_b_op);
    const sort = z3.Z3_get_sort(z3_ctx.ctx, ub_ast);
    const zero = z3.Z3_mk_unsigned_int64(z3_ctx.ctx, 0, sort);
    const one = z3.Z3_mk_unsigned_int64(z3_ctx.ctx, 1, sort);
    const ub_le_lb = z3.Z3_mk_bvule(z3_ctx.ctx, ub_ast, lb_ast);
    const distance = z3.Z3_mk_bv_sub(z3_ctx.ctx, ub_ast, lb_ast);
    const trip_count = z3.Z3_mk_ite(
        z3_ctx.ctx,
        ub_le_lb,
        zero,
        z3.Z3_mk_bv_add(
            z3_ctx.ctx,
            z3.Z3_mk_bv_udiv(z3_ctx.ctx, z3.Z3_mk_bv_sub(z3_ctx.ctx, distance, one), step_ast),
            one,
        ),
    );
    const total_delta = try encoder.encodeArithmeticOp(.Mul, trip_count, delta_b_ast);
    const expected = try encoder.encodeArithmeticOp(.Add, init_b_ast, total_delta);

    var solver = try Solver.init(&z3_ctx, testing.allocator);
    defer solver.deinit();
    solver.assert(z3.Z3_mk_not(z3_ctx.ctx, z3.Z3_mk_eq(z3_ctx.ctx, encoded, expected)));
    try testing.expectEqual(@as(z3.Z3_lbool, z3.Z3_L_FALSE), solver.check());
}

test "arith div emits safety obligation" {
    var z3_ctx = try Context.init(testing.allocator);
    defer z3_ctx.deinit();

    var encoder = Encoder.init(&z3_ctx, testing.allocator);
    defer encoder.deinit();

    const mlir_ctx = mlir.oraContextCreate();
    defer mlir.oraContextDestroy(mlir_ctx);
    loadAllDialects(mlir_ctx);
    _ = mlir.oraDialectRegister(mlir_ctx);

    const loc = mlir.oraLocationUnknownGet(mlir_ctx);
    const i256_ty = mlir.oraIntegerTypeCreate(mlir_ctx, 256);

    const lhs_attr = mlir.oraIntegerAttrCreateI64FromType(i256_ty, 6);
    const rhs_attr = mlir.oraIntegerAttrCreateI64FromType(i256_ty, 2);
    const lhs_const = mlir.oraArithConstantOpCreate(mlir_ctx, loc, i256_ty, lhs_attr);
    const rhs_const = mlir.oraArithConstantOpCreate(mlir_ctx, loc, i256_ty, rhs_attr);
    const lhs = mlir.oraOperationGetResult(lhs_const, 0);
    const rhs = mlir.oraOperationGetResult(rhs_const, 0);

    const div_op = mlir.oraArithDivUIOpCreate(mlir_ctx, loc, lhs, rhs);
    _ = try encoder.encodeOperation(div_op);

    const obligations = try encoder.takeObligations(testing.allocator);
    defer if (obligations.len > 0) testing.allocator.free(obligations);
    try testing.expect(obligations.len > 0);
}

test "ora.assert simplifies checked unsigned multiplication overflow pattern" {
    var z3_ctx = try Context.init(testing.allocator);
    defer z3_ctx.deinit();

    var encoder = Encoder.init(&z3_ctx, testing.allocator);
    defer encoder.deinit();

    const mlir_ctx = mlir.oraContextCreate();
    defer mlir.oraContextDestroy(mlir_ctx);
    loadAllDialects(mlir_ctx);
    _ = mlir.oraDialectRegister(mlir_ctx);

    const loc = mlir.oraLocationUnknownGet(mlir_ctx);
    const i1_ty = mlir.oraIntegerTypeCreate(mlir_ctx, 1);
    const i256_ty = mlir.oraIntegerTypeCreate(mlir_ctx, 256);

    const lhs_placeholder_op = mlir.oraVariablePlaceholderOpCreate(mlir_ctx, loc, stringRef("lhs"), i256_ty);
    const lhs = mlir.oraOperationGetResult(lhs_placeholder_op, 0);

    const rhs_attr = mlir.oraIntegerAttrCreateI64FromType(i256_ty, 10000);
    const rhs_const = mlir.oraArithConstantOpCreate(mlir_ctx, loc, i256_ty, rhs_attr);
    const rhs = mlir.oraOperationGetResult(rhs_const, 0);

    const zero_attr = mlir.oraIntegerAttrCreateI64FromType(i256_ty, 0);
    const zero_const = mlir.oraArithConstantOpCreate(mlir_ctx, loc, i256_ty, zero_attr);
    const zero = mlir.oraOperationGetResult(zero_const, 0);

    const true_attr = mlir.oraIntegerAttrCreateI64FromType(i1_ty, 1);
    const true_const = mlir.oraArithConstantOpCreate(mlir_ctx, loc, i1_ty, true_attr);
    const bool_true = mlir.oraOperationGetResult(true_const, 0);

    const mul_op = mlir.oraArithMulIOpCreate(mlir_ctx, loc, lhs, rhs);
    const mul = mlir.oraOperationGetResult(mul_op, 0);
    const rhs_non_zero_op = mlir.oraArithCmpIOpCreate(mlir_ctx, loc, 1, rhs, zero); // ne
    const rhs_non_zero = mlir.oraOperationGetResult(rhs_non_zero_op, 0);
    const div_op = mlir.oraArithDivUIOpCreate(mlir_ctx, loc, mul, rhs);
    mlir.oraOperationSetAttributeByName(div_op, stringRef("ora.guard_internal"), mlir.oraBoolAttrCreate(mlir_ctx, true));
    const recovered = mlir.oraOperationGetResult(div_op, 0);
    const overflow_cmp_op = mlir.oraArithCmpIOpCreate(mlir_ctx, loc, 1, recovered, lhs); // ne
    const overflow_cmp = mlir.oraOperationGetResult(overflow_cmp_op, 0);
    const and_op = mlir.oraArithAndIOpCreate(mlir_ctx, loc, overflow_cmp, rhs_non_zero);
    const overflow_flag = mlir.oraOperationGetResult(and_op, 0);
    const not_overflow_op = mlir.oraArithXorIOpCreate(mlir_ctx, loc, overflow_flag, bool_true);
    const not_overflow = mlir.oraOperationGetResult(not_overflow_op, 0);
    const assert_op = mlir.oraAssertOpCreate(mlir_ctx, loc, not_overflow, stringRef("checked multiplication overflow"));

    _ = try encoder.encodeOperation(assert_op);

    const obligations = try encoder.takeObligations(testing.allocator);
    defer if (obligations.len > 0) testing.allocator.free(obligations);
    try testing.expect(obligations.len >= 1);

    const lhs_ast = try encoder.encodeValue(lhs);
    const rhs_ast = try encoder.encodeValue(rhs);
    const expected = z3.Z3_mk_not(z3_ctx.ctx, encoder.checkMulOverflow(lhs_ast, rhs_ast));
    var found_expected = false;
    for (obligations) |obligation| {
        const obligation_text = std.mem.span(z3.Z3_ast_to_string(z3_ctx.ctx, obligation));
        if (std.mem.indexOf(u8, obligation_text, "(bvmul") != null) continue;

        var solver = try Solver.init(&z3_ctx, testing.allocator);
        defer solver.deinit();
        solver.assert(z3.Z3_mk_not(z3_ctx.ctx, z3.Z3_mk_eq(z3_ctx.ctx, obligation, expected)));
        if (solver.check() == z3.Z3_L_FALSE) {
            found_expected = true;
            break;
        }
    }
    try testing.expect(found_expected);
}

test "ora.assert simplifies checked addition overflow pattern" {
    var z3_ctx = try Context.init(testing.allocator);
    defer z3_ctx.deinit();

    var encoder = Encoder.init(&z3_ctx, testing.allocator);
    defer encoder.deinit();

    const mlir_ctx = mlir.oraContextCreate();
    defer mlir.oraContextDestroy(mlir_ctx);
    loadAllDialects(mlir_ctx);
    _ = mlir.oraDialectRegister(mlir_ctx);

    const loc = mlir.oraLocationUnknownGet(mlir_ctx);
    const i1_ty = mlir.oraIntegerTypeCreate(mlir_ctx, 1);
    const i256_ty = mlir.oraIntegerTypeCreate(mlir_ctx, 256);

    const lhs_placeholder_op = mlir.oraVariablePlaceholderOpCreate(mlir_ctx, loc, stringRef("lhs"), i256_ty);
    const lhs = mlir.oraOperationGetResult(lhs_placeholder_op, 0);
    const rhs_attr = mlir.oraIntegerAttrCreateI64FromType(i256_ty, 7);
    const rhs_const = mlir.oraArithConstantOpCreate(mlir_ctx, loc, i256_ty, rhs_attr);
    const rhs = mlir.oraOperationGetResult(rhs_const, 0);
    const true_attr = mlir.oraIntegerAttrCreateI64FromType(i1_ty, 1);
    const true_const = mlir.oraArithConstantOpCreate(mlir_ctx, loc, i1_ty, true_attr);
    const bool_true = mlir.oraOperationGetResult(true_const, 0);

    const add_op = mlir.oraArithAddIOpCreate(mlir_ctx, loc, lhs, rhs);
    const sum = mlir.oraOperationGetResult(add_op, 0);
    const overflow_cmp_op = mlir.oraArithCmpIOpCreate(mlir_ctx, loc, 6, sum, lhs); // ult
    const overflow_cmp = mlir.oraOperationGetResult(overflow_cmp_op, 0);
    const not_overflow_op = mlir.oraArithXorIOpCreate(mlir_ctx, loc, overflow_cmp, bool_true);
    const not_overflow = mlir.oraOperationGetResult(not_overflow_op, 0);
    const assert_op = mlir.oraAssertOpCreate(mlir_ctx, loc, not_overflow, stringRef("checked addition overflow"));

    _ = try encoder.encodeOperation(assert_op);

    const obligations = try encoder.takeObligations(testing.allocator);
    defer if (obligations.len > 0) testing.allocator.free(obligations);
    try testing.expect(obligations.len >= 1);

    const lhs_ast = try encoder.encodeValue(lhs);
    const rhs_ast = try encoder.encodeValue(rhs);
    const expected = z3.Z3_mk_not(z3_ctx.ctx, encoder.checkAddOverflow(lhs_ast, rhs_ast));
    var found_expected = false;
    for (obligations) |obligation| {
        var solver = try Solver.init(&z3_ctx, testing.allocator);
        defer solver.deinit();
        solver.assert(z3.Z3_mk_not(z3_ctx.ctx, z3.Z3_mk_eq(z3_ctx.ctx, obligation, expected)));
        if (solver.check() == z3.Z3_L_FALSE) {
            found_expected = true;
            break;
        }
    }
    try testing.expect(found_expected);
}

test "ora.assert simplifies checked subtraction overflow pattern" {
    var z3_ctx = try Context.init(testing.allocator);
    defer z3_ctx.deinit();

    var encoder = Encoder.init(&z3_ctx, testing.allocator);
    defer encoder.deinit();

    const mlir_ctx = mlir.oraContextCreate();
    defer mlir.oraContextDestroy(mlir_ctx);
    loadAllDialects(mlir_ctx);
    _ = mlir.oraDialectRegister(mlir_ctx);

    const loc = mlir.oraLocationUnknownGet(mlir_ctx);
    const i1_ty = mlir.oraIntegerTypeCreate(mlir_ctx, 1);
    const i256_ty = mlir.oraIntegerTypeCreate(mlir_ctx, 256);

    const lhs_placeholder_op = mlir.oraVariablePlaceholderOpCreate(mlir_ctx, loc, stringRef("lhs"), i256_ty);
    const lhs = mlir.oraOperationGetResult(lhs_placeholder_op, 0);
    const rhs_attr = mlir.oraIntegerAttrCreateI64FromType(i256_ty, 7);
    const rhs_const = mlir.oraArithConstantOpCreate(mlir_ctx, loc, i256_ty, rhs_attr);
    const rhs = mlir.oraOperationGetResult(rhs_const, 0);
    const true_attr = mlir.oraIntegerAttrCreateI64FromType(i1_ty, 1);
    const true_const = mlir.oraArithConstantOpCreate(mlir_ctx, loc, i1_ty, true_attr);
    const bool_true = mlir.oraOperationGetResult(true_const, 0);

    const underflow_cmp_op = mlir.oraArithCmpIOpCreate(mlir_ctx, loc, 6, lhs, rhs); // ult
    const underflow_cmp = mlir.oraOperationGetResult(underflow_cmp_op, 0);
    const not_underflow_op = mlir.oraArithXorIOpCreate(mlir_ctx, loc, underflow_cmp, bool_true);
    const not_underflow = mlir.oraOperationGetResult(not_underflow_op, 0);
    const assert_op = mlir.oraAssertOpCreate(mlir_ctx, loc, not_underflow, stringRef("checked subtraction overflow"));

    _ = try encoder.encodeOperation(assert_op);

    const obligations = try encoder.takeObligations(testing.allocator);
    defer if (obligations.len > 0) testing.allocator.free(obligations);
    try testing.expect(obligations.len >= 1);

    const lhs_ast = try encoder.encodeValue(lhs);
    const rhs_ast = try encoder.encodeValue(rhs);
    const expected = z3.Z3_mk_not(z3_ctx.ctx, encoder.checkSubUnderflow(lhs_ast, rhs_ast));
    var found_expected = false;
    for (obligations) |obligation| {
        var solver = try Solver.init(&z3_ctx, testing.allocator);
        defer solver.deinit();
        solver.assert(z3.Z3_mk_not(z3_ctx.ctx, z3.Z3_mk_eq(z3_ctx.ctx, obligation, expected)));
        if (solver.check() == z3.Z3_L_FALSE) {
            found_expected = true;
            break;
        }
    }
    try testing.expect(found_expected);
}

test "arith divsi encodes signed division" {
    var z3_ctx = try Context.init(testing.allocator);
    defer z3_ctx.deinit();

    var encoder = Encoder.init(&z3_ctx, testing.allocator);
    defer encoder.deinit();

    const mlir_ctx = mlir.oraContextCreate();
    defer mlir.oraContextDestroy(mlir_ctx);
    loadAllDialects(mlir_ctx);
    _ = mlir.oraDialectRegister(mlir_ctx);

    const loc = mlir.oraLocationUnknownGet(mlir_ctx);
    const i256_ty = mlir.oraIntegerTypeCreate(mlir_ctx, 256);

    const lhs_attr = mlir.oraIntegerAttrCreateI64FromType(i256_ty, -10);
    const rhs_attr = mlir.oraIntegerAttrCreateI64FromType(i256_ty, 3);
    const lhs_const = mlir.oraArithConstantOpCreate(mlir_ctx, loc, i256_ty, lhs_attr);
    const rhs_const = mlir.oraArithConstantOpCreate(mlir_ctx, loc, i256_ty, rhs_attr);
    const lhs = mlir.oraOperationGetResult(lhs_const, 0);
    const rhs = mlir.oraOperationGetResult(rhs_const, 0);

    const div_op = mlir.oraArithDivSIOpCreate(mlir_ctx, loc, lhs, rhs);
    const ast = try encoder.encodeOperation(div_op);
    const ast_str = std.mem.span(z3.Z3_ast_to_string(z3_ctx.ctx, ast));
    try testing.expect(std.mem.indexOf(u8, ast_str, "bvsdiv") != null);
}

test "arith divui encodes unsigned division" {
    var z3_ctx = try Context.init(testing.allocator);
    defer z3_ctx.deinit();

    var encoder = Encoder.init(&z3_ctx, testing.allocator);
    defer encoder.deinit();

    const mlir_ctx = mlir.oraContextCreate();
    defer mlir.oraContextDestroy(mlir_ctx);
    loadAllDialects(mlir_ctx);
    _ = mlir.oraDialectRegister(mlir_ctx);

    const loc = mlir.oraLocationUnknownGet(mlir_ctx);
    const i256_ty = mlir.oraIntegerTypeCreate(mlir_ctx, 256);

    const lhs_attr = mlir.oraIntegerAttrCreateI64FromType(i256_ty, 10);
    const rhs_attr = mlir.oraIntegerAttrCreateI64FromType(i256_ty, 3);
    const lhs_const = mlir.oraArithConstantOpCreate(mlir_ctx, loc, i256_ty, lhs_attr);
    const rhs_const = mlir.oraArithConstantOpCreate(mlir_ctx, loc, i256_ty, rhs_attr);
    const lhs = mlir.oraOperationGetResult(lhs_const, 0);
    const rhs = mlir.oraOperationGetResult(rhs_const, 0);

    const div_op = mlir.oraArithDivUIOpCreate(mlir_ctx, loc, lhs, rhs);
    const ast = try encoder.encodeOperation(div_op);
    const ast_str = std.mem.span(z3.Z3_ast_to_string(z3_ctx.ctx, ast));
    try testing.expect(std.mem.indexOf(u8, ast_str, "bvudiv") != null);
}

test "ora.power encodes modular exponentiation" {
    var z3_ctx = try Context.init(testing.allocator);
    defer z3_ctx.deinit();

    var encoder = Encoder.init(&z3_ctx, testing.allocator);
    defer encoder.deinit();

    const mlir_ctx = mlir.oraContextCreate();
    defer mlir.oraContextDestroy(mlir_ctx);
    loadAllDialects(mlir_ctx);
    _ = mlir.oraDialectRegister(mlir_ctx);

    const loc = mlir.oraLocationUnknownGet(mlir_ctx);
    const i256_ty = mlir.oraIntegerTypeCreate(mlir_ctx, 256);

    const base_attr = mlir.oraIntegerAttrCreateI64FromType(i256_ty, 2);
    const exp10_attr = mlir.oraIntegerAttrCreateI64FromType(i256_ty, 10);
    const exp256_attr = mlir.oraIntegerAttrCreateI64FromType(i256_ty, 256);

    const base_op = mlir.oraArithConstantOpCreate(mlir_ctx, loc, i256_ty, base_attr);
    const exp10_op = mlir.oraArithConstantOpCreate(mlir_ctx, loc, i256_ty, exp10_attr);
    const exp256_op = mlir.oraArithConstantOpCreate(mlir_ctx, loc, i256_ty, exp256_attr);

    const base = mlir.oraOperationGetResult(base_op, 0);
    const exp10 = mlir.oraOperationGetResult(exp10_op, 0);
    const exp256 = mlir.oraOperationGetResult(exp256_op, 0);

    const pow10_op = mlir.oraPowerOpCreate(mlir_ctx, loc, base, exp10, i256_ty);
    const pow10 = try encoder.encodeOperation(pow10_op);
    const expected_1024 = try encoder.encodeIntegerConstant(1024, 256);

    var solver_pow10 = try Solver.init(&z3_ctx, testing.allocator);
    defer solver_pow10.deinit();
    solver_pow10.assert(z3.Z3_mk_not(z3_ctx.ctx, z3.Z3_mk_eq(z3_ctx.ctx, pow10, expected_1024)));
    try testing.expectEqual(@as(z3.Z3_lbool, z3.Z3_L_FALSE), solver_pow10.check());

    const pow256_op = mlir.oraPowerOpCreate(mlir_ctx, loc, base, exp256, i256_ty);
    const pow256 = try encoder.encodeOperation(pow256_op);
    const expected_zero = try encoder.encodeIntegerConstant(0, 256);

    var solver_pow256 = try Solver.init(&z3_ctx, testing.allocator);
    defer solver_pow256.deinit();
    solver_pow256.assert(z3.Z3_mk_not(z3_ctx.ctx, z3.Z3_mk_eq(z3_ctx.ctx, pow256, expected_zero)));
    try testing.expectEqual(@as(z3.Z3_lbool, z3.Z3_L_FALSE), solver_pow256.check());
}

test "arith.constant -1 is encoded as all-ones at target bit width" {
    var z3_ctx = try Context.init(testing.allocator);
    defer z3_ctx.deinit();

    var encoder = Encoder.init(&z3_ctx, testing.allocator);
    defer encoder.deinit();

    const mlir_ctx = mlir.oraContextCreate();
    defer mlir.oraContextDestroy(mlir_ctx);
    loadAllDialects(mlir_ctx);
    _ = mlir.oraDialectRegister(mlir_ctx);

    const loc = mlir.oraLocationUnknownGet(mlir_ctx);
    const i256_ty = mlir.oraIntegerTypeCreate(mlir_ctx, 256);
    const minus_one_attr = mlir.oraIntegerAttrCreateI64FromType(i256_ty, -1);
    const minus_one_op = mlir.oraArithConstantOpCreate(mlir_ctx, loc, i256_ty, minus_one_attr);

    const encoded = try encoder.encodeOperation(minus_one_op);
    const expected = try encoder.encodeIntegerConstant(std.math.maxInt(u256), 256);

    var solver = try Solver.init(&z3_ctx, testing.allocator);
    defer solver.deinit();
    const neq = z3.Z3_mk_not(z3_ctx.ctx, z3.Z3_mk_eq(z3_ctx.ctx, encoded, expected));
    solver.assert(neq);
    try testing.expectEqual(@as(z3.Z3_lbool, z3.Z3_L_FALSE), solver.check());
}

test "arith.constant preserves values wider than 64 bits" {
    var z3_ctx = try Context.init(testing.allocator);
    defer z3_ctx.deinit();

    var encoder = Encoder.init(&z3_ctx, testing.allocator);
    defer encoder.deinit();

    const mlir_ctx = mlir.oraContextCreate();
    defer mlir.oraContextDestroy(mlir_ctx);
    loadAllDialects(mlir_ctx);
    _ = mlir.oraDialectRegister(mlir_ctx);

    const loc = mlir.oraLocationUnknownGet(mlir_ctx);
    const i256_ty = mlir.oraIntegerTypeCreate(mlir_ctx, 256);
    const big_attr = mlir.oraIntegerAttrGetFromString(i256_ty, stringRef("1208925819614629174706181")); // 2^80 + 5
    const big_op = mlir.oraArithConstantOpCreate(mlir_ctx, loc, i256_ty, big_attr);

    const encoded = try encoder.encodeOperation(big_op);
    const expected_value: u256 = (@as(u256, 1) << 80) + 5;
    const expected = try encoder.encodeIntegerConstant(expected_value, 256);

    var solver = try Solver.init(&z3_ctx, testing.allocator);
    defer solver.deinit();
    const neq = z3.Z3_mk_not(z3_ctx.ctx, z3.Z3_mk_eq(z3_ctx.ctx, encoded, expected));
    solver.assert(neq);
    try testing.expectEqual(@as(z3.Z3_lbool, z3.Z3_L_FALSE), solver.check());
}

test "arith remsi encodes signed remainder" {
    var z3_ctx = try Context.init(testing.allocator);
    defer z3_ctx.deinit();

    var encoder = Encoder.init(&z3_ctx, testing.allocator);
    defer encoder.deinit();

    const mlir_ctx = mlir.oraContextCreate();
    defer mlir.oraContextDestroy(mlir_ctx);
    loadAllDialects(mlir_ctx);
    _ = mlir.oraDialectRegister(mlir_ctx);

    const loc = mlir.oraLocationUnknownGet(mlir_ctx);
    const i256_ty = mlir.oraIntegerTypeCreate(mlir_ctx, 256);

    const lhs_attr = mlir.oraIntegerAttrCreateI64FromType(i256_ty, -10);
    const rhs_attr = mlir.oraIntegerAttrCreateI64FromType(i256_ty, 3);
    const lhs_const = mlir.oraArithConstantOpCreate(mlir_ctx, loc, i256_ty, lhs_attr);
    const rhs_const = mlir.oraArithConstantOpCreate(mlir_ctx, loc, i256_ty, rhs_attr);
    const lhs = mlir.oraOperationGetResult(lhs_const, 0);
    const rhs = mlir.oraOperationGetResult(rhs_const, 0);

    const rem_op = mlir.oraArithRemSIOpCreate(mlir_ctx, loc, lhs, rhs);
    const ast = try encoder.encodeOperation(rem_op);
    const ast_str = std.mem.span(z3.Z3_ast_to_string(z3_ctx.ctx, ast));
    try testing.expect(std.mem.indexOf(u8, ast_str, "bvsrem") != null);
}

test "arith remui encodes unsigned remainder" {
    var z3_ctx = try Context.init(testing.allocator);
    defer z3_ctx.deinit();

    var encoder = Encoder.init(&z3_ctx, testing.allocator);
    defer encoder.deinit();

    const mlir_ctx = mlir.oraContextCreate();
    defer mlir.oraContextDestroy(mlir_ctx);
    loadAllDialects(mlir_ctx);
    _ = mlir.oraDialectRegister(mlir_ctx);

    const loc = mlir.oraLocationUnknownGet(mlir_ctx);
    const i256_ty = mlir.oraIntegerTypeCreate(mlir_ctx, 256);

    const lhs_attr = mlir.oraIntegerAttrCreateI64FromType(i256_ty, 10);
    const rhs_attr = mlir.oraIntegerAttrCreateI64FromType(i256_ty, 3);
    const lhs_const = mlir.oraArithConstantOpCreate(mlir_ctx, loc, i256_ty, lhs_attr);
    const rhs_const = mlir.oraArithConstantOpCreate(mlir_ctx, loc, i256_ty, rhs_attr);
    const lhs = mlir.oraOperationGetResult(lhs_const, 0);
    const rhs = mlir.oraOperationGetResult(rhs_const, 0);

    const rem_op = mlir.oraArithRemUIOpCreate(mlir_ctx, loc, lhs, rhs);
    const ast = try encoder.encodeOperation(rem_op);
    const ast_str = std.mem.span(z3.Z3_ast_to_string(z3_ctx.ctx, ast));
    try testing.expect(std.mem.indexOf(u8, ast_str, "bvurem") != null);
}

test "arith cmpi ult encodes unsigned comparison" {
    var z3_ctx = try Context.init(testing.allocator);
    defer z3_ctx.deinit();

    var encoder = Encoder.init(&z3_ctx, testing.allocator);
    defer encoder.deinit();

    const mlir_ctx = mlir.oraContextCreate();
    defer mlir.oraContextDestroy(mlir_ctx);
    loadAllDialects(mlir_ctx);
    _ = mlir.oraDialectRegister(mlir_ctx);

    const loc = mlir.oraLocationUnknownGet(mlir_ctx);
    const i256_ty = mlir.oraIntegerTypeCreate(mlir_ctx, 256);

    const lhs_attr = mlir.oraIntegerAttrCreateI64FromType(i256_ty, 1);
    const rhs_attr = mlir.oraIntegerAttrCreateI64FromType(i256_ty, 2);
    const lhs_const = mlir.oraArithConstantOpCreate(mlir_ctx, loc, i256_ty, lhs_attr);
    const rhs_const = mlir.oraArithConstantOpCreate(mlir_ctx, loc, i256_ty, rhs_attr);
    const lhs = mlir.oraOperationGetResult(lhs_const, 0);
    const rhs = mlir.oraOperationGetResult(rhs_const, 0);

    const cmp_op = mlir.oraArithCmpIOpCreate(mlir_ctx, loc, 6, lhs, rhs); // ult
    const ast = try encoder.encodeOperation(cmp_op);
    const ast_str = std.mem.span(z3.Z3_ast_to_string(z3_ctx.ctx, ast));
    try testing.expect(std.mem.indexOf(u8, ast_str, "bvult") != null);
}

test "arith cmpi slt encodes signed comparison" {
    var z3_ctx = try Context.init(testing.allocator);
    defer z3_ctx.deinit();

    var encoder = Encoder.init(&z3_ctx, testing.allocator);
    defer encoder.deinit();

    const mlir_ctx = mlir.oraContextCreate();
    defer mlir.oraContextDestroy(mlir_ctx);
    loadAllDialects(mlir_ctx);
    _ = mlir.oraDialectRegister(mlir_ctx);

    const loc = mlir.oraLocationUnknownGet(mlir_ctx);
    const i256_ty = mlir.oraIntegerTypeCreate(mlir_ctx, 256);

    const lhs_attr = mlir.oraIntegerAttrCreateI64FromType(i256_ty, -1);
    const rhs_attr = mlir.oraIntegerAttrCreateI64FromType(i256_ty, 1);
    const lhs_const = mlir.oraArithConstantOpCreate(mlir_ctx, loc, i256_ty, lhs_attr);
    const rhs_const = mlir.oraArithConstantOpCreate(mlir_ctx, loc, i256_ty, rhs_attr);
    const lhs = mlir.oraOperationGetResult(lhs_const, 0);
    const rhs = mlir.oraOperationGetResult(rhs_const, 0);

    const cmp_op = mlir.oraArithCmpIOpCreate(mlir_ctx, loc, 2, lhs, rhs); // slt
    const ast = try encoder.encodeOperation(cmp_op);
    const ast_str = std.mem.span(z3.Z3_ast_to_string(z3_ctx.ctx, ast));
    try testing.expect(std.mem.indexOf(u8, ast_str, "bvslt") != null);
}

test "arith shrui encodes logical right shift" {
    var z3_ctx = try Context.init(testing.allocator);
    defer z3_ctx.deinit();

    var encoder = Encoder.init(&z3_ctx, testing.allocator);
    defer encoder.deinit();

    const mlir_ctx = mlir.oraContextCreate();
    defer mlir.oraContextDestroy(mlir_ctx);
    loadAllDialects(mlir_ctx);
    _ = mlir.oraDialectRegister(mlir_ctx);

    const loc = mlir.oraLocationUnknownGet(mlir_ctx);
    const i256_ty = mlir.oraIntegerTypeCreate(mlir_ctx, 256);

    const lhs_attr = mlir.oraIntegerAttrCreateI64FromType(i256_ty, 8);
    const rhs_attr = mlir.oraIntegerAttrCreateI64FromType(i256_ty, 1);
    const lhs_const = mlir.oraArithConstantOpCreate(mlir_ctx, loc, i256_ty, lhs_attr);
    const rhs_const = mlir.oraArithConstantOpCreate(mlir_ctx, loc, i256_ty, rhs_attr);
    const lhs = mlir.oraOperationGetResult(lhs_const, 0);
    const rhs = mlir.oraOperationGetResult(rhs_const, 0);

    const shr_op = mlir.oraArithShrUIOpCreate(mlir_ctx, loc, lhs, rhs);
    const ast = try encoder.encodeOperation(shr_op);
    const ast_str = std.mem.span(z3.Z3_ast_to_string(z3_ctx.ctx, ast));
    try testing.expect(std.mem.indexOf(u8, ast_str, "bvlshr") != null);
}

test "arith shrsi encodes arithmetic right shift" {
    var z3_ctx = try Context.init(testing.allocator);
    defer z3_ctx.deinit();

    var encoder = Encoder.init(&z3_ctx, testing.allocator);
    defer encoder.deinit();

    const mlir_ctx = mlir.oraContextCreate();
    defer mlir.oraContextDestroy(mlir_ctx);
    loadAllDialects(mlir_ctx);
    _ = mlir.oraDialectRegister(mlir_ctx);

    const loc = mlir.oraLocationUnknownGet(mlir_ctx);
    const i256_ty = mlir.oraIntegerTypeCreate(mlir_ctx, 256);

    const lhs_attr = mlir.oraIntegerAttrCreateI64FromType(i256_ty, -8);
    const rhs_attr = mlir.oraIntegerAttrCreateI64FromType(i256_ty, 1);
    const lhs_const = mlir.oraArithConstantOpCreate(mlir_ctx, loc, i256_ty, lhs_attr);
    const rhs_const = mlir.oraArithConstantOpCreate(mlir_ctx, loc, i256_ty, rhs_attr);
    const lhs = mlir.oraOperationGetResult(lhs_const, 0);
    const rhs = mlir.oraOperationGetResult(rhs_const, 0);

    const shr_op = mlir.oraArithShrSIOpCreate(mlir_ctx, loc, lhs, rhs);
    const ast = try encoder.encodeOperation(shr_op);
    const ast_str = std.mem.span(z3.Z3_ast_to_string(z3_ctx.ctx, ast));
    try testing.expect(std.mem.indexOf(u8, ast_str, "bvashr") != null);
}

test "storage store threads into later load" {
    var z3_ctx = try Context.init(testing.allocator);
    defer z3_ctx.deinit();

    var encoder = Encoder.init(&z3_ctx, testing.allocator);
    defer encoder.deinit();

    const mlir_ctx = mlir.oraContextCreate();
    defer mlir.oraContextDestroy(mlir_ctx);
    loadAllDialects(mlir_ctx);
    _ = mlir.oraDialectRegister(mlir_ctx);

    const loc = mlir.oraLocationUnknownGet(mlir_ctx);
    const i256_ty = mlir.oraIntegerTypeCreate(mlir_ctx, 256);
    const value_attr = mlir.oraIntegerAttrCreateI64FromType(i256_ty, 42);
    const value_op = mlir.oraArithConstantOpCreate(mlir_ctx, loc, i256_ty, value_attr);
    const value = mlir.oraOperationGetResult(value_op, 0);

    const sstore = mlir.oraSStoreOpCreate(mlir_ctx, loc, value, stringRef("counter"));
    const stored_ast = try encoder.encodeOperation(sstore);

    const sload = mlir.oraSLoadOpCreate(mlir_ctx, loc, stringRef("counter"), i256_ty);
    const loaded_ast = try encoder.encodeOperation(sload);

    const stored_raw = z3.Z3_ast_to_string(z3_ctx.ctx, stored_ast);
    const stored_str = try testing.allocator.dupe(u8, std.mem.span(stored_raw));
    defer testing.allocator.free(stored_str);
    const loaded_raw = z3.Z3_ast_to_string(z3_ctx.ctx, loaded_ast);
    const loaded_str = try testing.allocator.dupe(u8, std.mem.span(loaded_raw));
    defer testing.allocator.free(loaded_str);
    try testing.expect(std.mem.eql(u8, stored_str, loaded_str));
}

test "func.call encoding is deterministic for identical signatures" {
    var z3_ctx = try Context.init(testing.allocator);
    defer z3_ctx.deinit();

    var encoder = Encoder.init(&z3_ctx, testing.allocator);
    defer encoder.deinit();

    const mlir_ctx = mlir.oraContextCreate();
    defer mlir.oraContextDestroy(mlir_ctx);
    loadAllDialects(mlir_ctx);
    _ = mlir.oraDialectRegister(mlir_ctx);

    const loc = mlir.oraLocationUnknownGet(mlir_ctx);
    const i256_ty = mlir.oraIntegerTypeCreate(mlir_ctx, 256);
    const arg_attr = mlir.oraIntegerAttrCreateI64FromType(i256_ty, 7);
    const arg_op = mlir.oraArithConstantOpCreate(mlir_ctx, loc, i256_ty, arg_attr);
    const arg = mlir.oraOperationGetResult(arg_op, 0);

    const operands = [_]mlir.MlirValue{arg};
    const results = [_]mlir.MlirType{i256_ty};
    const call_a = mlir.oraFuncCallOpCreate(mlir_ctx, loc, stringRef("foo"), &operands, operands.len, &results, results.len);
    const call_b = mlir.oraFuncCallOpCreate(mlir_ctx, loc, stringRef("foo"), &operands, operands.len, &results, results.len);

    const ast_a = try encoder.encodeOperation(call_a);
    const ast_b = try encoder.encodeOperation(call_b);
    var solver = try Solver.init(&z3_ctx, testing.allocator);
    defer solver.deinit();
    const neq = z3.Z3_mk_not(z3_ctx.ctx, z3.Z3_mk_eq(z3_ctx.ctx, ast_a, ast_b));
    solver.assert(neq);
    try testing.expectEqual(@as(z3.Z3_lbool, z3.Z3_L_FALSE), solver.check());
}

test "func.call multi-result encoding is deterministic per result index" {
    var z3_ctx = try Context.init(testing.allocator);
    defer z3_ctx.deinit();

    var encoder = Encoder.init(&z3_ctx, testing.allocator);
    defer encoder.deinit();

    const mlir_ctx = mlir.oraContextCreate();
    defer mlir.oraContextDestroy(mlir_ctx);
    loadAllDialects(mlir_ctx);
    _ = mlir.oraDialectRegister(mlir_ctx);

    const loc = mlir.oraLocationUnknownGet(mlir_ctx);
    const i256_ty = mlir.oraIntegerTypeCreate(mlir_ctx, 256);
    const arg_attr = mlir.oraIntegerAttrCreateI64FromType(i256_ty, 7);
    const arg_op = mlir.oraArithConstantOpCreate(mlir_ctx, loc, i256_ty, arg_attr);
    const arg = mlir.oraOperationGetResult(arg_op, 0);

    const operands = [_]mlir.MlirValue{arg};
    const results = [_]mlir.MlirType{ i256_ty, i256_ty };
    const call_a = mlir.oraFuncCallOpCreate(mlir_ctx, loc, stringRef("foo"), &operands, operands.len, &results, results.len);
    const call_b = mlir.oraFuncCallOpCreate(mlir_ctx, loc, stringRef("foo"), &operands, operands.len, &results, results.len);

    const a1 = try encoder.encodeValue(mlir.oraOperationGetResult(call_a, 1));
    const b1 = try encoder.encodeValue(mlir.oraOperationGetResult(call_b, 1));

    var solver = try Solver.init(&z3_ctx, testing.allocator);
    defer solver.deinit();
    const neq = z3.Z3_mk_not(z3_ctx.ctx, z3.Z3_mk_eq(z3_ctx.ctx, a1, b1));
    solver.assert(neq);
    try testing.expectEqual(@as(z3.Z3_lbool, z3.Z3_L_FALSE), solver.check());
}

test "func.call inlines simple callee return expression" {
    var z3_ctx = try Context.init(testing.allocator);
    defer z3_ctx.deinit();

    var encoder = Encoder.init(&z3_ctx, testing.allocator);
    defer encoder.deinit();

    const mlir_ctx = mlir.oraContextCreate();
    defer mlir.oraContextDestroy(mlir_ctx);
    loadAllDialects(mlir_ctx);
    _ = mlir.oraDialectRegister(mlir_ctx);

    const loc = mlir.oraLocationUnknownGet(mlir_ctx);
    const i256_ty = mlir.oraIntegerTypeCreate(mlir_ctx, 256);

    const sym_name_attr = mlir.oraStringAttrCreate(mlir_ctx, stringRef("inc"));
    const func_attrs = [_]mlir.MlirNamedAttribute{
        namedAttr(mlir_ctx, "sym_name", sym_name_attr),
    };
    const param_types = [_]mlir.MlirType{i256_ty};
    const param_locs = [_]mlir.MlirLocation{loc};
    const func_op = mlir.oraFuncFuncOpCreate(mlir_ctx, loc, &func_attrs, func_attrs.len, &param_types, &param_locs, param_types.len);
    const body = mlir.oraFuncOpGetBodyBlock(func_op);
    const arg0 = mlir.oraBlockGetArgument(body, 0);

    const one_attr = mlir.oraIntegerAttrCreateI64FromType(i256_ty, 1);
    const one_op = mlir.oraArithConstantOpCreate(mlir_ctx, loc, i256_ty, one_attr);
    const one = mlir.oraOperationGetResult(one_op, 0);
    const add_op = mlir.oraArithAddIOpCreate(mlir_ctx, loc, arg0, one);
    const add_res = mlir.oraOperationGetResult(add_op, 0);
    const ret_vals = [_]mlir.MlirValue{add_res};
    const ret_op = mlir.oraReturnOpCreate(mlir_ctx, loc, &ret_vals, ret_vals.len);
    mlir.oraBlockAppendOwnedOperation(body, one_op);
    mlir.oraBlockAppendOwnedOperation(body, add_op);
    mlir.oraBlockAppendOwnedOperation(body, ret_op);

    try encoder.registerFunctionOperation(func_op);

    const arg_attr = mlir.oraIntegerAttrCreateI64FromType(i256_ty, 41);
    const arg_op = mlir.oraArithConstantOpCreate(mlir_ctx, loc, i256_ty, arg_attr);
    const arg = mlir.oraOperationGetResult(arg_op, 0);
    const call_operands = [_]mlir.MlirValue{arg};
    const call_results = [_]mlir.MlirType{i256_ty};
    const call_op = mlir.oraFuncCallOpCreate(mlir_ctx, loc, stringRef("inc"), &call_operands, call_operands.len, &call_results, call_results.len);

    const encoded = try encoder.encodeOperation(call_op);
    const expected = try encoder.encodeIntegerConstant(42, 256);

    var solver = try Solver.init(&z3_ctx, testing.allocator);
    defer solver.deinit();
    solver.assert(z3.Z3_mk_not(z3_ctx.ctx, z3.Z3_mk_eq(z3_ctx.ctx, encoded, expected)));
    try testing.expectEqual(@as(z3.Z3_lbool, z3.Z3_L_FALSE), solver.check());
}

test "func.call with write slots updates storage state" {
    var z3_ctx = try Context.init(testing.allocator);
    defer z3_ctx.deinit();

    var encoder = Encoder.init(&z3_ctx, testing.allocator);
    defer encoder.deinit();

    const mlir_ctx = mlir.oraContextCreate();
    defer mlir.oraContextDestroy(mlir_ctx);
    loadAllDialects(mlir_ctx);
    _ = mlir.oraDialectRegister(mlir_ctx);

    const loc = mlir.oraLocationUnknownGet(mlir_ctx);
    const i256_ty = mlir.oraIntegerTypeCreate(mlir_ctx, 256);

    const sym_name_attr = mlir.oraStringAttrCreate(mlir_ctx, stringRef("setCounter"));
    const effect_attr = mlir.oraStringAttrCreate(mlir_ctx, stringRef("writes"));
    const slot_attr = mlir.oraStringAttrCreate(mlir_ctx, stringRef("counter"));
    const slot_array = [_]mlir.MlirAttribute{slot_attr};
    const write_slots_attr = mlir.oraArrayAttrCreate(mlir_ctx, slot_array.len, &slot_array);
    const func_attrs = [_]mlir.MlirNamedAttribute{
        namedAttr(mlir_ctx, "sym_name", sym_name_attr),
        namedAttr(mlir_ctx, "ora.effect", effect_attr),
        namedAttr(mlir_ctx, "ora.write_slots", write_slots_attr),
    };
    const param_types = [_]mlir.MlirType{i256_ty};
    const param_locs = [_]mlir.MlirLocation{loc};
    const func_op = mlir.oraFuncFuncOpCreate(mlir_ctx, loc, &func_attrs, func_attrs.len, &param_types, &param_locs, param_types.len);
    const body = mlir.oraFuncOpGetBodyBlock(func_op);
    const arg0 = mlir.oraBlockGetArgument(body, 0);
    const sstore = mlir.oraSStoreOpCreate(mlir_ctx, loc, arg0, stringRef("counter"));
    const ret_vals = [_]mlir.MlirValue{arg0};
    const ret_op = mlir.oraReturnOpCreate(mlir_ctx, loc, &ret_vals, ret_vals.len);
    mlir.oraBlockAppendOwnedOperation(body, sstore);
    mlir.oraBlockAppendOwnedOperation(body, ret_op);

    try encoder.registerFunctionOperation(func_op);

    const arg_attr = mlir.oraIntegerAttrCreateI64FromType(i256_ty, 77);
    const arg_op = mlir.oraArithConstantOpCreate(mlir_ctx, loc, i256_ty, arg_attr);
    const arg = mlir.oraOperationGetResult(arg_op, 0);
    const call_operands = [_]mlir.MlirValue{arg};
    const call_results = [_]mlir.MlirType{i256_ty};
    const call_op = mlir.oraFuncCallOpCreate(mlir_ctx, loc, stringRef("setCounter"), &call_operands, call_operands.len, &call_results, call_results.len);
    _ = try encoder.encodeOperation(call_op);

    const sload_after = mlir.oraSLoadOpCreate(mlir_ctx, loc, stringRef("counter"), i256_ty);
    const loaded = try encoder.encodeOperation(sload_after);
    const expected = try encoder.encodeValue(arg);

    var solver = try Solver.init(&z3_ctx, testing.allocator);
    defer solver.deinit();
    solver.assert(z3.Z3_mk_not(z3_ctx.ctx, z3.Z3_mk_eq(z3_ctx.ctx, loaded, expected)));
    try testing.expectEqual(@as(z3.Z3_lbool, z3.Z3_L_FALSE), solver.check());
}

test "func.call composes nested callee summaries" {
    var z3_ctx = try Context.init(testing.allocator);
    defer z3_ctx.deinit();

    var encoder = Encoder.init(&z3_ctx, testing.allocator);
    defer encoder.deinit();

    const mlir_ctx = mlir.oraContextCreate();
    defer mlir.oraContextDestroy(mlir_ctx);
    loadAllDialects(mlir_ctx);
    _ = mlir.oraDialectRegister(mlir_ctx);

    const loc = mlir.oraLocationUnknownGet(mlir_ctx);
    const i256_ty = mlir.oraIntegerTypeCreate(mlir_ctx, 256);

    // inc(x) { return x + 1; }
    const inc_name_attr = mlir.oraStringAttrCreate(mlir_ctx, stringRef("inc"));
    const inc_attrs = [_]mlir.MlirNamedAttribute{
        namedAttr(mlir_ctx, "sym_name", inc_name_attr),
    };
    const inc_param_types = [_]mlir.MlirType{i256_ty};
    const inc_param_locs = [_]mlir.MlirLocation{loc};
    const inc_func = mlir.oraFuncFuncOpCreate(mlir_ctx, loc, &inc_attrs, inc_attrs.len, &inc_param_types, &inc_param_locs, inc_param_types.len);
    const inc_body = mlir.oraFuncOpGetBodyBlock(inc_func);
    const inc_arg0 = mlir.oraBlockGetArgument(inc_body, 0);
    const one_attr = mlir.oraIntegerAttrCreateI64FromType(i256_ty, 1);
    const one_op = mlir.oraArithConstantOpCreate(mlir_ctx, loc, i256_ty, one_attr);
    const one = mlir.oraOperationGetResult(one_op, 0);
    const add_op = mlir.oraArithAddIOpCreate(mlir_ctx, loc, inc_arg0, one);
    const add_res = mlir.oraOperationGetResult(add_op, 0);
    const inc_ret_vals = [_]mlir.MlirValue{add_res};
    const inc_ret = mlir.oraReturnOpCreate(mlir_ctx, loc, &inc_ret_vals, inc_ret_vals.len);
    mlir.oraBlockAppendOwnedOperation(inc_body, one_op);
    mlir.oraBlockAppendOwnedOperation(inc_body, add_op);
    mlir.oraBlockAppendOwnedOperation(inc_body, inc_ret);

    // add2(y) { return inc(inc(y)); }
    const add2_name_attr = mlir.oraStringAttrCreate(mlir_ctx, stringRef("add2"));
    const add2_attrs = [_]mlir.MlirNamedAttribute{
        namedAttr(mlir_ctx, "sym_name", add2_name_attr),
    };
    const add2_param_types = [_]mlir.MlirType{i256_ty};
    const add2_param_locs = [_]mlir.MlirLocation{loc};
    const add2_func = mlir.oraFuncFuncOpCreate(mlir_ctx, loc, &add2_attrs, add2_attrs.len, &add2_param_types, &add2_param_locs, add2_param_types.len);
    const add2_body = mlir.oraFuncOpGetBodyBlock(add2_func);
    const add2_arg0 = mlir.oraBlockGetArgument(add2_body, 0);

    const call_result_types = [_]mlir.MlirType{i256_ty};
    const call1_operands = [_]mlir.MlirValue{add2_arg0};
    const call1 = mlir.oraFuncCallOpCreate(
        mlir_ctx,
        loc,
        stringRef("inc"),
        &call1_operands,
        call1_operands.len,
        &call_result_types,
        call_result_types.len,
    );
    const call1_res = mlir.oraOperationGetResult(call1, 0);
    const call2_operands = [_]mlir.MlirValue{call1_res};
    const call2 = mlir.oraFuncCallOpCreate(
        mlir_ctx,
        loc,
        stringRef("inc"),
        &call2_operands,
        call2_operands.len,
        &call_result_types,
        call_result_types.len,
    );
    const call2_res = mlir.oraOperationGetResult(call2, 0);
    const add2_ret_vals = [_]mlir.MlirValue{call2_res};
    const add2_ret = mlir.oraReturnOpCreate(mlir_ctx, loc, &add2_ret_vals, add2_ret_vals.len);
    mlir.oraBlockAppendOwnedOperation(add2_body, call1);
    mlir.oraBlockAppendOwnedOperation(add2_body, call2);
    mlir.oraBlockAppendOwnedOperation(add2_body, add2_ret);

    try encoder.registerFunctionOperation(inc_func);
    try encoder.registerFunctionOperation(add2_func);

    const forty_attr = mlir.oraIntegerAttrCreateI64FromType(i256_ty, 40);
    const forty_op = mlir.oraArithConstantOpCreate(mlir_ctx, loc, i256_ty, forty_attr);
    const forty = mlir.oraOperationGetResult(forty_op, 0);
    const outer_operands = [_]mlir.MlirValue{forty};
    const outer_results = [_]mlir.MlirType{i256_ty};
    const outer_call = mlir.oraFuncCallOpCreate(
        mlir_ctx,
        loc,
        stringRef("add2"),
        &outer_operands,
        outer_operands.len,
        &outer_results,
        outer_results.len,
    );

    const encoded = try encoder.encodeOperation(outer_call);
    const expected = try encoder.encodeIntegerConstant(42, 256);

    var solver = try Solver.init(&z3_ctx, testing.allocator);
    defer solver.deinit();
    solver.assert(z3.Z3_mk_not(z3_ctx.ctx, z3.Z3_mk_eq(z3_ctx.ctx, encoded, expected)));
    try testing.expectEqual(@as(z3.Z3_lbool, z3.Z3_L_FALSE), solver.check());
}

test "func.call composes nested state updates from unused inner results" {
    var z3_ctx = try Context.init(testing.allocator);
    defer z3_ctx.deinit();

    var encoder = Encoder.init(&z3_ctx, testing.allocator);
    defer encoder.deinit();

    const mlir_ctx = mlir.oraContextCreate();
    defer mlir.oraContextDestroy(mlir_ctx);
    loadAllDialects(mlir_ctx);
    _ = mlir.oraDialectRegister(mlir_ctx);

    const loc = mlir.oraLocationUnknownGet(mlir_ctx);
    const i256_ty = mlir.oraIntegerTypeCreate(mlir_ctx, 256);

    // setCounter(v) writes storage slot "counter" and returns v.
    const set_name_attr = mlir.oraStringAttrCreate(mlir_ctx, stringRef("setCounter"));
    const effect_attr = mlir.oraStringAttrCreate(mlir_ctx, stringRef("writes"));
    const slot_attr = mlir.oraStringAttrCreate(mlir_ctx, stringRef("counter"));
    const slot_array = [_]mlir.MlirAttribute{slot_attr};
    const write_slots_attr = mlir.oraArrayAttrCreate(mlir_ctx, slot_array.len, &slot_array);
    const set_attrs = [_]mlir.MlirNamedAttribute{
        namedAttr(mlir_ctx, "sym_name", set_name_attr),
        namedAttr(mlir_ctx, "ora.effect", effect_attr),
        namedAttr(mlir_ctx, "ora.write_slots", write_slots_attr),
    };
    const set_param_types = [_]mlir.MlirType{i256_ty};
    const set_param_locs = [_]mlir.MlirLocation{loc};
    const set_func = mlir.oraFuncFuncOpCreate(mlir_ctx, loc, &set_attrs, set_attrs.len, &set_param_types, &set_param_locs, set_param_types.len);
    const set_body = mlir.oraFuncOpGetBodyBlock(set_func);
    const set_arg0 = mlir.oraBlockGetArgument(set_body, 0);
    const set_store = mlir.oraSStoreOpCreate(mlir_ctx, loc, set_arg0, stringRef("counter"));
    const set_ret_vals = [_]mlir.MlirValue{set_arg0};
    const set_ret = mlir.oraReturnOpCreate(mlir_ctx, loc, &set_ret_vals, set_ret_vals.len);
    mlir.oraBlockAppendOwnedOperation(set_body, set_store);
    mlir.oraBlockAppendOwnedOperation(set_body, set_ret);

    // wrapper(v) calls setCounter(v) and returns 0; call result is unused.
    const wrapper_name_attr = mlir.oraStringAttrCreate(mlir_ctx, stringRef("wrapper"));
    const wrapper_attrs = [_]mlir.MlirNamedAttribute{
        namedAttr(mlir_ctx, "sym_name", wrapper_name_attr),
        namedAttr(mlir_ctx, "ora.effect", effect_attr),
        namedAttr(mlir_ctx, "ora.write_slots", write_slots_attr),
    };
    const wrapper_param_types = [_]mlir.MlirType{i256_ty};
    const wrapper_param_locs = [_]mlir.MlirLocation{loc};
    const wrapper_func = mlir.oraFuncFuncOpCreate(
        mlir_ctx,
        loc,
        &wrapper_attrs,
        wrapper_attrs.len,
        &wrapper_param_types,
        &wrapper_param_locs,
        wrapper_param_types.len,
    );
    const wrapper_body = mlir.oraFuncOpGetBodyBlock(wrapper_func);
    const wrapper_arg0 = mlir.oraBlockGetArgument(wrapper_body, 0);
    const nested_call_operands = [_]mlir.MlirValue{wrapper_arg0};
    const nested_call_results = [_]mlir.MlirType{i256_ty};
    const nested_call = mlir.oraFuncCallOpCreate(
        mlir_ctx,
        loc,
        stringRef("setCounter"),
        &nested_call_operands,
        nested_call_operands.len,
        &nested_call_results,
        nested_call_results.len,
    );
    const zero_attr = mlir.oraIntegerAttrCreateI64FromType(i256_ty, 0);
    const zero_op = mlir.oraArithConstantOpCreate(mlir_ctx, loc, i256_ty, zero_attr);
    const zero = mlir.oraOperationGetResult(zero_op, 0);
    const wrapper_ret_vals = [_]mlir.MlirValue{zero};
    const wrapper_ret = mlir.oraReturnOpCreate(mlir_ctx, loc, &wrapper_ret_vals, wrapper_ret_vals.len);
    mlir.oraBlockAppendOwnedOperation(wrapper_body, nested_call);
    mlir.oraBlockAppendOwnedOperation(wrapper_body, zero_op);
    mlir.oraBlockAppendOwnedOperation(wrapper_body, wrapper_ret);

    try encoder.registerFunctionOperation(set_func);
    try encoder.registerFunctionOperation(wrapper_func);

    const arg_attr = mlir.oraIntegerAttrCreateI64FromType(i256_ty, 77);
    const arg_op = mlir.oraArithConstantOpCreate(mlir_ctx, loc, i256_ty, arg_attr);
    const arg = mlir.oraOperationGetResult(arg_op, 0);
    const call_operands = [_]mlir.MlirValue{arg};
    const call_results = [_]mlir.MlirType{i256_ty};
    const outer_call = mlir.oraFuncCallOpCreate(
        mlir_ctx,
        loc,
        stringRef("wrapper"),
        &call_operands,
        call_operands.len,
        &call_results,
        call_results.len,
    );
    _ = try encoder.encodeOperation(outer_call);

    const sload_after = mlir.oraSLoadOpCreate(mlir_ctx, loc, stringRef("counter"), i256_ty);
    const loaded = try encoder.encodeOperation(sload_after);
    const expected = try encoder.encodeValue(arg);

    var solver = try Solver.init(&z3_ctx, testing.allocator);
    defer solver.deinit();
    solver.assert(z3.Z3_mk_not(z3_ctx.ctx, z3.Z3_mk_eq(z3_ctx.ctx, loaded, expected)));
    try testing.expectEqual(@as(z3.Z3_lbool, z3.Z3_L_FALSE), solver.check());
}

test "func.call infers transitive state writes without metadata" {
    var z3_ctx = try Context.init(testing.allocator);
    defer z3_ctx.deinit();

    var encoder = Encoder.init(&z3_ctx, testing.allocator);
    defer encoder.deinit();

    const mlir_ctx = mlir.oraContextCreate();
    defer mlir.oraContextDestroy(mlir_ctx);
    loadAllDialects(mlir_ctx);
    _ = mlir.oraDialectRegister(mlir_ctx);

    const loc = mlir.oraLocationUnknownGet(mlir_ctx);
    const i256_ty = mlir.oraIntegerTypeCreate(mlir_ctx, 256);

    // setCounterNoMeta(v) { sstore counter <- v; return v; }
    const set_name_attr = mlir.oraStringAttrCreate(mlir_ctx, stringRef("setCounterNoMeta"));
    const set_attrs = [_]mlir.MlirNamedAttribute{
        namedAttr(mlir_ctx, "sym_name", set_name_attr),
    };
    const set_param_types = [_]mlir.MlirType{i256_ty};
    const set_param_locs = [_]mlir.MlirLocation{loc};
    const set_func = mlir.oraFuncFuncOpCreate(mlir_ctx, loc, &set_attrs, set_attrs.len, &set_param_types, &set_param_locs, set_param_types.len);
    const set_body = mlir.oraFuncOpGetBodyBlock(set_func);
    const set_arg0 = mlir.oraBlockGetArgument(set_body, 0);
    const set_store = mlir.oraSStoreOpCreate(mlir_ctx, loc, set_arg0, stringRef("counter"));
    const set_ret_vals = [_]mlir.MlirValue{set_arg0};
    const set_ret = mlir.oraReturnOpCreate(mlir_ctx, loc, &set_ret_vals, set_ret_vals.len);
    mlir.oraBlockAppendOwnedOperation(set_body, set_store);
    mlir.oraBlockAppendOwnedOperation(set_body, set_ret);

    // wrapperNoMeta(v) { setCounterNoMeta(v); return 0; }
    const wrap_name_attr = mlir.oraStringAttrCreate(mlir_ctx, stringRef("wrapperNoMeta"));
    const wrap_attrs = [_]mlir.MlirNamedAttribute{
        namedAttr(mlir_ctx, "sym_name", wrap_name_attr),
    };
    const wrap_param_types = [_]mlir.MlirType{i256_ty};
    const wrap_param_locs = [_]mlir.MlirLocation{loc};
    const wrap_func = mlir.oraFuncFuncOpCreate(mlir_ctx, loc, &wrap_attrs, wrap_attrs.len, &wrap_param_types, &wrap_param_locs, wrap_param_types.len);
    const wrap_body = mlir.oraFuncOpGetBodyBlock(wrap_func);
    const wrap_arg0 = mlir.oraBlockGetArgument(wrap_body, 0);
    const nested_call_operands = [_]mlir.MlirValue{wrap_arg0};
    const nested_call_results = [_]mlir.MlirType{i256_ty};
    const nested_call = mlir.oraFuncCallOpCreate(
        mlir_ctx,
        loc,
        stringRef("setCounterNoMeta"),
        &nested_call_operands,
        nested_call_operands.len,
        &nested_call_results,
        nested_call_results.len,
    );
    const zero_attr = mlir.oraIntegerAttrCreateI64FromType(i256_ty, 0);
    const zero_op = mlir.oraArithConstantOpCreate(mlir_ctx, loc, i256_ty, zero_attr);
    const zero = mlir.oraOperationGetResult(zero_op, 0);
    const wrap_ret_vals = [_]mlir.MlirValue{zero};
    const wrap_ret = mlir.oraReturnOpCreate(mlir_ctx, loc, &wrap_ret_vals, wrap_ret_vals.len);
    mlir.oraBlockAppendOwnedOperation(wrap_body, nested_call);
    mlir.oraBlockAppendOwnedOperation(wrap_body, zero_op);
    mlir.oraBlockAppendOwnedOperation(wrap_body, wrap_ret);

    try encoder.registerFunctionOperation(set_func);
    try encoder.registerFunctionOperation(wrap_func);

    const arg_attr = mlir.oraIntegerAttrCreateI64FromType(i256_ty, 91);
    const arg_op = mlir.oraArithConstantOpCreate(mlir_ctx, loc, i256_ty, arg_attr);
    const arg = mlir.oraOperationGetResult(arg_op, 0);
    const outer_operands = [_]mlir.MlirValue{arg};
    const outer_results = [_]mlir.MlirType{i256_ty};
    const outer_call = mlir.oraFuncCallOpCreate(
        mlir_ctx,
        loc,
        stringRef("wrapperNoMeta"),
        &outer_operands,
        outer_operands.len,
        &outer_results,
        outer_results.len,
    );
    _ = try encoder.encodeOperation(outer_call);

    const sload_after = mlir.oraSLoadOpCreate(mlir_ctx, loc, stringRef("counter"), i256_ty);
    const loaded = try encoder.encodeOperation(sload_after);
    const expected = try encoder.encodeValue(arg);

    var solver = try Solver.init(&z3_ctx, testing.allocator);
    defer solver.deinit();
    solver.assert(z3.Z3_mk_not(z3_ctx.ctx, z3.Z3_mk_eq(z3_ctx.ctx, loaded, expected)));
    try testing.expectEqual(@as(z3.Z3_lbool, z3.Z3_L_FALSE), solver.check());
}

test "func.call preserves state summary for multi-return callee" {
    var z3_ctx = try Context.init(testing.allocator);
    defer z3_ctx.deinit();

    var encoder = Encoder.init(&z3_ctx, testing.allocator);
    defer encoder.deinit();

    const mlir_ctx = mlir.oraContextCreate();
    defer mlir.oraContextDestroy(mlir_ctx);
    loadAllDialects(mlir_ctx);
    _ = mlir.oraDialectRegister(mlir_ctx);

    const loc = mlir.oraLocationUnknownGet(mlir_ctx);
    const i1_ty = mlir.oraIntegerTypeCreate(mlir_ctx, 1);
    const i256_ty = mlir.oraIntegerTypeCreate(mlir_ctx, 256);

    const helper_name_attr = mlir.oraStringAttrCreate(mlir_ctx, stringRef("setCounterMaybe"));
    const effect_attr = mlir.oraStringAttrCreate(mlir_ctx, stringRef("writes"));
    const slot_attr = mlir.oraStringAttrCreate(mlir_ctx, stringRef("counter"));
    const slot_array = [_]mlir.MlirAttribute{slot_attr};
    const write_slots_attr = mlir.oraArrayAttrCreate(mlir_ctx, slot_array.len, &slot_array);
    const helper_attrs = [_]mlir.MlirNamedAttribute{
        namedAttr(mlir_ctx, "sym_name", helper_name_attr),
        namedAttr(mlir_ctx, "ora.effect", effect_attr),
        namedAttr(mlir_ctx, "ora.write_slots", write_slots_attr),
    };
    const helper_param_types = [_]mlir.MlirType{ i1_ty, i256_ty };
    const helper_param_locs = [_]mlir.MlirLocation{ loc, loc };
    const helper_func = mlir.oraFuncFuncOpCreate(mlir_ctx, loc, &helper_attrs, helper_attrs.len, &helper_param_types, &helper_param_locs, helper_param_types.len);
    const helper_body = mlir.oraFuncOpGetBodyBlock(helper_func);
    const helper_flag = mlir.oraBlockGetArgument(helper_body, 0);
    const helper_value = mlir.oraBlockGetArgument(helper_body, 1);

    const empty_result_types = [_]mlir.MlirType{};
    const if_op = mlir.oraScfIfOpCreate(mlir_ctx, loc, helper_flag, &empty_result_types, empty_result_types.len, false);
    const then_block = mlir.oraScfIfOpGetThenBlock(if_op);
    const then_store = mlir.oraSStoreOpCreate(mlir_ctx, loc, helper_value, stringRef("counter"));
    const empty_return_vals = [_]mlir.MlirValue{};
    const then_ret = mlir.oraReturnOpCreate(mlir_ctx, loc, &empty_return_vals, empty_return_vals.len);
    mlir.oraBlockAppendOwnedOperation(then_block, then_store);
    mlir.oraBlockAppendOwnedOperation(then_block, then_ret);

    const helper_store = mlir.oraSStoreOpCreate(mlir_ctx, loc, helper_value, stringRef("counter"));
    const helper_ret = mlir.oraReturnOpCreate(mlir_ctx, loc, &empty_return_vals, empty_return_vals.len);
    mlir.oraBlockAppendOwnedOperation(helper_body, if_op);
    mlir.oraBlockAppendOwnedOperation(helper_body, helper_store);
    mlir.oraBlockAppendOwnedOperation(helper_body, helper_ret);

    const wrapper_name_attr = mlir.oraStringAttrCreate(mlir_ctx, stringRef("wrapperMultiReturn"));
    const wrapper_attrs = [_]mlir.MlirNamedAttribute{
        namedAttr(mlir_ctx, "sym_name", wrapper_name_attr),
    };
    const wrapper_param_types = [_]mlir.MlirType{i256_ty};
    const wrapper_param_locs = [_]mlir.MlirLocation{loc};
    const wrapper_func = mlir.oraFuncFuncOpCreate(mlir_ctx, loc, &wrapper_attrs, wrapper_attrs.len, &wrapper_param_types, &wrapper_param_locs, wrapper_param_types.len);
    const wrapper_body = mlir.oraFuncOpGetBodyBlock(wrapper_func);
    const wrapper_value = mlir.oraBlockGetArgument(wrapper_body, 0);

    const true_attr = mlir.oraIntegerAttrCreateI64FromType(i1_ty, 1);
    const true_op = mlir.oraArithConstantOpCreate(mlir_ctx, loc, i1_ty, true_attr);
    const true_val = mlir.oraOperationGetResult(true_op, 0);
    const call_operands = [_]mlir.MlirValue{ true_val, wrapper_value };
    const no_results = [_]mlir.MlirType{};
    const call_op = mlir.oraFuncCallOpCreate(
        mlir_ctx,
        loc,
        stringRef("setCounterMaybe"),
        &call_operands,
        call_operands.len,
        &no_results,
        no_results.len,
    );
    const zero_attr = mlir.oraIntegerAttrCreateI64FromType(i256_ty, 0);
    const zero_op = mlir.oraArithConstantOpCreate(mlir_ctx, loc, i256_ty, zero_attr);
    const zero = mlir.oraOperationGetResult(zero_op, 0);
    const wrapper_ret_vals = [_]mlir.MlirValue{zero};
    const wrapper_ret = mlir.oraReturnOpCreate(mlir_ctx, loc, &wrapper_ret_vals, wrapper_ret_vals.len);
    mlir.oraBlockAppendOwnedOperation(wrapper_body, true_op);
    mlir.oraBlockAppendOwnedOperation(wrapper_body, call_op);
    mlir.oraBlockAppendOwnedOperation(wrapper_body, zero_op);
    mlir.oraBlockAppendOwnedOperation(wrapper_body, wrapper_ret);

    try encoder.registerFunctionOperation(helper_func);
    try encoder.registerFunctionOperation(wrapper_func);

    const arg_attr = mlir.oraIntegerAttrCreateI64FromType(i256_ty, 91);
    const arg_op = mlir.oraArithConstantOpCreate(mlir_ctx, loc, i256_ty, arg_attr);
    const arg = mlir.oraOperationGetResult(arg_op, 0);
    const outer_operands = [_]mlir.MlirValue{arg};
    const outer_results = [_]mlir.MlirType{i256_ty};
    const outer_call = mlir.oraFuncCallOpCreate(
        mlir_ctx,
        loc,
        stringRef("wrapperMultiReturn"),
        &outer_operands,
        outer_operands.len,
        &outer_results,
        outer_results.len,
    );
    _ = try encoder.encodeOperation(outer_call);

    const sload_after = mlir.oraSLoadOpCreate(mlir_ctx, loc, stringRef("counter"), i256_ty);
    const loaded = try encoder.encodeOperation(sload_after);
    const expected = try encoder.encodeValue(arg);

    var solver = try Solver.init(&z3_ctx, testing.allocator);
    defer solver.deinit();
    solver.assert(z3.Z3_mk_not(z3_ctx.ctx, z3.Z3_mk_eq(z3_ctx.ctx, loaded, expected)));
    try testing.expectEqual(@as(z3.Z3_lbool, z3.Z3_L_FALSE), solver.check());
}

test "func.call summary preserves branch-guarded storage writes" {
    var z3_ctx = try Context.init(testing.allocator);
    defer z3_ctx.deinit();

    var encoder = Encoder.init(&z3_ctx, testing.allocator);
    defer encoder.deinit();

    const mlir_ctx = mlir.oraContextCreate();
    defer mlir.oraContextDestroy(mlir_ctx);
    loadAllDialects(mlir_ctx);
    _ = mlir.oraDialectRegister(mlir_ctx);

    const loc = mlir.oraLocationUnknownGet(mlir_ctx);
    const i1_ty = mlir.oraIntegerTypeCreate(mlir_ctx, 1);
    const i256_ty = mlir.oraIntegerTypeCreate(mlir_ctx, 256);

    const helper_name_attr = mlir.oraStringAttrCreate(mlir_ctx, stringRef("conditionalStore"));
    const effect_attr = mlir.oraStringAttrCreate(mlir_ctx, stringRef("writes"));
    const slot_attr = mlir.oraStringAttrCreate(mlir_ctx, stringRef("counter"));
    const slot_array = [_]mlir.MlirAttribute{slot_attr};
    const write_slots_attr = mlir.oraArrayAttrCreate(mlir_ctx, slot_array.len, &slot_array);
    const helper_attrs = [_]mlir.MlirNamedAttribute{
        namedAttr(mlir_ctx, "sym_name", helper_name_attr),
        namedAttr(mlir_ctx, "ora.effect", effect_attr),
        namedAttr(mlir_ctx, "ora.write_slots", write_slots_attr),
    };
    const helper_param_types = [_]mlir.MlirType{ i1_ty, i256_ty };
    const helper_param_locs = [_]mlir.MlirLocation{ loc, loc };
    const helper_func = mlir.oraFuncFuncOpCreate(mlir_ctx, loc, &helper_attrs, helper_attrs.len, &helper_param_types, &helper_param_locs, helper_param_types.len);
    const helper_body = mlir.oraFuncOpGetBodyBlock(helper_func);
    const helper_flag = mlir.oraBlockGetArgument(helper_body, 0);
    const helper_value = mlir.oraBlockGetArgument(helper_body, 1);

    const no_results = [_]mlir.MlirType{};
    const if_op = mlir.oraScfIfOpCreate(mlir_ctx, loc, helper_flag, &no_results, no_results.len, true);
    const then_block = mlir.oraScfIfOpGetThenBlock(if_op);
    const else_block = mlir.oraScfIfOpGetElseBlock(if_op);
    const then_store = mlir.oraSStoreOpCreate(mlir_ctx, loc, helper_value, stringRef("counter"));
    mlir.oraBlockAppendOwnedOperation(then_block, then_store);
    mlir.oraBlockAppendOwnedOperation(then_block, mlir.oraScfYieldOpCreate(mlir_ctx, loc, &[_]mlir.MlirValue{}, 0));
    mlir.oraBlockAppendOwnedOperation(else_block, mlir.oraScfYieldOpCreate(mlir_ctx, loc, &[_]mlir.MlirValue{}, 0));

    const empty_return_vals = [_]mlir.MlirValue{};
    const helper_ret = mlir.oraReturnOpCreate(mlir_ctx, loc, &empty_return_vals, empty_return_vals.len);
    mlir.oraBlockAppendOwnedOperation(helper_body, if_op);
    mlir.oraBlockAppendOwnedOperation(helper_body, helper_ret);

    try encoder.registerFunctionOperation(helper_func);

    const seed_attr = mlir.oraIntegerAttrCreateI64FromType(i256_ty, 5);
    const seed_op = mlir.oraArithConstantOpCreate(mlir_ctx, loc, i256_ty, seed_attr);
    const seed = mlir.oraOperationGetResult(seed_op, 0);
    _ = try encoder.encodeOperation(mlir.oraSStoreOpCreate(mlir_ctx, loc, seed, stringRef("counter")));

    const false_attr = mlir.oraIntegerAttrCreateI64FromType(i1_ty, 0);
    const false_op = mlir.oraArithConstantOpCreate(mlir_ctx, loc, i1_ty, false_attr);
    const false_val = mlir.oraOperationGetResult(false_op, 0);
    const arg_attr = mlir.oraIntegerAttrCreateI64FromType(i256_ty, 9);
    const arg_op = mlir.oraArithConstantOpCreate(mlir_ctx, loc, i256_ty, arg_attr);
    const arg = mlir.oraOperationGetResult(arg_op, 0);

    const call_operands = [_]mlir.MlirValue{ false_val, arg };
    const call_op = mlir.oraFuncCallOpCreate(
        mlir_ctx,
        loc,
        stringRef("conditionalStore"),
        &call_operands,
        call_operands.len,
        &[_]mlir.MlirType{},
        0,
    );
    _ = try encoder.encodeOperation(call_op);

    const sload_after = mlir.oraSLoadOpCreate(mlir_ctx, loc, stringRef("counter"), i256_ty);
    const loaded = try encoder.encodeOperation(sload_after);
    const expected = try encoder.encodeValue(seed);

    var solver = try Solver.init(&z3_ctx, testing.allocator);
    defer solver.deinit();
    solver.assert(z3.Z3_mk_not(z3_ctx.ctx, z3.Z3_mk_eq(z3_ctx.ctx, loaded, expected)));
    try testing.expectEqual(@as(z3.Z3_lbool, z3.Z3_L_FALSE), solver.check());
}

test "func.call summary preserves switch-guarded storage writes" {
    var z3_ctx = try Context.init(testing.allocator);
    defer z3_ctx.deinit();

    var encoder = Encoder.init(&z3_ctx, testing.allocator);
    defer encoder.deinit();

    const mlir_ctx = mlir.oraContextCreate();
    defer mlir.oraContextDestroy(mlir_ctx);
    loadAllDialects(mlir_ctx);
    _ = mlir.oraDialectRegister(mlir_ctx);

    const loc = mlir.oraLocationUnknownGet(mlir_ctx);
    const i256_ty = mlir.oraIntegerTypeCreate(mlir_ctx, 256);

    const helper_name_attr = mlir.oraStringAttrCreate(mlir_ctx, stringRef("switchStore"));
    const effect_attr = mlir.oraStringAttrCreate(mlir_ctx, stringRef("writes"));
    const slot_attr = mlir.oraStringAttrCreate(mlir_ctx, stringRef("counter"));
    const slot_array = [_]mlir.MlirAttribute{slot_attr};
    const write_slots_attr = mlir.oraArrayAttrCreate(mlir_ctx, slot_array.len, &slot_array);
    const helper_attrs = [_]mlir.MlirNamedAttribute{
        namedAttr(mlir_ctx, "sym_name", helper_name_attr),
        namedAttr(mlir_ctx, "ora.effect", effect_attr),
        namedAttr(mlir_ctx, "ora.write_slots", write_slots_attr),
    };
    const helper_param_types = [_]mlir.MlirType{ i256_ty, i256_ty };
    const helper_param_locs = [_]mlir.MlirLocation{ loc, loc };
    const helper_func = mlir.oraFuncFuncOpCreate(mlir_ctx, loc, &helper_attrs, helper_attrs.len, &helper_param_types, &helper_param_locs, helper_param_types.len);
    const helper_body = mlir.oraFuncOpGetBodyBlock(helper_func);
    const helper_tag = mlir.oraBlockGetArgument(helper_body, 0);
    const helper_value = mlir.oraBlockGetArgument(helper_body, 1);

    const no_results = [_]mlir.MlirType{};
    const switch_op = mlir.oraSwitchOpCreateWithCases(mlir_ctx, loc, helper_tag, &no_results, 0, 2);
    const case0_block = mlir.oraSwitchOpGetCaseBlock(switch_op, 0);
    const default_block = mlir.oraSwitchOpGetCaseBlock(switch_op, 1);
    mlir.oraBlockAppendOwnedOperation(case0_block, mlir.oraSStoreOpCreate(mlir_ctx, loc, helper_value, stringRef("counter")));
    mlir.oraBlockAppendOwnedOperation(case0_block, mlir.oraYieldOpCreate(mlir_ctx, loc, &[_]mlir.MlirValue{}, 0));
    mlir.oraBlockAppendOwnedOperation(default_block, mlir.oraYieldOpCreate(mlir_ctx, loc, &[_]mlir.MlirValue{}, 0));

    var case_values = [_]i64{ 1, 0 };
    var range_starts = [_]i64{ 0, 0 };
    var range_ends = [_]i64{ 0, 0 };
    var case_kinds = [_]i64{ 0, 2 };
    mlir.oraSwitchOpSetCasePatterns(switch_op, &case_values, &range_starts, &range_ends, &case_kinds, 1, 2);

    const empty_return_vals = [_]mlir.MlirValue{};
    const helper_ret = mlir.oraReturnOpCreate(mlir_ctx, loc, &empty_return_vals, empty_return_vals.len);
    mlir.oraBlockAppendOwnedOperation(helper_body, switch_op);
    mlir.oraBlockAppendOwnedOperation(helper_body, helper_ret);

    try encoder.registerFunctionOperation(helper_func);

    const seed_attr = mlir.oraIntegerAttrCreateI64FromType(i256_ty, 5);
    const seed_op = mlir.oraArithConstantOpCreate(mlir_ctx, loc, i256_ty, seed_attr);
    const seed = mlir.oraOperationGetResult(seed_op, 0);
    _ = try encoder.encodeOperation(mlir.oraSStoreOpCreate(mlir_ctx, loc, seed, stringRef("counter")));

    const tag_attr = mlir.oraIntegerAttrCreateI64FromType(i256_ty, 2);
    const tag_op = mlir.oraArithConstantOpCreate(mlir_ctx, loc, i256_ty, tag_attr);
    const tag = mlir.oraOperationGetResult(tag_op, 0);
    const arg_attr = mlir.oraIntegerAttrCreateI64FromType(i256_ty, 9);
    const arg_op = mlir.oraArithConstantOpCreate(mlir_ctx, loc, i256_ty, arg_attr);
    const arg = mlir.oraOperationGetResult(arg_op, 0);

    const call_operands = [_]mlir.MlirValue{ tag, arg };
    const call_op = mlir.oraFuncCallOpCreate(
        mlir_ctx,
        loc,
        stringRef("switchStore"),
        &call_operands,
        call_operands.len,
        &[_]mlir.MlirType{},
        0,
    );
    _ = try encoder.encodeOperation(call_op);

    const sload_after = mlir.oraSLoadOpCreate(mlir_ctx, loc, stringRef("counter"), i256_ty);
    const loaded = try encoder.encodeOperation(sload_after);
    const expected = try encoder.encodeValue(seed);

    var solver = try Solver.init(&z3_ctx, testing.allocator);
    defer solver.deinit();
    solver.assert(z3.Z3_mk_not(z3_ctx.ctx, z3.Z3_mk_eq(z3_ctx.ctx, loaded, expected)));
    try testing.expectEqual(@as(z3.Z3_lbool, z3.Z3_L_FALSE), solver.check());
}

test "func.call summary preserves branch-guarded result and storage together" {
    var z3_ctx = try Context.init(testing.allocator);
    defer z3_ctx.deinit();

    var encoder = Encoder.init(&z3_ctx, testing.allocator);
    defer encoder.deinit();

    const mlir_ctx = mlir.oraContextCreate();
    defer mlir.oraContextDestroy(mlir_ctx);
    loadAllDialects(mlir_ctx);
    _ = mlir.oraDialectRegister(mlir_ctx);

    const loc = mlir.oraLocationUnknownGet(mlir_ctx);
    const i1_ty = mlir.oraIntegerTypeCreate(mlir_ctx, 1);
    const i256_ty = mlir.oraIntegerTypeCreate(mlir_ctx, 256);

    const helper_name_attr = mlir.oraStringAttrCreate(mlir_ctx, stringRef("branchResultAndStore"));
    const effect_attr = mlir.oraStringAttrCreate(mlir_ctx, stringRef("writes"));
    const slot_attr = mlir.oraStringAttrCreate(mlir_ctx, stringRef("counter"));
    const slot_array = [_]mlir.MlirAttribute{slot_attr};
    const write_slots_attr = mlir.oraArrayAttrCreate(mlir_ctx, slot_array.len, &slot_array);
    const helper_attrs = [_]mlir.MlirNamedAttribute{
        namedAttr(mlir_ctx, "sym_name", helper_name_attr),
        namedAttr(mlir_ctx, "ora.effect", effect_attr),
        namedAttr(mlir_ctx, "ora.write_slots", write_slots_attr),
    };
    const helper_param_types = [_]mlir.MlirType{ i1_ty, i256_ty };
    const helper_param_locs = [_]mlir.MlirLocation{ loc, loc };
    const helper_func = mlir.oraFuncFuncOpCreate(mlir_ctx, loc, &helper_attrs, helper_attrs.len, &helper_param_types, &helper_param_locs, helper_param_types.len);
    const helper_body = mlir.oraFuncOpGetBodyBlock(helper_func);
    const helper_flag = mlir.oraBlockGetArgument(helper_body, 0);
    const helper_value = mlir.oraBlockGetArgument(helper_body, 1);

    const zero_attr = mlir.oraIntegerAttrCreateI64FromType(i256_ty, 0);
    const zero_op = mlir.oraArithConstantOpCreate(mlir_ctx, loc, i256_ty, zero_attr);
    const zero = mlir.oraOperationGetResult(zero_op, 0);

    const if_results = [_]mlir.MlirType{i256_ty};
    const if_op = mlir.oraScfIfOpCreate(mlir_ctx, loc, helper_flag, &if_results, if_results.len, true);
    const then_block = mlir.oraScfIfOpGetThenBlock(if_op);
    const else_block = mlir.oraScfIfOpGetElseBlock(if_op);

    mlir.oraBlockAppendOwnedOperation(then_block, mlir.oraSStoreOpCreate(mlir_ctx, loc, helper_value, stringRef("counter")));
    mlir.oraBlockAppendOwnedOperation(then_block, mlir.oraScfYieldOpCreate(
        mlir_ctx,
        loc,
        &[_]mlir.MlirValue{helper_value},
        1,
    ));

    mlir.oraBlockAppendOwnedOperation(else_block, mlir.oraSStoreOpCreate(mlir_ctx, loc, zero, stringRef("counter")));
    mlir.oraBlockAppendOwnedOperation(else_block, mlir.oraScfYieldOpCreate(
        mlir_ctx,
        loc,
        &[_]mlir.MlirValue{zero},
        1,
    ));

    mlir.oraBlockAppendOwnedOperation(helper_body, zero_op);
    mlir.oraBlockAppendOwnedOperation(helper_body, if_op);
    mlir.oraBlockAppendOwnedOperation(helper_body, mlir.oraReturnOpCreate(
        mlir_ctx,
        loc,
        &[_]mlir.MlirValue{mlir.oraOperationGetResult(if_op, 0)},
        1,
    ));

    try encoder.registerFunctionOperation(helper_func);

    const call_flag = mlir.oraVariablePlaceholderOpCreate(mlir_ctx, loc, stringRef("branchResultStoreFlag"), i1_ty);
    const call_value = mlir.oraVariablePlaceholderOpCreate(mlir_ctx, loc, stringRef("branchResultStoreValue"), i256_ty);
    const call_operands = [_]mlir.MlirValue{
        mlir.oraOperationGetResult(call_flag, 0),
        mlir.oraOperationGetResult(call_value, 0),
    };
    const call = mlir.oraFuncCallOpCreate(
        mlir_ctx,
        loc,
        stringRef("branchResultAndStore"),
        &call_operands,
        call_operands.len,
        &[_]mlir.MlirType{i256_ty},
        1,
    );

    const encoded_result = try encoder.encodeOperation(call);
    const loaded = try encoder.encodeOperation(mlir.oraSLoadOpCreate(mlir_ctx, loc, stringRef("counter"), i256_ty));

    try testing.expect(!encoder.isDegraded());

    const cond_ast = try encoder.encodeValue(mlir.oraOperationGetResult(call_flag, 0));
    const value_ast = try encoder.encodeValue(mlir.oraOperationGetResult(call_value, 0));
    const zero_ast = try encoder.encodeValue(zero);
    const expected = z3.Z3_mk_ite(z3_ctx.ctx, encoder.coerceBoolean(cond_ast), value_ast, zero_ast);

    var solver = try Solver.init(&z3_ctx, testing.allocator);
    defer solver.deinit();
    solver.assert(z3.Z3_mk_not(z3_ctx.ctx, z3.Z3_mk_eq(z3_ctx.ctx, encoded_result, expected)));
    try testing.expectEqual(@as(z3.Z3_lbool, z3.Z3_L_FALSE), solver.check());

    try solver.resetChecked();
    try solver.assertChecked(z3.Z3_mk_not(z3_ctx.ctx, z3.Z3_mk_eq(z3_ctx.ctx, loaded, expected)));
    try testing.expectEqual(@as(z3.Z3_lbool, z3.Z3_L_FALSE), try solver.checkChecked());
}

test "func.call summary encodes switch-selected pure returns exactly" {
    var z3_ctx = try Context.init(testing.allocator);
    defer z3_ctx.deinit();

    var encoder = Encoder.init(&z3_ctx, testing.allocator);
    defer encoder.deinit();

    const mlir_ctx = mlir.oraContextCreate();
    defer mlir.oraContextDestroy(mlir_ctx);
    loadAllDialects(mlir_ctx);
    _ = mlir.oraDialectRegister(mlir_ctx);

    const loc = mlir.oraLocationUnknownGet(mlir_ctx);
    const i256_ty = mlir.oraIntegerTypeCreate(mlir_ctx, 256);

    const helper_attrs = [_]mlir.MlirNamedAttribute{
        namedAttr(mlir_ctx, "sym_name", mlir.oraStringAttrCreate(mlir_ctx, stringRef("switchPure"))),
    };
    const helper_param_types = [_]mlir.MlirType{i256_ty};
    const helper_param_locs = [_]mlir.MlirLocation{loc};
    const helper = mlir.oraFuncFuncOpCreate(mlir_ctx, loc, &helper_attrs, helper_attrs.len, &helper_param_types, &helper_param_locs, helper_param_types.len);
    const helper_body = mlir.oraFuncOpGetBodyBlock(helper);
    const helper_tag = mlir.oraBlockGetArgument(helper_body, 0);

    const switch_results = [_]mlir.MlirType{i256_ty};
    const switch_op = mlir.oraSwitchOpCreateWithCases(mlir_ctx, loc, helper_tag, &switch_results, switch_results.len, 2);
    const case0_block = mlir.oraSwitchOpGetCaseBlock(switch_op, 0);
    const default_block = mlir.oraSwitchOpGetCaseBlock(switch_op, 1);

    const one_attr = mlir.oraIntegerAttrCreateI64FromType(i256_ty, 1);
    const one_op = mlir.oraArithConstantOpCreate(mlir_ctx, loc, i256_ty, one_attr);
    const one_val = mlir.oraOperationGetResult(one_op, 0);
    mlir.oraBlockAppendOwnedOperation(case0_block, one_op);
    mlir.oraBlockAppendOwnedOperation(case0_block, mlir.oraReturnOpCreate(mlir_ctx, loc, &[_]mlir.MlirValue{one_val}, 1));

    const two_attr = mlir.oraIntegerAttrCreateI64FromType(i256_ty, 2);
    const two_op = mlir.oraArithConstantOpCreate(mlir_ctx, loc, i256_ty, two_attr);
    const two_val = mlir.oraOperationGetResult(two_op, 0);
    mlir.oraBlockAppendOwnedOperation(default_block, two_op);
    mlir.oraBlockAppendOwnedOperation(default_block, mlir.oraReturnOpCreate(mlir_ctx, loc, &[_]mlir.MlirValue{two_val}, 1));

    var case_values = [_]i64{ 1, 0 };
    var range_starts = [_]i64{ 0, 0 };
    var range_ends = [_]i64{ 0, 0 };
    var case_kinds = [_]i64{ 0, 2 };
    mlir.oraSwitchOpSetCasePatterns(switch_op, &case_values, &range_starts, &range_ends, &case_kinds, 1, 2);
    mlir.oraBlockAppendOwnedOperation(helper_body, switch_op);

    try encoder.registerFunctionOperation(helper);

    const tag1_attr = mlir.oraIntegerAttrCreateI64FromType(i256_ty, 1);
    const tag1_op = mlir.oraArithConstantOpCreate(mlir_ctx, loc, i256_ty, tag1_attr);
    const tag1 = mlir.oraOperationGetResult(tag1_op, 0);
    const tag1_ast = try encoder.encodeOperation(tag1_op);

    const tag2_attr = mlir.oraIntegerAttrCreateI64FromType(i256_ty, 2);
    const tag2_op = mlir.oraArithConstantOpCreate(mlir_ctx, loc, i256_ty, tag2_attr);
    const tag2 = mlir.oraOperationGetResult(tag2_op, 0);
    const tag2_ast = try encoder.encodeOperation(tag2_op);

    const call1 = mlir.oraFuncCallOpCreate(
        mlir_ctx,
        loc,
        stringRef("switchPure"),
        &[_]mlir.MlirValue{tag1},
        1,
        &[_]mlir.MlirType{i256_ty},
        1,
    );
    const call2 = mlir.oraFuncCallOpCreate(
        mlir_ctx,
        loc,
        stringRef("switchPure"),
        &[_]mlir.MlirValue{tag2},
        1,
        &[_]mlir.MlirType{i256_ty},
        1,
    );

    const encoded1 = try encoder.encodeOperation(call1);
    const encoded2 = try encoder.encodeOperation(call2);
    try testing.expect(!encoder.isDegraded());

    var solver = try Solver.init(&z3_ctx, testing.allocator);
    defer solver.deinit();
    solver.assert(z3.Z3_mk_not(z3_ctx.ctx, z3.Z3_mk_eq(z3_ctx.ctx, encoded1, tag1_ast)));
    try testing.expectEqual(@as(z3.Z3_lbool, z3.Z3_L_FALSE), solver.check());

    solver.reset();
    solver.assert(z3.Z3_mk_not(z3_ctx.ctx, z3.Z3_mk_eq(z3_ctx.ctx, encoded2, tag2_ast)));
    try testing.expectEqual(@as(z3.Z3_lbool, z3.Z3_L_FALSE), solver.check());
}

test "func.call summary preserves conditional return fallthrough state" {
    var z3_ctx = try Context.init(testing.allocator);
    defer z3_ctx.deinit();

    var encoder = Encoder.init(&z3_ctx, testing.allocator);
    defer encoder.deinit();

    const mlir_ctx = mlir.oraContextCreate();
    defer mlir.oraContextDestroy(mlir_ctx);
    loadAllDialects(mlir_ctx);
    _ = mlir.oraDialectRegister(mlir_ctx);

    const loc = mlir.oraLocationUnknownGet(mlir_ctx);
    const i1_ty = mlir.oraIntegerTypeCreate(mlir_ctx, 1);
    const i256_ty = mlir.oraIntegerTypeCreate(mlir_ctx, 256);

    const helper_name_attr = mlir.oraStringAttrCreate(mlir_ctx, stringRef("returnOrStore"));
    const effect_attr = mlir.oraStringAttrCreate(mlir_ctx, stringRef("writes"));
    const slot_attr = mlir.oraStringAttrCreate(mlir_ctx, stringRef("counter"));
    const slot_array = [_]mlir.MlirAttribute{slot_attr};
    const write_slots_attr = mlir.oraArrayAttrCreate(mlir_ctx, slot_array.len, &slot_array);
    const helper_attrs = [_]mlir.MlirNamedAttribute{
        namedAttr(mlir_ctx, "sym_name", helper_name_attr),
        namedAttr(mlir_ctx, "ora.effect", effect_attr),
        namedAttr(mlir_ctx, "ora.write_slots", write_slots_attr),
    };
    const helper_param_types = [_]mlir.MlirType{ i1_ty, i256_ty };
    const helper_param_locs = [_]mlir.MlirLocation{ loc, loc };
    const helper_func = mlir.oraFuncFuncOpCreate(mlir_ctx, loc, &helper_attrs, helper_attrs.len, &helper_param_types, &helper_param_locs, helper_param_types.len);
    const helper_body = mlir.oraFuncOpGetBodyBlock(helper_func);
    const helper_flag = mlir.oraBlockGetArgument(helper_body, 0);
    const helper_value = mlir.oraBlockGetArgument(helper_body, 1);

    const conditional_ret = mlir.oraConditionalReturnOpCreate(mlir_ctx, loc, helper_flag);
    const then_block = mlir.oraConditionalReturnOpGetThenBlock(conditional_ret);
    const else_block = mlir.oraConditionalReturnOpGetElseBlock(conditional_ret);
    mlir.oraBlockAppendOwnedOperation(then_block, mlir.oraReturnOpCreate(mlir_ctx, loc, &[_]mlir.MlirValue{}, 0));
    mlir.oraBlockAppendOwnedOperation(else_block, mlir.oraYieldOpCreate(mlir_ctx, loc, &[_]mlir.MlirValue{}, 0));

    const helper_store = mlir.oraSStoreOpCreate(mlir_ctx, loc, helper_value, stringRef("counter"));
    const helper_ret = mlir.oraReturnOpCreate(mlir_ctx, loc, &[_]mlir.MlirValue{}, 0);
    mlir.oraBlockAppendOwnedOperation(helper_body, conditional_ret);
    mlir.oraBlockAppendOwnedOperation(helper_body, helper_store);
    mlir.oraBlockAppendOwnedOperation(helper_body, helper_ret);

    try encoder.registerFunctionOperation(helper_func);

    const seed_attr = mlir.oraIntegerAttrCreateI64FromType(i256_ty, 5);
    const seed_op = mlir.oraArithConstantOpCreate(mlir_ctx, loc, i256_ty, seed_attr);
    const seed = mlir.oraOperationGetResult(seed_op, 0);
    _ = try encoder.encodeOperation(mlir.oraSStoreOpCreate(mlir_ctx, loc, seed, stringRef("counter")));

    const true_attr = mlir.oraIntegerAttrCreateI64FromType(i1_ty, 1);
    const true_op = mlir.oraArithConstantOpCreate(mlir_ctx, loc, i1_ty, true_attr);
    const true_val = mlir.oraOperationGetResult(true_op, 0);
    const arg_attr = mlir.oraIntegerAttrCreateI64FromType(i256_ty, 9);
    const arg_op = mlir.oraArithConstantOpCreate(mlir_ctx, loc, i256_ty, arg_attr);
    const arg = mlir.oraOperationGetResult(arg_op, 0);

    const call_operands = [_]mlir.MlirValue{ true_val, arg };
    const call_op = mlir.oraFuncCallOpCreate(
        mlir_ctx,
        loc,
        stringRef("returnOrStore"),
        &call_operands,
        call_operands.len,
        &[_]mlir.MlirType{},
        0,
    );
    _ = try encoder.encodeOperation(call_op);

    const sload_after = mlir.oraSLoadOpCreate(mlir_ctx, loc, stringRef("counter"), i256_ty);
    const loaded = try encoder.encodeOperation(sload_after);
    const expected = try encoder.encodeValue(seed);

    var solver = try Solver.init(&z3_ctx, testing.allocator);
    defer solver.deinit();
    solver.assert(z3.Z3_mk_not(z3_ctx.ctx, z3.Z3_mk_eq(z3_ctx.ctx, loaded, expected)));
    try testing.expectEqual(@as(z3.Z3_lbool, z3.Z3_L_FALSE), solver.check());
}

test "func.call summary preserves conditional return result and state together" {
    var z3_ctx = try Context.init(testing.allocator);
    defer z3_ctx.deinit();

    var encoder = Encoder.init(&z3_ctx, testing.allocator);
    defer encoder.deinit();

    const mlir_ctx = mlir.oraContextCreate();
    defer mlir.oraContextDestroy(mlir_ctx);
    loadAllDialects(mlir_ctx);
    _ = mlir.oraDialectRegister(mlir_ctx);

    const loc = mlir.oraLocationUnknownGet(mlir_ctx);
    const i1_ty = mlir.oraIntegerTypeCreate(mlir_ctx, 1);
    const i256_ty = mlir.oraIntegerTypeCreate(mlir_ctx, 256);

    const helper_name_attr = mlir.oraStringAttrCreate(mlir_ctx, stringRef("returnOrStoreValue"));
    const effect_attr = mlir.oraStringAttrCreate(mlir_ctx, stringRef("writes"));
    const slot_attr = mlir.oraStringAttrCreate(mlir_ctx, stringRef("counter"));
    const slot_array = [_]mlir.MlirAttribute{slot_attr};
    const write_slots_attr = mlir.oraArrayAttrCreate(mlir_ctx, slot_array.len, &slot_array);
    const helper_attrs = [_]mlir.MlirNamedAttribute{
        namedAttr(mlir_ctx, "sym_name", helper_name_attr),
        namedAttr(mlir_ctx, "ora.effect", effect_attr),
        namedAttr(mlir_ctx, "ora.write_slots", write_slots_attr),
    };
    const helper_param_types = [_]mlir.MlirType{ i1_ty, i256_ty };
    const helper_param_locs = [_]mlir.MlirLocation{ loc, loc };
    const helper_func = mlir.oraFuncFuncOpCreate(mlir_ctx, loc, &helper_attrs, helper_attrs.len, &helper_param_types, &helper_param_locs, helper_param_types.len);
    const helper_body = mlir.oraFuncOpGetBodyBlock(helper_func);
    const helper_flag = mlir.oraBlockGetArgument(helper_body, 0);
    const helper_value = mlir.oraBlockGetArgument(helper_body, 1);

    const zero_attr = mlir.oraIntegerAttrCreateI64FromType(i256_ty, 0);
    const zero_op = mlir.oraArithConstantOpCreate(mlir_ctx, loc, i256_ty, zero_attr);
    const zero = mlir.oraOperationGetResult(zero_op, 0);

    const conditional_ret = mlir.oraConditionalReturnOpCreate(mlir_ctx, loc, helper_flag);
    const then_block = mlir.oraConditionalReturnOpGetThenBlock(conditional_ret);
    const else_block = mlir.oraConditionalReturnOpGetElseBlock(conditional_ret);
    mlir.oraBlockAppendOwnedOperation(then_block, mlir.oraSStoreOpCreate(mlir_ctx, loc, helper_value, stringRef("counter")));
    mlir.oraBlockAppendOwnedOperation(then_block, mlir.oraReturnOpCreate(
        mlir_ctx,
        loc,
        &[_]mlir.MlirValue{helper_value},
        1,
    ));
    mlir.oraBlockAppendOwnedOperation(else_block, mlir.oraYieldOpCreate(mlir_ctx, loc, &[_]mlir.MlirValue{}, 0));

    mlir.oraBlockAppendOwnedOperation(helper_body, zero_op);
    mlir.oraBlockAppendOwnedOperation(helper_body, conditional_ret);
    mlir.oraBlockAppendOwnedOperation(helper_body, mlir.oraSStoreOpCreate(mlir_ctx, loc, zero, stringRef("counter")));
    mlir.oraBlockAppendOwnedOperation(helper_body, mlir.oraReturnOpCreate(
        mlir_ctx,
        loc,
        &[_]mlir.MlirValue{zero},
        1,
    ));

    try encoder.registerFunctionOperation(helper_func);

    const call_flag = mlir.oraVariablePlaceholderOpCreate(mlir_ctx, loc, stringRef("conditionalReturnStateFlag"), i1_ty);
    const call_value = mlir.oraVariablePlaceholderOpCreate(mlir_ctx, loc, stringRef("conditionalReturnStateValue"), i256_ty);
    const call_operands = [_]mlir.MlirValue{
        mlir.oraOperationGetResult(call_flag, 0),
        mlir.oraOperationGetResult(call_value, 0),
    };
    const call = mlir.oraFuncCallOpCreate(
        mlir_ctx,
        loc,
        stringRef("returnOrStoreValue"),
        &call_operands,
        call_operands.len,
        &[_]mlir.MlirType{i256_ty},
        1,
    );

    const encoded_result = try encoder.encodeOperation(call);
    const loaded = try encoder.encodeOperation(mlir.oraSLoadOpCreate(mlir_ctx, loc, stringRef("counter"), i256_ty));

    try testing.expect(!encoder.isDegraded());

    const cond_ast = try encoder.encodeValue(mlir.oraOperationGetResult(call_flag, 0));
    const value_ast = try encoder.encodeValue(mlir.oraOperationGetResult(call_value, 0));
    const zero_ast = try encoder.encodeValue(zero);
    const expected = z3.Z3_mk_ite(z3_ctx.ctx, encoder.coerceBoolean(cond_ast), value_ast, zero_ast);

    var solver = try Solver.init(&z3_ctx, testing.allocator);
    defer solver.deinit();
    solver.assert(z3.Z3_mk_not(z3_ctx.ctx, z3.Z3_mk_eq(z3_ctx.ctx, encoded_result, expected)));
    try testing.expectEqual(@as(z3.Z3_lbool, z3.Z3_L_FALSE), solver.check());

    solver.reset();
    solver.assert(z3.Z3_mk_not(z3_ctx.ctx, z3.Z3_mk_eq(z3_ctx.ctx, loaded, expected)));
    try testing.expectEqual(@as(z3.Z3_lbool, z3.Z3_L_FALSE), solver.check());
}

test "known branching pure callee result encodes exactly" {
    var z3_ctx = try Context.init(testing.allocator);
    defer z3_ctx.deinit();

    var encoder = Encoder.init(&z3_ctx, testing.allocator);
    defer encoder.deinit();

    const mlir_ctx = mlir.oraContextCreate();
    defer mlir.oraContextDestroy(mlir_ctx);
    loadAllDialects(mlir_ctx);
    _ = mlir.oraDialectRegister(mlir_ctx);

    const loc = mlir.oraLocationUnknownGet(mlir_ctx);
    const i1_ty = mlir.oraIntegerTypeCreate(mlir_ctx, 1);
    const i256_ty = mlir.oraIntegerTypeCreate(mlir_ctx, 256);

    const helper_name_attr = mlir.oraStringAttrCreate(mlir_ctx, stringRef("chooseValue"));
    const helper_attrs = [_]mlir.MlirNamedAttribute{
        namedAttr(mlir_ctx, "sym_name", helper_name_attr),
    };
    const helper_param_types = [_]mlir.MlirType{ i1_ty, i256_ty };
    const helper_param_locs = [_]mlir.MlirLocation{ loc, loc };
    const helper_func = mlir.oraFuncFuncOpCreate(mlir_ctx, loc, &helper_attrs, helper_attrs.len, &helper_param_types, &helper_param_locs, helper_param_types.len);
    const helper_body = mlir.oraFuncOpGetBodyBlock(helper_func);
    const helper_flag = mlir.oraBlockGetArgument(helper_body, 0);
    const helper_value = mlir.oraBlockGetArgument(helper_body, 1);
    const fallback_attr = mlir.oraIntegerAttrCreateI64FromType(i256_ty, 5);
    const fallback_op = mlir.oraArithConstantOpCreate(mlir_ctx, loc, i256_ty, fallback_attr);
    const fallback_value = mlir.oraOperationGetResult(fallback_op, 0);

    const no_results = [_]mlir.MlirType{};
    const if_op = mlir.oraScfIfOpCreate(mlir_ctx, loc, helper_flag, &no_results, no_results.len, false);
    const then_block = mlir.oraScfIfOpGetThenBlock(if_op);
    const then_ret_vals = [_]mlir.MlirValue{helper_value};
    const then_ret = mlir.oraReturnOpCreate(mlir_ctx, loc, &then_ret_vals, then_ret_vals.len);
    mlir.oraBlockAppendOwnedOperation(then_block, then_ret);

    const helper_ret_vals = [_]mlir.MlirValue{fallback_value};
    const helper_ret = mlir.oraReturnOpCreate(mlir_ctx, loc, &helper_ret_vals, helper_ret_vals.len);
    mlir.oraBlockAppendOwnedOperation(helper_body, fallback_op);
    mlir.oraBlockAppendOwnedOperation(helper_body, if_op);
    mlir.oraBlockAppendOwnedOperation(helper_body, helper_ret);

    try encoder.registerFunctionOperation(helper_func);

    const true_attr = mlir.oraIntegerAttrCreateI64FromType(i1_ty, 1);
    const true_op = mlir.oraArithConstantOpCreate(mlir_ctx, loc, i1_ty, true_attr);
    const arg_attr = mlir.oraIntegerAttrCreateI64FromType(i256_ty, 9);
    const arg_op = mlir.oraArithConstantOpCreate(mlir_ctx, loc, i256_ty, arg_attr);
    const result_types = [_]mlir.MlirType{i256_ty};
    const call_operands = [_]mlir.MlirValue{ mlir.oraOperationGetResult(true_op, 0), mlir.oraOperationGetResult(arg_op, 0) };
    const call_op = mlir.oraFuncCallOpCreate(
        mlir_ctx,
        loc,
        stringRef("chooseValue"),
        &call_operands,
        call_operands.len,
        &result_types,
        result_types.len,
    );

    const encoded = try encoder.encodeOperation(call_op);
    const expected = try encoder.encodeValue(mlir.oraOperationGetResult(arg_op, 0));

    try testing.expect(!encoder.isDegraded());

    var solver = try Solver.init(&z3_ctx, testing.allocator);
    defer solver.deinit();
    solver.assert(z3.Z3_mk_not(z3_ctx.ctx, z3.Z3_mk_eq(z3_ctx.ctx, encoded, expected)));
    try testing.expectEqual(@as(z3.Z3_lbool, z3.Z3_L_FALSE), solver.check());
}

test "known pure callee conditional return result encodes exactly" {
    var z3_ctx = try Context.init(testing.allocator);
    defer z3_ctx.deinit();

    var encoder = Encoder.init(&z3_ctx, testing.allocator);
    defer encoder.deinit();

    const mlir_ctx = mlir.oraContextCreate();
    defer mlir.oraContextDestroy(mlir_ctx);
    loadAllDialects(mlir_ctx);
    _ = mlir.oraDialectRegister(mlir_ctx);

    const loc = mlir.oraLocationUnknownGet(mlir_ctx);
    const i1_ty = mlir.oraIntegerTypeCreate(mlir_ctx, 1);
    const i256_ty = mlir.oraIntegerTypeCreate(mlir_ctx, 256);

    const helper_name_attr = mlir.oraStringAttrCreate(mlir_ctx, stringRef("chooseViaConditionalReturn"));
    const helper_attrs = [_]mlir.MlirNamedAttribute{
        namedAttr(mlir_ctx, "sym_name", helper_name_attr),
    };
    const helper_param_types = [_]mlir.MlirType{ i1_ty, i256_ty };
    const helper_param_locs = [_]mlir.MlirLocation{ loc, loc };
    const helper_func = mlir.oraFuncFuncOpCreate(mlir_ctx, loc, &helper_attrs, helper_attrs.len, &helper_param_types, &helper_param_locs, helper_param_types.len);
    const helper_body = mlir.oraFuncOpGetBodyBlock(helper_func);
    const helper_flag = mlir.oraBlockGetArgument(helper_body, 0);
    const helper_value = mlir.oraBlockGetArgument(helper_body, 1);

    const conditional_ret = mlir.oraConditionalReturnOpCreate(mlir_ctx, loc, helper_flag);
    const then_block = mlir.oraConditionalReturnOpGetThenBlock(conditional_ret);
    const else_block = mlir.oraConditionalReturnOpGetElseBlock(conditional_ret);
    const then_ret_vals = [_]mlir.MlirValue{helper_value};
    mlir.oraBlockAppendOwnedOperation(then_block, mlir.oraReturnOpCreate(mlir_ctx, loc, &then_ret_vals, then_ret_vals.len));
    mlir.oraBlockAppendOwnedOperation(else_block, mlir.oraYieldOpCreate(mlir_ctx, loc, &[_]mlir.MlirValue{}, 0));

    const fallback_attr = mlir.oraIntegerAttrCreateI64FromType(i256_ty, 5);
    const fallback_op = mlir.oraArithConstantOpCreate(mlir_ctx, loc, i256_ty, fallback_attr);
    const fallback_value = mlir.oraOperationGetResult(fallback_op, 0);
    const helper_ret_vals = [_]mlir.MlirValue{fallback_value};
    const helper_ret = mlir.oraReturnOpCreate(mlir_ctx, loc, &helper_ret_vals, helper_ret_vals.len);
    mlir.oraBlockAppendOwnedOperation(helper_body, conditional_ret);
    mlir.oraBlockAppendOwnedOperation(helper_body, fallback_op);
    mlir.oraBlockAppendOwnedOperation(helper_body, helper_ret);

    try encoder.registerFunctionOperation(helper_func);

    const true_attr = mlir.oraIntegerAttrCreateI64FromType(i1_ty, 1);
    const true_op = mlir.oraArithConstantOpCreate(mlir_ctx, loc, i1_ty, true_attr);
    const false_attr = mlir.oraIntegerAttrCreateI64FromType(i1_ty, 0);
    const false_op = mlir.oraArithConstantOpCreate(mlir_ctx, loc, i1_ty, false_attr);
    const arg_attr = mlir.oraIntegerAttrCreateI64FromType(i256_ty, 9);
    const arg_op = mlir.oraArithConstantOpCreate(mlir_ctx, loc, i256_ty, arg_attr);
    const result_types = [_]mlir.MlirType{i256_ty};

    const true_call_operands = [_]mlir.MlirValue{
        mlir.oraOperationGetResult(true_op, 0),
        mlir.oraOperationGetResult(arg_op, 0),
    };
    const true_call = mlir.oraFuncCallOpCreate(
        mlir_ctx,
        loc,
        stringRef("chooseViaConditionalReturn"),
        &true_call_operands,
        true_call_operands.len,
        &result_types,
        result_types.len,
    );

    const false_call_operands = [_]mlir.MlirValue{
        mlir.oraOperationGetResult(false_op, 0),
        mlir.oraOperationGetResult(arg_op, 0),
    };
    const false_call = mlir.oraFuncCallOpCreate(
        mlir_ctx,
        loc,
        stringRef("chooseViaConditionalReturn"),
        &false_call_operands,
        false_call_operands.len,
        &result_types,
        result_types.len,
    );

    const encoded_true = try encoder.encodeOperation(true_call);
    const encoded_false = try encoder.encodeOperation(false_call);
    const expected_true = try encoder.encodeValue(mlir.oraOperationGetResult(arg_op, 0));
    const expected_false = try encoder.encodeIntegerConstant(5, 256);

    try testing.expect(!encoder.isDegraded());

    var solver = try Solver.init(&z3_ctx, testing.allocator);
    defer solver.deinit();
    solver.assert(z3.Z3_mk_not(z3_ctx.ctx, z3.Z3_mk_eq(z3_ctx.ctx, encoded_true, expected_true)));
    try testing.expectEqual(@as(z3.Z3_lbool, z3.Z3_L_FALSE), solver.check());

    solver.reset();
    solver.assert(z3.Z3_mk_not(z3_ctx.ctx, z3.Z3_mk_eq(z3_ctx.ctx, encoded_false, expected_false)));
    try testing.expectEqual(@as(z3.Z3_lbool, z3.Z3_L_FALSE), solver.check());
}

test "known pure callee local memref merge result encodes exactly" {
    var z3_ctx = try Context.init(testing.allocator);
    defer z3_ctx.deinit();

    var encoder = Encoder.init(&z3_ctx, testing.allocator);
    defer encoder.deinit();

    const mlir_ctx = mlir.oraContextCreate();
    defer mlir.oraContextDestroy(mlir_ctx);
    loadAllDialects(mlir_ctx);
    _ = mlir.oraDialectRegister(mlir_ctx);

    const loc = mlir.oraLocationUnknownGet(mlir_ctx);
    const i1_ty = mlir.oraIntegerTypeCreate(mlir_ctx, 1);
    const i256_ty = mlir.oraIntegerTypeCreate(mlir_ctx, 256);
    const null_attr = mlir.MlirAttribute{ .ptr = null };
    const memref_i256_ty = mlir.oraMemRefTypeCreate(mlir_ctx, i256_ty, 0, null, null_attr, null_attr);

    const helper_name_attr = mlir.oraStringAttrCreate(mlir_ctx, stringRef("chooseViaLocalMemref"));
    const helper_attrs = [_]mlir.MlirNamedAttribute{
        namedAttr(mlir_ctx, "sym_name", helper_name_attr),
    };
    const helper_param_types = [_]mlir.MlirType{i1_ty};
    const helper_param_locs = [_]mlir.MlirLocation{loc};
    const helper_func = mlir.oraFuncFuncOpCreate(mlir_ctx, loc, &helper_attrs, helper_attrs.len, &helper_param_types, &helper_param_locs, helper_param_types.len);
    const helper_body = mlir.oraFuncOpGetBodyBlock(helper_func);
    const helper_flag = mlir.oraBlockGetArgument(helper_body, 0);

    const alloca = mlir.oraMemrefAllocaOpCreate(mlir_ctx, loc, memref_i256_ty);
    const alloca_val = mlir.oraOperationGetResult(alloca, 0);

    const no_results = [_]mlir.MlirType{};
    const if_op = mlir.oraScfIfOpCreate(mlir_ctx, loc, helper_flag, &no_results, no_results.len, true);
    const then_block = mlir.oraScfIfOpGetThenBlock(if_op);
    const else_block = mlir.oraScfIfOpGetElseBlock(if_op);

    const nine_attr = mlir.oraIntegerAttrCreateI64FromType(i256_ty, 9);
    const nine_op = mlir.oraArithConstantOpCreate(mlir_ctx, loc, i256_ty, nine_attr);
    mlir.oraBlockAppendOwnedOperation(then_block, nine_op);
    mlir.oraBlockAppendOwnedOperation(then_block, mlir.oraMemrefStoreOpCreate(
        mlir_ctx,
        loc,
        mlir.oraOperationGetResult(nine_op, 0),
        alloca_val,
        &[_]mlir.MlirValue{},
        0,
    ));
    mlir.oraBlockAppendOwnedOperation(then_block, mlir.oraScfYieldOpCreate(mlir_ctx, loc, &[_]mlir.MlirValue{}, 0));

    const five_attr = mlir.oraIntegerAttrCreateI64FromType(i256_ty, 5);
    const five_op = mlir.oraArithConstantOpCreate(mlir_ctx, loc, i256_ty, five_attr);
    mlir.oraBlockAppendOwnedOperation(else_block, five_op);
    mlir.oraBlockAppendOwnedOperation(else_block, mlir.oraMemrefStoreOpCreate(
        mlir_ctx,
        loc,
        mlir.oraOperationGetResult(five_op, 0),
        alloca_val,
        &[_]mlir.MlirValue{},
        0,
    ));
    mlir.oraBlockAppendOwnedOperation(else_block, mlir.oraScfYieldOpCreate(mlir_ctx, loc, &[_]mlir.MlirValue{}, 0));

    const load = mlir.oraMemrefLoadOpCreate(mlir_ctx, loc, alloca_val, &[_]mlir.MlirValue{}, 0, i256_ty);
    const ret = mlir.oraReturnOpCreate(mlir_ctx, loc, &[_]mlir.MlirValue{mlir.oraOperationGetResult(load, 0)}, 1);
    mlir.oraBlockAppendOwnedOperation(helper_body, alloca);
    mlir.oraBlockAppendOwnedOperation(helper_body, if_op);
    mlir.oraBlockAppendOwnedOperation(helper_body, load);
    mlir.oraBlockAppendOwnedOperation(helper_body, ret);

    try encoder.registerFunctionOperation(helper_func);

    const true_attr = mlir.oraIntegerAttrCreateI64FromType(i1_ty, 1);
    const true_op = mlir.oraArithConstantOpCreate(mlir_ctx, loc, i1_ty, true_attr);
    const false_attr = mlir.oraIntegerAttrCreateI64FromType(i1_ty, 0);
    const false_op = mlir.oraArithConstantOpCreate(mlir_ctx, loc, i1_ty, false_attr);
    const result_types = [_]mlir.MlirType{i256_ty};

    const true_call = mlir.oraFuncCallOpCreate(
        mlir_ctx,
        loc,
        stringRef("chooseViaLocalMemref"),
        &[_]mlir.MlirValue{mlir.oraOperationGetResult(true_op, 0)},
        1,
        &result_types,
        result_types.len,
    );
    const false_call = mlir.oraFuncCallOpCreate(
        mlir_ctx,
        loc,
        stringRef("chooseViaLocalMemref"),
        &[_]mlir.MlirValue{mlir.oraOperationGetResult(false_op, 0)},
        1,
        &result_types,
        result_types.len,
    );

    const encoded_true = try encoder.encodeOperation(true_call);
    const encoded_false = try encoder.encodeOperation(false_call);
    const expected_true = try encoder.encodeIntegerConstant(9, 256);
    const expected_false = try encoder.encodeIntegerConstant(5, 256);

    try testing.expect(!encoder.isDegraded());

    var solver = try Solver.init(&z3_ctx, testing.allocator);
    defer solver.deinit();
    solver.assert(z3.Z3_mk_not(z3_ctx.ctx, z3.Z3_mk_eq(z3_ctx.ctx, encoded_true, expected_true)));
    try testing.expectEqual(@as(z3.Z3_lbool, z3.Z3_L_FALSE), solver.check());

    solver.reset();
    solver.assert(z3.Z3_mk_not(z3_ctx.ctx, z3.Z3_mk_eq(z3_ctx.ctx, encoded_false, expected_false)));
    try testing.expectEqual(@as(z3.Z3_lbool, z3.Z3_L_FALSE), solver.check());
}

test "known pure callee conditional return state effects encode exactly" {
    var z3_ctx = try Context.init(testing.allocator);
    defer z3_ctx.deinit();

    var encoder = Encoder.init(&z3_ctx, testing.allocator);
    defer encoder.deinit();

    const mlir_ctx = mlir.oraContextCreate();
    defer mlir.oraContextDestroy(mlir_ctx);
    loadAllDialects(mlir_ctx);
    _ = mlir.oraDialectRegister(mlir_ctx);

    const loc = mlir.oraLocationUnknownGet(mlir_ctx);
    const i1_ty = mlir.oraIntegerTypeCreate(mlir_ctx, 1);
    const i256_ty = mlir.oraIntegerTypeCreate(mlir_ctx, 256);
    const null_attr = mlir.MlirAttribute{ .ptr = null };
    const memref_i256_ty = mlir.oraMemRefTypeCreate(mlir_ctx, i256_ty, 0, null, null_attr, null_attr);

    const helper_name_attr = mlir.oraStringAttrCreate(mlir_ctx, stringRef("chooseViaConditionalState"));
    const helper_attrs = [_]mlir.MlirNamedAttribute{
        namedAttr(mlir_ctx, "sym_name", helper_name_attr),
    };
    const helper_param_types = [_]mlir.MlirType{i1_ty};
    const helper_param_locs = [_]mlir.MlirLocation{loc};
    const helper_func = mlir.oraFuncFuncOpCreate(mlir_ctx, loc, &helper_attrs, helper_attrs.len, &helper_param_types, &helper_param_locs, helper_param_types.len);
    const helper_body = mlir.oraFuncOpGetBodyBlock(helper_func);
    const helper_flag = mlir.oraBlockGetArgument(helper_body, 0);

    const alloca = mlir.oraMemrefAllocaOpCreate(mlir_ctx, loc, memref_i256_ty);
    const alloca_val = mlir.oraOperationGetResult(alloca, 0);
    const five_attr = mlir.oraIntegerAttrCreateI64FromType(i256_ty, 5);
    const five_op = mlir.oraArithConstantOpCreate(mlir_ctx, loc, i256_ty, five_attr);
    const init_store = mlir.oraMemrefStoreOpCreate(
        mlir_ctx,
        loc,
        mlir.oraOperationGetResult(five_op, 0),
        alloca_val,
        &[_]mlir.MlirValue{},
        0,
    );

    const conditional = mlir.oraConditionalReturnOpCreate(mlir_ctx, loc, helper_flag);
    const then_block = mlir.oraConditionalReturnOpGetThenBlock(conditional);
    const else_block = mlir.oraConditionalReturnOpGetElseBlock(conditional);

    const nine_attr = mlir.oraIntegerAttrCreateI64FromType(i256_ty, 9);
    const nine_op = mlir.oraArithConstantOpCreate(mlir_ctx, loc, i256_ty, nine_attr);
    mlir.oraBlockAppendOwnedOperation(then_block, nine_op);
    mlir.oraBlockAppendOwnedOperation(then_block, mlir.oraReturnOpCreate(
        mlir_ctx,
        loc,
        &[_]mlir.MlirValue{mlir.oraOperationGetResult(nine_op, 0)},
        1,
    ));
    mlir.oraBlockAppendOwnedOperation(else_block, mlir.oraYieldOpCreate(mlir_ctx, loc, &[_]mlir.MlirValue{}, 0));

    const load = mlir.oraMemrefLoadOpCreate(mlir_ctx, loc, alloca_val, &[_]mlir.MlirValue{}, 0, i256_ty);
    const ret = mlir.oraReturnOpCreate(mlir_ctx, loc, &[_]mlir.MlirValue{mlir.oraOperationGetResult(load, 0)}, 1);
    mlir.oraBlockAppendOwnedOperation(helper_body, alloca);
    mlir.oraBlockAppendOwnedOperation(helper_body, five_op);
    mlir.oraBlockAppendOwnedOperation(helper_body, init_store);
    mlir.oraBlockAppendOwnedOperation(helper_body, conditional);
    mlir.oraBlockAppendOwnedOperation(helper_body, load);
    mlir.oraBlockAppendOwnedOperation(helper_body, ret);

    try encoder.registerFunctionOperation(helper_func);

    const true_attr = mlir.oraIntegerAttrCreateI64FromType(i1_ty, 1);
    const true_op = mlir.oraArithConstantOpCreate(mlir_ctx, loc, i1_ty, true_attr);
    const false_attr = mlir.oraIntegerAttrCreateI64FromType(i1_ty, 0);
    const false_op = mlir.oraArithConstantOpCreate(mlir_ctx, loc, i1_ty, false_attr);
    const result_types = [_]mlir.MlirType{i256_ty};

    const true_call = mlir.oraFuncCallOpCreate(
        mlir_ctx,
        loc,
        stringRef("chooseViaConditionalState"),
        &[_]mlir.MlirValue{mlir.oraOperationGetResult(true_op, 0)},
        1,
        &result_types,
        result_types.len,
    );
    const false_call = mlir.oraFuncCallOpCreate(
        mlir_ctx,
        loc,
        stringRef("chooseViaConditionalState"),
        &[_]mlir.MlirValue{mlir.oraOperationGetResult(false_op, 0)},
        1,
        &result_types,
        result_types.len,
    );

    const encoded_true = try encoder.encodeOperation(true_call);
    const encoded_false = try encoder.encodeOperation(false_call);
    const expected_true = try encoder.encodeIntegerConstant(9, 256);
    const expected_false = try encoder.encodeIntegerConstant(5, 256);

    try testing.expect(!encoder.isDegraded());

    var solver = try Solver.init(&z3_ctx, testing.allocator);
    defer solver.deinit();
    solver.assert(z3.Z3_mk_not(z3_ctx.ctx, z3.Z3_mk_eq(z3_ctx.ctx, encoded_true, expected_true)));
    try testing.expectEqual(@as(z3.Z3_lbool, z3.Z3_L_FALSE), solver.check());

    solver.reset();
    solver.assert(z3.Z3_mk_not(z3_ctx.ctx, z3.Z3_mk_eq(z3_ctx.ctx, encoded_false, expected_false)));
    try testing.expectEqual(@as(z3.Z3_lbool, z3.Z3_L_FALSE), solver.check());
}

test "known pure callee switch_expr state effects encode exactly" {
    var z3_ctx = try Context.init(testing.allocator);
    defer z3_ctx.deinit();

    var encoder = Encoder.init(&z3_ctx, testing.allocator);
    defer encoder.deinit();

    const mlir_ctx = mlir.oraContextCreate();
    defer mlir.oraContextDestroy(mlir_ctx);
    loadAllDialects(mlir_ctx);
    _ = mlir.oraDialectRegister(mlir_ctx);

    const loc = mlir.oraLocationUnknownGet(mlir_ctx);
    const i1_ty = mlir.oraIntegerTypeCreate(mlir_ctx, 1);
    const i256_ty = mlir.oraIntegerTypeCreate(mlir_ctx, 256);
    const null_attr = mlir.MlirAttribute{ .ptr = null };
    const memref_i256_ty = mlir.oraMemRefTypeCreate(mlir_ctx, i256_ty, 0, null, null_attr, null_attr);

    const helper_name_attr = mlir.oraStringAttrCreate(mlir_ctx, stringRef("chooseViaSwitchExprState"));
    const helper_attrs = [_]mlir.MlirNamedAttribute{
        namedAttr(mlir_ctx, "sym_name", helper_name_attr),
    };
    const helper_param_types = [_]mlir.MlirType{i1_ty};
    const helper_param_locs = [_]mlir.MlirLocation{loc};
    const helper_func = mlir.oraFuncFuncOpCreate(mlir_ctx, loc, &helper_attrs, helper_attrs.len, &helper_param_types, &helper_param_locs, helper_param_types.len);
    const helper_body = mlir.oraFuncOpGetBodyBlock(helper_func);
    const helper_flag = mlir.oraBlockGetArgument(helper_body, 0);

    const alloca = mlir.oraMemrefAllocaOpCreate(mlir_ctx, loc, memref_i256_ty);
    const alloca_val = mlir.oraOperationGetResult(alloca, 0);

    const switch_expr = mlir.oraSwitchExprOpCreateWithCases(
        mlir_ctx,
        loc,
        helper_flag,
        &[_]mlir.MlirType{i256_ty},
        1,
        2,
    );
    const case_values = [_]i64{ 0, 1 };
    const range_starts = [_]i64{ 0, 0 };
    const range_ends = [_]i64{ 0, 0 };
    const case_kinds = [_]i64{ 0, 0 };
    mlir.oraSwitchOpSetCasePatterns(
        switch_expr,
        &case_values,
        &range_starts,
        &range_ends,
        &case_kinds,
        -1,
        case_values.len,
    );

    const false_block = mlir.oraSwitchExprOpGetCaseBlock(switch_expr, 0);
    const true_block = mlir.oraSwitchExprOpGetCaseBlock(switch_expr, 1);

    const five_attr = mlir.oraIntegerAttrCreateI64FromType(i256_ty, 5);
    const five_op = mlir.oraArithConstantOpCreate(mlir_ctx, loc, i256_ty, five_attr);
    mlir.oraBlockAppendOwnedOperation(false_block, five_op);
    mlir.oraBlockAppendOwnedOperation(false_block, mlir.oraMemrefStoreOpCreate(
        mlir_ctx,
        loc,
        mlir.oraOperationGetResult(five_op, 0),
        alloca_val,
        &[_]mlir.MlirValue{},
        0,
    ));
    mlir.oraBlockAppendOwnedOperation(false_block, mlir.oraYieldOpCreate(
        mlir_ctx,
        loc,
        &[_]mlir.MlirValue{mlir.oraOperationGetResult(five_op, 0)},
        1,
    ));

    const nine_attr = mlir.oraIntegerAttrCreateI64FromType(i256_ty, 9);
    const nine_op = mlir.oraArithConstantOpCreate(mlir_ctx, loc, i256_ty, nine_attr);
    mlir.oraBlockAppendOwnedOperation(true_block, nine_op);
    mlir.oraBlockAppendOwnedOperation(true_block, mlir.oraMemrefStoreOpCreate(
        mlir_ctx,
        loc,
        mlir.oraOperationGetResult(nine_op, 0),
        alloca_val,
        &[_]mlir.MlirValue{},
        0,
    ));
    mlir.oraBlockAppendOwnedOperation(true_block, mlir.oraYieldOpCreate(
        mlir_ctx,
        loc,
        &[_]mlir.MlirValue{mlir.oraOperationGetResult(nine_op, 0)},
        1,
    ));

    const load = mlir.oraMemrefLoadOpCreate(mlir_ctx, loc, alloca_val, &[_]mlir.MlirValue{}, 0, i256_ty);
    const ret = mlir.oraReturnOpCreate(mlir_ctx, loc, &[_]mlir.MlirValue{mlir.oraOperationGetResult(load, 0)}, 1);
    mlir.oraBlockAppendOwnedOperation(helper_body, alloca);
    mlir.oraBlockAppendOwnedOperation(helper_body, switch_expr);
    mlir.oraBlockAppendOwnedOperation(helper_body, load);
    mlir.oraBlockAppendOwnedOperation(helper_body, ret);

    try encoder.registerFunctionOperation(helper_func);

    const true_attr = mlir.oraIntegerAttrCreateI64FromType(i1_ty, 1);
    const true_op = mlir.oraArithConstantOpCreate(mlir_ctx, loc, i1_ty, true_attr);
    const false_attr = mlir.oraIntegerAttrCreateI64FromType(i1_ty, 0);
    const false_op = mlir.oraArithConstantOpCreate(mlir_ctx, loc, i1_ty, false_attr);
    const result_types = [_]mlir.MlirType{i256_ty};

    const true_call = mlir.oraFuncCallOpCreate(
        mlir_ctx,
        loc,
        stringRef("chooseViaSwitchExprState"),
        &[_]mlir.MlirValue{mlir.oraOperationGetResult(true_op, 0)},
        1,
        &result_types,
        result_types.len,
    );
    const false_call = mlir.oraFuncCallOpCreate(
        mlir_ctx,
        loc,
        stringRef("chooseViaSwitchExprState"),
        &[_]mlir.MlirValue{mlir.oraOperationGetResult(false_op, 0)},
        1,
        &result_types,
        result_types.len,
    );

    const encoded_true = try encoder.encodeOperation(true_call);
    const encoded_false = try encoder.encodeOperation(false_call);
    const expected_true = try encoder.encodeIntegerConstant(9, 256);
    const expected_false = try encoder.encodeIntegerConstant(5, 256);

    try testing.expect(!encoder.isDegraded());

    var solver = try Solver.init(&z3_ctx, testing.allocator);
    defer solver.deinit();
    solver.assert(z3.Z3_mk_not(z3_ctx.ctx, z3.Z3_mk_eq(z3_ctx.ctx, encoded_true, expected_true)));
    try testing.expectEqual(@as(z3.Z3_lbool, z3.Z3_L_FALSE), solver.check());

    solver.reset();
    solver.assert(z3.Z3_mk_not(z3_ctx.ctx, z3.Z3_mk_eq(z3_ctx.ctx, encoded_false, expected_false)));
    try testing.expectEqual(@as(z3.Z3_lbool, z3.Z3_L_FALSE), solver.check());
}

test "known pure callee non-throwing ora.try_stmt local memref result encodes exactly" {
    var z3_ctx = try Context.init(testing.allocator);
    defer z3_ctx.deinit();

    var encoder = Encoder.init(&z3_ctx, testing.allocator);
    defer encoder.deinit();

    const mlir_ctx = mlir.oraContextCreate();
    defer mlir.oraContextDestroy(mlir_ctx);
    loadAllDialects(mlir_ctx);
    _ = mlir.oraDialectRegister(mlir_ctx);

    const loc = mlir.oraLocationUnknownGet(mlir_ctx);
    const i256_ty = mlir.oraIntegerTypeCreate(mlir_ctx, 256);
    const null_attr = mlir.MlirAttribute{ .ptr = null };
    const memref_i256_ty = mlir.oraMemRefTypeCreate(mlir_ctx, i256_ty, 0, null, null_attr, null_attr);

    const helper_attrs = [_]mlir.MlirNamedAttribute{
        namedAttr(mlir_ctx, "sym_name", mlir.oraStringAttrCreate(mlir_ctx, stringRef("tryLocalMemref"))),
    };
    const helper = mlir.oraFuncFuncOpCreate(mlir_ctx, loc, &helper_attrs, helper_attrs.len, &[_]mlir.MlirType{}, &[_]mlir.MlirLocation{}, 0);
    const body = mlir.oraFuncOpGetBodyBlock(helper);

    const alloca = mlir.oraMemrefAllocaOpCreate(mlir_ctx, loc, memref_i256_ty);
    const slot = mlir.oraOperationGetResult(alloca, 0);
    mlir.oraBlockAppendOwnedOperation(body, alloca);

    const init_attr = mlir.oraIntegerAttrCreateI64FromType(i256_ty, 0);
    const init_op = mlir.oraArithConstantOpCreate(mlir_ctx, loc, i256_ty, init_attr);
    mlir.oraBlockAppendOwnedOperation(body, init_op);
    mlir.oraBlockAppendOwnedOperation(body, mlir.oraMemrefStoreOpCreate(
        mlir_ctx,
        loc,
        mlir.oraOperationGetResult(init_op, 0),
        slot,
        &[_]mlir.MlirValue{},
        0,
    ));

    const try_stmt = mlir.oraTryStmtOpCreate(mlir_ctx, loc, &[_]mlir.MlirType{}, 0);
    const try_block = mlir.oraTryStmtOpGetTryBlock(try_stmt);
    const catch_block = mlir.oraTryStmtOpGetCatchBlock(try_stmt);
    const store_attr = mlir.oraIntegerAttrCreateI64FromType(i256_ty, 7);
    const store_op = mlir.oraArithConstantOpCreate(mlir_ctx, loc, i256_ty, store_attr);
    mlir.oraBlockAppendOwnedOperation(try_block, store_op);
    mlir.oraBlockAppendOwnedOperation(try_block, mlir.oraMemrefStoreOpCreate(
        mlir_ctx,
        loc,
        mlir.oraOperationGetResult(store_op, 0),
        slot,
        &[_]mlir.MlirValue{},
        0,
    ));
    mlir.oraBlockAppendOwnedOperation(try_block, mlir.oraYieldOpCreate(mlir_ctx, loc, &[_]mlir.MlirValue{}, 0));
    mlir.oraBlockAppendOwnedOperation(catch_block, mlir.oraYieldOpCreate(mlir_ctx, loc, &[_]mlir.MlirValue{}, 0));
    mlir.oraBlockAppendOwnedOperation(body, try_stmt);

    const final_load = mlir.oraMemrefLoadOpCreate(mlir_ctx, loc, slot, &[_]mlir.MlirValue{}, 0, i256_ty);
    mlir.oraBlockAppendOwnedOperation(body, final_load);
    mlir.oraBlockAppendOwnedOperation(body, mlir.oraReturnOpCreate(
        mlir_ctx,
        loc,
        &[_]mlir.MlirValue{mlir.oraOperationGetResult(final_load, 0)},
        1,
    ));

    try encoder.registerFunctionOperation(helper);

    const call = mlir.oraFuncCallOpCreate(
        mlir_ctx,
        loc,
        stringRef("tryLocalMemref"),
        &[_]mlir.MlirValue{},
        0,
        &[_]mlir.MlirType{i256_ty},
        1,
    );

    const encoded = try encoder.encodeOperation(call);
    try testing.expect(!encoder.isDegraded());

    const expected = z3.Z3_mk_numeral(z3_ctx.ctx, "7", z3.Z3_mk_bv_sort(z3_ctx.ctx, 256));
    var solver = try Solver.init(&z3_ctx, testing.allocator);
    defer solver.deinit();
    solver.assert(z3.Z3_mk_not(z3_ctx.ctx, z3.Z3_mk_eq(z3_ctx.ctx, encoded, expected)));
    try testing.expectEqual(@as(z3.Z3_lbool, z3.Z3_L_FALSE), solver.check());
}

test "known pure callee branch-local memref initialization encodes exactly" {
    var z3_ctx = try Context.init(testing.allocator);
    defer z3_ctx.deinit();

    var encoder = Encoder.init(&z3_ctx, testing.allocator);
    defer encoder.deinit();

    const mlir_ctx = mlir.oraContextCreate();
    defer mlir.oraContextDestroy(mlir_ctx);
    loadAllDialects(mlir_ctx);
    _ = mlir.oraDialectRegister(mlir_ctx);

    const loc = mlir.oraLocationUnknownGet(mlir_ctx);
    const i1_ty = mlir.oraIntegerTypeCreate(mlir_ctx, 1);
    const i256_ty = mlir.oraIntegerTypeCreate(mlir_ctx, 256);
    const null_attr = mlir.MlirAttribute{ .ptr = null };
    const memref_i256_ty = mlir.oraMemRefTypeCreate(mlir_ctx, i256_ty, 0, null, null_attr, null_attr);

    const helper_attrs = [_]mlir.MlirNamedAttribute{
        namedAttr(mlir_ctx, "sym_name", mlir.oraStringAttrCreate(mlir_ctx, stringRef("branchInitThenLoad"))),
    };
    const helper_param_types = [_]mlir.MlirType{i1_ty};
    const helper_param_locs = [_]mlir.MlirLocation{loc};
    const helper = mlir.oraFuncFuncOpCreate(mlir_ctx, loc, &helper_attrs, helper_attrs.len, &helper_param_types, &helper_param_locs, helper_param_types.len);
    const body = mlir.oraFuncOpGetBodyBlock(helper);
    const flag = mlir.oraBlockGetArgument(body, 0);

    const alloca = mlir.oraMemrefAllocaOpCreate(mlir_ctx, loc, memref_i256_ty);
    const slot = mlir.oraOperationGetResult(alloca, 0);

    const no_results = [_]mlir.MlirType{};
    const if_op = mlir.oraScfIfOpCreate(mlir_ctx, loc, flag, &no_results, no_results.len, true);
    const then_block = mlir.oraScfIfOpGetThenBlock(if_op);
    const else_block = mlir.oraScfIfOpGetElseBlock(if_op);

    const nine_attr = mlir.oraIntegerAttrCreateI64FromType(i256_ty, 9);
    const nine_op = mlir.oraArithConstantOpCreate(mlir_ctx, loc, i256_ty, nine_attr);
    mlir.oraBlockAppendOwnedOperation(then_block, nine_op);
    mlir.oraBlockAppendOwnedOperation(then_block, mlir.oraMemrefStoreOpCreate(
        mlir_ctx,
        loc,
        mlir.oraOperationGetResult(nine_op, 0),
        slot,
        &[_]mlir.MlirValue{},
        0,
    ));
    const then_load = mlir.oraMemrefLoadOpCreate(mlir_ctx, loc, slot, &[_]mlir.MlirValue{}, 0, i256_ty);
    mlir.oraBlockAppendOwnedOperation(then_block, then_load);
    mlir.oraBlockAppendOwnedOperation(then_block, mlir.oraReturnOpCreate(
        mlir_ctx,
        loc,
        &[_]mlir.MlirValue{mlir.oraOperationGetResult(then_load, 0)},
        1,
    ));

    const five_attr = mlir.oraIntegerAttrCreateI64FromType(i256_ty, 5);
    const five_op = mlir.oraArithConstantOpCreate(mlir_ctx, loc, i256_ty, five_attr);
    mlir.oraBlockAppendOwnedOperation(else_block, five_op);
    mlir.oraBlockAppendOwnedOperation(else_block, mlir.oraReturnOpCreate(
        mlir_ctx,
        loc,
        &[_]mlir.MlirValue{mlir.oraOperationGetResult(five_op, 0)},
        1,
    ));

    mlir.oraBlockAppendOwnedOperation(body, alloca);
    mlir.oraBlockAppendOwnedOperation(body, if_op);

    try encoder.registerFunctionOperation(helper);

    const true_attr = mlir.oraIntegerAttrCreateI64FromType(i1_ty, 1);
    const true_op = mlir.oraArithConstantOpCreate(mlir_ctx, loc, i1_ty, true_attr);
    const false_attr = mlir.oraIntegerAttrCreateI64FromType(i1_ty, 0);
    const false_op = mlir.oraArithConstantOpCreate(mlir_ctx, loc, i1_ty, false_attr);
    const result_types = [_]mlir.MlirType{i256_ty};

    const true_call = mlir.oraFuncCallOpCreate(
        mlir_ctx,
        loc,
        stringRef("branchInitThenLoad"),
        &[_]mlir.MlirValue{mlir.oraOperationGetResult(true_op, 0)},
        1,
        &result_types,
        result_types.len,
    );
    const false_call = mlir.oraFuncCallOpCreate(
        mlir_ctx,
        loc,
        stringRef("branchInitThenLoad"),
        &[_]mlir.MlirValue{mlir.oraOperationGetResult(false_op, 0)},
        1,
        &result_types,
        result_types.len,
    );

    const encoded_true = try encoder.encodeOperation(true_call);
    const encoded_false = try encoder.encodeOperation(false_call);
    const expected_true = try encoder.encodeIntegerConstant(9, 256);
    const expected_false = try encoder.encodeIntegerConstant(5, 256);

    try testing.expect(!encoder.isDegraded());

    var solver = try Solver.init(&z3_ctx, testing.allocator);
    defer solver.deinit();
    solver.assert(z3.Z3_mk_not(z3_ctx.ctx, z3.Z3_mk_eq(z3_ctx.ctx, encoded_true, expected_true)));
    try testing.expectEqual(@as(z3.Z3_lbool, z3.Z3_L_FALSE), solver.check());

    solver.reset();
    solver.assert(z3.Z3_mk_not(z3_ctx.ctx, z3.Z3_mk_eq(z3_ctx.ctx, encoded_false, expected_false)));
    try testing.expectEqual(@as(z3.Z3_lbool, z3.Z3_L_FALSE), solver.check());
}

test "known pure callee non-throwing ora.try_stmt does not preempt later explicit return" {
    var z3_ctx = try Context.init(testing.allocator);
    defer z3_ctx.deinit();

    var encoder = Encoder.init(&z3_ctx, testing.allocator);
    defer encoder.deinit();

    const mlir_ctx = mlir.oraContextCreate();
    defer mlir.oraContextDestroy(mlir_ctx);
    loadAllDialects(mlir_ctx);
    _ = mlir.oraDialectRegister(mlir_ctx);

    const loc = mlir.oraLocationUnknownGet(mlir_ctx);
    const i256_ty = mlir.oraIntegerTypeCreate(mlir_ctx, 256);

    const helper_attrs = [_]mlir.MlirNamedAttribute{
        namedAttr(mlir_ctx, "sym_name", mlir.oraStringAttrCreate(mlir_ctx, stringRef("tryThenReturn"))),
    };
    const helper = mlir.oraFuncFuncOpCreate(mlir_ctx, loc, &helper_attrs, helper_attrs.len, &[_]mlir.MlirType{}, &[_]mlir.MlirLocation{}, 0);
    const body = mlir.oraFuncOpGetBodyBlock(helper);

    const try_stmt = mlir.oraTryStmtOpCreate(mlir_ctx, loc, &[_]mlir.MlirType{i256_ty}, 1);
    const try_block = mlir.oraTryStmtOpGetTryBlock(try_stmt);
    const catch_block = mlir.oraTryStmtOpGetCatchBlock(try_stmt);

    const try_attr = mlir.oraIntegerAttrCreateI64FromType(i256_ty, 7);
    const try_op = mlir.oraArithConstantOpCreate(mlir_ctx, loc, i256_ty, try_attr);
    mlir.oraBlockAppendOwnedOperation(try_block, try_op);
    mlir.oraBlockAppendOwnedOperation(try_block, mlir.oraYieldOpCreate(
        mlir_ctx,
        loc,
        &[_]mlir.MlirValue{mlir.oraOperationGetResult(try_op, 0)},
        1,
    ));

    const catch_attr = mlir.oraIntegerAttrCreateI64FromType(i256_ty, 99);
    const catch_op = mlir.oraArithConstantOpCreate(mlir_ctx, loc, i256_ty, catch_attr);
    mlir.oraBlockAppendOwnedOperation(catch_block, catch_op);
    mlir.oraBlockAppendOwnedOperation(catch_block, mlir.oraYieldOpCreate(
        mlir_ctx,
        loc,
        &[_]mlir.MlirValue{mlir.oraOperationGetResult(catch_op, 0)},
        1,
    ));

    mlir.oraBlockAppendOwnedOperation(body, try_stmt);

    const ret_attr = mlir.oraIntegerAttrCreateI64FromType(i256_ty, 42);
    const ret_op = mlir.oraArithConstantOpCreate(mlir_ctx, loc, i256_ty, ret_attr);
    mlir.oraBlockAppendOwnedOperation(body, ret_op);
    mlir.oraBlockAppendOwnedOperation(body, mlir.oraReturnOpCreate(
        mlir_ctx,
        loc,
        &[_]mlir.MlirValue{mlir.oraOperationGetResult(ret_op, 0)},
        1,
    ));

    try encoder.registerFunctionOperation(helper);

    const call = mlir.oraFuncCallOpCreate(
        mlir_ctx,
        loc,
        stringRef("tryThenReturn"),
        &[_]mlir.MlirValue{},
        0,
        &[_]mlir.MlirType{i256_ty},
        1,
    );

    const encoded = try encoder.encodeOperation(call);
    const expected = try encoder.encodeOperation(ret_op);
    try testing.expect(!encoder.isDegraded());

    var solver = try Solver.init(&z3_ctx, testing.allocator);
    defer solver.deinit();
    solver.assert(z3.Z3_mk_not(z3_ctx.ctx, z3.Z3_mk_eq(z3_ctx.ctx, encoded, expected)));
    try testing.expectEqual(@as(z3.Z3_lbool, z3.Z3_L_FALSE), solver.check());
}

test "known callee result degradation reports callee and callsite" {
    var z3_ctx = try Context.init(testing.allocator);
    defer z3_ctx.deinit();

    var encoder = Encoder.init(&z3_ctx, testing.allocator);
    defer encoder.deinit();

    const mlir_ctx = mlir.oraContextCreate();
    defer mlir.oraContextDestroy(mlir_ctx);
    loadAllDialects(mlir_ctx);
    _ = mlir.oraDialectRegister(mlir_ctx);

    const loc = mlir.oraLocationFileLineColGet(mlir_ctx, stringRef("/tmp/debug.ora"), 42, 7);
    const i256_ty = mlir.oraIntegerTypeCreate(mlir_ctx, 256);

    const helper_name = "opaqueWhilePure";
    const helper_attrs = [_]mlir.MlirNamedAttribute{
        namedAttr(mlir_ctx, "sym_name", mlir.oraStringAttrCreate(mlir_ctx, stringRef(helper_name))),
    };
    const helper = mlir.oraFuncFuncOpCreate(mlir_ctx, loc, &helper_attrs, helper_attrs.len, &[_]mlir.MlirType{}, &[_]mlir.MlirLocation{}, 0);
    const body = mlir.oraFuncOpGetBodyBlock(helper);

    const init_attr = mlir.oraIntegerAttrCreateI64FromType(i256_ty, 7);
    const init_op = mlir.oraArithConstantOpCreate(mlir_ctx, loc, i256_ty, init_attr);
    mlir.oraBlockAppendOwnedOperation(body, init_op);
    const while_op = mlir.oraScfWhileOpCreate(
        mlir_ctx,
        loc,
        &[_]mlir.MlirValue{mlir.oraOperationGetResult(init_op, 0)},
        1,
        &[_]mlir.MlirType{i256_ty},
        1,
    );
    mlir.oraBlockAppendOwnedOperation(body, while_op);
    mlir.oraBlockAppendOwnedOperation(body, mlir.oraReturnOpCreate(
        mlir_ctx,
        loc,
        &[_]mlir.MlirValue{mlir.oraOperationGetResult(while_op, 0)},
        1,
    ));

    try encoder.registerFunctionOperation(helper);

    const call = mlir.oraFuncCallOpCreate(
        mlir_ctx,
        loc,
        stringRef(helper_name),
        &[_]mlir.MlirValue{},
        0,
        &[_]mlir.MlirType{i256_ty},
        1,
    );
    _ = try encoder.encodeOperation(call);

    try testing.expect(encoder.isDegraded());
    const reason = encoder.degradationReason().?;
    try testing.expect(std.mem.containsAtLeast(u8, reason, 1, helper_name));
    try testing.expect(std.mem.containsAtLeast(u8, reason, 1, "/tmp/debug.ora"));
    try testing.expect(std.mem.containsAtLeast(u8, reason, 1, "42:7"));
    try testing.expect(
        std.mem.containsAtLeast(u8, reason, 1, "opaque summary") or
            std.mem.containsAtLeast(u8, reason, 1, "known callee") or
            std.mem.containsAtLeast(u8, reason, 1, "structured control") or
            std.mem.containsAtLeast(u8, reason, 1, "loop state summary"),
    );
}

test "known pure callee first-iteration scf.for return encodes exactly" {
    var z3_ctx = try Context.init(testing.allocator);
    defer z3_ctx.deinit();

    var encoder = Encoder.init(&z3_ctx, testing.allocator);
    defer encoder.deinit();

    const mlir_ctx = mlir.oraContextCreate();
    defer mlir.oraContextDestroy(mlir_ctx);
    loadAllDialects(mlir_ctx);
    _ = mlir.oraDialectRegister(mlir_ctx);

    const loc = mlir.oraLocationUnknownGet(mlir_ctx);
    const index_ty = mlir.oraIndexTypeCreate(mlir_ctx);
    const i256_ty = mlir.oraIntegerTypeCreate(mlir_ctx, 256);

    const helper_attrs = [_]mlir.MlirNamedAttribute{
        namedAttr(mlir_ctx, "sym_name", mlir.oraStringAttrCreate(mlir_ctx, stringRef("firstForReturn"))),
    };
    const helper = mlir.oraFuncFuncOpCreate(mlir_ctx, loc, &helper_attrs, helper_attrs.len, &[_]mlir.MlirType{}, &[_]mlir.MlirLocation{}, 0);
    const body = mlir.oraFuncOpGetBodyBlock(helper);

    const c0 = mlir.oraArithConstantOpCreate(mlir_ctx, loc, index_ty, mlir.oraIntegerAttrCreateI64FromType(index_ty, 0));
    const c1 = mlir.oraArithConstantOpCreate(mlir_ctx, loc, index_ty, mlir.oraIntegerAttrCreateI64FromType(index_ty, 1));
    const c4 = mlir.oraArithConstantOpCreate(mlir_ctx, loc, index_ty, mlir.oraIntegerAttrCreateI64FromType(index_ty, 4));
    mlir.oraBlockAppendOwnedOperation(body, c0);
    mlir.oraBlockAppendOwnedOperation(body, c1);
    mlir.oraBlockAppendOwnedOperation(body, c4);

    const loop = mlir.oraScfForOpCreate(
        mlir_ctx,
        loc,
        mlir.oraOperationGetResult(c0, 0),
        mlir.oraOperationGetResult(c4, 0),
        mlir.oraOperationGetResult(c1, 0),
        &[_]mlir.MlirValue{},
        0,
        false,
    );
    const loop_body = mlir.oraScfForOpGetBodyBlock(loop);
    const seven = mlir.oraArithConstantOpCreate(mlir_ctx, loc, i256_ty, mlir.oraIntegerAttrCreateI64FromType(i256_ty, 7));
    mlir.oraBlockAppendOwnedOperation(loop_body, seven);
    mlir.oraBlockAppendOwnedOperation(loop_body, mlir.oraReturnOpCreate(
        mlir_ctx,
        loc,
        &[_]mlir.MlirValue{mlir.oraOperationGetResult(seven, 0)},
        1,
    ));
    mlir.oraBlockAppendOwnedOperation(body, loop);

    try encoder.registerFunctionOperation(helper);

    const call = mlir.oraFuncCallOpCreate(
        mlir_ctx,
        loc,
        stringRef("firstForReturn"),
        &[_]mlir.MlirValue{},
        0,
        &[_]mlir.MlirType{i256_ty},
        1,
    );
    const encoded = try encoder.encodeOperation(call);
    const expected = try encoder.encodeIntegerConstant(7, 256);

    try testing.expect(!encoder.isDegraded());

    var solver = try Solver.init(&z3_ctx, testing.allocator);
    defer solver.deinit();
    solver.assert(z3.Z3_mk_not(z3_ctx.ctx, z3.Z3_mk_eq(z3_ctx.ctx, encoded, expected)));
    try testing.expectEqual(@as(z3.Z3_lbool, z3.Z3_L_FALSE), solver.check());
}

test "known pure callee first-iteration scf.while return encodes exactly" {
    var z3_ctx = try Context.init(testing.allocator);
    defer z3_ctx.deinit();

    var encoder = Encoder.init(&z3_ctx, testing.allocator);
    defer encoder.deinit();

    const mlir_ctx = mlir.oraContextCreate();
    defer mlir.oraContextDestroy(mlir_ctx);
    loadAllDialects(mlir_ctx);
    _ = mlir.oraDialectRegister(mlir_ctx);

    const loc = mlir.oraLocationUnknownGet(mlir_ctx);
    const i1_ty = mlir.oraIntegerTypeCreate(mlir_ctx, 1);
    const i256_ty = mlir.oraIntegerTypeCreate(mlir_ctx, 256);

    const helper_attrs = [_]mlir.MlirNamedAttribute{
        namedAttr(mlir_ctx, "sym_name", mlir.oraStringAttrCreate(mlir_ctx, stringRef("firstWhileReturn"))),
    };
    const helper = mlir.oraFuncFuncOpCreate(mlir_ctx, loc, &helper_attrs, helper_attrs.len, &[_]mlir.MlirType{}, &[_]mlir.MlirLocation{}, 0);
    const body = mlir.oraFuncOpGetBodyBlock(helper);

    const true_op = mlir.oraArithConstantOpCreate(mlir_ctx, loc, i1_ty, mlir.oraIntegerAttrCreateI64FromType(i1_ty, 1));
    mlir.oraBlockAppendOwnedOperation(body, true_op);

    const while_op = mlir.oraScfWhileOpCreate(mlir_ctx, loc, &[_]mlir.MlirValue{}, 0, &[_]mlir.MlirType{}, 0);
    const before_block = mlir.oraScfWhileOpGetBeforeBlock(while_op);
    const after_block = mlir.oraScfWhileOpGetAfterBlock(while_op);
    mlir.oraBlockAppendOwnedOperation(before_block, mlir.oraScfConditionOpCreate(
        mlir_ctx,
        loc,
        mlir.oraOperationGetResult(true_op, 0),
        &[_]mlir.MlirValue{},
        0,
    ));

    const seven_op = mlir.oraArithConstantOpCreate(mlir_ctx, loc, i256_ty, mlir.oraIntegerAttrCreateI64FromType(i256_ty, 7));
    mlir.oraBlockAppendOwnedOperation(after_block, seven_op);
    mlir.oraBlockAppendOwnedOperation(after_block, mlir.oraReturnOpCreate(
        mlir_ctx,
        loc,
        &[_]mlir.MlirValue{mlir.oraOperationGetResult(seven_op, 0)},
        1,
    ));

    mlir.oraBlockAppendOwnedOperation(body, while_op);

    try encoder.registerFunctionOperation(helper);

    const call = mlir.oraFuncCallOpCreate(
        mlir_ctx,
        loc,
        stringRef("firstWhileReturn"),
        &[_]mlir.MlirValue{},
        0,
        &[_]mlir.MlirType{i256_ty},
        1,
    );
    const encoded = try encoder.encodeOperation(call);
    const expected = try encoder.encodeIntegerConstant(7, 256);

    try testing.expect(!encoder.isDegraded());

    var solver = try Solver.init(&z3_ctx, testing.allocator);
    defer solver.deinit();
    solver.assert(z3.Z3_mk_not(z3_ctx.ctx, z3.Z3_mk_eq(z3_ctx.ctx, encoded, expected)));
    try testing.expectEqual(@as(z3.Z3_lbool, z3.Z3_L_FALSE), solver.check());
}

test "unsigned mul overflow check proves bounded constant multiplier safe" {
    var z3_ctx = try Context.init(testing.allocator);
    defer z3_ctx.deinit();

    var encoder = Encoder.init(&z3_ctx, testing.allocator);
    defer encoder.deinit();

    const bv256 = z3.Z3_mk_bv_sort(z3_ctx.ctx, 256);
    const lhs_sym = z3.Z3_mk_const(
        z3_ctx.ctx,
        z3.Z3_mk_string_symbol(z3_ctx.ctx, "lhs"),
        bv256,
    );
    const rhs = z3.Z3_mk_numeral(
        z3_ctx.ctx,
        "10000",
        bv256,
    );
    const max = z3.Z3_mk_numeral(
        z3_ctx.ctx,
        "115792089237316195423570985008687907853269984665640564039457584007913129639935",
        bv256,
    );
    const bound = z3.Z3_mk_bv_udiv(z3_ctx.ctx, max, rhs);
    const overflow = encoder.checkMulOverflow(lhs_sym, rhs);

    var solver = try Solver.init(&z3_ctx, testing.allocator);
    defer solver.deinit();
    solver.assert(z3.Z3_mk_bvule(z3_ctx.ctx, lhs_sym, bound));
    solver.assert(overflow);
    try testing.expectEqual(@as(z3.Z3_lbool, z3.Z3_L_FALSE), solver.check());
}

test "func.call relation can be disabled" {
    var z3_ctx = try Context.init(testing.allocator);
    defer z3_ctx.deinit();

    var encoder = Encoder.init(&z3_ctx, testing.allocator);
    defer encoder.deinit();
    encoder.setVerifyCalls(false);

    const mlir_ctx = mlir.oraContextCreate();
    defer mlir.oraContextDestroy(mlir_ctx);
    loadAllDialects(mlir_ctx);
    _ = mlir.oraDialectRegister(mlir_ctx);

    const loc = mlir.oraLocationUnknownGet(mlir_ctx);
    const i256_ty = mlir.oraIntegerTypeCreate(mlir_ctx, 256);
    const arg_attr = mlir.oraIntegerAttrCreateI64FromType(i256_ty, 7);
    const arg_op = mlir.oraArithConstantOpCreate(mlir_ctx, loc, i256_ty, arg_attr);
    const arg = mlir.oraOperationGetResult(arg_op, 0);

    const operands = [_]mlir.MlirValue{arg};
    const results = [_]mlir.MlirType{i256_ty};
    const call_a = mlir.oraFuncCallOpCreate(mlir_ctx, loc, stringRef("foo"), &operands, operands.len, &results, results.len);
    const call_b = mlir.oraFuncCallOpCreate(mlir_ctx, loc, stringRef("foo"), &operands, operands.len, &results, results.len);

    const ast_a = try encoder.encodeOperation(call_a);
    const ast_b = try encoder.encodeOperation(call_b);
    var solver = try Solver.init(&z3_ctx, testing.allocator);
    defer solver.deinit();
    const neq = z3.Z3_mk_not(z3_ctx.ctx, z3.Z3_mk_eq(z3_ctx.ctx, ast_a, ast_b));
    solver.assert(neq);
    try testing.expectEqual(@as(z3.Z3_lbool, z3.Z3_L_TRUE), solver.check());
}

test "unresolved call sites use distinct opaque symbols" {
    var z3_ctx = try Context.init(testing.allocator);
    defer z3_ctx.deinit();

    var encoder = Encoder.init(&z3_ctx, testing.allocator);
    defer encoder.deinit();

    const mlir_ctx = mlir.oraContextCreate();
    defer mlir.oraContextDestroy(mlir_ctx);
    loadAllDialects(mlir_ctx);
    _ = mlir.oraDialectRegister(mlir_ctx);

    const loc = mlir.oraLocationUnknownGet(mlir_ctx);
    const i256_ty = mlir.oraIntegerTypeCreate(mlir_ctx, 256);
    const arg_attr = mlir.oraIntegerAttrCreateI64FromType(i256_ty, 7);
    const arg_op = mlir.oraArithConstantOpCreate(mlir_ctx, loc, i256_ty, arg_attr);
    const arg = mlir.oraOperationGetResult(arg_op, 0);

    const operands = [_]mlir.MlirValue{arg};
    const results = [_]mlir.MlirType{i256_ty};
    const empty = stringRef("");
    const call_a = mlir.oraFuncCallOpCreate(mlir_ctx, loc, empty, &operands, operands.len, &results, results.len);
    const call_b = mlir.oraFuncCallOpCreate(mlir_ctx, loc, empty, &operands, operands.len, &results, results.len);

    const ast_a = try encoder.encodeOperation(call_a);
    const ast_b = try encoder.encodeOperation(call_b);

    var solver = try Solver.init(&z3_ctx, testing.allocator);
    defer solver.deinit();
    solver.assert(z3.Z3_mk_not(z3_ctx.ctx, z3.Z3_mk_eq(z3_ctx.ctx, ast_a, ast_b)));
    try testing.expectEqual(@as(z3.Z3_lbool, z3.Z3_L_TRUE), solver.check());
}

test "call summary degradation propagates to caller encoder" {
    var z3_ctx = try Context.init(testing.allocator);
    defer z3_ctx.deinit();

    var encoder = Encoder.init(&z3_ctx, testing.allocator);
    defer encoder.deinit();

    const mlir_ctx = mlir.oraContextCreate();
    defer mlir.oraContextDestroy(mlir_ctx);
    loadAllDialects(mlir_ctx);
    _ = mlir.oraDialectRegister(mlir_ctx);

    const loc = mlir.oraLocationUnknownGet(mlir_ctx);
    const i256_ty = mlir.oraIntegerTypeCreate(mlir_ctx, 256);
    const func_attrs = [_]mlir.MlirNamedAttribute{
        namedAttr(mlir_ctx, "sym_name", mlir.oraStringAttrCreate(mlir_ctx, stringRef("helper"))),
    };
    const result_types = [_]mlir.MlirType{i256_ty};
    const result_locs = [_]mlir.MlirLocation{loc};
    const helper = mlir.oraFuncFuncOpCreate(mlir_ctx, loc, &func_attrs, func_attrs.len, &result_types, &result_locs, result_types.len);
    const body = mlir.oraFuncOpGetBodyBlock(helper);

    const init_attr = mlir.oraIntegerAttrCreateI64FromType(i256_ty, 7);
    const init_const = mlir.oraArithConstantOpCreate(mlir_ctx, loc, i256_ty, init_attr);
    const init_value = mlir.oraOperationGetResult(init_const, 0);
    const tstore = mlir.oraTStoreOpCreate(mlir_ctx, loc, init_value, stringRef("pending"));
    const tload = mlir.oraTLoadOpCreate(mlir_ctx, loc, stringRef("pending"), i256_ty);
    const ret_vals = [_]mlir.MlirValue{mlir.oraOperationGetResult(tload, 0)};
    const ret = mlir.oraReturnOpCreate(mlir_ctx, loc, &ret_vals, ret_vals.len);
    mlir.oraBlockAppendOwnedOperation(body, init_const);
    mlir.oraBlockAppendOwnedOperation(body, tstore);
    mlir.oraBlockAppendOwnedOperation(body, tload);
    mlir.oraBlockAppendOwnedOperation(body, ret);

    try encoder.registerFunctionOperation(helper);

    const call = mlir.oraFuncCallOpCreate(mlir_ctx, loc, stringRef("helper"), &[_]mlir.MlirValue{}, 0, &result_types, result_types.len);
    const call_result = try encoder.encodeOperation(call);

    try testing.expect(!encoder.isDegraded());
    try testing.expectEqualStrings(
        "#x0000000000000000000000000000000000000000000000000000000000000007",
        std.mem.span(z3.Z3_ast_to_string(z3_ctx.ctx, call_result)),
    );
}

test "known callee with unknown write set degrades encoder" {
    var z3_ctx = try Context.init(testing.allocator);
    defer z3_ctx.deinit();

    var encoder = Encoder.init(&z3_ctx, testing.allocator);
    defer encoder.deinit();

    const mlir_ctx = mlir.oraContextCreate();
    defer mlir.oraContextDestroy(mlir_ctx);
    loadAllDialects(mlir_ctx);
    _ = mlir.oraDialectRegister(mlir_ctx);

    const loc = mlir.oraLocationUnknownGet(mlir_ctx);
    const i256_ty = mlir.oraIntegerTypeCreate(mlir_ctx, 256);

    const helper_attrs = [_]mlir.MlirNamedAttribute{
        namedAttr(mlir_ctx, "sym_name", mlir.oraStringAttrCreate(mlir_ctx, stringRef("writerWithUnknownSlots"))),
        namedAttr(mlir_ctx, "ora.effect", mlir.oraStringAttrCreate(mlir_ctx, stringRef("writes"))),
    };
    const helper = mlir.oraFuncFuncOpCreate(mlir_ctx, loc, &helper_attrs, helper_attrs.len, &[_]mlir.MlirType{}, &[_]mlir.MlirLocation{}, 0);
    const helper_body = mlir.oraFuncOpGetBodyBlock(helper);

    const unresolved_call = mlir.oraFuncCallOpCreate(
        mlir_ctx,
        loc,
        stringRef("opaqueWriter"),
        &[_]mlir.MlirValue{},
        0,
        &[_]mlir.MlirType{},
        0,
    );
    const ret = mlir.oraReturnOpCreate(mlir_ctx, loc, &[_]mlir.MlirValue{}, 0);
    mlir.oraBlockAppendOwnedOperation(helper_body, unresolved_call);
    mlir.oraBlockAppendOwnedOperation(helper_body, ret);

    try encoder.registerFunctionOperation(helper);

    const zero_attr = mlir.oraIntegerAttrCreateI64FromType(i256_ty, 0);
    const zero_op = mlir.oraArithConstantOpCreate(mlir_ctx, loc, i256_ty, zero_attr);
    const zero = mlir.oraOperationGetResult(zero_op, 0);
    const seed_store = mlir.oraSStoreOpCreate(mlir_ctx, loc, zero, stringRef("counter"));
    _ = try encoder.encodeOperation(seed_store);

    const call = mlir.oraFuncCallOpCreate(
        mlir_ctx,
        loc,
        stringRef("writerWithUnknownSlots"),
        &[_]mlir.MlirValue{},
        0,
        &[_]mlir.MlirType{},
        0,
    );
    _ = try encoder.encodeOperation(call);

    try testing.expect(encoder.isDegraded());
    try testing.expect(std.mem.eql(u8, encoder.degradationReason().?, "failed to recover known callee write set exactly"));
}

test "summary precondition encoding failure degrades encoder" {
    var z3_ctx = try Context.init(testing.allocator);
    defer z3_ctx.deinit();

    var encoder = Encoder.init(&z3_ctx, testing.allocator);
    defer encoder.deinit();

    const mlir_ctx = mlir.oraContextCreate();
    defer mlir.oraContextDestroy(mlir_ctx);
    loadAllDialects(mlir_ctx);
    _ = mlir.oraDialectRegister(mlir_ctx);

    const loc = mlir.oraLocationUnknownGet(mlir_ctx);
    const i1_ty = mlir.oraIntegerTypeCreate(mlir_ctx, 1);
    const i256_ty = mlir.oraIntegerTypeCreate(mlir_ctx, 256);

    const helper_attrs = [_]mlir.MlirNamedAttribute{
        namedAttr(mlir_ctx, "sym_name", mlir.oraStringAttrCreate(mlir_ctx, stringRef("requiresMalformedCmp"))),
    };
    const helper = mlir.oraFuncFuncOpCreate(mlir_ctx, loc, &helper_attrs, helper_attrs.len, &[_]mlir.MlirType{i256_ty}, &[_]mlir.MlirLocation{loc}, 1);
    const helper_body = mlir.oraFuncOpGetBodyBlock(helper);
    const helper_arg = mlir.oraBlockGetArgument(helper_body, 0);

    const zero_attr = mlir.oraIntegerAttrCreateI64FromType(i256_ty, 0);
    const zero_op = mlir.oraArithConstantOpCreate(mlir_ctx, loc, i256_ty, zero_attr);
    const malformed_cmp = mlir.oraCmpOpCreate(mlir_ctx, loc, stringRef(""), helper_arg, mlir.oraOperationGetResult(zero_op, 0), i1_ty);
    const malformed_cond = mlir.oraOperationGetResult(malformed_cmp, 0);
    const precond = mlir.oraCfAssertOpCreate(mlir_ctx, loc, malformed_cond, stringRef("bad precondition"));
    mlir.oraOperationSetAttributeByName(precond, stringRef("ora.requires"), mlir.oraBoolAttrCreate(mlir_ctx, true));
    const ret = mlir.oraReturnOpCreate(mlir_ctx, loc, &[_]mlir.MlirValue{helper_arg}, 1);

    mlir.oraBlockAppendOwnedOperation(helper_body, zero_op);
    mlir.oraBlockAppendOwnedOperation(helper_body, malformed_cmp);
    mlir.oraBlockAppendOwnedOperation(helper_body, precond);
    mlir.oraBlockAppendOwnedOperation(helper_body, ret);

    try encoder.registerFunctionOperation(helper);

    const arg_attr = mlir.oraIntegerAttrCreateI64FromType(i256_ty, 7);
    const arg_op = mlir.oraArithConstantOpCreate(mlir_ctx, loc, i256_ty, arg_attr);
    const call = mlir.oraFuncCallOpCreate(
        mlir_ctx,
        loc,
        stringRef("requiresMalformedCmp"),
        &[_]mlir.MlirValue{mlir.oraOperationGetResult(arg_op, 0)},
        1,
        &[_]mlir.MlirType{i256_ty},
        1,
    );

    _ = try encoder.encodeOperation(call);

    try testing.expect(encoder.isDegraded());
    const reason = encoder.degradationReason().?;
    try testing.expect(
        std.mem.containsAtLeast(u8, reason, 1, "failed to encode summary precondition") or
            std.mem.containsAtLeast(u8, reason, 1, "ora.cmp missing predicate"),
    );
}

test "known zero-result stateful callee preserves state exactly" {
    var z3_ctx = try Context.init(testing.allocator);
    defer z3_ctx.deinit();

    var encoder = Encoder.init(&z3_ctx, testing.allocator);
    defer encoder.deinit();
    encoder.setVerifyState(true);

    const mlir_ctx = mlir.oraContextCreate();
    defer mlir.oraContextDestroy(mlir_ctx);
    loadAllDialects(mlir_ctx);
    _ = mlir.oraDialectRegister(mlir_ctx);

    const loc = mlir.oraLocationUnknownGet(mlir_ctx);
    const helper_attrs = [_]mlir.MlirNamedAttribute{
        namedAttr(mlir_ctx, "sym_name", mlir.oraStringAttrCreate(mlir_ctx, stringRef("declaredWriter"))),
        namedAttr(mlir_ctx, "ora.effect", mlir.oraStringAttrCreate(mlir_ctx, stringRef("writes"))),
        namedAttr(mlir_ctx, "ora.write_slots", mlir.oraArrayAttrCreate(mlir_ctx, 1, &[_]mlir.MlirAttribute{
            mlir.oraStringAttrCreate(mlir_ctx, stringRef("counter")),
        })),
    };
    const helper = mlir.oraFuncFuncOpCreate(mlir_ctx, loc, &helper_attrs, helper_attrs.len, &[_]mlir.MlirType{}, &[_]mlir.MlirLocation{}, 0);
    // Intentionally leave the body without encodable writes so summary falls back.

    try encoder.registerFunctionOperation(helper);

    const i256_ty = mlir.oraIntegerTypeCreate(mlir_ctx, 256);
    const zero_attr = mlir.oraIntegerAttrCreateI64FromType(i256_ty, 0);
    const zero_op = mlir.oraArithConstantOpCreate(mlir_ctx, loc, i256_ty, zero_attr);
    const zero = mlir.oraOperationGetResult(zero_op, 0);
    const zero_ast = try encoder.encodeOperation(zero_op);
    const seed_store = mlir.oraSStoreOpCreate(mlir_ctx, loc, zero, stringRef("counter"));
    _ = try encoder.encodeOperation(seed_store);

    const call = mlir.oraFuncCallOpCreate(
        mlir_ctx,
        loc,
        stringRef("declaredWriter"),
        &[_]mlir.MlirValue{},
        0,
        &[_]mlir.MlirType{},
        0,
    );
    _ = try encoder.encodeOperation(call);

    try testing.expect(!encoder.isDegraded());
    const counter = encoder.global_map.get("counter").?;

    var solver = try Solver.init(&z3_ctx, testing.allocator);
    defer solver.deinit();
    solver.assert(z3.Z3_mk_not(z3_ctx.ctx, z3.Z3_mk_eq(z3_ctx.ctx, counter, zero_ast)));
    try testing.expectEqual(@as(z3.Z3_lbool, z3.Z3_L_FALSE), solver.check());
}

test "map_store updates global map for later map_get" {
    var z3_ctx = try Context.init(testing.allocator);
    defer z3_ctx.deinit();

    var encoder = Encoder.init(&z3_ctx, testing.allocator);
    defer encoder.deinit();

    const mlir_ctx = mlir.oraContextCreate();
    defer mlir.oraContextDestroy(mlir_ctx);
    loadAllDialects(mlir_ctx);
    _ = mlir.oraDialectRegister(mlir_ctx);

    const loc = mlir.oraLocationUnknownGet(mlir_ctx);
    const i256_ty = mlir.oraIntegerTypeCreate(mlir_ctx, 256);
    const map_ty = mlir.oraMapTypeGet(mlir_ctx, i256_ty, i256_ty);

    const key_attr = mlir.oraIntegerAttrCreateI64FromType(i256_ty, 5);
    const val_attr = mlir.oraIntegerAttrCreateI64FromType(i256_ty, 99);
    const key_op = mlir.oraArithConstantOpCreate(mlir_ctx, loc, i256_ty, key_attr);
    const val_op = mlir.oraArithConstantOpCreate(mlir_ctx, loc, i256_ty, val_attr);
    const key = mlir.oraOperationGetResult(key_op, 0);
    const value = mlir.oraOperationGetResult(val_op, 0);

    const map_load_before = mlir.oraSLoadOpCreate(mlir_ctx, loc, stringRef("balances"), map_ty);
    const map_before = mlir.oraOperationGetResult(map_load_before, 0);
    _ = try encoder.encodeOperation(mlir.oraMapStoreOpCreate(mlir_ctx, loc, map_before, key, value));

    const map_load_after = mlir.oraSLoadOpCreate(mlir_ctx, loc, stringRef("balances"), map_ty);
    const map_after = mlir.oraOperationGetResult(map_load_after, 0);
    const map_get = mlir.oraMapGetOpCreate(mlir_ctx, loc, map_after, key, i256_ty);
    const loaded = try encoder.encodeOperation(map_get);
    const expected = try encoder.encodeValue(value);

    var solver = try Solver.init(&z3_ctx, testing.allocator);
    defer solver.deinit();
    const neq = z3.Z3_mk_not(z3_ctx.ctx, z3.Z3_mk_eq(z3_ctx.ctx, loaded, expected));
    solver.assert(neq);
    try testing.expectEqual(@as(z3.Z3_lbool, z3.Z3_L_FALSE), solver.check());
}

test "nested map_store rethreads inner update through outer map" {
    var z3_ctx = try Context.init(testing.allocator);
    defer z3_ctx.deinit();

    var encoder = Encoder.init(&z3_ctx, testing.allocator);
    defer encoder.deinit();

    const mlir_ctx = mlir.oraContextCreate();
    defer mlir.oraContextDestroy(mlir_ctx);
    loadAllDialects(mlir_ctx);
    _ = mlir.oraDialectRegister(mlir_ctx);

    const loc = mlir.oraLocationUnknownGet(mlir_ctx);
    const i256_ty = mlir.oraIntegerTypeCreate(mlir_ctx, 256);
    const inner_map_ty = mlir.oraMapTypeGet(mlir_ctx, i256_ty, i256_ty);
    const outer_map_ty = mlir.oraMapTypeGet(mlir_ctx, i256_ty, inner_map_ty);

    const owner_attr = mlir.oraIntegerAttrCreateI64FromType(i256_ty, 11);
    const spender_attr = mlir.oraIntegerAttrCreateI64FromType(i256_ty, 22);
    const value_attr = mlir.oraIntegerAttrCreateI64FromType(i256_ty, 99);

    const owner_op = mlir.oraArithConstantOpCreate(mlir_ctx, loc, i256_ty, owner_attr);
    const spender_op = mlir.oraArithConstantOpCreate(mlir_ctx, loc, i256_ty, spender_attr);
    const value_op = mlir.oraArithConstantOpCreate(mlir_ctx, loc, i256_ty, value_attr);

    const owner = mlir.oraOperationGetResult(owner_op, 0);
    const spender = mlir.oraOperationGetResult(spender_op, 0);
    const value = mlir.oraOperationGetResult(value_op, 0);

    const outer_before = mlir.oraSLoadOpCreate(mlir_ctx, loc, stringRef("allowances"), outer_map_ty);
    const outer_before_value = mlir.oraOperationGetResult(outer_before, 0);
    const inner_before = mlir.oraMapGetOpCreate(mlir_ctx, loc, outer_before_value, owner, inner_map_ty);
    const inner_before_value = mlir.oraOperationGetResult(inner_before, 0);

    _ = try encoder.encodeOperation(mlir.oraMapStoreOpCreate(mlir_ctx, loc, inner_before_value, spender, value));
    _ = try encoder.encodeOperation(mlir.oraMapStoreOpCreate(mlir_ctx, loc, outer_before_value, owner, inner_before_value));

    const outer_after = mlir.oraSLoadOpCreate(mlir_ctx, loc, stringRef("allowances"), outer_map_ty);
    const outer_after_value = mlir.oraOperationGetResult(outer_after, 0);
    const inner_after = mlir.oraMapGetOpCreate(mlir_ctx, loc, outer_after_value, owner, inner_map_ty);
    const inner_after_value = try encoder.encodeOperation(inner_after);
    const final_get = mlir.oraMapGetOpCreate(mlir_ctx, loc, mlir.oraOperationGetResult(inner_after, 0), spender, i256_ty);
    const loaded = try encoder.encodeOperation(final_get);
    const expected = try encoder.encodeValue(value);

    var solver = try Solver.init(&z3_ctx, testing.allocator);
    defer solver.deinit();
    solver.assert(z3.Z3_mk_not(z3_ctx.ctx, z3.Z3_mk_eq(z3_ctx.ctx, loaded, expected)));
    try testing.expectEqual(@as(z3.Z3_lbool, z3.Z3_L_FALSE), solver.check());
    _ = inner_after_value;
}

test "map key operands are coerced to map domain width" {
    var z3_ctx = try Context.init(testing.allocator);
    defer z3_ctx.deinit();

    var encoder = Encoder.init(&z3_ctx, testing.allocator);
    defer encoder.deinit();

    const mlir_ctx = mlir.oraContextCreate();
    defer mlir.oraContextDestroy(mlir_ctx);
    loadAllDialects(mlir_ctx);
    _ = mlir.oraDialectRegister(mlir_ctx);

    const loc = mlir.oraLocationUnknownGet(mlir_ctx);
    const i160_ty = mlir.oraIntegerTypeCreate(mlir_ctx, 160);
    const i256_ty = mlir.oraIntegerTypeCreate(mlir_ctx, 256);
    const map_ty = mlir.oraMapTypeGet(mlir_ctx, i256_ty, i256_ty);

    const key160_attr = mlir.oraIntegerAttrCreateI64FromType(i160_ty, 5);
    const key256_attr = mlir.oraIntegerAttrCreateI64FromType(i256_ty, 5);
    const val_attr = mlir.oraIntegerAttrCreateI64FromType(i256_ty, 99);
    const key160_op = mlir.oraArithConstantOpCreate(mlir_ctx, loc, i160_ty, key160_attr);
    const key256_op = mlir.oraArithConstantOpCreate(mlir_ctx, loc, i256_ty, key256_attr);
    const value_op = mlir.oraArithConstantOpCreate(mlir_ctx, loc, i256_ty, val_attr);
    const key160 = mlir.oraOperationGetResult(key160_op, 0);
    const key256 = mlir.oraOperationGetResult(key256_op, 0);
    const value = mlir.oraOperationGetResult(value_op, 0);

    const map_load_before = mlir.oraSLoadOpCreate(mlir_ctx, loc, stringRef("balances"), map_ty);
    const map_before = mlir.oraOperationGetResult(map_load_before, 0);
    _ = try encoder.encodeOperation(mlir.oraMapStoreOpCreate(mlir_ctx, loc, map_before, key160, value));

    const map_load_after = mlir.oraSLoadOpCreate(mlir_ctx, loc, stringRef("balances"), map_ty);
    const map_after = mlir.oraOperationGetResult(map_load_after, 0);
    const map_get = mlir.oraMapGetOpCreate(mlir_ctx, loc, map_after, key256, i256_ty);
    const loaded = try encoder.encodeOperation(map_get);
    const expected = try encoder.encodeValue(value);

    var solver = try Solver.init(&z3_ctx, testing.allocator);
    defer solver.deinit();
    const neq = z3.Z3_mk_not(z3_ctx.ctx, z3.Z3_mk_eq(z3_ctx.ctx, loaded, expected));
    solver.assert(neq);
    try testing.expectEqual(@as(z3.Z3_lbool, z3.Z3_L_FALSE), solver.check());
}

test "map_store emits quantified frame constraint" {
    var z3_ctx = try Context.init(testing.allocator);
    defer z3_ctx.deinit();

    var encoder = Encoder.init(&z3_ctx, testing.allocator);
    defer encoder.deinit();

    const mlir_ctx = mlir.oraContextCreate();
    defer mlir.oraContextDestroy(mlir_ctx);
    loadAllDialects(mlir_ctx);
    _ = mlir.oraDialectRegister(mlir_ctx);

    const loc = mlir.oraLocationUnknownGet(mlir_ctx);
    const i256_ty = mlir.oraIntegerTypeCreate(mlir_ctx, 256);
    const map_ty = mlir.oraMapTypeGet(mlir_ctx, i256_ty, i256_ty);

    const key_attr = mlir.oraIntegerAttrCreateI64FromType(i256_ty, 9);
    const value_attr = mlir.oraIntegerAttrCreateI64FromType(i256_ty, 13);
    const key_op = mlir.oraArithConstantOpCreate(mlir_ctx, loc, i256_ty, key_attr);
    const value_op = mlir.oraArithConstantOpCreate(mlir_ctx, loc, i256_ty, value_attr);
    const map_load = mlir.oraSLoadOpCreate(mlir_ctx, loc, stringRef("balances"), map_ty);
    const map_store = mlir.oraMapStoreOpCreate(
        mlir_ctx,
        loc,
        mlir.oraOperationGetResult(map_load, 0),
        mlir.oraOperationGetResult(key_op, 0),
        mlir.oraOperationGetResult(value_op, 0),
    );
    _ = try encoder.encodeOperation(map_store);

    const constraints = try encoder.takeConstraints(testing.allocator);
    defer if (constraints.len > 0) testing.allocator.free(constraints);
    try testing.expect(constraints.len > 0);

    var saw_forall = false;
    for (constraints) |cst| {
        const text = std.mem.span(z3.Z3_ast_to_string(z3_ctx.ctx, cst));
        if (std.mem.indexOf(u8, text, "forall") != null) {
            saw_forall = true;
            break;
        }
    }
    try testing.expect(saw_forall);
}

test "func.call adds quantified frame constraint for untouched map slot" {
    var z3_ctx = try Context.init(testing.allocator);
    defer z3_ctx.deinit();

    var encoder = Encoder.init(&z3_ctx, testing.allocator);
    defer encoder.deinit();

    const mlir_ctx = mlir.oraContextCreate();
    defer mlir.oraContextDestroy(mlir_ctx);
    loadAllDialects(mlir_ctx);
    _ = mlir.oraDialectRegister(mlir_ctx);

    const loc = mlir.oraLocationUnknownGet(mlir_ctx);
    const i256_ty = mlir.oraIntegerTypeCreate(mlir_ctx, 256);
    const map_ty = mlir.oraMapTypeGet(mlir_ctx, i256_ty, i256_ty);

    // Seed an array-typed global slot in current state so call-summary frame logic
    // has a non-written map slot to preserve.
    _ = try encoder.encodeOperation(mlir.oraSLoadOpCreate(mlir_ctx, loc, stringRef("balances"), map_ty));

    const sym_name_attr = mlir.oraStringAttrCreate(mlir_ctx, stringRef("setCounter"));
    const effect_attr = mlir.oraStringAttrCreate(mlir_ctx, stringRef("writes"));
    const slot_attr = mlir.oraStringAttrCreate(mlir_ctx, stringRef("counter"));
    const slot_array = [_]mlir.MlirAttribute{slot_attr};
    const write_slots_attr = mlir.oraArrayAttrCreate(mlir_ctx, slot_array.len, &slot_array);
    const func_attrs = [_]mlir.MlirNamedAttribute{
        namedAttr(mlir_ctx, "sym_name", sym_name_attr),
        namedAttr(mlir_ctx, "ora.effect", effect_attr),
        namedAttr(mlir_ctx, "ora.write_slots", write_slots_attr),
    };
    const param_types = [_]mlir.MlirType{i256_ty};
    const param_locs = [_]mlir.MlirLocation{loc};
    const func_op = mlir.oraFuncFuncOpCreate(mlir_ctx, loc, &func_attrs, func_attrs.len, &param_types, &param_locs, param_types.len);
    const body = mlir.oraFuncOpGetBodyBlock(func_op);
    const arg0 = mlir.oraBlockGetArgument(body, 0);
    const sstore = mlir.oraSStoreOpCreate(mlir_ctx, loc, arg0, stringRef("counter"));
    const ret_vals = [_]mlir.MlirValue{arg0};
    const ret_op = mlir.oraReturnOpCreate(mlir_ctx, loc, &ret_vals, ret_vals.len);
    mlir.oraBlockAppendOwnedOperation(body, sstore);
    mlir.oraBlockAppendOwnedOperation(body, ret_op);
    try encoder.registerFunctionOperation(func_op);

    const arg_attr = mlir.oraIntegerAttrCreateI64FromType(i256_ty, 77);
    const arg_op = mlir.oraArithConstantOpCreate(mlir_ctx, loc, i256_ty, arg_attr);
    const call_operands = [_]mlir.MlirValue{mlir.oraOperationGetResult(arg_op, 0)};
    const call_results = [_]mlir.MlirType{i256_ty};
    const call_op = mlir.oraFuncCallOpCreate(mlir_ctx, loc, stringRef("setCounter"), &call_operands, call_operands.len, &call_results, call_results.len);
    _ = try encoder.encodeOperation(call_op);

    const constraints = try encoder.takeConstraints(testing.allocator);
    defer if (constraints.len > 0) testing.allocator.free(constraints);
    try testing.expect(constraints.len > 0);

    var saw_frame_forall = false;
    for (constraints) |cst| {
        const text = std.mem.span(z3.Z3_ast_to_string(z3_ctx.ctx, cst));
        if (std.mem.indexOf(u8, text, "forall") != null and std.mem.indexOf(u8, text, "frame_eq_k_") != null) {
            saw_frame_forall = true;
            break;
        }
    }
    try testing.expect(saw_frame_forall);
}

test "state threading can be disabled" {
    var z3_ctx = try Context.init(testing.allocator);
    defer z3_ctx.deinit();

    var encoder = Encoder.init(&z3_ctx, testing.allocator);
    defer encoder.deinit();
    encoder.setVerifyState(false);

    const mlir_ctx = mlir.oraContextCreate();
    defer mlir.oraContextDestroy(mlir_ctx);
    loadAllDialects(mlir_ctx);
    _ = mlir.oraDialectRegister(mlir_ctx);

    const loc = mlir.oraLocationUnknownGet(mlir_ctx);
    const i256_ty = mlir.oraIntegerTypeCreate(mlir_ctx, 256);
    const value_attr = mlir.oraIntegerAttrCreateI64FromType(i256_ty, 42);
    const value_op = mlir.oraArithConstantOpCreate(mlir_ctx, loc, i256_ty, value_attr);
    const value = mlir.oraOperationGetResult(value_op, 0);

    _ = try encoder.encodeOperation(mlir.oraSStoreOpCreate(mlir_ctx, loc, value, stringRef("counter")));
    const loaded = try encoder.encodeOperation(mlir.oraSLoadOpCreate(mlir_ctx, loc, stringRef("counter"), i256_ty));
    const expected = try encoder.encodeValue(value);

    var solver = try Solver.init(&z3_ctx, testing.allocator);
    defer solver.deinit();
    const neq = z3.Z3_mk_not(z3_ctx.ctx, z3.Z3_mk_eq(z3_ctx.ctx, loaded, expected));
    solver.assert(neq);
    try testing.expectEqual(@as(z3.Z3_lbool, z3.Z3_L_TRUE), solver.check());
}

test "scf.if multi-result encoding uses matching yield index" {
    var z3_ctx = try Context.init(testing.allocator);
    defer z3_ctx.deinit();

    var encoder = Encoder.init(&z3_ctx, testing.allocator);
    defer encoder.deinit();

    const mlir_ctx = mlir.oraContextCreate();
    defer mlir.oraContextDestroy(mlir_ctx);
    loadAllDialects(mlir_ctx);
    _ = mlir.oraDialectRegister(mlir_ctx);

    const loc = mlir.oraLocationUnknownGet(mlir_ctx);
    const i1_ty = mlir.oraIntegerTypeCreate(mlir_ctx, 1);
    const i256_ty = mlir.oraIntegerTypeCreate(mlir_ctx, 256);

    const cond_attr = mlir.oraIntegerAttrCreateI64FromType(i1_ty, 1);
    const cond_op = mlir.oraArithConstantOpCreate(mlir_ctx, loc, i1_ty, cond_attr);
    const cond = mlir.oraOperationGetResult(cond_op, 0);

    const then0_attr = mlir.oraIntegerAttrCreateI64FromType(i256_ty, 11);
    const then1_attr = mlir.oraIntegerAttrCreateI64FromType(i256_ty, 22);
    const else0_attr = mlir.oraIntegerAttrCreateI64FromType(i256_ty, 33);
    const else1_attr = mlir.oraIntegerAttrCreateI64FromType(i256_ty, 44);

    const then0_op = mlir.oraArithConstantOpCreate(mlir_ctx, loc, i256_ty, then0_attr);
    const then1_op = mlir.oraArithConstantOpCreate(mlir_ctx, loc, i256_ty, then1_attr);
    const else0_op = mlir.oraArithConstantOpCreate(mlir_ctx, loc, i256_ty, else0_attr);
    const else1_op = mlir.oraArithConstantOpCreate(mlir_ctx, loc, i256_ty, else1_attr);

    const then0 = mlir.oraOperationGetResult(then0_op, 0);
    const then1 = mlir.oraOperationGetResult(then1_op, 0);
    const else0 = mlir.oraOperationGetResult(else0_op, 0);
    const else1 = mlir.oraOperationGetResult(else1_op, 0);

    const result_types = [_]mlir.MlirType{ i256_ty, i256_ty };
    const if_op = mlir.oraScfIfOpCreate(mlir_ctx, loc, cond, &result_types, result_types.len, true);
    const then_block = mlir.oraScfIfOpGetThenBlock(if_op);
    const else_block = mlir.oraScfIfOpGetElseBlock(if_op);

    const then_vals = [_]mlir.MlirValue{ then0, then1 };
    const else_vals = [_]mlir.MlirValue{ else0, else1 };
    const then_yield = mlir.oraScfYieldOpCreate(mlir_ctx, loc, &then_vals, then_vals.len);
    const else_yield = mlir.oraScfYieldOpCreate(mlir_ctx, loc, &else_vals, else_vals.len);
    mlir.oraBlockAppendOwnedOperation(then_block, then_yield);
    mlir.oraBlockAppendOwnedOperation(else_block, else_yield);

    const if_r0 = mlir.oraOperationGetResult(if_op, 0);
    const if_r1 = mlir.oraOperationGetResult(if_op, 1);
    const enc_r0 = try encoder.encodeValue(if_r0);
    const enc_r1 = try encoder.encodeValue(if_r1);
    const enc_then0 = try encoder.encodeValue(then0);
    const enc_then1 = try encoder.encodeValue(then1);

    var solver = try Solver.init(&z3_ctx, testing.allocator);
    defer solver.deinit();

    // With constant true condition, each result should match the corresponding then-yield operand.
    solver.assert(z3.Z3_mk_not(z3_ctx.ctx, z3.Z3_mk_eq(z3_ctx.ctx, enc_r0, enc_then0)));
    try testing.expectEqual(@as(z3.Z3_lbool, z3.Z3_L_FALSE), solver.check());
    solver.reset();

    solver.assert(z3.Z3_mk_not(z3_ctx.ctx, z3.Z3_mk_eq(z3_ctx.ctx, enc_r1, enc_then1)));
    try testing.expectEqual(@as(z3.Z3_lbool, z3.Z3_L_FALSE), solver.check());
    solver.reset();

    // Results must remain distinct when yielded values differ.
    solver.assert(z3.Z3_mk_eq(z3_ctx.ctx, enc_r0, enc_r1));
    try testing.expectEqual(@as(z3.Z3_lbool, z3.Z3_L_FALSE), solver.check());
}

test "scf.if result encoding degrades when a branch yield is missing" {
    var z3_ctx = try Context.init(testing.allocator);
    defer z3_ctx.deinit();

    var encoder = Encoder.init(&z3_ctx, testing.allocator);
    defer encoder.deinit();

    const mlir_ctx = mlir.oraContextCreate();
    defer mlir.oraContextDestroy(mlir_ctx);
    loadAllDialects(mlir_ctx);
    _ = mlir.oraDialectRegister(mlir_ctx);

    const loc = mlir.oraLocationUnknownGet(mlir_ctx);
    const i1_ty = mlir.oraIntegerTypeCreate(mlir_ctx, 1);
    const i256_ty = mlir.oraIntegerTypeCreate(mlir_ctx, 256);

    const cond_attr = mlir.oraIntegerAttrCreateI64FromType(i1_ty, 1);
    const cond_op = mlir.oraArithConstantOpCreate(mlir_ctx, loc, i1_ty, cond_attr);
    const cond = mlir.oraOperationGetResult(cond_op, 0);

    const then_attr = mlir.oraIntegerAttrCreateI64FromType(i256_ty, 11);
    const then_op = mlir.oraArithConstantOpCreate(mlir_ctx, loc, i256_ty, then_attr);
    const then_val = mlir.oraOperationGetResult(then_op, 0);

    const result_types = [_]mlir.MlirType{i256_ty};
    const if_op = mlir.oraScfIfOpCreate(mlir_ctx, loc, cond, &result_types, result_types.len, true);
    const then_block = mlir.oraScfIfOpGetThenBlock(if_op);
    const then_vals = [_]mlir.MlirValue{then_val};
    const then_yield = mlir.oraScfYieldOpCreate(mlir_ctx, loc, &then_vals, then_vals.len);
    mlir.oraBlockAppendOwnedOperation(then_block, then_yield);

    const if_result = mlir.oraOperationGetResult(if_op, 0);
    _ = try encoder.encodeValue(if_result);

    try testing.expect(encoder.isDegraded());
}

test "scf.while result encoding degrades exact SMT modeling" {
    var z3_ctx = try Context.init(testing.allocator);
    defer z3_ctx.deinit();

    var encoder = Encoder.init(&z3_ctx, testing.allocator);
    defer encoder.deinit();

    const mlir_ctx = mlir.oraContextCreate();
    defer mlir.oraContextDestroy(mlir_ctx);
    loadAllDialects(mlir_ctx);
    _ = mlir.oraDialectRegister(mlir_ctx);

    const loc = mlir.oraLocationUnknownGet(mlir_ctx);
    const i256_ty = mlir.oraIntegerTypeCreate(mlir_ctx, 256);
    const init_attr = mlir.oraIntegerAttrCreateI64FromType(i256_ty, 7);
    const init_op = mlir.oraArithConstantOpCreate(mlir_ctx, loc, i256_ty, init_attr);
    const init = mlir.oraOperationGetResult(init_op, 0);

    const init_vals = [_]mlir.MlirValue{init};
    const result_types = [_]mlir.MlirType{i256_ty};
    const while_op = mlir.oraScfWhileOpCreate(mlir_ctx, loc, &init_vals, init_vals.len, &result_types, result_types.len);
    const while_result = mlir.oraOperationGetResult(while_op, 0);
    const encoded = try encoder.encodeValue(while_result);

    const encoded_text = std.mem.span(z3.Z3_ast_to_string(z3_ctx.ctx, encoded));
    try testing.expect(std.mem.indexOf(u8, encoded_text, "scf_while_result_") != null);
    try testing.expect(encoder.isDegraded());
}

test "scf.while zero-iteration result encodes exactly" {
    var z3_ctx = try Context.init(testing.allocator);
    defer z3_ctx.deinit();

    var encoder = Encoder.init(&z3_ctx, testing.allocator);
    defer encoder.deinit();

    const mlir_ctx = mlir.oraContextCreate();
    defer mlir.oraContextDestroy(mlir_ctx);
    loadAllDialects(mlir_ctx);
    _ = mlir.oraDialectRegister(mlir_ctx);

    const loc = mlir.oraLocationUnknownGet(mlir_ctx);
    const i1_ty = mlir.oraIntegerTypeCreate(mlir_ctx, 1);
    const i256_ty = mlir.oraIntegerTypeCreate(mlir_ctx, 256);

    const init_attr = mlir.oraIntegerAttrCreateI64FromType(i256_ty, 7);
    const init_op = mlir.oraArithConstantOpCreate(mlir_ctx, loc, i256_ty, init_attr);
    const init = mlir.oraOperationGetResult(init_op, 0);
    const expected = try encoder.encodeOperation(init_op);

    const false_attr = mlir.oraIntegerAttrCreateI64FromType(i1_ty, 0);
    const false_op = mlir.oraArithConstantOpCreate(mlir_ctx, loc, i1_ty, false_attr);
    const false_val = mlir.oraOperationGetResult(false_op, 0);

    const init_vals = [_]mlir.MlirValue{init};
    const result_types = [_]mlir.MlirType{i256_ty};
    const while_op = mlir.oraScfWhileOpCreate(mlir_ctx, loc, &init_vals, init_vals.len, &result_types, result_types.len);
    const before_block = mlir.oraScfWhileOpGetBeforeBlock(while_op);
    mlir.oraBlockAppendOwnedOperation(before_block, false_op);
    mlir.oraBlockAppendOwnedOperation(before_block, mlir.oraScfConditionOpCreate(
        mlir_ctx,
        loc,
        false_val,
        &[_]mlir.MlirValue{init},
        1,
    ));

    const encoded = try encoder.encodeValue(mlir.oraOperationGetResult(while_op, 0));
    try testing.expect(!encoder.isDegraded());

    var solver = try Solver.init(&z3_ctx, testing.allocator);
    defer solver.deinit();
    solver.assert(z3.Z3_mk_not(z3_ctx.ctx, z3.Z3_mk_eq(z3_ctx.ctx, encoded, expected)));
    try testing.expectEqual(@as(z3.Z3_lbool, z3.Z3_L_FALSE), solver.check());
}

test "scf.while carried false init result encodes exactly" {
    var z3_ctx = try Context.init(testing.allocator);
    defer z3_ctx.deinit();

    var encoder = Encoder.init(&z3_ctx, testing.allocator);
    defer encoder.deinit();

    const mlir_ctx = mlir.oraContextCreate();
    defer mlir.oraContextDestroy(mlir_ctx);
    loadAllDialects(mlir_ctx);
    _ = mlir.oraDialectRegister(mlir_ctx);

    const loc = mlir.oraLocationUnknownGet(mlir_ctx);
    const i1_ty = mlir.oraIntegerTypeCreate(mlir_ctx, 1);

    const false_attr = mlir.oraIntegerAttrCreateI64FromType(i1_ty, 0);
    const false_op = mlir.oraArithConstantOpCreate(mlir_ctx, loc, i1_ty, false_attr);
    const false_val = mlir.oraOperationGetResult(false_op, 0);
    const expected = try encoder.encodeOperation(false_op);

    const init_vals = [_]mlir.MlirValue{false_val};
    const result_types = [_]mlir.MlirType{i1_ty};
    const while_op = mlir.oraScfWhileOpCreate(mlir_ctx, loc, &init_vals, init_vals.len, &result_types, result_types.len);
    const before_block = mlir.oraScfWhileOpGetBeforeBlock(while_op);
    _ = mlir.mlirBlockAddArgument(before_block, i1_ty, loc);
    const before_arg = mlir.oraBlockGetArgument(before_block, 0);

    mlir.oraBlockAppendOwnedOperation(before_block, mlir.oraScfConditionOpCreate(
        mlir_ctx,
        loc,
        before_arg,
        &[_]mlir.MlirValue{before_arg},
        1,
    ));

    const encoded = try encoder.encodeValue(mlir.oraOperationGetResult(while_op, 0));
    try testing.expect(!encoder.isDegraded());

    var solver = try Solver.init(&z3_ctx, testing.allocator);
    defer solver.deinit();
    solver.assert(z3.Z3_mk_not(z3_ctx.ctx, z3.Z3_mk_eq(z3_ctx.ctx, encoded, expected)));
    try testing.expectEqual(@as(z3.Z3_lbool, z3.Z3_L_FALSE), solver.check());
}

test "scf.while single-iteration result encodes exactly" {
    var z3_ctx = try Context.init(testing.allocator);
    defer z3_ctx.deinit();

    var encoder = Encoder.init(&z3_ctx, testing.allocator);
    defer encoder.deinit();

    const mlir_ctx = mlir.oraContextCreate();
    defer mlir.oraContextDestroy(mlir_ctx);
    loadAllDialects(mlir_ctx);
    _ = mlir.oraDialectRegister(mlir_ctx);

    const loc = mlir.oraLocationUnknownGet(mlir_ctx);
    const i1_ty = mlir.oraIntegerTypeCreate(mlir_ctx, 1);

    const true_attr = mlir.oraIntegerAttrCreateI64FromType(i1_ty, 1);
    const false_attr = mlir.oraIntegerAttrCreateI64FromType(i1_ty, 0);
    const true_op = mlir.oraArithConstantOpCreate(mlir_ctx, loc, i1_ty, true_attr);
    const false_op = mlir.oraArithConstantOpCreate(mlir_ctx, loc, i1_ty, false_attr);
    const true_val = mlir.oraOperationGetResult(true_op, 0);
    const false_val = mlir.oraOperationGetResult(false_op, 0);
    const expected = try encoder.encodeOperation(false_op);

    const init_vals = [_]mlir.MlirValue{true_val};
    const result_types = [_]mlir.MlirType{i1_ty};
    const while_op = mlir.oraScfWhileOpCreate(mlir_ctx, loc, &init_vals, init_vals.len, &result_types, result_types.len);
    const before_block = mlir.oraScfWhileOpGetBeforeBlock(while_op);
    const after_block = mlir.oraScfWhileOpGetAfterBlock(while_op);
    _ = mlir.mlirBlockAddArgument(before_block, i1_ty, loc);
    _ = mlir.mlirBlockAddArgument(after_block, i1_ty, loc);
    const before_arg = mlir.oraBlockGetArgument(before_block, 0);

    mlir.oraBlockAppendOwnedOperation(before_block, mlir.oraScfConditionOpCreate(
        mlir_ctx,
        loc,
        before_arg,
        &[_]mlir.MlirValue{before_arg},
        1,
    ));
    mlir.oraBlockAppendOwnedOperation(after_block, mlir.oraScfYieldOpCreate(
        mlir_ctx,
        loc,
        &[_]mlir.MlirValue{false_val},
        1,
    ));

    const encoded = try encoder.encodeValue(mlir.oraOperationGetResult(while_op, 0));
    try testing.expect(!encoder.isDegraded());

    var solver = try Solver.init(&z3_ctx, testing.allocator);
    defer solver.deinit();
    solver.assert(z3.Z3_mk_not(z3_ctx.ctx, z3.Z3_mk_eq(z3_ctx.ctx, encoded, expected)));
    try testing.expectEqual(@as(z3.Z3_lbool, z3.Z3_L_FALSE), solver.check());
}

test "scf.while canonical symbolic increment result encodes exactly" {
    var z3_ctx = try Context.init(testing.allocator);
    defer z3_ctx.deinit();

    var encoder = Encoder.init(&z3_ctx, testing.allocator);
    defer encoder.deinit();

    const mlir_ctx = mlir.oraContextCreate();
    defer mlir.oraContextDestroy(mlir_ctx);
    loadAllDialects(mlir_ctx);
    _ = mlir.oraDialectRegister(mlir_ctx);

    const loc = mlir.oraLocationUnknownGet(mlir_ctx);
    const i256_ty = mlir.oraIntegerTypeCreate(mlir_ctx, 256);

    const zero_attr = mlir.oraIntegerAttrCreateI64FromType(i256_ty, 0);
    const one_attr = mlir.oraIntegerAttrCreateI64FromType(i256_ty, 1);
    const zero_op = mlir.oraArithConstantOpCreate(mlir_ctx, loc, i256_ty, zero_attr);
    const one_op = mlir.oraArithConstantOpCreate(mlir_ctx, loc, i256_ty, one_attr);
    const init = mlir.oraOperationGetResult(zero_op, 0);
    const bound_op = mlir.oraVariablePlaceholderOpCreate(mlir_ctx, loc, stringRef("boundValue"), i256_ty);
    const bound = mlir.oraOperationGetResult(bound_op, 0);

    const init_vals = [_]mlir.MlirValue{init};
    const result_types = [_]mlir.MlirType{i256_ty};
    const while_op = mlir.oraScfWhileOpCreate(mlir_ctx, loc, &init_vals, init_vals.len, &result_types, result_types.len);
    const before_block = mlir.oraScfWhileOpGetBeforeBlock(while_op);
    const after_block = mlir.oraScfWhileOpGetAfterBlock(while_op);
    _ = mlir.mlirBlockAddArgument(before_block, i256_ty, loc);
    _ = mlir.mlirBlockAddArgument(after_block, i256_ty, loc);
    const before_arg = mlir.oraBlockGetArgument(before_block, 0);
    const after_arg = mlir.oraBlockGetArgument(after_block, 0);

    const cmp_op = mlir.oraArithCmpIOpCreate(mlir_ctx, loc, 6, before_arg, bound); // ult
    mlir.oraBlockAppendOwnedOperation(before_block, cmp_op);
    mlir.oraBlockAppendOwnedOperation(before_block, mlir.oraScfConditionOpCreate(
        mlir_ctx,
        loc,
        mlir.oraOperationGetResult(cmp_op, 0),
        &[_]mlir.MlirValue{before_arg},
        1,
    ));

    const next_op = mlir.oraArithAddIOpCreate(mlir_ctx, loc, after_arg, mlir.oraOperationGetResult(one_op, 0));
    mlir.oraBlockAppendOwnedOperation(after_block, next_op);
    mlir.oraBlockAppendOwnedOperation(after_block, mlir.oraScfYieldOpCreate(
        mlir_ctx,
        loc,
        &[_]mlir.MlirValue{mlir.oraOperationGetResult(next_op, 0)},
        1,
    ));

    _ = try encoder.encodeOperation(zero_op);
    _ = try encoder.encodeOperation(one_op);

    const encoded = try encoder.encodeValue(mlir.oraOperationGetResult(while_op, 0));
    try testing.expect(!encoder.isDegraded());
    const expected = try encoder.encodeValue(bound);

    var solver = try Solver.init(&z3_ctx, testing.allocator);
    defer solver.deinit();
    solver.assert(z3.Z3_mk_not(z3_ctx.ctx, z3.Z3_mk_eq(z3_ctx.ctx, encoded, expected)));
    try testing.expectEqual(@as(z3.Z3_lbool, z3.Z3_L_FALSE), solver.check());
}

test "scf.while canonical affine carried-step result encodes exactly" {
    var z3_ctx = try Context.init(testing.allocator);
    defer z3_ctx.deinit();

    var encoder = Encoder.init(&z3_ctx, testing.allocator);
    defer encoder.deinit();

    const mlir_ctx = mlir.oraContextCreate();
    defer mlir.oraContextDestroy(mlir_ctx);
    loadAllDialects(mlir_ctx);
    _ = mlir.oraDialectRegister(mlir_ctx);

    const loc = mlir.oraLocationUnknownGet(mlir_ctx);
    const i256_ty = mlir.oraIntegerTypeCreate(mlir_ctx, 256);

    const zero = mlir.oraOperationGetResult(mlir.oraArithConstantOpCreate(
        mlir_ctx, loc, i256_ty, mlir.oraIntegerAttrCreateI64FromType(i256_ty, 0),
    ), 0);
    const one = mlir.oraOperationGetResult(mlir.oraArithConstantOpCreate(
        mlir_ctx, loc, i256_ty, mlir.oraIntegerAttrCreateI64FromType(i256_ty, 1),
    ), 0);
    const step = mlir.oraOperationGetResult(mlir.oraArithConstantOpCreate(
        mlir_ctx, loc, i256_ty, mlir.oraIntegerAttrCreateI64FromType(i256_ty, 3),
    ), 0);
    const bound = mlir.oraOperationGetResult(mlir.oraVariablePlaceholderOpCreate(mlir_ctx, loc, stringRef("affineWhileBound"), i256_ty), 0);

    const init_vals = [_]mlir.MlirValue{ zero, zero, step };
    const result_types = [_]mlir.MlirType{ i256_ty, i256_ty, i256_ty };
    const while_op = mlir.oraScfWhileOpCreate(mlir_ctx, loc, &init_vals, init_vals.len, &result_types, result_types.len);
    const before_block = mlir.oraScfWhileOpGetBeforeBlock(while_op);
    const after_block = mlir.oraScfWhileOpGetAfterBlock(while_op);
    _ = mlir.mlirBlockAddArgument(before_block, i256_ty, loc);
    _ = mlir.mlirBlockAddArgument(before_block, i256_ty, loc);
    _ = mlir.mlirBlockAddArgument(before_block, i256_ty, loc);
    _ = mlir.mlirBlockAddArgument(after_block, i256_ty, loc);
    _ = mlir.mlirBlockAddArgument(after_block, i256_ty, loc);
    _ = mlir.mlirBlockAddArgument(after_block, i256_ty, loc);
    const before_i = mlir.oraBlockGetArgument(before_block, 1);
    const after_sum = mlir.oraBlockGetArgument(after_block, 0);
    const after_i = mlir.oraBlockGetArgument(after_block, 1);
    const after_step = mlir.oraBlockGetArgument(after_block, 2);

    const cmp_op = mlir.oraArithCmpIOpCreate(mlir_ctx, loc, 6, before_i, bound);
    mlir.oraBlockAppendOwnedOperation(before_block, cmp_op);
    mlir.oraBlockAppendOwnedOperation(before_block, mlir.oraScfConditionOpCreate(
        mlir_ctx, loc, mlir.oraOperationGetResult(cmp_op, 0), &[_]mlir.MlirValue{
            mlir.oraBlockGetArgument(before_block, 0),
            before_i,
            mlir.oraBlockGetArgument(before_block, 2),
        }, 3,
    ));

    const next_sum = mlir.oraArithAddIOpCreate(mlir_ctx, loc, after_sum, after_step);
    const next_i = mlir.oraArithAddIOpCreate(mlir_ctx, loc, after_i, one);
    mlir.oraBlockAppendOwnedOperation(after_block, next_sum);
    mlir.oraBlockAppendOwnedOperation(after_block, next_i);
    mlir.oraBlockAppendOwnedOperation(after_block, mlir.oraScfYieldOpCreate(
        mlir_ctx, loc, &[_]mlir.MlirValue{
            mlir.oraOperationGetResult(next_sum, 0),
            mlir.oraOperationGetResult(next_i, 0),
            after_step,
        }, 3,
    ));

    const encoded = try encoder.encodeValue(mlir.oraOperationGetResult(while_op, 0));
    try testing.expect(!encoder.isDegraded());

    const bound_ast = try encoder.encodeValue(bound);
    const step_ast = try encoder.encodeValue(step);
    const expected = try encoder.encodeArithmeticOp(.Mul, bound_ast, step_ast);

    var solver = try Solver.init(&z3_ctx, testing.allocator);
    defer solver.deinit();
    solver.assert(z3.Z3_mk_not(z3_ctx.ctx, z3.Z3_mk_eq(z3_ctx.ctx, encoded, expected)));
    try testing.expectEqual(@as(z3.Z3_lbool, z3.Z3_L_FALSE), solver.check());
}

test "known pure callee canonical affine carried-step scf.while return encodes exactly" {
    var z3_ctx = try Context.init(testing.allocator);
    defer z3_ctx.deinit();

    var encoder = Encoder.init(&z3_ctx, testing.allocator);
    defer encoder.deinit();

    const mlir_ctx = mlir.oraContextCreate();
    defer mlir.oraContextDestroy(mlir_ctx);
    loadAllDialects(mlir_ctx);
    _ = mlir.oraDialectRegister(mlir_ctx);

    const loc = mlir.oraLocationUnknownGet(mlir_ctx);
    const i256_ty = mlir.oraIntegerTypeCreate(mlir_ctx, 256);

    const helper_attrs = [_]mlir.MlirNamedAttribute{
        namedAttr(mlir_ctx, "sym_name", mlir.oraStringAttrCreate(mlir_ctx, stringRef("affineWhileHelper"))),
    };
    const helper_param_types = [_]mlir.MlirType{i256_ty};
    const helper_param_locs = [_]mlir.MlirLocation{loc};
    const helper = mlir.oraFuncFuncOpCreate(mlir_ctx, loc, &helper_attrs, helper_attrs.len, &helper_param_types, &helper_param_locs, helper_param_types.len);
    const body = mlir.oraFuncOpGetBodyBlock(helper);

    const zero = mlir.oraOperationGetResult(mlir.oraArithConstantOpCreate(
        mlir_ctx, loc, i256_ty, mlir.oraIntegerAttrCreateI64FromType(i256_ty, 0),
    ), 0);
    const one = mlir.oraOperationGetResult(mlir.oraArithConstantOpCreate(
        mlir_ctx, loc, i256_ty, mlir.oraIntegerAttrCreateI64FromType(i256_ty, 1),
    ), 0);
    const step = mlir.oraOperationGetResult(mlir.oraArithConstantOpCreate(
        mlir_ctx, loc, i256_ty, mlir.oraIntegerAttrCreateI64FromType(i256_ty, 3),
    ), 0);
    const bound = mlir.oraBlockGetArgument(body, 0);

    const init_vals = [_]mlir.MlirValue{ zero, zero, step };
    const result_types = [_]mlir.MlirType{ i256_ty, i256_ty, i256_ty };
    const while_op = mlir.oraScfWhileOpCreate(mlir_ctx, loc, &init_vals, init_vals.len, &result_types, result_types.len);
    const before_block = mlir.oraScfWhileOpGetBeforeBlock(while_op);
    const after_block = mlir.oraScfWhileOpGetAfterBlock(while_op);
    _ = mlir.mlirBlockAddArgument(before_block, i256_ty, loc);
    _ = mlir.mlirBlockAddArgument(before_block, i256_ty, loc);
    _ = mlir.mlirBlockAddArgument(before_block, i256_ty, loc);
    _ = mlir.mlirBlockAddArgument(after_block, i256_ty, loc);
    _ = mlir.mlirBlockAddArgument(after_block, i256_ty, loc);
    _ = mlir.mlirBlockAddArgument(after_block, i256_ty, loc);
    const before_i = mlir.oraBlockGetArgument(before_block, 1);
    const after_sum = mlir.oraBlockGetArgument(after_block, 0);
    const after_i = mlir.oraBlockGetArgument(after_block, 1);
    const after_step = mlir.oraBlockGetArgument(after_block, 2);

    const cmp_op = mlir.oraArithCmpIOpCreate(mlir_ctx, loc, 6, before_i, bound);
    mlir.oraBlockAppendOwnedOperation(before_block, cmp_op);
    mlir.oraBlockAppendOwnedOperation(before_block, mlir.oraScfConditionOpCreate(
        mlir_ctx, loc, mlir.oraOperationGetResult(cmp_op, 0), &[_]mlir.MlirValue{
            mlir.oraBlockGetArgument(before_block, 0),
            before_i,
            mlir.oraBlockGetArgument(before_block, 2),
        }, 3,
    ));

    const next_sum = mlir.oraArithAddIOpCreate(mlir_ctx, loc, after_sum, after_step);
    const next_i = mlir.oraArithAddIOpCreate(mlir_ctx, loc, after_i, one);
    mlir.oraBlockAppendOwnedOperation(after_block, next_sum);
    mlir.oraBlockAppendOwnedOperation(after_block, next_i);
    mlir.oraBlockAppendOwnedOperation(after_block, mlir.oraScfYieldOpCreate(
        mlir_ctx, loc, &[_]mlir.MlirValue{
            mlir.oraOperationGetResult(next_sum, 0),
            mlir.oraOperationGetResult(next_i, 0),
            after_step,
        }, 3,
    ));
    mlir.oraBlockAppendOwnedOperation(body, while_op);
    mlir.oraBlockAppendOwnedOperation(body, mlir.oraReturnOpCreate(mlir_ctx, loc, &[_]mlir.MlirValue{mlir.oraOperationGetResult(while_op, 0)}, 1));
    try encoder.registerFunctionOperation(helper);

    const call = mlir.oraFuncCallOpCreate(mlir_ctx, loc, stringRef("affineWhileHelper"), &[_]mlir.MlirValue{bound}, 1, &[_]mlir.MlirType{i256_ty}, 1);
    const encoded = try encoder.encodeValue(mlir.oraOperationGetResult(call, 0));
    try testing.expect(!encoder.isDegraded());

    const bound_ast = try encoder.encodeValue(bound);
    const step_ast = try encoder.encodeValue(step);
    const expected = try encoder.encodeArithmeticOp(.Mul, bound_ast, step_ast);

    var solver = try Solver.init(&z3_ctx, testing.allocator);
    defer solver.deinit();
    const constraints = try encoder.takeConstraints(testing.allocator);
    defer if (constraints.len > 0) testing.allocator.free(constraints);
    for (constraints) |cst| solver.assert(cst);
    solver.assert(z3.Z3_mk_not(z3_ctx.ctx, z3.Z3_mk_eq(z3_ctx.ctx, encoded, expected)));
    try testing.expectEqual(@as(z3.Z3_lbool, z3.Z3_L_FALSE), solver.check());
}

test "scf.while canonical symbolic decrement result encodes exactly" {
    var z3_ctx = try Context.init(testing.allocator);
    defer z3_ctx.deinit();

    var encoder = Encoder.init(&z3_ctx, testing.allocator);
    defer encoder.deinit();

    const mlir_ctx = mlir.oraContextCreate();
    defer mlir.oraContextDestroy(mlir_ctx);
    loadAllDialects(mlir_ctx);
    _ = mlir.oraDialectRegister(mlir_ctx);

    const loc = mlir.oraLocationUnknownGet(mlir_ctx);
    const i256_ty = mlir.oraIntegerTypeCreate(mlir_ctx, 256);

    const zero_attr = mlir.oraIntegerAttrCreateI64FromType(i256_ty, 0);
    const one_attr = mlir.oraIntegerAttrCreateI64FromType(i256_ty, 1);
    const zero_op = mlir.oraArithConstantOpCreate(mlir_ctx, loc, i256_ty, zero_attr);
    const one_op = mlir.oraArithConstantOpCreate(mlir_ctx, loc, i256_ty, one_attr);
    const init_op = mlir.oraVariablePlaceholderOpCreate(mlir_ctx, loc, stringRef("initDecValue"), i256_ty);
    const init = mlir.oraOperationGetResult(init_op, 0);

    const init_vals = [_]mlir.MlirValue{init};
    const result_types = [_]mlir.MlirType{i256_ty};
    const while_op = mlir.oraScfWhileOpCreate(mlir_ctx, loc, &init_vals, init_vals.len, &result_types, result_types.len);
    const before_block = mlir.oraScfWhileOpGetBeforeBlock(while_op);
    const after_block = mlir.oraScfWhileOpGetAfterBlock(while_op);
    _ = mlir.mlirBlockAddArgument(before_block, i256_ty, loc);
    _ = mlir.mlirBlockAddArgument(after_block, i256_ty, loc);
    const before_arg = mlir.oraBlockGetArgument(before_block, 0);
    const after_arg = mlir.oraBlockGetArgument(after_block, 0);

    const cmp_op = mlir.oraArithCmpIOpCreate(mlir_ctx, loc, 8, before_arg, mlir.oraOperationGetResult(zero_op, 0)); // ugt
    mlir.oraBlockAppendOwnedOperation(before_block, cmp_op);
    mlir.oraBlockAppendOwnedOperation(before_block, mlir.oraScfConditionOpCreate(
        mlir_ctx,
        loc,
        mlir.oraOperationGetResult(cmp_op, 0),
        &[_]mlir.MlirValue{before_arg},
        1,
    ));

    const next_op = mlir.oraArithSubIOpCreate(mlir_ctx, loc, after_arg, mlir.oraOperationGetResult(one_op, 0));
    mlir.oraBlockAppendOwnedOperation(after_block, next_op);
    mlir.oraBlockAppendOwnedOperation(after_block, mlir.oraScfYieldOpCreate(
        mlir_ctx,
        loc,
        &[_]mlir.MlirValue{mlir.oraOperationGetResult(next_op, 0)},
        1,
    ));

    _ = try encoder.encodeOperation(zero_op);
    _ = try encoder.encodeOperation(one_op);

    const encoded = try encoder.encodeValue(mlir.oraOperationGetResult(while_op, 0));
    try testing.expect(!encoder.isDegraded());
    const expected = try encoder.encodeValue(mlir.oraOperationGetResult(zero_op, 0));

    var solver = try Solver.init(&z3_ctx, testing.allocator);
    defer solver.deinit();
    solver.assert(z3.Z3_mk_not(z3_ctx.ctx, z3.Z3_mk_eq(z3_ctx.ctx, encoded, expected)));
    try testing.expectEqual(@as(z3.Z3_lbool, z3.Z3_L_FALSE), solver.check());
}

test "scf.while canonical signed symbolic increment result encodes exactly" {
    var z3_ctx = try Context.init(testing.allocator);
    defer z3_ctx.deinit();

    var encoder = Encoder.init(&z3_ctx, testing.allocator);
    defer encoder.deinit();

    const mlir_ctx = mlir.oraContextCreate();
    defer mlir.oraContextDestroy(mlir_ctx);
    loadAllDialects(mlir_ctx);
    _ = mlir.oraDialectRegister(mlir_ctx);

    const loc = mlir.oraLocationUnknownGet(mlir_ctx);
    const i256_ty = mlir.oraIntegerTypeCreate(mlir_ctx, 256);

    const zero_attr = mlir.oraIntegerAttrCreateI64FromType(i256_ty, 0);
    const one_attr = mlir.oraIntegerAttrCreateI64FromType(i256_ty, 1);
    const zero_op = mlir.oraArithConstantOpCreate(mlir_ctx, loc, i256_ty, zero_attr);
    const one_op = mlir.oraArithConstantOpCreate(mlir_ctx, loc, i256_ty, one_attr);
    const init = mlir.oraOperationGetResult(zero_op, 0);
    const bound_op = mlir.oraVariablePlaceholderOpCreate(mlir_ctx, loc, stringRef("signedBoundValue"), i256_ty);
    const bound = mlir.oraOperationGetResult(bound_op, 0);

    const init_vals = [_]mlir.MlirValue{init};
    const result_types = [_]mlir.MlirType{i256_ty};
    const while_op = mlir.oraScfWhileOpCreate(mlir_ctx, loc, &init_vals, init_vals.len, &result_types, result_types.len);
    const before_block = mlir.oraScfWhileOpGetBeforeBlock(while_op);
    const after_block = mlir.oraScfWhileOpGetAfterBlock(while_op);
    _ = mlir.mlirBlockAddArgument(before_block, i256_ty, loc);
    _ = mlir.mlirBlockAddArgument(after_block, i256_ty, loc);
    const before_arg = mlir.oraBlockGetArgument(before_block, 0);
    const after_arg = mlir.oraBlockGetArgument(after_block, 0);

    const cmp_op = mlir.oraArithCmpIOpCreate(mlir_ctx, loc, 2, before_arg, bound); // slt
    mlir.oraBlockAppendOwnedOperation(before_block, cmp_op);
    mlir.oraBlockAppendOwnedOperation(before_block, mlir.oraScfConditionOpCreate(
        mlir_ctx,
        loc,
        mlir.oraOperationGetResult(cmp_op, 0),
        &[_]mlir.MlirValue{before_arg},
        1,
    ));

    const next_op = mlir.oraArithAddIOpCreate(mlir_ctx, loc, after_arg, mlir.oraOperationGetResult(one_op, 0));
    mlir.oraBlockAppendOwnedOperation(after_block, next_op);
    mlir.oraBlockAppendOwnedOperation(after_block, mlir.oraScfYieldOpCreate(
        mlir_ctx,
        loc,
        &[_]mlir.MlirValue{mlir.oraOperationGetResult(next_op, 0)},
        1,
    ));

    _ = try encoder.encodeOperation(zero_op);
    _ = try encoder.encodeOperation(one_op);

    const encoded = try encoder.encodeValue(mlir.oraOperationGetResult(while_op, 0));
    try testing.expect(!encoder.isDegraded());
    const cmp_ast = try encoder.encodeValue(mlir.oraOperationGetResult(cmp_op, 0));
    const bound_ast = try encoder.encodeValue(bound);
    const zero_ast = try encoder.encodeValue(init);
    const expected = try encoder.encodeControlFlow("scf.if", cmp_ast, bound_ast, zero_ast);
    const expected_str = try testing.allocator.dupe(u8, std.mem.span(z3.Z3_ast_to_string(z3_ctx.ctx, expected)));
    defer testing.allocator.free(expected_str);
    const encoded_str = try testing.allocator.dupe(u8, std.mem.span(z3.Z3_ast_to_string(z3_ctx.ctx, encoded)));
    defer testing.allocator.free(encoded_str);
    try testing.expectEqualStrings(
        expected_str,
        encoded_str,
    );
}

test "scf.while canonical signed positive-delta increment result encodes exactly" {
    var z3_ctx = try Context.init(testing.allocator);
    defer z3_ctx.deinit();

    var encoder = Encoder.init(&z3_ctx, testing.allocator);
    defer encoder.deinit();

    const mlir_ctx = mlir.oraContextCreate();
    defer mlir.oraContextDestroy(mlir_ctx);
    loadAllDialects(mlir_ctx);
    _ = mlir.oraDialectRegister(mlir_ctx);

    const loc = mlir.oraLocationUnknownGet(mlir_ctx);
    const i256_ty = mlir.oraIntegerTypeCreate(mlir_ctx, 256);

    const init_attr = mlir.oraIntegerAttrCreateI64FromType(i256_ty, 0);
    const delta_attr = mlir.oraIntegerAttrCreateI64FromType(i256_ty, 3);
    const init_op = mlir.oraArithConstantOpCreate(mlir_ctx, loc, i256_ty, init_attr);
    const delta_op = mlir.oraArithConstantOpCreate(mlir_ctx, loc, i256_ty, delta_attr);
    const init = mlir.oraOperationGetResult(init_op, 0);
    const bound_op = mlir.oraVariablePlaceholderOpCreate(mlir_ctx, loc, stringRef("signedDeltaBoundValue"), i256_ty);
    const bound = mlir.oraOperationGetResult(bound_op, 0);

    const init_vals = [_]mlir.MlirValue{init};
    const result_types = [_]mlir.MlirType{i256_ty};
    const while_op = mlir.oraScfWhileOpCreate(mlir_ctx, loc, &init_vals, init_vals.len, &result_types, result_types.len);
    const before_block = mlir.oraScfWhileOpGetBeforeBlock(while_op);
    const after_block = mlir.oraScfWhileOpGetAfterBlock(while_op);
    _ = mlir.mlirBlockAddArgument(before_block, i256_ty, loc);
    _ = mlir.mlirBlockAddArgument(after_block, i256_ty, loc);
    const before_arg = mlir.oraBlockGetArgument(before_block, 0);
    const after_arg = mlir.oraBlockGetArgument(after_block, 0);

    const cmp_op = mlir.oraArithCmpIOpCreate(mlir_ctx, loc, 2, before_arg, bound); // slt
    mlir.oraBlockAppendOwnedOperation(before_block, cmp_op);
    mlir.oraBlockAppendOwnedOperation(before_block, mlir.oraScfConditionOpCreate(
        mlir_ctx,
        loc,
        mlir.oraOperationGetResult(cmp_op, 0),
        &[_]mlir.MlirValue{before_arg},
        1,
    ));

    const next_op = mlir.oraArithAddIOpCreate(mlir_ctx, loc, after_arg, mlir.oraOperationGetResult(delta_op, 0));
    mlir.oraBlockAppendOwnedOperation(after_block, next_op);
    mlir.oraBlockAppendOwnedOperation(after_block, mlir.oraScfYieldOpCreate(
        mlir_ctx,
        loc,
        &[_]mlir.MlirValue{mlir.oraOperationGetResult(next_op, 0)},
        1,
    ));

    _ = try encoder.encodeOperation(init_op);
    _ = try encoder.encodeOperation(delta_op);

    const encoded = try encoder.encodeValue(mlir.oraOperationGetResult(while_op, 0));
    try testing.expect(!encoder.isDegraded());

    const init_ast = try encoder.encodeValue(init);
    const bound_ast = try encoder.encodeValue(bound);
    const delta_ast = try encoder.encodeValue(mlir.oraOperationGetResult(delta_op, 0));
    const sort = z3.Z3_get_sort(z3_ctx.ctx, bound_ast);
    const zero = z3.Z3_mk_unsigned_int64(z3_ctx.ctx, 0, sort);
    const one = z3.Z3_mk_unsigned_int64(z3_ctx.ctx, 1, sort);
    const bound_le_init = z3.Z3_mk_bvsle(z3_ctx.ctx, bound_ast, init_ast);
    const distance = z3.Z3_mk_ite(
        z3_ctx.ctx,
        bound_le_init,
        zero,
        z3.Z3_mk_bv_sub(z3_ctx.ctx, bound_ast, init_ast),
    );
    const distance_is_zero = z3.Z3_mk_eq(z3_ctx.ctx, distance, zero);
    const step_count = z3.Z3_mk_ite(
        z3_ctx.ctx,
        distance_is_zero,
        zero,
        z3.Z3_mk_bv_add(
            z3_ctx.ctx,
            z3.Z3_mk_bv_udiv(z3_ctx.ctx, z3.Z3_mk_bv_sub(z3_ctx.ctx, distance, one), delta_ast),
            one,
        ),
    );
    const total_delta = try encoder.encodeArithmeticOp(.Mul, step_count, delta_ast);
    const expected = try encoder.encodeArithmeticOp(.Add, init_ast, total_delta);

    var solver = try Solver.init(&z3_ctx, testing.allocator);
    defer solver.deinit();
    solver.assert(z3.Z3_mk_not(z3_ctx.ctx, z3.Z3_mk_eq(z3_ctx.ctx, encoded, expected)));
    try testing.expectEqual(@as(z3.Z3_lbool, z3.Z3_L_FALSE), solver.check());
}

test "scf.while canonical signed swapped-compare increment result encodes exactly" {
    var z3_ctx = try Context.init(testing.allocator);
    defer z3_ctx.deinit();

    var encoder = Encoder.init(&z3_ctx, testing.allocator);
    defer encoder.deinit();

    const mlir_ctx = mlir.oraContextCreate();
    defer mlir.oraContextDestroy(mlir_ctx);
    loadAllDialects(mlir_ctx);
    _ = mlir.oraDialectRegister(mlir_ctx);

    const loc = mlir.oraLocationUnknownGet(mlir_ctx);
    const i256_ty = mlir.oraIntegerTypeCreate(mlir_ctx, 256);

    const init_attr = mlir.oraIntegerAttrCreateI64FromType(i256_ty, 0);
    const delta_attr = mlir.oraIntegerAttrCreateI64FromType(i256_ty, 3);
    const init_op = mlir.oraArithConstantOpCreate(mlir_ctx, loc, i256_ty, init_attr);
    const delta_op = mlir.oraArithConstantOpCreate(mlir_ctx, loc, i256_ty, delta_attr);
    const bound_op = mlir.oraVariablePlaceholderOpCreate(mlir_ctx, loc, stringRef("signedBoundSwapped"), i256_ty);

    const init_vals = [_]mlir.MlirValue{mlir.oraOperationGetResult(init_op, 0)};
    const result_types = [_]mlir.MlirType{i256_ty};
    const while_op = mlir.oraScfWhileOpCreate(mlir_ctx, loc, &init_vals, init_vals.len, &result_types, result_types.len);
    const before_block = mlir.oraScfWhileOpGetBeforeBlock(while_op);
    const after_block = mlir.oraScfWhileOpGetAfterBlock(while_op);
    _ = mlir.mlirBlockAddArgument(before_block, i256_ty, loc);
    _ = mlir.mlirBlockAddArgument(after_block, i256_ty, loc);
    const before_arg = mlir.oraBlockGetArgument(before_block, 0);
    const after_arg = mlir.oraBlockGetArgument(after_block, 0);

    const cmp_op = mlir.oraArithCmpIOpCreate(mlir_ctx, loc, 4, mlir.oraOperationGetResult(bound_op, 0), before_arg); // bound > current => current < bound
    mlir.oraBlockAppendOwnedOperation(before_block, cmp_op);
    mlir.oraBlockAppendOwnedOperation(before_block, mlir.oraScfConditionOpCreate(
        mlir_ctx,
        loc,
        mlir.oraOperationGetResult(cmp_op, 0),
        &[_]mlir.MlirValue{before_arg},
        1,
    ));

    const next_op = mlir.oraArithAddIOpCreate(mlir_ctx, loc, after_arg, mlir.oraOperationGetResult(delta_op, 0));
    mlir.oraBlockAppendOwnedOperation(after_block, next_op);
    mlir.oraBlockAppendOwnedOperation(after_block, mlir.oraScfYieldOpCreate(
        mlir_ctx,
        loc,
        &[_]mlir.MlirValue{mlir.oraOperationGetResult(next_op, 0)},
        1,
    ));

    _ = try encoder.encodeOperation(init_op);
    _ = try encoder.encodeOperation(delta_op);

    const encoded = try encoder.encodeValue(mlir.oraOperationGetResult(while_op, 0));
    try testing.expect(!encoder.isDegraded());

    const init_ast = try encoder.encodeValue(mlir.oraOperationGetResult(init_op, 0));
    const bound_ast = try encoder.encodeValue(mlir.oraOperationGetResult(bound_op, 0));
    const sort = z3.Z3_get_sort(z3_ctx.ctx, init_ast);
    const zero = z3.Z3_mk_unsigned_int64(z3_ctx.ctx, 0, sort);
    const upper_le_lower = z3.Z3_mk_bvsle(z3_ctx.ctx, bound_ast, init_ast);
    const raw_distance = z3.Z3_mk_bv_sub(z3_ctx.ctx, bound_ast, init_ast);
    const distance = z3.Z3_mk_ite(z3_ctx.ctx, upper_le_lower, zero, raw_distance);
    const one = z3.Z3_mk_unsigned_int64(z3_ctx.ctx, 1, sort);
    const delta_const = z3.Z3_mk_unsigned_int64(z3_ctx.ctx, 3, sort);
    const distance_is_zero = z3.Z3_mk_eq(z3_ctx.ctx, distance, zero);
    const adjusted_distance = z3.Z3_mk_bv_sub(z3_ctx.ctx, distance, one);
    const quotient = z3.Z3_mk_bv_udiv(z3_ctx.ctx, adjusted_distance, delta_const);
    const step_count = z3.Z3_mk_ite(
        z3_ctx.ctx,
        distance_is_zero,
        zero,
        z3.Z3_mk_bv_add(z3_ctx.ctx, quotient, one),
    );
    const delta_ast = try encoder.encodeValue(mlir.oraOperationGetResult(delta_op, 0));
    const total_delta = try encoder.encodeArithmeticOp(.Mul, step_count, delta_ast);
    const expected = try encoder.encodeArithmeticOp(.Add, init_ast, total_delta);

    var solver = try Solver.init(&z3_ctx, testing.allocator);
    defer solver.deinit();
    solver.assert(z3.Z3_mk_not(z3_ctx.ctx, z3.Z3_mk_eq(z3_ctx.ctx, encoded, expected)));
    try testing.expectEqual(@as(z3.Z3_lbool, z3.Z3_L_FALSE), solver.check());
}

test "func.call summary with canonical signed positive-delta no-write scf.while preserves state exactly" {
    var z3_ctx = try Context.init(testing.allocator);
    defer z3_ctx.deinit();

    var encoder = Encoder.init(&z3_ctx, testing.allocator);
    defer encoder.deinit();

    const mlir_ctx = mlir.oraContextCreate();
    defer mlir.oraContextDestroy(mlir_ctx);
    loadAllDialects(mlir_ctx);
    _ = mlir.oraDialectRegister(mlir_ctx);

    const loc = mlir.oraLocationUnknownGet(mlir_ctx);
    const i256_ty = mlir.oraIntegerTypeCreate(mlir_ctx, 256);

    const helper_attrs = [_]mlir.MlirNamedAttribute{
        namedAttr(mlir_ctx, "sym_name", mlir.oraStringAttrCreate(mlir_ctx, stringRef("symbolicSignedDeltaNoWriteWhile"))),
    };
    const helper_param_types = [_]mlir.MlirType{i256_ty};
    const helper_param_locs = [_]mlir.MlirLocation{loc};
    const helper = mlir.oraFuncFuncOpCreate(mlir_ctx, loc, &helper_attrs, helper_attrs.len, &helper_param_types, &helper_param_locs, helper_param_types.len);
    const body = mlir.oraFuncOpGetBodyBlock(helper);

    const init_attr = mlir.oraIntegerAttrCreateI64FromType(i256_ty, 0);
    const delta_attr = mlir.oraIntegerAttrCreateI64FromType(i256_ty, 3);
    const init_op = mlir.oraArithConstantOpCreate(mlir_ctx, loc, i256_ty, init_attr);
    const delta_op = mlir.oraArithConstantOpCreate(mlir_ctx, loc, i256_ty, delta_attr);
    mlir.oraBlockAppendOwnedOperation(body, init_op);
    mlir.oraBlockAppendOwnedOperation(body, delta_op);

    const bound = mlir.oraBlockGetArgument(body, 0);
    const init_vals = [_]mlir.MlirValue{mlir.oraOperationGetResult(init_op, 0)};
    const result_types = [_]mlir.MlirType{i256_ty};
    const while_op = mlir.oraScfWhileOpCreate(mlir_ctx, loc, &init_vals, init_vals.len, &result_types, result_types.len);
    const before_block = mlir.oraScfWhileOpGetBeforeBlock(while_op);
    const after_block = mlir.oraScfWhileOpGetAfterBlock(while_op);
    _ = mlir.mlirBlockAddArgument(before_block, i256_ty, loc);
    _ = mlir.mlirBlockAddArgument(after_block, i256_ty, loc);
    const before_arg = mlir.oraBlockGetArgument(before_block, 0);
    const after_arg = mlir.oraBlockGetArgument(after_block, 0);

    const cmp_op = mlir.oraArithCmpIOpCreate(mlir_ctx, loc, 2, before_arg, bound); // slt
    mlir.oraBlockAppendOwnedOperation(before_block, cmp_op);
    mlir.oraBlockAppendOwnedOperation(before_block, mlir.oraScfConditionOpCreate(
        mlir_ctx,
        loc,
        mlir.oraOperationGetResult(cmp_op, 0),
        &[_]mlir.MlirValue{before_arg},
        1,
    ));

    const next_op = mlir.oraArithAddIOpCreate(mlir_ctx, loc, after_arg, mlir.oraOperationGetResult(delta_op, 0));
    mlir.oraBlockAppendOwnedOperation(after_block, next_op);
    mlir.oraBlockAppendOwnedOperation(after_block, mlir.oraScfYieldOpCreate(
        mlir_ctx,
        loc,
        &[_]mlir.MlirValue{mlir.oraOperationGetResult(next_op, 0)},
        1,
    ));

    mlir.oraBlockAppendOwnedOperation(body, while_op);
    mlir.oraBlockAppendOwnedOperation(body, mlir.oraReturnOpCreate(mlir_ctx, loc, &[_]mlir.MlirValue{}, 0));

    try encoder.registerFunctionOperation(helper);

    const counter_sort = z3.Z3_mk_bv_sort(z3_ctx.ctx, 256);
    const pre_counter = z3.Z3_mk_const(z3_ctx.ctx, z3.Z3_mk_string_symbol(z3_ctx.ctx, "pre_counter_signed_delta_while"), counter_sort);
    try encoder.global_map.put(try testing.allocator.dupe(u8, "counter"), pre_counter);

    const symbolic_bound = mlir.oraVariablePlaceholderOpCreate(mlir_ctx, loc, stringRef("signedDeltaWhileBound"), i256_ty);
    const call = mlir.oraFuncCallOpCreate(
        mlir_ctx,
        loc,
        stringRef("symbolicSignedDeltaNoWriteWhile"),
        &[_]mlir.MlirValue{mlir.oraOperationGetResult(symbolic_bound, 0)},
        1,
        &[_]mlir.MlirType{},
        0,
    );
    _ = try encoder.encodeOperation(call);

    try testing.expect(!encoder.isDegraded());
    const post_counter = encoder.global_map.get("counter").?;

    var solver = try Solver.init(&z3_ctx, testing.allocator);
    defer solver.deinit();
    solver.assert(z3.Z3_mk_not(z3_ctx.ctx, z3.Z3_mk_eq(z3_ctx.ctx, post_counter, pre_counter)));
    try testing.expectEqual(@as(z3.Z3_lbool, z3.Z3_L_FALSE), solver.check());
}

test "known pure callee canonical signed positive-delta scf.while return encodes exactly" {
    var z3_ctx = try Context.init(testing.allocator);
    defer z3_ctx.deinit();

    var encoder = Encoder.init(&z3_ctx, testing.allocator);
    defer encoder.deinit();

    const mlir_ctx = mlir.oraContextCreate();
    defer mlir.oraContextDestroy(mlir_ctx);
    loadAllDialects(mlir_ctx);
    _ = mlir.oraDialectRegister(mlir_ctx);

    const loc = mlir.oraLocationUnknownGet(mlir_ctx);
    const i256_ty = mlir.oraIntegerTypeCreate(mlir_ctx, 256);

    const helper_attrs = [_]mlir.MlirNamedAttribute{
        namedAttr(mlir_ctx, "sym_name", mlir.oraStringAttrCreate(mlir_ctx, stringRef("symbolicSignedDeltaWhileReturn"))),
    };
    const helper_param_types = [_]mlir.MlirType{i256_ty};
    const helper_param_locs = [_]mlir.MlirLocation{loc};
    const helper = mlir.oraFuncFuncOpCreate(mlir_ctx, loc, &helper_attrs, helper_attrs.len, &helper_param_types, &helper_param_locs, helper_param_types.len);
    const body = mlir.oraFuncOpGetBodyBlock(helper);

    const init_attr = mlir.oraIntegerAttrCreateI64FromType(i256_ty, 0);
    const delta_attr = mlir.oraIntegerAttrCreateI64FromType(i256_ty, 3);
    const init_op = mlir.oraArithConstantOpCreate(mlir_ctx, loc, i256_ty, init_attr);
    const delta_op = mlir.oraArithConstantOpCreate(mlir_ctx, loc, i256_ty, delta_attr);
    mlir.oraBlockAppendOwnedOperation(body, init_op);
    mlir.oraBlockAppendOwnedOperation(body, delta_op);

    const bound = mlir.oraBlockGetArgument(body, 0);
    const init_vals = [_]mlir.MlirValue{mlir.oraOperationGetResult(init_op, 0)};
    const result_types = [_]mlir.MlirType{i256_ty};
    const while_op = mlir.oraScfWhileOpCreate(mlir_ctx, loc, &init_vals, init_vals.len, &result_types, result_types.len);
    const before_block = mlir.oraScfWhileOpGetBeforeBlock(while_op);
    const after_block = mlir.oraScfWhileOpGetAfterBlock(while_op);
    _ = mlir.mlirBlockAddArgument(before_block, i256_ty, loc);
    _ = mlir.mlirBlockAddArgument(after_block, i256_ty, loc);
    const before_arg = mlir.oraBlockGetArgument(before_block, 0);
    const after_arg = mlir.oraBlockGetArgument(after_block, 0);

    const cmp_op = mlir.oraArithCmpIOpCreate(mlir_ctx, loc, 2, before_arg, bound); // slt
    mlir.oraBlockAppendOwnedOperation(before_block, cmp_op);
    mlir.oraBlockAppendOwnedOperation(before_block, mlir.oraScfConditionOpCreate(
        mlir_ctx,
        loc,
        mlir.oraOperationGetResult(cmp_op, 0),
        &[_]mlir.MlirValue{before_arg},
        1,
    ));

    const next_op = mlir.oraArithAddIOpCreate(mlir_ctx, loc, after_arg, mlir.oraOperationGetResult(delta_op, 0));
    mlir.oraBlockAppendOwnedOperation(after_block, next_op);
    mlir.oraBlockAppendOwnedOperation(after_block, mlir.oraScfYieldOpCreate(
        mlir_ctx,
        loc,
        &[_]mlir.MlirValue{mlir.oraOperationGetResult(next_op, 0)},
        1,
    ));
    mlir.oraBlockAppendOwnedOperation(body, while_op);
    mlir.oraBlockAppendOwnedOperation(body, mlir.oraReturnOpCreate(
        mlir_ctx,
        loc,
        &[_]mlir.MlirValue{mlir.oraOperationGetResult(while_op, 0)},
        1,
    ));

    try encoder.registerFunctionOperation(helper);

    const caller_bound = mlir.oraVariablePlaceholderOpCreate(mlir_ctx, loc, stringRef("callerSignedDeltaBound"), i256_ty);
    const call = mlir.oraFuncCallOpCreate(
        mlir_ctx,
        loc,
        stringRef("symbolicSignedDeltaWhileReturn"),
        &[_]mlir.MlirValue{mlir.oraOperationGetResult(caller_bound, 0)},
        1,
        &[_]mlir.MlirType{i256_ty},
        1,
    );
    const encoded = try encoder.encodeOperation(call);
    try testing.expect(!encoder.isDegraded());

    const init_ast = try encoder.encodeOperation(init_op);
    const bound_ast = try encoder.encodeOperation(caller_bound);
    const delta_ast = try encoder.encodeOperation(delta_op);
    const sort = z3.Z3_get_sort(z3_ctx.ctx, bound_ast);
    const zero = z3.Z3_mk_unsigned_int64(z3_ctx.ctx, 0, sort);
    const one = z3.Z3_mk_unsigned_int64(z3_ctx.ctx, 1, sort);
    const bound_le_init = z3.Z3_mk_bvsle(z3_ctx.ctx, bound_ast, init_ast);
    const distance = z3.Z3_mk_ite(
        z3_ctx.ctx,
        bound_le_init,
        zero,
        z3.Z3_mk_bv_sub(z3_ctx.ctx, bound_ast, init_ast),
    );
    const distance_is_zero = z3.Z3_mk_eq(z3_ctx.ctx, distance, zero);
    const step_count = z3.Z3_mk_ite(
        z3_ctx.ctx,
        distance_is_zero,
        zero,
        z3.Z3_mk_bv_add(
            z3_ctx.ctx,
            z3.Z3_mk_bv_udiv(z3_ctx.ctx, z3.Z3_mk_bv_sub(z3_ctx.ctx, distance, one), delta_ast),
            one,
        ),
    );
    const total_delta = try encoder.encodeArithmeticOp(.Mul, step_count, delta_ast);
    const expected = try encoder.encodeArithmeticOp(.Add, init_ast, total_delta);

    var solver = try Solver.init(&z3_ctx, testing.allocator);
    defer solver.deinit();
    solver.assert(z3.Z3_mk_not(z3_ctx.ctx, z3.Z3_mk_eq(z3_ctx.ctx, encoded, expected)));
    try testing.expectEqual(@as(z3.Z3_lbool, z3.Z3_L_FALSE), solver.check());
}

test "scf.while canonical signed positive-delta decrement result encodes exactly" {
    var z3_ctx = try Context.init(testing.allocator);
    defer z3_ctx.deinit();

    var encoder = Encoder.init(&z3_ctx, testing.allocator);
    defer encoder.deinit();

    const mlir_ctx = mlir.oraContextCreate();
    defer mlir.oraContextDestroy(mlir_ctx);
    loadAllDialects(mlir_ctx);
    _ = mlir.oraDialectRegister(mlir_ctx);

    const loc = mlir.oraLocationUnknownGet(mlir_ctx);
    const i256_ty = mlir.oraIntegerTypeCreate(mlir_ctx, 256);

    const init_attr = mlir.oraIntegerAttrCreateI64FromType(i256_ty, 10);
    const delta_attr = mlir.oraIntegerAttrCreateI64FromType(i256_ty, 3);
    const init_op = mlir.oraArithConstantOpCreate(mlir_ctx, loc, i256_ty, init_attr);
    const delta_op = mlir.oraArithConstantOpCreate(mlir_ctx, loc, i256_ty, delta_attr);
    const init = mlir.oraOperationGetResult(init_op, 0);
    const bound_op = mlir.oraVariablePlaceholderOpCreate(mlir_ctx, loc, stringRef("signedDeltaDecBoundValue"), i256_ty);
    const bound = mlir.oraOperationGetResult(bound_op, 0);

    const init_vals = [_]mlir.MlirValue{init};
    const result_types = [_]mlir.MlirType{i256_ty};
    const while_op = mlir.oraScfWhileOpCreate(mlir_ctx, loc, &init_vals, init_vals.len, &result_types, result_types.len);
    const before_block = mlir.oraScfWhileOpGetBeforeBlock(while_op);
    const after_block = mlir.oraScfWhileOpGetAfterBlock(while_op);
    _ = mlir.mlirBlockAddArgument(before_block, i256_ty, loc);
    _ = mlir.mlirBlockAddArgument(after_block, i256_ty, loc);
    const before_arg = mlir.oraBlockGetArgument(before_block, 0);
    const after_arg = mlir.oraBlockGetArgument(after_block, 0);

    const cmp_op = mlir.oraArithCmpIOpCreate(mlir_ctx, loc, 4, before_arg, bound); // sgt
    mlir.oraBlockAppendOwnedOperation(before_block, cmp_op);
    mlir.oraBlockAppendOwnedOperation(before_block, mlir.oraScfConditionOpCreate(
        mlir_ctx,
        loc,
        mlir.oraOperationGetResult(cmp_op, 0),
        &[_]mlir.MlirValue{before_arg},
        1,
    ));

    const next_op = mlir.oraArithSubIOpCreate(mlir_ctx, loc, after_arg, mlir.oraOperationGetResult(delta_op, 0));
    mlir.oraBlockAppendOwnedOperation(after_block, next_op);
    mlir.oraBlockAppendOwnedOperation(after_block, mlir.oraScfYieldOpCreate(
        mlir_ctx,
        loc,
        &[_]mlir.MlirValue{mlir.oraOperationGetResult(next_op, 0)},
        1,
    ));

    _ = try encoder.encodeOperation(init_op);
    _ = try encoder.encodeOperation(delta_op);

    const encoded = try encoder.encodeValue(mlir.oraOperationGetResult(while_op, 0));
    try testing.expect(!encoder.isDegraded());

    const init_ast = try encoder.encodeValue(init);
    const bound_ast = try encoder.encodeValue(bound);
    const delta_ast = try encoder.encodeValue(mlir.oraOperationGetResult(delta_op, 0));
    const sort = z3.Z3_get_sort(z3_ctx.ctx, bound_ast);
    const zero = z3.Z3_mk_unsigned_int64(z3_ctx.ctx, 0, sort);
    const one = z3.Z3_mk_unsigned_int64(z3_ctx.ctx, 1, sort);
    const init_le_bound = z3.Z3_mk_bvsle(z3_ctx.ctx, init_ast, bound_ast);
    const distance = z3.Z3_mk_ite(
        z3_ctx.ctx,
        init_le_bound,
        zero,
        z3.Z3_mk_bv_sub(z3_ctx.ctx, init_ast, bound_ast),
    );
    const distance_is_zero = z3.Z3_mk_eq(z3_ctx.ctx, distance, zero);
    const step_count = z3.Z3_mk_ite(
        z3_ctx.ctx,
        distance_is_zero,
        zero,
        z3.Z3_mk_bv_add(
            z3_ctx.ctx,
            z3.Z3_mk_bv_udiv(z3_ctx.ctx, z3.Z3_mk_bv_sub(z3_ctx.ctx, distance, one), delta_ast),
            one,
        ),
    );
    const total_delta = try encoder.encodeArithmeticOp(.Mul, step_count, delta_ast);
    const expected = try encoder.encodeArithmeticOp(.Sub, init_ast, total_delta);

    var solver = try Solver.init(&z3_ctx, testing.allocator);
    defer solver.deinit();
    solver.assert(z3.Z3_mk_not(z3_ctx.ctx, z3.Z3_mk_eq(z3_ctx.ctx, encoded, expected)));
    try testing.expectEqual(@as(z3.Z3_lbool, z3.Z3_L_FALSE), solver.check());
}

test "known pure callee canonical signed positive-delta decrement scf.while return encodes exactly" {
    var z3_ctx = try Context.init(testing.allocator);
    defer z3_ctx.deinit();

    var encoder = Encoder.init(&z3_ctx, testing.allocator);
    defer encoder.deinit();

    const mlir_ctx = mlir.oraContextCreate();
    defer mlir.oraContextDestroy(mlir_ctx);
    loadAllDialects(mlir_ctx);
    _ = mlir.oraDialectRegister(mlir_ctx);

    const loc = mlir.oraLocationUnknownGet(mlir_ctx);
    const i256_ty = mlir.oraIntegerTypeCreate(mlir_ctx, 256);

    const helper_attrs = [_]mlir.MlirNamedAttribute{
        namedAttr(mlir_ctx, "sym_name", mlir.oraStringAttrCreate(mlir_ctx, stringRef("symbolicSignedDeltaDecWhileReturn"))),
    };
    const helper_param_types = [_]mlir.MlirType{i256_ty};
    const helper_param_locs = [_]mlir.MlirLocation{loc};
    const helper = mlir.oraFuncFuncOpCreate(mlir_ctx, loc, &helper_attrs, helper_attrs.len, &helper_param_types, &helper_param_locs, helper_param_types.len);
    const body = mlir.oraFuncOpGetBodyBlock(helper);

    const init_attr = mlir.oraIntegerAttrCreateI64FromType(i256_ty, 10);
    const delta_attr = mlir.oraIntegerAttrCreateI64FromType(i256_ty, 3);
    const init_op = mlir.oraArithConstantOpCreate(mlir_ctx, loc, i256_ty, init_attr);
    const delta_op = mlir.oraArithConstantOpCreate(mlir_ctx, loc, i256_ty, delta_attr);
    mlir.oraBlockAppendOwnedOperation(body, init_op);
    mlir.oraBlockAppendOwnedOperation(body, delta_op);

    const bound = mlir.oraBlockGetArgument(body, 0);
    const init_vals = [_]mlir.MlirValue{mlir.oraOperationGetResult(init_op, 0)};
    const result_types = [_]mlir.MlirType{i256_ty};
    const while_op = mlir.oraScfWhileOpCreate(mlir_ctx, loc, &init_vals, init_vals.len, &result_types, result_types.len);
    const before_block = mlir.oraScfWhileOpGetBeforeBlock(while_op);
    const after_block = mlir.oraScfWhileOpGetAfterBlock(while_op);
    _ = mlir.mlirBlockAddArgument(before_block, i256_ty, loc);
    _ = mlir.mlirBlockAddArgument(after_block, i256_ty, loc);
    const before_arg = mlir.oraBlockGetArgument(before_block, 0);
    const after_arg = mlir.oraBlockGetArgument(after_block, 0);

    const cmp_op = mlir.oraArithCmpIOpCreate(mlir_ctx, loc, 4, before_arg, bound); // sgt
    mlir.oraBlockAppendOwnedOperation(before_block, cmp_op);
    mlir.oraBlockAppendOwnedOperation(before_block, mlir.oraScfConditionOpCreate(
        mlir_ctx,
        loc,
        mlir.oraOperationGetResult(cmp_op, 0),
        &[_]mlir.MlirValue{before_arg},
        1,
    ));

    const next_op = mlir.oraArithSubIOpCreate(mlir_ctx, loc, after_arg, mlir.oraOperationGetResult(delta_op, 0));
    mlir.oraBlockAppendOwnedOperation(after_block, next_op);
    mlir.oraBlockAppendOwnedOperation(after_block, mlir.oraScfYieldOpCreate(
        mlir_ctx,
        loc,
        &[_]mlir.MlirValue{mlir.oraOperationGetResult(next_op, 0)},
        1,
    ));
    mlir.oraBlockAppendOwnedOperation(body, while_op);
    mlir.oraBlockAppendOwnedOperation(body, mlir.oraReturnOpCreate(
        mlir_ctx,
        loc,
        &[_]mlir.MlirValue{mlir.oraOperationGetResult(while_op, 0)},
        1,
    ));

    try encoder.registerFunctionOperation(helper);

    const caller_bound = mlir.oraVariablePlaceholderOpCreate(mlir_ctx, loc, stringRef("callerSignedDeltaDecBound"), i256_ty);
    const call = mlir.oraFuncCallOpCreate(
        mlir_ctx,
        loc,
        stringRef("symbolicSignedDeltaDecWhileReturn"),
        &[_]mlir.MlirValue{mlir.oraOperationGetResult(caller_bound, 0)},
        1,
        &[_]mlir.MlirType{i256_ty},
        1,
    );
    const encoded = try encoder.encodeOperation(call);
    try testing.expect(!encoder.isDegraded());

    const init_ast = try encoder.encodeOperation(init_op);
    const bound_ast = try encoder.encodeOperation(caller_bound);
    const delta_ast = try encoder.encodeOperation(delta_op);
    const sort = z3.Z3_get_sort(z3_ctx.ctx, bound_ast);
    const zero = z3.Z3_mk_unsigned_int64(z3_ctx.ctx, 0, sort);
    const one = z3.Z3_mk_unsigned_int64(z3_ctx.ctx, 1, sort);
    const init_le_bound = z3.Z3_mk_bvsle(z3_ctx.ctx, init_ast, bound_ast);
    const distance = z3.Z3_mk_ite(
        z3_ctx.ctx,
        init_le_bound,
        zero,
        z3.Z3_mk_bv_sub(z3_ctx.ctx, init_ast, bound_ast),
    );
    const distance_is_zero = z3.Z3_mk_eq(z3_ctx.ctx, distance, zero);
    const step_count = z3.Z3_mk_ite(
        z3_ctx.ctx,
        distance_is_zero,
        zero,
        z3.Z3_mk_bv_add(
            z3_ctx.ctx,
            z3.Z3_mk_bv_udiv(z3_ctx.ctx, z3.Z3_mk_bv_sub(z3_ctx.ctx, distance, one), delta_ast),
            one,
        ),
    );
    const total_delta = try encoder.encodeArithmeticOp(.Mul, step_count, delta_ast);
    const expected = try encoder.encodeArithmeticOp(.Sub, init_ast, total_delta);

    var solver = try Solver.init(&z3_ctx, testing.allocator);
    defer solver.deinit();
    solver.assert(z3.Z3_mk_not(z3_ctx.ctx, z3.Z3_mk_eq(z3_ctx.ctx, encoded, expected)));
    try testing.expectEqual(@as(z3.Z3_lbool, z3.Z3_L_FALSE), solver.check());
}

test "direct canonical signed multi-result scf.while encodes derived result exactly" {
    var z3_ctx = try Context.init(testing.allocator);
    defer z3_ctx.deinit();

    var encoder = Encoder.init(&z3_ctx, testing.allocator);
    defer encoder.deinit();

    const mlir_ctx = mlir.oraContextCreate();
    defer mlir.oraContextDestroy(mlir_ctx);
    loadAllDialects(mlir_ctx);
    _ = mlir.oraDialectRegister(mlir_ctx);

    const loc = mlir.oraLocationUnknownGet(mlir_ctx);
    const i256_ty = mlir.oraIntegerTypeCreate(mlir_ctx, 256);

    const init_ctrl_attr = mlir.oraIntegerAttrCreateI64FromType(i256_ty, 0);
    const init_sum_attr = mlir.oraIntegerAttrCreateI64FromType(i256_ty, 10);
    const ctrl_delta_attr = mlir.oraIntegerAttrCreateI64FromType(i256_ty, 3);
    const sum_delta_attr = mlir.oraIntegerAttrCreateI64FromType(i256_ty, 5);
    const init_ctrl_op = mlir.oraArithConstantOpCreate(mlir_ctx, loc, i256_ty, init_ctrl_attr);
    const init_sum_op = mlir.oraArithConstantOpCreate(mlir_ctx, loc, i256_ty, init_sum_attr);
    const ctrl_delta_op = mlir.oraArithConstantOpCreate(mlir_ctx, loc, i256_ty, ctrl_delta_attr);
    const sum_delta_op = mlir.oraArithConstantOpCreate(mlir_ctx, loc, i256_ty, sum_delta_attr);
    const bound_op = mlir.oraVariablePlaceholderOpCreate(mlir_ctx, loc, stringRef("signedMultiBound"), i256_ty);

    const init_vals = [_]mlir.MlirValue{
        mlir.oraOperationGetResult(init_ctrl_op, 0),
        mlir.oraOperationGetResult(init_sum_op, 0),
    };
    const result_types = [_]mlir.MlirType{ i256_ty, i256_ty };
    const while_op = mlir.oraScfWhileOpCreate(mlir_ctx, loc, &init_vals, init_vals.len, &result_types, result_types.len);
    const before_block = mlir.oraScfWhileOpGetBeforeBlock(while_op);
    const after_block = mlir.oraScfWhileOpGetAfterBlock(while_op);
    _ = mlir.mlirBlockAddArgument(before_block, i256_ty, loc);
    _ = mlir.mlirBlockAddArgument(before_block, i256_ty, loc);
    _ = mlir.mlirBlockAddArgument(after_block, i256_ty, loc);
    _ = mlir.mlirBlockAddArgument(after_block, i256_ty, loc);

    const before_ctrl = mlir.oraBlockGetArgument(before_block, 0);
    const before_sum = mlir.oraBlockGetArgument(before_block, 1);
    const cmp_op = mlir.oraArithCmpIOpCreate(mlir_ctx, loc, 2, before_ctrl, mlir.oraOperationGetResult(bound_op, 0)); // slt
    mlir.oraBlockAppendOwnedOperation(before_block, cmp_op);
    mlir.oraBlockAppendOwnedOperation(before_block, mlir.oraScfConditionOpCreate(
        mlir_ctx,
        loc,
        mlir.oraOperationGetResult(cmp_op, 0),
        &[_]mlir.MlirValue{ before_ctrl, before_sum },
        2,
    ));

    const after_ctrl = mlir.oraBlockGetArgument(after_block, 0);
    const after_sum = mlir.oraBlockGetArgument(after_block, 1);
    const next_ctrl = mlir.oraArithAddIOpCreate(mlir_ctx, loc, after_ctrl, mlir.oraOperationGetResult(ctrl_delta_op, 0));
    const next_sum = mlir.oraArithAddIOpCreate(mlir_ctx, loc, after_sum, mlir.oraOperationGetResult(sum_delta_op, 0));
    mlir.oraBlockAppendOwnedOperation(after_block, next_ctrl);
    mlir.oraBlockAppendOwnedOperation(after_block, next_sum);
    mlir.oraBlockAppendOwnedOperation(after_block, mlir.oraScfYieldOpCreate(
        mlir_ctx,
        loc,
        &[_]mlir.MlirValue{
            mlir.oraOperationGetResult(next_ctrl, 0),
            mlir.oraOperationGetResult(next_sum, 0),
        },
        2,
    ));

    const encoded = try encoder.encodeValue(mlir.oraOperationGetResult(while_op, 1));
    try testing.expect(!encoder.isDegraded());

    const init_ctrl_ast = try encoder.encodeOperation(init_ctrl_op);
    const init_sum_ast = try encoder.encodeOperation(init_sum_op);
    const ctrl_delta_ast = try encoder.encodeOperation(ctrl_delta_op);
    const sum_delta_ast = try encoder.encodeOperation(sum_delta_op);
    const bound_ast = try encoder.encodeOperation(bound_op);
    const zero = try encoder.encodeIntegerConstant(0, 256);
    const one = try encoder.encodeIntegerConstant(1, 256);
    const bound_le_init = z3.Z3_mk_bvsle(z3_ctx.ctx, bound_ast, init_ctrl_ast);
    const distance = z3.Z3_mk_ite(
        z3_ctx.ctx,
        bound_le_init,
        zero,
        z3.Z3_mk_bv_sub(z3_ctx.ctx, bound_ast, init_ctrl_ast),
    );
    const distance_is_zero = z3.Z3_mk_eq(z3_ctx.ctx, distance, zero);
    const step_count = z3.Z3_mk_ite(
        z3_ctx.ctx,
        distance_is_zero,
        zero,
        z3.Z3_mk_bv_add(
            z3_ctx.ctx,
            z3.Z3_mk_bv_udiv(z3_ctx.ctx, z3.Z3_mk_bv_sub(z3_ctx.ctx, distance, one), ctrl_delta_ast),
            one,
        ),
    );
    const total_delta = try encoder.encodeArithmeticOp(.Mul, step_count, sum_delta_ast);
    const expected = try encoder.encodeArithmeticOp(.Add, init_sum_ast, total_delta);

    var solver = try Solver.init(&z3_ctx, testing.allocator);
    defer solver.deinit();
    solver.assert(z3.Z3_mk_not(z3_ctx.ctx, z3.Z3_mk_eq(z3_ctx.ctx, encoded, expected)));
    try testing.expectEqual(@as(z3.Z3_lbool, z3.Z3_L_FALSE), solver.check());
}

test "known pure callee canonical signed multi-result scf.while return encodes derived result exactly" {
    var z3_ctx = try Context.init(testing.allocator);
    defer z3_ctx.deinit();

    var encoder = Encoder.init(&z3_ctx, testing.allocator);
    defer encoder.deinit();

    const mlir_ctx = mlir.oraContextCreate();
    defer mlir.oraContextDestroy(mlir_ctx);
    loadAllDialects(mlir_ctx);
    _ = mlir.oraDialectRegister(mlir_ctx);

    const loc = mlir.oraLocationUnknownGet(mlir_ctx);
    const i256_ty = mlir.oraIntegerTypeCreate(mlir_ctx, 256);

    const helper_attrs = [_]mlir.MlirNamedAttribute{
        namedAttr(mlir_ctx, "sym_name", mlir.oraStringAttrCreate(mlir_ctx, stringRef("symbolicSignedMultiWhileReturn"))),
    };
    const helper_param_types = [_]mlir.MlirType{i256_ty};
    const helper_param_locs = [_]mlir.MlirLocation{loc};
    const helper = mlir.oraFuncFuncOpCreate(mlir_ctx, loc, &helper_attrs, helper_attrs.len, &helper_param_types, &helper_param_locs, helper_param_types.len);
    const body = mlir.oraFuncOpGetBodyBlock(helper);

    const init_ctrl_attr = mlir.oraIntegerAttrCreateI64FromType(i256_ty, 0);
    const init_sum_attr = mlir.oraIntegerAttrCreateI64FromType(i256_ty, 10);
    const ctrl_delta_attr = mlir.oraIntegerAttrCreateI64FromType(i256_ty, 3);
    const sum_delta_attr = mlir.oraIntegerAttrCreateI64FromType(i256_ty, 5);
    const init_ctrl_op = mlir.oraArithConstantOpCreate(mlir_ctx, loc, i256_ty, init_ctrl_attr);
    const init_sum_op = mlir.oraArithConstantOpCreate(mlir_ctx, loc, i256_ty, init_sum_attr);
    const ctrl_delta_op = mlir.oraArithConstantOpCreate(mlir_ctx, loc, i256_ty, ctrl_delta_attr);
    const sum_delta_op = mlir.oraArithConstantOpCreate(mlir_ctx, loc, i256_ty, sum_delta_attr);
    mlir.oraBlockAppendOwnedOperation(body, init_ctrl_op);
    mlir.oraBlockAppendOwnedOperation(body, init_sum_op);
    mlir.oraBlockAppendOwnedOperation(body, ctrl_delta_op);
    mlir.oraBlockAppendOwnedOperation(body, sum_delta_op);

    const bound = mlir.oraBlockGetArgument(body, 0);
    const init_vals = [_]mlir.MlirValue{
        mlir.oraOperationGetResult(init_ctrl_op, 0),
        mlir.oraOperationGetResult(init_sum_op, 0),
    };
    const result_types = [_]mlir.MlirType{ i256_ty, i256_ty };
    const while_op = mlir.oraScfWhileOpCreate(mlir_ctx, loc, &init_vals, init_vals.len, &result_types, result_types.len);
    const before_block = mlir.oraScfWhileOpGetBeforeBlock(while_op);
    const after_block = mlir.oraScfWhileOpGetAfterBlock(while_op);
    _ = mlir.mlirBlockAddArgument(before_block, i256_ty, loc);
    _ = mlir.mlirBlockAddArgument(before_block, i256_ty, loc);
    _ = mlir.mlirBlockAddArgument(after_block, i256_ty, loc);
    _ = mlir.mlirBlockAddArgument(after_block, i256_ty, loc);

    const before_ctrl = mlir.oraBlockGetArgument(before_block, 0);
    const before_sum = mlir.oraBlockGetArgument(before_block, 1);
    const cmp_op = mlir.oraArithCmpIOpCreate(mlir_ctx, loc, 2, before_ctrl, bound); // slt
    mlir.oraBlockAppendOwnedOperation(before_block, cmp_op);
    mlir.oraBlockAppendOwnedOperation(before_block, mlir.oraScfConditionOpCreate(
        mlir_ctx,
        loc,
        mlir.oraOperationGetResult(cmp_op, 0),
        &[_]mlir.MlirValue{ before_ctrl, before_sum },
        2,
    ));

    const after_ctrl = mlir.oraBlockGetArgument(after_block, 0);
    const after_sum = mlir.oraBlockGetArgument(after_block, 1);
    const next_ctrl = mlir.oraArithAddIOpCreate(mlir_ctx, loc, after_ctrl, mlir.oraOperationGetResult(ctrl_delta_op, 0));
    const next_sum = mlir.oraArithAddIOpCreate(mlir_ctx, loc, after_sum, mlir.oraOperationGetResult(sum_delta_op, 0));
    mlir.oraBlockAppendOwnedOperation(after_block, next_ctrl);
    mlir.oraBlockAppendOwnedOperation(after_block, next_sum);
    mlir.oraBlockAppendOwnedOperation(after_block, mlir.oraScfYieldOpCreate(
        mlir_ctx,
        loc,
        &[_]mlir.MlirValue{
            mlir.oraOperationGetResult(next_ctrl, 0),
            mlir.oraOperationGetResult(next_sum, 0),
        },
        2,
    ));
    mlir.oraBlockAppendOwnedOperation(body, while_op);
    mlir.oraBlockAppendOwnedOperation(body, mlir.oraReturnOpCreate(
        mlir_ctx,
        loc,
        &[_]mlir.MlirValue{mlir.oraOperationGetResult(while_op, 1)},
        1,
    ));

    try encoder.registerFunctionOperation(helper);

    const caller_bound = mlir.oraVariablePlaceholderOpCreate(mlir_ctx, loc, stringRef("callerSignedMultiBound"), i256_ty);
    const call = mlir.oraFuncCallOpCreate(
        mlir_ctx,
        loc,
        stringRef("symbolicSignedMultiWhileReturn"),
        &[_]mlir.MlirValue{mlir.oraOperationGetResult(caller_bound, 0)},
        1,
        &[_]mlir.MlirType{i256_ty},
        1,
    );
    const encoded = try encoder.encodeOperation(call);
    try testing.expect(!encoder.isDegraded());

    const init_ctrl_ast = try encoder.encodeOperation(init_ctrl_op);
    const init_sum_ast = try encoder.encodeOperation(init_sum_op);
    const ctrl_delta_ast = try encoder.encodeOperation(ctrl_delta_op);
    const sum_delta_ast = try encoder.encodeOperation(sum_delta_op);
    const bound_ast = try encoder.encodeOperation(caller_bound);
    const zero = try encoder.encodeIntegerConstant(0, 256);
    const one = try encoder.encodeIntegerConstant(1, 256);
    const bound_le_init = z3.Z3_mk_bvsle(z3_ctx.ctx, bound_ast, init_ctrl_ast);
    const distance = z3.Z3_mk_ite(
        z3_ctx.ctx,
        bound_le_init,
        zero,
        z3.Z3_mk_bv_sub(z3_ctx.ctx, bound_ast, init_ctrl_ast),
    );
    const distance_is_zero = z3.Z3_mk_eq(z3_ctx.ctx, distance, zero);
    const step_count = z3.Z3_mk_ite(
        z3_ctx.ctx,
        distance_is_zero,
        zero,
        z3.Z3_mk_bv_add(
            z3_ctx.ctx,
            z3.Z3_mk_bv_udiv(z3_ctx.ctx, z3.Z3_mk_bv_sub(z3_ctx.ctx, distance, one), ctrl_delta_ast),
            one,
        ),
    );
    const total_delta = try encoder.encodeArithmeticOp(.Mul, step_count, sum_delta_ast);
    const expected = try encoder.encodeArithmeticOp(.Add, init_sum_ast, total_delta);

    var solver = try Solver.init(&z3_ctx, testing.allocator);
    defer solver.deinit();
    solver.assert(z3.Z3_mk_not(z3_ctx.ctx, z3.Z3_mk_eq(z3_ctx.ctx, encoded, expected)));
    try testing.expectEqual(@as(z3.Z3_lbool, z3.Z3_L_FALSE), solver.check());
}

test "direct canonical signed scf.while with nonzero control index encodes derived result exactly" {
    var z3_ctx = try Context.init(testing.allocator);
    defer z3_ctx.deinit();

    var encoder = Encoder.init(&z3_ctx, testing.allocator);
    defer encoder.deinit();

    const mlir_ctx = mlir.oraContextCreate();
    defer mlir.oraContextDestroy(mlir_ctx);
    loadAllDialects(mlir_ctx);
    _ = mlir.oraDialectRegister(mlir_ctx);

    const loc = mlir.oraLocationUnknownGet(mlir_ctx);
    const i256_ty = mlir.oraIntegerTypeCreate(mlir_ctx, 256);

    const init_sum_attr = mlir.oraIntegerAttrCreateI64FromType(i256_ty, 10);
    const init_ctrl_attr = mlir.oraIntegerAttrCreateI64FromType(i256_ty, 0);
    const sum_delta_attr = mlir.oraIntegerAttrCreateI64FromType(i256_ty, 5);
    const ctrl_delta_attr = mlir.oraIntegerAttrCreateI64FromType(i256_ty, 3);
    const init_sum_op = mlir.oraArithConstantOpCreate(mlir_ctx, loc, i256_ty, init_sum_attr);
    const init_ctrl_op = mlir.oraArithConstantOpCreate(mlir_ctx, loc, i256_ty, init_ctrl_attr);
    const sum_delta_op = mlir.oraArithConstantOpCreate(mlir_ctx, loc, i256_ty, sum_delta_attr);
    const ctrl_delta_op = mlir.oraArithConstantOpCreate(mlir_ctx, loc, i256_ty, ctrl_delta_attr);
    const bound_op = mlir.oraVariablePlaceholderOpCreate(mlir_ctx, loc, stringRef("signedControlSecondBound"), i256_ty);

    const init_vals = [_]mlir.MlirValue{
        mlir.oraOperationGetResult(init_sum_op, 0),
        mlir.oraOperationGetResult(init_ctrl_op, 0),
    };
    const result_types = [_]mlir.MlirType{ i256_ty, i256_ty };
    const while_op = mlir.oraScfWhileOpCreate(mlir_ctx, loc, &init_vals, init_vals.len, &result_types, result_types.len);
    const before_block = mlir.oraScfWhileOpGetBeforeBlock(while_op);
    const after_block = mlir.oraScfWhileOpGetAfterBlock(while_op);
    _ = mlir.mlirBlockAddArgument(before_block, i256_ty, loc);
    _ = mlir.mlirBlockAddArgument(before_block, i256_ty, loc);
    _ = mlir.mlirBlockAddArgument(after_block, i256_ty, loc);
    _ = mlir.mlirBlockAddArgument(after_block, i256_ty, loc);

    const before_ctrl = mlir.oraBlockGetArgument(before_block, 1);
    const cmp_op = mlir.oraArithCmpIOpCreate(mlir_ctx, loc, 2, before_ctrl, mlir.oraOperationGetResult(bound_op, 0)); // slt
    mlir.oraBlockAppendOwnedOperation(before_block, cmp_op);
    mlir.oraBlockAppendOwnedOperation(before_block, mlir.oraScfConditionOpCreate(
        mlir_ctx,
        loc,
        mlir.oraOperationGetResult(cmp_op, 0),
        &[_]mlir.MlirValue{
            mlir.oraBlockGetArgument(before_block, 0),
            before_ctrl,
        },
        2,
    ));

    const after_sum = mlir.oraBlockGetArgument(after_block, 0);
    const after_ctrl = mlir.oraBlockGetArgument(after_block, 1);
    const next_sum = mlir.oraArithAddIOpCreate(mlir_ctx, loc, after_sum, mlir.oraOperationGetResult(sum_delta_op, 0));
    const next_ctrl = mlir.oraArithAddIOpCreate(mlir_ctx, loc, after_ctrl, mlir.oraOperationGetResult(ctrl_delta_op, 0));
    mlir.oraBlockAppendOwnedOperation(after_block, next_sum);
    mlir.oraBlockAppendOwnedOperation(after_block, next_ctrl);
    mlir.oraBlockAppendOwnedOperation(after_block, mlir.oraScfYieldOpCreate(
        mlir_ctx,
        loc,
        &[_]mlir.MlirValue{
            mlir.oraOperationGetResult(next_sum, 0),
            mlir.oraOperationGetResult(next_ctrl, 0),
        },
        2,
    ));

    const encoded = try encoder.encodeValue(mlir.oraOperationGetResult(while_op, 0));
    try testing.expect(!encoder.isDegraded());

    const init_sum_ast = try encoder.encodeOperation(init_sum_op);
    const init_ctrl_ast = try encoder.encodeOperation(init_ctrl_op);
    const sum_delta_ast = try encoder.encodeOperation(sum_delta_op);
    const ctrl_delta_ast = try encoder.encodeOperation(ctrl_delta_op);
    const bound_ast = try encoder.encodeOperation(bound_op);
    const zero = try encoder.encodeIntegerConstant(0, 256);
    const one = try encoder.encodeIntegerConstant(1, 256);
    const bound_le_init = z3.Z3_mk_bvsle(z3_ctx.ctx, bound_ast, init_ctrl_ast);
    const distance = z3.Z3_mk_ite(
        z3_ctx.ctx,
        bound_le_init,
        zero,
        z3.Z3_mk_bv_sub(z3_ctx.ctx, bound_ast, init_ctrl_ast),
    );
    const distance_is_zero = z3.Z3_mk_eq(z3_ctx.ctx, distance, zero);
    const step_count = z3.Z3_mk_ite(
        z3_ctx.ctx,
        distance_is_zero,
        zero,
        z3.Z3_mk_bv_add(
            z3_ctx.ctx,
            z3.Z3_mk_bv_udiv(z3_ctx.ctx, z3.Z3_mk_bv_sub(z3_ctx.ctx, distance, one), ctrl_delta_ast),
            one,
        ),
    );
    const total_delta = try encoder.encodeArithmeticOp(.Mul, step_count, sum_delta_ast);
    const expected = try encoder.encodeArithmeticOp(.Add, init_sum_ast, total_delta);

    var solver = try Solver.init(&z3_ctx, testing.allocator);
    defer solver.deinit();
    solver.assert(z3.Z3_mk_not(z3_ctx.ctx, z3.Z3_mk_eq(z3_ctx.ctx, encoded, expected)));
    try testing.expectEqual(@as(z3.Z3_lbool, z3.Z3_L_FALSE), solver.check());
}

test "known pure callee canonical signed scf.while with nonzero control index return encodes derived result exactly" {
    var z3_ctx = try Context.init(testing.allocator);
    defer z3_ctx.deinit();

    var encoder = Encoder.init(&z3_ctx, testing.allocator);
    defer encoder.deinit();

    const mlir_ctx = mlir.oraContextCreate();
    defer mlir.oraContextDestroy(mlir_ctx);
    loadAllDialects(mlir_ctx);
    _ = mlir.oraDialectRegister(mlir_ctx);

    const loc = mlir.oraLocationUnknownGet(mlir_ctx);
    const i256_ty = mlir.oraIntegerTypeCreate(mlir_ctx, 256);

    const helper_attrs = [_]mlir.MlirNamedAttribute{
        namedAttr(mlir_ctx, "sym_name", mlir.oraStringAttrCreate(mlir_ctx, stringRef("signedWhileControlSecondReturn"))),
    };
    const helper_param_types = [_]mlir.MlirType{i256_ty};
    const helper_param_locs = [_]mlir.MlirLocation{loc};
    const helper = mlir.oraFuncFuncOpCreate(mlir_ctx, loc, &helper_attrs, helper_attrs.len, &helper_param_types, &helper_param_locs, helper_param_types.len);
    const body = mlir.oraFuncOpGetBodyBlock(helper);

    const init_sum_attr = mlir.oraIntegerAttrCreateI64FromType(i256_ty, 10);
    const init_ctrl_attr = mlir.oraIntegerAttrCreateI64FromType(i256_ty, 0);
    const sum_delta_attr = mlir.oraIntegerAttrCreateI64FromType(i256_ty, 5);
    const ctrl_delta_attr = mlir.oraIntegerAttrCreateI64FromType(i256_ty, 3);
    const init_sum_op = mlir.oraArithConstantOpCreate(mlir_ctx, loc, i256_ty, init_sum_attr);
    const init_ctrl_op = mlir.oraArithConstantOpCreate(mlir_ctx, loc, i256_ty, init_ctrl_attr);
    const sum_delta_op = mlir.oraArithConstantOpCreate(mlir_ctx, loc, i256_ty, sum_delta_attr);
    const ctrl_delta_op = mlir.oraArithConstantOpCreate(mlir_ctx, loc, i256_ty, ctrl_delta_attr);
    mlir.oraBlockAppendOwnedOperation(body, init_sum_op);
    mlir.oraBlockAppendOwnedOperation(body, init_ctrl_op);
    mlir.oraBlockAppendOwnedOperation(body, sum_delta_op);
    mlir.oraBlockAppendOwnedOperation(body, ctrl_delta_op);

    const bound = mlir.oraBlockGetArgument(body, 0);
    const init_vals = [_]mlir.MlirValue{
        mlir.oraOperationGetResult(init_sum_op, 0),
        mlir.oraOperationGetResult(init_ctrl_op, 0),
    };
    const result_types = [_]mlir.MlirType{ i256_ty, i256_ty };
    const while_op = mlir.oraScfWhileOpCreate(mlir_ctx, loc, &init_vals, init_vals.len, &result_types, result_types.len);
    const before_block = mlir.oraScfWhileOpGetBeforeBlock(while_op);
    const after_block = mlir.oraScfWhileOpGetAfterBlock(while_op);
    _ = mlir.mlirBlockAddArgument(before_block, i256_ty, loc);
    _ = mlir.mlirBlockAddArgument(before_block, i256_ty, loc);
    _ = mlir.mlirBlockAddArgument(after_block, i256_ty, loc);
    _ = mlir.mlirBlockAddArgument(after_block, i256_ty, loc);

    const before_ctrl = mlir.oraBlockGetArgument(before_block, 1);
    const cmp_op = mlir.oraArithCmpIOpCreate(mlir_ctx, loc, 2, before_ctrl, bound); // slt
    mlir.oraBlockAppendOwnedOperation(before_block, cmp_op);
    mlir.oraBlockAppendOwnedOperation(before_block, mlir.oraScfConditionOpCreate(
        mlir_ctx,
        loc,
        mlir.oraOperationGetResult(cmp_op, 0),
        &[_]mlir.MlirValue{
            mlir.oraBlockGetArgument(before_block, 0),
            before_ctrl,
        },
        2,
    ));

    const after_sum = mlir.oraBlockGetArgument(after_block, 0);
    const after_ctrl = mlir.oraBlockGetArgument(after_block, 1);
    const next_sum = mlir.oraArithAddIOpCreate(mlir_ctx, loc, after_sum, mlir.oraOperationGetResult(sum_delta_op, 0));
    const next_ctrl = mlir.oraArithAddIOpCreate(mlir_ctx, loc, after_ctrl, mlir.oraOperationGetResult(ctrl_delta_op, 0));
    mlir.oraBlockAppendOwnedOperation(after_block, next_sum);
    mlir.oraBlockAppendOwnedOperation(after_block, next_ctrl);
    mlir.oraBlockAppendOwnedOperation(after_block, mlir.oraScfYieldOpCreate(
        mlir_ctx,
        loc,
        &[_]mlir.MlirValue{
            mlir.oraOperationGetResult(next_sum, 0),
            mlir.oraOperationGetResult(next_ctrl, 0),
        },
        2,
    ));
    mlir.oraBlockAppendOwnedOperation(body, while_op);
    mlir.oraBlockAppendOwnedOperation(body, mlir.oraReturnOpCreate(
        mlir_ctx,
        loc,
        &[_]mlir.MlirValue{mlir.oraOperationGetResult(while_op, 0)},
        1,
    ));

    try encoder.registerFunctionOperation(helper);

    const caller_bound = mlir.oraVariablePlaceholderOpCreate(mlir_ctx, loc, stringRef("callerSignedControlSecondBound"), i256_ty);
    const call = mlir.oraFuncCallOpCreate(
        mlir_ctx,
        loc,
        stringRef("signedWhileControlSecondReturn"),
        &[_]mlir.MlirValue{mlir.oraOperationGetResult(caller_bound, 0)},
        1,
        &[_]mlir.MlirType{i256_ty},
        1,
    );
    const encoded = try encoder.encodeOperation(call);
    try testing.expect(!encoder.isDegraded());

    const init_sum_ast = try encoder.encodeOperation(init_sum_op);
    const init_ctrl_ast = try encoder.encodeOperation(init_ctrl_op);
    const sum_delta_ast = try encoder.encodeOperation(sum_delta_op);
    const ctrl_delta_ast = try encoder.encodeOperation(ctrl_delta_op);
    const bound_ast = try encoder.encodeOperation(caller_bound);
    const zero = try encoder.encodeIntegerConstant(0, 256);
    const one = try encoder.encodeIntegerConstant(1, 256);
    const bound_le_init = z3.Z3_mk_bvsle(z3_ctx.ctx, bound_ast, init_ctrl_ast);
    const distance = z3.Z3_mk_ite(
        z3_ctx.ctx,
        bound_le_init,
        zero,
        z3.Z3_mk_bv_sub(z3_ctx.ctx, bound_ast, init_ctrl_ast),
    );
    const distance_is_zero = z3.Z3_mk_eq(z3_ctx.ctx, distance, zero);
    const step_count = z3.Z3_mk_ite(
        z3_ctx.ctx,
        distance_is_zero,
        zero,
        z3.Z3_mk_bv_add(
            z3_ctx.ctx,
            z3.Z3_mk_bv_udiv(z3_ctx.ctx, z3.Z3_mk_bv_sub(z3_ctx.ctx, distance, one), ctrl_delta_ast),
            one,
        ),
    );
    const total_delta = try encoder.encodeArithmeticOp(.Mul, step_count, sum_delta_ast);
    const expected = try encoder.encodeArithmeticOp(.Add, init_sum_ast, total_delta);

    var solver = try Solver.init(&z3_ctx, testing.allocator);
    defer solver.deinit();
    solver.assert(z3.Z3_mk_not(z3_ctx.ctx, z3.Z3_mk_eq(z3_ctx.ctx, encoded, expected)));
    try testing.expectEqual(@as(z3.Z3_lbool, z3.Z3_L_FALSE), solver.check());
}

test "quantified operation encodes to z3 quantifier" {
    var z3_ctx = try Context.init(testing.allocator);
    defer z3_ctx.deinit();

    var encoder = Encoder.init(&z3_ctx, testing.allocator);
    defer encoder.deinit();

    const mlir_ctx = mlir.oraContextCreate();
    defer mlir.oraContextDestroy(mlir_ctx);
    loadAllDialects(mlir_ctx);
    _ = mlir.oraDialectRegister(mlir_ctx);

    const loc = mlir.oraLocationUnknownGet(mlir_ctx);
    const i1_ty = mlir.oraIntegerTypeCreate(mlir_ctx, 1);
    const body_attr = mlir.oraIntegerAttrCreateI64FromType(i1_ty, 1);
    const body_op = mlir.oraArithConstantOpCreate(mlir_ctx, loc, i1_ty, body_attr);
    const body = mlir.oraOperationGetResult(body_op, 0);

    const qop = mlir.oraQuantifiedOpCreate(
        mlir_ctx,
        loc,
        stringRef("forall"),
        stringRef("i"),
        stringRef("u256"),
        mlir.MlirValue{ .ptr = null },
        false,
        body,
        i1_ty,
    );
    const qast = try encoder.encodeOperation(qop);
    const qstr = std.mem.span(z3.Z3_ast_to_string(z3_ctx.ctx, qast));
    try testing.expect(std.mem.indexOf(u8, qstr, "forall") != null);
}

test "quantified operation with placeholder keeps bound variable name" {
    var z3_ctx = try Context.init(testing.allocator);
    defer z3_ctx.deinit();

    var encoder = Encoder.init(&z3_ctx, testing.allocator);
    defer encoder.deinit();

    const mlir_ctx = mlir.oraContextCreate();
    defer mlir.oraContextDestroy(mlir_ctx);
    loadAllDialects(mlir_ctx);
    _ = mlir.oraDialectRegister(mlir_ctx);

    const loc = mlir.oraLocationUnknownGet(mlir_ctx);
    const i256_ty = mlir.oraIntegerTypeCreate(mlir_ctx, 256);
    const i1_ty = mlir.oraIntegerTypeCreate(mlir_ctx, 1);

    const placeholder_op = mlir.oraVariablePlaceholderOpCreate(mlir_ctx, loc, stringRef("i"), i256_ty);
    const placeholder = mlir.oraOperationGetResult(placeholder_op, 0);

    const zero_attr = mlir.oraIntegerAttrCreateI64FromType(i256_ty, 0);
    const zero_op = mlir.oraArithConstantOpCreate(mlir_ctx, loc, i256_ty, zero_attr);
    const zero = mlir.oraOperationGetResult(zero_op, 0);

    const cmp_op = mlir.oraArithCmpIOpCreate(mlir_ctx, loc, 6, placeholder, zero); // ult
    const body = mlir.oraOperationGetResult(cmp_op, 0);

    const qop = mlir.oraQuantifiedOpCreate(
        mlir_ctx,
        loc,
        stringRef("forall"),
        stringRef("i"),
        stringRef("u256"),
        mlir.MlirValue{ .ptr = null },
        false,
        body,
        i1_ty,
    );
    const qast = try encoder.encodeOperation(qop);
    const qstr = std.mem.span(z3.Z3_ast_to_string(z3_ctx.ctx, qast));
    try testing.expect(std.mem.indexOf(u8, qstr, "forall") != null);
    try testing.expect(std.mem.indexOf(u8, qstr, "i") != null);
}

test "quantified forall and exists have expected solver semantics" {
    var z3_ctx = try Context.init(testing.allocator);
    defer z3_ctx.deinit();

    var encoder = Encoder.init(&z3_ctx, testing.allocator);
    defer encoder.deinit();

    const mlir_ctx = mlir.oraContextCreate();
    defer mlir.oraContextDestroy(mlir_ctx);
    loadAllDialects(mlir_ctx);
    _ = mlir.oraDialectRegister(mlir_ctx);

    const loc = mlir.oraLocationUnknownGet(mlir_ctx);
    const i8_ty = mlir.oraIntegerTypeCreate(mlir_ctx, 8);
    const i1_ty = mlir.oraIntegerTypeCreate(mlir_ctx, 1);

    // forall i:u8. i == 7  ==> UNSAT
    const ph_forall = mlir.oraVariablePlaceholderOpCreate(mlir_ctx, loc, stringRef("i"), i8_ty);
    const ph_forall_v = mlir.oraOperationGetResult(ph_forall, 0);
    const seven_attr = mlir.oraIntegerAttrCreateI64FromType(i8_ty, 7);
    const seven_op = mlir.oraArithConstantOpCreate(mlir_ctx, loc, i8_ty, seven_attr);
    const seven_v = mlir.oraOperationGetResult(seven_op, 0);
    const eq_forall = mlir.oraArithCmpIOpCreate(mlir_ctx, loc, 0, ph_forall_v, seven_v); // eq
    const eq_forall_v = mlir.oraOperationGetResult(eq_forall, 0);
    const forall_op = mlir.oraQuantifiedOpCreate(
        mlir_ctx,
        loc,
        stringRef("forall"),
        stringRef("i"),
        stringRef("u8"),
        mlir.MlirValue{ .ptr = null },
        false,
        eq_forall_v,
        i1_ty,
    );
    const forall_ast = try encoder.encodeOperation(forall_op);

    var solver = try Solver.init(&z3_ctx, testing.allocator);
    defer solver.deinit();
    solver.assert(forall_ast);
    try testing.expectEqual(@as(z3.Z3_lbool, z3.Z3_L_FALSE), solver.check());
    solver.reset();

    // exists i:u8. i == 7  ==> SAT
    const ph_exists = mlir.oraVariablePlaceholderOpCreate(mlir_ctx, loc, stringRef("i"), i8_ty);
    const ph_exists_v = mlir.oraOperationGetResult(ph_exists, 0);
    const eq_exists = mlir.oraArithCmpIOpCreate(mlir_ctx, loc, 0, ph_exists_v, seven_v); // eq
    const eq_exists_v = mlir.oraOperationGetResult(eq_exists, 0);
    const exists_op = mlir.oraQuantifiedOpCreate(
        mlir_ctx,
        loc,
        stringRef("exists"),
        stringRef("i"),
        stringRef("u8"),
        mlir.MlirValue{ .ptr = null },
        false,
        eq_exists_v,
        i1_ty,
    );
    const exists_ast = try encoder.encodeOperation(exists_op);

    solver.assert(exists_ast);
    try testing.expectEqual(@as(z3.Z3_lbool, z3.Z3_L_TRUE), solver.check());
}
