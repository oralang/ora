const prelude = @import("encoder_test_prelude.zig");
const std = prelude.std;
const testing = prelude.testing;
const z3 = prelude.z3;
const mlir = prelude.mlir;
const Context = prelude.Context;
const Encoder = prelude.Encoder;
const Solver = prelude.Solver;
const stringRef = prelude.stringRef;
const namedAttr = prelude.namedAttr;
const loadAllDialects = prelude.loadAllDialects;
const expectAstEquivalent = prelude.expectAstEquivalent;
const expectSingleSelectTrigger = prelude.expectSingleSelectTrigger;
const expectNoQuantifiedConstraints = prelude.expectNoQuantifiedConstraints;
const astContainsQuantifier = prelude.astContainsQuantifier;

const production_z3_sources = [_][]const u8{
    @embedFile("c.zig"),
    @embedFile("context.zig"),
    @embedFile("encoder.zig"),
    @embedFile("errors.zig"),
    @embedFile("mlir_helpers.zig"),
    @embedFile("mod.zig"),
    @embedFile("solver.zig"),
    @embedFile("verification.zig"),
};

test "tag-only enum constants encode as distinct datatype constructors" {
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
    const enum_decl = mlir.oraEnumDeclOpCreate(mlir_ctx, loc, stringRef("Status"), i256_ty);
    mlir.oraOperationSetAttributeByName(
        enum_decl,
        stringRef("sym_name"),
        mlir.oraStringAttrCreate(mlir_ctx, stringRef("Status")),
    );
    const variant_names = [_]mlir.MlirAttribute{
        mlir.oraStringAttrCreate(mlir_ctx, stringRef("Active")),
        mlir.oraStringAttrCreate(mlir_ctx, stringRef("Paused")),
    };
    mlir.oraOperationSetAttributeByName(
        enum_decl,
        stringRef("ora.variant_names"),
        mlir.oraArrayAttrCreate(mlir_ctx, variant_names.len, &variant_names),
    );
    try encoder.registerEnumDeclOperation(enum_decl);

    const enum_ty = mlir.mlirTypeParseGet(mlir_ctx, stringRef("!ora.enum<\"Status\", i256>"));
    try testing.expect(!mlir.oraTypeIsNull(enum_ty));
    const enum_sort = try encoder.encodeMLIRType(enum_ty);
    try testing.expectEqual(@as(u32, z3.Z3_DATATYPE_SORT), @as(u32, @intCast(z3.Z3_get_sort_kind(z3_ctx.ctx, enum_sort))));

    const active_op = mlir.oraEnumConstantOpCreate(mlir_ctx, loc, stringRef("Status"), stringRef("Active"), enum_ty);
    const paused_op = mlir.oraEnumConstantOpCreate(mlir_ctx, loc, stringRef("Status"), stringRef("Paused"), enum_ty);
    const active = try encoder.encodeOperation(active_op);
    const paused = try encoder.encodeOperation(paused_op);

    var solver = try Solver.init(&z3_ctx, testing.allocator);
    defer solver.deinit();
    solver.assert(z3.Z3_mk_eq(z3_ctx.ctx, active, paused));
    try testing.expectEqual(@as(z3.Z3_lbool, z3.Z3_L_FALSE), solver.check());
    try testing.expect(!encoder.isDegraded());
}

test "tag-only enum constructors are scoped by enum sort" {
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
    const status_decl = mlir.oraEnumDeclOpCreate(mlir_ctx, loc, stringRef("Status"), i256_ty);
    const mode_decl = mlir.oraEnumDeclOpCreate(mlir_ctx, loc, stringRef("Mode"), i256_ty);
    const status_names = [_]mlir.MlirAttribute{
        mlir.oraStringAttrCreate(mlir_ctx, stringRef("Active")),
    };
    const mode_names = [_]mlir.MlirAttribute{
        mlir.oraStringAttrCreate(mlir_ctx, stringRef("Active")),
    };
    mlir.oraOperationSetAttributeByName(
        status_decl,
        stringRef("ora.variant_names"),
        mlir.oraArrayAttrCreate(mlir_ctx, status_names.len, &status_names),
    );
    mlir.oraOperationSetAttributeByName(
        mode_decl,
        stringRef("ora.variant_names"),
        mlir.oraArrayAttrCreate(mlir_ctx, mode_names.len, &mode_names),
    );
    try encoder.registerEnumDeclOperation(status_decl);
    try encoder.registerEnumDeclOperation(mode_decl);

    const status_ty = mlir.mlirTypeParseGet(mlir_ctx, stringRef("!ora.enum<\"Status\", i256>"));
    const mode_ty = mlir.mlirTypeParseGet(mlir_ctx, stringRef("!ora.enum<\"Mode\", i256>"));
    const status_active_op = mlir.oraEnumConstantOpCreate(mlir_ctx, loc, stringRef("Status"), stringRef("Active"), status_ty);
    const mode_active_op = mlir.oraEnumConstantOpCreate(mlir_ctx, loc, stringRef("Mode"), stringRef("Active"), mode_ty);
    const status_active = try encoder.encodeOperation(status_active_op);
    const mode_active = try encoder.encodeOperation(mode_active_op);

    const status_sort = z3.Z3_get_sort(z3_ctx.ctx, status_active);
    const mode_sort = z3.Z3_get_sort(z3_ctx.ctx, mode_active);
    try testing.expect(status_sort != mode_sort);
    try testing.expectEqual(@as(u32, z3.Z3_DATATYPE_SORT), @as(u32, @intCast(z3.Z3_get_sort_kind(z3_ctx.ctx, status_sort))));
    try testing.expectEqual(@as(u32, z3.Z3_DATATYPE_SORT), @as(u32, @intCast(z3.Z3_get_sort_kind(z3_ctx.ctx, mode_sort))));
    try testing.expect(!encoder.isDegraded());
}

test "ADT construct tag and payload encode as datatype constructors" {
    var z3_ctx = try Context.init(testing.allocator);
    defer z3_ctx.deinit();

    var encoder = Encoder.init(&z3_ctx, testing.allocator);
    defer encoder.deinit();

    const mlir_ctx = mlir.oraContextCreate();
    defer mlir.oraContextDestroy(mlir_ctx);
    loadAllDialects(mlir_ctx);
    _ = mlir.oraDialectRegister(mlir_ctx);

    const text =
        \\module {
        \\  func.func @f(%payload: !ora.int<256, false>) -> !ora.int<256, false> {
        \\    %event = ora.adt.construct "Value"(%payload) : (!ora.int<256, false>) -> !ora.adt<"Event", ("Empty", none), ("Value", !ora.int<256, false>)>
        \\    %tag = ora.adt.tag %event : !ora.adt<"Event", ("Empty", none), ("Value", !ora.int<256, false>)> -> !ora.int<256, false>
        \\    %extracted = ora.adt.payload %event, "Value" : !ora.adt<"Event", ("Empty", none), ("Value", !ora.int<256, false>)> -> !ora.int<256, false>
        \\    func.return %extracted : !ora.int<256, false>
        \\  }
        \\}
    ;
    const module = mlir.oraModuleCreateParse(mlir_ctx, mlir.oraStringRefCreate(text.ptr, text.len));
    defer mlir.oraModuleDestroy(module);
    try testing.expect(!mlir.oraModuleIsNull(module));

    const module_body = mlir.oraModuleGetBody(module);
    const func_op = mlir.oraBlockGetFirstOperation(module_body);
    const func_region = mlir.oraOperationGetRegion(func_op, 0);
    const func_block = mlir.oraRegionGetFirstBlock(func_region);
    const payload_arg = mlir.oraBlockGetArgument(func_block, 0);

    const construct_op = mlir.oraBlockGetFirstOperation(func_block);
    const tag_op = mlir.oraOperationGetNextInBlock(construct_op);
    const payload_op = mlir.oraOperationGetNextInBlock(tag_op);

    const payload_arg_ast = try encoder.encodeValue(payload_arg);
    const extracted = try encoder.encodeOperation(payload_op);
    const tag = try encoder.encodeOperation(tag_op);

    var payload_solver = try Solver.init(&z3_ctx, testing.allocator);
    defer payload_solver.deinit();
    payload_solver.assert(z3.Z3_mk_not(z3_ctx.ctx, z3.Z3_mk_eq(z3_ctx.ctx, extracted, payload_arg_ast)));
    try testing.expectEqual(@as(z3.Z3_lbool, z3.Z3_L_FALSE), payload_solver.check());

    const tag_sort = z3.Z3_get_sort(z3_ctx.ctx, tag);
    const tag_width = z3.Z3_get_bv_sort_size(z3_ctx.ctx, tag_sort);
    const one = try encoder.encodeIntegerConstant(1, tag_width);
    var tag_solver = try Solver.init(&z3_ctx, testing.allocator);
    defer tag_solver.deinit();
    tag_solver.assert(z3.Z3_mk_not(z3_ctx.ctx, z3.Z3_mk_eq(z3_ctx.ctx, tag, one)));
    try testing.expectEqual(@as(z3.Z3_lbool, z3.Z3_L_FALSE), tag_solver.check());
    try testing.expect(!encoder.isDegraded());
}

test "error_union uses distinct constructors for declared error payload types" {
    var z3_ctx = try Context.init(testing.allocator);
    defer z3_ctx.deinit();

    var encoder = Encoder.init(&z3_ctx, testing.allocator);
    defer encoder.deinit();

    const mlir_ctx = mlir.oraContextCreate();
    defer mlir.oraContextDestroy(mlir_ctx);
    loadAllDialects(mlir_ctx);
    _ = mlir.oraDialectRegister(mlir_ctx);

    const loc = mlir.oraLocationUnknownGet(mlir_ctx);
    const ok_ty = mlir.oraIntegerTypeCreate(mlir_ctx, 256);
    const err_a_ty = mlir.oraIntegerTypeCreate(mlir_ctx, 32);
    const err_b_ty = mlir.oraIntegerTypeCreate(mlir_ctx, 64);
    var error_types = [_]mlir.MlirType{ err_a_ty, err_b_ty };
    const eu_ty = mlir.oraErrorUnionTypeGetWithErrors(mlir_ctx, ok_ty, error_types.len, &error_types);

    const err_a_attr = mlir.oraIntegerAttrCreateI64FromType(err_a_ty, 7);
    const err_a_value_op = mlir.oraArithConstantOpCreate(mlir_ctx, loc, err_a_ty, err_a_attr);
    const err_a_op = mlir.oraErrorErrOpCreate(mlir_ctx, loc, mlir.oraOperationGetResult(err_a_value_op, 0), eu_ty);

    const err_b_attr = mlir.oraIntegerAttrCreateI64FromType(err_b_ty, 9);
    const err_b_value_op = mlir.oraArithConstantOpCreate(mlir_ctx, loc, err_b_ty, err_b_attr);
    const err_b_op = mlir.oraErrorErrOpCreate(mlir_ctx, loc, mlir.oraOperationGetResult(err_b_value_op, 0), eu_ty);

    const err_a = try encoder.encodeOperation(err_a_op);
    const err_b = try encoder.encodeOperation(err_b_op);
    const err_a_decl = z3.Z3_get_app_decl(z3_ctx.ctx, z3.Z3_to_app(z3_ctx.ctx, err_a));
    const err_b_decl = z3.Z3_get_app_decl(z3_ctx.ctx, z3.Z3_to_app(z3_ctx.ctx, err_b));
    try testing.expect(err_a_decl != err_b_decl);
    try testing.expectEqual(@as(c_uint, 1), z3.Z3_get_app_num_args(z3_ctx.ctx, z3.Z3_to_app(z3_ctx.ctx, err_a)));
    try testing.expectEqual(@as(c_uint, 1), z3.Z3_get_app_num_args(z3_ctx.ctx, z3.Z3_to_app(z3_ctx.ctx, err_b)));
    try testing.expectEqual(z3.Z3_get_sort(z3_ctx.ctx, err_a), z3.Z3_get_sort(z3_ctx.ctx, err_b));
    var solver = try Solver.init(&z3_ctx, testing.allocator);
    defer solver.deinit();
    solver.assert(z3.Z3_mk_eq(z3_ctx.ctx, err_a, err_b));
    try testing.expectEqual(@as(z3.Z3_lbool, z3.Z3_L_FALSE), solver.check());
    try testing.expect(!encoder.isDegraded());
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

test "constraints are path-guarded by default while global constraints remain raw" {
    var z3_ctx = try Context.init(testing.allocator);
    defer z3_ctx.deinit();

    var encoder = Encoder.init(&z3_ctx, testing.allocator);
    defer encoder.deinit();

    const bool_sort = z3.Z3_mk_bool_sort(z3_ctx.ctx);
    const path = try encoder.mkVariable("path", bool_sort);
    const fact = try encoder.mkVariable("fact", bool_sort);

    try encoder.return_path_assumptions.append(testing.allocator, path);
    encoder.addConstraintForTesting(fact);

    const guarded_constraints = try encoder.takeConstraints(testing.allocator);
    defer if (guarded_constraints.len > 0) testing.allocator.free(guarded_constraints);
    try testing.expectEqual(@as(usize, 1), guarded_constraints.len);

    var solver = try Solver.init(&z3_ctx, testing.allocator);
    defer solver.deinit();
    solver.assert(guarded_constraints[0]);
    solver.assert(path);
    solver.assert(z3.Z3_mk_not(z3_ctx.ctx, fact));
    try testing.expectEqual(@as(z3.Z3_lbool, z3.Z3_L_FALSE), solver.check());

    solver.reset();
    solver.assert(guarded_constraints[0]);
    solver.assert(z3.Z3_mk_not(z3_ctx.ctx, path));
    solver.assert(z3.Z3_mk_not(z3_ctx.ctx, fact));
    try testing.expectEqual(@as(z3.Z3_lbool, z3.Z3_L_TRUE), solver.check());

    encoder.addGlobalConstraintForTesting(fact);
    const global_constraints = try encoder.takeConstraints(testing.allocator);
    defer if (global_constraints.len > 0) testing.allocator.free(global_constraints);
    try testing.expectEqual(@as(usize, 1), global_constraints.len);

    solver.reset();
    solver.assert(global_constraints[0]);
    solver.assert(z3.Z3_mk_not(z3_ctx.ctx, path));
    solver.assert(z3.Z3_mk_not(z3_ctx.ctx, fact));
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

test "caller struct_field_update recovers untouched fields from known callee scoped struct declaration metadata" {
    var z3_ctx = try Context.init(testing.allocator);
    defer z3_ctx.deinit();

    var encoder = Encoder.init(&z3_ctx, testing.allocator);
    defer encoder.deinit();

    const mlir_ctx = mlir.oraContextCreate();
    defer mlir.oraContextDestroy(mlir_ctx);
    loadAllDialects(mlir_ctx);
    _ = mlir.oraDialectRegister(mlir_ctx);

    const loc = mlir.oraLocationUnknownGet(mlir_ctx);
    const module = mlir.oraModuleCreateEmpty(loc);
    defer mlir.oraModuleDestroy(module);
    const module_body = mlir.oraModuleGetBody(module);

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
    mlir.oraBlockAppendOwnedOperation(module_body, struct_decl);

    const helper_attrs = [_]mlir.MlirNamedAttribute{
        namedAttr(mlir_ctx, "sym_name", mlir.oraStringAttrCreate(mlir_ctx, stringRef("buildPairViaScopedDecl"))),
    };
    const helper = mlir.oraFuncFuncOpCreate(mlir_ctx, loc, &helper_attrs, helper_attrs.len, &[_]mlir.MlirType{}, &[_]mlir.MlirLocation{}, 0);
    mlir.oraBlockAppendOwnedOperation(module_body, helper);
    const helper_body = mlir.oraFuncOpGetBodyBlock(helper);

    const one_op = mlir.oraArithConstantOpCreate(
        mlir_ctx,
        loc,
        i256_ty,
        mlir.oraIntegerAttrCreateI64FromType(i256_ty, 1),
    );
    const two_op = mlir.oraArithConstantOpCreate(
        mlir_ctx,
        loc,
        i256_ty,
        mlir.oraIntegerAttrCreateI64FromType(i256_ty, 2),
    );
    const seven_op = mlir.oraArithConstantOpCreate(
        mlir_ctx,
        loc,
        i256_ty,
        mlir.oraIntegerAttrCreateI64FromType(i256_ty, 7),
    );
    mlir.oraBlockAppendOwnedOperation(helper_body, one_op);
    mlir.oraBlockAppendOwnedOperation(helper_body, two_op);
    mlir.oraBlockAppendOwnedOperation(helper_body, seven_op);

    const one = mlir.oraOperationGetResult(one_op, 0);
    const two = mlir.oraOperationGetResult(two_op, 0);
    const seven = mlir.oraOperationGetResult(seven_op, 0);
    const init_op = mlir.oraStructInstantiateOpCreate(mlir_ctx, loc, stringRef("Pair__u256"), &[_]mlir.MlirValue{ one, two }, 2, struct_ty);
    mlir.oraBlockAppendOwnedOperation(helper_body, init_op);
    mlir.oraBlockAppendOwnedOperation(helper_body, mlir.oraReturnOpCreate(
        mlir_ctx,
        loc,
        &[_]mlir.MlirValue{mlir.oraOperationGetResult(init_op, 0)},
        1,
    ));
    try encoder.registerFunctionOperation(helper);

    const call = mlir.oraFuncCallOpCreate(mlir_ctx, loc, stringRef("buildPairViaScopedDecl"), &[_]mlir.MlirValue{}, 0, &[_]mlir.MlirType{struct_ty}, 1);
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

test "caller struct_field_update from known callee scf.for carried block arg metadata preserves untouched fields exactly" {
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
    const struct_ty = mlir.oraStructTypeGet(mlir_ctx, stringRef("Pair__u256"));

    const helper_attrs = [_]mlir.MlirNamedAttribute{
        namedAttr(mlir_ctx, "sym_name", mlir.oraStringAttrCreate(mlir_ctx, stringRef("buildPairViaForCarry"))),
    };
    const helper = mlir.oraFuncFuncOpCreate(mlir_ctx, loc, &helper_attrs, helper_attrs.len, &[_]mlir.MlirType{}, &[_]mlir.MlirLocation{}, 0);
    const helper_body = mlir.oraFuncOpGetBodyBlock(helper);

    const zero = mlir.oraOperationGetResult(mlir.oraArithConstantOpCreate(
        mlir_ctx,
        loc,
        index_ty,
        mlir.oraIntegerAttrCreateI64FromType(index_ty, 0),
    ), 0);
    const one_idx = mlir.oraOperationGetResult(mlir.oraArithConstantOpCreate(
        mlir_ctx,
        loc,
        index_ty,
        mlir.oraIntegerAttrCreateI64FromType(index_ty, 1),
    ), 0);
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
    const init_op = mlir.oraStructInitOpCreate(mlir_ctx, loc, &[_]mlir.MlirValue{ one, two }, 2, struct_ty);
    mlir.oraOperationSetAttributeByName(init_op, stringRef("ora.field_names"), mlir.oraArrayAttrCreate(mlir_ctx, field_name_attrs.len, &field_name_attrs));
    const pair = mlir.oraOperationGetResult(init_op, 0);

    const loop = mlir.oraScfForOpCreate(
        mlir_ctx,
        loc,
        zero,
        one_idx,
        one_idx,
        &[_]mlir.MlirValue{pair},
        1,
        false,
    );
    const body = mlir.oraScfForOpGetBodyBlock(loop);
    const carried_pair = mlir.oraBlockGetArgument(body, 1);
    const update_op = mlir.oraStructFieldUpdateOpCreate(mlir_ctx, loc, carried_pair, stringRef("left"), seven);
    mlir.oraBlockAppendOwnedOperation(body, update_op);
    mlir.oraBlockAppendOwnedOperation(body, mlir.oraScfYieldOpCreate(
        mlir_ctx,
        loc,
        &[_]mlir.MlirValue{mlir.oraOperationGetResult(update_op, 0)},
        1,
    ));

    mlir.oraBlockAppendOwnedOperation(helper_body, init_op);
    mlir.oraBlockAppendOwnedOperation(helper_body, loop);
    mlir.oraBlockAppendOwnedOperation(helper_body, mlir.oraReturnOpCreate(
        mlir_ctx,
        loc,
        &[_]mlir.MlirValue{mlir.oraOperationGetResult(loop, 0)},
        1,
    ));
    try encoder.registerFunctionOperation(helper);

    const call = mlir.oraFuncCallOpCreate(mlir_ctx, loc, stringRef("buildPairViaForCarry"), &[_]mlir.MlirValue{}, 0, &[_]mlir.MlirType{struct_ty}, 1);
    const updated = mlir.oraStructFieldUpdateOpCreate(mlir_ctx, loc, mlir.oraOperationGetResult(call, 0), stringRef("left"), seven);
    const updated_pair = mlir.oraOperationGetResult(updated, 0);
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

test "caller struct_field_update from known callee scf.while carried block arg metadata preserves untouched fields exactly" {
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
        namedAttr(mlir_ctx, "sym_name", mlir.oraStringAttrCreate(mlir_ctx, stringRef("buildPairViaWhileCarry"))),
    };
    const helper = mlir.oraFuncFuncOpCreate(mlir_ctx, loc, &helper_attrs, helper_attrs.len, &[_]mlir.MlirType{}, &[_]mlir.MlirLocation{}, 0);
    const helper_body = mlir.oraFuncOpGetBodyBlock(helper);

    const true_val = mlir.oraOperationGetResult(mlir.oraArithConstantOpCreate(
        mlir_ctx,
        loc,
        i1_ty,
        mlir.oraIntegerAttrCreateI64FromType(i1_ty, 1),
    ), 0);
    const false_val = mlir.oraOperationGetResult(mlir.oraArithConstantOpCreate(
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
    const init_op = mlir.oraStructInitOpCreate(mlir_ctx, loc, &[_]mlir.MlirValue{ one, two }, 2, struct_ty);
    mlir.oraOperationSetAttributeByName(init_op, stringRef("ora.field_names"), mlir.oraArrayAttrCreate(mlir_ctx, field_name_attrs.len, &field_name_attrs));
    const pair = mlir.oraOperationGetResult(init_op, 0);

    const while_op = mlir.oraScfWhileOpCreate(
        mlir_ctx,
        loc,
        &[_]mlir.MlirValue{ pair, true_val },
        2,
        &[_]mlir.MlirType{ struct_ty, i1_ty },
        2,
    );
    const before_block = mlir.oraScfWhileOpGetBeforeBlock(while_op);
    const after_block = mlir.oraScfWhileOpGetAfterBlock(while_op);
    _ = mlir.mlirBlockAddArgument(before_block, struct_ty, loc);
    _ = mlir.mlirBlockAddArgument(before_block, i1_ty, loc);
    _ = mlir.mlirBlockAddArgument(after_block, struct_ty, loc);
    _ = mlir.mlirBlockAddArgument(after_block, i1_ty, loc);
    const before_pair = mlir.oraBlockGetArgument(before_block, 0);
    mlir.oraBlockAppendOwnedOperation(before_block, mlir.oraScfConditionOpCreate(
        mlir_ctx,
        loc,
        mlir.oraBlockGetArgument(before_block, 1),
        &[_]mlir.MlirValue{ before_pair, mlir.oraBlockGetArgument(before_block, 1) },
        2,
    ));

    const update_op = mlir.oraStructFieldUpdateOpCreate(mlir_ctx, loc, mlir.oraBlockGetArgument(after_block, 0), stringRef("left"), seven);
    mlir.oraBlockAppendOwnedOperation(after_block, update_op);
    mlir.oraBlockAppendOwnedOperation(after_block, mlir.oraScfYieldOpCreate(
        mlir_ctx,
        loc,
        &[_]mlir.MlirValue{
            mlir.oraOperationGetResult(update_op, 0),
            false_val,
        },
        2,
    ));

    mlir.oraBlockAppendOwnedOperation(helper_body, init_op);
    mlir.oraBlockAppendOwnedOperation(helper_body, while_op);
    mlir.oraBlockAppendOwnedOperation(helper_body, mlir.oraReturnOpCreate(
        mlir_ctx,
        loc,
        &[_]mlir.MlirValue{mlir.oraOperationGetResult(while_op, 0)},
        1,
    ));
    try encoder.registerFunctionOperation(helper);

    const call = mlir.oraFuncCallOpCreate(mlir_ctx, loc, stringRef("buildPairViaWhileCarry"), &[_]mlir.MlirValue{}, 0, &[_]mlir.MlirType{struct_ty}, 1);
    const updated = mlir.oraStructFieldUpdateOpCreate(mlir_ctx, loc, mlir.oraOperationGetResult(call, 0), stringRef("left"), seven);
    const updated_pair = mlir.oraOperationGetResult(updated, 0);
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
    const nine = mlir.oraOperationGetResult(mlir.oraArithConstantOpCreate(
        mlir_ctx,
        loc,
        i256_ty,
        mlir.oraIntegerAttrCreateI64FromType(i256_ty, 9),
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

test "coerceBoolean degrades on non-bool non-bv sort" {
    var z3_ctx = try Context.init(testing.allocator);
    defer z3_ctx.deinit();

    var encoder = Encoder.init(&z3_ctx, testing.allocator);
    defer encoder.deinit();

    const index_sort = z3.Z3_mk_bv_sort(z3_ctx.ctx, 256);
    const value_sort = z3.Z3_mk_bv_sort(z3_ctx.ctx, 256);
    const array_sort = z3.Z3_mk_array_sort(z3_ctx.ctx, index_sort, value_sort);
    const array_ast = try encoder.mkVariable("array_bool_coercion", array_sort);

    _ = encoder.coerceBoolean(array_ast);

    try testing.expect(encoder.isDegraded());
    try testing.expectEqualStrings("coerceToBool on non-bool non-bv sort", encoder.degradationReason().?);
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
    const shape: [2]i64 = .{ 4, 5 };
    const tensor_ty = mlir.oraRankedTensorTypeCreate(mlir_ctx, 2, &shape, i256_ty, null_attr);

    const empty_attrs = [_]mlir.MlirNamedAttribute{};
    const param_types = [_]mlir.MlirType{tensor_ty};
    const param_locs = [_]mlir.MlirLocation{loc};
    const func = mlir.oraFuncFuncOpCreate(mlir_ctx, loc, &empty_attrs, empty_attrs.len, &param_types, &param_locs, param_types.len);
    const body = mlir.oraFuncOpGetBodyBlock(func);
    const tensor_arg = mlir.oraBlockGetArgument(body, 0);

    const c0_attr = mlir.oraIntegerAttrCreateI64FromType(index_ty, 0);
    const c0_op = mlir.oraArithConstantOpCreate(mlir_ctx, loc, index_ty, c0_attr);
    const c0 = mlir.oraOperationGetResult(c0_op, 0);
    const c1_attr = mlir.oraIntegerAttrCreateI64FromType(index_ty, 1);
    const c1_op = mlir.oraArithConstantOpCreate(mlir_ctx, loc, index_ty, c1_attr);
    const c1 = mlir.oraOperationGetResult(c1_op, 0);

    const value_attr = mlir.oraIntegerAttrCreateI64FromType(i256_ty, 42);
    const value_op = mlir.oraArithConstantOpCreate(mlir_ctx, loc, i256_ty, value_attr);
    const value = mlir.oraOperationGetResult(value_op, 0);

    const insert_operands = [_]mlir.MlirValue{ value, tensor_arg, c0, c1 };
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

    const extract_indices = [_]mlir.MlirValue{ c0, c1 };
    const extract_op = mlir.oraTensorExtractOpCreate(mlir_ctx, loc, inserted_tensor, &extract_indices, extract_indices.len, i256_ty);
    const extracted = try encoder.encodeOperation(extract_op);
    const expected = try encoder.encodeValue(value);

    var solver = try Solver.init(&z3_ctx, testing.allocator);
    defer solver.deinit();
    solver.assert(z3.Z3_mk_not(z3_ctx.ctx, z3.Z3_mk_eq(z3_ctx.ctx, extracted, expected)));
    try testing.expectEqual(@as(z3.Z3_lbool, z3.Z3_L_FALSE), solver.check());
    try testing.expect(!encoder.isDegraded());
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

test "ora.evm.caller shares symbol and assumes non-zero" {
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

    // Repeated caller reads within one function should resolve to the same symbol.
    var solver_same = try Solver.init(&z3_ctx, testing.allocator);
    defer solver_same.deinit();
    for (constraints) |cst| solver_same.assert(cst);
    const neq = z3.Z3_mk_not(z3_ctx.ctx, z3.Z3_mk_eq(z3_ctx.ctx, caller_a, caller_b));
    solver_same.assert(neq);
    try testing.expectEqual(@as(z3.Z3_lbool, z3.Z3_L_FALSE), solver_same.check());

    // Ora's std.msg.sender() is modeled as NonZeroAddress, so the SMT
    // environment must exclude ZERO_ADDRESS for caller.
    var solver_zero = try Solver.init(&z3_ctx, testing.allocator);
    defer solver_zero.deinit();
    for (constraints) |cst| solver_zero.assert(cst);
    const caller_sort = z3.Z3_get_sort(z3_ctx.ctx, caller_a);
    const zero = z3.Z3_mk_unsigned_int64(z3_ctx.ctx, 0, caller_sort);
    solver_zero.assert(z3.Z3_mk_eq(z3_ctx.ctx, caller_a, zero));
    try testing.expectEqual(@as(z3.Z3_lbool, z3.Z3_L_FALSE), solver_zero.check());
}

test "loop result entry points agree for canonical scf.while and scf.for" {
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
    const index_ty = mlir.oraIndexTypeCreate(mlir_ctx);
    const i256_ty = mlir.oraIntegerTypeCreate(mlir_ctx, 256);

    const false_attr = mlir.oraIntegerAttrCreateI64FromType(i1_ty, 0);
    const i256_seven_attr = mlir.oraIntegerAttrCreateI64FromType(i256_ty, 7);
    const index_zero_attr = mlir.oraIntegerAttrCreateI64FromType(index_ty, 0);
    const index_one_attr = mlir.oraIntegerAttrCreateI64FromType(index_ty, 1);

    const false_op = mlir.oraArithConstantOpCreate(mlir_ctx, loc, i1_ty, false_attr);
    const init_op = mlir.oraArithConstantOpCreate(mlir_ctx, loc, i256_ty, i256_seven_attr);
    const index_zero_op = mlir.oraArithConstantOpCreate(mlir_ctx, loc, index_ty, index_zero_attr);
    const index_one_op = mlir.oraArithConstantOpCreate(mlir_ctx, loc, index_ty, index_one_attr);

    const while_init = mlir.oraOperationGetResult(init_op, 0);
    const while_results = [_]mlir.MlirType{i256_ty};
    const while_op = mlir.oraScfWhileOpCreate(mlir_ctx, loc, &[_]mlir.MlirValue{while_init}, 1, &while_results, while_results.len);
    const before_block = mlir.oraScfWhileOpGetBeforeBlock(while_op);
    mlir.oraBlockAppendOwnedOperation(before_block, mlir.oraScfConditionOpCreate(
        mlir_ctx,
        loc,
        mlir.oraOperationGetResult(false_op, 0),
        &[_]mlir.MlirValue{while_init},
        1,
    ));

    const while_via_operation = try encoder.encodeOperationWithModeForTesting(while_op, .Current);
    const while_via_result = try encoder.encodeOperationResultWithModeForTesting(while_op, 0, .Current);

    const for_op = mlir.oraScfForOpCreate(
        mlir_ctx,
        loc,
        mlir.oraOperationGetResult(index_zero_op, 0),
        mlir.oraOperationGetResult(index_zero_op, 0),
        mlir.oraOperationGetResult(index_one_op, 0),
        &[_]mlir.MlirValue{mlir.oraOperationGetResult(init_op, 0)},
        1,
        false,
    );

    const for_via_operation = try encoder.encodeOperationWithModeForTesting(for_op, .Current);
    const for_via_result = try encoder.encodeOperationResultWithModeForTesting(for_op, 0, .Current);

    try testing.expect(!encoder.isDegraded());

    var solver = try Solver.init(&z3_ctx, testing.allocator);
    defer solver.deinit();
    solver.assert(z3.Z3_mk_not(z3_ctx.ctx, z3.Z3_mk_eq(z3_ctx.ctx, while_via_operation, while_via_result)));
    try testing.expectEqual(@as(z3.Z3_lbool, z3.Z3_L_FALSE), solver.check());

    solver.reset();
    solver.assert(z3.Z3_mk_not(z3_ctx.ctx, z3.Z3_mk_eq(z3_ctx.ctx, for_via_operation, for_via_result)));
    try testing.expectEqual(@as(z3.Z3_lbool, z3.Z3_L_FALSE), solver.check());
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

test "arithmetic div and rem by zero encode EVM zero result" {
    var z3_ctx = try Context.init(testing.allocator);
    defer z3_ctx.deinit();

    var encoder = Encoder.init(&z3_ctx, testing.allocator);
    defer encoder.deinit();

    const lhs = try encoder.encodeIntegerConstant(10, 256);
    const rhs_zero = try encoder.encodeIntegerConstant(0, 256);
    const expected_zero = try encoder.encodeIntegerConstant(0, 256);

    inline for (.{ Encoder.ArithmeticOp.DivUnsigned, Encoder.ArithmeticOp.RemUnsigned, Encoder.ArithmeticOp.DivSigned, Encoder.ArithmeticOp.RemSigned }) |op| {
        const ast = try encoder.encodeArithmeticOp(op, lhs, rhs_zero);
        try expectAstEquivalent(&z3_ctx, ast, expected_zero);
    }
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

test "ora.assert records checked power overflow obligations without degradation" {
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

    const base_op = mlir.oraVariablePlaceholderOpCreate(mlir_ctx, loc, stringRef("powerBase"), i8_ty);
    const exp_op = mlir.oraVariablePlaceholderOpCreate(mlir_ctx, loc, stringRef("powerExp"), i8_ty);
    const base_arg = mlir.oraOperationGetResult(base_op, 0);
    const exp_arg = mlir.oraOperationGetResult(exp_op, 0);

    const pow_op = mlir.oraPowerOpCreate(mlir_ctx, loc, base_arg, exp_arg, i8_ty);
    const pow = mlir.oraOperationGetResult(pow_op, 0);

    const limit_attr = mlir.oraIntegerAttrCreateI64FromType(i8_ty, 81);
    const limit_const = mlir.oraArithConstantOpCreate(mlir_ctx, loc, i8_ty, limit_attr);
    const limit = mlir.oraOperationGetResult(limit_const, 0);

    const cmp_op = mlir.oraArithCmpIOpCreate(mlir_ctx, loc, 6, pow, limit); // ule
    const within_bound = mlir.oraOperationGetResult(cmp_op, 0);
    const assert_op = mlir.oraAssertOpCreate(mlir_ctx, loc, within_bound, stringRef("checked power overflow"));

    _ = try encoder.encodeOperation(assert_op);

    const obligations = try encoder.takeObligations(testing.allocator);
    defer if (obligations.len > 0) testing.allocator.free(obligations);
    try testing.expect(obligations.len >= 1);
    try testing.expect(!encoder.isDegraded());
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

test "mixed-width signed comparison sign-extends narrower operand" {
    var z3_ctx = try Context.init(testing.allocator);
    defer z3_ctx.deinit();

    var encoder = Encoder.init(&z3_ctx, testing.allocator);
    defer encoder.deinit();

    const i8_sort = z3.Z3_mk_bv_sort(z3_ctx.ctx, 8);
    const i16_sort = z3.Z3_mk_bv_sort(z3_ctx.ctx, 16);
    const minus_one_i8 = z3.Z3_mk_numeral(z3_ctx.ctx, "255", i8_sort);
    const plus_one_i16 = z3.Z3_mk_numeral(z3_ctx.ctx, "1", i16_sort);

    const signed_lt = try encoder.encodeCmpOp(2, &[_]z3.Z3_ast{ minus_one_i8, plus_one_i16 });
    const unsigned_lt = try encoder.encodeCmpOp(6, &[_]z3.Z3_ast{ minus_one_i8, plus_one_i16 });

    var solver = try Solver.init(&z3_ctx, testing.allocator);
    defer solver.deinit();

    solver.assert(z3.Z3_mk_not(z3_ctx.ctx, signed_lt));
    try testing.expectEqual(@as(z3.Z3_lbool, z3.Z3_L_FALSE), solver.check());

    solver.push();
    defer solver.pop();
    solver.assert(unsigned_lt);
    try testing.expectEqual(@as(z3.Z3_lbool, z3.Z3_L_FALSE), solver.check());
}

test "ora.cmp mixed-width signed comparison sign-extends narrower operand" {
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
    const i8_ty = mlir.oraIntegerTypeCreate(mlir_ctx, 8);
    const i16_ty = mlir.oraIntegerTypeCreate(mlir_ctx, 16);

    const lhs_attr = mlir.oraIntegerAttrCreateI64FromType(i8_ty, -1);
    const rhs_attr = mlir.oraIntegerAttrCreateI64FromType(i16_ty, 1);
    const lhs_const = mlir.oraArithConstantOpCreate(mlir_ctx, loc, i8_ty, lhs_attr);
    const rhs_const = mlir.oraArithConstantOpCreate(mlir_ctx, loc, i16_ty, rhs_attr);
    const lhs = mlir.oraOperationGetResult(lhs_const, 0);
    const rhs = mlir.oraOperationGetResult(rhs_const, 0);

    const signed_cmp = mlir.oraCmpOpCreate(mlir_ctx, loc, stringRef("slt"), lhs, rhs, i1_ty);
    const unsigned_cmp = mlir.oraCmpOpCreate(mlir_ctx, loc, stringRef("ult"), lhs, rhs, i1_ty);
    const signed_lt = try encoder.encodeOperation(signed_cmp);
    const unsigned_lt = try encoder.encodeOperation(unsigned_cmp);

    var solver = try Solver.init(&z3_ctx, testing.allocator);
    defer solver.deinit();

    solver.assert(z3.Z3_mk_not(z3_ctx.ctx, signed_lt));
    try testing.expectEqual(@as(z3.Z3_lbool, z3.Z3_L_FALSE), solver.check());

    solver.push();
    defer solver.pop();
    solver.assert(unsigned_lt);
    try testing.expectEqual(@as(z3.Z3_lbool, z3.Z3_L_FALSE), solver.check());
}

test "ora.shl_wrapping coerces narrow shift amount" {
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
    const i256_ty = mlir.oraIntegerTypeCreate(mlir_ctx, 256);

    const lhs_attr = mlir.oraIntegerAttrCreateI64FromType(i256_ty, 1);
    const rhs_attr = mlir.oraIntegerAttrCreateI64FromType(i8_ty, 8);
    const lhs_const = mlir.oraArithConstantOpCreate(mlir_ctx, loc, i256_ty, lhs_attr);
    const rhs_const = mlir.oraArithConstantOpCreate(mlir_ctx, loc, i8_ty, rhs_attr);
    const lhs = mlir.oraOperationGetResult(lhs_const, 0);
    const rhs = mlir.oraOperationGetResult(rhs_const, 0);

    const shl_op = mlir.oraShlWrappingOpCreate(mlir_ctx, loc, lhs, rhs, i256_ty);
    const ast = try encoder.encodeOperation(shl_op);
    const expected = try encoder.encodeIntegerConstant(256, 256);

    var solver = try Solver.init(&z3_ctx, testing.allocator);
    defer solver.deinit();
    solver.assert(z3.Z3_mk_not(z3_ctx.ctx, z3.Z3_mk_eq(z3_ctx.ctx, ast, expected)));
    try testing.expectEqual(@as(z3.Z3_lbool, z3.Z3_L_FALSE), solver.check());

    const obligations = try encoder.takeObligations(testing.allocator);
    defer if (obligations.len > 0) testing.allocator.free(obligations);
    try testing.expectEqual(@as(usize, 0), obligations.len);
}

test "ora.shl_wrapping keeps bvshl for signed integer type" {
    var z3_ctx = try Context.init(testing.allocator);
    defer z3_ctx.deinit();

    var encoder = Encoder.init(&z3_ctx, testing.allocator);
    defer encoder.deinit();

    const mlir_ctx = mlir.oraContextCreate();
    defer mlir.oraContextDestroy(mlir_ctx);
    loadAllDialects(mlir_ctx);
    _ = mlir.oraDialectRegister(mlir_ctx);

    const loc = mlir.oraLocationUnknownGet(mlir_ctx);
    const i256_signed_ty = mlir.oraIntegerTypeGet(mlir_ctx, 256, true);

    const lhs_attr = mlir.oraIntegerAttrCreateI64FromType(i256_signed_ty, -8);
    const rhs_attr = mlir.oraIntegerAttrCreateI64FromType(i256_signed_ty, 1);
    const lhs_const = mlir.oraArithConstantOpCreate(mlir_ctx, loc, i256_signed_ty, lhs_attr);
    const rhs_const = mlir.oraArithConstantOpCreate(mlir_ctx, loc, i256_signed_ty, rhs_attr);
    const lhs = mlir.oraOperationGetResult(lhs_const, 0);
    const rhs = mlir.oraOperationGetResult(rhs_const, 0);

    const shl_op = mlir.oraShlWrappingOpCreate(mlir_ctx, loc, lhs, rhs, i256_signed_ty);
    const ast = try encoder.encodeOperation(shl_op);
    const ast_str = std.mem.span(z3.Z3_ast_to_string(z3_ctx.ctx, ast));
    try testing.expect(std.mem.indexOf(u8, ast_str, "bvshl") != null);
    try testing.expect(std.mem.indexOf(u8, ast_str, "bvashr") == null);
    try testing.expect(std.mem.indexOf(u8, ast_str, "bvlshr") == null);
}

test "ora.shr_wrapping uses arithmetic shift for signed integer type" {
    var z3_ctx = try Context.init(testing.allocator);
    defer z3_ctx.deinit();

    var encoder = Encoder.init(&z3_ctx, testing.allocator);
    defer encoder.deinit();

    const mlir_ctx = mlir.oraContextCreate();
    defer mlir.oraContextDestroy(mlir_ctx);
    loadAllDialects(mlir_ctx);
    _ = mlir.oraDialectRegister(mlir_ctx);

    const loc = mlir.oraLocationUnknownGet(mlir_ctx);
    const i256_signed_ty = mlir.oraIntegerTypeGet(mlir_ctx, 256, true);

    const lhs_attr = mlir.oraIntegerAttrCreateI64FromType(i256_signed_ty, -8);
    const rhs_attr = mlir.oraIntegerAttrCreateI64FromType(i256_signed_ty, 1);
    const expected_attr = mlir.oraIntegerAttrCreateI64FromType(i256_signed_ty, -4);
    const lhs_const = mlir.oraArithConstantOpCreate(mlir_ctx, loc, i256_signed_ty, lhs_attr);
    const rhs_const = mlir.oraArithConstantOpCreate(mlir_ctx, loc, i256_signed_ty, rhs_attr);
    const expected_const = mlir.oraArithConstantOpCreate(mlir_ctx, loc, i256_signed_ty, expected_attr);
    const lhs = mlir.oraOperationGetResult(lhs_const, 0);
    const rhs = mlir.oraOperationGetResult(rhs_const, 0);

    const shr_op = mlir.oraShrWrappingOpCreate(mlir_ctx, loc, lhs, rhs, i256_signed_ty);
    const ast = try encoder.encodeOperation(shr_op);
    const expected = try encoder.encodeOperation(expected_const);

    const ast_str = std.mem.span(z3.Z3_ast_to_string(z3_ctx.ctx, ast));
    try testing.expect(std.mem.indexOf(u8, ast_str, "bvashr") != null);

    var solver = try Solver.init(&z3_ctx, testing.allocator);
    defer solver.deinit();
    solver.assert(z3.Z3_mk_not(z3_ctx.ctx, z3.Z3_mk_eq(z3_ctx.ctx, ast, expected)));
    try testing.expectEqual(@as(z3.Z3_lbool, z3.Z3_L_FALSE), solver.check());
}

test "ora.shr_wrapping coerces narrow shift amount for signed integer type" {
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
    const i256_signed_ty = mlir.oraIntegerTypeGet(mlir_ctx, 256, true);

    const lhs_attr = mlir.oraIntegerAttrCreateI64FromType(i256_signed_ty, -8);
    const rhs_attr = mlir.oraIntegerAttrCreateI64FromType(i8_ty, 1);
    const expected_attr = mlir.oraIntegerAttrCreateI64FromType(i256_signed_ty, -4);
    const lhs_const = mlir.oraArithConstantOpCreate(mlir_ctx, loc, i256_signed_ty, lhs_attr);
    const rhs_const = mlir.oraArithConstantOpCreate(mlir_ctx, loc, i8_ty, rhs_attr);
    const expected_const = mlir.oraArithConstantOpCreate(mlir_ctx, loc, i256_signed_ty, expected_attr);
    const lhs = mlir.oraOperationGetResult(lhs_const, 0);
    const rhs = mlir.oraOperationGetResult(rhs_const, 0);

    const shr_op = mlir.oraShrWrappingOpCreate(mlir_ctx, loc, lhs, rhs, i256_signed_ty);
    const ast = try encoder.encodeOperation(shr_op);
    const expected = try encoder.encodeOperation(expected_const);

    const ast_str = std.mem.span(z3.Z3_ast_to_string(z3_ctx.ctx, ast));
    try testing.expect(std.mem.indexOf(u8, ast_str, "bvashr") != null);

    var solver = try Solver.init(&z3_ctx, testing.allocator);
    defer solver.deinit();
    solver.assert(z3.Z3_mk_not(z3_ctx.ctx, z3.Z3_mk_eq(z3_ctx.ctx, ast, expected)));
    try testing.expectEqual(@as(z3.Z3_lbool, z3.Z3_L_FALSE), solver.check());
}

test "ora.shr_wrapping keeps logical shift for unsigned integer type" {
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

    const shr_op = mlir.oraShrWrappingOpCreate(mlir_ctx, loc, lhs, rhs, i256_ty);
    mlir.oraOperationSetAttributeByName(
        shr_op,
        stringRef("ora.integer_signed"),
        mlir.oraBoolAttrCreate(mlir_ctx, false),
    );
    const ast = try encoder.encodeOperation(shr_op);
    const ast_str = std.mem.span(z3.Z3_ast_to_string(z3_ctx.ctx, ast));
    try testing.expect(std.mem.indexOf(u8, ast_str, "bvlshr") != null);
}

test "bitvector width-changing coercions degrade instead of silently widening or narrowing" {
    var z3_ctx = try Context.init(testing.allocator);
    defer z3_ctx.deinit();

    var encoder = Encoder.init(&z3_ctx, testing.allocator);
    defer encoder.deinit();

    const bv8_sort = z3.Z3_mk_bv_sort(z3_ctx.ctx, 8);
    const bv16_sort = z3.Z3_mk_bv_sort(z3_ctx.ctx, 16);
    const ast = z3.Z3_mk_const(
        z3_ctx.ctx,
        z3.Z3_mk_string_symbol(z3_ctx.ctx, "w"),
        bv8_sort,
    );

    const widened = encoder.coerceAstToSortForTesting(ast, bv16_sort);
    try testing.expect(encoder.isDegraded());
    try testing.expectEqualStrings("bitvector widening requires explicit signedness-aware coercion", encoder.degradationReason().?);
    try testing.expectEqual(@as(u32, 16), @as(u32, @intCast(z3.Z3_get_bv_sort_size(z3_ctx.ctx, z3.Z3_get_sort(z3_ctx.ctx, widened)))));

    var encoder_narrow = Encoder.init(&z3_ctx, testing.allocator);
    defer encoder_narrow.deinit();
    const narrowed = encoder_narrow.coerceAstToSortForTesting(widened, bv8_sort);
    try testing.expect(encoder_narrow.isDegraded());
    try testing.expectEqualStrings("bitvector narrowing requires explicit width-fit proof", encoder_narrow.degradationReason().?);
    try testing.expectEqual(@as(u32, 8), @as(u32, @intCast(z3.Z3_get_bv_sort_size(z3_ctx.ctx, z3.Z3_get_sort(z3_ctx.ctx, narrowed)))));
}

test "ora.length to u256 result is exact under bounded byte-sequence model" {
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
    const result_ty = mlir.oraIntegerTypeCreate(mlir_ctx, 256);
    const bytes_op = mlir.oraBytesConstantOpCreate(mlir_ctx, loc, stringRef("0xdeadbeef"), bytes_ty);
    const length_op = mlir.oraLengthOpCreate(mlir_ctx, loc, mlir.oraOperationGetResult(bytes_op, 0), result_ty);

    const ast = try encoder.encodeOperation(length_op);
    try testing.expect(!encoder.isDegraded());

    var solver = try Solver.init(&z3_ctx, testing.allocator);
    defer solver.deinit();
    const expected = try encoder.encodeIntegerConstant(4, 256);
    solver.assert(z3.Z3_mk_not(z3_ctx.ctx, z3.Z3_mk_eq(z3_ctx.ctx, ast, expected)));
    try testing.expectEqual(@as(z3.Z3_lbool, z3.Z3_L_FALSE), solver.check());
}

test "byte-sequence concat length composes without quantified frame axioms" {
    var z3_ctx = try Context.init(testing.allocator);
    defer z3_ctx.deinit();

    var encoder = Encoder.init(&z3_ctx, testing.allocator);
    defer encoder.deinit();

    const seq_sort = z3.Z3_mk_seq_sort(z3_ctx.ctx, z3.Z3_mk_bv_sort(z3_ctx.ctx, 8));
    const lhs = try encoder.mkVariable("lhs_bytes", seq_sort);
    const rhs = try encoder.mkVariable("rhs_bytes", seq_sort);
    const concat = try encoder.encodeByteSequenceConcatForTesting(lhs, rhs);

    const constraints = try encoder.takeConstraints(testing.allocator);
    defer if (constraints.len > 0) testing.allocator.free(constraints);
    try expectNoQuantifiedConstraints(&z3_ctx, constraints);

    var solver = try Solver.init(&z3_ctx, testing.allocator);
    defer solver.deinit();
    for (constraints) |constraint| solver.assert(constraint);

    const lhs_len = encoder.encodeByteSequenceLengthBvForTesting(lhs);
    const rhs_len = encoder.encodeByteSequenceLengthBvForTesting(rhs);
    const concat_len = encoder.encodeByteSequenceLengthBvForTesting(concat);
    const expected = z3.Z3_mk_bv_add(z3_ctx.ctx, lhs_len, rhs_len);
    solver.assert(z3.Z3_mk_not(z3_ctx.ctx, z3.Z3_mk_eq(z3_ctx.ctx, concat_len, expected)));
    try testing.expectEqual(@as(z3.Z3_lbool, z3.Z3_L_FALSE), solver.check());
}

test "byte-sequence slice length composes without quantified frame axioms" {
    var z3_ctx = try Context.init(testing.allocator);
    defer z3_ctx.deinit();

    var encoder = Encoder.init(&z3_ctx, testing.allocator);
    defer encoder.deinit();

    const seq_sort = z3.Z3_mk_seq_sort(z3_ctx.ctx, z3.Z3_mk_bv_sort(z3_ctx.ctx, 8));
    const bytes = try encoder.mkVariable("slice_source", seq_sort);
    const offset = try encoder.encodeIntegerConstant(2, 256);
    const len = try encoder.encodeIntegerConstant(4, 256);
    const slice = try encoder.encodeByteSequenceSliceForTesting(bytes, offset, len);

    const constraints = try encoder.takeConstraints(testing.allocator);
    defer if (constraints.len > 0) testing.allocator.free(constraints);
    try expectNoQuantifiedConstraints(&z3_ctx, constraints);

    var solver = try Solver.init(&z3_ctx, testing.allocator);
    defer solver.deinit();
    for (constraints) |constraint| solver.assert(constraint);

    const slice_len = encoder.encodeByteSequenceLengthBvForTesting(slice);
    solver.assert(z3.Z3_mk_not(z3_ctx.ctx, z3.Z3_mk_eq(z3_ctx.ctx, slice_len, len)));
    try testing.expectEqual(@as(z3.Z3_lbool, z3.Z3_L_FALSE), solver.check());
}

test "ora.concat operation length composes under bounded byte-sequence model" {
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
    const lhs_op = mlir.oraBytesConstantOpCreate(mlir_ctx, loc, stringRef("0xdead"), bytes_ty);
    const rhs_op = mlir.oraBytesConstantOpCreate(mlir_ctx, loc, stringRef("0xbeefca"), bytes_ty);
    const concat_op = mlir.oraConcatOpCreate(mlir_ctx, loc, mlir.oraOperationGetResult(lhs_op, 0), mlir.oraOperationGetResult(rhs_op, 0), bytes_ty);

    const ast = try encoder.encodeOperation(concat_op);
    try testing.expect(!encoder.isDegraded());

    const constraints = try encoder.takeConstraints(testing.allocator);
    defer if (constraints.len > 0) testing.allocator.free(constraints);
    try expectNoQuantifiedConstraints(&z3_ctx, constraints);

    var solver = try Solver.init(&z3_ctx, testing.allocator);
    defer solver.deinit();
    for (constraints) |constraint| solver.assert(constraint);

    const expected_len = try encoder.encodeIntegerConstant(5, 256);
    const actual_len = encoder.encodeByteSequenceLengthBvForTesting(ast);
    solver.assert(z3.Z3_mk_not(z3_ctx.ctx, z3.Z3_mk_eq(z3_ctx.ctx, actual_len, expected_len)));
    try testing.expectEqual(@as(z3.Z3_lbool, z3.Z3_L_FALSE), solver.check());
}

test "ora.slice operation length composes under bounded byte-sequence model" {
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
    const i256_ty = mlir.oraIntegerTypeCreate(mlir_ctx, 256);
    const source_op = mlir.oraBytesConstantOpCreate(mlir_ctx, loc, stringRef("0xdeadbeefcafe"), bytes_ty);
    const start_attr = mlir.oraIntegerAttrCreateI64FromType(i256_ty, 1);
    const length_attr = mlir.oraIntegerAttrCreateI64FromType(i256_ty, 3);
    const start_op = mlir.oraArithConstantOpCreate(mlir_ctx, loc, i256_ty, start_attr);
    const length_op = mlir.oraArithConstantOpCreate(mlir_ctx, loc, i256_ty, length_attr);
    const slice_op = mlir.oraSliceOpCreate(
        mlir_ctx,
        loc,
        mlir.oraOperationGetResult(source_op, 0),
        mlir.oraOperationGetResult(start_op, 0),
        mlir.oraOperationGetResult(length_op, 0),
        bytes_ty,
    );

    const ast = try encoder.encodeOperation(slice_op);
    try testing.expect(!encoder.isDegraded());

    const constraints = try encoder.takeConstraints(testing.allocator);
    defer if (constraints.len > 0) testing.allocator.free(constraints);
    try expectNoQuantifiedConstraints(&z3_ctx, constraints);

    var solver = try Solver.init(&z3_ctx, testing.allocator);
    defer solver.deinit();
    for (constraints) |constraint| solver.assert(constraint);

    const expected_len = try encoder.encodeIntegerConstant(3, 256);
    const actual_len = encoder.encodeByteSequenceLengthBvForTesting(ast);
    solver.assert(z3.Z3_mk_not(z3_ctx.ctx, z3.Z3_mk_eq(z3_ctx.ctx, actual_len, expected_len)));
    try testing.expectEqual(@as(z3.Z3_lbool, z3.Z3_L_FALSE), solver.check());
}

test "ora.length to narrow bitvector result degrades instead of applying unchecked int2bv" {
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
    const result_ty = mlir.oraIntegerTypeCreate(mlir_ctx, 8);
    const bytes_op = mlir.oraBytesConstantOpCreate(mlir_ctx, loc, stringRef("0xdeadbeef"), bytes_ty);
    const length_op = mlir.oraLengthOpCreate(mlir_ctx, loc, mlir.oraOperationGetResult(bytes_op, 0), result_ty);

    const ast = try encoder.encodeOperation(length_op);
    try testing.expect(encoder.isDegraded());
    try testing.expectEqualStrings("sequence length to narrow bitvector requires explicit bound proof", encoder.degradationReason().?);
    try testing.expectEqual(@as(u32, 8), @as(u32, @intCast(z3.Z3_get_bv_sort_size(z3_ctx.ctx, z3.Z3_get_sort(z3_ctx.ctx, ast)))));
}

test "ora.byte_at zero-extends extracted byte to wider bitvector results" {
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
    const i256_ty = mlir.oraIntegerTypeCreate(mlir_ctx, 256);
    const index_ty = mlir.oraIntegerTypeCreate(mlir_ctx, 256);
    const bytes_op = mlir.oraBytesConstantOpCreate(mlir_ctx, loc, stringRef("0xab"), bytes_ty);
    const index_attr = mlir.oraIntegerAttrCreateI64FromType(index_ty, 0);
    const index_op = mlir.oraArithConstantOpCreate(mlir_ctx, loc, index_ty, index_attr);
    const byte_at = mlir.oraByteAtOpCreate(mlir_ctx, loc, mlir.oraOperationGetResult(bytes_op, 0), mlir.oraOperationGetResult(index_op, 0), i256_ty);

    const ast = try encoder.encodeOperation(byte_at);
    try testing.expect(!encoder.isDegraded());

    const expected = z3.Z3_mk_numeral(z3_ctx.ctx, "171", z3.Z3_get_sort(z3_ctx.ctx, ast));
    var solver = try Solver.init(&z3_ctx, testing.allocator);
    defer solver.deinit();
    solver.assert(z3.Z3_mk_not(z3_ctx.ctx, z3.Z3_mk_eq(z3_ctx.ctx, ast, expected)));
    try testing.expectEqual(@as(z3.Z3_lbool, z3.Z3_L_FALSE), solver.check());
}

test "ora.keccak256 encodes as deterministic uninterpreted function over bytes" {
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
    const i256_ty = mlir.oraIntegerTypeCreate(mlir_ctx, 256);

    const bytes_a_op = mlir.oraBytesConstantOpCreate(mlir_ctx, loc, stringRef("0xdeadbeef"), bytes_ty);
    const bytes_b_op = mlir.oraBytesConstantOpCreate(mlir_ctx, loc, stringRef("0xcafebabe"), bytes_ty);
    const bytes_a = mlir.oraOperationGetResult(bytes_a_op, 0);
    const bytes_b = mlir.oraOperationGetResult(bytes_b_op, 0);
    var operands_a = [_]mlir.MlirValue{bytes_a};
    var operands_b = [_]mlir.MlirValue{bytes_b};

    const hash_a_op = mlir.oraEvmOpCreate(mlir_ctx, loc, stringRef("ora.keccak256"), &operands_a, operands_a.len, i256_ty);
    const hash_a_again_op = mlir.oraEvmOpCreate(mlir_ctx, loc, stringRef("ora.keccak256"), &operands_a, operands_a.len, i256_ty);
    const hash_b_op = mlir.oraEvmOpCreate(mlir_ctx, loc, stringRef("ora.keccak256"), &operands_b, operands_b.len, i256_ty);

    const hash_a = try encoder.encodeOperation(hash_a_op);
    const hash_a_again = try encoder.encodeOperation(hash_a_again_op);
    const hash_b = try encoder.encodeOperation(hash_b_op);
    try testing.expect(!encoder.isDegraded());

    try expectAstEquivalent(&z3_ctx, hash_a, hash_a_again);

    var solver_eq = try Solver.init(&z3_ctx, testing.allocator);
    defer solver_eq.deinit();
    solver_eq.assert(z3.Z3_mk_eq(z3_ctx.ctx, hash_a, hash_b));
    try testing.expectEqual(@as(z3.Z3_lbool, z3.Z3_L_TRUE), solver_eq.check());

    var solver_neq = try Solver.init(&z3_ctx, testing.allocator);
    defer solver_neq.deinit();
    solver_neq.assert(z3.Z3_mk_not(z3_ctx.ctx, z3.Z3_mk_eq(z3_ctx.ctx, hash_a, hash_b)));
    try testing.expectEqual(@as(z3.Z3_lbool, z3.Z3_L_TRUE), solver_neq.check());
}

test "precompile boundaries encode as deterministic uninterpreted functions" {
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
    const addr_ty = mlir.oraAddressTypeGet(mlir_ctx);
    const i8_ty = mlir.oraIntegerTypeCreate(mlir_ctx, 8);
    const i160_ty = mlir.oraIntegerTypeCreate(mlir_ctx, 160);
    const i256_ty = mlir.oraIntegerTypeCreate(mlir_ctx, 256);

    const data_op = mlir.oraVariablePlaceholderOpCreate(mlir_ctx, loc, stringRef("precompileData"), bytes_ty);
    const other_data_op = mlir.oraVariablePlaceholderOpCreate(mlir_ctx, loc, stringRef("precompileOtherData"), bytes_ty);
    const hash_op = mlir.oraVariablePlaceholderOpCreate(mlir_ctx, loc, stringRef("precompileHash"), i256_ty);
    const v_op = mlir.oraVariablePlaceholderOpCreate(mlir_ctx, loc, stringRef("precompileV"), i8_ty);
    const r_op = mlir.oraVariablePlaceholderOpCreate(mlir_ctx, loc, stringRef("precompileR"), i256_ty);
    const s_op = mlir.oraVariablePlaceholderOpCreate(mlir_ctx, loc, stringRef("precompileS"), i256_ty);
    const x1_op = mlir.oraVariablePlaceholderOpCreate(mlir_ctx, loc, stringRef("precompileX1"), i256_ty);
    const y1_op = mlir.oraVariablePlaceholderOpCreate(mlir_ctx, loc, stringRef("precompileY1"), i256_ty);
    const x2_op = mlir.oraVariablePlaceholderOpCreate(mlir_ctx, loc, stringRef("precompileX2"), i256_ty);
    const y2_op = mlir.oraVariablePlaceholderOpCreate(mlir_ctx, loc, stringRef("precompileY2"), i256_ty);
    const scalar_op = mlir.oraVariablePlaceholderOpCreate(mlir_ctx, loc, stringRef("precompileScalar"), i256_ty);
    const rounds_op = mlir.oraVariablePlaceholderOpCreate(mlir_ctx, loc, stringRef("precompileRounds"), i256_ty);
    const t0_op = mlir.oraVariablePlaceholderOpCreate(mlir_ctx, loc, stringRef("precompileT0"), i256_ty);
    const t1_op = mlir.oraVariablePlaceholderOpCreate(mlir_ctx, loc, stringRef("precompileT1"), i256_ty);
    const final_op = mlir.oraVariablePlaceholderOpCreate(mlir_ctx, loc, stringRef("precompileFinal"), i8_ty);

    const data = mlir.oraOperationGetResult(data_op, 0);
    const other_data = mlir.oraOperationGetResult(other_data_op, 0);
    const hash = mlir.oraOperationGetResult(hash_op, 0);
    const v = mlir.oraOperationGetResult(v_op, 0);
    const r = mlir.oraOperationGetResult(r_op, 0);
    const s = mlir.oraOperationGetResult(s_op, 0);
    const x1 = mlir.oraOperationGetResult(x1_op, 0);
    const y1 = mlir.oraOperationGetResult(y1_op, 0);
    const x2 = mlir.oraOperationGetResult(x2_op, 0);
    const y2 = mlir.oraOperationGetResult(y2_op, 0);
    const scalar = mlir.oraOperationGetResult(scalar_op, 0);
    const rounds = mlir.oraOperationGetResult(rounds_op, 0);
    const t0 = mlir.oraOperationGetResult(t0_op, 0);
    const t1 = mlir.oraOperationGetResult(t1_op, 0);
    const final = mlir.oraOperationGetResult(final_op, 0);

    const Helper = struct {
        fn expectPrecompile(
            ctx: *Context,
            enc: *Encoder,
            mlir_ctx_inner: mlir.MlirContext,
            loc_inner: mlir.MlirLocation,
            comptime op_name: []const u8,
            comptime symbol_name: []const u8,
            operands: []mlir.MlirValue,
            result_ty: mlir.MlirType,
            expected_bv_bits: ?u32,
        ) !z3.Z3_ast {
            const first_op = mlir.oraEvmOpCreate(mlir_ctx_inner, loc_inner, stringRef(op_name), operands.ptr, operands.len, result_ty);
            const second_op = mlir.oraEvmOpCreate(mlir_ctx_inner, loc_inner, stringRef(op_name), operands.ptr, operands.len, result_ty);

            const first = try enc.encodeOperation(first_op);
            const second = try enc.encodeOperation(second_op);
            try expectAstEquivalent(ctx, first, second);

            const encoded_text = std.mem.span(z3.Z3_ast_to_string(ctx.ctx, first));
            try testing.expect(std.mem.indexOf(u8, encoded_text, symbol_name) != null);

            const result_sort = z3.Z3_get_sort(ctx.ctx, first);
            if (expected_bv_bits) |bits| {
                try testing.expectEqual(@as(u32, z3.Z3_BV_SORT), @as(u32, @intCast(z3.Z3_get_sort_kind(ctx.ctx, result_sort))));
                try testing.expectEqual(bits, @as(u32, @intCast(z3.Z3_get_bv_sort_size(ctx.ctx, result_sort))));
            } else {
                try testing.expect(z3.Z3_is_seq_sort(ctx.ctx, result_sort));
            }

            return first;
        }
    };

    var ecrecover_operands = [_]mlir.MlirValue{ hash, v, r, s };
    _ = try Helper.expectPrecompile(&z3_ctx, &encoder, mlir_ctx, loc, "ora.precompile.ecrecover", "precompile_ecrecover", &ecrecover_operands, addr_ty, 160);

    var sha_operands = [_]mlir.MlirValue{data};
    const sha_a = try Helper.expectPrecompile(&z3_ctx, &encoder, mlir_ctx, loc, "ora.precompile.sha256", "precompile_sha256", &sha_operands, i256_ty, 256);

    var ripemd_operands = [_]mlir.MlirValue{data};
    _ = try Helper.expectPrecompile(&z3_ctx, &encoder, mlir_ctx, loc, "ora.precompile.ripemd160", "precompile_ripemd160", &ripemd_operands, i160_ty, 160);

    var identity_operands = [_]mlir.MlirValue{data};
    _ = try Helper.expectPrecompile(&z3_ctx, &encoder, mlir_ctx, loc, "ora.precompile.identity", "precompile_identity", &identity_operands, bytes_ty, null);

    var modexp_operands = [_]mlir.MlirValue{ data, other_data, data };
    _ = try Helper.expectPrecompile(&z3_ctx, &encoder, mlir_ctx, loc, "ora.precompile.modexp", "precompile_modexp", &modexp_operands, i256_ty, 256);

    var ecadd_operands = [_]mlir.MlirValue{ x1, y1, x2, y2 };
    _ = try Helper.expectPrecompile(&z3_ctx, &encoder, mlir_ctx, loc, "ora.precompile.ecadd", "precompile_ecadd", &ecadd_operands, i256_ty, 256);

    var ecmul_operands = [_]mlir.MlirValue{ x1, y1, scalar };
    _ = try Helper.expectPrecompile(&z3_ctx, &encoder, mlir_ctx, loc, "ora.precompile.ecmul", "precompile_ecmul", &ecmul_operands, i256_ty, 256);

    var ecpairing_operands = [_]mlir.MlirValue{data};
    const ecpairing = try Helper.expectPrecompile(&z3_ctx, &encoder, mlir_ctx, loc, "ora.precompile.ecpairing", "precompile_ecpairing", &ecpairing_operands, i256_ty, 256);

    var blake2f_operands = [_]mlir.MlirValue{ rounds, data, other_data, t0, t1, final };
    _ = try Helper.expectPrecompile(&z3_ctx, &encoder, mlir_ctx, loc, "ora.precompile.blake2f", "precompile_blake2f", &blake2f_operands, i256_ty, 256);

    var sha_b_operands = [_]mlir.MlirValue{other_data};
    const sha_b_op = mlir.oraEvmOpCreate(mlir_ctx, loc, stringRef("ora.precompile.sha256"), &sha_b_operands, sha_b_operands.len, i256_ty);
    const sha_b = try encoder.encodeOperation(sha_b_op);
    try testing.expect(!encoder.isDegraded());

    var solver_eq = try Solver.init(&z3_ctx, testing.allocator);
    defer solver_eq.deinit();
    solver_eq.assert(z3.Z3_mk_eq(z3_ctx.ctx, sha_a, sha_b));
    try testing.expectEqual(@as(z3.Z3_lbool, z3.Z3_L_TRUE), solver_eq.check());

    var solver_neq = try Solver.init(&z3_ctx, testing.allocator);
    defer solver_neq.deinit();
    solver_neq.assert(z3.Z3_mk_not(z3_ctx.ctx, z3.Z3_mk_eq(z3_ctx.ctx, sha_a, sha_b)));
    try testing.expectEqual(@as(z3.Z3_lbool, z3.Z3_L_TRUE), solver_neq.check());

    var solver_cross_precompile = try Solver.init(&z3_ctx, testing.allocator);
    defer solver_cross_precompile.deinit();
    solver_cross_precompile.assert(z3.Z3_mk_not(z3_ctx.ctx, z3.Z3_mk_eq(z3_ctx.ctx, sha_a, ecpairing)));
    try testing.expectEqual(@as(z3.Z3_lbool, z3.Z3_L_TRUE), solver_cross_precompile.check());
}

test "precompile boundaries reject unregistered and mismatched shapes" {
    var z3_ctx = try Context.init(testing.allocator);
    defer z3_ctx.deinit();

    const mlir_ctx = mlir.oraContextCreate();
    defer mlir.oraContextDestroy(mlir_ctx);
    loadAllDialects(mlir_ctx);
    _ = mlir.oraDialectRegister(mlir_ctx);

    const loc = mlir.oraLocationUnknownGet(mlir_ctx);
    const bytes_ty = mlir.oraBytesTypeGet(mlir_ctx);
    const i256_ty = mlir.oraIntegerTypeCreate(mlir_ctx, 256);
    const data_op = mlir.oraVariablePlaceholderOpCreate(mlir_ctx, loc, stringRef("unregisteredPrecompileData"), bytes_ty);
    const data = mlir.oraOperationGetResult(data_op, 0);

    var unregistered_encoder = Encoder.init(&z3_ctx, testing.allocator);
    defer unregistered_encoder.deinit();
    var unregistered_operands = [_]mlir.MlirValue{data};
    const unregistered = mlir.oraEvmOpCreate(mlir_ctx, loc, stringRef("ora.precompile.modexp_alt"), &unregistered_operands, unregistered_operands.len, i256_ty);
    try testing.expectError(error.UnsupportedOperation, unregistered_encoder.encodeOperation(unregistered));
    try testing.expect(unregistered_encoder.isDegraded());

    var width_encoder = Encoder.init(&z3_ctx, testing.allocator);
    defer width_encoder.deinit();
    var ecrecover_operands = [_]mlir.MlirValue{ data, data, data, data };
    const mistyped_ecrecover = mlir.oraEvmOpCreate(mlir_ctx, loc, stringRef("ora.precompile.ecrecover"), &ecrecover_operands, ecrecover_operands.len, i256_ty);
    try testing.expectError(error.UnsupportedOperation, width_encoder.encodeOperation(mistyped_ecrecover));
    try testing.expect(width_encoder.isDegraded());

    var arity_encoder = Encoder.init(&z3_ctx, testing.allocator);
    defer arity_encoder.deinit();
    var short_modexp_operands = [_]mlir.MlirValue{data};
    const short_modexp = mlir.oraEvmOpCreate(mlir_ctx, loc, stringRef("ora.precompile.modexp"), &short_modexp_operands, short_modexp_operands.len, i256_ty);
    try testing.expectError(error.UnsupportedOperation, arity_encoder.encodeOperation(short_modexp));
    try testing.expect(arity_encoder.isDegraded());
}

test "degradation reason keeps first recorded cause" {
    var z3_ctx = try Context.init(testing.allocator);
    defer z3_ctx.deinit();

    var encoder = Encoder.init(&z3_ctx, testing.allocator);
    defer encoder.deinit();

    encoder.noteDegradation("first degradation");
    encoder.noteDegradation("second degradation");
    encoder.noteDegradation("third degradation");

    try testing.expect(encoder.isDegraded());
    try testing.expectEqualStrings("first degradation", encoder.degradationReason().?);
}

test "degradation reasons are capped and preserve encounter order" {
    var z3_ctx = try Context.init(testing.allocator);
    defer z3_ctx.deinit();

    var encoder = Encoder.init(&z3_ctx, testing.allocator);
    defer encoder.deinit();

    inline for (0..12) |idx| {
        const reason = switch (idx) {
            0 => "reason-0",
            1 => "reason-1",
            2 => "reason-2",
            3 => "reason-3",
            4 => "reason-4",
            5 => "reason-5",
            6 => "reason-6",
            7 => "reason-7",
            8 => "reason-8",
            9 => "reason-9",
            10 => "reason-10",
            11 => "reason-11",
            else => unreachable,
        };
        encoder.noteDegradation(reason);
    }

    const reasons = encoder.degradationReasons();
    try testing.expectEqual(@as(usize, 10), reasons.len);
    try testing.expectEqualStrings("reason-0", reasons[0]);
    try testing.expectEqualStrings("reason-9", reasons[9]);
    try testing.expectEqualStrings("reason-0", encoder.degradationReason().?);
}

test "soundness loss kinds are structured and capped" {
    comptime std.debug.assert(std.meta.fields(Encoder.SoundnessLoss).len > 10);

    var z3_ctx = try Context.init(testing.allocator);
    defer z3_ctx.deinit();

    var encoder = Encoder.init(&z3_ctx, testing.allocator);
    defer encoder.deinit();

    // Pillar 4.5 self-introspection: labels are a stable machine-facing API.
    inline for (std.meta.fields(Encoder.SoundnessLoss)) |field| {
        const loss: Encoder.SoundnessLoss = @enumFromInt(field.value);
        const label = Encoder.soundnessLossLabel(loss);
        try testing.expectEqualStrings(field.name, label);
        encoder.noteSoundnessLoss(loss, label);
    }

    const losses = encoder.soundnessLosses();
    try testing.expect(encoder.isDegraded());
    try testing.expectEqual(@as(usize, 10), losses.len);
    try testing.expectEqual(Encoder.SoundnessLoss.user_disabled_state_verification, losses[0]);
    try testing.expectEqual(Encoder.SoundnessLoss.soundness_loss_cap_exceeded, losses[9]);
}

test "soundness loss cap reports truncation explicitly" {
    var z3_ctx = try Context.init(testing.allocator);
    defer z3_ctx.deinit();

    var encoder = Encoder.init(&z3_ctx, testing.allocator);
    defer encoder.deinit();

    encoder.noteSoundnessLoss(.unsupported_operation, "loss-0");
    encoder.noteSoundnessLoss(.unsupported_sort_coercion, "loss-1");
    encoder.noteSoundnessLoss(.unsupported_error_encoding, "loss-2");
    encoder.noteSoundnessLoss(.missing_type_metadata, "loss-3");
    encoder.noteSoundnessLoss(.missing_product_metadata, "loss-4");
    encoder.noteSoundnessLoss(.missing_control_flow_summary, "loss-5");
    encoder.noteSoundnessLoss(.unresolved_callee, "loss-6");
    encoder.noteSoundnessLoss(.inexact_call_summary, "loss-7");
    encoder.noteSoundnessLoss(.inexact_state_summary, "loss-8");
    encoder.noteSoundnessLoss(.internal_encoding_failure, "loss-9");
    encoder.noteSoundnessLoss(.user_disabled_state_verification, "loss-10");

    const losses = encoder.soundnessLosses();
    try testing.expectEqual(@as(usize, 10), losses.len);
    try testing.expectEqual(Encoder.SoundnessLoss.unsupported_operation, losses[0]);
    try testing.expectEqual(Encoder.SoundnessLoss.inexact_state_summary, losses[8]);
    try testing.expectEqual(Encoder.SoundnessLoss.soundness_loss_cap_exceeded, losses[9]);
}

test "production z3 code avoids legacy undef and degradation escape hatches" {
    for (production_z3_sources) |source| {
        try testing.expectEqual(@as(usize, 0), std.mem.count(u8, source, ".noteDegradationAtOp("));
        try testing.expectEqual(@as(usize, 0), std.mem.count(u8, source, "mkUndefValue"));
        try testing.expectEqual(@as(usize, 0), std.mem.count(u8, source, "degradeToUndef"));
    }
}

test "encoder source keeps fresh-symbol and soundness-loss undef APIs separate" {
    const source = @embedFile("encoder.zig");

    try testing.expect(std.mem.indexOf(u8, source, "mkUndefValue") == null);
    try testing.expect(std.mem.indexOf(u8, source, "degradeToUndef") == null);
    try testing.expect(std.mem.indexOf(u8, source, "fn freshSymbol") != null);
    try testing.expect(std.mem.indexOf(u8, source, "fn soundnessLossUndef") != null);
    try testing.expect(std.mem.indexOf(u8, source, "\"try_state_global\"") != null);
    try testing.expect(std.mem.indexOf(u8, source, ".inexact_state_summary") != null);
    try testing.expect(std.mem.indexOf(u8, source, "ora_degraded_coercion_fallback_{d}") != null);
    try testing.expect(std.mem.indexOf(u8, source, "\"ora_degraded_coercion_fallback\"") == null);
}

test "production verifier uses typed soundness loss instead of annotation degradation escape hatch" {
    const verifier_source = @embedFile("verification.zig");
    const mlir_helpers_source = @embedFile("mlir_helpers.zig");
    var found_typed_annotation_failure = false;
    for (production_z3_sources) |source| {
        try testing.expectEqual(@as(usize, 0), std.mem.count(u8, source, ".noteDegradationAtOp("));
        found_typed_annotation_failure = found_typed_annotation_failure or
            std.mem.indexOf(u8, source, ".noteSoundnessLossAtOp(.unsupported_operation") != null;
    }
    try testing.expect(std.mem.indexOf(u8, verifier_source, "restoreEncoderBranchState(&base_encoder_state) catch {};") == null);
    try testing.expect(std.mem.indexOf(u8, verifier_source, ".noteSoundnessLoss(.internal_encoding_failure") != null);
    try testing.expect(std.mem.indexOf(u8, mlir_helpers_source, "if (mlir.oraAttributeIsNull(unsigned_attr)) return false;") == null);
    try testing.expect(std.mem.indexOf(u8, mlir_helpers_source, "return error.UnsupportedOperation;") != null);
    try testing.expect(found_typed_annotation_failure);
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

test "imported-summary opaque modifies metadata fallback preserves only disjoint map key" {
    var z3_ctx = try Context.init(testing.allocator);
    defer z3_ctx.deinit();

    var encoder = Encoder.init(&z3_ctx, testing.allocator);
    defer encoder.deinit();
    encoder.setVerifyState(true);
    encoder.setSummaryOnlyImportedCalls(true);

    const mlir_ctx = mlir.oraContextCreate();
    defer mlir.oraContextDestroy(mlir_ctx);
    loadAllDialects(mlir_ctx);
    _ = mlir.oraDialectRegister(mlir_ctx);

    const loc = mlir.oraLocationUnknownGet(mlir_ctx);
    const i256_ty = mlir.oraIntegerTypeCreate(mlir_ctx, 256);
    const map_ty = mlir.oraMapTypeGet(mlir_ctx, i256_ty, i256_ty);

    const modifies_slots = [_]mlir.MlirAttribute{
        mlir.oraStringAttrCreate(mlir_ctx, stringRef("balances[param#0]")),
    };
    const helper_attrs = [_]mlir.MlirNamedAttribute{
        namedAttr(mlir_ctx, "sym_name", mlir.oraStringAttrCreate(mlir_ctx, stringRef("opaqueSetBalance"))),
        namedAttr(mlir_ctx, "ora.effect", mlir.oraStringAttrCreate(mlir_ctx, stringRef("writes"))),
        namedAttr(mlir_ctx, "ora.modifies_slots", mlir.oraArrayAttrCreate(mlir_ctx, modifies_slots.len, &modifies_slots)),
    };
    const helper_param_types = [_]mlir.MlirType{ i256_ty, i256_ty };
    const helper_param_locs = [_]mlir.MlirLocation{ loc, loc };
    const helper = mlir.oraFuncFuncOpCreate(
        mlir_ctx,
        loc,
        &helper_attrs,
        helper_attrs.len,
        &helper_param_types,
        &helper_param_locs,
        helper_param_types.len,
    );
    const helper_body = mlir.oraFuncOpGetBodyBlock(helper);
    const user_arg = mlir.oraBlockGetArgument(helper_body, 0);
    const value_arg = mlir.oraBlockGetArgument(helper_body, 1);
    const balances_before = mlir.oraSLoadOpCreate(mlir_ctx, loc, stringRef("balances"), map_ty);
    const balances_after = mlir.oraMapStoreOpCreate(
        mlir_ctx,
        loc,
        mlir.oraOperationGetResult(balances_before, 0),
        user_arg,
        value_arg,
    );
    const ret = mlir.oraReturnOpCreate(mlir_ctx, loc, &[_]mlir.MlirValue{}, 0);
    mlir.oraBlockAppendOwnedOperation(helper_body, balances_before);
    mlir.oraBlockAppendOwnedOperation(helper_body, balances_after);
    mlir.oraBlockAppendOwnedOperation(helper_body, ret);
    try encoder.registerFunctionOperation(helper);

    const user_op = mlir.oraArithConstantOpCreate(mlir_ctx, loc, i256_ty, mlir.oraIntegerAttrCreateI64FromType(i256_ty, 11));
    const other_op = mlir.oraArithConstantOpCreate(mlir_ctx, loc, i256_ty, mlir.oraIntegerAttrCreateI64FromType(i256_ty, 22));
    const value_op = mlir.oraArithConstantOpCreate(mlir_ctx, loc, i256_ty, mlir.oraIntegerAttrCreateI64FromType(i256_ty, 99));
    const user = mlir.oraOperationGetResult(user_op, 0);
    const other = mlir.oraOperationGetResult(other_op, 0);
    const value = mlir.oraOperationGetResult(value_op, 0);

    const pre_load = mlir.oraSLoadOpCreate(mlir_ctx, loc, stringRef("balances"), map_ty);
    const pre_get = mlir.oraMapGetOpCreate(mlir_ctx, loc, mlir.oraOperationGetResult(pre_load, 0), other, i256_ty);
    const pre_other = try encoder.encodeOperation(pre_get);
    const pre_user_get = mlir.oraMapGetOpCreate(mlir_ctx, loc, mlir.oraOperationGetResult(pre_load, 0), user, i256_ty);
    const pre_user = try encoder.encodeOperation(pre_user_get);

    const call = mlir.oraFuncCallOpCreate(
        mlir_ctx,
        loc,
        stringRef("opaqueSetBalance"),
        &[_]mlir.MlirValue{ user, value },
        2,
        &[_]mlir.MlirType{},
        0,
    );
    mlir.oraOperationSetAttributeByName(call, stringRef("ora.imported_call"), mlir.oraBoolAttrCreate(mlir_ctx, true));
    _ = try encoder.encodeOperation(call);

    try testing.expect(!encoder.isDegraded());
    try testing.expectEqual(@as(usize, 0), encoder.soundnessLosses().len);
    try testing.expectEqual(@as(usize, 0), encoder.precisionNotes().len);

    const post_load = mlir.oraSLoadOpCreate(mlir_ctx, loc, stringRef("balances"), map_ty);
    const post_get = mlir.oraMapGetOpCreate(mlir_ctx, loc, mlir.oraOperationGetResult(post_load, 0), other, i256_ty);
    const post_other = try encoder.encodeOperation(post_get);
    const post_user_get = mlir.oraMapGetOpCreate(mlir_ctx, loc, mlir.oraOperationGetResult(post_load, 0), user, i256_ty);
    const post_user = try encoder.encodeOperation(post_user_get);

    var preserved_solver = try Solver.init(&z3_ctx, testing.allocator);
    defer preserved_solver.deinit();
    preserved_solver.assert(z3.Z3_mk_not(z3_ctx.ctx, z3.Z3_mk_eq(z3_ctx.ctx, pre_other, post_other)));
    try testing.expectEqual(@as(z3.Z3_lbool, z3.Z3_L_FALSE), preserved_solver.check());

    var changed_solver = try Solver.init(&z3_ctx, testing.allocator);
    defer changed_solver.deinit();
    changed_solver.assert(z3.Z3_mk_not(z3_ctx.ctx, z3.Z3_mk_eq(z3_ctx.ctx, pre_user, post_user)));
    try testing.expectEqual(@as(z3.Z3_lbool, z3.Z3_L_TRUE), changed_solver.check());
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

test "map key widening uses typed exact coercion when signedness is known" {
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

    try testing.expect(!encoder.isDegraded());

    const expected = try encoder.encodeValue(value);
    var solver = try Solver.init(&z3_ctx, testing.allocator);
    defer solver.deinit();
    const neq = z3.Z3_mk_not(z3_ctx.ctx, z3.Z3_mk_eq(z3_ctx.ctx, loaded, expected));
    solver.assert(neq);
    try testing.expectEqual(@as(z3.Z3_lbool, z3.Z3_L_FALSE), solver.check());
}

test "production encoder quantifiers are created only through mkQuantifier" {
    const source = @embedFile("encoder.zig");
    const needle_forall = "Z3_mk_forall_const";
    const needle_exists = "Z3_mk_exists_const";

    try testing.expectEqual(@as(usize, 1), std.mem.count(u8, source, needle_forall));
    try testing.expectEqual(@as(usize, 1), std.mem.count(u8, source, needle_exists));

    const forall_index = std.mem.indexOf(u8, source, needle_forall).?;
    const exists_index = std.mem.indexOf(u8, source, needle_exists).?;
    const helper_index = std.mem.indexOf(u8, source, "fn mkQuantifier").?;
    const helper_end = std.mem.indexOfPos(u8, source, helper_index, "// SENTINEL: end_of_mkQuantifier_helpers") orelse {
        std.debug.print("encoder.zig sentinel moved or reworded; update the raw-quantifier regression test\n", .{});
        return error.TestUnexpectedResult;
    };

    try testing.expect(forall_index > helper_index and forall_index < helper_end);
    try testing.expect(exists_index > helper_index and exists_index < helper_end);
}

test "array-store frame quantifier is reserved for opaque indexed modifies fallback" {
    const source = @embedFile("encoder.zig");
    const helper_needle = "fn addArrayStoreFrameConstraint";
    const call_needle = "self.addArrayStoreFrameConstraint(";

    try testing.expectEqual(@as(usize, 1), std.mem.count(u8, source, helper_needle));
    try testing.expectEqual(@as(usize, 0), std.mem.count(u8, source, "pub fn addArrayStoreFrameConstraint"));
    try testing.expectEqual(@as(usize, 1), std.mem.count(u8, source, call_needle));

    const call_index = std.mem.indexOf(u8, source, call_needle).?;
    const opaque_index = std.mem.indexOf(u8, source, "fn encodeOpaqueIndexedPathStore").?;
    const opaque_end = std.mem.indexOfPos(u8, source, opaque_index, "// SENTINEL: end_of_encodeOpaqueIndexedPathStore") orelse {
        std.debug.print("encoder.zig opaque indexed modifies sentinel moved; update the array-store frame quantifier guardrail\n", .{});
        return error.TestUnexpectedResult;
    };

    try testing.expect(call_index > opaque_index and call_index < opaque_end);
}
