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
    const constraints = try encoder.takeConstraints(testing.allocator);
    defer if (constraints.len > 0) testing.allocator.free(constraints);
    try expectNoQuantifiedConstraints(&z3_ctx, constraints);

    const counter = encoder.global_map.get("counter").?;
    const expected = z3.Z3_mk_numeral(z3_ctx.ctx, "7", z3.Z3_mk_bv_sort(z3_ctx.ctx, 256));

    var solver = try Solver.init(&z3_ctx, testing.allocator);
    defer solver.deinit();
    for (constraints) |cst| solver.assert(cst);
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

test "func.call summary with canonical inclusive symbolic no-write scf.while preserves state exactly" {
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
        namedAttr(mlir_ctx, "sym_name", mlir.oraStringAttrCreate(mlir_ctx, stringRef("symbolicNoWriteWhileInclusive"))),
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

    const cmp_op = mlir.oraArithCmpIOpCreate(mlir_ctx, loc, 7, before_arg, ub); // ule
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
    const pre_counter = z3.Z3_mk_const(z3_ctx.ctx, z3.Z3_mk_string_symbol(z3_ctx.ctx, "pre_counter_symbolic_inclusive_while"), counter_sort);
    try encoder.global_map.put(try testing.allocator.dupe(u8, "counter"), pre_counter);

    const symbolic_ub = mlir.oraVariablePlaceholderOpCreate(mlir_ctx, loc, stringRef("ubWhileInclusiveValue"), index_ty);
    const call = mlir.oraFuncCallOpCreate(
        mlir_ctx,
        loc,
        stringRef("symbolicNoWriteWhileInclusive"),
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

test "func.call summary with zero-result symbolic no-write scf.while preserves exact state modeling" {
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

    try testing.expect(!encoder.isDegraded());
}

test "func.call summary with noncanonical-result symbolic no-write scf.while preserves exact state modeling" {
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
        namedAttr(mlir_ctx, "sym_name", mlir.oraStringAttrCreate(mlir_ctx, stringRef("symbolicNoWriteNoncanonicalResultWhile"))),
    };
    const helper_param_types = [_]mlir.MlirType{i1_ty};
    const helper_param_locs = [_]mlir.MlirLocation{loc};
    const helper = mlir.oraFuncFuncOpCreate(mlir_ctx, loc, &helper_attrs, helper_attrs.len, &helper_param_types, &helper_param_locs, helper_param_types.len);
    const body = mlir.oraFuncOpGetBodyBlock(helper);

    const cond = mlir.oraBlockGetArgument(body, 0);
    const init_attr = mlir.oraIntegerAttrCreateI64FromType(i256_ty, 7);
    const init_op = mlir.oraArithConstantOpCreate(mlir_ctx, loc, i256_ty, init_attr);
    mlir.oraBlockAppendOwnedOperation(body, init_op);
    const init = mlir.oraOperationGetResult(init_op, 0);

    const while_op = mlir.oraScfWhileOpCreate(mlir_ctx, loc, &[_]mlir.MlirValue{init}, 1, &[_]mlir.MlirType{i256_ty}, 1);
    const before_block = mlir.oraScfWhileOpGetBeforeBlock(while_op);
    const after_block = mlir.oraScfWhileOpGetAfterBlock(while_op);
    _ = mlir.mlirBlockAddArgument(before_block, i256_ty, loc);
    _ = mlir.mlirBlockAddArgument(after_block, i256_ty, loc);
    const before_arg = mlir.oraBlockGetArgument(before_block, 0);
    const after_arg = mlir.oraBlockGetArgument(after_block, 0);

    mlir.oraBlockAppendOwnedOperation(before_block, mlir.oraScfConditionOpCreate(
        mlir_ctx,
        loc,
        cond,
        &[_]mlir.MlirValue{before_arg},
        1,
    ));

    // Use a non-canonical carried result update so result extraction remains
    // inexact while the state summary should still remain exact.
    const mul_op = mlir.oraArithMulIOpCreate(mlir_ctx, loc, after_arg, after_arg);
    mlir.oraBlockAppendOwnedOperation(after_block, mul_op);
    mlir.oraBlockAppendOwnedOperation(after_block, mlir.oraScfYieldOpCreate(
        mlir_ctx,
        loc,
        &[_]mlir.MlirValue{mlir.oraOperationGetResult(mul_op, 0)},
        1,
    ));

    mlir.oraBlockAppendOwnedOperation(body, while_op);
    mlir.oraBlockAppendOwnedOperation(body, mlir.oraReturnOpCreate(mlir_ctx, loc, &[_]mlir.MlirValue{}, 0));

    try encoder.registerFunctionOperation(helper);

    const counter_sort = z3.Z3_mk_bv_sort(z3_ctx.ctx, 256);
    const pre_counter = z3.Z3_mk_const(z3_ctx.ctx, z3.Z3_mk_string_symbol(z3_ctx.ctx, "pre_counter_symbolic_noncanonical_while_state"), counter_sort);
    try encoder.global_map.put(try testing.allocator.dupe(u8, "counter"), pre_counter);

    const call_cond = mlir.oraVariablePlaceholderOpCreate(mlir_ctx, loc, stringRef("noncanonicalWhileCond"), i1_ty);
    const call = mlir.oraFuncCallOpCreate(
        mlir_ctx,
        loc,
        stringRef("symbolicNoWriteNoncanonicalResultWhile"),
        &[_]mlir.MlirValue{mlir.oraOperationGetResult(call_cond, 0)},
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

    const outer_maybe = mlir.oraVariablePlaceholderOpCreate(mlir_ctx, loc, stringRef("maybeValue"), eu_ty);
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

    const outer_maybe = mlir.oraVariablePlaceholderOpCreate(mlir_ctx, loc, stringRef("maybeValue"), eu_ty);
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

    const outer_maybe = mlir.oraVariablePlaceholderOpCreate(mlir_ctx, loc, stringRef("maybeValue"), eu_ty);
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
    const outer_maybe = mlir.oraVariablePlaceholderOpCreate(mlir_ctx, loc, stringRef("maybeValue"), eu_ty);
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
    const outer_maybe = mlir.oraVariablePlaceholderOpCreate(mlir_ctx, loc, stringRef("maybeValue"), eu_ty);
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
    const outer_maybe = mlir.oraVariablePlaceholderOpCreate(mlir_ctx, loc, stringRef("maybeValue"), eu_ty);
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

test "func.call summary with switch ora.try_stmt state effects encodes exactly" {
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
        namedAttr(mlir_ctx, "sym_name", mlir.oraStringAttrCreate(mlir_ctx, stringRef("switchTryWriter"))),
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

    const switch_op = mlir.oraSwitchOpCreateWithCases(mlir_ctx, loc, helper_flag, &[_]mlir.MlirType{}, 0, 2);
    const false_block = mlir.oraSwitchOpGetCaseBlock(switch_op, 0);
    const true_block = mlir.oraSwitchOpGetCaseBlock(switch_op, 1);
    var case_values = [_]i64{ 0, 1 };
    var range_starts = [_]i64{ 0, 0 };
    var range_ends = [_]i64{ 0, 0 };
    var case_kinds = [_]i64{ 0, 0 };
    mlir.oraSwitchOpSetCasePatterns(
        switch_op,
        &case_values,
        &range_starts,
        &range_ends,
        &case_kinds,
        -1,
        case_values.len,
    );

    mlir.oraBlockAppendOwnedOperation(false_block, mlir.oraYieldOpCreate(mlir_ctx, loc, &[_]mlir.MlirValue{}, 0));

    const unwrap_op = mlir.oraErrorUnwrapOpCreate(mlir_ctx, loc, helper_maybe, i256_ty);
    mlir.oraBlockAppendOwnedOperation(true_block, unwrap_op);
    mlir.oraBlockAppendOwnedOperation(true_block, mlir.oraYieldOpCreate(mlir_ctx, loc, &[_]mlir.MlirValue{}, 0));

    const try_store_attr = mlir.oraIntegerAttrCreateI64FromType(i256_ty, 11);
    const try_store_op = mlir.oraArithConstantOpCreate(mlir_ctx, loc, i256_ty, try_store_attr);
    mlir.oraBlockAppendOwnedOperation(try_block, switch_op);
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
    const outer_maybe = mlir.oraVariablePlaceholderOpCreate(mlir_ctx, loc, stringRef("maybeValue"), eu_ty);
    const helper_call = mlir.oraFuncCallOpCreate(
        mlir_ctx,
        loc,
        stringRef("switchTryWriter"),
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

test "func.call summary with fixed-iteration symbolic scf.for ora.try_stmt encodes differing slots exactly" {
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
        namedAttr(mlir_ctx, "sym_name", mlir.oraStringAttrCreate(mlir_ctx, stringRef("unresolvedTryWriter"))),
        namedAttr(mlir_ctx, "ora.effect", mlir.oraStringAttrCreate(mlir_ctx, stringRef("writes"))),
        namedAttr(mlir_ctx, "ora.write_slots", mlir.oraArrayAttrCreate(mlir_ctx, 1, &[_]mlir.MlirAttribute{
            mlir.oraStringAttrCreate(mlir_ctx, stringRef("counter")),
        })),
    };
    const helper_param_types = [_]mlir.MlirType{ index_ty, eu_ty };
    const helper_param_locs = [_]mlir.MlirLocation{ loc, loc };
    const helper = mlir.oraFuncFuncOpCreate(mlir_ctx, loc, &helper_attrs, helper_attrs.len, &helper_param_types, &helper_param_locs, helper_param_types.len);
    const body = mlir.oraFuncOpGetBodyBlock(helper);

    const limit_arg = mlir.oraBlockGetArgument(body, 0);
    const maybe_arg = mlir.oraBlockGetArgument(body, 1);

    const zero_idx_attr = mlir.oraIntegerAttrCreateI64FromType(index_ty, 0);
    const one_idx_attr = mlir.oraIntegerAttrCreateI64FromType(index_ty, 1);
    const zero_idx_op = mlir.oraArithConstantOpCreate(mlir_ctx, loc, index_ty, zero_idx_attr);
    const one_idx_op = mlir.oraArithConstantOpCreate(mlir_ctx, loc, index_ty, one_idx_attr);
    mlir.oraBlockAppendOwnedOperation(body, zero_idx_op);
    mlir.oraBlockAppendOwnedOperation(body, one_idx_op);

    const try_stmt = mlir.oraTryStmtOpCreate(mlir_ctx, loc, &[_]mlir.MlirType{}, 0);
    const try_block = mlir.oraTryStmtOpGetTryBlock(try_stmt);
    const catch_block = mlir.oraTryStmtOpGetCatchBlock(try_stmt);

    const for_op = mlir.oraScfForOpCreate(
        mlir_ctx,
        loc,
        mlir.oraOperationGetResult(zero_idx_op, 0),
        limit_arg,
        mlir.oraOperationGetResult(one_idx_op, 0),
        &[_]mlir.MlirValue{},
        0,
        false,
    );
    const for_body = mlir.oraScfForOpGetBodyBlock(for_op);
    const iv = mlir.oraBlockGetArgument(for_body, 0);
    const cmp_op = mlir.oraArithCmpIOpCreate(mlir_ctx, loc, 0, iv, mlir.oraOperationGetResult(one_idx_op, 0));
    mlir.oraBlockAppendOwnedOperation(for_body, cmp_op);
    const no_results = [_]mlir.MlirType{};
    const if_op = mlir.oraScfIfOpCreate(mlir_ctx, loc, mlir.oraOperationGetResult(cmp_op, 0), &no_results, no_results.len, true);
    const then_block = mlir.oraScfIfOpGetThenBlock(if_op);
    const else_block = mlir.oraScfIfOpGetElseBlock(if_op);
    const unwrap_op = mlir.oraErrorUnwrapOpCreate(mlir_ctx, loc, maybe_arg, i256_ty);
    mlir.oraBlockAppendOwnedOperation(then_block, unwrap_op);
    mlir.oraBlockAppendOwnedOperation(then_block, mlir.oraScfYieldOpCreate(mlir_ctx, loc, &[_]mlir.MlirValue{}, 0));
    mlir.oraBlockAppendOwnedOperation(else_block, mlir.oraScfYieldOpCreate(mlir_ctx, loc, &[_]mlir.MlirValue{}, 0));
    mlir.oraBlockAppendOwnedOperation(for_body, if_op);
    mlir.oraBlockAppendOwnedOperation(for_body, mlir.oraScfYieldOpCreate(mlir_ctx, loc, &[_]mlir.MlirValue{}, 0));

    mlir.oraBlockAppendOwnedOperation(try_block, for_op);
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

    const counter_pre = try encoder.getOrCreateCurrentGlobal("counter", z3.Z3_mk_bv_sort(z3_ctx.ctx, 256));
    const stable_pre = try encoder.getOrCreateCurrentGlobal("stable", z3.Z3_mk_bv_sort(z3_ctx.ctx, 256));

    const limit = mlir.oraVariablePlaceholderOpCreate(mlir_ctx, loc, stringRef("loopLimit"), index_ty);
    const maybe = mlir.oraVariablePlaceholderOpCreate(mlir_ctx, loc, stringRef("maybeValue"), eu_ty);
    const call = mlir.oraFuncCallOpCreate(
        mlir_ctx,
        loc,
        stringRef("unresolvedTryWriter"),
        &[_]mlir.MlirValue{
            mlir.oraOperationGetResult(limit, 0),
            mlir.oraOperationGetResult(maybe, 0),
        },
        2,
        &[_]mlir.MlirType{},
        0,
    );
    _ = try encoder.encodeOperation(call);

    try testing.expect(!encoder.isDegraded());
    const stable = encoder.global_map.get("stable").?;
    var solver = try Solver.init(&z3_ctx, testing.allocator);
    defer solver.deinit();
    solver.assert(z3.Z3_mk_not(z3_ctx.ctx, z3.Z3_mk_eq(z3_ctx.ctx, stable, stable_pre)));
    try testing.expectEqual(@as(z3.Z3_lbool, z3.Z3_L_FALSE), solver.check());

    const counter = encoder.global_map.get("counter").?;
    const counter_text = std.mem.span(z3.Z3_ast_to_string(z3_ctx.ctx, counter));
    _ = counter_pre;
    try testing.expect(std.mem.indexOf(u8, counter_text, "undef_try_state_global") == null);
}

test "func.call summary with fixed-iteration symbolic-target scf.for ora.try_stmt encodes differing slots exactly" {
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
        namedAttr(mlir_ctx, "sym_name", mlir.oraStringAttrCreate(mlir_ctx, stringRef("symbolicTargetForTryWriter"))),
        namedAttr(mlir_ctx, "ora.effect", mlir.oraStringAttrCreate(mlir_ctx, stringRef("writes"))),
        namedAttr(mlir_ctx, "ora.write_slots", mlir.oraArrayAttrCreate(mlir_ctx, 1, &[_]mlir.MlirAttribute{
            mlir.oraStringAttrCreate(mlir_ctx, stringRef("counter")),
        })),
    };
    const helper_param_types = [_]mlir.MlirType{ index_ty, index_ty, eu_ty };
    const helper_param_locs = [_]mlir.MlirLocation{ loc, loc, loc };
    const helper = mlir.oraFuncFuncOpCreate(mlir_ctx, loc, &helper_attrs, helper_attrs.len, &helper_param_types, &helper_param_locs, helper_param_types.len);
    const body = mlir.oraFuncOpGetBodyBlock(helper);

    const limit_arg = mlir.oraBlockGetArgument(body, 0);
    const target_arg = mlir.oraBlockGetArgument(body, 1);
    const maybe_arg = mlir.oraBlockGetArgument(body, 2);

    const zero_idx_op = mlir.oraArithConstantOpCreate(
        mlir_ctx,
        loc,
        index_ty,
        mlir.oraIntegerAttrCreateI64FromType(index_ty, 0),
    );
    const one_idx_op = mlir.oraArithConstantOpCreate(
        mlir_ctx,
        loc,
        index_ty,
        mlir.oraIntegerAttrCreateI64FromType(index_ty, 1),
    );
    mlir.oraBlockAppendOwnedOperation(body, zero_idx_op);
    mlir.oraBlockAppendOwnedOperation(body, one_idx_op);

    const try_stmt = mlir.oraTryStmtOpCreate(mlir_ctx, loc, &[_]mlir.MlirType{}, 0);
    const try_block = mlir.oraTryStmtOpGetTryBlock(try_stmt);
    const catch_block = mlir.oraTryStmtOpGetCatchBlock(try_stmt);

    const for_op = mlir.oraScfForOpCreate(
        mlir_ctx,
        loc,
        mlir.oraOperationGetResult(zero_idx_op, 0),
        limit_arg,
        mlir.oraOperationGetResult(one_idx_op, 0),
        &[_]mlir.MlirValue{},
        0,
        false,
    );
    const for_body = mlir.oraScfForOpGetBodyBlock(for_op);
    const iv = mlir.oraBlockGetArgument(for_body, 0);
    const cmp_op = mlir.oraArithCmpIOpCreate(mlir_ctx, loc, 0, iv, target_arg);
    mlir.oraBlockAppendOwnedOperation(for_body, cmp_op);
    const no_results = [_]mlir.MlirType{};
    const if_op = mlir.oraScfIfOpCreate(mlir_ctx, loc, mlir.oraOperationGetResult(cmp_op, 0), &no_results, no_results.len, true);
    const then_block = mlir.oraScfIfOpGetThenBlock(if_op);
    const else_block = mlir.oraScfIfOpGetElseBlock(if_op);
    const unwrap_op = mlir.oraErrorUnwrapOpCreate(mlir_ctx, loc, maybe_arg, i256_ty);
    mlir.oraBlockAppendOwnedOperation(then_block, unwrap_op);
    mlir.oraBlockAppendOwnedOperation(then_block, mlir.oraScfYieldOpCreate(mlir_ctx, loc, &[_]mlir.MlirValue{}, 0));
    mlir.oraBlockAppendOwnedOperation(else_block, mlir.oraScfYieldOpCreate(mlir_ctx, loc, &[_]mlir.MlirValue{}, 0));
    mlir.oraBlockAppendOwnedOperation(for_body, if_op);
    mlir.oraBlockAppendOwnedOperation(for_body, mlir.oraScfYieldOpCreate(mlir_ctx, loc, &[_]mlir.MlirValue{}, 0));

    mlir.oraBlockAppendOwnedOperation(try_block, for_op);
    const try_store_op = mlir.oraArithConstantOpCreate(
        mlir_ctx,
        loc,
        i256_ty,
        mlir.oraIntegerAttrCreateI64FromType(i256_ty, 7),
    );
    mlir.oraBlockAppendOwnedOperation(try_block, try_store_op);
    mlir.oraBlockAppendOwnedOperation(try_block, mlir.oraSStoreOpCreate(
        mlir_ctx,
        loc,
        mlir.oraOperationGetResult(try_store_op, 0),
        stringRef("counter"),
    ));
    mlir.oraBlockAppendOwnedOperation(try_block, mlir.oraYieldOpCreate(mlir_ctx, loc, &[_]mlir.MlirValue{}, 0));

    const catch_store_op = mlir.oraArithConstantOpCreate(
        mlir_ctx,
        loc,
        i256_ty,
        mlir.oraIntegerAttrCreateI64FromType(i256_ty, 33),
    );
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

    const stable_pre = try encoder.getOrCreateCurrentGlobal("stable", z3.Z3_mk_bv_sort(z3_ctx.ctx, 256));

    const limit = mlir.oraVariablePlaceholderOpCreate(mlir_ctx, loc, stringRef("symbolicTargetLoopLimit"), index_ty);
    const target = mlir.oraVariablePlaceholderOpCreate(mlir_ctx, loc, stringRef("symbolicTargetLoopTarget"), index_ty);
    const maybe = mlir.oraVariablePlaceholderOpCreate(mlir_ctx, loc, stringRef("symbolicTargetMaybe"), eu_ty);
    const call = mlir.oraFuncCallOpCreate(
        mlir_ctx,
        loc,
        stringRef("symbolicTargetForTryWriter"),
        &[_]mlir.MlirValue{
            mlir.oraOperationGetResult(limit, 0),
            mlir.oraOperationGetResult(target, 0),
            mlir.oraOperationGetResult(maybe, 0),
        },
        3,
        &[_]mlir.MlirType{},
        0,
    );
    _ = try encoder.encodeOperation(call);

    try testing.expect(!encoder.isDegraded());

    const stable = encoder.global_map.get("stable").?;
    var stable_solver = try Solver.init(&z3_ctx, testing.allocator);
    defer stable_solver.deinit();
    stable_solver.assert(z3.Z3_mk_not(z3_ctx.ctx, z3.Z3_mk_eq(z3_ctx.ctx, stable, stable_pre)));
    try testing.expectEqual(@as(z3.Z3_lbool, z3.Z3_L_FALSE), stable_solver.check());

    const counter = encoder.global_map.get("counter").?;
    const limit_ast = try encoder.encodeValue(mlir.oraOperationGetResult(limit, 0));
    const target_ast = try encoder.encodeValue(mlir.oraOperationGetResult(target, 0));
    const is_error_op = mlir.oraErrorIsErrorOpCreate(mlir_ctx, loc, mlir.oraOperationGetResult(maybe, 0));
    const is_error = try encoder.encodeOperation(is_error_op);
    const seven = try encoder.encodeIntegerConstant(7, 256);
    const thirty_three = try encoder.encodeIntegerConstant(33, 256);
    const reaches_target = z3.Z3_mk_bvsgt(z3_ctx.ctx, limit_ast, target_ast);
    const catches = encoder.encodeAnd(&.{ reaches_target, encoder.coerceBoolean(is_error) });
    const expected = z3.Z3_mk_ite(z3_ctx.ctx, catches, thirty_three, seven);

    var counter_solver = try Solver.init(&z3_ctx, testing.allocator);
    defer counter_solver.deinit();
    counter_solver.assert(z3.Z3_mk_not(z3_ctx.ctx, z3.Z3_mk_eq(z3_ctx.ctx, counter, expected)));
    try testing.expectEqual(@as(z3.Z3_lbool, z3.Z3_L_FALSE), counter_solver.check());
}

test "func.call summary with multi-target symbolic scf.for ora.try_stmt encodes differing slots exactly" {
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
        namedAttr(mlir_ctx, "sym_name", mlir.oraStringAttrCreate(mlir_ctx, stringRef("multiTargetForTryWriter"))),
        namedAttr(mlir_ctx, "ora.effect", mlir.oraStringAttrCreate(mlir_ctx, stringRef("writes"))),
        namedAttr(mlir_ctx, "ora.write_slots", mlir.oraArrayAttrCreate(mlir_ctx, 1, &[_]mlir.MlirAttribute{
            mlir.oraStringAttrCreate(mlir_ctx, stringRef("counter")),
        })),
    };
    const helper_param_types = [_]mlir.MlirType{ index_ty, index_ty, index_ty, eu_ty };
    const helper_param_locs = [_]mlir.MlirLocation{ loc, loc, loc, loc };
    const helper = mlir.oraFuncFuncOpCreate(mlir_ctx, loc, &helper_attrs, helper_attrs.len, &helper_param_types, &helper_param_locs, helper_param_types.len);
    const body = mlir.oraFuncOpGetBodyBlock(helper);

    const limit_arg = mlir.oraBlockGetArgument(body, 0);
    const target_a_arg = mlir.oraBlockGetArgument(body, 1);
    const target_b_arg = mlir.oraBlockGetArgument(body, 2);
    const maybe_arg = mlir.oraBlockGetArgument(body, 3);

    const zero_idx_op = mlir.oraArithConstantOpCreate(
        mlir_ctx,
        loc,
        index_ty,
        mlir.oraIntegerAttrCreateI64FromType(index_ty, 0),
    );
    const one_idx_op = mlir.oraArithConstantOpCreate(
        mlir_ctx,
        loc,
        index_ty,
        mlir.oraIntegerAttrCreateI64FromType(index_ty, 1),
    );
    mlir.oraBlockAppendOwnedOperation(body, zero_idx_op);
    mlir.oraBlockAppendOwnedOperation(body, one_idx_op);

    const try_stmt = mlir.oraTryStmtOpCreate(mlir_ctx, loc, &[_]mlir.MlirType{}, 0);
    const try_block = mlir.oraTryStmtOpGetTryBlock(try_stmt);
    const catch_block = mlir.oraTryStmtOpGetCatchBlock(try_stmt);

    const for_op = mlir.oraScfForOpCreate(
        mlir_ctx,
        loc,
        mlir.oraOperationGetResult(zero_idx_op, 0),
        limit_arg,
        mlir.oraOperationGetResult(one_idx_op, 0),
        &[_]mlir.MlirValue{},
        0,
        false,
    );
    const for_body = mlir.oraScfForOpGetBodyBlock(for_op);
    const iv = mlir.oraBlockGetArgument(for_body, 0);
    const cmp_a = mlir.oraArithCmpIOpCreate(mlir_ctx, loc, 0, iv, target_a_arg);
    mlir.oraBlockAppendOwnedOperation(for_body, cmp_a);
    const no_results = [_]mlir.MlirType{};
    const if_a = mlir.oraScfIfOpCreate(mlir_ctx, loc, mlir.oraOperationGetResult(cmp_a, 0), &no_results, no_results.len, true);
    const then_a = mlir.oraScfIfOpGetThenBlock(if_a);
    const else_a = mlir.oraScfIfOpGetElseBlock(if_a);
    const unwrap_a = mlir.oraErrorUnwrapOpCreate(mlir_ctx, loc, maybe_arg, i256_ty);
    mlir.oraBlockAppendOwnedOperation(then_a, unwrap_a);
    mlir.oraBlockAppendOwnedOperation(then_a, mlir.oraScfYieldOpCreate(mlir_ctx, loc, &[_]mlir.MlirValue{}, 0));

    const cmp_b = mlir.oraArithCmpIOpCreate(mlir_ctx, loc, 0, iv, target_b_arg);
    mlir.oraBlockAppendOwnedOperation(else_a, cmp_b);
    const if_b = mlir.oraScfIfOpCreate(mlir_ctx, loc, mlir.oraOperationGetResult(cmp_b, 0), &no_results, no_results.len, true);
    const then_b = mlir.oraScfIfOpGetThenBlock(if_b);
    const else_b = mlir.oraScfIfOpGetElseBlock(if_b);
    const unwrap_b = mlir.oraErrorUnwrapOpCreate(mlir_ctx, loc, maybe_arg, i256_ty);
    mlir.oraBlockAppendOwnedOperation(then_b, unwrap_b);
    mlir.oraBlockAppendOwnedOperation(then_b, mlir.oraScfYieldOpCreate(mlir_ctx, loc, &[_]mlir.MlirValue{}, 0));
    mlir.oraBlockAppendOwnedOperation(else_b, mlir.oraScfYieldOpCreate(mlir_ctx, loc, &[_]mlir.MlirValue{}, 0));
    mlir.oraBlockAppendOwnedOperation(else_a, if_b);
    mlir.oraBlockAppendOwnedOperation(else_a, mlir.oraScfYieldOpCreate(mlir_ctx, loc, &[_]mlir.MlirValue{}, 0));

    mlir.oraBlockAppendOwnedOperation(for_body, if_a);
    mlir.oraBlockAppendOwnedOperation(for_body, mlir.oraScfYieldOpCreate(mlir_ctx, loc, &[_]mlir.MlirValue{}, 0));

    mlir.oraBlockAppendOwnedOperation(try_block, for_op);
    const try_store_op = mlir.oraArithConstantOpCreate(
        mlir_ctx,
        loc,
        i256_ty,
        mlir.oraIntegerAttrCreateI64FromType(i256_ty, 7),
    );
    mlir.oraBlockAppendOwnedOperation(try_block, try_store_op);
    mlir.oraBlockAppendOwnedOperation(try_block, mlir.oraSStoreOpCreate(
        mlir_ctx,
        loc,
        mlir.oraOperationGetResult(try_store_op, 0),
        stringRef("counter"),
    ));
    mlir.oraBlockAppendOwnedOperation(try_block, mlir.oraYieldOpCreate(mlir_ctx, loc, &[_]mlir.MlirValue{}, 0));

    const catch_store_op = mlir.oraArithConstantOpCreate(
        mlir_ctx,
        loc,
        i256_ty,
        mlir.oraIntegerAttrCreateI64FromType(i256_ty, 33),
    );
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

    const stable_pre = try encoder.getOrCreateCurrentGlobal("stable", z3.Z3_mk_bv_sort(z3_ctx.ctx, 256));

    const limit = mlir.oraVariablePlaceholderOpCreate(mlir_ctx, loc, stringRef("multiTargetLoopLimit"), index_ty);
    const target_a = mlir.oraVariablePlaceholderOpCreate(mlir_ctx, loc, stringRef("multiTargetLoopA"), index_ty);
    const target_b = mlir.oraVariablePlaceholderOpCreate(mlir_ctx, loc, stringRef("multiTargetLoopB"), index_ty);
    const maybe = mlir.oraVariablePlaceholderOpCreate(mlir_ctx, loc, stringRef("multiTargetLoopMaybe"), eu_ty);
    const call = mlir.oraFuncCallOpCreate(
        mlir_ctx,
        loc,
        stringRef("multiTargetForTryWriter"),
        &[_]mlir.MlirValue{
            mlir.oraOperationGetResult(limit, 0),
            mlir.oraOperationGetResult(target_a, 0),
            mlir.oraOperationGetResult(target_b, 0),
            mlir.oraOperationGetResult(maybe, 0),
        },
        4,
        &[_]mlir.MlirType{},
        0,
    );
    _ = try encoder.encodeOperation(call);
    try testing.expect(!encoder.isDegraded());

    const stable = encoder.global_map.get("stable").?;
    var stable_solver = try Solver.init(&z3_ctx, testing.allocator);
    defer stable_solver.deinit();
    stable_solver.assert(z3.Z3_mk_not(z3_ctx.ctx, z3.Z3_mk_eq(z3_ctx.ctx, stable, stable_pre)));
    try testing.expectEqual(@as(z3.Z3_lbool, z3.Z3_L_FALSE), stable_solver.check());

    const counter = encoder.global_map.get("counter").?;
    const limit_ast = try encoder.encodeValue(mlir.oraOperationGetResult(limit, 0));
    const target_a_ast = try encoder.encodeValue(mlir.oraOperationGetResult(target_a, 0));
    const target_b_ast = try encoder.encodeValue(mlir.oraOperationGetResult(target_b, 0));
    const is_error_op = mlir.oraErrorIsErrorOpCreate(mlir_ctx, loc, mlir.oraOperationGetResult(maybe, 0));
    const is_error = try encoder.encodeOperation(is_error_op);
    const seven = try encoder.encodeIntegerConstant(7, 256);
    const thirty_three = try encoder.encodeIntegerConstant(33, 256);
    const reaches_a = z3.Z3_mk_bvsgt(z3_ctx.ctx, limit_ast, target_a_ast);
    const reaches_b = z3.Z3_mk_bvsgt(z3_ctx.ctx, limit_ast, target_b_ast);
    const catches = encoder.encodeAnd(&.{ encoder.encodeOr(&.{ reaches_a, reaches_b }), encoder.coerceBoolean(is_error) });
    const expected = z3.Z3_mk_ite(z3_ctx.ctx, catches, thirty_three, seven);

    var counter_solver = try Solver.init(&z3_ctx, testing.allocator);
    defer counter_solver.deinit();
    counter_solver.assert(z3.Z3_mk_not(z3_ctx.ctx, z3.Z3_mk_eq(z3_ctx.ctx, counter, expected)));
    try testing.expectEqual(@as(z3.Z3_lbool, z3.Z3_L_FALSE), counter_solver.check());
}

test "func.call summary with multi-target symbolic scf.while ora.try_stmt encodes differing slots exactly" {
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
        namedAttr(mlir_ctx, "sym_name", mlir.oraStringAttrCreate(mlir_ctx, stringRef("multiTargetWhileTryWriter"))),
        namedAttr(mlir_ctx, "ora.effect", mlir.oraStringAttrCreate(mlir_ctx, stringRef("writes"))),
        namedAttr(mlir_ctx, "ora.write_slots", mlir.oraArrayAttrCreate(mlir_ctx, 1, &[_]mlir.MlirAttribute{
            mlir.oraStringAttrCreate(mlir_ctx, stringRef("counter")),
        })),
    };
    const helper_param_types = [_]mlir.MlirType{ index_ty, index_ty, index_ty, eu_ty };
    const helper_param_locs = [_]mlir.MlirLocation{ loc, loc, loc, loc };
    const helper = mlir.oraFuncFuncOpCreate(mlir_ctx, loc, &helper_attrs, helper_attrs.len, &helper_param_types, &helper_param_locs, helper_param_types.len);
    const body = mlir.oraFuncOpGetBodyBlock(helper);

    const limit_arg = mlir.oraBlockGetArgument(body, 0);
    const target_a_arg = mlir.oraBlockGetArgument(body, 1);
    const target_b_arg = mlir.oraBlockGetArgument(body, 2);
    const maybe_arg = mlir.oraBlockGetArgument(body, 3);

    const zero_idx_op = mlir.oraArithConstantOpCreate(
        mlir_ctx,
        loc,
        index_ty,
        mlir.oraIntegerAttrCreateI64FromType(index_ty, 0),
    );
    const one_idx_op = mlir.oraArithConstantOpCreate(
        mlir_ctx,
        loc,
        index_ty,
        mlir.oraIntegerAttrCreateI64FromType(index_ty, 1),
    );
    mlir.oraBlockAppendOwnedOperation(body, zero_idx_op);
    mlir.oraBlockAppendOwnedOperation(body, one_idx_op);

    const try_stmt = mlir.oraTryStmtOpCreate(mlir_ctx, loc, &[_]mlir.MlirType{}, 0);
    const try_block = mlir.oraTryStmtOpGetTryBlock(try_stmt);
    const catch_block = mlir.oraTryStmtOpGetCatchBlock(try_stmt);

    const init_vals = [_]mlir.MlirValue{mlir.oraOperationGetResult(zero_idx_op, 0)};
    const result_types = [_]mlir.MlirType{index_ty};
    const while_op = mlir.oraScfWhileOpCreate(mlir_ctx, loc, &init_vals, init_vals.len, &result_types, result_types.len);
    const before_block = mlir.oraScfWhileOpGetBeforeBlock(while_op);
    const after_block = mlir.oraScfWhileOpGetAfterBlock(while_op);
    _ = mlir.mlirBlockAddArgument(before_block, index_ty, loc);
    _ = mlir.mlirBlockAddArgument(after_block, index_ty, loc);
    const before_idx = mlir.oraBlockGetArgument(before_block, 0);
    const after_idx = mlir.oraBlockGetArgument(after_block, 0);

    const continue_cmp = mlir.oraArithCmpIOpCreate(mlir_ctx, loc, 6, before_idx, limit_arg);
    mlir.oraBlockAppendOwnedOperation(before_block, continue_cmp);
    mlir.oraBlockAppendOwnedOperation(before_block, mlir.oraScfConditionOpCreate(
        mlir_ctx,
        loc,
        mlir.oraOperationGetResult(continue_cmp, 0),
        &[_]mlir.MlirValue{before_idx},
        1,
    ));

    const cmp_a = mlir.oraArithCmpIOpCreate(mlir_ctx, loc, 0, after_idx, target_a_arg);
    mlir.oraBlockAppendOwnedOperation(after_block, cmp_a);
    const no_results = [_]mlir.MlirType{};
    const if_a = mlir.oraScfIfOpCreate(mlir_ctx, loc, mlir.oraOperationGetResult(cmp_a, 0), &no_results, no_results.len, true);
    const then_a = mlir.oraScfIfOpGetThenBlock(if_a);
    const else_a = mlir.oraScfIfOpGetElseBlock(if_a);
    const unwrap_a = mlir.oraErrorUnwrapOpCreate(mlir_ctx, loc, maybe_arg, i256_ty);
    mlir.oraBlockAppendOwnedOperation(then_a, unwrap_a);
    mlir.oraBlockAppendOwnedOperation(then_a, mlir.oraScfYieldOpCreate(mlir_ctx, loc, &[_]mlir.MlirValue{}, 0));

    const cmp_b = mlir.oraArithCmpIOpCreate(mlir_ctx, loc, 0, after_idx, target_b_arg);
    mlir.oraBlockAppendOwnedOperation(else_a, cmp_b);
    const if_b = mlir.oraScfIfOpCreate(mlir_ctx, loc, mlir.oraOperationGetResult(cmp_b, 0), &no_results, no_results.len, true);
    const then_b = mlir.oraScfIfOpGetThenBlock(if_b);
    const else_b = mlir.oraScfIfOpGetElseBlock(if_b);
    const unwrap_b = mlir.oraErrorUnwrapOpCreate(mlir_ctx, loc, maybe_arg, i256_ty);
    mlir.oraBlockAppendOwnedOperation(then_b, unwrap_b);
    mlir.oraBlockAppendOwnedOperation(then_b, mlir.oraScfYieldOpCreate(mlir_ctx, loc, &[_]mlir.MlirValue{}, 0));
    mlir.oraBlockAppendOwnedOperation(else_b, mlir.oraScfYieldOpCreate(mlir_ctx, loc, &[_]mlir.MlirValue{}, 0));
    mlir.oraBlockAppendOwnedOperation(else_a, if_b);
    mlir.oraBlockAppendOwnedOperation(else_a, mlir.oraScfYieldOpCreate(mlir_ctx, loc, &[_]mlir.MlirValue{}, 0));
    mlir.oraBlockAppendOwnedOperation(after_block, if_a);

    const next_idx = mlir.oraArithAddIOpCreate(
        mlir_ctx,
        loc,
        after_idx,
        mlir.oraOperationGetResult(one_idx_op, 0),
    );
    mlir.oraBlockAppendOwnedOperation(after_block, next_idx);
    mlir.oraBlockAppendOwnedOperation(after_block, mlir.oraScfYieldOpCreate(
        mlir_ctx,
        loc,
        &[_]mlir.MlirValue{mlir.oraOperationGetResult(next_idx, 0)},
        1,
    ));

    mlir.oraBlockAppendOwnedOperation(try_block, while_op);
    const try_store_op = mlir.oraArithConstantOpCreate(
        mlir_ctx,
        loc,
        i256_ty,
        mlir.oraIntegerAttrCreateI64FromType(i256_ty, 7),
    );
    mlir.oraBlockAppendOwnedOperation(try_block, try_store_op);
    mlir.oraBlockAppendOwnedOperation(try_block, mlir.oraSStoreOpCreate(
        mlir_ctx,
        loc,
        mlir.oraOperationGetResult(try_store_op, 0),
        stringRef("counter"),
    ));
    mlir.oraBlockAppendOwnedOperation(try_block, mlir.oraYieldOpCreate(mlir_ctx, loc, &[_]mlir.MlirValue{}, 0));

    const catch_store_op = mlir.oraArithConstantOpCreate(
        mlir_ctx,
        loc,
        i256_ty,
        mlir.oraIntegerAttrCreateI64FromType(i256_ty, 33),
    );
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

    const stable_pre = try encoder.getOrCreateCurrentGlobal("stable", z3.Z3_mk_bv_sort(z3_ctx.ctx, 256));

    const limit = mlir.oraVariablePlaceholderOpCreate(mlir_ctx, loc, stringRef("multiTargetWhileLimit"), index_ty);
    const target_a = mlir.oraVariablePlaceholderOpCreate(mlir_ctx, loc, stringRef("multiTargetWhileA"), index_ty);
    const target_b = mlir.oraVariablePlaceholderOpCreate(mlir_ctx, loc, stringRef("multiTargetWhileB"), index_ty);
    const maybe = mlir.oraVariablePlaceholderOpCreate(mlir_ctx, loc, stringRef("multiTargetWhileMaybe"), eu_ty);
    const call = mlir.oraFuncCallOpCreate(
        mlir_ctx,
        loc,
        stringRef("multiTargetWhileTryWriter"),
        &[_]mlir.MlirValue{
            mlir.oraOperationGetResult(limit, 0),
            mlir.oraOperationGetResult(target_a, 0),
            mlir.oraOperationGetResult(target_b, 0),
            mlir.oraOperationGetResult(maybe, 0),
        },
        4,
        &[_]mlir.MlirType{},
        0,
    );
    _ = try encoder.encodeOperation(call);
    try testing.expect(!encoder.isDegraded());

    const stable = encoder.global_map.get("stable").?;
    var stable_solver = try Solver.init(&z3_ctx, testing.allocator);
    defer stable_solver.deinit();
    stable_solver.assert(z3.Z3_mk_not(z3_ctx.ctx, z3.Z3_mk_eq(z3_ctx.ctx, stable, stable_pre)));
    try testing.expectEqual(@as(z3.Z3_lbool, z3.Z3_L_FALSE), stable_solver.check());

    const counter = encoder.global_map.get("counter").?;
    const limit_ast = try encoder.encodeValue(mlir.oraOperationGetResult(limit, 0));
    const target_a_ast = try encoder.encodeValue(mlir.oraOperationGetResult(target_a, 0));
    const target_b_ast = try encoder.encodeValue(mlir.oraOperationGetResult(target_b, 0));
    const is_error_op = mlir.oraErrorIsErrorOpCreate(mlir_ctx, loc, mlir.oraOperationGetResult(maybe, 0));
    const is_error = try encoder.encodeOperation(is_error_op);
    const seven = try encoder.encodeIntegerConstant(7, 256);
    const thirty_three = try encoder.encodeIntegerConstant(33, 256);
    const reaches_a = z3.Z3_mk_bvugt(z3_ctx.ctx, limit_ast, target_a_ast);
    const reaches_b = z3.Z3_mk_bvugt(z3_ctx.ctx, limit_ast, target_b_ast);
    const catches = encoder.encodeAnd(&.{ encoder.encodeOr(&.{ reaches_a, reaches_b }), encoder.coerceBoolean(is_error) });
    const expected = z3.Z3_mk_ite(z3_ctx.ctx, catches, thirty_three, seven);

    var counter_solver = try Solver.init(&z3_ctx, testing.allocator);
    defer counter_solver.deinit();
    counter_solver.assert(z3.Z3_mk_not(z3_ctx.ctx, z3.Z3_mk_eq(z3_ctx.ctx, counter, expected)));
    try testing.expectEqual(@as(z3.Z3_lbool, z3.Z3_L_FALSE), counter_solver.check());
}

test "func.call summary with symbolic loop ora.try_stmt encodes differing slots exactly" {
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
        namedAttr(mlir_ctx, "sym_name", mlir.oraStringAttrCreate(mlir_ctx, stringRef("symbolicLoopTryWriter"))),
        namedAttr(mlir_ctx, "ora.effect", mlir.oraStringAttrCreate(mlir_ctx, stringRef("writes"))),
        namedAttr(mlir_ctx, "ora.write_slots", mlir.oraArrayAttrCreate(mlir_ctx, 1, &[_]mlir.MlirAttribute{
            mlir.oraStringAttrCreate(mlir_ctx, stringRef("counter")),
        })),
    };
    const helper_param_types = [_]mlir.MlirType{ index_ty, eu_ty };
    const helper_param_locs = [_]mlir.MlirLocation{ loc, loc };
    const helper = mlir.oraFuncFuncOpCreate(mlir_ctx, loc, &helper_attrs, helper_attrs.len, &helper_param_types, &helper_param_locs, helper_param_types.len);
    const body = mlir.oraFuncOpGetBodyBlock(helper);

    const limit_arg = mlir.oraBlockGetArgument(body, 0);
    const maybe_arg = mlir.oraBlockGetArgument(body, 1);
    const try_stmt = mlir.oraTryStmtOpCreate(mlir_ctx, loc, &[_]mlir.MlirType{}, 0);
    const try_block = mlir.oraTryStmtOpGetTryBlock(try_stmt);
    const catch_block = mlir.oraTryStmtOpGetCatchBlock(try_stmt);

    const zero_idx_op = mlir.oraArithConstantOpCreate(
        mlir_ctx,
        loc,
        index_ty,
        mlir.oraIntegerAttrCreateI64FromType(index_ty, 0),
    );
    const one_idx_op = mlir.oraArithConstantOpCreate(
        mlir_ctx,
        loc,
        index_ty,
        mlir.oraIntegerAttrCreateI64FromType(index_ty, 1),
    );
    mlir.oraBlockAppendOwnedOperation(body, zero_idx_op);
    mlir.oraBlockAppendOwnedOperation(body, one_idx_op);

    const loop = mlir.oraScfForOpCreate(
        mlir_ctx,
        loc,
        mlir.oraOperationGetResult(zero_idx_op, 0),
        limit_arg,
        mlir.oraOperationGetResult(one_idx_op, 0),
        &[_]mlir.MlirValue{},
        0,
        false,
    );
    const loop_body = mlir.oraScfForOpGetBodyBlock(loop);
    const unwrap_op = mlir.oraErrorUnwrapOpCreate(mlir_ctx, loc, maybe_arg, i256_ty);
    mlir.oraBlockAppendOwnedOperation(loop_body, unwrap_op);
    mlir.oraBlockAppendOwnedOperation(loop_body, mlir.oraScfYieldOpCreate(mlir_ctx, loc, &[_]mlir.MlirValue{}, 0));
    mlir.oraBlockAppendOwnedOperation(try_block, loop);

    const try_store_op = mlir.oraArithConstantOpCreate(
        mlir_ctx,
        loc,
        i256_ty,
        mlir.oraIntegerAttrCreateI64FromType(i256_ty, 7),
    );
    mlir.oraBlockAppendOwnedOperation(try_block, try_store_op);
    mlir.oraBlockAppendOwnedOperation(try_block, mlir.oraSStoreOpCreate(
        mlir_ctx,
        loc,
        mlir.oraOperationGetResult(try_store_op, 0),
        stringRef("counter"),
    ));
    mlir.oraBlockAppendOwnedOperation(try_block, mlir.oraYieldOpCreate(mlir_ctx, loc, &[_]mlir.MlirValue{}, 0));

    const catch_store_op = mlir.oraArithConstantOpCreate(
        mlir_ctx,
        loc,
        i256_ty,
        mlir.oraIntegerAttrCreateI64FromType(i256_ty, 33),
    );
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

    const counter_pre = try encoder.getOrCreateCurrentGlobal("counter", z3.Z3_mk_bv_sort(z3_ctx.ctx, 256));
    const stable_pre = try encoder.getOrCreateCurrentGlobal("stable", z3.Z3_mk_bv_sort(z3_ctx.ctx, 256));

    const limit = mlir.oraVariablePlaceholderOpCreate(mlir_ctx, loc, stringRef("loopLimit"), index_ty);
    const maybe = mlir.oraVariablePlaceholderOpCreate(mlir_ctx, loc, stringRef("indirectMaybe"), eu_ty);
    const call = mlir.oraFuncCallOpCreate(
        mlir_ctx,
        loc,
        stringRef("symbolicLoopTryWriter"),
        &[_]mlir.MlirValue{
            mlir.oraOperationGetResult(limit, 0),
            mlir.oraOperationGetResult(maybe, 0),
        },
        2,
        &[_]mlir.MlirType{},
        0,
    );
    _ = try encoder.encodeOperation(call);

    try testing.expect(!encoder.isDegraded());

    const stable = encoder.global_map.get("stable").?;
    var solver = try Solver.init(&z3_ctx, testing.allocator);
    defer solver.deinit();
    solver.assert(z3.Z3_mk_not(z3_ctx.ctx, z3.Z3_mk_eq(z3_ctx.ctx, stable, stable_pre)));
    try testing.expectEqual(@as(z3.Z3_lbool, z3.Z3_L_FALSE), solver.check());

    const counter = encoder.global_map.get("counter").?;
    const counter_text = std.mem.span(z3.Z3_ast_to_string(z3_ctx.ctx, counter));
    _ = counter_pre;
    try testing.expect(std.mem.indexOf(u8, counter_text, "undef_try_state_global") == null);
}

test "func.call summary with symbolic single-entry scf.while in ora.try_stmt preserves differing slots exactly" {
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
        namedAttr(mlir_ctx, "sym_name", mlir.oraStringAttrCreate(mlir_ctx, stringRef("symbolicSingleEntryTryWhileWriter"))),
        namedAttr(mlir_ctx, "ora.effect", mlir.oraStringAttrCreate(mlir_ctx, stringRef("writes"))),
        namedAttr(mlir_ctx, "ora.write_slots", mlir.oraArrayAttrCreate(mlir_ctx, 1, &[_]mlir.MlirAttribute{
            mlir.oraStringAttrCreate(mlir_ctx, stringRef("counter")),
        })),
    };
    const helper_param_types = [_]mlir.MlirType{ i1_ty, eu_ty };
    const helper_param_locs = [_]mlir.MlirLocation{ loc, loc };
    const helper = mlir.oraFuncFuncOpCreate(mlir_ctx, loc, &helper_attrs, helper_attrs.len, &helper_param_types, &helper_param_locs, helper_param_types.len);
    const body = mlir.oraFuncOpGetBodyBlock(helper);

    const enter_arg = mlir.oraBlockGetArgument(body, 0);
    const maybe_arg = mlir.oraBlockGetArgument(body, 1);
    const try_stmt = mlir.oraTryStmtOpCreate(mlir_ctx, loc, &[_]mlir.MlirType{}, 0);
    const try_block = mlir.oraTryStmtOpGetTryBlock(try_stmt);
    const catch_block = mlir.oraTryStmtOpGetCatchBlock(try_stmt);

    const false_op = mlir.oraArithConstantOpCreate(
        mlir_ctx,
        loc,
        i1_ty,
        mlir.oraIntegerAttrCreateI64FromType(i1_ty, 0),
    );
    mlir.oraBlockAppendOwnedOperation(try_block, false_op);

    const init_vals = [_]mlir.MlirValue{enter_arg};
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

    const unwrap_op = mlir.oraErrorUnwrapOpCreate(mlir_ctx, loc, maybe_arg, i256_ty);
    mlir.oraBlockAppendOwnedOperation(after_block, unwrap_op);
    mlir.oraBlockAppendOwnedOperation(after_block, mlir.oraScfYieldOpCreate(
        mlir_ctx,
        loc,
        &[_]mlir.MlirValue{mlir.oraOperationGetResult(false_op, 0)},
        1,
    ));
    mlir.oraBlockAppendOwnedOperation(try_block, while_op);

    const try_store_op = mlir.oraArithConstantOpCreate(
        mlir_ctx,
        loc,
        i256_ty,
        mlir.oraIntegerAttrCreateI64FromType(i256_ty, 7),
    );
    mlir.oraBlockAppendOwnedOperation(try_block, try_store_op);
    mlir.oraBlockAppendOwnedOperation(try_block, mlir.oraSStoreOpCreate(
        mlir_ctx,
        loc,
        mlir.oraOperationGetResult(try_store_op, 0),
        stringRef("counter"),
    ));
    mlir.oraBlockAppendOwnedOperation(try_block, mlir.oraYieldOpCreate(mlir_ctx, loc, &[_]mlir.MlirValue{}, 0));

    const catch_store_op = mlir.oraArithConstantOpCreate(
        mlir_ctx,
        loc,
        i256_ty,
        mlir.oraIntegerAttrCreateI64FromType(i256_ty, 33),
    );
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

    const counter_pre = try encoder.getOrCreateCurrentGlobal("counter", z3.Z3_mk_bv_sort(z3_ctx.ctx, 256));
    const stable_pre = try encoder.getOrCreateCurrentGlobal("stable", z3.Z3_mk_bv_sort(z3_ctx.ctx, 256));

    const enter = mlir.oraVariablePlaceholderOpCreate(mlir_ctx, loc, stringRef("shouldEnter"), i1_ty);
    const maybe = mlir.oraVariablePlaceholderOpCreate(mlir_ctx, loc, stringRef("whileMaybe"), eu_ty);
    const call = mlir.oraFuncCallOpCreate(
        mlir_ctx,
        loc,
        stringRef("symbolicSingleEntryTryWhileWriter"),
        &[_]mlir.MlirValue{
            mlir.oraOperationGetResult(enter, 0),
            mlir.oraOperationGetResult(maybe, 0),
        },
        2,
        &[_]mlir.MlirType{},
        0,
    );
    _ = try encoder.encodeOperation(call);

    try testing.expect(!encoder.isDegraded());

    const stable = encoder.global_map.get("stable").?;
    var solver = try Solver.init(&z3_ctx, testing.allocator);
    defer solver.deinit();
    solver.assert(z3.Z3_mk_not(z3_ctx.ctx, z3.Z3_mk_eq(z3_ctx.ctx, stable, stable_pre)));
    try testing.expectEqual(@as(z3.Z3_lbool, z3.Z3_L_FALSE), solver.check());

    const counter = encoder.global_map.get("counter").?;
    const counter_text = std.mem.span(z3.Z3_ast_to_string(z3_ctx.ctx, counter));
    _ = counter_pre;
    try testing.expect(std.mem.indexOf(u8, counter_text, "undef_try_state_global") == null);
}

test "func.call summary with multi-iteration symbolic scf.while in ora.try_stmt encodes differing slots exactly" {
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
        namedAttr(mlir_ctx, "sym_name", mlir.oraStringAttrCreate(mlir_ctx, stringRef("symbolicMultiIterTryWhileWriter"))),
        namedAttr(mlir_ctx, "ora.effect", mlir.oraStringAttrCreate(mlir_ctx, stringRef("writes"))),
        namedAttr(mlir_ctx, "ora.write_slots", mlir.oraArrayAttrCreate(mlir_ctx, 1, &[_]mlir.MlirAttribute{
            mlir.oraStringAttrCreate(mlir_ctx, stringRef("counter")),
        })),
    };
    const helper_param_types = [_]mlir.MlirType{ index_ty, index_ty, eu_ty };
    const helper_param_locs = [_]mlir.MlirLocation{ loc, loc, loc };
    const helper = mlir.oraFuncFuncOpCreate(mlir_ctx, loc, &helper_attrs, helper_attrs.len, &helper_param_types, &helper_param_locs, helper_param_types.len);
    const body = mlir.oraFuncOpGetBodyBlock(helper);

    const limit_arg = mlir.oraBlockGetArgument(body, 0);
    const target_arg = mlir.oraBlockGetArgument(body, 1);
    const maybe_arg = mlir.oraBlockGetArgument(body, 2);

    const zero_idx_op = mlir.oraArithConstantOpCreate(
        mlir_ctx,
        loc,
        index_ty,
        mlir.oraIntegerAttrCreateI64FromType(index_ty, 0),
    );
    const one_idx_op = mlir.oraArithConstantOpCreate(
        mlir_ctx,
        loc,
        index_ty,
        mlir.oraIntegerAttrCreateI64FromType(index_ty, 1),
    );
    mlir.oraBlockAppendOwnedOperation(body, zero_idx_op);
    mlir.oraBlockAppendOwnedOperation(body, one_idx_op);

    const try_stmt = mlir.oraTryStmtOpCreate(mlir_ctx, loc, &[_]mlir.MlirType{}, 0);
    const try_block = mlir.oraTryStmtOpGetTryBlock(try_stmt);
    const catch_block = mlir.oraTryStmtOpGetCatchBlock(try_stmt);

    const init_vals = [_]mlir.MlirValue{mlir.oraOperationGetResult(zero_idx_op, 0)};
    const result_types = [_]mlir.MlirType{index_ty};
    const while_op = mlir.oraScfWhileOpCreate(mlir_ctx, loc, &init_vals, init_vals.len, &result_types, result_types.len);
    const before_block = mlir.oraScfWhileOpGetBeforeBlock(while_op);
    const after_block = mlir.oraScfWhileOpGetAfterBlock(while_op);
    _ = mlir.mlirBlockAddArgument(before_block, index_ty, loc);
    _ = mlir.mlirBlockAddArgument(after_block, index_ty, loc);
    const before_idx = mlir.oraBlockGetArgument(before_block, 0);
    const after_idx = mlir.oraBlockGetArgument(after_block, 0);

    const continue_cmp = mlir.oraArithCmpIOpCreate(mlir_ctx, loc, 6, before_idx, limit_arg); // ult
    mlir.oraBlockAppendOwnedOperation(before_block, continue_cmp);
    mlir.oraBlockAppendOwnedOperation(before_block, mlir.oraScfConditionOpCreate(
        mlir_ctx,
        loc,
        mlir.oraOperationGetResult(continue_cmp, 0),
        &[_]mlir.MlirValue{before_idx},
        1,
    ));

    const hit_cmp = mlir.oraArithCmpIOpCreate(mlir_ctx, loc, 0, after_idx, target_arg); // eq
    mlir.oraBlockAppendOwnedOperation(after_block, hit_cmp);
    const no_results = [_]mlir.MlirType{};
    const if_op = mlir.oraScfIfOpCreate(
        mlir_ctx,
        loc,
        mlir.oraOperationGetResult(hit_cmp, 0),
        &no_results,
        no_results.len,
        true,
    );
    const then_block = mlir.oraScfIfOpGetThenBlock(if_op);
    const else_block = mlir.oraScfIfOpGetElseBlock(if_op);
    const unwrap_op = mlir.oraErrorUnwrapOpCreate(mlir_ctx, loc, maybe_arg, i256_ty);
    mlir.oraBlockAppendOwnedOperation(then_block, unwrap_op);
    mlir.oraBlockAppendOwnedOperation(then_block, mlir.oraScfYieldOpCreate(mlir_ctx, loc, &[_]mlir.MlirValue{}, 0));
    mlir.oraBlockAppendOwnedOperation(else_block, mlir.oraScfYieldOpCreate(mlir_ctx, loc, &[_]mlir.MlirValue{}, 0));
    mlir.oraBlockAppendOwnedOperation(after_block, if_op);

    const next_idx = mlir.oraArithAddIOpCreate(
        mlir_ctx,
        loc,
        after_idx,
        mlir.oraOperationGetResult(one_idx_op, 0),
    );
    mlir.oraBlockAppendOwnedOperation(after_block, next_idx);
    mlir.oraBlockAppendOwnedOperation(after_block, mlir.oraScfYieldOpCreate(
        mlir_ctx,
        loc,
        &[_]mlir.MlirValue{mlir.oraOperationGetResult(next_idx, 0)},
        1,
    ));
    mlir.oraBlockAppendOwnedOperation(try_block, while_op);

    const try_store_op = mlir.oraArithConstantOpCreate(
        mlir_ctx,
        loc,
        i256_ty,
        mlir.oraIntegerAttrCreateI64FromType(i256_ty, 7),
    );
    mlir.oraBlockAppendOwnedOperation(try_block, try_store_op);
    mlir.oraBlockAppendOwnedOperation(try_block, mlir.oraSStoreOpCreate(
        mlir_ctx,
        loc,
        mlir.oraOperationGetResult(try_store_op, 0),
        stringRef("counter"),
    ));
    mlir.oraBlockAppendOwnedOperation(try_block, mlir.oraYieldOpCreate(mlir_ctx, loc, &[_]mlir.MlirValue{}, 0));

    const catch_store_op = mlir.oraArithConstantOpCreate(
        mlir_ctx,
        loc,
        i256_ty,
        mlir.oraIntegerAttrCreateI64FromType(i256_ty, 33),
    );
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

    const counter_pre = try encoder.getOrCreateCurrentGlobal("counter", z3.Z3_mk_bv_sort(z3_ctx.ctx, 256));
    const stable_pre = try encoder.getOrCreateCurrentGlobal("stable", z3.Z3_mk_bv_sort(z3_ctx.ctx, 256));

    const limit = mlir.oraVariablePlaceholderOpCreate(mlir_ctx, loc, stringRef("multiIterLimit"), index_ty);
    const target = mlir.oraVariablePlaceholderOpCreate(mlir_ctx, loc, stringRef("multiIterTarget"), index_ty);
    const maybe = mlir.oraVariablePlaceholderOpCreate(mlir_ctx, loc, stringRef("multiIterMaybe"), eu_ty);
    const call = mlir.oraFuncCallOpCreate(
        mlir_ctx,
        loc,
        stringRef("symbolicMultiIterTryWhileWriter"),
        &[_]mlir.MlirValue{
            mlir.oraOperationGetResult(limit, 0),
            mlir.oraOperationGetResult(target, 0),
            mlir.oraOperationGetResult(maybe, 0),
        },
        3,
        &[_]mlir.MlirType{},
        0,
    );
    _ = try encoder.encodeOperation(call);

    try testing.expect(!encoder.isDegraded());

    const stable = encoder.global_map.get("stable").?;
    var solver = try Solver.init(&z3_ctx, testing.allocator);
    defer solver.deinit();
    solver.assert(z3.Z3_mk_not(z3_ctx.ctx, z3.Z3_mk_eq(z3_ctx.ctx, stable, stable_pre)));
    try testing.expectEqual(@as(z3.Z3_lbool, z3.Z3_L_FALSE), solver.check());

    const counter = encoder.global_map.get("counter").?;
    const limit_ast = try encoder.encodeValue(mlir.oraOperationGetResult(limit, 0));
    const target_ast = try encoder.encodeValue(mlir.oraOperationGetResult(target, 0));
    const is_error_op = mlir.oraErrorIsErrorOpCreate(mlir_ctx, loc, mlir.oraOperationGetResult(maybe, 0));
    const is_error = try encoder.encodeOperation(is_error_op);
    const seven = try encoder.encodeIntegerConstant(7, 256);
    const thirty_three = try encoder.encodeIntegerConstant(33, 256);
    const reaches_target = z3.Z3_mk_bvugt(z3_ctx.ctx, limit_ast, target_ast);
    const catches = encoder.encodeAnd(&.{ reaches_target, encoder.coerceBoolean(is_error) });
    const expected = z3.Z3_mk_ite(z3_ctx.ctx, catches, thirty_three, seven);
    _ = counter_pre;
    var counter_solver = try Solver.init(&z3_ctx, testing.allocator);
    defer counter_solver.deinit();
    counter_solver.assert(z3.Z3_mk_not(z3_ctx.ctx, z3.Z3_mk_eq(z3_ctx.ctx, counter, expected)));
    try testing.expectEqual(@as(z3.Z3_lbool, z3.Z3_L_FALSE), counter_solver.check());
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

test "known pure callee canonical unsigned inclusive positive-delta scf.while return encodes exactly" {
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
        namedAttr(mlir_ctx, "sym_name", mlir.oraStringAttrCreate(mlir_ctx, stringRef("symbolicWhileDeltaInclusiveReturn"))),
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

    const cmp_op = mlir.oraArithCmpIOpCreate(mlir_ctx, loc, 7, before_arg, ub); // ule
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

    const caller_ub = mlir.oraVariablePlaceholderOpCreate(mlir_ctx, loc, stringRef("callerWhileDeltaInclusiveUb"), i256_ty);
    const call = mlir.oraFuncCallOpCreate(
        mlir_ctx,
        loc,
        stringRef("symbolicWhileDeltaInclusiveReturn"),
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
    const sort = z3.Z3_get_sort(z3_ctx.ctx, ub_ast);
    const zero = z3.Z3_mk_unsigned_int64(z3_ctx.ctx, 0, sort);
    const one = z3.Z3_mk_unsigned_int64(z3_ctx.ctx, 1, sort);
    const exclusive_bound = z3.Z3_mk_bv_add(z3_ctx.ctx, ub_ast, one);
    const ub_le_init = z3.Z3_mk_bvule(z3_ctx.ctx, exclusive_bound, init_ast);
    const distance = z3.Z3_mk_ite(
        z3_ctx.ctx,
        ub_le_init,
        zero,
        z3.Z3_mk_bv_sub(z3_ctx.ctx, exclusive_bound, init_ast),
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

test "known pure callee branch-conditioned ora.try_stmt result encodes exactly" {
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
        namedAttr(mlir_ctx, "sym_name", mlir.oraStringAttrCreate(mlir_ctx, stringRef("branchConditionedTryHelper"))),
    };
    const helper_param_types = [_]mlir.MlirType{ i1_ty, eu_ty };
    const helper_param_locs = [_]mlir.MlirLocation{ loc, loc };
    const helper = mlir.oraFuncFuncOpCreate(mlir_ctx, loc, &helper_attrs, helper_attrs.len, &helper_param_types, &helper_param_locs, helper_param_types.len);
    const body = mlir.oraFuncOpGetBodyBlock(helper);

    const cond_arg = mlir.oraBlockGetArgument(body, 0);
    const maybe_arg = mlir.oraBlockGetArgument(body, 1);

    const try_stmt = mlir.oraTryStmtOpCreate(mlir_ctx, loc, &[_]mlir.MlirType{i256_ty}, 1);
    const try_block = mlir.oraTryStmtOpGetTryBlock(try_stmt);
    const catch_block = mlir.oraTryStmtOpGetCatchBlock(try_stmt);

    const if_op = mlir.oraScfIfOpCreate(
        mlir_ctx,
        loc,
        cond_arg,
        &[_]mlir.MlirType{i256_ty},
        1,
        true,
    );
    const then_block = mlir.oraScfIfOpGetThenBlock(if_op);
    const else_block = mlir.oraScfIfOpGetElseBlock(if_op);

    const unwrap_op = mlir.oraErrorUnwrapOpCreate(mlir_ctx, loc, maybe_arg, i256_ty);
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

    mlir.oraBlockAppendOwnedOperation(body, try_stmt);
    mlir.oraBlockAppendOwnedOperation(body, mlir.oraReturnOpCreate(
        mlir_ctx,
        loc,
        &[_]mlir.MlirValue{mlir.oraOperationGetResult(try_stmt, 0)},
        1,
    ));

    try encoder.registerFunctionOperation(helper);

    const cond = mlir.oraVariablePlaceholderOpCreate(mlir_ctx, loc, stringRef("condValue"), i1_ty);
    const maybe = mlir.oraVariablePlaceholderOpCreate(mlir_ctx, loc, stringRef("maybeValue"), eu_ty);
    const call = mlir.oraFuncCallOpCreate(
        mlir_ctx,
        loc,
        stringRef("branchConditionedTryHelper"),
        &[_]mlir.MlirValue{
            mlir.oraOperationGetResult(cond, 0),
            mlir.oraOperationGetResult(maybe, 0),
        },
        2,
        &[_]mlir.MlirType{i256_ty},
        1,
    );

    const encoded = try encoder.encodeOperation(call);
    try testing.expect(!encoder.isDegraded());
    const encoded_text = std.mem.span(z3.Z3_ast_to_string(z3_ctx.ctx, encoded));
    try testing.expect(std.mem.indexOf(u8, encoded_text, "ite") != null);
    try testing.expect(std.mem.indexOf(u8, encoded_text, "condValue") != null);
    try testing.expect(std.mem.indexOf(u8, encoded_text, "maybeValue") != null);
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

    const constraints = try encoder.takeConstraints(testing.allocator);
    defer if (constraints.len > 0) testing.allocator.free(constraints);
    try expectNoQuantifiedConstraints(&z3_ctx, constraints);

    const sload_after = mlir.oraSLoadOpCreate(mlir_ctx, loc, stringRef("counter"), i256_ty);
    const loaded = try encoder.encodeOperation(sload_after);
    const expected = try encoder.encodeValue(seed);

    var solver = try Solver.init(&z3_ctx, testing.allocator);
    defer solver.deinit();
    for (constraints) |cst| solver.assert(cst);
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

    const constraints = try encoder.takeConstraints(testing.allocator);
    defer if (constraints.len > 0) testing.allocator.free(constraints);
    try expectNoQuantifiedConstraints(&z3_ctx, constraints);

    const sload_after = mlir.oraSLoadOpCreate(mlir_ctx, loc, stringRef("counter"), i256_ty);
    const loaded = try encoder.encodeOperation(sload_after);
    const expected = try encoder.encodeValue(seed);

    var solver = try Solver.init(&z3_ctx, testing.allocator);
    defer solver.deinit();
    for (constraints) |cst| solver.assert(cst);
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
            std.mem.containsAtLeast(u8, reason, 1, "known pure callee") or
            std.mem.containsAtLeast(u8, reason, 1, "structured control") or
            std.mem.containsAtLeast(u8, reason, 1, "loop state summary") or
            std.mem.containsAtLeast(u8, reason, 1, "loop summary"),
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

test "known pure callee finite two-iteration scf.while return encodes exactly" {
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
        namedAttr(mlir_ctx, "sym_name", mlir.oraStringAttrCreate(mlir_ctx, stringRef("twoIterWhileReturn"))),
    };
    const helper = mlir.oraFuncFuncOpCreate(mlir_ctx, loc, &helper_attrs, helper_attrs.len, &[_]mlir.MlirType{}, &[_]mlir.MlirLocation{}, 0);
    const body = mlir.oraFuncOpGetBodyBlock(helper);

    const true_op = mlir.oraArithConstantOpCreate(mlir_ctx, loc, i1_ty, mlir.oraIntegerAttrCreateI64FromType(i1_ty, 1));
    const false_op = mlir.oraArithConstantOpCreate(mlir_ctx, loc, i1_ty, mlir.oraIntegerAttrCreateI64FromType(i1_ty, 0));
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
    const after_next = mlir.oraBlockGetArgument(after_block, 1);

    mlir.oraBlockAppendOwnedOperation(before_block, mlir.oraScfConditionOpCreate(
        mlir_ctx,
        loc,
        before_flag,
        &[_]mlir.MlirValue{ before_flag, before_next },
        2,
    ));
    mlir.oraBlockAppendOwnedOperation(after_block, mlir.oraScfYieldOpCreate(
        mlir_ctx,
        loc,
        &[_]mlir.MlirValue{ after_next, mlir.oraOperationGetResult(false_op, 0) },
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

    const call = mlir.oraFuncCallOpCreate(
        mlir_ctx,
        loc,
        stringRef("twoIterWhileReturn"),
        &[_]mlir.MlirValue{},
        0,
        &[_]mlir.MlirType{i1_ty},
        1,
    );
    const encoded = try encoder.encodeOperation(call);
    const expected = encoder.encodeBoolConstant(false);

    try testing.expect(!encoder.isDegraded());

    var solver = try Solver.init(&z3_ctx, testing.allocator);
    defer solver.deinit();
    solver.assert(z3.Z3_mk_not(z3_ctx.ctx, z3.Z3_mk_eq(z3_ctx.ctx, encoded, expected)));
    try testing.expectEqual(@as(z3.Z3_lbool, z3.Z3_L_FALSE), solver.check());
}

test "known pure callee canonical unsigned swapped-compare scf.while return encodes exactly" {
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
        namedAttr(mlir_ctx, "sym_name", mlir.oraStringAttrCreate(mlir_ctx, stringRef("unsignedSwappedCmpWhileReturn"))),
    };
    const helper_param_types = [_]mlir.MlirType{i256_ty};
    const helper_param_locs = [_]mlir.MlirLocation{loc};
    const helper = mlir.oraFuncFuncOpCreate(mlir_ctx, loc, &helper_attrs, helper_attrs.len, &helper_param_types, &helper_param_locs, helper_param_types.len);
    const body = mlir.oraFuncOpGetBodyBlock(helper);

    const init_op = mlir.oraArithConstantOpCreate(mlir_ctx, loc, i256_ty, mlir.oraIntegerAttrCreateI64FromType(i256_ty, 0));
    const delta_op = mlir.oraArithConstantOpCreate(mlir_ctx, loc, i256_ty, mlir.oraIntegerAttrCreateI64FromType(i256_ty, 3));
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

    const cmp_op = mlir.oraArithCmpIOpCreate(mlir_ctx, loc, 8, bound, before_arg); // bound > current
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

    const caller_bound = mlir.oraVariablePlaceholderOpCreate(mlir_ctx, loc, stringRef("callerUnsignedSwappedCmpBound"), i256_ty);
    const call = mlir.oraFuncCallOpCreate(
        mlir_ctx,
        loc,
        stringRef("unsignedSwappedCmpWhileReturn"),
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
    const init_ge_bound = z3.Z3_mk_bvuge(z3_ctx.ctx, init_ast, bound_ast);
    const distance = z3.Z3_mk_ite(
        z3_ctx.ctx,
        init_ge_bound,
        zero,
        z3.Z3_mk_bv_sub(z3_ctx.ctx, bound_ast, init_ast),
    );
    const distance_is_zero = z3.Z3_mk_eq(z3_ctx.ctx, distance, zero);
    const three = z3.Z3_mk_unsigned_int64(z3_ctx.ctx, 3, sort);
    const step_count = z3.Z3_mk_ite(
        z3_ctx.ctx,
        distance_is_zero,
        zero,
        z3.Z3_mk_bv_add(
            z3_ctx.ctx,
            z3.Z3_mk_bv_udiv(z3_ctx.ctx, z3.Z3_mk_bv_sub(z3_ctx.ctx, distance, one), three),
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

test "known pure callee canonical signed symbolic increment scf.while return encodes exactly" {
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
        namedAttr(mlir_ctx, "sym_name", mlir.oraStringAttrCreate(mlir_ctx, stringRef("symbolicSignedIncWhileReturn"))),
    };
    const helper_param_types = [_]mlir.MlirType{i256_ty};
    const helper_param_locs = [_]mlir.MlirLocation{loc};
    const helper = mlir.oraFuncFuncOpCreate(mlir_ctx, loc, &helper_attrs, helper_attrs.len, &helper_param_types, &helper_param_locs, helper_param_types.len);
    const body = mlir.oraFuncOpGetBodyBlock(helper);

    const zero_op = mlir.oraArithConstantOpCreate(mlir_ctx, loc, i256_ty, mlir.oraIntegerAttrCreateI64FromType(i256_ty, 0));
    const one_op = mlir.oraArithConstantOpCreate(mlir_ctx, loc, i256_ty, mlir.oraIntegerAttrCreateI64FromType(i256_ty, 1));
    mlir.oraBlockAppendOwnedOperation(body, zero_op);
    mlir.oraBlockAppendOwnedOperation(body, one_op);

    const bound = mlir.oraBlockGetArgument(body, 0);
    const init_vals = [_]mlir.MlirValue{mlir.oraOperationGetResult(zero_op, 0)};
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
    mlir.oraBlockAppendOwnedOperation(body, while_op);
    mlir.oraBlockAppendOwnedOperation(body, mlir.oraReturnOpCreate(
        mlir_ctx,
        loc,
        &[_]mlir.MlirValue{mlir.oraOperationGetResult(while_op, 0)},
        1,
    ));

    try encoder.registerFunctionOperation(helper);

    const caller_bound = mlir.oraVariablePlaceholderOpCreate(mlir_ctx, loc, stringRef("callerSignedIncWhileBound"), i256_ty);
    const call = mlir.oraFuncCallOpCreate(
        mlir_ctx,
        loc,
        stringRef("symbolicSignedIncWhileReturn"),
        &[_]mlir.MlirValue{mlir.oraOperationGetResult(caller_bound, 0)},
        1,
        &[_]mlir.MlirType{i256_ty},
        1,
    );
    const encoded = try encoder.encodeOperation(call);
    try testing.expect(!encoder.isDegraded());

    const init_ast = try encoder.encodeValue(mlir.oraOperationGetResult(zero_op, 0));
    const bound_ast = try encoder.encodeValue(mlir.oraOperationGetResult(caller_bound, 0));
    const expected = try encoder.encodeControlFlow(
        "scf.if",
        z3.Z3_mk_bvslt(z3_ctx.ctx, init_ast, bound_ast),
        bound_ast,
        init_ast,
    );
    const expected_str = try testing.allocator.dupe(u8, std.mem.span(z3.Z3_ast_to_string(z3_ctx.ctx, expected)));
    defer testing.allocator.free(expected_str);
    const encoded_str = try testing.allocator.dupe(u8, std.mem.span(z3.Z3_ast_to_string(z3_ctx.ctx, encoded)));
    defer testing.allocator.free(encoded_str);
    try testing.expectEqualStrings(expected_str, encoded_str);
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

test "known callee with partially recovered write set still degrades encoder" {
    var z3_ctx = try Context.init(testing.allocator);
    defer z3_ctx.deinit();

    var encoder = Encoder.init(&z3_ctx, testing.allocator);
    defer encoder.deinit();
    encoder.setVerifyState(true);
    encoder.max_summary_inline_depth = 0;

    const mlir_ctx = mlir.oraContextCreate();
    defer mlir.oraContextDestroy(mlir_ctx);
    loadAllDialects(mlir_ctx);
    _ = mlir.oraDialectRegister(mlir_ctx);

    const loc = mlir.oraLocationUnknownGet(mlir_ctx);
    const i256_ty = mlir.oraIntegerTypeCreate(mlir_ctx, 256);

    const helper_attrs = [_]mlir.MlirNamedAttribute{
        namedAttr(mlir_ctx, "sym_name", mlir.oraStringAttrCreate(mlir_ctx, stringRef("writerWithPartialUnknownSlots"))),
        namedAttr(mlir_ctx, "ora.effect", mlir.oraStringAttrCreate(mlir_ctx, stringRef("writes"))),
    };
    const helper = mlir.oraFuncFuncOpCreate(mlir_ctx, loc, &helper_attrs, helper_attrs.len, &[_]mlir.MlirType{}, &[_]mlir.MlirLocation{}, 0);
    const helper_body = mlir.oraFuncOpGetBodyBlock(helper);

    const one_attr = mlir.oraIntegerAttrCreateI64FromType(i256_ty, 1);
    const one_op = mlir.oraArithConstantOpCreate(mlir_ctx, loc, i256_ty, one_attr);
    const one = mlir.oraOperationGetResult(one_op, 0);
    const known_store = mlir.oraSStoreOpCreate(mlir_ctx, loc, one, stringRef("counter"));
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
    mlir.oraBlockAppendOwnedOperation(helper_body, one_op);
    mlir.oraBlockAppendOwnedOperation(helper_body, known_store);
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
        stringRef("writerWithPartialUnknownSlots"),
        &[_]mlir.MlirValue{},
        0,
        &[_]mlir.MlirType{},
        0,
    );
    _ = try encoder.encodeOperation(call);

    try testing.expect(encoder.isDegraded());
    try testing.expect(std.mem.eql(u8, encoder.degradationReason().?, "failed to recover known callee write set exactly"));
}

test "known callee with unknown read set degrades encoder" {
    var z3_ctx = try Context.init(testing.allocator);
    defer z3_ctx.deinit();

    var encoder = Encoder.init(&z3_ctx, testing.allocator);
    defer encoder.deinit();
    encoder.setVerifyState(true);
    encoder.max_summary_inline_depth = 0;

    const mlir_ctx = mlir.oraContextCreate();
    defer mlir.oraContextDestroy(mlir_ctx);
    loadAllDialects(mlir_ctx);
    _ = mlir.oraDialectRegister(mlir_ctx);

    const loc = mlir.oraLocationUnknownGet(mlir_ctx);
    const i256_ty = mlir.oraIntegerTypeCreate(mlir_ctx, 256);

    const helper_attrs = [_]mlir.MlirNamedAttribute{
        namedAttr(mlir_ctx, "sym_name", mlir.oraStringAttrCreate(mlir_ctx, stringRef("readerWithUnknownSlots"))),
        namedAttr(mlir_ctx, "ora.effect", mlir.oraStringAttrCreate(mlir_ctx, stringRef("reads"))),
    };
    const helper = mlir.oraFuncFuncOpCreate(mlir_ctx, loc, &helper_attrs, helper_attrs.len, &[_]mlir.MlirType{}, &[_]mlir.MlirLocation{}, 0);
    const helper_body = mlir.oraFuncOpGetBodyBlock(helper);

    const unresolved_call = mlir.oraFuncCallOpCreate(
        mlir_ctx,
        loc,
        stringRef("opaqueReader"),
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
        stringRef("readerWithUnknownSlots"),
        &[_]mlir.MlirValue{},
        0,
        &[_]mlir.MlirType{},
        0,
    );
    _ = try encoder.encodeOperation(call);

    try testing.expect(encoder.isDegraded());
    var found_read_reason = false;
    for (encoder.degradationReasons()) |reason| {
        if (std.mem.eql(u8, reason, "failed to recover known callee read set exactly")) {
            found_read_reason = true;
            break;
        }
    }
    try testing.expect(found_read_reason);
}

test "known callee write metadata does not hide unresolved body calls" {
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
    const i256_ty = mlir.oraIntegerTypeCreate(mlir_ctx, 256);
    const write_slots = [_]mlir.MlirAttribute{
        mlir.oraStringAttrCreate(mlir_ctx, stringRef("counter")),
    };
    const helper_attrs = [_]mlir.MlirNamedAttribute{
        namedAttr(mlir_ctx, "sym_name", mlir.oraStringAttrCreate(mlir_ctx, stringRef("writerWithSummary"))),
        namedAttr(mlir_ctx, "ora.effect", mlir.oraStringAttrCreate(mlir_ctx, stringRef("writes"))),
        namedAttr(mlir_ctx, "ora.write_slots", mlir.oraArrayAttrCreate(mlir_ctx, write_slots.len, &write_slots)),
    };
    const helper = mlir.oraFuncFuncOpCreate(mlir_ctx, loc, &helper_attrs, helper_attrs.len, &[_]mlir.MlirType{i256_ty}, &[_]mlir.MlirLocation{loc}, 1);
    const helper_body = mlir.oraFuncOpGetBodyBlock(helper);
    const value_arg = mlir.oraBlockGetArgument(helper_body, 0);
    const known_store = mlir.oraSStoreOpCreate(mlir_ctx, loc, value_arg, stringRef("counter"));
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
    mlir.oraBlockAppendOwnedOperation(helper_body, known_store);
    mlir.oraBlockAppendOwnedOperation(helper_body, unresolved_call);
    mlir.oraBlockAppendOwnedOperation(helper_body, ret);
    try encoder.registerFunctionOperation(helper);

    const seed_attr = mlir.oraIntegerAttrCreateI64FromType(i256_ty, 0);
    const seed_op = mlir.oraArithConstantOpCreate(mlir_ctx, loc, i256_ty, seed_attr);
    _ = try encoder.encodeOperation(mlir.oraSStoreOpCreate(mlir_ctx, loc, mlir.oraOperationGetResult(seed_op, 0), stringRef("counter")));

    const value_attr = mlir.oraIntegerAttrCreateI64FromType(i256_ty, 42);
    const value_op = mlir.oraArithConstantOpCreate(mlir_ctx, loc, i256_ty, value_attr);
    const value = mlir.oraOperationGetResult(value_op, 0);
    const call = mlir.oraFuncCallOpCreate(
        mlir_ctx,
        loc,
        stringRef("writerWithSummary"),
        &[_]mlir.MlirValue{value},
        1,
        &[_]mlir.MlirType{},
        0,
    );
    _ = try encoder.encodeOperation(call);
    try testing.expect(encoder.isDegraded());

    var found_reason = false;
    for (encoder.degradationReasons()) |reason| {
        if (std.mem.eql(u8, reason, "unresolved external callee has no sound state summary")) {
            found_reason = true;
            break;
        }
    }
    try testing.expect(found_reason);
}

test "known callee write metadata omission degrades and widens write set" {
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
    const i256_ty = mlir.oraIntegerTypeCreate(mlir_ctx, 256);
    const write_slots = [_]mlir.MlirAttribute{
        mlir.oraStringAttrCreate(mlir_ctx, stringRef("counter")),
    };
    const helper_attrs = [_]mlir.MlirNamedAttribute{
        namedAttr(mlir_ctx, "sym_name", mlir.oraStringAttrCreate(mlir_ctx, stringRef("writerWithOmittedSlot"))),
        namedAttr(mlir_ctx, "ora.effect", mlir.oraStringAttrCreate(mlir_ctx, stringRef("writes"))),
        namedAttr(mlir_ctx, "ora.write_slots", mlir.oraArrayAttrCreate(mlir_ctx, write_slots.len, &write_slots)),
    };
    const helper = mlir.oraFuncFuncOpCreate(mlir_ctx, loc, &helper_attrs, helper_attrs.len, &[_]mlir.MlirType{}, &[_]mlir.MlirLocation{}, 0);
    const helper_body = mlir.oraFuncOpGetBodyBlock(helper);
    const value_attr = mlir.oraIntegerAttrCreateI64FromType(i256_ty, 99);
    const value_op = mlir.oraArithConstantOpCreate(mlir_ctx, loc, i256_ty, value_attr);
    const omitted_store = mlir.oraSStoreOpCreate(mlir_ctx, loc, mlir.oraOperationGetResult(value_op, 0), stringRef("secret"));
    const ret = mlir.oraReturnOpCreate(mlir_ctx, loc, &[_]mlir.MlirValue{}, 0);
    mlir.oraBlockAppendOwnedOperation(helper_body, value_op);
    mlir.oraBlockAppendOwnedOperation(helper_body, omitted_store);
    mlir.oraBlockAppendOwnedOperation(helper_body, ret);
    try encoder.registerFunctionOperation(helper);

    const seed_attr = mlir.oraIntegerAttrCreateI64FromType(i256_ty, 0);
    const seed_op = mlir.oraArithConstantOpCreate(mlir_ctx, loc, i256_ty, seed_attr);
    _ = try encoder.encodeOperation(mlir.oraSStoreOpCreate(mlir_ctx, loc, mlir.oraOperationGetResult(seed_op, 0), stringRef("secret")));

    const call = mlir.oraFuncCallOpCreate(
        mlir_ctx,
        loc,
        stringRef("writerWithOmittedSlot"),
        &[_]mlir.MlirValue{},
        0,
        &[_]mlir.MlirType{},
        0,
    );
    _ = try encoder.encodeOperation(call);

    try testing.expect(encoder.isDegraded());
    var found_loss = false;
    for (encoder.soundnessLosses()) |loss| {
        if (loss == .inexact_call_summary) {
            found_loss = true;
            break;
        }
    }
    try testing.expect(found_loss);

    var found_reason = false;
    for (encoder.degradationReasons()) |reason| {
        if (std.mem.eql(u8, reason, "ora.write_slots metadata omitted a body write")) {
            found_reason = true;
            break;
        }
    }
    try testing.expect(found_reason);
}

test "known callee empty write metadata omission degrades with specific reason" {
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
    const i256_ty = mlir.oraIntegerTypeCreate(mlir_ctx, 256);
    const no_write_slots = [_]mlir.MlirAttribute{};
    const helper_attrs = [_]mlir.MlirNamedAttribute{
        namedAttr(mlir_ctx, "sym_name", mlir.oraStringAttrCreate(mlir_ctx, stringRef("writerWithEmptySlots"))),
        namedAttr(mlir_ctx, "ora.effect", mlir.oraStringAttrCreate(mlir_ctx, stringRef("writes"))),
        namedAttr(mlir_ctx, "ora.write_slots", mlir.oraArrayAttrCreate(mlir_ctx, no_write_slots.len, &no_write_slots)),
    };
    const helper = mlir.oraFuncFuncOpCreate(mlir_ctx, loc, &helper_attrs, helper_attrs.len, &[_]mlir.MlirType{}, &[_]mlir.MlirLocation{}, 0);
    const helper_body = mlir.oraFuncOpGetBodyBlock(helper);
    const value_attr = mlir.oraIntegerAttrCreateI64FromType(i256_ty, 5);
    const value_op = mlir.oraArithConstantOpCreate(mlir_ctx, loc, i256_ty, value_attr);
    const omitted_store = mlir.oraSStoreOpCreate(mlir_ctx, loc, mlir.oraOperationGetResult(value_op, 0), stringRef("counter"));
    const ret = mlir.oraReturnOpCreate(mlir_ctx, loc, &[_]mlir.MlirValue{}, 0);
    mlir.oraBlockAppendOwnedOperation(helper_body, value_op);
    mlir.oraBlockAppendOwnedOperation(helper_body, omitted_store);
    mlir.oraBlockAppendOwnedOperation(helper_body, ret);
    try encoder.registerFunctionOperation(helper);

    const seed_attr = mlir.oraIntegerAttrCreateI64FromType(i256_ty, 0);
    const seed_op = mlir.oraArithConstantOpCreate(mlir_ctx, loc, i256_ty, seed_attr);
    _ = try encoder.encodeOperation(mlir.oraSStoreOpCreate(mlir_ctx, loc, mlir.oraOperationGetResult(seed_op, 0), stringRef("counter")));

    const call = mlir.oraFuncCallOpCreate(
        mlir_ctx,
        loc,
        stringRef("writerWithEmptySlots"),
        &[_]mlir.MlirValue{},
        0,
        &[_]mlir.MlirType{},
        0,
    );
    _ = try encoder.encodeOperation(call);

    try testing.expect(encoder.isDegraded());
    var found_reason = false;
    for (encoder.degradationReasons()) |reason| {
        if (std.mem.eql(u8, reason, "ora.write_slots metadata omitted a body write")) {
            found_reason = true;
            break;
        }
    }
    try testing.expect(found_reason);
}

test "known callee exact write metadata does not degrade" {
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
    const i256_ty = mlir.oraIntegerTypeCreate(mlir_ctx, 256);
    const write_slots = [_]mlir.MlirAttribute{
        mlir.oraStringAttrCreate(mlir_ctx, stringRef("counter")),
    };
    const helper_attrs = [_]mlir.MlirNamedAttribute{
        namedAttr(mlir_ctx, "sym_name", mlir.oraStringAttrCreate(mlir_ctx, stringRef("writerWithExactSlots"))),
        namedAttr(mlir_ctx, "ora.effect", mlir.oraStringAttrCreate(mlir_ctx, stringRef("writes"))),
        namedAttr(mlir_ctx, "ora.write_slots", mlir.oraArrayAttrCreate(mlir_ctx, write_slots.len, &write_slots)),
    };
    const helper = mlir.oraFuncFuncOpCreate(mlir_ctx, loc, &helper_attrs, helper_attrs.len, &[_]mlir.MlirType{}, &[_]mlir.MlirLocation{}, 0);
    const helper_body = mlir.oraFuncOpGetBodyBlock(helper);
    const value_attr = mlir.oraIntegerAttrCreateI64FromType(i256_ty, 7);
    const value_op = mlir.oraArithConstantOpCreate(mlir_ctx, loc, i256_ty, value_attr);
    const store = mlir.oraSStoreOpCreate(mlir_ctx, loc, mlir.oraOperationGetResult(value_op, 0), stringRef("counter"));
    const ret = mlir.oraReturnOpCreate(mlir_ctx, loc, &[_]mlir.MlirValue{}, 0);
    mlir.oraBlockAppendOwnedOperation(helper_body, value_op);
    mlir.oraBlockAppendOwnedOperation(helper_body, store);
    mlir.oraBlockAppendOwnedOperation(helper_body, ret);
    try encoder.registerFunctionOperation(helper);

    const seed_attr = mlir.oraIntegerAttrCreateI64FromType(i256_ty, 0);
    const seed_op = mlir.oraArithConstantOpCreate(mlir_ctx, loc, i256_ty, seed_attr);
    _ = try encoder.encodeOperation(mlir.oraSStoreOpCreate(mlir_ctx, loc, mlir.oraOperationGetResult(seed_op, 0), stringRef("counter")));

    const call = mlir.oraFuncCallOpCreate(
        mlir_ctx,
        loc,
        stringRef("writerWithExactSlots"),
        &[_]mlir.MlirValue{},
        0,
        &[_]mlir.MlirType{},
        0,
    );
    _ = try encoder.encodeOperation(call);

    try testing.expect(!encoder.isDegraded());
    try testing.expectEqual(@as(usize, 0), encoder.soundnessLosses().len);
    try testing.expectEqual(@as(usize, 0), encoder.precisionNotes().len);
}

test "known callee opaque modifies metadata fallback handles indexed and root paths" {
    var z3_ctx = try Context.init(testing.allocator);
    defer z3_ctx.deinit();

    var encoder = Encoder.init(&z3_ctx, testing.allocator);
    defer encoder.deinit();
    encoder.setVerifyState(true);
    encoder.max_summary_inline_depth = 0;

    const mlir_ctx = mlir.oraContextCreate();
    defer mlir.oraContextDestroy(mlir_ctx);
    loadAllDialects(mlir_ctx);
    _ = mlir.oraDialectRegister(mlir_ctx);

    const loc = mlir.oraLocationUnknownGet(mlir_ctx);
    const i256_ty = mlir.oraIntegerTypeCreate(mlir_ctx, 256);
    const map_ty = mlir.oraMapTypeGet(mlir_ctx, i256_ty, i256_ty);

    const modifies_slots = [_]mlir.MlirAttribute{
        mlir.oraStringAttrCreate(mlir_ctx, stringRef("balances[param#0]")),
        mlir.oraStringAttrCreate(mlir_ctx, stringRef("total_supply")),
    };
    const helper_attrs = [_]mlir.MlirNamedAttribute{
        namedAttr(mlir_ctx, "sym_name", mlir.oraStringAttrCreate(mlir_ctx, stringRef("opaqueSetBalanceAndTotal"))),
        namedAttr(mlir_ctx, "ora.effect", mlir.oraStringAttrCreate(mlir_ctx, stringRef("writes"))),
        namedAttr(mlir_ctx, "ora.modifies_slots", mlir.oraArrayAttrCreate(mlir_ctx, modifies_slots.len, &modifies_slots)),
    };
    const helper_param_types = [_]mlir.MlirType{ i256_ty, i256_ty, i256_ty };
    const helper_param_locs = [_]mlir.MlirLocation{ loc, loc, loc };
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
    const balance_arg = mlir.oraBlockGetArgument(helper_body, 1);
    const total_arg = mlir.oraBlockGetArgument(helper_body, 2);
    const balances_before = mlir.oraSLoadOpCreate(mlir_ctx, loc, stringRef("balances"), map_ty);
    const balances_after = mlir.oraMapStoreOpCreate(
        mlir_ctx,
        loc,
        mlir.oraOperationGetResult(balances_before, 0),
        user_arg,
        balance_arg,
    );
    const total_store = mlir.oraSStoreOpCreate(mlir_ctx, loc, total_arg, stringRef("total_supply"));
    const ret = mlir.oraReturnOpCreate(mlir_ctx, loc, &[_]mlir.MlirValue{}, 0);
    mlir.oraBlockAppendOwnedOperation(helper_body, balances_before);
    mlir.oraBlockAppendOwnedOperation(helper_body, balances_after);
    mlir.oraBlockAppendOwnedOperation(helper_body, total_store);
    mlir.oraBlockAppendOwnedOperation(helper_body, ret);
    try encoder.registerFunctionOperation(helper);

    const user_op = mlir.oraArithConstantOpCreate(mlir_ctx, loc, i256_ty, mlir.oraIntegerAttrCreateI64FromType(i256_ty, 11));
    const other_op = mlir.oraArithConstantOpCreate(mlir_ctx, loc, i256_ty, mlir.oraIntegerAttrCreateI64FromType(i256_ty, 22));
    const balance_op = mlir.oraArithConstantOpCreate(mlir_ctx, loc, i256_ty, mlir.oraIntegerAttrCreateI64FromType(i256_ty, 99));
    const total_op = mlir.oraArithConstantOpCreate(mlir_ctx, loc, i256_ty, mlir.oraIntegerAttrCreateI64FromType(i256_ty, 1000));
    const seed_total_op = mlir.oraArithConstantOpCreate(mlir_ctx, loc, i256_ty, mlir.oraIntegerAttrCreateI64FromType(i256_ty, 500));
    const user = mlir.oraOperationGetResult(user_op, 0);
    const other = mlir.oraOperationGetResult(other_op, 0);
    const balance = mlir.oraOperationGetResult(balance_op, 0);
    const total = mlir.oraOperationGetResult(total_op, 0);

    const pre_load = mlir.oraSLoadOpCreate(mlir_ctx, loc, stringRef("balances"), map_ty);
    const pre_other_get = mlir.oraMapGetOpCreate(mlir_ctx, loc, mlir.oraOperationGetResult(pre_load, 0), other, i256_ty);
    const pre_other = try encoder.encodeOperation(pre_other_get);
    const pre_user_get = mlir.oraMapGetOpCreate(mlir_ctx, loc, mlir.oraOperationGetResult(pre_load, 0), user, i256_ty);
    const pre_user = try encoder.encodeOperation(pre_user_get);
    _ = try encoder.encodeOperation(mlir.oraSStoreOpCreate(mlir_ctx, loc, mlir.oraOperationGetResult(seed_total_op, 0), stringRef("total_supply")));
    const pre_total_load = mlir.oraSLoadOpCreate(mlir_ctx, loc, stringRef("total_supply"), i256_ty);
    const pre_total = try encoder.encodeOperation(pre_total_load);

    const call = mlir.oraFuncCallOpCreate(
        mlir_ctx,
        loc,
        stringRef("opaqueSetBalanceAndTotal"),
        &[_]mlir.MlirValue{ user, balance, total },
        3,
        &[_]mlir.MlirType{},
        0,
    );
    _ = try encoder.encodeOperation(call);

    try testing.expect(!encoder.isDegraded());
    try testing.expectEqual(@as(usize, 0), encoder.soundnessLosses().len);
    try testing.expectEqual(@as(usize, 0), encoder.precisionNotes().len);

    const post_load = mlir.oraSLoadOpCreate(mlir_ctx, loc, stringRef("balances"), map_ty);
    const post_other_get = mlir.oraMapGetOpCreate(mlir_ctx, loc, mlir.oraOperationGetResult(post_load, 0), other, i256_ty);
    const post_other = try encoder.encodeOperation(post_other_get);
    const post_user_get = mlir.oraMapGetOpCreate(mlir_ctx, loc, mlir.oraOperationGetResult(post_load, 0), user, i256_ty);
    const post_user = try encoder.encodeOperation(post_user_get);
    const post_total_load = mlir.oraSLoadOpCreate(mlir_ctx, loc, stringRef("total_supply"), i256_ty);
    const post_total = try encoder.encodeOperation(post_total_load);

    var preserved_solver = try Solver.init(&z3_ctx, testing.allocator);
    defer preserved_solver.deinit();
    preserved_solver.assert(z3.Z3_mk_not(z3_ctx.ctx, z3.Z3_mk_eq(z3_ctx.ctx, pre_other, post_other)));
    try testing.expectEqual(@as(z3.Z3_lbool, z3.Z3_L_FALSE), preserved_solver.check());

    var changed_solver = try Solver.init(&z3_ctx, testing.allocator);
    defer changed_solver.deinit();
    changed_solver.assert(z3.Z3_mk_not(z3_ctx.ctx, z3.Z3_mk_eq(z3_ctx.ctx, pre_user, post_user)));
    changed_solver.assert(z3.Z3_mk_not(z3_ctx.ctx, z3.Z3_mk_eq(z3_ctx.ctx, pre_total, post_total)));
    try testing.expectEqual(@as(z3.Z3_lbool, z3.Z3_L_TRUE), changed_solver.check());
}

test "known callee opaque modifies metadata fallback preserves disjoint nested map key" {
    var z3_ctx = try Context.init(testing.allocator);
    defer z3_ctx.deinit();

    var encoder = Encoder.init(&z3_ctx, testing.allocator);
    defer encoder.deinit();
    encoder.setVerifyState(true);
    encoder.max_summary_inline_depth = 0;

    const mlir_ctx = mlir.oraContextCreate();
    defer mlir.oraContextDestroy(mlir_ctx);
    loadAllDialects(mlir_ctx);
    _ = mlir.oraDialectRegister(mlir_ctx);

    const loc = mlir.oraLocationUnknownGet(mlir_ctx);
    const i256_ty = mlir.oraIntegerTypeCreate(mlir_ctx, 256);
    const inner_map_ty = mlir.oraMapTypeGet(mlir_ctx, i256_ty, i256_ty);
    const outer_map_ty = mlir.oraMapTypeGet(mlir_ctx, i256_ty, inner_map_ty);

    const modifies_slots = [_]mlir.MlirAttribute{
        mlir.oraStringAttrCreate(mlir_ctx, stringRef("allowances[param#0][param#1]")),
    };
    const helper_attrs = [_]mlir.MlirNamedAttribute{
        namedAttr(mlir_ctx, "sym_name", mlir.oraStringAttrCreate(mlir_ctx, stringRef("opaqueSetAllowance"))),
        namedAttr(mlir_ctx, "ora.effect", mlir.oraStringAttrCreate(mlir_ctx, stringRef("writes"))),
        namedAttr(mlir_ctx, "ora.modifies_slots", mlir.oraArrayAttrCreate(mlir_ctx, modifies_slots.len, &modifies_slots)),
    };
    const helper_param_types = [_]mlir.MlirType{ i256_ty, i256_ty, i256_ty };
    const helper_param_locs = [_]mlir.MlirLocation{ loc, loc, loc };
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
    const owner_arg = mlir.oraBlockGetArgument(helper_body, 0);
    const spender_arg = mlir.oraBlockGetArgument(helper_body, 1);
    const value_arg = mlir.oraBlockGetArgument(helper_body, 2);
    const outer_before = mlir.oraSLoadOpCreate(mlir_ctx, loc, stringRef("allowances"), outer_map_ty);
    const inner_before = mlir.oraMapGetOpCreate(mlir_ctx, loc, mlir.oraOperationGetResult(outer_before, 0), owner_arg, inner_map_ty);
    const inner_after = mlir.oraMapStoreOpCreate(
        mlir_ctx,
        loc,
        mlir.oraOperationGetResult(inner_before, 0),
        spender_arg,
        value_arg,
    );
    const ret = mlir.oraReturnOpCreate(mlir_ctx, loc, &[_]mlir.MlirValue{}, 0);
    mlir.oraBlockAppendOwnedOperation(helper_body, outer_before);
    mlir.oraBlockAppendOwnedOperation(helper_body, inner_before);
    mlir.oraBlockAppendOwnedOperation(helper_body, inner_after);
    mlir.oraBlockAppendOwnedOperation(helper_body, ret);
    try encoder.registerFunctionOperation(helper);

    const owner_op = mlir.oraArithConstantOpCreate(mlir_ctx, loc, i256_ty, mlir.oraIntegerAttrCreateI64FromType(i256_ty, 11));
    const spender_op = mlir.oraArithConstantOpCreate(mlir_ctx, loc, i256_ty, mlir.oraIntegerAttrCreateI64FromType(i256_ty, 22));
    const other_spender_op = mlir.oraArithConstantOpCreate(mlir_ctx, loc, i256_ty, mlir.oraIntegerAttrCreateI64FromType(i256_ty, 33));
    const value_op = mlir.oraArithConstantOpCreate(mlir_ctx, loc, i256_ty, mlir.oraIntegerAttrCreateI64FromType(i256_ty, 99));
    const owner = mlir.oraOperationGetResult(owner_op, 0);
    const spender = mlir.oraOperationGetResult(spender_op, 0);
    const other_spender = mlir.oraOperationGetResult(other_spender_op, 0);
    const value = mlir.oraOperationGetResult(value_op, 0);

    const pre_outer_load = mlir.oraSLoadOpCreate(mlir_ctx, loc, stringRef("allowances"), outer_map_ty);
    const pre_owner_inner = mlir.oraMapGetOpCreate(mlir_ctx, loc, mlir.oraOperationGetResult(pre_outer_load, 0), owner, inner_map_ty);
    const pre_other_get = mlir.oraMapGetOpCreate(mlir_ctx, loc, mlir.oraOperationGetResult(pre_owner_inner, 0), other_spender, i256_ty);
    const pre_other = try encoder.encodeOperation(pre_other_get);
    const pre_spender_get = mlir.oraMapGetOpCreate(mlir_ctx, loc, mlir.oraOperationGetResult(pre_owner_inner, 0), spender, i256_ty);
    const pre_spender = try encoder.encodeOperation(pre_spender_get);

    const call = mlir.oraFuncCallOpCreate(
        mlir_ctx,
        loc,
        stringRef("opaqueSetAllowance"),
        &[_]mlir.MlirValue{ owner, spender, value },
        3,
        &[_]mlir.MlirType{},
        0,
    );
    _ = try encoder.encodeOperation(call);

    try testing.expect(!encoder.isDegraded());
    try testing.expectEqual(@as(usize, 0), encoder.soundnessLosses().len);
    try testing.expectEqual(@as(usize, 0), encoder.precisionNotes().len);

    const post_outer_load = mlir.oraSLoadOpCreate(mlir_ctx, loc, stringRef("allowances"), outer_map_ty);
    const post_owner_inner = mlir.oraMapGetOpCreate(mlir_ctx, loc, mlir.oraOperationGetResult(post_outer_load, 0), owner, inner_map_ty);
    const post_other_get = mlir.oraMapGetOpCreate(mlir_ctx, loc, mlir.oraOperationGetResult(post_owner_inner, 0), other_spender, i256_ty);
    const post_other = try encoder.encodeOperation(post_other_get);
    const post_spender_get = mlir.oraMapGetOpCreate(mlir_ctx, loc, mlir.oraOperationGetResult(post_owner_inner, 0), spender, i256_ty);
    const post_spender = try encoder.encodeOperation(post_spender_get);

    var preserved_solver = try Solver.init(&z3_ctx, testing.allocator);
    defer preserved_solver.deinit();
    preserved_solver.assert(z3.Z3_mk_not(z3_ctx.ctx, z3.Z3_mk_eq(z3_ctx.ctx, pre_other, post_other)));
    try testing.expectEqual(@as(z3.Z3_lbool, z3.Z3_L_FALSE), preserved_solver.check());

    var changed_solver = try Solver.init(&z3_ctx, testing.allocator);
    defer changed_solver.deinit();
    changed_solver.assert(z3.Z3_mk_not(z3_ctx.ctx, z3.Z3_mk_eq(z3_ctx.ctx, pre_spender, post_spender)));
    try testing.expectEqual(@as(z3.Z3_lbool, z3.Z3_L_TRUE), changed_solver.check());
}

test "known callee opaque modifies metadata fallback preserves disjoint struct field" {
    var z3_ctx = try Context.init(testing.allocator);
    defer z3_ctx.deinit();

    var encoder = Encoder.init(&z3_ctx, testing.allocator);
    defer encoder.deinit();
    encoder.setVerifyState(true);
    encoder.max_summary_inline_depth = 0;

    const mlir_ctx = mlir.oraContextCreate();
    defer mlir.oraContextDestroy(mlir_ctx);
    loadAllDialects(mlir_ctx);
    _ = mlir.oraDialectRegister(mlir_ctx);

    const loc = mlir.oraLocationUnknownGet(mlir_ctx);
    const i256_ty = mlir.oraIntegerTypeCreate(mlir_ctx, 256);
    const struct_ty = mlir.oraStructTypeGet(mlir_ctx, stringRef("Config__u256"));

    const struct_decl = mlir.oraStructDeclOpCreate(mlir_ctx, loc, stringRef("Config__u256"));
    const field_name_attrs = [_]mlir.MlirAttribute{
        mlir.oraStringAttrCreate(mlir_ctx, stringRef("owner")),
        mlir.oraStringAttrCreate(mlir_ctx, stringRef("admin")),
    };
    const field_type_attrs = [_]mlir.MlirAttribute{
        mlir.oraTypeAttrCreateFromType(i256_ty),
        mlir.oraTypeAttrCreateFromType(i256_ty),
    };
    mlir.oraOperationSetAttributeByName(struct_decl, stringRef("ora.field_names"), mlir.oraArrayAttrCreate(mlir_ctx, field_name_attrs.len, &field_name_attrs));
    mlir.oraOperationSetAttributeByName(struct_decl, stringRef("ora.field_types"), mlir.oraArrayAttrCreate(mlir_ctx, field_type_attrs.len, &field_type_attrs));
    try encoder.registerStructDeclOperation(struct_decl);

    const modifies_slots = [_]mlir.MlirAttribute{
        mlir.oraStringAttrCreate(mlir_ctx, stringRef("config.owner")),
    };
    const helper_attrs = [_]mlir.MlirNamedAttribute{
        namedAttr(mlir_ctx, "sym_name", mlir.oraStringAttrCreate(mlir_ctx, stringRef("opaqueSetOwner"))),
        namedAttr(mlir_ctx, "ora.effect", mlir.oraStringAttrCreate(mlir_ctx, stringRef("writes"))),
        namedAttr(mlir_ctx, "ora.modifies_slots", mlir.oraArrayAttrCreate(mlir_ctx, modifies_slots.len, &modifies_slots)),
    };
    const helper_param_types = [_]mlir.MlirType{i256_ty};
    const helper_param_locs = [_]mlir.MlirLocation{loc};
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
    const owner_arg = mlir.oraBlockGetArgument(helper_body, 0);
    const config_before = mlir.oraSLoadOpCreate(mlir_ctx, loc, stringRef("config"), struct_ty);
    const config_after = mlir.oraStructFieldUpdateOpCreate(
        mlir_ctx,
        loc,
        mlir.oraOperationGetResult(config_before, 0),
        stringRef("owner"),
        owner_arg,
    );
    const config_store = mlir.oraSStoreOpCreate(mlir_ctx, loc, mlir.oraOperationGetResult(config_after, 0), stringRef("config"));
    const ret = mlir.oraReturnOpCreate(mlir_ctx, loc, &[_]mlir.MlirValue{}, 0);
    mlir.oraBlockAppendOwnedOperation(helper_body, config_before);
    mlir.oraBlockAppendOwnedOperation(helper_body, config_after);
    mlir.oraBlockAppendOwnedOperation(helper_body, config_store);
    mlir.oraBlockAppendOwnedOperation(helper_body, ret);
    try encoder.registerFunctionOperation(helper);

    const owner_seed = mlir.oraOperationGetResult(mlir.oraArithConstantOpCreate(
        mlir_ctx,
        loc,
        i256_ty,
        mlir.oraIntegerAttrCreateI64FromType(i256_ty, 11),
    ), 0);
    const admin_seed = mlir.oraOperationGetResult(mlir.oraArithConstantOpCreate(
        mlir_ctx,
        loc,
        i256_ty,
        mlir.oraIntegerAttrCreateI64FromType(i256_ty, 22),
    ), 0);
    const next_owner = mlir.oraOperationGetResult(mlir.oraArithConstantOpCreate(
        mlir_ctx,
        loc,
        i256_ty,
        mlir.oraIntegerAttrCreateI64FromType(i256_ty, 99),
    ), 0);

    const initial_fields = [_]mlir.MlirValue{ owner_seed, admin_seed };
    const initial_config = mlir.oraStructInstantiateOpCreate(mlir_ctx, loc, stringRef("Config__u256"), &initial_fields, initial_fields.len, struct_ty);
    _ = try encoder.encodeOperation(mlir.oraSStoreOpCreate(mlir_ctx, loc, mlir.oraOperationGetResult(initial_config, 0), stringRef("config")));

    const pre_load = mlir.oraSLoadOpCreate(mlir_ctx, loc, stringRef("config"), struct_ty);
    const pre_owner_extract = mlir.oraStructFieldExtractOpCreate(mlir_ctx, loc, mlir.oraOperationGetResult(pre_load, 0), stringRef("owner"), i256_ty);
    const pre_admin_extract = mlir.oraStructFieldExtractOpCreate(mlir_ctx, loc, mlir.oraOperationGetResult(pre_load, 0), stringRef("admin"), i256_ty);
    const pre_owner = try encoder.encodeOperation(pre_owner_extract);
    const pre_admin = try encoder.encodeOperation(pre_admin_extract);

    const call = mlir.oraFuncCallOpCreate(
        mlir_ctx,
        loc,
        stringRef("opaqueSetOwner"),
        &[_]mlir.MlirValue{next_owner},
        1,
        &[_]mlir.MlirType{},
        0,
    );
    _ = try encoder.encodeOperation(call);

    try testing.expect(!encoder.isDegraded());
    try testing.expectEqual(@as(usize, 0), encoder.soundnessLosses().len);
    try testing.expectEqual(@as(usize, 0), encoder.precisionNotes().len);

    const post_load = mlir.oraSLoadOpCreate(mlir_ctx, loc, stringRef("config"), struct_ty);
    const post_owner_extract = mlir.oraStructFieldExtractOpCreate(mlir_ctx, loc, mlir.oraOperationGetResult(post_load, 0), stringRef("owner"), i256_ty);
    const post_admin_extract = mlir.oraStructFieldExtractOpCreate(mlir_ctx, loc, mlir.oraOperationGetResult(post_load, 0), stringRef("admin"), i256_ty);
    const post_owner = try encoder.encodeOperation(post_owner_extract);
    const post_admin = try encoder.encodeOperation(post_admin_extract);

    var preserved_solver = try Solver.init(&z3_ctx, testing.allocator);
    defer preserved_solver.deinit();
    preserved_solver.assert(z3.Z3_mk_not(z3_ctx.ctx, z3.Z3_mk_eq(z3_ctx.ctx, pre_admin, post_admin)));
    try testing.expectEqual(@as(z3.Z3_lbool, z3.Z3_L_FALSE), preserved_solver.check());

    var changed_solver = try Solver.init(&z3_ctx, testing.allocator);
    defer changed_solver.deinit();
    changed_solver.assert(z3.Z3_mk_not(z3_ctx.ctx, z3.Z3_mk_eq(z3_ctx.ctx, pre_owner, post_owner)));
    try testing.expectEqual(@as(z3.Z3_lbool, z3.Z3_L_TRUE), changed_solver.check());
}

test "known callee covered root opaque fallback records precision note when indexed path cannot apply" {
    var z3_ctx = try Context.init(testing.allocator);
    defer z3_ctx.deinit();

    var encoder = Encoder.init(&z3_ctx, testing.allocator);
    defer encoder.deinit();
    encoder.setVerifyState(true);
    encoder.max_summary_inline_depth = 0;

    const mlir_ctx = mlir.oraContextCreate();
    defer mlir.oraContextDestroy(mlir_ctx);
    loadAllDialects(mlir_ctx);
    _ = mlir.oraDialectRegister(mlir_ctx);

    const loc = mlir.oraLocationUnknownGet(mlir_ctx);
    const i256_ty = mlir.oraIntegerTypeCreate(mlir_ctx, 256);

    const modifies_slots = [_]mlir.MlirAttribute{
        mlir.oraStringAttrCreate(mlir_ctx, stringRef("counter[param#0]")),
    };
    const helper_attrs = [_]mlir.MlirNamedAttribute{
        namedAttr(mlir_ctx, "sym_name", mlir.oraStringAttrCreate(mlir_ctx, stringRef("opaqueSetCounterWithBadPathShape"))),
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
    const value_arg = mlir.oraBlockGetArgument(helper_body, 1);
    const store = mlir.oraSStoreOpCreate(mlir_ctx, loc, value_arg, stringRef("counter"));
    const ret = mlir.oraReturnOpCreate(mlir_ctx, loc, &[_]mlir.MlirValue{}, 0);
    mlir.oraBlockAppendOwnedOperation(helper_body, store);
    mlir.oraBlockAppendOwnedOperation(helper_body, ret);
    try encoder.registerFunctionOperation(helper);

    const user_op = mlir.oraArithConstantOpCreate(mlir_ctx, loc, i256_ty, mlir.oraIntegerAttrCreateI64FromType(i256_ty, 11));
    const value_op = mlir.oraArithConstantOpCreate(mlir_ctx, loc, i256_ty, mlir.oraIntegerAttrCreateI64FromType(i256_ty, 99));
    const user = mlir.oraOperationGetResult(user_op, 0);
    const value = mlir.oraOperationGetResult(value_op, 0);

    const seed_op = mlir.oraArithConstantOpCreate(mlir_ctx, loc, i256_ty, mlir.oraIntegerAttrCreateI64FromType(i256_ty, 0));
    _ = try encoder.encodeOperation(mlir.oraSStoreOpCreate(mlir_ctx, loc, mlir.oraOperationGetResult(seed_op, 0), stringRef("counter")));

    const caller_attrs = [_]mlir.MlirNamedAttribute{
        namedAttr(mlir_ctx, "sym_name", mlir.oraStringAttrCreate(mlir_ctx, stringRef("callerWithBadPathShape"))),
    };
    const caller = mlir.oraFuncFuncOpCreate(
        mlir_ctx,
        loc,
        &caller_attrs,
        caller_attrs.len,
        &[_]mlir.MlirType{},
        &[_]mlir.MlirLocation{},
        0,
    );
    const caller_body = mlir.oraFuncOpGetBodyBlock(caller);
    mlir.oraBlockAppendOwnedOperation(caller_body, user_op);
    mlir.oraBlockAppendOwnedOperation(caller_body, value_op);

    const call = mlir.oraFuncCallOpCreate(
        mlir_ctx,
        loc,
        stringRef("opaqueSetCounterWithBadPathShape"),
        &[_]mlir.MlirValue{ user, value },
        2,
        &[_]mlir.MlirType{},
        0,
    );
    mlir.oraBlockAppendOwnedOperation(caller_body, call);
    _ = try encoder.encodeOperation(call);

    try testing.expect(!encoder.isDegraded());
    try testing.expectEqual(@as(usize, 0), encoder.soundnessLosses().len);

    const notes = encoder.precisionNotes();
    try testing.expectEqual(@as(usize, 1), notes.len);
    try testing.expectEqual(Encoder.PrecisionNoteKind.path_precise_modifies_fallback_unavailable, notes[0]);

    const events = encoder.precisionNoteEvents();
    try testing.expectEqual(@as(usize, 1), events.len);
    try testing.expectEqual(Encoder.PrecisionNoteKind.path_precise_modifies_fallback_unavailable, events[0].kind);
    try testing.expectEqualStrings("callerWithBadPathShape", events[0].function_name.?);
    try testing.expectEqualStrings("opaqueSetCounterWithBadPathShape", events[0].callee.?);
    try testing.expectEqualStrings("counter", events[0].storage_root.?);
    try testing.expectEqualStrings("counter[param#0]", events[0].declared_path.?);
    try testing.expect(events[0].location != null);
}

test "known callee covered root opaque fallback records precision note for mixed indexed field metadata" {
    var z3_ctx = try Context.init(testing.allocator);
    defer z3_ctx.deinit();

    var encoder = Encoder.init(&z3_ctx, testing.allocator);
    defer encoder.deinit();
    encoder.setVerifyState(true);
    encoder.max_summary_inline_depth = 0;

    const mlir_ctx = mlir.oraContextCreate();
    defer mlir.oraContextDestroy(mlir_ctx);
    loadAllDialects(mlir_ctx);
    _ = mlir.oraDialectRegister(mlir_ctx);

    const loc = mlir.oraLocationUnknownGet(mlir_ctx);
    const i256_ty = mlir.oraIntegerTypeCreate(mlir_ctx, 256);

    const modifies_slots = [_]mlir.MlirAttribute{
        mlir.oraStringAttrCreate(mlir_ctx, stringRef("users[param#0].balance")),
    };
    const helper_attrs = [_]mlir.MlirNamedAttribute{
        namedAttr(mlir_ctx, "sym_name", mlir.oraStringAttrCreate(mlir_ctx, stringRef("opaqueSetUsersWithMixedPathShape"))),
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
    const value_arg = mlir.oraBlockGetArgument(helper_body, 1);
    const store = mlir.oraSStoreOpCreate(mlir_ctx, loc, value_arg, stringRef("users"));
    const ret = mlir.oraReturnOpCreate(mlir_ctx, loc, &[_]mlir.MlirValue{}, 0);
    mlir.oraBlockAppendOwnedOperation(helper_body, store);
    mlir.oraBlockAppendOwnedOperation(helper_body, ret);
    try encoder.registerFunctionOperation(helper);

    const user_op = mlir.oraArithConstantOpCreate(mlir_ctx, loc, i256_ty, mlir.oraIntegerAttrCreateI64FromType(i256_ty, 11));
    const value_op = mlir.oraArithConstantOpCreate(mlir_ctx, loc, i256_ty, mlir.oraIntegerAttrCreateI64FromType(i256_ty, 99));
    const user = mlir.oraOperationGetResult(user_op, 0);
    const value = mlir.oraOperationGetResult(value_op, 0);

    const seed_op = mlir.oraArithConstantOpCreate(mlir_ctx, loc, i256_ty, mlir.oraIntegerAttrCreateI64FromType(i256_ty, 0));
    _ = try encoder.encodeOperation(mlir.oraSStoreOpCreate(mlir_ctx, loc, mlir.oraOperationGetResult(seed_op, 0), stringRef("users")));

    const call = mlir.oraFuncCallOpCreate(
        mlir_ctx,
        loc,
        stringRef("opaqueSetUsersWithMixedPathShape"),
        &[_]mlir.MlirValue{ user, value },
        2,
        &[_]mlir.MlirType{},
        0,
    );
    _ = try encoder.encodeOperation(call);

    try testing.expect(!encoder.isDegraded());
    try testing.expectEqual(@as(usize, 0), encoder.soundnessLosses().len);

    const notes = encoder.precisionNotes();
    try testing.expectEqual(@as(usize, 1), notes.len);
    try testing.expectEqual(Encoder.PrecisionNoteKind.path_precise_modifies_fallback_unavailable, notes[0]);
}

test "known callee conservative write metadata superset does not degrade" {
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
    const i256_ty = mlir.oraIntegerTypeCreate(mlir_ctx, 256);
    const write_slots = [_]mlir.MlirAttribute{
        mlir.oraStringAttrCreate(mlir_ctx, stringRef("counter")),
        mlir.oraStringAttrCreate(mlir_ctx, stringRef("secret")),
    };
    const helper_attrs = [_]mlir.MlirNamedAttribute{
        namedAttr(mlir_ctx, "sym_name", mlir.oraStringAttrCreate(mlir_ctx, stringRef("writerWithSupersetSlots"))),
        namedAttr(mlir_ctx, "ora.effect", mlir.oraStringAttrCreate(mlir_ctx, stringRef("writes"))),
        namedAttr(mlir_ctx, "ora.write_slots", mlir.oraArrayAttrCreate(mlir_ctx, write_slots.len, &write_slots)),
    };
    const helper = mlir.oraFuncFuncOpCreate(mlir_ctx, loc, &helper_attrs, helper_attrs.len, &[_]mlir.MlirType{}, &[_]mlir.MlirLocation{}, 0);
    const helper_body = mlir.oraFuncOpGetBodyBlock(helper);
    const value_attr = mlir.oraIntegerAttrCreateI64FromType(i256_ty, 9);
    const value_op = mlir.oraArithConstantOpCreate(mlir_ctx, loc, i256_ty, value_attr);
    const store = mlir.oraSStoreOpCreate(mlir_ctx, loc, mlir.oraOperationGetResult(value_op, 0), stringRef("counter"));
    const ret = mlir.oraReturnOpCreate(mlir_ctx, loc, &[_]mlir.MlirValue{}, 0);
    mlir.oraBlockAppendOwnedOperation(helper_body, value_op);
    mlir.oraBlockAppendOwnedOperation(helper_body, store);
    mlir.oraBlockAppendOwnedOperation(helper_body, ret);
    try encoder.registerFunctionOperation(helper);

    const seed_attr = mlir.oraIntegerAttrCreateI64FromType(i256_ty, 0);
    const seed_op = mlir.oraArithConstantOpCreate(mlir_ctx, loc, i256_ty, seed_attr);
    const seed = mlir.oraOperationGetResult(seed_op, 0);
    _ = try encoder.encodeOperation(mlir.oraSStoreOpCreate(mlir_ctx, loc, seed, stringRef("counter")));
    _ = try encoder.encodeOperation(mlir.oraSStoreOpCreate(mlir_ctx, loc, seed, stringRef("secret")));

    const call = mlir.oraFuncCallOpCreate(
        mlir_ctx,
        loc,
        stringRef("writerWithSupersetSlots"),
        &[_]mlir.MlirValue{},
        0,
        &[_]mlir.MlirType{},
        0,
    );
    _ = try encoder.encodeOperation(call);

    try testing.expect(!encoder.isDegraded());
    try testing.expectEqual(@as(usize, 0), encoder.soundnessLosses().len);
}

test "known callee transient write metadata uses canonical transient prefix" {
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
    const i256_ty = mlir.oraIntegerTypeCreate(mlir_ctx, 256);
    const write_slots = [_]mlir.MlirAttribute{
        mlir.oraStringAttrCreate(mlir_ctx, stringRef("transient:pending")),
    };
    const helper_attrs = [_]mlir.MlirNamedAttribute{
        namedAttr(mlir_ctx, "sym_name", mlir.oraStringAttrCreate(mlir_ctx, stringRef("transientWriterWithSummary"))),
        namedAttr(mlir_ctx, "ora.effect", mlir.oraStringAttrCreate(mlir_ctx, stringRef("writes"))),
        namedAttr(mlir_ctx, "ora.write_slots", mlir.oraArrayAttrCreate(mlir_ctx, write_slots.len, &write_slots)),
    };
    const helper = mlir.oraFuncFuncOpCreate(mlir_ctx, loc, &helper_attrs, helper_attrs.len, &[_]mlir.MlirType{}, &[_]mlir.MlirLocation{}, 0);
    const helper_body = mlir.oraFuncOpGetBodyBlock(helper);
    const value_attr = mlir.oraIntegerAttrCreateI64FromType(i256_ty, 13);
    const value_op = mlir.oraArithConstantOpCreate(mlir_ctx, loc, i256_ty, value_attr);
    const store = mlir.oraTStoreOpCreate(mlir_ctx, loc, mlir.oraOperationGetResult(value_op, 0), stringRef("pending"));
    const ret = mlir.oraReturnOpCreate(mlir_ctx, loc, &[_]mlir.MlirValue{}, 0);
    mlir.oraBlockAppendOwnedOperation(helper_body, value_op);
    mlir.oraBlockAppendOwnedOperation(helper_body, store);
    mlir.oraBlockAppendOwnedOperation(helper_body, ret);
    try encoder.registerFunctionOperation(helper);

    const seed_attr = mlir.oraIntegerAttrCreateI64FromType(i256_ty, 0);
    const seed_op = mlir.oraArithConstantOpCreate(mlir_ctx, loc, i256_ty, seed_attr);
    _ = try encoder.encodeOperation(mlir.oraTStoreOpCreate(mlir_ctx, loc, mlir.oraOperationGetResult(seed_op, 0), stringRef("pending")));

    const call = mlir.oraFuncCallOpCreate(
        mlir_ctx,
        loc,
        stringRef("transientWriterWithSummary"),
        &[_]mlir.MlirValue{},
        0,
        &[_]mlir.MlirType{},
        0,
    );
    _ = try encoder.encodeOperation(call);

    try testing.expect(!encoder.isDegraded());
    try testing.expectEqual(@as(usize, 0), encoder.soundnessLosses().len);
}

test "known callee recursive write metadata omissions are still cross-checked" {
    var z3_ctx = try Context.init(testing.allocator);
    defer z3_ctx.deinit();

    var encoder = Encoder.init(&z3_ctx, testing.allocator);
    defer encoder.deinit();
    encoder.setVerifyState(true);
    encoder.max_summary_inline_depth = 0;

    const mlir_ctx = mlir.oraContextCreate();
    defer mlir.oraContextDestroy(mlir_ctx);
    loadAllDialects(mlir_ctx);
    _ = mlir.oraDialectRegister(mlir_ctx);

    const loc = mlir.oraLocationUnknownGet(mlir_ctx);
    const i256_ty = mlir.oraIntegerTypeCreate(mlir_ctx, 256);
    const caller_slots = [_]mlir.MlirAttribute{
        mlir.oraStringAttrCreate(mlir_ctx, stringRef("callee_secret")),
    };
    const caller_attrs = [_]mlir.MlirNamedAttribute{
        namedAttr(mlir_ctx, "sym_name", mlir.oraStringAttrCreate(mlir_ctx, stringRef("recursiveCaller"))),
        namedAttr(mlir_ctx, "ora.effect", mlir.oraStringAttrCreate(mlir_ctx, stringRef("writes"))),
        namedAttr(mlir_ctx, "ora.write_slots", mlir.oraArrayAttrCreate(mlir_ctx, caller_slots.len, &caller_slots)),
    };
    const callee_slots = [_]mlir.MlirAttribute{};
    const callee_attrs = [_]mlir.MlirNamedAttribute{
        namedAttr(mlir_ctx, "sym_name", mlir.oraStringAttrCreate(mlir_ctx, stringRef("recursiveCallee"))),
        namedAttr(mlir_ctx, "ora.effect", mlir.oraStringAttrCreate(mlir_ctx, stringRef("writes"))),
        namedAttr(mlir_ctx, "ora.write_slots", mlir.oraArrayAttrCreate(mlir_ctx, callee_slots.len, &callee_slots)),
    };

    const caller = mlir.oraFuncFuncOpCreate(mlir_ctx, loc, &caller_attrs, caller_attrs.len, &[_]mlir.MlirType{}, &[_]mlir.MlirLocation{}, 0);
    const caller_body = mlir.oraFuncOpGetBodyBlock(caller);
    const caller_value_attr = mlir.oraIntegerAttrCreateI64FromType(i256_ty, 21);
    const caller_value_op = mlir.oraArithConstantOpCreate(mlir_ctx, loc, i256_ty, caller_value_attr);
    const caller_store = mlir.oraSStoreOpCreate(mlir_ctx, loc, mlir.oraOperationGetResult(caller_value_op, 0), stringRef("caller_secret"));
    const call_callee = mlir.oraFuncCallOpCreate(mlir_ctx, loc, stringRef("recursiveCallee"), &[_]mlir.MlirValue{}, 0, &[_]mlir.MlirType{}, 0);
    const caller_ret = mlir.oraReturnOpCreate(mlir_ctx, loc, &[_]mlir.MlirValue{}, 0);
    mlir.oraBlockAppendOwnedOperation(caller_body, caller_value_op);
    mlir.oraBlockAppendOwnedOperation(caller_body, caller_store);
    mlir.oraBlockAppendOwnedOperation(caller_body, call_callee);
    mlir.oraBlockAppendOwnedOperation(caller_body, caller_ret);

    const callee = mlir.oraFuncFuncOpCreate(mlir_ctx, loc, &callee_attrs, callee_attrs.len, &[_]mlir.MlirType{}, &[_]mlir.MlirLocation{}, 0);
    const callee_body = mlir.oraFuncOpGetBodyBlock(callee);
    const callee_value_attr = mlir.oraIntegerAttrCreateI64FromType(i256_ty, 34);
    const callee_value_op = mlir.oraArithConstantOpCreate(mlir_ctx, loc, i256_ty, callee_value_attr);
    const callee_store = mlir.oraSStoreOpCreate(mlir_ctx, loc, mlir.oraOperationGetResult(callee_value_op, 0), stringRef("callee_secret"));
    const call_caller = mlir.oraFuncCallOpCreate(mlir_ctx, loc, stringRef("recursiveCaller"), &[_]mlir.MlirValue{}, 0, &[_]mlir.MlirType{}, 0);
    const callee_ret = mlir.oraReturnOpCreate(mlir_ctx, loc, &[_]mlir.MlirValue{}, 0);
    mlir.oraBlockAppendOwnedOperation(callee_body, callee_value_op);
    mlir.oraBlockAppendOwnedOperation(callee_body, callee_store);
    mlir.oraBlockAppendOwnedOperation(callee_body, call_caller);
    mlir.oraBlockAppendOwnedOperation(callee_body, callee_ret);

    try encoder.registerFunctionOperation(caller);
    try encoder.registerFunctionOperation(callee);

    const seed_attr = mlir.oraIntegerAttrCreateI64FromType(i256_ty, 0);
    const seed_op = mlir.oraArithConstantOpCreate(mlir_ctx, loc, i256_ty, seed_attr);
    const seed = mlir.oraOperationGetResult(seed_op, 0);
    _ = try encoder.encodeOperation(mlir.oraSStoreOpCreate(mlir_ctx, loc, seed, stringRef("caller_secret")));
    _ = try encoder.encodeOperation(mlir.oraSStoreOpCreate(mlir_ctx, loc, seed, stringRef("callee_secret")));

    const call = mlir.oraFuncCallOpCreate(
        mlir_ctx,
        loc,
        stringRef("recursiveCaller"),
        &[_]mlir.MlirValue{},
        0,
        &[_]mlir.MlirType{},
        0,
    );
    _ = try encoder.encodeOperation(call);

    var omission_count: usize = 0;
    for (encoder.degradationReasons()) |reason| {
        if (std.mem.eql(u8, reason, "ora.write_slots metadata omitted a body write")) {
            omission_count += 1;
        }
    }
    try testing.expect(omission_count >= 2);
}

test "known callee read metadata keeps unresolved body calls from widening the read set" {
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
    const i256_ty = mlir.oraIntegerTypeCreate(mlir_ctx, 256);
    const read_slots = [_]mlir.MlirAttribute{
        mlir.oraStringAttrCreate(mlir_ctx, stringRef("counter")),
    };
    const helper_attrs = [_]mlir.MlirNamedAttribute{
        namedAttr(mlir_ctx, "sym_name", mlir.oraStringAttrCreate(mlir_ctx, stringRef("readerWithSummary"))),
        namedAttr(mlir_ctx, "ora.effect", mlir.oraStringAttrCreate(mlir_ctx, stringRef("reads"))),
        namedAttr(mlir_ctx, "ora.read_slots", mlir.oraArrayAttrCreate(mlir_ctx, read_slots.len, &read_slots)),
    };
    const result_types = [_]mlir.MlirType{i256_ty};
    const result_locs = [_]mlir.MlirLocation{loc};
    const helper = mlir.oraFuncFuncOpCreate(mlir_ctx, loc, &helper_attrs, helper_attrs.len, &result_types, &result_locs, result_types.len);
    const helper_body = mlir.oraFuncOpGetBodyBlock(helper);
    const known_load = mlir.oraSLoadOpCreate(mlir_ctx, loc, stringRef("counter"), i256_ty);
    const unresolved_call = mlir.oraFuncCallOpCreate(
        mlir_ctx,
        loc,
        stringRef("opaqueReader"),
        &[_]mlir.MlirValue{},
        0,
        &[_]mlir.MlirType{},
        0,
    );
    const ret_vals = [_]mlir.MlirValue{mlir.oraOperationGetResult(known_load, 0)};
    const ret = mlir.oraReturnOpCreate(mlir_ctx, loc, &ret_vals, ret_vals.len);
    mlir.oraBlockAppendOwnedOperation(helper_body, known_load);
    mlir.oraBlockAppendOwnedOperation(helper_body, unresolved_call);
    mlir.oraBlockAppendOwnedOperation(helper_body, ret);
    try encoder.registerFunctionOperation(helper);

    const seed_attr = mlir.oraIntegerAttrCreateI64FromType(i256_ty, 17);
    const seed_op = mlir.oraArithConstantOpCreate(mlir_ctx, loc, i256_ty, seed_attr);
    const seed = mlir.oraOperationGetResult(seed_op, 0);
    _ = try encoder.encodeOperation(mlir.oraSStoreOpCreate(mlir_ctx, loc, seed, stringRef("counter")));

    const call = mlir.oraFuncCallOpCreate(
        mlir_ctx,
        loc,
        stringRef("readerWithSummary"),
        &[_]mlir.MlirValue{},
        0,
        &result_types,
        result_types.len,
    );
    const loaded = try encoder.encodeOperation(call);
    try testing.expect(!encoder.isDegraded());

    const expected = try encoder.encodeValue(seed);
    var solver = try Solver.init(&z3_ctx, testing.allocator);
    defer solver.deinit();
    solver.assert(z3.Z3_mk_not(z3_ctx.ctx, z3.Z3_mk_eq(z3_ctx.ctx, loaded, expected)));
    try testing.expectEqual(@as(z3.Z3_lbool, z3.Z3_L_FALSE), solver.check());
}

test "known callee read metadata omission degrades and widens read set" {
    var z3_ctx = try Context.init(testing.allocator);
    defer z3_ctx.deinit();

    var encoder = Encoder.init(&z3_ctx, testing.allocator);
    defer encoder.deinit();
    encoder.setVerifyState(true);
    encoder.max_summary_inline_depth = 0;

    const mlir_ctx = mlir.oraContextCreate();
    defer mlir.oraContextDestroy(mlir_ctx);
    loadAllDialects(mlir_ctx);
    _ = mlir.oraDialectRegister(mlir_ctx);

    const loc = mlir.oraLocationUnknownGet(mlir_ctx);
    const i256_ty = mlir.oraIntegerTypeCreate(mlir_ctx, 256);
    const read_slots = [_]mlir.MlirAttribute{
        mlir.oraStringAttrCreate(mlir_ctx, stringRef("counter")),
    };
    const helper_attrs = [_]mlir.MlirNamedAttribute{
        namedAttr(mlir_ctx, "sym_name", mlir.oraStringAttrCreate(mlir_ctx, stringRef("readerWithOmittedSlot"))),
        namedAttr(mlir_ctx, "ora.effect", mlir.oraStringAttrCreate(mlir_ctx, stringRef("reads"))),
        namedAttr(mlir_ctx, "ora.read_slots", mlir.oraArrayAttrCreate(mlir_ctx, read_slots.len, &read_slots)),
    };
    const result_types = [_]mlir.MlirType{i256_ty};
    const result_locs = [_]mlir.MlirLocation{loc};
    const helper = mlir.oraFuncFuncOpCreate(mlir_ctx, loc, &helper_attrs, helper_attrs.len, &result_types, &result_locs, result_types.len);
    const helper_body = mlir.oraFuncOpGetBodyBlock(helper);
    const omitted_load = mlir.oraSLoadOpCreate(mlir_ctx, loc, stringRef("secret"), i256_ty);
    const ret_vals = [_]mlir.MlirValue{mlir.oraOperationGetResult(omitted_load, 0)};
    const ret = mlir.oraReturnOpCreate(mlir_ctx, loc, &ret_vals, ret_vals.len);
    mlir.oraBlockAppendOwnedOperation(helper_body, omitted_load);
    mlir.oraBlockAppendOwnedOperation(helper_body, ret);
    try encoder.registerFunctionOperation(helper);

    const seed_attr = mlir.oraIntegerAttrCreateI64FromType(i256_ty, 11);
    const seed_op = mlir.oraArithConstantOpCreate(mlir_ctx, loc, i256_ty, seed_attr);
    _ = try encoder.encodeOperation(mlir.oraSStoreOpCreate(mlir_ctx, loc, mlir.oraOperationGetResult(seed_op, 0), stringRef("secret")));

    const call = mlir.oraFuncCallOpCreate(
        mlir_ctx,
        loc,
        stringRef("readerWithOmittedSlot"),
        &[_]mlir.MlirValue{},
        0,
        &result_types,
        result_types.len,
    );
    _ = try encoder.encodeOperation(call);

    try testing.expect(encoder.isDegraded());
    var found_loss = false;
    for (encoder.soundnessLosses()) |loss| {
        if (loss == .inexact_call_summary) {
            found_loss = true;
            break;
        }
    }
    try testing.expect(found_loss);

    var found_reason = false;
    for (encoder.degradationReasons()) |reason| {
        if (std.mem.eql(u8, reason, "ora.read_slots metadata omitted a body read")) {
            found_reason = true;
            break;
        }
    }
    try testing.expect(found_reason);
}

test "known callee nested map write set stays exact" {
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
    const i256_ty = mlir.oraIntegerTypeCreate(mlir_ctx, 256);
    const inner_map_ty = mlir.oraMapTypeGet(mlir_ctx, i256_ty, i256_ty);
    const outer_map_ty = mlir.oraMapTypeGet(mlir_ctx, i256_ty, inner_map_ty);

    const helper_attrs = [_]mlir.MlirNamedAttribute{
        namedAttr(mlir_ctx, "sym_name", mlir.oraStringAttrCreate(mlir_ctx, stringRef("writeAllowance"))),
        namedAttr(mlir_ctx, "ora.effect", mlir.oraStringAttrCreate(mlir_ctx, stringRef("writes"))),
    };
    const helper_param_types = [_]mlir.MlirType{ i256_ty, i256_ty, i256_ty };
    const helper_param_locs = [_]mlir.MlirLocation{ loc, loc, loc };
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
    const owner = mlir.oraBlockGetArgument(helper_body, 0);
    const spender = mlir.oraBlockGetArgument(helper_body, 1);
    const value = mlir.oraBlockGetArgument(helper_body, 2);

    const outer_before = mlir.oraSLoadOpCreate(mlir_ctx, loc, stringRef("allowances"), outer_map_ty);
    const outer_before_value = mlir.oraOperationGetResult(outer_before, 0);
    const inner_before = mlir.oraMapGetOpCreate(mlir_ctx, loc, outer_before_value, owner, inner_map_ty);
    const inner_before_value = mlir.oraOperationGetResult(inner_before, 0);
    const inner_after = mlir.oraMapStoreOpCreate(mlir_ctx, loc, inner_before_value, spender, value);
    const ret = mlir.oraReturnOpCreate(mlir_ctx, loc, &[_]mlir.MlirValue{}, 0);

    mlir.oraBlockAppendOwnedOperation(helper_body, outer_before);
    mlir.oraBlockAppendOwnedOperation(helper_body, inner_before);
    mlir.oraBlockAppendOwnedOperation(helper_body, inner_after);
    mlir.oraBlockAppendOwnedOperation(helper_body, ret);

    try encoder.registerFunctionOperation(helper);

    const owner_attr = mlir.oraIntegerAttrCreateI64FromType(i256_ty, 11);
    const spender_attr = mlir.oraIntegerAttrCreateI64FromType(i256_ty, 22);
    const value_attr = mlir.oraIntegerAttrCreateI64FromType(i256_ty, 99);
    const owner_op = mlir.oraArithConstantOpCreate(mlir_ctx, loc, i256_ty, owner_attr);
    const spender_op = mlir.oraArithConstantOpCreate(mlir_ctx, loc, i256_ty, spender_attr);
    const value_op = mlir.oraArithConstantOpCreate(mlir_ctx, loc, i256_ty, value_attr);
    const owner_val = mlir.oraOperationGetResult(owner_op, 0);
    const spender_val = mlir.oraOperationGetResult(spender_op, 0);
    const value_val = mlir.oraOperationGetResult(value_op, 0);

    const call = mlir.oraFuncCallOpCreate(
        mlir_ctx,
        loc,
        stringRef("writeAllowance"),
        &[_]mlir.MlirValue{ owner_val, spender_val, value_val },
        3,
        &[_]mlir.MlirType{},
        0,
    );
    _ = try encoder.encodeOperation(call);

    try testing.expect(!encoder.isDegraded());

    const outer_after_load = mlir.oraSLoadOpCreate(mlir_ctx, loc, stringRef("allowances"), outer_map_ty);
    const inner_after_load = mlir.oraMapGetOpCreate(mlir_ctx, loc, mlir.oraOperationGetResult(outer_after_load, 0), owner_val, inner_map_ty);
    const final_get = mlir.oraMapGetOpCreate(mlir_ctx, loc, mlir.oraOperationGetResult(inner_after_load, 0), spender_val, i256_ty);
    const loaded = try encoder.encodeOperation(final_get);
    const expected = try encoder.encodeValue(value_val);

    var solver = try Solver.init(&z3_ctx, testing.allocator);
    defer solver.deinit();
    solver.assert(z3.Z3_mk_not(z3_ctx.ctx, z3.Z3_mk_eq(z3_ctx.ctx, loaded, expected)));
    try testing.expectEqual(@as(z3.Z3_lbool, z3.Z3_L_FALSE), solver.check());
}

test "known callee loop-carried nested map write set stays exact" {
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
    const index_ty = mlir.oraIndexTypeCreate(mlir_ctx);
    const i256_ty = mlir.oraIntegerTypeCreate(mlir_ctx, 256);
    const inner_map_ty = mlir.oraMapTypeGet(mlir_ctx, i256_ty, i256_ty);
    const outer_map_ty = mlir.oraMapTypeGet(mlir_ctx, i256_ty, inner_map_ty);

    const helper_attrs = [_]mlir.MlirNamedAttribute{
        namedAttr(mlir_ctx, "sym_name", mlir.oraStringAttrCreate(mlir_ctx, stringRef("writeAllowanceViaLoop"))),
        namedAttr(mlir_ctx, "ora.effect", mlir.oraStringAttrCreate(mlir_ctx, stringRef("writes"))),
    };
    const helper_param_types = [_]mlir.MlirType{ i256_ty, i256_ty, i256_ty };
    const helper_param_locs = [_]mlir.MlirLocation{ loc, loc, loc };
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
    const owner = mlir.oraBlockGetArgument(helper_body, 0);
    const spender = mlir.oraBlockGetArgument(helper_body, 1);
    const value = mlir.oraBlockGetArgument(helper_body, 2);

    const zero_idx = mlir.oraOperationGetResult(mlir.oraArithConstantOpCreate(
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
    const outer_before = mlir.oraSLoadOpCreate(mlir_ctx, loc, stringRef("allowances"), outer_map_ty);
    const outer_before_value = mlir.oraOperationGetResult(outer_before, 0);
    const inner_before = mlir.oraMapGetOpCreate(mlir_ctx, loc, outer_before_value, owner, inner_map_ty);
    const inner_before_value = mlir.oraOperationGetResult(inner_before, 0);
    const loop = mlir.oraScfForOpCreate(
        mlir_ctx,
        loc,
        zero_idx,
        one_idx,
        one_idx,
        &[_]mlir.MlirValue{inner_before_value},
        1,
        false,
    );
    const loop_body = mlir.oraScfForOpGetBodyBlock(loop);
    const carried_inner = mlir.oraBlockGetArgument(loop_body, 1);
    const inner_after = mlir.oraMapStoreOpCreate(mlir_ctx, loc, carried_inner, spender, value);
    mlir.oraBlockAppendOwnedOperation(loop_body, inner_after);
    mlir.oraBlockAppendOwnedOperation(loop_body, mlir.oraScfYieldOpCreate(
        mlir_ctx,
        loc,
        &[_]mlir.MlirValue{carried_inner},
        1,
    ));
    mlir.oraBlockAppendOwnedOperation(helper_body, outer_before);
    mlir.oraBlockAppendOwnedOperation(helper_body, inner_before);
    mlir.oraBlockAppendOwnedOperation(helper_body, loop);
    mlir.oraBlockAppendOwnedOperation(helper_body, mlir.oraReturnOpCreate(mlir_ctx, loc, &[_]mlir.MlirValue{}, 0));

    try encoder.registerFunctionOperation(helper);

    const owner_val = mlir.oraOperationGetResult(mlir.oraArithConstantOpCreate(
        mlir_ctx,
        loc,
        i256_ty,
        mlir.oraIntegerAttrCreateI64FromType(i256_ty, 11),
    ), 0);
    const spender_val = mlir.oraOperationGetResult(mlir.oraArithConstantOpCreate(
        mlir_ctx,
        loc,
        i256_ty,
        mlir.oraIntegerAttrCreateI64FromType(i256_ty, 22),
    ), 0);
    const value_val = mlir.oraOperationGetResult(mlir.oraArithConstantOpCreate(
        mlir_ctx,
        loc,
        i256_ty,
        mlir.oraIntegerAttrCreateI64FromType(i256_ty, 99),
    ), 0);

    const call = mlir.oraFuncCallOpCreate(
        mlir_ctx,
        loc,
        stringRef("writeAllowanceViaLoop"),
        &[_]mlir.MlirValue{ owner_val, spender_val, value_val },
        3,
        &[_]mlir.MlirType{},
        0,
    );
    _ = try encoder.encodeOperation(call);

    try testing.expect(!encoder.isDegraded());
    const constraints = try encoder.takeConstraints(testing.allocator);
    defer if (constraints.len > 0) testing.allocator.free(constraints);
    try expectNoQuantifiedConstraints(&z3_ctx, constraints);

    const allowances = encoder.global_map.get("allowances").?;
    const owner_ast = try encoder.encodeValue(owner_val);
    const spender_ast = try encoder.encodeValue(spender_val);
    const value_ast = try encoder.encodeValue(value_val);
    const inner_map = encoder.encodeSelect(allowances, owner_ast);
    const observed = encoder.encodeSelect(inner_map, spender_ast);

    var solver = try Solver.init(&z3_ctx, testing.allocator);
    defer solver.deinit();
    for (constraints) |cst| solver.assert(cst);
    solver.assert(z3.Z3_mk_not(z3_ctx.ctx, z3.Z3_mk_eq(z3_ctx.ctx, observed, value_ast)));
    try testing.expectEqual(@as(z3.Z3_lbool, z3.Z3_L_FALSE), solver.check());
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

test "func.call preserves untouched map slot without redundant quantified frame" {
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

    const pre_balances = encoder.global_old_map.get("balances") orelse encoder.global_entry_map.get("balances") orelse unreachable;
    const post_balances = encoder.global_map.get("balances") orelse unreachable;
    const key = try encoder.encodeIntegerConstant(11, 256);

    var solver = try Solver.init(&z3_ctx, testing.allocator);
    defer solver.deinit();
    for (constraints) |cst| solver.assert(cst);
    solver.assert(z3.Z3_mk_not(
        z3_ctx.ctx,
        z3.Z3_mk_eq(
            z3_ctx.ctx,
            encoder.encodeSelect(post_balances, key),
            encoder.encodeSelect(pre_balances, key),
        ),
    ));
    try testing.expectEqual(@as(z3.Z3_lbool, z3.Z3_L_FALSE), solver.check());

    try expectNoQuantifiedConstraints(&z3_ctx, constraints);
}

test "func.call scf.if map state summary uses concrete array-store semantics without redundant quantifier" {
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
    const map_ty = mlir.oraMapTypeGet(mlir_ctx, i256_ty, i256_ty);

    _ = try encoder.encodeOperation(mlir.oraSLoadOpCreate(mlir_ctx, loc, stringRef("balances"), map_ty));

    const helper_attrs = [_]mlir.MlirNamedAttribute{
        namedAttr(mlir_ctx, "sym_name", mlir.oraStringAttrCreate(mlir_ctx, stringRef("branchMapWriter"))),
        namedAttr(mlir_ctx, "ora.effect", mlir.oraStringAttrCreate(mlir_ctx, stringRef("writes"))),
        namedAttr(mlir_ctx, "ora.write_slots", mlir.oraArrayAttrCreate(mlir_ctx, 1, &[_]mlir.MlirAttribute{
            mlir.oraStringAttrCreate(mlir_ctx, stringRef("balances")),
        })),
    };
    const helper_param_types = [_]mlir.MlirType{i1_ty};
    const helper_param_locs = [_]mlir.MlirLocation{loc};
    const helper = mlir.oraFuncFuncOpCreate(mlir_ctx, loc, &helper_attrs, helper_attrs.len, &helper_param_types, &helper_param_locs, helper_param_types.len);
    const body = mlir.oraFuncOpGetBodyBlock(helper);
    const flag_arg = mlir.oraBlockGetArgument(body, 0);

    const no_results = [_]mlir.MlirType{};
    const if_op = mlir.oraScfIfOpCreate(mlir_ctx, loc, flag_arg, &no_results, no_results.len, true);
    const then_block = mlir.oraScfIfOpGetThenBlock(if_op);
    const else_block = mlir.oraScfIfOpGetElseBlock(if_op);

    const key1_attr = mlir.oraIntegerAttrCreateI64FromType(i256_ty, 1);
    const key2_attr = mlir.oraIntegerAttrCreateI64FromType(i256_ty, 2);
    const val11_attr = mlir.oraIntegerAttrCreateI64FromType(i256_ty, 11);
    const val22_attr = mlir.oraIntegerAttrCreateI64FromType(i256_ty, 22);

    const then_key_op = mlir.oraArithConstantOpCreate(mlir_ctx, loc, i256_ty, key1_attr);
    const then_val_op = mlir.oraArithConstantOpCreate(mlir_ctx, loc, i256_ty, val11_attr);
    const then_load = mlir.oraSLoadOpCreate(mlir_ctx, loc, stringRef("balances"), map_ty);
    const then_store = mlir.oraMapStoreOpCreate(
        mlir_ctx,
        loc,
        mlir.oraOperationGetResult(then_load, 0),
        mlir.oraOperationGetResult(then_key_op, 0),
        mlir.oraOperationGetResult(then_val_op, 0),
    );
    mlir.oraBlockAppendOwnedOperation(then_block, then_key_op);
    mlir.oraBlockAppendOwnedOperation(then_block, then_val_op);
    mlir.oraBlockAppendOwnedOperation(then_block, then_load);
    mlir.oraBlockAppendOwnedOperation(then_block, then_store);
    mlir.oraBlockAppendOwnedOperation(then_block, mlir.oraScfYieldOpCreate(mlir_ctx, loc, &[_]mlir.MlirValue{}, 0));

    const else_key_op = mlir.oraArithConstantOpCreate(mlir_ctx, loc, i256_ty, key2_attr);
    const else_val_op = mlir.oraArithConstantOpCreate(mlir_ctx, loc, i256_ty, val22_attr);
    const else_load = mlir.oraSLoadOpCreate(mlir_ctx, loc, stringRef("balances"), map_ty);
    const else_store = mlir.oraMapStoreOpCreate(
        mlir_ctx,
        loc,
        mlir.oraOperationGetResult(else_load, 0),
        mlir.oraOperationGetResult(else_key_op, 0),
        mlir.oraOperationGetResult(else_val_op, 0),
    );
    mlir.oraBlockAppendOwnedOperation(else_block, else_key_op);
    mlir.oraBlockAppendOwnedOperation(else_block, else_val_op);
    mlir.oraBlockAppendOwnedOperation(else_block, else_load);
    mlir.oraBlockAppendOwnedOperation(else_block, else_store);
    mlir.oraBlockAppendOwnedOperation(else_block, mlir.oraScfYieldOpCreate(mlir_ctx, loc, &[_]mlir.MlirValue{}, 0));

    mlir.oraBlockAppendOwnedOperation(body, if_op);
    mlir.oraBlockAppendOwnedOperation(body, mlir.oraReturnOpCreate(mlir_ctx, loc, &[_]mlir.MlirValue{}, 0));

    try encoder.registerFunctionOperation(helper);

    const flag = mlir.oraVariablePlaceholderOpCreate(mlir_ctx, loc, stringRef("branch_map_flag"), i1_ty);
    const call_operands = [_]mlir.MlirValue{mlir.oraOperationGetResult(flag, 0)};
    const call_op = mlir.oraFuncCallOpCreate(mlir_ctx, loc, stringRef("branchMapWriter"), &call_operands, call_operands.len, &[_]mlir.MlirType{}, 0);
    _ = try encoder.encodeOperation(call_op);

    try testing.expect(!encoder.isDegraded());
    const constraints = try encoder.takeConstraints(testing.allocator);
    defer if (constraints.len > 0) testing.allocator.free(constraints);
    try expectNoQuantifiedConstraints(&z3_ctx, constraints);

    const pre_balances = encoder.global_old_map.get("balances") orelse encoder.global_entry_map.get("balances") orelse unreachable;
    const post_balances = encoder.global_map.get("balances") orelse unreachable;
    const flag_ast = try encoder.encodeValue(mlir.oraOperationGetResult(flag, 0));
    const key1 = try encoder.encodeIntegerConstant(1, 256);
    const key2 = try encoder.encodeIntegerConstant(2, 256);
    const expected_11 = try encoder.encodeIntegerConstant(11, 256);

    var solver = try Solver.init(&z3_ctx, testing.allocator);
    defer solver.deinit();
    for (constraints) |cst| solver.assert(cst);
    solver.assert(flag_ast);

    solver.assert(z3.Z3_mk_not(
        z3_ctx.ctx,
        z3.Z3_mk_eq(z3_ctx.ctx, encoder.encodeSelect(post_balances, key1), expected_11),
    ));
    try testing.expectEqual(@as(z3.Z3_lbool, z3.Z3_L_FALSE), solver.check());
    solver.reset();

    for (constraints) |cst| solver.assert(cst);
    solver.assert(flag_ast);
    solver.assert(z3.Z3_mk_not(
        z3_ctx.ctx,
        z3.Z3_mk_eq(
            z3_ctx.ctx,
            encoder.encodeSelect(post_balances, key2),
            encoder.encodeSelect(pre_balances, key2),
        ),
    ));
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
        mlir_ctx,
        loc,
        i256_ty,
        mlir.oraIntegerAttrCreateI64FromType(i256_ty, 0),
    ), 0);
    const one = mlir.oraOperationGetResult(mlir.oraArithConstantOpCreate(
        mlir_ctx,
        loc,
        i256_ty,
        mlir.oraIntegerAttrCreateI64FromType(i256_ty, 1),
    ), 0);
    const step = mlir.oraOperationGetResult(mlir.oraArithConstantOpCreate(
        mlir_ctx,
        loc,
        i256_ty,
        mlir.oraIntegerAttrCreateI64FromType(i256_ty, 3),
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
        mlir_ctx,
        loc,
        mlir.oraOperationGetResult(cmp_op, 0),
        &[_]mlir.MlirValue{
            mlir.oraBlockGetArgument(before_block, 0),
            before_i,
            mlir.oraBlockGetArgument(before_block, 2),
        },
        3,
    ));

    const next_sum = mlir.oraArithAddIOpCreate(mlir_ctx, loc, after_sum, after_step);
    const next_i = mlir.oraArithAddIOpCreate(mlir_ctx, loc, after_i, one);
    mlir.oraBlockAppendOwnedOperation(after_block, next_sum);
    mlir.oraBlockAppendOwnedOperation(after_block, next_i);
    mlir.oraBlockAppendOwnedOperation(after_block, mlir.oraScfYieldOpCreate(
        mlir_ctx,
        loc,
        &[_]mlir.MlirValue{
            mlir.oraOperationGetResult(next_sum, 0),
            mlir.oraOperationGetResult(next_i, 0),
            after_step,
        },
        3,
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

test "known pure callee canonical signed inclusive positive-delta decrement scf.while return encodes exactly" {
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
        namedAttr(mlir_ctx, "sym_name", mlir.oraStringAttrCreate(mlir_ctx, stringRef("symbolicSignedDeltaInclusiveDecWhileReturn"))),
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

    const cmp_op = mlir.oraArithCmpIOpCreate(mlir_ctx, loc, 5, before_arg, bound); // sge
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

    const caller_bound = mlir.oraVariablePlaceholderOpCreate(mlir_ctx, loc, stringRef("callerSignedDeltaInclusiveDecBound"), i256_ty);
    const call = mlir.oraFuncCallOpCreate(
        mlir_ctx,
        loc,
        stringRef("symbolicSignedDeltaInclusiveDecWhileReturn"),
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
    const exclusive_bound = z3.Z3_mk_bv_sub(z3_ctx.ctx, bound_ast, one);
    const init_le_bound = z3.Z3_mk_bvsle(z3_ctx.ctx, init_ast, exclusive_bound);
    const distance = z3.Z3_mk_ite(
        z3_ctx.ctx,
        init_le_bound,
        zero,
        z3.Z3_mk_bv_sub(z3_ctx.ctx, init_ast, exclusive_bound),
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
