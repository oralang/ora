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

test "struct_field_update degrades when declaration and source metadata are both absent" {
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

    const pair_placeholder = mlir.oraVariablePlaceholderOpCreate(mlir_ctx, loc, stringRef("opaquePair"), struct_ty);
    const update_value = mlir.oraOperationGetResult(mlir.oraArithConstantOpCreate(
        mlir_ctx,
        loc,
        i256_ty,
        mlir.oraIntegerAttrCreateI64FromType(i256_ty, 7),
    ), 0);

    const update_op = mlir.oraStructFieldUpdateOpCreate(
        mlir_ctx,
        loc,
        mlir.oraOperationGetResult(pair_placeholder, 0),
        stringRef("left"),
        update_value,
    );

    _ = try encoder.encodeOperation(update_op);

    try testing.expect(encoder.isDegraded());
    try testing.expectEqualStrings("missing struct declaration metadata for struct update", encoder.degradationReason().?);
}

test "struct_field_extract degrades when product metadata is absent" {
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
    const struct_ty = mlir.oraStructTypeGet(mlir_ctx, stringRef("Pair__u256_extract_missing"));

    const pair_placeholder = mlir.oraVariablePlaceholderOpCreate(mlir_ctx, loc, stringRef("opaquePair"), struct_ty);
    const extract_op = mlir.oraStructFieldExtractOpCreate(
        mlir_ctx,
        loc,
        mlir.oraOperationGetResult(pair_placeholder, 0),
        stringRef("left"),
        i256_ty,
    );

    _ = try encoder.encodeOperation(extract_op);

    try testing.expect(encoder.isDegraded());
    try testing.expectEqualStrings("struct_field_extract requires exact product metadata", encoder.degradationReason().?);
}

test "struct_field_update degrades when field type metadata is missing" {
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
    const struct_ty = mlir.oraStructTypeGet(mlir_ctx, stringRef("Pair__u256_missing_types"));

    const struct_decl = mlir.oraStructDeclOpCreate(mlir_ctx, loc, stringRef("Pair__u256_missing_types"));
    const field_name_attrs = [_]mlir.MlirAttribute{
        mlir.oraStringAttrCreate(mlir_ctx, stringRef("left")),
        mlir.oraStringAttrCreate(mlir_ctx, stringRef("right")),
    };
    mlir.oraOperationSetAttributeByName(
        struct_decl,
        stringRef("ora.field_names"),
        mlir.oraArrayAttrCreate(mlir_ctx, field_name_attrs.len, &field_name_attrs),
    );
    try encoder.registerStructDeclOperation(struct_decl);

    const pair_placeholder = mlir.oraVariablePlaceholderOpCreate(mlir_ctx, loc, stringRef("pairWithMissingTypes"), struct_ty);
    const update_value = mlir.oraOperationGetResult(mlir.oraArithConstantOpCreate(
        mlir_ctx,
        loc,
        i256_ty,
        mlir.oraIntegerAttrCreateI64FromType(i256_ty, 7),
    ), 0);

    const update_op = mlir.oraStructFieldUpdateOpCreate(
        mlir_ctx,
        loc,
        mlir.oraOperationGetResult(pair_placeholder, 0),
        stringRef("left"),
        update_value,
    );

    _ = try encoder.encodeOperation(update_op);

    try testing.expect(encoder.isDegraded());
    try testing.expectEqualStrings("missing struct field type metadata for struct update", encoder.degradationReason().?);
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

test "struct_field_update recovers untouched fields through ora.try_stmt source metadata" {
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

    const try_stmt = mlir.oraTryStmtOpCreate(mlir_ctx, loc, &[_]mlir.MlirType{struct_ty}, 1);
    const try_block = mlir.oraTryStmtOpGetTryBlock(try_stmt);
    const catch_block = mlir.oraTryStmtOpGetCatchBlock(try_stmt);
    const field_name_attrs = [_]mlir.MlirAttribute{
        mlir.oraStringAttrCreate(mlir_ctx, stringRef("left")),
        mlir.oraStringAttrCreate(mlir_ctx, stringRef("right")),
    };

    const try_init = mlir.oraStructInitOpCreate(mlir_ctx, loc, &[_]mlir.MlirValue{ one, two }, 2, struct_ty);
    mlir.oraOperationSetAttributeByName(try_init, stringRef("ora.field_names"), mlir.oraArrayAttrCreate(mlir_ctx, field_name_attrs.len, &field_name_attrs));
    mlir.oraBlockAppendOwnedOperation(try_block, try_init);
    mlir.oraBlockAppendOwnedOperation(try_block, mlir.oraYieldOpCreate(
        mlir_ctx,
        loc,
        &[_]mlir.MlirValue{mlir.oraOperationGetResult(try_init, 0)},
        1,
    ));

    const catch_init = mlir.oraStructInitOpCreate(mlir_ctx, loc, &[_]mlir.MlirValue{ three, two }, 2, struct_ty);
    mlir.oraOperationSetAttributeByName(catch_init, stringRef("ora.field_names"), mlir.oraArrayAttrCreate(mlir_ctx, field_name_attrs.len, &field_name_attrs));
    mlir.oraBlockAppendOwnedOperation(catch_block, catch_init);
    mlir.oraBlockAppendOwnedOperation(catch_block, mlir.oraYieldOpCreate(
        mlir_ctx,
        loc,
        &[_]mlir.MlirValue{mlir.oraOperationGetResult(catch_init, 0)},
        1,
    ));

    const pair = mlir.oraOperationGetResult(try_stmt, 0);
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

test "struct_field_update recovers untouched fields through scf.for carried block arg metadata" {
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

    const extract_right = mlir.oraStructFieldExtractOpCreate(mlir_ctx, loc, mlir.oraOperationGetResult(loop, 0), stringRef("right"), i256_ty);
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

test "struct_field_update recovers untouched fields through scf.while carried block arg metadata" {
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

    const init_vals = [_]mlir.MlirValue{ true_val, pair };
    const result_types = [_]mlir.MlirType{ i1_ty, struct_ty };
    const while_op = mlir.oraScfWhileOpCreate(mlir_ctx, loc, &init_vals, init_vals.len, &result_types, result_types.len);
    const before_block = mlir.oraScfWhileOpGetBeforeBlock(while_op);
    const after_block = mlir.oraScfWhileOpGetAfterBlock(while_op);
    _ = mlir.mlirBlockAddArgument(before_block, i1_ty, loc);
    _ = mlir.mlirBlockAddArgument(before_block, struct_ty, loc);
    _ = mlir.mlirBlockAddArgument(after_block, i1_ty, loc);
    _ = mlir.mlirBlockAddArgument(after_block, struct_ty, loc);

    const before_flag = mlir.oraBlockGetArgument(before_block, 0);
    const before_pair = mlir.oraBlockGetArgument(before_block, 1);
    mlir.oraBlockAppendOwnedOperation(before_block, mlir.oraScfConditionOpCreate(
        mlir_ctx,
        loc,
        before_flag,
        &[_]mlir.MlirValue{ before_flag, before_pair },
        2,
    ));

    const after_pair = mlir.oraBlockGetArgument(after_block, 1);
    const update_op = mlir.oraStructFieldUpdateOpCreate(mlir_ctx, loc, after_pair, stringRef("left"), seven);
    mlir.oraBlockAppendOwnedOperation(after_block, update_op);
    mlir.oraBlockAppendOwnedOperation(after_block, mlir.oraScfYieldOpCreate(
        mlir_ctx,
        loc,
        &[_]mlir.MlirValue{ false_val, mlir.oraOperationGetResult(update_op, 0) },
        2,
    ));

    const extract_right = mlir.oraStructFieldExtractOpCreate(mlir_ctx, loc, mlir.oraOperationGetResult(while_op, 1), stringRef("right"), i256_ty);
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

test "struct_field_update recovers untouched fields from scoped struct declaration metadata" {
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

    const sym_name_attr = mlir.oraStringAttrCreate(mlir_ctx, stringRef("scope_struct_update"));
    const func_attrs = [_]mlir.MlirNamedAttribute{
        namedAttr(mlir_ctx, "sym_name", sym_name_attr),
    };
    const empty_types = [_]mlir.MlirType{};
    const empty_locs = [_]mlir.MlirLocation{};
    const func_op = mlir.oraFuncFuncOpCreate(mlir_ctx, loc, &func_attrs, func_attrs.len, &empty_types, &empty_locs, 0);
    mlir.oraBlockAppendOwnedOperation(module_body, func_op);
    const func_body = mlir.oraFuncOpGetBodyBlock(func_op);

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
    mlir.oraBlockAppendOwnedOperation(func_body, one_op);
    mlir.oraBlockAppendOwnedOperation(func_body, two_op);
    mlir.oraBlockAppendOwnedOperation(func_body, seven_op);

    const one = mlir.oraOperationGetResult(one_op, 0);
    const two = mlir.oraOperationGetResult(two_op, 0);
    const seven = mlir.oraOperationGetResult(seven_op, 0);
    const fields = [_]mlir.MlirValue{ one, two };
    const init_op = mlir.oraStructInstantiateOpCreate(mlir_ctx, loc, stringRef("Pair__u256"), &fields, fields.len, struct_ty);
    mlir.oraBlockAppendOwnedOperation(func_body, init_op);
    const pair = mlir.oraOperationGetResult(init_op, 0);

    const update_op = mlir.oraStructFieldUpdateOpCreate(mlir_ctx, loc, pair, stringRef("left"), seven);
    mlir.oraBlockAppendOwnedOperation(func_body, update_op);
    const updated_pair = mlir.oraOperationGetResult(update_op, 0);

    try testing.expectEqual(@as(usize, 2), mlir.oraStructTypeGetFieldCountInScope(update_op, struct_ty));
    const scoped_right = mlir.oraStructTypeGetFieldNameInScope(update_op, struct_ty, 1);
    try testing.expect(scoped_right.data != null);
    try testing.expectEqualStrings("right", scoped_right.data[0..scoped_right.length]);
    try testing.expect(!mlir.oraTypeIsNull(mlir.oraStructTypeGetFieldTypeInScope(update_op, struct_ty, 1)));

    const extract_left = mlir.oraStructFieldExtractOpCreate(mlir_ctx, loc, updated_pair, stringRef("left"), i256_ty);
    const extract_right = mlir.oraStructFieldExtractOpCreate(mlir_ctx, loc, updated_pair, stringRef("right"), i256_ty);
    mlir.oraBlockAppendOwnedOperation(func_body, extract_left);
    mlir.oraBlockAppendOwnedOperation(func_body, extract_right);

    const ret_vals = [_]mlir.MlirValue{};
    mlir.oraBlockAppendOwnedOperation(func_body, mlir.oraReturnOpCreate(mlir_ctx, loc, &ret_vals, ret_vals.len));

    const left_ast = try encoder.encodeOperation(extract_left);
    const right_ast = try encoder.encodeOperation(extract_right);

    try testing.expect(!encoder.isDegraded());

    const constraints = try encoder.takeConstraints(testing.allocator);
    defer if (constraints.len > 0) testing.allocator.free(constraints);

    var left_solver = try Solver.init(&z3_ctx, testing.allocator);
    defer left_solver.deinit();
    for (constraints) |cst| left_solver.assert(cst);
    left_solver.assert(z3.Z3_mk_not(z3_ctx.ctx, z3.Z3_mk_eq(z3_ctx.ctx, left_ast, try encoder.encodeValue(seven))));
    try testing.expectEqual(@as(z3.Z3_lbool, z3.Z3_L_FALSE), left_solver.check());

    var right_solver = try Solver.init(&z3_ctx, testing.allocator);
    defer right_solver.deinit();
    for (constraints) |cst| right_solver.assert(cst);
    right_solver.assert(z3.Z3_mk_not(z3_ctx.ctx, z3.Z3_mk_eq(z3_ctx.ctx, right_ast, try encoder.encodeValue(two))));
    try testing.expectEqual(@as(z3.Z3_lbool, z3.Z3_L_FALSE), right_solver.check());
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

test "sequence index coercion preserves signedness" {
    var z3_ctx = try Context.init(testing.allocator);
    defer z3_ctx.deinit();

    var encoder = Encoder.init(&z3_ctx, testing.allocator);
    defer encoder.deinit();

    const bv8_sort = z3.Z3_mk_bv_sort(z3_ctx.ctx, 8);
    const minus_one = z3.Z3_mk_numeral(z3_ctx.ctx, "255", bv8_sort);
    const signed_index = encoder.coerceAstToSeqIndexIntForTesting(minus_one, true);
    const unsigned_index = encoder.coerceAstToSeqIndexIntForTesting(minus_one, false);
    const int_sort = z3.Z3_mk_int_sort(z3_ctx.ctx);
    const minus_one_int = z3.Z3_mk_numeral(z3_ctx.ctx, "-1", int_sort);
    const two_fifty_five_int = z3.Z3_mk_numeral(z3_ctx.ctx, "255", int_sort);

    var solver = try Solver.init(&z3_ctx, testing.allocator);
    defer solver.deinit();

    solver.assert(z3.Z3_mk_not(z3_ctx.ctx, z3.Z3_mk_eq(z3_ctx.ctx, signed_index, minus_one_int)));
    try testing.expectEqual(@as(z3.Z3_lbool, z3.Z3_L_FALSE), solver.check());

    try solver.resetChecked();
    solver.assert(z3.Z3_mk_not(z3_ctx.ctx, z3.Z3_mk_eq(z3_ctx.ctx, unsigned_index, two_fifty_five_int)));
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

test "map_store relies on concrete array-store semantics instead of redundant frame quantifier" {
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
    try testing.expectEqual(@as(usize, 0), constraints.len);
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
