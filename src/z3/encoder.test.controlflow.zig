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

fn parseModule(ctx: mlir.MlirContext, text: []const u8) !mlir.MlirModule {
    const module = mlir.oraModuleCreateParse(ctx, mlir.oraStringRefCreate(text.ptr, text.len));
    if (mlir.oraModuleIsNull(module)) return error.MlirParseFailed;
    return module;
}

fn findFirstOpByName(module: mlir.MlirModule, name: []const u8) ?mlir.MlirOperation {
    if (mlir.oraModuleIsNull(module)) return null;
    return findFirstOpByNameInOp(mlir.oraModuleGetOperation(module), name);
}

fn findFirstOpByNameInOp(op: mlir.MlirOperation, expected: []const u8) ?mlir.MlirOperation {
    if (mlir.oraOperationIsNull(op)) return null;

    const name_ref = mlir.oraOperationGetName(op);
    if (name_ref.data != null and std.mem.eql(u8, name_ref.data[0..name_ref.length], expected)) return op;

    const num_regions = mlir.oraOperationGetNumRegions(op);
    var region_index: usize = 0;
    while (region_index < num_regions) : (region_index += 1) {
        const region = mlir.oraOperationGetRegion(op, region_index);
        if (mlir.oraRegionIsNull(region)) continue;

        var block = mlir.oraRegionGetFirstBlock(region);
        while (!mlir.oraBlockIsNull(block)) : (block = mlir.oraBlockGetNextInRegion(block)) {
            var child = mlir.oraBlockGetFirstOperation(block);
            while (!mlir.oraOperationIsNull(child)) : (child = mlir.oraOperationGetNextInBlock(child)) {
                if (findFirstOpByNameInOp(child, expected)) |found| return found;
            }
        }
    }

    return null;
}

test "quantified bytes and string binders use sequence sort" {
    var z3_ctx = try Context.init(testing.allocator);
    defer z3_ctx.deinit();

    var encoder = Encoder.init(&z3_ctx, testing.allocator);
    defer encoder.deinit();

    const bytes_sort = try encoder.quantifiedVarSortFromTypeStringForTesting("bytes");
    const string_sort = try encoder.quantifiedVarSortFromTypeStringForTesting("string");

    try testing.expect(z3.Z3_is_seq_sort(z3_ctx.ctx, bytes_sort));
    try testing.expect(z3.Z3_is_seq_sort(z3_ctx.ctx, string_sort));

    const bytes_basis = z3.Z3_get_seq_sort_basis(z3_ctx.ctx, bytes_sort);
    const string_basis = z3.Z3_get_seq_sort_basis(z3_ctx.ctx, string_sort);
    try testing.expectEqual(@as(u32, 8), @as(u32, @intCast(z3.Z3_get_bv_sort_size(z3_ctx.ctx, bytes_basis))));
    try testing.expectEqual(@as(u32, 8), @as(u32, @intCast(z3.Z3_get_bv_sort_size(z3_ctx.ctx, string_basis))));
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

test "direct canonical unsigned inclusive positive-delta scf.while result encodes exactly" {
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
    const ub_op = mlir.oraVariablePlaceholderOpCreate(mlir_ctx, loc, stringRef("whileIncUbInclusive"), i256_ty);

    const init_vals = [_]mlir.MlirValue{mlir.oraOperationGetResult(init_op, 0)};
    const result_types = [_]mlir.MlirType{i256_ty};
    const while_op = mlir.oraScfWhileOpCreate(mlir_ctx, loc, &init_vals, init_vals.len, &result_types, result_types.len);
    const before_block = mlir.oraScfWhileOpGetBeforeBlock(while_op);
    const after_block = mlir.oraScfWhileOpGetAfterBlock(while_op);
    _ = mlir.mlirBlockAddArgument(before_block, i256_ty, loc);
    _ = mlir.mlirBlockAddArgument(after_block, i256_ty, loc);

    const before_arg = mlir.oraBlockGetArgument(before_block, 0);
    const cmp_op = mlir.oraArithCmpIOpCreate(mlir_ctx, loc, 7, before_arg, mlir.oraOperationGetResult(ub_op, 0)); // ule
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

    const maybe_value = mlir.oraVariablePlaceholderOpCreate(mlir_ctx, loc, stringRef("maybeValue"), eu_ty);
    const call_result = mlir.oraOperationGetResult(maybe_value, 0);
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

test "direct switch symbolic ora.try_stmt result encodes exactly" {
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

    const switch_op = mlir.oraSwitchOpCreateWithCases(
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
        switch_op,
        &case_values,
        &range_starts,
        &range_ends,
        &case_kinds,
        -1,
        case_values.len,
    );
    const false_block = mlir.oraSwitchOpGetCaseBlock(switch_op, 0);
    const true_block = mlir.oraSwitchOpGetCaseBlock(switch_op, 1);

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

    mlir.oraBlockAppendOwnedOperation(try_block, switch_op);
    mlir.oraBlockAppendOwnedOperation(try_block, mlir.oraYieldOpCreate(
        mlir_ctx,
        loc,
        &[_]mlir.MlirValue{mlir.oraOperationGetResult(switch_op, 0)},
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

test "direct ora.try_stmt yielding fixed-iteration symbolic-target scf.for result encodes exactly" {
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

    const limit_op = mlir.oraVariablePlaceholderOpCreate(mlir_ctx, loc, stringRef("symbolicTargetResultLimit"), index_ty);
    const target_op = mlir.oraVariablePlaceholderOpCreate(mlir_ctx, loc, stringRef("symbolicTargetResultTarget"), index_ty);
    const maybe_op = mlir.oraVariablePlaceholderOpCreate(mlir_ctx, loc, stringRef("symbolicTargetResultMaybe"), eu_ty);

    const outer_try = mlir.oraTryStmtOpCreate(mlir_ctx, loc, &[_]mlir.MlirType{i256_ty}, 1);
    const outer_try_block = mlir.oraTryStmtOpGetTryBlock(outer_try);
    const outer_catch_block = mlir.oraTryStmtOpGetCatchBlock(outer_try);

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
    mlir.oraBlockAppendOwnedOperation(outer_try_block, zero_idx_op);
    mlir.oraBlockAppendOwnedOperation(outer_try_block, one_idx_op);

    const for_op = mlir.oraScfForOpCreate(
        mlir_ctx,
        loc,
        mlir.oraOperationGetResult(zero_idx_op, 0),
        mlir.oraOperationGetResult(limit_op, 0),
        mlir.oraOperationGetResult(one_idx_op, 0),
        &[_]mlir.MlirValue{},
        0,
        false,
    );
    const body = mlir.oraScfForOpGetBodyBlock(for_op);
    const iv = mlir.oraBlockGetArgument(body, 0);
    const cmp_op = mlir.oraArithCmpIOpCreate(mlir_ctx, loc, 0, iv, mlir.oraOperationGetResult(target_op, 0)); // eq
    mlir.oraBlockAppendOwnedOperation(body, cmp_op);
    const no_results = [_]mlir.MlirType{};
    const if_op = mlir.oraScfIfOpCreate(mlir_ctx, loc, mlir.oraOperationGetResult(cmp_op, 0), &no_results, no_results.len, true);
    const then_block = mlir.oraScfIfOpGetThenBlock(if_op);
    const else_block = mlir.oraScfIfOpGetElseBlock(if_op);
    const unwrap_op = mlir.oraErrorUnwrapOpCreate(mlir_ctx, loc, mlir.oraOperationGetResult(maybe_op, 0), i256_ty);
    mlir.oraBlockAppendOwnedOperation(then_block, unwrap_op);
    mlir.oraBlockAppendOwnedOperation(then_block, mlir.oraScfYieldOpCreate(mlir_ctx, loc, &[_]mlir.MlirValue{}, 0));
    mlir.oraBlockAppendOwnedOperation(else_block, mlir.oraScfYieldOpCreate(mlir_ctx, loc, &[_]mlir.MlirValue{}, 0));
    mlir.oraBlockAppendOwnedOperation(body, if_op);
    mlir.oraBlockAppendOwnedOperation(body, mlir.oraScfYieldOpCreate(mlir_ctx, loc, &[_]mlir.MlirValue{}, 0));

    mlir.oraBlockAppendOwnedOperation(outer_try_block, for_op);
    const try_value_op = mlir.oraArithConstantOpCreate(
        mlir_ctx,
        loc,
        i256_ty,
        mlir.oraIntegerAttrCreateI64FromType(i256_ty, 7),
    );
    mlir.oraBlockAppendOwnedOperation(outer_try_block, try_value_op);
    mlir.oraBlockAppendOwnedOperation(outer_try_block, mlir.oraYieldOpCreate(
        mlir_ctx,
        loc,
        &[_]mlir.MlirValue{mlir.oraOperationGetResult(try_value_op, 0)},
        1,
    ));

    const catch_value_op = mlir.oraArithConstantOpCreate(
        mlir_ctx,
        loc,
        i256_ty,
        mlir.oraIntegerAttrCreateI64FromType(i256_ty, 33),
    );
    mlir.oraBlockAppendOwnedOperation(outer_catch_block, catch_value_op);
    mlir.oraBlockAppendOwnedOperation(outer_catch_block, mlir.oraYieldOpCreate(
        mlir_ctx,
        loc,
        &[_]mlir.MlirValue{mlir.oraOperationGetResult(catch_value_op, 0)},
        1,
    ));

    const encoded = try encoder.encodeValue(mlir.oraOperationGetResult(outer_try, 0));
    try testing.expect(!encoder.isDegraded());

    const limit_ast = try encoder.encodeValue(mlir.oraOperationGetResult(limit_op, 0));
    const target_ast = try encoder.encodeValue(mlir.oraOperationGetResult(target_op, 0));
    const is_error_op = mlir.oraErrorIsErrorOpCreate(mlir_ctx, loc, mlir.oraOperationGetResult(maybe_op, 0));
    const is_error = try encoder.encodeOperation(is_error_op);
    const seven = try encoder.encodeIntegerConstant(7, 256);
    const thirty_three = try encoder.encodeIntegerConstant(33, 256);
    const reaches_target = z3.Z3_mk_bvsgt(z3_ctx.ctx, limit_ast, target_ast);
    const catches = encoder.encodeAnd(&.{ reaches_target, encoder.coerceBoolean(is_error) });
    const expected = z3.Z3_mk_ite(z3_ctx.ctx, catches, thirty_three, seven);

    var solver = try Solver.init(&z3_ctx, testing.allocator);
    defer solver.deinit();
    solver.assert(z3.Z3_mk_not(z3_ctx.ctx, z3.Z3_mk_eq(z3_ctx.ctx, encoded, expected)));
    try testing.expectEqual(@as(z3.Z3_lbool, z3.Z3_L_FALSE), solver.check());
}

test "direct ora.try_stmt yielding multi-target symbolic scf.for result encodes exactly" {
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

    const limit_op = mlir.oraVariablePlaceholderOpCreate(mlir_ctx, loc, stringRef("multiTargetLimit"), index_ty);
    const target_a_op = mlir.oraVariablePlaceholderOpCreate(mlir_ctx, loc, stringRef("multiTargetA"), index_ty);
    const target_b_op = mlir.oraVariablePlaceholderOpCreate(mlir_ctx, loc, stringRef("multiTargetB"), index_ty);
    const maybe_op = mlir.oraVariablePlaceholderOpCreate(mlir_ctx, loc, stringRef("multiTargetMaybe"), eu_ty);

    const outer_try = mlir.oraTryStmtOpCreate(mlir_ctx, loc, &[_]mlir.MlirType{i256_ty}, 1);
    const outer_try_block = mlir.oraTryStmtOpGetTryBlock(outer_try);
    const outer_catch_block = mlir.oraTryStmtOpGetCatchBlock(outer_try);

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
    mlir.oraBlockAppendOwnedOperation(outer_try_block, zero_idx_op);
    mlir.oraBlockAppendOwnedOperation(outer_try_block, one_idx_op);

    const for_op = mlir.oraScfForOpCreate(
        mlir_ctx,
        loc,
        mlir.oraOperationGetResult(zero_idx_op, 0),
        mlir.oraOperationGetResult(limit_op, 0),
        mlir.oraOperationGetResult(one_idx_op, 0),
        &[_]mlir.MlirValue{},
        0,
        false,
    );
    const body = mlir.oraScfForOpGetBodyBlock(for_op);
    const iv = mlir.oraBlockGetArgument(body, 0);
    const cmp_a = mlir.oraArithCmpIOpCreate(mlir_ctx, loc, 0, iv, mlir.oraOperationGetResult(target_a_op, 0));
    mlir.oraBlockAppendOwnedOperation(body, cmp_a);
    const no_results = [_]mlir.MlirType{};
    const if_a = mlir.oraScfIfOpCreate(mlir_ctx, loc, mlir.oraOperationGetResult(cmp_a, 0), &no_results, no_results.len, true);
    const then_a = mlir.oraScfIfOpGetThenBlock(if_a);
    const else_a = mlir.oraScfIfOpGetElseBlock(if_a);
    const unwrap_a = mlir.oraErrorUnwrapOpCreate(mlir_ctx, loc, mlir.oraOperationGetResult(maybe_op, 0), i256_ty);
    mlir.oraBlockAppendOwnedOperation(then_a, unwrap_a);
    mlir.oraBlockAppendOwnedOperation(then_a, mlir.oraScfYieldOpCreate(mlir_ctx, loc, &[_]mlir.MlirValue{}, 0));

    const cmp_b = mlir.oraArithCmpIOpCreate(mlir_ctx, loc, 0, iv, mlir.oraOperationGetResult(target_b_op, 0));
    mlir.oraBlockAppendOwnedOperation(else_a, cmp_b);
    const if_b = mlir.oraScfIfOpCreate(mlir_ctx, loc, mlir.oraOperationGetResult(cmp_b, 0), &no_results, no_results.len, true);
    const then_b = mlir.oraScfIfOpGetThenBlock(if_b);
    const else_b = mlir.oraScfIfOpGetElseBlock(if_b);
    const unwrap_b = mlir.oraErrorUnwrapOpCreate(mlir_ctx, loc, mlir.oraOperationGetResult(maybe_op, 0), i256_ty);
    mlir.oraBlockAppendOwnedOperation(then_b, unwrap_b);
    mlir.oraBlockAppendOwnedOperation(then_b, mlir.oraScfYieldOpCreate(mlir_ctx, loc, &[_]mlir.MlirValue{}, 0));
    mlir.oraBlockAppendOwnedOperation(else_b, mlir.oraScfYieldOpCreate(mlir_ctx, loc, &[_]mlir.MlirValue{}, 0));
    mlir.oraBlockAppendOwnedOperation(else_a, if_b);
    mlir.oraBlockAppendOwnedOperation(else_a, mlir.oraScfYieldOpCreate(mlir_ctx, loc, &[_]mlir.MlirValue{}, 0));

    mlir.oraBlockAppendOwnedOperation(body, if_a);
    mlir.oraBlockAppendOwnedOperation(body, mlir.oraScfYieldOpCreate(mlir_ctx, loc, &[_]mlir.MlirValue{}, 0));

    mlir.oraBlockAppendOwnedOperation(outer_try_block, for_op);
    const try_value_op = mlir.oraArithConstantOpCreate(
        mlir_ctx,
        loc,
        i256_ty,
        mlir.oraIntegerAttrCreateI64FromType(i256_ty, 7),
    );
    mlir.oraBlockAppendOwnedOperation(outer_try_block, try_value_op);
    mlir.oraBlockAppendOwnedOperation(outer_try_block, mlir.oraYieldOpCreate(
        mlir_ctx,
        loc,
        &[_]mlir.MlirValue{mlir.oraOperationGetResult(try_value_op, 0)},
        1,
    ));

    const catch_value_op = mlir.oraArithConstantOpCreate(
        mlir_ctx,
        loc,
        i256_ty,
        mlir.oraIntegerAttrCreateI64FromType(i256_ty, 33),
    );
    mlir.oraBlockAppendOwnedOperation(outer_catch_block, catch_value_op);
    mlir.oraBlockAppendOwnedOperation(outer_catch_block, mlir.oraYieldOpCreate(
        mlir_ctx,
        loc,
        &[_]mlir.MlirValue{mlir.oraOperationGetResult(catch_value_op, 0)},
        1,
    ));

    const encoded = try encoder.encodeValue(mlir.oraOperationGetResult(outer_try, 0));
    try testing.expect(!encoder.isDegraded());

    const limit_ast = try encoder.encodeValue(mlir.oraOperationGetResult(limit_op, 0));
    const target_a_ast = try encoder.encodeValue(mlir.oraOperationGetResult(target_a_op, 0));
    const target_b_ast = try encoder.encodeValue(mlir.oraOperationGetResult(target_b_op, 0));
    const is_error_op = mlir.oraErrorIsErrorOpCreate(mlir_ctx, loc, mlir.oraOperationGetResult(maybe_op, 0));
    const is_error = try encoder.encodeOperation(is_error_op);
    const seven = try encoder.encodeIntegerConstant(7, 256);
    const thirty_three = try encoder.encodeIntegerConstant(33, 256);
    const reaches_a = z3.Z3_mk_bvsgt(z3_ctx.ctx, limit_ast, target_a_ast);
    const reaches_b = z3.Z3_mk_bvsgt(z3_ctx.ctx, limit_ast, target_b_ast);
    const catches = encoder.encodeAnd(&.{ encoder.encodeOr(&.{ reaches_a, reaches_b }), encoder.coerceBoolean(is_error) });
    const expected = z3.Z3_mk_ite(z3_ctx.ctx, catches, thirty_three, seven);

    var solver = try Solver.init(&z3_ctx, testing.allocator);
    defer solver.deinit();
    solver.assert(z3.Z3_mk_not(z3_ctx.ctx, z3.Z3_mk_eq(z3_ctx.ctx, encoded, expected)));
    try testing.expectEqual(@as(z3.Z3_lbool, z3.Z3_L_FALSE), solver.check());
}

test "direct ora.try_stmt yielding multi-target symbolic scf.while result encodes exactly" {
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

    const limit_op = mlir.oraVariablePlaceholderOpCreate(mlir_ctx, loc, stringRef("multiTargetWhileLimit"), index_ty);
    const target_a_op = mlir.oraVariablePlaceholderOpCreate(mlir_ctx, loc, stringRef("multiTargetWhileA"), index_ty);
    const target_b_op = mlir.oraVariablePlaceholderOpCreate(mlir_ctx, loc, stringRef("multiTargetWhileB"), index_ty);
    const maybe_op = mlir.oraVariablePlaceholderOpCreate(mlir_ctx, loc, stringRef("multiTargetWhileMaybe"), eu_ty);

    const outer_try = mlir.oraTryStmtOpCreate(mlir_ctx, loc, &[_]mlir.MlirType{i256_ty}, 1);
    const outer_try_block = mlir.oraTryStmtOpGetTryBlock(outer_try);
    const outer_catch_block = mlir.oraTryStmtOpGetCatchBlock(outer_try);

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
    mlir.oraBlockAppendOwnedOperation(outer_try_block, zero_idx_op);
    mlir.oraBlockAppendOwnedOperation(outer_try_block, one_idx_op);

    const init_vals = [_]mlir.MlirValue{mlir.oraOperationGetResult(zero_idx_op, 0)};
    const result_types = [_]mlir.MlirType{index_ty};
    const while_op = mlir.oraScfWhileOpCreate(mlir_ctx, loc, &init_vals, init_vals.len, &result_types, result_types.len);
    const before_block = mlir.oraScfWhileOpGetBeforeBlock(while_op);
    const after_block = mlir.oraScfWhileOpGetAfterBlock(while_op);
    _ = mlir.mlirBlockAddArgument(before_block, index_ty, loc);
    _ = mlir.mlirBlockAddArgument(after_block, index_ty, loc);
    const before_idx = mlir.oraBlockGetArgument(before_block, 0);
    const after_idx = mlir.oraBlockGetArgument(after_block, 0);

    const continue_cmp = mlir.oraArithCmpIOpCreate(mlir_ctx, loc, 6, before_idx, mlir.oraOperationGetResult(limit_op, 0)); // ult
    mlir.oraBlockAppendOwnedOperation(before_block, continue_cmp);
    mlir.oraBlockAppendOwnedOperation(before_block, mlir.oraScfConditionOpCreate(
        mlir_ctx,
        loc,
        mlir.oraOperationGetResult(continue_cmp, 0),
        &[_]mlir.MlirValue{before_idx},
        1,
    ));

    const cmp_a = mlir.oraArithCmpIOpCreate(mlir_ctx, loc, 0, after_idx, mlir.oraOperationGetResult(target_a_op, 0));
    mlir.oraBlockAppendOwnedOperation(after_block, cmp_a);
    const no_results = [_]mlir.MlirType{};
    const if_a = mlir.oraScfIfOpCreate(mlir_ctx, loc, mlir.oraOperationGetResult(cmp_a, 0), &no_results, no_results.len, true);
    const then_a = mlir.oraScfIfOpGetThenBlock(if_a);
    const else_a = mlir.oraScfIfOpGetElseBlock(if_a);
    const unwrap_a = mlir.oraErrorUnwrapOpCreate(mlir_ctx, loc, mlir.oraOperationGetResult(maybe_op, 0), i256_ty);
    mlir.oraBlockAppendOwnedOperation(then_a, unwrap_a);
    mlir.oraBlockAppendOwnedOperation(then_a, mlir.oraScfYieldOpCreate(mlir_ctx, loc, &[_]mlir.MlirValue{}, 0));

    const cmp_b = mlir.oraArithCmpIOpCreate(mlir_ctx, loc, 0, after_idx, mlir.oraOperationGetResult(target_b_op, 0));
    mlir.oraBlockAppendOwnedOperation(else_a, cmp_b);
    const if_b = mlir.oraScfIfOpCreate(mlir_ctx, loc, mlir.oraOperationGetResult(cmp_b, 0), &no_results, no_results.len, true);
    const then_b = mlir.oraScfIfOpGetThenBlock(if_b);
    const else_b = mlir.oraScfIfOpGetElseBlock(if_b);
    const unwrap_b = mlir.oraErrorUnwrapOpCreate(mlir_ctx, loc, mlir.oraOperationGetResult(maybe_op, 0), i256_ty);
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

    mlir.oraBlockAppendOwnedOperation(outer_try_block, while_op);
    const try_value_op = mlir.oraArithConstantOpCreate(
        mlir_ctx,
        loc,
        i256_ty,
        mlir.oraIntegerAttrCreateI64FromType(i256_ty, 7),
    );
    mlir.oraBlockAppendOwnedOperation(outer_try_block, try_value_op);
    mlir.oraBlockAppendOwnedOperation(outer_try_block, mlir.oraYieldOpCreate(
        mlir_ctx,
        loc,
        &[_]mlir.MlirValue{mlir.oraOperationGetResult(try_value_op, 0)},
        1,
    ));

    const catch_value_op = mlir.oraArithConstantOpCreate(
        mlir_ctx,
        loc,
        i256_ty,
        mlir.oraIntegerAttrCreateI64FromType(i256_ty, 33),
    );
    mlir.oraBlockAppendOwnedOperation(outer_catch_block, catch_value_op);
    mlir.oraBlockAppendOwnedOperation(outer_catch_block, mlir.oraYieldOpCreate(
        mlir_ctx,
        loc,
        &[_]mlir.MlirValue{mlir.oraOperationGetResult(catch_value_op, 0)},
        1,
    ));

    const encoded = try encoder.encodeValue(mlir.oraOperationGetResult(outer_try, 0));
    try testing.expect(!encoder.isDegraded());

    const limit_ast = try encoder.encodeValue(mlir.oraOperationGetResult(limit_op, 0));
    const target_a_ast = try encoder.encodeValue(mlir.oraOperationGetResult(target_a_op, 0));
    const target_b_ast = try encoder.encodeValue(mlir.oraOperationGetResult(target_b_op, 0));
    const is_error_op = mlir.oraErrorIsErrorOpCreate(mlir_ctx, loc, mlir.oraOperationGetResult(maybe_op, 0));
    const is_error = try encoder.encodeOperation(is_error_op);
    const seven = try encoder.encodeIntegerConstant(7, 256);
    const thirty_three = try encoder.encodeIntegerConstant(33, 256);
    const reaches_a = z3.Z3_mk_bvugt(z3_ctx.ctx, limit_ast, target_a_ast);
    const reaches_b = z3.Z3_mk_bvugt(z3_ctx.ctx, limit_ast, target_b_ast);
    const catches = encoder.encodeAnd(&.{ encoder.encodeOr(&.{ reaches_a, reaches_b }), encoder.coerceBoolean(is_error) });
    const expected = z3.Z3_mk_ite(z3_ctx.ctx, catches, thirty_three, seven);

    var solver = try Solver.init(&z3_ctx, testing.allocator);
    defer solver.deinit();
    solver.assert(z3.Z3_mk_not(z3_ctx.ctx, z3.Z3_mk_eq(z3_ctx.ctx, encoded, expected)));
    try testing.expectEqual(@as(z3.Z3_lbool, z3.Z3_L_FALSE), solver.check());
}

test "direct ora.try_stmt yielding multi-iteration symbolic scf.while result encodes exactly" {
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

    const limit_op = mlir.oraVariablePlaceholderOpCreate(mlir_ctx, loc, stringRef("multiIterResultLimit"), index_ty);
    const target_op = mlir.oraVariablePlaceholderOpCreate(mlir_ctx, loc, stringRef("multiIterResultTarget"), index_ty);
    const maybe_op = mlir.oraVariablePlaceholderOpCreate(mlir_ctx, loc, stringRef("multiIterResultMaybe"), eu_ty);

    const outer_try = mlir.oraTryStmtOpCreate(mlir_ctx, loc, &[_]mlir.MlirType{i256_ty}, 1);
    const outer_try_block = mlir.oraTryStmtOpGetTryBlock(outer_try);
    const outer_catch_block = mlir.oraTryStmtOpGetCatchBlock(outer_try);

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
    mlir.oraBlockAppendOwnedOperation(outer_try_block, zero_idx_op);
    mlir.oraBlockAppendOwnedOperation(outer_try_block, one_idx_op);

    const init_vals = [_]mlir.MlirValue{mlir.oraOperationGetResult(zero_idx_op, 0)};
    const result_types = [_]mlir.MlirType{index_ty};
    const while_op = mlir.oraScfWhileOpCreate(mlir_ctx, loc, &init_vals, init_vals.len, &result_types, result_types.len);
    const before_block = mlir.oraScfWhileOpGetBeforeBlock(while_op);
    const after_block = mlir.oraScfWhileOpGetAfterBlock(while_op);
    _ = mlir.mlirBlockAddArgument(before_block, index_ty, loc);
    _ = mlir.mlirBlockAddArgument(after_block, index_ty, loc);
    const before_idx = mlir.oraBlockGetArgument(before_block, 0);
    const after_idx = mlir.oraBlockGetArgument(after_block, 0);

    const continue_cmp = mlir.oraArithCmpIOpCreate(mlir_ctx, loc, 6, before_idx, mlir.oraOperationGetResult(limit_op, 0)); // ult
    mlir.oraBlockAppendOwnedOperation(before_block, continue_cmp);
    mlir.oraBlockAppendOwnedOperation(before_block, mlir.oraScfConditionOpCreate(
        mlir_ctx,
        loc,
        mlir.oraOperationGetResult(continue_cmp, 0),
        &[_]mlir.MlirValue{before_idx},
        1,
    ));

    const hit_cmp = mlir.oraArithCmpIOpCreate(mlir_ctx, loc, 0, after_idx, mlir.oraOperationGetResult(target_op, 0)); // eq
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
    const unwrap_op = mlir.oraErrorUnwrapOpCreate(mlir_ctx, loc, mlir.oraOperationGetResult(maybe_op, 0), i256_ty);
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

    mlir.oraBlockAppendOwnedOperation(outer_try_block, while_op);
    const try_value_op = mlir.oraArithConstantOpCreate(
        mlir_ctx,
        loc,
        i256_ty,
        mlir.oraIntegerAttrCreateI64FromType(i256_ty, 7),
    );
    mlir.oraBlockAppendOwnedOperation(outer_try_block, try_value_op);
    mlir.oraBlockAppendOwnedOperation(outer_try_block, mlir.oraYieldOpCreate(
        mlir_ctx,
        loc,
        &[_]mlir.MlirValue{mlir.oraOperationGetResult(try_value_op, 0)},
        1,
    ));

    const catch_value_op = mlir.oraArithConstantOpCreate(
        mlir_ctx,
        loc,
        i256_ty,
        mlir.oraIntegerAttrCreateI64FromType(i256_ty, 33),
    );
    mlir.oraBlockAppendOwnedOperation(outer_catch_block, catch_value_op);
    mlir.oraBlockAppendOwnedOperation(outer_catch_block, mlir.oraYieldOpCreate(
        mlir_ctx,
        loc,
        &[_]mlir.MlirValue{mlir.oraOperationGetResult(catch_value_op, 0)},
        1,
    ));

    const encoded = try encoder.encodeValue(mlir.oraOperationGetResult(outer_try, 0));
    try testing.expect(!encoder.isDegraded());

    const limit_ast = try encoder.encodeValue(mlir.oraOperationGetResult(limit_op, 0));
    const target_ast = try encoder.encodeValue(mlir.oraOperationGetResult(target_op, 0));
    const is_error_op = mlir.oraErrorIsErrorOpCreate(mlir_ctx, loc, mlir.oraOperationGetResult(maybe_op, 0));
    const is_error = try encoder.encodeOperation(is_error_op);
    const seven = try encoder.encodeIntegerConstant(7, 256);
    const thirty_three = try encoder.encodeIntegerConstant(33, 256);
    const reaches_target = z3.Z3_mk_bvugt(z3_ctx.ctx, limit_ast, target_ast);
    const catches = encoder.encodeAnd(&.{ reaches_target, encoder.coerceBoolean(is_error) });
    const expected = z3.Z3_mk_ite(z3_ctx.ctx, catches, thirty_three, seven);

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

test "direct ora.try_stmt encodes nested escaping try result exactly" {
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

    const cond_op = mlir.oraVariablePlaceholderOpCreate(mlir_ctx, loc, stringRef("condValueNestedYield"), i1_ty);

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
    try testing.expect(!encoder.isDegraded());

    const cond = try encoder.encodeOperation(cond_op);
    const five = try encoder.encodeIntegerConstant(5, 256);
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
    solver.assert(z3.Z3_mk_not(z3_ctx.ctx, z3.Z3_mk_eq(z3_ctx.ctx, encoded, five)));
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

test "direct symbolic geometric scf.for result encodes exactly" {
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

    const c0_op = mlir.oraArithConstantOpCreate(mlir_ctx, loc, index_ty, mlir.oraIntegerAttrCreateI64FromType(index_ty, 0));
    const c1_op = mlir.oraArithConstantOpCreate(mlir_ctx, loc, index_ty, mlir.oraIntegerAttrCreateI64FromType(index_ty, 1));
    const init_op = mlir.oraArithConstantOpCreate(mlir_ctx, loc, i256_ty, mlir.oraIntegerAttrCreateI64FromType(i256_ty, 13));
    const multiplier_op = mlir.oraArithConstantOpCreate(mlir_ctx, loc, i256_ty, mlir.oraIntegerAttrCreateI64FromType(i256_ty, 2));
    const ub_op = mlir.oraVariablePlaceholderOpCreate(mlir_ctx, loc, stringRef("forNonAffineUb"), index_ty);

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
    const next_op = mlir.oraArithMulIOpCreate(mlir_ctx, loc, carried_arg, mlir.oraOperationGetResult(multiplier_op, 0));
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
    const multiplier_ast = try encoder.encodeOperation(multiplier_op);
    const lb_ast = try encoder.encodeOperation(c0_op);
    const ub_ast = try encoder.encodeOperation(ub_op);
    const sort = z3.Z3_get_sort(z3_ctx.ctx, ub_ast);
    const zero = z3.Z3_mk_unsigned_int64(z3_ctx.ctx, 0, sort);
    const trip_count = z3.Z3_mk_ite(
        z3_ctx.ctx,
        z3.Z3_mk_bvule(z3_ctx.ctx, ub_ast, lb_ast),
        zero,
        z3.Z3_mk_bv_sub(z3_ctx.ctx, ub_ast, lb_ast),
    );
    const factor = try encoder.encodePowerOp(multiplier_ast, i256_ty, trip_count, index_ty, @intFromPtr(loop.ptr));
    const expected = try encoder.encodeArithmeticOp(.Mul, init_ast, factor);

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

test "scf.while without condition or yield degrades instead of fabricating a loop summary" {
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
    try testing.expectEqualStrings("scf.while result requires loop summary", encoder.degradationReason().?);
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

test "scf.while canonical signed inclusive positive-delta decrement result encodes exactly" {
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
    const bound_op = mlir.oraVariablePlaceholderOpCreate(mlir_ctx, loc, stringRef("signedDeltaDecInclusiveBoundValue"), i256_ty);
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

test "scf.while unsigned swapped-compare decrement degrades exact SMT modeling" {
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
    const bound_op = mlir.oraVariablePlaceholderOpCreate(mlir_ctx, loc, stringRef("unsignedSwappedDecBound"), i256_ty);
    const bound = mlir.oraOperationGetResult(bound_op, 0);

    const while_op = mlir.oraScfWhileOpCreate(mlir_ctx, loc, &[_]mlir.MlirValue{init}, 1, &[_]mlir.MlirType{i256_ty}, 1);
    const before_block = mlir.oraScfWhileOpGetBeforeBlock(while_op);
    const after_block = mlir.oraScfWhileOpGetAfterBlock(while_op);
    _ = mlir.mlirBlockAddArgument(before_block, i256_ty, loc);
    _ = mlir.mlirBlockAddArgument(after_block, i256_ty, loc);
    const before_arg = mlir.oraBlockGetArgument(before_block, 0);
    const after_arg = mlir.oraBlockGetArgument(after_block, 0);

    const cmp_op = mlir.oraArithCmpIOpCreate(mlir_ctx, loc, 8, bound, before_arg); // ugt swapped
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
    _ = try encoder.encodeValue(mlir.oraOperationGetResult(while_op, 0));
    try testing.expect(encoder.isDegraded());
    try testing.expectEqualStrings("scf.while result requires loop summary", encoder.degradationReason().?);
}

test "scf.while signed swapped-compare decrement degrades exact SMT modeling" {
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
    const bound_op = mlir.oraVariablePlaceholderOpCreate(mlir_ctx, loc, stringRef("signedSwappedDecBound"), i256_ty);
    const bound = mlir.oraOperationGetResult(bound_op, 0);

    const while_op = mlir.oraScfWhileOpCreate(mlir_ctx, loc, &[_]mlir.MlirValue{init}, 1, &[_]mlir.MlirType{i256_ty}, 1);
    const before_block = mlir.oraScfWhileOpGetBeforeBlock(while_op);
    const after_block = mlir.oraScfWhileOpGetAfterBlock(while_op);
    _ = mlir.mlirBlockAddArgument(before_block, i256_ty, loc);
    _ = mlir.mlirBlockAddArgument(after_block, i256_ty, loc);
    const before_arg = mlir.oraBlockGetArgument(before_block, 0);
    const after_arg = mlir.oraBlockGetArgument(after_block, 0);

    const cmp_op = mlir.oraArithCmpIOpCreate(mlir_ctx, loc, 4, bound, before_arg); // sgt swapped
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
    _ = try encoder.encodeValue(mlir.oraOperationGetResult(while_op, 0));
    try testing.expect(encoder.isDegraded());
    try testing.expectEqualStrings("scf.while result requires loop summary", encoder.degradationReason().?);
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
    try testing.expectEqual(@as(c_uint, 0), z3.Z3_get_quantifier_num_patterns(z3_ctx.ctx, qast));
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

test "quantified operation rejects unsupported quantifier instead of defaulting to forall" {
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
        stringRef("every"),
        stringRef("i"),
        stringRef("u256"),
        mlir.MlirValue{ .ptr = null },
        false,
        body,
        i1_ty,
    );

    try testing.expectError(error.UnsupportedOperation, encoder.encodeOperation(qop));
    try testing.expect(encoder.isDegraded());
    try testing.expectEqualStrings("ora.quantified unsupported quantifier attribute", encoder.degradationReason().?);
}

test "quantified operation rejects missing quantifier instead of defaulting to forall" {
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
        \\  func.func @bad(%flag: i1) {
        \\    %q = "ora.quantified"(%flag) <{variable = "i", variable_type = "u256"}> : (i1) -> i1
        \\    func.return
        \\  }
        \\}
    ;
    const module = try parseModule(mlir_ctx, text);
    defer mlir.oraModuleDestroy(module);

    const qop = findFirstOpByName(module, "ora.quantified") orelse return error.TestUnexpectedResult;
    try testing.expectError(error.UnsupportedOperation, encoder.encodeOperation(qop));
    try testing.expect(encoder.isDegraded());
    try testing.expectEqualStrings("ora.quantified missing quantifier attribute", encoder.degradationReason().?);
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
