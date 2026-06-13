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
    try testing.expect(z3.Z3_is_seq_sort(z3_ctx.ctx, bytes_sort));
    const basis = z3.Z3_get_seq_sort_basis(z3_ctx.ctx, bytes_sort);
    try testing.expectEqual(@as(u32, z3.Z3_BV_SORT), @as(u32, @intCast(z3.Z3_get_sort_kind(z3_ctx.ctx, basis))));
    try testing.expectEqual(@as(u32, 8), @as(u32, @intCast(z3.Z3_get_bv_sort_size(z3_ctx.ctx, basis))));
}

test "encodeMLIRType maps tuple to product sort" {
    var z3_ctx = try Context.init(testing.allocator);
    defer z3_ctx.deinit();

    var encoder = Encoder.init(&z3_ctx, testing.allocator);
    defer encoder.deinit();

    const mlir_ctx = mlir.oraContextCreate();
    defer mlir.oraContextDestroy(mlir_ctx);
    loadAllDialects(mlir_ctx);
    _ = mlir.oraDialectRegister(mlir_ctx);

    const elements = [_]mlir.MlirType{
        mlir.oraIntegerTypeCreate(mlir_ctx, 256),
        mlir.oraIntegerTypeCreate(mlir_ctx, 1),
    };
    const tuple_ty = mlir.oraTupleTypeGet(mlir_ctx, elements.len, &elements);
    const tuple_sort = try encoder.encodeMLIRType(tuple_ty);
    try testing.expectEqual(false, z3.Z3_get_sort_kind(z3_ctx.ctx, tuple_sort) == z3.Z3_BV_SORT);
    try testing.expect(!encoder.isDegraded());
}

test "encodeMLIRType maps anonymous struct to product sort" {
    var z3_ctx = try Context.init(testing.allocator);
    defer z3_ctx.deinit();

    var encoder = Encoder.init(&z3_ctx, testing.allocator);
    defer encoder.deinit();

    const mlir_ctx = mlir.oraContextCreate();
    defer mlir.oraContextDestroy(mlir_ctx);
    loadAllDialects(mlir_ctx);
    _ = mlir.oraDialectRegister(mlir_ctx);

    const field_names = [_]mlir.MlirStringRef{
        stringRef("value"),
        stringRef("overflow"),
    };
    const field_types = [_]mlir.MlirType{
        mlir.oraIntegerTypeCreate(mlir_ctx, 256),
        mlir.oraIntegerTypeCreate(mlir_ctx, 1),
    };
    const struct_ty = mlir.oraAnonymousStructTypeGet(mlir_ctx, field_names.len, &field_names, &field_types);
    const struct_sort = try encoder.encodeMLIRType(struct_ty);
    try testing.expectEqual(false, z3.Z3_get_sort_kind(z3_ctx.ctx, struct_sort) == z3.Z3_BV_SORT);
    try testing.expect(!encoder.isDegraded());
}

test "encodeMLIRType degrades named struct opaque fallback without declaration metadata" {
    var z3_ctx = try Context.init(testing.allocator);
    defer z3_ctx.deinit();

    var encoder = Encoder.init(&z3_ctx, testing.allocator);
    defer encoder.deinit();

    const mlir_ctx = mlir.oraContextCreate();
    defer mlir.oraContextDestroy(mlir_ctx);
    loadAllDialects(mlir_ctx);
    _ = mlir.oraDialectRegister(mlir_ctx);

    const struct_ty = mlir.oraStructTypeGet(mlir_ctx, stringRef("Pair__u256"));
    const struct_sort = try encoder.encodeMLIRType(struct_ty);
    try testing.expectEqual(@as(u32, z3.Z3_BV_SORT), @as(u32, @intCast(z3.Z3_get_sort_kind(z3_ctx.ctx, struct_sort))));
    try testing.expectEqual(@as(u32, 256), @as(u32, @intCast(z3.Z3_get_bv_sort_size(z3_ctx.ctx, struct_sort))));
    try testing.expect(encoder.isDegraded());
    try testing.expect(std.mem.indexOf(u8, encoder.degradationReason() orelse "", "product MLIR type missing exact metadata") != null);
}

test "sortFromPrintedType maps tuple and anonymous struct to product sorts" {
    var z3_ctx = try Context.init(testing.allocator);
    defer z3_ctx.deinit();

    var encoder = Encoder.init(&z3_ctx, testing.allocator);
    defer encoder.deinit();

    const tuple_sort = try encoder.sortFromPrintedTypeForTesting("!ora.tuple<i256, i1>");
    try testing.expectEqual(false, z3.Z3_get_sort_kind(z3_ctx.ctx, tuple_sort) == z3.Z3_BV_SORT);

    const anon_sort = try encoder.sortFromPrintedTypeForTesting("!ora.struct_anon<(\"value\", i256), (\"overflow\", i1)>");
    try testing.expectEqual(false, z3.Z3_get_sort_kind(z3_ctx.ctx, anon_sort) == z3.Z3_BV_SORT);
    try testing.expect(!encoder.isDegraded());
}

test "sortFromPrintedType maps named struct to product sort when declaration metadata is registered" {
    var z3_ctx = try Context.init(testing.allocator);
    defer z3_ctx.deinit();

    var encoder = Encoder.init(&z3_ctx, testing.allocator);
    defer encoder.deinit();

    const mlir_ctx = mlir.oraContextCreate();
    defer mlir.oraContextDestroy(mlir_ctx);
    loadAllDialects(mlir_ctx);
    _ = mlir.oraDialectRegister(mlir_ctx);

    const loc = mlir.mlirLocationUnknownGet(mlir_ctx);
    const struct_decl = mlir.oraStructDeclOpCreate(mlir_ctx, loc, stringRef("Pair__u256"));
    const field_name_attrs = [_]mlir.MlirAttribute{
        mlir.oraStringAttrCreate(mlir_ctx, stringRef("lhs")),
        mlir.oraStringAttrCreate(mlir_ctx, stringRef("rhs")),
    };
    const field_type_attrs = [_]mlir.MlirAttribute{
        mlir.oraTypeAttrCreateFromType(mlir.oraIntegerTypeCreate(mlir_ctx, 256)),
        mlir.oraTypeAttrCreateFromType(mlir.oraIntegerTypeCreate(mlir_ctx, 256)),
    };
    mlir.oraOperationSetAttributeByName(struct_decl, stringRef("ora.field_names"), mlir.oraArrayAttrCreate(mlir_ctx, field_name_attrs.len, &field_name_attrs));
    mlir.oraOperationSetAttributeByName(struct_decl, stringRef("ora.field_types"), mlir.oraArrayAttrCreate(mlir_ctx, field_type_attrs.len, &field_type_attrs));
    try encoder.registerStructDeclOperation(struct_decl);

    const struct_sort = try encoder.sortFromPrintedTypeForTesting("!ora.struct<\"Pair__u256\">");
    try testing.expectEqual(false, z3.Z3_get_sort_kind(z3_ctx.ctx, struct_sort) == z3.Z3_BV_SORT);
    try testing.expect(!encoder.isDegraded());
}

test "transparent cast rebuilds matching product value into target product sort" {
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
    const named_ty = mlir.oraStructTypeGet(mlir_ctx, stringRef("Pair__u256"));

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

    const anon_field_names = [_]mlir.MlirStringRef{
        stringRef("left"),
        stringRef("right"),
    };
    const anon_field_types = [_]mlir.MlirType{ i256_ty, i256_ty };
    const anon_ty = mlir.oraAnonymousStructTypeGet(mlir_ctx, anon_field_names.len, &anon_field_names, &anon_field_types);

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
    const one = mlir.oraOperationGetResult(one_op, 0);
    const two = mlir.oraOperationGetResult(two_op, 0);

    const init_op = mlir.oraStructInitOpCreate(mlir_ctx, loc, &[_]mlir.MlirValue{ one, two }, 2, anon_ty);
    const cast_op = mlir.oraUnrealizedConversionCastOpCreate(mlir_ctx, loc, mlir.oraOperationGetResult(init_op, 0), named_ty);
    const extract_op = mlir.oraStructFieldExtractOpCreate(mlir_ctx, loc, mlir.oraOperationGetResult(cast_op, 0), stringRef("right"), i256_ty);
    const right_ast = try encoder.encodeOperation(extract_op);

    try testing.expect(!encoder.isDegraded());

    var solver = try Solver.init(&z3_ctx, testing.allocator);
    defer solver.deinit();
    solver.assert(z3.Z3_mk_not(z3_ctx.ctx, z3.Z3_mk_eq(z3_ctx.ctx, right_ast, try encoder.encodeValue(two))));
    try testing.expectEqual(@as(z3.Z3_lbool, z3.Z3_L_FALSE), solver.check());
}

test "sortFromPrintedType degrades on named struct without declaration metadata" {
    var z3_ctx = try Context.init(testing.allocator);
    defer z3_ctx.deinit();

    var encoder = Encoder.init(&z3_ctx, testing.allocator);
    defer encoder.deinit();

    try testing.expectError(error.UnsupportedOperation, encoder.sortFromPrintedTypeForTesting("!ora.struct<\"Pair__u256\">"));
    try testing.expect(encoder.isDegraded());
    try testing.expectEqualStrings("printed product type requires exact metadata-backed sort reconstruction", encoder.degradationReason().?);
}

test "unknown quantified binder types degrade before opaque bv256 fallback" {
    var z3_ctx = try Context.init(testing.allocator);
    defer z3_ctx.deinit();

    var encoder = Encoder.init(&z3_ctx, testing.allocator);
    defer encoder.deinit();

    const opaque_sort = encoder.quantifiedVarSortFromTypeStringForTesting("FutureType");
    try testing.expectEqual(@as(u32, z3.Z3_BV_SORT), @as(u32, @intCast(z3.Z3_get_sort_kind(z3_ctx.ctx, opaque_sort))));
    try testing.expectEqual(@as(u32, 256), @as(u32, @intCast(z3.Z3_get_bv_sort_size(z3_ctx.ctx, opaque_sort))));
    try testing.expect(encoder.isDegraded());
    try testing.expectEqualStrings("unsupported quantified binder type encoded via opaque bv256 fallback", encoder.degradationReason().?);
}

test "printed memref and slice sorts match shaped array encoding" {
    var z3_ctx = try Context.init(testing.allocator);
    defer z3_ctx.deinit();

    var encoder = Encoder.init(&z3_ctx, testing.allocator);
    defer encoder.deinit();

    const memref_sort = try encoder.sortFromPrintedTypeForTesting("memref<4x5xi32>");
    try testing.expectEqual(@as(u32, z3.Z3_ARRAY_SORT), @as(u32, @intCast(z3.Z3_get_sort_kind(z3_ctx.ctx, memref_sort))));
    const memref_inner = z3.Z3_get_array_sort_range(z3_ctx.ctx, memref_sort);
    try testing.expectEqual(@as(u32, z3.Z3_ARRAY_SORT), @as(u32, @intCast(z3.Z3_get_sort_kind(z3_ctx.ctx, memref_inner))));
    const memref_leaf = z3.Z3_get_array_sort_range(z3_ctx.ctx, memref_inner);
    try testing.expectEqual(@as(u32, 32), @as(u32, @intCast(z3.Z3_get_bv_sort_size(z3_ctx.ctx, memref_leaf))));

    const slice_sort = try encoder.sortFromPrintedTypeForTesting("!ora.slice<!ora.bytes>");
    try testing.expectEqual(@as(u32, z3.Z3_ARRAY_SORT), @as(u32, @intCast(z3.Z3_get_sort_kind(z3_ctx.ctx, slice_sort))));
    const slice_leaf = z3.Z3_get_array_sort_range(z3_ctx.ctx, slice_sort);
    try testing.expect(z3.Z3_is_seq_sort(z3_ctx.ctx, slice_leaf));
}

test "unknown printed types degrade before opaque bv256 fallback" {
    var z3_ctx = try Context.init(testing.allocator);
    defer z3_ctx.deinit();

    var encoder = Encoder.init(&z3_ctx, testing.allocator);
    defer encoder.deinit();

    const opaque_sort = try encoder.sortFromPrintedTypeForTesting("!ora.future_magic_type");
    try testing.expectEqual(@as(u32, z3.Z3_BV_SORT), @as(u32, @intCast(z3.Z3_get_sort_kind(z3_ctx.ctx, opaque_sort))));
    try testing.expectEqual(@as(u32, 256), @as(u32, @intCast(z3.Z3_get_bv_sort_size(z3_ctx.ctx, opaque_sort))));
    try testing.expect(encoder.isDegraded());
    try testing.expectEqualStrings("unsupported printed type encoded via opaque bv256 fallback", encoder.degradationReason().?);
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

test "declared ora.evm environment ops encode as stable env symbols" {
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
    const i256_ty = mlir.oraIntegerTypeCreate(mlir_ctx, 256);

    const EnvCase = struct {
        op_name: []const u8,
        symbol: []const u8,
        address: bool = false,
    };

    const cases = [_]EnvCase{
        .{ .op_name = "ora.evm.origin", .symbol = "env_evm_origin", .address = true },
        .{ .op_name = "ora.evm.caller", .symbol = "env_evm_caller", .address = true },
        .{ .op_name = "ora.evm.gasprice", .symbol = "env_evm_gasprice" },
        .{ .op_name = "ora.evm.callvalue", .symbol = "env_evm_callvalue" },
        .{ .op_name = "ora.evm.gas", .symbol = "env_evm_gas" },
        .{ .op_name = "ora.evm.timestamp", .symbol = "env_evm_timestamp" },
        .{ .op_name = "ora.evm.number", .symbol = "env_evm_number" },
        .{ .op_name = "ora.evm.coinbase", .symbol = "env_evm_coinbase", .address = true },
        .{ .op_name = "ora.evm.difficulty", .symbol = "env_evm_difficulty" },
        .{ .op_name = "ora.evm.prevrandao", .symbol = "env_evm_prevrandao" },
        .{ .op_name = "ora.evm.gaslimit", .symbol = "env_evm_gaslimit" },
        .{ .op_name = "ora.evm.chainid", .symbol = "env_evm_chainid" },
        .{ .op_name = "ora.evm.basefee", .symbol = "env_evm_basefee" },
    };

    inline for (cases) |case| {
        const result_ty = if (case.address) addr_ty else i256_ty;
        const first_op = mlir.oraEvmOpCreate(mlir_ctx, loc, stringRef(case.op_name), null, 0, result_ty);
        const second_op = mlir.oraEvmOpCreate(mlir_ctx, loc, stringRef(case.op_name), null, 0, result_ty);

        const first = try encoder.encodeOperation(first_op);
        const second = try encoder.encodeOperation(second_op);
        try expectAstEquivalent(&z3_ctx, first, second);

        const encoded_text = std.mem.span(z3.Z3_ast_to_string(z3_ctx.ctx, first));
        try testing.expect(std.mem.indexOf(u8, encoded_text, case.symbol) != null);
    }

    try testing.expect(!encoder.isDegraded());
}

test "declared ora.evm environment ops reject mismatched result sorts" {
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
    const mistyped_coinbase = mlir.oraEvmOpCreate(mlir_ctx, loc, stringRef("ora.evm.coinbase"), null, 0, i256_ty);

    try testing.expectError(error.UnsupportedOperation, encoder.encodeOperation(mistyped_coinbase));
    try testing.expect(encoder.isDegraded());
}

test "unsupported sort coercion records degradation" {
    var z3_ctx = try Context.init(testing.allocator);
    defer z3_ctx.deinit();

    var encoder = Encoder.init(&z3_ctx, testing.allocator);
    defer encoder.deinit();

    const bv8_sort = z3.Z3_mk_bv_sort(z3_ctx.ctx, 8);
    const seq_bv8_sort = z3.Z3_mk_seq_sort(z3_ctx.ctx, bv8_sort);
    const ast = z3.Z3_mk_const(
        z3_ctx.ctx,
        z3.Z3_mk_string_symbol(z3_ctx.ctx, "x"),
        seq_bv8_sort,
    );
    const coerced = encoder.coerceAstToSortForTesting(ast, bv8_sort);

    try testing.expect(encoder.isDegraded());
    try testing.expectEqualStrings("unsupported AST sort coercion", encoder.degradationReason().?);
    try testing.expectEqual(@as(u32, z3.Z3_BV_SORT), @as(u32, @intCast(z3.Z3_get_sort_kind(z3_ctx.ctx, z3.Z3_get_sort(z3_ctx.ctx, coerced)))));
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

test "unresolved callee degrades even before storage is materialized" {
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
    const call = mlir.oraFuncCallOpCreate(
        mlir_ctx,
        loc,
        stringRef("external_thing"),
        &[_]mlir.MlirValue{},
        0,
        &[_]mlir.MlirType{},
        0,
    );

    try testing.expectEqual(@as(usize, 0), encoder.global_map.count());
    _ = try encoder.encodeOperation(call);

    try testing.expect(encoder.isDegraded());
    try testing.expectEqual(@as(usize, 1), encoder.soundnessLosses().len);
    try testing.expectEqual(Encoder.SoundnessLoss.unresolved_callee, encoder.soundnessLosses()[0]);
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

test "nested map_store with address keys rethreads inner update exactly" {
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
    const i256_ty = mlir.oraIntegerTypeCreate(mlir_ctx, 256);
    const inner_map_ty = mlir.oraMapTypeGet(mlir_ctx, addr_ty, i256_ty);
    const outer_map_ty = mlir.oraMapTypeGet(mlir_ctx, addr_ty, inner_map_ty);

    const value_attr = mlir.oraIntegerAttrCreateI64FromType(i256_ty, 99);

    const caller_op = mlir.oraEvmOpCreate(mlir_ctx, loc, stringRef("ora.evm.caller"), null, 0, addr_ty);
    const value_op = mlir.oraArithConstantOpCreate(mlir_ctx, loc, i256_ty, value_attr);

    const owner = mlir.oraOperationGetResult(caller_op, 0);
    const spender = mlir.oraOperationGetResult(caller_op, 0);
    const value = mlir.oraOperationGetResult(value_op, 0);

    const outer_before = mlir.oraSLoadOpCreate(mlir_ctx, loc, stringRef("allowances_by_owner"), outer_map_ty);
    const outer_before_value = mlir.oraOperationGetResult(outer_before, 0);
    const inner_before = mlir.oraMapGetOpCreate(mlir_ctx, loc, outer_before_value, owner, inner_map_ty);
    const inner_before_value = mlir.oraOperationGetResult(inner_before, 0);

    _ = try encoder.encodeOperation(mlir.oraMapStoreOpCreate(mlir_ctx, loc, inner_before_value, spender, value));
    _ = try encoder.encodeOperation(mlir.oraMapStoreOpCreate(mlir_ctx, loc, outer_before_value, owner, inner_before_value));
    try testing.expect(!encoder.isDegraded());

    const outer_after = mlir.oraSLoadOpCreate(mlir_ctx, loc, stringRef("allowances_by_owner"), outer_map_ty);
    const outer_after_value = mlir.oraOperationGetResult(outer_after, 0);
    const inner_after = mlir.oraMapGetOpCreate(mlir_ctx, loc, outer_after_value, owner, inner_map_ty);
    const final_get = mlir.oraMapGetOpCreate(mlir_ctx, loc, mlir.oraOperationGetResult(inner_after, 0), spender, i256_ty);
    const loaded = try encoder.encodeOperation(final_get);
    const expected = try encoder.encodeValue(value);

    var solver = try Solver.init(&z3_ctx, testing.allocator);
    defer solver.deinit();
    solver.assert(z3.Z3_mk_not(z3_ctx.ctx, z3.Z3_mk_eq(z3_ctx.ctx, loaded, expected)));
    try testing.expectEqual(@as(z3.Z3_lbool, z3.Z3_L_FALSE), solver.check());
}
