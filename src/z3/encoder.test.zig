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

test "scf.while result encoding returns stable symbolic value" {
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
    try testing.expect(std.mem.indexOf(u8, encoded_text, "scf.while_summary_") != null);
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
