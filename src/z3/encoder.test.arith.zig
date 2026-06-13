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

test "arith divui by zero still emits nonzero divisor obligation" {
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
    const rhs_attr = mlir.oraIntegerAttrCreateI64FromType(i256_ty, 0);
    const lhs_const = mlir.oraArithConstantOpCreate(mlir_ctx, loc, i256_ty, lhs_attr);
    const rhs_const = mlir.oraArithConstantOpCreate(mlir_ctx, loc, i256_ty, rhs_attr);
    const lhs = mlir.oraOperationGetResult(lhs_const, 0);
    const rhs = mlir.oraOperationGetResult(rhs_const, 0);

    const div_op = mlir.oraArithDivUIOpCreate(mlir_ctx, loc, lhs, rhs);
    const ast = try encoder.encodeOperation(div_op);
    const expected_zero = try encoder.encodeIntegerConstant(0, 256);
    try expectAstEquivalent(&z3_ctx, ast, expected_zero);

    const obligations = try encoder.takeObligations(testing.allocator);
    defer if (obligations.len > 0) testing.allocator.free(obligations);
    try testing.expectEqual(@as(usize, 1), obligations.len);
    var solver = try Solver.init(&z3_ctx, testing.allocator);
    defer solver.deinit();
    solver.assert(obligations[0]);
    try testing.expectEqual(@as(z3.Z3_lbool, z3.Z3_L_FALSE), solver.check());
}

test "signed div and rem int_min by negative one are explicit and portable" {
    var z3_ctx = try Context.init(testing.allocator);
    defer z3_ctx.deinit();

    var encoder = Encoder.init(&z3_ctx, testing.allocator);
    defer encoder.deinit();

    const min_int_value = @as(u256, 1) << 255;
    const lhs_min = try encoder.encodeIntegerConstant(min_int_value, 256);
    const rhs_neg_one = try encoder.encodeIntegerConstant(std.math.maxInt(u256), 256);
    const expected_div = try encoder.encodeIntegerConstant(min_int_value, 256);
    const expected_rem = try encoder.encodeIntegerConstant(0, 256);

    const div_ast = try encoder.encodeArithmeticOp(.DivSigned, lhs_min, rhs_neg_one);
    try expectAstEquivalent(&z3_ctx, div_ast, expected_div);

    const rem_ast = try encoder.encodeArithmeticOp(.RemSigned, lhs_min, rhs_neg_one);
    try expectAstEquivalent(&z3_ctx, rem_ast, expected_rem);
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

test "arith shli encodes left shift" {
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

    const shl_op = mlir.oraArithShlIOpCreate(mlir_ctx, loc, lhs, rhs);
    const ast = try encoder.encodeOperation(shl_op);
    const ast_str = std.mem.span(z3.Z3_ast_to_string(z3_ctx.ctx, ast));
    try testing.expect(std.mem.indexOf(u8, ast_str, "bvshl") != null);
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

test "precision note kinds are structured and labeled" {
    var z3_ctx = try Context.init(testing.allocator);
    defer z3_ctx.deinit();

    var encoder = Encoder.init(&z3_ctx, testing.allocator);
    defer encoder.deinit();

    inline for (std.meta.fields(Encoder.PrecisionNoteKind)) |field| {
        const note: Encoder.PrecisionNoteKind = @enumFromInt(field.value);
        const label = Encoder.precisionNoteLabel(note);
        try testing.expect(label.len > 0);
        try testing.expectEqualStrings(field.name, label);
        encoder.notePrecision(note);
    }

    const notes = encoder.precisionNotes();
    try testing.expectEqual(std.meta.fields(Encoder.PrecisionNoteKind).len, notes.len);
    try testing.expectEqual(Encoder.PrecisionNoteKind.path_precise_modifies_fallback_unavailable, notes[0]);
    try testing.expectEqual(Encoder.PrecisionNoteKind.precision_note_cap_exceeded, notes[1]);
}

test "precision note cap reports truncation explicitly" {
    var z3_ctx = try Context.init(testing.allocator);
    defer z3_ctx.deinit();

    var encoder = Encoder.init(&z3_ctx, testing.allocator);
    defer encoder.deinit();

    for (0..12) |_| {
        encoder.notePrecision(.path_precise_modifies_fallback_unavailable);
    }

    const notes = encoder.precisionNotes();
    try testing.expectEqual(@as(usize, 10), notes.len);
    try testing.expectEqual(Encoder.PrecisionNoteKind.path_precise_modifies_fallback_unavailable, notes[0]);
    try testing.expectEqual(Encoder.PrecisionNoteKind.precision_note_cap_exceeded, notes[9]);
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
