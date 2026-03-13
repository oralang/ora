const std = @import("std");
const testing = std.testing;
const lib = @import("ora_lib");
const mlir = @import("mod.zig");
const c = @import("mlir_c_api").c;
const mlir_c_api = @import("mlir_c_api");
const TypeResolver = lib.TypeResolver;

fn printOperationOwned(allocator: std.mem.Allocator, op: c.MlirOperation) ![]u8 {
    const text_ref = c.oraOperationPrintToString(op);
    defer mlir_c_api.freeStringRef(text_ref);

    if (text_ref.data == null or text_ref.length == 0) {
        return error.EmptyMlirText;
    }

    return allocator.dupe(u8, text_ref.data[0..text_ref.length]);
}

fn lowerSourceToMlirText(allocator: std.mem.Allocator, source: []const u8, filename: []const u8) ![]u8 {
    var lex = lib.lexer.Lexer.init(allocator, source);
    defer lex.deinit();
    const tokens = try lex.scanTokens();
    defer allocator.free(tokens);

    var arena = lib.ast_arena.AstArena.init(allocator);
    defer arena.deinit();
    var parser_instance = lib.parser.Parser.init(tokens, &arena);
    const nodes = try parser_instance.parse();
    defer arena.allocator().free(nodes);

    var sem = try lib.semantics.analyze(allocator, nodes);
    defer allocator.free(sem.diagnostics);
    defer sem.symbols.deinit();

    var type_resolver = TypeResolver.init(allocator, arena.allocator(), &sem.symbols);
    defer type_resolver.deinit();
    try type_resolver.resolveTypes(nodes);

    var mlir_arena = std.heap.ArenaAllocator.init(allocator);
    defer mlir_arena.deinit();
    const mlir_allocator = mlir_arena.allocator();

    const h = mlir.createContext(mlir_allocator);
    defer mlir.destroyContext(h);

    var lowering = try mlir.lower.lowerFunctionsToModuleWithSemanticTable(h.ctx, nodes, mlir_allocator, &sem.symbols, filename);
    defer lowering.deinit(mlir_allocator);
    defer c.oraModuleDestroy(lowering.module);

    try testing.expect(lowering.success);
    return printOperationOwned(allocator, c.oraModuleGetOperation(lowering.module));
}

fn parseModuleFromText(ctx: c.MlirContext, text: []const u8) !c.MlirModule {
    const ref = c.oraStringRefCreate(text.ptr, text.len);
    const module = c.oraModuleCreateParse(ctx, ref);
    if (c.oraModuleIsNull(module)) {
        return error.MlirParseFailed;
    }
    return module;
}

fn roundTripMlirText(allocator: std.mem.Allocator, text: []const u8) ![]u8 {
    var mlir_arena = std.heap.ArenaAllocator.init(allocator);
    defer mlir_arena.deinit();

    const h = mlir.createContext(mlir_arena.allocator());
    defer mlir.destroyContext(h);

    const module = parseModuleFromText(h.ctx, text) catch |err| {
        std.debug.print("Failed to parse round-trip MLIR:\n{s}\n", .{text});
        return err;
    };
    defer c.oraModuleDestroy(module);

    return printOperationOwned(allocator, c.oraModuleGetOperation(module));
}

fn expectRoundTripForSource(source: []const u8, filename: []const u8, expected_ops: []const []const u8) !void {
    const allocator = testing.allocator;
    const original = try lowerSourceToMlirText(allocator, source, filename);
    defer allocator.free(original);

    for (expected_ops) |op_name| {
        try testing.expect(std.mem.containsAtLeast(u8, original, 1, op_name));
    }

    const reparsed = try roundTripMlirText(allocator, original);
    defer allocator.free(reparsed);

    const reparsed_again = try roundTripMlirText(allocator, reparsed);
    defer allocator.free(reparsed_again);

    try testing.expectEqualStrings(original, reparsed);
    try testing.expectEqualStrings(reparsed, reparsed_again);
}

fn expectRoundTripForMlirText(text: []const u8, expected_ops: []const []const u8) !void {
    const allocator = testing.allocator;
    const reparsed = try roundTripMlirText(allocator, text);
    defer allocator.free(reparsed);

    for (expected_ops) |op_name| {
        try testing.expect(std.mem.containsAtLeast(u8, reparsed, 1, op_name));
    }
}

fn clearBlock(block: c.MlirBlock) void {
    var op = c.oraBlockGetFirstOperation(block);
    while (!c.oraOperationIsNull(op)) {
        const next = c.oraOperationGetNextInBlock(op);
        c.oraOperationErase(op);
        op = next;
    }
}

fn appendI64Constant(ctx: c.MlirContext, block: c.MlirBlock, loc: c.MlirLocation, ty: c.MlirType, value: i64) c.MlirValue {
    const attr = c.oraIntegerAttrCreateI64FromType(ty, value);
    const op = c.oraArithConstantOpCreate(ctx, loc, ty, attr);
    c.oraBlockAppendOwnedOperation(block, op);
    return c.oraOperationGetResult(op, 0);
}

fn buildTwoCaseSwitchExprModule(
    ctx: c.MlirContext,
    module: c.MlirModule,
    switch_loc: c.MlirLocation,
    case_loc: c.MlirLocation,
) ![]u8 {
    const module_body = c.oraModuleGetBody(module);
    const func_op = c.oraBlockGetFirstOperation(module_body);
    try testing.expect(!c.oraOperationIsNull(func_op));

    const func_body = c.oraFuncOpGetBodyBlock(func_op);
    try testing.expect(!c.oraBlockIsNull(func_body));
    clearBlock(func_body);

    const i256_ty = c.oraIntegerTypeCreate(ctx, 256);
    const condition = appendI64Constant(ctx, func_body, switch_loc, i256_ty, 0);

    const switch_op = c.oraSwitchExprOpCreateWithCases(
        ctx,
        switch_loc,
        condition,
        &[_]c.MlirType{i256_ty},
        1,
        2,
    );
    try testing.expect(!c.oraOperationIsNull(switch_op));
    c.oraBlockAppendOwnedOperation(func_body, switch_op);

    const case0 = c.oraSwitchExprOpGetCaseBlock(switch_op, 0);
    const case1 = c.oraSwitchExprOpGetCaseBlock(switch_op, 1);
    try testing.expect(!c.oraBlockIsNull(case0));
    try testing.expect(!c.oraBlockIsNull(case1));

    const case0_value = appendI64Constant(ctx, case0, case_loc, i256_ty, 1);
    const case0_yield = c.oraYieldOpCreate(ctx, case_loc, &[_]c.MlirValue{case0_value}, 1);
    c.oraBlockAppendOwnedOperation(case0, case0_yield);

    const case1_value = appendI64Constant(ctx, case1, case_loc, i256_ty, 2);
    const case1_yield = c.oraYieldOpCreate(ctx, case_loc, &[_]c.MlirValue{case1_value}, 1);
    c.oraBlockAppendOwnedOperation(case1, case1_yield);

    const case_values = [_]i64{ 0, 1 };
    const range_starts = [_]i64{ 0, 0 };
    const range_ends = [_]i64{ 0, 0 };
    const case_kinds = [_]i64{ 0, 0 };
    c.oraSwitchOpSetCasePatterns(
        switch_op,
        &case_values,
        &range_starts,
        &range_ends,
        &case_kinds,
        -1,
        2,
    );

    const switch_value = c.oraOperationGetResult(switch_op, 0);
    const ret = c.oraReturnOpCreate(ctx, switch_loc, &[_]c.MlirValue{switch_value}, 1);
    c.oraBlockAppendOwnedOperation(func_body, ret);

    return printOperationOwned(testing.allocator, c.oraModuleGetOperation(module));
}

test "mlir round-trips custom ora.contract ora.global and ora.sload assembly" {
    const source =
        \\contract Vault {
        \\    storage var balance: u256;
        \\
        \\    pub fn get() -> u256 {
        \\        return balance;
        \\    }
        \\}
    ;

    try expectRoundTripForSource(source, "roundtrip_contract_global_sload.ora", &.{
        "ora.contract",
        "ora.global",
        "ora.sload",
    });
}

test "mlir round-trips custom ora.switch assembly" {
    const source =
        \\contract Switches {
        \\    pub fn classify(v: u256) -> u256 {
        \\        switch (v) {
        \\            0 => {
        \\                return 10;
        \\            },
        \\            1...2 => {
        \\                return 20;
        \\            },
        \\            else => {
        \\                return 30;
        \\            }
        \\        }
        \\    }
        \\}
    ;

    try expectRoundTripForSource(source, "roundtrip_switch.ora", &.{
        "ora.switch",
    });
}

test "mlir round-trips custom ora.switch_expr assembly" {
    const source =
        \\contract SwitchExprs {
        \\    pub fn choose(tag: u256, start: u256) -> u256 {
        \\        return switch (tag) {
        \\            0 => start + 1,
        \\            1...2 => start + 2,
        \\            else => start + 3,
        \\        };
        \\    }
        \\}
    ;

    try expectRoundTripForSource(source, "roundtrip_switch_expr.ora", &.{
        "ora.switch_expr",
    });
}

test "mlir round-trips parsed ora.switch_expr literal assembly" {
    const text =
        \\module {
        \\  func.func @choose(%tag: i256) -> i256 {
        \\    %0 = ora.switch_expr %tag : i256 -> i256 {
        \\      case 3 => {
        \\        %1 = arith.constant 7 : i256
        \\        ora.yield %1 : i256
        \\      }
        \\      else => {
        \\        %2 = arith.constant 9 : i256
        \\        ora.yield %2 : i256
        \\      }
        \\    }
        \\    func.return %0 : i256
        \\  }
        \\}
    ;

    try expectRoundTripForMlirText(text, &.{
        "ora.switch_expr",
        "case 3 =>",
        "else =>",
    });
}

test "mlir round-trips custom single-case ora.switch_expr assembly" {
    const source =
        \\contract SwitchExprs {
        \\    pub fn choose(tag: u256) -> u256 {
        \\        return switch (tag) {
        \\            0 => 1,
        \\        };
        \\    }
        \\}
    ;

    try expectRoundTripForSource(source, "roundtrip_switch_expr_single_case.ora", &.{
        "ora.switch_expr",
        "case 0 =>",
    });
}

test "mlir round-trips parsed single-case ora.switch_expr literal assembly" {
    const text =
        \\module {
        \\  func.func @choose(%tag: i256) -> i256 {
        \\    %0 = ora.switch_expr %tag : i256 -> i256 {
        \\      case 0 => {
        \\        %1 = arith.constant 7 : i256
        \\        ora.yield %1 : i256
        \\      }
        \\    }
        \\    func.return %0 : i256
        \\  }
        \\}
    ;

    try expectRoundTripForMlirText(text, &.{
        "ora.switch_expr",
        "case 0 =>",
    });
}

test "mlir prints builder-created two-case ora.switch_expr" {
    var mlir_arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer mlir_arena.deinit();

    const h = mlir.createContext(mlir_arena.allocator());
    defer mlir.destroyContext(h);

    const module_text =
        \\module {
        \\  func.func @choose() -> i256 {
        \\    %0 = arith.constant 0 : i256
        \\    func.return %0 : i256
        \\  }
        \\}
    ;

    const module = try parseModuleFromText(h.ctx, module_text);
    defer c.oraModuleDestroy(module);

    const file_ref = c.oraStringRefCreate("builder_switch_expr.ora".ptr, "builder_switch_expr.ora".len);
    const loc = c.oraLocationFileLineColGet(h.ctx, file_ref, 1, 1);
    const case_loc = c.oraLocationFileLineColGet(h.ctx, file_ref, 2, 3);
    const text = try buildTwoCaseSwitchExprModule(h.ctx, module, loc, case_loc);
    defer testing.allocator.free(text);

    try testing.expect(std.mem.containsAtLeast(u8, text, 1, "ora.switch_expr"));
    try testing.expect(std.mem.containsAtLeast(u8, text, 1, "case 0 =>"));
    try testing.expect(std.mem.containsAtLeast(u8, text, 1, "case 1 =>"));
}

test "mlir prints builder-created two-case ora.switch_expr with file loc only on switch op" {
    var mlir_arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer mlir_arena.deinit();

    const h = mlir.createContext(mlir_arena.allocator());
    defer mlir.destroyContext(h);

    const module_text =
        \\module {
        \\  func.func @choose() -> i256 {
        \\    %0 = arith.constant 0 : i256
        \\    func.return %0 : i256
        \\  }
        \\}
    ;

    const module = try parseModuleFromText(h.ctx, module_text);
    defer c.oraModuleDestroy(module);

    const file_ref = c.oraStringRefCreate("builder_switch_expr.ora".ptr, "builder_switch_expr.ora".len);
    const loc = c.oraLocationFileLineColGet(h.ctx, file_ref, 1, 1);
    const unknown = c.oraLocationUnknownGet(h.ctx);

    const text = try buildTwoCaseSwitchExprModule(h.ctx, module, loc, unknown);
    defer testing.allocator.free(text);

    try testing.expect(std.mem.containsAtLeast(u8, text, 1, "ora.switch_expr"));
}

test "mlir prints builder-created two-case ora.switch_expr with file loc only inside cases" {
    var mlir_arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer mlir_arena.deinit();

    const h = mlir.createContext(mlir_arena.allocator());
    defer mlir.destroyContext(h);

    const module_text =
        \\module {
        \\  func.func @choose() -> i256 {
        \\    %0 = arith.constant 0 : i256
        \\    func.return %0 : i256
        \\  }
        \\}
    ;

    const module = try parseModuleFromText(h.ctx, module_text);
    defer c.oraModuleDestroy(module);

    const file_ref = c.oraStringRefCreate("builder_switch_expr.ora".ptr, "builder_switch_expr.ora".len);
    const case_loc = c.oraLocationFileLineColGet(h.ctx, file_ref, 2, 3);
    const unknown = c.oraLocationUnknownGet(h.ctx);

    const text = try buildTwoCaseSwitchExprModule(h.ctx, module, unknown, case_loc);
    defer testing.allocator.free(text);

    try testing.expect(std.mem.containsAtLeast(u8, text, 1, "ora.switch_expr"));
}

test "mlir prints builder-created two-case ora.switch_expr with case locs set after append" {
    var mlir_arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer mlir_arena.deinit();

    const h = mlir.createContext(mlir_arena.allocator());
    defer mlir.destroyContext(h);

    const module_text =
        \\module {
        \\  func.func @choose() -> i256 {
        \\    %0 = arith.constant 0 : i256
        \\    func.return %0 : i256
        \\  }
        \\}
    ;

    const module = try parseModuleFromText(h.ctx, module_text);
    defer c.oraModuleDestroy(module);

    const module_body = c.oraModuleGetBody(module);
    const func_op = c.oraBlockGetFirstOperation(module_body);
    try testing.expect(!c.oraOperationIsNull(func_op));

    const func_body = c.oraFuncOpGetBodyBlock(func_op);
    try testing.expect(!c.oraBlockIsNull(func_body));
    clearBlock(func_body);

    const file_ref = c.oraStringRefCreate("builder_switch_expr.ora".ptr, "builder_switch_expr.ora".len);
    const switch_loc = c.oraLocationFileLineColGet(h.ctx, file_ref, 1, 1);
    const case0_loc = c.oraLocationFileLineColGet(h.ctx, file_ref, 2, 3);
    const case1_loc = c.oraLocationFileLineColGet(h.ctx, file_ref, 3, 3);
    const unknown = c.oraLocationUnknownGet(h.ctx);
    const i256_ty = c.oraIntegerTypeCreate(h.ctx, 256);

    const cond_attr = c.oraIntegerAttrCreateI64FromType(i256_ty, 0);
    const cond_op = c.oraArithConstantOpCreate(h.ctx, unknown, i256_ty, cond_attr);
    c.oraOperationSetLocation(cond_op, switch_loc);
    c.oraBlockAppendOwnedOperation(func_body, cond_op);
    const condition = c.oraOperationGetResult(cond_op, 0);

    const switch_op = c.oraSwitchExprOpCreateWithCases(h.ctx, switch_loc, condition, &[_]c.MlirType{i256_ty}, 1, 2);
    try testing.expect(!c.oraOperationIsNull(switch_op));
    c.oraBlockAppendOwnedOperation(func_body, switch_op);

    const case0 = c.oraSwitchExprOpGetCaseBlock(switch_op, 0);
    const case1 = c.oraSwitchExprOpGetCaseBlock(switch_op, 1);
    try testing.expect(!c.oraBlockIsNull(case0));
    try testing.expect(!c.oraBlockIsNull(case1));

    const case0_attr = c.oraIntegerAttrCreateI64FromType(i256_ty, 1);
    const case0_const = c.oraArithConstantOpCreate(h.ctx, unknown, i256_ty, case0_attr);
    c.oraOperationSetLocation(case0_const, case0_loc);
    c.oraBlockAppendOwnedOperation(case0, case0_const);
    const case0_value = c.oraOperationGetResult(case0_const, 0);
    const case0_yield = c.oraYieldOpCreate(h.ctx, unknown, &[_]c.MlirValue{case0_value}, 1);
    c.oraOperationSetLocation(case0_yield, case0_loc);
    c.oraBlockAppendOwnedOperation(case0, case0_yield);

    const case1_attr = c.oraIntegerAttrCreateI64FromType(i256_ty, 2);
    const case1_const = c.oraArithConstantOpCreate(h.ctx, unknown, i256_ty, case1_attr);
    c.oraOperationSetLocation(case1_const, case1_loc);
    c.oraBlockAppendOwnedOperation(case1, case1_const);
    const case1_value = c.oraOperationGetResult(case1_const, 0);
    const case1_yield = c.oraYieldOpCreate(h.ctx, unknown, &[_]c.MlirValue{case1_value}, 1);
    c.oraOperationSetLocation(case1_yield, case1_loc);
    c.oraBlockAppendOwnedOperation(case1, case1_yield);

    const case_values = [_]i64{ 0, 1 };
    const range_starts = [_]i64{ 0, 0 };
    const range_ends = [_]i64{ 0, 0 };
    const case_kinds = [_]i64{ 0, 0 };
    c.oraSwitchOpSetCasePatterns(switch_op, &case_values, &range_starts, &range_ends, &case_kinds, -1, 2);

    const ret = c.oraReturnOpCreate(h.ctx, switch_loc, &[_]c.MlirValue{c.oraOperationGetResult(switch_op, 0)}, 1);
    c.oraBlockAppendOwnedOperation(func_body, ret);

    const text = try printOperationOwned(testing.allocator, c.oraModuleGetOperation(module));
    defer testing.allocator.free(text);

    try testing.expect(std.mem.containsAtLeast(u8, text, 1, "ora.switch_expr"));
    try testing.expect(std.mem.containsAtLeast(u8, text, 1, "case 0 =>"));
    try testing.expect(std.mem.containsAtLeast(u8, text, 1, "case 1 =>"));
}

test "mlir round-trips custom ora.conditional_return assembly" {
    const source =
        \\contract TryFlow {
        \\    error E();
        \\
        \\    pub fn caller(x: u256) -> !u256 | E {
        \\        let v: u256 = try callee(x);
        \\        return v;
        \\    }
        \\
        \\    fn callee(a: u256) -> !u256 | E {
        \\        return a;
        \\    }
        \\}
    ;

    try expectRoundTripForSource(source, "roundtrip_conditional_return.ora", &.{
        "ora.conditional_return",
    });
}
