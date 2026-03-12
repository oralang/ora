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
