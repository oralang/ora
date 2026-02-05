// ============================================================================
// Effect Summary MLIR Tests
// ============================================================================

const std = @import("std");
const testing = std.testing;
const lib = @import("ora_lib");
const mlir = @import("mod.zig");
const c = @import("mlir_c_api").c;
const TypeResolver = @import("ora_lib").TypeResolver;

test "mlir emits storage read effects on functions" {
    const allocator = testing.allocator;
    const source =
        \\contract Test {
        \\    storage var balance: u256;
        \\    pub fn get() -> u256 {
        \\        return balance;
        \\    }
        \\}
    ;

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

    var lowering = try mlir.lower.lowerFunctionsToModuleWithSemanticTable(h.ctx, nodes, mlir_allocator, &sem.symbols, "effects.ora");
    defer lowering.deinit(mlir_allocator);
    defer c.oraModuleDestroy(lowering.module);

    try testing.expect(lowering.success);

    const module_op = c.oraModuleGetOperation(lowering.module);
    const mlir_text_ref = c.oraOperationPrintToString(module_op);
    defer @import("mlir_c_api").freeStringRef(mlir_text_ref);
    const mlir_text = if (mlir_text_ref.data != null and mlir_text_ref.length > 0)
        mlir_text_ref.data[0..mlir_text_ref.length]
    else
        "";

    try testing.expect(std.mem.containsAtLeast(u8, mlir_text, 1, "ora.effect = \"reads\""));
    try testing.expect(std.mem.containsAtLeast(u8, mlir_text, 1, "ora.read_slots = [\"balance\"]"));
}

test "mlir emits storage readwrite effects on functions" {
    const allocator = testing.allocator;
    const source =
        \\contract Test {
        \\    storage var balance: u256;
        \\    pub fn bump() {
        \\        balance = balance + 1;
        \\    }
        \\}
    ;

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

    var lowering = try mlir.lower.lowerFunctionsToModuleWithSemanticTable(h.ctx, nodes, mlir_allocator, &sem.symbols, "effects_rw.ora");
    defer lowering.deinit(mlir_allocator);
    defer c.oraModuleDestroy(lowering.module);

    try testing.expect(lowering.success);

    const module_op = c.oraModuleGetOperation(lowering.module);
    const mlir_text_ref = c.oraOperationPrintToString(module_op);
    defer @import("mlir_c_api").freeStringRef(mlir_text_ref);
    const mlir_text = if (mlir_text_ref.data != null and mlir_text_ref.length > 0)
        mlir_text_ref.data[0..mlir_text_ref.length]
    else
        "";

    try testing.expect(std.mem.containsAtLeast(u8, mlir_text, 1, "ora.effect = \"readwrites\""));
    try testing.expect(std.mem.containsAtLeast(u8, mlir_text, 1, "ora.read_slots = [\"balance\"]"));
    try testing.expect(std.mem.containsAtLeast(u8, mlir_text, 1, "ora.write_slots = [\"balance\"]"));
}
