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

    var type_resolver = TypeResolver.init(allocator, &sem.symbols);
    defer type_resolver.deinit();
    try type_resolver.resolveTypes(nodes);

    var mlir_arena = std.heap.ArenaAllocator.init(allocator);
    defer mlir_arena.deinit();
    const mlir_allocator = mlir_arena.allocator();

    const h = mlir.createContext(mlir_allocator);
    defer mlir.destroyContext(h);

    var lowering = try mlir.lower.lowerFunctionsToModuleWithSemanticTable(h.ctx, nodes, mlir_allocator, &sem.symbols, "effects.ora");
    defer lowering.deinit(mlir_allocator);
    defer c.mlirModuleDestroy(lowering.module);

    try testing.expect(lowering.success);

    const module_op = c.mlirModuleGetOperation(lowering.module);
    var mlir_text_buffer = std.ArrayList(u8){};
    defer mlir_text_buffer.deinit(mlir_allocator);

    const PrintCallback = struct {
        buffer: *std.ArrayList(u8),
        allocator: std.mem.Allocator,
        fn callback(message: c.MlirStringRef, user_data: ?*anyopaque) callconv(.c) void {
            const self = @as(*@This(), @ptrCast(@alignCast(user_data)));
            const message_slice = message.data[0..message.length];
            self.buffer.appendSlice(self.allocator, message_slice) catch {};
        }
    };

    var callback = PrintCallback{ .buffer = &mlir_text_buffer, .allocator = mlir_allocator };
    c.mlirOperationPrint(module_op, PrintCallback.callback, @ptrCast(&callback));

    const mlir_text = try mlir_text_buffer.toOwnedSlice(mlir_allocator);
    defer mlir_allocator.free(mlir_text);

    try testing.expect(std.mem.containsAtLeast(u8, mlir_text, 1, "ora.effect = \"reads\""));
    try testing.expect(std.mem.containsAtLeast(u8, mlir_text, 1, "ora.read_slots = [\"balance\"]"));
}

test "mlir emits storage readwrite effects on functions" {
    const allocator = testing.allocator;
    const source =
        \\contract Test {
        \\    storage var balance: u256;
        \\    pub fn bump() -> void {
        \\        balance = balance + 1;
        \\        return;
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

    var type_resolver = TypeResolver.init(allocator, &sem.symbols);
    defer type_resolver.deinit();
    try type_resolver.resolveTypes(nodes);

    var mlir_arena = std.heap.ArenaAllocator.init(allocator);
    defer mlir_arena.deinit();
    const mlir_allocator = mlir_arena.allocator();

    const h = mlir.createContext(mlir_allocator);
    defer mlir.destroyContext(h);

    var lowering = try mlir.lower.lowerFunctionsToModuleWithSemanticTable(h.ctx, nodes, mlir_allocator, &sem.symbols, "effects_rw.ora");
    defer lowering.deinit(mlir_allocator);
    defer c.mlirModuleDestroy(lowering.module);

    try testing.expect(lowering.success);

    const module_op = c.mlirModuleGetOperation(lowering.module);
    var mlir_text_buffer = std.ArrayList(u8){};
    defer mlir_text_buffer.deinit(mlir_allocator);

    const PrintCallback = struct {
        buffer: *std.ArrayList(u8),
        allocator: std.mem.Allocator,
        fn callback(message: c.MlirStringRef, user_data: ?*anyopaque) callconv(.c) void {
            const self = @as(*@This(), @ptrCast(@alignCast(user_data)));
            const message_slice = message.data[0..message.length];
            self.buffer.appendSlice(self.allocator, message_slice) catch {};
        }
    };

    var callback = PrintCallback{ .buffer = &mlir_text_buffer, .allocator = mlir_allocator };
    c.mlirOperationPrint(module_op, PrintCallback.callback, @ptrCast(&callback));

    const mlir_text = try mlir_text_buffer.toOwnedSlice(mlir_allocator);
    defer mlir_allocator.free(mlir_text);

    try testing.expect(std.mem.containsAtLeast(u8, mlir_text, 1, "ora.effect = \"readwrites\""));
    try testing.expect(std.mem.containsAtLeast(u8, mlir_text, 1, "ora.read_slots = [\"balance\"]"));
    try testing.expect(std.mem.containsAtLeast(u8, mlir_text, 1, "ora.write_slots = [\"balance\"]"));
}
