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

test "mlir rewrites identity self-call in ensures to return value" {
    const allocator = testing.allocator;
    const source =
        \\contract Test {
        \\    pub fn id(n: u256) -> u256 {
        \\        return n;
        \\    }
        \\
        \\    pub fn sumToN(n: u256) -> u256
        \\        ensures(id(n) == sumToN(n))
        \\    {
        \\        return n;
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

    var lowering = try mlir.lower.lowerFunctionsToModuleWithSemanticTable(h.ctx, nodes, mlir_allocator, &sem.symbols, "ensures_self_call.ora");
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

    // Helper call remains a normal call in the postcondition.
    try testing.expect(std.mem.containsAtLeast(u8, mlir_text, 1, "call @id("));
    // Self-call in the same function postcondition must not lower to recursive call.
    try testing.expect(!std.mem.containsAtLeast(u8, mlir_text, 1, "call @sumToN("));
}

test "mlir gates ensures on success path for error-union returns" {
    const allocator = testing.allocator;
    const source =
        \\contract Test {
        \\    error E();
        \\    storage var balance: u256;
        \\
        \\    pub fn f(amount: u256) -> !bool | E
        \\        ensures(balance == old(balance) + amount)
        \\    {
        \\        if (amount > 10) {
        \\            return E();
        \\        }
        \\        balance = balance + amount;
        \\        return true;
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

    var lowering = try mlir.lower.lowerFunctionsToModuleWithSemanticTable(h.ctx, nodes, mlir_allocator, &sem.symbols, "ensures_error_union.ora");
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

    try testing.expect(std.mem.containsAtLeast(u8, mlir_text, 1, "ora.error.is_error"));
    try testing.expect(std.mem.containsAtLeast(u8, mlir_text, 1, "arith.ori"));
    try testing.expect(std.mem.containsAtLeast(u8, mlir_text, 1, "Postcondition 0 failed"));
}

test "mlir rethreads nested map assignment to outer map" {
    const allocator = testing.allocator;
    const source =
        \\contract Test {
        \\    storage var allowances: map<address, map<address, u256>>;
        \\
        \\    pub fn setAllowance(owner: address, spender: address, amount: u256) {
        \\        allowances[owner][spender] = amount;
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

    var lowering = try mlir.lower.lowerFunctionsToModuleWithSemanticTable(h.ctx, nodes, mlir_allocator, &sem.symbols, "nested_map_store.ora");
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

    const map_store_count = std.mem.count(u8, mlir_text, "ora.map_store");
    try testing.expect(map_store_count >= 2);
}

test "mlir encodes error-constructor call return as ora.error.err" {
    const allocator = testing.allocator;
    const source =
        \\contract Test {
        \\    error E(code: u256);
        \\
        \\    pub fn failIfPositive(x: u256) -> !bool | E {
        \\        if (x > 0) {
        \\            return E(x);
        \\        }
        \\        return true;
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

    var lowering = try mlir.lower.lowerFunctionsToModuleWithSemanticTable(h.ctx, nodes, mlir_allocator, &sem.symbols, "error_constructor_return.ora");
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

    try testing.expect(!std.mem.containsAtLeast(u8, mlir_text, 1, "call @E("));
    try testing.expect(std.mem.containsAtLeast(u8, mlir_text, 1, "ora.error.err"));
}

test "mlir does not emit func.call for payload error constructors in mixed unions" {
    const allocator = testing.allocator;
    const source =
        \\contract Test {
        \\    error E1;
        \\    error E2(code: u256);
        \\
        \\    pub fn mayFailMixed(x: u256) -> !bool | E1 | E2 {
        \\        if (x == 0) {
        \\            return E1;
        \\        }
        \\        if (x == 1) {
        \\            return E2(7);
        \\        }
        \\        return true;
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

    var lowering = try mlir.lower.lowerFunctionsToModuleWithSemanticTable(h.ctx, nodes, mlir_allocator, &sem.symbols, "error_constructor_mixed.ora");
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

    try testing.expect(!std.mem.containsAtLeast(u8, mlir_text, 1, "call @E2("));
    try testing.expect(std.mem.containsAtLeast(u8, mlir_text, 1, "ora.error.err"));
}

test "mlir keeps boolean ensures on state assignment" {
    const allocator = testing.allocator;
    const source =
        \\contract Test {
        \\    storage var is_paused: bool;
        \\
        \\    pub fn pause() -> bool
        \\        ensures(is_paused == true)
        \\    {
        \\        is_paused = true;
        \\        return true;
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

    var lowering = try mlir.lower.lowerFunctionsToModuleWithSemanticTable(h.ctx, nodes, mlir_allocator, &sem.symbols, "bool_ensures.ora");
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

    try testing.expect(std.mem.containsAtLeast(u8, mlir_text, 1, "\"is_paused\" : i1"));
    try testing.expect(!std.mem.containsAtLeast(u8, mlir_text, 1, "\"is_paused\" : i256"));
    try testing.expect(std.mem.containsAtLeast(u8, mlir_text, 1, "Postcondition 0 failed"));
}

test "mlir lowers arithmetic old() ensures expression" {
    const allocator = testing.allocator;
    const source =
        \\contract Test {
        \\    storage var total: u256;
        \\
        \\    pub fn addN(n: u256) -> u256
        \\        requires(n <= 100)
        \\        ensures(total == old(total) + n * (n + 1) / 2)
        \\    {
        \\        var i: u256 = 0;
        \\        while (i < n) {
        \\            i = i + 1;
        \\            total = total + i;
        \\        }
        \\        return total;
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

    var lowering = try mlir.lower.lowerFunctionsToModuleWithSemanticTable(h.ctx, nodes, mlir_allocator, &sem.symbols, "old_arith_ensures.ora");
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

    try testing.expect(std.mem.containsAtLeast(u8, mlir_text, 1, "ora.old"));
    try testing.expect(std.mem.containsAtLeast(u8, mlir_text, 1, "arith.muli"));
    try testing.expect(std.mem.containsAtLeast(u8, mlir_text, 1, "arith.divui"));
    try testing.expect(std.mem.containsAtLeast(u8, mlir_text, 1, "Postcondition 0 failed"));
}

test "mlir emits zero-result call for void callee" {
    const allocator = testing.allocator;
    const source =
        \\contract Test {
        \\    fn helper(x: u256) {
        \\    }
        \\
        \\    pub fn caller(a: u256) -> u256 {
        \\        helper(a);
        \\        return a;
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

    var lowering = try mlir.lower.lowerFunctionsToModuleWithSemanticTable(h.ctx, nodes, mlir_allocator, &sem.symbols, "void_call.ora");
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

    try testing.expect(std.mem.containsAtLeast(u8, mlir_text, 1, "call @helper("));
    try testing.expect(!std.mem.containsAtLeast(u8, mlir_text, 1, "= call @helper("));
    try testing.expect(!std.mem.containsAtLeast(u8, mlir_text, 1, "-> none"));
}

test "mlir infers forward callee error-union result type from typed call" {
    const allocator = testing.allocator;
    const source =
        \\contract Test {
        \\    error E();
        \\
        \\    pub fn caller() -> !u256 | E {
        \\        let v: u256 = try callee();
        \\        return v;
        \\    }
        \\
        \\    fn callee() -> !u256 | E {
        \\        return 1;
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

    var lowering = try mlir.lower.lowerFunctionsToModuleWithSemanticTable(h.ctx, nodes, mlir_allocator, &sem.symbols, "forward_error_union_call.ora");
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

    try testing.expect(std.mem.containsAtLeast(u8, mlir_text, 1, "call @callee()"));
    try testing.expect(std.mem.containsAtLeast(u8, mlir_text, 1, "-> !ora.error_union<i256>"));
    try testing.expect(!std.mem.containsAtLeast(u8, mlir_text, 1, "call @callee() {gas_cost = 10 : i64} : () -> i256"));
}

test "mlir infers forward callee param types and inserts arg conversion" {
    const allocator = testing.allocator;
    const source =
        \\contract Test {
        \\    pub fn caller() -> u256 {
        \\        return sink(std.msg.sender());
        \\    }
        \\
        \\    fn sink(a: address) -> u256 {
        \\        return 1;
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

    var lowering = try mlir.lower.lowerFunctionsToModuleWithSemanticTable(h.ctx, nodes, mlir_allocator, &sem.symbols, "forward_param_conversion.ora");
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

    try testing.expect(std.mem.containsAtLeast(u8, mlir_text, 1, "call @sink("));
    // Assert semantic intent rather than brittle call-print formatting:
    // forward callee param must resolve to address, and caller arg must be
    // adapted from non-zero-address via refinement_to_base.
    try testing.expect(std.mem.containsAtLeast(u8, mlir_text, 1, "func.func @sink(%arg0: !ora.address"));
    try testing.expect(std.mem.containsAtLeast(u8, mlir_text, 1, "ora.refinement_to_base"));
    try testing.expect(!std.mem.containsAtLeast(u8, mlir_text, 1, "func.func @sink(%arg0: !ora.non_zero_address"));
}
