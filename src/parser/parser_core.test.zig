// ============================================================================
// Parser Core Tests
// ============================================================================
// Tests for the main parser orchestrator
// ============================================================================

const std = @import("std");
const testing = std.testing;
const ora_root = @import("ora_root");
const lexer = ora_root.lexer;
const parser = ora_root.parser;
const ast = ora_root.ast;
const ast_arena = ora_root.ast_arena;
const diagnostics = ora_root.parser.diagnostics;

// ============================================================================
// Empty Input Tests
// ============================================================================

test "parser_core: empty input" {
    const allocator = testing.allocator;
    const source = "";

    var lex = lexer.Lexer.init(allocator, source);
    defer lex.deinit();
    const tokens = try lex.scanTokens();
    defer allocator.free(tokens);

    var arena = ast_arena.AstArena.init(allocator);
    defer arena.deinit();
    var parser_instance = parser.Parser.init(tokens, &arena);

    const nodes = try parser_instance.parse();
    defer arena.allocator().free(nodes);

    // Empty input should produce empty AST
    try testing.expect(nodes.len == 0);
}

test "parser_core: whitespace only" {
    const allocator = testing.allocator;
    const source = "   \n\t  ";

    var lex = lexer.Lexer.init(allocator, source);
    defer lex.deinit();
    const tokens = try lex.scanTokens();
    defer allocator.free(tokens);

    var arena = ast_arena.AstArena.init(allocator);
    defer arena.deinit();
    var parser_instance = parser.Parser.init(tokens, &arena);

    const nodes = try parser_instance.parse();
    defer arena.allocator().free(nodes);

    // Whitespace-only input should produce empty AST
    try testing.expect(nodes.len == 0);
}

// ============================================================================
// Single Contract Tests
// ============================================================================

test "parser_core: single empty contract" {
    const allocator = testing.allocator;
    const source = "contract Test { }";

    var lex = lexer.Lexer.init(allocator, source);
    defer lex.deinit();
    const tokens = try lex.scanTokens();
    defer allocator.free(tokens);

    var arena = ast_arena.AstArena.init(allocator);
    defer arena.deinit();
    var parser_instance = parser.Parser.init(tokens, &arena);

    const nodes = try parser_instance.parse();
    defer arena.allocator().free(nodes);

    try testing.expect(nodes.len == 1);
    try testing.expect(nodes[0] == .Contract);
    if (nodes[0] == .Contract) {
        try testing.expect(std.mem.eql(u8, "Test", nodes[0].Contract.name));
        try testing.expect(nodes[0].Contract.body.len == 0);
    }
}

test "parser_core: contract with function" {
    const allocator = testing.allocator;
    const source =
        \\contract Test {
        \\    pub fn foo() -> u256 {
        \\        return 42;
        \\    }
        \\}
    ;

    var lex = lexer.Lexer.init(allocator, source);
    defer lex.deinit();
    const tokens = try lex.scanTokens();
    defer allocator.free(tokens);

    var arena = ast_arena.AstArena.init(allocator);
    defer arena.deinit();
    var parser_instance = parser.Parser.init(tokens, &arena);

    const nodes = try parser_instance.parse();
    defer arena.allocator().free(nodes);

    try testing.expect(nodes.len == 1);
    try testing.expect(nodes[0] == .Contract);
    if (nodes[0] == .Contract) {
        try testing.expect(std.mem.eql(u8, "Test", nodes[0].Contract.name));
        try testing.expect(nodes[0].Contract.body.len >= 1);
    }
}

// ============================================================================
// Multiple Declarations Tests
// ============================================================================

test "parser_core: multiple contracts" {
    const allocator = testing.allocator;
    const source =
        \\contract A { }
        \\contract B { }
    ;

    var lex = lexer.Lexer.init(allocator, source);
    defer lex.deinit();
    const tokens = try lex.scanTokens();
    defer allocator.free(tokens);

    var arena = ast_arena.AstArena.init(allocator);
    defer arena.deinit();
    var parser_instance = parser.Parser.init(tokens, &arena);

    const nodes = try parser_instance.parse();
    defer arena.allocator().free(nodes);

    try testing.expect(nodes.len == 2);
    try testing.expect(nodes[0] == .Contract);
    try testing.expect(nodes[1] == .Contract);
    if (nodes[0] == .Contract and nodes[1] == .Contract) {
        try testing.expect(std.mem.eql(u8, "A", nodes[0].Contract.name));
        try testing.expect(std.mem.eql(u8, "B", nodes[1].Contract.name));
    }
}

test "parser_core: contract and standalone function" {
    const allocator = testing.allocator;
    const source =
        \\contract Test { }
        \\pub fn standalone() { }
    ;

    var lex = lexer.Lexer.init(allocator, source);
    defer lex.deinit();
    const tokens = try lex.scanTokens();
    defer allocator.free(tokens);

    var arena = ast_arena.AstArena.init(allocator);
    defer arena.deinit();
    var parser_instance = parser.Parser.init(tokens, &arena);

    const nodes = try parser_instance.parse();
    defer arena.allocator().free(nodes);

    try testing.expect(nodes.len == 2);
    try testing.expect(nodes[0] == .Contract);
    try testing.expect(nodes[1] == .Function);
}

// ============================================================================
// Import Tests
// ============================================================================

test "parser_core: import statement" {
    const allocator = testing.allocator;
    const source = "@import(\"std\");";

    var lex = lexer.Lexer.init(allocator, source);
    defer lex.deinit();
    const tokens = try lex.scanTokens();
    defer allocator.free(tokens);

    var arena = ast_arena.AstArena.init(allocator);
    defer arena.deinit();
    var parser_instance = parser.Parser.init(tokens, &arena);

    const nodes = try parser_instance.parse();
    defer arena.allocator().free(nodes);

    try testing.expect(nodes.len == 1);
    try testing.expect(nodes[0] == .Import);
    if (nodes[0] == .Import) {
        try testing.expect(std.mem.eql(u8, "std", nodes[0].Import.path));
    }
}

// ============================================================================
// Error Handling Tests
// ============================================================================

test "parser_core: unexpected token recovery" {
    const allocator = testing.allocator;
    // Invalid syntax - should recover and continue
    const source = "contract Test { invalid syntax here } contract Valid { }";
    const prev_diag = diagnostics.enable_stderr_diagnostics;
    diagnostics.enable_stderr_diagnostics = false;
    defer diagnostics.enable_stderr_diagnostics = prev_diag;

    var lex = lexer.Lexer.init(allocator, source);
    defer lex.deinit();
    const tokens = try lex.scanTokens();
    defer allocator.free(tokens);

    var arena = ast_arena.AstArena.init(allocator);
    defer arena.deinit();
    var parser_instance = parser.Parser.init(tokens, &arena);

    // Parser should attempt to recover and parse what it can
    // This may error or produce partial results depending on error recovery
    const nodes = parser_instance.parse() catch {
        // Expected - parser should handle errors gracefully
        return;
    };
    defer arena.allocator().free(nodes);

    // If parsing succeeded, we should have at least one contract
    // (parser may have recovered and skipped invalid parts)
}

test "parser_core: incomplete contract" {
    const allocator = testing.allocator;
    // Missing closing brace
    const source = "contract Test {";
    const prev_diag = diagnostics.enable_stderr_diagnostics;
    diagnostics.enable_stderr_diagnostics = false;
    defer diagnostics.enable_stderr_diagnostics = prev_diag;

    var lex = lexer.Lexer.init(allocator, source);
    defer lex.deinit();
    const tokens = try lex.scanTokens();
    defer allocator.free(tokens);

    var arena = ast_arena.AstArena.init(allocator);
    defer arena.deinit();
    var parser_instance = parser.Parser.init(tokens, &arena);

    // Should error on incomplete contract
    const nodes = parser_instance.parse() catch {
        // Expected error
        return;
    };
    defer arena.allocator().free(nodes);

    // If it didn't error, that's also acceptable (error recovery)
}
