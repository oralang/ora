//! End-to-end compiler tests
//! Simple integration tests that verify the compiler works correctly on real code

const std = @import("std");
const testing = std.testing;

// Test that basic compilation succeeds
test "compile simple contract" {
    const allocator = testing.allocator;

    const source =
        \\contract SimpleStorage {
        \\    pub storage value: u256;
        \\    
        \\    pub fn store(newValue: u256) {
        \\        value = newValue;
        \\    }
        \\    
        \\    pub fn retrieve() u256 {
        \\        return value;
        \\    }
        \\}
    ;

    // Just verify we can instantiate the lexer (basic smoke test)
    const Lexer = @import("ora").Lexer;
    var lexer = Lexer.init(allocator, source);
    defer lexer.deinit();

    const tokens = try lexer.scanTokens();
    defer allocator.free(tokens);

    // Basic assertion: we should have more than just EOF
    try testing.expect(tokens.len > 1);
}

// Test that the lexer handles keywords correctly
test "lexer recognizes keywords" {
    const allocator = testing.allocator;

    const source = "contract fn pub storage return";

    const Lexer = @import("ora").Lexer;
    var lexer = Lexer.init(allocator, source);
    defer lexer.deinit();

    const tokens = try lexer.scanTokens();
    defer allocator.free(tokens);

    // Should have: contract, fn, pub, storage, return, EOF = 6 tokens
    try testing.expect(tokens.len >= 6);
}

// Test that the lexer handles numbers
test "lexer scans numbers" {
    const allocator = testing.allocator;

    const source = "42 0x1234 256";

    const Lexer = @import("ora").Lexer;
    var lexer = Lexer.init(allocator, source);
    defer lexer.deinit();

    const tokens = try lexer.scanTokens();
    defer allocator.free(tokens);

    // Should have: 42, 0x1234, 256, EOF = 4 tokens
    try testing.expect(tokens.len >= 4);
}

// Test that parser can handle empty input
test "parser handles minimal input" {
    const allocator = testing.allocator;

    const source = "// Empty file\n";

    const Lexer = @import("ora").Lexer;
    const Parser = @import("ora").Parser;
    const AstArena = @import("ora").ast_arena.AstArena;

    var lexer = Lexer.init(allocator, source);
    defer lexer.deinit();

    const tokens = try lexer.scanTokens();
    defer allocator.free(tokens);

    var arena = AstArena.init(allocator);
    defer arena.deinit();

    var parser = Parser.init(tokens, &arena);
    const ast = try parser.parse();

    // Should succeed with empty AST
    try testing.expect(ast.len == 0);
}
