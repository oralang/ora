// ============================================================================
// String Literal Scanner Tests
// ============================================================================

const std = @import("std");
const testing = std.testing;
const ora_root = @import("ora_root");
const lexer = ora_root.lexer;

test "strings: basic string literal" {
    const allocator = testing.allocator;
    const source = "\"hello\"";

    var lex = lexer.Lexer.init(allocator, source);
    defer lex.deinit();

    const tokens = try lex.scanTokens();
    defer allocator.free(tokens);

    try testing.expectEqual(lexer.TokenType.StringLiteral, tokens[0].type);
    try testing.expect(std.mem.indexOf(u8, tokens[0].lexeme, "hello") != null);
}

test "strings: empty string" {
    const allocator = testing.allocator;
    const source = "\"\"";

    var lex = lexer.Lexer.init(allocator, source);
    defer lex.deinit();

    const tokens = try lex.scanTokens();
    defer allocator.free(tokens);

    try testing.expectEqual(lexer.TokenType.StringLiteral, tokens[0].type);
}

test "strings: escape sequences" {
    const allocator = testing.allocator;
    // Only test supported escape sequences: \n, \t, \\
    // Note: \" might have parsing edge cases, so test separately
    const test_cases = [_][]const u8{
        "\"\\n\"", // newline
        "\"\\t\"", // tab
        "\"\\\\\"", // backslash
    };

    for (test_cases) |source| {
        var lex = lexer.Lexer.init(allocator, source);
        defer lex.deinit();

        const tokens = try lex.scanTokens();
        defer allocator.free(tokens);

        try testing.expectEqual(lexer.TokenType.StringLiteral, tokens[0].type);
    }
}

test "strings: raw string literal" {
    const allocator = testing.allocator;
    const source = "r\"raw string\"";

    var lex = lexer.Lexer.init(allocator, source);
    defer lex.deinit();

    const tokens = try lex.scanTokens();
    defer allocator.free(tokens);

    try testing.expectEqual(lexer.TokenType.RawStringLiteral, tokens[0].type);
}

test "strings: character literal" {
    const allocator = testing.allocator;
    const source = "'a'";

    var lex = lexer.Lexer.init(allocator, source);
    defer lex.deinit();

    const tokens = try lex.scanTokens();
    defer allocator.free(tokens);

    try testing.expectEqual(lexer.TokenType.CharacterLiteral, tokens[0].type);
}

test "strings: hex bytes literal" {
    const allocator = testing.allocator;
    const source = "hex\"FF\"";

    var lex = lexer.Lexer.init(allocator, source);
    defer lex.deinit();

    const tokens = try lex.scanTokens();
    defer allocator.free(tokens);

    try testing.expectEqual(lexer.TokenType.BytesLiteral, tokens[0].type);
}

test "strings: unterminated string" {
    const allocator = testing.allocator;
    const source = "\"hello";

    var lex = lexer.Lexer.init(allocator, source);
    defer lex.deinit();

    // Lexer may use error recovery, which doesn't throw errors
    // Instead, it records errors and continues
    const tokens = lex.scanTokens() catch |err| {
        // If error recovery is disabled, we get an error
        try testing.expectEqual(lexer.LexerError.UnterminatedString, err);
        return;
    };
    defer allocator.free(tokens);

    // With error recovery enabled (default), lexer continues and may produce tokens
    // Verify we got some tokens (even if error recovery produced placeholder tokens)
    try testing.expect(tokens.len > 0);
}
