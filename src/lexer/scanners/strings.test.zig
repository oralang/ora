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
    try testing.expect(std.mem.indexOf(u8, lexer.tokenLexeme(source, tokens[0]), "hello") != null);
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
    const test_cases = [_][]const u8{
        "\"\\n\"", // newline
        "\"\\t\"", // tab
        "\"\\r\"", // carriage return
        "\"\\\\\"", // backslash
        "\"\\\"\"", // double quote
        "\"\\'\"", // single quote
        "\"\\0\"", // null byte
        "\"\\x41\"", // hex byte
        "\"\\xFF\"", // non-ASCII value byte via ASCII source escape
    };

    for (test_cases) |source| {
        var lex = lexer.Lexer.init(allocator, source);
        defer lex.deinit();

        const tokens = try lex.scanTokens();
        defer allocator.free(tokens);

        try testing.expectEqual(lexer.TokenType.StringLiteral, tokens[0].type);
    }
}

test "strings: escaped quote stays inside normal string" {
    const allocator = testing.allocator;
    const source = "\"a\\\"b\"";

    var lex = lexer.Lexer.init(allocator, source);
    defer lex.deinit();

    const tokens = try lex.scanTokens();
    defer allocator.free(tokens);

    try testing.expectEqual(lexer.TokenType.StringLiteral, tokens[0].type);
    try testing.expectEqualStrings("a\"b", tokens[0].value.?.*.string);
    try testing.expectEqual(lexer.TokenType.Eof, tokens[1].type);
}

test "strings: hex escape can produce non ASCII value byte" {
    const allocator = testing.allocator;
    const source = "\"\\xFF\"";

    var lex = lexer.Lexer.init(allocator, source);
    defer lex.deinit();

    const tokens = try lex.scanTokens();
    defer allocator.free(tokens);

    try testing.expectEqual(lexer.TokenType.StringLiteral, tokens[0].type);
    try testing.expectEqualSlices(u8, &[_]u8{0xff}, tokens[0].value.?.*.string);
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

test "strings: raw string literal rejects non ASCII source bytes" {
    const allocator = testing.allocator;
    const source = "r\"caf\xc3\xa9\"";

    var lex = lexer.Lexer.init(allocator, source);
    defer lex.deinit();

    try testing.expectError(lexer.LexerError.InvalidCharacterInString, lex.scanTokens());
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

    // lexer may use error recovery, which doesn't throw errors
    // instead, it records errors and continues
    const tokens = lex.scanTokens() catch |err| {
        // if error recovery is disabled, we get an error
        try testing.expectEqual(lexer.LexerError.UnterminatedString, err);
        return;
    };
    defer allocator.free(tokens);

    // with error recovery enabled (default), lexer continues and may produce tokens
    // verify we got some tokens (even if error recovery produced placeholder tokens)
    try testing.expect(tokens.len > 0);
}
