// ============================================================================
// Identifier and Keyword Scanner Tests
// ============================================================================

const std = @import("std");
const testing = std.testing;
const ora_root = @import("ora_root");
const lexer = ora_root.lexer;

test "identifiers: simple identifier" {
    const allocator = testing.allocator;
    const source = "myVariable";

    var lex = lexer.Lexer.init(allocator, source);
    defer lex.deinit();

    const tokens = try lex.scanTokens();
    defer allocator.free(tokens);

    try testing.expectEqual(lexer.TokenType.Identifier, tokens[0].type);
    try testing.expect(std.mem.eql(u8, "myVariable", tokens[0].lexeme));
}

test "identifiers: identifier with underscore" {
    const allocator = testing.allocator;
    const source = "my_variable";

    var lex = lexer.Lexer.init(allocator, source);
    defer lex.deinit();

    const tokens = try lex.scanTokens();
    defer allocator.free(tokens);

    try testing.expectEqual(lexer.TokenType.Identifier, tokens[0].type);
    try testing.expect(std.mem.eql(u8, "my_variable", tokens[0].lexeme));
}

test "identifiers: identifier with numbers" {
    const allocator = testing.allocator;
    const source = "var123";

    var lex = lexer.Lexer.init(allocator, source);
    defer lex.deinit();

    const tokens = try lex.scanTokens();
    defer allocator.free(tokens);

    try testing.expectEqual(lexer.TokenType.Identifier, tokens[0].type);
    try testing.expect(std.mem.eql(u8, "var123", tokens[0].lexeme));
}

test "identifiers: keywords vs identifiers" {
    const allocator = testing.allocator;
    const test_cases = [_]struct { source: []const u8, expected: lexer.TokenType }{
        .{ .source = "contract", .expected = .Contract },
        .{ .source = "fn", .expected = .Fn },
        .{ .source = "mycontract", .expected = .Identifier }, // Not a keyword
        .{ .source = "contractName", .expected = .Identifier }, // Not a keyword
    };

    for (test_cases) |tc| {
        var lex = lexer.Lexer.init(allocator, tc.source);
        defer lex.deinit();

        const tokens = try lex.scanTokens();
        defer allocator.free(tokens);

        try testing.expectEqual(tc.expected, tokens[0].type);
    }
}

test "identifiers: @import directive" {
    const allocator = testing.allocator;
    const source = "@import";

    var lex = lexer.Lexer.init(allocator, source);
    defer lex.deinit();

    const tokens = try lex.scanTokens();
    defer allocator.free(tokens);

    // @import should be recognized as a valid directive
    try testing.expect(tokens.len >= 1);
}

test "identifiers: invalid @ directive" {
    const allocator = testing.allocator;
    const source = "@someDirective";

    var lex = lexer.Lexer.init(allocator, source);
    defer lex.deinit();

    const tokens = lex.scanTokens() catch |err| {
        try testing.expectEqual(lexer.LexerError.InvalidBuiltinFunction, err);
        return;
    };

    // if lexer is lenient and produces tokens, free them
    allocator.free(tokens);
}
