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

test "identifiers: unknown @ name tokenizes for later semantic validation" {
    const allocator = testing.allocator;
    const source = "@someDirective";

    var lex = lexer.Lexer.init(allocator, source);
    defer lex.deinit();

    const tokens = try lex.scanTokens();
    defer allocator.free(tokens);

    try testing.expect(tokens.len >= 2);
    try testing.expectEqual(lexer.TokenType.At, tokens[0].type);
    try testing.expectEqual(lexer.TokenType.Identifier, tokens[1].type);
    try testing.expectEqualStrings("someDirective", tokens[1].lexeme);
}
