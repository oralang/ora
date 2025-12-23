// ============================================================================
// Lexer Unit Tests
// ============================================================================
// Tests for tokenization, position tracking, and error handling
// ============================================================================

const std = @import("std");
const testing = std.testing;
const ora_root = @import("root.zig");
const lexer = ora_root.lexer;

// ============================================================================
// Basic Tokenization Tests
// ============================================================================

test "keywords: contract token recognized" {
    const allocator = testing.allocator;
    const source = "contract";

    var lex = lexer.Lexer.init(allocator, source);
    defer lex.deinit();

    const tokens = try lex.scanTokens();
    defer allocator.free(tokens);

    try testing.expect(tokens.len >= 2); // contract + EOF
    try testing.expectEqual(lexer.TokenType.Contract, tokens[0].type);
    try testing.expect(std.mem.eql(u8, "contract", tokens[0].lexeme));
}

test "keywords: all keywords recognized" {
    const allocator = testing.allocator;
    const keywords = [_]struct { source: []const u8, expected: lexer.TokenType }{
        .{ .source = "contract", .expected = .Contract },
        .{ .source = "fn", .expected = .Fn },
        .{ .source = "pub", .expected = .Pub },
        .{ .source = "storage", .expected = .Storage },
        .{ .source = "return", .expected = .Return },
        .{ .source = "if", .expected = .If },
        .{ .source = "else", .expected = .Else },
        .{ .source = "while", .expected = .While },
        .{ .source = "for", .expected = .For },
        .{ .source = "switch", .expected = .Switch },
        .{ .source = "let", .expected = .Let },
        .{ .source = "var", .expected = .Var },
    };

    for (keywords) |kw| {
        var lex = lexer.Lexer.init(allocator, kw.source);
        defer lex.deinit();

        const tokens = try lex.scanTokens();
        defer allocator.free(tokens);

        try testing.expectEqual(kw.expected, tokens[0].type);
    }
}

test "operators: arithmetic operators" {
    const allocator = testing.allocator;
    const operators = [_]struct { source: []const u8, expected: lexer.TokenType }{
        .{ .source = "+", .expected = .Plus },
        .{ .source = "-", .expected = .Minus },
        .{ .source = "*", .expected = .Star },
        .{ .source = "/", .expected = .Slash },
        .{ .source = "%", .expected = .Percent },
    };

    for (operators) |op| {
        var lex = lexer.Lexer.init(allocator, op.source);
        defer lex.deinit();

        const tokens = try lex.scanTokens();
        defer allocator.free(tokens);

        try testing.expectEqual(op.expected, tokens[0].type);
    }
}

test "literals: integer literals" {
    const allocator = testing.allocator;
    const source = "123";

    var lex = lexer.Lexer.init(allocator, source);
    defer lex.deinit();

    const tokens = try lex.scanTokens();
    defer allocator.free(tokens);

    try testing.expectEqual(lexer.TokenType.IntegerLiteral, tokens[0].type);
    try testing.expect(std.mem.eql(u8, "123", tokens[0].lexeme));
}

test "literals: hex literals" {
    const allocator = testing.allocator;
    const source = "0xFF";

    var lex = lexer.Lexer.init(allocator, source);
    defer lex.deinit();

    const tokens = try lex.scanTokens();
    defer allocator.free(tokens);

    try testing.expectEqual(lexer.TokenType.HexLiteral, tokens[0].type);
    try testing.expect(std.mem.eql(u8, "0xFF", tokens[0].lexeme));
}

test "literals: string literals" {
    const allocator = testing.allocator;
    const source = "\"hello\"";

    var lex = lexer.Lexer.init(allocator, source);
    defer lex.deinit();

    const tokens = try lex.scanTokens();
    defer allocator.free(tokens);

    try testing.expectEqual(lexer.TokenType.StringLiteral, tokens[0].type);
    // Lexeme contains the string content (without quotes)
    try testing.expect(std.mem.indexOf(u8, tokens[0].lexeme, "hello") != null);
}

// ============================================================================
// Position Tracking Tests
// ============================================================================

test "position: line numbers increment" {
    const allocator = testing.allocator;
    const source = "contract\nfn\npub";

    var lex = lexer.Lexer.init(allocator, source);
    defer lex.deinit();

    const tokens = try lex.scanTokens();
    defer allocator.free(tokens);

    try testing.expectEqual(@as(u32, 1), tokens[0].range.start_line); // contract
    try testing.expectEqual(@as(u32, 2), tokens[1].range.start_line); // fn
    try testing.expectEqual(@as(u32, 3), tokens[2].range.start_line); // pub
}

test "position: column numbers reset on newline" {
    const allocator = testing.allocator;
    const source = "contract\nfn";

    var lex = lexer.Lexer.init(allocator, source);
    defer lex.deinit();

    const tokens = try lex.scanTokens();
    defer allocator.free(tokens);

    try testing.expectEqual(@as(u32, 1), tokens[0].range.start_column); // contract starts at col 1
    try testing.expectEqual(@as(u32, 1), tokens[1].range.start_column); // fn starts at col 1
}

// ============================================================================
// Error Handling Tests
// ============================================================================

test "errors: unterminated string" {
    const allocator = testing.allocator;
    const source = "\"hello";

    var lex = lexer.Lexer.init(allocator, source);
    defer lex.deinit();

    const tokens = lex.scanTokens() catch |err| {
        try testing.expectEqual(lexer.LexerError.UnterminatedString, err);
        return;
    };

    // Should not reach here
    allocator.free(tokens);
    try testing.expect(false);
}

// ============================================================================
// Edge Cases
// ============================================================================

test "edge cases: empty input" {
    const allocator = testing.allocator;
    const source = "";

    var lex = lexer.Lexer.init(allocator, source);
    defer lex.deinit();

    const tokens = try lex.scanTokens();
    defer allocator.free(tokens);

    try testing.expect(tokens.len >= 1); // At least EOF
    try testing.expectEqual(lexer.TokenType.Eof, tokens[tokens.len - 1].type);
}

test "edge cases: whitespace only" {
    const allocator = testing.allocator;
    const source = "   \n\t  ";

    var lex = lexer.Lexer.init(allocator, source);
    defer lex.deinit();

    const tokens = try lex.scanTokens();
    defer allocator.free(tokens);

    // Should have at least EOF (whitespace is trivia)
    try testing.expect(tokens.len >= 1);
    try testing.expectEqual(lexer.TokenType.Eof, tokens[tokens.len - 1].type);
}
