// ============================================================================
// Number Literal Scanner Tests
// ============================================================================

const std = @import("std");
const testing = std.testing;
const ora_root = @import("ora_root");
const lexer = ora_root.lexer;

test "numbers: decimal literals" {
    const allocator = testing.allocator;
    const test_cases = [_][]const u8{ "0", "123", "999999", "42" };

    for (test_cases) |source| {
        var lex = lexer.Lexer.init(allocator, source);
        defer lex.deinit();

        const tokens = try lex.scanTokens();
        defer allocator.free(tokens);

        try testing.expectEqual(lexer.TokenType.IntegerLiteral, tokens[0].type);
        try testing.expect(std.mem.eql(u8, source, tokens[0].lexeme));
    }
}

test "numbers: hex literals" {
    const allocator = testing.allocator;
    const test_cases = [_]struct { source: []const u8, expected: lexer.TokenType }{
        .{ .source = "0x0", .expected = .HexLiteral },
        .{ .source = "0xFF", .expected = .HexLiteral },
        .{ .source = "0x123ABC", .expected = .HexLiteral },
        .{ .source = "0XFF", .expected = .HexLiteral },
    };

    for (test_cases) |tc| {
        var lex = lexer.Lexer.init(allocator, tc.source);
        defer lex.deinit();

        const tokens = try lex.scanTokens();
        defer allocator.free(tokens);

        try testing.expectEqual(tc.expected, tokens[0].type);
        try testing.expect(std.mem.eql(u8, tc.source, tokens[0].lexeme));
    }
}

test "numbers: binary literals" {
    const allocator = testing.allocator;
    const test_cases = [_][]const u8{ "0b0", "0b1010", "0b11111111" };

    for (test_cases) |source| {
        var lex = lexer.Lexer.init(allocator, source);
        defer lex.deinit();

        const tokens = try lex.scanTokens();
        defer allocator.free(tokens);

        try testing.expectEqual(lexer.TokenType.BinaryLiteral, tokens[0].type);
        try testing.expect(std.mem.eql(u8, source, tokens[0].lexeme));
    }
}

test "numbers: invalid hex literal" {
    const allocator = testing.allocator;
    const source = "0xG";

    var lex = lexer.Lexer.init(allocator, source);
    defer lex.deinit();

    const tokens = lex.scanTokens() catch |err| {
        try testing.expectEqual(lexer.LexerError.InvalidHexLiteral, err);
        return;
    };

    allocator.free(tokens);
    try testing.expect(false);
}

test "numbers: hex with underscores" {
    const allocator = testing.allocator;
    const source = "0xFF_FF";

    var lex = lexer.Lexer.init(allocator, source);
    defer lex.deinit();

    const tokens = try lex.scanTokens();
    defer allocator.free(tokens);

    try testing.expectEqual(lexer.TokenType.HexLiteral, tokens[0].type);
}
