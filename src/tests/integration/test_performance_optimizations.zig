const std = @import("std");
const testing = std.testing;
const lexer = @import("src/lexer.zig");

test "Character classification lookup tables" {
    // Test digit classification
    try testing.expect(lexer.isDigit('0'));
    try testing.expect(lexer.isDigit('5'));
    try testing.expect(lexer.isDigit('9'));
    try testing.expect(!lexer.isDigit('a'));
    try testing.expect(!lexer.isDigit(' '));

    // Test hex digit classification
    try testing.expect(lexer.isHexDigit('0'));
    try testing.expect(lexer.isHexDigit('9'));
    try testing.expect(lexer.isHexDigit('a'));
    try testing.expect(lexer.isHexDigit('f'));
    try testing.expect(lexer.isHexDigit('A'));
    try testing.expect(lexer.isHexDigit('F'));
    try testing.expect(!lexer.isHexDigit('g'));
    try testing.expect(!lexer.isHexDigit('G'));

    // Test binary digit classification
    try testing.expect(lexer.isBinaryDigit('0'));
    try testing.expect(lexer.isBinaryDigit('1'));
    try testing.expect(!lexer.isBinaryDigit('2'));
    try testing.expect(!lexer.isBinaryDigit('a'));

    // Test alpha classification
    try testing.expect(lexer.isAlpha('a'));
    try testing.expect(lexer.isAlpha('z'));
    try testing.expect(lexer.isAlpha('A'));
    try testing.expect(lexer.isAlpha('Z'));
    try testing.expect(lexer.isAlpha('_'));
    try testing.expect(!lexer.isAlpha('0'));
    try testing.expect(!lexer.isAlpha(' '));

    // Test alphanumeric classification
    try testing.expect(lexer.isAlphaNumeric('a'));
    try testing.expect(lexer.isAlphaNumeric('Z'));
    try testing.expect(lexer.isAlphaNumeric('_'));
    try testing.expect(lexer.isAlphaNumeric('0'));
    try testing.expect(lexer.isAlphaNumeric('9'));
    try testing.expect(!lexer.isAlphaNumeric(' '));
    try testing.expect(!lexer.isAlphaNumeric('+'));

    // Test identifier start classification
    try testing.expect(lexer.isIdentifierStart('a'));
    try testing.expect(lexer.isIdentifierStart('Z'));
    try testing.expect(lexer.isIdentifierStart('_'));
    try testing.expect(!lexer.isIdentifierStart('0'));
    try testing.expect(!lexer.isIdentifierStart(' '));

    // Test whitespace classification
    try testing.expect(lexer.isWhitespace(' '));
    try testing.expect(lexer.isWhitespace('\t'));
    try testing.expect(lexer.isWhitespace('\r'));
    try testing.expect(lexer.isWhitespace('\n'));
    try testing.expect(!lexer.isWhitespace('a'));
    try testing.expect(!lexer.isWhitespace('0'));
}

test "Performance monitoring" {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    const source = "let x = hello; let y = world;";

    var config = lexer.LexerConfig.default();
    config.enable_performance_monitoring = true;
    config.enable_string_interning = true;

    var lex = lexer.Lexer.initWithConfig(allocator, source, config);
    defer lex.deinit();

    const tokens = try lex.scanTokens();
    defer allocator.free(tokens);

    // Check that performance monitoring is working
    if (lex.performance) |perf| {
        try testing.expect(perf.characters_processed > 0);
        try testing.expect(perf.getTokensPerCharacter() > 0.0);

        // Should have some string interning activity
        const total_interning = perf.string_interning_hits + perf.string_interning_misses;
        try testing.expect(total_interning > 0);
    } else {
        try testing.expect(false); // Performance monitoring should be enabled
    }
}

test "Token pre-allocation optimization" {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    // Create a source with many tokens to test pre-allocation
    const source = "let x = 1; let y = 2; let z = 3; fn test() { return x + y + z; }";

    var lex = lexer.Lexer.init(allocator, source);
    defer lex.deinit();

    const tokens = try lex.scanTokens();
    defer allocator.free(tokens);

    // Should have tokenized successfully with pre-allocation
    try testing.expect(tokens.len > 20); // Should have many tokens

    // Verify some key tokens are present
    var let_count: u32 = 0;
    var identifier_count: u32 = 0;
    var number_count: u32 = 0;

    for (tokens) |token| {
        switch (token.type) {
            .Let => let_count += 1,
            .Identifier => identifier_count += 1,
            .IntegerLiteral => number_count += 1,
            else => {},
        }
    }

    try testing.expectEqual(@as(u32, 3), let_count);
    try testing.expect(identifier_count >= 6); // x, y, z, test, x, y, z
    try testing.expectEqual(@as(u32, 3), number_count); // 1, 2, 3
}

test "Fast-path whitespace handling" {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    // Source with lots of whitespace
    const source = "  \t\n  let   \t x   =   \n  42  \t ;  \n  ";

    var lex = lexer.Lexer.init(allocator, source);
    defer lex.deinit();

    const tokens = try lex.scanTokens();
    defer allocator.free(tokens);

    // Should have tokenized correctly despite lots of whitespace
    var non_eof_tokens: u32 = 0;
    for (tokens) |token| {
        if (token.type != .Eof) {
            non_eof_tokens += 1;
        }
    }

    try testing.expectEqual(@as(u32, 5), non_eof_tokens); // let, x, =, 42, ;

    // Check that line numbers are tracked correctly
    var found_let = false;
    var found_number = false;

    for (tokens) |token| {
        if (token.type == .Let) {
            try testing.expectEqual(@as(u32, 2), token.line);
            found_let = true;
        } else if (token.type == .IntegerLiteral) {
            try testing.expectEqual(@as(u32, 3), token.line);
            found_number = true;
        }
    }

    try testing.expect(found_let);
    try testing.expect(found_number);
}

test "Performance comparison with and without optimizations" {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    // Create a moderately complex source
    const source =
        \\contract TestContract {
        \\    let x = 0x1234567890abcdef;
        \\    let y = 0b1010101010101010;
        \\    let name = "hello world";
        \\    let char = 'a';
        \\    
        \\    fn calculate(a: u256, b: u256) -> u256 {
        \\        requires a > 0;
        \\        ensures result > a;
        \\        return a + b * 2;
        \\    }
        \\}
    ;

    // Test with optimizations enabled
    var config_optimized = lexer.LexerConfig.default();
    config_optimized.enable_string_interning = true;
    config_optimized.enable_performance_monitoring = true;

    var lex_optimized = lexer.Lexer.initWithConfig(allocator, source, config_optimized);
    defer lex_optimized.deinit();

    const tokens_optimized = try lex_optimized.scanTokens();
    defer allocator.free(tokens_optimized);

    // Test with optimizations disabled
    var config_basic = lexer.LexerConfig.default();
    config_basic.enable_string_interning = false;
    config_basic.enable_performance_monitoring = false;

    var lex_basic = lexer.Lexer.initWithConfig(allocator, source, config_basic);
    defer lex_basic.deinit();

    const tokens_basic = try lex_basic.scanTokens();
    defer allocator.free(tokens_basic);

    // Both should produce the same number of tokens
    try testing.expectEqual(tokens_optimized.len, tokens_basic.len);

    // Both should produce tokens with the same types and content
    for (tokens_optimized, tokens_basic) |opt_token, basic_token| {
        try testing.expectEqual(opt_token.type, basic_token.type);
        try testing.expectEqualStrings(opt_token.lexeme, basic_token.lexeme);
    }

    // Check performance metrics for optimized version
    if (lex_optimized.performance) |perf| {
        try testing.expect(perf.characters_processed > 0);
        try testing.expect(perf.getTokensPerCharacter() > 0.0);
    }
}
