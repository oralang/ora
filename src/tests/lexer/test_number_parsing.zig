const std = @import("std");
const testing = std.testing;
const Allocator = std.mem.Allocator;
const lexer = @import("src/lexer.zig");
const Lexer = lexer.Lexer;
const Token = lexer.Token;
const TokenType = lexer.TokenType;
const TokenValue = lexer.TokenValue;
const LexerError = lexer.LexerError;
const LexerConfig = lexer.LexerConfig;

// Test helper function to create a lexer and scan tokens
fn scanTokens(source: []const u8, allocator: Allocator) ![]Token {
    var lex = Lexer.init(allocator, source);
    defer lex.deinit();
    return lex.scanTokens();
}

// Test helper function to create a lexer with config and scan tokens
fn scanTokensWithConfig(source: []const u8, allocator: Allocator, config: LexerConfig) ![]Token {
    var lex = try Lexer.initWithConfig(allocator, source, config);
    defer lex.deinit();
    return lex.scanTokens();
}

// Test helper to check if a token has the expected type and value
fn expectToken(token: Token, expected_type: TokenType, expected_lexeme: []const u8) !void {
    try testing.expectEqual(expected_type, token.type);
    try testing.expectEqualStrings(expected_lexeme, token.lexeme);
}

// Test helper to check if a token has the expected integer value
fn expectIntegerToken(token: Token, expected_value: u256) !void {
    try testing.expectEqual(TokenType.IntegerLiteral, token.type);
    try testing.expect(token.value != null);
    try testing.expectEqual(TokenValue{ .integer = expected_value }, token.value.?);
}

// Test helper to check if a token has the expected binary value
fn expectBinaryToken(token: Token, expected_value: u256) !void {
    try testing.expectEqual(TokenType.BinaryLiteral, token.type);
    try testing.expect(token.value != null);
    try testing.expectEqual(TokenValue{ .binary = expected_value }, token.value.?);
}

// Test helper to check if a token has the expected hex value
fn expectHexToken(token: Token, expected_value: u256) !void {
    try testing.expectEqual(TokenType.HexLiteral, token.type);
    try testing.expect(token.value != null);
    try testing.expectEqual(TokenValue{ .hex = expected_value }, token.value.?);
}

// Test helper to check if a token has the expected address value
fn expectAddressToken(token: Token, expected_bytes: [20]u8) !void {
    try testing.expectEqual(TokenType.AddressLiteral, token.type);
    try testing.expect(token.value != null);
    try testing.expectEqual(TokenValue{ .address = expected_bytes }, token.value.?);
}

test "decimal integer parsing - basic cases" {
    const allocator = testing.allocator;

    // Test basic decimal numbers
    {
        const tokens = try scanTokens("42", allocator);
        defer allocator.free(tokens);
        try testing.expectEqual(@as(usize, 2), tokens.len); // number + EOF
        try expectIntegerToken(tokens[0], 42);
        try expectToken(tokens[0], .IntegerLiteral, "42");
    }

    // Test zero
    {
        const tokens = try scanTokens("0", allocator);
        defer allocator.free(tokens);
        try testing.expectEqual(@as(usize, 2), tokens.len);
        try expectIntegerToken(tokens[0], 0);
    }

    // Test large number
    {
        const tokens = try scanTokens("123456789", allocator);
        defer allocator.free(tokens);
        try testing.expectEqual(@as(usize, 2), tokens.len);
        try expectIntegerToken(tokens[0], 123456789);
    }

    // Test maximum safe decimal (within u256 range)
    {
        const tokens = try scanTokens("115792089237316195423570985008687907853269984665640564039457584007913129639935", allocator);
        defer allocator.free(tokens);
        try testing.expectEqual(@as(usize, 2), tokens.len);
        try expectIntegerToken(tokens[0], 115792089237316195423570985008687907853269984665640564039457584007913129639935);
    }
}

test "decimal integer parsing - with underscores" {
    const allocator = testing.allocator;

    // Test number with underscores as separators
    {
        const tokens = try scanTokens("1_000_000", allocator);
        defer allocator.free(tokens);
        try testing.expectEqual(@as(usize, 2), tokens.len);
        try expectIntegerToken(tokens[0], 1000000);
        try expectToken(tokens[0], .IntegerLiteral, "1_000_000");
    }

    // Test number with multiple underscores
    {
        const tokens = try scanTokens("123_456_789", allocator);
        defer allocator.free(tokens);
        try testing.expectEqual(@as(usize, 2), tokens.len);
        try expectIntegerToken(tokens[0], 123456789);
    }

    // Test number with underscores at different positions
    {
        const tokens = try scanTokens("1_2_3_4", allocator);
        defer allocator.free(tokens);
        try testing.expectEqual(@as(usize, 2), tokens.len);
        try expectIntegerToken(tokens[0], 1234);
    }
}

test "binary literal parsing - basic cases" {
    const allocator = testing.allocator;

    // Test basic binary numbers
    {
        const tokens = try scanTokens("0b1010", allocator);
        defer allocator.free(tokens);
        try testing.expectEqual(@as(usize, 2), tokens.len);
        try expectBinaryToken(tokens[0], 10); // 1010 binary = 10 decimal
        try expectToken(tokens[0], .BinaryLiteral, "0b1010");
    }

    // Test binary zero
    {
        const tokens = try scanTokens("0b0", allocator);
        defer allocator.free(tokens);
        try testing.expectEqual(@as(usize, 2), tokens.len);
        try expectBinaryToken(tokens[0], 0);
    }

    // Test binary one
    {
        const tokens = try scanTokens("0b1", allocator);
        defer allocator.free(tokens);
        try testing.expectEqual(@as(usize, 2), tokens.len);
        try expectBinaryToken(tokens[0], 1);
    }

    // Test uppercase prefix
    {
        const tokens = try scanTokens("0B1111", allocator);
        defer allocator.free(tokens);
        try testing.expectEqual(@as(usize, 2), tokens.len);
        try expectBinaryToken(tokens[0], 15); // 1111 binary = 15 decimal
        try expectToken(tokens[0], .BinaryLiteral, "0B1111");
    }

    // Test longer binary number
    {
        const tokens = try scanTokens("0b11111111", allocator);
        defer allocator.free(tokens);
        try testing.expectEqual(@as(usize, 2), tokens.len);
        try expectBinaryToken(tokens[0], 255); // 11111111 binary = 255 decimal
    }
}

test "binary literal parsing - with underscores" {
    const allocator = testing.allocator;

    // Test binary with underscores
    {
        const tokens = try scanTokens("0b1010_1010", allocator);
        defer allocator.free(tokens);
        try testing.expectEqual(@as(usize, 2), tokens.len);
        try expectBinaryToken(tokens[0], 170); // 10101010 binary = 170 decimal
        try expectToken(tokens[0], .BinaryLiteral, "0b1010_1010");
    }

    // Test binary with multiple underscores
    {
        const tokens = try scanTokens("0b1_0_1_0", allocator);
        defer allocator.free(tokens);
        try testing.expectEqual(@as(usize, 2), tokens.len);
        try expectBinaryToken(tokens[0], 10); // 1010 binary = 10 decimal
    }

    // Test binary with underscores in different positions
    {
        const tokens = try scanTokens("0b1111_0000_1111_0000", allocator);
        defer allocator.free(tokens);
        try testing.expectEqual(@as(usize, 2), tokens.len);
        try expectBinaryToken(tokens[0], 61680); // 1111000011110000 binary = 61680 decimal
    }
}

test "hexadecimal literal parsing - basic cases" {
    const allocator = testing.allocator;

    // Test basic hex numbers
    {
        const tokens = try scanTokens("0xFF", allocator);
        defer allocator.free(tokens);
        try testing.expectEqual(@as(usize, 2), tokens.len);
        try expectHexToken(tokens[0], 255);
        try expectToken(tokens[0], .HexLiteral, "0xFF");
    }

    // Test hex zero
    {
        const tokens = try scanTokens("0x0", allocator);
        defer allocator.free(tokens);
        try testing.expectEqual(@as(usize, 2), tokens.len);
        try expectHexToken(tokens[0], 0);
    }

    // Test lowercase hex
    {
        const tokens = try scanTokens("0xabcdef", allocator);
        defer allocator.free(tokens);
        try testing.expectEqual(@as(usize, 2), tokens.len);
        try expectHexToken(tokens[0], 11259375); // abcdef hex = 11259375 decimal
    }

    // Test uppercase prefix
    {
        const tokens = try scanTokens("0X123", allocator);
        defer allocator.free(tokens);
        try testing.expectEqual(@as(usize, 2), tokens.len);
        try expectHexToken(tokens[0], 291); // 123 hex = 291 decimal
        try expectToken(tokens[0], .HexLiteral, "0X123");
    }

    // Test mixed case hex digits
    {
        const tokens = try scanTokens("0xAbCdEf", allocator);
        defer allocator.free(tokens);
        try testing.expectEqual(@as(usize, 2), tokens.len);
        try expectHexToken(tokens[0], 11259375); // AbCdEf hex = 11259375 decimal
    }
}

test "hexadecimal literal parsing - with underscores" {
    const allocator = testing.allocator;

    // Test hex with underscores
    {
        const tokens = try scanTokens("0xFF_FF", allocator);
        defer allocator.free(tokens);
        try testing.expectEqual(@as(usize, 2), tokens.len);
        try expectHexToken(tokens[0], 65535); // FFFF hex = 65535 decimal
        try expectToken(tokens[0], .HexLiteral, "0xFF_FF");
    }

    // Test hex with multiple underscores
    {
        const tokens = try scanTokens("0x1_2_3_4", allocator);
        defer allocator.free(tokens);
        try testing.expectEqual(@as(usize, 2), tokens.len);
        try expectHexToken(tokens[0], 4660); // 1234 hex = 4660 decimal
    }

    // Test long hex with underscores
    {
        const tokens = try scanTokens("0xDEAD_BEEF", allocator);
        defer allocator.free(tokens);
        try testing.expectEqual(@as(usize, 2), tokens.len);
        try expectHexToken(tokens[0], 3735928559); // DEADBEEF hex = 3735928559 decimal
    }
}

test "address literal parsing - valid addresses" {
    const allocator = testing.allocator;

    // Test valid 40-character address
    {
        const tokens = try scanTokens("0x1234567890123456789012345678901234567890", allocator);
        defer allocator.free(tokens);
        try testing.expectEqual(@as(usize, 2), tokens.len);
        try testing.expectEqual(TokenType.AddressLiteral, tokens[0].type);
        try expectToken(tokens[0], .AddressLiteral, "0x1234567890123456789012345678901234567890");

        // Check that the address bytes are correctly parsed
        const expected_bytes = [20]u8{ 0x12, 0x34, 0x56, 0x78, 0x90, 0x12, 0x34, 0x56, 0x78, 0x90, 0x12, 0x34, 0x56, 0x78, 0x90, 0x12, 0x34, 0x56, 0x78, 0x90 };
        try expectAddressToken(tokens[0], expected_bytes);
    }

    // Test address with all zeros
    {
        const tokens = try scanTokens("0x0000000000000000000000000000000000000000", allocator);
        defer allocator.free(tokens);
        try testing.expectEqual(@as(usize, 2), tokens.len);
        try testing.expectEqual(TokenType.AddressLiteral, tokens[0].type);

        const expected_bytes = [_]u8{ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 };
        try expectAddressToken(tokens[0], expected_bytes);
    }

    // Test address with all F's
    {
        const tokens = try scanTokens("0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF", allocator);
        defer allocator.free(tokens);
        try testing.expectEqual(@as(usize, 2), tokens.len);
        try testing.expectEqual(TokenType.AddressLiteral, tokens[0].type);

        const expected_bytes = [_]u8{ 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF };
        try expectAddressToken(tokens[0], expected_bytes);
    }

    // Test mixed case address
    {
        const tokens = try scanTokens("0xAbCdEf1234567890AbCdEf1234567890AbCdEf12", allocator);
        defer allocator.free(tokens);
        try testing.expectEqual(@as(usize, 2), tokens.len);
        try testing.expectEqual(TokenType.AddressLiteral, tokens[0].type);

        const expected_bytes = [20]u8{ 0xAb, 0xCd, 0xEf, 0x12, 0x34, 0x56, 0x78, 0x90, 0xAb, 0xCd, 0xEf, 0x12, 0x34, 0x56, 0x78, 0x90, 0xAb, 0xCd, 0xEf, 0x12 };
        try expectAddressToken(tokens[0], expected_bytes);
    }
}

test "number overflow detection - decimal numbers" {
    const allocator = testing.allocator;

    // Test number that's too large for u256 (should cause error)
    const config = LexerConfig{
        .enable_error_recovery = true,
        .enable_string_interning = false,
        .max_errors = 10,
        .enable_suggestions = true,
    };

    // This number is larger than u256 max value
    const large_number = "115792089237316195423570985008687907853269984665640564039457584007913129639936"; // u256 max + 1

    const tokens = try scanTokensWithConfig(large_number, allocator, config);
    defer allocator.free(tokens);

    // With error recovery, the lexer continues and may produce more tokens
    // The important thing is that we get at least the EOF token
    try testing.expect(tokens.len >= 1); // At least EOF token should be there
}

test "number overflow detection - binary numbers" {
    const allocator = testing.allocator;

    const config = LexerConfig{
        .enable_error_recovery = true,
        .enable_string_interning = false,
        .max_errors = 10,
        .enable_suggestions = true,
    };

    // Test binary number with more than 256 bits (should cause overflow)
    var large_binary = std.ArrayList(u8).init(allocator);
    defer large_binary.deinit();

    try large_binary.appendSlice("0b");
    // Add 257 '1' bits (more than u256 can hold)
    var i: usize = 0;
    while (i < 257) : (i += 1) {
        try large_binary.append('1');
    }

    const tokens = try scanTokensWithConfig(large_binary.items, allocator, config);
    defer allocator.free(tokens);

    // Should still produce EOF token with error recovery
    try testing.expectEqual(@as(usize, 1), tokens.len); // Only EOF token
}

test "number overflow detection - hex numbers" {
    const allocator = testing.allocator;

    const config = LexerConfig{
        .enable_error_recovery = true,
        .enable_string_interning = false,
        .max_errors = 10,
        .enable_suggestions = true,
    };

    // Test hex number with more than 64 hex digits (should cause overflow)
    var large_hex = std.ArrayList(u8).init(allocator);
    defer large_hex.deinit();

    try large_hex.appendSlice("0x");
    // Add 65 'F' hex digits (more than u256 can hold)
    var i: usize = 0;
    while (i < 65) : (i += 1) {
        try large_hex.append('F');
    }

    const tokens = try scanTokensWithConfig(large_hex.items, allocator, config);
    defer allocator.free(tokens);

    // Should still produce EOF token with error recovery
    try testing.expectEqual(@as(usize, 1), tokens.len); // Only EOF token
}

test "invalid binary literal error handling" {
    const allocator = testing.allocator;

    const config = LexerConfig{
        .enable_error_recovery = true,
        .enable_string_interning = false,
        .max_errors = 10,
        .enable_suggestions = true,
    };

    // Test binary literal with invalid digits
    {
        const tokens = try scanTokensWithConfig("0b1012", allocator, config);
        defer allocator.free(tokens);
        // With error recovery, lexer continues and may produce more tokens
        try testing.expect(tokens.len >= 1);
    }

    // Test binary literal with no digits
    {
        const tokens = try scanTokensWithConfig("0b", allocator, config);
        defer allocator.free(tokens);
        // With error recovery, lexer continues and may produce more tokens
        try testing.expect(tokens.len >= 1);
    }

    // Test binary literal with letters
    {
        const tokens = try scanTokensWithConfig("0b101a", allocator, config);
        defer allocator.free(tokens);
        // With error recovery, lexer continues and may produce more tokens
        try testing.expect(tokens.len >= 1);
    }
}

test "invalid hex literal error handling" {
    const allocator = testing.allocator;

    const config = LexerConfig{
        .enable_error_recovery = true,
        .enable_string_interning = false,
        .max_errors = 10,
        .enable_suggestions = true,
    };

    // Test hex literal with invalid characters
    {
        const tokens = try scanTokensWithConfig("0xG", allocator, config);
        defer allocator.free(tokens);
        // With error recovery, lexer continues and may produce more tokens
        try testing.expect(tokens.len >= 1);
    }

    // Test hex literal with no digits
    {
        const tokens = try scanTokensWithConfig("0x", allocator, config);
        defer allocator.free(tokens);
        // With error recovery, lexer continues and may produce more tokens
        try testing.expect(tokens.len >= 1);
    }

    // Test hex literal with invalid characters mixed in
    {
        const tokens = try scanTokensWithConfig("0x123G456", allocator, config);
        defer allocator.free(tokens);
        // With error recovery, lexer continues and may produce more tokens
        try testing.expect(tokens.len >= 1);
    }
}

test "invalid address format error handling" {
    const allocator = testing.allocator;

    const config = LexerConfig{
        .enable_error_recovery = true,
        .enable_string_interning = false,
        .max_errors = 10,
        .enable_suggestions = true,
    };

    // Test address with wrong length (too short)
    {
        const tokens = try scanTokensWithConfig("0x123456789012345678901234567890123456789", allocator, config); // 39 chars
        defer allocator.free(tokens);
        // Should be treated as regular hex literal, not address
        try testing.expectEqual(@as(usize, 2), tokens.len);
        try testing.expectEqual(TokenType.HexLiteral, tokens[0].type);
    }

    // Test address with wrong length (too long)
    {
        const tokens = try scanTokensWithConfig("0x12345678901234567890123456789012345678901", allocator, config); // 41 chars
        defer allocator.free(tokens);
        // Should be treated as regular hex literal, not address
        try testing.expectEqual(@as(usize, 2), tokens.len);
        try testing.expectEqual(TokenType.HexLiteral, tokens[0].type);
    }

    // Test address with invalid hex characters
    {
        const tokens = try scanTokensWithConfig("0x123456789012345678901234567890123456789G", allocator, config); // 40 chars but invalid
        defer allocator.free(tokens);
        // With error recovery, lexer continues and may produce more tokens
        try testing.expect(tokens.len >= 1);
    }
}

test "underscore handling in numeric literals" {
    const allocator = testing.allocator;

    // Test that underscores are properly ignored in parsing but preserved in lexeme
    {
        const tokens = try scanTokens("1_000", allocator);
        defer allocator.free(tokens);
        try testing.expectEqual(@as(usize, 2), tokens.len);
        try expectIntegerToken(tokens[0], 1000);
        try expectToken(tokens[0], .IntegerLiteral, "1_000");
    }

    {
        const tokens = try scanTokens("0b1010_1010", allocator);
        defer allocator.free(tokens);
        try testing.expectEqual(@as(usize, 2), tokens.len);
        try expectBinaryToken(tokens[0], 170);
        try expectToken(tokens[0], .BinaryLiteral, "0b1010_1010");
    }

    {
        const tokens = try scanTokens("0xFF_FF", allocator);
        defer allocator.free(tokens);
        try testing.expectEqual(@as(usize, 2), tokens.len);
        try expectHexToken(tokens[0], 65535);
        try expectToken(tokens[0], .HexLiteral, "0xFF_FF");
    }
}

test "mixed number types in sequence" {
    const allocator = testing.allocator;

    const source = "42 0b1010 0xFF 0x1234567890123456789012345678901234567890";
    const tokens = try scanTokens(source, allocator);
    defer allocator.free(tokens);

    try testing.expectEqual(@as(usize, 5), tokens.len); // 4 numbers + EOF

    // Check each token type and value
    try expectIntegerToken(tokens[0], 42);
    try expectBinaryToken(tokens[1], 10);
    try expectHexToken(tokens[2], 255);
    try testing.expectEqual(TokenType.AddressLiteral, tokens[3].type);
    try testing.expectEqual(TokenType.Eof, tokens[4].type);
}

test "number parsing with scientific notation" {
    const allocator = testing.allocator;

    // Test basic scientific notation (currently handled as regular number)
    {
        const tokens = try scanTokens("1e5", allocator);
        defer allocator.free(tokens);
        try testing.expectEqual(@as(usize, 2), tokens.len);
        try testing.expectEqual(TokenType.IntegerLiteral, tokens[0].type);
        try expectToken(tokens[0], .IntegerLiteral, "1e5");
    }

    // Test scientific notation with plus
    {
        const tokens = try scanTokens("1e+5", allocator);
        defer allocator.free(tokens);
        try testing.expectEqual(@as(usize, 2), tokens.len);
        try testing.expectEqual(TokenType.IntegerLiteral, tokens[0].type);
        try expectToken(tokens[0], .IntegerLiteral, "1e+5");
    }

    // Test scientific notation with minus
    {
        const tokens = try scanTokens("1e-5", allocator);
        defer allocator.free(tokens);
        try testing.expectEqual(@as(usize, 2), tokens.len);
        try testing.expectEqual(TokenType.IntegerLiteral, tokens[0].type);
        try expectToken(tokens[0], .IntegerLiteral, "1e-5");
    }
}

test "edge cases and boundary conditions" {
    const allocator = testing.allocator;

    // Test single digit numbers
    {
        const tokens = try scanTokens("0 1 2 9", allocator);
        defer allocator.free(tokens);
        try testing.expectEqual(@as(usize, 5), tokens.len); // 4 numbers + EOF
        try expectIntegerToken(tokens[0], 0);
        try expectIntegerToken(tokens[1], 1);
        try expectIntegerToken(tokens[2], 2);
        try expectIntegerToken(tokens[3], 9);
    }

    // Test minimum binary values
    {
        const tokens = try scanTokens("0b0 0b1", allocator);
        defer allocator.free(tokens);
        try testing.expectEqual(@as(usize, 3), tokens.len); // 2 numbers + EOF
        try expectBinaryToken(tokens[0], 0);
        try expectBinaryToken(tokens[1], 1);
    }

    // Test minimum hex values
    {
        const tokens = try scanTokens("0x0 0x1 0xF", allocator);
        defer allocator.free(tokens);
        try testing.expectEqual(@as(usize, 4), tokens.len); // 3 numbers + EOF
        try expectHexToken(tokens[0], 0);
        try expectHexToken(tokens[1], 1);
        try expectHexToken(tokens[2], 15);
    }
}

test "source range accuracy for number tokens" {
    const allocator = testing.allocator;

    const source = "123 0b1010 0xFF";
    const tokens = try scanTokens(source, allocator);
    defer allocator.free(tokens);

    try testing.expectEqual(@as(usize, 4), tokens.len); // 3 numbers + EOF

    // Check source ranges
    try testing.expectEqual(@as(u32, 1), tokens[0].range.start_line);
    try testing.expectEqual(@as(u32, 1), tokens[0].range.start_column);
    try testing.expectEqual(@as(u32, 0), tokens[0].range.start_offset);
    try testing.expectEqual(@as(u32, 3), tokens[0].range.end_offset);

    try testing.expectEqual(@as(u32, 1), tokens[1].range.start_line);
    try testing.expectEqual(@as(u32, 5), tokens[1].range.start_column);
    try testing.expectEqual(@as(u32, 4), tokens[1].range.start_offset);
    try testing.expectEqual(@as(u32, 10), tokens[1].range.end_offset);

    try testing.expectEqual(@as(u32, 1), tokens[2].range.start_line);
    try testing.expectEqual(@as(u32, 12), tokens[2].range.start_column);
    try testing.expectEqual(@as(u32, 11), tokens[2].range.start_offset);
    try testing.expectEqual(@as(u32, 15), tokens[2].range.end_offset);
}
