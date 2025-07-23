const std = @import("std");
const testing = std.testing;
const lexer = @import("src/lexer.zig");

test "SourceRange struct has all required fields" {
    const range = lexer.SourceRange{
        .start_line = 1,
        .start_column = 5,
        .end_line = 1,
        .end_column = 10,
        .start_offset = 4,
        .end_offset = 9,
    };

    try testing.expect(range.start_line == 1);
    try testing.expect(range.start_column == 5);
    try testing.expect(range.end_line == 1);
    try testing.expect(range.end_column == 10);
    try testing.expect(range.start_offset == 4);
    try testing.expect(range.end_offset == 9);
}

test "TokenValue union supports all required types" {
    const allocator = testing.allocator;

    // Test string value
    const string_val = lexer.TokenValue{ .string = "hello" };
    try testing.expect(std.mem.eql(u8, string_val.string, "hello"));

    // Test character value
    const char_val = lexer.TokenValue{ .character = 'A' };
    try testing.expect(char_val.character == 'A');

    // Test integer value
    const int_val = lexer.TokenValue{ .integer = 42 };
    try testing.expect(int_val.integer == 42);

    // Test binary value
    const bin_val = lexer.TokenValue{ .binary = 0b1010 };
    try testing.expect(bin_val.binary == 10);

    // Test hex value
    const hex_val = lexer.TokenValue{ .hex = 0xFF };
    try testing.expect(hex_val.hex == 255);

    // Test address value
    const addr_val = lexer.TokenValue{ .address = [_]u8{0} ** 20 };
    try testing.expect(addr_val.address.len == 20);

    // Test boolean value
    const bool_val = lexer.TokenValue{ .boolean = true };
    try testing.expect(bool_val.boolean == true);

    _ = allocator; // Suppress unused variable warning
}

test "Token struct includes SourceRange and optional TokenValue" {
    const range = lexer.SourceRange{
        .start_line = 1,
        .start_column = 1,
        .end_line = 1,
        .end_column = 5,
        .start_offset = 0,
        .end_offset = 4,
    };

    // Test token without value
    const token_no_value = lexer.Token{
        .type = .Identifier,
        .lexeme = "test",
        .range = range,
        .value = null,
        .line = 1,
        .column = 1,
    };

    try testing.expect(token_no_value.type == .Identifier);
    try testing.expect(std.mem.eql(u8, token_no_value.lexeme, "test"));
    try testing.expect(token_no_value.range.start_line == 1);
    try testing.expect(token_no_value.value == null);

    // Test token with value
    const token_with_value = lexer.Token{
        .type = .IntegerLiteral,
        .lexeme = "42",
        .range = range,
        .value = lexer.TokenValue{ .integer = 42 },
        .line = 1,
        .column = 1,
    };

    try testing.expect(token_with_value.type == .IntegerLiteral);
    try testing.expect(std.mem.eql(u8, token_with_value.lexeme, "42"));
    try testing.expect(token_with_value.value != null);
    try testing.expect(token_with_value.value.?.integer == 42);
}

test "Token creation methods populate source ranges accurately" {
    const allocator = testing.allocator;
    const source = "true false 42";

    var lex = lexer.Lexer.init(allocator, source);
    defer lex.deinit();

    const tokens = try lex.scanTokens();
    defer allocator.free(tokens);

    // Should have: true, false, 42, EOF
    try testing.expect(tokens.len == 4);

    // Test 'true' token
    const true_token = tokens[0];
    try testing.expect(true_token.type == .True);
    try testing.expect(std.mem.eql(u8, true_token.lexeme, "true"));
    try testing.expect(true_token.range.start_line == 1);
    try testing.expect(true_token.range.start_column == 1);
    try testing.expect(true_token.range.end_column == 5);
    try testing.expect(true_token.range.start_offset == 0);
    try testing.expect(true_token.range.end_offset == 4);
    try testing.expect(true_token.value != null);
    try testing.expect(true_token.value.?.boolean == true);

    // Test 'false' token
    const false_token = tokens[1];
    try testing.expect(false_token.type == .False);
    try testing.expect(std.mem.eql(u8, false_token.lexeme, "false"));
    try testing.expect(false_token.range.start_line == 1);
    try testing.expect(false_token.range.start_column == 6);
    try testing.expect(false_token.range.end_column == 11);
    try testing.expect(false_token.range.start_offset == 5);
    try testing.expect(false_token.range.end_offset == 10);
    try testing.expect(false_token.value != null);
    try testing.expect(false_token.value.?.boolean == false);

    // Test integer token
    const int_token = tokens[2];
    try testing.expect(int_token.type == .IntegerLiteral);
    try testing.expect(std.mem.eql(u8, int_token.lexeme, "42"));
    try testing.expect(int_token.range.start_line == 1);
    try testing.expect(int_token.range.start_column == 12);
    try testing.expect(int_token.range.end_column == 14);
    try testing.expect(int_token.range.start_offset == 11);
    try testing.expect(int_token.range.end_offset == 13);
    try testing.expect(int_token.value != null);
    try testing.expect(int_token.value.?.integer == 42);

    // Test EOF token
    const eof_token = tokens[3];
    try testing.expect(eof_token.type == .Eof);
    try testing.expect(std.mem.eql(u8, eof_token.lexeme, ""));
    try testing.expect(eof_token.range.start_line == 1);
    try testing.expect(eof_token.range.start_column == 14);
}

test "Multi-line source ranges are tracked correctly" {
    const allocator = testing.allocator;
    const source = "line1\nline2\nline3";

    var lex = lexer.Lexer.init(allocator, source);
    defer lex.deinit();

    const tokens = try lex.scanTokens();
    defer allocator.free(tokens);

    // Should have: line1, line2, line3, EOF
    try testing.expect(tokens.len == 4);

    // Test first identifier
    const token1 = tokens[0];
    try testing.expect(token1.type == .Identifier);
    try testing.expect(std.mem.eql(u8, token1.lexeme, "line1"));
    try testing.expect(token1.range.start_line == 1);
    try testing.expect(token1.range.start_column == 1);

    // Test second identifier
    const token2 = tokens[1];
    try testing.expect(token2.type == .Identifier);
    try testing.expect(std.mem.eql(u8, token2.lexeme, "line2"));
    try testing.expect(token2.range.start_line == 2);
    try testing.expect(token2.range.start_column == 1);

    // Test third identifier
    const token3 = tokens[2];
    try testing.expect(token3.type == .Identifier);
    try testing.expect(std.mem.eql(u8, token3.lexeme, "line3"));
    try testing.expect(token3.range.start_line == 3);
    try testing.expect(token3.range.start_column == 1);
}

test "String literals have correct TokenValue" {
    const allocator = testing.allocator;
    const source = "\"hello world\"";

    var lex = lexer.Lexer.init(allocator, source);
    defer lex.deinit();

    const tokens = try lex.scanTokens();
    defer allocator.free(tokens);

    // Should have: string literal, EOF
    try testing.expect(tokens.len == 2);

    const string_token = tokens[0];
    try testing.expect(string_token.type == .StringLiteral);
    try testing.expect(std.mem.eql(u8, string_token.lexeme, "hello world"));
    try testing.expect(string_token.value != null);
    try testing.expect(std.mem.eql(u8, string_token.value.?.string, "hello world"));
}

test "Character literals have correct TokenValue" {
    const allocator = testing.allocator;
    const source = "'A'";

    var lex = lexer.Lexer.init(allocator, source);
    defer lex.deinit();

    const tokens = try lex.scanTokens();
    defer allocator.free(tokens);

    // Should have: character literal, EOF
    try testing.expect(tokens.len == 2);

    const char_token = tokens[0];
    try testing.expect(char_token.type == .CharacterLiteral);
    try testing.expect(std.mem.eql(u8, char_token.lexeme, "A"));
    try testing.expect(char_token.value != null);
    try testing.expect(char_token.value.?.character == 'A');
}
