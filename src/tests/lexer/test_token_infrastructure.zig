const std = @import("std");
const lexer = @import("src/lexer.zig");
const Token = lexer.Token;
const TokenType = lexer.TokenType;
const SourceRange = lexer.SourceRange;
const TokenValue = lexer.TokenValue;

test "SourceRange struct has all required fields" {
    const range = SourceRange{
        .start_line = 1,
        .start_column = 5,
        .end_line = 1,
        .end_column = 10,
        .start_offset = 4,
        .end_offset = 9,
    };
    
    try std.testing.expect(range.start_line == 1);
    try std.testing.expect(range.start_column == 5);
    try std.testing.expect(range.end_line == 1);
    try std.testing.expect(range.end_column == 10);
    try std.testing.expect(range.start_offset == 4);
    try std.testing.expect(range.end_offset == 9);
}

test "TokenValue union supports all required types" {
    const string_val = TokenValue{ .string = "hello" };
    const char_val = TokenValue{ .character = 'A' };
    const int_val = TokenValue{ .integer = 42 };
    const bin_val = TokenValue{ .binary = 0b1010 };
    const hex_val = TokenValue{ .hex = 0xFF };
    const addr_val = TokenValue{ .address = [_]u8{0} ** 20 };
    const bool_val = TokenValue{ .boolean = true };
    
    switch (string_val) {
        .string => |s| try std.testing.expectEqualStrings("hello", s),
        else => try std.testing.expect(false),
    }
    
    switch (char_val) {
        .character => |c| try std.testing.expect(c == 'A'),
        else => try std.testing.expect(false),
    }
    
    switch (int_val) {
        .integer => |i| try std.testing.expect(i == 42),
        else => try std.testing.expect(false),
    }
    
    switch (bin_val) {
        .binary => |b| try std.testing.expect(b == 0b1010),
        else => try std.testing.expect(false),
    }
    
    switch (hex_val) {
        .hex => |h| try std.testing.expect(h == 0xFF),
        else => try std.testing.expect(false),
    }
    
    switch (addr_val) {
        .address => |a| try std.testing.expect(a.len == 20),
        else => try std.testing.expect(false),
    }
    
    switch (bool_val) {
        .boolean => |b| try std.testing.expect(b == true),
        else => try std.testing.expect(false),
    }
}

test "Token struct includes SourceRange and optional TokenValue" {
    const range = SourceRange{
        .start_line = 1,
        .start_column = 1,
        .end_line = 1,
        .end_column = 5,
        .start_offset = 0,
        .end_offset = 4,
    };
    
    const token_with_value = Token{
        .type = .StringLiteral,
        .lexeme = "test",
        .range = range,
        .value = TokenValue{ .string = "test" },
        .line = 1,
        .column = 1,
    };
    
    const token_without_value = Token{
        .type = .Identifier,
        .lexeme = "test",
        .range = range,
        .value = null,
        .line = 1,
        .column = 1,
    };
    
    try std.testing.expect(token_with_value.type == .StringLiteral);
    try std.testing.expectEqualStrings("test", token_with_value.lexeme);
    try std.testing.expect(token_with_value.range.start_line == 1);
    try std.testing.expect(token_with_value.value != null);
    
    try std.testing.expect(token_without_value.type == .Identifier);
    try std.testing.expectEqualStrings("test", token_without_value.lexeme);
    try std.testing.expect(token_without_value.range.start_line == 1);
    try std.testing.expect(token_without_value.value == null);
}

test "Token creation methods populate source ranges accurately" {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();
    
    const source = "let x = \"hello\";";
    const tokens = try lexer.scan(source, allocator);
    defer allocator.free(tokens);
    
    // Verify that tokens have accurate source ranges
    try std.testing.expect(tokens.len > 0);
    
    // Check the first token (let)
    const let_token = tokens[0];
    try std.testing.expect(let_token.type == .Let);
    try std.testing.expectEqualStrings("let", let_token.lexeme);
    try std.testing.expect(let_token.range.start_line == 1);
    try std.testing.expect(let_token.range.start_column == 1);
    try std.testing.expect(let_token.range.end_column == 4);
    try std.testing.expect(let_token.range.start_offset == 0);
    try std.testing.expect(let_token.range.end_offset == 3);
    
    // Check the string literal token
    var string_token: ?Token = null;
    for (tokens) |token| {
        if (token.type == .StringLiteral) {
            string_token = token;
            break;
        }
    }
    
    try std.testing.expect(string_token != null);
    if (string_token) |str_tok| {
        try std.testing.expectEqualStrings("hello", str_tok.lexeme);
        try std.testing.expect(str_tok.range.start_line == 1);
        try std.testing.expect(str_tok.range.start_column == 9); // Position of opening quote
        try std.testing.expect(str_tok.range.end_column == 16); // Position after closing quote
        try std.testing.expect(str_tok.value != null);
        if (str_tok.value) |val| {
            switch (val) {
                .string => |s| try std.testing.expectEqualStrings("hello", s),
                else => try std.testing.expect(false),
            }
        }
    }
}

test "Boolean literals have TokenValue" {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();
    
    const source = "true false";
    const tokens = try lexer.scan(source, allocator);
    defer allocator.free(tokens);
    
    var true_token: ?Token = null;
    var false_token: ?Token = null;
    
    for (tokens) |token| {
        if (token.type == .True) {
            true_token = token;
        } else if (token.type == .False) {
            false_token = token;
        }
    }
    
    try std.testing.expect(true_token != null);
    try std.testing.expect(false_token != null);
    
    if (true_token) |tok| {
        try std.testing.expect(tok.value != null);
        if (tok.value) |val| {
            switch (val) {
                .boolean => |b| try std.testing.expect(b == true),
                else => try std.testing.expect(false),
            }
        }
    }
    
    if (false_token) |tok| {
        try std.testing.expect(tok.value != null);
        if (tok.value) |val| {
            switch (val) {
                .boolean => |b| try std.testing.expect(b == false),
                else => try std.testing.expect(false),
            }
        }
    }
}