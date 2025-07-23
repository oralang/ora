const std = @import("std");
const testing = std.testing;
const Allocator = std.mem.Allocator;
const lexer = @import("src/lexer.zig");

const Lexer = lexer.Lexer;
const TokenType = lexer.TokenType;
const LexerError = lexer.LexerError;
const StringProcessor = lexer.StringProcessor;
const LexerConfig = lexer.LexerConfig;

test "string processing - basic escape sequences" {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    var processor = StringProcessor.init(allocator);

    // Test newline escape
    {
        const result = try processor.processString("Hello\\nWorld");
        defer allocator.free(result);
        try testing.expectEqualStrings("Hello\nWorld", result);
    }

    // Test tab escape
    {
        const result = try processor.processString("Hello\\tWorld");
        defer allocator.free(result);
        try testing.expectEqualStrings("Hello\tWorld", result);
    }

    // Test carriage return escape
    {
        const result = try processor.processString("Hello\\rWorld");
        defer allocator.free(result);
        try testing.expectEqualStrings("Hello\rWorld", result);
    }

    // Test backslash escape
    {
        const result = try processor.processString("Hello\\\\World");
        defer allocator.free(result);
        try testing.expectEqualStrings("Hello\\World", result);
    }

    // Test quote escape
    {
        const result = try processor.processString("Hello\\\"World");
        defer allocator.free(result);
        try testing.expectEqualStrings("Hello\"World", result);
    }

    // Test null character escape
    {
        const result = try processor.processString("Hello\\0World");
        defer allocator.free(result);
        try testing.expect(result.len == 11);
        try testing.expect(result[5] == 0);
        try testing.expectEqualStrings("Hello", result[0..5]);
        try testing.expectEqualStrings("World", result[6..]);
    }
}

test "string processing - hex escape sequences" {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    var processor = StringProcessor.init(allocator);

    // Test valid hex escapes
    {
        const result = try processor.processString("\\x41\\x42\\x43");
        defer allocator.free(result);
        try testing.expectEqualStrings("ABC", result);
    }

    // Test lowercase hex digits
    {
        const result = try processor.processString("\\x61\\x62\\x63");
        defer allocator.free(result);
        try testing.expectEqualStrings("abc", result);
    }

    // Test mixed case hex digits
    {
        const result = try processor.processString("\\x4A\\x6b\\x2F");
        defer allocator.free(result);
        try testing.expectEqualStrings("Jk/", result);
    }

    // Test hex escape with null byte
    {
        const result = try processor.processString("\\x00");
        defer allocator.free(result);
        try testing.expect(result.len == 1);
        try testing.expect(result[0] == 0);
    }

    // Test hex escape with high values
    {
        const result = try processor.processString("\\xFF\\xFE");
        defer allocator.free(result);
        try testing.expect(result.len == 2);
        try testing.expect(result[0] == 255);
        try testing.expect(result[1] == 254);
    }
}

test "string processing - invalid escape sequences" {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    var processor = StringProcessor.init(allocator);

    // Test invalid single character escape
    try testing.expectError(LexerError.InvalidEscapeSequence, processor.processString("\\z"));

    // Test incomplete hex escape
    try testing.expectError(LexerError.InvalidEscapeSequence, processor.processString("\\x4"));

    // Test invalid hex digits
    try testing.expectError(LexerError.InvalidEscapeSequence, processor.processString("\\xGH"));

    // Test hex escape without digits
    try testing.expectError(LexerError.InvalidEscapeSequence, processor.processString("\\x"));

    // Test backslash at end of string
    try testing.expectError(LexerError.InvalidEscapeSequence, processor.processString("Hello\\"));
}

test "string processing - complex escape combinations" {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    var processor = StringProcessor.init(allocator);

    // Test multiple escapes in one string
    {
        const result = try processor.processString("Line1\\nLine2\\tTabbed\\\\Backslash\\\"Quote");
        defer allocator.free(result);
        try testing.expectEqualStrings("Line1\nLine2\tTabbed\\Backslash\"Quote", result);
    }

    // Test mixed hex and character escapes
    {
        const result = try processor.processString("\\x48\\x65\\x6C\\x6C\\x6F\\n\\x57\\x6F\\x72\\x6C\\x64");
        defer allocator.free(result);
        try testing.expectEqualStrings("Hello\nWorld", result);
    }

    // Test consecutive escapes
    {
        const result = try processor.processString("\\n\\t\\r\\\\\\\"");
        defer allocator.free(result);
        try testing.expectEqualStrings("\n\t\r\\\"", result);
    }
}

test "character literal processing - basic characters" {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    var processor = StringProcessor.init(allocator);

    // Test simple ASCII characters
    try testing.expectEqual(@as(u8, 'A'), try processor.processCharacterLiteral("A"));
    try testing.expectEqual(@as(u8, 'z'), try processor.processCharacterLiteral("z"));
    try testing.expectEqual(@as(u8, '0'), try processor.processCharacterLiteral("0"));
    try testing.expectEqual(@as(u8, ' '), try processor.processCharacterLiteral(" "));
    try testing.expectEqual(@as(u8, '@'), try processor.processCharacterLiteral("@"));
}

test "character literal processing - escape sequences" {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    var processor = StringProcessor.init(allocator);

    // Test character escapes
    try testing.expectEqual(@as(u8, '\n'), try processor.processCharacterLiteral("\\n"));
    try testing.expectEqual(@as(u8, '\t'), try processor.processCharacterLiteral("\\t"));
    try testing.expectEqual(@as(u8, '\r'), try processor.processCharacterLiteral("\\r"));
    try testing.expectEqual(@as(u8, '\\'), try processor.processCharacterLiteral("\\\\"));
    try testing.expectEqual(@as(u8, '"'), try processor.processCharacterLiteral("\\\""));
    try testing.expectEqual(@as(u8, '\''), try processor.processCharacterLiteral("\\'"));
    try testing.expectEqual(@as(u8, 0), try processor.processCharacterLiteral("\\0"));

    // Test hex escapes in character literals
    try testing.expectEqual(@as(u8, 'A'), try processor.processCharacterLiteral("\\x41"));
    try testing.expectEqual(@as(u8, 255), try processor.processCharacterLiteral("\\xFF"));
    try testing.expectEqual(@as(u8, 0), try processor.processCharacterLiteral("\\x00"));
}

test "character literal processing - error conditions" {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    var processor = StringProcessor.init(allocator);

    // Test empty character literal
    try testing.expectError(LexerError.EmptyCharacterLiteral, processor.processCharacterLiteral(""));

    // Test multiple characters
    try testing.expectError(LexerError.InvalidCharacterLiteral, processor.processCharacterLiteral("AB"));

    // Test invalid escape in character literal
    try testing.expectError(LexerError.InvalidEscapeSequence, processor.processCharacterLiteral("\\z"));

    // Test incomplete hex escape in character literal
    try testing.expectError(LexerError.InvalidEscapeSequence, processor.processCharacterLiteral("\\x4"));

    // Test character literal with extra content after escape
    try testing.expectError(LexerError.InvalidCharacterLiteral, processor.processCharacterLiteral("\\nX"));
}

test "raw string processing - basic functionality" {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    var processor = StringProcessor.init(allocator);

    // Test basic raw string (no escape processing)
    {
        const result = try processor.processRawString("Hello\\nWorld");
        defer allocator.free(result);
        try testing.expectEqualStrings("Hello\\nWorld", result);
    }

    // Test raw string with quotes
    {
        const result = try processor.processRawString("Hello\"World\"Test");
        defer allocator.free(result);
        try testing.expectEqualStrings("Hello\"World\"Test", result);
    }

    // Test raw string with backslashes
    {
        const result = try processor.processRawString("C:\\\\Path\\\\To\\\\File");
        defer allocator.free(result);
        try testing.expectEqualStrings("C:\\\\Path\\\\To\\\\File", result);
    }
}

test "lexer integration - string literals with escapes" {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    const source = "\"Hello\\nWorld\" \"Tab\\tSeparated\" \"Quote\\\"Inside\"";
    var lex = Lexer.init(allocator, source);
    defer lex.deinit();

    const tokens = try lex.scanTokens();

    try testing.expect(tokens.len >= 3);
    try testing.expectEqual(TokenType.StringLiteral, tokens[0].type);
    try testing.expectEqual(TokenType.StringLiteral, tokens[1].type);
    try testing.expectEqual(TokenType.StringLiteral, tokens[2].type);

    // Check that processed values are available
    if (tokens[0].value) |val| {
        switch (val) {
            .string => |s| try testing.expectEqualStrings("Hello\nWorld", s),
            else => try testing.expect(false),
        }
    }
}

test "lexer integration - character literals" {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    const source = "'A' '\\n' '\\x41' '\\''";
    var lex = try Lexer.initWithConfig(allocator, source, LexerConfig.development());
    defer lex.deinit();

    const tokens = try lex.scanTokens();

    try testing.expect(tokens.len >= 4);

    for (tokens[0..4]) |token| {
        try testing.expectEqual(TokenType.CharacterLiteral, token.type);
    }

    // Check processed character values
    if (tokens[0].value) |val| {
        switch (val) {
            .character => |c| try testing.expectEqual(@as(u8, 'A'), c),
            else => try testing.expect(false),
        }
    }

    if (tokens[1].value) |val| {
        switch (val) {
            .character => |c| try testing.expectEqual(@as(u8, '\n'), c),
            else => try testing.expect(false),
        }
    }
}

test "error recovery - string processing errors" {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    // Test string with invalid escape sequence
    const source = "\"Hello\\zWorld\" \"Valid string\" \"Another\\qBad\"";
    var lex = try Lexer.initWithConfig(allocator, source, LexerConfig.development());
    defer lex.deinit();

    const tokens = try lex.scanTokens();
    const diagnostics = lex.getDiagnostics();

    // Should have collected errors but continued scanning
    try testing.expect(diagnostics.len > 0);
    try testing.expect(tokens.len > 0); // Should have some valid tokens

    // Check that we got the expected error types
    var found_invalid_escape = false;
    for (diagnostics) |diagnostic| {
        if (diagnostic.error_type == LexerError.InvalidEscapeSequence) {
            found_invalid_escape = true;
        }
    }
    try testing.expect(found_invalid_escape);
}

test "error recovery - unterminated strings" {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    const source = "\"Unterminated string\n\"Complete string\"";
    var lex = try Lexer.initWithConfig(allocator, source, LexerConfig.development());
    defer lex.deinit();

    const tokens = try lex.scanTokens();
    const diagnostics = lex.getDiagnostics();

    // Should have error for unterminated string
    try testing.expect(diagnostics.len > 0);

    var found_unterminated = false;
    for (diagnostics) |diagnostic| {
        if (diagnostic.error_type == LexerError.UnterminatedString) {
            found_unterminated = true;
        }
    }
    try testing.expect(found_unterminated);

    // Should still have processed the complete string
    try testing.expect(tokens.len > 0);
}

test "error recovery - character literal errors" {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    const source = "'' 'AB' 'valid' '\\z'";
    var lex = try Lexer.initWithConfig(allocator, source, LexerConfig.development());
    defer lex.deinit();

    const tokens = try lex.scanTokens();
    const diagnostics = lex.getDiagnostics();

    // Should have multiple errors
    try testing.expect(diagnostics.len >= 2);

    var found_empty = false;
    var found_invalid = false;
    var found_invalid_escape = false;

    for (diagnostics) |diagnostic| {
        switch (diagnostic.error_type) {
            LexerError.EmptyCharacterLiteral => found_empty = true,
            LexerError.InvalidCharacterLiteral => found_invalid = true,
            LexerError.InvalidEscapeSequence => found_invalid_escape = true,
            else => {},
        }
    }

    try testing.expect(found_empty or found_invalid or found_invalid_escape);
}

test "string interning - duplicate strings" {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    const source = "\"hello\" \"world\" \"hello\" \"hello\" \"world\"";
    var config = LexerConfig.development();
    config.enable_string_interning = true;

    var lex = try Lexer.initWithConfig(allocator, source, config);
    defer lex.deinit();

    const tokens = try lex.scanTokens();

    // All string tokens should be present
    var string_count: u32 = 0;
    for (tokens) |token| {
        if (token.type == TokenType.StringLiteral) {
            string_count += 1;
        }
    }
    try testing.expectEqual(@as(u32, 5), string_count);

    // Check that string interning worked (same strings should have same pointer)
    var hello_ptr: ?[]const u8 = null;
    var world_ptr: ?[]const u8 = null;

    for (tokens) |token| {
        if (token.type == TokenType.StringLiteral) {
            if (std.mem.eql(u8, token.lexeme, "\"hello\"")) {
                if (hello_ptr == null) {
                    hello_ptr = token.lexeme;
                } else {
                    // Same string should have same pointer due to interning
                    try testing.expectEqual(hello_ptr.?.ptr, token.lexeme.ptr);
                }
            } else if (std.mem.eql(u8, token.lexeme, "\"world\"")) {
                if (world_ptr == null) {
                    world_ptr = token.lexeme;
                } else {
                    // Same string should have same pointer due to interning
                    try testing.expectEqual(world_ptr.?.ptr, token.lexeme.ptr);
                }
            }
        }
    }
}

test "performance - string processing overhead" {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    // Create a source with many strings containing escapes
    var source_buffer = std.ArrayList(u8).init(allocator);
    defer source_buffer.deinit();

    const writer = source_buffer.writer();
    for (0..100) |i| {
        try writer.print("\"String{}\\nWith\\tEscapes\\\\And\\\"Quotes\" ", .{i});
    }

    const source = source_buffer.items;

    var config = LexerConfig.performance();
    config.enable_performance_monitoring = true;

    var lex = try Lexer.initWithConfig(allocator, source, config);
    defer lex.deinit();

    const start_time = std.time.nanoTimestamp();
    const tokens = try lex.scanTokens();
    const end_time = std.time.nanoTimestamp();

    const duration_ns = end_time - start_time;
    const duration_ms = @as(f64, @floatFromInt(duration_ns)) / 1_000_000.0;

    // Should process 100 strings in reasonable time (less than 100ms)
    try testing.expect(duration_ms < 100.0);
    try testing.expect(tokens.len >= 100); // Should have at least 100 string tokens
}

test "edge cases - empty strings and special characters" {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    var processor = StringProcessor.init(allocator);

    // Test empty string
    {
        const result = try processor.processString("");
        defer allocator.free(result);
        try testing.expectEqualStrings("", result);
    }

    // Test string with only escapes
    {
        const result = try processor.processString("\\n\\t\\r");
        defer allocator.free(result);
        try testing.expectEqualStrings("\n\t\r", result);
    }

    // Test string with Unicode-like hex escapes (high values)
    {
        const result = try processor.processString("\\xFF\\xFE\\xFD");
        defer allocator.free(result);
        try testing.expect(result.len == 3);
        try testing.expect(result[0] == 255);
        try testing.expect(result[1] == 254);
        try testing.expect(result[2] == 253);
    }
}

test "lexer configuration - string processing features" {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    // Test with character literals disabled
    {
        const source = "'A' \"string\"";
        var config = LexerConfig.default();
        config.enable_character_literals = false;

        var lex = try Lexer.initWithConfig(allocator, source, config);
        defer lex.deinit();

        const tokens = try lex.scanTokens();
        const diagnostics = lex.getDiagnostics();

        // Should have error for character literal when disabled
        try testing.expect(diagnostics.len > 0);
    }

    // Test with raw strings disabled
    {
        const source = "r\"raw string\" \"normal string\"";
        var config = LexerConfig.default();
        config.enable_raw_strings = false;

        var lex = try Lexer.initWithConfig(allocator, source, config);
        defer lex.deinit();

        const tokens = try lex.scanTokens();

        // Should treat 'r' as identifier and process string normally
        try testing.expect(tokens.len >= 2);
        try testing.expectEqual(TokenType.Identifier, tokens[0].type);
    }
}
