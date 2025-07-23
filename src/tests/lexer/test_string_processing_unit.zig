const std = @import("std");
const testing = std.testing;
const Allocator = std.mem.Allocator;
const lexer = @import("src/lexer.zig");

const StringProcessor = lexer.StringProcessor;
const LexerError = lexer.LexerError;

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

    // Test backslash at end of string (should be treated as literal backslash)
    {
        const result = try processor.processString("Hello\\");
        defer allocator.free(result);
        try testing.expectEqualStrings("Hello\\", result);
    }
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
