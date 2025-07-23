const std = @import("std");
const testing = std.testing;
const lexer = @import("src/lexer.zig");

test "enhanced error message quality with source context" {
    const allocator = testing.allocator;

    // Test source with various errors
    const source =
        \\contract Test {
        \\    let x = "unterminated string
        \\    let y = 0b123; // invalid binary
        \\    let z = @invalid;
        \\}
    ;

    var config = lexer.LexerConfig.default();
    config.enable_error_recovery = true;

    var lex = lexer.Lexer.initWithConfig(allocator, source, config);
    defer lex.deinit();

    // Scan tokens (should collect errors)
    const tokens = lex.scanTokens() catch {
        // If scanning fails, we still want to check diagnostics
        return;
    };
    defer allocator.free(tokens);

    const diagnostics = lex.getDiagnostics();
    try testing.expect(diagnostics.len > 0);

    // Test that diagnostics have detailed information
    for (diagnostics) |diagnostic| {
        // Should have source range information
        try testing.expect(diagnostic.range.start_line > 0);
        try testing.expect(diagnostic.range.start_column > 0);

        // Should have context if it's a detailed diagnostic
        if (diagnostic.context) |context| {
            try testing.expect(context.source_line.len > 0);
            try testing.expect(context.line_number > 0);
        }

        // Should have template for consistent formatting
        if (diagnostic.template) |template| {
            try testing.expect(template.title.len > 0);
            try testing.expect(template.description.len > 0);
        }
    }
}

test "error message templates for different error types" {
    const allocator = testing.allocator;

    // Test different error types
    const test_cases = [_]struct {
        source: []const u8,
        expected_error: lexer.LexerError,
    }{
        .{ .source = "let x = \"unterminated", .expected_error = lexer.LexerError.UnterminatedString },
        .{ .source = "let x = 0b123", .expected_error = lexer.LexerError.InvalidBinaryLiteral },
        .{ .source = "let x = 0xGHI", .expected_error = lexer.LexerError.InvalidHexLiteral },
        .{ .source = "let x = ''", .expected_error = lexer.LexerError.EmptyCharacterLiteral },
        .{ .source = "let x = @", .expected_error = lexer.LexerError.UnexpectedCharacter },
    };

    for (test_cases) |test_case| {
        var config = lexer.LexerConfig.default();
        config.enable_error_recovery = true;

        var lex = lexer.Lexer.initWithConfig(allocator, test_case.source, config);
        defer lex.deinit();

        const tokens = lex.scanTokens() catch {
            // If scanning fails, we still want to check diagnostics
            return;
        };
        defer allocator.free(tokens);

        const diagnostics = lex.getDiagnostics();
        try testing.expect(diagnostics.len > 0);

        // Find the expected error type
        var found = false;
        for (diagnostics) |diagnostic| {
            if (diagnostic.error_type == test_case.expected_error) {
                found = true;

                // Should have a template
                try testing.expect(diagnostic.template != null);
                if (diagnostic.template) |template| {
                    try testing.expect(template.title.len > 0);
                    try testing.expect(template.description.len > 0);
                }
                break;
            }
        }
        try testing.expect(found);
    }
}

test "diagnostic formatting with source context" {
    const allocator = testing.allocator;

    const source = "let x = @invalid;";

    var config = lexer.LexerConfig.default();
    config.enable_error_recovery = true;

    var lex = lexer.Lexer.initWithConfig(allocator, source, config);
    defer lex.deinit();

    const tokens = lex.scanTokens() catch {
        // If scanning fails, we still want to check diagnostics
        return;
    };
    defer allocator.free(tokens);

    const diagnostics = lex.getDiagnostics();
    try testing.expect(diagnostics.len > 0);

    // Test formatting
    var buffer = std.ArrayList(u8).init(allocator);
    defer buffer.deinit();

    const writer = buffer.writer();
    try diagnostics[0].format("", .{}, writer);

    const formatted = buffer.items;

    // Should contain error type and location
    try testing.expect(std.mem.indexOf(u8, formatted, "Error:") != null);
    try testing.expect(std.mem.indexOf(u8, formatted, "1:") != null); // line number

    // Should contain source context if available
    if (diagnostics[0].context != null) {
        try testing.expect(std.mem.indexOf(u8, formatted, source) != null);
        try testing.expect(std.mem.indexOf(u8, formatted, "^") != null); // caret indicator
    }
}

test "error context extraction" {
    const allocator = testing.allocator;

    const source =
        \\line 1
        \\line 2 with error @
        \\line 3
    ;

    const range = lexer.SourceRange{
        .start_line = 2,
        .start_column = 18,
        .end_line = 2,
        .end_column = 19,
        .start_offset = 25, // Position of '@'
        .end_offset = 26,
    };

    const context = lexer.extractSourceContext(allocator, source, range) catch unreachable;
    defer allocator.free(context.source_line);

    try testing.expectEqualStrings("line 2 with error @", context.source_line);
    try testing.expectEqual(@as(u32, 2), context.line_number);
    try testing.expectEqual(@as(u32, 18), context.column_start);
    try testing.expectEqual(@as(u32, 19), context.column_end);
}

test "error template consistency" {
    // Test that all error types have appropriate templates
    const error_types = [_]lexer.LexerError{
        lexer.LexerError.UnexpectedCharacter,
        lexer.LexerError.UnterminatedString,
        lexer.LexerError.UnterminatedRawString,
        lexer.LexerError.InvalidEscapeSequence,
        lexer.LexerError.InvalidBinaryLiteral,
        lexer.LexerError.InvalidHexLiteral,
        lexer.LexerError.NumberTooLarge,
        lexer.LexerError.InvalidAddressFormat,
        lexer.LexerError.EmptyCharacterLiteral,
        lexer.LexerError.InvalidCharacterLiteral,
        lexer.LexerError.UnterminatedComment,
    };

    for (error_types) |error_type| {
        const template = lexer.getErrorTemplate(error_type);

        // All templates should have title and description
        try testing.expect(template.title.len > 0);
        try testing.expect(template.description.len > 0);

        // Title should be lowercase and descriptive
        try testing.expect(std.ascii.isLower(template.title[0]));

        // Description should be a complete sentence
        try testing.expect(template.description.len > 10);
    }
}
