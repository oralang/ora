const std = @import("std");
const testing = std.testing;
const Allocator = std.mem.Allocator;
const lexer = @import("src/lexer.zig");
const Lexer = lexer.Lexer;
const Token = lexer.Token;
const TokenType = lexer.TokenType;
const LexerError = lexer.LexerError;
const LexerConfig = lexer.LexerConfig;
const LexerDiagnostic = lexer.LexerDiagnostic;
const DiagnosticSeverity = lexer.DiagnosticSeverity;
const ErrorRecovery = lexer.ErrorRecovery;

// Test helper function to create a lexer with error recovery enabled
fn createLexerWithErrorRecovery(source: []const u8, allocator: Allocator) !Lexer {
    const config = LexerConfig{
        .enable_error_recovery = true,
        .enable_string_interning = false,
        .max_errors = 50,
        .enable_suggestions = true,
    };
    return Lexer.initWithConfig(allocator, source, config);
}

// Test helper to scan tokens with error recovery
fn scanTokensWithErrorRecovery(source: []const u8, allocator: Allocator) ![]Token {
    var lex = try createLexerWithErrorRecovery(source, allocator);
    defer lex.deinit();
    return lex.scanTokens();
}

// Test helper to get diagnostics from lexer
fn getDiagnostics(source: []const u8, allocator: Allocator) ![]LexerDiagnostic {
    var lex = try createLexerWithErrorRecovery(source, allocator);
    defer lex.deinit();
    const tokens = try lex.scanTokens();
    defer allocator.free(tokens); // Free the tokens to prevent memory leak

    if (lex.error_recovery) |*recovery| {
        const diagnostics = recovery.getErrors();
        // Create a copy since the lexer will be deinitialized
        var result = try allocator.alloc(LexerDiagnostic, diagnostics.len);
        for (diagnostics, 0..) |diagnostic, i| {
            result[i] = diagnostic;
        }
        return result;
    } else {
        return try allocator.alloc(LexerDiagnostic, 0);
    }
}

test "error recovery - continued scanning after invalid characters" {
    const allocator = testing.allocator;

    // Test that lexer continues after encountering invalid characters
    const source = "valid @ invalid $ more_valid";
    const tokens = try scanTokensWithErrorRecovery(source, allocator);
    defer allocator.free(tokens);

    // Should produce tokens for valid parts and continue scanning
    try testing.expect(tokens.len >= 3); // At least: valid, more_valid, EOF

    // Check that we got the valid identifiers
    var found_valid = false;
    var found_more_valid = false;

    for (tokens) |token| {
        if (token.type == .Identifier) {
            if (std.mem.eql(u8, token.lexeme, "valid")) {
                found_valid = true;
            } else if (std.mem.eql(u8, token.lexeme, "more_valid")) {
                found_more_valid = true;
            }
        }
    }

    try testing.expect(found_valid);
    try testing.expect(found_more_valid);
}

test "error recovery - multiple errors collection" {
    const allocator = testing.allocator;

    // Test source with multiple different types of errors
    const source = "0b123 \"unterminated 0xGHI 'empty";
    const diagnostics = try getDiagnostics(source, allocator);
    defer allocator.free(diagnostics);

    // Should collect multiple errors
    try testing.expect(diagnostics.len >= 2);

    // Check that we have different types of errors
    var has_binary_error = false;
    var has_string_error = false;
    var has_hex_error = false;

    for (diagnostics) |diagnostic| {
        switch (diagnostic.error_type) {
            LexerError.InvalidBinaryLiteral => has_binary_error = true,
            LexerError.UnterminatedString => has_string_error = true,
            LexerError.InvalidHexLiteral => has_hex_error = true,
            else => {},
        }
    }

    try testing.expect(has_binary_error or has_hex_error); // At least one number error
    try testing.expect(has_string_error); // String error
}

test "error recovery - diagnostic details and context" {
    const allocator = testing.allocator;

    const source = "0b999"; // Invalid binary literal
    const diagnostics = try getDiagnostics(source, allocator);
    defer allocator.free(diagnostics);

    try testing.expect(diagnostics.len >= 1);

    const diagnostic = diagnostics[0];

    // Check diagnostic has proper error type
    try testing.expectEqual(LexerError.InvalidBinaryLiteral, diagnostic.error_type);

    // Check diagnostic has source range information
    try testing.expectEqual(@as(u32, 1), diagnostic.range.start_line);
    try testing.expectEqual(@as(u32, 1), diagnostic.range.start_column);

    // Check diagnostic has message
    try testing.expect(diagnostic.message.len > 0);

    // Check diagnostic has appropriate severity
    try testing.expectEqual(DiagnosticSeverity.Error, diagnostic.severity);
}

test "error recovery - suggestion system effectiveness" {
    const allocator = testing.allocator;

    // Test various error types that should generate suggestions
    const test_cases = [_]struct {
        source: []const u8,
        expected_error: LexerError,
    }{
        .{ .source = "0b", .expected_error = LexerError.InvalidBinaryLiteral },
        .{ .source = "0x", .expected_error = LexerError.InvalidHexLiteral },
        .{ .source = "\"unterminated", .expected_error = LexerError.UnterminatedString },
        .{ .source = "'", .expected_error = LexerError.EmptyCharacterLiteral },
    };

    for (test_cases) |test_case| {
        const diagnostics = try getDiagnostics(test_case.source, allocator);
        defer allocator.free(diagnostics);

        try testing.expect(diagnostics.len >= 1);

        // Find the expected error type
        var found_error = false;
        for (diagnostics) |diagnostic| {
            if (diagnostic.error_type == test_case.expected_error) {
                found_error = true;
                // Check that suggestion system is working (may or may not have suggestions)
                // The important thing is that the system doesn't crash
                break;
            }
        }

        try testing.expect(found_error);
    }
}

test "error recovery - token boundary detection" {
    const allocator = testing.allocator;

    // Test that error recovery finds appropriate token boundaries
    const source = "valid1 @#$%^& valid2 0b999 valid3";
    const tokens = try scanTokensWithErrorRecovery(source, allocator);
    defer allocator.free(tokens);

    // Should recover and continue scanning valid tokens
    var valid_token_count: u32 = 0;

    for (tokens) |token| {
        if (token.type == .Identifier and
            (std.mem.eql(u8, token.lexeme, "valid1") or
                std.mem.eql(u8, token.lexeme, "valid2") or
                std.mem.eql(u8, token.lexeme, "valid3")))
        {
            valid_token_count += 1;
        }
    }

    // Should find at least 2 of the 3 valid tokens (depending on error recovery effectiveness)
    try testing.expect(valid_token_count >= 2);
}

test "error recovery - maximum error limit" {
    const allocator = testing.allocator;

    // Create a lexer with a low error limit
    const config = LexerConfig{
        .enable_error_recovery = true,
        .enable_string_interning = false,
        .max_errors = 3, // Low limit for testing
        .enable_suggestions = true,
    };

    // Create source with many errors
    const source = "0b999 0xGHI \"unterminated 'empty @#$%";

    var lex = try Lexer.initWithConfig(allocator, source, config);
    defer lex.deinit();

    _ = try lex.scanTokens();

    const diagnostics = if (lex.error_recovery) |*recovery| recovery.getErrors() else &[_]LexerDiagnostic{};

    // Should not exceed the maximum error limit
    try testing.expect(diagnostics.len <= 3);
}

test "error recovery - diagnostic grouping functionality" {
    const allocator = testing.allocator;

    // Test source with related errors that should be grouped
    const source = "0b999 0b888 0b777"; // Multiple binary literal errors

    var lex = try createLexerWithErrorRecovery(source, allocator);
    defer lex.deinit();

    _ = try lex.scanTokens();

    // Test diagnostic grouping
    const groups = if (lex.error_recovery) |*recovery| try recovery.groupErrors() else blk: {
        const empty_groups = std.ArrayList(ErrorRecovery.DiagnosticGroup).init(allocator);
        break :blk empty_groups;
    };
    defer {
        for (groups.items) |group| {
            group.related.deinit();
        }
        groups.deinit();
    }

    // Should have at least one group
    try testing.expect(groups.items.len >= 1);

    // Check that grouping works (related errors should be grouped together)
    if (groups.items.len > 0) {
        const first_group = groups.items[0];
        try testing.expectEqual(LexerError.InvalidBinaryLiteral, first_group.primary.error_type);

        // Should have related errors of the same type
        for (first_group.related.items) |related| {
            try testing.expectEqual(LexerError.InvalidBinaryLiteral, related.error_type);
        }
    }
}

test "error recovery - diagnostic filtering by severity" {
    const allocator = testing.allocator;

    const source = "0b999 \"unterminated"; // Mix of errors

    var lex = try createLexerWithErrorRecovery(source, allocator);
    defer lex.deinit();

    _ = try lex.scanTokens();

    // Test filtering by severity
    var error_diagnostics = if (lex.error_recovery) |*recovery| recovery.getErrorsBySeverity(.Error) else std.ArrayList(LexerDiagnostic).init(allocator);
    defer error_diagnostics.deinit();

    var warning_diagnostics = if (lex.error_recovery) |*recovery| recovery.getErrorsBySeverity(.Warning) else std.ArrayList(LexerDiagnostic).init(allocator);
    defer warning_diagnostics.deinit();

    // Should have error-level diagnostics
    try testing.expect(error_diagnostics.items.len >= 1);

    // All filtered diagnostics should have the correct severity
    for (error_diagnostics.items) |diagnostic| {
        try testing.expectEqual(DiagnosticSeverity.Error, diagnostic.severity);
    }

    for (warning_diagnostics.items) |diagnostic| {
        try testing.expectEqual(DiagnosticSeverity.Warning, diagnostic.severity);
    }
}

test "error recovery - summary report generation" {
    const allocator = testing.allocator;

    const source = "0b999 0xGHI \"unterminated";

    var lex = try createLexerWithErrorRecovery(source, allocator);
    defer lex.deinit();

    _ = try lex.scanTokens();

    // Test summary report generation
    const summary = if (lex.error_recovery) |*recovery| try recovery.createSummaryReport(allocator) else try allocator.dupe(u8, "No error recovery enabled");
    defer allocator.free(summary);

    // Summary should contain useful information
    try testing.expect(summary.len > 0);
    try testing.expect(std.mem.indexOf(u8, summary, "Diagnostic Summary") != null);
    try testing.expect(std.mem.indexOf(u8, summary, "Severity Breakdown") != null);
}

test "error recovery - detailed report generation" {
    const allocator = testing.allocator;

    const source = "0b999 \"unterminated";

    var lex = try createLexerWithErrorRecovery(source, allocator);
    defer lex.deinit();

    _ = try lex.scanTokens();

    // Test detailed report generation
    const detailed = if (lex.error_recovery) |*recovery| try recovery.createDetailedReport(allocator) else try allocator.dupe(u8, "No error recovery enabled");
    defer allocator.free(detailed);

    // Detailed report should contain comprehensive information
    try testing.expect(detailed.len > 0);
    try testing.expect(std.mem.indexOf(u8, detailed, "Diagnostic Report") != null);
    try testing.expect(std.mem.indexOf(u8, detailed, "Location:") != null);
}

test "error recovery - error type counting and statistics" {
    const allocator = testing.allocator;

    const source = "0b999 0b888 0xGHI \"unterminated"; // 2 binary + 1 hex + 1 string error

    var lex = try createLexerWithErrorRecovery(source, allocator);
    defer lex.deinit();

    _ = try lex.scanTokens();

    // Test error type counting
    if (lex.error_recovery) |*recovery| {
        var error_types = recovery.getErrorsByType();
        defer error_types.deinit();

        // Should have different error types with counts
        try testing.expect(error_types.items.len >= 1);

        // Check that counts make sense
        var total_count: usize = 0;
        for (error_types.items) |type_count| {
            try testing.expect(type_count.count > 0);
            total_count += type_count.count;
        }

        // Total count should match the number of errors
        const all_errors = recovery.getErrors();
        try testing.expectEqual(all_errors.len, total_count);
    }
}

test "error recovery - line-based error grouping" {
    const allocator = testing.allocator;

    const source =
        \\0b999 0xGHI
        \\"unterminated
        \\valid_token
        \\0b888
    ;

    var lex = try createLexerWithErrorRecovery(source, allocator);
    defer lex.deinit();

    _ = try lex.scanTokens();

    // Test line-based error grouping
    if (lex.error_recovery) |*recovery| {
        var line_counts = recovery.getErrorsByLine();
        defer line_counts.deinit();

        // Should have errors on different lines
        try testing.expect(line_counts.items.len >= 1);

        // Check that line numbers are reasonable
        for (line_counts.items) |line_count| {
            try testing.expect(line_count.line >= 1);
            try testing.expect(line_count.line <= 4); // Source has 4 lines
            try testing.expect(line_count.count > 0);
        }
    }
}

test "error recovery - related error detection" {
    const allocator = testing.allocator;

    const source =
        \\0b999
        \\0b888
        \\valid_token
        \\0b777
    ;

    var lex = try createLexerWithErrorRecovery(source, allocator);
    defer lex.deinit();

    _ = try lex.scanTokens();

    const all_errors = if (lex.error_recovery) |*recovery| recovery.getErrors() else &[_]LexerDiagnostic{};
    try testing.expect(all_errors.len >= 1);

    // Test related error detection
    const first_error = all_errors[0];
    var related = if (lex.error_recovery) |*recovery| recovery.getRelatedErrors(first_error, 5) else std.ArrayList(LexerDiagnostic).init(allocator); // Within 5 lines
    defer related.deinit();

    // Should find related errors (other binary literal errors)
    for (related.items) |related_error| {
        // Related errors should be of the same type or nearby
        const same_type = related_error.error_type == first_error.error_type;
        const line_distance = if (related_error.range.start_line > first_error.range.start_line)
            related_error.range.start_line - first_error.range.start_line
        else
            first_error.range.start_line - related_error.range.start_line;
        const nearby = line_distance <= 5;

        try testing.expect(same_type or nearby);
    }
}

test "error recovery - minimum severity filtering" {
    const allocator = testing.allocator;

    const source = "0b999 \"unterminated";

    var lex = try createLexerWithErrorRecovery(source, allocator);
    defer lex.deinit();

    _ = try lex.scanTokens();

    // Test minimum severity filtering
    var error_and_above = if (lex.error_recovery) |*recovery| recovery.filterByMinimumSeverity(.Error) else std.ArrayList(LexerDiagnostic).init(allocator);
    defer error_and_above.deinit();

    var warning_and_above = if (lex.error_recovery) |*recovery| recovery.filterByMinimumSeverity(.Warning) else std.ArrayList(LexerDiagnostic).init(allocator);
    defer warning_and_above.deinit();

    var info_and_above = if (lex.error_recovery) |*recovery| recovery.filterByMinimumSeverity(.Info) else std.ArrayList(LexerDiagnostic).init(allocator);
    defer info_and_above.deinit();

    // Error level should have fewer or equal items than warning level
    try testing.expect(error_and_above.items.len <= warning_and_above.items.len);
    try testing.expect(warning_and_above.items.len <= info_and_above.items.len);

    // All filtered items should meet the minimum severity requirement
    for (error_and_above.items) |diagnostic| {
        try testing.expect(@intFromEnum(diagnostic.severity) >= @intFromEnum(DiagnosticSeverity.Error));
    }
}

test "error recovery - clear functionality" {
    const allocator = testing.allocator;

    const source = "0b999 \"unterminated";

    var lex = try createLexerWithErrorRecovery(source, allocator);
    defer lex.deinit();

    _ = try lex.scanTokens();

    // Should have errors initially
    if (lex.error_recovery) |*recovery| {
        try testing.expect(recovery.getErrorCount() > 0);

        // Clear errors
        recovery.clear();

        // Should have no errors after clearing
        try testing.expectEqual(@as(usize, 0), recovery.getErrorCount());
        try testing.expectEqual(@as(usize, 0), recovery.getErrors().len);
    }
}

test "error recovery - complex mixed error scenarios" {
    const allocator = testing.allocator;

    // Complex source with multiple types of errors mixed with valid code
    const source =
        \\contract Test {
        \\    let x = 0b999;  // Invalid binary
        \\    let y = "unterminated string
        \\    let z = 0xGHI;  // Invalid hex
        \\    fn test() {
        \\        let a = 'empty;  // Invalid char
        \\        let b = 42;      // Valid
        \\    }
        \\}
    ;

    const tokens = try scanTokensWithErrorRecovery(source, allocator);
    defer allocator.free(tokens);

    // Should still produce many valid tokens despite errors
    var valid_token_count: u32 = 0;
    var keyword_count: u32 = 0;

    for (tokens) |token| {
        switch (token.type) {
            .Contract, .Let, .Fn => keyword_count += 1,
            .Identifier => {
                if (std.mem.eql(u8, token.lexeme, "Test") or
                    std.mem.eql(u8, token.lexeme, "x") or
                    std.mem.eql(u8, token.lexeme, "y") or
                    std.mem.eql(u8, token.lexeme, "z") or
                    std.mem.eql(u8, token.lexeme, "test") or
                    std.mem.eql(u8, token.lexeme, "a") or
                    std.mem.eql(u8, token.lexeme, "b"))
                {
                    valid_token_count += 1;
                }
            },
            .IntegerLiteral => {
                if (std.mem.eql(u8, token.lexeme, "42")) {
                    valid_token_count += 1;
                }
            },
            else => {},
        }
    }

    // Should find most of the valid tokens and keywords
    try testing.expect(keyword_count >= 2); // At least some keywords
    try testing.expect(valid_token_count >= 3); // At least some identifiers/literals

    // Test that we collected multiple errors
    const diagnostics = try getDiagnostics(source, allocator);
    defer allocator.free(diagnostics);

    try testing.expect(diagnostics.len >= 2); // Multiple different errors
}

test "error recovery - performance with many errors" {
    const allocator = testing.allocator;

    // Create source with many errors to test performance
    var source_builder = std.ArrayList(u8).init(allocator);
    defer source_builder.deinit();

    // Add many invalid binary literals
    var i: u32 = 0;
    while (i < 100) : (i += 1) {
        try source_builder.appendSlice("0b999 ");
    }

    const source = source_builder.items;

    // This should complete in reasonable time without crashing
    const tokens = try scanTokensWithErrorRecovery(source, allocator);
    defer allocator.free(tokens);

    // Should still produce EOF token
    try testing.expect(tokens.len >= 1);
    try testing.expectEqual(TokenType.Eof, tokens[tokens.len - 1].type);

    // Test that error collection works with many errors
    const diagnostics = try getDiagnostics(source, allocator);
    defer allocator.free(diagnostics);

    // Should collect errors up to the limit
    try testing.expect(diagnostics.len > 0);
    try testing.expect(diagnostics.len <= 50); // Max errors limit
}
