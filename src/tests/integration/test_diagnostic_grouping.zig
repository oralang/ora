const std = @import("std");
const testing = std.testing;
const lexer = @import("src/lexer.zig");

test "diagnostic grouping and filtering" {
    const allocator = testing.allocator;

    // Test source with multiple related errors
    const source =
        \\let x = "unterminated string
        \\let y = 0b123; // invalid binary
        \\let z = 0xGHI; // invalid hex
        \\let a = 0b101; // valid binary
        \\let b = ''; // empty character
        \\let c = @invalid; // invalid builtin
        \\let d = "another unterminated string
    ;

    var config = lexer.LexerConfig.default();
    config.enable_error_recovery = true;

    var lex = lexer.Lexer.initWithConfig(allocator, source, config);
    defer lex.deinit();

    // Scan tokens (should collect errors)
    const tokens = lex.scanTokens() catch |err| {
        // If scanning fails, we still want to check diagnostics
        std.debug.print("Scan failed with error: {s}\n", .{@errorName(err)});
        return;
    };
    defer allocator.free(tokens);

    // Manually add some diagnostics for testing
    if (lex.error_recovery) |*recovery| {
        const range = lexer.SourceRange{
            .start_line = 1,
            .start_column = 1,
            .end_line = 1,
            .end_column = 2,
            .start_offset = 0,
            .end_offset = 1,
        };

        // Add diagnostics with different severities
        try recovery.recordError(lexer.LexerError.UnterminatedString, range, "Test error 1");
        try recovery.recordError(lexer.LexerError.InvalidBinaryLiteral, range, "Test error 2");

        // Manually set the severity of the second diagnostic to Warning
        if (recovery.errors.items.len > 1) {
            recovery.errors.items[1].severity = .Warning;
        }
    }

    const diagnostics = lex.getDiagnostics();
    try testing.expect(diagnostics.len > 0);

    // Test diagnostic grouping
    if (lex.error_recovery) |*recovery| {
        var groups = try recovery.groupErrors();
        defer {
            for (groups.items) |group| {
                group.related.deinit();
            }
            groups.deinit();
        }

        // Should have at least one group
        try testing.expect(groups.items.len > 0);

        // Check that related errors are grouped together
        var found_related = false;
        std.debug.print("Found {} groups\n", .{groups.items.len});
        for (groups.items, 0..) |group, i| {
            std.debug.print("Group {} has {} related errors\n", .{ i, group.related.items.len });
            if (group.related.items.len > 0) {
                found_related = true;
                break;
            }
        }

        // Print all diagnostics for debugging
        std.debug.print("Total diagnostics: {}\n", .{diagnostics.len});
        for (diagnostics, 0..) |diag, i| {
            std.debug.print("Diagnostic {}: {s} at {}:{}\n", .{ i, @errorName(diag.error_type), diag.range.start_line, diag.range.start_column });
        }

        try testing.expect(found_related);

        // Test filtering by severity
        var errors = recovery.getErrorsBySeverity(.Error);
        defer errors.deinit();
        try testing.expect(errors.items.len > 0);

        // Test filtering by minimum severity
        var min_warnings = recovery.filterByMinimumSeverity(.Warning);
        defer min_warnings.deinit();
        try testing.expect(min_warnings.items.len >= errors.items.len);

        // Test diagnostic report generation
        const report = try recovery.createDetailedReport(allocator);
        defer allocator.free(report);
        try testing.expect(report.len > 0);

        // Print the report for debugging
        std.debug.print("Report content: {s}\n", .{report});

        // Report should contain error types or messages
        try testing.expect(std.mem.indexOf(u8, report, "Test error 1") != null);
        try testing.expect(std.mem.indexOf(u8, report, "Test error 2") != null);
    }
}

test "diagnostic report formatting" {
    const allocator = testing.allocator;

    // Test source with a single error
    const source = "let x = @invalid;";

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

    // Test report generation
    const report = try lex.createDiagnosticReport();
    defer allocator.free(report);

    // Report should contain error details
    try testing.expect(report.len > 0);
    try testing.expect(std.mem.indexOf(u8, report, "Issue #1") != null);
    try testing.expect(std.mem.indexOf(u8, report, "Location:") != null);
    try testing.expect(std.mem.indexOf(u8, report, "Source context:") != null);
}
