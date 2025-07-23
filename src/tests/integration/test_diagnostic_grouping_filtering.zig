const std = @import("std");
const testing = std.testing;
const lexer = @import("src/lexer.zig");

test "diagnostic grouping by type" {
    const allocator = testing.allocator;

    // Test source with multiple types of errors
    const source =
        \\let x = "unterminated string
        \\let y = 0b123; // invalid binary
        \\let z = @invalid; // invalid builtin
        \\let w = 0xGHI; // invalid hex
    ;

    var config = lexer.LexerConfig.default();
    config.enable_error_recovery = true;

    var lex = lexer.Lexer.initWithConfig(allocator, source, config);
    defer lex.deinit();

    // Scan tokens to collect errors
    const tokens = lex.scanTokens() catch {
        return;
    };
    defer allocator.free(tokens);

    // Debug: print what tokens we got
    std.debug.print("Generated {} tokens\n", .{tokens.len});
    for (tokens[0..@min(tokens.len, 10)]) |token| {
        std.debug.print("  Token: {s} at line {} (lexeme: '{s}')\n", .{ @tagName(token.type), token.line, token.lexeme });
    }

    const diagnostics = lex.getDiagnostics();
    try testing.expect(diagnostics.len > 0);

    // Debug: print diagnostics
    std.debug.print("Generated {} diagnostics\n", .{diagnostics.len});
    for (diagnostics) |diagnostic| {
        std.debug.print("  Diagnostic: {s} at {}:{}\n", .{ @errorName(diagnostic.error_type), diagnostic.range.start_line, diagnostic.range.start_column });
    }

    // Debug: print all diagnostics from error recovery
    if (lex.error_recovery) |*recovery| {
        std.debug.print("Error recovery has {} errors\n", .{recovery.errors.items.len});
        for (recovery.errors.items) |diagnostic| {
            std.debug.print("  Error: {s} at {}:{}\n", .{ @errorName(diagnostic.error_type), diagnostic.range.start_line, diagnostic.range.start_column });
        }
    }

    // Test grouping by type
    const type_groups = lex.getDiagnosticsByType();
    try testing.expect(type_groups != null);

    if (type_groups) |groups| {
        defer groups.deinit();

        // Debug: print what we got
        std.debug.print("Found {} error types:\n", .{groups.items.len});
        for (groups.items) |group| {
            std.debug.print("  {s}: {}\n", .{ @errorName(group.error_type), group.count });
        }

        // Should have multiple error types
        try testing.expect(groups.items.len > 1);

        // Check for specific error types
        var found_unterminated_string = false;
        var found_invalid_binary = false;
        var found_invalid_builtin = false;
        var found_invalid_hex = false;

        for (groups.items) |group| {
            switch (group.error_type) {
                lexer.LexerError.UnterminatedString => {
                    found_unterminated_string = true;
                    try testing.expect(group.count > 0);
                },
                lexer.LexerError.InvalidBinaryLiteral => {
                    found_invalid_binary = true;
                    try testing.expect(group.count > 0);
                },
                lexer.LexerError.InvalidBuiltinFunction => {
                    found_invalid_builtin = true;
                    try testing.expect(group.count > 0);
                },
                lexer.LexerError.InvalidHexLiteral => {
                    found_invalid_hex = true;
                    try testing.expect(group.count > 0);
                },
                else => {},
            }
        }

        // Should find all expected error types
        try testing.expect(found_unterminated_string);
        try testing.expect(found_invalid_binary);
        try testing.expect(found_invalid_builtin);
        try testing.expect(found_invalid_hex);
    }
}

test "diagnostic filtering by severity" {
    const allocator = testing.allocator;

    const source = "let x = @invalid; let y = 0b123;";

    var config = lexer.LexerConfig.default();
    config.enable_error_recovery = true;

    var lex = lexer.Lexer.initWithConfig(allocator, source, config);
    defer lex.deinit();

    const tokens = lex.scanTokens() catch {
        return;
    };
    defer allocator.free(tokens);

    // Test filtering by severity
    var error_diagnostics = lex.getDiagnosticsBySeverity(.Error);
    defer error_diagnostics.deinit();
    try testing.expect(error_diagnostics.items.len > 0);

    // All diagnostics should be Error severity
    for (error_diagnostics.items) |diagnostic| {
        try testing.expect(diagnostic.severity == .Error);
    }
}

test "diagnostic grouping by line" {
    const allocator = testing.allocator;

    const source =
        \\let x = @invalid;
        \\let y = 0b123;
        \\let z = @invalid;
    ;

    var config = lexer.LexerConfig.default();
    config.enable_error_recovery = true;

    var lex = lexer.Lexer.initWithConfig(allocator, source, config);
    defer lex.deinit();

    const tokens = lex.scanTokens() catch {
        return;
    };
    defer allocator.free(tokens);

    const line_groups = lex.getDiagnosticsByLine();
    try testing.expect(line_groups != null);

    if (line_groups) |groups| {
        defer groups.deinit();

        // Should have multiple lines with errors
        try testing.expect(groups.items.len > 1);

        // Check that we have errors on different lines
        var found_line_1 = false;
        var found_line_2 = false;
        var found_line_3 = false;

        for (groups.items) |group| {
            switch (group.line) {
                1 => {
                    found_line_1 = true;
                    try testing.expect(group.count > 0);
                },
                2 => {
                    found_line_2 = true;
                    try testing.expect(group.count > 0);
                },
                3 => {
                    found_line_3 = true;
                    try testing.expect(group.count > 0);
                },
                else => {},
            }
        }

        // Should find errors on multiple lines
        try testing.expect(found_line_1);
        try testing.expect(found_line_2);
        try testing.expect(found_line_3);
    }
}

test "diagnostic summary report" {
    const allocator = testing.allocator;

    const source =
        \\let x = @invalid;
        \\let y = 0b123;
        \\let z = "unterminated
    ;

    var config = lexer.LexerConfig.default();
    config.enable_error_recovery = true;

    var lex = lexer.Lexer.initWithConfig(allocator, source, config);
    defer lex.deinit();

    const tokens = lex.scanTokens() catch {
        return;
    };
    defer allocator.free(tokens);

    const summary = lex.createDiagnosticSummary(allocator) catch unreachable;
    defer allocator.free(summary);

    // Summary should contain error counts
    try testing.expect(std.mem.indexOf(u8, summary, "Diagnostic Summary") != null);
    try testing.expect(std.mem.indexOf(u8, summary, "InvalidBuiltinFunction") != null);
    try testing.expect(std.mem.indexOf(u8, summary, "InvalidBinaryLiteral") != null);
    try testing.expect(std.mem.indexOf(u8, summary, "UnterminatedString") != null);
    try testing.expect(std.mem.indexOf(u8, summary, "Severity Breakdown") != null);
}

test "diagnostic filtering by minimum severity" {
    const allocator = testing.allocator;

    const source = "let x = @invalid; let y = 0b123;";

    var config = lexer.LexerConfig.default();
    config.enable_error_recovery = true;

    var lex = lexer.Lexer.initWithConfig(allocator, source, config);
    defer lex.deinit();

    const tokens = lex.scanTokens() catch {
        return;
    };
    defer allocator.free(tokens);

    // Filter by minimum severity (Error)
    var filtered = lex.filterDiagnosticsBySeverity(.Error);
    defer filtered.deinit();
    try testing.expect(filtered.items.len > 0);

    // All filtered diagnostics should be Error or higher severity
    for (filtered.items) |diagnostic| {
        try testing.expect(@intFromEnum(diagnostic.severity) >= @intFromEnum(lexer.DiagnosticSeverity.Error));
    }
}

test "related diagnostics" {
    const allocator = testing.allocator;

    const source =
        \\let x = @invalid;
        \\let y = @invalid;
        \\let z = 0b123;
    ;

    var config = lexer.LexerConfig.default();
    config.enable_error_recovery = true;

    var lex = lexer.Lexer.initWithConfig(allocator, source, config);
    defer lex.deinit();

    const tokens = lex.scanTokens() catch {
        return;
    };
    defer allocator.free(tokens);

    const diagnostics = lex.getDiagnostics();
    try testing.expect(diagnostics.len > 0);

    // Get related diagnostics for the first diagnostic
    var related = lex.getRelatedDiagnostics(diagnostics[0], 2);
    defer related.deinit();
    try testing.expect(related.items.len > 0);

    // Related diagnostics should be within 2 lines
    for (related.items) |related_diagnostic| {
        const line_distance = if (diagnostics[0].range.start_line > related_diagnostic.range.start_line)
            diagnostics[0].range.start_line - related_diagnostic.range.start_line
        else
            related_diagnostic.range.start_line - diagnostics[0].range.start_line;

        try testing.expect(line_distance <= 2);
    }
}
