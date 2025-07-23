const std = @import("std");
const lexer = @import("src/lexer.zig");
const testing = std.testing;

test "error collection infrastructure - basic functionality" {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    // Test ErrorRecovery struct
    var error_recovery = lexer.ErrorRecovery.init(allocator, 10);
    defer error_recovery.deinit();

    // Test recording errors
    const range = lexer.SourceRange{
        .start_line = 1,
        .start_column = 1,
        .end_line = 1,
        .end_column = 2,
        .start_offset = 0,
        .end_offset = 1,
    };

    try error_recovery.recordError(lexer.LexerError.UnexpectedCharacter, range, "Test error message");
    try testing.expect(error_recovery.getErrorCount() == 1);

    const errors = error_recovery.getErrors();
    try testing.expect(errors.len == 1);
    try testing.expect(errors[0].error_type == lexer.LexerError.UnexpectedCharacter);
    try testing.expect(std.mem.eql(u8, errors[0].message, "Test error message"));
    try testing.expect(errors[0].severity == .Error);
}

test "error collection infrastructure - with suggestions" {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    var error_recovery = lexer.ErrorRecovery.init(allocator, 10);
    defer error_recovery.deinit();

    const range = lexer.SourceRange{
        .start_line = 1,
        .start_column = 1,
        .end_line = 1,
        .end_column = 2,
        .start_offset = 0,
        .end_offset = 1,
    };

    try error_recovery.recordErrorWithSuggestion(lexer.LexerError.InvalidBinaryLiteral, range, "Invalid binary literal", "Use 0b followed by binary digits");

    const errors = error_recovery.getErrors();
    try testing.expect(errors.len == 1);
    try testing.expect(errors[0].suggestion != null);
    try testing.expect(std.mem.eql(u8, errors[0].suggestion.?, "Use 0b followed by binary digits"));
}

test "error collection infrastructure - max errors limit" {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    var error_recovery = lexer.ErrorRecovery.init(allocator, 2); // Limit to 2 errors
    defer error_recovery.deinit();

    const range = lexer.SourceRange{
        .start_line = 1,
        .start_column = 1,
        .end_line = 1,
        .end_column = 2,
        .start_offset = 0,
        .end_offset = 1,
    };

    // Record first error - should succeed
    try error_recovery.recordError(lexer.LexerError.UnexpectedCharacter, range, "Error 1");
    try testing.expect(error_recovery.getErrorCount() == 1);

    // Record second error - should succeed
    try error_recovery.recordError(lexer.LexerError.UnexpectedCharacter, range, "Error 2");
    try testing.expect(error_recovery.getErrorCount() == 2);

    // Record third error - should fail with TooManyErrors
    const result = error_recovery.recordError(lexer.LexerError.UnexpectedCharacter, range, "Error 3");
    try testing.expectError(lexer.LexerError.TooManyErrors, result);
    try testing.expect(error_recovery.getErrorCount() == 2); // Should still be 2
}

test "lexer with error recovery - invalid characters" {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    const source = "let x = 123; @ invalid $ chars #";
    const config = lexer.LexerConfig{
        .enable_error_recovery = true,
        .max_errors = 10,
        .enable_suggestions = true,
    };

    var lex = lexer.Lexer.initWithConfig(allocator, source, config);
    defer lex.deinit();

    // Should not fail even with invalid characters
    const tokens = try lex.scanTokens();
    defer allocator.free(tokens);

    // Should have collected some errors
    const diagnostics = lex.getDiagnostics();
    try testing.expect(diagnostics.len > 0);

    // Should still have valid tokens
    var valid_token_count: usize = 0;
    for (tokens) |token| {
        if (token.type != .Eof) {
            valid_token_count += 1;
        }
    }
    try testing.expect(valid_token_count > 0);
}

test "lexer without error recovery - fails on first error" {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    const source = "let x = 123; $ invalid";

    var lex = lexer.Lexer.init(allocator, source); // Default config has error recovery disabled
    lex.config.enable_error_recovery = false;
    defer lex.deinit();

    // Should fail on the first invalid character
    const result = lex.scanTokens();
    try testing.expectError(lexer.LexerError.UnexpectedCharacter, result);
}

test "diagnostic severity levels" {
    const range = lexer.SourceRange{
        .start_line = 1,
        .start_column = 1,
        .end_line = 1,
        .end_column = 2,
        .start_offset = 0,
        .end_offset = 1,
    };

    // Test different severity levels
    const error_diag = lexer.LexerDiagnostic.init(lexer.LexerError.UnexpectedCharacter, range, "Error message");
    try testing.expect(error_diag.severity == .Error);

    const warning_diag = error_diag.withSeverity(.Warning);
    try testing.expect(warning_diag.severity == .Warning);

    const info_diag = error_diag.withSeverity(.Info);
    try testing.expect(info_diag.severity == .Info);

    const hint_diag = error_diag.withSeverity(.Hint);
    try testing.expect(hint_diag.severity == .Hint);
}
