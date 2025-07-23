const std = @import("std");
const lexer = @import("src/lexer.zig");
const testing = std.testing;

test "findNextTokenBoundary - whitespace boundaries" {
    const source = "let x @ = 123;";
    const boundary = lexer.ErrorRecovery.findNextTokenBoundary(source, 6); // Start at '@'
    try testing.expect(boundary == 7); // Should stop at space after '@'
}

test "findNextTokenBoundary - token start boundaries" {
    const source = "let x@=123;";
    const boundary = lexer.ErrorRecovery.findNextTokenBoundary(source, 5); // Start at '@'
    try testing.expect(boundary == 6); // Should stop at '=' (token start)
}

test "findNextTokenBoundary - identifier boundaries" {
    const source = "let x@abc";
    const boundary = lexer.ErrorRecovery.findNextTokenBoundary(source, 5); // Start at '@'
    try testing.expect(boundary == 6); // Should stop at 'a' (identifier start)
}

test "findNextTokenBoundary - number boundaries" {
    const source = "let x@123";
    const boundary = lexer.ErrorRecovery.findNextTokenBoundary(source, 5); // Start at '@'
    try testing.expect(boundary == 6); // Should stop at '1' (number start)
}

test "findNextTokenBoundary - string boundaries" {
    const source = "let x@\"hello\"";
    const boundary = lexer.ErrorRecovery.findNextTokenBoundary(source, 5); // Start at '@'
    try testing.expect(boundary == 6); // Should stop at '"' (string start)
}

test "findNextTokenBoundary - end of source" {
    const source = "let x@";
    const boundary = lexer.ErrorRecovery.findNextTokenBoundary(source, 5); // Start at '@'
    try testing.expect(boundary == 6); // Should return end of source
}

test "suggestFix - unexpected character suggestions" {
    // Test dollar sign suggestion
    const dollar_suggestion = lexer.ErrorRecovery.suggestFix(lexer.LexerError.UnexpectedCharacter, "$");
    try testing.expect(dollar_suggestion != null);
    try testing.expect(std.mem.indexOf(u8, dollar_suggestion.?, "$") != null);

    // Test at sign suggestion
    const at_suggestion = lexer.ErrorRecovery.suggestFix(lexer.LexerError.UnexpectedCharacter, "@");
    try testing.expect(at_suggestion != null);
    try testing.expect(std.mem.indexOf(u8, at_suggestion.?, "@") != null);

    // Test hash suggestion
    const hash_suggestion = lexer.ErrorRecovery.suggestFix(lexer.LexerError.UnexpectedCharacter, "#");
    try testing.expect(hash_suggestion != null);
    try testing.expect(std.mem.indexOf(u8, hash_suggestion.?, "//") != null);

    // Test backtick suggestion
    const backtick_suggestion = lexer.ErrorRecovery.suggestFix(lexer.LexerError.UnexpectedCharacter, "`");
    try testing.expect(backtick_suggestion != null);
    try testing.expect(std.mem.indexOf(u8, backtick_suggestion.?, "double quotes") != null);
}

test "suggestFix - string error suggestions" {
    const unterminated_suggestion = lexer.ErrorRecovery.suggestFix(lexer.LexerError.UnterminatedString, "");
    try testing.expect(unterminated_suggestion != null);
    try testing.expect(std.mem.indexOf(u8, unterminated_suggestion.?, "closing quote") != null);

    const raw_unterminated_suggestion = lexer.ErrorRecovery.suggestFix(lexer.LexerError.UnterminatedRawString, "");
    try testing.expect(raw_unterminated_suggestion != null);
    try testing.expect(std.mem.indexOf(u8, raw_unterminated_suggestion.?, "closing quote") != null);
}

test "suggestFix - number error suggestions" {
    const binary_suggestion = lexer.ErrorRecovery.suggestFix(lexer.LexerError.InvalidBinaryLiteral, "");
    try testing.expect(binary_suggestion != null);
    try testing.expect(std.mem.indexOf(u8, binary_suggestion.?, "0b") != null);

    const hex_suggestion = lexer.ErrorRecovery.suggestFix(lexer.LexerError.InvalidHexLiteral, "");
    try testing.expect(hex_suggestion != null);
    try testing.expect(std.mem.indexOf(u8, hex_suggestion.?, "0x") != null);

    const address_suggestion = lexer.ErrorRecovery.suggestFix(lexer.LexerError.InvalidAddressFormat, "");
    try testing.expect(address_suggestion != null);
    try testing.expect(std.mem.indexOf(u8, address_suggestion.?, "40") != null);
}

test "suggestFix - character error suggestions" {
    const empty_char_suggestion = lexer.ErrorRecovery.suggestFix(lexer.LexerError.EmptyCharacterLiteral, "");
    try testing.expect(empty_char_suggestion != null);
    try testing.expect(std.mem.indexOf(u8, empty_char_suggestion.?, "character") != null);

    const invalid_char_suggestion = lexer.ErrorRecovery.suggestFix(lexer.LexerError.InvalidCharacterLiteral, "");
    try testing.expect(invalid_char_suggestion != null);
    try testing.expect(std.mem.indexOf(u8, invalid_char_suggestion.?, "exactly one") != null);
}

test "lexer error recovery - multiple invalid characters" {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    const source = "let x = 123; $ @ # invalid chars";
    const config = lexer.LexerConfig{
        .enable_error_recovery = true,
        .max_errors = 10,
        .enable_suggestions = true,
    };

    var lex = lexer.Lexer.initWithConfig(allocator, source, config);
    defer lex.deinit();

    const tokens = try lex.scanTokens();
    defer allocator.free(tokens);

    // Should have collected multiple errors (at least 2, since error recovery might group some)
    const diagnostics = lex.getDiagnostics();
    try testing.expect(diagnostics.len >= 2); // At least 2 invalid characters

    // Should still have valid tokens before the errors
    var valid_tokens: usize = 0;
    for (tokens) |token| {
        if (token.type != .Eof) {
            valid_tokens += 1;
        }
    }
    try testing.expect(valid_tokens >= 4); // let, x, =, 123
}

test "lexer error recovery - with suggestions" {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    const source = "let x = 123; # comment should be //";
    const config = lexer.LexerConfig{
        .enable_error_recovery = true,
        .max_errors = 10,
        .enable_suggestions = true,
    };

    var lex = lexer.Lexer.initWithConfig(allocator, source, config);
    defer lex.deinit();

    const tokens = try lex.scanTokens();
    defer allocator.free(tokens);

    const diagnostics = lex.getDiagnostics();
    try testing.expect(diagnostics.len >= 1);

    // Check that we got a suggestion for the hash character
    var found_hash_suggestion = false;
    for (diagnostics) |diagnostic| {
        if (diagnostic.error_type == lexer.LexerError.UnexpectedCharacter and diagnostic.suggestion != null) {
            if (std.mem.indexOf(u8, diagnostic.suggestion.?, "//") != null) {
                found_hash_suggestion = true;
                break;
            }
        }
    }
    try testing.expect(found_hash_suggestion);
}

test "lexer error recovery - boundary detection" {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    const source = "let x = @@@@@; let y = 456;";
    const config = lexer.LexerConfig{
        .enable_error_recovery = true,
        .max_errors = 10,
        .enable_suggestions = true,
    };

    var lex = lexer.Lexer.initWithConfig(allocator, source, config);
    defer lex.deinit();

    const tokens = try lex.scanTokens();
    defer allocator.free(tokens);

    // Should recover and continue parsing after the invalid characters
    var found_y = false;
    var found_456 = false;
    for (tokens) |token| {
        if (std.mem.eql(u8, token.lexeme, "y")) {
            found_y = true;
        }
        if (std.mem.eql(u8, token.lexeme, "456")) {
            found_456 = true;
        }
    }
    try testing.expect(found_y);
    try testing.expect(found_456);
}
