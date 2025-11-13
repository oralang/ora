// ============================================================================
// String Literal Scanners
// ============================================================================
//
// Handles scanning of string literals, raw strings, and character literals.
//
// ============================================================================

const std = @import("std");
const LexerError = @import("../../lexer.zig").LexerError;
const SourceRange = @import("../../lexer.zig").SourceRange;
const TokenValue = @import("../../lexer.zig").TokenValue;
const Token = @import("../../lexer.zig").Token;
const TokenType = @import("../../lexer.zig").TokenType;
const StringProcessor = @import("../trivia.zig").StringProcessor;

// Forward declaration - Lexer is defined in lexer.zig
const Lexer = @import("../../lexer.zig").Lexer;

/// Scan a string literal
pub fn scanString(lexer: *Lexer) LexerError!void {
    while (lexer.peek() != '"' and !lexer.isAtEnd()) {
        if (lexer.peek() == '\n') {
            lexer.line += 1;
            lexer.column = 1;
        }
        _ = lexer.advance();
    }

    if (lexer.isAtEnd()) {
        // Unterminated string - record detailed error with accurate span
        const start_offset = lexer.start;
        const start_line = lexer.line;
        const start_column = lexer.start_column;
        if (lexer.hasErrorRecovery()) {
            const range = SourceRange{
                .start_line = start_line,
                .start_column = start_column,
                .end_line = lexer.line,
                .end_column = lexer.column,
                .start_offset = start_offset,
                .end_offset = lexer.current,
            };
            lexer.error_recovery.?.recordDetailedError(LexerError.UnterminatedString, range, lexer.source, "Unterminated string literal") catch {};
            return;
        } else {
            return LexerError.UnterminatedString;
        }
    }

    // Consume closing "
    _ = lexer.advance();
    try addStringToken(lexer);
}

/// Scan a raw string literal
pub fn scanRawString(lexer: *Lexer) LexerError!void {
    // Raw strings don't process escape sequences, so we scan until we find the closing "
    while (lexer.peek() != '"' and !lexer.isAtEnd()) {
        if (lexer.peek() == '\n') {
            lexer.line += 1;
            lexer.column = 1;
        }
        _ = lexer.advance();
    }

    if (lexer.isAtEnd()) {
        // Unterminated raw string - use error recovery if enabled
        if (lexer.hasErrorRecovery()) {
            lexer.recordError(LexerError.UnterminatedRawString, "Unterminated raw string literal");
            return; // Skip adding the token
        } else {
            return LexerError.UnterminatedRawString;
        }
    }

    // Consume closing "
    _ = lexer.advance();
    try addRawStringToken(lexer);
}

/// Scan a character literal
pub fn scanCharacter(lexer: *Lexer) LexerError!void {
    const start_line = lexer.line;
    const start_column = lexer.column;

    // Enhanced character literal scanning with proper escape sequence handling
    while (lexer.peek() != '\'' and !lexer.isAtEnd()) {
        if (lexer.peek() == '\\') {
            // Handle escape sequence - advance past the backslash
            _ = lexer.advance();
            if (!lexer.isAtEnd()) {
                // Advance past the escaped character
                if (lexer.peek() == '\n') {
                    lexer.line += 1;
                    lexer.column = 1;
                }
                _ = lexer.advance();
            }
        } else {
            if (lexer.peek() == '\n') {
                lexer.line += 1;
                lexer.column = 1;
            }
            _ = lexer.advance();
        }
    }

    if (lexer.isAtEnd()) {
        // Create proper error range for unterminated character literal
        const range = SourceRange{
            .start_line = start_line,
            .start_column = start_column,
            .end_line = lexer.line,
            .end_column = lexer.column,
            .start_offset = lexer.start,
            .end_offset = lexer.current,
        };

        // Use error recovery if enabled
        if (lexer.config.enable_error_recovery and lexer.error_recovery != null) {
            try lexer.error_recovery.?.recordDetailedError(LexerError.InvalidCharacterLiteral, range, lexer.source, "Character literal is missing closing single quote");
            return;
        } else {
            return LexerError.InvalidCharacterLiteral;
        }
    }

    // Consume closing '
    _ = lexer.advance();
    try addCharacterToken(lexer);
}

/// Add a string literal token
pub fn addStringToken(lexer: *Lexer) LexerError!void {
    // Strip surrounding quotes from string literal
    const text = lexer.source[lexer.start + 1 .. lexer.current - 1];
    const range = SourceRange{
        .start_line = lexer.line,
        .start_column = lexer.start_column,
        .end_line = lexer.line,
        .end_column = lexer.column,
        .start_offset = lexer.start,
        .end_offset = lexer.current,
    };

    // Process escape sequences in the string content using arena allocator
    var string_processor = StringProcessor.init(lexer.arena.allocator());
    const processed_text = string_processor.processString(text) catch |err| {
        // Use error recovery if enabled for better error reporting
        if (lexer.config.enable_error_recovery and lexer.error_recovery != null) {
            const error_message = switch (err) {
                LexerError.InvalidEscapeSequence => "Invalid escape sequence in string literal",
                else => "Error processing string literal",
            };

            const suggestion = "Use valid escape sequences: \\n, \\t, \\r, \\\\, \\\", \\', \\0, or \\xNN";

            try lexer.error_recovery.?.recordDetailedErrorWithSuggestion(err, range, lexer.source, error_message, suggestion);

            // Return a default empty string for error recovery
            const token_value = TokenValue{ .string = "" };
            try lexer.tokens.append(lexer.allocator, Token{
                .type = .StringLiteral,
                .lexeme = "",
                .range = range,
                .value = token_value,
                .line = lexer.line,
                .column = lexer.start_column,
            });
            return;
        } else {
            return err;
        }
    };

    const token_value = TokenValue{ .string = processed_text };

    // Track performance metrics if enabled
    if (lexer.performance) |*perf| {
        perf.tokens_scanned += 1;
    }

    try lexer.tokens.append(lexer.allocator, Token{
        .type = .StringLiteral,
        .lexeme = processed_text, // Content with escape sequences processed
        .range = range,
        .value = token_value,
        // Legacy fields for backward compatibility
        .line = lexer.line,
        .column = lexer.start_column,
    });
}

/// Add a raw string literal token
pub fn addRawStringToken(lexer: *Lexer) LexerError!void {
    // Strip surrounding r" and " from raw string literal
    // The lexeme includes the 'r' prefix, so we need to skip r" at start and " at end
    const text = lexer.source[lexer.start + 2 .. lexer.current - 1];
    const range = SourceRange{
        .start_line = lexer.line,
        .start_column = lexer.start_column,
        .end_line = lexer.line,
        .end_column = lexer.column,
        .start_offset = lexer.start,
        .end_offset = lexer.current,
    };

    // Raw strings don't process escape sequences, store content as-is
    const token_value = TokenValue{ .string = text };

    try lexer.tokens.append(lexer.allocator, Token{
        .type = .RawStringLiteral,
        .lexeme = text, // Content without r" and "
        .range = range,
        .value = token_value,
        // Legacy fields for backward compatibility
        .line = lexer.line,
        .column = lexer.start_column,
    });
}

/// Add a character literal token
pub fn addCharacterToken(lexer: *Lexer) LexerError!void {
    // Strip surrounding single quotes from character literal
    const text = lexer.source[lexer.start + 1 .. lexer.current - 1];
    const range = SourceRange{
        .start_line = lexer.line,
        .start_column = lexer.start_column,
        .end_line = lexer.line,
        .end_column = lexer.column,
        .start_offset = lexer.start,
        .end_offset = lexer.current,
    };

    // Process the character literal using StringProcessor with enhanced error handling
    const char_value = StringProcessor.processCharacterLiteral(text) catch |err| {
        // Use error recovery if enabled for better error reporting
        if (lexer.config.enable_error_recovery and lexer.error_recovery != null) {
            const error_message = switch (err) {
                LexerError.EmptyCharacterLiteral => "Character literal cannot be empty",
                LexerError.InvalidCharacterLiteral => "Character literal must contain exactly one character or valid escape sequence",
                LexerError.InvalidEscapeSequence => "Invalid escape sequence in character literal",
                else => "Error processing character literal",
            };

            const suggestion = switch (err) {
                LexerError.EmptyCharacterLiteral => "Add a character between the single quotes, e.g., 'A'",
                LexerError.InvalidCharacterLiteral => "Use exactly one character or a valid escape sequence like '\\n'",
                LexerError.InvalidEscapeSequence => "Use valid escape sequences: \\n, \\t, \\r, \\\\, \\', \\\", \\0, or \\xNN",
                else => null,
            };

            if (suggestion) |s| {
                try lexer.error_recovery.?.recordDetailedErrorWithSuggestion(err, range, lexer.source, error_message, s);
            } else {
                try lexer.error_recovery.?.recordDetailedError(err, range, lexer.source, error_message);
            }

            // Return a default character value for error recovery
            const token_value = TokenValue{ .character = 0 };
            try lexer.tokens.append(lexer.allocator, Token{
                .type = .CharacterLiteral,
                .lexeme = text,
                .range = range,
                .value = token_value,
                .line = lexer.line,
                .column = lexer.start_column,
            });
            return;
        } else {
            return err;
        }
    };

    const token_value = TokenValue{ .character = char_value };

    try lexer.tokens.append(lexer.allocator, Token{
        .type = .CharacterLiteral,
        .lexeme = text, // Content without quotes
        .range = range,
        .value = token_value,
        // Legacy fields for backward compatibility
        .line = lexer.line,
        .column = lexer.start_column,
    });
}
