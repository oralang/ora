// String, raw-string, hex-bytes, and character literal scanners.

const std = @import("std");
const lexer_mod = @import("../../lexer.zig");
const LexerError = lexer_mod.LexerError;
const SourceRange = lexer_mod.SourceRange;
const TokenValue = lexer_mod.TokenValue;
const Token = lexer_mod.Token;
const Lexer = lexer_mod.Lexer;
const StringProcessor = @import("../trivia.zig").StringProcessor;

/// Scan a "..." string literal.
pub fn scanString(lexer: *Lexer) LexerError!void {
    while (lexer.peek() != '"' and !lexer.isAtEnd()) {
        if (lexer.peek() == '\n') {
            lexer.line += 1;
            lexer.column = 1;
        }
        _ = lexer.advance();
    }

    if (lexer.isAtEnd()) {
        try lexer.reportLexError(
            LexerError.UnterminatedString,
            lexer.currentRange(),
            "Unterminated string literal",
            null,
        );
        return;
    }

    _ = lexer.advance(); // consume closing "
    try addStringToken(lexer);
}

/// Scan an r"..." raw string literal (no escape processing).
pub fn scanRawString(lexer: *Lexer) LexerError!void {
    while (lexer.peek() != '"' and !lexer.isAtEnd()) {
        if (lexer.peek() == '\n') {
            lexer.line += 1;
            lexer.column = 1;
        }
        _ = lexer.advance();
    }

    if (lexer.isAtEnd()) {
        if (lexer.error_recovery != null) {
            lexer.recordError(LexerError.UnterminatedRawString, "Unterminated raw string literal");
            return;
        }
        return LexerError.UnterminatedRawString;
    }

    _ = lexer.advance(); // consume closing "
    try addRawStringToken(lexer);
}

/// Scan a hex"..." bytes literal (only hex characters allowed inside).
pub fn scanHexBytes(lexer: *Lexer) LexerError!void {
    const content_start = lexer.current;

    while (lexer.peek() != '"' and !lexer.isAtEnd()) {
        if (lexer.peek() == '\n') {
            lexer.line += 1;
            lexer.column = 1;
        }
        const c = lexer.advance();
        const is_hex = (c >= '0' and c <= '9') or (c >= 'a' and c <= 'f') or (c >= 'A' and c <= 'F');
        if (!is_hex) {
            // span just the bad char (already advanced past it)
            const bad_range = SourceRange{
                .start_line = lexer.line,
                .start_column = lexer.column - 1,
                .end_line = lexer.line,
                .end_column = lexer.column,
                .start_offset = lexer.current - 1,
                .end_offset = lexer.current,
            };
            try lexer.reportLexError(LexerError.InvalidHexLiteral, bad_range, "Invalid hex character in bytes literal", null);
            // keep scanning for more bad chars; literal-level error already recorded
        }
    }

    if (lexer.isAtEnd()) {
        if (lexer.error_recovery != null) {
            lexer.recordError(LexerError.UnterminatedString, "Unterminated hex bytes literal");
            return;
        }
        return LexerError.UnterminatedString;
    }

    _ = lexer.advance(); // consume closing "

    const hex_content = lexer.source[content_start..lexer.current];
    const token_value: ?TokenValue = if (lexer.config.store_token_values) .{ .string = hex_content } else null;
    try lexer.appendTokenWithValue(.BytesLiteral, token_value);
}

/// Scan a '.' character literal (one char or one escape sequence).
pub fn scanCharacter(lexer: *Lexer) LexerError!void {
    while (lexer.peek() != '\'' and !lexer.isAtEnd()) {
        if (lexer.peek() == '\\') {
            _ = lexer.advance();
            if (!lexer.isAtEnd()) {
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
        try lexer.reportLexError(
            LexerError.InvalidCharacterLiteral,
            lexer.currentRange(),
            "Character literal is missing closing single quote",
            null,
        );
        return;
    }

    _ = lexer.advance(); // consume closing '
    try addCharacterToken(lexer);
}

pub fn addStringToken(lexer: *Lexer) LexerError!void {
    const text = lexer.source[lexer.start + 1 .. lexer.current - 1]; // strip quotes
    const range = lexer.currentRange();

    if (!lexer.config.store_token_values) {
        StringProcessor.validateString(text) catch |err| {
            if (lexer.error_recovery != null) {
                const error_message = switch (err) {
                    LexerError.InvalidEscapeSequence => "Invalid escape sequence in string literal",
                    else => "Error processing string literal",
                };
                try lexer.reportLexError(err, range, error_message, "Use valid escape sequences: \\n, \\t, \\r, \\\\, \\\", \\', \\0, or \\xNN");
                try lexer.appendTokenWithValue(.StringLiteral, null);
                return;
            }
            return err;
        };
        try lexer.appendTokenWithValue(.StringLiteral, null);
        return;
    }

    var string_processor = StringProcessor.init(lexer.arena.allocator());
    const processed_text = string_processor.processString(text) catch |err| {
        if (lexer.error_recovery != null) {
            const error_message = switch (err) {
                LexerError.InvalidEscapeSequence => "Invalid escape sequence in string literal",
                else => "Error processing string literal",
            };
            try lexer.reportLexError(err, range, error_message, "Use valid escape sequences: \\n, \\t, \\r, \\\\, \\\", \\', \\0, or \\xNN");
            try lexer.appendTokenWithValue(.StringLiteral, .{ .string = "" });
            return;
        }
        return err;
    };

    try lexer.appendTokenWithValue(.StringLiteral, .{ .string = processed_text });
}

pub fn addRawStringToken(lexer: *Lexer) LexerError!void {
    const text = lexer.source[lexer.start + 2 .. lexer.current - 1]; // strip r" and "
    const token_value: ?TokenValue = if (lexer.config.store_token_values) .{ .string = text } else null;
    try lexer.appendTokenWithValue(.RawStringLiteral, token_value);
}

pub fn addCharacterToken(lexer: *Lexer) LexerError!void {
    const text = lexer.source[lexer.start + 1 .. lexer.current - 1]; // strip quotes
    const range = lexer.currentRange();

    const char_value = StringProcessor.processCharacterLiteral(text) catch |err| {
        if (lexer.error_recovery != null) {
            const error_message = switch (err) {
                LexerError.EmptyCharacterLiteral => "Character literal cannot be empty",
                LexerError.InvalidCharacterLiteral => "Character literal must contain exactly one character or valid escape sequence",
                LexerError.InvalidEscapeSequence => "Invalid escape sequence in character literal",
                else => "Error processing character literal",
            };
            const suggestion: ?[]const u8 = switch (err) {
                LexerError.EmptyCharacterLiteral => "Add a character between the single quotes, e.g., 'A'",
                LexerError.InvalidCharacterLiteral => "Use exactly one character or a valid escape sequence like '\\n'",
                LexerError.InvalidEscapeSequence => "Use valid escape sequences: \\n, \\t, \\r, \\\\, \\', \\\", \\0, or \\xNN",
                else => null,
            };
            try lexer.reportLexError(err, range, error_message, suggestion);
            try lexer.appendTokenWithValue(.CharacterLiteral, .{ .character = 0 });
            return;
        }
        return err;
    };

    const token_value: ?TokenValue = if (lexer.config.store_token_values) .{ .character = char_value } else null;
    try lexer.appendTokenWithValue(.CharacterLiteral, token_value);
}
