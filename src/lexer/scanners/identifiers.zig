// ============================================================================
// Identifier and Directive Scanners
// ============================================================================
//
// Handles scanning of identifiers, keywords, and @ directives.
//
// ============================================================================

const std = @import("std");
const LexerError = @import("../../lexer.zig").LexerError;
const SourceRange = @import("../../lexer.zig").SourceRange;
const Token = @import("../../lexer.zig").Token;
const TokenType = @import("../../lexer.zig").TokenType;
const isAlpha = @import("../../lexer.zig").isAlpha;
const isAlphaNumeric = @import("../../lexer.zig").isAlphaNumeric;

// Forward declaration - Lexer is defined in lexer.zig
const Lexer = @import("../../lexer.zig").Lexer;

// Import keywords map from lexer.zig
// Note: We'll need to access it through the lexer module
const keywords = @import("../../lexer.zig").keywords;

/// Scan an identifier or keyword
pub fn scanIdentifier(lexer: *Lexer) LexerError!void {
    // Validate that identifier starts with a valid character
    if (!isAlpha(lexer.source[lexer.start])) {
        lexer.last_bad_char = lexer.source[lexer.start];
        return LexerError.UnexpectedCharacter;
    }

    while (isAlphaNumeric(lexer.peek())) {
        _ = lexer.advance();
    }

    // Check if it's a keyword
    const text = lexer.source[lexer.start..lexer.current];
    const token_type = keywords.get(text) orelse .Identifier;

    // Use string interning for identifiers and keywords
    try lexer.addTokenWithInterning(token_type);
}

/// Scan an @ directive (built-in functions or import)
pub fn scanAtDirective(lexer: *Lexer) LexerError!void {
    // Check if the next character is a letter (start of identifier)
    if (!isAlpha(lexer.peek())) {
        // '@' followed by non-letter - this is an unexpected character
        if (lexer.hasErrorRecovery()) {
            const message = "Unexpected character '@'";
            lexer.recordError(LexerError.UnexpectedCharacter, message);
            return;
        } else {
            return LexerError.UnexpectedCharacter;
        }
    }

    // Scan the identifier after '@'
    while (isAlphaNumeric(lexer.peek())) {
        _ = lexer.advance();
    }

    // Get the directive/function name (without the '@')
    const directive_name = lexer.source[lexer.start + 1 .. lexer.current];

    // Check if we have an empty identifier (shouldn't happen due to isAlpha check above)
    if (directive_name.len == 0) {
        if (lexer.hasErrorRecovery()) {
            const message = "Unexpected character '@'";
            lexer.recordError(LexerError.UnexpectedCharacter, message);
            return;
        } else {
            return LexerError.UnexpectedCharacter;
        }
    }

    // Check if it's an import directive
    if (std.mem.eql(u8, directive_name, "import")) {
        // For @import, produce separate tokens: @ and import
        // First, add the @ token
        const at_text = lexer.source[lexer.start .. lexer.start + 1];
        const at_range = SourceRange{
            .start_line = lexer.line,
            .start_column = lexer.start_column,
            .end_line = lexer.line,
            .end_column = lexer.start_column + 1,
            .start_offset = lexer.start,
            .end_offset = lexer.start + 1,
        };
        try lexer.tokens.append(lexer.allocator, Token{
            .type = .At,
            .lexeme = at_text,
            .range = at_range,
            .value = null,
            .line = lexer.line,
            .column = lexer.start_column,
        });

        // Then, add the import token
        const import_text = lexer.source[lexer.start + 1 .. lexer.current];
        const import_range = SourceRange{
            .start_line = lexer.line,
            .start_column = lexer.start_column + 1,
            .end_line = lexer.line,
            .end_column = lexer.column,
            .start_offset = lexer.start + 1,
            .end_offset = lexer.current,
        };
        try lexer.tokens.append(lexer.allocator, Token{
            .type = .Import,
            .lexeme = import_text,
            .range = import_range,
            .value = null,
            .line = lexer.line,
            .column = lexer.start_column + 1,
        });
        return;
    }

    // Check if it's a valid built-in function
    const is_valid_builtin = std.mem.eql(u8, directive_name, "divTrunc") or
        std.mem.eql(u8, directive_name, "divFloor") or
        std.mem.eql(u8, directive_name, "divCeil") or
        std.mem.eql(u8, directive_name, "divExact") or
        std.mem.eql(u8, directive_name, "divMod");

    if (!is_valid_builtin) {
        // Invalid directive/function - record error and continue
        if (lexer.hasErrorRecovery()) {
            const message = std.fmt.allocPrint(lexer.allocator, "Invalid directive or built-in function '@{s}'", .{directive_name}) catch "Invalid directive or built-in function";
            defer lexer.allocator.free(message);

            const range = SourceRange{
                .start_line = lexer.line,
                .start_column = lexer.start_column,
                .end_line = lexer.line,
                .end_column = lexer.column,
                .start_offset = lexer.start,
                .end_offset = lexer.current,
            };

            if (lexer.error_recovery) |*recovery| {
                recovery.recordDetailedError(LexerError.InvalidBuiltinFunction, range, lexer.source, message) catch {
                    // If we can't record more errors, we've hit the limit
                    return;
                };
            }
        } else {
            return LexerError.InvalidBuiltinFunction;
        }
    }

    // For built-in functions, add the @ token (same as before)
    try lexer.addToken(.At);
}
