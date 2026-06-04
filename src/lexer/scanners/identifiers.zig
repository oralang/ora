// Identifier, keyword, and @ directive scanners.

const std = @import("std");
const lexer_mod = @import("../../lexer.zig");
const LexerError = lexer_mod.LexerError;
const SourceRange = lexer_mod.SourceRange;
const Token = lexer_mod.Token;
const Lexer = lexer_mod.Lexer;
const isAlpha = lexer_mod.isAlpha;
const isAlphaNumeric = lexer_mod.isAlphaNumeric;
const keywords = lexer_mod.keywords;

/// Scan an identifier or keyword.
pub fn scanIdentifier(lexer: *Lexer) LexerError!void {
    if (!isAlpha(lexer.source[lexer.start])) {
        lexer.last_bad_char = lexer.source[lexer.start];
        return LexerError.UnexpectedCharacter;
    }

    while (isAlphaNumeric(lexer.peek())) _ = lexer.advance();

    const text = lexer.source[lexer.start..lexer.current];
    const token_type = keywords.get(text) orelse .Identifier;
    try lexer.addTokenWithInterning(token_type);
}

/// Append a token with an explicit lexeme slice and range — used to split @-prefixed forms
/// into `.At` + `.Identifier`/`.Import`.
fn appendSplitToken(lexer: *Lexer, token_type: lexer_mod.TokenType, lexeme: []const u8, range: SourceRange) LexerError!void {
    try lexer.tokens.append(lexer.allocator, Token{
        .type = token_type,
        .lexeme = lexeme,
        .range = range,
        .value = null,
        .line = range.start_line,
        .column = range.start_column,
    });
}

/// Scan an @-prefixed identifier. The lexer only splits `@` from the following
/// name; parser/sema decide whether that name is a valid directive or builtin.
pub fn scanAtDirective(lexer: *Lexer) LexerError!void {
    if (!isAlpha(lexer.peek())) {
        if (lexer.error_recovery != null) {
            lexer.recordError(LexerError.UnexpectedCharacter, "Unexpected character '@'");
            return;
        }
        return LexerError.UnexpectedCharacter;
    }

    while (isAlphaNumeric(lexer.peek())) _ = lexer.advance();

    const directive_name = lexer.source[lexer.start + 1 .. lexer.current];
    if (directive_name.len == 0) {
        if (lexer.error_recovery != null) {
            lexer.recordError(LexerError.UnexpectedCharacter, "Unexpected character '@'");
            return;
        }
        return LexerError.UnexpectedCharacter;
    }

    // Emit .At + (.Import for @import, .Identifier for every other @name).
    const is_import = std.mem.eql(u8, directive_name, "import");
    const at_range = SourceRange{
        .start_line = lexer.line,
        .start_column = lexer.start_column,
        .end_line = lexer.line,
        .end_column = lexer.start_column + 1,
        .start_offset = lexer.start,
        .end_offset = lexer.start + 1,
    };
    try appendSplitToken(lexer, .At, lexer.source[lexer.start .. lexer.start + 1], at_range);

    const tail_range = SourceRange{
        .start_line = lexer.line,
        .start_column = lexer.start_column + 1,
        .end_line = lexer.line,
        .end_column = lexer.column,
        .start_offset = lexer.start + 1,
        .end_offset = lexer.current,
    };
    const tail_type: lexer_mod.TokenType = if (is_import) .Import else .Identifier;
    try appendSplitToken(lexer, tail_type, lexer.source[lexer.start + 1 .. lexer.current], tail_range);
}
