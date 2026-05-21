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

/// Scan an @-directive: @import, @lock, @unlock, or @builtin. All forms emit `.At` + a trailing token.
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

    const is_import = std.mem.eql(u8, directive_name, "import");
    const is_lock_or_unlock = std.mem.eql(u8, directive_name, "lock") or std.mem.eql(u8, directive_name, "unlock");

    if (!is_import and !is_lock_or_unlock and !isValidBuiltin(directive_name)) {
        if (lexer.error_recovery) |*recovery| {
            const message = std.fmt.allocPrint(lexer.allocator, "Invalid directive or built-in function '@{s}'", .{directive_name}) catch "Invalid directive or built-in function";
            defer lexer.allocator.free(message);
            recovery.recordDetailedError(LexerError.InvalidBuiltinFunction, lexer.currentRange(), lexer.source, message) catch return;
        } else {
            return LexerError.InvalidBuiltinFunction;
        }
    }

    // emit .At + (.Import for @import, .Identifier for @lock/@unlock/@builtin)
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

fn isValidBuiltin(name: []const u8) bool {
    const valid = [_][]const u8{
        "divTrunc",        "divFloor",          "divCeil",         "divExact",        "divMod",          "divmod",
        "cast",            "truncate",          "addWithOverflow", "subWithOverflow", "mulWithOverflow", "divWithOverflow",
        "modWithOverflow", "powerWithOverflow", "negWithOverflow", "shlWithOverflow", "shrWithOverflow", "bitCast",
        "bits",            "concat",            "slice",           "compileError",    "selector",        "abiSignature",
        "eventTopic",      "eip712TypeHash",    "chainId",         "structFields",    "traitMethods",
    };
    for (valid) |v| {
        if (std.mem.eql(u8, name, v)) return true;
    }
    return false;
}
