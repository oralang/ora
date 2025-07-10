const std = @import("std");
const Allocator = std.mem.Allocator;

/// Lexer-specific errors for better diagnostics
pub const LexerError = error{
    UnexpectedCharacter,
    UnterminatedString,
    InvalidHexLiteral,
    UnterminatedComment,
    OutOfMemory,
};

/// Token types for ZigOra DSL
pub const TokenType = enum {
    // End of file
    Eof,

    // Keywords
    Contract,
    Pub,
    Fn,
    Let,
    Var,
    Const,
    Immutable,
    Storage,
    Memory,
    Tstore,
    Init,
    Log,
    If,
    Else,
    While,
    Break,
    Continue,
    Return,
    Requires,
    Ensures,
    Invariant,
    Old,
    Comptime,
    As,
    Import,
    Struct,
    Enum,
    True,
    False,

    // Error handling keywords
    Error,
    Try,
    Catch,

    // Identifiers and literals
    Identifier,
    StringLiteral,
    IntegerLiteral,
    HexLiteral,
    AddressLiteral,

    // Symbols and operators
    Plus, // +
    Minus, // -
    Star, // *
    Slash, // /
    Percent, // %
    Equal, // =
    EqualEqual, // ==
    BangEqual, // !=
    Less, // <
    LessEqual, // <=
    Greater, // >
    GreaterEqual, // >=
    Bang, // !
    Ampersand, // &
    Pipe, // |
    Caret, // ^
    LeftShift, // <<
    RightShift, // >>
    PlusEqual, // +=
    MinusEqual, // -=
    StarEqual, // *=
    Arrow, // ->

    // Delimiters
    LeftParen, // (
    RightParen, // )
    LeftBrace, // {
    RightBrace, // }
    LeftBracket, // [
    RightBracket, // ]
    Comma, // ,
    Semicolon, // ;
    Colon, // :
    Dot, // .
    At, // @
};

/// Token with location information
pub const Token = struct {
    type: TokenType,
    lexeme: []const u8,
    line: u32,
    column: u32,

    pub fn format(self: Token, comptime fmt: []const u8, options: std.fmt.FormatOptions, writer: anytype) !void {
        _ = fmt;
        _ = options;
        try writer.writeAll("Token{ .type = ");
        try writer.writeAll(@tagName(self.type));
        try writer.writeAll(", .lexeme = \"");
        try writer.writeAll(self.lexeme);
        try writer.writeAll("\", .line = ");
        try writer.print("{}", .{self.line});
        try writer.writeAll(", .column = ");
        try writer.print("{}", .{self.column});
        try writer.writeAll(" }");
    }
};

/// Lexer for ZigOra DSL
pub const Lexer = struct {
    source: []const u8,
    tokens: std.ArrayList(Token),
    start: u32,
    current: u32,
    line: u32,
    column: u32,
    start_column: u32, // Track start position for accurate token positioning
    last_bad_char: ?u8, // Track the character that caused an error
    allocator: Allocator,

    /// Keywords map for efficient lookup
    const keywords = std.StaticStringMap(TokenType).initComptime(.{
        .{ "contract", .Contract },
        .{ "pub", .Pub },
        .{ "fn", .Fn },
        .{ "let", .Let },
        .{ "var", .Var },
        .{ "const", .Const },
        .{ "immutable", .Immutable },
        .{ "storage", .Storage },
        .{ "memory", .Memory },
        .{ "tstore", .Tstore },
        .{ "init", .Init },
        .{ "log", .Log },
        .{ "if", .If },
        .{ "else", .Else },
        .{ "while", .While },
        .{ "break", .Break },
        .{ "continue", .Continue },
        .{ "return", .Return },
        .{ "requires", .Requires },
        .{ "ensures", .Ensures },
        .{ "invariant", .Invariant },
        .{ "old", .Old },
        .{ "comptime", .Comptime },
        .{ "as", .As },
        .{ "import", .Import },
        .{ "struct", .Struct },
        .{ "enum", .Enum },
        .{ "true", .True },
        .{ "false", .False },
        .{ "error", .Error },
        .{ "try", .Try },
        .{ "catch", .Catch },
    });

    pub fn init(allocator: Allocator, source: []const u8) Lexer {
        return Lexer{
            .source = source,
            .tokens = std.ArrayList(Token).init(allocator),
            .start = 0,
            .current = 0,
            .line = 1,
            .column = 1,
            .start_column = 1,
            .last_bad_char = null,
            .allocator = allocator,
        };
    }

    pub fn deinit(self: *Lexer) void {
        self.tokens.deinit();
    }

    /// Get details about the last error
    pub fn getErrorDetails(self: *Lexer, allocator: Allocator) ![]u8 {
        if (self.last_bad_char) |c| {
            if (std.ascii.isPrint(c)) {
                return std.fmt.allocPrint(allocator, "Unexpected character '{c}' at line {}, column {}", .{ c, self.line, self.column - 1 });
            } else {
                return std.fmt.allocPrint(allocator, "Unexpected character (ASCII {}) at line {}, column {}", .{ c, self.line, self.column - 1 });
            }
        }
        return std.fmt.allocPrint(allocator, "Error at line {}, column {}", .{ self.line, self.column });
    }

    /// Tokenize the entire source
    pub fn scanTokens(self: *Lexer) LexerError![]Token {
        // Pre-allocate some capacity for better performance
        try self.tokens.ensureTotalCapacity(32);

        while (!self.isAtEnd()) {
            self.start = self.current;
            self.start_column = self.column;
            try self.scanToken();
        }

        // Add EOF token
        try self.tokens.append(Token{
            .type = .Eof,
            .lexeme = "",
            .line = self.line,
            .column = self.column,
        });

        return self.tokens.toOwnedSlice();
    }

    fn scanToken(self: *Lexer) LexerError!void {
        const c = self.advance();

        switch (c) {
            // Whitespace
            ' ', '\r', '\t' => {
                // Skip whitespace
            },
            '\n' => {
                self.line += 1;
                self.column = 1;
            },

            // Single character tokens
            '(' => try self.addToken(.LeftParen),
            ')' => try self.addToken(.RightParen),
            '{' => try self.addToken(.LeftBrace),
            '}' => try self.addToken(.RightBrace),
            '[' => try self.addToken(.LeftBracket),
            ']' => try self.addToken(.RightBracket),
            ',' => try self.addToken(.Comma),
            ';' => try self.addToken(.Semicolon),
            ':' => try self.addToken(.Colon),
            '.' => try self.addToken(.Dot),
            '@' => try self.addToken(.At),
            '%' => try self.addToken(.Percent),
            '^' => try self.addToken(.Caret),

            // Operators that might have compound forms
            '+' => {
                if (self.match('=')) {
                    try self.addToken(.PlusEqual);
                } else {
                    try self.addToken(.Plus);
                }
            },
            '-' => {
                if (self.match('=')) {
                    try self.addToken(.MinusEqual);
                } else if (self.match('>')) {
                    try self.addToken(.Arrow);
                } else {
                    try self.addToken(.Minus);
                }
            },
            '*' => {
                if (self.match('=')) {
                    try self.addToken(.StarEqual);
                } else {
                    try self.addToken(.Star);
                }
            },
            '/' => {
                if (self.match('/')) {
                    // Single-line comment
                    while (self.peek() != '\n' and !self.isAtEnd()) {
                        _ = self.advance();
                    }
                } else if (self.match('*')) {
                    // Multi-line comment
                    try self.scanMultiLineComment();
                } else {
                    try self.addToken(.Slash);
                }
            },
            '!' => {
                if (self.match('=')) {
                    try self.addToken(.BangEqual);
                } else {
                    try self.addToken(.Bang);
                }
            },
            '=' => {
                if (self.match('=')) {
                    try self.addToken(.EqualEqual);
                } else {
                    try self.addToken(.Equal);
                }
            },
            '<' => {
                if (self.match('=')) {
                    try self.addToken(.LessEqual);
                } else if (self.match('<')) {
                    try self.addToken(.LeftShift);
                } else {
                    try self.addToken(.Less);
                }
            },
            '>' => {
                if (self.match('=')) {
                    try self.addToken(.GreaterEqual);
                } else if (self.match('>')) {
                    try self.addToken(.RightShift);
                } else {
                    try self.addToken(.Greater);
                }
            },
            '&' => try self.addToken(.Ampersand),
            '|' => try self.addToken(.Pipe),

            // String literals
            '"' => try self.scanString(),

            // Number literals (including hex and addresses)
            '0' => {
                if (self.match('x') or self.match('X')) {
                    try self.scanHexLiteral();
                } else {
                    try self.scanNumber();
                }
            },

            else => {
                if (isDigit(c)) {
                    try self.scanNumber();
                } else if (isAlpha(c)) {
                    try self.scanIdentifier();
                } else {
                    // Invalid character
                    self.last_bad_char = c;
                    return LexerError.UnexpectedCharacter;
                }
            },
        }
    }

    fn scanMultiLineComment(self: *Lexer) LexerError!void {
        var nesting: u32 = 1;

        while (nesting > 0 and !self.isAtEnd()) {
            if (self.peek() == '/' and self.peekNext() == '*') {
                _ = self.advance(); // consume '/'
                _ = self.advance(); // consume '*'
                nesting += 1;
            } else if (self.peek() == '*' and self.peekNext() == '/') {
                _ = self.advance(); // consume '*'
                _ = self.advance(); // consume '/'
                nesting -= 1;
            } else if (self.peek() == '\n') {
                self.line += 1;
                self.column = 1;
                _ = self.advance();
            } else {
                _ = self.advance();
            }
        }

        if (nesting > 0) {
            // Unclosed comment
            return LexerError.UnterminatedComment;
        }
    }

    fn scanString(self: *Lexer) LexerError!void {
        while (self.peek() != '"' and !self.isAtEnd()) {
            if (self.peek() == '\n') {
                self.line += 1;
                self.column = 1;
            }
            _ = self.advance();
        }

        if (self.isAtEnd()) {
            // Unterminated string
            return LexerError.UnterminatedString;
        }

        // Consume closing "
        _ = self.advance();
        try self.addStringToken();
    }

    fn scanHexLiteral(self: *Lexer) LexerError!void {
        var digit_count: u32 = 0;

        while (isHexDigit(self.peek())) {
            _ = self.advance();
            digit_count += 1;
        }

        if (digit_count == 0) {
            // Invalid hex literal (just "0x")
            return LexerError.InvalidHexLiteral;
        }

        // Check if it's an address (40 hex digits)
        if (digit_count == 40) {
            try self.addToken(.AddressLiteral);
        } else {
            try self.addToken(.HexLiteral);
        }
    }

    fn scanNumber(self: *Lexer) LexerError!void {
        // Scan integer part
        while (isDigit(self.peek()) or self.peek() == '_') {
            _ = self.advance();
        }

        // Check for scientific notation
        if (self.peek() == 'e' or self.peek() == 'E') {
            _ = self.advance();
            if (self.peek() == '+' or self.peek() == '-') {
                _ = self.advance();
            }
            while (isDigit(self.peek())) {
                _ = self.advance();
            }
        }

        // TODO: Future feature - type suffixes (e.g., 100u256, 5u128)
        // if (self.peek() == 'u') {
        //     _ = self.advance();
        //     while (isDigit(self.peek())) {
        //         _ = self.advance();
        //     }
        // }

        try self.addToken(.IntegerLiteral);
    }

    fn scanIdentifier(self: *Lexer) LexerError!void {
        // Validate that identifier starts with a valid character
        if (!isAlpha(self.source[self.start])) {
            self.last_bad_char = self.source[self.start];
            return LexerError.UnexpectedCharacter;
        }

        while (isAlphaNumeric(self.peek())) {
            _ = self.advance();
        }

        // Check if it's a keyword
        const text = self.source[self.start..self.current];
        const token_type = keywords.get(text) orelse .Identifier;
        try self.addToken(token_type);
    }

    fn isAtEnd(self: *Lexer) bool {
        return self.current >= self.source.len;
    }

    fn advance(self: *Lexer) u8 {
        const c = self.source[self.current];
        self.current += 1;
        self.column += 1;
        return c;
    }

    fn match(self: *Lexer, expected: u8) bool {
        if (self.isAtEnd()) return false;
        if (self.source[self.current] != expected) return false;

        self.current += 1;
        self.column += 1;
        return true;
    }

    fn peek(self: *Lexer) u8 {
        if (self.isAtEnd()) return 0;
        return self.source[self.current];
    }

    fn peekNext(self: *Lexer) u8 {
        if (self.current + 1 >= self.source.len) return 0;
        return self.source[self.current + 1];
    }

    fn addToken(self: *Lexer, token_type: TokenType) LexerError!void {
        const text = self.source[self.start..self.current];
        try self.tokens.append(Token{
            .type = token_type,
            .lexeme = text,
            .line = self.line,
            .column = self.start_column,
        });
    }

    fn addStringToken(self: *Lexer) LexerError!void {
        // Strip surrounding quotes from string literal
        const text = self.source[self.start + 1 .. self.current - 1];
        try self.tokens.append(Token{
            .type = .StringLiteral,
            .lexeme = text, // Content without quotes
            .line = self.line,
            .column = self.start_column,
        });
    }
};

/// Convenience function for testing - tokenizes source and returns tokens
pub fn scan(source: []const u8, allocator: Allocator) LexerError![]Token {
    var lexer = Lexer.init(allocator, source);
    defer lexer.deinit();
    return lexer.scanTokens();
}

// Helper functions
fn isDigit(c: u8) bool {
    return c >= '0' and c <= '9';
}

fn isHexDigit(c: u8) bool {
    return isDigit(c) or (c >= 'a' and c <= 'f') or (c >= 'A' and c <= 'F');
}

fn isAlpha(c: u8) bool {
    return (c >= 'a' and c <= 'z') or (c >= 'A' and c <= 'Z') or c == '_';
}

fn isAlphaNumeric(c: u8) bool {
    return isAlpha(c) or isDigit(c);
}

// Token utility functions for parser use
pub fn isKeyword(token_type: TokenType) bool {
    return switch (token_type) {
        .Contract, .Pub, .Fn, .Let, .Var, .Const, .Immutable, .Storage, .Memory, .Tstore, .Init, .Log, .If, .Else, .While, .Break, .Continue, .Return, .Requires, .Ensures, .Invariant, .Old, .Comptime, .As, .Import, .Struct, .Enum, .True, .False => true,
        else => false,
    };
}

pub fn isLiteral(token_type: TokenType) bool {
    return switch (token_type) {
        .StringLiteral, .IntegerLiteral, .HexLiteral, .AddressLiteral, .True, .False => true,
        else => false,
    };
}

pub fn isOperator(token_type: TokenType) bool {
    return switch (token_type) {
        .Plus, .Minus, .Star, .Slash, .Percent, .Equal, .EqualEqual, .BangEqual, .Less, .LessEqual, .Greater, .GreaterEqual, .Bang, .Ampersand, .Pipe, .Caret, .LeftShift, .RightShift, .PlusEqual, .MinusEqual, .StarEqual, .Arrow => true,
        else => false,
    };
}

pub fn isDelimiter(token_type: TokenType) bool {
    return switch (token_type) {
        .LeftParen, .RightParen, .LeftBrace, .RightBrace, .LeftBracket, .RightBracket, .Comma, .Semicolon, .Colon, .Dot, .At => true,
        else => false,
    };
}
