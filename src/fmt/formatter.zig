const std = @import("std");
const lib = @import("ora_lib");
const Writer = @import("writer.zig").Writer;

pub const FormatError = error{
    ParseError,
    OutOfMemory,
    WriteError,
    UnsupportedNode,
    UnsupportedType,
};

pub const FormatOptions = struct {
    line_width: u32 = 100,
    indent_size: u32 = 4,
};

pub const Formatter = struct {
    allocator: std.mem.Allocator,
    options: FormatOptions,
    writer: Writer,
    source: []const u8,
    tokens: []const lib.Token = &.{},
    trivia: []const lib.lexer.TriviaPiece = &.{},
    paren_depth: usize = 0,
    bracket_depth: usize = 0,
    pending_space: bool = false,
    last_token_type: ?lib.TokenType = null,

    pub fn init(allocator: std.mem.Allocator, source: []const u8, options: FormatOptions) Formatter {
        return .{
            .allocator = allocator,
            .options = options,
            .writer = Writer.init(allocator, options.indent_size, options.line_width),
            .source = source,
        };
    }

    pub fn deinit(self: *Formatter) void {
        self.writer.deinit();
    }

    pub fn format(self: *Formatter) FormatError![]u8 {
        var parse_result = lib.compiler.syntax.parse(self.allocator, lib.compiler.FileId.fromIndex(0), self.source) catch {
            return FormatError.ParseError;
        };
        defer parse_result.deinit();
        if (!parse_result.diagnostics.isEmpty()) {
            return FormatError.ParseError;
        }

        var lower_result = lib.compiler.ast.lower(self.allocator, &parse_result.tree) catch {
            return FormatError.ParseError;
        };
        defer lower_result.deinit();
        if (!lower_result.diagnostics.isEmpty()) {
            return FormatError.ParseError;
        }

        var lex = lib.Lexer.init(self.allocator, self.source);
        defer lex.deinit();

        self.tokens = lex.scanTokens() catch {
            return FormatError.ParseError;
        };
        defer self.allocator.free(self.tokens);
        self.trivia = lex.getTrivia();

        for (self.tokens, 0..) |token, index| {
            if (token.type == .Eof) break;
            const next = self.peekNonEof(index + 1);
            try self.emitLeadingComments(token);
            try self.formatToken(token, next);
        }

        if (self.writer.current_line_length > 0) {
            try self.writer.newline();
        }
        return self.writer.toOwnedSlice() catch FormatError.OutOfMemory;
    }

    fn peekNonEof(self: *const Formatter, start: usize) ?lib.Token {
        var i = start;
        while (i < self.tokens.len) : (i += 1) {
            const token = self.tokens[i];
            if (token.type != .Eof) return token;
        }
        return null;
    }

    fn formatToken(self: *Formatter, token: lib.Token, next: ?lib.Token) FormatError!void {
        switch (token.type) {
            .LeftBrace => try self.formatLeftBrace(token),
            .RightBrace => try self.formatRightBrace(token, next),
            .Semicolon => try self.formatSemicolon(),
            .Comma => try self.formatComma(next),
            .Colon => try self.formatColon(next),
            .Arrow => try self.formatArrow(),
            .LeftParen => try self.formatLeftParen(token),
            .RightParen => try self.formatRightParen(),
            .LeftBracket => try self.formatLeftBracket(token),
            .RightBracket => try self.formatRightBracket(),
            .StringLiteral, .RawStringLiteral, .CharacterLiteral => {
                try self.emitPreTokenSpacing(token);
                try self.writer.write(self.originalSpan(token));
                try self.emitPostTokenSpacing(token, next);
            },
            else => {
                try self.emitPreTokenSpacing(token);
                try self.writer.write(token.lexeme);
                try self.emitPostTokenSpacing(token, next);
            },
        }
        self.last_token_type = token.type;
    }

    fn emitLeadingComments(self: *Formatter, token: lib.Token) FormatError!void {
        var i = token.leading_trivia_start;
        const end = token.leading_trivia_start + token.leading_trivia_len;
        while (i < end and i < self.trivia.len) : (i += 1) {
            const piece = self.trivia[i];
            switch (piece.kind) {
                .LineComment, .DocLineComment, .BlockComment, .DocBlockComment => {
                    try self.emitComment(piece);
                },
                .Newline => {
                    if (self.writer.current_line_length > 0) {
                        try self.writer.newline();
                    }
                },
                else => {},
            }
        }
    }

    fn emitComment(self: *Formatter, piece: lib.lexer.TriviaPiece) FormatError!void {
        const start: usize = @intCast(piece.span.start_offset);
        const end: usize = @intCast(piece.span.end_offset);
        if (start >= end or end > self.source.len) return;

        if (piece.span.start_column == 1) {
            if (self.writer.current_line_length > 0) {
                try self.writer.newline();
            }
        } else if (self.writer.current_line_length > 0) {
            try self.writer.space();
        }

        try self.writer.write(self.source[start..end]);
        switch (piece.kind) {
            .LineComment, .DocLineComment => try self.writer.newline(),
            .BlockComment, .DocBlockComment => {
                if (piece.span.end_line > piece.span.start_line or piece.span.start_column == 1) {
                    try self.writer.newline();
                } else {
                    self.pending_space = true;
                }
            },
            else => {},
        }
    }

    fn formatLeftBrace(self: *Formatter, token: lib.Token) FormatError!void {
        try self.emitPreTokenSpacing(token);
        try self.writer.write("{");
        self.writer.indent();
        self.pending_space = false;
        try self.writer.newline();
    }

    fn formatRightBrace(self: *Formatter, token: lib.Token, next: ?lib.Token) FormatError!void {
        _ = token;
        self.writer.dedent();
        if (self.writer.current_line_length > 0) {
            try self.writer.newline();
        }
        try self.writer.write("}");
        self.pending_space = if (next) |n| isBraceTrailer(n.type) else false;
    }

    fn formatSemicolon(self: *Formatter) FormatError!void {
        try self.writer.write(";");
        self.pending_space = false;
        if (self.paren_depth == 0 and self.bracket_depth == 0) {
            try self.writer.newline();
        } else {
            try self.writer.space();
        }
    }

    fn formatComma(self: *Formatter, next: ?lib.Token) FormatError!void {
        _ = next;
        try self.writer.write(",");
        self.pending_space = true;
    }

    fn formatColon(self: *Formatter, next: ?lib.Token) FormatError!void {
        try self.writer.write(":");
        if (next) |next_token| {
            self.pending_space = switch (next_token.type) {
                .RightBrace, .RightParen, .Comma => false,
                else => true,
            };
        } else {
            self.pending_space = false;
        }
    }

    fn formatArrow(self: *Formatter) FormatError!void {
        try self.ensureSpaceBefore();
        try self.writer.write("->");
        self.pending_space = true;
    }

    fn formatLeftParen(self: *Formatter, token: lib.Token) FormatError!void {
        try self.emitPreTokenSpacing(token);
        try self.writer.write("(");
        self.paren_depth += 1;
        self.pending_space = false;
    }

    fn formatRightParen(self: *Formatter) FormatError!void {
        if (self.paren_depth > 0) self.paren_depth -= 1;
        try self.writer.write(")");
        self.pending_space = false;
    }

    fn formatLeftBracket(self: *Formatter, token: lib.Token) FormatError!void {
        try self.emitPreTokenSpacing(token);
        try self.writer.write("[");
        self.bracket_depth += 1;
        self.pending_space = false;
    }

    fn formatRightBracket(self: *Formatter) FormatError!void {
        if (self.bracket_depth > 0) self.bracket_depth -= 1;
        try self.writer.write("]");
        self.pending_space = false;
    }

    fn emitPreTokenSpacing(self: *Formatter, token: lib.Token) FormatError!void {
        if (token.type == .RightBrace) return;

        if (self.pending_space and needsLeadingSpace(token.type, self.last_token_type)) {
            try self.ensureSpaceBefore();
        } else if (shouldForceLeadingSpace(token.type, self.last_token_type)) {
            try self.ensureSpaceBefore();
        }
        self.pending_space = false;
    }

    fn emitPostTokenSpacing(self: *Formatter, token: lib.Token, next: ?lib.Token) FormatError!void {
        if (isKeywordThatNeedsSpaceAfter(token.type, next)) {
            self.pending_space = true;
            return;
        }

        if (isAlwaysSpacedOperator(token.type)) {
            self.pending_space = true;
            return;
        }

        if (isPrefixOperator(token.type, self.last_token_type)) {
            self.pending_space = false;
            return;
        }

        self.pending_space = false;
    }

    fn originalSpan(self: *const Formatter, token: lib.Token) []const u8 {
        const start: usize = @intCast(token.range.start_offset);
        const end: usize = @intCast(token.range.end_offset);
        if (start <= end and end <= self.source.len) return self.source[start..end];
        return token.lexeme;
    }

    fn ensureSpaceBefore(self: *Formatter) FormatError!void {
        if (self.writer.current_line_length == 0 or self.writer.needs_indent) return;
        const written = self.writer.getWritten();
        if (written.len == 0 or written[written.len - 1] == ' ' or written[written.len - 1] == '\n') return;
        try self.writer.space();
    }
};

fn isBraceTrailer(token: lib.TokenType) bool {
    return switch (token) {
        .Else, .Catch => true,
        else => false,
    };
}

fn needsLeadingSpace(current: lib.TokenType, previous: ?lib.TokenType) bool {
    _ = previous;
    return switch (current) {
        .Comma, .Semicolon, .RightParen, .RightBracket, .Dot, .LeftParen, .LeftBracket => false,
        else => true,
    };
}

fn shouldForceLeadingSpace(current: lib.TokenType, previous: ?lib.TokenType) bool {
    const prev = previous orelse return false;
    return switch (current) {
        .LeftBrace => switch (prev) {
            .LeftParen, .LeftBracket, .Dot => false,
            else => true,
        },
        .LeftParen => switch (prev) {
            .If, .Else, .While, .For, .Switch, .Catch => true,
            else => false,
        },
        .Dot => !canEndExpression(prev),
        else => isAlwaysSpacedOperator(current) and !isPrefixOperator(current, previous),
    };
}

fn isKeywordThatNeedsSpaceAfter(token: lib.TokenType, next: ?lib.Token) bool {
    _ = next;
    return switch (token) {
        .Contract,
        .Pub,
        .Fn,
        .Let,
        .Var,
        .Const,
        .Immutable,
        .Storage,
        .Memory,
        .Tstore,
        .Init,
        .Log,
        .If,
        .Else,
        .While,
        .For,
        .Break,
        .Continue,
        .Return,
        .Requires,
        .Ensures,
        .Invariant,
        .Assume,
        .Havoc,
        .Comptime,
        .Import,
        .Struct,
        .Bitfield,
        .Enum,
        .Extern,
        .Trait,
        .Impl,
        .Call,
        .Staticcall,
        .Errors,
        .Error,
        .Try,
        .Catch,
        .Ghost,
        .Assert,
        .From,
        .To,
        .Forall,
        .Exists,
        .Where,
        => true,
        else => false,
    };
}

fn isAlwaysSpacedOperator(token: lib.TokenType) bool {
    return switch (token) {
        .Equal,
        .EqualEqual,
        .BangEqual,
        .Plus,
        .Minus,
        .Star,
        .Slash,
        .Percent,
        .StarStar,
        .StarStarPercent,
        .LessEqual,
        .GreaterEqual,
        .AmpersandEqual,
        .AmpersandAmpersand,
        .PipeEqual,
        .PipePipe,
        .CaretEqual,
        .PlusPercent,
        .PlusPercentEqual,
        .MinusPercent,
        .MinusPercentEqual,
        .StarPercent,
        .StarPercentEqual,
        .LessLessEqual,
        .GreaterGreaterEqual,
        .PlusEqual,
        .MinusEqual,
        .StarEqual,
        .SlashEqual,
        .PercentEqual,
        .StarStarEqual,
        => true,
        else => false,
    };
}

fn isPrefixOperator(token: lib.TokenType, previous: ?lib.TokenType) bool {
    return switch (token) {
        .Bang, .Tilde => true,
        .Ampersand, .Star, .Minus => previous == null or !canEndExpression(previous.?),
        else => false,
    };
}

fn canEndExpression(token: lib.TokenType) bool {
    return switch (token) {
        .Identifier,
        .StringLiteral,
        .RawStringLiteral,
        .CharacterLiteral,
        .IntegerLiteral,
        .BinaryLiteral,
        .HexLiteral,
        .AddressLiteral,
        .BytesLiteral,
        .True,
        .False,
        .RightParen,
        .RightBracket,
        .RightBrace,
        => true,
        else => false,
    };
}
