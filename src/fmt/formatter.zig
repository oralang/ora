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
    /// High-water mark of trivia pieces already emitted inline (same-line
    /// trailing comments hoisted by formatSemicolon), so emitLeadingComments
    /// never re-emits them on the next line.
    trivia_emitted_until: u32 = 0,
    paren_depth: usize = 0,
    bracket_depth: usize = 0,
    pending_space: bool = false,
    /// Set when the previous token was a spaced binary operator (`=`, `+`,
    /// `>>`, …). Distinguishes `a + (b)` from call-style keywords like
    /// `call(...)`, which set `pending_space` but must keep `(` tight.
    pending_operator_space: bool = false,
    last_token_type: ?lib.TokenType = null,
    /// Offsets of `<` / `>` / `>>` tokens in binary-operator position. The
    /// same token types close generic argument lists (`>>` as a one-token
    /// double closer), so spacing them needs AST classification, not a
    /// token-type rule.
    angle_binary_offsets: std.AutoHashMapUnmanaged(u32, void) = .{},

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

        try self.collectAngleBinaryOffsets(&lower_result.file);
        defer self.angle_binary_offsets.deinit(self.allocator);

        for (self.tokens, 0..) |token, index| {
            if (token.type == .Eof) break;
            const next = self.peekNonEof(index + 1);
            if (self.last_token_type != .At) {
                try self.emitLeadingComments(token);
            }
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
            .Semicolon => try self.formatSemicolon(token, next),
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
                if (!try self.emitRecoveredKeywordPrefix(token, next)) {
                    try self.writer.write(self.originalSpan(token));
                }
                try self.emitPostTokenSpacing(token, next);
            },
        }
        self.last_token_type = token.type;
    }

    fn emitLeadingComments(self: *Formatter, token: lib.Token) FormatError!void {
        var i = @max(token.leading_trivia_start, self.trivia_emitted_until);
        const end = token.leading_trivia_start + token.leading_trivia_len;
        var newline_count: u32 = 0;
        while (i < end and i < self.trivia.len) : (i += 1) {
            const piece = self.trivia[i];
            switch (piece.kind) {
                .LineComment, .DocLineComment, .BlockComment, .DocBlockComment => {
                    newline_count = 0;
                    try self.emitComment(piece);
                },
                .Newline => {
                    newline_count += 1;
                    if (self.writer.current_line_length > 0) {
                        try self.writer.newline();
                    } else if (newline_count == 2 and self.paren_depth == 0 and self.bracket_depth == 0) {
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

    fn formatSemicolon(self: *Formatter, token: lib.Token, next: ?lib.Token) FormatError!void {
        try self.writer.write(";");
        self.pending_space = false;
        if (self.paren_depth == 0 and self.bracket_depth == 0) {
            // A comment on the same line as the semicolon lexes as leading
            // trivia of the NEXT token; hoist it back before the newline so
            // `x = 1; // why` keeps its author's layout.
            if (next) |next_token| {
                var i = @max(next_token.leading_trivia_start, self.trivia_emitted_until);
                const end = next_token.leading_trivia_start + next_token.leading_trivia_len;
                // Skip whitespace pieces; stop at the first newline, comment,
                // or end — only a comment still on the semicolon's line hoists.
                while (i < end and i < self.trivia.len and self.trivia[i].kind == .Whitespace) i += 1;
                if (i < end and i < self.trivia.len) {
                    const piece = self.trivia[i];
                    const is_comment = switch (piece.kind) {
                        .LineComment, .DocLineComment, .BlockComment, .DocBlockComment => true,
                        else => false,
                    };
                    const ends_line = i + 1 >= end or i + 1 >= self.trivia.len or self.trivia[i + 1].kind == .Newline;
                    if (is_comment and ends_line and piece.span.start_line == token.range.end_line) {
                        const start: usize = @intCast(piece.span.start_offset);
                        const stop: usize = @intCast(piece.span.end_offset);
                        if (start < stop and stop <= self.source.len) {
                            try self.writer.space();
                            try self.writer.write(self.source[start..stop]);
                            self.trivia_emitted_until = i + 1;
                        }
                    }
                }
            }
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

    fn collectAngleBinaryOffsets(self: *Formatter, file: *const lib.compiler.ast.AstFile) FormatError!void {
        for (file.expressions) |expr| {
            const bin = switch (expr) {
                .Binary => |b| b,
                else => continue,
            };
            const wanted: lib.TokenType = switch (bin.op) {
                .lt => .Less,
                .gt => .Greater,
                .shr => .GreaterGreater,
                else => continue,
            };
            const window_start = lib.compiler.source.rangeOf(file.expression(bin.lhs).*).end;
            const window_end = lib.compiler.source.rangeOf(file.expression(bin.rhs).*).start;
            var index = lowerBoundByOffset(self.tokens, window_start);
            while (index < self.tokens.len) : (index += 1) {
                const token = self.tokens[index];
                if (token.range.start_offset >= window_end) break;
                if (token.type == wanted) {
                    self.angle_binary_offsets.put(self.allocator, token.range.start_offset, {}) catch
                        return FormatError.OutOfMemory;
                    break;
                }
            }
        }
    }

    fn isAngleBinaryOperator(self: *const Formatter, token: lib.Token) bool {
        return switch (token.type) {
            .Less, .Greater, .GreaterGreater => self.angle_binary_offsets.contains(token.range.start_offset),
            else => false,
        };
    }

    fn emitPreTokenSpacing(self: *Formatter, token: lib.Token) FormatError!void {
        if (token.type == .RightBrace) return;

        if (self.pending_space and needsLeadingSpace(token.type, self.last_token_type)) {
            try self.ensureSpaceBefore();
        } else if (shouldForceLeadingSpace(token.type, self.last_token_type)) {
            try self.ensureSpaceBefore();
        } else if (self.isAngleBinaryOperator(token)) {
            try self.ensureSpaceBefore();
        } else if (self.pending_operator_space and
            (token.type == .LeftParen or token.type == .LeftBracket))
        {
            try self.ensureSpaceBefore();
        }
        self.pending_space = false;
        self.pending_operator_space = false;
    }

    fn emitPostTokenSpacing(self: *Formatter, token: lib.Token, next: ?lib.Token) FormatError!void {
        if (isKeywordThatNeedsSpaceAfter(self.source, token, next)) {
            self.pending_space = true;
            return;
        }

        // Prefix position wins over the spaced-operator rule (`-1`, not `- 1`).
        if (isPrefixOperator(token.type, self.last_token_type)) {
            self.pending_space = false;
            return;
        }

        if (isAlwaysSpacedOperator(token.type)) {
            self.pending_space = true;
            self.pending_operator_space = true;
            return;
        }

        if (self.isAngleBinaryOperator(token)) {
            self.pending_space = true;
            self.pending_operator_space = true;
            return;
        }

        self.pending_space = false;
    }

    fn originalSpan(self: *const Formatter, token: lib.Token) []const u8 {
        const start: usize = @intCast(token.range.start_offset);
        const end: usize = @intCast(token.range.end_offset);
        if (start <= end and end <= self.source.len) return self.source[start..end];
        return "";
    }

    fn ensureSpaceBefore(self: *Formatter) FormatError!void {
        if (self.writer.current_line_length == 0 or self.writer.needs_indent) return;
        const written = self.writer.getWritten();
        if (written.len == 0 or written[written.len - 1] == ' ' or written[written.len - 1] == '\n') return;
        try self.writer.space();
    }

    fn emitRecoveredKeywordPrefix(self: *Formatter, token: lib.Token, next: ?lib.Token) FormatError!bool {
        const split = recoveredKeywordPrefix(self.originalSpan(token), token, next) orelse return false;
        try self.writer.write(split.keyword);
        try self.writer.space();
        try self.writer.write(split.suffix);
        return true;
    }
};

const KeywordPrefixSplit = struct {
    keyword: []const u8,
    suffix: []const u8,
};

fn recoveredKeywordPrefix(text: []const u8, token: lib.Token, next: ?lib.Token) ?KeywordPrefixSplit {
    if (token.type != .Identifier) return null;
    const keyword = "modifies";
    if (!std.mem.startsWith(u8, text, keyword)) return null;
    if (text.len == keyword.len) return null;

    const suffix = text[keyword.len..];
    if (suffix.len == 0 or !lib.lexer.isIdentifierStart(suffix[0])) return null;
    const next_token = next orelse return null;
    switch (next_token.type) {
        .LeftBracket, .Dot, .Comma, .LeftBrace => {},
        else => return null,
    }
    return .{ .keyword = keyword, .suffix = suffix };
}

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
    if (isWordBoundaryToken(current) and (isWordBoundaryToken(prev) or canEndExpression(prev))) return true;
    return switch (current) {
        .LeftBrace => switch (prev) {
            .LeftParen, .LeftBracket, .Dot => false,
            else => true,
        },
        .LeftBracket => switch (prev) {
            .Equal => true,
            else => false,
        },
        .LeftParen => switch (prev) {
            .If, .Else, .While, .For, .Switch, .Catch => true,
            else => false,
        },
        .Dot => !canEndExpression(prev),
        else => isAlwaysSpacedOperator(current) and !isPrefixOperator(current, previous),
    };
}

fn lowerBoundByOffset(tokens: []const lib.Token, offset: u32) usize {
    var low: usize = 0;
    var high: usize = tokens.len;
    while (low < high) {
        const mid = low + (high - low) / 2;
        if (tokens[mid].range.start_offset < offset) {
            low = mid + 1;
        } else {
            high = mid;
        }
    }
    return low;
}

fn isKeywordThatNeedsSpaceAfter(source: []const u8, token: lib.Token, next: ?lib.Token) bool {
    _ = next;
    if (token.type == .Identifier and std.mem.eql(u8, lib.lexer.tokenLexeme(source, token), "type")) return true;
    return switch (token.type) {
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
        .Guard,
        .Ensures,
        .EnsuresOk,
        .EnsuresErr,
        .Invariant,
        .Modifies,
        .Decreases,
        .Increases,
        .Assume,
        .Havoc,
        .Inline,
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
        .Resource,
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

fn isWordBoundaryToken(token: lib.TokenType) bool {
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
        .Guard,
        .Ensures,
        .EnsuresOk,
        .EnsuresErr,
        .Invariant,
        .Old,
        .Result,
        .Modifies,
        .Decreases,
        .Increases,
        .Assume,
        .Havoc,
        .Inline,
        .Comptime,
        .As,
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
        .True,
        .False,
        .Error,
        .Resource,
        .Try,
        .Catch,
        .Switch,
        .Match,
        .Ghost,
        .Assert,
        .Void,
        .From,
        .To,
        .Forall,
        .Exists,
        .Where,
        .U8,
        .U16,
        .U32,
        .U64,
        .U128,
        .U160,
        .U256,
        .I8,
        .I16,
        .I32,
        .I64,
        .I128,
        .I256,
        .Bool,
        .Address,
        .String,
        .Map,
        .Slice,
        .Bytes,
        .Identifier,
        .StringLiteral,
        .RawStringLiteral,
        .CharacterLiteral,
        .IntegerLiteral,
        .BinaryLiteral,
        .HexLiteral,
        .AddressLiteral,
        .BytesLiteral,
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
        .Ampersand,
        .Caret,
        .LessLess,
        .LessLessPercent,
        .GreaterGreaterPercent,
        .LessEqual,
        .GreaterEqual,
        .AmpersandEqual,
        .AmpersandAmpersand,
        .Pipe,
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
        .Bang, .Tilde, .At => true,
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
