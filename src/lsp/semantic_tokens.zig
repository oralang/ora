const std = @import("std");
const lexer_mod = @import("ora_lexer");
const frontend = @import("frontend.zig");
const semantic_index = @import("semantic_index.zig");

const Allocator = std.mem.Allocator;
const TokenType = lexer_mod.TokenType;

/// LSP semantic token types — indices into the legend array.
pub const SemanticTokenKind = enum(u32) {
    namespace = 0, // contract names
    type = 1, // type keywords, struct/enum/bitfield names
    @"struct" = 2,
    @"enum" = 3,
    enumMember = 4,
    parameter = 5,
    variable = 6,
    property = 7, // fields
    function = 8,
    method = 9,
    macro = 10, // comptime
    keyword = 11,
    comment = 12,
    string = 13,
    number = 14,
    operator = 15,
    decorator = 16, // @region attributes
    event = 17, // log declarations

    pub const legend = [_][]const u8{
        "namespace",
        "type",
        "struct",
        "enum",
        "enumMember",
        "parameter",
        "variable",
        "property",
        "function",
        "method",
        "macro",
        "keyword",
        "comment",
        "string",
        "number",
        "operator",
        "decorator",
        "event",
    };
};

/// LSP semantic token modifiers — bitmask positions.
pub const SemanticTokenModifier = enum(u5) {
    declaration = 0,
    definition = 1,
    readonly = 2,
    @"static" = 3,
    defaultLibrary = 4,

    pub const legend = [_][]const u8{
        "declaration",
        "definition",
        "readonly",
        "static",
        "defaultLibrary",
    };

    pub fn mask(self: SemanticTokenModifier) u32 {
        return @as(u32, 1) << @intFromEnum(self);
    }
};

pub const SemanticToken = struct {
    line: u32,
    start_char: u32,
    length: u32,
    kind: SemanticTokenKind,
    modifiers: u32,
};

/// Tokenize source into semantic tokens for LSP.
pub fn tokenize(allocator: Allocator, source: []const u8) ![]SemanticToken {
    var tokens_list = std.ArrayList(SemanticToken){};
    errdefer tokens_list.deinit(allocator);

    // Phase 1: Lex source and classify tokens.
    var lex = try lexer_mod.Lexer.initWithConfig(allocator, source, lexer_mod.LexerConfig.development());
    defer lex.deinit();
    const tokens = lex.scanTokens() catch return try tokens_list.toOwnedSlice(allocator);
    defer allocator.free(tokens);

    // Phase 2: Build semantic index for identifier enrichment.
    var index = semantic_index.indexDocument(allocator, source) catch null;
    defer if (index) |*idx| idx.deinit(allocator);

    // Phase 3: Extract comment ranges from trivia.
    try extractComments(&tokens_list, allocator, tokens, source);

    // Phase 4: Classify each lexer token.
    for (tokens) |token| {
        if (token.type == .Eof) continue;
        const len: u32 = @intCast(token.lexeme.len);
        if (len == 0) continue;

        const tok_line = if (token.line > 0) token.line - 1 else 0;
        const tok_char = if (token.column > 0) token.column - 1 else 0;

        if (classifyToken(token, index)) |classification| {
            try tokens_list.append(allocator, .{
                .line = tok_line,
                .start_char = tok_char,
                .length = len,
                .kind = classification.kind,
                .modifiers = classification.modifiers,
            });
        }
    }

    // Sort by position (line, then character).
    std.sort.heap(SemanticToken, tokens_list.items, {}, lessThanSemanticToken);

    return tokens_list.toOwnedSlice(allocator);
}

const Classification = struct {
    kind: SemanticTokenKind,
    modifiers: u32,
};

fn classifyToken(token: lexer_mod.Token, maybe_index: ?semantic_index.SemanticIndex) ?Classification {
    const tt = token.type;

    // Keywords
    if (isVerificationKeyword(tt)) {
        return .{ .kind = .keyword, .modifiers = 0 };
    }
    if (isControlFlowKeyword(tt) or isDeclarationKeyword(tt) or isOtherKeyword(tt)) {
        return .{ .kind = .keyword, .modifiers = 0 };
    }

    // Comptime
    if (tt == .Comptime) {
        return .{ .kind = .macro, .modifiers = 0 };
    }

    // Type keywords (primitives)
    if (isTypeKeyword(tt)) {
        return .{ .kind = .type, .modifiers = SemanticTokenModifier.mask(.defaultLibrary) };
    }

    // Literals
    if (tt == .True or tt == .False) {
        return .{ .kind = .keyword, .modifiers = 0 };
    }
    if (tt == .StringLiteral or tt == .RawStringLiteral or tt == .CharacterLiteral) {
        return .{ .kind = .string, .modifiers = 0 };
    }
    if (tt == .IntegerLiteral or tt == .BinaryLiteral or tt == .HexLiteral or tt == .AddressLiteral or tt == .BytesLiteral) {
        return .{ .kind = .number, .modifiers = 0 };
    }

    // Operators and delimiters
    if (isOperator(tt)) {
        return .{ .kind = .operator, .modifiers = 0 };
    }

    // @ symbol (decorator-like)
    if (tt == .At) {
        return .{ .kind = .decorator, .modifiers = 0 };
    }

    // Identifiers — enrich via semantic index
    if (tt == .Identifier) {
        return classifyIdentifier(token, maybe_index);
    }

    // Delimiters — skip (not typically highlighted semantically)
    return null;
}

fn classifyIdentifier(token: lexer_mod.Token, maybe_index: ?semantic_index.SemanticIndex) Classification {
    const idx = maybe_index orelse return .{ .kind = .variable, .modifiers = 0 };
    const tok_line = if (token.line > 0) token.line - 1 else 0;
    const tok_char = if (token.column > 0) token.column - 1 else 0;
    const tok_end = tok_char + @as(u32, @intCast(token.lexeme.len));

    // Check if this identifier matches a symbol's selection range (declaration site).
    for (idx.symbols) |symbol| {
        const sel = symbol.selection_range;
        if (sel.start.line == tok_line and sel.start.character == tok_char and sel.end.character == tok_end) {
            return .{
                .kind = symbolKindToTokenKind(symbol.kind),
                .modifiers = symbolKindToModifiers(symbol.kind) | SemanticTokenModifier.mask(.declaration) | SemanticTokenModifier.mask(.definition),
            };
        }
    }

    // Not at declaration — try to match by name for type-level symbols.
    for (idx.symbols) |symbol| {
        if (std.mem.eql(u8, symbol.name, token.lexeme)) {
            return .{
                .kind = symbolKindToTokenKind(symbol.kind),
                .modifiers = symbolKindToModifiers(symbol.kind),
            };
        }
    }

    return .{ .kind = .variable, .modifiers = 0 };
}

fn symbolKindToTokenKind(kind: semantic_index.SymbolKind) SemanticTokenKind {
    return switch (kind) {
        .contract => .namespace,
        .function => .function,
        .method => .method,
        .variable => .variable,
        .field => .property,
        .constant => .variable,
        .parameter => .parameter,
        .struct_decl => .@"struct",
        .bitfield_decl => .@"struct",
        .enum_decl => .@"enum",
        .enum_member => .enumMember,
        .event => .event,
        .error_decl => .type,
    };
}

fn symbolKindToModifiers(kind: semantic_index.SymbolKind) u32 {
    return switch (kind) {
        .constant => SemanticTokenModifier.mask(.readonly),
        else => 0,
    };
}

fn extractComments(
    tokens_list: *std.ArrayList(SemanticToken),
    allocator: Allocator,
    tokens: []const lexer_mod.Token,
    source: []const u8,
) !void {
    for (tokens) |token| {
        if (token.leading_trivia_len == 0) continue;
        const trivia_start: usize = token.leading_trivia_start;
        const trivia_end: usize = trivia_start + token.leading_trivia_len;
        if (trivia_end > source.len) continue;

        const trivia = source[trivia_start..trivia_end];
        var i: usize = 0;
        while (i < trivia.len) {
            if (i + 1 < trivia.len and trivia[i] == '/' and trivia[i + 1] == '/') {
                // Line comment — find end.
                const comment_start = trivia_start + i;
                const start_i = i;
                while (i < trivia.len and trivia[i] != '\n') : (i += 1) {}
                const comment_len = i - start_i;
                const pos = byteOffsetToPosition(source, comment_start);
                try tokens_list.append(allocator, .{
                    .line = pos.line,
                    .start_char = pos.character,
                    .length = @intCast(comment_len),
                    .kind = .comment,
                    .modifiers = 0,
                });
            } else if (i + 1 < trivia.len and trivia[i] == '/' and trivia[i + 1] == '*') {
                // Block comment — find end.
                const comment_start = trivia_start + i;
                i += 2;
                while (i + 1 < trivia.len) : (i += 1) {
                    if (trivia[i] == '*' and trivia[i + 1] == '/') {
                        i += 2;
                        break;
                    }
                }
                const comment_len = (trivia_start + i) - comment_start;
                // For block comments spanning multiple lines, emit one token per line.
                const block_text = source[comment_start .. comment_start + comment_len];
                try emitMultilineComment(tokens_list, allocator, source, comment_start, block_text);
            } else {
                i += 1;
            }
        }
    }
}

fn emitMultilineComment(
    tokens_list: *std.ArrayList(SemanticToken),
    allocator: Allocator,
    source: []const u8,
    start_offset: usize,
    text: []const u8,
) !void {
    var offset: usize = 0;
    while (offset < text.len) {
        const line_start = offset;
        while (offset < text.len and text[offset] != '\n') : (offset += 1) {}
        const line_len = offset - line_start;
        if (line_len > 0) {
            const pos = byteOffsetToPosition(source, start_offset + line_start);
            try tokens_list.append(allocator, .{
                .line = pos.line,
                .start_char = pos.character,
                .length = @intCast(line_len),
                .kind = .comment,
                .modifiers = 0,
            });
        }
        if (offset < text.len) offset += 1; // skip \n
    }
}

fn byteOffsetToPosition(source: []const u8, offset: usize) frontend.Position {
    var line: u32 = 0;
    var col: u32 = 0;
    for (source[0..@min(offset, source.len)]) |ch| {
        if (ch == '\n') {
            line += 1;
            col = 0;
        } else {
            col += 1;
        }
    }
    return .{ .line = line, .character = col };
}

/// Encode semantic tokens as LSP relative-position u32 array.
pub fn encodeTokens(allocator: Allocator, tokens: []const SemanticToken) ![]u32 {
    const data = try allocator.alloc(u32, tokens.len * 5);
    var prev_line: u32 = 0;
    var prev_char: u32 = 0;

    for (tokens, 0..) |token, i| {
        const base = i * 5;
        const delta_line = token.line - prev_line;
        const delta_char = if (delta_line == 0) token.start_char - prev_char else token.start_char;
        data[base + 0] = delta_line;
        data[base + 1] = delta_char;
        data[base + 2] = token.length;
        data[base + 3] = @intFromEnum(token.kind);
        data[base + 4] = token.modifiers;
        prev_line = token.line;
        prev_char = token.start_char;
    }

    return data;
}

// --- Token classification helpers ---

fn isVerificationKeyword(tt: TokenType) bool {
    return switch (tt) {
        .Requires, .Ensures, .Invariant, .Old, .Result, .Modifies, .Decreases, .Increases, .Assume, .Havoc, .Ghost, .Assert, .Forall, .Exists, .Where => true,
        else => false,
    };
}

fn isControlFlowKeyword(tt: TokenType) bool {
    return switch (tt) {
        .If, .Else, .While, .For, .Break, .Continue, .Return, .Switch, .Try, .Catch => true,
        else => false,
    };
}

fn isDeclarationKeyword(tt: TokenType) bool {
    return switch (tt) {
        .Contract, .Fn, .Let, .Var, .Const, .Pub, .Import, .Struct, .Bitfield, .Enum, .Extern, .Trait, .Impl, .Log, .Error, .Init, .Errors => true,
        else => false,
    };
}

fn isOtherKeyword(tt: TokenType) bool {
    return switch (tt) {
        .Immutable, .Storage, .Memory, .Tstore, .As, .Call, .Staticcall, .Void, .From, .To => true,
        else => false,
    };
}

fn isTypeKeyword(tt: TokenType) bool {
    return switch (tt) {
        .U8, .U16, .U32, .U64, .U128, .U256, .I8, .I16, .I32, .I64, .I128, .I256, .Bool, .Address, .String, .Map, .Slice, .Bytes => true,
        else => false,
    };
}

fn isOperator(tt: TokenType) bool {
    return switch (tt) {
        .Plus, .Minus, .Star, .Slash, .Percent, .StarStar, .StarStarPercent, .Equal, .EqualEqual, .BangEqual, .Less, .LessEqual, .Greater, .GreaterEqual, .LessLess, .LessLessEqual, .GreaterGreater, .GreaterGreaterEqual, .Bang, .Tilde, .Ampersand, .AmpersandEqual, .AmpersandAmpersand, .Pipe, .PipeEqual, .PipePipe, .Caret, .CaretEqual, .PlusPercent, .PlusPercentEqual, .MinusPercent, .MinusPercentEqual, .StarPercent, .StarPercentEqual, .LessLessPercent, .GreaterGreaterPercent, .PlusEqual, .MinusEqual, .StarEqual, .SlashEqual, .PercentEqual, .StarStarEqual, .Arrow, .DotDot, .DotDotDot => true,
        else => false,
    };
}

fn lessThanSemanticToken(_: void, a: SemanticToken, b: SemanticToken) bool {
    if (a.line != b.line) return a.line < b.line;
    return a.start_char < b.start_char;
}
