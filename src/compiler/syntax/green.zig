const std = @import("std");
const lexer = @import("ora_lexer");
const source = @import("../source/mod.zig");
const SyntaxKind = @import("kinds.zig").SyntaxKind;

pub const TokenKind = lexer.TokenType;
pub const TriviaKind = lexer.TriviaKind;

pub const GreenNodeId = enum(u32) {
    _,

    const Self = @This();

    pub fn fromIndex(idx: usize) Self {
        return @enumFromInt(idx);
    }

    pub fn index(self: Self) usize {
        return @intFromEnum(self);
    }
};

pub const GreenTokenId = enum(u32) {
    _,

    const Self = @This();

    pub fn fromIndex(idx: usize) Self {
        return @enumFromInt(idx);
    }

    pub fn index(self: Self) usize {
        return @intFromEnum(self);
    }
};

pub const GreenTrivia = struct {
    kind: TriviaKind,
    range: source.TextRange,
};

pub const GreenToken = struct {
    kind: TokenKind,
    range: source.TextRange,
    leading_trivia_start: u32,
    leading_trivia_len: u32,
    trailing_trivia_start: u32,
    trailing_trivia_len: u32,
};

pub const GreenChild = union(enum) {
    node: GreenNodeId,
    token: GreenTokenId,
};

pub const GreenNode = struct {
    kind: SyntaxKind,
    children_start: u32,
    children_len: u32,
    range: source.TextRange,
};

pub const SyntaxTree = struct {
    allocator: std.mem.Allocator,
    file_id: source.FileId,
    source_text: []const u8,
    trivia: []GreenTrivia,
    tokens: []GreenToken,
    children: []GreenChild,
    nodes: []GreenNode,
    root: GreenNodeId,

    pub fn deinit(self: *SyntaxTree) void {
        self.allocator.free(self.trivia);
        self.allocator.free(self.tokens);
        self.allocator.free(self.children);
        self.allocator.free(self.nodes);
    }

    pub fn rootRange(self: *const SyntaxTree) source.TextRange {
        return self.nodes[self.root.index()].range;
    }

    pub fn sourceSlice(self: *const SyntaxTree, range: source.TextRange) []const u8 {
        return self.source_text[range.start..range.end];
    }

    pub fn token(self: *const SyntaxTree, id: GreenTokenId) GreenToken {
        return self.tokens[id.index()];
    }

    pub fn tokenText(self: *const SyntaxTree, id: GreenTokenId) []const u8 {
        return self.sourceSlice(self.token(id).range);
    }

    pub fn triviaPiece(self: *const SyntaxTree, index: usize) GreenTrivia {
        return self.trivia[index];
    }

    pub fn triviaText(self: *const SyntaxTree, index: usize) []const u8 {
        return self.sourceSlice(self.trivia[index].range);
    }

    pub fn reconstructSource(self: *const SyntaxTree, allocator: std.mem.Allocator) ![]u8 {
        return allocator.dupe(u8, self.source_text);
    }
};
