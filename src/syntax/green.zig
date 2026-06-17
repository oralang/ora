const std = @import("std");
const lexer = @import("ora_lexer");
const source = @import("../source/mod.zig");
const SyntaxKind = @import("kinds.zig").SyntaxKind;

pub const TokenKind = lexer.TokenType;
pub const TriviaKind = lexer.TriviaKind;

pub const GreenNodeId = source.defineId("GreenNodeId");
pub const GreenTokenId = source.defineId("GreenTokenId");

pub const GreenTrivia = struct {
    kind: TriviaKind,
    range: source.TextRange,
};

pub const GreenToken = struct {
    kind: TokenKind,
    range: source.TextRange,
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
    trivia_capacity: usize,
    tokens: []GreenToken,
    tokens_capacity: usize,
    children: []GreenChild,
    children_capacity: usize,
    nodes: []GreenNode,
    nodes_capacity: usize,
    root: GreenNodeId,

    pub fn deinit(self: *SyntaxTree) void {
        freeBuffer(GreenTrivia, self.allocator, self.trivia, self.trivia_capacity);
        freeBuffer(GreenToken, self.allocator, self.tokens, self.tokens_capacity);
        freeBuffer(GreenChild, self.allocator, self.children, self.children_capacity);
        freeBuffer(GreenNode, self.allocator, self.nodes, self.nodes_capacity);
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

fn freeBuffer(comptime T: type, allocator: std.mem.Allocator, items: []T, capacity: usize) void {
    if (capacity == 0) return;
    allocator.free(items.ptr[0..capacity]);
}
