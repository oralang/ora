const std = @import("std");
const source = @import("../source/mod.zig");
const green = @import("green.zig");
const SyntaxKind = @import("kinds.zig").SyntaxKind;

pub const SyntaxElement = union(enum) {
    node: SyntaxNode,
    token: SyntaxToken,
};

pub const SyntaxToken = struct {
    tree: *const green.SyntaxTree,
    id: green.GreenTokenId,
    parent: ?green.GreenNodeId,

    pub fn kind(self: SyntaxToken) green.TokenKind {
        return self.tree.token(self.id).kind;
    }

    pub fn range(self: SyntaxToken) source.TextRange {
        return self.tree.token(self.id).range;
    }

    pub fn text(self: SyntaxToken) []const u8 {
        return self.tree.tokenText(self.id);
    }
};

pub const SyntaxNodePtr = struct {
    kind: SyntaxKind,
    range: source.TextRange,

    pub fn resolve(self: SyntaxNodePtr, tree: *const green.SyntaxTree) ?SyntaxNode {
        return resolveInNode(tree, tree.root, null, self.kind, self.range);
    }
};

pub const SyntaxNode = struct {
    tree: *const green.SyntaxTree,
    id: green.GreenNodeId,
    parent: ?green.GreenNodeId,

    pub fn kind(self: SyntaxNode) SyntaxKind {
        return self.tree.nodes[self.id.index()].kind;
    }

    pub fn range(self: SyntaxNode) source.TextRange {
        return self.tree.nodes[self.id.index()].range;
    }

    pub fn childCount(self: SyntaxNode) usize {
        return self.tree.nodes[self.id.index()].children_len;
    }

    pub fn childAt(self: SyntaxNode, index: usize) ?SyntaxElement {
        const node = self.tree.nodes[self.id.index()];
        if (index >= node.children_len) return null;
        const child = self.tree.children[node.children_start + index];
        return switch (child) {
            .node => |child_id| .{ .node = .{ .tree = self.tree, .id = child_id, .parent = self.id } },
            .token => |token_id| .{ .token = .{ .tree = self.tree, .id = token_id, .parent = self.id } },
        };
    }

    pub fn children(self: SyntaxNode) ChildIterator {
        const node = self.tree.nodes[self.id.index()];
        return .{
            .tree = self.tree,
            .parent = self.id,
            .slice = self.tree.children[node.children_start .. node.children_start + node.children_len],
            .index = 0,
        };
    }

    pub fn firstToken(self: SyntaxNode) ?SyntaxToken {
        var iterator = self.children();
        while (iterator.next()) |child| {
            switch (child) {
                .token => |token| return token,
                .node => |node| if (node.firstToken()) |token| return token,
            }
        }
        return null;
    }

    pub fn lastToken(self: SyntaxNode) ?SyntaxToken {
        const count = self.childCount();
        var index: usize = count;
        while (index > 0) {
            index -= 1;
            const child = self.childAt(index).?;
            switch (child) {
                .token => |token| return token,
                .node => |node| if (node.lastToken()) |token| return token,
            }
        }
        return null;
    }

    pub fn ptr(self: SyntaxNode) SyntaxNodePtr {
        return .{
            .kind = self.kind(),
            .range = self.range(),
        };
    }
};

pub const ChildIterator = struct {
    tree: *const green.SyntaxTree,
    parent: green.GreenNodeId,
    slice: []const green.GreenChild,
    index: usize,

    pub fn next(self: *ChildIterator) ?SyntaxElement {
        if (self.index >= self.slice.len) return null;
        defer self.index += 1;
        return switch (self.slice[self.index]) {
            .node => |child_id| .{ .node = .{ .tree = self.tree, .id = child_id, .parent = self.parent } },
            .token => |token_id| .{ .token = .{ .tree = self.tree, .id = token_id, .parent = self.parent } },
        };
    }
};

fn resolveInNode(
    tree: *const green.SyntaxTree,
    node_id: green.GreenNodeId,
    parent: ?green.GreenNodeId,
    kind: SyntaxKind,
    range: source.TextRange,
) ?SyntaxNode {
    const node = tree.nodes[node_id.index()];
    if (node.kind == kind and node.range.start == range.start and node.range.end == range.end) {
        return .{ .tree = tree, .id = node_id, .parent = parent };
    }

    const children = tree.children[node.children_start .. node.children_start + node.children_len];
    for (children) |child| {
        if (child == .node) {
            if (resolveInNode(tree, child.node, node_id, kind, range)) |found| {
                return found;
            }
        }
    }
    return null;
}
