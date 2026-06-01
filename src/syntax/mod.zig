pub const SyntaxKind = @import("kinds.zig").SyntaxKind;
const lexer = @import("ora_lexer");
pub const green = @import("green.zig");
pub const red = @import("red.zig");
pub const parser = @import("parser.zig");

pub const TokenKind = green.TokenKind;
pub const TriviaKind = green.TriviaKind;
pub const SyntaxTree = green.SyntaxTree;
pub const SyntaxNode = red.SyntaxNode;
pub const SyntaxToken = red.SyntaxToken;
pub const SyntaxElement = red.SyntaxElement;
pub const SyntaxNodePtr = red.SyntaxNodePtr;
pub const ParseResult = parser.ParseResult;
pub const parse = parser.parse;
pub const isBuiltinTypeKeyword = lexer.isBuiltinTypeKeyword;
pub const isTypeKeyword = lexer.isTypeKeyword;

pub fn rootNode(tree: *const SyntaxTree) SyntaxNode {
    return .{ .tree = tree, .id = tree.root, .parent = null };
}
