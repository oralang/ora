const lsp = @import("lsp");
const ora_root = @import("ora_root");

const semantic_tokens = ora_root.lsp.semantic_tokens;

const Allocator = @import("std").mem.Allocator;
const types = lsp.types;

pub const BuildResult = struct {
    response: types.SemanticTokens,
    data_count: usize,
};

pub fn build(data: []u32) types.SemanticTokens {
    return .{ .data = data };
}

pub fn buildWithStats(arena: Allocator, tokens: []const semantic_tokens.SemanticToken) !BuildResult {
    const data = try semantic_tokens.encodeTokens(arena, tokens);
    return .{
        .response = build(data),
        .data_count = data.len,
    };
}
