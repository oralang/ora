const lsp = @import("lsp");

const types = lsp.types;

pub fn build(data: []u32) types.SemanticTokens {
    return .{ .data = data };
}
