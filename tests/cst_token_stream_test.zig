const std = @import("std");
const ora = @import("ora");

test "CST token stream root builds" {
    const src = "let x: u8 = 1;\n";
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    var lex = ora.lexer.Lexer.init(allocator, src);
    defer lex.deinit();
    const toks = try lex.scanTokens();
    defer allocator.free(toks);

    var builder = ora.cst.CstBuilder.init(allocator);
    defer builder.deinit();
    const root = try builder.buildTokenStream(toks);
    try std.testing.expect(root.kind == .Root);
    try std.testing.expect(root.child_nodes.len == 1);
    try std.testing.expect(root.child_nodes[0].kind == .TokenStream);
    try std.testing.expect(root.child_nodes[0].token_indices.len > 0);
}
