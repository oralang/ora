const std = @import("std");
const ora = @import("ora");

test "CST populated by parser for top-level declarations" {
    const src =
        "@import(\"std\");\n" ++
        "error MyErr();\n" ++
        "struct S { a: u8; }\n" ++
        "enum E { A, }\n" ++
        "log Ev(a: u8);\n" ++
        "let x: u8;\n" ++
        "fn f() -> void { }\n" ++
        "contract C { }\n";

    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    var lex = ora.lexer.Lexer.init(allocator, src);
    defer lex.deinit();
    const toks = try lex.scanTokens();
    defer allocator.free(toks);

    var arena = ora.ast_arena.AstArena.init(allocator);
    defer arena.deinit();

    var parser = ora.Parser.init(toks, &arena);
    parser.setFileId(7);
    var builder = ora.cst.CstBuilder.init(allocator);
    defer builder.deinit();
    parser.withCst(&builder);

    _ = try parser.parse();

    const root = try builder.buildRoot(toks);
    defer {
        // child_nodes are allocated with the builder allocator; just deinit the builder after test
        // No explicit free needed beyond builder.deinit()
    }
    // Expect 8 top-level nodes in the given order
    try std.testing.expect(root.child_nodes.len == 8);
    const kinds = root.child_nodes;
    try std.testing.expect(kinds[0].kind == .ImportDecl);
    try std.testing.expect(kinds[1].kind == .ErrorDecl);
    try std.testing.expect(kinds[2].kind == .StructDecl);
    try std.testing.expect(kinds[3].kind == .EnumDecl);
    try std.testing.expect(kinds[4].kind == .LogDecl);
    try std.testing.expect(kinds[5].kind == .VarDecl);
    try std.testing.expect(kinds[6].kind == .FunctionDecl);
    try std.testing.expect(kinds[7].kind == .ContractDecl);
}
