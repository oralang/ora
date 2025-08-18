const std = @import("std");
const ora = @import("ora");

test "doc comment extraction: line and block" {
    const src =
        "/// adds two numbers\n" ++
        "/// returns sum\n" ++
        "pub fn add(x: u8, y: u8) -> u8 { return x + y; }\n\n" ++
        "/** type holds a field */\n" ++
        "struct S { a: u32, }\n";

    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    var lex = ora.lexer.Lexer.init(allocator, src);
    defer lex.deinit();
    const toks = try lex.scanTokens();
    defer allocator.free(toks);

    const entries = try ora.doc_comments.extractDocComments(allocator, src, toks, lex.getTrivia());
    defer {
        for (entries) |e| allocator.free(e.text);
        allocator.free(entries);
    }

    try std.testing.expect(entries.len >= 2);
    // First doc should contain 'adds two numbers'
    try std.testing.expect(std.mem.indexOf(u8, entries[0].text, "adds two numbers") != null);
    // Second doc should contain 'type holds a field'
    var found = false;
    for (entries) |e| {
        if (std.mem.indexOf(u8, e.text, "type holds a field") != null) found = true;
    }
    try std.testing.expect(found);
}
