const std = @import("std");
const ora = @import("ora");
const Lossless = ora.lossless_printer;

test "lossless round-trip: simple function" {
    const src = "pub fn add(x: u8, y: u8) -> u8 {\n    // sum\n    return x + y;\n}\n";
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    var lex = ora.lexer.Lexer.init(allocator, src);
    defer lex.deinit();
    const toks = try lex.scanTokens();
    defer allocator.free(toks);

    const printed = try Lossless.printLossless(allocator, src, toks, lex.getTrivia());
    defer allocator.free(printed);
    try std.testing.expectEqualStrings(src, printed);
}

test "lossless round-trip: comments and whitespace" {
    const src = "// header\n\nstruct S { /* field */ a: u32, }\n\n";
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    var lex = ora.lexer.Lexer.init(allocator, src);
    defer lex.deinit();
    const toks = try lex.scanTokens();
    defer allocator.free(toks);

    const printed = try Lossless.printLossless(allocator, src, toks, lex.getTrivia());
    defer allocator.free(printed);
    try std.testing.expectEqualStrings(src, printed);
}
