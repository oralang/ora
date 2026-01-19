// ============================================================================
// Type Resolver Try/Catch Tests
// ============================================================================

const std = @import("std");
const testing = std.testing;
const ora_root = @import("ora_root");
const lexer = ora_root.lexer;
const parser_mod = ora_root.parser;

test "catch variable typed as u256 for error union payload" {
    const allocator = testing.allocator;
    const source =
        \\contract TryCatchTest {
        \\  error E1;
        \\  pub fn mayFail(x: u256) -> !u256 | E1 {
        \\    if (x == 0) { return E1; }
        \\    return x + 1;
        \\  }
        \\  pub fn handle(x: u256) -> u256 {
        \\    try {
        \\      let y: u256 = mayFail(x);
        \\      return y;
        \\    } catch (e) {
        \\      let _err: u256 = e;
        \\      return _err;
        \\    }
        \\  }
        \\}
    ;

    var lex = lexer.Lexer.init(allocator, source);
    defer lex.deinit();
    const tokens = try lex.scanTokens();
    defer allocator.free(tokens);

    _ = try parser_mod.parse(allocator, tokens);
}
