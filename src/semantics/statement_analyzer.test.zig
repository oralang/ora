// ============================================================================
// Statement Analyzer Tests
// ============================================================================

const std = @import("std");
const testing = std.testing;
const ora_root = @import("ora_root");
const lexer = ora_root.lexer;
const parser_mod = ora_root.parser;
fn expectTypeResolutionError(allocator: std.mem.Allocator, source: []const u8) !void {
    var lex = lexer.Lexer.init(allocator, source);
    defer lex.deinit();
    const tokens = try lex.scanTokens();
    defer allocator.free(tokens);

    _ = parser_mod.parseWithArena(allocator, tokens) catch |err| {
        try testing.expect(err == parser_mod.ParserError.TypeResolutionFailed);
        return;
    };
    return error.TestUnexpectedResult;
}

test "try requires error union operand" {
    const allocator = testing.allocator;
    const source =
        \\fn ok() -> u256 { return 1; }
        \\fn caller() -> u256 {
        \\  let x = try ok();
        \\  return x;
        \\}
    ;

    try expectTypeResolutionError(allocator, source);
}

test "error union call must be handled in expression statement" {
    const allocator = testing.allocator;
    const source =
        \\error Fail;
        \\fn mayFail() -> !u256 { return error.Fail; }
        \\fn caller() {
        \\  mayFail();
        \\}
    ;

    try expectTypeResolutionError(allocator, source);
}

test "error union assignment to non error type is rejected" {
    const allocator = testing.allocator;
    const source =
        \\error Fail;
        \\fn mayFail() -> !u256 { return error.Fail; }
        \\fn caller() {
        \\  let x: u256 = mayFail();
        \\  _ = x;
        \\}
    ;

    try expectTypeResolutionError(allocator, source);
}

test "error union can be assigned to error union variable" {
    const allocator = testing.allocator;
    const source =
        \\error Fail;
        \\fn mayFail() -> !u256 { return error.Fail; }
        \\fn caller() -> !u256 {
        \\  let x: !u256 = mayFail();
        \\  return x;
        \\}
    ;

    var lex = lexer.Lexer.init(allocator, source);
    defer lex.deinit();
    const tokens = try lex.scanTokens();
    defer allocator.free(tokens);

    _ = try parser_mod.parseWithArena(allocator, tokens);
}
