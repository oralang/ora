// ============================================================================
// Type Resolver Log Validation Tests
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

    _ = parser_mod.parse(allocator, tokens) catch |err| {
        try testing.expect(err == parser_mod.ParserError.TypeResolutionFailed);
        return;
    };
    return error.TestUnexpectedResult;
}

test "log unknown event fails type resolution" {
    const allocator = testing.allocator;
    const source =
        \\contract LogTest {
        \\  pub fn run() {
        \\    log MissingEvent();
        \\  }
        \\}
    ;

    try expectTypeResolutionError(allocator, source);
}

test "log argument count mismatch fails type resolution" {
    const allocator = testing.allocator;
    const source =
        \\contract LogTest {
        \\  log Transfer(from: address, amount: u256);
        \\  pub fn run(addr: address) {
        \\    log Transfer(addr);
        \\  }
        \\}
    ;

    try expectTypeResolutionError(allocator, source);
}

test "log argument type mismatch fails type resolution" {
    const allocator = testing.allocator;
    const source =
        \\contract LogTest {
        \\  log Transfer(from: address, amount: u256);
        \\  pub fn run() {
        \\    log Transfer(1, 2);
        \\  }
        \\}
    ;

    try expectTypeResolutionError(allocator, source);
}

test "log with matching signature passes type resolution" {
    const allocator = testing.allocator;
    const source =
        \\contract LogTest {
        \\  log Transfer(from: address, amount: u256);
        \\  pub fn run(addr: address) {
        \\    log Transfer(addr, 2);
        \\  }
        \\}
    ;

    var lex = lexer.Lexer.init(allocator, source);
    defer lex.deinit();
    const tokens = try lex.scanTokens();
    defer allocator.free(tokens);

    _ = try parser_mod.parse(allocator, tokens);
}
