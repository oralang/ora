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

fn expectTypeResolutionSuccess(allocator: std.mem.Allocator, source: []const u8) !void {
    var lex = lexer.Lexer.init(allocator, source);
    defer lex.deinit();
    const tokens = try lex.scanTokens();
    defer allocator.free(tokens);

    _ = try parser_mod.parse(allocator, tokens);
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

test "checked compile-time arithmetic overflow fails type resolution" {
    const allocator = testing.allocator;
    const source =
        \\contract OverflowTest {
        \\  pub fn run() {
        \\    var value: u256 = 0;
        \\    value = 2**256 - 1;
        \\  }
        \\}
    ;

    try expectTypeResolutionError(allocator, source);
}

test "checked compile-time call overflow fails type resolution" {
    const allocator = testing.allocator;
    const source =
        \\contract OverflowCallTest {
        \\  fn inc(x: u256) -> u256 {
        \\    return x + 1;
        \\  }
        \\
        \\  pub fn run() {
        \\    const max: u256 = 115792089237316195423570985008687907853269984665640564039457584007913129639935;
        \\    let _ = inc(max);
        \\  }
        \\}
    ;

    try expectTypeResolutionError(allocator, source);
}

test "explicit comptime while with runtime condition fails type resolution" {
    const allocator = testing.allocator;
    const source =
        \\contract ComptimeLoopFail {
        \\  pub fn run(n: u256) -> u256 {
        \\    var i: u256 = n;
        \\    comptime while (i > 0) {
        \\      i = i - 1;
        \\    }
        \\    return i;
        \\  }
        \\}
    ;

    try expectTypeResolutionError(allocator, source);
}

test "definitely-known switch expression without else fails type resolution" {
    const allocator = testing.allocator;
    const source =
        \\contract SwitchKnownNoElseFail {
        \\  pub fn run() {
        \\    let value: u256 = switch (7) {
        \\      1 => 42
        \\    };
        \\  }
        \\}
    ;

    try expectTypeResolutionError(allocator, source);
}

test "definitely-known switch via pure call without else fails type resolution" {
    const allocator = testing.allocator;
    const source =
        \\contract SwitchKnownCallNoElseFail {
        \\  fn id(x: u256) -> u256 {
        \\    return x;
        \\  }
        \\
        \\  pub fn run() {
        \\    let value: u256 = switch (id(7)) {
        \\      1 => 42
        \\    };
        \\  }
        \\}
    ;

    try expectTypeResolutionError(allocator, source);
}

test "runtime-dependent switch expression with else passes type resolution" {
    const allocator = testing.allocator;
    const source =
        \\contract SwitchRuntimeElsePass {
        \\  pub fn run(n: u256) -> u256 {
        \\    let value: u256 = switch (n) {
        \\      1 => 42,
        \\      else => 0
        \\    };
        \\    return value;
        \\  }
        \\}
    ;

    try expectTypeResolutionSuccess(allocator, source);
}
