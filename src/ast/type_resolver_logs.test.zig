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

test "runtime storage field access via local alias passes type resolution" {
    const allocator = testing.allocator;
    const source =
        \\bitfield TokenFlags : u256 {
        \\  decimals: u8;
        \\}
        \\
        \\contract StorageAliasRead {
        \\  storage var token_flags: TokenFlags;
        \\
        \\  pub fn decimals() -> u8 {
        \\    let f: TokenFlags = token_flags;
        \\    return f.decimals;
        \\  }
        \\}
    ;

    try expectTypeResolutionSuccess(allocator, source);
}

test "mutable var does not force comptime call overflow check" {
    const allocator = testing.allocator;
    const source =
        \\contract MutableVarRuntimeBinding {
        \\  fn inc(x: u256) -> u256 {
        \\    return x + 1;
        \\  }
        \\
        \\  pub fn run() -> u256 {
        \\    var max: u256 = 115792089237316195423570985008687907853269984665640564039457584007913129639935;
        \\    return inc(max);
        \\  }
        \\}
    ;

    try expectTypeResolutionSuccess(allocator, source);
}

test "storage bitfield alias read after init write stays runtime" {
    const allocator = testing.allocator;
    const source =
        \\bitfield TokenFlags : u256 {
        \\  initialized: bool;
        \\  paused: bool;
        \\  decimals: u8;
        \\}
        \\
        \\contract StorageAliasAfterInit {
        \\  storage var token_flags: TokenFlags;
        \\
        \\  fn ct(comptime T: type, comptime value: T) -> T {
        \\    return value;
        \\  }
        \\
        \\  pub fn init() {
        \\    let f: TokenFlags = .{
        \\      .initialized = true,
        \\      .paused = false,
        \\      .decimals = ct(u8, 18),
        \\    };
        \\    token_flags = f;
        \\  }
        \\
        \\  pub fn decimals() -> u8 {
        \\    let f: TokenFlags = token_flags;
        \\    return f.decimals;
        \\  }
        \\}
    ;

    try expectTypeResolutionSuccess(allocator, source);
}

test "explicit comptime while with mutable local counter passes type resolution" {
    const allocator = testing.allocator;
    const source =
        \\contract ComptimeLoopProbe {
        \\  pub fn test_comptime_while() -> u256 {
        \\    var total: u256 = 0;
        \\    comptime while (total < 5) {
        \\      total = total + 1;
        \\    }
        \\    return total;
        \\  }
        \\}
    ;

    try expectTypeResolutionSuccess(allocator, source);
}

test "mutable while condition is not frozen before comptime call evaluation" {
    const allocator = testing.allocator;
    const source =
        \\contract MutableWhileCondition {
        \\  fn ten() -> u256 {
        \\    var i: u256 = 0;
        \\    while (i < 10) {
        \\      i = i + 1;
        \\    }
        \\    return i;
        \\  }
        \\
        \\  pub fn run() -> u256 {
        \\    const value: u256 = ten();
        \\    return value;
        \\  }
        \\}
    ;

    try expectTypeResolutionSuccess(allocator, source);
}

test "mutable for progression is not frozen before comptime call evaluation" {
    const allocator = testing.allocator;
    const source =
        \\contract MutableForProgression {
        \\  fn sum_to(n: u256) -> u256 {
        \\    var sum: u256 = 0;
        \\    var i: u256 = 0;
        \\    for (n) |x| {
        \\      sum = sum + i;
        \\      i = i + 1;
        \\    }
        \\    return sum;
        \\  }
        \\
        \\  pub fn run() -> u256 {
        \\    const value: u256 = sum_to(10);
        \\    return value;
        \\  }
        \\}
    ;

    try expectTypeResolutionSuccess(allocator, source);
}

test "wrapping add in narrow integer type does not trigger checked overflow" {
    const allocator = testing.allocator;
    const source =
        \\contract ComptimeWrappingU8 {
        \\  pub fn run() -> u8 {
        \\    const x: u8 = 255 +% 1;
        \\    const y: u8 = x + 1;
        \\    return y;
        \\  }
        \\}
    ;

    try expectTypeResolutionSuccess(allocator, source);
}

test "if with unknown condition does not hard-fail both branches on checked arithmetic" {
    const allocator = testing.allocator;
    const source =
        \\contract IfUnknownBranchPolicy {
        \\  pub fn run(flag: bool) -> bool {
        \\    if (flag) {
        \\      return true;
        \\    } else {
        \\      let z: u256 = 0;
        \\      let q: u256 = 1 / z;
        \\      return q == 1;
        \\    }
        \\  }
        \\}
    ;

    try expectTypeResolutionSuccess(allocator, source);
}

test "if with known true condition skips dead else branch hard arithmetic checks" {
    const allocator = testing.allocator;
    const source =
        \\contract IfKnownTrueSkipsElse {
        \\  pub fn run() -> u256 {
        \\    if (true) {
        \\      return 1;
        \\    } else {
        \\      let z: u256 = 0;
        \\      let q: u256 = 1 / z;
        \\      return q;
        \\    }
        \\  }
        \\}
    ;

    try expectTypeResolutionSuccess(allocator, source);
}

test "if with known true condition still hard-fails checked arithmetic in reachable branch" {
    const allocator = testing.allocator;
    const source =
        \\contract IfKnownTrueThenBranchFails {
        \\  pub fn run() -> u256 {
        \\    if (true) {
        \\      let z: u256 = 0;
        \\      let q: u256 = 1 / z;
        \\      return q;
        \\    }
        \\    return 0;
        \\  }
        \\}
    ;

    try expectTypeResolutionError(allocator, source);
}

test "requires clause remains proof assumption and does not force dead branch fold failure" {
    const allocator = testing.allocator;
    const source =
        \\contract RequiresAssumptionPolicy {
        \\  pub fn g(a: NonZeroAddress) -> bool
        \\      requires(a == std.msg.sender())
        \\      ensures(a == std.msg.sender())
        \\  {
        \\      let s: NonZeroAddress = std.msg.sender();
        \\      if (s == a) {
        \\          return true;
        \\      } else {
        \\          let z: u256 = 0;
        \\          let q: u256 = 1 / z;
        \\          return q == 1;
        \\      }
        \\  }
        \\}
    ;

    try expectTypeResolutionSuccess(allocator, source);
}
