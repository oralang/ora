//! Tiny side-effect-free expression evaluator for the debugger TUI.
//!
//! Used by `:print <expr>` (when the input is more than a bare binding
//! name) and by `:break <line> when <expr>` predicates. Identifiers are
//! resolved through a caller-provided `Resolver`; the evaluator itself
//! does not touch EVM state.
//!
//! Grammar (recursive descent):
//!   expr        := orExpr
//!   orExpr      := andExpr ('||' andExpr)*
//!   andExpr     := equality ('&&' equality)*
//!   equality    := comparison (('==' | '!=') comparison)*
//!   comparison  := additive (('<' | '>' | '<=' | '>=') additive)*
//!   additive    := multiplicative (('+' | '-') multiplicative)*
//!   multiplicative := unary (('*' | '/' | '%') unary)*
//!   unary       := ('!' | '-')? primary
//!   primary     := number | bool | identifier | '(' expr ')'
//!
//! Numbers: decimal `[0-9]+`, hex `0x[0-9a-fA-F]+`. Booleans: `true` / `false`.

const std = @import("std");

pub const Value = union(enum) {
    num: u256,
    bool_: bool,

    pub fn asBool(self: Value) bool {
        return switch (self) {
            .num => |n| n != 0,
            .bool_ => |b| b,
        };
    }

    pub fn asNum(self: Value) u256 {
        return switch (self) {
            .num => |n| n,
            .bool_ => |b| if (b) 1 else 0,
        };
    }
};

pub const EvalError = error{
    ParseError,
    UnknownIdentifier,
    BindingUnavailable,
    DivisionByZero,
    Overflow,
    OutOfMemory,
};

/// Caller-supplied identifier resolver. `resolveFn` returns:
///   - `Value` if the name resolves to a known value at this stop,
///   - `null` if the name is unknown (caller surfaces "binding not visible"),
///   - `error.BindingUnavailable` if the name is known but its current
///     value can't be read (e.g. SSA optimised away).
pub const Resolver = struct {
    ctx: *anyopaque,
    resolveFn: *const fn (ctx: *anyopaque, name: []const u8) EvalError!?Value,

    fn resolve(self: Resolver, name: []const u8) EvalError!?Value {
        return self.resolveFn(self.ctx, name);
    }
};

pub fn evaluate(expr: []const u8, resolver: Resolver) EvalError!Value {
    var p = Parser{ .src = expr, .pos = 0, .resolver = resolver };
    p.skipWs();
    const value = try p.parseExpr();
    p.skipWs();
    if (p.pos != p.src.len) return error.ParseError;
    return value;
}

const Parser = struct {
    src: []const u8,
    pos: usize,
    resolver: Resolver,

    fn skipWs(self: *Parser) void {
        while (self.pos < self.src.len and (self.src[self.pos] == ' ' or self.src[self.pos] == '\t')) {
            self.pos += 1;
        }
    }

    fn peek(self: *Parser) ?u8 {
        if (self.pos >= self.src.len) return null;
        return self.src[self.pos];
    }

    fn consumeIf(self: *Parser, lit: []const u8) bool {
        if (self.pos + lit.len > self.src.len) return false;
        if (!std.mem.eql(u8, self.src[self.pos .. self.pos + lit.len], lit)) return false;
        self.pos += lit.len;
        self.skipWs();
        return true;
    }

    fn parseExpr(self: *Parser) EvalError!Value {
        return self.parseOr();
    }

    fn parseOr(self: *Parser) EvalError!Value {
        var lhs = try self.parseAnd();
        while (self.consumeIf("||")) {
            const rhs = try self.parseAnd();
            lhs = Value{ .bool_ = lhs.asBool() or rhs.asBool() };
        }
        return lhs;
    }

    fn parseAnd(self: *Parser) EvalError!Value {
        var lhs = try self.parseEquality();
        while (self.consumeIf("&&")) {
            const rhs = try self.parseEquality();
            lhs = Value{ .bool_ = lhs.asBool() and rhs.asBool() };
        }
        return lhs;
    }

    fn parseEquality(self: *Parser) EvalError!Value {
        var lhs = try self.parseComparison();
        while (true) {
            if (self.consumeIf("==")) {
                const rhs = try self.parseComparison();
                lhs = Value{ .bool_ = lhs.asNum() == rhs.asNum() };
            } else if (self.consumeIf("!=")) {
                const rhs = try self.parseComparison();
                lhs = Value{ .bool_ = lhs.asNum() != rhs.asNum() };
            } else break;
        }
        return lhs;
    }

    fn parseComparison(self: *Parser) EvalError!Value {
        var lhs = try self.parseAdditive();
        while (true) {
            // Two-char ops first so they don't get partial-eaten by '<' or '>'.
            if (self.consumeIf("<=")) {
                const rhs = try self.parseAdditive();
                lhs = Value{ .bool_ = lhs.asNum() <= rhs.asNum() };
            } else if (self.consumeIf(">=")) {
                const rhs = try self.parseAdditive();
                lhs = Value{ .bool_ = lhs.asNum() >= rhs.asNum() };
            } else if (self.consumeIf("<")) {
                const rhs = try self.parseAdditive();
                lhs = Value{ .bool_ = lhs.asNum() < rhs.asNum() };
            } else if (self.consumeIf(">")) {
                const rhs = try self.parseAdditive();
                lhs = Value{ .bool_ = lhs.asNum() > rhs.asNum() };
            } else break;
        }
        return lhs;
    }

    fn parseAdditive(self: *Parser) EvalError!Value {
        var lhs = try self.parseMultiplicative();
        while (true) {
            if (self.consumeIf("+")) {
                const rhs = try self.parseMultiplicative();
                lhs = Value{ .num = lhs.asNum() +% rhs.asNum() };
            } else if (self.consumeIf("-")) {
                const rhs = try self.parseMultiplicative();
                lhs = Value{ .num = lhs.asNum() -% rhs.asNum() };
            } else break;
        }
        return lhs;
    }

    fn parseMultiplicative(self: *Parser) EvalError!Value {
        var lhs = try self.parseUnary();
        while (true) {
            if (self.consumeIf("*")) {
                const rhs = try self.parseUnary();
                lhs = Value{ .num = lhs.asNum() *% rhs.asNum() };
            } else if (self.consumeIf("/")) {
                const rhs = try self.parseUnary();
                const r = rhs.asNum();
                if (r == 0) return error.DivisionByZero;
                lhs = Value{ .num = lhs.asNum() / r };
            } else if (self.consumeIf("%")) {
                const rhs = try self.parseUnary();
                const r = rhs.asNum();
                if (r == 0) return error.DivisionByZero;
                lhs = Value{ .num = lhs.asNum() % r };
            } else break;
        }
        return lhs;
    }

    fn parseUnary(self: *Parser) EvalError!Value {
        if (self.consumeIf("!")) {
            const inner = try self.parseUnary();
            return Value{ .bool_ = !inner.asBool() };
        }
        if (self.consumeIf("-")) {
            const inner = try self.parseUnary();
            return Value{ .num = 0 -% inner.asNum() };
        }
        return self.parsePrimary();
    }

    fn parsePrimary(self: *Parser) EvalError!Value {
        self.skipWs();
        if (self.consumeIf("(")) {
            const inner = try self.parseExpr();
            if (!self.consumeIf(")")) return error.ParseError;
            return inner;
        }
        const c = self.peek() orelse return error.ParseError;
        if (c >= '0' and c <= '9') return self.parseNumber();
        if ((c >= 'a' and c <= 'z') or (c >= 'A' and c <= 'Z') or c == '_') return self.parseIdentOrBool();
        return error.ParseError;
    }

    fn parseNumber(self: *Parser) EvalError!Value {
        const start = self.pos;
        if (self.pos + 2 <= self.src.len and self.src[self.pos] == '0' and (self.src[self.pos + 1] == 'x' or self.src[self.pos + 1] == 'X')) {
            self.pos += 2;
            const hex_start = self.pos;
            while (self.pos < self.src.len and isHexDigit(self.src[self.pos])) self.pos += 1;
            if (self.pos == hex_start) return error.ParseError;
            const value = parseHex(self.src[hex_start..self.pos]) catch return error.Overflow;
            self.skipWs();
            return Value{ .num = value };
        }
        while (self.pos < self.src.len and self.src[self.pos] >= '0' and self.src[self.pos] <= '9') self.pos += 1;
        const value = std.fmt.parseUnsigned(u256, self.src[start..self.pos], 10) catch return error.Overflow;
        self.skipWs();
        return Value{ .num = value };
    }

    fn parseIdentOrBool(self: *Parser) EvalError!Value {
        const start = self.pos;
        while (self.pos < self.src.len) : (self.pos += 1) {
            const c = self.src[self.pos];
            const is_ident = (c >= 'a' and c <= 'z') or (c >= 'A' and c <= 'Z') or (c >= '0' and c <= '9') or c == '_';
            if (!is_ident) break;
        }
        const ident = self.src[start..self.pos];
        self.skipWs();
        if (std.mem.eql(u8, ident, "true")) return Value{ .bool_ = true };
        if (std.mem.eql(u8, ident, "false")) return Value{ .bool_ = false };
        const resolved = (try self.resolver.resolve(ident)) orelse return error.UnknownIdentifier;
        return resolved;
    }
};

fn isHexDigit(c: u8) bool {
    return (c >= '0' and c <= '9') or (c >= 'a' and c <= 'f') or (c >= 'A' and c <= 'F');
}

fn parseHex(s: []const u8) error{Overflow}!u256 {
    var value: u256 = 0;
    for (s) |c| {
        const digit: u256 = switch (c) {
            '0'...'9' => @intCast(c - '0'),
            'a'...'f' => @intCast(c - 'a' + 10),
            'A'...'F' => @intCast(c - 'A' + 10),
            else => unreachable,
        };
        const shifted = std.math.shlExact(u256, value, 4) catch return error.Overflow;
        value = shifted | digit;
    }
    return value;
}

// =============================================================================
// Tests
// =============================================================================

const NoBindings = struct {
    fn resolve(_: *anyopaque, _: []const u8) EvalError!?Value {
        return null;
    }
};

fn noBindingResolver() Resolver {
    var dummy: u8 = 0;
    return .{ .ctx = @ptrCast(&dummy), .resolveFn = NoBindings.resolve };
}

const StaticBindings = struct {
    map: *const std.StringHashMap(Value),

    fn resolve(ctx: *anyopaque, name: []const u8) EvalError!?Value {
        const self: *StaticBindings = @alignCast(@ptrCast(ctx));
        if (self.map.get(name)) |v| return v;
        return null;
    }
};

const testing = std.testing;

test "evaluator: literal arithmetic" {
    const r = noBindingResolver();
    try testing.expectEqual(@as(u256, 42), (try evaluate("42", r)).asNum());
    try testing.expectEqual(@as(u256, 0xff), (try evaluate("0xff", r)).asNum());
    try testing.expectEqual(@as(u256, 6), (try evaluate("1 + 2 + 3", r)).asNum());
    try testing.expectEqual(@as(u256, 7), (try evaluate("1 + 2 * 3", r)).asNum());
    try testing.expectEqual(@as(u256, 9), (try evaluate("(1 + 2) * 3", r)).asNum());
    try testing.expectEqual(@as(u256, 3), (try evaluate("10 - 5 - 2", r)).asNum());
    try testing.expectEqual(@as(u256, 4), (try evaluate("8 / 2", r)).asNum());
    try testing.expectEqual(@as(u256, 1), (try evaluate("7 % 3", r)).asNum());
}

test "evaluator: comparisons & booleans" {
    const r = noBindingResolver();
    try testing.expectEqual(true, (try evaluate("1 == 1", r)).asBool());
    try testing.expectEqual(false, (try evaluate("1 == 2", r)).asBool());
    try testing.expectEqual(true, (try evaluate("3 < 5", r)).asBool());
    try testing.expectEqual(true, (try evaluate("5 <= 5", r)).asBool());
    try testing.expectEqual(true, (try evaluate("5 >= 5", r)).asBool());
    try testing.expectEqual(true, (try evaluate("5 > 3", r)).asBool());
    try testing.expectEqual(true, (try evaluate("true && true", r)).asBool());
    try testing.expectEqual(false, (try evaluate("true && false", r)).asBool());
    try testing.expectEqual(true, (try evaluate("false || true", r)).asBool());
    try testing.expectEqual(true, (try evaluate("!false", r)).asBool());
    try testing.expectEqual(false, (try evaluate("!(1 < 2)", r)).asBool());
}

test "evaluator: identifier resolution" {
    const allocator = testing.allocator;
    var map = std.StringHashMap(Value).init(allocator);
    defer map.deinit();
    try map.put("x", Value{ .num = 10 });
    try map.put("y", Value{ .num = 3 });

    var bindings = StaticBindings{ .map = &map };
    const r = Resolver{ .ctx = @ptrCast(&bindings), .resolveFn = StaticBindings.resolve };

    try testing.expectEqual(@as(u256, 13), (try evaluate("x + y", r)).asNum());
    try testing.expectEqual(true, (try evaluate("x > y", r)).asBool());
    try testing.expectEqual(@as(u256, 30), (try evaluate("x * 3", r)).asNum());
    try testing.expectError(error.UnknownIdentifier, evaluate("x + z", r));
}

test "evaluator: parse errors" {
    const r = noBindingResolver();
    try testing.expectError(error.ParseError, evaluate("", r));
    try testing.expectError(error.ParseError, evaluate("1 +", r));
    try testing.expectError(error.ParseError, evaluate("(1 + 2", r));
    try testing.expectError(error.DivisionByZero, evaluate("5 / 0", r));
}
