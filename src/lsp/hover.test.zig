const std = @import("std");
const testing = std.testing;
const ora_root = @import("ora_root");

const hover = ora_root.lsp.hover;
const frontend = ora_root.lsp.frontend;
const test_analysis = @import("test_analysis.zig");

fn cachedHover(source: []const u8, position: frontend.Position) !?hover.Hover {
    var index = try test_analysis.semanticIndex(testing.allocator, source);
    defer index.deinit(testing.allocator);
    return hover.hoverAtIndex(testing.allocator, source, position, &index);
}

test "lsp hover: returns function signature at function name" {
    const source =
        \\contract Wallet {
        \\    pub fn deposit(amount: u256) -> u256 { return amount; }
        \\}
    ;

    const maybe_hover = try cachedHover(source, .{
        .line = 1,
        .character = 12,
    });
    try testing.expect(maybe_hover != null);

    var value = maybe_hover.?;
    defer value.deinit(testing.allocator);

    try testing.expect(std.mem.indexOf(u8, value.contents, "fn deposit(amount: u256) -> u256") != null);
}

test "lsp hover: explains a callHint state inside the directive parens" {
    const source =
        \\contract C {
        \\    storage var x: u256;
        \\    pub fn bump() {
        \\        @callHint(cold);
        \\        x = 1;
        \\    }
        \\}
    ;

    // Position on "cold" inside the parens of @callHint(cold).
    const maybe_hover = try cachedHover(source, .{
        .line = 3,
        .character = 19,
    });
    try testing.expect(maybe_hover != null);

    var value = maybe_hover.?;
    defer value.deinit(testing.allocator);

    try testing.expect(std.mem.indexOf(u8, value.contents, "@callHint(cold)") != null);
    try testing.expect(std.mem.indexOf(u8, value.contents, "excluded from the hot-set count") != null);
}

test "lsp hover: callHint state name outside directive parens gets no state hover" {
    const source =
        \\contract C {
        \\    storage var cold: u256;
        \\    pub fn bump() {
        \\        cold = 1;
        \\    }
        \\}
    ;

    // Position on the storage variable "cold" — must not claim directive docs.
    const maybe_hover = try cachedHover(source, .{
        .line = 3,
        .character = 9,
    });
    if (maybe_hover) |*value_const| {
        var value = value_const.*;
        defer value.deinit(testing.allocator);
        try testing.expect(std.mem.indexOf(u8, value.contents, "hot-set count") == null);
    }
}

test "lsp hover: indexed path uses caller-owned semantic index" {
    const source =
        \\contract Wallet {
        \\    pub fn deposit(amount: u256) -> u256 { return amount; }
        \\}
    ;

    var index = try test_analysis.semanticIndex(testing.allocator, source);
    defer index.deinit(testing.allocator);

    const maybe_hover = try hover.hoverAtIndex(testing.allocator, source, .{
        .line = 1,
        .character = 12,
    }, &index);
    try testing.expect(maybe_hover != null);

    var value = maybe_hover.?;
    defer value.deinit(testing.allocator);

    try testing.expect(std.mem.indexOf(u8, value.contents, "fn deposit(amount: u256) -> u256") != null);
}

test "lsp hover: marks inline function signatures" {
    const source =
        \\contract Math {
        \\    inline fn choose(mode: u256) -> u256 { return mode; }
        \\    pub fn run(mode: u256) -> u256 { return choose(mode); }
        \\}
    ;

    const maybe_hover = try cachedHover(source, positionOf(source, "choose"));
    try testing.expect(maybe_hover != null);

    var value = maybe_hover.?;
    defer value.deinit(testing.allocator);

    try testing.expect(std.mem.indexOf(u8, value.contents, "inline fn choose(mode: u256) -> u256") != null);
}

test "lsp hover: uses /// docs and ignores // internal comments" {
    const source =
        \\// Internal note that must not become public docs.
        \\/// Returns `value` unchanged.
        \\pub fn helper(value: u256) -> u256 { return value; }
    ;

    const maybe_hover = try cachedHover(source, .{
        .line = 2,
        .character = 8,
    });
    try testing.expect(maybe_hover != null);

    var value = maybe_hover.?;
    defer value.deinit(testing.allocator);

    try testing.expect(std.mem.indexOf(u8, value.contents, "Returns `value` unchanged.") != null);
    try testing.expect(std.mem.indexOf(u8, value.contents, "Internal note") == null);
}

test "lsp hover: returns parameter type at parameter position" {
    const source =
        \\contract Wallet {
        \\    pub fn deposit(amount: u256) -> u256 { return amount; }
        \\}
    ;

    const maybe_hover = try cachedHover(source, .{
        .line = 1,
        .character = 20,
    });
    try testing.expect(maybe_hover != null);

    var value = maybe_hover.?;
    defer value.deinit(testing.allocator);

    try testing.expect(std.mem.indexOf(u8, value.contents, "amount: u256") != null);
}

test "lsp hover: keyword docs come from shared guarded table" {
    const source =
        \\pub fn ready(left: bool, right: bool) -> bool {
        \\    return left and right;
        \\}
    ;

    const maybe_hover = try cachedHover(source, positionOf(source, "and"));
    try testing.expect(maybe_hover != null);

    var value = maybe_hover.?;
    defer value.deinit(testing.allocator);

    try testing.expect(std.mem.indexOf(u8, value.contents, "Logical conjunction operator.") != null);
}

test "lsp hover: builtin docs resolve from at-prefixed calls" {
    const source =
        \\resource TokenUnit = u256;
        \\
        \\contract Vault {
        \\    storage var balances: map<address, Resource<TokenUnit>>;
        \\
        \\    pub fn run(to: address, amount: TokenUnit) {
        \\        let wide: u256 = @cast(u256, amount);
        \\        @create(balances[to], amount);
        \\        @move(balances[to], balances[to], amount);
        \\        @destroy(balances[to], amount);
        \\    }
        \\}
    ;

    inline for (.{
        .{ "cast", "@cast(T, value) -> T", "checked conversion rules" },
        .{ "create", "@create(place, amount) -> void", "resource-domain boundary delta" },
        .{ "move", "@move(from, to, amount) -> void", "same resource domain" },
        .{ "destroy", "@destroy(place, amount) -> void", "resource place" },
    }) |case| {
        const maybe_hover = try cachedHover(source, positionOf(source, case[0]));
        try testing.expect(maybe_hover != null);

        var value = maybe_hover.?;
        defer value.deinit(testing.allocator);

        try testing.expect(std.mem.indexOf(u8, value.contents, case[1]) != null);
        try testing.expect(std.mem.indexOf(u8, value.contents, case[2]) != null);
    }
}

test "lsp hover: builtin docs resolve when cursor is on at sign" {
    const source =
        \\pub fn run(value: u64) -> u256 {
        \\    return @cast(u256, value);
        \\}
    ;

    const at_offset = std.mem.indexOf(u8, source, "@cast") orelse return error.TestExpectedEqual;
    const maybe_hover = try cachedHover(source, byteIndexToPosition(source, at_offset));
    try testing.expect(maybe_hover != null);

    var value = maybe_hover.?;
    defer value.deinit(testing.allocator);

    try testing.expect(std.mem.indexOf(u8, value.contents, "@cast(T, value) -> T") != null);
}

test "lsp hover: Resource generic capability type is documented" {
    const source =
        \\resource TokenUnit = u256;
        \\contract Vault {
        \\    storage var balance: Resource<TokenUnit>;
        \\}
    ;

    const maybe_hover = try cachedHover(source, positionOf(source, "Resource"));
    try testing.expect(maybe_hover != null);

    var value = maybe_hover.?;
    defer value.deinit(testing.allocator);

    try testing.expect(std.mem.indexOf(u8, value.contents, "Resource<T>") != null);
    try testing.expect(std.mem.indexOf(u8, value.contents, "@create") != null);
}

test "lsp hover: returns registry-backed refinement docs" {
    const source =
        \\pub fn ready(owner: NonZeroAddress) -> NonZeroAddress {
        \\    return owner;
        \\}
    ;

    const maybe_hover = try cachedHover(source, positionOf(source, "NonZeroAddress"));
    try testing.expect(maybe_hover != null);

    var value = maybe_hover.?;
    defer value.deinit(testing.allocator);

    try testing.expect(std.mem.indexOf(u8, value.contents, "NonZeroAddress") != null);
    try testing.expect(std.mem.indexOf(u8, value.contents, "exclude the zero address") != null);
}

test "lsp hover: returns null when no symbol is under cursor" {
    const source = "contract Wallet { }";

    const maybe_hover = try cachedHover(source, .{
        .line = 0,
        .character = 17,
    });
    try testing.expect(maybe_hover == null);
}

fn positionOf(source: []const u8, needle: []const u8) frontend.Position {
    const offset = std.mem.indexOf(u8, source, needle) orelse @panic("needle not found");
    return byteIndexToPosition(source, offset);
}

fn byteIndexToPosition(source: []const u8, offset: usize) frontend.Position {
    var line: u32 = 0;
    var character: u32 = 0;
    for (source[0..offset]) |byte| {
        if (byte == '\n') {
            line += 1;
            character = 0;
        } else {
            character += 1;
        }
    }
    return .{ .line = line, .character = character };
}

test "lsp hover: parse failure returns null hover" {
    const source = "@import(\"std\");";

    const maybe_hover = try cachedHover(source, .{
        .line = 0,
        .character = 2,
    });
    try testing.expect(maybe_hover == null);
}
