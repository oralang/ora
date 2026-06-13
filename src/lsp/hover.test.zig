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
