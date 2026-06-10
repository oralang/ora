const std = @import("std");
const testing = std.testing;
const ora_root = @import("ora_root");

const hover = ora_root.lsp.hover;
const semantic_index = ora_root.lsp.semantic_index;

test "lsp hover: returns function signature at function name" {
    const source =
        \\contract Wallet {
        \\    pub fn deposit(amount: u256) -> u256 { return amount; }
        \\}
    ;

    const maybe_hover = try hover.hoverAt(testing.allocator, source, .{
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

    var index = try semantic_index.indexDocument(testing.allocator, source);
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

    const maybe_hover = try hover.hoverAt(testing.allocator, source, .{
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

    const maybe_hover = try hover.hoverAt(testing.allocator, source, .{
        .line = 1,
        .character = 20,
    });
    try testing.expect(maybe_hover != null);

    var value = maybe_hover.?;
    defer value.deinit(testing.allocator);

    try testing.expect(std.mem.indexOf(u8, value.contents, "amount: u256") != null);
}

test "lsp hover: returns null when no symbol is under cursor" {
    const source = "contract Wallet { }";

    const maybe_hover = try hover.hoverAt(testing.allocator, source, .{
        .line = 0,
        .character = 17,
    });
    try testing.expect(maybe_hover == null);
}

test "lsp hover: parse failure returns null hover" {
    const source = "@import(\"std\");";

    const maybe_hover = try hover.hoverAt(testing.allocator, source, .{
        .line = 0,
        .character = 2,
    });
    try testing.expect(maybe_hover == null);
}
