const std = @import("std");
const testing = std.testing;
const ora_root = @import("ora_root");

const hover = ora_root.lsp.hover;

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
