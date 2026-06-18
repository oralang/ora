const std = @import("std");
const testing = std.testing;

const formatting = @import("formatting.zig");

test "lsp formatting: formats and is idempotent" {
    const source =
        \\contract Wallet{
        \\pub fn run()->u256{return 1;}
        \\}
    ;

    const formatted_once = try formatting.formatSourceAlloc(testing.allocator, source, .{
        .line_width = 100,
        .indent_size = 4,
    });
    defer testing.allocator.free(formatted_once);

    try testing.expect(!std.mem.eql(u8, source, formatted_once));

    const formatted_twice = try formatting.formatSourceAlloc(testing.allocator, formatted_once, .{
        .line_width = 100,
        .indent_size = 4,
    });
    defer testing.allocator.free(formatted_twice);

    try testing.expectEqualStrings(formatted_once, formatted_twice);
}

test "lsp formatting: parse errors surface as ParseError" {
    const invalid = "@import(\"std\");";
    try testing.expectError(
        error.ParseError,
        formatting.formatSourceAlloc(testing.allocator, invalid, .{}),
    );
}

test "lsp formatting: preserves type alias spacing" {
    const source =
        \\type Amount = u256;
        \\type AmountArray4 = [Amount; 4];
        \\type AliasArray(comptime T: type, comptime N: u256) = [T; N];
    ;

    const formatted = try formatting.formatSourceAlloc(testing.allocator, source, .{
        .line_width = 100,
        .indent_size = 4,
    });
    defer testing.allocator.free(formatted);

    try testing.expectEqualStrings(source ++ "\n", formatted);
}

test "lsp formatting: preserves keyword boundaries for inline impl and modifies" {
    const source =
        \\trait SettledBalance {
        \\    fn settled(self: Settled) -> u256;
        \\}
        \\
        \\struct Settled {
        \\    value: u256;
        \\}
        \\
        \\impl SettledBalance for Settled {
        \\    fn settled(self: Settled) -> u256 {
        \\        return self.value;
        \\    }
        \\}
        \\
        \\contract Token {
        \\    inline fn amountDelta(amount: u256) -> u256 {
        \\        return amount;
        \\    }
        \\
        \\    pub fn approve(spender: address, amount: u256)
        \\        modifies allowances[msg.sender][spender]
        \\    {
        \\        return;
        \\    }
        \\}
    ;

    const formatted_once = try formatting.formatSourceAlloc(testing.allocator, source, .{
        .line_width = 100,
        .indent_size = 4,
    });
    defer testing.allocator.free(formatted_once);

    const formatted_twice = try formatting.formatSourceAlloc(testing.allocator, formatted_once, .{
        .line_width = 100,
        .indent_size = 4,
    });
    defer testing.allocator.free(formatted_twice);

    try testing.expectEqualStrings(formatted_once, formatted_twice);
    try testing.expect(std.mem.indexOf(u8, formatted_once, "inline fn amountDelta") != null);
    try testing.expect(std.mem.indexOf(u8, formatted_once, "impl SettledBalance for Settled") != null);
    try testing.expect(std.mem.indexOf(u8, formatted_once, "modifies allowances[msg.sender][spender]") != null);
    try testing.expect(std.mem.indexOf(u8, formatted_twice, "forfor") == null);
    try testing.expect(std.mem.indexOf(u8, formatted_twice, "inlinefn") == null);
    try testing.expect(std.mem.indexOf(u8, formatted_twice, "modifiesallowances") == null);

    const corrupted =
        \\contract Token {
        \\    pub fn approve(spender: address, amount: u256)
        \\    modifiesallowances[msg.sender][spender]
        \\    {
        \\        return;
        \\    }
        \\}
    ;

    const repaired = try formatting.formatSourceAlloc(testing.allocator, corrupted, .{
        .line_width = 100,
        .indent_size = 4,
    });
    defer testing.allocator.free(repaired);

    try testing.expect(std.mem.indexOf(u8, repaired, "modifies allowances[msg.sender][spender]") != null);
    try testing.expect(std.mem.indexOf(u8, repaired, "modifiesallowances") == null);
}
