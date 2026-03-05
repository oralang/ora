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
