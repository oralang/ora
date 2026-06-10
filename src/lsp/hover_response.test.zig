const std = @import("std");
const lsp = @import("lsp");
const ora_root = @import("ora_root");
const hover_response = @import("hover_response.zig");

const frontend = ora_root.lsp.frontend;
const hover = ora_root.lsp.hover;
const line_index = ora_root.lsp.line_index;
const types = lsp.types;

test "lsp hover response: wraps markdown contents and converts range" {
    const source = "let marker = \"é\"; let amount = 1;";
    var lines = try line_index.LineIndex.init(std.testing.allocator, source);
    defer lines.deinit(std.testing.allocator);

    const amount_offset = std.mem.indexOf(u8, source, "amount") orelse return error.ExpectedAmount;
    const item = hover.Hover{
        .contents = "```ora\nlet amount: u256\n```",
        .range = .{
            .start = lines.offsetToPosition(source, @intCast(amount_offset), .utf8),
            .end = lines.offsetToPosition(source, @intCast(amount_offset + "amount".len), .utf8),
        },
    };
    const response = hover_response.build(source, &lines, .utf16, item) orelse return error.ExpectedHover;
    const expected_start = lines.offsetToPosition(source, @intCast(amount_offset), .utf16);
    const expected_end = lines.offsetToPosition(source, @intCast(amount_offset + "amount".len), .utf16);

    switch (response.contents) {
        .MarkupContent => |markup| {
            try std.testing.expectEqual(types.MarkupKind.markdown, markup.kind);
            try std.testing.expectEqualStrings(item.contents, markup.value);
        },
        else => return error.ExpectedMarkupContent,
    }
    const range = response.range orelse return error.ExpectedRange;
    try std.testing.expectEqual(expected_start.character, range.start.character);
    try std.testing.expectEqual(expected_end.character, range.end.character);
}

test "lsp hover response: rejects ranges outside the source" {
    const source = "pub fn run() -> u256 { return 1; }";
    var lines = try line_index.LineIndex.init(std.testing.allocator, source);
    defer lines.deinit(std.testing.allocator);

    const item = hover.Hover{
        .contents = "bad",
        .range = frontend.Range{
            .start = .{ .line = 99, .character = 0 },
            .end = .{ .line = 99, .character = 3 },
        },
    };

    try std.testing.expect(hover_response.build(source, &lines, .utf16, item) == null);
}
