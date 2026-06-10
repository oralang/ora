const std = @import("std");
const lsp = @import("lsp");
const ora_root = @import("ora_root");
const completion_items = @import("completion_items.zig");

const completion = ora_root.lsp.completion;
const frontend = ora_root.lsp.frontend;

const types = lsp.types;

test "lsp completion items: maps semantic items to protocol items" {
    var label = [_]u8{ 'r', 'e', 't', 'u', 'r', 'n' };
    var detail = [_]u8{ 'k', 'e', 'y', 'w', 'o', 'r', 'd' };
    var doc = [_]u8{ 'R', 'e', 't', 'u', 'r', 'n', 's' };
    const items = [_]completion.Item{.{
        .label = label[0..],
        .detail = detail[0..],
        .documentation = doc[0..],
        .kind = .keyword,
    }};

    var arena_state = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena_state.deinit();

    const result = try completion_items.build(
        arena_state.allocator(),
        "let value = 1;",
        position(0, 4),
        ".",
        &items,
    );

    try std.testing.expectEqual(@as(usize, 1), result.len);
    try std.testing.expectEqualStrings("return", result[0].label);
    try std.testing.expectEqual(types.CompletionItemKind.Keyword, result[0].kind.?);
    try std.testing.expectEqualStrings("keyword", result[0].detail.?);
    const documentation = result[0].documentation orelse return error.ExpectedDocumentation;
    switch (documentation) {
        .MarkupContent => |markup| {
            try std.testing.expectEqual(types.MarkupKind.markdown, markup.kind);
            try std.testing.expectEqualStrings("Returns", markup.value);
        },
        else => return error.ExpectedMarkupDocumentation,
    }
}

test "lsp completion items: adds snippets at line start without trigger" {
    var arena_state = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena_state.deinit();

    const result = try completion_items.build(
        arena_state.allocator(),
        "    ",
        position(0, 4),
        null,
        &.{},
    );

    try std.testing.expect(result.len > 0);
    try std.testing.expectEqualStrings("contract", result[0].label);
    try std.testing.expectEqual(types.CompletionItemKind.Snippet, result[0].kind.?);
    try std.testing.expectEqual(types.InsertTextFormat.Snippet, result[0].insertTextFormat.?);
}

test "lsp completion items: suppresses snippets away from line start or with trigger" {
    var arena_state = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena_state.deinit();

    const non_line_start = try completion_items.build(
        arena_state.allocator(),
        "let value",
        position(0, 8),
        null,
        &.{},
    );
    try std.testing.expectEqual(@as(usize, 0), non_line_start.len);

    const triggered = try completion_items.build(
        arena_state.allocator(),
        "    ",
        position(0, 4),
        ".",
        &.{},
    );
    try std.testing.expectEqual(@as(usize, 0), triggered.len);
}

fn position(line: u32, character: u32) frontend.Position {
    return .{ .line = line, .character = character };
}
