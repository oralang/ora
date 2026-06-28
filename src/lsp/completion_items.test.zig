const std = @import("std");
const lsp = @import("lsp");
const ora_root = @import("ora_root");
const completion_items = @import("completion_items.zig");
const test_analysis = @import("test_analysis.zig");

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

test "lsp completion items: owns semantic strings and reports byte totals" {
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

    const result = try completion_items.buildWithStats(
        arena_state.allocator(),
        "let value = 1;",
        position(0, 4),
        ".",
        &items,
    );

    label[0] = 'x';
    detail[0] = 'x';
    doc[0] = 'x';

    try std.testing.expectEqual(@as(usize, 1), result.items.len);
    try std.testing.expectEqualStrings("return", result.items[0].label);
    try std.testing.expectEqualStrings("keyword", result.items[0].detail.?);
    switch (result.items[0].documentation orelse return error.ExpectedDocumentation) {
        .MarkupContent => |markup| try std.testing.expectEqualStrings("Returns", markup.value),
        else => return error.ExpectedMarkupDocumentation,
    }
    try std.testing.expectEqual(@as(usize, "return".len + "keyword".len), result.string_bytes);
    try std.testing.expectEqual(@as(usize, "Returns".len), result.markdown_bytes);
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

test "lsp completion items: includes snippet strings in byte totals" {
    var arena_state = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena_state.deinit();

    const result = try completion_items.buildWithStats(
        arena_state.allocator(),
        "    ",
        position(0, 4),
        null,
        &.{},
    );

    try std.testing.expect(result.items.len > 0);
    try std.testing.expect(result.string_bytes > 0);
    try std.testing.expectEqual(@as(usize, 0), result.markdown_bytes);
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

test "lsp completion items: protocol path includes builtin docs and Resource type" {
    var arena_state = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena_state.deinit();

    const source =
        \\resource TokenUnit = u256;
        \\contract Vault {
        \\    storage var balance: Resource<TokenUnit>;
        \\    pub fn run(value: u64) -> u256 {
        \\        return @cast(u256, value);
        \\    }
        \\}
    ;

    var index = try test_analysis.semanticIndex(std.testing.allocator, source);
    defer index.deinit(std.testing.allocator);

    const result = try completion_items.buildFromSemanticIndexWithStats(
        arena_state.allocator(),
        source,
        position(4, 17),
        null,
        &index,
    );

    const cast = itemWithLabel(result.items, "cast") orelse return error.ExpectedCompletion;
    try std.testing.expectEqual(types.CompletionItemKind.Function, cast.kind.?);
    try std.testing.expectEqualStrings("@cast(T, value) -> T", cast.detail.?);
    switch (cast.documentation orelse return error.ExpectedDocumentation) {
        .MarkupContent => |markup| try std.testing.expect(std.mem.indexOf(u8, markup.value, "checked conversion rules") != null),
        else => return error.ExpectedMarkupDocumentation,
    }

    const all = try completion_items.buildFromSemanticIndexWithStats(
        arena_state.allocator(),
        source,
        position(4, 16),
        null,
        &index,
    );
    _ = itemWithLabel(all.items, "Resource") orelse return error.ExpectedCompletion;
    _ = itemWithLabel(all.items, "move") orelse return error.ExpectedCompletion;
    _ = itemWithLabel(all.items, "create") orelse return error.ExpectedCompletion;
    _ = itemWithLabel(all.items, "destroy") orelse return error.ExpectedCompletion;
}

fn itemWithLabel(items: []const types.CompletionItem, label: []const u8) ?types.CompletionItem {
    for (items) |item| {
        if (std.mem.eql(u8, item.label, label)) return item;
    }
    return null;
}

fn position(line: u32, character: u32) frontend.Position {
    return .{ .line = line, .character = character };
}
