const std = @import("std");
const lsp = @import("lsp");
const call_hierarchy_prepare = @import("call_hierarchy_prepare.zig");
const ora_root = @import("ora_root");

const line_index = ora_root.lsp.line_index;
const semantic_index = ora_root.lsp.semantic_index;

fn lspPositionAt(source: []const u8, lines: *const line_index.LineIndex, needle: []const u8, encoding: ora_root.lsp.text_edits.PositionEncoding) !lsp.types.Position {
    const offset = std.mem.indexOf(u8, source, needle) orelse return error.ExpectedNeedle;
    const position = lines.offsetToPosition(source, @intCast(offset), encoding);
    return .{ .line = position.line, .character = position.character };
}

test "lsp call hierarchy prepare: returns callable item" {
    const source =
        \\pub fn helper(value: u256) -> u256 {
        \\    return value;
        \\}
    ;

    var index = try semantic_index.indexDocument(std.testing.allocator, source);
    defer index.deinit(std.testing.allocator);

    var lines = try line_index.LineIndex.init(std.testing.allocator, source);
    defer lines.deinit(std.testing.allocator);

    const position = try lspPositionAt(source, &lines, "helper", .utf8);

    var response_arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer response_arena.deinit();

    const items = (try call_hierarchy_prepare.build(response_arena.allocator(), "file:///helper.ora", source, &lines, .utf8, &index, position)) orelse return error.ExpectedItems;

    try std.testing.expectEqual(@as(usize, 1), items.len);
    try std.testing.expectEqualStrings("helper", items[0].name);
    try std.testing.expectEqualStrings("file:///helper.ora", items[0].uri);
}

test "lsp call hierarchy prepare: converts item ranges to utf16" {
    const source = "/* é */ pub fn helper() -> u256 { return 1; }";

    var index = try semantic_index.indexDocument(std.testing.allocator, source);
    defer index.deinit(std.testing.allocator);

    var lines = try line_index.LineIndex.init(std.testing.allocator, source);
    defer lines.deinit(std.testing.allocator);

    const position = try lspPositionAt(source, &lines, "helper", .utf16);
    const helper_offset = std.mem.indexOf(u8, source, "helper") orelse return error.ExpectedHelper;
    const expected_start = lines.offsetToPosition(source, @intCast(helper_offset), .utf16);
    const expected_end = lines.offsetToPosition(source, @intCast(helper_offset + "helper".len), .utf16);

    var response_arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer response_arena.deinit();

    const items = (try call_hierarchy_prepare.build(response_arena.allocator(), "file:///helper.ora", source, &lines, .utf16, &index, position)) orelse return error.ExpectedItems;

    try std.testing.expectEqual(expected_start.line, items[0].selectionRange.start.line);
    try std.testing.expectEqual(expected_start.character, items[0].selectionRange.start.character);
    try std.testing.expectEqual(expected_end.line, items[0].selectionRange.end.line);
    try std.testing.expectEqual(expected_end.character, items[0].selectionRange.end.character);
}

test "lsp call hierarchy prepare: returns null outside callable symbols" {
    const source =
        \\contract Wallet {
        \\    storage var balance: u256;
        \\}
    ;

    var index = try semantic_index.indexDocument(std.testing.allocator, source);
    defer index.deinit(std.testing.allocator);

    var lines = try line_index.LineIndex.init(std.testing.allocator, source);
    defer lines.deinit(std.testing.allocator);

    const position = try lspPositionAt(source, &lines, "Wallet", .utf8);

    var response_arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer response_arena.deinit();

    const items = try call_hierarchy_prepare.build(response_arena.allocator(), "file:///wallet.ora", source, &lines, .utf8, &index, position);
    try std.testing.expect(items == null);
}
