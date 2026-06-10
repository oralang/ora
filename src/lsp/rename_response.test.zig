const std = @import("std");
const lsp = @import("lsp");
const ora_root = @import("ora_root");

const frontend = ora_root.lsp.frontend;
const line_index = ora_root.lsp.line_index;
const rename_response = @import("rename_response.zig");
const uri_ranges = @import("uri_ranges.zig");

const testing = std.testing;
const types = lsp.types;

test "lsp rename response: prepareRename wraps range and placeholder" {
    const source = "contract Wallet {\n    pub fn value() -> u256 { return 1; }\n}\n";
    var lines = try line_index.LineIndex.init(testing.allocator, source);
    defer lines.deinit(testing.allocator);

    var arena_state = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena_state.deinit();

    const original_name = "value";
    const maybe_result = try rename_response.buildPrepare(
        arena_state.allocator(),
        source,
        &lines,
        .utf16,
        rangeFor(source, &lines, "value"),
        original_name,
    );
    const result = maybe_result orelse return error.ExpectedPrepareRenameResult;

    switch (result) {
        .literal_1 => |prepare| {
            try testing.expectEqual(@as(u32, 1), prepare.range.start.line);
            try testing.expectEqualStrings("value", prepare.placeholder);
            try testing.expect(prepare.placeholder.ptr != original_name.ptr);
        },
        else => return error.ExpectedPrepareRenameLiteral,
    }
}

test "lsp rename response: builds text edits with one shared replacement" {
    const source = "pub fn value() -> u256 { return value(); }\n";
    var lines = try line_index.LineIndex.init(testing.allocator, source);
    defer lines.deinit(testing.allocator);
    const indexed = uri_ranges.IndexedSource{ .source = source, .line_index = &lines };
    const ranges = [_]frontend.Range{
        rangeFor(source, &lines, "value"),
        rangeForFrom(source, &lines, "value", 10),
    };

    var arena_state = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena_state.deinit();

    const edits = try rename_response.buildTextEdits(arena_state.allocator(), indexed, &ranges, "balance", .utf16);

    try testing.expectEqual(@as(usize, 2), edits.len);
    try testing.expectEqualStrings("balance", edits[0].newText);
    try testing.expectEqual(edits[0].newText.ptr, edits[1].newText.ptr);
}

test "lsp rename response: inserts workspace edit changes by uri" {
    const source = "pub fn value() -> u256 { return value(); }\n";
    var lines = try line_index.LineIndex.init(testing.allocator, source);
    defer lines.deinit(testing.allocator);
    const indexed = uri_ranges.IndexedSource{ .source = source, .line_index = &lines };
    const ranges = [_]frontend.Range{rangeFor(source, &lines, "value")};

    var arena_state = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena_state.deinit();

    var changes: lsp.parser.Map(types.DocumentUri, []const types.TextEdit) = .{};
    const edits = try rename_response.putChange(
        arena_state.allocator(),
        &changes,
        "file:///wallet.ora",
        indexed,
        &ranges,
        "balance",
        .utf16,
    );
    const edit = rename_response.workspaceEdit(changes);

    try testing.expectEqual(@as(usize, 1), edits.len);
    const map = edit.changes orelse return error.ExpectedChanges;
    const stored = map.map.get("file:///wallet.ora") orelse return error.ExpectedUriChange;
    try testing.expectEqual(@as(usize, 1), stored.len);
    try testing.expectEqualStrings("balance", stored[0].newText);
}

test "lsp rename response: empty change does not insert uri" {
    var arena_state = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena_state.deinit();

    var changes: lsp.parser.Map(types.DocumentUri, []const types.TextEdit) = .{};
    const edits = try rename_response.putChange(
        arena_state.allocator(),
        &changes,
        "file:///wallet.ora",
        null,
        &.{},
        "balance",
        .utf16,
    );

    try testing.expectEqual(@as(usize, 0), edits.len);
    try testing.expectEqual(@as(usize, 0), changes.map.count());
}

fn rangeFor(source: []const u8, lines: *const line_index.LineIndex, needle: []const u8) frontend.Range {
    const start = std.mem.indexOf(u8, source, needle).?;
    return rangeAt(source, lines, start, needle.len);
}

fn rangeForFrom(source: []const u8, lines: *const line_index.LineIndex, needle: []const u8, start_index: usize) frontend.Range {
    const start = std.mem.indexOfPos(u8, source, start_index, needle).?;
    return rangeAt(source, lines, start, needle.len);
}

fn rangeAt(source: []const u8, lines: *const line_index.LineIndex, start: usize, len: usize) frontend.Range {
    return .{
        .start = lines.offsetToPosition(source, @intCast(start), .utf8),
        .end = lines.offsetToPosition(source, @intCast(start + len), .utf8),
    };
}
