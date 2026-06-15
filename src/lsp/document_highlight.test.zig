const std = @import("std");
const lsp = @import("lsp");
const document_highlight = @import("document_highlight.zig");
const ora_root = @import("ora_root");

const frontend = ora_root.lsp.frontend;
const line_index = ora_root.lsp.line_index;
const references = ora_root.lsp.references;
const token_cache = ora_root.lsp.token_cache;
const test_analysis = @import("test_analysis.zig");

fn positionOfNth(source: []const u8, needle: []const u8, nth: usize) !frontend.Position {
    var from: usize = 0;
    var seen: usize = 0;
    while (true) {
        const rel = std.mem.indexOfPos(u8, source, from, needle) orelse return error.TestExpectedEqual;
        if (seen == nth) return byteIndexToPosition(source, rel);
        seen += 1;
        from = rel + needle.len;
    }
}

fn byteIndexToPosition(source: []const u8, byte_index: usize) frontend.Position {
    var line: u32 = 0;
    var character: u32 = 0;
    for (source[0..byte_index]) |byte| {
        if (byte == '\n') {
            line += 1;
            character = 0;
        } else {
            character += 1;
        }
    }
    return .{ .line = line, .character = character };
}

test "lsp document highlight: builds highlights from occurrence index" {
    const source =
        \\pub fn run() -> u256 {
        \\    let amount: u256 = 1;
        \\    return amount + amount;
        \\}
    ;

    var fixture: test_analysis.TestAnalysis = undefined;
    try fixture.init(std.testing.allocator, source);
    defer fixture.deinit();

    var cache = try token_cache.Cache.init(std.testing.allocator, source);
    defer cache.deinit(std.testing.allocator);

    var index = try references.OccurrenceIndex.init(std.testing.allocator, source, cache.tokens, &fixture.analysis);
    defer index.deinit(std.testing.allocator);

    const query = try positionOfNth(source, "amount", 2);

    var response_arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer response_arena.deinit();

    var lines = try line_index.LineIndex.init(std.testing.allocator, source);
    defer lines.deinit(std.testing.allocator);

    const highlights = (try document_highlight.build(response_arena.allocator(), source, &lines, .utf8, &index, query)).?;

    try std.testing.expectEqual(@as(usize, 3), highlights.len);
    for (highlights) |highlight| {
        try std.testing.expectEqual(@as(?lsp.types.DocumentHighlightKind, .Text), highlight.kind);
    }
}

test "lsp document highlight: converts byte ranges to utf16" {
    const source =
        \\pub fn run() -> u256 {
        \\    let marker = "é"; let amount: u256 = 1;
        \\    return amount + amount;
        \\}
    ;

    var fixture: test_analysis.TestAnalysis = undefined;
    try fixture.init(std.testing.allocator, source);
    defer fixture.deinit();

    var cache = try token_cache.Cache.init(std.testing.allocator, source);
    defer cache.deinit(std.testing.allocator);

    var index = try references.OccurrenceIndex.init(std.testing.allocator, source, cache.tokens, &fixture.analysis);
    defer index.deinit(std.testing.allocator);

    var lines = try line_index.LineIndex.init(std.testing.allocator, source);
    defer lines.deinit(std.testing.allocator);

    const query = try positionOfNth(source, "amount", 1);
    const declaration_offset = std.mem.indexOf(u8, source, "amount") orelse return error.ExpectedAmount;
    const expected_start = lines.offsetToPosition(source, @intCast(declaration_offset), .utf16);
    const expected_end = lines.offsetToPosition(source, @intCast(declaration_offset + "amount".len), .utf16);

    var response_arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer response_arena.deinit();

    const highlights = (try document_highlight.build(response_arena.allocator(), source, &lines, .utf16, &index, query)).?;

    var found_declaration = false;
    for (highlights) |highlight| {
        if (highlight.range.start.line == expected_start.line and
            highlight.range.start.character == expected_start.character and
            highlight.range.end.line == expected_end.line and
            highlight.range.end.character == expected_end.character)
        {
            found_declaration = true;
        }
    }
    try std.testing.expect(found_declaration);
}
