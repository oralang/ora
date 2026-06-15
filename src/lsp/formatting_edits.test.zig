const std = @import("std");
const formatting_edits = @import("formatting_edits.zig");

test "lsp formatting edits: returns no edits when text is unchanged" {
    var arena_state = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena_state.deinit();

    const edits = try formatting_edits.buildFullDocumentEdit(
        arena_state.allocator(),
        "pub fn run() {}\n",
        "pub fn run() {}\n",
        .utf16,
    );

    try std.testing.expectEqual(@as(usize, 0), edits.len);
}

test "lsp formatting edits: replaces the full document with utf16 end position" {
    const source = "let smile = \"😀\";\nlet amount = 1;";
    const formatted = "let smile = \"😀\";\nlet amount = 2;\n";

    var arena_state = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena_state.deinit();

    const edits = try formatting_edits.buildFullDocumentEdit(
        arena_state.allocator(),
        source,
        formatted,
        .utf16,
    );

    try std.testing.expectEqual(@as(usize, 1), edits.len);
    try std.testing.expectEqual(@as(u32, 0), edits[0].range.start.line);
    try std.testing.expectEqual(@as(u32, 0), edits[0].range.start.character);
    try std.testing.expectEqual(@as(u32, 1), edits[0].range.end.line);
    try std.testing.expectEqual(@as(u32, 15), edits[0].range.end.character);
    try std.testing.expectEqualStrings(formatted, edits[0].newText);
}
