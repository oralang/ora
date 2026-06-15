const std = @import("std");
const ora_root = @import("ora_root");

const compiler = ora_root.compiler;
const line_index = ora_root.lsp.line_index;

test "line index converts ascii positions and ranges" {
    const source = "abc\ndef\n";
    var index = try line_index.LineIndex.init(std.testing.allocator, source);
    defer index.deinit(std.testing.allocator);

    try std.testing.expectEqual(@as(u32, 0), index.positionToOffset(source, 0, 0, .utf16).?);
    try std.testing.expectEqual(@as(u32, 6), index.positionToOffset(source, 1, 2, .utf16).?);
    try std.testing.expectEqual(@as(u32, 8), index.positionToOffset(source, 2, 0, .utf16).?);
    try std.testing.expectEqual(@as(?u32, null), index.positionToOffset(source, 3, 0, .utf16));

    try std.testing.expectEqualDeep(
        ora_root.lsp.frontend.Position{ .line = 1, .character = 2 },
        index.offsetToPosition(source, 6, .utf16),
    );
    try std.testing.expectEqualDeep(
        ora_root.lsp.frontend.Range{
            .start = .{ .line = 0, .character = 1 },
            .end = .{ .line = 1, .character = 2 },
        },
        index.textRangeToRange(source, compiler.TextRange{ .start = 1, .end = 6 }, .utf16),
    );
}

test "line index respects negotiated UTF encodings" {
    const source = "aé😀\n";
    var index = try line_index.LineIndex.init(std.testing.allocator, source);
    defer index.deinit(std.testing.allocator);

    try std.testing.expectEqual(@as(u32, 1), index.positionToOffset(source, 0, 1, .utf8).?);
    try std.testing.expectEqual(@as(u32, 3), index.positionToOffset(source, 0, 3, .utf8).?);
    try std.testing.expectEqual(@as(u32, 7), index.positionToOffset(source, 0, 7, .utf8).?);

    try std.testing.expectEqual(@as(u32, 1), index.positionToOffset(source, 0, 1, .utf16).?);
    try std.testing.expectEqual(@as(u32, 3), index.positionToOffset(source, 0, 2, .utf16).?);
    try std.testing.expectEqual(@as(?u32, null), index.positionToOffset(source, 0, 3, .utf16));
    try std.testing.expectEqual(@as(u32, 7), index.positionToOffset(source, 0, 4, .utf16).?);

    try std.testing.expectEqual(@as(u32, 7), index.positionToOffset(source, 0, 3, .utf32).?);
    try std.testing.expectEqualDeep(
        ora_root.lsp.frontend.Position{ .line = 0, .character = 4 },
        index.offsetToPosition(source, 7, .utf16),
    );
    try std.testing.expectEqualDeep(
        ora_root.lsp.frontend.Position{ .line = 0, .character = 3 },
        index.offsetToPosition(source, 7, .utf32),
    );
}

test "line index handles CRLF and EOF edge cases" {
    const source = "a\r\nb\n";
    var index = try line_index.LineIndex.init(std.testing.allocator, source);
    defer index.deinit(std.testing.allocator);

    try std.testing.expectEqual(@as(u32, 2), index.positionToOffset(source, 0, 2, .utf8).?);
    try std.testing.expectEqual(@as(?u32, null), index.positionToOffset(source, 0, 3, .utf8));
    try std.testing.expectEqual(@as(u32, 4), index.positionToOffset(source, 1, 1, .utf8).?);
    try std.testing.expectEqual(@as(u32, 5), index.positionToOffset(source, 2, 0, .utf8).?);

    try std.testing.expectEqualDeep(
        ora_root.lsp.frontend.Position{ .line = 2, .character = 0 },
        index.offsetToPosition(source, @intCast(source.len), .utf8),
    );
}
