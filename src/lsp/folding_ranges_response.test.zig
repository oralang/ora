const std = @import("std");
const ora_root = @import("ora_root");
const folding_ranges_response = @import("folding_ranges_response.zig");

const folding = ora_root.lsp.folding;

test "lsp folding ranges response: maps folding ranges to protocol ranges" {
    const ranges = [_]folding.FoldingRange{
        .{ .start_line = 0, .end_line = 3, .kind = .region },
        .{ .start_line = 5, .end_line = 8, .kind = .comment },
        .{ .start_line = 10, .end_line = 12, .kind = .imports },
    };

    var arena_state = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena_state.deinit();

    const response = (try folding_ranges_response.build(arena_state.allocator(), &ranges)) orelse return error.ExpectedFoldingRanges;

    try std.testing.expectEqual(@as(usize, 3), response.len);
    try std.testing.expectEqual(@as(u32, 0), response[0].startLine);
    try std.testing.expectEqual(@as(u32, 3), response[0].endLine);
    try std.testing.expectEqual(.region, response[0].kind.?);
    try std.testing.expectEqual(.comment, response[1].kind.?);
    try std.testing.expectEqual(.imports, response[2].kind.?);
}

test "lsp folding ranges response: empty input returns null" {
    var arena_state = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena_state.deinit();

    try std.testing.expect((try folding_ranges_response.build(
        arena_state.allocator(),
        &[_]folding.FoldingRange{},
    )) == null);
}
