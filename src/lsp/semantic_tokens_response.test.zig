const std = @import("std");
const semantic_tokens_response = @import("semantic_tokens_response.zig");
const ora_root = @import("ora_root");

const semantic_tokens = ora_root.lsp.semantic_tokens;

test "lsp semantic tokens response: wraps encoded token data" {
    var data = [_]u32{ 0, 1, 4, 8, 0 };

    const response = semantic_tokens_response.build(data[0..]);

    try std.testing.expectEqual(@as(usize, 5), response.data.len);
    try std.testing.expectEqual(@as(u32, 8), response.data[3]);
}

test "lsp semantic tokens response: encodes token data and reports count" {
    const tokens = [_]semantic_tokens.SemanticToken{
        .{
            .line = 1,
            .start_char = 2,
            .length = 3,
            .kind = .function,
            .modifiers = semantic_tokens.SemanticTokenModifier.mask(.declaration),
        },
        .{
            .line = 1,
            .start_char = 8,
            .length = 5,
            .kind = .variable,
            .modifiers = 0,
        },
    };

    var arena_state = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena_state.deinit();

    const result = try semantic_tokens_response.buildWithStats(arena_state.allocator(), &tokens);

    try std.testing.expectEqual(@as(usize, 10), result.data_count);
    try std.testing.expectEqual(@as(usize, 10), result.response.data.len);
    try std.testing.expectEqual(@as(u32, 1), result.response.data[0]);
    try std.testing.expectEqual(@as(u32, 2), result.response.data[1]);
    try std.testing.expectEqual(@as(u32, 0), result.response.data[5]);
    try std.testing.expectEqual(@as(u32, 6), result.response.data[6]);
}
