const std = @import("std");
const semantic_tokens_response = @import("semantic_tokens_response.zig");

test "lsp semantic tokens response: wraps encoded token data" {
    var data = [_]u32{ 0, 1, 4, 8, 0 };

    const response = semantic_tokens_response.build(data[0..]);

    try std.testing.expectEqual(@as(usize, 5), response.data.len);
    try std.testing.expectEqual(@as(u32, 8), response.data[3]);
}
