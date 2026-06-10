const std = @import("std");
const ora_root = @import("ora_root");

const allocation_stats = ora_root.lsp.allocation_stats;

test "lsp allocation stats tracks allocator activity" {
    var tracker = allocation_stats.CountingAllocator.init(std.testing.allocator);
    const allocator = tracker.allocator();

    const first = try allocator.alloc(u8, 32);
    try std.testing.expectEqual(@as(usize, 1), tracker.stats.alloc_calls);
    try std.testing.expectEqual(@as(usize, 32), tracker.stats.bytes_allocated);
    try std.testing.expectEqual(@as(usize, 32), tracker.stats.bytes_live);
    try std.testing.expectEqual(@as(usize, 32), tracker.stats.bytes_peak);
    try std.testing.expectEqual(@as(usize, 1), tracker.stats.scope(.unscoped).alloc_calls);
    try std.testing.expectEqual(@as(usize, 32), tracker.stats.scope(.unscoped).bytes_allocated);

    {
        const scope = tracker.beginScope(.response);
        defer scope.deinit();
        const scoped = try allocator.alloc(u8, 8);
        errdefer allocator.free(scoped);

        try std.testing.expectEqual(@as(usize, 2), tracker.stats.alloc_calls);
        try std.testing.expectEqual(@as(usize, 40), tracker.stats.bytes_allocated);
        try std.testing.expectEqual(@as(usize, 40), tracker.stats.bytes_live);
        try std.testing.expectEqual(@as(usize, 40), tracker.stats.bytes_peak);
        try std.testing.expectEqual(@as(usize, 1), tracker.stats.scope(.response).alloc_calls);
        try std.testing.expectEqual(@as(usize, 8), tracker.stats.scope(.response).bytes_allocated);
        allocator.free(scoped);
    }
    try std.testing.expectEqual(allocation_stats.Scope.unscoped, tracker.current_scope);

    const second = try allocator.alloc(u8, 8);
    try std.testing.expectEqual(@as(usize, 3), tracker.stats.alloc_calls);
    try std.testing.expectEqual(@as(usize, 48), tracker.stats.bytes_allocated);
    try std.testing.expectEqual(@as(usize, 40), tracker.stats.bytes_live);
    try std.testing.expectEqual(@as(usize, 40), tracker.stats.bytes_peak);

    tracker.setScope(.request_protocol);
    const third = try allocator.alloc(u8, 4);
    try std.testing.expectEqual(@as(usize, 1), tracker.stats.scope(.request_protocol).alloc_calls);
    try std.testing.expectEqual(@as(usize, 4), tracker.stats.scope(.request_protocol).bytes_allocated);

    allocator.free(first);
    try std.testing.expectEqual(@as(usize, 2), tracker.stats.free_calls);
    try std.testing.expectEqual(@as(usize, 40), tracker.stats.bytes_freed);
    try std.testing.expectEqual(@as(usize, 12), tracker.stats.bytes_live);
    try std.testing.expectEqual(@as(usize, 44), tracker.stats.bytes_peak);

    allocator.free(second);
    allocator.free(third);
    try std.testing.expectEqual(@as(usize, 4), tracker.stats.free_calls);
    try std.testing.expectEqual(@as(usize, 52), tracker.stats.bytes_freed);
    try std.testing.expectEqual(@as(usize, 0), tracker.stats.bytes_live);
}
