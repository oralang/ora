const std = @import("std");
const ora_root = @import("ora_root");
const diagnostics_response = @import("diagnostics_response.zig");

const diagnostics = ora_root.lsp.diagnostics;
const frontend = ora_root.lsp.frontend;
const line_index = ora_root.lsp.line_index;

test "lsp diagnostics response: empty cache publishes empty diagnostics" {
    var arena_state = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena_state.deinit();

    const params = try diagnostics_response.buildPublishParams(
        arena_state.allocator(),
        "file:///empty.ora",
        "",
        null,
        .utf16,
        null,
    );

    try std.testing.expectEqualStrings("file:///empty.ora", params.uri);
    try std.testing.expectEqual(@as(usize, 0), params.diagnostics.len);
}

test "lsp diagnostics response: maps cached diagnostics and owns messages" {
    const source = "let marker = \"é\"; let amount = 1;";
    var lines = try line_index.LineIndex.init(std.testing.allocator, source);
    defer lines.deinit(std.testing.allocator);

    const amount_offset = std.mem.indexOf(u8, source, "amount") orelse return error.ExpectedAmount;
    const expected_start = lines.offsetToPosition(source, @intCast(amount_offset), .utf16);
    const expected_end = lines.offsetToPosition(source, @intCast(amount_offset + "amount".len), .utf16);
    var message = [_]u8{ 'b', 'a', 'd', ' ', 'n', 'a', 'm', 'e' };
    var cached_diagnostics = [_]diagnostics.Diagnostic{.{
        .source = .sema,
        .severity = .warning,
        .range = frontend.Range{
            .start = lines.offsetToPosition(source, @intCast(amount_offset), .utf8),
            .end = lines.offsetToPosition(source, @intCast(amount_offset + "amount".len), .utf8),
        },
        .message = message[0..],
    }};
    const cache_entry = diagnostics.CacheEntry{
        .version = 1,
        .generation = 1,
        .depth = .full,
        .diagnostics = cached_diagnostics[0..],
    };

    var arena_state = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena_state.deinit();

    const params = try diagnostics_response.buildPublishParams(
        arena_state.allocator(),
        "file:///diagnostic.ora",
        source,
        &lines,
        .utf16,
        &cache_entry,
    );
    message = [_]u8{ 'm', 'u', 't', 'a', 't', 'e', 'd', '!' };

    try std.testing.expectEqual(@as(usize, 1), params.diagnostics.len);
    try std.testing.expectEqualStrings("file:///diagnostic.ora", params.uri);
    try std.testing.expectEqual(expected_start.character, params.diagnostics[0].range.start.character);
    try std.testing.expectEqual(expected_end.character, params.diagnostics[0].range.end.character);
    try std.testing.expectEqual(.Warning, params.diagnostics[0].severity.?);
    try std.testing.expectEqualStrings("ora-sema", params.diagnostics[0].source.?);
    try std.testing.expectEqualStrings("bad name", params.diagnostics[0].message);
}
