const std = @import("std");
const ora_root = @import("ora_root");
const code_lens_response = @import("code_lens_response.zig");

const code_lens = ora_root.lsp.code_lens;
const line_index = ora_root.lsp.line_index;

test "lsp code lens response: maps verification lenses to protocol lenses" {
    const source = "let marker = \"é\"; pub fn verify() {}";
    var lines = try line_index.LineIndex.init(std.testing.allocator, source);
    defer lines.deinit(std.testing.allocator);

    const fn_offset = std.mem.indexOf(u8, source, "pub") orelse return error.ExpectedFunction;
    var title = [_]u8{
        '1', ' ', 'v', 'e', 'r', 'i', 'f', 'i', 'c', 'a', 't',
        'i', 'o', 'n', ' ', 'c', 'l', 'a', 'u', 's', 'e',
    };
    const lenses = [_]code_lens.VerificationLens{.{
        .range = .{
            .start = lines.offsetToPosition(source, @intCast(fn_offset), .utf8),
            .end = lines.offsetToPosition(source, @intCast(source.len), .utf8),
        },
        .title = title[0..],
    }};

    var arena_state = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena_state.deinit();

    const response = (try code_lens_response.build(arena_state.allocator(), source, &lines, .utf16, &lenses)) orelse return error.ExpectedCodeLens;
    const expected_start = lines.offsetToPosition(source, @intCast(fn_offset), .utf16);

    try std.testing.expectEqual(@as(usize, 1), response.len);
    try std.testing.expectEqual(expected_start.character, response[0].range.start.character);
    const command = response[0].command orelse return error.ExpectedCommand;
    try std.testing.expectEqualStrings("1 verification clause", command.title);
    try std.testing.expectEqualStrings("ora.verify", command.command);
}

test "lsp code lens response: empty input returns null" {
    const source = "pub fn run() {}";
    var lines = try line_index.LineIndex.init(std.testing.allocator, source);
    defer lines.deinit(std.testing.allocator);

    var arena_state = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena_state.deinit();

    try std.testing.expect((try code_lens_response.build(
        arena_state.allocator(),
        source,
        &lines,
        .utf16,
        &[_]code_lens.VerificationLens{},
    )) == null);
}
