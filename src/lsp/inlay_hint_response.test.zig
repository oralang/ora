const std = @import("std");
const lsp = @import("lsp");
const ora_root = @import("ora_root");
const inlay_hint_response = @import("inlay_hint_response.zig");

const frontend = ora_root.lsp.frontend;
const inlay_hints = ora_root.lsp.inlay_hints;
const line_index = ora_root.lsp.line_index;
const types = lsp.types;

test "lsp inlay hint response: maps hints and preserves negotiated positions" {
    const source = "let marker = \"é\"; amount";
    var lines = try line_index.LineIndex.init(std.testing.allocator, source);
    defer lines.deinit(std.testing.allocator);

    const amount_offset = std.mem.indexOf(u8, source, "amount") orelse return error.ExpectedAmount;
    var label = [_]u8{ 'v', 'a', 'l', 'u', 'e', ':' };
    const position = lines.offsetToPosition(source, @intCast(amount_offset), .utf16);
    const hints = [_]inlay_hints.InlayHint{.{
        .position = position,
        .label = label[0..],
        .kind = .parameter_hint,
        .padding_left = true,
        .padding_right = false,
    }};

    var arena_state = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena_state.deinit();

    const result = try inlay_hint_response.buildWithStats(arena_state.allocator(), &hints);
    const response = result.items orelse return error.ExpectedInlayHints;

    try std.testing.expectEqual(@as(usize, 1), response.len);
    try std.testing.expectEqual(position.character, response[0].position.character);
    switch (response[0].label) {
        .string => |value| try std.testing.expectEqualStrings("value:", value),
        else => return error.ExpectedStringLabel,
    }
    try std.testing.expectEqual(types.InlayHintKind.Parameter, response[0].kind.?);
    try std.testing.expectEqual(true, response[0].paddingLeft.?);
    try std.testing.expectEqual(false, response[0].paddingRight.?);
    try std.testing.expectEqual(@as(usize, "value:".len), result.string_bytes);
}

test "lsp inlay hint response: empty input returns null" {
    var arena_state = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena_state.deinit();

    try std.testing.expect((try inlay_hint_response.build(
        arena_state.allocator(),
        &[_]inlay_hints.InlayHint{},
    )) == null);
}

test "lsp inlay hint response: owns labels" {
    var label = [_]u8{ 't', 'y', 'p', 'e' };
    const hints = [_]inlay_hints.InlayHint{.{
        .position = frontend.Position{ .line = 1, .character = 2 },
        .label = label[0..],
        .kind = .type_hint,
        .padding_left = false,
        .padding_right = true,
    }};

    var arena_state = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena_state.deinit();

    const response = (try inlay_hint_response.build(arena_state.allocator(), &hints)) orelse return error.ExpectedInlayHints;
    label = [_]u8{ 'b', 'a', 'd', '!' };

    switch (response[0].label) {
        .string => |value| try std.testing.expectEqualStrings("type", value),
        else => return error.ExpectedStringLabel,
    }
}
