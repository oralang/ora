const std = @import("std");
const lsp = @import("lsp");
const selection_range = @import("selection_range.zig");
const ora_root = @import("ora_root");

const line_index = ora_root.lsp.line_index;
const test_analysis = @import("test_analysis.zig");

fn lspPositionAt(source: []const u8, lines: *const line_index.LineIndex, needle: []const u8, encoding: ora_root.lsp.text_edits.PositionEncoding) !lsp.types.Position {
    const offset = std.mem.indexOf(u8, source, needle) orelse return error.ExpectedNeedle;
    const position = lines.offsetToPosition(source, @intCast(offset), encoding);
    return .{ .line = position.line, .character = position.character };
}

test "lsp selection range: builds nested statement parents" {
    const source =
        \\pub fn run(flag: bool) -> u256 {
        \\    if (flag) {
        \\        return 1;
        \\    }
        \\    return 0;
        \\}
    ;

    var fixture: test_analysis.TestAnalysis = undefined;
    try fixture.init(std.testing.allocator, source);
    defer fixture.deinit();

    var lines = try line_index.LineIndex.init(std.testing.allocator, source);
    defer lines.deinit(std.testing.allocator);

    const positions = [_]lsp.types.Position{try lspPositionAt(source, &lines, "return 1", .utf8)};

    var response_arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer response_arena.deinit();

    const ranges = try selection_range.build(response_arena.allocator(), source, &lines, .utf8, fixture.analysis.ast_file, &positions);

    try std.testing.expectEqual(@as(usize, 1), ranges.len);
    try std.testing.expectEqual(@as(u32, 2), ranges[0].range.start.line);

    const if_parent = ranges[0].parent orelse return error.ExpectedIfParent;
    try std.testing.expectEqual(@as(u32, 1), if_parent.range.start.line);

    const fn_parent = if_parent.parent orelse return error.ExpectedFunctionParent;
    try std.testing.expectEqual(@as(u32, 0), fn_parent.range.start.line);
    try std.testing.expect(fn_parent.parent == null);
}

test "lsp selection range: converts parent ranges to utf16" {
    const source =
        \\/* é */ pub fn run() -> u256 {
        \\    return 1;
        \\}
    ;

    var fixture: test_analysis.TestAnalysis = undefined;
    try fixture.init(std.testing.allocator, source);
    defer fixture.deinit();

    var lines = try line_index.LineIndex.init(std.testing.allocator, source);
    defer lines.deinit(std.testing.allocator);

    const positions = [_]lsp.types.Position{try lspPositionAt(source, &lines, "return 1", .utf16)};

    var response_arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer response_arena.deinit();

    const ranges = try selection_range.build(response_arena.allocator(), source, &lines, .utf16, fixture.analysis.ast_file, &positions);

    const pub_offset = std.mem.indexOf(u8, source, "pub fn") orelse return error.ExpectedFunction;
    const expected_start = lines.offsetToPosition(source, @intCast(pub_offset), .utf16);

    const fn_parent = ranges[0].parent orelse return error.ExpectedFunctionParent;
    try std.testing.expectEqual(expected_start.line, fn_parent.range.start.line);
    try std.testing.expectEqual(expected_start.character, fn_parent.range.start.character);
}

test "lsp selection range: returns zero range outside selectable ast ranges" {
    const source =
        \\pub fn run() -> u256 {
        \\    return 1;
        \\}
        \\// trailing comment
    ;

    var fixture: test_analysis.TestAnalysis = undefined;
    try fixture.init(std.testing.allocator, source);
    defer fixture.deinit();

    var lines = try line_index.LineIndex.init(std.testing.allocator, source);
    defer lines.deinit(std.testing.allocator);

    const positions = [_]lsp.types.Position{.{ .line = 3, .character = 2 }};

    var response_arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer response_arena.deinit();

    const ranges = try selection_range.build(response_arena.allocator(), source, &lines, .utf8, fixture.analysis.ast_file, &positions);

    try std.testing.expectEqual(@as(usize, 1), ranges.len);
    try std.testing.expectEqual(@as(u32, 0), ranges[0].range.start.line);
    try std.testing.expectEqual(@as(u32, 0), ranges[0].range.start.character);
    try std.testing.expectEqual(@as(u32, 0), ranges[0].range.end.line);
    try std.testing.expectEqual(@as(u32, 0), ranges[0].range.end.character);
    try std.testing.expect(ranges[0].parent == null);
}
