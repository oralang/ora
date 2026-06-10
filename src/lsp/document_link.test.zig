const std = @import("std");
const document_link = @import("document_link.zig");
const ora_root = @import("ora_root");

const line_index = ora_root.lsp.line_index;
const workspace = ora_root.lsp.workspace;

test "lsp document link: builds import links" {
    const source =
        \\const Math = @import("lib/math.ora");
        \\pub fn run() -> u256 { return 1; }
    ;

    var lines = try line_index.LineIndex.init(std.testing.allocator, source);
    defer lines.deinit(std.testing.allocator);

    const imports = [_]workspace.ResolvedImport{.{
        .specifier = "lib/math.ora",
        .alias = "Math",
        .resolved_path = "/tmp/lib/math.ora",
    }};

    var response_arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer response_arena.deinit();

    const links = (try document_link.build(response_arena.allocator(), source, &lines, .utf8, &imports)) orelse return error.ExpectedLinks;

    try std.testing.expectEqual(@as(usize, 1), links.len);
    try std.testing.expectEqualStrings("file:///tmp/lib/math.ora", links[0].target.?);
    try std.testing.expectEqualStrings("Open lib/math.ora", links[0].tooltip.?);
}

test "lsp document link: converts import path ranges to utf16" {
    const source = "const marker = \"é\"; const Math = @import(\"lib/math.ora\");";

    var lines = try line_index.LineIndex.init(std.testing.allocator, source);
    defer lines.deinit(std.testing.allocator);

    const imports = [_]workspace.ResolvedImport{.{
        .specifier = "lib/math.ora",
        .alias = "Math",
        .resolved_path = "/tmp/lib/math.ora",
    }};

    var response_arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer response_arena.deinit();

    const links = (try document_link.build(response_arena.allocator(), source, &lines, .utf16, &imports)) orelse return error.ExpectedLinks;

    const specifier_offset = std.mem.indexOf(u8, source, "lib/math.ora") orelse return error.ExpectedSpecifier;
    const expected_start = lines.offsetToPosition(source, @intCast(specifier_offset), .utf16);
    const expected_end = lines.offsetToPosition(source, @intCast(specifier_offset + "lib/math.ora".len), .utf16);

    try std.testing.expectEqual(expected_start.line, links[0].range.start.line);
    try std.testing.expectEqual(expected_start.character, links[0].range.start.character);
    try std.testing.expectEqual(expected_end.line, links[0].range.end.line);
    try std.testing.expectEqual(expected_end.character, links[0].range.end.character);
}

test "lsp document link: returns null when no import specifier range matches" {
    const source = "pub fn run() -> u256 { return 1; }";

    var lines = try line_index.LineIndex.init(std.testing.allocator, source);
    defer lines.deinit(std.testing.allocator);

    const imports = [_]workspace.ResolvedImport{.{
        .specifier = "missing.ora",
        .alias = null,
        .resolved_path = "/tmp/missing.ora",
    }};

    var response_arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer response_arena.deinit();

    const links = try document_link.build(response_arena.allocator(), source, &lines, .utf8, &imports);
    try std.testing.expect(links == null);
}
