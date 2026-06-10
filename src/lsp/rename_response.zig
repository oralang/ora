const std = @import("std");
const lsp = @import("lsp");
const ora_root = @import("ora_root");
const protocol_ranges = @import("protocol_ranges.zig");
const uri_ranges = @import("uri_ranges.zig");

const frontend = ora_root.lsp.frontend;
const line_index = ora_root.lsp.line_index;
const text_edits = ora_root.lsp.text_edits;

const Allocator = std.mem.Allocator;
const types = lsp.types;

pub fn buildPrepare(
    arena: Allocator,
    source: []const u8,
    lines: *const line_index.LineIndex,
    encoding: text_edits.PositionEncoding,
    definition_range: frontend.Range,
    placeholder: []const u8,
) !lsp.ResultType("textDocument/prepareRename") {
    return .{ .literal_1 = .{
        .range = protocol_ranges.byteRangeToLsp(source, lines, encoding, definition_range) orelse return null,
        .placeholder = try arena.dupe(u8, placeholder),
    } };
}

pub fn putChange(
    arena: Allocator,
    changes: *lsp.parser.Map(types.DocumentUri, []const types.TextEdit),
    uri: types.DocumentUri,
    source: ?uri_ranges.IndexedSource,
    ranges: []const frontend.Range,
    new_name: []const u8,
    encoding: text_edits.PositionEncoding,
) ![]const types.TextEdit {
    if (ranges.len == 0) return &.{};

    const edits = try buildTextEdits(arena, source, ranges, new_name, encoding);
    try changes.map.put(arena, try arena.dupe(u8, uri), edits);
    return edits;
}

pub fn workspaceEdit(changes: lsp.parser.Map(types.DocumentUri, []const types.TextEdit)) types.WorkspaceEdit {
    return .{ .changes = changes };
}

pub fn buildTextEdits(
    arena: Allocator,
    source: ?uri_ranges.IndexedSource,
    ranges: []const frontend.Range,
    new_name: []const u8,
    encoding: text_edits.PositionEncoding,
) ![]const types.TextEdit {
    const edits = try arena.alloc(types.TextEdit, ranges.len);
    const replacement = try arena.dupe(u8, new_name);
    uri_ranges.fillRenameTextEdits(edits, source, ranges, replacement, encoding);
    return edits;
}
