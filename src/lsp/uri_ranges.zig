const std = @import("std");
const lsp = @import("lsp");
const ora_root = @import("ora_root");

const frontend = ora_root.lsp.frontend;
const line_index_api = ora_root.lsp.line_index;
const text_edits = ora_root.lsp.text_edits;
const protocol_ranges = @import("protocol_ranges.zig");

const Allocator = std.mem.Allocator;
const types = lsp.types;

pub const IndexedSource = struct {
    source: []const u8,
    line_index: *const line_index_api.LineIndex,
};

pub fn byteRangeToLsp(
    source: ?IndexedSource,
    encoding: text_edits.PositionEncoding,
    range: frontend.Range,
) types.Range {
    if (source) |indexed| {
        return protocol_ranges.byteRangeToLspOrRaw(indexed.source, indexed.line_index, encoding, range);
    }
    return protocol_ranges.rawRange(range);
}

pub fn appendLocationsAssumeCapacity(
    locations: *std.ArrayList(types.Location),
    uri: []const u8,
    ranges: []const frontend.Range,
    source: ?IndexedSource,
    encoding: text_edits.PositionEncoding,
) void {
    for (ranges) |range| {
        locations.appendAssumeCapacity(.{
            .uri = uri,
            .range = byteRangeToLsp(source, encoding, range),
        });
    }
}

pub fn rangesToLsp(
    arena: Allocator,
    source: ?IndexedSource,
    encoding: text_edits.PositionEncoding,
    ranges: []const frontend.Range,
) ![]types.Range {
    const result = try arena.alloc(types.Range, ranges.len);
    fillLspRanges(result, source, encoding, ranges);
    return result;
}

pub fn fillLspRanges(
    result: []types.Range,
    source: ?IndexedSource,
    encoding: text_edits.PositionEncoding,
    ranges: []const frontend.Range,
) void {
    for (ranges, 0..) |range, index| {
        result[index] = byteRangeToLsp(source, encoding, range);
    }
}

pub fn lspRangeToByte(
    source: ?IndexedSource,
    encoding: text_edits.PositionEncoding,
    range: types.Range,
) ?frontend.Range {
    if (source) |indexed| {
        return protocol_ranges.lspRangeToByte(indexed.source, indexed.line_index, encoding, range);
    }
    return .{
        .start = toFrontendPosition(range.start),
        .end = toFrontendPosition(range.end),
    };
}

pub fn fillRenameTextEdits(
    edits: []types.TextEdit,
    source: ?IndexedSource,
    ranges: []const frontend.Range,
    replacement: []const u8,
    encoding: text_edits.PositionEncoding,
) void {
    for (ranges, 0..) |range, index| {
        edits[index] = .{
            .range = byteRangeToLsp(source, encoding, range),
            .newText = replacement,
        };
    }
}

fn toFrontendPosition(pos: types.Position) frontend.Position {
    return .{ .line = pos.line, .character = pos.character };
}
