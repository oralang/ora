const lsp = @import("lsp");
const ora_root = @import("ora_root");

const compiler = ora_root.compiler;
const frontend = ora_root.lsp.frontend;
const line_index_api = ora_root.lsp.line_index;
const text_edits = ora_root.lsp.text_edits;

const types = lsp.types;

pub fn lspPositionToBytePosition(
    source: []const u8,
    line_index: *const line_index_api.LineIndex,
    encoding: text_edits.PositionEncoding,
    position: types.Position,
) ?frontend.Position {
    if (fastLspPositionToBytePosition(source, line_index, encoding, position)) |byte_position| {
        return byte_position;
    }
    const offset = line_index.positionToOffset(source, position.line, position.character, encoding) orelse return null;
    return line_index.offsetToPosition(source, offset, .utf8);
}

pub fn lspRangeToByte(
    source: []const u8,
    line_index: *const line_index_api.LineIndex,
    encoding: text_edits.PositionEncoding,
    range: types.Range,
) ?frontend.Range {
    const start = lspPositionToBytePosition(source, line_index, encoding, range.start) orelse return null;
    const end = lspPositionToBytePosition(source, line_index, encoding, range.end) orelse return null;
    if (positionLessThan(end, start)) return null;
    return .{ .start = start, .end = end };
}

pub fn bytePositionToLsp(
    source: []const u8,
    line_index: *const line_index_api.LineIndex,
    encoding: text_edits.PositionEncoding,
    position: frontend.Position,
) ?types.Position {
    if (fastBytePositionToLsp(source, line_index, encoding, position)) |lsp_position| {
        return lsp_position;
    }
    const offset = line_index.positionToOffset(source, position.line, position.character, .utf8) orelse return null;
    const converted = line_index.offsetToPosition(source, offset, encoding);
    return .{ .line = converted.line, .character = converted.character };
}

pub fn byteRangeToLsp(
    source: []const u8,
    line_index: *const line_index_api.LineIndex,
    encoding: text_edits.PositionEncoding,
    range: frontend.Range,
) ?types.Range {
    const start = bytePositionToLsp(source, line_index, encoding, range.start) orelse return null;
    const end = bytePositionToLsp(source, line_index, encoding, range.end) orelse return null;
    return .{ .start = start, .end = end };
}

pub fn byteRangeToLspOrRaw(
    source: []const u8,
    line_index: *const line_index_api.LineIndex,
    encoding: text_edits.PositionEncoding,
    range: frontend.Range,
) types.Range {
    return byteRangeToLsp(source, line_index, encoding, range) orelse rawRange(range);
}

pub fn textRangeToLsp(
    source: []const u8,
    line_index: *const line_index_api.LineIndex,
    encoding: text_edits.PositionEncoding,
    range: compiler.TextRange,
) types.Range {
    return rawRange(line_index.textRangeToRange(source, range, encoding));
}

pub fn rawRange(range: frontend.Range) types.Range {
    return .{
        .start = .{ .line = range.start.line, .character = range.start.character },
        .end = .{ .line = range.end.line, .character = range.end.character },
    };
}

fn positionLessThan(a: frontend.Position, b: frontend.Position) bool {
    if (a.line != b.line) return a.line < b.line;
    return a.character < b.character;
}

fn fastLspPositionToBytePosition(
    source: []const u8,
    line_index: *const line_index_api.LineIndex,
    encoding: text_edits.PositionEncoding,
    position: types.Position,
) ?frontend.Position {
    if (source.len != line_index.source_len) return null;
    const line_no: usize = @intCast(position.line);
    if (line_no >= line_index.line_starts.len) return null;

    const can_reuse_character = encoding == .utf8 or
        (line_index.line_ascii.len == line_index.line_starts.len and line_index.line_ascii[line_no]);
    if (!can_reuse_character) return null;

    const line_start = line_index.line_starts[line_no];
    const next_line = line_no + 1;
    const line_end = if (next_line < line_index.line_starts.len)
        line_index.line_starts[next_line] - 1
    else
        line_index.source_len;
    if (position.character > line_end - line_start) return null;

    return .{ .line = position.line, .character = position.character };
}

fn fastBytePositionToLsp(
    source: []const u8,
    line_index: *const line_index_api.LineIndex,
    encoding: text_edits.PositionEncoding,
    position: frontend.Position,
) ?types.Position {
    if (source.len != line_index.source_len) return null;
    const line_no: usize = @intCast(position.line);
    if (line_no >= line_index.line_starts.len) return null;

    const can_reuse_character = encoding == .utf8 or
        (line_index.line_ascii.len == line_index.line_starts.len and line_index.line_ascii[line_no]);
    if (!can_reuse_character) return null;

    const line_start = line_index.line_starts[line_no];
    const next_line = line_no + 1;
    const line_end = if (next_line < line_index.line_starts.len)
        line_index.line_starts[next_line] - 1
    else
        line_index.source_len;
    if (position.character > line_end - line_start) return null;

    return .{ .line = position.line, .character = position.character };
}
