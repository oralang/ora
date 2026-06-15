const std = @import("std");
const lsp = @import("lsp");
const ora_root = @import("ora_root");

const completion = ora_root.lsp.completion;
const frontend = ora_root.lsp.frontend;
const line_index_api = ora_root.lsp.line_index;
const text_edits = ora_root.lsp.text_edits;
const protocol_ranges = @import("protocol_ranges.zig");

const types = lsp.types;

pub const ImportedMemberAccess = struct {
    alias: []const u8,
    member_name: []const u8,
    member_range: frontend.Range,
};

const IdentifierBounds = struct {
    start: usize,
    end: usize,
};

pub fn isAtLineStart(source: []const u8, position: frontend.Position) bool {
    var line: u32 = 0;
    var line_start: usize = 0;
    for (source, 0..) |c, i| {
        if (line == position.line) {
            const prefix = source[line_start..@min(line_start + position.character, source.len)];
            for (prefix) |ch| {
                if (ch != ' ' and ch != '\t') return false;
            }
            return true;
        }
        if (c == '\n') {
            line += 1;
            line_start = i + 1;
        }
    }
    return line == position.line;
}

pub fn occurrenceIsMemberAccess(
    source: []const u8,
    line_index: *const line_index_api.LineIndex,
    range: frontend.Range,
) bool {
    const start: usize = @intCast(line_index.positionToOffset(
        source,
        range.start.line,
        range.start.character,
        .utf8,
    ) orelse return true);
    const end: usize = @intCast(line_index.positionToOffset(
        source,
        range.end.line,
        range.end.character,
        .utf8,
    ) orelse return true);

    if (start > 0 and source[start - 1] == '.') return true;
    if (end < source.len and source[end] == '.') return true;
    return false;
}

pub fn importedMemberAccessAtLspPosition(
    source: []const u8,
    line_index: *const line_index_api.LineIndex,
    encoding: text_edits.PositionEncoding,
    lsp_position: types.Position,
) ?ImportedMemberAccess {
    const raw_offset = line_index.positionToOffset(
        source,
        lsp_position.line,
        lsp_position.character,
        encoding,
    ) orelse return null;

    var cursor: usize = @intCast(raw_offset);
    if (cursor >= source.len or !isOraIdentifierByte(source[cursor])) {
        if (cursor == 0 or !isOraIdentifierByte(source[cursor - 1])) return null;
        cursor -= 1;
    }

    const ident = identifierBoundsAtOffset(source, cursor) orelse return null;
    if (importedMemberAccessFromAlias(source, line_index, ident)) |access| return access;
    return importedMemberAccessFromMember(source, line_index, ident);
}

pub fn frontendRangesEqual(a: frontend.Range, b: frontend.Range) bool {
    return a.start.line == b.start.line and
        a.start.character == b.start.character and
        a.end.line == b.end.line and
        a.end.character == b.end.character;
}

pub fn definitionLineLooksLikeImportAlias(
    source: []const u8,
    line_index: *const line_index_api.LineIndex,
    range: frontend.Range,
) bool {
    const line_no: usize = range.start.line;
    if (line_no >= line_index.line_starts.len) return true;

    const start: usize = @intCast(line_index.line_starts[line_no]);
    const end: usize = if (line_no + 1 < line_index.line_starts.len)
        @intCast(line_index.line_starts[line_no + 1] - 1)
    else
        source.len;

    if (start > source.len or end > source.len or start > end) return true;
    return std.mem.indexOf(u8, source[start..end], "@import") != null;
}

pub fn diagnosticRangeToLsp(
    source: []const u8,
    line_index: ?*const line_index_api.LineIndex,
    encoding: text_edits.PositionEncoding,
    range: frontend.Range,
) types.Range {
    const index = line_index orelse return protocol_ranges.rawRange(range);
    return protocol_ranges.byteRangeToLspOrRaw(source, index, encoding, range);
}

pub fn lspPositionToBytePosition(
    source: []const u8,
    line_index: *const line_index_api.LineIndex,
    encoding: text_edits.PositionEncoding,
    position: frontend.Position,
) ?frontend.Position {
    return protocol_ranges.lspPositionToBytePosition(
        source,
        line_index,
        encoding,
        .{ .line = position.line, .character = position.character },
    );
}

pub fn toLspPosition(pos: frontend.Position) types.Position {
    return .{ .line = pos.line, .character = pos.character };
}

pub fn frontendSeverityToLsp(severity: frontend.Severity) types.DiagnosticSeverity {
    return switch (severity) {
        .err => .Error,
        .warning => .Warning,
        .information => .Information,
        .hint => .Hint,
    };
}

pub fn completionKindToLsp(kind: completion.Kind) types.CompletionItemKind {
    return switch (kind) {
        .keyword => .Keyword,
        .contract => .Class,
        .function => .Function,
        .method => .Method,
        .variable => .Variable,
        .field => .Field,
        .constant => .Constant,
        .parameter => .Variable,
        .struct_decl => .Struct,
        .bitfield_decl => .Struct,
        .enum_decl => .Enum,
        .enum_member => .EnumMember,
        .trait_decl => .Interface,
        .impl_decl => .Class,
        .type_alias => .Struct,
        .event => .Event,
        .error_decl => .Class,
    };
}

pub fn normalizeIndentSize(tab_size: u32) u32 {
    if (tab_size == 0) return 4;
    return @min(tab_size, 16);
}

pub fn lastFullText(changes: []const types.TextDocumentContentChangeEvent) ?[]const u8 {
    var index = changes.len;
    while (index > 0) {
        index -= 1;
        switch (changes[index]) {
            .literal_1 => |full| return full.text,
            .literal_0 => {},
        }
    }
    return null;
}

pub fn negotiatePositionEncoding(params: types.InitializeParams) text_edits.PositionEncoding {
    if (params.capabilities.general) |general| {
        if (general.positionEncodings) |encodings| {
            for (encodings) |encoding| {
                if (encoding == .@"utf-16") return .utf16;
            }

            // LSP guarantees UTF-16 support even if omitted, but handle non-compliant clients.
            for (encodings) |encoding| {
                switch (encoding) {
                    .@"utf-8" => return .utf8,
                    .@"utf-32" => return .utf32,
                    else => {},
                }
            }
        }
    }

    return .utf16;
}

pub fn toLspPositionEncoding(encoding: text_edits.PositionEncoding) types.PositionEncodingKind {
    return switch (encoding) {
        .utf8 => .@"utf-8",
        .utf16 => .@"utf-16",
        .utf32 => .@"utf-32",
    };
}

pub fn containsString(items: []const []u8, needle: []const u8) bool {
    for (items) |item| {
        if (std.mem.eql(u8, item, needle)) return true;
    }
    return false;
}

fn importedMemberAccessFromAlias(
    source: []const u8,
    line_index: *const line_index_api.LineIndex,
    alias_bounds: IdentifierBounds,
) ?ImportedMemberAccess {
    if (alias_bounds.end >= source.len or source[alias_bounds.end] != '.') return null;
    const member_start = alias_bounds.end + 1;
    if (member_start >= source.len or !isOraIdentifierByte(source[member_start])) return null;

    var member_end = member_start;
    while (member_end < source.len and isOraIdentifierByte(source[member_end])) member_end += 1;

    return .{
        .alias = source[alias_bounds.start..alias_bounds.end],
        .member_name = source[member_start..member_end],
        .member_range = byteRangeFromOffsets(source, line_index, member_start, member_end) orelse return null,
    };
}

fn importedMemberAccessFromMember(
    source: []const u8,
    line_index: *const line_index_api.LineIndex,
    member_bounds: IdentifierBounds,
) ?ImportedMemberAccess {
    if (member_bounds.start == 0 or source[member_bounds.start - 1] != '.') return null;

    const alias_end = member_bounds.start - 1;
    var alias_start = alias_end;
    while (alias_start > 0 and isOraIdentifierByte(source[alias_start - 1])) alias_start -= 1;
    if (alias_start == alias_end) return null;

    return .{
        .alias = source[alias_start..alias_end],
        .member_name = source[member_bounds.start..member_bounds.end],
        .member_range = byteRangeFromOffsets(source, line_index, member_bounds.start, member_bounds.end) orelse return null,
    };
}

fn identifierBoundsAtOffset(source: []const u8, offset: usize) ?IdentifierBounds {
    if (offset >= source.len or !isOraIdentifierByte(source[offset])) return null;

    var start = offset;
    while (start > 0 and isOraIdentifierByte(source[start - 1])) start -= 1;

    var end = offset + 1;
    while (end < source.len and isOraIdentifierByte(source[end])) end += 1;

    return .{ .start = start, .end = end };
}

fn byteRangeFromOffsets(
    source: []const u8,
    line_index: *const line_index_api.LineIndex,
    start: usize,
    end: usize,
) ?frontend.Range {
    const start_u32 = std.math.cast(u32, start) orelse return null;
    const end_u32 = std.math.cast(u32, end) orelse return null;
    return .{
        .start = line_index.offsetToPosition(source, start_u32, .utf8),
        .end = line_index.offsetToPosition(source, end_u32, .utf8),
    };
}

fn isOraIdentifierByte(byte: u8) bool {
    return std.ascii.isAlphanumeric(byte) or byte == '_';
}
