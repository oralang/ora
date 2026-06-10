const std = @import("std");
const frontend = @import("frontend.zig");
const keyword_docs = @import("keyword_docs.zig");
const refinement_docs = @import("refinement_docs.zig");
const semantic_index = @import("semantic_index.zig");

const Allocator = std.mem.Allocator;

pub const Hover = struct {
    contents: []const u8,
    range: frontend.Range,

    pub fn deinit(self: *Hover, allocator: Allocator) void {
        allocator.free(self.contents);
    }
};

pub fn hoverAtIndex(
    allocator: Allocator,
    source: []const u8,
    position: frontend.Position,
    index: *const semantic_index.SemanticIndex,
) !?Hover {
    if (!index.parse_succeeded) return null;

    if (semantic_index.findSymbolAtPosition(index, position)) |symbol_index| {
        const candidate = index.symbols[symbol_index];
        if (rangeContainsPosition(candidate.selection_range, position)) {
            const value = try formatHoverValueAlloc(allocator, candidate);
            return .{ .contents = value, .range = candidate.selection_range };
        }
    }

    if (try keywordHoverAt(allocator, source, position)) |hover| return hover;
    return refinementHoverAt(allocator, source, position);
}

fn formatHoverValueAlloc(allocator: Allocator, symbol: semantic_index.Symbol) ![]u8 {
    const signature = try formatSignatureAlloc(allocator, symbol);
    defer allocator.free(signature);

    if (symbol.doc_comment) |doc| {
        return std.fmt.allocPrint(allocator, "```ora\n{s}\n```\n---\n{s}", .{ signature, doc });
    }

    return std.fmt.allocPrint(allocator, "```ora\n{s}\n```", .{signature});
}

fn formatSignatureAlloc(allocator: Allocator, symbol: semantic_index.Symbol) ![]u8 {
    return switch (symbol.kind) {
        .contract => std.fmt.allocPrint(allocator, "contract {s}", .{symbol.name}),
        .function, .method => if (symbol.detail) |detail|
            std.fmt.allocPrint(allocator, "fn {s}{s}", .{ symbol.name, detail })
        else
            std.fmt.allocPrint(allocator, "fn {s}()", .{symbol.name}),
        .variable => if (symbol.detail) |detail|
            std.fmt.allocPrint(allocator, "var {s}: {s}", .{ symbol.name, detail })
        else
            std.fmt.allocPrint(allocator, "var {s}", .{symbol.name}),
        .field => if (symbol.detail) |detail|
            std.fmt.allocPrint(allocator, "field {s}: {s}", .{ symbol.name, detail })
        else
            std.fmt.allocPrint(allocator, "field {s}", .{symbol.name}),
        .constant => if (symbol.detail) |detail|
            std.fmt.allocPrint(allocator, "const {s}: {s}", .{ symbol.name, detail })
        else
            std.fmt.allocPrint(allocator, "const {s}", .{symbol.name}),
        .parameter => if (symbol.detail) |detail|
            std.fmt.allocPrint(allocator, "{s}: {s}", .{ symbol.name, detail })
        else
            std.fmt.allocPrint(allocator, "{s}", .{symbol.name}),
        .struct_decl => std.fmt.allocPrint(allocator, "struct {s}", .{symbol.name}),
        .bitfield_decl => std.fmt.allocPrint(allocator, "bitfield {s}", .{symbol.name}),
        .enum_decl => if (symbol.detail) |detail|
            std.fmt.allocPrint(allocator, "enum {s}{s}", .{ symbol.name, detail })
        else
            std.fmt.allocPrint(allocator, "enum {s}", .{symbol.name}),
        .enum_member => if (symbol.detail) |detail|
            std.fmt.allocPrint(allocator, "enum member {s}{s}", .{ symbol.name, detail })
        else
            std.fmt.allocPrint(allocator, "enum member {s}", .{symbol.name}),
        .trait_decl => std.fmt.allocPrint(allocator, "trait {s}", .{symbol.name}),
        .impl_decl => if (symbol.detail) |detail|
            std.fmt.allocPrint(allocator, "{s} {s}", .{ symbol.name, detail })
        else
            std.fmt.allocPrint(allocator, "{s}", .{symbol.name}),
        .type_alias => if (symbol.detail) |detail|
            std.fmt.allocPrint(allocator, "type {s} = {s}", .{ symbol.name, detail })
        else
            std.fmt.allocPrint(allocator, "type {s}", .{symbol.name}),
        .event => if (symbol.detail) |detail|
            std.fmt.allocPrint(allocator, "log {s}{s}", .{ symbol.name, detail })
        else
            std.fmt.allocPrint(allocator, "log {s}", .{symbol.name}),
        .error_decl => if (symbol.detail) |detail|
            std.fmt.allocPrint(allocator, "error {s}{s}", .{ symbol.name, detail })
        else
            std.fmt.allocPrint(allocator, "error {s}", .{symbol.name}),
    };
}

fn keywordHoverAt(allocator: Allocator, source: []const u8, position: frontend.Position) !?Hover {
    const word = wordAtPosition(source, position) orelse return null;
    const doc = keyword_docs.documentation(word.text) orelse return null;

    const value = try std.fmt.allocPrint(allocator, "```ora\n{s}\n```\n---\n{s}", .{ word.text, doc });
    return .{ .contents = value, .range = word.range };
}

fn refinementHoverAt(allocator: Allocator, source: []const u8, position: frontend.Position) !?Hover {
    const word = wordAtPosition(source, position) orelse return null;
    const entry = refinement_docs.entryForName(word.text) orelse return null;
    const value = try refinement_docs.markdownAlloc(allocator, entry);
    return .{ .contents = value, .range = word.range };
}

const WordAtPosition = struct { text: []const u8, range: frontend.Range };

fn wordAtPosition(source: []const u8, position: frontend.Position) ?WordAtPosition {
    var cursor: usize = 0;
    var line: u32 = 0;
    while (cursor < source.len and line < position.line) : (cursor += 1) {
        if (source[cursor] == '\n') line += 1;
    }
    if (line != position.line) return null;

    const line_start = cursor;
    const col: usize = @intCast(position.character);
    const pos = @min(line_start + col, source.len);

    var start = pos;
    while (start > line_start and isIdentChar(source[start - 1])) start -= 1;
    var end = pos;
    while (end < source.len and source[end] != '\n' and isIdentChar(source[end])) end += 1;
    if (start == end) return null;

    const start_char: u32 = @intCast(start - line_start);
    const end_char: u32 = @intCast(end - line_start);
    return .{
        .text = source[start..end],
        .range = .{
            .start = .{ .line = position.line, .character = start_char },
            .end = .{ .line = position.line, .character = end_char },
        },
    };
}

fn isIdentChar(ch: u8) bool {
    return (ch >= 'a' and ch <= 'z') or (ch >= 'A' and ch <= 'Z') or (ch >= '0' and ch <= '9') or ch == '_';
}

fn rangeContainsPosition(range: frontend.Range, position: frontend.Position) bool {
    if (positionLessThan(position, range.start)) return false;
    if (!positionLessThan(position, range.end)) return false;
    return true;
}

fn positionLessThan(lhs: frontend.Position, rhs: frontend.Position) bool {
    if (lhs.line < rhs.line) return true;
    if (lhs.line > rhs.line) return false;
    return lhs.character < rhs.character;
}
