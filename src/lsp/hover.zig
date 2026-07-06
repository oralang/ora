const std = @import("std");
const builtin_docs = @import("builtin_docs.zig");
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

    if (try builtinHoverAt(allocator, source, position)) |hover| return hover;
    if (try directiveArgHoverAt(allocator, source, position)) |hover| return hover;
    if (try keywordHoverAt(allocator, source, position)) |hover| return hover;
    if (try resourceTypeHoverAt(allocator, source, position)) |hover| return hover;
    return refinementHoverAt(allocator, source, position);
}

fn formatHoverValueAlloc(allocator: Allocator, symbol: semantic_index.Symbol) ![]u8 {
    const value = try allocator.alloc(u8, hoverValueCapacity(symbol));
    var cursor: usize = 0;

    appendBytes(value, &cursor, "```ora\n");
    appendSignature(value, &cursor, symbol);
    appendBytes(value, &cursor, "\n```");
    if (symbol.doc_comment) |doc| {
        appendBytes(value, &cursor, "\n---\n");
        appendBytes(value, &cursor, doc);
    }

    std.debug.assert(cursor == value.len);
    return value;
}

fn appendSignature(buffer: []u8, cursor: *usize, symbol: semantic_index.Symbol) void {
    switch (symbol.kind) {
        .contract => {
            appendBytes(buffer, cursor, "contract ");
            appendBytes(buffer, cursor, symbol.name);
        },
        .function, .method => {
            const prefix = functionPrefix(symbol);
            if (symbol.detail) |detail|
                appendNameDetail(buffer, cursor, prefix, symbol.name, detail, "")
            else
                appendNameDetail(buffer, cursor, prefix, symbol.name, "", "()");
        },
        .variable => if (symbol.detail) |detail|
            appendNameDetail(buffer, cursor, "var ", symbol.name, detail, ": ")
        else
            appendNameDetail(buffer, cursor, "var ", symbol.name, "", ""),
        .field => if (symbol.detail) |detail|
            appendNameDetail(buffer, cursor, "field ", symbol.name, detail, ": ")
        else
            appendNameDetail(buffer, cursor, "field ", symbol.name, "", ""),
        .constant => if (symbol.detail) |detail|
            appendNameDetail(buffer, cursor, "const ", symbol.name, detail, ": ")
        else
            appendNameDetail(buffer, cursor, "const ", symbol.name, "", ""),
        .parameter => if (symbol.detail) |detail|
            appendNameDetail(buffer, cursor, "", symbol.name, detail, ": ")
        else
            appendNameDetail(buffer, cursor, "", symbol.name, "", ""),
        .struct_decl => {
            appendBytes(buffer, cursor, "struct ");
            appendBytes(buffer, cursor, symbol.name);
        },
        .bitfield_decl => {
            appendBytes(buffer, cursor, "bitfield ");
            appendBytes(buffer, cursor, symbol.name);
        },
        .enum_decl => if (symbol.detail) |detail|
            appendNameDetail(buffer, cursor, "enum ", symbol.name, detail, "")
        else
            appendNameDetail(buffer, cursor, "enum ", symbol.name, "", ""),
        .enum_member => if (symbol.detail) |detail|
            appendNameDetail(buffer, cursor, "enum member ", symbol.name, detail, "")
        else
            appendNameDetail(buffer, cursor, "enum member ", symbol.name, "", ""),
        .trait_decl => {
            appendBytes(buffer, cursor, "trait ");
            appendBytes(buffer, cursor, symbol.name);
        },
        .impl_decl => if (symbol.detail) |detail|
            appendNameDetail(buffer, cursor, "", symbol.name, detail, " ")
        else
            appendNameDetail(buffer, cursor, "", symbol.name, "", ""),
        .type_alias => if (symbol.detail) |detail|
            appendNameDetail(buffer, cursor, "type ", symbol.name, detail, " = ")
        else
            appendNameDetail(buffer, cursor, "type ", symbol.name, "", ""),
        .event => if (symbol.detail) |detail|
            appendNameDetail(buffer, cursor, "log ", symbol.name, detail, "")
        else
            appendNameDetail(buffer, cursor, "log ", symbol.name, "", ""),
        .error_decl => if (symbol.detail) |detail|
            appendNameDetail(buffer, cursor, "error ", symbol.name, detail, "")
        else
            appendNameDetail(buffer, cursor, "error ", symbol.name, "", ""),
    }
}

fn appendNameDetail(buffer: []u8, cursor: *usize, prefix: []const u8, name: []const u8, detail: []const u8, separator: []const u8) void {
    appendBytes(buffer, cursor, prefix);
    appendBytes(buffer, cursor, name);
    appendBytes(buffer, cursor, separator);
    appendBytes(buffer, cursor, detail);
}

fn appendBytes(buffer: []u8, cursor: *usize, value: []const u8) void {
    @memcpy(buffer[cursor.*..][0..value.len], value);
    cursor.* += value.len;
}

fn functionPrefix(symbol: semantic_index.Symbol) []const u8 {
    return if (symbol.is_inline) "inline fn " else "fn ";
}

fn hoverValueCapacity(symbol: semantic_index.Symbol) usize {
    var total: usize = "```ora\n".len + "\n```".len + signatureCapacity(symbol);
    if (symbol.doc_comment) |doc| total += "\n---\n".len + doc.len;
    return total;
}

fn signatureCapacity(symbol: semantic_index.Symbol) usize {
    const detail_len = if (symbol.detail) |detail| detail.len else 0;
    return symbol.name.len + detail_len + switch (symbol.kind) {
        .contract => "contract ".len,
        .function, .method => functionPrefix(symbol).len + if (symbol.detail == null) "()".len else 0,
        .variable => "var ".len + if (symbol.detail != null) ": ".len else 0,
        .field => "field ".len + if (symbol.detail != null) ": ".len else 0,
        .constant => "const ".len + if (symbol.detail != null) ": ".len else 0,
        .parameter => if (symbol.detail != null) ": ".len else 0,
        .struct_decl => "struct ".len,
        .bitfield_decl => "bitfield ".len,
        .enum_decl => "enum ".len,
        .enum_member => "enum member ".len,
        .trait_decl => "trait ".len,
        .impl_decl => if (symbol.detail != null) " ".len else 0,
        .type_alias => "type  = ".len,
        .event => "log ".len,
        .error_decl => "error ".len,
    };
}

fn keywordHoverAt(allocator: Allocator, source: []const u8, position: frontend.Position) !?Hover {
    const word = wordAtPosition(source, position) orelse return null;
    const doc = keyword_docs.documentation(word.text) orelse return null;

    const value = try std.fmt.allocPrint(allocator, "```ora\n{s}\n```\n---\n{s}", .{ word.text, doc });
    return .{ .contents = value, .range = word.range };
}

fn builtinHoverAt(allocator: Allocator, source: []const u8, position: frontend.Position) !?Hover {
    const word = builtinWordAtPosition(source, position) orelse return null;
    const entry = builtin_docs.entryForName(word.text) orelse return null;
    const markdown = try builtin_docs.markdownAlloc(allocator, entry);
    errdefer allocator.free(markdown);
    const value = try std.fmt.allocPrint(allocator, "```ora\n{s}\n```\n---\n{s}", .{ entry.signature, markdown });
    allocator.free(markdown);
    return .{ .contents = value, .range = word.range };
}

/// Hover on a keyword-like state inside a directive builtin's parens,
/// e.g. `cold` in `@callHint(cold)`. The value set and docs come from the
/// builtin registry, so this stays in lockstep with the language enum.
fn directiveArgHoverAt(allocator: Allocator, source: []const u8, position: frontend.Position) !?Hover {
    const word = wordAtPosition(source, position) orelse return null;
    const line_start = lineStartOffset(source, position.line) orelse return null;
    var i = line_start + @as(usize, @intCast(word.range.start.character));
    while (i > line_start and (source[i - 1] == ' ' or source[i - 1] == '\t')) i -= 1;
    if (i == line_start or source[i - 1] != '(') return null;
    i -= 1;
    while (i > line_start and (source[i - 1] == ' ' or source[i - 1] == '\t')) i -= 1;
    const name_end = i;
    while (i > line_start and isIdentChar(source[i - 1])) i -= 1;
    if (i == name_end or i == line_start or source[i - 1] != '@') return null;
    const entry = builtin_docs.entryForName(source[i..name_end]) orelse return null;
    for (entry.arg_values) |value| {
        if (!std.mem.eql(u8, value.name, word.text)) continue;
        const contents = try std.fmt.allocPrint(
            allocator,
            "```ora\n@{s}({s})\n```\n---\n{s}",
            .{ entry.name, value.name, value.documentation },
        );
        return .{ .contents = contents, .range = word.range };
    }
    return null;
}

fn resourceTypeHoverAt(allocator: Allocator, source: []const u8, position: frontend.Position) !?Hover {
    const word = wordAtPosition(source, position) orelse return null;
    if (!std.mem.eql(u8, word.text, "Resource")) return null;
    const value = try allocator.dupe(
        u8,
        "```ora\nResource<T>\n```\n---\nOpaque storage or transient resource place for a declared `resource` domain. `Resource<T>` values are capabilities: they are stored in contract state, observed through `@amount`, and mutated through `@create`, `@destroy`, and `@move`; they cannot be exposed through ABI parameters, returns, or logs.",
    );
    return .{ .contents = value, .range = word.range };
}

fn refinementHoverAt(allocator: Allocator, source: []const u8, position: frontend.Position) !?Hover {
    const word = wordAtPosition(source, position) orelse return null;
    const entry = refinement_docs.entryForName(word.text) orelse return null;
    const value = try refinement_docs.markdownAlloc(allocator, entry);
    return .{ .contents = value, .range = word.range };
}

const WordAtPosition = struct { text: []const u8, range: frontend.Range };

fn builtinWordAtPosition(source: []const u8, position: frontend.Position) ?WordAtPosition {
    const word = wordAtPosition(source, position) orelse return wordAfterAtAtPosition(source, position);
    if (word.range.start.character == 0) return null;
    const line_start = lineStartOffset(source, position.line) orelse return null;
    const at_offset = line_start + @as(usize, @intCast(word.range.start.character)) - 1;
    if (at_offset >= source.len or source[at_offset] != '@') return null;
    return .{
        .text = word.text,
        .range = .{
            .start = .{ .line = word.range.start.line, .character = word.range.start.character - 1 },
            .end = word.range.end,
        },
    };
}

fn wordAfterAtAtPosition(source: []const u8, position: frontend.Position) ?WordAtPosition {
    const line_start = lineStartOffset(source, position.line) orelse return null;
    const cursor = @min(line_start + @as(usize, @intCast(position.character)), source.len);
    if (cursor >= source.len or source[cursor] != '@') return null;
    var end = cursor + 1;
    while (end < source.len and source[end] != '\n' and isIdentChar(source[end])) end += 1;
    if (end == cursor + 1) return null;
    return .{
        .text = source[cursor + 1 .. end],
        .range = .{
            .start = .{ .line = position.line, .character = @intCast(cursor - line_start) },
            .end = .{ .line = position.line, .character = @intCast(end - line_start) },
        },
    };
}

fn wordAtPosition(source: []const u8, position: frontend.Position) ?WordAtPosition {
    const line_start = lineStartOffset(source, position.line) orelse return null;
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

fn lineStartOffset(source: []const u8, target_line: u32) ?usize {
    var cursor: usize = 0;
    var line: u32 = 0;
    while (cursor < source.len and line < target_line) : (cursor += 1) {
        if (source[cursor] == '\n') line += 1;
    }
    if (line != target_line) return null;
    return cursor;
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
