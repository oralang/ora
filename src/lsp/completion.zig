const std = @import("std");
const frontend = @import("frontend.zig");
const semantic_index = @import("semantic_index.zig");

const Allocator = std.mem.Allocator;

pub const Kind = enum {
    keyword,
    contract,
    function,
    method,
    variable,
    field,
    constant,
    parameter,
    struct_decl,
    bitfield_decl,
    enum_decl,
    enum_member,
    event,
    error_decl,
};

pub const Item = struct {
    label: []u8,
    detail: ?[]u8 = null,
    kind: Kind,

    pub fn deinit(self: *Item, allocator: Allocator) void {
        allocator.free(self.label);
        if (self.detail) |detail| allocator.free(detail);
    }
};

const keyword_candidates = [_][]const u8{
    "contract",
    "struct",
    "enum",
    "bitfield",
    "pub",
    "fn",
    "let",
    "var",
    "const",
    "storage",
    "if",
    "else",
    "while",
    "for",
    "return",
    "import",
    "log",
    "error",
    "true",
    "false",
};

pub fn completionAt(
    allocator: Allocator,
    source: []const u8,
    position: frontend.Position,
) ![]Item {
    const prefix = identifierPrefixAtPosition(source, position);

    var seen = std.StringHashMap(void).init(allocator);
    defer seen.deinit();

    var items = std.ArrayList(Item){};
    errdefer {
        for (items.items) |*item| item.deinit(allocator);
        items.deinit(allocator);
    }

    for (keyword_candidates) |keyword| {
        if (!matchesPrefix(keyword, prefix)) continue;
        if (seen.contains(keyword)) continue;

        try seen.put(keyword, {});
        try items.append(allocator, .{
            .label = try allocator.dupe(u8, keyword),
            .kind = .keyword,
        });
    }

    var index = try semantic_index.indexDocument(allocator, source);
    defer index.deinit(allocator);

    for (index.symbols) |symbol| {
        if (!matchesPrefix(symbol.name, prefix)) continue;
        if (seen.contains(symbol.name)) continue;

        try seen.put(symbol.name, {});
        try items.append(allocator, .{
            .label = try allocator.dupe(u8, symbol.name),
            .detail = if (symbol.detail) |detail| try allocator.dupe(u8, detail) else null,
            .kind = symbolKindToCompletionKind(symbol.kind),
        });
    }

    std.sort.heap(Item, items.items, {}, lessItemByLabel);

    return items.toOwnedSlice(allocator);
}

pub fn deinitItems(allocator: Allocator, items: []Item) void {
    for (items) |*item| item.deinit(allocator);
    allocator.free(items);
}

fn symbolKindToCompletionKind(kind: semantic_index.SymbolKind) Kind {
    return switch (kind) {
        .contract => .contract,
        .function => .function,
        .method => .method,
        .variable => .variable,
        .field => .field,
        .constant => .constant,
        .parameter => .parameter,
        .struct_decl => .struct_decl,
        .bitfield_decl => .bitfield_decl,
        .enum_decl => .enum_decl,
        .enum_member => .enum_member,
        .event => .event,
        .error_decl => .error_decl,
    };
}

fn matchesPrefix(candidate: []const u8, prefix: []const u8) bool {
    if (prefix.len == 0) return true;
    return std.mem.startsWith(u8, candidate, prefix);
}

fn lessItemByLabel(_: void, a: Item, b: Item) bool {
    return std.mem.lessThan(u8, a.label, b.label);
}

fn identifierPrefixAtPosition(source: []const u8, position: frontend.Position) []const u8 {
    const cursor = positionToByteOffsetOnLine(source, position);

    var start = cursor;
    while (start > 0 and source[start - 1] != '\n' and isIdentifierContinue(source[start - 1])) {
        start -= 1;
    }

    return source[start..cursor];
}

fn positionToByteOffsetOnLine(source: []const u8, position: frontend.Position) usize {
    var cursor: usize = 0;
    var current_line: u32 = 0;

    while (cursor < source.len and current_line < position.line) : (cursor += 1) {
        if (source[cursor] == '\n') current_line += 1;
    }

    if (current_line != position.line) return source.len;

    const line_start = cursor;
    while (cursor < source.len and source[cursor] != '\n') : (cursor += 1) {}
    const line_end = cursor;

    const requested_col: usize = @intCast(position.character);
    return @min(line_start + requested_col, line_end);
}

fn isIdentifierStart(ch: u8) bool {
    return (ch >= 'a' and ch <= 'z') or
        (ch >= 'A' and ch <= 'Z') or
        ch == '_';
}

fn isIdentifierContinue(ch: u8) bool {
    return isIdentifierStart(ch) or (ch >= '0' and ch <= '9');
}
