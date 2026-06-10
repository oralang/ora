const std = @import("std");
const frontend = @import("frontend.zig");
const lexer = @import("ora_lexer");
const refinements = @import("ora_refinements");
const semantic_index = @import("semantic_index.zig");
const keyword_docs = @import("keyword_docs.zig");
const refinement_docs = @import("refinement_docs.zig");

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
    trait_decl,
    impl_decl,
    type_alias,
    event,
    error_decl,
};

pub const Item = struct {
    label: []u8,
    detail: ?[]u8 = null,
    documentation: ?[]u8 = null,
    kind: Kind,

    pub fn deinit(self: *Item, allocator: Allocator) void {
        allocator.free(self.label);
        if (self.detail) |detail| allocator.free(detail);
        if (self.documentation) |doc| allocator.free(doc);
    }
};

pub fn completionAtIndex(
    allocator: Allocator,
    source: []const u8,
    position: frontend.Position,
    trigger_char: ?[]const u8,
    index: *const semantic_index.SemanticIndex,
) ![]Item {
    // Dot-triggered member completion.
    if (isDotTrigger(trigger_char, source, position)) {
        return memberCompletion(allocator, source, position, index);
    }

    const prefix = identifierPrefixAtPosition(source, position);

    var seen = std.StringHashMap(void).init(allocator);
    defer seen.deinit();

    var items = std.ArrayList(Item){};
    errdefer {
        for (items.items) |*item| item.deinit(allocator);
        items.deinit(allocator);
    }

    const keyword_keys = lexer.keywords.kvs.keys[0..lexer.keywords.kvs.len];
    for (keyword_keys) |keyword| {
        try appendKeywordCompletion(allocator, &items, &seen, prefix, keyword);
    }
    for (keyword_docs.contextual_keywords) |keyword| {
        try appendKeywordCompletion(allocator, &items, &seen, prefix, keyword);
    }

    for (refinements.entries) |entry| {
        if (!matchesPrefix(entry.name, prefix)) continue;
        if (seen.contains(entry.name)) continue;

        const docs = refinement_docs.entryForName(entry.name);
        try seen.put(entry.name, {});
        try items.append(allocator, .{
            .label = try allocator.dupe(u8, entry.name),
            .detail = if (docs) |doc| try allocator.dupe(u8, doc.signature) else null,
            .documentation = if (docs) |doc| try refinement_docs.markdownAlloc(allocator, doc) else null,
            .kind = .type_alias,
        });
    }

    for (index.symbols) |symbol| {
        if (!matchesPrefix(symbol.name, prefix)) continue;
        if (seen.contains(symbol.name)) continue;

        try seen.put(symbol.name, {});
        try items.append(allocator, .{
            .label = try allocator.dupe(u8, symbol.name),
            .detail = if (symbol.detail) |detail| try allocator.dupe(u8, detail) else null,
            .documentation = if (symbol.doc_comment) |doc| try allocator.dupe(u8, doc) else null,
            .kind = symbolKindToCompletionKind(symbol.kind),
        });
    }

    std.sort.heap(Item, items.items, {}, lessItemByLabel);

    return items.toOwnedSlice(allocator);
}

fn appendKeywordCompletion(
    allocator: Allocator,
    items: *std.ArrayList(Item),
    seen: *std.StringHashMap(void),
    prefix: []const u8,
    keyword: []const u8,
) !void {
    if (!matchesPrefix(keyword, prefix)) return;
    if (seen.contains(keyword)) return;

    try seen.put(keyword, {});
    try items.append(allocator, .{
        .label = try allocator.dupe(u8, keyword),
        .documentation = if (keyword_docs.documentation(keyword)) |doc| try allocator.dupe(u8, doc) else null,
        .kind = .keyword,
    });
}

fn isDotTrigger(trigger_char: ?[]const u8, source: []const u8, position: frontend.Position) bool {
    if (trigger_char) |tc| {
        if (std.mem.eql(u8, tc, ".")) return true;
    }
    // Also check if the character just before cursor is a dot.
    const cursor = positionToByteOffsetOnLine(source, position);
    if (cursor > 0 and source[cursor - 1] == '.') return true;
    return false;
}

fn memberCompletion(
    allocator: Allocator,
    source: []const u8,
    position: frontend.Position,
    index: *const semantic_index.SemanticIndex,
) ![]Item {
    var items = std.ArrayList(Item){};
    errdefer {
        for (items.items) |*item| item.deinit(allocator);
        items.deinit(allocator);
    }

    // Extract the base name before the dot.
    const cursor = positionToByteOffsetOnLine(source, position);
    const base_name = extractBaseBeforeDot(source, cursor) orelse return items.toOwnedSlice(allocator);

    // Find the symbol matching the base name.
    const base_idx = findSymbolByName(index.symbols, base_name) orelse return items.toOwnedSlice(allocator);

    // Collect prefix after the dot for filtering.
    const prefix = identifierPrefixAtPosition(source, position);

    // If the base is a type (contract/struct/bitfield/enum), offer its children.
    const base_symbol = index.symbols[base_idx];
    switch (base_symbol.kind) {
        .contract, .struct_decl, .bitfield_decl, .enum_decl => {
            for (index.symbols) |symbol| {
                if (symbol.parent == null or symbol.parent.? != base_idx) continue;
                if (!matchesPrefix(symbol.name, prefix)) continue;

                try items.append(allocator, .{
                    .label = try allocator.dupe(u8, symbol.name),
                    .detail = if (symbol.detail) |detail| try allocator.dupe(u8, detail) else null,
                    .documentation = if (symbol.doc_comment) |doc| try allocator.dupe(u8, doc) else null,
                    .kind = symbolKindToCompletionKind(symbol.kind),
                });
            }
        },
        .variable, .field, .parameter, .constant => {
            if (base_symbol.detail) |type_name| {
                const trimmed = std.mem.trim(u8, type_name, " ");
                if (findSymbolByName(index.symbols, trimmed)) |type_idx| {
                    for (index.symbols) |symbol| {
                        if (symbol.parent == null or symbol.parent.? != type_idx) continue;
                        if (!matchesPrefix(symbol.name, prefix)) continue;

                        try items.append(allocator, .{
                            .label = try allocator.dupe(u8, symbol.name),
                            .detail = if (symbol.detail) |detail| try allocator.dupe(u8, detail) else null,
                            .documentation = if (symbol.doc_comment) |doc| try allocator.dupe(u8, doc) else null,
                            .kind = symbolKindToCompletionKind(symbol.kind),
                        });
                    }
                }
            }
        },
        else => {},
    }

    std.sort.heap(Item, items.items, {}, lessItemByLabel);
    return items.toOwnedSlice(allocator);
}

fn extractBaseBeforeDot(source: []const u8, cursor: usize) ?[]const u8 {
    if (cursor == 0) return null;

    // Walk back past the dot.
    var pos = cursor;
    // Skip any prefix typed after the dot.
    while (pos > 0 and isIdentifierContinue(source[pos - 1])) : (pos -= 1) {}
    // Now we should be at the dot.
    if (pos == 0 or source[pos - 1] != '.') return null;
    pos -= 1; // skip dot

    // Extract the identifier before the dot.
    var end = pos;
    while (end > 0 and (source[end - 1] == ' ' or source[end - 1] == '\t')) : (end -= 1) {}
    if (end == 0) return null;

    var start = end;
    while (start > 0 and isIdentifierContinue(source[start - 1])) : (start -= 1) {}

    if (start == end) return null;
    return source[start..end];
}

fn findSymbolByName(symbols: []const semantic_index.Symbol, name: []const u8) ?usize {
    for (symbols, 0..) |symbol, i| {
        if (std.mem.eql(u8, symbol.name, name)) {
            // Prefer type declarations.
            if (symbol.kind == .contract or symbol.kind == .struct_decl or
                symbol.kind == .bitfield_decl or symbol.kind == .enum_decl or
                symbol.kind == .trait_decl or symbol.kind == .type_alias or
                symbol.kind == .event or symbol.kind == .error_decl) return i;
        }
    }
    // Fallback: any symbol with matching name.
    for (symbols, 0..) |symbol, i| {
        if (std.mem.eql(u8, symbol.name, name)) return i;
    }
    return null;
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
        .trait_decl => .trait_decl,
        .impl_decl => .impl_decl,
        .type_alias => .type_alias,
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
