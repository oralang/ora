const std = @import("std");
const lsp = @import("lsp");
const ora_root = @import("ora_root");
const builtin_docs = ora_root.lsp.builtin_docs;

const completion = ora_root.lsp.completion;
const frontend = ora_root.lsp.frontend;
const lexer = ora_root.lexer;
const keyword_docs = ora_root.lsp.keyword_docs;
const refinement_docs = ora_root.lsp.refinement_docs;
const semantic_index = ora_root.lsp.semantic_index;
const protocol_helpers = @import("protocol_helpers.zig");

const Allocator = std.mem.Allocator;
const types = lsp.types;

const Snippet = struct {
    label: []const u8,
    body: []const u8,
    detail: []const u8,
};

const snippets = [_]Snippet{
    .{ .label = "contract", .body = "contract ${1:Name} {\n\t$0\n}", .detail = "Contract declaration" },
    .{ .label = "fn", .body = "fn ${1:name}(${2}) -> ${3:u256} {\n\t$0\n}", .detail = "Function declaration" },
    .{ .label = "pub fn", .body = "pub fn ${1:name}(${2}) -> ${3:u256} {\n\t$0\n}", .detail = "Public function declaration" },
    .{ .label = "if", .body = "if (${1:condition}) {\n\t$0\n}", .detail = "If statement" },
    .{ .label = "while", .body = "while (${1:condition}) {\n\t$0\n}", .detail = "While loop" },
    .{ .label = "for", .body = "for (${1:item} in ${2:iterable}) {\n\t$0\n}", .detail = "For loop" },
    .{ .label = "struct", .body = "struct ${1:Name} {\n\t${2:field}: ${3:u256},\n}", .detail = "Struct declaration" },
    .{ .label = "requires", .body = "requires(${1:condition})", .detail = "Precondition clause" },
    .{ .label = "ensures", .body = "ensures(${1:condition})", .detail = "Postcondition clause" },
    .{ .label = "import", .body = "const ${1:name} = @import(\"${2:path}.ora\");", .detail = "Import declaration" },
    .{ .label = "storage", .body = "storage var ${1:name}: ${2:u256};", .detail = "Storage variable" },
    .{ .label = "event", .body = "log ${1:Name}(${2:param}: ${3:u256});", .detail = "Event/log declaration" },
};

pub const BuildResult = struct {
    items: []types.CompletionItem,
    string_bytes: usize,
    markdown_bytes: usize,
};

pub fn build(
    arena: Allocator,
    source: []const u8,
    byte_position: frontend.Position,
    trigger_char: ?[]const u8,
    items: []const completion.Item,
) ![]types.CompletionItem {
    return (try buildWithStats(arena, source, byte_position, trigger_char, items)).items;
}

pub fn buildWithStats(
    arena: Allocator,
    source: []const u8,
    byte_position: frontend.Position,
    trigger_char: ?[]const u8,
    items: []const completion.Item,
) !BuildResult {
    const snippet_count: usize = if (protocol_helpers.isAtLineStart(source, byte_position) and trigger_char == null) snippets.len else 0;
    const result = try arena.alloc(types.CompletionItem, items.len + snippet_count);
    var string_bytes: usize = 0;
    var markdown_bytes: usize = 0;

    for (items, 0..) |item, i| {
        string_bytes = addSat(string_bytes, item.label.len);
        if (item.detail) |detail| string_bytes = addSat(string_bytes, detail.len);
        if (item.documentation) |doc| markdown_bytes = addSat(markdown_bytes, doc.len);
        result[i] = .{
            .label = try arena.dupe(u8, item.label),
            .kind = protocol_helpers.completionKindToLsp(item.kind),
            .detail = if (item.detail) |detail| try arena.dupe(u8, detail) else null,
            .documentation = if (item.documentation) |doc| .{ .MarkupContent = .{
                .kind = .markdown,
                .value = try arena.dupe(u8, doc),
            } } else null,
        };
    }

    for (snippets[0..snippet_count], items.len..) |snippet, i| {
        string_bytes = addSat(string_bytes, snippet.label.len);
        string_bytes = addSat(string_bytes, snippet.detail.len);
        string_bytes = addSat(string_bytes, snippet.body.len);
        result[i] = .{
            .label = snippet.label,
            .kind = .Snippet,
            .detail = snippet.detail,
            .insertTextFormat = .Snippet,
            .insertText = snippet.body,
        };
    }

    return .{
        .items = result,
        .string_bytes = string_bytes,
        .markdown_bytes = markdown_bytes,
    };
}

pub fn buildFromSemanticIndexWithStats(
    arena: Allocator,
    source: []const u8,
    byte_position: frontend.Position,
    trigger_char: ?[]const u8,
    index: *const semantic_index.SemanticIndex,
) !BuildResult {
    const prefix = completion.identifierPrefixAtPosition(source, byte_position);
    const snippet_count: usize = if (protocol_helpers.isAtLineStart(source, byte_position) and trigger_char == null) snippets.len else 0;

    var result = std.ArrayList(types.CompletionItem).empty;
    errdefer result.deinit(arena);

    var string_bytes: usize = 0;
    var markdown_bytes: usize = 0;

    const keyword_keys = lexer.keywords.kvs.keys[0..lexer.keywords.kvs.len];
    for (keyword_keys) |keyword| {
        try appendBorrowedCompletion(arena, &result, &string_bytes, &markdown_bytes, prefix, keyword, null, keyword_docs.documentation(keyword), .keyword);
    }
    for (keyword_docs.contextual_keywords) |keyword| {
        try appendBorrowedCompletion(arena, &result, &string_bytes, &markdown_bytes, prefix, keyword, null, keyword_docs.documentation(keyword), .keyword);
    }
    try appendBorrowedCompletion(
        arena,
        &result,
        &string_bytes,
        &markdown_bytes,
        prefix,
        "Resource",
        "Resource<T>",
        "Opaque storage or transient resource place for a declared `resource` domain. Observe resource amounts with `@amount`; mutate places with `@create`, `@destroy`, and `@move`.",
        .type_alias,
    );

    for (builtin_docs.entries) |entry| {
        if (!matchesPrefix(entry.name, prefix)) continue;
        if (containsCompletionLabel(result.items, entry.name)) continue;

        const markdown = try builtin_docs.markdownAlloc(arena, entry);
        try appendBorrowedCompletion(
            arena,
            &result,
            &string_bytes,
            &markdown_bytes,
            prefix,
            entry.name,
            entry.signature,
            markdown,
            .function,
        );
    }

    for (completion.refinement_entries) |entry| {
        if (!matchesPrefix(entry.name, prefix)) continue;
        if (containsCompletionLabel(result.items, entry.name)) continue;

        const docs = refinement_docs.entryForName(entry.name);
        const markdown = if (docs) |doc| try refinement_docs.markdownAlloc(arena, doc) else null;
        try appendBorrowedCompletion(
            arena,
            &result,
            &string_bytes,
            &markdown_bytes,
            prefix,
            entry.name,
            if (docs) |doc| doc.signature else null,
            markdown,
            .type_alias,
        );
    }

    if (semantic_index.symbolIndexesWithNamePrefix(index, prefix)) |symbol_indexes| {
        for (symbol_indexes) |symbol_index| {
            const symbol = index.symbols[symbol_index];
            try appendBorrowedCompletion(
                arena,
                &result,
                &string_bytes,
                &markdown_bytes,
                prefix,
                symbol.name,
                symbol.detail,
                symbol.doc_comment,
                completion.symbolKindToCompletionKind(symbol.kind),
            );
        }
    } else {
        for (index.symbols) |symbol| {
            try appendBorrowedCompletion(
                arena,
                &result,
                &string_bytes,
                &markdown_bytes,
                prefix,
                symbol.name,
                symbol.detail,
                symbol.doc_comment,
                completion.symbolKindToCompletionKind(symbol.kind),
            );
        }
    }

    std.mem.sort(types.CompletionItem, result.items, {}, lessCompletionItemByLabel);
    for (snippets[0..snippet_count]) |snippet| {
        string_bytes = addSat(string_bytes, snippet.label.len);
        string_bytes = addSat(string_bytes, snippet.detail.len);
        string_bytes = addSat(string_bytes, snippet.body.len);
        try result.append(arena, .{
            .label = snippet.label,
            .kind = .Snippet,
            .detail = snippet.detail,
            .insertTextFormat = .Snippet,
            .insertText = snippet.body,
        });
    }

    return .{
        .items = try result.toOwnedSlice(arena),
        .string_bytes = string_bytes,
        .markdown_bytes = markdown_bytes,
    };
}

fn appendBorrowedCompletion(
    arena: Allocator,
    result: *std.ArrayList(types.CompletionItem),
    string_bytes: *usize,
    markdown_bytes: *usize,
    prefix: []const u8,
    label: []const u8,
    detail: ?[]const u8,
    documentation: ?[]const u8,
    kind: completion.Kind,
) !void {
    if (!matchesPrefix(label, prefix)) return;
    if (containsCompletionLabel(result.items, label)) return;

    string_bytes.* = addSat(string_bytes.*, label.len);
    if (detail) |value| string_bytes.* = addSat(string_bytes.*, value.len);
    if (documentation) |value| markdown_bytes.* = addSat(markdown_bytes.*, value.len);

    try result.append(arena, .{
        .label = label,
        .kind = protocol_helpers.completionKindToLsp(kind),
        .detail = detail,
        .documentation = if (documentation) |doc| .{ .MarkupContent = .{
            .kind = .markdown,
            .value = doc,
        } } else null,
    });
}

fn containsCompletionLabel(items: []const types.CompletionItem, label: []const u8) bool {
    for (items) |item| {
        if (std.mem.eql(u8, item.label, label)) return true;
    }
    return false;
}

fn matchesPrefix(candidate: []const u8, prefix: []const u8) bool {
    if (prefix.len == 0) return true;
    return std.mem.startsWith(u8, candidate, prefix);
}

fn lessCompletionItemByLabel(_: void, a: types.CompletionItem, b: types.CompletionItem) bool {
    return std.mem.lessThan(u8, a.label, b.label);
}

fn addSat(a: usize, b: usize) usize {
    return std.math.add(usize, a, b) catch std.math.maxInt(usize);
}
