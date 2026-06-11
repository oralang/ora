const std = @import("std");
const lsp = @import("lsp");
const ora_root = @import("ora_root");

const completion_api = ora_root.lsp.completion;
const formatting_api = @import("formatting.zig");
const frontend = ora_root.lsp.frontend;
const hover_api = ora_root.lsp.hover;
const semantic_index = ora_root.lsp.semantic_index;
const std_docs_api = ora_root.lsp.std_docs;

const code_action_response = @import("code_action.zig");
const completion_items_response = @import("completion_items.zig");
const hover_response = @import("hover_response.zig");
const protocol_helpers = @import("protocol_helpers.zig");
const response_stats = @import("response_stats.zig");
const semantic_tokens_response = @import("semantic_tokens_response.zig");

const Allocator = std.mem.Allocator;
const types = lsp.types;

pub fn documentSymbol(
    server: anytype,
    _: Allocator,
    params: types.DocumentSymbolParams,
) !lsp.ResultType("textDocument/documentSymbol") {
    const uri = params.textDocument.uri;
    if (!server.supports_hierarchical_document_symbols) {
        const flat = (try server.docs.flatDocumentSymbolsForUri(
            uri,
            server.position_encoding,
            &server.phase_counters,
        )) orelse return null;

        server.response_counters.recordItems(.document_symbol, types.SymbolInformation, flat.items.len);
        server.response_counters.recordStringBytes(.document_symbol, flat.string_bytes);
        return .{ .array_of_SymbolInformation = flat.items };
    }

    const result = (try server.docs.documentSymbolsForUri(
        uri,
        server.position_encoding,
        &server.phase_counters,
    )) orelse return null;

    server.response_counters.recordItems(.document_symbol, types.DocumentSymbol, result.symbol_count);
    server.response_counters.recordStringBytes(.document_symbol, result.string_bytes);
    return .{ .array_of_DocumentSymbol = result.symbols };
}

pub fn hover(
    server: anytype,
    arena: Allocator,
    params: types.HoverParams,
) !?types.Hover {
    const uri = params.textDocument.uri;
    if (server.docs.cachedHoverForUri(uri, params.position, server.position_encoding)) |cached| {
        server.response_counters.recordItems(.hover, types.Hover, 1);
        server.response_counters.recordMarkdownBytes(.hover, cached.markdown_bytes);
        return cached.item;
    }
    if (try server.docs.hoverForUri(
        uri,
        params.position,
        server.position_encoding,
        &server.phase_counters,
    )) |cached| {
        server.response_counters.recordItems(.hover, types.Hover, 1);
        server.response_counters.recordMarkdownBytes(.hover, cached.markdown_bytes);
        return cached.item;
    }

    const source = server.docs.sourceForUri(uri) orelse return null;
    const line_index = (try server.docs.lineIndexForUri(uri)) orelse return null;
    const position: frontend.Position = .{
        .line = params.position.line,
        .character = params.position.character,
    };
    const byte_position = protocol_helpers.lspPositionToBytePosition(
        source,
        line_index,
        server.position_encoding,
        position,
    ) orelse return null;

    const maybe_hover = try stdHoverAt(server, arena, uri, source, byte_position);
    if (maybe_hover == null) return null;

    const hover_result = maybe_hover.?;
    var response_scope = server.responseScope();
    defer response_scope.deinit();
    const response = hover_response.build(source, line_index, server.position_encoding, hover_result) orelse return null;
    server.response_counters.recordItems(.hover, types.Hover, 1);
    server.response_counters.recordMarkdownBytes(.hover, hover_result.contents.len);
    return response;
}
pub fn completion(
    server: anytype,
    arena: Allocator,
    params: types.CompletionParams,
) !lsp.ResultType("textDocument/completion") {
    const uri = params.textDocument.uri;
    const trigger_char: ?[]const u8 = if (params.context) |ctx| ctx.triggerCharacter else null;
    if (trigger_char == null) {
        if (server.docs.cachedSemanticCompletionForUri(uri, params.position, server.position_encoding)) |result| {
            server.response_counters.recordItems(.completion_item, types.CompletionItem, result.items.len);
            server.response_counters.recordStringBytes(.completion_item, result.string_bytes);
            server.response_counters.recordMarkdownBytes(.completion_item, result.markdown_bytes);
            return .{ .array_of_CompletionItem = result.items };
        }
    }

    const source = server.docs.sourceForUri(uri) orelse return null;
    const line_index = (try server.docs.lineIndexForUri(uri)) orelse return null;
    const position: frontend.Position = .{
        .line = params.position.line,
        .character = params.position.character,
    };
    const byte_position = protocol_helpers.lspPositionToBytePosition(
        source,
        line_index,
        server.position_encoding,
        position,
    ) orelse return null;

    if (!completion_api.isDotTrigger(trigger_char, source, byte_position) and
        !std_docs_api.isCompletionAccessContext(source, byte_position))
    {
        if (trigger_char == null) {
            const result = (try server.docs.semanticCompletionForUri(
                uri,
                params.position,
                server.position_encoding,
                &server.phase_counters,
            )) orelse return null;
            server.response_counters.recordItems(.completion_item, types.CompletionItem, result.items.len);
            server.response_counters.recordStringBytes(.completion_item, result.string_bytes);
            server.response_counters.recordMarkdownBytes(.completion_item, result.markdown_bytes);
            return .{ .array_of_CompletionItem = result.items };
        } else {
            const index = (try server.docs.semanticIndexForUri(uri, &server.phase_counters)) orelse return null;
            var response_scope = server.responseScope();
            defer response_scope.deinit();
            const result = try completion_items_response.buildFromSemanticIndexWithStats(
                arena,
                source,
                byte_position,
                trigger_char,
                index,
            );
            server.response_counters.recordItems(.completion_item, types.CompletionItem, result.items.len);
            server.response_counters.recordStringBytes(.completion_item, result.string_bytes);
            server.response_counters.recordMarkdownBytes(.completion_item, result.markdown_bytes);
            return .{ .array_of_CompletionItem = result.items };
        }
    }

    const index = (try server.docs.semanticIndexForUri(uri, &server.phase_counters)) orelse return null;
    var temp_scope = server.tempAnalysisScope();
    defer temp_scope.deinit();
    const completion_items = try completion_api.completionAtIndex(server.allocator, source, byte_position, trigger_char, index);
    defer completion_api.deinitItems(server.allocator, completion_items);
    const std_completion_items = try stdCompletionItemsAt(server, uri, source, byte_position);
    defer completion_api.deinitItems(server.allocator, std_completion_items);
    const needs_merge = completion_items.len != 0 and std_completion_items.len != 0;
    const merged_items: []const completion_api.Item = if (std_completion_items.len == 0)
        completion_items
    else if (completion_items.len == 0)
        std_completion_items
    else
        try mergeCompletionItems(server.allocator, completion_items, std_completion_items);
    defer if (needs_merge) server.allocator.free(@constCast(merged_items));

    var response_scope = server.responseScope();
    defer response_scope.deinit();
    const result = try completion_items_response.buildWithStats(
        arena,
        source,
        byte_position,
        trigger_char,
        merged_items,
    );

    server.response_counters.recordItems(.completion_item, types.CompletionItem, result.items.len);
    server.response_counters.recordStringBytes(.completion_item, result.string_bytes);
    server.response_counters.recordMarkdownBytes(.completion_item, result.markdown_bytes);
    return .{ .array_of_CompletionItem = result.items };
}

fn stdHoverAt(
    server: anytype,
    allocator: Allocator,
    uri: []const u8,
    source: []const u8,
    byte_position: frontend.Position,
) !?hover_api.Hover {
    const aliases = (try server.docs.stdImportAliasesForUri(uri, &server.phase_counters)) orelse return null;
    if (aliases.len == 0) return null;

    const std_index = try server.stdDocsIndex();
    const maybe_hover = try std_docs_api.hoverAt(allocator, source, byte_position, std_index, aliases);
    if (maybe_hover == null) return null;

    const value = maybe_hover.?;
    return .{
        .contents = value.contents,
        .range = value.range,
    };
}

fn stdCompletionItemsAt(
    server: anytype,
    uri: []const u8,
    source: []const u8,
    byte_position: frontend.Position,
) ![]const completion_api.Item {
    if (!std_docs_api.isCompletionAccessContext(source, byte_position)) {
        return &.{};
    }

    const aliases = (try server.docs.stdImportAliasesForUri(uri, &server.phase_counters)) orelse
        return &.{};
    if (aliases.len == 0) return &.{};

    const std_index = try server.stdDocsIndex();
    const symbols = try std_docs_api.completionCandidatesAtPosition(
        server.allocator,
        source,
        byte_position,
        std_index,
        aliases,
    );
    defer server.allocator.free(symbols);

    const items = try server.allocator.alloc(completion_api.Item, symbols.len);
    var built: usize = 0;
    errdefer {
        for (items[0..built]) |*item| item.deinit(server.allocator);
        server.allocator.free(items);
    }

    for (symbols, 0..) |symbol, index| {
        items[index] = .{
            .label = try server.allocator.dupe(u8, symbol.name),
            .detail = if (symbol.detail) |detail| try server.allocator.dupe(u8, detail) else null,
            .documentation = if (symbol.doc_comment) |doc| try server.allocator.dupe(u8, doc) else null,
            .kind = stdSymbolKindToCompletionKind(symbol.kind),
        };
        built = index + 1;
    }

    return items;
}

fn mergeCompletionItems(
    allocator: Allocator,
    first: []const completion_api.Item,
    second: []const completion_api.Item,
) ![]const completion_api.Item {
    const merged = try allocator.alloc(completion_api.Item, first.len + second.len);
    @memcpy(merged[0..first.len], first);
    @memcpy(merged[first.len..], second);
    return merged;
}

fn stdSymbolKindToCompletionKind(kind: semantic_index.SymbolKind) completion_api.Kind {
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

pub fn inlayHint(
    server: anytype,
    _: Allocator,
    params: types.InlayHintParams,
) !lsp.ResultType("textDocument/inlayHint") {
    const uri = params.textDocument.uri;
    const range: frontend.Range = .{
        .start = .{ .line = params.range.start.line, .character = params.range.start.character },
        .end = .{ .line = params.range.end.line, .character = params.range.end.character },
    };
    const result = (try server.docs.inlayHintsForUri(
        uri,
        range,
        server.position_encoding,
        &server.phase_counters,
    )) orelse return null;
    if (result.items) |items| {
        server.response_counters.recordItems(.inlay_hint, types.InlayHint, items.len);
        server.response_counters.recordStringBytes(.inlay_hint, result.string_bytes);
    }
    return result.items;
}

pub fn codeLens(
    server: anytype,
    _: Allocator,
    params: types.CodeLensParams,
) !lsp.ResultType("textDocument/codeLens") {
    const uri = params.textDocument.uri;
    const result = (try server.docs.codeLensesForUri(
        uri,
        server.position_encoding,
        &server.phase_counters,
    )) orelse return null;
    if (result.items) |items| {
        server.response_counters.recordItems(.code_lens, types.CodeLens, items.len);
        server.response_counters.recordStringBytes(.code_lens, result.string_bytes);
    }
    return result.items;
}

pub fn signatureHelp(
    server: anytype,
    _: Allocator,
    params: types.SignatureHelpParams,
) !?types.SignatureHelp {
    const uri = params.textDocument.uri;
    const result = if (server.docs.cachedSignatureHelpForUri(uri, params.position, server.position_encoding)) |cached|
        cached
    else
        (try server.docs.signatureHelpForUri(
            uri,
            params.position,
            server.position_encoding,
            &server.phase_counters,
        )) orelse return null;
    server.response_counters.recordItems(.signature_help, types.SignatureInformation, result.signature_count);
    server.response_counters.recordItems(.signature_help, types.ParameterInformation, result.parameter_count);
    server.response_counters.recordStringBytes(.signature_help, result.string_bytes);
    server.response_counters.recordMarkdownBytes(.signature_help, result.markdown_bytes);
    return result.item;
}

pub fn semanticTokensFull(
    server: anytype,
    arena: Allocator,
    params: types.SemanticTokensParams,
) !lsp.ResultType("textDocument/semanticTokens/full") {
    _ = arena;
    const data = (try server.docs.semanticTokenDataForUri(params.textDocument.uri, &server.phase_counters)) orelse return null;

    server.response_counters.recordItems(.semantic_token_data, u32, data.len);
    return semantic_tokens_response.build(data);
}

pub fn formatting(
    server: anytype,
    _: Allocator,
    params: types.DocumentFormattingParams,
) !lsp.ResultType("textDocument/formatting") {
    const uri = params.textDocument.uri;
    const options: formatting_api.Options = .{
        .line_width = 100,
        .indent_size = protocol_helpers.normalizeIndentSize(params.options.tabSize),
    };
    const result = (server.docs.formattingEditsForUri(uri, options, server.position_encoding, &server.phase_counters) catch |err| switch (err) {
        error.ParseError => return null,
        else => return err,
    }) orelse return null;

    server.response_counters.recordItems(.formatting_edit, types.TextEdit, result.edits.len);
    if (result.string_bytes != 0) server.response_counters.recordStringBytes(.formatting_edit, result.string_bytes);
    return result.edits;
}

pub fn documentHighlight(
    server: anytype,
    _: Allocator,
    params: types.DocumentHighlightParams,
) !lsp.ResultType("textDocument/documentHighlight") {
    const uri = params.textDocument.uri;
    const result = (try server.docs.documentHighlightsForUri(
        uri,
        params.position,
        server.position_encoding,
        server.workspaceRootPaths(),
        &server.phase_counters,
    )) orelse return null;
    server.response_counters.recordItems(.document_highlight, types.DocumentHighlight, result.items.len);
    return result.items;
}

pub fn foldingRange(
    server: anytype,
    _: Allocator,
    params: types.FoldingRangeParams,
) !lsp.ResultType("textDocument/foldingRange") {
    const uri = params.textDocument.uri;
    const result = (try server.docs.foldingRangesForUri(uri, &server.phase_counters)) orelse return null;
    if (result.items.len == 0) return null;

    server.response_counters.recordItems(.folding_range, types.FoldingRange, result.items.len);
    return result.items;
}

pub fn codeAction(
    server: anytype,
    arena: Allocator,
    params: types.CodeActionParams,
) !lsp.ResultType("textDocument/codeAction") {
    const uri = params.textDocument.uri;
    if (server.docs.codeActionCacheForUri(uri, params.range, params.context.diagnostics)) |cached| {
        server.response_counters.recordItems(.code_action, code_action_response.CodeActionOrCommand, cached.actions.len);
        server.response_counters.recordStringBytes(.code_action, cached.string_bytes);
        return cached.actions;
    }

    var response_scope = server.responseScope();
    defer response_scope.deinit();
    const result = (try code_action_response.build(
        arena,
        uri,
        params.context.diagnostics,
    )) orelse return null;
    const string_bytes = response_stats.codeActionStringBytes(result);
    const cached = (try server.docs.cacheCodeActionsForUri(
        uri,
        params.range,
        params.context.diagnostics,
        result,
        string_bytes,
    )) orelse {
        server.response_counters.recordItems(.code_action, code_action_response.CodeActionOrCommand, result.len);
        server.response_counters.recordStringBytes(.code_action, string_bytes);
        return result;
    };
    server.response_counters.recordItems(.code_action, code_action_response.CodeActionOrCommand, cached.actions.len);
    server.response_counters.recordStringBytes(.code_action, cached.string_bytes);
    return cached.actions;
}

pub fn selectionRange(
    server: anytype,
    _: Allocator,
    params: types.SelectionRangeParams,
) !lsp.ResultType("textDocument/selectionRange") {
    const uri = params.textDocument.uri;
    const result = (try server.docs.selectionRangesForUri(
        uri,
        params.positions,
        server.position_encoding,
        &server.phase_counters,
    )) orelse return null;
    server.response_counters.recordItems(.selection_range, types.SelectionRange, result.node_count);
    return result.items;
}

pub fn documentLink(
    server: anytype,
    _: Allocator,
    params: types.DocumentLinkParams,
) !lsp.ResultType("textDocument/documentLink") {
    const uri = params.textDocument.uri;
    const result = (try server.docs.documentLinksForUri(
        uri,
        server.position_encoding,
        server.workspaceRootPaths(),
        &server.phase_counters,
    )) orelse return null;
    server.response_counters.recordItems(.document_link, types.DocumentLink, result.links.len);
    server.response_counters.recordStringBytes(.document_link, result.string_bytes);
    return result.links;
}

fn addSat(a: usize, b: usize) usize {
    return std.math.add(usize, a, b) catch std.math.maxInt(usize);
}
