const std = @import("std");
const lsp = @import("lsp");
const ora_root = @import("ora_root");

const code_lens_api = ora_root.lsp.code_lens;
const completion_api = ora_root.lsp.completion;
const folding_api = ora_root.lsp.folding;
const formatting_api = @import("formatting.zig");
const frontend = ora_root.lsp.frontend;
const hover_api = ora_root.lsp.hover;
const inlay_hints_api = ora_root.lsp.inlay_hints;
const semantic_index = ora_root.lsp.semantic_index;
const signature_help_api = ora_root.lsp.signature_help;
const std_docs_api = ora_root.lsp.std_docs;

const code_action_response = @import("code_action.zig");
const code_lens_response = @import("code_lens_response.zig");
const completion_items_response = @import("completion_items.zig");
const document_highlight_response = @import("document_highlight.zig");
const document_link_response = @import("document_link.zig");
const document_symbol_response = @import("document_symbol.zig");
const folding_ranges_response = @import("folding_ranges_response.zig");
const formatting_edits = @import("formatting_edits.zig");
const hover_response = @import("hover_response.zig");
const inlay_hint_response = @import("inlay_hint_response.zig");
const protocol_helpers = @import("protocol_helpers.zig");
const response_stats = @import("response_stats.zig");
const selection_range_response = @import("selection_range.zig");
const semantic_tokens_response = @import("semantic_tokens_response.zig");
const signature_help_response = @import("signature_help_response.zig");

const Allocator = std.mem.Allocator;
const types = lsp.types;

pub fn documentSymbol(
    server: anytype,
    arena: Allocator,
    params: types.DocumentSymbolParams,
) !lsp.ResultType("textDocument/documentSymbol") {
    const uri = params.textDocument.uri;
    const source = server.docs.sourceForUri(uri) orelse return null;
    const line_index = (try server.docs.lineIndexForUri(uri)) orelse return null;
    const index = (try server.docs.semanticIndexForUri(uri, &server.phase_counters)) orelse return null;

    var response_scope = server.responseScope();
    defer response_scope.deinit();

    const result = try document_symbol_response.build(
        arena,
        source,
        line_index,
        server.position_encoding,
        index,
    );

    server.response_counters.recordItems(.document_symbol, types.DocumentSymbol, index.symbols.len);
    server.response_counters.recordStringBytes(.document_symbol, response_stats.semanticSymbolStringBytes(index.symbols));
    return .{ .array_of_DocumentSymbol = result };
}

pub fn hover(
    server: anytype,
    arena: Allocator,
    params: types.HoverParams,
) !?types.Hover {
    const uri = params.textDocument.uri;
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
    const index = (try server.docs.semanticIndexForUri(uri, &server.phase_counters)) orelse return null;

    var temp_scope = server.tempAnalysisScope();
    defer temp_scope.deinit();
    var maybe_hover = try hover_api.hoverAtIndex(server.allocator, source, byte_position, index);
    if (maybe_hover == null) {
        maybe_hover = try stdHoverAt(server, uri, source, byte_position);
    }
    if (maybe_hover == null) return null;
    defer maybe_hover.?.deinit(server.allocator);

    const hover_result = maybe_hover.?;
    var response_scope = server.responseScope();
    defer response_scope.deinit();
    const arena_hover: hover_api.Hover = .{
        .contents = try arena.dupe(u8, hover_result.contents),
        .range = hover_result.range,
    };
    const response = hover_response.build(source, line_index, server.position_encoding, arena_hover) orelse return null;
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
    const index = (try server.docs.semanticIndexForUri(uri, &server.phase_counters)) orelse return null;

    const trigger_char: ?[]const u8 = if (params.context) |ctx| ctx.triggerCharacter else null;
    var temp_scope = server.tempAnalysisScope();
    defer temp_scope.deinit();
    const completion_items = try completion_api.completionAtIndex(server.allocator, source, byte_position, trigger_char, index);
    defer completion_api.deinitItems(server.allocator, completion_items);
    const std_completion_items = try stdCompletionItemsAt(server, uri, source, byte_position);
    defer completion_api.deinitItems(server.allocator, std_completion_items);
    const merged_items = try mergeCompletionItems(server.allocator, completion_items, std_completion_items);
    defer server.allocator.free(merged_items);

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
    uri: []const u8,
    source: []const u8,
    byte_position: frontend.Position,
) !?hover_api.Hover {
    const aliases = (try server.docs.stdImportAliasesForUri(uri, &server.phase_counters)) orelse return null;
    if (aliases.len == 0) return null;

    const std_index = try server.stdDocsIndex();
    const maybe_hover = try std_docs_api.hoverAt(server.allocator, source, byte_position, std_index, aliases);
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
) ![]completion_api.Item {
    const aliases = (try server.docs.stdImportAliasesForUri(uri, &server.phase_counters)) orelse
        return server.allocator.alloc(completion_api.Item, 0);
    if (aliases.len == 0) return server.allocator.alloc(completion_api.Item, 0);

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
) ![]completion_api.Item {
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
    arena: Allocator,
    params: types.InlayHintParams,
) !lsp.ResultType("textDocument/inlayHint") {
    const uri = params.textDocument.uri;
    const source = server.docs.sourceForUri(uri) orelse return null;
    const line_index = (try server.docs.lineIndexForUri(uri)) orelse return null;
    const ast_file = (try server.docs.astFileForUri(uri, &server.phase_counters)) orelse return null;
    const index = (try server.docs.semanticIndexForUri(uri, &server.phase_counters)) orelse return null;
    const range: frontend.Range = .{
        .start = .{ .line = params.range.start.line, .character = params.range.start.character },
        .end = .{ .line = params.range.end.line, .character = params.range.end.character },
    };

    var temp_scope = server.tempAnalysisScope();
    defer temp_scope.deinit();
    const hints = try inlay_hints_api.hintsInRangeCached(
        server.allocator,
        source,
        range,
        line_index,
        server.position_encoding,
        ast_file,
        index,
    );
    defer inlay_hints_api.deinitHints(server.allocator, hints);

    if (hints.len == 0) return null;

    var response_scope = server.responseScope();
    defer response_scope.deinit();
    const result = try inlay_hint_response.buildWithStats(arena, hints);
    if (result.items) |items| {
        server.response_counters.recordItems(.inlay_hint, types.InlayHint, items.len);
        server.response_counters.recordStringBytes(.inlay_hint, result.string_bytes);
    }
    return result.items;
}

pub fn codeLens(
    server: anytype,
    arena: Allocator,
    params: types.CodeLensParams,
) !lsp.ResultType("textDocument/codeLens") {
    const uri = params.textDocument.uri;
    const source = server.docs.sourceForUri(uri) orelse return null;
    const line_index = (try server.docs.lineIndexForUri(uri)) orelse return null;
    const ast_file = (try server.docs.astFileForUri(uri, &server.phase_counters)) orelse return null;

    var temp_scope = server.tempAnalysisScope();
    defer temp_scope.deinit();
    const lenses = try code_lens_api.findVerificationLensesInAst(server.allocator, source, ast_file, line_index);
    defer code_lens_api.deinitLenses(server.allocator, lenses);

    var response_scope = server.responseScope();
    defer response_scope.deinit();
    const result = try code_lens_response.buildWithStats(arena, source, line_index, server.position_encoding, lenses);
    if (result.items) |items| {
        server.response_counters.recordItems(.code_lens, types.CodeLens, items.len);
        server.response_counters.recordStringBytes(.code_lens, result.string_bytes);
    }
    return result.items;
}

pub fn signatureHelp(
    server: anytype,
    arena: Allocator,
    params: types.SignatureHelpParams,
) !?types.SignatureHelp {
    const uri = params.textDocument.uri;
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
    const index = (try server.docs.semanticIndexForUri(uri, &server.phase_counters)) orelse return null;

    var temp_scope = server.tempAnalysisScope();
    defer temp_scope.deinit();
    var sig_val = (signature_help_api.signatureAtIndex(server.allocator, source, byte_position, index) catch return null) orelse return null;
    defer sig_val.deinit(server.allocator);

    var response_scope = server.responseScope();
    defer response_scope.deinit();
    const result = try signature_help_response.build(arena, sig_val);
    var string_bytes: usize = sig_val.label.len;
    for (sig_val.parameters) |parameter| {
        string_bytes = addSat(string_bytes, parameter.label.len);
    }
    server.response_counters.recordItems(.signature_help, types.SignatureInformation, 1);
    server.response_counters.recordItems(.signature_help, types.ParameterInformation, sig_val.parameters.len);
    server.response_counters.recordStringBytes(.signature_help, string_bytes);
    if (sig_val.documentation) |doc| server.response_counters.recordMarkdownBytes(.signature_help, doc.len);
    return result;
}

pub fn semanticTokensFull(
    server: anytype,
    arena: Allocator,
    params: types.SemanticTokensParams,
) !lsp.ResultType("textDocument/semanticTokens/full") {
    const tokens = (try server.docs.semanticTokensForUri(params.textDocument.uri, &server.phase_counters)) orelse return null;

    var response_scope = server.responseScope();
    defer response_scope.deinit();
    const result = try semantic_tokens_response.buildWithStats(arena, tokens);

    server.response_counters.recordItems(.semantic_token_data, u32, result.data_count);
    return result.response;
}

pub fn formatting(
    server: anytype,
    arena: Allocator,
    params: types.DocumentFormattingParams,
) !lsp.ResultType("textDocument/formatting") {
    const uri = params.textDocument.uri;
    const source = server.docs.sourceForUri(uri) orelse return null;
    const options: formatting_api.Options = .{
        .line_width = 100,
        .indent_size = protocol_helpers.normalizeIndentSize(params.options.tabSize),
    };
    const formatted = (server.docs.formattingCacheForUri(uri, options, &server.phase_counters) catch |err| switch (err) {
        error.ParseError => return null,
        else => return err,
    }) orelse return null;

    var response_scope = server.responseScope();
    defer response_scope.deinit();
    const edits = try formatting_edits.buildFullDocumentEdit(arena, source, formatted, server.position_encoding);
    server.response_counters.recordItems(.formatting_edit, types.TextEdit, edits.len);
    if (edits.len != 0) server.response_counters.recordStringBytes(.formatting_edit, formatted.len);
    return edits;
}

pub fn documentHighlight(
    server: anytype,
    arena: Allocator,
    params: types.DocumentHighlightParams,
) !lsp.ResultType("textDocument/documentHighlight") {
    const uri = params.textDocument.uri;
    const source = server.docs.sourceForUri(uri) orelse return null;
    const workspace_entry = (try server.docs.workspaceEntryForUri(uri, .{ .occurrences = true }, server.workspaceRootPaths(), &server.phase_counters)) orelse return null;
    const occurrence_index = workspace_entry.occurrenceIndex();
    const position: frontend.Position = .{
        .line = params.position.line,
        .character = params.position.character,
    };

    var response_scope = server.responseScope();
    defer response_scope.deinit();
    const result = try document_highlight_response.build(arena, source, &workspace_entry.line_index, server.position_encoding, &occurrence_index, position);
    if (result) |items| {
        server.response_counters.recordItems(.document_highlight, types.DocumentHighlight, items.len);
    }
    return result;
}

pub fn foldingRange(
    server: anytype,
    arena: Allocator,
    params: types.FoldingRangeParams,
) !lsp.ResultType("textDocument/foldingRange") {
    const uri = params.textDocument.uri;
    const source = server.docs.sourceForUri(uri) orelse return null;
    const line_index = (try server.docs.lineIndexForUri(uri)) orelse return null;
    const ast_file = (try server.docs.astFileForUri(uri, &server.phase_counters)) orelse return null;

    var temp_scope = server.tempAnalysisScope();
    defer temp_scope.deinit();
    const ranges = try folding_api.foldingRangesInAst(server.allocator, source, ast_file, line_index);
    defer folding_api.deinitRanges(server.allocator, ranges);

    var response_scope = server.responseScope();
    defer response_scope.deinit();
    const result = try folding_ranges_response.buildWithStats(arena, ranges);
    if (result.items != null) {
        server.response_counters.recordItems(.folding_range, types.FoldingRange, result.item_count);
    }
    return result.items;
}

pub fn codeAction(
    server: anytype,
    arena: Allocator,
    params: types.CodeActionParams,
) !lsp.ResultType("textDocument/codeAction") {
    var response_scope = server.responseScope();
    defer response_scope.deinit();
    const result = (try code_action_response.build(
        arena,
        params.textDocument.uri,
        params.context.diagnostics,
    )) orelse return null;
    server.response_counters.recordItems(.code_action, code_action_response.CodeActionOrCommand, result.len);
    server.response_counters.recordStringBytes(.code_action, response_stats.codeActionStringBytes(result));
    return result;
}

pub fn selectionRange(
    server: anytype,
    arena: Allocator,
    params: types.SelectionRangeParams,
) !lsp.ResultType("textDocument/selectionRange") {
    const uri = params.textDocument.uri;
    const source = server.docs.sourceForUri(uri) orelse return null;
    const line_index = (try server.docs.lineIndexForUri(uri)) orelse return null;
    const ast_file = (try server.docs.astFileForUri(uri, &server.phase_counters)) orelse return null;

    var response_scope = server.responseScope();
    defer response_scope.deinit();
    const result = try selection_range_response.build(
        arena,
        source,
        line_index,
        server.position_encoding,
        ast_file,
        params.positions,
    );
    server.response_counters.recordItems(.selection_range, types.SelectionRange, response_stats.selectionRangeNodeCount(result));
    return result;
}

pub fn documentLink(
    server: anytype,
    arena: Allocator,
    params: types.DocumentLinkParams,
) !lsp.ResultType("textDocument/documentLink") {
    const uri = params.textDocument.uri;
    const source = server.docs.sourceForUri(uri) orelse return null;
    const line_index = (try server.docs.lineIndexForUri(uri)) orelse return null;
    const resolution = (try server.docs.importResolutionForUri(uri, server.workspaceRootPaths(), &server.phase_counters)) orelse return null;

    var response_scope = server.responseScope();
    defer response_scope.deinit();
    const result = (try document_link_response.build(
        arena,
        source,
        line_index,
        server.position_encoding,
        resolution.imports,
    )) orelse return null;
    server.response_counters.recordItems(.document_link, types.DocumentLink, result.len);
    server.response_counters.recordStringBytes(.document_link, response_stats.documentLinkStringBytes(result));
    return result;
}

fn addSat(a: usize, b: usize) usize {
    return std.math.add(usize, a, b) catch std.math.maxInt(usize);
}
