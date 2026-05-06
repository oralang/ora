const std = @import("std");
const lsp = @import("lsp");
const types = lsp.types;
const ora_root = @import("ora_root");

const lexer_mod = ora_root.lexer;
const frontend = ora_root.lsp.frontend;
const workspace = ora_root.lsp.workspace;
const dependency_graph = ora_root.lsp.dependency_graph;
const semantic_index = ora_root.lsp.semantic_index;
const text_edits = ora_root.lsp.text_edits;
const hover_api = ora_root.lsp.hover;
const definition_api = ora_root.lsp.definition;
const references_api = ora_root.lsp.references;
const rename_api = ora_root.lsp.rename;
const completion_api = ora_root.lsp.completion;
const semantic_tokens_api = ora_root.lsp.semantic_tokens;
const signature_help_api = ora_root.lsp.signature_help;
const code_lens_api = ora_root.lsp.code_lens;
const inlay_hints_api = ora_root.lsp.inlay_hints;
const folding_api = ora_root.lsp.folding;
const formatting_api = @import("formatting.zig");
const compiler = ora_root.compiler;
const Allocator = std.mem.Allocator;

const DocumentStore = struct {
    allocator: Allocator,
    docs: std.StringHashMap([]u8),

    fn init(allocator: Allocator) DocumentStore {
        return .{
            .allocator = allocator,
            .docs = std.StringHashMap([]u8).init(allocator),
        };
    }

    fn deinit(self: *DocumentStore) void {
        var it = self.docs.iterator();
        while (it.next()) |entry| {
            self.allocator.free(entry.key_ptr.*);
            self.allocator.free(entry.value_ptr.*);
        }
        self.docs.deinit();
    }

    fn put(self: *DocumentStore, uri: []const u8, text: []const u8) !void {
        const uri_copy = try self.allocator.dupe(u8, uri);
        errdefer self.allocator.free(uri_copy);

        const text_copy = try self.allocator.dupe(u8, text);
        errdefer self.allocator.free(text_copy);

        if (self.docs.fetchRemove(uri)) |removed| {
            self.allocator.free(removed.key);
            self.allocator.free(removed.value);
        }

        try self.docs.put(uri_copy, text_copy);
    }

    fn remove(self: *DocumentStore, uri: []const u8) void {
        if (self.docs.fetchRemove(uri)) |removed| {
            self.allocator.free(removed.key);
            self.allocator.free(removed.value);
        }
    }
};

pub const Handler = struct {
    allocator: Allocator,
    transport: *lsp.Transport,
    docs: DocumentStore,
    workspace_roots: std.ArrayList([]const u8),
    dependencies: dependency_graph.Graph,
    position_encoding: text_edits.PositionEncoding,

    fn init(allocator: Allocator, transport: *lsp.Transport) Handler {
        return .{
            .allocator = allocator,
            .transport = transport,
            .docs = DocumentStore.init(allocator),
            .workspace_roots = .{},
            .dependencies = dependency_graph.Graph.init(allocator),
            .position_encoding = .utf16,
        };
    }

    fn deinit(self: *Handler) void {
        self.docs.deinit();
        for (self.workspace_roots.items) |root| {
            self.allocator.free(root);
        }
        self.workspace_roots.deinit(self.allocator);
        self.dependencies.deinit();
    }

    pub fn initialize(
        self: *Handler,
        _: Allocator,
        params: types.InitializeParams,
    ) !types.InitializeResult {
        try self.configureWorkspaceRoots(params);
        self.position_encoding = negotiatePositionEncoding(params);

        const capabilities: types.ServerCapabilities = .{
            .positionEncoding = toLspPositionEncoding(self.position_encoding),
            .textDocumentSync = .{
                .TextDocumentSyncOptions = .{
                    .openClose = true,
                    .change = .Incremental,
                    .save = .{ .SaveOptions = .{ .includeText = false } },
                },
            },
            .hoverProvider = .{ .bool = true },
            .definitionProvider = .{ .bool = true },
            .referencesProvider = .{ .bool = true },
            .documentSymbolProvider = .{ .bool = true },
            .completionProvider = .{
                .triggerCharacters = &[_][]const u8{ ".", ":" },
                .resolveProvider = false,
            },
            .renameProvider = .{ .RenameOptions = .{ .prepareProvider = true } },
            .documentHighlightProvider = .{ .bool = true },
            .foldingRangeProvider = .{ .bool = true },
            .workspaceSymbolProvider = .{ .bool = true },
            .codeActionProvider = .{ .CodeActionOptions = .{
                .codeActionKinds = &.{ .quickfix, .@"source.organizeImports" },
            } },
            .selectionRangeProvider = .{ .bool = true },
            .documentLinkProvider = .{ .resolveProvider = false },
            .callHierarchyProvider = .{ .bool = true },
            .documentFormattingProvider = .{ .bool = true },
            .inlayHintProvider = .{ .InlayHintOptions = .{ .resolveProvider = false } },
            .codeLensProvider = .{ .resolveProvider = false },
            .signatureHelpProvider = .{
                .triggerCharacters = &[_][]const u8{ "(", "," },
                .retriggerCharacters = &[_][]const u8{","},
            },
            .semanticTokensProvider = .{
                .SemanticTokensOptions = .{
                    .legend = .{
                        .tokenTypes = &semantic_tokens_api.SemanticTokenKind.legend,
                        .tokenModifiers = &semantic_tokens_api.SemanticTokenModifier.legend,
                    },
                    .full = .{ .bool = true },
                },
            },
        };

        if (@import("builtin").mode == .Debug) {
            lsp.basic_server.validateServerCapabilities(Handler, capabilities);
        }

        return .{
            .capabilities = capabilities,
            .serverInfo = .{
                .name = "ora-lsp",
                .version = "0.1.0",
            },
        };
    }

    pub fn initialized(_: *Handler, _: Allocator, _: types.InitializedParams) void {}

    pub fn shutdown(_: *Handler, _: Allocator, _: void) ?void {
        return null;
    }

    pub fn exit(_: *Handler, _: Allocator, _: void) void {}

    pub fn onResponse(_: *Handler, _: Allocator, _: lsp.JsonRPCMessage.Response) void {}

    pub fn @"textDocument/didOpen"(self: *Handler, arena: Allocator, notification: types.DidOpenTextDocumentParams) !void {
        const uri = notification.textDocument.uri;
        const text = notification.textDocument.text;
        try self.docs.put(uri, text);
        try self.updateDocumentDependencies(uri, text);
        try self.publishDiagnostics(arena, uri, text);

        if (self.dependencies.getPathForUri(uri)) |changed_path| {
            try self.publishDependentsDiagnostics(arena, changed_path, uri);
        }
    }

    pub fn @"textDocument/didChange"(self: *Handler, arena: Allocator, notification: types.DidChangeTextDocumentParams) !void {
        const uri = notification.textDocument.uri;
        const changes = notification.contentChanges;
        if (changes.len == 0) return;

        const current = self.docs.docs.get(uri) orelse return;
        const edit_changes = try self.allocator.alloc(text_edits.Change, changes.len);
        defer self.allocator.free(edit_changes);

        for (changes, 0..) |change, i| {
            edit_changes[i] = switch (change) {
                .literal_1 => |full| .{ .full = .{ .text = full.text } },
                .literal_0 => |inc| .{ .incremental = .{
                    .range = .{
                        .start = .{ .line = inc.range.start.line, .character = inc.range.start.character },
                        .end = .{ .line = inc.range.end.line, .character = inc.range.end.character },
                    },
                    .text = inc.text,
                } },
            };
        }

        const updated = text_edits.applyChangesAllocWithEncoding(self.allocator, current, edit_changes, self.position_encoding) catch |err| switch (err) {
            error.InvalidRange => {
                if (lastFullText(changes)) |full_text| {
                    try self.docs.put(uri, full_text);
                    try self.updateDocumentDependencies(uri, full_text);
                    try self.publishDiagnostics(arena, uri, full_text);
                }
                return;
            },
            else => return err,
        };
        defer self.allocator.free(updated);

        try self.docs.put(uri, updated);
        try self.updateDocumentDependencies(uri, updated);
        try self.publishDiagnostics(arena, uri, updated);

        if (self.dependencies.getPathForUri(uri)) |changed_path| {
            try self.publishDependentsDiagnostics(arena, changed_path, uri);
        }
    }

    pub fn @"textDocument/didClose"(self: *Handler, arena: Allocator, notification: types.DidCloseTextDocumentParams) !void {
        const uri = notification.textDocument.uri;

        const removed_path = try self.dependencies.remove(uri);
        defer if (removed_path) |path| self.allocator.free(path);

        self.docs.remove(uri);
        try self.publishDiagnostics(arena, uri, "");

        if (removed_path) |path| {
            try self.publishDependentsDiagnostics(arena, path, null);
        }
    }

    pub fn @"textDocument/didSave"(self: *Handler, arena: Allocator, notification: types.DidSaveTextDocumentParams) !void {
        const uri = notification.textDocument.uri;
        const source = self.docs.docs.get(uri) orelse return;
        try self.updateDocumentDependencies(uri, source);
        try self.publishDiagnostics(arena, uri, source);

        if (self.dependencies.getPathForUri(uri)) |changed_path| {
            try self.publishDependentsDiagnostics(arena, changed_path, uri);
        }
    }

    pub fn @"textDocument/documentSymbol"(self: *Handler, arena: Allocator, params: types.DocumentSymbolParams) !lsp.ResultType("textDocument/documentSymbol") {
        const source = self.docs.docs.get(params.textDocument.uri) orelse return null;

        var index = try semantic_index.indexDocument(self.allocator, source);
        defer index.deinit(self.allocator);

        const symbols = try semantic_index.buildDocumentSymbols(self.allocator, index.symbols);
        defer semantic_index.deinitDocumentSymbols(self.allocator, symbols);

        const result = try arena.alloc(types.DocumentSymbol, symbols.len);
        for (symbols, 0..) |symbol, i| {
            result[i] = convertDocumentSymbol(arena, symbol) catch return null;
        }

        return .{ .array_of_DocumentSymbol = result };
    }

    pub fn @"textDocument/hover"(self: *Handler, arena: Allocator, params: types.HoverParams) !?types.Hover {
        const source = self.docs.docs.get(params.textDocument.uri) orelse return null;
        const position: frontend.Position = .{
            .line = params.position.line,
            .character = params.position.character,
        };

        var maybe_hover = try hover_api.hoverAt(self.allocator, source, position);
        if (maybe_hover == null) return null;
        defer maybe_hover.?.deinit(self.allocator);

        const hover = maybe_hover.?;
        return .{
            .contents = .{ .MarkupContent = .{
                .kind = .markdown,
                .value = try arena.dupe(u8, hover.contents),
            } },
            .range = toLspRange(hover.range),
        };
    }

    pub fn @"textDocument/definition"(self: *Handler, _: Allocator, params: types.DefinitionParams) !lsp.ResultType("textDocument/definition") {
        const uri = params.textDocument.uri;
        const source = self.docs.docs.get(uri) orelse return null;
        const position: frontend.Position = .{
            .line = params.position.line,
            .character = params.position.character,
        };

        const cross_file = try self.buildCrossFileContext(uri, source);
        defer self.allocator.free(cross_file.bindings);
        defer for (cross_file.bindings) |b| {
            self.allocator.free(b.target_uri);
            if (b.target_source) |s| self.allocator.free(s);
        };

        const maybe_definition = try definition_api.definitionAtCrossFile(self.allocator, source, position, cross_file);
        if (maybe_definition == null) return null;

        const definition = maybe_definition.?;
        return .{ .Definition = .{ .Location = .{
            .uri = definition.uri orelse uri,
            .range = toLspRange(definition.range),
        } } };
    }

    pub fn @"textDocument/references"(self: *Handler, arena: Allocator, params: types.ReferenceParams) !lsp.ResultType("textDocument/references") {
        const uri = params.textDocument.uri;
        const source = self.docs.docs.get(uri) orelse return null;
        const position: frontend.Position = .{
            .line = params.position.line,
            .character = params.position.character,
        };

        const include_declaration = if (params.context.includeDeclaration) true else false;

        const maybe_target = try definition_api.definitionAt(self.allocator, source, position);
        if (maybe_target == null) return null;
        const target_name = try self.identifierAtPosition(source, position) orelse return null;

        var all_locations = std.ArrayList(types.Location){};
        errdefer all_locations.deinit(arena);

        const same_file_refs = try references_api.referencesAt(self.allocator, source, position, include_declaration);
        defer self.allocator.free(same_file_refs);
        for (same_file_refs) |ref| {
            try all_locations.append(arena, .{
                .uri = uri,
                .range = toLspRange(.{
                    .start = .{ .line = ref.start.line, .character = ref.start.character },
                    .end = .{ .line = ref.end.line, .character = ref.end.character },
                }),
            });
        }

        var doc_it = self.docs.docs.iterator();
        while (doc_it.next()) |entry| {
            const other_uri = entry.key_ptr.*;
            if (std.mem.eql(u8, other_uri, uri)) continue;
            const other_source = entry.value_ptr.*;

            const other_refs = try self.findImportedReferencesInSource(other_uri, other_source, uri, target_name);
            defer self.allocator.free(other_refs);
            for (other_refs) |ref| {
                try all_locations.append(arena, .{
                    .uri = other_uri,
                    .range = toLspRange(ref),
                });
            }
        }

        return try all_locations.toOwnedSlice(arena);
    }

    pub fn @"textDocument/completion"(self: *Handler, arena: Allocator, params: types.CompletionParams) !lsp.ResultType("textDocument/completion") {
        const source = self.docs.docs.get(params.textDocument.uri) orelse return null;
        const position: frontend.Position = .{
            .line = params.position.line,
            .character = params.position.character,
        };

        const trigger_char: ?[]const u8 = if (params.context) |ctx| ctx.triggerCharacter else null;
        const completion_items = try completion_api.completionAt(self.allocator, source, position, trigger_char);
        defer completion_api.deinitItems(self.allocator, completion_items);

        const snippets = &[_]struct { label: []const u8, body: []const u8, detail: []const u8 }{
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

        const is_line_start = isAtLineStart(source, position);
        const snippet_count: usize = if (is_line_start and trigger_char == null) snippets.len else 0;
        const result = try arena.alloc(types.CompletionItem, completion_items.len + snippet_count);
        for (completion_items, 0..) |item, i| {
            result[i] = .{
                .label = try arena.dupe(u8, item.label),
                .kind = completionKindToLsp(item.kind),
                .detail = if (item.detail) |detail| try arena.dupe(u8, detail) else null,
                .documentation = if (item.documentation) |doc| .{ .MarkupContent = .{
                    .kind = .markdown,
                    .value = try arena.dupe(u8, doc),
                } } else null,
            };
        }
        for (snippets[0..snippet_count], completion_items.len..) |snip, i| {
            result[i] = .{
                .label = snip.label,
                .kind = .Snippet,
                .detail = snip.detail,
                .insertTextFormat = .Snippet,
                .insertText = snip.body,
            };
        }

        return .{ .array_of_CompletionItem = result };
    }

    pub fn @"textDocument/prepareRename"(self: *Handler, arena: Allocator, params: types.PrepareRenameParams) !lsp.ResultType("textDocument/prepareRename") {
        const uri = params.textDocument.uri;
        const source = self.docs.docs.get(uri) orelse return null;
        const position: frontend.Position = .{
            .line = params.position.line,
            .character = params.position.character,
        };

        const identifier = try self.identifierAtPosition(source, position) orelse return null;
        const maybe_definition = try definition_api.definitionAt(self.allocator, source, position);
        if (maybe_definition == null) return null;

        return .{ .literal_1 = .{
            .range = toLspRange(maybe_definition.?.range),
            .placeholder = try arena.dupe(u8, identifier),
        } };
    }

    pub fn @"textDocument/rename"(self: *Handler, arena: Allocator, params: types.RenameParams) !lsp.ResultType("textDocument/rename") {
        if (!rename_api.isValidIdentifier(params.newName)) return null;

        const uri = params.textDocument.uri;
        const source = self.docs.docs.get(uri) orelse return null;
        const position: frontend.Position = .{
            .line = params.position.line,
            .character = params.position.character,
        };

        const target_name = try self.identifierAtPosition(source, position) orelse return null;

        const same_file_ranges = try rename_api.renameRangesAt(self.allocator, source, position);
        defer self.allocator.free(same_file_ranges);
        if (same_file_ranges.len == 0) return null;

        var changes: lsp.parser.Map(types.DocumentUri, []const types.TextEdit) = .{};
        try changes.map.put(arena, try arena.dupe(u8, uri), try buildRenameTextEdits(arena, same_file_ranges, params.newName));

        var doc_it = self.docs.docs.iterator();
        while (doc_it.next()) |entry| {
            const other_uri = entry.key_ptr.*;
            if (std.mem.eql(u8, other_uri, uri)) continue;

            const other_source = entry.value_ptr.*;
            const other_ranges = try self.findImportedReferencesInSource(other_uri, other_source, uri, target_name);
            defer self.allocator.free(other_ranges);
            if (other_ranges.len == 0) continue;

            try changes.map.put(arena, try arena.dupe(u8, other_uri), try buildRenameTextEdits(arena, other_ranges, params.newName));
        }

        return .{ .changes = changes };
    }

    pub fn @"textDocument/inlayHint"(self: *Handler, arena: Allocator, params: types.InlayHintParams) !lsp.ResultType("textDocument/inlayHint") {
        const source = self.docs.docs.get(params.textDocument.uri) orelse return null;
        const range: frontend.Range = .{
            .start = .{ .line = params.range.start.line, .character = params.range.start.character },
            .end = .{ .line = params.range.end.line, .character = params.range.end.character },
        };

        const hints = try inlay_hints_api.hintsInRange(self.allocator, source, range);
        defer inlay_hints_api.deinitHints(self.allocator, hints);

        if (hints.len == 0) return null;

        const result = try arena.alloc(types.InlayHint, hints.len);
        for (hints, 0..) |hint, i| {
            result[i] = .{
                .position = toLspPosition(hint.position),
                .label = .{ .string = try arena.dupe(u8, hint.label) },
                .kind = switch (hint.kind) {
                    .type_hint => .Type,
                    .parameter_hint => .Parameter,
                },
                .paddingLeft = hint.padding_left,
                .paddingRight = hint.padding_right,
            };
        }

        return result;
    }

    pub fn @"textDocument/codeLens"(self: *Handler, arena: Allocator, params: types.CodeLensParams) !lsp.ResultType("textDocument/codeLens") {
        const source = self.docs.docs.get(params.textDocument.uri) orelse return null;

        const lenses = try code_lens_api.findVerificationLenses(self.allocator, source);
        defer code_lens_api.deinitLenses(self.allocator, lenses);

        if (lenses.len == 0) return null;

        const result = try arena.alloc(types.CodeLens, lenses.len);
        for (lenses, 0..) |lens, i| {
            result[i] = .{
                .range = toLspRange(lens.range),
                .command = .{
                    .title = try arena.dupe(u8, lens.title),
                    .command = "ora.verify",
                },
            };
        }

        return result;
    }

    pub fn @"textDocument/signatureHelp"(self: *Handler, arena: Allocator, params: types.SignatureHelpParams) !?types.SignatureHelp {
        const source = self.docs.docs.get(params.textDocument.uri) orelse return null;
        const position: frontend.Position = .{
            .line = params.position.line,
            .character = params.position.character,
        };

        var sig_val = (signature_help_api.signatureAt(self.allocator, source, position) catch return null) orelse return null;
        defer sig_val.deinit(self.allocator);

        const param_infos = try arena.alloc(types.ParameterInformation, sig_val.parameters.len);
        for (sig_val.parameters, 0..) |param, i| {
            param_infos[i] = .{
                .label = .{ .string = try arena.dupe(u8, param.label) },
                .documentation = null,
            };
        }

        const sig_info: types.SignatureInformation = .{
            .label = try arena.dupe(u8, sig_val.label),
            .documentation = if (sig_val.documentation) |doc| .{ .MarkupContent = .{
                .kind = .markdown,
                .value = try arena.dupe(u8, doc),
            } } else null,
            .parameters = param_infos,
            .activeParameter = sig_val.active_parameter,
        };

        return .{
            .signatures = &[_]types.SignatureInformation{sig_info},
            .activeSignature = 0,
            .activeParameter = sig_val.active_parameter,
        };
    }

    pub fn @"textDocument/semanticTokens/full"(self: *Handler, arena: Allocator, params: types.SemanticTokensParams) !lsp.ResultType("textDocument/semanticTokens/full") {
        const source = self.docs.docs.get(params.textDocument.uri) orelse return null;

        const tokens = try semantic_tokens_api.tokenize(self.allocator, source);
        defer self.allocator.free(tokens);

        const data = try semantic_tokens_api.encodeTokens(arena, tokens);

        return .{ .data = data };
    }

    pub fn @"textDocument/formatting"(self: *Handler, arena: Allocator, params: types.DocumentFormattingParams) !lsp.ResultType("textDocument/formatting") {
        const source = self.docs.docs.get(params.textDocument.uri) orelse return null;

        const formatted = formatting_api.formatSourceAlloc(self.allocator, source, .{
            .line_width = 100,
            .indent_size = normalizeIndentSize(params.options.tabSize),
        }) catch |err| switch (err) {
            error.ParseError => return null,
            else => return err,
        };
        defer self.allocator.free(formatted);

        if (std.mem.eql(u8, source, formatted)) {
            return try arena.alloc(types.TextEdit, 0);
        }

        const edits = try arena.alloc(types.TextEdit, 1);
        edits[0] = .{
            .range = .{
                .start = .{ .line = 0, .character = 0 },
                .end = textEndPosition(source, self.position_encoding),
            },
            .newText = try arena.dupe(u8, formatted),
        };
        return edits;
    }

    pub fn @"textDocument/documentHighlight"(self: *Handler, arena: Allocator, params: types.DocumentHighlightParams) !lsp.ResultType("textDocument/documentHighlight") {
        const source = self.docs.docs.get(params.textDocument.uri) orelse return null;
        const position: frontend.Position = .{
            .line = params.position.line,
            .character = params.position.character,
        };

        const refs = try references_api.referencesAt(self.allocator, source, position, true);
        defer self.allocator.free(refs);
        if (refs.len == 0) return null;

        const result = try arena.alloc(types.DocumentHighlight, refs.len);
        for (refs, 0..) |ref, i| {
            result[i] = .{
                .range = toLspRange(ref),
                .kind = .Text,
            };
        }

        return result;
    }

    pub fn @"textDocument/foldingRange"(self: *Handler, arena: Allocator, params: types.FoldingRangeParams) !lsp.ResultType("textDocument/foldingRange") {
        const source = self.docs.docs.get(params.textDocument.uri) orelse return null;

        const ranges = try folding_api.foldingRanges(self.allocator, source);
        defer folding_api.deinitRanges(self.allocator, ranges);
        if (ranges.len == 0) return null;

        const result = try arena.alloc(types.FoldingRange, ranges.len);
        for (ranges, 0..) |range, i| {
            result[i] = .{
                .startLine = range.start_line,
                .endLine = range.end_line,
                .kind = switch (range.kind) {
                    .region => .region,
                    .comment => .comment,
                    .imports => .imports,
                },
            };
        }

        return result;
    }

    pub fn @"textDocument/codeAction"(self: *Handler, arena: Allocator, params: types.CodeActionParams) !lsp.ResultType("textDocument/codeAction") {
        _ = self;

        const Slice = @typeInfo(lsp.ResultType("textDocument/codeAction")).optional.child;
        const CodeActionOrCommand = @typeInfo(Slice).pointer.child;

        var results = std.ArrayList(CodeActionOrCommand){};

        for (params.context.diagnostics) |diag| {
            if (std.mem.indexOf(u8, diag.message, "expected ';'") != null) {
                const insert_pos = diag.range.end;
                const edits = try arena.alloc(types.TextEdit, 1);
                edits[0] = .{ .range = .{ .start = insert_pos, .end = insert_pos }, .newText = ";" };
                var changes: lsp.parser.Map(types.DocumentUri, []const types.TextEdit) = .{};
                try changes.map.put(arena, params.textDocument.uri, edits);
                try results.append(arena, .{ .CodeAction = .{
                    .title = "Insert missing ';'",
                    .kind = .quickfix,
                    .diagnostics = try arena.dupe(types.Diagnostic, &.{diag}),
                    .edit = .{ .changes = changes },
                    .isPreferred = true,
                } });
            }
        }

        if (results.items.len == 0) return null;
        return try results.toOwnedSlice(arena);
    }

    pub fn @"workspace/symbol"(self: *Handler, arena: Allocator, params: types.WorkspaceSymbolParams) !lsp.ResultType("workspace/symbol") {
        const query = params.query;
        var symbols = std.ArrayList(types.SymbolInformation){};

        var doc_it = self.docs.docs.iterator();
        while (doc_it.next()) |entry| {
            const doc_uri = entry.key_ptr.*;
            const doc_source = entry.value_ptr.*;

            var index = try semantic_index.indexDocument(self.allocator, doc_source);
            defer index.deinit(self.allocator);

            for (index.symbols) |sym| {
                if (sym.parent != null) continue;
                if (query.len > 0 and !fuzzyMatch(sym.name, query)) continue;

                try symbols.append(arena, .{
                    .name = try arena.dupe(u8, sym.name),
                    .kind = @enumFromInt(semantic_index.toLspKind(sym.kind)),
                    .location = .{
                        .uri = try arena.dupe(u8, doc_uri),
                        .range = toLspRange(sym.selection_range),
                    },
                    .containerName = if (sym.detail) |d| try arena.dupe(u8, d) else null,
                });
            }
        }

        if (symbols.items.len == 0) return null;
        return .{ .array_of_SymbolInformation = try symbols.toOwnedSlice(arena) };
    }

    // --- Selection Range ---

    pub fn @"textDocument/selectionRange"(self: *Handler, arena: Allocator, params: types.SelectionRangeParams) !lsp.ResultType("textDocument/selectionRange") {
        const source = self.docs.docs.get(params.textDocument.uri) orelse return null;

        var parse_result = compiler.syntax.parse(self.allocator, compiler.FileId.fromIndex(0), source) catch return null;
        defer parse_result.deinit();

        var lower_result = compiler.ast.lower(self.allocator, &parse_result.tree) catch return null;
        defer lower_result.deinit();

        const result = try arena.alloc(types.SelectionRange, params.positions.len);
        for (params.positions, 0..) |pos, i| {
            const offset = positionToOffset(source, pos.line, pos.character);
            result[i] = try buildSelectionRange(arena, &lower_result.file, source, offset);
        }
        return result;
    }

    fn buildSelectionRange(
        arena: Allocator,
        file: *const compiler.ast.AstFile,
        source: []const u8,
        offset: u32,
    ) !types.SelectionRange {
        var ranges = std.ArrayList(types.Range){};
        defer ranges.deinit(arena);

        for (file.root_items) |item_id| {
            try collectContainingRanges(&ranges, arena, file, source, item_id, offset);
        }

        std.mem.sort(types.Range, ranges.items, {}, struct {
            fn cmp(_: void, a: types.Range, b: types.Range) bool {
                return rangeSpan(a) > rangeSpan(b);
            }
            fn rangeSpan(r: types.Range) u64 {
                const lines = if (r.end.line >= r.start.line) r.end.line - r.start.line else 0;
                const chars = if (r.end.character >= r.start.character) r.end.character - r.start.character else 0;
                return @as(u64, lines) * 100000 + chars;
            }
        }.cmp);

        var current: types.SelectionRange = if (ranges.items.len > 0)
            .{ .range = ranges.items[0] }
        else
            .{ .range = .{ .start = .{ .line = 0, .character = 0 }, .end = .{ .line = 0, .character = 0 } } };

        for (ranges.items[1..]) |r| {
            const parent = try arena.create(types.SelectionRange);
            parent.* = current;
            current = .{ .range = r, .parent = parent };
        }
        return current;
    }

    fn collectContainingRanges(
        ranges: *std.ArrayList(types.Range),
        arena: Allocator,
        file: *const compiler.ast.AstFile,
        source: []const u8,
        item_id: compiler.ast.ItemId,
        offset: u32,
    ) !void {
        const item = file.item(item_id).*;
        const item_range = itemTextRange(item) orelse return;
        if (offset < item_range.start or offset > item_range.end) return;

        try ranges.append(arena, offsetRangeToLsp(source, item_range));

        switch (item) {
            .Contract => |c| {
                for (c.members) |mid| try collectContainingRanges(ranges, arena, file, source, mid, offset);
            },
            .Function => |f| try collectBodyContainingRanges(ranges, arena, file, source, f.body, offset),
            else => {},
        }
    }

    fn collectBodyContainingRanges(
        ranges: *std.ArrayList(types.Range),
        arena: Allocator,
        file: *const compiler.ast.AstFile,
        source: []const u8,
        body_id: compiler.ast.BodyId,
        offset: u32,
    ) !void {
        const body = file.body(body_id).*;
        for (body.statements) |stmt_id| {
            const stmt = file.statement(stmt_id).*;
            const sr = stmtTextRange(stmt) orelse continue;
            if (offset < sr.start or offset > sr.end) continue;
            try ranges.append(arena, offsetRangeToLsp(source, sr));
            switch (stmt) {
                .If => |s| {
                    try collectBodyContainingRanges(ranges, arena, file, source, s.then_body, offset);
                    if (s.else_body) |eb| try collectBodyContainingRanges(ranges, arena, file, source, eb, offset);
                },
                .While => |s| try collectBodyContainingRanges(ranges, arena, file, source, s.body, offset),
                .For => |s| try collectBodyContainingRanges(ranges, arena, file, source, s.body, offset),
                .Block => |s| try collectBodyContainingRanges(ranges, arena, file, source, s.body, offset),
                else => {},
            }
        }
    }

    fn itemTextRange(item: compiler.ast.nodes.Item) ?compiler.TextRange {
        return switch (item) {
            .Import => |i| i.range,
            .Contract => |c| c.range,
            .Function => |f| f.range,
            .Struct => |s| s.range,
            .Bitfield => |b| b.range,
            .Enum => |e| e.range,
            .Trait => |t| t.range,
            .Impl => |i| i.range,
            .TypeAlias => |t| t.range,
            .LogDecl => |l| l.range,
            .ErrorDecl => |e| e.range,
            .Field => |f| f.range,
            .Constant => |c| c.range,
            .GhostBlock => |g| g.range,
            .Error => null,
        };
    }

    fn stmtTextRange(stmt: compiler.ast.nodes.Stmt) ?compiler.TextRange {
        return switch (stmt) {
            .Error => null,
            inline else => |s| s.range,
        };
    }

    // --- Document Link ---

    pub fn @"textDocument/documentLink"(self: *Handler, arena: Allocator, params: types.DocumentLinkParams) !lsp.ResultType("textDocument/documentLink") {
        const uri = params.textDocument.uri;
        const source = self.docs.docs.get(uri) orelse return null;

        var resolution = try workspace.resolveDocumentImports(
            self.allocator,
            uri,
            source,
            .{ .workspace_roots = self.workspace_roots.items },
        );
        defer resolution.deinit(self.allocator);

        if (resolution.imports.len == 0) return null;

        var links = std.ArrayList(types.DocumentLink){};
        for (resolution.imports) |imp| {
            const target_uri = try workspace.pathToFileUri(arena, imp.resolved_path);
            const path_range = findImportPathRange(source, imp.specifier) orelse continue;
            try links.append(arena, .{
                .range = path_range,
                .target = target_uri,
                .tooltip = try std.fmt.allocPrint(arena, "Open {s}", .{imp.specifier}),
            });
        }

        if (links.items.len == 0) return null;
        return try links.toOwnedSlice(arena);
    }

    fn findImportPathRange(source: []const u8, specifier: []const u8) ?types.Range {
        var line: u32 = 0;
        var col: u32 = 0;
        var i: usize = 0;
        while (i < source.len) : (i += 1) {
            if (source[i] == '\n') {
                line += 1;
                col = 0;
                continue;
            }
            if (i + specifier.len <= source.len and std.mem.eql(u8, source[i .. i + specifier.len], specifier)) {
                // Verify it's inside a string literal (preceded by quote)
                if (i > 0 and source[i - 1] == '"') {
                    return .{
                        .start = .{ .line = line, .character = col },
                        .end = .{ .line = line, .character = col + @as(u32, @intCast(specifier.len)) },
                    };
                }
            }
            col += 1;
        }
        return null;
    }

    // --- Call Hierarchy ---

    pub fn @"textDocument/prepareCallHierarchy"(self: *Handler, arena: Allocator, params: types.CallHierarchyPrepareParams) !lsp.ResultType("textDocument/prepareCallHierarchy") {
        const uri = params.textDocument.uri;
        const source = self.docs.docs.get(uri) orelse return null;
        const position: frontend.Position = .{
            .line = params.position.line,
            .character = params.position.character,
        };

        var index = try semantic_index.indexDocument(self.allocator, source);
        defer index.deinit(self.allocator);

        const sym_idx = semantic_index.findSymbolAtPosition(index.symbols, position) orelse return null;
        const sym = index.symbols[sym_idx];
        if (sym.kind != .function and sym.kind != .method) return null;

        const result = try arena.alloc(types.CallHierarchyItem, 1);
        result[0] = .{
            .name = try arena.dupe(u8, sym.name),
            .kind = @enumFromInt(semantic_index.toLspKind(sym.kind)),
            .uri = try arena.dupe(u8, uri),
            .range = toLspRange(sym.range),
            .selectionRange = toLspRange(sym.selection_range),
            .detail = if (sym.detail) |d| try arena.dupe(u8, d) else null,
        };
        return result;
    }

    pub fn @"callHierarchy/incomingCalls"(self: *Handler, arena: Allocator, params: types.CallHierarchyIncomingCallsParams) !lsp.ResultType("callHierarchy/incomingCalls") {
        const target_name = params.item.name;
        var calls = std.ArrayList(types.CallHierarchyIncomingCall){};

        var doc_it = self.docs.docs.iterator();
        while (doc_it.next()) |entry| {
            const doc_uri = entry.key_ptr.*;
            const doc_source = entry.value_ptr.*;

            var index = try semantic_index.indexDocument(self.allocator, doc_source);
            defer index.deinit(self.allocator);

            // Find call sites: scan for Call expressions to target_name inside functions
            var lex = try lexer_mod.Lexer.initWithConfig(self.allocator, doc_source, lexer_mod.LexerConfig.development());
            defer lex.deinit();
            const tokens = lex.scanTokens() catch continue;
            defer self.allocator.free(tokens);

            for (index.symbols) |sym| {
                if (sym.kind != .function and sym.kind != .method) continue;

                var has_call = false;
                var call_ranges = std.ArrayList(types.Range){};
                defer call_ranges.deinit(arena);

                for (tokens) |tok| {
                    if (tok.type != .Identifier) continue;
                    if (!std.mem.eql(u8, tok.lexeme, target_name)) continue;
                    const tok_line = if (tok.line > 0) tok.line - 1 else 0;
                    const tok_char = if (tok.column > 0) tok.column - 1 else 0;
                    const pos = frontend.Position{ .line = tok_line, .character = tok_char };

                    if (positionInRange(pos, sym.range)) {
                        has_call = true;
                        const len = @as(u32, @intCast(tok.lexeme.len));
                        try call_ranges.append(arena, .{
                            .start = .{ .line = tok_line, .character = tok_char },
                            .end = .{ .line = tok_line, .character = tok_char + len },
                        });
                    }
                }

                if (has_call) {
                    try calls.append(arena, .{
                        .from = .{
                            .name = try arena.dupe(u8, sym.name),
                            .kind = @enumFromInt(semantic_index.toLspKind(sym.kind)),
                            .uri = try arena.dupe(u8, doc_uri),
                            .range = toLspRange(sym.range),
                            .selectionRange = toLspRange(sym.selection_range),
                            .detail = if (sym.detail) |d| try arena.dupe(u8, d) else null,
                        },
                        .fromRanges = try call_ranges.toOwnedSlice(arena),
                    });
                }
            }
        }

        if (calls.items.len == 0) return null;
        return try calls.toOwnedSlice(arena);
    }

    pub fn @"callHierarchy/outgoingCalls"(self: *Handler, arena: Allocator, params: types.CallHierarchyOutgoingCallsParams) !lsp.ResultType("callHierarchy/outgoingCalls") {
        const caller_uri = params.item.uri;
        const caller_source = self.docs.docs.get(caller_uri) orelse return null;
        const caller_range = params.item.range;

        var lex = try lexer_mod.Lexer.initWithConfig(self.allocator, caller_source, lexer_mod.LexerConfig.development());
        defer lex.deinit();
        const tokens = lex.scanTokens() catch return null;
        defer self.allocator.free(tokens);

        var index = try semantic_index.indexDocument(self.allocator, caller_source);
        defer index.deinit(self.allocator);

        // Collect unique call targets within the caller's range
        const CallTarget = struct { name: []const u8, range: types.Range };
        var targets = std.ArrayList(CallTarget){};
        defer targets.deinit(self.allocator);

        for (tokens, 0..) |tok, ti| {
            if (tok.type != .Identifier) continue;
            const tok_line = if (tok.line > 0) tok.line - 1 else 0;
            const tok_char = if (tok.column > 0) tok.column - 1 else 0;
            const pos = frontend.Position{ .line = tok_line, .character = tok_char };

            if (!positionInRange(pos, .{
                .start = .{ .line = caller_range.start.line, .character = caller_range.start.character },
                .end = .{ .line = caller_range.end.line, .character = caller_range.end.character },
            })) continue;

            // Check if this identifier is followed by `(`
            if (ti + 1 < tokens.len and tokens[ti + 1].type == .LeftParen) {
                const len = @as(u32, @intCast(tok.lexeme.len));
                var already_listed = false;
                for (targets.items) |t| {
                    if (std.mem.eql(u8, t.name, tok.lexeme)) {
                        already_listed = true;
                        break;
                    }
                }
                if (!already_listed) {
                    try targets.append(self.allocator, .{
                        .name = tok.lexeme,
                        .range = .{
                            .start = .{ .line = tok_line, .character = tok_char },
                            .end = .{ .line = tok_line, .character = tok_char + len },
                        },
                    });
                }
            }
        }

        if (targets.items.len == 0) return null;

        var calls = std.ArrayList(types.CallHierarchyOutgoingCall){};
        for (targets.items) |target| {
            // Try to find the function symbol
            for (index.symbols) |sym| {
                if (sym.kind != .function and sym.kind != .method) continue;
                if (!std.mem.eql(u8, sym.name, target.name)) continue;

                try calls.append(arena, .{
                    .to = .{
                        .name = try arena.dupe(u8, sym.name),
                        .kind = @enumFromInt(semantic_index.toLspKind(sym.kind)),
                        .uri = try arena.dupe(u8, caller_uri),
                        .range = toLspRange(sym.range),
                        .selectionRange = toLspRange(sym.selection_range),
                        .detail = if (sym.detail) |d| try arena.dupe(u8, d) else null,
                    },
                    .fromRanges = try arena.dupe(types.Range, &.{target.range}),
                });
                break;
            }
        }

        if (calls.items.len == 0) return null;
        return try calls.toOwnedSlice(arena);
    }

    fn positionInRange(pos: frontend.Position, range: frontend.Range) bool {
        if (pos.line < range.start.line) return false;
        if (pos.line > range.end.line) return false;
        if (pos.line == range.start.line and pos.character < range.start.character) return false;
        if (pos.line == range.end.line and pos.character > range.end.character) return false;
        return true;
    }

    fn identifierAtPosition(self: *Handler, source: []const u8, position: frontend.Position) !?[]const u8 {
        var lex = try lexer_mod.Lexer.initWithConfig(self.allocator, source, lexer_mod.LexerConfig.development());
        defer lex.deinit();
        const tokens = lex.scanTokens() catch return null;
        defer self.allocator.free(tokens);

        for (tokens) |token| {
            if (token.type != .Identifier) continue;
            const tok_line = if (token.line > 0) token.line - 1 else 0;
            const tok_char = if (token.column > 0) token.column - 1 else 0;
            const tok_end = tok_char + @as(u32, @intCast(token.lexeme.len));
            if (tok_line == position.line and tok_char <= position.character and position.character < tok_end) {
                return token.lexeme;
            }
        }
        return null;
    }

    /// Find references to `target_name` (defined in `target_uri`) inside
    /// another file.  Only returns matches where the other file actually
    /// imports the target file — an `alias.target_name` field-access pattern.
    fn findImportedReferencesInSource(
        self: *Handler,
        other_uri: []const u8,
        other_source: []const u8,
        target_uri: []const u8,
        target_name: []const u8,
    ) ![]frontend.Range {
        const empty = try self.allocator.alloc(frontend.Range, 0);

        const target_path = try workspace.fileUriToPathAlloc(self.allocator, target_uri) orelse return empty;
        defer self.allocator.free(target_path);
        const norm_target = try workspace.normalizePathAlloc(self.allocator, target_path);
        defer self.allocator.free(norm_target);

        var resolution = try workspace.resolveDocumentImports(
            self.allocator,
            other_uri,
            other_source,
            .{ .workspace_roots = self.workspace_roots.items },
        );
        defer resolution.deinit(self.allocator);

        var aliases = std.ArrayList([]const u8){};
        defer aliases.deinit(self.allocator);
        for (resolution.imports) |imp| {
            const norm_imp = try workspace.normalizePathAlloc(self.allocator, imp.resolved_path);
            defer self.allocator.free(norm_imp);
            if (std.mem.eql(u8, norm_imp, norm_target)) {
                if (imp.alias) |alias| try aliases.append(self.allocator, alias);
            }
        }
        if (aliases.items.len == 0) {
            self.allocator.free(empty);
            return try self.allocator.alloc(frontend.Range, 0);
        }

        var lex = try lexer_mod.Lexer.initWithConfig(self.allocator, other_source, lexer_mod.LexerConfig.development());
        defer lex.deinit();
        const tokens = lex.scanTokens() catch {
            self.allocator.free(empty);
            return try self.allocator.alloc(frontend.Range, 0);
        };
        defer self.allocator.free(tokens);

        self.allocator.free(empty);
        var ranges = std.ArrayList(frontend.Range){};
        errdefer ranges.deinit(self.allocator);

        // Match patterns:  alias . target_name
        var i: usize = 0;
        while (i < tokens.len) : (i += 1) {
            const tok = tokens[i];
            if (tok.type != .Identifier) continue;

            var is_alias = false;
            for (aliases.items) |alias| {
                if (std.mem.eql(u8, tok.lexeme, alias)) {
                    is_alias = true;
                    break;
                }
            }
            if (!is_alias) continue;

            if (i + 2 < tokens.len and tokens[i + 1].type == .Dot) {
                const member = tokens[i + 2];
                if (member.type == .Identifier and std.mem.eql(u8, member.lexeme, target_name)) {
                    const start_line = if (member.line > 0) member.line - 1 else 0;
                    const start_char = if (member.column > 0) member.column - 1 else 0;
                    const len = @as(u32, @intCast(member.lexeme.len));
                    try ranges.append(self.allocator, .{
                        .start = .{ .line = start_line, .character = start_char },
                        .end = .{ .line = start_line, .character = start_char + len },
                    });
                    i += 2;
                }
            }
        }

        return ranges.toOwnedSlice(self.allocator);
    }

    // --- internal helpers ---

    fn configureWorkspaceRoots(self: *Handler, params: types.InitializeParams) !void {
        self.clearWorkspaceRoots();

        var added_root = false;

        if (params.workspaceFolders) |folders| {
            for (folders) |folder| {
                if (try self.addWorkspaceRootFromUri(folder.uri)) {
                    added_root = true;
                }
            }
        }

        if (!added_root) {
            if (params.rootUri) |root_uri| {
                added_root = try self.addWorkspaceRootFromUri(root_uri);
            }
        }

        if (!added_root) {
            const cwd = try std.fs.cwd().realpathAlloc(self.allocator, ".");
            defer self.allocator.free(cwd);
            _ = try self.addWorkspaceRootFromPath(cwd);
        }
    }

    fn clearWorkspaceRoots(self: *Handler) void {
        for (self.workspace_roots.items) |root| {
            self.allocator.free(root);
        }
        self.workspace_roots.clearRetainingCapacity();
    }

    fn hasWorkspaceRoot(self: *const Handler, candidate: []const u8) bool {
        for (self.workspace_roots.items) |existing| {
            if (std.mem.eql(u8, existing, candidate)) return true;
        }
        return false;
    }

    fn addWorkspaceRootFromUri(self: *Handler, uri: []const u8) !bool {
        const maybe_path = try workspace.fileUriToPathAlloc(self.allocator, uri);
        if (maybe_path == null) return false;
        const path = maybe_path.?;
        defer self.allocator.free(path);
        return try self.addWorkspaceRootFromPath(path);
    }

    fn addWorkspaceRootFromPath(self: *Handler, root_path: []const u8) !bool {
        const normalized = try workspace.normalizePathAlloc(self.allocator, root_path);
        errdefer self.allocator.free(normalized);

        if (self.hasWorkspaceRoot(normalized)) {
            self.allocator.free(normalized);
            return false;
        }

        try self.workspace_roots.append(self.allocator, normalized);
        return true;
    }

    fn buildCrossFileContext(self: *Handler, uri: []const u8, source: []const u8) !definition_api.CrossFileContext {
        var import_resolution = try workspace.resolveDocumentImports(
            self.allocator,
            uri,
            source,
            .{ .workspace_roots = self.workspace_roots.items },
        );
        defer import_resolution.deinit(self.allocator);

        var bindings = std.ArrayList(definition_api.ImportBinding){};
        errdefer {
            for (bindings.items) |b| {
                self.allocator.free(b.target_uri);
                if (b.target_source) |s| self.allocator.free(s);
            }
            bindings.deinit(self.allocator);
        }

        for (import_resolution.imports) |resolved| {
            const alias = resolved.alias orelse aliasFromSpecifier(resolved.specifier) orelse continue;
            const target_uri = try workspace.pathToFileUri(self.allocator, resolved.resolved_path);
            errdefer self.allocator.free(target_uri);

            const target_source = self.getSourceForPath(target_uri, resolved.resolved_path);
            try bindings.append(self.allocator, .{
                .alias = alias,
                .target_uri = target_uri,
                .target_source = target_source,
            });
        }

        return .{ .bindings = try bindings.toOwnedSlice(self.allocator) };
    }

    fn aliasFromSpecifier(specifier: []const u8) ?[]const u8 {
        const basename = std.fs.path.stem(specifier);
        if (basename.len == 0) return null;
        return basename;
    }

    fn getSourceForPath(self: *Handler, target_uri: []const u8, fs_path: []const u8) ?[]u8 {
        if (self.docs.docs.get(target_uri)) |text| {
            return self.allocator.dupe(u8, text) catch return null;
        }
        return std.fs.cwd().readFileAlloc(self.allocator, fs_path, 10 * 1024 * 1024) catch null;
    }

    fn updateDocumentDependencies(self: *Handler, uri: []const u8, source: []const u8) !void {
        const maybe_doc_path = try workspace.fileUriToPathAlloc(self.allocator, uri);
        const normalized_doc_path = if (maybe_doc_path) |doc_path| blk: {
            defer self.allocator.free(doc_path);
            break :blk try workspace.normalizePathAlloc(self.allocator, doc_path);
        } else null;
        defer if (normalized_doc_path) |path| self.allocator.free(path);

        var import_resolution = try workspace.resolveDocumentImports(
            self.allocator,
            uri,
            source,
            .{ .workspace_roots = self.workspace_roots.items },
        );
        defer import_resolution.deinit(self.allocator);

        const import_paths = try self.allocator.alloc([]const u8, import_resolution.imports.len);
        defer self.allocator.free(import_paths);
        for (import_resolution.imports, 0..) |item, i| {
            import_paths[i] = item.resolved_path;
        }

        try self.dependencies.upsert(uri, normalized_doc_path, import_paths);
    }

    fn publishDependentsDiagnostics(self: *Handler, arena: Allocator, changed_path: []const u8, exclude_uri: ?[]const u8) !void {
        const dependents = try self.dependencies.collectDependents(self.allocator, changed_path);
        defer self.allocator.free(dependents);

        for (dependents) |dependent_uri| {
            if (exclude_uri) |excluded| {
                if (std.mem.eql(u8, dependent_uri, excluded)) continue;
            }
            const dependent_source = self.docs.docs.get(dependent_uri) orelse continue;
            try self.publishDiagnostics(arena, dependent_uri, dependent_source);
        }
    }

    fn publishDiagnostics(self: *Handler, arena: Allocator, uri: []const u8, source: []const u8) !void {
        var analysis = try frontend.analyzeDocument(self.allocator, source);
        defer analysis.deinit(self.allocator);

        var import_resolution = try workspace.resolveDocumentImports(
            self.allocator,
            uri,
            source,
            .{ .workspace_roots = self.workspace_roots.items },
        );
        defer import_resolution.deinit(self.allocator);

        const total = analysis.diagnostics.len + import_resolution.diagnostics.len;
        const diagnostics = try arena.alloc(types.Diagnostic, total);

        for (analysis.diagnostics, 0..) |diag, i| {
            diagnostics[i] = .{
                .range = toLspRange(diag.range),
                .severity = frontendSeverityToLsp(diag.severity),
                .source = diagnosticSourceName(diag.source),
                .message = try arena.dupe(u8, diag.message),
            };
        }

        var offset: usize = analysis.diagnostics.len;
        for (import_resolution.diagnostics) |diag| {
            diagnostics[offset] = .{
                .range = toLspRange(diag.range),
                .severity = .Error,
                .source = "ora-imports",
                .message = try arena.dupe(u8, diag.message),
            };
            offset += 1;
        }

        try self.transport.writeNotification(
            self.allocator,
            "textDocument/publishDiagnostics",
            types.PublishDiagnosticsParams,
            .{ .uri = uri, .diagnostics = diagnostics },
            .{ .emit_null_optional_fields = false },
        );
    }
};

fn isAtLineStart(source: []const u8, position: frontend.Position) bool {
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

fn positionToOffset(source: []const u8, line: u32, character: u32) u32 {
    var cur_line: u32 = 0;
    var i: u32 = 0;
    while (i < source.len and cur_line < line) : (i += 1) {
        if (source[i] == '\n') cur_line += 1;
    }
    return i + character;
}

fn offsetToLspPosition(source: []const u8, offset: u32) types.Position {
    var line: u32 = 0;
    var col: u32 = 0;
    const end = @min(offset, @as(u32, @intCast(source.len)));
    for (source[0..end]) |c| {
        if (c == '\n') {
            line += 1;
            col = 0;
        } else {
            col += 1;
        }
    }
    return .{ .line = line, .character = col };
}

fn offsetRangeToLsp(source: []const u8, tr: compiler.TextRange) types.Range {
    return .{
        .start = offsetToLspPosition(source, tr.start),
        .end = offsetToLspPosition(source, tr.end),
    };
}

fn toLspRange(range: frontend.Range) types.Range {
    return .{
        .start = .{ .line = range.start.line, .character = range.start.character },
        .end = .{ .line = range.end.line, .character = range.end.character },
    };
}

fn toLspPosition(pos: frontend.Position) types.Position {
    return .{ .line = pos.line, .character = pos.character };
}

fn buildRenameTextEdits(arena: Allocator, ranges: []const frontend.Range, new_name: []const u8) ![]types.TextEdit {
    const edits = try arena.alloc(types.TextEdit, ranges.len);
    const replacement = try arena.dupe(u8, new_name);

    for (ranges, 0..) |range, i| {
        edits[i] = .{ .range = toLspRange(range), .newText = replacement };
    }

    return edits;
}

fn frontendSeverityToLsp(severity: frontend.Severity) types.DiagnosticSeverity {
    return switch (severity) {
        .err => .Error,
        .warning => .Warning,
        .information => .Information,
        .hint => .Hint,
    };
}

fn diagnosticSourceName(source: frontend.DiagnosticSource) []const u8 {
    return switch (source) {
        .lexer => "ora-lexer",
        .parser => "ora-parser",
        .sema => "ora-sema",
    };
}

fn completionKindToLsp(kind: completion_api.Kind) types.CompletionItemKind {
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

fn normalizeIndentSize(tab_size: u32) u32 {
    if (tab_size == 0) return 4;
    return @min(tab_size, 16);
}

fn textEndPosition(source: []const u8, encoding: text_edits.PositionEncoding) types.Position {
    var line: u32 = 0;
    var character: u32 = 0;
    var i: usize = 0;

    while (i < source.len) {
        const byte = source[i];
        if (byte == '\n') {
            line += 1;
            character = 0;
            i += 1;
            continue;
        }

        switch (encoding) {
            .utf8 => {
                character += 1;
                i += 1;
            },
            .utf16, .utf32 => {
                const seq_len = std.unicode.utf8ByteSequenceLength(byte) catch {
                    character += 1;
                    i += 1;
                    continue;
                };
                if (i + seq_len > source.len) {
                    character += 1;
                    i += 1;
                    continue;
                }

                const cp = std.unicode.utf8Decode(source[i .. i + seq_len]) catch {
                    character += 1;
                    i += 1;
                    continue;
                };

                character += switch (encoding) {
                    .utf16 => if (cp <= 0xFFFF) 1 else 2,
                    .utf32 => 1,
                    .utf8 => unreachable,
                };
                i += seq_len;
            },
        }
    }

    return .{ .line = line, .character = character };
}

fn lastFullText(changes: []const types.TextDocumentContentChangeEvent) ?[]const u8 {
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

fn negotiatePositionEncoding(params: types.InitializeParams) text_edits.PositionEncoding {
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

fn toLspPositionEncoding(encoding: text_edits.PositionEncoding) types.PositionEncodingKind {
    return switch (encoding) {
        .utf8 => .@"utf-8",
        .utf16 => .@"utf-16",
        .utf32 => .@"utf-32",
    };
}

/// Case-insensitive subsequence match (the standard for workspace symbol filtering).
fn fuzzyMatch(name: []const u8, query: []const u8) bool {
    var qi: usize = 0;
    for (name) |c| {
        if (qi >= query.len) break;
        if (std.ascii.toLower(c) == std.ascii.toLower(query[qi])) qi += 1;
    }
    return qi == query.len;
}

fn convertDocumentSymbol(arena: Allocator, symbol: semantic_index.DocumentSymbol) !types.DocumentSymbol {
    const children = if (symbol.children.len > 0) blk: {
        const result = try arena.alloc(types.DocumentSymbol, symbol.children.len);
        for (symbol.children, 0..) |child, i| {
            result[i] = try convertDocumentSymbol(arena, child);
        }
        break :blk result;
    } else &[_]types.DocumentSymbol{};

    return .{
        .name = try arena.dupe(u8, symbol.name),
        .detail = if (symbol.detail) |d| try arena.dupe(u8, d) else null,
        .kind = @enumFromInt(symbol.kind),
        .range = toLspRange(symbol.range),
        .selectionRange = toLspRange(symbol.selectionRange),
        .children = children,
    };
}

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    var read_buffer: [256]u8 = undefined;
    var stdio_transport: lsp.Transport.Stdio = .init(&read_buffer, .stdin(), .stdout());
    const transport: *lsp.Transport = &stdio_transport.transport;

    var handler: Handler = .init(allocator, transport);
    defer handler.deinit();

    try lsp.basic_server.run(allocator, transport, &handler, std.log.err);
}
