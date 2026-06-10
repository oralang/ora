const std = @import("std");
const lsp = @import("lsp");
const types = lsp.types;
const ora_root = @import("ora_root");

const lexer_mod = ora_root.lexer;
const frontend = ora_root.lsp.frontend;
const workspace = ora_root.lsp.workspace;
const dependency_graph = ora_root.lsp.dependency_graph;
const semantic_index = ora_root.lsp.semantic_index;
const phase_stats = ora_root.lsp.phase_stats;
const allocation_stats = ora_root.lsp.allocation_stats;
const line_index_api = ora_root.lsp.line_index;
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
const token_cache_api = ora_root.lsp.token_cache;
const call_hierarchy_api = ora_root.lsp.call_hierarchy;
const workspace_index_api = ora_root.lsp.workspace_index;
const formatting_api = @import("formatting.zig");
const code_lens_response = @import("code_lens_response.zig");
const folding_ranges_response = @import("folding_ranges_response.zig");
const cache_stats_response = @import("cache_stats_response.zig");
const hover_response = @import("hover_response.zig");
const definition_response = @import("definition_response.zig");
const signature_help_response = @import("signature_help_response.zig");
const document_symbol_response = @import("document_symbol.zig");
const document_highlight_response = @import("document_highlight.zig");
const document_link_response = @import("document_link.zig");
const code_action_response = @import("code_action.zig");
const workspace_symbol_response = @import("workspace_symbol_response.zig");
const call_hierarchy_prepare_response = @import("call_hierarchy_prepare.zig");
const call_hierarchy_calls_response = @import("call_hierarchy_calls.zig");
const references_response = @import("references_response.zig");
const rename_response = @import("rename_response.zig");
const formatting_edits = @import("formatting_edits.zig");
const protocol_ranges = @import("protocol_ranges.zig");
const selection_range_response = @import("selection_range.zig");
const uri_ranges = @import("uri_ranges.zig");
const compiler = ora_root.compiler;
const Allocator = std.mem.Allocator;

const DocumentState = struct {
    arena: std.heap.ArenaAllocator,
    line_index: ?line_index_api.LineIndex = null,
    token_cache: ?token_cache_api.Cache = null,
    semantic_tokens: ?[]semantic_tokens_api.SemanticToken = null,
    formatting_cache: ?FormattingCacheEntry = null,
    import_resolution: ?ImportResolutionCacheEntry = null,
    semantic_index: ?semantic_index.SemanticIndex = null,
    occurrence_index: ?references_api.OccurrenceIndex = null,
    imported_member_index: ?references_api.ImportedMemberIndex = null,
    call_edge_index: ?call_hierarchy_api.CallEdgeIndex = null,
    diagnostic_cache: ?DiagnosticCacheEntry = null,

    fn init(backing_allocator: Allocator) DocumentState {
        return .{ .arena = std.heap.ArenaAllocator.init(backing_allocator) };
    }

    fn allocator(self: *DocumentState) Allocator {
        return self.arena.allocator();
    }

    fn deinit(self: *DocumentState) void {
        self.clearCachedFields();
        self.arena.deinit();
        self.* = undefined;
    }

    fn reset(self: *DocumentState) void {
        self.clearCachedFields();
        _ = self.arena.reset(.free_all);
    }

    fn clearCachedFields(self: *DocumentState) void {
        self.line_index = null;
        self.token_cache = null;
        self.semantic_tokens = null;
        self.formatting_cache = null;
        self.import_resolution = null;
        self.semantic_index = null;
        self.occurrence_index = null;
        self.imported_member_index = null;
        self.call_edge_index = null;
        self.diagnostic_cache = null;
    }

    fn invalidateLineIndex(self: *DocumentState) void {
        self.line_index = null;
    }

    fn lineIndexBytes(self: *const DocumentState) usize {
        if (self.line_index) |*index| return index.estimatedByteSize();
        return 0;
    }

    fn invalidateTokenCache(self: *DocumentState) void {
        self.invalidateSemanticTokens();
        self.token_cache = null;
    }

    fn invalidateSemanticTokens(self: *DocumentState) void {
        self.semantic_tokens = null;
    }

    fn tokenCacheBytes(self: *const DocumentState) usize {
        var total: usize = 0;
        if (self.token_cache) |*cache| {
            total = addSat(total, cache.estimatedByteSize());
        }
        if (self.semantic_tokens) |tokens| {
            total = addSat(total, mulSat(tokens.len, @sizeOf(semantic_tokens_api.SemanticToken)));
        }
        return total;
    }

    fn tokenCount(self: *const DocumentState) usize {
        if (self.token_cache) |*cache| return cache.tokens.len;
        return 0;
    }

    fn tokenDiagnosticCount(self: *const DocumentState) usize {
        if (self.token_cache) |*cache| return cache.diagnostics.len;
        return 0;
    }

    fn cacheBuilderCapacityRequested(self: *const DocumentState) usize {
        var total: usize = 0;
        if (self.token_cache) |*cache| total = addSat(total, cache.builderCapacityRequested());
        if (self.semantic_index) |*index| total = addSat(total, index.builderCapacityRequested());
        if (self.occurrence_index) |*index| total = addSat(total, index.builderCapacityRequested());
        if (self.imported_member_index) |*index| total = addSat(total, index.builderCapacityRequested());
        if (self.call_edge_index) |*index| total = addSat(total, index.builderCapacityRequested());
        return total;
    }

    fn cacheBuilderItemsBuilt(self: *const DocumentState) usize {
        var total: usize = 0;
        if (self.token_cache) |*cache| total = addSat(total, cache.builderItemsBuilt());
        if (self.semantic_index) |*index| total = addSat(total, index.builderItemsBuilt());
        if (self.occurrence_index) |*index| total = addSat(total, index.builderItemsBuilt());
        if (self.imported_member_index) |*index| total = addSat(total, index.builderItemsBuilt());
        if (self.call_edge_index) |*index| total = addSat(total, index.builderItemsBuilt());
        return total;
    }

    fn cacheBuilderUnusedCapacity(self: *const DocumentState) usize {
        var total: usize = 0;
        if (self.token_cache) |*cache| total = addSat(total, cache.builderUnusedCapacity());
        if (self.semantic_index) |*index| total = addSat(total, index.builderUnusedCapacity());
        if (self.occurrence_index) |*index| total = addSat(total, index.builderUnusedCapacity());
        if (self.imported_member_index) |*index| total = addSat(total, index.builderUnusedCapacity());
        if (self.call_edge_index) |*index| total = addSat(total, index.builderUnusedCapacity());
        return total;
    }

    fn cacheBuilderGrowthEvents(self: *const DocumentState) usize {
        var total: usize = 0;
        if (self.token_cache) |*cache| total = addSat(total, cache.builderGrowthEvents());
        if (self.semantic_index) |*index| total = addSat(total, index.builderGrowthEvents());
        if (self.occurrence_index) |*index| total = addSat(total, index.builderGrowthEvents());
        if (self.imported_member_index) |*index| total = addSat(total, index.builderGrowthEvents());
        if (self.call_edge_index) |*index| total = addSat(total, index.builderGrowthEvents());
        return total;
    }

    fn invalidateFormattingCache(self: *DocumentState) void {
        self.formatting_cache = null;
    }

    fn invalidateImportResolution(self: *DocumentState) void {
        self.import_resolution = null;
    }

    fn formattingCacheBytes(self: *const DocumentState) usize {
        if (self.formatting_cache) |*entry| return entry.estimatedByteSize();
        return 0;
    }

    fn formattingCacheEntries(self: *const DocumentState) usize {
        return if (self.formatting_cache != null) 1 else 0;
    }

    fn importResolutionBytes(self: *const DocumentState) usize {
        if (self.import_resolution) |*entry| return entry.estimatedByteSize();
        return 0;
    }

    fn invalidateSemanticIndex(self: *DocumentState) void {
        self.invalidateSemanticTokens();
        self.invalidateCallEdgeIndex();
        self.semantic_index = null;
    }

    fn invalidateOccurrenceIndex(self: *DocumentState) void {
        self.occurrence_index = null;
    }

    fn invalidateImportedMemberIndex(self: *DocumentState) void {
        self.imported_member_index = null;
    }

    fn invalidateCallEdgeIndex(self: *DocumentState) void {
        self.call_edge_index = null;
    }

    fn invalidateDiagnosticCache(self: *DocumentState) void {
        self.diagnostic_cache = null;
    }

    fn semanticIndexBytes(self: *const DocumentState) usize {
        if (self.semantic_index) |*index| return index.estimatedByteSize();
        return 0;
    }

    fn symbolCount(self: *const DocumentState) usize {
        if (self.semantic_index) |*index| return index.symbols.len;
        return 0;
    }

    fn occurrenceIndexBytes(self: *const DocumentState) usize {
        if (self.occurrence_index) |*index| return index.estimatedByteSize();
        return 0;
    }

    fn occurrenceCount(self: *const DocumentState) usize {
        if (self.occurrence_index) |*index| return index.occurrences.len;
        return 0;
    }

    fn importedMemberIndexBytes(self: *const DocumentState) usize {
        if (self.imported_member_index) |*index| return index.estimatedByteSize();
        return 0;
    }

    fn importedMemberCount(self: *const DocumentState) usize {
        if (self.imported_member_index) |*index| return index.occurrences.len;
        return 0;
    }

    fn callEdgeIndexBytes(self: *const DocumentState) usize {
        if (self.call_edge_index) |*index| return index.estimatedByteSize();
        return 0;
    }

    fn callEdgeCount(self: *const DocumentState) usize {
        if (self.call_edge_index) |*index| return index.edges.len;
        return 0;
    }

    fn diagnosticCacheBytes(self: *const DocumentState) usize {
        if (self.diagnostic_cache) |*entry| return entry.estimatedByteSize();
        return 0;
    }
};

const DocumentDbRecord = struct {
    file_id: compiler.FileId,
    module_id: compiler.ModuleId,
    version: i32,
    generation: u64,
    state: DocumentState,

    fn deinit(self: *DocumentDbRecord) void {
        self.state.deinit();
        self.* = undefined;
    }
};

const max_workspace_discovery_files: usize = 512;
const max_cold_file_bytes: usize = 10 * 1024 * 1024;

const DocumentVersionState = struct {
    version: i32,
    generation: u64,
    is_cold: bool,
};

const ColdDocumentRecord = struct {
    normalized_path: []u8,
    source: []u8,
    version: i32 = 0,
    generation: u64 = 1,
    state: DocumentState,

    fn deinit(self: *ColdDocumentRecord, allocator: Allocator) void {
        self.state.deinit();
        allocator.free(self.normalized_path);
        allocator.free(self.source);
        self.* = undefined;
    }
};

const DiscoveredImporter = struct {
    uri: []u8,
    normalized_path: []u8,

    fn deinit(self: *DiscoveredImporter, allocator: Allocator) void {
        allocator.free(self.uri);
        allocator.free(self.normalized_path);
        self.* = undefined;
    }
};

const WorkspaceDiscoveryCacheEntry = struct {
    importers: []DiscoveredImporter,

    fn deinit(self: *WorkspaceDiscoveryCacheEntry, allocator: Allocator) void {
        for (self.importers) |*importer| importer.deinit(allocator);
        allocator.free(self.importers);
        self.* = undefined;
    }
};

const FormattingCacheEntry = struct {
    version: i32,
    generation: u64,
    line_width: u32,
    indent_size: u32,
    formatted: []u8,

    fn matches(
        self: *const FormattingCacheEntry,
        state: DocumentVersionState,
        options: formatting_api.Options,
    ) bool {
        return self.version == state.version and
            self.generation == state.generation and
            self.line_width == options.line_width and
            self.indent_size == options.indent_size;
    }

    fn deinit(self: *FormattingCacheEntry, allocator: Allocator) void {
        allocator.free(self.formatted);
        self.* = undefined;
    }

    fn estimatedByteSize(self: *const FormattingCacheEntry) usize {
        return addSat(@sizeOf(FormattingCacheEntry), self.formatted.len);
    }
};

const ImportResolutionCacheEntry = struct {
    version: i32,
    generation: u64,
    result: workspace.ResolutionResult,

    fn matches(self: *const ImportResolutionCacheEntry, state: DocumentVersionState) bool {
        return self.version == state.version and self.generation == state.generation;
    }

    fn deinit(self: *ImportResolutionCacheEntry, allocator: Allocator) void {
        self.result.deinit(allocator);
        self.* = undefined;
    }

    fn estimatedByteSize(self: *const ImportResolutionCacheEntry) usize {
        var total: usize = @sizeOf(ImportResolutionCacheEntry);
        total = addSat(total, mulSat(self.result.diagnostics.len, @sizeOf(workspace.ImportResolutionDiagnostic)));
        for (self.result.diagnostics) |diagnostic| {
            total = addSat(total, diagnostic.message.len);
        }
        total = addSat(total, mulSat(self.result.imports.len, @sizeOf(workspace.ResolvedImport)));
        for (self.result.imports) |import_item| {
            total = addSat(total, import_item.specifier.len);
            if (import_item.alias) |alias| total = addSat(total, alias.len);
            total = addSat(total, import_item.resolved_path.len);
        }
        return total;
    }
};

fn cloneImportResolutionResult(allocator: Allocator, source: workspace.ResolutionResult) !workspace.ResolutionResult {
    const diagnostics = try allocator.alloc(workspace.ImportResolutionDiagnostic, source.diagnostics.len);
    var diagnostic_count: usize = 0;
    errdefer {
        for (diagnostics[0..diagnostic_count]) |diagnostic| allocator.free(diagnostic.message);
        allocator.free(diagnostics);
    }
    for (source.diagnostics, 0..) |diagnostic, i| {
        diagnostics[i] = .{
            .range = diagnostic.range,
            .message = try allocator.dupe(u8, diagnostic.message),
        };
        diagnostic_count = i + 1;
    }

    const imports = try allocator.alloc(workspace.ResolvedImport, source.imports.len);
    var import_count: usize = 0;
    errdefer {
        for (imports[0..import_count]) |import_item| {
            allocator.free(import_item.specifier);
            if (import_item.alias) |alias| allocator.free(alias);
            allocator.free(import_item.resolved_path);
        }
        allocator.free(imports);
    }
    for (source.imports, 0..) |import_item, i| {
        const specifier = try allocator.dupe(u8, import_item.specifier);
        errdefer allocator.free(specifier);
        const alias = if (import_item.alias) |alias_text| try allocator.dupe(u8, alias_text) else null;
        errdefer if (alias) |alias_text| allocator.free(alias_text);
        const resolved_path = try allocator.dupe(u8, import_item.resolved_path);
        imports[i] = .{
            .specifier = specifier,
            .alias = alias,
            .resolved_path = resolved_path,
        };
        import_count = i + 1;
    }

    return .{
        .diagnostics = diagnostics,
        .imports = imports,
    };
}

const CachedDiagnosticSource = enum {
    lexer,
    parser,
    sema,
    imports,
};

const CachedDiagnostic = struct {
    source: CachedDiagnosticSource,
    severity: frontend.Severity,
    range: frontend.Range,
    message: []u8,
};

const DiagnosticCacheEntry = struct {
    version: i32,
    generation: u64,
    diagnostics: []CachedDiagnostic,

    fn matches(self: *const DiagnosticCacheEntry, state: DocumentVersionState) bool {
        return self.version == state.version and self.generation == state.generation;
    }

    fn deinit(self: *DiagnosticCacheEntry, allocator: Allocator) void {
        for (self.diagnostics) |diagnostic| {
            allocator.free(diagnostic.message);
        }
        allocator.free(self.diagnostics);
        self.* = undefined;
    }

    fn estimatedByteSize(self: *const DiagnosticCacheEntry) usize {
        var total: usize = @sizeOf(DiagnosticCacheEntry);
        total = addSat(total, mulSat(self.diagnostics.len, @sizeOf(CachedDiagnostic)));
        for (self.diagnostics) |diagnostic| {
            total = addSat(total, diagnostic.message.len);
        }
        return total;
    }
};

const OptionalAllocationScope = struct {
    guard: ?allocation_stats.ScopeGuard = null,

    fn deinit(self: *OptionalAllocationScope) void {
        if (self.guard) |guard| guard.deinit();
        self.guard = null;
    }
};

fn beginAllocationScope(
    tracker: ?*allocation_stats.CountingAllocator,
    scope: allocation_stats.Scope,
) OptionalAllocationScope {
    return .{ .guard = if (tracker) |allocator_tracker| allocator_tracker.beginScope(scope) else null };
}

fn freeCachedDiagnostics(allocator: Allocator, diagnostics: []CachedDiagnostic) void {
    for (diagnostics) |diagnostic| {
        allocator.free(diagnostic.message);
    }
}

fn appendCachedDiagnostic(
    allocator: Allocator,
    diagnostics: *std.ArrayList(CachedDiagnostic),
    source: CachedDiagnosticSource,
    severity: frontend.Severity,
    range: frontend.Range,
    message: []const u8,
) !void {
    const message_copy = try allocator.dupe(u8, message);
    errdefer allocator.free(message_copy);
    try diagnostics.append(allocator, .{
        .source = source,
        .severity = severity,
        .range = range,
        .message = message_copy,
    });
}

fn appendLexerDiagnostics(
    allocator: Allocator,
    diagnostics: *std.ArrayList(CachedDiagnostic),
    lexer_diagnostics: []const token_cache_api.Diagnostic,
) !void {
    for (lexer_diagnostics) |diagnostic| {
        const message = if (diagnostic.suggestion) |suggestion|
            try std.fmt.allocPrint(allocator, "{s} (suggestion: {s})", .{ diagnostic.message, suggestion })
        else
            try allocator.dupe(u8, diagnostic.message);
        errdefer allocator.free(message);
        try diagnostics.append(allocator, .{
            .source = .lexer,
            .severity = lexerSeverityToFrontend(diagnostic.severity),
            .range = lexerRangeToFrontend(diagnostic.range),
            .message = message,
        });
    }
}

fn appendCompilerDiagnostics(
    allocator: Allocator,
    diagnostics: *std.ArrayList(CachedDiagnostic),
    sources: *const compiler.source.SourceStore,
    file_id: compiler.FileId,
    source: CachedDiagnosticSource,
    diagnostic_list: *const compiler.diagnostics.DiagnosticList,
) !void {
    for (diagnostic_list.items.items) |diagnostic| {
        const range = compilerDiagnosticRange(sources, file_id, diagnostic) orelse continue;
        try appendCachedDiagnostic(
            allocator,
            diagnostics,
            source,
            compilerSeverityToFrontend(diagnostic.severity),
            range,
            diagnostic.message,
        );
    }
}

fn appendImportDiagnostics(
    allocator: Allocator,
    diagnostics: *std.ArrayList(CachedDiagnostic),
    import_diagnostics: []const workspace.ImportResolutionDiagnostic,
) !void {
    for (import_diagnostics) |diagnostic| {
        try appendCachedDiagnostic(
            allocator,
            diagnostics,
            .imports,
            .err,
            diagnostic.range,
            diagnostic.message,
        );
    }
}

fn compilerDiagnosticRange(
    sources: *const compiler.source.SourceStore,
    file_id: compiler.FileId,
    diagnostic: compiler.diagnostics.Diagnostic,
) ?frontend.Range {
    const range = if (diagnostic.labels.len > 0) blk: {
        const label = diagnostic.labels[0];
        if (label.location.file_id != file_id) return null;
        break :blk label.location.range;
    } else compiler.TextRange.empty(0);

    return compilerTextRangeToFrontend(sources, file_id, range);
}

fn compilerTextRangeToFrontend(
    sources: *const compiler.source.SourceStore,
    file_id: compiler.FileId,
    range: compiler.TextRange,
) frontend.Range {
    const start = sources.lineColumn(.{
        .file_id = file_id,
        .range = .{ .start = range.start, .end = range.start },
    });
    const end = sources.lineColumn(.{
        .file_id = file_id,
        .range = .{ .start = range.end, .end = range.end },
    });
    return .{
        .start = .{
            .line = oneBasedToZeroBased(start.line),
            .character = oneBasedToZeroBased(start.column),
        },
        .end = .{
            .line = oneBasedToZeroBased(end.line),
            .character = oneBasedToZeroBased(end.column),
        },
    };
}

fn lexerRangeToFrontend(range: lexer_mod.SourceRange) frontend.Range {
    var end_character = oneBasedToZeroBased(range.end_column);
    const start_character = oneBasedToZeroBased(range.start_column);
    if (end_character < start_character) end_character = start_character;
    return .{
        .start = .{
            .line = oneBasedToZeroBased(range.start_line),
            .character = start_character,
        },
        .end = .{
            .line = oneBasedToZeroBased(range.end_line),
            .character = end_character,
        },
    };
}

fn compilerSeverityToFrontend(severity: compiler.diagnostics.Severity) frontend.Severity {
    return switch (severity) {
        .Error => .err,
        .Warning => .warning,
        .Note => .information,
        .Help => .hint,
    };
}

fn lexerSeverityToFrontend(severity: lexer_mod.DiagnosticSeverity) frontend.Severity {
    return switch (severity) {
        .Error => .err,
        .Warning => .warning,
        .Info => .information,
        .Hint => .hint,
    };
}

fn oneBasedToZeroBased(value: u32) u32 {
    return if (value == 0) 0 else value - 1;
}

const IncomingCallTargetUris = std.ArrayList([]u8);

const DocumentStateSource = struct {
    state: *DocumentState,
    source: []const u8,
};

const DocumentStore = struct {
    allocator: Allocator,
    allocator_tracker: ?*allocation_stats.CountingAllocator,
    compiler_db: compiler.CompilerDb,
    package_id: ?compiler.PackageId,
    docs: std.StringHashMap(void),
    cold_docs: std.StringHashMap(ColdDocumentRecord),
    db_records: std.StringHashMap(DocumentDbRecord),
    workspace_index: workspace_index_api.Index,
    incoming_call_targets: std.StringHashMap(IncomingCallTargetUris),
    incoming_call_target_index_built: bool,
    incoming_call_target_index_builds: usize,
    semantic_index_builds: usize,
    call_edge_index_builds: usize,
    diagnostic_cache_builds: usize,
    workspace_index_builds: usize,
    cold_workspace_index_builds: usize,

    fn init(allocator: Allocator, allocator_tracker: ?*allocation_stats.CountingAllocator) DocumentStore {
        return .{
            .allocator = allocator,
            .allocator_tracker = allocator_tracker,
            .compiler_db = compiler.CompilerDb.init(allocator),
            .package_id = null,
            .docs = std.StringHashMap(void).init(allocator),
            .cold_docs = std.StringHashMap(ColdDocumentRecord).init(allocator),
            .db_records = std.StringHashMap(DocumentDbRecord).init(allocator),
            .workspace_index = workspace_index_api.Index.init(allocator),
            .incoming_call_targets = std.StringHashMap(IncomingCallTargetUris).init(allocator),
            .incoming_call_target_index_built = false,
            .incoming_call_target_index_builds = 0,
            .semantic_index_builds = 0,
            .call_edge_index_builds = 0,
            .diagnostic_cache_builds = 0,
            .workspace_index_builds = 0,
            .cold_workspace_index_builds = 0,
        };
    }

    fn allocationScope(self: *DocumentStore, scope: allocation_stats.Scope) OptionalAllocationScope {
        return beginAllocationScope(self.allocator_tracker, scope);
    }

    fn deinit(self: *DocumentStore) void {
        self.clearIncomingCallTargetIndex();
        self.incoming_call_targets.deinit();

        self.workspace_index.deinit();

        var db_it = self.db_records.iterator();
        while (db_it.next()) |entry| {
            self.allocator.free(entry.key_ptr.*);
            entry.value_ptr.deinit();
        }
        self.db_records.deinit();

        var it = self.docs.iterator();
        while (it.next()) |entry| {
            self.allocator.free(entry.key_ptr.*);
        }
        self.docs.deinit();

        var cold_it = self.cold_docs.iterator();
        while (cold_it.next()) |entry| {
            self.allocator.free(entry.key_ptr.*);
            entry.value_ptr.deinit(self.allocator);
        }
        self.cold_docs.deinit();

        self.compiler_db.deinitFrontendOnly();
    }

    fn put(self: *DocumentStore, uri: []const u8, text: []const u8, version: i32) !void {
        self.removeColdDocument(uri);
        self.invalidateWorkspaceIndex(uri);
        self.clearIncomingCallTargetIndex();
        try self.registerDbRecord(uri, text, version);

        const uri_copy = try self.allocator.dupe(u8, uri);
        errdefer self.allocator.free(uri_copy);

        if (self.docs.fetchRemove(uri)) |removed| {
            self.allocator.free(removed.key);
        }

        try self.docs.put(uri_copy, {});
    }

    fn remove(self: *DocumentStore, uri: []const u8) void {
        self.invalidateWorkspaceIndex(uri);
        self.clearIncomingCallTargetIndex();
        if (self.db_records.fetchRemove(uri)) |removed| {
            var record = removed.value;
            record.deinit();
            self.allocator.free(removed.key);
        }
        if (self.docs.fetchRemove(uri)) |removed| {
            self.allocator.free(removed.key);
        }
    }

    fn putColdDocument(self: *DocumentStore, uri: []const u8, normalized_path: []const u8, source: []const u8) !void {
        if (self.docs.contains(uri) or self.db_records.contains(uri)) return;
        if (self.cold_docs.contains(uri)) return;

        var scope = self.allocationScope(.cache_build);
        defer scope.deinit();

        const uri_copy = try self.allocator.dupe(u8, uri);
        errdefer self.allocator.free(uri_copy);

        const path_copy = try self.allocator.dupe(u8, normalized_path);
        errdefer self.allocator.free(path_copy);

        const source_copy = try self.allocator.dupe(u8, source);
        errdefer self.allocator.free(source_copy);

        var state = DocumentState.init(self.allocator);
        errdefer state.deinit();

        try self.cold_docs.put(uri_copy, .{
            .normalized_path = path_copy,
            .source = source_copy,
            .state = state,
        });
    }

    fn removeColdDocument(self: *DocumentStore, uri: []const u8) void {
        if (self.cold_docs.fetchRemove(uri)) |removed| {
            self.invalidateWorkspaceIndex(uri);
            self.clearIncomingCallTargetIndex();
            self.allocator.free(removed.key);
            var record = removed.value;
            record.deinit(self.allocator);
        }
    }

    fn documentVersionStateForUri(self: *DocumentStore, uri: []const u8) ?DocumentVersionState {
        if (self.db_records.get(uri)) |record| {
            return .{ .version = record.version, .generation = record.generation, .is_cold = false };
        }
        if (self.cold_docs.get(uri)) |record| {
            return .{ .version = record.version, .generation = record.generation, .is_cold = true };
        }
        return null;
    }

    fn isStaleOpenVersion(self: *DocumentStore, uri: []const u8, version: i32) bool {
        const state = self.documentVersionStateForUri(uri) orelse return false;
        return !state.is_cold and version <= state.version;
    }

    fn sourceForUri(self: *DocumentStore, uri: []const u8) ?[]const u8 {
        if (self.db_records.get(uri)) |record| {
            return self.compiler_db.sourceText(record.file_id);
        }
        if (self.cold_docs.get(uri)) |record| return record.source;
        return null;
    }

    fn documentStateAndSourceForUri(self: *DocumentStore, uri: []const u8) ?DocumentStateSource {
        if (self.db_records.getPtr(uri)) |record| {
            return .{
                .state = &record.state,
                .source = self.compiler_db.sourceText(record.file_id),
            };
        }
        if (self.cold_docs.getPtr(uri)) |record| {
            return .{
                .state = &record.state,
                .source = record.source,
            };
        }
        return null;
    }

    fn isOpenDocument(self: *DocumentStore, uri: []const u8) bool {
        return self.docs.contains(uri);
    }

    fn openDocumentCount(self: *DocumentStore) usize {
        return self.docs.count();
    }

    fn astFileForUri(self: *DocumentStore, uri: []const u8, stats: ?*phase_stats.Stats) !?*const compiler.ast.AstFile {
        const record = self.db_records.get(uri) orelse return null;
        const needs_parse = !self.compiler_db.hasSyntaxResult(record.file_id);
        const needs_lower = !self.compiler_db.hasAstResult(record.file_id);
        if (needs_parse) phase_stats.record(stats, .parse);
        if (needs_lower) phase_stats.record(stats, .ast_lower);
        return try self.compiler_db.astFile(record.file_id);
    }

    fn definitionAnalysisForUri(self: *DocumentStore, uri: []const u8, source: []const u8, stats: ?*phase_stats.Stats) !?definition_api.Analysis {
        const record = self.db_records.get(uri) orelse return null;

        var scope = self.allocationScope(.cache_build);
        defer scope.deinit();

        const needs_parse = !self.compiler_db.hasSyntaxResult(record.file_id);
        const needs_lower = !self.compiler_db.hasAstResult(record.file_id);
        const needs_item_index = !self.compiler_db.hasItemIndexResult(record.module_id);
        const needs_resolution = !self.compiler_db.hasResolutionResult(record.module_id);
        if (needs_parse) phase_stats.record(stats, .parse);
        if (needs_lower) phase_stats.record(stats, .ast_lower);
        const ast_file = try self.compiler_db.astFile(record.file_id);
        if (needs_item_index) phase_stats.record(stats, .item_index);
        const item_index = try self.compiler_db.itemIndex(record.module_id);
        if (needs_resolution) phase_stats.record(stats, .resolve);
        const resolution = try self.compiler_db.resolveNames(record.module_id);

        return definition_api.Analysis.initBorrowed(
            self.allocator,
            &self.compiler_db.sources,
            record.file_id,
            record.module_id,
            source,
            ast_file,
            item_index,
            resolution,
        );
    }

    fn lineIndexForUri(self: *DocumentStore, uri: []const u8) !?*const line_index_api.LineIndex {
        if (self.db_records.getPtr(uri)) |record| {
            if (record.state.line_index == null) {
                var scope = self.allocationScope(.cache_build);
                defer scope.deinit();

                const source = self.compiler_db.sourceText(record.file_id);
                const state_allocator = record.state.allocator();
                var index = try line_index_api.LineIndex.init(state_allocator, source);
                errdefer index.deinit(state_allocator);
                record.state.line_index = index;
            }
            return &record.state.line_index.?;
        }

        const record = self.cold_docs.getPtr(uri) orelse return null;
        if (record.state.line_index == null) {
            var scope = self.allocationScope(.cache_build);
            defer scope.deinit();

            const state_allocator = record.state.allocator();
            var index = try line_index_api.LineIndex.init(state_allocator, record.source);
            errdefer index.deinit(state_allocator);
            record.state.line_index = index;
        }
        return &record.state.line_index.?;
    }

    fn occurrenceIndexForUri(self: *DocumentStore, uri: []const u8) ?*const references_api.OccurrenceIndex {
        const state_source = self.documentStateAndSourceForUri(uri) orelse return null;
        return if (state_source.state.occurrence_index) |*index| index else null;
    }

    fn importedMemberIndexForUri(self: *DocumentStore, uri: []const u8) ?*const references_api.ImportedMemberIndex {
        const state_source = self.documentStateAndSourceForUri(uri) orelse return null;
        return if (state_source.state.imported_member_index) |*index| index else null;
    }

    fn tokenCacheForUri(self: *DocumentStore, uri: []const u8, stats: ?*phase_stats.Stats) !?*const token_cache_api.Cache {
        const state_source = self.documentStateAndSourceForUri(uri) orelse return null;
        if (state_source.state.token_cache == null) {
            var scope = self.allocationScope(.cache_build);
            defer scope.deinit();

            phase_stats.record(stats, .lex);
            const state_allocator = state_source.state.allocator();
            var cache = try token_cache_api.Cache.initWithScratch(state_allocator, self.allocator, state_source.source);
            errdefer cache.deinit(state_allocator);

            state_source.state.token_cache = cache;
        }
        return &state_source.state.token_cache.?;
    }

    fn semanticIndexForUri(self: *DocumentStore, uri: []const u8, stats: ?*phase_stats.Stats) !?*const semantic_index.SemanticIndex {
        if (self.db_records.getPtr(uri)) |record| {
            if (record.state.semantic_index) |*cached| return cached;

            var scope = self.allocationScope(.cache_build);
            defer scope.deinit();

            const needs_parse = !self.compiler_db.hasSyntaxResult(record.file_id);
            const needs_lower = !self.compiler_db.hasAstResult(record.file_id);
            if (needs_parse) phase_stats.record(stats, .parse);
            const syntax_diagnostics = try self.compiler_db.syntaxDiagnostics(record.file_id);
            if (needs_lower) phase_stats.record(stats, .ast_lower);
            const ast_file = try self.compiler_db.astFile(record.file_id);
            const ast_diagnostics = try self.compiler_db.astDiagnostics(record.file_id);
            const source = self.compiler_db.sourceText(record.file_id);
            const state_allocator = record.state.allocator();
            var index = try semantic_index.indexAstFileWithSourceStoreAlloc(
                state_allocator,
                self.allocator,
                &self.compiler_db.sources,
                record.file_id,
                source,
                ast_file,
                syntax_diagnostics.isEmpty() and ast_diagnostics.isEmpty(),
            );
            errdefer index.deinit(state_allocator);

            record.state.semantic_index = index;
            self.semantic_index_builds = addSat(self.semantic_index_builds, 1);
            return &record.state.semantic_index.?;
        }

        const state_source = self.documentStateAndSourceForUri(uri) orelse return null;
        if (state_source.state.semantic_index) |*cached| return cached;

        var scope = self.allocationScope(.cache_build);
        defer scope.deinit();

        const state_allocator = state_source.state.allocator();
        var index = try semantic_index.indexDocumentWithStatsAlloc(state_allocator, self.allocator, state_source.source, stats);
        errdefer index.deinit(state_allocator);

        state_source.state.semantic_index = index;
        self.semantic_index_builds = addSat(self.semantic_index_builds, 1);
        return &state_source.state.semantic_index.?;
    }

    fn semanticTokensForUri(self: *DocumentStore, uri: []const u8, stats: ?*phase_stats.Stats) !?[]const semantic_tokens_api.SemanticToken {
        const state_source = self.documentStateAndSourceForUri(uri) orelse return null;
        if (state_source.state.semantic_tokens) |cached| return cached;

        var scope = self.allocationScope(.cache_build);
        defer scope.deinit();

        const tokens = (try self.tokenCacheForUri(uri, stats)) orelse return null;

        const maybe_index_ptr = self.semanticIndexForUri(uri, stats) catch |err| switch (err) {
            error.OutOfMemory => return error.OutOfMemory,
            else => null,
        };

        const semantic_tokens = try semantic_tokens_api.tokenizeCached(
            state_source.state.allocator(),
            state_source.source,
            tokens.tokens,
            if (maybe_index_ptr) |index| index.* else null,
        );

        state_source.state.semantic_tokens = semantic_tokens;
        return state_source.state.semantic_tokens.?;
    }

    fn callEdgeIndexForUri(self: *DocumentStore, uri: []const u8, stats: ?*phase_stats.Stats) !?*const call_hierarchy_api.CallEdgeIndex {
        const state_source = self.documentStateAndSourceForUri(uri) orelse return null;
        if (state_source.state.call_edge_index) |*cached| return cached;

        var scope = self.allocationScope(.cache_build);
        defer scope.deinit();

        const tokens = (try self.tokenCacheForUri(uri, stats)) orelse return null;
        const index = (try self.semanticIndexForUri(uri, stats)) orelse return null;

        const state_allocator = state_source.state.allocator();
        var call_edges = try call_hierarchy_api.CallEdgeIndex.init(state_allocator, tokens.tokens, index.symbols);
        errdefer call_edges.deinit(state_allocator);

        state_source.state.call_edge_index = call_edges;
        self.call_edge_index_builds = addSat(self.call_edge_index_builds, 1);
        return &state_source.state.call_edge_index.?;
    }

    fn formattingCacheForUri(
        self: *DocumentStore,
        uri: []const u8,
        options: formatting_api.Options,
        stats: ?*phase_stats.Stats,
    ) !?[]const u8 {
        const state = self.documentVersionStateForUri(uri) orelse return null;
        const state_source = self.documentStateAndSourceForUri(uri) orelse return null;
        if (state_source.state.formatting_cache) |*cached| {
            if (cached.matches(state, options)) return cached.formatted;
            state_source.state.invalidateFormattingCache();
        }

        var scope = self.allocationScope(.cache_build);
        defer scope.deinit();

        phase_stats.record(stats, .formatter);
        const formatted_temp = try formatting_api.formatSourceAlloc(self.allocator, state_source.source, options);
        defer self.allocator.free(formatted_temp);

        const state_allocator = state_source.state.allocator();
        const formatted = try state_allocator.dupe(u8, formatted_temp);

        state_source.state.formatting_cache = .{
            .version = state.version,
            .generation = state.generation,
            .line_width = options.line_width,
            .indent_size = options.indent_size,
            .formatted = formatted,
        };
        return state_source.state.formatting_cache.?.formatted;
    }

    fn importResolutionForUri(
        self: *DocumentStore,
        uri: []const u8,
        workspace_roots: []const []const u8,
        stats: ?*phase_stats.Stats,
    ) !?*const workspace.ResolutionResult {
        const state = self.documentVersionStateForUri(uri) orelse return null;
        const state_source = self.documentStateAndSourceForUri(uri) orelse return null;
        if (state_source.state.import_resolution) |*cached| {
            if (cached.matches(state)) return &cached.result;
            state_source.state.invalidateImportResolution();
        }

        var scope = self.allocationScope(.cache_build);
        defer scope.deinit();

        phase_stats.record(stats, .lex);
        var result_temp = try workspace.resolveDocumentImports(
            self.allocator,
            uri,
            state_source.source,
            .{ .workspace_roots = workspace_roots },
        );
        defer result_temp.deinit(self.allocator);

        const state_allocator = state_source.state.allocator();
        const result = try cloneImportResolutionResult(state_allocator, result_temp);

        state_source.state.import_resolution = .{
            .version = state.version,
            .generation = state.generation,
            .result = result,
        };
        return &state_source.state.import_resolution.?.result;
    }

    fn diagnosticCacheForUri(
        self: *DocumentStore,
        uri: []const u8,
        workspace_roots: []const []const u8,
        stats: ?*phase_stats.Stats,
    ) !?*const DiagnosticCacheEntry {
        const state = self.documentVersionStateForUri(uri) orelse return null;
        const state_source = self.documentStateAndSourceForUri(uri) orelse return null;
        if (state_source.state.diagnostic_cache) |*cached| {
            if (cached.matches(state)) return cached;
            state_source.state.invalidateDiagnosticCache();
        }

        var scope = self.allocationScope(.cache_build);
        defer scope.deinit();

        if (self.db_records.getPtr(uri)) |record| {
            const state_allocator = state_source.state.allocator();
            var diagnostics = std.ArrayList(CachedDiagnostic){};
            errdefer freeCachedDiagnostics(state_allocator, diagnostics.items);
            defer diagnostics.deinit(state_allocator);

            if (try self.tokenCacheForUri(uri, stats)) |token_cache| {
                try appendLexerDiagnostics(state_allocator, &diagnostics, token_cache.diagnostics);
            }

            const needs_parse = !self.compiler_db.hasSyntaxResult(record.file_id);
            const needs_lower = !self.compiler_db.hasAstResult(record.file_id);
            if (needs_parse) phase_stats.record(stats, .parse);
            const syntax_diagnostics = try self.compiler_db.syntaxDiagnostics(record.file_id);
            try appendCompilerDiagnostics(
                state_allocator,
                &diagnostics,
                &self.compiler_db.sources,
                record.file_id,
                .parser,
                syntax_diagnostics,
            );

            if (needs_lower) phase_stats.record(stats, .ast_lower);
            const ast_file = try self.compiler_db.astFile(record.file_id);
            const ast_diagnostics = try self.compiler_db.astDiagnostics(record.file_id);
            try appendCompilerDiagnostics(
                state_allocator,
                &diagnostics,
                &self.compiler_db.sources,
                record.file_id,
                .parser,
                ast_diagnostics,
            );

            if (syntax_diagnostics.isEmpty() and ast_diagnostics.isEmpty()) {
                if (!self.compiler_db.hasItemIndexResult(record.module_id)) phase_stats.record(stats, .item_index);
                if (!self.compiler_db.hasResolutionResult(record.module_id)) phase_stats.record(stats, .resolve);
                const resolution_diagnostics = try self.compiler_db.resolutionDiagnostics(record.module_id);
                try appendCompilerDiagnostics(
                    state_allocator,
                    &diagnostics,
                    &self.compiler_db.sources,
                    record.file_id,
                    .sema,
                    resolution_diagnostics,
                );

                if (!self.compiler_db.hasConstEvalResult(record.module_id)) phase_stats.record(stats, .const_eval);
                const const_eval_diagnostics = try self.compiler_db.constEvalDiagnostics(record.module_id);
                try appendCompilerDiagnostics(
                    state_allocator,
                    &diagnostics,
                    &self.compiler_db.sources,
                    record.file_id,
                    .sema,
                    const_eval_diagnostics,
                );

                if (ast_file.bodies.len > 0) {
                    const key: compiler.sema.TypeCheckKey = .{ .body = compiler.ast.BodyId.fromIndex(0) };
                    if (!self.compiler_db.hasTypeCheckResult(record.module_id, key)) phase_stats.record(stats, .type_check);
                    const typecheck_diagnostics = try self.compiler_db.typeCheckDiagnostics(record.module_id, key);
                    try appendCompilerDiagnostics(
                        state_allocator,
                        &diagnostics,
                        &self.compiler_db.sources,
                        record.file_id,
                        .sema,
                        typecheck_diagnostics,
                    );
                }
            }

            if (try self.importResolutionForUri(uri, workspace_roots, stats)) |import_resolution| {
                try appendImportDiagnostics(state_allocator, &diagnostics, import_resolution.diagnostics);
            }

            state_source.state.diagnostic_cache = .{
                .version = state.version,
                .generation = state.generation,
                .diagnostics = try diagnostics.toOwnedSlice(state_allocator),
            };
            self.diagnostic_cache_builds = addSat(self.diagnostic_cache_builds, 1);
            return &state_source.state.diagnostic_cache.?;
        }

        var analysis = try frontend.analyzeDocumentWithStats(self.allocator, state_source.source, stats);
        defer analysis.deinit(self.allocator);

        const import_diagnostics: []const workspace.ImportResolutionDiagnostic = if (try self.importResolutionForUri(uri, workspace_roots, stats)) |import_resolution|
            import_resolution.diagnostics
        else
            &.{};

        const total = analysis.diagnostics.len + import_diagnostics.len;
        const state_allocator = state_source.state.allocator();
        const diagnostics = try state_allocator.alloc(CachedDiagnostic, total);
        var count: usize = 0;
        errdefer {
            for (diagnostics[0..count]) |diagnostic| {
                state_allocator.free(diagnostic.message);
            }
            state_allocator.free(diagnostics);
        }

        for (analysis.diagnostics) |diagnostic| {
            diagnostics[count] = .{
                .source = cachedDiagnosticSourceFromFrontend(diagnostic.source),
                .severity = diagnostic.severity,
                .range = diagnostic.range,
                .message = try state_allocator.dupe(u8, diagnostic.message),
            };
            count += 1;
        }

        for (import_diagnostics) |diagnostic| {
            diagnostics[count] = .{
                .source = .imports,
                .severity = .err,
                .range = diagnostic.range,
                .message = try state_allocator.dupe(u8, diagnostic.message),
            };
            count += 1;
        }

        state_source.state.diagnostic_cache = .{
            .version = state.version,
            .generation = state.generation,
            .diagnostics = diagnostics,
        };
        self.diagnostic_cache_builds = addSat(self.diagnostic_cache_builds, 1);
        return &state_source.state.diagnostic_cache.?;
    }

    fn workspaceEntryForUri(
        self: *DocumentStore,
        uri: []const u8,
        features: workspace_index_api.FeatureSet,
        stats: ?*phase_stats.Stats,
    ) !?*const workspace_index_api.FileEntry {
        const document_state = self.documentVersionStateForUri(uri) orelse return null;
        if (self.workspace_index.getFresh(uri, document_state.version, document_state.generation, features)) |entry| return entry;

        var requested_features = features;
        if (self.workspace_index.getFreshAny(uri, document_state.version, document_state.generation)) |entry| {
            requested_features = entry.features.merged(features);
        }

        var scope = self.allocationScope(.cache_build);
        defer scope.deinit();

        const line_index = (try self.lineIndexForUri(uri)) orelse return null;
        const semantic_symbols = if (requested_features.symbols)
            ((try self.semanticIndexForUri(uri, stats)) orelse return null).symbols
        else
            &[_]semantic_index.Symbol{};
        const occurrence_index = if (requested_features.occurrences)
            self.occurrenceIndexForUri(uri) orelse return null
        else
            null;
        const imported_member_index = if (requested_features.imported_members)
            self.importedMemberIndexForUri(uri) orelse return null
        else
            null;
        const call_edge_index = if (requested_features.call_edges)
            (try self.callEdgeIndexForUri(uri, stats)) orelse return null
        else
            null;
        const empty_imports: []const workspace.ResolvedImport = &.{};

        var entry = try workspace_index_api.FileEntry.init(
            self.allocator,
            uri,
            document_state.version,
            document_state.generation,
            document_state.is_cold,
            requested_features,
            line_index,
            semantic_symbols,
            empty_imports,
            occurrence_index,
            imported_member_index,
            call_edge_index,
        );
        errdefer entry.deinit();

        try self.workspace_index.upsert(entry);
        if (document_state.is_cold) {
            self.cold_workspace_index_builds = addSat(self.cold_workspace_index_builds, 1);
        } else {
            self.workspace_index_builds = addSat(self.workspace_index_builds, 1);
        }
        return self.workspace_index.getFresh(uri, document_state.version, document_state.generation, features).?;
    }

    fn ensureIncomingCallTargetIndex(self: *DocumentStore, stats: ?*phase_stats.Stats) !void {
        if (self.incoming_call_target_index_built) return;

        var scope = self.allocationScope(.cache_build);
        defer scope.deinit();

        self.clearIncomingCallTargetIndex();
        errdefer self.clearIncomingCallTargetIndex();

        var doc_it = self.docs.iterator();
        while (doc_it.next()) |entry| {
            const doc_uri = entry.key_ptr.*;
            const workspace_entry = (try self.workspaceEntryForUri(doc_uri, .calls, stats)) orelse continue;
            for (workspace_entry.call_edges) |edge| {
                try self.appendIncomingCallTarget(edge.callee_name, doc_uri);
            }
        }

        self.incoming_call_target_index_built = true;
        self.incoming_call_target_index_builds = addSat(self.incoming_call_target_index_builds, 1);
    }

    fn incomingCallTargetUris(self: *DocumentStore, target_name: []const u8) ?[]const []const u8 {
        const uris = self.incoming_call_targets.getPtr(target_name) orelse return null;
        return uris.items;
    }

    fn appendIncomingCallTarget(self: *DocumentStore, callee_name: []const u8, uri: []const u8) !void {
        if (self.incoming_call_targets.getPtr(callee_name)) |uris| {
            if (containsString(uris.items, uri)) return;

            const uri_copy = try self.allocator.dupe(u8, uri);
            uris.append(self.allocator, uri_copy) catch |err| {
                self.allocator.free(uri_copy);
                return err;
            };
            return;
        }

        const key_copy = try self.allocator.dupe(u8, callee_name);
        errdefer self.allocator.free(key_copy);

        var uris = IncomingCallTargetUris{};
        errdefer {
            for (uris.items) |owned_uri| self.allocator.free(owned_uri);
            uris.deinit(self.allocator);
        }

        const uri_copy = try self.allocator.dupe(u8, uri);
        uris.append(self.allocator, uri_copy) catch |err| {
            self.allocator.free(uri_copy);
            return err;
        };
        try self.incoming_call_targets.put(key_copy, uris);
    }

    fn clearIncomingCallTargetIndex(self: *DocumentStore) void {
        var it = self.incoming_call_targets.iterator();
        while (it.next()) |entry| {
            self.allocator.free(entry.key_ptr.*);
            for (entry.value_ptr.items) |uri| self.allocator.free(uri);
            entry.value_ptr.deinit(self.allocator);
        }
        self.incoming_call_targets.clearRetainingCapacity();
        self.incoming_call_target_index_built = false;
    }

    fn rebuildOccurrenceIndex(self: *DocumentStore, uri: []const u8, source: []const u8, stats: ?*phase_stats.Stats) !void {
        self.invalidateOccurrenceIndex(uri);
        const state_source = self.documentStateAndSourceForUri(uri) orelse return;
        const token_cache = (try self.tokenCacheForUri(uri, stats)) orelse return;

        var scope = self.allocationScope(.cache_build);
        defer scope.deinit();

        const state_allocator = state_source.state.allocator();
        if (try self.definitionAnalysisForUri(uri, source, stats)) |analysis_value| {
            var analysis = analysis_value;
            defer analysis.deinit();

            var index = try references_api.OccurrenceIndex.init(state_allocator, source, token_cache.tokens, &analysis);
            errdefer index.deinit(state_allocator);

            if (index.occurrences.len == 0) {
                index.deinit(state_allocator);
                return;
            }

            state_source.state.occurrence_index = index;
            return;
        }

        phase_stats.record(stats, .parse);
        phase_stats.record(stats, .ast_lower);
        phase_stats.record(stats, .item_index);
        phase_stats.record(stats, .resolve);
        var analysis = (definition_api.Analysis.init(self.allocator, source) catch |err| switch (err) {
            error.OutOfMemory => return error.OutOfMemory,
            else => return,
        }) orelse return;
        defer analysis.deinit();

        var index = try references_api.OccurrenceIndex.init(state_allocator, source, token_cache.tokens, &analysis);
        errdefer index.deinit(state_allocator);

        if (index.occurrences.len == 0) {
            index.deinit(state_allocator);
            return;
        }

        state_source.state.occurrence_index = index;
    }

    fn rebuildImportedMemberIndex(
        self: *DocumentStore,
        uri: []const u8,
        imports: []const workspace.ResolvedImport,
        stats: ?*phase_stats.Stats,
    ) !void {
        self.invalidateImportedMemberIndex(uri);
        const state_source = self.documentStateAndSourceForUri(uri) orelse return;
        const token_cache = (try self.tokenCacheForUri(uri, stats)) orelse return;

        var scope = self.allocationScope(.cache_build);
        defer scope.deinit();

        const import_bindings = try self.allocator.alloc(references_api.ImportBinding, imports.len);
        defer self.allocator.free(import_bindings);

        var binding_count: usize = 0;
        defer {
            for (import_bindings[0..binding_count]) |binding| {
                self.allocator.free(binding.resolved_path);
            }
        }
        for (imports) |import_item| {
            const alias = import_item.alias orelse continue;
            const normalized_path = try workspace.normalizePathAlloc(self.allocator, import_item.resolved_path);
            import_bindings[binding_count] = .{
                .alias = alias,
                .resolved_path = normalized_path,
            };
            binding_count += 1;
        }

        const state_allocator = state_source.state.allocator();
        var index = try references_api.buildImportedMemberIndexFromTokens(
            state_allocator,
            token_cache.tokens,
            import_bindings[0..binding_count],
        );
        errdefer references_api.deinitImportedMemberIndex(state_allocator, &index);

        if (index.occurrences.len == 0) {
            references_api.deinitImportedMemberIndex(state_allocator, &index);
            return;
        }

        state_source.state.imported_member_index = index;
    }

    fn registerDbRecord(self: *DocumentStore, uri: []const u8, text: []const u8, version: i32) !void {
        if (self.db_records.getPtr(uri)) |record| {
            try self.compiler_db.updateSourceFileFrontendOnly(record.file_id, text);
            record.version = version;
            record.generation = bumpGeneration(record.generation);
            record.state.reset();
            return;
        }

        const db_path = try self.dbPathForUri(uri);
        defer self.allocator.free(db_path);

        const package_id = try self.getOrCreatePackage();
        const file_id = try self.compiler_db.addSourceFile(db_path, text);
        const module_id = try self.compiler_db.addModule(package_id, file_id, moduleNameForPath(db_path));

        const uri_copy = try self.allocator.dupe(u8, uri);
        errdefer self.allocator.free(uri_copy);

        var state = DocumentState.init(self.allocator);
        errdefer state.deinit();

        const record: DocumentDbRecord = .{
            .file_id = file_id,
            .module_id = module_id,
            .version = version,
            .generation = 1,
            .state = state,
        };
        try self.db_records.put(uri_copy, record);
    }

    fn getOrCreatePackage(self: *DocumentStore) !compiler.PackageId {
        if (self.package_id) |package_id| return package_id;
        const package_id = try self.compiler_db.addPackage("lsp-open-documents");
        self.package_id = package_id;
        return package_id;
    }

    fn dbPathForUri(self: *DocumentStore, uri: []const u8) ![]u8 {
        if (try workspace.fileUriToPathAlloc(self.allocator, uri)) |path| {
            return path;
        }
        return try self.allocator.dupe(u8, uri);
    }

    fn invalidateLineIndex(self: *DocumentStore, uri: []const u8) void {
        if (self.db_records.getPtr(uri)) |record| record.state.invalidateLineIndex();
        if (self.cold_docs.getPtr(uri)) |record| record.state.invalidateLineIndex();
    }

    fn invalidateTokenCache(self: *DocumentStore, uri: []const u8) void {
        if (self.db_records.getPtr(uri)) |record| record.state.invalidateTokenCache();
        if (self.cold_docs.getPtr(uri)) |record| record.state.invalidateTokenCache();
    }

    fn invalidateSemanticIndex(self: *DocumentStore, uri: []const u8) void {
        if (self.db_records.getPtr(uri)) |record| record.state.invalidateSemanticIndex();
        if (self.cold_docs.getPtr(uri)) |record| record.state.invalidateSemanticIndex();
        self.invalidateWorkspaceIndex(uri);
        self.clearIncomingCallTargetIndex();
    }

    fn invalidateSemanticTokenCache(self: *DocumentStore, uri: []const u8) void {
        if (self.db_records.getPtr(uri)) |record| record.state.invalidateSemanticTokens();
        if (self.cold_docs.getPtr(uri)) |record| record.state.invalidateSemanticTokens();
    }

    fn invalidateOccurrenceIndex(self: *DocumentStore, uri: []const u8) void {
        if (self.db_records.getPtr(uri)) |record| record.state.invalidateOccurrenceIndex();
        if (self.cold_docs.getPtr(uri)) |record| record.state.invalidateOccurrenceIndex();
        self.invalidateWorkspaceIndex(uri);
    }

    fn invalidateImportedMemberIndex(self: *DocumentStore, uri: []const u8) void {
        if (self.db_records.getPtr(uri)) |record| record.state.invalidateImportedMemberIndex();
        if (self.cold_docs.getPtr(uri)) |record| record.state.invalidateImportedMemberIndex();
        self.invalidateWorkspaceIndex(uri);
    }

    fn invalidateCallEdgeIndex(self: *DocumentStore, uri: []const u8) void {
        if (self.db_records.getPtr(uri)) |record| record.state.invalidateCallEdgeIndex();
        if (self.cold_docs.getPtr(uri)) |record| record.state.invalidateCallEdgeIndex();
        self.invalidateWorkspaceIndex(uri);
        self.clearIncomingCallTargetIndex();
    }

    fn invalidateDiagnosticCache(self: *DocumentStore, uri: []const u8) void {
        if (self.db_records.getPtr(uri)) |record| record.state.invalidateDiagnosticCache();
        if (self.cold_docs.getPtr(uri)) |record| record.state.invalidateDiagnosticCache();
    }

    fn invalidateFormattingCache(self: *DocumentStore, uri: []const u8) void {
        if (self.db_records.getPtr(uri)) |record| record.state.invalidateFormattingCache();
        if (self.cold_docs.getPtr(uri)) |record| record.state.invalidateFormattingCache();
    }

    fn invalidateImportResolution(self: *DocumentStore, uri: []const u8) void {
        if (self.db_records.getPtr(uri)) |record| {
            record.state.invalidateImportResolution();
            record.state.invalidateImportedMemberIndex();
        }
        if (self.cold_docs.getPtr(uri)) |record| {
            record.state.invalidateImportResolution();
            record.state.invalidateImportedMemberIndex();
        }
        self.invalidateDiagnosticCache(uri);
        self.invalidateWorkspaceIndex(uri);
        self.clearIncomingCallTargetIndex();
    }

    fn clearImportResolutionCache(self: *DocumentStore) void {
        var db_iterator = self.db_records.iterator();
        while (db_iterator.next()) |entry| {
            entry.value_ptr.state.reset();
            self.invalidateWorkspaceIndex(entry.key_ptr.*);
        }
        var cold_iterator = self.cold_docs.iterator();
        while (cold_iterator.next()) |entry| {
            entry.value_ptr.state.reset();
            self.invalidateWorkspaceIndex(entry.key_ptr.*);
        }
        self.clearIncomingCallTargetIndex();
    }

    fn invalidateWorkspaceIndex(self: *DocumentStore, uri: []const u8) void {
        self.workspace_index.invalidate(uri);
    }

    fn openSourceBytes(self: *DocumentStore) usize {
        var total: usize = 0;
        var iterator = self.docs.keyIterator();
        while (iterator.next()) |uri| {
            if (self.sourceForUri(uri.*)) |source| {
                total = addSat(total, source.len);
            }
        }
        return total;
    }

    fn coldDocumentCount(self: *DocumentStore) usize {
        return self.cold_docs.count();
    }

    fn coldSourceBytes(self: *DocumentStore) usize {
        var total: usize = 0;
        var iterator = self.cold_docs.valueIterator();
        while (iterator.next()) |record| {
            total = addSat(total, record.source.len);
        }
        return total;
    }

    fn lineIndexBytes(self: *DocumentStore) usize {
        var total: usize = 0;
        var db_iterator = self.db_records.valueIterator();
        while (db_iterator.next()) |record| {
            total = addSat(total, record.state.lineIndexBytes());
        }
        var cold_iterator = self.cold_docs.valueIterator();
        while (cold_iterator.next()) |record| {
            total = addSat(total, record.state.lineIndexBytes());
        }
        return total;
    }

    fn tokenCacheBytes(self: *DocumentStore) usize {
        var total: usize = 0;
        var db_iterator = self.db_records.valueIterator();
        while (db_iterator.next()) |record| {
            total = addSat(total, record.state.tokenCacheBytes());
        }
        var cold_iterator = self.cold_docs.valueIterator();
        while (cold_iterator.next()) |record| {
            total = addSat(total, record.state.tokenCacheBytes());
        }
        return total;
    }

    fn tokenCount(self: *DocumentStore) usize {
        var total: usize = 0;
        var db_iterator = self.db_records.valueIterator();
        while (db_iterator.next()) |record| {
            total = addSat(total, record.state.tokenCount());
        }
        var cold_iterator = self.cold_docs.valueIterator();
        while (cold_iterator.next()) |record| {
            total = addSat(total, record.state.tokenCount());
        }
        return total;
    }

    fn tokenDiagnosticCount(self: *DocumentStore) usize {
        var total: usize = 0;
        var db_iterator = self.db_records.valueIterator();
        while (db_iterator.next()) |record| {
            total = addSat(total, record.state.tokenDiagnosticCount());
        }
        var cold_iterator = self.cold_docs.valueIterator();
        while (cold_iterator.next()) |record| {
            total = addSat(total, record.state.tokenDiagnosticCount());
        }
        return total;
    }

    fn importResolutionBytes(self: *DocumentStore) usize {
        var total: usize = 0;
        var db_iterator = self.db_records.valueIterator();
        while (db_iterator.next()) |record| {
            total = addSat(total, record.state.importResolutionBytes());
        }
        var cold_iterator = self.cold_docs.valueIterator();
        while (cold_iterator.next()) |record| {
            total = addSat(total, record.state.importResolutionBytes());
        }
        return total;
    }

    fn semanticIndexBytes(self: *DocumentStore) usize {
        var total: usize = 0;
        var db_iterator = self.db_records.valueIterator();
        while (db_iterator.next()) |record| {
            total = addSat(total, record.state.semanticIndexBytes());
        }
        var cold_iterator = self.cold_docs.valueIterator();
        while (cold_iterator.next()) |record| {
            total = addSat(total, record.state.semanticIndexBytes());
        }
        return total;
    }

    fn symbolCount(self: *DocumentStore) usize {
        var total: usize = 0;
        var db_iterator = self.db_records.valueIterator();
        while (db_iterator.next()) |record| {
            total = addSat(total, record.state.symbolCount());
        }
        var cold_iterator = self.cold_docs.valueIterator();
        while (cold_iterator.next()) |record| {
            total = addSat(total, record.state.symbolCount());
        }
        return total;
    }

    fn importedMemberIndexBytes(self: *DocumentStore) usize {
        var total: usize = 0;
        var db_iterator = self.db_records.valueIterator();
        while (db_iterator.next()) |record| {
            total = addSat(total, record.state.importedMemberIndexBytes());
        }
        var cold_iterator = self.cold_docs.valueIterator();
        while (cold_iterator.next()) |record| {
            total = addSat(total, record.state.importedMemberIndexBytes());
        }
        return total;
    }

    fn importedMemberCount(self: *DocumentStore) usize {
        var total: usize = 0;
        var db_iterator = self.db_records.valueIterator();
        while (db_iterator.next()) |record| {
            total = addSat(total, record.state.importedMemberCount());
        }
        var cold_iterator = self.cold_docs.valueIterator();
        while (cold_iterator.next()) |record| {
            total = addSat(total, record.state.importedMemberCount());
        }
        return total;
    }

    fn occurrenceIndexBytes(self: *DocumentStore) usize {
        var total: usize = 0;
        var db_iterator = self.db_records.valueIterator();
        while (db_iterator.next()) |record| {
            total = addSat(total, record.state.occurrenceIndexBytes());
        }
        var cold_iterator = self.cold_docs.valueIterator();
        while (cold_iterator.next()) |record| {
            total = addSat(total, record.state.occurrenceIndexBytes());
        }
        return total;
    }

    fn occurrenceCount(self: *DocumentStore) usize {
        var total: usize = 0;
        var db_iterator = self.db_records.valueIterator();
        while (db_iterator.next()) |record| {
            total = addSat(total, record.state.occurrenceCount());
        }
        var cold_iterator = self.cold_docs.valueIterator();
        while (cold_iterator.next()) |record| {
            total = addSat(total, record.state.occurrenceCount());
        }
        return total;
    }

    fn callEdgeIndexBytes(self: *DocumentStore) usize {
        var total: usize = 0;
        var db_iterator = self.db_records.valueIterator();
        while (db_iterator.next()) |record| {
            total = addSat(total, record.state.callEdgeIndexBytes());
        }
        var cold_iterator = self.cold_docs.valueIterator();
        while (cold_iterator.next()) |record| {
            total = addSat(total, record.state.callEdgeIndexBytes());
        }
        return total;
    }

    fn callEdgeCount(self: *DocumentStore) usize {
        var total: usize = 0;
        var db_iterator = self.db_records.valueIterator();
        while (db_iterator.next()) |record| {
            total = addSat(total, record.state.callEdgeCount());
        }
        var cold_iterator = self.cold_docs.valueIterator();
        while (cold_iterator.next()) |record| {
            total = addSat(total, record.state.callEdgeCount());
        }
        return total;
    }

    fn diagnosticCacheBytes(self: *DocumentStore) usize {
        var total: usize = 0;
        var db_iterator = self.db_records.valueIterator();
        while (db_iterator.next()) |record| {
            total = addSat(total, record.state.diagnosticCacheBytes());
        }
        var cold_iterator = self.cold_docs.valueIterator();
        while (cold_iterator.next()) |record| {
            total = addSat(total, record.state.diagnosticCacheBytes());
        }
        return total;
    }

    fn formattingCacheBytes(self: *DocumentStore) usize {
        var total: usize = 0;
        var db_iterator = self.db_records.valueIterator();
        while (db_iterator.next()) |record| {
            total = addSat(total, record.state.formattingCacheBytes());
        }
        var cold_iterator = self.cold_docs.valueIterator();
        while (cold_iterator.next()) |record| {
            total = addSat(total, record.state.formattingCacheBytes());
        }
        return total;
    }

    fn formattingCacheEntries(self: *DocumentStore) usize {
        var total: usize = 0;
        var db_iterator = self.db_records.valueIterator();
        while (db_iterator.next()) |record| {
            total = addSat(total, record.state.formattingCacheEntries());
        }
        var cold_iterator = self.cold_docs.valueIterator();
        while (cold_iterator.next()) |record| {
            total = addSat(total, record.state.formattingCacheEntries());
        }
        return total;
    }

    fn cacheBuilderCapacityRequested(self: *DocumentStore) usize {
        var total: usize = 0;
        var db_state_iterator = self.db_records.valueIterator();
        while (db_state_iterator.next()) |record| {
            total = addSat(total, record.state.cacheBuilderCapacityRequested());
        }
        var cold_state_iterator = self.cold_docs.valueIterator();
        while (cold_state_iterator.next()) |record| {
            total = addSat(total, record.state.cacheBuilderCapacityRequested());
        }
        total = addSat(total, self.workspace_index.builderCapacityRequested());
        return total;
    }

    fn cacheBuilderItemsBuilt(self: *DocumentStore) usize {
        var total: usize = 0;
        var db_state_iterator = self.db_records.valueIterator();
        while (db_state_iterator.next()) |record| {
            total = addSat(total, record.state.cacheBuilderItemsBuilt());
        }
        var cold_state_iterator = self.cold_docs.valueIterator();
        while (cold_state_iterator.next()) |record| {
            total = addSat(total, record.state.cacheBuilderItemsBuilt());
        }
        total = addSat(total, self.workspace_index.builderItemsBuilt());
        return total;
    }

    fn cacheBuilderUnusedCapacity(self: *DocumentStore) usize {
        var total: usize = 0;
        var db_state_iterator = self.db_records.valueIterator();
        while (db_state_iterator.next()) |record| {
            total = addSat(total, record.state.cacheBuilderUnusedCapacity());
        }
        var cold_state_iterator = self.cold_docs.valueIterator();
        while (cold_state_iterator.next()) |record| {
            total = addSat(total, record.state.cacheBuilderUnusedCapacity());
        }
        total = addSat(total, self.workspace_index.builderUnusedCapacity());
        return total;
    }

    fn cacheBuilderGrowthEvents(self: *DocumentStore) usize {
        var total: usize = 0;
        var db_state_iterator = self.db_records.valueIterator();
        while (db_state_iterator.next()) |record| {
            total = addSat(total, record.state.cacheBuilderGrowthEvents());
        }
        var cold_state_iterator = self.cold_docs.valueIterator();
        while (cold_state_iterator.next()) |record| {
            total = addSat(total, record.state.cacheBuilderGrowthEvents());
        }
        total = addSat(total, self.workspace_index.builderGrowthEvents());
        return total;
    }

    fn cacheSideMapCapacityRequested(self: *DocumentStore) usize {
        return self.workspace_index.sideMapCapacityRequested();
    }

    fn cacheSideMapItemsBuilt(self: *DocumentStore) usize {
        return self.workspace_index.sideMapItemsBuilt();
    }

    fn cacheSideMapUnusedCapacity(self: *DocumentStore) usize {
        return self.workspace_index.sideMapUnusedCapacity();
    }

    fn cacheSideMapGrowthEvents(self: *DocumentStore) usize {
        return self.workspace_index.sideMapGrowthEvents();
    }

    fn documentStateBytes(self: *DocumentStore) usize {
        var total: usize = 0;
        total = addSat(total, self.openSourceBytes());
        total = addSat(total, self.lineIndexBytes());
        total = addSat(total, self.tokenCacheBytes());
        total = addSat(total, self.semanticIndexBytes());
        total = addSat(total, self.occurrenceIndexBytes());
        total = addSat(total, self.importedMemberIndexBytes());
        total = addSat(total, self.callEdgeIndexBytes());
        total = addSat(total, self.formattingCacheBytes());
        total = addSat(total, self.importResolutionBytes());
        total = addSat(total, self.diagnosticCacheBytes());
        total = addSat(total, self.db_records.count() * @sizeOf(DocumentDbRecord));
        return total;
    }

    fn workspaceIndexEntries(self: *DocumentStore) usize {
        return self.workspace_index.entries.count();
    }

    fn workspaceIndexBytes(self: *DocumentStore) usize {
        return self.workspace_index.current_bytes;
    }

    fn workspaceIndexMaxBytes(self: *DocumentStore) usize {
        return self.workspace_index.max_bytes;
    }

    fn workspaceIndexEvictions(self: *DocumentStore) usize {
        return self.workspace_index.evictions;
    }

    fn workspaceIndexSymbolCount(self: *DocumentStore) usize {
        var total: usize = 0;
        var iterator = self.workspace_index.entries.valueIterator();
        while (iterator.next()) |entry| {
            total = addSat(total, entry.symbols.len);
        }
        return total;
    }

    fn workspaceIndexRootSymbolCount(self: *DocumentStore) usize {
        var total: usize = 0;
        var iterator = self.workspace_index.entries.valueIterator();
        while (iterator.next()) |entry| {
            total = addSat(total, entry.root_symbol_indexes.len);
        }
        return total;
    }

    fn workspaceIndexCallableSymbolCount(self: *DocumentStore) usize {
        var total: usize = 0;
        var iterator = self.workspace_index.entries.valueIterator();
        while (iterator.next()) |entry| {
            total = addSat(total, entry.callable_symbol_indexes.len);
        }
        return total;
    }

    fn workspaceIndexImportCount(self: *DocumentStore) usize {
        var total: usize = 0;
        var iterator = self.workspace_index.entries.valueIterator();
        while (iterator.next()) |entry| {
            total = addSat(total, entry.imports.len);
        }
        return total;
    }

    fn workspaceIndexOccurrenceCount(self: *DocumentStore) usize {
        var total: usize = 0;
        var iterator = self.workspace_index.entries.valueIterator();
        while (iterator.next()) |entry| {
            total = addSat(total, entry.occurrences.len);
        }
        return total;
    }

    fn workspaceIndexImportedMemberCount(self: *DocumentStore) usize {
        var total: usize = 0;
        var iterator = self.workspace_index.entries.valueIterator();
        while (iterator.next()) |entry| {
            total = addSat(total, entry.imported_members.len);
        }
        return total;
    }

    fn workspaceIndexCallEdgeCount(self: *DocumentStore) usize {
        var total: usize = 0;
        var iterator = self.workspace_index.entries.valueIterator();
        while (iterator.next()) |entry| {
            total = addSat(total, entry.call_edges.len);
        }
        return total;
    }

    fn workspaceIndexInternedStringBytes(self: *DocumentStore) usize {
        var total: usize = 0;
        var iterator = self.workspace_index.entries.valueIterator();
        while (iterator.next()) |entry| {
            total = addSat(total, entry.interned_string_bytes);
        }
        return total;
    }

    fn workspaceIndexInternedStringCapacityRequested(self: *DocumentStore) usize {
        return self.workspace_index.internedStringCapacityRequested();
    }

    fn workspaceIndexInternedStringItemsBuilt(self: *DocumentStore) usize {
        return self.workspace_index.internedStringItemsBuilt();
    }

    fn workspaceIndexInternedStringUnusedCapacity(self: *DocumentStore) usize {
        return self.workspace_index.internedStringUnusedCapacity();
    }

    fn workspaceIndexInternedStringGrowthEvents(self: *DocumentStore) usize {
        return self.workspace_index.internedStringGrowthEvents();
    }

    fn coldWorkspaceIndexEntries(self: *DocumentStore) usize {
        return self.workspace_index.coldEntryCount();
    }

    fn coldWorkspaceIndexBytes(self: *DocumentStore) usize {
        return self.workspace_index.coldBytes();
    }

    fn coldWorkspaceIndexInternedStringBytes(self: *DocumentStore) usize {
        return self.workspace_index.coldInternedStringBytes();
    }

    fn coldWorkspaceIndexInternedStringCount(self: *DocumentStore) usize {
        return self.workspace_index.coldInternedStringCount();
    }

    fn coldWorkspaceIndexDuplicateStringBytesSaved(self: *DocumentStore) usize {
        return self.workspace_index.coldDuplicateStringBytesSaved();
    }

    fn coldWorkspaceIndexInternedStringCapacityRequested(self: *DocumentStore) usize {
        return self.workspace_index.coldInternedStringCapacityRequested();
    }

    fn coldWorkspaceIndexInternedStringItemsBuilt(self: *DocumentStore) usize {
        return self.workspace_index.coldInternedStringItemsBuilt();
    }

    fn coldWorkspaceIndexInternedStringUnusedCapacity(self: *DocumentStore) usize {
        return self.workspace_index.coldInternedStringUnusedCapacity();
    }

    fn coldWorkspaceIndexInternedStringGrowthEvents(self: *DocumentStore) usize {
        return self.workspace_index.coldInternedStringGrowthEvents();
    }

    fn workspaceIndexInternedStringCount(self: *DocumentStore) usize {
        var total: usize = 0;
        var iterator = self.workspace_index.entries.valueIterator();
        while (iterator.next()) |entry| {
            total = addSat(total, entry.interned_string_count);
        }
        return total;
    }

    fn workspaceIndexDuplicateStringBytesSaved(self: *DocumentStore) usize {
        var total: usize = 0;
        var iterator = self.workspace_index.entries.valueIterator();
        while (iterator.next()) |entry| {
            total = addSat(total, entry.duplicate_string_bytes_saved);
        }
        return total;
    }

    fn incomingCallTargetNameCount(self: *DocumentStore) usize {
        return self.incoming_call_targets.count();
    }

    fn incomingCallTargetUriCount(self: *DocumentStore) usize {
        var total: usize = 0;
        var iterator = self.incoming_call_targets.valueIterator();
        while (iterator.next()) |uris| {
            total = addSat(total, uris.items.len);
        }
        return total;
    }

    fn incomingCallTargetIndexBytes(self: *DocumentStore) usize {
        var total: usize = 0;
        var iterator = self.incoming_call_targets.iterator();
        while (iterator.next()) |entry| {
            total = addSat(total, entry.key_ptr.*.len);
            total = addSat(total, mulSat(entry.value_ptr.capacity, @sizeOf([]u8)));
            for (entry.value_ptr.items) |uri| {
                total = addSat(total, uri.len);
            }
        }
        return total;
    }
};

const OpenDocumentRangeConverter = struct {
    arena: Allocator,
    docs: *DocumentStore,
    encoding: text_edits.PositionEncoding,

    pub fn byteRangeToLsp(self: *OpenDocumentRangeConverter, uri: []const u8, range: frontend.Range) !types.Range {
        const source = self.docs.sourceForUri(uri) orelse return protocol_ranges.rawRange(range);
        const line_index = (try self.docs.lineIndexForUri(uri)) orelse return protocol_ranges.rawRange(range);
        return protocol_ranges.byteRangeToLspOrRaw(source, line_index, self.encoding, range);
    }

    pub fn rangesToLsp(self: *OpenDocumentRangeConverter, uri: []const u8, ranges: []const frontend.Range) ![]types.Range {
        const result = try self.arena.alloc(types.Range, ranges.len);
        for (ranges, 0..) |range, index| {
            result[index] = try self.byteRangeToLsp(uri, range);
        }
        return result;
    }
};

const CrossFileDefinitionRangeConverter = struct {
    arena: Allocator,
    docs: *DocumentStore,
    encoding: text_edits.PositionEncoding,
    bindings: []const definition_api.ImportBinding,

    pub fn byteRangeToLsp(self: *CrossFileDefinitionRangeConverter, uri: []const u8, range: frontend.Range) !types.Range {
        if (self.docs.sourceForUri(uri)) |source| {
            const line_index = (try self.docs.lineIndexForUri(uri)) orelse return protocol_ranges.rawRange(range);
            return protocol_ranges.byteRangeToLspOrRaw(source, line_index, self.encoding, range);
        }

        for (self.bindings) |binding| {
            if (!std.mem.eql(u8, binding.target_uri, uri)) continue;
            const target_source = binding.target_source orelse return protocol_ranges.rawRange(range);
            var line_index = try line_index_api.LineIndex.init(self.arena, target_source);
            return protocol_ranges.byteRangeToLspOrRaw(target_source, &line_index, self.encoding, range);
        }

        return protocol_ranges.rawRange(range);
    }
};

const ImportedMemberAccess = struct {
    alias: []const u8,
    member_name: []const u8,
    member_range: frontend.Range,
};

fn bumpGeneration(generation: u64) u64 {
    return if (generation == std.math.maxInt(u64)) generation else generation + 1;
}

fn moduleNameForPath(path: []const u8) []const u8 {
    const stem = std.fs.path.stem(path);
    return if (stem.len == 0) path else stem;
}

pub const Handler = struct {
    allocator: Allocator,
    transport: *lsp.Transport,
    docs: DocumentStore,
    workspace_roots: std.ArrayList([]const u8),
    workspace_discovery_cache: std.StringHashMap(WorkspaceDiscoveryCacheEntry),
    dependencies: dependency_graph.Graph,
    position_encoding: text_edits.PositionEncoding,
    response_builder_items_built: usize = 0,
    response_builder_capacity_bytes: usize = 0,
    response_location_capacity_bytes: usize = 0,
    response_text_edit_capacity_bytes: usize = 0,
    response_completion_item_capacity_bytes: usize = 0,
    response_semantic_token_data_capacity_bytes: usize = 0,
    response_workspace_symbol_capacity_bytes: usize = 0,
    response_call_hierarchy_capacity_bytes: usize = 0,
    response_hover_capacity_bytes: usize = 0,
    response_definition_capacity_bytes: usize = 0,
    response_document_symbol_capacity_bytes: usize = 0,
    response_document_highlight_capacity_bytes: usize = 0,
    response_inlay_hint_capacity_bytes: usize = 0,
    response_code_lens_capacity_bytes: usize = 0,
    response_formatting_edit_capacity_bytes: usize = 0,
    response_selection_range_capacity_bytes: usize = 0,
    response_folding_range_capacity_bytes: usize = 0,
    response_document_link_capacity_bytes: usize = 0,
    response_code_action_capacity_bytes: usize = 0,
    response_signature_help_capacity_bytes: usize = 0,
    response_prepare_rename_capacity_bytes: usize = 0,
    response_string_bytes: usize = 0,
    response_markdown_bytes: usize = 0,
    response_location_string_bytes: usize = 0,
    response_text_edit_string_bytes: usize = 0,
    response_completion_string_bytes: usize = 0,
    response_completion_markdown_bytes: usize = 0,
    response_hover_string_bytes: usize = 0,
    response_hover_markdown_bytes: usize = 0,
    response_definition_string_bytes: usize = 0,
    response_signature_string_bytes: usize = 0,
    response_signature_markdown_bytes: usize = 0,
    response_document_symbol_string_bytes: usize = 0,
    response_workspace_symbol_string_bytes: usize = 0,
    response_call_hierarchy_string_bytes: usize = 0,
    response_inlay_hint_string_bytes: usize = 0,
    response_code_lens_string_bytes: usize = 0,
    response_formatting_edit_string_bytes: usize = 0,
    response_document_link_string_bytes: usize = 0,
    response_code_action_string_bytes: usize = 0,
    response_prepare_rename_string_bytes: usize = 0,
    stale_document_change_skips: usize = 0,
    occurrence_index_builds: usize = 0,
    imported_member_index_builds: usize = 0,
    phase_counters: phase_stats.Stats = .{},
    allocator_tracker: ?*allocation_stats.CountingAllocator = null,
    workspace_discovery_runs: usize = 0,
    workspace_discovery_files_seen: usize = 0,
    workspace_discovery_files_enqueued: usize = 0,
    workspace_discovery_skipped: usize = 0,
    workspace_discovery_limit_hits: usize = 0,
    workspace_discovery_cache_hits: usize = 0,
    workspace_discovery_cache_rebuilds: usize = 0,
    workspace_discovery_max_files: usize = max_workspace_discovery_files,

    fn init(allocator: Allocator, transport: *lsp.Transport, allocator_tracker: ?*allocation_stats.CountingAllocator) Handler {
        return .{
            .allocator = allocator,
            .transport = transport,
            .docs = DocumentStore.init(allocator, allocator_tracker),
            .workspace_roots = .{},
            .workspace_discovery_cache = std.StringHashMap(WorkspaceDiscoveryCacheEntry).init(allocator),
            .dependencies = dependency_graph.Graph.init(allocator),
            .position_encoding = .utf16,
            .allocator_tracker = allocator_tracker,
        };
    }

    fn allocationScope(self: *Handler, scope: allocation_stats.Scope) OptionalAllocationScope {
        return beginAllocationScope(self.allocator_tracker, scope);
    }

    fn allocatorStats(self: *Handler) ?*const allocation_stats.Stats {
        return if (self.allocator_tracker) |tracker| &tracker.stats else null;
    }

    fn scopedAllocatorStats(self: *Handler, scope: allocation_stats.Scope) allocation_stats.ScopedStats {
        return if (self.allocator_tracker) |tracker| tracker.stats.scope(scope) else .{};
    }

    fn deinit(self: *Handler) void {
        self.docs.deinit();
        for (self.workspace_roots.items) |root| {
            self.allocator.free(root);
        }
        self.workspace_roots.deinit(self.allocator);
        self.clearWorkspaceDiscoveryCache();
        self.workspace_discovery_cache.deinit();
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
            .executeCommandProvider = .{ .commands = &[_][]const u8{"ora.cacheStats"} },
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
        try self.docs.put(uri, text, notification.textDocument.version);
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

        if (self.docs.isStaleOpenVersion(uri, notification.textDocument.version)) {
            self.stale_document_change_skips = addSat(self.stale_document_change_skips, 1);
            return;
        }

        const current = self.docs.sourceForUri(uri) orelse return;
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
                    try self.docs.put(uri, full_text, notification.textDocument.version);
                    try self.updateDocumentDependencies(uri, full_text);
                    try self.publishDiagnostics(arena, uri, full_text);
                }
                return;
            },
            else => return err,
        };
        defer self.allocator.free(updated);

        try self.docs.put(uri, updated, notification.textDocument.version);
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
        const source = self.docs.sourceForUri(uri) orelse return;
        try self.updateDocumentDependencies(uri, source);
        try self.publishDiagnostics(arena, uri, source);

        if (self.dependencies.getPathForUri(uri)) |changed_path| {
            try self.publishDependentsDiagnostics(arena, changed_path, uri);
        }
    }

    pub fn @"textDocument/documentSymbol"(self: *Handler, arena: Allocator, params: types.DocumentSymbolParams) !lsp.ResultType("textDocument/documentSymbol") {
        const uri = params.textDocument.uri;
        const source = self.docs.sourceForUri(uri) orelse return null;
        const line_index = (try self.docs.lineIndexForUri(uri)) orelse return null;
        const index = (try self.docs.semanticIndexForUri(uri, &self.phase_counters)) orelse return null;

        var response_scope = self.allocationScope(.response);
        defer response_scope.deinit();

        const result = try document_symbol_response.build(
            arena,
            source,
            line_index,
            self.position_encoding,
            index,
        );

        self.recordResponseItems(.document_symbol, types.DocumentSymbol, index.symbols.len);
        self.recordResponseStringBytes(.document_symbol, semanticSymbolStringBytes(index.symbols));
        return .{ .array_of_DocumentSymbol = result };
    }

    pub fn @"textDocument/hover"(self: *Handler, arena: Allocator, params: types.HoverParams) !?types.Hover {
        const uri = params.textDocument.uri;
        const source = self.docs.sourceForUri(uri) orelse return null;
        const line_index = (try self.docs.lineIndexForUri(uri)) orelse return null;
        const position: frontend.Position = .{
            .line = params.position.line,
            .character = params.position.character,
        };
        const index = (try self.docs.semanticIndexForUri(uri, &self.phase_counters)) orelse return null;

        var temp_scope = self.allocationScope(.temp_analysis);
        defer temp_scope.deinit();
        var maybe_hover = try hover_api.hoverAtIndex(self.allocator, source, position, index);
        if (maybe_hover == null) return null;
        defer maybe_hover.?.deinit(self.allocator);

        const hover = maybe_hover.?;
        var response_scope = self.allocationScope(.response);
        defer response_scope.deinit();
        const arena_hover: hover_api.Hover = .{
            .contents = try arena.dupe(u8, hover.contents),
            .range = hover.range,
        };
        const response = hover_response.build(source, line_index, self.position_encoding, arena_hover) orelse return null;
        self.recordResponseItems(.hover, types.Hover, 1);
        self.recordResponseMarkdownBytes(.hover, hover.contents.len);
        return response;
    }

    pub fn @"textDocument/definition"(self: *Handler, arena: Allocator, params: types.DefinitionParams) !lsp.ResultType("textDocument/definition") {
        const uri = params.textDocument.uri;
        const source = self.docs.sourceForUri(uri) orelse return null;
        const position: frontend.Position = .{
            .line = params.position.line,
            .character = params.position.character,
        };

        if (try self.cachedSameFileDefinitionFromOccurrences(arena, uri, source, params.position)) |location| {
            return .{ .Definition = .{ .Location = location } };
        }
        if (try self.cachedImportedMemberDefinition(arena, uri, source, params.position)) |location| {
            return .{ .Definition = .{ .Location = location } };
        }

        var temp_scope = self.allocationScope(.temp_analysis);
        defer temp_scope.deinit();
        const cross_file = try self.buildCrossFileContext(uri);
        defer self.allocator.free(cross_file.bindings);
        defer for (cross_file.bindings) |b| {
            self.allocator.free(b.alias);
            self.allocator.free(b.target_uri);
            if (b.target_source) |s| self.allocator.free(s);
        };

        const maybe_definition = if (try self.docs.definitionAnalysisForUri(uri, source, &self.phase_counters)) |analysis_value| blk: {
            var analysis = analysis_value;
            defer analysis.deinit();
            break :blk try definition_api.definitionAtCachedCrossFile(&analysis, source, position, cross_file);
        } else try definition_api.definitionAtCrossFile(self.allocator, source, position, cross_file);
        if (maybe_definition == null) return null;

        const definition = maybe_definition.?;
        const response_uri = definition.uri orelse uri;
        var range_converter = CrossFileDefinitionRangeConverter{
            .arena = arena,
            .docs = &self.docs,
            .encoding = self.position_encoding,
            .bindings = cross_file.bindings,
        };
        var response_scope = self.allocationScope(.response);
        defer response_scope.deinit();
        const location = try definition_response.build(arena, uri, definition, &range_converter);
        self.recordResponseItems(.definition, types.Location, 1);
        self.recordResponseStringBytes(.definition, response_uri.len);
        return .{ .Definition = .{ .Location = location } };
    }

    fn cachedSameFileDefinitionFromOccurrences(
        self: *Handler,
        arena: Allocator,
        uri: []const u8,
        source: []const u8,
        lsp_position: types.Position,
    ) !?types.Location {
        const occurrence_index = self.docs.occurrenceIndexForUri(uri) orelse return null;
        const line_index = (try self.docs.lineIndexForUri(uri)) orelse return null;
        const position = protocol_ranges.lspPositionToBytePosition(
            source,
            line_index,
            self.position_encoding,
            lsp_position,
        ) orelse frontend.Position{
            .line = lsp_position.line,
            .character = lsp_position.character,
        };
        const target = occurrence_index.occurrenceAt(position) orelse return null;

        if (occurrenceIsMemberAccess(source, line_index, target.range)) return null;
        if (definitionLineLooksLikeImportAlias(source, line_index, target.definition_range)) return null;

        var range_converter = OpenDocumentRangeConverter{
            .arena = arena,
            .docs = &self.docs,
            .encoding = self.position_encoding,
        };
        var response_scope = self.allocationScope(.response);
        defer response_scope.deinit();
        const location = try definition_response.build(
            arena,
            uri,
            .{ .range = target.definition_range },
            &range_converter,
        );
        self.recordResponseItems(.definition, types.Location, 1);
        self.recordResponseStringBytes(.definition, uri.len);
        return location;
    }

    fn cachedImportedMemberDefinition(
        self: *Handler,
        arena: Allocator,
        uri: []const u8,
        source: []const u8,
        lsp_position: types.Position,
    ) !?types.Location {
        const line_index = (try self.docs.lineIndexForUri(uri)) orelse return null;
        const access = importedMemberAccessAtLspPosition(source, line_index, self.position_encoding, lsp_position) orelse return null;
        const imported_index = self.docs.importedMemberIndexForUri(uri) orelse return null;

        for (imported_index.occurrences) |occurrence| {
            if (!std.mem.eql(u8, occurrence.alias, access.alias)) continue;
            if (!std.mem.eql(u8, occurrence.member_name, access.member_name)) continue;
            if (!frontendRangesEqual(occurrence.range, access.member_range)) continue;

            const target_uri = try workspace.pathToFileUri(arena, occurrence.imported_path);
            try self.ensureColdDocumentForPath(target_uri, occurrence.imported_path);

            const target_source = self.docs.sourceForUri(target_uri) orelse return null;
            const target_entry = (try self.docs.workspaceEntryForUri(target_uri, .symbols_only, &self.phase_counters)) orelse return null;
            const symbol = target_entry.rootSymbolNamed(occurrence.member_name) orelse return null;
            const range = protocol_ranges.byteRangeToLspOrRaw(
                target_source,
                &target_entry.line_index,
                self.position_encoding,
                symbol.selection_range,
            );

            var response_scope = self.allocationScope(.response);
            defer response_scope.deinit();
            const location: types.Location = .{
                .uri = target_uri,
                .range = range,
            };
            self.recordResponseItems(.definition, types.Location, 1);
            self.recordResponseStringBytes(.definition, target_uri.len);
            return location;
        }

        return null;
    }

    pub fn @"textDocument/references"(self: *Handler, arena: Allocator, params: types.ReferenceParams) !lsp.ResultType("textDocument/references") {
        const uri = params.textDocument.uri;
        const position: frontend.Position = .{
            .line = params.position.line,
            .character = params.position.character,
        };

        const include_declaration = if (params.context.includeDeclaration) true else false;
        const workspace_entry = (try self.docs.workspaceEntryForUri(uri, .{ .occurrences = true }, &self.phase_counters)) orelse return null;
        const occurrence_index = workspace_entry.occurrenceIndex();
        const target = occurrence_index.occurrenceAt(position) orelse return null;

        var response_scope = self.allocationScope(.response);
        defer response_scope.deinit();
        var all_locations = std.ArrayList(types.Location){};
        errdefer all_locations.deinit(arena);

        const same_file_refs = try references_api.referencesAtOccurrenceIndex(
            arena,
            &occurrence_index,
            target.name,
            target.definition_range,
            include_declaration,
        );
        var range_converter = OpenDocumentRangeConverter{
            .arena = arena,
            .docs = &self.docs,
            .encoding = self.position_encoding,
        };
        try references_response.appendLocations(arena, &all_locations, uri, same_file_refs, &range_converter);
        try self.appendCachedImportedReferenceLocations(arena, &all_locations, uri, target.name, &range_converter);

        const result = try all_locations.toOwnedSlice(arena);
        self.recordResponseItems(.location, types.Location, result.len);
        self.recordResponseStringBytes(.location, locationUriBytes(result));
        return result;
    }

    pub fn @"textDocument/completion"(self: *Handler, arena: Allocator, params: types.CompletionParams) !lsp.ResultType("textDocument/completion") {
        const uri = params.textDocument.uri;
        const source = self.docs.sourceForUri(uri) orelse return null;
        const position: frontend.Position = .{
            .line = params.position.line,
            .character = params.position.character,
        };
        const index = (try self.docs.semanticIndexForUri(uri, &self.phase_counters)) orelse return null;

        const trigger_char: ?[]const u8 = if (params.context) |ctx| ctx.triggerCharacter else null;
        var temp_scope = self.allocationScope(.temp_analysis);
        defer temp_scope.deinit();
        const completion_items = try completion_api.completionAtIndex(self.allocator, source, position, trigger_char, index);
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
        var response_scope = self.allocationScope(.response);
        defer response_scope.deinit();
        const result = try arena.alloc(types.CompletionItem, completion_items.len + snippet_count);
        var string_bytes: usize = 0;
        var markdown_bytes: usize = 0;
        for (completion_items, 0..) |item, i| {
            string_bytes = addSat(string_bytes, item.label.len);
            if (item.detail) |detail| string_bytes = addSat(string_bytes, detail.len);
            if (item.documentation) |doc| markdown_bytes = addSat(markdown_bytes, doc.len);
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
            string_bytes = addSat(string_bytes, snip.label.len);
            string_bytes = addSat(string_bytes, snip.detail.len);
            string_bytes = addSat(string_bytes, snip.body.len);
            result[i] = .{
                .label = snip.label,
                .kind = .Snippet,
                .detail = snip.detail,
                .insertTextFormat = .Snippet,
                .insertText = snip.body,
            };
        }

        self.recordResponseItems(.completion_item, types.CompletionItem, result.len);
        self.recordResponseStringBytes(.completion_item, string_bytes);
        self.recordResponseMarkdownBytes(.completion_item, markdown_bytes);
        return .{ .array_of_CompletionItem = result };
    }

    pub fn @"textDocument/prepareRename"(self: *Handler, arena: Allocator, params: types.PrepareRenameParams) !lsp.ResultType("textDocument/prepareRename") {
        const uri = params.textDocument.uri;
        const position: frontend.Position = .{
            .line = params.position.line,
            .character = params.position.character,
        };

        const source = self.docs.sourceForUri(uri) orelse return null;
        const workspace_entry = (try self.docs.workspaceEntryForUri(uri, .{ .occurrences = true }, &self.phase_counters)) orelse return null;
        const occurrence_index = workspace_entry.occurrenceIndex();
        const target = occurrence_index.occurrenceAt(position) orelse return null;

        var response_scope = self.allocationScope(.response);
        defer response_scope.deinit();
        const result = try rename_response.buildPrepare(
            arena,
            source,
            &workspace_entry.line_index,
            self.position_encoding,
            target.definition_range,
            target.name,
        );
        if (result == null) return null;
        self.recordResponseItems(.prepare_rename, types.Range, 1);
        self.recordResponseStringBytes(.prepare_rename, target.name.len);
        return result;
    }

    pub fn @"textDocument/rename"(self: *Handler, arena: Allocator, params: types.RenameParams) !lsp.ResultType("textDocument/rename") {
        if (!rename_api.isValidIdentifier(params.newName)) return null;

        const uri = params.textDocument.uri;
        const position: frontend.Position = .{
            .line = params.position.line,
            .character = params.position.character,
        };

        const source = self.docs.sourceForUri(uri) orelse return null;
        const workspace_entry = (try self.docs.workspaceEntryForUri(uri, .{ .occurrences = true }, &self.phase_counters)) orelse return null;
        const occurrence_index = workspace_entry.occurrenceIndex();
        const target = occurrence_index.occurrenceAt(position) orelse return null;
        var response_scope = self.allocationScope(.response);
        defer response_scope.deinit();
        const same_file_ranges = try references_api.referencesAtOccurrenceIndex(
            arena,
            &occurrence_index,
            target.name,
            target.definition_range,
            true,
        );
        if (same_file_ranges.len == 0) return null;

        var changes: lsp.parser.Map(types.DocumentUri, []const types.TextEdit) = .{};
        _ = try rename_response.putChange(
            arena,
            &changes,
            uri,
            .{ .source = source, .line_index = &workspace_entry.line_index },
            same_file_ranges,
            params.newName,
            self.position_encoding,
        );

        const imported_edit_count = try self.appendCachedImportedRenameChanges(arena, &changes, uri, target.name, params.newName);
        const edit_count = addSat(same_file_ranges.len, imported_edit_count);
        self.recordResponseItems(.text_edit, types.TextEdit, edit_count);
        self.recordResponseStringBytes(.text_edit, mulSat(edit_count, params.newName.len));

        return .{ .changes = changes };
    }

    pub fn @"textDocument/inlayHint"(self: *Handler, arena: Allocator, params: types.InlayHintParams) !lsp.ResultType("textDocument/inlayHint") {
        const uri = params.textDocument.uri;
        const source = self.docs.sourceForUri(uri) orelse return null;
        const line_index = (try self.docs.lineIndexForUri(uri)) orelse return null;
        const ast_file = (try self.docs.astFileForUri(uri, &self.phase_counters)) orelse return null;
        const index = (try self.docs.semanticIndexForUri(uri, &self.phase_counters)) orelse return null;
        const range: frontend.Range = .{
            .start = .{ .line = params.range.start.line, .character = params.range.start.character },
            .end = .{ .line = params.range.end.line, .character = params.range.end.character },
        };

        var temp_scope = self.allocationScope(.temp_analysis);
        defer temp_scope.deinit();
        const hints = try inlay_hints_api.hintsInRangeCached(
            self.allocator,
            source,
            range,
            line_index,
            self.position_encoding,
            ast_file,
            index,
        );
        defer inlay_hints_api.deinitHints(self.allocator, hints);

        if (hints.len == 0) return null;

        var response_scope = self.allocationScope(.response);
        defer response_scope.deinit();
        const result = try arena.alloc(types.InlayHint, hints.len);
        var string_bytes: usize = 0;
        for (hints, 0..) |hint, i| {
            string_bytes = addSat(string_bytes, hint.label.len);
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

        self.recordResponseItems(.inlay_hint, types.InlayHint, result.len);
        self.recordResponseStringBytes(.inlay_hint, string_bytes);
        return result;
    }

    pub fn @"textDocument/codeLens"(self: *Handler, arena: Allocator, params: types.CodeLensParams) !lsp.ResultType("textDocument/codeLens") {
        const uri = params.textDocument.uri;
        const source = self.docs.sourceForUri(uri) orelse return null;
        const line_index = (try self.docs.lineIndexForUri(uri)) orelse return null;
        const ast_file = (try self.docs.astFileForUri(uri, &self.phase_counters)) orelse return null;

        var temp_scope = self.allocationScope(.temp_analysis);
        defer temp_scope.deinit();
        const lenses = try code_lens_api.findVerificationLensesInAst(self.allocator, source, ast_file, line_index);
        defer code_lens_api.deinitLenses(self.allocator, lenses);

        var response_scope = self.allocationScope(.response);
        defer response_scope.deinit();
        const result = try code_lens_response.build(arena, source, line_index, self.position_encoding, lenses);
        if (result) |items| {
            self.recordResponseItems(.code_lens, types.CodeLens, items.len);
            var string_bytes: usize = 0;
            for (lenses) |lens| {
                string_bytes = addSat(string_bytes, lens.title.len);
                string_bytes = addSat(string_bytes, "ora.verify".len);
            }
            self.recordResponseStringBytes(.code_lens, string_bytes);
        }
        return result;
    }

    pub fn @"textDocument/signatureHelp"(self: *Handler, arena: Allocator, params: types.SignatureHelpParams) !?types.SignatureHelp {
        const uri = params.textDocument.uri;
        const source = self.docs.sourceForUri(uri) orelse return null;
        const position: frontend.Position = .{
            .line = params.position.line,
            .character = params.position.character,
        };
        const index = (try self.docs.semanticIndexForUri(uri, &self.phase_counters)) orelse return null;

        var temp_scope = self.allocationScope(.temp_analysis);
        defer temp_scope.deinit();
        var sig_val = (signature_help_api.signatureAtIndex(self.allocator, source, position, index) catch return null) orelse return null;
        defer sig_val.deinit(self.allocator);

        var response_scope = self.allocationScope(.response);
        defer response_scope.deinit();
        const result = try signature_help_response.build(arena, sig_val);
        var string_bytes: usize = sig_val.label.len;
        for (sig_val.parameters) |parameter| {
            string_bytes = addSat(string_bytes, parameter.label.len);
        }
        self.recordResponseItems(.signature_help, types.SignatureInformation, 1);
        self.recordResponseItems(.signature_help, types.ParameterInformation, sig_val.parameters.len);
        self.recordResponseStringBytes(.signature_help, string_bytes);
        if (sig_val.documentation) |doc| self.recordResponseMarkdownBytes(.signature_help, doc.len);
        return result;
    }

    pub fn @"textDocument/semanticTokens/full"(self: *Handler, arena: Allocator, params: types.SemanticTokensParams) !lsp.ResultType("textDocument/semanticTokens/full") {
        const tokens = (try self.docs.semanticTokensForUri(params.textDocument.uri, &self.phase_counters)) orelse return null;

        var response_scope = self.allocationScope(.response);
        defer response_scope.deinit();
        const data = try semantic_tokens_api.encodeTokens(arena, tokens);

        self.recordResponseItems(.semantic_token_data, u32, data.len);
        return .{ .data = data };
    }

    pub fn @"textDocument/formatting"(self: *Handler, arena: Allocator, params: types.DocumentFormattingParams) !lsp.ResultType("textDocument/formatting") {
        const uri = params.textDocument.uri;
        const source = self.docs.sourceForUri(uri) orelse return null;
        const options: formatting_api.Options = .{
            .line_width = 100,
            .indent_size = normalizeIndentSize(params.options.tabSize),
        };
        const formatted = (self.docs.formattingCacheForUri(uri, options, &self.phase_counters) catch |err| switch (err) {
            error.ParseError => return null,
            else => return err,
        }) orelse return null;

        var response_scope = self.allocationScope(.response);
        defer response_scope.deinit();
        const edits = try formatting_edits.buildFullDocumentEdit(arena, source, formatted, self.position_encoding);
        self.recordResponseItems(.formatting_edit, types.TextEdit, edits.len);
        if (edits.len != 0) self.recordResponseStringBytes(.formatting_edit, formatted.len);
        return edits;
    }

    pub fn @"textDocument/documentHighlight"(self: *Handler, arena: Allocator, params: types.DocumentHighlightParams) !lsp.ResultType("textDocument/documentHighlight") {
        const uri = params.textDocument.uri;
        const source = self.docs.sourceForUri(uri) orelse return null;
        const workspace_entry = (try self.docs.workspaceEntryForUri(uri, .{ .occurrences = true }, &self.phase_counters)) orelse return null;
        const occurrence_index = workspace_entry.occurrenceIndex();
        const position: frontend.Position = .{
            .line = params.position.line,
            .character = params.position.character,
        };

        var response_scope = self.allocationScope(.response);
        defer response_scope.deinit();
        const result = try document_highlight_response.build(arena, source, &workspace_entry.line_index, self.position_encoding, &occurrence_index, position);
        if (result) |items| {
            self.recordResponseItems(.document_highlight, types.DocumentHighlight, items.len);
        }
        return result;
    }

    pub fn @"textDocument/foldingRange"(self: *Handler, arena: Allocator, params: types.FoldingRangeParams) !lsp.ResultType("textDocument/foldingRange") {
        const uri = params.textDocument.uri;
        const source = self.docs.sourceForUri(uri) orelse return null;
        const line_index = (try self.docs.lineIndexForUri(uri)) orelse return null;
        const ast_file = (try self.docs.astFileForUri(uri, &self.phase_counters)) orelse return null;

        var temp_scope = self.allocationScope(.temp_analysis);
        defer temp_scope.deinit();
        const ranges = try folding_api.foldingRangesInAst(self.allocator, source, ast_file, line_index);
        defer folding_api.deinitRanges(self.allocator, ranges);

        var response_scope = self.allocationScope(.response);
        defer response_scope.deinit();
        const result = try folding_ranges_response.build(arena, ranges);
        if (result) |items| {
            self.recordResponseItems(.folding_range, types.FoldingRange, items.len);
        }
        return result;
    }

    pub fn @"workspace/executeCommand"(self: *Handler, arena: Allocator, params: types.ExecuteCommandParams) !lsp.ResultType("workspace/executeCommand") {
        if (!std.mem.eql(u8, params.command, "ora.cacheStats")) return null;

        const allocator_totals = self.allocatorStats();
        const unscoped_allocator = self.scopedAllocatorStats(.unscoped);
        const request_allocator = self.scopedAllocatorStats(.request_protocol);
        const response_allocator = self.scopedAllocatorStats(.response);
        const cache_build_allocator = self.scopedAllocatorStats(.cache_build);
        const temp_analysis_allocator = self.scopedAllocatorStats(.temp_analysis);
        const snapshot: cache_stats_response.Snapshot = .{
            .open_documents = self.docs.openDocumentCount(),
            .open_source_bytes = self.docs.openSourceBytes(),
            .document_state_bytes = self.docs.documentStateBytes(),
            .line_index_bytes = self.docs.lineIndexBytes(),
            .formatting_cache_entries = self.docs.formattingCacheEntries(),
            .formatting_cache_bytes = self.docs.formattingCacheBytes(),
            .cache_builder_capacity_requested = self.docs.cacheBuilderCapacityRequested(),
            .cache_builder_items_built = self.docs.cacheBuilderItemsBuilt(),
            .cache_builder_unused_capacity = self.docs.cacheBuilderUnusedCapacity(),
            .cache_builder_growth_events = self.docs.cacheBuilderGrowthEvents(),
            .cache_side_map_capacity_requested = self.docs.cacheSideMapCapacityRequested(),
            .cache_side_map_items_built = self.docs.cacheSideMapItemsBuilt(),
            .cache_side_map_unused_capacity = self.docs.cacheSideMapUnusedCapacity(),
            .cache_side_map_growth_events = self.docs.cacheSideMapGrowthEvents(),
            .response_builder_items_built = self.response_builder_items_built,
            .response_builder_capacity_bytes = self.response_builder_capacity_bytes,
            .response_builder_item_bytes = self.response_builder_capacity_bytes,
            .response_location_capacity_bytes = self.response_location_capacity_bytes,
            .response_text_edit_capacity_bytes = self.response_text_edit_capacity_bytes,
            .response_completion_item_capacity_bytes = self.response_completion_item_capacity_bytes,
            .response_semantic_token_data_capacity_bytes = self.response_semantic_token_data_capacity_bytes,
            .response_workspace_symbol_capacity_bytes = self.response_workspace_symbol_capacity_bytes,
            .response_call_hierarchy_capacity_bytes = self.response_call_hierarchy_capacity_bytes,
            .response_hover_capacity_bytes = self.response_hover_capacity_bytes,
            .response_definition_capacity_bytes = self.response_definition_capacity_bytes,
            .response_document_symbol_capacity_bytes = self.response_document_symbol_capacity_bytes,
            .response_document_highlight_capacity_bytes = self.response_document_highlight_capacity_bytes,
            .response_inlay_hint_capacity_bytes = self.response_inlay_hint_capacity_bytes,
            .response_code_lens_capacity_bytes = self.response_code_lens_capacity_bytes,
            .response_formatting_edit_capacity_bytes = self.response_formatting_edit_capacity_bytes,
            .response_selection_range_capacity_bytes = self.response_selection_range_capacity_bytes,
            .response_folding_range_capacity_bytes = self.response_folding_range_capacity_bytes,
            .response_document_link_capacity_bytes = self.response_document_link_capacity_bytes,
            .response_code_action_capacity_bytes = self.response_code_action_capacity_bytes,
            .response_signature_help_capacity_bytes = self.response_signature_help_capacity_bytes,
            .response_prepare_rename_capacity_bytes = self.response_prepare_rename_capacity_bytes,
            .response_string_bytes = self.response_string_bytes,
            .response_markdown_bytes = self.response_markdown_bytes,
            .response_location_string_bytes = self.response_location_string_bytes,
            .response_text_edit_string_bytes = self.response_text_edit_string_bytes,
            .response_completion_string_bytes = self.response_completion_string_bytes,
            .response_completion_markdown_bytes = self.response_completion_markdown_bytes,
            .response_hover_string_bytes = self.response_hover_string_bytes,
            .response_hover_markdown_bytes = self.response_hover_markdown_bytes,
            .response_definition_string_bytes = self.response_definition_string_bytes,
            .response_signature_string_bytes = self.response_signature_string_bytes,
            .response_signature_markdown_bytes = self.response_signature_markdown_bytes,
            .response_document_symbol_string_bytes = self.response_document_symbol_string_bytes,
            .response_workspace_symbol_string_bytes = self.response_workspace_symbol_string_bytes,
            .response_call_hierarchy_string_bytes = self.response_call_hierarchy_string_bytes,
            .response_inlay_hint_string_bytes = self.response_inlay_hint_string_bytes,
            .response_code_lens_string_bytes = self.response_code_lens_string_bytes,
            .response_formatting_edit_string_bytes = self.response_formatting_edit_string_bytes,
            .response_document_link_string_bytes = self.response_document_link_string_bytes,
            .response_code_action_string_bytes = self.response_code_action_string_bytes,
            .response_prepare_rename_string_bytes = self.response_prepare_rename_string_bytes,
            .stale_document_change_skips = self.stale_document_change_skips,
            .workspace_index_builds = self.docs.workspace_index_builds,
            .cold_workspace_index_builds = self.docs.cold_workspace_index_builds,
            .incoming_call_target_index_builds = self.docs.incoming_call_target_index_builds,
            .workspace_discovery_runs = self.workspace_discovery_runs,
            .workspace_discovery_files_seen = self.workspace_discovery_files_seen,
            .workspace_discovery_files_enqueued = self.workspace_discovery_files_enqueued,
            .workspace_discovery_skipped = self.workspace_discovery_skipped,
            .workspace_discovery_limit_hits = self.workspace_discovery_limit_hits,
            .workspace_discovery_cache_hits = self.workspace_discovery_cache_hits,
            .workspace_discovery_cache_rebuilds = self.workspace_discovery_cache_rebuilds,
            .workspace_discovery_max_files = self.workspace_discovery_max_files,
            .diagnostic_cache_builds = self.docs.diagnostic_cache_builds,
            .workspace_index_entries = self.docs.workspaceIndexEntries(),
            .workspace_index_bytes = self.docs.workspaceIndexBytes(),
            .workspace_index_max_bytes = self.docs.workspaceIndexMaxBytes(),
            .workspace_index_evictions = self.docs.workspaceIndexEvictions(),
            .workspace_index_symbol_count = self.docs.workspaceIndexSymbolCount(),
            .workspace_index_root_symbol_count = self.docs.workspaceIndexRootSymbolCount(),
            .workspace_index_callable_symbol_count = self.docs.workspaceIndexCallableSymbolCount(),
            .workspace_index_import_count = self.docs.workspaceIndexImportCount(),
            .workspace_index_occurrence_count = self.docs.workspaceIndexOccurrenceCount(),
            .workspace_index_imported_member_count = self.docs.workspaceIndexImportedMemberCount(),
            .workspace_index_call_edge_count = self.docs.workspaceIndexCallEdgeCount(),
            .incoming_call_target_name_count = self.docs.incomingCallTargetNameCount(),
            .incoming_call_target_uri_count = self.docs.incomingCallTargetUriCount(),
            .workspace_index_interned_string_bytes = self.docs.workspaceIndexInternedStringBytes(),
            .workspace_index_interned_string_count = self.docs.workspaceIndexInternedStringCount(),
            .workspace_index_duplicate_string_bytes_saved = self.docs.workspaceIndexDuplicateStringBytesSaved(),
            .workspace_index_interned_string_capacity_requested = self.docs.workspaceIndexInternedStringCapacityRequested(),
            .workspace_index_interned_string_items_built = self.docs.workspaceIndexInternedStringItemsBuilt(),
            .workspace_index_interned_string_unused_capacity = self.docs.workspaceIndexInternedStringUnusedCapacity(),
            .workspace_index_interned_string_growth_events = self.docs.workspaceIndexInternedStringGrowthEvents(),
            .cold_workspace_index_entries = self.docs.coldWorkspaceIndexEntries(),
            .cold_workspace_index_bytes = self.docs.coldWorkspaceIndexBytes(),
            .cold_workspace_index_interned_string_bytes = self.docs.coldWorkspaceIndexInternedStringBytes(),
            .cold_workspace_index_interned_string_count = self.docs.coldWorkspaceIndexInternedStringCount(),
            .cold_workspace_index_duplicate_string_bytes_saved = self.docs.coldWorkspaceIndexDuplicateStringBytesSaved(),
            .cold_workspace_index_interned_string_capacity_requested = self.docs.coldWorkspaceIndexInternedStringCapacityRequested(),
            .cold_workspace_index_interned_string_items_built = self.docs.coldWorkspaceIndexInternedStringItemsBuilt(),
            .cold_workspace_index_interned_string_unused_capacity = self.docs.coldWorkspaceIndexInternedStringUnusedCapacity(),
            .cold_workspace_index_interned_string_growth_events = self.docs.coldWorkspaceIndexInternedStringGrowthEvents(),
            .occurrence_index_builds = self.occurrence_index_builds,
            .open_occurrence_count = self.docs.occurrenceCount(),
            .occurrence_index_bytes = self.docs.occurrenceIndexBytes(),
            .imported_member_index_builds = self.imported_member_index_builds,
            .open_imported_member_count = self.docs.importedMemberCount(),
            .imported_member_index_bytes = self.docs.importedMemberIndexBytes(),
            .semantic_index_builds = self.docs.semantic_index_builds,
            .open_symbol_count = self.docs.symbolCount(),
            .semantic_index_bytes = self.docs.semanticIndexBytes(),
            .call_edge_index_builds = self.docs.call_edge_index_builds,
            .open_call_edge_count = self.docs.callEdgeCount(),
            .call_edge_index_bytes = self.docs.callEdgeIndexBytes(),
            .incoming_call_target_index_bytes = self.docs.incomingCallTargetIndexBytes(),
            .cold_documents = self.docs.coldDocumentCount(),
            .cold_source_bytes = self.docs.coldSourceBytes(),
            .open_token_count = self.docs.tokenCount(),
            .open_token_diagnostic_count = self.docs.tokenDiagnosticCount(),
            .token_cache_bytes = self.docs.tokenCacheBytes(),
            .import_resolution_bytes = self.docs.importResolutionBytes(),
            .diagnostic_cache_bytes = self.docs.diagnosticCacheBytes(),
            .lex_builds = self.phase_counters.lex_builds,
            .parse_builds = self.phase_counters.parse_builds,
            .ast_lower_builds = self.phase_counters.ast_lower_builds,
            .item_index_builds = self.phase_counters.item_index_builds,
            .resolve_builds = self.phase_counters.resolve_builds,
            .const_eval_builds = self.phase_counters.const_eval_builds,
            .type_check_builds = self.phase_counters.type_check_builds,
            .formatter_builds = self.phase_counters.formatter_builds,
            .server_allocator_alloc_calls = if (allocator_totals) |stats| stats.alloc_calls else 0,
            .server_allocator_resize_calls = if (allocator_totals) |stats| stats.resize_calls else 0,
            .server_allocator_remap_calls = if (allocator_totals) |stats| stats.remap_calls else 0,
            .server_allocator_free_calls = if (allocator_totals) |stats| stats.free_calls else 0,
            .server_allocator_bytes_allocated = if (allocator_totals) |stats| stats.bytes_allocated else 0,
            .server_allocator_bytes_freed = if (allocator_totals) |stats| stats.bytes_freed else 0,
            .server_allocator_bytes_live = if (allocator_totals) |stats| stats.bytes_live else 0,
            .server_allocator_bytes_peak = if (allocator_totals) |stats| stats.bytes_peak else 0,
            .server_allocator_unscoped_alloc_calls = unscoped_allocator.alloc_calls,
            .server_allocator_unscoped_bytes_allocated = unscoped_allocator.bytes_allocated,
            .server_allocator_request_alloc_calls = request_allocator.alloc_calls,
            .server_allocator_request_bytes_allocated = request_allocator.bytes_allocated,
            .server_allocator_response_alloc_calls = response_allocator.alloc_calls,
            .server_allocator_response_bytes_allocated = response_allocator.bytes_allocated,
            .server_allocator_cache_build_alloc_calls = cache_build_allocator.alloc_calls,
            .server_allocator_cache_build_bytes_allocated = cache_build_allocator.bytes_allocated,
            .server_allocator_temp_analysis_alloc_calls = temp_analysis_allocator.alloc_calls,
            .server_allocator_temp_analysis_bytes_allocated = temp_analysis_allocator.bytes_allocated,
        };
        return .{ .object = try cache_stats_response.build(arena, snapshot) };
    }

    const ResponseKind = enum {
        location,
        text_edit,
        completion_item,
        semantic_token_data,
        workspace_symbol,
        call_hierarchy,
        hover,
        definition,
        document_symbol,
        document_highlight,
        inlay_hint,
        code_lens,
        formatting_edit,
        selection_range,
        folding_range,
        document_link,
        code_action,
        signature_help,
        prepare_rename,
    };

    fn recordResponseItems(self: *Handler, kind: ResponseKind, comptime Item: type, count: usize) void {
        const bytes = mulSat(count, @sizeOf(Item));
        self.response_builder_items_built = addSat(self.response_builder_items_built, count);
        self.response_builder_capacity_bytes = addSat(self.response_builder_capacity_bytes, bytes);
        switch (kind) {
            .location => self.response_location_capacity_bytes = addSat(self.response_location_capacity_bytes, bytes),
            .text_edit => self.response_text_edit_capacity_bytes = addSat(self.response_text_edit_capacity_bytes, bytes),
            .completion_item => self.response_completion_item_capacity_bytes = addSat(self.response_completion_item_capacity_bytes, bytes),
            .semantic_token_data => self.response_semantic_token_data_capacity_bytes = addSat(self.response_semantic_token_data_capacity_bytes, bytes),
            .workspace_symbol => self.response_workspace_symbol_capacity_bytes = addSat(self.response_workspace_symbol_capacity_bytes, bytes),
            .call_hierarchy => self.response_call_hierarchy_capacity_bytes = addSat(self.response_call_hierarchy_capacity_bytes, bytes),
            .hover => self.response_hover_capacity_bytes = addSat(self.response_hover_capacity_bytes, bytes),
            .definition => self.response_definition_capacity_bytes = addSat(self.response_definition_capacity_bytes, bytes),
            .document_symbol => self.response_document_symbol_capacity_bytes = addSat(self.response_document_symbol_capacity_bytes, bytes),
            .document_highlight => self.response_document_highlight_capacity_bytes = addSat(self.response_document_highlight_capacity_bytes, bytes),
            .inlay_hint => self.response_inlay_hint_capacity_bytes = addSat(self.response_inlay_hint_capacity_bytes, bytes),
            .code_lens => self.response_code_lens_capacity_bytes = addSat(self.response_code_lens_capacity_bytes, bytes),
            .formatting_edit => self.response_formatting_edit_capacity_bytes = addSat(self.response_formatting_edit_capacity_bytes, bytes),
            .selection_range => self.response_selection_range_capacity_bytes = addSat(self.response_selection_range_capacity_bytes, bytes),
            .folding_range => self.response_folding_range_capacity_bytes = addSat(self.response_folding_range_capacity_bytes, bytes),
            .document_link => self.response_document_link_capacity_bytes = addSat(self.response_document_link_capacity_bytes, bytes),
            .code_action => self.response_code_action_capacity_bytes = addSat(self.response_code_action_capacity_bytes, bytes),
            .signature_help => self.response_signature_help_capacity_bytes = addSat(self.response_signature_help_capacity_bytes, bytes),
            .prepare_rename => self.response_prepare_rename_capacity_bytes = addSat(self.response_prepare_rename_capacity_bytes, bytes),
        }
    }

    fn recordResponseStringBytes(self: *Handler, kind: ResponseKind, bytes: usize) void {
        if (bytes == 0) return;
        self.response_string_bytes = addSat(self.response_string_bytes, bytes);
        switch (kind) {
            .location => self.response_location_string_bytes = addSat(self.response_location_string_bytes, bytes),
            .text_edit => self.response_text_edit_string_bytes = addSat(self.response_text_edit_string_bytes, bytes),
            .completion_item => self.response_completion_string_bytes = addSat(self.response_completion_string_bytes, bytes),
            .workspace_symbol => self.response_workspace_symbol_string_bytes = addSat(self.response_workspace_symbol_string_bytes, bytes),
            .call_hierarchy => self.response_call_hierarchy_string_bytes = addSat(self.response_call_hierarchy_string_bytes, bytes),
            .hover => self.response_hover_string_bytes = addSat(self.response_hover_string_bytes, bytes),
            .definition => self.response_definition_string_bytes = addSat(self.response_definition_string_bytes, bytes),
            .document_symbol => self.response_document_symbol_string_bytes = addSat(self.response_document_symbol_string_bytes, bytes),
            .inlay_hint => self.response_inlay_hint_string_bytes = addSat(self.response_inlay_hint_string_bytes, bytes),
            .code_lens => self.response_code_lens_string_bytes = addSat(self.response_code_lens_string_bytes, bytes),
            .formatting_edit => self.response_formatting_edit_string_bytes = addSat(self.response_formatting_edit_string_bytes, bytes),
            .document_link => self.response_document_link_string_bytes = addSat(self.response_document_link_string_bytes, bytes),
            .code_action => self.response_code_action_string_bytes = addSat(self.response_code_action_string_bytes, bytes),
            .prepare_rename => self.response_prepare_rename_string_bytes = addSat(self.response_prepare_rename_string_bytes, bytes),
            .signature_help => self.response_signature_string_bytes = addSat(self.response_signature_string_bytes, bytes),
            .semantic_token_data, .document_highlight, .selection_range, .folding_range => {},
        }
    }

    fn recordResponseMarkdownBytes(self: *Handler, kind: ResponseKind, bytes: usize) void {
        if (bytes == 0) return;
        self.recordResponseStringBytes(kind, bytes);
        self.response_markdown_bytes = addSat(self.response_markdown_bytes, bytes);
        switch (kind) {
            .completion_item => self.response_completion_markdown_bytes = addSat(self.response_completion_markdown_bytes, bytes),
            .hover => self.response_hover_markdown_bytes = addSat(self.response_hover_markdown_bytes, bytes),
            .signature_help => self.response_signature_markdown_bytes = addSat(self.response_signature_markdown_bytes, bytes),
            else => {},
        }
    }

    pub fn @"textDocument/codeAction"(self: *Handler, arena: Allocator, params: types.CodeActionParams) !lsp.ResultType("textDocument/codeAction") {
        var response_scope = self.allocationScope(.response);
        defer response_scope.deinit();
        const result = (try code_action_response.build(
            arena,
            params.textDocument.uri,
            params.context.diagnostics,
        )) orelse return null;
        self.recordResponseItems(.code_action, code_action_response.CodeActionOrCommand, result.len);
        self.recordResponseStringBytes(.code_action, codeActionStringBytes(result));
        return result;
    }

    pub fn @"workspace/symbol"(self: *Handler, arena: Allocator, params: types.WorkspaceSymbolParams) !lsp.ResultType("workspace/symbol") {
        const query = params.query;
        var response_scope = self.allocationScope(.response);
        defer response_scope.deinit();
        var symbols = std.ArrayList(types.SymbolInformation){};
        var string_bytes: usize = 0;

        var doc_it = self.docs.docs.iterator();
        while (doc_it.next()) |entry| {
            const doc_uri = entry.key_ptr.*;
            const workspace_entry = (try self.docs.workspaceEntryForUri(doc_uri, .symbols_only, &self.phase_counters)) orelse continue;
            var range_converter = OpenDocumentRangeConverter{
                .arena = arena,
                .docs = &self.docs,
                .encoding = self.position_encoding,
            };

            const stats = try workspace_symbol_response.appendEntrySymbols(
                arena,
                &symbols,
                doc_uri,
                workspace_entry,
                query,
                &range_converter,
            );
            string_bytes = addSat(string_bytes, stats.string_bytes);
        }

        if (symbols.items.len == 0) return null;
        const result = try symbols.toOwnedSlice(arena);
        self.recordResponseItems(.workspace_symbol, types.SymbolInformation, result.len);
        self.recordResponseStringBytes(.workspace_symbol, string_bytes);
        return .{ .array_of_SymbolInformation = result };
    }

    // --- Selection Range ---

    pub fn @"textDocument/selectionRange"(self: *Handler, arena: Allocator, params: types.SelectionRangeParams) !lsp.ResultType("textDocument/selectionRange") {
        const uri = params.textDocument.uri;
        const source = self.docs.sourceForUri(uri) orelse return null;
        const line_index = (try self.docs.lineIndexForUri(uri)) orelse return null;
        const ast_file = (try self.docs.astFileForUri(uri, &self.phase_counters)) orelse return null;

        var response_scope = self.allocationScope(.response);
        defer response_scope.deinit();
        const result = try selection_range_response.build(
            arena,
            source,
            line_index,
            self.position_encoding,
            ast_file,
            params.positions,
        );
        self.recordResponseItems(.selection_range, types.SelectionRange, selectionRangeNodeCount(result));
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
        const source = self.docs.sourceForUri(uri) orelse return null;
        const line_index = (try self.docs.lineIndexForUri(uri)) orelse return null;
        const resolution = (try self.docs.importResolutionForUri(uri, self.workspace_roots.items, &self.phase_counters)) orelse return null;

        var response_scope = self.allocationScope(.response);
        defer response_scope.deinit();
        const result = (try document_link_response.build(
            arena,
            source,
            line_index,
            self.position_encoding,
            resolution.imports,
        )) orelse return null;
        self.recordResponseItems(.document_link, types.DocumentLink, result.len);
        self.recordResponseStringBytes(.document_link, documentLinkStringBytes(result));
        return result;
    }

    // --- Call Hierarchy ---

    pub fn @"textDocument/prepareCallHierarchy"(self: *Handler, arena: Allocator, params: types.CallHierarchyPrepareParams) !lsp.ResultType("textDocument/prepareCallHierarchy") {
        const uri = params.textDocument.uri;
        const source = self.docs.sourceForUri(uri) orelse return null;
        const line_index = (try self.docs.lineIndexForUri(uri)) orelse return null;
        const index = (try self.docs.semanticIndexForUri(uri, &self.phase_counters)) orelse return null;

        var response_scope = self.allocationScope(.response);
        defer response_scope.deinit();
        const result = (try call_hierarchy_prepare_response.build(
            arena,
            uri,
            source,
            line_index,
            self.position_encoding,
            index,
            params.position,
        )) orelse return null;
        self.recordResponseItems(.call_hierarchy, types.CallHierarchyItem, result.len);
        self.recordResponseStringBytes(.call_hierarchy, callHierarchyItemStringBytes(result));
        return result;
    }

    pub fn @"callHierarchy/incomingCalls"(self: *Handler, arena: Allocator, params: types.CallHierarchyIncomingCallsParams) !lsp.ResultType("callHierarchy/incomingCalls") {
        const target_name = params.item.name;
        var response_scope = self.allocationScope(.response);
        defer response_scope.deinit();
        var calls = std.ArrayList(types.CallHierarchyIncomingCall){};
        try self.docs.ensureIncomingCallTargetIndex(&self.phase_counters);
        var range_converter = OpenDocumentRangeConverter{
            .arena = arena,
            .docs = &self.docs,
            .encoding = self.position_encoding,
        };

        if (self.docs.incomingCallTargetUris(target_name)) |candidate_uris| {
            for (candidate_uris) |doc_uri| {
                const workspace_entry = (try self.docs.workspaceEntryForUri(doc_uri, .calls, &self.phase_counters)) orelse continue;
                try call_hierarchy_calls_response.appendIncomingMatches(arena, &calls, doc_uri, workspace_entry, target_name, &range_converter);
            }
        }
        try self.appendColdImportedIncomingCalls(
            arena,
            &calls,
            params.item.uri,
            target_name,
            isImportableCallHierarchyKind(params.item.kind),
            &range_converter,
        );

        if (calls.items.len == 0) return null;
        const result = try calls.toOwnedSlice(arena);
        self.recordResponseItems(.call_hierarchy, types.CallHierarchyIncomingCall, result.len);
        self.recordResponseStringBytes(.call_hierarchy, incomingCallStringBytes(result));
        return result;
    }

    pub fn @"callHierarchy/outgoingCalls"(self: *Handler, arena: Allocator, params: types.CallHierarchyOutgoingCallsParams) !lsp.ResultType("callHierarchy/outgoingCalls") {
        const caller_uri = params.item.uri;
        const caller_source = self.docs.sourceForUri(caller_uri) orelse return null;
        const line_index = (try self.docs.lineIndexForUri(caller_uri)) orelse return null;
        const caller_state = self.docs.documentVersionStateForUri(caller_uri) orelse return null;
        const has_imported_members = self.docs.importedMemberIndexForUri(caller_uri) != null or
            (caller_state.is_cold and try self.ensureImportedMemberIndexForUri(caller_uri));
        const workspace_entry = if (has_imported_members)
            (try self.docs.workspaceEntryForUri(caller_uri, .{ .call_edges = true, .imported_members = true }, &self.phase_counters)) orelse return null
        else
            (try self.docs.workspaceEntryForUri(caller_uri, .calls, &self.phase_counters)) orelse return null;

        const caller_range = protocol_ranges.lspRangeToByte(caller_source, line_index, self.position_encoding, params.item.range) orelse return null;
        var range_converter = OpenDocumentRangeConverter{
            .arena = arena,
            .docs = &self.docs,
            .encoding = self.position_encoding,
        };

        const ImportedOutgoingTargetResolver = struct {
            handler: *Handler,
            arena: Allocator,

            pub fn resolve(
                resolver: *@This(),
                occurrence: references_api.ImportedMemberOccurrence,
            ) !?call_hierarchy_calls_response.ResolvedTarget {
                const target_uri = try workspace.pathToFileUri(resolver.arena, occurrence.imported_path);
                try resolver.handler.ensureColdDocumentForPath(target_uri, occurrence.imported_path);
                const target_entry = (try resolver.handler.docs.workspaceEntryForUri(target_uri, .symbols_only, &resolver.handler.phase_counters)) orelse return null;
                const symbol = target_entry.callableSymbolNamed(occurrence.member_name) orelse return null;
                return .{ .uri = target_uri, .symbol = symbol.* };
            }
        };
        var imported_resolver = ImportedOutgoingTargetResolver{ .handler = self, .arena = arena };
        var response_scope = self.allocationScope(.response);
        defer response_scope.deinit();
        const result = (try call_hierarchy_calls_response.outgoingCallsWithResolver(
            arena,
            caller_uri,
            workspace_entry,
            caller_range,
            &range_converter,
            &imported_resolver,
        )) orelse return null;
        self.recordResponseItems(.call_hierarchy, types.CallHierarchyOutgoingCall, result.len);
        self.recordResponseStringBytes(.call_hierarchy, outgoingCallStringBytes(result));
        return result;
    }

    fn appendColdImportedIncomingCalls(
        self: *Handler,
        arena: Allocator,
        calls: *std.ArrayList(types.CallHierarchyIncomingCall),
        target_uri: []const u8,
        target_name: []const u8,
        allow_discovery: bool,
        range_converter: *OpenDocumentRangeConverter,
    ) !void {
        const borrowed_target_path = self.borrowedNormalizedPathForUri(target_uri);
        const owned_target_path = if (borrowed_target_path == null) try self.normalizedPathForUri(target_uri) else null;
        defer if (owned_target_path) |path| self.allocator.free(path);
        const target_path = borrowed_target_path orelse owned_target_path orelse return;

        const cold_importers = if (self.cachedDiscoveredImportersForTargetPath(target_path)) |cached|
            cached
        else if (allow_discovery)
            try self.discoveredImportersForTargetPath(target_path)
        else
            return;
        for (cold_importers) |importer| {
            if (std.mem.eql(u8, importer.uri, target_uri)) continue;
            if (self.docs.isOpenDocument(importer.uri)) continue;

            const workspace_entry = (try self.coldImportedCallWorkspaceEntry(importer)) orelse continue;
            try call_hierarchy_calls_response.appendIncomingImportedMatches(
                arena,
                calls,
                importer.uri,
                workspace_entry,
                target_path,
                target_name,
                range_converter,
            );
        }
    }

    fn isImportableCallHierarchyKind(kind: types.SymbolKind) bool {
        return kind == .Function or kind == .Method;
    }

    fn appendCachedImportedReferenceLocations(
        self: *Handler,
        arena: Allocator,
        locations: *std.ArrayList(types.Location),
        target_uri: []const u8,
        target_name: []const u8,
        range_converter: *OpenDocumentRangeConverter,
    ) !void {
        const borrowed_target_path = self.borrowedNormalizedPathForUri(target_uri);
        const owned_target_path = if (borrowed_target_path == null) try self.normalizedPathForUri(target_uri) else null;
        defer if (owned_target_path) |path| self.allocator.free(path);
        const target_path = borrowed_target_path orelse owned_target_path orelse return;

        for (self.dependencies.directImporters(target_path)) |other_uri| {
            if (std.mem.eql(u8, other_uri, target_uri)) continue;
            if (!self.docs.isOpenDocument(other_uri)) continue;
            const workspace_entry = (try self.docs.workspaceEntryForUri(other_uri, .{ .symbols = false, .imported_members = true }, &self.phase_counters)) orelse continue;
            const imported_member_index = workspace_entry.importedMemberIndex();
            for (imported_member_index.occurrences) |occurrence| {
                if (!std.mem.eql(u8, occurrence.imported_path, target_path)) continue;
                if (!std.mem.eql(u8, occurrence.member_name, target_name)) continue;
                try locations.append(arena, .{
                    .uri = other_uri,
                    .range = try range_converter.byteRangeToLsp(other_uri, occurrence.range),
                });
            }
        }

        const cold_importers = try self.discoveredImportersForTargetPath(target_path);
        for (cold_importers) |importer| {
            if (std.mem.eql(u8, importer.uri, target_uri)) continue;
            if (self.docs.isOpenDocument(importer.uri)) continue;

            const workspace_entry = (try self.coldImportedMemberWorkspaceEntry(importer)) orelse continue;
            const source = self.docs.sourceForUri(importer.uri) orelse continue;
            const imported_member_index = workspace_entry.importedMemberIndex();
            const indexed_source = uri_ranges.IndexedSource{ .source = source, .line_index = &workspace_entry.line_index };
            for (imported_member_index.occurrences) |occurrence| {
                if (!std.mem.eql(u8, occurrence.imported_path, target_path)) continue;
                if (!std.mem.eql(u8, occurrence.member_name, target_name)) continue;
                try locations.append(arena, .{
                    .uri = importer.uri,
                    .range = uri_ranges.byteRangeToLsp(indexed_source, self.position_encoding, occurrence.range),
                });
            }
        }
    }

    fn appendCachedImportedRenameChanges(
        self: *Handler,
        arena: Allocator,
        changes: *lsp.parser.Map(types.DocumentUri, []const types.TextEdit),
        target_uri: []const u8,
        target_name: []const u8,
        new_name: []const u8,
    ) !usize {
        const borrowed_target_path = self.borrowedNormalizedPathForUri(target_uri);
        const owned_target_path = if (borrowed_target_path == null) try self.normalizedPathForUri(target_uri) else null;
        defer if (owned_target_path) |path| self.allocator.free(path);
        const target_path = borrowed_target_path orelse owned_target_path orelse return 0;

        var total_edits: usize = 0;
        for (self.dependencies.directImporters(target_path)) |other_uri| {
            if (std.mem.eql(u8, other_uri, target_uri)) continue;
            if (!self.docs.isOpenDocument(other_uri)) continue;
            const workspace_entry = (try self.docs.workspaceEntryForUri(other_uri, .{ .symbols = false, .imported_members = true }, &self.phase_counters)) orelse continue;
            const imported_member_index = workspace_entry.importedMemberIndex();

            var match_count: usize = 0;
            for (imported_member_index.occurrences) |occurrence| {
                if (!std.mem.eql(u8, occurrence.imported_path, target_path)) continue;
                if (!std.mem.eql(u8, occurrence.member_name, target_name)) continue;
                match_count += 1;
            }
            if (match_count == 0) continue;

            const match_ranges = try arena.alloc(frontend.Range, match_count);
            var range_index: usize = 0;
            for (imported_member_index.occurrences) |occurrence| {
                if (!std.mem.eql(u8, occurrence.imported_path, target_path)) continue;
                if (!std.mem.eql(u8, occurrence.member_name, target_name)) continue;
                match_ranges[range_index] = occurrence.range;
                range_index += 1;
            }

            const indexed_source: ?uri_ranges.IndexedSource = if (self.docs.sourceForUri(other_uri)) |source|
                .{ .source = source, .line_index = &workspace_entry.line_index }
            else
                null;
            _ = try rename_response.putChange(
                arena,
                changes,
                other_uri,
                indexed_source,
                match_ranges,
                new_name,
                self.position_encoding,
            );
            total_edits = addSat(total_edits, match_count);
        }

        const cold_importers = try self.discoveredImportersForTargetPath(target_path);
        for (cold_importers) |importer| {
            if (std.mem.eql(u8, importer.uri, target_uri)) continue;
            if (self.docs.isOpenDocument(importer.uri)) continue;

            const workspace_entry = (try self.coldImportedMemberWorkspaceEntry(importer)) orelse continue;
            const source = self.docs.sourceForUri(importer.uri) orelse continue;
            const imported_member_index = workspace_entry.importedMemberIndex();

            var match_count: usize = 0;
            for (imported_member_index.occurrences) |occurrence| {
                if (!std.mem.eql(u8, occurrence.imported_path, target_path)) continue;
                if (!std.mem.eql(u8, occurrence.member_name, target_name)) continue;
                match_count += 1;
            }
            if (match_count == 0) continue;

            const match_ranges = try arena.alloc(frontend.Range, match_count);
            var range_index: usize = 0;
            for (imported_member_index.occurrences) |occurrence| {
                if (!std.mem.eql(u8, occurrence.imported_path, target_path)) continue;
                if (!std.mem.eql(u8, occurrence.member_name, target_name)) continue;
                match_ranges[range_index] = occurrence.range;
                range_index += 1;
            }

            _ = try rename_response.putChange(
                arena,
                changes,
                importer.uri,
                .{ .source = source, .line_index = &workspace_entry.line_index },
                match_ranges,
                new_name,
                self.position_encoding,
            );
            total_edits = addSat(total_edits, match_count);
        }
        return total_edits;
    }

    fn normalizedPathForUri(self: *Handler, uri: []const u8) !?[]u8 {
        var scope = self.allocationScope(.temp_analysis);
        defer scope.deinit();

        const maybe_path = try workspace.fileUriToPathAlloc(self.allocator, uri);
        const path = maybe_path orelse return null;
        defer self.allocator.free(path);
        return try workspace.normalizePathAlloc(self.allocator, path);
    }

    fn borrowedNormalizedPathForUri(self: *Handler, uri: []const u8) ?[]const u8 {
        return self.dependencies.getPathForUri(uri);
    }

    fn discoveredImportersForTargetPath(self: *Handler, target_path: []const u8) ![]const DiscoveredImporter {
        if (self.workspace_discovery_cache.get(target_path)) |entry| {
            self.workspace_discovery_cache_hits = addSat(self.workspace_discovery_cache_hits, 1);
            return entry.importers;
        }

        var scope = self.allocationScope(.cache_build);
        defer scope.deinit();

        self.workspace_discovery_runs = addSat(self.workspace_discovery_runs, 1);
        self.workspace_discovery_cache_rebuilds = addSat(self.workspace_discovery_cache_rebuilds, 1);

        var importers = std.ArrayList(DiscoveredImporter){};
        errdefer {
            for (importers.items) |*importer| importer.deinit(self.allocator);
            importers.deinit(self.allocator);
        }

        var reached_limit = false;
        var files_seen_this_run: usize = 0;
        for (self.workspace_roots.items) |root| {
            if (reached_limit) break;

            var root_dir = std.fs.openDirAbsolute(root, .{ .iterate = true }) catch {
                self.workspace_discovery_skipped = addSat(self.workspace_discovery_skipped, 1);
                continue;
            };
            defer root_dir.close();

            var walker = root_dir.walk(self.allocator) catch {
                self.workspace_discovery_skipped = addSat(self.workspace_discovery_skipped, 1);
                continue;
            };
            defer walker.deinit();

            while (try walker.next()) |entry| {
                if (entry.kind != .file) continue;

                files_seen_this_run = addSat(files_seen_this_run, 1);
                self.workspace_discovery_files_seen = addSat(self.workspace_discovery_files_seen, 1);
                if (files_seen_this_run > self.workspace_discovery_max_files) {
                    self.workspace_discovery_limit_hits = addSat(self.workspace_discovery_limit_hits, 1);
                    reached_limit = true;
                    break;
                }

                if (!std.mem.endsWith(u8, entry.path, ".ora")) continue;
                self.workspace_discovery_files_enqueued = addSat(self.workspace_discovery_files_enqueued, 1);

                const joined_path = try std.fs.path.join(self.allocator, &.{ root, entry.path });
                defer self.allocator.free(joined_path);

                const normalized_path = workspace.normalizePathAlloc(self.allocator, joined_path) catch {
                    self.workspace_discovery_skipped = addSat(self.workspace_discovery_skipped, 1);
                    continue;
                };
                defer self.allocator.free(normalized_path);

                if (std.mem.eql(u8, normalized_path, target_path)) continue;

                const importer_uri = workspace.pathToFileUri(self.allocator, normalized_path) catch {
                    self.workspace_discovery_skipped = addSat(self.workspace_discovery_skipped, 1);
                    continue;
                };
                defer self.allocator.free(importer_uri);

                if (self.docs.isOpenDocument(importer_uri)) continue;

                const source = std.fs.cwd().readFileAlloc(self.allocator, normalized_path, max_cold_file_bytes) catch {
                    self.workspace_discovery_skipped = addSat(self.workspace_discovery_skipped, 1);
                    continue;
                };
                defer self.allocator.free(source);

                var resolution = workspace.resolveDocumentImports(
                    self.allocator,
                    importer_uri,
                    source,
                    .{ .workspace_roots = self.workspace_roots.items },
                ) catch {
                    self.workspace_discovery_skipped = addSat(self.workspace_discovery_skipped, 1);
                    continue;
                };
                defer resolution.deinit(self.allocator);

                if (!importsPath(resolution.imports, target_path)) continue;

                try self.docs.putColdDocument(importer_uri, normalized_path, source);
                try self.appendDiscoveredImporter(&importers, importer_uri, normalized_path);
            }
        }

        const key = try self.allocator.dupe(u8, target_path);
        errdefer self.allocator.free(key);

        const owned_importers = try importers.toOwnedSlice(self.allocator);
        errdefer {
            for (owned_importers) |*importer| importer.deinit(self.allocator);
            self.allocator.free(owned_importers);
        }

        try self.workspace_discovery_cache.put(key, .{ .importers = owned_importers });
        return self.workspace_discovery_cache.get(target_path).?.importers;
    }

    fn cachedDiscoveredImportersForTargetPath(self: *Handler, target_path: []const u8) ?[]const DiscoveredImporter {
        if (self.workspace_discovery_cache.get(target_path)) |entry| {
            self.workspace_discovery_cache_hits = addSat(self.workspace_discovery_cache_hits, 1);
            return entry.importers;
        }
        return null;
    }

    fn appendDiscoveredImporter(
        self: *Handler,
        importers: *std.ArrayList(DiscoveredImporter),
        uri: []const u8,
        normalized_path: []const u8,
    ) !void {
        const uri_copy = try self.allocator.dupe(u8, uri);
        errdefer self.allocator.free(uri_copy);

        const path_copy = try self.allocator.dupe(u8, normalized_path);
        errdefer self.allocator.free(path_copy);

        try importers.append(self.allocator, .{
            .uri = uri_copy,
            .normalized_path = path_copy,
        });
    }

    fn importsPath(imports: []const workspace.ResolvedImport, target_path: []const u8) bool {
        for (imports) |import_item| {
            if (std.mem.eql(u8, import_item.resolved_path, target_path)) return true;
        }
        return false;
    }

    fn coldImportedMemberWorkspaceEntry(self: *Handler, importer: DiscoveredImporter) !?*const workspace_index_api.FileEntry {
        const features: workspace_index_api.FeatureSet = .{ .symbols = false, .imported_members = true };
        if (!(try self.ensureColdImportedMemberIndex(importer))) return null;

        return try self.docs.workspaceEntryForUri(importer.uri, features, &self.phase_counters);
    }

    fn coldImportedCallWorkspaceEntry(self: *Handler, importer: DiscoveredImporter) !?*const workspace_index_api.FileEntry {
        const features: workspace_index_api.FeatureSet = .{ .call_edges = true, .imported_members = true };
        if (!(try self.ensureColdImportedMemberIndex(importer))) return null;

        return try self.docs.workspaceEntryForUri(importer.uri, features, &self.phase_counters);
    }

    fn ensureImportedMemberIndexForUri(self: *Handler, uri: []const u8) !bool {
        if (self.docs.importedMemberIndexForUri(uri) != null) return true;

        const resolution = (try self.docs.importResolutionForUri(uri, self.workspace_roots.items, &self.phase_counters)) orelse return false;

        if (resolution.imports.len == 0) return false;
        try self.docs.rebuildImportedMemberIndex(uri, resolution.imports, &self.phase_counters);
        self.imported_member_index_builds = addSat(self.imported_member_index_builds, 1);
        return self.docs.importedMemberIndexForUri(uri) != null;
    }

    fn ensureColdImportedMemberIndex(self: *Handler, importer: DiscoveredImporter) !bool {
        if (self.docs.sourceForUri(importer.uri) == null) {
            var temp_scope = self.allocationScope(.temp_analysis);
            defer temp_scope.deinit();
            const loaded = std.fs.cwd().readFileAlloc(self.allocator, importer.normalized_path, max_cold_file_bytes) catch |err| switch (err) {
                error.OutOfMemory => return error.OutOfMemory,
                else => return false,
            };
            defer self.allocator.free(loaded);
            try self.docs.putColdDocument(importer.uri, importer.normalized_path, loaded);
            if (self.docs.sourceForUri(importer.uri) == null) return false;
        }

        if (self.docs.importedMemberIndexForUri(importer.uri) == null) {
            const resolution = (try self.docs.importResolutionForUri(importer.uri, self.workspace_roots.items, &self.phase_counters)) orelse return false;

            try self.docs.rebuildImportedMemberIndex(importer.uri, resolution.imports, &self.phase_counters);
            self.imported_member_index_builds = addSat(self.imported_member_index_builds, 1);
        }

        return self.docs.importedMemberIndexForUri(importer.uri) != null;
    }

    fn ensureColdDocumentForPath(self: *Handler, uri: []const u8, normalized_path: []const u8) !void {
        if (self.docs.sourceForUri(uri) != null) return;

        var temp_scope = self.allocationScope(.temp_analysis);
        defer temp_scope.deinit();
        const loaded = std.fs.cwd().readFileAlloc(self.allocator, normalized_path, max_cold_file_bytes) catch |err| switch (err) {
            error.OutOfMemory => return error.OutOfMemory,
            else => return,
        };
        defer self.allocator.free(loaded);

        try self.docs.putColdDocument(uri, normalized_path, loaded);
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
        self.docs.clearImportResolutionCache();
        self.clearWorkspaceDiscoveryCache();
    }

    fn clearWorkspaceDiscoveryCache(self: *Handler) void {
        var it = self.workspace_discovery_cache.iterator();
        while (it.next()) |entry| {
            self.allocator.free(entry.key_ptr.*);
            entry.value_ptr.deinit(self.allocator);
        }
        self.workspace_discovery_cache.clearRetainingCapacity();
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

    fn buildCrossFileContext(self: *Handler, uri: []const u8) !definition_api.CrossFileContext {
        const import_resolution = (try self.docs.importResolutionForUri(uri, self.workspace_roots.items, &self.phase_counters)) orelse return .{
            .bindings = try self.allocator.alloc(definition_api.ImportBinding, 0),
        };
        var bindings = std.ArrayList(definition_api.ImportBinding){};
        errdefer {
            for (bindings.items) |b| {
                self.allocator.free(b.alias);
                self.allocator.free(b.target_uri);
                if (b.target_source) |s| self.allocator.free(s);
            }
            bindings.deinit(self.allocator);
        }

        for (import_resolution.imports) |resolved| {
            const alias = resolved.alias orelse aliasFromSpecifier(resolved.specifier) orelse continue;
            const alias_copy = try self.allocator.dupe(u8, alias);
            errdefer self.allocator.free(alias_copy);

            const target_uri = try workspace.pathToFileUri(self.allocator, resolved.resolved_path);
            errdefer self.allocator.free(target_uri);

            const target_source = self.getSourceForPath(target_uri, resolved.resolved_path);
            errdefer if (target_source) |s| self.allocator.free(s);
            try bindings.append(self.allocator, .{
                .alias = alias_copy,
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
        if (self.docs.sourceForUri(target_uri)) |text| {
            return self.allocator.dupe(u8, text) catch return null;
        }
        return std.fs.cwd().readFileAlloc(self.allocator, fs_path, 10 * 1024 * 1024) catch null;
    }

    fn updateDocumentDependencies(self: *Handler, uri: []const u8, source: []const u8) !void {
        var temp_scope = self.allocationScope(.temp_analysis);
        defer temp_scope.deinit();

        const maybe_doc_path = try workspace.fileUriToPathAlloc(self.allocator, uri);
        const normalized_doc_path = if (maybe_doc_path) |doc_path| blk: {
            defer self.allocator.free(doc_path);
            break :blk try workspace.normalizePathAlloc(self.allocator, doc_path);
        } else null;
        defer if (normalized_doc_path) |path| self.allocator.free(path);

        const import_resolution = (try self.docs.importResolutionForUri(uri, self.workspace_roots.items, &self.phase_counters)) orelse return;

        try self.docs.rebuildOccurrenceIndex(uri, source, &self.phase_counters);
        self.occurrence_index_builds = addSat(self.occurrence_index_builds, 1);

        try self.docs.rebuildImportedMemberIndex(uri, import_resolution.imports, &self.phase_counters);
        self.imported_member_index_builds = addSat(self.imported_member_index_builds, 1);

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
            const dependent_source = self.docs.sourceForUri(dependent_uri) orelse continue;
            try self.publishDiagnostics(arena, dependent_uri, dependent_source);
        }
    }

    fn publishDiagnostics(self: *Handler, arena: Allocator, uri: []const u8, source: []const u8) !void {
        const diagnostic_cache = (try self.docs.diagnosticCacheForUri(uri, self.workspace_roots.items, &self.phase_counters)) orelse {
            const diagnostics = try arena.alloc(types.Diagnostic, 0);
            try self.transport.writeNotification(
                self.allocator,
                "textDocument/publishDiagnostics",
                types.PublishDiagnosticsParams,
                .{ .uri = uri, .diagnostics = diagnostics },
                .{ .emit_null_optional_fields = false },
            );
            return;
        };
        const total = diagnostic_cache.diagnostics.len;
        const diagnostics = try arena.alloc(types.Diagnostic, total);
        const line_index = if (total == 0) null else try self.docs.lineIndexForUri(uri);

        for (diagnostic_cache.diagnostics, 0..) |diag, i| {
            diagnostics[i] = .{
                .range = diagnosticRangeToLsp(source, line_index, self.position_encoding, diag.range),
                .severity = frontendSeverityToLsp(diag.severity),
                .source = cachedDiagnosticSourceName(diag.source),
                .message = try arena.dupe(u8, diag.message),
            };
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

fn occurrenceIsMemberAccess(
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

const IdentifierBounds = struct {
    start: usize,
    end: usize,
};

fn importedMemberAccessAtLspPosition(
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

fn frontendRangesEqual(a: frontend.Range, b: frontend.Range) bool {
    return a.start.line == b.start.line and
        a.start.character == b.start.character and
        a.end.line == b.end.line and
        a.end.character == b.end.character;
}

fn isOraIdentifierByte(byte: u8) bool {
    return std.ascii.isAlphanumeric(byte) or byte == '_';
}

fn definitionLineLooksLikeImportAlias(
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

fn diagnosticRangeToLsp(
    source: []const u8,
    line_index: ?*const line_index_api.LineIndex,
    encoding: text_edits.PositionEncoding,
    range: frontend.Range,
) types.Range {
    const index = line_index orelse return protocol_ranges.rawRange(range);
    return protocol_ranges.byteRangeToLspOrRaw(source, index, encoding, range);
}

fn toLspPosition(pos: frontend.Position) types.Position {
    return .{ .line = pos.line, .character = pos.character };
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

fn cachedDiagnosticSourceFromFrontend(source: frontend.DiagnosticSource) CachedDiagnosticSource {
    return switch (source) {
        .lexer => .lexer,
        .parser => .parser,
        .sema => .sema,
    };
}

fn cachedDiagnosticSourceName(source: CachedDiagnosticSource) []const u8 {
    return switch (source) {
        .lexer => "ora-lexer",
        .parser => "ora-parser",
        .sema => "ora-sema",
        .imports => "ora-imports",
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

fn semanticSymbolStringBytes(symbols: []const semantic_index.Symbol) usize {
    var total: usize = 0;
    for (symbols) |symbol| {
        total = addSat(total, symbol.name.len);
        if (symbol.detail) |detail| total = addSat(total, detail.len);
    }
    return total;
}

fn callHierarchyItemStringBytes(items: []const types.CallHierarchyItem) usize {
    var total: usize = 0;
    for (items) |item| {
        total = addSat(total, item.name.len);
        total = addSat(total, item.uri.len);
        if (item.detail) |detail| total = addSat(total, detail.len);
    }
    return total;
}

fn incomingCallStringBytes(items: []const types.CallHierarchyIncomingCall) usize {
    var total: usize = 0;
    for (items) |item| {
        total = addSat(total, item.from.name.len);
        total = addSat(total, item.from.uri.len);
        if (item.from.detail) |detail| total = addSat(total, detail.len);
    }
    return total;
}

fn outgoingCallStringBytes(items: []const types.CallHierarchyOutgoingCall) usize {
    var total: usize = 0;
    for (items) |item| {
        total = addSat(total, item.to.name.len);
        total = addSat(total, item.to.uri.len);
        if (item.to.detail) |detail| total = addSat(total, detail.len);
    }
    return total;
}

fn locationUriBytes(locations: []const types.Location) usize {
    var total: usize = 0;
    for (locations) |location| {
        total = addSat(total, location.uri.len);
    }
    return total;
}

fn documentLinkStringBytes(links: []const types.DocumentLink) usize {
    var total: usize = 0;
    for (links) |link| {
        if (link.target) |target| total = addSat(total, target.len);
        if (link.tooltip) |tooltip| total = addSat(total, tooltip.len);
    }
    return total;
}

fn codeActionStringBytes(actions: []const code_action_response.CodeActionOrCommand) usize {
    var total: usize = 0;
    for (actions) |action_or_command| {
        switch (action_or_command) {
            .CodeAction => |action| {
                total = addSat(total, action.title.len);
                if (action.diagnostics) |diagnostics| {
                    for (diagnostics) |diagnostic| {
                        total = addSat(total, diagnostic.message.len);
                        if (diagnostic.source) |source| total = addSat(total, source.len);
                    }
                }
                if (action.edit) |edit| total = addSat(total, workspaceEditStringBytes(edit));
                if (action.command) |command| total = addSat(total, commandStringBytes(command));
            },
            .Command => |command| total = addSat(total, commandStringBytes(command)),
        }
    }
    return total;
}

fn workspaceEditStringBytes(edit: types.WorkspaceEdit) usize {
    var total: usize = 0;
    if (edit.changes) |changes| {
        var iterator = changes.map.iterator();
        while (iterator.next()) |entry| {
            total = addSat(total, entry.key_ptr.*.len);
            for (entry.value_ptr.*) |text_edit| {
                total = addSat(total, text_edit.newText.len);
            }
        }
    }
    return total;
}

fn commandStringBytes(command: types.Command) usize {
    return addSat(command.title.len, command.command.len);
}

fn selectionRangeNodeCount(ranges: []const types.SelectionRange) usize {
    var total: usize = 0;
    for (ranges) |range| {
        total = addSat(total, selectionRangeNodeCountOne(range));
    }
    return total;
}

fn selectionRangeNodeCountOne(range: types.SelectionRange) usize {
    var total: usize = 1;
    var parent = range.parent;
    while (parent) |next| {
        total = addSat(total, 1);
        parent = next.parent;
    }
    return total;
}

fn addSat(a: usize, b: usize) usize {
    return std.math.add(usize, a, b) catch std.math.maxInt(usize);
}

fn mulSat(a: usize, b: usize) usize {
    return std.math.mul(usize, a, b) catch std.math.maxInt(usize);
}

fn containsString(items: []const []u8, needle: []const u8) bool {
    for (items) |item| {
        if (std.mem.eql(u8, item, needle)) return true;
    }
    return false;
}

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    var counting_allocator = allocation_stats.CountingAllocator.init(gpa.allocator());
    const allocator = counting_allocator.allocator();

    var read_buffer: [256]u8 = undefined;
    var stdio_transport: lsp.Transport.Stdio = .init(&read_buffer, .stdin(), .stdout());
    const transport: *lsp.Transport = &stdio_transport.transport;

    var handler: Handler = .init(allocator, transport, &counting_allocator);
    defer handler.deinit();

    counting_allocator.setScope(.request_protocol);
    try lsp.basic_server.run(allocator, transport, &handler, std.log.err);
}
