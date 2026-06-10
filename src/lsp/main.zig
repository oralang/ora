const std = @import("std");
const lsp = @import("lsp");
const types = lsp.types;
const ora_root = @import("ora_root");

const frontend = ora_root.lsp.frontend;
const workspace = ora_root.lsp.workspace;
const dependency_graph = ora_root.lsp.dependency_graph;
const semantic_index = ora_root.lsp.semantic_index;
const phase_stats = ora_root.lsp.phase_stats;
const allocation_stats = ora_root.lsp.allocation_stats;
const diagnostics_api = ora_root.lsp.diagnostics;
const diagnostic_debounce = ora_root.lsp.diagnostic_debounce;
const line_index_api = ora_root.lsp.line_index;
const text_edits = ora_root.lsp.text_edits;
const definition_api = ora_root.lsp.definition;
const references_api = ora_root.lsp.references;
const semantic_tokens_api = ora_root.lsp.semantic_tokens;
const std_docs_api = ora_root.lsp.std_docs;
const token_cache_api = ora_root.lsp.token_cache;
const call_hierarchy_api = ora_root.lsp.call_hierarchy;
const workspace_index_api = ora_root.lsp.workspace_index;
const workspace_discovery_api = ora_root.lsp.workspace_discovery;
const workspace_roots_api = ora_root.lsp.workspace_roots;
const formatting_api = @import("formatting.zig");
const definition_requests = @import("definition_requests.zig");
const document_feature_requests = @import("document_feature_requests.zig");
const document_sync_requests = @import("document_sync_requests.zig");
const initialization_requests = @import("initialization_requests.zig");
const call_hierarchy_requests = @import("call_hierarchy_requests.zig");
const references_requests = @import("references_requests.zig");
const rename_requests = @import("rename_requests.zig");
const workspace_requests = @import("workspace_requests.zig");
const protocol_helpers = @import("protocol_helpers.zig");
const response_stats = @import("response_stats.zig");
const compiler = ora_root.compiler;
const Allocator = std.mem.Allocator;

const DocumentState = struct {
    arena: std.heap.ArenaAllocator,
    line_index: ?line_index_api.LineIndex = null,
    token_cache: ?token_cache_api.Cache = null,
    semantic_tokens: ?[]semantic_tokens_api.SemanticToken = null,
    formatting_cache: ?FormattingCacheEntry = null,
    import_resolution: ?ImportResolutionCacheEntry = null,
    std_import_aliases: ?[]std_docs_api.ImportAlias = null,
    semantic_index: ?semantic_index.SemanticIndex = null,
    occurrence_index: ?references_api.OccurrenceIndex = null,
    imported_member_index: ?references_api.ImportedMemberIndex = null,
    call_edge_index: ?call_hierarchy_api.CallEdgeIndex = null,
    diagnostic_cache: ?diagnostics_api.CacheEntry = null,

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
        self.std_import_aliases = null;
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
        self.std_import_aliases = null;
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
        self.std_import_aliases = null;
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

fn invalidateImportDependentState(state: *DocumentState) void {
    state.invalidateSemanticIndex();
    state.invalidateOccurrenceIndex();
    state.invalidateDiagnosticCache();
}

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
const max_cold_document_count: usize = max_workspace_discovery_files;
const max_cold_source_bytes: usize = 64 * 1024 * 1024;

const DocumentVersionState = struct {
    version: i32,
    generation: u64,
    is_cold: bool,
};

const ColdDocumentFreshness = enum {
    missing,
    fresh,
    refreshed,
    stale_removed,
};

const ColdFileFingerprint = struct {
    size: u64,
    mtime: i128,
};

const ColdDocumentRecord = struct {
    normalized_path: []u8,
    file_id: compiler.FileId,
    module_id: compiler.ModuleId,
    version: i32 = 0,
    generation: u64 = 1,
    source_bytes: usize,
    file_size: u64,
    mtime: i128,
    last_access: u64 = 0,
    state: DocumentState,

    fn deinit(self: *ColdDocumentRecord, allocator: Allocator) void {
        self.state.deinit();
        allocator.free(self.normalized_path);
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

const OptionalAllocationScope = struct {
    guard: ?allocation_stats.ScopeGuard = null,

    pub fn deinit(self: *OptionalAllocationScope) void {
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

const IncomingCallTargetUris = std.ArrayList([]u8);

const DocumentStateSource = struct {
    state: *DocumentState,
    source: []const u8,
};

const CompilerDbDocument = struct {
    file_id: compiler.FileId,
    module_id: compiler.ModuleId,
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
    occurrence_index_builds: usize,
    imported_member_index_builds: usize,
    call_edge_index_builds: usize,
    diagnostic_cache_builds: usize,
    diagnostic_fast_builds: usize,
    diagnostic_full_builds: usize,
    workspace_index_builds: usize,
    cold_workspace_index_builds: usize,
    cold_document_access_clock: u64,
    cold_document_source_bytes: usize,
    cold_document_evictions: usize,
    cold_document_refresh_checks: usize,
    cold_document_refreshes: usize,
    cold_document_stale_removals: usize,

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
            .occurrence_index_builds = 0,
            .imported_member_index_builds = 0,
            .call_edge_index_builds = 0,
            .diagnostic_cache_builds = 0,
            .diagnostic_fast_builds = 0,
            .diagnostic_full_builds = 0,
            .workspace_index_builds = 0,
            .cold_workspace_index_builds = 0,
            .cold_document_access_clock = 0,
            .cold_document_source_bytes = 0,
            .cold_document_evictions = 0,
            .cold_document_refresh_checks = 0,
            .cold_document_refreshes = 0,
            .cold_document_stale_removals = 0,
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

    pub fn put(self: *DocumentStore, uri: []const u8, text: []const u8, version: i32) !void {
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

    pub fn remove(self: *DocumentStore, uri: []const u8) void {
        self.invalidateWorkspaceIndex(uri);
        self.clearIncomingCallTargetIndex();
        if (self.db_records.fetchRemove(uri)) |removed| {
            var record = removed.value;
            self.compiler_db.releaseSourceFileFrontendOnly(record.file_id);
            record.deinit();
            self.allocator.free(removed.key);
        }
        if (self.docs.fetchRemove(uri)) |removed| {
            self.allocator.free(removed.key);
        }
    }

    pub fn putColdDocument(self: *DocumentStore, uri: []const u8, normalized_path: []const u8, source: []const u8) !void {
        if (self.docs.contains(uri) or self.db_records.contains(uri)) return;
        if (self.cold_docs.getPtr(uri)) |record| {
            self.touchColdDocumentRecord(record);
            return;
        }

        var scope = self.allocationScope(.cache_build);
        defer scope.deinit();

        const fingerprint = coldFileFingerprint(normalized_path) catch ColdFileFingerprint{
            .size = @intCast(source.len),
            .mtime = 0,
        };

        const uri_copy = try self.allocator.dupe(u8, uri);
        errdefer self.allocator.free(uri_copy);

        const path_copy = try self.allocator.dupe(u8, normalized_path);
        errdefer self.allocator.free(path_copy);

        const package_id = try self.getOrCreatePackage();
        const file_id = try self.compiler_db.addSourceFile(normalized_path, source);
        const module_id = try self.compiler_db.addModule(package_id, file_id, moduleNameForPath(normalized_path));

        var state = DocumentState.init(self.allocator);
        errdefer state.deinit();

        try self.cold_docs.put(uri_copy, .{
            .normalized_path = path_copy,
            .file_id = file_id,
            .module_id = module_id,
            .source_bytes = source.len,
            .file_size = fingerprint.size,
            .mtime = fingerprint.mtime,
            .state = state,
        });
        self.cold_document_source_bytes = addSat(self.cold_document_source_bytes, source.len);
        self.touchColdDocument(uri);
        self.evictColdDocumentsToBudget(uri);
    }

    fn removeColdDocument(self: *DocumentStore, uri: []const u8) void {
        if (self.cold_docs.fetchRemove(uri)) |removed| {
            self.invalidateWorkspaceIndex(uri);
            self.clearIncomingCallTargetIndex();
            self.allocator.free(removed.key);
            var record = removed.value;
            self.compiler_db.releaseSourceFileFrontendOnly(record.file_id);
            self.subtractColdSourceBytes(record.source_bytes);
            record.deinit(self.allocator);
        }
    }

    pub fn ensureFreshColdDocument(self: *DocumentStore, uri: []const u8) !ColdDocumentFreshness {
        const record = self.cold_docs.getPtr(uri) orelse return .missing;
        self.cold_document_refresh_checks = addSat(self.cold_document_refresh_checks, 1);

        const fingerprint = coldFileFingerprint(record.normalized_path) catch {
            self.removeStaleColdDocument(uri);
            return .stale_removed;
        };

        if (record.file_size == fingerprint.size and record.mtime == fingerprint.mtime) {
            self.touchColdDocumentRecord(record);
            return .fresh;
        }

        if (fingerprint.size > max_cold_file_bytes) {
            self.removeStaleColdDocument(uri);
            return .stale_removed;
        }

        const loaded = std.fs.cwd().readFileAlloc(self.allocator, record.normalized_path, max_cold_file_bytes) catch |err| switch (err) {
            error.OutOfMemory => return error.OutOfMemory,
            else => {
                self.removeStaleColdDocument(uri);
                return .stale_removed;
            },
        };
        defer self.allocator.free(loaded);

        try self.compiler_db.updateSourceFileFrontendOnly(record.file_id, loaded);
        self.subtractColdSourceBytes(record.source_bytes);
        self.cold_document_source_bytes = addSat(self.cold_document_source_bytes, loaded.len);
        record.source_bytes = loaded.len;
        record.file_size = fingerprint.size;
        record.mtime = fingerprint.mtime;
        record.generation = bumpGeneration(record.generation);
        record.state.reset();
        self.invalidateWorkspaceIndex(uri);
        self.clearIncomingCallTargetIndex();
        self.cold_document_refreshes = addSat(self.cold_document_refreshes, 1);
        self.touchColdDocumentRecord(record);
        self.evictColdDocumentsToBudget(uri);
        return .refreshed;
    }

    fn removeStaleColdDocument(self: *DocumentStore, uri: []const u8) void {
        const before = self.cold_docs.count();
        self.removeColdDocument(uri);
        if (self.cold_docs.count() < before) {
            self.cold_document_stale_removals = addSat(self.cold_document_stale_removals, 1);
        }
    }

    fn touchColdDocument(self: *DocumentStore, uri: []const u8) void {
        if (self.cold_docs.getPtr(uri)) |record| {
            self.touchColdDocumentRecord(record);
        }
    }

    fn touchColdDocumentRecord(self: *DocumentStore, record: *ColdDocumentRecord) void {
        self.cold_document_access_clock +%= 1;
        record.last_access = self.cold_document_access_clock;
    }

    fn evictColdDocumentsToBudget(self: *DocumentStore, protected_uri: []const u8) void {
        while (self.cold_docs.count() > 1 and self.coldDocumentsOverBudget()) {
            const oldest_uri = self.oldestEvictableColdDocumentUri(protected_uri) orelse return;
            self.evictColdDocument(oldest_uri);
        }
    }

    fn coldDocumentsOverBudget(self: *const DocumentStore) bool {
        return self.cold_docs.count() > max_cold_document_count or
            self.cold_document_source_bytes > max_cold_source_bytes;
    }

    fn oldestEvictableColdDocumentUri(self: *DocumentStore, protected_uri: []const u8) ?[]const u8 {
        var oldest_uri: ?[]const u8 = null;
        var oldest_access: u64 = std.math.maxInt(u64);

        var it = self.cold_docs.iterator();
        while (it.next()) |entry| {
            if (std.mem.eql(u8, entry.key_ptr.*, protected_uri)) continue;
            if (entry.value_ptr.last_access < oldest_access) {
                oldest_access = entry.value_ptr.last_access;
                oldest_uri = entry.key_ptr.*;
            }
        }

        return oldest_uri;
    }

    fn evictColdDocument(self: *DocumentStore, uri: []const u8) void {
        const before = self.cold_docs.count();
        self.removeColdDocument(uri);
        if (self.cold_docs.count() < before) {
            self.cold_document_evictions = addSat(self.cold_document_evictions, 1);
        }
    }

    fn subtractColdSourceBytes(self: *DocumentStore, byte_count: usize) void {
        self.cold_document_source_bytes = if (byte_count > self.cold_document_source_bytes)
            0
        else
            self.cold_document_source_bytes - byte_count;
    }

    pub fn documentVersionStateForUri(self: *DocumentStore, uri: []const u8) ?DocumentVersionState {
        if (self.db_records.get(uri)) |record| {
            return .{ .version = record.version, .generation = record.generation, .is_cold = false };
        }
        if (self.cold_docs.get(uri)) |record| {
            return .{ .version = record.version, .generation = record.generation, .is_cold = true };
        }
        return null;
    }

    pub fn isStaleOpenVersion(self: *DocumentStore, uri: []const u8, version: i32) bool {
        const state = self.documentVersionStateForUri(uri) orelse return false;
        return !state.is_cold and version <= state.version;
    }

    pub fn sourceForUri(self: *DocumentStore, uri: []const u8) ?[]const u8 {
        if (self.db_records.get(uri)) |record| {
            return self.compiler_db.sourceText(record.file_id);
        }
        if (self.cold_docs.getPtr(uri)) |record| {
            self.touchColdDocumentRecord(record);
            return self.compiler_db.sourceText(record.file_id);
        }
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
            self.touchColdDocumentRecord(record);
            return .{
                .state = &record.state,
                .source = self.compiler_db.sourceText(record.file_id),
            };
        }
        return null;
    }

    fn compilerDbDocumentForUri(self: *DocumentStore, uri: []const u8) ?CompilerDbDocument {
        if (self.db_records.getPtr(uri)) |record| {
            return .{
                .file_id = record.file_id,
                .module_id = record.module_id,
                .state = &record.state,
                .source = self.compiler_db.sourceText(record.file_id),
            };
        }
        if (self.cold_docs.getPtr(uri)) |record| {
            self.touchColdDocumentRecord(record);
            return .{
                .file_id = record.file_id,
                .module_id = record.module_id,
                .state = &record.state,
                .source = self.compiler_db.sourceText(record.file_id),
            };
        }
        return null;
    }

    pub fn isOpenDocument(self: *DocumentStore, uri: []const u8) bool {
        return self.docs.contains(uri);
    }

    pub fn openDocumentCount(self: *DocumentStore) usize {
        return self.docs.count();
    }

    pub fn astFileForUri(self: *DocumentStore, uri: []const u8, stats: ?*phase_stats.Stats) !?*const compiler.ast.AstFile {
        const document = self.compilerDbDocumentForUri(uri) orelse return null;
        const needs_parse = !self.compiler_db.hasSyntaxResult(document.file_id);
        const needs_lower = !self.compiler_db.hasAstResult(document.file_id);
        if (needs_parse) phase_stats.record(stats, .parse);
        if (needs_lower) phase_stats.record(stats, .ast_lower);
        return try self.compiler_db.astFile(document.file_id);
    }

    pub fn definitionAnalysisForUri(self: *DocumentStore, uri: []const u8, stats: ?*phase_stats.Stats) !?definition_api.Analysis {
        const document = self.compilerDbDocumentForUri(uri) orelse return null;

        var scope = self.allocationScope(.cache_build);
        defer scope.deinit();

        const needs_parse = !self.compiler_db.hasSyntaxResult(document.file_id);
        const needs_lower = !self.compiler_db.hasAstResult(document.file_id);
        const needs_item_index = !self.compiler_db.hasItemIndexResult(document.module_id);
        const needs_resolution = !self.compiler_db.hasResolutionResult(document.module_id);
        if (needs_parse) phase_stats.record(stats, .parse);
        if (needs_lower) phase_stats.record(stats, .ast_lower);
        const ast_file = try self.compiler_db.astFile(document.file_id);
        if (needs_item_index) phase_stats.record(stats, .item_index);
        const item_index = try self.compiler_db.itemIndex(document.module_id);
        if (needs_resolution) phase_stats.record(stats, .resolve);
        const resolution = try self.compiler_db.resolveNames(document.module_id);

        return definition_api.Analysis.initBorrowed(
            self.allocator,
            &self.compiler_db.sources,
            document.file_id,
            document.module_id,
            document.source,
            ast_file,
            item_index,
            resolution,
        );
    }

    pub fn lineIndexForUri(self: *DocumentStore, uri: []const u8) !?*const line_index_api.LineIndex {
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
            var index = try line_index_api.LineIndex.init(state_allocator, self.compiler_db.sourceText(record.file_id));
            errdefer index.deinit(state_allocator);
            record.state.line_index = index;
        }
        return &record.state.line_index.?;
    }

    pub fn occurrenceIndexForUri(self: *DocumentStore, uri: []const u8) ?*const references_api.OccurrenceIndex {
        const state_source = self.documentStateAndSourceForUri(uri) orelse return null;
        return if (state_source.state.occurrence_index) |*index| index else null;
    }

    pub fn importedMemberIndexForUri(self: *DocumentStore, uri: []const u8) ?*const references_api.ImportedMemberIndex {
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

    pub fn stdImportAliasesForUri(self: *DocumentStore, uri: []const u8, stats: ?*phase_stats.Stats) !?[]const std_docs_api.ImportAlias {
        const state_source = self.documentStateAndSourceForUri(uri) orelse return null;
        if (state_source.state.std_import_aliases) |aliases| return aliases;

        var scope = self.allocationScope(.cache_build);
        defer scope.deinit();

        const tokens = (try self.tokenCacheForUri(uri, stats)) orelse return null;
        const aliases = try std_docs_api.collectImportAliases(state_source.state.allocator(), tokens.tokens);
        state_source.state.std_import_aliases = aliases;
        return state_source.state.std_import_aliases.?;
    }

    pub fn semanticIndexForUri(self: *DocumentStore, uri: []const u8, stats: ?*phase_stats.Stats) !?*const semantic_index.SemanticIndex {
        const document = self.compilerDbDocumentForUri(uri) orelse return null;
        if (document.state.semantic_index) |*cached| return cached;

        var scope = self.allocationScope(.cache_build);
        defer scope.deinit();

        const needs_parse = !self.compiler_db.hasSyntaxResult(document.file_id);
        const needs_lower = !self.compiler_db.hasAstResult(document.file_id);
        if (needs_parse) phase_stats.record(stats, .parse);
        const syntax_diagnostics = try self.compiler_db.syntaxDiagnostics(document.file_id);
        if (needs_lower) phase_stats.record(stats, .ast_lower);
        const ast_file = try self.compiler_db.astFile(document.file_id);
        const ast_diagnostics = try self.compiler_db.astDiagnostics(document.file_id);
        const state_allocator = document.state.allocator();
        var index = try semantic_index.indexAstFileWithSourceStoreAlloc(
            state_allocator,
            self.allocator,
            &self.compiler_db.sources,
            document.file_id,
            document.source,
            ast_file,
            syntax_diagnostics.isEmpty() and ast_diagnostics.isEmpty(),
        );
        errdefer index.deinit(state_allocator);

        document.state.semantic_index = index;
        self.semantic_index_builds = addSat(self.semantic_index_builds, 1);
        return &document.state.semantic_index.?;
    }

    pub fn semanticTokensForUri(self: *DocumentStore, uri: []const u8, stats: ?*phase_stats.Stats) !?[]const semantic_tokens_api.SemanticToken {
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

    pub fn formattingCacheForUri(
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

    pub fn importResolutionForUri(
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

    pub fn diagnosticCacheForUri(
        self: *DocumentStore,
        uri: []const u8,
        workspace_roots: []const []const u8,
        depth: diagnostics_api.Depth,
        stats: ?*phase_stats.Stats,
    ) !?*const diagnostics_api.CacheEntry {
        const state = self.documentVersionStateForUri(uri) orelse return null;
        const state_source = self.documentStateAndSourceForUri(uri) orelse return null;
        if (state_source.state.diagnostic_cache) |*cached| {
            if (cached.matches(state.version, state.generation, depth)) return cached;
            state_source.state.invalidateDiagnosticCache();
        }

        var scope = self.allocationScope(.cache_build);
        defer scope.deinit();

        const document = self.compilerDbDocumentForUri(uri) orelse return null;
        const state_allocator = state_source.state.allocator();
        var diagnostics = std.ArrayList(diagnostics_api.Diagnostic){};
        defer diagnostics.deinit(state_allocator);

        if (try self.tokenCacheForUri(uri, stats)) |token_cache| {
            try diagnostics_api.appendLexerDiagnostics(state_allocator, &diagnostics, token_cache.diagnostics);
        }

        const needs_parse = !self.compiler_db.hasSyntaxResult(document.file_id);
        const needs_lower = !self.compiler_db.hasAstResult(document.file_id);
        if (needs_parse) phase_stats.record(stats, .parse);
        const syntax_diagnostics = try self.compiler_db.syntaxDiagnostics(document.file_id);
        try diagnostics_api.appendCompilerDiagnostics(
            state_allocator,
            &diagnostics,
            &self.compiler_db.sources,
            document.file_id,
            .parser,
            syntax_diagnostics,
        );

        if (needs_lower) phase_stats.record(stats, .ast_lower);
        const ast_file = try self.compiler_db.astFile(document.file_id);
        const ast_diagnostics = try self.compiler_db.astDiagnostics(document.file_id);
        try diagnostics_api.appendCompilerDiagnostics(
            state_allocator,
            &diagnostics,
            &self.compiler_db.sources,
            document.file_id,
            .parser,
            ast_diagnostics,
        );

        if (depth == .full and syntax_diagnostics.isEmpty() and ast_diagnostics.isEmpty()) {
            if (!self.compiler_db.hasItemIndexResult(document.module_id)) phase_stats.record(stats, .item_index);
            if (!self.compiler_db.hasResolutionResult(document.module_id)) phase_stats.record(stats, .resolve);
            const resolution_diagnostics = try self.compiler_db.resolutionDiagnostics(document.module_id);
            try diagnostics_api.appendCompilerDiagnostics(
                state_allocator,
                &diagnostics,
                &self.compiler_db.sources,
                document.file_id,
                .sema,
                resolution_diagnostics,
            );

            if (!self.compiler_db.hasConstEvalResult(document.module_id)) phase_stats.record(stats, .const_eval);
            const const_eval_diagnostics = try self.compiler_db.constEvalDiagnostics(document.module_id);
            try diagnostics_api.appendCompilerDiagnostics(
                state_allocator,
                &diagnostics,
                &self.compiler_db.sources,
                document.file_id,
                .sema,
                const_eval_diagnostics,
            );

            if (ast_file.bodies.len > 0) {
                const key: compiler.sema.TypeCheckKey = .{ .body = compiler.ast.BodyId.fromIndex(0) };
                if (!self.compiler_db.hasTypeCheckResult(document.module_id, key)) phase_stats.record(stats, .type_check);
                const typecheck_diagnostics = try self.compiler_db.typeCheckDiagnostics(document.module_id, key);
                try diagnostics_api.appendCompilerDiagnostics(
                    state_allocator,
                    &diagnostics,
                    &self.compiler_db.sources,
                    document.file_id,
                    .sema,
                    typecheck_diagnostics,
                );
            }
        }

        if (try self.importResolutionForUri(uri, workspace_roots, stats)) |import_resolution| {
            try diagnostics_api.appendImportDiagnostics(state_allocator, &diagnostics, import_resolution.diagnostics);
        }

        state_source.state.diagnostic_cache = .{
            .version = state.version,
            .generation = state.generation,
            .depth = depth,
            .diagnostics = try diagnostics.toOwnedSlice(state_allocator),
        };
        self.diagnostic_cache_builds = addSat(self.diagnostic_cache_builds, 1);
        switch (depth) {
            .fast => self.diagnostic_fast_builds = addSat(self.diagnostic_fast_builds, 1),
            .full => self.diagnostic_full_builds = addSat(self.diagnostic_full_builds, 1),
        }
        return &state_source.state.diagnostic_cache.?;
    }

    pub fn workspaceEntryForUri(
        self: *DocumentStore,
        uri: []const u8,
        features: workspace_index_api.FeatureSet,
        workspace_roots: []const []const u8,
        stats: ?*phase_stats.Stats,
    ) !?*const workspace_index_api.FileEntry {
        const document_state = self.documentVersionStateForUri(uri) orelse return null;
        if (self.workspace_index.getFresh(uri, document_state.version, document_state.generation, features)) |entry| return entry;

        var requested_features = features;
        if (self.workspace_index.getFreshAny(uri, document_state.version, document_state.generation)) |entry| {
            requested_features = entry.features.merged(features);
        }

        var resolved_imports: []const workspace.ResolvedImport = &.{};
        if (requested_features.occurrences and self.occurrenceIndexForUri(uri) == null) {
            const source = self.sourceForUri(uri) orelse return null;
            try self.rebuildOccurrenceIndex(uri, source, stats);
        }
        if (requested_features.imported_members) {
            const import_resolution = (try self.importResolutionForUri(uri, workspace_roots, stats)) orelse return null;
            resolved_imports = import_resolution.imports;
            if (self.importedMemberIndexForUri(uri) == null) {
                try self.rebuildImportedMemberIndex(uri, import_resolution.imports, stats);
            }
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

        var entry = try workspace_index_api.FileEntry.init(
            self.allocator,
            uri,
            document_state.version,
            document_state.generation,
            document_state.is_cold,
            requested_features,
            line_index,
            semantic_symbols,
            resolved_imports,
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

    pub fn ensureIncomingCallTargetIndex(
        self: *DocumentStore,
        workspace_roots: []const []const u8,
        stats: ?*phase_stats.Stats,
    ) !void {
        if (self.incoming_call_target_index_built) return;

        var scope = self.allocationScope(.cache_build);
        defer scope.deinit();

        self.clearIncomingCallTargetIndex();
        errdefer self.clearIncomingCallTargetIndex();

        var doc_it = self.docs.iterator();
        while (doc_it.next()) |entry| {
            const doc_uri = entry.key_ptr.*;
            const workspace_entry = (try self.workspaceEntryForUri(doc_uri, .calls, workspace_roots, stats)) orelse continue;
            for (workspace_entry.call_edges) |edge| {
                try self.appendIncomingCallTarget(edge.callee_name, doc_uri);
            }
        }

        self.incoming_call_target_index_built = true;
        self.incoming_call_target_index_builds = addSat(self.incoming_call_target_index_builds, 1);
    }

    pub fn incomingCallTargetUris(self: *DocumentStore, target_name: []const u8) ?[]const []const u8 {
        const uris = self.incoming_call_targets.getPtr(target_name) orelse return null;
        return uris.items;
    }

    fn appendIncomingCallTarget(self: *DocumentStore, callee_name: []const u8, uri: []const u8) !void {
        if (self.incoming_call_targets.getPtr(callee_name)) |uris| {
            if (protocol_helpers.containsString(uris.items, uri)) return;

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

    pub fn rebuildOccurrenceIndex(self: *DocumentStore, uri: []const u8, source: []const u8, stats: ?*phase_stats.Stats) !void {
        self.invalidateOccurrenceIndex(uri);
        const state_source = self.documentStateAndSourceForUri(uri) orelse return;
        const token_cache = (try self.tokenCacheForUri(uri, stats)) orelse return;

        var scope = self.allocationScope(.cache_build);
        defer scope.deinit();

        const state_allocator = state_source.state.allocator();
        var analysis = (try self.definitionAnalysisForUri(uri, stats)) orelse return;
        defer analysis.deinit();

        var index = try references_api.OccurrenceIndex.init(state_allocator, source, token_cache.tokens, &analysis);
        errdefer index.deinit(state_allocator);

        if (index.occurrences.len == 0) {
            index.deinit(state_allocator);
            return;
        }

        state_source.state.occurrence_index = index;
        self.occurrence_index_builds = addSat(self.occurrence_index_builds, 1);
    }

    pub fn rebuildImportedMemberIndex(
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
        self.imported_member_index_builds = addSat(self.imported_member_index_builds, 1);
    }

    fn registerDbRecord(self: *DocumentStore, uri: []const u8, text: []const u8, version: i32) !void {
        if (self.db_records.getPtr(uri)) |record| {
            try self.compiler_db.updateSourceFileFrontendOnly(record.file_id, text);
            record.version = version;
            record.generation = bumpGeneration(record.generation);
            record.state.reset();
            return;
        }

        if (self.cold_docs.fetchRemove(uri)) |removed| {
            var cold_record = removed.value;
            var state_moved = false;
            self.subtractColdSourceBytes(cold_record.source_bytes);
            errdefer {
                self.allocator.free(removed.key);
                self.allocator.free(cold_record.normalized_path);
                if (!state_moved) cold_record.state.deinit();
            }

            try self.compiler_db.updateSourceFileFrontendOnly(cold_record.file_id, text);
            cold_record.state.reset();

            const record: DocumentDbRecord = .{
                .file_id = cold_record.file_id,
                .module_id = cold_record.module_id,
                .version = version,
                .generation = bumpGeneration(cold_record.generation),
                .state = cold_record.state,
            };
            try self.db_records.put(removed.key, record);
            state_moved = true;
            self.allocator.free(cold_record.normalized_path);
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

    pub fn invalidateImportDependentCaches(self: *DocumentStore, uri: []const u8) void {
        if (self.db_records.getPtr(uri)) |record| {
            invalidateImportDependentState(&record.state);
        }
        if (self.cold_docs.getPtr(uri)) |record| {
            invalidateImportDependentState(&record.state);
        }
        self.invalidateWorkspaceIndex(uri);
        self.clearIncomingCallTargetIndex();
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

    pub fn clearImportResolutionCache(self: *DocumentStore) void {
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

    pub fn openSourceBytes(self: *DocumentStore) usize {
        var total: usize = 0;
        var iterator = self.docs.keyIterator();
        while (iterator.next()) |uri| {
            if (self.sourceForUri(uri.*)) |source| {
                total = addSat(total, source.len);
            }
        }
        return total;
    }

    pub fn coldDocumentCount(self: *DocumentStore) usize {
        return self.cold_docs.count();
    }

    pub fn coldSourceBytes(self: *DocumentStore) usize {
        return self.cold_document_source_bytes;
    }

    pub fn coldSourceMaxBytes(_: *DocumentStore) usize {
        return max_cold_source_bytes;
    }

    pub fn coldDocumentMaxCount(_: *DocumentStore) usize {
        return max_cold_document_count;
    }

    pub fn coldDocumentEvictions(self: *DocumentStore) usize {
        return self.cold_document_evictions;
    }

    pub fn coldDocumentRefreshChecks(self: *DocumentStore) usize {
        return self.cold_document_refresh_checks;
    }

    pub fn coldDocumentRefreshes(self: *DocumentStore) usize {
        return self.cold_document_refreshes;
    }

    pub fn coldDocumentStaleRemovals(self: *DocumentStore) usize {
        return self.cold_document_stale_removals;
    }

    pub fn lineIndexBytes(self: *DocumentStore) usize {
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

    pub fn tokenCacheBytes(self: *DocumentStore) usize {
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

    pub fn tokenCount(self: *DocumentStore) usize {
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

    pub fn tokenDiagnosticCount(self: *DocumentStore) usize {
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

    pub fn importResolutionBytes(self: *DocumentStore) usize {
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

    pub fn semanticIndexBytes(self: *DocumentStore) usize {
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

    pub fn symbolCount(self: *DocumentStore) usize {
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

    pub fn importedMemberIndexBytes(self: *DocumentStore) usize {
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

    pub fn importedMemberCount(self: *DocumentStore) usize {
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

    pub fn occurrenceIndexBytes(self: *DocumentStore) usize {
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

    pub fn occurrenceCount(self: *DocumentStore) usize {
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

    pub fn callEdgeIndexBytes(self: *DocumentStore) usize {
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

    pub fn callEdgeCount(self: *DocumentStore) usize {
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

    pub fn diagnosticCacheBytes(self: *DocumentStore) usize {
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

    pub fn formattingCacheBytes(self: *DocumentStore) usize {
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

    pub fn formattingCacheEntries(self: *DocumentStore) usize {
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

    pub fn cacheBuilderCapacityRequested(self: *DocumentStore) usize {
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

    pub fn cacheBuilderItemsBuilt(self: *DocumentStore) usize {
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

    pub fn cacheBuilderUnusedCapacity(self: *DocumentStore) usize {
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

    pub fn cacheBuilderGrowthEvents(self: *DocumentStore) usize {
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

    pub fn cacheSideMapCapacityRequested(self: *DocumentStore) usize {
        return self.workspace_index.sideMapCapacityRequested();
    }

    pub fn cacheSideMapItemsBuilt(self: *DocumentStore) usize {
        return self.workspace_index.sideMapItemsBuilt();
    }

    pub fn cacheSideMapUnusedCapacity(self: *DocumentStore) usize {
        return self.workspace_index.sideMapUnusedCapacity();
    }

    pub fn cacheSideMapGrowthEvents(self: *DocumentStore) usize {
        return self.workspace_index.sideMapGrowthEvents();
    }

    pub fn documentStateBytes(self: *DocumentStore) usize {
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

    pub fn workspaceIndexEntries(self: *DocumentStore) usize {
        return self.workspace_index.entries.count();
    }

    pub fn workspaceIndexBytes(self: *DocumentStore) usize {
        return self.workspace_index.current_bytes;
    }

    pub fn workspaceIndexMaxBytes(self: *DocumentStore) usize {
        return self.workspace_index.max_bytes;
    }

    pub fn workspaceIndexEvictions(self: *DocumentStore) usize {
        return self.workspace_index.evictions;
    }

    pub fn workspaceIndexSymbolCount(self: *DocumentStore) usize {
        var total: usize = 0;
        var iterator = self.workspace_index.entries.valueIterator();
        while (iterator.next()) |entry| {
            total = addSat(total, entry.*.symbols.len);
        }
        return total;
    }

    pub fn workspaceIndexRootSymbolCount(self: *DocumentStore) usize {
        var total: usize = 0;
        var iterator = self.workspace_index.entries.valueIterator();
        while (iterator.next()) |entry| {
            total = addSat(total, entry.*.root_symbol_indexes.len);
        }
        return total;
    }

    pub fn workspaceIndexCallableSymbolCount(self: *DocumentStore) usize {
        var total: usize = 0;
        var iterator = self.workspace_index.entries.valueIterator();
        while (iterator.next()) |entry| {
            total = addSat(total, entry.*.callable_symbol_indexes.len);
        }
        return total;
    }

    pub fn workspaceIndexImportCount(self: *DocumentStore) usize {
        var total: usize = 0;
        var iterator = self.workspace_index.entries.valueIterator();
        while (iterator.next()) |entry| {
            total = addSat(total, entry.*.imports.len);
        }
        return total;
    }

    pub fn workspaceIndexOccurrenceCount(self: *DocumentStore) usize {
        var total: usize = 0;
        var iterator = self.workspace_index.entries.valueIterator();
        while (iterator.next()) |entry| {
            total = addSat(total, entry.*.occurrences.len);
        }
        return total;
    }

    pub fn workspaceIndexImportedMemberCount(self: *DocumentStore) usize {
        var total: usize = 0;
        var iterator = self.workspace_index.entries.valueIterator();
        while (iterator.next()) |entry| {
            total = addSat(total, entry.*.imported_members.len);
        }
        return total;
    }

    pub fn workspaceIndexCallEdgeCount(self: *DocumentStore) usize {
        var total: usize = 0;
        var iterator = self.workspace_index.entries.valueIterator();
        while (iterator.next()) |entry| {
            total = addSat(total, entry.*.call_edges.len);
        }
        return total;
    }

    pub fn workspaceIndexInternedStringBytes(self: *DocumentStore) usize {
        var total: usize = 0;
        var iterator = self.workspace_index.entries.valueIterator();
        while (iterator.next()) |entry| {
            total = addSat(total, entry.*.interned_string_bytes);
        }
        return total;
    }

    pub fn workspaceIndexInternedStringCapacityRequested(self: *DocumentStore) usize {
        return self.workspace_index.internedStringCapacityRequested();
    }

    pub fn workspaceIndexInternedStringItemsBuilt(self: *DocumentStore) usize {
        return self.workspace_index.internedStringItemsBuilt();
    }

    pub fn workspaceIndexInternedStringUnusedCapacity(self: *DocumentStore) usize {
        return self.workspace_index.internedStringUnusedCapacity();
    }

    pub fn workspaceIndexInternedStringGrowthEvents(self: *DocumentStore) usize {
        return self.workspace_index.internedStringGrowthEvents();
    }

    pub fn coldWorkspaceIndexEntries(self: *DocumentStore) usize {
        return self.workspace_index.coldEntryCount();
    }

    pub fn coldWorkspaceIndexBytes(self: *DocumentStore) usize {
        return self.workspace_index.coldBytes();
    }

    pub fn coldWorkspaceIndexInternedStringBytes(self: *DocumentStore) usize {
        return self.workspace_index.coldInternedStringBytes();
    }

    pub fn coldWorkspaceIndexInternedStringCount(self: *DocumentStore) usize {
        return self.workspace_index.coldInternedStringCount();
    }

    pub fn coldWorkspaceIndexDuplicateStringBytesSaved(self: *DocumentStore) usize {
        return self.workspace_index.coldDuplicateStringBytesSaved();
    }

    pub fn coldWorkspaceIndexInternedStringCapacityRequested(self: *DocumentStore) usize {
        return self.workspace_index.coldInternedStringCapacityRequested();
    }

    pub fn coldWorkspaceIndexInternedStringItemsBuilt(self: *DocumentStore) usize {
        return self.workspace_index.coldInternedStringItemsBuilt();
    }

    pub fn coldWorkspaceIndexInternedStringUnusedCapacity(self: *DocumentStore) usize {
        return self.workspace_index.coldInternedStringUnusedCapacity();
    }

    pub fn coldWorkspaceIndexInternedStringGrowthEvents(self: *DocumentStore) usize {
        return self.workspace_index.coldInternedStringGrowthEvents();
    }

    pub fn workspaceIndexInternedStringCount(self: *DocumentStore) usize {
        var total: usize = 0;
        var iterator = self.workspace_index.entries.valueIterator();
        while (iterator.next()) |entry| {
            total = addSat(total, entry.*.interned_string_count);
        }
        return total;
    }

    pub fn workspaceIndexDuplicateStringBytesSaved(self: *DocumentStore) usize {
        var total: usize = 0;
        var iterator = self.workspace_index.entries.valueIterator();
        while (iterator.next()) |entry| {
            total = addSat(total, entry.*.duplicate_string_bytes_saved);
        }
        return total;
    }

    pub fn incomingCallTargetNameCount(self: *DocumentStore) usize {
        return self.incoming_call_targets.count();
    }

    pub fn incomingCallTargetUriCount(self: *DocumentStore) usize {
        var total: usize = 0;
        var iterator = self.incoming_call_targets.valueIterator();
        while (iterator.next()) |uris| {
            total = addSat(total, uris.items.len);
        }
        return total;
    }

    pub fn incomingCallTargetIndexBytes(self: *DocumentStore) usize {
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

fn bumpGeneration(generation: u64) u64 {
    return if (generation == std.math.maxInt(u64)) generation else generation + 1;
}

fn moduleNameForPath(path: []const u8) []const u8 {
    const stem = std.fs.path.stem(path);
    return if (stem.len == 0) path else stem;
}

fn coldFileFingerprint(path: []const u8) !ColdFileFingerprint {
    const stat = try std.fs.cwd().statFile(path);
    return .{
        .size = stat.size,
        .mtime = stat.mtime,
    };
}

pub const Handler = struct {
    allocator: Allocator,
    transport: *lsp.Transport,
    docs: DocumentStore,
    std_docs_index: ?std_docs_api.Index = null,
    workspace_roots: workspace_roots_api.Store,
    workspace_discovery: workspace_discovery_api.Cache,
    dependencies: dependency_graph.Graph,
    diagnostics: diagnostic_debounce.Debouncer,
    position_encoding: text_edits.PositionEncoding,
    response_counters: response_stats.Stats = .{},
    stale_document_change_skips: usize = 0,
    edit_diagnostic_fast_publishes: usize = 0,
    edit_diagnostic_full_skips: usize = 0,
    dependent_diagnostic_publish_runs: usize = 0,
    dependent_diagnostic_published_documents: usize = 0,
    dependent_diagnostic_publish_skips: usize = 0,
    phase_counters: phase_stats.Stats = .{},
    allocator_tracker: ?*allocation_stats.CountingAllocator = null,

    fn init(allocator: Allocator, transport: *lsp.Transport, allocator_tracker: ?*allocation_stats.CountingAllocator) Handler {
        return .{
            .allocator = allocator,
            .transport = transport,
            .docs = DocumentStore.init(allocator, allocator_tracker),
            .workspace_roots = workspace_roots_api.Store.init(allocator),
            .workspace_discovery = workspace_discovery_api.Cache.init(allocator, max_workspace_discovery_files),
            .dependencies = dependency_graph.Graph.init(allocator),
            .diagnostics = diagnostic_debounce.Debouncer.init(allocator),
            .position_encoding = .utf16,
            .allocator_tracker = allocator_tracker,
        };
    }

    fn allocationScope(self: *Handler, scope: allocation_stats.Scope) OptionalAllocationScope {
        return beginAllocationScope(self.allocator_tracker, scope);
    }

    pub fn responseScope(self: *Handler) OptionalAllocationScope {
        return self.allocationScope(.response);
    }

    pub fn tempAnalysisScope(self: *Handler) OptionalAllocationScope {
        return self.allocationScope(.temp_analysis);
    }

    pub fn workspaceRootPaths(self: *const Handler) []const []const u8 {
        return self.workspace_roots.paths();
    }

    pub fn allocatorStats(self: *Handler) ?*const allocation_stats.Stats {
        return if (self.allocator_tracker) |tracker| &tracker.stats else null;
    }

    pub fn scopedAllocatorStats(self: *Handler, scope: allocation_stats.Scope) allocation_stats.ScopedStats {
        return if (self.allocator_tracker) |tracker| tracker.stats.scope(scope) else .{};
    }

    pub fn stdDocsIndex(self: *Handler) !*const std_docs_api.Index {
        if (self.std_docs_index == null) {
            var scope = self.allocationScope(.cache_build);
            defer scope.deinit();
            self.std_docs_index = try std_docs_api.Index.init(self.allocator);
        }
        return &self.std_docs_index.?;
    }

    fn deinit(self: *Handler) void {
        if (self.std_docs_index) |*index| index.deinit();
        self.docs.deinit();
        self.workspace_roots.deinit();
        self.workspace_discovery.deinit();
        self.dependencies.deinit();
        self.diagnostics.deinit();
    }

    pub fn initialize(
        self: *Handler,
        _: Allocator,
        params: types.InitializeParams,
    ) !types.InitializeResult {
        return initialization_requests.initialize(self, params);
    }

    pub fn initialized(_: *Handler, _: Allocator, _: types.InitializedParams) void {}

    pub fn shutdown(_: *Handler, _: Allocator, _: void) ?void {
        return null;
    }

    pub fn exit(_: *Handler, _: Allocator, _: void) void {}

    pub fn onResponse(_: *Handler, _: Allocator, _: lsp.JsonRPCMessage.Response) void {}

    pub fn @"textDocument/didOpen"(self: *Handler, arena: Allocator, notification: types.DidOpenTextDocumentParams) !void {
        try document_sync_requests.didOpen(self, arena, notification);
    }

    pub fn @"textDocument/didChange"(self: *Handler, arena: Allocator, notification: types.DidChangeTextDocumentParams) !void {
        try document_sync_requests.didChange(self, arena, notification);
    }

    pub fn @"textDocument/didClose"(self: *Handler, arena: Allocator, notification: types.DidCloseTextDocumentParams) !void {
        try document_sync_requests.didClose(self, arena, notification);
    }

    pub fn @"textDocument/didSave"(self: *Handler, arena: Allocator, notification: types.DidSaveTextDocumentParams) !void {
        try document_sync_requests.didSave(self, arena, notification);
    }

    pub fn @"textDocument/documentSymbol"(self: *Handler, arena: Allocator, params: types.DocumentSymbolParams) !lsp.ResultType("textDocument/documentSymbol") {
        return document_feature_requests.documentSymbol(self, arena, params);
    }

    pub fn @"textDocument/hover"(self: *Handler, arena: Allocator, params: types.HoverParams) !?types.Hover {
        return document_feature_requests.hover(self, arena, params);
    }

    pub fn @"textDocument/definition"(self: *Handler, arena: Allocator, params: types.DefinitionParams) !lsp.ResultType("textDocument/definition") {
        return definition_requests.definition(self, arena, params);
    }

    pub fn @"textDocument/references"(self: *Handler, arena: Allocator, params: types.ReferenceParams) !lsp.ResultType("textDocument/references") {
        return references_requests.references(self, arena, params);
    }

    pub fn @"textDocument/completion"(self: *Handler, arena: Allocator, params: types.CompletionParams) !lsp.ResultType("textDocument/completion") {
        return document_feature_requests.completion(self, arena, params);
    }

    pub fn @"textDocument/prepareRename"(self: *Handler, arena: Allocator, params: types.PrepareRenameParams) !lsp.ResultType("textDocument/prepareRename") {
        return rename_requests.prepareRename(self, arena, params);
    }

    pub fn @"textDocument/rename"(self: *Handler, arena: Allocator, params: types.RenameParams) !lsp.ResultType("textDocument/rename") {
        return rename_requests.rename(self, arena, params);
    }

    pub fn @"textDocument/inlayHint"(self: *Handler, arena: Allocator, params: types.InlayHintParams) !lsp.ResultType("textDocument/inlayHint") {
        return document_feature_requests.inlayHint(self, arena, params);
    }

    pub fn @"textDocument/codeLens"(self: *Handler, arena: Allocator, params: types.CodeLensParams) !lsp.ResultType("textDocument/codeLens") {
        return document_feature_requests.codeLens(self, arena, params);
    }

    pub fn @"textDocument/signatureHelp"(self: *Handler, arena: Allocator, params: types.SignatureHelpParams) !?types.SignatureHelp {
        return document_feature_requests.signatureHelp(self, arena, params);
    }

    pub fn @"textDocument/semanticTokens/full"(self: *Handler, arena: Allocator, params: types.SemanticTokensParams) !lsp.ResultType("textDocument/semanticTokens/full") {
        return document_feature_requests.semanticTokensFull(self, arena, params);
    }

    pub fn @"textDocument/formatting"(self: *Handler, arena: Allocator, params: types.DocumentFormattingParams) !lsp.ResultType("textDocument/formatting") {
        return document_feature_requests.formatting(self, arena, params);
    }

    pub fn @"textDocument/documentHighlight"(self: *Handler, arena: Allocator, params: types.DocumentHighlightParams) !lsp.ResultType("textDocument/documentHighlight") {
        return document_feature_requests.documentHighlight(self, arena, params);
    }

    pub fn @"textDocument/foldingRange"(self: *Handler, arena: Allocator, params: types.FoldingRangeParams) !lsp.ResultType("textDocument/foldingRange") {
        return document_feature_requests.foldingRange(self, arena, params);
    }

    pub fn @"workspace/executeCommand"(self: *Handler, arena: Allocator, params: types.ExecuteCommandParams) !lsp.ResultType("workspace/executeCommand") {
        return workspace_requests.executeCommand(self, arena, params);
    }

    pub fn @"textDocument/codeAction"(self: *Handler, arena: Allocator, params: types.CodeActionParams) !lsp.ResultType("textDocument/codeAction") {
        return document_feature_requests.codeAction(self, arena, params);
    }

    pub fn @"workspace/symbol"(self: *Handler, arena: Allocator, params: types.WorkspaceSymbolParams) !lsp.ResultType("workspace/symbol") {
        return workspace_requests.symbol(self, arena, params);
    }

    pub fn @"textDocument/selectionRange"(self: *Handler, arena: Allocator, params: types.SelectionRangeParams) !lsp.ResultType("textDocument/selectionRange") {
        return document_feature_requests.selectionRange(self, arena, params);
    }

    pub fn @"textDocument/documentLink"(self: *Handler, arena: Allocator, params: types.DocumentLinkParams) !lsp.ResultType("textDocument/documentLink") {
        return document_feature_requests.documentLink(self, arena, params);
    }

    pub fn @"textDocument/prepareCallHierarchy"(self: *Handler, arena: Allocator, params: types.CallHierarchyPrepareParams) !lsp.ResultType("textDocument/prepareCallHierarchy") {
        return call_hierarchy_requests.prepare(self, arena, params);
    }

    pub fn @"callHierarchy/incomingCalls"(self: *Handler, arena: Allocator, params: types.CallHierarchyIncomingCallsParams) !lsp.ResultType("callHierarchy/incomingCalls") {
        return call_hierarchy_requests.incoming(self, arena, params);
    }

    pub fn @"callHierarchy/outgoingCalls"(self: *Handler, arena: Allocator, params: types.CallHierarchyOutgoingCallsParams) !lsp.ResultType("callHierarchy/outgoingCalls") {
        return call_hierarchy_requests.outgoing(self, arena, params);
    }

    pub fn normalizedPathForUri(self: *Handler, uri: []const u8) !?[]u8 {
        var scope = self.allocationScope(.temp_analysis);
        defer scope.deinit();

        const maybe_path = try workspace.fileUriToPathAlloc(self.allocator, uri);
        const path = maybe_path orelse return null;
        defer self.allocator.free(path);
        return try workspace.normalizePathAlloc(self.allocator, path);
    }

    pub fn borrowedNormalizedPathForUri(self: *Handler, uri: []const u8) ?[]const u8 {
        return self.dependencies.getPathForUri(uri);
    }

    pub fn discoveredImportersForTargetPath(self: *Handler, target_path: []const u8) ![]const workspace_discovery_api.DiscoveredImporter {
        if (self.workspace_discovery.getCached(target_path)) |cached| return cached;
        var scope = self.allocationScope(.cache_build);
        defer scope.deinit();
        return try self.workspace_discovery.discoverImportersForTargetPath(
            &self.docs,
            self.workspaceRootPaths(),
            target_path,
            max_cold_file_bytes,
        );
    }

    pub fn cachedDiscoveredImportersForTargetPath(self: *Handler, target_path: []const u8) ?[]const workspace_discovery_api.DiscoveredImporter {
        return self.workspace_discovery.getCached(target_path);
    }

    pub fn coldImportedMemberWorkspaceEntry(self: *Handler, importer: workspace_discovery_api.DiscoveredImporter) !?*const workspace_index_api.FileEntry {
        const features: workspace_index_api.FeatureSet = .{ .symbols = false, .imported_members = true };
        if (!(try self.ensureColdImportedMemberIndex(importer))) return null;

        return try self.docs.workspaceEntryForUri(importer.uri, features, self.workspaceRootPaths(), &self.phase_counters);
    }

    pub fn coldImportedCallWorkspaceEntry(self: *Handler, importer: workspace_discovery_api.DiscoveredImporter) !?*const workspace_index_api.FileEntry {
        const features: workspace_index_api.FeatureSet = .{ .call_edges = true, .imported_members = true };
        if (!(try self.ensureColdImportedMemberIndex(importer))) return null;

        return try self.docs.workspaceEntryForUri(importer.uri, features, self.workspaceRootPaths(), &self.phase_counters);
    }

    pub fn ensureImportedMemberIndexForUri(self: *Handler, uri: []const u8) !bool {
        if (self.docs.importedMemberIndexForUri(uri) != null) return true;

        const resolution = (try self.docs.importResolutionForUri(uri, self.workspaceRootPaths(), &self.phase_counters)) orelse return false;

        if (resolution.imports.len == 0) return false;
        try self.docs.rebuildImportedMemberIndex(uri, resolution.imports, &self.phase_counters);
        return self.docs.importedMemberIndexForUri(uri) != null;
    }

    pub fn hasImportedOutgoingMembers(self: *Handler, uri: []const u8) !bool {
        if (self.docs.importedMemberIndexForUri(uri) != null) return true;
        const state = self.docs.documentVersionStateForUri(uri) orelse return false;
        return state.is_cold and try self.ensureImportedMemberIndexForUri(uri);
    }

    fn ensureColdImportedMemberIndex(self: *Handler, importer: workspace_discovery_api.DiscoveredImporter) !bool {
        try self.ensureColdDocumentForPath(importer.uri, importer.normalized_path);
        if (self.docs.sourceForUri(importer.uri) == null) return false;

        if (self.docs.importedMemberIndexForUri(importer.uri) == null) {
            const resolution = (try self.docs.importResolutionForUri(importer.uri, self.workspaceRootPaths(), &self.phase_counters)) orelse return false;

            try self.docs.rebuildImportedMemberIndex(importer.uri, resolution.imports, &self.phase_counters);
        }

        return self.docs.importedMemberIndexForUri(importer.uri) != null;
    }

    pub fn ensureColdDocumentForPath(self: *Handler, uri: []const u8, normalized_path: []const u8) !void {
        if (self.docs.isOpenDocument(uri)) return;

        switch (try self.docs.ensureFreshColdDocument(uri)) {
            .fresh => return,
            .refreshed => {
                self.workspace_discovery.clear();
                return;
            },
            .stale_removed => self.workspace_discovery.clear(),
            .missing => {},
        }

        var temp_scope = self.allocationScope(.temp_analysis);
        defer temp_scope.deinit();
        const loaded = std.fs.cwd().readFileAlloc(self.allocator, normalized_path, max_cold_file_bytes) catch |err| switch (err) {
            error.OutOfMemory => return error.OutOfMemory,
            else => return,
        };
        defer self.allocator.free(loaded);

        try self.docs.putColdDocument(uri, normalized_path, loaded);
    }
};

fn addSat(a: usize, b: usize) usize {
    return std.math.add(usize, a, b) catch std.math.maxInt(usize);
}

fn mulSat(a: usize, b: usize) usize {
    return std.math.mul(usize, a, b) catch std.math.maxInt(usize);
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
