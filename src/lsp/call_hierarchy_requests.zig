const std = @import("std");
const lsp = @import("lsp");
const ora_root = @import("ora_root");

const frontend = ora_root.lsp.frontend;
const references_api = ora_root.lsp.references;
const workspace = ora_root.lsp.workspace;
const workspace_index = ora_root.lsp.workspace_index;

const call_hierarchy_calls_response = @import("call_hierarchy_calls.zig");
const call_hierarchy_prepare_response = @import("call_hierarchy_prepare.zig");
const protocol_ranges = @import("protocol_ranges.zig");
const range_converters = @import("range_converters.zig");
const response_stats = @import("response_stats.zig");

const Allocator = std.mem.Allocator;
const types = lsp.types;

pub fn prepare(
    server: anytype,
    arena: Allocator,
    params: types.CallHierarchyPrepareParams,
) !lsp.ResultType("textDocument/prepareCallHierarchy") {
    const uri = params.textDocument.uri;
    if (server.docs.callHierarchyPrepareCacheForUri(uri, params.position, server.position_encoding)) |cached| {
        server.response_counters.recordItems(.call_hierarchy, types.CallHierarchyItem, cached.items.len);
        server.response_counters.recordStringBytes(.call_hierarchy, cached.string_bytes);
        return cached.items;
    }

    const source = server.docs.sourceForUri(uri) orelse return null;
    const line_index = (try server.docs.lineIndexForUri(uri)) orelse return null;
    const index = (try server.docs.semanticIndexForUri(uri, &server.phase_counters)) orelse return null;

    var response_scope = server.responseScope();
    defer response_scope.deinit();
    const result = (try call_hierarchy_prepare_response.build(
        arena,
        uri,
        source,
        line_index,
        server.position_encoding,
        index,
        params.position,
    )) orelse return null;
    const string_bytes = response_stats.callHierarchyItemStringBytes(result);
    const cached = (try server.docs.cacheCallHierarchyPrepareForUri(
        uri,
        params.position,
        server.position_encoding,
        result,
        string_bytes,
    )) orelse {
        server.response_counters.recordItems(.call_hierarchy, types.CallHierarchyItem, result.len);
        server.response_counters.recordStringBytes(.call_hierarchy, string_bytes);
        return result;
    };
    server.response_counters.recordItems(.call_hierarchy, types.CallHierarchyItem, cached.items.len);
    server.response_counters.recordStringBytes(.call_hierarchy, cached.string_bytes);
    return cached.items;
}

pub fn incoming(
    server: anytype,
    arena: Allocator,
    params: types.CallHierarchyIncomingCallsParams,
) !lsp.ResultType("callHierarchy/incomingCalls") {
    const target_name = params.item.name;
    const dependency_generation = server.dependencies.generation();
    if (server.docs.incomingCallCacheForUri(
        params.item.uri,
        target_name,
        params.item.range,
        server.position_encoding,
        dependency_generation,
    )) |cached| {
        server.response_counters.recordItems(.call_hierarchy, types.CallHierarchyIncomingCall, cached.calls.len);
        server.response_counters.recordStringBytes(.call_hierarchy, cached.string_bytes);
        return cached.calls;
    }

    var response_scope = server.responseScope();
    defer response_scope.deinit();
    var calls = std.ArrayList(types.CallHierarchyIncomingCall).empty;
    try server.docs.ensureIncomingCallTargetIndex(server.workspaceRootPaths(), &server.phase_counters);
    var range_converter = range_converters.OpenDocument(@TypeOf(server)){
        .arena = arena,
        .handler = server,
        .encoding = server.position_encoding,
    };

    if (server.docs.incomingCallTargetUris(target_name)) |candidate_uris| {
        for (candidate_uris) |doc_uri| {
            const workspace_entry = (try server.docs.workspaceEntryForUri(doc_uri, .calls, server.workspaceRootPaths(), &server.phase_counters)) orelse continue;
            try call_hierarchy_calls_response.appendIncomingMatches(arena, &calls, doc_uri, workspace_entry, target_name, &range_converter);
        }
    }
    try appendColdImportedIncomingCalls(
        server,
        arena,
        &calls,
        params.item.uri,
        target_name,
        isImportableKind(params.item.kind),
        &range_converter,
    );

    if (calls.items.len == 0) return null;
    const result = try calls.toOwnedSlice(arena);
    const string_bytes = response_stats.incomingCallStringBytes(result);
    const cached = (try server.docs.cacheIncomingCallsForUri(
        params.item.uri,
        target_name,
        params.item.range,
        server.position_encoding,
        dependency_generation,
        result,
        string_bytes,
    )) orelse {
        server.response_counters.recordItems(.call_hierarchy, types.CallHierarchyIncomingCall, result.len);
        server.response_counters.recordStringBytes(.call_hierarchy, string_bytes);
        return result;
    };
    server.response_counters.recordItems(.call_hierarchy, types.CallHierarchyIncomingCall, cached.calls.len);
    server.response_counters.recordStringBytes(.call_hierarchy, cached.string_bytes);
    return cached.calls;
}

pub fn outgoing(
    server: anytype,
    arena: Allocator,
    params: types.CallHierarchyOutgoingCallsParams,
) !lsp.ResultType("callHierarchy/outgoingCalls") {
    const caller_uri = params.item.uri;
    const dependency_generation = server.dependencies.generation();
    if (server.docs.outgoingCallCacheForUri(caller_uri, params.item.range, server.position_encoding, dependency_generation)) |cached| {
        server.response_counters.recordItems(.call_hierarchy, types.CallHierarchyOutgoingCall, cached.calls.len);
        server.response_counters.recordStringBytes(.call_hierarchy, cached.string_bytes);
        return cached.calls;
    }

    const caller_source = server.docs.sourceForUri(caller_uri) orelse return null;
    const line_index = (try server.docs.lineIndexForUri(caller_uri)) orelse return null;
    const caller_state = server.docs.documentVersionStateForUri(caller_uri) orelse return null;
    const feature_set: workspace_index.FeatureSet = if (caller_state.is_cold or server.docs.importedMemberIndexForUri(caller_uri) != null)
        .{ .call_edges = true, .imported_members = true }
    else
        .calls;
    const workspace_entry = (try server.docs.workspaceEntryForUri(caller_uri, feature_set, server.workspaceRootPaths(), &server.phase_counters)) orelse return null;

    const caller_range = protocol_ranges.lspRangeToByte(caller_source, line_index, server.position_encoding, params.item.range) orelse return null;
    var range_converter = range_converters.OpenDocument(@TypeOf(server)){
        .arena = arena,
        .handler = server,
        .encoding = server.position_encoding,
    };

    const ImportedOutgoingTargetResolver = struct {
        handler: @TypeOf(server),
        arena: Allocator,

        pub fn resolve(
            resolver: *@This(),
            occurrence: references_api.ImportedMemberOccurrence,
        ) !?call_hierarchy_calls_response.ResolvedTarget {
            const target_uri = resolver.handler.dependencies.uriForPath(occurrence.imported_path) orelse
                try workspace.pathToFileUri(resolver.arena, occurrence.imported_path);
            try resolver.handler.ensureColdDocumentForPath(target_uri, occurrence.imported_path);
            const target_entry = (try resolver.handler.docs.workspaceEntryForUri(target_uri, .symbols_only, resolver.handler.workspaceRootPaths(), &resolver.handler.phase_counters)) orelse return null;
            const symbol = target_entry.callableSymbolNamed(occurrence.member_name) orelse return null;
            return .{ .uri = target_uri, .symbol = symbol.* };
        }
    };
    var imported_resolver = ImportedOutgoingTargetResolver{ .handler = server, .arena = arena };
    var response_scope = server.responseScope();
    defer response_scope.deinit();
    const result = (try call_hierarchy_calls_response.outgoingCallsWithResolver(
        arena,
        caller_uri,
        workspace_entry,
        caller_range,
        &range_converter,
        &imported_resolver,
    )) orelse return null;
    const string_bytes = response_stats.outgoingCallStringBytes(result);
    const cached = (try server.docs.cacheOutgoingCallsForUri(
        caller_uri,
        params.item.range,
        server.position_encoding,
        dependency_generation,
        result,
        string_bytes,
    )) orelse {
        server.response_counters.recordItems(.call_hierarchy, types.CallHierarchyOutgoingCall, result.len);
        server.response_counters.recordStringBytes(.call_hierarchy, string_bytes);
        return result;
    };
    server.response_counters.recordItems(.call_hierarchy, types.CallHierarchyOutgoingCall, cached.calls.len);
    server.response_counters.recordStringBytes(.call_hierarchy, cached.string_bytes);
    return cached.calls;
}

fn appendColdImportedIncomingCalls(
    server: anytype,
    arena: Allocator,
    calls: *std.ArrayList(types.CallHierarchyIncomingCall),
    target_uri: []const u8,
    target_name: []const u8,
    allow_discovery: bool,
    range_converter: anytype,
) !void {
    const borrowed_target_path = server.borrowedNormalizedPathForUri(target_uri);
    const owned_target_path = if (borrowed_target_path == null) try server.normalizedPathForUri(target_uri) else null;
    defer if (owned_target_path) |path| server.allocator.free(path);
    const target_path = borrowed_target_path orelse owned_target_path orelse return;

    const cold_importers = if (server.cachedDiscoveredImportersForTargetPath(target_path)) |cached|
        cached
    else if (allow_discovery)
        try server.discoveredImportersForTargetPath(target_path)
    else
        return;
    for (cold_importers) |importer| {
        if (std.mem.eql(u8, importer.uri, target_uri)) continue;
        if (server.docs.isOpenDocument(importer.uri)) continue;

        const workspace_entry = (try server.coldImportedCallWorkspaceEntry(importer)) orelse continue;
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

fn isImportableKind(kind: types.SymbolKind) bool {
    return kind == .Function or kind == .Method;
}
