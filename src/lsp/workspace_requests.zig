const std = @import("std");
const lsp = @import("lsp");

const cache_stats_response = @import("cache_stats_response.zig");
const cache_stats_snapshot = @import("cache_stats_snapshot.zig");
const response_stats = @import("response_stats.zig");
const workspace_symbol_response = @import("workspace_symbol_response.zig");

const Allocator = std.mem.Allocator;
const types = lsp.types;

pub fn executeCommand(
    server: anytype,
    arena: Allocator,
    params: types.ExecuteCommandParams,
) !lsp.ResultType("workspace/executeCommand") {
    if (!std.mem.eql(u8, params.command, "ora.cacheStats")) return null;

    const snapshot = cache_stats_snapshot.build(
        &server.docs,
        &server.workspace_discovery,
        &server.diagnostics,
        &server.response_counters,
        server.phase_counters,
        .{
            .dependent_diagnostic_publish_runs = server.dependent_diagnostic_publish_runs,
            .dependent_diagnostic_published_documents = server.dependent_diagnostic_published_documents,
            .dependent_diagnostic_publish_skips = server.dependent_diagnostic_publish_skips,
            .stale_document_change_skips = server.stale_document_change_skips,
            .edit_diagnostic_fast_publishes = server.edit_diagnostic_fast_publishes,
            .edit_diagnostic_full_skips = server.edit_diagnostic_full_skips,
        },
        .{
            .totals = server.allocatorStats(),
            .unscoped = server.scopedAllocatorStats(.unscoped),
            .request = server.scopedAllocatorStats(.request_protocol),
            .response = server.scopedAllocatorStats(.response),
            .cache_build = server.scopedAllocatorStats(.cache_build),
            .temp_analysis = server.scopedAllocatorStats(.temp_analysis),
        },
    );
    return .{ .object = try cache_stats_response.build(arena, snapshot) };
}

pub fn symbol(
    server: anytype,
    arena: Allocator,
    params: types.WorkspaceSymbolParams,
) !lsp.ResultType("workspace/symbol") {
    const query = params.query;
    if (server.docs.workspaceSymbolCache(query, server.position_encoding)) |cached| {
        server.response_counters.recordItems(.workspace_symbol, types.SymbolInformation, cached.items.len);
        server.response_counters.recordStringBytes(.workspace_symbol, cached.string_bytes);
        return .{ .array_of_SymbolInformation = cached.items };
    }

    var response_scope = server.responseScope();
    defer response_scope.deinit();
    var symbols = std.ArrayList(types.SymbolInformation){};
    var string_bytes: usize = 0;

    var doc_it = server.docs.docs.iterator();
    while (doc_it.next()) |entry| {
        const doc_uri = entry.key_ptr.*;
        const source = server.docs.sourceForUri(doc_uri) orelse continue;
        const line_index = (try server.docs.lineIndexForUri(doc_uri)) orelse continue;
        const semantic_index = (try server.docs.semanticIndexForUri(doc_uri, &server.phase_counters)) orelse continue;

        const stats = try workspace_symbol_response.appendOpenDocumentSymbols(
            arena,
            &symbols,
            source,
            line_index,
            server.position_encoding,
            doc_uri,
            semantic_index,
            query,
        );
        string_bytes = addSat(string_bytes, stats.string_bytes);
    }

    if (symbols.items.len == 0) return null;
    const result = try symbols.toOwnedSlice(arena);
    const cached = try server.docs.cacheWorkspaceSymbols(query, server.position_encoding, result, string_bytes);
    server.response_counters.recordItems(.workspace_symbol, types.SymbolInformation, cached.items.len);
    server.response_counters.recordStringBytes(.workspace_symbol, cached.string_bytes);
    return .{ .array_of_SymbolInformation = cached.items };
}

fn addSat(a: usize, b: usize) usize {
    return std.math.add(usize, a, b) catch std.math.maxInt(usize);
}
