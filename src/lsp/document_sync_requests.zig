const std = @import("std");
const lsp = @import("lsp");
const ora_root = @import("ora_root");

const diagnostics_api = ora_root.lsp.diagnostics;
const text_edits = ora_root.lsp.text_edits;
const workspace = ora_root.lsp.workspace;

const diagnostics_response = @import("diagnostics_response.zig");
const protocol_helpers = @import("protocol_helpers.zig");

const Allocator = std.mem.Allocator;
const types = lsp.types;

pub fn didOpen(
    server: anytype,
    arena: Allocator,
    notification: types.DidOpenTextDocumentParams,
) !void {
    const uri = notification.textDocument.uri;
    const text = notification.textDocument.text;
    _ = server.diagnostics.clear(uri);
    try server.docs.put(uri, text, notification.textDocument.version);
    try updateDocumentDependencies(server, uri, text, true);
    try publishDiagnostics(server, arena, uri, text, .full);

    if (server.dependencies.getPathForUri(uri)) |changed_path| {
        try publishDependentsDiagnostics(server, arena, changed_path, uri);
    }
}

pub fn didChange(
    server: anytype,
    arena: Allocator,
    notification: types.DidChangeTextDocumentParams,
) !void {
    const uri = notification.textDocument.uri;
    const changes = notification.contentChanges;
    if (changes.len == 0) return;

    if (server.docs.isStaleOpenVersion(uri, notification.textDocument.version)) {
        server.stale_document_change_skips = addSat(server.stale_document_change_skips, 1);
        return;
    }

    const current = server.docs.sourceForUri(uri) orelse return;
    const edit_changes = try server.allocator.alloc(text_edits.Change, changes.len);
    defer server.allocator.free(edit_changes);

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

    const updated = text_edits.applyChangesAllocWithEncoding(server.allocator, current, edit_changes, server.position_encoding) catch |err| switch (err) {
        error.InvalidRange => {
            if (protocol_helpers.lastFullText(changes)) |full_text| {
                try server.docs.put(uri, full_text, notification.textDocument.version);
                try updateDocumentDependencies(server, uri, full_text, false);
                try scheduleFullDiagnosticsAfterEdit(server, uri);
                try publishDiagnostics(server, arena, uri, full_text, .fast);
                server.edit_diagnostic_fast_publishes = addSat(server.edit_diagnostic_fast_publishes, 1);
                server.edit_diagnostic_full_skips = addSat(server.edit_diagnostic_full_skips, 1);
                try skipDependentDiagnosticsForEdit(server, uri);
            }
            return;
        },
        else => return err,
    };
    defer server.allocator.free(updated);

    try server.docs.put(uri, updated, notification.textDocument.version);
    try updateDocumentDependencies(server, uri, updated, false);
    try scheduleFullDiagnosticsAfterEdit(server, uri);
    try publishDiagnostics(server, arena, uri, updated, .fast);
    server.edit_diagnostic_fast_publishes = addSat(server.edit_diagnostic_fast_publishes, 1);
    server.edit_diagnostic_full_skips = addSat(server.edit_diagnostic_full_skips, 1);
    try skipDependentDiagnosticsForEdit(server, uri);
}

pub fn didClose(
    server: anytype,
    arena: Allocator,
    notification: types.DidCloseTextDocumentParams,
) !void {
    const uri = notification.textDocument.uri;

    const removed_path = try server.dependencies.remove(uri);
    defer if (removed_path) |path| server.allocator.free(path);

    server.docs.remove(uri);
    _ = server.diagnostics.clear(uri);
    try publishDiagnostics(server, arena, uri, "", .full);

    if (removed_path) |path| {
        try publishDependentsDiagnostics(server, arena, path, null);
    }
}

pub fn didSave(
    server: anytype,
    arena: Allocator,
    notification: types.DidSaveTextDocumentParams,
) !void {
    const uri = notification.textDocument.uri;
    const source = server.docs.sourceForUri(uri) orelse return;
    try updateDocumentDependencies(server, uri, source, true);
    try publishDiagnostics(server, arena, uri, source, .full);
    _ = server.diagnostics.flush(uri);

    if (server.dependencies.getPathForUri(uri)) |changed_path| {
        try publishDependentsDiagnostics(server, arena, changed_path, uri);
    }
}

fn scheduleFullDiagnosticsAfterEdit(server: anytype, uri: []const u8) !void {
    const state = server.docs.documentVersionStateForUri(uri) orelse return;
    try server.diagnostics.schedule(uri, .{
        .version = state.version,
        .generation = state.generation,
    });
}

fn updateDocumentDependencies(server: anytype, uri: []const u8, source: []const u8, eager_indexes: bool) !void {
    var temp_scope = server.tempAnalysisScope();
    defer temp_scope.deinit();

    const maybe_doc_path = try workspace.fileUriToPathAlloc(server.allocator, uri);
    const normalized_doc_path = if (maybe_doc_path) |doc_path| blk: {
        defer server.allocator.free(doc_path);
        break :blk try workspace.normalizePathAlloc(server.allocator, doc_path);
    } else null;
    defer if (normalized_doc_path) |path| server.allocator.free(path);

    const import_resolution = (try server.docs.importResolutionForUri(uri, server.workspaceRootPaths(), &server.phase_counters)) orelse return;

    if (eager_indexes) {
        try server.docs.rebuildOccurrenceIndex(uri, source, &server.phase_counters);
        try server.docs.rebuildImportedMemberIndex(uri, import_resolution.imports, &server.phase_counters);
    }

    const import_paths = try server.allocator.alloc([]const u8, import_resolution.imports.len);
    defer server.allocator.free(import_paths);
    for (import_resolution.imports, 0..) |item, i| {
        import_paths[i] = item.resolved_path;
    }

    try server.dependencies.upsert(uri, normalized_doc_path, import_paths);
}

fn skipDependentDiagnosticsForEdit(server: anytype, uri: []const u8) !void {
    const changed_path = server.dependencies.getPathForUri(uri) orelse return;
    const invalidated = try invalidateDependentsForChangedPath(server, changed_path, uri);
    server.dependent_diagnostic_publish_skips = addSat(
        server.dependent_diagnostic_publish_skips,
        if (invalidated == 0) 1 else invalidated,
    );
}

fn publishDependentsDiagnostics(server: anytype, arena: Allocator, changed_path: []const u8, exclude_uri: ?[]const u8) !void {
    server.dependent_diagnostic_publish_runs = addSat(server.dependent_diagnostic_publish_runs, 1);
    const dependents = try server.dependencies.collectDependents(server.allocator, changed_path);
    defer server.allocator.free(dependents);

    for (dependents) |dependent_uri| {
        if (exclude_uri) |excluded| {
            if (std.mem.eql(u8, dependent_uri, excluded)) {
                server.dependent_diagnostic_publish_skips = addSat(server.dependent_diagnostic_publish_skips, 1);
                continue;
            }
        }
        server.docs.invalidateImportDependentCaches(dependent_uri);
        const dependent_source = server.docs.sourceForUri(dependent_uri) orelse {
            server.dependent_diagnostic_publish_skips = addSat(server.dependent_diagnostic_publish_skips, 1);
            continue;
        };
        try publishDiagnostics(server, arena, dependent_uri, dependent_source, .full);
        server.dependent_diagnostic_published_documents = addSat(server.dependent_diagnostic_published_documents, 1);
    }
}

fn invalidateDependentsForChangedPath(server: anytype, changed_path: []const u8, exclude_uri: ?[]const u8) !usize {
    const dependents = try server.dependencies.collectDependents(server.allocator, changed_path);
    defer server.allocator.free(dependents);

    var invalidated: usize = 0;
    for (dependents) |dependent_uri| {
        if (exclude_uri) |excluded| {
            if (std.mem.eql(u8, dependent_uri, excluded)) continue;
        }
        server.docs.invalidateImportDependentCaches(dependent_uri);
        invalidated = addSat(invalidated, 1);
    }
    return invalidated;
}

fn publishDiagnostics(server: anytype, arena: Allocator, uri: []const u8, source: []const u8, depth: diagnostics_api.Depth) !void {
    const diagnostic_cache = try server.docs.diagnosticCacheForUri(uri, server.workspaceRootPaths(), depth, &server.phase_counters);
    const line_index = if (diagnostic_cache) |cache|
        if (cache.diagnostics.len == 0) null else try server.docs.lineIndexForUri(uri)
    else
        null;
    const params = try diagnostics_response.buildPublishParams(
        arena,
        uri,
        source,
        line_index,
        server.position_encoding,
        diagnostic_cache,
    );

    try server.transport.writeNotification(
        server.allocator,
        "textDocument/publishDiagnostics",
        types.PublishDiagnosticsParams,
        params,
        .{ .emit_null_optional_fields = false },
    );
}

fn addSat(a: usize, b: usize) usize {
    return std.math.add(usize, a, b) catch std.math.maxInt(usize);
}
