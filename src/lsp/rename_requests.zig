const std = @import("std");
const lsp = @import("lsp");
const ora_root = @import("ora_root");

const references_api = ora_root.lsp.references;
const rename_api = ora_root.lsp.rename;
const text_edits = ora_root.lsp.text_edits;

const protocol_ranges = @import("protocol_ranges.zig");
const rename_response = @import("rename_response.zig");
const uri_ranges = @import("uri_ranges.zig");

const Allocator = std.mem.Allocator;
const types = lsp.types;

pub fn prepareRename(
    server: anytype,
    _: Allocator,
    params: types.PrepareRenameParams,
) !lsp.ResultType("textDocument/prepareRename") {
    const uri = params.textDocument.uri;
    const prepared = (try server.docs.prepareRenameForUri(uri, params.position, server.position_encoding)) orelse return null;

    const result: lsp.ResultType("textDocument/prepareRename") = .{ .literal_1 = .{
        .range = prepared.range,
        .placeholder = prepared.placeholder,
    } };
    server.response_counters.recordItems(.prepare_rename, types.Range, 1);
    server.response_counters.recordStringBytes(.prepare_rename, prepared.placeholder.len);
    return result;
}

pub fn rename(
    server: anytype,
    arena: Allocator,
    params: types.RenameParams,
) !lsp.ResultType("textDocument/rename") {
    const uri = params.textDocument.uri;
    const dependency_generation = server.dependencies.generation();
    if (server.docs.renameCacheForUri(uri, params.position, server.position_encoding, dependency_generation, params.newName)) |cached| {
        server.response_counters.recordItems(.text_edit, types.TextEdit, cached.edit_count);
        server.response_counters.recordStringBytes(.text_edit, cached.string_bytes);
        return .{ .changes = cached.changes };
    }
    if (!rename_api.isValidIdentifier(params.newName)) return null;

    const source = server.docs.sourceForUri(uri) orelse return null;
    const line_index = (try server.docs.lineIndexForUri(uri)) orelse return null;
    if (server.docs.occurrenceIndexForUri(uri) == null) {
        try server.docs.rebuildOccurrenceIndex(uri, source, &server.phase_counters);
    }
    const occurrence_index = server.docs.occurrenceIndexForUri(uri) orelse return null;
    const position = protocol_ranges.lspPositionToBytePosition(
        source,
        line_index,
        server.position_encoding,
        params.position,
    ) orelse return null;
    const target = occurrence_index.occurrenceAt(position) orelse return null;

    var response_scope = server.responseScope();
    defer response_scope.deinit();
    const same_file_ranges = try references_api.referencesAtOccurrenceIndex(
        arena,
        occurrence_index,
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
        .{ .source = source, .line_index = line_index },
        same_file_ranges,
        params.newName,
        server.position_encoding,
    );

    const imported_edit_count = try appendImportedChanges(server, arena, &changes, uri, target.name, params.newName);
    const edit_count = addSat(same_file_ranges.len, imported_edit_count);
    const string_bytes = mulSat(edit_count, params.newName.len);
    if (try server.docs.cacheRenameForUri(
        uri,
        params.position,
        server.position_encoding,
        dependency_generation,
        params.newName,
        changes,
        edit_count,
        string_bytes,
    )) |cached| {
        server.response_counters.recordItems(.text_edit, types.TextEdit, cached.edit_count);
        server.response_counters.recordStringBytes(.text_edit, cached.string_bytes);
        return .{ .changes = cached.changes };
    }

    server.response_counters.recordItems(.text_edit, types.TextEdit, edit_count);
    server.response_counters.recordStringBytes(.text_edit, string_bytes);

    return .{ .changes = changes };
}

fn appendImportedChanges(
    server: anytype,
    arena: Allocator,
    changes: *lsp.parser.Map(types.DocumentUri, []const types.TextEdit),
    target_uri: []const u8,
    target_name: []const u8,
    new_name: []const u8,
) !usize {
    const borrowed_target_path = server.borrowedNormalizedPathForUri(target_uri);
    const owned_target_path = if (borrowed_target_path == null) try server.normalizedPathForUri(target_uri) else null;
    defer if (owned_target_path) |path| server.allocator.free(path);
    const target_path = borrowed_target_path orelse owned_target_path orelse return 0;

    var total_edits: usize = 0;
    for (server.dependencies.directImporters(target_path)) |other_uri| {
        if (std.mem.eql(u8, other_uri, target_uri)) continue;
        if (!server.docs.isOpenDocument(other_uri)) continue;
        if (try putCachedImportedChange(server, arena, changes, other_uri, target_path, target_name, new_name)) |count| {
            total_edits = addSat(total_edits, count);
            continue;
        }
        const imported_doc = (try server.importedMemberDocumentForUri(other_uri)) orelse continue;
        total_edits = addSat(total_edits, try putImportedChange(
            server,
            arena,
            changes,
            other_uri,
            .{ .source = imported_doc.source, .line_index = imported_doc.line_index },
            imported_doc.index.occurrences,
            target_path,
            target_name,
            new_name,
            server.position_encoding,
        ));
    }

    const cold_importers = try server.discoveredImportersForTargetPath(target_path);
    for (cold_importers) |importer| {
        if (std.mem.eql(u8, importer.uri, target_uri)) continue;
        if (server.docs.isOpenDocument(importer.uri)) continue;

        try server.ensureColdDocumentForPath(importer.uri, importer.normalized_path);
        if (try putCachedImportedChange(server, arena, changes, importer.uri, target_path, target_name, new_name)) |count| {
            total_edits = addSat(total_edits, count);
            continue;
        }
        const imported_doc = (try server.coldImportedMemberDocument(importer)) orelse continue;
        total_edits = addSat(total_edits, try putImportedChange(
            server,
            arena,
            changes,
            importer.uri,
            .{ .source = imported_doc.source, .line_index = imported_doc.line_index },
            imported_doc.index.occurrences,
            target_path,
            target_name,
            new_name,
            server.position_encoding,
        ));
    }
    return total_edits;
}

fn putCachedImportedChange(
    server: anytype,
    arena: Allocator,
    changes: *lsp.parser.Map(types.DocumentUri, []const types.TextEdit),
    uri: []const u8,
    target_path: []const u8,
    target_name: []const u8,
    new_name: []const u8,
) !?usize {
    if (server.docs.importedMemberRenameCacheForUri(
        uri,
        target_path,
        target_name,
        new_name,
        server.position_encoding,
    )) |cached| {
        try changes.map.put(arena, uri, cached.edits);
        return cached.edits.len;
    }
    return null;
}

fn putImportedChange(
    server: anytype,
    arena: Allocator,
    changes: *lsp.parser.Map(types.DocumentUri, []const types.TextEdit),
    uri: []const u8,
    indexed_source: uri_ranges.IndexedSource,
    occurrences: anytype,
    target_path: []const u8,
    target_name: []const u8,
    new_name: []const u8,
    encoding: text_edits.PositionEncoding,
) !usize {
    if (server.docs.importedMemberRenameCacheForUri(
        uri,
        target_path,
        target_name,
        new_name,
        encoding,
    )) |cached| {
        try changes.map.put(arena, uri, cached.edits);
        return cached.edits.len;
    }

    var match_count: usize = 0;
    for (occurrences) |occurrence| {
        if (!std.mem.eql(u8, occurrence.imported_path, target_path)) continue;
        if (!std.mem.eql(u8, occurrence.member_name, target_name)) continue;
        match_count += 1;
    }
    if (match_count == 0) return 0;

    const edits = try arena.alloc(types.TextEdit, match_count);
    var edit_index: usize = 0;
    for (occurrences) |occurrence| {
        if (!std.mem.eql(u8, occurrence.imported_path, target_path)) continue;
        if (!std.mem.eql(u8, occurrence.member_name, target_name)) continue;
        edits[edit_index] = .{
            .range = uri_ranges.byteRangeToLsp(indexed_source, encoding, occurrence.range),
            .newText = new_name,
        };
        edit_index += 1;
    }

    const cached = (try server.docs.cacheImportedMemberRenameForUri(
        uri,
        target_path,
        target_name,
        new_name,
        encoding,
        edits,
        mulSat(edits.len, new_name.len),
    )) orelse {
        try changes.map.put(arena, uri, edits);
        return edits.len;
    };

    try changes.map.put(arena, uri, cached.edits);
    return cached.edits.len;
}

fn addSat(a: usize, b: usize) usize {
    return std.math.add(usize, a, b) catch std.math.maxInt(usize);
}

fn mulSat(a: usize, b: usize) usize {
    return std.math.mul(usize, a, b) catch std.math.maxInt(usize);
}
