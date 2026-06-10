const std = @import("std");
const lsp = @import("lsp");
const ora_root = @import("ora_root");

const frontend = ora_root.lsp.frontend;
const references_api = ora_root.lsp.references;
const rename_api = ora_root.lsp.rename;

const rename_response = @import("rename_response.zig");
const uri_ranges = @import("uri_ranges.zig");

const Allocator = std.mem.Allocator;
const types = lsp.types;

pub fn prepareRename(
    server: anytype,
    arena: Allocator,
    params: types.PrepareRenameParams,
) !lsp.ResultType("textDocument/prepareRename") {
    const uri = params.textDocument.uri;
    const position: frontend.Position = .{
        .line = params.position.line,
        .character = params.position.character,
    };

    const source = server.docs.sourceForUri(uri) orelse return null;
    const workspace_entry = (try server.docs.workspaceEntryForUri(uri, .{ .occurrences = true }, server.workspaceRootPaths(), &server.phase_counters)) orelse return null;
    const occurrence_index = workspace_entry.occurrenceIndex();
    const target = occurrence_index.occurrenceAt(position) orelse return null;

    var response_scope = server.responseScope();
    defer response_scope.deinit();
    const result = try rename_response.buildPrepare(
        arena,
        source,
        &workspace_entry.line_index,
        server.position_encoding,
        target.definition_range,
        target.name,
    );
    if (result == null) return null;
    server.response_counters.recordItems(.prepare_rename, types.Range, 1);
    server.response_counters.recordStringBytes(.prepare_rename, target.name.len);
    return result;
}

pub fn rename(
    server: anytype,
    arena: Allocator,
    params: types.RenameParams,
) !lsp.ResultType("textDocument/rename") {
    if (!rename_api.isValidIdentifier(params.newName)) return null;

    const uri = params.textDocument.uri;
    const position: frontend.Position = .{
        .line = params.position.line,
        .character = params.position.character,
    };

    const source = server.docs.sourceForUri(uri) orelse return null;
    const workspace_entry = (try server.docs.workspaceEntryForUri(uri, .{ .occurrences = true }, server.workspaceRootPaths(), &server.phase_counters)) orelse return null;
    const occurrence_index = workspace_entry.occurrenceIndex();
    const target = occurrence_index.occurrenceAt(position) orelse return null;

    var response_scope = server.responseScope();
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
        server.position_encoding,
    );

    const imported_edit_count = try appendImportedChanges(server, arena, &changes, uri, target.name, params.newName);
    const edit_count = addSat(same_file_ranges.len, imported_edit_count);
    server.response_counters.recordItems(.text_edit, types.TextEdit, edit_count);
    server.response_counters.recordStringBytes(.text_edit, mulSat(edit_count, params.newName.len));

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
        const workspace_entry = (try server.docs.workspaceEntryForUri(other_uri, .{ .symbols = false, .imported_members = true }, server.workspaceRootPaths(), &server.phase_counters)) orelse continue;
        const imported_member_index = workspace_entry.importedMemberIndex();
        total_edits = addSat(total_edits, try putImportedChange(
            server,
            arena,
            changes,
            other_uri,
            workspace_entry,
            imported_member_index.occurrences,
            target_path,
            target_name,
            new_name,
        ));
    }

    const cold_importers = try server.discoveredImportersForTargetPath(target_path);
    for (cold_importers) |importer| {
        if (std.mem.eql(u8, importer.uri, target_uri)) continue;
        if (server.docs.isOpenDocument(importer.uri)) continue;

        const workspace_entry = (try server.coldImportedMemberWorkspaceEntry(importer)) orelse continue;
        const imported_member_index = workspace_entry.importedMemberIndex();
        total_edits = addSat(total_edits, try putImportedChange(
            server,
            arena,
            changes,
            importer.uri,
            workspace_entry,
            imported_member_index.occurrences,
            target_path,
            target_name,
            new_name,
        ));
    }
    return total_edits;
}

fn putImportedChange(
    server: anytype,
    arena: Allocator,
    changes: *lsp.parser.Map(types.DocumentUri, []const types.TextEdit),
    uri: []const u8,
    workspace_entry: anytype,
    occurrences: anytype,
    target_path: []const u8,
    target_name: []const u8,
    new_name: []const u8,
) !usize {
    var match_count: usize = 0;
    for (occurrences) |occurrence| {
        if (!std.mem.eql(u8, occurrence.imported_path, target_path)) continue;
        if (!std.mem.eql(u8, occurrence.member_name, target_name)) continue;
        match_count += 1;
    }
    if (match_count == 0) return 0;

    const match_ranges = try arena.alloc(frontend.Range, match_count);
    var range_index: usize = 0;
    for (occurrences) |occurrence| {
        if (!std.mem.eql(u8, occurrence.imported_path, target_path)) continue;
        if (!std.mem.eql(u8, occurrence.member_name, target_name)) continue;
        match_ranges[range_index] = occurrence.range;
        range_index += 1;
    }

    const indexed_source: ?uri_ranges.IndexedSource = if (server.docs.sourceForUri(uri)) |source|
        .{ .source = source, .line_index = &workspace_entry.line_index }
    else
        null;
    _ = try rename_response.putChange(
        arena,
        changes,
        uri,
        indexed_source,
        match_ranges,
        new_name,
        server.position_encoding,
    );
    return match_count;
}

fn addSat(a: usize, b: usize) usize {
    return std.math.add(usize, a, b) catch std.math.maxInt(usize);
}

fn mulSat(a: usize, b: usize) usize {
    return std.math.mul(usize, a, b) catch std.math.maxInt(usize);
}
