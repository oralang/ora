const std = @import("std");
const lsp = @import("lsp");
const ora_root = @import("ora_root");

const frontend = ora_root.lsp.frontend;
const references_api = ora_root.lsp.references;

const range_converters = @import("range_converters.zig");
const references_response = @import("references_response.zig");
const response_stats = @import("response_stats.zig");
const uri_ranges = @import("uri_ranges.zig");

const Allocator = std.mem.Allocator;
const types = lsp.types;

pub fn references(
    server: anytype,
    arena: Allocator,
    params: types.ReferenceParams,
) !lsp.ResultType("textDocument/references") {
    const uri = params.textDocument.uri;
    const position: frontend.Position = .{
        .line = params.position.line,
        .character = params.position.character,
    };

    const include_declaration = if (params.context.includeDeclaration) true else false;
    const workspace_entry = (try server.docs.workspaceEntryForUri(uri, .{ .occurrences = true }, server.workspaceRootPaths(), &server.phase_counters)) orelse return null;
    const occurrence_index = workspace_entry.occurrenceIndex();
    const target = occurrence_index.occurrenceAt(position) orelse return null;

    var response_scope = server.responseScope();
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
    var range_converter = range_converters.OpenDocument(@TypeOf(server)){
        .arena = arena,
        .handler = server,
        .encoding = server.position_encoding,
    };
    try references_response.appendLocations(arena, &all_locations, uri, same_file_refs, &range_converter);
    try appendImportedLocations(server, arena, &all_locations, uri, target.name, &range_converter);

    const result = try all_locations.toOwnedSlice(arena);
    server.response_counters.recordItems(.location, types.Location, result.len);
    server.response_counters.recordStringBytes(.location, response_stats.locationUriBytes(result));
    return result;
}

fn appendImportedLocations(
    server: anytype,
    arena: Allocator,
    locations: *std.ArrayList(types.Location),
    target_uri: []const u8,
    target_name: []const u8,
    range_converter: anytype,
) !void {
    const borrowed_target_path = server.borrowedNormalizedPathForUri(target_uri);
    const owned_target_path = if (borrowed_target_path == null) try server.normalizedPathForUri(target_uri) else null;
    defer if (owned_target_path) |path| server.allocator.free(path);
    const target_path = borrowed_target_path orelse owned_target_path orelse return;

    for (server.dependencies.directImporters(target_path)) |other_uri| {
        if (std.mem.eql(u8, other_uri, target_uri)) continue;
        if (!server.docs.isOpenDocument(other_uri)) continue;
        const workspace_entry = (try server.docs.workspaceEntryForUri(other_uri, .{ .symbols = false, .imported_members = true }, server.workspaceRootPaths(), &server.phase_counters)) orelse continue;
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

    const cold_importers = try server.discoveredImportersForTargetPath(target_path);
    for (cold_importers) |importer| {
        if (std.mem.eql(u8, importer.uri, target_uri)) continue;
        if (server.docs.isOpenDocument(importer.uri)) continue;

        const workspace_entry = (try server.coldImportedMemberWorkspaceEntry(importer)) orelse continue;
        const source = server.docs.sourceForUri(importer.uri) orelse continue;
        const imported_member_index = workspace_entry.importedMemberIndex();
        const indexed_source = uri_ranges.IndexedSource{ .source = source, .line_index = &workspace_entry.line_index };
        for (imported_member_index.occurrences) |occurrence| {
            if (!std.mem.eql(u8, occurrence.imported_path, target_path)) continue;
            if (!std.mem.eql(u8, occurrence.member_name, target_name)) continue;
            try locations.append(arena, .{
                .uri = importer.uri,
                .range = uri_ranges.byteRangeToLsp(indexed_source, server.position_encoding, occurrence.range),
            });
        }
    }
}
