const std = @import("std");
const lsp = @import("lsp");
const ora_root = @import("ora_root");

const frontend = ora_root.lsp.frontend;
const references_api = ora_root.lsp.references;
const text_edits = ora_root.lsp.text_edits;

const response_stats = @import("response_stats.zig");
const protocol_ranges = @import("protocol_ranges.zig");
const uri_ranges = @import("uri_ranges.zig");

const Allocator = std.mem.Allocator;
const types = lsp.types;

pub fn references(
    server: anytype,
    arena: Allocator,
    params: types.ReferenceParams,
) !lsp.ResultType("textDocument/references") {
    const uri = params.textDocument.uri;
    const include_declaration = if (params.context.includeDeclaration) true else false;
    const dependency_generation = server.dependencies.generation();
    if (server.docs.referencesCacheForUri(uri, params.position, server.position_encoding, include_declaration, dependency_generation)) |cached| {
        server.response_counters.recordItems(.location, types.Location, cached.locations.len);
        server.response_counters.recordStringBytes(.location, cached.string_bytes);
        return cached.locations;
    }

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
    var all_locations = std.ArrayList(types.Location){};
    errdefer all_locations.deinit(arena);

    const same_file_ref_capacity = references_api.referenceRangeCapacityHintAtOccurrenceIndex(
        occurrence_index,
        target.name,
        include_declaration,
    );
    try all_locations.ensureUnusedCapacity(arena, same_file_ref_capacity);
    var same_file_appender = SameFileReferenceAppender{
        .locations = &all_locations,
        .uri = uri,
        .source = source,
        .line_index = line_index,
        .encoding = server.position_encoding,
    };
    try references_api.appendReferenceRangesAtOccurrenceIndex(
        occurrence_index,
        target.name,
        target.definition_range,
        include_declaration,
        &same_file_appender,
        SameFileReferenceAppender.append,
    );
    _ = try appendImportedLocations(server, arena, &all_locations, uri, target.name);

    const result = try all_locations.toOwnedSlice(arena);
    const string_bytes = response_stats.locationUriBytes(result);
    const cache_dependency_generation = server.dependencies.generation();

    const cached = (try server.docs.cacheReferencesForUri(
        uri,
        params.position,
        server.position_encoding,
        include_declaration,
        cache_dependency_generation,
        result,
        string_bytes,
    )) orelse {
        server.response_counters.recordItems(.location, types.Location, result.len);
        server.response_counters.recordStringBytes(.location, string_bytes);
        return result;
    };
    server.response_counters.recordItems(.location, types.Location, cached.locations.len);
    server.response_counters.recordStringBytes(.location, cached.string_bytes);
    return cached.locations;
}

fn appendImportedLocations(
    server: anytype,
    arena: Allocator,
    locations: *std.ArrayList(types.Location),
    target_uri: []const u8,
    target_name: []const u8,
) !usize {
    const borrowed_target_path = server.borrowedNormalizedPathForUri(target_uri);
    const owned_target_path = if (borrowed_target_path == null) try server.normalizedPathForUri(target_uri) else null;
    defer if (owned_target_path) |path| server.allocator.free(path);
    const target_path = borrowed_target_path orelse owned_target_path orelse return 0;

    var added: usize = 0;
    for (server.dependencies.directImporters(target_path)) |other_uri| {
        if (std.mem.eql(u8, other_uri, target_uri)) continue;
        if (!server.docs.isOpenDocument(other_uri)) continue;
        if (try appendCachedImportedLocations(server, arena, locations, other_uri, target_path, target_name)) |count| {
            added = addSat(added, count);
            continue;
        }
        const imported_doc = (try server.importedMemberDocumentForUri(other_uri)) orelse continue;
        added = addSat(added, try appendImportedLocationsForDocument(
            server,
            arena,
            locations,
            other_uri,
            .{ .source = imported_doc.source, .line_index = imported_doc.line_index },
            imported_doc.index.occurrences,
            target_path,
            target_name,
        ));
    }

    const cold_importers = try server.discoveredImportersForTargetPath(target_path);
    for (cold_importers) |importer| {
        if (std.mem.eql(u8, importer.uri, target_uri)) continue;
        if (server.docs.isOpenDocument(importer.uri)) continue;

        try server.ensureColdDocumentForPath(importer.uri, importer.normalized_path);
        if (try appendCachedImportedLocations(server, arena, locations, importer.uri, target_path, target_name)) |count| {
            added = addSat(added, count);
            continue;
        }
        const imported_doc = (try server.coldImportedMemberDocument(importer)) orelse continue;
        added = addSat(added, try appendImportedLocationsForDocument(
            server,
            arena,
            locations,
            importer.uri,
            .{ .source = imported_doc.source, .line_index = imported_doc.line_index },
            imported_doc.index.occurrences,
            target_path,
            target_name,
        ));
    }
    return added;
}

fn appendCachedImportedLocations(
    server: anytype,
    arena: Allocator,
    locations: *std.ArrayList(types.Location),
    importer_uri: []const u8,
    target_path: []const u8,
    target_name: []const u8,
) !?usize {
    if (server.docs.importedMemberReferencesCacheForUri(
        importer_uri,
        target_path,
        target_name,
        server.position_encoding,
    )) |cached| {
        try locations.appendSlice(arena, cached.locations);
        return cached.locations.len;
    }
    return null;
}

fn appendImportedLocationsForDocument(
    server: anytype,
    arena: Allocator,
    locations: *std.ArrayList(types.Location),
    importer_uri: []const u8,
    indexed_source: uri_ranges.IndexedSource,
    occurrences: anytype,
    target_path: []const u8,
    target_name: []const u8,
) !usize {
    if (server.docs.importedMemberReferencesCacheForUri(
        importer_uri,
        target_path,
        target_name,
        server.position_encoding,
    )) |cached| {
        try locations.appendSlice(arena, cached.locations);
        return cached.locations.len;
    }

    var match_count: usize = 0;
    for (occurrences) |occurrence| {
        if (!std.mem.eql(u8, occurrence.imported_path, target_path)) continue;
        if (!std.mem.eql(u8, occurrence.member_name, target_name)) continue;
        match_count += 1;
    }

    const built = try arena.alloc(types.Location, match_count);
    var index: usize = 0;
    for (occurrences) |occurrence| {
        if (!std.mem.eql(u8, occurrence.imported_path, target_path)) continue;
        if (!std.mem.eql(u8, occurrence.member_name, target_name)) continue;
        built[index] = .{
            .uri = importer_uri,
            .range = uri_ranges.byteRangeToLsp(indexed_source, server.position_encoding, occurrence.range),
        };
        index += 1;
    }

    const string_bytes = response_stats.locationUriBytes(built);
    const cached = (try server.docs.cacheImportedMemberReferencesForUri(
        importer_uri,
        target_path,
        target_name,
        server.position_encoding,
        built,
        string_bytes,
    )) orelse return 0;
    try locations.appendSlice(arena, cached.locations);
    return cached.locations.len;
}

const SameFileReferenceAppender = struct {
    locations: *std.ArrayList(types.Location),
    uri: []const u8,
    source: []const u8,
    line_index: *const ora_root.lsp.line_index.LineIndex,
    encoding: text_edits.PositionEncoding,

    fn append(self: *@This(), range: frontend.Range) !void {
        self.locations.appendAssumeCapacity(.{
            .uri = self.uri,
            .range = uri_ranges.byteRangeToLsp(.{ .source = self.source, .line_index = self.line_index }, self.encoding, range),
        });
    }
};

fn addSat(a: usize, b: usize) usize {
    return std.math.add(usize, a, b) catch std.math.maxInt(usize);
}
