const std = @import("std");
const lsp = @import("lsp");
const ora_root = @import("ora_root");

const definition_api = ora_root.lsp.definition;
const frontend = ora_root.lsp.frontend;
const std_docs_api = ora_root.lsp.std_docs;
const workspace = ora_root.lsp.workspace;

const definition_response = @import("definition_response.zig");
const protocol_helpers = @import("protocol_helpers.zig");
const protocol_ranges = @import("protocol_ranges.zig");
const range_converters = @import("range_converters.zig");

const Allocator = std.mem.Allocator;
const types = lsp.types;

pub fn definition(
    server: anytype,
    arena: Allocator,
    params: types.DefinitionParams,
) !lsp.ResultType("textDocument/definition") {
    const uri = params.textDocument.uri;

    if (server.docs.definitionCacheForUri(uri, params.position, server.position_encoding, &server.dependencies)) |cached| {
        server.response_counters.recordItems(.definition, types.Location, 1);
        server.response_counters.recordStringBytes(.definition, cached.string_bytes);
        return .{ .Definition = .{ .Location = cached.location } };
    }

    if (try server.docs.sameFileDefinitionRangeForUri(uri, params.position, server.position_encoding, &server.phase_counters)) |range| {
        server.response_counters.recordItems(.definition, types.Location, 1);
        server.response_counters.recordStringBytes(.definition, uri.len);
        return .{ .Definition = .{ .Location = .{ .uri = uri, .range = range } } };
    }

    const source = server.docs.sourceForUri(uri) orelse return null;
    const position: frontend.Position = .{
        .line = params.position.line,
        .character = params.position.character,
    };

    const dependency_generation = server.dependencies.generation();
    if (try cachedImportedMemberDefinition(server, arena, uri, source, params.position, dependency_generation)) |location| {
        return .{ .Definition = .{ .Location = location } };
    }
    if (try stdDocsDefinition(server, arena, uri, source, params.position)) |location| {
        return .{ .Definition = .{ .Location = location } };
    }

    var temp_scope = server.tempAnalysisScope();
    defer temp_scope.deinit();
    const cross_file = try buildCrossFileContext(server, uri);
    defer server.allocator.free(cross_file.bindings);
    defer for (cross_file.bindings) |b| {
        server.allocator.free(b.alias);
        server.allocator.free(b.target_uri);
    };

    var analysis = (try server.docs.definitionAnalysisForUri(uri, &server.phase_counters)) orelse return null;
    defer analysis.deinit();
    const maybe_definition = try definition_api.definitionAtCachedCrossFile(&analysis, source, position, cross_file);
    if (maybe_definition == null) return null;

    const resolved = maybe_definition.?;
    const response_uri = resolved.uri orelse uri;
    var range_converter = range_converters.OpenDocument(@TypeOf(server)){
        .arena = arena,
        .handler = server,
        .encoding = server.position_encoding,
    };
    var response_scope = server.responseScope();
    defer response_scope.deinit();
    const location = try definition_response.build(arena, uri, resolved, &range_converter);
    server.response_counters.recordItems(.definition, types.Location, 1);
    server.response_counters.recordStringBytes(.definition, response_uri.len);
    return .{ .Definition = .{ .Location = location } };
}

fn stdDocsDefinition(
    server: anytype,
    arena: Allocator,
    uri: []const u8,
    source: []const u8,
    lsp_position: types.Position,
) !?types.Location {
    const line_index = (try server.docs.lineIndexForUri(uri)) orelse return null;
    const position = protocol_ranges.lspPositionToBytePosition(
        source,
        line_index,
        server.position_encoding,
        lsp_position,
    ) orelse frontend.Position{
        .line = lsp_position.line,
        .character = lsp_position.character,
    };
    const aliases = (try server.docs.stdImportAliasesForUri(uri, &server.phase_counters)) orelse return null;
    if (aliases.len == 0) return null;

    const std_index = try server.stdDocsIndex();
    const std_definition = std_docs_api.definitionAt(source, position, std_index, aliases) orelse return null;

    var response_scope = server.responseScope();
    defer response_scope.deinit();
    server.response_counters.recordItems(.definition, types.Location, 1);
    server.response_counters.recordStringBytes(.definition, std_definition.uri.len);
    return .{
        .uri = try arena.dupe(u8, std_definition.uri),
        .range = protocol_ranges.rawRange(std_definition.range),
    };
}

fn cachedImportedMemberDefinition(
    server: anytype,
    arena: Allocator,
    uri: []const u8,
    source: []const u8,
    lsp_position: types.Position,
    dependency_generation: u64,
) !?types.Location {
    const line_index = (try server.docs.lineIndexForUri(uri)) orelse return null;
    const access = protocol_helpers.importedMemberAccessAtLspPosition(source, line_index, server.position_encoding, lsp_position) orelse return null;
    if (server.docs.importedMemberIndexForUri(uri) == null) {
        const import_resolution = (try server.docs.importResolutionForUri(
            uri,
            server.workspaceRootPaths(),
            &server.phase_counters,
        )) orelse return null;
        try server.docs.rebuildImportedMemberIndex(uri, import_resolution.imports, &server.phase_counters);
    }
    const imported_index = server.docs.importedMemberIndexForUri(uri) orelse return null;

    for (imported_index.occurrences) |occurrence| {
        if (!std.mem.eql(u8, occurrence.alias, access.alias)) continue;
        if (!std.mem.eql(u8, occurrence.member_name, access.member_name)) continue;
        if (!protocol_helpers.frontendRangesEqual(occurrence.range, access.member_range)) continue;

        const target_uri = server.dependencies.uriForPath(occurrence.imported_path) orelse
            try workspace.pathToFileUri(arena, occurrence.imported_path);
        try server.ensureColdDocumentForPath(target_uri, occurrence.imported_path);

        const target_text = server.docs.sourceForUri(target_uri) orelse return null;
        const target_entry = (try server.docs.workspaceEntryForUri(target_uri, .symbols_only, server.workspaceRootPaths(), &server.phase_counters)) orelse return null;
        const symbol = target_entry.rootSymbolNamed(occurrence.member_name) orelse return null;
        const range = protocol_ranges.byteRangeToLspOrRaw(
            target_text,
            &target_entry.line_index,
            server.position_encoding,
            symbol.selection_range,
        );

        var response_scope = server.responseScope();
        defer response_scope.deinit();
        const cached = (try server.docs.cacheImportedDefinitionForUri(
            uri,
            lsp_position,
            server.position_encoding,
            dependency_generation,
            target_uri,
            range,
        )) orelse return null;
        server.response_counters.recordItems(.definition, types.Location, 1);
        server.response_counters.recordStringBytes(.definition, cached.string_bytes);
        return cached.location;
    }

    return null;
}

fn buildCrossFileContext(server: anytype, uri: []const u8) !definition_api.CrossFileContext {
    const import_resolution = (try server.docs.importResolutionForUri(uri, server.workspaceRootPaths(), &server.phase_counters)) orelse return .{
        .bindings = try server.allocator.alloc(definition_api.ImportBinding, 0),
    };
    var bindings = std.ArrayList(definition_api.ImportBinding){};
    errdefer {
        for (bindings.items) |b| {
            server.allocator.free(b.alias);
            server.allocator.free(b.target_uri);
        }
        bindings.deinit(server.allocator);
    }

    for (import_resolution.imports) |resolved| {
        const alias = resolved.alias orelse aliasFromSpecifier(resolved.specifier) orelse continue;
        const alias_copy = try server.allocator.dupe(u8, alias);
        errdefer server.allocator.free(alias_copy);

        const target_uri = try workspace.pathToFileUri(server.allocator, resolved.resolved_path);
        errdefer server.allocator.free(target_uri);

        try server.ensureColdDocumentForPath(target_uri, resolved.resolved_path);
        try bindings.append(server.allocator, .{
            .alias = alias_copy,
            .target_uri = target_uri,
        });
    }

    return .{ .bindings = try bindings.toOwnedSlice(server.allocator) };
}

fn aliasFromSpecifier(specifier: []const u8) ?[]const u8 {
    const basename = std.fs.path.stem(specifier);
    if (basename.len == 0) return null;
    return basename;
}
