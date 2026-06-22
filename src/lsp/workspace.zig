const std = @import("std");
const lexer = @import("ora_lexer");
const compiler = @import("../compiler.zig");
const frontend = @import("frontend.zig");
const token_cache = @import("token_cache.zig");

const Allocator = std.mem.Allocator;

pub const ResolveOptions = struct {
    workspace_roots: []const []const u8 = &.{},
};

pub const ImportResolutionDiagnostic = struct {
    range: frontend.Range,
    message: []const u8,
};

pub const ResolvedImport = struct {
    specifier: []const u8,
    alias: ?[]const u8,
    resolved_path: []const u8,
    specifier_range: ?compiler.TextRange = null,
};

pub const ResolutionResult = struct {
    diagnostics: []ImportResolutionDiagnostic,
    imports: []ResolvedImport,

    pub fn deinit(self: *ResolutionResult, allocator: Allocator) void {
        for (self.diagnostics) |diagnostic| {
            allocator.free(diagnostic.message);
        }
        allocator.free(self.diagnostics);

        for (self.imports) |item| {
            allocator.free(item.specifier);
            if (item.alias) |a| allocator.free(a);
            allocator.free(item.resolved_path);
        }
        allocator.free(self.imports);
    }
};

const ImportRef = struct {
    specifier: []const u8,
    alias: ?[]const u8,
    range: compiler.TextRange,
    specifier_range: ?compiler.TextRange = null,
};

pub fn pathToFileUri(allocator: Allocator, path: []const u8) ![]u8 {
    var uri = std.Io.Writer.Allocating.init(allocator);
    errdefer uri.deinit();
    const writer = &uri.writer;
    try writer.writeAll("file://");
    for (path) |byte| {
        if (std.ascii.isAlphanumeric(byte) or byte == '/' or byte == '.' or byte == '-' or byte == '_') {
            try writer.writeByte(byte);
        } else {
            try writer.print("%{X:0>2}", .{byte});
        }
    }
    return uri.toOwnedSlice();
}

pub fn fileUriToPathAlloc(allocator: Allocator, uri: []const u8) !?[]u8 {
    var parsed = std.Uri.parse(uri) catch return null;
    if (!std.mem.eql(u8, parsed.scheme, "file")) return null;

    const decode_buffer = try allocator.alloc(u8, uri.len);
    defer allocator.free(decode_buffer);
    const raw_path = try parsed.path.toRaw(decode_buffer);
    if (raw_path.len == 0) return null;

    if (!std.fs.path.isAbsolute(raw_path)) return null;
    return try allocator.dupe(u8, raw_path);
}

pub fn normalizePathAlloc(allocator: Allocator, path: []const u8) ![]u8 {
    const io = std.Io.Threaded.global_single_threaded.io();
    const absolute = if (std.fs.path.isAbsolute(path))
        try std.fs.path.resolve(allocator, &.{path})
    else blk: {
        const cwd = try std.process.currentPathAlloc(io, allocator);
        defer allocator.free(cwd);
        break :blk try std.fs.path.resolve(allocator, &.{ cwd, path });
    };
    errdefer allocator.free(absolute);

    const canonical_z = std.Io.Dir.realPathFileAbsoluteAlloc(io, absolute, allocator) catch {
        return absolute;
    };
    defer allocator.free(canonical_z);
    const canonical = try allocator.dupe(u8, canonical_z);
    allocator.free(absolute);
    return canonical;
}

pub fn resolveDocumentImports(
    allocator: Allocator,
    uri: []const u8,
    source: []const u8,
    options: ResolveOptions,
) !ResolutionResult {
    const import_refs = try collectImports(allocator, source);
    defer freeImportRefs(allocator, import_refs);
    return resolveDocumentImportRefs(allocator, uri, source, options, import_refs);
}

pub fn resolveDocumentImportsFromTokens(
    allocator: Allocator,
    uri: []const u8,
    source: []const u8,
    tokens: []const token_cache.Token,
    options: ResolveOptions,
) !ResolutionResult {
    const import_refs = try collectImportsFromCachedTokens(allocator, source, tokens);
    defer freeImportRefs(allocator, import_refs);
    return resolveDocumentImportRefs(allocator, uri, source, options, import_refs);
}

fn resolveDocumentImportRefs(
    allocator: Allocator,
    uri: []const u8,
    source: []const u8,
    options: ResolveOptions,
    import_refs: []const ImportRef,
) !ResolutionResult {
    var diagnostics = std.ArrayList(ImportResolutionDiagnostic).empty;
    errdefer {
        for (diagnostics.items) |diagnostic| {
            allocator.free(diagnostic.message);
        }
        diagnostics.deinit(allocator);
    }

    var imports = std.ArrayList(ResolvedImport).empty;
    errdefer {
        for (imports.items) |item| {
            allocator.free(item.specifier);
            if (item.alias) |a| allocator.free(a);
            allocator.free(item.resolved_path);
        }
        imports.deinit(allocator);
    }

    const maybe_doc_path = try fileUriToPathAlloc(allocator, uri);
    if (maybe_doc_path == null) {
        return .{
            .diagnostics = try diagnostics.toOwnedSlice(allocator),
            .imports = try imports.toOwnedSlice(allocator),
        };
    }

    const doc_path = maybe_doc_path.?;
    defer allocator.free(doc_path);

    const normalized_doc_path = try normalizePathAlloc(allocator, doc_path);
    defer allocator.free(normalized_doc_path);

    for (import_refs) |import_ref| {
        if (!isRelativeSpecifier(import_ref.specifier)) continue;

        if (!std.mem.endsWith(u8, import_ref.specifier, ".ora")) {
            const message = try std.fmt.allocPrint(
                allocator,
                "Relative import must include '.ora' extension: '{s}'",
                .{import_ref.specifier},
            );
            try diagnostics.append(allocator, .{
                .range = try spanToRange(allocator, source, import_ref.range),
                .message = message,
            });
            continue;
        }

        const maybe_resolved_path = resolveRelativeImportPathAlloc(allocator, normalized_doc_path, import_ref.specifier) catch null;
        if (maybe_resolved_path == null) {
            const message = try std.fmt.allocPrint(
                allocator,
                "Import target not found: '{s}'",
                .{import_ref.specifier},
            );
            try diagnostics.append(allocator, .{
                .range = try spanToRange(allocator, source, import_ref.range),
                .message = message,
            });
            continue;
        }

        const resolved_path = maybe_resolved_path.?;

        const allowed = try isAllowedInWorkspace(allocator, resolved_path, options.workspace_roots);
        if (!allowed) {
            const message = try std.fmt.allocPrint(
                allocator,
                "Relative import escapes workspace roots: '{s}'",
                .{import_ref.specifier},
            );
            try diagnostics.append(allocator, .{
                .range = try spanToRange(allocator, source, import_ref.range),
                .message = message,
            });
            allocator.free(resolved_path);
            continue;
        }

        const alias_copy = if (import_ref.alias) |a| try allocator.dupe(u8, a) else null;
        errdefer if (alias_copy) |a| allocator.free(a);

        try imports.append(allocator, .{
            .specifier = try allocator.dupe(u8, import_ref.specifier),
            .alias = alias_copy,
            .resolved_path = resolved_path,
            .specifier_range = import_ref.specifier_range,
        });
    }

    return .{
        .diagnostics = try diagnostics.toOwnedSlice(allocator),
        .imports = try imports.toOwnedSlice(allocator),
    };
}

pub fn sourceImportsTargetPath(
    allocator: Allocator,
    uri: []const u8,
    normalized_doc_path: []const u8,
    source: []const u8,
    options: ResolveOptions,
    target_path: []const u8,
) !bool {
    switch (try scanSourceForTargetImport(allocator, normalized_doc_path, source, options, target_path)) {
        .match => return true,
        .no_match => return false,
        .needs_full_resolution => {},
    }

    var resolution = try resolveDocumentImports(allocator, uri, source, options);
    defer resolution.deinit(allocator);
    return importsPath(resolution.imports, target_path);
}

fn collectImports(allocator: Allocator, source: []const u8) ![]ImportRef {
    var collected = std.ArrayList(ImportRef).empty;
    errdefer {
        for (collected.items) |item| {
            allocator.free(item.specifier);
            if (item.alias) |a| allocator.free(a);
        }
        collected.deinit(allocator);
    }

    var lex = lexer.Lexer.initWithRecovery(allocator, source);
    defer lex.deinit();

    const tokens = lex.scanTokens() catch {
        return try collected.toOwnedSlice(allocator);
    };
    defer allocator.free(tokens);

    var index: usize = 0;
    while (index < tokens.len) : (index += 1) {
        const token = tokens[index];
        if (token.type != .Const or index + 6 >= tokens.len) continue;

        const alias_token = tokens[index + 1];
        const equal_token = tokens[index + 2];
        const at_token = tokens[index + 3];
        const import_token = tokens[index + 4];
        const left_paren_token = tokens[index + 5];
        const path_token = tokens[index + 6];

        if (alias_token.type != .Identifier or
            equal_token.type != .Equal or
            at_token.type != .At or
            import_token.type != .Import or
            left_paren_token.type != .LeftParen or
            (path_token.type != .StringLiteral and path_token.type != .RawStringLiteral))
        {
            continue;
        }

        const value_ptr = path_token.value orelse continue;
        const string_value = switch (value_ptr.*) {
            .string => |value| value,
            else => continue,
        };
        const specifier_range = specifierTextRange(source, path_token.range, string_value);
        const specifier = try allocator.dupe(u8, string_value);
        errdefer allocator.free(specifier);
        const alias = try allocator.dupe(u8, lexer.tokenLexeme(source, alias_token));

        var end_offset = path_token.range.end_offset;
        if (index + 7 < tokens.len and tokens[index + 7].type == .RightParen) {
            end_offset = tokens[index + 7].range.end_offset;
        }
        if (index + 8 < tokens.len and tokens[index + 8].type == .Semicolon) {
            end_offset = tokens[index + 8].range.end_offset;
        }

        try collected.append(allocator, .{
            .specifier = specifier,
            .alias = alias,
            .range = .{
                .start = path_token.range.start_offset,
                .end = end_offset,
            },
            .specifier_range = specifier_range,
        });
    }

    return try collected.toOwnedSlice(allocator);
}

fn collectImportsFromCachedTokens(
    allocator: Allocator,
    source: []const u8,
    tokens: []const token_cache.Token,
) ![]ImportRef {
    var collected = std.ArrayList(ImportRef).empty;
    errdefer {
        for (collected.items) |item| {
            allocator.free(item.specifier);
            if (item.alias) |a| allocator.free(a);
        }
        collected.deinit(allocator);
    }

    var index: usize = 0;
    while (index < tokens.len) : (index += 1) {
        const token = tokens[index];
        if (token.type != .Const or index + 6 >= tokens.len) continue;

        const alias_token = tokens[index + 1];
        const equal_token = tokens[index + 2];
        const at_token = tokens[index + 3];
        const import_token = tokens[index + 4];
        const left_paren_token = tokens[index + 5];
        const path_token = tokens[index + 6];

        if (alias_token.type != .Identifier or
            equal_token.type != .Equal or
            at_token.type != .At or
            import_token.type != .Import or
            left_paren_token.type != .LeftParen or
            (path_token.type != .StringLiteral and path_token.type != .RawStringLiteral))
        {
            continue;
        }

        const string_value = path_token.string_value orelse continue;
        const specifier_range = specifierTextRange(source, path_token.range, string_value);
        const specifier = try allocator.dupe(u8, string_value);
        errdefer allocator.free(specifier);
        const alias = try allocator.dupe(u8, alias_token.lexeme);

        var end_offset = path_token.range.end_offset;
        if (index + 7 < tokens.len and tokens[index + 7].type == .RightParen) {
            end_offset = tokens[index + 7].range.end_offset;
        }
        if (index + 8 < tokens.len and tokens[index + 8].type == .Semicolon) {
            end_offset = tokens[index + 8].range.end_offset;
        }

        try collected.append(allocator, .{
            .specifier = specifier,
            .alias = alias,
            .range = .{
                .start = path_token.range.start_offset,
                .end = end_offset,
            },
            .specifier_range = specifier_range,
        });
    }

    return try collected.toOwnedSlice(allocator);
}

const TargetImportScan = enum {
    no_match,
    match,
    needs_full_resolution,
};

const ParsedImportSpecifier = union(enum) {
    none,
    value: []const u8,
    needs_full_resolution,
};

fn scanSourceForTargetImport(
    allocator: Allocator,
    normalized_doc_path: []const u8,
    source: []const u8,
    options: ResolveOptions,
    target_path: []const u8,
) !TargetImportScan {
    var cursor: usize = 0;
    while (std.mem.indexOfScalarPos(u8, source, cursor, '@')) |at_index| {
        cursor = at_index + 1;

        var index = skipWhitespace(source, cursor);
        if (!startsWithKeyword(source, index, "import")) continue;
        index += "import".len;

        index = skipWhitespace(source, index);
        if (index >= source.len or source[index] != '(') continue;
        index = skipWhitespace(source, index + 1);

        switch (parseImportSpecifier(source, index)) {
            .none => continue,
            .needs_full_resolution => return .needs_full_resolution,
            .value => |specifier| {
                if (!isRelativeSpecifier(specifier)) continue;
                if (!std.mem.endsWith(u8, specifier, ".ora")) continue;

                const maybe_resolved_path = try resolveRelativeImportPathAlloc(allocator, normalized_doc_path, specifier);
                const resolved_path = maybe_resolved_path orelse continue;
                defer allocator.free(resolved_path);

                const allowed = try isAllowedInWorkspace(allocator, resolved_path, options.workspace_roots);
                if (!allowed) continue;
                if (std.mem.eql(u8, resolved_path, target_path)) return .match;
            },
        }
    }
    return .no_match;
}

fn parseImportSpecifier(source: []const u8, start: usize) ParsedImportSpecifier {
    if (start >= source.len) return .none;

    if (source[start] == 'r' and start + 1 < source.len and source[start + 1] == '"') {
        const value_start = start + 2;
        const value_end = std.mem.indexOfScalarPos(u8, source, value_start, '"') orelse return .none;
        return .{ .value = source[value_start..value_end] };
    }

    if (source[start] != '"') return .none;
    const value_start = start + 1;
    var index = value_start;
    while (index < source.len) : (index += 1) {
        switch (source[index]) {
            '\\' => return .needs_full_resolution,
            '"' => return .{ .value = source[value_start..index] },
            else => {},
        }
    }
    return .none;
}

fn skipWhitespace(source: []const u8, start: usize) usize {
    var index = start;
    while (index < source.len and std.ascii.isWhitespace(source[index])) : (index += 1) {}
    return index;
}

fn startsWithKeyword(source: []const u8, start: usize, keyword: []const u8) bool {
    if (start + keyword.len > source.len) return false;
    if (!std.mem.eql(u8, source[start .. start + keyword.len], keyword)) return false;
    const after = start + keyword.len;
    return after >= source.len or !isIdentifierByte(source[after]);
}

fn isIdentifierByte(byte: u8) bool {
    return std.ascii.isAlphanumeric(byte) or byte == '_';
}

fn freeImportRefs(allocator: Allocator, refs: []ImportRef) void {
    for (refs) |item| {
        allocator.free(item.specifier);
        if (item.alias) |a| allocator.free(a);
    }
    allocator.free(refs);
}

fn specifierTextRange(source: []const u8, token_range: lexer.SourceRange, specifier: []const u8) ?compiler.TextRange {
    if (token_range.start_offset >= source.len or token_range.end_offset > source.len) return null;
    const token_text = source[token_range.start_offset..token_range.end_offset];
    if (token_text.len == 0) return null;

    const start_offset = if (std.mem.startsWith(u8, token_text, "r\""))
        token_range.start_offset + 2
    else if (token_text[0] == '"')
        token_range.start_offset + 1
    else
        return null;
    const end_offset = std.math.add(u32, start_offset, @as(u32, @intCast(specifier.len))) catch return null;
    if (end_offset > token_range.end_offset or end_offset > source.len) return null;
    if (!std.mem.eql(u8, source[start_offset..end_offset], specifier)) return null;

    return .{ .start = start_offset, .end = end_offset };
}

fn spanToRange(allocator: Allocator, source_text: []const u8, range: compiler.TextRange) !frontend.Range {
    var sources = compiler.source.SourceStore.init(allocator);
    defer sources.deinit();

    const file_id = try sources.addFile("<lsp>", source_text);
    const start = sources.lineColumn(.{ .file_id = file_id, .range = .{ .start = range.start, .end = range.start } });
    const end = sources.lineColumn(.{ .file_id = file_id, .range = .{ .start = range.end, .end = range.end } });

    return .{
        .start = .{
            .line = if (start.line > 0) start.line - 1 else 0,
            .character = if (start.column > 0) start.column - 1 else 0,
        },
        .end = .{
            .line = if (end.line > 0) end.line - 1 else 0,
            .character = if (end.column > 0) end.column - 1 else 0,
        },
    };
}

fn isRelativeSpecifier(specifier: []const u8) bool {
    return std.mem.startsWith(u8, specifier, "./") or std.mem.startsWith(u8, specifier, "../");
}

fn importsPath(imports: []const ResolvedImport, target_path: []const u8) bool {
    for (imports) |import_item| {
        if (std.mem.eql(u8, import_item.resolved_path, target_path)) return true;
    }
    return false;
}

fn resolveRelativeImportPathAlloc(allocator: Allocator, importer_path: []const u8, specifier: []const u8) !?[]u8 {
    const importer_dir = std.fs.path.dirname(importer_path) orelse ".";
    const joined = try std.fs.path.join(allocator, &.{ importer_dir, specifier });
    defer allocator.free(joined);

    const real_z = std.Io.Dir.cwd().realPathFileAlloc(std.Io.Threaded.global_single_threaded.io(), joined, allocator) catch {
        return null;
    };
    defer allocator.free(real_z);
    return try allocator.dupe(u8, real_z);
}

fn isAllowedInWorkspace(allocator: Allocator, resolved_path: []const u8, workspace_roots: []const []const u8) !bool {
    if (workspace_roots.len == 0) return true;

    for (workspace_roots) |root| {
        const normalized_root = try normalizePathAlloc(allocator, root);
        defer allocator.free(normalized_root);
        if (isPathWithinRoot(resolved_path, normalized_root)) return true;
    }

    return false;
}

fn isPathWithinRoot(path: []const u8, root: []const u8) bool {
    if (!std.mem.startsWith(u8, path, root)) return false;
    if (path.len == root.len) return true;
    if (root.len == 0) return false;
    if (root[root.len - 1] == std.fs.path.sep) return true;
    return path[root.len] == std.fs.path.sep;
}
