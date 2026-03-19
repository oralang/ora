const std = @import("std");
const lexer = @import("ora_lexer");
const compiler = @import("../compiler.zig");
const frontend = @import("frontend.zig");

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
};

pub fn pathToFileUri(allocator: Allocator, path: []const u8) ![]u8 {
    var uri = std.ArrayList(u8){};
    errdefer uri.deinit(allocator);
    const writer = uri.writer(allocator);
    try writer.writeAll("file://");
    for (path) |byte| {
        if (std.ascii.isAlphanumeric(byte) or byte == '/' or byte == '.' or byte == '-' or byte == '_') {
            try writer.writeByte(byte);
        } else {
            try writer.print("%{X:0>2}", .{byte});
        }
    }
    return uri.toOwnedSlice(allocator);
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
    const absolute = if (std.fs.path.isAbsolute(path))
        try std.fs.path.resolve(allocator, &.{path})
    else blk: {
        const cwd = try std.fs.cwd().realpathAlloc(allocator, ".");
        defer allocator.free(cwd);
        break :blk try std.fs.path.resolve(allocator, &.{ cwd, path });
    };
    errdefer allocator.free(absolute);

    const canonical = std.fs.cwd().realpathAlloc(allocator, absolute) catch {
        return absolute;
    };
    allocator.free(absolute);
    return canonical;
}

pub fn resolveDocumentImports(
    allocator: Allocator,
    uri: []const u8,
    source: []const u8,
    options: ResolveOptions,
) !ResolutionResult {
    var diagnostics = std.ArrayList(ImportResolutionDiagnostic){};
    errdefer {
        for (diagnostics.items) |diagnostic| {
            allocator.free(diagnostic.message);
        }
        diagnostics.deinit(allocator);
    }

    var imports = std.ArrayList(ResolvedImport){};
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

    const import_refs = try collectImports(allocator, source);
    defer freeImportRefs(allocator, import_refs);

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
        });
    }

    return .{
        .diagnostics = try diagnostics.toOwnedSlice(allocator),
        .imports = try imports.toOwnedSlice(allocator),
    };
}

fn collectImports(allocator: Allocator, source: []const u8) ![]ImportRef {
    var collected = std.ArrayList(ImportRef){};
    errdefer {
        for (collected.items) |item| {
            allocator.free(item.specifier);
            if (item.alias) |a| allocator.free(a);
        }
        collected.deinit(allocator);
    }

    var lex = try lexer.Lexer.initWithConfig(allocator, source, lexer.LexerConfig.development());
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

        const string_value = switch (path_token.value orelse continue) {
            .string => |value| value,
            else => continue,
        };
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
        });
    }

    return try collected.toOwnedSlice(allocator);
}

fn freeImportRefs(allocator: Allocator, refs: []ImportRef) void {
    for (refs) |item| {
        allocator.free(item.specifier);
        if (item.alias) |a| allocator.free(a);
    }
    allocator.free(refs);
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

fn resolveRelativeImportPathAlloc(allocator: Allocator, importer_path: []const u8, specifier: []const u8) !?[]u8 {
    const importer_dir = std.fs.path.dirname(importer_path) orelse ".";
    const joined = try std.fs.path.join(allocator, &.{ importer_dir, specifier });
    defer allocator.free(joined);

    const real = std.fs.cwd().realpathAlloc(allocator, joined) catch {
        return null;
    };
    return real;
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
