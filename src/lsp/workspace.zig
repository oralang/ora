const std = @import("std");
const lexer = @import("ora_lexer");
const parser = @import("../parser.zig");
const ast = @import("ora_ast");
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
    span: ast.SourceSpan,
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
                .range = spanToRange(import_ref.span),
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
                .range = spanToRange(import_ref.span),
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
                .range = spanToRange(import_ref.span),
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

    const previous_parser_stderr = parser.diagnostics.enable_stderr_diagnostics;
    parser.diagnostics.enable_stderr_diagnostics = false;
    defer parser.diagnostics.enable_stderr_diagnostics = previous_parser_stderr;

    var parse_result = parser.parseRaw(allocator, tokens) catch {
        return try collected.toOwnedSlice(allocator);
    };
    defer parse_result.arena.deinit();

    for (parse_result.nodes) |node| {
        if (node != .Import) continue;
        const specifier = try allocator.dupe(u8, node.Import.path);
        errdefer allocator.free(specifier);
        const alias = if (node.Import.alias) |a| try allocator.dupe(u8, a) else null;
        try collected.append(allocator, .{
            .specifier = specifier,
            .alias = alias,
            .span = node.Import.span,
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

fn spanToRange(span: ast.SourceSpan) frontend.Range {
    const start_line = if (span.line > 0) span.line - 1 else 0;
    const start_char = if (span.column > 0) span.column - 1 else 0;
    var end_char = start_char + span.length;
    if (end_char < start_char) end_char = start_char;

    return .{
        .start = .{ .line = start_line, .character = start_char },
        .end = .{ .line = start_line, .character = end_char },
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
