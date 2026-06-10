const std = @import("std");

const lexer_mod = @import("ora_lexer");
const compiler = @import("../compiler.zig");
const frontend = @import("frontend.zig");
const token_cache = @import("token_cache.zig");
const workspace = @import("workspace.zig");

const Allocator = std.mem.Allocator;

pub const Source = enum {
    lexer,
    parser,
    sema,
    imports,
};

pub const Diagnostic = struct {
    source: Source,
    severity: frontend.Severity,
    range: frontend.Range,
    message: []u8,
};

pub const Depth = enum {
    fast,
    full,

    fn satisfies(self: Depth, requested: Depth) bool {
        return self == .full or requested == .fast;
    }
};

pub const CacheEntry = struct {
    version: i32,
    generation: u64,
    depth: Depth,
    diagnostics: []Diagnostic,

    pub fn matches(self: *const CacheEntry, version: i32, generation: u64, depth: Depth) bool {
        return self.version == version and
            self.generation == generation and
            self.depth.satisfies(depth);
    }

    pub fn estimatedByteSize(self: *const CacheEntry) usize {
        var total: usize = @sizeOf(CacheEntry);
        total = addSat(total, mulSat(self.diagnostics.len, @sizeOf(Diagnostic)));
        for (self.diagnostics) |diagnostic| {
            total = addSat(total, diagnostic.message.len);
        }
        return total;
    }
};

pub fn appendLexerDiagnostics(
    allocator: Allocator,
    diagnostics: *std.ArrayList(Diagnostic),
    lexer_diagnostics: []const token_cache.Diagnostic,
) !void {
    for (lexer_diagnostics) |diagnostic| {
        const message = if (diagnostic.suggestion) |suggestion|
            try std.fmt.allocPrint(allocator, "{s} (suggestion: {s})", .{ diagnostic.message, suggestion })
        else
            try allocator.dupe(u8, diagnostic.message);
        errdefer allocator.free(message);
        try diagnostics.append(allocator, .{
            .source = .lexer,
            .severity = lexerSeverityToFrontend(diagnostic.severity),
            .range = lexerRangeToFrontend(diagnostic.range),
            .message = message,
        });
    }
}

pub fn appendCompilerDiagnostics(
    allocator: Allocator,
    diagnostics: *std.ArrayList(Diagnostic),
    sources: *const compiler.source.SourceStore,
    file_id: compiler.FileId,
    source: Source,
    diagnostic_list: *const compiler.diagnostics.DiagnosticList,
) !void {
    for (diagnostic_list.items.items) |diagnostic| {
        const range = compilerDiagnosticRange(sources, file_id, diagnostic) orelse continue;
        try appendCachedDiagnostic(
            allocator,
            diagnostics,
            source,
            compilerSeverityToFrontend(diagnostic.severity),
            range,
            diagnostic.message,
        );
    }
}

pub fn appendImportDiagnostics(
    allocator: Allocator,
    diagnostics: *std.ArrayList(Diagnostic),
    import_diagnostics: []const workspace.ImportResolutionDiagnostic,
) !void {
    for (import_diagnostics) |diagnostic| {
        try appendCachedDiagnostic(
            allocator,
            diagnostics,
            .imports,
            .err,
            diagnostic.range,
            diagnostic.message,
        );
    }
}

pub fn sourceName(source: Source) []const u8 {
    return switch (source) {
        .lexer => "ora-lexer",
        .parser => "ora-parser",
        .sema => "ora-sema",
        .imports => "ora-imports",
    };
}

fn appendCachedDiagnostic(
    allocator: Allocator,
    diagnostics: *std.ArrayList(Diagnostic),
    source: Source,
    severity: frontend.Severity,
    range: frontend.Range,
    message: []const u8,
) !void {
    const message_copy = try allocator.dupe(u8, message);
    errdefer allocator.free(message_copy);
    try diagnostics.append(allocator, .{
        .source = source,
        .severity = severity,
        .range = range,
        .message = message_copy,
    });
}

fn compilerDiagnosticRange(
    sources: *const compiler.source.SourceStore,
    file_id: compiler.FileId,
    diagnostic: compiler.diagnostics.Diagnostic,
) ?frontend.Range {
    const range = if (diagnostic.labels.len > 0) blk: {
        const label = diagnostic.labels[0];
        if (label.location.file_id != file_id) return null;
        break :blk label.location.range;
    } else compiler.TextRange.empty(0);

    return compilerTextRangeToFrontend(sources, file_id, range);
}

fn compilerTextRangeToFrontend(
    sources: *const compiler.source.SourceStore,
    file_id: compiler.FileId,
    range: compiler.TextRange,
) frontend.Range {
    const start = sources.lineColumn(.{
        .file_id = file_id,
        .range = .{ .start = range.start, .end = range.start },
    });
    const end = sources.lineColumn(.{
        .file_id = file_id,
        .range = .{ .start = range.end, .end = range.end },
    });
    return .{
        .start = .{
            .line = oneBasedToZeroBased(start.line),
            .character = oneBasedToZeroBased(start.column),
        },
        .end = .{
            .line = oneBasedToZeroBased(end.line),
            .character = oneBasedToZeroBased(end.column),
        },
    };
}

fn lexerRangeToFrontend(range: lexer_mod.SourceRange) frontend.Range {
    var end_character = oneBasedToZeroBased(range.end_column);
    const start_character = oneBasedToZeroBased(range.start_column);
    if (end_character < start_character) end_character = start_character;
    return .{
        .start = .{
            .line = oneBasedToZeroBased(range.start_line),
            .character = start_character,
        },
        .end = .{
            .line = oneBasedToZeroBased(range.end_line),
            .character = end_character,
        },
    };
}

fn compilerSeverityToFrontend(severity: compiler.diagnostics.Severity) frontend.Severity {
    return switch (severity) {
        .Error => .err,
        .Warning => .warning,
        .Note => .information,
        .Help => .hint,
    };
}

fn lexerSeverityToFrontend(severity: lexer_mod.DiagnosticSeverity) frontend.Severity {
    return switch (severity) {
        .Error => .err,
        .Warning => .warning,
        .Info => .information,
        .Hint => .hint,
    };
}

fn oneBasedToZeroBased(value: u32) u32 {
    return if (value == 0) 0 else value - 1;
}

fn addSat(a: usize, b: usize) usize {
    return std.math.add(usize, a, b) catch std.math.maxInt(usize);
}

fn mulSat(a: usize, b: usize) usize {
    return std.math.mul(usize, a, b) catch std.math.maxInt(usize);
}
