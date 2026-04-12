const std = @import("std");
const lexer = @import("ora_lexer");
const compiler = @import("../compiler.zig");

const Allocator = std.mem.Allocator;

pub const DiagnosticSource = enum {
    lexer,
    parser,
    sema,
};

pub const Severity = enum {
    err,
    warning,
    information,
    hint,
};

pub const Position = struct {
    line: u32,
    character: u32,
};

pub const Range = struct {
    start: Position,
    end: Position,
};

pub const Diagnostic = struct {
    source: DiagnosticSource,
    severity: Severity,
    range: Range,
    message: []const u8,
};

pub const DocumentAnalysis = struct {
    diagnostics: []Diagnostic,
    parse_succeeded: bool,

    pub fn deinit(self: *DocumentAnalysis, allocator: Allocator) void {
        for (self.diagnostics) |diagnostic| {
            allocator.free(diagnostic.message);
        }
        allocator.free(self.diagnostics);
    }

    pub fn hasErrors(self: DocumentAnalysis) bool {
        for (self.diagnostics) |diagnostic| {
            if (diagnostic.severity == .err) return true;
        }
        return false;
    }
};

pub fn analyzeDocument(allocator: Allocator, source: []const u8) !DocumentAnalysis {
    const file_id = compiler.FileId.fromIndex(0);
    var diagnostics = std.ArrayList(Diagnostic){};
    errdefer {
        for (diagnostics.items) |diagnostic| {
            allocator.free(diagnostic.message);
        }
        diagnostics.deinit(allocator);
    }

    var lex = try lexer.Lexer.initWithConfig(allocator, source, lexer.LexerConfig.development());
    defer lex.deinit();

    const tokens = try lex.scanTokens();
    defer allocator.free(tokens);

    for (lex.getDiagnostics()) |lex_diagnostic| {
        const message = try formatLexerMessage(allocator, lex_diagnostic);
        try diagnostics.append(allocator, Diagnostic{
            .source = .lexer,
            .severity = mapLexerSeverity(lex_diagnostic.severity),
            .range = mapLexerRange(lex_diagnostic.range),
            .message = message,
        });
    }

    var parse_result = try compiler.syntax.parse(allocator, file_id, source);
    defer parse_result.deinit();

    const parse_had_diagnostics = parse_result.diagnostics.items.items.len != 0;
    for (parse_result.diagnostics.items.items) |diag| {
        const label = if (diag.labels.len > 0) diag.labels[0].location.range else compiler.TextRange.empty(0);
        const message = try allocator.dupe(u8, diag.message);
        try diagnostics.append(allocator, Diagnostic{
            .source = .parser,
            .severity = mapCompilerSeverity(diag.severity),
            .range = textRangeToRange(allocator, source, label) catch eofTokenRange(tokens),
            .message = message,
        });
    }

    var lower_result = try compiler.ast.lower(allocator, &parse_result.tree);
    defer lower_result.deinit();

    const lower_had_diagnostics = lower_result.diagnostics.items.items.len != 0;
    for (lower_result.diagnostics.items.items) |diag| {
        const label = if (diag.labels.len > 0) diag.labels[0].location.range else compiler.TextRange.empty(0);
        const message = try allocator.dupe(u8, diag.message);
        try diagnostics.append(allocator, Diagnostic{
            .source = .parser,
            .severity = mapCompilerSeverity(diag.severity),
            .range = textRangeToRange(allocator, source, label) catch eofTokenRange(tokens),
            .message = message,
        });
    }

    // Run sema type checking if parsing succeeded — produces type errors,
    // region errors, effect errors, and refinement diagnostics.
    var sema_had_diagnostics = false;
    if (!parse_had_diagnostics and !lower_had_diagnostics) {
        var item_index = try compiler.sema.buildItemIndex(allocator, &lower_result.file);
        defer item_index.deinit();
        var resolution = try compiler.sema.resolveNames(allocator, file_id, &lower_result.file, &item_index);
        defer resolution.deinit();
        var const_eval_result = try compiler.comptime_eval.constEval(allocator, &lower_result.file, .{});
        defer const_eval_result.deinit();

        var typecheck_result = try compiler.sema.typeCheck(
            allocator,
            compiler.ModuleId.fromIndex(0),
            file_id,
            &lower_result.file,
            &item_index,
            &resolution,
            &const_eval_result,
            .{ .body = compiler.ast.BodyId.fromIndex(0) },
            null,
        );
        defer typecheck_result.deinit();

        sema_had_diagnostics = typecheck_result.diagnostics.items.items.len != 0;
        for (typecheck_result.diagnostics.items.items) |diag| {
            const label = if (diag.labels.len > 0) diag.labels[0].location.range else compiler.TextRange.empty(0);
            const message = try allocator.dupe(u8, diag.message);
            try diagnostics.append(allocator, Diagnostic{
                .source = .sema,
                .severity = mapCompilerSeverity(diag.severity),
                .range = textRangeToRange(allocator, source, label) catch eofTokenRange(tokens),
                .message = message,
            });
        }

        // Also collect comptime diagnostics.
        for (const_eval_result.diagnostics.items.items) |diag| {
            const label = if (diag.labels.len > 0) diag.labels[0].location.range else compiler.TextRange.empty(0);
            const message = try allocator.dupe(u8, diag.message);
            try diagnostics.append(allocator, Diagnostic{
                .source = .sema,
                .severity = mapCompilerSeverity(diag.severity),
                .range = textRangeToRange(allocator, source, label) catch eofTokenRange(tokens),
                .message = message,
            });
        }
    }

    return .{
        .diagnostics = try diagnostics.toOwnedSlice(allocator),
        .parse_succeeded = !(parse_had_diagnostics or lower_had_diagnostics or sema_had_diagnostics),
    };
}

fn mapLexerSeverity(severity: lexer.DiagnosticSeverity) Severity {
    return switch (severity) {
        .Error => .err,
        .Warning => .warning,
        .Info => .information,
        .Hint => .hint,
    };
}

fn mapLexerRange(range: lexer.SourceRange) Range {
    var end_character = toZeroBased(range.end_column);
    const start_character = toZeroBased(range.start_column);

    if (end_character < start_character) {
        end_character = start_character;
    }

    return .{
        .start = .{
            .line = toZeroBased(range.start_line),
            .character = start_character,
        },
        .end = .{
            .line = toZeroBased(range.end_line),
            .character = end_character,
        },
    };
}

fn formatLexerMessage(allocator: Allocator, diagnostic: lexer.LexerDiagnostic) ![]const u8 {
    if (diagnostic.suggestion) |suggestion| {
        return std.fmt.allocPrint(allocator, "{s} (suggestion: {s})", .{ diagnostic.message, suggestion });
    }

    return allocator.dupe(u8, diagnostic.message);
}

fn eofTokenRange(tokens: []const lexer.Token) Range {
    if (tokens.len == 0) {
        return .{
            .start = .{ .line = 0, .character = 0 },
            .end = .{ .line = 0, .character = 0 },
        };
    }

    const eof = tokens[tokens.len - 1];
    const line = toZeroBased(eof.line);
    const character = toZeroBased(eof.column);
    return .{
        .start = .{ .line = line, .character = character },
        .end = .{ .line = line, .character = character },
    };
}

fn toZeroBased(value: u32) u32 {
    return if (value == 0) 0 else value - 1;
}

fn mapCompilerSeverity(severity: compiler.diagnostics.Severity) Severity {
    return switch (severity) {
        .Error => .err,
        .Warning => .warning,
        .Note => .information,
        .Help => .hint,
    };
}

fn textRangeToRange(allocator: Allocator, source_text: []const u8, range: compiler.TextRange) !Range {
    var sources = compiler.source.SourceStore.init(allocator);
    defer sources.deinit();

    const file_id = try sources.addFile("<lsp>", source_text);
    const start = sources.lineColumn(.{ .file_id = file_id, .range = .{ .start = range.start, .end = range.start } });
    const end = sources.lineColumn(.{ .file_id = file_id, .range = .{ .start = range.end, .end = range.end } });

    return .{
        .start = .{
            .line = toZeroBased(start.line),
            .character = toZeroBased(start.column),
        },
        .end = .{
            .line = toZeroBased(end.line),
            .character = toZeroBased(end.column),
        },
    };
}
