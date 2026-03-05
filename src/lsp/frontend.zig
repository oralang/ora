const std = @import("std");
const lexer = @import("ora_lexer");
const parser = @import("../parser.zig");

const Allocator = std.mem.Allocator;

pub const DiagnosticSource = enum {
    lexer,
    parser,
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

    const previous_parser_stderr = parser.diagnostics.enable_stderr_diagnostics;
    parser.diagnostics.enable_stderr_diagnostics = false;
    defer parser.diagnostics.enable_stderr_diagnostics = previous_parser_stderr;

    var parse_result = parser.parseRaw(allocator, tokens) catch |err| {
        const message = try std.fmt.allocPrint(allocator, "Parser error: {s}", .{parserErrorMessage(err)});
        try diagnostics.append(allocator, Diagnostic{
            .source = .parser,
            .severity = .err,
            .range = eofTokenRange(tokens),
            .message = message,
        });

        return .{
            .diagnostics = try diagnostics.toOwnedSlice(allocator),
            .parse_succeeded = false,
        };
    };
    parse_result.arena.deinit();

    return .{
        .diagnostics = try diagnostics.toOwnedSlice(allocator),
        .parse_succeeded = true,
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

fn parserErrorMessage(err: parser.ParserError) []const u8 {
    return switch (err) {
        error.UnexpectedToken => "unexpected token",
        error.ExpectedToken => "expected token",
        error.ExpectedIdentifier => "expected identifier",
        error.ExpectedType => "expected type",
        error.ExpectedExpression => "expected expression",
        error.ExpectedRangeExpression => "expected range expression",
        error.UnexpectedEof => "unexpected end of file",
        error.OutOfMemory => "out of memory",
        error.InvalidMemoryRegion => "invalid memory region",
        error.InvalidReturnType => "invalid return type",
        error.UnresolvedType => "unresolved type",
        error.TypeResolutionFailed => "type resolution failed",
    };
}
