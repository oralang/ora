const std = @import("std");
const lexer = @import("ora_lexer");

const Allocator = std.mem.Allocator;

pub const Token = struct {
    type: lexer.TokenType,
    lexeme: []const u8,
    string_value: ?[]const u8 = null,
    range: lexer.SourceRange,
    line: u32,
    column: u32,
    leading_trivia_start: u32 = 0,
    leading_trivia_len: u32 = 0,
    trailing_trivia_start: u32 = 0,
    trailing_trivia_len: u32 = 0,
};

pub const Diagnostic = struct {
    error_type: lexer.LexerError,
    range: lexer.SourceRange,
    message: []const u8,
    suggestion: ?[]const u8,
    severity: lexer.DiagnosticSeverity,

    fn deinit(self: Diagnostic, allocator: Allocator) void {
        allocator.free(self.message);
        if (self.suggestion) |suggestion| allocator.free(suggestion);
    }
};

pub const Cache = struct {
    tokens: []Token,
    diagnostics: []Diagnostic,
    tokens_reserved: usize = 0,
    diagnostics_reserved: usize = 0,
    builder_growth_events: usize = 0,

    pub fn init(allocator: Allocator, source: []const u8) !Cache {
        return initWithScratch(allocator, allocator, source);
    }

    pub fn initWithScratch(result_allocator: Allocator, scratch_allocator: Allocator, source: []const u8) !Cache {
        var lex = lexer.Lexer.initWithRecovery(scratch_allocator, source);
        defer lex.deinit();

        const scanned = lex.scanTokens() catch |err| switch (err) {
            error.OutOfMemory => return error.OutOfMemory,
            else => {
                const diagnostics = try cloneDiagnostics(result_allocator, lex.getDiagnostics());
                errdefer freeDiagnostics(result_allocator, diagnostics);
                return .{
                    .tokens = try result_allocator.alloc(Token, 0),
                    .diagnostics = diagnostics,
                    .tokens_reserved = 0,
                    .diagnostics_reserved = diagnostics.len,
                };
            },
        };
        defer scratch_allocator.free(scanned);

        const diagnostics = try cloneDiagnostics(result_allocator, lex.getDiagnostics());
        errdefer freeDiagnostics(result_allocator, diagnostics);

        const tokens = try result_allocator.alloc(Token, scanned.len);
        var built: usize = 0;
        errdefer {
            for (tokens[0..built]) |token| {
                if (token.string_value) |value| result_allocator.free(value);
            }
            result_allocator.free(tokens);
        }

        for (scanned, 0..) |token, i| {
            tokens[i] = .{
                .type = token.type,
                .lexeme = sourceLexeme(source, token),
                .string_value = try cloneStringValue(result_allocator, token),
                .range = token.range,
                .line = token.line,
                .column = token.column,
                .leading_trivia_start = token.leading_trivia_start,
                .leading_trivia_len = token.leading_trivia_len,
                .trailing_trivia_start = token.trailing_trivia_start,
                .trailing_trivia_len = token.trailing_trivia_len,
            };
            built = i + 1;
        }

        return .{
            .tokens = tokens,
            .diagnostics = diagnostics,
            .tokens_reserved = scanned.len,
            .diagnostics_reserved = diagnostics.len,
        };
    }

    pub fn deinit(self: *Cache, allocator: Allocator) void {
        for (self.tokens) |token| {
            if (token.string_value) |value| allocator.free(value);
        }
        allocator.free(self.tokens);
        freeDiagnostics(allocator, self.diagnostics);
        self.* = .{
            .tokens = &.{},
            .diagnostics = &.{},
            .tokens_reserved = 0,
            .diagnostics_reserved = 0,
            .builder_growth_events = 0,
        };
    }

    pub fn estimatedByteSize(self: *const Cache) usize {
        var total = bytesFor(Token, self.tokens.len);
        total = addSat(total, bytesFor(Diagnostic, self.diagnostics.len));
        for (self.tokens) |token| {
            if (token.string_value) |value| total = addSat(total, value.len);
        }
        for (self.diagnostics) |diagnostic| {
            total = addSat(total, diagnostic.message.len);
            if (diagnostic.suggestion) |suggestion| total = addSat(total, suggestion.len);
        }
        return total;
    }

    pub fn builderCapacityRequested(self: *const Cache) usize {
        return addSat(self.tokens_reserved, self.diagnostics_reserved);
    }

    pub fn builderItemsBuilt(self: *const Cache) usize {
        return addSat(self.tokens.len, self.diagnostics.len);
    }

    pub fn builderUnusedCapacity(self: *const Cache) usize {
        const requested = self.builderCapacityRequested();
        const used = self.builderItemsBuilt();
        return if (requested > used) requested - used else 0;
    }

    pub fn builderGrowthEvents(self: *const Cache) usize {
        return self.builder_growth_events;
    }
};

fn cloneDiagnostics(allocator: Allocator, diagnostics: []const lexer.LexerDiagnostic) ![]Diagnostic {
    const cloned = try allocator.alloc(Diagnostic, diagnostics.len);
    var built: usize = 0;
    errdefer {
        for (cloned[0..built]) |diagnostic| diagnostic.deinit(allocator);
        allocator.free(cloned);
    }

    for (diagnostics, 0..) |diagnostic, i| {
        cloned[i] = .{
            .error_type = diagnostic.error_type,
            .range = diagnostic.range,
            .message = try allocator.dupe(u8, diagnostic.message),
            .suggestion = if (diagnostic.suggestion) |suggestion| try allocator.dupe(u8, suggestion) else null,
            .severity = diagnostic.severity,
        };
        built = i + 1;
    }

    return cloned;
}

fn freeDiagnostics(allocator: Allocator, diagnostics: []Diagnostic) void {
    for (diagnostics) |diagnostic| diagnostic.deinit(allocator);
    allocator.free(diagnostics);
}

fn cloneStringValue(allocator: Allocator, token: lexer.Token) !?[]const u8 {
    const value = token.value orelse return null;
    return switch (value) {
        .string => |string| try allocator.dupe(u8, string),
        else => null,
    };
}

fn sourceLexeme(source: []const u8, token: lexer.Token) []const u8 {
    const start: usize = @intCast(token.range.start_offset);
    const end: usize = @intCast(token.range.end_offset);
    if (start <= end and end <= source.len) {
        return source[start..end];
    }
    return "";
}

fn bytesFor(comptime T: type, len: usize) usize {
    return std.math.mul(usize, @sizeOf(T), len) catch std.math.maxInt(usize);
}

fn addSat(a: usize, b: usize) usize {
    return std.math.add(usize, a, b) catch std.math.maxInt(usize);
}
