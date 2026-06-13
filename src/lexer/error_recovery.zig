// ============================================================================
// Error Recovery System
// ============================================================================
//
// Comprehensive error recovery and diagnostic system for the lexer.
// Handles error collection, grouping, filtering, and reporting.
//
// ============================================================================

const std = @import("std");
const Allocator = std.mem.Allocator;

// Import types from lexer
const LexerError = @import("../lexer.zig").LexerError;
const SourceRange = @import("../lexer.zig").SourceRange;

/// Diagnostic severity levels for error reporting
pub const DiagnosticSeverity = enum {
    Error,
    Warning,
    Info,
    Hint,

    pub fn format(self: DiagnosticSeverity, comptime fmt: []const u8, options: std.fmt.FormatOptions, writer: anytype) !void {
        _ = fmt;
        _ = options;
        try writer.writeAll(@tagName(self));
    }
};

/// Error message template for consistent formatting
pub const ErrorMessageTemplate = struct {
    title: []const u8,
    description: []const u8,
    help: ?[]const u8 = null,

    pub fn format(self: ErrorMessageTemplate, comptime fmt: []const u8, options: std.fmt.FormatOptions, writer: anytype) !void {
        _ = fmt;
        _ = options;
        try writer.print("{s}: {s}", .{ self.title, self.description });
        if (self.help) |help| {
            try writer.print("\n  help: {s}", .{help});
        }
    }
};

/// Context information for error reporting
pub const ErrorContext = struct {
    source_line: []const u8,
    line_number: u32,
    column_start: u32,
    column_end: u32,

    pub fn format(self: ErrorContext, comptime fmt: []const u8, options: std.fmt.FormatOptions, writer: anytype) !void {
        _ = fmt;
        _ = options;

        // show line number and source line
        try writer.print("   {d} | {s}\n", .{ self.line_number, self.source_line });

        // show error indicator with carets
        try writer.writeAll("     | ");
        var i: u32 = 1;
        while (i < self.column_start) : (i += 1) {
            try writer.writeAll(" ");
        }

        // draw carets under the problematic area
        const caret_count = @max(1, self.column_end - self.column_start);
        var j: u32 = 0;
        while (j < caret_count) : (j += 1) {
            try writer.writeAll("^");
        }
    }
};

/// Lexer diagnostic with error details and suggestions
pub const LexerDiagnostic = struct {
    error_type: LexerError,
    range: SourceRange,
    message: []const u8,
    message_owned: bool = false, // Track if message needs to be freed
    suggestion: ?[]const u8,
    suggestion_owned: bool = false, // Track if suggestion needs to be freed
    severity: DiagnosticSeverity,
    context: ?ErrorContext = null,
    template: ?ErrorMessageTemplate = null,

    pub fn init(error_type: LexerError, range: SourceRange, message: []const u8) LexerDiagnostic {
        return LexerDiagnostic{
            .error_type = error_type,
            .range = range,
            .message = message,
            .suggestion = null,
            .severity = .Error,
        };
    }

    pub fn withSuggestion(self: LexerDiagnostic, suggestion: []const u8) LexerDiagnostic {
        var result = self;
        result.suggestion = suggestion;
        return result;
    }

    /// Create a detailed diagnostic with source context
    pub fn createDetailed(
        allocator: Allocator,
        error_type: LexerError,
        range: SourceRange,
        source: []const u8,
        message: []const u8,
    ) !LexerDiagnostic {
        const context = try extractSourceContext(allocator, source, range);
        const template = getErrorTemplate(error_type);

        // allocate owned copy of the message
        const owned_message = try allocator.dupe(u8, message);
        var diagnostic = LexerDiagnostic.init(error_type, range, owned_message);
        diagnostic.message_owned = true; // Mark message as owned
        diagnostic.context = context;
        diagnostic.template = template;

        return diagnostic;
    }

    pub fn format(self: LexerDiagnostic, comptime fmt: []const u8, options: std.fmt.FormatOptions, writer: anytype) !void {
        _ = fmt;
        _ = options;

        // use template if available, otherwise use basic message
        if (self.template) |template| {
            try writer.print("{s}: {s}", .{ @tagName(self.severity), template.title });
            try writer.print("\n  --> {d}:{d}\n", .{ self.range.start_line, self.range.start_column });

            // show source context if available
            if (self.context) |context| {
                try writer.writeAll("\n");
                try context.format("", .{}, writer);
                try writer.writeAll("\n");
            }

            try writer.print("\n{s}", .{template.description});

            if (template.help) |help| {
                try writer.print("\n  help: {s}", .{help});
            }
        } else {
            try writer.print("{s}: {s} at {any}", .{ @tagName(self.severity), self.message, self.range });
        }

        if (self.suggestion) |suggestion| {
            try writer.print("\n  suggestion: {s}", .{suggestion});
        }
    }
};

/// Extract source context for error reporting
pub fn extractSourceContext(allocator: Allocator, source: []const u8, range: SourceRange) !ErrorContext {
    // find the start of the line containing the error
    var line_start: u32 = 0;
    var current_line: u32 = 1;
    var i: u32 = 0;

    while (i < range.start_offset and i < source.len) {
        if (source[i] == '\n') {
            current_line += 1;
            line_start = i + 1;
        }
        i += 1;
    }

    // find the end of the line
    var line_end = line_start;
    while (line_end < source.len and source[line_end] != '\n') {
        line_end += 1;
    }

    // extract the source line
    const source_line = try allocator.dupe(u8, source[line_start..line_end]);

    return ErrorContext{
        .source_line = source_line,
        .line_number = range.start_line,
        .column_start = range.start_column,
        .column_end = range.end_column,
    };
}

/// Get error message template for consistent formatting
pub fn getErrorTemplate(error_type: LexerError) ErrorMessageTemplate {
    return switch (error_type) {
        LexerError.UnexpectedCharacter => ErrorMessageTemplate{
            .title = "unexpected character",
            .description = "This character is not valid in this context",
            .help = "Remove the character or use it within a string literal",
        },
        LexerError.UnterminatedString => ErrorMessageTemplate{
            .title = "unterminated string literal",
            .description = "String literal is missing a closing quote",
            .help = "Add a closing quote (\") to complete the string",
        },
        LexerError.UnterminatedRawString => ErrorMessageTemplate{
            .title = "unterminated raw string literal",
            .description = "Raw string literal is missing a closing quote",
            .help = "Add a closing quote (\") to complete the raw string",
        },
        LexerError.InvalidEscapeSequence => ErrorMessageTemplate{
            .title = "invalid escape sequence",
            .description = "This escape sequence is not recognized",
            .help = "Use valid escape sequences like \\n, \\t, \\r, \\\\, \\\", or \\xNN",
        },
        LexerError.InvalidBinaryLiteral => ErrorMessageTemplate{
            .title = "invalid binary literal",
            .description = "Binary literals must contain only 0 and 1 digits",
            .help = "Use '0b' prefix followed by binary digits (0 and 1)",
        },
        LexerError.InvalidHexLiteral => ErrorMessageTemplate{
            .title = "invalid hexadecimal literal",
            .description = "Hexadecimal literals must contain only valid hex digits",
            .help = "Use '0x' prefix followed by hex digits (0-9, a-f, A-F)",
        },
        LexerError.NumberTooLarge => ErrorMessageTemplate{
            .title = "number literal too large",
            .description = "This number exceeds the maximum supported value",
            .help = "Use a smaller number that fits within the u256 range",
        },
        LexerError.InvalidAddressFormat => ErrorMessageTemplate{
            .title = "invalid address format",
            .description = "Address literals must be exactly 40 hexadecimal characters",
            .help = "Ensure the address has exactly 40 hex digits after '0x'",
        },
        LexerError.EmptyCharacterLiteral => ErrorMessageTemplate{
            .title = "empty character literal",
            .description = "Character literals cannot be empty",
            .help = "Add a character between the single quotes",
        },
        LexerError.InvalidCharacterLiteral => ErrorMessageTemplate{
            .title = "invalid character literal",
            .description = "Character literals must contain exactly one character",
            .help = "Use single quotes around exactly one character",
        },
        LexerError.UnterminatedComment => ErrorMessageTemplate{
            .title = "unterminated comment",
            .description = "Multi-line comment is missing closing */",
            .help = "Add */ to close the comment",
        },
        LexerError.InvalidRangePattern => ErrorMessageTemplate{
            .title = "invalid range pattern",
            .description = "Range patterns must have valid syntax with proper bounds",
            .help = "Use range patterns like 'a..z', '0..9', or 'start..end' with valid characters or numbers",
        },
        LexerError.InvalidSwitchSyntax => ErrorMessageTemplate{
            .title = "invalid switch syntax",
            .description = "Switch statements require proper syntax with cases and optional default",
            .help = "Ensure switch has proper case labels, expressions, and braces: switch (expr) { case value: ... default: ... }",
        },
        else => ErrorMessageTemplate{
            .title = "lexical error",
            .description = "An error occurred during lexical analysis",
            .help = null,
        },
    };
}

/// Error recovery system for collecting multiple errors during lexing
pub const ErrorRecovery = struct {
    errors: std.ArrayList(LexerDiagnostic),
    max_errors: u32,
    allocator: Allocator,

    pub fn init(allocator: Allocator, max_errors: u32) ErrorRecovery {
        return ErrorRecovery{
            .errors = std.ArrayList(LexerDiagnostic){},
            .max_errors = max_errors,
            .allocator = allocator,
        };
    }

    pub fn deinit(self: *ErrorRecovery) void {
        // free allocated diagnostic messages and context
        for (self.errors.items) |diagnostic| {
            if (diagnostic.suggestion_owned and diagnostic.suggestion != null) {
                self.allocator.free(diagnostic.suggestion.?);
            }
            if (diagnostic.context) |context| {
                self.allocator.free(context.source_line);
            }
            if (diagnostic.message_owned) {
                self.allocator.free(diagnostic.message);
            }
        }
        self.errors.deinit(self.allocator);
    }

    /// Record an error with the given details
    pub fn recordError(self: *ErrorRecovery, error_type: LexerError, range: SourceRange, message: []const u8) !void {
        if (self.errors.items.len >= self.max_errors) {
            return LexerError.TooManyErrors;
        }

        // allocate owned copy of the message
        const owned_message = try self.allocator.dupe(u8, message);
        var diagnostic = LexerDiagnostic.init(error_type, range, owned_message);
        diagnostic.message_owned = true; // Mark message as owned
        try self.errors.append(self.allocator, diagnostic);
    }

    /// Record an error with a suggestion
    pub fn recordErrorWithSuggestion(self: *ErrorRecovery, error_type: LexerError, range: SourceRange, message: []const u8, suggestion: []const u8) !void {
        if (self.errors.items.len >= self.max_errors) {
            return LexerError.TooManyErrors;
        }

        // allocate owned copies of message and suggestion
        const owned_message = try self.allocator.dupe(u8, message);
        const owned_suggestion = try self.allocator.dupe(u8, suggestion);
        var diagnostic = LexerDiagnostic.init(error_type, range, owned_message).withSuggestion(owned_suggestion);
        diagnostic.message_owned = true; // Mark message as owned
        diagnostic.suggestion_owned = true; // Mark suggestion as owned
        try self.errors.append(self.allocator, diagnostic);
    }

    /// Record a detailed error with source context
    pub fn recordDetailedError(self: *ErrorRecovery, error_type: LexerError, range: SourceRange, source: []const u8, message: []const u8) !void {
        if (self.errors.items.len >= self.max_errors) {
            return LexerError.TooManyErrors;
        }

        const diagnostic = try LexerDiagnostic.createDetailed(self.allocator, error_type, range, source, message);
        try self.errors.append(self.allocator, diagnostic);
    }

    /// Record a detailed error with source context and suggestion
    pub fn recordDetailedErrorWithSuggestion(self: *ErrorRecovery, error_type: LexerError, range: SourceRange, source: []const u8, message: []const u8, suggestion: []const u8) !void {
        if (self.errors.items.len >= self.max_errors) {
            return LexerError.TooManyErrors;
        }

        var diagnostic = try LexerDiagnostic.createDetailed(self.allocator, error_type, range, source, message);
        const owned_suggestion = try self.allocator.dupe(u8, suggestion);
        diagnostic = diagnostic.withSuggestion(owned_suggestion);
        diagnostic.suggestion_owned = true; // Mark suggestion as owned
        try self.errors.append(self.allocator, diagnostic);
    }

    /// Get all collected errors
    pub fn getErrors(self: *ErrorRecovery) []const LexerDiagnostic {
        return self.errors.items;
    }

    /// Get error count
    pub fn getErrorCount(self: *const ErrorRecovery) usize {
        return self.errors.items.len;
    }

    /// Clear all collected errors
    pub fn clear(self: *ErrorRecovery) void {
        // free allocated diagnostic messages and context
        for (self.errors.items) |diagnostic| {
            if (diagnostic.suggestion_owned and diagnostic.suggestion != null) {
                self.allocator.free(diagnostic.suggestion.?);
            }
            if (diagnostic.context) |context| {
                self.allocator.free(context.source_line);
            }
            if (diagnostic.message_owned) {
                self.allocator.free(diagnostic.message);
            }
        }
        self.errors.clearAndFree(self.allocator);
    }

    /// Find the next safe token boundary for error recovery
    pub fn findNextTokenBoundary(source: []const u8, current: u32) u32 {
        if (current >= source.len) return current;

        var pos = current;

        // skip to next whitespace, newline, or known token start character
        while (pos < source.len) {
            const c = source[pos];

            // stop at whitespace or newline (safe boundaries)
            if (std.ascii.isWhitespace(c)) {
                return pos;
            }

            // stop at common token start characters
            switch (c) {
                '(', ')', '{', '}', '[', ']', ';', ',', '.', ':', '=', '+', '-', '*', '/', '%', '!', '<', '>', '&', '|', '^', '~' => {
                    return pos;
                },
                // stop at quote characters (string boundaries)
                '"', '\'' => {
                    return pos;
                },
                // stop at digits (number boundaries)
                '0'...'9' => {
                    return pos;
                },
                // stop at letters (identifier boundaries)
                'a'...'z', 'A'...'Z', '_' => {
                    return pos;
                },
                else => {
                    pos += 1;
                },
            }
        }

        return pos;
    }

    /// Suggest a fix for common error patterns
    pub fn suggestFix(error_type: LexerError, context: []const u8) ?[]const u8 {
        return switch (error_type) {
            LexerError.UnexpectedCharacter => blk: {
                if (context.len > 0) {
                    const c = context[0];
                    switch (c) {
                        '$' => break :blk "Remove the '$' character or use it in a string literal",
                        '@' => break :blk "Remove the '@' character or use it in a string literal",
                        '#' => break :blk "Use '//' for comments instead of '#'",
                        '`' => break :blk "Use double quotes for strings instead of backticks",
                        else => break :blk "Remove or replace the invalid character",
                    }
                }
                break :blk "Remove the invalid character";
            },
            LexerError.UnterminatedString => "Add a closing quote to terminate the string",
            LexerError.UnterminatedRawString => "Add a closing quote to terminate the raw string",
            LexerError.InvalidBinaryLiteral => "Use '0b' followed by only binary digits (0 and 1)",
            LexerError.InvalidHexLiteral => "Use '0x' followed by valid hexadecimal digits (0-9, a-f, A-F)",
            LexerError.InvalidAddressFormat => "Address literals must be exactly 40 hexadecimal characters",
            LexerError.NumberTooLarge => "Use a smaller number that fits within the supported range",
            LexerError.EmptyCharacterLiteral => "Add a character between the single quotes",
            LexerError.InvalidCharacterLiteral => "Character literals must contain exactly one character",
            LexerError.InvalidEscapeSequence => "Use a valid escape sequence like \\n, \\t, \\r, \\\\, or \\\"",
            else => null,
        };
    }
};
