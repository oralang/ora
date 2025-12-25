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
        LexerError.InvalidBuiltinFunction => ErrorMessageTemplate{
            .title = "invalid built-in function",
            .description = "This is not a valid built-in function name",
            .help = "Use a valid built-in function like @divTrunc, @divFloor, @divCeil, @divExact, or @divmod",
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

        std.debug.print("Added error to recovery: {s} at {}:{}\n", .{ @errorName(error_type), range.start_line, range.start_column });
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

    /// Record a diagnostic with custom severity
    pub fn recordDiagnostic(self: *ErrorRecovery, diagnostic: LexerDiagnostic) !void {
        if (self.errors.items.len >= self.max_errors) {
            return LexerError.TooManyErrors;
        }

        // create a deep copy of the diagnostic to avoid double-free issues
        var new_diagnostic = diagnostic;

        // copy message if owned
        if (diagnostic.message_owned) {
            new_diagnostic.message = try self.allocator.dupe(u8, diagnostic.message);
        }

        // copy suggestion if owned
        if (diagnostic.suggestion_owned and diagnostic.suggestion != null) {
            new_diagnostic.suggestion = try self.allocator.dupe(u8, diagnostic.suggestion.?);
        }

        // copy context if present
        if (diagnostic.context != null) {
            const context = diagnostic.context.?;
            const source_line = try self.allocator.dupe(u8, context.source_line);
            new_diagnostic.context = .{
                .source_line = source_line,
                .line_number = context.line_number,
                .column_start = context.column_start,
                .column_end = context.column_end,
            };
        }

        try self.errors.append(new_diagnostic);
    }

    /// Check if we have reached the maximum error limit
    pub fn hasReachedLimit(self: *ErrorRecovery) bool {
        return self.errors.items.len >= self.max_errors;
    }

    /// Get all collected errors
    pub fn getErrors(self: *ErrorRecovery) []const LexerDiagnostic {
        return self.errors.items;
    }

    /// Get error count
    pub fn getErrorCount(self: *const ErrorRecovery) usize {
        return self.errors.items.len;
    }

    /// Get errors filtered by severity
    pub fn getErrorsBySeverity(self: *ErrorRecovery, severity: DiagnosticSeverity) std.ArrayList(LexerDiagnostic) {
        var filtered = std.ArrayList(LexerDiagnostic){};

        for (self.errors.items) |diagnostic| {
            if (diagnostic.severity == severity) {
                filtered.append(diagnostic) catch continue;
            }
        }

        return filtered;
    }

    pub const ErrorTypeCount = struct {
        error_type: LexerError,
        count: usize,
    };

    /// Get errors grouped by type - returns a slice of error types and their diagnostic counts
    pub fn getErrorsByType(self: *ErrorRecovery) std.ArrayList(ErrorTypeCount) {
        var type_counts = std.AutoHashMap(LexerError, usize).init(self.allocator);
        defer type_counts.deinit();

        for (self.errors.items) |diagnostic| {
            const current = type_counts.get(diagnostic.error_type) orelse 0;
            type_counts.put(diagnostic.error_type, current + 1) catch continue;
        }

        var result = std.ArrayList(ErrorTypeCount){};
        var iterator = type_counts.iterator();
        while (iterator.next()) |entry| {
            result.append(.{ .error_type = entry.key_ptr.*, .count = entry.value_ptr.* }) catch continue;
        }

        return result;
    }

    pub const LineCount = struct {
        line: u32,
        count: usize,
    };

    /// Get errors grouped by line number - returns a slice of line numbers and their diagnostic counts
    pub fn getErrorsByLine(self: *ErrorRecovery) std.ArrayList(LineCount) {
        var line_counts = std.AutoHashMap(u32, usize).init(self.allocator);
        defer line_counts.deinit();

        for (self.errors.items) |diagnostic| {
            const line = diagnostic.range.start_line;
            const current = line_counts.get(line) orelse 0;
            line_counts.put(line, current + 1) catch continue;
        }

        var result = std.ArrayList(LineCount){};
        var iterator = line_counts.iterator();
        while (iterator.next()) |entry| {
            result.append(.{ .line = entry.key_ptr.*, .count = entry.value_ptr.* }) catch continue;
        }

        return result;
    }

    /// Create a summary report of all errors
    pub fn createSummaryReport(self: *ErrorRecovery, allocator: Allocator) ![]u8 {
        var buffer = std.ArrayList(u8){};
        defer buffer.deinit(allocator);

        const writer = buffer.writer(allocator);

        // summary header
        try writer.print("Diagnostic Summary ({d} errors)\n", .{self.errors.items.len});
        try writer.writeAll("=" ** 50);
        try writer.writeAll("\n\n");

        // group by error type
        var type_groups = self.getErrorsByType();
        defer type_groups.deinit();

        for (type_groups.items) |group| {
            try writer.print("{s}: {d} occurrences\n", .{ @errorName(group.error_type), group.count });

            // find first occurrence for details
            for (self.errors.items) |diagnostic| {
                if (diagnostic.error_type == group.error_type) {
                    try writer.print("  First occurrence: {d}:{d}\n", .{ diagnostic.range.start_line, diagnostic.range.start_column });
                    break;
                }
            }

            if (group.count > 1) {
                try writer.print("  Additional occurrences: {d}\n", .{group.count - 1});
            }
            try writer.writeAll("\n");
        }

        // severity breakdown
        try writer.writeAll("Severity Breakdown:\n");
        try writer.writeAll("-" ** 20);
        try writer.writeAll("\n");

        var error_diagnostics = self.getErrorsBySeverity(.Error);
        defer error_diagnostics.deinit();
        var warning_diagnostics = self.getErrorsBySeverity(.Warning);
        defer warning_diagnostics.deinit();
        var info_diagnostics = self.getErrorsBySeverity(.Info);
        defer info_diagnostics.deinit();
        var hint_diagnostics = self.getErrorsBySeverity(.Hint);
        defer hint_diagnostics.deinit();

        const error_count = error_diagnostics.items.len;
        const warning_count = warning_diagnostics.items.len;
        const info_count = info_diagnostics.items.len;
        const hint_count = hint_diagnostics.items.len;

        try writer.print("Errors: {d}\n", .{error_count});
        try writer.print("Warnings: {d}\n", .{warning_count});
        try writer.print("Info: {d}\n", .{info_count});
        try writer.print("Hints: {d}\n", .{hint_count});

        return buffer.toOwnedSlice(allocator);
    }

    /// Create a detailed diagnostic report with grouped errors and source context
    pub fn createDetailedReport(self: *ErrorRecovery, allocator: Allocator) ![]u8 {
        var buffer = std.ArrayList(u8){};
        defer buffer.deinit(allocator);

        const writer = buffer.writer(allocator);

        // report header
        try writer.print("Diagnostic Report ({d} issues found)\n", .{self.errors.items.len});
        try writer.writeAll("=" ** 50);
        try writer.writeAll("\n\n");

        // group errors for better organization
        var groups = try self.groupErrors();
        defer {
            for (groups.items) |*group| {
                var related = group.related;
                related.deinit(self.allocator);
            }
            groups.deinit(self.allocator);
        }

        // print each group
        for (groups.items, 0..) |group, i| {
            const primary = group.primary;

            // group header
            try writer.print("Issue #{d}: {s}\n", .{ i + 1, @errorName(primary.error_type) });
            try writer.writeAll("-" ** 40);
            try writer.writeAll("\n");

            // primary error details
            try writer.print("Location: {d}:{d}\n", .{ primary.range.start_line, primary.range.start_column });

            if (primary.template) |template| {
                try writer.print("Description: {s}\n", .{template.description});
                if (template.help) |help| {
                    try writer.print("Help: {s}\n", .{help});
                }
            } else {
                try writer.print("Message: {s}\n", .{primary.message});
            }

            // source context
            if (primary.context) |context| {
                try writer.writeAll("\nSource context:\n");
                try context.format("", .{}, writer);
                try writer.writeAll("\n");
            }

            // related errors
            if (group.related.items.len > 0) {
                try writer.print("\nRelated issues ({d} similar problems):\n", .{group.related.items.len});

                for (group.related.items, 0..) |related, j| {
                    if (j >= 3) {
                        try writer.print("... and {d} more\n", .{group.related.items.len - 3});
                        break;
                    }

                    try writer.print("- {d}:{d} {s}\n", .{ related.range.start_line, related.range.start_column, if (related.template) |t| t.title else related.message });
                }
            }

            // suggestions
            if (primary.suggestion) |suggestion| {
                try writer.print("\nSuggestion: {s}\n", .{suggestion});
            }

            try writer.writeAll("\n\n");
        }

        return buffer.toOwnedSlice(allocator);
    }

    /// Filter diagnostics by severity level (minimum severity)
    pub fn filterByMinimumSeverity(self: *ErrorRecovery, min_severity: DiagnosticSeverity) std.ArrayList(LexerDiagnostic) {
        var filtered = std.ArrayList(LexerDiagnostic){};

        for (self.errors.items) |diagnostic| {
            if (@intFromEnum(diagnostic.severity) >= @intFromEnum(min_severity)) {
                filtered.append(diagnostic) catch continue;
            }
        }

        return filtered;
    }

    /// Get related errors (same line or nearby)
    pub fn getRelatedErrors(self: *ErrorRecovery, diagnostic: LexerDiagnostic, max_distance: u32) std.ArrayList(LexerDiagnostic) {
        var related = std.ArrayList(LexerDiagnostic){};

        for (self.errors.items) |other| {
            // skip self by comparing memory addresses
            if (std.mem.eql(u8, diagnostic.message, other.message) and
                diagnostic.range.start_line == other.range.start_line and
                diagnostic.range.start_column == other.range.start_column) continue;

            const line_distance = if (diagnostic.range.start_line > other.range.start_line)
                diagnostic.range.start_line - other.range.start_line
            else
                other.range.start_line - diagnostic.range.start_line;

            if (line_distance <= max_distance) {
                related.append(other) catch continue;
            }
        }

        return related;
    }

    pub const DiagnosticGroup = struct {
        primary: LexerDiagnostic,
        related: std.ArrayList(LexerDiagnostic),
    };

    /// Group errors by type and location for better organization
    pub fn groupErrors(self: *ErrorRecovery) !std.ArrayList(DiagnosticGroup) {
        var groups = std.ArrayList(DiagnosticGroup){};

        if (self.errors.items.len == 0) {
            return groups;
        }

        var processed = std.AutoHashMap(usize, void).init(self.allocator);
        defer processed.deinit();

        // process all errors
        for (self.errors.items, 0..) |diagnostic, i| {
            // skip if already processed as part of another group
            if (processed.contains(i)) continue;

            // mark this error as processed
            try processed.put(i, {});

            // create a new group with this error as primary
            var related = std.ArrayList(LexerDiagnostic){};

            // find related errors (same type or nearby location)
            for (self.errors.items, 0..) |other, j| {
                if (i == j) continue; // Skip self
                if (processed.contains(j)) continue; // Skip already processed

                const same_type = diagnostic.error_type == other.error_type;
                const line_distance = if (diagnostic.range.start_line > other.range.start_line)
                    diagnostic.range.start_line - other.range.start_line
                else
                    other.range.start_line - diagnostic.range.start_line;

                const nearby = line_distance <= 3; // Within 3 lines

                if (same_type or nearby) {
                    try related.append(self.allocator, other);
                    try processed.put(j, {}); // Mark as processed
                }
            }

            // add the group
            try groups.append(self.allocator, .{
                .primary = diagnostic,
                .related = related,
            });
        }

        return groups;
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
            LexerError.InvalidBuiltinFunction => "Use a valid built-in function like @divTrunc, @divFloor, @divCeil, @divExact, or @divmod",
            else => null,
        };
    }
};
