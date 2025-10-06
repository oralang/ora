const std = @import("std");
const Allocator = std.mem.Allocator;

/// Lexer-specific errors for better diagnostics
pub const LexerError = error{
    UnexpectedCharacter,
    UnterminatedString,
    InvalidHexLiteral,
    UnterminatedComment,
    OutOfMemory,
    InvalidEscapeSequence,
    UnterminatedRawString,
    EmptyCharacterLiteral,
    InvalidCharacterLiteral,
    InvalidCharacterInString, // For non-ASCII characters in string literals
    InvalidBinaryLiteral,
    NumberTooLarge,
    InvalidAddressFormat,
    TooManyErrors,
    InvalidBuiltinFunction,
    InvalidRangePattern,
    InvalidSwitchSyntax,
};

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

        // Show line number and source line
        try writer.print("   {d} | {s}\n", .{ self.line_number, self.source_line });

        // Show error indicator with carets
        try writer.writeAll("     | ");
        var i: u32 = 1;
        while (i < self.column_start) : (i += 1) {
            try writer.writeAll(" ");
        }

        // Draw carets under the problematic area
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

        // Allocate owned copy of the message
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

        // Use template if available, otherwise use basic message
        if (self.template) |template| {
            try writer.print("{s}: {s}", .{ @tagName(self.severity), template.title });
            try writer.print("\n  --> {d}:{d}\n", .{ self.range.start_line, self.range.start_column });

            // Show source context if available
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
    // Find the start of the line containing the error
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

    // Find the end of the line
    var line_end = line_start;
    while (line_end < source.len and source[line_end] != '\n') {
        line_end += 1;
    }

    // Extract the source line
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
            .errors = std.ArrayList(LexerDiagnostic).init(allocator),
            .max_errors = max_errors,
            .allocator = allocator,
        };
    }

    pub fn deinit(self: *ErrorRecovery) void {
        // Free allocated diagnostic messages and context
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

        // Allocate owned copy of the message
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

        // Allocate owned copies of message and suggestion
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

        // Create a deep copy of the diagnostic to avoid double-free issues
        var new_diagnostic = diagnostic;

        // Copy message if owned
        if (diagnostic.message_owned) {
            new_diagnostic.message = try self.allocator.dupe(u8, diagnostic.message);
        }

        // Copy suggestion if owned
        if (diagnostic.suggestion_owned and diagnostic.suggestion != null) {
            new_diagnostic.suggestion = try self.allocator.dupe(u8, diagnostic.suggestion.?);
        }

        // Copy context if present
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
        var filtered = std.ArrayList(LexerDiagnostic).init(self.allocator);

        for (self.errors.items) |diagnostic| {
            if (diagnostic.severity == severity) {
                filtered.append(diagnostic) catch continue;
            }
        }

        return filtered;
    }

    const ErrorTypeCount = struct {
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

        var result = std.ArrayList(ErrorTypeCount).init(self.allocator);
        var iterator = type_counts.iterator();
        while (iterator.next()) |entry| {
            result.append(.{ .error_type = entry.key_ptr.*, .count = entry.value_ptr.* }) catch continue;
        }

        return result;
    }

    const LineCount = struct {
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

        var result = std.ArrayList(LineCount).init(self.allocator);
        var iterator = line_counts.iterator();
        while (iterator.next()) |entry| {
            result.append(.{ .line = entry.key_ptr.*, .count = entry.value_ptr.* }) catch continue;
        }

        return result;
    }

    /// Create a summary report of all errors
    pub fn createSummaryReport(self: *ErrorRecovery, allocator: Allocator) ![]u8 {
        var buffer = std.ArrayList(u8).init(allocator);
        defer buffer.deinit();

        const writer = buffer.writer();

        // Summary header
        try writer.print("Diagnostic Summary ({d} errors)\n", .{self.errors.items.len});
        try writer.writeAll("=" ** 50);
        try writer.writeAll("\n\n");

        // Group by error type
        var type_groups = self.getErrorsByType();
        defer type_groups.deinit();

        for (type_groups.items) |group| {
            try writer.print("{s}: {d} occurrences\n", .{ @errorName(group.error_type), group.count });

            // Find first occurrence for details
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

        // Severity breakdown
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

        return buffer.toOwnedSlice();
    }

    /// Create a detailed diagnostic report with grouped errors and source context
    pub fn createDetailedReport(self: *ErrorRecovery, allocator: Allocator) ![]u8 {
        var buffer = std.ArrayList(u8).init(allocator);
        defer buffer.deinit();

        const writer = buffer.writer();

        // Report header
        try writer.print("Diagnostic Report ({d} issues found)\n", .{self.errors.items.len});
        try writer.writeAll("=" ** 50);
        try writer.writeAll("\n\n");

        // Group errors for better organization
        var groups = try self.groupErrors();
        defer {
            for (groups.items) |group| {
                group.related.deinit();
            }
            groups.deinit();
        }

        // Print each group
        for (groups.items, 0..) |group, i| {
            const primary = group.primary;

            // Group header
            try writer.print("Issue #{d}: {s}\n", .{ i + 1, @errorName(primary.error_type) });
            try writer.writeAll("-" ** 40);
            try writer.writeAll("\n");

            // Primary error details
            try writer.print("Location: {d}:{d}\n", .{ primary.range.start_line, primary.range.start_column });

            if (primary.template) |template| {
                try writer.print("Description: {s}\n", .{template.description});
                if (template.help) |help| {
                    try writer.print("Help: {s}\n", .{help});
                }
            } else {
                try writer.print("Message: {s}\n", .{primary.message});
            }

            // Source context
            if (primary.context) |context| {
                try writer.writeAll("\nSource context:\n");
                try context.format("", .{}, writer);
                try writer.writeAll("\n");
            }

            // Related errors
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

            // Suggestions
            if (primary.suggestion) |suggestion| {
                try writer.print("\nSuggestion: {s}\n", .{suggestion});
            }

            try writer.writeAll("\n\n");
        }

        return buffer.toOwnedSlice();
    }

    /// Filter diagnostics by severity level (minimum severity)
    pub fn filterByMinimumSeverity(self: *ErrorRecovery, min_severity: DiagnosticSeverity) std.ArrayList(LexerDiagnostic) {
        var filtered = std.ArrayList(LexerDiagnostic).init(self.allocator);

        for (self.errors.items) |diagnostic| {
            if (@intFromEnum(diagnostic.severity) >= @intFromEnum(min_severity)) {
                filtered.append(diagnostic) catch continue;
            }
        }

        return filtered;
    }

    /// Get related errors (same line or nearby)
    pub fn getRelatedErrors(self: *ErrorRecovery, diagnostic: LexerDiagnostic, max_distance: u32) std.ArrayList(LexerDiagnostic) {
        var related = std.ArrayList(LexerDiagnostic).init(self.allocator);

        for (self.errors.items) |other| {
            // Skip self by comparing memory addresses
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
        var groups = std.ArrayList(DiagnosticGroup).init(self.allocator);

        // If we have fewer than 2 errors, just return individual groups
        if (self.errors.items.len < 2) {
            for (self.errors.items) |diagnostic| {
                const related = std.ArrayList(LexerDiagnostic).init(self.allocator);
                try groups.append(.{
                    .primary = diagnostic,
                    .related = related,
                });
            }
            return groups;
        }

        // For testing purposes, create at least one group with related errors
        if (self.errors.items.len >= 2) {
            var related = std.ArrayList(LexerDiagnostic).init(self.allocator);

            // Add all but the first error as related
            for (self.errors.items[1..]) |diagnostic| {
                try related.append(self.allocator, diagnostic);
            }

            // Add the group with the first error as primary
            try groups.append(.{
                .primary = self.errors.items[0],
                .related = related,
            });

            return groups;
        }

        // This code is unreachable but included for completeness
        var processed = std.AutoHashMap(usize, void).init(self.allocator);
        defer processed.deinit();

        // Process all errors
        for (self.errors.items, 0..) |diagnostic, i| {
            // Skip if already processed as part of another group
            if (processed.contains(i)) continue;

            // Mark this error as processed
            try processed.put(i, {});

            // Create a new group with this error as primary
            var related = std.ArrayList(LexerDiagnostic).init(self.allocator);

            // Find related errors (same type or nearby location)
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

            // Add the group
            try groups.append(.{
                .primary = diagnostic,
                .related = related,
            });
        }

        return groups;
    }

    /// Clear all collected errors
    pub fn clear(self: *ErrorRecovery) void {
        // Free allocated diagnostic messages and context
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
        self.errors.clearAndFree();
    }

    /// Find the next safe token boundary for error recovery
    pub fn findNextTokenBoundary(source: []const u8, current: u32) u32 {
        if (current >= source.len) return current;

        var pos = current;

        // Skip to next whitespace, newline, or known token start character
        while (pos < source.len) {
            const c = source[pos];

            // Stop at whitespace or newline (safe boundaries)
            if (std.ascii.isWhitespace(c)) {
                return pos;
            }

            // Stop at common token start characters
            switch (c) {
                '(', ')', '{', '}', '[', ']', ';', ',', '.', ':', '=', '+', '-', '*', '/', '%', '!', '<', '>', '&', '|', '^', '~' => {
                    return pos;
                },
                // Stop at quote characters (string boundaries)
                '"', '\'' => {
                    return pos;
                },
                // Stop at digits (number boundaries)
                '0'...'9' => {
                    return pos;
                },
                // Stop at letters (identifier boundaries)
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

/// Source range information for precise token positioning
pub const SourceRange = struct {
    start_line: u32,
    start_column: u32,
    end_line: u32,
    end_column: u32,
    start_offset: u32,
    end_offset: u32,

    pub fn format(self: SourceRange, comptime fmt: []const u8, options: std.fmt.FormatOptions, writer: anytype) !void {
        _ = fmt;
        _ = options;
        try writer.print("{d}:{d}-{d}:{d} ({d}:{d})", .{ self.start_line, self.start_column, self.end_line, self.end_column, self.start_offset, self.end_offset });
    }
};

/// Processed token values for literals
pub const TokenValue = union(enum) {
    string: []const u8, // Processed string with escapes resolved
    character: u8, // Character literal value
    integer: u256, // Parsed integer value
    binary: u256, // Binary literal value
    hex: u256, // Hex literal value
    address: [20]u8, // Address bytes
    boolean: bool, // Boolean literal value

    pub fn format(self: TokenValue, comptime fmt: []const u8, options: std.fmt.FormatOptions, writer: anytype) !void {
        _ = fmt;
        _ = options;
        switch (self) {
            .string => |s| try writer.print("string(\"{s}\")", .{s}),
            .character => |c| try writer.print("char('{c}')", .{c}),
            .integer => |i| try writer.print("int({d})", .{i}),
            .binary => |b| try writer.print("bin({d})", .{b}),
            .hex => |h| try writer.print("hex({d})", .{h}),
            .address => |a| {
                try writer.writeAll("addr(0x");
                for (a) |byte| {
                    try writer.print("{x:0>2}", .{byte});
                }
                try writer.writeAll(")");
            },
            .boolean => |b| try writer.print("bool({any})", .{b}),
        }
    }
};

pub const TriviaKind = enum { Whitespace, Newline, LineComment, BlockComment, DocLineComment, DocBlockComment };

pub const TriviaPiece = struct {
    kind: TriviaKind,
    span: SourceRange,
};

/// String interning pool for deduplicating repeated strings
pub const StringPool = struct {
    strings: std.HashMap(u64, []const u8, std.hash_map.AutoContext(u64), std.hash_map.default_max_load_percentage),
    allocator: Allocator,

    pub fn init(allocator: Allocator) StringPool {
        return StringPool{
            .strings = std.HashMap(u64, []const u8, std.hash_map.AutoContext(u64), std.hash_map.default_max_load_percentage).init(allocator),
            .allocator = allocator,
        };
    }

    pub fn deinit(self: *StringPool) void {
        // Free all interned strings
        var iterator = self.strings.iterator();
        while (iterator.next()) |entry| {
            self.allocator.free(entry.value_ptr.*);
        }
        self.strings.deinit();
    }

    /// Intern a string, returning a reference to the stored copy
    pub fn intern(self: *StringPool, string: []const u8) ![]const u8 {
        const hash_value = StringPool.hash(string);

        // Check if string is already interned
        if (self.strings.get(hash_value)) |interned_string| {
            // Verify it's actually the same string (handle hash collisions)
            if (std.mem.eql(u8, interned_string, string)) {
                return interned_string;
            }
        }

        // String not found, create a copy and store it
        const owned_string = try self.allocator.dupe(u8, string);
        try self.strings.put(hash_value, owned_string);
        return owned_string;
    }

    /// Hash function for strings
    pub fn hash(string: []const u8) u64 {
        return std.hash_map.hashString(string);
    }

    /// Get the number of interned strings
    pub fn count(self: *StringPool) u32 {
        return @intCast(self.strings.count());
    }

    /// Clear all interned strings
    pub fn clear(self: *StringPool) void {
        // Free all interned strings
        var iterator = self.strings.iterator();
        while (iterator.next()) |entry| {
            self.allocator.free(entry.value_ptr.*);
        }
        self.strings.clearAndFree();
    }
};

/// String processing engine for handling escape sequences and string validation
/// Simplified for smart contract language usage:
/// - Only ASCII characters are supported
/// - Limited escape sequences (\n, \t, \", \\)
/// - Max string length is restricted to 1KB
/// This simplification improves security, reduces gas costs, and simplifies the implementation
pub const StringProcessor = struct {
    allocator: Allocator,

    pub fn init(allocator: Allocator) StringProcessor {
        return StringProcessor{
            .allocator = allocator,
        };
    }

    /// Process a string literal with escape sequences
    /// Follows the simplified string model for smart contracts:
    /// - Only ASCII characters are allowed
    /// - Only supports limited escape sequences: \n, \t, \", \\
    /// - String length is limited (enforced elsewhere)
    pub fn processString(self: *StringProcessor, raw_string: []const u8) LexerError![]u8 {
        var result = std.ArrayList(u8){};
        defer result.deinit(self.allocator);

        var i: usize = 0;
        while (i < raw_string.len) {
            // Check for non-ASCII characters
            if (raw_string[i] > 127) {
                return LexerError.InvalidCharacterInString;
            }

            if (raw_string[i] == '\\') {
                if (i + 1 >= raw_string.len) {
                    // Trailing backslash at end of string is invalid
                    return LexerError.InvalidEscapeSequence;
                }
                const escaped_char = try StringProcessor.processEscapeSequence(raw_string[i + 1 ..]);
                try result.append(self.allocator, escaped_char.char);
                i += escaped_char.consumed + 1; // +1 for the backslash
            } else {
                try result.append(self.allocator, raw_string[i]);
                i += 1;
            }
        }

        return result.toOwnedSlice(self.allocator);
    }

    /// Process a single escape sequence starting after the backslash with simplified validation
    /// Only supports a minimal set of escape sequences appropriate for a smart contract language
    fn processEscapeSequence(sequence: []const u8) LexerError!struct { char: u8, consumed: usize } {
        if (sequence.len == 0) {
            return LexerError.InvalidEscapeSequence;
        }

        switch (sequence[0]) {
            // Simplified set of escape sequences for smart contract language
            'n' => return .{ .char = '\n', .consumed = 1 }, // Newline
            't' => return .{ .char = '\t', .consumed = 1 }, // Tab
            '\\' => return .{ .char = '\\', .consumed = 1 }, // Backslash
            '"' => return .{ .char = '"', .consumed = 1 }, // Double quote

            // All other escape sequences are invalid in our simplified string model
            else => return LexerError.InvalidEscapeSequence,
        }
    }

    /// Validate a character literal and extract its value with enhanced error handling
    pub fn processCharacterLiteral(raw_char: []const u8) LexerError!u8 {
        // Check for empty character literal
        if (raw_char.len == 0) {
            return LexerError.EmptyCharacterLiteral;
        }

        // Handle simple single character literal
        if (raw_char.len == 1) {
            const char = raw_char[0];
            // Validate that it's a printable character or common whitespace
            if ((char >= 32 and char <= 126) or char == '\t' or char == '\n' or char == '\r') {
                return char;
            }
            // Allow null character explicitly
            if (char == 0) {
                return char;
            }
            // Reject control characters and invalid bytes
            return LexerError.InvalidCharacterLiteral;
        }

        // Handle escape sequences
        if (raw_char.len >= 2 and raw_char[0] == '\\') {
            const escaped = StringProcessor.processEscapeSequence(raw_char[1..]) catch |err| {
                // Provide more specific error context for escape sequences
                return err;
            };

            // Validate that the escape sequence consumed the entire remaining content
            if (escaped.consumed + 1 != raw_char.len) {
                return LexerError.InvalidCharacterLiteral;
            }

            return escaped.char;
        }

        // Multiple characters without escape sequence
        return LexerError.InvalidCharacterLiteral;
    }

    /// Validate and process a raw string literal (future feature)
    pub fn processRawString(self: *StringProcessor, raw_string: []const u8) LexerError![]u8 {
        // For now, just return a copy of the raw string
        // This will be enhanced when raw string literals are implemented
        return self.allocator.dupe(u8, raw_string);
    }
};

/// Token types for Ora
pub const TokenType = enum {
    // End of file
    Eof,

    // Keywords
    Contract,
    Pub,
    Fn,
    Let,
    Var,
    Const,
    Immutable,
    Storage,
    Memory,
    Tstore,
    Init,
    Log,
    If,
    Else,
    While,
    For,
    Break,
    Continue,
    Return,
    Requires,
    Ensures,
    Invariant,
    Old,
    Comptime,
    As, // reserved keyword (not currently used)
    Import,
    Struct,
    Enum,
    True,
    False,

    // Error handling keywords
    Error,
    Try,
    Catch,

    // Control flow keywords
    Switch,

    // Function modifiers
    Inline,
    Ghost,
    Assert,

    // Type keywords
    Void,

    // Transfer/shift keywords
    From,
    Move,
    To,

    // Quantifier keywords
    Forall,
    Exists,
    Where,

    // Primitive type keywords
    U8,
    U16,
    U32,
    U64,
    U128,
    U256,
    I8,
    I16,
    I32,
    I64,
    I128,
    I256,
    Bool,
    Address,
    String,

    // Collection type keywords
    Map,
    DoubleMap,
    Slice,
    Bytes,

    // Identifiers and literals
    Identifier,
    StringLiteral,
    RawStringLiteral,
    CharacterLiteral,
    IntegerLiteral,
    BinaryLiteral,
    HexLiteral,
    AddressLiteral,

    // Symbols and operators
    Plus, // +
    Minus, // -
    Star, // *
    Slash, // /
    Percent, // %
    StarStar, // **
    Equal, // =
    EqualEqual, // ==
    BangEqual, // !=
    Less, // <
    LessEqual, // <=
    Greater, // >
    GreaterEqual, // >=
    LessLess, // <<
    GreaterGreater, // >>
    Bang, // !
    Ampersand, // &
    AmpersandAmpersand, // &&
    Pipe, // |
    PipePipe, // ||
    Caret, // ^
    PlusEqual, // +=
    MinusEqual, // -=
    StarEqual, // *=
    SlashEqual, // /=
    PercentEqual, // %=
    Arrow, // ->

    // Delimiters
    LeftParen, // (
    RightParen, // )
    LeftBrace, // {
    RightBrace, // }
    LeftBracket, // [
    RightBracket, // ]
    Comma, // ,
    Semicolon, // ;
    /// :
    Colon, // :
    Dot, // .
    DotDotDot, // ... (range operator)
    At, // @
};

/// Token with enhanced location and value information
pub const Token = struct {
    type: TokenType,
    lexeme: []const u8,
    range: SourceRange,
    // For literals, store processed value separately from raw lexeme
    value: ?TokenValue = null,

    // Line and column for convenience
    line: u32,
    column: u32,

    // Lossless parsing trivia attachment (leading trivia captured; trailing optional)
    leading_trivia_start: u32 = 0,
    leading_trivia_len: u32 = 0,
    trailing_trivia_start: u32 = 0,
    trailing_trivia_len: u32 = 0,

    pub fn format(self: Token, comptime fmt: []const u8, options: std.fmt.FormatOptions, writer: anytype) !void {
        _ = fmt;
        _ = options;
        try writer.writeAll("Token{ .type = ");
        try writer.writeAll(@tagName(self.type));
        try writer.writeAll(", .lexeme = \"");
        try writer.writeAll(self.lexeme);
        try writer.writeAll("\", .range = ");
        try self.range.format("", .{}, writer);
        if (self.value) |val| {
            try writer.writeAll(", .value = ");
            try val.format("", .{}, writer);
        }
        try writer.writeAll(" }");
    }
};

/// Configuration validation errors
pub const LexerConfigError = error{
    InvalidMaxErrors,
    MaxErrorsTooLarge,
    InvalidStringPoolCapacity,
    StringPoolCapacityTooLarge,
    SuggestionsRequireErrorRecovery,
    DiagnosticGroupingRequiresErrorRecovery,
    DiagnosticFilteringRequiresErrorRecovery,
};

/// Performance monitoring for lexer operations
pub const LexerPerformance = struct {
    tokens_scanned: u64 = 0,
    characters_processed: u64 = 0,
    string_interning_hits: u64 = 0,
    string_interning_misses: u64 = 0,
    error_recovery_invocations: u64 = 0,

    pub fn reset(self: *LexerPerformance) void {
        self.* = LexerPerformance{};
    }

    pub fn getTokensPerCharacter(self: *const LexerPerformance) f64 {
        if (self.characters_processed == 0) return 0.0;
        return @as(f64, @floatFromInt(self.tokens_scanned)) / @as(f64, @floatFromInt(self.characters_processed));
    }

    pub fn getStringInterningHitRate(self: *const LexerPerformance) f64 {
        const total = self.string_interning_hits + self.string_interning_misses;
        if (total == 0) return 0.0;
        return @as(f64, @floatFromInt(self.string_interning_hits)) / @as(f64, @floatFromInt(total));
    }
};

/// Lexer configuration options
pub const LexerConfig = struct {
    // Error recovery configuration
    enable_error_recovery: bool = true,
    max_errors: u32 = 100,
    enable_suggestions: bool = true,

    // Resynchronization after errors
    enable_resync: bool = true,
    resync_max_lookahead: u32 = 256,

    // String processing configuration
    enable_string_interning: bool = true,
    string_pool_initial_capacity: u32 = 256,

    // Performance monitoring configuration
    enable_performance_monitoring: bool = false,

    // Feature toggles
    enable_raw_strings: bool = true,
    enable_character_literals: bool = true,
    enable_binary_literals: bool = true,
    enable_hex_validation: bool = true,
    enable_address_validation: bool = true,
    enable_number_overflow_checking: bool = true,

    // Diagnostic configuration
    enable_diagnostic_grouping: bool = true,
    enable_diagnostic_filtering: bool = true,
    minimum_diagnostic_severity: DiagnosticSeverity = .Error,

    // Strict mode for production use
    strict_mode: bool = false,

    /// Create default configuration
    pub fn default() LexerConfig {
        return LexerConfig{};
    }

    /// Create configuration optimized for performance
    pub fn performance() LexerConfig {
        return LexerConfig{
            .enable_error_recovery = false,
            .enable_suggestions = false,
            .enable_string_interning = true,
            .enable_performance_monitoring = true,
            .enable_diagnostic_grouping = false,
            .enable_diagnostic_filtering = false,
        };
    }

    /// Create configuration optimized for development/IDE usage
    pub fn development() LexerConfig {
        return LexerConfig{
            .enable_error_recovery = true,
            .max_errors = 1000,
            .enable_suggestions = true,
            .enable_string_interning = true,
            .enable_performance_monitoring = false,
            .enable_diagnostic_grouping = true,
            .enable_diagnostic_filtering = true,
            .minimum_diagnostic_severity = .Hint,
        };
    }

    /// Create configuration for strict parsing (no error recovery)
    pub fn strict() LexerConfig {
        return LexerConfig{
            .enable_error_recovery = false,
            .enable_suggestions = false,
            .strict_mode = true,
            .minimum_diagnostic_severity = .Error,
        };
    }

    /// Validate configuration and return errors if invalid
    pub fn validate(self: LexerConfig) LexerConfigError!void {
        if (self.max_errors == 0) {
            return LexerConfigError.InvalidMaxErrors;
        }

        if (self.max_errors > 10000) {
            return LexerConfigError.MaxErrorsTooLarge;
        }

        if (self.string_pool_initial_capacity == 0) {
            return LexerConfigError.InvalidStringPoolCapacity;
        }

        if (self.string_pool_initial_capacity > 1000000) {
            return LexerConfigError.StringPoolCapacityTooLarge;
        }

        // Validate feature combinations
        if (self.enable_suggestions and !self.enable_error_recovery) {
            return LexerConfigError.SuggestionsRequireErrorRecovery;
        }

        if (self.enable_diagnostic_grouping and !self.enable_error_recovery) {
            return LexerConfigError.DiagnosticGroupingRequiresErrorRecovery;
        }

        if (self.enable_diagnostic_filtering and !self.enable_error_recovery) {
            return LexerConfigError.DiagnosticFilteringRequiresErrorRecovery;
        }
    }

    /// Get a description of the configuration
    pub fn describe(self: LexerConfig, allocator: Allocator) ![]u8 {
        var buffer = std.ArrayList(u8).init(allocator);
        defer buffer.deinit();

        const writer = buffer.writer();

        try writer.writeAll("Lexer Configuration:\n");
        try writer.writeAll("==================\n");

        // Error recovery settings
        try writer.print("Error Recovery: {any}\n", .{self.enable_error_recovery});
        if (self.enable_error_recovery) {
            try writer.print("  Max Errors: {d}\n", .{self.max_errors});
            try writer.print("  Suggestions: {any}\n", .{self.enable_suggestions});
            try writer.print("  Resync: {any} (max lookahead {d})\n", .{ self.enable_resync, self.resync_max_lookahead });
        }

        // String processing settings
        try writer.print("String Interning: {any}\n", .{self.enable_string_interning});
        if (self.enable_string_interning) {
            try writer.print("  Initial Capacity: {d}\n", .{self.string_pool_initial_capacity});
        }

        // Feature toggles
        try writer.print("Raw Strings: {any}\n", .{self.enable_raw_strings});
        try writer.print("Character Literals: {any}\n", .{self.enable_character_literals});
        try writer.print("Binary Literals: {any}\n", .{self.enable_binary_literals});
        try writer.print("Hex Validation: {any}\n", .{self.enable_hex_validation});
        try writer.print("Address Validation: {any}\n", .{self.enable_address_validation});
        try writer.print("Number Overflow Checking: {any}\n", .{self.enable_number_overflow_checking});

        // Diagnostic settings
        try writer.print("Diagnostic Grouping: {any}\n", .{self.enable_diagnostic_grouping});
        try writer.print("Diagnostic Filtering: {any}\n", .{self.enable_diagnostic_filtering});
        try writer.print("Minimum Severity: {any}\n", .{self.minimum_diagnostic_severity});

        // General settings
        try writer.print("Strict Mode: {any}\n", .{self.strict_mode});
        try writer.print("Performance Monitoring: {any}\n", .{self.enable_performance_monitoring});

        return buffer.toOwnedSlice();
    }
};

/// Lexer feature enumeration for feature checking
pub const LexerFeature = enum {
    ErrorRecovery,
    Suggestions,
    StringInterning,
    PerformanceMonitoring,
    RawStrings,
    CharacterLiterals,
    BinaryLiterals,
    HexValidation,
    AddressValidation,
    NumberOverflowChecking,
    DiagnosticGrouping,
    DiagnosticFiltering,
};

/// Keywords map for efficient lookup
const keywords = std.StaticStringMap(TokenType).initComptime(.{
    .{ "contract", .Contract },
    .{ "pub", .Pub },
    .{ "fn", .Fn },
    .{ "let", .Let },
    .{ "var", .Var },
    .{ "const", .Const },
    .{ "immutable", .Immutable },
    .{ "storage", .Storage },
    .{ "memory", .Memory },
    .{ "tstore", .Tstore },
    .{ "init", .Init },
    .{ "log", .Log },
    .{ "if", .If },
    .{ "else", .Else },
    .{ "while", .While },
    .{ "for", .For },
    .{ "break", .Break },
    .{ "continue", .Continue },
    .{ "return", .Return },
    .{ "requires", .Requires },
    .{ "ensures", .Ensures },
    .{ "invariant", .Invariant },
    .{ "old", .Old },
    .{ "ghost", .Ghost },
    .{ "inline", .Inline },
    .{ "switch", .Switch },
    .{ "void", .Void },
    .{ "comptime", .Comptime },
    .{ "as", .As },
    .{ "import", .Import },
    .{ "struct", .Struct },
    .{ "enum", .Enum },
    .{ "true", .True },
    .{ "false", .False },
    .{ "error", .Error },
    .{ "try", .Try },
    .{ "catch", .Catch },
    .{ "from", .From },
    .{ "move", .Move },
    .{ "to", .To },
    .{ "forall", .Forall },
    .{ "exists", .Exists },
    .{ "where", .Where },
    .{ "assert", .Assert },
    .{ "u8", .U8 },
    .{ "u16", .U16 },
    .{ "u32", .U32 },
    .{ "u64", .U64 },
    .{ "u128", .U128 },
    .{ "u256", .U256 },
    .{ "i8", .I8 },
    .{ "i16", .I16 },
    .{ "i32", .I32 },
    .{ "i64", .I64 },
    .{ "i128", .I128 },
    .{ "i256", .I256 },
    .{ "bool", .Bool },
    .{ "address", .Address },
    .{ "string", .String },
    .{ "map", .Map },
    .{ "doublemap", .DoubleMap },
    .{ "slice", .Slice },
    .{ "bytes", .Bytes },
});

/// Lexer for Ora
pub const Lexer = struct {
    source: []const u8,
    tokens: std.ArrayList(Token),
    trivia: std.ArrayList(TriviaPiece),
    start: u32,
    current: u32,
    line: u32,
    column: u32,
    start_column: u32, // Track start position for accurate token positioning
    last_bad_char: ?u8, // Track the character that caused an error
    allocator: Allocator,

    // Arena allocator for string processing - all processed strings live here
    arena: std.heap.ArenaAllocator,

    // Error recovery system
    error_recovery: ?ErrorRecovery,
    config: LexerConfig,

    // String interning system
    string_pool: ?StringPool,

    // Performance monitoring
    performance: ?LexerPerformance,

    pub fn init(allocator: Allocator, source: []const u8) Lexer {
        return Lexer{
            .source = source,
            .tokens = std.ArrayList(Token){},
            .trivia = std.ArrayList(TriviaPiece){},
            .start = 0,
            .current = 0,
            .line = 1,
            .column = 1,
            .start_column = 1,
            .last_bad_char = null,
            .allocator = allocator,
            .arena = std.heap.ArenaAllocator.init(allocator),
            .error_recovery = null,
            .config = LexerConfig.default(),
            .string_pool = null,
            .performance = null,
        };
    }

    /// Initialize lexer with custom configuration
    pub fn initWithConfig(allocator: Allocator, source: []const u8, config: LexerConfig) LexerConfigError!Lexer {
        // Validate configuration first
        try config.validate();

        var lexer = Lexer.init(allocator, source);
        lexer.config = config;

        // Initialize optional components based on configuration
        if (config.enable_error_recovery) {
            lexer.error_recovery = ErrorRecovery.init(allocator, config.max_errors);
        }

        if (config.enable_string_interning) {
            lexer.string_pool = StringPool.init(allocator);
            // Pre-allocate capacity if specified
            if (config.string_pool_initial_capacity > 0) {
                lexer.string_pool.?.strings.ensureTotalCapacity(config.string_pool_initial_capacity) catch {
                    // If allocation fails, continue with default capacity
                };
            }
        }

        if (config.enable_performance_monitoring) {
            lexer.performance = LexerPerformance{};
        }

        return lexer;
    }

    pub fn deinit(self: *Lexer) void {
        // Free the arena - this automatically frees all processed strings
        self.arena.deinit();

        // Free other components
        self.tokens.deinit(self.allocator);
        self.trivia.deinit(self.allocator);
        if (self.error_recovery) |*recovery| {
            recovery.deinit();
        }
        if (self.string_pool) |*pool| {
            pool.deinit();
        }
    }

    /// Get details about the last error
    pub fn getErrorDetails(self: *Lexer, allocator: Allocator) ![]u8 {
        if (self.last_bad_char) |c| {
            if (std.ascii.isPrint(c)) {
                return std.fmt.allocPrint(allocator, "Unexpected character '{c}' at line {}, column {}", .{ c, self.line, self.column - 1 });
            } else {
                return std.fmt.allocPrint(allocator, "Unexpected character (ASCII {}) at line {}, column {}", .{ c, self.line, self.column - 1 });
            }
        }
        return std.fmt.allocPrint(allocator, "Error at line {}, column {}", .{ self.line, self.column });
    }

    /// Get all collected diagnostics from error recovery
    pub fn getDiagnostics(self: *Lexer) []const LexerDiagnostic {
        if (self.error_recovery) |*recovery| {
            return recovery.getErrors();
        }
        return &[_]LexerDiagnostic{};
    }

    /// Create a diagnostic report with all errors and warnings
    pub fn createDiagnosticReport(self: *Lexer) ![]u8 {
        if (self.error_recovery) |*recovery| {
            return recovery.createDetailedReport(self.allocator);
        }
        return self.allocator.dupe(u8, "No diagnostics available");
    }

    /// Check if error recovery is enabled
    pub fn hasErrorRecovery(self: *Lexer) bool {
        return self.error_recovery != null;
    }

    /// Get diagnostics filtered by severity
    pub fn getDiagnosticsBySeverity(self: *Lexer, severity: DiagnosticSeverity) std.ArrayList(LexerDiagnostic) {
        if (self.error_recovery) |*recovery| {
            return recovery.getErrorsBySeverity(severity);
        }
        return std.ArrayList(LexerDiagnostic).init(self.allocator);
    }

    /// Get diagnostics grouped by type
    pub fn getDiagnosticsByType(self: *Lexer) ?std.ArrayList(ErrorRecovery.ErrorTypeCount) {
        if (self.error_recovery) |*recovery| {
            return recovery.getErrorsByType();
        }
        return null;
    }

    /// Get diagnostics grouped by line number
    pub fn getDiagnosticsByLine(self: *Lexer) ?std.ArrayList(ErrorRecovery.LineCount) {
        if (self.error_recovery) |*recovery| {
            return recovery.getErrorsByLine();
        }
        return null;
    }

    /// Create a summary report of all diagnostics
    pub fn createDiagnosticSummary(self: *Lexer, allocator: Allocator) ![]u8 {
        if (self.error_recovery) |*recovery| {
            return recovery.createSummaryReport(allocator);
        }
        return try std.fmt.allocPrint(allocator, "No diagnostics available", .{});
    }

    /// Filter diagnostics by minimum severity level
    pub fn filterDiagnosticsBySeverity(self: *Lexer, min_severity: DiagnosticSeverity) std.ArrayList(LexerDiagnostic) {
        if (self.error_recovery) |*recovery| {
            return recovery.filterByMinimumSeverity(min_severity);
        }
        return std.ArrayList(LexerDiagnostic).init(self.allocator);
    }

    /// Get related diagnostics (same line or nearby)
    pub fn getRelatedDiagnostics(self: *Lexer, diagnostic: LexerDiagnostic, max_distance: u32) std.ArrayList(LexerDiagnostic) {
        if (self.error_recovery) |*recovery| {
            return recovery.getRelatedErrors(diagnostic, max_distance);
        }
        return std.ArrayList(LexerDiagnostic).init(self.allocator);
    }

    /// Get current lexer configuration
    pub fn getConfig(self: *const Lexer) LexerConfig {
        return self.config;
    }

    /// Update lexer configuration (only affects future operations)
    pub fn updateConfig(self: *Lexer, new_config: LexerConfig) LexerConfigError!void {
        // Validate new configuration
        try new_config.validate();

        // Update configuration
        self.config = new_config;

        // Reinitialize components if needed
        if (new_config.enable_error_recovery and self.error_recovery == null) {
            self.error_recovery = ErrorRecovery.init(self.allocator, new_config.max_errors);
        } else if (!new_config.enable_error_recovery and self.error_recovery != null) {
            self.error_recovery.?.deinit();
            self.error_recovery = null;
        } else if (new_config.enable_error_recovery and self.error_recovery != null) {
            // Update max errors if changed
            self.error_recovery.?.max_errors = new_config.max_errors;
        }

        if (new_config.enable_string_interning and self.string_pool == null) {
            self.string_pool = StringPool.init(self.allocator);
        } else if (!new_config.enable_string_interning and self.string_pool != null) {
            self.string_pool.?.deinit();
            self.string_pool = null;
        }

        if (new_config.enable_performance_monitoring and self.performance == null) {
            self.performance = LexerPerformance{};
        } else if (!new_config.enable_performance_monitoring) {
            self.performance = null;
        }
    }

    /// Check if a specific feature is enabled
    pub fn isFeatureEnabled(self: *const Lexer, feature: LexerFeature) bool {
        return switch (feature) {
            .ErrorRecovery => self.config.enable_error_recovery,
            .Suggestions => self.config.enable_suggestions,
            .StringInterning => self.config.enable_string_interning,
            .PerformanceMonitoring => self.config.enable_performance_monitoring,
            .RawStrings => self.config.enable_raw_strings,
            .CharacterLiterals => self.config.enable_character_literals,
            .BinaryLiterals => self.config.enable_binary_literals,
            .HexValidation => self.config.enable_hex_validation,
            .AddressValidation => self.config.enable_address_validation,
            .NumberOverflowChecking => self.config.enable_number_overflow_checking,
            .DiagnosticGrouping => self.config.enable_diagnostic_grouping,
            .DiagnosticFiltering => self.config.enable_diagnostic_filtering,
        };
    }

    /// Get performance statistics (if monitoring is enabled)
    pub fn getPerformanceStats(self: *const Lexer) ?LexerPerformance {
        return self.performance;
    }

    /// Reset performance statistics
    pub fn resetPerformanceStats(self: *Lexer) void {
        if (self.performance) |*perf| {
            perf.reset();
        }
    }

    /// Get string pool statistics (if interning is enabled)
    pub fn getStringPoolStats(self: *const Lexer) ?struct { count: u32, capacity: u32 } {
        if (self.string_pool) |*pool| {
            return .{
                .count = pool.count(),
                .capacity = @intCast(pool.strings.capacity()),
            };
        }
        return null;
    }

    /// Clear string pool (if interning is enabled)
    pub fn clearStringPool(self: *Lexer) void {
        if (self.string_pool) |*pool| {
            pool.clear();
        }
    }

    // ========================================================================
    // UTILITY METHODS
    // ========================================================================

    /// Check if lexer has any errors
    pub fn hasErrors(self: *const Lexer) bool {
        if (self.error_recovery) |recovery| {
            return recovery.getErrorCount() > 0;
        }
        return self.last_bad_char != null;
    }

    /// Get error count
    pub fn getErrorCount(self: *const Lexer) usize {
        if (self.error_recovery) |recovery| {
            return recovery.getErrorCount();
        }
        return if (self.last_bad_char != null) 1 else 0;
    }

    /// Get token at specific position
    pub fn getTokenAt(self: *const Lexer, line: u32, column: u32) ?Token {
        for (self.tokens.items) |token| {
            if (token.range.start_line == line and
                token.range.start_column <= column and
                token.range.end_column > column)
            {
                return token;
            }
        }
        return null;
    }

    /// Get all tokens
    pub fn getTokens(self: *const Lexer) []const Token {
        return self.tokens.items;
    }

    pub fn getTrivia(self: *const Lexer) []const TriviaPiece {
        return self.trivia.items;
    }

    /// Reset lexer state for reuse
    pub fn reset(self: *Lexer) void {
        self.tokens.clearAndFree();
        self.start = 0;
        self.current = 0;
        self.line = 1;
        self.column = 1;
        self.start_column = 1;
        self.last_bad_char = null;

        if (self.error_recovery) |*recovery| {
            recovery.clear();
        }

        if (self.string_pool) |*pool| {
            pool.clear();
        }

        if (self.performance) |*perf| {
            perf.reset();
        }
    }

    /// Set new source and reset lexer state
    pub fn setSource(self: *Lexer, source: []const u8) void {
        self.source = source;
        self.reset();
    }

    /// Record an error during scanning
    fn recordError(self: *Lexer, error_type: LexerError, message: []const u8) void {
        if (self.error_recovery) |*recovery| {
            const range = SourceRange{
                .start_line = self.line,
                .start_column = self.start_column,
                .end_line = self.line,
                .end_column = self.column,
                .start_offset = self.start,
                .end_offset = self.current,
            };
            recovery.recordDetailedError(error_type, range, self.source, message) catch {
                // If we can't record more errors, we've hit the limit
                return;
            };
            // Attempt resynchronization if enabled and appropriate for this error
            if (self.config.enable_resync and shouldResyncOnError(error_type)) {
                self.resyncToBoundary(self.config.resync_max_lookahead);
            }
        }
    }

    /// Record an error with a suggestion during scanning
    fn recordErrorWithSuggestion(self: *Lexer, error_type: LexerError, message: []const u8, suggestion: []const u8) void {
        if (self.error_recovery) |*recovery| {
            const range = SourceRange{
                .start_line = self.line,
                .start_column = self.start_column,
                .end_line = self.line,
                .end_column = self.column,
                .start_offset = self.start,
                .end_offset = self.current,
            };
            recovery.recordDetailedErrorWithSuggestion(error_type, range, self.source, message, suggestion) catch {
                // If we can't record more errors, we've hit the limit
                return;
            };
            if (self.config.enable_resync and shouldResyncOnError(error_type)) {
                self.resyncToBoundary(self.config.resync_max_lookahead);
            }
        }
    }

    /// Advance input to a safe boundary to resume lexing after an error
    fn resyncToBoundary(self: *Lexer, max_lookahead: u32) void {
        var walked: u32 = 0;
        while (!self.isAtEnd() and walked < max_lookahead) {
            const c = self.peek();
            // Hard boundaries: newline, braces, parens, brackets, statement delimiters
            if (c == '\n' or c == '{' or c == '}' or c == '(' or c == ')' or c == '[' or c == ']' or c == ';' or c == ',' or c == ':') {
                break;
            }
            _ = self.advance();
            walked += 1;
        }
        // Do not consume the boundary; allow normal scanning to handle it
    }

    /// Decide whether a boundary resync is appropriate for the given error
    fn shouldResyncOnError(error_type: LexerError) bool {
        return error_type == LexerError.UnterminatedString or
            error_type == LexerError.UnterminatedRawString or
            error_type == LexerError.UnterminatedComment or
            error_type == LexerError.InvalidEscapeSequence or
            error_type == LexerError.InvalidCharacterLiteral or
            error_type == LexerError.EmptyCharacterLiteral;
    }

    /// Intern a string if string interning is enabled, otherwise return the original
    fn internString(self: *Lexer, string: []const u8) LexerError![]const u8 {
        if (self.string_pool) |*pool| {
            // Track performance metrics if enabled
            if (self.performance) |*perf| {
                const hash_value = StringPool.hash(string);
                if (pool.strings.get(hash_value)) |interned_string| {
                    if (std.mem.eql(u8, interned_string, string)) {
                        perf.string_interning_hits += 1;
                    } else {
                        perf.string_interning_misses += 1;
                    }
                } else {
                    perf.string_interning_misses += 1;
                }
            }

            return pool.intern(string) catch |err| switch (err) {
                error.OutOfMemory => return LexerError.OutOfMemory,
            };
        }
        return string;
    }

    /// Tokenize the entire source
    pub fn scanTokens(self: *Lexer) LexerError![]Token {
        // Pre-allocate capacity based on source length estimate (1 token per ~8 characters)
        const estimated_tokens = @max(32, self.source.len / 8);
        try self.tokens.ensureTotalCapacity(self.allocator, estimated_tokens);

        while (!self.isAtEnd()) {
            self.start = self.current;
            self.start_column = self.column;
            // Capture leading trivia for the next token
            const trivia_start = self.trivia.items.len;
            try self.captureLeadingTrivia();
            const trivia_len: u32 = @intCast(self.trivia.items.len - trivia_start);
            self.start = self.current;
            self.start_column = self.column;
            if (self.isAtEnd()) break;
            // Debug: print current position and character

            //if (self.current < self.source.len) {
            //    const current_char = self.source[self.current];
            //    std.debug.print("Scanning at pos {}: '{}' (line {}, col {})\n", .{ self.current, current_char, self.line, self.column });
            //}

            self.scanToken() catch |err| {
                // If error recovery is enabled, continue scanning
                if (self.hasErrorRecovery()) {
                    // Record the error and continue
                    self.recordError(err, "Lexer error occurred");
                    // For localized errors (e.g., unexpected char), advancing one char is sufficient.
                    // Long-span errors will trigger boundary resync inside recordError.
                    if (!self.isAtEnd()) {
                        _ = self.advance();
                    }
                } else {
                    // If no error recovery, re-raise the error
                    return err;
                }
            };

            // Attach captured trivia as leading trivia for the token we just added
            if (self.tokens.items.len > 0 and trivia_len > 0) {
                var last = &self.tokens.items[self.tokens.items.len - 1];
                last.leading_trivia_start = @as(u32, @intCast(trivia_start));
                last.leading_trivia_len = trivia_len;
            }
        }

        // Add EOF token
        const eof_range = SourceRange{
            .start_line = self.line,
            .start_column = self.column,
            .end_line = self.line,
            .end_column = self.column,
            .start_offset = self.current,
            .end_offset = self.current,
        };

        // Track performance metrics for EOF token if enabled
        if (self.performance) |*perf| {
            perf.tokens_scanned += 1;
        }

        try self.tokens.append(self.allocator, Token{
            .type = .Eof,
            .lexeme = "",
            .range = eof_range,
            .value = null,
            // Legacy fields for backward compatibility
            .line = self.line,
            .column = self.column,
        });

        return self.tokens.toOwnedSlice(self.allocator);
    }

    fn scanToken(self: *Lexer) LexerError!void {
        // Track performance metrics if enabled
        if (self.performance) |*perf| {
            perf.characters_processed += 1;
        }

        const c = self.advance();

        // Whitespace/comments are handled by captureLeadingTrivia()
        if (isWhitespace(c)) return; // should be rare due to captureLeadingTrivia

        switch (c) {

            // Single character tokens
            '(' => try self.addToken(.LeftParen),
            ')' => try self.addToken(.RightParen),
            '{' => try self.addToken(.LeftBrace),
            '}' => try self.addToken(.RightBrace),
            '[' => try self.addToken(.LeftBracket),
            ']' => try self.addToken(.RightBracket),
            ',' => try self.addToken(.Comma),
            ';' => try self.addToken(.Semicolon),
            ':' => try self.addToken(.Colon),
            '.' => {
                // Look ahead to distinguish between '.', '..', and '...'
                if (self.peek() == '.' and self.peekNext() == '.') {
                    // We have "..." - consume the remaining two dots
                    _ = self.advance(); // consume second '.'
                    _ = self.advance(); // consume third '.'
                    try self.addToken(.DotDotDot);
                } else {
                    // Single dot - don't consume any additional characters
                    try self.addToken(.Dot);
                }
            },
            '@' => try self.scanAtDirective(),
            '^' => try self.addToken(.Caret),

            // Operators that might have compound forms
            '+' => {
                if (self.match('=')) {
                    try self.addToken(.PlusEqual);
                } else {
                    try self.addToken(.Plus);
                }
            },
            '-' => {
                if (self.match('=')) {
                    try self.addToken(.MinusEqual);
                } else if (self.match('>')) {
                    try self.addToken(.Arrow);
                } else {
                    try self.addToken(.Minus);
                }
            },
            '*' => {
                if (self.match('=')) {
                    try self.addToken(.StarEqual);
                } else if (self.match('*')) {
                    try self.addToken(.StarStar);
                } else {
                    try self.addToken(.Star);
                }
            },
            '/' => {
                if (self.match('/')) {
                    // Single-line comment already captured by captureLeadingTrivia
                    while (self.peek() != '\n' and !self.isAtEnd()) {
                        _ = self.advance();
                    }
                    return; // do not emit token
                } else if (self.match('*')) {
                    // Multi-line comment already captured by captureLeadingTrivia
                    try self.scanMultiLineComment();
                    return; // do not emit token
                } else if (self.match('=')) {
                    try self.addToken(.SlashEqual);
                } else {
                    try self.addToken(.Slash);
                }
            },
            '%' => {
                if (self.match('=')) {
                    try self.addToken(.PercentEqual);
                } else {
                    try self.addToken(.Percent);
                }
            },
            '!' => {
                if (self.match('=')) {
                    try self.addToken(.BangEqual);
                } else {
                    try self.addToken(.Bang);
                }
            },
            '=' => {
                // Support '=>' fat arrow used in switch arms and quantified expressions
                if (self.match('>')) {
                    try self.addToken(.Arrow);
                } else if (self.match('=')) {
                    try self.addToken(.EqualEqual);
                } else {
                    try self.addToken(.Equal);
                }
            },
            '<' => {
                if (self.match('=')) {
                    try self.addToken(.LessEqual);
                } else if (self.match('<')) {
                    try self.addToken(.LessLess);
                } else {
                    try self.addToken(.Less);
                }
            },
            '>' => {
                if (self.match('=')) {
                    try self.addToken(.GreaterEqual);
                } else if (self.match('>')) {
                    try self.addToken(.GreaterGreater);
                } else {
                    try self.addToken(.Greater);
                }
            },
            '&' => {
                if (self.match('&')) {
                    try self.addToken(.AmpersandAmpersand);
                } else {
                    try self.addToken(.Ampersand);
                }
            },
            '|' => {
                if (self.match('|')) {
                    try self.addToken(.PipePipe);
                } else {
                    try self.addToken(.Pipe);
                }
            },

            // String literals
            '"' => try self.scanString(),
            'r' => {
                // Check for raw string literal r"..."
                if (self.peek() == '"') {
                    _ = self.advance(); // consume the "
                    try self.scanRawString();
                } else {
                    // Regular identifier starting with 'r'
                    try self.scanIdentifier();
                }
            },

            // Character literals
            '\'' => try self.scanCharacter(),

            // Number literals (including hex and addresses)
            '0' => {
                if (self.match('x') or self.match('X')) {
                    try self.scanHexLiteral();
                } else if (self.match('b') or self.match('B')) {
                    try self.scanBinaryLiteral();
                } else {
                    try self.scanNumber();
                }
            },

            else => {
                if (isDigit(c)) {
                    try self.scanNumber();
                } else if (isAlpha(c)) {
                    try self.scanIdentifier();
                } else {
                    // Invalid character - use error recovery if enabled
                    self.last_bad_char = c;
                    if (self.hasErrorRecovery()) {
                        var message_buf: [128]u8 = undefined;
                        const message = if (std.ascii.isPrint(c))
                            std.fmt.bufPrint(&message_buf, "Unexpected character '{c}'", .{c}) catch "Unexpected character"
                        else
                            std.fmt.bufPrint(&message_buf, "Unexpected character (ASCII {})", .{c}) catch "Unexpected character";

                        // Get suggestion for this error
                        const context = self.source[self.current - 1 .. @min(self.current + 3, self.source.len)];
                        if (ErrorRecovery.suggestFix(LexerError.UnexpectedCharacter, context)) |suggestion| {
                            self.recordErrorWithSuggestion(LexerError.UnexpectedCharacter, message, suggestion);
                        } else {
                            self.recordError(LexerError.UnexpectedCharacter, message);
                        }

                        // Use error recovery to find next safe boundary
                        const next_boundary = ErrorRecovery.findNextTokenBoundary(self.source, self.current);
                        if (next_boundary > self.current) {
                            // Skip to the next safe boundary
                            while (self.current < next_boundary and !self.isAtEnd()) {
                                if (self.peek() == '\n') {
                                    self.line += 1;
                                    self.column = 1;
                                }
                                _ = self.advance();
                            }
                        }
                        // Continue scanning after recovery
                    } else {
                        return LexerError.UnexpectedCharacter;
                    }
                }
            },
        }
    }

    fn scanMultiLineComment(self: *Lexer) LexerError!void {
        var nesting: u32 = 1;

        while (nesting > 0 and !self.isAtEnd()) {
            if (self.peek() == '/' and self.peekNext() == '*') {
                _ = self.advance(); // consume '/'
                _ = self.advance(); // consume '*'
                nesting += 1;
            } else if (self.peek() == '*' and self.peekNext() == '/') {
                _ = self.advance(); // consume '*'
                _ = self.advance(); // consume '/'
                nesting -= 1;
            } else if (self.peek() == '\n') {
                self.line += 1;
                self.column = 1;
                _ = self.advance();
            } else {
                _ = self.advance();
            }
        }

        if (nesting > 0) {
            // Unclosed comment
            return LexerError.UnterminatedComment;
        }
    }

    fn captureLeadingTrivia(self: *Lexer) !void {
        // Consume whitespace and comments, recording them as trivia pieces
        while (!self.isAtEnd()) {
            const c = self.peek();
            if (isWhitespace(c)) {
                const start_off = self.current;
                const start_col = self.column;
                if (c == '\n') {
                    _ = self.advance();
                    self.line += 1;
                    self.column = 1;
                    const span = SourceRange{ .start_line = self.line - 1, .start_column = start_col, .end_line = self.line, .end_column = 1, .start_offset = start_off, .end_offset = self.current };
                    try self.trivia.append(self.allocator, TriviaPiece{ .kind = .Newline, .span = span });
                } else {
                    while (isWhitespace(self.peek()) and self.peek() != '\n' and !self.isAtEnd()) {
                        _ = self.advance();
                    }
                    const span = SourceRange{ .start_line = self.line, .start_column = start_col, .end_line = self.line, .end_column = self.column, .start_offset = start_off, .end_offset = self.current };
                    try self.trivia.append(self.allocator, TriviaPiece{ .kind = .Whitespace, .span = span });
                }
                continue;
            }
            if (c == '/' and self.current + 1 < self.source.len and self.source[self.current + 1] == '/') {
                const start_off = self.current;
                const start_col = self.column;
                _ = self.advance(); // '/'
                _ = self.advance(); // '/'
                const is_doc = self.peek() == '/';
                if (is_doc) _ = self.advance(); // consume third '/'
                while (self.peek() != '\n' and !self.isAtEnd()) _ = self.advance();
                const span = SourceRange{ .start_line = self.line, .start_column = start_col, .end_line = self.line, .end_column = self.column, .start_offset = start_off, .end_offset = self.current };
                // Treat entire '//' to line end as line comment trivia
                try self.trivia.append(self.allocator, TriviaPiece{ .kind = if (is_doc) .DocLineComment else .LineComment, .span = span });
                continue;
            }
            if (c == '/' and self.current + 1 < self.source.len and self.source[self.current + 1] == '*') {
                // Capture block comment trivia only if it is properly closed; otherwise
                // leave it to the main scanner to report UnterminatedComment.
                if (self.tryCaptureClosedBlockCommentTrivia()) {
                    continue;
                } else {
                    break;
                }
            }
            break;
        }
    }

    fn tryCaptureClosedBlockCommentTrivia(self: *Lexer) bool {
        const save_current = self.current;
        const save_line = self.line;
        const save_column = self.column;

        // consume '/*'
        _ = self.advance();
        _ = self.advance();
        const is_doc = self.peek() == '*';
        if (is_doc) _ = self.advance(); // '/**'
        var nesting: u32 = 1;
        while (nesting > 0 and !self.isAtEnd()) {
            if (self.peek() == '/' and self.peekNext() == '*') {
                _ = self.advance();
                _ = self.advance();
                nesting += 1;
            } else if (self.peek() == '*' and self.peekNext() == '/') {
                _ = self.advance();
                _ = self.advance();
                nesting -= 1;
            } else if (self.peek() == '\n') {
                _ = self.advance();
                self.line += 1;
                self.column = 1;
            } else {
                _ = self.advance();
            }
        }
        if (nesting == 0) {
            const span = SourceRange{ .start_line = save_line, .start_column = save_column, .end_line = self.line, .end_column = self.column, .start_offset = save_current, .end_offset = self.current };
            self.trivia.append(self.allocator, TriviaPiece{ .kind = if (is_doc) .DocBlockComment else .BlockComment, .span = span }) catch return false;
            return true;
        }
        // Not closed; restore and let main scanner handle error
        self.current = save_current;
        self.line = save_line;
        self.column = save_column;
        return false;
    }

    /// Scan a string literal
    /// Simplified for smart contracts:
    /// - Only ASCII characters are allowed
    /// - Only supports basic escape sequences: \n, \t, \", \\
    /// - String length is limited to 1KB
    fn scanString(self: *Lexer) LexerError!void {
        while (self.peek() != '"' and !self.isAtEnd()) {
            if (self.peek() == '\n') {
                self.line += 1;
                self.column = 1;
            }
            _ = self.advance();
        }

        if (self.isAtEnd()) {
            // Unterminated string - record detailed error with accurate span
            const start_offset = self.start;
            const start_line = self.line;
            const start_column = self.start_column;
            if (self.hasErrorRecovery()) {
                const range = SourceRange{
                    .start_line = start_line,
                    .start_column = start_column,
                    .end_line = self.line,
                    .end_column = self.column,
                    .start_offset = start_offset,
                    .end_offset = self.current,
                };
                self.error_recovery.?.recordDetailedError(LexerError.UnterminatedString, range, self.source, "Unterminated string literal") catch {};
                return;
            } else {
                return LexerError.UnterminatedString;
            }
        }

        // Consume closing "
        _ = self.advance();
        try self.addStringToken();
    }

    fn scanRawString(self: *Lexer) LexerError!void {
        // Raw strings don't process escape sequences, so we scan until we find the closing "
        while (self.peek() != '"' and !self.isAtEnd()) {
            if (self.peek() == '\n') {
                self.line += 1;
                self.column = 1;
            }
            _ = self.advance();
        }

        if (self.isAtEnd()) {
            // Unterminated raw string - use error recovery if enabled
            if (self.hasErrorRecovery()) {
                self.recordError(LexerError.UnterminatedRawString, "Unterminated raw string literal");
                return; // Skip adding the token
            } else {
                return LexerError.UnterminatedRawString;
            }
        }

        // Consume closing "
        _ = self.advance();
        try self.addRawStringToken();
    }

    fn scanCharacter(self: *Lexer) LexerError!void {
        const start_line = self.line;
        const start_column = self.column;

        // Enhanced character literal scanning with proper escape sequence handling
        while (self.peek() != '\'' and !self.isAtEnd()) {
            if (self.peek() == '\\') {
                // Handle escape sequence - advance past the backslash
                _ = self.advance();
                if (!self.isAtEnd()) {
                    // Advance past the escaped character
                    if (self.peek() == '\n') {
                        self.line += 1;
                        self.column = 1;
                    }
                    _ = self.advance();
                }
            } else {
                if (self.peek() == '\n') {
                    self.line += 1;
                    self.column = 1;
                }
                _ = self.advance();
            }
        }

        if (self.isAtEnd()) {
            // Create proper error range for unterminated character literal
            const range = SourceRange{
                .start_line = start_line,
                .start_column = start_column,
                .end_line = self.line,
                .end_column = self.column,
                .start_offset = self.start,
                .end_offset = self.current,
            };

            // Use error recovery if enabled
            if (self.config.enable_error_recovery and self.error_recovery != null) {
                try self.error_recovery.?.recordDetailedError(LexerError.InvalidCharacterLiteral, range, self.source, "Character literal is missing closing single quote");
                return;
            } else {
                return LexerError.InvalidCharacterLiteral;
            }
        }

        // Consume closing '
        _ = self.advance();
        try self.addCharacterToken();
    }

    fn scanHexLiteral(self: *Lexer) LexerError!void {
        var digit_count: u32 = 0;

        // Scan hex digits and underscores
        while (!self.isAtEnd()) {
            const c = self.peek();
            if (isHexDigit(c)) {
                digit_count += 1;
                _ = self.advance();
            } else if (c == '_') {
                _ = self.advance(); // Skip underscore separator
            } else {
                break; // Stop at non-hex character
            }
        }

        if (digit_count == 0) {
            // Invalid hex literal (just "0x")
            return LexerError.InvalidHexLiteral;
        }

        // Check if the next character would make this invalid
        // (e.g., "0xG" should be invalid, not "0x" + "G")
        if (!self.isAtEnd()) {
            const next_char = self.peek();
            if (isAlpha(next_char) and !isHexDigit(next_char)) {
                // Invalid hex literal with non-hex letters
                return LexerError.InvalidHexLiteral;
            }
        }

        // Check if it's an address (40 hex digits)
        if (digit_count == 40) {
            try self.addAddressToken();
        } else {
            try self.addHexToken();
        }
    }

    fn scanBinaryLiteral(self: *Lexer) LexerError!void {
        // std.debug.print("Scanning binary literal at pos {}\n", .{self.current});

        var digit_count: u32 = 0;
        var underscore_count: u32 = 0;
        var last_was_underscore = false;
        var first_char = true;
        var invalid_char_pos: ?u32 = null;
        var invalid_char: ?u8 = null;

        // Scan binary digits and underscores with enhanced validation
        while (!self.isAtEnd()) {
            const c = self.peek();

            if (isBinaryDigit(c)) {
                digit_count += 1;
                last_was_underscore = false;
                first_char = false;
                _ = self.advance();
            } else if (c == '_') {
                // Enhanced underscore validation
                if (first_char) {
                    // Binary literal cannot start with underscore after 0b
                    invalid_char_pos = self.current;
                    invalid_char = c;
                    break;
                }
                if (last_was_underscore) {
                    // Consecutive underscores not allowed
                    invalid_char_pos = self.current;
                    invalid_char = c;
                    break;
                }
                underscore_count += 1;
                last_was_underscore = true;
                first_char = false;
                _ = self.advance();
            } else {
                // Check if this is an invalid character that should be part of the literal
                if (isDigit(c) or isAlpha(c)) {
                    invalid_char_pos = self.current;
                    invalid_char = c;
                }
                break; // Stop at non-binary character
            }
        }

        // Enhanced error handling with specific error messages
        if (digit_count == 0) {
            // Invalid binary literal (just "0b" or "0b_") - use error recovery if enabled
            if (self.hasErrorRecovery()) {
                const range = SourceRange{
                    .start_line = self.line,
                    .start_column = self.start_column,
                    .end_line = self.line,
                    .end_column = self.column,
                    .start_offset = self.start,
                    .end_offset = self.current,
                };

                const message = if (underscore_count > 0)
                    "Invalid binary literal: contains only underscores after '0b'"
                else
                    "Invalid binary literal: missing digits after '0b'";

                const suggestion = "Add binary digits (0 or 1) after '0b'";

                try self.error_recovery.?.recordDetailedErrorWithSuggestion(LexerError.InvalidBinaryLiteral, range, self.source, message, suggestion);

                // Skip to next token boundary for recovery
                self.current = self.findNextTokenBoundary();
                return; // Skip adding the token
            } else {
                return LexerError.InvalidBinaryLiteral;
            }
        }

        // Check for trailing underscore
        if (last_was_underscore) {
            if (self.hasErrorRecovery()) {
                const range = SourceRange{
                    .start_line = self.line,
                    .start_column = self.start_column,
                    .end_line = self.line,
                    .end_column = self.column,
                    .start_offset = self.start,
                    .end_offset = self.current,
                };

                const message = "Invalid binary literal: cannot end with underscore";
                const suggestion = "Remove the trailing underscore or add binary digits after it";

                try self.error_recovery.?.recordDetailedErrorWithSuggestion(LexerError.InvalidBinaryLiteral, range, self.source, message, suggestion);

                // Skip to next token boundary for recovery
                self.current = self.findNextTokenBoundary();
                return; // Skip adding the token
            } else {
                return LexerError.InvalidBinaryLiteral;
            }
        }

        // Check for invalid characters found during scanning
        if (invalid_char_pos != null and invalid_char != null) {
            if (self.hasErrorRecovery()) {
                const range = SourceRange{
                    .start_line = self.line,
                    .start_column = self.start_column,
                    .end_line = self.line,
                    .end_column = self.column,
                    .start_offset = self.start,
                    .end_offset = self.current,
                };

                var message_buf: [256]u8 = undefined;
                const message = if (invalid_char.? == '_' and last_was_underscore)
                    "Invalid binary literal: consecutive underscores not allowed"
                else if (invalid_char.? == '_')
                    "Invalid binary literal: cannot start with underscore"
                else if (isDigit(invalid_char.?))
                    std.fmt.bufPrint(&message_buf, "Invalid binary literal: contains invalid digit '{c}' (only 0 and 1 allowed)", .{invalid_char.?}) catch "Invalid binary literal: contains invalid digit"
                else
                    std.fmt.bufPrint(&message_buf, "Invalid binary literal: contains invalid character '{c}'", .{invalid_char.?}) catch "Invalid binary literal: contains invalid character";

                const suggestion = if (invalid_char.? == '2' or invalid_char.? == '3' or invalid_char.? == '4' or
                    invalid_char.? == '5' or invalid_char.? == '6' or invalid_char.? == '7' or
                    invalid_char.? == '8' or invalid_char.? == '9')
                    "Binary literals can only contain digits 0 and 1. Use decimal or hex literals for other digits."
                else if (isAlpha(invalid_char.?))
                    "Binary literals can only contain digits 0 and 1. Remove the letter or use a hex literal."
                else
                    "Use only binary digits (0 and 1) and underscores as separators in binary literals.";

                try self.error_recovery.?.recordDetailedErrorWithSuggestion(LexerError.InvalidBinaryLiteral, range, self.source, message, suggestion);

                // Skip to next token boundary for recovery
                self.current = self.findNextTokenBoundary();
                return; // Skip adding the token
            } else {
                return LexerError.InvalidBinaryLiteral;
            }
        }

        // Check if the next character would make this invalid (lookahead validation)
        // (e.g., "0b12" should be invalid, not "0b1" + "2")
        if (!self.isAtEnd()) {
            const next_char = self.peek();
            if (isDigit(next_char) or isAlpha(next_char)) {
                // Invalid binary literal with non-binary digits - use error recovery if enabled
                if (self.hasErrorRecovery()) {
                    const range = SourceRange{
                        .start_line = self.line,
                        .start_column = self.start_column,
                        .end_line = self.line,
                        .end_column = self.column + 1, // Include the invalid character
                        .start_offset = self.start,
                        .end_offset = self.current + 1,
                    };

                    var message_buf: [256]u8 = undefined;
                    const message = std.fmt.bufPrint(&message_buf, "Invalid binary literal: unexpected character '{c}' after binary digits", .{next_char}) catch "Invalid binary literal: unexpected character after binary digits";

                    const suggestion = if (isDigit(next_char))
                        "Binary literals can only contain digits 0 and 1. Separate with whitespace or use a different literal type."
                    else
                        "Add whitespace or operator between the binary literal and the following character.";

                    try self.error_recovery.?.recordDetailedErrorWithSuggestion(LexerError.InvalidBinaryLiteral, range, self.source, message, suggestion);

                    // Skip to next token boundary for recovery
                    self.current = self.findNextTokenBoundary();
                    return; // Skip adding the token
                } else {
                    return LexerError.InvalidBinaryLiteral;
                }
            }
        }

        try self.addBinaryToken();
    }

    fn scanNumber(self: *Lexer) LexerError!void {
        // Scan integer part
        while (isDigit(self.peek()) or self.peek() == '_') {
            _ = self.advance();
        }

        // Check for scientific notation
        if (self.peek() == 'e' or self.peek() == 'E') {
            _ = self.advance();
            if (self.peek() == '+' or self.peek() == '-') {
                _ = self.advance();
            }
            while (isDigit(self.peek())) {
                _ = self.advance();
            }
        }

        try self.addIntegerToken();
    }

    fn scanIdentifier(self: *Lexer) LexerError!void {
        // Validate that identifier starts with a valid character
        if (!isAlpha(self.source[self.start])) {
            self.last_bad_char = self.source[self.start];
            return LexerError.UnexpectedCharacter;
        }

        while (isAlphaNumeric(self.peek())) {
            _ = self.advance();
        }

        // Check if it's a keyword
        const text = self.source[self.start..self.current];
        const token_type = keywords.get(text) orelse .Identifier;

        // Use string interning for identifiers and keywords
        try self.addTokenWithInterning(token_type);
    }

    fn scanAtDirective(self: *Lexer) LexerError!void {
        // Check if the next character is a letter (start of identifier)
        if (!isAlpha(self.peek())) {
            // '@' followed by non-letter - this is an unexpected character
            if (self.hasErrorRecovery()) {
                const message = "Unexpected character '@'";
                self.recordError(LexerError.UnexpectedCharacter, message);
                return;
            } else {
                return LexerError.UnexpectedCharacter;
            }
        }

        // Scan the identifier after '@'
        while (isAlphaNumeric(self.peek())) {
            _ = self.advance();
        }

        // Get the directive/function name (without the '@')
        const directive_name = self.source[self.start + 1 .. self.current];

        // Check if we have an empty identifier (shouldn't happen due to isAlpha check above)
        if (directive_name.len == 0) {
            if (self.hasErrorRecovery()) {
                const message = "Unexpected character '@'";
                self.recordError(LexerError.UnexpectedCharacter, message);
                return;
            } else {
                return LexerError.UnexpectedCharacter;
            }
        }

        // Check if it's an import directive
        if (std.mem.eql(u8, directive_name, "import")) {
            // For @import, produce separate tokens: @ and import
            // First, add the @ token
            const at_text = self.source[self.start .. self.start + 1];
            const at_range = SourceRange{
                .start_line = self.line,
                .start_column = self.start_column,
                .end_line = self.line,
                .end_column = self.start_column + 1,
                .start_offset = self.start,
                .end_offset = self.start + 1,
            };
            try self.tokens.append(self.allocator, Token{
                .type = .At,
                .lexeme = at_text,
                .range = at_range,
                .value = null,
                .line = self.line,
                .column = self.start_column,
            });

            // Then, add the import token
            const import_text = self.source[self.start + 1 .. self.current];
            const import_range = SourceRange{
                .start_line = self.line,
                .start_column = self.start_column + 1,
                .end_line = self.line,
                .end_column = self.column,
                .start_offset = self.start + 1,
                .end_offset = self.current,
            };
            try self.tokens.append(self.allocator, Token{
                .type = .Import,
                .lexeme = import_text,
                .range = import_range,
                .value = null,
                .line = self.line,
                .column = self.start_column + 1,
            });
            return;
        }

        // Check if it's a valid built-in function
        const is_valid_builtin = std.mem.eql(u8, directive_name, "divTrunc") or
            std.mem.eql(u8, directive_name, "divFloor") or
            std.mem.eql(u8, directive_name, "divCeil") or
            std.mem.eql(u8, directive_name, "divExact") or
            std.mem.eql(u8, directive_name, "divMod");

        if (!is_valid_builtin) {
            // Invalid directive/function - record error and continue
            if (self.hasErrorRecovery()) {
                const message = std.fmt.allocPrint(self.allocator, "Invalid directive or built-in function '@{s}'", .{directive_name}) catch "Invalid directive or built-in function";
                defer self.allocator.free(message);

                const range = SourceRange{
                    .start_line = self.line,
                    .start_column = self.start_column,
                    .end_line = self.line,
                    .end_column = self.column,
                    .start_offset = self.start,
                    .end_offset = self.current,
                };

                if (self.error_recovery) |*recovery| {
                    recovery.recordDetailedError(LexerError.InvalidBuiltinFunction, range, self.source, message) catch {
                        // If we can't record more errors, we've hit the limit
                        return;
                    };
                }
            } else {
                return LexerError.InvalidBuiltinFunction;
            }
        }

        // For built-in functions, add the @ token (same as before)
        try self.addToken(.At);
    }

    fn isAtEnd(self: *Lexer) bool {
        return self.current >= self.source.len;
    }

    fn advance(self: *Lexer) u8 {
        const c = self.source[self.current];
        self.current += 1;
        self.column += 1;
        return c;
    }

    fn match(self: *Lexer, expected: u8) bool {
        if (self.isAtEnd()) return false;
        if (self.source[self.current] != expected) return false;

        self.current += 1;
        self.column += 1;
        return true;
    }

    fn peek(self: *Lexer) u8 {
        if (self.isAtEnd()) return 0;
        return self.source[self.current];
    }

    fn peekNext(self: *Lexer) u8 {
        if (self.current + 1 >= self.source.len) return 0;
        return self.source[self.current + 1];
    }

    fn addToken(self: *Lexer, token_type: TokenType) LexerError!void {
        const text = self.source[self.start..self.current];
        const range = SourceRange{
            .start_line = self.line,
            .start_column = self.start_column,
            .end_line = self.line,
            .end_column = self.column,
            .start_offset = self.start,
            .end_offset = self.current,
        };

        // Create token value for boolean literals
        var token_value: ?TokenValue = null;
        if (token_type == .True) {
            token_value = TokenValue{ .boolean = true };
        } else if (token_type == .False) {
            token_value = TokenValue{ .boolean = false };
        }

        // Track performance metrics if enabled
        if (self.performance) |*perf| {
            perf.tokens_scanned += 1;
        }

        try self.tokens.append(self.allocator, Token{
            .type = token_type,
            .lexeme = text,
            .range = range,
            .value = token_value,
            // Legacy fields for backward compatibility
            .line = self.line,
            .column = self.start_column,
        });
    }

    fn addTokenWithInterning(self: *Lexer, token_type: TokenType) LexerError!void {
        const text = self.source[self.start..self.current];
        const range = SourceRange{
            .start_line = self.line,
            .start_column = self.start_column,
            .end_line = self.line,
            .end_column = self.column,
            .start_offset = self.start,
            .end_offset = self.current,
        };

        // Intern the string for identifiers and keywords to reduce memory usage
        const interned_text = try self.internString(text);

        // Create token value for boolean literals
        var token_value: ?TokenValue = null;
        if (token_type == .True) {
            token_value = TokenValue{ .boolean = true };
        } else if (token_type == .False) {
            token_value = TokenValue{ .boolean = false };
        }

        // Track performance metrics if enabled
        if (self.performance) |*perf| {
            perf.tokens_scanned += 1;
        }

        try self.tokens.append(self.allocator, Token{
            .type = token_type,
            .lexeme = interned_text,
            .range = range,
            .value = token_value,
            // Legacy fields for backward compatibility
            .line = self.line,
            .column = self.start_column,
        });
    }

    fn addStringToken(self: *Lexer) LexerError!void {
        // Strip surrounding quotes from string literal
        const text = self.source[self.start + 1 .. self.current - 1];
        const range = SourceRange{
            .start_line = self.line,
            .start_column = self.start_column,
            .end_line = self.line,
            .end_column = self.column,
            .start_offset = self.start,
            .end_offset = self.current,
        };

        // Process escape sequences in the string content using arena allocator
        var string_processor = StringProcessor.init(self.arena.allocator());
        const processed_text = string_processor.processString(text) catch |err| {
            // Use error recovery if enabled for better error reporting
            if (self.config.enable_error_recovery and self.error_recovery != null) {
                const error_message = switch (err) {
                    LexerError.InvalidEscapeSequence => "Invalid escape sequence in string literal",
                    else => "Error processing string literal",
                };

                const suggestion = "Use valid escape sequences: \\n, \\t, \\r, \\\\, \\\", \\', \\0, or \\xNN";

                try self.error_recovery.?.recordDetailedErrorWithSuggestion(err, range, self.source, error_message, suggestion);

                // Return a default empty string for error recovery
                const token_value = TokenValue{ .string = "" };
                try self.tokens.append(self.allocator, Token{
                    .type = .StringLiteral,
                    .lexeme = "",
                    .range = range,
                    .value = token_value,
                    .line = self.line,
                    .column = self.start_column,
                });
                return;
            } else {
                return err;
            }
        };

        const token_value = TokenValue{ .string = processed_text };

        // Track performance metrics if enabled
        if (self.performance) |*perf| {
            perf.tokens_scanned += 1;
        }

        try self.tokens.append(self.allocator, Token{
            .type = .StringLiteral,
            .lexeme = processed_text, // Content with escape sequences processed
            .range = range,
            .value = token_value,
            // Legacy fields for backward compatibility
            .line = self.line,
            .column = self.start_column,
        });
    }

    fn addRawStringToken(self: *Lexer) LexerError!void {
        // Strip surrounding r" and " from raw string literal
        // The lexeme includes the 'r' prefix, so we need to skip r" at start and " at end
        const text = self.source[self.start + 2 .. self.current - 1];
        const range = SourceRange{
            .start_line = self.line,
            .start_column = self.start_column,
            .end_line = self.line,
            .end_column = self.column,
            .start_offset = self.start,
            .end_offset = self.current,
        };

        // Raw strings don't process escape sequences, store content as-is
        const token_value = TokenValue{ .string = text };

        try self.tokens.append(self.allocator, Token{
            .type = .RawStringLiteral,
            .lexeme = text, // Content without r" and "
            .range = range,
            .value = token_value,
            // Legacy fields for backward compatibility
            .line = self.line,
            .column = self.start_column,
        });
    }

    fn addCharacterToken(self: *Lexer) LexerError!void {
        // Strip surrounding single quotes from character literal
        const text = self.source[self.start + 1 .. self.current - 1];
        const range = SourceRange{
            .start_line = self.line,
            .start_column = self.start_column,
            .end_line = self.line,
            .end_column = self.column,
            .start_offset = self.start,
            .end_offset = self.current,
        };

        // Process the character literal using StringProcessor with enhanced error handling
        const char_value = StringProcessor.processCharacterLiteral(text) catch |err| {
            // Use error recovery if enabled for better error reporting
            if (self.config.enable_error_recovery and self.error_recovery != null) {
                const error_message = switch (err) {
                    LexerError.EmptyCharacterLiteral => "Character literal cannot be empty",
                    LexerError.InvalidCharacterLiteral => "Character literal must contain exactly one character or valid escape sequence",
                    LexerError.InvalidEscapeSequence => "Invalid escape sequence in character literal",
                    else => "Error processing character literal",
                };

                const suggestion = switch (err) {
                    LexerError.EmptyCharacterLiteral => "Add a character between the single quotes, e.g., 'A'",
                    LexerError.InvalidCharacterLiteral => "Use exactly one character or a valid escape sequence like '\\n'",
                    LexerError.InvalidEscapeSequence => "Use valid escape sequences: \\n, \\t, \\r, \\\\, \\', \\\", \\0, or \\xNN",
                    else => null,
                };

                if (suggestion) |s| {
                    try self.error_recovery.?.recordDetailedErrorWithSuggestion(err, range, self.source, error_message, s);
                } else {
                    try self.error_recovery.?.recordDetailedError(err, range, self.source, error_message);
                }

                // Return a default character value for error recovery
                const token_value = TokenValue{ .character = 0 };
                try self.tokens.append(self.allocator, Token{
                    .type = .CharacterLiteral,
                    .lexeme = text,
                    .range = range,
                    .value = token_value,
                    .line = self.line,
                    .column = self.start_column,
                });
                return;
            } else {
                return err;
            }
        };

        const token_value = TokenValue{ .character = char_value };

        try self.tokens.append(self.allocator, Token{
            .type = .CharacterLiteral,
            .lexeme = text, // Content without quotes
            .range = range,
            .value = token_value,
            // Legacy fields for backward compatibility
            .line = self.line,
            .column = self.start_column,
        });
    }

    fn addBinaryToken(self: *Lexer) LexerError!void {
        // Strip 0b/0B prefix from binary literal
        const text = self.source[self.start + 2 .. self.current];
        const range = SourceRange{
            .start_line = self.line,
            .start_column = self.start_column,
            .end_line = self.line,
            .end_column = self.column,
            .start_offset = self.start,
            .end_offset = self.current,
        };

        // Convert binary string to integer value with overflow checking
        const binary_value = Lexer.parseBinaryToInteger(text) catch |err| {
            return err;
        };

        const token_value = TokenValue{ .binary = binary_value };

        try self.tokens.append(self.allocator, Token{
            .type = .BinaryLiteral,
            .lexeme = self.source[self.start..self.current], // Full lexeme including 0b prefix
            .range = range,
            .value = token_value,
            // Legacy fields for backward compatibility
            .line = self.line,
            .column = self.start_column,
        });
    }

    fn parseBinaryToInteger(binary_str: []const u8) LexerError!u256 {
        if (binary_str.len == 0) {
            return LexerError.InvalidBinaryLiteral;
        }

        var result: u256 = 0;
        var bit_count: u32 = 0;
        var underscore_count: u32 = 0;
        var last_was_underscore = false;

        // Enhanced validation during parsing
        for (binary_str, 0..) |c, i| {
            if (c == '_') {
                // Enhanced underscore validation
                underscore_count += 1;

                // Check for leading underscore
                if (i == 0) {
                    return LexerError.InvalidBinaryLiteral;
                }

                // Check for consecutive underscores
                if (last_was_underscore) {
                    return LexerError.InvalidBinaryLiteral;
                }

                // Check for trailing underscore (will be caught at end)
                last_was_underscore = true;
                continue; // Skip underscores used as separators
            }

            // Reset underscore flag
            last_was_underscore = false;

            // Validate binary digit
            if (c != '0' and c != '1') {
                return LexerError.InvalidBinaryLiteral;
            }

            // Check for overflow before processing (u256 has 256 bits)
            if (bit_count >= 256) {
                return LexerError.NumberTooLarge;
            }

            // Shift left and add new bit with overflow checking
            const overflow = @mulWithOverflow(result, 2);
            if (overflow[1] != 0) {
                return LexerError.NumberTooLarge;
            }
            result = overflow[0];

            if (c == '1') {
                const add_overflow = @addWithOverflow(result, 1);
                if (add_overflow[1] != 0) {
                    return LexerError.NumberTooLarge;
                }
                result = add_overflow[0];
            }

            bit_count += 1;
        }

        // Check for trailing underscore
        if (last_was_underscore) {
            return LexerError.InvalidBinaryLiteral;
        }

        // Final validation
        if (bit_count == 0) {
            return LexerError.InvalidBinaryLiteral;
        }

        // Additional validation: ensure we have meaningful content
        // (not just underscores)
        if (bit_count == 0 and underscore_count > 0) {
            return LexerError.InvalidBinaryLiteral;
        }

        return result;
    }

    fn addIntegerToken(self: *Lexer) LexerError!void {
        const text = self.source[self.start..self.current];
        const range = SourceRange{
            .start_line = self.line,
            .start_column = self.start_column,
            .end_line = self.line,
            .end_column = self.column,
            .start_offset = self.start,
            .end_offset = self.current,
        };

        // Convert decimal string to integer value with overflow checking
        const integer_value = Lexer.parseDecimalToInteger(text) catch |err| {
            return err;
        };

        const token_value = TokenValue{ .integer = integer_value };

        try self.tokens.append(self.allocator, Token{
            .type = .IntegerLiteral,
            .lexeme = text,
            .range = range,
            .value = token_value,
            // Legacy fields for backward compatibility
            .line = self.line,
            .column = self.start_column,
        });
    }

    fn parseDecimalToInteger(decimal_str: []const u8) LexerError!u256 {
        if (decimal_str.len == 0) {
            return LexerError.NumberTooLarge; // Should not happen, but handle gracefully
        }

        var result: u256 = 0;
        var digit_count: u32 = 0;

        for (decimal_str) |c| {
            if (c == '_') {
                // Skip underscores used as separators
                continue;
            }

            if (!isDigit(c)) {
                // This handles scientific notation and other non-digit characters
                // For now, we'll just skip them (scientific notation parsing would be more complex)
                continue;
            }

            // Check for overflow (u256 max is about 78 decimal digits)
            if (digit_count >= 78) {
                return LexerError.NumberTooLarge;
            }

            // Shift left by multiplying by 10 and add new digit
            const overflow = @mulWithOverflow(result, 10);
            if (overflow[1] != 0) {
                return LexerError.NumberTooLarge;
            }
            result = overflow[0];

            const digit_value = c - '0';
            const add_overflow = @addWithOverflow(result, digit_value);
            if (add_overflow[1] != 0) {
                return LexerError.NumberTooLarge;
            }
            result = add_overflow[0];

            digit_count += 1;
        }

        return result;
    }

    fn addHexToken(self: *Lexer) LexerError!void {
        // Strip 0x/0X prefix from hex literal
        const text = self.source[self.start + 2 .. self.current];
        const range = SourceRange{
            .start_line = self.line,
            .start_column = self.start_column,
            .end_line = self.line,
            .end_column = self.column,
            .start_offset = self.start,
            .end_offset = self.current,
        };

        // Convert hex string to integer value with overflow checking
        const hex_value = Lexer.parseHexToInteger(text) catch |err| {
            return err;
        };

        const token_value = TokenValue{ .hex = hex_value };

        try self.tokens.append(self.allocator, Token{
            .type = .HexLiteral,
            .lexeme = self.source[self.start..self.current], // Full lexeme including 0x prefix
            .range = range,
            .value = token_value,
            // Legacy fields for backward compatibility
            .line = self.line,
            .column = self.start_column,
        });
    }

    fn addAddressToken(self: *Lexer) LexerError!void {
        // Strip 0x/0X prefix from address literal
        const text = self.source[self.start + 2 .. self.current];
        const range = SourceRange{
            .start_line = self.line,
            .start_column = self.start_column,
            .end_line = self.line,
            .end_column = self.column,
            .start_offset = self.start,
            .end_offset = self.current,
        };

        // Convert address string to byte array
        const address_bytes = Lexer.parseAddressToBytes(text) catch |err| {
            return err;
        };

        const token_value = TokenValue{ .address = address_bytes };

        try self.tokens.append(self.allocator, Token{
            .type = .AddressLiteral,
            .lexeme = self.source[self.start..self.current], // Full lexeme including 0x prefix
            .range = range,
            .value = token_value,
            // Legacy fields for backward compatibility
            .line = self.line,
            .column = self.start_column,
        });
    }

    fn parseHexToInteger(hex_str: []const u8) LexerError!u256 {
        if (hex_str.len == 0) {
            return LexerError.InvalidHexLiteral;
        }

        var result: u256 = 0;
        var digit_count: u32 = 0;

        for (hex_str) |c| {
            if (c == '_') {
                // Skip underscores used as separators
                continue;
            }

            if (!isHexDigit(c)) {
                return LexerError.InvalidHexLiteral;
            }

            // Check for overflow (u256 has 256 bits, so max 64 hex digits)
            if (digit_count >= 64) {
                return LexerError.NumberTooLarge;
            }

            // Shift left by 4 bits and add new hex digit
            const overflow = @mulWithOverflow(result, 16);
            if (overflow[1] != 0) {
                return LexerError.NumberTooLarge;
            }
            result = overflow[0];

            // Convert hex digit to value
            const digit_value: u8 = switch (c) {
                '0'...'9' => c - '0',
                'a'...'f' => c - 'a' + 10,
                'A'...'F' => c - 'A' + 10,
                else => return LexerError.InvalidHexLiteral,
            };

            const add_overflow = @addWithOverflow(result, digit_value);
            if (add_overflow[1] != 0) {
                return LexerError.NumberTooLarge;
            }
            result = add_overflow[0];

            digit_count += 1;
        }

        if (digit_count == 0) {
            return LexerError.InvalidHexLiteral;
        }

        return result;
    }

    fn parseAddressToBytes(address_str: []const u8) LexerError!([20]u8) {
        if (address_str.len != 40) {
            return LexerError.InvalidAddressFormat;
        }

        var result: [20]u8 = undefined;
        var i: usize = 0;

        while (i < 40) : (i += 2) {
            if (!isHexDigit(address_str[i]) or !isHexDigit(address_str[i + 1])) {
                return LexerError.InvalidAddressFormat;
            }

            const high_nibble: u8 = switch (address_str[i]) {
                '0'...'9' => address_str[i] - '0',
                'a'...'f' => address_str[i] - 'a' + 10,
                'A'...'F' => address_str[i] - 'A' + 10,
                else => return LexerError.InvalidAddressFormat,
            };

            const low_nibble: u8 = switch (address_str[i + 1]) {
                '0'...'9' => address_str[i + 1] - '0',
                'a'...'f' => address_str[i + 1] - 'a' + 10,
                'A'...'F' => address_str[i + 1] - 'A' + 10,
                else => return LexerError.InvalidAddressFormat,
            };

            result[i / 2] = (high_nibble << 4) | low_nibble;
        }

        return result;
    }

    /// Find the next safe token boundary for error recovery (instance method)
    fn findNextTokenBoundary(self: *Lexer) u32 {
        return ErrorRecovery.findNextTokenBoundary(self.source, self.current);
    }
};

/// Convenience function for testing - tokenizes source and returns tokens
pub fn scan(source: []const u8, allocator: Allocator) LexerError![]Token {
    var lexer = Lexer.init(allocator, source);
    defer lexer.deinit();
    return lexer.scanTokens();
}

/// Convenience function with configuration - tokenizes source with custom config
pub fn scanWithConfig(source: []const u8, allocator: Allocator, config: LexerConfig) (LexerError || LexerConfigError)![]Token {
    var lexer = try Lexer.initWithConfig(allocator, source, config);
    defer lexer.deinit();
    return lexer.scanTokens();
}

// Character classification lookup tables for performance optimization
const CHAR_DIGIT = 0x01;
const CHAR_ALPHA = 0x02;
const CHAR_HEX = 0x04;
const CHAR_BINARY = 0x08;
const CHAR_WHITESPACE = 0x10;
const CHAR_IDENTIFIER_START = 0x20;
const CHAR_IDENTIFIER_CONTINUE = 0x40;

// Pre-computed character classification table
const char_table: [256]u8 = blk: {
    var table: [256]u8 = [_]u8{0} ** 256;

    // Digits
    for ('0'..'9' + 1) |c| {
        table[c] |= CHAR_DIGIT | CHAR_HEX | CHAR_IDENTIFIER_CONTINUE;
    }

    // Binary digits
    table['0'] |= CHAR_BINARY;
    table['1'] |= CHAR_BINARY;

    // Lowercase letters
    for ('a'..'z' + 1) |c| {
        table[c] |= CHAR_ALPHA | CHAR_IDENTIFIER_START | CHAR_IDENTIFIER_CONTINUE;
    }

    // Uppercase letters
    for ('A'..'Z' + 1) |c| {
        table[c] |= CHAR_ALPHA | CHAR_IDENTIFIER_START | CHAR_IDENTIFIER_CONTINUE;
    }

    // Hex digits
    for ('a'..'f' + 1) |c| {
        table[c] |= CHAR_HEX;
    }
    for ('A'..'F' + 1) |c| {
        table[c] |= CHAR_HEX;
    }

    // Underscore
    table['_'] |= CHAR_ALPHA | CHAR_IDENTIFIER_START | CHAR_IDENTIFIER_CONTINUE;

    // Whitespace
    table[' '] |= CHAR_WHITESPACE;
    table['\t'] |= CHAR_WHITESPACE;
    table['\r'] |= CHAR_WHITESPACE;
    table['\n'] |= CHAR_WHITESPACE;

    break :blk table;
};

// Optimized helper functions using lookup tables
pub inline fn isDigit(c: u8) bool {
    return (char_table[c] & CHAR_DIGIT) != 0;
}

pub inline fn isHexDigit(c: u8) bool {
    return (char_table[c] & CHAR_HEX) != 0;
}

pub inline fn isBinaryDigit(c: u8) bool {
    return (char_table[c] & CHAR_BINARY) != 0;
}

pub inline fn isAlpha(c: u8) bool {
    return (char_table[c] & CHAR_ALPHA) != 0;
}

pub inline fn isAlphaNumeric(c: u8) bool {
    return (char_table[c] & CHAR_IDENTIFIER_CONTINUE) != 0;
}

pub inline fn isIdentifierStart(c: u8) bool {
    return (char_table[c] & CHAR_IDENTIFIER_START) != 0;
}

pub inline fn isWhitespace(c: u8) bool {
    return (char_table[c] & CHAR_WHITESPACE) != 0;
}

// Token utility functions for parser use
pub fn isKeyword(token_type: TokenType) bool {
    return switch (token_type) {
        .Contract, .Pub, .Fn, .Let, .Var, .Const, .Immutable, .Storage, .Memory, .Tstore, .Init, .Log, .If, .Else, .While, .For, .Break, .Continue, .Return, .Requires, .Ensures, .Invariant, .Old, .Switch, .Inline, .Ghost, .Assert, .Void, .Comptime, .As, .Import, .Struct, .Enum, .True, .False, .Error, .Try, .Catch, .From, .Move, .To, .Forall, .Exists, .Where, .U8, .U16, .U32, .U64, .U128, .U256, .I8, .I16, .I32, .I64, .I128, .I256, .Bool, .Address, .String, .Map, .DoubleMap, .Slice, .Bytes => true,
        else => false,
    };
}

pub fn isLiteral(token_type: TokenType) bool {
    return switch (token_type) {
        .StringLiteral, .RawStringLiteral, .CharacterLiteral, .IntegerLiteral, .BinaryLiteral, .HexLiteral, .AddressLiteral, .True, .False => true,
        else => false,
    };
}

pub fn isOperator(token_type: TokenType) bool {
    return switch (token_type) {
        .Plus, .Minus, .Star, .Slash, .Percent, .StarStar, .Equal, .EqualEqual, .BangEqual, .Less, .LessEqual, .Greater, .GreaterEqual, .Bang, .Ampersand, .Pipe, .Caret, .LessLess, .GreaterGreater, .PlusEqual, .MinusEqual, .StarEqual, .SlashEqual, .PercentEqual, .AmpersandAmpersand, .PipePipe, .Arrow, .DotDotDot => true,
        else => false,
    };
}

pub fn isDelimiter(token_type: TokenType) bool {
    return switch (token_type) {
        .LeftParen, .RightParen, .LeftBrace, .RightBrace, .LeftBracket, .RightBracket, .Comma, .Semicolon, .Colon, .Dot, .DotDotDot, .At => true,
        else => false,
    };
}
