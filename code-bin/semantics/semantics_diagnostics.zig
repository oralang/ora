const std = @import("std");
pub const ast = @import("../ast.zig");
const semantics_errors = @import("semantics_errors.zig");
const semantics_memory_safety = @import("semantics_memory_safety.zig");

// Forward declaration for SemanticAnalyzer
const SemanticAnalyzer = @import("semantics_core.zig").SemanticAnalyzer;

/// Enhanced diagnostic information
pub const EnhancedDiagnostic = struct {
    base_diagnostic: semantics_errors.Diagnostic,
    diagnostic_id: u32,
    category: DiagnosticCategory,
    source_context: SourceContext,
    suggestions: []Suggestion,
    related_diagnostics: []u32,
    fix_suggestions: []FixSuggestion,
    metadata: DiagnosticMetadata,

    pub const DiagnosticCategory = enum {
        Syntax,
        Semantic,
        Type,
        Memory,
        Performance,
        Style,
        Security,
        Compatibility,
    };

    pub const SourceContext = struct {
        file_path: ?[]const u8,
        surrounding_lines: [][]const u8,
        line_numbers: []u32,
        highlighted_ranges: []HighlightRange,

        pub const HighlightRange = struct {
            start_line: u32,
            start_column: u32,
            end_line: u32,
            end_column: u32,
            highlight_type: HighlightType,

            pub const HighlightType = enum {
                Error,
                Warning,
                Info,
                Suggestion,
            };
        };
    };

    pub const Suggestion = struct {
        message: []const u8,
        suggestion_type: SuggestionType,
        confidence: f32, // 0.0 to 1.0

        pub const SuggestionType = enum {
            Alternative,
            BestPractice,
            Performance,
            Security,
            Readability,
        };
    };

    pub const FixSuggestion = struct {
        description: []const u8,
        replacement_text: []const u8,
        replacement_span: ast.SourceSpan,
        fix_type: FixType,
        applies_to_all: bool,

        pub const FixType = enum {
            Replace,
            Insert,
            Delete,
            Reorder,
        };
    };

    pub const DiagnosticMetadata = struct {
        timestamp: i64,
        analysis_phase: []const u8,
        node_type: ?std.meta.Tag(ast.AstNode),
        rule_id: ?[]const u8,
        help_url: ?[]const u8,
    };

    pub fn deinit(self: *EnhancedDiagnostic, allocator: std.mem.Allocator) void {
        allocator.free(self.suggestions);
        allocator.free(self.related_diagnostics);
        allocator.free(self.fix_suggestions);
        if (self.source_context.surrounding_lines.len > 0) {
            for (self.source_context.surrounding_lines) |line| {
                allocator.free(line);
            }
            allocator.free(self.source_context.surrounding_lines);
            allocator.free(self.source_context.line_numbers);
            allocator.free(self.source_context.highlighted_ranges);
        }
    }
};

/// Diagnostic formatter for different output formats
pub const DiagnosticFormatter = struct {
    allocator: std.mem.Allocator,
    format_style: FormatStyle,
    color_enabled: bool,
    show_source_context: bool,
    show_suggestions: bool,

    pub const FormatStyle = enum {
        Compact,
        Detailed,
        Json,
        Xml,
        Markdown,
    };

    pub fn init(allocator: std.mem.Allocator, style: FormatStyle) DiagnosticFormatter {
        return DiagnosticFormatter{
            .allocator = allocator,
            .format_style = style,
            .color_enabled = true,
            .show_source_context = true,
            .show_suggestions = true,
        };
    }

    /// Format a single diagnostic
    pub fn formatDiagnostic(self: *DiagnosticFormatter, diagnostic: *EnhancedDiagnostic) ![]const u8 {
        return switch (self.format_style) {
            .Compact => self.formatCompact(diagnostic),
            .Detailed => self.formatDetailed(diagnostic),
            .Json => self.formatJson(diagnostic),
            .Xml => self.formatXml(diagnostic),
            .Markdown => self.formatMarkdown(diagnostic),
        };
    }

    /// Format multiple diagnostics
    pub fn formatDiagnostics(self: *DiagnosticFormatter, diagnostics: []EnhancedDiagnostic) ![]const u8 {
        var output = std.ArrayList(u8).init(self.allocator);
        defer output.deinit();

        for (diagnostics, 0..) |*diagnostic, i| {
            if (i > 0) {
                try output.appendSlice("\n");
            }
            const formatted = try self.formatDiagnostic(diagnostic);
            defer self.allocator.free(formatted);
            try output.appendSlice(formatted);
        }

        return output.toOwnedSlice();
    }

    fn formatCompact(self: *DiagnosticFormatter, diagnostic: *EnhancedDiagnostic) ![]const u8 {
        const severity_str = @tagName(diagnostic.base_diagnostic.severity);
        const category_str = @tagName(diagnostic.category);

        return std.fmt.allocPrint(self.allocator, "[{s}:{s}] {s} at line {d}, column {d}: {s}", .{ severity_str, category_str, diagnostic.base_diagnostic.span.line, diagnostic.base_diagnostic.span.column, diagnostic.base_diagnostic.message });
    }

    fn formatDetailed(self: *DiagnosticFormatter, diagnostic: *EnhancedDiagnostic) ![]const u8 {
        var output = std.ArrayList(u8).init(self.allocator);
        defer output.deinit();

        // Header
        const severity_str = @tagName(diagnostic.base_diagnostic.severity);
        const category_str = @tagName(diagnostic.category);
        try output.writer().print("{s} [{s}]: {s}\n", .{ severity_str, category_str, diagnostic.base_diagnostic.message });

        // Location
        try output.writer().print("  --> line {d}, column {d}\n", .{ diagnostic.base_diagnostic.span.line, diagnostic.base_diagnostic.span.column });

        // Source context
        if (self.show_source_context and diagnostic.source_context.surrounding_lines.len > 0) {
            try output.appendSlice("\n");
            for (diagnostic.source_context.surrounding_lines, 0..) |line, i| {
                const line_num = diagnostic.source_context.line_numbers[i];
                try output.writer().print("{d:4} | {s}\n", .{ line_num, line });
            }
        }

        // Suggestions
        if (self.show_suggestions and diagnostic.suggestions.len > 0) {
            try output.appendSlice("\nSuggestions:\n");
            for (diagnostic.suggestions) |suggestion| {
                try output.writer().print("  - {s}\n", .{suggestion.message});
            }
        }

        // Fix suggestions
        if (diagnostic.fix_suggestions.len > 0) {
            try output.appendSlice("\nPossible fixes:\n");
            for (diagnostic.fix_suggestions) |fix| {
                try output.writer().print("  - {s}\n", .{fix.description});
            }
        }

        return output.toOwnedSlice();
    }

    fn formatJson(self: *DiagnosticFormatter, diagnostic: *EnhancedDiagnostic) ![]const u8 {
        _ = diagnostic;
        // TODO: Implement JSON formatting
        return try self.allocator.dupe(u8, "{}");
    }

    fn formatXml(self: *DiagnosticFormatter, diagnostic: *EnhancedDiagnostic) ![]const u8 {
        _ = diagnostic;
        // TODO: Implement XML formatting
        return try self.allocator.dupe(u8, "<diagnostic/>");
    }

    fn formatMarkdown(self: *DiagnosticFormatter, diagnostic: *EnhancedDiagnostic) ![]const u8 {
        var output = std.ArrayList(u8).init(self.allocator);
        defer output.deinit();

        const severity_str = @tagName(diagnostic.base_diagnostic.severity);
        const category_str = @tagName(diagnostic.category);

        try output.writer().print("## {s} ({s})\n\n", .{ severity_str, category_str });
        try output.writer().print("**Message:** {s}\n\n", .{diagnostic.base_diagnostic.message});
        try output.writer().print("**Location:** Line {d}, Column {d}\n\n", .{ diagnostic.base_diagnostic.span.line, diagnostic.base_diagnostic.span.column });

        if (diagnostic.suggestions.len > 0) {
            try output.appendSlice("### Suggestions\n\n");
            for (diagnostic.suggestions) |suggestion| {
                try output.writer().print("- {s}\n", .{suggestion.message});
            }
            try output.appendSlice("\n");
        }

        return output.toOwnedSlice();
    }
};

/// Diagnostic collector and manager
pub const DiagnosticManager = struct {
    allocator: std.mem.Allocator,
    diagnostics: std.ArrayList(EnhancedDiagnostic),
    next_diagnostic_id: u32,
    diagnostic_counts: DiagnosticCounts,
    filters: DiagnosticFilters,

    pub const DiagnosticCounts = struct {
        errors: u32,
        warnings: u32,
        info: u32,
        total: u32,

        pub fn init() DiagnosticCounts {
            return DiagnosticCounts{
                .errors = 0,
                .warnings = 0,
                .info = 0,
                .total = 0,
            };
        }
    };

    pub const DiagnosticFilters = struct {
        min_severity: semantics_errors.Diagnostic.Severity,
        categories: std.EnumSet(EnhancedDiagnostic.DiagnosticCategory),
        max_diagnostics: ?u32,

        pub fn init() DiagnosticFilters {
            return DiagnosticFilters{
                .min_severity = .Info,
                .categories = std.EnumSet(EnhancedDiagnostic.DiagnosticCategory).initFull(),
                .max_diagnostics = null,
            };
        }
    };

    pub fn init(allocator: std.mem.Allocator) DiagnosticManager {
        return DiagnosticManager{
            .allocator = allocator,
            .diagnostics = std.ArrayList(EnhancedDiagnostic).init(allocator),
            .next_diagnostic_id = 1,
            .diagnostic_counts = DiagnosticCounts.init(),
            .filters = DiagnosticFilters.init(),
        };
    }

    pub fn deinit(self: *DiagnosticManager) void {
        for (self.diagnostics.items) |*diagnostic| {
            diagnostic.deinit(self.allocator);
        }
        self.diagnostics.deinit();
    }

    /// Add an enhanced diagnostic
    pub fn addDiagnostic(self: *DiagnosticManager, diagnostic: EnhancedDiagnostic) !void {
        // Apply filters
        if (!self.shouldIncludeDiagnostic(&diagnostic)) {
            return;
        }

        // Check max diagnostics limit
        if (self.filters.max_diagnostics) |max| {
            if (self.diagnostic_counts.total >= max) {
                return;
            }
        }

        var enhanced_diagnostic = diagnostic;
        enhanced_diagnostic.diagnostic_id = self.next_diagnostic_id;
        self.next_diagnostic_id += 1;

        try self.diagnostics.append(enhanced_diagnostic);
        self.updateCounts(&enhanced_diagnostic);
    }

    /// Create and add a diagnostic from basic information
    pub fn createDiagnostic(self: *DiagnosticManager, severity: semantics_errors.Diagnostic.Severity, category: EnhancedDiagnostic.DiagnosticCategory, message: []const u8, span: ast.SourceSpan) !void {
        const base_diagnostic = semantics_errors.Diagnostic{
            .message = message,
            .span = span,
            .severity = severity,
            .context = null,
        };

        const enhanced_diagnostic = EnhancedDiagnostic{
            .base_diagnostic = base_diagnostic,
            .diagnostic_id = 0, // Will be set by addDiagnostic
            .category = category,
            .source_context = EnhancedDiagnostic.SourceContext{
                .file_path = null,
                .surrounding_lines = &[_][]const u8{},
                .line_numbers = &[_]u32{},
                .highlighted_ranges = &[_]EnhancedDiagnostic.SourceContext.HighlightRange{},
            },
            .suggestions = &[_]EnhancedDiagnostic.Suggestion{},
            .related_diagnostics = &[_]u32{},
            .fix_suggestions = &[_]EnhancedDiagnostic.FixSuggestion{},
            .metadata = EnhancedDiagnostic.DiagnosticMetadata{
                .timestamp = std.time.timestamp(),
                .analysis_phase = "semantic_analysis",
                .node_type = null,
                .rule_id = null,
                .help_url = null,
            },
        };

        try self.addDiagnostic(enhanced_diagnostic);
    }

    /// Get all diagnostics
    pub fn getDiagnostics(self: *DiagnosticManager) []EnhancedDiagnostic {
        return self.diagnostics.items;
    }

    /// Get diagnostics by severity
    pub fn getDiagnosticsBySeverity(self: *DiagnosticManager, severity: semantics_errors.Diagnostic.Severity) ![]EnhancedDiagnostic {
        var filtered = std.ArrayList(EnhancedDiagnostic).init(self.allocator);
        defer filtered.deinit();

        for (self.diagnostics.items) |diagnostic| {
            if (diagnostic.base_diagnostic.severity == severity) {
                try filtered.append(diagnostic);
            }
        }

        return filtered.toOwnedSlice();
    }

    /// Get diagnostics by category
    pub fn getDiagnosticsByCategory(self: *DiagnosticManager, category: EnhancedDiagnostic.DiagnosticCategory) ![]EnhancedDiagnostic {
        var filtered = std.ArrayList(EnhancedDiagnostic).init(self.allocator);
        defer filtered.deinit();

        for (self.diagnostics.items) |diagnostic| {
            if (diagnostic.category == category) {
                try filtered.append(diagnostic);
            }
        }

        return filtered.toOwnedSlice();
    }

    /// Clear all diagnostics
    pub fn clear(self: *DiagnosticManager) void {
        for (self.diagnostics.items) |*diagnostic| {
            diagnostic.deinit(self.allocator);
        }
        self.diagnostics.clearRetainingCapacity();
        self.diagnostic_counts = DiagnosticCounts.init();
        self.next_diagnostic_id = 1;
    }

    /// Get diagnostic counts
    pub fn getCounts(self: *DiagnosticManager) DiagnosticCounts {
        return self.diagnostic_counts;
    }

    /// Set diagnostic filters
    pub fn setFilters(self: *DiagnosticManager, filters: DiagnosticFilters) void {
        self.filters = filters;
    }

    /// Private helper methods
    fn shouldIncludeDiagnostic(self: *DiagnosticManager, diagnostic: *const EnhancedDiagnostic) bool {
        // Check severity filter
        const severity_order = [_]semantics_errors.Diagnostic.Severity{ .Info, .Warning, .Error };
        const diagnostic_severity_index = for (severity_order, 0..) |sev, i| {
            if (sev == diagnostic.base_diagnostic.severity) break i;
        } else 0;

        const min_severity_index = for (severity_order, 0..) |sev, i| {
            if (sev == self.filters.min_severity) break i;
        } else 0;

        if (diagnostic_severity_index < min_severity_index) {
            return false;
        }

        // Check category filter
        if (!self.filters.categories.contains(diagnostic.category)) {
            return false;
        }

        return true;
    }

    fn updateCounts(self: *DiagnosticManager, diagnostic: *const EnhancedDiagnostic) void {
        switch (diagnostic.base_diagnostic.severity) {
            .Error => self.diagnostic_counts.errors += 1,
            .Warning => self.diagnostic_counts.warnings += 1,
            .Info => self.diagnostic_counts.info += 1,
        }
        self.diagnostic_counts.total += 1;
    }
};

/// Enhanced diagnostic system integration
pub const DiagnosticSystem = struct {
    manager: DiagnosticManager,
    formatter: DiagnosticFormatter,

    pub fn init(allocator: std.mem.Allocator, format_style: DiagnosticFormatter.FormatStyle) DiagnosticSystem {
        return DiagnosticSystem{
            .manager = DiagnosticManager.init(allocator),
            .formatter = DiagnosticFormatter.init(allocator, format_style),
        };
    }

    pub fn deinit(self: *DiagnosticSystem) void {
        self.manager.deinit();
    }

    /// Add error with enhanced diagnostics
    pub fn addError(self: *DiagnosticSystem, message: []const u8, span: ast.SourceSpan, category: EnhancedDiagnostic.DiagnosticCategory) !void {
        try self.manager.createDiagnostic(.Error, category, message, span);
    }

    /// Add warning with enhanced diagnostics
    pub fn addWarning(self: *DiagnosticSystem, message: []const u8, span: ast.SourceSpan, category: EnhancedDiagnostic.DiagnosticCategory) !void {
        try self.manager.createDiagnostic(.Warning, category, message, span);
    }

    /// Add info with enhanced diagnostics
    pub fn addInfo(self: *DiagnosticSystem, message: []const u8, span: ast.SourceSpan, category: EnhancedDiagnostic.DiagnosticCategory) !void {
        try self.manager.createDiagnostic(.Info, category, message, span);
    }

    /// Generate formatted diagnostic report
    pub fn generateReport(self: *DiagnosticSystem) ![]const u8 {
        const diagnostics = self.manager.getDiagnostics();
        return self.formatter.formatDiagnostics(diagnostics);
    }

    /// Get diagnostic summary
    pub fn getSummary(self: *DiagnosticSystem) DiagnosticManager.DiagnosticCounts {
        return self.manager.getCounts();
    }
};

/// Create enhanced diagnostic system for analyzer
pub fn createDiagnosticSystem(analyzer: *SemanticAnalyzer, format_style: DiagnosticFormatter.FormatStyle) DiagnosticSystem {
    return DiagnosticSystem.init(analyzer.allocator, format_style);
}

/// Add enhanced error diagnostic
pub fn addEnhancedError(analyzer: *SemanticAnalyzer, message: []const u8, span: ast.SourceSpan, category: EnhancedDiagnostic.DiagnosticCategory) semantics_errors.SemanticError!void {
    _ = category; // TODO: Use category in enhanced implementation
    // For now, fall back to the basic error system
    // In a full implementation, this would use the enhanced diagnostic system
    try semantics_errors.addErrorStatic(analyzer, message, span);
}

/// Add enhanced warning diagnostic
pub fn addEnhancedWarning(analyzer: *SemanticAnalyzer, message: []const u8, span: ast.SourceSpan, category: EnhancedDiagnostic.DiagnosticCategory) semantics_errors.SemanticError!void {
    _ = category; // TODO: Use category in enhanced implementation
    // For now, fall back to the basic warning system
    // In a full implementation, this would use the enhanced diagnostic system
    try semantics_errors.addWarningStatic(analyzer, message, span);
}
