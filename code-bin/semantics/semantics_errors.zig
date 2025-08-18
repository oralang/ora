const std = @import("std");
pub const ast = @import("../ast.zig");

/// Semantic analysis errors
pub const SemanticError = error{
    // Contract-level errors
    MissingInitFunction,
    InvalidContractStructure,
    DuplicateInitFunction,
    InitFunctionNotPublic,

    // Memory semantics errors
    InvalidStorageAccess,
    ImmutableViolation,
    InvalidMemoryTransition,
    StorageInNonPersistentContext,

    // Function semantics errors
    MissingReturnStatement,
    UnreachableCode,
    InvalidReturnType,
    VoidReturnInNonVoidFunction,

    // Formal verification errors
    InvalidRequiresClause,
    InvalidEnsuresClause,
    OldExpressionInRequires,
    InvalidInvariant,

    // Error union semantic errors
    DuplicateErrorDeclaration,
    UndeclaredError,
    InvalidErrorType,
    InvalidErrorUnionCast,
    InvalidErrorUnionTarget,

    // General semantic errors
    UndeclaredIdentifier,
    TypeMismatch,
    InvalidOperation,
    CircularDependency,
    OutOfMemory,

    // Memory safety errors
    PointerValidationFailed,
    MessageValidationFailed,
    AnalysisStateCorrupted,
    RecoveryFailed,
};

/// Enhanced diagnostic with context for better error reporting
pub const DiagnosticContext = struct {
    node_type: std.meta.Tag(ast.AstNode),
    analysis_phase: @import("semantics_state.zig").AnalysisState.AnalysisPhase,
    recovery_attempted: bool,
    additional_info: ?[]const u8,
};

/// Semantic diagnostic with location and severity
pub const Diagnostic = struct {
    message: []const u8,
    span: ast.SourceSpan,
    severity: Severity,
    context: ?DiagnosticContext,

    pub const Severity = enum {
        Error,
        Warning,
        Info,
    };

    pub fn format(self: Diagnostic, comptime fmt: []const u8, options: std.fmt.FormatOptions, writer: anytype) !void {
        _ = fmt;
        _ = options;

        // Safe message handling with validation
        const safe_message = self.validateMessage();
        const safe_span = self.validateSpan();

        // Enhanced formatting with context
        try writer.print("{s} at line {}, column {}: {s}", .{ @tagName(self.severity), safe_span.line, safe_span.column, safe_message });

        // Add context information if available
        if (self.context) |ctx| {
            try writer.print(" [Node: {s}, Phase: {s}]", .{ @tagName(ctx.node_type), @tagName(ctx.analysis_phase) });
        }
    }

    /// Validate and sanitize diagnostic message
    fn validateMessage(self: *const Diagnostic) []const u8 {
        if (self.message.len == 0) {
            return "<empty message>";
        }

        // Check for reasonable message length
        if (self.message.len > 1024) {
            return "<message too long>";
        }

        // Basic UTF-8 validation - check if all bytes are printable ASCII or valid UTF-8 start bytes
        for (self.message) |byte| {
            if (byte < 32 or byte == 127) { // Non-printable ASCII
                if (byte != '\t' and byte != '\n' and byte != '\r') {
                    return "<corrupted message>";
                }
            }
        }

        return self.message;
    }

    /// Validate and sanitize source span
    fn validateSpan(self: *const Diagnostic) ast.SourceSpan {
        return ast.SourceSpan{
            .line = if (self.span.line > 1000000) 0 else self.span.line,
            .column = if (self.span.column > 10000) 0 else self.span.column,
            .length = if (self.span.length > 10000) 0 else self.span.length,
        };
    }
};

// Forward declaration for SemanticAnalyzer
const SemanticAnalyzer = @import("semantics_core.zig").SemanticAnalyzer;

/// Add an error diagnostic (takes ownership of message)
pub fn addError(analyzer: *SemanticAnalyzer, message: []const u8, span: ast.SourceSpan) SemanticError!void {
    const diagnostic = Diagnostic{
        .message = message,
        .span = span,
        .severity = .Error,
        .context = if (analyzer.analysis_state.current_node_type) |node_type| DiagnosticContext{
            .node_type = node_type,
            .analysis_phase = analyzer.analysis_state.phase,
            .recovery_attempted = analyzer.error_recovery_mode,
            .additional_info = null,
        } else null,
    };

    try analyzer.diagnostics.append(diagnostic);
    analyzer.analysis_state.error_count += 1;
    analyzer.validation_coverage.validation_stats.errors_found += 1;
}

/// Add an error diagnostic with static message (does not take ownership)
pub fn addErrorStatic(analyzer: *SemanticAnalyzer, message: []const u8, span: ast.SourceSpan) SemanticError!void {
    const owned_message = try analyzer.allocator.dupe(u8, message);
    try addError(analyzer, owned_message, span);
}

/// Add a warning diagnostic (takes ownership of message)
pub fn addWarning(analyzer: *SemanticAnalyzer, message: []const u8, span: ast.SourceSpan) SemanticError!void {
    const diagnostic = Diagnostic{
        .message = message,
        .span = span,
        .severity = .Warning,
        .context = if (analyzer.analysis_state.current_node_type) |node_type| DiagnosticContext{
            .node_type = node_type,
            .analysis_phase = analyzer.analysis_state.phase,
            .recovery_attempted = analyzer.error_recovery_mode,
            .additional_info = null,
        } else null,
    };

    try analyzer.diagnostics.append(diagnostic);
    analyzer.analysis_state.warning_count += 1;
    analyzer.validation_coverage.validation_stats.warnings_generated += 1;
}

/// Add a warning diagnostic with static message (does not take ownership)
pub fn addWarningStatic(analyzer: *SemanticAnalyzer, message: []const u8, span: ast.SourceSpan) SemanticError!void {
    const owned_message = try analyzer.allocator.dupe(u8, message);
    try addWarning(analyzer, owned_message, span);
}

/// Add an error diagnostic with allocated message (takes ownership)
pub fn addErrorAllocated(analyzer: *SemanticAnalyzer, message: []const u8, span: ast.SourceSpan) SemanticError!void {
    try addError(analyzer, message, span);
}

/// Add a warning diagnostic with allocated message (takes ownership)
pub fn addWarningAllocated(analyzer: *SemanticAnalyzer, message: []const u8, span: ast.SourceSpan) SemanticError!void {
    try addWarning(analyzer, message, span);
}

/// Add an info diagnostic (takes ownership of message)
pub fn addInfo(analyzer: *SemanticAnalyzer, message: []const u8, span: ast.SourceSpan) SemanticError!void {
    const diagnostic = Diagnostic{
        .message = message,
        .span = span,
        .severity = .Info,
        .context = if (analyzer.analysis_state.current_node_type) |node_type| DiagnosticContext{
            .node_type = node_type,
            .analysis_phase = analyzer.analysis_state.phase,
            .recovery_attempted = analyzer.error_recovery_mode,
            .additional_info = null,
        } else null,
    };

    try analyzer.diagnostics.append(diagnostic);
}
