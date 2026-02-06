//! Comptime Error Types
//!
//! Distinguishes between:
//! - Knownness failures (not_comptime): value depends on runtime inputs
//! - Stage violations (stage_violation): runtime-only op in must_eval
//! - Runtime lowering errors (comptime_only_at_runtime): comptime-only reached runtime

const std = @import("std");

/// Source span (local definition to avoid import issues)
pub const SourceSpan = struct {
    file_id: u32 = 0,
    line: u32,
    column: u32,
    length: u32,
    byte_offset: u32 = 0,
    lexeme: ?[]const u8 = null,
};

/// Kind of comptime error
pub const CtErrorKind = enum {
    // === Knownness failures ===
    /// Value depends on runtime inputs (knownness fails in must_eval)
    not_comptime,

    // === Stage violations ===
    /// Runtime-only operation attempted in must_eval mode
    stage_violation,
    /// Comptime-only operation reached runtime lowering
    comptime_only_at_runtime,

    // === Arithmetic errors ===
    overflow,
    underflow,
    division_by_zero,

    // === Limit errors ===
    recursion_limit,
    iteration_limit,
    step_limit,
    memory_limit,

    // === Type/value errors ===
    type_mismatch,
    invalid_cast,
    index_out_of_bounds,
    field_not_found,
    undefined_identifier,

    // === Internal ===
    internal_error,

    /// Get a human-readable description
    pub fn description(self: CtErrorKind) []const u8 {
        return switch (self) {
            .not_comptime => "value is not compile-time known",
            .stage_violation => "operation is runtime-only",
            .comptime_only_at_runtime => "comptime-only operation cannot be lowered to runtime",
            .overflow => "arithmetic overflow",
            .underflow => "arithmetic underflow",
            .division_by_zero => "division by zero",
            .recursion_limit => "maximum recursion depth exceeded",
            .iteration_limit => "maximum loop iterations exceeded",
            .step_limit => "maximum evaluation steps exceeded",
            .memory_limit => "maximum memory allocation exceeded",
            .type_mismatch => "type mismatch",
            .invalid_cast => "invalid type cast",
            .index_out_of_bounds => "index out of bounds",
            .field_not_found => "field not found",
            .undefined_identifier => "undefined identifier",
            .internal_error => "internal evaluator error",
        };
    }

    /// Check if this is a stage-related error
    pub fn isStageError(self: CtErrorKind) bool {
        return self == .stage_violation or self == .comptime_only_at_runtime;
    }

    /// Check if this is a limit-related error
    pub fn isLimitError(self: CtErrorKind) bool {
        return switch (self) {
            .recursion_limit, .iteration_limit, .step_limit, .memory_limit => true,
            else => false,
        };
    }

    /// Check if this is an arithmetic error
    pub fn isArithmeticError(self: CtErrorKind) bool {
        return switch (self) {
            .overflow, .underflow, .division_by_zero => true,
            else => false,
        };
    }
};

/// Comptime evaluation error
pub const CtError = struct {
    kind: CtErrorKind,
    span: SourceSpan,
    message: []const u8,
    reason: ?[]const u8 = null,

    // NOTE: trace is not stored here; diagnostics captures traces at emission time.
    // This avoids lifetime coupling between errors and the evaluation environment.

    /// Create a basic error
    pub fn init(kind: CtErrorKind, span: SourceSpan, message: []const u8) CtError {
        return .{
            .kind = kind,
            .span = span,
            .message = message,
            .reason = null,
        };
    }

    /// Create an error with reason
    pub fn withReason(kind: CtErrorKind, span: SourceSpan, message: []const u8, reason: []const u8) CtError {
        return .{
            .kind = kind,
            .span = span,
            .message = message,
            .reason = reason,
        };
    }

    /// Create a "not comptime" error
    pub fn notComptime(span: SourceSpan, reason: []const u8) CtError {
        return withReason(.not_comptime, span, "expression is not compile-time known", reason);
    }

    /// Create a stage violation error
    pub fn stageViolation(span: SourceSpan, op_name: []const u8) CtError {
        return withReason(.stage_violation, span, "runtime-only operation in comptime context", op_name);
    }

    /// Create a comptime-only at runtime error
    pub fn comptimeOnlyAtRuntime(span: SourceSpan, op_name: []const u8) CtError {
        return withReason(.comptime_only_at_runtime, span, "comptime-only operation cannot reach runtime", op_name);
    }

    /// Format error for display
    pub fn format(
        self: CtError,
        comptime fmt: []const u8,
        options: std.fmt.FormatOptions,
        writer: anytype,
    ) !void {
        _ = fmt;
        _ = options;

        try writer.print("{s} at {d}:{d}", .{
            self.message,
            self.span.line,
            self.span.column,
        });

        if (self.reason) |r| {
            try writer.print(" ({s})", .{r});
        }
    }
};

/// Try-eval error handling policy
pub const TryEvalPolicy = enum {
    /// Errors in try_eval become .runtime (default, avoids surprising failures)
    forgiving,
    /// Errors in try_eval are always reported
    strict,
};

// ============================================================================
// Tests
// ============================================================================

test "CtErrorKind classification" {
    try std.testing.expect(CtErrorKind.stage_violation.isStageError());
    try std.testing.expect(CtErrorKind.comptime_only_at_runtime.isStageError());
    try std.testing.expect(!CtErrorKind.overflow.isStageError());

    try std.testing.expect(CtErrorKind.step_limit.isLimitError());
    try std.testing.expect(!CtErrorKind.overflow.isLimitError());

    try std.testing.expect(CtErrorKind.overflow.isArithmeticError());
    try std.testing.expect(!CtErrorKind.step_limit.isArithmeticError());
}

test "CtError creation" {
    const span = SourceSpan{ .line = 10, .column = 5, .length = 3 };

    const err1 = CtError.notComptime(span, "depends on runtime parameter");
    try std.testing.expectEqual(CtErrorKind.not_comptime, err1.kind);
    try std.testing.expect(err1.reason != null);

    const err2 = CtError.stageViolation(span, "sload");
    try std.testing.expectEqual(CtErrorKind.stage_violation, err2.kind);
}
