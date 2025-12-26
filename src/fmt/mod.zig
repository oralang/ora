// ============================================================================
// Ora Formatter Module
// ============================================================================
//
// Canonical formatter for Ora source code (ora fmt)
//
// DESIGN:
//   • AST-driven formatting (not regex-based)
//   • Comment preservation via lexer trivia
//   • Deterministic and idempotent
//   • Semantics-preserving
//
// ============================================================================

pub const Formatter = @import("formatter.zig").Formatter;
pub const FormatOptions = @import("formatter.zig").FormatOptions;
pub const FormatError = @import("formatter.zig").FormatError;

pub const Writer = @import("writer.zig").Writer;
