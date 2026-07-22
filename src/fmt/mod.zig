// ============================================================================
// Ora Formatter Module
// ============================================================================
//
// Canonical formatter for Ora source code (ora fmt)
//
// DESIGN:
//   • Token-stream emitter; parse + AST lowering gate formatting (files with
//     diagnostics are refused) and classify ambiguous angle tokens
//     (`<` `>` `>>` as binary operators vs generic brackets)
//   • Comment preservation via lexer trivia
//   • Deterministic and idempotent
//   • Semantics-preserving
//
// ============================================================================

pub const Formatter = @import("formatter.zig").Formatter;
pub const FormatOptions = @import("formatter.zig").FormatOptions;
pub const FormatError = @import("formatter.zig").FormatError;

pub const Writer = @import("writer.zig").Writer;
