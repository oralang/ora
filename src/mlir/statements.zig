// ============================================================================
// Statement Lowering
// ============================================================================
//
// Re-exports from modular statement lowering implementation.
// Implementation is split across multiple files in statements/ directory.
//
// ============================================================================

// Re-export core StatementLowerer and types
pub const StatementLowerer = @import("statements/statement_lowerer.zig").StatementLowerer;
pub const LabelContext = @import("statements/statement_lowerer.zig").LabelContext;
