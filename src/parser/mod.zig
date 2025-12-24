// ============================================================================
// Parser Module - Public API
// ============================================================================
//
// Re-exports all parser components for external use.
//
// ============================================================================

// Parser module exports
pub const Parser = @import("parser_core.zig").Parser;
pub const ExpressionParser = @import("expression_parser.zig").ExpressionParser;
pub const StatementParser = @import("statement_parser.zig").StatementParser;
pub const TypeParser = @import("type_parser.zig").TypeParser;
pub const DeclarationParser = @import("declaration_parser.zig").DeclarationParser;
pub const diagnostics = @import("diagnostics.zig");

pub const ParserError = @import("parser_core.zig").ParserError;
pub const parse = @import("parser_core.zig").parse;
pub const parseWithArena = @import("parser_core.zig").parseWithArena;
