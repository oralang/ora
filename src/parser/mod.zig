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
pub const ParseResult = @import("parser_core.zig").ParseResult;
pub const parseRaw = @import("parser_core.zig").parseRaw;

const pipeline = @import("pipeline.zig");
pub const parse = pipeline.parse;
pub const parseWithArena = pipeline.parseWithArena;
