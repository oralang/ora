// Parser module exports
pub const Parser = @import("parser_core.zig").Parser;
pub const ExpressionParser = @import("expression_parser.zig").ExpressionParser;
pub const StatementParser = @import("statement_parser.zig").StatementParser;
pub const TypeParser = @import("type_parser.zig").TypeParser;
pub const DeclarationParser = @import("declaration_parser.zig").DeclarationParser;

pub const ParserError = @import("parser_core.zig").ParserError;
pub const parse = @import("parser_core.zig").parse;
