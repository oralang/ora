// Re-export the modular parser
pub const parser_mod = @import("parser/mod.zig");
pub const Parser = parser_mod.Parser;
pub const ParserError = parser_mod.ParserError;
pub const parse = parser_mod.parse;
pub const parseWithArena = parser_mod.parseWithArena;
