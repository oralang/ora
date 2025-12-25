// ============================================================================
// Parser Common Utilities
// ============================================================================
//
// Shared utilities and base infrastructure for all parser modules.
//
// KEY COMPONENTS:
//   • ParserCommon: Helper functions (isMemoryRegionKeyword, etc.)
//   • ParserError: Error types for parsing
//   • BaseParser: Base state and token manipulation
//
// ============================================================================

const std = @import("std");
const lexer = @import("../lexer.zig");
const ast = @import("../ast.zig");
const Diag = @import("diagnostics.zig");
const AstArena = @import("../ast/ast_arena.zig").AstArena;

const Token = lexer.Token;

/// Common parser utilities and helper functions
pub const ParserCommon = struct {
    /// Helper function to create SourceSpan from Token
    pub fn makeSpan(token: Token) ast.SourceSpan {
        return ast.SourceSpan{
            .file_id = 0,
            .line = token.line,
            .column = token.column,
            .length = @intCast(token.lexeme.len),
            .byte_offset = token.range.start_offset,
            .lexeme = token.lexeme,
        };
    }

    /// Check if a keyword can be used as an identifier in certain contexts
    pub fn isKeywordThatCanBeIdentifier(token_type: lexer.TokenType) bool {
        return switch (token_type) {
            .From => true, // 'from' can be used as a parameter name in log declarations
            .To => true, // 'to' can be used as a parameter name
            .Error => true, // 'error' can be used as a variable name in catch blocks
            else => false,
        };
    }

    /// Check if token type is a memory region keyword
    /// Includes: storage, memory, tstore, const, immutable
    pub fn isMemoryRegionKeyword(token_type: lexer.TokenType) bool {
        return token_type == .Storage or
            token_type == .Memory or
            token_type == .Tstore or
            token_type == .Const or
            token_type == .Immutable;
    }
};

/// Unified parser error set shared by all parser components
pub const ParserError = error{
    UnexpectedToken,
    ExpectedToken,
    ExpectedIdentifier,
    ExpectedType,
    ExpectedExpression,
    ExpectedRangeExpression,
    UnexpectedEof,
    OutOfMemory,
    InvalidMemoryRegion,
    InvalidReturnType,
    UnresolvedType,
    TypeResolutionFailed,
};

/// Base parser trait that all sub-parsers implement
pub const BaseParser = struct {
    tokens: []const Token,
    current: usize,
    arena: *AstArena,
    file_id: u32 = 0,

    pub fn init(tokens: []const Token, arena: *AstArena) BaseParser {
        return BaseParser{
            .tokens = tokens,
            .current = 0,
            .arena = arena,
            .file_id = 0,
        };
    }

    // common token manipulation methods
    pub fn match(self: *BaseParser, token_type: lexer.TokenType) bool {
        if (self.check(token_type)) {
            _ = self.advance();
            return true;
        }
        return false;
    }

    pub fn check(self: *BaseParser, token_type: lexer.TokenType) bool {
        if (self.isAtEnd()) return false;
        return self.peek().type == token_type;
    }

    pub fn advance(self: *BaseParser) Token {
        if (!self.isAtEnd()) self.current += 1;
        return self.previous();
    }

    pub fn isAtEnd(self: *BaseParser) bool {
        return self.peek().type == .Eof;
    }

    pub fn peek(self: *BaseParser) Token {
        return self.tokens[self.current];
    }

    pub fn previous(self: *BaseParser) Token {
        return self.tokens[self.current - 1];
    }

    pub fn consume(self: *BaseParser, token_type: lexer.TokenType, message: []const u8) !Token {
        if (self.check(token_type)) {
            return self.advance();
        }

        const current_token = self.peek();
        Diag.print("Parser error at line {}, column {}: {s} (expected {s}, got {s})\n", .{ current_token.line, current_token.column, message, @tagName(token_type), @tagName(current_token.type) });

        // add more context with the lexeme if available
        if (current_token.lexeme.len > 0) {
            Diag.print("   Found: '{s}'\n", .{current_token.lexeme});
        }

        return error.ExpectedToken;
    }

    pub fn errorAtCurrent(self: *BaseParser, message: []const u8) !void {
        const current_token = self.peek();
        Diag.print("Parser error at line {}, column {}: {s}\n", .{ current_token.line, current_token.column, message });
        // add more context with the lexeme if available
        if (current_token.lexeme.len > 0) {
            Diag.print("   Found: '{s}'\n", .{current_token.lexeme});
        }
        return error.UnexpectedToken;
    }

    /// Consume an identifier or keyword that can be used as an identifier in certain contexts
    pub fn consumeIdentifierOrKeyword(self: *BaseParser, message: []const u8) !Token {
        const current_token = self.peek();
        if (current_token.type == .Identifier or ParserCommon.isKeywordThatCanBeIdentifier(current_token.type)) {
            return self.advance();
        }

        std.debug.print("Parser error at line {}, column {}: {s} (got {s})\n", .{ current_token.line, current_token.column, message, @tagName(current_token.type) });

        // add more context with the lexeme if available
        if (current_token.lexeme.len > 0) {
            std.debug.print("   Found: '{s}'\n", .{current_token.lexeme});
        }

        return error.ExpectedToken;
    }

    /// Match a keyword that can be used as an identifier
    pub fn matchKeywordAsIdentifier(self: *BaseParser) bool {
        if (ParserCommon.isKeywordThatCanBeIdentifier(self.peek().type)) {
            _ = self.advance();
            return true;
        }
        return false;
    }

    /// Get the current token's source span
    pub fn currentSpan(self: *BaseParser) ast.SourceSpan {
        const t = self.peek();
        return ast.SourceSpan{
            .file_id = self.file_id,
            .line = t.line,
            .column = t.column,
            .length = @intCast(t.lexeme.len),
            .byte_offset = t.range.start_offset,
            .lexeme = t.lexeme,
        };
    }

    /// Create a SourceSpan for an arbitrary token using current parser file_id context
    pub fn spanFromToken(self: *BaseParser, token: Token) ast.SourceSpan {
        return ast.SourceSpan{
            .file_id = self.file_id,
            .line = token.line,
            .column = token.column,
            .length = @intCast(token.lexeme.len),
            .byte_offset = token.range.start_offset,
            .lexeme = token.lexeme,
        };
    }
};
