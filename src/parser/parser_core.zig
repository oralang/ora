// ============================================================================
// Parser Core - Main Orchestrator
// ============================================================================
//
// Main parser entry point that coordinates all sub-parsers.
//
// ARCHITECTURE:
//   Embeds BaseParser for state management and delegates to specialized parsers
//   (ExpressionParser, StatementParser, DeclarationParser, TypeParser).
//
// ENTRY POINT: parse() - Converts tokens → AST nodes
//
// ============================================================================

const std = @import("std");
const lexer = @import("../lexer.zig");
const ast = @import("../ast.zig");
const ast_arena = @import("../ast/ast_arena.zig");

const Token = lexer.Token;
const TokenType = lexer.TokenType;
const AstNode = ast.AstNode;
const Allocator = std.mem.Allocator;

const ExpressionParser = @import("expression_parser.zig").ExpressionParser;
const StatementParser = @import("statement_parser.zig").StatementParser;
const TypeParser = @import("type_parser.zig").TypeParser;
const DeclarationParser = @import("declaration_parser.zig").DeclarationParser;
const Diag = @import("diagnostics.zig");
const common = @import("common.zig");

/// Parser errors with detailed diagnostics (alias common.ParserError for consistency)
pub const ParserError = common.ParserError;

/// Main parser for Ora language
pub const Parser = struct {
    base: common.BaseParser,

    // Sub-parsers
    expr_parser: ExpressionParser,
    stmt_parser: StatementParser,
    type_parser: TypeParser,
    decl_parser: DeclarationParser,

    pub fn init(tokens: []const Token, arena: *ast_arena.AstArena) Parser {
        return Parser{
            .base = common.BaseParser.init(tokens, arena),
            .expr_parser = ExpressionParser.init(tokens, arena),
            .stmt_parser = StatementParser.init(tokens, arena),
            .type_parser = TypeParser.init(tokens, arena),
            .decl_parser = DeclarationParser.init(tokens, arena),
        };
    }

    pub fn setFileId(self: *Parser, file_id: u32) void {
        self.base.file_id = file_id;
        self.expr_parser.base.file_id = file_id;
        self.stmt_parser.base.file_id = file_id;
        self.type_parser.base.file_id = file_id;
        self.decl_parser.base.file_id = file_id;
    }

    fn currentSpanForTopLevel(self: *Parser, node: AstNode) ast.SourceSpan {
        return switch (node) {
            .Contract => |c| c.span,
            .Function => |f| f.span,
            .VariableDecl => |v| v.span,
            .StructDecl => |s| s.span,
            .EnumDecl => |e| e.span,
            .LogDecl => |l| l.span,
            .Import => |i| i.span,
            .ErrorDecl => |e| e.span,
            else => self.makeEmptySpan(),
        };
    }

    fn makeEmptySpan(self: *Parser) ast.SourceSpan {
        _ = self;
        return .{ .file_id = 0, .line = 1, .column = 1, .length = 0, .byte_offset = 0, .lexeme = null };
    }

    /// Parse tokens into a list of top-level AST nodes
    pub fn parse(self: *Parser) ParserError![]AstNode {
        var nodes = std.ArrayList(AstNode){};
        defer nodes.deinit(self.base.arena.allocator());

        while (!self.isAtEnd()) {
            if (self.check(.Eof)) break;

            const node = try self.parseTopLevel();
            try nodes.append(self.base.arena.allocator(), node);
        }

        return nodes.toOwnedSlice(self.base.arena.allocator());
    }

    /// Parse a top-level declaration
    fn parseTopLevel(self: *Parser) ParserError!AstNode {
        // Sync parser state with sub-parsers
        self.syncSubParsers();

        // Handle imports (@import("path"))
        if (self.check(.At)) {
            _ = self.advance(); // consume '@'
            if (self.check(.Import)) {
                const result = try self.getDeclParser().parseImport();
                self.updateFromSubParser(self.decl_parser.base.current);
                return result;
            } else {
                self.errorAtCurrent("Unknown @ directive at top-level; only @import is supported") catch {};
                // Attempt to recover: skip to next semicolon or newline-like boundary
                while (!self.isAtEnd() and !self.check(.Semicolon) and !self.check(.RightBrace)) {
                    _ = self.advance();
                }
                _ = self.match(.Semicolon);
            }
        }

        // Handle const imports (const std = @import("std"))
        if (self.check(.Const)) {
            const result = try self.getDeclParser().parseConstImport();
            self.updateFromSubParser(self.decl_parser.base.current);
            return result;
        }

        // Handle contracts
        if (self.match(.Contract)) {
            const result = try self.getDeclParser().parseContract(&self.type_parser, &self.expr_parser);
            self.updateFromSubParser(self.decl_parser.base.current);
            return result;
        }

        // Handle standalone functions (for modules)
        if (self.check(.Pub) or self.check(.Fn) or self.check(.Inline)) {
            // Parse function header with declaration parser
            const hdr = try self.getDeclParser().parseFunction(&self.type_parser, &self.expr_parser);
            self.updateFromSubParser(self.decl_parser.base.current);
            // Now parse the body with StatementParser; it will consume '{' itself
            const body = try self.getStmtParser().parseBlock();
            self.updateFromSubParser(self.stmt_parser.base.current);
            var func_full = hdr;
            func_full.body = body;
            return AstNode{ .Function = func_full };
        }

        // Handle variable declarations
        if (self.isMemoryRegionKeyword() or self.check(.Let) or self.check(.Var)) {
            const var_decl = try self.getDeclParser().parseVariableDecl(&self.type_parser, &self.expr_parser);
            self.updateFromSubParser(self.decl_parser.base.current);
            return AstNode{ .VariableDecl = var_decl };
        }

        // Handle struct declarations
        if (self.match(.Struct)) {
            const result = try self.getDeclParser().parseStruct(&self.type_parser);
            self.updateFromSubParser(self.decl_parser.base.current);
            return result;
        }

        // Handle enum declarations
        if (self.match(.Enum)) {
            const result = try self.getDeclParser().parseEnum(&self.type_parser, &self.expr_parser);
            self.updateFromSubParser(self.decl_parser.base.current);
            return result;
        }

        // Handle log declarations
        if (self.match(.Log)) {
            const result = try self.getDeclParser().parseLogDecl(&self.type_parser);
            self.updateFromSubParser(self.decl_parser.base.current);
            return result;
        }

        // Handle error declarations
        if (self.match(.Error)) {
            const err_decl = try self.getDeclParser().parseErrorDeclTopLevel(&self.type_parser);
            self.updateFromSubParser(self.decl_parser.base.current);
            return AstNode{ .ErrorDecl = err_decl };
        }

        try self.errorAtCurrent("Expected top-level declaration");
        return error.UnexpectedToken;
    }

    // Token manipulation methods - delegate to base parser
    pub fn match(self: *Parser, token_type: TokenType) bool {
        return self.base.match(token_type);
    }

    pub fn check(self: *Parser, token_type: TokenType) bool {
        return self.base.check(token_type);
    }

    pub fn advance(self: *Parser) Token {
        return self.base.advance();
    }

    pub fn isAtEnd(self: *Parser) bool {
        return self.base.isAtEnd();
    }

    pub fn peek(self: *Parser) Token {
        return self.base.peek();
    }

    pub fn previous(self: *Parser) Token {
        return self.base.previous();
    }

    pub fn consume(self: *Parser, token_type: TokenType, message: []const u8) ParserError!Token {
        return self.base.consume(token_type, message);
    }

    pub fn errorAtCurrent(self: *Parser, message: []const u8) ParserError!void {
        return self.base.errorAtCurrent(message);
    }

    /// Check if current token is a memory region keyword
    fn isMemoryRegionKeyword(self: *Parser) bool {
        return common.ParserCommon.isMemoryRegionKeyword(self.peek().type);
    }

    /// Sync parser state with sub-parsers
    fn syncSubParsers(self: *Parser) void {
        self.expr_parser.base.current = self.base.current;
        self.stmt_parser.base.current = self.base.current;
        self.type_parser.base.current = self.base.current;
        self.decl_parser.base.current = self.base.current;
    }

    /// Update current position from sub-parser
    pub fn updateFromSubParser(self: *Parser, new_current: usize) void {
        self.base.current = new_current;
        self.syncSubParsers();
    }

    /// Get expression parser with synced state
    pub fn getExprParser(self: *Parser) *ExpressionParser {
        self.syncSubParsers();
        return &self.expr_parser;
    }

    /// Get statement parser with synced state
    pub fn getStmtParser(self: *Parser) *StatementParser {
        self.syncSubParsers();
        return &self.stmt_parser;
    }

    /// Get type parser with synced state
    pub fn getTypeParser(self: *Parser) *TypeParser {
        self.syncSubParsers();
        return &self.type_parser;
    }

    /// Get declaration parser with synced state
    pub fn getDeclParser(self: *Parser) *DeclarationParser {
        self.syncSubParsers();
        return &self.decl_parser;
    }
};

/// Convenience function for parsing tokens into AST with type resolution
pub fn parse(allocator: Allocator, tokens: []const Token) ParserError![]AstNode {
    var ast_arena_instance = ast_arena.AstArena.init(allocator);
    defer ast_arena_instance.deinit();

    var parser = Parser.init(tokens, &ast_arena_instance);
    const nodes = try parser.parse();

    // Perform type resolution on the parsed AST
    var type_resolver = ast.TypeResolver.init(allocator);
    type_resolver.resolveTypes(nodes) catch |err| {
        std.debug.print("Type resolution error: {s}\n", .{@errorName(err)});
        // Type resolution errors are reported but don't prevent returning the AST
        // Full type checking happens in the semantics phase
    };

    return nodes;
}
