const std = @import("std");
const lexer = @import("../lexer.zig");
const ast = @import("../ast.zig");
const ast_arena = @import("../ast/ast_arena.zig");

const Token = lexer.Token;
const TokenType = lexer.TokenType;
const cst = @import("../cst.zig");
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
    tokens: []const Token,
    current: usize,
    arena: *ast_arena.AstArena,
    // Optional CST builder to record a thin CST alongside AST
    cst_builder: ?*cst.CstBuilder = null,

    // Sub-parsers
    expr_parser: ExpressionParser,
    stmt_parser: StatementParser,
    type_parser: TypeParser,
    decl_parser: DeclarationParser,

    pub fn init(tokens: []const Token, arena: *ast_arena.AstArena) Parser {
        return Parser{
            .tokens = tokens,
            .current = 0,
            .arena = arena,
            .expr_parser = ExpressionParser.init(tokens, arena),
            .stmt_parser = StatementParser.init(tokens, arena),
            .type_parser = TypeParser.init(tokens, arena),
            .decl_parser = DeclarationParser.init(tokens, arena),
        };
    }

    pub fn setFileId(self: *Parser, file_id: u32) void {
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
    pub fn withCst(self: *Parser, builder: *cst.CstBuilder) void {
        self.cst_builder = builder;
    }

    /// Parse tokens into a list of top-level AST nodes
    pub fn parse(self: *Parser) ParserError![]AstNode {
        var nodes = std.ArrayList(AstNode){};
        defer nodes.deinit(self.arena.allocator());

        while (!self.isAtEnd()) {
            if (self.check(.Eof)) break;

            const node = try self.parseTopLevel();
            try nodes.append(self.arena.allocator(), node);
        }

        return nodes.toOwnedSlice(self.arena.allocator());
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
                if (self.cst_builder) |builder| {
                    const span = self.currentSpanForTopLevel(result);
                    _ = try builder.createTopLevel(cst.CstKind.ImportDecl, span, @as(u32, @intCast(self.current)), @as(u32, @intCast(self.current)));
                }
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
            if (self.cst_builder) |builder| {
                const span = self.currentSpanForTopLevel(result);
                _ = try builder.createTopLevel(cst.CstKind.ImportDecl, span, @as(u32, @intCast(self.current)), @as(u32, @intCast(self.current)));
            }
            return result;
        }

        // Handle contracts
        if (self.match(.Contract)) {
            const result = try self.getDeclParser().parseContract();
            self.updateFromSubParser(self.decl_parser.base.current);
            if (self.cst_builder) |builder| {
                const span = self.currentSpanForTopLevel(result);
                _ = try builder.createTopLevel(cst.CstKind.ContractDecl, span, @as(u32, @intCast(self.current)), @as(u32, @intCast(self.current)));
            }
            return result;
        }

        // Handle standalone functions (for modules)
        if (self.check(.Pub) or self.check(.Fn) or self.check(.Inline)) {
            // Parse function header with declaration parser
            const hdr = try self.getDeclParser().parseFunction();
            self.updateFromSubParser(self.decl_parser.base.current);
            // Now parse the body with StatementParser; it will consume '{' itself
            const body = try self.getStmtParser().parseBlock();
            self.updateFromSubParser(self.stmt_parser.base.current);
            var func_full = hdr;
            func_full.body = body;
            if (self.cst_builder) |builder| {
                const span = func_full.span;
                _ = try builder.createTopLevel(cst.CstKind.FunctionDecl, span, @as(u32, @intCast(self.current)), @as(u32, @intCast(self.current)));
            }
            return AstNode{ .Function = func_full };
        }

        // Handle variable declarations
        if (self.isMemoryRegionKeyword() or self.check(.Let) or self.check(.Var)) {
            const var_decl = try self.getDeclParser().parseVariableDecl();
            self.updateFromSubParser(self.decl_parser.base.current);
            if (self.cst_builder) |builder| {
                const span = var_decl.span;
                _ = try builder.createTopLevel(cst.CstKind.VarDecl, span, @as(u32, @intCast(self.current)), @as(u32, @intCast(self.current)));
            }
            return AstNode{ .VariableDecl = var_decl };
        }

        // Handle struct declarations
        if (self.match(.Struct)) {
            const result = try self.getDeclParser().parseStruct();
            self.updateFromSubParser(self.decl_parser.base.current);
            if (self.cst_builder) |builder| {
                const span = switch (result) {
                    .StructDecl => |s| s.span,
                    else => self.makeEmptySpan(),
                };
                _ = try builder.createTopLevel(cst.CstKind.StructDecl, span, @as(u32, @intCast(self.current)), @as(u32, @intCast(self.current)));
            }
            return result;
        }

        // Handle enum declarations
        if (self.match(.Enum)) {
            const result = try self.getDeclParser().parseEnum();
            self.updateFromSubParser(self.decl_parser.base.current);
            if (self.cst_builder) |builder| {
                const span = switch (result) {
                    .EnumDecl => |e| e.span,
                    else => self.makeEmptySpan(),
                };
                _ = try builder.createTopLevel(cst.CstKind.EnumDecl, span, @as(u32, @intCast(self.current)), @as(u32, @intCast(self.current)));
            }
            return result;
        }

        // Handle log declarations
        if (self.match(.Log)) {
            const result = try self.getDeclParser().parseLogDecl();
            self.updateFromSubParser(self.decl_parser.base.current);
            if (self.cst_builder) |builder| {
                const span = switch (result) {
                    .LogDecl => |l| l.span,
                    else => self.makeEmptySpan(),
                };
                _ = try builder.createTopLevel(cst.CstKind.LogDecl, span, @as(u32, @intCast(self.current)), @as(u32, @intCast(self.current)));
            }
            return result;
        }

        // Handle error declarations
        if (self.match(.Error)) {
            const err_decl = try self.getDeclParser().parseErrorDeclTopLevel();
            self.updateFromSubParser(self.decl_parser.base.current);
            if (self.cst_builder) |builder| {
                const span = err_decl.span;
                _ = try builder.createTopLevel(cst.CstKind.ErrorDecl, span, @as(u32, @intCast(self.current)), @as(u32, @intCast(self.current)));
            }
            return AstNode{ .ErrorDecl = err_decl };
        }

        return self.errorAtCurrent("Expected top-level declaration");
    }

    // Token manipulation methods
    pub fn match(self: *Parser, token_type: TokenType) bool {
        if (self.check(token_type)) {
            _ = self.advance();
            return true;
        }
        return false;
    }

    pub fn check(self: *Parser, token_type: TokenType) bool {
        if (self.isAtEnd()) return false;
        return self.peek().type == token_type;
    }

    pub fn advance(self: *Parser) Token {
        if (!self.isAtEnd()) self.current += 1;
        return self.previous();
    }

    pub fn isAtEnd(self: *Parser) bool {
        return self.peek().type == .Eof;
    }

    pub fn peek(self: *Parser) Token {
        return self.tokens[self.current];
    }

    pub fn previous(self: *Parser) Token {
        return self.tokens[self.current - 1];
    }

    pub fn consume(self: *Parser, token_type: TokenType, message: []const u8) ParserError!Token {
        if (self.check(token_type)) {
            return self.advance();
        }

        Diag.print("Parser error: {s} (got {s})\n", .{ message, @tagName(self.peek().type) });
        return ParserError.ExpectedToken;
    }

    pub fn errorAtCurrent(self: *Parser, message: []const u8) ParserError {
        const current_token = self.peek();
        Diag.print("Parser error at line {}, column {}: {s}\n", .{ current_token.line, current_token.column, message });
        // Add more context with the lexeme if available
        if (current_token.lexeme.len > 0) {
            Diag.print("   Found: '{s}'\n", .{current_token.lexeme});
        }
        return ParserError.UnexpectedToken;
    }

    /// Check if current token is a memory region keyword
    fn isMemoryRegionKeyword(self: *Parser) bool {
        return self.check(.Const) or self.check(.Immutable) or
            self.check(.Storage) or self.check(.Memory) or self.check(.Tstore);
    }

    /// Sync parser state with sub-parsers
    fn syncSubParsers(self: *Parser) void {
        self.expr_parser.base.current = self.current;
        self.stmt_parser.base.current = self.current;
        self.type_parser.base.current = self.current;
        self.decl_parser.base.current = self.current;
    }

    /// Update current position from sub-parser
    pub fn updateFromSubParser(self: *Parser, new_current: usize) void {
        self.current = new_current;
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
