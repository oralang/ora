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
const builtin = @import("builtin");
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
const common = @import("common.zig");
const log = @import("log");

/// Parser errors with detailed diagnostics (alias common.ParserError for consistency)
pub const ParserError = common.ParserError;

/// Main parser for Ora language
pub const Parser = struct {
    base: common.BaseParser,

    // sub-parsers
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
        // sync parser state with sub-parsers
        self.syncSubParsers();

        // top-level bare @import("...") is intentionally disallowed in v1.
        // Require: const alias = @import("...");
        if (self.check(.At)) {
            _ = self.advance(); // consume '@'
            if (self.check(.Import)) {
                try self.errorAtCurrent("Bare @import is not allowed; use 'const name = @import(\"...\");'");
                return error.UnexpectedToken;
            } else {
                self.errorAtCurrent("Unknown @ directive at top-level") catch {};
                return error.UnexpectedToken;
            }
        }

        // handle comptime const imports (comptime const math = @import("comptime/math"))
        if (self.check(.Comptime)) {
            const comptime_token = self.advance(); // consume 'comptime'
            if (self.check(.Const)) {
                _ = self.advance(); // consume 'const'
                const result = try self.getDeclParser().parseConstImport(true, comptime_token);
                self.updateFromSubParser(self.decl_parser.base.current);
                return result;
            } else {
                try self.errorAtCurrent("Expected 'const' after 'comptime' for import declaration");
                return error.UnexpectedToken;
            }
        }

        // handle const imports (const std = @import("std"))
        if (self.check(.Const)) {
            const const_token = self.advance(); // consume 'const'
            const result = try self.getDeclParser().parseConstImport(false, const_token);
            self.updateFromSubParser(self.decl_parser.base.current);
            return result;
        }

        // handle contracts
        if (self.match(.Contract)) {
            const result = try self.getDeclParser().parseContract(&self.type_parser, &self.expr_parser);
            self.updateFromSubParser(self.decl_parser.base.current);
            return result;
        }

        // handle standalone functions (for modules)
        if (self.check(.Pub) or self.check(.Fn)) {
            // parse function header with declaration parser
            const hdr = try self.getDeclParser().parseFunction(&self.type_parser, &self.expr_parser);
            self.updateFromSubParser(self.decl_parser.base.current);
            // now parse the body with StatementParser; it will consume '{' itself
            const body = try self.getStmtParser().parseBlock();
            self.updateFromSubParser(self.stmt_parser.base.current);
            var func_full = hdr;
            func_full.body = body;
            return AstNode{ .Function = func_full };
        }

        // handle variable declarations
        if (self.isMemoryRegionKeyword() or self.check(.Let) or self.check(.Var)) {
            const var_decl = try self.getDeclParser().parseVariableDecl(&self.type_parser, &self.expr_parser);
            self.updateFromSubParser(self.decl_parser.base.current);
            return AstNode{ .VariableDecl = var_decl };
        }

        // handle struct declarations
        if (self.match(.Struct)) {
            const result = try self.getDeclParser().parseStruct(&self.type_parser);
            self.updateFromSubParser(self.decl_parser.base.current);
            return result;
        }

        // handle bitfield declarations
        if (self.match(.Bitfield)) {
            const result = try self.getDeclParser().parseBitfield(&self.type_parser);
            self.updateFromSubParser(self.decl_parser.base.current);
            return result;
        }

        // handle enum declarations
        if (self.match(.Enum)) {
            const result = try self.getDeclParser().parseEnum(&self.type_parser, &self.expr_parser);
            self.updateFromSubParser(self.decl_parser.base.current);
            return result;
        }

        // handle log declarations
        if (self.match(.Log)) {
            const result = try self.getDeclParser().parseLogDecl(&self.type_parser);
            self.updateFromSubParser(self.decl_parser.base.current);
            return result;
        }

        // handle error declarations
        if (self.match(.Error)) {
            const err_decl = try self.getDeclParser().parseErrorDeclTopLevel(&self.type_parser);
            self.updateFromSubParser(self.decl_parser.base.current);
            return AstNode{ .ErrorDecl = err_decl };
        }

        try self.errorAtCurrent("Expected top-level declaration");
        return error.UnexpectedToken;
    }

    // token manipulation methods - delegate to base parser
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
    pub fn syncSubParsers(self: *Parser) void {
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
/// Returns both the AST nodes and the arena (caller must keep arena alive while using nodes)
pub fn parseWithArena(allocator: Allocator, tokens: []const Token) ParserError!struct { nodes: []AstNode, arena: ast_arena.AstArena } {
    var ast_arena_instance = ast_arena.AstArena.init(allocator);
    // note: arena is NOT deinitialized here - caller must deinit it

    var parser = Parser.init(tokens, &ast_arena_instance);
    const nodes = try parser.parse();

    // collect symbols (errors, functions, structs, etc.) into symbol table before type resolution
    // use analyzePhase1 which creates contract scopes and function scopes properly
    const semantics_core = @import("../semantics/core.zig");
    var semantics_result = try semantics_core.analyzePhase1(allocator, nodes);
    defer allocator.free(semantics_result.diagnostics);
    defer semantics_result.symbols.deinit();
    ensureLogSignatures(&semantics_result.symbols, nodes) catch |err| {
        if (!builtin.is_test) {
            log.err("Failed to collect log signatures: {s}\n", .{@errorName(err)});
        }
        return ParserError.TypeResolutionFailed;
    };

    // perform type resolution on the parsed AST
    const TypeResolver = @import("../ast/type_resolver/mod.zig").TypeResolver;
    var type_resolver = TypeResolver.init(allocator, ast_arena_instance.allocator(), &semantics_result.symbols);
    errdefer type_resolver.deinit();
    type_resolver.resolveTypes(nodes) catch |err| {
        // type resolution errors (especially TypeMismatch) should stop compilation
        // these indicate invalid type assignments that cannot be safely compiled
        if (!builtin.is_test) {
            const is_user_facing_type_error = err == @import("../ast/type_resolver/mod.zig").TypeResolutionError.ErrorUnionOutsideTry or
                err == @import("../ast/type_resolver/mod.zig").TypeResolutionError.GenericContractNotSupported or
                err == @import("../ast/type_resolver/mod.zig").TypeResolutionError.TopLevelGenericInstantiationNotSupported or
                err == @import("../ast/type_resolver/mod.zig").TypeResolutionError.ComptimeArithmeticError or
                err == @import("../ast/type_resolver/mod.zig").TypeResolutionError.ComptimeEvaluationError;
            log.err("Type resolution failed: {s}\n", .{@errorName(err)});
            if (err == @import("../ast/type_resolver/mod.zig").TypeResolutionError.ErrorUnionOutsideTry) {
                log.help("use `try` to unwrap error unions or wrap the code in a try/catch block\n", .{});
            } else if (err == @import("../ast/type_resolver/mod.zig").TypeResolutionError.GenericContractNotSupported) {
                log.help("generic contracts are parsed but not implemented yet; remove type parameters for now\n", .{});
            } else if (err == @import("../ast/type_resolver/mod.zig").TypeResolutionError.TopLevelGenericInstantiationNotSupported) {
                log.help("generic functions/structs currently require a contract scope; move the generic usage inside a contract\n", .{});
            } else if (err == @import("../ast/type_resolver/mod.zig").TypeResolutionError.ComptimeArithmeticError) {
                log.help("checked arithmetic in a compile-time-known expression failed; use wrapping operators (e.g. **%) or smaller values\n", .{});
            } else if (err == @import("../ast/type_resolver/mod.zig").TypeResolutionError.ComptimeEvaluationError) {
                log.help("explicit/known comptime evaluation failed and cannot fall back to runtime\n", .{});
            }
            // best-effort stack trace in debug builds
            if (!is_user_facing_type_error) {
                const trace = @errorReturnTrace();
                if (trace) |t| std.debug.dumpStackTrace(t.*);
            }
        }
        return ParserError.TypeResolutionFailed;
    };
    type_resolver.deinit();

    return .{ .nodes = nodes, .arena = ast_arena_instance };
}

/// Convenience function for parsing tokens into AST with type resolution
/// WARNING: For --emit-ast, use parseWithArena instead to keep arena alive
pub fn parse(allocator: Allocator, tokens: []const Token) ParserError![]AstNode {
    var ast_arena_instance = ast_arena.AstArena.init(allocator);
    defer ast_arena_instance.deinit();

    var parser = Parser.init(tokens, &ast_arena_instance);
    const nodes = try parser.parse();

    // collect symbols (errors, functions, structs, etc.) into symbol table before type resolution
    // use analyzePhase1 which creates contract scopes and function scopes properly
    const semantics_core = @import("../semantics/core.zig");
    var semantics_result = try semantics_core.analyzePhase1(allocator, nodes);
    defer allocator.free(semantics_result.diagnostics);
    defer semantics_result.symbols.deinit();
    ensureLogSignatures(&semantics_result.symbols, nodes) catch |err| {
        if (!builtin.is_test) {
            log.err("Failed to collect log signatures: {s}\n", .{@errorName(err)});
        }
        return ParserError.TypeResolutionFailed;
    };

    // perform type resolution on the parsed AST
    const TypeResolver = @import("../ast/type_resolver/mod.zig").TypeResolver;
    var type_resolver = TypeResolver.init(allocator, ast_arena_instance.allocator(), &semantics_result.symbols);
    errdefer type_resolver.deinit();
    type_resolver.resolveTypes(nodes) catch |err| {
        // type resolution errors (especially TypeMismatch) should stop compilation
        // these indicate invalid type assignments that cannot be safely compiled
        if (!builtin.is_test) {
            const is_user_facing_type_error = err == @import("../ast/type_resolver/mod.zig").TypeResolutionError.ErrorUnionOutsideTry or
                err == @import("../ast/type_resolver/mod.zig").TypeResolutionError.GenericContractNotSupported or
                err == @import("../ast/type_resolver/mod.zig").TypeResolutionError.TopLevelGenericInstantiationNotSupported or
                err == @import("../ast/type_resolver/mod.zig").TypeResolutionError.ComptimeArithmeticError or
                err == @import("../ast/type_resolver/mod.zig").TypeResolutionError.ComptimeEvaluationError;
            log.err("Type resolution failed: {s}\n", .{@errorName(err)});
            if (err == @import("../ast/type_resolver/mod.zig").TypeResolutionError.ErrorUnionOutsideTry) {
                log.help("use `try` to unwrap error unions or wrap the code in a try/catch block\n", .{});
            } else if (err == @import("../ast/type_resolver/mod.zig").TypeResolutionError.GenericContractNotSupported) {
                log.help("generic contracts are parsed but not implemented yet; remove type parameters for now\n", .{});
            } else if (err == @import("../ast/type_resolver/mod.zig").TypeResolutionError.TopLevelGenericInstantiationNotSupported) {
                log.help("generic functions/structs currently require a contract scope; move the generic usage inside a contract\n", .{});
            } else if (err == @import("../ast/type_resolver/mod.zig").TypeResolutionError.ComptimeArithmeticError) {
                log.help("checked arithmetic in a compile-time-known expression failed; use wrapping operators (e.g. **%) or smaller values\n", .{});
            } else if (err == @import("../ast/type_resolver/mod.zig").TypeResolutionError.ComptimeEvaluationError) {
                log.help("explicit/known comptime evaluation failed and cannot fall back to runtime\n", .{});
            }
            // best-effort stack trace in debug builds
            if (!is_user_facing_type_error) {
                const trace = @errorReturnTrace();
                if (trace) |t| std.debug.dumpStackTrace(t.*);
            }
        }
        return ParserError.TypeResolutionFailed;
    };
    type_resolver.deinit();

    return nodes;
}

fn ensureLogSignatures(symbols: *@import("../semantics/state.zig").SymbolTable, nodes: []const AstNode) !void {
    for (nodes) |node| switch (node) {
        .LogDecl => |l| {
            if (symbols.log_signatures.get(l.name) == null) {
                try symbols.log_signatures.put(l.name, l.fields);
            }
        },
        .Contract => |c| {
            if (symbols.contract_log_signatures.getPtr(c.name) == null) {
                const log_map = std.StringHashMap([]const ast.LogField).init(symbols.allocator);
                try symbols.contract_log_signatures.put(c.name, log_map);
            }
            for (c.body) |member| switch (member) {
                .LogDecl => |l| {
                    if (symbols.contract_log_signatures.getPtr(c.name)) |log_map| {
                        if (log_map.get(l.name) == null) {
                            try log_map.put(l.name, l.fields);
                        }
                    }
                },
                else => {},
            };
        },
        else => {},
    };
}
