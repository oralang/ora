// ============================================================================
// Declaration Parser
// ============================================================================
//
// Parses top-level and contract-level declarations.
//
// SUPPORTED DECLARATIONS:
//   • Contracts, functions, structs, enums, logs, errors
//   • Constants, variables, imports
//   • Storage variables with memory regions
//
// ============================================================================

const std = @import("std");
const lexer = @import("../lexer.zig");
const ast = @import("../ast.zig");
const common = @import("common.zig");
const AstArena = @import("../ast/ast_arena.zig").AstArena;

const Token = lexer.Token;
const TokenType = lexer.TokenType;
const BaseParser = common.BaseParser;
const ParserCommon = common.ParserCommon;
const ParserError = @import("parser_core.zig").ParserError;

// Import other parsers for cross-parser communication
const TypeParser = @import("type_parser.zig").TypeParser;
const ExpressionParser = @import("expression_parser.zig").ExpressionParser;
const StatementParser = @import("statement_parser.zig").StatementParser;

// Import declaration parsers
const contract = @import("declarations/contract.zig");
const function_parser = @import("declarations/function.zig");
const types = @import("declarations/types.zig");

/// Specialized parser for declarations (functions, structs, enums, etc.)
/// Uses dependency injection: methods that need other parsers receive them as parameters
pub const DeclarationParser = struct {
    base: BaseParser,

    pub fn init(tokens: []const Token, arena: *AstArena) DeclarationParser {
        return DeclarationParser{
            .base = BaseParser.init(tokens, arena),
        };
    }

    /// Parse import statement (@import("path"))
    pub fn parseImport(self: *DeclarationParser) !ast.AstNode {
        const import_token = self.base.previous(); // The '@' token
        _ = try self.base.consume(.Import, "Expected 'import' after '@'");
        _ = try self.base.consume(.LeftParen, "Expected '(' after 'import'");

        const path_token = try self.base.consume(.StringLiteral, "Expected string path");

        _ = try self.base.consume(.RightParen, "Expected ')' after import path");
        _ = try self.base.consume(.Semicolon, "Expected ';' after @import(...) statement");

        return ast.AstNode{
            .Import = ast.ImportNode{
                .alias = null, // No alias in @import syntax
                .path = path_token.lexeme,
                .span = self.base.spanFromToken(import_token),
            },
        };
    }

    /// Parse const import statement (const std = @import("std"))
    pub fn parseConstImport(self: *DeclarationParser) !ast.AstNode {
        const const_token = self.base.previous(); // The 'const' token
        const name_token = try self.base.consume(.Identifier, "Expected namespace name");
        _ = try self.base.consume(.Equal, "Expected '=' after namespace name");

        // Parse @import("path")
        _ = try self.base.consume(.At, "Expected '@' before import");
        _ = try self.base.consume(.Import, "Expected 'import' after '@'");
        _ = try self.base.consume(.LeftParen, "Expected '(' after 'import'");

        const path_token = try self.base.consume(.StringLiteral, "Expected string path");
        _ = try self.base.consume(.RightParen, "Expected ')' after import path");
        _ = try self.base.consume(.Semicolon, "Expected ';' after const import");

        return ast.AstNode{
            .Import = ast.ImportNode{
                .alias = name_token.lexeme, // Use the const name as alias
                .path = path_token.lexeme,
                .span = self.base.spanFromToken(const_token),
            },
        };
    }

    /// Parse top-level import declaration - unified entry point
    pub fn parseImportDeclaration(self: *DeclarationParser) !ast.AstNode {
        // Handle different import patterns:
        // 1. @import("path") - direct import
        // 2. const name = @import("path") - aliased import

        if (self.base.check(.At)) {
            _ = self.base.advance(); // consume '@'
            return self.parseImport();
        } else if (self.base.check(.Const)) {
            _ = self.base.advance(); // consume 'const'
            return self.parseConstImport();
        } else {
            try self.base.errorAtCurrent("Expected '@' or 'const' for import declaration");
            return error.UnexpectedToken;
        }
    }

    /// Parse contract declaration
    pub fn parseContract(
        self: *DeclarationParser,
        type_parser: *TypeParser,
        expr_parser: *ExpressionParser,
    ) !ast.AstNode {
        return contract.parseContract(self, type_parser, expr_parser);
    }

    /// Parse function declaration
    pub fn parseFunction(
        self: *DeclarationParser,
        type_parser: *TypeParser,
        expr_parser: *ExpressionParser,
    ) !ast.FunctionNode {
        return function_parser.parseFunction(self, type_parser, expr_parser);
    }

    /// Parse variable declaration
    pub fn parseVariableDecl(
        self: *DeclarationParser,
        type_parser: *TypeParser,
        expr_parser: *ExpressionParser,
    ) !ast.Statements.VariableDeclNode {
        return self.parseVariableDeclWithLock(type_parser, expr_parser, false);
    }

    /// Parse struct declaration with complete field definitions
    pub fn parseStruct(self: *DeclarationParser, type_parser: *TypeParser) !ast.AstNode {
        return types.parseStruct(self, type_parser);
    }

    /// Parse enum declaration with value assignments and base types
    pub fn parseEnum(
        self: *DeclarationParser,
        type_parser: *TypeParser,
        expr_parser: *ExpressionParser,
    ) !ast.AstNode {
        return types.parseEnum(self, type_parser, expr_parser);
    }

    /// Parse enum variant value (public wrapper for use by enum parser)
    pub fn parseEnumVariantValue(
        self: *DeclarationParser,
        expr_parser: *ExpressionParser,
        underlying_type: ?ast.Types.TypeInfo,
    ) !ast.Expressions.ExprNode {
        return types.parseEnumVariantValue(self, expr_parser, underlying_type);
    }

    /// Parse log declaration with indexed parameter support
    pub fn parseLogDecl(self: *DeclarationParser, type_parser: *TypeParser) !ast.AstNode {
        const name_token = try self.base.consume(.Identifier, "Expected log name");
        _ = try self.base.consume(.LeftParen, "Expected '(' after log name");

        var fields = std.ArrayList(ast.LogField){};
        defer fields.deinit(self.base.arena.allocator());

        if (!self.base.check(.RightParen)) {
            repeat: while (true) {
                // Optional 'indexed' modifier for searchable fields
                const is_indexed = if (self.base.check(.Identifier) and std.mem.eql(u8, self.base.peek().lexeme, "indexed")) blk: {
                    _ = self.base.advance();
                    break :blk true;
                } else false;

                const field_name = try self.base.consumeIdentifierOrKeyword("Expected field name");
                _ = try self.base.consume(.Colon, "Expected ':' after field name");

                // Use type parser
                type_parser.base.current = self.base.current;
                const field_type = try type_parser.parseTypeWithContext(.LogField);
                self.base.current = type_parser.base.current;

                try fields.append(self.base.arena.allocator(), ast.LogField{
                    .name = field_name.lexeme,
                    .type_info = field_type,
                    .indexed = is_indexed,
                    .span = self.base.spanFromToken(field_name),
                });

                if (!self.base.match(.Comma)) break :repeat;
            }
        }

        _ = try self.base.consume(.RightParen, "Expected ')' after log fields");
        _ = try self.base.consume(.Semicolon, "Expected ';' after error declaration");

        return ast.AstNode{ .LogDecl = ast.LogDeclNode{
            .name = name_token.lexeme,
            .fields = try fields.toOwnedSlice(self.base.arena.allocator()),
            .span = self.base.spanFromToken(name_token),
        } };
    }
    /// Parse contract invariant declaration
    /// Syntax: invariant name(condition);
    /// Parse contract invariant (public wrapper)
    pub fn parseContractInvariant(
        self: *DeclarationParser,
        expr_parser: *ExpressionParser,
    ) !ast.AstNode {
        return contract.parseContractInvariant(self, expr_parser);
    }

    /// Parse ghost declaration (public wrapper)
    pub fn parseGhostDeclaration(
        self: *DeclarationParser,
        type_parser: *TypeParser,
        expr_parser: *ExpressionParser,
    ) !ast.AstNode {
        return contract.parseGhostDeclaration(self, type_parser, expr_parser);
    }

    /// Parse contract member (public wrapper)
    pub fn parseContractMember(
        self: *DeclarationParser,
        type_parser: *TypeParser,
        expr_parser: *ExpressionParser,
    ) !ast.AstNode {
        return contract.parseContractMember(self, type_parser, expr_parser);
    }

    /// Parse function parameter (public wrapper)
    pub fn parseParameter(self: *DeclarationParser, type_parser: *TypeParser) !ast.ParameterNode {
        return function_parser.parseParameter(self, type_parser);
    }

    /// Parse function parameter with default value support (public wrapper)
    pub fn parseParameterWithDefaults(
        self: *DeclarationParser,
        type_parser: *TypeParser,
        expr_parser: *ExpressionParser,
    ) !ast.ParameterNode {
        return function_parser.parseParameterWithDefaults(self, type_parser, expr_parser);
    }

    /// Try to parse @lock annotation, returns variable declaration if found
    pub fn tryParseLockAnnotation(
        self: *DeclarationParser,
        type_parser: *TypeParser,
        expr_parser: *ExpressionParser,
    ) !?ast.Statements.VariableDeclNode {
        if (self.base.check(.At)) {
            const saved_pos = self.base.current;
            _ = self.base.advance(); // consume @
            if (self.base.check(.Identifier) and std.mem.eql(u8, self.base.peek().lexeme, "lock")) {
                _ = self.base.advance(); // consume "lock"

                // Check if this is followed by a variable declaration
                if (self.isMemoryRegionKeyword() or self.base.check(.Let) or self.base.check(.Var)) {
                    return try self.parseVariableDeclWithLock(type_parser, expr_parser, true);
                }
            }

            // Not @lock or not followed by variable declaration, restore position
            self.base.current = saved_pos;
        }
        return null;
    }

    /// Parse variable declaration with lock annotation flag - ENHANCED FOR MEMORY REGIONS
    fn parseVariableDeclWithLock(
        self: *DeclarationParser,
        type_parser: *TypeParser,
        expr_parser: *ExpressionParser,
        is_locked: bool,
    ) !ast.Statements.VariableDeclNode {
        // Parse memory region and variable kind together
        const region_and_kind = try self.parseMemoryRegionAndKind();
        const region = region_and_kind.region;
        const kind = region_and_kind.kind;

        // Tuple unpacking is not supported; use struct destructuring: let .{ a, b } = expr;
        if (self.base.check(.LeftParen)) {
            try self.base.errorAtCurrent("Tuple destructuring is not supported; use '.{ ... }' instead");
            return error.UnexpectedToken;
        }

        // Regular variable declaration
        const name_token = try self.base.consume(.Identifier, "Expected variable name");

        // Type annotation is optional if there's an initializer (enables type inference)
        var var_type: ast.Types.TypeInfo = ast.Types.TypeInfo.unknown();
        if (self.base.match(.Colon)) {
            // Explicit type annotation provided
            type_parser.base.current = self.base.current;
            var_type = try type_parser.parseTypeWithContext(.Variable);
            self.base.current = type_parser.base.current;
        }

        // Parse optional initializer
        var initializer: ?*ast.Expressions.ExprNode = null;
        if (self.base.match(.Equal)) {
            // Use expression parser
            expr_parser.base.current = self.base.current;
            const expr = try expr_parser.parseExpression();
            self.base.current = expr_parser.base.current;
            const expr_ptr = try self.base.arena.createNode(ast.Expressions.ExprNode);
            expr_ptr.* = expr;
            initializer = expr_ptr;
        } else if (!var_type.isResolved()) {
            // No type annotation and no initializer - error
            try self.base.errorAtCurrent("Variable declaration requires either a type annotation or an initializer");
            return error.UnexpectedToken;
        }

        _ = try self.base.consume(.Semicolon, "Expected ';' after error declaration");

        return ast.Statements.VariableDeclNode{
            .name = name_token.lexeme,
            .region = region,
            .kind = kind,
            .locked = is_locked,
            .type_info = var_type,
            .value = initializer,
            .span = ParserCommon.makeSpan(name_token),
            .tuple_names = null,
        };
    }

    /// Parse memory region and variable kind together
    fn parseMemoryRegionAndKind(self: *DeclarationParser) !struct { region: ast.Memory.Region, kind: ast.Memory.VariableKind } {
        // Handle const and immutable as special cases (they define both region and kind)
        if (self.base.match(.Const)) {
            return .{ .region = .Stack, .kind = .Const };
        }
        if (self.base.match(.Immutable)) {
            return .{ .region = .Stack, .kind = .Immutable };
        }

        // Parse explicit memory region qualifiers
        var region: ast.Memory.Region = .Stack; // Default to stack
        if (self.base.match(.Storage)) {
            region = .Storage;
        } else if (self.base.match(.Memory)) {
            region = .Memory;
        } else if (self.base.match(.Tstore)) {
            region = .TStore;
        }

        // Parse variable kind (var/let) - required for non-const/immutable variables
        var kind: ast.Memory.VariableKind = undefined;
        if (self.base.match(.Var)) {
            kind = .Var;
        } else if (self.base.match(.Let)) {
            kind = .Let;
        } else {
            // If no explicit var/let, default based on context
            // For storage/memory/tstore without explicit kind, default to var (mutable)
            // For stack without explicit kind, default to let (immutable)
            kind = if (region == .Stack) .Let else .Var;
        }

        return .{ .region = region, .kind = kind };
    }

    /// Parse constant declaration (const NAME: type = value;)
    /// Parse constant declaration (public for use by contract module)
    pub fn parseConstantDecl(
        self: *DeclarationParser,
        type_parser: *TypeParser,
        expr_parser: *ExpressionParser,
    ) !ast.ConstantNode {
        // Consume the 'const' keyword
        _ = try self.base.consume(.Const, "Expected 'const' keyword");

        // Parse constant name
        const name_token = try self.base.consume(.Identifier, "Expected constant name");

        // Parse type annotation
        _ = try self.base.consume(.Colon, "Expected ':' after constant name");
        type_parser.base.current = self.base.current;
        const const_type = try type_parser.parseTypeWithContext(.Variable);
        self.base.current = type_parser.base.current;

        // Parse initializer
        _ = try self.base.consume(.Equal, "Expected '=' after constant type");
        expr_parser.base.current = self.base.current;
        const value_expr = try expr_parser.parseExpression();
        self.base.current = expr_parser.base.current;

        // Create the value expression node
        const value_ptr = try self.base.arena.createNode(ast.Expressions.ExprNode);
        value_ptr.* = value_expr;

        _ = try self.base.consume(.Semicolon, "Expected ';' after constant declaration");

        return ast.ConstantNode{
            .name = name_token.lexeme,
            .typ = const_type,
            .value = value_ptr,
            .visibility = .Private, // Constants are private by default
            .span = ParserCommon.makeSpan(name_token),
        };
    }

    /// Parse error declaration with optional parameter list (error ErrorName(param: type);)
    /// Public for use by contract module
    pub fn parseErrorDecl(self: *DeclarationParser, type_parser: *TypeParser) !ast.Statements.ErrorDeclNode {
        const name_token = try self.base.consume(.Identifier, "Expected error name");

        // Parse optional parameter list
        var parameters: ?[]ast.ParameterNode = null;
        if (self.base.match(.LeftParen)) {
            var params = std.ArrayList(ast.ParameterNode){};
            defer params.deinit(self.base.arena.allocator());

            if (!self.base.check(.RightParen)) {
                repeat: while (true) {
                    const param = try self.parseParameter(type_parser);
                    try params.append(self.base.arena.allocator(), param);

                    if (!self.base.match(.Comma)) break :repeat;
                }
            }

            _ = try self.base.consume(.RightParen, "Expected ')' after error parameters");
            parameters = try params.toOwnedSlice(self.base.arena.allocator());
        }

        _ = try self.base.consume(.Semicolon, "Expected ';' after error declaration");

        return ast.Statements.ErrorDeclNode{
            .name = name_token.lexeme,
            .parameters = parameters,
            .span = self.base.spanFromToken(name_token),
        };
    }

    pub fn parseErrorDeclTopLevel(self: *DeclarationParser, type_parser: *TypeParser) !ast.Statements.ErrorDeclNode {
        return self.parseErrorDecl(type_parser);
    }

    /// Parse block (temporary - should delegate to statement parser)
    fn parseBlock(self: *DeclarationParser) !ast.Statements.BlockNode {
        _ = try self.base.consume(.LeftBrace, "Expected '{'");

        var statements = std.ArrayList(ast.Statements.StmtNode).init(self.base.arena.allocator());
        defer statements.deinit();

        while (!self.base.check(.RightBrace) and !self.base.isAtEnd()) {
            // Statement parsing should be handled by statement_parser.zig
            // This is a placeholder until proper integration is implemented
            _ = self.base.advance();
        }

        const end_token = try self.base.consume(.RightBrace, "Expected '}' after block");

        return ast.Statements.BlockNode{
            .statements = try statements.toOwnedSlice(self.base.arena.allocator()),
            .span = self.base.spanFromToken(end_token),
        };
    }

    /// Check if current token is a memory region keyword
    /// Public for use by contract module
    pub fn isMemoryRegionKeyword(self: *DeclarationParser) bool {
        return ParserCommon.isMemoryRegionKeyword(self.base.peek().type);
    }
};
