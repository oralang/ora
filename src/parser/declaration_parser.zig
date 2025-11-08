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
        const name_token = try self.base.consume(.Identifier, "Expected contract name");

        // Parse optional inheritance: contract Child extends Parent
        // TODO: Add Extends and Implements tokens to lexer for proper keyword support
        var extends: ?[]const u8 = null;
        if (self.base.check(.Identifier) and std.mem.eql(u8, self.base.peek().lexeme, "extends")) {
            _ = self.base.advance(); // consume "extends"
            const parent_token = try self.base.consume(.Identifier, "Expected parent contract name after 'extends'");
            extends = parent_token.lexeme;
        }

        // Parse optional interface implementations: contract MyContract implements Interface1, Interface2
        var implements = std.ArrayList([]const u8){};
        defer implements.deinit(self.base.arena.allocator());

        if (self.base.check(.Identifier) and std.mem.eql(u8, self.base.peek().lexeme, "implements")) {
            _ = self.base.advance(); // consume "implements"
            // Parse comma-separated list of interface names
            repeat: while (true) {
                const interface_token = try self.base.consume(.Identifier, "Expected interface name");
                try implements.append(self.base.arena.allocator(), interface_token.lexeme);

                if (!self.base.match(.Comma)) break :repeat;
            }
        }

        _ = try self.base.consume(.LeftBrace, "Expected '{' after contract declaration");

        var body = std.ArrayList(ast.AstNode){};
        defer body.deinit(self.base.arena.allocator());
        errdefer {
            // Clean up any partially parsed members on error
            for (body.items) |*member| {
                ast.deinitAstNode(self.base.arena.allocator(), member);
            }
        }

        while (!self.base.check(.RightBrace) and !self.base.isAtEnd()) {
            const member = try self.parseContractMember(type_parser, expr_parser);
            try body.append(self.base.arena.allocator(), member);
        }

        _ = try self.base.consume(.RightBrace, "Expected '}' after contract body");

        return ast.AstNode{ .Contract = ast.ContractNode{
            .name = name_token.lexeme,
            .extends = extends,
            .implements = try implements.toOwnedSlice(self.base.arena.allocator()),
            .attributes = &[_]u8{},
            .body = try body.toOwnedSlice(self.base.arena.allocator()),
            .span = self.base.spanFromToken(name_token),
        } };
    }

    /// Parse function declaration
    pub fn parseFunction(
        self: *DeclarationParser,
        type_parser: *TypeParser,
        expr_parser: *ExpressionParser,
    ) !ast.FunctionNode {
        // Parse inline modifier
        const is_inline = self.base.match(.Inline);

        // Parse visibility
        const is_pub = self.base.match(.Pub);

        _ = try self.base.consume(.Fn, "Expected 'fn'");

        // Allow both regular identifiers and 'init' keyword as function names
        const name_token = if (self.base.check(.Identifier))
            self.base.advance()
        else if (self.base.check(.Init))
            self.base.advance()
        else {
            try self.base.errorAtCurrent("Expected function name");
            return error.ExpectedToken;
        };
        const is_init_fn = std.mem.eql(u8, name_token.lexeme, "init");

        // Disallow inline on init
        if (is_inline and is_init_fn) {
            try self.base.errorAtCurrent("'init' cannot be marked inline");
            return error.UnexpectedToken;
        }
        _ = try self.base.consume(.LeftParen, "Expected '(' after function name");

        // Parse parameters with default values support
        var params = std.ArrayList(ast.ParameterNode){};
        defer params.deinit(self.base.arena.allocator());
        errdefer {
            // Clean up parameters on error
            for (params.items) |*param| {
                if (param.default_value) |default_val| {
                    ast.deinitExprNode(self.base.arena.allocator(), default_val);
                    self.base.arena.allocator().destroy(default_val);
                }
            }
        }

        if (!self.base.check(.RightParen)) {
            repeat: while (true) {
                const param = try self.parseParameterWithDefaults(type_parser, expr_parser);
                try params.append(self.base.arena.allocator(), param);

                if (!self.base.match(.Comma)) break :repeat;
            }
        }

        _ = try self.base.consume(.RightParen, "Expected ')' after parameters");

        // Parse optional return type using arrow syntax: fn foo(...) -> Type
        var return_type_info: ?ast.Types.TypeInfo = null;
        if (self.base.check(.Arrow)) {
            if (is_init_fn) {
                try self.base.errorAtCurrent("'init' cannot have a return type");
                return error.UnexpectedToken;
            }
            _ = self.base.advance(); // consume '->'
            // Use type parser
            type_parser.base.current = self.base.current;
            const parsed_type = try type_parser.parseReturnType();
            self.base.current = type_parser.base.current;
            return_type_info = parsed_type;
        }

        // Parse requires clauses
        var requires_clauses = std.ArrayList(*ast.Expressions.ExprNode){};
        defer requires_clauses.deinit(self.base.arena.allocator());

        while (self.base.match(.Requires)) {
            _ = try self.base.consume(.LeftParen, "Expected '(' after 'requires'");

            // Parse the condition expression
            expr_parser.base.current = self.base.current;
            const condition = try expr_parser.parseExpression();
            self.base.current = expr_parser.base.current;

            _ = try self.base.consume(.RightParen, "Expected ')' after requires condition");
            if (self.base.match(.Semicolon)) {
                try self.base.errorAtCurrent("Unexpected ';' after requires(...) (no semicolon allowed)");
                return error.UnexpectedToken;
            }

            // Store the expression in arena
            const condition_ptr = try self.base.arena.createNode(ast.Expressions.ExprNode);
            condition_ptr.* = condition;
            try requires_clauses.append(self.base.arena.allocator(), condition_ptr);
        }

        // Parse ensures clauses
        var ensures_clauses = std.ArrayList(*ast.Expressions.ExprNode){};
        defer ensures_clauses.deinit(self.base.arena.allocator());

        while (self.base.match(.Ensures)) {
            _ = try self.base.consume(.LeftParen, "Expected '(' after 'ensures'");

            // Parse the condition expression
            expr_parser.base.current = self.base.current;
            const condition = try expr_parser.parseExpression();
            self.base.current = expr_parser.base.current;

            _ = try self.base.consume(.RightParen, "Expected ')' after ensures condition");
            if (self.base.match(.Semicolon)) {
                try self.base.errorAtCurrent("Unexpected ';' after ensures(...) (no semicolon allowed)");
                return error.UnexpectedToken;
            }

            // Store the expression in arena
            const condition_ptr = try self.base.arena.createNode(ast.Expressions.ExprNode);
            condition_ptr.* = condition;
            try ensures_clauses.append(self.base.arena.allocator(), condition_ptr);
        }

        // Parse modifies clause (frame conditions)
        var modifies_clause: ?[]*ast.Expressions.ExprNode = null;
        if (self.base.match(.Modifies)) {
            _ = try self.base.consume(.LeftParen, "Expected '(' after 'modifies'");

            var modifies_list = std.ArrayList(*ast.Expressions.ExprNode){};
            defer modifies_list.deinit(self.base.arena.allocator());

            // Parse expression list (comma-separated)
            if (!self.base.check(.RightParen)) {
                repeat: while (true) {
                    expr_parser.base.current = self.base.current;
                    const expr = try expr_parser.parseExpression();
                    self.base.current = expr_parser.base.current;

                    const expr_ptr = try self.base.arena.createNode(ast.Expressions.ExprNode);
                    expr_ptr.* = expr;
                    try modifies_list.append(self.base.arena.allocator(), expr_ptr);

                    if (!self.base.match(.Comma)) break :repeat;
                }
            }

            _ = try self.base.consume(.RightParen, "Expected ')' after modifies list");
            if (self.base.match(.Semicolon)) {
                try self.base.errorAtCurrent("Unexpected ';' after modifies(...) (no semicolon allowed)");
                return error.UnexpectedToken;
            }

            modifies_clause = try modifies_list.toOwnedSlice(self.base.arena.allocator());
        }

        // Parse function body - delegate to statement parser
        // Placeholder body: parser_core will parse the real block with StatementParser
        // Do not consume any tokens here; leave current at '{'
        const body = ast.Statements.BlockNode{ .statements = &[_]ast.Statements.StmtNode{}, .span = self.base.currentSpan() };

        return ast.FunctionNode{
            .name = name_token.lexeme,
            .parameters = try params.toOwnedSlice(self.base.arena.allocator()),
            .return_type_info = return_type_info,
            .body = body,
            .visibility = if (is_pub) ast.Visibility.Public else ast.Visibility.Private,
            .attributes = &[_]u8{}, // Empty attributes
            .is_inline = is_inline,
            .requires_clauses = try requires_clauses.toOwnedSlice(self.base.arena.allocator()),
            .ensures_clauses = try ensures_clauses.toOwnedSlice(self.base.arena.allocator()),
            .modifies_clause = modifies_clause,
            .span = self.base.spanFromToken(name_token),
        };
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
        const name_token = try self.base.consume(.Identifier, "Expected struct name");
        _ = try self.base.consume(.LeftBrace, "Expected '{' after struct name");

        var fields = std.ArrayList(ast.StructField){};
        defer fields.deinit(self.base.arena.allocator());

        while (!self.base.check(.RightBrace) and !self.base.isAtEnd()) {
            // Skip any field attributes/annotations for now
            while (self.base.match(.At)) {
                _ = try self.base.consume(.Identifier, "Expected annotation name after '@'");
                if (self.base.match(.LeftParen)) {
                    var paren_depth: u32 = 1;
                    while (paren_depth > 0 and !self.base.isAtEnd()) {
                        if (self.base.match(.LeftParen)) {
                            paren_depth += 1;
                        } else if (self.base.match(.RightParen)) {
                            paren_depth -= 1;
                        } else {
                            _ = self.base.advance();
                        }
                    }
                }
            }

            const field_name = try self.base.consumeIdentifierOrKeyword("Expected field name");
            _ = try self.base.consume(.Colon, "Expected ':' after field name");

            // Use type parser with complete type information
            type_parser.base.current = self.base.current;
            const field_type = try type_parser.parseTypeWithContext(.StructField);
            self.base.current = type_parser.base.current;

            _ = try self.base.consume(.Semicolon, "Expected ';' after field");

            try fields.append(self.base.arena.allocator(), ast.StructField{
                .name = field_name.lexeme,
                .type_info = field_type,
                .span = self.base.spanFromToken(field_name),
            });
        }

        _ = try self.base.consume(.RightBrace, "Expected '}' after struct fields");

        return ast.AstNode{ .StructDecl = ast.StructDeclNode{
            .name = name_token.lexeme,
            .fields = try fields.toOwnedSlice(self.base.arena.allocator()),
            .span = self.base.spanFromToken(name_token),
        } };
    }

    /// Parse a single enum variant value with proper precedence handling
    /// This avoids using the general expression parser which would interpret commas as operators
    /// The enum's underlying type is used for integer literals instead of generic integers
    fn parseEnumVariantValue(
        self: *DeclarationParser,
        expr_parser: *ExpressionParser,
        underlying_type: ?ast.Types.TypeInfo,
    ) !ast.Expressions.ExprNode {
        // Use the expression parser but stop at comma level to avoid enum separator confusion
        // This ensures proper operator precedence and left-associativity
        expr_parser.base.current = self.base.current;
        var expr = try expr_parser.parseLogicalOr();
        self.base.current = expr_parser.base.current;

        // If this is a simple literal and we have an underlying type for the enum,
        // apply the enum's underlying type immediately. For complex expressions, leave as unknown
        // and let the semantic analyzer resolve with the enum type constraint.
        if (expr == .Literal and underlying_type != null) {
            var updated_type_info = underlying_type.?;
            updated_type_info.source = .explicit; // Mark as explicit since the user provided this value

            switch (expr.Literal) {
                .Integer => |*int_lit| {
                    int_lit.type_info = updated_type_info;
                },
                .String => |*str_lit| {
                    str_lit.type_info = updated_type_info;
                },
                .Bool => |*bool_lit| {
                    bool_lit.type_info = updated_type_info;
                },
                .Address => |*addr_lit| {
                    addr_lit.type_info = updated_type_info;
                },
                .Hex => |*hex_lit| {
                    hex_lit.type_info = updated_type_info;
                },
                .Binary => |*bin_lit| {
                    bin_lit.type_info = updated_type_info;
                },
                .Character => |*char_lit| {
                    char_lit.type_info = updated_type_info;
                },
                .Bytes => |*bytes_lit| {
                    bytes_lit.type_info = updated_type_info;
                },
            }
        }
        // For complex expressions, we leave them as-is with unknown types
        // The semantic analyzer will resolve them with the enum type constraint

        return expr;
    }

    /// Parse enum declaration with value assignments and base types
    pub fn parseEnum(
        self: *DeclarationParser,
        type_parser: *TypeParser,
        expr_parser: *ExpressionParser,
    ) !ast.AstNode {
        const name_token = try self.base.consume(.Identifier, "Expected enum name");

        // Parse optional underlying type: enum Status : u8 { ... }
        var base_type: ?ast.Types.TypeInfo = null; // No default - let type system infer
        if (self.base.match(.Colon)) {
            // Use type parser for underlying type
            type_parser.base.current = self.base.current;
            base_type = try type_parser.parseTypeWithContext(.EnumUnderlying);
            self.base.current = type_parser.base.current;
        }

        _ = try self.base.consume(.LeftBrace, "Expected '{' after enum name");

        var variants = std.ArrayList(ast.EnumVariant){};
        defer variants.deinit(self.base.arena.allocator());

        // Track if any variants have explicit values
        var has_explicit_values = false;

        // Track the current implicit value, starting from 0
        var next_implicit_value: u32 = 0;

        // Parse all enum variants
        while (!self.base.check(.RightBrace) and !self.base.isAtEnd()) {
            const variant_name = try self.base.consume(.Identifier, "Expected variant name");

            // Parse optional explicit value assignment: VariantName = value
            var value: ?ast.Expressions.ExprNode = null;
            if (self.base.match(.Equal)) {
                has_explicit_values = true; // Mark that we have at least one explicit value

                // Use our dedicated enum variant value parser to ensure complex expressions
                // like (1 + 2) * 3 are parsed correctly without misinterpreting operators
                // as enum variant separators
                const expr = try self.parseEnumVariantValue(expr_parser, base_type);
                value = expr;

                // If this is an integer literal, we need to update the next_implicit_value
                if (expr == .Literal and expr.Literal == .Integer) {
                    // Try to parse the integer value to determine the next implicit value
                    const int_str = expr.Literal.Integer.value;
                    if (std.fmt.parseInt(u32, int_str, 0)) |int_val| {
                        // Set the next implicit value to one more than this explicit value
                        next_implicit_value = int_val + 1;
                    } else |_| {
                        // If we can't parse it (e.g., it's a complex expression),
                        // just continue with the current next_implicit_value
                    }
                }
            } else {
                // Create an implicit value for this variant
                // We always add integer values even when not explicitly assigned
                const int_literal = ast.expressions.IntegerLiteral{
                    .value = try std.fmt.allocPrint(self.base.arena.allocator(), "{d}", .{next_implicit_value}),
                    .type_info = if (base_type) |bt| blk: {
                        var type_info = bt;
                        type_info.source = .inferred; // Mark as inferred since compiler generated this
                        break :blk type_info;
                    } else blk: {
                        // Default to u32 for enum values if no underlying type specified
                        var type_info = ast.Types.TypeInfo.fromOraType(.u32);
                        type_info.source = .inferred;
                        break :blk type_info;
                    },
                    .span = self.base.spanFromToken(variant_name),
                };

                // Create an ExprNode with the integer literal (no need to store in arena)
                value = ast.Expressions.ExprNode{ .Literal = .{ .Integer = int_literal } };

                // Increment the implicit value for the next variant
                next_implicit_value += 1;
            }

            // Add the variant to our list
            try variants.append(self.base.arena.allocator(), ast.EnumVariant{
                .name = variant_name.lexeme,
                .value = value,
                .span = self.base.spanFromToken(variant_name),
            });

            // Check for comma separator or end of enum body
            if (self.base.check(.RightBrace)) {
                break; // No more variants
            } else if (!self.base.match(.Comma)) {
                // If not a comma and not the end, it's an error
                try self.base.errorAtCurrent("Expected ',' between enum variants");
                return error.ExpectedToken;
            }
        }

        _ = try self.base.consume(.RightBrace, "Expected '}' after enum variants");

        // Create and return the enum declaration node
        return ast.AstNode{
            .EnumDecl = ast.EnumDeclNode{
                .name = name_token.lexeme,
                .variants = try variants.toOwnedSlice(self.base.arena.allocator()),
                .underlying_type_info = base_type,
                .span = self.base.spanFromToken(name_token),
                .has_explicit_values = true, // We now always have values (implicit or explicit)
            },
        };
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
    pub fn parseContractInvariant(
        self: *DeclarationParser,
        expr_parser: *ExpressionParser,
    ) !ast.AstNode {
        const invariant_token = self.base.previous(); // The 'invariant' keyword

        const name_token = try self.base.consume(.Identifier, "Expected invariant name");
        const name = try self.base.arena.createString(name_token.lexeme);

        _ = try self.base.consume(.LeftParen, "Expected '(' after invariant name");

        // Parse the condition expression
        expr_parser.base.current = self.base.current;
        const condition = try expr_parser.parseExpression();
        self.base.current = expr_parser.base.current;

        _ = try self.base.consume(.RightParen, "Expected ')' after invariant condition");
        _ = try self.base.consume(.Semicolon, "Expected ';' after invariant declaration");

        // Store the expression in arena
        const condition_ptr = try self.base.arena.createNode(ast.Expressions.ExprNode);
        condition_ptr.* = condition;

        return ast.AstNode{ .ContractInvariant = ast.ContractInvariant{
            .name = name,
            .condition = condition_ptr,
            .span = self.base.spanFromToken(invariant_token),
            .is_specification = true,
        } };
    }

    /// Parse ghost declaration (ghost variable/function/block)
    pub fn parseGhostDeclaration(
        self: *DeclarationParser,
        type_parser: *TypeParser,
        expr_parser: *ExpressionParser,
    ) !ast.AstNode {
        _ = try self.base.consume(.Ghost, "Expected 'ghost' keyword");

        // Ghost can precede: variable declaration, function declaration, or block
        if (self.base.check(.Pub) or self.base.check(.Fn) or self.base.check(.Inline)) {
            // Ghost function
            var fn_node = try self.parseFunction(type_parser, expr_parser);

            // Parse the function body block using a local StatementParser
            var stmt_parser = StatementParser.init(self.base.tokens, self.base.arena);
            stmt_parser.base.current = self.base.current;
            const body_block = try stmt_parser.parseBlock();

            // Update current position from statement parser
            self.base.current = stmt_parser.base.current;

            // Attach the parsed body
            fn_node.body = body_block;
            fn_node.is_ghost = true; // Mark as ghost

            return ast.AstNode{ .Function = fn_node };
        } else if (self.isMemoryRegionKeyword() or self.base.check(.Let) or self.base.check(.Var) or self.base.check(.Immutable) or self.base.check(.Const)) {
            // Ghost variable or constant
            if (self.base.check(.Const)) {
                var const_node = try self.parseConstantDecl(type_parser, expr_parser);
                const_node.is_ghost = true; // Mark as ghost
                return ast.AstNode{ .Constant = const_node };
            } else {
                var var_node = try self.parseVariableDecl(type_parser, expr_parser);
                var_node.is_ghost = true; // Mark as ghost
                return ast.AstNode{ .VariableDecl = var_node };
            }
        } else if (self.base.check(.LeftBrace)) {
            // Ghost block - parse as a statement block
            var stmt_parser = StatementParser.init(self.base.tokens, self.base.arena);
            stmt_parser.base.current = self.base.current;
            var ghost_block = try stmt_parser.parseBlock();
            self.base.current = stmt_parser.base.current;

            ghost_block.is_ghost = true; // Mark as ghost

            return ast.AstNode{ .Block = ghost_block };
        } else {
            try self.base.errorAtCurrent("Expected function, variable, or block after 'ghost'");
            return error.UnexpectedToken;
        }
    }

    /// Parse contract member (function, variable, etc.) with proper scoping
    pub fn parseContractMember(
        self: *DeclarationParser,
        type_parser: *TypeParser,
        expr_parser: *ExpressionParser,
    ) !ast.AstNode {
        // Check for contract invariants
        if (self.base.check(.Invariant)) {
            _ = self.base.advance(); // consume 'invariant'
            return try self.parseContractInvariant(expr_parser);
        }

        // Check for ghost declarations
        if (self.base.check(.Ghost)) {
            return try self.parseGhostDeclaration(type_parser, expr_parser);
        }

        // Check for @lock annotation before variable declarations
        if (try self.tryParseLockAnnotation(type_parser, expr_parser)) |var_decl| {
            return ast.AstNode{ .VariableDecl = var_decl };
        }

        // Reject unknown @ directives at declaration position
        if (self.base.check(.At)) {
            const saved = self.base.current;
            _ = self.base.advance();
            // Only @lock is meaningful before a variable declaration (handled above)
            const is_known = self.base.check(.Identifier) and std.mem.eql(u8, self.base.peek().lexeme, "lock");
            if (!is_known) {
                _ = self.base.errorAtCurrent("Unknown @ directive before declaration; attributes are not supported") catch {};
                // Recover: skip to semicolon or next declaration boundary
                while (!self.base.isAtEnd() and !self.base.check(.Semicolon) and !self.base.check(.RightBrace)) {
                    _ = self.base.advance();
                }
                _ = self.base.match(.Semicolon);
            } else {
                // Restore for lock-annotation path to handle normally
                self.base.current = saved;
            }
        }

        // Functions (can be public or private within contracts)
        if (self.base.check(.Pub) or self.base.check(.Fn) or self.base.check(.Inline)) {
            // Parse the function header first (leaves current at '{')
            var fn_node = try self.parseFunction(type_parser, expr_parser);

            // Parse the function body block using a local StatementParser
            var stmt_parser = StatementParser.init(self.base.tokens, self.base.arena);
            stmt_parser.base.current = self.base.current;
            const body_block = try stmt_parser.parseBlock();

            // Update current position from statement parser
            self.base.current = stmt_parser.base.current;

            // Attach the parsed body
            fn_node.body = body_block;

            return ast.AstNode{ .Function = fn_node };
        }

        // Constant declarations (contract constants) - check before variables
        if (self.base.check(.Const)) {
            return ast.AstNode{ .Constant = try self.parseConstantDecl(type_parser, expr_parser) };
        }

        // Variable declarations (contract state variables)
        if (self.isMemoryRegionKeyword() or self.base.check(.Let) or self.base.check(.Var) or self.base.check(.Immutable)) {
            return ast.AstNode{ .VariableDecl = try self.parseVariableDecl(type_parser, expr_parser) };
        }

        // Error declarations (contract-specific errors)
        if (self.base.match(.Error)) {
            return ast.AstNode{ .ErrorDecl = try self.parseErrorDecl(type_parser) };
        }

        // Log declarations (contract events)
        if (self.base.match(.Log)) {
            return self.parseLogDecl(type_parser);
        }

        // Struct declarations (contract-scoped structs)
        if (self.base.match(.Struct)) {
            return self.parseStruct(type_parser);
        }

        // Enum declarations (contract-scoped enums)
        if (self.base.match(.Enum)) {
            return self.parseEnum(type_parser, expr_parser);
        }

        try self.base.errorAtCurrent("Expected contract member (function, variable, struct, enum, log, or error)");
        return error.UnexpectedToken;
    }

    /// Parse function parameter
    fn parseParameter(self: *DeclarationParser, type_parser: *TypeParser) !ast.ParameterNode {
        const name_token = try self.base.consumeIdentifierOrKeyword("Expected parameter name");
        _ = try self.base.consume(.Colon, "Expected ':' after parameter name");
        // Use type parser
        type_parser.base.current = self.base.current;
        const param_type = try type_parser.parseTypeWithContext(.Parameter);
        self.base.current = type_parser.base.current;

        return ast.ParameterNode{
            .name = name_token.lexeme,
            .type_info = param_type,
            .is_mutable = false, // Default to immutable
            .default_value = null, // No default value
            .span = self.base.spanFromToken(name_token),
        };
    }

    /// Parse function parameter with default value support
    fn parseParameterWithDefaults(
        self: *DeclarationParser,
        type_parser: *TypeParser,
        expr_parser: *ExpressionParser,
    ) !ast.ParameterNode {
        // Check for mutable parameter modifier (mut param_name)
        const is_mutable = if (self.base.check(.Identifier) and std.mem.eql(u8, self.base.peek().lexeme, "mut")) blk: {
            _ = self.base.advance(); // consume "mut"
            break :blk true;
        } else false;

        const name_token = try self.base.consumeIdentifierOrKeyword("Expected parameter name");
        _ = try self.base.consume(.Colon, "Expected ':' after parameter name");

        // Use type parser
        type_parser.base.current = self.base.current;
        const param_type = try type_parser.parseTypeWithContext(.Parameter);
        self.base.current = type_parser.base.current;

        // Parse optional default value
        var default_value: ?*ast.Expressions.ExprNode = null;
        if (self.base.match(.Equal)) {
            // Parse default value expression
            expr_parser.base.current = self.base.current;
            const expr = try expr_parser.parseExpression();
            self.base.current = expr_parser.base.current;

            // Store in arena
            const expr_ptr = try self.base.arena.createNode(ast.Expressions.ExprNode);
            expr_ptr.* = expr;
            default_value = expr_ptr;
        }

        return ast.ParameterNode{
            .name = name_token.lexeme,
            .type_info = param_type,
            .is_mutable = is_mutable,
            .default_value = default_value,
            .span = ParserCommon.makeSpan(name_token),
        };
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
        var var_type: ast.Types.TypeInfo = undefined;

        // Support both explicit type and type inference
        if (self.base.match(.Colon)) {
            // Explicit type: let x: u32 = value
            type_parser.base.current = self.base.current;
            var_type = try type_parser.parseTypeWithContext(.Variable);
            self.base.current = type_parser.base.current;
        } else if (self.base.check(.Equal)) {
            // Type inference: let x = value
            var_type = ast.Types.TypeInfo.unknown(); // Will be inferred
        } else {
            try self.base.errorAtCurrent("Expected ':' for type annotation or '=' for assignment");
            return error.ExpectedToken;
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
    fn parseConstantDecl(
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
    fn parseErrorDecl(self: *DeclarationParser, type_parser: *TypeParser) !ast.Statements.ErrorDeclNode {
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
    fn isMemoryRegionKeyword(self: *DeclarationParser) bool {
        return ParserCommon.isMemoryRegionKeyword(self.base.peek().type);
    }
};
