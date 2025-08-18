const std = @import("std");
const lexer = @import("../lexer.zig");
const ast = @import("../ast.zig");
const common = @import("common.zig");

const Token = lexer.Token;
const TokenType = lexer.TokenType;
const BaseParser = common.BaseParser;
const ParserCommon = common.ParserCommon;
const ParserError = @import("parser_core.zig").ParserError;

// Import other parsers for cross-parser communication
const TypeParser = @import("type_parser.zig").TypeParser;
const ExpressionParser = @import("expression_parser.zig").ExpressionParser;

/// Specialized parser for declarations (functions, structs, enums, etc.)
pub const DeclarationParser = struct {
    base: BaseParser,
    type_parser: TypeParser,
    expr_parser: ExpressionParser,

    pub fn init(tokens: []const Token, arena: *@import("../ast/ast_arena.zig").AstArena) DeclarationParser {
        return DeclarationParser{
            .base = BaseParser.init(tokens, arena),
            .type_parser = TypeParser.init(tokens, arena),
            .expr_parser = ExpressionParser.init(tokens, arena),
        };
    }

    /// Sync sub-parser states with current position
    fn syncSubParsers(self: *DeclarationParser) void {
        self.type_parser.base.current = self.base.current;
        self.expr_parser.base.current = self.base.current;
    }

    /// Update current position from sub-parser
    fn updateFromSubParser(self: *DeclarationParser, new_current: usize) void {
        self.base.current = new_current;
        self.syncSubParsers();
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
    pub fn parseContract(self: *DeclarationParser) !ast.AstNode {
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
        var implements = std.ArrayList([]const u8).init(self.base.arena.allocator());
        defer implements.deinit();

        if (self.base.check(.Identifier) and std.mem.eql(u8, self.base.peek().lexeme, "implements")) {
            _ = self.base.advance(); // consume "implements"
            // Parse comma-separated list of interface names
            repeat: while (true) {
                const interface_token = try self.base.consume(.Identifier, "Expected interface name");
                try implements.append(interface_token.lexeme);

                if (!self.base.match(.Comma)) break :repeat;
            }
        }

        _ = try self.base.consume(.LeftBrace, "Expected '{' after contract declaration");

        var body = std.ArrayList(ast.AstNode).init(self.base.arena.allocator());
        defer body.deinit();
        errdefer {
            // Clean up any partially parsed members on error
            for (body.items) |*member| {
                ast.deinitAstNode(self.base.arena.allocator(), member);
            }
        }

        while (!self.base.check(.RightBrace) and !self.base.isAtEnd()) {
            const member = try self.parseContractMember();
            try body.append(member);
        }

        _ = try self.base.consume(.RightBrace, "Expected '}' after contract body");

        return ast.AstNode{ .Contract = ast.ContractNode{
            .name = name_token.lexeme,
            .extends = extends,
            .implements = try implements.toOwnedSlice(),
            .attributes = &[_]u8{},
            .body = try body.toOwnedSlice(),
            .span = self.base.spanFromToken(name_token),
        } };
    }

    /// Parse function declaration
    pub fn parseFunction(self: *DeclarationParser) !ast.FunctionNode {
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
        _ = try self.base.consume(.LeftParen, "Expected '(' after function name");

        // Parse parameters with default values support
        var params = std.ArrayList(ast.ParameterNode).init(self.base.arena.allocator());
        defer params.deinit();
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
                const param = try self.parseParameterWithDefaults();
                try params.append(param);

                if (!self.base.match(.Comma)) break :repeat;
            }
        }

        _ = try self.base.consume(.RightParen, "Expected ')' after parameters");

        // Parse optional return type using arrow syntax: fn foo(...) -> Type
        var return_type_info: ?ast.Types.TypeInfo = null;
        if (self.base.match(.Arrow)) {
            // Use integrated type parser
            self.syncSubParsers();
            const parsed_type = try self.type_parser.parseReturnType();
            return_type_info = parsed_type;
            self.updateFromSubParser(self.type_parser.base.current);
        }

        // Parse requires clauses
        var requires_clauses = std.ArrayList(*ast.Expressions.ExprNode).init(self.base.arena.allocator());
        defer requires_clauses.deinit();

        while (self.base.match(.Requires)) {
            _ = try self.base.consume(.LeftParen, "Expected '(' after 'requires'");

            // Parse the condition expression
            self.syncSubParsers();
            const condition = try self.expr_parser.parseExpression();
            self.updateFromSubParser(self.expr_parser.base.current);

            _ = try self.base.consume(.RightParen, "Expected ')' after requires condition");
            if (self.base.match(.Semicolon)) {
                try self.base.errorAtCurrent("Unexpected ';' after requires(...) (no semicolon allowed)");
                return error.UnexpectedToken;
            }

            // Store the expression in arena
            const condition_ptr = try self.base.arena.createNode(ast.Expressions.ExprNode);
            condition_ptr.* = condition;
            try requires_clauses.append(condition_ptr);
        }

        // Parse ensures clauses
        var ensures_clauses = std.ArrayList(*ast.Expressions.ExprNode).init(self.base.arena.allocator());
        defer ensures_clauses.deinit();

        while (self.base.match(.Ensures)) {
            _ = try self.base.consume(.LeftParen, "Expected '(' after 'ensures'");

            // Parse the condition expression
            self.syncSubParsers();
            const condition = try self.expr_parser.parseExpression();
            self.updateFromSubParser(self.expr_parser.base.current);

            _ = try self.base.consume(.RightParen, "Expected ')' after ensures condition");
            if (self.base.match(.Semicolon)) {
                try self.base.errorAtCurrent("Unexpected ';' after ensures(...) (no semicolon allowed)");
                return error.UnexpectedToken;
            }

            // Store the expression in arena
            const condition_ptr = try self.base.arena.createNode(ast.Expressions.ExprNode);
            condition_ptr.* = condition;
            try ensures_clauses.append(condition_ptr);
        }

        // Parse function body - delegate to statement parser
        // Placeholder body: parser_core will parse the real block with StatementParser
        // Do not consume any tokens here; leave current at '{'
        const body = ast.Statements.BlockNode{ .statements = &[_]ast.Statements.StmtNode{}, .span = self.base.currentSpan() };

        return ast.FunctionNode{
            .name = name_token.lexeme,
            .parameters = try params.toOwnedSlice(),
            .return_type_info = return_type_info,
            .body = body,
            .visibility = if (is_pub) ast.Visibility.Public else ast.Visibility.Private,
            .attributes = &[_]u8{}, // Empty attributes
            .is_inline = is_inline,
            .requires_clauses = try requires_clauses.toOwnedSlice(),
            .ensures_clauses = try ensures_clauses.toOwnedSlice(),
            .span = self.base.spanFromToken(name_token),
        };
    }

    /// Parse variable declaration
    pub fn parseVariableDecl(self: *DeclarationParser) !ast.Statements.VariableDeclNode {
        return self.parseVariableDeclWithLock(false);
    }

    /// Parse struct declaration with complete field definitions
    pub fn parseStruct(self: *DeclarationParser) !ast.AstNode {
        const name_token = try self.base.consume(.Identifier, "Expected struct name");
        _ = try self.base.consume(.LeftBrace, "Expected '{' after struct name");

        var fields = std.ArrayList(ast.StructField).init(self.base.arena.allocator());
        defer fields.deinit();

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

            // Use integrated type parser with complete type information
            self.syncSubParsers();
            const field_type = try self.type_parser.parseTypeWithContext(.StructField);
            self.updateFromSubParser(self.type_parser.base.current);

            _ = try self.base.consume(.Semicolon, "Expected ';' after field");

            try fields.append(ast.StructField{
                .name = field_name.lexeme,
                .type_info = field_type,
                .span = self.base.spanFromToken(field_name),
            });
        }

        _ = try self.base.consume(.RightBrace, "Expected '}' after struct fields");

        return ast.AstNode{ .StructDecl = ast.StructDeclNode{
            .name = name_token.lexeme,
            .fields = try fields.toOwnedSlice(),
            .span = self.base.spanFromToken(name_token),
        } };
    }

    /// Parse a single enum variant value with proper precedence handling
    /// This avoids using the general expression parser which would interpret commas as operators
    /// The enum's underlying type is used for integer literals instead of generic integers
    fn parseEnumVariantValue(self: *DeclarationParser, underlying_type: ?ast.Types.TypeInfo) !ast.Expressions.ExprNode {
        // Use the expression parser but stop at comma level to avoid enum separator confusion
        // This ensures proper operator precedence and left-associativity
        self.syncSubParsers();
        var expr = try self.expr_parser.parseLogicalOr();
        self.updateFromSubParser(self.expr_parser.base.current);

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
            }
        }
        // For complex expressions, we leave them as-is with unknown types
        // The semantic analyzer will resolve them with the enum type constraint

        return expr;
    }

    /// Parse enum declaration with value assignments and base types
    pub fn parseEnum(self: *DeclarationParser) !ast.AstNode {
        const name_token = try self.base.consume(.Identifier, "Expected enum name");

        // Parse optional underlying type: enum Status : u8 { ... }
        var base_type: ?ast.Types.TypeInfo = null; // No default - let type system infer
        if (self.base.match(.Colon)) {
            // Use integrated type parser for underlying type
            self.syncSubParsers();
            base_type = try self.type_parser.parseTypeWithContext(.EnumUnderlying);
            self.updateFromSubParser(self.type_parser.base.current);
        }

        _ = try self.base.consume(.LeftBrace, "Expected '{' after enum name");

        var variants = std.ArrayList(ast.EnumVariant).init(self.base.arena.allocator());
        defer variants.deinit();

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
                const expr = try self.parseEnumVariantValue(base_type);
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
            try variants.append(ast.EnumVariant{
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
                .variants = try variants.toOwnedSlice(),
                .underlying_type_info = base_type,
                .span = self.base.spanFromToken(name_token),
                .has_explicit_values = true, // We now always have values (implicit or explicit)
            },
        };
    }

    /// Parse log declaration with indexed parameter support
    pub fn parseLogDecl(self: *DeclarationParser) !ast.AstNode {
        const name_token = try self.base.consume(.Identifier, "Expected log name");
        _ = try self.base.consume(.LeftParen, "Expected '(' after log name");

        var fields = std.ArrayList(ast.LogField).init(self.base.arena.allocator());
        defer fields.deinit();

        if (!self.base.check(.RightParen)) {
            repeat: while (true) {
                // Optional 'indexed' modifier for searchable fields
                const is_indexed = if (self.base.check(.Identifier) and std.mem.eql(u8, self.base.peek().lexeme, "indexed")) blk: {
                    _ = self.base.advance();
                    break :blk true;
                } else false;

                const field_name = try self.base.consumeIdentifierOrKeyword("Expected field name");
                _ = try self.base.consume(.Colon, "Expected ':' after field name");

                // Use integrated type parser
                self.syncSubParsers();
                const field_type = try self.type_parser.parseTypeWithContext(.LogField);
                self.updateFromSubParser(self.type_parser.base.current);

                try fields.append(ast.LogField{
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
            .fields = try fields.toOwnedSlice(),
            .span = self.base.spanFromToken(name_token),
        } };
    }
    /// Parse contract member (function, variable, etc.) with proper scoping
    pub fn parseContractMember(self: *DeclarationParser) !ast.AstNode {
        // Check for @lock annotation before variable declarations
        if (try self.tryParseLockAnnotation()) |var_decl| {
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
            return ast.AstNode{ .Function = try self.parseFunction() };
        }

        // Variable declarations (contract state variables)
        if (self.isMemoryRegionKeyword() or self.base.check(.Let) or self.base.check(.Var)) {
            return ast.AstNode{ .VariableDecl = try self.parseVariableDecl() };
        }

        // Constant declarations (contract constants)
        if (self.base.check(.Const)) {
            return ast.AstNode{ .VariableDecl = try self.parseVariableDecl() };
        }

        // Error declarations (contract-specific errors)
        if (self.base.match(.Error)) {
            return ast.AstNode{ .ErrorDecl = try self.parseErrorDecl() };
        }

        // Log declarations (contract events)
        if (self.base.match(.Log)) {
            return self.parseLogDecl();
        }

        // Struct declarations (contract-scoped structs)
        if (self.base.match(.Struct)) {
            return self.parseStruct();
        }

        // Enum declarations (contract-scoped enums)
        if (self.base.match(.Enum)) {
            return self.parseEnum();
        }

        try self.base.errorAtCurrent("Expected contract member (function, variable, struct, enum, log, or error)");
        return error.UnexpectedToken;
    }

    /// Parse function parameter
    fn parseParameter(self: *DeclarationParser) !ast.ParameterNode {
        const name_token = try self.base.consumeIdentifierOrKeyword("Expected parameter name");
        _ = try self.base.consume(.Colon, "Expected ':' after parameter name");
        // Use integrated type parser
        self.syncSubParsers();
        const param_type = try self.type_parser.parseTypeWithContext(.Parameter);
        self.updateFromSubParser(self.type_parser.base.current);

        return ast.ParameterNode{
            .name = name_token.lexeme,
            .type_info = param_type,
            .is_mutable = false, // Default to immutable
            .default_value = null, // No default value
            .span = self.base.spanFromToken(name_token),
        };
    }

    /// Parse function parameter with default value support
    fn parseParameterWithDefaults(self: *DeclarationParser) !ast.ParameterNode {
        // Check for mutable parameter modifier (mut param_name)
        const is_mutable = if (self.base.check(.Identifier) and std.mem.eql(u8, self.base.peek().lexeme, "mut")) blk: {
            _ = self.base.advance(); // consume "mut"
            break :blk true;
        } else false;

        const name_token = try self.base.consumeIdentifierOrKeyword("Expected parameter name");
        _ = try self.base.consume(.Colon, "Expected ':' after parameter name");

        // Use integrated type parser
        self.syncSubParsers();
        const param_type = try self.type_parser.parseTypeWithContext(.Parameter);
        self.updateFromSubParser(self.type_parser.base.current);

        // Parse optional default value
        var default_value: ?*ast.Expressions.ExprNode = null;
        if (self.base.match(.Equal)) {
            // Parse default value expression
            self.syncSubParsers();
            const expr = try self.expr_parser.parseExpression();
            self.updateFromSubParser(self.expr_parser.base.current);

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
    pub fn tryParseLockAnnotation(self: *DeclarationParser) !?ast.Statements.VariableDeclNode {
        if (self.base.check(.At)) {
            const saved_pos = self.base.current;
            _ = self.base.advance(); // consume @
            if (self.base.check(.Identifier) and std.mem.eql(u8, self.base.peek().lexeme, "lock")) {
                _ = self.base.advance(); // consume "lock"

                // Check if this is followed by a variable declaration
                if (self.isMemoryRegionKeyword() or self.base.check(.Let) or self.base.check(.Var)) {
                    return try self.parseVariableDeclWithLock(true);
                }
            }

            // Not @lock or not followed by variable declaration, restore position
            self.base.current = saved_pos;
        }
        return null;
    }

    /// Parse variable declaration with lock annotation flag - ENHANCED FOR MEMORY REGIONS
    fn parseVariableDeclWithLock(self: *DeclarationParser, is_locked: bool) !ast.Statements.VariableDeclNode {
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
            self.syncSubParsers();
            var_type = try self.type_parser.parseTypeWithContext(.Variable);
            self.updateFromSubParser(self.type_parser.base.current);
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
            // Use integrated expression parser
            self.syncSubParsers();
            const expr = try self.expr_parser.parseExpression();
            self.updateFromSubParser(self.expr_parser.base.current);
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

    /// Parse error declaration with optional parameter list (error ErrorName(param: type);)
    fn parseErrorDecl(self: *DeclarationParser) !ast.Statements.ErrorDeclNode {
        const name_token = try self.base.consume(.Identifier, "Expected error name");

        // Parse optional parameter list
        var parameters: ?[]ast.ParameterNode = null;
        if (self.base.match(.LeftParen)) {
            var params = std.ArrayList(ast.ParameterNode).init(self.base.arena.allocator());
            defer params.deinit();

            if (!self.base.check(.RightParen)) {
                repeat: while (true) {
                    const param = try self.parseParameter();
                    try params.append(param);

                    if (!self.base.match(.Comma)) break :repeat;
                }
            }

            _ = try self.base.consume(.RightParen, "Expected ')' after error parameters");
            parameters = try params.toOwnedSlice();
        }

        _ = try self.base.consume(.Semicolon, "Expected ';' after error declaration");

        return ast.Statements.ErrorDeclNode{
            .name = name_token.lexeme,
            .parameters = parameters,
            .span = self.base.spanFromToken(name_token),
        };
    }

    pub fn parseErrorDeclTopLevel(self: *DeclarationParser) !ast.Statements.ErrorDeclNode {
        return self.parseErrorDecl();
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
            .statements = try statements.toOwnedSlice(),
            .span = self.base.spanFromToken(end_token),
        };
    }

    /// Check if current token is a memory region keyword
    fn isMemoryRegionKeyword(self: *DeclarationParser) bool {
        return self.base.check(.Const) or self.base.check(.Immutable) or
            self.base.check(.Storage) or self.base.check(.Memory) or self.base.check(.Tstore);
    }
};
