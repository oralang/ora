const std = @import("std");
const lexer = @import("lexer.zig");
const ast = @import("ast.zig");

const Token = lexer.Token;
const TokenType = lexer.TokenType;
const AstNode = ast.AstNode;
const Allocator = std.mem.Allocator;

/// Parser errors with detailed diagnostics
pub const ParserError = error{
    UnexpectedToken,
    ExpectedToken,
    ExpectedIdentifier,
    ExpectedType,
    ExpectedExpression,
    UnexpectedEof,
    OutOfMemory,
    InvalidMemoryRegion,
    InvalidReturnType,
};

/// Parser for Ora
pub const Parser = struct {
    tokens: []const Token,
    current: usize,
    allocator: Allocator,

    pub fn init(allocator: Allocator, tokens: []const Token) Parser {
        return Parser{
            .tokens = tokens,
            .current = 0,
            .allocator = allocator,
        };
    }

    /// Parse tokens into a list of top-level AST nodes
    pub fn parse(self: *Parser) ParserError![]AstNode {
        var nodes = std.ArrayList(AstNode).init(self.allocator);
        defer nodes.deinit();

        while (!self.isAtEnd()) {
            if (self.check(.Eof)) break;

            const node = try self.parseTopLevel();
            try nodes.append(node);
        }

        return nodes.toOwnedSlice();
    }

    /// Parse a top-level declaration
    fn parseTopLevel(self: *Parser) ParserError!AstNode {
        // Handle imports
        if (self.match(.At)) {
            return self.parseImport();
        }

        // Handle contracts
        if (self.match(.Contract)) {
            return self.parseContract();
        }

        // Handle standalone functions (for modules)
        if (self.check(.Pub) or self.check(.Fn)) {
            return AstNode{ .Function = try self.parseFunction() };
        }

        // Handle variable declarations
        if (self.isMemoryRegionKeyword() or self.check(.Let) or self.check(.Var)) {
            return AstNode{ .VariableDecl = try self.parseVariableDecl() };
        }

        // Handle struct declarations
        if (self.match(.Struct)) {
            return self.parseStruct();
        }

        // Handle enum declarations
        if (self.match(.Enum)) {
            return self.parseEnum();
        }

        // Handle log declarations
        if (self.match(.Log)) {
            return self.parseLogDecl();
        }

        return self.errorAtCurrent("Expected top-level declaration");
    }

    /// Parse import statement
    fn parseImport(self: *Parser) ParserError!AstNode {
        _ = try self.consume(.Import, "Expected 'import'");
        _ = try self.consume(.LeftParen, "Expected '(' after 'import'");

        const path_token = try self.consume(.StringLiteral, "Expected string path");
        _ = try self.consume(.RightParen, "Expected ')' after import path");

        return AstNode{
            .Import = ast.ImportNode{
                .name = "", // Will be set by variable assignment later
                .path = path_token.lexeme,
                .span = ast.SourceSpan{
                    .line = path_token.line,
                    .column = path_token.column,
                    .length = @intCast(path_token.lexeme.len),
                },
            },
        };
    }

    /// Parse contract declaration
    fn parseContract(self: *Parser) ParserError!AstNode {
        const name_token = try self.consume(.Identifier, "Expected contract name");
        _ = try self.consume(.LeftBrace, "Expected '{' after contract name");

        var body = std.ArrayList(AstNode).init(self.allocator);
        defer body.deinit();
        errdefer {
            // Clean up any partially parsed members on error
            for (body.items) |*member| {
                ast.deinitAstNode(self.allocator, member);
            }
        }

        while (!self.check(.RightBrace) and !self.isAtEnd()) {
            const member = try self.parseContractMember();
            try body.append(member);
        }

        _ = try self.consume(.RightBrace, "Expected '}' after contract body");

        return AstNode{ .Contract = ast.ContractNode{
            .name = name_token.lexeme,
            .body = try body.toOwnedSlice(),
            .span = ast.SourceSpan{
                .line = name_token.line,
                .column = name_token.column,
                .length = @intCast(name_token.lexeme.len),
            },
        } };
    }

    /// Parse contract member (function, variable, etc.)
    fn parseContractMember(self: *Parser) ParserError!AstNode {
        // Check for @lock annotation before variable declarations
        if (try self.tryParseLockAnnotation()) |var_decl| {
            return AstNode{ .VariableDecl = var_decl };
        }

        // Skip other annotations for now (consume but ignore)
        while (self.match(.At)) {
            // Consume annotation name
            _ = try self.consume(.Identifier, "Expected annotation name after '@'");

            // If annotation has parameters, consume them
            if (self.match(.LeftParen)) {
                var paren_depth: u32 = 1;
                while (paren_depth > 0 and !self.isAtEnd()) {
                    if (self.match(.LeftParen)) {
                        paren_depth += 1;
                    } else if (self.match(.RightParen)) {
                        paren_depth -= 1;
                    } else {
                        _ = self.advance(); // Skip any token inside parentheses
                    }
                }
            }
        }

        // Functions
        if (self.check(.Pub) or self.check(.Fn)) {
            return AstNode{ .Function = try self.parseFunction() };
        }

        // Variable declarations
        if (self.isMemoryRegionKeyword() or self.check(.Let) or self.check(.Var)) {
            return AstNode{ .VariableDecl = try self.parseVariableDecl() };
        }

        // Log declarations
        if (self.match(.Log)) {
            return self.parseLogDecl();
        }

        // Struct declarations
        if (self.match(.Struct)) {
            return self.parseStruct();
        }

        // Enum declarations
        if (self.match(.Enum)) {
            return self.parseEnum();
        }

        return self.errorAtCurrent("Expected contract member");
    }

    /// Parse function declaration
    fn parseFunction(self: *Parser) ParserError!ast.FunctionNode {
        const is_pub = self.match(.Pub);
        _ = try self.consume(.Fn, "Expected 'fn'");

        // Allow both regular identifiers and 'init' keyword as function names
        const name_token = if (self.check(.Identifier))
            self.advance()
        else if (self.check(.Init))
            self.advance()
        else
            return self.errorAtCurrent("Expected function name");
        _ = try self.consume(.LeftParen, "Expected '(' after function name");

        // Parse parameters
        var params = std.ArrayList(ast.ParamNode).init(self.allocator);
        defer params.deinit();
        errdefer {
            // Clean up parameters on error
            for (params.items) |*param| {
                ast.deinitTypeRef(self.allocator, &param.typ);
            }
        }

        if (!self.check(.RightParen)) {
            repeat: while (true) {
                const param = try self.parseParameter();
                try params.append(param);

                if (!self.match(.Comma)) break :repeat;
            }
        }

        _ = try self.consume(.RightParen, "Expected ')' after parameters");

        // Parse return type (optional)
        var return_type: ?ast.TypeRef = null;
        if (self.match(.Arrow)) {
            return_type = try self.parseType();
        }

        // Parse preconditions (requires clauses)
        var requires_clauses = std.ArrayList(ast.ExprNode).init(self.allocator);
        defer requires_clauses.deinit();
        errdefer {
            // Clean up requires clauses on error
            for (requires_clauses.items) |*clause| {
                ast.deinitExprNode(self.allocator, clause);
            }
        }

        while (self.match(.Requires)) {
            _ = try self.consume(.LeftParen, "Expected '(' after 'requires'");
            const condition = try self.parseExpression();
            try requires_clauses.append(condition);
            _ = try self.consume(.RightParen, "Expected ')' after requires condition");
            _ = self.match(.Semicolon); // Optional semicolon
        }

        // Parse postconditions (ensures clauses)
        var ensures_clauses = std.ArrayList(ast.ExprNode).init(self.allocator);
        defer ensures_clauses.deinit();
        errdefer {
            // Clean up ensures clauses on error
            for (ensures_clauses.items) |*clause| {
                ast.deinitExprNode(self.allocator, clause);
            }
        }

        while (self.match(.Ensures)) {
            _ = try self.consume(.LeftParen, "Expected '(' after 'ensures'");
            const condition = try self.parseExpression();
            try ensures_clauses.append(condition);
            _ = try self.consume(.RightParen, "Expected ')' after ensures condition");
            _ = self.match(.Semicolon); // Optional semicolon
        }

        // Parse function body
        const body = try self.parseBlock();

        return ast.FunctionNode{
            .pub_ = is_pub,
            .name = name_token.lexeme,
            .parameters = try params.toOwnedSlice(),
            .return_type = return_type,
            .requires_clauses = try requires_clauses.toOwnedSlice(),
            .ensures_clauses = try ensures_clauses.toOwnedSlice(),
            .body = body,
            .span = ast.SourceSpan{
                .line = name_token.line,
                .column = name_token.column,
                .length = @intCast(name_token.lexeme.len),
            },
        };
    }

    /// Parse function parameter
    fn parseParameter(self: *Parser) ParserError!ast.ParamNode {
        const name_token = try self.consumeIdentifierOrKeyword("Expected parameter name");
        _ = try self.consume(.Colon, "Expected ':' after parameter name");
        const param_type = try self.parseType();

        return ast.ParamNode{
            .name = name_token.lexeme,
            .typ = param_type,
            .span = ast.SourceSpan{
                .line = name_token.line,
                .column = name_token.column,
                .length = @intCast(name_token.lexeme.len),
            },
        };
    }

    /// Parse type reference
    fn parseType(self: *Parser) ParserError!ast.TypeRef {
        // Handle map type directly
        if (self.match(.Map)) {
            _ = try self.consume(.LeftBracket, "Expected '[' after 'map'");
            const key_type = try self.allocator.create(ast.TypeRef);
            errdefer self.allocator.destroy(key_type);
            key_type.* = try self.parseType();
            _ = try self.consume(.Comma, "Expected ',' after map key type");
            const value_type = try self.allocator.create(ast.TypeRef);
            errdefer {
                ast.deinitTypeRef(self.allocator, key_type);
                self.allocator.destroy(key_type);
                self.allocator.destroy(value_type);
            }
            value_type.* = try self.parseType();
            _ = try self.consume(.RightBracket, "Expected ']' after map value type");

            return ast.TypeRef{ .Mapping = ast.MappingType{
                .key = key_type,
                .value = value_type,
            } };
        }

        // Handle bytes type directly
        if (self.match(.Bytes)) {
            return .Bytes;
        }

        // Handle array types [T; N] and [T]
        if (self.match(.LeftBracket)) {
            const elem_type = try self.allocator.create(ast.TypeRef);
            errdefer self.allocator.destroy(elem_type);
            elem_type.* = try self.parseType();

            if (self.match(.Semicolon)) {
                // Fixed array: [T; N]
                _ = try self.consume(.IntegerLiteral, "Expected array size after ';'");
                _ = try self.consume(.RightBracket, "Expected ']' after array size");
                return ast.TypeRef{ .Slice = elem_type }; // For now, treat as slice
            } else {
                // Dynamic array: [T]
                _ = try self.consume(.RightBracket, "Expected ']' after array element type");
                return ast.TypeRef{ .Slice = elem_type };
            }
        }

        if (self.match(.Identifier)) {
            const type_name = self.previous().lexeme;

            // Check for built-in types
            if (std.mem.eql(u8, type_name, "bool")) return .Bool;
            if (std.mem.eql(u8, type_name, "address")) return .Address;
            if (std.mem.eql(u8, type_name, "u8")) return .U8;
            if (std.mem.eql(u8, type_name, "u16")) return .U16;
            if (std.mem.eql(u8, type_name, "u32")) return .U32;
            if (std.mem.eql(u8, type_name, "u64")) return .U64;
            if (std.mem.eql(u8, type_name, "u128")) return .U128;
            if (std.mem.eql(u8, type_name, "u256")) return .U256;
            if (std.mem.eql(u8, type_name, "i8")) return .I8;
            if (std.mem.eql(u8, type_name, "i16")) return .I16;
            if (std.mem.eql(u8, type_name, "i32")) return .I32;
            if (std.mem.eql(u8, type_name, "i64")) return .I64;
            if (std.mem.eql(u8, type_name, "i128")) return .I128;
            if (std.mem.eql(u8, type_name, "i256")) return .I256;
            if (std.mem.eql(u8, type_name, "string")) return .String;

            // Check for complex types
            if (std.mem.eql(u8, type_name, "slice")) {
                _ = try self.consume(.LeftBracket, "Expected '[' after 'slice'");
                const elem_type = try self.allocator.create(ast.TypeRef);
                errdefer self.allocator.destroy(elem_type);
                elem_type.* = try self.parseType();
                _ = try self.consume(.RightBracket, "Expected ']' after slice element type");
                return ast.TypeRef{ .Slice = elem_type };
            }

            if (std.mem.eql(u8, type_name, "doublemap")) {
                _ = try self.consume(.LeftBracket, "Expected '[' after 'doublemap'");
                const key1_type = try self.allocator.create(ast.TypeRef);
                errdefer self.allocator.destroy(key1_type);
                key1_type.* = try self.parseType();
                _ = try self.consume(.Comma, "Expected ',' after first key type");
                const key2_type = try self.allocator.create(ast.TypeRef);
                errdefer {
                    ast.deinitTypeRef(self.allocator, key1_type);
                    self.allocator.destroy(key1_type);
                    self.allocator.destroy(key2_type);
                }
                key2_type.* = try self.parseType();
                _ = try self.consume(.Comma, "Expected ',' after second key type");
                const value_type = try self.allocator.create(ast.TypeRef);
                errdefer {
                    ast.deinitTypeRef(self.allocator, key1_type);
                    ast.deinitTypeRef(self.allocator, key2_type);
                    self.allocator.destroy(key1_type);
                    self.allocator.destroy(key2_type);
                    self.allocator.destroy(value_type);
                }
                value_type.* = try self.parseType();
                _ = try self.consume(.RightBracket, "Expected ']' after doublemap value type");

                return ast.TypeRef{ .DoubleMap = ast.DoubleMapType{
                    .key1 = key1_type,
                    .key2 = key2_type,
                    .value = value_type,
                } };
            }

            // Result[T, E] type
            if (std.mem.eql(u8, type_name, "Result")) {
                _ = try self.consume(.LeftBracket, "Expected '[' after 'Result'");
                const ok_type = try self.allocator.create(ast.TypeRef);
                errdefer self.allocator.destroy(ok_type);
                ok_type.* = try self.parseType();
                _ = try self.consume(.Comma, "Expected ',' after Result ok type");
                const error_type = try self.allocator.create(ast.TypeRef);
                errdefer {
                    ast.deinitTypeRef(self.allocator, ok_type);
                    self.allocator.destroy(ok_type);
                    self.allocator.destroy(error_type);
                }
                error_type.* = try self.parseType();
                _ = try self.consume(.RightBracket, "Expected ']' after Result error type");

                return ast.TypeRef{ .Result = ast.ResultType{
                    .ok_type = ok_type,
                    .error_type = error_type,
                } };
            }

            // Custom type (struct/enum)
            return ast.TypeRef{ .Identifier = type_name };
        }

        // Error union type (!T)
        if (self.match(.Bang)) {
            const success_type = try self.allocator.create(ast.TypeRef);
            errdefer self.allocator.destroy(success_type);
            success_type.* = try self.parseType();

            return ast.TypeRef{ .ErrorUnion = ast.ErrorUnionType{
                .success_type = success_type,
            } };
        }

        return self.errorAtCurrent("Expected type");
    }

    /// Parse variable declaration
    fn parseVariableDecl(self: *Parser) ParserError!ast.VariableDeclNode {
        return self.parseVariableDeclWithLock(false);
    }

    /// Parse variable declaration with lock annotation flag
    fn parseVariableDeclWithLock(self: *Parser, is_locked: bool) ParserError!ast.VariableDeclNode {
        // Parse memory region
        const region = try self.parseMemoryRegion();

        // Parse mutability - handle const/immutable after storage
        const is_mutable = blk: {
            if (region == .Const or region == .Immutable) {
                break :blk false;
            } else if (region == .Storage or region == .Memory or region == .TStore) {
                // For storage/memory/tstore, check for const/mut/let after region
                if (self.match(.Const)) {
                    // storage const name: type - not mutable
                    break :blk false;
                } else if (self.match(.Var)) {
                    // storage var name: type - mutable
                    break :blk true;
                } else if (self.match(.Let)) {
                    // storage let name: type - not mutable
                    break :blk false;
                } else {
                    // Default to mutable for storage if no modifier specified
                    break :blk true;
                }
            } else {
                // For stack variables (default region), check for var or let
                if (self.match(.Var)) {
                    break :blk true;
                } else if (self.match(.Let)) {
                    break :blk false;
                } else {
                    break :blk false; // Default to immutable
                }
            }
        };

        // Check for tuple unpacking: let (a, b) = expr
        var tuple_names: ?[][]const u8 = null;
        var name_token: lexer.Token = undefined;
        var var_type: ast.TypeRef = undefined;

        if (self.check(.LeftParen)) {
            // Tuple unpacking
            _ = self.advance(); // consume '('

            var names = std.ArrayList([]const u8).init(self.allocator);
            defer names.deinit();

            // Parse tuple variable names
            if (!self.check(.RightParen)) {
                repeat: while (true) {
                    const tuple_name = try self.consume(.Identifier, "Expected variable name in tuple");
                    try names.append(tuple_name.lexeme);

                    if (!self.match(.Comma)) break :repeat;
                }
            }

            _ = try self.consume(.RightParen, "Expected ')' after tuple variables");

            // For tuple unpacking, we don't specify explicit type yet (inferred from RHS)
            _ = try self.consume(.Equal, "Expected '=' after tuple variables");

            tuple_names = try names.toOwnedSlice();
            name_token = self.previous(); // Use ')' token for span
            var_type = ast.TypeRef.Unknown; // Will be inferred
        } else {
            // Regular variable declaration
            name_token = try self.consume(.Identifier, "Expected variable name");
            _ = try self.consume(.Colon, "Expected ':' after variable name");
            var_type = try self.parseType();
        }

        // Parse optional initializer
        var initializer: ?ast.ExprNode = null;
        if (self.match(.Equal)) {
            initializer = try self.parseExpression();
        }

        _ = self.match(.Semicolon); // Optional semicolon

        return ast.VariableDeclNode{
            .name = if (tuple_names) |_| "" else name_token.lexeme, // Empty name for tuple unpacking
            .region = region,
            .mutable = is_mutable,
            .locked = is_locked,
            .typ = var_type,
            .value = initializer,
            .span = ast.SourceSpan{
                .line = name_token.line,
                .column = name_token.column,
                .length = @intCast(name_token.lexeme.len),
            },
            .tuple_names = tuple_names,
        };
    }

    /// Parse memory region keywords
    fn parseMemoryRegion(self: *Parser) ParserError!ast.MemoryRegion {
        // Check for explicit memory regions first
        if (self.match(.Storage)) return .Storage;
        if (self.match(.Memory)) return .Memory;
        if (self.match(.Tstore)) return .TStore;
        if (self.match(.Immutable)) return .Immutable;

        // const can only be a memory region if it's not preceded by storage/memory/tstore
        if (self.match(.Const)) return .Const;

        // Stack region for let/var variables
        if (self.check(.Let) or self.check(.Var)) return .Stack;

        return .Stack; // Default to stack
    }

    /// Check if current token is a memory region keyword
    fn isMemoryRegionKeyword(self: *Parser) bool {
        return self.check(.Const) or self.check(.Immutable) or
            self.check(.Storage) or self.check(.Memory) or self.check(.Tstore);
    }

    /// Try to parse @lock annotation, returns variable declaration if found
    fn tryParseLockAnnotation(self: *Parser) ParserError!?ast.VariableDeclNode {
        if (self.check(.At)) {
            const saved_pos = self.current;
            _ = self.advance(); // consume @
            if (self.check(.Identifier) and std.mem.eql(u8, self.peek().lexeme, "lock")) {
                _ = self.advance(); // consume "lock"

                // Check if this is followed by a variable declaration
                if (self.isMemoryRegionKeyword() or self.check(.Let) or self.check(.Var)) {
                    return try self.parseVariableDeclWithLock(true);
                }
            }

            // Not @lock or not followed by variable declaration, restore position
            self.current = saved_pos;
        }
        return null;
    }

    /// Parse block statement
    fn parseBlock(self: *Parser) ParserError!ast.BlockNode {
        _ = try self.consume(.LeftBrace, "Expected '{'");

        var statements = std.ArrayList(ast.StmtNode).init(self.allocator);
        defer statements.deinit();
        errdefer {
            // Clean up statements on error
            for (statements.items) |*stmt| {
                ast.deinitStmtNode(self.allocator, stmt);
            }
        }

        while (!self.check(.RightBrace) and !self.isAtEnd()) {
            const stmt = try self.parseStatement();
            try statements.append(stmt);
        }

        const end_token = try self.consume(.RightBrace, "Expected '}' after block");

        return ast.BlockNode{
            .statements = try statements.toOwnedSlice(),
            .span = ast.SourceSpan{
                .line = end_token.line,
                .column = end_token.column,
                .length = 1,
            },
        };
    }

    /// Parse statement
    fn parseStatement(self: *Parser) ParserError!ast.StmtNode {
        // Check for @lock annotation before variable declarations
        if (try self.tryParseLockAnnotation()) |var_decl| {
            return ast.StmtNode{ .VariableDecl = var_decl };
        }

        // Skip other annotations for now (consume but ignore)
        while (self.match(.At)) {
            // Consume annotation name
            _ = try self.consume(.Identifier, "Expected annotation name after '@'");

            // If annotation has parameters, consume them
            if (self.match(.LeftParen)) {
                var paren_depth: u32 = 1;
                while (paren_depth > 0 and !self.isAtEnd()) {
                    if (self.match(.LeftParen)) {
                        paren_depth += 1;
                    } else if (self.match(.RightParen)) {
                        paren_depth -= 1;
                    } else {
                        _ = self.advance(); // Skip any token inside parentheses
                    }
                }
            }

            // Consume optional semicolon after annotation
            _ = self.match(.Semicolon);
        }

        // Variable declarations
        if (self.isMemoryRegionKeyword() or self.check(.Let) or self.check(.Var)) {
            return ast.StmtNode{ .VariableDecl = try self.parseVariableDecl() };
        }

        // Return statements
        if (self.match(.Return)) {
            var value: ?ast.ExprNode = null;
            if (!self.check(.Semicolon) and !self.check(.RightBrace)) {
                value = try self.parseExpression();
            }
            _ = self.match(.Semicolon);

            return ast.StmtNode{ .Return = ast.ReturnNode{
                .value = value,
                .span = makeSpan(self.previous()),
            } };
        }

        // Log statements
        if (self.match(.Log)) {
            const event_name_token = try self.consume(.Identifier, "Expected event name");
            _ = try self.consume(.LeftParen, "Expected '(' after event name");

            var args = std.ArrayList(ast.ExprNode).init(self.allocator);
            defer args.deinit();

            if (!self.check(.RightParen)) {
                repeat: while (true) {
                    const arg = try self.parseExpression();
                    try args.append(arg);
                    if (!self.match(.Comma)) break :repeat;
                }
            }

            _ = try self.consume(.RightParen, "Expected ')' after log arguments");
            _ = self.match(.Semicolon);

            return ast.StmtNode{ .Log = ast.LogNode{
                .event_name = event_name_token.lexeme,
                .args = try args.toOwnedSlice(),
                .span = makeSpan(event_name_token),
            } };
        }

        // Requires statements
        if (self.match(.Requires)) {
            _ = try self.consume(.LeftParen, "Expected '(' after 'requires'");
            const condition = try self.parseExpression();
            _ = try self.consume(.RightParen, "Expected ')' after requires condition");
            _ = self.match(.Semicolon);

            return ast.StmtNode{ .Requires = ast.RequiresNode{
                .condition = condition,
                .span = makeSpan(self.previous()),
            } };
        }

        // Ensures statements
        if (self.match(.Ensures)) {
            _ = try self.consume(.LeftParen, "Expected '(' after 'ensures'");
            const condition = try self.parseExpression();
            _ = try self.consume(.RightParen, "Expected ')' after ensures condition");
            _ = self.match(.Semicolon);

            return ast.StmtNode{ .Ensures = ast.EnsuresNode{
                .condition = condition,
                .span = makeSpan(self.previous()),
            } };
        }

        // Error declarations
        if (self.match(.Error)) {
            const error_token = self.previous();
            const name_token = try self.consume(.Identifier, "Expected error name");
            _ = try self.consume(.Semicolon, "Expected ';' after error declaration");

            return ast.StmtNode{ .ErrorDecl = ast.ErrorDeclNode{
                .name = name_token.lexeme,
                .span = makeSpan(error_token),
            } };
        }

        // Try-catch blocks
        if (self.match(.Try)) {
            const try_token = self.previous();
            const try_block = try self.parseBlock();

            var catch_block: ?ast.CatchBlock = null;
            if (self.match(.Catch)) {
                var error_variable: ?[]const u8 = null;

                // Optional catch variable: catch(e) { ... }
                if (self.match(.LeftParen)) {
                    const var_token = try self.consume(.Identifier, "Expected variable name in catch");
                    error_variable = var_token.lexeme;
                    _ = try self.consume(.RightParen, "Expected ')' after catch variable");
                }

                const catch_body = try self.parseBlock();
                catch_block = ast.CatchBlock{
                    .error_variable = error_variable,
                    .block = catch_body,
                    .span = makeSpan(self.previous()),
                };
            }

            return ast.StmtNode{ .TryBlock = ast.TryBlockNode{
                .try_block = try_block,
                .catch_block = catch_block,
                .span = makeSpan(try_token),
            } };
        }

        // If statements
        if (self.match(.If)) {
            _ = try self.consume(.LeftParen, "Expected '(' after 'if'");
            const condition = try self.parseExpression();
            _ = try self.consume(.RightParen, "Expected ')' after if condition");

            const then_branch = try self.parseBlock();

            var else_branch: ?ast.BlockNode = null;
            if (self.match(.Else)) {
                else_branch = try self.parseBlock();
            }

            return ast.StmtNode{ .If = ast.IfNode{
                .condition = condition,
                .then_branch = then_branch,
                .else_branch = else_branch,
                .span = makeSpan(self.previous()),
            } };
        }

        // While statements
        if (self.match(.While)) {
            _ = try self.consume(.LeftParen, "Expected '(' after 'while'");
            const condition = try self.parseExpression();
            _ = try self.consume(.RightParen, "Expected ')' after while condition");

            const body = try self.parseBlock();

            // Parse optional invariants (for now, empty array)
            var invariants = std.ArrayList(ast.ExprNode).init(self.allocator);
            defer invariants.deinit();

            return ast.StmtNode{ .While = ast.WhileNode{
                .condition = condition,
                .body = body,
                .invariants = try invariants.toOwnedSlice(),
                .span = makeSpan(self.previous()),
            } };
        }

        // Break statements
        if (self.match(.Break)) {
            _ = self.match(.Semicolon);
            return ast.StmtNode{ .Break = makeSpan(self.previous()) };
        }

        // Continue statements
        if (self.match(.Continue)) {
            _ = self.match(.Semicolon);
            return ast.StmtNode{ .Continue = makeSpan(self.previous()) };
        }

        // Expression statements
        const expr = try self.parseExpression();
        _ = self.match(.Semicolon);
        return ast.StmtNode{ .Expr = expr };
    }

    /// Parse expression with precedence climbing
    fn parseExpression(self: *Parser) ParserError!ast.ExprNode {
        return self.parseAssignment();
    }

    /// Parse assignment expressions
    fn parseAssignment(self: *Parser) ParserError!ast.ExprNode {
        const expr = try self.parseLogicalOr();

        // Check for shift syntax: mapping from source -> dest : amount
        if (self.match(.From)) {
            const from_token = self.previous();
            const source = try self.parseLogicalOr();
            _ = try self.consume(.Arrow, "Expected '->' in shift expression");
            const dest = try self.parseLogicalOr();
            _ = try self.consume(.Colon, "Expected ':' in shift expression");
            const amount = try self.parseAssignment();

            const mapping_ptr = try self.allocator.create(ast.ExprNode);
            mapping_ptr.* = expr;
            const source_ptr = try self.allocator.create(ast.ExprNode);
            source_ptr.* = source;
            const dest_ptr = try self.allocator.create(ast.ExprNode);
            dest_ptr.* = dest;
            const amount_ptr = try self.allocator.create(ast.ExprNode);
            amount_ptr.* = amount;

            return ast.ExprNode{ .Shift = ast.ShiftExpr{
                .mapping = mapping_ptr,
                .source = source_ptr,
                .dest = dest_ptr,
                .amount = amount_ptr,
                .span = makeSpan(from_token),
            } };
        }

        if (self.match(.Equal)) {
            const value = try self.parseAssignment();
            const expr_ptr = try self.allocator.create(ast.ExprNode);
            expr_ptr.* = expr;
            const value_ptr = try self.allocator.create(ast.ExprNode);
            value_ptr.* = value;

            return ast.ExprNode{ .Assignment = ast.AssignmentExpr{
                .target = expr_ptr,
                .value = value_ptr,
                .span = makeSpan(self.previous()),
            } };
        }

        // Compound assignments
        if (self.match(.PlusEqual) or self.match(.MinusEqual) or self.match(.StarEqual)) {
            const op_token = self.previous();
            const value = try self.parseAssignment();
            const expr_ptr = try self.allocator.create(ast.ExprNode);
            expr_ptr.* = expr;
            const value_ptr = try self.allocator.create(ast.ExprNode);
            value_ptr.* = value;

            const compound_op: ast.CompoundAssignmentOp = switch (op_token.type) {
                .PlusEqual => .PlusEqual,
                .MinusEqual => .MinusEqual,
                .StarEqual => .StarEqual,
                else => unreachable,
            };

            return ast.ExprNode{ .CompoundAssignment = ast.CompoundAssignmentExpr{
                .target = expr_ptr,
                .operator = compound_op,
                .value = value_ptr,
                .span = makeSpan(op_token),
            } };
        }

        return expr;
    }

    /// Parse logical OR expressions
    fn parseLogicalOr(self: *Parser) ParserError!ast.ExprNode {
        var expr = try self.parseLogicalAnd();

        while (self.match(.Pipe)) {
            const op_token = self.previous();
            const right = try self.parseLogicalAnd();

            const left_ptr = try self.allocator.create(ast.ExprNode);
            left_ptr.* = expr;
            const right_ptr = try self.allocator.create(ast.ExprNode);
            right_ptr.* = right;

            expr = ast.ExprNode{
                .Binary = ast.BinaryExpr{
                    .lhs = left_ptr,
                    .operator = .Or, // Logical OR
                    .rhs = right_ptr,
                    .span = makeSpan(op_token),
                },
            };
        }

        return expr;
    }

    /// Parse logical AND expressions
    fn parseLogicalAnd(self: *Parser) ParserError!ast.ExprNode {
        var expr = try self.parseEquality();

        while (self.match(.Ampersand)) {
            const op_token = self.previous();
            const right = try self.parseEquality();

            const left_ptr = try self.allocator.create(ast.ExprNode);
            left_ptr.* = expr;
            const right_ptr = try self.allocator.create(ast.ExprNode);
            right_ptr.* = right;

            expr = ast.ExprNode{
                .Binary = ast.BinaryExpr{
                    .lhs = left_ptr,
                    .operator = .And, // Logical AND
                    .rhs = right_ptr,
                    .span = makeSpan(op_token),
                },
            };
        }

        return expr;
    }

    /// Parse equality expressions
    fn parseEquality(self: *Parser) ParserError!ast.ExprNode {
        var expr = try self.parseComparison();

        while (self.match(.EqualEqual) or self.match(.BangEqual)) {
            const op_token = self.previous();
            const right = try self.parseComparison();

            const operator: ast.BinaryOp = switch (op_token.type) {
                .EqualEqual => .EqualEqual,
                .BangEqual => .BangEqual,
                else => unreachable,
            };

            const left_ptr = try self.allocator.create(ast.ExprNode);
            left_ptr.* = expr;
            const right_ptr = try self.allocator.create(ast.ExprNode);
            right_ptr.* = right;

            expr = ast.ExprNode{ .Binary = ast.BinaryExpr{
                .lhs = left_ptr,
                .operator = operator,
                .rhs = right_ptr,
                .span = makeSpan(op_token),
            } };
        }

        return expr;
    }

    /// Parse comparison expressions
    fn parseComparison(self: *Parser) ParserError!ast.ExprNode {
        var expr = try self.parseTerm();

        while (self.match(.Greater) or self.match(.GreaterEqual) or
            self.match(.Less) or self.match(.LessEqual))
        {
            const op_token = self.previous();
            const right = try self.parseTerm();

            const operator: ast.BinaryOp = switch (op_token.type) {
                .Greater => .Greater,
                .GreaterEqual => .GreaterEqual,
                .Less => .Less,
                .LessEqual => .LessEqual,
                else => unreachable,
            };

            const left_ptr = try self.allocator.create(ast.ExprNode);
            left_ptr.* = expr;
            const right_ptr = try self.allocator.create(ast.ExprNode);
            right_ptr.* = right;

            expr = ast.ExprNode{ .Binary = ast.BinaryExpr{
                .lhs = left_ptr,
                .operator = operator,
                .rhs = right_ptr,
                .span = makeSpan(op_token),
            } };
        }

        return expr;
    }

    /// Parse term expressions (+ -)
    fn parseTerm(self: *Parser) ParserError!ast.ExprNode {
        var expr = try self.parseFactor();

        while (self.match(.Plus) or self.match(.Minus)) {
            const op_token = self.previous();
            const right = try self.parseFactor();

            const operator: ast.BinaryOp = switch (op_token.type) {
                .Plus => .Plus,
                .Minus => .Minus,
                else => unreachable,
            };

            const left_ptr = try self.allocator.create(ast.ExprNode);
            left_ptr.* = expr;
            const right_ptr = try self.allocator.create(ast.ExprNode);
            right_ptr.* = right;

            expr = ast.ExprNode{ .Binary = ast.BinaryExpr{
                .lhs = left_ptr,
                .operator = operator,
                .rhs = right_ptr,
                .span = makeSpan(op_token),
            } };
        }

        return expr;
    }

    /// Parse factor expressions (* / %)
    fn parseFactor(self: *Parser) ParserError!ast.ExprNode {
        var expr = try self.parseUnary();

        while (self.match(.Star) or self.match(.Slash) or self.match(.Percent)) {
            const op_token = self.previous();
            const right = try self.parseUnary();

            const operator: ast.BinaryOp = switch (op_token.type) {
                .Star => .Star,
                .Slash => .Slash,
                .Percent => .Percent,
                else => unreachable,
            };

            const left_ptr = try self.allocator.create(ast.ExprNode);
            left_ptr.* = expr;
            const right_ptr = try self.allocator.create(ast.ExprNode);
            right_ptr.* = right;

            expr = ast.ExprNode{ .Binary = ast.BinaryExpr{
                .lhs = left_ptr,
                .operator = operator,
                .rhs = right_ptr,
                .span = makeSpan(op_token),
            } };
        }

        return expr;
    }

    /// Parse unary expressions
    fn parseUnary(self: *Parser) ParserError!ast.ExprNode {
        if (self.match(.Bang) or self.match(.Minus)) {
            const op_token = self.previous();
            const right = try self.parseUnary();

            const operator: ast.UnaryOp = switch (op_token.type) {
                .Bang => .Bang,
                .Minus => .Minus,
                else => unreachable,
            };

            const right_ptr = try self.allocator.create(ast.ExprNode);
            right_ptr.* = right;

            return ast.ExprNode{ .Unary = ast.UnaryExpr{
                .operator = operator,
                .operand = right_ptr,
                .span = makeSpan(op_token),
            } };
        }

        return self.parseCall();
    }

    /// Parse function calls and member access
    fn parseCall(self: *Parser) ParserError!ast.ExprNode {
        var expr = try self.parsePrimary();

        while (true) {
            if (self.match(.LeftParen)) {
                expr = try self.finishCall(expr);
            } else if (self.match(.Dot)) {
                const name_token = try self.consume(.Identifier, "Expected property name after '.'");
                const expr_ptr = try self.allocator.create(ast.ExprNode);
                expr_ptr.* = expr;

                // Check if this might be an enum literal (EnumType.VariantName)
                // We parse it as an enum literal if the left side is a simple identifier
                if (expr == .Identifier) {
                    const enum_name = expr.Identifier.name;
                    expr = ast.ExprNode{ .EnumLiteral = ast.EnumLiteralExpr{
                        .enum_name = enum_name,
                        .variant_name = name_token.lexeme,
                        .span = makeSpan(name_token),
                    } };
                } else {
                    // Otherwise, treat it as field access
                    expr = ast.ExprNode{ .FieldAccess = ast.FieldAccessExpr{
                        .target = expr_ptr,
                        .field = name_token.lexeme,
                        .span = makeSpan(name_token),
                    } };
                }
            } else if (self.match(.LeftBracket)) {
                const index = try self.parseExpression();

                // Check for double mapping access: target[key1, key2]
                if (self.match(.Comma)) {
                    const second_index = try self.parseExpression();
                    _ = try self.consume(.RightBracket, "Expected ']' after double mapping index");

                    const expr_ptr = try self.allocator.create(ast.ExprNode);
                    expr_ptr.* = expr;
                    const index_ptr = try self.allocator.create(ast.ExprNode);
                    index_ptr.* = index;
                    const second_index_ptr = try self.allocator.create(ast.ExprNode);
                    second_index_ptr.* = second_index;

                    // Create a nested index expression for double mapping: target[key1][key2]
                    const first_index = ast.ExprNode{ .Index = ast.IndexExpr{
                        .target = expr_ptr,
                        .index = index_ptr,
                        .span = makeSpan(self.previous()),
                    } };

                    const first_index_ptr = try self.allocator.create(ast.ExprNode);
                    first_index_ptr.* = first_index;

                    expr = ast.ExprNode{ .Index = ast.IndexExpr{
                        .target = first_index_ptr,
                        .index = second_index_ptr,
                        .span = makeSpan(self.previous()),
                    } };
                } else {
                    _ = try self.consume(.RightBracket, "Expected ']' after array index");

                    const expr_ptr = try self.allocator.create(ast.ExprNode);
                    expr_ptr.* = expr;
                    const index_ptr = try self.allocator.create(ast.ExprNode);
                    index_ptr.* = index;

                    expr = ast.ExprNode{ .Index = ast.IndexExpr{
                        .target = expr_ptr,
                        .index = index_ptr,
                        .span = makeSpan(self.previous()),
                    } };
                }
            } else {
                break;
            }
        }

        return expr;
    }

    /// Finish parsing a function call
    fn finishCall(self: *Parser, callee: ast.ExprNode) ParserError!ast.ExprNode {
        var arguments = std.ArrayList(ast.ExprNode).init(self.allocator);
        defer arguments.deinit();

        if (!self.check(.RightParen)) {
            repeat: while (true) {
                const arg = try self.parseExpression();
                try arguments.append(arg);
                if (!self.match(.Comma)) break :repeat;
            }
        }

        const paren_token = try self.consume(.RightParen, "Expected ')' after arguments");

        const callee_ptr = try self.allocator.create(ast.ExprNode);
        callee_ptr.* = callee;

        return ast.ExprNode{ .Call = ast.CallExpr{
            .callee = callee_ptr,
            .arguments = try arguments.toOwnedSlice(),
            .span = makeSpan(paren_token),
        } };
    }

    /// Parse primary expressions (literals, identifiers, parentheses)
    fn parsePrimary(self: *Parser) ParserError!ast.ExprNode {
        // Boolean literals
        if (self.match(.True)) {
            return ast.ExprNode{ .Literal = .{ .Bool = ast.BoolLiteral{
                .value = true,
                .span = makeSpan(self.previous()),
            } } };
        }

        if (self.match(.False)) {
            return ast.ExprNode{ .Literal = .{ .Bool = ast.BoolLiteral{
                .value = false,
                .span = makeSpan(self.previous()),
            } } };
        }

        // Number literals
        if (self.match(.IntegerLiteral)) {
            const token = self.previous();
            return ast.ExprNode{ .Literal = .{ .Integer = ast.IntegerLiteral{
                .value = token.lexeme,
                .span = makeSpan(token),
            } } };
        }

        // String literals
        if (self.match(.StringLiteral)) {
            const token = self.previous();
            return ast.ExprNode{ .Literal = .{ .String = ast.StringLiteral{
                .value = token.lexeme,
                .span = makeSpan(token),
            } } };
        }

        // Address literals
        if (self.match(.AddressLiteral)) {
            const token = self.previous();
            return ast.ExprNode{ .Literal = .{ .Address = ast.AddressLiteral{
                .value = token.lexeme,
                .span = makeSpan(token),
            } } };
        }

        // Hex literals
        if (self.match(.HexLiteral)) {
            const token = self.previous();
            return ast.ExprNode{ .Literal = .{ .Hex = ast.HexLiteral{
                .value = token.lexeme,
                .span = makeSpan(token),
            } } };
        }

        // Identifiers (including keywords that can be used as identifiers)
        if (self.match(.Identifier) or self.matchKeywordAsIdentifier()) {
            const token = self.previous();

            // Check if this is struct instantiation (identifier followed by {)
            if (self.check(.LeftBrace)) {
                return try self.parseStructInstantiation(token);
            }

            return ast.ExprNode{ .Identifier = ast.IdentifierExpr{
                .name = token.lexeme,
                .span = makeSpan(token),
            } };
        }

        // Try expressions
        if (self.match(.Try)) {
            const try_token = self.previous();
            const expr = try self.parseUnary();

            const expr_ptr = try self.allocator.create(ast.ExprNode);
            expr_ptr.* = expr;

            return ast.ExprNode{ .Try = ast.TryExpr{
                .expr = expr_ptr,
                .span = makeSpan(try_token),
            } };
        }

        // Old expressions (old(expr) for postconditions)
        if (self.match(.Old)) {
            const old_token = self.previous();
            _ = try self.consume(.LeftParen, "Expected '(' after 'old'");
            const expr = try self.parseExpression();
            _ = try self.consume(.RightParen, "Expected ')' after old expression");

            const expr_ptr = try self.allocator.create(ast.ExprNode);
            expr_ptr.* = expr;

            return ast.ExprNode{ .Old = ast.OldExpr{
                .expr = expr_ptr,
                .span = makeSpan(old_token),
            } };
        }

        // Error expressions (error.SomeError)
        if (self.match(.Error)) {
            const error_token = self.previous();
            _ = try self.consume(.Dot, "Expected '.' after 'error'");
            const name_token = try self.consume(.Identifier, "Expected error name after 'error.'");

            return ast.ExprNode{ .ErrorReturn = ast.ErrorReturnExpr{
                .error_name = name_token.lexeme,
                .span = makeSpan(error_token),
            } };
        }

        // Builtin functions starting with @
        if (self.match(.At)) {
            const at_token = self.previous();
            const name_token = try self.consume(.Identifier, "Expected builtin function name after '@'");

            // Check if it's a valid builtin function
            const builtin_name = name_token.lexeme;
            if (std.mem.eql(u8, builtin_name, "divTrunc") or
                std.mem.eql(u8, builtin_name, "divFloor") or
                std.mem.eql(u8, builtin_name, "divCeil") or
                std.mem.eql(u8, builtin_name, "divExact") or
                std.mem.eql(u8, builtin_name, "divmod"))
            {
                _ = try self.consume(.LeftParen, "Expected '(' after builtin function name");

                var args = std.ArrayList(ast.ExprNode).init(self.allocator);
                defer args.deinit();

                if (!self.check(.RightParen)) {
                    const first_arg = try self.parseExpression();
                    try args.append(first_arg);

                    while (self.match(.Comma)) {
                        const arg = try self.parseExpression();
                        try args.append(arg);
                    }
                }

                _ = try self.consume(.RightParen, "Expected ')' after arguments");

                // Create the builtin function call
                const full_name = try std.fmt.allocPrint(self.allocator, "@{s}", .{builtin_name});

                // Create identifier for the function name
                const name_expr = try self.allocator.create(ast.ExprNode);
                name_expr.* = ast.ExprNode{ .Identifier = ast.IdentifierExpr{
                    .name = full_name,
                    .span = makeSpan(at_token),
                } };

                return ast.ExprNode{ .Call = ast.CallExpr{
                    .callee = name_expr,
                    .arguments = try args.toOwnedSlice(),
                    .span = makeSpan(at_token),
                } };
            } else {
                return self.errorAtCurrent("Unknown builtin function");
            }
        }

        // Parenthesized expressions or tuples
        if (self.match(.LeftParen)) {
            const paren_token = self.previous();

            // Check for empty tuple
            if (self.check(.RightParen)) {
                _ = self.advance();
                var empty_elements = std.ArrayList(ast.ExprNode).init(self.allocator);
                defer empty_elements.deinit();

                return ast.ExprNode{ .Tuple = ast.TupleExpr{
                    .elements = try empty_elements.toOwnedSlice(),
                    .span = makeSpan(paren_token),
                } };
            }

            const first_expr = try self.parseExpression();

            // Check if it's a tuple (has comma)
            if (self.match(.Comma)) {
                var elements = std.ArrayList(ast.ExprNode).init(self.allocator);
                defer elements.deinit();

                try elements.append(first_expr);

                // Handle trailing comma case: (a,)
                if (!self.check(.RightParen)) {
                    repeat: while (true) {
                        const element = try self.parseExpression();
                        try elements.append(element);

                        if (!self.match(.Comma)) break :repeat;
                        if (self.check(.RightParen)) break :repeat; // Trailing comma
                    }
                }

                _ = try self.consume(.RightParen, "Expected ')' after tuple elements");

                return ast.ExprNode{ .Tuple = ast.TupleExpr{
                    .elements = try elements.toOwnedSlice(),
                    .span = makeSpan(paren_token),
                } };
            } else {
                // Single parenthesized expression
                _ = try self.consume(.RightParen, "Expected ')' after expression");
                return first_expr;
            }
        }

        return self.errorAtCurrent("Expected expression");
    }

    /// Parse struct declaration (placeholder)
    fn parseStruct(self: *Parser) ParserError!AstNode {
        const name_token = try self.consume(.Identifier, "Expected struct name");
        _ = try self.consume(.LeftBrace, "Expected '{' after struct name");

        var fields = std.ArrayList(ast.StructField).init(self.allocator);
        defer fields.deinit();

        while (!self.check(.RightBrace) and !self.isAtEnd()) {
            const field_name = try self.consume(.Identifier, "Expected field name");
            _ = try self.consume(.Colon, "Expected ':' after field name");
            const field_type = try self.parseType();
            _ = try self.consume(.Comma, "Expected ',' after field");

            try fields.append(ast.StructField{
                .name = field_name.lexeme,
                .typ = field_type,
                .span = makeSpan(field_name),
            });
        }

        _ = try self.consume(.RightBrace, "Expected '}' after struct fields");

        return AstNode{ .StructDecl = ast.StructDeclNode{
            .name = name_token.lexeme,
            .fields = try fields.toOwnedSlice(),
            .span = makeSpan(name_token),
        } };
    }

    /// Parse enum declaration with value assignments and base types
    fn parseEnum(self: *Parser) ParserError!AstNode {
        const name_token = try self.consume(.Identifier, "Expected enum name");

        // Parse optional base type: enum Status : u8 { ... }
        var base_type: ?ast.TypeRef = null;
        if (self.match(.Colon)) {
            base_type = try self.parseType();
        }

        _ = try self.consume(.LeftBrace, "Expected '{' after enum name");

        var variants = std.ArrayList(ast.EnumVariant).init(self.allocator);
        defer variants.deinit();

        var has_explicit_values = false;

        while (!self.check(.RightBrace) and !self.isAtEnd()) {
            const variant_name = try self.consume(.Identifier, "Expected variant name");

            // Parse optional explicit value assignment: VariantName = value
            var value: ?ast.ExprNode = null;
            if (self.match(.Equal)) {
                value = try self.parseExpression();
                has_explicit_values = true;
            }

            try variants.append(ast.EnumVariant{
                .name = variant_name.lexeme,
                .value = value,
                .span = makeSpan(variant_name),
                .discriminant = null, // Will be filled during semantic analysis
            });

            // Handle comma separator
            if (self.check(.RightBrace)) {
                break; // Allow trailing comma
            }
            _ = try self.consume(.Comma, "Expected ',' between enum variants");
        }

        _ = try self.consume(.RightBrace, "Expected '}' after enum variants");

        return AstNode{ .EnumDecl = ast.EnumDeclNode{
            .name = name_token.lexeme,
            .variants = try variants.toOwnedSlice(),
            .span = makeSpan(name_token),
            .base_type = base_type,
            .has_explicit_values = has_explicit_values,
        } };
    }

    /// Parse log declaration
    fn parseLogDecl(self: *Parser) ParserError!AstNode {
        const name_token = try self.consume(.Identifier, "Expected log name");
        _ = try self.consume(.LeftParen, "Expected '(' after log name");

        var fields = std.ArrayList(ast.LogField).init(self.allocator);
        defer fields.deinit();

        if (!self.check(.RightParen)) {
            repeat: while (true) {
                const field_name = try self.consumeIdentifierOrKeyword("Expected field name");
                _ = try self.consume(.Colon, "Expected ':' after field name");
                const field_type = try self.parseType();

                try fields.append(ast.LogField{
                    .name = field_name.lexeme,
                    .typ = field_type,
                    .span = makeSpan(field_name),
                });

                if (!self.match(.Comma)) break :repeat;
            }
        }

        _ = try self.consume(.RightParen, "Expected ')' after log fields");
        _ = self.match(.Semicolon); // Optional semicolon

        return AstNode{ .LogDecl = ast.LogDeclNode{
            .name = name_token.lexeme,
            .fields = try fields.toOwnedSlice(),
            .span = makeSpan(name_token),
        } };
    }

    // Helper methods

    fn match(self: *Parser, token_type: TokenType) bool {
        if (self.check(token_type)) {
            _ = self.advance();
            return true;
        }
        return false;
    }

    fn check(self: *Parser, token_type: TokenType) bool {
        if (self.isAtEnd()) return false;
        return self.peek().type == token_type;
    }

    fn advance(self: *Parser) Token {
        if (!self.isAtEnd()) self.current += 1;
        return self.previous();
    }

    fn isAtEnd(self: *Parser) bool {
        return self.peek().type == .Eof;
    }

    fn peek(self: *Parser) Token {
        return self.tokens[self.current];
    }

    fn previous(self: *Parser) Token {
        return self.tokens[self.current - 1];
    }

    fn consume(self: *Parser, token_type: TokenType, message: []const u8) ParserError!Token {
        if (self.check(token_type)) {
            return self.advance();
        }

        std.debug.print("Parser error: {s} (got {s})\n", .{ message, @tagName(self.peek().type) });
        return ParserError.ExpectedToken;
    }

    fn errorAtCurrent(self: *Parser, message: []const u8) ParserError {
        const current_token = self.peek();
        std.debug.print("Parser error at line {}, column {}: {s}\n", .{ current_token.line, current_token.column, message });
        return ParserError.UnexpectedToken;
    }

    /// Consume an identifier or keyword that can be used as an identifier in certain contexts
    fn consumeIdentifierOrKeyword(self: *Parser, message: []const u8) ParserError!Token {
        const current_token = self.peek();
        if (current_token.type == .Identifier or self.isKeywordThatCanBeIdentifier(current_token.type)) {
            return self.advance();
        }

        std.debug.print("Parser error: {s} (got {s})\n", .{ message, @tagName(current_token.type) });
        return ParserError.ExpectedToken;
    }

    /// Check if a keyword can be used as an identifier in certain contexts
    fn isKeywordThatCanBeIdentifier(self: *Parser, token_type: TokenType) bool {
        _ = self;
        return switch (token_type) {
            .From => true, // 'from' can be used as a parameter name in log declarations
            else => false,
        };
    }

    /// Match a keyword that can be used as an identifier
    fn matchKeywordAsIdentifier(self: *Parser) bool {
        if (self.isKeywordThatCanBeIdentifier(self.peek().type)) {
            _ = self.advance();
            return true;
        }
        return false;
    }

    /// Parse a struct instantiation expression (e.g., `MyStruct { a: 1, b: 2 }`)
    fn parseStructInstantiation(self: *Parser, name_token: Token) ParserError!ast.ExprNode {
        _ = try self.consume(.LeftBrace, "Expected '{' after struct name");

        var fields = std.ArrayList(ast.StructInstantiationField).init(self.allocator);
        defer fields.deinit();

        // Parse field initializers (field_name: value)
        while (!self.check(.RightBrace) and !self.isAtEnd()) {
            const field_name = try self.consumeIdentifierOrKeyword("Expected field name in struct instantiation");
            _ = try self.consume(.Colon, "Expected ':' after field name in struct instantiation");

            const field_value = try self.parseExpression();
            const field_value_ptr = try self.allocator.create(ast.ExprNode);
            field_value_ptr.* = field_value;

            try fields.append(ast.StructInstantiationField{
                .name = field_name.lexeme,
                .value = field_value_ptr,
                .span = makeSpan(field_name),
            });

            // Optional comma (but don't require it for last field)
            if (!self.check(.RightBrace)) {
                _ = try self.consume(.Comma, "Expected ',' after field in struct instantiation");
            } else {
                _ = self.match(.Comma); // Consume trailing comma if present
            }
        }

        _ = try self.consume(.RightBrace, "Expected '}' after struct instantiation fields");

        // Create the struct name identifier
        const struct_name_ptr = try self.allocator.create(ast.ExprNode);
        struct_name_ptr.* = ast.ExprNode{ .Identifier = ast.IdentifierExpr{
            .name = name_token.lexeme,
            .span = makeSpan(name_token),
        } };

        return ast.ExprNode{ .StructInstantiation = ast.StructInstantiationExpr{
            .struct_name = struct_name_ptr,
            .fields = try fields.toOwnedSlice(),
            .span = makeSpan(name_token),
        } };
    }
};

/// Helper extension for Token to create SourceSpan
fn makeSpan(token: Token) ast.SourceSpan {
    return ast.SourceSpan{
        .line = token.line,
        .column = token.column,
        .length = @intCast(token.lexeme.len),
    };
}

/// Convenience function for parsing tokens into AST
pub fn parse(allocator: Allocator, tokens: []const Token) ParserError![]AstNode {
    var parser = Parser.init(allocator, tokens);
    return parser.parse();
}

// Tests
test "parse simple contract" {
    const testing = std.testing;
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const allocator = arena.allocator();

    const source =
        \\contract Token {
        \\    pub fn hello() {
        \\        log "Hello";
        \\    }
        \\}
    ;

    const tokens = try lexer.scan(source, allocator);
    const nodes = try parse(allocator, tokens);

    try testing.expect(nodes.len == 1);
    try testing.expect(nodes[0] == .Contract);
    try testing.expectEqualStrings("Token", nodes[0].Contract.name);
}
