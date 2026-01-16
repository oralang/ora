// ============================================================================
// Statement Parser
// ============================================================================
//
// Parses statements and blocks.
//
// SUPPORTED STATEMENTS:
//   • Variable declarations, assignments
//   • Control flow: if, while, for, try-catch, return
//   • Expressions, blocks, log statements
//   • Requires/ensures/invariant (spec statements)
//
// ============================================================================

const std = @import("std");
const lexer = @import("../lexer.zig");
const ast = @import("../ast.zig");
const common = @import("common.zig");
const common_parsers = @import("common_parsers.zig");
const AstArena = @import("../ast/ast_arena.zig").AstArena;

const Token = lexer.Token;
const TokenType = lexer.TokenType;
const BaseParser = common.BaseParser;
const ParserCommon = common.ParserCommon;

// Import other parsers for cross-parser communication
const ExpressionParser = @import("expression_parser.zig").ExpressionParser;
const DeclarationParser = @import("declaration_parser.zig").DeclarationParser;

// Import control flow parsers
const control_flow = @import("statements/control_flow.zig");

// Import spec statement parsers
const spec = @import("statements/spec.zig");

/// Specialized parser for statements
pub const StatementParser = struct {
    base: BaseParser,
    expr_parser: ExpressionParser,
    decl_parser: DeclarationParser,

    pub fn init(tokens: []const Token, arena: *AstArena) StatementParser {
        return StatementParser{
            .base = BaseParser.init(tokens, arena),
            .expr_parser = ExpressionParser.init(tokens, arena),
            .decl_parser = DeclarationParser.init(tokens, arena),
        };
    }

    /// Sync sub-parser states with current position
    pub fn syncSubParsers(self: *StatementParser) void {
        self.expr_parser.base.current = self.base.current;
        self.decl_parser.base.current = self.base.current;
    }

    /// Update current position from sub-parser
    pub fn updateFromSubParser(self: *StatementParser, new_current: usize) void {
        self.base.current = new_current;
        self.syncSubParsers();
    }

    /// Parse statement
    pub fn parseStatement(self: *StatementParser) common.ParserError!ast.Statements.StmtNode {
        // check for @lock annotation before variable declarations
        if (try self.tryParseLockAnnotation()) |var_decl| {
            return ast.Statements.StmtNode{ .VariableDecl = var_decl };
        }

        // check for @unlock annotation
        if (try self.tryParseUnlockAnnotation()) |unlock_stmt| {
            return ast.Statements.StmtNode{ .Unlock = unlock_stmt };
        }

        // standalone @lock statement: @lock(expr);
        if (self.base.check(.At)) {
            const saved = self.base.current;
            _ = self.base.advance(); // consume '@'
            if (self.base.check(.Identifier) and std.mem.eql(u8, self.base.peek().lexeme, "lock")) {
                _ = self.base.advance(); // consume 'lock'
                _ = try self.base.consume(.LeftParen, "Expected '(' after @lock");
                self.syncSubParsers();
                const path_expr = try self.expr_parser.parseExpression();
                self.updateFromSubParser(self.expr_parser.base.current);
                _ = try self.base.consume(.RightParen, "Expected ')' after @lock expression");
                _ = self.base.match(.Semicolon);
                return ast.Statements.StmtNode{ .Lock = .{ .path = path_expr, .span = ParserCommon.makeSpan(self.base.previous()) } };
            }
            // not a lock statement; restore and continue
            self.base.current = saved;
        }

        // if an unknown @ directive appears, report it and recover
        if (self.base.check(.At)) {
            const saved = self.base.current;
            _ = self.base.advance();
            // known statements handled above: @lock, @unlock
            if (!(self.base.check(.Identifier) and (std.mem.eql(u8, self.base.peek().lexeme, "lock") or std.mem.eql(u8, self.base.peek().lexeme, "unlock")))) {
                _ = self.base.errorAtCurrent("Unknown @ directive; only @lock/@unlock statements are supported here") catch {};
                // recover to next statement boundary
                while (!self.base.isAtEnd() and !self.base.check(.Semicolon) and !self.base.check(.RightBrace)) {
                    _ = self.base.advance();
                }
                _ = self.base.match(.Semicolon);
            } else {
                // restore for normal handling if it's a known directive
                self.base.current = saved;
            }
        }

        // check for labeled statement (label: statement)
        // only check for labels before control flow statements (for, while)
        var label: ?[]const u8 = null;
        if (self.base.check(.Identifier)) {
            const saved = self.base.current;
            const label_token = self.base.peek();
            _ = self.base.advance(); // consume identifier
            if (self.base.match(.Colon)) {
                // check if next token is for or while
                if (self.base.check(.For) or self.base.check(.While)) {
                    // this is a label before a loop
                    label = try self.base.arena.createString(label_token.lexeme);
                } else {
                    // not a label before loop, restore position
                    self.base.current = saved;
                }
            } else {
                // not a label, restore position
                self.base.current = saved;
            }
        }

        // variable declarations and destructuring assignments
        if (self.isMemoryRegionKeyword() or self.base.check(.Let) or self.base.check(.Var)) {
            // check if this is a destructuring assignment (let .{...} = ...)
            if (try self.tryParseDestructuringAssignment()) |dest_assign| {
                return ast.Statements.StmtNode{ .DestructuringAssignment = dest_assign };
            }
            return ast.Statements.StmtNode{ .VariableDecl = try self.parseVariableDecl() };
        }

        // return statements
        if (self.base.match(.Return)) {
            return try self.parseReturnStatement();
        }

        // log statements
        if (self.base.match(.Log)) {
            return try self.parseLogStatement();
        }

        // assert statements (formal verification)
        if (self.base.match(.Assert)) {
            return try spec.parseAssertStatement(self);
        }

        // assume statements (formal verification)
        if (self.base.match(.Assume)) {
            return try spec.parseAssumeStatement(self);
        }

        // havoc statements (formal verification)
        if (self.base.match(.Havoc)) {
            return try spec.parseHavocStatement(self);
        }

        // requires statements
        if (self.base.match(.Requires)) {
            return try spec.parseRequiresStatement(self);
        }

        // ensures statements
        if (self.base.match(.Ensures)) {
            return try spec.parseEnsuresStatement(self);
        }

        // error declarations
        if (self.base.match(.Error)) {
            return try self.parseErrorDeclStatement();
        }

        // try-catch blocks
        if (self.base.match(.Try)) {
            return try control_flow.parseTryStatement(self);
        }

        // if statements
        if (self.base.match(.If)) {
            return try control_flow.parseIfStatement(self);
        }

        // while statements
        if (self.base.match(.While)) {
            const while_stmt = try control_flow.parseWhileStatement(self);
            // add label if present
            if (label) |l| {
                switch (while_stmt) {
                    .While => |w| {
                        var w_copy = w;
                        w_copy.label = l;
                        return ast.Statements.StmtNode{ .While = w_copy };
                    },
                    else => return while_stmt,
                }
            }
            return while_stmt;
        }

        // for statements
        if (self.base.match(.For)) {
            const for_stmt = try control_flow.parseForStatement(self);
            // add label if present
            if (label) |l| {
                switch (for_stmt) {
                    .ForLoop => |fl| {
                        var fl_copy = fl;
                        fl_copy.label = l;
                        return ast.Statements.StmtNode{ .ForLoop = fl_copy };
                    },
                    else => return for_stmt,
                }
            }
            return for_stmt;
        }

        // switch statements
        if (self.base.match(.Switch)) {
            return try control_flow.parseSwitchStatement(self);
        }

        // break statements
        if (self.base.match(.Break)) {
            return try control_flow.parseBreakStatement(self);
        }

        // continue statements
        if (self.base.match(.Continue)) {
            return try control_flow.parseContinueStatement(self);
        }

        // compound assignment statements (a += b, etc)
        if (try self.tryParseCompoundAssignmentStatement()) |comp_assign| {
            return ast.Statements.StmtNode{ .CompoundAssignment = comp_assign };
        }

        // labeled blocks (label: { statements })
        if (try self.tryParseLabeledBlock()) |labeled_block| {
            return ast.Statements.StmtNode{ .LabeledBlock = labeled_block };
        }

        // expression statements (fallback)
        return try self.parseExpressionStatement();
    }

    /// Parse block statement
    pub fn parseBlock(self: *StatementParser) common.ParserError!ast.Statements.BlockNode {
        _ = try self.base.consume(.LeftBrace, "Expected '{'");

        var statements = std.ArrayList(ast.Statements.StmtNode){};
        defer statements.deinit(self.base.arena.allocator());
        errdefer {
            // clean up statements on error
            for (statements.items) |*stmt| {
                ast.deinitStmtNode(self.base.arena.allocator(), stmt);
            }
        }

        while (!self.base.check(.RightBrace) and !self.base.isAtEnd()) {
            const stmt = try self.parseStatement();
            try statements.append(self.base.arena.allocator(), stmt);
        }

        const end_token = try self.base.consume(.RightBrace, "Expected '}' after block");

        return ast.Statements.BlockNode{
            .statements = try statements.toOwnedSlice(self.base.arena.allocator()),
            .span = ParserCommon.makeSpan(end_token),
        };
    }

    /// Parse variable declaration - DELEGATE TO DECLARATION PARSER
    fn parseVariableDecl(self: *StatementParser) !ast.Statements.VariableDeclNode {
        // delegate to declaration parser
        // create temporary TypeParser - this is part of circular deps that need fixing
        var type_parser = @import("type_parser.zig").TypeParser.init(self.base.tokens, self.base.arena);
        type_parser.base.current = self.base.current;
        type_parser.base.file_id = self.base.file_id;
        self.expr_parser.base.current = self.base.current;
        self.decl_parser.base.current = self.base.current;
        const result = try self.decl_parser.parseVariableDecl(&type_parser, &self.expr_parser);
        self.base.current = self.decl_parser.base.current;
        self.syncSubParsers();
        return result;
    }

    /// Parse return statement
    fn parseReturnStatement(self: *StatementParser) common.ParserError!ast.Statements.StmtNode {
        var value: ?ast.Expressions.ExprNode = null;
        if (!self.base.check(.Semicolon) and !self.base.check(.RightBrace)) {
            self.syncSubParsers();
            value = try self.expr_parser.parseExpression();
            self.updateFromSubParser(self.expr_parser.base.current);
        }
        _ = self.base.match(.Semicolon);

        return ast.Statements.StmtNode{ .Return = ast.Statements.ReturnNode{
            .value = value,
            .span = ParserCommon.makeSpan(self.base.previous()),
        } };
    }

    /// Parse log statement
    fn parseLogStatement(self: *StatementParser) common.ParserError!ast.Statements.StmtNode {
        const event_name_token = try self.base.consume(.Identifier, "Expected event name");
        _ = try self.base.consume(.LeftParen, "Expected '(' after event name");

        var args = std.ArrayList(ast.Expressions.ExprNode){};
        defer args.deinit(self.base.arena.allocator());

        if (!self.base.check(.RightParen)) {
            repeat: while (true) {
                // use expression parser for log arguments
                self.syncSubParsers();
                const arg = try self.expr_parser.parseExpressionNoComma();
                self.updateFromSubParser(self.expr_parser.base.current);
                try args.append(self.base.arena.allocator(), arg);
                if (!self.base.match(.Comma)) break :repeat;
            }
        }

        _ = try self.base.consume(.RightParen, "Expected ')' after log arguments");
        _ = try self.base.consume(.Semicolon, "Expected ';' after log statement");

        return ast.Statements.StmtNode{ .Log = ast.Statements.LogNode{
            .event_name = try self.base.arena.createString(event_name_token.lexeme),
            .args = try args.toOwnedSlice(self.base.arena.allocator()),
            .span = ParserCommon.makeSpan(event_name_token),
        } };
    }

    /// Parse error declaration statement
    fn parseErrorDeclStatement(self: *StatementParser) common.ParserError!ast.Statements.StmtNode {
        const error_token = self.base.previous();
        const name_token = try self.base.consume(.Identifier, "Expected error name");
        _ = try self.base.consume(.Semicolon, "Expected ';' after error declaration");

        return ast.Statements.StmtNode{ .ErrorDecl = ast.Statements.ErrorDeclNode{
            .name = try self.base.arena.createString(name_token.lexeme),
            .parameters = null,
            .span = ParserCommon.makeSpan(error_token),
        } };
    }

    /// Check if current token is a memory region keyword
    fn isMemoryRegionKeyword(self: *StatementParser) bool {
        return ParserCommon.isMemoryRegionKeyword(self.base.peek().type);
    }
    /// Try to parse @lock annotation, returns variable declaration if found
    fn tryParseLockAnnotation(self: *StatementParser) !?ast.Statements.VariableDeclNode {
        if (self.base.check(.At)) {
            const saved_pos = self.base.current;
            _ = self.base.advance(); // consume @
            if (self.base.check(.Identifier) and std.mem.eql(u8, self.base.peek().lexeme, "lock")) {
                _ = self.base.advance(); // consume "lock"

                // check if this is followed by a variable declaration
                if (self.isMemoryRegionKeyword() or self.base.check(.Let) or self.base.check(.Var)) {
                    // delegate to declaration parser (get TypeParser from parent)
                    // for now, create temporary parsers - this is part of circular deps that need fixing
                    self.base.current = saved_pos;
                    var type_parser = @import("type_parser.zig").TypeParser.init(self.base.tokens, self.base.arena);
                    type_parser.base.current = self.base.current;
                    type_parser.base.file_id = self.base.file_id;
                    self.expr_parser.base.current = self.base.current;
                    if (try self.decl_parser.tryParseLockAnnotation(&type_parser, &self.expr_parser)) |var_decl| {
                        self.base.current = self.decl_parser.base.current;
                        self.syncSubParsers();
                        return var_decl;
                    }
                    return error.UnexpectedToken;
                }
            }

            // not @lock or not followed by variable declaration, restore position
            self.base.current = saved_pos;
        }
        return null;
    }

    /// Try to parse destructuring assignment (let .{field1, field2} = expr)
    fn tryParseDestructuringAssignment(self: *StatementParser) common.ParserError!?ast.Statements.DestructuringAssignmentNode {
        const saved_pos = self.base.current;

        // parse memory region and variable kind
        // parse declaration keyword and region quickly, we don't need the value here
        if (!(self.base.match(.Let) or self.base.match(.Var) or self.base.match(.Storage) or self.base.match(.Memory) or self.base.match(.Tstore))) {
            self.base.current = saved_pos;
            return null;
        }

        // check for destructuring pattern (starts with .{)
        if (!self.base.check(.Dot)) {
            // not a destructuring assignment, restore position
            self.base.current = saved_pos;
            return null;
        }

        _ = self.base.advance(); // consume '.'
        _ = try self.base.consume(.LeftBrace, "Expected '{' after '.' in destructuring pattern");

        // parse struct destructuring fields
        var fields = std.ArrayList(ast.Expressions.StructDestructureField){};
        defer fields.deinit(self.base.arena.allocator());

        if (!self.base.check(.RightBrace)) {
            repeat: while (true) {
                const field_name_token = try self.base.consume(.Identifier, "Expected field name in destructuring pattern");
                var variable_name = try self.base.arena.createString(field_name_token.lexeme);

                // check for field renaming (field: variable)
                if (self.base.match(.Colon)) {
                    const var_name_token = try self.base.consume(.Identifier, "Expected variable name after ':' in destructuring pattern");
                    variable_name = try self.base.arena.createString(var_name_token.lexeme);
                }

                try fields.append(self.base.arena.allocator(), ast.Expressions.StructDestructureField{
                    .name = try self.base.arena.createString(field_name_token.lexeme),
                    .variable = variable_name,
                    .span = common.ParserCommon.makeSpan(field_name_token),
                });

                if (!self.base.match(.Comma)) break :repeat;
            }
        }

        _ = try self.base.consume(.RightBrace, "Expected '}' after destructuring fields");
        _ = try self.base.consume(.Equal, "Expected '=' after destructuring pattern");

        // parse the value expression
        self.syncSubParsers();
        const value_expr = try self.expr_parser.parseExpression();
        self.updateFromSubParser(self.expr_parser.base.current);

        const value_ptr = try self.base.arena.createNode(ast.Expressions.ExprNode);
        value_ptr.* = value_expr;

        _ = self.base.match(.Semicolon);

        const pattern = ast.Expressions.DestructuringPattern{ .Struct = try fields.toOwnedSlice(self.base.arena.allocator()) };

        return ast.Statements.DestructuringAssignmentNode{
            .pattern = pattern,
            .value = value_ptr,
            .span = common.ParserCommon.makeSpan(self.base.previous()),
        };
    }

    /// Try to parse @unlock annotation
    fn tryParseUnlockAnnotation(self: *StatementParser) common.ParserError!?ast.Statements.UnlockNode {
        if (self.base.check(.At)) {
            const saved_pos = self.base.current;
            _ = self.base.advance(); // consume @

            if (self.base.check(.Identifier) and std.mem.eql(u8, self.base.peek().lexeme, "unlock")) {
                _ = self.base.advance(); // consume "unlock"

                _ = try self.base.consume(.LeftParen, "Expected '(' after @unlock");

                // parse the path expression
                self.syncSubParsers();
                const path_expr = try self.expr_parser.parseExpression();
                self.updateFromSubParser(self.expr_parser.base.current);

                _ = try self.base.consume(.RightParen, "Expected ')' after @unlock expression");
                _ = self.base.match(.Semicolon);

                return ast.Statements.UnlockNode{
                    .path = path_expr,
                    .span = common.ParserCommon.makeSpan(self.base.previous()),
                };
            }

            // not @unlock, restore position
            self.base.current = saved_pos;
        }
        return null;
    }

    /// Try to parse labeled block (label: { statements })
    fn tryParseLabeledBlock(self: *StatementParser) !?ast.Statements.LabeledBlockNode {
        if (self.base.check(.Identifier)) {
            const saved_pos = self.base.current;
            const label_token = self.base.advance(); // consume identifier

            if (self.base.match(.Colon)) {
                // case 1: label: { ... }
                if (self.base.check(.LeftBrace)) {
                    const block = try self.parseBlock();
                    return ast.Statements.LabeledBlockNode{
                        .label = try self.base.arena.createString(label_token.lexeme),
                        .block = block,
                        .span = common.ParserCommon.makeSpan(label_token),
                    };
                }
                // case 2: label: switch (...) { ... }
                if (self.base.match(.Switch)) {
                    const switch_stmt = try control_flow.parseSwitchStatement(self);
                    var stmts = std.ArrayList(ast.Statements.StmtNode){};
                    defer stmts.deinit(self.base.arena.allocator());
                    try stmts.append(self.base.arena.allocator(), switch_stmt);
                    const block = ast.Statements.BlockNode{
                        .statements = try self.base.arena.createSlice(ast.Statements.StmtNode, stmts.items.len),
                        .span = common.ParserCommon.makeSpan(label_token),
                    };
                    for (stmts.items, 0..) |s, i| block.statements[i] = s;
                    return ast.Statements.LabeledBlockNode{
                        .label = try self.base.arena.createString(label_token.lexeme),
                        .block = block,
                        .span = common.ParserCommon.makeSpan(label_token),
                    };
                }
                // unknown after label:, restore
                self.base.current = saved_pos;
                return null;
            }

            // not a labeled block, restore position
            self.base.current = saved_pos;
        }
        return null;
    }

    /// Parse expression statement (fallback)
    fn parseExpressionStatement(self: *StatementParser) common.ParserError!ast.Statements.StmtNode {
        // use integrated expression parser for expression statements
        self.syncSubParsers();
        const expr = try self.expr_parser.parseExpression();
        self.updateFromSubParser(self.expr_parser.base.current);
        _ = self.base.match(.Semicolon);
        return ast.Statements.StmtNode{ .Expr = expr };
    }

    /// Try to parse a compound assignment statement (a += b, a -= b, etc.)
    fn tryParseCompoundAssignmentStatement(self: *StatementParser) common.ParserError!?ast.statements.CompoundAssignmentNode {
        const saved_pos = self.base.current;

        // we need to parse a potential lvalue first
        self.syncSubParsers();
        if (self.base.isAtEnd()) return null;

        // parse the left-hand side (must be a valid lvalue)
        const lhs = try self.expr_parser.parseExpression();
        self.updateFromSubParser(self.expr_parser.base.current);

        // check if the next token is a compound operator
        if (self.base.match(.PlusEqual) or self.base.match(.MinusEqual) or
            self.base.match(.StarEqual) or self.base.match(.SlashEqual) or
            self.base.match(.PercentEqual))
        {
            const op_token = self.base.previous();

            // map the token type to CompoundAssignmentOp
            const compound_op: ast.Operators.Compound = switch (op_token.type) {
                .PlusEqual => .PlusEqual,
                .MinusEqual => .MinusEqual,
                .StarEqual => .StarEqual,
                .SlashEqual => .SlashEqual,
                .PercentEqual => .PercentEqual,
                else => unreachable, // We already checked these token types
            };

            // parse the right-hand side
            self.syncSubParsers();
            const rhs = try self.expr_parser.parseExpression();
            self.updateFromSubParser(self.expr_parser.base.current);

            // expect a semicolon
            _ = try self.base.consume(.Semicolon, "Expected ';' after compound assignment");

            // create lhs and rhs pointers
            const lhs_ptr = try self.base.arena.createNode(ast.Expressions.ExprNode);
            lhs_ptr.* = lhs;
            const rhs_ptr = try self.base.arena.createNode(ast.Expressions.ExprNode);
            rhs_ptr.* = rhs;

            return ast.statements.CompoundAssignmentNode{
                .target = lhs_ptr,
                .operator = compound_op,
                .value = rhs_ptr,
                .span = common.ParserCommon.makeSpan(op_token),
            };
        }

        // if we didn't find a compound assignment, restore position and return null
        self.base.current = saved_pos;
        return null;
    }
};
