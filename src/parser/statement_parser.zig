const std = @import("std");
const lexer = @import("../lexer.zig");
const ast = @import("../ast.zig");
const common = @import("common.zig");
const common_parsers = @import("common_parsers.zig");

const Token = lexer.Token;
const TokenType = lexer.TokenType;
const BaseParser = common.BaseParser;
const ParserCommon = common.ParserCommon;

// Import other parsers for cross-parser communication
const ExpressionParser = @import("expression_parser.zig").ExpressionParser;
const DeclarationParser = @import("declaration_parser.zig").DeclarationParser;

/// Specialized parser for statements
pub const StatementParser = struct {
    base: BaseParser,
    expr_parser: ExpressionParser,
    decl_parser: DeclarationParser,

    pub fn init(tokens: []const Token, arena: *@import("../ast/ast_arena.zig").AstArena) StatementParser {
        return StatementParser{
            .base = BaseParser.init(tokens, arena),
            .expr_parser = ExpressionParser.init(tokens, arena),
            .decl_parser = DeclarationParser.init(tokens, arena),
        };
    }

    /// Sync sub-parser states with current position
    fn syncSubParsers(self: *StatementParser) void {
        self.expr_parser.base.current = self.base.current;
        self.decl_parser.base.current = self.base.current;
    }

    /// Update current position from sub-parser
    fn updateFromSubParser(self: *StatementParser, new_current: usize) void {
        self.base.current = new_current;
        self.syncSubParsers();
    }

    /// Parse statement
    pub fn parseStatement(self: *StatementParser) common.ParserError!ast.Statements.StmtNode {
        // Check for @lock annotation before variable declarations
        if (try self.tryParseLockAnnotation()) |var_decl| {
            return ast.Statements.StmtNode{ .VariableDecl = var_decl };
        }

        // Check for @unlock annotation
        if (try self.tryParseUnlockAnnotation()) |unlock_stmt| {
            return ast.Statements.StmtNode{ .Unlock = unlock_stmt };
        }

        // Standalone @lock statement: @lock(expr);
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
                return ast.Statements.StmtNode{ .Lock = .{ .path = path_expr, .span = ParserCommon.makeStmtSpan(self.base.previous()) } };
            }
            // Not a lock statement; restore and continue
            self.base.current = saved;
        }

        // If an unknown @ directive appears, report it and recover
        if (self.base.check(.At)) {
            const saved = self.base.current;
            _ = self.base.advance();
            // Known statements handled above: @lock, @unlock
            if (!(self.base.check(.Identifier) and (std.mem.eql(u8, self.base.peek().lexeme, "lock") or std.mem.eql(u8, self.base.peek().lexeme, "unlock")))) {
                _ = self.base.errorAtCurrent("Unknown @ directive; only @lock/@unlock statements are supported here") catch {};
                // Recover to next statement boundary
                while (!self.base.isAtEnd() and !self.base.check(.Semicolon) and !self.base.check(.RightBrace)) {
                    _ = self.base.advance();
                }
                _ = self.base.match(.Semicolon);
            } else {
                // Restore for normal handling if it's a known directive
                self.base.current = saved;
            }
        }

        // Variable declarations and destructuring assignments
        if (self.isMemoryRegionKeyword() or self.base.check(.Let) or self.base.check(.Var)) {
            // Check if this is a destructuring assignment (let .{...} = ...)
            if (try self.tryParseDestructuringAssignment()) |dest_assign| {
                return ast.Statements.StmtNode{ .DestructuringAssignment = dest_assign };
            }
            return ast.Statements.StmtNode{ .VariableDecl = try self.parseVariableDecl() };
        }

        // Return statements
        if (self.base.match(.Return)) {
            return try self.parseReturnStatement();
        }

        // Log statements
        if (self.base.match(.Log)) {
            return try self.parseLogStatement();
        }

        // Requires statements
        if (self.base.match(.Requires)) {
            return try self.parseRequiresStatement();
        }

        // Ensures statements
        if (self.base.match(.Ensures)) {
            return try self.parseEnsuresStatement();
        }

        // Error declarations
        if (self.base.match(.Error)) {
            return try self.parseErrorDeclStatement();
        }

        // Try-catch blocks
        if (self.base.match(.Try)) {
            return try self.parseTryStatement();
        }

        // If statements
        if (self.base.match(.If)) {
            return try self.parseIfStatement();
        }

        // While statements
        if (self.base.match(.While)) {
            return try self.parseWhileStatement();
        }

        // For statements
        if (self.base.match(.For)) {
            return try self.parseForStatement();
        }

        // Switch statements
        if (self.base.match(.Switch)) {
            return try self.parseSwitchStatement();
        }

        // Break statements
        if (self.base.match(.Break)) {
            return try self.parseBreakStatement();
        }

        // Continue statements
        if (self.base.match(.Continue)) {
            return try self.parseContinueStatement();
        }

        // Compound assignment statements (a += b, etc)
        if (try self.tryParseCompoundAssignmentStatement()) |comp_assign| {
            return ast.Statements.StmtNode{ .CompoundAssignment = comp_assign };
        }

        // Move statements (move amount from src to dest;)
        if (try self.tryParseMoveStatement()) |move_stmt| {
            return ast.Statements.StmtNode{ .Move = move_stmt };
        }

        // Labeled blocks (label: { statements })
        if (try self.tryParseLabeledBlock()) |labeled_block| {
            return ast.Statements.StmtNode{ .LabeledBlock = labeled_block };
        }

        // Expression statements (fallback)
        return try self.parseExpressionStatement();
    }

    /// Parse block statement
    pub fn parseBlock(self: *StatementParser) common.ParserError!ast.Statements.BlockNode {
        _ = try self.base.consume(.LeftBrace, "Expected '{'");

        var statements = std.ArrayList(ast.Statements.StmtNode).init(self.base.arena.allocator());
        defer statements.deinit();
        errdefer {
            // Clean up statements on error
            for (statements.items) |*stmt| {
                ast.deinitStmtNode(self.base.arena.allocator(), stmt);
            }
        }

        while (!self.base.check(.RightBrace) and !self.base.isAtEnd()) {
            const stmt = try self.parseStatement();
            try statements.append(stmt);
        }

        const end_token = try self.base.consume(.RightBrace, "Expected '}' after block");

        return ast.Statements.BlockNode{
            .statements = try statements.toOwnedSlice(),
            .span = ParserCommon.makeSpan(end_token),
        };
    }

    /// Parse variable declaration - DELEGATE TO DECLARATION PARSER
    fn parseVariableDecl(self: *StatementParser) !ast.Statements.VariableDeclNode {
        // Delegate to declaration parser
        self.syncSubParsers();
        const result = try self.decl_parser.parseVariableDecl();
        self.updateFromSubParser(self.decl_parser.base.current);
        return result;
    }

    /// Parse return statement - MIGRATED FROM ORIGINAL
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
            .span = ParserCommon.makeStmtSpan(self.base.previous()),
        } };
    }

    /// Parse log statement
    fn parseLogStatement(self: *StatementParser) common.ParserError!ast.Statements.StmtNode {
        const event_name_token = try self.base.consume(.Identifier, "Expected event name");
        _ = try self.base.consume(.LeftParen, "Expected '(' after event name");

        var args = std.ArrayList(ast.Expressions.ExprNode).init(self.base.arena.allocator());
        defer args.deinit();

        if (!self.base.check(.RightParen)) {
            repeat: while (true) {
                // Use expression parser for log arguments
                self.syncSubParsers();
                const arg = try self.expr_parser.parseExpression();
                self.updateFromSubParser(self.expr_parser.base.current);
                try args.append(arg);
                if (!self.base.match(.Comma)) break :repeat;
            }
        }

        _ = try self.base.consume(.RightParen, "Expected ')' after log arguments");
        _ = try self.base.consume(.Semicolon, "Expected ';' after log statement");

        return ast.Statements.StmtNode{ .Log = ast.Statements.LogNode{
            .event_name = try self.base.arena.createString(event_name_token.lexeme),
            .args = try args.toOwnedSlice(),
            .span = ParserCommon.makeStmtSpan(event_name_token),
        } };
    }

    /// Parse requires statement - MIGRATED FROM ORIGINAL
    fn parseRequiresStatement(self: *StatementParser) common.ParserError!ast.Statements.StmtNode {
        const requires_token = self.base.previous();
        // Only requires(condition) is allowed
        _ = try self.base.consume(.LeftParen, "Expected '(' after 'requires'");
        // Use expression parser for requires condition
        self.syncSubParsers();
        const condition = try self.expr_parser.parseExpression();
        self.updateFromSubParser(self.expr_parser.base.current);
        _ = try self.base.consume(.RightParen, "Expected ')' after requires condition");
        // Strict: disallow trailing semicolon
        if (self.base.match(.Semicolon)) {
            try self.base.errorAtCurrent("Unexpected ';' after requires(...) (no semicolon allowed)");
            return error.UnexpectedToken;
        }

        return ast.Statements.StmtNode{ .Requires = ast.Statements.RequiresNode{
            .condition = condition,
            .span = ParserCommon.makeStmtSpan(requires_token),
        } };
    }

    /// Parse ensures statement - MIGRATED FROM ORIGINAL
    fn parseEnsuresStatement(self: *StatementParser) common.ParserError!ast.Statements.StmtNode {
        const ensures_token = self.base.previous();
        // Only ensures(condition) is allowed
        _ = try self.base.consume(.LeftParen, "Expected '(' after 'ensures'");
        // Use expression parser for ensures condition
        self.syncSubParsers();
        const condition = try self.expr_parser.parseExpression();
        self.updateFromSubParser(self.expr_parser.base.current);
        _ = try self.base.consume(.RightParen, "Expected ')' after ensures condition");
        // Strict: disallow trailing semicolon
        if (self.base.match(.Semicolon)) {
            try self.base.errorAtCurrent("Unexpected ';' after ensures(...) (no semicolon allowed)");
            return error.UnexpectedToken;
        }

        return ast.Statements.StmtNode{ .Ensures = ast.Statements.EnsuresNode{
            .condition = condition,
            .span = ParserCommon.makeStmtSpan(ensures_token),
        } };
    }

    /// Parse error declaration statement - MIGRATED FROM ORIGINAL
    fn parseErrorDeclStatement(self: *StatementParser) common.ParserError!ast.Statements.StmtNode {
        const error_token = self.base.previous();
        const name_token = try self.base.consume(.Identifier, "Expected error name");
        _ = try self.base.consume(.Semicolon, "Expected ';' after error declaration");

        return ast.Statements.StmtNode{ .ErrorDecl = ast.Statements.ErrorDeclNode{
            .name = try self.base.arena.createString(name_token.lexeme),
            .parameters = null,
            .span = ParserCommon.makeStmtSpan(error_token),
        } };
    }

    /// Check if current token is a memory region keyword
    fn isMemoryRegionKeyword(self: *StatementParser) bool {
        return self.base.check(.Const) or self.base.check(.Immutable) or
            self.base.check(.Storage) or self.base.check(.Memory) or self.base.check(.Tstore);
    }
    /// Try to parse @lock annotation, returns variable declaration if found
    fn tryParseLockAnnotation(self: *StatementParser) !?ast.Statements.VariableDeclNode {
        if (self.base.check(.At)) {
            const saved_pos = self.base.current;
            _ = self.base.advance(); // consume @
            if (self.base.check(.Identifier) and std.mem.eql(u8, self.base.peek().lexeme, "lock")) {
                _ = self.base.advance(); // consume "lock"

                // Check if this is followed by a variable declaration
                if (self.isMemoryRegionKeyword() or self.base.check(.Let) or self.base.check(.Var)) {
                    // Delegate to declaration parser
                    // Restore position to start of @lock and let declaration parser handle it
                    self.base.current = saved_pos;
                    self.syncSubParsers();
                    if (try self.decl_parser.tryParseLockAnnotation()) |var_decl| {
                        self.updateFromSubParser(self.decl_parser.base.current);
                        return var_decl;
                    }
                    return error.UnexpectedToken;
                }
            }

            // Not @lock or not followed by variable declaration, restore position
            self.base.current = saved_pos;
        }
        return null;
    }

    /// Try to parse destructuring assignment (let .{field1, field2} = expr)
    fn tryParseDestructuringAssignment(self: *StatementParser) common.ParserError!?ast.Statements.DestructuringAssignmentNode {
        const saved_pos = self.base.current;

        // Parse memory region and variable kind
        // Parse declaration keyword and region quickly, we don't need the value here
        if (!(self.base.match(.Let) or self.base.match(.Var) or self.base.match(.Storage) or self.base.match(.Memory) or self.base.match(.Tstore))) {
            self.base.current = saved_pos;
            return null;
        }

        // Check for destructuring pattern (starts with .{)
        if (!self.base.check(.Dot)) {
            // Not a destructuring assignment, restore position
            self.base.current = saved_pos;
            return null;
        }

        _ = self.base.advance(); // consume '.'
        _ = try self.base.consume(.LeftBrace, "Expected '{' after '.' in destructuring pattern");

        // Parse struct destructuring fields
        var fields = std.ArrayList(ast.Expressions.StructDestructureField).init(self.base.arena.allocator());
        defer fields.deinit();

        if (!self.base.check(.RightBrace)) {
            repeat: while (true) {
                const field_name_token = try self.base.consume(.Identifier, "Expected field name in destructuring pattern");
                var variable_name = try self.base.arena.createString(field_name_token.lexeme);

                // Check for field renaming (field: variable)
                if (self.base.match(.Colon)) {
                    const var_name_token = try self.base.consume(.Identifier, "Expected variable name after ':' in destructuring pattern");
                    variable_name = try self.base.arena.createString(var_name_token.lexeme);
                }

                try fields.append(ast.Expressions.StructDestructureField{
                    .name = try self.base.arena.createString(field_name_token.lexeme),
                    .variable = variable_name,
                    .span = common.ParserCommon.makeSpan(field_name_token),
                });

                if (!self.base.match(.Comma)) break :repeat;
            }
        }

        _ = try self.base.consume(.RightBrace, "Expected '}' after destructuring fields");
        _ = try self.base.consume(.Equal, "Expected '=' after destructuring pattern");

        // Parse the value expression
        self.syncSubParsers();
        const value_expr = try self.expr_parser.parseExpression();
        self.updateFromSubParser(self.expr_parser.base.current);

        const value_ptr = try self.base.arena.createNode(ast.Expressions.ExprNode);
        value_ptr.* = value_expr;

        _ = self.base.match(.Semicolon);

        const pattern = ast.Expressions.DestructuringPattern{ .Struct = try fields.toOwnedSlice() };

        return ast.Statements.DestructuringAssignmentNode{
            .pattern = pattern,
            .value = value_ptr,
            .span = common.ParserCommon.makeSpan(self.base.previous()),
        };
    }

    /// Try to parse move statement: move <amount> from <source> to <dest>;
    fn tryParseMoveStatement(self: *StatementParser) common.ParserError!?ast.Statements.MoveNode {
        if (!self.base.match(.Move)) return null;

        // Parse amount expression
        self.syncSubParsers();
        const amount = try self.expr_parser.parseExpression();
        self.updateFromSubParser(self.expr_parser.base.current);

        // Expect 'from'
        _ = try self.base.consume(.From, "Expected 'from' after move amount");

        // Parse source expression
        self.syncSubParsers();
        const source = try self.expr_parser.parseExpression();
        self.updateFromSubParser(self.expr_parser.base.current);

        // Expect 'to'
        _ = try self.base.consume(.To, "Expected 'to' after source in move statement");

        // Parse destination expression
        self.syncSubParsers();
        const dest = try self.expr_parser.parseExpression();
        self.updateFromSubParser(self.expr_parser.base.current);

        _ = self.base.match(.Semicolon);

        return ast.Statements.MoveNode{
            .expr = amount,
            .source = source,
            .dest = dest,
            .amount = amount,
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

                // Parse the path expression
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

            // Not @unlock, restore position
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
                // This is a labeled block
                const block = try self.parseBlock();

                return ast.Statements.LabeledBlockNode{
                    .label = try self.base.arena.createString(label_token.lexeme),
                    .block = block,
                    .span = common.ParserCommon.makeSpan(label_token),
                };
            }

            // Not a labeled block, restore position
            self.base.current = saved_pos;
        }
        return null;
    }

    /// Parse expression statement (fallback)
    fn parseExpressionStatement(self: *StatementParser) common.ParserError!ast.Statements.StmtNode {
        // Use integrated expression parser for expression statements
        self.syncSubParsers();
        const expr = try self.expr_parser.parseExpression();
        self.updateFromSubParser(self.expr_parser.base.current);
        _ = self.base.match(.Semicolon);
        return ast.Statements.StmtNode{ .Expr = expr };
    }

    /// Parse if statement
    fn parseIfStatement(self: *StatementParser) common.ParserError!ast.Statements.StmtNode {
        _ = try self.base.consume(.LeftParen, "Expected '(' after 'if'");

        // Use expression parser for condition
        self.syncSubParsers();
        const condition = try self.expr_parser.parseExpression();
        self.updateFromSubParser(self.expr_parser.base.current);

        _ = try self.base.consume(.RightParen, "Expected ')' after if condition");

        const then_branch = try self.parseBlock();

        var else_branch: ?ast.Statements.BlockNode = null;
        if (self.base.match(.Else)) {
            else_branch = try self.parseBlock();
        }

        return ast.Statements.StmtNode{ .If = ast.Statements.IfNode{
            .condition = condition,
            .then_branch = then_branch,
            .else_branch = else_branch,
            .span = self.base.spanFromToken(self.base.previous()),
        } };
    }

    /// Parse while statement - MIGRATED FROM ORIGINAL
    fn parseWhileStatement(self: *StatementParser) common.ParserError!ast.Statements.StmtNode {
        _ = try self.base.consume(.LeftParen, "Expected '(' after 'while'");

        // Use expression parser for condition
        self.syncSubParsers();
        const condition = try self.expr_parser.parseExpression();
        self.updateFromSubParser(self.expr_parser.base.current);

        _ = try self.base.consume(.RightParen, "Expected ')' after while condition");

        const body = try self.parseBlock();

        // TODO: Invariants are not currently part of the language syntax, but they should be
        var invariants = std.ArrayList(ast.Expressions.ExprNode).init(self.base.arena.allocator());
        defer invariants.deinit();

        return ast.Statements.StmtNode{ .While = ast.Statements.WhileNode{
            .condition = condition,
            .body = body,
            .invariants = try invariants.toOwnedSlice(),
            .span = self.base.spanFromToken(self.base.previous()),
        } };
    }

    /// Parse for statement (Zig-style: for (expr) |var1, var2| stmt) - MIGRATED FROM ORIGINAL
    fn parseForStatement(self: *StatementParser) common.ParserError!ast.Statements.StmtNode {
        const for_token = self.base.previous();

        _ = try self.base.consume(.LeftParen, "Expected '(' after 'for'");

        // Use expression parser for iterable
        self.syncSubParsers();
        const iterable = try self.expr_parser.parseExpression();
        self.updateFromSubParser(self.expr_parser.base.current);

        _ = try self.base.consume(.RightParen, "Expected ')' after for expression");

        _ = try self.base.consume(.Pipe, "Expected '|' after for expression");
        const var1_token = try self.base.consume(.Identifier, "Expected loop variable");
        const var1 = try self.base.arena.createString(var1_token.lexeme);

        var var2: ?[]const u8 = null;
        if (self.base.match(.Comma)) {
            const var2_token = try self.base.consume(.Identifier, "Expected second loop variable");
            var2 = try self.base.arena.createString(var2_token.lexeme);
        }

        _ = try self.base.consume(.Pipe, "Expected '|' after loop variables");

        const body = try self.parseBlock();

        const pattern = if (var2) |v2|
            ast.Statements.LoopPattern{ .IndexPair = .{ .item = var1, .index = v2, .span = self.base.spanFromToken(var1_token) } }
        else
            ast.Statements.LoopPattern{ .Single = .{ .name = var1, .span = self.base.spanFromToken(var1_token) } };

        return ast.Statements.StmtNode{ .ForLoop = ast.Statements.ForLoopNode{
            .iterable = iterable,
            .pattern = pattern,
            .body = body,
            .span = self.base.spanFromToken(for_token),
        } };
    }

    /// Parse switch statement - MIGRATED FROM ORIGINAL
    fn parseSwitchStatement(self: *StatementParser) common.ParserError!ast.Statements.StmtNode {
        const switch_token = self.base.previous();

        // Parse required switch condition: switch (expr)
        _ = try self.base.consume(.LeftParen, "Expected '(' after 'switch'");

        // Use expression parser for condition
        self.syncSubParsers();
        const condition = try self.expr_parser.parseExpression();
        self.updateFromSubParser(self.expr_parser.base.current);

        _ = try self.base.consume(.RightParen, "Expected ')' after switch condition");

        _ = try self.base.consume(.LeftBrace, "Expected '{' after switch condition");

        var cases = std.ArrayList(ast.Switch.Case).init(self.base.arena.allocator());
        defer cases.deinit();

        var default_case: ?ast.Statements.BlockNode = null;

        // Parse switch arms
        while (!self.base.check(.RightBrace) and !self.base.isAtEnd()) {
            if (self.base.match(.Else)) {
                // Parse else clause
                _ = try self.base.consume(.Arrow, "Expected '=>' after 'else'");
                const else_body = try common_parsers.parseSwitchBody(&self.base, &self.expr_parser, .StatementArm);
                switch (else_body) {
                    .Block => |block| {
                        default_case = block;
                    },
                    .LabeledBlock => |labeled| {
                        default_case = labeled.block;
                    },
                    .Expression => |expr_ptr| {
                        // Wrap expression into a block for statement AST
                        var stmts = try self.base.arena.createSlice(ast.Statements.StmtNode, 1);
                        stmts[0] = ast.Statements.StmtNode{ .Expr = expr_ptr.* };
                        default_case = ast.Statements.BlockNode{
                            .statements = stmts,
                            .span = ParserCommon.makeSpan(self.base.previous()),
                        };
                    },
                }
                break;
            }

            // Parse switch pattern using common parser
            const pattern = try common_parsers.parseSwitchPattern(&self.base, &self.expr_parser);
            _ = try self.base.consume(.Arrow, "Expected '=>' after switch pattern");

            // Parse switch body using common parser (statement arms require ';' after expr bodies)
            const body = try common_parsers.parseSwitchBody(&self.base, &self.expr_parser, .StatementArm);

            const case = ast.Switch.Case{
                .pattern = pattern,
                .body = body,
                .span = ParserCommon.makeSpan(self.base.previous()),
            };

            try cases.append(case);

            // Optional comma between cases
            _ = self.base.match(.Comma);
        }

        _ = try self.base.consume(.RightBrace, "Expected '}' after switch cases");

        return ast.Statements.StmtNode{ .Switch = ast.Statements.SwitchNode{
            .condition = condition,
            .cases = try cases.toOwnedSlice(),
            .default_case = default_case,
            .span = self.base.spanFromToken(switch_token),
        } };
    }

    /// Parse break statement - MIGRATED FROM ORIGINAL
    fn parseBreakStatement(self: *StatementParser) common.ParserError!ast.Statements.StmtNode {
        const break_token = self.base.previous();
        var label: ?[]const u8 = null;
        var value: ?*ast.Expressions.ExprNode = null;

        // Check for labeled break (break :label)
        if (self.base.match(.Colon)) {
            const label_token = try self.base.consume(.Identifier, "Expected label after ':' in break statement");
            label = try self.base.arena.createString(label_token.lexeme);
        }

        // Check for break with value (break value or break :label value)
        if (!self.base.check(.Semicolon) and !self.base.isAtEnd()) {
            // Use expression parser for break value
            self.syncSubParsers();
            const expr = try self.expr_parser.parseExpression();
            self.updateFromSubParser(self.expr_parser.base.current);
            const expr_ptr = try self.base.arena.createNode(ast.Expressions.ExprNode);
            expr_ptr.* = expr;
            value = expr_ptr;
        }

        _ = self.base.match(.Semicolon);

        return ast.Statements.StmtNode{ .Break = ast.Statements.BreakNode{
            .label = label,
            .value = value,
            .span = ParserCommon.makeSpan(break_token),
        } };
    }

    /// Parse continue statement - MIGRATED FROM ORIGINAL
    fn parseContinueStatement(self: *StatementParser) common.ParserError!ast.Statements.StmtNode {
        const continue_token = self.base.previous();

        // Check for optional label (continue :label)
        var label: ?[]const u8 = null;
        if (self.base.match(.Colon)) {
            const label_token = try self.base.consume(.Identifier, "Expected label name after ':'");
            label = try self.base.arena.createString(label_token.lexeme);
        }

        _ = self.base.match(.Semicolon);
        return ast.Statements.StmtNode{ .Continue = ast.Statements.ContinueNode{
            .label = label,
            .span = ParserCommon.makeSpan(continue_token),
        } };
    }

    /// Parse try-catch statement
    fn parseTryStatement(self: *StatementParser) common.ParserError!ast.Statements.StmtNode {
        const try_token = self.base.previous();
        const try_block = try self.parseBlock();

        var catch_block: ?ast.Statements.CatchBlock = null;
        if (self.base.match(.Catch)) {
            var error_variable: ?[]const u8 = null;

            // Optional catch variable: catch(e) { ... }
            if (self.base.match(.LeftParen)) {
                const var_token = try self.base.consume(.Identifier, "Expected variable name in catch");
                error_variable = try self.base.arena.createString(var_token.lexeme);
                _ = try self.base.consume(.RightParen, "Expected ')' after catch variable");
            }

            const catch_body = try self.parseBlock();
            catch_block = ast.Statements.CatchBlock{
                .error_variable = error_variable,
                .block = catch_body,
                .span = ParserCommon.makeSpan(self.base.previous()),
            };
        }

        return ast.Statements.StmtNode{ .TryBlock = ast.Statements.TryBlockNode{
            .try_block = try_block,
            .catch_block = catch_block,
            .span = ParserCommon.makeSpan(try_token),
        } };
    }

    // Using common parseSwitchPattern from common_parsers.zig
    // Using common parseSwitchBody from common_parsers.zig

    /// Try to parse a compound assignment statement (a += b, a -= b, etc.)
    fn tryParseCompoundAssignmentStatement(self: *StatementParser) common.ParserError!?ast.statements.CompoundAssignmentNode {
        const saved_pos = self.base.current;

        // We need to parse a potential lvalue first
        self.syncSubParsers();
        if (self.base.isAtEnd()) return null;

        // Parse the left-hand side (must be a valid lvalue)
        const lhs = try self.expr_parser.parseExpression();
        self.updateFromSubParser(self.expr_parser.base.current);

        // Check if the next token is a compound operator
        if (self.base.match(.PlusEqual) or self.base.match(.MinusEqual) or
            self.base.match(.StarEqual) or self.base.match(.SlashEqual) or
            self.base.match(.PercentEqual))
        {
            const op_token = self.base.previous();

            // Map the token type to CompoundAssignmentOp
            const compound_op: ast.Operators.Compound = switch (op_token.type) {
                .PlusEqual => .PlusEqual,
                .MinusEqual => .MinusEqual,
                .StarEqual => .StarEqual,
                .SlashEqual => .SlashEqual,
                .PercentEqual => .PercentEqual,
                else => unreachable, // We already checked these token types
            };

            // Parse the right-hand side
            self.syncSubParsers();
            const rhs = try self.expr_parser.parseExpression();
            self.updateFromSubParser(self.expr_parser.base.current);

            // Expect a semicolon
            _ = try self.base.consume(.Semicolon, "Expected ';' after compound assignment");

            // Create lhs and rhs pointers
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

        // If we didn't find a compound assignment, restore position and return null
        self.base.current = saved_pos;
        return null;
    }
};
