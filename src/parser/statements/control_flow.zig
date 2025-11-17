// ============================================================================
// Control Flow Statement Parsers
// ============================================================================
//
// Handles parsing of control flow statements:
//   • if/else
//   • while loops (with invariants, decreases, increases)
//   • for loops (with invariants, decreases, increases)
//   • switch statements
//   • break/continue (with labels and values)
//   • try-catch blocks
//
// ============================================================================

const std = @import("std");
const ast = @import("../../ast.zig");
const common = @import("../common.zig");
const common_parsers = @import("../common_parsers.zig");
const ParserCommon = common.ParserCommon;

// Forward declaration - StatementParser is defined in statement_parser.zig
const StatementParser = @import("../statement_parser.zig").StatementParser;

/// Parse if statement
pub fn parseIfStatement(parser: *StatementParser) common.ParserError!ast.Statements.StmtNode {
    _ = try parser.base.consume(.LeftParen, "Expected '(' after 'if'");

    // Use expression parser for condition
    parser.syncSubParsers();
    const condition = try parser.expr_parser.parseExpression();
    parser.updateFromSubParser(parser.expr_parser.base.current);

    _ = try parser.base.consume(.RightParen, "Expected ')' after if condition");

    const then_branch = try parser.parseBlock();

    var else_branch: ?ast.Statements.BlockNode = null;
    if (parser.base.match(.Else)) {
        else_branch = try parser.parseBlock();
    }

    return ast.Statements.StmtNode{ .If = ast.Statements.IfNode{
        .condition = condition,
        .then_branch = then_branch,
        .else_branch = else_branch,
        .span = parser.base.spanFromToken(parser.base.previous()),
    } };
}

/// Parse while statement
pub fn parseWhileStatement(parser: *StatementParser) common.ParserError!ast.Statements.StmtNode {
    _ = try parser.base.consume(.LeftParen, "Expected '(' after 'while'");

    // Use expression parser for condition
    parser.syncSubParsers();
    const condition = try parser.expr_parser.parseExpression();
    parser.updateFromSubParser(parser.expr_parser.base.current);

    _ = try parser.base.consume(.RightParen, "Expected ')' after while condition");

    // Parse optional loop invariants: invariant(expr) invariant(expr) ...
    var invariants = std.ArrayList(ast.Expressions.ExprNode){};
    defer invariants.deinit(parser.base.arena.allocator());

    while (parser.base.match(.Invariant)) {
        _ = try parser.base.consume(.LeftParen, "Expected '(' after 'invariant'");
        parser.syncSubParsers();
        const inv_expr = try parser.expr_parser.parseExpression();
        parser.updateFromSubParser(parser.expr_parser.base.current);
        _ = try parser.base.consume(.RightParen, "Expected ')' after invariant expression");
        try invariants.append(parser.base.arena.allocator(), inv_expr);
    }

    // Parse optional decreases clause
    var decreases_expr: ?*ast.Expressions.ExprNode = null;
    if (parser.base.match(.Decreases)) {
        _ = try parser.base.consume(.LeftParen, "Expected '(' after 'decreases'");
        parser.syncSubParsers();
        const dec_expr = try parser.expr_parser.parseExpression();
        parser.updateFromSubParser(parser.expr_parser.base.current);
        _ = try parser.base.consume(.RightParen, "Expected ')' after decreases expression");
        const dec_ptr = try parser.base.arena.createNode(ast.Expressions.ExprNode);
        dec_ptr.* = dec_expr;
        decreases_expr = dec_ptr;
    }

    // Parse optional increases clause
    var increases_expr: ?*ast.Expressions.ExprNode = null;
    if (parser.base.match(.Increases)) {
        _ = try parser.base.consume(.LeftParen, "Expected '(' after 'increases'");
        parser.syncSubParsers();
        const inc_expr = try parser.expr_parser.parseExpression();
        parser.updateFromSubParser(parser.expr_parser.base.current);
        _ = try parser.base.consume(.RightParen, "Expected ')' after increases expression");
        const inc_ptr = try parser.base.arena.createNode(ast.Expressions.ExprNode);
        inc_ptr.* = inc_expr;
        increases_expr = inc_ptr;
    }

    const body = try parser.parseBlock();

    return ast.Statements.StmtNode{ .While = ast.Statements.WhileNode{
        .condition = condition,
        .body = body,
        .invariants = try invariants.toOwnedSlice(parser.base.arena.allocator()),
        .decreases = decreases_expr,
        .increases = increases_expr,
        .span = parser.base.spanFromToken(parser.base.previous()),
    } };
}

/// Parse for statement (Zig-style: for (expr) |var1, var2| stmt)
pub fn parseForStatement(parser: *StatementParser) common.ParserError!ast.Statements.StmtNode {
    const for_token = parser.base.previous();

    _ = try parser.base.consume(.LeftParen, "Expected '(' after 'for'");

    // Use expression parser for iterable
    parser.syncSubParsers();
    const iterable = try parser.expr_parser.parseExpression();
    parser.updateFromSubParser(parser.expr_parser.base.current);

    _ = try parser.base.consume(.RightParen, "Expected ')' after for expression");

    _ = try parser.base.consume(.Pipe, "Expected '|' after for expression");
    const var1_token = try parser.base.consume(.Identifier, "Expected loop variable");
    const var1 = try parser.base.arena.createString(var1_token.lexeme);

    var var2: ?[]const u8 = null;
    if (parser.base.match(.Comma)) {
        const var2_token = try parser.base.consume(.Identifier, "Expected second loop variable");
        var2 = try parser.base.arena.createString(var2_token.lexeme);
    }

    _ = try parser.base.consume(.Pipe, "Expected '|' after loop variables");

    // Parse optional loop invariants: invariant(expr) invariant(expr) ...
    var invariants = std.ArrayList(ast.Expressions.ExprNode){};
    defer invariants.deinit(parser.base.arena.allocator());

    while (parser.base.match(.Invariant)) {
        _ = try parser.base.consume(.LeftParen, "Expected '(' after 'invariant'");
        parser.syncSubParsers();
        const inv_expr = try parser.expr_parser.parseExpression();
        parser.updateFromSubParser(parser.expr_parser.base.current);
        _ = try parser.base.consume(.RightParen, "Expected ')' after invariant expression");
        try invariants.append(parser.base.arena.allocator(), inv_expr);
    }

    // Parse optional decreases clause
    var decreases_expr: ?*ast.Expressions.ExprNode = null;
    if (parser.base.match(.Decreases)) {
        _ = try parser.base.consume(.LeftParen, "Expected '(' after 'decreases'");
        parser.syncSubParsers();
        const dec_expr = try parser.expr_parser.parseExpression();
        parser.updateFromSubParser(parser.expr_parser.base.current);
        _ = try parser.base.consume(.RightParen, "Expected ')' after decreases expression");
        const dec_ptr = try parser.base.arena.createNode(ast.Expressions.ExprNode);
        dec_ptr.* = dec_expr;
        decreases_expr = dec_ptr;
    }

    // Parse optional increases clause
    var increases_expr: ?*ast.Expressions.ExprNode = null;
    if (parser.base.match(.Increases)) {
        _ = try parser.base.consume(.LeftParen, "Expected '(' after 'increases'");
        parser.syncSubParsers();
        const inc_expr = try parser.expr_parser.parseExpression();
        parser.updateFromSubParser(parser.expr_parser.base.current);
        _ = try parser.base.consume(.RightParen, "Expected ')' after increases expression");
        const inc_ptr = try parser.base.arena.createNode(ast.Expressions.ExprNode);
        inc_ptr.* = inc_expr;
        increases_expr = inc_ptr;
    }

    const body = try parser.parseBlock();

    const pattern = if (var2) |v2|
        ast.Statements.LoopPattern{ .IndexPair = .{ .item = var1, .index = v2, .span = parser.base.spanFromToken(var1_token) } }
    else
        ast.Statements.LoopPattern{ .Single = .{ .name = var1, .span = parser.base.spanFromToken(var1_token) } };

    return ast.Statements.StmtNode{
        .ForLoop = ast.Statements.ForLoopNode{
            .iterable = iterable,
            .pattern = pattern,
            .body = body,
            .invariants = try invariants.toOwnedSlice(parser.base.arena.allocator()),
            .span = parser.base.spanFromToken(for_token),
        },
    };
}

/// Parse switch statement
pub fn parseSwitchStatement(parser: *StatementParser) common.ParserError!ast.Statements.StmtNode {
    const switch_token = parser.base.previous();

    // Parse required switch condition: switch (expr)
    _ = try parser.base.consume(.LeftParen, "Expected '(' after 'switch'");

    // Use expression parser for condition
    parser.syncSubParsers();
    const condition = try parser.expr_parser.parseExpression();
    parser.updateFromSubParser(parser.expr_parser.base.current);

    _ = try parser.base.consume(.RightParen, "Expected ')' after switch condition");

    _ = try parser.base.consume(.LeftBrace, "Expected '{' after switch condition");

    var cases = std.ArrayList(ast.Switch.Case){};
    defer cases.deinit(parser.base.arena.allocator());

    var default_case: ?ast.Statements.BlockNode = null;

    // Parse switch arms
    while (!parser.base.check(.RightBrace) and !parser.base.isAtEnd()) {
        if (parser.base.match(.Else)) {
            // Parse else clause
            _ = try parser.base.consume(.Arrow, "Expected '=>' after 'else'");

            // Handle labeled block: Identifier ':' '{' ... '}'
            if (parser.base.check(.Identifier)) {
                const cur = parser.base.current;
                if (cur + 2 < parser.base.tokens.len and
                    parser.base.tokens[cur + 1].type == .Colon and
                    parser.base.tokens[cur + 2].type == .LeftBrace)
                {
                    _ = parser.base.advance(); // consume Identifier label
                    _ = parser.base.advance(); // ':'
                    const block = try parser.parseBlock();
                    default_case = block;
                    break;
                }
            }

            // Handle plain block body
            if (parser.base.check(.LeftBrace)) {
                const block = try parser.parseBlock();
                default_case = block;
                break;
            }

            // Fallback: parse as expression arm and wrap into a block
            const else_body = try common_parsers.parseSwitchBody(&parser.base, &parser.expr_parser, .StatementArm);
            switch (else_body) {
                .Block => |block| {
                    default_case = block;
                },
                .LabeledBlock => |labeled| {
                    default_case = labeled.block;
                },
                .Expression => |expr_ptr| {
                    var stmts = try parser.base.arena.createSlice(ast.Statements.StmtNode, 1);
                    stmts[0] = ast.Statements.StmtNode{ .Expr = expr_ptr.* };
                    default_case = ast.Statements.BlockNode{
                        .statements = stmts,
                        .span = ParserCommon.makeSpan(parser.base.previous()),
                    };
                },
            }
            break;
        }

        // Parse switch pattern using common parser
        const pattern = try common_parsers.parseSwitchPattern(&parser.base, &parser.expr_parser);
        _ = try parser.base.consume(.Arrow, "Expected '=>' after switch pattern");

        // Parse switch body: handle labeled blocks, plain blocks, or expression arms
        const body = blk: {
            // Labeled block detection: Identifier ':' '{'
            if (parser.base.check(.Identifier)) {
                const cur = parser.base.current;
                if (cur + 2 < parser.base.tokens.len and
                    parser.base.tokens[cur + 1].type == .Colon and
                    parser.base.tokens[cur + 2].type == .LeftBrace)
                {
                    const label_tok = parser.base.advance(); // Identifier
                    _ = parser.base.advance(); // ':'
                    const block = try parser.parseBlock();
                    break :blk ast.Switch.Body{ .LabeledBlock = .{
                        .label = label_tok.lexeme,
                        .block = block,
                        .span = ParserCommon.makeSpan(label_tok),
                    } };
                }
            }
            if (parser.base.check(.LeftBrace)) {
                const block = try parser.parseBlock();
                break :blk ast.Switch.Body{ .Block = block };
            }
            // Otherwise parse as an expression arm requiring ';'
            const b = try common_parsers.parseSwitchBody(&parser.base, &parser.expr_parser, .StatementArm);
            break :blk b;
        };

        const case = ast.Switch.Case{
            .pattern = pattern,
            .body = body,
            .span = ParserCommon.makeSpan(parser.base.previous()),
        };

        try cases.append(parser.base.arena.allocator(), case);

        // Optional comma between cases
        _ = parser.base.match(.Comma);
    }

    _ = try parser.base.consume(.RightBrace, "Expected '}' after switch cases");

    return ast.Statements.StmtNode{ .Switch = ast.Statements.SwitchNode{
        .condition = condition,
        .cases = try cases.toOwnedSlice(parser.base.arena.allocator()),
        .default_case = default_case,
        .span = parser.base.spanFromToken(switch_token),
    } };
}

/// Parse break statement
pub fn parseBreakStatement(parser: *StatementParser) common.ParserError!ast.Statements.StmtNode {
    const break_token = parser.base.previous();
    var label: ?[]const u8 = null;
    var value: ?*ast.Expressions.ExprNode = null;

    // Check for labeled break (break :label)
    if (parser.base.match(.Colon)) {
        const label_token = try parser.base.consume(.Identifier, "Expected label after ':' in break statement");
        label = try parser.base.arena.createString(label_token.lexeme);
    }

    // Check for break with value (break value or break :label value)
    if (!parser.base.check(.Semicolon) and !parser.base.isAtEnd()) {
        // Use expression parser for break value
        parser.syncSubParsers();
        const expr = try parser.expr_parser.parseExpression();
        parser.updateFromSubParser(parser.expr_parser.base.current);
        const expr_ptr = try parser.base.arena.createNode(ast.Expressions.ExprNode);
        expr_ptr.* = expr;
        value = expr_ptr;
    }

    _ = parser.base.match(.Semicolon);

    return ast.Statements.StmtNode{ .Break = ast.Statements.BreakNode{
        .label = label,
        .value = value,
        .span = ParserCommon.makeSpan(break_token),
    } };
}

/// Parse continue statement
pub fn parseContinueStatement(parser: *StatementParser) common.ParserError!ast.Statements.StmtNode {
    const continue_token = parser.base.previous();
    // Syntax: continue [:label] [value]? ;
    // If labeled, optional replacement operand expression (value) is allowed.

    // Optional label
    var label: ?[]const u8 = null;
    if (parser.base.match(.Colon)) {
        const label_token = try parser.base.consume(.Identifier, "Expected label name after ':'");
        label = try parser.base.arena.createString(label_token.lexeme);
    }

    // Optional value expression before semicolon
    var value: ?*ast.Expressions.ExprNode = null;
    if (!parser.base.check(.Semicolon) and !parser.base.check(.RightBrace)) {
        parser.syncSubParsers();
        const expr = try parser.expr_parser.parseExpression();
        parser.updateFromSubParser(parser.expr_parser.base.current);
        const expr_ptr = try parser.base.arena.createNode(ast.Expressions.ExprNode);
        expr_ptr.* = expr;
        value = expr_ptr;
    }

    _ = parser.base.match(.Semicolon);
    return ast.Statements.StmtNode{ .Continue = ast.Statements.ContinueNode{
        .label = label,
        .value = value,
        .span = ParserCommon.makeSpan(continue_token),
    } };
}

/// Parse try-catch statement
pub fn parseTryStatement(parser: *StatementParser) common.ParserError!ast.Statements.StmtNode {
    const try_token = parser.base.previous();
    const try_block = try parser.parseBlock();

    var catch_block: ?ast.Statements.CatchBlock = null;
    if (parser.base.match(.Catch)) {
        var error_variable: ?[]const u8 = null;

        // Optional catch variable: catch(e) { ... }
        if (parser.base.match(.LeftParen)) {
            const var_token = try parser.base.consumeIdentifierOrKeyword("Expected variable name in catch");
            error_variable = try parser.base.arena.createString(var_token.lexeme);
            _ = try parser.base.consume(.RightParen, "Expected ')' after catch variable");
        }

        const catch_body = try parser.parseBlock();
        catch_block = ast.Statements.CatchBlock{
            .error_variable = error_variable,
            .block = catch_body,
            .span = ParserCommon.makeSpan(parser.base.previous()),
        };
    }

    return ast.Statements.StmtNode{ .TryBlock = ast.Statements.TryBlockNode{
        .try_block = try_block,
        .catch_block = catch_block,
        .span = ParserCommon.makeSpan(try_token),
    } };
}
