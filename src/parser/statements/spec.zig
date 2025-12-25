// ============================================================================
// Specification Statement Parsers
// ============================================================================
//
// Handles parsing of specification and formal verification statements:
//   • requires (preconditions)
//   • ensures (postconditions)
//   • assert (runtime assertions)
//   • assume (assumptions for verification)
//   • havoc (havoc statements for verification)
//
// ============================================================================

const std = @import("std");
const ast = @import("../../ast.zig");
const common = @import("../common.zig");
const ParserCommon = common.ParserCommon;

// Forward declaration - StatementParser is defined in statement_parser.zig
const StatementParser = @import("../statement_parser.zig").StatementParser;

/// Parse assert statement
pub fn parseAssertStatement(parser: *StatementParser) common.ParserError!ast.Statements.StmtNode {
    const assert_token = parser.base.previous();
    _ = try parser.base.consume(.LeftParen, "Expected '(' after 'assert'");

    // parse the condition expression
    parser.syncSubParsers();
    const condition = try parser.expr_parser.parseExpression();
    parser.updateFromSubParser(parser.expr_parser.base.current);

    // optional message: assert(condition, "message")
    var message: ?[]const u8 = null;
    if (parser.base.match(.Comma)) {
        const msg_token = try parser.base.consume(.StringLiteral, "Expected string message after ',' in assert");
        message = try parser.base.arena.createString(msg_token.lexeme);
    }

    _ = try parser.base.consume(.RightParen, "Expected ')' after assert condition");
    _ = try parser.base.consume(.Semicolon, "Expected ';' after assert statement");

    return ast.Statements.StmtNode{
        .Assert = ast.Statements.AssertNode{
            .condition = condition,
            .message = message,
            .is_ghost = false, // Runtime assertion by default (ghost context handled elsewhere)
            .span = ParserCommon.makeSpan(assert_token),
        },
    };
}

/// Parse requires statement
pub fn parseRequiresStatement(parser: *StatementParser) common.ParserError!ast.Statements.StmtNode {
    const requires_token = parser.base.previous();
    // only requires(condition) is allowed
    _ = try parser.base.consume(.LeftParen, "Expected '(' after 'requires'");
    // use expression parser for requires condition
    parser.syncSubParsers();
    const condition = try parser.expr_parser.parseExpression();
    parser.updateFromSubParser(parser.expr_parser.base.current);
    _ = try parser.base.consume(.RightParen, "Expected ')' after requires condition");
    // strict: disallow trailing semicolon
    if (parser.base.match(.Semicolon)) {
        try parser.base.errorAtCurrent("Unexpected ';' after requires(...) (no semicolon allowed)");
        return error.UnexpectedToken;
    }

    return ast.Statements.StmtNode{ .Requires = ast.Statements.RequiresNode{
        .condition = condition,
        .span = ParserCommon.makeSpan(requires_token),
    } };
}

/// Parse assume statement (formal verification)
pub fn parseAssumeStatement(parser: *StatementParser) common.ParserError!ast.Statements.StmtNode {
    const assume_token = parser.base.previous();
    _ = try parser.base.consume(.LeftParen, "Expected '(' after 'assume'");

    // parse the condition expression
    parser.syncSubParsers();
    const condition = try parser.expr_parser.parseExpression();
    parser.updateFromSubParser(parser.expr_parser.base.current);

    _ = try parser.base.consume(.RightParen, "Expected ')' after assume condition");
    _ = try parser.base.consume(.Semicolon, "Expected ';' after assume statement");

    return ast.Statements.StmtNode{ .Assume = ast.Statements.AssumeNode{
        .condition = condition,
        .span = ParserCommon.makeSpan(assume_token),
    } };
}

/// Parse havoc statement (formal verification)
pub fn parseHavocStatement(parser: *StatementParser) common.ParserError!ast.Statements.StmtNode {
    const havoc_token = parser.base.previous();
    const var_token = try parser.base.consume(.Identifier, "Expected variable name after 'havoc'");
    _ = try parser.base.consume(.Semicolon, "Expected ';' after havoc statement");

    const var_name = try parser.base.arena.createString(var_token.lexeme);

    return ast.Statements.StmtNode{ .Havoc = ast.Statements.HavocNode{
        .variable_name = var_name,
        .span = ParserCommon.makeSpan(havoc_token),
    } };
}

/// Parse ensures statement
pub fn parseEnsuresStatement(parser: *StatementParser) common.ParserError!ast.Statements.StmtNode {
    const ensures_token = parser.base.previous();
    // only ensures(condition) is allowed
    _ = try parser.base.consume(.LeftParen, "Expected '(' after 'ensures'");
    // use expression parser for ensures condition
    parser.syncSubParsers();
    const condition = try parser.expr_parser.parseExpression();
    parser.updateFromSubParser(parser.expr_parser.base.current);
    _ = try parser.base.consume(.RightParen, "Expected ')' after ensures condition");
    // strict: disallow trailing semicolon
    if (parser.base.match(.Semicolon)) {
        try parser.base.errorAtCurrent("Unexpected ';' after ensures(...) (no semicolon allowed)");
        return error.UnexpectedToken;
    }

    return ast.Statements.StmtNode{ .Ensures = ast.Statements.EnsuresNode{
        .condition = condition,
        .span = ParserCommon.makeSpan(ensures_token),
    } };
}
