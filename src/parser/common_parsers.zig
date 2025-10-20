// ============================================================================
// Common Parser Functions
//
// This module contains shared parsing functions used by multiple parser modules
// to avoid code duplication and ensure consistent parsing behavior.
// ============================================================================
const std = @import("std");
const lexer = @import("../lexer.zig");
const ast = @import("../ast.zig");
const TypeInfo = @import("../ast/type_info.zig").TypeInfo;
const common = @import("common.zig");

const Token = lexer.Token;
const TokenType = lexer.TokenType;
const BaseParser = common.BaseParser;
const ParserCommon = common.ParserCommon;
const ParserError = @import("parser_core.zig").ParserError;
const ExpressionParser = @import("expression_parser.zig").ExpressionParser;

/// Parse a switch pattern that can be used in both statements and expressions
pub fn parseSwitchPattern(
    base: *BaseParser,
    expr_parser: *ExpressionParser,
) ParserError!ast.Switch.Pattern {
    // Ensure expression parser starts from the same cursor as base
    expr_parser.base.current = base.current;
    const pattern_token = base.peek();

    // Check for else pattern
    if (base.match(.Else)) {
        return ast.Switch.Pattern{ .Else = base.spanFromToken(pattern_token) };
    }

    // Add support for explicit enum patterns with ranges: EnumName.Start...EnumName.End
    if (base.check(.Identifier)) {
        const saved_pos = base.current;
        _ = base.advance(); // Consume identifier

        if (base.match(.Dot) and base.check(.Identifier)) {
            _ = base.advance(); // Consume variant name

            if (base.match(.DotDotDot)) {
                // This is an enum range pattern: EnumName.Start...EnumName.End
                base.current = saved_pos; // Restore position

                // Use the range expression parser
                const range_expr = try expr_parser.parseRangeExpression();

                if (range_expr == .Range) {
                    return ast.Switch.Pattern{
                        .Range = .{
                            .start = range_expr.Range.start,
                            .end = range_expr.Range.end,
                            .inclusive = range_expr.Range.inclusive,
                            .span = range_expr.Range.span,
                            .type_info = TypeInfo.unknown(), // Type will be inferred
                        },
                    };
                } else {
                    return error.ExpectedRangeExpression;
                }
            } else {
                // Not a range, restore position to after the first token
                base.current = saved_pos;
            }
        } else {
            // Not an enum pattern, restore position
            base.current = saved_pos;
        }
    }

    // General range pattern: parse an expression and if followed by '...'
    // parse another expression to form a Range pattern
    {
        const saved_pos_base = base.current;
        const saved_pos_expr = expr_parser.base.current;
        const start_expr = expr_parser.parseExpression() catch blk: {
            base.current = saved_pos_base;
            expr_parser.base.current = saved_pos_expr;
            break :blk null;
        };
        if (start_expr) |se| {
            // Keep base in sync with expr parser after parsing the start expression
            base.current = expr_parser.base.current;

            if (base.match(.DotDotDot)) {
                // Keep expr parser in sync for parsing the end expression
                expr_parser.base.current = base.current;
                const end_expr = try expr_parser.parseExpression();
                base.current = expr_parser.base.current;
                const start_ptr = try base.arena.createNode(ast.Expressions.ExprNode);
                start_ptr.* = se;
                const end_ptr = try base.arena.createNode(ast.Expressions.ExprNode);
                end_ptr.* = end_expr;
                return ast.Switch.Pattern{ .Range = .{
                    .start = start_ptr,
                    .end = end_ptr,
                    .inclusive = true,
                    .span = base.spanFromToken(pattern_token),
                    .type_info = TypeInfo.unknown(),
                } };
            } else {
                // Not a range; restore and continue with other pattern kinds
                base.current = saved_pos_base;
                expr_parser.base.current = saved_pos_expr;
            }
        }
    }

    // Check for enum value pattern (EnumName.VariantName)
    if (base.check(.Identifier)) {
        const first_token = base.advance();

        if (base.match(.Dot)) {
            const variant_token = try base.consume(.Identifier, "Expected variant name after '.'");

            return ast.Switch.Pattern{ .EnumValue = .{
                .enum_name = first_token.lexeme,
                .variant_name = variant_token.lexeme,
                .span = base.spanFromToken(first_token),
            } };
        } else {
            // Single identifier - parsed as enum variant without explicit enum name
            // Type resolution happens in semantics phase
            return ast.Switch.Pattern{ .EnumValue = .{
                .enum_name = "",
                .variant_name = first_token.lexeme,
                .span = base.spanFromToken(first_token),
            } };
        }
    }

    // Check for literal patterns
    if (base.check(.IntegerLiteral) or base.check(.StringLiteral) or
        base.check(.BinaryLiteral) or base.check(.True) or base.check(.False))
    {
        const literal_token = base.advance();

        const literal_value = switch (literal_token.type) {
            .IntegerLiteral => ast.Expressions.LiteralExpr{
                .Integer = ast.expressions.IntegerLiteral{
                    .value = literal_token.lexeme,
                    .type_info = ast.Types.TypeInfo.fromOraType(.u256), // Default to u256 for integers
                    .span = base.spanFromToken(literal_token),
                },
            },
            .BinaryLiteral => ast.Expressions.LiteralExpr{
                .Binary = ast.Expressions.BinaryLiteral{
                    .value = literal_token.lexeme,
                    .type_info = ast.Types.TypeInfo.fromOraType(.u256), // Default to u256 for binary literals
                    .span = base.spanFromToken(literal_token),
                },
            },
            .StringLiteral => ast.Expressions.LiteralExpr{ .String = ast.expressions.StringLiteral{
                .value = literal_token.lexeme,
                .type_info = ast.Types.TypeInfo.fromOraType(.string),
                .span = base.spanFromToken(literal_token),
            } },
            .True => ast.Expressions.LiteralExpr{ .Bool = ast.expressions.BoolLiteral{
                .value = true,
                .type_info = ast.Types.TypeInfo.fromOraType(.bool),
                .span = base.spanFromToken(literal_token),
            } },
            .False => ast.Expressions.LiteralExpr{ .Bool = ast.expressions.BoolLiteral{
                .value = false,
                .type_info = ast.Types.TypeInfo.fromOraType(.bool),
                .span = base.spanFromToken(literal_token),
            } },
            else => unreachable,
        };

        return ast.Switch.Pattern{ .Literal = .{
            .value = literal_value,
            .span = base.spanFromToken(literal_token),
        } };
    }

    // Default case or other patterns (underscore)
    if (base.check(.Identifier)) {
        const saved_pos = base.current;
        const token = base.advance();

        // Check if the identifier is a single underscore
        if (std.mem.eql(u8, token.lexeme, "_")) {
            return ast.Switch.Pattern{ .Else = base.spanFromToken(token) };
        }

        // Not an underscore, rewind
        base.current = saved_pos;
    }

    return ParserError.UnexpectedToken;
}

/// Parse a switch body that can be used in both statements and expressions
/// Mode controls whether statement arms must end with ';'
pub const SwitchBodyMode = enum { StatementArm, ExpressionArm };

pub fn parseSwitchBody(
    base: *BaseParser,
    expr_parser: *ExpressionParser,
    mode: SwitchBodyMode,
) ParserError!ast.Switch.Body {
    // Ensure expression parser starts from the same cursor as base
    expr_parser.base.current = base.current;

    // Check for a labeled block: Identifier ":" "{" ... "}"
    if (base.check(.Identifier)) {
        const saved_pos = base.current;
        const label_token = base.advance();
        if (base.match(.Colon) and base.check(.LeftBrace)) {
            if (mode == .ExpressionArm) {
                try base.errorAtCurrent("Labeled block not allowed in switch expression arm");
                return ParserError.UnexpectedToken;
            }

            // Parse the block content
            _ = base.advance(); // consume '{'

            var statements = std.ArrayList(ast.Statements.StmtNode){};
            defer statements.deinit(base.arena.allocator());

            while (!base.check(.RightBrace) and !base.isAtEnd()) {
                // Parse statements as expressions for now; optional semicolons
                const e = try expr_parser.parseExpression();
                base.current = expr_parser.base.current;
                try statements.append(base.arena.allocator(), ast.Statements.StmtNode{ .Expr = e });
                _ = base.match(.Semicolon);
            }

            _ = try base.consume(.RightBrace, "Expected '}' after labeled block body");

            const block = ast.Statements.BlockNode{
                .statements = try base.arena.createSlice(ast.Statements.StmtNode, statements.items.len),
                .span = base.spanFromToken(label_token),
            };
            // Copy statements into the allocated slice
            for (statements.items, 0..) |stmt, i| block.statements[i] = stmt;

            return ast.Switch.Body{ .LabeledBlock = .{
                .label = label_token.lexeme,
                .block = block,
                .span = base.spanFromToken(label_token),
            } };
        } else {
            // Not a labeled block; rewind
            base.current = saved_pos;
            expr_parser.base.current = saved_pos;
        }
    }

    // Check for a block body
    if (base.match(.LeftBrace)) {
        if (mode == .ExpressionArm) {
            try base.errorAtCurrent("Block body not allowed in switch expression arm");
            return ParserError.UnexpectedToken;
        }
        var statements = std.ArrayList(ast.Statements.StmtNode){};
        defer statements.deinit(base.arena.allocator());

        // Parse statements inside the block
        while (!base.check(.RightBrace) and !base.isAtEnd()) {
            // If using ExpressionParser, the statement will be an expression statement
            const e = try expr_parser.parseExpression();
            // Keep base and expr parser in sync after expression parse
            base.current = expr_parser.base.current;
            try statements.append(base.arena.allocator(), ast.Statements.StmtNode{ .Expr = e });

            // Consume optional semicolon
            _ = base.match(.Semicolon);
        }

        _ = try base.consume(.RightBrace, "Expected '}' after switch case block");

        return ast.Switch.Body{ .Block = .{
            .statements = try base.arena.createSlice(ast.Statements.StmtNode, statements.items.len),
            .span = base.spanFromToken(base.previous()),
        } };
    }

    // Single expression body
    const expr = try expr_parser.parseExpression();
    // Keep base and expr parser in sync after expression parse
    base.current = expr_parser.base.current;
    if (mode == .StatementArm) {
        _ = try base.consume(.Semicolon, "Expected ';' after switch statement arm expression");
    }
    const expr_ptr = try base.arena.createNode(ast.Expressions.ExprNode);
    expr_ptr.* = expr;
    return ast.Switch.Body{ .Expression = expr_ptr };
}
