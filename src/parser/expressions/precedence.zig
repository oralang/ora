// ============================================================================
// Expression Precedence Chain
// ============================================================================
//
// Implements the precedence climbing algorithm for binary operators.
// Handles all precedence levels from assignment (lowest) to unary (highest).
//
// PRECEDENCE LEVELS (low to high):
//   Assignment → Logical OR/AND → Bitwise OR/XOR/AND →
//   Equality → Comparison → Shifts → Add/Sub → Mul/Div/Mod →
//   Exponentiation → Unary
//
// ============================================================================

const ast = @import("../../ast.zig");
const common = @import("../common.zig");
const ParserCommon = common.ParserCommon;
const ParserError = @import("../parser_core.zig").ParserError;

// Forward declaration - ExpressionParser is defined in expression_parser.zig
const ExpressionParser = @import("../expression_parser.zig").ExpressionParser;

/// Parse assignment expressions (precedence 14)
pub fn parseAssignment(parser: *ExpressionParser) ParserError!ast.Expressions.ExprNode {
    const expr = try parseLogicalOr(parser);

    // simple assignment
    if (parser.base.match(.Equal)) {
        // validate that the left side is a valid L-value
        ast.expressions.validateLValue(&expr) catch |err| {
            const error_msg = switch (err) {
                ast.expressions.LValueError.LiteralNotAssignable => "Cannot assign to literal value",
                ast.expressions.LValueError.CallNotAssignable => "Cannot assign to function call result",
                ast.expressions.LValueError.BinaryExprNotAssignable => "Cannot assign to binary expression",
                ast.expressions.LValueError.UnaryExprNotAssignable => "Cannot assign to unary expression",
                else => "Invalid assignment target",
            };
            try parser.base.errorAtCurrent(error_msg);
            return error.UnexpectedToken;
        };

        const value = try parseAssignment(parser); // Right-associative
        const expr_ptr = try parser.base.arena.createNode(ast.Expressions.ExprNode);
        expr_ptr.* = expr;
        const value_ptr = try parser.base.arena.createNode(ast.Expressions.ExprNode);
        value_ptr.* = value;

        return ast.Expressions.ExprNode{ .Assignment = ast.Expressions.AssignmentExpr{
            .target = expr_ptr,
            .value = value_ptr,
            .span = parser.base.spanFromToken(parser.base.previous()),
        } };
    }

    // compound assignments
    if (parser.base.match(.PlusEqual) or parser.base.match(.MinusEqual) or parser.base.match(.StarEqual) or
        parser.base.match(.SlashEqual) or parser.base.match(.PercentEqual))
    {
        const op_token = parser.base.previous();

        // validate that the left side is a valid L-value
        ast.expressions.validateLValue(&expr) catch |err| {
            const error_msg = switch (err) {
                ast.expressions.LValueError.LiteralNotAssignable => "Cannot assign to literal value",
                ast.expressions.LValueError.CallNotAssignable => "Cannot assign to function call result",
                ast.expressions.LValueError.BinaryExprNotAssignable => "Cannot assign to binary expression",
                ast.expressions.LValueError.UnaryExprNotAssignable => "Cannot assign to unary expression",
                else => "Invalid assignment target",
            };
            try parser.base.errorAtCurrent(error_msg);
            return error.UnexpectedToken;
        };

        const value = try parseAssignment(parser); // Right-associative
        const expr_ptr = try parser.base.arena.createNode(ast.Expressions.ExprNode);
        expr_ptr.* = expr;
        const value_ptr = try parser.base.arena.createNode(ast.Expressions.ExprNode);
        value_ptr.* = value;

        const compound_op: ast.Operators.Compound = switch (op_token.type) {
            .PlusEqual => .PlusEqual,
            .MinusEqual => .MinusEqual,
            .StarEqual => .StarEqual,
            .SlashEqual => .SlashEqual,
            .PercentEqual => .PercentEqual,
            else => unreachable,
        };

        return ast.Expressions.ExprNode{ .CompoundAssignment = ast.Expressions.CompoundAssignmentExpr{
            .target = expr_ptr,
            .operator = compound_op,
            .value = value_ptr,
            .span = parser.base.spanFromToken(op_token),
        } };
    }

    return expr;
}

/// Parse logical OR expressions (precedence 13)
pub fn parseLogicalOr(parser: *ExpressionParser) ParserError!ast.Expressions.ExprNode {
    var expr = try parseLogicalAnd(parser);

    while (parser.base.match(.PipePipe)) {
        const op_token = parser.base.previous();
        const right = try parseLogicalAnd(parser);

        const left_ptr = try parser.base.arena.createNode(ast.Expressions.ExprNode);
        left_ptr.* = expr;
        const right_ptr = try parser.base.arena.createNode(ast.Expressions.ExprNode);
        right_ptr.* = right;

        expr = ast.Expressions.ExprNode{
            .Binary = ast.Expressions.BinaryExpr{
                .lhs = left_ptr,
                .operator = .Or, // Logical OR
                .rhs = right_ptr,
                .type_info = ast.Types.CommonTypes.bool_type(), // Logical operations return bool
                .span = parser.base.spanFromToken(op_token),
            },
        };
    }

    return expr;
}

/// Parse logical AND expressions (precedence 12)
pub fn parseLogicalAnd(parser: *ExpressionParser) ParserError!ast.Expressions.ExprNode {
    var expr = try parseBitwiseOr(parser);

    while (parser.base.match(.AmpersandAmpersand)) {
        const op_token = parser.base.previous();
        const right = try parseBitwiseOr(parser);

        const left_ptr = try parser.base.arena.createNode(ast.Expressions.ExprNode);
        left_ptr.* = expr;
        const right_ptr = try parser.base.arena.createNode(ast.Expressions.ExprNode);
        right_ptr.* = right;

        expr = ast.Expressions.ExprNode{
            .Binary = ast.Expressions.BinaryExpr{
                .lhs = left_ptr,
                .operator = .And, // Logical AND
                .rhs = right_ptr,
                .type_info = ast.Types.CommonTypes.bool_type(), // Logical operations return bool
                .span = parser.base.spanFromToken(op_token),
            },
        };
    }

    return expr;
}

/// Parse bitwise OR expressions (precedence 11)
pub fn parseBitwiseOr(parser: *ExpressionParser) ParserError!ast.Expressions.ExprNode {
    var expr = try parseBitwiseXor(parser);

    while (parser.base.match(.Pipe)) {
        const op_token = parser.base.previous();
        const right = try parseBitwiseXor(parser);

        const left_ptr = try parser.base.arena.createNode(ast.Expressions.ExprNode);
        left_ptr.* = expr;
        const right_ptr = try parser.base.arena.createNode(ast.Expressions.ExprNode);
        right_ptr.* = right;

        expr = ast.Expressions.ExprNode{
            .Binary = ast.Expressions.BinaryExpr{
                .lhs = left_ptr,
                .operator = .BitwiseOr,
                .rhs = right_ptr,
                .type_info = ast.Types.TypeInfo.unknown(), // Type will be inferred from operands
                .span = parser.base.spanFromToken(op_token),
            },
        };
    }

    return expr;
}

/// Parse bitwise XOR expressions (precedence 10)
pub fn parseBitwiseXor(parser: *ExpressionParser) ParserError!ast.Expressions.ExprNode {
    var expr = try parseBitwiseAnd(parser);

    while (parser.base.match(.Caret)) {
        const op_token = parser.base.previous();
        const right = try parseBitwiseAnd(parser);

        const left_ptr = try parser.base.arena.createNode(ast.Expressions.ExprNode);
        left_ptr.* = expr;
        const right_ptr = try parser.base.arena.createNode(ast.Expressions.ExprNode);
        right_ptr.* = right;

        expr = ast.Expressions.ExprNode{
            .Binary = ast.Expressions.BinaryExpr{
                .lhs = left_ptr,
                .operator = .BitwiseXor,
                .rhs = right_ptr,
                .type_info = ast.Types.TypeInfo.unknown(), // Type will be inferred from operands
                .span = parser.base.spanFromToken(op_token),
            },
        };
    }

    return expr;
}

/// Parse bitwise AND expressions (precedence 9)
pub fn parseBitwiseAnd(parser: *ExpressionParser) ParserError!ast.Expressions.ExprNode {
    var expr = try parseEquality(parser);

    while (parser.base.match(.Ampersand)) {
        const op_token = parser.base.previous();
        const right = try parseEquality(parser);

        const left_ptr = try parser.base.arena.createNode(ast.Expressions.ExprNode);
        left_ptr.* = expr;
        const right_ptr = try parser.base.arena.createNode(ast.Expressions.ExprNode);
        right_ptr.* = right;

        expr = ast.Expressions.ExprNode{
            .Binary = ast.Expressions.BinaryExpr{
                .lhs = left_ptr,
                .operator = .BitwiseAnd,
                .rhs = right_ptr,
                .type_info = ast.Types.TypeInfo.unknown(), // Type will be inferred from operands
                .span = parser.base.spanFromToken(op_token),
            },
        };
    }

    return expr;
}

/// Parse equality expressions (precedence 8)
pub fn parseEquality(parser: *ExpressionParser) ParserError!ast.Expressions.ExprNode {
    var expr = try parseComparison(parser);

    while (parser.base.match(.EqualEqual) or parser.base.match(.BangEqual)) {
        const op_token = parser.base.previous();
        const right = try parseComparison(parser);

        const operator: ast.Operators.Binary = switch (op_token.type) {
            .EqualEqual => .EqualEqual,
            .BangEqual => .BangEqual,
            else => unreachable,
        };

        const left_ptr = try parser.base.arena.createNode(ast.Expressions.ExprNode);
        left_ptr.* = expr;
        const right_ptr = try parser.base.arena.createNode(ast.Expressions.ExprNode);
        right_ptr.* = right;

        expr = ast.Expressions.ExprNode{
            .Binary = ast.Expressions.BinaryExpr{
                .lhs = left_ptr,
                .operator = operator,
                .rhs = right_ptr,
                .type_info = ast.Types.CommonTypes.bool_type(), // Equality operations return bool
                .span = parser.base.spanFromToken(op_token),
            },
        };
    }

    return expr;
}

/// Parse comparison expressions (precedence 7)
pub fn parseComparison(parser: *ExpressionParser) ParserError!ast.Expressions.ExprNode {
    var expr = try parseBitwiseShift(parser);

    while (parser.base.match(.Less) or parser.base.match(.LessEqual) or parser.base.match(.Greater) or parser.base.match(.GreaterEqual)) {
        const op_token = parser.base.previous();
        const right = try parseBitwiseShift(parser);

        const operator: ast.Operators.Binary = switch (op_token.type) {
            .Less => .Less,
            .LessEqual => .LessEqual,
            .Greater => .Greater,
            .GreaterEqual => .GreaterEqual,
            else => unreachable,
        };

        const left_ptr = try parser.base.arena.createNode(ast.Expressions.ExprNode);
        left_ptr.* = expr;
        const right_ptr = try parser.base.arena.createNode(ast.Expressions.ExprNode);
        right_ptr.* = right;

        expr = ast.Expressions.ExprNode{
            .Binary = ast.Expressions.BinaryExpr{
                .lhs = left_ptr,
                .operator = operator,
                .rhs = right_ptr,
                .type_info = ast.Types.CommonTypes.bool_type(), // Comparison operations return bool
                .span = parser.base.spanFromToken(op_token),
            },
        };
    }

    return expr;
}

/// Parse bitwise shift expressions (precedence 6)
pub fn parseBitwiseShift(parser: *ExpressionParser) ParserError!ast.Expressions.ExprNode {
    var expr = try parseTerm(parser);

    while (parser.base.match(.LessLess) or parser.base.match(.GreaterGreater) or
        parser.base.match(.LessLessPercent) or parser.base.match(.GreaterGreaterPercent))
    {
        const op_token = parser.base.previous();
        const right = try parseTerm(parser);

        const operator: ast.Operators.Binary = switch (op_token.type) {
            .LessLess => .LeftShift,
            .GreaterGreater => .RightShift,
            .LessLessPercent => .WrappingShl,
            .GreaterGreaterPercent => .WrappingShr,
            else => unreachable,
        };

        const left_ptr = try parser.base.arena.createNode(ast.Expressions.ExprNode);
        left_ptr.* = expr;
        const right_ptr = try parser.base.arena.createNode(ast.Expressions.ExprNode);
        right_ptr.* = right;

        expr = ast.Expressions.ExprNode{
            .Binary = ast.Expressions.BinaryExpr{
                .lhs = left_ptr,
                .operator = operator,
                .rhs = right_ptr,
                .type_info = ast.Types.TypeInfo.unknown(), // Type will be inferred from left operand
                .span = ParserCommon.makeSpan(op_token),
            },
        };
    }

    return expr;
}

/// Parse term expressions (precedence 5: + -)
pub fn parseTerm(parser: *ExpressionParser) ParserError!ast.Expressions.ExprNode {
    var expr = try parseFactor(parser);

    while (parser.base.match(.Plus) or parser.base.match(.Minus) or
        parser.base.match(.PlusPercent) or parser.base.match(.MinusPercent))
    {
        const op_token = parser.base.previous();
        const right = try parseFactor(parser);

        const operator: ast.Operators.Binary = switch (op_token.type) {
            .Plus => .Plus,
            .Minus => .Minus,
            .PlusPercent => .WrappingAdd,
            .MinusPercent => .WrappingSub,
            else => unreachable,
        };

        const left_ptr = try parser.base.arena.createNode(ast.Expressions.ExprNode);
        left_ptr.* = expr;
        const right_ptr = try parser.base.arena.createNode(ast.Expressions.ExprNode);
        right_ptr.* = right;

        expr = ast.Expressions.ExprNode{
            .Binary = ast.Expressions.BinaryExpr{
                .lhs = left_ptr,
                .operator = operator,
                .rhs = right_ptr,
                .type_info = ast.Types.TypeInfo.unknown(), // Type will be inferred from operands
                .span = ParserCommon.makeSpan(op_token),
            },
        };
    }

    return expr;
}

/// Parse factor expressions (precedence 4: * / %)
pub fn parseFactor(parser: *ExpressionParser) ParserError!ast.Expressions.ExprNode {
    var expr = try parseExponent(parser);

    while (parser.base.match(.Star) or parser.base.match(.Slash) or parser.base.match(.Percent) or
        parser.base.match(.StarPercent))
    {
        const op_token = parser.base.previous();
        const right = try parseExponent(parser);

        const operator: ast.Operators.Binary = switch (op_token.type) {
            .Star => .Star,
            .Slash => .Slash,
            .Percent => .Percent,
            .StarPercent => .WrappingMul,
            else => unreachable,
        };

        const left_ptr = try parser.base.arena.createNode(ast.Expressions.ExprNode);
        left_ptr.* = expr;
        const right_ptr = try parser.base.arena.createNode(ast.Expressions.ExprNode);
        right_ptr.* = right;

        expr = ast.Expressions.ExprNode{
            .Binary = ast.Expressions.BinaryExpr{
                .lhs = left_ptr,
                .operator = operator,
                .rhs = right_ptr,
                .type_info = ast.Types.TypeInfo.unknown(), // Type will be inferred from operands
                .span = ParserCommon.makeSpan(op_token),
            },
        };
    }

    return expr;
}

/// Parse exponentiation expressions (precedence 3: **, **%)
pub fn parseExponent(parser: *ExpressionParser) ParserError!ast.Expressions.ExprNode {
    var expr = try parseUnary(parser);

    while (parser.base.match(.StarStar) or parser.base.match(.StarStarPercent)) {
        const op_token = parser.base.previous();
        const right = try parseExponent(parser); // Right-associative

        const operator: ast.Operators.Binary = switch (op_token.type) {
            .StarStar => .StarStar,
            .StarStarPercent => .WrappingPow,
            else => unreachable,
        };

        const left_ptr = try parser.base.arena.createNode(ast.Expressions.ExprNode);
        left_ptr.* = expr;
        const right_ptr = try parser.base.arena.createNode(ast.Expressions.ExprNode);
        right_ptr.* = right;

        expr = ast.Expressions.ExprNode{
            .Binary = ast.Expressions.BinaryExpr{
                .lhs = left_ptr,
                .operator = operator,
                .rhs = right_ptr,
                .type_info = ast.Types.TypeInfo.unknown(), // Type will be inferred from operands
                .span = ParserCommon.makeSpan(op_token),
            },
        };
    }

    return expr;
}

/// Parse unary expressions (precedence 2: ! -)
/// Note: This calls parseCall from the main expression parser
pub fn parseUnary(parser: *ExpressionParser) ParserError!ast.Expressions.ExprNode {
    if (parser.base.match(.Bang) or parser.base.match(.Minus)) {
        const op_token = parser.base.previous();
        const right = try parseUnary(parser); // Right-associative for unary operators

        const operator: ast.Operators.Unary = switch (op_token.type) {
            .Bang => .Bang,
            .Minus => .Minus,
            else => unreachable,
        };

        const right_ptr = try parser.base.arena.createNode(ast.Expressions.ExprNode);
        right_ptr.* = right;

        return ast.Expressions.ExprNode{
            .Unary = ast.Expressions.UnaryExpr{
                .operator = operator,
                .operand = right_ptr,
                .type_info = ast.Types.TypeInfo.unknown(), // Type will be inferred from operand
                .span = ParserCommon.makeSpan(op_token),
            },
        };
    }

    // delegate to parseCall in the main expression parser
    return parser.parseCall();
}
