// ============================================================================
// Constant Evaluation
// ============================================================================
// Evaluates constant expressions during type resolution for:
// - Compile-time validation of literals against refinement constraints
// - Guard optimization (skip guards for constants that satisfy constraints)
// - Type checking decisions based on constant values
// ============================================================================

const std = @import("std");
const ast = @import("../../../ast.zig");

/// Result of constant evaluation
pub const ConstantValue = union(enum) {
    Integer: u256,
    NotConstant, // Expression is not a compile-time constant
    Error, // Evaluation error (e.g., overflow, division by zero)
};

/// Evaluate a constant expression to a u256 value
/// Returns null if the expression is not a compile-time constant
pub fn evaluateConstantExpression(
    allocator: std.mem.Allocator,
    expr: *ast.Expressions.ExprNode,
) !ConstantValue {
    _ = allocator; // Reserved for future use (e.g., error messages)

    return switch (expr.*) {
        .Literal => |*lit| evaluateLiteral(lit),
        .Binary => |*bin| evaluateBinary(bin),
        .Unary => |*unary| evaluateUnary(unary),
        else => ConstantValue.NotConstant,
    };
}

/// Extract integer value from a literal expression
/// Returns null if not an integer literal or if parsing fails
pub fn extractIntegerValue(literal: *const ast.Expressions.LiteralExpr) ?u256 {
    return switch (literal.*) {
        .Integer => |int_lit| extractIntegerLiteralValue(int_lit),
        else => null,
    };
}

/// Extract integer value from an IntegerLiteral
fn extractIntegerLiteralValue(int_lit: ast.Expressions.IntegerLiteral) ?u256 {
    // Parse the string value to u256
    return std.fmt.parseInt(u256, int_lit.value, 0) catch null;
}

/// Evaluate a literal expression
fn evaluateLiteral(lit: *ast.Expressions.LiteralExpr) ConstantValue {
    return switch (lit.*) {
        .Integer => |int_lit| {
            if (extractIntegerLiteralValue(int_lit)) |value| {
                return ConstantValue{ .Integer = value };
            } else {
                return ConstantValue.Error;
            }
        },
        else => ConstantValue.NotConstant, // Only integer literals are constants for now
    };
}

/// Evaluate a binary expression
fn evaluateBinary(bin: *ast.Expressions.BinaryExpr) ConstantValue {
    // Recursively evaluate left and right operands
    const lhs_result = evaluateConstantExpressionRecursive(bin.lhs) catch return ConstantValue.Error;
    const rhs_result = evaluateConstantExpressionRecursive(bin.rhs) catch return ConstantValue.Error;

    // Both operands must be constants
    const lhs_val = switch (lhs_result) {
        .Integer => |v| v,
        else => return ConstantValue.NotConstant,
    };
    const rhs_val = switch (rhs_result) {
        .Integer => |v| v,
        else => return ConstantValue.NotConstant,
    };

    // Evaluate based on operator
    return switch (bin.operator) {
        // Arithmetic operations
        .Plus => arithmeticAdd(lhs_val, rhs_val),
        .Minus => arithmeticSub(lhs_val, rhs_val),
        .Star => arithmeticMul(lhs_val, rhs_val),
        .Slash => arithmeticDiv(lhs_val, rhs_val),
        .Percent => arithmeticMod(lhs_val, rhs_val),
        .StarStar => arithmeticPow(lhs_val, rhs_val),

        // Comparison operations (return boolean, but we only support integers for now)
        .EqualEqual,
        .BangEqual,
        .Less,
        .LessEqual,
        .Greater,
        .GreaterEqual,
        .And,
        .Or, // Logical operators
        .BitwiseAnd,
        .BitwiseOr,
        .BitwiseXor,
        .LeftShift,
        .RightShift, // Bitwise operators
        .Comma,
        => ConstantValue.NotConstant, // Not constant expressions

        // Note: We only evaluate arithmetic operations as constants
        // Comparison and logical operations are runtime checks
    };
}

/// Evaluate a unary expression
fn evaluateUnary(unary: *ast.Expressions.UnaryExpr) ConstantValue {
    // Recursively evaluate operand
    const operand_result = evaluateConstantExpressionRecursive(unary.operand) catch return ConstantValue.Error;

    const operand_val = switch (operand_result) {
        .Integer => |v| v,
        else => return ConstantValue.NotConstant,
    };

    return switch (unary.operator) {
        .Minus => {
            // Unary minus: for unsigned types, this is a compile-time error or runtime behavior
            // In Ora, unary minus on u256 would wrap around (2's complement)
            // For constant evaluation, we can compute the wrapped value
            const result = @as(u256, 0) -% operand_val; // Wrapping subtraction
            return ConstantValue{ .Integer = result };
        },
        .Bang, .BitNot => ConstantValue.NotConstant, // Logical/bitwise not - not constant for our purposes
    };
}

/// Recursive helper that doesn't require allocator
fn evaluateConstantExpressionRecursive(expr: *ast.Expressions.ExprNode) !ConstantValue {
    return switch (expr.*) {
        .Literal => |*lit| evaluateLiteral(lit),
        .Binary => |*bin| evaluateBinary(bin),
        .Unary => |*unary| evaluateUnary(unary),
        else => ConstantValue.NotConstant,
    };
}

// ============================================================================
// Arithmetic Operations
// ============================================================================

fn arithmeticAdd(lhs: u256, rhs: u256) ConstantValue {
    // Check for overflow
    const result = lhs +% rhs;
    if (result < lhs) {
        // Overflow occurred
        return ConstantValue.Error;
    }
    return ConstantValue{ .Integer = result };
}

fn arithmeticSub(lhs: u256, rhs: u256) ConstantValue {
    // Check for underflow
    if (rhs > lhs) {
        // Underflow occurred
        return ConstantValue.Error;
    }
    return ConstantValue{ .Integer = lhs - rhs };
}

fn arithmeticMul(lhs: u256, rhs: u256) ConstantValue {
    // Check for overflow
    const result = lhs *% rhs;
    if (rhs != 0 and result / rhs != lhs) {
        // Overflow occurred
        return ConstantValue.Error;
    }
    return ConstantValue{ .Integer = result };
}

fn arithmeticDiv(lhs: u256, rhs: u256) ConstantValue {
    // Check for division by zero
    if (rhs == 0) {
        return ConstantValue.Error;
    }
    return ConstantValue{ .Integer = lhs / rhs };
}

fn arithmeticMod(lhs: u256, rhs: u256) ConstantValue {
    // Check for division by zero
    if (rhs == 0) {
        return ConstantValue.Error;
    }
    return ConstantValue{ .Integer = lhs % rhs };
}

fn arithmeticPow(base: u256, exponent: u256) ConstantValue {
    // For constant evaluation, we only handle small exponents
    // Large exponents would cause overflow and are not practical for compile-time evaluation
    if (exponent == 0) {
        return ConstantValue{ .Integer = 1 };
    }
    if (exponent == 1) {
        return ConstantValue{ .Integer = base };
    }

    // For exponent > 1, check if result would overflow
    // Simple check: if base > 1 and exponent > 256, it's likely to overflow
    if (base > 1 and exponent > 256) {
        return ConstantValue.Error;
    }

    // Compute power iteratively with overflow checking
    var result: u256 = 1;
    var exp: u256 = exponent;
    var b: u256 = base;

    while (exp > 0) {
        if (exp & 1 == 1) {
            // Check overflow before multiplying
            const mul_result = arithmeticMul(result, b);
            switch (mul_result) {
                .Integer => |v| result = v,
                else => return ConstantValue.Error,
            }
        }
        exp >>= 1;
        if (exp > 0) {
            // Check overflow before squaring
            const mul_result = arithmeticMul(b, b);
            switch (mul_result) {
                .Integer => |v| b = v,
                else => return ConstantValue.Error,
            }
        }
    }

    return ConstantValue{ .Integer = result };
}
