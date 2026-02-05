// ============================================================================
// Shared Constant Evaluation (Comptime)
// ============================================================================
// Evaluates constant expressions for both type resolution and MLIR lowering.
// This module lives at the root to avoid cross-module path conflicts.
// ============================================================================

const std = @import("std");
const ast = @import("ast.zig");

/// Result of constant evaluation
pub const ConstantValue = union(enum) {
    Integer: u256,
    Bool: bool,
    Array: []ConstantValue,
    Range: struct { start: u256, end: u256, inclusive: bool },
    NotConstant, // Expression is not a compile-time constant
    Error, // Evaluation error (e.g., overflow, division by zero)
};

pub const IdentifierLookup = struct {
    ctx: *anyopaque,
    func: *const fn (ctx: *anyopaque, name: []const u8) ?ConstantValue,
    enum_func: ?*const fn (ctx: *anyopaque, enum_name: []const u8, variant_name: []const u8) ?ConstantValue = null,
};

/// Evaluate a constant expression to a u256 value
/// Returns null if the expression is not a compile-time constant
pub fn evaluateConstantExpression(
    allocator: std.mem.Allocator,
    expr: *ast.Expressions.ExprNode,
) !ConstantValue {
    return evaluateConstantExpressionWithLookup(allocator, expr, null);
}

pub fn evaluateConstantExpressionWithLookup(
    allocator: std.mem.Allocator,
    expr: *ast.Expressions.ExprNode,
    lookup: ?*const IdentifierLookup,
) !ConstantValue {
    return evaluateConstantExpressionRecursive(expr, lookup, allocator);
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
    // parse the string value to u256
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
        .Bool => |bool_lit| ConstantValue{ .Bool = bool_lit.value },
        else => ConstantValue.NotConstant, // Only integer literals are constants for now
    };
}

/// Evaluate a binary expression
fn evaluateBinary(bin: *ast.Expressions.BinaryExpr) ConstantValue {
    // recursively evaluate left and right operands
    const lhs_result = evaluateConstantExpressionRecursive(bin.lhs, null, std.heap.page_allocator) catch return ConstantValue.Error;
    const rhs_result = evaluateConstantExpressionRecursive(bin.rhs, null, std.heap.page_allocator) catch return ConstantValue.Error;

    // both operands must be constants
    const lhs_val = switch (lhs_result) {
        .Integer => |v| v,
        else => return ConstantValue.NotConstant,
    };
    const rhs_val = switch (rhs_result) {
        .Integer => |v| v,
        else => return ConstantValue.NotConstant,
    };

    // evaluate based on operator
    return switch (bin.operator) {
        // arithmetic operations
        .Plus => arithmeticAdd(lhs_val, rhs_val),
        .Minus => arithmeticSub(lhs_val, rhs_val),
        .Star => arithmeticMul(lhs_val, rhs_val),
        .Slash => arithmeticDiv(lhs_val, rhs_val),
        .Percent => arithmeticMod(lhs_val, rhs_val),
        .StarStar => arithmeticPow(lhs_val, rhs_val),

        // comparison operations (return boolean, but we only support integers for now)
        .EqualEqual => ConstantValue{ .Bool = lhs_val == rhs_val },
        .BangEqual => ConstantValue{ .Bool = lhs_val != rhs_val },
        .Less => ConstantValue{ .Bool = lhs_val < rhs_val },
        .LessEqual => ConstantValue{ .Bool = lhs_val <= rhs_val },
        .Greater => ConstantValue{ .Bool = lhs_val > rhs_val },
        .GreaterEqual => ConstantValue{ .Bool = lhs_val >= rhs_val },
        .And, .Or, // Logical operators (handled below for Bool operands)
        .BitwiseAnd => ConstantValue{ .Integer = lhs_val & rhs_val },
        .BitwiseOr => ConstantValue{ .Integer = lhs_val | rhs_val },
        .BitwiseXor => ConstantValue{ .Integer = lhs_val ^ rhs_val },
        .LeftShift => arithmeticShiftLeft(lhs_val, rhs_val),
        .RightShift => arithmeticShiftRight(lhs_val, rhs_val), // Bitwise operators
        .Comma,
        => ConstantValue.NotConstant, // Not constant expressions

        // note: We only evaluate arithmetic operations as constants
        // comparison and logical operations are runtime checks
    };
}

/// Evaluate a unary expression
fn evaluateUnary(unary: *ast.Expressions.UnaryExpr) ConstantValue {
    // recursively evaluate operand
    const operand_result = evaluateConstantExpressionRecursive(unary.operand, null, std.heap.page_allocator) catch return ConstantValue.Error;

    const operand_val = switch (operand_result) {
        .Integer => |v| v,
        else => return ConstantValue.NotConstant,
    };

    return switch (unary.operator) {
        .Minus => {
            // unary minus: for unsigned types, this is a compile-time error or runtime behavior
            // in Ora, unary minus on u256 would wrap around (2's complement)
            // for constant evaluation, we can compute the wrapped value
            const result = @as(u256, 0) -% operand_val; // Wrapping subtraction
            return ConstantValue{ .Integer = result };
        },
        .BitNot => ConstantValue{ .Integer = ~operand_val },
        .Bang => ConstantValue.NotConstant, // Logical not handled in bool path
    };
}

/// Recursive helper that doesn't require allocator
fn evaluateConstantExpressionRecursive(expr: *ast.Expressions.ExprNode, lookup: ?*const IdentifierLookup, allocator: std.mem.Allocator) !ConstantValue {
    return switch (expr.*) {
        .Literal => |*lit| evaluateLiteral(lit),
        .Binary => |*bin| evaluateBinaryWithLookup(bin, lookup, allocator),
        .Unary => |*unary| evaluateUnaryWithLookup(unary, lookup, allocator),
        .Call => |*call| evaluateCall(call, lookup, allocator),
        .Index => |*idx| evaluateIndex(idx, lookup, allocator),
        .FieldAccess => |*fa| evaluateFieldAccess(fa, lookup, allocator),
        .Identifier => |*id| {
            if (lookup) |lk| {
                if (lk.func(lk.ctx, id.name)) |value| {
                    return value;
                }
            }
            return ConstantValue.NotConstant;
        },
        .EnumLiteral => |*el| {
            if (lookup) |lk| {
                if (lk.enum_func) |enum_lookup| {
                    if (enum_lookup(lk.ctx, el.enum_name, el.variant_name)) |value| {
                        return value;
                    }
                }
            }
            return ConstantValue.NotConstant;
        },
        .Cast => |*cast_expr| evaluateCast(cast_expr, lookup, allocator),
        .ErrorCast => |*err_cast| evaluateErrorCast(err_cast, lookup, allocator),
        .SwitchExpression => |*sw| evaluateSwitchExpression(sw, lookup, allocator),
        .Comptime => |*ct| evaluateComptimeBlock(ct, lookup, allocator),
        .ArrayLiteral => |*arr| evaluateArrayLiteral(arr, lookup, allocator),
        .Tuple => |*tuple_expr| evaluateTuple(tuple_expr, lookup, allocator),
        .Range => |*range| evaluateRange(range, lookup, allocator),
        else => ConstantValue.NotConstant,
    };
}

fn evaluateTuple(tuple_expr: *ast.Expressions.TupleExpr, lookup: ?*const IdentifierLookup, allocator: std.mem.Allocator) ConstantValue {
    var elements = std.ArrayList(ConstantValue){};
    defer elements.deinit(allocator);
    for (tuple_expr.elements) |el| {
        const value = evaluateConstantExpressionRecursive(el, lookup, allocator) catch return ConstantValue.Error;
        if (value == .NotConstant or value == .Error) return ConstantValue.NotConstant;
        elements.append(allocator, value) catch return ConstantValue.Error;
    }
    const owned = elements.toOwnedSlice(allocator) catch return ConstantValue.Error;
    return ConstantValue{ .Array = owned };
}

fn evaluateCast(cast_expr: *ast.Expressions.CastExpr, lookup: ?*const IdentifierLookup, allocator: std.mem.Allocator) ConstantValue {
    const operand_result = evaluateConstantExpressionRecursive(cast_expr.operand, lookup, allocator) catch return ConstantValue.Error;
    return switch (operand_result) {
        .Integer => operand_result,
        .Bool => operand_result,
        else => ConstantValue.NotConstant,
    };
}

fn evaluateErrorCast(err_cast: *ast.Expressions.ErrorCastExpr, lookup: ?*const IdentifierLookup, allocator: std.mem.Allocator) ConstantValue {
    const operand_result = evaluateConstantExpressionRecursive(err_cast.operand, lookup, allocator) catch return ConstantValue.Error;
    return switch (operand_result) {
        .Integer => operand_result,
        .Bool => operand_result,
        else => ConstantValue.NotConstant,
    };
}

fn evaluateBinaryWithLookup(bin: *ast.Expressions.BinaryExpr, lookup: ?*const IdentifierLookup, allocator: std.mem.Allocator) ConstantValue {
    const lhs_result = evaluateConstantExpressionRecursive(bin.lhs, lookup, allocator) catch return ConstantValue.Error;
    const rhs_result = evaluateConstantExpressionRecursive(bin.rhs, lookup, allocator) catch return ConstantValue.Error;
    switch (bin.operator) {
        .And, .Or => {
            const lhs_bool = switch (lhs_result) {
                .Bool => |v| v,
                else => return ConstantValue.NotConstant,
            };
            const rhs_bool = switch (rhs_result) {
                .Bool => |v| v,
                else => return ConstantValue.NotConstant,
            };
            return ConstantValue{ .Bool = if (bin.operator == .And) (lhs_bool and rhs_bool) else (lhs_bool or rhs_bool) };
        },
        else => {
            const lhs_val = switch (lhs_result) {
                .Integer => |v| v,
                else => return ConstantValue.NotConstant,
            };
            const rhs_val = switch (rhs_result) {
                .Integer => |v| v,
                else => return ConstantValue.NotConstant,
            };
            return switch (bin.operator) {
                .Plus => arithmeticAdd(lhs_val, rhs_val),
                .Minus => arithmeticSub(lhs_val, rhs_val),
                .Star => arithmeticMul(lhs_val, rhs_val),
                .Slash => arithmeticDiv(lhs_val, rhs_val),
                .Percent => arithmeticMod(lhs_val, rhs_val),
                .StarStar => arithmeticPow(lhs_val, rhs_val),
                .EqualEqual => ConstantValue{ .Bool = lhs_val == rhs_val },
                .BangEqual => ConstantValue{ .Bool = lhs_val != rhs_val },
                .Less => ConstantValue{ .Bool = lhs_val < rhs_val },
                .LessEqual => ConstantValue{ .Bool = lhs_val <= rhs_val },
                .Greater => ConstantValue{ .Bool = lhs_val > rhs_val },
                .GreaterEqual => ConstantValue{ .Bool = lhs_val >= rhs_val },
                .BitwiseAnd => ConstantValue{ .Integer = lhs_val & rhs_val },
                .BitwiseOr => ConstantValue{ .Integer = lhs_val | rhs_val },
                .BitwiseXor => ConstantValue{ .Integer = lhs_val ^ rhs_val },
                .LeftShift => arithmeticShiftLeft(lhs_val, rhs_val),
                .RightShift => arithmeticShiftRight(lhs_val, rhs_val),
                .Comma,
                => ConstantValue.NotConstant,
                .And, .Or => ConstantValue.NotConstant,
            };
        },
    }
}

fn evaluateUnaryWithLookup(unary: *ast.Expressions.UnaryExpr, lookup: ?*const IdentifierLookup, allocator: std.mem.Allocator) ConstantValue {
    const operand_result = evaluateConstantExpressionRecursive(unary.operand, lookup, allocator) catch return ConstantValue.Error;
    return switch (unary.operator) {
        .Minus => {
            const operand_val = switch (operand_result) {
                .Integer => |v| v,
                else => return ConstantValue.NotConstant,
            };
            const result = @as(u256, 0) -% operand_val;
            return ConstantValue{ .Integer = result };
        },
        .BitNot => {
            const operand_val = switch (operand_result) {
                .Integer => |v| v,
                else => return ConstantValue.NotConstant,
            };
            return ConstantValue{ .Integer = ~operand_val };
        },
        .Bang => {
            const operand_bool = switch (operand_result) {
                .Bool => |v| v,
                else => return ConstantValue.NotConstant,
            };
            return ConstantValue{ .Bool = !operand_bool };
        },
    };
}

fn evaluateComptimeBlock(ct: *ast.Expressions.ComptimeExpr, lookup: ?*const IdentifierLookup, allocator: std.mem.Allocator) ConstantValue {
    for (ct.block.statements) |*stmt| {
        switch (stmt.*) {
            .Return => |ret| {
                if (ret.value) |*expr| {
                    return evaluateConstantExpressionRecursive(@constCast(expr), lookup, allocator) catch ConstantValue.Error;
                }
            },
            .Expr => |expr_node| {
                return evaluateConstantExpressionRecursive(@constCast(&expr_node), lookup, allocator) catch ConstantValue.Error;
            },
            else => return ConstantValue.NotConstant,
        }
    }
    return ConstantValue.NotConstant;
}

fn evaluateSwitchExpression(sw: *ast.Expressions.SwitchExprNode, lookup: ?*const IdentifierLookup, allocator: std.mem.Allocator) ConstantValue {
    const condition = evaluateConstantExpressionRecursive(sw.condition, lookup, allocator) catch return ConstantValue.Error;
    if (condition == .NotConstant) return ConstantValue.NotConstant;

    for (sw.cases) |*case| {
        if (!switchPatternMatches(&case.pattern, condition, lookup, allocator)) continue;
        return evaluateSwitchBody(&case.body, lookup, allocator);
    }

    if (sw.default_case) |*default_block| {
        return evaluateBlockExpression(default_block, lookup, allocator);
    }

    return ConstantValue.NotConstant;
}

fn switchPatternMatches(pattern: *const ast.Expressions.SwitchPattern, condition: ConstantValue, lookup: ?*const IdentifierLookup, allocator: std.mem.Allocator) bool {
    return switch (pattern.*) {
        .Literal => |lit| {
            var expr = ast.Expressions.ExprNode{ .Literal = lit.value };
            const lit_val = evaluateConstantExpressionRecursive(&expr, lookup, allocator) catch return false;
            return constantEquals(condition, lit_val);
        },
        .Range => |range| {
            const start_val = evaluateConstantExpressionRecursive(range.start, lookup, allocator) catch return false;
            const end_val = evaluateConstantExpressionRecursive(range.end, lookup, allocator) catch return false;
            if (condition != .Integer or start_val != .Integer or end_val != .Integer) return false;
            var max_val: u256 = end_val.Integer;
            if (!range.inclusive and max_val > 0) {
                max_val -= 1;
            }
            return condition.Integer >= start_val.Integer and condition.Integer <= max_val;
        },
        .EnumValue => |ev| {
            if (lookup) |lk| {
                if (lk.enum_func) |enum_lookup| {
                    if (enum_lookup(lk.ctx, ev.enum_name, ev.variant_name)) |value| {
                        return constantEquals(condition, value);
                    }
                }
            }
            return false;
        },
        .Else => true,
    };
}

fn constantEquals(a: ConstantValue, b: ConstantValue) bool {
    return switch (a) {
        .Integer => |av| b == .Integer and av == b.Integer,
        .Bool => |ab| b == .Bool and ab == b.Bool,
        else => false,
    };
}

fn evaluateSwitchBody(body: *const ast.Expressions.SwitchBody, lookup: ?*const IdentifierLookup, allocator: std.mem.Allocator) ConstantValue {
    return switch (body.*) {
        .Expression => |expr| evaluateConstantExpressionRecursive(expr, lookup, allocator) catch ConstantValue.Error,
        .Block => |*block| evaluateBlockExpression(block, lookup, allocator),
        .LabeledBlock => |*lb| evaluateBlockExpression(&lb.block, lookup, allocator),
    };
}

fn evaluateBlockExpression(block: *const ast.Statements.BlockNode, lookup: ?*const IdentifierLookup, allocator: std.mem.Allocator) ConstantValue {
    if (block.statements.len == 0) return ConstantValue.NotConstant;
    const stmt = &block.statements[0];
    return switch (stmt.*) {
        .Expr => |expr_node| evaluateConstantExpressionRecursive(@constCast(&expr_node), lookup, allocator) catch ConstantValue.Error,
        .Return => |ret| blk: {
            if (ret.value) |*expr| break :blk evaluateConstantExpressionRecursive(@constCast(expr), lookup, allocator) catch ConstantValue.Error;
            break :blk ConstantValue.NotConstant;
        },
        else => ConstantValue.NotConstant,
    };
}

fn evaluateArrayLiteral(arr: *ast.Expressions.ArrayLiteralExpr, lookup: ?*const IdentifierLookup, allocator: std.mem.Allocator) ConstantValue {
    const len = arr.elements.len;
    var values = allocator.alloc(ConstantValue, len) catch return ConstantValue.Error;
    for (arr.elements, 0..) |elem, i| {
        const elem_val = evaluateConstantExpressionRecursive(elem, lookup, allocator) catch return ConstantValue.Error;
        switch (elem_val) {
            .Integer, .Bool => values[i] = elem_val,
            else => return ConstantValue.NotConstant,
        }
    }
    return ConstantValue{ .Array = values };
}

fn evaluateRange(range: *ast.Expressions.RangeExpr, lookup: ?*const IdentifierLookup, allocator: std.mem.Allocator) ConstantValue {
    const start_val = evaluateConstantExpressionRecursive(range.start, lookup, allocator) catch return ConstantValue.Error;
    const end_val = evaluateConstantExpressionRecursive(range.end, lookup, allocator) catch return ConstantValue.Error;
    if (start_val != .Integer or end_val != .Integer) return ConstantValue.NotConstant;
    return ConstantValue{ .Range = .{ .start = start_val.Integer, .end = end_val.Integer, .inclusive = range.inclusive } };
}

fn evaluateCall(call: *ast.Expressions.CallExpr, lookup: ?*const IdentifierLookup, allocator: std.mem.Allocator) ConstantValue {
    const builtin_name = switch (call.callee.*) {
        .Identifier => |id| id.name,
        else => return ConstantValue.NotConstant,
    };

    if (std.mem.eql(u8, builtin_name, "@divTrunc") or
        std.mem.eql(u8, builtin_name, "@divFloor") or
        std.mem.eql(u8, builtin_name, "@divCeil") or
        std.mem.eql(u8, builtin_name, "@divExact"))
    {
        if (call.arguments.len != 2) return ConstantValue.NotConstant;
        const lhs = evaluateConstantExpressionRecursive(call.arguments[0], lookup, allocator) catch return ConstantValue.Error;
        const rhs = evaluateConstantExpressionRecursive(call.arguments[1], lookup, allocator) catch return ConstantValue.Error;
        if (lhs != .Integer or rhs != .Integer) return ConstantValue.NotConstant;
        if (rhs.Integer == 0) return ConstantValue.Error;
        const dividend = lhs.Integer;
        const divisor = rhs.Integer;

        if (std.mem.eql(u8, builtin_name, "@divExact")) {
            if (dividend % divisor != 0) return ConstantValue.Error;
            return ConstantValue{ .Integer = dividend / divisor };
        }

        if (std.mem.eql(u8, builtin_name, "@divTrunc")) {
            return ConstantValue{ .Integer = dividend / divisor };
        }

        if (std.mem.eql(u8, builtin_name, "@divFloor")) {
            return ConstantValue{ .Integer = dividend / divisor };
        }

        if (std.mem.eql(u8, builtin_name, "@divCeil")) {
            const q = dividend / divisor;
            const r = dividend % divisor;
            if (r == 0) return ConstantValue{ .Integer = q };
            return ConstantValue{ .Integer = q + 1 };
        }
    }

    return ConstantValue.NotConstant;
}

fn evaluateIndex(idx: *ast.Expressions.IndexExpr, lookup: ?*const IdentifierLookup, allocator: std.mem.Allocator) ConstantValue {
    const target = evaluateConstantExpressionRecursive(idx.target, lookup, allocator) catch return ConstantValue.Error;
    const index = evaluateConstantExpressionRecursive(idx.index, lookup, allocator) catch return ConstantValue.Error;

    if (target != .Array or index != .Integer) return ConstantValue.NotConstant;
    const i: usize = @intCast(index.Integer);
    if (i >= target.Array.len) return ConstantValue.Error;
    return target.Array[i];
}

fn evaluateFieldAccess(fa: *ast.Expressions.FieldAccessExpr, lookup: ?*const IdentifierLookup, allocator: std.mem.Allocator) ConstantValue {
    const target = evaluateConstantExpressionRecursive(fa.target, lookup, allocator) catch return ConstantValue.Error;
    if (target != .Array) return ConstantValue.NotConstant;
    const field_name = if (fa.field.len > 0 and fa.field[0] == '_') fa.field[1..] else fa.field;
    const index = std.fmt.parseInt(usize, field_name, 10) catch return ConstantValue.NotConstant;
    if (index >= target.Array.len) return ConstantValue.Error;
    return target.Array[index];
}

// ============================================================================
// Arithmetic Operations
// ============================================================================

fn arithmeticAdd(lhs: u256, rhs: u256) ConstantValue {
    // check for overflow
    const result = lhs +% rhs;
    if (result < lhs) {
        // overflow occurred
        return ConstantValue.Error;
    }
    return ConstantValue{ .Integer = result };
}

fn arithmeticSub(lhs: u256, rhs: u256) ConstantValue {
    // check for underflow
    if (rhs > lhs) {
        // underflow occurred
        return ConstantValue.Error;
    }
    return ConstantValue{ .Integer = lhs - rhs };
}

fn arithmeticMul(lhs: u256, rhs: u256) ConstantValue {
    // check for overflow
    const result = lhs *% rhs;
    if (rhs != 0 and result / rhs != lhs) {
        // overflow occurred
        return ConstantValue.Error;
    }
    return ConstantValue{ .Integer = result };
}

fn arithmeticDiv(lhs: u256, rhs: u256) ConstantValue {
    // check for division by zero
    if (rhs == 0) {
        return ConstantValue.Error;
    }
    return ConstantValue{ .Integer = lhs / rhs };
}

fn arithmeticMod(lhs: u256, rhs: u256) ConstantValue {
    // check for division by zero
    if (rhs == 0) {
        return ConstantValue.Error;
    }
    return ConstantValue{ .Integer = lhs % rhs };
}

fn arithmeticPow(base: u256, exponent: u256) ConstantValue {
    // for constant evaluation, we only handle small exponents
    // large exponents would cause overflow and are not practical for compile-time evaluation
    if (exponent == 0) {
        return ConstantValue{ .Integer = 1 };
    }
    if (exponent == 1) {
        return ConstantValue{ .Integer = base };
    }

    // for exponent > 1, check if result would overflow
    // simple check: if base > 1 and exponent > 256, it's likely to overflow
    if (base > 1 and exponent > 256) {
        return ConstantValue.Error;
    }

    // compute power iteratively with overflow checking
    var result: u256 = 1;
    var exp: u256 = exponent;
    var b: u256 = base;

    while (exp > 0) {
        if (exp & 1 == 1) {
            // check overflow before multiplying
            const mul_result = arithmeticMul(result, b);
            switch (mul_result) {
                .Integer => |v| result = v,
                else => return ConstantValue.Error,
            }
        }
        exp >>= 1;
        if (exp > 0) {
            // check overflow before squaring
            const mul_result = arithmeticMul(b, b);
            switch (mul_result) {
                .Integer => |v| b = v,
                else => return ConstantValue.Error,
            }
        }
    }

    return ConstantValue{ .Integer = result };
}

fn arithmeticShiftLeft(lhs: u256, rhs: u256) ConstantValue {
    if (rhs >= 256) return ConstantValue.Error;
    const shift: u8 = @intCast(rhs);
    return ConstantValue{ .Integer = lhs << shift };
}

fn arithmeticShiftRight(lhs: u256, rhs: u256) ConstantValue {
    if (rhs >= 256) return ConstantValue.Error;
    const shift: u8 = @intCast(rhs);
    return ConstantValue{ .Integer = lhs >> shift };
}
