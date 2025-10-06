const std = @import("std");
const lexer = @import("../lexer.zig");
const ast = @import("../ast.zig");
const TypeInfo = @import("../ast/type_info.zig").TypeInfo;
const common = @import("common.zig");
const common_parsers = @import("common_parsers.zig");
const AstArena = @import("../ast/ast_arena.zig").AstArena;
const TypeParser = @import("type_parser.zig").TypeParser;

const Token = lexer.Token;
const TokenType = lexer.TokenType;
const BaseParser = common.BaseParser;
const ParserCommon = common.ParserCommon;
const ParserError = @import("parser_core.zig").ParserError;

// Import common parser functions
const parseSwitchPattern = common_parsers.parseSwitchPattern;
const parseSwitchBody = common_parsers.parseSwitchBody;

/// Specialized parser for expressions using precedence climbing
pub const ExpressionParser = struct {
    base: BaseParser,

    pub fn init(tokens: []const Token, arena: *AstArena) ExpressionParser {
        return ExpressionParser{
            .base = BaseParser.init(tokens, arena),
        };
    }

    /// Parse expression with precedence climbing (entry point)
    pub fn parseExpression(self: *ExpressionParser) ParserError!ast.Expressions.ExprNode {
        return self.parseComma();
    }

    /// Parse an expression that must NOT consume top-level commas.
    ///
    /// This is used in contexts where a comma is a list/case separator
    /// (e.g., switch expression arms, function arg separators handled elsewhere),
    /// so we start from assignment-precedence instead of comma-precedence.
    pub fn parseExpressionNoComma(self: *ExpressionParser) ParserError!ast.Expressions.ExprNode {
        return self.parseAssignment();
    }

    /// Parse comma expressions (lowest precedence)
    fn parseComma(self: *ExpressionParser) ParserError!ast.Expressions.ExprNode {
        var expr = try self.parseAssignment();

        while (self.base.match(.Comma)) {
            const comma_token = self.base.previous();
            const right = try self.parseAssignment();

            const left_ptr = try self.base.arena.createNode(ast.Expressions.ExprNode);
            left_ptr.* = expr;
            const right_ptr = try self.base.arena.createNode(ast.Expressions.ExprNode);
            right_ptr.* = right;

            expr = ast.Expressions.ExprNode{
                .Binary = ast.Expressions.BinaryExpr{
                    .lhs = left_ptr,
                    .operator = .Comma,
                    .rhs = right_ptr,
                    .type_info = ast.Types.TypeInfo.unknown(), // Result type will be resolved later
                    .span = self.base.spanFromToken(comma_token),
                },
            };
        }

        return expr;
    }

    /// Parse assignment expressions (precedence 14)
    fn parseAssignment(self: *ExpressionParser) ParserError!ast.Expressions.ExprNode {
        const expr = try self.parseLogicalOr();

        // Legacy move/shift syntax removed; handled by statement parser as 'move ... from ... to ...;'

        // Simple assignment
        if (self.base.match(.Equal)) {
            // Validate that the left side is a valid L-value
            ast.expressions.validateLValue(&expr) catch |err| {
                const error_msg = switch (err) {
                    ast.expressions.LValueError.LiteralNotAssignable => "Cannot assign to literal value",
                    ast.expressions.LValueError.CallNotAssignable => "Cannot assign to function call result",
                    ast.expressions.LValueError.BinaryExprNotAssignable => "Cannot assign to binary expression",
                    ast.expressions.LValueError.UnaryExprNotAssignable => "Cannot assign to unary expression",
                    else => "Invalid assignment target",
                };
                try self.base.errorAtCurrent(error_msg);
                return error.UnexpectedToken;
            };

            const value = try self.parseAssignment(); // Right-associative
            const expr_ptr = try self.base.arena.createNode(ast.Expressions.ExprNode);
            expr_ptr.* = expr;
            const value_ptr = try self.base.arena.createNode(ast.Expressions.ExprNode);
            value_ptr.* = value;

            return ast.Expressions.ExprNode{ .Assignment = ast.Expressions.AssignmentExpr{
                .target = expr_ptr,
                .value = value_ptr,
                .span = self.base.spanFromToken(self.base.previous()),
            } };
        }

        // Compound assignments
        if (self.base.match(.PlusEqual) or self.base.match(.MinusEqual) or self.base.match(.StarEqual) or
            self.base.match(.SlashEqual) or self.base.match(.PercentEqual))
        {
            const op_token = self.base.previous();

            // Validate that the left side is a valid L-value
            ast.expressions.validateLValue(&expr) catch |err| {
                const error_msg = switch (err) {
                    ast.expressions.LValueError.LiteralNotAssignable => "Cannot assign to literal value",
                    ast.expressions.LValueError.CallNotAssignable => "Cannot assign to function call result",
                    ast.expressions.LValueError.BinaryExprNotAssignable => "Cannot assign to binary expression",
                    ast.expressions.LValueError.UnaryExprNotAssignable => "Cannot assign to unary expression",
                    else => "Invalid assignment target",
                };
                try self.base.errorAtCurrent(error_msg);
                return error.UnexpectedToken;
            };

            const value = try self.parseAssignment(); // Right-associative
            const expr_ptr = try self.base.arena.createNode(ast.Expressions.ExprNode);
            expr_ptr.* = expr;
            const value_ptr = try self.base.arena.createNode(ast.Expressions.ExprNode);
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
                .span = self.base.spanFromToken(op_token),
            } };
        }

        return expr;
    }

    /// Parse logical OR expressions (precedence 13) - MIGRATED FROM ORIGINAL
    pub fn parseLogicalOr(self: *ExpressionParser) ParserError!ast.Expressions.ExprNode {
        var expr = try self.parseLogicalAnd();

        while (self.base.match(.PipePipe)) {
            const op_token = self.base.previous();
            const right = try self.parseLogicalAnd();

            const left_ptr = try self.base.arena.createNode(ast.Expressions.ExprNode);
            left_ptr.* = expr;
            const right_ptr = try self.base.arena.createNode(ast.Expressions.ExprNode);
            right_ptr.* = right;

            expr = ast.Expressions.ExprNode{
                .Binary = ast.Expressions.BinaryExpr{
                    .lhs = left_ptr,
                    .operator = .Or, // Logical OR
                    .rhs = right_ptr,
                    .type_info = ast.Types.CommonTypes.bool_type(), // Logical operations return bool
                    .span = self.base.spanFromToken(op_token),
                },
            };
        }

        return expr;
    }

    /// Parse logical AND expressions (precedence 12) - MIGRATED FROM ORIGINAL
    fn parseLogicalAnd(self: *ExpressionParser) ParserError!ast.Expressions.ExprNode {
        var expr = try self.parseBitwiseOr();

        while (self.base.match(.AmpersandAmpersand)) {
            const op_token = self.base.previous();
            const right = try self.parseBitwiseOr();

            const left_ptr = try self.base.arena.createNode(ast.Expressions.ExprNode);
            left_ptr.* = expr;
            const right_ptr = try self.base.arena.createNode(ast.Expressions.ExprNode);
            right_ptr.* = right;

            expr = ast.Expressions.ExprNode{
                .Binary = ast.Expressions.BinaryExpr{
                    .lhs = left_ptr,
                    .operator = .And, // Logical AND
                    .rhs = right_ptr,
                    .type_info = ast.Types.CommonTypes.bool_type(), // Logical operations return bool
                    .span = self.base.spanFromToken(op_token),
                },
            };
        }

        return expr;
    }

    /// Parse bitwise OR expressions (precedence 11) - MIGRATED FROM ORIGINAL
    fn parseBitwiseOr(self: *ExpressionParser) ParserError!ast.Expressions.ExprNode {
        var expr = try self.parseBitwiseXor();

        while (self.base.match(.Pipe)) {
            const op_token = self.base.previous();
            const right = try self.parseBitwiseXor();

            const left_ptr = try self.base.arena.createNode(ast.Expressions.ExprNode);
            left_ptr.* = expr;
            const right_ptr = try self.base.arena.createNode(ast.Expressions.ExprNode);
            right_ptr.* = right;

            expr = ast.Expressions.ExprNode{
                .Binary = ast.Expressions.BinaryExpr{
                    .lhs = left_ptr,
                    .operator = .BitwiseOr,
                    .rhs = right_ptr,
                    .type_info = ast.Types.TypeInfo.unknown(), // Type will be inferred from operands
                    .span = self.base.spanFromToken(op_token),
                },
            };
        }

        return expr;
    }

    /// Parse bitwise XOR expressions (precedence 10) - MIGRATED FROM ORIGINAL
    fn parseBitwiseXor(self: *ExpressionParser) ParserError!ast.Expressions.ExprNode {
        var expr = try self.parseBitwiseAnd();

        while (self.base.match(.Caret)) {
            const op_token = self.base.previous();
            const right = try self.parseBitwiseAnd();

            const left_ptr = try self.base.arena.createNode(ast.Expressions.ExprNode);
            left_ptr.* = expr;
            const right_ptr = try self.base.arena.createNode(ast.Expressions.ExprNode);
            right_ptr.* = right;

            expr = ast.Expressions.ExprNode{
                .Binary = ast.Expressions.BinaryExpr{
                    .lhs = left_ptr,
                    .operator = .BitwiseXor,
                    .rhs = right_ptr,
                    .type_info = ast.Types.TypeInfo.unknown(), // Type will be inferred from operands
                    .span = self.base.spanFromToken(op_token),
                },
            };
        }

        return expr;
    }

    /// Parse bitwise AND expressions (precedence 9) - MIGRATED FROM ORIGINAL
    fn parseBitwiseAnd(self: *ExpressionParser) ParserError!ast.Expressions.ExprNode {
        var expr = try self.parseEquality();

        while (self.base.match(.Ampersand)) {
            const op_token = self.base.previous();
            const right = try self.parseEquality();

            const left_ptr = try self.base.arena.createNode(ast.Expressions.ExprNode);
            left_ptr.* = expr;
            const right_ptr = try self.base.arena.createNode(ast.Expressions.ExprNode);
            right_ptr.* = right;

            expr = ast.Expressions.ExprNode{
                .Binary = ast.Expressions.BinaryExpr{
                    .lhs = left_ptr,
                    .operator = .BitwiseAnd,
                    .rhs = right_ptr,
                    .type_info = ast.Types.TypeInfo.unknown(), // Type will be inferred from operands
                    .span = self.base.spanFromToken(op_token),
                },
            };
        }

        return expr;
    }

    /// Parse equality expressions (precedence 8) - MIGRATED FROM ORIGINAL
    fn parseEquality(self: *ExpressionParser) ParserError!ast.Expressions.ExprNode {
        var expr = try self.parseComparison();

        while (self.base.match(.EqualEqual) or self.base.match(.BangEqual)) {
            const op_token = self.base.previous();
            const right = try self.parseComparison();

            const operator: ast.Operators.Binary = switch (op_token.type) {
                .EqualEqual => .EqualEqual,
                .BangEqual => .BangEqual,
                else => unreachable,
            };

            const left_ptr = try self.base.arena.createNode(ast.Expressions.ExprNode);
            left_ptr.* = expr;
            const right_ptr = try self.base.arena.createNode(ast.Expressions.ExprNode);
            right_ptr.* = right;

            expr = ast.Expressions.ExprNode{
                .Binary = ast.Expressions.BinaryExpr{
                    .lhs = left_ptr,
                    .operator = operator,
                    .rhs = right_ptr,
                    .type_info = ast.Types.CommonTypes.bool_type(), // Equality operations return bool
                    .span = self.base.spanFromToken(op_token),
                },
            };
        }

        return expr;
    }

    /// Parse comparison expressions (precedence 7) - MIGRATED FROM ORIGINAL
    fn parseComparison(self: *ExpressionParser) ParserError!ast.Expressions.ExprNode {
        var expr = try self.parseBitwiseShift();

        while (self.base.match(.Less) or self.base.match(.LessEqual) or self.base.match(.Greater) or self.base.match(.GreaterEqual)) {
            const op_token = self.base.previous();
            const right = try self.parseBitwiseShift();

            const operator: ast.Operators.Binary = switch (op_token.type) {
                .Less => .Less,
                .LessEqual => .LessEqual,
                .Greater => .Greater,
                .GreaterEqual => .GreaterEqual,
                else => unreachable,
            };

            const left_ptr = try self.base.arena.createNode(ast.Expressions.ExprNode);
            left_ptr.* = expr;
            const right_ptr = try self.base.arena.createNode(ast.Expressions.ExprNode);
            right_ptr.* = right;

            expr = ast.Expressions.ExprNode{
                .Binary = ast.Expressions.BinaryExpr{
                    .lhs = left_ptr,
                    .operator = operator,
                    .rhs = right_ptr,
                    .type_info = ast.Types.CommonTypes.bool_type(), // Comparison operations return bool
                    .span = self.base.spanFromToken(op_token),
                },
            };
        }

        return expr;
    }

    /// Parse bitwise shift expressions (precedence 6) - MIGRATED FROM ORIGINAL
    fn parseBitwiseShift(self: *ExpressionParser) ParserError!ast.Expressions.ExprNode {
        var expr = try self.parseTerm();

        while (self.base.match(.LessLess) or self.base.match(.GreaterGreater)) {
            const op_token = self.base.previous();
            const right = try self.parseTerm();

            const operator: ast.Operators.Binary = switch (op_token.type) {
                .LessLess => .LeftShift,
                .GreaterGreater => .RightShift,
                else => unreachable,
            };

            const left_ptr = try self.base.arena.createNode(ast.Expressions.ExprNode);
            left_ptr.* = expr;
            const right_ptr = try self.base.arena.createNode(ast.Expressions.ExprNode);
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

    /// Parse term expressions (precedence 5: + -) - MIGRATED FROM ORIGINAL
    fn parseTerm(self: *ExpressionParser) ParserError!ast.Expressions.ExprNode {
        var expr = try self.parseFactor();

        while (self.base.match(.Plus) or self.base.match(.Minus)) {
            const op_token = self.base.previous();
            const right = try self.parseFactor();

            const operator: ast.Operators.Binary = switch (op_token.type) {
                .Plus => .Plus,
                .Minus => .Minus,
                else => unreachable,
            };

            const left_ptr = try self.base.arena.createNode(ast.Expressions.ExprNode);
            left_ptr.* = expr;
            const right_ptr = try self.base.arena.createNode(ast.Expressions.ExprNode);
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

    /// Parse factor expressions (precedence 4: * / %) - MIGRATED FROM ORIGINAL
    fn parseFactor(self: *ExpressionParser) ParserError!ast.Expressions.ExprNode {
        var expr = try self.parseExponent();

        while (self.base.match(.Star) or self.base.match(.Slash) or self.base.match(.Percent)) {
            const op_token = self.base.previous();
            const right = try self.parseExponent();

            const operator: ast.Operators.Binary = switch (op_token.type) {
                .Star => .Star,
                .Slash => .Slash,
                .Percent => .Percent,
                else => unreachable,
            };

            const left_ptr = try self.base.arena.createNode(ast.Expressions.ExprNode);
            left_ptr.* = expr;
            const right_ptr = try self.base.arena.createNode(ast.Expressions.ExprNode);
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

    /// Parse exponentiation expressions (precedence 3: **) - MIGRATED FROM ORIGINAL
    fn parseExponent(self: *ExpressionParser) ParserError!ast.Expressions.ExprNode {
        var expr = try self.parseUnary();

        while (self.base.match(.StarStar)) {
            const op_token = self.base.previous();
            const right = try self.parseExponent(); // Right-associative

            const left_ptr = try self.base.arena.createNode(ast.Expressions.ExprNode);
            left_ptr.* = expr;
            const right_ptr = try self.base.arena.createNode(ast.Expressions.ExprNode);
            right_ptr.* = right;

            expr = ast.Expressions.ExprNode{
                .Binary = ast.Expressions.BinaryExpr{
                    .lhs = left_ptr,
                    .operator = .StarStar,
                    .rhs = right_ptr,
                    .type_info = ast.Types.TypeInfo.unknown(), // Type will be inferred from operands
                    .span = ParserCommon.makeSpan(op_token),
                },
            };
        }

        return expr;
    }

    /// Parse unary expressions (precedence 2: ! -) - MIGRATED FROM ORIGINAL
    fn parseUnary(self: *ExpressionParser) ParserError!ast.Expressions.ExprNode {
        if (self.base.match(.Bang) or self.base.match(.Minus)) {
            const op_token = self.base.previous();
            const right = try self.parseUnary(); // Right-associative for unary operators

            const operator: ast.Operators.Unary = switch (op_token.type) {
                .Bang => .Bang,
                .Minus => .Minus,
                else => unreachable,
            };

            const right_ptr = try self.base.arena.createNode(ast.Expressions.ExprNode);
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

        return self.parseCall();
    }

    /// Parse function calls and member access - MIGRATED FROM ORIGINAL
    fn parseCall(self: *ExpressionParser) ParserError!ast.Expressions.ExprNode {
        var expr = try self.parsePrimary();

        while (true) {
            if (self.base.match(.LeftParen)) {
                expr = try self.finishCall(expr);
            } else if (self.base.match(.Dot)) {
                const name_token = try self.base.consume(.Identifier, "Expected property name after '.'");

                // Check if this might be an enum literal (EnumType.VariantName)
                // But exclude known module/namespace patterns
                if (expr == .Identifier) {
                    const enum_name = expr.Identifier.name;

                    // Don't treat standard library and module access as enum literals
                    const is_module_access = std.mem.eql(u8, enum_name, "std") or
                        std.mem.eql(u8, enum_name, "constants") or
                        std.mem.eql(u8, enum_name, "transaction") or
                        std.mem.eql(u8, enum_name, "block") or
                        std.mem.eql(u8, enum_name, "math");

                    if (is_module_access) {
                        // Treat as field access for module patterns
                        const expr_ptr = try self.base.arena.createNode(ast.Expressions.ExprNode);
                        expr_ptr.* = expr;
                        const field_name = try self.base.arena.createString(name_token.lexeme);
                        expr = ast.Expressions.ExprNode{
                            .FieldAccess = ast.Expressions.FieldAccessExpr{
                                .target = expr_ptr,
                                .field = field_name,
                                .type_info = ast.Types.TypeInfo.unknown(), // Field type will be resolved from struct definition
                                .span = self.base.spanFromToken(name_token),
                            },
                        };
                    } else {
                        // Treat as potential enum literal
                        const variant_name = try self.base.arena.createString(name_token.lexeme);
                        expr = ast.Expressions.ExprNode{ .EnumLiteral = ast.Expressions.EnumLiteralExpr{
                            .enum_name = enum_name,
                            .variant_name = variant_name,
                            .span = self.base.spanFromToken(name_token),
                        } };
                    }
                } else {
                    // Complex expressions are always field access
                    const expr_ptr = try self.base.arena.createNode(ast.Expressions.ExprNode);
                    expr_ptr.* = expr;
                    const field_name = try self.base.arena.createString(name_token.lexeme);
                    expr = ast.Expressions.ExprNode{
                        .FieldAccess = ast.Expressions.FieldAccessExpr{
                            .target = expr_ptr,
                            .field = field_name,
                            .type_info = ast.Types.TypeInfo.unknown(), // Will be resolved during type checking
                            .span = self.base.spanFromToken(name_token),
                        },
                    };
                }
            } else if (self.base.match(.LeftBracket)) {
                const index = try self.parseExpression();

                // Check for double mapping access: target[key1, key2]
                if (self.base.match(.Comma)) {
                    const second_index = try self.parseExpression();
                    _ = try self.base.consume(.RightBracket, "Expected ']' after double mapping index");

                    // Create pointers for the nested structure
                    const expr_ptr = try self.base.arena.createNode(ast.Expressions.ExprNode);
                    expr_ptr.* = expr;
                    const index_ptr = try self.base.arena.createNode(ast.Expressions.ExprNode);
                    index_ptr.* = index;
                    const second_index_ptr = try self.base.arena.createNode(ast.Expressions.ExprNode);
                    second_index_ptr.* = second_index;

                    // Create a nested index expression for double mapping: target[key1][key2]
                    const first_index = ast.Expressions.ExprNode{ .Index = ast.Expressions.IndexExpr{
                        .target = expr_ptr,
                        .index = index_ptr,
                        .span = self.base.spanFromToken(self.base.previous()),
                    } };

                    const first_index_ptr = try self.base.arena.createNode(ast.Expressions.ExprNode);
                    first_index_ptr.* = first_index;

                    expr = ast.Expressions.ExprNode{ .Index = ast.Expressions.IndexExpr{
                        .target = first_index_ptr,
                        .index = second_index_ptr,
                        .span = self.base.spanFromToken(self.base.previous()),
                    } };
                } else {
                    _ = try self.base.consume(.RightBracket, "Expected ']' after array index");

                    const expr_ptr = try self.base.arena.createNode(ast.Expressions.ExprNode);
                    expr_ptr.* = expr;
                    const index_ptr = try self.base.arena.createNode(ast.Expressions.ExprNode);
                    index_ptr.* = index;

                    expr = ast.Expressions.ExprNode{ .Index = ast.Expressions.IndexExpr{
                        .target = expr_ptr,
                        .index = index_ptr,
                        .span = self.base.spanFromToken(self.base.previous()),
                    } };
                }
            } else {
                break;
            }
        }

        return expr;
    }

    // Note: '@cast(Type, expr)' is the only supported cast syntax.

    /// Finish parsing a function call - MIGRATED FROM ORIGINAL
    fn finishCall(self: *ExpressionParser, callee: ast.Expressions.ExprNode) ParserError!ast.Expressions.ExprNode {
        var arguments = std.ArrayList(*ast.Expressions.ExprNode){};
        defer arguments.deinit(self.base.arena.allocator());

        if (!self.base.check(.RightParen)) {
            repeat: while (true) {
                const arg = try self.parseExpression();
                const arg_ptr = try self.base.arena.createNode(ast.Expressions.ExprNode);
                arg_ptr.* = arg;
                try arguments.append(self.base.arena.allocator(), arg_ptr);
                if (!self.base.match(.Comma)) break :repeat;
            }
        }

        const paren_token = try self.base.consume(.RightParen, "Expected ')' after arguments");

        const callee_ptr = try self.base.arena.createNode(ast.Expressions.ExprNode);
        callee_ptr.* = callee;

        return ast.Expressions.ExprNode{
            .Call = ast.Expressions.CallExpr{
                .callee = callee_ptr,
                .arguments = try arguments.toOwnedSlice(self.base.arena.allocator()),
                .type_info = ast.Types.TypeInfo.unknown(), // Return type will be resolved from function signature
                .span = self.base.spanFromToken(paren_token),
            },
        };
    }

    /// Parse field access (obj.field)
    fn parseFieldAccess(self: *ExpressionParser, target: ast.Expressions.ExprNode) ParserError!ast.Expressions.ExprNode {
        const name_token = try self.base.consume(.Identifier, "Expected property name after '.'");

        // Check if this might be an enum literal (EnumType.VariantName)
        if (target == .Identifier) {
            const enum_name = target.Identifier.name;

            // Don't treat standard library and module access as enum literals
            const is_module_access = std.mem.eql(u8, enum_name, "std") or
                std.mem.eql(u8, enum_name, "constants") or
                std.mem.eql(u8, enum_name, "transaction") or
                std.mem.eql(u8, enum_name, "block") or
                std.mem.eql(u8, enum_name, "math");

            if (!is_module_access) {
                // Treat as potential enum literal
                return ast.Expressions.ExprNode{ .EnumLiteral = ast.Expressions.EnumLiteralExpr{
                    .enum_name = enum_name,
                    .variant_name = name_token.lexeme,
                    .span = self.base.spanFromToken(name_token),
                } };
            }
        }

        // Regular field access
        const expr_ptr = try self.base.arena.createNode(ast.Expressions.ExprNode);
        expr_ptr.* = target;
        return ast.Expressions.ExprNode{ .FieldAccess = ast.Expressions.FieldAccessExpr{
            .target = expr_ptr,
            .field = name_token.lexeme,
            .span = self.base.spanFromToken(name_token),
        } };
    }

    /// Parse index access (obj[index] or obj[key1, key2])
    fn parseIndexAccess(self: *ExpressionParser, target: ast.Expressions.ExprNode) ParserError!ast.Expressions.ExprNode {
        const index = try self.parseExpression();

        // Check for double mapping access: target[key1, key2]
        if (self.base.match(.Comma)) {
            const second_index = try self.parseExpression();
            _ = try self.base.consume(.RightBracket, "Expected ']' after double mapping index");

            // Create pointers for the nested structure
            const expr_ptr = try self.base.arena.createNode(ast.Expressions.ExprNode);
            expr_ptr.* = target;
            const index_ptr = try self.base.arena.createNode(ast.Expressions.ExprNode);
            index_ptr.* = index;
            const second_index_ptr = try self.base.arena.createNode(ast.Expressions.ExprNode);
            second_index_ptr.* = second_index;

            // Create a nested index expression for double mapping: target[key1][key2]
            const first_index = ast.Expressions.ExprNode{ .Index = ast.Expressions.IndexExpr{
                .target = expr_ptr,
                .index = index_ptr,
                .span = self.base.spanFromToken(self.base.previous()),
            } };

            const first_index_ptr = try self.base.arena.createNode(ast.Expressions.ExprNode);
            first_index_ptr.* = first_index;

            return ast.Expressions.ExprNode{ .Index = ast.Expressions.IndexExpr{
                .target = first_index_ptr,
                .index = second_index_ptr,
                .span = self.base.spanFromToken(self.base.previous()),
            } };
        } else {
            _ = try self.base.consume(.RightBracket, "Expected ']' after array index");

            const expr_ptr = try self.base.arena.createNode(ast.Expressions.ExprNode);
            expr_ptr.* = target;
            const index_ptr = try self.base.arena.createNode(ast.Expressions.ExprNode);
            index_ptr.* = index;

            return ast.Expressions.ExprNode{ .Index = ast.Expressions.IndexExpr{
                .target = expr_ptr,
                .index = index_ptr,
                .span = self.base.spanFromToken(self.base.previous()),
            } };
        }
    }

    /// Parse primary expressions (literals, identifiers, parentheses) - MIGRATED FROM ORIGINAL
    fn parsePrimary(self: *ExpressionParser) ParserError!ast.Expressions.ExprNode {
        // Boolean literals
        if (self.base.match(.True)) {
            const token = self.base.previous();
            return ast.Expressions.ExprNode{ .Literal = .{ .Bool = ast.expressions.BoolLiteral{
                .value = true,
                .type_info = ast.Types.TypeInfo.explicit(.Bool, .bool, self.base.spanFromToken(token)),
                .span = self.base.spanFromToken(token),
            } } };
        }

        if (self.base.match(.False)) {
            const token = self.base.previous();
            return ast.Expressions.ExprNode{ .Literal = .{ .Bool = ast.expressions.BoolLiteral{
                .value = false,
                .type_info = ast.Types.TypeInfo.explicit(.Bool, .bool, self.base.spanFromToken(token)),
                .span = self.base.spanFromToken(token),
            } } };
        }

        // Number literals
        if (self.base.match(.IntegerLiteral)) {
            const token = self.base.previous();
            const value_copy = try self.base.arena.createString(token.lexeme);
            return ast.Expressions.ExprNode{
                .Literal = .{
                    .Integer = ast.expressions.IntegerLiteral{
                        .value = value_copy,
                        .type_info = ast.Types.CommonTypes.unknown_integer(), // Will be resolved by type resolver
                        .span = self.base.spanFromToken(token),
                    },
                },
            };
        }

        // String literals
        if (self.base.match(.StringLiteral)) {
            const token = self.base.previous();
            const value_copy = try self.base.arena.createString(token.lexeme);
            return ast.Expressions.ExprNode{ .Literal = .{ .String = ast.expressions.StringLiteral{
                .value = value_copy,
                .type_info = ast.Types.TypeInfo.explicit(.String, .string, self.base.spanFromToken(token)),
                .span = self.base.spanFromToken(token),
            } } };
        }

        // Address literals
        if (self.base.match(.AddressLiteral)) {
            const token = self.base.previous();
            return ast.Expressions.ExprNode{ .Literal = .{ .Address = ast.Expressions.AddressLiteral{
                .value = token.lexeme,
                .type_info = ast.Types.TypeInfo.explicit(.Address, .address, self.base.spanFromToken(token)),
                .span = self.base.spanFromToken(token),
            } } };
        }

        // Hex literals
        if (self.base.match(.HexLiteral)) {
            const token = self.base.previous();
            return ast.Expressions.ExprNode{
                .Literal = .{
                    .Hex = ast.Expressions.HexLiteral{
                        .value = token.lexeme,
                        .type_info = ast.Types.CommonTypes.unknown_integer(), // Hex can be various integer types
                        .span = self.base.spanFromToken(token),
                    },
                },
            };
        }

        // Binary literals
        if (self.base.match(.BinaryLiteral)) {
            const token = self.base.previous();
            return ast.Expressions.ExprNode{
                .Literal = .{
                    .Binary = ast.Expressions.BinaryLiteral{
                        .value = token.lexeme,
                        .type_info = ast.Types.CommonTypes.unknown_integer(), // Binary can be various integer types
                        .span = self.base.spanFromToken(token),
                    },
                },
            };
        }

        // Identifiers (including keywords that can be used as identifiers)
        if (self.base.match(.Identifier) or self.base.matchKeywordAsIdentifier()) {
            const token = self.base.previous();

            // Check if this is struct instantiation (identifier followed by {)
            if (self.base.check(.LeftBrace)) {
                return try self.parseStructInstantiation(token);
            }

            // Start with the identifier - store name in arena
            const name_copy = try self.base.arena.createString(token.lexeme);
            var current_expr = ast.Expressions.ExprNode{ .Identifier = ast.Expressions.IdentifierExpr{
                .name = name_copy,
                .type_info = ast.Types.TypeInfo.unknown(),
                .span = self.base.spanFromToken(token),
            } };

            // Handle field access (identifier.field)
            while (self.base.match(.Dot)) {
                const field_token = try self.base.consume(.Identifier, "Expected field name after '.'");

                // Store field name in arena
                const field_name = try self.base.arena.createString(field_token.lexeme);

                // Create field access expression
                const field_expr = ast.Expressions.ExprNode{
                    .FieldAccess = ast.Expressions.FieldAccessExpr{
                        .target = try self.base.arena.createNode(ast.Expressions.ExprNode),
                        .field = field_name,
                        .type_info = ast.Types.TypeInfo.unknown(), // Will be resolved during type checking
                        .span = self.base.spanFromToken(token),
                    },
                };

                // Set the target field
                field_expr.FieldAccess.target.* = current_expr;

                // Update current expression for potential chaining
                current_expr = field_expr;
            }

            return current_expr;
        }

        // Try expressions
        if (self.base.match(.Try)) {
            const try_token = self.base.previous();
            const expr = try self.parseUnary();

            const expr_ptr = try self.base.arena.createNode(ast.Expressions.ExprNode);
            expr_ptr.* = expr;

            return ast.Expressions.ExprNode{ .Try = ast.Expressions.TryExpr{
                .expr = expr_ptr,
                .span = self.base.spanFromToken(try_token),
            } };
        }

        // Switch expressions
        if (self.base.match(.Switch)) {
            return try self.parseSwitchExpression();
        }

        // Parentheses
        if (self.base.match(.LeftParen)) {
            const expr = try self.parseExpression();
            _ = try self.base.consume(.RightParen, "Expected ')' after expression");
            return expr;
        }
        // Quantified expressions: forall/exists i: T (where predicate)? => body
        if (self.base.match(.Forall) or self.base.match(.Exists)) {
            const quant_token = self.base.previous();
            const quantifier: ast.Expressions.QuantifierType = if (quant_token.type == .Forall) .Forall else .Exists;

            // Parse verification attributes if present
            const verification_attributes = try self.parseVerificationAttributes();

            // Bound variable name
            const var_token = try self.base.consume(.Identifier, "Expected bound variable name after quantifier");

            _ = try self.base.consume(.Colon, "Expected ':' after bound variable name");

            // Parse variable type using TypeParser
            var type_parser = TypeParser.init(self.base.tokens, self.base.arena);
            type_parser.base.current = self.base.current;
            const var_type = try type_parser.parseType();
            self.base.current = type_parser.base.current;

            // Optional where clause
            var where_ptr: ?*ast.Expressions.ExprNode = null;
            if (self.base.match(.Where)) {
                const where_expr = try self.parseExpression();
                const tmp_ptr = try self.base.arena.createNode(ast.Expressions.ExprNode);
                tmp_ptr.* = where_expr;
                where_ptr = tmp_ptr;
            }

            _ = try self.base.consume(.Arrow, "Expected '=>' after quantifier header");

            // Parse body expression
            const body_expr = try self.parseExpression();
            const body_ptr = try self.base.arena.createNode(ast.Expressions.ExprNode);
            body_ptr.* = body_expr;

            // Create verification metadata for the quantified expression
            const verification_metadata = try self.base.arena.createNode(ast.Verification.QuantifiedMetadata);
            verification_metadata.* = ast.Verification.QuantifiedMetadata.init(quantifier, var_token.lexeme, var_type, self.base.spanFromToken(quant_token));
            verification_metadata.has_condition = where_ptr != null;

            return ast.Expressions.ExprNode{ .Quantified = ast.Expressions.QuantifiedExpr{
                .quantifier = quantifier,
                .variable = var_token.lexeme,
                .variable_type = var_type,
                .condition = where_ptr,
                .body = body_ptr,
                .span = self.base.spanFromToken(quant_token),
                .verification_metadata = verification_metadata,
                .verification_attributes = verification_attributes,
            } };
        }

        // Old expressions (old(expr) for postconditions)
        if (self.base.match(.Old)) {
            const old_token = self.base.previous();
            _ = try self.base.consume(.LeftParen, "Expected '(' after 'old'");
            const expr = try self.parseExpression();
            _ = try self.base.consume(.RightParen, "Expected ')' after old expression");

            const expr_ptr = try self.base.arena.createNode(ast.Expressions.ExprNode);
            expr_ptr.* = expr;

            return ast.Expressions.ExprNode{ .Old = ast.Expressions.OldExpr{
                .expr = expr_ptr,
                .span = self.base.spanFromToken(old_token),
            } };
        }

        // Comptime expressions (comptime { ... })
        if (self.base.match(.Comptime)) {
            const comptime_token = self.base.previous();
            const block = try self.parseBlock();

            return ast.Expressions.ExprNode{ .Comptime = ast.Expressions.ComptimeExpr{
                .block = block,
                .span = self.base.spanFromToken(comptime_token),
            } };
        }

        // Error expressions (error.SomeError)
        if (self.base.match(.Error)) {
            const error_token = self.base.previous();
            _ = try self.base.consume(.Dot, "Expected '.' after 'error'");
            const name_token = try self.base.consume(.Identifier, "Expected error name after 'error.'");

            return ast.Expressions.ExprNode{ .ErrorReturn = ast.Expressions.ErrorReturnExpr{
                .error_name = name_token.lexeme,
                .span = self.base.spanFromToken(error_token),
            } };
        }

        // Builtin functions starting with @
        if (self.base.match(.At)) {
            return try self.parseBuiltinFunction();
        }

        // Address literals
        if (self.base.match(.AddressLiteral)) {
            const token = self.base.previous();
            return ast.Expressions.ExprNode{ .Literal = .{ .Address = ast.Expressions.AddressLiteral{
                .value = token.lexeme,
                .type_info = ast.Types.TypeInfo.explicit(.Address, .address, self.base.spanFromToken(token)),
                .span = self.base.spanFromToken(token),
            } } };
        }

        // Hex literals
        if (self.base.match(.HexLiteral)) {
            const token = self.base.previous();
            return ast.Expressions.ExprNode{
                .Literal = .{
                    .Hex = ast.Expressions.HexLiteral{
                        .value = token.lexeme,
                        .type_info = ast.Types.CommonTypes.unknown_integer(), // Hex can be various integer types
                        .span = self.base.spanFromToken(token),
                    },
                },
            };
        }

        // Try expressions
        if (self.base.match(.Try)) {
            const try_token = self.base.previous();
            const expr = try self.parseUnary();

            const expr_ptr = try self.base.arena.createNode(ast.Expressions.ExprNode);
            expr_ptr.* = expr;

            return ast.Expressions.ExprNode{ .Try = ast.Expressions.TryExpr{
                .expr = expr_ptr,
                .span = self.base.spanFromToken(try_token),
            } };
        }

        // Switch expressions
        if (self.base.match(.Switch)) {
            return try self.parseSwitchExpression();
        }

        // Old expressions (old(expr) for postconditions)
        if (self.base.match(.Old)) {
            const old_token = self.base.previous();
            _ = try self.base.consume(.LeftParen, "Expected '(' after 'old'");
            const expr = try self.parseExpression();
            _ = try self.base.consume(.RightParen, "Expected ')' after old expression");

            const expr_ptr = try self.base.arena.createNode(ast.Expressions.ExprNode);
            expr_ptr.* = expr;

            return ast.Expressions.ExprNode{ .Old = ast.Expressions.OldExpr{
                .expr = expr_ptr,
                .span = self.base.spanFromToken(old_token),
            } };
        }

        // Error expressions (error.SomeError)
        if (self.base.match(.Error)) {
            const error_token = self.base.previous();
            _ = try self.base.consume(.Dot, "Expected '.' after 'error'");
            const name_token = try self.base.consume(.Identifier, "Expected error name after 'error.'");

            return ast.Expressions.ExprNode{ .ErrorReturn = ast.Expressions.ErrorReturnExpr{
                .error_name = name_token.lexeme,
                .span = self.base.spanFromToken(error_token),
            } };
        }

        // Builtin functions starting with @
        if (self.base.match(.At)) {
            return try self.parseBuiltinFunction();
        }

        // Anonymous struct literals (.{field = value, ...})
        if (self.base.check(.Dot) and self.base.current + 1 < self.base.tokens.len and
            self.base.tokens[self.base.current + 1].type == .LeftBrace)
        {
            _ = self.base.advance(); // consume the dot
            return try self.parseAnonymousStructLiteral();
        }

        // Array literals ([element1, element2, ...])
        if (self.base.match(.LeftBracket)) {
            return try self.parseArrayLiteral();
        }

        // Switch expressions
        if (self.base.match(.Switch)) {
            return try self.parseSwitchExpression();
        }

        // Parentheses or tuples
        if (self.base.match(.LeftParen)) {
            return try self.parseParenthesizedOrTuple();
        }

        // Struct instantiation (handled in parseCall via field access)
        if (self.base.check(.Identifier)) {
            if (self.base.current + 1 < self.base.tokens.len and
                self.base.tokens[self.base.current + 1].type == .LeftBrace)
            {
                const name_token = self.base.advance();
                return try self.parseStructInstantiation(name_token);
            }
        }

        try self.base.errorAtCurrent("Expected expression");
        return error.UnexpectedToken;
    }

    /// Parse switch expression (returns a value) - MIGRATED FROM ORIGINAL
    fn parseSwitchExpression(self: *ExpressionParser) ParserError!ast.Expressions.ExprNode {
        const switch_token = self.base.previous();

        // Parse required switch condition: switch (expr)
        _ = try self.base.consume(.LeftParen, "Expected '(' after 'switch'");
        const condition = try self.parseExpression();
        _ = try self.base.consume(.RightParen, "Expected ')' after switch condition");

        _ = try self.base.consume(.LeftBrace, "Expected '{' after switch condition");

        // parse switch arms

        var cases = std.ArrayList(ast.Switch.Case){};
        defer cases.deinit(self.base.arena.allocator());

        var default_case: ?ast.Statements.BlockNode = null;

        // Parse switch arms
        while (!self.base.check(.RightBrace) and !self.base.isAtEnd()) {
            if (self.base.match(.Else)) {
                // Parse else clause
                _ = try self.base.consume(.Arrow, "Expected '=>' after 'else'");
                // For switch expressions, else body must be an expression
                const else_expr = try self.parseExpressionNoComma();
                const expr_ptr = try self.base.arena.createNode(ast.Expressions.ExprNode);
                expr_ptr.* = else_expr;
                // Synthesize a block node with a single expression statement
                var stmts = try self.base.arena.createSlice(ast.Statements.StmtNode, 1);
                stmts[0] = ast.Statements.StmtNode{ .Expr = expr_ptr.* };
                default_case = ast.Statements.BlockNode{
                    .statements = stmts,
                    .span = self.base.spanFromToken(self.base.previous()),
                };
                // Optional trailing comma after else arm
                _ = self.base.match(.Comma);
                break;
            }

            // Parse pattern for switch expression using common parser
            const pattern = try common_parsers.parseSwitchPattern(&self.base, self);

            // Sync parser state defensively before consuming '=>' and parsing the arm body
            const sync_pos = self.base.current;
            self.base.current = sync_pos;

            _ = try self.base.consume(.Arrow, "Expected '=>' after switch pattern");
            // Defensive: if cursor drifted, ensure any stray '=>' is consumed
            if (self.base.check(.Arrow)) {
                _ = self.base.advance();
            }

            // Parse body directly as an expression for switch expressions.
            // IMPORTANT: Do not allow the comma operator to swallow the next case.
            const before_idx = self.base.current;
            const arm_expr = try self.parseExpressionNoComma();
            const expr_ptr2 = try self.base.arena.createNode(ast.Expressions.ExprNode);
            expr_ptr2.* = arm_expr;
            const body = ast.Switch.Body{ .Expression = expr_ptr2 };

            // no debug output

            // Create switch case
            const case = ast.Switch.Case{
                .pattern = pattern,
                .body = body,
                .span = self.base.spanFromToken(switch_token),
            };

            try cases.append(self.base.arena.allocator(), case);

            // Optional comma between cases
            _ = self.base.match(.Comma);
            // Defensive: if parser did not advance and next is '=>', consume it
            if (self.base.check(.Arrow) and self.base.current == before_idx) {
                _ = self.base.advance();
                _ = self.base.match(.Comma);
            }
            // no debug output
        }

        _ = try self.base.consume(.RightBrace, "Expected '}' after switch cases");

        const condition_ptr = try self.base.arena.createNode(ast.Expressions.ExprNode);
        condition_ptr.* = condition;

        return ast.Expressions.ExprNode{ .SwitchExpression = ast.Switch.ExprNode{
            .condition = condition_ptr,
            .cases = try cases.toOwnedSlice(self.base.arena.allocator()),
            .default_case = default_case,
            .span = self.base.spanFromToken(switch_token),
        } };
    }

    // Using common parseSwitchPattern from common_parsers.zig

    // Using common parseSwitchBody from common_parsers.zig

    /// Parse block (needed for switch bodies)
    fn parseBlock(self: *ExpressionParser) !ast.Statements.BlockNode {
        _ = try self.base.consume(.LeftBrace, "Expected '{'");

        var statements = std.ArrayList(ast.Statements.StmtNode){};
        defer statements.deinit(self.base.arena.allocator());

        while (!self.base.check(.RightBrace) and !self.base.isAtEnd()) {
            // Statement parsing should be handled by statement_parser.zig
            // This is a placeholder until proper integration is implemented
            _ = self.base.advance();
        }

        const end_token = try self.base.consume(.RightBrace, "Expected '}' after block");

        return ast.Statements.BlockNode{
            .statements = try statements.toOwnedSlice(self.base.arena.allocator()),
            .span = self.base.spanFromToken(end_token),
        };
    }

    /// Parse builtin function (@function_name(...)) - MIGRATED FROM ORIGINAL
    fn parseBuiltinFunction(self: *ExpressionParser) ParserError!ast.Expressions.ExprNode {
        const at_token = self.base.previous();
        const name_token = try self.base.consume(.Identifier, "Expected builtin function name after '@'");

        // Check if it's a valid builtin function
        const builtin_name = name_token.lexeme;

        // Handle @cast(Type, expr)
        if (std.mem.eql(u8, builtin_name, "cast")) {
            _ = try self.base.consume(.LeftParen, "Expected '(' after builtin function name");

            // Parse target type via TypeParser
            var type_parser = TypeParser.init(self.base.tokens, self.base.arena);
            type_parser.base.current = self.base.current;
            const target_type = try type_parser.parseType();
            self.base.current = type_parser.base.current;

            _ = try self.base.consume(.Comma, "Expected ',' after type in @cast");

            // Parse operand expression
            const operand_expr = try self.parseExpression();
            _ = try self.base.consume(.RightParen, "Expected ')' after @cast arguments");

            const operand_ptr = try self.base.arena.createNode(ast.Expressions.ExprNode);
            operand_ptr.* = operand_expr;

            return ast.Expressions.ExprNode{ .Cast = ast.Expressions.CastExpr{
                .operand = operand_ptr,
                .target_type = target_type,
                .cast_type = .Unsafe,
                .span = self.base.spanFromToken(at_token),
            } };
        }

        // Handle other builtin functions
        if (std.mem.eql(u8, builtin_name, "divTrunc") or
            std.mem.eql(u8, builtin_name, "divFloor") or
            std.mem.eql(u8, builtin_name, "divCeil") or
            std.mem.eql(u8, builtin_name, "divExact") or
            std.mem.eql(u8, builtin_name, "divmod"))
        {
            _ = try self.base.consume(.LeftParen, "Expected '(' after builtin function name");

            var args = std.ArrayList(*ast.Expressions.ExprNode){};
            defer args.deinit(self.base.arena.allocator());

            if (!self.base.check(.RightParen)) {
                const first_arg = try self.parseExpression();
                const first_arg_ptr = try self.base.arena.createNode(ast.Expressions.ExprNode);
                first_arg_ptr.* = first_arg;
                try args.append(self.base.arena.allocator(), first_arg_ptr);

                while (self.base.match(.Comma)) {
                    const arg = try self.parseExpression();
                    const arg_ptr = try self.base.arena.createNode(ast.Expressions.ExprNode);
                    arg_ptr.* = arg;
                    try args.append(self.base.arena.allocator(), arg_ptr);
                }
            }

            _ = try self.base.consume(.RightParen, "Expected ')' after arguments");

            // Create the builtin function call
            const full_name = try std.fmt.allocPrint(self.base.arena.allocator(), "@{s}", .{builtin_name});

            // Create identifier for the function name
            const name_expr = try self.base.arena.createNode(ast.Expressions.ExprNode);
            name_expr.* = ast.Expressions.ExprNode{ .Identifier = ast.Expressions.IdentifierExpr{
                .name = full_name,
                .type_info = ast.Types.TypeInfo.unknown(),
                .span = self.base.spanFromToken(at_token),
            } };

            return ast.Expressions.ExprNode{
                .Call = ast.Expressions.CallExpr{
                    .callee = name_expr,
                    .arguments = try args.toOwnedSlice(self.base.arena.allocator()),
                    .type_info = ast.Types.TypeInfo.unknown(), // Will be resolved during type checking
                    .span = self.base.spanFromToken(at_token),
                },
            };
        } else {
            try self.base.errorAtCurrent("Unknown builtin function");
            return error.UnexpectedToken;
        }
    }

    /// Parse parenthesized expressions or tuples
    fn parseParenthesizedOrTuple(self: *ExpressionParser) ParserError!ast.Expressions.ExprNode {
        const paren_token = self.base.previous();

        // Check for empty tuple
        if (self.base.check(.RightParen)) {
            _ = self.base.advance();
            var empty_elements = std.ArrayList(*ast.Expressions.ExprNode){};
            defer empty_elements.deinit(self.base.arena.allocator());

            return ast.Expressions.ExprNode{ .Tuple = ast.Expressions.TupleExpr{
                .elements = try empty_elements.toOwnedSlice(self.base.arena.allocator()),
                .span = self.base.spanFromToken(paren_token),
            } };
        }

        const first_expr = try self.parseExpression();

        // Check if it's a tuple (has comma)
        if (self.base.match(.Comma)) {
            var elements = std.ArrayList(*ast.Expressions.ExprNode){};
            defer elements.deinit(self.base.arena.allocator());

            // Convert first_expr to pointer
            const first_ptr = try self.base.arena.createNode(ast.Expressions.ExprNode);
            first_ptr.* = first_expr;
            try elements.append(self.base.arena.allocator(), first_ptr);

            // Handle trailing comma case: (a,)
            if (!self.base.check(.RightParen)) {
                repeat: while (true) {
                    const element = try self.parseExpression();
                    const element_ptr = try self.base.arena.createNode(ast.Expressions.ExprNode);
                    element_ptr.* = element;
                    try elements.append(self.base.arena.allocator(), element_ptr);

                    if (!self.base.match(.Comma)) break :repeat;
                    if (self.base.check(.RightParen)) break :repeat; // Trailing comma
                }
            }

            _ = try self.base.consume(.RightParen, "Expected ')' after tuple elements");

            return ast.Expressions.ExprNode{ .Tuple = ast.Expressions.TupleExpr{
                .elements = try elements.toOwnedSlice(self.base.arena.allocator()),
                .span = self.base.spanFromToken(paren_token),
            } };
        } else {
            // Single parenthesized expression
            _ = try self.base.consume(.RightParen, "Expected ')' after expression");
            return first_expr;
        }
    }

    /// Parse struct instantiation expression (e.g., `MyStruct { a: 1, b: 2 }`)
    fn parseStructInstantiation(self: *ExpressionParser, name_token: Token) ParserError!ast.Expressions.ExprNode {
        _ = try self.base.consume(.LeftBrace, "Expected '{' after struct name");

        var fields = std.ArrayList(ast.Expressions.StructInstantiationField){};
        defer fields.deinit(self.base.arena.allocator());

        // Parse field initializers (field_name: value)
        while (!self.base.check(.RightBrace) and !self.base.isAtEnd()) {
            const field_name = try self.base.consumeIdentifierOrKeyword("Expected field name in struct instantiation");
            _ = try self.base.consume(.Colon, "Expected ':' after field name in struct instantiation");

            const field_value = try self.parseAssignment();
            const field_value_ptr = try self.base.arena.createNode(ast.Expressions.ExprNode);
            field_value_ptr.* = field_value;

            try fields.append(self.base.arena.allocator(), ast.Expressions.StructInstantiationField{
                .name = field_name.lexeme,
                .value = field_value_ptr,
                .span = self.base.spanFromToken(field_name),
            });

            // Optional comma (but don't require it for last field)
            if (!self.base.check(.RightBrace)) {
                if (!self.base.match(.Comma)) {
                    // No comma found, but we're not at the end, so this is an error
                    return error.ExpectedToken;
                }
            } else {
                _ = self.base.match(.Comma); // Consume trailing comma if present
            }
        }

        _ = try self.base.consume(.RightBrace, "Expected '}' after struct instantiation fields");

        // Create the struct name identifier
        const struct_name_ptr = try self.base.arena.createNode(ast.Expressions.ExprNode);
        struct_name_ptr.* = ast.Expressions.ExprNode{ .Identifier = ast.Expressions.IdentifierExpr{
            .name = name_token.lexeme,
            .type_info = ast.Types.TypeInfo.unknown(),
            .span = self.base.spanFromToken(name_token),
        } };

        return ast.Expressions.ExprNode{ .StructInstantiation = ast.Expressions.StructInstantiationExpr{
            .struct_name = struct_name_ptr,
            .fields = try fields.toOwnedSlice(self.base.arena.allocator()),
            .span = self.base.spanFromToken(name_token),
        } };
    }

    /// Parse anonymous struct literal (.{field = value, ...})
    fn parseAnonymousStructLiteral(self: *ExpressionParser) ParserError!ast.Expressions.ExprNode {
        const dot_token = self.base.previous();
        _ = try self.base.consume(.LeftBrace, "Expected '{' after '.' in anonymous struct literal");

        var fields = std.ArrayList(ast.Expressions.AnonymousStructField){};
        defer fields.deinit(self.base.arena.allocator());

        // Parse field initializers (.{ .field = value, ... })
        while (!self.base.check(.RightBrace) and !self.base.isAtEnd()) {
            // Enforce leading dot before each field name
            _ = try self.base.consume(.Dot, "Expected '.' before field name in anonymous struct literal");
            const field_name = try self.base.consume(.Identifier, "Expected field name in anonymous struct literal");
            _ = try self.base.consume(.Equal, "Expected '=' after field name in anonymous struct literal");

            const field_value = try self.parseAssignment();
            const field_value_ptr = try self.base.arena.createNode(ast.Expressions.ExprNode);
            field_value_ptr.* = field_value;

            try fields.append(self.base.arena.allocator(), ast.Expressions.AnonymousStructField{
                .name = field_name.lexeme,
                .value = field_value_ptr,
                .span = self.base.spanFromToken(field_name),
            });

            // Optional comma (but don't require it for last field)
            if (!self.base.check(.RightBrace)) {
                if (!self.base.match(.Comma)) {
                    // No comma found, but we're not at the end, so this is an error
                    return error.ExpectedToken;
                }
            } else {
                _ = self.base.match(.Comma); // Consume trailing comma if present
            }
        }

        _ = try self.base.consume(.RightBrace, "Expected '}' after anonymous struct literal fields");

        return ast.Expressions.ExprNode{ .AnonymousStruct = ast.Expressions.AnonymousStructExpr{
            .fields = try fields.toOwnedSlice(self.base.arena.allocator()),
            .span = self.base.spanFromToken(dot_token),
        } };
    }

    /// Parse array literal ([element1, element2, ...])
    fn parseArrayLiteral(self: *ExpressionParser) ParserError!ast.Expressions.ExprNode {
        const bracket_token = self.base.previous();

        var elements = std.ArrayList(*ast.Expressions.ExprNode){};
        defer elements.deinit(self.base.arena.allocator());

        // Parse array elements
        while (!self.base.check(.RightBracket) and !self.base.isAtEnd()) {
            const element = try self.parseAssignment();
            const element_ptr = try self.base.arena.createNode(ast.Expressions.ExprNode);
            element_ptr.* = element;
            try elements.append(self.base.arena.allocator(), element_ptr);

            // Optional comma (but don't require it for last element)
            if (!self.base.check(.RightBracket)) {
                if (!self.base.match(.Comma)) {
                    // No comma found, but we're not at the end, so this is an error
                    return error.ExpectedToken;
                }
            } else {
                _ = self.base.match(.Comma); // Consume trailing comma if present
            }
        }

        _ = try self.base.consume(.RightBracket, "Expected ']' after array elements");

        return ast.Expressions.ExprNode{
            .ArrayLiteral = ast.Literals.Array{
                .elements = try elements.toOwnedSlice(self.base.arena.allocator()),
                .element_type = null, // Type will be inferred
                .span = self.base.spanFromToken(bracket_token),
            },
        };
    }

    /// Parse verification attributes (e.g., @ora.quantified, @ora.assertion, etc.)
    fn parseVerificationAttributes(self: *ExpressionParser) ParserError![]ast.Verification.VerificationAttribute {
        var attributes = std.ArrayList(ast.Verification.VerificationAttribute){};
        defer attributes.deinit(self.base.arena.allocator());

        // Parse attributes in the format @ora.attribute_name or @ora.attribute_name(value)
        while (self.base.match(.At)) {
            const at_token = self.base.previous();

            // Expect 'ora' namespace
            _ = try self.base.consume(.Identifier, "Expected 'ora' after '@'");
            if (!std.mem.eql(u8, self.base.previous().lexeme, "ora")) {
                try self.base.errorAtCurrent("Expected 'ora' namespace for verification attributes");
                return error.UnexpectedToken;
            }

            _ = try self.base.consume(.Dot, "Expected '.' after 'ora'");

            // Parse attribute name
            const attr_name_token = try self.base.consume(.Identifier, "Expected attribute name after 'ora.'");
            const attr_name = attr_name_token.lexeme;

            // Parse optional value in parentheses
            var attr_value: ?[]const u8 = null;
            if (self.base.match(.LeftParen)) {
                const value_token = try self.base.consume(.String, "Expected string value for attribute");
                attr_value = value_token.lexeme;
                _ = try self.base.consume(.RightParen, "Expected ')' after attribute value");
            }

            // Create verification attribute
            const attr_type = if (std.mem.eql(u8, attr_name, "quantified"))
                ast.Verification.VerificationAttributeType.Quantified
            else if (std.mem.eql(u8, attr_name, "assertion"))
                ast.Verification.VerificationAttributeType.Assertion
            else if (std.mem.eql(u8, attr_name, "invariant"))
                ast.Verification.VerificationAttributeType.Invariant
            else if (std.mem.eql(u8, attr_name, "precondition"))
                ast.Verification.VerificationAttributeType.Precondition
            else if (std.mem.eql(u8, attr_name, "postcondition"))
                ast.Verification.VerificationAttributeType.Postcondition
            else if (std.mem.eql(u8, attr_name, "loop_invariant"))
                ast.Verification.VerificationAttributeType.LoopInvariant
            else
                ast.Verification.VerificationAttributeType.Custom;

            const attr = if (attr_type == .Custom)
                ast.Verification.VerificationAttribute.initCustom(attr_name, attr_value, self.base.spanFromToken(at_token))
            else
                ast.Verification.VerificationAttribute.init(attr_type, self.base.spanFromToken(at_token));

            try attributes.append(self.base.arena.allocator(), attr);
        }

        return try attributes.toOwnedSlice(self.base.arena.allocator());
    }

    /// Parse a range expression (start...end)
    /// This creates a RangeExpr which can be used both in switch patterns and directly as expressions
    pub fn parseRangeExpression(self: *ExpressionParser) ParserError!ast.Expressions.ExprNode {
        const start_token = self.base.peek();

        // Parse start expression
        const start_expr = try self.parseExpression();

        _ = try self.base.consume(.DotDotDot, "Expected '...' in range expression");

        // Parse end expression
        const end_expr = try self.parseExpression();

        // Create pointers to the expressions
        const start_ptr = try self.base.arena.createNode(ast.Expressions.ExprNode);
        start_ptr.* = start_expr;
        const end_ptr = try self.base.arena.createNode(ast.Expressions.ExprNode);
        end_ptr.* = end_expr;

        return ast.Expressions.ExprNode{
            .Range = ast.Expressions.RangeExpr{
                .start = start_ptr,
                .end = end_ptr,
                .inclusive = true, // Default to inclusive range
                .type_info = TypeInfo.unknown(), // Type will be inferred
                .span = self.base.spanFromToken(start_token),
            },
        };
    }
};
