// ============================================================================
// Expression Parser
// ============================================================================
//
// Implements expression parsing using precedence climbing.
//
// PRECEDENCE LEVELS (low to high):
//   Comma → Assignment → Logical OR/AND → Bitwise OR/XOR/AND →
//   Equality → Comparison → Shifts → Add/Sub → Mul/Div/Mod →
//   Exponentiation → Unary → Postfix → Primary
//
// SECTIONS:
//   • Setup & entry points
//   • Precedence chain (17 levels)
//   • Unary & postfix operators
//   • Primary expressions (literals, identifiers)
//   • Complex expressions (switch, struct, array, etc.)
//
// ============================================================================

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

// Import precedence chain
const precedence = @import("expressions/precedence.zig");

// Import complex expressions
const complex = @import("expressions/complex.zig");

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
        // Ora doesn't use the comma operator for expressions; treat commas as list separators.
        return precedence.parseAssignment(self);
    }

    /// Parse an expression that must NOT consume top-level commas.
    ///
    /// This is used in contexts where a comma is a list/case separator
    /// (e.g., switch expression arms, function arg separators handled elsewhere),
    /// so we start from assignment-precedence instead of comma-precedence.
    pub fn parseExpressionNoComma(self: *ExpressionParser) ParserError!ast.Expressions.ExprNode {
        return precedence.parseAssignment(self);
    }

    /// Parse logical OR expression (public for use by other parsers)
    pub fn parseLogicalOr(self: *ExpressionParser) ParserError!ast.Expressions.ExprNode {
        return precedence.parseLogicalOr(self);
    }

    // ========================================================================
    // unary & POSTFIX OPERATORS
    // ========================================================================

    /// Parse function calls and member access
    /// Made public so precedence module can call it
    pub fn parseCall(self: *ExpressionParser) ParserError!ast.Expressions.ExprNode {
        var expr = try self.parsePrimary();

        while (true) {
            if (self.base.match(.LeftParen)) {
                expr = try self.finishCall(expr);
                // Generic struct instantiation: Pair(u256) { ... }
                if (self.base.check(.LeftBrace)) {
                    if (expr == .Call and expr.Call.callee.* == .Identifier) {
                        return try complex.parseGenericStructInstantiation(self, &expr.Call);
                    }
                }
            } else if (self.base.check(.DotDot) or self.base.check(.DotDotDot)) {
                // this is a range expression - parse it
                const start_expr = expr;
                const is_inclusive = self.base.check(.DotDotDot);
                _ = self.base.advance(); // Consume DotDot or DotDotDot token
                // parse end expression (but don't consume beyond it - stop at closing paren, etc.)
                // use parseLogicalOr to avoid consuming too much (stops at comma, paren, etc.)
                const end_expr = try precedence.parseLogicalOr(self);
                const start_ptr = try self.base.arena.createNode(ast.Expressions.ExprNode);
                start_ptr.* = start_expr;
                const end_ptr = try self.base.arena.createNode(ast.Expressions.ExprNode);
                end_ptr.* = end_expr;
                expr = ast.Expressions.ExprNode{
                    .Range = ast.Expressions.RangeExpr{
                        .start = start_ptr,
                        .end = end_ptr,
                        .inclusive = is_inclusive,
                        .type_info = ast.Types.TypeInfo.unknown(),
                        .span = self.base.spanFromToken(self.base.previous()),
                    },
                };
                // break out of the loop after parsing range - ranges are complete expressions
                break;
            } else if (self.base.match(.Dot)) {
                const name_token = try self.base.consume(.Identifier, "Expected property name after '.'");

                // check if this might be an enum literal (EnumType.VariantName)
                // but exclude known module/namespace patterns
                // also, if followed by '=' (assignment), treat as field access (L-value)
                const is_assignment_target = self.base.check(.Equal) or
                    self.base.check(.PlusEqual) or self.base.check(.MinusEqual) or
                    self.base.check(.StarEqual) or self.base.check(.SlashEqual) or
                    self.base.check(.PercentEqual);

                if (expr == .Identifier) {
                    const enum_name = expr.Identifier.name;

                    // don't treat standard library and module access as enum literals
                    const is_module_access = std.mem.eql(u8, enum_name, "std") or
                        std.mem.eql(u8, enum_name, "constants") or
                        std.mem.eql(u8, enum_name, "transaction") or
                        std.mem.eql(u8, enum_name, "block") or
                        std.mem.eql(u8, enum_name, "math");

                    // always treat as field access if it's an assignment target (L-value)
                    // this allows field assignment like user.balance = value
                    if (is_module_access or is_assignment_target) {
                        // treat as field access for module patterns or assignment targets (L-values)
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
                        // treat as potential enum literal (only when NOT an assignment target)
                        const variant_name = try self.base.arena.createString(name_token.lexeme);
                        expr = ast.Expressions.ExprNode{ .EnumLiteral = ast.Expressions.EnumLiteralExpr{
                            .enum_name = enum_name,
                            .variant_name = variant_name,
                            .span = self.base.spanFromToken(name_token),
                        } };
                    }
                } else {
                    // complex expressions are always field access
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

                // check for double mapping access: target[key1, key2]
                if (self.base.match(.Comma)) {
                    const second_index = try self.parseExpression();
                    _ = try self.base.consume(.RightBracket, "Expected ']' after double mapping index");

                    // create pointers for the nested structure
                    const expr_ptr = try self.base.arena.createNode(ast.Expressions.ExprNode);
                    expr_ptr.* = expr;
                    const index_ptr = try self.base.arena.createNode(ast.Expressions.ExprNode);
                    index_ptr.* = index;
                    const second_index_ptr = try self.base.arena.createNode(ast.Expressions.ExprNode);
                    second_index_ptr.* = second_index;

                    // create a nested index expression for double mapping: target[key1][key2]
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

    // note: '@cast(Type, expr)' is the only supported cast syntax.

    /// Finish parsing a function call
    fn finishCall(self: *ExpressionParser, callee: ast.Expressions.ExprNode) ParserError!ast.Expressions.ExprNode {
        var arguments = std.ArrayList(*ast.Expressions.ExprNode){};
        defer arguments.deinit(self.base.arena.allocator());

        if (!self.base.check(.RightParen)) {
            repeat: while (true) {
                // Use parseExpressionNoComma to avoid consuming commas as binary operators
                const arg = try self.parseExpressionNoComma();
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
        var name_token: Token = undefined;
        var is_numeric = false;
        if (self.base.check(.IntegerLiteral)) {
            name_token = self.base.advance();
            is_numeric = true;
        } else {
            name_token = try self.base.consume(.Identifier, "Expected property name after '.'");
        }

        // check if this might be an enum literal (EnumType.VariantName)
        // but if followed by '=' (assignment), treat as field access (L-value)
        const is_assignment_target = self.base.check(.Equal) or
            self.base.check(.PlusEqual) or self.base.check(.MinusEqual) or
            self.base.check(.StarEqual) or self.base.check(.SlashEqual) or
            self.base.check(.PercentEqual);

        if (!is_numeric and target == .Identifier) {
            const enum_name = target.Identifier.name;

            // don't treat standard library and module access as enum literals
            const is_module_access = std.mem.eql(u8, enum_name, "std") or
                std.mem.eql(u8, enum_name, "constants") or
                std.mem.eql(u8, enum_name, "transaction") or
                std.mem.eql(u8, enum_name, "block") or
                std.mem.eql(u8, enum_name, "math");

            if (!is_module_access and !is_assignment_target) {
                // treat as potential enum literal (only if not an assignment target)
                return ast.Expressions.ExprNode{ .EnumLiteral = ast.Expressions.EnumLiteralExpr{
                    .enum_name = enum_name,
                    .variant_name = name_token.lexeme,
                    .span = self.base.spanFromToken(name_token),
                } };
            }
        }

        // regular field access (for modules, assignment targets, or complex expressions)
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

        // check for double mapping access: target[key1, key2]
        if (self.base.match(.Comma)) {
            const second_index = try self.parseExpression();
            _ = try self.base.consume(.RightBracket, "Expected ']' after double mapping index");

            // create pointers for the nested structure
            const expr_ptr = try self.base.arena.createNode(ast.Expressions.ExprNode);
            expr_ptr.* = target;
            const index_ptr = try self.base.arena.createNode(ast.Expressions.ExprNode);
            index_ptr.* = index;
            const second_index_ptr = try self.base.arena.createNode(ast.Expressions.ExprNode);
            second_index_ptr.* = second_index;

            // create a nested index expression for double mapping: target[key1][key2]
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

    // ========================================================================
    // primary EXPRESSIONS (literals, identifiers, etc.)
    // ========================================================================

    /// Parse primary expressions (literals, identifiers, parentheses)
    pub fn parsePrimary(self: *ExpressionParser) ParserError!ast.Expressions.ExprNode {
        // quantified expressions (forall, exists) - Formal verification
        if (self.base.match(.Forall) or self.base.match(.Exists)) {
            return try complex.parseQuantifiedExpression(self);
        }

        // boolean literals
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

        // number literals
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

        // string literals
        if (self.base.match(.StringLiteral)) {
            const token = self.base.previous();
            const value_copy = try self.base.arena.createString(token.lexeme);
            return ast.Expressions.ExprNode{ .Literal = .{ .String = ast.expressions.StringLiteral{
                .value = value_copy,
                .type_info = ast.Types.TypeInfo.explicit(.String, .string, self.base.spanFromToken(token)),
                .span = self.base.spanFromToken(token),
            } } };
        }

        // address literals
        if (self.base.match(.AddressLiteral)) {
            const token = self.base.previous();
            return ast.Expressions.ExprNode{ .Literal = .{ .Address = ast.Expressions.AddressLiteral{
                .value = token.lexeme,
                .type_info = ast.Types.TypeInfo.explicit(.Address, .address, self.base.spanFromToken(token)),
                .span = self.base.spanFromToken(token),
            } } };
        }

        // hex literals
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

        // bytes literals (hex"...")
        if (self.base.match(.BytesLiteral)) {
            const token = self.base.previous();
            // extract hex content from token value (token.value.string contains just the hex content)
            const hex_content = if (token.value) |val| switch (val) {
                .string => |s| s,
                else => token.lexeme,
            } else token.lexeme;
            // token.value.string should already contain just the hex digits (no quotes)
            // but if it doesn't, strip any remaining quotes
            const clean_content = if (std.mem.endsWith(u8, hex_content, "\""))
                hex_content[0 .. hex_content.len - 1]
            else
                hex_content;
            const value_copy = try self.base.arena.createString(clean_content);
            return ast.Expressions.ExprNode{ .Literal = .{ .Bytes = ast.Expressions.BytesLiteral{
                .value = value_copy,
                .type_info = ast.Types.TypeInfo.explicit(.Bytes, .bytes, self.base.spanFromToken(token)),
                .span = self.base.spanFromToken(token),
            } } };
        }

        // binary literals
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

        // type keywords as expressions (for `comptime T: type` arguments)
        if (self.matchTypeKeyword()) |type_token| {
            const name_copy = try self.base.arena.createString(type_token.lexeme);
            return ast.Expressions.ExprNode{ .Identifier = ast.Expressions.IdentifierExpr{
                .name = name_copy,
                .type_info = ast.Types.TypeInfo.explicit(.Type, .{ .@"type" = {} }, self.base.spanFromToken(type_token)),
                .span = self.base.spanFromToken(type_token),
            } };
        }

        // identifiers (including keywords that can be used as identifiers)
        if (self.base.match(.Identifier) or self.base.matchKeywordAsIdentifier()) {
            const token = self.base.previous();

            // check if this is a labeled block expression (label: { ... })
            if (self.base.check(.Colon) and self.base.current + 1 < self.base.tokens.len and
                self.base.tokens[self.base.current + 1].type == .LeftBrace)
            {
                _ = self.base.advance(); // consume ':'
                const block = try self.parseBlock();
                return ast.Expressions.ExprNode{ .LabeledBlock = ast.Expressions.LabeledBlockExpr{
                    .label = try self.base.arena.createString(token.lexeme),
                    .block = block,
                    .span = self.base.spanFromToken(token),
                } };
            }

            // check if this is struct instantiation (identifier followed by {)
            if (self.base.check(.LeftBrace)) {
                return try complex.parseStructInstantiation(self, token);
            }

            // start with the identifier - store name in arena
            const name_copy = try self.base.arena.createString(token.lexeme);
            var current_expr = ast.Expressions.ExprNode{ .Identifier = ast.Expressions.IdentifierExpr{
                .name = name_copy,
                .type_info = ast.Types.TypeInfo.unknown(),
                .span = self.base.spanFromToken(token),
            } };

            // handle field access (identifier.field) or enum literal (EnumType.VariantName)
            while (self.base.match(.Dot)) {
                var field_token: Token = undefined;
                var is_numeric = false;
                if (self.base.check(.IntegerLiteral)) {
                    field_token = self.base.advance();
                    is_numeric = true;
                } else {
                    field_token = try self.base.consume(.Identifier, "Expected field name after '.'");
                }

                // check if this might be an enum literal (EnumType.VariantName)
                // but if followed by '=' (assignment), treat as field access (L-value)
                const is_assignment_target = self.base.check(.Equal) or
                    self.base.check(.PlusEqual) or self.base.check(.MinusEqual) or
                    self.base.check(.StarEqual) or self.base.check(.SlashEqual) or
                    self.base.check(.PercentEqual);

                if (!is_numeric and current_expr == .Identifier) {
                    const enum_name = current_expr.Identifier.name;

                    // don't treat standard library and module access as enum literals
                    const is_module_access = std.mem.eql(u8, enum_name, "std") or
                        std.mem.eql(u8, enum_name, "constants") or
                        std.mem.eql(u8, enum_name, "transaction") or
                        std.mem.eql(u8, enum_name, "block") or
                        std.mem.eql(u8, enum_name, "math");

                    if (!is_module_access and !is_assignment_target) {
                        if (std.mem.eql(u8, enum_name, "error")) {
                            var parameters: ?[]*ast.Expressions.ExprNode = null;
                            if (self.base.match(.LeftParen)) {
                                var args = std.ArrayList(*ast.Expressions.ExprNode){};
                                defer args.deinit(self.base.arena.allocator());

                                if (!self.base.check(.RightParen)) {
                                    repeat: while (true) {
                                        const arg = try self.parseExpression();
                                        const arg_ptr = try self.base.arena.createNode(ast.Expressions.ExprNode);
                                        arg_ptr.* = arg;
                                        try args.append(self.base.arena.allocator(), arg_ptr);
                                        if (!self.base.match(.Comma)) break :repeat;
                                    }
                                }

                                _ = try self.base.consume(.RightParen, "Expected ')' after error parameters");
                                parameters = try args.toOwnedSlice(self.base.arena.allocator());
                            }

                            return ast.Expressions.ExprNode{ .ErrorReturn = ast.Expressions.ErrorReturnExpr{
                                .error_name = field_token.lexeme,
                                .parameters = parameters,
                                .span = self.base.spanFromToken(field_token),
                            } };
                        }

                        // treat as potential enum literal (only if not an assignment target)
                        const variant_name = try self.base.arena.createString(field_token.lexeme);
                        current_expr = ast.Expressions.ExprNode{ .EnumLiteral = ast.Expressions.EnumLiteralExpr{
                            .enum_name = enum_name,
                            .variant_name = variant_name,
                            .span = self.base.spanFromToken(field_token),
                        } };
                        continue;
                    }
                }

                // regular field access
                const field_name = try self.base.arena.createString(field_token.lexeme);
                const field_expr = ast.Expressions.ExprNode{
                    .FieldAccess = ast.Expressions.FieldAccessExpr{
                        .target = try self.base.arena.createNode(ast.Expressions.ExprNode),
                        .field = field_name,
                        .type_info = ast.Types.TypeInfo.unknown(), // Will be resolved during type checking
                        .span = self.base.spanFromToken(field_token),
                    },
                };

                // set the target field
                field_expr.FieldAccess.target.* = current_expr;

                // update current expression for potential chaining
                current_expr = field_expr;
            }

            return current_expr;
        }

        // try expressions
        if (self.base.match(.Try)) {
            const try_token = self.base.previous();
            const expr = try precedence.parseUnary(self);

            const expr_ptr = try self.base.arena.createNode(ast.Expressions.ExprNode);
            expr_ptr.* = expr;

            return ast.Expressions.ExprNode{ .Try = ast.Expressions.TryExpr{
                .expr = expr_ptr,
                .span = self.base.spanFromToken(try_token),
            } };
        }

        // switch expressions
        if (self.base.match(.Switch)) {
            return try complex.parseSwitchExpression(self);
        }

        // quantified expressions: forall/exists i: T (where predicate)? => body
        if (self.base.match(.Forall) or self.base.match(.Exists)) {
            const quant_token = self.base.previous();
            const quantifier: ast.Expressions.QuantifierType = if (quant_token.type == .Forall) .Forall else .Exists;

            // parse verification attributes if present
            const verification_attributes = try self.parseVerificationAttributes();

            // bound variable name
            const var_token = try self.base.consume(.Identifier, "Expected bound variable name after quantifier");

            _ = try self.base.consume(.Colon, "Expected ':' after bound variable name");

            // parse variable type using TypeParser
            var type_parser = TypeParser.init(self.base.tokens, self.base.arena);
            type_parser.base.current = self.base.current;
            const var_type = try type_parser.parseType();
            self.base.current = type_parser.base.current;

            // optional where clause
            var where_ptr: ?*ast.Expressions.ExprNode = null;
            if (self.base.match(.Where)) {
                const where_expr = try self.parseExpression();
                const tmp_ptr = try self.base.arena.createNode(ast.Expressions.ExprNode);
                tmp_ptr.* = where_expr;
                where_ptr = tmp_ptr;
            }

            _ = try self.base.consume(.Arrow, "Expected '=>' after quantifier header");

            // parse body expression
            const body_expr = try self.parseExpression();
            const body_ptr = try self.base.arena.createNode(ast.Expressions.ExprNode);
            body_ptr.* = body_expr;

            // create verification metadata for the quantified expression
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

        // old expressions (old(expr) for postconditions)
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

        // comptime expressions (comptime { ... })
        if (self.base.match(.Comptime)) {
            const comptime_token = self.base.previous();
            const block = try self.parseBlock();

            return ast.Expressions.ExprNode{ .Comptime = ast.Expressions.ComptimeExpr{
                .block = block,
                .span = self.base.spanFromToken(comptime_token),
                .type_info = ast.type_info.TypeInfo.unknown(),
            } };
        }

        // error expressions (error.SomeError or error.SomeError(args))
        if (self.base.match(.Error)) {
            const error_token = self.base.previous();
            _ = try self.base.consume(.Dot, "Expected '.' after 'error'");
            const name_token = try self.base.consume(.Identifier, "Expected error name after 'error.'");

            // check if there are parameters (error.SomeError(args))
            var parameters: ?[]*ast.Expressions.ExprNode = null;
            if (self.base.match(.LeftParen)) {
                var args = std.ArrayList(*ast.Expressions.ExprNode){};
                defer args.deinit(self.base.arena.allocator());

                if (!self.base.check(.RightParen)) {
                    repeat: while (true) {
                        const arg = try self.parseExpression();
                        const arg_ptr = try self.base.arena.createNode(ast.Expressions.ExprNode);
                        arg_ptr.* = arg;
                        try args.append(self.base.arena.allocator(), arg_ptr);
                        if (!self.base.match(.Comma)) break :repeat;
                    }
                }

                _ = try self.base.consume(.RightParen, "Expected ')' after error parameters");
                parameters = try args.toOwnedSlice(self.base.arena.allocator());
            }

            return ast.Expressions.ExprNode{ .ErrorReturn = ast.Expressions.ErrorReturnExpr{
                .error_name = name_token.lexeme,
                .parameters = parameters,
                .span = self.base.spanFromToken(error_token),
            } };
        }

        // builtin functions starting with @
        if (self.base.match(.At)) {
            return try complex.parseBuiltinFunction(self);
        }

        // address literals
        if (self.base.match(.AddressLiteral)) {
            const token = self.base.previous();
            return ast.Expressions.ExprNode{ .Literal = .{ .Address = ast.Expressions.AddressLiteral{
                .value = token.lexeme,
                .type_info = ast.Types.TypeInfo.explicit(.Address, .address, self.base.spanFromToken(token)),
                .span = self.base.spanFromToken(token),
            } } };
        }

        // hex literals
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

        // try expressions
        if (self.base.match(.Try)) {
            const try_token = self.base.previous();
            const expr = try precedence.parseUnary(self);

            const expr_ptr = try self.base.arena.createNode(ast.Expressions.ExprNode);
            expr_ptr.* = expr;

            return ast.Expressions.ExprNode{ .Try = ast.Expressions.TryExpr{
                .expr = expr_ptr,
                .span = self.base.spanFromToken(try_token),
            } };
        }

        // switch expressions
        if (self.base.match(.Switch)) {
            return try complex.parseSwitchExpression(self);
        }

        // old expressions (old(expr) for postconditions)
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

        // builtin functions starting with @
        if (self.base.match(.At)) {
            return try complex.parseBuiltinFunction(self);
        }

        // anonymous struct literals (.{field = value, ...})
        if (self.base.check(.Dot) and self.base.current + 1 < self.base.tokens.len and
            self.base.tokens[self.base.current + 1].type == .LeftBrace)
        {
            _ = self.base.advance(); // consume the dot
            return try complex.parseAnonymousStructLiteral(self);
        }

        // array literals ([element1, element2, ...])
        if (self.base.match(.LeftBracket)) {
            return try complex.parseArrayLiteral(self);
        }

        // switch expressions
        if (self.base.match(.Switch)) {
            return try complex.parseSwitchExpression(self);
        }

        // parentheses or tuples
        if (self.base.match(.LeftParen)) {
            return try complex.parseParenthesizedOrTuple(self);
        }

        // struct instantiation (handled in parseCall via field access)
        if (self.base.check(.Identifier)) {
            if (self.base.current + 1 < self.base.tokens.len and
                self.base.tokens[self.base.current + 1].type == .LeftBrace)
            {
                const name_token = self.base.advance();
                return try complex.parseStructInstantiation(self, name_token);
            }
        }

        try self.base.errorAtCurrent("Expected expression");
        return error.UnexpectedToken;
    }

    /// Parse verification attributes (e.g., @ora.quantified, @ora.assertion, etc.)
    fn parseVerificationAttributes(self: *ExpressionParser) ParserError![]ast.Verification.VerificationAttribute {
        var attributes = std.ArrayList(ast.Verification.VerificationAttribute){};
        defer attributes.deinit(self.base.arena.allocator());

        // parse attributes in the format @ora.attribute_name or @ora.attribute_name(value)
        while (self.base.match(.At)) {
            const at_token = self.base.previous();

            // expect 'ora' namespace
            _ = try self.base.consume(.Identifier, "Expected 'ora' after '@'");
            if (!std.mem.eql(u8, self.base.previous().lexeme, "ora")) {
                try self.base.errorAtCurrent("Expected 'ora' namespace for verification attributes");
                return error.UnexpectedToken;
            }

            _ = try self.base.consume(.Dot, "Expected '.' after 'ora'");

            // parse attribute name
            const attr_name_token = try self.base.consume(.Identifier, "Expected attribute name after 'ora.'");
            const attr_name = attr_name_token.lexeme;

            // parse optional value in parentheses
            var attr_value: ?[]const u8 = null;
            if (self.base.match(.LeftParen)) {
                const value_token = try self.base.consume(.String, "Expected string value for attribute");
                attr_value = value_token.lexeme;
                _ = try self.base.consume(.RightParen, "Expected ')' after attribute value");
            }

            // create verification attribute
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
    /// Public for use by other parsers (e.g., switch pattern parsing)
    pub fn parseRangeExpression(self: *ExpressionParser) ParserError!ast.Expressions.ExprNode {
        return complex.parseRangeExpression(self);
    }

    /// Parse a block (for comptime expressions and labeled blocks)
    /// Delegates to StatementParser for full statement support including if, while, etc.
    fn parseBlock(self: *ExpressionParser) ParserError!ast.Statements.BlockNode {
        const StatementParser = @import("statement_parser.zig").StatementParser;
        var stmt_parser = StatementParser.init(self.base.tokens, self.base.arena);
        stmt_parser.base.current = self.base.current;
        stmt_parser.base.file_id = self.base.file_id;
        stmt_parser.syncSubParsers();

        const block = try stmt_parser.parseBlock();
        self.base.current = stmt_parser.base.current;
        return block;
    }

    /// Match a type keyword token (u8, u256, bool, etc.) in expression position.
    /// Returns the token if matched, null otherwise.
    fn matchTypeKeyword(self: *ExpressionParser) ?Token {
        const tt = self.base.peek().type;
        const is_type_kw = switch (tt) {
            .U8, .U16, .U32, .U64, .U128, .U256, .I8, .I16, .I32, .I64, .I128, .I256, .Bool, .Address, .String, .Bytes => true,
            else => false,
        };
        if (is_type_kw) {
            return self.base.advance();
        }
        return null;
    }
};
