// ============================================================================
// Complex Expression Parsers
// ============================================================================
//
// Handles parsing of complex expression types:
//   • Switch expressions
//   • Builtin functions (@cast, @divTrunc, etc.)
//   • Struct instantiation and anonymous struct literals
//   • Array literals
//   • Range expressions
//   • Quantified expressions (forall, exists)
//   • Parenthesized expressions and tuples
//
// ============================================================================

const std = @import("std");
const ast = @import("../../ast.zig");
const TypeInfo = @import("../../ast/type_info.zig").TypeInfo;
const common_parsers = @import("../common_parsers.zig");
const TypeParser = @import("../type_parser.zig").TypeParser;
const ParserError = @import("../parser_core.zig").ParserError;
const precedence = @import("precedence.zig");

// Forward declaration - ExpressionParser is defined in expression_parser.zig
const ExpressionParser = @import("../expression_parser.zig").ExpressionParser;
const Token = @import("../../lexer.zig").Token;

/// Parse switch expression (returns a value)
pub fn parseSwitchExpression(parser: *ExpressionParser) ParserError!ast.Expressions.ExprNode {
    const switch_token = parser.base.previous();

    // parse required switch condition: switch (expr)
    _ = try parser.base.consume(.LeftParen, "Expected '(' after 'switch'");
    const condition = try parser.parseExpression();
    _ = try parser.base.consume(.RightParen, "Expected ')' after switch condition");

    _ = try parser.base.consume(.LeftBrace, "Expected '{' after switch condition");

    // parse switch arms

    var cases = std.ArrayList(ast.Switch.Case){};
    defer cases.deinit(parser.base.arena.allocator());

    var default_case: ?ast.Statements.BlockNode = null;

    // parse switch arms
    while (!parser.base.check(.RightBrace) and !parser.base.isAtEnd()) {
        if (parser.base.match(.Else)) {
            // parse else clause
            _ = try parser.base.consume(.Arrow, "Expected '=>' after 'else'");
            // for switch expressions, else body must be an expression
            const else_expr = try parser.parseExpressionNoComma();
            const expr_ptr = try parser.base.arena.createNode(ast.Expressions.ExprNode);
            expr_ptr.* = else_expr;
            // synthesize a block node with a single expression statement
            var stmts = try parser.base.arena.createSlice(ast.Statements.StmtNode, 1);
            stmts[0] = ast.Statements.StmtNode{ .Expr = expr_ptr.* };
            default_case = ast.Statements.BlockNode{
                .statements = stmts,
                .span = parser.base.spanFromToken(parser.base.previous()),
            };
            // optional trailing comma after else arm
            _ = parser.base.match(.Comma);
            break;
        }

        // parse pattern for switch expression using common parser
        const pattern = try common_parsers.parseSwitchPattern(&parser.base, parser);

        // sync parser state defensively before consuming '=>' and parsing the arm body
        const sync_pos = parser.base.current;
        parser.base.current = sync_pos;

        _ = try parser.base.consume(.Arrow, "Expected '=>' after switch pattern");
        // defensive: if cursor drifted, ensure any stray '=>' is consumed
        if (parser.base.check(.Arrow)) {
            _ = parser.base.advance();
        }

        // parse body directly as an expression for switch expressions.
        // important: Do not allow the comma operator to swallow the next case.
        const before_idx = parser.base.current;
        const arm_expr = try parser.parseExpressionNoComma();
        const expr_ptr2 = try parser.base.arena.createNode(ast.Expressions.ExprNode);
        expr_ptr2.* = arm_expr;
        const body = ast.Switch.Body{ .Expression = expr_ptr2 };

        // create switch case
        const case = ast.Switch.Case{
            .pattern = pattern,
            .body = body,
            .span = parser.base.spanFromToken(switch_token),
        };

        try cases.append(parser.base.arena.allocator(), case);

        // optional comma between cases
        _ = parser.base.match(.Comma);
        // defensive: if parser did not advance and next is '=>', consume it
        if (parser.base.check(.Arrow) and parser.base.current == before_idx) {
            _ = parser.base.advance();
            _ = parser.base.match(.Comma);
        }
    }

    _ = try parser.base.consume(.RightBrace, "Expected '}' after switch cases");

    const condition_ptr = try parser.base.arena.createNode(ast.Expressions.ExprNode);
    condition_ptr.* = condition;

    return ast.Expressions.ExprNode{ .SwitchExpression = ast.Switch.ExprNode{
        .condition = condition_ptr,
        .cases = try cases.toOwnedSlice(parser.base.arena.allocator()),
        .default_case = default_case,
        .span = parser.base.spanFromToken(switch_token),
    } };
}

/// Parse builtin function (@function_name(...))
pub fn parseBuiltinFunction(parser: *ExpressionParser) ParserError!ast.Expressions.ExprNode {
    const at_token = parser.base.previous();
    const name_token = try parser.base.consume(.Identifier, "Expected builtin function name after '@'");

    // check if it's a valid builtin function
    const builtin_name = name_token.lexeme;

    if (std.mem.eql(u8, builtin_name, "lock") or std.mem.eql(u8, builtin_name, "unlock")) {
        try parser.base.errorAtCurrent("@lock/@unlock are statement-only; use @lock(path); or @unlock(path);");
        return error.UnexpectedToken;
    }

    // handle @cast(Type, expr)
    if (std.mem.eql(u8, builtin_name, "cast")) {
        _ = try parser.base.consume(.LeftParen, "Expected '(' after builtin function name");

        // parse target type via TypeParser
        var type_parser = TypeParser.init(parser.base.tokens, parser.base.arena);
        type_parser.base.current = parser.base.current;
        const target_type = try type_parser.parseType();
        parser.base.current = type_parser.base.current;

        _ = try parser.base.consume(.Comma, "Expected ',' after type in @cast");

        // parse operand expression
        const operand_expr = try parser.parseExpression();
        _ = try parser.base.consume(.RightParen, "Expected ')' after @cast arguments");

        const operand_ptr = try parser.base.arena.createNode(ast.Expressions.ExprNode);
        operand_ptr.* = operand_expr;

        return ast.Expressions.ExprNode{ .Cast = ast.Expressions.CastExpr{
            .operand = operand_ptr,
            .target_type = target_type,
            .cast_type = .Unsafe,
            .span = parser.base.spanFromToken(at_token),
        } };
    }

    // handle other builtin functions
    if (std.mem.eql(u8, builtin_name, "divTrunc") or
        std.mem.eql(u8, builtin_name, "divFloor") or
        std.mem.eql(u8, builtin_name, "divCeil") or
        std.mem.eql(u8, builtin_name, "divExact") or
        std.mem.eql(u8, builtin_name, "divmod") or
        std.mem.eql(u8, builtin_name, "addWithOverflow") or
        std.mem.eql(u8, builtin_name, "subWithOverflow") or
        std.mem.eql(u8, builtin_name, "mulWithOverflow") or
        std.mem.eql(u8, builtin_name, "divWithOverflow") or
        std.mem.eql(u8, builtin_name, "modWithOverflow") or
        std.mem.eql(u8, builtin_name, "negWithOverflow") or
        std.mem.eql(u8, builtin_name, "shlWithOverflow") or
        std.mem.eql(u8, builtin_name, "shrWithOverflow") or
        std.mem.eql(u8, builtin_name, "truncate"))
    {
        _ = try parser.base.consume(.LeftParen, "Expected '(' after builtin function name");

        var args = std.ArrayList(*ast.Expressions.ExprNode){};
        defer args.deinit(parser.base.arena.allocator());

        if (!parser.base.check(.RightParen)) {
            const first_arg = try parser.parseExpression();
            const first_arg_ptr = try parser.base.arena.createNode(ast.Expressions.ExprNode);
            first_arg_ptr.* = first_arg;
            try args.append(parser.base.arena.allocator(), first_arg_ptr);

            while (parser.base.match(.Comma)) {
                const arg = try parser.parseExpression();
                const arg_ptr = try parser.base.arena.createNode(ast.Expressions.ExprNode);
                arg_ptr.* = arg;
                try args.append(parser.base.arena.allocator(), arg_ptr);
            }
        }

        _ = try parser.base.consume(.RightParen, "Expected ')' after arguments");

        // create the builtin function call
        const full_name = try std.fmt.allocPrint(parser.base.arena.allocator(), "@{s}", .{builtin_name});

        // create identifier for the function name
        const name_expr = try parser.base.arena.createNode(ast.Expressions.ExprNode);
        name_expr.* = ast.Expressions.ExprNode{ .Identifier = ast.Expressions.IdentifierExpr{
            .name = full_name,
            .type_info = ast.Types.TypeInfo.unknown(),
            .span = parser.base.spanFromToken(at_token),
        } };

        return ast.Expressions.ExprNode{
            .Call = ast.Expressions.CallExpr{
                .callee = name_expr,
                .arguments = try args.toOwnedSlice(parser.base.arena.allocator()),
                .type_info = ast.Types.TypeInfo.unknown(), // Will be resolved during type checking
                .span = parser.base.spanFromToken(at_token),
            },
        };
    } else {
        try parser.base.errorAtCurrent("Unknown builtin function");
        return error.UnexpectedToken;
    }
}

/// Parse parenthesized expressions or tuples
pub fn parseParenthesizedOrTuple(parser: *ExpressionParser) ParserError!ast.Expressions.ExprNode {
    const paren_token = parser.base.previous();

    // check for empty tuple
    if (parser.base.check(.RightParen)) {
        _ = parser.base.advance();
        var empty_elements = std.ArrayList(*ast.Expressions.ExprNode){};
        defer empty_elements.deinit(parser.base.arena.allocator());

        const tuple_expr = ast.Expressions.ExprNode{ .Tuple = ast.Expressions.TupleExpr{
            .elements = try empty_elements.toOwnedSlice(parser.base.arena.allocator()),
            .span = parser.base.spanFromToken(paren_token),
        } };
        return tuple_expr;
    }

    const first_expr = try parser.parseExpressionNoComma();

    // check if it's a tuple (has comma)
    if (parser.base.match(.Comma)) {
        var elements = std.ArrayList(*ast.Expressions.ExprNode){};
        defer elements.deinit(parser.base.arena.allocator());

        // convert first_expr to pointer
        const first_ptr = try parser.base.arena.createNode(ast.Expressions.ExprNode);
        first_ptr.* = first_expr;
        try elements.append(parser.base.arena.allocator(), first_ptr);

        // handle trailing comma case: (a,)
        if (!parser.base.check(.RightParen)) {
            repeat: while (true) {
                const element = try parser.parseExpressionNoComma();
                const element_ptr = try parser.base.arena.createNode(ast.Expressions.ExprNode);
                element_ptr.* = element;
                try elements.append(parser.base.arena.allocator(), element_ptr);

                if (!parser.base.match(.Comma)) break :repeat;
                if (parser.base.check(.RightParen)) break :repeat; // Trailing comma
            }
        }

        _ = try parser.base.consume(.RightParen, "Expected ')' after tuple elements");

        const tuple_expr = ast.Expressions.ExprNode{ .Tuple = ast.Expressions.TupleExpr{
            .elements = try elements.toOwnedSlice(parser.base.arena.allocator()),
            .span = parser.base.spanFromToken(paren_token),
        } };
        return tuple_expr;
    } else {
        // single parenthesized expression
        _ = try parser.base.consume(.RightParen, "Expected ')' after expression");
        return first_expr;
    }
}

/// Parse struct instantiation expression (e.g., `MyStruct { a: 1, b: 2 }`)
pub fn parseStructInstantiation(parser: *ExpressionParser, name_token: Token) ParserError!ast.Expressions.ExprNode {
    _ = try parser.base.consume(.LeftBrace, "Expected '{' after struct name");

    var fields = std.ArrayList(ast.Expressions.StructInstantiationField){};
    defer fields.deinit(parser.base.arena.allocator());

    // parse field initializers (field_name: value)
    while (!parser.base.check(.RightBrace) and !parser.base.isAtEnd()) {
        const field_name = try parser.base.consumeIdentifierOrKeyword("Expected field name in struct instantiation");
        _ = try parser.base.consume(.Colon, "Expected ':' after field name in struct instantiation");

        const field_value = try precedence.parseAssignment(parser);
        const field_value_ptr = try parser.base.arena.createNode(ast.Expressions.ExprNode);
        field_value_ptr.* = field_value;

        try fields.append(parser.base.arena.allocator(), ast.Expressions.StructInstantiationField{
            .name = field_name.lexeme,
            .value = field_value_ptr,
            .span = parser.base.spanFromToken(field_name),
        });

        // optional comma (but don't require it for last field)
        if (!parser.base.check(.RightBrace)) {
            if (!parser.base.match(.Comma)) {
                // no comma found, but we're not at the end, so this is an error
                return error.ExpectedToken;
            }
        } else {
            _ = parser.base.match(.Comma); // Consume trailing comma if present
        }
    }

    _ = try parser.base.consume(.RightBrace, "Expected '}' after struct instantiation fields");

    // create the struct name identifier
    const struct_name_ptr = try parser.base.arena.createNode(ast.Expressions.ExprNode);
    struct_name_ptr.* = ast.Expressions.ExprNode{ .Identifier = ast.Expressions.IdentifierExpr{
        .name = name_token.lexeme,
        .type_info = ast.Types.TypeInfo.unknown(),
        .span = parser.base.spanFromToken(name_token),
    } };

    return ast.Expressions.ExprNode{ .StructInstantiation = ast.Expressions.StructInstantiationExpr{
        .struct_name = struct_name_ptr,
        .fields = try fields.toOwnedSlice(parser.base.arena.allocator()),
        .span = parser.base.spanFromToken(name_token),
    } };
}

/// Parse generic struct instantiation: Pair(u256) { first: value, ... }
/// Called after Pair(u256) has been parsed as a Call expression.
/// Stores the Call expression as the struct_name — the type resolver will
/// detect the Call pattern and trigger monomorphization.
pub fn parseGenericStructInstantiation(
    parser: *ExpressionParser,
    call: *const ast.Expressions.CallExpr,
) ParserError!ast.Expressions.ExprNode {
    const alloc = parser.base.arena.allocator();
    _ = try parser.base.consume(.LeftBrace, "Expected '{' after generic struct type arguments");

    var fields = std.ArrayList(ast.Expressions.StructInstantiationField){};
    defer fields.deinit(alloc);

    while (!parser.base.check(.RightBrace) and !parser.base.isAtEnd()) {
        const field_name = try parser.base.consumeIdentifierOrKeyword("Expected field name in struct instantiation");
        _ = try parser.base.consume(.Colon, "Expected ':' after field name in struct instantiation");
        const field_value = try precedence.parseAssignment(parser);
        const field_value_ptr = try parser.base.arena.createNode(ast.Expressions.ExprNode);
        field_value_ptr.* = field_value;
        try fields.append(alloc, ast.Expressions.StructInstantiationField{
            .name = field_name.lexeme,
            .value = field_value_ptr,
            .span = parser.base.spanFromToken(field_name),
        });
        if (!parser.base.check(.RightBrace)) {
            if (!parser.base.match(.Comma)) return error.ExpectedToken;
        } else {
            _ = parser.base.match(.Comma);
        }
    }
    _ = try parser.base.consume(.RightBrace, "Expected '}' after struct instantiation fields");

    // Store the original Call expression as struct_name — preserves the
    // callee (struct name) and arguments (type args) for the type resolver.
    const call_expr_ptr = try parser.base.arena.createNode(ast.Expressions.ExprNode);
    call_expr_ptr.* = ast.Expressions.ExprNode{ .Call = call.* };

    return ast.Expressions.ExprNode{ .StructInstantiation = ast.Expressions.StructInstantiationExpr{
        .struct_name = call_expr_ptr,
        .fields = try fields.toOwnedSlice(alloc),
        .span = call.span,
    } };
}

/// Parse anonymous struct literal (.{field = value, ...})
pub fn parseAnonymousStructLiteral(parser: *ExpressionParser) ParserError!ast.Expressions.ExprNode {
    const dot_token = parser.base.previous();
    _ = try parser.base.consume(.LeftBrace, "Expected '{' after '.' in anonymous struct literal");

    var fields = std.ArrayList(ast.Expressions.AnonymousStructField){};
    defer fields.deinit(parser.base.arena.allocator());

    // parse field initializers (.{ .field = value, ... })
    while (!parser.base.check(.RightBrace) and !parser.base.isAtEnd()) {
        // enforce leading dot before each field name
        _ = try parser.base.consume(.Dot, "Expected '.' before field name in anonymous struct literal");
        const field_name = try parser.base.consume(.Identifier, "Expected field name in anonymous struct literal");
        _ = try parser.base.consume(.Equal, "Expected '=' after field name in anonymous struct literal");

        const field_value = try precedence.parseAssignment(parser);
        const field_value_ptr = try parser.base.arena.createNode(ast.Expressions.ExprNode);
        field_value_ptr.* = field_value;

        try fields.append(parser.base.arena.allocator(), ast.Expressions.AnonymousStructField{
            .name = field_name.lexeme,
            .value = field_value_ptr,
            .span = parser.base.spanFromToken(field_name),
        });

        // optional comma (but don't require it for last field)
        if (!parser.base.check(.RightBrace)) {
            if (!parser.base.match(.Comma)) {
                // no comma found, but we're not at the end, so this is an error
                return error.ExpectedToken;
            }
        } else {
            _ = parser.base.match(.Comma); // Consume trailing comma if present
        }
    }

    _ = try parser.base.consume(.RightBrace, "Expected '}' after anonymous struct literal fields");

    return ast.Expressions.ExprNode{ .AnonymousStruct = ast.Expressions.AnonymousStructExpr{
        .fields = try fields.toOwnedSlice(parser.base.arena.allocator()),
        .span = parser.base.spanFromToken(dot_token),
    } };
}

/// Parse array literal ([element1, element2, ...])
pub fn parseArrayLiteral(parser: *ExpressionParser) ParserError!ast.Expressions.ExprNode {
    const bracket_token = parser.base.previous();

    var elements = std.ArrayList(*ast.Expressions.ExprNode){};
    defer elements.deinit(parser.base.arena.allocator());

    // parse array elements
    while (!parser.base.check(.RightBracket) and !parser.base.isAtEnd()) {
        const element = try precedence.parseAssignment(parser);
        const element_ptr = try parser.base.arena.createNode(ast.Expressions.ExprNode);
        element_ptr.* = element;
        try elements.append(parser.base.arena.allocator(), element_ptr);

        // optional comma (but don't require it for last element)
        if (!parser.base.check(.RightBracket)) {
            if (!parser.base.match(.Comma)) {
                // no comma found, but we're not at the end, so this is an error
                return error.ExpectedToken;
            }
        } else {
            _ = parser.base.match(.Comma); // Consume trailing comma if present
        }
    }

    _ = try parser.base.consume(.RightBracket, "Expected ']' after array elements");

    return ast.Expressions.ExprNode{
        .ArrayLiteral = ast.Literals.Array{
            .elements = try elements.toOwnedSlice(parser.base.arena.allocator()),
            .element_type = null, // Type will be inferred
            .span = parser.base.spanFromToken(bracket_token),
        },
    };
}

/// Parse a range expression (start...end)
/// This creates a RangeExpr which can be used both in switch patterns and directly as expressions
pub fn parseRangeExpression(parser: *ExpressionParser) ParserError!ast.Expressions.ExprNode {
    const start_token = parser.base.peek();

    // parse start expression
    const start_expr = try parser.parseExpression();

    // accept both .. and ... for range expressions
    if (parser.base.match(.DotDot)) {
        // .. range (exclusive end)
    } else if (parser.base.match(.DotDotDot)) {
        // ... range (inclusive end)
    } else {
        try parser.base.errorAtCurrent("Expected '..' or '...' in range expression");
        return error.ExpectedToken;
    }

    // parse end expression
    const end_expr = try parser.parseExpression();

    // create pointers to the expressions
    const start_ptr = try parser.base.arena.createNode(ast.Expressions.ExprNode);
    start_ptr.* = start_expr;
    const end_ptr = try parser.base.arena.createNode(ast.Expressions.ExprNode);
    end_ptr.* = end_expr;

    return ast.Expressions.ExprNode{
        .Range = ast.Expressions.RangeExpr{
            .start = start_ptr,
            .end = end_ptr,
            .inclusive = true, // Default to inclusive range
            .type_info = TypeInfo.unknown(), // Type will be inferred
            .span = parser.base.spanFromToken(start_token),
        },
    };
}

/// Parse quantified expressions for formal verification
/// Syntax: forall variable: Type where condition => body
///         exists variable: Type where condition => body
pub fn parseQuantifiedExpression(parser: *ExpressionParser) ParserError!ast.Expressions.ExprNode {
    const quant_token = parser.base.previous();

    // determine quantifier type
    const quantifier: ast.Expressions.QuantifierType = if (std.mem.eql(u8, quant_token.lexeme, "forall"))
        .Forall
    else
        .Exists;

    // parse variable name
    const var_token = try parser.base.consume(.Identifier, "Expected variable name after quantifier");
    const variable = try parser.base.arena.createString(var_token.lexeme);

    // expect ":"
    _ = try parser.base.consume(.Colon, "Expected ':' after variable name");

    // parse variable type
    const type_parser = TypeParser.init(parser.base.tokens, parser.base.arena);
    var type_parser_mut = type_parser;
    type_parser_mut.base.current = parser.base.current;
    const variable_type = try type_parser_mut.parseType();
    parser.base.current = type_parser_mut.base.current;

    // parse optional where clause
    var condition: ?*ast.Expressions.ExprNode = null;
    if (parser.base.match(.Where)) {
        const cond_expr = try parser.parseExpression();
        const cond_ptr = try parser.base.arena.createNode(ast.Expressions.ExprNode);
        cond_ptr.* = cond_expr;
        condition = cond_ptr;
    }

    // expect "=>" (single Arrow token)
    _ = try parser.base.consume(.Arrow, "Expected '=>' after quantifier condition");

    // parse body expression
    const body_expr = try parser.parseExpression();
    const body_ptr = try parser.base.arena.createNode(ast.Expressions.ExprNode);
    body_ptr.* = body_expr;

    return ast.Expressions.ExprNode{
        .Quantified = ast.Expressions.QuantifiedExpr{
            .quantifier = quantifier,
            .variable = variable,
            .variable_type = variable_type,
            .condition = condition,
            .body = body_ptr,
            .span = parser.base.spanFromToken(quant_token),
            .is_specification = true,
            .verification_metadata = null,
            .verification_attributes = &[_]ast.verification.VerificationAttribute{},
        },
    };
}
