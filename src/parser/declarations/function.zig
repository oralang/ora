// ============================================================================
// Function Declaration Parser
// ============================================================================
//
// Handles parsing of function declarations:
//   • Function signatures (name, parameters, return type)
//   • Function parameters (with default values, mutability)
//   • Specification clauses (requires, ensures, modifies)
//
// ============================================================================

const std = @import("std");
const ast = @import("../../ast.zig");
const TypeParser = @import("../type_parser.zig").TypeParser;
const ExpressionParser = @import("../expression_parser.zig").ExpressionParser;

// Forward declaration - DeclarationParser is defined in declaration_parser.zig
const DeclarationParser = @import("../declaration_parser.zig").DeclarationParser;
const ParserCommon = @import("../common.zig").ParserCommon;

/// Parse function declaration
pub fn parseFunction(
    parser: *DeclarationParser,
    type_parser: *TypeParser,
    expr_parser: *ExpressionParser,
) !ast.FunctionNode {
    // Parse visibility
    const is_pub = parser.base.match(.Pub);

    _ = try parser.base.consume(.Fn, "Expected 'fn'");

    // Allow both regular identifiers and 'init' keyword as function names
    const name_token = if (parser.base.check(.Identifier))
        parser.base.advance()
    else if (parser.base.check(.Init))
        parser.base.advance()
    else {
        try parser.base.errorAtCurrent("Expected function name");
        return error.ExpectedToken;
    };
    const is_init_fn = std.mem.eql(u8, name_token.lexeme, "init");

    _ = try parser.base.consume(.LeftParen, "Expected '(' after function name");

    // Parse parameters with default values support
    var params = std.ArrayList(ast.ParameterNode){};
    defer params.deinit(parser.base.arena.allocator());
    errdefer {
        // Clean up parameters on error
        for (params.items) |*param| {
            if (param.default_value) |default_val| {
                ast.deinitExprNode(parser.base.arena.allocator(), default_val);
                parser.base.arena.allocator().destroy(default_val);
            }
        }
    }

    if (!parser.base.check(.RightParen)) {
        repeat: while (true) {
            const param = try parseParameterWithDefaults(parser, type_parser, expr_parser);
            try params.append(parser.base.arena.allocator(), param);

            if (!parser.base.match(.Comma)) break :repeat;
        }
    }

    _ = try parser.base.consume(.RightParen, "Expected ')' after parameters");

    // Parse optional return type using arrow syntax: fn foo(...) -> Type
    var return_type_info: ?ast.Types.TypeInfo = null;
    if (parser.base.check(.Arrow)) {
        if (is_init_fn) {
            try parser.base.errorAtCurrent("'init' cannot have a return type");
            return error.UnexpectedToken;
        }
        _ = parser.base.advance(); // consume '->'
        // Use type parser
        type_parser.base.current = parser.base.current;
        const parsed_type = try type_parser.parseReturnType();
        parser.base.current = type_parser.base.current;
        return_type_info = parsed_type;
    }

    // Parse requires clauses
    var requires_clauses = std.ArrayList(*ast.Expressions.ExprNode){};
    defer requires_clauses.deinit(parser.base.arena.allocator());

    while (parser.base.match(.Requires)) {
        _ = try parser.base.consume(.LeftParen, "Expected '(' after 'requires'");

        // Parse the condition expression
        expr_parser.base.current = parser.base.current;
        const condition = try expr_parser.parseExpression();
        parser.base.current = expr_parser.base.current;

        _ = try parser.base.consume(.RightParen, "Expected ')' after requires condition");
        if (parser.base.match(.Semicolon)) {
            try parser.base.errorAtCurrent("Unexpected ';' after requires(...) (no semicolon allowed)");
            return error.UnexpectedToken;
        }

        // Store the expression in arena
        const condition_ptr = try parser.base.arena.createNode(ast.Expressions.ExprNode);
        condition_ptr.* = condition;
        try requires_clauses.append(parser.base.arena.allocator(), condition_ptr);
    }

    // Parse ensures clauses
    var ensures_clauses = std.ArrayList(*ast.Expressions.ExprNode){};
    defer ensures_clauses.deinit(parser.base.arena.allocator());

    while (parser.base.match(.Ensures)) {
        _ = try parser.base.consume(.LeftParen, "Expected '(' after 'ensures'");

        // Parse the condition expression
        expr_parser.base.current = parser.base.current;
        const condition = try expr_parser.parseExpression();
        parser.base.current = expr_parser.base.current;

        _ = try parser.base.consume(.RightParen, "Expected ')' after ensures condition");
        if (parser.base.match(.Semicolon)) {
            try parser.base.errorAtCurrent("Unexpected ';' after ensures(...) (no semicolon allowed)");
            return error.UnexpectedToken;
        }

        // Store the expression in arena
        const condition_ptr = try parser.base.arena.createNode(ast.Expressions.ExprNode);
        condition_ptr.* = condition;
        try ensures_clauses.append(parser.base.arena.allocator(), condition_ptr);
    }

    // Parse modifies clause (frame conditions)
    var modifies_clause: ?[]*ast.Expressions.ExprNode = null;
    if (parser.base.match(.Modifies)) {
        _ = try parser.base.consume(.LeftParen, "Expected '(' after 'modifies'");

        var modifies_list = std.ArrayList(*ast.Expressions.ExprNode){};
        defer modifies_list.deinit(parser.base.arena.allocator());

        // Parse expression list (comma-separated)
        if (!parser.base.check(.RightParen)) {
            repeat: while (true) {
                expr_parser.base.current = parser.base.current;
                const expr = try expr_parser.parseExpression();
                parser.base.current = expr_parser.base.current;

                const expr_ptr = try parser.base.arena.createNode(ast.Expressions.ExprNode);
                expr_ptr.* = expr;
                try modifies_list.append(parser.base.arena.allocator(), expr_ptr);

                if (!parser.base.match(.Comma)) break :repeat;
            }
        }

        _ = try parser.base.consume(.RightParen, "Expected ')' after modifies list");
        if (parser.base.match(.Semicolon)) {
            try parser.base.errorAtCurrent("Unexpected ';' after modifies(...) (no semicolon allowed)");
            return error.UnexpectedToken;
        }

        modifies_clause = try modifies_list.toOwnedSlice(parser.base.arena.allocator());
    }

    // Parse function body - delegate to statement parser
    // Placeholder body: parser_core will parse the real block with StatementParser
    // Do not consume any tokens here; leave current at '{'
    const body = ast.Statements.BlockNode{ .statements = &[_]ast.Statements.StmtNode{}, .span = parser.base.currentSpan() };

    return ast.FunctionNode{
        .name = name_token.lexeme,
        .parameters = try params.toOwnedSlice(parser.base.arena.allocator()),
        .return_type_info = return_type_info,
        .body = body,
        .visibility = if (is_pub) ast.Visibility.Public else ast.Visibility.Private,
        .attributes = &[_]u8{}, // Empty attributes
        .requires_clauses = try requires_clauses.toOwnedSlice(parser.base.arena.allocator()),
        .ensures_clauses = try ensures_clauses.toOwnedSlice(parser.base.arena.allocator()),
        .modifies_clause = modifies_clause,
        .span = parser.base.spanFromToken(name_token),
    };
}

/// Parse function parameter
pub fn parseParameter(parser: *DeclarationParser, type_parser: *TypeParser) !ast.ParameterNode {
    const name_token = try parser.base.consumeIdentifierOrKeyword("Expected parameter name");
    _ = try parser.base.consume(.Colon, "Expected ':' after parameter name");
    // Use type parser
    type_parser.base.current = parser.base.current;
    const param_type = try type_parser.parseTypeWithContext(.Parameter);
    parser.base.current = type_parser.base.current;

    return ast.ParameterNode{
        .name = name_token.lexeme,
        .type_info = param_type,
        .is_mutable = false, // Default to immutable
        .default_value = null, // No default value
        .span = parser.base.spanFromToken(name_token),
    };
}

/// Parse function parameter with default value support
pub fn parseParameterWithDefaults(
    parser: *DeclarationParser,
    type_parser: *TypeParser,
    expr_parser: *ExpressionParser,
) !ast.ParameterNode {
    // Check for mutable parameter modifier (mut param_name)
    const is_mutable = if (parser.base.check(.Identifier) and std.mem.eql(u8, parser.base.peek().lexeme, "mut")) blk: {
        _ = parser.base.advance(); // consume "mut"
        break :blk true;
    } else false;

    const name_token = try parser.base.consumeIdentifierOrKeyword("Expected parameter name");
    _ = try parser.base.consume(.Colon, "Expected ':' after parameter name");

    // Use type parser
    type_parser.base.current = parser.base.current;
    const param_type = try type_parser.parseTypeWithContext(.Parameter);
    parser.base.current = type_parser.base.current;

    // Parse optional default value
    var default_value: ?*ast.Expressions.ExprNode = null;
    if (parser.base.match(.Equal)) {
        // Parse default value expression
        expr_parser.base.current = parser.base.current;
        const expr = try expr_parser.parseExpression();
        parser.base.current = expr_parser.base.current;

        // Store in arena
        const expr_ptr = try parser.base.arena.createNode(ast.Expressions.ExprNode);
        expr_ptr.* = expr;
        default_value = expr_ptr;
    }

    return ast.ParameterNode{
        .name = name_token.lexeme,
        .type_info = param_type,
        .is_mutable = is_mutable,
        .default_value = default_value,
        .span = ParserCommon.makeSpan(name_token),
    };
}
