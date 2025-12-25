// ============================================================================
// Contract Declaration Parser
// ============================================================================
//
// Handles parsing of contract declarations and their members:
//   • Contract declarations (with extends/implements)
//   • Contract members (functions, variables, structs, enums, etc.)
//   • Contract invariants
//   • Ghost declarations
//
// ============================================================================

const std = @import("std");
const ast = @import("../../ast.zig");
const common = @import("../common.zig");
const TypeParser = @import("../type_parser.zig").TypeParser;
const ExpressionParser = @import("../expression_parser.zig").ExpressionParser;
const StatementParser = @import("../statement_parser.zig").StatementParser;

// Forward declaration - DeclarationParser is defined in declaration_parser.zig
const DeclarationParser = @import("../declaration_parser.zig").DeclarationParser;

/// Parse contract declaration
pub fn parseContract(
    parser: *DeclarationParser,
    type_parser: *TypeParser,
    expr_parser: *ExpressionParser,
) !ast.AstNode {
    const name_token = try parser.base.consume(.Identifier, "Expected contract name");

    // parse optional inheritance: contract Child extends Parent
    var extends: ?[]const u8 = null;
    if (parser.base.match(.Extends)) {
        const parent_token = try parser.base.consume(.Identifier, "Expected parent contract name after 'extends'");
        extends = parent_token.lexeme;
    }

    // parse optional interface implementations: contract MyContract implements Interface1, Interface2
    var implements = std.ArrayList([]const u8){};
    defer implements.deinit(parser.base.arena.allocator());

    if (parser.base.match(.Implements)) {
        // parse comma-separated list of interface names
        repeat: while (true) {
            const interface_token = try parser.base.consume(.Identifier, "Expected interface name");
            try implements.append(parser.base.arena.allocator(), interface_token.lexeme);

            if (!parser.base.match(.Comma)) break :repeat;
        }
    }

    _ = try parser.base.consume(.LeftBrace, "Expected '{' after contract declaration");

    var body = std.ArrayList(ast.AstNode){};
    defer body.deinit(parser.base.arena.allocator());
    errdefer {
        // clean up any partially parsed members on error
        for (body.items) |*member| {
            ast.deinitAstNode(parser.base.arena.allocator(), member);
        }
    }

    while (!parser.base.check(.RightBrace) and !parser.base.isAtEnd()) {
        const member = try parseContractMember(parser, type_parser, expr_parser);
        try body.append(parser.base.arena.allocator(), member);
    }

    _ = try parser.base.consume(.RightBrace, "Expected '}' after contract body");

    return ast.AstNode{ .Contract = ast.ContractNode{
        .name = name_token.lexeme,
        .extends = extends,
        .implements = try implements.toOwnedSlice(parser.base.arena.allocator()),
        .attributes = &[_]u8{},
        .body = try body.toOwnedSlice(parser.base.arena.allocator()),
        .span = parser.base.spanFromToken(name_token),
    } };
}

/// Parse contract invariant
pub fn parseContractInvariant(
    parser: *DeclarationParser,
    expr_parser: *ExpressionParser,
) !ast.AstNode {
    const invariant_token = parser.base.previous(); // The 'invariant' keyword

    const name_token = try parser.base.consume(.Identifier, "Expected invariant name");
    const name = try parser.base.arena.createString(name_token.lexeme);

    _ = try parser.base.consume(.LeftParen, "Expected '(' after invariant name");

    // parse the condition expression
    expr_parser.base.current = parser.base.current;
    const condition = try expr_parser.parseExpression();
    parser.base.current = expr_parser.base.current;

    _ = try parser.base.consume(.RightParen, "Expected ')' after invariant condition");
    _ = try parser.base.consume(.Semicolon, "Expected ';' after invariant declaration");

    // store the expression in arena
    const condition_ptr = try parser.base.arena.createNode(ast.Expressions.ExprNode);
    condition_ptr.* = condition;

    return ast.AstNode{ .ContractInvariant = ast.ContractInvariant{
        .name = name,
        .condition = condition_ptr,
        .span = parser.base.spanFromToken(invariant_token),
        .is_specification = true,
    } };
}

/// Parse ghost declaration (ghost variable/function/block)
pub fn parseGhostDeclaration(
    parser: *DeclarationParser,
    type_parser: *TypeParser,
    expr_parser: *ExpressionParser,
) !ast.AstNode {
    _ = try parser.base.consume(.Ghost, "Expected 'ghost' keyword");

    // ghost can precede: variable declaration, function declaration, or block
    if (parser.base.check(.Pub) or parser.base.check(.Fn)) {
        // ghost function
        var fn_node = try parser.parseFunction(type_parser, expr_parser);

        // parse the function body block using a local StatementParser
        var stmt_parser = StatementParser.init(parser.base.tokens, parser.base.arena);
        stmt_parser.base.current = parser.base.current;
        const body_block = try stmt_parser.parseBlock();

        // update current position from statement parser
        parser.base.current = stmt_parser.base.current;

        // attach the parsed body
        fn_node.body = body_block;
        fn_node.is_ghost = true; // Mark as ghost

        return ast.AstNode{ .Function = fn_node };
    } else if (parser.isMemoryRegionKeyword() or parser.base.check(.Let) or parser.base.check(.Var) or parser.base.check(.Immutable) or parser.base.check(.Const)) {
        // ghost variable or constant
        if (parser.base.check(.Const)) {
            var const_node = try parser.parseConstantDecl(type_parser, expr_parser);
            const_node.is_ghost = true; // Mark as ghost
            return ast.AstNode{ .Constant = const_node };
        } else {
            var var_node = try parser.parseVariableDecl(type_parser, expr_parser);
            var_node.is_ghost = true; // Mark as ghost
            return ast.AstNode{ .VariableDecl = var_node };
        }
    } else if (parser.base.check(.LeftBrace)) {
        // ghost block - parse as a statement block
        var stmt_parser = StatementParser.init(parser.base.tokens, parser.base.arena);
        stmt_parser.base.current = parser.base.current;
        var ghost_block = try stmt_parser.parseBlock();
        parser.base.current = stmt_parser.base.current;

        ghost_block.is_ghost = true; // Mark as ghost

        return ast.AstNode{ .Block = ghost_block };
    } else {
        try parser.base.errorAtCurrent("Expected function, variable, or block after 'ghost'");
        return error.UnexpectedToken;
    }
}

/// Parse contract member (function, variable, etc.) with proper scoping
pub fn parseContractMember(
    parser: *DeclarationParser,
    type_parser: *TypeParser,
    expr_parser: *ExpressionParser,
) !ast.AstNode {
    // check for contract invariants
    if (parser.base.check(.Invariant)) {
        _ = parser.base.advance(); // consume 'invariant'
        return try parseContractInvariant(parser, expr_parser);
    }

    // check for ghost declarations
    if (parser.base.check(.Ghost)) {
        return try parseGhostDeclaration(parser, type_parser, expr_parser);
    }

    // check for @lock annotation before variable declarations
    if (try parser.tryParseLockAnnotation(type_parser, expr_parser)) |var_decl| {
        return ast.AstNode{ .VariableDecl = var_decl };
    }

    // reject unknown @ directives at declaration position
    if (parser.base.check(.At)) {
        const saved = parser.base.current;
        _ = parser.base.advance();
        // only @lock is meaningful before a variable declaration (handled above)
        const is_known = parser.base.check(.Identifier) and std.mem.eql(u8, parser.base.peek().lexeme, "lock");
        if (!is_known) {
            _ = parser.base.errorAtCurrent("Unknown @ directive before declaration; attributes are not supported") catch {};
            // recover: skip to semicolon or next declaration boundary
            while (!parser.base.isAtEnd() and !parser.base.check(.Semicolon) and !parser.base.check(.RightBrace)) {
                _ = parser.base.advance();
            }
            _ = parser.base.match(.Semicolon);
        } else {
            // restore for lock-annotation path to handle normally
            parser.base.current = saved;
        }
    }

    // functions (can be public or private within contracts)
    if (parser.base.check(.Pub) or parser.base.check(.Fn)) {
        // parse the function header first (leaves current at '{')
        var fn_node = try parser.parseFunction(type_parser, expr_parser);

        // parse the function body block using a local StatementParser
        var stmt_parser = StatementParser.init(parser.base.tokens, parser.base.arena);
        stmt_parser.base.current = parser.base.current;
        const body_block = try stmt_parser.parseBlock();

        // update current position from statement parser
        parser.base.current = stmt_parser.base.current;

        // attach the parsed body
        fn_node.body = body_block;

        return ast.AstNode{ .Function = fn_node };
    }

    // constant declarations (contract constants) - check before variables
    if (parser.base.check(.Const)) {
        return ast.AstNode{ .Constant = try parser.parseConstantDecl(type_parser, expr_parser) };
    }

    // variable declarations (contract state variables)
    if (parser.isMemoryRegionKeyword() or parser.base.check(.Let) or parser.base.check(.Var) or parser.base.check(.Immutable)) {
        return ast.AstNode{ .VariableDecl = try parser.parseVariableDecl(type_parser, expr_parser) };
    }

    // error declarations (contract-specific errors)
    if (parser.base.match(.Error)) {
        return ast.AstNode{ .ErrorDecl = try parser.parseErrorDecl(type_parser) };
    }

    // log declarations (contract events)
    if (parser.base.match(.Log)) {
        return parser.parseLogDecl(type_parser);
    }

    // struct declarations (contract-scoped structs)
    if (parser.base.match(.Struct)) {
        return parser.parseStruct(type_parser);
    }

    // enum declarations (contract-scoped enums)
    if (parser.base.match(.Enum)) {
        return parser.parseEnum(type_parser, expr_parser);
    }

    try parser.base.errorAtCurrent("Expected contract member (function, variable, struct, enum, log, or error)");
    return error.UnexpectedToken;
}
