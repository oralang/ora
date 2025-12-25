// ============================================================================
// Type Declaration Parser
// ============================================================================
//
// Handles parsing of type declarations:
//   • Struct declarations
//   • Enum declarations (with underlying types and variant values)
//   • Enum variant value parsing
//
// ============================================================================

const std = @import("std");
const ast = @import("../../ast.zig");
const TypeParser = @import("../type_parser.zig").TypeParser;
const ExpressionParser = @import("../expression_parser.zig").ExpressionParser;

// Forward declaration - DeclarationParser is defined in declaration_parser.zig
const DeclarationParser = @import("../declaration_parser.zig").DeclarationParser;

/// Parse struct declaration with complete field definitions
pub fn parseStruct(parser: *DeclarationParser, type_parser: *TypeParser) !ast.AstNode {
    const name_token = try parser.base.consume(.Identifier, "Expected struct name");
    _ = try parser.base.consume(.LeftBrace, "Expected '{' after struct name");

    var fields = std.ArrayList(ast.StructField){};
    defer fields.deinit(parser.base.arena.allocator());

    while (!parser.base.check(.RightBrace) and !parser.base.isAtEnd()) {
        // skip any field attributes/annotations for now
        while (parser.base.match(.At)) {
            _ = try parser.base.consume(.Identifier, "Expected annotation name after '@'");
            if (parser.base.match(.LeftParen)) {
                var paren_depth: u32 = 1;
                while (paren_depth > 0 and !parser.base.isAtEnd()) {
                    if (parser.base.match(.LeftParen)) {
                        paren_depth += 1;
                    } else if (parser.base.match(.RightParen)) {
                        paren_depth -= 1;
                    } else {
                        _ = parser.base.advance();
                    }
                }
            }
        }

        const field_name = try parser.base.consumeIdentifierOrKeyword("Expected field name");
        _ = try parser.base.consume(.Colon, "Expected ':' after field name");

        // use type parser with complete type information
        type_parser.base.current = parser.base.current;
        const field_type = try type_parser.parseTypeWithContext(.StructField);
        parser.base.current = type_parser.base.current;

        _ = try parser.base.consume(.Semicolon, "Expected ';' after field");

        try fields.append(parser.base.arena.allocator(), ast.StructField{
            .name = field_name.lexeme,
            .type_info = field_type,
            .span = parser.base.spanFromToken(field_name),
        });
    }

    _ = try parser.base.consume(.RightBrace, "Expected '}' after struct fields");

    return ast.AstNode{ .StructDecl = ast.StructDeclNode{
        .name = name_token.lexeme,
        .fields = try fields.toOwnedSlice(parser.base.arena.allocator()),
        .span = parser.base.spanFromToken(name_token),
    } };
}

/// Parse a single enum variant value with proper precedence handling
/// This avoids using the general expression parser which would interpret commas as operators
/// The enum's underlying type is used for integer literals instead of generic integers
pub fn parseEnumVariantValue(
    parser: *DeclarationParser,
    expr_parser: *ExpressionParser,
    underlying_type: ?ast.Types.TypeInfo,
) !ast.Expressions.ExprNode {
    // use the expression parser but stop at comma level to avoid enum separator confusion
    // this ensures proper operator precedence and left-associativity
    expr_parser.base.current = parser.base.current;
    var expr = try expr_parser.parseLogicalOr();
    parser.base.current = expr_parser.base.current;

    // if this is a simple literal and we have an underlying type for the enum,
    // apply the enum's underlying type immediately. For complex expressions, leave as unknown
    // and let the semantic analyzer resolve with the enum type constraint.
    if (expr == .Literal and underlying_type != null) {
        var updated_type_info = underlying_type.?;
        updated_type_info.source = .explicit; // Mark as explicit since the user provided this value

        switch (expr.Literal) {
            .Integer => |*int_lit| {
                int_lit.type_info = updated_type_info;
            },
            .String => |*str_lit| {
                str_lit.type_info = updated_type_info;
            },
            .Bool => |*bool_lit| {
                bool_lit.type_info = updated_type_info;
            },
            .Address => |*addr_lit| {
                addr_lit.type_info = updated_type_info;
            },
            .Hex => |*hex_lit| {
                hex_lit.type_info = updated_type_info;
            },
            .Binary => |*bin_lit| {
                bin_lit.type_info = updated_type_info;
            },
            .Character => |*char_lit| {
                char_lit.type_info = updated_type_info;
            },
            .Bytes => |*bytes_lit| {
                bytes_lit.type_info = updated_type_info;
            },
        }
    }
    // for complex expressions, we leave them as-is with unknown types
    // the semantic analyzer will resolve them with the enum type constraint

    return expr;
}

/// Parse enum declaration with value assignments and base types
pub fn parseEnum(
    parser: *DeclarationParser,
    type_parser: *TypeParser,
    expr_parser: *ExpressionParser,
) !ast.AstNode {
    const name_token = try parser.base.consume(.Identifier, "Expected enum name");

    // parse optional underlying type: enum Status : u8 { ... }
    var base_type: ?ast.Types.TypeInfo = null; // No default - let type system infer
    if (parser.base.match(.Colon)) {
        // use type parser for underlying type
        type_parser.base.current = parser.base.current;
        base_type = try type_parser.parseTypeWithContext(.EnumUnderlying);
        parser.base.current = type_parser.base.current;
    }

    _ = try parser.base.consume(.LeftBrace, "Expected '{' after enum name");

    var variants = std.ArrayList(ast.EnumVariant){};
    defer variants.deinit(parser.base.arena.allocator());

    // track if any variants have explicit values
    var has_explicit_values = false;

    // track the current implicit value, starting from 0
    var next_implicit_value: u32 = 0;

    // parse all enum variants
    while (!parser.base.check(.RightBrace) and !parser.base.isAtEnd()) {
        const variant_name = try parser.base.consume(.Identifier, "Expected variant name");

        // parse optional explicit value assignment: VariantName = value
        var value: ?ast.Expressions.ExprNode = null;
        if (parser.base.match(.Equal)) {
            has_explicit_values = true; // Mark that we have at least one explicit value

            // use our dedicated enum variant value parser to ensure complex expressions
            // like (1 + 2) * 3 are parsed correctly without misinterpreting operators
            // as enum variant separators
            const expr = try parseEnumVariantValue(parser, expr_parser, base_type);
            value = expr;

            // if this is an integer literal, we need to update the next_implicit_value
            if (expr == .Literal and expr.Literal == .Integer) {
                // try to parse the integer value to determine the next implicit value
                const int_str = expr.Literal.Integer.value;
                if (std.fmt.parseInt(u32, int_str, 0)) |int_val| {
                    // set the next implicit value to one more than this explicit value
                    next_implicit_value = int_val + 1;
                } else |_| {
                    // if we can't parse it (e.g., it's a complex expression),
                    // just continue with the current next_implicit_value
                }
            }
        } else {
            // create an implicit value for this variant
            // we always add integer values even when not explicitly assigned
            const int_literal = ast.expressions.IntegerLiteral{
                .value = try std.fmt.allocPrint(parser.base.arena.allocator(), "{d}", .{next_implicit_value}),
                .type_info = if (base_type) |bt| blk: {
                    var type_info = bt;
                    type_info.source = .inferred; // Mark as inferred since compiler generated this
                    break :blk type_info;
                } else blk: {
                    // default to u32 for enum values if no underlying type specified
                    var type_info = ast.Types.TypeInfo.fromOraType(.u32);
                    type_info.source = .inferred;
                    break :blk type_info;
                },
                .span = parser.base.spanFromToken(variant_name),
            };

            // create an ExprNode with the integer literal (no need to store in arena)
            value = ast.Expressions.ExprNode{ .Literal = .{ .Integer = int_literal } };

            // increment the implicit value for the next variant
            next_implicit_value += 1;
        }

        // add the variant to our list
        try variants.append(parser.base.arena.allocator(), ast.EnumVariant{
            .name = variant_name.lexeme,
            .value = value,
            .span = parser.base.spanFromToken(variant_name),
        });

        // check for comma separator or end of enum body
        if (parser.base.check(.RightBrace)) {
            break; // No more variants
        } else if (!parser.base.match(.Comma)) {
            // if not a comma and not the end, it's an error
            try parser.base.errorAtCurrent("Expected ',' between enum variants");
            return error.ExpectedToken;
        }
    }

    _ = try parser.base.consume(.RightBrace, "Expected '}' after enum variants");

    // create and return the enum declaration node
    return ast.AstNode{
        .EnumDecl = ast.EnumDeclNode{
            .name = name_token.lexeme,
            .variants = try variants.toOwnedSlice(parser.base.arena.allocator()),
            .underlying_type_info = base_type,
            .span = parser.base.spanFromToken(name_token),
            .has_explicit_values = true, // We now always have values (implicit or explicit)
        },
    };
}
