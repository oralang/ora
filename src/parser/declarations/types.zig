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
/// Supports generic structs: struct Pair(comptime T: type) { first: T, second: T }
pub fn parseStruct(parser: *DeclarationParser, type_parser: *TypeParser) !ast.AstNode {
    const name_token = try parser.base.consume(.Identifier, "Expected struct name");
    const alloc = parser.base.arena.allocator();

    // Parse optional comptime type parameters: (comptime T: type, ...)
    var type_param_names = std.ArrayList([]const u8){};
    defer type_param_names.deinit(alloc);
    var is_generic = false;

    if (parser.base.match(.LeftParen)) {
        while (!parser.base.check(.RightParen) and !parser.base.isAtEnd()) {
            _ = try parser.base.consume(.Comptime, "Expected 'comptime' in struct type parameter");
            const param_name = try parser.base.consume(.Identifier, "Expected type parameter name");
            _ = try parser.base.consume(.Colon, "Expected ':' after type parameter name");
            // Expect 'type' keyword (parsed as Identifier since it's not a keyword)
            const type_tok = try parser.base.consume(.Identifier, "Expected 'type' after ':'");
            if (!std.mem.eql(u8, type_tok.lexeme, "type")) {
                try parser.base.errorAtCurrent("Expected 'type' keyword for struct type parameter");
                return error.UnexpectedToken;
            }
            try type_param_names.append(alloc, param_name.lexeme);
            if (!parser.base.match(.Comma)) break;
        }
        _ = try parser.base.consume(.RightParen, "Expected ')' after struct type parameters");
        is_generic = type_param_names.items.len > 0;
    }

    _ = try parser.base.consume(.LeftBrace, "Expected '{' after struct name");

    var fields = std.ArrayList(ast.StructField){};
    defer fields.deinit(alloc);

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

        try fields.append(alloc, ast.StructField{
            .name = field_name.lexeme,
            .type_info = field_type,
            .span = parser.base.spanFromToken(field_name),
        });
    }

    _ = try parser.base.consume(.RightBrace, "Expected '}' after struct fields");

    return ast.AstNode{ .StructDecl = ast.StructDeclNode{
        .name = name_token.lexeme,
        .fields = try fields.toOwnedSlice(alloc),
        .is_generic = is_generic,
        .type_param_names = if (is_generic) try type_param_names.toOwnedSlice(alloc) else &.{},
        .span = parser.base.spanFromToken(name_token),
    } };
}

/// Parse bitfield declaration: bitfield Name : BaseInt { fields }
pub fn parseBitfield(parser: *DeclarationParser, type_parser: *TypeParser) !ast.AstNode {
    const name_token = try parser.base.consume(.Identifier, "Expected bitfield name");

    // Parse optional base type (: BaseInt)
    var base_type_info = ast.Types.TypeInfo.fromOraType(.u256); // default u256
    if (parser.base.match(.Colon)) {
        type_parser.base.current = parser.base.current;
        base_type_info = try type_parser.parseType();
        parser.base.current = type_parser.base.current;
    }

    _ = try parser.base.consume(.LeftBrace, "Expected '{' after bitfield declaration");

    var fields = std.ArrayList(ast.BitfieldField){};
    defer fields.deinit(parser.base.arena.allocator());
    var auto_packed = true;
    var any_at = false;
    var any_no_at = false;

    while (!parser.base.check(.RightBrace) and !parser.base.isAtEnd()) {
        const field_name = try parser.base.consumeIdentifierOrKeyword("Expected field name");
        _ = try parser.base.consume(.Colon, "Expected ':' after field name");

        // Parse field type
        type_parser.base.current = parser.base.current;
        const field_type = try type_parser.parseType();
        parser.base.current = type_parser.base.current;

        // Parse optional width annotation: (width)
        var explicit_width: ?u32 = null;
        if (parser.base.match(.LeftParen)) {
            const w_tok = try parser.base.consume(.IntegerLiteral, "Expected width in field width annotation");
            explicit_width = std.fmt.parseInt(u32, w_tok.lexeme, 10) catch null;
            _ = try parser.base.consume(.RightParen, "Expected ')' after width");
        }

        // Parse optional @at(offset, width) or @bits(start..end)
        var offset: ?u32 = null;
        var width: ?u32 = explicit_width;
        if (parser.base.match(.At)) {
            const at_name = try parser.base.consume(.Identifier, "Expected 'at' or 'bits' after '@'");
            if (std.mem.eql(u8, at_name.lexeme, "at")) {
                _ = try parser.base.consume(.LeftParen, "Expected '(' after '@at'");
                const off_tok = try parser.base.consume(.IntegerLiteral, "Expected offset in @at");
                offset = std.fmt.parseInt(u32, off_tok.lexeme, 10) catch null;
                _ = try parser.base.consume(.Comma, "Expected ',' between offset and width in @at");
                const w_tok = try parser.base.consume(.IntegerLiteral, "Expected width in @at");
                width = std.fmt.parseInt(u32, w_tok.lexeme, 10) catch null;
                _ = try parser.base.consume(.RightParen, "Expected ')' after @at arguments");
            } else if (std.mem.eql(u8, at_name.lexeme, "bits")) {
                // @bits(start..end) desugars to @at(start, end - start)
                _ = try parser.base.consume(.LeftParen, "Expected '(' after '@bits'");
                const start_tok = try parser.base.consume(.IntegerLiteral, "Expected start bit in @bits");
                const start = std.fmt.parseInt(u32, start_tok.lexeme, 10) catch null;
                _ = try parser.base.consume(.DotDot, "Expected '..' in @bits range");
                const end_tok = try parser.base.consume(.IntegerLiteral, "Expected end bit in @bits");
                const end = std.fmt.parseInt(u32, end_tok.lexeme, 10) catch null;
                _ = try parser.base.consume(.RightParen, "Expected ')' after @bits arguments");
                if (start != null and end != null) {
                    offset = start;
                    width = end.? - start.?;
                }
            } else {
                try parser.base.errorAtCurrent("Expected '@at' or '@bits' annotation for bitfield field");
                return error.UnexpectedToken;
            }
            any_at = true;
        } else {
            any_no_at = true;
        }

        _ = try parser.base.consume(.Semicolon, "Expected ';' after field");

        try fields.append(parser.base.arena.allocator(), ast.BitfieldField{
            .name = field_name.lexeme,
            .type_info = field_type,
            .offset = offset,
            .width = width,
            .span = parser.base.spanFromToken(field_name),
        });
    }

    _ = try parser.base.consume(.RightBrace, "Expected '}' after bitfield fields");

    // Mixing @at and auto-packed is an error
    if (any_at and any_no_at) {
        try parser.base.errorAtCurrent("Cannot mix @at() and auto-packed fields in a bitfield");
        return error.UnexpectedToken;
    }
    auto_packed = !any_at;

    // Resolve auto-packed fields: derive widths from types and compute cumulative offsets
    if (auto_packed) {
        var cursor: u32 = 0;
        const items = fields.items;
        for (items) |*field| {
            // Derive width from the field type if not explicitly given
            if (field.width == null) {
                field.width = if (field.type_info.ora_type) |ot| ot.bitWidth() else null;
            }
            field.offset = cursor;
            cursor += field.width orelse 0;
        }
    } else {
        // Explicit @at: derive width from type if only offset was given via @bits or inline (width)
        const items = fields.items;
        for (items) |*field| {
            if (field.width == null) {
                field.width = if (field.type_info.ora_type) |ot| ot.bitWidth() else null;
            }
        }
    }

    return ast.AstNode{ .BitfieldDecl = ast.BitfieldDeclNode{
        .name = name_token.lexeme,
        .base_type_info = base_type_info,
        .fields = try fields.toOwnedSlice(parser.base.arena.allocator()),
        .auto_packed = auto_packed,
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
