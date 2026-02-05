// ============================================================================
// Type Parser
// ============================================================================
//
// Parses type expressions and annotations.
//
// SUPPORTED TYPES:
//   • Primitives: uint, int, bool, address, string, bytes
//   • Collections: arrays, maps, slices
//   • Complex: structs, enums, function types, error unions
//
// ============================================================================

const std = @import("std");
const lexer = @import("../lexer.zig");
const ast = @import("../ast.zig");
const common = @import("common.zig");

const Token = lexer.Token;
const TokenType = lexer.TokenType;
const BaseParser = common.BaseParser;
const ParserCommon = common.ParserCommon;
const ParserError = @import("parser_core.zig").ParserError;
const TypeInfo = ast.Types.TypeInfo;
const OraType = ast.Types.OraType;
const TypeCategory = ast.TypeCategory;
const SourceSpan = ast.SourceSpan;

/// Represents the context in which a type is being parsed
pub const TypeParseContext = enum {
    Any, // Any general context where a type is needed
    ReturnType, // Function return type position
    Parameter, // Function parameter type position
    Variable, // Variable declaration type position
    StructField, // Struct field type position
    EnumUnderlying, // Enum underlying type position
    LogField, // Log field type position
    ErrorUnionMember, // Parsing a type inside an error union (after |)
};

/// Specialized parser for type expressions
pub const TypeParser = struct {
    base: BaseParser,

    pub fn init(tokens: []const Token, arena: *@import("../ast/ast_arena.zig").AstArena) TypeParser {
        return TypeParser{
            .base = BaseParser.init(tokens, arena),
        };
    }

    /// Parse type information with default context
    pub fn parseType(self: *TypeParser) ParserError!TypeInfo {
        return self.parseTypeWithContext(.Any);
    }

    /// Parse a return type, which can include void
    pub fn parseReturnType(self: *TypeParser) ParserError!TypeInfo {
        return self.parseTypeWithContext(.ReturnType);
    }

    /// Parse type information with context awareness
    pub fn parseTypeWithContext(self: *TypeParser, context: TypeParseContext) ParserError!TypeInfo {
        const span = self.base.currentSpan();

        // handle primitive types using TypeInfo.explicit()
        if (self.base.match(.U8)) return TypeInfo.explicit(.Integer, .u8, span);
        if (self.base.match(.U16)) return TypeInfo.explicit(.Integer, .u16, span);
        if (self.base.match(.U32)) return TypeInfo.explicit(.Integer, .u32, span);
        if (self.base.match(.U64)) return TypeInfo.explicit(.Integer, .u64, span);
        if (self.base.match(.U128)) return TypeInfo.explicit(.Integer, .u128, span);
        if (self.base.match(.U256)) return TypeInfo.explicit(.Integer, .u256, span);
        if (self.base.match(.I8)) return TypeInfo.explicit(.Integer, .i8, span);
        if (self.base.match(.I16)) return TypeInfo.explicit(.Integer, .i16, span);
        if (self.base.match(.I32)) return TypeInfo.explicit(.Integer, .i32, span);
        if (self.base.match(.I64)) return TypeInfo.explicit(.Integer, .i64, span);
        if (self.base.match(.I128)) return TypeInfo.explicit(.Integer, .i128, span);
        if (self.base.match(.I256)) return TypeInfo.explicit(.Integer, .i256, span);
        if (self.base.match(.Bool)) return TypeInfo.explicit(.Bool, .bool, span);
        if (self.base.match(.Address)) return TypeInfo.explicit(.Address, .address, span);
        if (self.base.match(.String)) return TypeInfo.explicit(.String, .string, span);
        if (self.base.match(.Bytes)) return TypeInfo.explicit(.Bytes, .bytes, span);

        // void is only valid in return type positions
        if (self.base.match(.Void)) {
            if (context == .ReturnType) {
                return TypeInfo.explicit(.Void, .void, span);
            } else {
                try self.base.errorAtCurrent("'void' can only be used as a function return type");
                return error.UnexpectedToken;
            }
        }

        // tuple type: (T1, T2, ...)
        if (self.base.match(.LeftParen)) {
            return try self.parseTupleType(context, span);
        }

        // handle complex types
        if (self.base.match(.Map)) {
            return try self.parseMapType(context);
        }

        if (self.base.match(.Slice)) {
            return try self.parseSliceType(context);
        }

        if (self.base.match(.LeftBracket)) {
            return try self.parseArrayType(context);
        }

        if (self.base.match(.Identifier)) {
            const left = try self.parseNamedType(context);
            // parse error unions only: !T | E or similar not supported; '|' only allowed if left is an error union
            // note: When parsing error union members (ErrorUnionMember context), we don't handle | here
            // because the parent parseErrorUnionType handles all | tokens.
            if (context != .ErrorUnionMember and self.base.match(.Pipe)) {
                // only allow union when the left is an error union (!T)
                if (left.category != .ErrorUnion) {
                    try self.base.errorAtCurrent("'|' is only allowed in error unions (e.g., !T | E)");
                    return error.UnexpectedToken;
                }
                var types = std.ArrayList(ast.Types.OraType){};
                defer types.deinit(self.base.arena.allocator());
                // push first
                try types.append(self.base.arena.allocator(), left.ora_type orelse return error.UnresolvedType);
                // parse subsequent types
                while (true) {
                    const next = try self.parseTypeWithContext(context);
                    try types.append(self.base.arena.allocator(), next.ora_type orelse return error.UnresolvedType);
                    if (!self.base.match(.Pipe)) break;
                }
                const slice_types = try self.base.arena.createSlice(ast.Types.OraType, types.items.len);
                for (types.items, 0..) |t, i| slice_types[i] = t;
                return TypeInfo{
                    .category = .ErrorUnion,
                    .ora_type = ast.Types.OraType{ ._union = slice_types },
                    .source = .explicit,
                    .span = span,
                };
            }
            return left;
        }

        // anonymous struct type: struct { field: T, ... }
        if (self.base.match(.Struct)) {
            _ = try self.base.consume(.LeftBrace, "Expected '{' after 'struct'");
            var fields = std.ArrayList(ast.Types.AnonymousStructFieldType){};
            defer fields.deinit(self.base.arena.allocator());
            if (!self.base.check(.RightBrace)) {
                repeat: while (true) {
                    const name_tok = try self.base.consume(.Identifier, "Expected field name in anonymous struct type");
                    _ = try self.base.consume(.Colon, "Expected ':' after field name");
                    const field_type = try self.parseTypeWithContext(context);
                    const field_type_ptr = try self.base.arena.createNode(ast.Types.OraType);
                    field_type_ptr.* = field_type.ora_type orelse return error.UnresolvedType;
                    try fields.append(self.base.arena.allocator(), .{ .name = name_tok.lexeme, .typ = field_type_ptr });
                    if (!self.base.match(.Comma)) break :repeat;
                }
            }
            _ = try self.base.consume(.RightBrace, "Expected '}' after anonymous struct fields");
            const field_slice = try self.base.arena.createSlice(ast.type_info.AnonymousStructFieldType, fields.items.len);
            for (fields.items, 0..) |f, i| field_slice[i] = f;
            return TypeInfo{
                .category = .Struct,
                .ora_type = ast.Types.OraType{ .anonymous_struct = field_slice },
                .source = .explicit,
                .span = span,
            };
        }

        // error union type (!T)
        if (self.base.match(.Bang)) {
            return try self.parseErrorUnionType(context);
        }

        try self.base.errorAtCurrent("Expected type");
        return error.UnexpectedToken;
    }

    /// Parse tuple type: (T1, T2, ...)
    fn parseTupleType(self: *TypeParser, context: TypeParseContext, span: SourceSpan) ParserError!TypeInfo {
        // empty tuple: ()
        if (self.base.check(.RightParen)) {
            _ = try self.base.consume(.RightParen, "Expected ')' after tuple type");
            const empty = try self.base.arena.createSlice(OraType, 0);
            return TypeInfo{
                .category = .Tuple,
                .ora_type = OraType{ .tuple = empty },
                .source = .explicit,
                .span = span,
            };
        }

        const first = try self.parseTypeWithContext(context);

        // single type in parens: (T) -> T
        if (!self.base.match(.Comma)) {
            _ = try self.base.consume(.RightParen, "Expected ')' after type");
            return first;
        }

        var types = std.ArrayList(TypeInfo){};
        defer types.deinit(self.base.arena.allocator());
        try types.append(self.base.arena.allocator(), first);

        if (!self.base.check(.RightParen)) {
            repeat: while (true) {
                const next = try self.parseTypeWithContext(context);
                try types.append(self.base.arena.allocator(), next);
                if (!self.base.match(.Comma)) break :repeat;
                if (self.base.check(.RightParen)) break :repeat;
            }
        }

        _ = try self.base.consume(.RightParen, "Expected ')' after tuple elements");

        const ora_types = try self.base.arena.createSlice(OraType, types.items.len);
        for (types.items, 0..) |t, i| {
            if (t.ora_type == null) return ParserError.UnresolvedType;
            ora_types[i] = t.ora_type.?;
        }

        return TypeInfo{
            .category = .Tuple,
            .ora_type = OraType{ .tuple = ora_types },
            .source = .explicit,
            .span = span,
        };
    }

    /// Parse map type: map[K, V]
    fn parseMapType(self: *TypeParser, context: TypeParseContext) ParserError!TypeInfo {
        const span = self.base.currentSpan();
        _ = try self.base.consume(.LeftBracket, "Expected '[' after 'map'");

        const key_type_info = try self.parseTypeWithContext(context);
        _ = try self.base.consume(.Comma, "Expected ',' after map key type");
        const value_type_info = try self.parseTypeWithContext(context);
        _ = try self.base.consume(.RightBracket, "Expected ']' after map value type");

        // create OraType for mapping
        const key_ora_type = try self.base.arena.createNode(OraType);
        key_ora_type.* = key_type_info.ora_type orelse return error.UnresolvedType;

        const value_ora_type = try self.base.arena.createNode(OraType);
        value_ora_type.* = value_type_info.ora_type orelse return error.UnresolvedType;

        const map_type = ast.type_info.MapType{
            .key = key_ora_type,
            .value = value_ora_type,
        };

        return TypeInfo{
            .category = .Map,
            .ora_type = OraType{ .map = map_type },
            .source = .explicit,
            .span = span,
        };
    }

    /// Parse array types [T; N] and [T]
    fn parseArrayType(self: *TypeParser, context: TypeParseContext) ParserError!TypeInfo {
        const span = self.base.currentSpan();
        const elem_type_info = try self.parseTypeWithContext(context);

        if (self.base.match(.Semicolon)) {
            // fixed array: [T; N]
            const size_tok = try self.base.consume(.IntegerLiteral, "Expected array size after ';'");
            _ = try self.base.consume(.RightBracket, "Expected ']' after array size");

            // create OraType for array
            const elem_ora_type = try self.base.arena.createNode(OraType);
            elem_ora_type.* = elem_type_info.ora_type orelse return error.UnresolvedType;

            const size_val: u64 = std.fmt.parseInt(u64, size_tok.lexeme, 10) catch 0;

            return TypeInfo{
                .category = .Array,
                .ora_type = OraType{ .array = .{ .elem = elem_ora_type, .len = size_val } },
                .source = .explicit,
                .span = span,
            };
        } else {
            // deprecated dynamic array syntax [T] is not supported.
            // users should write slice[T] for dynamic sequences.
            _ = try self.base.consume(.RightBracket, "Expected ']' after array element type");
            try self.base.errorAtCurrent("Dynamic sequence syntax '[T]' is not supported; use 'slice[T]' instead");
            return error.UnexpectedToken;
        }
    }

    /// Parse named types (identifiers that could be structs, enums, or complex types)
    fn parseNamedType(self: *TypeParser, context: TypeParseContext) ParserError!TypeInfo {
        const span = self.base.currentSpan();
        const type_name = self.base.previous().lexeme;

        // check for complex types
        if (std.mem.eql(u8, type_name, "slice")) {
            return try self.parseSliceType(context);
        }

        // check for refinement types (MinValue, MaxValue, InRange, Scaled, Exact, NonZero, BasisPoints)
        if (std.mem.eql(u8, type_name, "MinValue")) {
            return try self.parseMinValueType(context);
        }
        if (std.mem.eql(u8, type_name, "MaxValue")) {
            return try self.parseMaxValueType(context);
        }
        if (std.mem.eql(u8, type_name, "InRange")) {
            return try self.parseInRangeType(context);
        }
        if (std.mem.eql(u8, type_name, "Scaled")) {
            return try self.parseScaledType(context);
        }
        if (std.mem.eql(u8, type_name, "Exact")) {
            return try self.parseExactType(context);
        }
        if (std.mem.eql(u8, type_name, "NonZero")) {
            return try self.parseNonZeroType(context);
        }
        if (std.mem.eql(u8, type_name, "BasisPoints")) {
            return try self.parseBasisPointsType(context);
        }
        if (std.mem.eql(u8, type_name, "NonZeroAddress")) {
            return try self.parseNonZeroAddressType(context);
        }

        // note: 'Result[T,E]' is not supported; use error unions '!T | E' instead.

        // custom type (struct or enum - will be resolved during semantic analysis)
        return TypeInfo{
            .category = .Struct, // Assume struct for now, will be resolved later
            .ora_type = OraType{ .struct_type = type_name },
            .source = .explicit,
            .span = span,
        };
    }

    /// Parse slice type: slice[T]
    fn parseSliceType(self: *TypeParser, context: TypeParseContext) ParserError!TypeInfo {
        const span = self.base.currentSpan();
        _ = try self.base.consume(.LeftBracket, "Expected '[' after 'slice'");
        const elem_type_info = try self.parseTypeWithContext(context);
        _ = try self.base.consume(.RightBracket, "Expected ']' after slice element type");

        // create OraType for slice
        const elem_ora_type = try self.base.arena.createNode(OraType);
        elem_ora_type.* = elem_type_info.ora_type orelse return error.UnresolvedType;

        return TypeInfo{
            .category = .Slice,
            .ora_type = OraType{ .slice = elem_ora_type },
            .source = .explicit,
            .span = span,
        };
    }

    // result[T,E] removed; prefer error unions '!T | E'.

    /// Parse error union type: !T
    fn parseErrorUnionType(self: *TypeParser, context: TypeParseContext) ParserError!TypeInfo {
        const span = self.base.currentSpan();
        const success_type_info = try self.parseTypeWithContext(context);

        // create OraType for error union
        const success_ora_type = try self.base.arena.createNode(OraType);
        success_ora_type.* = success_type_info.ora_type orelse return error.UnresolvedType;

        // support optional explicit error list continuation: !T | E1 | E2 ...
        if (self.base.match(.Pipe)) {
            var types = std.ArrayList(ast.Types.OraType){};
            defer types.deinit(self.base.arena.allocator());
            // first member is the initial !T
            try types.append(self.base.arena.allocator(), OraType{ .error_union = success_ora_type });
            // parse subsequent types with ErrorUnionMember context to prevent | handling
            while (true) {
                const next = try self.parseTypeWithContext(.ErrorUnionMember);
                try types.append(self.base.arena.allocator(), next.ora_type orelse return error.UnresolvedType);
                if (!self.base.match(.Pipe)) break;
            }
            const slice_types = try self.base.arena.createSlice(ast.Types.OraType, types.items.len);
            for (types.items, 0..) |t, i| slice_types[i] = t;
            return TypeInfo{
                .category = .ErrorUnion,
                .ora_type = OraType{ ._union = slice_types },
                .source = .explicit,
                .span = span,
            };
        }

        return TypeInfo{
            .category = .ErrorUnion,
            .ora_type = OraType{ .error_union = success_ora_type },
            .source = .explicit,
            .span = span,
        };
    }

    // ============================================================================
    // refinement Type Parsers
    // ============================================================================

    /// Parse a compile-time integer constant (for refinement type parameters)
    fn parseCompileTimeInt(self: *TypeParser) ParserError!u256 {
        if (!self.base.match(.IntegerLiteral)) {
            try self.base.errorAtCurrent("Expected compile-time integer constant");
            return error.UnexpectedToken;
        }
        const token = self.base.previous();
        // parse integer literal (supports decimal, hex, binary)
        const value = std.fmt.parseInt(u256, token.lexeme, 0) catch {
            try self.base.errorAtCurrent("Integer constant too large");
            return error.UnexpectedToken;
        };
        return value;
    }

    /// Parse MinValue<T, N> refinement type
    fn parseMinValueType(self: *TypeParser, context: TypeParseContext) ParserError!TypeInfo {
        const span = self.base.currentSpan();
        _ = try self.base.consume(.Less, "Expected '<' after 'MinValue'");

        // parse base type
        const base_type_info = try self.parseTypeWithContext(context);
        const base_ora_type = base_type_info.ora_type orelse return error.UnresolvedType;
        _ = try self.base.consume(.Comma, "Expected ',' after base type");

        // parse minimum value (compile-time constant)
        const min_value = try self.parseCompileTimeInt();
        _ = try self.base.consume(.Greater, "Expected '>' after MinValue parameters");

        // create base type pointer
        const base_type_ptr = try self.base.arena.createNode(OraType);
        base_type_ptr.* = base_ora_type;

        return TypeInfo{
            .category = base_type_info.category,
            .ora_type = OraType{ .min_value = .{ .base = base_type_ptr, .min = min_value } },
            .source = .explicit,
            .span = span,
        };
    }

    /// Parse MaxValue<T, N> refinement type
    fn parseMaxValueType(self: *TypeParser, context: TypeParseContext) ParserError!TypeInfo {
        const span = self.base.currentSpan();
        _ = try self.base.consume(.Less, "Expected '<' after 'MaxValue'");

        // parse base type
        const base_type_info = try self.parseTypeWithContext(context);
        const base_ora_type = base_type_info.ora_type orelse return error.UnresolvedType;
        _ = try self.base.consume(.Comma, "Expected ',' after base type");

        // parse maximum value (compile-time constant)
        const max_value = try self.parseCompileTimeInt();
        _ = try self.base.consume(.Greater, "Expected '>' after MaxValue parameters");

        // create base type pointer
        const base_type_ptr = try self.base.arena.createNode(OraType);
        base_type_ptr.* = base_ora_type;

        return TypeInfo{
            .category = base_type_info.category,
            .ora_type = OraType{ .max_value = .{ .base = base_type_ptr, .max = max_value } },
            .source = .explicit,
            .span = span,
        };
    }

    /// Parse InRange<T, MIN, MAX> refinement type
    fn parseInRangeType(self: *TypeParser, context: TypeParseContext) ParserError!TypeInfo {
        const span = self.base.currentSpan();
        _ = try self.base.consume(.Less, "Expected '<' after 'InRange'");

        // parse base type
        const base_type_info = try self.parseTypeWithContext(context);
        const base_ora_type = base_type_info.ora_type orelse return error.UnresolvedType;
        _ = try self.base.consume(.Comma, "Expected ',' after base type");

        // parse minimum value
        const min_value = try self.parseCompileTimeInt();
        _ = try self.base.consume(.Comma, "Expected ',' after minimum value");

        // parse maximum value
        const max_value = try self.parseCompileTimeInt();
        _ = try self.base.consume(.Greater, "Expected '>' after InRange parameters");

        // validate: MIN <= MAX (will be checked in type resolution, but catch obvious errors here)
        if (min_value > max_value) {
            try self.base.errorAtCurrent("Minimum value must be <= maximum value");
            return error.UnexpectedToken;
        }

        // create base type pointer
        const base_type_ptr = try self.base.arena.createNode(OraType);
        base_type_ptr.* = base_ora_type;

        return TypeInfo{
            .category = base_type_info.category,
            .ora_type = OraType{ .in_range = .{ .base = base_type_ptr, .min = min_value, .max = max_value } },
            .source = .explicit,
            .span = span,
        };
    }

    /// Parse Scaled<T, D> refinement type
    fn parseScaledType(self: *TypeParser, context: TypeParseContext) ParserError!TypeInfo {
        const span = self.base.currentSpan();
        _ = try self.base.consume(.Less, "Expected '<' after 'Scaled'");

        // parse base type
        const base_type_info = try self.parseTypeWithContext(context);
        const base_ora_type = base_type_info.ora_type orelse return error.UnresolvedType;
        _ = try self.base.consume(.Comma, "Expected ',' after base type");

        // parse decimals (compile-time constant, u32)
        const decimals_token = try self.base.consume(.IntegerLiteral, "Expected decimal places constant");
        const decimals = std.fmt.parseInt(u32, decimals_token.lexeme, 0) catch {
            try self.base.errorAtCurrent("Decimal places must be a valid u32");
            return error.UnexpectedToken;
        };
        _ = try self.base.consume(.Greater, "Expected '>' after Scaled parameters");

        // create base type pointer
        const base_type_ptr = try self.base.arena.createNode(OraType);
        base_type_ptr.* = base_ora_type;

        return TypeInfo{
            .category = base_type_info.category,
            .ora_type = OraType{ .scaled = .{ .base = base_type_ptr, .decimals = decimals } },
            .source = .explicit,
            .span = span,
        };
    }

    /// Parse Exact<T> refinement type
    fn parseExactType(self: *TypeParser, context: TypeParseContext) ParserError!TypeInfo {
        const span = self.base.currentSpan();
        _ = try self.base.consume(.Less, "Expected '<' after 'Exact'");

        // parse base type
        const base_type_info = try self.parseTypeWithContext(context);
        const base_ora_type = base_type_info.ora_type orelse return error.UnresolvedType;
        _ = try self.base.consume(.Greater, "Expected '>' after Exact base type");

        // create base type pointer
        const base_type_ptr = try self.base.arena.createNode(OraType);
        base_type_ptr.* = base_ora_type;

        return TypeInfo{
            .category = base_type_info.category,
            .ora_type = OraType{ .exact = base_type_ptr },
            .source = .explicit,
            .span = span,
        };
    }

    /// Parse NonZero<T> type alias (alias for MinValue<T, 1>)
    fn parseNonZeroType(self: *TypeParser, context: TypeParseContext) ParserError!TypeInfo {
        const span = self.base.currentSpan();
        _ = try self.base.consume(.Less, "Expected '<' after 'NonZero'");

        // parse base type
        const base_type_info = try self.parseTypeWithContext(context);
        const base_ora_type = base_type_info.ora_type orelse return error.UnresolvedType;
        _ = try self.base.consume(.Greater, "Expected '>' after NonZero base type");

        // create base type pointer
        const base_type_ptr = try self.base.arena.createNode(OraType);
        base_type_ptr.* = base_ora_type;

        // nonZero<T> = MinValue<T, 1>
        return TypeInfo{
            .category = base_type_info.category,
            .ora_type = OraType{ .min_value = .{ .base = base_type_ptr, .min = 1 } },
            .source = .explicit,
            .span = span,
        };
    }

    /// Parse BasisPoints<T> type alias (alias for InRange<T, 0, 10000>)
    fn parseBasisPointsType(self: *TypeParser, context: TypeParseContext) ParserError!TypeInfo {
        const span = self.base.currentSpan();
        _ = try self.base.consume(.Less, "Expected '<' after 'BasisPoints'");

        // parse base type
        const base_type_info = try self.parseTypeWithContext(context);
        const base_ora_type = base_type_info.ora_type orelse return error.UnresolvedType;
        _ = try self.base.consume(.Greater, "Expected '>' after BasisPoints base type");

        // create base type pointer
        const base_type_ptr = try self.base.arena.createNode(OraType);
        base_type_ptr.* = base_ora_type;

        // basisPoints<T> = InRange<T, 0, 10000>
        return TypeInfo{
            .category = base_type_info.category,
            .ora_type = OraType{ .in_range = .{ .base = base_type_ptr, .min = 0, .max = 10000 } },
            .source = .explicit,
            .span = span,
        };
    }

    /// Parse NonZeroAddress type (address that cannot be zero)
    fn parseNonZeroAddressType(self: *TypeParser, _: TypeParseContext) ParserError!TypeInfo {
        const span = self.base.currentSpan();
        // nonZeroAddress doesn't take type parameters - it's just NonZeroAddress
        // check if there's a < after NonZeroAddress (shouldn't be)
        if (self.base.check(.Less)) {
            try self.base.errorAtCurrent("NonZeroAddress does not take type parameters");
            return error.UnexpectedToken;
        }

        return TypeInfo{
            .category = .Address,
            .ora_type = OraType{ .non_zero_address = {} },
            .source = .explicit,
            .span = span,
        };
    }
};
