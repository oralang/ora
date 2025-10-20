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

/// Represents the context in which a type is being parsed
pub const TypeParseContext = enum {
    Any, // Any general context where a type is needed
    ReturnType, // Function return type position
    Parameter, // Function parameter type position
    Variable, // Variable declaration type position
    StructField, // Struct field type position
    EnumUnderlying, // Enum underlying type position
    LogField, // Log field type position
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

        // Handle primitive types using TypeInfo.explicit()
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

        // Handle complex types
        if (self.base.match(.Map)) {
            return try self.parseMapType(context);
        }

        if (self.base.match(.DoubleMap)) {
            return try self.parseDoubleMapType(context);
        }

        if (self.base.match(.Slice)) {
            return try self.parseSliceType(context);
        }

        if (self.base.match(.LeftBracket)) {
            return try self.parseArrayType(context);
        }

        if (self.base.match(.Identifier)) {
            const left = try self.parseNamedType(context);
            // Parse error unions only: !T | E or similar not supported; '|' only allowed if left is an error union
            if (self.base.match(.Pipe)) {
                // Only allow union when the left is an error union (!T)
                if (left.category != .ErrorUnion) {
                    try self.base.errorAtCurrent("'|' is only allowed in error unions (e.g., !T | E)");
                    return error.UnexpectedToken;
                }
                var types = std.ArrayList(ast.Types.OraType){};
                defer types.deinit(self.base.arena.allocator());
                // Push first
                try types.append(self.base.arena.allocator(), left.ora_type orelse return error.UnresolvedType);
                // Parse subsequent types
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

        // Anonymous struct type: struct { field: T, ... }
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

        // Error union type (!T)
        if (self.base.match(.Bang)) {
            return try self.parseErrorUnionType(context);
        }

        try self.base.errorAtCurrent("Expected type");
        return error.UnexpectedToken;
    }

    /// Parse map type: map[K, V]
    fn parseMapType(self: *TypeParser, context: TypeParseContext) ParserError!TypeInfo {
        const span = self.base.currentSpan();
        _ = try self.base.consume(.LeftBracket, "Expected '[' after 'map'");

        const key_type_info = try self.parseTypeWithContext(context);
        _ = try self.base.consume(.Comma, "Expected ',' after map key type");
        const value_type_info = try self.parseTypeWithContext(context);
        _ = try self.base.consume(.RightBracket, "Expected ']' after map value type");

        // Create OraType for mapping
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

    /// Parse doublemap type: doublemap[K1, K2, V]
    fn parseDoubleMapType(self: *TypeParser, context: TypeParseContext) ParserError!TypeInfo {
        const span = self.base.currentSpan();
        _ = try self.base.consume(.LeftBracket, "Expected '[' after 'doublemap'");

        const key1_type_info = try self.parseTypeWithContext(context);
        _ = try self.base.consume(.Comma, "Expected ',' after first key type");

        const key2_type_info = try self.parseTypeWithContext(context);
        _ = try self.base.consume(.Comma, "Expected ',' after second key type");

        const value_type_info = try self.parseTypeWithContext(context);
        _ = try self.base.consume(.RightBracket, "Expected ']' after doublemap value type");

        // Create OraType for double mapping
        const key1_ora_type = try self.base.arena.createNode(OraType);
        key1_ora_type.* = key1_type_info.ora_type orelse return error.UnresolvedType;

        const key2_ora_type = try self.base.arena.createNode(OraType);
        key2_ora_type.* = key2_type_info.ora_type orelse return error.UnresolvedType;

        const value_ora_type = try self.base.arena.createNode(OraType);
        value_ora_type.* = value_type_info.ora_type orelse return error.UnresolvedType;

        const doublemap_type = ast.type_info.DoubleMapType{
            .key1 = key1_ora_type,
            .key2 = key2_ora_type,
            .value = value_ora_type,
        };

        return TypeInfo{
            .category = .DoubleMap,
            .ora_type = OraType{ .double_map = doublemap_type },
            .source = .explicit,
            .span = span,
        };
    }

    /// Parse array types [T; N] and [T]
    fn parseArrayType(self: *TypeParser, context: TypeParseContext) ParserError!TypeInfo {
        const span = self.base.currentSpan();
        const elem_type_info = try self.parseTypeWithContext(context);

        if (self.base.match(.Semicolon)) {
            // Fixed array: [T; N]
            const size_tok = try self.base.consume(.IntegerLiteral, "Expected array size after ';'");
            _ = try self.base.consume(.RightBracket, "Expected ']' after array size");

            // Create OraType for array
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
            // Deprecated dynamic array syntax [T] is not supported.
            // Users should write slice[T] for dynamic sequences.
            _ = try self.base.consume(.RightBracket, "Expected ']' after array element type");
            try self.base.errorAtCurrent("Dynamic sequence syntax '[T]' is not supported; use 'slice[T]' instead");
            return error.UnexpectedToken;
        }
    }

    /// Parse named types (identifiers that could be structs, enums, or complex types)
    fn parseNamedType(self: *TypeParser, context: TypeParseContext) ParserError!TypeInfo {
        const span = self.base.currentSpan();
        const type_name = self.base.previous().lexeme;

        // Check for complex types
        if (std.mem.eql(u8, type_name, "slice")) {
            return try self.parseSliceType(context);
        }

        if (std.mem.eql(u8, type_name, "doublemap")) {
            return try self.parseDoubleMapType(context);
        }

        // Note: 'Result[T,E]' is not supported; use error unions '!T | E' instead.

        // Custom type (struct or enum - will be resolved during semantic analysis)
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

        // Create OraType for slice
        const elem_ora_type = try self.base.arena.createNode(OraType);
        elem_ora_type.* = elem_type_info.ora_type orelse return error.UnresolvedType;

        return TypeInfo{
            .category = .Slice,
            .ora_type = OraType{ .slice = elem_ora_type },
            .source = .explicit,
            .span = span,
        };
    }

    // Result[T,E] removed; prefer error unions '!T | E'.

    /// Parse error union type: !T
    fn parseErrorUnionType(self: *TypeParser, context: TypeParseContext) ParserError!TypeInfo {
        const span = self.base.currentSpan();
        const success_type_info = try self.parseTypeWithContext(context);

        // Create OraType for error union
        const success_ora_type = try self.base.arena.createNode(OraType);
        success_ora_type.* = success_type_info.ora_type orelse return error.UnresolvedType;

        // Support optional explicit error list continuation: !T | E1 | E2 ...
        if (self.base.match(.Pipe)) {
            var types = std.ArrayList(ast.Types.OraType){};
            defer types.deinit(self.base.arena.allocator());
            // First member is the initial !T
            try types.append(self.base.arena.allocator(), OraType{ .error_union = success_ora_type });
            // Parse subsequent types
            while (true) {
                const next = try self.parseTypeWithContext(context);
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
};
