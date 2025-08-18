const std = @import("std");
pub const ast = @import("../ast.zig");
pub const typer = @import("../typer.zig");
const semantics_errors = @import("semantics_errors.zig");
const semantics_memory_safety = @import("semantics_memory_safety.zig");

// Forward declaration for SemanticAnalyzer
const SemanticAnalyzer = @import("semantics_core.zig").SemanticAnalyzer;

/// Utility functions for semantic analysis
pub const SemanticUtils = struct {
    /// Check if an identifier is a valid Ora identifier
    pub fn isValidIdentifier(name: []const u8) bool {
        if (name.len == 0) return false;

        // First character must be letter or underscore
        if (!std.ascii.isAlphabetic(name[0]) and name[0] != '_') {
            return false;
        }

        // Remaining characters must be alphanumeric or underscore
        for (name[1..]) |c| {
            if (!std.ascii.isAlphanumeric(c) and c != '_') {
                return false;
            }
        }

        // Check against reserved keywords
        return !isReservedKeyword(name);
    }

    /// Check if a name is a reserved keyword
    pub fn isReservedKeyword(name: []const u8) bool {
        const keywords = [_][]const u8{
            "contract", "function",  "struct", "enum",    "import",    "export",
            "pub",      "const",     "var",    "let",     "if",        "else",
            "while",    "for",       "return", "break",   "continue",  "try",
            "catch",    "throw",     "true",   "false",   "null",      "undefined",
            "void",     "u8",        "u16",    "u32",     "u64",       "u128",
            "u256",     "i8",        "i16",    "i32",     "i64",       "i128",
            "i256",     "bool",      "string", "address", "mapping",   "array",
            "storage",  "memory",    "stack",  "heap",    "immutable", "requires",
            "ensures",  "invariant", "old",    "init",    "self",      "super",
            "this",
        };

        for (keywords) |keyword| {
            if (std.mem.eql(u8, name, keyword)) {
                return true;
            }
        }

        return false;
    }

    /// Extract identifier name from expression if it's an identifier
    pub fn extractIdentifierName(expr: *ast.ExprNode) ?[]const u8 {
        return switch (expr.*) {
            .Identifier => |*ident| ident.name,
            else => null,
        };
    }

    /// Check if expression is a literal
    pub fn isLiteralExpression(expr: *ast.ExprNode) bool {
        return switch (expr.*) {
            .Literal => true,
            else => false,
        };
    }

    /// Check if expression is a compile-time constant
    pub fn isComptimeConstant(expr: *ast.ExprNode) bool {
        return switch (expr.*) {
            .Literal => true,
            .Comptime => true,
            .Identifier => |*ident| {
                // Check if identifier refers to a const variable
                // This would require symbol table lookup in a full implementation
                _ = ident;
                return false;
            },
            .Binary => |*bin| {
                // Binary operations on constants are constant
                return isComptimeConstant(bin.left) and isComptimeConstant(bin.right);
            },
            .Unary => |*un| {
                return isComptimeConstant(un.operand);
            },
            else => false,
        };
    }

    /// Get the base type name from a type node
    pub fn getBaseTypeName(type_node: *ast.TypeNode) ?[]const u8 {
        return switch (type_node.*) {
            .Identifier => |*ident| ident.name,
            .Array => |*array| getBaseTypeName(array.element_type),
            .Mapping => |*mapping| getBaseTypeName(mapping.value_type),
            .ErrorUnion => |*error_union| getBaseTypeName(error_union.value_type),
            else => null,
        };
    }

    /// Check if type is a primitive type
    pub fn isPrimitiveType(type_name: []const u8) bool {
        const primitive_types = [_][]const u8{
            "u8",   "u16",    "u32",     "u64",  "u128", "u256",
            "i8",   "i16",    "i32",     "i64",  "i128", "i256",
            "bool", "string", "address", "void",
        };

        for (primitive_types) |primitive| {
            if (std.mem.eql(u8, type_name, primitive)) {
                return true;
            }
        }

        return false;
    }

    /// Check if type is a numeric type
    pub fn isNumericType(type_name: []const u8) bool {
        const numeric_types = [_][]const u8{
            "u8", "u16", "u32", "u64", "u128", "u256",
            "i8", "i16", "i32", "i64", "i128", "i256",
        };

        for (numeric_types) |numeric| {
            if (std.mem.eql(u8, type_name, numeric)) {
                return true;
            }
        }

        return false;
    }

    /// Check if type is an unsigned integer type
    pub fn isUnsignedIntegerType(type_name: []const u8) bool {
        const unsigned_types = [_][]const u8{
            "u8", "u16", "u32", "u64", "u128", "u256",
        };

        for (unsigned_types) |unsigned| {
            if (std.mem.eql(u8, type_name, unsigned)) {
                return true;
            }
        }

        return false;
    }

    /// Check if type is a signed integer type
    pub fn isSignedIntegerType(type_name: []const u8) bool {
        const signed_types = [_][]const u8{
            "i8", "i16", "i32", "i64", "i128", "i256",
        };

        for (signed_types) |signed| {
            if (std.mem.eql(u8, type_name, signed)) {
                return true;
            }
        }

        return false;
    }

    /// Get the bit width of an integer type
    pub fn getIntegerBitWidth(type_name: []const u8) ?u32 {
        if (std.mem.eql(u8, type_name, "u8") or std.mem.eql(u8, type_name, "i8")) return 8;
        if (std.mem.eql(u8, type_name, "u16") or std.mem.eql(u8, type_name, "i16")) return 16;
        if (std.mem.eql(u8, type_name, "u32") or std.mem.eql(u8, type_name, "i32")) return 32;
        if (std.mem.eql(u8, type_name, "u64") or std.mem.eql(u8, type_name, "i64")) return 64;
        if (std.mem.eql(u8, type_name, "u128") or std.mem.eql(u8, type_name, "i128")) return 128;
        if (std.mem.eql(u8, type_name, "u256") or std.mem.eql(u8, type_name, "i256")) return 256;
        return null;
    }

    /// Check if binary operation is valid for given types
    pub fn isBinaryOperationValid(op: ast.BinaryOp, left_type: []const u8, right_type: []const u8) bool {
        switch (op) {
            .Add, .Subtract, .Multiply, .Divide, .Modulo => {
                return isNumericType(left_type) and isNumericType(right_type);
            },
            .Equal, .NotEqual => {
                // Equality can be checked between same types
                return std.mem.eql(u8, left_type, right_type);
            },
            .LessThan, .LessThanOrEqual, .GreaterThan, .GreaterThanOrEqual => {
                return isNumericType(left_type) and isNumericType(right_type);
            },
            .And, .Or => {
                return std.mem.eql(u8, left_type, "bool") and std.mem.eql(u8, right_type, "bool");
            },
            .BitwiseAnd, .BitwiseOr, .BitwiseXor => {
                return isNumericType(left_type) and isNumericType(right_type);
            },
            .LeftShift, .RightShift => {
                return isNumericType(left_type) and isNumericType(right_type);
            },
        }
    }

    /// Check if unary operation is valid for given type
    pub fn isUnaryOperationValid(op: ast.UnaryOp, operand_type: []const u8) bool {
        switch (op) {
            .Not => return std.mem.eql(u8, operand_type, "bool"),
            .Negate => return isNumericType(operand_type),
            .BitwiseNot => return isNumericType(operand_type),
            .AddressOf => return true, // Can take address of most things
            .Dereference => return true, // Would need pointer type checking
        }
    }

    /// Generate a unique name for anonymous items
    pub fn generateUniqueName(allocator: std.mem.Allocator, prefix: []const u8, counter: *u32) ![]const u8 {
        const name = try std.fmt.allocPrint(allocator, "{s}_{d}", .{ prefix, counter.* });
        counter.* += 1;
        return name;
    }

    /// Sanitize a string for use as an identifier
    pub fn sanitizeIdentifier(allocator: std.mem.Allocator, input: []const u8) ![]const u8 {
        if (input.len == 0) {
            return try allocator.dupe(u8, "_");
        }

        var result = std.ArrayList(u8).init(allocator);
        defer result.deinit();

        // Ensure first character is valid
        if (std.ascii.isAlphabetic(input[0]) or input[0] == '_') {
            try result.append(input[0]);
        } else {
            try result.append('_');
        }

        // Process remaining characters
        for (input[1..]) |c| {
            if (std.ascii.isAlphanumeric(c) or c == '_') {
                try result.append(c);
            } else {
                try result.append('_');
            }
        }

        const sanitized = try result.toOwnedSlice();

        // Ensure it's not a reserved keyword
        if (isReservedKeyword(sanitized)) {
            const new_name = try std.fmt.allocPrint(allocator, "{s}_", .{sanitized});
            allocator.free(sanitized);
            return new_name;
        }

        return sanitized;
    }

    /// Calculate the complexity score of an expression
    pub fn calculateExpressionComplexity(expr: *ast.ExprNode) u32 {
        return switch (expr.*) {
            .Literal => 1,
            .Identifier => 1,
            .Binary => |*bin| {
                return 1 + calculateExpressionComplexity(bin.left) + calculateExpressionComplexity(bin.right);
            },
            .Unary => |*un| {
                return 1 + calculateExpressionComplexity(un.operand);
            },
            .Call => |*call| {
                var complexity: u32 = 2; // Base complexity for function call
                complexity += calculateExpressionComplexity(call.function);
                for (call.args) |*arg| {
                    complexity += calculateExpressionComplexity(arg);
                }
                return complexity;
            },
            .Index => |*index| {
                return 2 + calculateExpressionComplexity(index.object) + calculateExpressionComplexity(index.index);
            },
            .FieldAccess => |*field| {
                return 2 + calculateExpressionComplexity(field.object);
            },
            .Assignment => |*assign| {
                return 2 + calculateExpressionComplexity(assign.target) + calculateExpressionComplexity(assign.value);
            },
            .CompoundAssignment => |*comp| {
                return 3 + calculateExpressionComplexity(comp.target) + calculateExpressionComplexity(comp.value);
            },
            .Cast => |*cast| {
                return 2 + calculateExpressionComplexity(cast.expr);
            },
            .Tuple => |*tuple| {
                var complexity: u32 = 1;
                for (tuple.elements) |*elem| {
                    complexity += calculateExpressionComplexity(elem);
                }
                return complexity;
            },
            .Try => |*try_expr| {
                return 3 + calculateExpressionComplexity(try_expr.expr);
            },
            .ErrorReturn => |*error_ret| {
                return 2 + calculateExpressionComplexity(error_ret.expr);
            },
            .ErrorCast => |*error_cast| {
                return 3 + calculateExpressionComplexity(error_cast.expr);
            },
            else => 1,
        };
    }

    /// Check if function is a constructor (init function)
    pub fn isConstructorFunction(function: *ast.FunctionNode) bool {
        return std.mem.eql(u8, function.name, "init");
    }

    /// Check if function is public
    pub fn isFunctionPublic(function: *ast.FunctionNode) bool {
        return function.visibility == .Public;
    }

    /// Check if variable is mutable
    pub fn isVariableMutable(var_decl: *ast.VariableDeclNode) bool {
        return var_decl.mutable;
    }

    /// Check if variable is in storage region
    pub fn isStorageVariable(var_decl: *ast.VariableDeclNode) bool {
        return var_decl.region == .Storage;
    }

    /// Check if variable is immutable
    pub fn isImmutableVariable(var_decl: *ast.VariableDeclNode) bool {
        return var_decl.region == .Immutable or (var_decl.region == .Storage and !var_decl.mutable);
    }

    /// Get memory region name as string
    pub fn getMemoryRegionName(region: ast.MemoryRegion) []const u8 {
        return switch (region) {
            .Stack => "stack",
            .Heap => "heap",
            .Storage => "storage",
            .Immutable => "immutable",
        };
    }

    /// Parse integer literal value
    pub fn parseIntegerLiteral(literal: []const u8) !u256 {
        // Handle different number bases
        if (std.mem.startsWith(u8, literal, "0x") or std.mem.startsWith(u8, literal, "0X")) {
            // Hexadecimal
            return std.fmt.parseInt(u256, literal[2..], 16);
        } else if (std.mem.startsWith(u8, literal, "0b") or std.mem.startsWith(u8, literal, "0B")) {
            // Binary
            return std.fmt.parseInt(u256, literal[2..], 2);
        } else if (std.mem.startsWith(u8, literal, "0o") or std.mem.startsWith(u8, literal, "0O")) {
            // Octal
            return std.fmt.parseInt(u256, literal[2..], 8);
        } else {
            // Decimal
            return std.fmt.parseInt(u256, literal, 10);
        }
    }

    /// Validate address literal format
    pub fn isValidAddressLiteral(address: []const u8) bool {
        // Ethereum-style address: 0x followed by 40 hex characters
        if (address.len != 42) return false;
        if (!std.mem.startsWith(u8, address, "0x")) return false;

        for (address[2..]) |c| {
            if (!std.ascii.isHex(c)) return false;
        }

        return true;
    }

    /// Validate hex literal format
    pub fn isValidHexLiteral(hex: []const u8) bool {
        if (hex.len < 3) return false; // At least "0x" + one digit
        if (!std.mem.startsWith(u8, hex, "0x")) return false;

        for (hex[2..]) |c| {
            if (!std.ascii.isHex(c)) return false;
        }

        return true;
    }

    /// Create a safe default span for error reporting
    pub fn createDefaultSpan() ast.SourceSpan {
        return ast.SourceSpan{ .line = 0, .column = 0, .length = 0 };
    }

    /// Create a span from line and column information
    pub fn createSpan(line: u32, column: u32, length: u32) ast.SourceSpan {
        return ast.SourceSpan{ .line = line, .column = column, .length = length };
    }

    /// Merge two source spans to create a span covering both
    pub fn mergeSpans(span1: ast.SourceSpan, span2: ast.SourceSpan) ast.SourceSpan {
        const start_line = @min(span1.line, span2.line);
        const start_column = if (span1.line == span2.line) @min(span1.column, span2.column) else if (span1.line < span2.line) span1.column else span2.column;

        const end_line = @max(span1.line, span2.line);
        const end_column = if (span1.line == span2.line) @max(span1.column + span1.length, span2.column + span2.length) else if (span1.line > span2.line) span1.column + span1.length else span2.column + span2.length;

        const length = if (start_line == end_line) end_column - start_column else 0;

        return ast.SourceSpan{ .line = start_line, .column = start_column, .length = length };
    }
};

/// String utilities for semantic analysis
pub const StringUtils = struct {
    /// Escape string for safe output
    pub fn escapeString(allocator: std.mem.Allocator, input: []const u8) ![]const u8 {
        var result = std.ArrayList(u8).init(allocator);
        defer result.deinit();

        for (input) |c| {
            switch (c) {
                '\n' => try result.appendSlice("\\n"),
                '\r' => try result.appendSlice("\\r"),
                '\t' => try result.appendSlice("\\t"),
                '\\' => try result.appendSlice("\\\\"),
                '"' => try result.appendSlice("\\\""),
                '\'' => try result.appendSlice("\\'"),
                else => try result.append(c),
            }
        }

        return result.toOwnedSlice();
    }

    /// Unescape string literal
    pub fn unescapeString(allocator: std.mem.Allocator, input: []const u8) ![]const u8 {
        var result = std.ArrayList(u8).init(allocator);
        defer result.deinit();

        var i: usize = 0;
        while (i < input.len) {
            if (input[i] == '\\' and i + 1 < input.len) {
                switch (input[i + 1]) {
                    'n' => try result.append('\n'),
                    'r' => try result.append('\r'),
                    't' => try result.append('\t'),
                    '\\' => try result.append('\\'),
                    '"' => try result.append('"'),
                    '\'' => try result.append('\''),
                    else => {
                        try result.append(input[i]);
                        try result.append(input[i + 1]);
                    },
                }
                i += 2;
            } else {
                try result.append(input[i]);
                i += 1;
            }
        }

        return result.toOwnedSlice();
    }

    /// Check if string is a valid UTF-8 sequence
    pub fn isValidUtf8(input: []const u8) bool {
        return std.unicode.utf8ValidateSlice(input);
    }

    /// Calculate string hash for fast comparison
    pub fn hashString(input: []const u8) u64 {
        return std.hash_map.hashString(input);
    }
};

/// Collection utilities for semantic analysis
pub const CollectionUtils = struct {
    /// Check if array contains element
    pub fn arrayContains(comptime T: type, array: []const T, element: T) bool {
        for (array) |item| {
            if (std.meta.eql(item, element)) {
                return true;
            }
        }
        return false;
    }

    /// Find index of element in array
    pub fn arrayIndexOf(comptime T: type, array: []const T, element: T) ?usize {
        for (array, 0..) |item, i| {
            if (std.meta.eql(item, element)) {
                return i;
            }
        }
        return null;
    }

    /// Remove duplicates from array
    pub fn removeDuplicates(comptime T: type, allocator: std.mem.Allocator, array: []const T) ![]T {
        var result = std.ArrayList(T).init(allocator);
        defer result.deinit();

        for (array) |item| {
            var found = false;
            for (result.items) |existing| {
                if (std.meta.eql(item, existing)) {
                    found = true;
                    break;
                }
            }
            if (!found) {
                try result.append(item);
            }
        }

        return result.toOwnedSlice();
    }
};

/// Validation utilities
pub fn validateIdentifier(analyzer: *SemanticAnalyzer, name: []const u8, span: ast.SourceSpan) semantics_errors.SemanticError!bool {
    if (!semantics_memory_safety.isValidString(analyzer, name)) {
        try semantics_errors.addErrorStatic(analyzer, "Invalid identifier string", span);
        return false;
    }

    if (!SemanticUtils.isValidIdentifier(name)) {
        const error_msg = try std.fmt.allocPrint(analyzer.allocator, "Invalid identifier: {s}", .{name});
        try semantics_errors.addError(analyzer, error_msg, span);
        return false;
    }

    return true;
}

/// Validate type name
pub fn validateTypeName(analyzer: *SemanticAnalyzer, type_name: []const u8, span: ast.SourceSpan) semantics_errors.SemanticError!bool {
    if (!semantics_memory_safety.isValidString(analyzer, type_name)) {
        try semantics_errors.addErrorStatic(analyzer, "Invalid type name string", span);
        return false;
    }

    if (!SemanticUtils.isValidIdentifier(type_name) and !SemanticUtils.isPrimitiveType(type_name)) {
        const error_msg = try std.fmt.allocPrint(analyzer.allocator, "Invalid type name: {s}", .{type_name});
        try semantics_errors.addError(analyzer, error_msg, span);
        return false;
    }

    return true;
}

/// Validate literal value
pub fn validateLiteral(analyzer: *SemanticAnalyzer, literal: *ast.LiteralExpr) semantics_errors.SemanticError!bool {
    switch (literal.*) {
        .Integer => |*int| {
            if (!semantics_memory_safety.isValidString(analyzer, int.value)) {
                try semantics_errors.addErrorStatic(analyzer, "Invalid integer literal", int.span);
                return false;
            }

            // Try to parse the integer
            SemanticUtils.parseIntegerLiteral(int.value) catch {
                const error_msg = try std.fmt.allocPrint(analyzer.allocator, "Invalid integer literal: {s}", .{int.value});
                try semantics_errors.addError(analyzer, error_msg, int.span);
                return false;
            };
        },
        .String => |*str| {
            if (!semantics_memory_safety.isValidString(analyzer, str.value)) {
                try semantics_errors.addErrorStatic(analyzer, "Invalid string literal", str.span);
                return false;
            }

            if (!StringUtils.isValidUtf8(str.value)) {
                try semantics_errors.addErrorStatic(analyzer, "String literal contains invalid UTF-8", str.span);
                return false;
            }
        },
        .Address => |*addr| {
            if (!semantics_memory_safety.isValidString(analyzer, addr.value)) {
                try semantics_errors.addErrorStatic(analyzer, "Invalid address literal", addr.span);
                return false;
            }

            if (!SemanticUtils.isValidAddressLiteral(addr.value)) {
                try semantics_errors.addErrorStatic(analyzer, "Invalid address format", addr.span);
                return false;
            }
        },
        .Hex => |*hex| {
            if (!semantics_memory_safety.isValidString(analyzer, hex.value)) {
                try semantics_errors.addErrorStatic(analyzer, "Invalid hex literal", hex.span);
                return false;
            }

            if (!SemanticUtils.isValidHexLiteral(hex.value)) {
                try semantics_errors.addErrorStatic(analyzer, "Invalid hex format", hex.span);
                return false;
            }
        },
        .Bool => {
            // Boolean literals are always valid
        },
    }

    return true;
}
