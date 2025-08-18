const std = @import("std");
const ast = @import("ast.zig");
const typer = @import("typer.zig");
const ast_arena = @import("ast/ast_arena.zig");

/// Validation error kinds for detailed error categorization
pub const ValidationErrorKind = enum {
    type_mismatch,
    undefined_type,
    invalid_operation,
    circular_dependency,
    invalid_constraint,
    unsupported_feature,
    memory_region_violation,
    mutability_violation,
    access_violation,
    constraint_violation,
    inference_failure,
    generic_resolution_failure,
    structural_inconsistency,
};

/// Comprehensive validation error with detailed information
pub const ValidationError = struct {
    kind: ValidationErrorKind,
    message: []const u8,
    span: ast.SourceSpan,
    suggestions: [][]const u8,
    related_spans: []ast.SourceSpan,
    severity: Severity,

    // Type-specific error data
    expected_type: ?typer.OraType,
    actual_type: ?typer.OraType,
    operation: ?[]const u8,
    constraint: ?[]const u8,

    pub const Severity = enum {
        err,
        warning,
        info,
        hint,
    };

    pub fn init(
        kind: ValidationErrorKind,
        message: []const u8,
        span: ast.SourceSpan,
    ) ValidationError {
        return ValidationError{
            .kind = kind,
            .message = message,
            .span = span,
            .suggestions = &[_][]const u8{},
            .related_spans = &[_]ast.SourceSpan{},
            .severity = .err,
            .expected_type = null,
            .actual_type = null,
            .operation = null,
            .constraint = null,
        };
    }

    pub fn withSuggestions(self: ValidationError, suggestions: [][]const u8) ValidationError {
        var result = self;
        result.suggestions = suggestions;
        return result;
    }

    pub fn withSeverity(self: ValidationError, severity: Severity) ValidationError {
        var result = self;
        result.severity = severity;
        return result;
    }

    pub fn withTypes(self: ValidationError, expected: ?typer.OraType, actual: ?typer.OraType) ValidationError {
        var result = self;
        result.expected_type = expected;
        result.actual_type = actual;
        return result;
    }

    pub fn withOperation(self: ValidationError, operation: []const u8) ValidationError {
        var result = self;
        result.operation = operation;
        return result;
    }

    pub fn getSpan(self: ValidationError) ast.SourceSpan {
        return self.span;
    }
};

/// Comprehensive validation result with detailed error information
pub const ValidationResult = struct {
    allocator: std.mem.Allocator,
    errors: std.ArrayList(ValidationError),
    warnings: std.ArrayList(ValidationError),
    infos: std.ArrayList(ValidationError),
    hints: std.ArrayList(ValidationError),

    pub fn init(allocator: std.mem.Allocator) ValidationResult {
        return ValidationResult{
            .allocator = allocator,
            .errors = std.ArrayList(ValidationError).init(allocator),
            .warnings = std.ArrayList(ValidationError).init(allocator),
            .infos = std.ArrayList(ValidationError).init(allocator),
            .hints = std.ArrayList(ValidationError).init(allocator),
        };
    }

    pub fn deinit(self: *ValidationResult) void {
        self.errors.deinit();
        self.warnings.deinit();
        self.infos.deinit();
        self.hints.deinit();
    }

    pub fn addError(self: *ValidationResult, err: ValidationError) !void {
        switch (err.severity) {
            .err => try self.errors.append(err),
            .warning => try self.warnings.append(err),
            .info => try self.infos.append(err),
            .hint => try self.hints.append(err),
        }
    }

    pub fn addWarning(self: *ValidationResult, warning: ValidationError) !void {
        try self.warnings.append(warning);
    }

    pub fn isValid(self: ValidationResult) bool {
        return self.errors.items.len == 0;
    }

    pub fn hasErrors(self: ValidationResult) bool {
        return self.errors.items.len > 0;
    }

    pub fn hasWarnings(self: ValidationResult) bool {
        return self.warnings.items.len > 0;
    }

    pub fn getErrors(self: ValidationResult) []const ValidationError {
        return self.errors.items;
    }

    pub fn getWarnings(self: ValidationResult) []const ValidationError {
        return self.warnings.items;
    }

    pub fn getInfos(self: ValidationResult) []const ValidationError {
        return self.infos.items;
    }

    pub fn getHints(self: ValidationResult) []const ValidationError {
        return self.hints.items;
    }

    pub fn getTotalCount(self: ValidationResult) usize {
        return self.errors.items.len + self.warnings.items.len + self.infos.items.len + self.hints.items.len;
    }
};

/// Comprehensive type validator enforcing Ora's explicit typing rules
pub const TypeValidator = struct {
    allocator: std.mem.Allocator,
    arena: *ast_arena.AstArena,
    typer_instance: *typer.Typer,
    validation_stack: std.ArrayList([]const u8), // For circular dependency detection
    strict_mode: bool, // Enforce strict Ora typing rules

    pub fn init(allocator: std.mem.Allocator, arena: *ast_arena.AstArena, typer_instance: *typer.Typer) TypeValidator {
        return TypeValidator{
            .allocator = allocator,
            .arena = arena,
            .typer_instance = typer_instance,
            .validation_stack = std.ArrayList([]const u8).init(allocator),
            .strict_mode = true, // Ora enforces strict typing by default
        };
    }

    pub fn deinit(self: *TypeValidator) void {
        self.validation_stack.deinit();
    }

    /// Set strict mode for Ora's explicit typing enforcement
    pub fn setStrictMode(self: *TypeValidator, strict: bool) void {
        self.strict_mode = strict;
    }

    /// Validate an AST node recursively
    pub fn validateNode(self: *TypeValidator, node: *ast.AstNode) !ValidationResult {
        var result = ValidationResult.init(self.allocator);

        switch (node.*) {
            .Contract => |*contract| {
                try self.validateContract(contract, &result);
            },
            .Function => |*function| {
                try self.validateFunction(function, &result);
            },
            .VariableDecl => |*var_decl| {
                try self.validateVariableDecl(var_decl, &result);
            },
            .StructDecl => |*struct_decl| {
                try self.validateStructDecl(struct_decl, &result);
            },
            .EnumDecl => |*enum_decl| {
                try self.validateEnumDecl(enum_decl, &result);
            },
            .Block => |*block| {
                try self.validateBlock(block, &result);
            },
            .Expression => |*expr| {
                _ = try self.validateExpression(expr, &result);
            },
            .Statement => |*stmt| {
                try self.validateStatement(stmt, &result);
            },
            else => {
                // Add validation for other node types as needed
            },
        }

        return result;
    }

    /// Validate a type reference
    pub fn validateType(self: *TypeValidator, type_ref: *const ast.TypeRef, result: *ValidationResult) !typer.OraType {
        return switch (type_ref.*) {
            // Primitive types are always valid
            .Bool => typer.OraType.Bool,
            .Address => typer.OraType.Address,
            .U8 => typer.OraType.U8,
            .U16 => typer.OraType.U16,
            .U32 => typer.OraType.U32,
            .U64 => typer.OraType.U64,
            .U128 => typer.OraType.U128,
            .U256 => typer.OraType.U256,
            .I8 => typer.OraType.I8,
            .I16 => typer.OraType.I16,
            .I32 => typer.OraType.I32,
            .I64 => typer.OraType.I64,
            .I128 => typer.OraType.I128,
            .I256 => typer.OraType.I256,
            .String => typer.OraType.String,
            .Bytes => typer.OraType.Bytes,

            .Slice => |element_type| {
                const validated_element = try self.validateType(element_type, result);
                const element_ptr = try self.allocator.create(typer.OraType);
                element_ptr.* = validated_element;
                return typer.OraType{ .Slice = element_ptr };
            },

            .Mapping => |mapping| {
                const validated_key = try self.validateType(mapping.key, result);
                const validated_value = try self.validateType(mapping.value, result);

                // Validate that key type is valid for mapping
                if (!self.isValidMappingKeyType(validated_key)) {
                    const error_msg = try std.fmt.allocPrint(self.allocator, "Invalid key type '{}' for mapping", .{validated_key});
                    const error_obj = ValidationError.init(
                        .invalid_operation,
                        error_msg,
                        .{ .line = 0, .column = 0, .length = 0 },
                    ).withOperation("mapping key").withTypes(null, validated_key);
                    try result.addError(error_obj);
                }

                const key_ptr = try self.allocator.create(typer.OraType);
                const value_ptr = try self.allocator.create(typer.OraType);
                key_ptr.* = validated_key;
                value_ptr.* = validated_value;

                return typer.OraType{ .Mapping = .{
                    .key = key_ptr,
                    .value = value_ptr,
                } };
            },

            .DoubleMap => |double_map| {
                const validated_key1 = try self.validateType(double_map.key1, result);
                const validated_key2 = try self.validateType(double_map.key2, result);
                const validated_value = try self.validateType(double_map.value, result);

                // Validate that key types are valid for mapping
                if (!self.isValidMappingKeyType(validated_key1)) {
                    const error_msg = try std.fmt.allocPrint(self.allocator, "Invalid key type '{}' for double map key1", .{validated_key1});
                    const error_obj = ValidationError.init(
                        .invalid_operation,
                        error_msg,
                        .{ .line = 0, .column = 0, .length = 0 },
                    ).withOperation("double map key1").withTypes(null, validated_key1);
                    try result.addError(error_obj);
                }

                if (!self.isValidMappingKeyType(validated_key2)) {
                    const error_msg = try std.fmt.allocPrint(self.allocator, "Invalid key type '{}' for double map key2", .{validated_key2});
                    const error_obj = ValidationError.init(
                        .invalid_operation,
                        error_msg,
                        .{ .line = 0, .column = 0, .length = 0 },
                    ).withOperation("double map key2").withTypes(null, validated_key2);
                    try result.addError(error_obj);
                }

                const key1_ptr = try self.allocator.create(typer.OraType);
                const key2_ptr = try self.allocator.create(typer.OraType);
                const value_ptr = try self.allocator.create(typer.OraType);
                key1_ptr.* = validated_key1;
                key2_ptr.* = validated_key2;
                value_ptr.* = validated_value;

                return typer.OraType{ .DoubleMap = .{
                    .key1 = key1_ptr,
                    .key2 = key2_ptr,
                    .value = value_ptr,
                } };
            },

            .Identifier => |name| {
                // Look up custom type in symbol table
                if (self.typer_instance.current_scope.lookup(name)) |symbol| {
                    return symbol.typ;
                }

                // Check if it's a struct type
                if (self.typer_instance.getStructType(name)) |struct_type| {
                    const struct_ptr = try self.allocator.create(typer.StructType);
                    struct_ptr.* = struct_type.*;
                    return typer.OraType{ .Struct = struct_ptr };
                }

                const error_msg = try std.fmt.allocPrint(self.allocator, "Undefined type '{s}'", .{name});
                const error_obj = ValidationError.init(
                    .undefined_type,
                    error_msg,
                    .{ .line = 0, .column = 0, .length = 0 },
                );
                try result.addError(error_obj);
                return typer.OraType.Unknown;
            },

            .Tuple => |tuple| {
                const element_types = try self.allocator.alloc(typer.OraType, tuple.types.len);
                for (tuple.types, 0..) |*element_type, i| {
                    element_types[i] = try self.validateType(element_type, result);
                }
                return typer.OraType{ .Tuple = .{ .types = element_types } };
            },

            .ErrorUnion => |error_union| {
                // For now, convert error union to Unknown type
                // TODO: Implement proper error union handling
                _ = error_union;
                return typer.OraType.Unknown;
            },

            .Result => |result_type| {
                // For now, convert Result to Unknown type
                // TODO: Implement proper Result type handling
                _ = result_type;
                return typer.OraType.Unknown;
            },

            .Unknown => typer.OraType.Unknown,
            .Inferred => |inferred| {
                // For inferred types, validate the base type
                return try self.validateType(&inferred.base_type, result);
            },
        };
    }

    /// Validate an expression and return its type
    pub fn validateExpression(self: *TypeValidator, expr: *ast.ExprNode, result: *ValidationResult) !typer.OraType {
        return switch (expr.*) {
            .Identifier => |*ident| {
                if (self.typer_instance.current_scope.lookup(ident.name)) |symbol| {
                    return symbol.typ;
                } else {
                    const error_msg = try std.fmt.allocPrint(self.allocator, "Undefined identifier '{s}'", .{ident.name});
                    const error_obj = ValidationError.init(
                        .undefined_type,
                        error_msg,
                        ident.span,
                    );
                    try result.addError(error_obj);
                    return typer.OraType.Unknown;
                }
            },

            .Literal => |*literal| {
                return try self.validateLiteral(literal, result);
            },

            .Binary => |*binary| {
                const lhs_type = try self.validateExpression(binary.lhs, result);
                const rhs_type = try self.validateExpression(binary.rhs, result);
                return try self.validateBinaryOperation(binary.operator, lhs_type, rhs_type, binary.span, result);
            },

            .Unary => |*unary| {
                const operand_type = try self.validateExpression(unary.operand, result);
                return try self.validateUnaryOperation(unary.operator, operand_type, unary.span, result);
            },

            .Call => |*call| {
                return try self.validateFunctionCall(call, result);
            },

            .Index => |*index| {
                const target_type = try self.validateExpression(index.target, result);
                const index_type = try self.validateExpression(index.index, result);
                return try self.validateIndexOperation(target_type, index_type, index.span, result);
            },

            .FieldAccess => |*field| {
                const target_type = try self.validateExpression(field.target, result);
                return try self.validateFieldAccess(target_type, field.field, field.span, result);
            },

            .Cast => |*cast| {
                const operand_type = try self.validateExpression(cast.operand, result);
                const target_type = try self.validateType(&cast.target_type, result);
                return try self.validateCast(operand_type, target_type, cast.cast_type, cast.span, result);
            },

            .Tuple => |*tuple| {
                const element_types = try self.allocator.alloc(typer.OraType, tuple.elements.len);
                for (tuple.elements, 0..) |*element, i| {
                    element_types[i] = try self.validateExpression(element, result);
                }
                return typer.OraType{ .Tuple = .{ .types = element_types } };
            },

            .Try => |*try_expr| {
                const expr_type = try self.validateExpression(try_expr.expr, result);
                return try self.validateTryExpression(expr_type, try_expr.span, result);
            },

            .If => |*if_expr| {
                const condition_type = try self.validateExpression(if_expr.condition, result);
                if (condition_type != .Bool) {
                    try result.addError(ValidationError{ .TypeMismatch = .{
                        .expected = typer.OraType.Bool,
                        .actual = condition_type,
                        .span = if_expr.span,
                    } });
                }

                const then_type = try self.validateExpression(if_expr.then_expr, result);
                const else_type = if (if_expr.else_expr) |else_expr|
                    try self.validateExpression(else_expr, result)
                else
                    typer.OraType.Unknown;

                // Both branches should have compatible types
                if (then_type != else_type and else_type != .Unknown) {
                    try result.addError(ValidationError{ .TypeMismatch = .{
                        .expected = then_type,
                        .actual = else_type,
                        .span = if_expr.span,
                    } });
                }

                return then_type;
            },

            .Block => |*block| {
                // Validate all statements in the block
                for (block.statements) |*stmt| {
                    _ = try self.validateStatement(stmt, result);
                }

                // Return type of last expression if it's an expression statement
                if (block.statements.len > 0) {
                    const last_stmt = block.statements[block.statements.len - 1];
                    if (last_stmt.* == .Expression) {
                        return try self.validateExpression(&last_stmt.Expression, result);
                    }
                }

                return typer.OraType.Unknown;
            },
        };
    }

    // Private validation methods

    /// Validate literal types - Ora requires explicit type annotations for numeric literals
    pub fn validateLiteral(self: *TypeValidator, literal: *ast.LiteralNode, result: *ValidationResult) !typer.OraType {
        return switch (literal.*) {
            .Integer => |*int_lit| {
                if (self.strict_mode) {
                    // In Ora, integer literals require explicit type annotations
                    // Example: let a: u256 = 100; (correct) vs let a = 100; (wrong)
                    const error_msg = try std.fmt.allocPrint(self.allocator, "Integer literal requires explicit type annotation. Use 'let var_name: u256 = {s}' instead of 'let var_name = {s}'", .{ int_lit.value, int_lit.value });

                    const suggestions = try self.allocator.alloc([]const u8, 6);
                    suggestions[0] = try std.fmt.allocPrint(self.allocator, "let var_name: u8 = {s}", .{int_lit.value});
                    suggestions[1] = try std.fmt.allocPrint(self.allocator, "let var_name: u32 = {s}", .{int_lit.value});
                    suggestions[2] = try std.fmt.allocPrint(self.allocator, "let var_name: u64 = {s}", .{int_lit.value});
                    suggestions[3] = try std.fmt.allocPrint(self.allocator, "let var_name: u128 = {s}", .{int_lit.value});
                    suggestions[4] = try std.fmt.allocPrint(self.allocator, "let var_name: u256 = {s}", .{int_lit.value});
                    suggestions[5] = try std.fmt.allocPrint(self.allocator, "let var_name: i256 = {s}", .{int_lit.value});

                    const error_obj = ValidationError.init(
                        .invalid_operation,
                        error_msg,
                        int_lit.span,
                    ).withSuggestions(suggestions).withOperation("integer literal without explicit type");

                    try result.addError(error_obj);
                }
                return typer.OraType.Unknown;
            },
            .String => typer.OraType.String,
            .Bool => typer.OraType.Bool,
            .Address => typer.OraType.Address,
            .Hex => typer.OraType.U256, // Hex literals default to u256
        };
    }

    pub fn validateBinaryOperation(self: *TypeValidator, operator: ast.BinaryOp, lhs: typer.OraType, rhs: typer.OraType, span: ast.SourceSpan, result: *ValidationResult) !typer.OraType {
        return switch (operator) {
            .Plus, .Minus, .Star, .Slash, .Percent, .StarStar => {
                if (!self.isNumericType(lhs) or !self.isNumericType(rhs)) {
                    const error_msg = try std.fmt.allocPrint(self.allocator, "Arithmetic operation '{}' requires numeric operands, got '{}' and '{}'", .{ operator, lhs, rhs });

                    const error_obj = ValidationError.init(
                        .invalid_operation,
                        error_msg,
                        span,
                    ).withTypes(lhs, rhs).withOperation("arithmetic operation");

                    try result.addError(error_obj);
                    return typer.OraType.Unknown;
                }

                // Ora requires exact type match - no implicit conversions
                if (!self.areTypesExactMatch(lhs, rhs)) {
                    const error_msg = try std.fmt.allocPrint(self.allocator, "Arithmetic operation requires operands of the same type. Got '{}' and '{}'. Use explicit casting if conversion is intended.", .{ lhs, rhs });

                    const suggestions = try self.allocator.alloc([]const u8, 2);
                    suggestions[0] = try std.fmt.allocPrint(self.allocator, "Cast left operand: (operand as {}) {} operand", .{ rhs, operator });
                    suggestions[1] = try std.fmt.allocPrint(self.allocator, "Cast right operand: operand {} (operand as {})", .{ operator, lhs });

                    const error_obj = ValidationError.init(
                        .type_mismatch,
                        error_msg,
                        span,
                    ).withTypes(lhs, rhs).withSuggestions(suggestions);

                    try result.addError(error_obj);
                    return typer.OraType.Unknown;
                }

                // Check for division by zero (would need runtime check)
                if (operator == .Slash or operator == .Percent) {
                    const warning_msg = "Division operations should include runtime checks for zero divisor";
                    const warning_obj = ValidationError.init(
                        .invalid_operation,
                        warning_msg,
                        span,
                    ).withSeverity(.warning).withOperation("division operation");

                    try result.addError(warning_obj);
                }

                return lhs; // Return the common type
            },
            .BitwiseAnd, .BitwiseOr, .BitwiseXor, .LeftShift, .RightShift => {
                if (!self.isIntegerType(lhs) or !self.isIntegerType(rhs)) {
                    const error_msg = try std.fmt.allocPrint(self.allocator, "Bitwise operation '{}' requires integer operands, got '{}' and '{}'", .{ operator, lhs, rhs });

                    const error_obj = ValidationError.init(
                        .invalid_operation,
                        error_msg,
                        span,
                    ).withTypes(lhs, rhs).withOperation("bitwise operation");

                    try result.addError(error_obj);
                    return typer.OraType.Unknown;
                }

                // Ora requires exact type match - no implicit conversions
                if (!self.areTypesExactMatch(lhs, rhs)) {
                    const error_msg = try std.fmt.allocPrint(self.allocator, "Bitwise operation requires operands of the same type. Got '{}' and '{}'. Use explicit casting if conversion is intended.", .{ lhs, rhs });

                    const error_obj = ValidationError.init(
                        .type_mismatch,
                        error_msg,
                        span,
                    ).withTypes(lhs, rhs);

                    try result.addError(error_obj);
                    return typer.OraType.Unknown;
                }

                return lhs; // Return the common type
            },
            .EqualEqual, .BangEqual, .Less, .LessEqual, .Greater, .GreaterEqual => {
                // Comparison operators require exact type match in Ora
                if (!self.areTypesExactMatch(lhs, rhs)) {
                    const error_msg = try std.fmt.allocPrint(self.allocator, "Comparison operation requires operands of the same type. Got '{}' and '{}'. Use explicit casting if conversion is intended.", .{ lhs, rhs });

                    const error_obj = ValidationError.init(
                        .type_mismatch,
                        error_msg,
                        span,
                    ).withTypes(lhs, rhs);

                    try result.addError(error_obj);
                }
                return typer.OraType.Bool;
            },
            .And, .Or => {
                if (!self.areTypesExactMatch(lhs, typer.OraType.Bool) or !self.areTypesExactMatch(rhs, typer.OraType.Bool)) {
                    const error_msg = try std.fmt.allocPrint(self.allocator, "Logical operation '{}' requires boolean operands, got '{}' and '{}'", .{ operator, lhs, rhs });

                    const error_obj = ValidationError.init(
                        .type_mismatch,
                        error_msg,
                        span,
                    ).withTypes(typer.OraType.Bool, lhs);

                    try result.addError(error_obj);
                    return typer.OraType.Unknown;
                }
                return typer.OraType.Bool;
            },
            .Comma => {
                // Comma operator returns the type of the right operand
                return rhs;
            },
        };
    }

    fn validateUnaryOperation(self: *TypeValidator, operator: ast.UnaryOperator, operand: typer.OraType, span: ast.SourceSpan, result: *ValidationResult) !typer.OraType {
        return switch (operator) {
            .Minus => {
                if (!self.isNumericType(operand)) {
                    try result.addError(ValidationError{ .InvalidOperation = .{
                        .operation = "negation",
                        .type_ = operand,
                        .span = span,
                    } });
                    return typer.OraType.Unknown;
                }
                return operand;
            },
            .Bang => {
                if (operand != .Bool) {
                    try result.addError(ValidationError{ .TypeMismatch = .{
                        .expected = typer.OraType.Bool,
                        .actual = operand,
                        .span = span,
                    } });
                    return typer.OraType.Unknown;
                }
                return typer.OraType.Bool;
            },
            .BitNot => {
                if (!self.isIntegerType(operand)) {
                    try result.addError(ValidationError{ .InvalidOperation = .{
                        .operation = "bitwise not",
                        .type_ = operand,
                        .span = span,
                    } });
                    return typer.OraType.Unknown;
                }
                return operand;
            },
        };
    }

    fn validateFunctionCall(self: *TypeValidator, call: *ast.CallExpr, result: *ValidationResult) !typer.OraType {
        _ = try self.validateExpression(call.callee, result);

        // Validate all arguments
        for (call.arguments) |*arg| {
            _ = try self.validateExpression(arg, result);
        }

        // Handle built-in functions
        if (call.callee.* == .Identifier) {
            const func_name = call.callee.Identifier.name;

            if (std.mem.eql(u8, func_name, "require") or std.mem.eql(u8, func_name, "assert")) {
                if (call.arguments.len != 1) {
                    try result.addError(ValidationError{ .InvalidOperation = .{
                        .operation = "require/assert",
                        .type_ = typer.OraType.Unknown,
                        .span = call.span,
                    } });
                }
                return typer.OraType.Bool;
            }

            if (std.mem.eql(u8, func_name, "hash")) {
                return typer.OraType.U256;
            }

            if (std.mem.eql(u8, func_name, "len")) {
                return typer.OraType.U256;
            }
        }

        // TODO: Implement function type lookup from symbol table
        return typer.OraType.Unknown;
    }

    fn validateIndexOperation(self: *TypeValidator, target_type: typer.OraType, index_type: typer.OraType, span: ast.SourceSpan, result: *ValidationResult) !typer.OraType {
        return switch (target_type) {
            .Slice => |element_type| {
                if (!self.isIntegerType(index_type)) {
                    try result.addError(ValidationError{ .TypeMismatch = .{
                        .expected = typer.OraType.U256,
                        .actual = index_type,
                        .span = span,
                    } });
                }
                return element_type.*;
            },
            .Mapping => |mapping| {
                if (!self.areTypesCompatible(mapping.key.*, index_type)) {
                    try result.addError(ValidationError{ .TypeMismatch = .{
                        .expected = mapping.key.*,
                        .actual = index_type,
                        .span = span,
                    } });
                }
                return mapping.value.*;
            },
            .DoubleMap => |double_map| {
                // For double maps, we need to check both key types
                // This is a simplified implementation
                return double_map.value.*;
            },
            else => {
                try result.addError(ValidationError{ .InvalidOperation = .{
                    .operation = "index",
                    .type_ = target_type,
                    .span = span,
                } });
                return typer.OraType.Unknown;
            },
        };
    }

    fn validateFieldAccess(self: *TypeValidator, target_type: typer.OraType, field_name: []const u8, span: ast.SourceSpan, result: *ValidationResult) !typer.OraType {
        _ = self;
        return switch (target_type) {
            .Struct => |struct_type| {
                for (struct_type.fields) |struct_field| {
                    if (std.mem.eql(u8, struct_field.name, field_name)) {
                        return struct_field.typ;
                    }
                }
                try result.addError(ValidationError{ .UndefinedType = .{
                    .name = field_name,
                    .span = span,
                } });
                return typer.OraType.Unknown;
            },
            else => {
                try result.addError(ValidationError{ .InvalidOperation = .{
                    .operation = "field access",
                    .type_ = target_type,
                    .span = span,
                } });
                return typer.OraType.Unknown;
            },
        };
    }

    fn validateCast(self: *TypeValidator, from_type: typer.OraType, to_type: typer.OraType, cast_type: ast.CastType, span: ast.SourceSpan, result: *ValidationResult) !typer.OraType {
        return switch (cast_type) {
            .Unsafe => {
                // Unsafe casts are always allowed but may generate warnings
                if (!self.isValidUnsafeCast(from_type, to_type)) {
                    try result.addWarning(ValidationError{ .InvalidOperation = .{
                        .operation = "unsafe cast",
                        .type_ = from_type,
                        .span = span,
                    } });
                }
                return to_type;
            },
            .Safe => {
                if (!self.isValidSafeCast(from_type, to_type)) {
                    try result.addError(ValidationError{ .InvalidOperation = .{
                        .operation = "safe cast",
                        .type_ = from_type,
                        .span = span,
                    } });
                    return typer.OraType.Unknown;
                }
                return to_type;
            },
            .Forced => {
                // Forced casts are always allowed
                return to_type;
            },
        };
    }

    fn validateTryExpression(self: *TypeValidator, expr_type: typer.OraType, span: ast.SourceSpan, result: *ValidationResult) !typer.OraType {
        _ = self;
        _ = span;
        _ = result;
        return switch (expr_type) {
            .ErrorUnion => |error_union| {
                return error_union.success_type.*;
            },
            else => {
                // Try on non-error type is allowed but has no effect
                return expr_type;
            },
        };
    }

    // Contract and function validation methods

    fn validateContract(self: *TypeValidator, contract: *ast.ContractNode, result: *ValidationResult) !void {
        _ = result;
        // Validate all contract members
        for (contract.body) |*member| {
            _ = try self.validateNode(member);
        }
    }

    fn validateFunction(self: *TypeValidator, function: *ast.FunctionNode, result: *ValidationResult) !void {
        // Validate parameters
        for (function.parameters) |*param| {
            _ = try self.validateType(&param.typ, result);
        }

        // Validate return type
        _ = try self.validateType(&function.return_type, result);

        // Validate requires clauses
        for (function.requires_clauses) |*clause| {
            const clause_type = try self.validateExpression(clause, result);
            if (clause_type != .Bool) {
                try result.addError(ValidationError{ .TypeMismatch = .{
                    .expected = typer.OraType.Bool,
                    .actual = clause_type,
                    .span = function.span,
                } });
            }
        }

        // Validate ensures clauses
        for (function.ensures_clauses) |*clause| {
            const clause_type = try self.validateExpression(clause, result);
            if (clause_type != .Bool) {
                try result.addError(ValidationError{ .TypeMismatch = .{
                    .expected = typer.OraType.Bool,
                    .actual = clause_type,
                    .span = function.span,
                } });
            }
        }

        // Validate function body
        try self.validateBlock(&function.body, result);
    }

    pub fn validateVariableDecl(self: *TypeValidator, var_decl: *ast.VariableDeclNode, result: *ValidationResult) !void {
        // Validate that type annotation is present and explicit
        if (self.strict_mode) {
            try self.validateExplicitTypeAnnotation(&var_decl.typ, var_decl.span, result);
        }

        // Validate type annotation
        const declared_type = try self.validateType(&var_decl.typ, result);

        // Validate initial value if present
        if (var_decl.value) |*value| {
            const value_type = try self.validateExpression(value, result);

            // In Ora, no implicit conversions are allowed
            if (!self.areTypesExactMatch(declared_type, value_type)) {
                const error_msg = try std.fmt.allocPrint(self.allocator, "Type mismatch: cannot assign value of type '{}' to variable of type '{}'. Ora requires explicit type conversions.", .{ value_type, declared_type });

                const error_obj = ValidationError.init(
                    .type_mismatch,
                    error_msg,
                    var_decl.span,
                ).withTypes(declared_type, value_type);

                try result.addError(error_obj);
            }
        }
    }

    /// Validate that a type annotation is explicit and not inferred
    fn validateExplicitTypeAnnotation(self: *TypeValidator, type_ref: *const ast.TypeRef, span: ast.SourceSpan, result: *ValidationResult) !void {
        switch (type_ref.*) {
            .Unknown => {
                const error_msg = "Variable declaration must have explicit type annotation. Ora does not support type inference.";
                const suggestions = try self.allocator.alloc([]const u8, 3);
                suggestions[0] = "Add explicit type: let var_name: u256 = value;";
                suggestions[1] = "Add explicit type: let var_name: bool = value;";
                suggestions[2] = "Add explicit type: let var_name: address = value;";

                const error_obj = ValidationError.init(
                    .invalid_operation,
                    error_msg,
                    span,
                ).withSuggestions(suggestions).withOperation("missing explicit type annotation");

                try result.addError(error_obj);
            },
            .Inferred => {
                const error_msg = "Inferred types are not allowed in Ora. All variables must have explicit type annotations.";
                const error_obj = ValidationError.init(
                    .unsupported_feature,
                    error_msg,
                    span,
                ).withOperation("type inference");

                try result.addError(error_obj);
            },
            else => {
                // Explicit type annotation is present, which is correct for Ora
            },
        }
    }

    /// Check if two types are exactly the same (no implicit conversions in Ora)
    pub fn areTypesExactMatch(self: *TypeValidator, type1: typer.OraType, type2: typer.OraType) bool {
        _ = self;
        return std.meta.eql(type1, type2);
    }

    fn validateStructDecl(self: *TypeValidator, struct_decl: *ast.StructDeclNode, result: *ValidationResult) !void {
        // Validate all struct fields
        for (struct_decl.fields) |*field| {
            _ = try self.validateType(&field.typ, result);
        }
    }

    fn validateEnumDecl(self: *TypeValidator, enum_decl: *ast.EnumDeclNode, result: *ValidationResult) !void {
        // Validate base type if specified
        if (enum_decl.base_type) |*base_type| {
            const validated_base = try self.validateType(base_type, result);
            if (!self.isValidEnumBaseType(validated_base)) {
                try result.addError(ValidationError{ .InvalidOperation = .{
                    .operation = "enum base type",
                    .type_ = validated_base,
                    .span = enum_decl.span,
                } });
            }
        }

        // Validate all enum variants
        for (enum_decl.variants) |*variant| {
            if (variant.value) |*value| {
                _ = try self.validateExpression(value, result);
            }
        }
    }

    fn validateBlock(self: *TypeValidator, block: *ast.BlockNode, result: *ValidationResult) !void {
        // Validate all statements in the block
        for (block.statements) |*stmt| {
            _ = try self.validateStatement(stmt, result);
        }
    }

    fn validateStatement(self: *TypeValidator, stmt: *ast.StmtNode, result: *ValidationResult) !typer.OraType {
        return switch (stmt.*) {
            .Expression => |*expr| {
                return try self.validateExpression(expr, result);
            },
            .VariableDecl => |*var_decl| {
                try self.validateVariableDecl(var_decl, result);
                return typer.OraType.Unknown;
            },
            .Assignment => |*assign| {
                const target_type = try self.validateExpression(assign.target, result);
                const value_type = try self.validateExpression(assign.value, result);

                if (!self.areTypesCompatible(target_type, value_type)) {
                    try result.addError(ValidationError{ .TypeMismatch = .{
                        .expected = target_type,
                        .actual = value_type,
                        .span = assign.span,
                    } });
                }

                return target_type;
            },
            .If => |*if_stmt| {
                const condition_type = try self.validateExpression(if_stmt.condition, result);
                if (condition_type != .Bool) {
                    try result.addError(ValidationError{ .TypeMismatch = .{
                        .expected = typer.OraType.Bool,
                        .actual = condition_type,
                        .span = if_stmt.span,
                    } });
                }

                try self.validateBlock(if_stmt.then_block, result);
                if (if_stmt.else_block) |*else_block| {
                    try self.validateBlock(else_block, result);
                }

                return typer.OraType.Unknown;
            },
            .While => |*while_stmt| {
                const condition_type = try self.validateExpression(while_stmt.condition, result);
                if (condition_type != .Bool) {
                    try result.addError(ValidationError{ .TypeMismatch = .{
                        .expected = typer.OraType.Bool,
                        .actual = condition_type,
                        .span = while_stmt.span,
                    } });
                }

                try self.validateBlock(while_stmt.body, result);
                return typer.OraType.Unknown;
            },
            .Return => |*return_stmt| {
                if (return_stmt.value) |*value| {
                    return try self.validateExpression(value, result);
                }
                return typer.OraType.Unknown;
            },
            .Break, .Continue => {
                return typer.OraType.Unknown;
            },
        };
    }

    // Type checking utility methods

    fn areTypesCompatible(self: *TypeValidator, lhs: typer.OraType, rhs: typer.OraType) bool {
        return self.typer_instance.typesCompatible(lhs, rhs);
    }

    pub fn isNumericType(self: *TypeValidator, type_: typer.OraType) bool {
        _ = self;
        return switch (type_) {
            .U8, .U16, .U32, .U64, .U128, .U256, .I8, .I16, .I32, .I64, .I128, .I256 => true,
            else => false,
        };
    }

    pub fn isIntegerType(self: *TypeValidator, type_: typer.OraType) bool {
        return self.isNumericType(type_);
    }

    fn isValidMappingKeyType(self: *TypeValidator, type_: typer.OraType) bool {
        _ = self;
        return switch (type_) {
            .U8, .U16, .U32, .U64, .U128, .U256, .I8, .I16, .I32, .I64, .I128, .I256, .Bool, .Address, .String => true,
            else => false,
        };
    }

    fn isValidEnumBaseType(self: *TypeValidator, type_: typer.OraType) bool {
        _ = self;
        return switch (type_) {
            .U8, .U16, .U32, .U64, .U128, .U256, .I8, .I16, .I32, .I64, .I128, .I256 => true,
            else => false,
        };
    }

    fn isValidUnsafeCast(self: *TypeValidator, from: typer.OraType, to: typer.OraType) bool {
        _ = self;
        // Most casts are allowed as unsafe
        return switch (from) {
            .Unknown => false,
            else => switch (to) {
                .Unknown => false,
                else => true,
            },
        };
    }

    fn isValidSafeCast(self: *TypeValidator, from: typer.OraType, to: typer.OraType) bool {
        _ = self;
        // Safe casts are more restrictive
        return switch (from) {
            .U8 => switch (to) {
                .U16, .U32, .U64, .U128, .U256 => true,
                else => false,
            },
            .U16 => switch (to) {
                .U32, .U64, .U128, .U256 => true,
                else => false,
            },
            .U32 => switch (to) {
                .U64, .U128, .U256 => true,
                else => false,
            },
            .U64 => switch (to) {
                .U128, .U256 => true,
                else => false,
            },
            .U128 => switch (to) {
                .U256 => true,
                else => false,
            },
            else => from == to,
        };
    }

    fn promoteNumericTypes(self: *TypeValidator, lhs: typer.OraType, rhs: typer.OraType) !typer.OraType {
        _ = self;
        const size1 = typer.getTypeSize(lhs);
        const size2 = typer.getTypeSize(rhs);

        if (size1 >= size2) {
            return lhs;
        } else {
            return rhs;
        }
    }
};
