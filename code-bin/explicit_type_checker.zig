const std = @import("std");
const ast = @import("ast.zig");
const typer = @import("typer.zig");
const type_validator = @import("type_validator.zig");
const ast_arena = @import("ast/ast_arena.zig");

/// Explicit type checker enforcing Ora's strict typing rules
/// Ora philosophy: All types must be explicitly declared, no implicit conversions
pub const ExplicitTypeChecker = struct {
    allocator: std.mem.Allocator,
    arena: *ast_arena.AstArena,
    typer_instance: *typer.Typer,
    validator: *type_validator.TypeValidator,

    pub fn init(
        allocator: std.mem.Allocator,
        arena: *ast_arena.AstArena,
        typer_instance: *typer.Typer,
        validator: *type_validator.TypeValidator,
    ) ExplicitTypeChecker {
        return ExplicitTypeChecker{
            .allocator = allocator,
            .arena = arena,
            .typer_instance = typer_instance,
            .validator = validator,
        };
    }

    /// Check that a literal requires explicit type annotation
    /// In Ora: let a = 100; (WRONG) vs let a: u256 = 100; (CORRECT)
    pub fn checkLiteralRequiresExplicitType(
        self: *ExplicitTypeChecker,
        literal: *ast.LiteralNode,
        context_type: ?typer.OraType,
        result: *type_validator.ValidationResult,
    ) !void {
        switch (literal.*) {
            .Integer => |*int_lit| {
                if (context_type == null) {
                    const error_msg = try std.fmt.allocPrint(self.allocator, "Integer literal '{s}' requires explicit type annotation. In Ora, you must specify the type explicitly.", .{int_lit.value});

                    const suggestions = try self.allocator.alloc([]const u8, 8);
                    suggestions[0] = try std.fmt.allocPrint(self.allocator, "let var_name: u8 = {s};", .{int_lit.value});
                    suggestions[1] = try std.fmt.allocPrint(self.allocator, "let var_name: u16 = {s};", .{int_lit.value});
                    suggestions[2] = try std.fmt.allocPrint(self.allocator, "let var_name: u32 = {s};", .{int_lit.value});
                    suggestions[3] = try std.fmt.allocPrint(self.allocator, "let var_name: u64 = {s};", .{int_lit.value});
                    suggestions[4] = try std.fmt.allocPrint(self.allocator, "let var_name: u128 = {s};", .{int_lit.value});
                    suggestions[5] = try std.fmt.allocPrint(self.allocator, "let var_name: u256 = {s};", .{int_lit.value});
                    suggestions[6] = try std.fmt.allocPrint(self.allocator, "let var_name: i128 = {s};", .{int_lit.value});
                    suggestions[7] = try std.fmt.allocPrint(self.allocator, "let var_name: i256 = {s};", .{int_lit.value});

                    const error_obj = type_validator.ValidationError.init(
                        .invalid_operation,
                        error_msg,
                        int_lit.span,
                    ).withSuggestions(suggestions).withOperation("integer literal without explicit type");

                    try result.addError(error_obj);
                }
            },
            .String, .Bool, .Address, .Hex => {
                // These literals have implicit types and are allowed
            },
        }
    }

    /// Check that no implicit conversions are performed
    /// In Ora: u32 + u64 is not allowed without explicit casting
    pub fn checkNoImplicitConversion(
        self: *ExplicitTypeChecker,
        from_type: typer.OraType,
        to_type: typer.OraType,
        span: ast.SourceSpan,
        operation_context: []const u8,
        result: *type_validator.ValidationResult,
    ) !bool {
        // In Ora, types must match exactly - no implicit conversions
        if (!self.areTypesExactMatch(from_type, to_type)) {
            const error_msg = try std.fmt.allocPrint(self.allocator, "No implicit conversion from '{}' to '{}' in {}. Ora requires explicit type conversions.", .{ from_type, to_type, operation_context });

            const suggestions = try self.allocator.alloc([]const u8, 2);
            suggestions[0] = try std.fmt.allocPrint(self.allocator, "Use explicit cast: (value as {})", .{to_type});
            suggestions[1] = try std.fmt.allocPrint(self.allocator, "Ensure both operands have the same type: {} and {}", .{ from_type, to_type });

            const error_obj = type_validator.ValidationError.init(
                .type_mismatch,
                error_msg,
                span,
            ).withTypes(to_type, from_type).withSuggestions(suggestions).withOperation("implicit conversion");

            try result.addError(error_obj);
            return false;
        }
        return true;
    }

    /// Validate that a variable declaration has explicit type annotation
    /// In Ora: let a = value; (WRONG) vs let a: Type = value; (CORRECT)
    pub fn validateExplicitTypeAnnotation(
        self: *ExplicitTypeChecker,
        var_decl: *ast.VariableDeclNode,
        result: *type_validator.ValidationResult,
    ) !void {
        switch (var_decl.typ) {
            .Unknown => {
                const error_msg = try std.fmt.allocPrint(self.allocator, "Variable '{s}' must have explicit type annotation. Ora does not support type inference.", .{var_decl.name});

                const suggestions = try self.allocator.alloc([]const u8, 5);
                suggestions[0] = try std.fmt.allocPrint(self.allocator, "let {s}: u256 = value;", .{var_decl.name});
                suggestions[1] = try std.fmt.allocPrint(self.allocator, "let {s}: bool = value;", .{var_decl.name});
                suggestions[2] = try std.fmt.allocPrint(self.allocator, "let {s}: address = value;", .{var_decl.name});
                suggestions[3] = try std.fmt.allocPrint(self.allocator, "let {s}: string = value;", .{var_decl.name});
                suggestions[4] = try std.fmt.allocPrint(self.allocator, "let {s}: bytes = value;", .{var_decl.name});

                const error_obj = type_validator.ValidationError.init(
                    .invalid_operation,
                    error_msg,
                    var_decl.span,
                ).withSuggestions(suggestions).withOperation("missing explicit type annotation");

                try result.addError(error_obj);
            },
            .Inferred => {
                const error_msg = try std.fmt.allocPrint(self.allocator, "Inferred types are not allowed for variable '{s}'. Ora requires explicit type annotations.", .{var_decl.name});

                const error_obj = type_validator.ValidationError.init(
                    .unsupported_feature,
                    error_msg,
                    var_decl.span,
                ).withOperation("type inference");

                try result.addError(error_obj);
            },
            else => {
                // Explicit type annotation is present, which is correct for Ora
                // Validate that the type is valid
                _ = try self.validator.validateType(&var_decl.typ, result);
            },
        }
    }

    /// Check type compatibility without allowing implicit conversions
    /// This is stricter than normal type compatibility checking
    pub fn checkTypeCompatibilityStrict(
        self: *ExplicitTypeChecker,
        expected: typer.OraType,
        actual: typer.OraType,
        span: ast.SourceSpan,
        context: []const u8,
        result: *type_validator.ValidationResult,
    ) !bool {
        if (!self.areTypesExactMatch(expected, actual)) {
            const error_msg = try std.fmt.allocPrint(self.allocator, "Type mismatch in {s}: expected '{}', got '{}'. Ora requires exact type matches.", .{ context, expected, actual });

            const suggestions = try self.allocator.alloc([]const u8, 3);
            suggestions[0] = try std.fmt.allocPrint(self.allocator, "Use explicit cast: (value as {})", .{expected});
            suggestions[1] = try std.fmt.allocPrint(self.allocator, "Change variable type to match: let var: {} = value;", .{actual});
            suggestions[2] = "Ensure all operands in expressions have the same type";

            const error_obj = type_validator.ValidationError.init(
                .type_mismatch,
                error_msg,
                span,
            ).withTypes(expected, actual).withSuggestions(suggestions);

            try result.addError(error_obj);
            return false;
        }
        return true;
    }

    /// Validate arithmetic operations with strict type checking
    /// In Ora: u32 + u64 requires explicit casting
    pub fn validateArithmeticOperation(
        self: *ExplicitTypeChecker,
        operator: ast.BinaryOp,
        lhs_type: typer.OraType,
        rhs_type: typer.OraType,
        span: ast.SourceSpan,
        result: *type_validator.ValidationResult,
    ) !typer.OraType {
        // Check that both operands are numeric
        if (!self.isNumericType(lhs_type) or !self.isNumericType(rhs_type)) {
            const error_msg = try std.fmt.allocPrint(self.allocator, "Arithmetic operation '{}' requires numeric operands, got '{}' and '{}'", .{ operator, lhs_type, rhs_type });

            const error_obj = type_validator.ValidationError.init(
                .invalid_operation,
                error_msg,
                span,
            ).withTypes(lhs_type, rhs_type).withOperation("arithmetic operation");

            try result.addError(error_obj);
            return typer.OraType.Unknown;
        }

        // Check for exact type match (no implicit conversions)
        if (!self.areTypesExactMatch(lhs_type, rhs_type)) {
            const error_msg = try std.fmt.allocPrint(self.allocator, "Arithmetic operation '{}' requires operands of the same type. Got '{}' and '{}'. Use explicit casting.", .{ operator, lhs_type, rhs_type });

            const suggestions = try self.allocator.alloc([]const u8, 2);
            suggestions[0] = try std.fmt.allocPrint(self.allocator, "Cast left operand: (lhs as {}) {} rhs", .{ rhs_type, operator });
            suggestions[1] = try std.fmt.allocPrint(self.allocator, "Cast right operand: lhs {} (rhs as {})", .{ operator, lhs_type });

            const error_obj = type_validator.ValidationError.init(
                .type_mismatch,
                error_msg,
                span,
            ).withTypes(lhs_type, rhs_type).withSuggestions(suggestions);

            try result.addError(error_obj);
            return typer.OraType.Unknown;
        }

        // Additional checks for division operations
        if (operator == .Slash or operator == .Percent) {
            const warning_msg = "Division operations should include runtime checks for zero divisor to prevent panics";
            const warning_obj = type_validator.ValidationError.init(
                .invalid_operation,
                warning_msg,
                span,
            ).withSeverity(.warning).withOperation("division operation");

            try result.addError(warning_obj);
        }

        return lhs_type; // Both types are the same, return either one
    }

    /// Validate assignment operations with strict type checking
    pub fn validateAssignment(
        self: *ExplicitTypeChecker,
        target_type: typer.OraType,
        value_type: typer.OraType,
        span: ast.SourceSpan,
        result: *type_validator.ValidationResult,
    ) !bool {
        return self.checkTypeCompatibilityStrict(
            target_type,
            value_type,
            span,
            "assignment",
            result,
        );
    }

    /// Validate function call arguments with strict type checking
    pub fn validateFunctionCallArguments(
        self: *ExplicitTypeChecker,
        expected_params: []typer.OraType,
        actual_args: []typer.OraType,
        call_span: ast.SourceSpan,
        result: *type_validator.ValidationResult,
    ) !bool {
        if (expected_params.len != actual_args.len) {
            const error_msg = try std.fmt.allocPrint(self.allocator, "Function call argument count mismatch: expected {}, got {}", .{ expected_params.len, actual_args.len });

            const error_obj = type_validator.ValidationError.init(
                .invalid_operation,
                error_msg,
                call_span,
            ).withOperation("function call");

            try result.addError(error_obj);
            return false;
        }

        var all_match = true;
        for (expected_params, actual_args, 0..) |expected, actual, i| {
            if (!self.areTypesExactMatch(expected, actual)) {
                const error_msg = try std.fmt.allocPrint(self.allocator, "Function call argument {} type mismatch: expected '{}', got '{}'", .{ i + 1, expected, actual });

                const error_obj = type_validator.ValidationError.init(
                    .type_mismatch,
                    error_msg,
                    call_span,
                ).withTypes(expected, actual);

                try result.addError(error_obj);
                all_match = false;
            }
        }

        return all_match;
    }

    /// Check if a cast operation is explicit and valid
    pub fn validateExplicitCast(
        self: *ExplicitTypeChecker,
        from_type: typer.OraType,
        to_type: typer.OraType,
        cast_type: ast.CastType,
        span: ast.SourceSpan,
        result: *type_validator.ValidationResult,
    ) !typer.OraType {
        switch (cast_type) {
            .Unsafe => {
                // Unsafe casts are explicit and allowed, but warn about potential issues
                if (!self.isValidUnsafeCast(from_type, to_type)) {
                    const warning_msg = try std.fmt.allocPrint(self.allocator, "Unsafe cast from '{}' to '{}' may cause data loss or unexpected behavior", .{ from_type, to_type });

                    const warning_obj = type_validator.ValidationError.init(
                        .invalid_operation,
                        warning_msg,
                        span,
                    ).withSeverity(.warning).withOperation("unsafe cast");

                    try result.addError(warning_obj);
                }
                return to_type;
            },
            .Safe => {
                // Safe casts must be valid conversions
                if (!self.isValidSafeCast(from_type, to_type)) {
                    const error_msg = try std.fmt.allocPrint(self.allocator, "Safe cast from '{}' to '{}' is not valid. Use unsafe cast if intended.", .{ from_type, to_type });

                    const error_obj = type_validator.ValidationError.init(
                        .invalid_operation,
                        error_msg,
                        span,
                    ).withOperation("safe cast");

                    try result.addError(error_obj);
                    return typer.OraType.Unknown;
                }
                return to_type;
            },
            .Forced => {
                // Forced casts are always allowed but should be used sparingly
                const info_msg = try std.fmt.allocPrint(self.allocator, "Forced cast from '{}' to '{}' bypasses all safety checks", .{ from_type, to_type });

                const info_obj = type_validator.ValidationError.init(
                    .invalid_operation,
                    info_msg,
                    span,
                ).withSeverity(.info).withOperation("forced cast");

                try result.addError(info_obj);
                return to_type;
            },
        }
    }

    // Helper methods

    /// Check if two types are exactly the same (no implicit conversions)
    fn areTypesExactMatch(self: *ExplicitTypeChecker, type1: typer.OraType, type2: typer.OraType) bool {
        _ = self;
        return std.meta.eql(type1, type2);
    }

    /// Check if a type is numeric
    fn isNumericType(self: *ExplicitTypeChecker, type_: typer.OraType) bool {
        _ = self;
        return switch (type_) {
            .U8, .U16, .U32, .U64, .U128, .U256, .I8, .I16, .I32, .I64, .I128, .I256 => true,
            else => false,
        };
    }

    /// Check if a type is integer
    fn isIntegerType(self: *ExplicitTypeChecker, type_: typer.OraType) bool {
        return self.isNumericType(type_);
    }

    /// Check if an unsafe cast is valid (basic validation)
    fn isValidUnsafeCast(self: *ExplicitTypeChecker, from: typer.OraType, to: typer.OraType) bool {
        _ = self;
        // Most unsafe casts are allowed, but some combinations don't make sense
        return switch (from) {
            .Unknown => false,
            else => switch (to) {
                .Unknown => false,
                else => true, // Allow most unsafe casts
            },
        };
    }

    /// Check if a safe cast is valid
    fn isValidSafeCast(self: *ExplicitTypeChecker, from: typer.OraType, to: typer.OraType) bool {
        _ = self;
        // Safe casts only allow widening conversions for numeric types
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
            .I8 => switch (to) {
                .I16, .I32, .I64, .I128, .I256 => true,
                else => false,
            },
            .I16 => switch (to) {
                .I32, .I64, .I128, .I256 => true,
                else => false,
            },
            .I32 => switch (to) {
                .I64, .I128, .I256 => true,
                else => false,
            },
            .I64 => switch (to) {
                .I128, .I256 => true,
                else => false,
            },
            .I128 => switch (to) {
                .I256 => true,
                else => false,
            },
            else => std.meta.eql(from, to), // Same type is always safe
        };
    }
};
