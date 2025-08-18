const std = @import("std");
const ast = @import("../ast.zig");
const TypeInfo = @import("type_info.zig").TypeInfo;
const TypeCategory = @import("type_info.zig").TypeCategory;
const OraType = @import("type_info.zig").OraType;
const TypeSource = @import("type_info.zig").TypeSource;
const CommonTypes = @import("type_info.zig").CommonTypes;

/// Type resolution errors
pub const TypeResolutionError = error{
    UnknownType,
    TypeMismatch,
    CircularReference,
    InvalidEnumValue,
    OutOfMemory,
    IncompatibleTypes,
    UndefinedIdentifier,
};

/// Context for type resolution
pub const TypeContext = struct {
    expected_type: ?TypeInfo = null, // Expected type from parent context
    enum_underlying_type: ?TypeInfo = null, // For enum variant resolution
    function_return_type: ?TypeInfo = null, // For return statement checking

    pub fn withExpectedType(expected: TypeInfo) TypeContext {
        return TypeContext{ .expected_type = expected };
    }

    pub fn withEnumType(enum_type: TypeInfo) TypeContext {
        return TypeContext{ .enum_underlying_type = enum_type };
    }
};

/// Modern type resolver using unified TypeInfo system
pub const TypeResolver = struct {
    allocator: std.mem.Allocator,

    pub fn init(allocator: std.mem.Allocator) TypeResolver {
        return TypeResolver{
            .allocator = allocator,
        };
    }

    /// Resolve types for an entire AST
    pub fn resolveTypes(self: *TypeResolver, nodes: []ast.AstNode) TypeResolutionError!void {
        for (nodes) |*node| {
            try self.resolveNodeTypes(node, TypeContext{});
        }
    }

    /// Resolve types for a single AST node with context
    fn resolveNodeTypes(self: *TypeResolver, node: *ast.AstNode, context: TypeContext) TypeResolutionError!void {
        switch (node.*) {
            .EnumDecl => |*enum_decl| {
                try self.resolveEnumTypes(enum_decl);
            },
            .Contract => |*contract| {
                for (contract.body) |*child| {
                    try self.resolveNodeTypes(child, context);
                }
            },
            .Function => |*function| {
                // Parameters should already have explicit types, just validate them
                for (function.parameters) |*param| {
                    try self.validateTypeInfo(&param.type_info);
                }

                // Return type should be explicit or void
                if (function.return_type_info) |*ret_type| {
                    try self.validateTypeInfo(ret_type);
                }

                // Create context for function body with return type
                var func_context = context;
                func_context.function_return_type = function.return_type_info;

                // Resolve requires/ensures expressions
                for (function.requires_clauses) |*clause| {
                    try self.resolveExpressionTypes(clause, func_context);
                }
                for (function.ensures_clauses) |*clause| {
                    try self.resolveExpressionTypes(clause, func_context);
                }
            },
            .VariableDecl => |*var_decl| {
                try self.validateTypeInfo(&var_decl.type_info);
                if (var_decl.value) |*value| {
                    // Create context with expected type from variable declaration
                    const var_context = TypeContext.withExpectedType(var_decl.type_info);
                    try self.resolveExpressionTypes(value, var_context);
                }
            },
            else => {
                // Handle other node types as needed
            },
        }
    }

    /// Resolve enum types and infer variant values
    fn resolveEnumTypes(self: *TypeResolver, enum_decl: *ast.EnumDeclNode) TypeResolutionError!void {
        // Get the underlying type info, default to u8 if not specified
        const underlying_type_info = enum_decl.underlying_type_info orelse CommonTypes.u8_type();

        // Create context for enum variant resolution
        const enum_context = TypeContext.withEnumType(underlying_type_info);

        for (enum_decl.variants) |*variant| {
            if (variant.value) |*value_expr| {
                // Resolve the expression with enum context
                try self.resolveExpressionTypes(value_expr, enum_context);

                // The resolved_value field was removed - constant evaluation
                // should be done in a separate pass if needed
            }
            // Auto-increment logic removed - this should be handled in
            // a separate constant evaluation pass
        }
    }

    /// Validate that a TypeInfo is properly resolved
    fn validateTypeInfo(self: *TypeResolver, type_info: *TypeInfo) TypeResolutionError!void {
        _ = self; // unused for now

        if (!type_info.isResolved()) {
            return TypeResolutionError.UnknownType;
        }

        // Additional validation can be added here
        // e.g., check if custom types (structs, enums) are defined
    }

    /// Update literal expression types based on context
    fn updateLiteralType(self: *TypeResolver, expr: *ast.Expressions.ExprNode, target_type_info: TypeInfo) TypeResolutionError!void {
        switch (expr.*) {
            .Literal => |*literal| {
                switch (literal.*) {
                    .Integer => |*int_literal| {
                        // Update integer type info based on target type
                        if (target_type_info.ora_type) |ora_type| {
                            int_literal.type_info = TypeInfo.inferred(target_type_info.category, ora_type, int_literal.span);
                        }
                    },
                    .String => |*str_literal| {
                        if (target_type_info.category == .String) {
                            str_literal.type_info = target_type_info;
                        }
                    },
                    .Bool => |*bool_literal| {
                        if (target_type_info.category == .Bool) {
                            bool_literal.type_info = target_type_info;
                        }
                    },
                    .Address => |*addr_literal| {
                        if (target_type_info.category == .Address) {
                            addr_literal.type_info = target_type_info;
                        }
                    },
                    .Hex => |*hex_literal| {
                        if (target_type_info.category == .Hex) {
                            hex_literal.type_info = target_type_info;
                        }
                    },
                }
            },
            .Binary => |*binary| {
                // Recursively update binary expression operands
                try self.updateLiteralType(binary.lhs, target_type_info);
                try self.updateLiteralType(binary.rhs, target_type_info);
                // Update the binary expression's result type
                binary.type_info = target_type_info;
            },
            .Unary => |*unary| {
                try self.updateLiteralType(unary.operand, target_type_info);
                // Update the unary expression's result type
                unary.type_info = target_type_info;
            },
            else => {}, // Other expression types don't need updating
        }
    }

    /// Evaluate constant expressions to compute their values
    fn evaluateConstantExpression(self: *TypeResolver, expr: *ast.Expressions.ExprNode) TypeResolutionError!?u256 {
        switch (expr.*) {
            .Literal => |*literal| {
                switch (literal.*) {
                    .Integer => |*int_literal| {
                        return std.fmt.parseInt(u256, int_literal.value, 0) catch null;
                    },
                    .Bool => |*bool_literal| {
                        return if (bool_literal.value) 1 else 0;
                    },
                    else => return null,
                }
            },
            .Binary => |*binary| {
                const lhs = try self.evaluateConstantExpression(binary.lhs) orelse return null;
                const rhs = try self.evaluateConstantExpression(binary.rhs) orelse return null;

                return switch (binary.operator) {
                    .Plus => lhs + rhs,
                    .Minus => if (lhs >= rhs) lhs - rhs else null,
                    .Star => lhs * rhs,
                    .Slash => if (rhs != 0) lhs / rhs else null,
                    .Percent => if (rhs != 0) lhs % rhs else null,
                    .BitwiseAnd => lhs & rhs,
                    .BitwiseOr => lhs | rhs,
                    .BitwiseXor => lhs ^ rhs,
                    .LeftShift => lhs << @intCast(rhs),
                    .RightShift => lhs >> @intCast(rhs),
                    else => null, // Non-arithmetic operations
                };
            },
            .Unary => |*unary| {
                const operand = try self.evaluateConstantExpression(unary.operand) orelse return null;

                return switch (unary.operator) {
                    .Minus => if (operand == 0) 0 else null, // Only allow -0 for unsigned
                    .Bang => if (operand == 0) 1 else 0,
                    else => null,
                };
            },
            .Identifier => |*identifier| {
                // Identifier constant evaluation not implemented
                // Would require symbol table access for scope lookup
                _ = identifier;
                return null;
            },
            else => return null,
        }
    }

    /// Resolve expression types with context
    fn resolveExpressionTypes(self: *TypeResolver, expr: *ast.Expressions.ExprNode, context: TypeContext) TypeResolutionError!void {
        switch (expr.*) {
            .Literal => {
                // If we have expected type from context, apply it
                if (context.expected_type) |expected| {
                    try self.updateLiteralType(expr, expected);
                } else if (context.enum_underlying_type) |enum_type| {
                    try self.updateLiteralType(expr, enum_type);
                }
            },
            .Binary => |*binary| {
                try self.resolveExpressionTypes(binary.lhs, context);
                try self.resolveExpressionTypes(binary.rhs, context);

                // Infer result type from operands or context
                if (context.expected_type) |expected| {
                    binary.type_info = expected;
                } else if (context.enum_underlying_type) |enum_type| {
                    binary.type_info = enum_type;
                }
            },
            .Unary => |*unary| {
                try self.resolveExpressionTypes(unary.operand, context);

                // Infer result type from operand or context
                if (context.expected_type) |expected| {
                    unary.type_info = expected;
                } else if (context.enum_underlying_type) |enum_type| {
                    unary.type_info = enum_type;
                }
            },
            .Call => |*call| {
                try self.resolveExpressionTypes(call.callee, context);
                for (call.arguments) |*arg| {
                    try self.resolveExpressionTypes(arg, context);
                }
                // Call result type should be resolved from function signature
            },
            .Index => |*index| {
                try self.resolveExpressionTypes(index.target, context);
                try self.resolveExpressionTypes(index.index, context);
                // Index result type should be resolved from target type
            },
            .FieldAccess => |*field_access| {
                try self.resolveExpressionTypes(field_access.target, context);
                // Field access result type should be resolved from struct definition
            },
            .Cast => |*cast| {
                try self.resolveExpressionTypes(cast.operand, context);
                // Cast target type should already be explicit
                try self.validateTypeInfo(&cast.target_type);
            },
            else => {
                // Other expression types don't need special handling
            },
        }
    }
};
