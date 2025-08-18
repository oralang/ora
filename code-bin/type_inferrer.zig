const std = @import("std");
const ast = @import("ast.zig");
const typer = @import("typer.zig");
const type_validator = @import("type_validator.zig");
const ast_arena = @import("ast/ast_arena.zig");

/// Type constraint kinds for type validation
pub const ConstraintKind = enum {
    /// Type must be convertible to another type
    ConvertibleTo,
    /// Type must have a specific size
    SizeEquals,
    /// Type must be numeric
    Numeric,
    /// Type must be integral
    Integral,
    /// Type must support specific operations
    SupportsOp,
};

/// Type constraint for type validation
pub const TypeConstraint = struct {
    kind: ConstraintKind,
    target_type: ?typer.OraType,
    operation: ?[]const u8,
    size: ?u32,

    pub fn isSatisfiedBy(self: TypeConstraint, type_: typer.OraType) bool {
        return switch (self.kind) {
            .ConvertibleTo => {
                if (self.target_type) |target| {
                    return isConvertible(type_, target);
                }
                return false;
            },
            .SizeEquals => {
                if (self.size) |size| {
                    return typer.getTypeSize(type_) == size;
                }
                return false;
            },
            .Numeric => {
                return isNumericType(type_);
            },
            .Integral => {
                return isIntegralType(type_);
            },
            .SupportsOp => {
                if (self.operation) |op| {
                    return supportsOperation(type_, op);
                }
                return false;
            },
        };
    }
};

/// Inference context for tracking type variables and constraints
pub const InferenceContext = struct {
    allocator: std.mem.Allocator,
    type_variables: std.HashMap([]const u8, typer.OraType, std.hash_map.StringContext, std.hash_map.default_max_load_percentage),
    constraints: std.ArrayList(TypeConstraint),

    pub fn init(allocator: std.mem.Allocator) InferenceContext {
        return InferenceContext{
            .allocator = allocator,
            .type_variables = std.HashMap([]const u8, typer.OraType, std.hash_map.StringContext, std.hash_map.default_max_load_percentage).init(allocator),
            .constraints = std.ArrayList(TypeConstraint).init(allocator),
        };
    }

    pub fn deinit(self: *InferenceContext) void {
        self.type_variables.deinit();
        self.constraints.deinit();
    }

    pub fn addConstraint(self: *InferenceContext, constraint: TypeConstraint) !void {
        try self.constraints.append(constraint);
    }

    pub fn setTypeVariable(self: *InferenceContext, name: []const u8, type_: typer.OraType) !void {
        try self.type_variables.put(name, type_);
    }

    pub fn getTypeVariable(self: *InferenceContext, name: []const u8) ?typer.OraType {
        return self.type_variables.get(name);
    }
};

/// Type inference engine - Ora philosophy: explicit over implicit
pub const TypeInferrer = struct {
    allocator: std.mem.Allocator,
    arena: *ast_arena.AstArena,
    typer_instance: *typer.Typer,
    validator: *type_validator.TypeValidator,
    context: InferenceContext,

    pub fn init(allocator: std.mem.Allocator, arena: *ast_arena.AstArena, typer_instance: *typer.Typer, validator: *type_validator.TypeValidator) TypeInferrer {
        return TypeInferrer{
            .allocator = allocator,
            .arena = arena,
            .typer_instance = typer_instance,
            .validator = validator,
            .context = InferenceContext.init(allocator),
        };
    }

    pub fn deinit(self: *TypeInferrer) void {
        self.context.deinit();
    }

    /// Infer the type of an expression (excluding literals which require explicit types)
    pub fn inferExpressionType(self: *TypeInferrer, expr: *ast.ExprNode) !typer.OraType {
        return switch (expr.*) {
            .Identifier => |*ident| {
                // Look up symbol in current scope
                if (self.typer_instance.current_scope.lookup(ident.name)) |symbol| {
                    return symbol.typ;
                }
                // Check if it's a type variable
                if (self.context.getTypeVariable(ident.name)) |type_var| {
                    return type_var;
                }
                // Unknown identifier
                return typer.OraType.Unknown;
            },

            .Literal => |*literal| {
                // Literals require explicit type annotations in Ora
                return try self.validateLiteralType(literal);
            },

            .Binary => |*binary| {
                const lhs_type = try self.inferExpressionType(binary.lhs);
                const rhs_type = try self.inferExpressionType(binary.rhs);
                return try self.inferBinaryExpressionType(binary.operator, lhs_type, rhs_type);
            },

            .Unary => |*unary| {
                const operand_type = try self.inferExpressionType(unary.operand);
                return try self.inferUnaryExpressionType(unary.operator, operand_type);
            },

            .Call => |*call| {
                return try self.inferCallExpressionType(call);
            },

            .Index => |*index| {
                const target_type = try self.inferExpressionType(index.target);
                const index_type = try self.inferExpressionType(index.index);
                return try self.inferIndexExpressionType(target_type, index_type);
            },

            .FieldAccess => |*field| {
                const target_type = try self.inferExpressionType(field.target);
                return try self.inferFieldAccessType(target_type, field.field);
            },

            .Cast => |*cast| {
                const operand_type = try self.inferExpressionType(cast.operand);
                return try self.validateCastType(cast.target_type, operand_type);
            },

            .Tuple => |*tuple| {
                const element_types = try self.allocator.alloc(typer.OraType, tuple.elements.len);
                for (tuple.elements, 0..) |*element, i| {
                    element_types[i] = try self.inferExpressionType(element);
                }
                return typer.OraType{ .Tuple = .{ .types = element_types } };
            },

            .Try => |*try_expr| {
                const expr_type = try self.inferExpressionType(try_expr.expr);
                return try self.inferTryExpressionType(expr_type);
            },

            .If => |*if_expr| {
                _ = try self.inferExpressionType(if_expr.condition);
                const then_type = try self.inferExpressionType(if_expr.then_expr);
                const else_type = if (if_expr.else_expr) |else_expr|
                    try self.inferExpressionType(else_expr)
                else
                    typer.OraType.Unknown;
                return try self.unifyTypes(then_type, else_type);
            },

            .Block => |*block| {
                // For block expressions, return the type of the last expression
                if (block.statements.len > 0) {
                    const last_stmt = block.statements[block.statements.len - 1];
                    if (last_stmt.* == .Expression) {
                        return try self.inferExpressionType(&last_stmt.Expression);
                    }
                }
                return typer.OraType.Unknown;
            },
        };
    }

    /// Unify two types, finding a common type or creating constraints
    /// Note: No automatic type promotion - types must be explicitly compatible
    pub fn unifyTypes(self: *TypeInferrer, type1: typer.OraType, type2: typer.OraType) !typer.OraType {
        // If either type is Unknown, return the other
        if (type1 == .Unknown) return type2;
        if (type2 == .Unknown) return type1;

        // If types are identical, return either
        if (std.meta.eql(type1, type2)) return type1;

        // For smart contracts, we don't automatically promote types
        // Types must be explicitly compatible
        if (self.areTypesCompatible(type1, type2)) {
            // Return the more specific type if compatible
            return if (self.isMoreSpecific(type1, type2)) type1 else type2;
        }

        // Handle tuple unification
        if (type1 == .Tuple and type2 == .Tuple) {
            const tuple1 = type1.Tuple;
            const tuple2 = type2.Tuple;

            if (tuple1.types.len != tuple2.types.len) {
                return typer.OraType.Unknown; // Incompatible tuple sizes
            }

            const unified_types = try self.allocator.alloc(typer.OraType, tuple1.types.len);
            for (tuple1.types, tuple2.types, 0..) |t1, t2, i| {
                unified_types[i] = try self.unifyTypes(t1, t2);
            }

            return typer.OraType{ .Tuple = .{ .types = unified_types } };
        }

        // Handle slice unification
        if (type1 == .Slice and type2 == .Slice) {
            const element1 = type1.Slice.*;
            const element2 = type2.Slice.*;
            const unified_element = try self.unifyTypes(element1, element2);
            const element_ptr = try self.allocator.create(typer.OraType);
            element_ptr.* = unified_element;
            return typer.OraType{ .Slice = element_ptr };
        }

        // Handle mapping unification
        if (type1 == .Mapping and type2 == .Mapping) {
            const mapping1 = type1.Mapping;
            const mapping2 = type2.Mapping;

            const unified_key = try self.unifyTypes(mapping1.key.*, mapping2.key.*);
            const unified_value = try self.unifyTypes(mapping1.value.*, mapping2.value.*);

            const key_ptr = try self.allocator.create(typer.OraType);
            const value_ptr = try self.allocator.create(typer.OraType);
            key_ptr.* = unified_key;
            value_ptr.* = unified_value;

            return typer.OraType{ .Mapping = .{
                .key = key_ptr,
                .value = value_ptr,
            } };
        }

        // If no unification is possible, return Unknown
        return typer.OraType.Unknown;
    }

    /// Resolve a type with the given constraints
    pub fn resolveTypeWithConstraints(self: *TypeInferrer, type_name: []const u8, constraints: []TypeConstraint) !typer.OraType {
        // Check if we have a concrete type for this type variable
        if (self.context.getTypeVariable(type_name)) |concrete_type| {
            // Verify all constraints are satisfied
            for (constraints) |constraint| {
                if (!constraint.isSatisfiedBy(concrete_type)) {
                    return typer.OraType.Unknown; // Constraint violation
                }
            }
            return concrete_type;
        }

        // Try to find a type that satisfies all constraints
        const candidate_types = [_]typer.OraType{
            typer.OraType.U8,   typer.OraType.U16,     typer.OraType.U32,    typer.OraType.U64,   typer.OraType.U128, typer.OraType.U256,
            typer.OraType.I8,   typer.OraType.I16,     typer.OraType.I32,    typer.OraType.I64,   typer.OraType.I128, typer.OraType.I256,
            typer.OraType.Bool, typer.OraType.Address, typer.OraType.String, typer.OraType.Bytes,
        };

        for (candidate_types) |candidate| {
            var satisfies_all = true;
            for (constraints) |constraint| {
                if (!constraint.isSatisfiedBy(candidate)) {
                    satisfies_all = false;
                    break;
                }
            }
            if (satisfies_all) {
                try self.context.setTypeVariable(type_name, candidate);
                return candidate;
            }
        }

        // No suitable type found
        return typer.OraType.Unknown;
    }

    // Private helper methods

    /// Validate literal types - literals require explicit type annotations in Ora
    fn validateLiteralType(self: *TypeInferrer, literal: *ast.LiteralNode) !typer.OraType {
        _ = self;
        return switch (literal.*) {
            .Integer => |*int_lit| {
                _ = int_lit;
                // Integer literals must have explicit type annotations
                // Return Unknown to force explicit annotation
                return typer.OraType.Unknown;
            },
            .String => typer.OraType.String,
            .Bool => typer.OraType.Bool,
            .Address => typer.OraType.Address,
            .Hex => typer.OraType.U256,
        };
    }

    fn inferBinaryExpressionType(self: *TypeInferrer, operator: ast.BinaryOperator, lhs: typer.OraType, rhs: typer.OraType) !typer.OraType {
        return switch (operator) {
            .Add, .Sub, .Mul, .Div, .Mod => {
                if (self.isNumericType(lhs) and self.isNumericType(rhs)) {
                    // For smart contracts, require explicit type compatibility
                    if (lhs == rhs) {
                        return lhs;
                    }
                    // Don't automatically promote - require explicit cast
                    return typer.OraType.Unknown;
                }
                return typer.OraType.Unknown;
            },
            .BitAnd, .BitOr, .BitXor, .BitShl, .BitShr => {
                if (self.isIntegerType(lhs) and self.isIntegerType(rhs)) {
                    // For smart contracts, require explicit type compatibility
                    if (lhs == rhs) {
                        return lhs;
                    }
                    // Don't automatically promote - require explicit cast
                    return typer.OraType.Unknown;
                }
                return typer.OraType.Unknown;
            },
            .Eq, .Ne, .Lt, .Le, .Gt, .Ge => {
                // Comparison operators return boolean
                return typer.OraType.Bool;
            },
            .And, .Or => {
                // Logical operators require boolean operands and return boolean
                if (lhs == .Bool and rhs == .Bool) {
                    return typer.OraType.Bool;
                }
                return typer.OraType.Unknown;
            },
        };
    }

    fn inferUnaryExpressionType(self: *TypeInferrer, operator: ast.UnaryOperator, operand: typer.OraType) !typer.OraType {
        return switch (operator) {
            .Minus => {
                if (self.isNumericType(operand)) {
                    return operand;
                }
                return typer.OraType.Unknown;
            },
            .Bang => {
                if (operand == .Bool) {
                    return typer.OraType.Bool;
                }
                return typer.OraType.Unknown;
            },
            .BitNot => {
                if (self.isIntegerType(operand)) {
                    return operand;
                }
                return typer.OraType.Unknown;
            },
        };
    }

    fn inferCallExpressionType(self: *TypeInferrer, call: *ast.CallExpr) !typer.OraType {
        _ = try self.inferExpressionType(call.callee);

        // Handle built-in functions
        if (call.callee.* == .Identifier) {
            const func_name = call.callee.Identifier.name;

            if (std.mem.eql(u8, func_name, "require") or std.mem.eql(u8, func_name, "assert")) {
                return typer.OraType.Bool;
            }

            if (std.mem.eql(u8, func_name, "hash")) {
                return typer.OraType.U256;
            }

            if (std.mem.eql(u8, func_name, "len")) {
                // len() returns u256 for array/slice length
                return typer.OraType.U256;
            }
        }

        // TODO: Implement function type lookup from symbol table
        return typer.OraType.Unknown;
    }

    fn inferIndexExpressionType(self: *TypeInferrer, target: typer.OraType, index: typer.OraType) !typer.OraType {
        return switch (target) {
            .Slice => |element_type| {
                if (!self.isIntegerType(index)) {
                    return typer.OraType.Unknown;
                }
                return element_type.*;
            },
            .Mapping => |mapping| {
                if (!self.areTypesCompatible(mapping.key.*, index)) {
                    return typer.OraType.Unknown;
                }
                return mapping.value.*;
            },
            .DoubleMap => |double_map| {
                // For double maps, we need to check both key types
                // This is a simplified implementation
                return double_map.value.*;
            },
            else => typer.OraType.Unknown,
        };
    }

    fn inferFieldAccessType(self: *TypeInferrer, target: typer.OraType, field: []const u8) !typer.OraType {
        _ = self;
        return switch (target) {
            .Struct => |struct_type| {
                // Look up field in struct definition
                for (struct_type.fields) |struct_field| {
                    if (std.mem.eql(u8, struct_field.name, field)) {
                        return struct_field.typ;
                    }
                }
                return typer.OraType.Unknown;
            },
            else => typer.OraType.Unknown,
        };
    }

    fn validateCastType(self: *TypeInferrer, target_type: ast.TypeRef, operand_type: typer.OraType) !typer.OraType {
        var result = type_validator.ValidationResult.init(self.allocator);
        defer result.deinit();

        const validated_target = try self.validator.validateType(&target_type, &result);

        // Check if cast is valid
        if (!self.isValidCast(operand_type, validated_target)) {
            return typer.OraType.Unknown;
        }

        return validated_target;
    }

    fn inferTryExpressionType(self: *TypeInferrer, expr_type: typer.OraType) !typer.OraType {
        _ = self;
        return switch (expr_type) {
            .ErrorUnion => |error_union| {
                return error_union.success_type.*;
            },
            else => expr_type, // For non-error types, try has no effect
        };
    }

    fn isMoreSpecific(self: *TypeInferrer, type1: typer.OraType, type2: typer.OraType) bool {
        _ = self;
        // Simple heuristic: smaller types are more specific
        const size1 = typer.getTypeSize(type1);
        const size2 = typer.getTypeSize(type2);
        return size1 < size2;
    }

    fn isNumericType(self: *TypeInferrer, type_: typer.OraType) bool {
        _ = self;
        return switch (type_) {
            .U8, .U16, .U32, .U64, .U128, .U256, .I8, .I16, .I32, .I64, .I128, .I256 => true,
            else => false,
        };
    }

    fn isIntegerType(self: *TypeInferrer, type_: typer.OraType) bool {
        return self.isNumericType(type_);
    }

    fn areTypesCompatible(self: *TypeInferrer, lhs: typer.OraType, rhs: typer.OraType) bool {
        return self.typer_instance.typesCompatible(lhs, rhs);
    }

    fn isValidCast(self: *TypeInferrer, from: typer.OraType, to: typer.OraType) bool {
        _ = self;
        // Implement cast validity checking
        return switch (from) {
            .Unknown => false,
            else => switch (to) {
                .Unknown => false,
                else => true, // Simplified for now
            },
        };
    }
};

// Helper functions for type constraints

fn isConvertible(from: typer.OraType, to: typer.OraType) bool {
    // Implement type conversion rules
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
        else => std.meta.eql(from, to),
    };
}

fn isNumericType(type_: typer.OraType) bool {
    return switch (type_) {
        .U8, .U16, .U32, .U64, .U128, .U256, .I8, .I16, .I32, .I64, .I128, .I256 => true,
        else => false,
    };
}

fn isIntegralType(type_: typer.OraType) bool {
    return isNumericType(type_);
}

fn supportsOperation(type_: typer.OraType, operation: []const u8) bool {
    return switch (type_) {
        .U8, .U16, .U32, .U64, .U128, .U256, .I8, .I16, .I32, .I64, .I128, .I256 => {
            return std.mem.eql(u8, operation, "add") or
                std.mem.eql(u8, operation, "sub") or
                std.mem.eql(u8, operation, "mul") or
                std.mem.eql(u8, operation, "div") or
                std.mem.eql(u8, operation, "mod") or
                std.mem.eql(u8, operation, "bit_and") or
                std.mem.eql(u8, operation, "bit_or") or
                std.mem.eql(u8, operation, "bit_xor") or
                std.mem.eql(u8, operation, "bit_shl") or
                std.mem.eql(u8, operation, "bit_shr");
        },
        .Bool => {
            return std.mem.eql(u8, operation, "and") or
                std.mem.eql(u8, operation, "or") or
                std.mem.eql(u8, operation, "not");
        },
        else => false,
    };
}
