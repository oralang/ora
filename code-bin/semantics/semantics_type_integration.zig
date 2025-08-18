const std = @import("std");
pub const ast = @import("../ast.zig");
pub const typer = @import("../typer.zig");
const semantics_errors = @import("semantics_errors.zig");

// Forward declaration for SemanticAnalyzer
const SemanticAnalyzer = @import("semantics_core.zig").SemanticAnalyzer;

/// Type system integration for seamless cooperation with type checker
pub const TypeSystemIntegration = struct {
    type_checker: *typer.Typer,
    allocator: std.mem.Allocator,
    type_cache: std.HashMap(TypeCacheKey, typer.OraType, TypeCacheContext, std.hash_map.default_max_load_percentage),

    pub const TypeCacheKey = struct {
        node_ptr: *ast.AstNode,
        context_hash: u64,

        pub fn hash(self: TypeCacheKey) u64 {
            var hasher = std.hash.Wyhash.init(0);
            hasher.update(std.mem.asBytes(&self.node_ptr));
            hasher.update(std.mem.asBytes(&self.context_hash));
            return hasher.final();
        }

        pub fn eql(a: TypeCacheKey, b: TypeCacheKey) bool {
            return a.node_ptr == b.node_ptr and a.context_hash == b.context_hash;
        }
    };

    pub const TypeCacheContext = struct {
        pub fn hash(self: @This(), key: TypeCacheKey) u64 {
            _ = self;
            return key.hash();
        }

        pub fn eql(self: @This(), a: TypeCacheKey, b: TypeCacheKey) bool {
            _ = self;
            return a.eql(b);
        }
    };

    pub fn init(allocator: std.mem.Allocator, type_checker: *typer.Typer) TypeSystemIntegration {
        return TypeSystemIntegration{
            .type_checker = type_checker,
            .allocator = allocator,
            .type_cache = std.HashMap(TypeCacheKey, typer.OraType, TypeCacheContext, std.hash_map.default_max_load_percentage).init(allocator),
        };
    }

    pub fn deinit(self: *TypeSystemIntegration) void {
        self.type_cache.deinit();
    }

    /// Get type information for an AST node
    pub fn getNodeType(self: *TypeSystemIntegration, node: *ast.AstNode) !typer.OraType {
        const cache_key = TypeCacheKey{
            .node_ptr = node,
            .context_hash = self.computeContextHash(node),
        };

        if (self.type_cache.get(cache_key)) |cached_type| {
            return cached_type;
        }

        const node_type = try self.computeNodeType(node);
        try self.type_cache.put(cache_key, node_type);
        return node_type;
    }

    /// Validate type compatibility between two nodes
    pub fn validateTypeCompatibility(self: *TypeSystemIntegration, left: *ast.AstNode, right: *ast.AstNode) !bool {
        const left_type = try self.getNodeType(left);
        const right_type = try self.getNodeType(right);

        return self.type_checker.areTypesCompatible(left_type, right_type);
    }

    /// Check if a type is a generic type
    pub fn isGenericType(self: *TypeSystemIntegration, node_type: typer.OraType) bool {
        return switch (node_type) {
            .Generic => true,
            .Array => |array_type| self.isGenericType(array_type.element_type.*),
            .Mapping => |mapping_type| self.isGenericType(mapping_type.key_type.*) or self.isGenericType(mapping_type.value_type.*),
            .ErrorUnion => |error_union| self.isGenericType(error_union.value_type.*),
            else => false,
        };
    }

    /// Validate generic type constraints
    pub fn validateGenericConstraints(self: *TypeSystemIntegration, generic_type: typer.OraType, constraints: []const typer.TypeConstraint) !bool {
        _ = self;
        _ = generic_type;
        _ = constraints;
        // TODO: Implement generic constraint validation
        return true;
    }

    /// Check if a type is an error union type
    pub fn isErrorUnionType(self: *TypeSystemIntegration, node_type: typer.OraType) bool {
        _ = self; // Parameter not used in this simple check
        return switch (node_type) {
            .ErrorUnion => true,
            else => false,
        };
    }

    /// Validate error union propagation
    pub fn validateErrorUnionPropagation(self: *TypeSystemIntegration, source: *ast.AstNode, target_context: ErrorUnionContext) !bool {
        const source_type = try self.getNodeType(source);

        if (!self.isErrorUnionType(source_type)) {
            return false;
        }

        return switch (target_context) {
            .FunctionReturn => true, // Functions can return error unions
            .TryExpression => true, // Try expressions handle error unions
            .Assignment => false, // Regular assignments don't propagate errors
        };
    }

    /// Check if a type is a mapping type
    pub fn isMappingType(self: *TypeSystemIntegration, node_type: typer.OraType) bool {
        _ = self;
        return switch (node_type) {
            .Mapping => true,
            else => false,
        };
    }

    /// Validate mapping key and value types
    pub fn validateMappingTypes(self: *TypeSystemIntegration, key_type: typer.OraType, value_type: typer.OraType) !bool {
        // Key types must be hashable (basic types, addresses, strings)
        const key_valid = switch (key_type) {
            .U8, .U16, .U32, .U64, .U128, .U256 => true,
            .I8, .I16, .I32, .I64, .I128, .I256 => true,
            .Bool => true,
            .Address => true,
            .String => true,
            else => false,
        };

        if (!key_valid) {
            return false;
        }

        // Value types can be any valid type
        return self.isValidType(value_type);
    }

    /// Check if a type is valid in the current context
    pub fn isValidType(self: *TypeSystemIntegration, node_type: typer.OraType) bool {
        return switch (node_type) {
            .Unknown => false,
            .Void => true,
            .U8, .U16, .U32, .U64, .U128, .U256 => true,
            .I8, .I16, .I32, .I64, .I128, .I256 => true,
            .Bool => true,
            .Address => true,
            .String => true,
            .Array => |array_type| self.isValidType(array_type.element_type.*),
            .Mapping => |mapping_type| self.isValidType(mapping_type.key_type.*) and self.isValidType(mapping_type.value_type.*),
            .Struct => true,
            .Enum => true,
            .Function => true,
            .ErrorUnion => |error_union| self.isValidType(error_union.value_type.*),
            .Generic => true,
        };
    }

    /// Get the underlying type of an error union
    pub fn getErrorUnionValueType(self: *TypeSystemIntegration, error_union_type: typer.OraType) ?typer.OraType {
        _ = self;
        return switch (error_union_type) {
            .ErrorUnion => |error_union| error_union.value_type.*,
            else => null,
        };
    }

    /// Validate struct field access
    pub fn validateStructFieldAccess(self: *TypeSystemIntegration, struct_type: typer.OraType, field_name: []const u8) !bool {
        _ = self;
        _ = struct_type;
        _ = field_name;
        // TODO: Implement struct field validation with actual struct definitions
        return true;
    }

    /// Get the type of a struct field
    pub fn getStructFieldType(self: *TypeSystemIntegration, struct_type: typer.OraType, field_name: []const u8) !?typer.OraType {
        _ = self;
        _ = struct_type;
        _ = field_name;
        // TODO: Implement struct field type lookup
        return null;
    }

    /// Validate function signature compatibility
    pub fn validateFunctionSignature(self: *TypeSystemIntegration, function_type: typer.OraType, args: []const typer.OraType) !bool {
        _ = self;
        _ = function_type;
        _ = args;
        // TODO: Implement function signature validation
        return true;
    }

    /// Private helper methods
    fn computeContextHash(self: *TypeSystemIntegration, node: *ast.AstNode) u64 {
        _ = self;
        // Simple hash based on node type and position
        var hasher = std.hash.Wyhash.init(0);
        hasher.update(std.mem.asBytes(&@as(std.meta.Tag(ast.AstNode), node.*)));
        return hasher.final();
    }

    fn computeNodeType(self: *TypeSystemIntegration, node: *ast.AstNode) !typer.OraType {
        return switch (node.*) {
            .Expr => |*expr| try self.computeExpressionType(expr),
            .Type => |*type_node| try self.type_checker.convertAstTypeToOraType(type_node),
            else => typer.OraType.Unknown,
        };
    }

    fn computeExpressionType(self: *TypeSystemIntegration, expr: *ast.ExprNode) !typer.OraType {
        return switch (expr.*) {
            .Literal => |*lit| self.computeLiteralType(lit),
            .Identifier => |*ident| try self.type_checker.getIdentifierType(ident.name),
            .Binary => |*bin| try self.computeBinaryType(bin),
            .Unary => |*un| try self.computeUnaryType(un),
            .Call => |*call| try self.computeCallType(call),
            .FieldAccess => |*field| try self.computeFieldAccessType(field),
            .Cast => |*cast| try self.type_checker.convertAstTypeToOraType(&cast.target_type),
            else => typer.OraType.Unknown,
        };
    }

    fn computeLiteralType(self: *TypeSystemIntegration, lit: *ast.LiteralExpr) typer.OraType {
        _ = self;
        return switch (lit.*) {
            .Integer => typer.OraType.U256, // Default integer type
            .String => typer.OraType.String,
            .Bool => typer.OraType.Bool,
            .Address => typer.OraType.Address,
            .Hex => typer.OraType.U256,
        };
    }

    fn computeBinaryType(self: *TypeSystemIntegration, bin: *ast.BinaryNode) !typer.OraType {
        const left_type = try self.computeExpressionType(bin.left);
        const right_type = try self.computeExpressionType(bin.right);

        return switch (bin.op) {
            .Add, .Subtract, .Multiply, .Divide, .Modulo => blk: {
                // Arithmetic operations preserve the larger type
                if (self.type_checker.areTypesCompatible(left_type, right_type)) {
                    break :blk left_type;
                } else {
                    break :blk typer.OraType.Unknown;
                }
            },
            .Equal, .NotEqual, .LessThan, .LessThanOrEqual, .GreaterThan, .GreaterThanOrEqual => typer.OraType.Bool,
            .And, .Or => typer.OraType.Bool,
            else => typer.OraType.Unknown,
        };
    }

    fn computeUnaryType(self: *TypeSystemIntegration, un: *ast.UnaryNode) !typer.OraType {
        const operand_type = try self.computeExpressionType(un.operand);

        return switch (un.op) {
            .Not => typer.OraType.Bool,
            .Negate => operand_type,
            else => typer.OraType.Unknown,
        };
    }

    fn computeCallType(self: *TypeSystemIntegration, call: *ast.CallNode) !typer.OraType {
        _ = self;
        _ = call;
        // TODO: Implement function call type resolution
        return typer.OraType.Unknown;
    }

    fn computeFieldAccessType(self: *TypeSystemIntegration, field: *ast.FieldAccessNode) !typer.OraType {
        const object_type = try self.computeExpressionType(field.object);
        return self.getStructFieldType(object_type, field.field) catch typer.OraType.Unknown orelse typer.OraType.Unknown;
    }
};

/// Error union context for validation
pub const ErrorUnionContext = enum {
    FunctionReturn,
    TryExpression,
    Assignment,
};

/// Integrate type system with semantic analysis
pub fn integrateWithTypeChecker(analyzer: *SemanticAnalyzer, integration: *TypeSystemIntegration) !void {
    // Set up bidirectional communication between semantic analyzer and type checker
    integration.type_checker = &analyzer.type_checker;

    // Initialize type cache for performance
    integration.type_cache.clearRetainingCapacity();
}

/// Validate type compatibility for assignment
pub fn validateAssignmentTypes(analyzer: *SemanticAnalyzer, integration: *TypeSystemIntegration, target: *ast.ExprNode, value: *ast.ExprNode) !bool {
    const target_type = try integration.getNodeType(&ast.AstNode{ .Expr = target });
    const value_type = try integration.getNodeType(&ast.AstNode{ .Expr = value });

    const compatible = try integration.validateTypeCompatibility(&ast.AstNode{ .Expr = target }, &ast.AstNode{ .Expr = value });

    if (!compatible) {
        const error_msg = try std.fmt.allocPrint(analyzer.allocator, "Type mismatch in assignment: cannot assign {s} to {s}", .{ @tagName(value_type), @tagName(target_type) });
        try semantics_errors.addError(analyzer, error_msg, target.*.Identifier.span); // Assuming target is identifier for simplicity
        return false;
    }

    return true;
}

/// Validate generic type instantiation
pub fn validateGenericInstantiation(analyzer: *SemanticAnalyzer, integration: *TypeSystemIntegration, generic_type: typer.OraType, type_args: []const typer.OraType) !bool {
    _ = analyzer;
    _ = integration;
    _ = generic_type;
    _ = type_args;
    // TODO: Implement generic type instantiation validation
    return true;
}
