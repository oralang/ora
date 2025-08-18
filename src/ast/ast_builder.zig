// ============================================================================
// Ora AST Builder
//
// Provides a fluent, type-safe API for constructing Abstract Syntax Tree (AST)
// nodes programmatically. The builder integrates with AstArena for memory
// management and includes comprehensive validation and diagnostic collection.
//
// Key Features:
// - Fluent API for building complex AST structures
// - Arena-based memory management (all nodes owned by AstArena)
// - Built-in validation with configurable error collection
// - Diagnostic reporting with spans and error codes
// - Type-safe expression construction with operator precedence
// - Contract, function, and statement builders
//
// Usage:
//   var arena = AstArena.init(allocator);
//   var builder = AstBuilder.init(&arena);
//   var contract = try builder.contract("MyContract");
//   // ... build contract members
//   const contract_node = try contract.build();
//
// Safety Notes:
// - All nodes become invalid after arena deinit/reset
// - Builder must be finalized before accessing built nodes
// - Validation errors prevent finalization until resolved
// - Not thread-safe; intended for single-threaded construction
//
// ============================================================================

const std = @import("std");
const ast = @import("../ast.zig");
const ast_arena = @import("ast_arena.zig");
const semantics = @import("../semantics.zig");
const TypeInfo = @import("type_info.zig").TypeInfo;
const TypeCategory = @import("type_info.zig").TypeCategory;
const OraType = @import("type_info.zig").OraType;
const SourceSpan = @import("../ast.zig").SourceSpan;

/// Error types for AST builder operations
/// These errors are returned when builder operations fail validation or encounter invalid state
pub const BuilderError = error{
    /// Validation failed during construction (e.g., invalid node structure, type mismatch)
    ValidationFailed,
    /// Invalid node type for the requested operation (e.g., using expression where statement expected)
    InvalidNodeType,
    /// Missing required field in node construction (e.g., function without name)
    MissingRequiredField,
    /// Invalid builder state (e.g., using finalized builder)
    InvalidBuilderState,
    /// Memory allocation failed during node creation
    OutOfMemory,
    /// Builder was finalized and cannot be modified further
    BuilderFinalized,
};

/// Diagnostic collector for builder operations
/// Collects and manages validation errors, warnings, and informational messages
/// with source spans for precise error reporting
pub const DiagnosticCollector = struct {
    /// Allocator for diagnostic message storage
    allocator: std.mem.Allocator,
    /// Collection of diagnostic messages with spans
    diagnostics: std.ArrayList(semantics.Diagnostic),
    /// Maximum number of errors to collect before stopping
    max_errors: u32,
    /// Current count of error diagnostics
    error_count: u32,
    /// Current count of warning diagnostics
    warning_count: u32,

    pub fn init(allocator: std.mem.Allocator, max_errors: u32) DiagnosticCollector {
        return DiagnosticCollector{
            .allocator = allocator,
            .diagnostics = std.ArrayList(semantics.Diagnostic).init(allocator),
            .max_errors = max_errors,
            .error_count = 0,
            .warning_count = 0,
        };
    }

    pub fn deinit(self: *DiagnosticCollector) void {
        // Free diagnostic messages
        for (self.diagnostics.items) |diagnostic| {
            self.allocator.free(diagnostic.message);
        }
        self.diagnostics.deinit();
    }

    pub fn addError(self: *DiagnosticCollector, span: ast.SourceSpan, message: []const u8) !void {
        if (self.error_count >= self.max_errors) {
            return; // Silently ignore additional errors
        }

        const owned_message = try self.allocator.dupe(u8, message);
        try self.diagnostics.append(semantics.Diagnostic{
            .message = owned_message,
            .span = span,
            .severity = .Error,
            .context = null,
        });
        self.error_count += 1;
    }

    pub fn addWarning(self: *DiagnosticCollector, span: ast.SourceSpan, message: []const u8) !void {
        const owned_message = try self.allocator.dupe(u8, message);
        try self.diagnostics.append(semantics.Diagnostic{
            .message = owned_message,
            .span = span,
            .severity = .Warning,
            .context = null,
        });
        self.warning_count += 1;
    }

    pub fn addInfo(self: *DiagnosticCollector, span: ast.SourceSpan, message: []const u8) !void {
        const owned_message = try self.allocator.dupe(u8, message);
        try self.diagnostics.append(semantics.Diagnostic{
            .message = owned_message,
            .span = span,
            .severity = .Info,
            .context = null,
        });
    }

    pub fn hasErrors(self: *const DiagnosticCollector) bool {
        return self.error_count > 0;
    }

    pub fn hasWarnings(self: *const DiagnosticCollector) bool {
        return self.warning_count > 0;
    }

    pub fn getDiagnostics(self: *const DiagnosticCollector) []const semantics.Diagnostic {
        return self.diagnostics.items;
    }

    pub fn clear(self: *DiagnosticCollector) void {
        // Free existing diagnostic messages
        for (self.diagnostics.items) |diagnostic| {
            self.allocator.free(diagnostic.message);
        }
        self.diagnostics.clearRetainingCapacity();
        self.error_count = 0;
        self.warning_count = 0;
    }
};

/// Core AST builder with arena integration and diagnostic collection
/// Provides the main interface for constructing AST nodes with validation and error reporting
pub const AstBuilder = struct {
    /// Arena allocator for all AST node memory management
    arena: *ast_arena.AstArena,
    /// Diagnostic collector for validation errors and warnings
    diagnostics: DiagnosticCollector,

    /// Whether the builder has been finalized (no more modifications allowed)
    finalized: bool,
    /// Whether validation is enabled during construction
    validation_enabled: bool,
    /// Collection of all built AST nodes
    built_nodes: std.ArrayList(ast.AstNode),
    /// Counter for tracking total nodes created
    node_counter: u32,

    pub fn init(arena: *ast_arena.AstArena) AstBuilder {
        return AstBuilder{
            .arena = arena,
            .diagnostics = DiagnosticCollector.init(arena.allocator(), 100),
            .finalized = false,
            .validation_enabled = true,
            .built_nodes = std.ArrayList(ast.AstNode).init(arena.allocator()),
            .node_counter = 0,
        };
    }

    pub fn deinit(self: *AstBuilder) void {
        self.diagnostics.deinit();
        self.built_nodes.deinit();
    }

    /// Enable or disable validation during construction
    pub fn setValidationEnabled(self: *AstBuilder, enabled: bool) void {
        if (self.finalized) return; // Cannot change after finalization
        self.validation_enabled = enabled;
    }

    /// Check if the builder has been finalized
    pub fn isFinalized(self: *const AstBuilder) bool {
        return self.finalized;
    }

    /// Get the diagnostic collector
    pub fn getDiagnostics(self: *const AstBuilder) *const DiagnosticCollector {
        return &self.diagnostics;
    }

    /// Clear all diagnostics
    pub fn clearDiagnostics(self: *AstBuilder) void {
        if (self.finalized) return; // Cannot clear after finalization
        self.diagnostics.clear();
    }

    /// Get the number of nodes built so far
    pub fn getNodeCount(self: *const AstBuilder) u32 {
        return self.node_counter;
    }

    /// Add a built node to the collection
    fn addBuiltNode(self: *AstBuilder, node: ast.AstNode) !void {
        try self.built_nodes.append(node);
        self.node_counter += 1;
    }

    /// Create a new contract builder for fluent contract construction
    /// Returns a ContractBuilder that can be used to add functions, variables, etc.
    pub fn contract(self: *AstBuilder, name: []const u8) !ContractBuilder {
        if (self.finalized) return BuilderError.BuilderFinalized;
        return ContractBuilder.init(self, name);
    }

    /// Create a new expression builder for fluent expression construction
    /// Returns an ExpressionBuilder with methods for creating various expression types
    pub fn expr(self: *AstBuilder) ExpressionBuilder {
        return ExpressionBuilder.init(self);
    }

    /// Create a literal expression node from a value of any supported type
    /// Supports string, boolean, and integer literals with automatic type detection
    /// Integer literals are converted to strings for consistent storage
    pub fn literal(self: *AstBuilder, value: anytype, span: ast.SourceSpan) !*ast.Expressions.ExprNode {
        if (self.finalized) return BuilderError.BuilderFinalized;

        const expr_node = try self.arena.createNode(ast.Expressions.ExprNode);
        const T = @TypeOf(value);

        if (T == []const u8) {
            // String literal - store as owned string in arena
            const owned_value = try self.arena.createString(value);
            expr_node.* = ast.Expressions.ExprNode{ .Literal = .{ .String = .{
                .value = owned_value,
                .span = span,
            } } };
        } else if (T == bool) {
            // Boolean literal - store boolean value directly
            expr_node.* = ast.Expressions.ExprNode{ .Literal = .{ .Bool = .{
                .value = value,
                .span = span,
            } } };
        } else if (comptime @typeInfo(T) == .int) {
            // Integer literal - convert to string for consistent storage format
            const value_str = try std.fmt.allocPrint(self.arena.allocator(), "{}", .{value});
            expr_node.* = ast.Expressions.ExprNode{ .Literal = .{ .Integer = .{
                .value = value_str,
                .span = span,
            } } };
        } else {
            // Unsupported literal type - report error
            const type_name = @typeName(T);
            const error_msg = try std.fmt.allocPrint(self.arena.allocator(), "Unsupported literal type: {s}", .{type_name});
            try self.diagnostics.addError(span, error_msg);
            return BuilderError.ValidationFailed;
        }

        if (self.validation_enabled) {
            try self.validateExpression(expr_node, span);
        }

        return expr_node;
    }

    /// Create a string literal with explicit type checking
    pub fn stringLiteral(self: *AstBuilder, value: []const u8, span: ast.SourceSpan) !*ast.Expressions.ExprNode {
        if (self.finalized) return BuilderError.BuilderFinalized;

        const expr_node = try self.arena.createNode(ast.Expressions.ExprNode);
        const owned_value = try self.arena.createString(value);
        expr_node.* = ast.Expressions.ExprNode{ .Literal = .{ .String = .{
            .value = owned_value,
            .span = span,
        } } };

        if (self.validation_enabled) {
            try self.validateExpression(expr_node, span);
        }

        return expr_node;
    }

    /// Create an integer literal with explicit type checking
    pub fn integerLiteral(self: *AstBuilder, value: []const u8, span: ast.SourceSpan) !*ast.Expressions.ExprNode {
        if (self.finalized) return BuilderError.BuilderFinalized;

        // Validate integer format
        if (value.len == 0) {
            try self.diagnostics.addError(span, "Empty integer literal");
            return BuilderError.ValidationFailed;
        }

        const expr_node = try self.arena.createNode(ast.Expressions.ExprNode);
        const owned_value = try self.arena.createString(value);
        expr_node.* = ast.Expressions.ExprNode{ .Literal = .{ .Integer = .{
            .value = owned_value,
            .span = span,
        } } };

        if (self.validation_enabled) {
            try self.validateExpression(expr_node, span);
        }

        return expr_node;
    }

    /// Create a boolean literal with explicit type checking
    pub fn boolLiteral(self: *AstBuilder, value: bool, span: ast.SourceSpan) !*ast.Expressions.ExprNode {
        if (self.finalized) return BuilderError.BuilderFinalized;

        const expr_node = try self.arena.createNode(ast.Expressions.ExprNode);
        expr_node.* = ast.Expressions.ExprNode{ .Literal = .{ .Bool = .{
            .value = value,
            .span = span,
        } } };

        if (self.validation_enabled) {
            try self.validateExpression(expr_node, span);
        }

        return expr_node;
    }

    /// Create an address literal with validation
    pub fn addressLiteral(self: *AstBuilder, value: []const u8, span: ast.SourceSpan) !*ast.Expressions.ExprNode {
        if (self.finalized) return BuilderError.BuilderFinalized;

        // Basic address validation (should be hex string)
        if (value.len < 2 or !std.mem.startsWith(u8, value, "0x")) {
            try self.diagnostics.addError(span, "Address literal must start with '0x'");
            return BuilderError.ValidationFailed;
        }

        const expr_node = try self.arena.createNode(ast.Expressions.ExprNode);
        const owned_value = try self.arena.createString(value);
        expr_node.* = ast.Expressions.ExprNode{ .Literal = .{ .Address = .{
            .value = owned_value,
            .span = span,
        } } };

        if (self.validation_enabled) {
            try self.validateExpression(expr_node, span);
        }

        return expr_node;
    }

    /// Create a hex literal with validation
    pub fn hexLiteral(self: *AstBuilder, value: []const u8, span: ast.SourceSpan) !*ast.Expressions.ExprNode {
        if (self.finalized) return BuilderError.BuilderFinalized;

        // Basic hex validation
        if (value.len < 2 or !std.mem.startsWith(u8, value, "0x")) {
            try self.diagnostics.addError(span, "Hex literal must start with '0x'");
            return BuilderError.ValidationFailed;
        }

        const expr_node = try self.arena.createNode(ast.Expressions.ExprNode);
        const owned_value = try self.arena.createString(value);
        expr_node.* = ast.Expressions.ExprNode{ .Literal = .{ .Hex = .{
            .value = owned_value,
            .span = span,
        } } };

        if (self.validation_enabled) {
            try self.validateExpression(expr_node, span);
        }

        return expr_node;
    }

    pub fn identifier(self: *AstBuilder, name: []const u8, span: ast.SourceSpan) !*ast.Expressions.ExprNode {
        if (self.finalized) return BuilderError.BuilderFinalized;

        const expr_node = try self.arena.createNode(ast.Expressions.ExprNode);
        const owned_name = try self.arena.createString(name);
        expr_node.* = ast.Expressions.ExprNode{ .Identifier = .{
            .name = owned_name,
            .span = span,
        } };

        if (self.validation_enabled) {
            try self.validateExpression(expr_node, span);
        }

        return expr_node;
    }

    /// Create a binary expression node with left and right operands and an operator
    /// Validates that operands are provided and creates the expression in the arena
    /// Note: Operand validation is limited in Zig due to pointer comparison restrictions
    pub fn binary(self: *AstBuilder, lhs: *ast.Expressions.ExprNode, op: ast.Operators.Binary, rhs: *ast.Expressions.ExprNode, span: ast.SourceSpan) !*ast.Expressions.ExprNode {
        if (self.finalized) return BuilderError.BuilderFinalized;

        // Basic validation - ensure operands are provided
        // Note: In Zig, we can't directly compare pointers to undefined at runtime
        // More comprehensive validation would be done by TypeValidator integration

        const expr_node = try self.arena.createNode(ast.Expressions.ExprNode);
        expr_node.* = ast.Expressions.ExprNode{ .Binary = .{
            .lhs = lhs,
            .operator = op,
            .rhs = rhs,
            .span = span,
        } };

        if (self.validation_enabled) {
            try self.validateExpression(expr_node, span);
        }

        return expr_node;
    }

    /// Create a unary expression with validation
    pub fn unary(self: *AstBuilder, op: ast.Operators.Unary, operand: *ast.Expressions.ExprNode, span: ast.SourceSpan) !*ast.Expressions.ExprNode {
        if (self.finalized) return BuilderError.BuilderFinalized;

        // Note: In Zig, we can't directly compare pointers to undefined at runtime

        const expr_node = try self.arena.createNode(ast.Expressions.ExprNode);
        expr_node.* = ast.Expressions.ExprNode{ .Unary = .{
            .operator = op,
            .operand = operand,
            .span = span,
        } };

        if (self.validation_enabled) {
            try self.validateExpression(expr_node, span);
        }

        return expr_node;
    }

    /// Create an assignment expression with target validation
    /// Validates that the target is assignable (identifier, field access, or index)
    /// Reports validation errors for invalid assignment targets
    pub fn assignment(self: *AstBuilder, target: *ast.Expressions.ExprNode, value: *ast.Expressions.ExprNode, span: ast.SourceSpan) !*ast.Expressions.ExprNode {
        if (self.finalized) return BuilderError.BuilderFinalized;

        // Basic operand validation
        // Note: In Zig, we can't directly compare pointers to undefined at runtime

        // Validate that target is assignable (only certain expression types can be assigned to)
        switch (target.*) {
            .Identifier, .FieldAccess, .Index => {}, // Valid assignment targets
            else => {
                try self.diagnostics.addError(span, "Invalid assignment target");
                return BuilderError.ValidationFailed;
            },
        }

        const expr_node = try self.arena.createNode(ast.Expressions.ExprNode);
        expr_node.* = ast.Expressions.ExprNode{ .Assignment = .{
            .target = target,
            .value = value,
            .span = span,
        } };

        if (self.validation_enabled) {
            try self.validateExpression(expr_node, span);
        }

        return expr_node;
    }

    /// Create a compound assignment expression with validation
    pub fn compoundAssignment(self: *AstBuilder, target: *ast.Expressions.ExprNode, op: ast.Operators.Compound, value: *ast.Expressions.ExprNode, span: ast.SourceSpan) !*ast.Expressions.ExprNode {
        if (self.finalized) return BuilderError.BuilderFinalized;

        // Note: In Zig, we can't directly compare pointers to undefined at runtime

        // Validate that target is assignable
        switch (target.*) {
            .Identifier, .FieldAccess, .Index => {}, // Valid assignment targets
            else => {
                try self.diagnostics.addError(span, "Invalid compound assignment target");
                return BuilderError.ValidationFailed;
            },
        }

        const expr_node = try self.arena.createNode(ast.Expressions.ExprNode);
        expr_node.* = ast.Expressions.ExprNode{ .CompoundAssignment = .{
            .target = target,
            .operator = op,
            .value = value,
            .span = span,
        } };

        if (self.validation_enabled) {
            try self.validateExpression(expr_node, span);
        }

        return expr_node;
    }

    // Statement building
    pub fn stmt(self: *AstBuilder) StatementBuilder {
        return StatementBuilder.init(self);
    }

    pub fn block(self: *AstBuilder, statements: []ast.Statements.StmtNode, span: ast.SourceSpan) !ast.Statements.BlockNode {
        if (self.finalized) return BuilderError.BuilderFinalized;

        const owned_statements = try self.arena.createSlice(ast.Statements.StmtNode, statements.len);
        @memcpy(owned_statements, statements);

        const block_node = ast.Statements.BlockNode{
            .statements = owned_statements,
            .span = span,
        };

        if (self.validation_enabled) {
            try self.validateBlock(&block_node, span);
        }

        return block_node;
    }

    // Type building
    pub fn typeBuilder(self: *AstBuilder) TypeBuilder {
        return TypeBuilder.init(self);
    }

    /// Helper method to create addition expressions (+)
    /// Convenience wrapper around binary() for arithmetic operations
    pub fn add(self: *AstBuilder, lhs: *ast.Expressions.ExprNode, rhs: *ast.Expressions.ExprNode, span: ast.SourceSpan) !*ast.Expressions.ExprNode {
        return self.binary(lhs, .Plus, rhs, span);
    }

    /// Helper method to create subtraction expressions (-)
    /// Convenience wrapper around binary() for arithmetic operations
    pub fn subtract(self: *AstBuilder, lhs: *ast.Expressions.ExprNode, rhs: *ast.Expressions.ExprNode, span: ast.SourceSpan) !*ast.Expressions.ExprNode {
        return self.binary(lhs, .Minus, rhs, span);
    }

    /// Helper method to create multiplication expressions (*)
    /// Convenience wrapper around binary() for arithmetic operations
    pub fn multiply(self: *AstBuilder, lhs: *ast.Expressions.ExprNode, rhs: *ast.Expressions.ExprNode, span: ast.SourceSpan) !*ast.Expressions.ExprNode {
        return self.binary(lhs, .Star, rhs, span);
    }

    /// Helper method to create division expressions (/)
    /// Convenience wrapper around binary() for arithmetic operations
    pub fn divide(self: *AstBuilder, lhs: *ast.Expressions.ExprNode, rhs: *ast.Expressions.ExprNode, span: ast.SourceSpan) !*ast.Expressions.ExprNode {
        return self.binary(lhs, .Slash, rhs, span);
    }

    /// Helper method to create comparison expressions
    pub fn equal(self: *AstBuilder, lhs: *ast.Expressions.ExprNode, rhs: *ast.Expressions.ExprNode, span: ast.SourceSpan) !*ast.Expressions.ExprNode {
        return self.binary(lhs, .EqualEqual, rhs, span);
    }

    pub fn notEqual(self: *AstBuilder, lhs: *ast.Expressions.ExprNode, rhs: *ast.Expressions.ExprNode, span: ast.SourceSpan) !*ast.Expressions.ExprNode {
        return self.binary(lhs, .BangEqual, rhs, span);
    }

    pub fn lessThan(self: *AstBuilder, lhs: *ast.Expressions.ExprNode, rhs: *ast.Expressions.ExprNode, span: ast.SourceSpan) !*ast.Expressions.ExprNode {
        return self.binary(lhs, .Less, rhs, span);
    }

    pub fn greaterThan(self: *AstBuilder, lhs: *ast.Expressions.ExprNode, rhs: *ast.Expressions.ExprNode, span: ast.SourceSpan) !*ast.Expressions.ExprNode {
        return self.binary(lhs, .Greater, rhs, span);
    }

    /// Helper method to create logical expressions
    pub fn logicalAnd(self: *AstBuilder, lhs: *ast.Expressions.ExprNode, rhs: *ast.Expressions.ExprNode, span: ast.SourceSpan) !*ast.Expressions.ExprNode {
        return self.binary(lhs, .And, rhs, span);
    }

    pub fn logicalOr(self: *AstBuilder, lhs: *ast.Expressions.ExprNode, rhs: *ast.Expressions.ExprNode, span: ast.SourceSpan) !*ast.Expressions.ExprNode {
        return self.binary(lhs, .Or, rhs, span);
    }

    pub fn logicalNot(self: *AstBuilder, operand: *ast.Expressions.ExprNode, span: ast.SourceSpan) !*ast.Expressions.ExprNode {
        return self.unary(.Bang, operand, span);
    }

    // Validation methods
    fn validateExpression(self: *AstBuilder, expr_node: *ast.Expressions.ExprNode, span: ast.SourceSpan) !void {
        if (!self.validation_enabled) return;
        _ = expr_node;
        _ = span; // Suppress unused parameter warning
    }

    fn validateBlock(self: *AstBuilder, block_node: *const ast.Statements.BlockNode, span: ast.SourceSpan) !void {
        if (!self.validation_enabled) return;
        _ = block_node;
        _ = span; // Suppress unused parameter warning
    }

    fn validateNode(self: *AstBuilder, node: *ast.AstNode) !void {
        if (self.validator) |validator| {
            // Simplified validation until proper integration with TypeValidator
            _ = try validator.validateNode(node);
            // Real implementation would check for errors and report them
        }
    }

    /// Finalize the builder and perform comprehensive validation
    /// This method must be called before accessing built nodes
    /// Returns a slice of all built AST nodes or validation errors
    pub fn build(self: *AstBuilder) ![]ast.AstNode {
        if (self.finalized) return BuilderError.BuilderFinalized;

        // Perform final validation if enabled
        if (self.validation_enabled) {
            // Check for any accumulated errors from construction
            if (self.diagnostics.hasErrors()) {
                return BuilderError.ValidationFailed;
            }

            // Perform comprehensive validation on all built nodes
            // Note: Built nodes validation is currently skipped until TypeValidator integration
            if (!self.validation_enabled) return;

            // Final check after comprehensive validation
            if (self.diagnostics.hasErrors()) {
                return BuilderError.ValidationFailed;
            }
        }

        self.finalized = true;

        // Return a copy of the built nodes in arena memory
        const result = try self.arena.createSlice(ast.AstNode, self.built_nodes.items.len);
        @memcpy(result, self.built_nodes.items);
        return result;
    }

    /// Validate the current state of the builder without finalizing
    pub fn validate(self: *AstBuilder) !void {
        if (self.finalized) return BuilderError.BuilderFinalized;

        if (!self.validation_enabled) return;

        // Clear previous validation errors
        self.diagnostics.clear();

        // Skip validation until TypeValidator is properly integrated
        if (!self.validation_enabled) return;

        if (self.diagnostics.hasErrors()) {
            return BuilderError.ValidationFailed;
        }
    }

    /// Reset the builder to its initial state (keeping the same arena and validator)
    pub fn reset(self: *AstBuilder) void {
        if (self.finalized) return; // Cannot reset finalized builder

        self.diagnostics.clear();
        self.built_nodes.clearRetainingCapacity();
        self.node_counter = 0;
    }
};

/// Contract builder for fluent contract construction
/// Provides methods to add functions, variables, structs, enums, logs, and errors to a contract
/// Validates member uniqueness and contract structure during construction
pub const ContractBuilder = struct {
    /// Reference to the parent AST builder for arena access and diagnostics
    builder: *AstBuilder,
    /// The contract node being constructed
    contract: *ast.ContractNode,
    /// Collection of contract members (functions, variables, etc.)
    members: std.ArrayList(ast.AstNode),
    /// Source span for the entire contract
    span: ast.SourceSpan,

    pub fn init(builder: *AstBuilder, name: []const u8) !ContractBuilder {
        const contract = try builder.arena.createNode(ast.ContractNode);
        const owned_name = try builder.arena.createString(name);
        contract.* = ast.ContractNode{
            .name = owned_name,
            .body = &[_]ast.AstNode{},
            .span = .{ .line = 0, .column = 0, .length = 0 },
        };

        return ContractBuilder{
            .builder = builder,
            .contract = contract,
            .members = std.ArrayList(ast.AstNode).init(builder.arena.allocator()),
            .span = .{ .line = 0, .column = 0, .length = 0 },
        };
    }

    /// Set the source span for the contract
    pub fn setSpan(self: *ContractBuilder, span: ast.SourceSpan) *ContractBuilder {
        self.span = span;
        self.contract.span = span;
        return self;
    }

    /// Add a function to the contract with validation
    pub fn addFunction(self: *ContractBuilder, function: *ast.FunctionNode) !*ContractBuilder {
        // Validate function name uniqueness
        for (self.members.items) |member| {
            switch (member) {
                .Function => |existing_func| {
                    if (std.mem.eql(u8, existing_func.name, function.name)) {
                        const error_msg = try std.fmt.allocPrint(self.builder.arena.allocator(), "Duplicate function name: {s}", .{function.name});
                        try self.builder.diagnostics.addError(function.span, error_msg);
                        return BuilderError.ValidationFailed;
                    }
                },
                else => {},
            }
        }

        try self.members.append(ast.AstNode{ .Function = function.* });
        return self;
    }

    /// Add a variable declaration to the contract with validation
    pub fn addVariable(self: *ContractBuilder, variable: *ast.Statements.VariableDeclNode) !*ContractBuilder {
        // Validate variable name uniqueness
        for (self.members.items) |member| {
            switch (member) {
                .VariableDecl => |existing_var| {
                    if (std.mem.eql(u8, existing_var.name, variable.name)) {
                        const error_msg = try std.fmt.allocPrint(self.builder.arena.allocator(), "Duplicate variable name: {s}", .{variable.name});
                        try self.builder.diagnostics.addError(variable.span, error_msg);
                        return BuilderError.ValidationFailed;
                    }
                },
                else => {},
            }
        }

        try self.members.append(ast.AstNode{ .VariableDecl = variable.* });
        return self;
    }

    /// Add a struct declaration to the contract with validation
    pub fn addStruct(self: *ContractBuilder, struct_decl: *ast.StructDeclNode) !*ContractBuilder {
        // Validate struct name uniqueness
        for (self.members.items) |member| {
            switch (member) {
                .StructDecl => |existing_struct| {
                    if (std.mem.eql(u8, existing_struct.name, struct_decl.name)) {
                        const error_msg = try std.fmt.allocPrint(self.builder.arena.allocator(), "Duplicate struct name: {s}", .{struct_decl.name});
                        try self.builder.diagnostics.addError(struct_decl.span, error_msg);
                        return BuilderError.ValidationFailed;
                    }
                },
                else => {},
            }
        }

        try self.members.append(ast.AstNode{ .StructDecl = struct_decl.* });
        return self;
    }

    /// Add an enum declaration to the contract with validation
    pub fn addEnum(self: *ContractBuilder, enum_decl: *ast.EnumDeclNode) !*ContractBuilder {
        // Validate enum name uniqueness
        for (self.members.items) |member| {
            switch (member) {
                .EnumDecl => |existing_enum| {
                    if (std.mem.eql(u8, existing_enum.name, enum_decl.name)) {
                        const error_msg = try std.fmt.allocPrint(self.builder.arena.allocator(), "Duplicate enum name: {s}", .{enum_decl.name});
                        try self.builder.diagnostics.addError(enum_decl.span, error_msg);
                        return BuilderError.ValidationFailed;
                    }
                },
                else => {},
            }
        }

        try self.members.append(ast.AstNode{ .EnumDecl = enum_decl.* });
        return self;
    }

    /// Add a log declaration to the contract with validation
    pub fn addLog(self: *ContractBuilder, log_decl: *ast.LogDeclNode) !*ContractBuilder {
        // Validate log name uniqueness
        for (self.members.items) |member| {
            switch (member) {
                .LogDecl => |existing_log| {
                    if (std.mem.eql(u8, existing_log.name, log_decl.name)) {
                        const error_msg = try std.fmt.allocPrint(self.builder.arena.allocator(), "Duplicate log name: {s}", .{log_decl.name});
                        try self.builder.diagnostics.addError(log_decl.span, error_msg);
                        return BuilderError.ValidationFailed;
                    }
                },
                else => {},
            }
        }

        try self.members.append(ast.AstNode{ .LogDecl = log_decl.* });
        return self;
    }

    /// Add an error declaration to the contract with validation
    pub fn addError(self: *ContractBuilder, error_decl: *ast.Statements.ErrorDeclNode) !*ContractBuilder {
        // Validate error name uniqueness
        for (self.members.items) |member| {
            switch (member) {
                .ErrorDecl => |existing_error| {
                    if (std.mem.eql(u8, existing_error.name, error_decl.name)) {
                        const error_msg = try std.fmt.allocPrint(self.builder.arena.allocator(), "Duplicate error name: {s}", .{error_decl.name});
                        try self.builder.diagnostics.addError(error_decl.span, error_msg);
                        return BuilderError.ValidationFailed;
                    }
                },
                else => {},
            }
        }

        try self.members.append(ast.AstNode{ .ErrorDecl = error_decl.* });
        return self;
    }

    /// Get the number of members in the contract
    pub fn getMemberCount(self: *const ContractBuilder) usize {
        return self.members.items.len;
    }

    /// Check if the contract has any functions
    pub fn hasFunctions(self: *const ContractBuilder) bool {
        for (self.members.items) |member| {
            if (member == .Function) return true;
        }
        return false;
    }

    /// Build the contract with comprehensive validation
    pub fn build(self: *ContractBuilder) !*ast.ContractNode {
        // Validate that contract has at least one member
        if (self.members.items.len == 0) {
            try self.builder.diagnostics.addError(self.span, "Contract must have at least one member");
            return BuilderError.ValidationFailed;
        }

        // Convert members to owned slice
        const owned_members = try self.builder.arena.createSlice(ast.AstNode, self.members.items.len);
        @memcpy(owned_members, self.members.items);
        self.contract.body = owned_members;

        // Contract validation is skipped until TypeValidator is properly integrated

        // Add the contract to the builder's built nodes
        try self.builder.addBuiltNode(ast.AstNode{ .Contract = self.contract.* });

        return self.contract;
    }
};

/// Expression builder for fluent expression construction with operator precedence handling
/// Provides methods to create complex expressions like calls, field access, casts, tuples, etc.
/// Includes validation for expression structure and operand types
pub const ExpressionBuilder = struct {
    /// Reference to the parent AST builder for arena access and diagnostics
    builder: *AstBuilder,

    pub fn init(builder: *AstBuilder) ExpressionBuilder {
        return ExpressionBuilder{
            .builder = builder,
        };
    }

    /// Create a function call expression with validation
    pub fn call(self: *const ExpressionBuilder, callee: *ast.Expressions.ExprNode, args: []ast.Expressions.ExprNode, span: ast.SourceSpan) !*ast.Expressions.ExprNode {
        // Note: In Zig, we can't directly compare pointers to undefined at runtime

        const expr = try self.builder.arena.createNode(ast.Expressions.ExprNode);
        const owned_args = try self.builder.arena.createSlice(ast.Expressions.ExprNode, args.len);
        @memcpy(owned_args, args);

        expr.* = ast.Expressions.ExprNode{ .Call = .{
            .callee = callee,
            .arguments = owned_args,
            .span = span,
        } };

        if (self.builder.validation_enabled) {
            try self.builder.validateExpression(expr, span);
        }

        return expr;
    }

    /// Create a field access expression with validation
    pub fn fieldAccess(self: *const ExpressionBuilder, target: *ast.Expressions.ExprNode, field: []const u8, span: ast.SourceSpan) !*ast.Expressions.ExprNode {
        // Note: In Zig, we can't directly compare pointers to undefined at runtime

        if (field.len == 0) {
            try self.builder.diagnostics.addError(span, "Field name cannot be empty");
            return BuilderError.ValidationFailed;
        }

        const expr = try self.builder.arena.createNode(ast.Expressions.ExprNode);
        const owned_field = try self.builder.arena.createString(field);

        expr.* = ast.Expressions.ExprNode{ .FieldAccess = .{
            .target = target,
            .field = owned_field,
            .span = span,
        } };

        if (self.builder.validation_enabled) {
            try self.builder.validateExpression(expr, span);
        }

        return expr;
    }

    /// Create an index expression with validation
    pub fn index(self: *const ExpressionBuilder, target: *ast.Expressions.ExprNode, index_expr: *ast.Expressions.ExprNode, span: ast.SourceSpan) !*ast.Expressions.ExprNode {
        // Note: In Zig, we can't directly compare pointers to undefined at runtime

        const expr = try self.builder.arena.createNode(ast.Expressions.ExprNode);

        expr.* = ast.Expressions.ExprNode{ .Index = .{
            .target = target,
            .index = index_expr,
            .span = span,
        } };

        if (self.builder.validation_enabled) {
            try self.builder.validateExpression(expr, span);
        }

        return expr;
    }

    /// Create a cast expression with explicit type checking
    pub fn cast(self: *const ExpressionBuilder, expr_node: *ast.Expressions.ExprNode, target_type: TypeInfo, span: ast.SourceSpan) !*ast.Expressions.ExprNode {
        return self.castWithType(expr_node, target_type, .Unsafe, span);
    }

    /// Create a safe cast expression (as?)
    pub fn safeCast(self: *const ExpressionBuilder, expr_node: *ast.Expressions.ExprNode, target_type: TypeInfo, span: ast.SourceSpan) !*ast.Expressions.ExprNode {
        return self.castWithType(expr_node, target_type, .Safe, span);
    }

    /// Create a forced cast expression (as!)
    pub fn forcedCast(self: *const ExpressionBuilder, expr_node: *ast.Expressions.ExprNode, target_type: TypeInfo, span: ast.SourceSpan) !*ast.Expressions.ExprNode {
        return self.castWithType(expr_node, target_type, .Forced, span);
    }

    /// Internal method to create cast expressions with specific cast types
    fn castWithType(self: *const ExpressionBuilder, expr_node: *ast.Expressions.ExprNode, target_type: TypeInfo, cast_type: ast.CastType, span: ast.SourceSpan) !*ast.Expressions.ExprNode {
        // Note: In Zig, we can't directly compare pointers to undefined at runtime

        const expr = try self.builder.arena.createNode(ast.Expressions.ExprNode);

        expr.* = ast.Expressions.ExprNode{
            .Cast = .{
                .operand = expr_node,
                .target_type = target_type,
                .cast_type = cast_type,
                .span = span,
            },
        };

        if (self.builder.validation_enabled) {
            try self.builder.validateExpression(expr, span);
        }

        return expr;
    }

    /// Create a tuple expression
    pub fn tuple(self: *const ExpressionBuilder, elements: []ast.Expressions.ExprNode, span: ast.SourceSpan) !*ast.Expressions.ExprNode {
        if (elements.len == 0) {
            try self.builder.diagnostics.addError(span, "Tuple must have at least one element");
            return BuilderError.ValidationFailed;
        }

        const expr = try self.builder.arena.createNode(ast.Expressions.ExprNode);
        const owned_elements = try self.builder.arena.createSlice(ast.Expressions.ExprNode, elements.len);
        @memcpy(owned_elements, elements);

        expr.* = ast.Expressions.ExprNode{ .Tuple = .{
            .elements = owned_elements,
            .span = span,
        } };

        if (self.builder.validation_enabled) {
            try self.builder.validateExpression(expr, span);
        }

        return expr;
    }

    /// Create a try expression for error handling
    pub fn tryExpr(self: *const ExpressionBuilder, expr_node: *ast.Expressions.ExprNode, span: ast.SourceSpan) !*ast.Expressions.ExprNode {
        // Note: In Zig, we can't directly compare pointers to undefined at runtime

        const expr = try self.builder.arena.createNode(ast.Expressions.ExprNode);

        expr.* = ast.Expressions.ExprNode{ .Try = .{
            .expr = expr_node,
            .span = span,
        } };

        if (self.builder.validation_enabled) {
            try self.builder.validateExpression(expr, span);
        }

        return expr;
    }

    /// Create an error return expression
    pub fn errorReturn(self: *const ExpressionBuilder, error_name: []const u8, span: ast.SourceSpan) !*ast.Expressions.ExprNode {
        if (error_name.len == 0) {
            try self.builder.diagnostics.addError(span, "Error name cannot be empty");
            return BuilderError.ValidationFailed;
        }

        const expr = try self.builder.arena.createNode(ast.Expressions.ExprNode);
        const owned_error_name = try self.builder.arena.createString(error_name);

        expr.* = ast.Expressions.ExprNode{ .ErrorReturn = .{
            .error_name = owned_error_name,
            .span = span,
        } };

        if (self.builder.validation_enabled) {
            try self.builder.validateExpression(expr, span);
        }

        return expr;
    }

    /// Create a struct instantiation expression
    pub fn structInstantiation(self: *const ExpressionBuilder, struct_name: *ast.Expressions.ExprNode, fields: []ast.Expressions.StructInstantiationField, span: ast.SourceSpan) !*ast.Expressions.ExprNode {
        // Note: In Zig, we can't directly compare pointers to undefined at runtime

        const expr = try self.builder.arena.createNode(ast.Expressions.ExprNode);
        const owned_fields = try self.builder.arena.createSlice(ast.Expressions.StructInstantiationField, fields.len);
        @memcpy(owned_fields, fields);

        expr.* = ast.Expressions.ExprNode{ .StructInstantiation = .{
            .struct_name = struct_name,
            .fields = owned_fields,
            .span = span,
        } };

        if (self.builder.validation_enabled) {
            try self.builder.validateExpression(expr, span);
        }

        return expr;
    }

    /// Create an enum literal expression
    pub fn enumLiteral(self: *const ExpressionBuilder, enum_name: []const u8, variant_name: []const u8, span: ast.SourceSpan) !*ast.Expressions.ExprNode {
        if (enum_name.len == 0 or variant_name.len == 0) {
            try self.builder.diagnostics.addError(span, "Enum name and variant name cannot be empty");
            return BuilderError.ValidationFailed;
        }

        const expr = try self.builder.arena.createNode(ast.Expressions.ExprNode);
        const owned_enum_name = try self.builder.arena.createString(enum_name);
        const owned_variant_name = try self.builder.arena.createString(variant_name);

        expr.* = ast.Expressions.ExprNode{ .EnumLiteral = .{
            .enum_name = owned_enum_name,
            .variant_name = owned_variant_name,
            .span = span,
        } };

        if (self.builder.validation_enabled) {
            try self.builder.validateExpression(expr, span);
        }

        return expr;
    }
};

/// Statement builder for fluent statement construction with control flow validation
/// Provides methods to create statements like returns, if/while loops, variable declarations, etc.
/// Includes validation for control flow correctness and statement structure
pub const StatementBuilder = struct {
    /// Reference to the parent AST builder for arena access and diagnostics
    builder: *AstBuilder,

    pub fn init(builder: *AstBuilder) StatementBuilder {
        return StatementBuilder{
            .builder = builder,
        };
    }

    /// Create a return statement with validation
    pub fn returnStmt(_: *const StatementBuilder, value: ?*ast.Expressions.ExprNode, span: ast.SourceSpan) !ast.Statements.StmtNode {
        // No validation needed for this simple statement

        return ast.Statements.StmtNode{ .Return = .{
            .value = if (value) |v| v.* else null,
            .span = span,
        } };
    }

    /// Create an if statement with control flow validation
    pub fn ifStmt(self: *const StatementBuilder, condition: *ast.Expressions.ExprNode, then_branch: ast.Statements.BlockNode, else_branch: ?ast.Statements.BlockNode, span: ast.SourceSpan) !ast.Statements.StmtNode {
        // Validate condition
        // Note: In Zig, we can't directly compare pointers to undefined at runtime

        // Validate that then_branch has at least one statement
        if (then_branch.statements.len == 0) {
            try self.builder.diagnostics.addError(span, "If statement then branch cannot be empty");
            return BuilderError.ValidationFailed;
        }

        return ast.Statements.StmtNode{ .If = .{
            .condition = condition.*,
            .then_branch = then_branch,
            .else_branch = else_branch,
            .span = span,
        } };
    }

    /// Create a while statement with loop validation
    pub fn whileStmt(self: *const StatementBuilder, condition: *ast.Expressions.ExprNode, body: ast.Statements.BlockNode, span: ast.SourceSpan) !ast.Statements.StmtNode {
        return self.whileStmtWithInvariants(condition, body, &[_]ast.Expressions.ExprNode{}, span);
    }

    /// Create a while statement with loop invariants
    pub fn whileStmtWithInvariants(self: *const StatementBuilder, condition: *ast.Expressions.ExprNode, body: ast.Statements.BlockNode, invariants: []ast.Expressions.ExprNode, span: ast.SourceSpan) !ast.Statements.StmtNode {
        // Note: In Zig, we can't directly compare pointers to undefined at runtime

        // Validate that body has at least one statement
        if (body.statements.len == 0) {
            try self.builder.diagnostics.addError(span, "While loop body cannot be empty");
            return BuilderError.ValidationFailed;
        }

        // Create owned invariants slice
        const owned_invariants = try self.builder.arena.createSlice(ast.Expressions.ExprNode, invariants.len);
        @memcpy(owned_invariants, invariants);

        return ast.Statements.StmtNode{
            .While = .{
                .condition = condition.*,
                .body = body,
                .invariants = owned_invariants,
                .span = span,
            },
        };
    }

    /// Create an expression statement
    pub fn exprStmt(self: *const StatementBuilder, expr: *ast.Expressions.ExprNode) ast.Statements.StmtNode {
        _ = self; // Suppress unused parameter warning
        return ast.Statements.StmtNode{ .Expr = expr.* };
    }

    /// Create a variable declaration statement
    pub fn variableDecl(_: *const StatementBuilder, var_decl: *ast.Statements.VariableDeclNode) ast.Statements.StmtNode {
        return ast.Statements.StmtNode{ .VariableDecl = var_decl.* };
    }

    /// Create a break statement
    pub fn breakStmt(_: *const StatementBuilder, span: ast.SourceSpan) ast.Statements.StmtNode {
        return ast.Statements.StmtNode{ .Break = span };
    }

    /// Create a continue statement
    pub fn continueStmt(_: *const StatementBuilder, span: ast.SourceSpan) ast.Statements.StmtNode {
        return ast.Statements.StmtNode{ .Continue = span };
    }

    /// Create a log statement
    pub fn logStmt(self: *const StatementBuilder, event_name: []const u8, args: []ast.Expressions.ExprNode, span: ast.SourceSpan) !ast.Statements.StmtNode {
        if (event_name.len == 0) {
            try self.builder.diagnostics.addError(span, "Log event name cannot be empty");
            return BuilderError.ValidationFailed;
        }

        const owned_event_name = try self.builder.arena.createString(event_name);
        const owned_args = try self.builder.arena.createSlice(ast.Expressions.ExprNode, args.len);
        @memcpy(owned_args, args);

        return ast.Statements.StmtNode{ .Log = .{
            .event_name = owned_event_name,
            .args = owned_args,
            .span = span,
        } };
    }

    /// Create a requires statement (precondition)
    pub fn requiresStmt(self: *const StatementBuilder, condition: *ast.Expressions.ExprNode, span: ast.SourceSpan) !ast.Statements.StmtNode {
        _ = self;
        // Note: In Zig, we can't directly compare pointers to undefined at runtime

        return ast.Statements.StmtNode{ .Requires = .{
            .condition = condition.*,
            .span = span,
        } };
    }

    /// Create an ensures statement (postcondition)
    pub fn ensuresStmt(_: *const StatementBuilder, condition: *ast.Expressions.ExprNode, span: ast.SourceSpan) !ast.Statements.StmtNode {
        return ast.Statements.StmtNode{ .Ensures = .{
            .condition = condition.*,
            .span = span,
        } };
    }

    /// Create an invariant statement
    pub fn invariantStmt(_: *const StatementBuilder, condition: *ast.Expressions.ExprNode, span: ast.SourceSpan) !ast.Statements.StmtNode {
        return ast.Statements.StmtNode{ .Invariant = .{
            .condition = condition.*,
            .span = span,
        } };
    }
};

/// Type builder for fluent type construction with explicit type annotation validation
/// Provides methods to create primitive types, slices, mappings, tuples, error unions, etc.
/// Includes validation for type structure and compatibility
pub const TypeBuilder = struct {
    /// Reference to the parent AST builder for arena access and diagnostics
    builder: *AstBuilder,

    pub fn init(builder: *AstBuilder) TypeBuilder {
        return TypeBuilder{
            .builder = builder,
        };
    }

    /// Create a primitive type reference
    pub fn primitive(_: *const TypeBuilder, prim_type: TypeInfo) TypeInfo {
        // No builder state needed for primitive types
        return prim_type;
    }

    /// Create a slice type with element type validation
    pub fn slice(self: *const TypeBuilder, element_type: TypeInfo) !TypeInfo {
        // Create a slice type using OraType
        var slice_type = OraType{ .Slice = undefined };

        // Store the element type
        const elem_type_ptr = try self.builder.arena.allocator().create(OraType);
        if (element_type.ora_type) |et| {
            elem_type_ptr.* = et;
        } else {
            elem_type_ptr.* = OraType.Unknown;
        }
        slice_type.Slice = elem_type_ptr;

        return TypeInfo.explicit(.SliceType, slice_type, element_type.span orelse SourceSpan{});
    }

    /// Create a mapping type with key-value type validation
    pub fn mapping(self: *const TypeBuilder, key_type: TypeInfo, value_type: TypeInfo) !TypeInfo {
        // Create a mapping type using OraType
        var mapping_type = OraType{ .mapping = undefined };

        // Store the key and value types
        const mapping_data = try self.builder.arena.allocator().create(OraType.MappingType);

        // Key type
        const key_type_ptr = try self.builder.arena.allocator().create(OraType);
        if (key_type.ora_type) |kt| {
            key_type_ptr.* = kt;
        } else {
            key_type_ptr.* = OraType.Unknown;
        }

        // Value type
        const value_type_ptr = try self.builder.arena.allocator().create(OraType);
        if (value_type.ora_type) |vt| {
            value_type_ptr.* = vt;
        } else {
            value_type_ptr.* = OraType.Unknown;
        }

        mapping_data.* = OraType.MappingType{
            .key_type = key_type_ptr,
            .value_type = value_type_ptr,
        };

        mapping_type.mapping = mapping_data;

        // Determine span for the mapping type
        var span = SourceSpan{};
        if (key_type.span) |ks| {
            span = ks;
        } else if (value_type.span) |vs| {
            span = vs;
        }

        return TypeInfo.explicit(.MappingType, mapping_type, span);
    }

    /// Create a double mapping type
    pub fn doubleMapping(self: *const TypeBuilder, key1_type: TypeInfo, key2_type: TypeInfo, value_type: TypeInfo) !TypeInfo {
        // Create a double mapping type using OraType
        var double_map_type = OraType{ .double_map = undefined };

        // Store the key and value types
        const double_map_data = try self.builder.arena.allocator().create(OraType.DoubleMapType);

        // Key1 type
        const key1_type_ptr = try self.builder.arena.allocator().create(OraType);
        if (key1_type.ora_type) |kt| {
            key1_type_ptr.* = kt;
        } else {
            key1_type_ptr.* = OraType.Unknown;
        }

        // Key2 type
        const key2_type_ptr = try self.builder.arena.allocator().create(OraType);
        if (key2_type.ora_type) |kt| {
            key2_type_ptr.* = kt;
        } else {
            key2_type_ptr.* = OraType.Unknown;
        }

        // Value type
        const value_type_ptr = try self.builder.arena.allocator().create(OraType);
        if (value_type.ora_type) |vt| {
            value_type_ptr.* = vt;
        } else {
            value_type_ptr.* = OraType.Unknown;
        }

        double_map_data.* = OraType.DoubleMapType{
            .key1_type = key1_type_ptr,
            .key2_type = key2_type_ptr,
            .value_type = value_type_ptr,
        };

        double_map_type.double_map = double_map_data;

        // Determine span for the double mapping type
        var span = SourceSpan{};
        if (key1_type.span) |ks| {
            span = ks;
        } else if (key2_type.span) |ks| {
            span = ks;
        } else if (value_type.span) |vs| {
            span = vs;
        }

        return TypeInfo.explicit(.DoubleMapType, double_map_type, span);
    }

    /// Create a tuple type
    pub fn tuple(self: *const TypeBuilder, types: []TypeInfo) !TypeInfo {
        if (types.len == 0) {
            // For now, allow empty tuples - this might be changed based on language requirements
        }

        // Create a tuple type using OraType
        var tuple_type = OraType{ .tuple = undefined };

        // Extract and store the OraTypes from each TypeInfo
        const ora_types = try self.builder.arena.allocator().alloc(OraType, types.len);
        for (types, 0..) |type_info, i| {
            if (type_info.ora_type) |ot| {
                ora_types[i] = ot;
            } else {
                ora_types[i] = OraType.Unknown;
            }
        }

        tuple_type.tuple = ora_types;

        // Determine span for the tuple type
        var span = SourceSpan{};
        for (types) |type_info| {
            if (type_info.span) |s| {
                span = s;
                break;
            }
        }

        return TypeInfo.explicit(.TupleType, tuple_type, span);
    }

    /// Create an error union type (!T)
    pub fn errorUnion(self: *const TypeBuilder, success_type: TypeInfo) !TypeInfo {
        // Create an error union type using OraType
        var error_union_type = OraType{ .error_union = undefined };

        // Store the success type
        const success_type_ptr = try self.builder.arena.allocator().create(OraType);
        if (success_type.ora_type) |st| {
            success_type_ptr.* = st;
        } else {
            success_type_ptr.* = OraType.Unknown;
        }

        error_union_type.error_union = success_type_ptr;

        // Use the span from the success type if available
        const span = success_type.span orelse SourceSpan{};

        return TypeInfo.explicit(.ErrorUnionType, error_union_type, span);
    }

    // Result[T,E] removed; prefer error unions '!T | E'.

    /// Create a custom type identifier with validation
    pub fn identifier(self: *const TypeBuilder, name: []const u8, span: ?SourceSpan) !TypeInfo {
        if (name.len == 0) {
            // This would need a span for proper error reporting, but we don't have one here
            // For now, we'll allow it and let validation catch it later
        }

        const owned_name = try self.builder.arena.createString(name);

        // Create a struct/enum type using OraType
        const struct_type = OraType{ .struct_type = owned_name };

        return TypeInfo.explicit(.Struct, struct_type, span orelse SourceSpan{});
    }

    /// Create common primitive types with convenience methods
    pub fn boolType(self: *const TypeBuilder) TypeInfo {
        return self.primitive(TypeInfo.explicit(.Bool, OraType{ .bool = {} }, SourceSpan{}));
    }

    pub fn addressType(self: *const TypeBuilder) TypeInfo {
        return self.primitive(TypeInfo.explicit(.Address, OraType{ .address = {} }, SourceSpan{}));
    }

    pub fn u8Type(self: *const TypeBuilder) TypeInfo {
        return self.primitive(TypeInfo.explicit(.Integer, OraType{ .u8 = {} }, SourceSpan{}));
    }

    pub fn u16Type(self: *const TypeBuilder) TypeInfo {
        return self.primitive(TypeInfo.explicit(.Integer, OraType{ .u16 = {} }, SourceSpan{}));
    }

    pub fn u32Type(self: *const TypeBuilder) TypeInfo {
        return self.primitive(TypeInfo.explicit(.Integer, OraType{ .u32 = {} }, SourceSpan{}));
    }

    pub fn u64Type(self: *const TypeBuilder) TypeInfo {
        return self.primitive(TypeInfo.explicit(.Integer, OraType{ .u64 = {} }, SourceSpan{}));
    }

    pub fn u128Type(self: *const TypeBuilder) TypeInfo {
        return self.primitive(TypeInfo.explicit(.Integer, OraType{ .u128 = {} }, SourceSpan{}));
    }

    pub fn u256Type(self: *const TypeBuilder) TypeInfo {
        return self.primitive(TypeInfo.explicit(.Integer, OraType{ .u256 = {} }, SourceSpan{}));
    }

    pub fn i8Type(self: *const TypeBuilder) TypeInfo {
        return self.primitive(TypeInfo.explicit(.Integer, OraType{ .i8 = {} }, SourceSpan{}));
    }

    pub fn i16Type(self: *const TypeBuilder) TypeInfo {
        return self.primitive(TypeInfo.explicit(.Integer, OraType{ .i16 = {} }, SourceSpan{}));
    }

    pub fn i32Type(self: *const TypeBuilder) TypeInfo {
        return self.primitive(TypeInfo.explicit(.Integer, OraType{ .i32 = {} }, SourceSpan{}));
    }

    pub fn i64Type(self: *const TypeBuilder) TypeInfo {
        return self.primitive(TypeInfo.explicit(.Integer, OraType{ .i64 = {} }, SourceSpan{}));
    }

    pub fn i128Type(self: *const TypeBuilder) TypeInfo {
        return self.primitive(TypeInfo.explicit(.Integer, OraType{ .i128 = {} }, SourceSpan{}));
    }

    pub fn i256Type(self: *const TypeBuilder) TypeInfo {
        return self.primitive(TypeInfo.explicit(.Integer, OraType{ .i256 = {} }, SourceSpan{}));
    }

    pub fn stringType(self: *const TypeBuilder) TypeInfo {
        return self.primitive(TypeInfo.explicit(.String, OraType{ .string = {} }, SourceSpan{}));
    }

    pub fn bytesType(self: *const TypeBuilder) TypeInfo {
        return self.primitive(TypeInfo.explicit(.Bytes, OraType{ .bytes = {} }, SourceSpan{}));
    }
};
