const std = @import("std");
const c = @import("c.zig").c;
const lib = @import("ora_lib");

/// Comprehensive error handling and validation system for MLIR lowering
pub const ErrorHandler = struct {
    allocator: std.mem.Allocator,
    errors: std.ArrayList(LoweringError),
    warnings: std.ArrayList(LoweringWarning),
    context_stack: std.ArrayList(ErrorContext),

    pub fn init(allocator: std.mem.Allocator) ErrorHandler {
        return .{
            .allocator = allocator,
            .errors = std.ArrayList(LoweringError).init(allocator),
            .warnings = std.ArrayList(LoweringWarning).init(allocator),
            .context_stack = std.ArrayList(ErrorContext).init(allocator),
        };
    }

    pub fn deinit(self: *ErrorHandler) void {
        self.errors.deinit();
        self.warnings.deinit();
        self.context_stack.deinit();
    }

    /// Push an error context onto the stack
    pub fn pushContext(self: *ErrorHandler, context: ErrorContext) !void {
        try self.context_stack.append(context);
    }

    /// Pop the current error context
    pub fn popContext(self: *ErrorHandler) void {
        if (self.context_stack.items.len > 0) {
            _ = self.context_stack.pop();
        }
    }

    /// Report an error with source location information
    pub fn reportError(self: *ErrorHandler, error_type: ErrorType, span: ?lib.ast.SourceSpan, message: []const u8, suggestion: ?[]const u8) !void {
        const error_info = LoweringError{
            .error_type = error_type,
            .span = span,
            .message = try self.allocator.dupe(u8, message),
            .suggestion = if (suggestion) |s| try self.allocator.dupe(u8, s) else null,
            .context = if (self.context_stack.items.len > 0) self.context_stack.items[self.context_stack.items.len - 1] else null,
        };
        try self.errors.append(error_info);
    }

    /// Report a warning with source location information
    pub fn reportWarning(self: *ErrorHandler, warning_type: WarningType, span: ?lib.ast.SourceSpan, message: []const u8) !void {
        const warning_info = LoweringWarning{
            .warning_type = warning_type,
            .span = span,
            .message = try self.allocator.dupe(u8, message),
        };
        try self.warnings.append(warning_info);
    }

    /// Check if there are any errors
    pub fn hasErrors(self: *const ErrorHandler) bool {
        return self.errors.items.len > 0;
    }

    /// Check if there are any warnings
    pub fn hasWarnings(self: *const ErrorHandler) bool {
        return self.warnings.items.len > 0;
    }

    /// Get all errors
    pub fn getErrors(self: *const ErrorHandler) []const LoweringError {
        return self.errors.items;
    }

    /// Get all warnings
    pub fn getWarnings(self: *const ErrorHandler) []const LoweringWarning {
        return self.warnings.items;
    }

    /// Format and print all errors and warnings
    pub fn printDiagnostics(self: *const ErrorHandler, writer: anytype) !void {
        // Print errors
        for (self.errors.items) |err| {
            try self.printError(writer, err);
        }

        // Print warnings
        for (self.warnings.items) |warn| {
            try self.printWarning(writer, warn);
        }
    }

    /// Print a single error with formatting
    fn printError(self: *const ErrorHandler, writer: anytype, err: LoweringError) !void {
        _ = self;
        try writer.writeAll("error: ");
        try writer.writeAll(err.message);

        if (err.span) |span| {
            try writer.print(" at line {d}, column {d}", .{ span.start, span.start });
        }

        try writer.writeByte('\n');

        if (err.suggestion) |suggestion| {
            try writer.writeAll("  suggestion: ");
            try writer.writeAll(suggestion);
            try writer.writeByte('\n');
        }
    }

    /// Print a single warning with formatting
    fn printWarning(self: *const ErrorHandler, writer: anytype, warn: LoweringWarning) !void {
        _ = self;
        try writer.writeAll("warning: ");
        try writer.writeAll(warn.message);

        if (warn.span) |span| {
            try writer.print(" at line {d}, column {d}", .{ span.start, span.start });
        }

        try writer.writeByte('\n');
    }

    /// Validate type compatibility and report errors
    pub fn validateTypeCompatibility(self: *ErrorHandler, expected_type: lib.ast.type_info.OraType, actual_type: lib.ast.type_info.OraType, span: ?lib.ast.SourceSpan) !bool {
        if (!lib.ast.type_info.OraType.equals(expected_type, actual_type)) {
            var message_buf: [512]u8 = undefined;
            var expected_buf: [128]u8 = undefined;
            var actual_buf: [128]u8 = undefined;

            var expected_stream = std.io.fixedBufferStream(&expected_buf);
            var actual_stream = std.io.fixedBufferStream(&actual_buf);

            try expected_type.render(expected_stream.writer());
            try actual_type.render(actual_stream.writer());

            const message = try std.fmt.bufPrint(&message_buf, "type mismatch: expected '{}', found '{}'", .{
                expected_stream.getWritten(),
                actual_stream.getWritten(),
            });

            const suggestion = "check the type of the expression or add an explicit cast";
            try self.reportError(.TypeMismatch, span, message, suggestion);
            return false;
        }
        return true;
    }

    /// Validate memory region constraints
    pub fn validateMemoryRegion(self: *ErrorHandler, region: []const u8, operation: []const u8, span: ?lib.ast.SourceSpan) !bool {
        const valid_regions = [_][]const u8{ "storage", "memory", "tstore" };

        for (valid_regions) |valid_region| {
            if (std.mem.eql(u8, region, valid_region)) {
                return true;
            }
        }

        var message_buf: [256]u8 = undefined;
        const message = try std.fmt.bufPrint(&message_buf, "invalid memory region '{s}' for operation '{s}'", .{ region, operation });
        const suggestion = "use 'storage', 'memory', or 'tstore'";
        try self.reportError(.InvalidMemoryRegion, span, message, suggestion);
        return false;
    }

    /// Validate AST node structure
    pub fn validateAstNode(self: *ErrorHandler, node: anytype, span: ?lib.ast.SourceSpan) !bool {
        const T = @TypeOf(node);

        // Check for null pointers in required fields
        switch (T) {
            lib.ast.expressions.BinaryExpr => {
                if (node.lhs == null or node.rhs == null) {
                    try self.reportError(.MalformedAst, span, "binary operation missing operands", "ensure both left and right operands are provided");
                    return false;
                }
            },
            lib.ast.expressions.UnaryExpr => {
                if (node.operand == null) {
                    try self.reportError(.MalformedAst, span, "unary operation missing operand", "provide an operand for the unary operation");
                    return false;
                }
            },
            lib.ast.expressions.CallExpr => {
                if (node.callee == null) {
                    try self.reportError(.MalformedAst, span, "function call missing callee", "provide a function name or expression");
                    return false;
                }
            },
            else => {
                // Generic validation for other node types
            },
        }

        return true;
    }

    /// Graceful error recovery - create placeholder operations
    pub fn createErrorRecoveryOp(self: *ErrorHandler, ctx: c.MlirContext, block: c.MlirBlock, result_type: c.MlirType, span: ?lib.ast.SourceSpan) c.MlirValue {
        _ = self;

        const location = if (span) |s|
            c.mlirLocationFileLineColGet(ctx, c.mlirStringRefCreateFromCString(""), @intCast(s.start), @intCast(s.start))
        else
            c.mlirLocationUnknownGet(ctx);

        // Create a placeholder constant operation for error recovery
        if (c.mlirTypeIsAInteger(result_type)) {
            const zero_attr = c.mlirIntegerAttrGet(result_type, 0);
            const op_name = c.mlirStringRefCreateFromCString("arith.constant");
            const op_state = c.mlirOperationStateGet(op_name, location);
            c.mlirOperationStateAddResults(&op_state, 1, &result_type);
            c.mlirOperationStateAddAttributes(&op_state, 1, &c.mlirNamedAttributeGet(c.mlirIdentifierGet(ctx, c.mlirStringRefCreateFromCString("value")), zero_attr));
            const op = c.mlirOperationCreate(&op_state);
            c.mlirBlockAppendOwnedOperation(block, op);
            return c.mlirOperationGetResult(op, 0);
        }

        // For non-integer types, create a dummy operation
        // This is a fallback that should rarely be used
        const op_name = c.mlirStringRefCreateFromCString("ora.error_placeholder");
        const op_state = c.mlirOperationStateGet(op_name, location);
        c.mlirOperationStateAddResults(&op_state, 1, &result_type);
        const op = c.mlirOperationCreate(&op_state);
        c.mlirBlockAppendOwnedOperation(block, op);
        return c.mlirOperationGetResult(op, 0);
    }

    /// Validate MLIR operation correctness
    pub fn validateMlirOperation(self: *ErrorHandler, operation: c.MlirOperation, span: ?lib.ast.SourceSpan) !bool {
        if (c.mlirOperationIsNull(operation)) {
            try self.reportError(.MlirOperationFailed, span, "failed to create MLIR operation", "check operation parameters and types");
            return false;
        }

        // Additional validation can be added here
        // For example, checking operation attributes, operand types, etc.

        return true;
    }

    /// Provide actionable error messages with context
    pub fn getActionableErrorMessage(self: *const ErrorHandler, error_type: ErrorType) []const u8 {
        _ = self;
        return switch (error_type) {
            .UnsupportedAstNode => "This AST node type is not yet supported in MLIR lowering. Consider using a simpler construct or file a feature request.",
            .TypeMismatch => "The types don't match. Check your variable declarations and ensure consistent types throughout your code.",
            .UndefinedSymbol => "This symbol is not defined in the current scope. Check for typos or ensure the variable/function is declared before use.",
            .InvalidMemoryRegion => "Invalid memory region specified. Use 'storage' for persistent state, 'memory' for temporary data, or 'tstore' for transient storage.",
            .MalformedAst => "The AST structure is invalid. This might indicate a parser error or corrupted AST node.",
            .MlirOperationFailed => "Failed to create MLIR operation. Check that all operands and types are valid.",
        };
    }
};

/// Types of errors that can occur during MLIR lowering
pub const ErrorType = enum {
    UnsupportedAstNode,
    TypeMismatch,
    UndefinedSymbol,
    InvalidMemoryRegion,
    MalformedAst,
    MlirOperationFailed,
};

/// Types of warnings that can occur during MLIR lowering
pub const WarningType = enum {
    UnusedVariable,
    ImplicitTypeConversion,
    DeprecatedFeature,
    PerformanceWarning,
};

/// Detailed error information
pub const LoweringError = struct {
    error_type: ErrorType,
    span: ?lib.ast.SourceSpan,
    message: []const u8,
    suggestion: ?[]const u8,
    context: ?ErrorContext,
};

/// Warning information
pub const LoweringWarning = struct {
    warning_type: WarningType,
    span: ?lib.ast.SourceSpan,
    message: []const u8,
};

/// Context information for error reporting
pub const ErrorContext = struct {
    function_name: ?[]const u8,
    contract_name: ?[]const u8,
    operation_type: []const u8,

    pub fn function(name: []const u8) ErrorContext {
        return .{
            .function_name = name,
            .contract_name = null,
            .operation_type = "function",
        };
    }

    pub fn contract(name: []const u8) ErrorContext {
        return .{
            .function_name = null,
            .contract_name = name,
            .operation_type = "contract",
        };
    }

    pub fn expression() ErrorContext {
        return .{
            .function_name = null,
            .contract_name = null,
            .operation_type = "expression",
        };
    }

    pub fn statement() ErrorContext {
        return .{
            .function_name = null,
            .contract_name = null,
            .operation_type = "statement",
        };
    }
};

/// Validation utilities
pub const Validator = struct {
    /// Validate that all required AST fields are present
    pub fn validateRequiredFields(comptime T: type, node: T) bool {
        const type_info = @typeInfo(T);
        if (type_info != .Struct) return true;

        // Check for null pointers in pointer fields
        inline for (type_info.Struct.fields) |field| {
            const field_type_info = @typeInfo(field.type);
            if (field_type_info == .Pointer) {
                const field_value = @field(node, field.name);
                if (field_value == null) {
                    return false;
                }
            }
        }

        return true;
    }

    /// Validate integer bounds
    pub fn validateIntegerBounds(value: i64, bit_width: u32) bool {
        const max_value = (@as(i64, 1) << @intCast(bit_width - 1)) - 1;
        const min_value = -(@as(i64, 1) << @intCast(bit_width - 1));
        return value >= min_value and value <= max_value;
    }

    /// Validate identifier names
    pub fn validateIdentifier(name: []const u8) bool {
        if (name.len == 0) return false;

        // First character must be letter or underscore
        if (!std.ascii.isAlphabetic(name[0]) and name[0] != '_') {
            return false;
        }

        // Remaining characters must be alphanumeric or underscore
        for (name[1..]) |char| {
            if (!std.ascii.isAlphanumeric(char) and char != '_') {
                return false;
            }
        }

        return true;
    }
};
