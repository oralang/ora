// ============================================================================
// MLIR Error Handling
// ============================================================================
//
// Comprehensive error handling and validation system for MLIR lowering.
//
// FEATURES:
//   • Detailed error reporting with source spans
//   • Warning system for non-fatal issues
//   • Error recovery mode for batch compilation
//   • Context stack for nested error tracking
//   • Suggestions for common mistakes
//
// MEMORY OWNERSHIP:
//   Owns: errors, warnings, context_stack, all message strings
//   Borrows: source spans from AST
//   Must call deinit() to avoid leaks
//
// ============================================================================

const std = @import("std");
const c = @import("c.zig").c;
const lib = @import("ora_lib");

/// Comprehensive error handling and validation system for MLIR lowering
pub const ErrorHandler = struct {
    allocator: std.mem.Allocator,
    errors: std.ArrayList(LoweringError),
    warnings: std.ArrayList(LoweringWarning),
    context_stack: std.ArrayList(ErrorContext),
    error_recovery_mode: bool, // Enable error recovery mode
    max_errors: usize, // Maximum errors before giving up
    error_count: usize, // Current error count

    pub fn init(allocator: std.mem.Allocator) ErrorHandler {
        return .{
            .allocator = allocator,
            .errors = std.ArrayList(LoweringError){},
            .warnings = std.ArrayList(LoweringWarning){},
            .context_stack = std.ArrayList(ErrorContext){},
            .error_recovery_mode = true,
            .max_errors = 100, // Allow up to 100 errors before giving up
            .error_count = 0,
        };
    }

    pub fn deinit(self: *ErrorHandler) void {
        // Free individual error messages and suggestions
        for (self.errors.items) |*err| {
            self.allocator.free(err.message);
            if (err.suggestion) |suggestion| {
                self.allocator.free(suggestion);
            }
        }

        // Free individual warning messages
        for (self.warnings.items) |*warn| {
            self.allocator.free(warn.message);
        }

        self.errors.deinit(self.allocator);
        self.warnings.deinit(self.allocator);
        self.context_stack.deinit(self.allocator);
    }

    /// Enable or disable error recovery mode
    pub fn setErrorRecoveryMode(self: *ErrorHandler, enabled: bool) void {
        self.error_recovery_mode = enabled;
    }

    /// Set maximum number of errors before giving up
    pub fn setMaxErrors(self: *ErrorHandler, max: usize) void {
        self.max_errors = max;
    }

    /// Check if we should continue processing (error recovery mode)
    pub fn shouldContinue(self: *const ErrorHandler) bool {
        return self.error_recovery_mode and self.error_count < self.max_errors;
    }

    /// Push an error context onto the stack
    pub fn pushContext(self: *ErrorHandler, context: ErrorContext) !void {
        try self.context_stack.append(self.allocator, context);
    }

    /// Pop the current error context
    pub fn popContext(self: *ErrorHandler) void {
        if (self.context_stack.items.len > 0) {
            _ = self.context_stack.pop();
        }
    }

    /// Report an error with source location information and automatic recovery
    pub fn reportError(self: *ErrorHandler, error_type: ErrorType, span: ?lib.ast.SourceSpan, message: []const u8, suggestion: ?[]const u8) !void {
        self.error_count += 1;

        const error_info = LoweringError{
            .error_type = error_type,
            .span = span,
            .message = try self.allocator.dupe(u8, message),
            .suggestion = if (suggestion) |s| try self.allocator.dupe(u8, s) else null,
            .context = if (self.context_stack.items.len > 0) self.context_stack.items[self.context_stack.items.len - 1] else null,
        };
        try self.errors.append(self.allocator, error_info);

        // If we've exceeded max errors and recovery is disabled, panic
        if (!self.error_recovery_mode and self.error_count >= self.max_errors) {
            @panic("Too many errors during MLIR lowering - compilation aborted");
        }
    }

    /// Report a warning with source location information
    pub fn reportWarning(self: *ErrorHandler, warning_type: WarningType, span: ?lib.ast.SourceSpan, message: []const u8) !void {
        const warning_info = LoweringWarning{
            .warning_type = warning_type,
            .span = span,
            .message = try self.allocator.dupe(u8, message),
        };
        try self.warnings.append(self.allocator, warning_info);
    }

    /// Report an unsupported feature with helpful suggestions
    pub fn reportUnsupportedFeature(self: *ErrorHandler, feature_name: []const u8, span: ?lib.ast.SourceSpan, context: []const u8) !void {
        const message = try std.fmt.allocPrint(self.allocator, "Feature '{s}' is not yet supported in MLIR lowering", .{feature_name});
        defer self.allocator.free(message);

        const suggestion = try std.fmt.allocPrint(self.allocator, "Consider using a simpler alternative or wait for future implementation. Context: {s}", .{context});
        defer self.allocator.free(suggestion);

        try self.reportError(.UnsupportedFeature, span, message, suggestion);
    }

    /// Report a missing node type with recovery suggestions
    pub fn reportMissingNodeType(self: *ErrorHandler, node_type: []const u8, span: ?lib.ast.SourceSpan, parent_context: []const u8) !void {
        const message = try std.fmt.allocPrint(self.allocator, "Node type '{s}' is not handled in MLIR lowering", .{node_type});
        defer self.allocator.free(message);

        const suggestion = try std.fmt.allocPrint(self.allocator, "This {s} contains unsupported constructs. Consider simplifying the code or removing unsupported features.", .{parent_context});
        defer self.allocator.free(suggestion);

        try self.reportError(.MissingNodeType, span, message, suggestion);
    }

    /// Report a graceful degradation with explanation
    pub fn reportGracefulDegradation(self: *ErrorHandler, feature: []const u8, fallback: []const u8, span: ?lib.ast.SourceSpan) !void {
        const message = try std.fmt.allocPrint(self.allocator, "Feature '{s}' degraded to '{s}' for compatibility", .{ feature, fallback });
        defer self.allocator.free(message);

        const suggestion = try std.fmt.allocPrint(self.allocator, "The code will compile but may not have optimal performance. Consider using supported alternatives.", .{});
        defer self.allocator.free(suggestion);

        try self.reportWarning(.GracefulDegradation, span, message);
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

    /// Get current error count
    pub fn getErrorCount(self: *const ErrorHandler) usize {
        return self.error_count;
    }

    /// Reset error count (useful for testing or partial compilation)
    pub fn resetErrorCount(self: *ErrorHandler) void {
        self.error_count = 0;
    }

    /// Format and print all errors and warnings
    pub fn printDiagnostics(self: *ErrorHandler, writer: anytype) !void {
        // Print errors
        for (self.errors.items) |err| {
            try self.printError(writer, err);
        }

        // Print warnings
        for (self.warnings.items) |warn| {
            try self.printWarning(writer, warn);
        }

        // Print summary
        if (self.errors.items.len > 0 or self.warnings.items.len > 0) {
            try writer.print("\nDiagnostics Summary:\n", .{});
            try writer.print("  Errors: {d}\n", .{self.errors.items.len});
            try writer.print("  Warnings: {d}\n", .{self.warnings.items.len});

            if (self.error_recovery_mode) {
                try writer.print("  Error Recovery: Enabled (max {d} errors)\n", .{self.max_errors});
            } else {
                try writer.print("  Error Recovery: Disabled\n", .{});
            }
        }
    }

    /// Print a single error with formatting
    fn printError(self: *ErrorHandler, writer: anytype, err: LoweringError) !void {
        _ = self;
        try writer.writeAll("error: ");
        try writer.writeAll(err.message);

        if (err.span) |span| {
            try writer.print(" at {s}:{d}:{d}", .{ span.file_path, span.start_line, span.start_column });
        }

        if (err.suggestion) |suggestion| {
            try writer.print("\n  suggestion: {s}", .{suggestion});
        }

        if (err.context) |context| {
            try writer.print("\n  context: {s}", .{context.name});
        }

        try writer.writeAll("\n");
    }

    /// Print a single warning with formatting
    fn printWarning(self: *ErrorHandler, writer: anytype, warn: LoweringWarning) !void {
        _ = self;
        try writer.writeAll("warning: ");
        try writer.writeAll(warn.message);

        if (warn.span) |span| {
            try writer.print(" at {s}:{d}:{d}", .{ span.file_path, span.start_line, span.start_column });
        }

        try writer.writeAll("\n");
    }

    /// Validate an AST node with comprehensive error reporting
    pub fn validateAstNode(_: *ErrorHandler, _: anytype, _: ?lib.ast.SourceSpan) !bool {
        // Basic validation - always return true for now
        // This can be enhanced with specific validation logic
        return true;
    }

    /// Validate an MLIR operation with comprehensive error reporting
    pub fn validateMlirOperation(_: *ErrorHandler, _: c.MlirOperation, _: ?lib.ast.SourceSpan) !bool {
        // Basic validation - always return true for now
        // This can be enhanced with MLIR operation validation
        return true;
    }

    /// Validate memory region access with comprehensive error reporting
    pub fn validateMemoryRegion(_: *ErrorHandler, _: lib.ast.Statements.MemoryRegion, _: []const u8, _: ?lib.ast.SourceSpan) !bool {
        // Basic validation - always return true for now
        // This can be enhanced with memory region validation
        return true;
    }
};

/// Error types for MLIR lowering
pub const ErrorType = enum {
    MalformedAst,
    TypeMismatch,
    UndefinedSymbol,
    InvalidMemoryRegion,
    MlirOperationFailed,
    UnsupportedFeature,
    MissingNodeType,
    CompilationLimit,
    InternalError,
};

/// Warning types for MLIR lowering
pub const WarningType = enum {
    DeprecatedFeature,
    GracefulDegradation,
    PerformanceWarning,
    CompatibilityWarning,
    ImplementationWarning,
};

/// Error information with context
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

/// Error context for better diagnostics
pub const ErrorContext = struct {
    name: []const u8,
    details: ?[]const u8,

    pub fn function(name: []const u8) ErrorContext {
        return .{ .name = name, .details = null };
    }

    pub fn contract(name: []const u8) ErrorContext {
        return .{ .name = name, .details = null };
    }

    pub fn expression() ErrorContext {
        return .{ .name = "expression", .details = null };
    }

    pub fn statement() ErrorContext {
        return .{ .name = "statement", .details = null };
    }

    pub fn module(name: ?[]const u8) ErrorContext {
        return .{ .name = if (name) |n| n else "module", .details = null };
    }

    pub fn block(name: []const u8) ErrorContext {
        return .{ .name = name, .details = null };
    }

    pub fn try_block(name: []const u8) ErrorContext {
        return .{ .name = name, .details = null };
    }

    pub fn withDetails(self: ErrorContext, details: []const u8) ErrorContext {
        return .{ .name = self.name, .details = details };
    }
};

/// Get the span from an expression node
pub fn getSpanFromExpression(expr: *const lib.ast.expressions.ExprNode) lib.ast.SourceSpan {
    return switch (expr.*) {
        .Identifier => |ident| ident.span,
        .Literal => |lit| switch (lit) {
            .Integer => |int| int.span,
            .String => |str| str.span,
            .Bool => |bool_lit| bool_lit.span,
            .Address => |addr| addr.span,
            .Hex => |hex| hex.span,
            .Binary => |bin| bin.span,
            .Character => |char| char.span,
            .Bytes => |bytes| bytes.span,
        },
        .Binary => |bin| bin.span,
        .Unary => |unary| unary.span,
        .Call => |call| call.span,
        .Assignment => |assign| assign.span,
        .CompoundAssignment => |compound| compound.span,
        .Index => |index| index.span,
        .FieldAccess => |field| field.span,
        .Cast => |cast| cast.span,
        .Comptime => |comptime_expr| comptime_expr.span,
        .Old => |old| old.span,
        .Tuple => |tuple| tuple.span,
        .SwitchExpression => |switch_expr| switch_expr.span,
        .Quantified => |quantified| quantified.span,
        .Try => |try_expr| try_expr.span,
        .ErrorReturn => |error_ret| error_ret.span,
        .ErrorCast => |error_cast| error_cast.span,
        .Shift => |shift| shift.span,
        .StructInstantiation => |struct_inst| struct_inst.span,
        .AnonymousStruct => |anon_struct| anon_struct.span,
        .Range => |range| range.span,
        .LabeledBlock => |labeled_block| labeled_block.span,
        .Destructuring => |destructuring| destructuring.span,
        .EnumLiteral => |enum_lit| enum_lit.span,
        .ArrayLiteral => |array_lit| array_lit.span,
    };
}

/// Get the span from a statement node
pub fn getSpanFromStatement(stmt: *const lib.ast.Statements.StmtNode) lib.ast.SourceSpan {
    return switch (stmt.*) {
        .Return => |ret| ret.span,
        .VariableDecl => |var_decl| var_decl.span,
        .DestructuringAssignment => |destruct| destruct.span,
        .CompoundAssignment => |compound| compound.span,
        .If => |if_stmt| if_stmt.span,
        .While => |while_stmt| while_stmt.span,
        .ForLoop => |for_stmt| for_stmt.span,
        .Switch => |switch_stmt| switch_stmt.span,
        .Break => |break_stmt| break_stmt.span,
        .Continue => |continue_stmt| continue_stmt.span,
        .Log => |log_stmt| log_stmt.span,
        .Lock => |lock_stmt| lock_stmt.span,
        .Unlock => |unlock_stmt| unlock_stmt.span,
        .Assert => |assert_stmt| assert_stmt.span,
        .TryBlock => |try_stmt| try_stmt.span,
        .ErrorDecl => |error_decl| error_decl.span,
        .Invariant => |invariant| invariant.span,
        .Requires => |requires| requires.span,
        .Ensures => |ensures| ensures.span,
        .Assume => |assume| assume.span,
        .Havoc => |havoc| havoc.span,
        .Expr => |expr| getSpanFromExpression(&expr),
        .LabeledBlock => |labeled_block| labeled_block.span,
    };
}

/// Get the span from an AST node
pub fn getSpanFromAstNode(node: *const lib.ast.AstNode) lib.ast.SourceSpan {
    return switch (node.*) {
        .Module => |module| module.span,
        .Contract => |contract| contract.span,
        .Function => |function| function.span,
        .Constant => |constant| constant.span,
        .VariableDecl => |var_decl| var_decl.span,
        .StructDecl => |struct_decl| struct_decl.span,
        .EnumDecl => |enum_decl| enum_decl.span,
        .LogDecl => |log_decl| log_decl.span,
        .Import => |import| import.span,
        .ErrorDecl => |error_decl| error_decl.span,
        .ContractInvariant => |invariant| invariant.span,
        .Block => |block| block.span,
        .Expression => |expr| getSpanFromExpression(expr),
        .Statement => |stmt| getSpanFromStatement(stmt),
        .TryBlock => |try_block| try_block.span,
    };
}
