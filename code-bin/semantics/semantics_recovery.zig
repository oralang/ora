const std = @import("std");
pub const ast = @import("../ast.zig");
const semantics_errors = @import("semantics_errors.zig");
const semantics_memory_safety = @import("semantics_memory_safety.zig");

// Forward declaration for SemanticAnalyzer
const SemanticAnalyzer = @import("semantics_core.zig").SemanticAnalyzer;

/// Error recovery strategy
pub const RecoveryStrategy = enum {
    Skip, // Skip the problematic node
    Substitute, // Replace with a safe default
    Partial, // Analyze what we can
    Backtrack, // Go back to a safe state
    Continue, // Continue with warnings
};

/// Recovery context information
pub const RecoveryContext = struct {
    error_type: semantics_errors.SemanticError,
    node_type: std.meta.Tag(ast.AstNode),
    error_span: ast.SourceSpan,
    analysis_phase: AnalysisPhase,
    recovery_attempts: u32,
    max_recovery_attempts: u32,

    pub const AnalysisPhase = enum {
        PreInitialization,
        TypeChecking,
        SemanticAnalysis,
        Validation,
        PostProcessing,
    };

    pub fn init(error_type: semantics_errors.SemanticError, node_type: std.meta.Tag(ast.AstNode), span: ast.SourceSpan, phase: AnalysisPhase) RecoveryContext {
        return RecoveryContext{
            .error_type = error_type,
            .node_type = node_type,
            .error_span = span,
            .analysis_phase = phase,
            .recovery_attempts = 0,
            .max_recovery_attempts = 3,
        };
    }

    pub fn canAttemptRecovery(self: *RecoveryContext) bool {
        return self.recovery_attempts < self.max_recovery_attempts;
    }

    pub fn incrementAttempts(self: *RecoveryContext) void {
        self.recovery_attempts += 1;
    }
};

/// Recovery action result
pub const RecoveryResult = struct {
    success: bool,
    strategy_used: RecoveryStrategy,
    recovery_message: []const u8,
    continue_analysis: bool,
    substitute_node: ?*ast.AstNode,
    warnings_generated: [][]const u8,

    pub fn deinit(self: *RecoveryResult, allocator: std.mem.Allocator) void {
        allocator.free(self.recovery_message);
        for (self.warnings_generated) |warning| {
            allocator.free(warning);
        }
        allocator.free(self.warnings_generated);
    }
};

/// Recovery statistics
pub const RecoveryStatistics = struct {
    total_recovery_attempts: u32,
    successful_recoveries: u32,
    failed_recoveries: u32,
    recovery_by_strategy: std.EnumMap(RecoveryStrategy, u32),
    recovery_by_error_type: std.HashMap(semantics_errors.SemanticError, u32, std.hash_map.AutoContext(semantics_errors.SemanticError), std.hash_map.default_max_load_percentage),
    recovery_by_node_type: std.EnumMap(std.meta.Tag(ast.AstNode), u32),

    pub fn init(allocator: std.mem.Allocator) RecoveryStatistics {
        return RecoveryStatistics{
            .total_recovery_attempts = 0,
            .successful_recoveries = 0,
            .failed_recoveries = 0,
            .recovery_by_strategy = std.EnumMap(RecoveryStrategy, u32).init(.{}),
            .recovery_by_error_type = std.HashMap(semantics_errors.SemanticError, u32, std.hash_map.AutoContext(semantics_errors.SemanticError), std.hash_map.default_max_load_percentage).init(allocator),
            .recovery_by_node_type = std.EnumMap(std.meta.Tag(ast.AstNode), u32).init(.{}),
        };
    }

    pub fn deinit(self: *RecoveryStatistics) void {
        self.recovery_by_error_type.deinit();
    }

    pub fn recordRecoveryAttempt(self: *RecoveryStatistics, context: *RecoveryContext, result: *RecoveryResult) !void {
        self.total_recovery_attempts += 1;

        if (result.success) {
            self.successful_recoveries += 1;
        } else {
            self.failed_recoveries += 1;
        }

        // Update strategy statistics
        const current_strategy_count = self.recovery_by_strategy.get(result.strategy_used) orelse 0;
        self.recovery_by_strategy.put(result.strategy_used, current_strategy_count + 1);

        // Update error type statistics
        const current_error_count = self.recovery_by_error_type.get(context.error_type) orelse 0;
        try self.recovery_by_error_type.put(context.error_type, current_error_count + 1);

        // Update node type statistics
        const current_node_count = self.recovery_by_node_type.get(context.node_type) orelse 0;
        self.recovery_by_node_type.put(context.node_type, current_node_count + 1);
    }

    pub fn getSuccessRate(self: *RecoveryStatistics) f32 {
        if (self.total_recovery_attempts == 0) return 0.0;
        return @as(f32, @floatFromInt(self.successful_recoveries)) / @as(f32, @floatFromInt(self.total_recovery_attempts)) * 100.0;
    }
};

/// Error recovery manager
pub const ErrorRecoveryManager = struct {
    allocator: std.mem.Allocator,
    recovery_statistics: RecoveryStatistics,
    recovery_strategies: std.HashMap(RecoveryKey, RecoveryStrategy, RecoveryKeyContext, std.hash_map.default_max_load_percentage),
    recovery_enabled: bool,

    const RecoveryKey = struct {
        error_type: semantics_errors.SemanticError,
        node_type: std.meta.Tag(ast.AstNode),

        pub fn hash(self: RecoveryKey) u64 {
            var hasher = std.hash.Wyhash.init(0);
            hasher.update(std.mem.asBytes(&self.error_type));
            hasher.update(std.mem.asBytes(&self.node_type));
            return hasher.final();
        }

        pub fn eql(a: RecoveryKey, b: RecoveryKey) bool {
            return a.error_type == b.error_type and a.node_type == b.node_type;
        }
    };

    const RecoveryKeyContext = struct {
        pub fn hash(self: @This(), key: RecoveryKey) u64 {
            _ = self;
            return key.hash();
        }

        pub fn eql(self: @This(), a: RecoveryKey, b: RecoveryKey) bool {
            _ = self;
            return a.eql(b);
        }
    };

    pub fn init(allocator: std.mem.Allocator) ErrorRecoveryManager {
        var manager = ErrorRecoveryManager{
            .allocator = allocator,
            .recovery_statistics = RecoveryStatistics.init(allocator),
            .recovery_strategies = std.HashMap(RecoveryKey, RecoveryStrategy, RecoveryKeyContext, std.hash_map.default_max_load_percentage).init(allocator),
            .recovery_enabled = true,
        };

        // Initialize default recovery strategies
        manager.initializeDefaultStrategies() catch {};
        return manager;
    }

    pub fn deinit(self: *ErrorRecoveryManager) void {
        self.recovery_statistics.deinit();
        self.recovery_strategies.deinit();
    }

    /// Initialize default recovery strategies for common error/node combinations
    fn initializeDefaultStrategies(self: *ErrorRecoveryManager) !void {
        // Memory safety errors - skip problematic nodes
        try self.setRecoveryStrategy(.PointerValidationFailed, .Expr, .Skip);
        try self.setRecoveryStrategy(.MessageValidationFailed, .Expr, .Skip);

        // Type errors - substitute with safe defaults
        try self.setRecoveryStrategy(.TypeMismatch, .Expr, .Substitute);
        try self.setRecoveryStrategy(.InvalidOperation, .Expr, .Substitute);

        // Semantic errors - partial analysis
        try self.setRecoveryStrategy(.UndeclaredIdentifier, .Expr, .Partial);
        try self.setRecoveryStrategy(.InvalidReturnType, .Function, .Partial);

        // Contract errors - continue with warnings
        try self.setRecoveryStrategy(.MissingInitFunction, .Contract, .Continue);
        try self.setRecoveryStrategy(.InvalidContractStructure, .Contract, .Continue);

        // Memory region errors - substitute with stack region
        try self.setRecoveryStrategy(.InvalidStorageAccess, .VariableDecl, .Substitute);
        try self.setRecoveryStrategy(.ImmutableViolation, .Assignment, .Skip);

        // Circular dependency errors - skip problematic dependencies
        try self.setRecoveryStrategy(.CircularDependency, .Import, .Skip);
        try self.setRecoveryStrategy(.CircularDependency, .StructDecl, .Partial);
    }

    /// Set recovery strategy for specific error/node combination
    pub fn setRecoveryStrategy(self: *ErrorRecoveryManager, error_type: semantics_errors.SemanticError, node_type: std.meta.Tag(ast.AstNode), strategy: RecoveryStrategy) !void {
        const key = RecoveryKey{ .error_type = error_type, .node_type = node_type };
        try self.recovery_strategies.put(key, strategy);
    }

    /// Get recovery strategy for error/node combination
    pub fn getRecoveryStrategy(self: *ErrorRecoveryManager, error_type: semantics_errors.SemanticError, node_type: std.meta.Tag(ast.AstNode)) RecoveryStrategy {
        const key = RecoveryKey{ .error_type = error_type, .node_type = node_type };
        return self.recovery_strategies.get(key) orelse .Skip; // Default to skip
    }

    /// Attempt error recovery
    pub fn attemptRecovery(self: *ErrorRecoveryManager, analyzer: *SemanticAnalyzer, context: *RecoveryContext, node: *ast.AstNode) !RecoveryResult {
        if (!self.recovery_enabled or !context.canAttemptRecovery()) {
            return RecoveryResult{
                .success = false,
                .strategy_used = .Skip,
                .recovery_message = try self.allocator.dupe(u8, "Recovery disabled or max attempts reached"),
                .continue_analysis = false,
                .substitute_node = null,
                .warnings_generated = &[_][]const u8{},
            };
        }

        context.incrementAttempts();

        const strategy = self.getRecoveryStrategy(context.error_type, context.node_type);
        var result = try self.executeRecoveryStrategy(analyzer, context, node, strategy);

        // Record recovery attempt
        try self.recovery_statistics.recordRecoveryAttempt(context, &result);

        return result;
    }

    /// Execute specific recovery strategy
    fn executeRecoveryStrategy(self: *ErrorRecoveryManager, analyzer: *SemanticAnalyzer, context: *RecoveryContext, node: *ast.AstNode, strategy: RecoveryStrategy) !RecoveryResult {
        switch (strategy) {
            .Skip => return self.executeSkipStrategy(analyzer, context, node),
            .Substitute => return self.executeSubstituteStrategy(analyzer, context, node),
            .Partial => return self.executePartialStrategy(analyzer, context, node),
            .Backtrack => return self.executeBacktrackStrategy(analyzer, context, node),
            .Continue => return self.executeContinueStrategy(analyzer, context, node),
        }
    }

    fn executeSkipStrategy(self: *ErrorRecoveryManager, analyzer: *SemanticAnalyzer, context: *RecoveryContext, node: *ast.AstNode) !RecoveryResult {
        _ = node;

        const message = try std.fmt.allocPrint(self.allocator, "Skipped analysis of {s} node due to {s}", .{ @tagName(context.node_type), @errorName(context.error_type) });

        const warning = try std.fmt.allocPrint(analyzer.allocator, "Skipped problematic node at line {d}", .{context.error_span.line});
        try semantics_errors.addWarning(analyzer, warning, context.error_span);

        return RecoveryResult{
            .success = true,
            .strategy_used = .Skip,
            .recovery_message = message,
            .continue_analysis = true,
            .substitute_node = null,
            .warnings_generated = &[_][]const u8{warning},
        };
    }

    fn executeSubstituteStrategy(self: *ErrorRecoveryManager, analyzer: *SemanticAnalyzer, context: *RecoveryContext, node: *ast.AstNode) !RecoveryResult {
        _ = node;

        const message = try std.fmt.allocPrint(self.allocator, "Substituted safe default for {s} node due to {s}", .{ @tagName(context.node_type), @errorName(context.error_type) });

        const warning = try std.fmt.allocPrint(analyzer.allocator, "Used safe default for problematic node at line {d}", .{context.error_span.line});
        try semantics_errors.addWarning(analyzer, warning, context.error_span);

        // TODO: Create appropriate substitute node based on node type

        return RecoveryResult{
            .success = true,
            .strategy_used = .Substitute,
            .recovery_message = message,
            .continue_analysis = true,
            .substitute_node = null, // Would be set to actual substitute
            .warnings_generated = &[_][]const u8{warning},
        };
    }

    fn executePartialStrategy(self: *ErrorRecoveryManager, analyzer: *SemanticAnalyzer, context: *RecoveryContext, node: *ast.AstNode) !RecoveryResult {
        _ = node;

        const message = try std.fmt.allocPrint(self.allocator, "Performed partial analysis of {s} node despite {s}", .{ @tagName(context.node_type), @errorName(context.error_type) });

        const warning = try std.fmt.allocPrint(analyzer.allocator, "Partial analysis performed for node at line {d}", .{context.error_span.line});
        try semantics_errors.addWarning(analyzer, warning, context.error_span);

        return RecoveryResult{
            .success = true,
            .strategy_used = .Partial,
            .recovery_message = message,
            .continue_analysis = true,
            .substitute_node = null,
            .warnings_generated = &[_][]const u8{warning},
        };
    }

    fn executeBacktrackStrategy(self: *ErrorRecoveryManager, analyzer: *SemanticAnalyzer, context: *RecoveryContext, node: *ast.AstNode) !RecoveryResult {
        _ = node;

        const message = try std.fmt.allocPrint(self.allocator, "Backtracked from {s} node due to {s}", .{ @tagName(context.node_type), @errorName(context.error_type) });

        const warning = try std.fmt.allocPrint(analyzer.allocator, "Backtracked from problematic analysis at line {d}", .{context.error_span.line});
        try semantics_errors.addWarning(analyzer, warning, context.error_span);

        return RecoveryResult{
            .success = true,
            .strategy_used = .Backtrack,
            .recovery_message = message,
            .continue_analysis = false, // Stop current analysis branch
            .substitute_node = null,
            .warnings_generated = &[_][]const u8{warning},
        };
    }

    fn executeContinueStrategy(self: *ErrorRecoveryManager, analyzer: *SemanticAnalyzer, context: *RecoveryContext, node: *ast.AstNode) !RecoveryResult {
        _ = node;

        const message = try std.fmt.allocPrint(self.allocator, "Continued analysis despite {s} in {s} node", .{ @errorName(context.error_type), @tagName(context.node_type) });

        const warning = try std.fmt.allocPrint(analyzer.allocator, "Continuing analysis with potential issues at line {d}", .{context.error_span.line});
        try semantics_errors.addWarning(analyzer, warning, context.error_span);

        return RecoveryResult{
            .success = true,
            .strategy_used = .Continue,
            .recovery_message = message,
            .continue_analysis = true,
            .substitute_node = null,
            .warnings_generated = &[_][]const u8{warning},
        };
    }

    /// Enable or disable error recovery
    pub fn setRecoveryEnabled(self: *ErrorRecoveryManager, enabled: bool) void {
        self.recovery_enabled = enabled;
    }

    /// Get recovery statistics
    pub fn getStatistics(self: *ErrorRecoveryManager) RecoveryStatistics {
        return self.recovery_statistics;
    }

    /// Generate recovery report
    pub fn generateRecoveryReport(self: *ErrorRecoveryManager) ![]const u8 {
        var output = std.ArrayList(u8).init(self.allocator);
        defer output.deinit();

        const stats = &self.recovery_statistics;

        try output.writer().print("=== Error Recovery Report ===\n\n");
        try output.writer().print("Total Recovery Attempts: {d}\n", .{stats.total_recovery_attempts});
        try output.writer().print("Successful Recoveries: {d}\n", .{stats.successful_recoveries});
        try output.writer().print("Failed Recoveries: {d}\n", .{stats.failed_recoveries});
        try output.writer().print("Success Rate: {d:.1}%\n\n", .{stats.getSuccessRate()});

        try output.appendSlice("Recovery by Strategy:\n");
        const strategy_fields = @typeInfo(RecoveryStrategy).Enum.fields;
        for (strategy_fields) |field| {
            const strategy = @as(RecoveryStrategy, @enumFromInt(field.value));
            const count = stats.recovery_by_strategy.get(strategy) orelse 0;
            try output.writer().print("  {s}: {d}\n", .{ field.name, count });
        }

        return output.toOwnedSlice();
    }
};

/// Attempt error recovery for analyzer
pub fn attemptErrorRecovery(analyzer: *SemanticAnalyzer, error_type: semantics_errors.SemanticError, node: *ast.AstNode, span: ast.SourceSpan) !bool {
    // Create a temporary recovery manager for this attempt
    var recovery_manager = ErrorRecoveryManager.init(analyzer.allocator);
    defer recovery_manager.deinit();

    // Create recovery context
    var context = RecoveryContext.init(error_type, @as(std.meta.Tag(ast.AstNode), node.*), span, .SemanticAnalysis);

    // Attempt recovery
    var result = recovery_manager.attemptRecovery(analyzer, &context, node) catch |err| {
        switch (err) {
            error.OutOfMemory => return semantics_errors.SemanticError.OutOfMemory,
            else => return false,
        }
    };
    defer result.deinit(analyzer.allocator);

    // Update analyzer recovery statistics
    analyzer.validation_coverage.validation_stats.recovery_attempts += 1;

    return result.success and result.continue_analysis;
}

/// Check if error is recoverable
pub fn isRecoverableError(error_type: semantics_errors.SemanticError) bool {
    return switch (error_type) {
        .OutOfMemory => false, // Cannot recover from OOM
        .AnalysisStateCorrupted => false, // Cannot recover from corrupted state
        .RecoveryFailed => false, // Cannot recover from recovery failure
        else => true, // Most other errors are potentially recoverable
    };
}

/// Create safe recovery context
pub fn createRecoveryContext(error_type: semantics_errors.SemanticError, node_type: std.meta.Tag(ast.AstNode), span: ast.SourceSpan) RecoveryContext {
    return RecoveryContext.init(error_type, node_type, span, .SemanticAnalysis);
}
