const std = @import("std");
const ast = @import("ast.zig");
const static_verifier = @import("static_verifier.zig");
const comptime_eval = @import("comptime_eval.zig");
const Allocator = std.mem.Allocator;

/// Optimization errors
pub const OptimizationError = error{
    InvalidTransformation,
    OutOfMemory,
};

/// Types of optimizations that can be performed
pub const OptimizationType = enum {
    RedundantCheckElimination,
    DeadCodeElimination,
    ConstantFolding,
    BoundsCheckElimination,
    NullCheckElimination,
    TautologyElimination,
    ContradictionElimination,
    LoopInvariantHoisting,
};

/// Optimization result tracking
pub const OptimizationResult = struct {
    transformations_applied: u32,
    checks_eliminated: u32,
    instructions_removed: u32,
    optimizations: []OptimizationInfo,

    pub const OptimizationInfo = struct {
        type: OptimizationType,
        description: []const u8,
        span: ast.SourceSpan,
        savings: OptimizationSavings,
    };

    pub const OptimizationSavings = struct { gas_saved: u64, instructions_saved: u32, runtime_checks_eliminated: u32 };
};

/// Context for optimization passes
pub const OptimizationContext = struct {
    verification_results: []static_verifier.VerificationResult,
    proven_conditions: std.HashMap([]const u8, bool, std.hash_map.StringContext, std.hash_map.default_max_load_percentage),
    comptime_constants: std.HashMap([]const u8, comptime_eval.ComptimeValue, std.hash_map.StringContext, std.hash_map.default_max_load_percentage),
    allocator: Allocator,

    pub fn init(allocator: Allocator) OptimizationContext {
        return OptimizationContext{
            .verification_results = &[_]static_verifier.VerificationResult{},
            .proven_conditions = std.HashMap([]const u8, bool, std.hash_map.StringContext, std.hash_map.default_max_load_percentage).init(allocator),
            .comptime_constants = std.HashMap([]const u8, comptime_eval.ComptimeValue, std.hash_map.StringContext, std.hash_map.default_max_load_percentage).init(allocator),
            .allocator = allocator,
        };
    }

    pub fn deinit(self: *OptimizationContext) void {
        self.proven_conditions.deinit();
        self.comptime_constants.deinit();
    }
};

/// The main optimizer
pub const Optimizer = struct {
    allocator: Allocator,
    context: OptimizationContext,
    optimizations: std.ArrayList(OptimizationResult.OptimizationInfo),

    pub fn init(allocator: Allocator) Optimizer {
        return Optimizer{
            .allocator = allocator,
            .context = OptimizationContext.init(allocator),
            .optimizations = std.ArrayList(OptimizationResult.OptimizationInfo).init(allocator),
        };
    }

    pub fn deinit(self: *Optimizer) void {
        self.context.deinit();
        self.optimizations.deinit();
    }

    /// Set verification results to use for optimization
    pub fn setVerificationResults(self: *Optimizer, results: []static_verifier.VerificationResult) void {
        self.context.verification_results = results;
    }

    /// Add a proven condition that can be used for optimization
    pub fn addProvenCondition(self: *Optimizer, condition: []const u8, is_true: bool) !void {
        try self.context.proven_conditions.put(condition, is_true);
    }

    /// Add a compile-time constant
    pub fn addConstant(self: *Optimizer, name: []const u8, value: comptime_eval.ComptimeValue) !void {
        try self.context.comptime_constants.put(name, value);
    }

    /// Optimize a function
    pub fn optimizeFunction(self: *Optimizer, function: *ast.FunctionNode) !OptimizationResult {
        self.optimizations.clearRetainingCapacity();

        var transformations: u32 = 0;
        var checks_eliminated: u32 = 0;
        var instructions_removed: u32 = 0;

        // Pass 1: Eliminate redundant runtime checks
        const redundant_result = try self.eliminateRedundantChecks(function);
        transformations += redundant_result.transformations;
        checks_eliminated += redundant_result.checks_eliminated;

        // Pass 2: Constant folding
        const constant_result = try self.performConstantFolding(function);
        transformations += constant_result.transformations;
        instructions_removed += constant_result.instructions_removed;

        // Pass 3: Dead code elimination
        const dead_code_result = try self.eliminateDeadCode(function);
        transformations += dead_code_result.transformations;
        instructions_removed += dead_code_result.instructions_removed;

        // Pass 4: Tautology elimination
        const tautology_result = try self.eliminateTautologies(function);
        transformations += tautology_result.transformations;
        checks_eliminated += tautology_result.checks_eliminated;

        // Pass 5: Bounds check elimination
        const bounds_result = try self.eliminateBoundsChecks(function);
        transformations += bounds_result.transformations;
        checks_eliminated += bounds_result.checks_eliminated;

        return OptimizationResult{
            .transformations_applied = transformations,
            .checks_eliminated = checks_eliminated,
            .instructions_removed = instructions_removed,
            .optimizations = try self.optimizations.toOwnedSlice(),
        };
    }

    /// Pass 1: Eliminate redundant runtime checks
    fn eliminateRedundantChecks(self: *Optimizer, function: *ast.FunctionNode) !struct { transformations: u32, checks_eliminated: u32 } {
        var transformations: u32 = 0;
        var checks_eliminated: u32 = 0;

        // Analyze requires clauses to find proven conditions
        for (function.requires_clauses) |*clause| {
            if (self.isProvenAtCompileTime(clause)) {
                try self.recordOptimization(.RedundantCheckElimination, "Requires clause proven at compile time", getExprSpan(clause), OptimizationResult.OptimizationSavings{
                    .gas_saved = 50, // Typical gas cost of a runtime check
                    .instructions_saved = 1,
                    .runtime_checks_eliminated = 1,
                });
                checks_eliminated += 1;
                transformations += 1;
            }
        }

        // Look for redundant checks in function body
        try self.eliminateRedundantChecksInBlock(&function.body, &transformations, &checks_eliminated);

        return .{ .transformations = transformations, .checks_eliminated = checks_eliminated };
    }

    /// Eliminate redundant checks in a block
    fn eliminateRedundantChecksInBlock(self: *Optimizer, block: *ast.BlockNode, transformations: *u32, checks_eliminated: *u32) !void {
        for (block.statements) |*stmt| {
            switch (stmt.*) {
                .Requires => |*req| {
                    if (self.isProvenAtCompileTime(&req.condition)) {
                        try self.recordOptimization(.RedundantCheckElimination, "Runtime requires check eliminated", req.span, OptimizationResult.OptimizationSavings{
                            .gas_saved = 50,
                            .instructions_saved = 1,
                            .runtime_checks_eliminated = 1,
                        });
                        checks_eliminated.* += 1;
                        transformations.* += 1;
                    }
                },
                .If => |*if_stmt| {
                    // Check if condition is compile-time constant
                    if (self.isCompileTimeConstant(&if_stmt.condition)) {
                        try self.recordOptimization(.DeadCodeElimination, "If condition is compile-time constant", ast.SourceSpan{ .line = 0, .column = 0, .length = 0 }, OptimizationResult.OptimizationSavings{
                            .gas_saved = 30,
                            .instructions_saved = 1,
                            .runtime_checks_eliminated = 1,
                        });
                        transformations.* += 1;
                    }

                    // Recursively check branches
                    try self.eliminateRedundantChecksInBlock(&if_stmt.then_branch, transformations, checks_eliminated);
                    if (if_stmt.else_branch) |*else_branch| {
                        try self.eliminateRedundantChecksInBlock(else_branch, transformations, checks_eliminated);
                    }
                },
                .While => |*while_stmt| {
                    // Check for invariants that eliminate checks
                    for (while_stmt.invariants) |*inv| {
                        if (self.isProvenAtCompileTime(inv)) {
                            try self.recordOptimization(.LoopInvariantHoisting, "Loop invariant proven at compile time", ast.SourceSpan{ .line = 0, .column = 0, .length = 0 }, OptimizationResult.OptimizationSavings{
                                .gas_saved = 100, // Saves gas on every iteration
                                .instructions_saved = 1,
                                .runtime_checks_eliminated = 1,
                            });
                            checks_eliminated.* += 1;
                            transformations.* += 1;
                        }
                    }

                    try self.eliminateRedundantChecksInBlock(&while_stmt.body, transformations, checks_eliminated);
                },
                else => {
                    // TODO: Add optimization for: Expr, VariableDecl, Return, Log, Break, Continue, Invariant, Requires, Ensures, ErrorDecl, TryBlock
                },
            }
        }
    }

    /// Pass 2: Constant folding
    fn performConstantFolding(self: *Optimizer, function: *ast.FunctionNode) !struct { transformations: u32, instructions_removed: u32 } {
        var transformations: u32 = 0;
        var instructions_removed: u32 = 0;

        // Fold constants in requires/ensures clauses
        for (function.requires_clauses) |*clause| {
            if (self.canFoldToConstant(clause)) {
                try self.recordOptimization(.ConstantFolding, "Constant folded in requires clause", getExprSpan(clause), OptimizationResult.OptimizationSavings{
                    .gas_saved = 20,
                    .instructions_saved = 1,
                    .runtime_checks_eliminated = 0,
                });
                transformations += 1;
                instructions_removed += 1;
            }
        }

        for (function.ensures_clauses) |*clause| {
            if (self.canFoldToConstant(clause)) {
                try self.recordOptimization(.ConstantFolding, "Constant folded in ensures clause", getExprSpan(clause), OptimizationResult.OptimizationSavings{
                    .gas_saved = 20,
                    .instructions_saved = 1,
                    .runtime_checks_eliminated = 0,
                });
                transformations += 1;
                instructions_removed += 1;
            }
        }

        // Fold constants in function body
        try self.foldConstantsInBlock(&function.body, &transformations, &instructions_removed);

        return .{ .transformations = transformations, .instructions_removed = instructions_removed };
    }

    /// Fold constants in a block
    fn foldConstantsInBlock(self: *Optimizer, block: *ast.BlockNode, transformations: *u32, instructions_removed: *u32) !void {
        for (block.statements) |*stmt| {
            switch (stmt.*) {
                .Expr => |*expr| {
                    if (self.canFoldToConstant(expr)) {
                        try self.recordOptimization(.ConstantFolding, "Expression folded to constant", ast.SourceSpan{ .line = 0, .column = 0, .length = 0 }, OptimizationResult.OptimizationSavings{
                            .gas_saved = 15,
                            .instructions_saved = 1,
                            .runtime_checks_eliminated = 0,
                        });
                        transformations.* += 1;
                        instructions_removed.* += 1;
                    }
                },
                .VariableDecl => |*var_decl| {
                    if (var_decl.value) |*init_expr| {
                        if (self.canFoldToConstant(init_expr)) {
                            try self.recordOptimization(.ConstantFolding, "Variable initializer folded to constant", var_decl.span, OptimizationResult.OptimizationSavings{
                                .gas_saved = 10,
                                .instructions_saved = 1,
                                .runtime_checks_eliminated = 0,
                            });
                            transformations.* += 1;
                            instructions_removed.* += 1;
                        }
                    }
                },
                .If => |*if_stmt| {
                    try self.foldConstantsInBlock(&if_stmt.then_branch, transformations, instructions_removed);
                    if (if_stmt.else_branch) |*else_branch| {
                        try self.foldConstantsInBlock(else_branch, transformations, instructions_removed);
                    }
                },
                .While => |*while_stmt| {
                    try self.foldConstantsInBlock(&while_stmt.body, transformations, instructions_removed);
                },
                else => {},
            }
        }
    }

    /// Pass 3: Dead code elimination
    fn eliminateDeadCode(self: *Optimizer, function: *ast.FunctionNode) !struct { transformations: u32, instructions_removed: u32 } {
        var transformations: u32 = 0;
        var instructions_removed: u32 = 0;

        // Look for unreachable code after proven false conditions
        try self.eliminateDeadCodeInBlock(&function.body, &transformations, &instructions_removed);

        return .{ .transformations = transformations, .instructions_removed = instructions_removed };
    }

    /// Eliminate dead code in a block
    fn eliminateDeadCodeInBlock(self: *Optimizer, block: *ast.BlockNode, transformations: *u32, instructions_removed: *u32) !void {
        for (block.statements) |*stmt| {
            switch (stmt.*) {
                .If => |*if_stmt| {
                    if (self.isCompileTimeConstant(&if_stmt.condition)) {
                        // Check if condition is always false
                        if (self.isAlwaysFalse(&if_stmt.condition)) {
                            try self.recordOptimization(.DeadCodeElimination, "Dead code eliminated (always false condition)", ast.SourceSpan{ .line = 0, .column = 0, .length = 0 }, OptimizationResult.OptimizationSavings{
                                .gas_saved = 200, // Significant savings from removing entire branch
                                .instructions_saved = 10,
                                .runtime_checks_eliminated = 1,
                            });
                            transformations.* += 1;
                            instructions_removed.* += 10; // Estimate
                        }
                        // Check if condition is always true (eliminate else branch)
                        else if (self.isAlwaysTrue(&if_stmt.condition) and if_stmt.else_branch != null) {
                            try self.recordOptimization(.DeadCodeElimination, "Dead else branch eliminated (always true condition)", ast.SourceSpan{ .line = 0, .column = 0, .length = 0 }, OptimizationResult.OptimizationSavings{
                                .gas_saved = 100,
                                .instructions_saved = 5,
                                .runtime_checks_eliminated = 0,
                            });
                            transformations.* += 1;
                            instructions_removed.* += 5;
                        }
                    }

                    // Recursively check branches
                    try self.eliminateDeadCodeInBlock(&if_stmt.then_branch, transformations, instructions_removed);
                    if (if_stmt.else_branch) |*else_branch| {
                        try self.eliminateDeadCodeInBlock(else_branch, transformations, instructions_removed);
                    }
                },
                .While => |*while_stmt| {
                    // Check for infinite loops (condition always true) or loops that never run (condition always false)
                    if (self.isCompileTimeConstant(&while_stmt.condition)) {
                        if (self.isAlwaysFalse(&while_stmt.condition)) {
                            try self.recordOptimization(.DeadCodeElimination, "Dead loop eliminated (condition always false)", ast.SourceSpan{ .line = 0, .column = 0, .length = 0 }, OptimizationResult.OptimizationSavings{
                                .gas_saved = 300,
                                .instructions_saved = 15,
                                .runtime_checks_eliminated = 1,
                            });
                            transformations.* += 1;
                            instructions_removed.* += 15;
                        }
                    }

                    try self.eliminateDeadCodeInBlock(&while_stmt.body, transformations, instructions_removed);
                },
                else => {},
            }
        }
    }

    /// Pass 4: Eliminate tautologies
    fn eliminateTautologies(self: *Optimizer, function: *ast.FunctionNode) !struct { transformations: u32, checks_eliminated: u32 } {
        var transformations: u32 = 0;
        var checks_eliminated: u32 = 0;

        // Check requires clauses for tautologies
        for (function.requires_clauses) |*clause| {
            if (self.isTautology(clause)) {
                try self.recordOptimization(.TautologyElimination, "Tautological requires clause eliminated", getExprSpan(clause), OptimizationResult.OptimizationSavings{
                    .gas_saved = 50,
                    .instructions_saved = 1,
                    .runtime_checks_eliminated = 1,
                });
                transformations += 1;
                checks_eliminated += 1;
            }
        }

        // Check ensures clauses for tautologies
        for (function.ensures_clauses) |*clause| {
            if (self.isTautology(clause)) {
                try self.recordOptimization(.TautologyElimination, "Tautological ensures clause eliminated", getExprSpan(clause), OptimizationResult.OptimizationSavings{
                    .gas_saved = 30,
                    .instructions_saved = 1,
                    .runtime_checks_eliminated = 1,
                });
                transformations += 1;
                checks_eliminated += 1;
            }
        }

        return .{ .transformations = transformations, .checks_eliminated = checks_eliminated };
    }

    /// Pass 5: Eliminate bounds checks
    fn eliminateBoundsChecks(self: *Optimizer, function: *ast.FunctionNode) !struct { transformations: u32, checks_eliminated: u32 } {
        var transformations: u32 = 0;
        var checks_eliminated: u32 = 0;

        // Look for array access patterns where bounds are statically verified
        try self.eliminateBoundsChecksInBlock(&function.body, &transformations, &checks_eliminated);

        return .{ .transformations = transformations, .checks_eliminated = checks_eliminated };
    }

    /// Eliminate bounds checks in a block
    fn eliminateBoundsChecksInBlock(self: *Optimizer, block: *ast.BlockNode, transformations: *u32, checks_eliminated: *u32) !void {
        for (block.statements) |*stmt| {
            switch (stmt.*) {
                .Expr => |*expr| {
                    try self.checkForBoundsOptimizations(expr, transformations, checks_eliminated);
                },
                .VariableDecl => |*var_decl| {
                    if (var_decl.value) |*init_expr| {
                        try self.checkForBoundsOptimizations(init_expr, transformations, checks_eliminated);
                    }
                },
                .If => |*if_stmt| {
                    try self.eliminateBoundsChecksInBlock(&if_stmt.then_branch, transformations, checks_eliminated);
                    if (if_stmt.else_branch) |*else_branch| {
                        try self.eliminateBoundsChecksInBlock(else_branch, transformations, checks_eliminated);
                    }
                },
                .While => |*while_stmt| {
                    try self.eliminateBoundsChecksInBlock(&while_stmt.body, transformations, checks_eliminated);
                },
                else => {},
            }
        }
    }

    /// Check for bounds check optimization opportunities in an expression
    fn checkForBoundsOptimizations(self: *Optimizer, expr: *ast.ExprNode, transformations: *u32, checks_eliminated: *u32) !void {
        switch (expr.*) {
            .Index => |*index| {
                // Check if index is statically proven to be in bounds
                if (self.isIndexInBounds(index)) {
                    try self.recordOptimization(.BoundsCheckElimination, "Array bounds check eliminated", ast.SourceSpan{ .line = 0, .column = 0, .length = 0 }, OptimizationResult.OptimizationSavings{
                        .gas_saved = 80, // Bounds checks are expensive
                        .instructions_saved = 3,
                        .runtime_checks_eliminated = 1,
                    });
                    transformations.* += 1;
                    checks_eliminated.* += 1;
                }

                // Recursively check sub-expressions
                try self.checkForBoundsOptimizations(index.target, transformations, checks_eliminated);
                try self.checkForBoundsOptimizations(index.index, transformations, checks_eliminated);
            },
            .Binary => |*binary| {
                try self.checkForBoundsOptimizations(binary.lhs, transformations, checks_eliminated);
                try self.checkForBoundsOptimizations(binary.rhs, transformations, checks_eliminated);
            },
            .Unary => |*unary| {
                try self.checkForBoundsOptimizations(unary.operand, transformations, checks_eliminated);
            },
            .Call => |*call| {
                try self.checkForBoundsOptimizations(call.callee, transformations, checks_eliminated);
                for (call.arguments) |*arg| {
                    try self.checkForBoundsOptimizations(arg, transformations, checks_eliminated);
                }
            },
            else => {},
        }
    }

    // Helper methods for optimization analysis

    /// Check if expression is proven at compile time
    fn isProvenAtCompileTime(self: *Optimizer, expr: *const ast.ExprNode) bool {
        // Check if this exact condition was proven by static verification
        const expr_hash = self.hashExpression(expr);
        return self.context.proven_conditions.get(expr_hash) orelse false;
    }

    /// Check if expression is a compile-time constant
    pub fn isCompileTimeConstant(self: *Optimizer, expr: *const ast.ExprNode) bool {
        return switch (expr.*) {
            .Literal => true,
            .Identifier => |*ident| self.context.comptime_constants.contains(ident.name),
            .Binary => |*binary| self.isCompileTimeConstant(binary.lhs) and self.isCompileTimeConstant(binary.rhs),
            .Unary => |*unary| self.isCompileTimeConstant(unary.operand),
            else => false,
        };
    }

    /// Check if expression can be folded to a constant
    pub fn canFoldToConstant(self: *Optimizer, expr: *const ast.ExprNode) bool {
        return self.isCompileTimeConstant(expr);
    }

    /// Check if expression is always true
    pub fn isAlwaysTrue(self: *Optimizer, expr: *const ast.ExprNode) bool {
        return switch (expr.*) {
            .Literal => |*lit| switch (lit.*) {
                .Bool => |*b| b.value == true,
                else => false,
            },
            else => self.isProvenAtCompileTime(expr) and (self.context.proven_conditions.get(self.hashExpression(expr)) orelse false),
        };
    }

    /// Check if expression is always false
    pub fn isAlwaysFalse(self: *Optimizer, expr: *const ast.ExprNode) bool {
        return switch (expr.*) {
            .Literal => |*lit| switch (lit.*) {
                .Bool => |*b| b.value == false,
                else => false,
            },
            else => self.isProvenAtCompileTime(expr) and !(self.context.proven_conditions.get(self.hashExpression(expr)) orelse true),
        };
    }

    /// Check if expression is a tautology
    pub fn isTautology(self: *Optimizer, expr: *const ast.ExprNode) bool {
        return switch (expr.*) {
            .Literal => |*lit| switch (lit.*) {
                .Bool => |*b| b.value == true,
                else => false,
            },
            .Binary => |*binary| {
                // Check for patterns like x == x, x >= x, etc.
                if (binary.operator == .EqualEqual or binary.operator == .LessEqual or binary.operator == .GreaterEqual) {
                    return self.expressionsEqual(binary.lhs, binary.rhs);
                }
                return false;
            },
            else => false,
        };
    }

    /// Check if index access is proven to be in bounds
    fn isIndexInBounds(self: *Optimizer, index: *ast.IndexExpr) bool {
        _ = self;
        _ = index;
        // TODO: Implement sophisticated bounds analysis
        // This would check if:
        // 1. Index is a constant within known array bounds
        // 2. Previous checks guarantee index is in bounds
        // 3. Loop invariants prove bounds safety
        return false;
    }

    /// Check if two expressions are equivalent
    fn expressionsEqual(self: *Optimizer, lhs: *ast.ExprNode, rhs: *ast.ExprNode) bool {
        return switch (lhs.*) {
            .Identifier => |*lhs_ident| switch (rhs.*) {
                .Identifier => |*rhs_ident| std.mem.eql(u8, lhs_ident.name, rhs_ident.name),
                else => false,
            },
            .Literal => |*lhs_lit| switch (rhs.*) {
                .Literal => |*rhs_lit| self.literalsEqual(lhs_lit, rhs_lit),
                else => false,
            },
            else => false, // More sophisticated comparison would be needed for complex expressions
        };
    }

    /// Check if two literals are equal
    fn literalsEqual(self: *Optimizer, lhs: *ast.LiteralNode, rhs: *ast.LiteralNode) bool {
        _ = self;
        return switch (lhs.*) {
            .Integer => |*lhs_int| switch (rhs.*) {
                .Integer => |*rhs_int| std.mem.eql(u8, lhs_int.value, rhs_int.value),
                else => false,
            },
            .Bool => |*lhs_bool| switch (rhs.*) {
                .Bool => |*rhs_bool| lhs_bool.value == rhs_bool.value,
                else => false,
            },
            .String => |*lhs_str| switch (rhs.*) {
                .String => |*rhs_str| std.mem.eql(u8, lhs_str.value, rhs_str.value),
                else => false,
            },
            else => false,
        };
    }

    /// Create a hash for an expression (simplified)
    fn hashExpression(self: *Optimizer, expr: *const ast.ExprNode) []const u8 {
        _ = self;
        _ = expr;
        // TODO: Implement proper expression hashing
        return "expression_hash"; // Placeholder
    }

    /// Record an optimization
    fn recordOptimization(self: *Optimizer, opt_type: OptimizationType, description: []const u8, span: ast.SourceSpan, savings: OptimizationResult.OptimizationSavings) !void {
        try self.optimizations.append(OptimizationResult.OptimizationInfo{
            .type = opt_type,
            .description = description,
            .span = span,
            .savings = savings,
        });
    }

    /// Get total gas savings from optimizations
    pub fn getTotalGasSavings(self: *Optimizer) u64 {
        var total: u64 = 0;
        for (self.optimizations.items) |opt| {
            total += opt.savings.gas_saved;
        }
        return total;
    }

    /// Get total instructions saved
    pub fn getTotalInstructionsSaved(self: *Optimizer) u32 {
        var total: u32 = 0;
        for (self.optimizations.items) |opt| {
            total += opt.savings.instructions_saved;
        }
        return total;
    }

    /// Get total runtime checks eliminated
    pub fn getTotalRuntimeChecksEliminated(self: *Optimizer) u32 {
        var total: u32 = 0;
        for (self.optimizations.items) |opt| {
            total += opt.savings.runtime_checks_eliminated;
        }
        return total;
    }
};

// Extension to AST for getting spans (helper)
const ASTExtensions = struct {
    pub fn getSpan(expr: *ast.ExprNode) ast.SourceSpan {
        return switch (expr.*) {
            .Literal => |*lit| switch (lit.*) {
                .Integer => |*int| int.span,
                .Bool => |*b| b.span,
                .String => |*s| s.span,
                .Address => |*a| a.span,
                .Hex => |*h| h.span,
            },
            .Identifier => |*ident| ast.SourceSpan{ .line = 0, .column = 0, .length = @intCast(ident.name.len) },
            else => ast.SourceSpan{ .line = 0, .column = 0, .length = 0 },
        };
    }
};

// Extend ExprNode with getSpan method
pub fn getExprSpan(expr: *ast.ExprNode) ast.SourceSpan {
    return ASTExtensions.getSpan(expr);
}

// Tests
const testing = std.testing;

test "optimizer initialization" {
    var optimizer = Optimizer.init(testing.allocator);
    defer optimizer.deinit();

    try testing.expect(optimizer.optimizations.items.len == 0);
    try testing.expect(optimizer.context.proven_conditions.count() == 0);
}

test "constant optimization detection" {
    var optimizer = Optimizer.init(testing.allocator);
    defer optimizer.deinit();

    // Add a compile-time constant
    try optimizer.addConstant("MAX_VALUE", comptime_eval.ComptimeValue{ .u64 = 1000 });

    // Create an identifier expression
    var identifier = ast.IdentifierExpr{ .name = "MAX_VALUE" };
    var expr = ast.ExprNode{ .Identifier = &identifier };

    try testing.expect(optimizer.isCompileTimeConstant(&expr));
}

test "tautology detection" {
    var optimizer = Optimizer.init(testing.allocator);
    defer optimizer.deinit();

    // Create a tautology: true
    var true_literal = ast.LiteralNode{ .Bool = ast.expressions.BoolLiteral{ .value = true, .span = ast.SourceSpan{ .line = 1, .column = 1, .length = 4 } } };
    var true_expr = ast.ExprNode{ .Literal = &true_literal };

    try testing.expect(optimizer.isTautology(&true_expr));

    // Create a tautology: x == x
    var identifier1 = ast.IdentifierExpr{ .name = "x" };
    var identifier2 = ast.IdentifierExpr{ .name = "x" };
    var expr1 = ast.ExprNode{ .Identifier = &identifier1 };
    var expr2 = ast.ExprNode{ .Identifier = &identifier2 };

    var binary_expr = ast.BinaryExpr{
        .lhs = &expr1,
        .rhs = &expr2,
        .operator = .EqualEqual,
    };
    var binary_node = ast.ExprNode{ .Binary = &binary_expr };

    try testing.expect(optimizer.isTautology(&binary_node));
}

test "gas savings calculation" {
    var optimizer = Optimizer.init(testing.allocator);
    defer optimizer.deinit();

    // Record some optimizations
    try optimizer.recordOptimization(.RedundantCheckElimination, "Test optimization 1", ast.SourceSpan{ .line = 1, .column = 1, .length = 5 }, OptimizationResult.OptimizationSavings{
        .gas_saved = 100,
        .instructions_saved = 2,
        .runtime_checks_eliminated = 1,
    });

    try optimizer.recordOptimization(.ConstantFolding, "Test optimization 2", ast.SourceSpan{ .line = 2, .column = 1, .length = 3 }, OptimizationResult.OptimizationSavings{
        .gas_saved = 50,
        .instructions_saved = 1,
        .runtime_checks_eliminated = 0,
    });

    try testing.expectEqual(@as(u64, 150), optimizer.getTotalGasSavings());
    try testing.expectEqual(@as(u32, 3), optimizer.getTotalInstructionsSaved());
    try testing.expectEqual(@as(u32, 1), optimizer.getTotalRuntimeChecksEliminated());
}
