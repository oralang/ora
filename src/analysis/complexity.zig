// ============================================================================
// Complexity Analysis
// ============================================================================
//
// Provides complexity metrics for Ora functions to support:
// - Inline function size validation
// - Code quality warnings
// - Future gas estimation
//
// PHASE 1 (ASUKA): Simple AST node counting
// - Count total statements in function body
// - Warn on large inline functions
//
// FUTURE: Deep expression analysis, cyclomatic complexity, gas estimation
// ============================================================================

const std = @import("std");
const ast = @import("../ast.zig");
const Statements = @import("../ast/statements.zig");

/// Complexity metrics for a function
pub const ComplexityMetrics = struct {
    /// Total statements
    node_count: u32 = 0,

    /// Number of statements
    statement_count: u32 = 0,

    /// Maximum nesting depth
    max_depth: u32 = 0,

    /// Check if function is simple (good for inline)
    pub fn isSimple(self: ComplexityMetrics) bool {
        return self.node_count < 20;
    }

    /// Check if function is moderate complexity
    pub fn isModerate(self: ComplexityMetrics) bool {
        return self.node_count >= 20 and self.node_count <= 100;
    }

    /// Check if function is complex (warn for inline)
    pub fn isComplex(self: ComplexityMetrics) bool {
        return self.node_count > 100;
    }

    /// Get human-readable complexity level
    pub fn getLevel(self: ComplexityMetrics) []const u8 {
        if (self.isSimple()) return "Simple";
        if (self.isModerate()) return "Moderate";
        return "Complex";
    }
};

/// Complexity analyzer using visitor pattern
pub const ComplexityAnalyzer = struct {
    allocator: std.mem.Allocator,
    current_depth: u32 = 0,
    metrics: ComplexityMetrics = .{},

    pub fn init(allocator: std.mem.Allocator) ComplexityAnalyzer {
        return .{
            .allocator = allocator,
            .current_depth = 0,
            .metrics = .{},
        };
    }

    /// Analyze a function's complexity
    pub fn analyzeFunction(self: *ComplexityAnalyzer, func: *const ast.FunctionNode) ComplexityMetrics {
        // Reset metrics for new analysis
        self.metrics = .{};
        self.current_depth = 0;

        // Analyze function body
        self.analyzeBlock(func.body);

        return self.metrics;
    }

    /// Analyze a block of statements
    fn analyzeBlock(self: *ComplexityAnalyzer, block: Statements.BlockNode) void {
        self.current_depth += 1;
        if (self.current_depth > self.metrics.max_depth) {
            self.metrics.max_depth = self.current_depth;
        }

        for (block.statements) |*stmt| {
            self.analyzeStatement(stmt);
        }

        self.current_depth -= 1;
    }

    /// Analyze a statement (simplified - just count and recurse into blocks)
    fn analyzeStatement(self: *ComplexityAnalyzer, stmt: *const Statements.StmtNode) void {
        self.metrics.node_count += 1;
        self.metrics.statement_count += 1;

        // Only recurse into nested blocks
        switch (stmt.*) {
            .If => |if_stmt| {
                self.analyzeBlock(if_stmt.then_branch);
                if (if_stmt.else_branch) |else_branch| {
                    self.analyzeBlock(else_branch);
                }
            },
            .While => |while_stmt| {
                self.analyzeBlock(while_stmt.body);
            },
            .ForLoop => |for_stmt| {
                self.analyzeBlock(for_stmt.body);
            },
            .Switch => |switch_stmt| {
                for (switch_stmt.cases) |*case| {
                    switch (case.body) {
                        .Block => |block| self.analyzeBlock(block),
                        .LabeledBlock => |labeled| self.analyzeBlock(labeled.block),
                        else => {}, // Expression - just count as 1 statement
                    }
                }
                if (switch_stmt.default_case) |default_case| {
                    self.analyzeBlock(default_case);
                }
            },
            .TryBlock => |try_stmt| {
                self.analyzeBlock(try_stmt.try_block);
                if (try_stmt.catch_block) |catch_block| {
                    self.analyzeBlock(catch_block.block);
                }
            },
            .LabeledBlock => |labeled| {
                self.analyzeBlock(labeled.block);
            },
            // All other statements - just count, don't recurse
            else => {},
        }
    }
};
