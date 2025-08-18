const std = @import("std");
pub const ast = @import("../ast.zig");
const semantics_errors = @import("semantics_errors.zig");

// Forward declaration for SemanticAnalyzer
const SemanticAnalyzer = @import("semantics_core.zig").SemanticAnalyzer;

/// Coverage tracking for different analysis aspects
pub const CoverageTracker = struct {
    allocator: std.mem.Allocator,
    node_coverage: NodeCoverage,
    validation_coverage: ValidationCoverage,
    rule_coverage: RuleCoverage,
    feature_coverage: FeatureCoverage,
    analysis_metrics: AnalysisMetrics,

    pub const NodeCoverage = struct {
        visited_nodes: std.EnumSet(std.meta.Tag(ast.AstNode)),
        node_counts: std.EnumMap(std.meta.Tag(ast.AstNode), u32),
        unhandled_nodes: std.ArrayList(NodeInfo),

        pub const NodeInfo = struct {
            node_type: std.meta.Tag(ast.AstNode),
            span: ast.SourceSpan,
            context: []const u8,
        };

        pub fn init(allocator: std.mem.Allocator) NodeCoverage {
            return NodeCoverage{
                .visited_nodes = std.EnumSet(std.meta.Tag(ast.AstNode)).initEmpty(),
                .node_counts = std.EnumMap(std.meta.Tag(ast.AstNode), u32).init(.{}),
                .unhandled_nodes = std.ArrayList(NodeInfo).init(allocator),
            };
        }

        pub fn deinit(self: *NodeCoverage) void {
            self.unhandled_nodes.deinit();
        }

        pub fn visitNode(self: *NodeCoverage, node_type: std.meta.Tag(ast.AstNode)) void {
            self.visited_nodes.insert(node_type);
            const current_count = self.node_counts.get(node_type) orelse 0;
            self.node_counts.put(node_type, current_count + 1);
        }

        pub fn markUnhandled(self: *NodeCoverage, node_type: std.meta.Tag(ast.AstNode), span: ast.SourceSpan, context: []const u8) !void {
            try self.unhandled_nodes.append(NodeInfo{
                .node_type = node_type,
                .span = span,
                .context = context,
            });
        }

        pub fn getCoveragePercentage(self: *NodeCoverage) f32 {
            const total_node_types = @typeInfo(std.meta.Tag(ast.AstNode)).Enum.fields.len;
            const visited_count = self.visited_nodes.count();
            return @as(f32, @floatFromInt(visited_count)) / @as(f32, @floatFromInt(total_node_types)) * 100.0;
        }

        pub fn getUnvisitedNodes(self: *NodeCoverage) []std.meta.Tag(ast.AstNode) {
            var unvisited = std.ArrayList(std.meta.Tag(ast.AstNode)).init(self.unhandled_nodes.allocator);
            defer unvisited.deinit();

            const all_node_types = @typeInfo(std.meta.Tag(ast.AstNode)).Enum.fields;
            for (all_node_types) |field| {
                const node_type = @as(std.meta.Tag(ast.AstNode), @enumFromInt(field.value));
                if (!self.visited_nodes.contains(node_type)) {
                    unvisited.append(node_type) catch {};
                }
            }

            return unvisited.toOwnedSlice() catch &[_]std.meta.Tag(ast.AstNode){};
        }
    };

    pub const ValidationCoverage = struct {
        validation_rules: std.HashMap([]const u8, ValidationRuleInfo, std.hash_map.StringContext, std.hash_map.default_max_load_percentage),
        rule_executions: std.HashMap([]const u8, u32, std.hash_map.StringContext, std.hash_map.default_max_load_percentage),
        skipped_validations: std.ArrayList(SkippedValidation),

        pub const ValidationRuleInfo = struct {
            rule_name: []const u8,
            rule_category: RuleCategory,
            description: []const u8,
            is_critical: bool,
            execution_count: u32,

            pub const RuleCategory = enum {
                Syntax,
                Semantic,
                Type,
                Memory,
                Security,
                Performance,
                Style,
            };
        };

        pub const SkippedValidation = struct {
            rule_name: []const u8,
            reason: []const u8,
            span: ast.SourceSpan,
            node_type: std.meta.Tag(ast.AstNode),
        };

        pub fn init(allocator: std.mem.Allocator) ValidationCoverage {
            return ValidationCoverage{
                .validation_rules = std.HashMap([]const u8, ValidationRuleInfo, std.hash_map.StringContext, std.hash_map.default_max_load_percentage).init(allocator),
                .rule_executions = std.HashMap([]const u8, u32, std.hash_map.StringContext, std.hash_map.default_max_load_percentage).init(allocator),
                .skipped_validations = std.ArrayList(SkippedValidation).init(allocator),
            };
        }

        pub fn deinit(self: *ValidationCoverage) void {
            self.validation_rules.deinit();
            self.rule_executions.deinit();
            self.skipped_validations.deinit();
        }

        pub fn registerRule(self: *ValidationCoverage, rule_info: ValidationRuleInfo) !void {
            try self.validation_rules.put(rule_info.rule_name, rule_info);
            try self.rule_executions.put(rule_info.rule_name, 0);
        }

        pub fn executeRule(self: *ValidationCoverage, rule_name: []const u8) void {
            const current_count = self.rule_executions.get(rule_name) orelse 0;
            self.rule_executions.put(rule_name, current_count + 1) catch {};
        }

        pub fn skipValidation(self: *ValidationCoverage, rule_name: []const u8, reason: []const u8, span: ast.SourceSpan, node_type: std.meta.Tag(ast.AstNode)) !void {
            try self.skipped_validations.append(SkippedValidation{
                .rule_name = rule_name,
                .reason = reason,
                .span = span,
                .node_type = node_type,
            });
        }

        pub fn getExecutedRules(self: *ValidationCoverage) [][]const u8 {
            var executed = std.ArrayList([]const u8).init(self.skipped_validations.allocator);
            defer executed.deinit();

            var iterator = self.rule_executions.iterator();
            while (iterator.next()) |entry| {
                if (entry.value_ptr.* > 0) {
                    executed.append(entry.key_ptr.*) catch {};
                }
            }

            return executed.toOwnedSlice() catch &[_][]const u8{};
        }

        pub fn getUnexecutedRules(self: *ValidationCoverage) [][]const u8 {
            var unexecuted = std.ArrayList([]const u8).init(self.skipped_validations.allocator);
            defer unexecuted.deinit();

            var iterator = self.rule_executions.iterator();
            while (iterator.next()) |entry| {
                if (entry.value_ptr.* == 0) {
                    unexecuted.append(entry.key_ptr.*) catch {};
                }
            }

            return unexecuted.toOwnedSlice() catch &[_][]const u8{};
        }
    };

    pub const RuleCoverage = struct {
        semantic_rules: std.HashMap([]const u8, RuleExecutionInfo, std.hash_map.StringContext, std.hash_map.default_max_load_percentage),
        type_rules: std.HashMap([]const u8, RuleExecutionInfo, std.hash_map.StringContext, std.hash_map.default_max_load_percentage),
        memory_rules: std.HashMap([]const u8, RuleExecutionInfo, std.hash_map.StringContext, std.hash_map.default_max_load_percentage),

        pub const RuleExecutionInfo = struct {
            rule_name: []const u8,
            execution_count: u32,
            success_count: u32,
            failure_count: u32,
            average_execution_time: f64,
            last_execution_time: i64,
        };

        pub fn init(allocator: std.mem.Allocator) RuleCoverage {
            return RuleCoverage{
                .semantic_rules = std.HashMap([]const u8, RuleExecutionInfo, std.hash_map.StringContext, std.hash_map.default_max_load_percentage).init(allocator),
                .type_rules = std.HashMap([]const u8, RuleExecutionInfo, std.hash_map.StringContext, std.hash_map.default_max_load_percentage).init(allocator),
                .memory_rules = std.HashMap([]const u8, RuleExecutionInfo, std.hash_map.StringContext, std.hash_map.default_max_load_percentage).init(allocator),
            };
        }

        pub fn deinit(self: *RuleCoverage) void {
            self.semantic_rules.deinit();
            self.type_rules.deinit();
            self.memory_rules.deinit();
        }

        pub fn recordRuleExecution(self: *RuleCoverage, rule_category: RuleCategory, rule_name: []const u8, success: bool, execution_time: f64) !void {
            const rules_map = switch (rule_category) {
                .Semantic => &self.semantic_rules,
                .Type => &self.type_rules,
                .Memory => &self.memory_rules,
            };

            const result = try rules_map.getOrPut(rule_name);
            if (!result.found_existing) {
                result.value_ptr.* = RuleExecutionInfo{
                    .rule_name = rule_name,
                    .execution_count = 0,
                    .success_count = 0,
                    .failure_count = 0,
                    .average_execution_time = 0.0,
                    .last_execution_time = std.time.timestamp(),
                };
            }

            var info = result.value_ptr;
            info.execution_count += 1;
            if (success) {
                info.success_count += 1;
            } else {
                info.failure_count += 1;
            }

            // Update average execution time
            info.average_execution_time = (info.average_execution_time * @as(f64, @floatFromInt(info.execution_count - 1)) + execution_time) / @as(f64, @floatFromInt(info.execution_count));
            info.last_execution_time = std.time.timestamp();
        }

        pub const RuleCategory = enum {
            Semantic,
            Type,
            Memory,
        };
    };

    pub const FeatureCoverage = struct {
        language_features: std.HashMap([]const u8, FeatureInfo, std.hash_map.StringContext, std.hash_map.default_max_load_percentage),
        tested_features: std.ArrayList([]const u8),
        untested_features: std.ArrayList([]const u8),

        pub const FeatureInfo = struct {
            feature_name: []const u8,
            feature_category: FeatureCategory,
            is_tested: bool,
            test_count: u32,
            complexity_score: f32,

            pub const FeatureCategory = enum {
                Core,
                Advanced,
                Experimental,
                Deprecated,
            };
        };

        pub fn init(allocator: std.mem.Allocator) FeatureCoverage {
            var coverage = FeatureCoverage{
                .language_features = std.HashMap([]const u8, FeatureInfo, std.hash_map.StringContext, std.hash_map.default_max_load_percentage).init(allocator),
                .tested_features = std.ArrayList([]const u8).init(allocator),
                .untested_features = std.ArrayList([]const u8).init(allocator),
            };

            // Initialize with known language features
            coverage.initializeLanguageFeatures() catch {};
            return coverage;
        }

        pub fn deinit(self: *FeatureCoverage) void {
            self.language_features.deinit();
            self.tested_features.deinit();
            self.untested_features.deinit();
        }

        fn initializeLanguageFeatures(self: *FeatureCoverage) !void {
            const features = [_]FeatureInfo{
                .{ .feature_name = "contracts", .feature_category = .Core, .is_tested = false, .test_count = 0, .complexity_score = 0.8 },
                .{ .feature_name = "functions", .feature_category = .Core, .is_tested = false, .test_count = 0, .complexity_score = 0.6 },
                .{ .feature_name = "variables", .feature_category = .Core, .is_tested = false, .test_count = 0, .complexity_score = 0.4 },
                .{ .feature_name = "structs", .feature_category = .Core, .is_tested = false, .test_count = 0, .complexity_score = 0.7 },
                .{ .feature_name = "enums", .feature_category = .Core, .is_tested = false, .test_count = 0, .complexity_score = 0.5 },
                .{ .feature_name = "mappings", .feature_category = .Core, .is_tested = false, .test_count = 0, .complexity_score = 0.6 },
                .{ .feature_name = "arrays", .feature_category = .Core, .is_tested = false, .test_count = 0, .complexity_score = 0.5 },
                .{ .feature_name = "error_handling", .feature_category = .Advanced, .is_tested = false, .test_count = 0, .complexity_score = 0.9 },
                .{ .feature_name = "generics", .feature_category = .Advanced, .is_tested = false, .test_count = 0, .complexity_score = 1.0 },
                .{ .feature_name = "formal_verification", .feature_category = .Advanced, .is_tested = false, .test_count = 0, .complexity_score = 1.0 },
                .{ .feature_name = "memory_regions", .feature_category = .Advanced, .is_tested = false, .test_count = 0, .complexity_score = 0.8 },
                .{ .feature_name = "imports", .feature_category = .Core, .is_tested = false, .test_count = 0, .complexity_score = 0.6 },
            };

            for (features) |feature| {
                try self.language_features.put(feature.feature_name, feature);
                try self.untested_features.append(feature.feature_name);
            }
        }

        pub fn markFeatureTested(self: *FeatureCoverage, feature_name: []const u8) !void {
            if (self.language_features.getPtr(feature_name)) |feature| {
                if (!feature.is_tested) {
                    feature.is_tested = true;
                    try self.tested_features.append(feature_name);

                    // Remove from untested features
                    for (self.untested_features.items, 0..) |untested, i| {
                        if (std.mem.eql(u8, untested, feature_name)) {
                            _ = self.untested_features.swapRemove(i);
                            break;
                        }
                    }
                }
                feature.test_count += 1;
            }
        }

        pub fn getFeatureCoveragePercentage(self: *FeatureCoverage) f32 {
            const total_features = self.language_features.count();
            const tested_features = self.tested_features.items.len;
            if (total_features == 0) return 0.0;
            return @as(f32, @floatFromInt(tested_features)) / @as(f32, @floatFromInt(total_features)) * 100.0;
        }
    };

    pub const AnalysisMetrics = struct {
        total_nodes_analyzed: u32,
        total_analysis_time: f64,
        average_node_analysis_time: f64,
        memory_usage_peak: usize,
        error_recovery_attempts: u32,
        successful_recoveries: u32,

        pub fn init() AnalysisMetrics {
            return AnalysisMetrics{
                .total_nodes_analyzed = 0,
                .total_analysis_time = 0.0,
                .average_node_analysis_time = 0.0,
                .memory_usage_peak = 0,
                .error_recovery_attempts = 0,
                .successful_recoveries = 0,
            };
        }

        pub fn recordNodeAnalysis(self: *AnalysisMetrics, analysis_time: f64) void {
            self.total_nodes_analyzed += 1;
            self.total_analysis_time += analysis_time;
            self.average_node_analysis_time = self.total_analysis_time / @as(f64, @floatFromInt(self.total_nodes_analyzed));
        }

        pub fn recordMemoryUsage(self: *AnalysisMetrics, memory_usage: usize) void {
            if (memory_usage > self.memory_usage_peak) {
                self.memory_usage_peak = memory_usage;
            }
        }

        pub fn recordErrorRecovery(self: *AnalysisMetrics, successful: bool) void {
            self.error_recovery_attempts += 1;
            if (successful) {
                self.successful_recoveries += 1;
            }
        }

        pub fn getErrorRecoveryRate(self: *AnalysisMetrics) f32 {
            if (self.error_recovery_attempts == 0) return 0.0;
            return @as(f32, @floatFromInt(self.successful_recoveries)) / @as(f32, @floatFromInt(self.error_recovery_attempts)) * 100.0;
        }
    };

    pub fn init(allocator: std.mem.Allocator) CoverageTracker {
        return CoverageTracker{
            .allocator = allocator,
            .node_coverage = NodeCoverage.init(allocator),
            .validation_coverage = ValidationCoverage.init(allocator),
            .rule_coverage = RuleCoverage.init(allocator),
            .feature_coverage = FeatureCoverage.init(allocator),
            .analysis_metrics = AnalysisMetrics.init(),
        };
    }

    pub fn deinit(self: *CoverageTracker) void {
        self.node_coverage.deinit();
        self.validation_coverage.deinit();
        self.rule_coverage.deinit();
        self.feature_coverage.deinit();
    }

    /// Record node visit for coverage tracking
    pub fn visitNode(self: *CoverageTracker, node_type: std.meta.Tag(ast.AstNode), analysis_time: f64) void {
        self.node_coverage.visitNode(node_type);
        self.analysis_metrics.recordNodeAnalysis(analysis_time);
    }

    /// Record validation rule execution
    pub fn executeValidationRule(self: *CoverageTracker, rule_name: []const u8) void {
        self.validation_coverage.executeRule(rule_name);
    }

    /// Record feature usage
    pub fn recordFeatureUsage(self: *CoverageTracker, feature_name: []const u8) !void {
        try self.feature_coverage.markFeatureTested(feature_name);
    }

    /// Generate coverage report
    pub fn generateCoverageReport(self: *CoverageTracker) !CoverageReport {
        return CoverageReport{
            .node_coverage_percentage = self.node_coverage.getCoveragePercentage(),
            .feature_coverage_percentage = self.feature_coverage.getFeatureCoveragePercentage(),
            .total_nodes_analyzed = self.analysis_metrics.total_nodes_analyzed,
            .total_analysis_time = self.analysis_metrics.total_analysis_time,
            .error_recovery_rate = self.analysis_metrics.getErrorRecoveryRate(),
            .unvisited_nodes = self.node_coverage.getUnvisitedNodes(),
            .unexecuted_rules = self.validation_coverage.getUnexecutedRules(),
            .untested_features = self.feature_coverage.untested_features.items,
        };
    }
};

/// Coverage report structure
pub const CoverageReport = struct {
    node_coverage_percentage: f32,
    feature_coverage_percentage: f32,
    total_nodes_analyzed: u32,
    total_analysis_time: f64,
    error_recovery_rate: f32,
    unvisited_nodes: []std.meta.Tag(ast.AstNode),
    unexecuted_rules: [][]const u8,
    untested_features: [][]const u8,

    pub fn format(self: CoverageReport, allocator: std.mem.Allocator) ![]const u8 {
        var output = std.ArrayList(u8).init(allocator);
        defer output.deinit();

        try output.writer().print("=== Semantic Analysis Coverage Report ===\n\n");
        try output.writer().print("Node Coverage: {d:.1}%\n", .{self.node_coverage_percentage});
        try output.writer().print("Feature Coverage: {d:.1}%\n", .{self.feature_coverage_percentage});
        try output.writer().print("Total Nodes Analyzed: {d}\n", .{self.total_nodes_analyzed});
        try output.writer().print("Total Analysis Time: {d:.3}s\n", .{self.total_analysis_time});
        try output.writer().print("Error Recovery Rate: {d:.1}%\n\n", .{self.error_recovery_rate});

        if (self.unvisited_nodes.len > 0) {
            try output.appendSlice("Unvisited Node Types:\n");
            for (self.unvisited_nodes) |node_type| {
                try output.writer().print("  - {s}\n", .{@tagName(node_type)});
            }
            try output.appendSlice("\n");
        }

        if (self.unexecuted_rules.len > 0) {
            try output.appendSlice("Unexecuted Validation Rules:\n");
            for (self.unexecuted_rules) |rule| {
                try output.writer().print("  - {s}\n", .{rule});
            }
            try output.appendSlice("\n");
        }

        if (self.untested_features.len > 0) {
            try output.appendSlice("Untested Language Features:\n");
            for (self.untested_features) |feature| {
                try output.writer().print("  - {s}\n", .{feature});
            }
        }

        return output.toOwnedSlice();
    }
};

/// Initialize coverage tracking for analyzer
pub fn initializeCoverageTracking(analyzer: *SemanticAnalyzer) !*CoverageTracker {
    const tracker = try analyzer.allocator.create(CoverageTracker);
    tracker.* = CoverageTracker.init(analyzer.allocator);
    return tracker;
}

/// Record node analysis for coverage
pub fn recordNodeAnalysis(analyzer: *SemanticAnalyzer, node_type: std.meta.Tag(ast.AstNode)) void {
    // Update the existing validation coverage in the analyzer
    analyzer.validation_coverage.visited_node_types.insert(node_type);
    analyzer.validation_coverage.validation_stats.nodes_analyzed += 1;
}

/// Generate coverage report for analyzer
pub fn generateAnalyzerCoverageReport(analyzer: *SemanticAnalyzer) ![]const u8 {
    var output = std.ArrayList(u8).init(analyzer.allocator);
    defer output.deinit();

    const visited_count = analyzer.validation_coverage.visited_node_types.count();
    const total_node_types = @typeInfo(std.meta.Tag(ast.AstNode)).Enum.fields.len;
    const coverage_percentage = @as(f32, @floatFromInt(visited_count)) / @as(f32, @floatFromInt(total_node_types)) * 100.0;

    try output.writer().print("=== Semantic Analysis Coverage Report ===\n\n");
    try output.writer().print("Node Coverage: {d:.1}% ({d}/{d})\n", .{ coverage_percentage, visited_count, total_node_types });
    try output.writer().print("Nodes Analyzed: {d}\n", .{analyzer.validation_coverage.validation_stats.nodes_analyzed});
    try output.writer().print("Errors Found: {d}\n", .{analyzer.validation_coverage.validation_stats.errors_found});
    try output.writer().print("Warnings Generated: {d}\n", .{analyzer.validation_coverage.validation_stats.warnings_generated});
    try output.writer().print("Recovery Attempts: {d}\n", .{analyzer.validation_coverage.validation_stats.recovery_attempts});

    return output.toOwnedSlice();
}
