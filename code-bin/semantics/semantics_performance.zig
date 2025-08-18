const std = @import("std");
pub const ast = @import("../ast.zig");
const semantics_errors = @import("semantics_errors.zig");

// Forward declaration for SemanticAnalyzer
const SemanticAnalyzer = @import("semantics_core.zig").SemanticAnalyzer;

/// Performance metrics for semantic analysis
pub const PerformanceMetrics = struct {
    analysis_start_time: i64,
    analysis_end_time: i64,
    total_analysis_time: f64,
    phase_timings: std.EnumMap(AnalysisPhase, PhaseMetrics),
    node_timings: std.EnumMap(std.meta.Tag(ast.AstNode), NodeMetrics),
    memory_metrics: MemoryMetrics,
    throughput_metrics: ThroughputMetrics,

    pub const AnalysisPhase = enum {
        PreInitialization,
        TypeChecking,
        SemanticAnalysis,
        Validation,
        PostProcessing,
    };

    pub const PhaseMetrics = struct {
        start_time: i64,
        end_time: i64,
        duration: f64,
        nodes_processed: u32,
        errors_found: u32,
        warnings_generated: u32,
        memory_used: usize,

        pub fn init() PhaseMetrics {
            return PhaseMetrics{
                .start_time = 0,
                .end_time = 0,
                .duration = 0.0,
                .nodes_processed = 0,
                .errors_found = 0,
                .warnings_generated = 0,
                .memory_used = 0,
            };
        }

        pub fn startPhase(self: *PhaseMetrics) void {
            self.start_time = std.time.nanoTimestamp();
        }

        pub fn endPhase(self: *PhaseMetrics) void {
            self.end_time = std.time.nanoTimestamp();
            self.duration = @as(f64, @floatFromInt(self.end_time - self.start_time)) / 1_000_000_000.0; // Convert to seconds
        }
    };

    pub const NodeMetrics = struct {
        total_count: u32,
        total_time: f64,
        average_time: f64,
        min_time: f64,
        max_time: f64,
        error_count: u32,
        warning_count: u32,

        pub fn init() NodeMetrics {
            return NodeMetrics{
                .total_count = 0,
                .total_time = 0.0,
                .average_time = 0.0,
                .min_time = std.math.inf(f64),
                .max_time = 0.0,
                .error_count = 0,
                .warning_count = 0,
            };
        }

        pub fn recordNodeAnalysis(self: *NodeMetrics, analysis_time: f64, had_error: bool, had_warning: bool) void {
            self.total_count += 1;
            self.total_time += analysis_time;
            self.average_time = self.total_time / @as(f64, @floatFromInt(self.total_count));

            if (analysis_time < self.min_time) {
                self.min_time = analysis_time;
            }
            if (analysis_time > self.max_time) {
                self.max_time = analysis_time;
            }

            if (had_error) {
                self.error_count += 1;
            }
            if (had_warning) {
                self.warning_count += 1;
            }
        }
    };

    pub const MemoryMetrics = struct {
        initial_memory: usize,
        peak_memory: usize,
        final_memory: usize,
        allocations: u32,
        deallocations: u32,
        memory_efficiency: f32,

        pub fn init() MemoryMetrics {
            return MemoryMetrics{
                .initial_memory = 0,
                .peak_memory = 0,
                .final_memory = 0,
                .allocations = 0,
                .deallocations = 0,
                .memory_efficiency = 0.0,
            };
        }

        pub fn recordMemoryUsage(self: *MemoryMetrics, current_memory: usize) void {
            if (current_memory > self.peak_memory) {
                self.peak_memory = current_memory;
            }
            self.final_memory = current_memory;
        }

        pub fn recordAllocation(self: *MemoryMetrics) void {
            self.allocations += 1;
        }

        pub fn recordDeallocation(self: *MemoryMetrics) void {
            self.deallocations += 1;
        }

        pub fn calculateEfficiency(self: *MemoryMetrics) void {
            if (self.peak_memory > 0) {
                self.memory_efficiency = @as(f32, @floatFromInt(self.final_memory)) / @as(f32, @floatFromInt(self.peak_memory)) * 100.0;
            }
        }
    };

    pub const ThroughputMetrics = struct {
        nodes_per_second: f64,
        lines_per_second: f64,
        bytes_per_second: f64,
        total_nodes: u32,
        total_lines: u32,
        total_bytes: usize,

        pub fn init() ThroughputMetrics {
            return ThroughputMetrics{
                .nodes_per_second = 0.0,
                .lines_per_second = 0.0,
                .bytes_per_second = 0.0,
                .total_nodes = 0,
                .total_lines = 0,
                .total_bytes = 0,
            };
        }

        pub fn calculateThroughput(self: *ThroughputMetrics, total_time: f64) void {
            if (total_time > 0.0) {
                self.nodes_per_second = @as(f64, @floatFromInt(self.total_nodes)) / total_time;
                self.lines_per_second = @as(f64, @floatFromInt(self.total_lines)) / total_time;
                self.bytes_per_second = @as(f64, @floatFromInt(self.total_bytes)) / total_time;
            }
        }
    };

    pub fn init() PerformanceMetrics {
        return PerformanceMetrics{
            .analysis_start_time = 0,
            .analysis_end_time = 0,
            .total_analysis_time = 0.0,
            .phase_timings = std.EnumMap(AnalysisPhase, PhaseMetrics).init(.{}),
            .node_timings = std.EnumMap(std.meta.Tag(ast.AstNode), NodeMetrics).init(.{}),
            .memory_metrics = MemoryMetrics.init(),
            .throughput_metrics = ThroughputMetrics.init(),
        };
    }

    pub fn startAnalysis(self: *PerformanceMetrics) void {
        self.analysis_start_time = std.time.nanoTimestamp();
    }

    pub fn endAnalysis(self: *PerformanceMetrics) void {
        self.analysis_end_time = std.time.nanoTimestamp();
        self.total_analysis_time = @as(f64, @floatFromInt(self.analysis_end_time - self.analysis_start_time)) / 1_000_000_000.0;

        // Calculate final throughput metrics
        self.throughput_metrics.calculateThroughput(self.total_analysis_time);
        self.memory_metrics.calculateEfficiency();
    }

    pub fn startPhase(self: *PerformanceMetrics, phase: AnalysisPhase) void {
        var phase_metrics = self.phase_timings.getPtr(phase) orelse blk: {
            self.phase_timings.put(phase, PhaseMetrics.init());
            break :blk self.phase_timings.getPtr(phase).?;
        };
        phase_metrics.startPhase();
    }

    pub fn endPhase(self: *PerformanceMetrics, phase: AnalysisPhase) void {
        if (self.phase_timings.getPtr(phase)) |phase_metrics| {
            phase_metrics.endPhase();
        }
    }

    pub fn recordNodeAnalysis(self: *PerformanceMetrics, node_type: std.meta.Tag(ast.AstNode), analysis_time: f64, had_error: bool, had_warning: bool) void {
        var node_metrics = self.node_timings.getPtr(node_type) orelse blk: {
            self.node_timings.put(node_type, NodeMetrics.init());
            break :blk self.node_timings.getPtr(node_type).?;
        };

        node_metrics.recordNodeAnalysis(analysis_time, had_error, had_warning);
        self.throughput_metrics.total_nodes += 1;
    }
};

/// Performance profiler for semantic analysis
pub const PerformanceProfiler = struct {
    allocator: std.mem.Allocator,
    metrics: PerformanceMetrics,
    profiling_enabled: bool,
    detailed_profiling: bool,
    current_phase: ?PerformanceMetrics.AnalysisPhase,
    node_analysis_stack: std.ArrayList(NodeAnalysisContext),

    const NodeAnalysisContext = struct {
        node_type: std.meta.Tag(ast.AstNode),
        start_time: i64,
        span: ast.SourceSpan,
    };

    pub fn init(allocator: std.mem.Allocator) PerformanceProfiler {
        return PerformanceProfiler{
            .allocator = allocator,
            .metrics = PerformanceMetrics.init(),
            .profiling_enabled = true,
            .detailed_profiling = false,
            .current_phase = null,
            .node_analysis_stack = std.ArrayList(NodeAnalysisContext).init(allocator),
        };
    }

    pub fn deinit(self: *PerformanceProfiler) void {
        self.node_analysis_stack.deinit();
    }

    /// Enable or disable profiling
    pub fn setProfilingEnabled(self: *PerformanceProfiler, enabled: bool) void {
        self.profiling_enabled = enabled;
    }

    /// Enable or disable detailed profiling
    pub fn setDetailedProfiling(self: *PerformanceProfiler, enabled: bool) void {
        self.detailed_profiling = enabled;
    }

    /// Start analysis profiling
    pub fn startAnalysis(self: *PerformanceProfiler) void {
        if (!self.profiling_enabled) return;
        self.metrics.startAnalysis();
    }

    /// End analysis profiling
    pub fn endAnalysis(self: *PerformanceProfiler) void {
        if (!self.profiling_enabled) return;
        self.metrics.endAnalysis();
    }

    /// Start phase profiling
    pub fn startPhase(self: *PerformanceProfiler, phase: PerformanceMetrics.AnalysisPhase) void {
        if (!self.profiling_enabled) return;
        self.current_phase = phase;
        self.metrics.startPhase(phase);
    }

    /// End phase profiling
    pub fn endPhase(self: *PerformanceProfiler, phase: PerformanceMetrics.AnalysisPhase) void {
        if (!self.profiling_enabled) return;
        self.metrics.endPhase(phase);
        if (self.current_phase == phase) {
            self.current_phase = null;
        }
    }

    /// Start node analysis profiling
    pub fn startNodeAnalysis(self: *PerformanceProfiler, node_type: std.meta.Tag(ast.AstNode), span: ast.SourceSpan) !void {
        if (!self.profiling_enabled or !self.detailed_profiling) return;

        const context = NodeAnalysisContext{
            .node_type = node_type,
            .start_time = std.time.nanoTimestamp(),
            .span = span,
        };

        try self.node_analysis_stack.append(context);
    }

    /// End node analysis profiling
    pub fn endNodeAnalysis(self: *PerformanceProfiler, node_type: std.meta.Tag(ast.AstNode), had_error: bool, had_warning: bool) void {
        if (!self.profiling_enabled or !self.detailed_profiling) return;

        if (self.node_analysis_stack.items.len == 0) return;

        const context = self.node_analysis_stack.pop();
        if (context.node_type != node_type) {
            // Mismatched node types - log warning but continue
            return;
        }

        const end_time = std.time.nanoTimestamp();
        const analysis_time = @as(f64, @floatFromInt(end_time - context.start_time)) / 1_000_000_000.0;

        self.metrics.recordNodeAnalysis(node_type, analysis_time, had_error, had_warning);
    }

    /// Record memory usage
    pub fn recordMemoryUsage(self: *PerformanceProfiler, memory_usage: usize) void {
        if (!self.profiling_enabled) return;
        self.metrics.memory_metrics.recordMemoryUsage(memory_usage);
    }

    /// Record allocation
    pub fn recordAllocation(self: *PerformanceProfiler) void {
        if (!self.profiling_enabled) return;
        self.metrics.memory_metrics.recordAllocation();
    }

    /// Record deallocation
    pub fn recordDeallocation(self: *PerformanceProfiler) void {
        if (!self.profiling_enabled) return;
        self.metrics.memory_metrics.recordDeallocation();
    }

    /// Get current metrics
    pub fn getMetrics(self: *PerformanceProfiler) PerformanceMetrics {
        return self.metrics;
    }

    /// Generate performance report
    pub fn generatePerformanceReport(self: *PerformanceProfiler) ![]const u8 {
        var output = std.ArrayList(u8).init(self.allocator);
        defer output.deinit();

        const metrics = &self.metrics;

        try output.writer().print("=== Semantic Analysis Performance Report ===\n\n");
        try output.writer().print("Total Analysis Time: {d:.3}s\n", .{metrics.total_analysis_time});
        try output.writer().print("Total Nodes Processed: {d}\n", .{metrics.throughput_metrics.total_nodes});
        try output.writer().print("Throughput: {d:.1} nodes/sec\n\n", .{metrics.throughput_metrics.nodes_per_second});

        // Phase timings
        try output.appendSlice("Phase Timings:\n");
        const phase_fields = @typeInfo(PerformanceMetrics.AnalysisPhase).Enum.fields;
        for (phase_fields) |field| {
            const phase = @as(PerformanceMetrics.AnalysisPhase, @enumFromInt(field.value));
            if (metrics.phase_timings.get(phase)) |phase_metrics| {
                if (phase_metrics.duration > 0.0) {
                    const percentage = (phase_metrics.duration / metrics.total_analysis_time) * 100.0;
                    try output.writer().print("  {s}: {d:.3}s ({d:.1}%)\n", .{ field.name, phase_metrics.duration, percentage });
                }
            }
        }

        // Memory metrics
        try output.writer().print("\nMemory Usage:\n");
        try output.writer().print("  Peak Memory: {d} bytes\n", .{metrics.memory_metrics.peak_memory});
        try output.writer().print("  Final Memory: {d} bytes\n", .{metrics.memory_metrics.final_memory});
        try output.writer().print("  Memory Efficiency: {d:.1}%\n", .{metrics.memory_metrics.memory_efficiency});
        try output.writer().print("  Allocations: {d}\n", .{metrics.memory_metrics.allocations});
        try output.writer().print("  Deallocations: {d}\n", .{metrics.memory_metrics.deallocations});

        // Top slowest node types
        if (self.detailed_profiling) {
            try output.appendSlice("\nSlowest Node Types:\n");
            var slowest_nodes = std.ArrayList(struct { node_type: std.meta.Tag(ast.AstNode), avg_time: f64 }).init(self.allocator);
            defer slowest_nodes.deinit();

            const node_fields = @typeInfo(std.meta.Tag(ast.AstNode)).Enum.fields;
            for (node_fields) |field| {
                const node_type = @as(std.meta.Tag(ast.AstNode), @enumFromInt(field.value));
                if (metrics.node_timings.get(node_type)) |node_metrics| {
                    if (node_metrics.total_count > 0) {
                        try slowest_nodes.append(.{ .node_type = node_type, .avg_time = node_metrics.average_time });
                    }
                }
            }

            // Sort by average time (descending)
            std.sort.insertion(@TypeOf(slowest_nodes.items[0]), slowest_nodes.items, {}, struct {
                fn lessThan(context: void, a: @TypeOf(slowest_nodes.items[0]), b: @TypeOf(slowest_nodes.items[0])) bool {
                    _ = context;
                    return a.avg_time > b.avg_time;
                }
            }.lessThan);

            // Show top 10
            const max_show = @min(10, slowest_nodes.items.len);
            for (slowest_nodes.items[0..max_show]) |item| {
                const node_metrics = metrics.node_timings.get(item.node_type).?;
                try output.writer().print("  {s}: {d:.6}s avg ({d} nodes)\n", .{ @tagName(item.node_type), item.avg_time, node_metrics.total_count });
            }
        }

        return output.toOwnedSlice();
    }

    /// Get performance summary
    pub fn getPerformanceSummary(self: *PerformanceProfiler) PerformanceSummary {
        const metrics = &self.metrics;
        return PerformanceSummary{
            .total_time = metrics.total_analysis_time,
            .nodes_processed = metrics.throughput_metrics.total_nodes,
            .throughput = metrics.throughput_metrics.nodes_per_second,
            .peak_memory = metrics.memory_metrics.peak_memory,
            .memory_efficiency = metrics.memory_metrics.memory_efficiency,
        };
    }
};

/// Performance summary structure
pub const PerformanceSummary = struct {
    total_time: f64,
    nodes_processed: u32,
    throughput: f64,
    peak_memory: usize,
    memory_efficiency: f32,
};

/// Performance monitoring utilities
pub const PerformanceMonitor = struct {
    profiler: PerformanceProfiler,
    warning_thresholds: PerformanceThresholds,

    pub const PerformanceThresholds = struct {
        max_analysis_time: f64,
        max_node_time: f64,
        max_memory_usage: usize,
        min_throughput: f64,

        pub fn init() PerformanceThresholds {
            return PerformanceThresholds{
                .max_analysis_time = 10.0, // 10 seconds
                .max_node_time = 0.001, // 1ms per node
                .max_memory_usage = 100 * 1024 * 1024, // 100MB
                .min_throughput = 1000.0, // 1000 nodes/sec
            };
        }
    };

    pub fn init(allocator: std.mem.Allocator) PerformanceMonitor {
        return PerformanceMonitor{
            .profiler = PerformanceProfiler.init(allocator),
            .warning_thresholds = PerformanceThresholds.init(),
        };
    }

    pub fn deinit(self: *PerformanceMonitor) void {
        self.profiler.deinit();
    }

    /// Check performance and generate warnings
    pub fn checkPerformance(self: *PerformanceMonitor, analyzer: *SemanticAnalyzer) !void {
        const summary = self.profiler.getPerformanceSummary();

        // Check analysis time
        if (summary.total_time > self.warning_thresholds.max_analysis_time) {
            const warning = try std.fmt.allocPrint(analyzer.allocator, "Analysis took {d:.1}s, exceeding threshold of {d:.1}s", .{ summary.total_time, self.warning_thresholds.max_analysis_time });
            try semantics_errors.addWarning(analyzer, warning, ast.SourceSpan{ .line = 0, .column = 0, .length = 0 });
        }

        // Check memory usage
        if (summary.peak_memory > self.warning_thresholds.max_memory_usage) {
            const warning = try std.fmt.allocPrint(analyzer.allocator, "Peak memory usage {d} bytes exceeds threshold of {d} bytes", .{ summary.peak_memory, self.warning_thresholds.max_memory_usage });
            try semantics_errors.addWarning(analyzer, warning, ast.SourceSpan{ .line = 0, .column = 0, .length = 0 });
        }

        // Check throughput
        if (summary.throughput < self.warning_thresholds.min_throughput and summary.nodes_processed > 100) {
            const warning = try std.fmt.allocPrint(analyzer.allocator, "Analysis throughput {d:.1} nodes/sec below threshold of {d:.1} nodes/sec", .{ summary.throughput, self.warning_thresholds.min_throughput });
            try semantics_errors.addWarning(analyzer, warning, ast.SourceSpan{ .line = 0, .column = 0, .length = 0 });
        }
    }
};

/// Initialize performance monitoring for analyzer
pub fn initializePerformanceMonitoring(analyzer: *SemanticAnalyzer) !*PerformanceMonitor {
    const monitor = try analyzer.allocator.create(PerformanceMonitor);
    monitor.* = PerformanceMonitor.init(analyzer.allocator);
    return monitor;
}

/// Record performance metrics for node analysis
pub fn recordNodePerformance(analyzer: *SemanticAnalyzer, node_type: std.meta.Tag(ast.AstNode), analysis_time_ns: i64) void {
    // Update existing validation coverage with performance info
    analyzer.validation_coverage.validation_stats.nodes_analyzed += 1;

    // Convert nanoseconds to seconds for consistency
    const analysis_time_s = @as(f64, @floatFromInt(analysis_time_ns)) / 1_000_000_000.0;

    // Simple performance tracking - could be enhanced with full profiler integration
    _ = analysis_time_s;
    _ = node_type;
}
