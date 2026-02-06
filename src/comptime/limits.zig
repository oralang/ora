//! Comptime Evaluation Limits
//!
//! Ensures determinism and safety by limiting:
//! - Recursion depth
//! - Loop iterations (secondary guard)
//! - Total steps (primary fuel limit)
//! - Memory allocation
//! - Array length

const std = @import("std");

/// Configuration for comptime evaluation limits
pub const EvalConfig = struct {
    /// Maximum recursion depth for function calls
    max_recursion_depth: u32 = 1000,

    /// Maximum loop iterations (secondary guard)
    max_loop_iterations: u64 = 1_000_000,

    /// Maximum evaluation steps (primary fuel limit, per IR op)
    max_steps: u64 = 10_000_000,

    /// Maximum memory allocation in bytes
    max_memory_bytes: usize = 16 * 1024 * 1024, // 16 MB

    /// Maximum array length
    max_array_length: usize = 65536,

    /// Enable evaluation tracing (routed to diagnostics)
    trace_enabled: bool = false,

    /// Default configuration
    pub const default = EvalConfig{};

    /// Strict configuration (lower limits for testing)
    pub const strict = EvalConfig{
        .max_recursion_depth = 100,
        .max_loop_iterations = 10_000,
        .max_steps = 100_000,
        .max_memory_bytes = 1024 * 1024, // 1 MB
        .max_array_length = 1024,
        .trace_enabled = false,
    };

    /// Permissive configuration (higher limits)
    pub const permissive = EvalConfig{
        .max_recursion_depth = 10000,
        .max_loop_iterations = 100_000_000,
        .max_steps = 1_000_000_000,
        .max_memory_bytes = 256 * 1024 * 1024, // 256 MB
        .max_array_length = 1024 * 1024,
        .trace_enabled = false,
    };
};

/// Statistics tracked during evaluation
pub const EvalStats = struct {
    /// Current recursion depth
    current_recursion_depth: u32 = 0,

    /// Peak recursion depth reached
    peak_recursion_depth: u32 = 0,

    /// Total evaluation steps consumed
    total_steps: u64 = 0,

    /// Total bytes allocated in heap
    allocated_bytes: usize = 0,

    /// Number of heap allocations
    heap_allocations: u64 = 0,

    /// Number of COW clones performed
    cow_clones: u64 = 0,

    /// Reset statistics
    pub fn reset(self: *EvalStats) void {
        self.* = EvalStats{};
    }

    /// Record entering a function
    pub fn enterFunction(self: *EvalStats) void {
        self.current_recursion_depth += 1;
        if (self.current_recursion_depth > self.peak_recursion_depth) {
            self.peak_recursion_depth = self.current_recursion_depth;
        }
    }

    /// Record leaving a function
    pub fn leaveFunction(self: *EvalStats) void {
        if (self.current_recursion_depth > 0) {
            self.current_recursion_depth -= 1;
        }
    }

    /// Record a step
    pub fn recordStep(self: *EvalStats) void {
        self.total_steps += 1;
    }

    /// Record memory allocation
    pub fn recordAllocation(self: *EvalStats, bytes: usize) void {
        self.allocated_bytes += bytes;
        self.heap_allocations += 1;
    }

    /// Record a COW clone
    pub fn recordCowClone(self: *EvalStats) void {
        self.cow_clones += 1;
    }

    /// Format statistics for display
    pub fn format(
        self: EvalStats,
        comptime fmt: []const u8,
        options: std.fmt.FormatOptions,
        writer: anytype,
    ) !void {
        _ = fmt;
        _ = options;

        try writer.print(
            "steps: {d}, depth: {d}/{d}, memory: {d} bytes, allocs: {d}, clones: {d}",
            .{
                self.total_steps,
                self.current_recursion_depth,
                self.peak_recursion_depth,
                self.allocated_bytes,
                self.heap_allocations,
                self.cow_clones,
            },
        );
    }
};

/// Check if a limit has been exceeded
pub const LimitCheck = struct {
    config: EvalConfig,
    stats: *EvalStats,

    pub fn init(config: EvalConfig, stats: *EvalStats) LimitCheck {
        return .{ .config = config, .stats = stats };
    }

    /// Check recursion limit, returns error kind if exceeded
    pub fn checkRecursion(self: LimitCheck) ?@import("error.zig").CtErrorKind {
        if (self.stats.current_recursion_depth >= self.config.max_recursion_depth) {
            return .recursion_limit;
        }
        return null;
    }

    /// Check step limit, returns error kind if exceeded
    pub fn checkSteps(self: LimitCheck) ?@import("error.zig").CtErrorKind {
        if (self.stats.total_steps >= self.config.max_steps) {
            return .step_limit;
        }
        return null;
    }

    /// Check memory limit, returns error kind if exceeded
    pub fn checkMemory(self: LimitCheck, additional_bytes: usize) ?@import("error.zig").CtErrorKind {
        if (self.stats.allocated_bytes + additional_bytes > self.config.max_memory_bytes) {
            return .memory_limit;
        }
        return null;
    }

    /// Check array length limit
    pub fn checkArrayLength(self: LimitCheck, length: usize) ?@import("error.zig").CtErrorKind {
        if (length > self.config.max_array_length) {
            return .memory_limit;
        }
        return null;
    }

    /// Get remaining steps
    pub fn remainingSteps(self: LimitCheck) u64 {
        if (self.stats.total_steps >= self.config.max_steps) return 0;
        return self.config.max_steps - self.stats.total_steps;
    }

    /// Get remaining memory
    pub fn remainingMemory(self: LimitCheck) usize {
        if (self.stats.allocated_bytes >= self.config.max_memory_bytes) return 0;
        return self.config.max_memory_bytes - self.stats.allocated_bytes;
    }
};

// ============================================================================
// Tests
// ============================================================================

test "EvalConfig presets" {
    const default = EvalConfig.default;
    const strict = EvalConfig.strict;

    try std.testing.expect(default.max_steps > strict.max_steps);
    try std.testing.expect(default.max_recursion_depth > strict.max_recursion_depth);
}

test "EvalStats tracking" {
    var stats = EvalStats{};

    stats.enterFunction();
    stats.enterFunction();
    try std.testing.expectEqual(@as(u32, 2), stats.current_recursion_depth);
    try std.testing.expectEqual(@as(u32, 2), stats.peak_recursion_depth);

    stats.leaveFunction();
    try std.testing.expectEqual(@as(u32, 1), stats.current_recursion_depth);
    try std.testing.expectEqual(@as(u32, 2), stats.peak_recursion_depth);

    stats.recordStep();
    stats.recordStep();
    try std.testing.expectEqual(@as(u64, 2), stats.total_steps);

    stats.recordAllocation(1024);
    try std.testing.expectEqual(@as(usize, 1024), stats.allocated_bytes);
    try std.testing.expectEqual(@as(u64, 1), stats.heap_allocations);
}

test "LimitCheck" {
    var stats = EvalStats{};
    const config = EvalConfig{ .max_recursion_depth = 2, .max_steps = 5 };
    const check = LimitCheck.init(config, &stats);

    try std.testing.expectEqual(@as(?@import("error.zig").CtErrorKind, null), check.checkRecursion());

    stats.enterFunction();
    stats.enterFunction();
    try std.testing.expectEqual(@import("error.zig").CtErrorKind.recursion_limit, check.checkRecursion().?);

    try std.testing.expectEqual(@as(u64, 5), check.remainingSteps());
    stats.total_steps = 3;
    try std.testing.expectEqual(@as(u64, 2), check.remainingSteps());
}
