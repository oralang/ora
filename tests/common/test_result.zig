//! Test Result Types - Consistent test result reporting
//!
//! Provides standardized result types for test execution, failure reporting,
//! and performance benchmarking.

const std = @import("std");
const Allocator = std.mem.Allocator;

/// Test execution result
pub const TestResult = union(enum) {
    pass: void,
    fail: TestFailure,
    skip: []const u8, // reason

    /// Create a passing test result
    pub fn passed() TestResult {
        return TestResult{ .pass = {} };
    }

    /// Create a failing test result
    pub fn failed(failure: TestFailure) TestResult {
        return TestResult{ .fail = failure };
    }

    /// Create a skipped test result
    pub fn skipped(reason: []const u8) TestResult {
        return TestResult{ .skip = reason };
    }

    /// Check if the test passed
    pub fn isPassed(self: TestResult) bool {
        return switch (self) {
            .pass => true,
            else => false,
        };
    }

    /// Check if the test failed
    pub fn isFailed(self: TestResult) bool {
        return switch (self) {
            .fail => true,
            else => false,
        };
    }

    /// Check if the test was skipped
    pub fn isSkipped(self: TestResult) bool {
        return switch (self) {
            .skip => true,
            else => false,
        };
    }

    /// Get failure information if test failed
    pub fn getFailure(self: TestResult) ?TestFailure {
        return switch (self) {
            .fail => |failure| failure,
            else => null,
        };
    }

    /// Format the test result for display
    pub fn format(self: TestResult, comptime fmt: []const u8, options: std.fmt.FormatOptions, writer: anytype) !void {
        _ = fmt;
        _ = options;

        switch (self) {
            .pass => try writer.writeAll("PASS"),
            .fail => |failure| {
                try writer.writeAll("FAIL: ");
                try writer.writeAll(failure.message);
            },
            .skip => |reason| {
                try writer.writeAll("SKIP: ");
                try writer.writeAll(reason);
            },
        }
    }
};

/// Test failure information
pub const TestFailure = struct {
    message: []const u8,
    expected: ?[]const u8 = null,
    actual: ?[]const u8 = null,
    source_location: ?SourceLocation = null,
    diff: ?[]const u8 = null,

    /// Create a simple test failure
    pub fn simple(message: []const u8) TestFailure {
        return TestFailure{
            .message = message,
        };
    }

    /// Create a comparison test failure
    pub fn comparison(message: []const u8, expected: []const u8, actual: []const u8) TestFailure {
        return TestFailure{
            .message = message,
            .expected = expected,
            .actual = actual,
        };
    }

    /// Create a test failure with source location
    pub fn withLocation(message: []const u8, location: SourceLocation) TestFailure {
        return TestFailure{
            .message = message,
            .source_location = location,
        };
    }

    /// Create a test failure with diff
    pub fn withDiff(message: []const u8, expected: []const u8, actual: []const u8, diff: []const u8) TestFailure {
        return TestFailure{
            .message = message,
            .expected = expected,
            .actual = actual,
            .diff = diff,
        };
    }

    /// Format the test failure for display
    pub fn format(self: TestFailure, comptime fmt: []const u8, options: std.fmt.FormatOptions, writer: anytype) !void {
        _ = fmt;
        _ = options;

        try writer.writeAll(self.message);

        if (self.source_location) |location| {
            try writer.print(" at {d}:{d}", .{ location.line, location.column });
        }

        if (self.expected) |expected| {
            try writer.writeAll("\nExpected: ");
            try writer.writeAll(expected);
        }

        if (self.actual) |actual| {
            try writer.writeAll("\nActual: ");
            try writer.writeAll(actual);
        }

        if (self.diff) |diff| {
            try writer.writeAll("\nDiff:\n");
            try writer.writeAll(diff);
        }
    }
};

/// Source location information
pub const SourceLocation = struct {
    file: []const u8,
    line: u32,
    column: u32,

    pub fn format(self: SourceLocation, comptime fmt: []const u8, options: std.fmt.FormatOptions, writer: anytype) !void {
        _ = fmt;
        _ = options;
        try writer.print("{s}:{d}:{d}", .{ self.file, self.line, self.column });
    }
};

/// Benchmark result for performance tests
pub const BenchmarkResult = struct {
    duration_ns: u64,
    memory_bytes: u64,
    iterations: u32,
    throughput_ops_per_sec: f64,

    /// Calculate operations per second
    pub fn calculateThroughput(duration_ns: u64, iterations: u32) f64 {
        if (duration_ns == 0) return 0.0;
        const duration_sec = @as(f64, @floatFromInt(duration_ns)) / 1_000_000_000.0;
        return @as(f64, @floatFromInt(iterations)) / duration_sec;
    }

    /// Create a benchmark result
    pub fn init(duration_ns: u64, memory_bytes: u64, iterations: u32) BenchmarkResult {
        return BenchmarkResult{
            .duration_ns = duration_ns,
            .memory_bytes = memory_bytes,
            .iterations = iterations,
            .throughput_ops_per_sec = calculateThroughput(duration_ns, iterations),
        };
    }

    /// Format the benchmark result for display
    pub fn format(self: BenchmarkResult, comptime fmt: []const u8, options: std.fmt.FormatOptions, writer: anytype) !void {
        _ = fmt;
        _ = options;

        const duration_ms = @as(f64, @floatFromInt(self.duration_ns)) / 1_000_000.0;
        const memory_mb = @as(f64, @floatFromInt(self.memory_bytes)) / (1024.0 * 1024.0);

        try writer.print("{d:.2}ms, {d:.2}MB, {d:.0} ops/sec ({d} iterations)", .{
            duration_ms,
            memory_mb,
            self.throughput_ops_per_sec,
            self.iterations,
        });
    }
};

/// Memory usage result
pub const MemoryUsageResult = struct {
    peak_bytes: u64,
    final_bytes: u64,
    allocations: u32,
    deallocations: u32,
    leaks: []MemoryLeak,

    /// Check if there are memory leaks
    pub fn hasLeaks(self: MemoryUsageResult) bool {
        return self.leaks.len > 0;
    }

    /// Get the number of leaked bytes
    pub fn getLeakedBytes(self: MemoryUsageResult) u64 {
        var total: u64 = 0;
        for (self.leaks) |leak| {
            total += leak.size;
        }
        return total;
    }

    /// Format the memory usage result for display
    pub fn format(self: MemoryUsageResult, comptime fmt: []const u8, options: std.fmt.FormatOptions, writer: anytype) !void {
        _ = fmt;
        _ = options;

        const peak_mb = @as(f64, @floatFromInt(self.peak_bytes)) / (1024.0 * 1024.0);
        const final_mb = @as(f64, @floatFromInt(self.final_bytes)) / (1024.0 * 1024.0);

        try writer.print("Peak: {d:.2}MB, Final: {d:.2}MB, Allocs: {d}, Deallocs: {d}", .{
            peak_mb,
            final_mb,
            self.allocations,
            self.deallocations,
        });

        if (self.leaks.len > 0) {
            const leaked_mb = @as(f64, @floatFromInt(self.getLeakedBytes())) / (1024.0 * 1024.0);
            try writer.print(", Leaks: {d} ({d:.2}MB)", .{ self.leaks.len, leaked_mb });
        }
    }
};

/// Memory leak information
pub const MemoryLeak = struct {
    address: usize,
    size: usize,
    timestamp: i64,

    pub fn format(self: MemoryLeak, comptime fmt: []const u8, options: std.fmt.FormatOptions, writer: anytype) !void {
        _ = fmt;
        _ = options;
        try writer.print("0x{x}: {d} bytes", .{ self.address, self.size });
    }
};

/// Test suite result aggregation
pub const TestSuiteResult = struct {
    name: []const u8,
    total_tests: u32,
    passed_tests: u32,
    failed_tests: u32,
    skipped_tests: u32,
    duration_ms: f64,
    failures: []TestFailure,

    /// Calculate success rate
    pub fn getSuccessRate(self: TestSuiteResult) f64 {
        if (self.total_tests == 0) return 0.0;
        return @as(f64, @floatFromInt(self.passed_tests)) / @as(f64, @floatFromInt(self.total_tests));
    }

    /// Check if all tests passed
    pub fn allPassed(self: TestSuiteResult) bool {
        return self.failed_tests == 0 and self.passed_tests == self.total_tests - self.skipped_tests;
    }

    /// Format the test suite result for display
    pub fn format(self: TestSuiteResult, comptime fmt: []const u8, options: std.fmt.FormatOptions, writer: anytype) !void {
        _ = fmt;
        _ = options;

        const success_rate = self.getSuccessRate() * 100.0;

        try writer.print("{s}: {d}/{d} passed ({d:.1}%), {d} failed, {d} skipped ({d:.2}ms)", .{
            self.name,
            self.passed_tests,
            self.total_tests,
            success_rate,
            self.failed_tests,
            self.skipped_tests,
            self.duration_ms,
        });
    }
};

// Tests
test "TestResult basic functionality" {
    const pass_result = TestResult.passed();
    try std.testing.expect(pass_result.isPassed());
    try std.testing.expect(!pass_result.isFailed());
    try std.testing.expect(!pass_result.isSkipped());

    const fail_result = TestResult.failed(TestFailure.simple("test failed"));
    try std.testing.expect(!fail_result.isPassed());
    try std.testing.expect(fail_result.isFailed());
    try std.testing.expect(!fail_result.isSkipped());

    const skip_result = TestResult.skipped("not implemented");
    try std.testing.expect(!skip_result.isPassed());
    try std.testing.expect(!skip_result.isFailed());
    try std.testing.expect(skip_result.isSkipped());
}

test "BenchmarkResult calculations" {
    const result = BenchmarkResult.init(1_000_000_000, 1024 * 1024, 1000); // 1 second, 1MB, 1000 iterations
    try std.testing.expect(result.throughput_ops_per_sec == 1000.0);
}

test "MemoryUsageResult leak detection" {
    const leaks = [_]MemoryLeak{
        MemoryLeak{ .address = 0x1000, .size = 100, .timestamp = 0 },
        MemoryLeak{ .address = 0x2000, .size = 200, .timestamp = 0 },
    };

    const result = MemoryUsageResult{
        .peak_bytes = 1024,
        .final_bytes = 300,
        .allocations = 10,
        .deallocations = 8,
        .leaks = @constCast(&leaks),
    };

    try std.testing.expect(result.hasLeaks());
    try std.testing.expect(result.getLeakedBytes() == 300);
}
