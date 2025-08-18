//! Test Runner - Base test execution with parallel support
//!
//! Provides a flexible test runner that can execute tests in parallel,
//! collect results, and generate comprehensive reports.

const std = @import("std");
const Allocator = std.mem.Allocator;
const TestResult = @import("test_result.zig").TestResult;
const TestFailure = @import("test_result.zig").TestFailure;
const TestSuiteResult = @import("test_result.zig").TestSuiteResult;
const TestArena = @import("test_arena.zig").TestArena;

/// Test function signature
pub const TestFn = *const fn (allocator: Allocator) anyerror!TestResult;

/// Test case definition
pub const TestCase = struct {
    name: []const u8,
    test_fn: TestFn,
    timeout_ms: ?u32 = null,
    skip: bool = false,
    skip_reason: ?[]const u8 = null,
};

/// Test runner configuration
pub const TestRunnerConfig = struct {
    /// Enable parallel test execution
    parallel_execution: bool = true,
    /// Maximum number of concurrent tests
    max_concurrent_tests: u32 = 0, // 0 = use CPU count
    /// Default timeout for tests in milliseconds
    default_timeout_ms: u32 = 30000,
    /// Memory limit per test in bytes
    memory_limit_bytes: u64 = 1024 * 1024 * 1024, // 1GB
    /// Enable memory tracking for leak detection
    enable_memory_tracking: bool = true,
    /// Verbose output
    verbose: bool = false,
};

/// Test runner for executing test suites
pub const TestRunner = struct {
    allocator: Allocator,
    config: TestRunnerConfig,

    /// Initialize a new test runner
    pub fn init(allocator: Allocator, config: TestRunnerConfig) TestRunner {
        var final_config = config;
        if (final_config.max_concurrent_tests == 0) {
            final_config.max_concurrent_tests = @intCast(std.Thread.getCpuCount() catch 4);
        }

        return TestRunner{
            .allocator = allocator,
            .config = final_config,
        };
    }

    /// Run a test suite
    pub fn runSuite(self: *TestRunner, suite_name: []const u8, test_cases: []const TestCase) !TestSuiteResult {
        const start_time = std.time.milliTimestamp();

        var results = std.ArrayList(TestResult).init(self.allocator);
        defer results.deinit();

        var failures = std.ArrayList(TestFailure).init(self.allocator);
        defer failures.deinit();

        if (self.config.parallel_execution and test_cases.len > 1) {
            try self.runTestsParallel(test_cases, &results, &failures);
        } else {
            try self.runTestsSequential(test_cases, &results, &failures);
        }

        const end_time = std.time.milliTimestamp();
        const duration_ms = @as(f64, @floatFromInt(end_time - start_time));

        // Count results
        var passed: u32 = 0;
        var failed: u32 = 0;
        var skipped: u32 = 0;

        for (results.items) |result| {
            switch (result) {
                .pass => passed += 1,
                .fail => failed += 1,
                .skip => skipped += 1,
            }
        }

        return TestSuiteResult{
            .name = suite_name,
            .total_tests = @intCast(test_cases.len),
            .passed_tests = passed,
            .failed_tests = failed,
            .skipped_tests = skipped,
            .duration_ms = duration_ms,
            .failures = try failures.toOwnedSlice(),
        };
    }

    /// Run tests sequentially
    fn runTestsSequential(self: *TestRunner, test_cases: []const TestCase, results: *std.ArrayList(TestResult), failures: *std.ArrayList(TestFailure)) !void {
        for (test_cases) |test_case| {
            const result = try self.runSingleTest(test_case);
            try results.append(result);

            if (result.getFailure()) |failure| {
                try failures.append(failure);
            }

            if (self.config.verbose) {
                std.log.info("Test '{s}': {}", .{ test_case.name, result });
            }
        }
    }

    /// Run tests in parallel
    fn runTestsParallel(self: *TestRunner, test_cases: []const TestCase, results: *std.ArrayList(TestResult), failures: *std.ArrayList(TestFailure)) !void {
        // For now, implement sequential execution as parallel execution is complex
        // TODO: Implement proper parallel execution with thread pool
        try self.runTestsSequential(test_cases, results, failures);
    }

    /// Run a single test case
    fn runSingleTest(self: *TestRunner, test_case: TestCase) !TestResult {
        // Check if test should be skipped
        if (test_case.skip) {
            return TestResult.skipped(test_case.skip_reason orelse "Test skipped");
        }

        // Set up test arena
        var test_arena = TestArena.init(self.allocator, self.config.enable_memory_tracking);
        defer test_arena.deinit();

        const test_allocator = test_arena.allocator();

        // Run the test with timeout
        const timeout_ms = test_case.timeout_ms orelse self.config.default_timeout_ms;

        const result = self.runTestWithTimeout(test_case.test_fn, test_allocator, timeout_ms) catch |err| {
            const error_message = switch (err) {
                error.OutOfMemory => "Test ran out of memory",
            };
            return TestResult.failed(TestFailure.simple(error_message));
        };

        // Check for memory leaks
        if (self.config.enable_memory_tracking) {
            const stats = test_arena.getMemoryStats();
            if (stats.current_allocated > 0) {
                const leak_message = try std.fmt.allocPrint(self.allocator, "Memory leak detected: {} bytes", .{stats.current_allocated});
                return TestResult.failed(TestFailure.simple(leak_message));
            }
        }

        return result;
    }

    /// Run test with timeout (simplified implementation)
    fn runTestWithTimeout(self: *TestRunner, test_fn: TestFn, test_allocator: Allocator, timeout_ms: u32) !TestResult {
        _ = self;
        _ = timeout_ms; // TODO: Implement actual timeout mechanism

        return test_fn(test_allocator) catch |err| {
            const error_message = try std.fmt.allocPrint(test_allocator, "Test failed with error: {}", .{err});
            return TestResult.failed(TestFailure.simple(error_message));
        };
    }

    /// Print test suite results
    pub fn printResults(self: *TestRunner, suite_result: TestSuiteResult) void {
        _ = self;

        std.log.info("{}", .{suite_result});

        if (suite_result.failures.len > 0) {
            std.log.info("Failures:");
            for (suite_result.failures) |failure| {
                std.log.info("  {}", .{failure});
            }
        }
    }
};

/// Helper function to create a test case
pub fn testCase(name: []const u8, test_fn: TestFn) TestCase {
    return TestCase{
        .name = name,
        .test_fn = test_fn,
    };
}

/// Helper function to create a skipped test case
pub fn skipTest(name: []const u8, test_fn: TestFn, reason: []const u8) TestCase {
    return TestCase{
        .name = name,
        .test_fn = test_fn,
        .skip = true,
        .skip_reason = reason,
    };
}

/// Helper function to create a test case with timeout
pub fn testCaseWithTimeout(name: []const u8, test_fn: TestFn, timeout_ms: u32) TestCase {
    return TestCase{
        .name = name,
        .test_fn = test_fn,
        .timeout_ms = timeout_ms,
    };
}

// Example test functions for testing the runner itself
fn examplePassingTest(allocator: Allocator) !TestResult {
    _ = allocator;
    return TestResult.passed();
}

fn exampleFailingTest(allocator: Allocator) !TestResult {
    _ = allocator;
    return TestResult.failed(TestFailure.simple("This test always fails"));
}

fn exampleMemoryTest(allocator: Allocator) !TestResult {
    const memory = try allocator.alloc(u8, 1024);
    defer allocator.free(memory);
    return TestResult.passed();
}

// Tests
test "TestRunner basic functionality" {
    var runner = TestRunner.init(std.testing.allocator, TestRunnerConfig{
        .parallel_execution = false,
        .verbose = false,
    });

    const test_cases = [_]TestCase{
        testCase("passing_test", examplePassingTest),
        testCase("failing_test", exampleFailingTest),
        testCase("memory_test", exampleMemoryTest),
    };

    const result = try runner.runSuite("example_suite", &test_cases);
    defer std.testing.allocator.free(result.failures);

    try std.testing.expect(result.total_tests == 3);
    try std.testing.expect(result.passed_tests == 2);
    try std.testing.expect(result.failed_tests == 1);
    try std.testing.expect(result.skipped_tests == 0);
}

test "TestRunner skip functionality" {
    var runner = TestRunner.init(std.testing.allocator, TestRunnerConfig{
        .parallel_execution = false,
        .verbose = false,
    });

    const test_cases = [_]TestCase{
        testCase("passing_test", examplePassingTest),
        skipTest("skipped_test", exampleFailingTest, "Not implemented yet"),
    };

    const result = try runner.runSuite("skip_suite", &test_cases);
    defer std.testing.allocator.free(result.failures);

    try std.testing.expect(result.total_tests == 2);
    try std.testing.expect(result.passed_tests == 1);
    try std.testing.expect(result.failed_tests == 0);
    try std.testing.expect(result.skipped_tests == 1);
}
