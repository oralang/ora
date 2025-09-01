//! Test Framework - Main entry point for the compiler testing framework
//!
//! This module provides the main interface for running compiler tests,
//! integrating all test components and providing a unified API.

const std = @import("std");
const Allocator = std.mem.Allocator;

// Import test infrastructure components
pub const TestArena = @import("common/test_arena.zig").TestArena;
pub const MemoryTracker = @import("common/test_arena.zig").MemoryTracker;
pub const MemoryStats = @import("common/test_arena.zig").MemoryStats;
pub const MemoryLeak = @import("common/test_arena.zig").MemoryLeak;

pub const TestResult = @import("common/test_result.zig").TestResult;
pub const TestFailure = @import("common/test_result.zig").TestFailure;
pub const SourceLocation = @import("common/test_result.zig").SourceLocation;
pub const BenchmarkResult = @import("common/test_result.zig").BenchmarkResult;
pub const MemoryUsageResult = @import("common/test_result.zig").MemoryUsageResult;
pub const TestSuiteResult = @import("common/test_result.zig").TestSuiteResult;

pub const TestRunner = @import("common/test_runner.zig").TestRunner;
pub const TestRunnerConfig = @import("common/test_runner.zig").TestRunnerConfig;
pub const TestCase = @import("common/test_runner.zig").TestCase;
pub const TestFn = @import("common/test_runner.zig").TestFn;
pub const testCase = @import("common/test_runner.zig").testCase;
pub const skipTest = @import("common/test_runner.zig").skipTest;
pub const testCaseWithTimeout = @import("common/test_runner.zig").testCaseWithTimeout;

// Import test utilities and assertions
pub const assertions = @import("common/assertions.zig");
pub const assertTokenEqual = assertions.assertTokenEqual;
pub const assertTokenType = assertions.assertTokenType;
pub const assertTokenLexeme = assertions.assertTokenLexeme;
pub const assertTokenSequence = assertions.assertTokenSequence;
pub const assertTokenPosition = assertions.assertTokenPosition;
pub const assertAstNodeType = assertions.assertAstNodeType;
pub const assertValidSourceSpan = assertions.assertValidSourceSpan;
pub const assertTypeInfoConsistent = assertions.assertTypeInfoConsistent;
pub const assertAstNodesEqual = assertions.assertAstNodesEqual;
pub const assertExpressionType = assertions.assertExpressionType;

pub const test_helpers = @import("common/test_helpers.zig");
pub const LexerTestHelper = test_helpers.LexerTestHelper;
pub const ParserTestHelper = test_helpers.ParserTestHelper;
pub const AstTestHelper = test_helpers.AstTestHelper;
pub const TestDataGenerator = test_helpers.TestDataGenerator;
pub const PerformanceTestHelper = test_helpers.PerformanceTestHelper;

// Import fixture management
pub const fixtures = @import("common/fixtures.zig");
pub const TestFixture = fixtures.TestFixture;
pub const FixtureMetadata = fixtures.FixtureMetadata;
pub const FixtureCategory = fixtures.FixtureCategory;
pub const FixtureManager = fixtures.FixtureManager;

pub const FixtureCache = @import("common/fixture_cache.zig").FixtureCache;
pub const CacheConfig = @import("common/fixture_cache.zig").CacheConfig;
pub const CacheStats = @import("common/fixture_cache.zig").CacheStats;

/// Main test framework configuration
pub const TestFrameworkConfig = struct {
    /// Test runner configuration
    runner: TestRunnerConfig = .{},
    /// Fixture cache configuration
    cache: CacheConfig = .{},
    /// Enable verbose output
    verbose: bool = false,
    /// Enable performance benchmarking
    enable_benchmarks: bool = false,
    /// Test filtering pattern (regex)
    filter_pattern: ?[]const u8 = null,
    /// Maximum test execution time in milliseconds
    max_test_time_ms: u32 = 60000,
};

/// Main test framework instance
pub const TestFramework = struct {
    allocator: Allocator,
    config: TestFrameworkConfig,
    runner: TestRunner,
    fixture_manager: FixtureManager,
    fixture_cache: FixtureCache,

    pub fn init(allocator: Allocator, config: TestFrameworkConfig) TestFramework {
        return TestFramework{
            .allocator = allocator,
            .config = config,
            .runner = TestRunner.init(allocator, config.runner),
            .fixture_manager = FixtureManager.init(allocator),
            .fixture_cache = FixtureCache.init(allocator, config.cache),
        };
    }

    /// Walk a directory and feed all .ora files to a callback
    pub fn forEachOraFile(self: *TestFramework, dir_path: []const u8, cb: *const fn (allocator: Allocator, path: []const u8) anyerror!void) !void {
        var dir = try std.fs.cwd().openDir(dir_path, .{ .iterate = true });
        defer dir.close();

        var walker = try dir.walk(self.allocator);
        defer walker.deinit();

        while (try walker.next()) |entry| {
            if (entry.kind != .file) continue;
            if (!std.mem.endsWith(u8, entry.basename, ".ora")) continue;
            const path = try std.fs.path.join(self.allocator, &.{ dir_path, entry.path });
            defer self.allocator.free(path);
            try cb(self.allocator, path);
        }
    }

    pub fn deinit(self: *TestFramework) void {
        self.fixture_cache.deinit();
        self.fixture_manager.deinit();
    }

    /// Run a test suite with the framework
    pub fn runSuite(self: *TestFramework, suite_name: []const u8, test_cases: []const TestCase) !TestSuiteResult {
        if (self.config.verbose) {
            std.log.info("Running test suite: {s}", .{suite_name});
        }

        // Filter tests if pattern is provided
        var filtered_tests = std.ArrayList(TestCase).init(self.allocator);
        defer filtered_tests.deinit();

        if (self.config.filter_pattern) |pattern| {
            for (test_cases) |test_case| {
                if (std.mem.indexOf(u8, test_case.name, pattern) != null) {
                    try filtered_tests.append(test_case);
                }
            }
        } else {
            try filtered_tests.appendSlice(test_cases);
        }

        const result = try self.runner.runSuite(suite_name, filtered_tests.items);

        if (self.config.verbose) {
            self.runner.printResults(result);
        }

        return result;
    }

    /// Get fixture from cache
    pub fn getFixture(self: *TestFramework, name: []const u8) !TestFixture {
        return self.fixture_cache.getFixture(name);
    }

    /// Preload fixtures for better performance
    pub fn preloadFixtures(self: *TestFramework, category: FixtureCategory) !void {
        try self.fixture_cache.preloadCategory(category);
    }

    /// Get framework statistics
    pub fn getStats(self: *TestFramework) FrameworkStats {
        const cache_stats = self.fixture_cache.getStats();

        return FrameworkStats{
            .cache_stats = cache_stats,
        };
    }

    /// Clean up resources and prepare for shutdown
    pub fn cleanup(self: *TestFramework) !void {
        try self.fixture_cache.evictStale();
    }
};

/// Framework statistics
pub const FrameworkStats = struct {
    cache_stats: CacheStats,

    pub fn format(self: FrameworkStats, comptime fmt: []const u8, options: std.fmt.FormatOptions, writer: anytype) !void {
        _ = fmt;
        _ = options;

        try writer.writeAll("Test Framework Stats:\n");
        try writer.print("  {any}\n", .{self.cache_stats});
    }
};

/// Convenience function to create a test framework with default configuration
pub fn createDefaultFramework(allocator: Allocator) TestFramework {
    return TestFramework.init(allocator, TestFrameworkConfig{});
}

/// Convenience function to create a test framework with verbose output
pub fn createVerboseFramework(allocator: Allocator) TestFramework {
    return TestFramework.init(allocator, TestFrameworkConfig{
        .verbose = true,
        .runner = .{ .verbose = true },
    });
}

/// Convenience function to create a test framework for benchmarking
pub fn createBenchmarkFramework(allocator: Allocator) TestFramework {
    return TestFramework.init(allocator, TestFrameworkConfig{
        .enable_benchmarks = true,
        .runner = .{ .enable_memory_tracking = true },
    });
}

// Tests for the framework itself
test "TestFramework initialization" {
    var framework = createDefaultFramework(std.testing.allocator);
    defer framework.deinit();

    const stats = framework.getStats();
    try std.testing.expect(stats.cache_stats.cached_fixtures == 0);
}

test "TestFramework with verbose config" {
    var framework = createVerboseFramework(std.testing.allocator);
    defer framework.deinit();

    try std.testing.expect(framework.config.verbose == true);
    try std.testing.expect(framework.config.runner.verbose == true);
}

test "TestFramework benchmark config" {
    var framework = createBenchmarkFramework(std.testing.allocator);
    defer framework.deinit();

    try std.testing.expect(framework.config.enable_benchmarks == true);
    try std.testing.expect(framework.config.runner.enable_memory_tracking == true);
}
