//! CI/CD Integration - Helpers for continuous integration
//!
//! Provides utilities for integrating the test framework with CI/CD systems,
//! including test result reporting and exit code handling.

const std = @import("std");
const Allocator = std.mem.Allocator;
const TestSuiteResult = @import("test_result.zig").TestSuiteResult;
const CoverageStats = @import("coverage.zig").CoverageStats;

/// CI/CD test result format
pub const CIResultFormat = enum {
    junit_xml,
    github_actions,
    plain_text,
    json,
};

/// CI/CD integration configuration
pub const CIConfig = struct {
    /// Output format for test results
    format: CIResultFormat = .plain_text,
    /// Output file path (null for stdout)
    output_file: ?[]const u8 = null,
    /// Fail build on test failures
    fail_on_test_failure: bool = true,
    /// Fail build on coverage below threshold
    fail_on_coverage_threshold: bool = false,
    /// Minimum coverage percentage required
    coverage_threshold: f64 = 80.0,
    /// Enable verbose output
    verbose: bool = false,
};

/// CI/CD integration helper
pub const CIIntegration = struct {
    allocator: Allocator,
    config: CIConfig,

    pub fn init(allocator: Allocator, config: CIConfig) CIIntegration {
        return CIIntegration{
            .allocator = allocator,
            .config = config,
        };
    }

    /// Report test results in CI format
    pub fn reportTestResults(self: *CIIntegration, results: []const TestSuiteResult) !void {
        const output = switch (self.config.format) {
            .junit_xml => try self.generateJUnitXML(results),
            .github_actions => try self.generateGitHubActions(results),
            .plain_text => try self.generatePlainText(results),
            .json => try self.generateJSON(results),
        };
        defer self.allocator.free(output);

        if (self.config.output_file) |file_path| {
            try self.writeToFile(file_path, output);
        } else {
            std.log.info("{s}", .{output});
        }
    }

    /// Report coverage results
    pub fn reportCoverage(self: *CIIntegration, coverage: CoverageStats) !void {
        const output = switch (self.config.format) {
            .github_actions => try self.generateCoverageGitHubActions(coverage),
            else => try self.generateCoveragePlainText(coverage),
        };
        defer self.allocator.free(output);

        if (self.config.verbose) {
            std.log.info("Coverage Report:\n{s}", .{output});
        }

        // Check coverage threshold
        if (self.config.fail_on_coverage_threshold) {
            const overall_percentage = coverage.getOverallPercentage();
            if (overall_percentage < self.config.coverage_threshold) {
                std.log.err("Coverage {d:.1}% is below threshold {d:.1}%", .{ overall_percentage, self.config.coverage_threshold });
                std.process.exit(1);
            }
        }
    }

    /// Determine exit code based on test results
    pub fn getExitCode(self: *CIIntegration, results: []const TestSuiteResult) u8 {
        if (!self.config.fail_on_test_failure) {
            return 0;
        }

        for (results) |result| {
            if (result.failed_tests > 0) {
                return 1;
            }
        }

        return 0;
    }

    /// Generate JUnit XML format
    fn generateJUnitXML(self: *CIIntegration, results: []const TestSuiteResult) ![]u8 {
        var buffer = std.ArrayList(u8).init(self.allocator);
        const writer = buffer.writer();

        try writer.writeAll("<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n");
        try writer.writeAll("<testsuites>\n");

        for (results) |result| {
            try writer.print("  <testsuite name=\"{s}\" tests=\"{}\" failures=\"{}\" skipped=\"{}\" time=\"{d:.3}\">\n", .{
                result.name,
                result.total_tests,
                result.failed_tests,
                result.skipped_tests,
                result.duration_ms / 1000.0,
            });

            // Add individual test cases (simplified)
            const passed_tests = result.total_tests - result.failed_tests - result.skipped_tests;
            var i: u32 = 0;
            while (i < passed_tests) : (i += 1) {
                try writer.print("    <testcase name=\"test_{}\"/>\n", .{i});
            }

            // Add failed test cases
            for (result.failures) |failure| {
                try writer.print("    <testcase name=\"failed_test\">\n");
                try writer.print("      <failure message=\"{s}\"/>\n", .{failure.message});
                try writer.print("    </testcase>\n");
            }

            try writer.writeAll("  </testsuite>\n");
        }

        try writer.writeAll("</testsuites>\n");

        return buffer.toOwnedSlice();
    }

    /// Generate GitHub Actions format
    fn generateGitHubActions(self: *CIIntegration, results: []const TestSuiteResult) ![]u8 {
        var buffer = std.ArrayList(u8).init(self.allocator);
        const writer = buffer.writer();

        for (results) |result| {
            if (result.failed_tests > 0) {
                try writer.print("::error title=Test Failures::Suite '{s}' has {} failed tests\n", .{ result.name, result.failed_tests });

                for (result.failures) |failure| {
                    try writer.print("::error::{s}\n", .{failure.message});
                }
            } else {
                try writer.print("::notice title=Test Success::Suite '{s}' passed all {} tests\n", .{ result.name, result.total_tests });
            }
        }

        return buffer.toOwnedSlice();
    }

    /// Generate plain text format
    fn generatePlainText(self: *CIIntegration, results: []const TestSuiteResult) ![]u8 {
        var buffer = std.ArrayList(u8).init(self.allocator);
        const writer = buffer.writer();

        try writer.writeAll("Test Results Summary\n");
        try writer.writeAll("===================\n\n");

        var total_tests: u32 = 0;
        var total_passed: u32 = 0;
        var total_failed: u32 = 0;
        var total_skipped: u32 = 0;

        for (results) |result| {
            try writer.print("{}\n", .{result});

            total_tests += result.total_tests;
            total_passed += result.passed_tests;
            total_failed += result.failed_tests;
            total_skipped += result.skipped_tests;

            if (result.failed_tests > 0) {
                try writer.writeAll("  Failures:\n");
                for (result.failures) |failure| {
                    try writer.print("    - {s}\n", .{failure.message});
                }
            }
            try writer.writeAll("\n");
        }

        try writer.writeAll("Overall Summary:\n");
        try writer.writeAll("---------------\n");
        try writer.print("Total: {}, Passed: {}, Failed: {}, Skipped: {}\n", .{ total_tests, total_passed, total_failed, total_skipped });

        if (total_failed > 0) {
            try writer.writeAll("❌ Some tests failed\n");
        } else {
            try writer.writeAll("✅ All tests passed\n");
        }

        return buffer.toOwnedSlice();
    }

    /// Generate JSON format
    fn generateJSON(self: *CIIntegration, results: []const TestSuiteResult) ![]u8 {
        var buffer = std.ArrayList(u8).init(self.allocator);
        const writer = buffer.writer();

        try writer.writeAll("{\n  \"test_results\": [\n");

        for (results, 0..) |result, i| {
            try writer.print("    {{\n");
            try writer.print("      \"name\": \"{s}\",\n", .{result.name});
            try writer.print("      \"total_tests\": {},\n", .{result.total_tests});
            try writer.print("      \"passed_tests\": {},\n", .{result.passed_tests});
            try writer.print("      \"failed_tests\": {},\n", .{result.failed_tests});
            try writer.print("      \"skipped_tests\": {},\n", .{result.skipped_tests});
            try writer.print("      \"duration_ms\": {d:.2}\n", .{result.duration_ms});
            try writer.print("    }}");

            if (i < results.len - 1) {
                try writer.writeAll(",");
            }
            try writer.writeAll("\n");
        }

        try writer.writeAll("  ]\n}\n");

        return buffer.toOwnedSlice();
    }

    /// Generate coverage report for GitHub Actions
    fn generateCoverageGitHubActions(self: *CIIntegration, coverage: CoverageStats) ![]u8 {
        var buffer = std.ArrayList(u8).init(self.allocator);
        const writer = buffer.writer();

        const percentage = coverage.getOverallPercentage();

        if (percentage >= 90.0) {
            try writer.print("::notice title=Coverage::Excellent coverage: {d:.1}%\n", .{percentage});
        } else if (percentage >= 80.0) {
            try writer.print("::notice title=Coverage::Good coverage: {d:.1}%\n", .{percentage});
        } else if (percentage >= 70.0) {
            try writer.print("::warning title=Coverage::Acceptable coverage: {d:.1}%\n", .{percentage});
        } else {
            try writer.print("::error title=Coverage::Poor coverage: {d:.1}%\n", .{percentage});
        }

        return buffer.toOwnedSlice();
    }

    /// Generate plain text coverage report
    fn generateCoveragePlainText(self: *CIIntegration, coverage: CoverageStats) ![]u8 {
        var buffer = std.ArrayList(u8).init(self.allocator);
        const writer = buffer.writer();

        try writer.print("Coverage: {d:.1}% ({}/{} lines)\n", .{
            coverage.getOverallPercentage(),
            coverage.covered_lines,
            coverage.total_lines,
        });

        return buffer.toOwnedSlice();
    }

    /// Write output to file
    fn writeToFile(self: *CIIntegration, file_path: []const u8, content: []const u8) !void {
        _ = self;
        const file = try std.fs.cwd().createFile(file_path, .{});
        defer file.close();

        try file.writeAll(content);
    }
};

/// Detect CI environment
pub fn detectCIEnvironment() ?CIEnvironment {
    if (std.process.hasEnvVarConstant("GITHUB_ACTIONS")) {
        return .github_actions;
    } else if (std.process.hasEnvVarConstant("GITLAB_CI")) {
        return .gitlab_ci;
    } else if (std.process.hasEnvVarConstant("JENKINS_URL")) {
        return .jenkins;
    } else if (std.process.hasEnvVarConstant("TRAVIS")) {
        return .travis_ci;
    }

    return null;
}

/// CI environment types
pub const CIEnvironment = enum {
    github_actions,
    gitlab_ci,
    jenkins,
    travis_ci,
    unknown,
};

/// Create CI configuration based on detected environment
pub fn createCIConfig(environment: ?CIEnvironment) CIConfig {
    const env = environment orelse .unknown;

    return switch (env) {
        .github_actions => CIConfig{
            .format = .github_actions,
            .verbose = true,
        },
        .gitlab_ci => CIConfig{
            .format = .junit_xml,
            .output_file = "test-results.xml",
        },
        .jenkins => CIConfig{
            .format = .junit_xml,
            .output_file = "test-results.xml",
        },
        .travis_ci => CIConfig{
            .format = .plain_text,
            .verbose = true,
        },
        .unknown => CIConfig{
            .format = .plain_text,
        },
    };
}

// Tests
test "CIIntegration exit code determination" {
    const config = CIConfig{ .fail_on_test_failure = true };
    var ci = CIIntegration.init(std.testing.allocator, config);

    const passing_result = TestSuiteResult{
        .name = "test_suite",
        .total_tests = 10,
        .passed_tests = 10,
        .failed_tests = 0,
        .skipped_tests = 0,
        .duration_ms = 100.0,
        .failures = &.{},
    };

    const failing_result = TestSuiteResult{
        .name = "test_suite",
        .total_tests = 10,
        .passed_tests = 8,
        .failed_tests = 2,
        .skipped_tests = 0,
        .duration_ms = 100.0,
        .failures = &.{},
    };

    try std.testing.expect(ci.getExitCode(&.{passing_result}) == 0);
    try std.testing.expect(ci.getExitCode(&.{failing_result}) == 1);
}

test "CI environment detection" {
    // This test would need environment variables to be set
    // For now, just test that the function doesn't crash
    const env = detectCIEnvironment();
    _ = env;
}
