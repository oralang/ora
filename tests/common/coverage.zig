//! Test Coverage - Coverage analysis and reporting
//!
//! Provides basic test coverage analysis and reporting capabilities
//! for the compiler testing framework.

const std = @import("std");
const Allocator = std.mem.Allocator;

/// Coverage data for a source file
pub const FileCoverage = struct {
    file_path: []const u8,
    total_lines: u32,
    covered_lines: u32,
    uncovered_lines: []const u32,

    pub fn getCoveragePercentage(self: FileCoverage) f64 {
        if (self.total_lines == 0) return 0.0;
        return (@as(f64, @floatFromInt(self.covered_lines)) / @as(f64, @floatFromInt(self.total_lines))) * 100.0;
    }

    pub fn format(self: FileCoverage, comptime fmt: []const u8, options: std.fmt.FormatOptions, writer: anytype) !void {
        _ = fmt;
        _ = options;

        try writer.print("{s}: {d:.1}% ({}/{} lines)", .{
            self.file_path,
            self.getCoveragePercentage(),
            self.covered_lines,
            self.total_lines,
        });
    }
};

/// Overall coverage statistics
pub const CoverageStats = struct {
    total_files: u32,
    total_lines: u32,
    covered_lines: u32,
    files: []FileCoverage,

    pub fn getOverallPercentage(self: CoverageStats) f64 {
        if (self.total_lines == 0) return 0.0;
        return (@as(f64, @floatFromInt(self.covered_lines)) / @as(f64, @floatFromInt(self.total_lines))) * 100.0;
    }

    pub fn format(self: CoverageStats, comptime fmt: []const u8, options: std.fmt.FormatOptions, writer: anytype) !void {
        _ = fmt;
        _ = options;

        try writer.print("Overall Coverage: {d:.1}% ({}/{} lines across {} files)", .{
            self.getOverallPercentage(),
            self.covered_lines,
            self.total_lines,
            self.total_files,
        });
    }
};

/// Coverage reporter for generating coverage reports
pub const CoverageReporter = struct {
    allocator: Allocator,

    pub fn init(allocator: Allocator) CoverageReporter {
        return CoverageReporter{ .allocator = allocator };
    }

    /// Generate a simple coverage report
    pub fn generateReport(self: *CoverageReporter, stats: CoverageStats) ![]u8 {
        var buffer = std.ArrayList(u8).init(self.allocator);
        const writer = buffer.writer();

        try writer.writeAll("Test Coverage Report\n");
        try writer.writeAll("===================\n\n");

        try writer.print("{}\n\n", .{stats});

        try writer.writeAll("File Coverage:\n");
        try writer.writeAll("--------------\n");

        for (stats.files) |file_coverage| {
            try writer.print("{}\n", .{file_coverage});
        }

        try writer.writeAll("\nCoverage Thresholds:\n");
        try writer.writeAll("-------------------\n");

        const overall_percentage = stats.getOverallPercentage();
        if (overall_percentage >= 90.0) {
            try writer.writeAll("✅ Excellent coverage (≥90%)\n");
        } else if (overall_percentage >= 80.0) {
            try writer.writeAll("✅ Good coverage (≥80%)\n");
        } else if (overall_percentage >= 70.0) {
            try writer.writeAll("⚠️  Acceptable coverage (≥70%)\n");
        } else {
            try writer.writeAll("❌ Poor coverage (<70%)\n");
        }

        return buffer.toOwnedSlice();
    }

    /// Generate HTML coverage report
    pub fn generateHtmlReport(self: *CoverageReporter, stats: CoverageStats) ![]u8 {
        var buffer = std.ArrayList(u8).init(self.allocator);
        const writer = buffer.writer();

        try writer.writeAll(
            \\<!DOCTYPE html>
            \\<html>
            \\<head>
            \\    <title>Test Coverage Report</title>
            \\    <style>
            \\        body { font-family: Arial, sans-serif; margin: 20px; }
            \\        .header { background-color: #f0f0f0; padding: 20px; border-radius: 5px; }
            \\        .stats { margin: 20px 0; }
            \\        .file { margin: 10px 0; padding: 10px; border: 1px solid #ddd; border-radius: 3px; }
            \\        .good { background-color: #d4edda; }
            \\        .warning { background-color: #fff3cd; }
            \\        .poor { background-color: #f8d7da; }
            \\    </style>
            \\</head>
            \\<body>
            \\    <div class="header">
            \\        <h1>Test Coverage Report</h1>
            \\
        );

        try writer.print("        <p>Overall Coverage: <strong>{d:.1}%</strong> ({}/{} lines across {} files)</p>\n", .{
            stats.getOverallPercentage(),
            stats.covered_lines,
            stats.total_lines,
            stats.total_files,
        });

        try writer.writeAll("    </div>\n\n    <div class=\"stats\">\n        <h2>File Coverage</h2>\n");

        for (stats.files) |file_coverage| {
            const percentage = file_coverage.getCoveragePercentage();
            const css_class = if (percentage >= 80.0) "good" else if (percentage >= 60.0) "warning" else "poor";

            try writer.print("        <div class=\"file {s}\">\n", .{css_class});
            try writer.print("            <strong>{s}</strong>: {d:.1}% ({}/{} lines)\n", .{
                file_coverage.file_path,
                percentage,
                file_coverage.covered_lines,
                file_coverage.total_lines,
            });
            try writer.writeAll("        </div>\n");
        }

        try writer.writeAll("    </div>\n</body>\n</html>\n");

        return buffer.toOwnedSlice();
    }
};

/// Mock coverage data for testing (since we don't have actual coverage instrumentation)
pub fn generateMockCoverageStats(allocator: Allocator) !CoverageStats {
    var files = std.ArrayList(FileCoverage).init(allocator);

    try files.append(FileCoverage{
        .file_path = "src/lexer.zig",
        .total_lines = 1500,
        .covered_lines = 1350,
        .uncovered_lines = &.{ 45, 67, 123, 456, 789 },
    });

    try files.append(FileCoverage{
        .file_path = "src/parser.zig",
        .total_lines = 2000,
        .covered_lines = 1600,
        .uncovered_lines = &.{ 12, 34, 56, 78, 90 },
    });

    try files.append(FileCoverage{
        .file_path = "src/ast.zig",
        .total_lines = 800,
        .covered_lines = 720,
        .uncovered_lines = &.{ 100, 200, 300 },
    });

    const files_slice = try files.toOwnedSlice();

    var total_lines: u32 = 0;
    var covered_lines: u32 = 0;

    for (files_slice) |file| {
        total_lines += file.total_lines;
        covered_lines += file.covered_lines;
    }

    return CoverageStats{
        .total_files = @intCast(files_slice.len),
        .total_lines = total_lines,
        .covered_lines = covered_lines,
        .files = files_slice,
    };
}

// Tests
test "FileCoverage percentage calculation" {
    const file_coverage = FileCoverage{
        .file_path = "test.zig",
        .total_lines = 100,
        .covered_lines = 85,
        .uncovered_lines = &.{},
    };

    try std.testing.expect(file_coverage.getCoveragePercentage() == 85.0);
}

test "CoverageStats overall percentage" {
    var files = [_]FileCoverage{
        FileCoverage{
            .file_path = "file1.zig",
            .total_lines = 100,
            .covered_lines = 80,
            .uncovered_lines = &.{},
        },
        FileCoverage{
            .file_path = "file2.zig",
            .total_lines = 200,
            .covered_lines = 160,
            .uncovered_lines = &.{},
        },
    };

    const stats = CoverageStats{
        .total_files = 2,
        .total_lines = 300,
        .covered_lines = 240,
        .files = files[0..],
    };

    try std.testing.expect(stats.getOverallPercentage() == 80.0);
}

test "CoverageReporter report generation" {
    var reporter = CoverageReporter.init(std.testing.allocator);

    const stats = try generateMockCoverageStats(std.testing.allocator);
    defer std.testing.allocator.free(stats.files);

    const report = try reporter.generateReport(stats);
    defer std.testing.allocator.free(report);

    try std.testing.expect(report.len > 0);
    try std.testing.expect(std.mem.indexOf(u8, report, "Test Coverage Report") != null);
}
