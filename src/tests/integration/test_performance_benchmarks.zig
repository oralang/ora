const std = @import("std");
const testing = std.testing;
const Allocator = std.mem.Allocator;
const Timer = std.time.Timer;
const lexer = @import("src/lexer.zig");
const Lexer = lexer.Lexer;
const Token = lexer.Token;
const TokenType = lexer.TokenType;
const LexerConfig = lexer.LexerConfig;

// Benchmark configuration
const BENCHMARK_ITERATIONS = 100;
const LARGE_FILE_SIZE = 10000; // Number of tokens in large file test
const MEMORY_MEASUREMENT_ITERATIONS = 10;

// Helper function to create minimal config (no error recovery)
fn createMinimalConfig() LexerConfig {
    return LexerConfig{
        .enable_error_recovery = false,
        .enable_string_interning = false,
        .max_errors = 10,
        .enable_suggestions = false,
        .enable_diagnostic_grouping = false,
        .enable_diagnostic_filtering = false,
    };
}

// Helper function to create config with string interning
fn createStringInterningConfig() LexerConfig {
    return LexerConfig{
        .enable_error_recovery = false,
        .enable_string_interning = true,
        .max_errors = 10,
        .enable_suggestions = false,
        .enable_diagnostic_grouping = false,
        .enable_diagnostic_filtering = false,
    };
}

// Benchmark result structure
const BenchmarkResult = struct {
    name: []const u8,
    avg_time_ns: u64,
    min_time_ns: u64,
    max_time_ns: u64,
    iterations: u32,
    tokens_per_second: f64,
    memory_used_bytes: usize,
};

// Helper function to run a benchmark
fn runBenchmark(
    allocator: Allocator,
    name: []const u8,
    source: []const u8,
    config: LexerConfig,
    iterations: u32,
) !BenchmarkResult {
    var timer = try Timer.start();
    var total_time: u64 = 0;
    var min_time: u64 = std.math.maxInt(u64);
    var max_time: u64 = 0;
    var total_tokens: usize = 0;
    var memory_used: usize = 0;

    // Run benchmark iterations
    for (0..iterations) |_| {
        timer.reset();

        var lex = try Lexer.initWithConfig(allocator, source, config);
        defer lex.deinit();

        const start_time = timer.read();
        const tokens = try lex.scanTokens();
        const end_time = timer.read();

        const iteration_time = end_time - start_time;
        total_time += iteration_time;
        min_time = @min(min_time, iteration_time);
        max_time = @max(max_time, iteration_time);
        total_tokens = tokens.len;

        // Estimate memory usage (rough approximation)
        memory_used = tokens.len * @sizeOf(Token) + source.len;

        allocator.free(tokens);
    }

    const avg_time = total_time / iterations;
    const tokens_per_second = if (avg_time > 0)
        (@as(f64, @floatFromInt(total_tokens)) * 1_000_000_000.0) / @as(f64, @floatFromInt(avg_time))
    else
        0.0;

    return BenchmarkResult{
        .name = name,
        .avg_time_ns = avg_time,
        .min_time_ns = min_time,
        .max_time_ns = max_time,
        .iterations = iterations,
        .tokens_per_second = tokens_per_second,
        .memory_used_bytes = memory_used,
    };
}

// Helper function to print benchmark results
fn printBenchmarkResult(result: BenchmarkResult) void {
    std.debug.print("\n=== {s} ===\n", .{result.name});
    std.debug.print("Iterations: {}\n", .{result.iterations});
    std.debug.print("Average time: {d:.2} ms\n", .{@as(f64, @floatFromInt(result.avg_time_ns)) / 1_000_000.0});
    std.debug.print("Min time: {d:.2} ms\n", .{@as(f64, @floatFromInt(result.min_time_ns)) / 1_000_000.0});
    std.debug.print("Max time: {d:.2} ms\n", .{@as(f64, @floatFromInt(result.max_time_ns)) / 1_000_000.0});
    std.debug.print("Tokens/second: {d:.0}\n", .{result.tokens_per_second});
    std.debug.print("Memory used: {} bytes\n", .{result.memory_used_bytes});
}

// Generate test source code with various token types
fn generateTestSource(allocator: Allocator, token_count: usize) ![]u8 {
    var source = std.ArrayList(u8).init(allocator);
    defer source.deinit();

    const token_patterns = [_][]const u8{
        "let x = 42;\n",
        "fn test() { return true; }\n",
        "contract MyContract {\n",
        "    pub storage value: u256;\n",
        "    let binary = 0b1010;\n",
        "    let hex = 0xFF;\n",
        "    let str = \"hello world\";\n",
        "    if (x > 0) { x += 1; }\n",
        "}\n",
    };

    var tokens_generated: usize = 0;
    while (tokens_generated < token_count) {
        for (token_patterns) |pattern| {
            try source.appendSlice(pattern);
            tokens_generated += 10; // Rough estimate of tokens per pattern
            if (tokens_generated >= token_count) break;
        }
    }

    return source.toOwnedSlice();
}

test "performance benchmark - basic tokenization speed" {
    const allocator = testing.allocator;

    const source =
        \\contract Test {
        \\    let x = 42;
        \\    let y = 0xFF;
        \\    let z = 0b1010;
        \\    fn test() -> bool {
        \\        return x > 0;
        \\    }
        \\}
    ;

    const config = createMinimalConfig();

    const result = try runBenchmark(allocator, "Basic Tokenization", source, config, BENCHMARK_ITERATIONS);
    printBenchmarkResult(result);

    // Basic performance assertions
    try testing.expect(result.avg_time_ns > 0);
    try testing.expect(result.tokens_per_second > 1000); // Should process at least 1000 tokens/second
}

test "performance benchmark - large file processing" {
    const allocator = testing.allocator;

    const source = try generateTestSource(allocator, LARGE_FILE_SIZE);
    defer allocator.free(source);

    const config = LexerConfig{
        .enable_error_recovery = false,
        .enable_string_interning = false,
        .max_errors = 10,
        .enable_suggestions = false,
    };

    const result = try runBenchmark(allocator, "Large File Processing", source, config, 10); // Fewer iterations for large files
    printBenchmarkResult(result);

    // Performance should scale reasonably with file size
    try testing.expect(result.avg_time_ns > 0);
    try testing.expect(result.tokens_per_second > 500); // Should still be reasonably fast
}

test "performance benchmark - string interning impact" {
    const allocator = testing.allocator;

    // Source with many repeated identifiers (good for string interning)
    const source =
        \\let variable = 42;
        \\let variable2 = variable + variable;
        \\let variable3 = variable * variable2;
        \\fn function() { return variable; }
        \\fn function2() { return variable2; }
        \\fn function3() { return variable3; }
    ;

    // Benchmark without string interning
    const config_no_interning = LexerConfig{
        .enable_error_recovery = false,
        .enable_string_interning = false,
        .max_errors = 10,
        .enable_suggestions = false,
    };

    const result_no_interning = try runBenchmark(allocator, "Without String Interning", source, config_no_interning, BENCHMARK_ITERATIONS);
    printBenchmarkResult(result_no_interning);

    // Benchmark with string interning
    const config_with_interning = LexerConfig{
        .enable_error_recovery = false,
        .enable_string_interning = true,
        .max_errors = 10,
        .enable_suggestions = false,
    };

    const result_with_interning = try runBenchmark(allocator, "With String Interning", source, config_with_interning, BENCHMARK_ITERATIONS);
    printBenchmarkResult(result_with_interning);

    // String interning might be slightly slower for small files but should use less memory
    try testing.expect(result_no_interning.avg_time_ns > 0);
    try testing.expect(result_with_interning.avg_time_ns > 0);

    std.debug.print("\nString Interning Impact:\n", .{});
    std.debug.print("Time difference: {d:.2}%\n", .{(@as(f64, @floatFromInt(result_with_interning.avg_time_ns)) - @as(f64, @floatFromInt(result_no_interning.avg_time_ns))) /
        @as(f64, @floatFromInt(result_no_interning.avg_time_ns)) * 100.0});
}

test "performance benchmark - error recovery overhead" {
    const allocator = testing.allocator;

    // Source with some errors
    const source =
        \\let x = 42;
        \\let y = 0b999; // Invalid binary
        \\let z = "unterminated string
        \\fn test() { return true; }
    ;

    // Benchmark without error recovery
    const config_no_recovery = LexerConfig{
        .enable_error_recovery = false,
        .enable_string_interning = false,
        .max_errors = 10,
        .enable_suggestions = false,
    };

    // This will likely fail, but we can measure the time until failure
    var no_recovery_time: u64 = 0;
    var timer = try Timer.start();

    for (0..10) |_| {
        timer.reset();
        var lex = Lexer.initWithConfig(allocator, source, config_no_recovery) catch continue;
        defer lex.deinit();

        const start_time = timer.read();
        _ = lex.scanTokens() catch {};
        const end_time = timer.read();

        no_recovery_time += (end_time - start_time);
    }
    no_recovery_time /= 10;

    // Benchmark with error recovery
    const config_with_recovery = LexerConfig{
        .enable_error_recovery = true,
        .enable_string_interning = false,
        .max_errors = 10,
        .enable_suggestions = true,
    };

    const result_with_recovery = try runBenchmark(allocator, "With Error Recovery", source, config_with_recovery, BENCHMARK_ITERATIONS);
    printBenchmarkResult(result_with_recovery);

    std.debug.print("\nError Recovery Overhead:\n", .{});
    std.debug.print("Without recovery (avg): {d:.2} ms\n", .{@as(f64, @floatFromInt(no_recovery_time)) / 1_000_000.0});
    std.debug.print("With recovery (avg): {d:.2} ms\n", .{@as(f64, @floatFromInt(result_with_recovery.avg_time_ns)) / 1_000_000.0});

    if (no_recovery_time > 0) {
        std.debug.print("Overhead: {d:.2}%\n", .{(@as(f64, @floatFromInt(result_with_recovery.avg_time_ns)) - @as(f64, @floatFromInt(no_recovery_time))) /
            @as(f64, @floatFromInt(no_recovery_time)) * 100.0});
    }

    try testing.expect(result_with_recovery.avg_time_ns > 0);
}

test "performance benchmark - number parsing performance" {
    const allocator = testing.allocator;

    // Source with various number formats
    const source =
        \\let decimal = 123456789;
        \\let binary = 0b11111111;
        \\let hex = 0xDEADBEEF;
        \\let address = 0x1234567890123456789012345678901234567890;
        \\let with_underscores = 1_000_000;
        \\let binary_underscores = 0b1010_1010;
        \\let hex_underscores = 0xFF_FF_FF;
    ;

    const config = LexerConfig{
        .enable_error_recovery = false,
        .enable_string_interning = false,
        .max_errors = 10,
        .enable_suggestions = false,
    };

    const result = try runBenchmark(allocator, "Number Parsing", source, config, BENCHMARK_ITERATIONS);
    printBenchmarkResult(result);

    try testing.expect(result.avg_time_ns > 0);
    try testing.expect(result.tokens_per_second > 1000);
}

test "performance benchmark - string processing performance" {
    const allocator = testing.allocator;

    // Source with various string types
    const source =
        \\let simple = "hello world";
        \\let with_escapes = "hello\nworld\t!";
        \\let raw_string = r"no escapes here\n\t";
        \\let char_literal = 'x';
        \\let escaped_char = '\n';
        \\let unicode = "unicode: \x41\x42\x43";
    ;

    const config = LexerConfig{
        .enable_error_recovery = false,
        .enable_string_interning = false,
        .max_errors = 10,
        .enable_suggestions = false,
    };

    const result = try runBenchmark(allocator, "String Processing", source, config, BENCHMARK_ITERATIONS);
    printBenchmarkResult(result);

    try testing.expect(result.avg_time_ns > 0);
    try testing.expect(result.tokens_per_second > 1000);
}

test "performance benchmark - memory usage comparison" {
    const allocator = testing.allocator;

    const source = try generateTestSource(allocator, 1000);
    defer allocator.free(source);

    // Test different configurations and their memory impact
    const configs = [_]struct {
        name: []const u8,
        config: LexerConfig,
    }{
        .{
            .name = "Minimal Config",
            .config = LexerConfig{
                .enable_error_recovery = false,
                .enable_string_interning = false,
                .max_errors = 10,
                .enable_suggestions = false,
            },
        },
        .{
            .name = "With String Interning",
            .config = LexerConfig{
                .enable_error_recovery = false,
                .enable_string_interning = true,
                .max_errors = 10,
                .enable_suggestions = false,
            },
        },
        .{
            .name = "With Error Recovery",
            .config = LexerConfig{
                .enable_error_recovery = true,
                .enable_string_interning = false,
                .max_errors = 50,
                .enable_suggestions = true,
            },
        },
        .{
            .name = "Full Features",
            .config = LexerConfig{
                .enable_error_recovery = true,
                .enable_string_interning = true,
                .max_errors = 50,
                .enable_suggestions = true,
            },
        },
    };

    std.debug.print("\n=== Memory Usage Comparison ===\n", .{});

    for (configs) |test_config| {
        const result = try runBenchmark(allocator, test_config.name, source, test_config.config, MEMORY_MEASUREMENT_ITERATIONS);
        std.debug.print("{s}: {} bytes, {d:.2} ms avg\n", .{
            test_config.name,
            result.memory_used_bytes,
            @as(f64, @floatFromInt(result.avg_time_ns)) / 1_000_000.0,
        });
    }
}

test "performance benchmark - scalability test" {
    const allocator = testing.allocator;

    const file_sizes = [_]usize{ 100, 500, 1000, 2000, 5000 };

    const config = LexerConfig{
        .enable_error_recovery = false,
        .enable_string_interning = false,
        .max_errors = 10,
        .enable_suggestions = false,
    };

    std.debug.print("\n=== Scalability Test ===\n", .{});
    std.debug.print("File Size (tokens) | Time (ms) | Tokens/sec\n", .{});
    std.debug.print("-------------------|-----------|----------\n", .{});

    for (file_sizes) |size| {
        const source = try generateTestSource(allocator, size);
        defer allocator.free(source);

        const iterations = if (size > 2000) 5 else 20; // Fewer iterations for larger files
        const result = try runBenchmark(allocator, "Scalability", source, config, iterations);

        std.debug.print("{:>18} | {:>8.2} | {:>9.0}\n", .{
            size,
            @as(f64, @floatFromInt(result.avg_time_ns)) / 1_000_000.0,
            result.tokens_per_second,
        });

        // Performance should not degrade dramatically with size
        try testing.expect(result.tokens_per_second > 100);
    }
}

test "performance benchmark - baseline vs enhanced comparison" {
    const allocator = testing.allocator;

    const source =
        \\contract Performance {
        \\    let numbers = [42, 0xFF, 0b1010];
        \\    let strings = ["hello", r"raw", 'c'];
        \\    fn process() {
        \\        let x = 1_000_000;
        \\        let addr = 0x1234567890123456789012345678901234567890;
        \\        return x + addr;
        \\    }
        \\}
    ;

    // Baseline configuration (minimal features)
    const baseline_config = LexerConfig{
        .enable_error_recovery = false,
        .enable_string_interning = false,
        .max_errors = 10,
        .enable_suggestions = false,
    };

    // Enhanced configuration (all features)
    const enhanced_config = LexerConfig{
        .enable_error_recovery = true,
        .enable_string_interning = true,
        .max_errors = 50,
        .enable_suggestions = true,
    };

    const baseline_result = try runBenchmark(allocator, "Baseline Lexer", source, baseline_config, BENCHMARK_ITERATIONS);
    const enhanced_result = try runBenchmark(allocator, "Enhanced Lexer", source, enhanced_config, BENCHMARK_ITERATIONS);

    printBenchmarkResult(baseline_result);
    printBenchmarkResult(enhanced_result);

    std.debug.print("\n=== Baseline vs Enhanced Comparison ===\n", .{});
    std.debug.print("Performance overhead: {d:.2}%\n", .{(@as(f64, @floatFromInt(enhanced_result.avg_time_ns)) - @as(f64, @floatFromInt(baseline_result.avg_time_ns))) /
        @as(f64, @floatFromInt(baseline_result.avg_time_ns)) * 100.0});

    std.debug.print("Memory overhead: {d:.2}%\n", .{(@as(f64, @floatFromInt(enhanced_result.memory_used_bytes)) - @as(f64, @floatFromInt(baseline_result.memory_used_bytes))) /
        @as(f64, @floatFromInt(baseline_result.memory_used_bytes)) * 100.0});

    // Enhanced version should not be more than 50% slower than baseline
    const performance_overhead = (@as(f64, @floatFromInt(enhanced_result.avg_time_ns)) - @as(f64, @floatFromInt(baseline_result.avg_time_ns))) /
        @as(f64, @floatFromInt(baseline_result.avg_time_ns)) * 100.0;

    try testing.expect(performance_overhead < 50.0);
    try testing.expect(baseline_result.tokens_per_second > 1000);
    try testing.expect(enhanced_result.tokens_per_second > 500);
}

test "performance benchmark - stress test with errors" {
    const allocator = testing.allocator;

    // Generate source with many errors to stress test error recovery
    var source = std.ArrayList(u8).init(allocator);
    defer source.deinit();

    // Add valid code mixed with errors
    for (0..100) |i| {
        if (i % 3 == 0) {
            try source.appendSlice("0b999 "); // Invalid binary
        } else if (i % 3 == 1) {
            try source.appendSlice("\"unterminated "); // Unterminated string
        } else {
            try source.appendSlice("let x = 42; "); // Valid code
        }
    }

    const config = LexerConfig{
        .enable_error_recovery = true,
        .enable_string_interning = false,
        .max_errors = 200, // High limit for stress test
        .enable_suggestions = true,
    };

    const result = try runBenchmark(allocator, "Stress Test with Errors", source.items, config, 10);
    printBenchmarkResult(result);

    // Should still complete in reasonable time even with many errors
    try testing.expect(result.avg_time_ns > 0);
    try testing.expect(result.avg_time_ns < 100_000_000); // Less than 100ms
}
