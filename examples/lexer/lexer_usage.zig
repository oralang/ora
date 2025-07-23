const std = @import("std");
const Lexer = @import("../../src/lexer.zig").Lexer;
const LexerConfig = @import("../../src/lexer.zig").LexerConfig;
const TokenType = @import("../../src/lexer.zig").TokenType;
const TokenValue = @import("../../src/lexer.zig").TokenValue;
const DiagnosticSeverity = @import("../../src/lexer.zig").DiagnosticSeverity;

pub fn main() !void {
    // Initialize allocator
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    // Sample source code with various features
    const source =
        \\// String literals
        \\let regular = "Hello, world!";
        \\let escaped = "Line 1\nLine 2\tTabbed";
        \\let raw = r"C:\Program Files\App";
        \\let character = 'A';
        \\
        \\// Number literals
        \\let decimal = 12345;
        \\let binary = 0b1010;
        \\let hex = 0xDEADBEEF;
        \\let address = 0x742d35Cc6634C0532925a3b8D404fAbCe4649681;
        \\
        \\// Error example (intentional)
        \\let error_string = "Unterminated
        \\let error_char = 'abc';
        \\
        \\// Valid code after errors
        \\let valid = 42;
    ;

    // Example 1: Basic usage with default configuration
    std.debug.print("\n=== Example 1: Basic Usage ===\n", .{});
    try basicUsageExample(allocator, source);

    // Example 2: Error recovery mode
    std.debug.print("\n=== Example 2: Error Recovery ===\n", .{});
    try errorRecoveryExample(allocator, source);

    // Example 3: Accessing processed token values
    std.debug.print("\n=== Example 3: Token Values ===\n", .{});
    try tokenValueExample(allocator, source);

    // Example 4: Performance configuration
    std.debug.print("\n=== Example 4: Performance Configuration ===\n", .{});
    try performanceExample(allocator, source);

    // Example 5: Diagnostic reporting
    std.debug.print("\n=== Example 5: Diagnostic Reporting ===\n", .{});
    try diagnosticReportingExample(allocator, source);
}

// Example 1: Basic usage with default configuration
fn basicUsageExample(allocator: std.mem.Allocator, source: []const u8) !void {
    // Create default configuration
    const config = LexerConfig.default();

    // Initialize lexer
    var lexer = try Lexer.init(allocator, source, config);
    defer lexer.deinit();

    // Traditional token-by-token processing
    std.debug.print("Tokens:\n", .{});
    var token_count: usize = 0;

    while (true) {
        const token = lexer.nextToken() catch |err| {
            std.debug.print("Error: {s} at line {}, column {}\n", .{ @errorName(err), lexer.line, lexer.column });
            break;
        };

        if (token.type == .Eof) break;

        token_count += 1;
        if (token_count <= 5) { // Only print first 5 tokens
            std.debug.print("  {}: {s} at {}:{}\n", .{
                @tagName(token.type),
                token.lexeme,
                token.line,
                token.column,
            });
        }
    }

    std.debug.print("Total tokens: {}\n", .{token_count});
}

// Example 2: Error recovery mode
fn errorRecoveryExample(allocator: std.mem.Allocator, source: []const u8) !void {
    // Create development configuration with error recovery
    var config = LexerConfig.development();
    config.max_errors = 10;

    // Initialize lexer
    var lexer = try Lexer.init(allocator, source, config);
    defer lexer.deinit();

    // Tokenize with error recovery
    const tokens = try lexer.tokenizeWithRecovery();
    defer tokens.deinit();

    // Print token count
    std.debug.print("Successfully tokenized {} tokens despite errors\n", .{tokens.items.len});

    // Check for errors
    const errors = lexer.error_recovery.getErrors();
    std.debug.print("Found {} errors:\n", .{errors.len});

    for (errors, 0..) |diagnostic, i| {
        if (i >= 3) { // Only print first 3 errors
            std.debug.print("  ... and {} more errors\n", .{errors.len - 3});
            break;
        }

        std.debug.print("  Error {}: {s} at {}:{}\n", .{
            i + 1,
            @errorName(diagnostic.error_type),
            diagnostic.range.start_line,
            diagnostic.range.start_column,
        });

        if (diagnostic.suggestion) |suggestion| {
            std.debug.print("    Suggestion: {s}\n", .{suggestion});
        }
    }
}

// Example 3: Accessing processed token values
fn tokenValueExample(allocator: std.mem.Allocator, source: []const u8) !void {
    // Create default configuration
    const config = LexerConfig.default();

    // Initialize lexer
    var lexer = try Lexer.init(allocator, source, config);
    defer lexer.deinit();

    // Process tokens and extract values
    std.debug.print("Processed token values:\n", .{});

    while (true) {
        const token = lexer.nextToken() catch |err| {
            std.debug.print("Error: {s}\n", .{@errorName(err)});
            continue;
        };

        if (token.type == .Eof) break;

        // Only process certain token types
        switch (token.type) {
            .StringLiteral, .RawStringLiteral, .CharacterLiteral, .IntegerLiteral, .BinaryLiteral, .HexLiteral, .AddressLiteral => {
                std.debug.print("  {}: {s} = ", .{ @tagName(token.type), token.lexeme });

                if (token.value) |value| {
                    switch (value) {
                        .string => |str| {
                            std.debug.print("string: \"{s}\"", .{str});
                        },
                        .character => |ch| {
                            std.debug.print("char: '{c}' ({})", .{ ch, ch });
                        },
                        .integer => |int| {
                            std.debug.print("integer: {}", .{int});
                        },
                        .binary => |bin| {
                            std.debug.print("binary: {} (decimal: {})", .{ token.lexeme, bin });
                        },
                        .hex => |hex| {
                            std.debug.print("hex: {} (decimal: {})", .{ token.lexeme, hex });
                        },
                        .address => |addr| {
                            std.debug.print("address: 0x", .{});
                            for (addr) |byte| {
                                std.debug.print("{x:0>2}", .{byte});
                            }
                        },
                        .boolean => |b| {
                            std.debug.print("boolean: {}", .{b});
                        },
                    }
                } else {
                    std.debug.print("no value", .{});
                }
                std.debug.print("\n", .{});
            },
            else => {},
        }
    }
}

// Example 4: Performance configuration
fn performanceExample(allocator: std.mem.Allocator, source: []const u8) !void {
    // Create performance-optimized configuration
    var config = LexerConfig.performance();
    config.enable_performance_monitoring = true;

    // Initialize lexer
    var lexer = try Lexer.init(allocator, source, config);
    defer lexer.deinit();

    // Tokenize multiple times for benchmarking
    var total_tokens: usize = 0;
    const iterations = 100;

    const start_time = std.time.milliTimestamp();

    for (0..iterations) |_| {
        lexer.reset(); // Reset to beginning of source

        while (true) {
            const token = lexer.nextToken() catch break;
            if (token.type == .Eof) break;
            total_tokens += 1;
        }
    }

    const end_time = std.time.milliTimestamp();
    const elapsed_ms = @as(f64, @floatFromInt(end_time - start_time));

    std.debug.print("Performance results:\n", .{});
    std.debug.print("  Iterations: {}\n", .{iterations});
    std.debug.print("  Total tokens: {}\n", .{total_tokens});
    std.debug.print("  Time: {d:.2} ms\n", .{elapsed_ms});
    std.debug.print("  Tokens per second: {d:.2}\n", .{@as(f64, @floatFromInt(total_tokens)) / (elapsed_ms / 1000.0)});

    // If performance monitoring is implemented
    if (@hasDecl(@TypeOf(lexer), "getPerformanceStats")) {
        const stats = lexer.getPerformanceStats();
        std.debug.print("  Memory usage: {} bytes\n", .{stats.memory_usage});
    }
}

// Example 5: Diagnostic reporting
fn diagnosticReportingExample(allocator: std.mem.Allocator, source: []const u8) !void {
    // Create development configuration
    var config = LexerConfig.development();
    config.enable_diagnostic_grouping = true;

    // Initialize lexer
    var lexer = try Lexer.init(allocator, source, config);
    defer lexer.deinit();

    // Tokenize with error recovery
    _ = try lexer.tokenizeWithRecovery();

    // Generate diagnostic reports
    if (lexer.error_recovery.getErrorCount() > 0) {
        // Create summary report
        const summary = try lexer.error_recovery.createSummaryReport(allocator);
        defer allocator.free(summary);

        std.debug.print("Diagnostic Summary:\n{s}\n", .{summary});

        // Create detailed report
        const detailed = try lexer.error_recovery.createDetailedReport(allocator);
        defer allocator.free(detailed);

        std.debug.print("\nDetailed Diagnostics (excerpt):\n", .{});

        // Print just the first part of the detailed report
        const max_len = @min(detailed.len, 500);
        std.debug.print("{s}...\n", .{detailed[0..max_len]});

        // Group errors
        var groups = try lexer.error_recovery.groupErrors();
        defer {
            for (groups.items) |group| {
                group.related.deinit();
            }
            groups.deinit();
        }

        std.debug.print("\nGrouped Errors: {} groups\n", .{groups.items.len});
        for (groups.items, 0..) |group, i| {
            if (i >= 1) break; // Only show first group

            std.debug.print("  Group {}: {s} with {} related errors\n", .{
                i + 1,
                @errorName(group.primary.error_type),
                group.related.items.len,
            });
        }
    } else {
        std.debug.print("No errors found.\n", .{});
    }
}
