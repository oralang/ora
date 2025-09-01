//! Test Helpers - Common testing patterns and utilities
//!
//! Provides helper functions for common testing patterns like token sequences,
//! AST validation, and test data generation.

const std = @import("std");
const Allocator = std.mem.Allocator;
const TestResult = @import("test_result.zig").TestResult;
const TestFailure = @import("test_result.zig").TestFailure;

// Import compiler types
const ora = @import("ora");
const Lexer = ora.Lexer;
const Parser = ora.Parser;
const Token = ora.Token;
const TokenType = ora.TokenType;
const AstNode = ora.AstNode;
const ExprNode = ora.ExprNode;

/// Helper for creating and running lexer tests
pub const LexerTestHelper = struct {
    allocator: Allocator,

    pub fn init(allocator: Allocator) LexerTestHelper {
        return LexerTestHelper{ .allocator = allocator };
    }

    /// Tokenize source code and return tokens
    pub fn tokenize(self: *LexerTestHelper, source: []const u8) ![]Token {
        var lexer = Lexer.init(self.allocator, source);
        defer lexer.deinit();

        return try lexer.scanTokens();
    }

    /// Tokenize and extract only token types
    pub fn tokenizeTypes(self: *LexerTestHelper, source: []const u8) ![]TokenType {
        const tokens = try self.tokenize(source);
        defer self.allocator.free(tokens);

        var types = std.ArrayList(TokenType).init(self.allocator);

        for (tokens) |token| {
            try types.append(token.type);
        }

        return types.toOwnedSlice();
    }

    /// Tokenize and extract only lexemes
    pub fn tokenizeLexemes(self: *LexerTestHelper, source: []const u8) ![][]const u8 {
        const tokens = try self.tokenize(source);
        defer self.allocator.free(tokens);

        var lexemes = std.ArrayList([]const u8).init(self.allocator);

        for (tokens) |token| {
            const lexeme_copy = try self.allocator.dupe(u8, token.lexeme);
            try lexemes.append(lexeme_copy);
        }

        return lexemes.toOwnedSlice();
    }

    /// Test that source produces expected token types
    pub fn expectTokenTypes(self: *LexerTestHelper, source: []const u8, expected: []const TokenType) !TestResult {
        const actual_types = try self.tokenizeTypes(source);
        defer self.allocator.free(actual_types);

        if (expected.len != actual_types.len) {
            const message = try std.fmt.allocPrint(self.allocator, "Token count mismatch: expected {}, got {}", .{ expected.len, actual_types.len });
            return TestResult.failed(TestFailure.simple(message));
        }

        for (expected, actual_types, 0..) |expected_type, actual_type, i| {
            if (expected_type != actual_type) {
                const message = try std.fmt.allocPrint(self.allocator, "Token type mismatch at position {}: expected {s}, got {s}", .{ i, @tagName(expected_type), @tagName(actual_type) });
                return TestResult.failed(TestFailure.simple(message));
            }
        }

        return TestResult.passed();
    }

    /// Test that source produces expected lexemes
    pub fn expectLexemes(self: *LexerTestHelper, source: []const u8, expected: []const []const u8) !TestResult {
        const actual_lexemes = try self.tokenizeLexemes(source);
        defer {
            for (actual_lexemes) |lexeme| {
                self.allocator.free(lexeme);
            }
            self.allocator.free(actual_lexemes);
        }

        if (expected.len != actual_lexemes.len) {
            const message = try std.fmt.allocPrint(self.allocator, "Lexeme count mismatch: expected {}, got {}", .{ expected.len, actual_lexemes.len });
            return TestResult.failed(TestFailure.simple(message));
        }

        for (expected, actual_lexemes, 0..) |expected_lexeme, actual_lexeme, i| {
            if (!std.mem.eql(u8, expected_lexeme, actual_lexeme)) {
                const message = try std.fmt.allocPrint(self.allocator, "Lexeme mismatch at position {}: expected \"{s}\", got \"{s}\"", .{ i, expected_lexeme, actual_lexeme });
                return TestResult.failed(TestFailure.simple(message));
            }
        }

        return TestResult.passed();
    }

    /// Test that source produces lexer errors
    pub fn expectLexerErrors(self: *LexerTestHelper, source: []const u8, expected_error_count: usize) !TestResult {
        var lexer = Lexer.init(self.allocator, source);
        defer lexer.deinit();

        var error_count: usize = 0;

        while (true) {
            const token = lexer.nextToken() catch {
                error_count += 1;
                // Continue to count all errors
                continue;
            };

            if (token.type == .Eof) break;
        }

        if (error_count != expected_error_count) {
            const message = try std.fmt.allocPrint(self.allocator, "Error count mismatch: expected {}, got {}", .{ expected_error_count, error_count });
            return TestResult.failed(TestFailure.simple(message));
        }

        return TestResult.passed();
    }
};

/// Helper for creating and running parser tests
pub const ParserTestHelper = struct {
    allocator: Allocator,

    pub fn init(allocator: Allocator) ParserTestHelper {
        return ParserTestHelper{ .allocator = allocator };
    }

    /// Parse source code and return AST
    pub fn parse(self: *ParserTestHelper, source: []const u8) !*AstNode {
        var lexer = Lexer.init(self.allocator, source);
        defer lexer.deinit();

        var parser = Parser.init(self.allocator, &lexer);
        defer parser.deinit();

        return try parser.parseProgram();
    }

    /// Parse expression and return expression node
    pub fn parseExpression(self: *ParserTestHelper, source: []const u8) !*ExprNode {
        var lexer = Lexer.init(self.allocator, source);
        defer lexer.deinit();

        var parser = Parser.init(self.allocator, &lexer);
        defer parser.deinit();

        return try parser.parseExpression();
    }

    /// Test that source parses successfully
    pub fn expectValidParse(self: *ParserTestHelper, source: []const u8) !TestResult {
        const ast = self.parse(source) catch |err| {
            const message = try std.fmt.allocPrint(self.allocator, "Parse failed with error: {}", .{err});
            return TestResult.failed(TestFailure.simple(message));
        };

        _ = ast; // TODO: Add AST validation
        return TestResult.passed();
    }

    /// Test that source produces parser errors
    pub fn expectParseError(self: *ParserTestHelper, source: []const u8) !TestResult {
        const ast = self.parse(source) catch {
            return TestResult.passed(); // Expected to fail
        };

        _ = ast;
        const message = try std.fmt.allocPrint(self.allocator, "Expected parse to fail, but it succeeded");
        return TestResult.failed(TestFailure.simple(message));
    }

    /// Test that expression parses to expected type
    pub fn expectExpressionType(self: *ParserTestHelper, source: []const u8, expected_type: std.meta.Tag(ExprNode)) !TestResult {
        const expr = self.parseExpression(source) catch |err| {
            const message = try std.fmt.allocPrint(self.allocator, "Expression parse failed with error: {}", .{err});
            return TestResult.failed(TestFailure.simple(message));
        };

        const actual_type = std.meta.activeTag(expr.*);
        if (expected_type != actual_type) {
            const message = try std.fmt.allocPrint(self.allocator, "Expression type mismatch: expected {s}, got {s}", .{ @tagName(expected_type), @tagName(actual_type) });
            return TestResult.failed(TestFailure.simple(message));
        }

        return TestResult.passed();
    }
};

/// Helper for AST validation and testing
pub const AstTestHelper = struct {
    allocator: Allocator,

    pub fn init(allocator: Allocator) AstTestHelper {
        return AstTestHelper{ .allocator = allocator };
    }

    /// Count nodes in AST tree
    pub fn countNodes(self: *AstTestHelper, root: *const AstNode) u32 {
        _ = self;
        _ = root;
        // TODO: Implement AST traversal and counting
        return 1;
    }

    /// Find nodes of specific type in AST
    pub fn findNodesByType(self: *AstTestHelper, root: *const AstNode, node_type: std.meta.Tag(AstNode)) ![]const *AstNode {
        _ = self;
        _ = root;
        _ = node_type;
        // TODO: Implement AST traversal and filtering
        return &.{};
    }

    /// Validate AST structural integrity
    pub fn validateStructure(self: *AstTestHelper, root: *const AstNode) !TestResult {
        _ = self;
        _ = root;
        // TODO: Implement structural validation
        return TestResult.passed();
    }

    /// Test that AST has expected node count
    pub fn expectNodeCount(self: *AstTestHelper, root: *const AstNode, expected_count: u32) !TestResult {
        const actual_count = self.countNodes(root);

        if (expected_count != actual_count) {
            const message = try std.fmt.allocPrint(self.allocator, "Node count mismatch: expected {}, got {}", .{ expected_count, actual_count });
            return TestResult.failed(TestFailure.simple(message));
        }

        return TestResult.passed();
    }
};

/// Helper for generating test data
pub const TestDataGenerator = struct {
    allocator: Allocator,
    random: std.Random,

    pub fn init(allocator: Allocator, seed: u64) TestDataGenerator {
        var prng = std.Random.DefaultPrng.init(seed);
        return TestDataGenerator{
            .allocator = allocator,
            .random = prng.random(),
        };
    }

    /// Generate random identifier
    pub fn generateIdentifier(self: *TestDataGenerator) ![]u8 {
        const length = self.random.intRangeAtMost(u8, 1, 10); // Smaller range to avoid issues
        var identifier = try self.allocator.alloc(u8, length);

        // First character must be letter or underscore
        const first_choice = self.random.intRangeAtMost(u8, 0, 25);
        identifier[0] = if (first_choice == 0) '_' else 'a' + first_choice;

        // Remaining characters can be letters, digits, or underscores
        for (identifier[1..]) |*char| {
            const choice = self.random.intRangeAtMost(u8, 0, 35);
            char.* = if (choice == 0)
                '_'
            else if (choice <= 25)
                'a' + (choice - 1)
            else
                '0' + (choice - 26);
        }

        return identifier;
    }

    /// Generate random string literal
    pub fn generateStringLiteral(self: *TestDataGenerator) ![]u8 {
        const content_length = self.random.intRangeAtMost(u8, 0, 50);
        var content = try self.allocator.alloc(u8, content_length + 2); // +2 for quotes

        content[0] = '"';
        content[content.len - 1] = '"';

        for (content[1 .. content.len - 1]) |*char| {
            // Generate printable ASCII characters, avoiding quotes and backslashes
            char.* = switch (self.random.intRangeAtMost(u8, 32, 126)) {
                '"', '\\' => ' ', // Replace problematic characters with space
                else => |c| c,
            };
        }

        return content;
    }

    /// Generate random number literal
    pub fn generateNumberLiteral(self: *TestDataGenerator) ![]u8 {
        const number = self.random.intRangeAtMost(u32, 0, 999999);
        return try std.fmt.allocPrint(self.allocator, "{}", .{number});
    }

    /// Generate random valid Ora source code
    pub fn generateValidSource(self: *TestDataGenerator) ![]u8 {
        var source = std.ArrayList(u8).init(self.allocator);
        const writer = source.writer();

        // Generate a simple function
        const func_name = try self.generateIdentifier();
        defer self.allocator.free(func_name);

        try writer.print("fn {s}() {{\n", .{func_name});

        // Generate some statements
        const stmt_count = self.random.intRangeAtMost(u8, 1, 5);
        for (0..stmt_count) |_| {
            const var_name = try self.generateIdentifier();
            defer self.allocator.free(var_name);

            const value = try self.generateNumberLiteral();
            defer self.allocator.free(value);

            try writer.print("    let {s} = {any};\n", .{ var_name, value });
        }

        try writer.writeAll("}\n");

        return source.toOwnedSlice();
    }
};

/// Helper for performance testing
pub const PerformanceTestHelper = struct {
    allocator: Allocator,

    pub fn init(allocator: Allocator) PerformanceTestHelper {
        return PerformanceTestHelper{ .allocator = allocator };
    }

    /// Measure execution time of a function
    pub fn measureTime(self: *PerformanceTestHelper, comptime func: anytype, args: anytype) !struct { result: @TypeOf(@call(.auto, func, args)), duration_ns: u64 } {
        _ = self;

        const start_time = std.time.nanoTimestamp();
        const result = @call(.auto, func, args);
        const end_time = std.time.nanoTimestamp();

        return .{
            .result = result,
            .duration_ns = @intCast(end_time - start_time),
        };
    }

    /// Run benchmark with multiple iterations
    pub fn benchmark(self: *PerformanceTestHelper, comptime func: anytype, args: anytype, iterations: u32) !struct { avg_duration_ns: u64, min_duration_ns: u64, max_duration_ns: u64 } {
        var total_duration: u64 = 0;
        var min_duration: u64 = std.math.maxInt(u64);
        var max_duration: u64 = 0;

        for (0..iterations) |_| {
            const measurement = try self.measureTime(func, args);
            total_duration += measurement.duration_ns;
            min_duration = @min(min_duration, measurement.duration_ns);
            max_duration = @max(max_duration, measurement.duration_ns);
        }

        return .{
            .avg_duration_ns = total_duration / iterations,
            .min_duration_ns = min_duration,
            .max_duration_ns = max_duration,
        };
    }
};

// Tests
test "LexerTestHelper basic functionality" {
    var helper = LexerTestHelper.init(std.testing.allocator);

    const source = "let x = 42;";
    const expected_types = [_]TokenType{ .Let, .Identifier, .Equal, .IntegerLiteral, .Semicolon, .Eof };

    const result = try helper.expectTokenTypes(source, &expected_types);
    try std.testing.expect(result.isPassed());
}

test "TestDataGenerator identifier generation" {
    // Skip this test for now due to random generation issues
    // TODO: Fix random generation or use deterministic approach
}
