//! Test Assertions - Compiler-specific assertion functions
//!
//! Provides specialized assertion functions for tokens, AST nodes, and types
//! with detailed error reporting and diff generation.

const std = @import("std");
const Allocator = std.mem.Allocator;
const TestResult = @import("test_result.zig").TestResult;
const TestFailure = @import("test_result.zig").TestFailure;
const SourceLocation = @import("test_result.zig").SourceLocation;

// Import compiler types
const ora = @import("ora");
const Token = ora.Token;
const TokenType = ora.TokenType;
const AstNode = ora.AstNode;
const ExprNode = ora.ExprNode;
const TypeInfo = ora.TypeInfo;

/// Assert that two tokens are equal
pub fn assertTokenEqual(allocator: Allocator, expected: Token, actual: Token) !TestResult {
    if (expected.type != actual.type) {
        const message = try std.fmt.allocPrint(allocator, "Token types differ", .{});
        const expected_str = try std.fmt.allocPrint(allocator, "{s}", .{@tagName(expected.type)});
        const actual_str = try std.fmt.allocPrint(allocator, "{s}", .{@tagName(actual.type)});
        const diff = try generateTokenDiff(allocator, expected, actual);

        return TestResult.failed(TestFailure.withDiff(message, expected_str, actual_str, diff));
    }

    if (!std.mem.eql(u8, expected.lexeme, actual.lexeme)) {
        const message = try std.fmt.allocPrint(allocator, "Token lexemes differ", .{});
        const diff = try generateTokenDiff(allocator, expected, actual);

        return TestResult.failed(TestFailure.withDiff(message, expected.lexeme, actual.lexeme, diff));
    }

    return TestResult.passed();
}

/// Assert that a token has the expected type
pub fn assertTokenType(allocator: Allocator, expected_type: TokenType, actual: Token) !TestResult {
    _ = allocator;
    if (expected_type != actual.type) {
        // Use a simple static message to avoid memory leaks in tests
        return TestResult.failed(TestFailure.simple("Token type mismatch"));
    }

    return TestResult.passed();
}

/// Assert that a token has the expected lexeme
pub fn assertTokenLexeme(allocator: Allocator, expected_lexeme: []const u8, actual: Token) !TestResult {
    if (!std.mem.eql(u8, expected_lexeme, actual.lexeme)) {
        const message = try std.fmt.allocPrint(allocator, "Token lexeme mismatch");
        const diff = try generateStringDiff(allocator, expected_lexeme, actual.lexeme);

        return TestResult.failed(TestFailure.withDiff(message, expected_lexeme, actual.lexeme, diff));
    }

    return TestResult.passed();
}

/// Assert that a token sequence matches expected types
pub fn assertTokenSequence(allocator: Allocator, expected_types: []const TokenType, actual_tokens: []const Token) !TestResult {
    if (expected_types.len != actual_tokens.len) {
        const message = try std.fmt.allocPrint(allocator, "Token sequence length mismatch: expected {}, got {}", .{ expected_types.len, actual_tokens.len });

        return TestResult.failed(TestFailure.simple(message));
    }

    for (expected_types, actual_tokens, 0..) |expected_type, actual_token, i| {
        if (expected_type != actual_token.type) {
            const message = try std.fmt.allocPrint(allocator, "Token type mismatch at position {}: expected {s}, got {s}", .{ i, @tagName(expected_type), @tagName(actual_token.type) });

            return TestResult.failed(TestFailure.simple(message));
        }
    }

    return TestResult.passed();
}

/// Assert that source positions are accurate
pub fn assertTokenPosition(allocator: Allocator, expected_line: u32, expected_column: u32, actual: Token) !TestResult {
    if (actual.line != expected_line or actual.column != expected_column) {
        const message = try std.fmt.allocPrint(allocator, "Token position mismatch: expected {}:{}, got {}:{}", .{ expected_line, expected_column, actual.line, actual.column });

        return TestResult.failed(TestFailure.simple(message));
    }

    return TestResult.passed();
}

/// Assert that an AST node has the expected type
pub fn assertAstNodeType(allocator: Allocator, expected_type: std.meta.Tag(AstNode), actual: *const AstNode) !TestResult {
    const actual_type = std.meta.activeTag(actual.*);
    if (expected_type != actual_type) {
        const message = try std.fmt.allocPrint(allocator, "AST node type mismatch: expected {s}, got {s}", .{ @tagName(expected_type), @tagName(actual_type) });

        return TestResult.failed(TestFailure.simple(message));
    }

    return TestResult.passed();
}

/// Assert that an AST node has valid source span
pub fn assertValidSourceSpan(allocator: Allocator, node: *const AstNode, source: []const u8) !TestResult {
    const span = getNodeSourceSpan(node);

    if (span.start_offset >= source.len) {
        const message = try std.fmt.allocPrint(allocator, "Source span start offset {} exceeds source length {}", .{ span.start_offset, source.len });

        return TestResult.failed(TestFailure.simple(message));
    }

    if (span.end_offset > source.len) {
        const message = try std.fmt.allocPrint(allocator, "Source span end offset {} exceeds source length {}", .{ span.end_offset, source.len });

        return TestResult.failed(TestFailure.simple(message));
    }

    if (span.start_offset > span.end_offset) {
        const message = try std.fmt.allocPrint(allocator, "Invalid source span: start {} > end {}", .{ span.start_offset, span.end_offset });

        return TestResult.failed(TestFailure.simple(message));
    }

    return TestResult.passed();
}

/// Assert that type information is consistent
pub fn assertTypeInfoConsistent(allocator: Allocator, node: *const AstNode) !TestResult {
    const type_info = getNodeTypeInfo(node);

    if (type_info == null) {
        const message = try std.fmt.allocPrint(allocator, "AST node missing type information");
        return TestResult.failed(TestFailure.simple(message));
    }

    // Additional type consistency checks can be added here
    return TestResult.passed();
}

/// Assert that two AST nodes are structurally equal
pub fn assertAstNodesEqual(allocator: Allocator, expected: *const AstNode, actual: *const AstNode) !TestResult {
    if (!astNodesEqual(expected, actual)) {
        const message = try std.fmt.allocPrint(allocator, "AST nodes are not structurally equal");
        const diff = try generateAstDiff(allocator, expected, actual);

        return TestResult.failed(TestFailure.withDiff(message, "expected AST", "actual AST", diff));
    }

    return TestResult.passed();
}

/// Assert that an expression has the expected type
pub fn assertExpressionType(allocator: Allocator, expected_type: TypeInfo, actual_expr: *const ExprNode) !TestResult {
    const actual_type = getExpressionTypeInfo(actual_expr);

    if (actual_type == null) {
        const message = try std.fmt.allocPrint(allocator, "Expression missing type information");
        return TestResult.failed(TestFailure.simple(message));
    }

    if (!typeInfoEqual(expected_type, actual_type.?)) {
        const message = try std.fmt.allocPrint(allocator, "Expression type mismatch");
        const expected_str = try formatTypeInfo(allocator, expected_type);
        const actual_str = try formatTypeInfo(allocator, actual_type.?);

        return TestResult.failed(TestFailure.comparison(message, expected_str, actual_str));
    }

    return TestResult.passed();
}

/// Generate diff between two tokens
fn generateTokenDiff(allocator: Allocator, expected: Token, actual: Token) ![]u8 {
    var buffer = std.ArrayList(u8).init(allocator);
    const writer = buffer.writer();

    try writer.writeAll("Token differences:\n");

    if (expected.type != actual.type) {
        try writer.print("  Type: expected {s}, got {s}\n", .{ @tagName(expected.type), @tagName(actual.type) });
    }

    if (!std.mem.eql(u8, expected.lexeme, actual.lexeme)) {
        try writer.print("  Lexeme: expected \"{s}\", got \"{s}\"\n", .{ expected.lexeme, actual.lexeme });
    }

    if (expected.line != actual.line or expected.column != actual.column) {
        try writer.print("  Position: expected {d}:{d}, got {d}:{d}\n", .{ expected.line, expected.column, actual.line, actual.column });
    }

    return buffer.toOwnedSlice();
}

/// Generate diff between two strings
fn generateStringDiff(allocator: Allocator, expected: []const u8, actual: []const u8) ![]u8 {
    var buffer = std.ArrayList(u8).init(allocator);
    const writer = buffer.writer();

    try writer.writeAll("String differences:\n");
    try writer.print("- Expected: \"{s}\"\n", .{expected});
    try writer.print("+ Actual:   \"{s}\"\n", .{actual});

    // Simple character-by-character diff
    const min_len = @min(expected.len, actual.len);
    var diff_start: ?usize = null;

    for (0..min_len) |i| {
        if (expected[i] != actual[i]) {
            diff_start = i;
            break;
        }
    }

    if (diff_start) |start| {
        try writer.print("  First difference at position {d}: expected '{c}', got '{c}'\n", .{ start, expected[start], actual[start] });
    } else if (expected.len != actual.len) {
        try writer.print("  Length difference: expected {d}, got {d}\n", .{ expected.len, actual.len });
    }

    return buffer.toOwnedSlice();
}

/// Generate diff between two AST nodes
fn generateAstDiff(allocator: Allocator, expected: *const AstNode, actual: *const AstNode) ![]u8 {
    var buffer = std.ArrayList(u8).init(allocator);
    const writer = buffer.writer();

    try writer.writeAll("AST node differences:\n");

    const expected_type = std.meta.activeTag(expected.*);
    const actual_type = std.meta.activeTag(actual.*);

    if (expected_type != actual_type) {
        try writer.print("  Node type: expected {s}, got {s}\n", .{ @tagName(expected_type), @tagName(actual_type) });
    }

    // TODO: Add more detailed AST comparison
    try writer.writeAll("  (Detailed AST comparison not yet implemented)\n");

    return buffer.toOwnedSlice();
}

/// Helper functions for AST node inspection
fn getNodeSourceSpan(_: *const AstNode) struct { start_offset: u32, end_offset: u32 } {
    // This is a simplified implementation - actual implementation would depend on AST structure
    return .{ .start_offset = 0, .end_offset = 0 };
}

fn getNodeTypeInfo(node: *const AstNode) ?TypeInfo {
    _ = node;
    // This is a placeholder - actual implementation would extract type info from node
    return null;
}

fn getExpressionTypeInfo(expr: *const ExprNode) ?TypeInfo {
    _ = expr;
    // This is a placeholder - actual implementation would extract type info from expression
    return null;
}

fn astNodesEqual(a: *const AstNode, b: *const AstNode) bool {
    _ = a;
    _ = b;
    // This is a placeholder - actual implementation would do deep structural comparison
    return false;
}

fn typeInfoEqual(a: TypeInfo, b: TypeInfo) bool {
    _ = a;
    _ = b;
    // This is a placeholder - actual implementation would compare type info
    return false;
}

fn formatTypeInfo(allocator: Allocator, type_info: TypeInfo) ![]u8 {
    _ = type_info;
    // This is a placeholder - actual implementation would format type info
    return try allocator.dupe(u8, "TypeInfo");
}

// Tests
test "assertTokenEqual basic functionality" {
    const token1 = Token{
        .type = .Identifier,
        .lexeme = "test",
        .range = .{
            .start_offset = 0,
            .end_offset = 4,
            .start_line = 1,
            .start_column = 1,
            .end_line = 1,
            .end_column = 5,
        },
        .line = 1,
        .column = 1,
    };

    const token2 = Token{
        .type = .Identifier,
        .lexeme = "test",
        .range = .{
            .start_offset = 0,
            .end_offset = 4,
            .start_line = 1,
            .start_column = 1,
            .end_line = 1,
            .end_column = 5,
        },
        .line = 1,
        .column = 1,
    };

    const result = try assertTokenEqual(std.testing.allocator, token1, token2);
    try std.testing.expect(result.isPassed());
}

test "assertTokenType functionality" {
    const token = Token{
        .type = .Identifier,
        .lexeme = "test",
        .range = .{
            .start_offset = 0,
            .end_offset = 4,
            .start_line = 1,
            .start_column = 1,
            .end_line = 1,
            .end_column = 5,
        },
        .line = 1,
        .column = 1,
    };

    const result = try assertTokenType(std.testing.allocator, .Identifier, token);
    try std.testing.expect(result.isPassed());

    const fail_result = try assertTokenType(std.testing.allocator, .StringLiteral, token);
    try std.testing.expect(fail_result.isFailed());
}
