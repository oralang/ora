// ============================================================================
// Error Recovery Grouping Tests
// ============================================================================

const std = @import("std");
const testing = std.testing;
const ora_root = @import("ora_root");
const lexer = ora_root.lexer;

const ErrorRecovery = lexer.ErrorRecovery;
const LexerError = lexer.LexerError;
const SourceRange = lexer.SourceRange;

fn makeRange(line: u32, column: u32) SourceRange {
    return .{
        .start_line = line,
        .start_column = column,
        .end_line = line,
        .end_column = column + 1,
        .start_offset = 0,
        .end_offset = 1,
    };
}

test "error recovery groups nearby errors and separates distant ones" {
    const allocator = testing.allocator;
    var recovery = ErrorRecovery.init(allocator, 10);
    defer recovery.deinit();

    try recovery.recordError(LexerError.UnexpectedCharacter, makeRange(1, 1), "bad char");
    try recovery.recordError(LexerError.UnterminatedString, makeRange(2, 4), "unterminated");
    try recovery.recordError(LexerError.InvalidHexLiteral, makeRange(10, 2), "invalid hex");

    var groups = try recovery.groupErrors();
    defer {
        for (groups.items) |*group| {
            group.related.deinit(allocator);
        }
        groups.deinit(allocator);
    }

    try testing.expectEqual(@as(usize, 2), groups.items.len);
    try testing.expectEqual(@as(usize, 1), groups.items[0].related.items.len);
    try testing.expectEqual(@as(usize, 0), groups.items[1].related.items.len);
}

test "error recovery groups same-type errors even when far apart" {
    const allocator = testing.allocator;
    var recovery = ErrorRecovery.init(allocator, 10);
    defer recovery.deinit();

    try recovery.recordError(LexerError.UnexpectedCharacter, makeRange(1, 1), "bad char");
    try recovery.recordError(LexerError.UnexpectedCharacter, makeRange(20, 3), "bad char again");

    var groups = try recovery.groupErrors();
    defer {
        for (groups.items) |*group| {
            group.related.deinit(allocator);
        }
        groups.deinit(allocator);
    }

    try testing.expectEqual(@as(usize, 1), groups.items.len);
    try testing.expectEqual(@as(usize, 1), groups.items[0].related.items.len);
}

test "error recovery groups different types on the same line" {
    const allocator = testing.allocator;
    var recovery = ErrorRecovery.init(allocator, 10);
    defer recovery.deinit();

    try recovery.recordError(LexerError.InvalidHexLiteral, makeRange(3, 2), "invalid hex");
    try recovery.recordError(LexerError.UnterminatedString, makeRange(3, 12), "unterminated");

    var groups = try recovery.groupErrors();
    defer {
        for (groups.items) |*group| {
            group.related.deinit(allocator);
        }
        groups.deinit(allocator);
    }

    try testing.expectEqual(@as(usize, 1), groups.items.len);
    try testing.expectEqual(@as(usize, 1), groups.items[0].related.items.len);
}
