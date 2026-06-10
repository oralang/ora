const std = @import("std");
const testing = std.testing;
const ora_root = @import("ora_root");

const frontend = ora_root.lsp.frontend;
const inlay_hints = ora_root.lsp.inlay_hints;
const line_index = ora_root.lsp.line_index;
const test_analysis = @import("test_analysis.zig");

fn cachedHints(allocator: std.mem.Allocator, source: []const u8, range: frontend.Range) ![]inlay_hints.InlayHint {
    var fixture: test_analysis.TestAnalysis = undefined;
    try fixture.init(allocator, source);
    defer fixture.deinit();

    var index = try fixture.buildSemanticIndex(allocator, source);
    defer index.deinit(allocator);

    var lines = try line_index.LineIndex.init(allocator, source);
    defer lines.deinit(allocator);

    return inlay_hints.hintsInRangeCached(
        allocator,
        source,
        range,
        &lines,
        .utf8,
        fixture.analysis.ast_file,
        &index,
    );
}

test "inlay hints: cached helper handles parse diagnostics without hidden failures" {
    const source =
        \\contract Broken {
        \\    pub fn value() -> u256 {
        \\        return "unterminated;
        \\    }
        \\}
    ;

    const hints = try cachedHints(testing.allocator, source, frontend.Range{
        .start = .{ .line = 0, .character = 0 },
        .end = .{ .line = 5, .character = 0 },
    });
    defer inlay_hints.deinitHints(testing.allocator, hints);

    try testing.expectEqual(@as(usize, 0), hints.len);
}

test "inlay hints: parameter hints resolve callable through semantic name map" {
    const source =
        \\pub fn helper(value: u256) -> u256 {
        \\    return value;
        \\}
        \\pub fn run() -> u256 {
        \\    return helper(1);
        \\}
    ;

    const hints = try cachedHints(testing.allocator, source, frontend.Range{
        .start = .{ .line = 0, .character = 0 },
        .end = .{ .line = 6, .character = 0 },
    });
    defer inlay_hints.deinitHints(testing.allocator, hints);

    var found = false;
    for (hints) |hint| {
        if (std.mem.eql(u8, hint.label, "value:")) found = true;
    }
    try testing.expect(found);
}

test "inlay hints: cached entrypoint reuses parsed AST and semantic index" {
    const source =
        \\pub fn helper(value: u256) -> u256 {
        \\    return value;
        \\}
        \\pub fn run() -> u256 {
        \\    return helper(1);
        \\}
    ;

    var fixture: test_analysis.TestAnalysis = undefined;
    try fixture.init(testing.allocator, source);
    defer fixture.deinit();

    var index = try fixture.buildSemanticIndex(testing.allocator, source);
    defer index.deinit(testing.allocator);

    var lines = try line_index.LineIndex.init(testing.allocator, source);
    defer lines.deinit(testing.allocator);

    const hints = try inlay_hints.hintsInRangeCached(
        testing.allocator,
        source,
        frontend.Range{
            .start = .{ .line = 0, .character = 0 },
            .end = .{ .line = 6, .character = 0 },
        },
        &lines,
        .utf16,
        fixture.analysis.ast_file,
        &index,
    );
    defer inlay_hints.deinitHints(testing.allocator, hints);

    var found = false;
    for (hints) |hint| {
        if (std.mem.eql(u8, hint.label, "value:")) found = true;
    }
    try testing.expect(found);
}

test "inlay hints: propagates parameter-name allocation failures" {
    const source =
        \\pub fn helper(value: u256) -> u256 {
        \\    return value;
        \\}
        \\pub fn run() -> u256 {
        \\    return helper(1);
        \\}
    ;

    const range = frontend.Range{
        .start = .{ .line = 0, .character = 0 },
        .end = .{ .line = 6, .character = 0 },
    };

    var observed_induced_failure = false;
    for (0..128) |fail_index| {
        var backing_arena = std.heap.ArenaAllocator.init(testing.allocator);
        defer backing_arena.deinit();

        var failing = testing.FailingAllocator.init(backing_arena.allocator(), .{ .fail_index = fail_index });
        const allocator = failing.allocator();

        if (cachedHints(allocator, source, range)) |hints| {
            inlay_hints.deinitHints(allocator, hints);
            try testing.expect(!failing.has_induced_failure);
            if (observed_induced_failure) break;
        } else |err| switch (err) {
            error.OutOfMemory => {
                try testing.expect(failing.has_induced_failure);
                observed_induced_failure = true;
            },
            else => return err,
        }
    }

    try testing.expect(observed_induced_failure);
}
