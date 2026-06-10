const std = @import("std");
const testing = std.testing;
const ora_root = @import("ora_root");

const compiler = ora_root.compiler;
const frontend = ora_root.lsp.frontend;
const inlay_hints = ora_root.lsp.inlay_hints;
const line_index = ora_root.lsp.line_index;
const semantic_index = ora_root.lsp.semantic_index;

test "inlay hints: standalone helper handles parse diagnostics without hidden failures" {
    const source =
        \\contract Broken {
        \\    pub fn value() -> u256 {
        \\        return "unterminated;
        \\    }
        \\}
    ;

    const hints = try inlay_hints.hintsInRange(testing.allocator, source, frontend.Range{
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

    const hints = try inlay_hints.hintsInRange(testing.allocator, source, frontend.Range{
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

    var sources = compiler.source.SourceStore.init(testing.allocator);
    defer sources.deinit();
    const file_id = try sources.addFile("<cached-inlay>", source);

    var parse_result = try compiler.syntax.parse(testing.allocator, file_id, source);
    defer parse_result.deinit();

    var lower_result = try compiler.ast.lower(testing.allocator, &parse_result.tree);
    defer lower_result.deinit();

    var index = try semantic_index.indexDocument(testing.allocator, source);
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
        &lower_result.file,
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

        if (inlay_hints.hintsInRange(allocator, source, range)) |hints| {
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
