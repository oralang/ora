const std = @import("std");
const testing = std.testing;
const ora_root = @import("ora_root");

const compiler = ora_root.compiler;
const folding = ora_root.lsp.folding;
const line_index = ora_root.lsp.line_index;

test "lsp folding: returns comment and AST region folds" {
    const source =
        \\// first
        \\// second
        \\contract Wallet {
        \\    pub fn value() -> u256 {
        \\        return 1;
        \\    }
        \\}
    ;

    const ranges = try folding.foldingRanges(testing.allocator, source);
    defer folding.deinitRanges(testing.allocator, ranges);

    var found_comment = false;
    var found_contract = false;
    var found_function = false;
    for (ranges) |range| {
        if (range.kind == .comment and range.start_line == 0 and range.end_line == 1) {
            found_comment = true;
        }
        if (range.kind == .region and range.start_line == 2 and range.end_line == 6) {
            found_contract = true;
        }
        if (range.kind == .region and range.start_line == 3 and range.end_line == 5) {
            found_function = true;
        }
    }

    try testing.expect(found_comment);
    try testing.expect(found_contract);
    try testing.expect(found_function);
}

test "lsp folding: cached AST path returns comment and AST region folds" {
    const source =
        \\// first
        \\// second
        \\contract Wallet {
        \\    pub fn value() -> u256 {
        \\        return 1;
        \\    }
        \\}
    ;

    var parse_result = try compiler.syntax.parse(testing.allocator, compiler.FileId.fromIndex(0), source);
    defer parse_result.deinit();

    var lower_result = try compiler.ast.lower(testing.allocator, &parse_result.tree);
    defer lower_result.deinit();

    var lines = try line_index.LineIndex.init(testing.allocator, source);
    defer lines.deinit(testing.allocator);

    const ranges = try folding.foldingRangesInAst(testing.allocator, source, &lower_result.file, &lines);
    defer folding.deinitRanges(testing.allocator, ranges);

    var found_comment = false;
    var found_contract = false;
    var found_function = false;
    for (ranges) |range| {
        if (range.kind == .comment and range.start_line == 0 and range.end_line == 1) {
            found_comment = true;
        }
        if (range.kind == .region and range.start_line == 2 and range.end_line == 6) {
            found_contract = true;
        }
        if (range.kind == .region and range.start_line == 3 and range.end_line == 5) {
            found_function = true;
        }
    }

    try testing.expect(found_comment);
    try testing.expect(found_contract);
    try testing.expect(found_function);
}

test "lsp folding: propagates allocator failures instead of returning partial folds" {
    const source =
        \\// first
        \\// second
        \\contract Wallet {
        \\    storage var balance: u256;
        \\}
    ;

    var observed_induced_failure = false;
    for (0..96) |fail_index| {
        var backing_arena = std.heap.ArenaAllocator.init(testing.allocator);
        defer backing_arena.deinit();

        var failing = testing.FailingAllocator.init(backing_arena.allocator(), .{ .fail_index = fail_index });
        const allocator = failing.allocator();

        if (folding.foldingRanges(allocator, source)) |ranges| {
            folding.deinitRanges(allocator, ranges);
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
