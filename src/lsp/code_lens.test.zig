const std = @import("std");
const testing = std.testing;
const ora_root = @import("ora_root");

const code_lens = ora_root.lsp.code_lens;
const compiler = ora_root.compiler;
const line_index = ora_root.lsp.line_index;

test "lsp code lens: reports verification clauses" {
    const source =
        \\contract Wallet {
        \\    invariant(balance >= 0);
        \\    storage var balance: u256;
        \\
        \\    pub fn deposit(amount: u256) -> u256
        \\        requires(amount > 0)
        \\        ensures(result >= amount)
        \\    {
        \\        return amount;
        \\    }
        \\}
    ;

    const lenses = try code_lens.findVerificationLenses(testing.allocator, source);
    defer code_lens.deinitLenses(testing.allocator, lenses);

    try testing.expectEqual(@as(usize, 2), lenses.len);
    try testing.expectEqualStrings("contract invariant: 1 clause", lenses[0].title);
    try testing.expectEqualStrings("1 requires, 1 ensures", lenses[1].title);
}

test "lsp code lens: cached AST path reports verification clauses" {
    const source =
        \\contract Wallet {
        \\    invariant(balance >= 0);
        \\    storage var balance: u256;
        \\
        \\    pub fn deposit(amount: u256) -> u256
        \\        requires(amount > 0)
        \\        ensures(result >= amount)
        \\    {
        \\        return amount;
        \\    }
        \\}
    ;

    var parse_result = try compiler.syntax.parse(testing.allocator, compiler.FileId.fromIndex(0), source);
    defer parse_result.deinit();

    var lower_result = try compiler.ast.lower(testing.allocator, &parse_result.tree);
    defer lower_result.deinit();

    var lines = try line_index.LineIndex.init(testing.allocator, source);
    defer lines.deinit(testing.allocator);

    const lenses = try code_lens.findVerificationLensesInAst(testing.allocator, source, &lower_result.file, &lines);
    defer code_lens.deinitLenses(testing.allocator, lenses);

    try testing.expectEqual(@as(usize, 2), lenses.len);
    try testing.expectEqualStrings("contract invariant: 1 clause", lenses[0].title);
    try testing.expectEqualStrings("1 requires, 1 ensures", lenses[1].title);
}

test "lsp code lens: propagates allocator failures instead of returning empty lenses" {
    const source =
        \\contract Wallet {
        \\    invariant(balance >= 0);
        \\    storage var balance: u256;
        \\}
    ;

    var observed_induced_failure = false;
    for (0..64) |fail_index| {
        var backing_arena = std.heap.ArenaAllocator.init(testing.allocator);
        defer backing_arena.deinit();

        var failing = testing.FailingAllocator.init(backing_arena.allocator(), .{ .fail_index = fail_index });
        const allocator = failing.allocator();

        if (code_lens.findVerificationLenses(allocator, source)) |lenses| {
            code_lens.deinitLenses(allocator, lenses);
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
