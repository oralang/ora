const std = @import("std");
const testing = std.testing;
const ora_root = @import("ora_root");

const text_edits = ora_root.lsp.text_edits;

test "lsp text edits: full change replaces document" {
    const changes = [_]text_edits.Change{
        .{ .full = .{ .text = "new text" } },
    };

    const updated = try text_edits.applyChangesAlloc(testing.allocator, "old text", changes[0..]);
    defer testing.allocator.free(updated);

    try testing.expectEqualStrings("new text", updated);
}

test "lsp text edits: incremental insertion" {
    const changes = [_]text_edits.Change{
        .{ .incremental = .{
            .range = .{
                .start = .{ .line = 0, .character = 5 },
                .end = .{ .line = 0, .character = 5 },
            },
            .text = " brave",
        } },
    };

    const updated = try text_edits.applyChangesAlloc(testing.allocator, "hello world", changes[0..]);
    defer testing.allocator.free(updated);

    try testing.expectEqualStrings("hello brave world", updated);
}

test "lsp text edits: incremental multiline replace" {
    const changes = [_]text_edits.Change{
        .{ .incremental = .{
            .range = .{
                .start = .{ .line = 1, .character = 0 },
                .end = .{ .line = 1, .character = 4 },
            },
            .text = "zig",
        } },
    };

    const updated = try text_edits.applyChangesAlloc(testing.allocator, "alpha\nbeta\ngamma", changes[0..]);
    defer testing.allocator.free(updated);

    try testing.expectEqualStrings("alpha\nzig\ngamma", updated);
}

test "lsp text edits: applies multiple changes in order" {
    const changes = [_]text_edits.Change{
        .{ .incremental = .{
            .range = .{
                .start = .{ .line = 0, .character = 0 },
                .end = .{ .line = 0, .character = 5 },
            },
            .text = "hello",
        } },
        .{ .incremental = .{
            .range = .{
                .start = .{ .line = 0, .character = 5 },
                .end = .{ .line = 0, .character = 5 },
            },
            .text = " world",
        } },
    };

    const updated = try text_edits.applyChangesAlloc(testing.allocator, "start", changes[0..]);
    defer testing.allocator.free(updated);

    try testing.expectEqualStrings("hello world", updated);
}

test "lsp text edits: invalid range returns error" {
    const changes = [_]text_edits.Change{
        .{ .incremental = .{
            .range = .{
                .start = .{ .line = 2, .character = 0 },
                .end = .{ .line = 2, .character = 1 },
            },
            .text = "x",
        } },
    };

    try testing.expectError(
        text_edits.ApplyError.InvalidRange,
        text_edits.applyChangesAlloc(testing.allocator, "one line", changes[0..]),
    );
}

test "lsp text edits: utf16 handles surrogate pair insertion" {
    const changes = [_]text_edits.Change{
        .{ .incremental = .{
            .range = .{
                .start = .{ .line = 0, .character = 3 },
                .end = .{ .line = 0, .character = 3 },
            },
            .text = "X",
        } },
    };

    const updated = try text_edits.applyChangesAllocWithEncoding(
        testing.allocator,
        "a😀b",
        changes[0..],
        .utf16,
    );
    defer testing.allocator.free(updated);

    try testing.expectEqualStrings("a😀Xb", updated);
}

test "lsp text edits: utf16 rejects split surrogate positions" {
    const changes = [_]text_edits.Change{
        .{ .incremental = .{
            .range = .{
                .start = .{ .line = 0, .character = 2 },
                .end = .{ .line = 0, .character = 2 },
            },
            .text = "X",
        } },
    };

    try testing.expectError(
        text_edits.ApplyError.InvalidRange,
        text_edits.applyChangesAllocWithEncoding(
            testing.allocator,
            "a😀b",
            changes[0..],
            .utf16,
        ),
    );
}
