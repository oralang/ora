const std = @import("std");
const lsp = @import("lsp");
const code_action = @import("code_action.zig");

const types = lsp.types;

test "lsp code action: builds missing semicolon quick fix" {
    const uri: types.DocumentUri = "file:///wallet.ora";
    const diagnostics = [_]types.Diagnostic{.{
        .range = .{
            .start = .{ .line = 3, .character = 12 },
            .end = .{ .line = 3, .character = 18 },
        },
        .severity = .Error,
        .source = "ora-parser",
        .message = "expected ';'",
    }};

    var response_arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer response_arena.deinit();

    const actions = (try code_action.build(response_arena.allocator(), uri, &diagnostics)) orelse return error.ExpectedActions;

    try std.testing.expectEqual(@as(usize, 1), actions.len);
    const action = actions[0].CodeAction;
    try std.testing.expectEqualStrings("Insert missing ';'", action.title);
    try std.testing.expectEqual(@as(?types.CodeActionKind, .quickfix), action.kind);
    try std.testing.expect(action.isPreferred orelse false);

    const edit = action.edit.?.changes.?.map.get(uri) orelse return error.ExpectedEdit;
    try std.testing.expectEqual(@as(usize, 1), edit.len);
    try std.testing.expectEqualStrings(";", edit[0].newText);
    try std.testing.expectEqual(diagnostics[0].range.end.line, edit[0].range.start.line);
    try std.testing.expectEqual(diagnostics[0].range.end.character, edit[0].range.start.character);
}

test "lsp code action: returns null when diagnostics have no supported action" {
    const diagnostics = [_]types.Diagnostic{.{
        .range = .{
            .start = .{ .line = 0, .character = 0 },
            .end = .{ .line = 0, .character = 4 },
        },
        .severity = .Warning,
        .source = "ora",
        .message = "unused variable",
    }};

    var response_arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer response_arena.deinit();

    const actions = try code_action.build(response_arena.allocator(), "file:///wallet.ora", &diagnostics);
    try std.testing.expect(actions == null);
}
