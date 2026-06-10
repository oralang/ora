const std = @import("std");
const lsp = @import("lsp");

const Allocator = std.mem.Allocator;
const types = lsp.types;
pub const CodeActionOrCommand = @typeInfo(@typeInfo(lsp.ResultType("textDocument/codeAction")).optional.child).pointer.child;

pub fn build(
    arena: Allocator,
    uri: types.DocumentUri,
    diagnostics: []const types.Diagnostic,
) !lsp.ResultType("textDocument/codeAction") {
    var results = std.ArrayList(CodeActionOrCommand){};
    errdefer results.deinit(arena);

    for (diagnostics) |diagnostic| {
        if (std.mem.indexOf(u8, diagnostic.message, "expected ';'") != null) {
            try results.append(arena, .{ .CodeAction = try missingSemicolonAction(arena, uri, diagnostic) });
        }
    }

    if (results.items.len == 0) return null;
    return try results.toOwnedSlice(arena);
}

fn missingSemicolonAction(
    arena: Allocator,
    uri: types.DocumentUri,
    diagnostic: types.Diagnostic,
) !types.CodeAction {
    const insert_pos = diagnostic.range.end;
    const edits = try arena.alloc(types.TextEdit, 1);
    edits[0] = .{ .range = .{ .start = insert_pos, .end = insert_pos }, .newText = ";" };

    var changes: lsp.parser.Map(types.DocumentUri, []const types.TextEdit) = .{};
    try changes.map.put(arena, uri, edits);

    return .{
        .title = "Insert missing ';'",
        .kind = .quickfix,
        .diagnostics = try arena.dupe(types.Diagnostic, &.{diagnostic}),
        .edit = .{ .changes = changes },
        .isPreferred = true,
    };
}
