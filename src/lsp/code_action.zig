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
    var first_match: ?usize = null;
    for (diagnostics, 0..) |diagnostic, diagnostic_index| {
        if (isMissingSemicolonDiagnostic(diagnostic)) {
            first_match = diagnostic_index;
            break;
        }
    }
    const start_index = first_match orelse return null;

    const results = try arena.alloc(CodeActionOrCommand, diagnostics.len - start_index);
    results[0] = .{ .CodeAction = try missingSemicolonAction(arena, uri, diagnostics[start_index]) };
    var out_index: usize = 1;
    for (diagnostics[start_index + 1 ..]) |diagnostic| {
        if (!isMissingSemicolonDiagnostic(diagnostic)) continue;
        results[out_index] = .{
            .CodeAction = try missingSemicolonAction(arena, uri, diagnostic),
        };
        out_index += 1;
    }

    return results[0..out_index];
}

fn isMissingSemicolonDiagnostic(diagnostic: types.Diagnostic) bool {
    return std.mem.startsWith(u8, diagnostic.message, "expected ';'");
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
        .diagnostics = null,
        .edit = .{ .changes = changes },
        .isPreferred = true,
    };
}
