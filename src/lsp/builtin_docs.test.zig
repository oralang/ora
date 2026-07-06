const std = @import("std");
const ora_root = @import("ora_root");

const builtin_docs = ora_root.lsp.builtin_docs;
const compiler = ora_root.compiler;

test "lsp builtin docs: every builtin has signature docs and example" {
    for (builtin_docs.entries) |entry| {
        try std.testing.expect(entry.name.len != 0);
        try std.testing.expect(entry.signature.len != 0);
        try std.testing.expect(entry.documentation.len != 0);
        try std.testing.expect(entry.example.len != 0);
        try std.testing.expect(builtin_docs.entryForName(entry.name) != null);
    }
}

test "lsp builtin docs: formats markdown with example" {
    const entry = builtin_docs.entryForName("keccak256") orelse return error.TestExpectedEqual;
    const markdown = try builtin_docs.markdownAlloc(std.testing.allocator, entry);
    defer std.testing.allocator.free(markdown);

    try std.testing.expect(std.mem.indexOf(u8, markdown, "Keccak-256") != null);
    try std.testing.expect(std.mem.indexOf(u8, markdown, "```ora") != null);
}

test "lsp builtin docs: covers cast and resource builtins" {
    inline for (.{ "cast", "amount", "move", "create", "destroy" }) |name| {
        const entry = builtin_docs.entryForName(name) orelse return error.TestExpectedEqual;
        try std.testing.expect(std.mem.startsWith(u8, entry.signature, "@"));
        try std.testing.expect(entry.documentation.len > 0);
        try std.testing.expect(entry.example.len > 0);
    }
}

test "lsp builtin docs: every example snippet parses" {
    for (builtin_docs.entries) |entry| {
        try expectExampleParses(entry);
    }
}

fn expectExampleParses(entry: builtin_docs.Entry) !void {
    const source = try std.fmt.allocPrint(
        std.testing.allocator,
        \\contract BuiltinDocExample {{
        \\    pub fn run() {{
        \\        {s}
        \\    }}
        \\}}
    ,
        .{entry.example},
    );
    defer std.testing.allocator.free(source);

    var db = compiler.CompilerDb.init(std.testing.allocator);
    defer db.deinitFrontendOnly();

    const file_id = try db.addSourceFile("<builtin-doc-example>", source);
    _ = try db.astFile(file_id);

    if (!(try db.syntaxDiagnostics(file_id)).isEmpty()) {
        std.debug.print("builtin example parse failed for @{s}: {s}\n", .{ entry.name, entry.example });
        return error.TestUnexpectedResult;
    }

    const ast_diagnostics = try db.astDiagnostics(file_id);
    if (!ast_diagnostics.isEmpty()) {
        std.debug.print("builtin example lowering failed for @{s}: {s}\n", .{ entry.name, entry.example });
        for (ast_diagnostics.items.items) |diagnostic| {
            std.debug.print("  {s}\n", .{diagnostic.message});
        }
        return error.TestUnexpectedResult;
    }
}
