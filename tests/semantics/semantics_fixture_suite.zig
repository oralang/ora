const std = @import("std");
const ora = @import("ora");
const Lexer = ora.lexer.Lexer;
const ParserCore = @import("../../src/parser/parser_core.zig");
const Parser = ParserCore.Parser;
const Semantics = @import("../../src/semantics.zig");
const Core = @import("../../src/semantics/core.zig");
const AstArena = @import("ora").ast_arena.AstArena;

fn analyzePath(allocator: std.mem.Allocator, path: []const u8) ![]Core.Diagnostic {
    const file = try std.fs.cwd().openFile(path, .{});
    defer file.close();
    const src = try file.readToEndAlloc(allocator, 16 * 1024 * 1024);
    defer allocator.free(src);

    var lex = Lexer.init(allocator, src);
    defer lex.deinit();
    const tokens_or_err = lex.scanTokens();
    if (tokens_or_err) |tokens| {
        defer allocator.free(tokens);

        var arena = AstArena.init(allocator);
        defer arena.deinit();

        var parser = Parser.init(tokens, &arena);
        const nodes = parser.parse() catch return error.ParseFailed;

        var result = try Semantics.analyze(allocator, nodes);
        const diags = result.diagnostics;
        // Cleanup symbols to avoid leaks under GPA
        result.symbols.deinit();
        return diags;
    } else |_| {
        return error.LexFailed;
    }
}

fn runDir(allocator: std.mem.Allocator, dir_path: []const u8, expect_ok: bool) !void {
    var dir = try std.fs.cwd().openDir(dir_path, .{ .iterate = true });
    defer dir.close();
    var walker = try dir.walk(allocator);
    defer walker.deinit();
    while (try walker.next()) |entry| {
        if (entry.kind != .file) continue;
        if (!std.mem.endsWith(u8, entry.basename, ".ora")) continue;
        const path = try std.fs.path.join(allocator, &.{ dir_path, entry.path });
        defer allocator.free(path);
        const diags = analyzePath(allocator, path) catch |e| switch (e) {
            error.LexFailed, error.ParseFailed => {
                // Fixtures should at least lex/parse; treat as failure
                return e;
            },
            else => return e,
        };
        defer allocator.free(diags);
        if (expect_ok) {
            if (diags.len != 0) {
                std.debug.print("Unexpected diagnostics for {s}:\n", .{path});
                for (diags) |d| std.debug.print("  - {s} at {d}:{d}\n", .{ d.message, d.span.line, d.span.column });
                return error.UnexpectedDiagnostics;
            }
        } else {
            if (diags.len == 0) {
                std.debug.print("Expected diagnostics but got none for {s}\n", .{path});
                return error.ExpectedDiagnostics;
            }
        }
    }
}

test "Semantics fixtures: valid should have no diagnostics" {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    try runDir(gpa.allocator(), "tests/fixtures/semantics/valid", true);
}

test "Semantics fixtures: invalid should have diagnostics" {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    try runDir(gpa.allocator(), "tests/fixtures/semantics/invalid", false);
}
