//! Parser Invalid Fixture Suite - walks invalid parser fixtures and asserts parse failure

const std = @import("std");
const ora = @import("ora");
const Lexer = ora.lexer.Lexer;
const ParserCore = @import("../../src/parser/parser_core.zig");
const Parser = ParserCore.Parser;
const Diag = @import("../../src/parser/diagnostics.zig");
const AstArena = @import("ora").ast_arena.AstArena;

fn expectParseFailure(allocator: std.mem.Allocator, path: []const u8) !void {
    const file = try std.fs.cwd().openFile(path, .{});
    defer file.close();
    const content = try file.readToEndAlloc(allocator, 16 * 1024 * 1024);
    defer allocator.free(content);

    var lexer_instance = Lexer.init(allocator, content);
    defer lexer_instance.deinit();
    const tokens_or_err = lexer_instance.scanTokens();

    // If lexing failed, that's acceptable for invalid fixtures
    if (tokens_or_err) |tokens_ret| {
        defer allocator.free(tokens_ret);
        // If lexing succeeded but diagnostics exist, treat as failure as expected
        if (lexer_instance.getDiagnostics().len > 0) return; // expected failure

        // If no tokens produced (defensive), treat as failure
        if (tokens_ret.len == 0) return; // expected failure

        // Otherwise attempt to parse and expect an error
        var arena = AstArena.init(allocator);
        defer arena.deinit();
        var parser = Parser.init(tokens_ret, &arena);
        if (parser.parse()) |_| {
            // Unexpected success
            std.debug.print("Unexpected parse success for invalid fixture: {s}\n", .{path});
            return error.UnexpectedSuccess;
        } else |_| {
            // Expected parse error
            return;
        }
    } else |_| {
        // Lexing threw; acceptable for invalid
        return;
    }
}

test "Parser invalid fixtures" {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    // Silence parser diagnostics for invalid suite to keep output clean
    Diag.enable_stderr_diagnostics = false;

    const dirs = [_][]const u8{
        "tests/fixtures/parser_invalid/expressions",
        "tests/fixtures/parser_invalid/statements",
        "tests/fixtures/parser_invalid/declarations",
    };

    for (dirs) |dir_path| {
        // Skip missing dirs gracefully
        var dir = std.fs.cwd().openDir(dir_path, .{ .iterate = true }) catch |err| switch (err) {
            error.FileNotFound => continue,
            else => return err,
        };
        defer dir.close();
        var walker = try dir.walk(allocator);
        defer walker.deinit();
        while (try walker.next()) |entry| {
            if (entry.kind != .file) continue;
            if (!std.mem.endsWith(u8, entry.basename, ".ora")) continue;
            const path = try std.fs.path.join(allocator, &.{ dir_path, entry.path });
            defer allocator.free(path);
            try expectParseFailure(allocator, path);
        }
    }
}
