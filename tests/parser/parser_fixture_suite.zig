//! Parser Fixture Suite - walks parser and integration fixtures and asserts parse success

const std = @import("std");
const ora = @import("ora");
const Lexer = ora.lexer.Lexer;
const ParserCore = @import("../../src/parser/parser_core.zig");
const Parser = ParserCore.Parser;
const AstArena = @import("../../src/ast/ast_arena.zig").AstArena;

fn parseFile(allocator: std.mem.Allocator, path: []const u8) !void {
    const file = try std.fs.cwd().openFile(path, .{});
    defer file.close();
    const content = try file.readToEndAlloc(allocator, 16 * 1024 * 1024); // 16MB cap
    defer allocator.free(content);

    var lexer_instance = Lexer.init(allocator, content);
    defer lexer_instance.deinit();
    _ = try lexer_instance.scanTokens();
    if (lexer_instance.getDiagnostics().len > 0) return error.LexerDiagnosticsPresent;

    var arena = AstArena.init(allocator);
    defer arena.deinit();

    var parser = Parser.init(lexer_instance.getTokens(), &arena);
    _ = try parser.parse();
}

test "Parser fixtures: declarations and contracts" {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    const dirs = [_][]const u8{
        "tests/fixtures/parser/expressions",
        "tests/fixtures/parser/statements",
        "tests/fixtures/parser/declarations",
        "tests/fixtures/parser/contracts",
        "tests/fixtures/integration/complex_programs",
    };

    for (dirs) |dir_path| {
        var dir = try std.fs.cwd().openDir(dir_path, .{ .iterate = true });
        defer dir.close();
        var walker = try dir.walk(allocator);
        defer walker.deinit();
        while (try walker.next()) |entry| {
            if (entry.kind != .file) continue;
            if (!std.mem.endsWith(u8, entry.basename, ".ora")) continue;
            const path = try std.fs.path.join(allocator, &.{ dir_path, entry.path });
            defer allocator.free(path);
            try parseFile(allocator, path);
        }
    }
}
