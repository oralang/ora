// ============================================================================
// Statement Analyzer Tests
// ============================================================================

const std = @import("std");
const testing = std.testing;
const ora_root = @import("ora_root");
const lexer = ora_root.lexer;
const parser_mod = ora_root.parser;
const semantics = ora_root.semantics;
const AstArena = ora_root.AstArena;

fn analyzeSource(allocator: std.mem.Allocator, source: []const u8) ![]semantics.Diagnostic {
    var arena = AstArena.init(allocator);
    defer arena.deinit();

    var lex = lexer.Lexer.init(allocator, source);
    defer lex.deinit();
    const tokens = try lex.scanTokens();
    defer allocator.free(tokens);

    var parser = parser_mod.Parser.init(tokens, &arena);
    const nodes = try parser.parse();

    var sem = try semantics.analyze(allocator, nodes);
    defer sem.symbols.deinit();
    return sem.diagnostics;
}

test "try requires error union operand" {
    const allocator = testing.allocator;
    const source =
        \\fn ok() -> u256 { return 1; }
        \\fn caller() -> u256 {
        \\  let x = try ok();
        \\  return x;
        \\}
    ;

    const diags = try analyzeSource(allocator, source);
    defer allocator.free(diags);
    try testing.expect(diags.len > 0);
}

test "error union call must be handled in expression statement" {
    const allocator = testing.allocator;
    const source =
        \\error Fail;
        \\fn mayFail() -> !u256 { return error.Fail; }
        \\fn caller() {
        \\  mayFail();
        \\}
    ;

    const diags = try analyzeSource(allocator, source);
    defer allocator.free(diags);
    try testing.expect(diags.len > 0);
}

test "error union assignment to non error type is rejected" {
    const allocator = testing.allocator;
    const source =
        \\error Fail;
        \\fn mayFail() -> !u256 { return error.Fail; }
        \\fn caller() {
        \\  let x: u256 = mayFail();
        \\  _ = x;
        \\}
    ;

    const diags = try analyzeSource(allocator, source);
    defer allocator.free(diags);
    try testing.expect(diags.len > 0);
}

test "error union can be assigned to error union variable" {
    const allocator = testing.allocator;
    const source =
        \\error Fail;
        \\fn mayFail() -> !u256 { return error.Fail; }
        \\fn caller() -> !u256 {
        \\  let x: !u256 = mayFail();
        \\  return x;
        \\}
    ;

    const diags = try analyzeSource(allocator, source);
    defer allocator.free(diags);
    try testing.expectEqual(@as(usize, 0), diags.len);
}
