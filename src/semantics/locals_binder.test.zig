// ============================================================================
// Locals Binder Tests
// ============================================================================

const std = @import("std");
const testing = std.testing;
const ora_root = @import("ora_root");
const lexer = ora_root.lexer;
const parser_mod = ora_root.parser;
const semantics = ora_root.semantics;
const AstArena = ora_root.AstArena;

fn isErrorUnionType(ti: ora_root.ast.Types.TypeInfo) bool {
    if (ti.category == .ErrorUnion) return true;
    if (ti.ora_type) |ot| {
        return switch (ot) {
            .error_union => true,
            ._union => |members| members.len > 0 and members[0] == .error_union,
            else => false,
        };
    }
    return false;
}

test "catch variable uses error union type from try expression" {
    const allocator = testing.allocator;
    const source =
        \\error Foo;
        \\fn mayFail() -> !u256 {
        \\  return error.Foo;
        \\}
        \\fn caller() {
        \\  try {
        \\    let _ = try mayFail();
        \\  } catch (e) {
        \\    let x = e;
        \\  }
        \\}
    ;

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
    defer allocator.free(sem.diagnostics);

    var caller_fn: ?*ora_root.ast.FunctionNode = null;
    for (nodes) |*node| switch (node.*) {
        .Function => |*fn_node| {
            if (std.mem.eql(u8, fn_node.name, "caller")) {
                caller_fn = fn_node;
            }
        },
        else => {},
    };
    try testing.expect(caller_fn != null);

    var catch_block: ?*ora_root.ast.Statements.CatchBlock = null;
    for (caller_fn.?.body.statements) |*stmt| {
        if (stmt.* == .TryBlock) {
            if (stmt.TryBlock.catch_block) |*cb| {
                catch_block = cb;
                break;
            }
        }
    }
    try testing.expect(catch_block != null);

    const catch_key: usize = @intFromPtr(&catch_block.?.block);
    const catch_scope = sem.symbols.block_scopes.get(catch_key) orelse return error.TestExpectedEqual;
    const idx = catch_scope.findInCurrent("e") orelse return error.TestExpectedEqual;
    const sym = catch_scope.symbols.items[idx];
    try testing.expect(sym.typ != null);
    try testing.expect(isErrorUnionType(sym.typ.?));
}
