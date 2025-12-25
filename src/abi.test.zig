// ============================================================================
// ABI Tests
// ============================================================================
//
// tests for ABI generation and mutability inference.
//
// ============================================================================

const std = @import("std");
const testing = std.testing;
const ora_root = @import("ora_root");
const lexer = ora_root.lexer;
const parser = ora_root.parser;
const ast_arena = ora_root.ast_arena;
const abi = ora_root.abi;

test "abi includes storage effects and gas estimate" {
    const allocator = testing.allocator;
    const source =
        \\contract Test {
        \\    storage var counter: u256;
        \\    pub fn pureFn() -> u256 { let x: u256 = 1; return x; }
        \\    pub fn viewFn() -> u256 { return counter; }
        \\    pub fn writeFn() { counter = 1; }
        \\}
    ;

    var lex = lexer.Lexer.init(allocator, source);
    defer lex.deinit();
    const tokens = try lex.scanTokens();
    defer allocator.free(tokens);

    var arena = ast_arena.AstArena.init(allocator);
    defer arena.deinit();
    var parser_instance = parser.Parser.init(tokens, &arena);

    const nodes = try parser_instance.parse();
    defer arena.allocator().free(nodes);

    var generator = try abi.AbiGenerator.init(allocator);
    defer generator.deinit();

    var contract_abi = try generator.generate(nodes);
    defer contract_abi.deinit();
    const solidity_json = try contract_abi.toSolidityJson(allocator);
    defer allocator.free(solidity_json);

    var write_seen = false;
    var read_seen = false;
    var gas_seen = false;

    for (contract_abi.functions) |func| {
        if (std.mem.eql(u8, func.name, "pureFn")) {
            try testing.expect(func.effects.len == 0);
            try testing.expect(func.gas_estimate != null);
            gas_seen = true;
        } else if (std.mem.eql(u8, func.name, "viewFn")) {
            read_seen = true;
            try testing.expect(func.effects.len > 0);
            try testing.expect(func.gas_estimate != null);
        } else if (std.mem.eql(u8, func.name, "writeFn")) {
            write_seen = true;
            try testing.expect(func.effects.len > 0);
            try testing.expect(func.gas_estimate != null);
        }
    }

    try testing.expect(read_seen and write_seen and gas_seen);
    try testing.expect(std.mem.indexOf(u8, solidity_json, "\"stateMutability\": \"view\"") != null);
    try testing.expect(std.mem.indexOf(u8, solidity_json, "\"stateMutability\": \"nonpayable\"") != null);
}
