// ============================================================================
// ABI Tests
// ============================================================================

const std = @import("std");
const testing = std.testing;
const ora_root = @import("ora_root");
const lexer = ora_root.lexer;
const parser = ora_root.parser;
const ast_arena = ora_root.ast_arena;
const abi = ora_root.abi;

fn generateAbiForSource(
    allocator: std.mem.Allocator,
    source: []const u8,
    arena: *ast_arena.AstArena,
) !abi.ContractAbi {
    var lex = lexer.Lexer.init(allocator, source);
    defer lex.deinit();
    const tokens = try lex.scanTokens();
    defer allocator.free(tokens);

    var parser_instance = parser.Parser.init(tokens, arena);
    const nodes = try parser_instance.parse();
    defer arena.allocator().free(nodes);

    var generator = try abi.AbiGenerator.init(allocator);
    defer generator.deinit();

    return try generator.generate(nodes);
}

fn findCallable(contract_abi: *const abi.ContractAbi, kind: abi.CallableKind, name: []const u8) ?*const abi.AbiCallable {
    for (contract_abi.callables) |*callable| {
        if (callable.kind == kind and std.mem.eql(u8, callable.name, name)) {
            return callable;
        }
    }
    return null;
}

fn countCallables(contract_abi: *const abi.ContractAbi, kind: abi.CallableKind, name: []const u8) usize {
    var count: usize = 0;
    for (contract_abi.callables) |callable| {
        if (callable.kind == kind and std.mem.eql(u8, callable.name, name)) {
            count += 1;
        }
    }
    return count;
}

fn hasEffectKind(callable: *const abi.AbiCallable, kind: abi.AbiEffectKind) bool {
    for (callable.effects) |effect| {
        if (effect.kind == kind) return true;
    }
    return false;
}

test "abi manifest includes functions errors events effects and hashed type ids" {
    const allocator = testing.allocator;
    const source =
        \\contract Test {
        \\    storage var counter: u256;
        \\    error InvalidAmount(amount: u256);
        \\    log Transfer(indexed from: address, amount: u256);
        \\    pub fn pureFn() -> u256 { let x: u256 = 1; return x; }
        \\    pub fn viewFn() -> u256 { return counter; }
        \\    pub fn writeFn() { counter = 1; }
        \\    pub fn transferLike(from: address, amount: u256) { counter = amount; }
        \\}
    ;

    var arena = ast_arena.AstArena.init(allocator);
    defer arena.deinit();

    var contract_abi = try generateAbiForSource(allocator, source, &arena);
    defer contract_abi.deinit();

    try testing.expectEqual(@as(usize, 1), contract_abi.contract_count);
    try testing.expectEqualStrings("Test", contract_abi.contract_name);

    const pure_fn = findCallable(&contract_abi, .function, "pureFn") orelse return error.TestUnexpectedResult;
    const view_fn = findCallable(&contract_abi, .function, "viewFn") orelse return error.TestUnexpectedResult;
    const write_fn = findCallable(&contract_abi, .function, "writeFn") orelse return error.TestUnexpectedResult;
    const invalid_amount = findCallable(&contract_abi, .@"error", "InvalidAmount") orelse return error.TestUnexpectedResult;
    const transfer = findCallable(&contract_abi, .event, "Transfer") orelse return error.TestUnexpectedResult;

    try testing.expect(pure_fn.selector != null);
    try testing.expectEqual(@as(usize, 1), pure_fn.outputs.len);
    try testing.expectEqual(@as(usize, 0), pure_fn.effects.len);

    try testing.expect(view_fn.selector != null);
    try testing.expect(hasEffectKind(view_fn, .reads));

    try testing.expect(write_fn.selector != null);
    try testing.expect(hasEffectKind(write_fn, .writes));

    try testing.expect(invalid_amount.selector != null);
    try testing.expectEqual(@as(usize, 1), invalid_amount.inputs.len);

    try testing.expect(transfer.selector == null);
    try testing.expectEqual(@as(usize, 2), transfer.inputs.len);
    try testing.expect(transfer.inputs[0].indexed orelse false);
    try testing.expect(!(transfer.inputs[1].indexed orelse false));

    for (contract_abi.types) |typ| {
        const type_id = typ.type_id orelse return error.TestUnexpectedResult;
        try testing.expect(std.mem.startsWith(u8, type_id, "t:"));
    }

    var u256_count: usize = 0;
    for (contract_abi.types) |typ| {
        if (typ.name) |name| {
            if (std.mem.eql(u8, name, "u256")) u256_count += 1;
        }
    }
    try testing.expectEqual(@as(usize, 1), u256_count);

    const manifest_json = try contract_abi.toJson(allocator);
    defer allocator.free(manifest_json);

    try testing.expect(std.mem.indexOf(u8, manifest_json, "\"schemaVersion\":\"ora-abi-0.1\"") != null);
    try testing.expect(std.mem.indexOf(u8, manifest_json, "\"contract\":{\"name\":\"Test\"") != null);
    try testing.expect(std.mem.indexOf(u8, manifest_json, "\"wireProfiles\":[{\"id\":\"evm-default\"") != null);
    try testing.expect(std.mem.indexOf(u8, manifest_json, "\"kind\":\"error\"") != null);
    try testing.expect(std.mem.indexOf(u8, manifest_json, "\"kind\":\"event\"") != null);
    try testing.expect(std.mem.indexOf(u8, manifest_json, "\"kind\":\"reads\"") != null);
    try testing.expect(std.mem.indexOf(u8, manifest_json, "\"kind\":\"writes\"") != null);
    try testing.expect(std.mem.indexOf(u8, manifest_json, "\"group\":\"Read\"") == null);
    try testing.expect(std.mem.indexOf(u8, manifest_json, "\"dangerLevel\"") == null);
    try testing.expect(std.mem.indexOf(u8, manifest_json, "\"messageTemplate\"") == null);

    const extras_json = try contract_abi.toExtrasJson(allocator);
    defer allocator.free(extras_json);
    try testing.expect(std.mem.indexOf(u8, extras_json, "\"schemaVersion\":\"ora-abi-extras-0.1\"") != null);
    try testing.expect(std.mem.indexOf(u8, extras_json, "\"baseSchemaVersion\":\"ora-abi-0.1\"") != null);
    try testing.expect(std.mem.indexOf(u8, extras_json, "\"ui\":{\"group\":\"Read\",\"dangerLevel\":\"info\"") != null);
    try testing.expect(std.mem.indexOf(u8, extras_json, "\"ui\":{\"group\":\"Write\",\"dangerLevel\":\"normal\"") != null);
    try testing.expect(std.mem.indexOf(u8, extras_json, "\"forms\":{\"from\":{\"widget\":\"address\"},\"amount\":{\"widget\":\"number\"}}") != null);
    try testing.expect(std.mem.indexOf(u8, extras_json, "\"group\":\"Errors\"") != null);
    try testing.expect(std.mem.indexOf(u8, extras_json, "\"messageTemplate\":\"InvalidAmount: amount={amount}\"") != null);
    try testing.expect(std.mem.indexOf(u8, extras_json, "\"group\":\"Events\"") != null);
    try testing.expect(std.mem.indexOf(u8, extras_json, "\"ui\":{\"widget\":\"number\"}") != null);

    const solidity_json = try contract_abi.toSolidityJson(allocator);
    defer allocator.free(solidity_json);

    try testing.expect(std.mem.indexOf(u8, solidity_json, "\"stateMutability\":\"pure\"") != null);
    try testing.expect(std.mem.indexOf(u8, solidity_json, "\"stateMutability\":\"view\"") != null);
    try testing.expect(std.mem.indexOf(u8, solidity_json, "\"stateMutability\":\"nonpayable\"") != null);
    try testing.expect(std.mem.indexOf(u8, solidity_json, "\"type\":\"error\"") != null);
    try testing.expect(std.mem.indexOf(u8, solidity_json, "\"type\":\"event\"") != null);
}

test "abi emits one bundle for multiple contracts and disambiguates callable ids" {
    const allocator = testing.allocator;
    const source =
        \\contract A {
        \\    error Boom(code: u256);
        \\    pub fn ping() {}
        \\}
        \\contract B {
        \\    error Boom(code: u256);
        \\    pub fn ping() {}
        \\}
    ;

    var arena = ast_arena.AstArena.init(allocator);
    defer arena.deinit();

    var contract_abi = try generateAbiForSource(allocator, source, &arena);
    defer contract_abi.deinit();

    try testing.expectEqual(@as(usize, 2), contract_abi.contract_count);
    try testing.expectEqualStrings("bundle", contract_abi.contract_name);

    try testing.expectEqual(@as(usize, 2), countCallables(&contract_abi, .function, "ping"));
    try testing.expectEqual(@as(usize, 2), countCallables(&contract_abi, .@"error", "Boom"));

    var found_a_fn = false;
    var found_b_fn = false;
    var found_a_error = false;
    var found_b_error = false;

    for (contract_abi.callables) |callable| {
        if (std.mem.eql(u8, callable.id, "c:A.ping()")) found_a_fn = true;
        if (std.mem.eql(u8, callable.id, "c:B.ping()")) found_b_fn = true;
        if (std.mem.eql(u8, callable.id, "c:A.Boom(uint256)")) found_a_error = true;
        if (std.mem.eql(u8, callable.id, "c:B.Boom(uint256)")) found_b_error = true;
    }

    try testing.expect(found_a_fn);
    try testing.expect(found_b_fn);
    try testing.expect(found_a_error);
    try testing.expect(found_b_error);
}

test "abi type ids are stable across repeated generation for same source" {
    const allocator = testing.allocator;
    const source =
        \\contract Stable {
        \\    pub fn f(a: u256, b: address) -> u256 { return a; }
        \\}
    ;

    var arena_a = ast_arena.AstArena.init(allocator);
    defer arena_a.deinit();
    var abi_a = try generateAbiForSource(allocator, source, &arena_a);
    defer abi_a.deinit();

    var arena_b = ast_arena.AstArena.init(allocator);
    defer arena_b.deinit();
    var abi_b = try generateAbiForSource(allocator, source, &arena_b);
    defer abi_b.deinit();

    try testing.expectEqual(abi_a.types.len, abi_b.types.len);

    for (abi_a.types) |left| {
        const left_id = left.type_id orelse return error.TestUnexpectedResult;
        var found = false;
        for (abi_b.types) |right| {
            const right_id = right.type_id orelse return error.TestUnexpectedResult;
            if (std.mem.eql(u8, left_id, right_id)) {
                found = true;
                break;
            }
        }
        try testing.expect(found);
    }

    const json_a = try abi_a.toJson(allocator);
    defer allocator.free(json_a);
    const json_b = try abi_b.toJson(allocator);
    defer allocator.free(json_b);

    try testing.expectEqualStrings(json_a, json_b);
}
