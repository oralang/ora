const std = @import("std");
const testing = std.testing;
const ora_root = @import("ora_root");

const lexer = ora_root.lexer;
const semantic_tokens = ora_root.lsp.semantic_tokens;
const token_cache = ora_root.lsp.token_cache;

fn cachedTokens(
    allocator: std.mem.Allocator,
    source: []const u8,
    maybe_index: ?ora_root.lsp.semantic_index.SemanticIndex,
) ![]semantic_tokens.SemanticToken {
    var cache = try token_cache.Cache.init(allocator, source);
    defer cache.deinit(allocator);
    return semantic_tokens.tokenizeCached(allocator, source, cache.tokens, maybe_index);
}

test "semantic tokens: cached tokenizer handles syntax-broken source lexically" {
    const source =
        \\contract Broken {
        \\    pub fn value() -> u256 {
        \\        return "unterminated;
        \\    }
        \\}
    ;

    const tokens = try cachedTokens(testing.allocator, source, null);
    defer testing.allocator.free(tokens);

    try testing.expect(tokens.len > 0);
    var saw_keyword = false;
    for (tokens) |token| {
        if (token.kind == .keyword) saw_keyword = true;
    }
    try testing.expect(saw_keyword);
}

test "semantic tokens: every lexer keyword lexeme classifies" {
    const keyword_keys = lexer.keywords.kvs.keys[0..lexer.keywords.kvs.len];
    for (keyword_keys) |keyword| {
        const tokens = try cachedTokens(testing.allocator, keyword, null);
        defer testing.allocator.free(tokens);

        try expectSemanticTokenForLexeme(keyword, tokens, keyword, null);
    }
}

test "semantic tokens: keyword classification follows lexer lexemes" {
    const source = "and && or || comptime u160 void return";
    const tokens = try cachedTokens(testing.allocator, source, null);
    defer testing.allocator.free(tokens);

    try expectSemanticTokenForLexeme(source, tokens, "and", .keyword);
    try expectSemanticTokenForLexeme(source, tokens, "&&", .operator);
    try expectSemanticTokenForLexeme(source, tokens, "or", .keyword);
    try expectSemanticTokenForLexeme(source, tokens, "||", .operator);
    try expectSemanticTokenForLexeme(source, tokens, "comptime", .macro);
    try expectSemanticTokenForLexeme(source, tokens, "u160", .type);
    try expectSemanticTokenForLexeme(source, tokens, "void", .type);
    try expectSemanticTokenForLexeme(source, tokens, "return", .keyword);
}

test "semantic tokens: refinement type names classify as default library types" {
    const source = "NonZeroAddress MinValue";
    const tokens = try cachedTokens(testing.allocator, source, null);
    defer testing.allocator.free(tokens);

    try expectSemanticTokenForLexeme(source, tokens, "NonZeroAddress", .type);
    try expectSemanticTokenForLexeme(source, tokens, "MinValue", .type);

    for (tokens) |token| {
        if (token.line == 0 and token.start_char == 0) {
            try testing.expect((token.modifiers & semantic_tokens.SemanticTokenModifier.mask(.defaultLibrary)) != 0);
            return;
        }
    }
    return error.TestExpectedEqual;
}

test "semantic tokens: Resource and at-prefixed builtins classify as default library" {
    const source = "@move Resource @cast";
    const tokens = try cachedTokens(testing.allocator, source, null);
    defer testing.allocator.free(tokens);

    const move = try semanticTokenForLexeme(source, tokens, "move");
    try testing.expectEqual(semantic_tokens.SemanticTokenKind.function, move.kind);
    try testing.expect((move.modifiers & semantic_tokens.SemanticTokenModifier.mask(.defaultLibrary)) != 0);

    const resource = try semanticTokenForLexeme(source, tokens, "Resource");
    try testing.expectEqual(semantic_tokens.SemanticTokenKind.type, resource.kind);
    try testing.expect((resource.modifiers & semantic_tokens.SemanticTokenModifier.mask(.defaultLibrary)) != 0);

    const cast = try semanticTokenForLexeme(source, tokens, "cast");
    try testing.expectEqual(semantic_tokens.SemanticTokenKind.function, cast.kind);
    try testing.expect((cast.modifiers & semantic_tokens.SemanticTokenModifier.mask(.defaultLibrary)) != 0);
}

test "semantic tokens: builtin environment namespaces and constants classify as default library" {
    const source = "msg.sender i256.MAX std.msg.value std.constants.U256_MAX";
    const tokens = try cachedTokens(testing.allocator, source, null);
    defer testing.allocator.free(tokens);

    const msg = try semanticTokenForLexeme(source, tokens, "msg");
    try testing.expectEqual(semantic_tokens.SemanticTokenKind.namespace, msg.kind);
    try testing.expect((msg.modifiers & semantic_tokens.SemanticTokenModifier.mask(.defaultLibrary)) != 0);

    const sender = try semanticTokenForLexeme(source, tokens, "sender");
    try testing.expectEqual(semantic_tokens.SemanticTokenKind.property, sender.kind);
    try testing.expect((sender.modifiers & semantic_tokens.SemanticTokenModifier.mask(.defaultLibrary)) != 0);
    try testing.expect((sender.modifiers & semantic_tokens.SemanticTokenModifier.mask(.readonly)) != 0);

    const u256_max = try semanticTokenForLexeme(source, tokens, "U256_MAX");
    try testing.expectEqual(semantic_tokens.SemanticTokenKind.variable, u256_max.kind);
    try testing.expect((u256_max.modifiers & semantic_tokens.SemanticTokenModifier.mask(.defaultLibrary)) != 0);
    try testing.expect((u256_max.modifiers & semantic_tokens.SemanticTokenModifier.mask(.readonly)) != 0);

    const max = try semanticTokenForLexeme(source, tokens, "MAX");
    try testing.expectEqual(semantic_tokens.SemanticTokenKind.variable, max.kind);
    try testing.expect((max.modifiers & semantic_tokens.SemanticTokenModifier.mask(.defaultLibrary)) != 0);
    try testing.expect((max.modifiers & semantic_tokens.SemanticTokenModifier.mask(.readonly)) != 0);
}

test "semantic tokens: cached tokenizer propagates allocator failure" {
    const source =
        \\contract Wallet {
        \\    storage var balance: u256;
        \\    pub fn deposit(amount: u256) -> u256 {
        \\        let next: u256 = balance + amount;
        \\        return next;
        \\    }
        \\}
    ;

    var observed_induced_failure = false;
    for (0..128) |fail_index| {
        var backing_arena = std.heap.ArenaAllocator.init(testing.allocator);
        defer backing_arena.deinit();

        var failing = testing.FailingAllocator.init(backing_arena.allocator(), .{ .fail_index = fail_index });
        const allocator = failing.allocator();

        if (cachedTokens(allocator, source, null)) |tokens| {
            allocator.free(tokens);
            try testing.expect(!failing.has_induced_failure);
            if (observed_induced_failure) break;
        } else |err| switch (err) {
            error.OutOfMemory => {
                try testing.expect(failing.has_induced_failure);
                observed_induced_failure = true;
            },
            else => return err,
        }
    }

    try testing.expect(observed_induced_failure);
}

fn expectSemanticTokenForLexeme(
    source: []const u8,
    tokens: []const semantic_tokens.SemanticToken,
    lexeme: []const u8,
    expected: ?semantic_tokens.SemanticTokenKind,
) !void {
    const offset = std.mem.indexOf(u8, source, lexeme) orelse return error.TestExpectedEqual;
    const start_char: u32 = @intCast(offset);
    const length: u32 = @intCast(lexeme.len);

    for (tokens) |token| {
        if (token.line != 0 or token.start_char != start_char or token.length != length) continue;
        if (expected) |kind| {
            try testing.expectEqual(kind, token.kind);
            return;
        }
        return;
    }

    return error.TestExpectedEqual;
}

fn semanticTokenForLexeme(
    source: []const u8,
    tokens: []const semantic_tokens.SemanticToken,
    lexeme: []const u8,
) !semantic_tokens.SemanticToken {
    const offset = std.mem.indexOf(u8, source, lexeme) orelse return error.TestExpectedEqual;
    const start_char: u32 = @intCast(offset);
    const length: u32 = @intCast(lexeme.len);

    for (tokens) |token| {
        if (token.line == 0 and token.start_char == start_char and token.length == length) {
            return token;
        }
    }

    return error.TestExpectedEqual;
}
