const std = @import("std");
const testing = std.testing;
const ora_root = @import("ora_root");

const lexer = ora_root.lexer;
const semantic_tokens = ora_root.lsp.semantic_tokens;

test "semantic tokens: standalone tokenizer handles parse diagnostics without hiding errors" {
    const source =
        \\contract Broken {
        \\    pub fn value() -> u256 {
        \\        return "unterminated;
        \\    }
        \\}
    ;

    const tokens = try semantic_tokens.tokenize(testing.allocator, source);
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
        const tokens = try semantic_tokens.tokenizeWithIndex(testing.allocator, keyword, null);
        defer testing.allocator.free(tokens);

        try expectSemanticTokenForLexeme(keyword, tokens, keyword, null);
    }
}

test "semantic tokens: keyword classification follows lexer lexemes" {
    const source = "and && or || comptime u160 void return";
    const tokens = try semantic_tokens.tokenizeWithIndex(testing.allocator, source, null);
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

test "semantic tokens: standalone tokenizer propagates allocator failure" {
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

        if (semantic_tokens.tokenize(allocator, source)) |tokens| {
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
