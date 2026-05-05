const std = @import("std");
const testing = std.testing;
const ora_root = @import("ora_root");

const semantic_index = ora_root.lsp.semantic_index;

fn findSymbolIndex(
    symbols: []const semantic_index.Symbol,
    name: []const u8,
    kind: semantic_index.SymbolKind,
) ?usize {
    for (symbols, 0..) |symbol, i| {
        if (symbol.kind == kind and std.mem.eql(u8, symbol.name, name)) {
            return i;
        }
    }
    return null;
}

test "lsp semantic index: collects declarations and scope links" {
    const source =
        \\contract Wallet {
        \\    storage var balance: u256;
        \\    pub fn deposit(amount: u256) { }
        \\}
        \\pub fn helper(value: u256) { }
        \\struct Point { x: u256; y: u256; }
        \\enum Mode { Off, On }
    ;

    var index = try semantic_index.indexDocument(testing.allocator, source);
    defer index.deinit(testing.allocator);

    try testing.expect(index.parse_succeeded);

    const wallet_idx = findSymbolIndex(index.symbols, "Wallet", .contract) orelse return error.TestExpectedEqual;
    const deposit_idx = findSymbolIndex(index.symbols, "deposit", .method) orelse return error.TestExpectedEqual;
    const amount_idx = findSymbolIndex(index.symbols, "amount", .parameter) orelse return error.TestExpectedEqual;
    const balance_idx = findSymbolIndex(index.symbols, "balance", .field) orelse return error.TestExpectedEqual;
    const helper_idx = findSymbolIndex(index.symbols, "helper", .function) orelse return error.TestExpectedEqual;
    const helper_value_idx = findSymbolIndex(index.symbols, "value", .parameter) orelse return error.TestExpectedEqual;
    const point_idx = findSymbolIndex(index.symbols, "Point", .struct_decl) orelse return error.TestExpectedEqual;
    const point_x_idx = findSymbolIndex(index.symbols, "x", .field) orelse return error.TestExpectedEqual;
    const mode_idx = findSymbolIndex(index.symbols, "Mode", .enum_decl) orelse return error.TestExpectedEqual;
    const mode_off_idx = findSymbolIndex(index.symbols, "Off", .enum_member) orelse return error.TestExpectedEqual;

    try testing.expectEqual(@as(?usize, null), index.symbols[wallet_idx].parent);
    try testing.expectEqual(@as(?usize, wallet_idx), index.symbols[deposit_idx].parent);
    try testing.expectEqual(@as(?usize, deposit_idx), index.symbols[amount_idx].parent);
    try testing.expectEqual(@as(?usize, wallet_idx), index.symbols[balance_idx].parent);
    try testing.expectEqual(@as(?usize, null), index.symbols[helper_idx].parent);
    try testing.expectEqual(@as(?usize, helper_idx), index.symbols[helper_value_idx].parent);
    try testing.expectEqual(@as(?usize, null), index.symbols[point_idx].parent);
    try testing.expectEqual(@as(?usize, point_idx), index.symbols[point_x_idx].parent);
    try testing.expectEqual(@as(?usize, null), index.symbols[mode_idx].parent);
    try testing.expectEqual(@as(?usize, mode_idx), index.symbols[mode_off_idx].parent);
}

test "lsp semantic index: parse failure returns no symbols" {
    const source = "@import(\"std\");";

    var index = try semantic_index.indexDocument(testing.allocator, source);
    defer index.deinit(testing.allocator);

    try testing.expect(!index.parse_succeeded);
    try testing.expectEqual(@as(usize, 0), index.symbols.len);
}

test "lsp semantic index: builds nested document symbols" {
    const source =
        \\contract Wallet {
        \\    pub fn deposit(amount: u256) { }
        \\}
    ;

    var index = try semantic_index.indexDocument(testing.allocator, source);
    defer index.deinit(testing.allocator);

    try testing.expect(index.parse_succeeded);

    const doc_symbols = try semantic_index.buildDocumentSymbols(testing.allocator, index.symbols);
    defer semantic_index.deinitDocumentSymbols(testing.allocator, doc_symbols);

    try testing.expectEqual(@as(usize, 1), doc_symbols.len);
    try testing.expectEqualStrings("Wallet", doc_symbols[0].name);
    try testing.expectEqual(@as(u8, 5), doc_symbols[0].kind);
    try testing.expectEqual(@as(usize, 1), doc_symbols[0].children.len);
    try testing.expectEqualStrings("deposit", doc_symbols[0].children[0].name);
    try testing.expectEqual(@as(u8, 6), doc_symbols[0].children[0].kind);
    try testing.expectEqual(@as(usize, 1), doc_symbols[0].children[0].children.len);
    try testing.expectEqualStrings("amount", doc_symbols[0].children[0].children[0].name);
}

test "lsp semantic index: collects traits impls aliases and ADT variant details" {
    const source =
        \\error InsufficientBalance;
        \\type Amount = u256;
        \\enum Event: u8 {
        \\    Value(u256) = 1,
        \\    Pair { left: u256, right: u256 } = 2,
        \\}
        \\struct Vault {}
        \\extern trait Token {
        \\    call fn transfer(self, to: address, amount: Amount) -> bool errors(InsufficientBalance);
        \\}
        \\impl Token for Vault {
        \\    fn transfer(self, to: address, amount: Amount) -> bool { return true; }
        \\}
    ;

    var index = try semantic_index.indexDocument(testing.allocator, source);
    defer index.deinit(testing.allocator);

    try testing.expect(index.parse_succeeded);

    const amount_idx = findSymbolIndex(index.symbols, "Amount", .type_alias) orelse return error.TestExpectedEqual;
    const event_idx = findSymbolIndex(index.symbols, "Event", .enum_decl) orelse return error.TestExpectedEqual;
    const value_idx = findSymbolIndex(index.symbols, "Value", .enum_member) orelse return error.TestExpectedEqual;
    const pair_idx = findSymbolIndex(index.symbols, "Pair", .enum_member) orelse return error.TestExpectedEqual;
    const token_idx = findSymbolIndex(index.symbols, "Token", .trait_decl) orelse return error.TestExpectedEqual;
    const impl_idx = findSymbolIndex(index.symbols, "impl Token for Vault", .impl_decl) orelse return error.TestExpectedEqual;
    const trait_transfer_idx = findSymbolIndex(index.symbols, "transfer", .method) orelse return error.TestExpectedEqual;

    try testing.expectEqualStrings("u256", index.symbols[amount_idx].detail.?);
    try testing.expectEqualStrings(": u8", index.symbols[event_idx].detail.?);
    try testing.expectEqualStrings("(u256) = 1", index.symbols[value_idx].detail.?);
    try testing.expectEqualStrings(" { left: u256, right: u256 } = 2", index.symbols[pair_idx].detail.?);
    try testing.expectEqual(@as(?usize, event_idx), index.symbols[value_idx].parent);
    try testing.expectEqual(@as(?usize, event_idx), index.symbols[pair_idx].parent);
    try testing.expectEqual(@as(?usize, null), index.symbols[token_idx].parent);
    try testing.expectEqual(@as(?usize, null), index.symbols[impl_idx].parent);
    try testing.expectEqual(@as(?usize, token_idx), index.symbols[trait_transfer_idx].parent);
    try testing.expect(std.mem.indexOf(u8, index.symbols[trait_transfer_idx].detail.?, "errors(InsufficientBalance)") != null);
}
