// ============================================================================
// Declaration Parser Tests
// ============================================================================
// Tests for parsing declarations (functions, contracts, types)
// ============================================================================

const std = @import("std");
const testing = std.testing;
const ora_root = @import("ora_root");
const lexer = ora_root.lexer;
const parser = ora_root.parser;
const ast = ora_root.ast;
const ast_arena = ora_root.ast_arena;

// ============================================================================
// Function Declaration Tests
// ============================================================================

test "declarations: public function without parameters" {
    const allocator = testing.allocator;
    const source = "pub fn foo() { }";

    var lex = lexer.Lexer.init(allocator, source);
    defer lex.deinit();
    const tokens = try lex.scanTokens();
    defer allocator.free(tokens);

    var arena = ast_arena.AstArena.init(allocator);
    defer arena.deinit();
    var parser_instance = parser.Parser.init(tokens, &arena);

    const nodes = try parser_instance.parse();
    defer arena.allocator().free(nodes);

    try testing.expect(nodes.len == 1);
    try testing.expect(nodes[0] == .Function);
    if (nodes[0] == .Function) {
        try testing.expect(std.mem.eql(u8, "foo", nodes[0].Function.name));
        try testing.expectEqual(ast.Visibility.Public, nodes[0].Function.visibility);
        try testing.expect(nodes[0].Function.parameters.len == 0);
    }
}

test "declarations: private function" {
    const allocator = testing.allocator;
    const source = "fn bar() { }";

    var lex = lexer.Lexer.init(allocator, source);
    defer lex.deinit();
    const tokens = try lex.scanTokens();
    defer allocator.free(tokens);

    var arena = ast_arena.AstArena.init(allocator);
    defer arena.deinit();
    var parser_instance = parser.Parser.init(tokens, &arena);

    const nodes = try parser_instance.parse();
    defer arena.allocator().free(nodes);

    try testing.expect(nodes.len == 1);
    try testing.expect(nodes[0] == .Function);
    if (nodes[0] == .Function) {
        try testing.expect(std.mem.eql(u8, "bar", nodes[0].Function.name));
        try testing.expectEqual(ast.Visibility.Private, nodes[0].Function.visibility);
    }
}

test "declarations: function with parameters" {
    const allocator = testing.allocator;
    const source = "pub fn add(x: u256, y: u256) -> u256 { return x + y; }";

    var lex = lexer.Lexer.init(allocator, source);
    defer lex.deinit();
    const tokens = try lex.scanTokens();
    defer allocator.free(tokens);

    var arena = ast_arena.AstArena.init(allocator);
    defer arena.deinit();
    var parser_instance = parser.Parser.init(tokens, &arena);

    const nodes = try parser_instance.parse();
    defer arena.allocator().free(nodes);

    try testing.expect(nodes.len == 1);
    try testing.expect(nodes[0] == .Function);
    if (nodes[0] == .Function) {
        try testing.expect(std.mem.eql(u8, "add", nodes[0].Function.name));
        try testing.expect(nodes[0].Function.parameters.len == 2);
        try testing.expect(nodes[0].Function.return_type_info != null);
    }
}

test "declarations: function with return type" {
    const allocator = testing.allocator;
    const source = "pub fn getValue() -> u256 { return 42; }";

    var lex = lexer.Lexer.init(allocator, source);
    defer lex.deinit();
    const tokens = try lex.scanTokens();
    defer allocator.free(tokens);

    var arena = ast_arena.AstArena.init(allocator);
    defer arena.deinit();
    var parser_instance = parser.Parser.init(tokens, &arena);

    const nodes = try parser_instance.parse();
    defer arena.allocator().free(nodes);

    try testing.expect(nodes.len == 1);
    try testing.expect(nodes[0] == .Function);
    if (nodes[0] == .Function) {
        try testing.expect(nodes[0].Function.return_type_info != null);
    }
}

// ============================================================================
// Contract Declaration Tests
// ============================================================================

test "declarations: empty contract" {
    const allocator = testing.allocator;
    const source = "contract Test { }";

    var lex = lexer.Lexer.init(allocator, source);
    defer lex.deinit();
    const tokens = try lex.scanTokens();
    defer allocator.free(tokens);

    var arena = ast_arena.AstArena.init(allocator);
    defer arena.deinit();
    var parser_instance = parser.Parser.init(tokens, &arena);

    const nodes = try parser_instance.parse();
    defer arena.allocator().free(nodes);

    try testing.expect(nodes.len == 1);
    try testing.expect(nodes[0] == .Contract);
    if (nodes[0] == .Contract) {
        try testing.expect(std.mem.eql(u8, "Test", nodes[0].Contract.name));
        try testing.expect(nodes[0].Contract.body.len == 0);
    }
}

test "declarations: contract with function" {
    const allocator = testing.allocator;
    const source =
        \\contract Test {
        \\    pub fn foo() { }
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

    try testing.expect(nodes.len == 1);
    try testing.expect(nodes[0] == .Contract);
    if (nodes[0] == .Contract) {
        try testing.expect(nodes[0].Contract.body.len >= 1);
    }
}

test "declarations: contract with storage variable" {
    const allocator = testing.allocator;
    const source =
        \\contract Test {
        \\    storage var counter: u256;
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

    try testing.expect(nodes.len == 1);
    try testing.expect(nodes[0] == .Contract);
    if (nodes[0] == .Contract) {
        try testing.expect(nodes[0].Contract.body.len >= 1);
    }
}

// ============================================================================
// Struct Declaration Tests
// ============================================================================

test "declarations: struct declaration" {
    const allocator = testing.allocator;
    const source = "struct Point { x: u256; y: u256; }";

    var lex = lexer.Lexer.init(allocator, source);
    defer lex.deinit();
    const tokens = try lex.scanTokens();
    defer allocator.free(tokens);

    var arena = ast_arena.AstArena.init(allocator);
    defer arena.deinit();
    var parser_instance = parser.Parser.init(tokens, &arena);

    const nodes = try parser_instance.parse();
    defer arena.allocator().free(nodes);

    try testing.expect(nodes.len == 1);
    try testing.expect(nodes[0] == .StructDecl);
    if (nodes[0] == .StructDecl) {
        try testing.expect(std.mem.eql(u8, "Point", nodes[0].StructDecl.name));
        try testing.expect(nodes[0].StructDecl.fields.len == 2);
    }
}

test "declarations: struct with single field" {
    const allocator = testing.allocator;
    const source = "struct Counter { value: u256; }";

    var lex = lexer.Lexer.init(allocator, source);
    defer lex.deinit();
    const tokens = try lex.scanTokens();
    defer allocator.free(tokens);

    var arena = ast_arena.AstArena.init(allocator);
    defer arena.deinit();
    var parser_instance = parser.Parser.init(tokens, &arena);

    const nodes = try parser_instance.parse();
    defer arena.allocator().free(nodes);

    try testing.expect(nodes.len == 1);
    try testing.expect(nodes[0] == .StructDecl);
    if (nodes[0] == .StructDecl) {
        try testing.expect(nodes[0].StructDecl.fields.len == 1);
    }
}

// ============================================================================
// Enum Declaration Tests
// ============================================================================

test "declarations: enum declaration" {
    const allocator = testing.allocator;
    const source = "enum Status { Active, Inactive }";

    var lex = lexer.Lexer.init(allocator, source);
    defer lex.deinit();
    const tokens = try lex.scanTokens();
    defer allocator.free(tokens);

    var arena = ast_arena.AstArena.init(allocator);
    defer arena.deinit();
    var parser_instance = parser.Parser.init(tokens, &arena);

    const nodes = try parser_instance.parse();
    defer arena.allocator().free(nodes);

    try testing.expect(nodes.len == 1);
    try testing.expect(nodes[0] == .EnumDecl);
    if (nodes[0] == .EnumDecl) {
        try testing.expect(std.mem.eql(u8, "Status", nodes[0].EnumDecl.name));
        try testing.expect(nodes[0].EnumDecl.variants.len == 2);
    }
}

test "declarations: enum with underlying type" {
    const allocator = testing.allocator;
    const source = "enum TokenStandard : u16 { ERC20 = 20, ERC721 = 721 }";

    var lex = lexer.Lexer.init(allocator, source);
    defer lex.deinit();
    const tokens = try lex.scanTokens();
    defer allocator.free(tokens);

    var arena = ast_arena.AstArena.init(allocator);
    defer arena.deinit();
    var parser_instance = parser.Parser.init(tokens, &arena);

    const nodes = try parser_instance.parse();
    defer arena.allocator().free(nodes);

    try testing.expect(nodes.len == 1);
    try testing.expect(nodes[0] == .EnumDecl);
    if (nodes[0] == .EnumDecl) {
        try testing.expect(nodes[0].EnumDecl.underlying_type_info != null);
        try testing.expect(nodes[0].EnumDecl.variants.len == 2);
    }
}

// ============================================================================
// Import Declaration Tests
// ============================================================================

test "declarations: import statement" {
    const allocator = testing.allocator;
    const source = "@import(\"std\");";

    var lex = lexer.Lexer.init(allocator, source);
    defer lex.deinit();
    const tokens = try lex.scanTokens();
    defer allocator.free(tokens);

    var arena = ast_arena.AstArena.init(allocator);
    defer arena.deinit();
    var parser_instance = parser.Parser.init(tokens, &arena);

    const nodes = try parser_instance.parse();
    defer arena.allocator().free(nodes);

    try testing.expect(nodes.len == 1);
    try testing.expect(nodes[0] == .Import);
    if (nodes[0] == .Import) {
        try testing.expect(std.mem.eql(u8, "std", nodes[0].Import.path));
    }
}

test "declarations: const import with alias" {
    const allocator = testing.allocator;
    const source = "const std = @import(\"std\");";

    var lex = lexer.Lexer.init(allocator, source);
    defer lex.deinit();
    const tokens = try lex.scanTokens();
    defer allocator.free(tokens);

    var arena = ast_arena.AstArena.init(allocator);
    defer arena.deinit();
    var parser_instance = parser.Parser.init(tokens, &arena);

    const nodes = try parser_instance.parse();
    defer arena.allocator().free(nodes);

    try testing.expect(nodes.len == 1);
    try testing.expect(nodes[0] == .Import);
    if (nodes[0] == .Import) {
        try testing.expect(nodes[0].Import.alias != null);
        if (nodes[0].Import.alias) |alias| {
            try testing.expect(std.mem.eql(u8, "std", alias));
        }
    }
}
