// ============================================================================
// Type Parser Tests
// ============================================================================
// Tests for parsing type expressions
// ============================================================================

const std = @import("std");
const testing = std.testing;
const ora_root = @import("ora_root");
const lexer = ora_root.lexer;
const parser = ora_root.parser;
const ast = ora_root.ast;
const ast_arena = ora_root.ast_arena;
const TypeCategory = ast.Types.TypeCategory;

// Helper function to parse a type from source
fn parseTypeFromSource(allocator: std.mem.Allocator, source: []const u8) !ast.Types.TypeInfo {
    var lex = lexer.Lexer.init(allocator, source);
    defer lex.deinit();
    const tokens = try lex.scanTokens();
    defer allocator.free(tokens);

    var arena = ast_arena.AstArena.init(allocator);
    defer arena.deinit();
    const TypeParser = parser.parser_mod.TypeParser;
    var type_parser = TypeParser.init(tokens, &arena);

    return type_parser.parseType();
}

// ============================================================================
// Base Type Tests
// ============================================================================

test "types: unsigned integer types" {
    const allocator = testing.allocator;
    const test_cases = [_][]const u8{ "u8", "u16", "u32", "u64", "u128", "u256" };

    for (test_cases) |source| {
        const type_info = try parseTypeFromSource(allocator, source);
        try testing.expectEqual(TypeCategory.Integer, type_info.category);
        try testing.expect(type_info.ora_type != null);
    }
}

test "types: signed integer types" {
    const allocator = testing.allocator;
    const test_cases = [_][]const u8{ "i8", "i16", "i32", "i64", "i128", "i256" };

    for (test_cases) |source| {
        const type_info = try parseTypeFromSource(allocator, source);
        try testing.expectEqual(TypeCategory.Integer, type_info.category);
        try testing.expect(type_info.ora_type != null);
    }
}

test "types: bool type" {
    const allocator = testing.allocator;
    const type_info = try parseTypeFromSource(allocator, "bool");

    try testing.expectEqual(TypeCategory.Bool, type_info.category);
    try testing.expect(type_info.ora_type != null);
}

test "types: address type" {
    const allocator = testing.allocator;
    const type_info = try parseTypeFromSource(allocator, "address");

    try testing.expectEqual(TypeCategory.Address, type_info.category);
    try testing.expect(type_info.ora_type != null);
}

test "types: string type" {
    const allocator = testing.allocator;
    const type_info = try parseTypeFromSource(allocator, "string");

    try testing.expectEqual(TypeCategory.String, type_info.category);
    try testing.expect(type_info.ora_type != null);
}

// ============================================================================
// Array Type Tests
// ============================================================================

test "types: fixed array type" {
    const allocator = testing.allocator;
    const type_info = try parseTypeFromSource(allocator, "[u256; 5]");

    try testing.expectEqual(TypeCategory.Array, type_info.category);
    try testing.expect(type_info.ora_type != null);
}

test "types: array with different element types" {
    const allocator = testing.allocator;
    const test_cases = [_][]const u8{
        "[u8; 10]",
        "[bool; 2]",
        "[address; 3]",
    };

    for (test_cases) |source| {
        const type_info = try parseTypeFromSource(allocator, source);
        try testing.expectEqual(TypeCategory.Array, type_info.category);
    }
}

// ============================================================================
// Slice Type Tests
// ============================================================================

test "types: slice type" {
    const allocator = testing.allocator;
    const type_info = try parseTypeFromSource(allocator, "slice[u256]");

    try testing.expectEqual(TypeCategory.Slice, type_info.category);
    try testing.expect(type_info.ora_type != null);
}

test "types: slice with different element types" {
    const allocator = testing.allocator;
    const test_cases = [_][]const u8{
        "slice[u8]",
        "slice[address]",
        "slice[bool]",
    };

    for (test_cases) |source| {
        const type_info = try parseTypeFromSource(allocator, source);
        try testing.expectEqual(TypeCategory.Slice, type_info.category);
    }
}

// ============================================================================
// Map Type Tests
// ============================================================================

test "types: map type angle brackets" {
    const allocator = testing.allocator;
    const type_info = try parseTypeFromSource(allocator, "map<address, u256>");

    try testing.expectEqual(TypeCategory.Map, type_info.category);
    try testing.expect(type_info.ora_type != null);
}

test "types: nested map type angle brackets" {
    const allocator = testing.allocator;
    const type_info = try parseTypeFromSource(allocator, "map<address, map<address, u256>>");

    try testing.expectEqual(TypeCategory.Map, type_info.category);
}

// ============================================================================
// Refinement Type Tests
// ============================================================================

test "types: MinValue refinement type" {
    const allocator = testing.allocator;
    const type_info = try parseTypeFromSource(allocator, "MinValue<u256, 100>");

    try testing.expectEqual(TypeCategory.Integer, type_info.category);
    try testing.expect(type_info.ora_type != null);
}

test "types: MaxValue refinement type" {
    const allocator = testing.allocator;
    const type_info = try parseTypeFromSource(allocator, "MaxValue<u256, 1000>");

    try testing.expectEqual(TypeCategory.Integer, type_info.category);
    try testing.expect(type_info.ora_type != null);
}

test "types: InRange refinement type" {
    const allocator = testing.allocator;
    const type_info = try parseTypeFromSource(allocator, "InRange<u256, 0, 100>");

    try testing.expectEqual(TypeCategory.Integer, type_info.category);
    try testing.expect(type_info.ora_type != null);
}

test "types: NonZeroAddress refinement type" {
    const allocator = testing.allocator;
    const type_info = try parseTypeFromSource(allocator, "NonZeroAddress");

    try testing.expectEqual(TypeCategory.Address, type_info.category);
    try testing.expect(type_info.ora_type != null);
}

// ============================================================================
// Error Union Type Tests
// ============================================================================

test "types: error union type" {
    const allocator = testing.allocator;
    const type_info = try parseTypeFromSource(allocator, "!u256");

    try testing.expectEqual(TypeCategory.ErrorUnion, type_info.category);
    try testing.expect(type_info.ora_type != null);
}

test "types: error union with multiple error types" {
    const allocator = testing.allocator;
    const type_info = try parseTypeFromSource(allocator, "!u256 | Error1 | Error2");

    try testing.expectEqual(TypeCategory.ErrorUnion, type_info.category);
}

// ============================================================================
// Return Type Tests
// ============================================================================

test "types: void return type" {
    const allocator = testing.allocator;
    var lex = lexer.Lexer.init(allocator, "void");
    defer lex.deinit();
    const tokens = try lex.scanTokens();
    defer allocator.free(tokens);

    var arena = ast_arena.AstArena.init(allocator);
    defer arena.deinit();
    const TypeParser = parser.parser_mod.TypeParser;
    var type_parser = TypeParser.init(tokens, &arena);

    const type_info = try type_parser.parseReturnType();

    try testing.expectEqual(TypeCategory.Void, type_info.category);
}

// ============================================================================
// Custom Type Tests
// ============================================================================

test "types: custom struct type" {
    const allocator = testing.allocator;
    const type_info = try parseTypeFromSource(allocator, "Point");

    // custom types are assumed to be structs initially
    try testing.expectEqual(TypeCategory.Struct, type_info.category);
}

test "types: custom enum type" {
    const allocator = testing.allocator;
    const type_info = try parseTypeFromSource(allocator, "Status");

    // custom types are assumed to be structs initially (will be resolved later)
    try testing.expectEqual(TypeCategory.Struct, type_info.category);
}
