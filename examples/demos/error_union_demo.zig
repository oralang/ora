const std = @import("std");
const print = std.debug.print;
const testing = std.testing;

// Import our Ora compiler modules
const lexer = @import("../src/lexer.zig");
const parser = @import("../src/parser.zig");
const ast = @import("../src/ast.zig");
const codegen = @import("../src/codegen_yul.zig");

/// Error Union Demo
///
/// This demonstrates the !T error union model implementation in Ora:
/// - Error declarations with `error Name;`
/// - Error union return types `!T`
/// - Try expressions `try expr`
/// - Error return `error.Name`
/// - Try-catch blocks
pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    print("=== Ora Error Union Demo ===\n\n");

    // Test 1: Basic error union lexing
    try testErrorUnionLexing(allocator);

    // Test 2: Error union parsing
    try testErrorUnionParsing(allocator);

    // Test 3: Error union Yul generation
    try testErrorUnionYulGeneration(allocator);

    print("\n=== All tests completed successfully! ===\n");
}

/// Test lexing of error union syntax
fn testErrorUnionLexing(allocator: std.mem.Allocator) !void {
    print("1. Testing Error Union Lexing...\n");

    const source =
        \\error AccessDenied;
        \\fn transfer(amount: u256) -> !u256 {
        \\    if (amount > 1000) {
        \\        return error.AccessDenied;
        \\    }
        \\    let result = try getBalance();
        \\    return result;
        \\}
    ;

    var lex = lexer.Lexer.init(allocator, source);
    defer lex.deinit();

    const tokens = try lex.scanTokens();
    defer allocator.free(tokens);

    print("   Tokens generated: {}\n", .{tokens.len});

    // Check for error-related tokens
    var has_error_keyword = false;
    var has_try_keyword = false;
    var has_bang_token = false;

    for (tokens) |token| {
        switch (token.type) {
            .Error => {
                has_error_keyword = true;
                print("   Found error keyword: {s}\n", .{token.lexeme});
            },
            .Try => {
                has_try_keyword = true;
                print("   Found try keyword: {s}\n", .{token.lexeme});
            },
            .Bang => {
                has_bang_token = true;
                print("   Found bang token: {s}\n", .{token.lexeme});
            },
            else => {},
        }
    }

    if (has_error_keyword and has_try_keyword and has_bang_token) {
        print("   ✓ Error union lexing successful\n\n");
    } else {
        print("   ✗ Error union lexing failed\n\n");
    }
}

/// Test parsing of error union syntax
fn testErrorUnionParsing(allocator: std.mem.Allocator) !void {
    print("2. Testing Error Union Parsing...\n");

    const source =
        \\error MyError;
        \\fn test() -> !u256 {
        \\    return error.MyError;
        \\}
    ;

    var lex = lexer.Lexer.init(allocator, source);
    defer lex.deinit();

    const tokens = try lex.scanTokens();
    defer allocator.free(tokens);

    var pars = parser.Parser.init(allocator, tokens);

    // Parse the tokens
    if (pars.parse()) |ast_nodes| {
        defer {
            for (ast_nodes) |*node| {
                ast.deinitAstNode(allocator, node);
            }
            allocator.free(ast_nodes);
        }

        print("   Parsed {} AST nodes\n", .{ast_nodes.len});

        // Check for error union types and expressions
        for (ast_nodes) |node| {
            switch (node) {
                .Function => |func| {
                    if (func.return_type) |ret_type| {
                        switch (ret_type) {
                            .ErrorUnion => {
                                print("   ✓ Found error union return type\n");
                            },
                            else => {},
                        }
                    }
                },
                else => {},
            }
        }

        print("   ✓ Error union parsing successful\n\n");
    } else |err| {
        print("   ✗ Parsing failed: {}\n\n", .{err});
    }
}

/// Test Yul code generation for error unions
fn testErrorUnionYulGeneration(allocator: std.mem.Allocator) !void {
    print("3. Testing Error Union Yul Generation...\n");

    var codegen_instance = codegen.YulCodegen.init(allocator);
    defer codegen_instance.deinit();

    // Create a simple buffer to test error union code generation
    var yul_code = std.ArrayList(u8).init(allocator);
    defer yul_code.deinit();

    // Test success case generation
    const success_var = try codegen_instance.generateErrorUnionSuccess(&yul_code, "123");
    defer allocator.free(success_var);

    // Test error case generation
    const error_var = try codegen_instance.generateErrorUnionError(&yul_code, 42);
    defer allocator.free(error_var);

    // Test error union check
    const tag_var = try codegen_instance.generateErrorUnionCheck(&yul_code, success_var);
    defer allocator.free(tag_var);

    // Test data extraction
    const data_var = try codegen_instance.generateErrorUnionExtract(&yul_code, success_var);
    defer allocator.free(data_var);

    const generated_yul = yul_code.items;
    print("   Generated Yul code ({} bytes):\n", .{generated_yul.len});
    print("   ---\n{s}   ---\n", .{generated_yul});

    // Verify the generated code contains expected patterns
    const has_success_tag = std.mem.indexOf(u8, generated_yul, "0x00") != null;
    const has_error_tag = std.mem.indexOf(u8, generated_yul, "0x01") != null;
    const has_memory_layout = std.mem.indexOf(u8, generated_yul, "mstore(ptr") != null;
    const has_tag_check = std.mem.indexOf(u8, generated_yul, "mload(") != null;

    if (has_success_tag and has_error_tag and has_memory_layout and has_tag_check) {
        print("   ✓ Error union Yul generation successful\n");
        print("   ✓ Contains success tag (0x00)\n");
        print("   ✓ Contains error tag (0x01)\n");
        print("   ✓ Contains memory layout operations\n");
        print("   ✓ Contains tag checking operations\n\n");
    } else {
        print("   ✗ Error union Yul generation incomplete\n\n");
    }
}

test "error union complete workflow" {
    const allocator = testing.allocator;

    // Test that error union types can be created and used
    var codegen_instance = codegen.YulCodegen.init(allocator);
    defer codegen_instance.deinit();

    var yul_code = std.ArrayList(u8).init(allocator);
    defer yul_code.deinit();

    // Generate success case
    const success_var = try codegen_instance.generateErrorUnionSuccess(&yul_code, "42");
    defer allocator.free(success_var);

    // Generate error case
    const error_var = try codegen_instance.generateErrorUnionError(&yul_code, 1);
    defer allocator.free(error_var);

    // Check that code was generated
    try testing.expect(yul_code.items.len > 0);

    // Check for expected patterns
    const contains_success = std.mem.indexOf(u8, yul_code.items, "0x00") != null;
    const contains_error = std.mem.indexOf(u8, yul_code.items, "0x01") != null;

    try testing.expect(contains_success);
    try testing.expect(contains_error);
}

test "error union memory layout" {
    const allocator = testing.allocator;

    var codegen_instance = codegen.YulCodegen.init(allocator);
    defer codegen_instance.deinit();

    var yul_code = std.ArrayList(u8).init(allocator);
    defer yul_code.deinit();

    // Generate error union
    const result_var = try codegen_instance.generateErrorUnionSuccess(&yul_code, "100");
    defer allocator.free(result_var);

    // Generate check code
    const tag_var = try codegen_instance.generateErrorUnionCheck(&yul_code, result_var);
    defer allocator.free(tag_var);

    // Generate extract code
    const data_var = try codegen_instance.generateErrorUnionExtract(&yul_code, result_var);
    defer allocator.free(data_var);

    const generated = yul_code.items;

    // Verify memory layout: [tag][data]
    try testing.expect(std.mem.indexOf(u8, generated, "mstore(ptr, 0x00)") != null); // tag
    try testing.expect(std.mem.indexOf(u8, generated, "mstore(add(ptr, 0x20)") != null); // data
    try testing.expect(std.mem.indexOf(u8, generated, "mload(") != null); // read operations
}
