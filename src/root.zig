//! Ora DSL Compiler Library
//!
//! This is the main library module for the Ora language compiler, providing
//! lexical analysis, parsing, semantic analysis, HIR generation, and Yul code generation.
//!
//! The Ora language is a domain-specific language for smart contract development
//! with formal verification capabilities and memory safety guarantees.

const std = @import("std");
const testing = std.testing;

// Export all public modules
pub const lexer = @import("lexer.zig");
pub const parser = @import("parser.zig");
pub const ast = @import("ast.zig");
pub const typer = @import("typer.zig");
pub const semantics = @import("semantics.zig");
pub const comptime_eval = @import("comptime_eval.zig");
pub const static_verifier = @import("static_verifier.zig");
pub const formal_verifier = @import("formal_verifier.zig");
pub const optimizer = @import("optimizer.zig");

// Re-export key types for convenience
/// Lexical analyzer for Ora source code
pub const Lexer = lexer.Lexer;
/// Parser for generating AST from tokens
pub const Parser = parser.Parser;
/// Token representation
pub const Token = lexer.Token;
/// Token type enumeration
pub const TokenType = lexer.TokenType;

// AST types and functions
/// Abstract Syntax Tree node
pub const AstNode = ast.AstNode;
/// Contract declaration node
pub const ContractNode = ast.ContractNode;
/// Function declaration node
pub const FunctionNode = ast.FunctionNode;
/// Variable declaration node
pub const VariableDeclNode = ast.VariableDeclNode;
/// Expression node
pub const ExprNode = ast.ExprNode;
/// Type reference
pub const TypeRef = ast.TypeRef;
/// Memory region specification
pub const MemoryRegion = ast.MemoryRegion;
/// Cleanup function for AST nodes
pub const deinitAstNodes = ast.deinitAstNodes;

// Analysis types
/// Type checker for Ora language
pub const Typer = typer.Typer;
/// Type representation
pub const OraType = typer.OraType;
/// Semantic analyzer
pub const SemanticAnalyzer = semantics.SemanticAnalyzer;
/// Diagnostic message
pub const Diagnostic = semantics.Diagnostic;

// Comptime evaluation
/// Compile-time evaluator
pub const ComptimeEvaluator = comptime_eval.ComptimeEvaluator;
/// Compile-time value representation
pub const ComptimeValue = comptime_eval.ComptimeValue;
/// Compile-time evaluation errors
pub const ComptimeError = comptime_eval.ComptimeError;

// Static verification
/// Static verifier for requires/ensures clauses
pub const StaticVerifier = static_verifier.StaticVerifier;
/// Verification condition
pub const VerificationCondition = static_verifier.VerificationCondition;
/// Verification result
pub const VerificationResult = static_verifier.VerificationResult;
/// Verification errors
pub const VerificationError = static_verifier.VerificationError;

// Optimization
/// Optimizer for eliminating redundant runtime checks
pub const Optimizer = optimizer.Optimizer;
/// Optimization result
pub const OptimizationResult = optimizer.OptimizationResult;
/// Optimization context
pub const OptimizationContext = optimizer.OptimizationContext;
/// Optimization type
pub const OptimizationType = optimizer.OptimizationType;
/// Optimization errors
pub const OptimizationError = optimizer.OptimizationError;

// IR system
/// High-level Intermediate Representation module
pub const ir = @import("ir.zig");
/// HIR builder for AST conversion
pub const IRBuilder = ir.IRBuilder;
/// HIR validator
pub const Validator = ir.Validator;
/// HIR program representation
pub const HIRProgram = ir.HIRProgram;
/// Validation result
pub const ValidationResult = ir.ValidationResult;
/// AST to HIR converter
pub const ASTToHIRConverter = ir.ASTToHIRConverter;
/// JSON serialization utilities
pub const JSONSerializer = ir.JSONSerializer;

// Yul integration
/// Yul compiler bindings
pub const yul_bindings = @import("yul_bindings.zig");
/// Yul code generation
pub const codegen_yul = @import("codegen_yul.zig");
/// Yul compiler interface
pub const YulCompiler = yul_bindings.YulCompiler;
/// Yul code generator
pub const YulCodegen = codegen_yul.YulCodegen;

// Test imports
const Lexer_test = Lexer;
const Token_test = Token;
const TokenType_test = TokenType;

test "empty source" {
    var lex = Lexer_test.init(testing.allocator, "");
    defer lex.deinit();

    const tokens = try lex.scanTokens();
    defer testing.allocator.free(tokens);

    try testing.expect(tokens.len == 1);
    try testing.expect(tokens[0].type == .Eof);
}

test "keywords" {
    const source = "contract pub fn let var const immutable storage memory tstore init log";
    var lex = Lexer_test.init(testing.allocator, source);
    defer lex.deinit();

    const tokens = try lex.scanTokens();
    defer testing.allocator.free(tokens);

    const expected = [_]TokenType_test{ .Contract, .Pub, .Fn, .Let, .Var, .Const, .Immutable, .Storage, .Memory, .Tstore, .Init, .Log, .Eof };

    try testing.expect(tokens.len == expected.len);
    for (tokens, expected) |token, expected_type| {
        try testing.expect(token.type == expected_type);
    }
}

test "identifiers and literals" {
    const source = "transfer balanceOf \"MyToken\" 42 0x1a2b3c";
    var lex = Lexer_test.init(testing.allocator, source);
    defer lex.deinit();

    const tokens = try lex.scanTokens();
    defer testing.allocator.free(tokens);

    const expected = [_]TokenType_test{ .Identifier, .Identifier, .StringLiteral, .IntegerLiteral, .HexLiteral, .Eof };

    try testing.expect(tokens.len == expected.len);
    for (tokens, expected) |token, expected_type| {
        try testing.expect(token.type == expected_type);
    }

    try testing.expectEqualStrings("transfer", tokens[0].lexeme);
    try testing.expectEqualStrings("balanceOf", tokens[1].lexeme);
    try testing.expectEqualStrings("MyToken", tokens[2].lexeme); // Quotes stripped
    try testing.expectEqualStrings("42", tokens[3].lexeme);
    try testing.expectEqualStrings("0x1a2b3c", tokens[4].lexeme);
}

test "operators and delimiters" {
    const source = "+ - * / == != <= >= -> ( ) { } [ ]";
    var lex = Lexer_test.init(testing.allocator, source);
    defer lex.deinit();

    const tokens = try lex.scanTokens();
    defer testing.allocator.free(tokens);

    const expected = [_]TokenType_test{ .Plus, .Minus, .Star, .Slash, .EqualEqual, .BangEqual, .LessEqual, .GreaterEqual, .Arrow, .LeftParen, .RightParen, .LeftBrace, .RightBrace, .LeftBracket, .RightBracket, .Eof };

    try testing.expect(tokens.len == expected.len);
    for (tokens, expected) |token, expected_type| {
        try testing.expect(token.type == expected_type);
    }
}

test "address literals" {
    const source = "0x742d35Cc6e6B002B96DCE91a5aa0B5E7b6E0e312";
    var lex = Lexer_test.init(testing.allocator, source);
    defer lex.deinit();

    const tokens = try lex.scanTokens();
    defer testing.allocator.free(tokens);

    try testing.expect(tokens.len == 2); // address + EOF
    try testing.expect(tokens[0].type == .AddressLiteral);
    try testing.expectEqualStrings("0x742d35Cc6e6B002B96DCE91a5aa0B5E7b6E0e312", tokens[0].lexeme);
}

test "comments" {
    const source =
        \\// Single line comment
        \\let x = /* inline comment */ 42;
        \\/* Multi-line
        \\   comment */
    ;
    var lex = Lexer_test.init(testing.allocator, source);
    defer lex.deinit();

    const tokens = try lex.scanTokens();
    defer testing.allocator.free(tokens);

    // Should only have: let, x, =, 42, ;, EOF
    const expected = [_]TokenType_test{ .Let, .Identifier, .Equal, .IntegerLiteral, .Semicolon, .Eof };

    try testing.expect(tokens.len == expected.len);
    for (tokens, expected) |token, expected_type| {
        try testing.expect(token.type == expected_type);
    }
}

test "real zigora code" {
    const source =
        \\contract ERC20 {
        \\    pub fn transfer(to: address, amount: u256) -> bool {
        \\        requires(balances[std.transaction.sender] >= amount);
        \\        balances[std.transaction.sender] -= amount;
        \\        return true;
        \\    }
        \\}
    ;
    var lex = Lexer_test.init(testing.allocator, source);
    defer lex.deinit();

    const tokens = try lex.scanTokens();
    defer testing.allocator.free(tokens);

    // Verify we have reasonable number of tokens
    try testing.expect(tokens.len > 30);

    // Verify first few tokens are correct
    const expected_start = [_]TokenType_test{ .Contract, .Identifier, .LeftBrace };
    for (tokens[0..3], expected_start) |token, expected_type| {
        try testing.expect(token.type == expected_type);
    }

    // Count some specific tokens
    var pub_count: u32 = 0;
    var fn_count: u32 = 0;
    var arrow_count: u32 = 0;

    for (tokens) |token| {
        switch (token.type) {
            .Pub => pub_count += 1,
            .Fn => fn_count += 1,
            .Arrow => arrow_count += 1,
            else => {},
        }
    }

    try testing.expect(pub_count == 1); // One public function
    try testing.expect(fn_count == 1); // One function total
    try testing.expect(arrow_count == 1); // One return type arrow
}

test "error handling" {
    // Test unterminated string
    {
        const source = "\"unterminated string";
        var lex = Lexer_test.init(testing.allocator, source);
        defer lex.deinit();

        const result = lex.scanTokens();
        try testing.expectError(lexer.LexerError.UnterminatedString, result);
    }

    // Test invalid hex literal
    {
        const source = "0x";
        var lex = Lexer_test.init(testing.allocator, source);
        defer lex.deinit();

        const result = lex.scanTokens();
        try testing.expectError(lexer.LexerError.InvalidHexLiteral, result);
    }

    // Test unterminated comment
    {
        const source = "/* unterminated comment";
        var lex = Lexer_test.init(testing.allocator, source);
        defer lex.deinit();

        const result = lex.scanTokens();
        try testing.expectError(lexer.LexerError.UnterminatedComment, result);
    }

    // Test unexpected character
    {
        const source = "let x = $ invalid";
        var lex = Lexer_test.init(testing.allocator, source);
        defer lex.deinit();

        const result = lex.scanTokens();
        try testing.expectError(lexer.LexerError.UnexpectedCharacter, result);
    }
}

test "convenience scan function" {
    const tokens = try lexer.scan("let x = 42;", testing.allocator);
    defer testing.allocator.free(tokens);

    try testing.expect(tokens.len == 6); // let, x, =, 42, ;, EOF
    try testing.expect(tokens[0].type == .Let);
    try testing.expect(tokens[3].type == .IntegerLiteral);
}

test "token utility functions" {
    try testing.expect(lexer.isKeyword(.Contract));
    try testing.expect(lexer.isKeyword(.Pub));
    try testing.expect(!lexer.isKeyword(.Identifier));

    try testing.expect(lexer.isLiteral(.StringLiteral));
    try testing.expect(lexer.isLiteral(.IntegerLiteral));
    try testing.expect(!lexer.isLiteral(.Plus));

    try testing.expect(lexer.isOperator(.Plus));
    try testing.expect(lexer.isOperator(.Arrow));
    try testing.expect(!lexer.isOperator(.Identifier));

    try testing.expect(lexer.isDelimiter(.LeftParen));
    try testing.expect(lexer.isDelimiter(.Comma));
    try testing.expect(!lexer.isDelimiter(.Plus));
}

test "string literal quote stripping" {
    const tokens = try lexer.scan("\"hello world\"", testing.allocator);
    defer testing.allocator.free(tokens);

    try testing.expect(tokens.len == 2); // string + EOF
    try testing.expect(tokens[0].type == .StringLiteral);
    try testing.expectEqualStrings("hello world", tokens[0].lexeme);
}

test "comprehensive token patterns" {
    const test_cases = [_]struct {
        input: []const u8,
        expected: []const TokenType_test,
    }{
        .{ .input = "let x = 5;", .expected = &[_]TokenType_test{ .Let, .Identifier, .Equal, .IntegerLiteral, .Semicolon, .Eof } },
        .{ .input = "\"string\"", .expected = &[_]TokenType_test{ .StringLiteral, .Eof } },
        .{ .input = "0x123abc", .expected = &[_]TokenType_test{ .HexLiteral, .Eof } },
        .{ .input = "0x123abc123abc123abc123abc123abc123abc123a", .expected = &[_]TokenType_test{ .AddressLiteral, .Eof } },
        .{ .input = "!= == <= >= ->", .expected = &[_]TokenType_test{ .BangEqual, .EqualEqual, .LessEqual, .GreaterEqual, .Arrow, .Eof } },
    };

    for (test_cases) |test_case| {
        const tokens = try lexer.scan(test_case.input, testing.allocator);
        defer testing.allocator.free(tokens);

        try testing.expect(tokens.len == test_case.expected.len);
        for (tokens, test_case.expected) |token, expected_type| {
            try testing.expect(token.type == expected_type);
        }
    }
}

test "nested comments" {
    const source = "/* outer /* inner */ outer */ let x = 42;";
    const tokens = try lexer.scan(source, testing.allocator);
    defer testing.allocator.free(tokens);

    // Should skip all comments and only have: let, x, =, 42, ;, EOF
    const expected = [_]TokenType_test{ .Let, .Identifier, .Equal, .IntegerLiteral, .Semicolon, .Eof };

    try testing.expect(tokens.len == expected.len);
    for (tokens, expected) |token, expected_type| {
        try testing.expect(token.type == expected_type);
    }
}

test "column tracking accuracy" {
    const source = "let x == 42;";
    const tokens = try lexer.scan(source, testing.allocator);
    defer testing.allocator.free(tokens);

    // Verify column positions are accurate
    try testing.expect(tokens[0].column == 1); // "let" starts at column 1
    try testing.expect(tokens[1].column == 5); // "x" starts at column 5
    try testing.expect(tokens[2].column == 7); // "==" starts at column 7
    try testing.expect(tokens[3].column == 10); // "42" starts at column 10
}

test "string content verification" {
    const test_cases = [_]struct {
        input: []const u8,
        expected_content: []const u8,
    }{
        .{ .input = "\"MyToken\"", .expected_content = "MyToken" },
        .{ .input = "\"MTK\"", .expected_content = "MTK" },
        .{ .input = "\"\"", .expected_content = "" },
        .{ .input = "\"Hello, World!\"", .expected_content = "Hello, World!" },
    };

    for (test_cases) |test_case| {
        const tokens = try lexer.scan(test_case.input, testing.allocator);
        defer testing.allocator.free(tokens);

        try testing.expect(tokens.len == 2); // string + EOF
        try testing.expect(tokens[0].type == .StringLiteral);
        try testing.expectEqualStrings(test_case.expected_content, tokens[0].lexeme);
    }
}

test "error detail reporting" {
    var lex = Lexer_test.init(testing.allocator, "let x = $ invalid");
    defer lex.deinit();

    const result = lex.scanTokens();
    try testing.expectError(lexer.LexerError.UnexpectedCharacter, result);

    // Check that error details are available
    try testing.expect(lex.last_bad_char == '$');

    const error_msg = try lex.getErrorDetails(testing.allocator);
    defer testing.allocator.free(error_msg);

    // Should contain the problematic character
    try testing.expect(std.mem.indexOf(u8, error_msg, "$") != null);
}

test "transient storage support" {
    const source = "tstore var counter: u256; tstore temp_hash: bytes32;";
    const tokens = try lexer.scan(source, testing.allocator);
    defer testing.allocator.free(tokens);

    // Should tokenize: tstore, var, counter, :, u256, ;, tstore, temp_hash, :, bytes32, ;, EOF
    const expected = [_]TokenType_test{ .Tstore, .Var, .Identifier, .Colon, .Identifier, .Semicolon, .Tstore, .Identifier, .Colon, .Identifier, .Semicolon, .Eof };

    try testing.expect(tokens.len == expected.len);
    for (tokens, expected) |token, expected_type| {
        try testing.expect(token.type == expected_type);
    }

    // Verify specific lexemes
    try testing.expectEqualStrings("tstore", tokens[0].lexeme);
    try testing.expectEqualStrings("var", tokens[1].lexeme);
    try testing.expectEqualStrings("counter", tokens[2].lexeme);
    try testing.expectEqualStrings("tstore", tokens[6].lexeme);
    try testing.expectEqualStrings("temp_hash", tokens[7].lexeme);
}

test "ast basic types" {
    _ = ast.SourceSpan{ .line = 1, .column = 1, .length = 10 };

    // Test TypeRef variants
    const type_u256 = ast.TypeRef.U256;
    const type_bool = ast.TypeRef.Bool;
    const type_address = ast.TypeRef.Address;

    try testing.expect(type_u256 == .U256);
    try testing.expect(type_bool == .Bool);
    try testing.expect(type_address == .Address);

    // Test MemoryRegion variants
    const region_stack = ast.MemoryRegion.Stack;
    const region_storage = ast.MemoryRegion.Storage;
    const region_tstore = ast.MemoryRegion.TStore;

    try testing.expect(region_stack == .Stack);
    try testing.expect(region_storage == .Storage);
    try testing.expect(region_tstore == .TStore);
}

test "ast node construction" {
    const span = ast.SourceSpan{ .line = 1, .column = 1, .length = 5 };

    // Test creating a variable declaration
    const var_decl = ast.VariableDeclNode{
        .name = "balance",
        .region = .Storage,
        .mutable = true,
        .locked = false,
        .typ = .U256,
        .value = null,
        .span = span,
    };

    try testing.expectEqualStrings("balance", var_decl.name);
    try testing.expect(var_decl.region == .Storage);
    try testing.expect(var_decl.mutable == true);
    try testing.expect(var_decl.typ == .U256);

    // Test creating a contract node
    const contract = ast.ContractNode{
        .name = "Token",
        .body = &[_]ast.AstNode{},
        .span = span,
    };

    try testing.expectEqualStrings("Token", contract.name);
    try testing.expect(contract.body.len == 0);
}

test "ast helper functions" {
    const span = ast.SourceSpan{ .line = 1, .column = 1, .length = 5 };

    // Test identifier creation
    const ident = try ast.createIdentifier(testing.allocator, "myVar", span);
    defer testing.allocator.destroy(ident);

    switch (ident.*) {
        .Identifier => |id| {
            try testing.expectEqualStrings("myVar", id.name);
            try testing.expect(id.span.line == 1);
        },
        else => try testing.expect(false),
    }
}

test "ast complex types" {
    const span = ast.SourceSpan{ .line = 1, .column = 1, .length = 10 };

    // Test mapping type
    const key_type = try testing.allocator.create(ast.TypeRef);
    defer testing.allocator.destroy(key_type);
    key_type.* = .Address;

    const value_type = try testing.allocator.create(ast.TypeRef);
    defer testing.allocator.destroy(value_type);
    value_type.* = .U256;

    const mapping = ast.MappingType{
        .key = key_type,
        .value = value_type,
    };

    try testing.expect(mapping.key.* == .Address);
    try testing.expect(mapping.value.* == .U256);

    // Test function node with requires/ensures
    const func = ast.FunctionNode{
        .pub_ = true,
        .name = "transfer",
        .parameters = &[_]ast.ParamNode{},
        .return_type = ast.TypeRef.Bool,
        .requires_clauses = &[_]ast.ExprNode{},
        .ensures_clauses = &[_]ast.ExprNode{},
        .body = ast.BlockNode{
            .statements = &[_]ast.StmtNode{},
            .span = span,
        },
        .span = span,
    };

    try testing.expect(func.pub_ == true);
    try testing.expectEqualStrings("transfer", func.name);
    try testing.expect(func.return_type.? == .Bool);
}
