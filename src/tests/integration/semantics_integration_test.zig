const std = @import("std");
const testing = std.testing;
const ast = @import("../src/ast.zig");
const semantics = @import("../src/semantics.zig");
const lexer = @import("../src/lexer.zig");
const parser = @import("../src/parser.zig");

/// Integration test helper for complete workflows
const IntegrationTestHelper = struct {
    allocator: std.mem.Allocator,

    fn init(allocator: std.mem.Allocator) IntegrationTestHelper {
        return IntegrationTestHelper{ .allocator = allocator };
    }

    /// Compile source code through lexing, parsing, and semantic analysis
    fn compileAndAnalyze(self: *IntegrationTestHelper, source: []const u8) !struct {
        tokens: []lexer.Token,
        ast_nodes: []ast.AstNode,
        diagnostics: []semantics.Diagnostic,
    } {
        // Lex the source
        const tokens = try lexer.scan(source, self.allocator);
        errdefer self.allocator.free(tokens);

        // Parse tokens to AST
        var p = parser.Parser.init(tokens, self.allocator);
        defer p.deinit();
        const ast_nodes = try p.parseProgram();
        errdefer self.allocator.free(ast_nodes);

        // Perform semantic analysis
        const diagnostics = try semantics.analyze(self.allocator, ast_nodes);

        return .{
            .tokens = tokens,
            .ast_nodes = ast_nodes,
            .diagnostics = diagnostics,
        };
    }

    fn expectNoErrors(diagnostics: []semantics.Diagnostic) !void {
        var error_count: u32 = 0;
        for (diagnostics) |diagnostic| {
            if (diagnostic.severity == .Error) {
                error_count += 1;
                std.debug.print("ERROR: {s}\n", .{diagnostic.message});
            }
        }
        try testing.expect(error_count == 0);
    }

    fn expectErrorCount(diagnostics: []semantics.Diagnostic, expected: u32) !void {
        var error_count: u32 = 0;
        for (diagnostics) |diagnostic| {
            if (diagnostic.severity == .Error) {
                error_count += 1;
            }
        }
        try testing.expectEqual(expected, error_count);
    }

    fn expectWarningCount(diagnostics: []semantics.Diagnostic, expected: u32) !void {
        var warning_count: u32 = 0;
        for (diagnostics) |diagnostic| {
            if (diagnostic.severity == .Warning) {
                warning_count += 1;
            }
        }
        try testing.expectEqual(expected, warning_count);
    }

    fn cleanup(self: *IntegrationTestHelper, result: anytype) void {
        self.allocator.free(result.tokens);
        self.allocator.free(result.ast_nodes);

        // Free diagnostic messages
        for (result.diagnostics) |diagnostic| {
            self.allocator.free(diagnostic.message);
        }
        self.allocator.free(result.diagnostics);
    }
};

// =============================================================================
// SIMPLE CONTRACT INTEGRATION TESTS
// =============================================================================

test "simple valid contract compilation" {
    var helper = IntegrationTestHelper.init(testing.allocator);

    const source =
        \\contract SimpleStorage {
        \\    storage value: u32;
        \\    
        \\    function init() {
        \\        value = 0;
        \\    }
        \\    
        \\    function setValue(newValue: u32) public {
        \\        value = newValue;
        \\    }
        \\}
    ;

    const result = try helper.compileAndAnalyze(source);
    defer helper.cleanup(result);

    // Should compile without errors
    try helper.expectNoErrors(result.diagnostics);

    // Verify we have contract, storage variable, and functions
    try testing.expect(result.ast_nodes.len >= 1);
}

test "contract missing init function" {
    var helper = IntegrationTestHelper.init(testing.allocator);

    const source =
        \\contract InvalidContract {
        \\    storage value: u32;
        \\    
        \\    function setValue(newValue: u32) public {
        \\        value = newValue;
        \\    }
        \\}
    ;

    const result = try helper.compileAndAnalyze(source);
    defer helper.cleanup(result);

    // Should have error for missing init function
    try helper.expectErrorCount(result.diagnostics, 1);
}

test "contract with invalid storage access" {
    var helper = IntegrationTestHelper.init(testing.allocator);

    const source =
        \\contract InvalidStorage {
        \\    function init() {
        \\        storage invalidVar: u32 = 42; // Storage var in function
        \\    }
        \\}
    ;

    const result = try helper.compileAndAnalyze(source);
    defer helper.cleanup(result);

    // Should have error for storage variable in function
    try helper.expectErrorCount(result.diagnostics, 1);
}

// =============================================================================
// IMMUTABLE VARIABLE INTEGRATION TESTS
// =============================================================================

test "immutable variable proper initialization" {
    var helper = IntegrationTestHelper.init(testing.allocator);

    const source =
        \\contract ImmutableTest {
        \\    immutable owner: address;
        \\    
        \\    function init() {
        \\        owner = 0x1234567890123456789012345678901234567890;
        \\    }
        \\}
    ;

    const result = try helper.compileAndAnalyze(source);
    defer helper.cleanup(result);

    // Should compile without errors
    try helper.expectNoErrors(result.diagnostics);
}

test "immutable variable not initialized" {
    var helper = IntegrationTestHelper.init(testing.allocator);

    const source =
        \\contract ImmutableTest {
        \\    immutable owner: address;
        \\    
        \\    function init() {
        \\        // owner not initialized
        \\    }
        \\}
    ;

    const result = try helper.compileAndAnalyze(source);
    defer helper.cleanup(result);

    // Should have error for uninitialized immutable variable
    try helper.expectErrorCount(result.diagnostics, 1);
}

test "immutable variable double initialization" {
    var helper = IntegrationTestHelper.init(testing.allocator);

    const source =
        \\contract ImmutableTest {
        \\    immutable owner: address;
        \\    
        \\    function init() {
        \\        owner = 0x1111111111111111111111111111111111111111;
        \\        owner = 0x2222222222222222222222222222222222222222; // Double init
        \\    }
        \\}
    ;

    const result = try helper.compileAndAnalyze(source);
    defer helper.cleanup(result);

    // Should have error for double initialization
    try helper.expectErrorCount(result.diagnostics, 1);
}

test "immutable variable assignment outside constructor" {
    var helper = IntegrationTestHelper.init(testing.allocator);

    const source =
        \\contract ImmutableTest {
        \\    immutable owner: address;
        \\    
        \\    function init() {
        \\        owner = 0x1111111111111111111111111111111111111111;
        \\    }
        \\    
        \\    function changeOwner() public {
        \\        owner = 0x2222222222222222222222222222222222222222; // Invalid
        \\    }
        \\}
    ;

    const result = try helper.compileAndAnalyze(source);
    defer helper.cleanup(result);

    // Should have error for assignment outside constructor
    try helper.expectErrorCount(result.diagnostics, 1);
}

// =============================================================================
// MEMORY REGION INTEGRATION TESTS
// =============================================================================

test "all memory regions in appropriate contexts" {
    var helper = IntegrationTestHelper.init(testing.allocator);

    const source =
        \\contract MemoryRegionTest {
        \\    storage storageVar: u32;
        \\    immutable immutableVar: u32;
        \\    const CONSTANT: u32 = 42;
        \\    
        \\    function init() {
        \\        storageVar = 1;
        \\        immutableVar = 2;
        \\    }
        \\    
        \\    function test() public {
        \\        let stackVar: u32 = 3;
        \\        memory memoryVar: u32 = 4;
        \\        tstore tstoreVar: u32 = 5;
        \\    }
        \\}
    ;

    const result = try helper.compileAndAnalyze(source);
    defer helper.cleanup(result);

    // Should compile without errors (proper memory region usage)
    try helper.expectNoErrors(result.diagnostics);
}

test "const variable without initializer" {
    var helper = IntegrationTestHelper.init(testing.allocator);

    const source =
        \\contract ConstTest {
        \\    const INVALID_CONST: u32; // No initializer
        \\    
        \\    function init() {
        \\    }
        \\}
    ;

    const result = try helper.compileAndAnalyze(source);
    defer helper.cleanup(result);

    // Should have error for const without initializer
    try helper.expectErrorCount(result.diagnostics, 1);
}

// =============================================================================
// FUNCTION ANALYSIS INTEGRATION TESTS
// =============================================================================

test "function with requires and ensures" {
    var helper = IntegrationTestHelper.init(testing.allocator);

    const source =
        \\contract FormalVerificationTest {
        \\    function init() {
        \\    }
        \\    
        \\    function add(a: u32, b: u32) -> u32
        \\    requires a > 0
        \\    requires b > 0
        \\    ensures result > a
        \\    ensures result > b
        \\    {
        \\        return a + b;
        \\    }
        \\}
    ;

    const result = try helper.compileAndAnalyze(source);
    defer helper.cleanup(result);

    // Should compile without errors (requires/ensures are syntactically valid)
    try helper.expectNoErrors(result.diagnostics);
}

test "requires clause with old expression" {
    var helper = IntegrationTestHelper.init(testing.allocator);

    const source =
        \\contract InvalidRequires {
        \\    storage value: u32;
        \\    
        \\    function init() {
        \\        value = 0;
        \\    }
        \\    
        \\    function updateValue(newValue: u32)
        \\    requires old(value) > 0  // Invalid: old() in requires
        \\    {
        \\        value = newValue;
        \\    }
        \\}
    ;

    const result = try helper.compileAndAnalyze(source);
    defer helper.cleanup(result);

    // Should have error for old() in requires clause
    try helper.expectErrorCount(result.diagnostics, 1);
}

test "function without return statement" {
    var helper = IntegrationTestHelper.init(testing.allocator);

    const source =
        \\contract MissingReturn {
        \\    function init() {
        \\    }
        \\    
        \\    function getValue() -> u32 {
        \\        // Missing return statement
        \\        let temp: u32 = 42;
        \\    }
        \\}
    ;

    const result = try helper.compileAndAnalyze(source);
    defer helper.cleanup(result);

    // Should have error for missing return statement
    try helper.expectErrorCount(result.diagnostics, 1);
}

// =============================================================================
// ERROR HANDLING INTEGRATION TESTS
// =============================================================================

test "error declaration and usage" {
    var helper = IntegrationTestHelper.init(testing.allocator);

    const source =
        \\contract ErrorTest {
        \\    error CustomError;
        \\    error AnotherError;
        \\    
        \\    function init() {
        \\    }
        \\    
        \\    function mayFail() -> !u32 {
        \\        return CustomError;
        \\    }
        \\}
    ;

    const result = try helper.compileAndAnalyze(source);
    defer helper.cleanup(result);

    // Should compile without errors (valid error usage)
    try helper.expectNoErrors(result.diagnostics);
}

test "duplicate error declaration" {
    var helper = IntegrationTestHelper.init(testing.allocator);

    const source =
        \\contract DuplicateError {
        \\    error SameError;
        \\    error SameError; // Duplicate
        \\    
        \\    function init() {
        \\    }
        \\}
    ;

    const result = try helper.compileAndAnalyze(source);
    defer helper.cleanup(result);

    // Should have error for duplicate error declaration
    try helper.expectErrorCount(result.diagnostics, 1);
}

test "undefined error usage" {
    var helper = IntegrationTestHelper.init(testing.allocator);

    const source =
        \\contract UndefinedError {
        \\    function init() {
        \\    }
        \\    
        \\    function mayFail() -> !u32 {
        \\        return UndefinedError; // Error not declared
        \\    }
        \\}
    ;

    const result = try helper.compileAndAnalyze(source);
    defer helper.cleanup(result);

    // Should have error for undefined error
    try helper.expectErrorCount(result.diagnostics, 1);
}

// =============================================================================
// IMPORT SYSTEM INTEGRATION TESTS
// =============================================================================

test "standard library imports" {
    var helper = IntegrationTestHelper.init(testing.allocator);

    const source =
        \\import std/transaction as tx;
        \\import std/block;
        \\import std/constants;
        \\
        \\contract ImportTest {
        \\    function init() {
        \\    }
        \\    
        \\    function getCurrentSender() -> address {
        \\        return tx.sender;
        \\    }
        \\    
        \\    function getBlockNumber() -> u256 {
        \\        return block.number;
        \\    }
        \\}
    ;

    const result = try helper.compileAndAnalyze(source);
    defer helper.cleanup(result);

    // Should compile without errors (valid std imports)
    try helper.expectNoErrors(result.diagnostics);
}

test "invalid import path" {
    var helper = IntegrationTestHelper.init(testing.allocator);

    const source =
        \\import invalid/module;
        \\
        \\contract ImportTest {
        \\    function init() {
        \\    }
        \\}
    ;

    const result = try helper.compileAndAnalyze(source);
    defer helper.cleanup(result);

    // Should have warning for unknown module
    try helper.expectWarningCount(result.diagnostics, 1);
}

test "invalid field access on std modules" {
    var helper = IntegrationTestHelper.init(testing.allocator);

    const source =
        \\import std/transaction;
        \\
        \\contract FieldAccessTest {
        \\    function init() {
        \\    }
        \\    
        \\    function getInvalidField() -> u256 {
        \\        return transaction.invalidField;
        \\    }
        \\}
    ;

    const result = try helper.compileAndAnalyze(source);
    defer helper.cleanup(result);

    // Should have error for invalid field access
    try helper.expectErrorCount(result.diagnostics, 1);
}

// =============================================================================
// STRUCT AND ENUM INTEGRATION TESTS
// =============================================================================

test "struct declaration and usage" {
    var helper = IntegrationTestHelper.init(testing.allocator);

    const source =
        \\struct Point {
        \\    x: u32;
        \\    y: u32;
        \\}
        \\
        \\contract StructTest {
        \\    storage origin: Point;
        \\    
        \\    function init() {
        \\        origin = Point{
        \\            x: 0,
        \\            y: 0
        \\        };
        \\    }
        \\}
    ;

    const result = try helper.compileAndAnalyze(source);
    defer helper.cleanup(result);

    // Should compile without errors
    try helper.expectNoErrors(result.diagnostics);
}

test "empty struct declaration" {
    var helper = IntegrationTestHelper.init(testing.allocator);

    const source =
        \\struct EmptyStruct {
        \\}
        \\
        \\contract StructTest {
        \\    function init() {
        \\    }
        \\}
    ;

    const result = try helper.compileAndAnalyze(source);
    defer helper.cleanup(result);

    // Should have warning for empty struct
    try helper.expectWarningCount(result.diagnostics, 1);
}

test "struct with duplicate field names" {
    var helper = IntegrationTestHelper.init(testing.allocator);

    const source =
        \\struct DuplicateFields {
        \\    field: u32;
        \\    field: u64; // Duplicate name
        \\}
        \\
        \\contract StructTest {
        \\    function init() {
        \\    }
        \\}
    ;

    const result = try helper.compileAndAnalyze(source);
    defer helper.cleanup(result);

    // Should have error for duplicate field names
    try helper.expectErrorCount(result.diagnostics, 1);
}

test "enum declaration and usage" {
    var helper = IntegrationTestHelper.init(testing.allocator);

    const source =
        \\enum Status: u8 {
        \\    Pending = 0,
        \\    Active = 1,
        \\    Inactive = 2
        \\}
        \\
        \\contract EnumTest {
        \\    storage currentStatus: Status;
        \\    
        \\    function init() {
        \\        currentStatus = Status.Pending;
        \\    }
        \\}
    ;

    const result = try helper.compileAndAnalyze(source);
    defer helper.cleanup(result);

    // Should compile without errors
    try helper.expectNoErrors(result.diagnostics);
}

// =============================================================================
// EXPRESSION ANALYSIS INTEGRATION TESTS
// =============================================================================

test "division by zero detection in expressions" {
    var helper = IntegrationTestHelper.init(testing.allocator);

    const source =
        \\contract DivisionTest {
        \\    function init() {
        \\    }
        \\    
        \\    function divide() -> u32 {
        \\        return 10 / 0; // Division by zero
        \\    }
        \\}
    ;

    const result = try helper.compileAndAnalyze(source);
    defer helper.cleanup(result);

    // Should have error for division by zero
    try helper.expectErrorCount(result.diagnostics, 1);
}

test "complex binary expressions" {
    var helper = IntegrationTestHelper.init(testing.allocator);

    const source =
        \\contract ExpressionTest {
        \\    function init() {
        \\    }
        \\    
        \\    function complexCalculation(a: u32, b: u32, c: u32) -> u32 {
        \\        return (a + b) * c - (a / (b + 1));
        \\    }
        \\}
    ;

    const result = try helper.compileAndAnalyze(source);
    defer helper.cleanup(result);

    // Should compile without errors
    try helper.expectNoErrors(result.diagnostics);
}

test "shift expressions with large amounts" {
    var helper = IntegrationTestHelper.init(testing.allocator);

    const source =
        \\contract ShiftTest {
        \\    function init() {
        \\    }
        \\    
        \\    function largeShift() -> u32 {
        \\        return 1 << 300; // Very large shift
        \\    }
        \\}
    ;

    const result = try helper.compileAndAnalyze(source);
    defer helper.cleanup(result);

    // Should have warning for large shift amount
    try helper.expectWarningCount(result.diagnostics, 1);
}

// =============================================================================
// COMPREHENSIVE WORKFLOW TESTS
// =============================================================================

test "complete ERC20-style token contract" {
    var helper = IntegrationTestHelper.init(testing.allocator);

    const source =
        \\import std/transaction as tx;
        \\
        \\contract SimpleToken {
        \\    storage totalSupply: u256;
        \\    storage balances: map[address, u256];
        \\    immutable name: string;
        \\    immutable symbol: string;
        \\    
        \\    error InsufficientBalance;
        \\    error InvalidTransfer;
        \\    
        \\    log Transfer(from: address, to: address, value: u256);
        \\    
        \\    function init() {
        \\        name = "SimpleToken";
        \\        symbol = "STK";
        \\        totalSupply = 1000000;
        \\        balances[tx.sender] = totalSupply;
        \\    }
        \\    
        \\    function transfer(to: address, amount: u256) -> !bool
        \\    requires amount > 0
        \\    requires balances[tx.sender] >= amount
        \\    ensures balances[tx.sender] == old(balances[tx.sender]) - amount
        \\    ensures balances[to] == old(balances[to]) + amount
        \\    {
        \\        if (balances[tx.sender] < amount) {
        \\            return InsufficientBalance;
        \\        }
        \\        
        \\        balances[tx.sender] -= amount;
        \\        balances[to] += amount;
        \\        
        \\        log Transfer(tx.sender, to, amount);
        \\        return true;
        \\    }
        \\    
        \\    function balanceOf(owner: address) -> u256 {
        \\        return balances[owner];
        \\    }
        \\}
    ;

    const result = try helper.compileAndAnalyze(source);
    defer helper.cleanup(result);

    // Complex contract should compile without errors
    try helper.expectNoErrors(result.diagnostics);
}

test "contract with all language features" {
    var helper = IntegrationTestHelper.init(testing.allocator);

    const source =
        \\import std/constants;
        \\import std/transaction as tx;
        \\
        \\struct UserData {
        \\    id: u32;
        \\    balance: u256;
        \\    active: bool;
        \\}
        \\
        \\enum UserStatus: u8 {
        \\    Inactive = 0,
        \\    Active = 1,
        \\    Suspended = 2
        \\}
        \\
        \\contract ComprehensiveTest {
        \\    storage users: map[address, UserData];
        \\    storage userCount: u32;
        \\    immutable owner: address;
        \\    const MAX_USERS: u32 = 10000;
        \\    
        \\    error UserNotFound;
        \\    error MaxUsersReached;
        \\    
        \\    log UserRegistered(user: address, id: u32);
        \\    log UserUpdated(user: address, status: UserStatus);
        \\    
        \\    function init() {
        \\        owner = tx.sender;
        \\        userCount = 0;
        \\    }
        \\    
        \\    function registerUser() -> !u32
        \\    requires userCount < MAX_USERS
        \\    ensures userCount == old(userCount) + 1
        \\    {
        \\        if (userCount >= MAX_USERS) {
        \\            return MaxUsersReached;
        \\        }
        \\        
        \\        let newId = userCount + 1;
        \\        users[tx.sender] = UserData{
        \\            id: newId,
        \\            balance: 0,
        \\            active: true
        \\        };
        \\        userCount = newId;
        \\        
        \\        log UserRegistered(tx.sender, newId);
        \\        return newId;
        \\    }
        \\    
        \\    function updateUserStatus(user: address, status: UserStatus) -> !bool
        \\    requires user != constants.ZERO_ADDRESS
        \\    {
        \\        if (users[user].id == 0) {
        \\            return UserNotFound;
        \\        }
        \\        
        \\        users[user].active = (status == UserStatus.Active);
        \\        log UserUpdated(user, status);
        \\        return true;
        \\    }
        \\}
    ;

    const result = try helper.compileAndAnalyze(source);
    defer helper.cleanup(result);

    // Comprehensive contract should compile successfully
    try helper.expectNoErrors(result.diagnostics);
}
