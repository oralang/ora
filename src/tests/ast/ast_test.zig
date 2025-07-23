const std = @import("std");
const testing = std.testing;
const lexer = @import("lexer.zig");
const parser = @import("parser.zig");
const ast = @import("ast.zig");

// =============================================================================
// AST TEST UTILITIES
// =============================================================================

/// Helper for parsing Ora source code into AST
const AstTestHelper = struct {
    allocator: std.mem.Allocator,

    fn init(allocator: std.mem.Allocator) AstTestHelper {
        return AstTestHelper{ .allocator = allocator };
    }

    /// Parse Ora source code into AST nodes
    fn parseSource(self: *AstTestHelper, source: []const u8) ![]ast.AstNode {
        // Step 1: Lex the source code
        const tokens = try lexer.scan(source, self.allocator);
        defer self.allocator.free(tokens);

        // Step 2: Parse tokens into AST
        var p = parser.Parser.init(self.allocator, tokens);
        return p.parse();
    }

    /// Cleanup AST nodes
    fn cleanup(self: *AstTestHelper, nodes: []ast.AstNode) void {
        ast.deinitAstNodes(self.allocator, nodes);
    }

    /// Assert that parsing succeeds without error
    fn expectParseSuccess(self: *AstTestHelper, source: []const u8) ![]ast.AstNode {
        return self.parseSource(source);
    }

    /// Assert that parsing fails with specific error
    fn expectParseError(self: *AstTestHelper, source: []const u8, expected_error: parser.ParserError) !void {
        const result = self.parseSource(source);
        try testing.expectError(expected_error, result);
    }

    /// Find the first node of a specific type in the AST
    fn findNode(self: *AstTestHelper, nodes: []ast.AstNode, comptime NodeType: type) ?*NodeType {
        _ = self;
        for (nodes) |*node| {
            switch (node.*) {
                inline else => |*n| {
                    if (@TypeOf(n.*) == NodeType) {
                        return @ptrCast(n);
                    }
                },
            }
        }
        return null;
    }

    /// Count nodes of a specific type
    fn countNodes(self: *AstTestHelper, nodes: []ast.AstNode, node_tag: std.meta.Tag(ast.AstNode)) u32 {
        _ = self;
        var count: u32 = 0;
        for (nodes) |*node| {
            if (@as(std.meta.Tag(ast.AstNode), node.*) == node_tag) {
                count += 1;
            }
        }
        return count;
    }
};

// =============================================================================
// BASIC AST PARSING TESTS
// =============================================================================

test "parse empty source" {
    var helper = AstTestHelper.init(testing.allocator);

    const nodes = try helper.expectParseSuccess("");
    defer helper.cleanup(nodes);

    try testing.expect(nodes.len == 0);
}

test "parse simple contract" {
    var helper = AstTestHelper.init(testing.allocator);

    const source =
        \\contract SimpleContract {
        \\    pub fn init() {
        \\    }
        \\}
    ;

    const nodes = try helper.expectParseSuccess(source);
    defer helper.cleanup(nodes);

    try testing.expect(nodes.len == 1);
    try testing.expect(@as(std.meta.Tag(ast.AstNode), nodes[0]) == .Contract);

    const contract = nodes[0].Contract;
    try testing.expect(std.mem.eql(u8, contract.name, "SimpleContract"));
    try testing.expect(contract.body.len == 1);
    try testing.expect(@as(std.meta.Tag(ast.AstNode), contract.body[0]) == .Function);
}

test "parse contract with storage variables" {
    var helper = AstTestHelper.init(testing.allocator);

    const source =
        \\contract StorageContract {
        \\    storage var balance: u256;
        \\    storage var owner: address;
        \\    immutable name: string;
        \\    
        \\    pub fn init() {
        \\        balance = 0;
        \\        owner = std.transaction.sender;
        \\        name = "TestContract";
        \\    }
        \\}
    ;

    const nodes = try helper.expectParseSuccess(source);
    defer helper.cleanup(nodes);

    try testing.expect(nodes.len == 1);
    const contract = nodes[0].Contract;
    try testing.expect(std.mem.eql(u8, contract.name, "StorageContract"));
    try testing.expect(contract.body.len == 4); // 3 variables + 1 function

    // Check storage variable
    try testing.expect(@as(std.meta.Tag(ast.AstNode), contract.body[0]) == .VariableDecl);
    const storage_var = contract.body[0].VariableDecl;
    try testing.expect(std.mem.eql(u8, storage_var.name, "balance"));
    try testing.expect(storage_var.region == .Storage);
    try testing.expect(@as(std.meta.Tag(ast.TypeRef), storage_var.typ) == .U256);

    // Check immutable variable
    try testing.expect(@as(std.meta.Tag(ast.AstNode), contract.body[2]) == .VariableDecl);
    const immutable_var = contract.body[2].VariableDecl;
    try testing.expect(std.mem.eql(u8, immutable_var.name, "name"));
    try testing.expect(immutable_var.region == .Immutable);
    try testing.expect(@as(std.meta.Tag(ast.TypeRef), immutable_var.typ) == .String);
}

test "parse function with parameters and return type" {
    var helper = AstTestHelper.init(testing.allocator);

    const source =
        \\contract MathContract {
        \\    pub fn add(a: u32, b: u32) -> u32 {
        \\        return a + b;
        \\    }
        \\}
    ;

    const nodes = try helper.expectParseSuccess(source);
    defer helper.cleanup(nodes);

    const contract = nodes[0].Contract;
    const function = contract.body[0].Function;

    try testing.expect(std.mem.eql(u8, function.name, "add"));
    try testing.expect(function.pub_ == true);
    try testing.expect(function.parameters.len == 2);
    try testing.expect(function.return_type != null);

    // Check parameters
    try testing.expect(std.mem.eql(u8, function.parameters[0].name, "a"));
    try testing.expect(@as(std.meta.Tag(ast.TypeRef), function.parameters[0].typ) == .U32);
    try testing.expect(std.mem.eql(u8, function.parameters[1].name, "b"));
    try testing.expect(@as(std.meta.Tag(ast.TypeRef), function.parameters[1].typ) == .U32);

    // Check return type
    try testing.expect(@as(std.meta.Tag(ast.TypeRef), function.return_type.?) == .U32);

    // Check function body
    try testing.expect(function.body.statements.len == 1);
    try testing.expect(@as(std.meta.Tag(ast.StmtNode), function.body.statements[0]) == .Return);
}

test "parse expressions and literals" {
    var helper = AstTestHelper.init(testing.allocator);

    const source =
        \\contract ExpressionTest {
        \\    pub fn test() {
        \\        var a: u32 = 42;
        \\        var b: bool = true;
        \\        var c: string = "hello";
        \\        var d: address = 0x1234567890123456789012345678901234567890;
        \\    }
        \\}
    ;

    const nodes = try helper.expectParseSuccess(source);
    defer helper.cleanup(nodes);

    const contract = nodes[0].Contract;
    const function = contract.body[0].Function;
    const statements = function.body.statements;

    try testing.expect(statements.len == 4);

    // Test integer literal
    const var_a = statements[0].VariableDecl;
    try testing.expect(std.mem.eql(u8, var_a.name, "a"));
    try testing.expect(var_a.value != null);
    try testing.expect(@as(std.meta.Tag(ast.ExprNode), var_a.value.?) == .Literal);
    try testing.expect(@as(std.meta.Tag(ast.LiteralNode), var_a.value.?.Literal) == .Integer);
    try testing.expect(std.mem.eql(u8, var_a.value.?.Literal.Integer.value, "42"));

    // Test bool literal
    const var_b = statements[1].VariableDecl;
    try testing.expect(@as(std.meta.Tag(ast.LiteralNode), var_b.value.?.Literal) == .Bool);
    try testing.expect(var_b.value.?.Literal.Bool.value == true);

    // Test string literal
    const var_c = statements[2].VariableDecl;
    try testing.expect(@as(std.meta.Tag(ast.LiteralNode), var_c.value.?.Literal) == .String);
    try testing.expect(std.mem.eql(u8, var_c.value.?.Literal.String.value, "hello"));

    // Test address literal
    const var_d = statements[3].VariableDecl;
    try testing.expect(@as(std.meta.Tag(ast.LiteralNode), var_d.value.?.Literal) == .Address);
    try testing.expect(std.mem.eql(u8, var_d.value.?.Literal.Address.value, "0x1234567890123456789012345678901234567890"));
}

test "parse binary expressions" {
    var helper = AstTestHelper.init(testing.allocator);

    const source =
        \\contract BinaryExprTest {
        \\    pub fn calc() -> u32 {
        \\        return 10 + 20 * 30;
        \\    }
        \\}
    ;

    const nodes = try helper.expectParseSuccess(source);
    defer helper.cleanup(nodes);

    const contract = nodes[0].Contract;
    const function = contract.body[0].Function;
    const return_stmt = function.body.statements[0].Return;

    try testing.expect(return_stmt.value != null);
    try testing.expect(@as(std.meta.Tag(ast.ExprNode), return_stmt.value.?) == .Binary);

    const binary_expr = return_stmt.value.?.Binary;
    try testing.expect(binary_expr.operator == .Plus);
    try testing.expect(@as(std.meta.Tag(ast.ExprNode), binary_expr.lhs.*) == .Literal);
    try testing.expect(@as(std.meta.Tag(ast.ExprNode), binary_expr.rhs.*) == .Binary);
}

test "parse if statement" {
    var helper = AstTestHelper.init(testing.allocator);

    const source =
        \\contract ControlFlowTest {
        \\    pub fn conditionalReturn(x: u32) -> bool {
        \\        if (x > 10) {
        \\            return true;
        \\        } else {
        \\            return false;
        \\        }
        \\    }
        \\}
    ;

    const nodes = try helper.expectParseSuccess(source);
    defer helper.cleanup(nodes);

    const contract = nodes[0].Contract;
    const function = contract.body[0].Function;
    const if_stmt = function.body.statements[0].If;

    // Check condition
    try testing.expect(@as(std.meta.Tag(ast.ExprNode), if_stmt.condition) == .Binary);

    // Check then branch
    try testing.expect(if_stmt.then_branch.statements.len == 1);
    try testing.expect(@as(std.meta.Tag(ast.StmtNode), if_stmt.then_branch.statements[0]) == .Return);

    // Check else branch
    try testing.expect(if_stmt.else_branch != null);
    try testing.expect(if_stmt.else_branch.?.statements.len == 1);
    try testing.expect(@as(std.meta.Tag(ast.StmtNode), if_stmt.else_branch.?.statements[0]) == .Return);
}

test "parse while loop with invariant" {
    var helper = AstTestHelper.init(testing.allocator);

    const source =
        \\contract LoopTest {
        \\    pub fn countdown(n: u32) {
        \\        while (n > 0) {
        \\            n = n - 1;
        \\        }
        \\    }
        \\}
    ;

    const nodes = try helper.expectParseSuccess(source);
    defer helper.cleanup(nodes);

    const contract = nodes[0].Contract;
    const function = contract.body[0].Function;
    const while_stmt = function.body.statements[0].While;

    // Check condition
    try testing.expect(@as(std.meta.Tag(ast.ExprNode), while_stmt.condition) == .Binary);

    // Check that invariants list is empty (since we removed the invariant from the source)
    try testing.expect(while_stmt.invariants.len == 0);

    // Check body
    try testing.expect(while_stmt.body.statements.len == 1);
    try testing.expect(@as(std.meta.Tag(ast.StmtNode), while_stmt.body.statements[0]) == .Expr);
}

// =============================================================================
// FORMAL VERIFICATION TESTS
// =============================================================================

test "parse function with requires and ensures" {
    var helper = AstTestHelper.init(testing.allocator);

    const source =
        \\contract FormalTest {
        \\    pub fn divide(a: u32, b: u32) -> u32 {
        \\        requires(b != 0);
        \\        requires(a >= b);
        \\        ensures(result <= a);
        \\        ensures(result >= 0);
        \\        return a / b;
        \\    }
        \\}
    ;

    const nodes = try helper.expectParseSuccess(source);
    defer helper.cleanup(nodes);

    const contract = nodes[0].Contract;
    const function = contract.body[0].Function;

    // Check that function body contains requires/ensures statements
    try testing.expect(function.body.statements.len == 5); // 4 requires/ensures + 1 return

    // Check first statement is requires
    try testing.expect(@as(std.meta.Tag(ast.StmtNode), function.body.statements[0]) == .Requires);
    // Check second statement is requires
    try testing.expect(@as(std.meta.Tag(ast.StmtNode), function.body.statements[1]) == .Requires);
    // Check third statement is ensures
    try testing.expect(@as(std.meta.Tag(ast.StmtNode), function.body.statements[2]) == .Ensures);
    // Check fourth statement is ensures
    try testing.expect(@as(std.meta.Tag(ast.StmtNode), function.body.statements[3]) == .Ensures);
    // Check fifth statement is return
    try testing.expect(@as(std.meta.Tag(ast.StmtNode), function.body.statements[4]) == .Return);
}

test "parse ensures with old() expression" {
    var helper = AstTestHelper.init(testing.allocator);

    const source =
        \\contract StateTest {
        \\    storage balance: u256;
        \\    
        \\    fn withdraw(amount: u256) {
        \\        ensures balance == old(balance) - amount;
        \\        balance = balance - amount;
        \\    }
        \\}
    ;

    const nodes = try helper.expectParseSuccess(source);
    defer helper.cleanup(nodes);

    const contract = nodes[0].Contract;
    const function = contract.body[1].Function;

    // Check that first statement is an ensures statement
    try testing.expect(function.body.statements.len == 2); // ensures + assignment
    try testing.expect(@as(std.meta.Tag(ast.StmtNode), function.body.statements[0]) == .Ensures);

    const ensures_stmt = function.body.statements[0].Ensures;
    const ensures_expr = ensures_stmt.condition;
    try testing.expect(@as(std.meta.Tag(ast.ExprNode), ensures_expr) == .Binary);

    // Check that old() expression is parsed correctly
    const binary = ensures_expr.Binary;
    try testing.expect(@as(std.meta.Tag(ast.ExprNode), binary.rhs.*) == .Binary);
    const rhs_binary = binary.rhs.Binary;
    try testing.expect(@as(std.meta.Tag(ast.ExprNode), rhs_binary.lhs.*) == .Old);
}

// =============================================================================
// ERROR HANDLING TESTS
// =============================================================================

test "parse error declarations and error returns" {
    var helper = AstTestHelper.init(testing.allocator);

    const source =
        \\contract ErrorTest {
        \\    error InsufficientFunds;
        \\    error InvalidOperation;
        \\    
        \\    fn withdraw(amount: u256) -> !bool {
        \\        if (balance < amount) {
        \\            return InsufficientFunds;
        \\        }
        \\        return true;
        \\    }
        \\}
    ;

    const nodes = try helper.expectParseSuccess(source);
    defer helper.cleanup(nodes);

    const contract = nodes[0].Contract;

    // Check error declarations
    try testing.expect(contract.body.len == 3); // 2 errors + 1 function
    try testing.expect(@as(std.meta.Tag(ast.AstNode), contract.body[0]) == .ErrorDecl);
    try testing.expect(@as(std.meta.Tag(ast.AstNode), contract.body[1]) == .ErrorDecl);

    const error1 = contract.body[0].ErrorDecl;
    try testing.expect(std.mem.eql(u8, error1.name, "InsufficientFunds"));

    // Check function with error union return type
    const function = contract.body[2].Function;
    try testing.expect(function.return_type != null);
    try testing.expect(@as(std.meta.Tag(ast.TypeRef), function.return_type.?) == .ErrorUnion);

    // Check error return statement
    const if_stmt = function.body.statements[0].If;
    const return_stmt = if_stmt.then_branch.statements[0].Return;
    try testing.expect(@as(std.meta.Tag(ast.ExprNode), return_stmt.value.?) == .ErrorReturn);

    const error_return = return_stmt.value.?.ErrorReturn;
    try testing.expect(std.mem.eql(u8, error_return.error_name, "InsufficientFunds"));
}

test "parse try-catch blocks" {
    var helper = AstTestHelper.init(testing.allocator);

    const source =
        \\contract TryTest {
        \\    fn handleErrors() {
        \\        try {
        \\            let result = riskyOperation();
        \\        } catch (e) {
        \\            log ErrorOccurred(e);
        \\        }
        \\    }
        \\}
    ;

    const nodes = try helper.expectParseSuccess(source);
    defer helper.cleanup(nodes);

    const contract = nodes[0].Contract;
    const function = contract.body[0].Function;
    const try_block = function.body.statements[0].TryBlock;

    // Check try block
    try testing.expect(try_block.try_block.statements.len == 1);
    try testing.expect(@as(std.meta.Tag(ast.StmtNode), try_block.try_block.statements[0]) == .VariableDecl);

    // Check catch block
    try testing.expect(try_block.catch_block != null);
    try testing.expect(std.mem.eql(u8, try_block.catch_block.?.error_variable.?, "e"));
    try testing.expect(try_block.catch_block.?.block.statements.len == 1);
}

// =============================================================================
// STRUCT AND ENUM TESTS
// =============================================================================

test "parse struct declaration" {
    var helper = AstTestHelper.init(testing.allocator);

    const source =
        \\struct Point {
        \\    x: u32,
        \\    y: u32,
        \\}
        \\
        \\contract StructTest {
        \\    storage var origin: Point;
        \\    
        \\    pub fn init() {
        \\        origin = Point {
        \\            x: 0,
        \\            y: 0
        \\        };
        \\    }
        \\}
    ;

    const nodes = try helper.expectParseSuccess(source);
    defer helper.cleanup(nodes);

    try testing.expect(nodes.len == 2);

    // Check struct declaration
    try testing.expect(@as(std.meta.Tag(ast.AstNode), nodes[0]) == .StructDecl);
    const struct_decl = nodes[0].StructDecl;
    try testing.expect(std.mem.eql(u8, struct_decl.name, "Point"));
    try testing.expect(struct_decl.fields.len == 2);

    // Check struct fields
    try testing.expect(std.mem.eql(u8, struct_decl.fields[0].name, "x"));
    try testing.expect(@as(std.meta.Tag(ast.TypeRef), struct_decl.fields[0].typ) == .U32);
    try testing.expect(std.mem.eql(u8, struct_decl.fields[1].name, "y"));
    try testing.expect(@as(std.meta.Tag(ast.TypeRef), struct_decl.fields[1].typ) == .U32);

    // Check struct usage in contract
    const contract = nodes[1].Contract;
    const storage_var = contract.body[0].VariableDecl;
    try testing.expect(@as(std.meta.Tag(ast.TypeRef), storage_var.typ) == .Identifier);
    try testing.expect(std.mem.eql(u8, storage_var.typ.Identifier, "Point"));
}

test "parse enum declaration" {
    var helper = AstTestHelper.init(testing.allocator);

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
        \\    fn setActive() {
        \\        currentStatus = Status.Active;
        \\    }
        \\}
    ;

    const nodes = try helper.expectParseSuccess(source);
    defer helper.cleanup(nodes);

    try testing.expect(nodes.len == 2);

    // Check enum declaration
    try testing.expect(@as(std.meta.Tag(ast.AstNode), nodes[0]) == .EnumDecl);
    const enum_decl = nodes[0].EnumDecl;
    try testing.expect(std.mem.eql(u8, enum_decl.name, "Status"));
    try testing.expect(enum_decl.base_type != null);
    try testing.expect(@as(std.meta.Tag(ast.TypeRef), enum_decl.base_type.?) == .U8);
    try testing.expect(enum_decl.variants.len == 3);

    // Check enum variants
    try testing.expect(std.mem.eql(u8, enum_decl.variants[0].name, "Pending"));
    try testing.expect(enum_decl.variants[0].value != null);
    try testing.expect(@as(std.meta.Tag(ast.ExprNode), enum_decl.variants[0].value.?) == .Literal);

    // Check enum literal usage
    const contract = nodes[1].Contract;
    const function = contract.body[1].Function;
    const assignment = function.body.statements[0].Expr.Assignment;
    try testing.expect(@as(std.meta.Tag(ast.ExprNode), assignment.value.*) == .EnumLiteral);

    const enum_literal = assignment.value.EnumLiteral;
    try testing.expect(std.mem.eql(u8, enum_literal.enum_name, "Status"));
    try testing.expect(std.mem.eql(u8, enum_literal.variant_name, "Active"));
}

// =============================================================================
// IMPORT SYSTEM TESTS
// =============================================================================

test "parse import statements" {
    var helper = AstTestHelper.init(testing.allocator);

    const source =
        \\import std/transaction as tx;
        \\import std/block;
        \\
        \\contract ImportTest {
        \\    fn getCurrentSender() -> address {
        \\        return tx.sender;
        \\    }
        \\}
    ;

    const nodes = try helper.expectParseSuccess(source);
    defer helper.cleanup(nodes);

    try testing.expect(nodes.len == 3); // 2 imports + 1 contract

    // Check first import with alias
    try testing.expect(@as(std.meta.Tag(ast.AstNode), nodes[0]) == .Import);
    const import1 = nodes[0].Import;
    try testing.expect(std.mem.eql(u8, import1.name, "tx"));
    try testing.expect(std.mem.eql(u8, import1.path, "std/transaction"));

    // Check second import without alias
    try testing.expect(@as(std.meta.Tag(ast.AstNode), nodes[1]) == .Import);
    const import2 = nodes[1].Import;
    try testing.expect(std.mem.eql(u8, import2.path, "std/block"));
}

// =============================================================================
// LOG EVENT TESTS
// =============================================================================

test "parse log declarations and usage" {
    var helper = AstTestHelper.init(testing.allocator);

    const source =
        \\contract EventTest {
        \\    log Transfer(from: address, to: address, value: u256);
        \\    log Approval(owner: address, spender: address, value: u256);
        \\    
        \\    fn transfer(to: address, amount: u256) {
        \\        log Transfer(tx.sender, to, amount);
        \\    }
        \\}
    ;

    const nodes = try helper.expectParseSuccess(source);
    defer helper.cleanup(nodes);

    const contract = nodes[0].Contract;

    // Check log declarations
    try testing.expect(@as(std.meta.Tag(ast.AstNode), contract.body[0]) == .LogDecl);
    try testing.expect(@as(std.meta.Tag(ast.AstNode), contract.body[1]) == .LogDecl);

    const log_decl = contract.body[0].LogDecl;
    try testing.expect(std.mem.eql(u8, log_decl.name, "Transfer"));
    try testing.expect(log_decl.fields.len == 3);
    try testing.expect(std.mem.eql(u8, log_decl.fields[0].name, "from"));
    try testing.expect(@as(std.meta.Tag(ast.TypeRef), log_decl.fields[0].typ) == .Address);

    // Check log usage
    const function = contract.body[2].Function;
    const log_stmt = function.body.statements[0].Log;
    try testing.expect(std.mem.eql(u8, log_stmt.event_name, "Transfer"));
    try testing.expect(log_stmt.args.len == 3);
}

// =============================================================================
// ERROR CASE TESTS
// =============================================================================

test "syntax error - missing semicolon" {
    var helper = AstTestHelper.init(testing.allocator);

    const source =
        \\contract SyntaxError {
        \\    fn test() {
        \\        let x = 42  // Missing semicolon
        \\        return x;
        \\    }
        \\}
    ;

    try helper.expectParseError(source, parser.ParserError.UnexpectedToken);
}

test "syntax error - invalid memory region" {
    var helper = AstTestHelper.init(testing.allocator);

    const source =
        \\contract InvalidMemoryRegion {
        \\    invalid_region x: u32;
        \\}
    ;

    try helper.expectParseError(source, parser.ParserError.InvalidMemoryRegion);
}

test "syntax error - unexpected EOF" {
    var helper = AstTestHelper.init(testing.allocator);

    const source =
        \\contract IncompleteContract {
        \\    fn test() {
        \\        let x = 42;
        \\        // Missing closing brace
    ;

    try helper.expectParseError(source, parser.ParserError.UnexpectedEof);
}

// =============================================================================
// COMPREHENSIVE INTEGRATION TESTS
// =============================================================================

test "parse complex ERC20-style contract" {
    var helper = AstTestHelper.init(testing.allocator);

    const source =
        \\import std/transaction as tx;
        \\
        \\contract ComplexToken {
        \\    storage totalSupply: u256;
        \\    storage balances: map[address, u256];
        \\    storage allowances: map[address, map[address, u256]];
        \\    
        \\    immutable name: string;
        \\    immutable symbol: string;
        \\    immutable decimals: u8;
        \\    
        \\    error InsufficientBalance;
        \\    error InsufficientAllowance;
        \\    
        \\    log Transfer(from: address, to: address, value: u256);
        \\    log Approval(owner: address, spender: address, value: u256);
        \\    
        \\    fn init() {
        \\        name = "ComplexToken";
        \\        symbol = "CTK";
        \\        decimals = 18;
        \\        totalSupply = 1000000 * 10 ** 18;
        \\        balances[tx.sender] = totalSupply;
        \\    }
        \\    
        \\    fn transfer(to: address, amount: u256) -> !bool 
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
        \\}
    ;

    const nodes = try helper.expectParseSuccess(source);
    defer helper.cleanup(nodes);

    try testing.expect(nodes.len == 2); // import + contract

    const contract = nodes[1].Contract;
    try testing.expect(std.mem.eql(u8, contract.name, "ComplexToken"));
    try testing.expect(contract.body.len == 9); // 3 storage + 3 immutable + 2 errors + 2 logs + 2 functions

    // Verify contract structure
    try testing.expect(helper.countNodes(contract.body, .VariableDecl) == 6);
    try testing.expect(helper.countNodes(contract.body, .ErrorDecl) == 2);
    try testing.expect(helper.countNodes(contract.body, .LogDecl) == 2);
    try testing.expect(helper.countNodes(contract.body, .Function) == 2);

    // Check transfer function complexity
    const transfer_fn = contract.body[8].Function;
    try testing.expect(std.mem.eql(u8, transfer_fn.name, "transfer"));
    try testing.expect(transfer_fn.requires_clauses.len == 2);
    try testing.expect(transfer_fn.ensures_clauses.len == 2);
    try testing.expect(transfer_fn.body.statements.len == 4); // if, assignment, assignment, log, return
}

// =============================================================================
// AST UTILITY TESTS
// =============================================================================

test "AST node counting utility" {
    var helper = AstTestHelper.init(testing.allocator);

    const source =
        \\struct Point { x: u32, y: u32, }
        \\enum Status { Active, Inactive }
        \\contract MultiNodeTest {
        \\    storage var data: Point;
        \\    pub fn init() { }
        \\    pub fn test() { }
        \\}
    ;

    const nodes = try helper.expectParseSuccess(source);
    defer helper.cleanup(nodes);

    // Test node counting
    try testing.expect(helper.countNodes(nodes, .StructDecl) == 1);
    try testing.expect(helper.countNodes(nodes, .EnumDecl) == 1);
    try testing.expect(helper.countNodes(nodes, .Contract) == 1);

    const contract = nodes[2].Contract;
    try testing.expect(helper.countNodes(contract.body, .VariableDecl) == 1);
    try testing.expect(helper.countNodes(contract.body, .Function) == 2);
}

test "AST memory cleanup" {
    var helper = AstTestHelper.init(testing.allocator);

    const source = "contract CleanupTest { pub fn init() { } }";

    // Test that multiple parse/cleanup cycles don't leak memory
    var i: u32 = 0;
    while (i < 10) : (i += 1) {
        const nodes = try helper.expectParseSuccess(source);
        helper.cleanup(nodes);
    }

    // If we reach here without error, cleanup is working
    try testing.expect(true);
}
