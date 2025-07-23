const std = @import("std");
const testing = std.testing;
const ast = @import("ast.zig");
const semantics = @import("semantics.zig");
const typer = @import("typer.zig");

// Test helpers and utilities
const TestHelper = struct {
    allocator: std.mem.Allocator,
    analyzer: semantics.SemanticAnalyzer,

    fn init(allocator: std.mem.Allocator) TestHelper {
        return TestHelper{
            .allocator = allocator,
            .analyzer = semantics.SemanticAnalyzer.init(allocator),
        };
    }

    fn deinit(self: *TestHelper) void {
        self.analyzer.deinit();
    }

    fn createTestContract(self: *TestHelper, name: []const u8) *ast.ContractNode {
        const contract = self.allocator.create(ast.ContractNode) catch unreachable;
        contract.* = ast.ContractNode{
            .name = name,
            .body = &[_]ast.AstNode{},
            .span = ast.SourceSpan{ .line = 1, .column = 1, .length = @intCast(name.len) },
        };
        return contract;
    }

    fn createTestFunction(self: *TestHelper, name: []const u8, pub_: bool) *ast.FunctionNode {
        const function = self.allocator.create(ast.FunctionNode) catch unreachable;
        function.* = ast.FunctionNode{
            .name = name,
            .params = &[_]ast.ParameterNode{},
            .return_type = null,
            .body = ast.BlockNode{
                .statements = &[_]ast.StmtNode{},
                .span = ast.SourceSpan{ .line = 1, .column = 1, .length = 1 },
            },
            .requires_clauses = &[_]ast.ExprNode{},
            .ensures_clauses = &[_]ast.ExprNode{},
            .pub_ = pub_,
            .span = ast.SourceSpan{ .line = 1, .column = 1, .length = @intCast(name.len) },
        };
        return function;
    }

    fn createTestVariable(self: *TestHelper, name: []const u8, region: ast.MemoryRegion, mutable: bool) *ast.VariableDeclNode {
        const variable = self.allocator.create(ast.VariableDeclNode) catch unreachable;
        variable.* = ast.VariableDeclNode{
            .name = name,
            .typ = ast.TypeRef{ .Named = "u32" },
            .value = null,
            .region = region,
            .mutable = mutable,
            .span = ast.SourceSpan{ .line = 1, .column = 1, .length = @intCast(name.len) },
        };
        return variable;
    }

    fn expectDiagnosticCount(self: *TestHelper, expected_errors: u32, expected_warnings: u32) !void {
        try testing.expect(self.analyzer.analysis_state.error_count == expected_errors);
        try testing.expect(self.analyzer.analysis_state.warning_count == expected_warnings);
    }

    fn expectNoDiagnostics(self: *TestHelper) !void {
        try self.expectDiagnosticCount(0, 0);
    }
};

// =============================================================================
// BASIC FUNCTIONALITY TESTS
// =============================================================================

test "semantic analyzer initialization" {
    var analyzer = semantics.SemanticAnalyzer.init(testing.allocator);
    defer analyzer.deinit();

    // Test initial state
    try testing.expect(analyzer.current_function == null);
    try testing.expect(analyzer.in_loop == false);
    try testing.expect(analyzer.in_assignment_target == false);
    try testing.expect(analyzer.in_error_propagation_context == false);
    try testing.expect(analyzer.current_function_returns_error_union == false);
    try testing.expect(analyzer.error_recovery_mode == false);
    try testing.expect(analyzer.analysis_state.phase == .PreInitialization);
    try testing.expect(analyzer.analysis_state.error_count == 0);
    try testing.expect(analyzer.analysis_state.warning_count == 0);
}

test "analysis state tracking" {
    var analyzer = semantics.SemanticAnalyzer.init(testing.allocator);
    defer analyzer.deinit();

    // Test analysis state progression
    analyzer.analysis_state.phase = .TypeChecking;
    try testing.expect(analyzer.analysis_state.phase == .TypeChecking);

    analyzer.analysis_state.phase = .SemanticAnalysis;
    try testing.expect(analyzer.analysis_state.phase == .SemanticAnalysis);

    analyzer.analysis_state.phase = .Validation;
    try testing.expect(analyzer.analysis_state.phase == .Validation);
}

test "validation coverage tracking" {
    var coverage = semantics.ValidationCoverage.init(testing.allocator);
    defer coverage.deinit();

    // Test initial state
    try testing.expect(coverage.validation_stats.nodes_analyzed == 0);
    try testing.expect(coverage.validation_stats.errors_found == 0);
    try testing.expect(coverage.validation_stats.warnings_generated == 0);
    try testing.expect(coverage.validation_stats.recovery_attempts == 0);

    // Test stats updates
    coverage.validation_stats.nodes_analyzed += 1;
    coverage.validation_stats.errors_found += 1;
    try testing.expect(coverage.validation_stats.nodes_analyzed == 1);
    try testing.expect(coverage.validation_stats.errors_found == 1);
}

// =============================================================================
// MEMORY SAFETY TESTS
// =============================================================================

test "pointer validation" {
    var helper = TestHelper.init(testing.allocator);
    defer helper.deinit();

    // Test null pointer detection
    try testing.expect(!helper.analyzer.isValidNodePointer(null));

    // Test valid pointer (using a real allocated node)
    const test_node = helper.allocator.create(ast.AstNode) catch unreachable;
    defer helper.allocator.destroy(test_node);
    test_node.* = ast.AstNode{ .Expression = ast.ExprNode{ .Literal = ast.LiteralExpr{ .Integer = ast.IntegerLiteral{
        .value = "42",
        .span = ast.SourceSpan{ .line = 1, .column = 1, .length = 2 },
    } } } };

    try testing.expect(helper.analyzer.isValidNodePointer(test_node));
}

test "string validation" {
    var helper = TestHelper.init(testing.allocator);
    defer helper.deinit();

    // Test valid string
    const valid_str = "test_string";
    try testing.expect(helper.analyzer.isValidString(valid_str));

    // Test empty string (should be valid)
    const empty_str = "";
    try testing.expect(helper.analyzer.isValidString(empty_str));
}

test "span validation" {
    var helper = TestHelper.init(testing.allocator);
    defer helper.deinit();

    // Test normal span
    const normal_span = ast.SourceSpan{ .line = 10, .column = 5, .length = 20 };
    const validated_span = helper.analyzer.validateSpan(normal_span);
    try testing.expect(validated_span.line == 10);
    try testing.expect(validated_span.column == 5);
    try testing.expect(validated_span.length == 20);

    // Test extreme values get clamped
    const extreme_span = ast.SourceSpan{ .line = 2000000, .column = 50000, .length = 50000 };
    const clamped_span = helper.analyzer.validateSpan(extreme_span);
    try testing.expect(clamped_span.line == 0);
    try testing.expect(clamped_span.column == 0);
    try testing.expect(clamped_span.length == 0);
}

// =============================================================================
// CONTRACT ANALYSIS TESTS
// =============================================================================

test "contract with init function validation" {
    var helper = TestHelper.init(testing.allocator);
    defer helper.deinit();

    // Create contract with public init function
    const contract = helper.createTestContract("TestContract");
    const init_func = helper.createTestFunction("init", true);

    // Create contract body with init function
    const contract_body = [_]ast.AstNode{
        ast.AstNode{ .Function = init_func.* },
    };
    contract.body = &contract_body;

    // Analyze contract
    const contract_node = ast.AstNode{ .Contract = contract.* };
    try helper.analyzer.safeAnalyzeNode(@constCast(&contract_node));

    // Should have no errors for valid contract
    try helper.expectNoDiagnostics();
}

test "contract missing init function" {
    var helper = TestHelper.init(testing.allocator);
    defer helper.deinit();

    // Create contract without init function
    const contract = helper.createTestContract("TestContract");
    const regular_func = helper.createTestFunction("regularFunction", true);

    const contract_body = [_]ast.AstNode{
        ast.AstNode{ .Function = regular_func.* },
    };
    contract.body = &contract_body;

    const contract_node = ast.AstNode{ .Contract = contract.* };
    try helper.analyzer.safeAnalyzeNode(@constCast(&contract_node));

    // Should have error for missing init function
    try testing.expect(helper.analyzer.analysis_state.error_count >= 1);
}

test "contract with non-public init function" {
    var helper = TestHelper.init(testing.allocator);
    defer helper.deinit();

    // Create contract with private init function
    const contract = helper.createTestContract("TestContract");
    const init_func = helper.createTestFunction("init", false); // non-public

    const contract_body = [_]ast.AstNode{
        ast.AstNode{ .Function = init_func.* },
    };
    contract.body = &contract_body;

    const contract_node = ast.AstNode{ .Contract = contract.* };
    try helper.analyzer.safeAnalyzeNode(@constCast(&contract_node));

    // Should have warning for non-public init function
    try testing.expect(helper.analyzer.analysis_state.warning_count >= 1);
}

test "duplicate init functions" {
    var helper = TestHelper.init(testing.allocator);
    defer helper.deinit();

    // Create contract with two init functions
    const contract = helper.createTestContract("TestContract");
    const init_func1 = helper.createTestFunction("init", true);
    const init_func2 = helper.createTestFunction("init", true);

    const contract_body = [_]ast.AstNode{
        ast.AstNode{ .Function = init_func1.* },
        ast.AstNode{ .Function = init_func2.* },
    };
    contract.body = &contract_body;

    const contract_node = ast.AstNode{ .Contract = contract.* };
    try helper.analyzer.safeAnalyzeNode(@constCast(&contract_node));

    // Should have error for duplicate init functions
    try testing.expect(helper.analyzer.analysis_state.error_count >= 1);
}

// =============================================================================
// VARIABLE DECLARATION TESTS
// =============================================================================

test "storage variable validation" {
    var helper = TestHelper.init(testing.allocator);
    defer helper.deinit();

    // Test valid storage variable
    const storage_var = helper.createTestVariable("storageVar", .Storage, true);
    const var_node = ast.AstNode{ .VariableDecl = storage_var.* };

    // Should be valid at contract level
    helper.analyzer.current_function = null; // Simulate contract level
    try helper.analyzer.safeAnalyzeNode(@constCast(&var_node));
    try helper.expectNoDiagnostics();

    // Should be invalid in function
    helper.analyzer.current_function = "testFunction";
    helper.analyzer.analysis_state.error_count = 0; // Reset count
    try helper.analyzer.safeAnalyzeNode(@constCast(&var_node));
    try testing.expect(helper.analyzer.analysis_state.error_count >= 1);
}

test "immutable variable validation" {
    var helper = TestHelper.init(testing.allocator);
    defer helper.deinit();

    // Test immutable variable
    const immutable_var = helper.createTestVariable("immutableVar", .Immutable, false);
    const var_node = ast.AstNode{ .VariableDecl = immutable_var.* };

    // Should be valid at contract level
    helper.analyzer.current_function = null;
    try helper.analyzer.safeAnalyzeNode(@constCast(&var_node));
    try helper.expectNoDiagnostics();

    // Should be invalid in function
    helper.analyzer.current_function = "testFunction";
    helper.analyzer.analysis_state.error_count = 0;
    try helper.analyzer.safeAnalyzeNode(@constCast(&var_node));
    try testing.expect(helper.analyzer.analysis_state.error_count >= 1);
}

test "const variable validation" {
    var helper = TestHelper.init(testing.allocator);
    defer helper.deinit();

    // Test const variable without initializer (should error)
    const const_var = helper.createTestVariable("constVar", .Const, false);
    const var_node = ast.AstNode{ .VariableDecl = const_var.* };

    try helper.analyzer.safeAnalyzeNode(@constCast(&var_node));
    try testing.expect(helper.analyzer.analysis_state.error_count >= 1);

    // Test const variable with initializer (would be valid - tested in integration)
}

test "memory region semantic validation" {
    var helper = TestHelper.init(testing.allocator);
    defer helper.deinit();

    // Test all memory regions
    const regions = [_]ast.MemoryRegion{ .Storage, .Immutable, .Stack, .Memory, .Const, .TStore };

    for (regions) |region| {
        const var_decl = helper.createTestVariable("testVar", region, true);
        const var_node = ast.AstNode{ .VariableDecl = var_decl.* };

        // Reset error counts
        helper.analyzer.analysis_state.error_count = 0;
        helper.analyzer.analysis_state.warning_count = 0;

        // Test at contract level
        helper.analyzer.current_function = null;
        try helper.analyzer.safeAnalyzeNode(@constCast(&var_node));

        // Storage and Immutable should be fine at contract level
        // Others should also be fine for this basic test
        switch (region) {
            .Storage, .Immutable => {
                // Should have no errors at contract level
                try testing.expect(helper.analyzer.analysis_state.error_count == 0);
            },
            else => {
                // Other regions are generally OK
            },
        }
    }
}

// =============================================================================
// FUNCTION ANALYSIS TESTS
// =============================================================================

test "function context tracking" {
    var helper = TestHelper.init(testing.allocator);
    defer helper.deinit();

    // Test function context is set correctly
    try testing.expect(helper.analyzer.current_function == null);

    helper.analyzer.current_function = "testFunction";
    try testing.expect(std.mem.eql(u8, helper.analyzer.current_function.?, "testFunction"));

    helper.analyzer.current_function = null;
    try testing.expect(helper.analyzer.current_function == null);
}

test "loop context tracking" {
    var helper = TestHelper.init(testing.allocator);
    defer helper.deinit();

    // Test loop context tracking
    try testing.expect(helper.analyzer.in_loop == false);

    helper.analyzer.in_loop = true;
    try testing.expect(helper.analyzer.in_loop == true);

    helper.analyzer.in_loop = false;
    try testing.expect(helper.analyzer.in_loop == false);
}

test "assignment target context tracking" {
    var helper = TestHelper.init(testing.allocator);
    defer helper.deinit();

    // Test assignment target context
    try testing.expect(helper.analyzer.in_assignment_target == false);

    helper.analyzer.in_assignment_target = true;
    try testing.expect(helper.analyzer.in_assignment_target == true);

    helper.analyzer.in_assignment_target = false;
    try testing.expect(helper.analyzer.in_assignment_target == false);
}

test "error propagation context tracking" {
    var helper = TestHelper.init(testing.allocator);
    defer helper.deinit();

    // Test error propagation context
    try testing.expect(helper.analyzer.in_error_propagation_context == false);
    try testing.expect(helper.analyzer.current_function_returns_error_union == false);

    helper.analyzer.in_error_propagation_context = true;
    helper.analyzer.current_function_returns_error_union = true;
    try testing.expect(helper.analyzer.in_error_propagation_context == true);
    try testing.expect(helper.analyzer.current_function_returns_error_union == true);
}

// =============================================================================
// EXPRESSION ANALYSIS TESTS
// =============================================================================

test "identifier analysis" {
    var helper = TestHelper.init(testing.allocator);
    defer helper.deinit();

    // Create identifier expression
    const ident = ast.IdentifierExpr{
        .name = "testVar",
        .span = ast.SourceSpan{ .line = 1, .column = 1, .length = 7 },
    };
    const expr = ast.ExprNode{ .Identifier = ident };

    // Analyze identifier
    try helper.analyzer.analyzeExpression(@constCast(&expr));

    // Basic identifier analysis should not error by itself
    // (undeclared identifier errors are handled by type checker)
}

test "literal analysis" {
    var helper = TestHelper.init(testing.allocator);
    defer helper.deinit();

    // Test integer literal
    const int_literal = ast.IntegerLiteral{
        .value = "42",
        .span = ast.SourceSpan{ .line = 1, .column = 1, .length = 2 },
    };
    const int_expr = ast.ExprNode{ .Literal = ast.LiteralExpr{ .Integer = int_literal } };

    try helper.analyzer.analyzeExpression(@constCast(&int_expr));
    try helper.expectNoDiagnostics();

    // Test string literal
    const str_literal = ast.StringLiteral{
        .value = "test string",
        .span = ast.SourceSpan{ .line = 1, .column = 1, .length = 11 },
    };
    const str_expr = ast.ExprNode{ .Literal = ast.LiteralExpr{ .String = str_literal } };

    try helper.analyzer.analyzeExpression(@constCast(&str_expr));
    try helper.expectNoDiagnostics();

    // Test boolean literal
    const bool_literal = ast.BoolLiteral{
        .value = true,
        .span = ast.SourceSpan{ .line = 1, .column = 1, .length = 4 },
    };
    const bool_expr = ast.ExprNode{ .Literal = ast.LiteralExpr{ .Bool = bool_literal } };

    try helper.analyzer.analyzeExpression(@constCast(&bool_expr));
    try helper.expectNoDiagnostics();
}

test "binary expression analysis" {
    var helper = TestHelper.init(testing.allocator);
    defer helper.deinit();

    // Create binary expression: 10 + 5
    const left_literal = ast.ExprNode{ .Literal = ast.LiteralExpr{ .Integer = ast.IntegerLiteral{
        .value = "10",
        .span = ast.SourceSpan{ .line = 1, .column = 1, .length = 2 },
    } } };

    const right_literal = ast.ExprNode{ .Literal = ast.LiteralExpr{ .Integer = ast.IntegerLiteral{
        .value = "5",
        .span = ast.SourceSpan{ .line = 1, .column = 6, .length = 1 },
    } } };

    const left_ptr = helper.allocator.create(ast.ExprNode) catch unreachable;
    const right_ptr = helper.allocator.create(ast.ExprNode) catch unreachable;
    defer helper.allocator.destroy(left_ptr);
    defer helper.allocator.destroy(right_ptr);

    left_ptr.* = left_literal;
    right_ptr.* = right_literal;

    const binary_expr = ast.BinaryExpr{
        .lhs = left_ptr,
        .operator = .Plus,
        .rhs = right_ptr,
        .span = ast.SourceSpan{ .line = 1, .column = 1, .length = 6 },
    };

    const expr = ast.ExprNode{ .Binary = binary_expr };
    try helper.analyzer.analyzeExpression(@constCast(&expr));
    try helper.expectNoDiagnostics();
}

test "division by zero detection" {
    var helper = TestHelper.init(testing.allocator);
    defer helper.deinit();

    // Create division by zero: 10 / 0
    const left_literal = ast.ExprNode{ .Literal = ast.LiteralExpr{ .Integer = ast.IntegerLiteral{
        .value = "10",
        .span = ast.SourceSpan{ .line = 1, .column = 1, .length = 2 },
    } } };

    const right_literal = ast.ExprNode{ .Literal = ast.LiteralExpr{ .Integer = ast.IntegerLiteral{
        .value = "0",
        .span = ast.SourceSpan{ .line = 1, .column = 6, .length = 1 },
    } } };

    const left_ptr = helper.allocator.create(ast.ExprNode) catch unreachable;
    const right_ptr = helper.allocator.create(ast.ExprNode) catch unreachable;
    defer helper.allocator.destroy(left_ptr);
    defer helper.allocator.destroy(right_ptr);

    left_ptr.* = left_literal;
    right_ptr.* = right_literal;

    const binary_expr = ast.BinaryExpr{
        .lhs = left_ptr,
        .operator = .Slash,
        .rhs = right_ptr,
        .span = ast.SourceSpan{ .line = 1, .column = 1, .length = 6 },
    };

    const expr = ast.ExprNode{ .Binary = binary_expr };
    try helper.analyzer.analyzeExpression(@constCast(&expr));

    // Should detect potential division by zero
    try testing.expect(helper.analyzer.analysis_state.error_count >= 1);
}

// =============================================================================
// IMMUTABLE VARIABLE TRACKING TESTS
// =============================================================================

test "immutable variable initialization tracking" {
    var helper = TestHelper.init(testing.allocator);
    defer helper.deinit();

    // Add an immutable variable to tracking
    const immutable_info = semantics.SemanticAnalyzer.ImmutableVarInfo{
        .name = "testVar",
        .declared_span = ast.SourceSpan{ .line = 1, .column = 1, .length = 7 },
        .initialized = false,
        .init_span = null,
    };

    try helper.analyzer.immutable_variables.put("testVar", immutable_info);

    // Verify it was added
    try testing.expect(helper.analyzer.immutable_variables.count() == 1);
    const retrieved = helper.analyzer.immutable_variables.get("testVar");
    try testing.expect(retrieved != null);
    try testing.expect(retrieved.?.initialized == false);
    try testing.expect(retrieved.?.init_span == null);

    // Mark as initialized
    var mutable_info = helper.analyzer.immutable_variables.getPtr("testVar").?;
    mutable_info.initialized = true;
    mutable_info.init_span = ast.SourceSpan{ .line = 2, .column = 1, .length = 7 };

    // Verify update
    const updated = helper.analyzer.immutable_variables.get("testVar");
    try testing.expect(updated.?.initialized == true);
    try testing.expect(updated.?.init_span != null);
}

test "constructor context tracking" {
    var helper = TestHelper.init(testing.allocator);
    defer helper.deinit();

    // Test constructor context
    try testing.expect(helper.analyzer.in_constructor == false);

    helper.analyzer.in_constructor = true;
    try testing.expect(helper.analyzer.in_constructor == true);

    helper.analyzer.in_constructor = false;
    try testing.expect(helper.analyzer.in_constructor == false);
}

// =============================================================================
// ERROR HANDLING TESTS
// =============================================================================

test "error diagnostic creation" {
    var helper = TestHelper.init(testing.allocator);
    defer helper.deinit();

    const test_span = ast.SourceSpan{ .line = 10, .column = 5, .length = 20 };

    // Test static error message
    try helper.analyzer.addErrorStatic("Test error message", test_span);
    try testing.expect(helper.analyzer.analysis_state.error_count == 1);
    try testing.expect(helper.analyzer.diagnostics.items.len == 1);

    const diagnostic = helper.analyzer.diagnostics.items[0];
    try testing.expect(diagnostic.severity == .Error);
    try testing.expect(std.mem.eql(u8, diagnostic.message, "Test error message"));
}

test "warning diagnostic creation" {
    var helper = TestHelper.init(testing.allocator);
    defer helper.deinit();

    const test_span = ast.SourceSpan{ .line = 10, .column = 5, .length = 20 };

    // Test static warning message
    try helper.analyzer.addWarningStatic("Test warning message", test_span);
    try testing.expect(helper.analyzer.analysis_state.warning_count == 1);
    try testing.expect(helper.analyzer.diagnostics.items.len == 1);

    const diagnostic = helper.analyzer.diagnostics.items[0];
    try testing.expect(diagnostic.severity == .Warning);
    try testing.expect(std.mem.eql(u8, diagnostic.message, "Test warning message"));
}

test "info diagnostic creation" {
    var helper = TestHelper.init(testing.allocator);
    defer helper.deinit();

    const test_span = ast.SourceSpan{ .line = 10, .column = 5, .length = 20 };

    // Test static info message
    try helper.analyzer.addInfoStatic("Test info message", test_span);
    try testing.expect(helper.analyzer.diagnostics.items.len == 1);

    const diagnostic = helper.analyzer.diagnostics.items[0];
    try testing.expect(diagnostic.severity == .Info);
    try testing.expect(std.mem.eql(u8, diagnostic.message, "Test info message"));
}

test "diagnostic context information" {
    var helper = TestHelper.init(testing.allocator);
    defer helper.deinit();

    // Set analysis context
    helper.analyzer.analysis_state.current_node_type = .Contract;
    helper.analyzer.analysis_state.phase = .SemanticAnalysis;
    helper.analyzer.error_recovery_mode = true;

    const test_span = ast.SourceSpan{ .line = 10, .column = 5, .length = 20 };
    try helper.analyzer.addErrorStatic("Test error with context", test_span);

    const diagnostic = helper.analyzer.diagnostics.items[0];
    try testing.expect(diagnostic.context != null);
    try testing.expect(diagnostic.context.?.node_type == .Contract);
    try testing.expect(diagnostic.context.?.analysis_phase == .SemanticAnalysis);
    try testing.expect(diagnostic.context.?.recovery_attempted == true);
}

// =============================================================================
// BUILTIN FUNCTION TESTS
// =============================================================================

test "builtin function recognition" {
    var helper = TestHelper.init(testing.allocator);
    defer helper.deinit();

    // Test formal verification builtins
    try testing.expect(helper.analyzer.isBuiltinFunction("requires"));
    try testing.expect(helper.analyzer.isBuiltinFunction("ensures"));
    try testing.expect(helper.analyzer.isBuiltinFunction("invariant"));
    try testing.expect(helper.analyzer.isBuiltinFunction("old"));
    try testing.expect(helper.analyzer.isBuiltinFunction("log"));

    // Test division builtins
    try testing.expect(helper.analyzer.isBuiltinFunction("@divmod"));
    try testing.expect(helper.analyzer.isBuiltinFunction("@divTrunc"));
    try testing.expect(helper.analyzer.isBuiltinFunction("@divFloor"));
    try testing.expect(helper.analyzer.isBuiltinFunction("@divCeil"));
    try testing.expect(helper.analyzer.isBuiltinFunction("@divExact"));

    // Test non-builtin functions
    try testing.expect(!helper.analyzer.isBuiltinFunction("customFunction"));
    try testing.expect(!helper.analyzer.isBuiltinFunction("println"));
    try testing.expect(!helper.analyzer.isBuiltinFunction("malloc"));
}

// =============================================================================
// TYPE SIZE CALCULATION TESTS
// =============================================================================

test "type size calculation" {
    var helper = TestHelper.init(testing.allocator);
    defer helper.deinit();

    // Test primitive type sizes
    try testing.expect(helper.analyzer.getTypeSize(typer.OraType.Bool) == 1);
    try testing.expect(helper.analyzer.getTypeSize(typer.OraType.U8) == 1);
    try testing.expect(helper.analyzer.getTypeSize(typer.OraType.U16) == 2);
    try testing.expect(helper.analyzer.getTypeSize(typer.OraType.U32) == 4);
    try testing.expect(helper.analyzer.getTypeSize(typer.OraType.U64) == 8);
    try testing.expect(helper.analyzer.getTypeSize(typer.OraType.U128) == 16);
    try testing.expect(helper.analyzer.getTypeSize(typer.OraType.U256) == 32);
    try testing.expect(helper.analyzer.getTypeSize(typer.OraType.Address) == 20);

    // Test complex types
    try testing.expect(helper.analyzer.getTypeSize(typer.OraType.String) == 32);
    try testing.expect(helper.analyzer.getTypeSize(typer.OraType.Bytes) == 32);
    try testing.expect(helper.analyzer.getTypeSize(typer.OraType.Slice) == 64);
    try testing.expect(helper.analyzer.getTypeSize(typer.OraType.Mapping) == 32);
    try testing.expect(helper.analyzer.getTypeSize(typer.OraType.DoubleMap) == 32);

    // Test special types
    try testing.expect(helper.analyzer.getTypeSize(typer.OraType.Void) == 0);
    try testing.expect(helper.analyzer.getTypeSize(typer.OraType.Unknown) == 0);
}

// =============================================================================
// RESERVED NAME TESTS
// =============================================================================

test "reserved field name detection" {
    var helper = TestHelper.init(testing.allocator);
    defer helper.deinit();

    // Test reserved names
    try testing.expect(helper.analyzer.isReservedFieldName("init"));
    try testing.expect(helper.analyzer.isReservedFieldName("deinit"));
    try testing.expect(helper.analyzer.isReservedFieldName("clone"));
    try testing.expect(helper.analyzer.isReservedFieldName("copy"));
    try testing.expect(helper.analyzer.isReservedFieldName("serialize"));
    try testing.expect(helper.analyzer.isReservedFieldName("hash"));
    try testing.expect(helper.analyzer.isReservedFieldName("equals"));

    // Test non-reserved names
    try testing.expect(!helper.analyzer.isReservedFieldName("customField"));
    try testing.expect(!helper.analyzer.isReservedFieldName("myData"));
    try testing.expect(!helper.analyzer.isReservedFieldName("value"));
}

// =============================================================================
// IDENTIFIER VALIDATION TESTS
// =============================================================================

test "identifier validation" {
    var helper = TestHelper.init(testing.allocator);
    defer helper.deinit();

    // Test valid identifiers
    try testing.expect(helper.analyzer.isValidIdentifier("validName"));
    try testing.expect(helper.analyzer.isValidIdentifier("_underscore"));
    try testing.expect(helper.analyzer.isValidIdentifier("camelCase"));
    try testing.expect(helper.analyzer.isValidIdentifier("snake_case"));
    try testing.expect(helper.analyzer.isValidIdentifier("with123numbers"));
    try testing.expect(helper.analyzer.isValidIdentifier("_"));

    // Test invalid identifiers
    try testing.expect(!helper.analyzer.isValidIdentifier(""));
    try testing.expect(!helper.analyzer.isValidIdentifier("123startWithNumber"));
    try testing.expect(!helper.analyzer.isValidIdentifier("has-dash"));
    try testing.expect(!helper.analyzer.isValidIdentifier("has spaces"));
    try testing.expect(!helper.analyzer.isValidIdentifier("has.dot"));
    try testing.expect(!helper.analyzer.isValidIdentifier("has@symbol"));
}

// =============================================================================
// MODULE FIELD VALIDATION TESTS
// =============================================================================

test "field validation for standard modules" {
    var helper = TestHelper.init(testing.allocator);
    defer helper.deinit();

    // Test transaction fields
    const transaction_fields = [_][]const u8{ "sender", "value", "origin", "gasprice" };
    for (transaction_fields) |field| {
        try testing.expect(helper.analyzer.isFieldValid(field, &transaction_fields));
    }
    try testing.expect(!helper.analyzer.isFieldValid("invalid", &transaction_fields));

    // Test block fields
    const block_fields = [_][]const u8{ "timestamp", "number", "coinbase", "difficulty", "gaslimit" };
    for (block_fields) |field| {
        try testing.expect(helper.analyzer.isFieldValid(field, &block_fields));
    }
    try testing.expect(!helper.analyzer.isFieldValid("invalid", &block_fields));

    // Test constants fields
    const constant_fields = [_][]const u8{ "ZERO_ADDRESS", "MAX_UINT256", "MIN_UINT256" };
    for (constant_fields) |field| {
        try testing.expect(helper.analyzer.isFieldValid(field, &constant_fields));
    }
    try testing.expect(!helper.analyzer.isFieldValid("invalid", &constant_fields));
}

// =============================================================================
// ERROR RECOVERY TESTS
// =============================================================================

test "error recovery mode" {
    var helper = TestHelper.init(testing.allocator);
    defer helper.deinit();

    // Test error recovery is disabled initially
    try testing.expect(helper.analyzer.error_recovery_mode == false);

    // Enable error recovery
    helper.analyzer.error_recovery_mode = true;
    try testing.expect(helper.analyzer.error_recovery_mode == true);

    // Test that error recovery affects diagnostic context
    helper.analyzer.analysis_state.current_node_type = .Function;
    const test_span = ast.SourceSpan{ .line = 1, .column = 1, .length = 10 };
    try helper.analyzer.addErrorStatic("Test error in recovery mode", test_span);

    const diagnostic = helper.analyzer.diagnostics.items[0];
    try testing.expect(diagnostic.context != null);
    try testing.expect(diagnostic.context.?.recovery_attempted == true);
}

test "safe node analysis with invalid pointer" {
    var helper = TestHelper.init(testing.allocator);
    defer helper.deinit();

    // Try to analyze null pointer (should be handled gracefully)
    const result = helper.analyzer.safeAnalyzeNode(null);
    try testing.expectError(semantics.SemanticError.PointerValidationFailed, result);

    // Should have recorded the error in diagnostics
    try testing.expect(helper.analyzer.analysis_state.error_count >= 1);
    try testing.expect(helper.analyzer.validation_coverage.validation_stats.recovery_attempts >= 1);
}

// =============================================================================
// INTEGRATION TESTS
// =============================================================================

test "complete contract analysis workflow" {
    var helper = TestHelper.init(testing.allocator);
    defer helper.deinit();

    // Create a complete valid contract
    const init_func = helper.createTestFunction("init", true);
    const storage_var = helper.createTestVariable("myStorage", .Storage, true);
    const immutable_var = helper.createTestVariable("myImmutable", .Immutable, false);

    const contract_body = [_]ast.AstNode{
        ast.AstNode{ .VariableDecl = storage_var.* },
        ast.AstNode{ .VariableDecl = immutable_var.* },
        ast.AstNode{ .Function = init_func.* },
    };

    const contract = helper.createTestContract("CompleteContract");
    contract.body = &contract_body;
    const contract_node = ast.AstNode{ .Contract = contract.* };

    // Analyze the complete contract
    try helper.analyzer.safeAnalyzeNode(@constCast(&contract_node));

    // Should complete without critical errors
    // (may have warnings about uninitialized immutable variables)
}

test "analyze function with context" {
    var helper = TestHelper.init(testing.allocator);
    defer helper.deinit();

    // Create function with proper context
    const function = helper.createTestFunction("testFunction", true);
    const function_node = ast.AstNode{ .Function = function.* };

    // Ensure we're not in a function context initially
    try testing.expect(helper.analyzer.current_function == null);

    // Analyze the function
    try helper.analyzer.safeAnalyzeNode(@constCast(&function_node));

    // Function analysis should complete
    // Context should be reset after analysis
    try testing.expect(helper.analyzer.current_function == null);
}

// =============================================================================
// PERFORMANCE AND STRESS TESTS
// =============================================================================

test "large number of diagnostics" {
    var helper = TestHelper.init(testing.allocator);
    defer helper.deinit();

    const test_span = ast.SourceSpan{ .line = 1, .column = 1, .length = 10 };

    // Add many diagnostics to test memory management
    var i: u32 = 0;
    while (i < 100) : (i += 1) {
        try helper.analyzer.addErrorStatic("Test error", test_span);
        try helper.analyzer.addWarningStatic("Test warning", test_span);
        try helper.analyzer.addInfoStatic("Test info", test_span);
    }

    try testing.expect(helper.analyzer.diagnostics.items.len == 300);
    try testing.expect(helper.analyzer.analysis_state.error_count == 100);
    try testing.expect(helper.analyzer.analysis_state.warning_count == 100);
}

test "validation statistics tracking" {
    var helper = TestHelper.init(testing.allocator);
    defer helper.deinit();

    // Verify initial statistics
    try testing.expect(helper.analyzer.validation_coverage.validation_stats.nodes_analyzed == 0);
    try testing.expect(helper.analyzer.validation_coverage.validation_stats.errors_found == 0);
    try testing.expect(helper.analyzer.validation_coverage.validation_stats.warnings_generated == 0);

    // Simulate some analysis work
    helper.analyzer.validation_coverage.validation_stats.nodes_analyzed += 10;
    helper.analyzer.validation_coverage.validation_stats.errors_found += 2;
    helper.analyzer.validation_coverage.validation_stats.warnings_generated += 3;

    try testing.expect(helper.analyzer.validation_coverage.validation_stats.nodes_analyzed == 10);
    try testing.expect(helper.analyzer.validation_coverage.validation_stats.errors_found == 2);
    try testing.expect(helper.analyzer.validation_coverage.validation_stats.warnings_generated == 3);
}
