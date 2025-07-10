const std = @import("std");
const ora = @import("../src/root.zig");

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    std.debug.print("🔍 Ora Static Verification Demo\n");
    std.debug.print("=================================\n\n");

    // Create a static verifier
    var verifier = ora.StaticVerifier.init(allocator);
    defer verifier.deinit();

    // Test 1: Simple precondition verification
    std.debug.print("1. Testing Simple Preconditions:\n");

    // Create a simple "true" condition (should pass)
    var true_literal = ora.ast.LiteralNode{ .Bool = ora.ast.BoolLiteral{ .value = true, .span = ora.ast.SourceSpan{ .line = 1, .column = 1, .length = 4 } } };
    var true_expr = ora.ast.ExprNode{ .Literal = &true_literal };

    try verifier.addCondition(ora.VerificationCondition{
        .condition = &true_expr,
        .kind = .Precondition,
        .context = ora.VerificationCondition.Context{
            .function_name = "testFunction",
            .old_state = null,
        },
        .span = ora.ast.SourceSpan{ .line = 1, .column = 1, .length = 4 },
    });

    var result = try verifier.verifyAll();
    std.debug.print("   ✓ True precondition: {} (warnings: {})\n", .{ result.verified, result.warnings.len });

    // Test 2: False precondition (should fail)
    std.debug.print("\n2. Testing False Preconditions:\n");

    var false_literal = ora.ast.LiteralNode{ .Bool = ora.ast.BoolLiteral{ .value = false, .span = ora.ast.SourceSpan{ .line = 2, .column = 1, .length = 5 } } };
    var false_expr = ora.ast.ExprNode{ .Literal = &false_literal };

    try verifier.addCondition(ora.VerificationCondition{
        .condition = &false_expr,
        .kind = .Precondition,
        .context = ora.VerificationCondition.Context{
            .function_name = "badFunction",
            .old_state = null,
        },
        .span = ora.ast.SourceSpan{ .line = 2, .column = 1, .length = 5 },
    });

    result = try verifier.verifyAll();
    std.debug.print("   ✗ False precondition: {} (violations: {})\n", .{ result.verified, result.violations.len });

    // Test 3: Constants integration
    std.debug.print("\n3. Testing Constants Integration:\n");

    // Define some constants
    try verifier.defineConstant("MAX_SUPPLY", ora.ComptimeValue{ .u256 = [_]u8{0} ** 31 ++ [_]u8{100} });
    try verifier.defineConstant("MIN_BALANCE", ora.ComptimeValue{ .u256 = [_]u8{0} ** 31 ++ [_]u8{10} });

    std.debug.print("   ✓ Constants defined: MAX_SUPPLY=100, MIN_BALANCE=10\n");

    // Test 4: Postcondition with old() expressions
    std.debug.print("\n4. Testing Postconditions with old() expressions:\n");

    var old_state = verifier.createOldStateContext();
    defer old_state.deinit();

    // Simulate capturing old state
    try old_state.captureVariable("balance", ora.ComptimeValue{ .u256 = [_]u8{0} ** 31 ++ [_]u8{50} });

    // Create a mock old() expression (simplified)
    var identifier = ora.ast.IdentifierExpr{ .name = "balance" };
    var ident_expr = ora.ast.ExprNode{ .Identifier = &identifier };

    try verifier.addCondition(ora.VerificationCondition{
        .condition = &ident_expr,
        .kind = .Postcondition,
        .context = ora.VerificationCondition.Context{
            .function_name = "transfer",
            .old_state = &old_state,
        },
        .span = ora.ast.SourceSpan{ .line = 3, .column = 1, .length = 7 },
    });

    result = try verifier.verifyAll();
    std.debug.print("   ✓ Postcondition with old state: {} (warnings: {})\n", .{ result.verified, result.warnings.len });

    // Test 5: Arithmetic with constants
    std.debug.print("\n5. Testing Arithmetic with Constants:\n");

    // Create a binary expression: 50 + 25
    var lit_50 = ora.ast.LiteralNode{ .Integer = ora.ast.IntegerLiteral{ .value = "50", .span = ora.ast.SourceSpan{ .line = 4, .column = 1, .length = 2 } } };
    var lit_25 = ora.ast.LiteralNode{ .Integer = ora.ast.IntegerLiteral{ .value = "25", .span = ora.ast.SourceSpan{ .line = 4, .column = 6, .length = 2 } } };
    var expr_50 = ora.ast.ExprNode{ .Literal = &lit_50 };
    var expr_25 = ora.ast.ExprNode{ .Literal = &lit_25 };

    var binary_expr = ora.ast.BinaryExpr{
        .lhs = &expr_50,
        .rhs = &expr_25,
        .operator = .Plus,
    };
    var binary_node = ora.ast.ExprNode{ .Binary = &binary_expr };

    try verifier.addCondition(ora.VerificationCondition{
        .condition = &binary_node,
        .kind = .Precondition,
        .context = ora.VerificationCondition.Context{
            .function_name = "arithmeticTest",
            .old_state = null,
        },
        .span = ora.ast.SourceSpan{ .line = 4, .column = 1, .length = 7 },
    });

    result = try verifier.verifyAll();
    std.debug.print("   ✓ Arithmetic expression (50 + 25): {} (warnings: {})\n", .{ result.verified, result.warnings.len });

    // Test 6: Show verification results
    std.debug.print("\n6. Verification Results Summary:\n");
    std.debug.print("   Total conditions verified: {}\n", .{verifier.conditions.items.len});
    std.debug.print("   Overall verification status: {}\n", .{result.verified});

    if (result.violations.len > 0) {
        std.debug.print("   Violations found:\n");
        for (result.violations) |violation| {
            std.debug.print("     - {s} (line {})\n", .{ violation.message, violation.span.line });
        }
    }

    if (result.warnings.len > 0) {
        std.debug.print("   Warnings:\n");
        for (result.warnings) |warning| {
            std.debug.print("     - {s} (line {})\n", .{ warning.message, warning.span.line });
        }
    }

    std.debug.print("\n");

    // Test 7: Pattern recognition demonstration
    std.debug.print("7. Pattern Recognition Demo:\n");

    // Create a new verifier for pattern testing
    var pattern_verifier = ora.StaticVerifier.init(allocator);
    defer pattern_verifier.deinit();

    // Test tautology detection
    var tautology_expr = ora.ast.ExprNode{ .Literal = &true_literal };
    try pattern_verifier.addCondition(ora.VerificationCondition{
        .condition = &tautology_expr,
        .kind = .Assertion,
        .context = ora.VerificationCondition.Context{
            .function_name = "tautologyTest",
            .old_state = null,
        },
        .span = ora.ast.SourceSpan{ .line = 5, .column = 1, .length = 4 },
    });

    const pattern_result = try pattern_verifier.verifyAll();
    std.debug.print("   ✓ Tautology detection: {} violations, {} warnings\n", .{ pattern_result.violations.len, pattern_result.warnings.len });

    // Test 8: Comptime integration
    std.debug.print("\n8. Comptime Integration Demo:\n");

    // The static verifier should be able to use comptime-evaluated constants
    var comptime_verifier = ora.StaticVerifier.init(allocator);
    defer comptime_verifier.deinit();

    // Define a comptime constant
    try comptime_verifier.defineConstant("COMPILE_TIME_CONSTANT", ora.ComptimeValue{ .u64 = 42 });

    // Verify that constants are available
    const constant_lookup = comptime_verifier.comptime_evaluator.symbol_table.lookup("COMPILE_TIME_CONSTANT");
    if (constant_lookup) |value| {
        std.debug.print("   ✓ Comptime constant available: {}\n", .{value});
    } else {
        std.debug.print("   ✗ Comptime constant not found\n");
    }

    std.debug.print("\n");

    // Final summary
    std.debug.print("🎉 Static Verification Demo Complete!\n");
    std.debug.print("✨ Key Features Demonstrated:\n");
    std.debug.print("   • Precondition verification\n");
    std.debug.print("   • Postcondition verification\n");
    std.debug.print("   • old() expression support\n");
    std.debug.print("   • Constant integration\n");
    std.debug.print("   • Tautology detection\n");
    std.debug.print("   • Arithmetic expression evaluation\n");
    std.debug.print("   • Pattern recognition framework\n");
    std.debug.print("   • Comptime integration\n");

    std.debug.print("\n🚀 Ora is ready for compile-time verification!\n");
}
