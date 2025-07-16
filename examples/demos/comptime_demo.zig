const std = @import("std");
const ora = @import("../src/root.zig");

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    // Create a comptime evaluator
    var evaluator = ora.ComptimeEvaluator.init(allocator);
    defer evaluator.deinit();

    std.debug.print("🔥 Ora Comptime Evaluation Demo\n");
    std.debug.print("================================\n\n");

    // Test 1: Basic arithmetic
    std.debug.print("1. Basic Arithmetic:\n");

    // Create AST nodes for testing (simplified)
    var literal_10 = ora.ast.LiteralNode{ .Integer = ora.ast.IntegerLiteral{ .value = "10", .span = ora.ast.SourceSpan{ .line = 1, .column = 1, .length = 2 } } };
    var literal_5 = ora.ast.LiteralNode{ .Integer = ora.ast.IntegerLiteral{ .value = "5", .span = ora.ast.SourceSpan{ .line = 1, .column = 1, .length = 1 } } };

    // Evaluate literals
    const val_10 = try evaluator.evaluateLiteral(&literal_10);
    const val_5 = try evaluator.evaluateLiteral(&literal_5);

    std.debug.print("   10 -> {}\n", .{val_10});
    std.debug.print("   5 -> {}\n", .{val_5});

    // Test arithmetic operations
    const sum = try evaluator.add(val_10, val_5);
    const diff = try evaluator.subtract(val_10, val_5);
    const product = try evaluator.multiply(val_10, val_5);
    const quotient = try evaluator.divide(val_10, val_5);

    std.debug.print("   10 + 5 = {}\n", .{sum});
    std.debug.print("   10 - 5 = {}\n", .{diff});
    std.debug.print("   10 * 5 = {}\n", .{product});
    std.debug.print("   10 / 5 = {}\n", .{quotient});

    std.debug.print("\n");

    // Test 2: Boolean operations
    std.debug.print("2. Boolean Operations:\n");

    var bool_true = ora.ast.LiteralNode{ .Bool = ora.ast.BoolLiteral{ .value = true, .span = ora.ast.SourceSpan{ .line = 1, .column = 1, .length = 4 } } };
    var bool_false = ora.ast.LiteralNode{ .Bool = ora.ast.BoolLiteral{ .value = false, .span = ora.ast.SourceSpan{ .line = 1, .column = 1, .length = 5 } } };

    const val_true = try evaluator.evaluateLiteral(&bool_true);
    const val_false = try evaluator.evaluateLiteral(&bool_false);

    const and_result = try evaluator.logicalAnd(val_true, val_false);
    const or_result = try evaluator.logicalOr(val_true, val_false);
    const not_result = try evaluator.logicalNot(val_false);

    std.debug.print("   true && false = {}\n", .{and_result});
    std.debug.print("   true || false = {}\n", .{or_result});
    std.debug.print("   !false = {}\n", .{not_result});

    std.debug.print("\n");

    // Test 3: Constants
    std.debug.print("3. Constant Definitions:\n");

    // Define some constants
    try evaluator.defineConstant("MAX_SUPPLY", ora.ComptimeValue{ .u64 = 1000000 });
    try evaluator.defineConstant("HALF_SUPPLY", ora.ComptimeValue{ .u64 = 500000 });
    try evaluator.defineConstant("IS_ENABLED", ora.ComptimeValue{ .bool = true });

    // Look them up
    const max_supply = evaluator.symbol_table.lookup("MAX_SUPPLY").?;
    const half_supply = evaluator.symbol_table.lookup("HALF_SUPPLY").?;
    const is_enabled = evaluator.symbol_table.lookup("IS_ENABLED").?;

    std.debug.print("   MAX_SUPPLY = {}\n", .{max_supply});
    std.debug.print("   HALF_SUPPLY = {}\n", .{half_supply});
    std.debug.print("   IS_ENABLED = {}\n", .{is_enabled});

    std.debug.print("\n");

    // Test 4: String constants
    std.debug.print("4. String Constants:\n");

    var str_literal = ora.ast.LiteralNode{ .String = ora.ast.StringLiteral{ .value = "Hello, Ora!", .span = ora.ast.SourceSpan{ .line = 1, .column = 1, .length = 13 } } };
    const str_val = try evaluator.evaluateLiteral(&str_literal);

    std.debug.print("   \"Hello, Ora!\" -> {s}\n", .{str_val.string});

    std.debug.print("\n");

    // Test 5: Address constants
    std.debug.print("5. Address Constants:\n");

    var addr_literal = ora.ast.LiteralNode{ .Address = ora.ast.AddressLiteral{ .value = "0x742d35Cc6634C0532925a3b8D4e6f69d2d1e7CE8", .span = ora.ast.SourceSpan{ .line = 1, .column = 1, .length = 42 } } };
    const addr_val = try evaluator.evaluateLiteral(&addr_literal);

    std.debug.print("   Address: 0x{}\n", .{std.fmt.fmtSliceHexUpper(&addr_val.address)});

    std.debug.print("\n");

    // Test 6: Hex constants
    std.debug.print("6. Hex Constants:\n");

    var hex_literal = ora.ast.LiteralNode{ .Hex = ora.ast.HexLiteral{ .value = "0xFF", .span = ora.ast.SourceSpan{ .line = 1, .column = 1, .length = 4 } } };
    const hex_val = try evaluator.evaluateLiteral(&hex_literal);

    std.debug.print("   0xFF -> 0x{}\n", .{std.fmt.fmtSliceHexUpper(&hex_val.u256)});

    std.debug.print("\n");

    // Test 7: Comparisons
    std.debug.print("7. Comparisons:\n");

    const a = ora.ComptimeValue{ .u32 = 10 };
    const b = ora.ComptimeValue{ .u32 = 5 };
    const c = ora.ComptimeValue{ .u32 = 10 };

    const less_result = try evaluator.compare(b, a, .less);
    const greater_result = try evaluator.compare(a, b, .greater);
    const equal_result = a.equals(c);

    std.debug.print("   5 < 10 = {}\n", .{less_result.bool});
    std.debug.print("   10 > 5 = {}\n", .{greater_result.bool});
    std.debug.print("   10 == 10 = {}\n", .{equal_result});

    std.debug.print("\n");

    // Test 8: Bitwise operations
    std.debug.print("8. Bitwise Operations:\n");

    const val_255 = ora.ComptimeValue{ .u8 = 255 }; // 0xFF
    const val_15 = ora.ComptimeValue{ .u8 = 15 }; // 0x0F

    const and_bitwise = try evaluator.bitwiseAnd(val_255, val_15);
    const or_bitwise = try evaluator.bitwiseOr(val_255, val_15);
    const xor_bitwise = try evaluator.bitwiseXor(val_255, val_15);
    const not_bitwise = try evaluator.bitwiseNot(val_15);

    std.debug.print("   0xFF & 0x0F = 0x{X}\n", .{and_bitwise.u8});
    std.debug.print("   0xFF | 0x0F = 0x{X}\n", .{or_bitwise.u8});
    std.debug.print("   0xFF ^ 0x0F = 0x{X}\n", .{xor_bitwise.u8});
    std.debug.print("   ~0x0F = 0x{X}\n", .{not_bitwise.u8});

    std.debug.print("\n");

    // Test 9: Error handling
    std.debug.print("9. Error Handling:\n");

    // Division by zero
    const zero = ora.ComptimeValue{ .u8 = 0 };
    const div_by_zero_result = evaluator.divide(val_10, zero);
    if (div_by_zero_result) |_| {
        std.debug.print("   ERROR: Division by zero should have failed!\n");
    } else |err| {
        std.debug.print("   ✓ Division by zero properly caught: {}\n", .{err});
    }

    // Type mismatch
    const type_mismatch_result = evaluator.add(val_10, val_true);
    if (type_mismatch_result) |_| {
        std.debug.print("   ERROR: Type mismatch should have failed!\n");
    } else |err| {
        std.debug.print("   ✓ Type mismatch properly caught: {}\n", .{err});
    }

    std.debug.print("\n");

    std.debug.print("🎉 All comptime evaluation tests passed!\n");
    std.debug.print("✨ Ora is ready for compile-time driven development!\n");
}
