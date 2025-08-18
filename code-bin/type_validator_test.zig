const std = @import("std");
const testing = std.testing;
const ast = @import("ast.zig");
const typer = @import("typer.zig");
const type_validator = @import("type_validator.zig");
const ast_arena = @import("ast/ast_arena.zig");

// Test helper to create a basic test setup
fn createTestSetup(allocator: std.mem.Allocator) !struct {
    arena: *ast_arena.AstArena,
    typer_instance: *typer.Typer,
    validator: *type_validator.TypeValidator,
} {
    const arena = try allocator.create(ast_arena.AstArena);
    arena.* = ast_arena.AstArena.init(allocator);

    const typer_instance = try allocator.create(typer.Typer);
    typer_instance.* = typer.Typer.init(allocator);

    const validator = try allocator.create(type_validator.TypeValidator);
    validator.* = type_validator.TypeValidator.init(allocator, arena, typer_instance);

    return .{
        .arena = arena,
        .typer_instance = typer_instance,
        .validator = validator,
    };
}

fn cleanupTestSetup(allocator: std.mem.Allocator, setup: anytype) void {
    setup.validator.deinit();
    setup.typer_instance.deinit();
    setup.arena.deinit();
    allocator.destroy(setup.validator);
    allocator.destroy(setup.typer_instance);
    allocator.destroy(setup.arena);
}

test "TypeValidator - explicit type annotation enforcement" {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    const setup = try createTestSetup(allocator);
    defer cleanupTestSetup(allocator, setup);

    // Test integer literal without explicit type annotation (should fail)
    var int_literal = ast.LiteralNode{
        .Integer = .{
            .value = "100",
            .span = .{ .line = 1, .column = 1, .length = 3 },
        },
    };

    var result = type_validator.ValidationResult.init(allocator);
    defer result.deinit();

    const literal_type = try setup.validator.validateLiteral(&int_literal, &result);

    // Should return Unknown type and have an error
    try testing.expect(literal_type == typer.OraType.Unknown);
    try testing.expect(result.hasErrors());
    try testing.expect(result.getErrors().len == 1);

    const error_obj = result.getErrors()[0];
    try testing.expect(error_obj.kind == .invalid_operation);
    try testing.expect(std.mem.indexOf(u8, error_obj.message, "explicit type annotation") != null);
}

test "TypeValidator - variable declaration without explicit type" {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    const setup = try createTestSetup(allocator);
    defer cleanupTestSetup(allocator, setup);

    // Test variable declaration with Unknown type (should fail)
    var var_decl = ast.VariableDeclNode{
        .name = "test_var",
        .region = .Stack,
        .mutable = false,
        .locked = false,
        .typ = ast.TypeRef.Unknown,
        .value = null,
        .span = .{ .line = 1, .column = 1, .length = 10 },
        .tuple_names = null,
    };

    var result = type_validator.ValidationResult.init(allocator);
    defer result.deinit();

    try setup.validator.validateVariableDecl(&var_decl, &result);

    // Should have an error for missing explicit type annotation
    try testing.expect(result.hasErrors());
    try testing.expect(result.getErrors().len == 1);

    const error_obj = result.getErrors()[0];
    try testing.expect(error_obj.kind == .invalid_operation);
    try testing.expect(std.mem.indexOf(u8, error_obj.message, "explicit type annotation") != null);
}

test "TypeValidator - binary operation with type mismatch" {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    const setup = try createTestSetup(allocator);
    defer cleanupTestSetup(allocator, setup);

    var result = type_validator.ValidationResult.init(allocator);
    defer result.deinit();

    const span = ast.SourceSpan{ .line = 1, .column = 1, .length = 5 };

    // Test addition of u32 and u64 (should fail - no implicit conversion)
    const result_type = try setup.validator.validateBinaryOperation(
        .Plus,
        typer.OraType.U32,
        typer.OraType.U64,
        span,
        &result,
    );

    try testing.expect(result_type == typer.OraType.Unknown);
    try testing.expect(result.hasErrors());
    try testing.expect(result.getErrors().len == 1);

    const error_obj = result.getErrors()[0];
    try testing.expect(error_obj.kind == .type_mismatch);
    try testing.expect(std.mem.indexOf(u8, error_obj.message, "same type") != null);
    try testing.expect(error_obj.expected_type != null);
    try testing.expect(error_obj.actual_type != null);
}

test "TypeValidator - binary operation with same types" {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    const setup = try createTestSetup(allocator);
    defer cleanupTestSetup(allocator, setup);

    var result = type_validator.ValidationResult.init(allocator);
    defer result.deinit();

    const span = ast.SourceSpan{ .line = 1, .column = 1, .length = 5 };

    // Test addition of u256 and u256 (should succeed)
    const result_type = try setup.validator.validateBinaryOperation(
        .Plus,
        typer.OraType.U256,
        typer.OraType.U256,
        span,
        &result,
    );

    try testing.expect(result_type == typer.OraType.U256);
    try testing.expect(!result.hasErrors());
}

test "TypeValidator - logical operation with non-boolean types" {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    const setup = try createTestSetup(allocator);
    defer cleanupTestSetup(allocator, setup);

    var result = type_validator.ValidationResult.init(allocator);
    defer result.deinit();

    const span = ast.SourceSpan{ .line = 1, .column = 1, .length = 5 };

    // Test logical AND with u256 operands (should fail)
    const result_type = try setup.validator.validateBinaryOperation(
        .And,
        typer.OraType.U256,
        typer.OraType.U256,
        span,
        &result,
    );

    try testing.expect(result_type == typer.OraType.Unknown);
    try testing.expect(result.hasErrors());
    try testing.expect(result.getErrors().len == 1);

    const error_obj = result.getErrors()[0];
    try testing.expect(error_obj.kind == .type_mismatch);
    try testing.expect(std.mem.indexOf(u8, error_obj.message, "boolean operands") != null);
}

test "TypeValidator - division operation warning" {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    const setup = try createTestSetup(allocator);
    defer cleanupTestSetup(allocator, setup);

    var result = type_validator.ValidationResult.init(allocator);
    defer result.deinit();

    const span = ast.SourceSpan{ .line = 1, .column = 1, .length = 5 };

    // Test division operation (should succeed but generate warning)
    const result_type = try setup.validator.validateBinaryOperation(
        .Slash,
        typer.OraType.U256,
        typer.OraType.U256,
        span,
        &result,
    );

    try testing.expect(result_type == typer.OraType.U256);
    try testing.expect(!result.hasErrors());
    try testing.expect(result.hasWarnings());
    try testing.expect(result.getWarnings().len == 1);

    const warning_obj = result.getWarnings()[0];
    try testing.expect(warning_obj.severity == .warning);
    try testing.expect(std.mem.indexOf(u8, warning_obj.message, "runtime checks") != null);
}

test "TypeValidator - exact type matching" {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    const setup = try createTestSetup(allocator);
    defer cleanupTestSetup(allocator, setup);

    // Test exact type matching
    try testing.expect(setup.validator.areTypesExactMatch(typer.OraType.U256, typer.OraType.U256));
    try testing.expect(!setup.validator.areTypesExactMatch(typer.OraType.U256, typer.OraType.U128));
    try testing.expect(!setup.validator.areTypesExactMatch(typer.OraType.U32, typer.OraType.I32));
}

test "TypeValidator - numeric type checking" {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    const setup = try createTestSetup(allocator);
    defer cleanupTestSetup(allocator, setup);

    // Test numeric type identification
    try testing.expect(setup.validator.isNumericType(typer.OraType.U256));
    try testing.expect(setup.validator.isNumericType(typer.OraType.I128));
    try testing.expect(!setup.validator.isNumericType(typer.OraType.Bool));
    try testing.expect(!setup.validator.isNumericType(typer.OraType.Address));
    try testing.expect(!setup.validator.isNumericType(typer.OraType.String));
}

test "TypeValidator - integer type checking" {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    const setup = try createTestSetup(allocator);
    defer cleanupTestSetup(allocator, setup);

    // Test integer type identification
    try testing.expect(setup.validator.isIntegerType(typer.OraType.U256));
    try testing.expect(setup.validator.isIntegerType(typer.OraType.I128));
    try testing.expect(!setup.validator.isIntegerType(typer.OraType.Bool));
    try testing.expect(!setup.validator.isIntegerType(typer.OraType.Address));
}

test "TypeValidator - validation result management" {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    var result = type_validator.ValidationResult.init(allocator);
    defer result.deinit();

    // Test adding different severity levels
    const error_obj = type_validator.ValidationError.init(
        .type_mismatch,
        "Test error",
        .{ .line = 1, .column = 1, .length = 1 },
    );

    const warning_obj = type_validator.ValidationError.init(
        .invalid_operation,
        "Test warning",
        .{ .line = 1, .column = 1, .length = 1 },
    ).withSeverity(.warning);

    const info_obj = type_validator.ValidationError.init(
        .invalid_operation,
        "Test info",
        .{ .line = 1, .column = 1, .length = 1 },
    ).withSeverity(.info);

    try result.addError(error_obj);
    try result.addError(warning_obj);
    try result.addError(info_obj);

    try testing.expect(result.hasErrors());
    try testing.expect(result.hasWarnings());
    try testing.expect(result.getErrors().len == 1);
    try testing.expect(result.getWarnings().len == 1);
    try testing.expect(result.getInfos().len == 1);
    try testing.expect(result.getTotalCount() == 3);
}
