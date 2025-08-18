const std = @import("std");
const testing = std.testing;
const ast = @import("ast.zig");
const typer = @import("typer.zig");
const type_validator = @import("type_validator.zig");
const explicit_type_checker = @import("explicit_type_checker.zig");
const ast_arena = @import("ast/ast_arena.zig");

// Test helper to create a basic test setup
fn createTestSetup(allocator: std.mem.Allocator) !struct {
    arena: *ast_arena.AstArena,
    typer_instance: *typer.Typer,
    validator: *type_validator.TypeValidator,
    checker: *explicit_type_checker.ExplicitTypeChecker,
} {
    const arena = try allocator.create(ast_arena.AstArena);
    arena.* = ast_arena.AstArena.init(allocator);

    const typer_instance = try allocator.create(typer.Typer);
    typer_instance.* = typer.Typer.init(allocator);

    const validator = try allocator.create(type_validator.TypeValidator);
    validator.* = type_validator.TypeValidator.init(allocator, arena, typer_instance);

    const checker = try allocator.create(explicit_type_checker.ExplicitTypeChecker);
    checker.* = explicit_type_checker.ExplicitTypeChecker.init(allocator, arena, typer_instance, validator);

    return .{
        .arena = arena,
        .typer_instance = typer_instance,
        .validator = validator,
        .checker = checker,
    };
}

fn cleanupTestSetup(allocator: std.mem.Allocator, setup: anytype) void {
    setup.validator.deinit();
    setup.typer_instance.deinit();
    setup.arena.deinit();
    allocator.destroy(setup.checker);
    allocator.destroy(setup.validator);
    allocator.destroy(setup.typer_instance);
    allocator.destroy(setup.arena);
}

test "ExplicitTypeChecker - integer literal requires explicit type" {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    const setup = try createTestSetup(allocator);
    defer cleanupTestSetup(allocator, setup);

    var int_literal = ast.LiteralNode{
        .Integer = .{
            .value = "42",
            .span = .{ .line = 1, .column = 1, .length = 2 },
        },
    };

    var result = type_validator.ValidationResult.init(allocator);
    defer result.deinit();

    // Test without context type (should fail)
    try setup.checker.checkLiteralRequiresExplicitType(&int_literal, null, &result);

    try testing.expect(result.hasErrors());
    try testing.expect(result.getErrors().len == 1);

    const error_obj = result.getErrors()[0];
    try testing.expect(error_obj.kind == .invalid_operation);
    try testing.expect(std.mem.indexOf(u8, error_obj.message, "explicit type annotation") != null);
    try testing.expect(error_obj.suggestions.len > 0);
}

test "ExplicitTypeChecker - string literal allowed without explicit type" {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    const setup = try createTestSetup(allocator);
    defer cleanupTestSetup(allocator, setup);

    var string_literal = ast.LiteralNode{
        .String = .{
            .value = "hello",
            .span = .{ .line = 1, .column = 1, .length = 7 },
        },
    };

    var result = type_validator.ValidationResult.init(allocator);
    defer result.deinit();

    // String literals are allowed without explicit type
    try setup.checker.checkLiteralRequiresExplicitType(&string_literal, null, &result);

    try testing.expect(!result.hasErrors());
}

test "ExplicitTypeChecker - no implicit conversion" {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    const setup = try createTestSetup(allocator);
    defer cleanupTestSetup(allocator, setup);

    var result = type_validator.ValidationResult.init(allocator);
    defer result.deinit();

    const span = ast.SourceSpan{ .line = 1, .column = 1, .length = 5 };

    // Test u32 to u64 conversion (should fail)
    const conversion_ok = try setup.checker.checkNoImplicitConversion(
        typer.OraType.U32,
        typer.OraType.U64,
        span,
        "assignment",
        &result,
    );

    try testing.expect(!conversion_ok);
    try testing.expect(result.hasErrors());
    try testing.expect(result.getErrors().len == 1);

    const error_obj = result.getErrors()[0];
    try testing.expect(error_obj.kind == .type_mismatch);
    try testing.expect(std.mem.indexOf(u8, error_obj.message, "No implicit conversion") != null);
    try testing.expect(error_obj.suggestions.len > 0);
}

test "ExplicitTypeChecker - same types allowed" {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    const setup = try createTestSetup(allocator);
    defer cleanupTestSetup(allocator, setup);

    var result = type_validator.ValidationResult.init(allocator);
    defer result.deinit();

    const span = ast.SourceSpan{ .line = 1, .column = 1, .length = 5 };

    // Test u256 to u256 (should succeed)
    const conversion_ok = try setup.checker.checkNoImplicitConversion(
        typer.OraType.U256,
        typer.OraType.U256,
        span,
        "assignment",
        &result,
    );

    try testing.expect(conversion_ok);
    try testing.expect(!result.hasErrors());
}

test "ExplicitTypeChecker - variable declaration without explicit type" {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    const setup = try createTestSetup(allocator);
    defer cleanupTestSetup(allocator, setup);

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

    try setup.checker.validateExplicitTypeAnnotation(&var_decl, &result);

    try testing.expect(result.hasErrors());
    try testing.expect(result.getErrors().len == 1);

    const error_obj = result.getErrors()[0];
    try testing.expect(error_obj.kind == .invalid_operation);
    try testing.expect(std.mem.indexOf(u8, error_obj.message, "explicit type annotation") != null);
    try testing.expect(error_obj.suggestions.len > 0);
}

test "ExplicitTypeChecker - variable declaration with explicit type" {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    const setup = try createTestSetup(allocator);
    defer cleanupTestSetup(allocator, setup);

    var var_decl = ast.VariableDeclNode{
        .name = "test_var",
        .region = .Stack,
        .mutable = false,
        .locked = false,
        .typ = ast.TypeRef.U256,
        .value = null,
        .span = .{ .line = 1, .column = 1, .length = 10 },
        .tuple_names = null,
    };

    var result = type_validator.ValidationResult.init(allocator);
    defer result.deinit();

    try setup.checker.validateExplicitTypeAnnotation(&var_decl, &result);

    // Should not have errors for explicit type annotation
    try testing.expect(!result.hasErrors());
}

test "ExplicitTypeChecker - arithmetic operation with type mismatch" {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    const setup = try createTestSetup(allocator);
    defer cleanupTestSetup(allocator, setup);

    var result = type_validator.ValidationResult.init(allocator);
    defer result.deinit();

    const span = ast.SourceSpan{ .line = 1, .column = 1, .length = 5 };

    // Test u32 + u64 (should fail)
    const result_type = try setup.checker.validateArithmeticOperation(
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
    try testing.expect(error_obj.suggestions.len > 0);
}

test "ExplicitTypeChecker - arithmetic operation with same types" {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    const setup = try createTestSetup(allocator);
    defer cleanupTestSetup(allocator, setup);

    var result = type_validator.ValidationResult.init(allocator);
    defer result.deinit();

    const span = ast.SourceSpan{ .line = 1, .column = 1, .length = 5 };

    // Test u256 + u256 (should succeed)
    const result_type = try setup.checker.validateArithmeticOperation(
        .Plus,
        typer.OraType.U256,
        typer.OraType.U256,
        span,
        &result,
    );

    try testing.expect(result_type == typer.OraType.U256);
    try testing.expect(!result.hasErrors());
}

test "ExplicitTypeChecker - division operation warning" {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    const setup = try createTestSetup(allocator);
    defer cleanupTestSetup(allocator, setup);

    var result = type_validator.ValidationResult.init(allocator);
    defer result.deinit();

    const span = ast.SourceSpan{ .line = 1, .column = 1, .length = 5 };

    // Test u256 / u256 (should succeed but warn)
    const result_type = try setup.checker.validateArithmeticOperation(
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

test "ExplicitTypeChecker - function call argument mismatch" {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    const setup = try createTestSetup(allocator);
    defer cleanupTestSetup(allocator, setup);

    var result = type_validator.ValidationResult.init(allocator);
    defer result.deinit();

    const span = ast.SourceSpan{ .line = 1, .column = 1, .length = 10 };

    const expected_params = [_]typer.OraType{ typer.OraType.U256, typer.OraType.Bool };
    const actual_args = [_]typer.OraType{ typer.OraType.U32, typer.OraType.Bool };

    const args_ok = try setup.checker.validateFunctionCallArguments(
        &expected_params,
        &actual_args,
        span,
        &result,
    );

    try testing.expect(!args_ok);
    try testing.expect(result.hasErrors());
    try testing.expect(result.getErrors().len == 1);

    const error_obj = result.getErrors()[0];
    try testing.expect(error_obj.kind == .type_mismatch);
    try testing.expect(std.mem.indexOf(u8, error_obj.message, "argument 1 type mismatch") != null);
}

test "ExplicitTypeChecker - safe cast validation" {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    const setup = try createTestSetup(allocator);
    defer cleanupTestSetup(allocator, setup);

    var result = type_validator.ValidationResult.init(allocator);
    defer result.deinit();

    const span = ast.SourceSpan{ .line = 1, .column = 1, .length = 5 };

    // Test safe cast u32 to u64 (should succeed)
    const result_type = try setup.checker.validateExplicitCast(
        typer.OraType.U32,
        typer.OraType.U64,
        .Safe,
        span,
        &result,
    );

    try testing.expect(result_type == typer.OraType.U64);
    try testing.expect(!result.hasErrors());
}

test "ExplicitTypeChecker - invalid safe cast" {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    const setup = try createTestSetup(allocator);
    defer cleanupTestSetup(allocator, setup);

    var result = type_validator.ValidationResult.init(allocator);
    defer result.deinit();

    const span = ast.SourceSpan{ .line = 1, .column = 1, .length = 5 };

    // Test safe cast u64 to u32 (should fail)
    const result_type = try setup.checker.validateExplicitCast(
        typer.OraType.U64,
        typer.OraType.U32,
        .Safe,
        span,
        &result,
    );

    try testing.expect(result_type == typer.OraType.Unknown);
    try testing.expect(result.hasErrors());
    try testing.expect(result.getErrors().len == 1);

    const error_obj = result.getErrors()[0];
    try testing.expect(error_obj.kind == .invalid_operation);
    try testing.expect(std.mem.indexOf(u8, error_obj.message, "not valid") != null);
}

test "ExplicitTypeChecker - unsafe cast warning" {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    const setup = try createTestSetup(allocator);
    defer cleanupTestSetup(allocator, setup);

    var result = type_validator.ValidationResult.init(allocator);
    defer result.deinit();

    const span = ast.SourceSpan{ .line = 1, .column = 1, .length = 5 };

    // Test unsafe cast u64 to u32 (should succeed with warning)
    const result_type = try setup.checker.validateExplicitCast(
        typer.OraType.U64,
        typer.OraType.U32,
        .Unsafe,
        span,
        &result,
    );

    try testing.expect(result_type == typer.OraType.U32);
    try testing.expect(!result.hasErrors());
    try testing.expect(result.hasWarnings());
    try testing.expect(result.getWarnings().len == 1);

    const warning_obj = result.getWarnings()[0];
    try testing.expect(warning_obj.severity == .warning);
    try testing.expect(std.mem.indexOf(u8, warning_obj.message, "data loss") != null);
}

test "ExplicitTypeChecker - forced cast info" {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    const setup = try createTestSetup(allocator);
    defer cleanupTestSetup(allocator, setup);

    var result = type_validator.ValidationResult.init(allocator);
    defer result.deinit();

    const span = ast.SourceSpan{ .line = 1, .column = 1, .length = 5 };

    // Test forced cast (should succeed with info)
    const result_type = try setup.checker.validateExplicitCast(
        typer.OraType.Bool,
        typer.OraType.U32,
        .Forced,
        span,
        &result,
    );

    try testing.expect(result_type == typer.OraType.U32);
    try testing.expect(!result.hasErrors());
    try testing.expect(!result.hasWarnings());
    try testing.expect(result.getInfos().len == 1);

    const info_obj = result.getInfos()[0];
    try testing.expect(info_obj.severity == .info);
    try testing.expect(std.mem.indexOf(u8, info_obj.message, "bypasses all safety checks") != null);
};