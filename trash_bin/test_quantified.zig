const std = @import("std");
const lib = @import("ora");
const c = @import("src/mlir/c.zig").c;
const ExpressionLowerer = @import("src/mlir/expressions.zig").ExpressionLowerer;
const TypeMapper = @import("src/mlir/types.zig").TypeMapper;
const LocationTracker = @import("src/mlir/locations.zig").LocationTracker;

test "quantified expression lowering" {
    // Initialize MLIR context
    const ctx = c.mlirContextCreate();
    defer c.mlirContextDestroy(ctx);

    // Create a simple quantified expression for testing
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const allocator = arena.allocator();

    // Create a forall expression: forall x: u256 => (x > 0)
    const variable_type = lib.ast.Types.TypeInfo{
        .ora_type = .u256,
        .is_mutable = false,
        .is_optional = false,
        .array_size = null,
        .key_type = null,
        .value_type = null,
        .struct_name = null,
        .enum_name = null,
        .error_types = null,
        .function_signature = null,
        .contract_name = null,
    };

    // Create body expression: x > 0
    const x_ident = try lib.ast.Expressions.createIdentifier(allocator, "x", lib.ast.SourceSpan{
        .line = 1,
        .column = 1,
        .length = 1,
        .byte_offset = 0,
    });

    const zero_literal = try lib.ast.Expressions.createUntypedIntegerLiteral(allocator, "0", lib.ast.SourceSpan{
        .line = 1,
        .column = 5,
        .length = 1,
        .byte_offset = 4,
    });

    const body_expr = try lib.ast.Expressions.createBinaryExpr(allocator, x_ident, .Greater, zero_literal, lib.ast.SourceSpan{
        .line = 1,
        .column = 1,
        .length = 5,
        .byte_offset = 0,
    });

    // Create quantified expression
    const quantified_expr = try lib.ast.Expressions.createQuantifiedExpr(allocator, .Forall, "x", variable_type, null, // no condition
        body_expr, lib.ast.SourceSpan{
        .line = 1,
        .column = 1,
        .length = 20,
        .byte_offset = 0,
    });

    // Create MLIR module and function for testing
    const module = c.mlirModuleCreateEmpty(c.mlirLocationUnknownGet(ctx));
    defer c.mlirModuleDestroy(module);

    // Create a function to contain our test
    const func_name = c.mlirStringRefCreateFromCString("test_func");
    const func_op = c.mlirOperationCreate(&c.MlirOperationState{
        .name = c.mlirStringRefCreateFromCString("func.func"),
        .location = c.mlirLocationUnknownGet(ctx),
        .nOperands = 0,
        .operands = null,
        .nResults = 0,
        .results = null,
        .nSuccessors = 0,
        .successors = null,
        .nRegions = 1,
        .regions = null,
        .nAttributes = 1,
        .attributes = &c.MlirNamedAttribute{
            .name = c.mlirIdentifierGet(ctx, c.mlirStringRefCreateFromCString("sym_name")),
            .attribute = c.mlirStringAttrGet(ctx, func_name),
        },
    });

    const region = c.mlirOperationGetRegion(func_op, 0);
    const block = c.mlirBlockCreate(0, null, null);
    c.mlirRegionAppendOwnedBlock(region, block);

    // Create expression lowerer
    const type_mapper = TypeMapper.init(ctx);
    const locations = LocationTracker.init();
    const expr_lowerer = ExpressionLowerer.init(ctx, block, &type_mapper, null, null, null, locations);

    // Lower the quantified expression
    const result = expr_lowerer.lowerExpression(quantified_expr);

    // Verify that we got a valid MLIR value
    try std.testing.expect(!c.mlirValueIsNull(result));

    // Verify the result type is boolean (i1)
    const result_type = c.mlirValueGetType(result);
    try std.testing.expect(c.mlirTypeIsAInteger(result_type));
    try std.testing.expect(c.mlirIntegerTypeGetWidth(result_type) == 1);

    std.debug.print("✓ Quantified expression lowering test passed\n", .{});
}
