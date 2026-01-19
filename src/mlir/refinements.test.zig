// ============================================================================
// Refinement Guard Tests
// ============================================================================

const std = @import("std");
const testing = std.testing;
const mlir = @import("mlir_c_api").c;
const lib = @import("ora_lib");
const ErrorHandler = @import("error_handling.zig").ErrorHandler;
const TypeMapper = @import("types.zig").TypeMapper;
const LocationTracker = @import("locations.zig").LocationTracker;
const OraDialect = @import("dialect.zig").OraDialect;
const DeclarationLowerer = @import("declarations/mod.zig").DeclarationLowerer;

test "refinement guard reports error for null value" {
    const allocator = testing.allocator;
    const ctx = mlir.oraContextCreate();
    defer mlir.oraContextDestroy(ctx);

    var handler = ErrorHandler.init(allocator);
    defer handler.deinit();

    var mapper = TypeMapper.init(ctx, allocator);
    defer mapper.deinit();

    var ora_dialect = OraDialect.init(ctx, allocator);
    try ora_dialect.register();

    const locations = LocationTracker.init(ctx);
    const lowerer = DeclarationLowerer.withErrorHandlerAndDialect(ctx, &mapper, locations, &handler, &ora_dialect);

    const base_type = lib.ast.type_info.OraType{ .u256 = {} };
    const refinement_type = lib.ast.type_info.OraType{
        .min_value = .{ .base = &base_type, .min = 1 },
    };
    const span = lib.ast.SourceSpan{ .line = 1, .column = 1, .length = 1 };

    try lowerer.insertRefinementGuard(
        mlir.MlirBlock{ .ptr = null },
        mlir.MlirValue{ .ptr = null },
        refinement_type,
        span,
        null,
        null,
        null,
    );

    try testing.expect(handler.hasErrors());
}

test "non_zero_address refinement emits guard op" {
    const allocator = testing.allocator;
    const ctx = mlir.oraContextCreate();
    defer mlir.oraContextDestroy(ctx);

    var handler = ErrorHandler.init(allocator);
    defer handler.deinit();

    var mapper = TypeMapper.init(ctx, allocator);
    defer mapper.deinit();

    var ora_dialect = OraDialect.init(ctx, allocator);
    try ora_dialect.register();

    const locations = LocationTracker.init(ctx);
    const lowerer = DeclarationLowerer.withErrorHandlerAndDialect(ctx, &mapper, locations, &handler, &ora_dialect);

    const loc = mlir.oraLocationUnknownGet(ctx);
    const module = mlir.oraModuleCreateEmpty(loc);
    defer mlir.oraModuleDestroy(module);
    const block = mlir.oraModuleGetBody(module);

    const i160_ty = mlir.oraIntegerTypeCreate(ctx, 160);
    const one_attr = mlir.oraIntegerAttrCreateI64FromType(i160_ty, 1);
    const one_const_op = mlir.oraArithConstantOpCreate(ctx, loc, i160_ty, one_attr);
    mlir.oraBlockAppendOwnedOperation(block, one_const_op);
    const one_value = mlir.oraOperationGetResult(one_const_op, 0);

    const addr_op = mlir.oraI160ToAddrOpCreate(ctx, loc, one_value);
    mlir.oraBlockAppendOwnedOperation(block, addr_op);
    const addr_value = mlir.oraOperationGetResult(addr_op, 0);

    const refinement_type = lib.ast.type_info.OraType{ .non_zero_address = {} };
    const span = lib.ast.SourceSpan{ .line = 1, .column = 1, .length = 1 };

    try lowerer.insertRefinementGuard(block, addr_value, refinement_type, span, null, null, null);

    var found_guard = false;
    var current = mlir.oraBlockGetFirstOperation(block);
    while (current.ptr != null) {
        const name = mlir.oraOperationGetName(current);
        if (name.data != null and name.length > 0) {
            if (std.mem.eql(u8, name.data[0..name.length], "ora.refinement_guard")) {
                found_guard = true;
                break;
            }
        }
        current = mlir.oraOperationGetNextInBlock(current);
    }

    try testing.expect(found_guard);
}
