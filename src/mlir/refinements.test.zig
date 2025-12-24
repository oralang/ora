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
    const ctx = mlir.mlirContextCreate();
    defer mlir.mlirContextDestroy(ctx);

    var handler = ErrorHandler.init(allocator);
    defer handler.deinit();

    var mapper = TypeMapper.init(ctx, allocator);
    defer mapper.deinit();

    var ora_dialect = OraDialect.init(ctx, allocator);

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
    );

    try testing.expect(handler.hasErrors());
}
