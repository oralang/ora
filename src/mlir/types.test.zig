// ============================================================================
// Type Mapper Tests
// ============================================================================

const std = @import("std");
const testing = std.testing;
const mlir = @import("mlir_c_api").c;
const lib = @import("ora_lib");
const TypeMapper = @import("types.zig").TypeMapper;
const ErrorHandler = @import("error_handling.zig").ErrorHandler;

test "toMlirType reports missing ora type" {
    const allocator = testing.allocator;
    const ctx = mlir.oraContextCreate();
    defer mlir.oraContextDestroy(ctx);

    var handler = ErrorHandler.init(allocator);
    defer handler.deinit();

    var mapper = TypeMapper.init(ctx, allocator);
    mapper.setErrorHandler(&handler);
    defer mapper.deinit();

    const unknown = lib.TypeInfo.unknown();
    const ty = mapper.toMlirType(unknown);

    try testing.expect(mlir.oraTypeIsNull(ty));
    try testing.expect(handler.hasErrors());
}
