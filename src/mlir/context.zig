const std = @import("std");
const c = @import("c.zig").c;
const OraDialect = @import("dialect.zig").OraDialect;

pub const MlirContextHandle = struct {
    ctx: c.MlirContext,
    ora_dialect: OraDialect,
};

pub fn createContext(allocator: std.mem.Allocator) MlirContextHandle {
    const ctx = c.mlirContextCreate();

    // Register all standard MLIR dialects
    const registry = c.mlirDialectRegistryCreate();
    c.mlirRegisterAllDialects(registry);
    c.mlirContextAppendDialectRegistry(ctx, registry);
    c.mlirDialectRegistryDestroy(registry);
    c.mlirContextLoadAllAvailableDialects(ctx);

    // Initialize the Ora dialect
    var ora_dialect = OraDialect.init(ctx, allocator);
    ora_dialect.register() catch {
        std.debug.print("Warning: Failed to register Ora dialect\n", .{});
    };

    return .{
        .ctx = ctx,
        .ora_dialect = ora_dialect,
    };
}

pub fn destroyContext(handle: MlirContextHandle) void {
    c.mlirContextDestroy(handle.ctx);
}
