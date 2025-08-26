const std = @import("std");
const c = @import("c.zig").c;

pub const MlirContextHandle = struct {
    ctx: c.MlirContext,
};

pub fn createContext() MlirContextHandle {
    const ctx = c.mlirContextCreate();
    const registry = c.mlirDialectRegistryCreate();
    c.mlirRegisterAllDialects(registry);
    c.mlirContextAppendDialectRegistry(ctx, registry);
    c.mlirDialectRegistryDestroy(registry);
    c.mlirContextLoadAllAvailableDialects(ctx);
    return .{ .ctx = ctx };
}

pub fn destroyContext(handle: MlirContextHandle) void {
    c.mlirContextDestroy(handle.ctx);
}
