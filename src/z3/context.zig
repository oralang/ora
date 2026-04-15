//===----------------------------------------------------------------------===//
//
// Z3 Context Management
//
//===----------------------------------------------------------------------===//
//
// Manages Z3 context lifecycle, configuration, and resource cleanup.
//
//===----------------------------------------------------------------------===//

const std = @import("std");
const c = @import("c.zig");
fn z3ErrorHandler(ctx: c.Z3_context, code: c.Z3_error_code) callconv(.c) void {
    _ = ctx;
    _ = code;
    // The wrappers inspect `lastErrorCode()` and surface structured Zig errors.
    // Printing here makes expected negative tests look like unexpected failures.
}

fn createContext(cfg: c.Z3_config) !c.Z3_context {
    const ctx = c.Z3_mk_context(cfg) orelse return error.Z3InitFailed;
    c.Z3_set_error_handler(ctx, z3ErrorHandler);
    return ctx;
}

/// Z3 Context wrapper for RAII-style resource management
pub const Context = struct {
    cfg: c.Z3_config,
    ctx: c.Z3_context,
    allocator: std.mem.Allocator,

    /// Initialize a new Z3 context with default configuration
    pub fn init(allocator: std.mem.Allocator) !Context {
        const cfg = c.Z3_mk_config() orelse return error.Z3InitFailed;
        errdefer c.Z3_del_config(cfg);

        const ctx = try createContext(cfg);

        return Context{
            .cfg = cfg,
            .ctx = ctx,
            .allocator = allocator,
        };
    }

    /// Initialize with custom configuration options
    pub fn initWithConfig(allocator: std.mem.Allocator, timeout_ms: u32) !Context {
        _ = timeout_ms;
        const cfg = c.Z3_mk_config() orelse return error.Z3InitFailed;
        errdefer c.Z3_del_config(cfg);

        const ctx = try createContext(cfg);

        return Context{
            .cfg = cfg,
            .ctx = ctx,
            .allocator = allocator,
        };
    }

    /// Clean up Z3 context and configuration
    pub fn deinit(self: *Context) void {
        c.Z3_del_context(self.ctx);
        c.Z3_del_config(self.cfg);
    }

    pub fn clearLastError(self: *Context) void {
        c.Z3_set_error(self.ctx, c.Z3_OK);
    }

    pub fn lastErrorCode(self: *Context) c.Z3_error_code {
        return c.Z3_get_error_code(self.ctx);
    }

    pub fn checkNoError(self: *Context) !void {
        if (self.lastErrorCode() == c.Z3_OK) return;
        return error.Z3ApiError;
    }
};

//===----------------------------------------------------------------------===//
// Tests
//===----------------------------------------------------------------------===//

const testing = std.testing;

test "Context init and deinit" {
    var ctx = try Context.init(testing.allocator);
    defer ctx.deinit();

    // context should be non-null
    try testing.expect(ctx.ctx != null);
}

test "Context initWithConfig registers a valid context" {
    var ctx = try Context.initWithConfig(testing.allocator, 50);
    defer ctx.deinit();

    try testing.expect(ctx.ctx != null);
    try testing.expectEqual(@as(c.Z3_error_code, c.Z3_OK), ctx.lastErrorCode());
}
