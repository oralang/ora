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

/// Z3 Context wrapper for RAII-style resource management
pub const Context = struct {
    cfg: c.Z3_config,
    ctx: c.Z3_context,
    allocator: std.mem.Allocator,

    /// Initialize a new Z3 context with default configuration
    pub fn init(allocator: std.mem.Allocator) !Context {
        const cfg = c.Z3_mk_config() orelse return error.Z3InitFailed;
        errdefer c.Z3_del_config(cfg);

        const ctx = c.Z3_mk_context(cfg) orelse return error.Z3InitFailed;

        return Context{
            .cfg = cfg,
            .ctx = ctx,
            .allocator = allocator,
        };
    }

    /// Initialize with custom configuration options
    pub fn initWithConfig(allocator: std.mem.Allocator, timeout_ms: u32) !Context {
        const cfg = c.Z3_mk_config() orelse return error.Z3InitFailed;
        errdefer c.Z3_del_config(cfg);

        // set timeout
        const timeout_str = try std.fmt.allocPrintZ(allocator, "{d}", .{timeout_ms});
        defer allocator.free(timeout_str);
        c.Z3_set_param_value(cfg, "timeout", timeout_str.ptr);

        const ctx = c.Z3_mk_context(cfg) orelse return error.Z3InitFailed;

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
