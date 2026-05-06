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
    pub const Options = struct {
        proofs_enabled: bool = false,
    };

    cfg: c.Z3_config,
    ctx: c.Z3_context,
    allocator: std.mem.Allocator,
    proofs_enabled: bool,

    /// Initialize a new Z3 context with default configuration
    pub fn init(allocator: std.mem.Allocator) !Context {
        return initWithOptions(allocator, .{});
    }

    pub fn initWithOptions(allocator: std.mem.Allocator, options: Options) !Context {
        const cfg = c.Z3_mk_config() orelse return error.Z3InitFailed;
        errdefer c.Z3_del_config(cfg);

        if (options.proofs_enabled) {
            c.Z3_set_param_value(cfg, "proof", "true");
        }

        const ctx = try createContext(cfg);

        return Context{
            .cfg = cfg,
            .ctx = ctx,
            .allocator = allocator,
            .proofs_enabled = options.proofs_enabled,
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

    pub fn lastErrorMessage(self: *Context) []const u8 {
        const code = self.lastErrorCode();
        const raw = c.Z3_get_error_msg(self.ctx, code);
        if (raw == null) return @errorName(error.Z3ApiError);
        return std.mem.span(raw);
    }

    pub fn lastErrorMessageOwned(self: *Context, allocator: std.mem.Allocator) ![]u8 {
        return try allocator.dupe(u8, self.lastErrorMessage());
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

test "Context initWithOptions default registers a valid context" {
    var ctx = try Context.initWithOptions(testing.allocator, .{});
    defer ctx.deinit();

    try testing.expect(ctx.ctx != null);
    try testing.expectEqual(@as(c.Z3_error_code, c.Z3_OK), ctx.lastErrorCode());
}

test "Context initWithOptions enables proof generation" {
    var ctx = try Context.initWithOptions(testing.allocator, .{ .proofs_enabled = true });
    defer ctx.deinit();

    try testing.expect(ctx.ctx != null);
    try testing.expect(ctx.proofs_enabled);
    try testing.expectEqual(@as(c.Z3_error_code, c.Z3_OK), ctx.lastErrorCode());
}

test "Context lastErrorMessage reflects Z3 API failures" {
    var ctx = try Context.init(testing.allocator);
    defer ctx.deinit();

    _ = c.Z3_parse_smtlib2_string(ctx.ctx, "(assert", 0, null, null, 0, null, null);
    try testing.expect(ctx.lastErrorCode() != c.Z3_OK);
    try testing.expect(ctx.lastErrorMessage().len > 0);
}
