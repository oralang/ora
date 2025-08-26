const std = @import("std");
const ctx_mod = @import("context.zig");
const c = @import("c.zig").c;

pub fn writeModuleToFile(module: c.MlirModule, path: []const u8) !void {
    const owned = try std.fs.cwd().createFile(path, .{});
    defer owned.close();

    const writer = owned.writer();

    const callback = struct {
        fn cb(str: c.MlirStringRef, user: ?*anyopaque) callconv(.C) void {
            const w: *std.fs.File.Writer = @ptrCast(@alignCast(user.?));
            _ = w.writeAll(str.data[0..str.length]) catch {};
        }
    };

    const op = c.mlirModuleGetOperation(module);
    c.mlirOperationPrint(op, callback.cb, @ptrCast(&writer));
}
