const std = @import("std");

pub fn addImports(b: *std.Build, module: *std.Build.Module, target: std.Build.ResolvedTarget, optimize: std.builtin.OptimizeMode, prefix: []const u8) void {
    const headers = b.addWriteFiles();
    inline for (.{ .{ "ckzg_c", "ckzg.h" }, .{ "blst_c", "blst.h" } }) |entry| {
        const translation = b.addTranslateC(.{
            .root_source_file = headers.add(entry[0] ++ ".h", "#include <" ++ entry[1] ++ ">\n"),
            .target = target,
            .optimize = optimize,
        });
        translation.addIncludePath(b.path(b.fmt("{s}/vendor/c-kzg-4844/src", .{prefix})));
        translation.addIncludePath(b.path(b.fmt("{s}/vendor/c-kzg-4844/blst/bindings", .{prefix})));
        module.addImport(entry[0], translation.createModule());
    }
}
