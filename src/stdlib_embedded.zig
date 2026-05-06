const std = @import("std");

pub const std_logical_path = "std";
pub const std_bytes_logical_path = "std/bytes";
pub const std_constants_logical_path = "std/constants";
pub const std_result_logical_path = "std/result";

pub const std_resolved_path = "embedded://std/std.ora";
pub const std_bytes_resolved_path = "embedded://std/bytes.ora";
pub const std_constants_resolved_path = "embedded://std/constants.ora";
pub const std_result_resolved_path = "embedded://std/result.ora";

pub const EmbeddedModule = struct {
    logical_path: []const u8,
    resolved_path: []const u8,
    source: []const u8,
};

// Embedded Ora stdlib modules compiled into the compiler binary.
const modules = [_]EmbeddedModule{
    .{
        .logical_path = std_logical_path,
        .resolved_path = std_resolved_path,
        .source = @embedFile("std/std.ora"),
    },
    .{
        .logical_path = std_bytes_logical_path,
        .resolved_path = std_bytes_resolved_path,
        .source = @embedFile("std/bytes.ora"),
    },
    .{
        .logical_path = std_constants_logical_path,
        .resolved_path = std_constants_resolved_path,
        .source = @embedFile("std/constants.ora"),
    },
    .{
        .logical_path = std_result_logical_path,
        .resolved_path = std_result_resolved_path,
        .source = @embedFile("std/result.ora"),
    },
};

pub fn all() []const EmbeddedModule {
    return &modules;
}

pub fn byLogicalPath(path: []const u8) ?EmbeddedModule {
    for (modules) |module| {
        if (std.mem.eql(u8, module.logical_path, path)) return module;
    }
    return null;
}

pub fn sourceForResolvedPath(path: []const u8) ?[]const u8 {
    for (modules) |module| {
        if (std.mem.eql(u8, module.resolved_path, path)) return module.source;
    }
    return null;
}
