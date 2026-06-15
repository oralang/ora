const std = @import("std");

pub const EmbeddedModule = struct {
    logical_path: []const u8,
    resolved_path: []const u8,
    source: []const u8,
};

// Embedded Ora stdlib modules compiled into the compiler binary.
const modules = [_]EmbeddedModule{
    .{
        .logical_path = "std",
        .resolved_path = "embedded://std/std.ora",
        .source = @embedFile("std/std.ora"),
    },
    .{
        .logical_path = "std/bytes",
        .resolved_path = "embedded://std/bytes.ora",
        .source = @embedFile("std/bytes.ora"),
    },
    .{
        .logical_path = "std/constants",
        .resolved_path = "embedded://std/constants.ora",
        .source = @embedFile("std/constants.ora"),
    },
    .{
        .logical_path = "std/result",
        .resolved_path = "embedded://std/result.ora",
        .source = @embedFile("std/result.ora"),
    },
    .{
        .logical_path = "std/interfaces",
        .resolved_path = "embedded://std/interfaces.ora",
        .source = @embedFile("std/interfaces.ora"),
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

pub fn byResolvedPath(path: []const u8) ?EmbeddedModule {
    for (modules) |module| {
        if (std.mem.eql(u8, module.resolved_path, path)) return module;
    }
    return null;
}

pub fn sourceForResolvedPath(path: []const u8) ?[]const u8 {
    return if (byResolvedPath(path)) |module| module.source else null;
}
