const std = @import("std");

pub const EmbeddedImport = struct {
    alias: []const u8,
    specifier: []const u8,
};

pub const EmbeddedModule = struct {
    logical_path: []const u8,
    resolved_path: []const u8,
    source: []const u8,
    imports: []const EmbeddedImport = &.{},
};

const std_imports = [_]EmbeddedImport{
    .{ .alias = "constants", .specifier = "std/constants" },
    .{ .alias = "bytes", .specifier = "std/bytes" },
    .{ .alias = "result", .specifier = "std/result" },
    .{ .alias = "interfaces", .specifier = "std/interfaces" },
};

const bytes_imports = [_]EmbeddedImport{
    .{ .alias = "constants", .specifier = "std/constants" },
};

const storage_imports = [_]EmbeddedImport{
    .{ .alias = "words", .specifier = "std/storage/words" },
};

// Embedded Ora stdlib modules compiled into the compiler binary.
const modules = [_]EmbeddedModule{
    .{
        .logical_path = "std",
        .resolved_path = "embedded://std/std.ora",
        .source = @embedFile("std/std.ora"),
        .imports = &std_imports,
    },
    .{
        .logical_path = "std/bytes",
        .resolved_path = "embedded://std/bytes.ora",
        .source = @embedFile("std/bytes.ora"),
        .imports = &bytes_imports,
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
        .logical_path = "std/storage",
        .resolved_path = "embedded://std/storage.ora",
        .source = @embedFile("std/storage.ora"),
        .imports = &storage_imports,
    },
    .{
        .logical_path = "std/storage/words",
        .resolved_path = "embedded://std/storage/words.ora",
        .source = @embedFile("std/storage/words.ora"),
    },
    .{
        .logical_path = "std/storage/fixed_size_data",
        .resolved_path = "embedded://std/storage/fixed_size_data.ora",
        .source = @embedFile("std/storage/fixed_size_data.ora"),
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
