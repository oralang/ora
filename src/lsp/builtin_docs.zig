const std = @import("std");
const builtins = @import("../builtins.zig");

pub const Entry = builtins.Entry;

pub const entries = builtins.entries;

pub fn entryForName(name: []const u8) ?Entry {
    return builtins.entryForName(name);
}

pub fn matchesPrefix(name: []const u8, prefix: []const u8) bool {
    return prefix.len == 0 or std.mem.startsWith(u8, name, prefix);
}

pub fn markdownAlloc(allocator: std.mem.Allocator, entry: Entry) ![]u8 {
    return std.fmt.allocPrint(
        allocator,
        "{s}\n\nExample:\n```ora\n{s}\n```",
        .{ entry.documentation, entry.example },
    );
}
