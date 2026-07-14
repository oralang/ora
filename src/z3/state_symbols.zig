const std = @import("std");

pub fn currentStorageName(allocator: std.mem.Allocator, root: []const u8) ![]u8 {
    return std.fmt.allocPrint(allocator, "g_{s}", .{root});
}

pub fn entryStorageName(allocator: std.mem.Allocator, root: []const u8) ![]u8 {
    return std.fmt.allocPrint(allocator, "g_entry_{s}", .{root});
}

pub fn oldStorageName(allocator: std.mem.Allocator, root: []const u8) ![]u8 {
    return std.fmt.allocPrint(allocator, "old_{s}", .{root});
}
