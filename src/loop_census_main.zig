const std = @import("std");
const ora_root = @import("ora_root");
const loop_census = @import("formal/loop_census.zig");

pub fn main(init: std.process.Init) !void {
    var gpa = std.heap.DebugAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    const args = try collectArgs(allocator, init.minimal.args);
    defer freeArgs(allocator, args);

    var stdout_buffer: [8192]u8 = undefined;
    var stdout_writer = std.Io.File.stdout().writer(std.Io.Threaded.global_single_threaded.io(), &stdout_buffer);
    const stdout = &stdout_writer.interface;

    if (args.len != 2) {
        try stdout.writeAll("usage: ora-loop-census <source.ora>\n");
        try stdout.flush();
        exitCli(2);
    }

    const path = args[1];
    var compilation = ora_root.compiler.compilePackageWithOptions(allocator, path, .{
        .compile_options = .{ .measure_loop_census = true },
    }) catch |err| {
        try loop_census.writeCompilerErrorJson(stdout, path, err);
        try stdout.flush();
        return;
    };
    defer compilation.deinit();

    loop_census.writeCompilationJson(stdout, allocator, path, &compilation) catch |err| {
        try loop_census.writeCompilerErrorJson(stdout, path, err);
    };
    try stdout.flush();
}

fn exitCli(code: u8) noreturn {
    std.process.exit(code);
}

fn collectArgs(allocator: std.mem.Allocator, process_args: std.process.Args) ![][]u8 {
    var iterator = try std.process.Args.Iterator.initAllocator(process_args, allocator);
    defer iterator.deinit();

    var list: std.ArrayList([]u8) = .empty;
    errdefer {
        for (list.items) |arg| allocator.free(arg);
        list.deinit(allocator);
    }
    while (iterator.next()) |arg| try list.append(allocator, try allocator.dupe(u8, arg));
    return list.toOwnedSlice(allocator);
}

fn freeArgs(allocator: std.mem.Allocator, args: [][]u8) void {
    for (args) |arg| allocator.free(arg);
    allocator.free(args);
}
