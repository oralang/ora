const std = @import("std");
const fmt_mod = @import("ora_fmt");

const Allocator = std.mem.Allocator;

pub const Options = struct {
    line_width: u32 = 100,
    indent_size: u32 = 4,
};

pub const Error = fmt_mod.FormatError || Allocator.Error;

pub fn formatSourceAlloc(allocator: Allocator, source: []const u8, options: Options) Error![]u8 {
    var formatter = fmt_mod.Formatter.init(allocator, source, .{
        .line_width = options.line_width,
        .indent_size = options.indent_size,
    });
    defer formatter.deinit();
    return formatter.format();
}
