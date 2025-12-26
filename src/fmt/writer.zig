// ============================================================================
// Formatter Writer
// ============================================================================
//
// Handles indentation, line width, and output formatting
//
// ============================================================================

const std = @import("std");

pub const Writer = struct {
    output: std.ArrayList(u8),
    indent_level: u32 = 0,
    indent_size: u32 = 4,
    line_width: u32 = 100,
    current_line_length: u32 = 0,
    needs_indent: bool = true,
    allocator: std.mem.Allocator,

    pub fn init(allocator: std.mem.Allocator, indent_size: u32, line_width: u32) Writer {
        return Writer{
            .output = std.ArrayList(u8){},
            .indent_level = 0,
            .indent_size = indent_size,
            .line_width = line_width,
            .current_line_length = 0,
            .needs_indent = true,
            .allocator = allocator,
        };
    }

    pub fn deinit(self: *Writer) void {
        self.output.deinit(self.allocator);
    }

    pub fn write(self: *Writer, text: []const u8) !void {
        if (self.needs_indent) {
            try self.writeIndent();
            self.needs_indent = false;
        }
        try self.output.appendSlice(self.allocator, text);
        self.current_line_length += @intCast(text.len);
    }

    pub fn writeByte(self: *Writer, byte: u8) !void {
        if (self.needs_indent) {
            try self.writeIndent();
            self.needs_indent = false;
        }
        try self.output.append(self.allocator, byte);
        self.current_line_length += 1;
    }

    pub fn newline(self: *Writer) !void {
        try self.output.append(self.allocator, '\n');
        self.current_line_length = 0;
        self.needs_indent = true;
    }

    pub fn writeIndent(self: *Writer) !void {
        var i: u32 = 0;
        while (i < self.indent_level * self.indent_size) : (i += 1) {
            try self.output.append(self.allocator, ' ');
        }
        self.current_line_length = self.indent_level * self.indent_size;
    }

    pub fn indent(self: *Writer) void {
        self.indent_level += 1;
    }

    pub fn dedent(self: *Writer) void {
        if (self.indent_level > 0) {
            self.indent_level -= 1;
        }
    }

    pub fn space(self: *Writer) !void {
        try self.write(" ");
    }

    pub fn toOwnedSlice(self: *Writer) ![]u8 {
        return try self.output.toOwnedSlice(self.allocator);
    }

    pub fn getWritten(self: *const Writer) []const u8 {
        return self.output.items;
    }

    pub fn wouldExceedWidth(self: *const Writer, additional: u32) bool {
        return self.current_line_length + additional > self.line_width;
    }
};
