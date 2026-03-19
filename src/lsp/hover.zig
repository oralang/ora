const std = @import("std");
const frontend = @import("frontend.zig");
const semantic_index = @import("semantic_index.zig");

const Allocator = std.mem.Allocator;

pub const Hover = struct {
    contents: []const u8,
    range: frontend.Range,

    pub fn deinit(self: *Hover, allocator: Allocator) void {
        allocator.free(self.contents);
    }
};

pub fn hoverAt(allocator: Allocator, source: []const u8, position: frontend.Position) !?Hover {
    var index = try semantic_index.indexDocument(allocator, source);
    defer index.deinit(allocator);

    if (!index.parse_succeeded) return null;

    const symbol = blk: {
        const symbol_index = semantic_index.findSymbolAtPosition(index.symbols, position) orelse return null;
        const candidate = index.symbols[symbol_index];
        if (!rangeContainsPosition(candidate.selection_range, position)) return null;
        break :blk candidate;
    };

    const value = try formatHoverValueAlloc(allocator, symbol);
    return .{
        .contents = value,
        .range = symbol.selection_range,
    };
}

fn formatHoverValueAlloc(allocator: Allocator, symbol: semantic_index.Symbol) ![]u8 {
    const signature = try formatSignatureAlloc(allocator, symbol);
    defer allocator.free(signature);

    return std.fmt.allocPrint(allocator, "```ora\n{s}\n```", .{signature});
}

fn formatSignatureAlloc(allocator: Allocator, symbol: semantic_index.Symbol) ![]u8 {
    return switch (symbol.kind) {
        .contract => std.fmt.allocPrint(allocator, "contract {s}", .{symbol.name}),
        .function, .method => if (symbol.detail) |detail|
            std.fmt.allocPrint(allocator, "fn {s}{s}", .{ symbol.name, detail })
        else
            std.fmt.allocPrint(allocator, "fn {s}()", .{symbol.name}),
        .variable => if (symbol.detail) |detail|
            std.fmt.allocPrint(allocator, "var {s}: {s}", .{ symbol.name, detail })
        else
            std.fmt.allocPrint(allocator, "var {s}", .{symbol.name}),
        .field => if (symbol.detail) |detail|
            std.fmt.allocPrint(allocator, "field {s}: {s}", .{ symbol.name, detail })
        else
            std.fmt.allocPrint(allocator, "field {s}", .{symbol.name}),
        .constant => if (symbol.detail) |detail|
            std.fmt.allocPrint(allocator, "const {s}: {s}", .{ symbol.name, detail })
        else
            std.fmt.allocPrint(allocator, "const {s}", .{symbol.name}),
        .parameter => if (symbol.detail) |detail|
            std.fmt.allocPrint(allocator, "{s}: {s}", .{ symbol.name, detail })
        else
            std.fmt.allocPrint(allocator, "{s}", .{symbol.name}),
        .struct_decl => std.fmt.allocPrint(allocator, "struct {s}", .{symbol.name}),
        .bitfield_decl => std.fmt.allocPrint(allocator, "bitfield {s}", .{symbol.name}),
        .enum_decl => std.fmt.allocPrint(allocator, "enum {s}", .{symbol.name}),
        .enum_member => std.fmt.allocPrint(allocator, "enum member {s}", .{symbol.name}),
        .event => if (symbol.detail) |detail|
            std.fmt.allocPrint(allocator, "log {s}{s}", .{ symbol.name, detail })
        else
            std.fmt.allocPrint(allocator, "log {s}", .{symbol.name}),
        .error_decl => if (symbol.detail) |detail|
            std.fmt.allocPrint(allocator, "error {s}{s}", .{ symbol.name, detail })
        else
            std.fmt.allocPrint(allocator, "error {s}", .{symbol.name}),
    };
}

fn rangeContainsPosition(range: frontend.Range, position: frontend.Position) bool {
    if (positionLessThan(position, range.start)) return false;
    if (!positionLessThan(position, range.end)) return false;
    return true;
}

fn positionLessThan(lhs: frontend.Position, rhs: frontend.Position) bool {
    if (lhs.line < rhs.line) return true;
    if (lhs.line > rhs.line) return false;
    return lhs.character < rhs.character;
}
