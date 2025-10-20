const std = @import("std");
const ast = @import("ast.zig");
const lexer = @import("lexer.zig");

pub const CstKind = enum {
    Root,
    TokenStream,
    ContractDecl,
    FunctionDecl,
    VarDecl,
    StructDecl,
    EnumDecl,
    LogDecl,
    ImportDecl,
    ErrorDecl,
};

pub const CstNode = struct {
    kind: CstKind,
    span: ast.SourceSpan,
    token_start: u32, // inclusive token index
    token_end: u32, // exclusive token index
    // For simple modes we keep either token indices or child nodes; not both populated
    token_indices: []u32 = &[_]u32{},
    child_nodes: []*CstNode = &[_]*CstNode{},
};

pub const CstBuilder = struct {
    arena: std.heap.ArenaAllocator,
    top_levels: std.ArrayList(*CstNode),
    backing_allocator: std.mem.Allocator,

    pub fn init(alloc: std.mem.Allocator) CstBuilder {
        return CstBuilder{ .arena = std.heap.ArenaAllocator.init(alloc), .top_levels = std.ArrayList(*CstNode){}, .backing_allocator = alloc };
    }

    pub fn deinit(self: *CstBuilder) void {
        // Free lists before destroying the arena that backs them
        self.top_levels.deinit(self.backing_allocator);
        self.arena.deinit();
    }

    pub fn getAllocator(self: *CstBuilder) std.mem.Allocator {
        return self.arena.allocator();
    }

    /// Build a shallow CST that represents the token stream under a single root
    pub fn buildTokenStream(self: *CstBuilder, tokens: []const lexer.Token) !*CstNode {
        const alloc = self.getAllocator();
        var root = try alloc.create(CstNode);
        // Build a root span compatible with ast.SourceSpan (line, column, length)
        const span0: ast.SourceSpan = blk: {
            if (tokens.len == 0) break :blk .{ .line = 1, .column = 1, .length = 0, .lexeme = null };
            // Find last non-EOF token to compute length
            var last_idx: usize = tokens.len - 1;
            while (last_idx > 0 and tokens[last_idx].type == .Eof) : (last_idx -= 1) {}
            const start_off: u32 = tokens[0].range.start_offset;
            const end_off: u32 = tokens[last_idx].range.end_offset;
            const len: u32 = if (end_off > start_off) end_off - start_off else 0;
            break :blk .{ .file_id = 0, .line = tokens[0].line, .column = tokens[0].column, .length = len, .byte_offset = tokens[0].range.start_offset, .lexeme = null };
        };
        root.* = .{
            .kind = .Root,
            .span = span0,
            .token_start = 0,
            .token_end = @as(u32, @intCast(tokens.len)),
            .token_indices = &[_]u32{},
            .child_nodes = &[_]*CstNode{},
        };

        var stream = try alloc.create(CstNode);
        stream.* = .{
            .kind = .TokenStream,
            .span = root.span,
            .token_start = 0,
            .token_end = root.token_end,
            .token_indices = &[_]u32{},
            .child_nodes = &[_]*CstNode{},
        };

        // Collect non-EOF token indices
        var indices = std.ArrayList(u32){};
        for (tokens, 0..) |t, idx| {
            if (t.type == .Eof) continue;
            try indices.append(alloc, @as(u32, @intCast(idx)));
        }
        stream.token_indices = try indices.toOwnedSlice(alloc);

        var kids = std.ArrayList(*CstNode){};
        try kids.append(alloc, stream);
        // Move to arena-owned slice
        root.child_nodes = try kids.toOwnedSlice(alloc);
        return root;
    }

    /// Create and record a top-level CST node
    pub fn createTopLevel(self: *CstBuilder, kind: CstKind, span: ast.SourceSpan, token_start: u32, token_end: u32) !*CstNode {
        const alloc = self.getAllocator();
        const n = try alloc.create(CstNode);
        n.* = .{
            .kind = kind,
            .span = span,
            .token_start = token_start,
            .token_end = token_end,
            .token_indices = &[_]u32{},
            .child_nodes = &[_]*CstNode{},
        };
        try self.top_levels.append(self.backing_allocator, n);
        return n;
    }

    /// Build a root node that contains all recorded top-level CST nodes
    pub fn buildRoot(self: *CstBuilder, tokens: []const lexer.Token) !*CstNode {
        const alloc = self.getAllocator();
        const root = try alloc.create(CstNode);
        // Compute simple span from first token
        const span0: ast.SourceSpan = blk: {
            if (tokens.len == 0) break :blk .{ .file_id = 0, .line = 1, .column = 1, .length = 0, .byte_offset = 0, .lexeme = null };
            var last_idx: usize = tokens.len - 1;
            while (last_idx > 0 and tokens[last_idx].type == .Eof) : (last_idx -= 1) {}
            const start_off: u32 = tokens[0].range.start_offset;
            const end_off: u32 = tokens[last_idx].range.end_offset;
            const len: u32 = if (end_off > start_off) end_off - start_off else 0;
            break :blk .{ .file_id = 0, .line = tokens[0].line, .column = tokens[0].column, .length = len, .byte_offset = tokens[0].range.start_offset, .lexeme = null };
        };
        root.* = .{
            .kind = .Root,
            .span = span0,
            .token_start = 0,
            .token_end = @as(u32, @intCast(tokens.len)),
            .token_indices = &[_]u32{},
            .child_nodes = &[_]*CstNode{},
        };
        // Move recorded top-levels to arena-owned slice
        const tmp = try self.top_levels.toOwnedSlice(self.backing_allocator);
        const dst = try self.getAllocator().alloc(*CstNode, tmp.len);
        std.mem.copyForwards(*CstNode, dst, tmp);
        self.backing_allocator.free(tmp);
        root.child_nodes = dst;
        // Reset internal list using backing allocator and free old buffer
        self.top_levels = std.ArrayList(*CstNode){};
        return root;
    }
};
