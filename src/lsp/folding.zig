const std = @import("std");
const compiler = @import("../compiler.zig");
const line_index_api = @import("line_index.zig");

const Allocator = std.mem.Allocator;

pub const FoldingRange = struct {
    start_line: u32,
    end_line: u32,
    kind: Kind,

    pub const Kind = enum { region, comment, imports };
};

pub fn foldingRangesInAst(
    allocator: Allocator,
    source: []const u8,
    file: *const compiler.ast.AstFile,
    line_index: *const line_index_api.LineIndex,
) ![]FoldingRange {
    var ranges = std.ArrayList(FoldingRange).empty;
    errdefer ranges.deinit(allocator);

    try collectCommentFolds(&ranges, allocator, source);

    const position_context: PositionContext = .{ .line_index = .{
        .source = source,
        .index = line_index,
    } };
    for (file.root_items) |item_id| {
        try collectItemFolds(&ranges, allocator, file, item_id, position_context);
    }

    return ranges.toOwnedSlice(allocator);
}

pub fn deinitRanges(allocator: Allocator, ranges: []FoldingRange) void {
    allocator.free(ranges);
}

const PositionContext = union(enum) {
    line_index: struct {
        source: []const u8,
        index: *const line_index_api.LineIndex,
    },

    fn lineForOffset(self: PositionContext, offset: u32) u32 {
        return switch (self) {
            .line_index => |ctx| ctx.index.offsetToPosition(ctx.source, offset, .utf8).line,
        };
    }
};

fn collectItemFolds(
    ranges: *std.ArrayList(FoldingRange),
    allocator: Allocator,
    file: *const compiler.ast.AstFile,
    item_id: compiler.ast.ItemId,
    position_context: PositionContext,
) !void {
    const item = file.item(item_id).*;
    switch (item) {
        .Contract => |decl| {
            try addBlockRange(ranges, allocator, position_context, decl.range);
            for (decl.members) |member_id| {
                try collectItemFolds(ranges, allocator, file, member_id, position_context);
            }
        },
        .Function => |decl| {
            try addBlockRange(ranges, allocator, position_context, decl.range);
            try collectBodyFolds(ranges, allocator, file, decl.body, position_context);
        },
        .Struct => |decl| try addBlockRange(ranges, allocator, position_context, decl.range),
        .Bitfield => |decl| try addBlockRange(ranges, allocator, position_context, decl.range),
        .Enum => |decl| try addBlockRange(ranges, allocator, position_context, decl.range),
        .Trait => |decl| {
            try addBlockRange(ranges, allocator, position_context, decl.range);
        },
        .LogDecl => |decl| try addBlockRange(ranges, allocator, position_context, decl.range),
        .ErrorDecl => |decl| try addBlockRange(ranges, allocator, position_context, decl.range),
        else => {},
    }
}

fn collectBodyFolds(
    ranges: *std.ArrayList(FoldingRange),
    allocator: Allocator,
    file: *const compiler.ast.AstFile,
    body_id: compiler.ast.BodyId,
    position_context: PositionContext,
) !void {
    const body = file.body(body_id).*;
    for (body.statements) |stmt_id| {
        const stmt = file.statement(stmt_id).*;
        switch (stmt) {
            .If => |s| {
                try addBlockRange(ranges, allocator, position_context, s.range);
                try collectBodyFolds(ranges, allocator, file, s.then_body, position_context);
                if (s.else_body) |else_body| {
                    try collectBodyFolds(ranges, allocator, file, else_body, position_context);
                }
            },
            .While => |s| {
                try addBlockRange(ranges, allocator, position_context, s.range);
                try collectBodyFolds(ranges, allocator, file, s.body, position_context);
            },
            .For => |s| {
                try addBlockRange(ranges, allocator, position_context, s.range);
                try collectBodyFolds(ranges, allocator, file, s.body, position_context);
            },
            .Block => |s| {
                try collectBodyFolds(ranges, allocator, file, s.body, position_context);
            },
            else => {},
        }
    }
}

fn addBlockRange(
    ranges: *std.ArrayList(FoldingRange),
    allocator: Allocator,
    position_context: PositionContext,
    range: compiler.TextRange,
) !void {
    const start_line = position_context.lineForOffset(range.start);
    const end_line = position_context.lineForOffset(range.end);
    if (end_line > start_line) {
        try ranges.append(allocator, .{
            .start_line = start_line,
            .end_line = end_line,
            .kind = .region,
        });
    }
}

fn collectCommentFolds(
    ranges: *std.ArrayList(FoldingRange),
    allocator: Allocator,
    source: []const u8,
) !void {
    var line: u32 = 0;
    var i: usize = 0;
    var comment_start: ?u32 = null;
    var comment_end: u32 = 0;

    while (i < source.len) {
        const line_start = i;
        while (i < source.len and source[i] != '\n') : (i += 1) {}

        const line_text = std.mem.trimStart(u8, source[line_start..i], " \t");
        if (std.mem.startsWith(u8, line_text, "//")) {
            if (comment_start == null) comment_start = line;
            comment_end = line;
        } else {
            if (comment_start) |start| {
                if (comment_end > start) {
                    try ranges.append(allocator, .{
                        .start_line = start,
                        .end_line = comment_end,
                        .kind = .comment,
                    });
                }
                comment_start = null;
            }
        }

        if (i < source.len) i += 1;
        line += 1;
    }

    if (comment_start) |start| {
        if (comment_end > start) {
            try ranges.append(allocator, .{
                .start_line = start,
                .end_line = comment_end,
                .kind = .comment,
            });
        }
    }
}
