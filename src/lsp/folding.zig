const std = @import("std");
const compiler = @import("../compiler.zig");
const frontend = @import("frontend.zig");

const Allocator = std.mem.Allocator;

pub const FoldingRange = struct {
    start_line: u32,
    end_line: u32,
    kind: Kind,

    pub const Kind = enum { region, comment, imports };
};

pub fn foldingRanges(allocator: Allocator, source: []const u8) ![]FoldingRange {
    var ranges = std.ArrayList(FoldingRange){};
    errdefer ranges.deinit(allocator);

    try collectCommentFolds(&ranges, allocator, source);

    var sources = compiler.source.SourceStore.init(allocator);
    defer sources.deinit();
    const file_id = try sources.addFile("<lsp>", source);

    var parse_result = compiler.syntax.parse(allocator, file_id, source) catch
        return ranges.toOwnedSlice(allocator);
    defer parse_result.deinit();

    var lower_result = compiler.ast.lower(allocator, &parse_result.tree) catch
        return ranges.toOwnedSlice(allocator);
    defer lower_result.deinit();

    for (lower_result.file.root_items) |item_id| {
        try collectItemFolds(&ranges, allocator, &lower_result.file, item_id, &sources, file_id);
    }

    return ranges.toOwnedSlice(allocator);
}

pub fn deinitRanges(allocator: Allocator, ranges: []FoldingRange) void {
    allocator.free(ranges);
}

fn collectItemFolds(
    ranges: *std.ArrayList(FoldingRange),
    allocator: Allocator,
    file: *const compiler.ast.AstFile,
    item_id: compiler.ast.ItemId,
    sources: *const compiler.source.SourceStore,
    file_id: compiler.FileId,
) !void {
    const item = file.item(item_id).*;
    switch (item) {
        .Contract => |decl| {
            try addBlockRange(ranges, allocator, sources, file_id, decl.range);
            for (decl.members) |member_id| {
                try collectItemFolds(ranges, allocator, file, member_id, sources, file_id);
            }
        },
        .Function => |decl| {
            try addBlockRange(ranges, allocator, sources, file_id, decl.range);
            try collectBodyFolds(ranges, allocator, file, decl.body, sources, file_id);
        },
        .Struct => |decl| try addBlockRange(ranges, allocator, sources, file_id, decl.range),
        .Bitfield => |decl| try addBlockRange(ranges, allocator, sources, file_id, decl.range),
        .Enum => |decl| try addBlockRange(ranges, allocator, sources, file_id, decl.range),
        .Trait => |decl| {
            try addBlockRange(ranges, allocator, sources, file_id, decl.range);
        },
        .LogDecl => |decl| try addBlockRange(ranges, allocator, sources, file_id, decl.range),
        .ErrorDecl => |decl| try addBlockRange(ranges, allocator, sources, file_id, decl.range),
        else => {},
    }
}

fn collectBodyFolds(
    ranges: *std.ArrayList(FoldingRange),
    allocator: Allocator,
    file: *const compiler.ast.AstFile,
    body_id: compiler.ast.BodyId,
    sources: *const compiler.source.SourceStore,
    file_id: compiler.FileId,
) !void {
    const body = file.body(body_id).*;
    for (body.statements) |stmt_id| {
        const stmt = file.statement(stmt_id).*;
        switch (stmt) {
            .If => |s| {
                try addBlockRange(ranges, allocator, sources, file_id, s.range);
                try collectBodyFolds(ranges, allocator, file, s.then_body, sources, file_id);
                if (s.else_body) |else_body| {
                    try collectBodyFolds(ranges, allocator, file, else_body, sources, file_id);
                }
            },
            .While => |s| {
                try addBlockRange(ranges, allocator, sources, file_id, s.range);
                try collectBodyFolds(ranges, allocator, file, s.body, sources, file_id);
            },
            .For => |s| {
                try addBlockRange(ranges, allocator, sources, file_id, s.range);
                try collectBodyFolds(ranges, allocator, file, s.body, sources, file_id);
            },
            .Block => |s| {
                try collectBodyFolds(ranges, allocator, file, s.body, sources, file_id);
            },
            else => {},
        }
    }
}

fn addBlockRange(
    ranges: *std.ArrayList(FoldingRange),
    allocator: Allocator,
    sources: *const compiler.source.SourceStore,
    file_id: compiler.FileId,
    range: compiler.TextRange,
) !void {
    const start = sources.lineColumn(.{
        .file_id = file_id,
        .range = .{ .start = range.start, .end = range.start },
    });
    const end = sources.lineColumn(.{
        .file_id = file_id,
        .range = .{ .start = range.end, .end = range.end },
    });
    const start_line = if (start.line > 0) start.line - 1 else 0;
    const end_line = if (end.line > 0) end.line - 1 else 0;
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

        const line_text = std.mem.trimLeft(u8, source[line_start..i], " \t");
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
