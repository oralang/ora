const std = @import("std");
const compiler = @import("../compiler.zig");
const frontend = @import("frontend.zig");

const Allocator = std.mem.Allocator;

pub const VerificationLens = struct {
    range: frontend.Range,
    title: []u8,

    pub fn deinit(self: *VerificationLens, allocator: Allocator) void {
        allocator.free(self.title);
    }
};

/// Scan source for functions/contracts with verification annotations and return
/// code lenses showing their spec clause summary.
pub fn findVerificationLenses(allocator: Allocator, source: []const u8) ![]VerificationLens {
    var lenses = std.ArrayList(VerificationLens){};
    errdefer {
        for (lenses.items) |*lens| lens.deinit(allocator);
        lenses.deinit(allocator);
    }

    var sources = compiler.source.SourceStore.init(allocator);
    defer sources.deinit();
    const file_id = try sources.addFile("<lsp>", source);

    var parse_result = compiler.syntax.parse(allocator, file_id, source) catch return try lenses.toOwnedSlice(allocator);
    defer parse_result.deinit();

    var lower_result = compiler.ast.lower(allocator, &parse_result.tree) catch return try lenses.toOwnedSlice(allocator);
    defer lower_result.deinit();

    for (lower_result.file.root_items) |item_id| {
        try collectLenses(&lenses, allocator, &lower_result.file, item_id, &sources, file_id);
    }

    return lenses.toOwnedSlice(allocator);
}

pub fn deinitLenses(allocator: Allocator, lenses: []VerificationLens) void {
    for (lenses) |*lens| lens.deinit(allocator);
    allocator.free(lenses);
}

fn collectLenses(
    lenses: *std.ArrayList(VerificationLens),
    allocator: Allocator,
    file: *const compiler.ast.AstFile,
    item_id: compiler.ast.ItemId,
    sources: *const compiler.source.SourceStore,
    file_id: compiler.FileId,
) !void {
    const item = file.item(item_id).*;
    switch (item) {
        .Contract => |contract_decl| {
            if (contract_decl.invariants.len > 0) {
                const title = try std.fmt.allocPrint(allocator, "contract invariant: {d} clause{s}", .{
                    contract_decl.invariants.len,
                    if (contract_decl.invariants.len != 1) "s" else "",
                });
                const range = textRangeToFrontendRange(sources, file_id, contract_decl.range);
                try lenses.append(allocator, .{ .range = range, .title = title });
            }
            for (contract_decl.members) |member_id| {
                try collectLenses(lenses, allocator, file, member_id, sources, file_id);
            }
        },
        .Function => |function_decl| {
            if (function_decl.is_ghost) {
                const range = textRangeToFrontendRange(sources, file_id, function_decl.range);
                try lenses.append(allocator, .{
                    .range = range,
                    .title = try allocator.dupe(u8, "ghost function"),
                });
            } else if (function_decl.clauses.len > 0) {
                const range = textRangeToFrontendRange(sources, file_id, function_decl.range);
                const title = try formatClausesSummary(allocator, function_decl.clauses);
                try lenses.append(allocator, .{ .range = range, .title = title });
            }
        },
        .Trait => |trait_decl| {
            for (trait_decl.methods) |method| {
                if (method.clauses.len > 0) {
                    const range = textRangeToFrontendRange(sources, file_id, method.range);
                    const title = try formatClausesSummary(allocator, method.clauses);
                    try lenses.append(allocator, .{ .range = range, .title = title });
                }
            }
        },
        else => {},
    }
}

fn formatClausesSummary(allocator: Allocator, clauses: []const compiler.ast.SpecClause) ![]u8 {
    var requires_count: u32 = 0;
    var ensures_count: u32 = 0;
    var invariant_count: u32 = 0;

    for (clauses) |clause| {
        switch (clause.kind) {
            .requires => requires_count += 1,
            .ensures => ensures_count += 1,
            .invariant => invariant_count += 1,
        }
    }

    var buffer = std.ArrayList(u8){};
    errdefer buffer.deinit(allocator);
    const writer = buffer.writer(allocator);

    var first = true;
    if (requires_count > 0) {
        try writer.print("{d} requires", .{requires_count});
        first = false;
    }
    if (ensures_count > 0) {
        if (!first) try writer.writeAll(", ");
        try writer.print("{d} ensures", .{ensures_count});
        first = false;
    }
    if (invariant_count > 0) {
        if (!first) try writer.writeAll(", ");
        try writer.print("{d} invariant", .{invariant_count});
    }

    return buffer.toOwnedSlice(allocator);
}

fn textRangeToFrontendRange(sources: *const compiler.source.SourceStore, file_id: compiler.FileId, range: compiler.TextRange) frontend.Range {
    const start = sources.lineColumn(.{
        .file_id = file_id,
        .range = .{ .start = range.start, .end = range.start },
    });
    const end = sources.lineColumn(.{
        .file_id = file_id,
        .range = .{ .start = range.end, .end = range.end },
    });
    return .{
        .start = .{
            .line = if (start.line > 0) start.line - 1 else 0,
            .character = if (start.column > 0) start.column - 1 else 0,
        },
        .end = .{
            .line = if (end.line > 0) end.line - 1 else 0,
            .character = if (end.column > 0) end.column - 1 else 0,
        },
    };
}
