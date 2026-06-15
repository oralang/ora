const std = @import("std");
const compiler = @import("../compiler.zig");
const frontend = @import("frontend.zig");
const line_index_api = @import("line_index.zig");

const Allocator = std.mem.Allocator;

pub const VerificationLens = struct {
    range: frontend.Range,
    title: []u8,

    pub fn deinit(self: *VerificationLens, allocator: Allocator) void {
        allocator.free(self.title);
    }
};

pub fn findVerificationLensesInAst(
    allocator: Allocator,
    source: []const u8,
    file: *const compiler.ast.AstFile,
    line_index: *const line_index_api.LineIndex,
) ![]VerificationLens {
    var lenses = std.ArrayList(VerificationLens){};
    errdefer {
        for (lenses.items) |*lens| lens.deinit(allocator);
        lenses.deinit(allocator);
    }

    const range_context: RangeContext = .{ .line_index = .{
        .source = source,
        .index = line_index,
    } };
    for (file.root_items) |item_id| {
        try collectLenses(&lenses, allocator, file, item_id, range_context);
    }

    return lenses.toOwnedSlice(allocator);
}

pub fn deinitLenses(allocator: Allocator, lenses: []VerificationLens) void {
    for (lenses) |*lens| lens.deinit(allocator);
    allocator.free(lenses);
}

const RangeContext = union(enum) {
    line_index: struct {
        source: []const u8,
        index: *const line_index_api.LineIndex,
    },

    fn textRangeToFrontendRange(self: RangeContext, range: compiler.TextRange) frontend.Range {
        return switch (self) {
            .line_index => |ctx| ctx.index.textRangeToRange(ctx.source, range, .utf8),
        };
    }
};

fn collectLenses(
    lenses: *std.ArrayList(VerificationLens),
    allocator: Allocator,
    file: *const compiler.ast.AstFile,
    item_id: compiler.ast.ItemId,
    range_context: RangeContext,
) !void {
    const item = file.item(item_id).*;
    switch (item) {
        .Contract => |contract_decl| {
            if (contract_decl.invariants.len > 0) {
                const title = try std.fmt.allocPrint(allocator, "contract invariant: {d} clause{s}", .{
                    contract_decl.invariants.len,
                    if (contract_decl.invariants.len != 1) "s" else "",
                });
                const range = range_context.textRangeToFrontendRange(contract_decl.range);
                try lenses.append(allocator, .{ .range = range, .title = title });
            }
            for (contract_decl.members) |member_id| {
                try collectLenses(lenses, allocator, file, member_id, range_context);
            }
        },
        .Function => |function_decl| {
            if (function_decl.is_ghost) {
                const range = range_context.textRangeToFrontendRange(function_decl.range);
                try lenses.append(allocator, .{
                    .range = range,
                    .title = try allocator.dupe(u8, "ghost function"),
                });
            } else if (function_decl.clauses.len > 0) {
                const range = range_context.textRangeToFrontendRange(function_decl.range);
                const title = try formatClausesSummary(allocator, function_decl.clauses);
                try lenses.append(allocator, .{ .range = range, .title = title });
            }
        },
        .Trait => |trait_decl| {
            for (trait_decl.methods) |method| {
                if (method.clauses.len > 0) {
                    const range = range_context.textRangeToFrontendRange(method.range);
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
    var guard_count: u32 = 0;
    var ensures_count: u32 = 0;
    var invariant_count: u32 = 0;
    var modifies_count: u32 = 0;

    for (clauses) |clause| {
        switch (clause.kind) {
            .requires => requires_count += 1,
            .guard => guard_count += 1,
            .ensures, .ensures_ok, .ensures_err => ensures_count += 1,
            .invariant => invariant_count += 1,
            .modifies => modifies_count += 1,
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
    if (guard_count > 0) {
        if (!first) try writer.writeAll(", ");
        try writer.print("{d} guard", .{guard_count});
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
        first = false;
    }
    if (modifies_count > 0) {
        if (!first) try writer.writeAll(", ");
        try writer.print("{d} modifies", .{modifies_count});
    }

    return buffer.toOwnedSlice(allocator);
}
