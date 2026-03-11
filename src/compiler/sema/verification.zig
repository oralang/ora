const std = @import("std");
const ast = @import("../ast/mod.zig");
const model = @import("model.zig");
const source = @import("../source/mod.zig");

const VerificationFact = model.VerificationFact;
const VerificationFactsKey = model.VerificationFactsKey;
const VerificationFactsResult = model.VerificationFactsResult;

pub fn verificationFacts(allocator: std.mem.Allocator, file: *const ast.AstFile, key: VerificationFactsKey) !VerificationFactsResult {
    var result = VerificationFactsResult{
        .arena = std.heap.ArenaAllocator.init(allocator),
        .key = key,
        .facts = &[_]VerificationFact{},
    };
    errdefer result.deinit();

    const arena = result.arena.allocator();
    var facts: std.ArrayList(VerificationFact) = .{};

    switch (key) {
        .item => |item_id| try collectFactsForItem(arena, file, item_id, &facts),
        .body => |body_id| {
            for (file.items, 0..) |item, index| {
                if (item == .Function and item.Function.body == body_id) {
                    try collectFactsForItem(arena, file, ast.ItemId.fromIndex(index), &facts);
                }
            }
        },
    }

    result.facts = try facts.toOwnedSlice(arena);
    return result;
}

fn collectFactsForItem(allocator: std.mem.Allocator, file: *const ast.AstFile, item_id: ast.ItemId, facts: *std.ArrayList(VerificationFact)) !void {
    switch (file.item(item_id).*) {
        .Contract => |contract| {
            for (contract.invariants) |expr_id| {
                try facts.append(allocator, .{ .kind = .invariant, .expr = expr_id, .range = source.rangeOf(file.expression(expr_id).*) });
            }
        },
        .Function => |function| {
            for (function.clauses) |clause| {
                try facts.append(allocator, .{ .kind = clause.kind, .expr = clause.expr, .range = clause.range });
            }
        },
        else => {},
    }
}
