const std = @import("std");
const ast = @import("../ast/mod.zig");
const model = @import("model.zig");
const source = @import("../source/mod.zig");

const VerificationFact = model.VerificationFact;
const VerificationFactKind = model.VerificationFactKind;
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
    var facts: std.ArrayList(VerificationFact) = .empty;

    try appendVerificationFacts(arena, file, key, &facts);

    result.facts = try facts.toOwnedSlice(arena);
    return result;
}

pub fn appendVerificationFacts(
    allocator: std.mem.Allocator,
    file: *const ast.AstFile,
    key: VerificationFactsKey,
    facts: *std.ArrayList(VerificationFact),
) !void {
    switch (key) {
        .item => |item_id| try collectFactsForItem(allocator, file, item_id, facts),
        .body => |body_id| {
            for (file.items, 0..) |item, index| {
                if (item == .Function and item.Function.body == body_id) {
                    try collectFactsForItem(allocator, file, ast.ItemId.fromIndex(index), facts);
                }
            }
        },
        .trait_method => |owner| try collectFactsForTraitMethod(allocator, file, owner, facts),
        .statement => |owner| try collectDirectFactsForStatement(allocator, file, owner, facts),
    }
}

fn collectFactsForItem(allocator: std.mem.Allocator, file: *const ast.AstFile, item_id: ast.ItemId, facts: *std.ArrayList(VerificationFact)) !void {
    switch (file.item(item_id).*) {
        .Contract => |contract| {
            for (contract.invariants) |expr_id| {
                try appendInvariantFact(allocator, file, facts, .{
                    .kind = .contract_invariant,
                    .owner = .{ .item = item_id },
                    .range = source.TextRange.empty(0),
                    .context = .contract,
                }, expr_id);
            }
            for (contract.members) |member_id| {
                try collectGhostFactsForItem(allocator, file, member_id, facts);
            }
        },
        .Trait => |trait_item| {
            for (trait_item.methods, 0..) |_, method_index| {
                try collectFactsForTraitMethod(allocator, file, .{
                    .trait_item = item_id,
                    .method_index = method_index,
                }, facts);
            }
            if (trait_item.ghost_block) |ghost_id| {
                const ghost = file.item(ghost_id).GhostBlock;
                try facts.append(allocator, .{
                    .kind = .ghost_block,
                    .owner = .{ .item = ghost_id },
                    .range = ghost.range,
                    .context = .trait_ghost_block,
                });
                try collectGhostBodyFacts(allocator, file, ghost.body, item_id, .trait_ghost_block, facts);
                try collectStatementFactsForBody(allocator, file, ghost.body, ghost_id, facts);
            }
        },
        .Function => |function| {
            if (function.is_ghost) {
                try facts.append(allocator, .{
                    .kind = .ghost_function,
                    .owner = .{ .item = item_id },
                    .range = function.range,
                    .context = .ghost_declaration,
                });
                try collectGhostBodyFacts(allocator, file, function.body, item_id, .ghost_declaration, facts);
            }
            for (function.clauses) |clause| {
                const fact_kind = VerificationFactKind.fromSpecClause(clause.kind) orelse
                    return error.InvalidVerificationSpecClause;
                try facts.append(allocator, .{
                    .kind = fact_kind,
                    .owner = .{ .item = item_id },
                    .expr = clause.expr,
                    .range = clause.range,
                });
            }
            try collectStatementFactsForBody(allocator, file, function.body, item_id, facts);
        },
        else => try collectGhostFactsForItem(allocator, file, item_id, facts),
    }
}

fn collectGhostFactsForItem(allocator: std.mem.Allocator, file: *const ast.AstFile, item_id: ast.ItemId, facts: *std.ArrayList(VerificationFact)) !void {
    switch (file.item(item_id).*) {
        .Function => |function| {
            if (!function.is_ghost) return;
            try facts.append(allocator, .{
                .kind = .ghost_function,
                .owner = .{ .item = item_id },
                .range = function.range,
                .context = .ghost_declaration,
            });
            try collectGhostBodyFacts(allocator, file, function.body, item_id, .ghost_declaration, facts);
        },
        .Field => |field| {
            if (!field.is_ghost) return;
            try facts.append(allocator, .{
                .kind = .ghost_field,
                .owner = .{ .item = item_id },
                .range = field.range,
                .context = .ghost_declaration,
            });
        },
        .Constant => |constant| {
            if (!constant.is_ghost) return;
            try facts.append(allocator, .{
                .kind = .ghost_constant,
                .owner = .{ .item = item_id },
                .range = constant.range,
                .context = .ghost_declaration,
            });
        },
        .GhostBlock => |ghost| {
            try facts.append(allocator, .{
                .kind = .ghost_block,
                .owner = .{ .item = item_id },
                .range = ghost.range,
                .context = .ghost_block,
            });
            try collectGhostBodyFacts(allocator, file, ghost.body, item_id, .ghost_block, facts);
            try collectStatementFactsForBody(allocator, file, ghost.body, item_id, facts);
        },
        else => {},
    }
}

const InvariantPayload = struct {
    expr_id: ast.ExprId,
    label: ?[]const u8,
};

fn invariantPayload(file: *const ast.AstFile, expr_id: ast.ExprId) InvariantPayload {
    return switch (file.expression(expr_id).*) {
        .Call => |call| blk: {
            if (call.args.len == 1) {
                const label = switch (file.expression(call.callee).*) {
                    .Name => |name| name.name,
                    else => null,
                };
                break :blk .{ .expr_id = call.args[0], .label = label };
            }
            break :blk .{ .expr_id = expr_id, .label = null };
        },
        else => .{ .expr_id = expr_id, .label = null },
    };
}

fn appendInvariantFact(
    allocator: std.mem.Allocator,
    file: *const ast.AstFile,
    facts: *std.ArrayList(VerificationFact),
    base: VerificationFact,
    expr_id: ast.ExprId,
) !void {
    const payload = invariantPayload(file, expr_id);
    var fact = base;
    fact.expr = payload.expr_id;
    fact.label = payload.label;
    fact.range = source.rangeOf(file.expression(expr_id).*);
    try facts.append(allocator, fact);
}

fn collectStatementFactsForBody(
    allocator: std.mem.Allocator,
    file: *const ast.AstFile,
    body_id: ast.BodyId,
    owner_id: ast.ItemId,
    facts: *std.ArrayList(VerificationFact),
) anyerror!void {
    const body = file.body(body_id).*;
    for (body.statements) |stmt_id| {
        try collectStatementFacts(allocator, file, owner_id, stmt_id, facts);
    }
}

fn collectStatementFacts(
    allocator: std.mem.Allocator,
    file: *const ast.AstFile,
    owner_id: ast.ItemId,
    stmt_id: ast.StmtId,
    facts: *std.ArrayList(VerificationFact),
) anyerror!void {
    try collectDirectFactsForStatement(allocator, file, .{ .item = owner_id, .stmt = stmt_id }, facts);

    switch (file.statement(stmt_id).*) {
        .If => |if_stmt| {
            try collectStatementFactsForBody(allocator, file, if_stmt.then_body, owner_id, facts);
            if (if_stmt.else_body) |else_body| {
                try collectStatementFactsForBody(allocator, file, else_body, owner_id, facts);
            }
        },
        .While => |while_stmt| try collectStatementFactsForBody(allocator, file, while_stmt.body, owner_id, facts),
        .For => |for_stmt| try collectStatementFactsForBody(allocator, file, for_stmt.body, owner_id, facts),
        .Switch => |switch_stmt| {
            for (switch_stmt.arms) |arm| {
                try collectStatementFactsForBody(allocator, file, arm.body, owner_id, facts);
            }
            if (switch_stmt.else_body) |else_body| {
                try collectStatementFactsForBody(allocator, file, else_body, owner_id, facts);
            }
        },
        .Try => |try_stmt| {
            try collectStatementFactsForBody(allocator, file, try_stmt.try_body, owner_id, facts);
            if (try_stmt.catch_clause) |catch_clause| {
                try collectStatementFactsForBody(allocator, file, catch_clause.body, owner_id, facts);
            }
        },
        .Block => |block| try collectStatementFactsForBody(allocator, file, block.body, owner_id, facts),
        .LabeledBlock => |block| try collectStatementFactsForBody(allocator, file, block.body, owner_id, facts),
        else => {},
    }
}

fn collectDirectFactsForStatement(
    allocator: std.mem.Allocator,
    file: *const ast.AstFile,
    owner: model.VerificationStatementOwner,
    facts: *std.ArrayList(VerificationFact),
) !void {
    switch (file.statement(owner.stmt).*) {
        .While => |while_stmt| {
            for (while_stmt.invariants) |expr_id| {
                try appendInvariantFact(allocator, file, facts, .{
                    .kind = .loop_invariant,
                    .owner = .{ .statement = owner },
                    .range = source.TextRange.empty(0),
                    .context = .loop,
                }, expr_id);
            }
        },
        .For => |for_stmt| {
            for (for_stmt.invariants) |expr_id| {
                try appendInvariantFact(allocator, file, facts, .{
                    .kind = .loop_invariant,
                    .owner = .{ .statement = owner },
                    .range = source.TextRange.empty(0),
                    .context = .loop,
                }, expr_id);
            }
        },
        .Switch => |switch_stmt| {
            for (switch_stmt.invariants) |expr_id| {
                try appendInvariantFact(allocator, file, facts, .{
                    .kind = .loop_invariant,
                    .owner = .{ .statement = owner },
                    .range = source.TextRange.empty(0),
                    .context = .loop,
                }, expr_id);
            }
        },
        .Havoc => |havoc_stmt| {
            try facts.append(allocator, .{
                .kind = .havoc,
                .owner = .{ .statement = owner },
                .target_name = havoc_stmt.name,
                .range = havoc_stmt.range,
            });
        },
        else => {},
    }
}

fn collectFactsForTraitMethod(
    allocator: std.mem.Allocator,
    file: *const ast.AstFile,
    owner: model.VerificationTraitMethodOwner,
    facts: *std.ArrayList(VerificationFact),
) !void {
    const trait_item = switch (file.item(owner.trait_item).*) {
        .Trait => |trait_item| trait_item,
        else => return,
    };
    if (owner.method_index >= trait_item.methods.len) return;

    const method = trait_item.methods[owner.method_index];
    for (method.clauses) |clause| {
        const fact_kind = VerificationFactKind.fromSpecClause(clause.kind) orelse
            return error.InvalidVerificationSpecClause;
        try facts.append(allocator, .{
            .kind = fact_kind,
            .owner = .{ .trait_method = owner },
            .expr = clause.expr,
            .range = clause.range,
            .context = .trait_method_contract,
        });
    }
}

fn collectGhostBodyFacts(
    allocator: std.mem.Allocator,
    file: *const ast.AstFile,
    body_id: ast.BodyId,
    owner_id: ast.ItemId,
    context: model.VerificationContext,
    facts: *std.ArrayList(VerificationFact),
) anyerror!void {
    const body = file.body(body_id).*;
    for (body.statements) |stmt_id| {
        try collectGhostStatementFacts(allocator, file, stmt_id, owner_id, context, facts);
    }
}

fn collectGhostStatementFacts(
    allocator: std.mem.Allocator,
    file: *const ast.AstFile,
    stmt_id: ast.StmtId,
    owner_id: ast.ItemId,
    context: model.VerificationContext,
    facts: *std.ArrayList(VerificationFact),
) anyerror!void {
    switch (file.statement(stmt_id).*) {
        .Assert => |assert_stmt| {
            try facts.append(allocator, .{
                .kind = .assert,
                .owner = .{ .item = owner_id },
                .expr = assert_stmt.condition,
                .range = assert_stmt.range,
                .context = context,
            });
        },
        .Assume => |assume_stmt| {
            try facts.append(allocator, .{
                .kind = .assume,
                .owner = .{ .item = owner_id },
                .expr = assume_stmt.condition,
                .range = assume_stmt.range,
                .context = context,
            });
        },
        .Expr => |expr_stmt| {
            if (context != .trait_ghost_block) return;
            try facts.append(allocator, .{
                .kind = .ghost_axiom,
                .owner = .{ .item = owner_id },
                .expr = expr_stmt.expr,
                .range = source.rangeOf(file.expression(expr_stmt.expr).*),
                .context = context,
            });
        },
        .Block => |block| try collectGhostBodyFacts(allocator, file, block.body, owner_id, context, facts),
        .LabeledBlock => |block| try collectGhostBodyFacts(allocator, file, block.body, owner_id, context, facts),
        .If => |if_stmt| {
            try collectGhostBodyFacts(allocator, file, if_stmt.then_body, owner_id, context, facts);
            if (if_stmt.else_body) |else_body| {
                try collectGhostBodyFacts(allocator, file, else_body, owner_id, context, facts);
            }
        },
        .While => |while_stmt| try collectGhostBodyFacts(allocator, file, while_stmt.body, owner_id, context, facts),
        .For => |for_stmt| try collectGhostBodyFacts(allocator, file, for_stmt.body, owner_id, context, facts),
        .Switch => |switch_stmt| {
            for (switch_stmt.arms) |arm| {
                try collectGhostBodyFacts(allocator, file, arm.body, owner_id, context, facts);
            }
            if (switch_stmt.else_body) |else_body| {
                try collectGhostBodyFacts(allocator, file, else_body, owner_id, context, facts);
            }
        },
        .Try => |try_stmt| {
            try collectGhostBodyFacts(allocator, file, try_stmt.try_body, owner_id, context, facts);
            if (try_stmt.catch_clause) |catch_clause| {
                try collectGhostBodyFacts(allocator, file, catch_clause.body, owner_id, context, facts);
            }
        },
        else => {},
    }
}
