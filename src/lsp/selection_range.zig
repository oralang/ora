const std = @import("std");
const lsp = @import("lsp");
const ora_root = @import("ora_root");

const compiler = ora_root.compiler;
const line_index_api = ora_root.lsp.line_index;
const protocol_ranges = @import("protocol_ranges.zig");
const text_edits = ora_root.lsp.text_edits;

const Allocator = std.mem.Allocator;
const types = lsp.types;

pub fn build(
    arena: Allocator,
    source: []const u8,
    line_index: *const line_index_api.LineIndex,
    encoding: text_edits.PositionEncoding,
    file: *const compiler.ast.AstFile,
    positions: []const types.Position,
) ![]types.SelectionRange {
    const result = try arena.alloc(types.SelectionRange, positions.len);
    for (positions, 0..) |position, i| {
        const offset = line_index.positionToOffset(
            source,
            position.line,
            position.character,
            encoding,
        ) orelse @as(u32, @intCast(source.len));
        result[i] = try buildAtOffset(arena, source, line_index, encoding, file, offset);
    }
    return result;
}

pub fn buildAtOffset(
    arena: Allocator,
    source: []const u8,
    line_index: *const line_index_api.LineIndex,
    encoding: text_edits.PositionEncoding,
    file: *const compiler.ast.AstFile,
    offset: u32,
) !types.SelectionRange {
    var ranges = std.ArrayList(types.Range).empty;
    defer ranges.deinit(arena);

    for (file.root_items) |item_id| {
        try collectContainingRanges(&ranges, arena, source, line_index, file, item_id, offset, encoding);
    }

    std.mem.sort(types.Range, ranges.items, {}, rangeLargerThan);

    if (ranges.items.len == 0) {
        return .{ .range = .{ .start = .{ .line = 0, .character = 0 }, .end = .{ .line = 0, .character = 0 } } };
    }

    var current: types.SelectionRange = .{ .range = ranges.items[0] };

    for (ranges.items[1..]) |range| {
        const parent = try arena.create(types.SelectionRange);
        parent.* = current;
        current = .{ .range = range, .parent = parent };
    }
    return current;
}

fn collectContainingRanges(
    ranges: *std.ArrayList(types.Range),
    arena: Allocator,
    source: []const u8,
    line_index: *const line_index_api.LineIndex,
    file: *const compiler.ast.AstFile,
    item_id: compiler.ast.ItemId,
    offset: u32,
    encoding: text_edits.PositionEncoding,
) !void {
    const item = file.item(item_id).*;
    const item_range = itemTextRange(item) orelse return;
    if (offset < item_range.start or offset > item_range.end) return;

    try ranges.append(arena, protocol_ranges.textRangeToLsp(source, line_index, encoding, item_range));

    switch (item) {
        .Contract => |contract_decl| {
            for (contract_decl.members) |member_id| {
                try collectContainingRanges(ranges, arena, source, line_index, file, member_id, offset, encoding);
            }
        },
        .Function => |function_decl| try collectBodyContainingRanges(ranges, arena, source, line_index, file, function_decl.body, offset, encoding),
        else => {},
    }
}

fn collectBodyContainingRanges(
    ranges: *std.ArrayList(types.Range),
    arena: Allocator,
    source: []const u8,
    line_index: *const line_index_api.LineIndex,
    file: *const compiler.ast.AstFile,
    body_id: compiler.ast.BodyId,
    offset: u32,
    encoding: text_edits.PositionEncoding,
) !void {
    const body = file.body(body_id).*;
    for (body.statements) |stmt_id| {
        const stmt = file.statement(stmt_id).*;
        const stmt_range = stmtTextRange(stmt) orelse continue;
        if (offset < stmt_range.start or offset > stmt_range.end) continue;

        try ranges.append(arena, protocol_ranges.textRangeToLsp(source, line_index, encoding, stmt_range));
        switch (stmt) {
            .If => |if_stmt| {
                try collectBodyContainingRanges(ranges, arena, source, line_index, file, if_stmt.then_body, offset, encoding);
                if (if_stmt.else_body) |else_body| {
                    try collectBodyContainingRanges(ranges, arena, source, line_index, file, else_body, offset, encoding);
                }
            },
            .While => |while_stmt| try collectBodyContainingRanges(ranges, arena, source, line_index, file, while_stmt.body, offset, encoding),
            .For => |for_stmt| try collectBodyContainingRanges(ranges, arena, source, line_index, file, for_stmt.body, offset, encoding),
            .Block => |block_stmt| try collectBodyContainingRanges(ranges, arena, source, line_index, file, block_stmt.body, offset, encoding),
            else => {},
        }
    }
}

fn itemTextRange(item: compiler.ast.nodes.Item) ?compiler.TextRange {
    return switch (item) {
        .Import => |import_decl| import_decl.range,
        .Contract => |contract_decl| contract_decl.range,
        .Function => |function_decl| function_decl.range,
        .Struct => |struct_decl| struct_decl.range,
        .Bitfield => |bitfield_decl| bitfield_decl.range,
        .Enum => |enum_decl| enum_decl.range,
        .Resource => |resource_decl| resource_decl.range,
        .Trait => |trait_decl| trait_decl.range,
        .Impl => |impl_decl| impl_decl.range,
        .TypeAlias => |type_alias| type_alias.range,
        .LogDecl => |log_decl| log_decl.range,
        .ErrorDecl => |error_decl| error_decl.range,
        .Field => |field_decl| field_decl.range,
        .Constant => |constant_decl| constant_decl.range,
        .GhostBlock => |ghost_block| ghost_block.range,
        .Error => null,
    };
}

fn stmtTextRange(stmt: compiler.ast.nodes.Stmt) ?compiler.TextRange {
    return switch (stmt) {
        .Error => null,
        inline else => |statement| statement.range,
    };
}

fn rangeLargerThan(_: void, lhs: types.Range, rhs: types.Range) bool {
    return rangeSpan(lhs) > rangeSpan(rhs);
}

fn rangeSpan(range: types.Range) u64 {
    const lines = if (range.end.line >= range.start.line) range.end.line - range.start.line else 0;
    const chars = if (range.end.character >= range.start.character) range.end.character - range.start.character else 0;
    return @as(u64, lines) * 100000 + chars;
}
