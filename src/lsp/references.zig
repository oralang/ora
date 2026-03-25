const std = @import("std");
const lexer = @import("ora_lexer");
const definition = @import("definition.zig");
const frontend = @import("frontend.zig");

const Allocator = std.mem.Allocator;

pub fn referencesAt(
    allocator: Allocator,
    source: []const u8,
    position: frontend.Position,
    include_declaration: bool,
) ![]frontend.Range {
    var analysis = (try definition.Analysis.init(allocator, source)) orelse
        return try allocator.alloc(frontend.Range, 0);
    defer analysis.deinit();

    const target_range = blk: {
        const def = definition.definitionAtCached(&analysis, source, position) orelse
            return try allocator.alloc(frontend.Range, 0);
        break :blk def.range;
    };

    var lex = try lexer.Lexer.initWithConfig(allocator, source, lexer.LexerConfig.development());
    defer lex.deinit();

    const tokens = lex.scanTokens() catch
        return try allocator.alloc(frontend.Range, 0);
    defer allocator.free(tokens);

    var ranges = std.ArrayList(frontend.Range){};
    errdefer ranges.deinit(allocator);

    for (tokens) |token| {
        if (token.type != .Identifier) continue;

        const token_range = tokenSelectionRange(token);
        const maybe_def = definition.definitionAtCached(&analysis, source, token_range.start);
        if (maybe_def == null) continue;

        if (!rangesEqual(maybe_def.?.range, target_range)) continue;
        if (!include_declaration and rangesEqual(token_range, target_range)) continue;

        try appendUniqueRange(allocator, &ranges, token_range);
    }

    if (include_declaration) {
        try appendUniqueRange(allocator, &ranges, target_range);
    }

    return ranges.toOwnedSlice(allocator);
}

fn tokenSelectionRange(token: lexer.Token) frontend.Range {
    const start_line = if (token.line > 0) token.line - 1 else 0;
    const start_char = if (token.column > 0) token.column - 1 else 0;

    const lexeme_len = std.math.cast(u32, token.lexeme.len) orelse std.math.maxInt(u32);
    const end_char = std.math.add(u32, start_char, lexeme_len) catch std.math.maxInt(u32);

    return .{
        .start = .{ .line = start_line, .character = start_char },
        .end = .{ .line = start_line, .character = end_char },
    };
}

fn appendUniqueRange(
    allocator: Allocator,
    ranges: *std.ArrayList(frontend.Range),
    range: frontend.Range,
) !void {
    for (ranges.items) |existing| {
        if (rangesEqual(existing, range)) return;
    }
    try ranges.append(allocator, range);
}

fn rangesEqual(a: frontend.Range, b: frontend.Range) bool {
    return a.start.line == b.start.line and
        a.start.character == b.start.character and
        a.end.line == b.end.line and
        a.end.character == b.end.character;
}
