// ============================================================================
// Documentation Comment Extraction
// ============================================================================
//
// Extracts documentation comments from token trivia.
// Supports both line comments (///) and block comments (/** */).
//
// ALGORITHM:
//   For each token with leading trivia:
//     • If doc block (/** */): use it alone
//     • Otherwise: collect contiguous doc line comments (///)
//     • Associate with token index
//
// Only immediate preceding comments attached (whitespace allowed between lines).
//
// ============================================================================

const std = @import("std");
const lexer = @import("lexer.zig");

const Token = lexer.Token;
const TriviaPiece = lexer.TriviaPiece;
const TriviaKind = lexer.TriviaKind;

pub const DocCommentEntry = struct {
    token_index: u32,
    text: []const u8,
};

/// Extract doc comments from leading trivia for tokens that likely start declarations.
/// We attach the last contiguous run of doc comments (/// lines or a /** block */)
/// immediately preceding a token, allowing only whitespace/newlines in between.
pub fn extractDocComments(
    allocator: std.mem.Allocator,
    source: []const u8,
    tokens: []const Token,
    trivia: []const TriviaPiece,
) ![]DocCommentEntry {
    var entries = std.ArrayList(DocCommentEntry).init(allocator);
    errdefer {
        for (entries.items) |e| allocator.free(e.text);
        entries.deinit();
    }

    var i: usize = 0;
    while (i < tokens.len) : (i += 1) {
        const tok = tokens[i];
        if (tok.type == .Eof) break;

        const tstart = @as(usize, tok.leading_trivia_start);
        const tlen = @as(usize, tok.leading_trivia_len);
        if (tlen == 0) continue;

        var idx: isize = @as(isize, @intCast(tstart + tlen)) - 1;
        // Skip trailing whitespace/newlines just before token
        while (idx >= 0) : (idx -= 1) {
            const k = trivia[@as(usize, @intCast(idx))].kind;
            if (k == .Whitespace or k == .Newline) continue;
            break;
        }
        if (idx < 0) continue;

        const end_idx: usize = @as(usize, @intCast(idx));

        // If the immediate trivia is a doc block, use it alone
        if (trivia[end_idx].kind == .DocBlockComment) {
            const span = trivia[end_idx].span;
            const text = try allocator.dupe(u8, source[span.start_offset..span.end_offset]);
            try entries.append(.{ .token_index = @as(u32, @intCast(i)), .text = text });
            continue;
        }

        // Otherwise collect contiguous doc line comments upward
        var start_idx_opt: ?usize = null;
        var j: isize = @as(isize, @intCast(end_idx));
        while (j >= 0) : (j -= 1) {
            const k = trivia[@as(usize, @intCast(j))].kind;
            if (k == .DocLineComment) {
                start_idx_opt = @as(usize, @intCast(j));
                // continue to include contiguous doc lines
                continue;
            }
            if (k == .Whitespace or k == .Newline) {
                // allow interleaving whitespace/newlines between doc lines
                continue;
            }
            break;
        }
        if (start_idx_opt) |start_idx| {
            // Combine from start_idx..=end_idx in original order
            var buf = std.ArrayList(u8).init(allocator);
            defer buf.deinit();
            var k: usize = start_idx;
            while (k <= end_idx) : (k += 1) {
                const piece = trivia[k];
                switch (piece.kind) {
                    .DocLineComment, .Whitespace, .Newline => {
                        try buf.appendSlice(source[piece.span.start_offset..piece.span.end_offset]);
                    },
                    else => {},
                }
            }
            const text = try buf.toOwnedSlice();
            try entries.append(.{ .token_index = @as(u32, @intCast(i)), .text = text });
        }
    }

    return entries.toOwnedSlice();
}
