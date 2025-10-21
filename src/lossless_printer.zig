// ============================================================================
// Lossless Printer
// ============================================================================
//
// Reconstructs source code from tokens and trivia with perfect fidelity.
// Ensures that printed output matches the original source byte-for-byte.
//
// ALGORITHM:
//   For each token: emit leading trivia → token text → trailing trivia
//   Finally, emit any remaining EOF trivia
//
// USE CASES:
//   • Code formatting preservation
//   • Syntax tree manipulation with comment preservation
//   • Testing lexer round-trip accuracy
//
// ============================================================================

const std = @import("std");
const lexer_mod = @import("lexer.zig");

const Token = lexer_mod.Token;
const TriviaPiece = lexer_mod.TriviaPiece;

pub fn printLossless(allocator: std.mem.Allocator, source: []const u8, tokens: []const Token, trivia: []const TriviaPiece) ![]u8 {
    var out = std.ArrayList(u8).init(allocator);
    errdefer out.deinit();

    var consumed_trivia_idx: usize = 0;

    // Iterate all tokens except EOF; emit leading + token + trailing
    var i: usize = 0;
    while (i < tokens.len) : (i += 1) {
        const tok = tokens[i];
        if (tok.type == .Eof) break;

        // Leading trivia
        if (tok.leading_trivia_len > 0) {
            const tstart = @as(usize, tok.leading_trivia_start);
            const tlen = @as(usize, tok.leading_trivia_len);
            var j: usize = 0;
            while (j < tlen) : (j += 1) {
                const piece = trivia[tstart + j];
                const begin = @as(usize, piece.span.start_offset);
                const end = @as(usize, piece.span.end_offset);
                try out.appendSlice(source[begin..end]);
            }
            consumed_trivia_idx = @max(consumed_trivia_idx, tstart + tlen);
        }

        const begin = @as(usize, tok.range.start_offset);
        const end = @as(usize, tok.range.end_offset);
        try out.appendSlice(source[begin..end]);

        // Trailing trivia
        if (tok.trailing_trivia_len > 0) {
            const tstart2 = @as(usize, tok.trailing_trivia_start);
            const tlen2 = @as(usize, tok.trailing_trivia_len);
            var k: usize = 0;
            while (k < tlen2) : (k += 1) {
                const piece = trivia[tstart2 + k];
                const b2 = @as(usize, piece.span.start_offset);
                const e2 = @as(usize, piece.span.end_offset);
                try out.appendSlice(source[b2..e2]);
            }
            consumed_trivia_idx = @max(consumed_trivia_idx, tstart2 + tlen2);
        }
    }

    // Emit any remaining trailing trivia not attached to a token (e.g., at EOF)
    while (consumed_trivia_idx < trivia.len) : (consumed_trivia_idx += 1) {
        const piece = trivia[consumed_trivia_idx];
        const begin = @as(usize, piece.span.start_offset);
        const end = @as(usize, piece.span.end_offset);
        try out.appendSlice(source[begin..end]);
    }

    return out.toOwnedSlice();
}
