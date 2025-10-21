// ============================================================================
// Documentation Attachment
// ============================================================================
//
// Attaches extracted doc comments to their corresponding AST nodes.
// Matches doc comments to declarations based on token positions.
//
// ALGORITHM:
//   1. Extract doc comments with token indices
//   2. Build map: token_index → doc_text
//   3. For each AST node, find first token and attach its doc comment
//
// TOKEN MATCHING:
//   Doc comments attach to the token immediately following them,
//   ensuring they associate with the right declaration.
//
// ============================================================================

const std = @import("std");
const ora = @import("root.zig");
const ast = ora.ast;
const lexer = ora.lexer;
const Doc = @import("doc_comments.zig");

pub const NodeDoc = struct { span: ast.SourceSpan, text: []const u8 };

inline fn findFirstTokenIndexInSpan(tokens: []const lexer.Token, span: ast.SourceSpan) ?usize {
    var idx: usize = 0;
    while (idx < tokens.len) : (idx += 1) {
        const t = tokens[idx];
        if (t.type == .Eof) break;
        if (t.range.start_offset >= span.start_offset and t.range.start_offset < span.end_offset) {
            return idx;
        }
    }
    return null;
}

/// Attach doc comments to top-level declarations by matching the first token inside the node span
pub fn attachTopLevelDocs(
    allocator: std.mem.Allocator,
    source: []const u8,
    tokens: []const lexer.Token,
    trivia: []const lexer.TriviaPiece,
    nodes: []const ast.AstNode,
) ![]NodeDoc {
    const entries = try Doc.extractDocComments(allocator, source, tokens, trivia);
    defer {
        // We will duplicate doc text into NodeDoc; free original entries' text
        for (entries) |e| allocator.free(e.text);
        allocator.free(entries);
    }

    var out = std.ArrayList(NodeDoc).init(allocator);
    errdefer {
        for (out.items) |nd| allocator.free(nd.text);
        out.deinit();
    }

    // (helper moved to file scope)

    // Build a map from token_index to doc text slice
    var tok_to_doc = std.AutoHashMap(u32, []const u8).init(allocator);
    defer {
        var it = tok_to_doc.iterator();
        while (it.next()) |kv| allocator.free(kv.value_ptr.*);
        tok_to_doc.deinit();
    }
    for (entries) |e| {
        // store a duplicate owned slice
        const copy = try allocator.dupe(u8, e.text);
        try tok_to_doc.put(e.token_index, copy);
    }

    for (nodes) |n| {
        const span = switch (n) {
            .Function => |f| f.span,
            .StructDecl => |s| s.span,
            .EnumDecl => |e| e.span,
            .LogDecl => |l| l.span,
            .Contract => |c| c.span,
            .ErrorDecl => |e| e.span,
            else => continue,
        };
        if (findFirstTokenIndexInSpan(tokens, span)) |tidx| {
            if (tok_to_doc.get(@as(u32, @intCast(tidx)))) |text| {
                // move ownership of text into out list
                try out.append(.{ .span = span, .text = text });
                // remove from map so we don't free twice in defer
                _ = tok_to_doc.remove(@as(u32, @intCast(tidx)));
            }
        }
    }

    return out.toOwnedSlice();
}
