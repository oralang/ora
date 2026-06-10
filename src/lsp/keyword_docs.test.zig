const std = @import("std");
const ora_root = @import("ora_root");

const keyword_docs = ora_root.lsp.keyword_docs;
const lexer = ora_root.lexer;

test "keyword docs cover lexer and contextual keywords" {
    const keyword_keys = lexer.keywords.kvs.keys[0..lexer.keywords.kvs.len];
    for (keyword_keys) |keyword| {
        const doc = keyword_docs.documentation(keyword) orelse return error.TestExpectedEqual;
        try std.testing.expect(doc.len != 0);
    }

    for (keyword_docs.contextual_keywords) |keyword| {
        const doc = keyword_docs.documentation(keyword) orelse return error.TestExpectedEqual;
        try std.testing.expect(doc.len != 0);
    }
}
