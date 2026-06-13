const std = @import("std");
const testing = std.testing;
const ora_root = @import("ora_root");

const rename = ora_root.lsp.rename;

test "lsp rename: identifier validation" {
    try testing.expect(rename.isValidIdentifier("new_name"));
    try testing.expect(rename.isValidIdentifier("newName2"));

    try testing.expect(!rename.isValidIdentifier(""));
    try testing.expect(!rename.isValidIdentifier("2name"));
    try testing.expect(!rename.isValidIdentifier("new-name"));
}
