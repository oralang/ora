const std = @import("std");
const testing = std.testing;
const ora_root = @import("ora_root");
const ora_types = @import("ora_types");

const completion = ora_root.lsp.completion;
const frontend = ora_root.lsp.frontend;
const keyword_docs = ora_root.lsp.keyword_docs;
const lexer = ora_root.lexer;
const refinements = ora_types.refinement_semantics;
const test_analysis = @import("test_analysis.zig");

fn hasLabel(items: []const completion.Item, label: []const u8) bool {
    for (items) |item| {
        if (std.mem.eql(u8, item.label, label)) return true;
    }
    return false;
}

fn itemWithLabel(items: []const completion.Item, label: []const u8) ?completion.Item {
    for (items) |item| {
        if (std.mem.eql(u8, item.label, label)) return item;
    }
    return null;
}

fn positionAfterNth(source: []const u8, needle: []const u8, nth: usize) !frontend.Position {
    var from: usize = 0;
    var seen: usize = 0;
    while (true) {
        const rel = std.mem.indexOfPos(u8, source, from, needle) orelse return error.TestExpectedEqual;
        if (seen == nth) return byteIndexToPosition(source, rel + needle.len);
        seen += 1;
        from = rel + needle.len;
    }
}

fn byteIndexToPosition(source: []const u8, byte_index: usize) frontend.Position {
    var line: u32 = 0;
    var character: u32 = 0;
    for (source[0..byte_index]) |byte| {
        if (byte == '\n') {
            line += 1;
            character = 0;
        } else {
            character += 1;
        }
    }
    return .{ .line = line, .character = character };
}

fn completionItems(source: []const u8, position: frontend.Position, trigger_char: ?[]const u8) ![]completion.Item {
    var index = try test_analysis.semanticIndex(testing.allocator, source);
    defer index.deinit(testing.allocator);
    return completion.completionAtIndex(testing.allocator, source, position, trigger_char, &index);
}

test "lsp completion: includes keyword suggestions by prefix" {
    const source =
        \\pub fn run() -> u256 {
        \\    ret
        \\}
    ;

    const position = try positionAfterNth(source, "ret", 0);

    const items = try completionItems(source, position, null);
    defer completion.deinitItems(testing.allocator, items);

    try testing.expect(hasLabel(items, "return"));
}

test "lsp completion: directive argument position offers exactly the callHint states" {
    const source =
        \\contract C {
        \\    storage var x: u256;
        \\    pub fn bump() {
        \\        @callHint(
        \\        x = 1;
        \\    }
        \\}
    ;

    const position = try positionAfterNth(source, "@callHint(", 0);

    const items = try completionItems(source, position, null);
    defer completion.deinitItems(testing.allocator, items);

    try testing.expectEqual(@as(usize, 4), items.len);
    try testing.expect(hasLabel(items, "likely"));
    try testing.expect(hasLabel(items, "unlikely"));
    try testing.expect(hasLabel(items, "cold"));
    try testing.expect(hasLabel(items, "none"));
}

test "lsp completion: directive argument prefix narrows the state list" {
    const source =
        \\contract C {
        \\    storage var x: u256;
        \\    pub fn bump() {
        \\        @callHint(co
        \\        x = 1;
        \\    }
        \\}
    ;

    const position = try positionAfterNth(source, "@callHint(co", 0);

    const items = try completionItems(source, position, null);
    defer completion.deinitItems(testing.allocator, items);

    try testing.expectEqual(@as(usize, 1), items.len);
    try testing.expect(hasLabel(items, "cold"));
}

test "lsp completion: includes symbols from semantic index" {
    const source =
        \\pub fn helper(x: u256) -> u256 { return x; }
        \\pub fn run() -> u256 { return hel; }
    ;

    const position = try positionAfterNth(source, "hel", 0);

    const items = try completionItems(source, position, null);
    defer completion.deinitItems(testing.allocator, items);

    try testing.expect(hasLabel(items, "helper"));
}

test "lsp completion: marks inline function details" {
    const source =
        \\contract Math {
        \\    inline fn choose(mode: u256) -> u256 { return mode; }
        \\    pub fn run(mode: u256) -> u256 { return cho; }
        \\}
    ;

    const position = try positionAfterNth(source, "cho", 1);

    const items = try completionItems(source, position, null);
    defer completion.deinitItems(testing.allocator, items);

    const choose = itemWithLabel(items, "choose") orelse return error.TestExpectedEqual;
    try testing.expectEqual(completion.Kind.method, choose.kind);
    try testing.expect(choose.detail != null);
    try testing.expectEqualStrings("inline (mode: u256) -> u256", choose.detail.?);
}

test "lsp completion: indexed path uses caller-owned semantic index" {
    const source =
        \\pub fn helper(x: u256) -> u256 { return x; }
        \\pub fn run() -> u256 { return hel; }
    ;

    var index = try test_analysis.semanticIndex(testing.allocator, source);
    defer index.deinit(testing.allocator);

    const position = try positionAfterNth(source, "hel", 0);

    const items = try completion.completionAtIndex(testing.allocator, source, position, null, &index);
    defer completion.deinitItems(testing.allocator, items);

    try testing.expect(hasLabel(items, "helper"));
}

test "lsp completion: non-matching prefix returns empty" {
    const source =
        \\pub fn helper(x: u256) -> u256 { return x; }
        \\pub fn run() -> u256 { return zzz; }
    ;

    const position = try positionAfterNth(source, "zzz", 0);

    const items = try completionItems(source, position, null);
    defer completion.deinitItems(testing.allocator, items);

    try testing.expectEqual(@as(usize, 0), items.len);
}

test "lsp completion: includes current language keywords and types" {
    const source =
        \\pub fn run() {
        \\
        \\}
    ;

    const position: frontend.Position = .{ .line = 1, .character = 0 };

    const items = try completionItems(source, position, null);
    defer completion.deinitItems(testing.allocator, items);

    try testing.expect(hasLabel(items, "match"));
    try testing.expect(hasLabel(items, "type"));
    try testing.expect(hasLabel(items, "tstore"));
    try testing.expect(hasLabel(items, "staticcall"));
    try testing.expect(hasLabel(items, "errors"));
    try testing.expect(hasLabel(items, "bytes"));
    try testing.expect(hasLabel(items, "slice"));
    try testing.expect(hasLabel(items, "result"));
}

test "lsp completion: includes builtin functions and Resource type" {
    const source =
        \\pub fn run(value: u64) -> u256 {
        \\    return @cast(u256, value);
        \\}
    ;

    const position = try positionAfterNth(source, "@c", 0);

    const items = try completionItems(source, position, null);
    defer completion.deinitItems(testing.allocator, items);

    const cast = itemWithLabel(items, "cast") orelse return error.TestExpectedEqual;
    try testing.expectEqual(completion.Kind.function, cast.kind);
    try testing.expect(cast.detail != null);
    try testing.expectEqualStrings("@cast(T, value) -> T", cast.detail.?);
    try testing.expect(cast.documentation != null);
    try testing.expect(std.mem.indexOf(u8, cast.documentation.?, "checked conversion rules") != null);
}

test "lsp completion: includes resource builtins and Resource type" {
    const source =
        "pub fn run() {\n" ++
        "    \n" ++
        "}\n";

    const position: frontend.Position = .{ .line = 1, .character = 4 };

    const items = try completionItems(source, position, null);
    defer completion.deinitItems(testing.allocator, items);

    inline for (.{ "amount", "move", "create", "destroy", "Resource" }) |label| {
        try testing.expect(hasLabel(items, label));
    }

    const resource = itemWithLabel(items, "Resource") orelse return error.TestExpectedEqual;
    try testing.expectEqual(completion.Kind.type_alias, resource.kind);
    try testing.expect(resource.documentation != null);
    try testing.expect(std.mem.indexOf(u8, resource.documentation.?, "@move") != null);
}

test "lsp completion: offers every lexer and contextual keyword" {
    const source =
        \\pub fn run() {
        \\
        \\}
    ;

    const position: frontend.Position = .{ .line = 1, .character = 0 };

    const items = try completionItems(source, position, null);
    defer completion.deinitItems(testing.allocator, items);

    const keyword_keys = lexer.keywords.kvs.keys[0..lexer.keywords.kvs.len];
    for (keyword_keys) |keyword| {
        try testing.expect(hasLabel(items, keyword));
    }
    for (keyword_docs.contextual_keywords) |keyword| {
        try testing.expect(hasLabel(items, keyword));
    }
}

test "lsp completion: includes registry-backed refinement types" {
    const source =
        \\pub fn run() {
        \\
        \\}
    ;

    const position: frontend.Position = .{ .line = 1, .character = 0 };

    const items = try completionItems(source, position, null);
    defer completion.deinitItems(testing.allocator, items);

    for (refinements.entries) |entry| try testing.expect(hasLabel(items, entry.name));
    const min_value = itemWithLabel(items, "MinValue") orelse return error.TestExpectedEqual;
    try testing.expectEqual(completion.Kind.type_alias, min_value.kind);
    try testing.expect(min_value.detail != null);
    try testing.expect(std.mem.indexOf(u8, min_value.detail.?, "MinValue<T, MIN>") != null);
    try testing.expect(min_value.documentation != null);
    try testing.expect(std.mem.indexOf(u8, min_value.documentation.?, "greater than or equal") != null);
}
