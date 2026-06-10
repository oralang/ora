const std = @import("std");
const testing = std.testing;
const ora_root = @import("ora_root");

const definition = ora_root.lsp.definition;
const frontend = ora_root.lsp.frontend;
const test_analysis = @import("test_analysis.zig");

fn cachedDefinition(source: []const u8, position: frontend.Position) !?definition.Definition {
    var fixture: test_analysis.TestAnalysis = undefined;
    try fixture.init(testing.allocator, source);
    defer fixture.deinit();

    return definition.definitionAtCached(&fixture.analysis, source, position);
}

fn cachedCrossFileDefinition(
    source: []const u8,
    position: frontend.Position,
    cross_file: definition.CrossFileContext,
) !?definition.Definition {
    var fixture: test_analysis.TestAnalysis = undefined;
    try fixture.init(testing.allocator, source);
    defer fixture.deinit();

    return try definition.definitionAtCachedCrossFile(&fixture.analysis, source, position, cross_file);
}

fn positionOfNth(source: []const u8, needle: []const u8, nth: usize) !frontend.Position {
    var from: usize = 0;
    var seen: usize = 0;
    while (true) {
        const rel = std.mem.indexOfPos(u8, source, from, needle) orelse return error.TestExpectedEqual;
        if (seen == nth) {
            return byteIndexToPosition(source, rel);
        }
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

fn positionOfWithOffset(source: []const u8, needle: []const u8, offset: usize) !frontend.Position {
    const start = std.mem.indexOf(u8, source, needle) orelse return error.TestExpectedEqual;
    return byteIndexToPosition(source, start + offset);
}

test "lsp definition: resolves top-level function call" {
    const source =
        \\pub fn main() -> u256 { return helper(1); }
        \\pub fn helper(x: u256) -> u256 { return x; }
    ;

    const query = try positionOfNth(source, "helper", 0);
    const expected = try positionOfNth(source, "helper", 1);

    const maybe_def = try cachedDefinition(source, query);
    try testing.expect(maybe_def != null);

    const def = maybe_def.?;
    try testing.expectEqual(expected.line, def.range.start.line);
    try testing.expectEqual(expected.character, def.range.start.character);
}

test "lsp definition: resolves parameter usage" {
    const source =
        \\pub fn amountOf(amount: u256) -> u256 {
        \\    return amount;
        \\}
    ;

    const query = try positionOfWithOffset(source, "return amount", 7);
    const expected = try positionOfWithOffset(source, "amount: u256", 0);

    const maybe_def = try cachedDefinition(source, query);
    try testing.expect(maybe_def != null);

    const def = maybe_def.?;
    try testing.expectEqual(expected.line, def.range.start.line);
    try testing.expectEqual(expected.character, def.range.start.character);
}

test "lsp definition: resolves local variable usage" {
    const source =
        \\pub fn run() -> u256 {
        \\    let amount: u256 = 42;
        \\    return amount;
        \\}
    ;

    const query = try positionOfWithOffset(source, "return amount", 7);
    const expected = try positionOfWithOffset(source, "let amount", 4);

    const maybe_def = try cachedDefinition(source, query);
    try testing.expect(maybe_def != null);

    const def = maybe_def.?;
    try testing.expectEqual(expected.line, def.range.start.line);
    try testing.expectEqual(expected.character, def.range.start.character);
}

test "lsp definition: resolves contract member function call" {
    const source =
        \\contract Wallet {
        \\    pub fn deposit(amount: u256) -> u256 { return amount; }
        \\    pub fn execute() -> u256 { return deposit(1); }
        \\}
    ;

    const query = try positionOfWithOffset(source, "return deposit(1)", 7);
    const expected = try positionOfWithOffset(source, "fn deposit(", 3);

    const maybe_def = try cachedDefinition(source, query);
    try testing.expect(maybe_def != null);

    const def = maybe_def.?;
    try testing.expectEqual(expected.line, def.range.start.line);
    try testing.expectEqual(expected.character, def.range.start.character);
}

test "lsp definition: declaration resolves to itself" {
    const source = "pub fn helper() -> u256 { return 1; }";
    const query = try positionOfNth(source, "helper", 0);

    const maybe_def = try cachedDefinition(source, query);
    try testing.expect(maybe_def != null);

    const def = maybe_def.?;
    try testing.expectEqual(query.line, def.range.start.line);
    try testing.expectEqual(query.character, def.range.start.character);
}

test "lsp definition: unknown symbol returns null" {
    const source = "pub fn run() -> u256 { return missing; }";
    const query = try positionOfNth(source, "missing", 0);

    const maybe_def = try cachedDefinition(source, query);
    try testing.expect(maybe_def == null);
}

test "lsp definition: parse failure returns null" {
    const source = "@import(\"std\");";
    const query = try positionOfNth(source, "import", 0);

    const maybe_def = try cachedDefinition(source, query);
    try testing.expect(maybe_def == null);
}

test "lsp definition: import alias resolves to target file with cross-file context" {
    const source =
        \\const math = @import("./math.ora");
        \\pub fn run() -> u256 { return math.add(1); }
    ;

    const query = try positionOfNth(source, "math", 2);

    const bindings = [_]definition.ImportBinding{.{
        .alias = "math",
        .target_uri = "file:///project/math.ora",
    }};
    const cross_file = definition.CrossFileContext{ .bindings = &bindings };

    const maybe_def = try cachedCrossFileDefinition(source, query, cross_file);
    try testing.expect(maybe_def != null);

    const def = maybe_def.?;
    try testing.expect(def.uri != null);
    try testing.expectEqualStrings("file:///project/math.ora", def.uri.?);
    try testing.expectEqual(@as(u32, 0), def.range.start.line);
}

test "lsp definition: cross-file fallback does not raw-parse imported members" {
    const source =
        \\const math = @import("./math.ora");
        \\pub fn run() -> u256 { return math.add(1); }
    ;

    const query = try positionOfWithOffset(source, "math.add", 5);

    const bindings = [_]definition.ImportBinding{.{
        .alias = "math",
        .target_uri = "file:///project/math.ora",
    }};
    const cross_file = definition.CrossFileContext{ .bindings = &bindings };

    const maybe_def = try cachedCrossFileDefinition(source, query, cross_file);
    try testing.expect(maybe_def == null);
}

test "lsp definition: cached cross-file analysis resolves import alias" {
    const source =
        \\const math = @import("./math.ora");
        \\pub fn run() -> u256 { return math.add(1); }
    ;

    const query = try positionOfWithOffset(source, "math.add", 0);

    const bindings = [_]definition.ImportBinding{.{
        .alias = "math",
        .target_uri = "file:///project/math.ora",
    }};
    const cross_file = definition.CrossFileContext{ .bindings = &bindings };

    const maybe_def = try cachedCrossFileDefinition(source, query, cross_file);
    try testing.expect(maybe_def != null);

    const def = maybe_def.?;
    try testing.expect(def.uri != null);
    try testing.expectEqualStrings("file:///project/math.ora", def.uri.?);
}

test "lsp definition: without cross-file context, import alias stays in-file" {
    const source =
        \\const math = @import("./math.ora");
        \\pub fn run() -> u256 { return math.add(1); }
    ;

    const query = try positionOfNth(source, "math", 2);

    const maybe_def = try cachedDefinition(source, query);
    try testing.expect(maybe_def != null);
    try testing.expect(maybe_def.?.uri == null);
}
