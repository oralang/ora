const std = @import("std");
const ora_root = @import("ora_root");

const frontend = ora_root.lsp.frontend;
const std_docs = ora_root.lsp.std_docs;
const token_cache = ora_root.lsp.token_cache;

test "lsp std docs: hover resolves chained std module members" {
    const allocator = std.testing.allocator;
    const source =
        \\const std = @import("std");
        \\
        \\pub fn read(data: bytes) -> !u8|std.bytes.OutOfBounds {
        \\    return std.bytes.at(data, 0);
        \\}
    ;

    var cache = try token_cache.Cache.init(allocator, source);
    defer cache.deinit(allocator);

    const aliases = try std_docs.collectImportAliases(allocator, cache.tokens);
    defer {
        for (aliases) |alias| {
            allocator.free(alias.alias);
            allocator.free(alias.logical_path);
        }
        allocator.free(aliases);
    }

    try std.testing.expectEqual(@as(usize, 1), aliases.len);
    try std.testing.expectEqualStrings("std", aliases[0].alias);
    try std.testing.expectEqualStrings("std", aliases[0].logical_path);

    var index = try std_docs.Index.init(allocator);
    defer index.deinit();

    const hover = (try std_docs.hoverAt(
        allocator,
        source,
        positionOf(source, "at(data"),
        &index,
        aliases,
    )).?;
    defer allocator.free(hover.contents);
    try std.testing.expect(std.mem.indexOf(u8, hover.contents, "fn at(data: bytes, index: u256) -> !u8 | OutOfBounds") != null);
    try std.testing.expect(std.mem.indexOf(u8, hover.contents, "Returns the byte at `index`.") != null);

    const definition = std_docs.definitionAt(source, positionOf(source, "at(data"), &index, aliases).?;
    try std.testing.expectEqualStrings("embedded://std/bytes.ora", definition.uri);
    try std.testing.expectEqual(@as(u32, 12), definition.range.start.line);
    try std.testing.expectEqual(@as(u32, 3), definition.range.start.character);
    try std.testing.expectEqual(@as(u32, 5), definition.range.end.character);
}

test "lsp std docs: direct std submodule aliases resolve" {
    const allocator = std.testing.allocator;
    const source =
        \\const bytes = @import("std/bytes");
        \\
        \\pub fn read(data: bytes) -> u8 {
        \\    return bytes.at(data, 0);
        \\}
    ;

    var cache = try token_cache.Cache.init(allocator, source);
    defer cache.deinit(allocator);

    const aliases = try std_docs.collectImportAliases(allocator, cache.tokens);
    defer {
        for (aliases) |alias| {
            allocator.free(alias.alias);
            allocator.free(alias.logical_path);
        }
        allocator.free(aliases);
    }

    var index = try std_docs.Index.init(allocator);
    defer index.deinit();

    try std.testing.expectEqual(@as(usize, 1), aliases.len);
    try std.testing.expectEqualStrings("bytes", aliases[0].alias);
    try std.testing.expectEqualStrings("std/bytes", aliases[0].logical_path);

    const maybe_hover = try std_docs.hoverAt(
        allocator,
        source,
        positionOf(source, "at(data"),
        &index,
        aliases,
    );
    const hover = maybe_hover.?;
    defer allocator.free(hover.contents);

    try std.testing.expect(std.mem.indexOf(u8, hover.contents, "Returns the byte at `index`.") != null);
}

fn positionOf(source: []const u8, needle: []const u8) frontend.Position {
    const offset = std.mem.indexOf(u8, source, needle) orelse @panic("needle not found");
    var line: u32 = 0;
    var character: u32 = 0;
    for (source[0..offset]) |byte| {
        if (byte == '\n') {
            line += 1;
            character = 0;
        } else {
            character += 1;
        }
    }
    return .{ .line = line, .character = character };
}
