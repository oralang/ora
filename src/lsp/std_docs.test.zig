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

test "lsp std docs: hover resolves synthetic transaction environment members" {
    const allocator = std.testing.allocator;
    const source =
        \\comptime const std = @import("std");
        \\
        \\contract Env {
        \\    pub fn caller() -> address {
        \\        return std.transaction.sender;
        \\    }
        \\}
    ;

    var cache = try token_cache.Cache.init(allocator, source);
    defer cache.deinit(allocator);

    const aliases = try std_docs.collectImportAliases(allocator, cache.tokens);
    defer freeAliases(allocator, aliases);

    var index = try std_docs.Index.init(allocator);
    defer index.deinit();

    const hover = (try std_docs.hoverAt(
        allocator,
        source,
        positionOf(source, "sender;"),
        &index,
        aliases,
    )).?;
    defer allocator.free(hover.contents);

    try std.testing.expect(std.mem.indexOf(u8, hover.contents, "std.transaction.sender: address") != null);
    try std.testing.expect(std.mem.indexOf(u8, hover.contents, "Alias of `std.msg.sender`") != null);
    try std.testing.expect(std_docs.definitionAt(source, positionOf(source, "sender;"), &index, aliases) == null);
}

test "lsp std docs: completion includes synthetic std namespaces and members" {
    const allocator = std.testing.allocator;
    const source =
        \\comptime const std = @import("std");
        \\
        \\pub fn caller() -> address {
        \\    return std.transaction.se;
        \\}
    ;

    var cache = try token_cache.Cache.init(allocator, source);
    defer cache.deinit(allocator);

    const aliases = try std_docs.collectImportAliases(allocator, cache.tokens);
    defer freeAliases(allocator, aliases);

    var index = try std_docs.Index.init(allocator);
    defer index.deinit();

    const member_candidates = try std_docs.completionCandidatesAtPosition(
        allocator,
        source,
        positionAfter(source, "se"),
        &index,
        aliases,
    );
    defer allocator.free(member_candidates);

    const sender = symbolWithName(member_candidates, "sender") orelse return error.TestExpectedEqual;
    try std.testing.expectEqualStrings("address", sender.detail.?);
    try std.testing.expect(std.mem.indexOf(u8, sender.doc_comment.?, "Alias of `std.msg.sender`") != null);

    const root_source =
        \\comptime const std = @import("std");
        \\
        \\pub fn caller() {
        \\    std.
        \\}
    ;
    var root_cache = try token_cache.Cache.init(allocator, root_source);
    defer root_cache.deinit(allocator);

    const root_aliases = try std_docs.collectImportAliases(allocator, root_cache.tokens);
    defer freeAliases(allocator, root_aliases);

    const root_candidates = try std_docs.completionCandidatesAtPosition(
        allocator,
        root_source,
        positionAfter(root_source, "std."),
        &index,
        root_aliases,
    );
    defer allocator.free(root_candidates);

    try std.testing.expect(symbolWithName(root_candidates, "transaction") != null);
    try std.testing.expect(symbolWithName(root_candidates, "constants") != null);
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

fn positionAfter(source: []const u8, needle: []const u8) frontend.Position {
    const offset = (std.mem.indexOf(u8, source, needle) orelse @panic("needle not found")) + needle.len;
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

fn symbolWithName(symbols: []const ora_root.lsp.semantic_index.Symbol, name: []const u8) ?ora_root.lsp.semantic_index.Symbol {
    for (symbols) |symbol| {
        if (std.mem.eql(u8, symbol.name, name)) return symbol;
    }
    return null;
}

fn freeAliases(allocator: std.mem.Allocator, aliases: []const std_docs.ImportAlias) void {
    for (aliases) |alias| {
        allocator.free(alias.alias);
        allocator.free(alias.logical_path);
    }
    allocator.free(aliases);
}
