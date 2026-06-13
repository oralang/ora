const std = @import("std");
const ora_root = @import("ora_root");

const workspace = ora_root.lsp.workspace;
const workspace_discovery = ora_root.lsp.workspace_discovery;

const testing = std.testing;

fn pathFromTmpAlloc(allocator: std.mem.Allocator, tmp: std.testing.TmpDir, rel_path: []const u8) ![]u8 {
    return std.fmt.allocPrint(allocator, ".zig-cache/tmp/{s}/{s}", .{ tmp.sub_path, rel_path });
}

const FakeDocs = struct {
    allocator: std.mem.Allocator,
    open_uri: ?[]const u8 = null,
    put_count: usize = 0,
    last_uri: ?[]u8 = null,
    last_path: ?[]u8 = null,

    fn deinit(self: *FakeDocs) void {
        if (self.last_uri) |uri| self.allocator.free(uri);
        if (self.last_path) |path| self.allocator.free(path);
        self.* = undefined;
    }

    pub fn isOpenDocument(self: *const FakeDocs, uri: []const u8) bool {
        return if (self.open_uri) |open| std.mem.eql(u8, open, uri) else false;
    }

    pub fn putColdDocument(self: *FakeDocs, uri: []const u8, normalized_path: []const u8, source: []const u8) !void {
        try testing.expect(source.len > 0);
        self.put_count += 1;

        if (self.last_uri) |old| self.allocator.free(old);
        if (self.last_path) |old| self.allocator.free(old);
        self.last_uri = try self.allocator.dupe(u8, uri);
        self.last_path = try self.allocator.dupe(u8, normalized_path);
    }
};

test "lsp workspace discovery: importsPath detects direct target import" {
    const imports = [_]workspace.ResolvedImport{
        .{
            .specifier = "./a.ora",
            .alias = "a",
            .resolved_path = "/workspace/a.ora",
        },
        .{
            .specifier = "./target.ora",
            .alias = "target",
            .resolved_path = "/workspace/target.ora",
        },
    };

    try testing.expect(workspace_discovery.importsPath(&imports, "/workspace/target.ora"));
    try testing.expect(!workspace_discovery.importsPath(&imports, "/workspace/missing.ora"));
}

test "lsp workspace discovery: discovers unopened importers and caches result" {
    const allocator = testing.allocator;
    var tmp = testing.tmpDir(.{});
    defer tmp.cleanup();

    try tmp.dir.makePath("project");
    try tmp.dir.writeFile(.{
        .sub_path = "project/target.ora",
        .data = "contract Target { pub fn called() {} }",
    });
    try tmp.dir.writeFile(.{
        .sub_path = "project/importer.ora",
        .data = "const target = @import(\"./target.ora\"); contract Importer {}",
    });
    try tmp.dir.writeFile(.{
        .sub_path = "project/other.ora",
        .data = "contract Other {}",
    });

    const target_rel = try pathFromTmpAlloc(allocator, tmp, "project/target.ora");
    defer allocator.free(target_rel);
    const target_path = try std.fs.cwd().realpathAlloc(allocator, target_rel);
    defer allocator.free(target_path);

    const root_rel = try pathFromTmpAlloc(allocator, tmp, "project");
    defer allocator.free(root_rel);
    const root = try std.fs.cwd().realpathAlloc(allocator, root_rel);
    defer allocator.free(root);

    var docs = FakeDocs{ .allocator = allocator };
    defer docs.deinit();

    var cache = workspace_discovery.Cache.init(allocator, 64);
    defer cache.deinit();

    const roots = [_][]const u8{root};
    const importers = try cache.discoverImportersForTargetPath(&docs, roots[0..], target_path, 1024 * 1024);

    try testing.expectEqual(@as(usize, 1), importers.len);
    try testing.expectEqualStrings("importer.ora", std.fs.path.basename(importers[0].normalized_path));
    try testing.expectEqual(@as(usize, 1), docs.put_count);
    try testing.expectEqual(@as(usize, 1), cache.stats.runs);
    try testing.expectEqual(@as(usize, 1), cache.stats.cache_rebuilds);
    try testing.expect(cache.stats.files_seen >= 3);
    try testing.expect(cache.stats.files_enqueued >= 3);

    const cached = cache.getCached(target_path) orelse return error.ExpectedCachedImporters;
    try testing.expectEqual(@as(usize, 1), cached.len);
    try testing.expectEqual(@as(usize, 1), cache.stats.cache_hits);
}
