const std = @import("std");
const workspace = @import("workspace.zig");

const Allocator = std.mem.Allocator;

pub const Stats = struct {
    runs: usize = 0,
    files_seen: usize = 0,
    files_enqueued: usize = 0,
    skipped: usize = 0,
    limit_hits: usize = 0,
    cache_hits: usize = 0,
    cache_rebuilds: usize = 0,
};

pub const DiscoveredImporter = struct {
    uri: []u8,
    normalized_path: []u8,

    pub fn deinit(self: *DiscoveredImporter, allocator: Allocator) void {
        allocator.free(self.uri);
        allocator.free(self.normalized_path);
        self.* = undefined;
    }
};

const CacheEntry = struct {
    importers: []DiscoveredImporter,

    fn deinit(self: *CacheEntry, allocator: Allocator) void {
        for (self.importers) |*importer| importer.deinit(allocator);
        allocator.free(self.importers);
        self.* = undefined;
    }
};

pub const Cache = struct {
    allocator: Allocator,
    entries: std.StringHashMap(CacheEntry),
    stats: Stats = .{},
    max_files: usize,

    pub fn init(allocator: Allocator, max_files: usize) Cache {
        return .{
            .allocator = allocator,
            .entries = std.StringHashMap(CacheEntry).init(allocator),
            .max_files = max_files,
        };
    }

    pub fn deinit(self: *Cache) void {
        self.clear();
        self.entries.deinit();
    }

    pub fn clear(self: *Cache) void {
        var it = self.entries.iterator();
        while (it.next()) |entry| {
            self.allocator.free(entry.key_ptr.*);
            entry.value_ptr.deinit(self.allocator);
        }
        self.entries.clearRetainingCapacity();
    }

    pub fn getCached(self: *Cache, target_path: []const u8) ?[]const DiscoveredImporter {
        if (self.entries.get(target_path)) |entry| {
            self.stats.cache_hits = addSat(self.stats.cache_hits, 1);
            return entry.importers;
        }
        return null;
    }

    pub fn discoverImportersForTargetPath(
        self: *Cache,
        docs: anytype,
        workspace_roots: []const []const u8,
        target_path: []const u8,
        max_file_bytes: usize,
    ) ![]const DiscoveredImporter {
        self.stats.runs = addSat(self.stats.runs, 1);
        self.stats.cache_rebuilds = addSat(self.stats.cache_rebuilds, 1);

        var importers = std.ArrayList(DiscoveredImporter){};
        errdefer {
            for (importers.items) |*importer| importer.deinit(self.allocator);
            importers.deinit(self.allocator);
        }

        var reached_limit = false;
        var files_seen_this_run: usize = 0;
        for (workspace_roots) |root| {
            if (reached_limit) break;

            var root_dir = std.fs.openDirAbsolute(root, .{ .iterate = true }) catch {
                self.stats.skipped = addSat(self.stats.skipped, 1);
                continue;
            };
            defer root_dir.close();

            var walker = root_dir.walk(self.allocator) catch {
                self.stats.skipped = addSat(self.stats.skipped, 1);
                continue;
            };
            defer walker.deinit();

            while (try walker.next()) |entry| {
                if (entry.kind != .file) continue;

                files_seen_this_run = addSat(files_seen_this_run, 1);
                self.stats.files_seen = addSat(self.stats.files_seen, 1);
                if (files_seen_this_run > self.max_files) {
                    self.stats.limit_hits = addSat(self.stats.limit_hits, 1);
                    reached_limit = true;
                    break;
                }

                if (!std.mem.endsWith(u8, entry.path, ".ora")) continue;
                self.stats.files_enqueued = addSat(self.stats.files_enqueued, 1);

                const joined_path = try std.fs.path.join(self.allocator, &.{ root, entry.path });
                defer self.allocator.free(joined_path);

                const normalized_path = workspace.normalizePathAlloc(self.allocator, joined_path) catch {
                    self.stats.skipped = addSat(self.stats.skipped, 1);
                    continue;
                };
                defer self.allocator.free(normalized_path);

                if (std.mem.eql(u8, normalized_path, target_path)) continue;

                const importer_uri = workspace.pathToFileUri(self.allocator, normalized_path) catch {
                    self.stats.skipped = addSat(self.stats.skipped, 1);
                    continue;
                };
                defer self.allocator.free(importer_uri);

                if (docs.isOpenDocument(importer_uri)) continue;

                const source = std.fs.cwd().readFileAlloc(self.allocator, normalized_path, max_file_bytes) catch {
                    self.stats.skipped = addSat(self.stats.skipped, 1);
                    continue;
                };
                defer self.allocator.free(source);

                var resolution = workspace.resolveDocumentImports(
                    self.allocator,
                    importer_uri,
                    source,
                    .{ .workspace_roots = workspace_roots },
                ) catch {
                    self.stats.skipped = addSat(self.stats.skipped, 1);
                    continue;
                };
                defer resolution.deinit(self.allocator);

                if (!importsPath(resolution.imports, target_path)) continue;

                try docs.putColdDocument(importer_uri, normalized_path, source);
                try self.appendDiscoveredImporter(&importers, importer_uri, normalized_path);
            }
        }

        const key = try self.allocator.dupe(u8, target_path);
        errdefer self.allocator.free(key);

        const owned_importers = try importers.toOwnedSlice(self.allocator);
        errdefer {
            for (owned_importers) |*importer| importer.deinit(self.allocator);
            self.allocator.free(owned_importers);
        }

        try self.entries.put(key, .{ .importers = owned_importers });
        return self.entries.get(target_path).?.importers;
    }

    fn appendDiscoveredImporter(
        self: *Cache,
        importers: *std.ArrayList(DiscoveredImporter),
        uri: []const u8,
        normalized_path: []const u8,
    ) !void {
        const uri_copy = try self.allocator.dupe(u8, uri);
        errdefer self.allocator.free(uri_copy);

        const path_copy = try self.allocator.dupe(u8, normalized_path);
        errdefer self.allocator.free(path_copy);

        try importers.append(self.allocator, .{
            .uri = uri_copy,
            .normalized_path = path_copy,
        });
    }
};

pub fn importsPath(imports: []const workspace.ResolvedImport, target_path: []const u8) bool {
    for (imports) |import_item| {
        if (std.mem.eql(u8, import_item.resolved_path, target_path)) return true;
    }
    return false;
}

fn addSat(a: usize, b: usize) usize {
    return std.math.add(usize, a, b) catch std.math.maxInt(usize);
}
