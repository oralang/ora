const std = @import("std");

const Allocator = std.mem.Allocator;

pub const Node = struct {
    normalized_path: ?[]const u8,
    imports: []const []const u8,

    fn deinit(self: *Node, allocator: Allocator) void {
        if (self.normalized_path) |path| {
            allocator.free(path);
        }
        for (self.imports) |path| {
            allocator.free(path);
        }
        allocator.free(self.imports);
    }
};

pub const Graph = struct {
    allocator: Allocator,
    nodes_by_uri: std.StringHashMap(Node),
    reverse_importers: std.StringHashMap([]const []const u8),

    pub fn init(allocator: Allocator) Graph {
        return .{
            .allocator = allocator,
            .nodes_by_uri = std.StringHashMap(Node).init(allocator),
            .reverse_importers = std.StringHashMap([]const []const u8).init(allocator),
        };
    }

    pub fn deinit(self: *Graph) void {
        self.clearReverseImporters();
        self.reverse_importers.deinit();

        var it = self.nodes_by_uri.iterator();
        while (it.next()) |entry| {
            self.allocator.free(entry.key_ptr.*);
            var node = entry.value_ptr.*;
            node.deinit(self.allocator);
        }
        self.nodes_by_uri.deinit();
    }

    pub fn upsert(self: *Graph, uri: []const u8, normalized_path: ?[]const u8, imports: []const []const u8) !void {
        var node = try self.cloneNode(normalized_path, imports);
        errdefer node.deinit(self.allocator);

        const uri_copy = try self.allocator.dupe(u8, uri);
        errdefer self.allocator.free(uri_copy);

        if (self.nodes_by_uri.fetchRemove(uri)) |removed| {
            self.allocator.free(removed.key);
            var old_node = removed.value;
            old_node.deinit(self.allocator);
        }

        try self.nodes_by_uri.put(uri_copy, node);
        try self.rebuildReverseImporters();
    }

    pub fn remove(self: *Graph, uri: []const u8) !?[]u8 {
        if (self.nodes_by_uri.fetchRemove(uri)) |removed| {
            const path_copy = if (removed.value.normalized_path) |path|
                try self.allocator.dupe(u8, path)
            else
                null;

            self.allocator.free(removed.key);
            var node = removed.value;
            node.deinit(self.allocator);

            try self.rebuildReverseImporters();
            return path_copy;
        }

        return null;
    }

    pub fn getPathForUri(self: *const Graph, uri: []const u8) ?[]const u8 {
        const node = self.nodes_by_uri.get(uri) orelse return null;
        return node.normalized_path;
    }

    /// Returns importer URIs that (transitively) depend on the given path.
    /// Returned slice must be freed by caller. Elements are borrowed from graph storage.
    pub fn collectDependents(self: *const Graph, allocator: Allocator, changed_path: []const u8) ![]const []const u8 {
        var queue_paths = std.ArrayList([]const u8){};
        defer queue_paths.deinit(allocator);

        var visited_paths = std.ArrayList([]const u8){};
        defer visited_paths.deinit(allocator);

        var affected_uris = std.ArrayList([]const u8){};
        defer affected_uris.deinit(allocator);

        try queue_paths.append(allocator, changed_path);
        try visited_paths.append(allocator, changed_path);

        var read_index: usize = 0;
        while (read_index < queue_paths.items.len) : (read_index += 1) {
            const current_path = queue_paths.items[read_index];
            const importers = self.reverse_importers.get(current_path) orelse continue;

            for (importers) |importer_uri| {
                if (!containsSlice(affected_uris.items, importer_uri)) {
                    try affected_uris.append(allocator, importer_uri);
                }

                if (self.nodes_by_uri.get(importer_uri)) |node| {
                    if (node.normalized_path) |importer_path| {
                        if (!containsSlice(visited_paths.items, importer_path)) {
                            try visited_paths.append(allocator, importer_path);
                            try queue_paths.append(allocator, importer_path);
                        }
                    }
                }
            }
        }

        return try affected_uris.toOwnedSlice(allocator);
    }

    fn cloneNode(self: *const Graph, normalized_path: ?[]const u8, imports: []const []const u8) !Node {
        const path_copy = if (normalized_path) |path|
            try self.allocator.dupe(u8, path)
        else
            null;
        errdefer if (path_copy) |path| self.allocator.free(path);

        const imports_copy = try self.allocator.alloc([]const u8, imports.len);
        var copied: usize = 0;
        errdefer {
            for (imports_copy[0..copied]) |item| {
                self.allocator.free(item);
            }
            self.allocator.free(imports_copy);
        }

        for (imports) |import_path| {
            imports_copy[copied] = try self.allocator.dupe(u8, import_path);
            copied += 1;
        }

        return .{
            .normalized_path = path_copy,
            .imports = imports_copy,
        };
    }

    fn rebuildReverseImporters(self: *Graph) !void {
        self.clearReverseImporters();

        var it = self.nodes_by_uri.iterator();
        while (it.next()) |entry| {
            const importer_uri = entry.key_ptr.*;
            const node = entry.value_ptr.*;
            for (node.imports) |imported_path| {
                try self.appendReverseImporter(imported_path, importer_uri);
            }
        }
    }

    fn appendReverseImporter(self: *Graph, imported_path: []const u8, importer_uri: []const u8) !void {
        if (self.reverse_importers.getPtr(imported_path)) |importers_ptr| {
            for (importers_ptr.*) |existing| {
                if (std.mem.eql(u8, existing, importer_uri)) return;
            }

            const importer_copy = try self.allocator.dupe(u8, importer_uri);
            errdefer self.allocator.free(importer_copy);

            const old_importers = importers_ptr.*;
            const new_importers = try self.allocator.alloc([]const u8, old_importers.len + 1);
            @memcpy(new_importers[0..old_importers.len], old_importers);
            new_importers[old_importers.len] = importer_copy;
            self.allocator.free(old_importers);
            importers_ptr.* = new_importers;
            return;
        }

        const key_copy = try self.allocator.dupe(u8, imported_path);
        errdefer self.allocator.free(key_copy);

        const importer_copy = try self.allocator.dupe(u8, importer_uri);
        errdefer self.allocator.free(importer_copy);

        const importers = try self.allocator.alloc([]const u8, 1);
        importers[0] = importer_copy;

        try self.reverse_importers.put(key_copy, importers);
    }

    fn clearReverseImporters(self: *Graph) void {
        var it = self.reverse_importers.iterator();
        while (it.next()) |entry| {
            self.allocator.free(entry.key_ptr.*);
            for (entry.value_ptr.*) |importer_uri| {
                self.allocator.free(importer_uri);
            }
            self.allocator.free(entry.value_ptr.*);
        }
        self.reverse_importers.clearRetainingCapacity();
    }
};

fn containsSlice(values: []const []const u8, candidate: []const u8) bool {
    for (values) |value| {
        if (std.mem.eql(u8, value, candidate)) return true;
    }
    return false;
}
