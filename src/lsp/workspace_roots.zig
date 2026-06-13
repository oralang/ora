const std = @import("std");

const workspace = @import("workspace.zig");

const Allocator = std.mem.Allocator;

pub const Store = struct {
    allocator: Allocator,
    roots: std.ArrayList([]const u8) = .{},

    pub fn init(allocator: Allocator) Store {
        return .{ .allocator = allocator };
    }

    pub fn deinit(self: *Store) void {
        self.clear();
        self.roots.deinit(self.allocator);
    }

    pub fn paths(self: *const Store) []const []const u8 {
        return self.roots.items;
    }

    pub fn clear(self: *Store) void {
        for (self.roots.items) |root| {
            self.allocator.free(root);
        }
        self.roots.clearRetainingCapacity();
    }

    pub fn addInitializeRoots(self: *Store, params: anytype) !void {
        var added_root = false;

        if (params.workspaceFolders) |folders| {
            for (folders) |folder| {
                if (try self.addUri(folder.uri)) {
                    added_root = true;
                }
            }
        }

        if (!added_root) {
            if (params.rootUri) |root_uri| {
                added_root = try self.addUri(root_uri);
            }
        }

        if (!added_root) {
            const cwd = try std.fs.cwd().realpathAlloc(self.allocator, ".");
            defer self.allocator.free(cwd);
            _ = try self.addPath(cwd);
        }
    }

    pub fn addUri(self: *Store, uri: []const u8) !bool {
        const maybe_path = try workspace.fileUriToPathAlloc(self.allocator, uri);
        const path = maybe_path orelse return false;
        defer self.allocator.free(path);
        return try self.addPath(path);
    }

    pub fn addPath(self: *Store, root_path: []const u8) !bool {
        const normalized = try workspace.normalizePathAlloc(self.allocator, root_path);
        errdefer self.allocator.free(normalized);

        if (self.contains(normalized)) {
            self.allocator.free(normalized);
            return false;
        }

        try self.roots.append(self.allocator, normalized);
        return true;
    }

    pub fn contains(self: *const Store, candidate: []const u8) bool {
        for (self.roots.items) |existing| {
            if (std.mem.eql(u8, existing, candidate)) return true;
        }
        return false;
    }
};
