const std = @import("std");

pub const TextRange = struct {
    start: u32,
    end: u32,

    pub fn init(start: u32, end: u32) TextRange {
        return .{ .start = start, .end = end };
    }

    pub fn empty(offset: u32) TextRange {
        return .{ .start = offset, .end = offset };
    }

    pub fn len(self: TextRange) u32 {
        return self.end - self.start;
    }

    pub fn isEmpty(self: TextRange) bool {
        return self.start == self.end;
    }

    pub fn contains(self: TextRange, offset: u32) bool {
        return offset >= self.start and offset < self.end;
    }

    pub fn merge(a: TextRange, b: TextRange) TextRange {
        return .{
            .start = @min(a.start, b.start),
            .end = @max(a.end, b.end),
        };
    }
};

pub const FileId = defineId("FileId");
pub const PackageId = defineId("PackageId");
pub const ModuleId = defineId("ModuleId");

pub const SourceLocation = struct {
    file_id: FileId,
    range: TextRange,
};

pub const LineColumn = struct {
    line: u32,
    column: u32,
};

pub const SourceFile = struct {
    id: FileId,
    path: []const u8,
    text: []const u8,
    line_starts: []u32,
};

pub const Package = struct {
    id: PackageId,
    name: []const u8,
    modules: std.ArrayList(ModuleId),

    fn deinit(self: *Package, allocator: std.mem.Allocator) void {
        allocator.free(self.name);
        self.modules.deinit(allocator);
    }
};

pub const Module = struct {
    id: ModuleId,
    package_id: PackageId,
    file_id: FileId,
    name: []const u8,

    fn deinit(self: *Module, allocator: std.mem.Allocator) void {
        allocator.free(self.name);
    }
};

pub const SourceStore = struct {
    allocator: std.mem.Allocator,
    files: std.ArrayList(SourceFile),
    packages: std.ArrayList(Package),
    modules: std.ArrayList(Module),

    pub fn init(allocator: std.mem.Allocator) SourceStore {
        return .{
            .allocator = allocator,
            .files = .{},
            .packages = .{},
            .modules = .{},
        };
    }

    pub fn deinit(self: *SourceStore) void {
        for (self.files.items) |source_file| {
            self.allocator.free(source_file.path);
            self.allocator.free(source_file.text);
            self.allocator.free(source_file.line_starts);
        }
        self.files.deinit(self.allocator);

        for (self.packages.items) |*package_record| {
            package_record.deinit(self.allocator);
        }
        self.packages.deinit(self.allocator);

        for (self.modules.items) |*module_record| {
            module_record.deinit(self.allocator);
        }
        self.modules.deinit(self.allocator);
    }

    pub fn addFile(self: *SourceStore, path: []const u8, text: []const u8) !FileId {
        const owned_path = try self.allocator.dupe(u8, path);
        errdefer self.allocator.free(owned_path);

        const owned_text = try self.allocator.dupe(u8, text);
        errdefer self.allocator.free(owned_text);

        const line_starts = try buildLineStarts(self.allocator, owned_text);
        errdefer self.allocator.free(line_starts);

        const id = FileId.fromIndex(self.files.items.len);
        try self.files.append(self.allocator, .{
            .id = id,
            .path = owned_path,
            .text = owned_text,
            .line_starts = line_starts,
        });
        return id;
    }

    pub fn updateFile(self: *SourceStore, file_id: FileId, text: []const u8) !void {
        const owned_text = try self.allocator.dupe(u8, text);
        errdefer self.allocator.free(owned_text);

        const line_starts = try buildLineStarts(self.allocator, owned_text);
        errdefer self.allocator.free(line_starts);

        const source_file = self.fileMut(file_id);
        self.allocator.free(source_file.text);
        self.allocator.free(source_file.line_starts);
        source_file.text = owned_text;
        source_file.line_starts = line_starts;
    }

    pub fn addPackage(self: *SourceStore, name: []const u8) !PackageId {
        const owned_name = try self.allocator.dupe(u8, name);
        errdefer self.allocator.free(owned_name);

        const id = PackageId.fromIndex(self.packages.items.len);
        try self.packages.append(self.allocator, .{
            .id = id,
            .name = owned_name,
            .modules = .{},
        });
        return id;
    }

    pub fn addModule(self: *SourceStore, package_id: PackageId, file_id: FileId, name: []const u8) !ModuleId {
        const owned_name = try self.allocator.dupe(u8, name);
        errdefer self.allocator.free(owned_name);

        const id = ModuleId.fromIndex(self.modules.items.len);
        try self.modules.append(self.allocator, .{
            .id = id,
            .package_id = package_id,
            .file_id = file_id,
            .name = owned_name,
        });
        try self.packages.items[package_id.index()].modules.append(self.allocator, id);
        return id;
    }

    pub fn file(self: *const SourceStore, file_id: FileId) *const SourceFile {
        return &self.files.items[file_id.index()];
    }

    pub fn fileMut(self: *SourceStore, file_id: FileId) *SourceFile {
        return &self.files.items[file_id.index()];
    }

    pub fn package(self: *const SourceStore, package_id: PackageId) *const Package {
        return &self.packages.items[package_id.index()];
    }

    pub fn module(self: *const SourceStore, module_id: ModuleId) *const Module {
        return &self.modules.items[module_id.index()];
    }

    pub fn moduleByFile(self: *const SourceStore, file_id: FileId) ?ModuleId {
        for (self.modules.items) |module_record| {
            if (module_record.file_id == file_id) return module_record.id;
        }
        return null;
    }

    pub fn sourceText(self: *const SourceStore, file_id: FileId) []const u8 {
        return self.file(file_id).text;
    }

    pub fn lineColumn(self: *const SourceStore, location: SourceLocation) LineColumn {
        const source_file = self.file(location.file_id);
        const offset = location.range.start;
        const line_index_plus_one = std.sort.upperBound(u32, source_file.line_starts, offset, lineStartLessThan);
        const line_index: usize = if (line_index_plus_one == 0) 0 else line_index_plus_one - 1;
        const line_start = source_file.line_starts[line_index];
        return .{
            .line = @intCast(line_index + 1),
            .column = offset - line_start + 1,
        };
    }
};

fn lineStartLessThan(context: u32, value: u32) std.math.Order {
    return std.math.order(context, value);
}

pub fn defineId(comptime name: []const u8) type {
    return enum(u32) {
        _,

        const Self = @This();

        pub fn fromIndex(idx: usize) Self {
            return @enumFromInt(idx);
        }

        pub fn index(self: Self) usize {
            return @intFromEnum(self);
        }

        pub fn format(self: Self, comptime fmt: []const u8, options: std.fmt.FormatOptions, writer: anytype) !void {
            _ = fmt;
            _ = options;
            try writer.print("{s}({d})", .{ name, self.index() });
        }
    };
}

pub fn rangeOf(value: anytype) TextRange {
    return switch (value) {
        inline else => |v| v.range,
    };
}

fn buildLineStarts(allocator: std.mem.Allocator, text: []const u8) ![]u32 {
    var starts: std.ArrayList(u32) = .{};
    defer starts.deinit(allocator);

    try starts.append(allocator, 0);
    for (text, 0..) |byte, index| {
        if (byte == '\n') {
            try starts.append(allocator, @intCast(index + 1));
        }
    }
    return starts.toOwnedSlice(allocator);
}
