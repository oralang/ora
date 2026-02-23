const std = @import("std");

pub const ConfigError = error{
    FileNotFound,
    InvalidToml,
    MissingSchemaVersion,
    UnsupportedSchemaVersion,
    MissingTargets,
    MissingTargetName,
    MissingTargetRoot,
    InvalidTargetKind,
    OutOfMemory,
};

pub const TargetKind = enum {
    contract,
    library,
};

pub const InitArg = struct {
    name: []const u8,
    value: []const u8,

    fn deinit(self: *InitArg, allocator: std.mem.Allocator) void {
        allocator.free(self.name);
        allocator.free(self.value);
    }
};

pub const Target = struct {
    name: []const u8,
    kind: TargetKind,
    root: []const u8,
    include_paths: []const []const u8,
    init_args: []InitArg,
    output_dir: ?[]const u8,

    fn deinit(self: *Target, allocator: std.mem.Allocator) void {
        allocator.free(self.name);
        allocator.free(self.root);
        for (self.include_paths) |include_path| {
            allocator.free(include_path);
        }
        allocator.free(self.include_paths);
        for (self.init_args) |*arg| {
            arg.deinit(allocator);
        }
        allocator.free(self.init_args);
        if (self.output_dir) |output_dir| {
            allocator.free(output_dir);
        }
    }
};

pub const ProjectConfig = struct {
    schema_version: []const u8,
    compiler_output_dir: ?[]const u8,
    compiler_init_args: []InitArg,
    targets: []Target,

    pub fn deinit(self: *ProjectConfig, allocator: std.mem.Allocator) void {
        allocator.free(self.schema_version);
        if (self.compiler_output_dir) |output_dir| {
            allocator.free(output_dir);
        }
        for (self.compiler_init_args) |*arg| {
            arg.deinit(allocator);
        }
        allocator.free(self.compiler_init_args);
        for (self.targets) |*target| {
            target.deinit(allocator);
        }
        allocator.free(self.targets);
        self.* = undefined;
    }
};

pub const LoadedProjectConfig = struct {
    config_path: []const u8,
    config_dir: []const u8,
    config: ProjectConfig,

    pub fn deinit(self: *LoadedProjectConfig, allocator: std.mem.Allocator) void {
        allocator.free(self.config_path);
        allocator.free(self.config_dir);
        self.config.deinit(allocator);
        self.* = undefined;
    }
};

const Section = enum {
    top,
    compiler,
    target,
    other,
};

const TargetBuilder = struct {
    name: ?[]const u8 = null,
    kind: TargetKind = .contract,
    root: ?[]const u8 = null,
    include_paths: std.ArrayList([]const u8) = .{},
    init_args: std.ArrayList(InitArg) = .{},
    output_dir: ?[]const u8 = null,

    fn deinit(self: *TargetBuilder, allocator: std.mem.Allocator) void {
        if (self.name) |name| allocator.free(name);
        if (self.root) |root| allocator.free(root);
        for (self.include_paths.items) |include_path| {
            allocator.free(include_path);
        }
        self.include_paths.deinit(allocator);
        for (self.init_args.items) |*arg| {
            arg.deinit(allocator);
        }
        self.init_args.deinit(allocator);
        if (self.output_dir) |output_dir| allocator.free(output_dir);
    }

    fn setName(self: *TargetBuilder, allocator: std.mem.Allocator, name: []const u8) void {
        if (self.name) |old| allocator.free(old);
        self.name = name;
    }

    fn setRoot(self: *TargetBuilder, allocator: std.mem.Allocator, root: []const u8) void {
        if (self.root) |old| allocator.free(old);
        self.root = root;
    }

    fn setOutputDir(self: *TargetBuilder, allocator: std.mem.Allocator, output_dir: []const u8) void {
        if (self.output_dir) |old| allocator.free(old);
        self.output_dir = output_dir;
    }

    fn setIncludePaths(self: *TargetBuilder, allocator: std.mem.Allocator, include_paths: []const []const u8) ConfigError!void {
        for (self.include_paths.items) |include_path| {
            allocator.free(include_path);
        }
        self.include_paths.clearRetainingCapacity();

        for (include_paths) |include_path| {
            const copy = allocator.dupe(u8, include_path) catch return ConfigError.OutOfMemory;
            errdefer allocator.free(copy);
            self.include_paths.append(allocator, copy) catch return ConfigError.OutOfMemory;
        }
    }

    fn setInitArgs(self: *TargetBuilder, allocator: std.mem.Allocator, raw_init_args: []const []const u8) ConfigError!void {
        for (self.init_args.items) |*arg| {
            arg.deinit(allocator);
        }
        self.init_args.clearRetainingCapacity();

        var seen_names = std.StringHashMap(void).init(allocator);
        defer seen_names.deinit();

        for (raw_init_args) |raw_arg| {
            const parsed = try parseInitArgOwned(allocator, raw_arg);
            errdefer {
                var parsed_copy = parsed;
                parsed_copy.deinit(allocator);
            }
            if (seen_names.contains(parsed.name)) {
                return ConfigError.InvalidToml;
            }
            try seen_names.put(parsed.name, {});
            self.init_args.append(allocator, parsed) catch return ConfigError.OutOfMemory;
        }
    }

    fn build(self: *const TargetBuilder, allocator: std.mem.Allocator) ConfigError!Target {
        const name = self.name orelse return ConfigError.MissingTargetName;
        const root = self.root orelse return ConfigError.MissingTargetRoot;

        const include_paths = allocator.alloc([]const u8, self.include_paths.items.len) catch return ConfigError.OutOfMemory;
        var include_idx: usize = 0;
        errdefer {
            for (include_paths[0..include_idx]) |include_path| {
                allocator.free(include_path);
            }
            allocator.free(include_paths);
        }

        for (self.include_paths.items) |include_path| {
            include_paths[include_idx] = allocator.dupe(u8, include_path) catch return ConfigError.OutOfMemory;
            include_idx += 1;
        }

        const init_args = allocator.alloc(InitArg, self.init_args.items.len) catch return ConfigError.OutOfMemory;
        var arg_idx: usize = 0;
        errdefer {
            for (init_args[0..arg_idx]) |*arg| {
                arg.deinit(allocator);
            }
            allocator.free(init_args);
        }
        for (self.init_args.items) |arg| {
            const name_copy = allocator.dupe(u8, arg.name) catch return ConfigError.OutOfMemory;
            errdefer allocator.free(name_copy);
            const value_copy = allocator.dupe(u8, arg.value) catch return ConfigError.OutOfMemory;
            init_args[arg_idx] = .{
                .name = name_copy,
                .value = value_copy,
            };
            arg_idx += 1;
        }

        const name_copy = allocator.dupe(u8, name) catch return ConfigError.OutOfMemory;
        errdefer allocator.free(name_copy);
        const root_copy = allocator.dupe(u8, root) catch return ConfigError.OutOfMemory;
        errdefer allocator.free(root_copy);
        const output_dir_copy = if (self.output_dir) |output_dir|
            allocator.dupe(u8, output_dir) catch return ConfigError.OutOfMemory
        else
            null;
        errdefer if (output_dir_copy) |output_dir| allocator.free(output_dir);

        return .{
            .name = name_copy,
            .kind = self.kind,
            .root = root_copy,
            .include_paths = include_paths,
            .init_args = init_args,
            .output_dir = output_dir_copy,
        };
    }
};

fn fileExists(path: []const u8) bool {
    std.fs.cwd().access(path, .{}) catch return false;
    return true;
}

fn stripInlineComment(line: []const u8) []const u8 {
    var in_string = false;
    var i: usize = 0;
    while (i < line.len) : (i += 1) {
        const c = line[i];
        if (c == '"') {
            var escaped = false;
            if (i > 0 and line[i - 1] == '\\') {
                escaped = true;
            }
            if (!escaped) {
                in_string = !in_string;
            }
            continue;
        }
        if (c == '#' and !in_string) {
            return line[0..i];
        }
    }
    return line;
}

fn parseTomlStringOwned(allocator: std.mem.Allocator, value: []const u8) ConfigError![]u8 {
    const trimmed = std.mem.trim(u8, value, " \t\r\n");
    if (trimmed.len < 2 or trimmed[0] != '"' or trimmed[trimmed.len - 1] != '"') {
        return ConfigError.InvalidToml;
    }
    return allocator.dupe(u8, trimmed[1 .. trimmed.len - 1]) catch ConfigError.OutOfMemory;
}

fn parseTomlStringArrayOwned(allocator: std.mem.Allocator, value: []const u8) ConfigError![]const []const u8 {
    const trimmed = std.mem.trim(u8, value, " \t\r\n");
    if (trimmed.len < 2 or trimmed[0] != '[' or trimmed[trimmed.len - 1] != ']') {
        return ConfigError.InvalidToml;
    }

    const inner = std.mem.trim(u8, trimmed[1 .. trimmed.len - 1], " \t\r\n");
    if (inner.len == 0) {
        return allocator.alloc([]const u8, 0) catch ConfigError.OutOfMemory;
    }

    var items = std.ArrayList([]const u8){};
    defer {
        for (items.items) |entry| allocator.free(entry);
        items.deinit(allocator);
    }

    var split = std.mem.splitScalar(u8, inner, ',');
    while (split.next()) |raw_item| {
        const item = std.mem.trim(u8, raw_item, " \t\r\n");
        if (item.len == 0) {
            return ConfigError.InvalidToml;
        }
        const parsed = try parseTomlStringOwned(allocator, item);
        items.append(allocator, parsed) catch return ConfigError.OutOfMemory;
    }

    const out = allocator.alloc([]const u8, items.items.len) catch return ConfigError.OutOfMemory;
    for (items.items, 0..) |entry, idx| {
        out[idx] = entry;
    }
    items.clearRetainingCapacity();
    return out;
}

fn isAsciiIdentifierStart(ch: u8) bool {
    return std.ascii.isAlphabetic(ch) or ch == '_';
}

fn isAsciiIdentifierContinue(ch: u8) bool {
    return std.ascii.isAlphanumeric(ch) or ch == '_';
}

fn isValidIdentifier(name: []const u8) bool {
    if (name.len == 0) return false;
    if (!isAsciiIdentifierStart(name[0])) return false;
    for (name[1..]) |ch| {
        if (!isAsciiIdentifierContinue(ch)) return false;
    }
    return true;
}

fn parseInitArgOwned(allocator: std.mem.Allocator, raw: []const u8) ConfigError!InitArg {
    const trimmed = std.mem.trim(u8, raw, " \t\r\n");
    if (trimmed.len == 0) return ConfigError.InvalidToml;

    const eq_idx = std.mem.indexOfScalar(u8, trimmed, '=') orelse return ConfigError.InvalidToml;
    const name_part = std.mem.trim(u8, trimmed[0..eq_idx], " \t\r\n");
    const value_part = std.mem.trim(u8, trimmed[eq_idx + 1 ..], " \t\r\n");

    if (!isValidIdentifier(name_part) or value_part.len == 0) {
        return ConfigError.InvalidToml;
    }

    const name_copy = allocator.dupe(u8, name_part) catch return ConfigError.OutOfMemory;
    errdefer allocator.free(name_copy);
    const value_copy = allocator.dupe(u8, value_part) catch return ConfigError.OutOfMemory;

    return .{
        .name = name_copy,
        .value = value_copy,
    };
}

fn targetKindFromString(value: []const u8) ConfigError!TargetKind {
    if (std.mem.eql(u8, value, "contract")) return .contract;
    if (std.mem.eql(u8, value, "library")) return .library;
    return ConfigError.InvalidTargetKind;
}

pub fn discoverConfigPathFromStartDir(allocator: std.mem.Allocator, start_dir: []const u8) ConfigError!?[]u8 {
    var probe = allocator.dupe(u8, start_dir) catch return ConfigError.OutOfMemory;
    defer allocator.free(probe);

    while (true) {
        const ora_toml = std.fs.path.join(allocator, &.{ probe, "ora.toml" }) catch return ConfigError.OutOfMemory;
        if (fileExists(ora_toml)) {
            return ora_toml;
        }
        allocator.free(ora_toml);

        const Ora_toml = std.fs.path.join(allocator, &.{ probe, "Ora.toml" }) catch return ConfigError.OutOfMemory;
        if (fileExists(Ora_toml)) {
            return Ora_toml;
        }
        allocator.free(Ora_toml);

        const probe_real = std.fs.cwd().realpathAlloc(allocator, probe) catch break;
        defer allocator.free(probe_real);
        const parent_real = std.fs.path.dirname(probe_real) orelse break;
        if (std.mem.eql(u8, parent_real, probe_real)) {
            break;
        }

        const next_probe = std.fs.path.join(allocator, &.{ probe, ".." }) catch return ConfigError.OutOfMemory;
        allocator.free(probe);
        probe = next_probe;
    }

    return null;
}

pub fn resolvePathFromConfigDir(allocator: std.mem.Allocator, config_dir: []const u8, input_path: []const u8) ConfigError![]u8 {
    if (std.fs.path.isAbsolute(input_path) or std.mem.eql(u8, config_dir, ".")) {
        return allocator.dupe(u8, input_path) catch ConfigError.OutOfMemory;
    }
    return std.fs.path.join(allocator, &.{ config_dir, input_path }) catch ConfigError.OutOfMemory;
}

pub fn loadProjectConfigFile(allocator: std.mem.Allocator, config_path: []const u8) ConfigError!ProjectConfig {
    const source = std.fs.cwd().readFileAlloc(allocator, config_path, 1024 * 1024) catch |err| switch (err) {
        error.FileNotFound => return ConfigError.FileNotFound,
        else => return ConfigError.InvalidToml,
    };
    defer allocator.free(source);

    var schema_version: ?[]const u8 = null;
    errdefer if (schema_version) |schema| allocator.free(schema);
    var compiler_output_dir: ?[]const u8 = null;
    errdefer if (compiler_output_dir) |output_dir| allocator.free(output_dir);
    var compiler_init_args = allocator.alloc(InitArg, 0) catch return ConfigError.OutOfMemory;
    errdefer {
        for (compiler_init_args) |*arg| arg.deinit(allocator);
        allocator.free(compiler_init_args);
    }
    var pending_array_key: ?[]u8 = null;
    defer if (pending_array_key) |key| allocator.free(key);
    var pending_array_section: Section = .top;
    var pending_array_value = std.ArrayList(u8){};
    defer pending_array_value.deinit(allocator);
    var section: Section = .top;

    var target_builders = std.ArrayList(TargetBuilder){};
    defer {
        for (target_builders.items) |*target_builder| {
            target_builder.deinit(allocator);
        }
        target_builders.deinit(allocator);
    }

    var lines = std.mem.splitScalar(u8, source, '\n');
    while (lines.next()) |raw_line| {
        const no_comment = stripInlineComment(raw_line);
        const line = std.mem.trim(u8, no_comment, " \t\r\n");
        if (line.len == 0) continue;

        if (pending_array_key != null) {
            pending_array_value.appendSlice(allocator, line) catch return ConfigError.OutOfMemory;
            if (!std.mem.endsWith(u8, line, "]")) {
                continue;
            }

            const key = pending_array_key.?;
            pending_array_key = null;
            defer allocator.free(key);

            const parsed = try parseTomlStringArrayOwned(allocator, pending_array_value.items);
            defer {
                for (parsed) |item| allocator.free(item);
                allocator.free(parsed);
            }
            pending_array_value.clearRetainingCapacity();

            switch (pending_array_section) {
                .compiler => {
                    if (std.mem.eql(u8, key, "init_args")) {
                        var new_args = allocator.alloc(InitArg, parsed.len) catch return ConfigError.OutOfMemory;
                        var new_arg_idx: usize = 0;
                        errdefer {
                            for (new_args[0..new_arg_idx]) |*arg| arg.deinit(allocator);
                            allocator.free(new_args);
                        }
                        var seen_names = std.StringHashMap(void).init(allocator);
                        defer seen_names.deinit();
                        for (parsed) |item| {
                            new_args[new_arg_idx] = try parseInitArgOwned(allocator, item);
                            if (seen_names.contains(new_args[new_arg_idx].name)) {
                                return ConfigError.InvalidToml;
                            }
                            try seen_names.put(new_args[new_arg_idx].name, {});
                            new_arg_idx += 1;
                        }

                        for (compiler_init_args) |*arg| arg.deinit(allocator);
                        allocator.free(compiler_init_args);
                        compiler_init_args = new_args;
                    } else if (std.mem.eql(u8, key, "defines")) {
                        return ConfigError.InvalidToml;
                    }
                },
                .target => {
                    if (target_builders.items.len == 0) return ConfigError.InvalidToml;
                    const current = &target_builders.items[target_builders.items.len - 1];
                    if (std.mem.eql(u8, key, "include_paths")) {
                        try current.setIncludePaths(allocator, parsed);
                    } else if (std.mem.eql(u8, key, "init_args")) {
                        try current.setInitArgs(allocator, parsed);
                    } else if (std.mem.eql(u8, key, "defines")) {
                        return ConfigError.InvalidToml;
                    }
                },
                else => {},
            }
            continue;
        }

        if (std.mem.startsWith(u8, line, "[[") and std.mem.endsWith(u8, line, "]]")) {
            if (std.mem.eql(u8, line, "[[targets]]")) {
                section = .target;
                target_builders.append(allocator, .{}) catch return ConfigError.OutOfMemory;
            } else {
                section = .other;
            }
            continue;
        }

        if (std.mem.startsWith(u8, line, "[") and std.mem.endsWith(u8, line, "]")) {
            if (std.mem.eql(u8, line, "[compiler]")) {
                section = .compiler;
            } else {
                section = .other;
            }
            continue;
        }

        const eq_idx = std.mem.indexOfScalar(u8, line, '=') orelse return ConfigError.InvalidToml;
        const key = std.mem.trim(u8, line[0..eq_idx], " \t\r\n");
        const value = std.mem.trim(u8, line[eq_idx + 1 ..], " \t\r\n");

        if (std.mem.startsWith(u8, value, "[") and !std.mem.endsWith(u8, value, "]")) {
            pending_array_key = allocator.dupe(u8, key) catch return ConfigError.OutOfMemory;
            pending_array_section = section;
            pending_array_value.clearRetainingCapacity();
            pending_array_value.appendSlice(allocator, value) catch return ConfigError.OutOfMemory;
            continue;
        }

        switch (section) {
            .top => {
                if (std.mem.eql(u8, key, "schema_version")) {
                    const parsed = try parseTomlStringOwned(allocator, value);
                    if (schema_version) |old| allocator.free(old);
                    schema_version = parsed;
                }
            },
            .compiler => {
                if (std.mem.eql(u8, key, "output_dir")) {
                    const parsed = try parseTomlStringOwned(allocator, value);
                    if (compiler_output_dir) |old| allocator.free(old);
                    compiler_output_dir = parsed;
                } else if (std.mem.eql(u8, key, "init_args")) {
                    const parsed = try parseTomlStringArrayOwned(allocator, value);
                    defer {
                        for (parsed) |arg| allocator.free(arg);
                        allocator.free(parsed);
                    }

                    var new_args = allocator.alloc(InitArg, parsed.len) catch return ConfigError.OutOfMemory;
                    var new_arg_idx: usize = 0;
                    errdefer {
                        for (new_args[0..new_arg_idx]) |*arg| arg.deinit(allocator);
                        allocator.free(new_args);
                    }
                    var seen_names = std.StringHashMap(void).init(allocator);
                    defer seen_names.deinit();
                    for (parsed) |item| {
                        new_args[new_arg_idx] = try parseInitArgOwned(allocator, item);
                        if (seen_names.contains(new_args[new_arg_idx].name)) {
                            return ConfigError.InvalidToml;
                        }
                        try seen_names.put(new_args[new_arg_idx].name, {});
                        new_arg_idx += 1;
                    }

                    for (compiler_init_args) |*arg| arg.deinit(allocator);
                    allocator.free(compiler_init_args);
                    compiler_init_args = new_args;
                } else if (std.mem.eql(u8, key, "defines")) {
                    return ConfigError.InvalidToml;
                }
            },
            .target => {
                if (target_builders.items.len == 0) return ConfigError.InvalidToml;
                const current = &target_builders.items[target_builders.items.len - 1];

                if (std.mem.eql(u8, key, "name")) {
                    const parsed = try parseTomlStringOwned(allocator, value);
                    current.setName(allocator, parsed);
                } else if (std.mem.eql(u8, key, "kind")) {
                    const parsed = try parseTomlStringOwned(allocator, value);
                    defer allocator.free(parsed);
                    current.kind = try targetKindFromString(parsed);
                } else if (std.mem.eql(u8, key, "root")) {
                    const parsed = try parseTomlStringOwned(allocator, value);
                    current.setRoot(allocator, parsed);
                } else if (std.mem.eql(u8, key, "output_dir")) {
                    const parsed = try parseTomlStringOwned(allocator, value);
                    current.setOutputDir(allocator, parsed);
                } else if (std.mem.eql(u8, key, "include_paths")) {
                    const parsed = try parseTomlStringArrayOwned(allocator, value);
                    defer {
                        for (parsed) |include_path| allocator.free(include_path);
                        allocator.free(parsed);
                    }
                    try current.setIncludePaths(allocator, parsed);
                } else if (std.mem.eql(u8, key, "init_args")) {
                    const parsed = try parseTomlStringArrayOwned(allocator, value);
                    defer {
                        for (parsed) |arg| allocator.free(arg);
                        allocator.free(parsed);
                    }
                    try current.setInitArgs(allocator, parsed);
                } else if (std.mem.eql(u8, key, "defines")) {
                    return ConfigError.InvalidToml;
                }
            },
            .other => {},
        }
    }

    if (pending_array_key != null) {
        return ConfigError.InvalidToml;
    }

    const schema = schema_version orelse return ConfigError.MissingSchemaVersion;
    errdefer allocator.free(schema);

    if (!std.mem.eql(u8, schema, "0.1")) {
        return ConfigError.UnsupportedSchemaVersion;
    }

    if (target_builders.items.len == 0) {
        return ConfigError.MissingTargets;
    }

    const targets = allocator.alloc(Target, target_builders.items.len) catch return ConfigError.OutOfMemory;
    var target_idx: usize = 0;
    errdefer {
        for (targets[0..target_idx]) |*target| {
            target.deinit(allocator);
        }
        allocator.free(targets);
    }

    for (target_builders.items) |*target_builder| {
        targets[target_idx] = try target_builder.build(allocator);
        target_idx += 1;
    }

    return .{
        .schema_version = schema,
        .compiler_output_dir = compiler_output_dir,
        .compiler_init_args = compiler_init_args,
        .targets = targets,
    };
}

pub fn loadDiscoveredFromStartDir(allocator: std.mem.Allocator, start_dir: []const u8) ConfigError!?LoadedProjectConfig {
    const discovered_path = try discoverConfigPathFromStartDir(allocator, start_dir);
    if (discovered_path == null) {
        return null;
    }

    const path = discovered_path.?;
    errdefer allocator.free(path);

    var config = try loadProjectConfigFile(allocator, path);
    errdefer config.deinit(allocator);

    const dir_path = std.fs.path.dirname(path) orelse ".";
    return .{
        .config_path = path,
        .config_dir = allocator.dupe(u8, dir_path) catch return ConfigError.OutOfMemory,
        .config = config,
    };
}

pub fn loadDiscovered(allocator: std.mem.Allocator) ConfigError!?LoadedProjectConfig {
    return loadDiscoveredFromStartDir(allocator, ".");
}

pub fn findMatchingTargetIndex(
    allocator: std.mem.Allocator,
    loaded: *const LoadedProjectConfig,
    entry_file_path: []const u8,
) ConfigError!?usize {
    const entry_real = std.fs.cwd().realpathAlloc(allocator, entry_file_path) catch {
        return null;
    };
    defer allocator.free(entry_real);

    for (loaded.config.targets, 0..) |target, idx| {
        const target_path = try resolvePathFromConfigDir(allocator, loaded.config_dir, target.root);
        defer allocator.free(target_path);

        const target_real = std.fs.cwd().realpathAlloc(allocator, target_path) catch {
            continue;
        };
        defer allocator.free(target_real);

        if (std.mem.eql(u8, entry_real, target_real)) {
            return idx;
        }
    }

    return null;
}
