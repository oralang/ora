// ============================================================================
// Import Resolution and Validation
// ============================================================================
//
// Phase 2 scope:
// - Relative import resolution (./, ../)
// - Package/workspace import resolution (name/path)
// - Canonical module IDs
// - Import cycle detection
// - Deterministic dependency-first module order
//
// Notes:
// - Builtin imports like "std" are ignored in graph output.
// - Resolver options provide workspace/include roots for package imports.
//
// ============================================================================

const std = @import("std");
const lib = @import("ora_lib");

pub const ImportValidationError = error{
    ImportTargetNotFound,
    ImportCycleDetected,
    ImportAliasRequired,
    DuplicateImportAlias,
    RelativeImportMustIncludeOraExtension,
    InvalidImportSpecifier,
    PackageRootConflict,
    ParseFailed,
    OutOfMemory,
};

pub const ModuleKind = enum {
    file,
    package,
};

pub const ResolverOptions = struct {
    include_roots: []const []const u8 = &.{},
    workspace_roots: []const []const u8 = &.{},
};

pub const ResolvedModule = struct {
    kind: ModuleKind,
    canonical_id: []const u8,
    resolved_path: []const u8,
    package_name: ?[]const u8 = null,
    package_module_path: ?[]const u8 = null,
    imports: []ResolvedImport = &.{},

    fn deinit(self: *ResolvedModule, allocator: std.mem.Allocator) void {
        allocator.free(self.canonical_id);
        allocator.free(self.resolved_path);
        if (self.package_name) |package_name| {
            allocator.free(package_name);
        }
        if (self.package_module_path) |package_module_path| {
            allocator.free(package_module_path);
        }
        for (self.imports) |*import_info| {
            import_info.deinit(allocator);
        }
        allocator.free(self.imports);
    }
};

pub const ImportGraph = struct {
    // Dependencies-first deterministic module order.
    modules: []ResolvedModule,
    entry_canonical_id: []const u8,

    pub fn deinit(self: *ImportGraph, allocator: std.mem.Allocator) void {
        for (self.modules) |*module| {
            module.deinit(allocator);
        }
        allocator.free(self.modules);
        allocator.free(self.entry_canonical_id);
        self.* = undefined;
    }
};

pub const ResolvedImport = struct {
    alias: []const u8,
    specifier: []const u8,
    target_canonical_id: []const u8,

    fn deinit(self: *ResolvedImport, allocator: std.mem.Allocator) void {
        allocator.free(self.alias);
        allocator.free(self.specifier);
        allocator.free(self.target_canonical_id);
    }
};

const VisitState = enum {
    visiting,
    done,
};

const ModuleDescriptor = struct {
    kind: ModuleKind,
    canonical_id: []const u8,
    resolved_path: []const u8,
    package_name: ?[]const u8 = null,
    package_module_path: ?[]const u8 = null,

    fn deinit(self: *ModuleDescriptor, allocator: std.mem.Allocator) void {
        allocator.free(self.canonical_id);
        allocator.free(self.resolved_path);
        if (self.package_name) |package_name| {
            allocator.free(package_name);
        }
        if (self.package_module_path) |package_module_path| {
            allocator.free(package_module_path);
        }
    }
};

const ModuleRecord = struct {
    kind: ModuleKind,
    canonical_id: []const u8,
    resolved_path: []const u8,
    package_name: ?[]const u8 = null,
    package_module_path: ?[]const u8 = null,
    dependencies: std.ArrayList([]const u8),
    imports: std.ArrayList(ResolvedImport),

    fn initFromDescriptor(descriptor: ModuleDescriptor) ModuleRecord {
        return .{
            .kind = descriptor.kind,
            .canonical_id = descriptor.canonical_id,
            .resolved_path = descriptor.resolved_path,
            .package_name = descriptor.package_name,
            .package_module_path = descriptor.package_module_path,
            .dependencies = .{},
            .imports = .{},
        };
    }

    fn addDependency(self: *ModuleRecord, allocator: std.mem.Allocator, dep_canonical_id: []const u8) ImportValidationError!void {
        for (self.dependencies.items) |existing_dep| {
            if (std.mem.eql(u8, existing_dep, dep_canonical_id)) {
                return;
            }
        }
        self.dependencies.append(allocator, dep_canonical_id) catch {
            return ImportValidationError.OutOfMemory;
        };
    }

    fn deinit(self: *ModuleRecord, allocator: std.mem.Allocator) void {
        self.dependencies.deinit(allocator);
        for (self.imports.items) |*import_info| {
            import_info.deinit(allocator);
        }
        self.imports.deinit(allocator);
        allocator.free(self.canonical_id);
        allocator.free(self.resolved_path);
        if (self.package_name) |package_name| {
            allocator.free(package_name);
        }
        if (self.package_module_path) |package_module_path| {
            allocator.free(package_module_path);
        }
    }

    fn addImport(
        self: *ModuleRecord,
        allocator: std.mem.Allocator,
        alias: []const u8,
        specifier: []const u8,
        target_canonical_id: []const u8,
    ) ImportValidationError!void {
        for (self.imports.items) |existing| {
            if (std.mem.eql(u8, existing.alias, alias)) {
                if (std.mem.eql(u8, existing.target_canonical_id, target_canonical_id)) return;
                std.log.warn("Duplicate import alias '{s}': maps to both '{s}' and '{s}'", .{
                    alias, existing.target_canonical_id, target_canonical_id,
                });
                return ImportValidationError.DuplicateImportAlias;
            }
        }

        const alias_copy = allocator.dupe(u8, alias) catch return ImportValidationError.OutOfMemory;
        errdefer allocator.free(alias_copy);
        const specifier_copy = allocator.dupe(u8, specifier) catch return ImportValidationError.OutOfMemory;
        errdefer allocator.free(specifier_copy);
        const target_copy = allocator.dupe(u8, target_canonical_id) catch return ImportValidationError.OutOfMemory;
        errdefer allocator.free(target_copy);

        self.imports.append(allocator, .{
            .alias = alias_copy,
            .specifier = specifier_copy,
            .target_canonical_id = target_copy,
        }) catch return ImportValidationError.OutOfMemory;
    }
};

const PackageSpecifier = struct {
    package_name: []const u8,
    module_path: []const u8,
};

const Resolver = struct {
    allocator: std.mem.Allocator,
    options: ResolverOptions,
    states: std.StringHashMap(VisitState),
    modules: std.StringHashMap(*ModuleRecord),
    package_bindings: std.StringHashMap([]const u8),
    stack: std.ArrayList([]const u8),
    entry_canonical_id: ?[]const u8,

    fn init(allocator: std.mem.Allocator, options: ResolverOptions) Resolver {
        return .{
            .allocator = allocator,
            .options = options,
            .states = std.StringHashMap(VisitState).init(allocator),
            .modules = std.StringHashMap(*ModuleRecord).init(allocator),
            .package_bindings = std.StringHashMap([]const u8).init(allocator),
            .stack = .{},
            .entry_canonical_id = null,
        };
    }

    fn deinit(self: *Resolver) void {
        self.states.deinit();
        self.stack.deinit(self.allocator);

        var pkg_it = self.package_bindings.iterator();
        while (pkg_it.next()) |entry| {
            self.allocator.free(entry.key_ptr.*);
            self.allocator.free(entry.value_ptr.*);
        }
        self.package_bindings.deinit();

        var module_it = self.modules.iterator();
        while (module_it.next()) |entry| {
            const record = entry.value_ptr.*;
            record.deinit(self.allocator);
            self.allocator.destroy(record);
        }
        self.modules.deinit();
    }

    fn isRelativeSpecifier(path: []const u8) bool {
        return std.mem.startsWith(u8, path, "./") or std.mem.startsWith(u8, path, "../");
    }

    fn lessThanStrings(_: void, a: []const u8, b: []const u8) bool {
        return std.mem.lessThan(u8, a, b);
    }

    fn reportCycle(self: *Resolver, cycle_target: []const u8) void {
        var start_idx: usize = 0;
        for (self.stack.items, 0..) |entry, i| {
            if (std.mem.eql(u8, entry, cycle_target)) {
                start_idx = i;
                break;
            }
        }

        std.log.warn("Import cycle detected:", .{});
        for (self.stack.items[start_idx..]) |entry| {
            std.log.warn("  -> {s}", .{self.displayModule(entry)});
        }
        std.log.warn("  -> {s}", .{self.displayModule(cycle_target)});
    }

    fn displayModule(self: *Resolver, canonical_id: []const u8) []const u8 {
        if (self.modules.get(canonical_id)) |record| {
            return record.resolved_path;
        }
        return canonical_id;
    }

    fn buildEntryDescriptor(self: *Resolver, entry_file_path: []const u8) ImportValidationError!ModuleDescriptor {
        const resolved_path = std.fs.cwd().realpathAlloc(self.allocator, entry_file_path) catch |err| {
            std.log.warn("Entry file not found: '{s}' ({s})", .{ entry_file_path, @errorName(err) });
            return ImportValidationError.ImportTargetNotFound;
        };

        const canonical_id = std.fmt.allocPrint(self.allocator, "file:{s}", .{resolved_path}) catch {
            self.allocator.free(resolved_path);
            return ImportValidationError.OutOfMemory;
        };

        return .{
            .kind = .file,
            .canonical_id = canonical_id,
            .resolved_path = resolved_path,
        };
    }

    fn ensureModule(self: *Resolver, descriptor: ModuleDescriptor) ImportValidationError!*ModuleRecord {
        if (self.modules.get(descriptor.canonical_id)) |existing| {
            var owned = descriptor;
            owned.deinit(self.allocator);
            return existing;
        }

        const record = self.allocator.create(ModuleRecord) catch {
            var owned = descriptor;
            owned.deinit(self.allocator);
            return ImportValidationError.OutOfMemory;
        };

        record.* = ModuleRecord.initFromDescriptor(descriptor);
        self.modules.put(record.canonical_id, record) catch {
            record.deinit(self.allocator);
            self.allocator.destroy(record);
            return ImportValidationError.OutOfMemory;
        };

        return record;
    }

    fn visitRecord(self: *Resolver, record: *ModuleRecord) ImportValidationError!void {
        if (self.states.get(record.canonical_id)) |state| {
            switch (state) {
                .done => return,
                .visiting => {
                    self.reportCycle(record.canonical_id);
                    return ImportValidationError.ImportCycleDetected;
                },
            }
        }

        try self.states.put(record.canonical_id, .visiting);
        try self.stack.append(self.allocator, record.canonical_id);
        defer _ = self.stack.pop();

        try self.scanModuleImports(record);
        try self.states.put(record.canonical_id, .done);
    }

    fn scanModuleImports(self: *Resolver, module: *ModuleRecord) ImportValidationError!void {
        const source = std.fs.cwd().readFileAlloc(self.allocator, module.resolved_path, 1024 * 1024) catch |err| {
            std.log.warn("Failed to read module '{s}': {s}", .{ module.resolved_path, @errorName(err) });
            return ImportValidationError.ImportTargetNotFound;
        };
        defer self.allocator.free(source);

        var lex = lib.Lexer.init(self.allocator, source);
        defer lex.deinit();

        const tokens = lex.scanTokens() catch |err| {
            std.log.warn("Failed to lex module '{s}': {s}", .{ module.resolved_path, @errorName(err) });
            return ImportValidationError.ParseFailed;
        };
        defer self.allocator.free(tokens);

        var arena = lib.ast_arena.AstArena.init(self.allocator);
        defer arena.deinit();

        var parser = lib.Parser.init(tokens, &arena);
        const nodes = parser.parse() catch |err| {
            std.log.warn("Failed to parse module '{s}' while resolving imports: {s}", .{ module.resolved_path, @errorName(err) });
            return ImportValidationError.ParseFailed;
        };

        for (nodes) |node| {
            if (node != .Import) continue;

            const import_decl = node.Import;
            const specifier = import_decl.path;

            if (import_decl.alias == null) {
                std.log.warn("Import alias required in '{s}' for '{s}'. Use: const name = @import(\"...\");", .{ module.resolved_path, specifier });
                return ImportValidationError.ImportAliasRequired;
            }

            const dependency_descriptor_opt = try self.resolveImportSpecifier(module, specifier);
            if (dependency_descriptor_opt) |dependency_descriptor| {
                const dependency_record = try self.ensureModule(dependency_descriptor);
                const alias = import_decl.alias.?;
                try module.addImport(self.allocator, alias, specifier, dependency_record.canonical_id);
                try module.addDependency(self.allocator, dependency_record.canonical_id);
                try self.visitRecord(dependency_record);
            }
        }
    }

    fn resolveImportSpecifier(self: *Resolver, importer: *const ModuleRecord, specifier: []const u8) ImportValidationError!?ModuleDescriptor {
        if (std.mem.eql(u8, specifier, "std")) {
            return null;
        }

        if (isRelativeSpecifier(specifier)) {
            return try self.resolveRelativeImport(importer, specifier);
        }

        return try self.resolvePackageImport(importer, specifier);
    }

    fn resolveRelativeImport(self: *Resolver, importer: *const ModuleRecord, specifier: []const u8) ImportValidationError!ModuleDescriptor {
        if (!std.mem.endsWith(u8, specifier, ".ora")) {
            std.log.warn("Relative import must include '.ora' extension: '{s}' in '{s}'", .{ specifier, importer.resolved_path });
            return ImportValidationError.RelativeImportMustIncludeOraExtension;
        }

        const module_dir = std.fs.path.dirname(importer.resolved_path) orelse ".";
        const joined = std.fs.path.join(self.allocator, &.{ module_dir, specifier }) catch {
            return ImportValidationError.OutOfMemory;
        };
        defer self.allocator.free(joined);

        const resolved_path = std.fs.cwd().realpathAlloc(self.allocator, joined) catch |err| {
            std.log.warn("Import target not found: '{s}' in module '{s}' ({s})", .{ specifier, importer.resolved_path, @errorName(err) });
            return ImportValidationError.ImportTargetNotFound;
        };

        const canonical_id = std.fmt.allocPrint(self.allocator, "file:{s}", .{resolved_path}) catch {
            self.allocator.free(resolved_path);
            return ImportValidationError.OutOfMemory;
        };

        return .{
            .kind = .file,
            .canonical_id = canonical_id,
            .resolved_path = resolved_path,
        };
    }

    fn parsePackageSpecifier(specifier: []const u8) ImportValidationError!PackageSpecifier {
        const slash_index = std.mem.indexOfScalar(u8, specifier, '/') orelse return ImportValidationError.InvalidImportSpecifier;
        if (slash_index == 0 or slash_index + 1 >= specifier.len) {
            return ImportValidationError.InvalidImportSpecifier;
        }

        const package_name = specifier[0..slash_index];
        const module_path = specifier[slash_index + 1 ..];
        if (package_name.len == 0 or module_path.len == 0) {
            return ImportValidationError.InvalidImportSpecifier;
        }

        return .{
            .package_name = package_name,
            .module_path = module_path,
        };
    }

    fn bindPackageRoot(self: *Resolver, package_name: []const u8, package_root_path: []const u8) ImportValidationError!void {
        if (self.package_bindings.get(package_name)) |existing_root| {
            if (!std.mem.eql(u8, existing_root, package_root_path)) {
                std.log.warn("Package root conflict for '{s}': '{s}' vs '{s}'", .{ package_name, existing_root, package_root_path });
                return ImportValidationError.PackageRootConflict;
            }
            return;
        }

        const owned_key = self.allocator.dupe(u8, package_name) catch return ImportValidationError.OutOfMemory;
        errdefer self.allocator.free(owned_key);

        const owned_value = self.allocator.dupe(u8, package_root_path) catch return ImportValidationError.OutOfMemory;
        errdefer self.allocator.free(owned_value);

        self.package_bindings.put(owned_key, owned_value) catch return ImportValidationError.OutOfMemory;
    }

    fn resolvePackageImport(self: *Resolver, importer: *const ModuleRecord, specifier: []const u8) ImportValidationError!ModuleDescriptor {
        const parsed = try parsePackageSpecifier(specifier);
        const module_path_with_extension = if (std.mem.endsWith(u8, parsed.module_path, ".ora"))
            self.allocator.dupe(u8, parsed.module_path) catch return ImportValidationError.OutOfMemory
        else
            std.fmt.allocPrint(self.allocator, "{s}.ora", .{parsed.module_path}) catch return ImportValidationError.OutOfMemory;
        defer self.allocator.free(module_path_with_extension);

        if (self.options.workspace_roots.len > 0) {
            if (try self.tryResolvePackageInRoots(parsed, module_path_with_extension, self.options.workspace_roots)) |descriptor| {
                return descriptor;
            }
        }

        if (self.options.include_roots.len > 0) {
            if (try self.tryResolvePackageInRoots(parsed, module_path_with_extension, self.options.include_roots)) |descriptor| {
                return descriptor;
            }
        }

        // Default root when no explicit roots are provided.
        if (self.options.workspace_roots.len == 0 and self.options.include_roots.len == 0) {
            const default_roots = [_][]const u8{"."};
            if (try self.tryResolvePackageInRoots(parsed, module_path_with_extension, default_roots[0..])) |descriptor| {
                return descriptor;
            }
        }

        std.log.warn("Package import target not found: '{s}' referenced in '{s}'", .{ specifier, importer.resolved_path });
        return ImportValidationError.ImportTargetNotFound;
    }

    fn tryResolvePackageInRoots(
        self: *Resolver,
        parsed: PackageSpecifier,
        module_path_with_extension: []const u8,
        roots: []const []const u8,
    ) ImportValidationError!?ModuleDescriptor {
        for (roots) |root| {
            const workspace_root = std.fs.cwd().realpathAlloc(self.allocator, root) catch {
                continue;
            };
            defer self.allocator.free(workspace_root);

            const package_root_joined = std.fs.path.join(self.allocator, &.{ workspace_root, parsed.package_name }) catch {
                return ImportValidationError.OutOfMemory;
            };
            defer self.allocator.free(package_root_joined);

            const package_root = std.fs.cwd().realpathAlloc(self.allocator, package_root_joined) catch {
                continue;
            };
            defer self.allocator.free(package_root);

            const module_joined = std.fs.path.join(self.allocator, &.{ package_root, module_path_with_extension }) catch {
                return ImportValidationError.OutOfMemory;
            };
            defer self.allocator.free(module_joined);

            const resolved_module = std.fs.cwd().realpathAlloc(self.allocator, module_joined) catch {
                continue;
            };
            errdefer self.allocator.free(resolved_module);

            try self.bindPackageRoot(parsed.package_name, package_root);

            const canonical_id = std.fmt.allocPrint(self.allocator, "pkg:{s}@workspace/{s}", .{ parsed.package_name, module_path_with_extension }) catch {
                self.allocator.free(resolved_module);
                return ImportValidationError.OutOfMemory;
            };

            const package_name = self.allocator.dupe(u8, parsed.package_name) catch {
                self.allocator.free(canonical_id);
                self.allocator.free(resolved_module);
                return ImportValidationError.OutOfMemory;
            };

            const package_module_path = self.allocator.dupe(u8, module_path_with_extension) catch {
                self.allocator.free(package_name);
                self.allocator.free(canonical_id);
                self.allocator.free(resolved_module);
                return ImportValidationError.OutOfMemory;
            };

            return ModuleDescriptor{
                .kind = .package,
                .canonical_id = canonical_id,
                .resolved_path = resolved_module,
                .package_name = package_name,
                .package_module_path = package_module_path,
            };
        }

        return null;
    }

    fn collectOrderedRecords(
        self: *Resolver,
        canonical_id: []const u8,
        visited: *std.StringHashMap(void),
        ordered: *std.ArrayList(*ModuleRecord),
    ) ImportValidationError!void {
        if (visited.contains(canonical_id)) return;

        visited.put(canonical_id, {}) catch return ImportValidationError.OutOfMemory;

        const module = self.modules.get(canonical_id) orelse {
            std.log.warn("Internal error: missing module record for '{s}'", .{canonical_id});
            return ImportValidationError.ParseFailed;
        };

        const dep_ids = self.allocator.dupe([]const u8, module.dependencies.items) catch {
            return ImportValidationError.OutOfMemory;
        };
        defer self.allocator.free(dep_ids);
        std.sort.heap([]const u8, dep_ids, {}, lessThanStrings);

        for (dep_ids) |dep_id| {
            try self.collectOrderedRecords(dep_id, visited, ordered);
        }

        ordered.append(self.allocator, module) catch return ImportValidationError.OutOfMemory;
    }

    fn buildGraph(self: *Resolver) ImportValidationError!ImportGraph {
        const entry_canonical_id = self.entry_canonical_id orelse {
            return ImportValidationError.ParseFailed;
        };

        var visited = std.StringHashMap(void).init(self.allocator);
        defer visited.deinit();

        var ordered_records = std.ArrayList(*ModuleRecord){};
        defer ordered_records.deinit(self.allocator);

        try self.collectOrderedRecords(entry_canonical_id, &visited, &ordered_records);

        const modules = self.allocator.alloc(ResolvedModule, ordered_records.items.len) catch {
            return ImportValidationError.OutOfMemory;
        };

        var idx: usize = 0;
        errdefer {
            for (modules[0..idx]) |*module| {
                module.deinit(self.allocator);
            }
            self.allocator.free(modules);
        }

        for (ordered_records.items) |record| {
            const imports_copy = self.allocator.alloc(ResolvedImport, record.imports.items.len) catch return ImportValidationError.OutOfMemory;
            var import_idx: usize = 0;
            errdefer {
                for (imports_copy[0..import_idx]) |*import_info| {
                    import_info.deinit(self.allocator);
                }
                self.allocator.free(imports_copy);
            }
            for (record.imports.items) |import_info| {
                imports_copy[import_idx] = .{
                    .alias = self.allocator.dupe(u8, import_info.alias) catch return ImportValidationError.OutOfMemory,
                    .specifier = self.allocator.dupe(u8, import_info.specifier) catch return ImportValidationError.OutOfMemory,
                    .target_canonical_id = self.allocator.dupe(u8, import_info.target_canonical_id) catch return ImportValidationError.OutOfMemory,
                };
                import_idx += 1;
            }

            modules[idx] = .{
                .kind = record.kind,
                .canonical_id = self.allocator.dupe(u8, record.canonical_id) catch return ImportValidationError.OutOfMemory,
                .resolved_path = self.allocator.dupe(u8, record.resolved_path) catch return ImportValidationError.OutOfMemory,
                .package_name = if (record.package_name) |package_name|
                    self.allocator.dupe(u8, package_name) catch return ImportValidationError.OutOfMemory
                else
                    null,
                .package_module_path = if (record.package_module_path) |package_module_path|
                    self.allocator.dupe(u8, package_module_path) catch return ImportValidationError.OutOfMemory
                else
                    null,
                .imports = imports_copy,
            };
            idx += 1;
        }

        return .{
            .modules = modules,
            .entry_canonical_id = self.allocator.dupe(u8, entry_canonical_id) catch return ImportValidationError.OutOfMemory,
        };
    }
};

pub fn resolveImportGraph(
    allocator: std.mem.Allocator,
    entry_file_path: []const u8,
    options: ResolverOptions,
) ImportValidationError!ImportGraph {
    var resolver = Resolver.init(allocator, options);
    defer resolver.deinit();

    const entry_descriptor = try resolver.buildEntryDescriptor(entry_file_path);
    const entry_record = try resolver.ensureModule(entry_descriptor);
    resolver.entry_canonical_id = entry_record.canonical_id;

    try resolver.visitRecord(entry_record);
    return try resolver.buildGraph();
}

pub fn validateNormalImportsWithOptions(
    allocator: std.mem.Allocator,
    entry_file_path: []const u8,
    options: ResolverOptions,
) ImportValidationError!void {
    var graph = try resolveImportGraph(allocator, entry_file_path, options);
    defer graph.deinit(allocator);
}

pub fn validateNormalImports(allocator: std.mem.Allocator, entry_file_path: []const u8) ImportValidationError!void {
    return validateNormalImportsWithOptions(allocator, entry_file_path, .{});
}
