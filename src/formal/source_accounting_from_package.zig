//! Package-wide source-accounting inventory.
//!
//! Per-file syntax and semantic adapters intentionally use file-local IDs.
//! This adapter is the only place that globalizes them: modules are ordered by
//! canonical package-relative path, then sites/templates retain their adapter
//! canonical order. Downstream stages consume `SiteBinding`/`TemplateBinding`
//! instead of matching source text or allocator-local compiler IDs.

const std = @import("std");
const compiler = @import("ora_root").compiler;
const source = compiler.source;
const syntax = compiler.syntax;
const accounting = @import("shared/source_accounting.zig");
const from_syntax = @import("source_accounting_from_syntax.zig");
const from_sema = @import("source_accounting_from_sema.zig");

pub const SiteBinding = struct {
    local_site_id: accounting.SiteId,
    source_fact_id: ?u32,
    kind: accounting.SourceFactKind,
    site_id: accounting.SiteId,
};

pub const TemplateBinding = struct {
    local_template_id: accounting.TemplateId,
    template_id: accounting.TemplateId,
    local_owner_key: []const u8,
    activation: accounting.TemplateActivation,
};

pub const ModuleInventory = struct {
    module_id: source.ModuleId,
    file_id: source.FileId,
    source_path: []const u8,
    canonical_path: []const u8,
    site_bindings: []const SiteBinding,
    template_bindings: []const TemplateBinding,

    pub fn siteForSourceFact(
        self: ModuleInventory,
        source_fact_id: u32,
        kind: accounting.SourceFactKind,
    ) ?accounting.SiteId {
        for (self.site_bindings) |binding| {
            if (binding.source_fact_id == source_fact_id and binding.kind == kind) return binding.site_id;
        }
        return null;
    }

    pub fn templateForOwner(
        self: ModuleInventory,
        local_owner_key: []const u8,
        activation: accounting.TemplateActivation,
    ) ?accounting.TemplateId {
        for (self.template_bindings) |binding| {
            if (binding.activation == activation and
                std.mem.eql(u8, binding.local_owner_key, local_owner_key)) return binding.template_id;
        }
        return null;
    }
};

pub const Result = struct {
    arena: std.heap.ArenaAllocator,
    inventory: accounting.SourceInventory,
    modules: []const ModuleInventory,

    pub fn deinit(self: *Result) void {
        self.arena.deinit();
        self.* = undefined;
    }

    pub fn module(self: *const Result, module_id: source.ModuleId) ?ModuleInventory {
        for (self.modules) |row| if (row.module_id == module_id) return row;
        return null;
    }

    pub fn moduleForFile(self: *const Result, file_id: source.FileId) ?ModuleInventory {
        for (self.modules) |row| if (row.file_id == file_id) return row;
        return null;
    }

    pub fn moduleForSourcePath(self: *const Result, source_path: []const u8) ?ModuleInventory {
        for (self.modules) |row| {
            if (std.mem.eql(u8, row.source_path, source_path)) return row;
        }
        return null;
    }

    pub fn template(self: *const Result, template_id: accounting.TemplateId) ?accounting.OwnerTemplate {
        for (self.inventory.owner_templates) |row| if (row.id == template_id) return row;
        return null;
    }
};

const ModuleSeed = struct {
    module_id: source.ModuleId,
    file_id: source.FileId,
    canonical_path: []const u8,
};

pub fn collect(
    allocator: std.mem.Allocator,
    compiler_db: anytype,
    package_id: source.PackageId,
    root_module_id: source.ModuleId,
) !Result {
    var result: Result = .{
        .arena = std.heap.ArenaAllocator.init(allocator),
        .inventory = .{},
        .modules = &.{},
    };
    errdefer result.deinit();
    const arena = result.arena.allocator();

    const package = compiler_db.sources.package(package_id);
    const seeds = try arena.alloc(ModuleSeed, package.modules.items.len);
    for (package.modules.items, seeds) |module_id, *seed| {
        const module = compiler_db.sources.module(module_id);
        seed.* = .{
            .module_id = module_id,
            .file_id = module.file_id,
            .canonical_path = try canonicalPackagePath(arena, &compiler_db.sources, root_module_id, module_id),
        };
    }
    std.mem.sort(ModuleSeed, seeds, {}, lessModuleSeed);
    for (seeds[1..], seeds[0..seeds.len -| 1]) |current, previous| {
        if (std.mem.eql(u8, current.canonical_path, previous.canonical_path)) {
            return error.DuplicateCanonicalSourceModulePath;
        }
    }

    var syntax_results: std.ArrayList(from_syntax.Result) = .empty;
    defer {
        for (syntax_results.items) |*inventory| inventory.deinit();
        syntax_results.deinit(allocator);
    }
    var sema_results: std.ArrayList(from_sema.Result) = .empty;
    defer {
        for (sema_results.items) |*inventory| inventory.deinit();
        sema_results.deinit(allocator);
    }

    var declared_count: usize = 0;
    for (seeds) |seed| {
        const tree = try compiler_db.syntaxTree(seed.file_id);
        const inventory = try from_syntax.collect(allocator, syntax.rootNode(tree), seed.canonical_path);
        declared_count += inventory.declared_sites.len;
        try syntax_results.append(allocator, inventory);
    }
    for (seeds, syntax_results.items) |seed, syntax_inventory| {
        const ast_file = try compiler_db.astFile(seed.file_id);
        const inventory = try from_sema.collect(
            allocator,
            ast_file,
            seed.canonical_path,
            syntax_inventory.declared_sites,
        );
        try sema_results.append(allocator, inventory);
    }

    var unknown_typed_count: usize = 0;
    var typed_count: usize = 0;
    var template_count: usize = 0;
    for (syntax_results.items, sema_results.items) |syntax_inventory, sema_inventory| {
        typed_count += sema_inventory.typed_sites.len;
        for (sema_inventory.typed_sites) |site| if (site.declared_site_id == null) {
            unknown_typed_count += 1;
        };
        template_count += sema_inventory.owner_templates.len;
        _ = syntax_inventory;
    }

    const declared = try arena.alloc(accounting.DeclaredSite, declared_count);
    // The syntax and semantic inventories are independent by construction.
    // Allocate the rows the semantic adapter actually produced; padding this
    // slice to the syntax count creates uninitialized pseudo-sites precisely
    // when conservation is broken, before the kernel can report it.
    const typed = try arena.alloc(accounting.TypedSite, typed_count);
    const templates = try arena.alloc(accounting.OwnerTemplate, template_count);
    const modules = try arena.alloc(ModuleInventory, seeds.len);

    var declared_index: usize = 0;
    var typed_index: usize = 0;
    var template_index: usize = 0;
    var next_unknown_site_id: accounting.SiteId = @intCast(declared_count + 1);

    for (seeds, syntax_results.items, sema_results.items, modules) |seed, syntax_inventory, sema_inventory, *module_row| {
        // Syntax and sema assign independent file-local IDs. Keep their maps
        // separate: templates refer to sema IDs, while declared_site_id refers
        // to syntax IDs. Reusing one map is only accidentally correct when the
        // two adapters happen to produce the same order.
        const declared_bindings = try arena.alloc(SiteBinding, syntax_inventory.declared_sites.len);

        for (syntax_inventory.declared_sites, declared_bindings) |local_site, *binding| {
            const global_id: accounting.SiteId = @intCast(declared_index + 1);
            declared[declared_index] = try cloneDeclaredSite(arena, local_site, global_id);
            declared_index += 1;
            binding.* = .{
                .local_site_id = local_site.id,
                .source_fact_id = local_site.key.range_start,
                .kind = local_site.key.kind,
                .site_id = global_id,
            };
        }

        const site_bindings = try arena.alloc(SiteBinding, sema_inventory.typed_sites.len);
        for (sema_inventory.typed_sites, site_bindings) |local_site, *binding| {
            const global_id = if (local_site.declared_site_id) |declared_id|
                remapSiteId(declared_bindings, declared_id) orelse return error.UnknownPackageDeclaredSite
            else blk: {
                const id = next_unknown_site_id;
                next_unknown_site_id = std.math.add(accounting.SiteId, next_unknown_site_id, 1) catch
                    return error.SourceAccountingSiteIdOverflow;
                break :blk id;
            };
            binding.* = .{
                .local_site_id = local_site.id,
                .source_fact_id = local_site.source_fact_id,
                .kind = local_site.kind,
                .site_id = global_id,
            };
            typed[typed_index] = try cloneTypedSite(arena, local_site, global_id);
            if (local_site.declared_site_id) |declared_id| {
                typed[typed_index].declared_site_id = remapSiteId(declared_bindings, declared_id);
            }
            typed_index += 1;
        }
        std.mem.sort(SiteBinding, site_bindings, {}, lessSiteBinding);

        const template_bindings = try arena.alloc(TemplateBinding, sema_inventory.owner_templates.len);
        for (sema_inventory.owner_templates, template_bindings) |local_template, *binding| {
            const global_id: accounting.TemplateId = @intCast(template_index + 1);
            binding.* = .{
                .local_template_id = local_template.id,
                .template_id = global_id,
                .local_owner_key = try arena.dupe(u8, local_template.owner_key),
                .activation = local_template.activation,
            };
            templates[template_index] = try cloneOwnerTemplate(
                arena,
                local_template,
                global_id,
                seed.canonical_path,
                site_bindings,
            );
            template_index += 1;
        }

        module_row.* = .{
            .module_id = seed.module_id,
            .file_id = seed.file_id,
            .source_path = try arena.dupe(u8, compiler_db.sources.file(seed.file_id).path),
            .canonical_path = try arena.dupe(u8, seed.canonical_path),
            .site_bindings = site_bindings,
            .template_bindings = template_bindings,
        };
    }

    std.debug.assert(declared_index == declared.len);
    std.debug.assert(typed_index == typed.len);
    std.debug.assert(template_index == templates.len);

    std.mem.sort(accounting.TypedSite, typed, {}, struct {
        fn less(_: void, lhs: accounting.TypedSite, rhs: accounting.TypedSite) bool {
            return lhs.id < rhs.id;
        }
    }.less);
    result.inventory = .{
        .declared_sites = declared,
        .typed_sites = typed,
        .owner_templates = templates,
    };
    result.modules = modules;
    return result;
}

fn canonicalPackagePath(
    allocator: std.mem.Allocator,
    sources: *const source.SourceStore,
    root_module_id: source.ModuleId,
    module_id: source.ModuleId,
) ![]const u8 {
    const module = sources.module(module_id);
    const path = sources.file(module.file_id).path;
    if (std.mem.startsWith(u8, path, "embedded://")) {
        return std.fmt.allocPrint(allocator, "@embedded/{s}.ora", .{module.name});
    }

    const root_path = sources.file(sources.module(root_module_id).file_id).path;
    const root_absolute = try std.fs.path.resolve(allocator, &.{root_path});
    defer allocator.free(root_absolute);
    const module_absolute = try std.fs.path.resolve(allocator, &.{path});
    defer allocator.free(module_absolute);
    const root_dir = std.fs.path.dirname(root_absolute) orelse root_absolute;
    return std.fs.path.relative(allocator, ".", null, root_dir, module_absolute);
}

fn lessModuleSeed(_: void, lhs: ModuleSeed, rhs: ModuleSeed) bool {
    const path_order = std.mem.order(u8, lhs.canonical_path, rhs.canonical_path);
    if (path_order != .eq) return path_order == .lt;
    return lhs.module_id.index() < rhs.module_id.index();
}

fn lessSiteBinding(_: void, lhs: SiteBinding, rhs: SiteBinding) bool {
    if (lhs.local_site_id != rhs.local_site_id) return lhs.local_site_id < rhs.local_site_id;
    return lhs.site_id < rhs.site_id;
}

fn remapSiteId(bindings: []const SiteBinding, local_id: accounting.SiteId) ?accounting.SiteId {
    for (bindings) |binding| if (binding.local_site_id == local_id) return binding.site_id;
    return null;
}

fn cloneDeclaredSite(
    allocator: std.mem.Allocator,
    site: accounting.DeclaredSite,
    id: accounting.SiteId,
) !accounting.DeclaredSite {
    return .{
        .id = id,
        .key = try cloneSiteKey(allocator, site.key),
        .label = if (site.label) |label| try allocator.dupe(u8, label) else null,
    };
}

fn cloneTypedSite(
    allocator: std.mem.Allocator,
    site: accounting.TypedSite,
    id: accounting.SiteId,
) !accounting.TypedSite {
    return .{
        .id = id,
        .origin = site.origin,
        .kind = site.kind,
        .key = try cloneSiteKey(allocator, site.key),
        .source_fact_id = site.source_fact_id,
        .declared_site_id = site.declared_site_id,
        .derivation_id = site.derivation_id,
    };
}

fn cloneSiteKey(allocator: std.mem.Allocator, key: accounting.SiteKey) !accounting.SiteKey {
    var cloned = key;
    cloned.path = try allocator.dupe(u8, key.path);
    cloned.owner = try allocator.dupe(u8, key.owner);
    return cloned;
}

fn cloneOwnerTemplate(
    allocator: std.mem.Allocator,
    template: accounting.OwnerTemplate,
    id: accounting.TemplateId,
    canonical_path: []const u8,
    site_bindings: []const SiteBinding,
) !accounting.OwnerTemplate {
    const uses = try allocator.alloc(accounting.UseTemplate, template.uses.len);
    for (template.uses, uses) |local_use, *use| {
        use.* = local_use;
        use.site_id = remapSiteId(site_bindings, local_use.site_id) orelse return error.UnknownPackageTypedSite;
    }

    const nodes = try allocator.alloc(accounting.ControlNodeTemplate, template.control_nodes.len);
    for (template.control_nodes, nodes) |local_node, *node| {
        node.* = local_node;
        node.range = try cloneRange(allocator, local_node.range);
        node.attached_use_ordinals = try allocator.dupe(u32, local_node.attached_use_ordinals);
    }
    return .{
        .id = id,
        .owner_key = try std.fmt.allocPrint(allocator, "{s}::{s}", .{ canonical_path, template.owner_key }),
        .activation = template.activation,
        .uses = uses,
        .control_nodes = nodes,
        .control_edges = try allocator.dupe(accounting.ControlEdgeTemplate, template.control_edges),
        .entry_slot = template.entry_slot,
        .terminal_slots = try allocator.dupe(u32, template.terminal_slots),
    };
}

fn cloneRange(allocator: std.mem.Allocator, range: accounting.SourceRange) !accounting.SourceRange {
    return .{
        .file = try allocator.dupe(u8, range.file),
        .start = range.start,
        .end = range.end,
    };
}
