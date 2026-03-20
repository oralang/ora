const std = @import("std");
const ast = @import("../ast/mod.zig");
const comptime_eval = @import("../comptime/mod.zig").compiler_ast_eval;
const diagnostics = @import("../diagnostics/mod.zig");
const hir = @import("../hir/mod.zig");
const sema = @import("../sema/mod.zig");
const source = @import("../source/mod.zig");
const syntax = @import("../syntax/mod.zig");

const TypeCheckCache = struct {
    entries: std.AutoHashMap(u64, *sema.TypeCheckResult),
    in_progress: std.AutoHashMap(u64, void),
    sentinels: std.AutoHashMap(u64, *sema.TypeCheckResult),

    fn init(allocator: std.mem.Allocator) TypeCheckCache {
        return .{
            .entries = std.AutoHashMap(u64, *sema.TypeCheckResult).init(allocator),
            .in_progress = std.AutoHashMap(u64, void).init(allocator),
            .sentinels = std.AutoHashMap(u64, *sema.TypeCheckResult).init(allocator),
        };
    }

    fn deinit(self: *TypeCheckCache, allocator: std.mem.Allocator) void {
        self.clear(allocator);
        self.entries.deinit();
        self.in_progress.deinit();
        self.sentinels.deinit();
    }

    fn clear(self: *TypeCheckCache, allocator: std.mem.Allocator) void {
        var iterator = self.entries.valueIterator();
        while (iterator.next()) |result| {
            result.*.deinit();
            allocator.destroy(result.*);
        }
        var sentinel_iterator = self.sentinels.valueIterator();
        while (sentinel_iterator.next()) |result| {
            result.*.deinit();
            allocator.destroy(result.*);
        }
        self.entries.clearRetainingCapacity();
        self.in_progress.clearRetainingCapacity();
        self.sentinels.clearRetainingCapacity();
    }

    fn lookup(self: *const TypeCheckCache, key: sema.TypeCheckKey) ?*const sema.TypeCheckResult {
        return self.entries.get(typeCheckCacheKey(key));
    }

    fn lookupSentinel(self: *const TypeCheckCache, key: sema.TypeCheckKey) ?*const sema.TypeCheckResult {
        return self.sentinels.get(typeCheckCacheKey(key));
    }
};

const VerificationCache = struct {
    entries: std.AutoHashMap(u64, *sema.VerificationFactsResult),

    fn init(allocator: std.mem.Allocator) VerificationCache {
        return .{ .entries = std.AutoHashMap(u64, *sema.VerificationFactsResult).init(allocator) };
    }

    fn deinit(self: *VerificationCache, allocator: std.mem.Allocator) void {
        self.clear(allocator);
        self.entries.deinit();
    }

    fn clear(self: *VerificationCache, allocator: std.mem.Allocator) void {
        var iterator = self.entries.valueIterator();
        while (iterator.next()) |result| {
            result.*.deinit();
            allocator.destroy(result.*);
        }
        self.entries.clearRetainingCapacity();
    }

    fn lookup(self: *const VerificationCache, key: sema.VerificationFactsKey) ?*const sema.VerificationFactsResult {
        return self.entries.get(verificationCacheKey(key));
    }
};

pub const CompilerDb = struct {
    allocator: std.mem.Allocator,
    sources: source.SourceStore,
    syntax_slots: std.ArrayList(?*syntax.ParseResult),
    ast_slots: std.ArrayList(?*ast.LowerResult),
    module_graph_slots: std.ArrayList(?*sema.ModuleGraphResult),
    item_index_slots: std.ArrayList(?*sema.ItemIndexResult),
    resolution_slots: std.ArrayList(?*sema.NameResolutionResult),
    typecheck_slots: std.ArrayList(TypeCheckCache),
    consteval_slots: std.ArrayList(?*sema.ConstEvalResult),
    consteval_in_progress: std.ArrayList(bool),
    consteval_sentinel_slots: std.ArrayList(?*sema.ConstEvalResult),
    verification_slots: std.ArrayList(VerificationCache),
    module_verification_slots: std.ArrayList(?*sema.ModuleVerificationFactsResult),
    hir_slots: std.ArrayList(?*hir.LoweringResult),

    pub fn init(allocator: std.mem.Allocator) CompilerDb {
        return .{
            .allocator = allocator,
            .sources = source.SourceStore.init(allocator),
            .syntax_slots = .{},
            .ast_slots = .{},
            .module_graph_slots = .{},
            .item_index_slots = .{},
            .resolution_slots = .{},
            .typecheck_slots = .{},
            .consteval_slots = .{},
            .consteval_in_progress = .{},
            .consteval_sentinel_slots = .{},
            .verification_slots = .{},
            .module_verification_slots = .{},
            .hir_slots = .{},
        };
    }

    pub fn deinit(self: *CompilerDb) void {
        deinitSlots(self.allocator, &self.syntax_slots);
        deinitSlots(self.allocator, &self.ast_slots);
        deinitSlots(self.allocator, &self.module_graph_slots);
        deinitSlots(self.allocator, &self.item_index_slots);
        deinitSlots(self.allocator, &self.resolution_slots);
        deinitCacheSlots(self.allocator, &self.typecheck_slots);
        deinitSlots(self.allocator, &self.consteval_slots);
        deinitSlots(self.allocator, &self.consteval_sentinel_slots);
        self.consteval_in_progress.deinit(self.allocator);
        deinitCacheSlots(self.allocator, &self.verification_slots);
        deinitSlots(self.allocator, &self.module_verification_slots);
        deinitSlots(self.allocator, &self.hir_slots);
        self.sources.deinit();
    }

    pub fn addSourceFile(self: *CompilerDb, path: []const u8, text: []const u8) !source.FileId {
        const file_id = try self.sources.addFile(path, text);
        try ensureSlots(self.allocator, &self.syntax_slots, file_id.index() + 1);
        try ensureSlots(self.allocator, &self.ast_slots, file_id.index() + 1);
        return file_id;
    }

    pub fn addPackage(self: *CompilerDb, name: []const u8) !source.PackageId {
        const package_id = try self.sources.addPackage(name);
        try ensureSlots(self.allocator, &self.module_graph_slots, package_id.index() + 1);
        return package_id;
    }

    pub fn addModule(self: *CompilerDb, package_id: source.PackageId, file_id: source.FileId, name: []const u8) !source.ModuleId {
        const module_id = try self.sources.addModule(package_id, file_id, name);
        const required = module_id.index() + 1;
        try ensureSlots(self.allocator, &self.item_index_slots, required);
        try ensureSlots(self.allocator, &self.resolution_slots, required);
        try self.typecheck_slots.append(self.allocator, TypeCheckCache.init(self.allocator));
        try ensureSlots(self.allocator, &self.consteval_slots, required);
        try self.consteval_in_progress.append(self.allocator, false);
        try ensureSlots(self.allocator, &self.consteval_sentinel_slots, required);
        try self.verification_slots.append(self.allocator, VerificationCache.init(self.allocator));
        try ensureSlots(self.allocator, &self.module_verification_slots, required);
        try ensureSlots(self.allocator, &self.hir_slots, required);
        return module_id;
    }

    pub fn updateSourceFile(self: *CompilerDb, file_id: source.FileId, text: []const u8) !void {
        try self.sources.updateFile(file_id, text);
        self.invalidateFile(file_id);
    }

    pub fn sourceText(self: *const CompilerDb, file_id: source.FileId) []const u8 {
        return self.sources.sourceText(file_id);
    }

    pub fn syntaxTree(self: *CompilerDb, file_id: source.FileId) !*const syntax.SyntaxTree {
        const result = try self.syntaxResult(file_id);
        return &result.tree;
    }

    pub fn syntaxDiagnostics(self: *CompilerDb, file_id: source.FileId) !*const diagnostics.DiagnosticList {
        const result = try self.syntaxResult(file_id);
        return &result.diagnostics;
    }

    pub fn astFile(self: *CompilerDb, file_id: source.FileId) !*const ast.AstFile {
        const result = try self.astResult(file_id);
        return &result.file;
    }

    pub fn astDiagnostics(self: *CompilerDb, file_id: source.FileId) !*const diagnostics.DiagnosticList {
        const result = try self.astResult(file_id);
        return &result.diagnostics;
    }

    pub fn moduleGraph(self: *CompilerDb, package_id: source.PackageId) !*const sema.ModuleGraphResult {
        const slot = &self.module_graph_slots.items[package_id.index()];
        if (slot.* == null) {
            const package = self.sources.package(package_id);
            var inputs: std.ArrayList(sema.ModuleGraphInput) = .{};
            defer inputs.deinit(self.allocator);
            for (package.modules.items) |module_id| {
                const module = self.sources.module(module_id);
                try inputs.append(self.allocator, .{
                    .module_id = module.id,
                    .file_id = module.file_id,
                    .path = module.name,
                    .file_path = self.sources.file(module.file_id).path,
                    .ast_file = try self.astFile(module.file_id),
                });
            }
            const result = try self.allocator.create(sema.ModuleGraphResult);
            result.* = try sema.buildModuleGraph(self.allocator, package_id, inputs.items);
            slot.* = result;
        }
        return slot.*.?;
    }

    pub fn itemIndex(self: *CompilerDb, module_id: source.ModuleId) !*const sema.ItemIndexResult {
        const slot = &self.item_index_slots.items[module_id.index()];
        if (slot.* == null) {
            const module = self.sources.module(module_id);
            const result = try self.allocator.create(sema.ItemIndexResult);
            result.* = try sema.buildItemIndex(self.allocator, try self.astFile(module.file_id));
            slot.* = result;
        }
        return slot.*.?;
    }

    pub fn resolveNames(self: *CompilerDb, module_id: source.ModuleId) !*const sema.NameResolutionResult {
        const slot = &self.resolution_slots.items[module_id.index()];
        if (slot.* == null) {
            const module = self.sources.module(module_id);
            const result = try self.allocator.create(sema.NameResolutionResult);
            result.* = try sema.resolveNames(self.allocator, module.file_id, try self.astFile(module.file_id), try self.itemIndex(module_id));
            slot.* = result;
        }
        return slot.*.?;
    }

    pub fn resolutionDiagnostics(self: *CompilerDb, module_id: source.ModuleId) !*const diagnostics.DiagnosticList {
        const result = try self.resolveNames(module_id);
        return &result.diagnostics;
    }

    pub fn typeCheck(self: *CompilerDb, module_id: source.ModuleId, key: sema.TypeCheckKey) !*const sema.TypeCheckResult {
        const cache = &self.typecheck_slots.items[module_id.index()];
        if (cache.lookup(key)) |cached| {
            return cached;
        }
        const cache_key = typeCheckCacheKey(key);
        if (cache.in_progress.contains(cache_key)) {
            return try self.unknownTypeCheckResult(module_id, key);
        }
        try cache.in_progress.put(cache_key, {});
        defer _ = cache.in_progress.remove(cache_key);
        const module = self.sources.module(module_id);
        const result = try self.allocator.create(sema.TypeCheckResult);
        errdefer self.allocator.destroy(result);
        result.* = try sema.typeCheck(
            self.allocator,
            module.file_id,
            try self.astFile(module.file_id),
            try self.itemIndex(module_id),
            try self.resolveNames(module_id),
            try self.constEval(module_id),
            key,
        );
        errdefer result.deinit();
        try cache.entries.put(typeCheckCacheKey(key), result);
        return result;
    }

    pub fn typeCheckDiagnostics(self: *CompilerDb, module_id: source.ModuleId, key: sema.TypeCheckKey) !*const diagnostics.DiagnosticList {
        const result = try self.typeCheck(module_id, key);
        return &result.diagnostics;
    }

    pub fn moduleTypeCheck(self: *CompilerDb, module_id: source.ModuleId) !*const sema.TypeCheckResult {
        const module = self.sources.module(module_id);
        const ast_file = try self.astFile(module.file_id);
        if (ast_file.root_items.len == 0) {
            // Empty modules have no real body. This sentinel key is only used to cache/query
            // the empty-module result and must never be dereferenced as file.bodies[0].
            return self.typeCheck(module_id, .{ .body = ast.BodyId.fromIndex(0) });
        }

        var primary: ?*const sema.TypeCheckResult = null;
        for (ast_file.root_items) |item_id| {
            const result = try self.typeCheck(module_id, .{ .item = item_id });
            if (primary == null) primary = result;
        }
        return primary.?;
    }

    pub fn constEval(self: *CompilerDb, module_id: source.ModuleId) !*const sema.ConstEvalResult {
        const slot = &self.consteval_slots.items[module_id.index()];
        if (slot.* != null) {
            return slot.*.?;
        }
        if (self.consteval_in_progress.items[module_id.index()]) {
            return try self.unknownConstEvalResult(module_id);
        }
        self.consteval_in_progress.items[module_id.index()] = true;
        defer self.consteval_in_progress.items[module_id.index()] = false;

        const module = self.sources.module(module_id);
        const ast_file = try self.astFile(module.file_id);
        const result = try self.allocator.create(sema.ConstEvalResult);
        result.* = try comptime_eval.constEval(self.allocator, ast_file, .{
            .module_id = module_id,
            .type_query = .{
                .context = self,
                .ensure_typecheck = ensureTypeCheckedForComptime,
                .module_typecheck = moduleTypeCheckForComptime,
                .ast_file = astFileForComptime,
                .lookup_item = lookupItemForComptime,
                .resolve_import_alias = resolveImportAliasForComptime,
            },
        });
        slot.* = result;
        return slot.*.?;
    }

    pub fn constEvalDiagnostics(self: *CompilerDb, module_id: source.ModuleId) !*const diagnostics.DiagnosticList {
        const result = try self.constEval(module_id);
        return &result.diagnostics;
    }

    pub fn verificationFacts(self: *CompilerDb, module_id: source.ModuleId, key: sema.VerificationFactsKey) !*const sema.VerificationFactsResult {
        const cache = &self.verification_slots.items[module_id.index()];
        if (cache.lookup(key)) |cached| {
            return cached;
        }
        const module = self.sources.module(module_id);
        const result = try self.allocator.create(sema.VerificationFactsResult);
        errdefer self.allocator.destroy(result);
        result.* = try sema.verificationFacts(self.allocator, try self.astFile(module.file_id), key);
        errdefer result.deinit();
        try cache.entries.put(verificationCacheKey(key), result);
        return result;
    }

    pub fn moduleVerificationFacts(self: *CompilerDb, module_id: source.ModuleId) !*const sema.ModuleVerificationFactsResult {
        const slot = &self.module_verification_slots.items[module_id.index()];
        if (slot.* == null) {
            const module = self.sources.module(module_id);
            const ast_file = try self.astFile(module.file_id);
            const result = try self.allocator.create(sema.ModuleVerificationFactsResult);
            errdefer self.allocator.destroy(result);

            result.* = .{
                .arena = std.heap.ArenaAllocator.init(self.allocator),
                .facts = &.{},
            };
            errdefer result.deinit();

            const arena = result.arena.allocator();
            var facts: std.ArrayList(sema.VerificationFact) = .{};
            defer facts.deinit(arena);

            if (ast_file.root_items.len == 0) {
                // Empty modules have no real body. This sentinel key is only used to cache/query
                // the empty-module result and must never be dereferenced as file.bodies[0].
                const root = try self.verificationFacts(module_id, .{ .body = ast.BodyId.fromIndex(0) });
                try facts.appendSlice(arena, root.facts);
            } else {
                for (ast_file.root_items) |item_id| {
                    const item_facts = try self.verificationFacts(module_id, .{ .item = item_id });
                    try facts.appendSlice(arena, item_facts.facts);
                }
            }

            result.facts = try facts.toOwnedSlice(arena);
            slot.* = result;
        }
        return slot.*.?;
    }

    pub fn lowerToHir(self: *CompilerDb, module_id: source.ModuleId) !*const hir.LoweringResult {
        const slot = &self.hir_slots.items[module_id.index()];
        if (slot.* == null) {
            const module = self.sources.module(module_id);
            const ast_file = try self.astFile(module.file_id);
            const item_index = try self.itemIndex(module_id);
            const resolution = try self.resolveNames(module_id);
            const typecheck = try self.moduleTypeCheck(module_id);
            _ = try self.moduleVerificationFacts(module_id);
            const result = try self.allocator.create(hir.LoweringResult);
            errdefer self.allocator.destroy(result);
            result.* = try hir.lowerModule(self.allocator, &self.sources, module_id, ast_file, item_index, resolution, try self.constEval(module_id), typecheck);
            slot.* = result;
        }
        return slot.*.?;
    }

    fn syntaxResult(self: *CompilerDb, file_id: source.FileId) !*syntax.ParseResult {
        const slot = &self.syntax_slots.items[file_id.index()];
        if (slot.* == null) {
            const result = try self.allocator.create(syntax.ParseResult);
            result.* = try syntax.parse(self.allocator, file_id, self.sources.sourceText(file_id));
            slot.* = result;
        }
        return slot.*.?;
    }

    fn astResult(self: *CompilerDb, file_id: source.FileId) !*ast.LowerResult {
        const slot = &self.ast_slots.items[file_id.index()];
        if (slot.* == null) {
            const result = try self.allocator.create(ast.LowerResult);
            result.* = try ast.lower(self.allocator, try self.syntaxTree(file_id));
            slot.* = result;
        }
        return slot.*.?;
    }

    fn invalidateFile(self: *CompilerDb, file_id: source.FileId) void {
        clearPtrSlot(self.allocator, syntax.ParseResult, &self.syntax_slots.items[file_id.index()]);
        clearPtrSlot(self.allocator, ast.LowerResult, &self.ast_slots.items[file_id.index()]);

        var invalidated_modules: std.AutoHashMap(source.ModuleId, void) = .init(self.allocator);
        defer invalidated_modules.deinit();
        var invalidated_packages: std.AutoHashMap(source.PackageId, void) = .init(self.allocator);
        defer invalidated_packages.deinit();

        for (self.sources.modules.items) |module_record| {
            if (module_record.file_id == file_id) {
                if (self.module_graph_slots.items[module_record.package_id.index()]) |graph| {
                    self.invalidateModuleDependents(graph, module_record.id, &invalidated_modules) catch {
                        self.invalidateModule(module_record.id);
                    };
                } else {
                    self.invalidateModule(module_record.id);
                    invalidated_modules.put(module_record.id, {}) catch {};
                }
                invalidated_packages.put(module_record.package_id, {}) catch {};
            }
        }

        var package_iterator = invalidated_packages.keyIterator();
        while (package_iterator.next()) |package_id| {
            self.invalidatePackage(package_id.*);
        }
    }

    fn invalidatePackage(self: *CompilerDb, package_id: source.PackageId) void {
        clearPtrSlot(self.allocator, sema.ModuleGraphResult, &self.module_graph_slots.items[package_id.index()]);
    }

    fn invalidateModule(self: *CompilerDb, module_id: source.ModuleId) void {
        clearPtrSlot(self.allocator, sema.ItemIndexResult, &self.item_index_slots.items[module_id.index()]);
        clearPtrSlot(self.allocator, sema.NameResolutionResult, &self.resolution_slots.items[module_id.index()]);
        self.typecheck_slots.items[module_id.index()].clear(self.allocator);
        clearPtrSlot(self.allocator, sema.ConstEvalResult, &self.consteval_slots.items[module_id.index()]);
        self.consteval_in_progress.items[module_id.index()] = false;
        clearPtrSlot(self.allocator, sema.ConstEvalResult, &self.consteval_sentinel_slots.items[module_id.index()]);
        self.verification_slots.items[module_id.index()].clear(self.allocator);
        clearPtrSlot(self.allocator, sema.ModuleVerificationFactsResult, &self.module_verification_slots.items[module_id.index()]);
        clearPtrSlot(self.allocator, hir.LoweringResult, &self.hir_slots.items[module_id.index()]);
    }

    fn invalidateModuleDependents(
        self: *CompilerDb,
        graph: *const sema.ModuleGraphResult,
        module_id: source.ModuleId,
        visited: *std.AutoHashMap(source.ModuleId, void),
    ) !void {
        if (visited.contains(module_id)) return;
        try visited.put(module_id, {});
        self.invalidateModule(module_id);

        for (graph.modules) |module_summary| {
            if (!moduleDependsOn(module_summary, module_id)) continue;
            try self.invalidateModuleDependents(graph, module_summary.module_id, visited);
        }
    }

    fn unknownTypeCheckResult(self: *CompilerDb, module_id: source.ModuleId, key: sema.TypeCheckKey) !*const sema.TypeCheckResult {
        const cache = &self.typecheck_slots.items[module_id.index()];
        if (cache.lookupSentinel(key)) |cached| {
            return cached;
        }

        const module = self.sources.module(module_id);
        const ast_file = try self.astFile(module.file_id);
        const result = try self.allocator.create(sema.TypeCheckResult);
        errdefer self.allocator.destroy(result);

        var arena = std.heap.ArenaAllocator.init(self.allocator);
        errdefer arena.deinit();
        const arena_allocator = arena.allocator();

        const item_types = try arena_allocator.alloc(sema.Type, ast_file.items.len);
        const item_regions = try arena_allocator.alloc(sema.Region, ast_file.items.len);
        const item_effects = try arena_allocator.alloc(sema.Effect, ast_file.items.len);
        const pattern_types = try arena_allocator.alloc(sema.LocatedType, ast_file.patterns.len);
        const expr_types = try arena_allocator.alloc(sema.Type, ast_file.expressions.len);
        const expr_effects = try arena_allocator.alloc(sema.Effect, ast_file.expressions.len);
        const body_types = try arena_allocator.alloc(sema.Type, ast_file.bodies.len);

        for (item_types) |*item_type| item_type.* = .{ .unknown = {} };
        for (item_regions) |*region| region.* = .none;
        for (item_effects) |*effect| effect.* = .pure;
        for (pattern_types) |*pattern_type| pattern_type.* = sema.LocatedType.unlocated(.{ .unknown = {} });
        for (expr_types) |*expr_type| expr_type.* = .{ .unknown = {} };
        for (expr_effects) |*effect| effect.* = .pure;
        for (body_types) |*body_type| body_type.* = .{ .unknown = {} };

        result.* = .{
            .arena = arena,
            .key = key,
            .item_types = item_types,
            .item_regions = item_regions,
            .item_effects = item_effects,
            .pattern_types = pattern_types,
            .expr_types = expr_types,
            .expr_effects = expr_effects,
            .body_types = body_types,
            .instantiated_structs = &.{},
            .instantiated_enums = &.{},
            .instantiated_bitfields = &.{},
            .trait_interfaces = &.{},
            .impl_interfaces = &.{},
            .diagnostics = diagnostics.DiagnosticList.init(self.allocator),
        };

        try cache.sentinels.put(typeCheckCacheKey(key), result);
        return result;
    }

    fn unknownConstEvalResult(self: *CompilerDb, module_id: source.ModuleId) !*const sema.ConstEvalResult {
        const slot = &self.consteval_sentinel_slots.items[module_id.index()];
        if (slot.*) |cached| {
            return cached;
        }

        const module = self.sources.module(module_id);
        const ast_file = try self.astFile(module.file_id);
        const result = try self.allocator.create(sema.ConstEvalResult);
        errdefer self.allocator.destroy(result);

        var arena = std.heap.ArenaAllocator.init(self.allocator);
        errdefer arena.deinit();
        const arena_allocator = arena.allocator();
        const values = try arena_allocator.alloc(?sema.ConstValue, ast_file.expressions.len);
        @memset(values, null);

        result.* = .{
            .arena = arena,
            .values = values,
            .diagnostics = diagnostics.DiagnosticList.init(self.allocator),
        };
        slot.* = result;
        return result;
    }
};

fn ensureTypeCheckedForComptime(context: *anyopaque, module_id: source.ModuleId, key: sema.TypeCheckKey) anyerror!*const sema.TypeCheckResult {
    const self: *CompilerDb = @ptrCast(@alignCast(context));
    return self.typeCheck(module_id, key);
}

fn moduleTypeCheckForComptime(context: *anyopaque, module_id: source.ModuleId) anyerror!*const sema.TypeCheckResult {
    const self: *CompilerDb = @ptrCast(@alignCast(context));
    return self.moduleTypeCheck(module_id);
}

fn astFileForComptime(context: *anyopaque, module_id: source.ModuleId) anyerror!*const ast.AstFile {
    const self: *CompilerDb = @ptrCast(@alignCast(context));
    return self.astFile(self.sources.module(module_id).file_id);
}

fn lookupItemForComptime(context: *anyopaque, module_id: source.ModuleId, name: []const u8) anyerror!?ast.ItemId {
    const self: *CompilerDb = @ptrCast(@alignCast(context));
    return (try self.itemIndex(module_id)).lookup(name);
}

fn resolveImportAliasForComptime(context: *anyopaque, module_id: source.ModuleId, alias: []const u8) anyerror!?source.ModuleId {
    const self: *CompilerDb = @ptrCast(@alignCast(context));
    const module = self.sources.module(module_id);
    const graph = try self.moduleGraph(module.package_id);
    const summary = for (graph.modules) |summary| {
        if (summary.module_id == module_id) break summary;
    } else return null;

    for (summary.imports) |import_info| {
        const import_alias = import_info.alias orelse std.fs.path.stem(std.fs.path.basename(import_info.path));
        if (std.mem.eql(u8, import_alias, alias)) return import_info.target_module_id;
    }
    return null;
}

fn ensureSlots(allocator: std.mem.Allocator, slots: anytype, len: usize) !void {
    while (slots.items.len < len) {
        try slots.append(allocator, null);
    }
}

fn deinitSlots(allocator: std.mem.Allocator, slots: anytype) void {
    for (slots.items) |slot| {
        if (slot) |ptr| {
            ptr.deinit();
            allocator.destroy(ptr);
        }
    }
    slots.deinit(allocator);
}

fn deinitCacheSlots(allocator: std.mem.Allocator, slots: anytype) void {
    for (slots.items) |*slot| {
        slot.deinit(allocator);
    }
    slots.deinit(allocator);
}

fn clearPtrSlot(allocator: std.mem.Allocator, comptime T: type, slot: *?*T) void {
    if (slot.*) |ptr| {
        ptr.deinit();
        allocator.destroy(ptr);
        slot.* = null;
    }
}

fn typeCheckCacheKey(key: sema.TypeCheckKey) u64 {
    return switch (key) {
        .item => |item_id| (@as(u64, 0) << 32) | @intFromEnum(item_id),
        .body => |body_id| (@as(u64, 1) << 32) | @intFromEnum(body_id),
    };
}

fn verificationCacheKey(key: sema.VerificationFactsKey) u64 {
    return switch (key) {
        .item => |item_id| (@as(u64, 0) << 32) | @intFromEnum(item_id),
        .body => |body_id| (@as(u64, 1) << 32) | @intFromEnum(body_id),
    };
}

fn moduleDependsOn(summary: sema.ModuleSummary, dependency: source.ModuleId) bool {
    for (summary.dependencies) |module_id| {
        if (module_id == dependency) return true;
    }
    return false;
}
