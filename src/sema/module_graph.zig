const std = @import("std");
const ast = @import("../ast/mod.zig");
const model = @import("model.zig");
const source = @import("../source/mod.zig");

const ModuleGraphInput = model.ModuleGraphInput;
const ModuleGraphResult = model.ModuleGraphResult;
const ModuleImport = model.ModuleImport;
const ModuleSummary = model.ModuleSummary;
const ItemIndexResult = model.ItemIndexResult;
const NamedItem = model.NamedItem;
const ImplEntry = model.ImplEntry;
const VisitState = enum(u2) { unvisited, visiting, done };

pub fn buildModuleGraph(allocator: std.mem.Allocator, package_id: source.PackageId, inputs: []const ModuleGraphInput) !ModuleGraphResult {
    var result = ModuleGraphResult{
        .arena = std.heap.ArenaAllocator.init(allocator),
        .package_id = package_id,
        .modules = &[_]ModuleSummary{},
        .topo_order = &.{},
        .has_cycles = false,
    };
    errdefer result.deinit();

    const arena = result.arena.allocator();
    var names: std.StringHashMap(source.ModuleId) = .init(arena);
    for (inputs) |input| {
        try names.put(input.path, input.module_id);
    }

    var modules: std.ArrayList(ModuleSummary) = .{};
    for (inputs) |input| {
        var imports: std.ArrayList(ModuleImport) = .{};
        var dependencies: std.ArrayList(source.ModuleId) = .{};
        for (input.ast_file.root_items) |item_id| {
            const item = input.ast_file.item(item_id).*;
            if (item == .Import) {
                const target_module_id = names.get(normalizeImportPath(item.Import.path));
                if (target_module_id) |dependency| {
                    if (!containsModuleId(dependencies.items, dependency)) {
                        try dependencies.append(arena, dependency);
                    }
                }
                try imports.append(arena, .{
                    .range = item.Import.range,
                    .path = item.Import.path,
                    .alias = item.Import.alias,
                    .target_module_id = target_module_id,
                });
            }
        }
        try modules.append(arena, .{
            .module_id = input.module_id,
            .file_id = input.file_id,
            .path = input.path,
            .imports = try imports.toOwnedSlice(arena),
            .dependencies = try dependencies.toOwnedSlice(arena),
        });
    }
    result.modules = try modules.toOwnedSlice(arena);
    result.topo_order = try buildTopoOrder(arena, result.modules, &result.has_cycles);
    return result;
}

pub fn buildItemIndex(allocator: std.mem.Allocator, file: *const ast.AstFile) !ItemIndexResult {
    var result = ItemIndexResult{
        .arena = std.heap.ArenaAllocator.init(allocator),
        .entries = &[_]NamedItem{},
        .impl_entries = &[_]ImplEntry{},
    };
    errdefer result.deinit();

    const arena = result.arena.allocator();
    var entries: std.ArrayList(NamedItem) = .{};
    var impl_entries: std.ArrayList(ImplEntry) = .{};
    for (file.root_items) |item_id| {
        try collectItemEntry(arena, file, item_id, null, &entries, &impl_entries);
    }
    std.sort.heap(NamedItem, entries.items, {}, struct {
        fn lessThan(_: void, lhs: NamedItem, rhs: NamedItem) bool {
            return std.mem.order(u8, lhs.name, rhs.name) == .lt;
        }
    }.lessThan);
    result.entries = try entries.toOwnedSlice(arena);
    result.impl_entries = try impl_entries.toOwnedSlice(arena);
    return result;
}

fn collectItemEntry(
    allocator: std.mem.Allocator,
    file: *const ast.AstFile,
    item_id: ast.ItemId,
    prefix: ?[]const u8,
    entries: *std.ArrayList(NamedItem),
    impl_entries: *std.ArrayList(ImplEntry),
) !void {
    const item = file.item(item_id).*;
    const name = switch (item) {
        .Contract => item.Contract.name,
        .Function => item.Function.name,
        .Struct => item.Struct.name,
        .Bitfield => item.Bitfield.name,
        .Enum => item.Enum.name,
        .Trait => item.Trait.name,
        .TypeAlias => item.TypeAlias.name,
        .LogDecl => item.LogDecl.name,
        .ErrorDecl => item.ErrorDecl.name,
        .Field => item.Field.name,
        .Constant => item.Constant.name,
        .Impl => {
            try impl_entries.append(allocator, .{
                .trait_name = item.Impl.trait_name,
                .target_name = item.Impl.target_name,
                .item_id = item_id,
            });
            return;
        },
        .GhostBlock => return,
        else => return,
    };

    if (prefix) |p| {
        const qualified = try std.fmt.allocPrint(allocator, "{s}.{s}", .{ p, name });
        try entries.append(allocator, .{ .name = qualified, .item_id = item_id });
    }
    try entries.append(allocator, .{ .name = name, .item_id = item_id });

    if (item == .Contract) {
        for (item.Contract.members) |member_id| {
            try collectItemEntry(allocator, file, member_id, item.Contract.name, entries, impl_entries);
        }
    }
}

fn normalizeImportPath(path: []const u8) []const u8 {
    return std.fs.path.stem(std.fs.path.basename(path));
}

fn containsModuleId(values: []const source.ModuleId, needle: source.ModuleId) bool {
    for (values) |value| {
        if (value == needle) return true;
    }
    return false;
}

fn buildTopoOrder(allocator: std.mem.Allocator, modules: []const ModuleSummary, has_cycles: *bool) ![]const source.ModuleId {
    const states = try allocator.alloc(VisitState, modules.len);
    @memset(states, .unvisited);
    var ordered: std.ArrayList(source.ModuleId) = .{};

    for (modules, 0..) |module_summary, index| {
        if (states[index] == .unvisited) {
            try visitModule(allocator, modules, module_summary.module_id, states, &ordered, has_cycles);
        }
    }

    return ordered.toOwnedSlice(allocator);
}

fn visitModule(
    allocator: std.mem.Allocator,
    modules: []const ModuleSummary,
    module_id: source.ModuleId,
    states: []VisitState,
    ordered: *std.ArrayList(source.ModuleId),
    has_cycles: *bool,
) !void {
    const index = findModuleIndex(modules, module_id) orelse return;
    switch (states[index]) {
        .done => return,
        .visiting => {
            has_cycles.* = true;
            return;
        },
        .unvisited => {},
    }

    states[index] = .visiting;
    for (modules[index].dependencies) |dependency| {
        try visitModule(allocator, modules, dependency, states, ordered, has_cycles);
    }
    states[index] = .done;
    try ordered.append(allocator, module_id);
}

fn findModuleIndex(modules: []const ModuleSummary, module_id: source.ModuleId) ?usize {
    for (modules, 0..) |module_summary, index| {
        if (module_summary.module_id == module_id) return index;
    }
    return null;
}
