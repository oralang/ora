const std = @import("std");
const ast = @import("../ast/mod.zig");
const lookup = @import("lookup.zig");
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

fn compilerPhaseDebugEnabled() bool {
    if (!@import("builtin").link_libc) return false;
    const value_ptr = std.c.getenv("ORA_COMPILER_PHASE_DEBUG") orelse return false;
    const value = std.mem.span(value_ptr);
    return value.len != 0 and !std.mem.eql(u8, value, "0");
}

fn compilerPhaseLog(comptime fmt: []const u8, args: anytype) void {
    if (!compilerPhaseDebugEnabled()) return;
    std.debug.print("compiler-phase: " ++ fmt ++ "\n", args);
}

pub fn buildModuleGraph(allocator: std.mem.Allocator, package_id: source.PackageId, inputs: []const ModuleGraphInput) !ModuleGraphResult {
    compilerPhaseLog("build-module-graph begin inputs={d}", .{inputs.len});
    var result = ModuleGraphResult{
        .arena = std.heap.ArenaAllocator.init(allocator),
        .package_id = package_id,
        .modules = &[_]ModuleSummary{},
        .topo_order = &.{},
        .has_cycles = false,
    };
    errdefer result.deinit();

    const arena = result.arena.allocator();
    var file_paths: std.StringHashMap(source.ModuleId) = .init(arena);
    if (inputs.len > 1) try file_paths.ensureTotalCapacity(@intCast(inputs.len));
    for (inputs) |input| {
        compilerPhaseLog("build-module-graph normalize {s}", .{input.file_path});
        try file_paths.put(try normalizeModuleFilePath(arena, input.file_path), input.module_id);
    }

    var modules: std.ArrayList(ModuleSummary) = .empty;
    if (inputs.len != 0) try modules.ensureTotalCapacityPrecise(arena, inputs.len);
    for (inputs) |input| {
        compilerPhaseLog("build-module-graph module {s} scan-imports begin root-items={d}", .{ input.path, input.ast_file.root_items.len });
        var import_count: usize = 0;
        for (input.ast_file.root_items) |item_id| {
            if (input.ast_file.item(item_id).* == .Import) import_count += 1;
        }

        var imports: std.ArrayList(ModuleImport) = .empty;
        var dependencies: std.ArrayList(source.ModuleId) = .empty;
        if (import_count != 0) {
            try imports.ensureTotalCapacityPrecise(arena, import_count);
            try dependencies.ensureTotalCapacityPrecise(arena, import_count);
        }
        for (input.ast_file.root_items) |item_id| {
            const item = input.ast_file.item(item_id).*;
            if (item == .Import) {
                const target_module_id = try resolveImportTargetModuleId(arena, input.file_path, item.Import.path, inputs, &file_paths);
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
        compilerPhaseLog("build-module-graph module {s} scan-imports done deps={d} imports={d}", .{ input.path, dependencies.items.len, imports.items.len });
        try modules.append(arena, .{
            .module_id = input.module_id,
            .file_id = input.file_id,
            .path = input.path,
            .imports = imports.items,
            .dependencies = dependencies.items,
        });
    }
    result.modules = modules.items;
    compilerPhaseLog("build-module-graph topo begin modules={d}", .{result.modules.len});
    result.topo_order = try buildTopoOrder(arena, result.modules, &result.has_cycles);
    compilerPhaseLog("build-module-graph topo done order={d} cycles={any}", .{ result.topo_order.len, result.has_cycles });
    return result;
}

fn resolveImportTargetModuleId(
    allocator: std.mem.Allocator,
    importer_file_path: []const u8,
    import_path: []const u8,
    inputs: []const ModuleGraphInput,
    file_paths: *const std.StringHashMap(source.ModuleId),
) !?source.ModuleId {
    if (std.mem.eql(u8, import_path, "std")) {
        for (inputs) |input| {
            if (std.mem.eql(u8, input.path, "std")) return input.module_id;
        }
        return null;
    }

    if (std.mem.startsWith(u8, import_path, "./") or std.mem.startsWith(u8, import_path, "../")) {
        const importer_dir = std.fs.path.dirname(importer_file_path) orelse ".";
        const resolved = try std.fs.path.resolve(allocator, &.{ importer_dir, import_path });
        return file_paths.get(resolved);
    }

    for (inputs) |input| {
        if (std.mem.eql(u8, input.path, import_path)) return input.module_id;
    }
    return null;
}

fn normalizeModuleFilePath(allocator: std.mem.Allocator, file_path: []const u8) ![]const u8 {
    return std.fs.path.resolve(allocator, &.{file_path});
}

pub fn buildItemIndex(allocator: std.mem.Allocator, file: *const ast.AstFile) !ItemIndexResult {
    var result = ItemIndexResult{
        .arena = std.heap.ArenaAllocator.init(allocator),
        .entries = &[_]NamedItem{},
        .impl_entries = &[_]ImplEntry{},
        .impl_lookup = &[_]lookup.PairEntry{},
        .trait_method_lookup = &[_]lookup.MemberEntry{},
        .impl_method_lookup = &[_]lookup.MemberEntry{},
        .impl_method_owner_lookup = &[_]lookup.IndexEntry{},
        .struct_field_lookup = &[_]lookup.MemberEntry{},
        .bitfield_field_lookup = &[_]lookup.MemberEntry{},
        .enum_variant_lookup = &[_]lookup.MemberEntry{},
        .contract_member_lookup = &[_]lookup.MemberEntry{},
    };
    errdefer result.deinit();

    const arena = result.arena.allocator();
    var entries: std.ArrayList(NamedItem) = .empty;
    var impl_entries: std.ArrayList(ImplEntry) = .empty;
    var trait_method_lookup: std.ArrayList(lookup.MemberEntry) = .empty;
    var impl_method_lookup: std.ArrayList(lookup.MemberEntry) = .empty;
    var impl_method_owner_lookup: std.ArrayList(lookup.IndexEntry) = .empty;
    var struct_field_lookup: std.ArrayList(lookup.MemberEntry) = .empty;
    var bitfield_field_lookup: std.ArrayList(lookup.MemberEntry) = .empty;
    var enum_variant_lookup: std.ArrayList(lookup.MemberEntry) = .empty;
    var contract_member_lookup: std.ArrayList(lookup.MemberEntry) = .empty;
    for (file.root_items) |item_id| {
        try collectItemEntry(arena, file, item_id, null, &entries, &impl_entries, &trait_method_lookup, &impl_method_lookup, &impl_method_owner_lookup, &struct_field_lookup, &bitfield_field_lookup, &enum_variant_lookup, &contract_member_lookup);
    }
    std.sort.heap(NamedItem, entries.items, {}, struct {
        fn lessThan(_: void, lhs: NamedItem, rhs: NamedItem) bool {
            return std.mem.order(u8, lhs.name, rhs.name) == .lt;
        }
    }.lessThan);
    lookup.sortMembers(trait_method_lookup.items);
    lookup.sortMembers(impl_method_lookup.items);
    lookup.sortIndexes(impl_method_owner_lookup.items);
    lookup.sortMembers(struct_field_lookup.items);
    lookup.sortMembers(bitfield_field_lookup.items);
    lookup.sortMembers(enum_variant_lookup.items);
    lookup.sortMembers(contract_member_lookup.items);
    result.entries = try entries.toOwnedSlice(arena);
    result.impl_entries = try impl_entries.toOwnedSlice(arena);
    result.impl_lookup = try lookup.buildPair(ImplEntry, arena, result.impl_entries, "trait_name", "target_name");
    result.trait_method_lookup = try trait_method_lookup.toOwnedSlice(arena);
    result.impl_method_lookup = try impl_method_lookup.toOwnedSlice(arena);
    result.impl_method_owner_lookup = try impl_method_owner_lookup.toOwnedSlice(arena);
    result.struct_field_lookup = try struct_field_lookup.toOwnedSlice(arena);
    result.bitfield_field_lookup = try bitfield_field_lookup.toOwnedSlice(arena);
    result.enum_variant_lookup = try enum_variant_lookup.toOwnedSlice(arena);
    result.contract_member_lookup = try contract_member_lookup.toOwnedSlice(arena);
    return result;
}

fn collectItemEntry(
    allocator: std.mem.Allocator,
    file: *const ast.AstFile,
    item_id: ast.ItemId,
    prefix: ?[]const u8,
    entries: *std.ArrayList(NamedItem),
    impl_entries: *std.ArrayList(ImplEntry),
    trait_method_lookup: *std.ArrayList(lookup.MemberEntry),
    impl_method_lookup: *std.ArrayList(lookup.MemberEntry),
    impl_method_owner_lookup: *std.ArrayList(lookup.IndexEntry),
    struct_field_lookup: *std.ArrayList(lookup.MemberEntry),
    bitfield_field_lookup: *std.ArrayList(lookup.MemberEntry),
    enum_variant_lookup: *std.ArrayList(lookup.MemberEntry),
    contract_member_lookup: *std.ArrayList(lookup.MemberEntry),
) !void {
    const item = file.item(item_id).*;
    const name = switch (item) {
        .Contract => item.Contract.name,
        .Function => item.Function.name,
        .Struct => blk: {
            for (item.Struct.fields, 0..) |field, field_index| {
                try struct_field_lookup.append(allocator, .{
                    .owner_index = item_id.index(),
                    .name = field.name,
                    .index = field_index,
                });
            }
            break :blk item.Struct.name;
        },
        .Bitfield => blk: {
            for (item.Bitfield.fields, 0..) |field, field_index| {
                try bitfield_field_lookup.append(allocator, .{
                    .owner_index = item_id.index(),
                    .name = field.name,
                    .index = field_index,
                });
            }
            break :blk item.Bitfield.name;
        },
        .Enum => blk: {
            for (item.Enum.variants, 0..) |variant, index| {
                try enum_variant_lookup.append(allocator, .{
                    .owner_index = item_id.index(),
                    .name = variant.name,
                    .index = index,
                });
            }
            break :blk item.Enum.name;
        },
        .Trait => blk: {
            for (item.Trait.methods, 0..) |method, method_index| {
                try trait_method_lookup.append(allocator, .{
                    .owner_index = item_id.index(),
                    .name = method.name,
                    .index = method_index,
                });
            }
            break :blk item.Trait.name;
        },
        .Resource => item.Resource.name,
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
            for (item.Impl.methods, 0..) |method_id, method_index| {
                const method = switch (file.item(method_id).*) {
                    .Function => |function| function,
                    else => continue,
                };
                try impl_method_lookup.append(allocator, .{
                    .owner_index = item_id.index(),
                    .name = method.name,
                    .index = method_index,
                });
                try impl_method_owner_lookup.append(allocator, .{
                    .key_index = method_id.index(),
                    .index = item_id.index(),
                });
            }
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
        for (item.Contract.members, 0..) |member_id, member_index| {
            if (contractMemberName(file.item(member_id).*)) |member_name| {
                try contract_member_lookup.append(allocator, .{
                    .owner_index = item_id.index(),
                    .name = member_name,
                    .index = member_index,
                });
            }
            try collectItemEntry(allocator, file, member_id, item.Contract.name, entries, impl_entries, trait_method_lookup, impl_method_lookup, impl_method_owner_lookup, struct_field_lookup, bitfield_field_lookup, enum_variant_lookup, contract_member_lookup);
        }
    }
}

fn contractMemberName(item: ast.Item) ?[]const u8 {
    return switch (item) {
        .Function => |function| function.name,
        .Field => |field| field.name,
        .Constant => |constant| constant.name,
        .Struct => |struct_item| struct_item.name,
        .Bitfield => |bitfield| bitfield.name,
        .Enum => |enum_item| enum_item.name,
        .Resource => |resource| resource.name,
        .Trait => |trait_item| trait_item.name,
        .TypeAlias => |type_alias| type_alias.name,
        .LogDecl => |log_decl| log_decl.name,
        .ErrorDecl => |error_decl| error_decl.name,
        else => null,
    };
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
    var ordered: std.ArrayList(source.ModuleId) = .empty;
    if (modules.len != 0) try ordered.ensureTotalCapacityPrecise(allocator, modules.len);

    for (modules, 0..) |module_summary, index| {
        if (states[index] == .unvisited) {
            try visitModule(allocator, modules, module_summary.module_id, states, &ordered, has_cycles);
        }
    }

    return ordered.items;
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
