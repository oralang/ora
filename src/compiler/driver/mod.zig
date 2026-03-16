const std = @import("std");
const import_graph = @import("../../imports/mod.zig");
const ast = @import("../ast/mod.zig");
const db = @import("../db/mod.zig");
const sema = @import("../sema/mod.zig");
const source = @import("../source/mod.zig");

pub const Compilation = struct {
    db: db.CompilerDb,
    package_id: source.PackageId,
    root_module_id: source.ModuleId,

    pub fn deinit(self: *Compilation) void {
        self.db.deinit();
    }
};

pub fn compilePackage(allocator: std.mem.Allocator, root_path: []const u8) !Compilation {
    return compilePackageWithResolverOptions(allocator, root_path, .{});
}

pub fn compilePackageWithResolverOptions(
    allocator: std.mem.Allocator,
    root_path: []const u8,
    resolver_options: import_graph.ResolverOptions,
) !Compilation {
    var compilation = try loadPackageSources(allocator, root_path, resolver_options);
    const module_graph = try compilation.db.moduleGraph(compilation.package_id);
    if (module_graph.has_cycles) {
        const package_name = compilation.db.sources.package(compilation.package_id).name;
        std.log.warn("Compiler package '{s}' contains import cycles; cross-module queries may be incomplete.", .{package_name});
    }

    const package = compilation.db.sources.package(compilation.package_id);
    for (package.modules.items) |module_id| {
        const module = compilation.db.sources.module(module_id);
        _ = try compilation.db.syntaxTree(module.file_id);
        _ = try compilation.db.astFile(module.file_id);
        _ = try compilation.db.itemIndex(module_id);
        _ = try compilation.db.resolveNames(module_id);
        _ = try compilation.db.moduleTypeCheck(module_id);
        _ = try compilation.db.constEval(module_id);
        _ = try compilation.db.moduleVerificationFacts(module_id);
    }
    _ = try compilation.db.lowerToHir(compilation.root_module_id);
    return compilation;
}

pub fn compileSource(allocator: std.mem.Allocator, path: []const u8, text: []const u8) !Compilation {
    var compiler_db = db.CompilerDb.init(allocator);
    errdefer compiler_db.deinit();

    const file_id = try compiler_db.addSourceFile(path, text);
    const package_id = try compiler_db.addPackage("main");
    const module_name = std.fs.path.stem(path);
    const module_id = try compiler_db.addModule(package_id, file_id, module_name);

    return .{
        .db = compiler_db,
        .package_id = package_id,
        .root_module_id = module_id,
    };
}

fn loadPackageSources(
    allocator: std.mem.Allocator,
    root_path: []const u8,
    resolver_options: import_graph.ResolverOptions,
) !Compilation {
    var compiler_db = db.CompilerDb.init(allocator);
    errdefer compiler_db.deinit();

    var graph = try import_graph.resolveImportGraph(allocator, root_path, resolver_options);
    defer graph.deinit(allocator);

    const package_id = try compiler_db.addPackage("main");
    var root_module_id: ?source.ModuleId = null;

    for (graph.modules) |module_info| {
        const source_text = try std.fs.cwd().readFileAlloc(allocator, module_info.resolved_path, 1024 * 1024);
        defer allocator.free(source_text);

        const file_id = try compiler_db.addSourceFile(module_info.resolved_path, source_text);
        const module_name = std.fs.path.stem(module_info.resolved_path);
        const module_id = try compiler_db.addModule(package_id, file_id, module_name);
        if (std.mem.eql(u8, module_info.canonical_id, graph.entry_canonical_id)) {
            root_module_id = module_id;
        }
    }

    return .{
        .db = compiler_db,
        .package_id = package_id,
        .root_module_id = root_module_id orelse return error.ModuleNotFound,
    };
}
