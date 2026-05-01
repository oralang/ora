const std = @import("std");
const import_graph = @import("ora_imports");
const ast = @import("../ast/mod.zig");
const db = @import("../db/mod.zig");
const diagnostics = @import("../diagnostics/mod.zig");
const sema = @import("../sema/mod.zig");
const source = @import("../source/mod.zig");
const embedded_stdlib = @import("../stdlib_embedded.zig");

fn compilerPhaseDebugEnabled() bool {
    const value = std.process.getEnvVarOwned(std.heap.page_allocator, "ORA_COMPILER_PHASE_DEBUG") catch return false;
    defer std.heap.page_allocator.free(value);
    return value.len != 0 and !std.mem.eql(u8, value, "0");
}

fn compilerPhaseLog(comptime fmt: []const u8, args: anytype) void {
    if (!compilerPhaseDebugEnabled()) return;
    std.debug.print("compiler-phase: " ++ fmt ++ "\n", args);
}

fn diagnosticsHaveErrors(list: *const diagnostics.DiagnosticList) bool {
    for (list.items.items) |diag| {
        if (diag.severity == .Error) return true;
    }
    return false;
}

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
    compilerPhaseLog("load-package begin root={s}", .{root_path});
    var compilation = try loadPackageSources(allocator, root_path, resolver_options);
    compilerPhaseLog("load-package done modules={d}", .{compilation.db.sources.modules.items.len});
    compilerPhaseLog("module-graph begin", .{});
    const module_graph = try compilation.db.moduleGraph(compilation.package_id);
    compilerPhaseLog("module-graph done has-cycles={any}", .{module_graph.has_cycles});
    if (module_graph.has_cycles) {
        const package_name = compilation.db.sources.package(compilation.package_id).name;
        std.log.warn("Compiler package '{s}' contains import cycles; cross-module queries may be incomplete.", .{package_name});
    }

    const package = compilation.db.sources.package(compilation.package_id);
    for (package.modules.items) |module_id| {
        const module = compilation.db.sources.module(module_id);
        compilerPhaseLog("module {s} begin", .{module.name});
        _ = try compilation.db.syntaxTree(module.file_id);
        compilerPhaseLog("module {s} syntax", .{module.name});
        _ = try compilation.db.astFile(module.file_id);
        compilerPhaseLog("module {s} ast", .{module.name});
        _ = try compilation.db.itemIndex(module_id);
        compilerPhaseLog("module {s} item-index", .{module.name});
        _ = try compilation.db.resolveNames(module_id);
        compilerPhaseLog("module {s} resolve", .{module.name});
        const typecheck = try compilation.db.moduleTypeCheck(module_id);
        compilerPhaseLog("module {s} typecheck", .{module.name});
        if (diagnosticsHaveErrors(&typecheck.diagnostics)) {
            return compilation;
        }
        _ = try compilation.db.constEval(module_id);
        compilerPhaseLog("module {s} consteval", .{module.name});
        _ = try compilation.db.moduleVerificationFacts(module_id);
        compilerPhaseLog("module {s} verification-facts", .{module.name});
    }
    compilerPhaseLog("root lower-to-hir begin", .{});
    _ = try compilation.db.lowerToHir(compilation.root_module_id);
    compilerPhaseLog("root lower-to-hir done", .{});
    return compilation;
}

pub fn compileSource(allocator: std.mem.Allocator, path: []const u8, text: []const u8) !Compilation {
    var compiler_db = db.CompilerDb.init(allocator);
    errdefer compiler_db.deinit();

    const package_id = try compiler_db.addPackage("main");
    try addEmbeddedStdModules(&compiler_db, package_id);
    const file_id = try compiler_db.addSourceFile(path, text);
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
        const embedded_source = embedded_stdlib.sourceForResolvedPath(module_info.resolved_path);
        const source_text = if (embedded_source) |text|
            text
        else
            try std.fs.cwd().readFileAlloc(allocator, module_info.resolved_path, 1024 * 1024);
        defer if (embedded_source == null) allocator.free(source_text);

        const file_id = try compiler_db.addSourceFile(module_info.resolved_path, source_text);
        const module_name = if (embedded_stdlib.byLogicalPath("std")) |std_module|
            if (std.mem.eql(u8, module_info.resolved_path, std_module.resolved_path))
                std_module.logical_path
            else if (embedded_stdlib.byLogicalPath("std/bytes")) |bytes_module|
                if (std.mem.eql(u8, module_info.resolved_path, bytes_module.resolved_path))
                    bytes_module.logical_path
                else if (embedded_stdlib.byLogicalPath("std/constants")) |constants_module|
                    if (std.mem.eql(u8, module_info.resolved_path, constants_module.resolved_path))
                        constants_module.logical_path
                    else if (embedded_stdlib.byLogicalPath("std/result")) |result_module|
                        if (std.mem.eql(u8, module_info.resolved_path, result_module.resolved_path))
                            result_module.logical_path
                        else
                            std.fs.path.stem(module_info.resolved_path)
                    else
                        std.fs.path.stem(module_info.resolved_path)
                else
                    std.fs.path.stem(module_info.resolved_path)
            else if (embedded_stdlib.byLogicalPath("std/constants")) |constants_module|
                if (std.mem.eql(u8, module_info.resolved_path, constants_module.resolved_path))
                    constants_module.logical_path
                else if (embedded_stdlib.byLogicalPath("std/result")) |result_module|
                    if (std.mem.eql(u8, module_info.resolved_path, result_module.resolved_path))
                        result_module.logical_path
                    else
                        std.fs.path.stem(module_info.resolved_path)
                else
                    std.fs.path.stem(module_info.resolved_path)
            else
                std.fs.path.stem(module_info.resolved_path)
        else
            std.fs.path.stem(module_info.resolved_path);
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

fn addEmbeddedStdModules(compiler_db: *db.CompilerDb, package_id: source.PackageId) !void {
    for (embedded_stdlib.all()) |module| {
        const file_id = try compiler_db.addSourceFile(module.resolved_path, module.source);
        _ = try compiler_db.addModule(package_id, file_id, module.logical_path);
    }
}
