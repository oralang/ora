const std = @import("std");
const import_graph = @import("ora_imports");
const ast = @import("../ast/mod.zig");
const db = @import("../db/mod.zig");
const diagnostics = @import("../diagnostics/mod.zig");
const hir = @import("../hir/mod.zig");
const sema = @import("../sema/mod.zig");
const source = @import("../source/mod.zig");
const embedded_stdlib = import_graph.embedded_stdlib;
const compile_options = @import("../compile_options.zig");
const Metrics = @import("../metrics.zig").Metrics;

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

fn beginMetric(instrumentation: ?*Metrics, name: []const u8) void {
    if (instrumentation) |metrics| metrics.begin(name);
}

fn endMetric(instrumentation: ?*Metrics, work_count: usize) void {
    if (instrumentation) |metrics| metrics.endWith(metricCount(work_count));
}

fn cancelMetric(instrumentation: ?*Metrics) void {
    if (instrumentation) |metrics| metrics.cancel();
}

fn metricCount(value: usize) u64 {
    return std.math.cast(u64, value) orelse std.math.maxInt(u64);
}

pub const Compilation = struct {
    db: db.CompilerDb,
    package_id: source.PackageId,
    root_module_id: source.ModuleId,
    artifact_blocked: bool = false,

    pub fn deinit(self: *Compilation) void {
        self.db.deinit();
    }

    pub fn isArtifactEmittable(self: *const Compilation) bool {
        return !self.artifact_blocked;
    }
};

pub const CompilePackageOptions = struct {
    resolver_options: import_graph.ResolverOptions = .{},
    compile_options: compile_options.CompileOptions = .{},
};

pub fn compilePackage(allocator: std.mem.Allocator, root_path: []const u8) !Compilation {
    return compilePackageWithOptions(allocator, root_path, .{});
}

pub fn compilePackageWithOptions(
    allocator: std.mem.Allocator,
    root_path: []const u8,
    options: CompilePackageOptions,
) !Compilation {
    compilerPhaseLog("load-package begin root={s}", .{root_path});
    var compilation = try loadPackageSources(allocator, root_path, options.resolver_options);
    errdefer compilation.deinit();
    compilation.db.setCompileOptions(options.compile_options);
    try finishCompilation(&compilation);
    return compilation;
}

pub fn compilePackageWithResolverOptions(
    allocator: std.mem.Allocator,
    root_path: []const u8,
    resolver_options: import_graph.ResolverOptions,
) !Compilation {
    return compilePackageWithOptions(allocator, root_path, .{ .resolver_options = resolver_options });
}

fn finishCompilation(compilation: *Compilation) !void {
    compilerPhaseLog("load-package done modules={d}", .{compilation.db.sources.modules.items.len});
    const instrumentation = compilation.db.options.instrumentation;
    const package = compilation.db.sources.package(compilation.package_id);

    for (package.modules.items) |module_id| {
        const module = compilation.db.sources.module(module_id);
        compilerPhaseLog("module {s} syntax begin", .{module.name});
        beginMetric(instrumentation, "syntax");
        const tree = compilation.db.syntaxTree(module.file_id) catch |err| {
            cancelMetric(instrumentation);
            return err;
        };
        endMetric(instrumentation, tree.tokens.len);
        compilerPhaseLog("module {s} syntax", .{module.name});
    }

    for (package.modules.items) |module_id| {
        const module = compilation.db.sources.module(module_id);
        compilerPhaseLog("module {s} ast begin", .{module.name});
        beginMetric(instrumentation, "ast-lower");
        const ast_file = compilation.db.astFile(module.file_id) catch |err| {
            cancelMetric(instrumentation);
            return err;
        };
        endMetric(instrumentation, ast_file.expressions.len);
        compilerPhaseLog("module {s} ast", .{module.name});
    }

    compilerPhaseLog("module-graph begin", .{});
    beginMetric(instrumentation, "module-graph");
    const module_graph = compilation.db.moduleGraph(compilation.package_id) catch |err| {
        cancelMetric(instrumentation);
        return err;
    };
    endMetric(instrumentation, module_graph.modules.len);
    compilerPhaseLog("module-graph done has-cycles={any}", .{module_graph.has_cycles});
    if (module_graph.has_cycles) {
        const package_name = compilation.db.sources.package(compilation.package_id).name;
        std.log.warn("Compiler package '{s}' contains import cycles; cross-module queries may be incomplete.", .{package_name});
    }

    for (package.modules.items) |module_id| {
        const module = compilation.db.sources.module(module_id);
        compilerPhaseLog("module {s} begin", .{module.name});
        beginMetric(instrumentation, "item-index");
        const item_index = compilation.db.itemIndex(module_id) catch |err| {
            cancelMetric(instrumentation);
            return err;
        };
        endMetric(instrumentation, item_index.entries.len);
        compilerPhaseLog("module {s} item-index", .{module.name});
        beginMetric(instrumentation, "resolve");
        const resolution = compilation.db.resolveNames(module_id) catch |err| {
            cancelMetric(instrumentation);
            return err;
        };
        endMetric(instrumentation, resolution.expr_bindings.len);
        compilerPhaseLog("module {s} resolve", .{module.name});
        beginMetric(instrumentation, "typecheck");
        const typecheck = compilation.db.moduleTypeCheck(module_id) catch |err| {
            cancelMetric(instrumentation);
            return err;
        };
        endMetric(instrumentation, typecheck.expr_types.len);
        compilerPhaseLog("module {s} typecheck", .{module.name});
        if (diagnosticsHaveErrors(&typecheck.diagnostics)) {
            compilation.artifact_blocked = true;
            return;
        }
        beginMetric(instrumentation, "const-eval");
        const const_eval = compilation.db.constEval(module_id) catch |err| {
            cancelMetric(instrumentation);
            return err;
        };
        endMetric(instrumentation, const_eval.values.len);
        compilerPhaseLog("module {s} consteval", .{module.name});
        beginMetric(instrumentation, "verify-facts");
        const verification_facts = compilation.db.moduleVerificationFacts(module_id) catch |err| {
            cancelMetric(instrumentation);
            return err;
        };
        endMetric(instrumentation, verification_facts.facts.len);
        compilerPhaseLog("module {s} verification-facts", .{module.name});
    }
    compilerPhaseLog("root lower-to-hir begin", .{});
    beginMetric(instrumentation, "hir-lower");
    const lowering = compilation.db.lowerToHir(compilation.root_module_id) catch |err| {
        cancelMetric(instrumentation);
        return err;
    };
    endMetric(instrumentation, lowering.items.len);
    compilerPhaseLog("root lower-to-hir done", .{});
    if (diagnosticsHaveErrors(&lowering.diagnostics)) {
        compilation.artifact_blocked = true;
        return;
    }
    if (!lowering.isEmittable() or hir.findExecutableFallback(lowering.module.raw_module) != null) {
        compilation.artifact_blocked = true;
        return;
    }
}

pub fn compileSource(allocator: std.mem.Allocator, path: []const u8, text: []const u8) !Compilation {
    return compileSourceWithOptions(allocator, path, text, .{});
}

pub fn compileSourceWithOptions(allocator: std.mem.Allocator, path: []const u8, text: []const u8, options: compile_options.CompileOptions) !Compilation {
    var compiler_db = db.CompilerDb.init(allocator);
    errdefer compiler_db.deinit();
    compiler_db.setCompileOptions(options);

    const package_id = try compiler_db.addPackage("main");
    const file_id = try compiler_db.addSourceFile(path, text);
    const module_name = std.fs.path.stem(path);
    const module_id = try compiler_db.addModule(package_id, file_id, module_name);
    try addEmbeddedStdModulesReferencedByAst(allocator, &compiler_db, package_id, try compiler_db.astFile(file_id));

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
        const module_name = if (embedded_stdlib.byResolvedPath(module_info.resolved_path)) |module|
            module.logical_path
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

fn addEmbeddedStdModulesReferencedByAst(
    allocator: std.mem.Allocator,
    compiler_db: *db.CompilerDb,
    package_id: source.PackageId,
    ast_file: *const ast.AstFile,
) anyerror!void {
    var added = std.StringHashMap(void).init(allocator);
    defer added.deinit();
    for (ast_file.root_items) |item_id| {
        const item = ast_file.item(item_id).*;
        if (item == .Import) {
            const module = embedded_stdlib.byLogicalPath(item.Import.path) orelse continue;
            try addEmbeddedStdModuleWithDependencies(allocator, compiler_db, package_id, module, &added);
        }
    }
}

fn addEmbeddedStdModuleWithDependencies(
    allocator: std.mem.Allocator,
    compiler_db: *db.CompilerDb,
    package_id: source.PackageId,
    module: embedded_stdlib.EmbeddedModule,
    added: *std.StringHashMap(void),
) anyerror!void {
    if (added.contains(module.logical_path)) return;
    try added.put(module.logical_path, {});
    const file_id = try compiler_db.addSourceFile(module.resolved_path, module.source);
    _ = try compiler_db.addModule(package_id, file_id, module.logical_path);
    for (module.imports) |import_info| {
        if (embedded_stdlib.byLogicalPath(import_info.specifier)) |dependency| {
            try addEmbeddedStdModuleWithDependencies(allocator, compiler_db, package_id, dependency, added);
        }
    }
}
