const std = @import("std");
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
    const source_text = try std.fs.cwd().readFileAlloc(allocator, root_path, 1024 * 1024);
    defer allocator.free(source_text);

    var compilation = try compileSource(allocator, root_path, source_text);
    const module_graph = try compilation.db.moduleGraph(compilation.package_id);
    if (module_graph.has_cycles) {
        const package_name = compilation.db.sources.package(compilation.package_id).name;
        std.log.warn("Compiler package '{s}' contains import cycles; cross-module queries may be incomplete.", .{package_name});
    }
    _ = try compilation.db.syntaxTree(compilation.db.sources.module(compilation.root_module_id).file_id);
    _ = try compilation.db.astFile(compilation.db.sources.module(compilation.root_module_id).file_id);
    _ = try compilation.db.itemIndex(compilation.root_module_id);
    _ = try compilation.db.resolveNames(compilation.root_module_id);
    _ = try compilation.db.moduleTypeCheck(compilation.root_module_id);
    _ = try compilation.db.constEval(compilation.root_module_id);
    _ = try compilation.db.moduleVerificationFacts(compilation.root_module_id);
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
