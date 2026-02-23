const std = @import("std");
const testing = std.testing;
const imports = @import("mod.zig");

fn pathFromTmpAlloc(allocator: std.mem.Allocator, tmp: std.testing.TmpDir, rel_path: []const u8) ![]u8 {
    return std.fmt.allocPrint(allocator, ".zig-cache/tmp/{s}/{s}", .{ tmp.sub_path, rel_path });
}

test "imports: deterministic order for relative diamond graph" {
    const allocator = testing.allocator;
    var tmp = testing.tmpDir(.{});
    defer tmp.cleanup();

    try tmp.dir.writeFile(.{
        .sub_path = "entry.ora",
        .data =
        \\const left = @import("./left.ora");
        \\const right = @import("./right.ora");
        ,
    });
    try tmp.dir.writeFile(.{
        .sub_path = "left.ora",
        .data = "const shared = @import(\"./shared.ora\");",
    });
    try tmp.dir.writeFile(.{
        .sub_path = "right.ora",
        .data = "const shared = @import(\"./shared.ora\");",
    });
    try tmp.dir.writeFile(.{
        .sub_path = "shared.ora",
        .data = "contract Shared { }",
    });

    const entry_path = try pathFromTmpAlloc(allocator, tmp, "entry.ora");
    defer allocator.free(entry_path);

    var graph = try imports.resolveImportGraph(allocator, entry_path, .{});
    defer graph.deinit(allocator);

    try testing.expectEqual(@as(usize, 4), graph.modules.len);
    try testing.expectEqualStrings("shared.ora", std.fs.path.basename(graph.modules[0].resolved_path));
    try testing.expectEqualStrings("left.ora", std.fs.path.basename(graph.modules[1].resolved_path));
    try testing.expectEqualStrings("right.ora", std.fs.path.basename(graph.modules[2].resolved_path));
    try testing.expectEqualStrings("entry.ora", std.fs.path.basename(graph.modules[3].resolved_path));
    try testing.expect(std.mem.eql(u8, graph.entry_canonical_id, graph.modules[3].canonical_id));

    const entry = graph.modules[3];
    try testing.expectEqual(@as(usize, 2), entry.imports.len);
    try testing.expectEqualStrings("left", entry.imports[0].alias);
    try testing.expectEqualStrings("./left.ora", entry.imports[0].specifier);
    try testing.expectEqualStrings(graph.modules[1].canonical_id, entry.imports[0].target_canonical_id);
    try testing.expectEqualStrings("right", entry.imports[1].alias);
    try testing.expectEqualStrings("./right.ora", entry.imports[1].specifier);
    try testing.expectEqualStrings(graph.modules[2].canonical_id, entry.imports[1].target_canonical_id);
}

test "imports: detects recursive cycle" {
    const allocator = testing.allocator;
    var tmp = testing.tmpDir(.{});
    defer tmp.cleanup();

    try tmp.dir.writeFile(.{
        .sub_path = "entry.ora",
        .data = "const a = @import(\"./a.ora\");",
    });
    try tmp.dir.writeFile(.{
        .sub_path = "a.ora",
        .data = "const b = @import(\"./b.ora\");",
    });
    try tmp.dir.writeFile(.{
        .sub_path = "b.ora",
        .data = "const a = @import(\"./a.ora\");",
    });

    const entry_path = try pathFromTmpAlloc(allocator, tmp, "entry.ora");
    defer allocator.free(entry_path);

    try testing.expectError(error.ImportCycleDetected, imports.validateNormalImports(allocator, entry_path));
}

test "imports: reports missing relative target" {
    const allocator = testing.allocator;
    var tmp = testing.tmpDir(.{});
    defer tmp.cleanup();

    try tmp.dir.writeFile(.{
        .sub_path = "entry.ora",
        .data = "const missing = @import(\"./missing.ora\");",
    });

    const entry_path = try pathFromTmpAlloc(allocator, tmp, "entry.ora");
    defer allocator.free(entry_path);

    try testing.expectError(error.ImportTargetNotFound, imports.validateNormalImports(allocator, entry_path));
}

test "imports: relative specifier must include .ora extension" {
    const allocator = testing.allocator;
    var tmp = testing.tmpDir(.{});
    defer tmp.cleanup();

    try tmp.dir.writeFile(.{
        .sub_path = "entry.ora",
        .data = "const lib = @import(\"./lib\");",
    });
    try tmp.dir.writeFile(.{
        .sub_path = "lib.ora",
        .data = "contract Lib { }",
    });

    const entry_path = try pathFromTmpAlloc(allocator, tmp, "entry.ora");
    defer allocator.free(entry_path);

    try testing.expectError(
        error.RelativeImportMustIncludeOraExtension,
        imports.validateNormalImports(allocator, entry_path),
    );
}

test "imports: package style resolves from workspace roots" {
    const allocator = testing.allocator;
    var tmp = testing.tmpDir(.{});
    defer tmp.cleanup();

    try tmp.dir.makePath("workspace/acme");
    try tmp.dir.writeFile(.{
        .sub_path = "workspace/acme/math.ora",
        .data = "pub fn add(x: u256, y: u256) -> u256 { return x + y; }",
    });
    try tmp.dir.writeFile(.{
        .sub_path = "entry.ora",
        .data = "const math = @import(\"acme/math\");",
    });

    const entry_path = try pathFromTmpAlloc(allocator, tmp, "entry.ora");
    defer allocator.free(entry_path);
    const workspace_root = try pathFromTmpAlloc(allocator, tmp, "workspace");
    defer allocator.free(workspace_root);

    const roots = [_][]const u8{workspace_root};
    var graph = try imports.resolveImportGraph(allocator, entry_path, .{
        .workspace_roots = roots[0..],
    });
    defer graph.deinit(allocator);

    try testing.expectEqual(@as(usize, 2), graph.modules.len);
    try testing.expectEqual(imports.ModuleKind.package, graph.modules[0].kind);
    try testing.expect(std.mem.startsWith(u8, graph.modules[0].canonical_id, "pkg:acme@workspace/math.ora"));
    try testing.expectEqualStrings("math.ora", std.fs.path.basename(graph.modules[0].resolved_path));
    try testing.expectEqual(imports.ModuleKind.file, graph.modules[1].kind);
    try testing.expectEqualStrings("entry.ora", std.fs.path.basename(graph.modules[1].resolved_path));
}

test "imports: package root conflict is rejected" {
    const allocator = testing.allocator;
    var tmp = testing.tmpDir(.{});
    defer tmp.cleanup();

    try tmp.dir.makePath("w1/acme");
    try tmp.dir.makePath("w2/acme");
    try tmp.dir.writeFile(.{
        .sub_path = "w1/acme/math.ora",
        .data = "pub fn math() -> u256 { return 1; }",
    });
    try tmp.dir.writeFile(.{
        .sub_path = "w2/acme/util.ora",
        .data = "pub fn util() -> u256 { return 2; }",
    });
    try tmp.dir.writeFile(.{
        .sub_path = "entry.ora",
        .data =
        \\const math = @import("acme/math");
        \\const util = @import("acme/util");
        ,
    });

    const entry_path = try pathFromTmpAlloc(allocator, tmp, "entry.ora");
    defer allocator.free(entry_path);
    const w1 = try pathFromTmpAlloc(allocator, tmp, "w1");
    defer allocator.free(w1);
    const w2 = try pathFromTmpAlloc(allocator, tmp, "w2");
    defer allocator.free(w2);

    const roots = [_][]const u8{ w1, w2 };
    try testing.expectError(
        error.PackageRootConflict,
        imports.resolveImportGraph(allocator, entry_path, .{ .workspace_roots = roots[0..] }),
    );
}

test "imports: std import is ignored by resolver graph" {
    const allocator = testing.allocator;
    var tmp = testing.tmpDir(.{});
    defer tmp.cleanup();

    try tmp.dir.writeFile(.{
        .sub_path = "entry.ora",
        .data =
        \\const std = @import("std");
        \\contract UsesStd { }
        ,
    });

    const entry_path = try pathFromTmpAlloc(allocator, tmp, "entry.ora");
    defer allocator.free(entry_path);

    var graph = try imports.resolveImportGraph(allocator, entry_path, .{});
    defer graph.deinit(allocator);

    try testing.expectEqual(@as(usize, 1), graph.modules.len);
    try testing.expectEqualStrings("entry.ora", std.fs.path.basename(graph.modules[0].resolved_path));
    try testing.expectEqual(@as(usize, 0), graph.modules[0].imports.len);
}

test "imports: package import must include package and module path" {
    const allocator = testing.allocator;
    var tmp = testing.tmpDir(.{});
    defer tmp.cleanup();

    try tmp.dir.writeFile(.{
        .sub_path = "entry.ora",
        .data = "const bad = @import(\"acme\");",
    });

    const entry_path = try pathFromTmpAlloc(allocator, tmp, "entry.ora");
    defer allocator.free(entry_path);

    try testing.expectError(
        error.InvalidImportSpecifier,
        imports.validateNormalImports(allocator, entry_path),
    );
}
