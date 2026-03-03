const std = @import("std");
const builtin = @import("builtin");
const testing = std.testing;
const ora_root = @import("ora_root");

const workspace = ora_root.lsp.workspace;

fn pathFromTmpAlloc(allocator: std.mem.Allocator, tmp: std.testing.TmpDir, rel_path: []const u8) ![]u8 {
    return std.fmt.allocPrint(allocator, ".zig-cache/tmp/{s}/{s}", .{ tmp.sub_path, rel_path });
}

fn toFileUriAlloc(allocator: std.mem.Allocator, path: []const u8) ![]u8 {
    return std.fmt.allocPrint(allocator, "file://{s}", .{path});
}

test "lsp workspace: decodes file uri to path" {
    const allocator = testing.allocator;
    const uri = "file:///tmp/ora%20example.ora";

    const maybe_path = try workspace.fileUriToPathAlloc(allocator, uri);
    try testing.expect(maybe_path != null);

    const path = maybe_path.?;
    defer allocator.free(path);

    try testing.expectEqualStrings("/tmp/ora example.ora", path);
}

test "lsp workspace: resolves relative imports in workspace" {
    const allocator = testing.allocator;
    var tmp = testing.tmpDir(.{});
    defer tmp.cleanup();

    try tmp.dir.makePath("project");
    try tmp.dir.writeFile(.{
        .sub_path = "project/entry.ora",
        .data = "const lib = @import(\"./lib.ora\");",
    });
    try tmp.dir.writeFile(.{
        .sub_path = "project/lib.ora",
        .data = "contract Lib { }",
    });

    const entry_path_rel = try pathFromTmpAlloc(allocator, tmp, "project/entry.ora");
    defer allocator.free(entry_path_rel);
    const entry_path = try std.fs.cwd().realpathAlloc(allocator, entry_path_rel);
    defer allocator.free(entry_path);

    const workspace_root_rel = try pathFromTmpAlloc(allocator, tmp, "project");
    defer allocator.free(workspace_root_rel);
    const workspace_root = try std.fs.cwd().realpathAlloc(allocator, workspace_root_rel);
    defer allocator.free(workspace_root);

    const uri = try toFileUriAlloc(allocator, entry_path);
    defer allocator.free(uri);

    const roots = [_][]const u8{workspace_root};
    var result = try workspace.resolveDocumentImports(
        allocator,
        uri,
        "const lib = @import(\"./lib.ora\");",
        .{ .workspace_roots = roots[0..] },
    );
    defer result.deinit(allocator);

    try testing.expectEqual(@as(usize, 0), result.diagnostics.len);
    try testing.expectEqual(@as(usize, 1), result.imports.len);
    try testing.expectEqualStrings("./lib.ora", result.imports[0].specifier);
    try testing.expectEqualStrings("lib.ora", std.fs.path.basename(result.imports[0].resolved_path));
}

test "lsp workspace: reports relative import without .ora extension" {
    const allocator = testing.allocator;
    var tmp = testing.tmpDir(.{});
    defer tmp.cleanup();

    try tmp.dir.makePath("project");
    try tmp.dir.writeFile(.{
        .sub_path = "project/entry.ora",
        .data = "const lib = @import(\"./lib\");",
    });

    const entry_path_rel = try pathFromTmpAlloc(allocator, tmp, "project/entry.ora");
    defer allocator.free(entry_path_rel);
    const entry_path = try std.fs.cwd().realpathAlloc(allocator, entry_path_rel);
    defer allocator.free(entry_path);

    const workspace_root_rel = try pathFromTmpAlloc(allocator, tmp, "project");
    defer allocator.free(workspace_root_rel);
    const workspace_root = try std.fs.cwd().realpathAlloc(allocator, workspace_root_rel);
    defer allocator.free(workspace_root);

    const uri = try toFileUriAlloc(allocator, entry_path);
    defer allocator.free(uri);

    const roots = [_][]const u8{workspace_root};
    var result = try workspace.resolveDocumentImports(
        allocator,
        uri,
        "const lib = @import(\"./lib\");",
        .{ .workspace_roots = roots[0..] },
    );
    defer result.deinit(allocator);

    try testing.expectEqual(@as(usize, 1), result.diagnostics.len);
    try testing.expect(std.mem.indexOf(u8, result.diagnostics[0].message, ".ora") != null);
}

test "lsp workspace: reports relative import escaping workspace root" {
    const allocator = testing.allocator;
    var tmp = testing.tmpDir(.{});
    defer tmp.cleanup();

    try tmp.dir.makePath("project/contracts");
    try tmp.dir.makePath("outside");
    try tmp.dir.writeFile(.{
        .sub_path = "project/contracts/entry.ora",
        .data = "const secret = @import(\"../../outside/secret.ora\");",
    });
    try tmp.dir.writeFile(.{
        .sub_path = "outside/secret.ora",
        .data = "contract Secret { }",
    });

    const entry_path_rel = try pathFromTmpAlloc(allocator, tmp, "project/contracts/entry.ora");
    defer allocator.free(entry_path_rel);
    const entry_path = try std.fs.cwd().realpathAlloc(allocator, entry_path_rel);
    defer allocator.free(entry_path);

    const workspace_root_rel = try pathFromTmpAlloc(allocator, tmp, "project");
    defer allocator.free(workspace_root_rel);
    const workspace_root = try std.fs.cwd().realpathAlloc(allocator, workspace_root_rel);
    defer allocator.free(workspace_root);

    const uri = try toFileUriAlloc(allocator, entry_path);
    defer allocator.free(uri);

    const roots = [_][]const u8{workspace_root};
    var result = try workspace.resolveDocumentImports(
        allocator,
        uri,
        "const secret = @import(\"../../outside/secret.ora\");",
        .{ .workspace_roots = roots[0..] },
    );
    defer result.deinit(allocator);

    try testing.expectEqual(@as(usize, 1), result.diagnostics.len);
    try testing.expect(std.mem.indexOf(u8, result.diagnostics[0].message, "escapes") != null);
}

test "lsp workspace: pathToFileUri converts simple path" {
    const allocator = testing.allocator;
    const uri = try workspace.pathToFileUri(allocator, "/tmp/project/main.ora");
    defer allocator.free(uri);
    try testing.expectEqualStrings("file:///tmp/project/main.ora", uri);
}

test "lsp workspace: pathToFileUri escapes spaces" {
    const allocator = testing.allocator;
    const uri = try workspace.pathToFileUri(allocator, "/tmp/my project/main.ora");
    defer allocator.free(uri);
    try testing.expectEqualStrings("file:///tmp/my%20project/main.ora", uri);
}

test "lsp workspace: accepts symlink workspace roots" {
    if (builtin.os.tag == .windows) return error.SkipZigTest;

    const allocator = testing.allocator;
    var tmp = testing.tmpDir(.{});
    defer tmp.cleanup();

    try tmp.dir.makePath("realroot");
    try tmp.dir.symLink("realroot", "aliasroot", .{ .is_directory = true });
    try tmp.dir.writeFile(.{
        .sub_path = "realroot/entry.ora",
        .data = "const lib = @import(\"./lib.ora\");",
    });
    try tmp.dir.writeFile(.{
        .sub_path = "realroot/lib.ora",
        .data = "contract Lib { }",
    });

    const cwd = try std.fs.cwd().realpathAlloc(allocator, ".");
    defer allocator.free(cwd);

    const entry_alias_rel = try pathFromTmpAlloc(allocator, tmp, "aliasroot/entry.ora");
    defer allocator.free(entry_alias_rel);
    const entry_alias_abs = try std.fs.path.resolve(allocator, &.{ cwd, entry_alias_rel });
    defer allocator.free(entry_alias_abs);

    const root_alias_rel = try pathFromTmpAlloc(allocator, tmp, "aliasroot");
    defer allocator.free(root_alias_rel);
    const root_alias_abs = try std.fs.path.resolve(allocator, &.{ cwd, root_alias_rel });
    defer allocator.free(root_alias_abs);

    const uri = try toFileUriAlloc(allocator, entry_alias_abs);
    defer allocator.free(uri);

    const roots = [_][]const u8{root_alias_abs};
    var result = try workspace.resolveDocumentImports(
        allocator,
        uri,
        "const lib = @import(\"./lib.ora\");",
        .{ .workspace_roots = roots[0..] },
    );
    defer result.deinit(allocator);

    try testing.expectEqual(@as(usize, 0), result.diagnostics.len);
    try testing.expectEqual(@as(usize, 1), result.imports.len);
    try testing.expectEqualStrings("lib.ora", std.fs.path.basename(result.imports[0].resolved_path));
}
