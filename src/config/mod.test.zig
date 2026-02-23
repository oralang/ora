const std = @import("std");
const testing = std.testing;
const config = @import("mod.zig");

fn pathFromTmpAlloc(allocator: std.mem.Allocator, tmp: std.testing.TmpDir, rel_path: []const u8) ![]u8 {
    return std.fmt.allocPrint(allocator, ".zig-cache/tmp/{s}/{s}", .{ tmp.sub_path, rel_path });
}

test "config: parse ora.toml targets and compiler output" {
    const allocator = testing.allocator;
    var tmp = testing.tmpDir(.{});
    defer tmp.cleanup();

    try tmp.dir.writeFile(.{
        .sub_path = "ora.toml",
        .data =
        \\schema_version = "0.1"
        \\
        \\[project]
        \\name = "demo"
        \\
        \\[compiler]
        \\output_dir = "build"
        \\init_args = ["owner=0x1111111111111111111111111111111111111111", "cap=1000"]
        \\
        \\[[targets]]
        \\name = "Token"
        \\kind = "contract"
        \\root = "contracts/Token.ora"
        \\include_paths = ["contracts", "lib"]
        \\init_args = ["cap=42"]
        \\
        \\[[targets]]
        \\name = "Math"
        \\kind = "library"
        \\root = "lib/Math.ora"
        \\output_dir = "out/math"
        ,
    });

    const config_path = try pathFromTmpAlloc(allocator, tmp, "ora.toml");
    defer allocator.free(config_path);

    var parsed = try config.loadProjectConfigFile(allocator, config_path);
    defer parsed.deinit(allocator);

    try testing.expectEqualStrings("0.1", parsed.schema_version);
    try testing.expect(parsed.compiler_output_dir != null);
    try testing.expectEqualStrings("build", parsed.compiler_output_dir.?);
    try testing.expectEqual(@as(usize, 2), parsed.compiler_init_args.len);
    try testing.expectEqualStrings("owner", parsed.compiler_init_args[0].name);
    try testing.expectEqualStrings("0x1111111111111111111111111111111111111111", parsed.compiler_init_args[0].value);
    try testing.expectEqualStrings("cap", parsed.compiler_init_args[1].name);
    try testing.expectEqualStrings("1000", parsed.compiler_init_args[1].value);

    try testing.expectEqual(@as(usize, 2), parsed.targets.len);

    try testing.expectEqualStrings("Token", parsed.targets[0].name);
    try testing.expectEqual(config.TargetKind.contract, parsed.targets[0].kind);
    try testing.expectEqualStrings("contracts/Token.ora", parsed.targets[0].root);
    try testing.expectEqual(@as(usize, 2), parsed.targets[0].include_paths.len);
    try testing.expectEqualStrings("contracts", parsed.targets[0].include_paths[0]);
    try testing.expectEqualStrings("lib", parsed.targets[0].include_paths[1]);
    try testing.expectEqual(@as(usize, 1), parsed.targets[0].init_args.len);
    try testing.expectEqualStrings("cap", parsed.targets[0].init_args[0].name);
    try testing.expectEqualStrings("42", parsed.targets[0].init_args[0].value);
    try testing.expect(parsed.targets[0].output_dir == null);

    try testing.expectEqualStrings("Math", parsed.targets[1].name);
    try testing.expectEqual(config.TargetKind.library, parsed.targets[1].kind);
    try testing.expectEqualStrings("lib/Math.ora", parsed.targets[1].root);
    try testing.expectEqual(@as(usize, 0), parsed.targets[1].init_args.len);
    try testing.expect(parsed.targets[1].output_dir != null);
    try testing.expectEqualStrings("out/math", parsed.targets[1].output_dir.?);
}

test "config: schema_version is required" {
    const allocator = testing.allocator;
    var tmp = testing.tmpDir(.{});
    defer tmp.cleanup();

    try tmp.dir.writeFile(.{
        .sub_path = "ora.toml",
        .data =
        \\[[targets]]
        \\name = "Token"
        \\root = "contracts/Token.ora"
        ,
    });

    const config_path = try pathFromTmpAlloc(allocator, tmp, "ora.toml");
    defer allocator.free(config_path);

    try testing.expectError(
        error.MissingSchemaVersion,
        config.loadProjectConfigFile(allocator, config_path),
    );
}

test "config: discovery walks up parent directories" {
    const allocator = testing.allocator;
    var tmp = testing.tmpDir(.{});
    defer tmp.cleanup();

    try tmp.dir.makePath("workspace/app/src");
    try tmp.dir.writeFile(.{
        .sub_path = "workspace/ora.toml",
        .data =
        \\schema_version = "0.1"
        \\
        \\[[targets]]
        \\name = "App"
        \\root = "app/src/main.ora"
        ,
    });

    const start_dir = try pathFromTmpAlloc(allocator, tmp, "workspace/app/src");
    defer allocator.free(start_dir);

    const loaded_opt = try config.loadDiscoveredFromStartDir(allocator, start_dir);
    try testing.expect(loaded_opt != null);

    var loaded = loaded_opt.?;
    defer loaded.deinit(allocator);

    try testing.expectEqualStrings("ora.toml", std.fs.path.basename(loaded.config_path));
    try testing.expect(loaded.config.targets.len == 1);
    try testing.expectEqualStrings("App", loaded.config.targets[0].name);

    const real_dir = try std.fs.cwd().realpathAlloc(allocator, loaded.config_dir);
    defer allocator.free(real_dir);
    try testing.expectEqualStrings("workspace", std.fs.path.basename(real_dir));
}

test "config: path resolution from config dir" {
    const allocator = testing.allocator;

    const resolved = try config.resolvePathFromConfigDir(allocator, "workspace", "contracts/Token.ora");
    defer allocator.free(resolved);
    try testing.expectEqualStrings("workspace/contracts/Token.ora", resolved);

    const same = try config.resolvePathFromConfigDir(allocator, ".", "contracts/Token.ora");
    defer allocator.free(same);
    try testing.expectEqualStrings("contracts/Token.ora", same);
}

test "config: find matching target index for entry file" {
    const allocator = testing.allocator;
    var tmp = testing.tmpDir(.{});
    defer tmp.cleanup();

    try tmp.dir.makePath("workspace/contracts");
    try tmp.dir.writeFile(.{
        .sub_path = "workspace/contracts/Token.ora",
        .data = "contract Token { }",
    });
    try tmp.dir.writeFile(.{
        .sub_path = "workspace/contracts/Pool.ora",
        .data = "contract Pool { }",
    });
    try tmp.dir.writeFile(.{
        .sub_path = "workspace/ora.toml",
        .data =
        \\schema_version = "0.1"
        \\
        \\[[targets]]
        \\name = "Token"
        \\root = "contracts/Token.ora"
        \\include_paths = ["lib"]
        \\
        \\[[targets]]
        \\name = "Pool"
        \\root = "contracts/Pool.ora"
        ,
    });

    const start_dir = try pathFromTmpAlloc(allocator, tmp, "workspace/contracts");
    defer allocator.free(start_dir);
    const entry_path = try pathFromTmpAlloc(allocator, tmp, "workspace/contracts/Pool.ora");
    defer allocator.free(entry_path);

    const loaded_opt = try config.loadDiscoveredFromStartDir(allocator, start_dir);
    try testing.expect(loaded_opt != null);

    var loaded = loaded_opt.?;
    defer loaded.deinit(allocator);

    const idx_opt = try config.findMatchingTargetIndex(allocator, &loaded, entry_path);
    try testing.expect(idx_opt != null);
    try testing.expectEqual(@as(usize, 1), idx_opt.?);
}

test "config: parse multiline string arrays for init_args and include_paths" {
    const allocator = testing.allocator;
    var tmp = testing.tmpDir(.{});
    defer tmp.cleanup();

    try tmp.dir.writeFile(.{
        .sub_path = "ora.toml",
        .data =
        \\schema_version = "0.1"
        \\
        \\[compiler]
        \\init_args = [
        \\  "owner=0x1111111111111111111111111111111111111111",
        \\  "cap=1000"
        \\]
        \\
        \\[[targets]]
        \\name = "Main"
        \\root = "contracts/Main.ora"
        \\include_paths = [
        \\  "contracts",
        \\  "lib"
        \\]
        \\init_args = [
        \\  "cap=7"
        \\]
        ,
    });

    const config_path = try pathFromTmpAlloc(allocator, tmp, "ora.toml");
    defer allocator.free(config_path);

    var parsed = try config.loadProjectConfigFile(allocator, config_path);
    defer parsed.deinit(allocator);

    try testing.expectEqual(@as(usize, 2), parsed.compiler_init_args.len);
    try testing.expectEqualStrings("owner", parsed.compiler_init_args[0].name);
    try testing.expectEqualStrings("0x1111111111111111111111111111111111111111", parsed.compiler_init_args[0].value);
    try testing.expectEqualStrings("cap", parsed.compiler_init_args[1].name);
    try testing.expectEqualStrings("1000", parsed.compiler_init_args[1].value);

    try testing.expectEqual(@as(usize, 1), parsed.targets.len);
    try testing.expectEqual(@as(usize, 2), parsed.targets[0].include_paths.len);
    try testing.expectEqualStrings("contracts", parsed.targets[0].include_paths[0]);
    try testing.expectEqualStrings("lib", parsed.targets[0].include_paths[1]);
    try testing.expectEqual(@as(usize, 1), parsed.targets[0].init_args.len);
    try testing.expectEqualStrings("cap", parsed.targets[0].init_args[0].name);
    try testing.expectEqualStrings("7", parsed.targets[0].init_args[0].value);
}

test "config: invalid init_args format errors" {
    const allocator = testing.allocator;
    var tmp = testing.tmpDir(.{});
    defer tmp.cleanup();

    try tmp.dir.writeFile(.{
        .sub_path = "ora.toml",
        .data =
        \\schema_version = "0.1"
        \\
        \\[compiler]
        \\init_args = ["broken"]
        \\
        \\[[targets]]
        \\name = "Main"
        \\root = "contracts/Main.ora"
        ,
    });

    const config_path = try pathFromTmpAlloc(allocator, tmp, "ora.toml");
    defer allocator.free(config_path);

    try testing.expectError(error.InvalidToml, config.loadProjectConfigFile(allocator, config_path));
}

test "config: duplicate init_args names error" {
    const allocator = testing.allocator;
    var tmp = testing.tmpDir(.{});
    defer tmp.cleanup();

    try tmp.dir.writeFile(.{
        .sub_path = "ora.toml",
        .data =
        \\schema_version = "0.1"
        \\
        \\[[targets]]
        \\name = "Main"
        \\root = "contracts/Main.ora"
        \\init_args = ["owner=0x1111111111111111111111111111111111111111", "owner=0x2222222222222222222222222222222222222222"]
        ,
    });

    const config_path = try pathFromTmpAlloc(allocator, tmp, "ora.toml");
    defer allocator.free(config_path);

    try testing.expectError(error.InvalidToml, config.loadProjectConfigFile(allocator, config_path));
}

test "config: legacy defines keys are rejected" {
    const allocator = testing.allocator;
    var tmp = testing.tmpDir(.{});
    defer tmp.cleanup();

    try tmp.dir.writeFile(.{
        .sub_path = "ora.toml",
        .data =
        \\schema_version = "0.1"
        \\
        \\[compiler]
        \\defines = ["FEATURE_ON=true"]
        \\
        \\[[targets]]
        \\name = "Main"
        \\root = "contracts/Main.ora"
        ,
    });

    const config_path = try pathFromTmpAlloc(allocator, tmp, "ora.toml");
    defer allocator.free(config_path);

    try testing.expectError(error.InvalidToml, config.loadProjectConfigFile(allocator, config_path));
}
