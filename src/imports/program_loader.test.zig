const std = @import("std");
const testing = std.testing;
const loader = @import("program_loader.zig");
const lib = @import("ora_lib");

fn pathFromTmpAlloc(allocator: std.mem.Allocator, tmp: std.testing.TmpDir, rel_path: []const u8) ![]u8 {
    return std.fmt.allocPrint(allocator, ".zig-cache/tmp/{s}/{s}", .{ tmp.sub_path, rel_path });
}

test "program loader: preserves module-qualified calls and injects imported runtime fn into contract" {
    const allocator = testing.allocator;
    var tmp = testing.tmpDir(.{});
    defer tmp.cleanup();

    try tmp.dir.writeFile(.{
        .sub_path = "math.ora",
        .data =
        \\pub fn add(a: u256, b: u256) -> u256 {
        \\    return a + b;
        \\}
        ,
    });

    try tmp.dir.writeFile(.{
        .sub_path = "entry.ora",
        .data =
        \\const math = @import("./math.ora");
        \\
        \\contract LoaderSmoke {
        \\    pub fn run() -> u256 {
        \\        return math.add(1, 2);
        \\    }
        \\}
        ,
    });

    const entry_path = try pathFromTmpAlloc(allocator, tmp, "entry.ora");
    defer allocator.free(entry_path);

    var program = try loader.loadProgramWithImportsRaw(allocator, entry_path);
    defer program.deinit();

    var saw_top_level_math_fn = false;
    var saw_injected_math_fn = false;
    var saw_contract = false;
    var saw_import = false;

    for (program.nodes) |node| {
        switch (node) {
            .Import => {
                saw_import = true;
            },
            .Function => |f| {
                if (std.mem.eql(u8, f.name, "add")) saw_top_level_math_fn = true;
            },
            .Contract => |c| {
                if (!std.mem.eql(u8, c.name, "LoaderSmoke")) continue;
                saw_contract = true;

                for (c.body) |member| {
                    if (member != .Function) continue;
                    const f = member.Function;
                    if (std.mem.eql(u8, f.name, "add")) {
                        try testing.expectEqual(lib.ast.Visibility.Private, f.visibility);
                        saw_injected_math_fn = true;
                        continue;
                    }
                    if (!std.mem.eql(u8, f.name, "run")) continue;

                    try testing.expectEqual(@as(usize, 1), f.body.statements.len);
                    const stmt = f.body.statements[0];
                    try testing.expect(stmt == .Return);
                    const ret_expr = stmt.Return.value.?;

                    // Callee must remain as FieldAccess(math, add) — namespace preserved.
                    try testing.expect(ret_expr == .Call);
                    const callee = ret_expr.Call.callee.*;
                    try testing.expect(callee == .FieldAccess);
                    try testing.expectEqualStrings("add", callee.FieldAccess.field);
                    try testing.expect(callee.FieldAccess.target.* == .Identifier);
                    try testing.expectEqualStrings("math", callee.FieldAccess.target.Identifier.name);
                }
            },
            else => {},
        }
    }

    try testing.expect(!saw_top_level_math_fn);
    try testing.expect(saw_injected_math_fn);
    try testing.expect(saw_contract);
    try testing.expect(saw_import);
}

test "program loader: preserves field access for non-exported member" {
    const allocator = testing.allocator;
    var tmp = testing.tmpDir(.{});
    defer tmp.cleanup();

    try tmp.dir.writeFile(.{
        .sub_path = "math.ora",
        .data = "pub fn add(a: u256, b: u256) -> u256 { return a + b; }",
    });

    try tmp.dir.writeFile(.{
        .sub_path = "entry.ora",
        .data =
        \\const math = @import("./math.ora");
        \\
        \\contract LoaderFieldAccess {
        \\    pub fn run() -> u256 {
        \\        return math.VERSION;
        \\    }
        \\}
        ,
    });

    const entry_path = try pathFromTmpAlloc(allocator, tmp, "entry.ora");
    defer allocator.free(entry_path);

    var program = try loader.loadProgramWithImportsRaw(allocator, entry_path);
    defer program.deinit();

    var checked = false;
    for (program.nodes) |node| {
        if (node != .Contract) continue;
        const c = node.Contract;
        if (!std.mem.eql(u8, c.name, "LoaderFieldAccess")) continue;

        for (c.body) |member| {
            if (member != .Function) continue;
            const f = member.Function;
            if (!std.mem.eql(u8, f.name, "run")) continue;

            const stmt = f.body.statements[0];
            try testing.expect(stmt == .Return);
            const ret_expr = stmt.Return.value.?;
            try testing.expect(ret_expr == .FieldAccess);
            try testing.expectEqualStrings("VERSION", ret_expr.FieldAccess.field);
            try testing.expect(ret_expr.FieldAccess.target.* == .Identifier);
            try testing.expectEqualStrings("math", ret_expr.FieldAccess.target.Identifier.name);
            checked = true;
        }
    }

    try testing.expect(checked);
}

test "program loader: keeps imported runtime fn at module level when entry has no contract" {
    const allocator = testing.allocator;
    var tmp = testing.tmpDir(.{});
    defer tmp.cleanup();

    try tmp.dir.writeFile(.{
        .sub_path = "math.ora",
        .data =
        \\pub fn add(a: u256, b: u256) -> u256 {
        \\    return a + b;
        \\}
        ,
    });

    try tmp.dir.writeFile(.{
        .sub_path = "entry.ora",
        .data =
        \\const math = @import("./math.ora");
        \\
        \\pub fn run() -> u256 {
        \\    return math.add(1, 2);
        \\}
        ,
    });

    const entry_path = try pathFromTmpAlloc(allocator, tmp, "entry.ora");
    defer allocator.free(entry_path);

    var program = try loader.loadProgramWithImportsRaw(allocator, entry_path);
    defer program.deinit();

    var saw_top_level_add = false;
    for (program.nodes) |node| {
        if (node == .Function and std.mem.eql(u8, node.Function.name, "add")) {
            saw_top_level_add = true;
        }
    }
    try testing.expect(saw_top_level_add);
}

test "program loader: preserves qualified struct access" {
    const allocator = testing.allocator;
    var tmp = testing.tmpDir(.{});
    defer tmp.cleanup();

    try tmp.dir.writeFile(.{
        .sub_path = "types.ora",
        .data =
        \\struct Point {
        \\    x: u256;
        \\    y: u256;
        \\}
        \\
        \\pub fn origin() -> u256 {
        \\    return 0;
        \\}
        ,
    });

    try tmp.dir.writeFile(.{
        .sub_path = "entry.ora",
        .data =
        \\const types = @import("./types.ora");
        \\
        \\contract Geometry {
        \\    pub fn getOrigin() -> u256 {
        \\        return types.origin();
        \\    }
        \\}
        ,
    });

    const entry_path = try pathFromTmpAlloc(allocator, tmp, "entry.ora");
    defer allocator.free(entry_path);

    var program = try loader.loadProgramWithImportsRaw(allocator, entry_path);
    defer program.deinit();

    var saw_struct = false;
    var checked_call = false;
    for (program.nodes) |node| {
        if (node == .StructDecl and std.mem.eql(u8, node.StructDecl.name, "Point")) {
            saw_struct = true;
        }
        if (node != .Contract) continue;
        if (!std.mem.eql(u8, node.Contract.name, "Geometry")) continue;

        for (node.Contract.body) |member| {
            if (member != .Function) continue;
            if (!std.mem.eql(u8, member.Function.name, "getOrigin")) continue;

            const stmt = member.Function.body.statements[0];
            try testing.expect(stmt == .Return);
            const ret_expr = stmt.Return.value.?;

            // types.origin() preserved as FieldAccess call target
            try testing.expect(ret_expr == .Call);
            try testing.expect(ret_expr.Call.callee.* == .FieldAccess);
            try testing.expectEqualStrings("origin", ret_expr.Call.callee.FieldAccess.field);
            try testing.expect(ret_expr.Call.callee.FieldAccess.target.* == .Identifier);
            try testing.expectEqualStrings("types", ret_expr.Call.callee.FieldAccess.target.Identifier.name);
            checked_call = true;
        }
    }

    try testing.expect(saw_struct);
    try testing.expect(checked_call);
}

test "program loader: same-named exports across modules allowed with namespaces" {
    const allocator = testing.allocator;
    var tmp = testing.tmpDir(.{});
    defer tmp.cleanup();

    try tmp.dir.writeFile(.{
        .sub_path = "a.ora",
        .data =
        \\pub fn helper() -> u256 {
        \\    return 1;
        \\}
        ,
    });

    try tmp.dir.writeFile(.{
        .sub_path = "b.ora",
        .data =
        \\pub fn helper() -> u256 {
        \\    return 2;
        \\}
        ,
    });

    try tmp.dir.writeFile(.{
        .sub_path = "entry.ora",
        .data =
        \\const a = @import("./a.ora");
        \\const b = @import("./b.ora");
        \\
        \\contract Collision {
        \\    pub fn run() -> u256 {
        \\        return a.helper() + b.helper();
        \\    }
        \\}
        ,
    });

    const entry_path = try pathFromTmpAlloc(allocator, tmp, "entry.ora");
    defer allocator.free(entry_path);

    var program = try loader.loadProgramWithImportsRaw(allocator, entry_path);
    defer program.deinit();

    // Both a.helper and b.helper should be in the ModuleExportMap
    const me = program.module_exports.?;
    try testing.expect(me.lookupExport("a", "helper") != null);
    try testing.expect(me.lookupExport("b", "helper") != null);
}

test "program loader: typed load succeeds for imported function usage" {
    const allocator = testing.allocator;
    var tmp = testing.tmpDir(.{});
    defer tmp.cleanup();

    try tmp.dir.writeFile(.{
        .sub_path = "math.ora",
        .data =
        \\pub fn add(a: u256, b: u256) -> u256 {
        \\    return a + b;
        \\}
        ,
    });

    try tmp.dir.writeFile(.{
        .sub_path = "entry.ora",
        .data =
        \\const math = @import("./math.ora");
        \\
        \\contract LoaderTyped {
        \\    pub fn run() -> u256 {
        \\        return math.add(40, 2);
        \\    }
        \\}
        ,
    });

    const entry_path = try pathFromTmpAlloc(allocator, tmp, "entry.ora");
    defer allocator.free(entry_path);

    var program = try loader.loadProgramWithImportsTyped(allocator, entry_path);
    defer program.deinit();

    var saw_contract = false;
    var saw_runtime_add = false;
    for (program.nodes) |node| {
        if (node == .Contract and std.mem.eql(u8, node.Contract.name, "LoaderTyped")) {
            saw_contract = true;
            for (node.Contract.body) |member| {
                if (member != .Function) continue;
                const f = member.Function;
                if (std.mem.eql(u8, f.name, "add")) {
                    try testing.expect(!f.is_comptime_only);
                    saw_runtime_add = true;
                }
            }
        }
    }

    try testing.expect(saw_contract);
    try testing.expect(saw_runtime_add);
}

test "program loader: preserves qualified access with include_roots" {
    const allocator = testing.allocator;
    var tmp = testing.tmpDir(.{});
    defer tmp.cleanup();

    try tmp.dir.makePath("deps/acme");
    try tmp.dir.writeFile(.{
        .sub_path = "deps/acme/math.ora",
        .data =
        \\pub fn add(a: u256, b: u256) -> u256 {
        \\    return a + b;
        \\}
        ,
    });

    try tmp.dir.writeFile(.{
        .sub_path = "entry.ora",
        .data =
        \\const math = @import("acme/math");
        \\
        \\contract LoaderIncludeRoots {
        \\    pub fn run() -> u256 {
        \\        return math.add(1, 2);
        \\    }
        \\}
        ,
    });

    const entry_path = try pathFromTmpAlloc(allocator, tmp, "entry.ora");
    defer allocator.free(entry_path);
    const deps_root = try pathFromTmpAlloc(allocator, tmp, "deps");
    defer allocator.free(deps_root);

    const include_roots = [_][]const u8{deps_root};
    var program = try loader.loadProgramWithImportsRawWithResolverOptions(allocator, entry_path, .{
        .include_roots = include_roots[0..],
    });
    defer program.deinit();

    var checked = false;
    for (program.nodes) |node| {
        if (node != .Contract) continue;
        if (!std.mem.eql(u8, node.Contract.name, "LoaderIncludeRoots")) continue;

        for (node.Contract.body) |member| {
            if (member != .Function) continue;
            if (!std.mem.eql(u8, member.Function.name, "run")) continue;

            const stmt = member.Function.body.statements[0];
            try testing.expect(stmt == .Return);
            const ret_expr = stmt.Return.value.?;
            try testing.expect(ret_expr == .Call);
            // Callee should be FieldAccess(math, add)
            try testing.expect(ret_expr.Call.callee.* == .FieldAccess);
            try testing.expectEqualStrings("add", ret_expr.Call.callee.FieldAccess.field);
            checked = true;
        }
    }

    try testing.expect(checked);
}

test "program loader: normalizes enum-literal style calls to qualified access" {
    const allocator = testing.allocator;
    var tmp = testing.tmpDir(.{});
    defer tmp.cleanup();

    try tmp.dir.writeFile(.{
        .sub_path = "fee_math.ora",
        .data =
        \\pub fn feeFromBps(amount: u256, bps: u256) -> u256 {
        \\    return amount + bps;
        \\}
        ,
    });

    try tmp.dir.writeFile(.{
        .sub_path = "entry.ora",
        .data =
        \\const fee_math = @import("./fee_math.ora");
        \\
        \\contract LoaderAliasUnderscore {
        \\    pub fn run() -> u256 {
        \\        return fee_math.feeFromBps(40, 2);
        \\    }
        \\}
        ,
    });

    const entry_path = try pathFromTmpAlloc(allocator, tmp, "entry.ora");
    defer allocator.free(entry_path);

    var program = try loader.loadProgramWithImportsRaw(allocator, entry_path);
    defer program.deinit();

    var checked = false;
    for (program.nodes) |node| {
        if (node != .Contract) continue;
        if (!std.mem.eql(u8, node.Contract.name, "LoaderAliasUnderscore")) continue;

        for (node.Contract.body) |member| {
            if (member != .Function) continue;
            if (!std.mem.eql(u8, member.Function.name, "run")) continue;

            const stmt = member.Function.body.statements[0];
            try testing.expect(stmt == .Return);
            const ret_expr = stmt.Return.value.?;
            try testing.expect(ret_expr == .Call);
            // After EnumLiteral normalization, callee is FieldAccess(fee_math, feeFromBps)
            const callee = ret_expr.Call.callee.*;
            try testing.expect(callee == .FieldAccess);
            try testing.expectEqualStrings("feeFromBps", callee.FieldAccess.field);
            try testing.expect(callee.FieldAccess.target.* == .Identifier);
            try testing.expectEqualStrings("fee_math", callee.FieldAccess.target.Identifier.name);
            checked = true;
        }
    }

    try testing.expect(checked);

    // Also ensure type-resolved loading succeeds for this pattern.
    var typed_program = try loader.loadProgramWithImportsTyped(allocator, entry_path);
    defer typed_program.deinit();
}

test "program loader: duplicate import alias to different target errors" {
    const allocator = testing.allocator;
    var tmp = testing.tmpDir(.{});
    defer tmp.cleanup();

    try tmp.dir.writeFile(.{
        .sub_path = "a.ora",
        .data = "pub fn helperA() -> u256 { return 1; }",
    });

    try tmp.dir.writeFile(.{
        .sub_path = "b.ora",
        .data = "pub fn helperB() -> u256 { return 2; }",
    });

    try tmp.dir.writeFile(.{
        .sub_path = "entry.ora",
        .data =
        \\const x = @import("./a.ora");
        \\const x = @import("./b.ora");
        \\
        \\contract DuplicateAlias {
        \\    pub fn run() -> u256 {
        \\        return x.helperA();
        \\    }
        \\}
        ,
    });

    const entry_path = try pathFromTmpAlloc(allocator, tmp, "entry.ora");
    defer allocator.free(entry_path);

    const result = loader.loadProgramWithImportsRaw(allocator, entry_path);
    try testing.expectError(error.DuplicateImportAlias, result);
}

test "program loader: same alias same target deduplicates without error" {
    const allocator = testing.allocator;
    var tmp = testing.tmpDir(.{});
    defer tmp.cleanup();

    try tmp.dir.writeFile(.{
        .sub_path = "math.ora",
        .data = "pub fn add(a: u256, b: u256) -> u256 { return a + b; }",
    });

    try tmp.dir.writeFile(.{
        .sub_path = "entry.ora",
        .data =
        \\const math = @import("./math.ora");
        \\const math = @import("./math.ora");
        \\
        \\contract DeduplicateAlias {
        \\    pub fn run() -> u256 {
        \\        return math.add(1, 2);
        \\    }
        \\}
        ,
    });

    const entry_path = try pathFromTmpAlloc(allocator, tmp, "entry.ora");
    defer allocator.free(entry_path);

    var program = try loader.loadProgramWithImportsRaw(allocator, entry_path);
    defer program.deinit();

    try testing.expect(program.module_exports != null);
    try testing.expect(program.module_exports.?.isModuleAlias("math"));
}

test "program loader: builds ModuleExportMap correctly" {
    const allocator = testing.allocator;
    var tmp = testing.tmpDir(.{});
    defer tmp.cleanup();

    try tmp.dir.writeFile(.{
        .sub_path = "math.ora",
        .data =
        \\pub fn add(a: u256, b: u256) -> u256 {
        \\    return a + b;
        \\}
        ,
    });

    try tmp.dir.writeFile(.{
        .sub_path = "entry.ora",
        .data =
        \\const math = @import("./math.ora");
        \\
        \\contract ExportMapTest {
        \\    pub fn run() -> u256 {
        \\        return math.add(1, 2);
        \\    }
        \\}
        ,
    });

    const entry_path = try pathFromTmpAlloc(allocator, tmp, "entry.ora");
    defer allocator.free(entry_path);

    var program = try loader.loadProgramWithImportsRaw(allocator, entry_path);
    defer program.deinit();

    try testing.expect(program.module_exports != null);
    const me = program.module_exports.?;
    try testing.expect(me.isModuleAlias("math"));
    try testing.expect(!me.isModuleAlias("nonexistent"));
    try testing.expect(me.lookupExport("math", "add") != null);
    try testing.expectEqual(loader.ExportKind.Function, me.lookupExport("math", "add").?);
    try testing.expect(me.lookupExport("math", "nonexistent") == null);
}

test "program loader: typed load does not inject compiler define-like names" {
    const allocator = testing.allocator;
    var tmp = testing.tmpDir(.{});
    defer tmp.cleanup();

    try tmp.dir.writeFile(.{
        .sub_path = "entry.ora",
        .data =
        \\contract DefineTypedMissing {
        \\    pub fn run() -> u256 {
        \\        if (FEATURE_ON) {
        \\            return 1;
        \\        }
        \\        return 0;
        \\    }
        \\}
        ,
    });

    const entry_path = try pathFromTmpAlloc(allocator, tmp, "entry.ora");
    defer allocator.free(entry_path);

    try testing.expectError(error.UndefinedIdentifier, loader.loadProgramWithImportsTyped(allocator, entry_path));
}

test "program loader: imported library rejects init function" {
    const allocator = testing.allocator;
    var tmp = testing.tmpDir(.{});
    defer tmp.cleanup();

    try tmp.dir.writeFile(.{
        .sub_path = "lib.ora",
        .data =
        \\pub fn init() {
        \\}
        ,
    });

    try tmp.dir.writeFile(.{
        .sub_path = "entry.ora",
        .data =
        \\const lib = @import("./lib.ora");
        \\
        \\contract InitReject {
        \\    pub fn run() -> u256 {
        \\        return 1;
        \\    }
        \\}
        ,
    });

    const entry_path = try pathFromTmpAlloc(allocator, tmp, "entry.ora");
    defer allocator.free(entry_path);

    const result = loader.loadProgramWithImportsRaw(allocator, entry_path);
    try testing.expectError(error.ImportedLibraryInitNotAllowed, result);
}

test "program loader: imported library rejects private runtime function" {
    const allocator = testing.allocator;
    var tmp = testing.tmpDir(.{});
    defer tmp.cleanup();

    try tmp.dir.writeFile(.{
        .sub_path = "lib.ora",
        .data =
        \\fn helper() -> u256 {
        \\    return 1;
        \\}
        ,
    });

    try tmp.dir.writeFile(.{
        .sub_path = "entry.ora",
        .data =
        \\const lib = @import("./lib.ora");
        \\
        \\contract PrivateReject {
        \\    pub fn run() -> u256 {
        \\        return 1;
        \\    }
        \\}
        ,
    });

    const entry_path = try pathFromTmpAlloc(allocator, tmp, "entry.ora");
    defer allocator.free(entry_path);

    const result = loader.loadProgramWithImportsRaw(allocator, entry_path);
    try testing.expectError(error.ImportedLibraryFunctionVisibilityNotAllowed, result);
}

test "program loader: imported library rejects contract declarations" {
    const allocator = testing.allocator;
    var tmp = testing.tmpDir(.{});
    defer tmp.cleanup();

    try tmp.dir.writeFile(.{
        .sub_path = "lib.ora",
        .data = "contract Helper { }",
    });

    try tmp.dir.writeFile(.{
        .sub_path = "entry.ora",
        .data =
        \\const lib = @import("./lib.ora");
        \\
        \\contract ContractReject {
        \\    pub fn run() -> u256 {
        \\        return 1;
        \\    }
        \\}
        ,
    });

    const entry_path = try pathFromTmpAlloc(allocator, tmp, "entry.ora");
    defer allocator.free(entry_path);

    const result = loader.loadProgramWithImportsRaw(allocator, entry_path);
    try testing.expectError(error.ImportedLibraryDeclarationNotAllowed, result);
}

test "program loader: imported library rejects top-level state variable" {
    const allocator = testing.allocator;
    var tmp = testing.tmpDir(.{});
    defer tmp.cleanup();

    try tmp.dir.writeFile(.{
        .sub_path = "lib.ora",
        .data = "storage var counter: u256 = 0;",
    });

    try tmp.dir.writeFile(.{
        .sub_path = "entry.ora",
        .data =
        \\const lib = @import("./lib.ora");
        \\
        \\contract StateReject {
        \\    pub fn run() -> u256 {
        \\        return 1;
        \\    }
        \\}
        ,
    });

    const entry_path = try pathFromTmpAlloc(allocator, tmp, "entry.ora");
    defer allocator.free(entry_path);

    const result = loader.loadProgramWithImportsRaw(allocator, entry_path);
    try testing.expectError(error.ImportedLibraryStateNotAllowed, result);
}

test "program loader: imported library rejects enum declarations in v1" {
    const allocator = testing.allocator;
    var tmp = testing.tmpDir(.{});
    defer tmp.cleanup();

    try tmp.dir.writeFile(.{
        .sub_path = "lib.ora",
        .data =
        \\enum Mode {
        \\    A,
        \\    B
        \\}
        ,
    });

    try tmp.dir.writeFile(.{
        .sub_path = "entry.ora",
        .data =
        \\const lib = @import("./lib.ora");
        \\
        \\contract EnumReject {
        \\    pub fn run() -> u256 {
        \\        return 1;
        \\    }
        \\}
        ,
    });

    const entry_path = try pathFromTmpAlloc(allocator, tmp, "entry.ora");
    defer allocator.free(entry_path);

    const result = loader.loadProgramWithImportsRaw(allocator, entry_path);
    try testing.expectError(error.ImportedLibraryDeclarationNotAllowed, result);
}

test "program loader: imported library allows pub fn, struct, and bitfield exports" {
    const allocator = testing.allocator;
    var tmp = testing.tmpDir(.{});
    defer tmp.cleanup();

    try tmp.dir.writeFile(.{
        .sub_path = "util.ora",
        .data =
        \\struct Point {
        \\    x: u256;
        \\    y: u256;
        \\}
        \\
        \\bitfield Flags : u256 {
        \\    enabled: bool;
        \\}
        \\
        \\pub fn add(a: u256, b: u256) -> u256 {
        \\    return a + b;
        \\}
        ,
    });

    try tmp.dir.writeFile(.{
        .sub_path = "entry.ora",
        .data =
        \\const util = @import("./util.ora");
        \\
        \\contract AllowedLib {
        \\    pub fn run() -> u256 {
        \\        return util.add(1, 2);
        \\    }
        \\}
        ,
    });

    const entry_path = try pathFromTmpAlloc(allocator, tmp, "entry.ora");
    defer allocator.free(entry_path);

    var program = try loader.loadProgramWithImportsRaw(allocator, entry_path);
    defer program.deinit();

    const me = program.module_exports.?;
    try testing.expectEqual(loader.ExportKind.Function, me.lookupExport("util", "add").?);
    try testing.expectEqual(loader.ExportKind.StructDecl, me.lookupExport("util", "Point").?);
    try testing.expectEqual(loader.ExportKind.BitfieldDecl, me.lookupExport("util", "Flags").?);
}

test "program loader: typed load rejects calling non-function module export" {
    const allocator = testing.allocator;
    var tmp = testing.tmpDir(.{});
    defer tmp.cleanup();

    try tmp.dir.writeFile(.{
        .sub_path = "util.ora",
        .data =
        \\struct Point {
        \\    x: u256;
        \\    y: u256;
        \\}
        ,
    });

    try tmp.dir.writeFile(.{
        .sub_path = "entry.ora",
        .data =
        \\const util = @import("./util.ora");
        \\
        \\contract BadCall {
        \\    pub fn run() -> u256 {
        \\        return util.Point();
        \\    }
        \\}
        ,
    });

    const entry_path = try pathFromTmpAlloc(allocator, tmp, "entry.ora");
    defer allocator.free(entry_path);

    try testing.expectError(error.TypeMismatch, loader.loadProgramWithImportsTyped(allocator, entry_path));
}

test "program loader: module export map carries exported type info" {
    const allocator = testing.allocator;
    var tmp = testing.tmpDir(.{});
    defer tmp.cleanup();

    try tmp.dir.writeFile(.{
        .sub_path = "util.ora",
        .data =
        \\struct Point {
        \\    x: u256;
        \\    y: u256;
        \\}
        \\
        \\bitfield Flags : u256 {
        \\    enabled: bool;
        \\}
        \\
        \\pub fn add(a: u256, b: u256) -> u256 {
        \\    return a + b;
        \\}
        ,
    });

    try tmp.dir.writeFile(.{
        .sub_path = "entry.ora",
        .data =
        \\const util = @import("./util.ora");
        \\
        \\contract ExportTypes {
        \\    pub fn run() -> u256 {
        \\        return util.add(1, 2);
        \\    }
        \\}
        ,
    });

    const entry_path = try pathFromTmpAlloc(allocator, tmp, "entry.ora");
    defer allocator.free(entry_path);

    var program = try loader.loadProgramWithImportsRaw(allocator, entry_path);
    defer program.deinit();

    const me = program.module_exports.?;

    const add_ty = me.lookupExportType("util", "add").?;
    try testing.expectEqual(lib.ast.Types.TypeCategory.Integer, add_ty.category);
    try testing.expect(add_ty.ora_type != null);
    try testing.expect(add_ty.ora_type.? == .u256);

    const point_ty = me.lookupExportType("util", "Point").?;
    try testing.expectEqual(lib.ast.Types.TypeCategory.Struct, point_ty.category);
    try testing.expect(point_ty.ora_type != null);
    try testing.expect(point_ty.ora_type.? == .struct_type);
    try testing.expectEqualStrings("Point", point_ty.ora_type.?.struct_type);

    const flags_ty = me.lookupExportType("util", "Flags").?;
    try testing.expectEqual(lib.ast.Types.TypeCategory.Bitfield, flags_ty.category);
    try testing.expect(flags_ty.ora_type != null);
    try testing.expect(flags_ty.ora_type.? == .bitfield_type);
    try testing.expectEqualStrings("Flags", flags_ty.ora_type.?.bitfield_type);
}

test "program loader: typed load infers imported type declaration for qualified access" {
    const allocator = testing.allocator;
    var tmp = testing.tmpDir(.{});
    defer tmp.cleanup();

    try tmp.dir.writeFile(.{
        .sub_path = "util.ora",
        .data =
        \\struct Point {
        \\    x: u256;
        \\    y: u256;
        \\}
        ,
    });

    try tmp.dir.writeFile(.{
        .sub_path = "entry.ora",
        .data =
        \\const util = @import("./util.ora");
        \\
        \\contract ImportedTypeInference {
        \\    pub fn run() -> u256 {
        \\        var p = util.Point;
        \\        return 0;
        \\    }
        \\}
        ,
    });

    const entry_path = try pathFromTmpAlloc(allocator, tmp, "entry.ora");
    defer allocator.free(entry_path);

    var program = try loader.loadProgramWithImportsTyped(allocator, entry_path);
    defer program.deinit();

    var saw_inferred_struct = false;
    for (program.nodes) |node| {
        if (node != .Contract or !std.mem.eql(u8, node.Contract.name, "ImportedTypeInference")) continue;
        for (node.Contract.body) |member| {
            if (member != .Function or !std.mem.eql(u8, member.Function.name, "run")) continue;
            if (member.Function.body.statements.len == 0) continue;
            const first_stmt = member.Function.body.statements[0];
            if (first_stmt != .VariableDecl) continue;

            const var_decl = first_stmt.VariableDecl;
            try testing.expectEqual(lib.ast.Types.TypeCategory.Struct, var_decl.type_info.category);
            try testing.expect(var_decl.type_info.ora_type != null);
            try testing.expect(var_decl.type_info.ora_type.? == .struct_type);
            try testing.expectEqualStrings("Point", var_decl.type_info.ora_type.?.struct_type);
            saw_inferred_struct = true;
        }
    }

    try testing.expect(saw_inferred_struct);
}

test "program loader: typed load supports transitive imports (A -> B -> C)" {
    const allocator = testing.allocator;
    var tmp = testing.tmpDir(.{});
    defer tmp.cleanup();

    try tmp.dir.writeFile(.{
        .sub_path = "c.ora",
        .data =
        \\pub fn fC() -> u256 {
        \\    return 1;
        \\}
        ,
    });

    try tmp.dir.writeFile(.{
        .sub_path = "b.ora",
        .data =
        \\const lib_c = @import("./c.ora");
        \\
        \\pub fn fB() -> u256 {
        \\    return lib_c.fC();
        \\}
        ,
    });

    try tmp.dir.writeFile(.{
        .sub_path = "a.ora",
        .data =
        \\const lib_b = @import("./b.ora");
        \\
        \\pub fn fA() -> u256 {
        \\    return lib_b.fB();
        \\}
        ,
    });

    try tmp.dir.writeFile(.{
        .sub_path = "entry.ora",
        .data =
        \\const lib_a = @import("./a.ora");
        \\
        \\contract TransitiveImports {
        \\    pub fn run() -> u256 {
        \\        return lib_a.fA();
        \\    }
        \\}
        ,
    });

    const entry_path = try pathFromTmpAlloc(allocator, tmp, "entry.ora");
    defer allocator.free(entry_path);

    var program = try loader.loadProgramWithImportsTyped(allocator, entry_path);
    defer program.deinit();

    const me = program.module_exports.?;
    try testing.expectEqual(loader.ExportKind.Function, me.lookupExport("lib_a", "fA").?);
    try testing.expectEqual(loader.ExportKind.Function, me.lookupExport("lib_b", "fB").?);
    try testing.expectEqual(loader.ExportKind.Function, me.lookupExport("lib_c", "fC").?);
}

test "program loader: cycle detection surfaces as loader error" {
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

    try testing.expectError(error.ImportCycleDetected, loader.loadProgramWithImportsRaw(allocator, entry_path));
}

test "program loader: allows multiple aliases to the same module target" {
    const allocator = testing.allocator;
    var tmp = testing.tmpDir(.{});
    defer tmp.cleanup();

    try tmp.dir.writeFile(.{
        .sub_path = "math.ora",
        .data =
        \\pub fn add(a: u256, b: u256) -> u256 {
        \\    return a + b;
        \\}
        ,
    });

    try tmp.dir.writeFile(.{
        .sub_path = "entry.ora",
        .data =
        \\const m1 = @import("./math.ora");
        \\const m2 = @import("./math.ora");
        \\
        \\contract MultiAliasSameTarget {
        \\    pub fn run() -> u256 {
        \\        return m1.add(1, 2) + m2.add(3, 4);
        \\    }
        \\}
        ,
    });

    const entry_path = try pathFromTmpAlloc(allocator, tmp, "entry.ora");
    defer allocator.free(entry_path);

    var program = try loader.loadProgramWithImportsTyped(allocator, entry_path);
    defer program.deinit();

    const me = program.module_exports.?;
    try testing.expectEqual(loader.ExportKind.Function, me.lookupExport("m1", "add").?);
    try testing.expectEqual(loader.ExportKind.Function, me.lookupExport("m2", "add").?);
}

test "program loader: module export map supports constant export metadata" {
    const allocator = testing.allocator;

    var me = loader.ModuleExportMap.init(allocator);
    defer me.deinit();

    const alias = try allocator.dupe(u8, "math");
    var exports = std.StringHashMap(loader.ExportInfo).init(allocator);
    const member = try allocator.dupe(u8, "FEE_BPS");
    try exports.put(member, .{
        .kind = .Constant,
        .type_info = lib.ast.Types.TypeInfo.fromOraType(.{ .u256 = {} }),
    });
    try me.entries.put(alias, exports);

    try testing.expectEqual(loader.ExportKind.Constant, me.lookupExport("math", "FEE_BPS").?);
    const type_info = me.lookupExportType("math", "FEE_BPS").?;
    try testing.expectEqual(lib.ast.Types.TypeCategory.Integer, type_info.category);
    try testing.expect(type_info.ora_type != null);
    try testing.expect(type_info.ora_type.? == .u256);
}

test "program loader: include_roots resolves transitive package imports with .ora auto-append" {
    const allocator = testing.allocator;
    var tmp = testing.tmpDir(.{});
    defer tmp.cleanup();

    try tmp.dir.makePath("deps/acme");
    try tmp.dir.writeFile(.{
        .sub_path = "deps/acme/util.ora",
        .data =
        \\pub fn one() -> u256 {
        \\    return 1;
        \\}
        ,
    });
    try tmp.dir.writeFile(.{
        .sub_path = "deps/acme/math.ora",
        .data =
        \\const util_pkg = @import("acme/util");
        \\
        \\pub fn addOne(x: u256) -> u256 {
        \\    return x + util_pkg.one();
        \\}
        ,
    });

    try tmp.dir.writeFile(.{
        .sub_path = "entry.ora",
        .data =
        \\const math_pkg = @import("acme/math");
        \\
        \\contract IncludeRootsTransitive {
        \\    pub fn run() -> u256 {
        \\        return math_pkg.addOne(41);
        \\    }
        \\}
        ,
    });

    const entry_path = try pathFromTmpAlloc(allocator, tmp, "entry.ora");
    defer allocator.free(entry_path);
    const deps_root = try pathFromTmpAlloc(allocator, tmp, "deps");
    defer allocator.free(deps_root);

    const include_roots = [_][]const u8{deps_root};
    var program = try loader.loadProgramWithImportsRawWithResolverOptions(allocator, entry_path, .{
        .include_roots = include_roots[0..],
    });
    defer program.deinit();

    const me = program.module_exports.?;
    try testing.expectEqual(loader.ExportKind.Function, me.lookupExport("math_pkg", "addOne").?);
    try testing.expectEqual(loader.ExportKind.Function, me.lookupExport("util_pkg", "one").?);
}
