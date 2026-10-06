const std = @import("std");
const testing = std.testing;
const helpers = @import("compiler.test.helpers.zig");
const mlir = @import("mlir_c_api").c;
const z3 = @import("ora_z3_verification");
const data = @import("formal/storage_lifecycle.zig");
const collector = @import("formal/storage_lifecycle_from_mlir.zig");
const emitter = @import("formal/storage_lifecycle_to_lean.zig");
const canonical_collector = @import("formal/obligation_from_mlir.zig");
const canonical_lean = @import("formal/obligation_to_lean.zig");
const canonical_z3 = @import("formal/obligation_to_z3.zig");

test "SL1 lifecycle inventory records all eight goal sources without a grant" {
    const sources = [_][]const u8{
        "ora-example/corpus/types/refinement/in_range_percentage.ora",
        "ora-example/corpus/types/refinement/in_range_u8.ora",
        "ora-example/corpus/types/refinement/max_value_in_function.ora",
        "ora-example/corpus/types/refinement/max_value_u256.ora",
        "ora-example/corpus/types/refinement/max_value_u8.ora",
        "ora-example/refinements/dispatcher_refinement_e2e.ora",
        "ora-example/refinements/patterns/supply_cap.ora",
        "ora-example/corpus/patterns/kitchen_sink.ora",
    };
    var pass = try z3.VerificationPass.init(testing.allocator);
    defer pass.deinit();
    for (sources, 0..) |source, index| {
        var compilation = try helpers.compilePackage(source);
        defer compilation.deinit();
        const hir = try compilation.db.lowerToHir(compilation.root_module_id);
        try testing.expect(hir.diagnostics.isEmpty());
        var snapshot = collector.collect(testing.allocator, hir.module.raw_module, &pass) catch |err| {
            std.debug.print("SL1 inventory {s}: {s}\n", .{ source, @errorName(err) });
            return err;
        };
        defer snapshot.deinit();
        const inv = snapshot.inventory;
        try testing.expect(inv.functions.len != 0);
        if (index == sources.len - 1) {
            try testing.expectEqual(@as(usize, 0), inv.claims.len);
        } else {
            try testing.expect(inv.claims.len != 0);
            var supported: usize = 0;
            for (inv.claims, 0..) |claim, claim_index| {
                if (claim.refusal != null) continue;
                supported += 1;
                try testing.expect(claim.zero_base);
                var module = std.Io.Writer.Allocating.init(testing.allocator);
                defer module.deinit();
                try emitter.writeClaim(&module.writer, testing.allocator, inv, claim_index);
                try testing.expect(std.mem.indexOf(u8, module.written(), "candidateClaim") != null);
                try testing.expect(std.mem.indexOf(u8, module.written(), "theorem") == null);
            }
            try testing.expect(supported != 0);
        }
        try collector.checkFresh(testing.allocator, hir.module.raw_module, &pass, inv);
    }
}

test "SL1 lifecycle inventory compares submitted rows against fresh HIR" {
    var compilation = try helpers.compileText(
        "contract Census { storage var cap: MaxValue<u256, 100>; pub fn init() { cap = 5; } fn hidden(x: MaxValue<u256, 100>) { if (x > 0) { cap = x; } } pub fn put(x: MaxValue<u256, 100>) { hidden(x); } pub fn get() -> u256 { return cap; } }",
    );
    defer compilation.deinit();
    const hir = try compilation.db.lowerToHir(compilation.root_module_id);
    try testing.expect(hir.diagnostics.isEmpty());
    var pass = try z3.VerificationPass.init(testing.allocator);
    defer pass.deinit();
    var snapshot = try collector.collect(testing.allocator, hir.module.raw_module, &pass);
    defer snapshot.deinit();
    const original = snapshot.inventory;
    try testing.expectEqual(@as(usize, 1), original.claims.len);
    try testing.expectEqual(@as(u256, 100), original.claims[0].bounds.?.maximum);
    var constructors: usize = 0;
    var writes: usize = 0;
    var hidden_runtime = false;
    for (original.functions) |function| {
        if (function.constructor) constructors += 1;
        if (std.mem.eql(u8, function.symbol, "hidden") and function.runtime) hidden_runtime = true;
        for (function.events) |event| {
            if (std.mem.eql(u8, event.operation, "ora.sstore")) {
                writes += 1;
                try testing.expectEqualStrings("cap", event.target.?);
            }
        }
    }
    try testing.expectEqual(@as(usize, 1), constructors);
    try testing.expect(writes >= 2);
    try testing.expect(hidden_runtime);
    var changed = original;
    changed.functions = original.functions[1..];
    try testing.expectError(error.LifecycleInventoryMismatch, collector.checkFresh(testing.allocator, hir.module.raw_module, &pass, changed));
    changed = original;
    var claim = original.claims[0];
    claim.bounds.?.maximum = 101;
    changed.claims = &.{claim};
    try testing.expectError(error.LifecycleInventoryMismatch, collector.checkFresh(testing.allocator, hir.module.raw_module, &pass, changed));
    claim = original.claims[0];
    claim.owner = "module/contract:Foreign";
    changed.claims = &.{claim};
    try testing.expectError(error.LifecycleInventoryMismatch, collector.checkFresh(testing.allocator, hir.module.raw_module, &pass, changed));
    changed.claims = &.{ original.claims[0], original.claims[0] };
    try testing.expectError(error.DuplicateLifecycleClaim, collector.checkFresh(testing.allocator, hir.module.raw_module, &pass, changed));
    changed = original;
    const functions = try testing.allocator.dupe(data.Function, original.functions);
    defer testing.allocator.free(functions);
    changed.functions = functions;
    functions[0].events = &.{};
    functions[0].hir_sha256[0] ^= 1;
    try testing.expectError(error.LifecycleInventoryMismatch, collector.checkFresh(testing.allocator, hir.module.raw_module, &pass, changed));
    // Each completeness field fails independently; a body hash cannot cover a
    // forged owner, omitted nested effect, or a changed constructor flag.
    const writer = found: {
        for (original.functions, 0..) |function, i| {
            for (function.events) |event| if (std.mem.eql(u8, event.operation, "ora.sstore")) break :found i;
        }
        return error.TestUnexpectedResult;
    };
    for (0..7) |mutation| {
        @memcpy(functions, original.functions);
        var events = try testing.allocator.dupe(data.Event, functions[writer].events);
        defer testing.allocator.free(events);
        switch (mutation) {
            0 => functions[writer].owner = "foreign/owner",
            1 => functions[writer].runtime = !functions[writer].runtime,
            2 => functions[writer].constructor = !functions[writer].constructor,
            3 => functions[writer].events = &.{},
            4 => functions[writer].hir_sha256[0] ^= 1,
            5 => {
                events[0].hir_sha256[0] ^= 1;
                functions[writer].events = events;
            },
            6 => {
                events[0].target = "foreign_slot";
                functions[writer].events = events;
            },
            else => unreachable,
        }
        try testing.expectError(error.LifecycleInventoryMismatch, collector.checkFresh(testing.allocator, hir.module.raw_module, &pass, changed));
    }
    changed = original;
    changed.functions = &.{ original.functions[0], original.functions[0] };
    try testing.expectError(error.AmbiguousLifecycleFunction, collector.checkFresh(testing.allocator, hir.module.raw_module, &pass, changed));
    changed.functions = &.{};
    changed.claims = &.{};
    try testing.expectError(error.LifecycleInventoryMismatch, collector.checkFresh(testing.allocator, hir.module.raw_module, &pass, changed));
    try testing.expectError(error.MissingLifecycleModule, collector.collect(testing.allocator, .{ .ptr = null }, &pass));
    pass.filter_function_name = "get";
    try testing.expectError(error.IncompleteRuntimeInventory, collector.collect(testing.allocator, hir.module.raw_module, &pass));
}

fn firstOperation(op: mlir.MlirOperation, expected: []const u8) ?mlir.MlirOperation {
    const name = mlir.oraOperationGetName(op);
    defer if (name.data != null) mlir.oraStringRefFree(name);
    if (name.data == null) return null;
    if (std.mem.eql(u8, name.data[0..name.length], expected)) return op;
    for (0..@intCast(mlir.oraOperationGetNumRegions(op))) |i| {
        var block = mlir.oraRegionGetFirstBlock(mlir.oraOperationGetRegion(op, i));
        while (!mlir.oraBlockIsNull(block)) : (block = mlir.oraBlockGetNextInRegion(block)) {
            var child = mlir.oraBlockGetFirstOperation(block);
            while (!mlir.oraOperationIsNull(child)) : (child = mlir.oraOperationGetNextInBlock(child)) {
                if (firstOperation(child, expected)) |found| return found;
            }
        }
    }
    return null;
}

test "SL1 live HIR mutations invalidate census and preserve unresolved-call refusal" {
    var compilation = try helpers.compileText("contract Fresh { storage var cap: MaxValue<u8, 100>; fn hidden(x: u8) { cap = x; } pub fn put(x: u8) { hidden(x); } }");
    defer compilation.deinit();
    const hir = try compilation.db.lowerToHir(compilation.root_module_id);
    try testing.expect(hir.diagnostics.isEmpty());
    const root = mlir.oraModuleGetOperation(hir.module.raw_module);
    var pass = try z3.VerificationPass.init(testing.allocator);
    defer pass.deinit();
    var before = try collector.collect(testing.allocator, hir.module.raw_module, &pass);
    defer before.deinit();
    const call = firstOperation(root, "func.call") orelse return error.TestUnexpectedResult;
    const context = mlir.mlirOperationGetContext(call);
    mlir.oraOperationSetAttributeByName(call, mlir.oraStringRefCreate("callee", 6), mlir.mlirFlatSymbolRefAttrGet(context, mlir.oraStringRefCreate("missing", 7)));
    try testing.expectError(error.LifecycleInventoryMismatch, collector.checkFresh(testing.allocator, hir.module.raw_module, &pass, before.inventory));
    var changed = try collector.collect(testing.allocator, hir.module.raw_module, &pass);
    defer changed.deinit();
    var unresolved = false;
    for (changed.inventory.functions) |function| {
        if (std.mem.eql(u8, function.symbol, "hidden")) try testing.expect(!function.runtime);
        for (function.events) |event| if (event.kind == .call) {
            try testing.expectEqual(.unresolved_call, event.refusal.?);
            unresolved = true;
        };
    }
    try testing.expect(unresolved);
    const global = firstOperation(root, "ora.global") orelse return error.TestUnexpectedResult;
    const integer_type = mlir.mlirIntegerTypeGet(context, 256);
    mlir.oraOperationSetAttributeByName(global, mlir.oraStringRefCreate("ora.compiler_type_id", 20), mlir.mlirIntegerAttrGet(integer_type, 6));
    var wrong_width = try collector.collect(testing.allocator, hir.module.raw_module, &pass);
    defer wrong_width.deinit();
    try testing.expectEqual(data.Refusal.width_mismatch, wrong_width.inventory.claims[0].refusal.?);
    mlir.oraOperationSetAttributeByName(global, mlir.oraStringRefCreate("ora.compiler_type_id", 20), mlir.mlirIntegerAttrGet(integer_type, 1));
    mlir.oraOperationSetAttributeByName(global, mlir.oraStringRefCreate("init", 4), mlir.mlirIntegerAttrGet(integer_type, 5));
    var initializer = try collector.collect(testing.allocator, hir.module.raw_module, &pass);
    defer initializer.deinit();
    try testing.expectEqual(data.Refusal.unsupported_initialization, initializer.inventory.claims[0].refusal.?);
    _ = mlir.mlirOperationRemoveAttributeByName(global, mlir.oraStringRefCreate("ora.compiler_type_id", 20));
    var missing = try collector.collect(testing.allocator, hir.module.raw_module, &pass);
    defer missing.deinit();
    try testing.expectEqual(data.Refusal.missing_type, missing.inventory.claims[0].refusal.?);
    _ = mlir.mlirOperationRemoveAttributeByName(global, mlir.oraStringRefCreate("type", 4));
    try testing.expectError(error.InvalidLifecycleType, collector.collect(testing.allocator, hir.module.raw_module, &pass));
}

test "SL1 imported callee cannot be omitted and external effects are not frames" {
    var tmp = testing.tmpDir(.{});
    defer tmp.cleanup();
    try tmp.dir.writeFile(testing.io, .{ .sub_path = "dep.ora", .data = "pub fn helper(x: u256) -> u256 { return x; }" });
    try tmp.dir.writeFile(testing.io, .{ .sub_path = "main.ora", .data = "comptime const dep = @import(\"./dep.ora\"); contract Imported { storage var cap: MaxValue<u256,100>; pub fn put(x:u256) { cap = dep.helper(x); } }" });
    const path = try tmp.dir.realPathFileAlloc(testing.io, "main.ora", testing.allocator);
    defer testing.allocator.free(path);
    var compilation = try helpers.compilePackage(path);
    defer compilation.deinit();
    const hir = try compilation.db.lowerToHir(compilation.root_module_id);
    try testing.expect(hir.diagnostics.isEmpty());
    var pass = try z3.VerificationPass.init(testing.allocator);
    defer pass.deinit();
    var snapshot = try collector.collect(testing.allocator, hir.module.raw_module, &pass);
    defer snapshot.deinit();
    const imported = for (snapshot.inventory.functions, 0..) |function, i| {
        if (std.mem.endsWith(u8, function.module_path, "dep.ora") and function.runtime) break i;
    } else return error.TestUnexpectedResult;
    const functions = try testing.allocator.dupe(data.Function, snapshot.inventory.functions);
    defer testing.allocator.free(functions);
    functions[imported].events = &.{};
    functions[imported].owner = "invented";
    var forged = snapshot.inventory;
    forged.functions = functions;
    try testing.expectError(error.LifecycleInventoryMismatch, collector.checkFresh(testing.allocator, hir.module.raw_module, &pass, forged));

    var external = try helpers.compilePackage("ora-example/smt/modifies/pass_staticcall_modifies_empty_frame.ora");
    defer external.deinit();
    const external_hir = try external.db.lowerToHir(external.root_module_id);
    try testing.expect(external_hir.diagnostics.isEmpty());
    var external_inventory = try collector.collect(testing.allocator, external_hir.module.raw_module, &pass);
    defer external_inventory.deinit();
    var refused = false;
    for (external_inventory.inventory.functions) |function| for (function.events) |event| {
        if (std.mem.eql(u8, event.operation, "ora.external_call")) {
            try testing.expectEqual(.external_interference, event.refusal.?);
            refused = true;
        }
        if (event.kind == .unknown) try testing.expectEqual(.unknown_effect, event.refusal.?);
    };
    try testing.expect(refused);
}

test "SL1 exact wide signed and unsupported type data is never widened" {
    const source = "contract Types { storage var wide: InRange<u256, 25108406941546723056364004793593481054836439088298861789185, 50216813883093446113068291954107900573173146272176910893061>; storage var narrow: MaxValue<i128, 100>; storage var unsupported: NonZeroAddress; pub fn raw() -> u256 { return wide; } }";
    var compilation = try helpers.compileText(source);
    defer compilation.deinit();
    const hir = try compilation.db.lowerToHir(compilation.root_module_id);
    try testing.expect(hir.diagnostics.isEmpty());
    var pass = try z3.VerificationPass.init(testing.allocator);
    defer pass.deinit();
    var snapshot = try collector.collect(testing.allocator, hir.module.raw_module, &pass);
    defer snapshot.deinit();
    try testing.expectEqual(@as(usize, 3), snapshot.inventory.claims.len);
    try testing.expectEqual(@as(u256, 25108406941546723056364004793593481054836439088298861789185), snapshot.inventory.claims[0].bounds.?.minimum);
    try testing.expectEqual(@as(u256, 50216813883093446113068291954107900573173146272176910893061), snapshot.inventory.claims[0].bounds.?.maximum);
    try testing.expectEqual(data.Refusal.unsupported_type, snapshot.inventory.claims[2].refusal.?);
    var output = std.Io.Writer.Allocating.init(testing.allocator);
    defer output.deinit();
    try emitter.writeClaim(&output.writer, testing.allocator, snapshot.inventory, 1);
    try testing.expect(std.mem.indexOf(u8, output.written(), ".signed .w128") != null);
    try testing.expect(std.mem.indexOf(u8, output.written(), "module/contract:Types/slot:narrow") != null);
}

test "SL1 lifecycle inventory refuses unresolved types and keeps actual initialization" {
    const source = "contract Initial { storage var slot: MinValue<u256, 1> = 5; pub fn raw() -> u256 { return slot; } }";
    var compilation = try helpers.compileText(source);
    defer compilation.deinit();
    const hir = try compilation.db.lowerToHir(compilation.root_module_id);
    try testing.expect(hir.diagnostics.isEmpty());
    var pass = try z3.VerificationPass.init(testing.allocator);
    defer pass.deinit();
    var snapshot = try collector.collect(testing.allocator, hir.module.raw_module, &pass);
    defer snapshot.deinit();
    try testing.expect(snapshot.inventory.claims[0].zero_base);
    try testing.expectEqual(@as(u256, 1), snapshot.inventory.claims[0].bounds.?.minimum);
    var changed = snapshot.inventory;
    var claim = changed.claims[0];
    claim.type_id = 0;
    changed.claims = &.{claim};
    try testing.expectError(error.InvalidLifecycleType, changed.validate());
    claim = snapshot.inventory.claims[0];
    claim.refusal = .unsupported_type;
    changed.claims = &.{claim};
    var output = std.Io.Writer.Allocating.init(testing.allocator);
    defer output.deinit();
    try testing.expectError(error.UnsupportedLifecycleClaim, emitter.writeClaim(&output.writer, testing.allocator, changed, 0));
}

test "SL1 emitted exact claim data kernel checks and invalid zero does not" {
    for ([_]bool{ false, true }) |bad| {
        const source = if (bad)
            "contract KernelClaim { storage var cap: MinValue<u8, 1>; pub fn get() -> u8 { return cap; } }"
        else
            "contract KernelClaim { storage var cap: MaxValue<u8, 100>; pub fn get() -> u8 { return cap; } }";
        var compilation = try helpers.compileText(source);
        defer compilation.deinit();
        const hir = try compilation.db.lowerToHir(compilation.root_module_id);
        try testing.expect(hir.diagnostics.isEmpty());
        var pass = try z3.VerificationPass.init(testing.allocator);
        defer pass.deinit();
        var snapshot = try collector.collect(testing.allocator, hir.module.raw_module, &pass);
        defer snapshot.deinit();
        var output = std.Io.Writer.Allocating.init(testing.allocator);
        defer output.deinit();
        try emitter.writeClaim(&output.writer, testing.allocator, snapshot.inventory, 0);
        try output.writer.writeAll(
            "open Ora.Generated.LifecycleInputs\n" ++
                "theorem zero_check : candidateClaim.holds zeroAtClaim := by\n",
        );
        try output.writer.writeAll(if (bad) "  change (1 : Nat) ≤ 0\n  decide\n" else "  change (0 : Nat) ≤ 100\n  decide\n");
        if (!bad) try output.writer.writeAll("#print axioms zero_check\n");
        var tmp = testing.tmpDir(.{});
        defer tmp.cleanup();
        try tmp.dir.writeFile(testing.io, .{ .sub_path = "claim.lean", .data = output.written() });
        const path = try tmp.dir.realPathFileAlloc(testing.io, "claim.lean", testing.allocator);
        defer testing.allocator.free(path);
        const home = std.c.getenv("HOME") orelse return error.LeanHomeMissing;
        const inherited_path = std.c.getenv("PATH") orelse return error.LeanPathMissing;
        const path_env = try std.fmt.allocPrint(testing.allocator, "PATH={s}/.elan/bin:{s}", .{ std.mem.span(home), std.mem.span(inherited_path) });
        defer testing.allocator.free(path_env);
        var io = std.Io.Threaded.init(testing.allocator, .{ .async_limit = .nothing, .concurrent_limit = .nothing });
        defer io.deinit();
        const run = try std.process.run(testing.allocator, io.io(), .{
            .argv = &.{ "/usr/bin/env", path_env, "lake", "env", "lean", path },
            .cwd = .{ .path = "formal" },
            .stdout_limit = .limited(1024 * 1024),
            .stderr_limit = .limited(1024 * 1024),
        });
        defer testing.allocator.free(run.stdout);
        defer testing.allocator.free(run.stderr);
        errdefer std.debug.print("SL1 kernel bad={}\n{s}\n{s}", .{ bad, run.stdout, run.stderr });
        try testing.expect(run.term == .exited);
        try testing.expectEqual(@as(u8, if (bad) 1 else 0), run.term.exited);
        if (bad) {
            try testing.expect(std.mem.indexOf(u8, run.stdout, "false") != null);
        } else {
            try testing.expect(std.mem.indexOf(u8, run.stdout, "zero_check") != null);
            try testing.expect(std.mem.indexOf(u8, run.stdout, "sorryAx") == null);
            try testing.expect(std.mem.indexOf(u8, run.stdout, "Classical") == null);
        }
    }
}

test "SL2A explicit fresh roots bind refined storage bodies without granting authority" {
    for ([_][]const u8{ "u8", "u256" }) |width| {
        const source = try std.fmt.allocPrint(
            testing.allocator,
            "contract Roots {{ storage var cap: MaxValue<{s}, 100>; pub fn init() {{ cap = 5; }} pub fn put(x: MaxValue<{s}, 100>) {{ cap = x; }} pub fn get() -> MaxValue<{s}, 100> {{ return cap; }} }}",
            .{ width, width, width },
        );
        defer testing.allocator.free(source);
        var compilation = try helpers.compileText(source);
        defer compilation.deinit();
        const hir = try compilation.db.lowerToHir(compilation.root_module_id);
        try testing.expect(hir.diagnostics.isEmpty());
        var pass = try z3.VerificationPass.init(testing.allocator);
        defer pass.deinit();
        var snapshot = try collector.collect(testing.allocator, hir.module.raw_module, &pass);
        defer snapshot.deinit();
        var projected = try collector.project(testing.allocator, hir.module.raw_module, &pass, snapshot.inventory);
        defer projected.deinit();
        try testing.expectEqual(@as(usize, 3), projected.bindings.len);
        try testing.expect(projected.canonical.set.artifactDecision() == .blocked);
        try testing.expectEqualStrings("cap", projected.snapshot.inventory.claims[0].root);
        var output = std.Io.Writer.Allocating.init(testing.allocator);
        defer output.deinit();
        try emitter.writeProjection(&output.writer, testing.allocator, projected.snapshot.inventory, projected.canonical.set, projected.bindings);
        for (projected.bindings) |binding| {
            const digest = try canonical_lean.functionVcSha256(projected.canonical.set, binding.function_id);
            try testing.expectEqualSlices(u8, &digest, &binding.function_vc_sha256);
            try testing.expect(std.mem.indexOf(u8, output.written(), &std.fmt.bytesToHex(digest, .lower)) != null);
        }
        try testing.expect(std.mem.indexOf(u8, output.written(), "lifecycleInventorySha256") != null);
        try testing.expect(std.mem.indexOf(u8, output.written(), "def emittedCallCustody_1") != null);
        try testing.expect(std.mem.indexOf(u8, output.written(), "sorry") == null);
        var ordinary = try canonical_collector.collect(testing.allocator, hir.module.raw_module, .{ .project_calls = true });
        defer ordinary.deinit();
        try testing.expectEqual(@as(usize, 0), ordinary.set.function_summaries.len);
        try lifecycleDataKernelCheck(output.written());
    }
}

fn lifecycleDataKernelCheck(source: []const u8) !void {
    var tmp = testing.tmpDir(.{});
    defer tmp.cleanup();
    try tmp.dir.writeFile(testing.io, .{ .sub_path = "projection.lean", .data = source });
    const path = try tmp.dir.realPathFileAlloc(testing.io, "projection.lean", testing.allocator);
    defer testing.allocator.free(path);
    const home = std.c.getenv("HOME") orelse return error.LeanHomeMissing;
    const executable = try std.fmt.allocPrint(testing.allocator, "{s}/.elan/bin/lake", .{std.mem.span(home)});
    defer testing.allocator.free(executable);
    var io = std.Io.Threaded.init(testing.allocator, .{ .async_limit = .nothing, .concurrent_limit = .nothing });
    defer io.deinit();
    const run = try std.process.run(testing.allocator, io.io(), .{ .argv = &.{ executable, "env", "lean", path }, .cwd = .{ .path = "formal" }, .stdout_limit = .limited(1024 * 1024), .stderr_limit = .limited(1024 * 1024) });
    defer testing.allocator.free(run.stdout);
    defer testing.allocator.free(run.stderr);
    errdefer std.debug.print("SL2A emitted data kernel check:\n{s}\n{s}", .{ run.stdout, run.stderr });
    try testing.expect(run.term == .exited);
    try testing.expectEqual(@as(u8, 0), run.term.exited);
}

test "SL2A roots reject absent duplicate foreign and cross-owner selection" {
    var compilation = try helpers.compileText("contract Owner { storage var cap: MaxValue<u8, 100>; pub fn put(x: MaxValue<u8, 100>) { cap = x; } pub fn get() -> u8 { return cap; } }");
    defer compilation.deinit();
    const hir = try compilation.db.lowerToHir(compilation.root_module_id);
    var pass = try z3.VerificationPass.init(testing.allocator);
    defer pass.deinit();
    var snapshot = try collector.collect(testing.allocator, hir.module.raw_module, &pass);
    defer snapshot.deinit();
    const op = snapshot.function_ops[0];
    try testing.expectError(error.MissingLifecycleRoots, canonical_collector.collect(testing.allocator, hir.module.raw_module, .{ .project_calls = true, .lifecycle_roots = &.{} }));
    try testing.expectError(error.MissingLifecycleRoots, canonical_collector.collect(testing.allocator, hir.module.raw_module, .{ .lifecycle_roots = &.{op} }));
    try testing.expectError(error.DuplicateLifecycleRoot, canonical_collector.collect(testing.allocator, hir.module.raw_module, .{ .project_calls = true, .lifecycle_roots = &.{ op, op } }));
    try testing.expectError(error.ForeignLifecycleRoot, canonical_collector.collect(testing.allocator, hir.module.raw_module, .{ .project_calls = true, .lifecycle_roots = &.{.{ .ptr = null }} }));
    var foreign = try helpers.compileText("contract Foreign { pub fn get() -> u8 { return 0; } }");
    defer foreign.deinit();
    const foreign_hir = try foreign.db.lowerToHir(foreign.root_module_id);
    const foreign_op = firstOperation(mlir.oraModuleGetOperation(foreign_hir.module.raw_module), "func.func").?;
    try testing.expectError(error.ForeignLifecycleRoot, canonical_collector.collect(testing.allocator, hir.module.raw_module, .{ .project_calls = true, .lifecycle_roots = &.{foreign_op} }));
    var ambiguous = try helpers.compileText("contract A { storage var cap: MaxValue<u8, 100>; pub fn a() -> u8 { return cap; } } contract B { storage var cap: MaxValue<u8, 100>; pub fn b() -> u8 { return cap; } }");
    defer ambiguous.deinit();
    const ambiguous_hir = try ambiguous.db.lowerToHir(ambiguous.root_module_id);
    var ambiguous_snapshot = try collector.collect(testing.allocator, ambiguous_hir.module.raw_module, &pass);
    defer ambiguous_snapshot.deinit();
    try testing.expectError(error.LifecycleOwnerUnsupported, collector.project(testing.allocator, ambiguous_hir.module.raw_module, &pass, ambiguous_snapshot.inventory));
}

test "SL2A fresh inventory and semantic digest both reject binding mutations" {
    var compilation = try helpers.compileText("contract Bound { storage var cap: MaxValue<u8, 100>; pub fn put(x: MaxValue<u8, 100>) { cap = x; } pub fn get() -> u8 { return cap; } }");
    defer compilation.deinit();
    const hir = try compilation.db.lowerToHir(compilation.root_module_id);
    var pass = try z3.VerificationPass.init(testing.allocator);
    defer pass.deinit();
    var snapshot = try collector.collect(testing.allocator, hir.module.raw_module, &pass);
    defer snapshot.deinit();
    var projected = try collector.project(testing.allocator, hir.module.raw_module, &pass, snapshot.inventory);
    defer projected.deinit();
    var omitted = snapshot.inventory;
    omitted.functions = omitted.functions[1..];
    try testing.expectError(error.LifecycleInventoryMismatch, collector.project(testing.allocator, hir.module.raw_module, &pass, omitted));
    const functions = try testing.allocator.dupe(data.Function, snapshot.inventory.functions);
    defer testing.allocator.free(functions);
    var changed = snapshot.inventory;
    changed.functions = functions;
    for (0..4) |mutation| {
        @memcpy(functions, snapshot.inventory.functions);
        switch (mutation) {
            0 => functions[0].owner = "module/contract:Foreign/function:put",
            1 => functions[0].runtime = false,
            2 => functions[0].hir_sha256[0] ^= 1,
            3 => functions[0].events = &.{},
            else => unreachable,
        }
        try testing.expectError(error.LifecycleInventoryMismatch, collector.project(testing.allocator, hir.module.raw_module, &pass, changed));
    }
    try testing.expectError(error.MissingLifecycleBinding, data.validateBindings(snapshot.inventory, projected.bindings[1..]));
    const bindings = try testing.allocator.dupe(data.FunctionBinding, projected.bindings);
    defer testing.allocator.free(bindings);
    var output = std.Io.Writer.Allocating.init(testing.allocator);
    defer output.deinit();
    bindings[0].function_vc_sha256[0] ^= 1;
    try testing.expectError(error.LifecycleDigestMismatch, emitter.writeProjection(&output.writer, testing.allocator, projected.snapshot.inventory, projected.canonical.set, bindings));
    try testing.expectEqual(@as(usize, 0), output.written().len);
    @memcpy(bindings, projected.bindings);
    bindings[1].function_id = bindings[0].function_id;
    try testing.expectError(error.DuplicateLifecycleBinding, data.validateBindings(snapshot.inventory, bindings));
    @memcpy(bindings, projected.bindings);
    bindings[0].function_ordinal = std.math.maxInt(u32);
    try testing.expectError(error.ForeignLifecycleBinding, data.validateBindings(snapshot.inventory, bindings));
    pass.filter_function_name = "get";
    try testing.expectError(error.IncompleteRuntimeInventory, collector.project(testing.allocator, hir.module.raw_module, &pass, snapshot.inventory));
}

test "SL2A storage metadata is exact and stale HIR cannot reuse a binding" {
    var compilation = try helpers.compileText("contract Metadata { storage var cap: MaxValue<u8, 100>; pub fn get() -> u8 { return cap; } }");
    defer compilation.deinit();
    const hir = try compilation.db.lowerToHir(compilation.root_module_id);
    const root = mlir.oraModuleGetOperation(hir.module.raw_module);
    const global = firstOperation(root, "ora.global").?;
    const context = mlir.mlirOperationGetContext(global);
    const attr_name = mlir.oraStringRefCreate("ora.compiler_type_id", 20);
    var pass = try z3.VerificationPass.init(testing.allocator);
    defer pass.deinit();
    var original = try collector.collect(testing.allocator, hir.module.raw_module, &pass);
    defer original.deinit();
    for ([_]u32{ 6, 0 }) |id| {
        if (id == 0) {
            try testing.expect(mlir.mlirOperationRemoveAttributeByName(global, attr_name));
        } else mlir.oraOperationSetAttributeByName(global, attr_name, mlir.mlirIntegerAttrGet(mlir.mlirIntegerTypeGet(context, 256), id));
        try testing.expectError(error.LifecycleInventoryMismatch, collector.project(testing.allocator, hir.module.raw_module, &pass, original.inventory));
        var fresh = try collector.collect(testing.allocator, hir.module.raw_module, &pass);
        defer fresh.deinit();
        try testing.expectEqual(if (id == 0) data.Refusal.missing_type else data.Refusal.width_mismatch, fresh.inventory.claims[0].refusal.?);
        try testing.expectError(error.LifecycleProjectionUnsupported, collector.project(testing.allocator, hir.module.raw_module, &pass, fresh.inventory));
        var rows = try canonical_collector.collect(testing.allocator, hir.module.raw_module, .{ .project_calls = true, .lifecycle_roots = fresh.function_ops });
        defer rows.deinit();
        const reason = if (id == 0) "LifecycleStorageTypeMissing" else "LifecycleStorageWidthMismatch";
        const named = for (rows.set.diagnostics) |diagnostic| {
            if (std.mem.indexOf(u8, diagnostic.message, reason) != null) break true;
        } else false;
        try testing.expect(named);
    }
}

test "SL2A private and imported dependencies remain in the fresh binding set" {
    for ([_]bool{ false, true }) |imported| {
        var tmp = testing.tmpDir(.{});
        defer tmp.cleanup();
        try tmp.dir.writeFile(testing.io, .{ .sub_path = "dep.ora", .data = "pub fn helper(x: u256) -> u256 { return x; }" });
        const source = if (imported)
            "comptime const dep = @import(\"./dep.ora\"); contract Imported { storage var cap: MaxValue<u256,100>; pub fn put(x:u256) { cap = dep.helper(x); } }"
        else
            "contract Private { storage var cap: MaxValue<u256,100>; fn helper(x: MaxValue<u256,100>) -> u256 { cap = x; return x; } pub fn put(x:MaxValue<u256,100>) -> u256 { return helper(x); } }";
        try tmp.dir.writeFile(testing.io, .{ .sub_path = "main.ora", .data = source });
        const path = try tmp.dir.realPathFileAlloc(testing.io, "main.ora", testing.allocator);
        defer testing.allocator.free(path);
        var compilation = try helpers.compilePackage(path);
        defer compilation.deinit();
        const hir = try compilation.db.lowerToHir(compilation.root_module_id);
        try testing.expect(hir.diagnostics.isEmpty());
        var pass = try z3.VerificationPass.init(testing.allocator);
        defer pass.deinit();
        var snapshot = try collector.collect(testing.allocator, hir.module.raw_module, &pass);
        defer snapshot.deinit();
        var projected = try collector.project(testing.allocator, hir.module.raw_module, &pass, snapshot.inventory);
        defer projected.deinit();
        const helper = for (snapshot.inventory.functions) |function| {
            if (std.mem.eql(u8, function.symbol, "helper") or std.mem.endsWith(u8, function.module_path, "dep.ora")) break function;
        } else return error.TestUnexpectedResult;
        try testing.expect(helper.runtime);
        const bound = for (projected.bindings) |binding| {
            if (binding.function_ordinal == helper.ordinal) break true;
        } else false;
        try testing.expect(bound);
        try testing.expect(projected.canonical.set.artifactDecision() == .blocked);
    }
}

test "SL3A unchanged source census pins static addresses and remaining caller refusals" {
    const sources = [_][]const u8{
        "ora-example/corpus/types/refinement/in_range_percentage.ora",
        "ora-example/corpus/types/refinement/in_range_u8.ora",
        "ora-example/corpus/types/refinement/max_value_in_function.ora",
        "ora-example/corpus/types/refinement/max_value_u256.ora",
        "ora-example/corpus/types/refinement/max_value_u8.ora",
        "ora-example/refinements/dispatcher_refinement_e2e.ora",
        "ora-example/refinements/patterns/supply_cap.ora",
    };
    var total: usize = 0;
    var projected_count: usize = 0;
    var boolean_count: usize = 0;
    var whole_count: usize = 0;
    var pass = try z3.VerificationPass.init(testing.allocator);
    defer pass.deinit();
    for (sources) |source| {
        var compilation = try helpers.compilePackage(source);
        defer compilation.deinit();
        const hir = try compilation.db.lowerToHir(compilation.root_module_id);
        try testing.expect(hir.diagnostics.isEmpty());
        var snapshot = try collector.collect(testing.allocator, hir.module.raw_module, &pass);
        defer snapshot.deinit();
        for (snapshot.inventory.functions, snapshot.function_ops) |function, op| {
            if (!function.runtime and !function.constructor) continue;
            total += 1;
            var rows = canonical_collector.collect(testing.allocator, hir.module.raw_module, .{ .project_calls = true, .lifecycle_roots = &.{op} }) catch |err| {
                std.debug.print("SL2A_CENSUS {s} {s} REFUSED {s}\n", .{ source, function.symbol, @errorName(err) });
                continue;
            };
            defer rows.deinit();
            if (rows.set.hasBlockingDiagnostic()) {
                for (rows.set.diagnostics) |diagnostic| if (diagnostic.blocks_artifacts)
                    std.debug.print("SL2A_CENSUS {s} {s} REFUSED {s}\n", .{ source, function.symbol, diagnostic.message });
            } else {
                try testing.expect(rows.set.function_summaries.len != 0);
                var output = std.Io.Writer.Allocating.init(testing.allocator);
                defer output.deinit();
                canonical_lean.writeCallCustodyModule(&output.writer, rows.set, "Ora.Generated.SL2ACensus") catch |err| {
                    std.debug.print("SL2A_CENSUS {s} {s} EMISSION_REFUSED {s}\n", .{ source, function.symbol, @errorName(err) });
                    continue;
                };
                projected_count += 1;
                const row = rows.set.function_summaries[0];
                if (row.outcomes[0].types.len == 1 and row.outcomes[0].types[0] == .spelling and std.mem.eql(u8, row.outcomes[0].types[0].spelling, "bool")) {
                    boolean_count += 1;
                    try lifecycleDataKernelCheck(output.written());
                }
                std.debug.print("SL2B_CENSUS {s} {s} PROJECTED\n", .{ source, function.symbol });
            }
            try testing.expect(rows.set.artifactDecision() == .blocked);
        }
        if (collector.project(testing.allocator, hir.module.raw_module, &pass, snapshot.inventory)) |value| {
            whole_count += 1;
            var projection = value;
            defer projection.deinit();
            try testing.expect(projection.canonical.set.artifactDecision() == .blocked);
            std.debug.print("SL2A_WHOLE {s} PROJECTED_NO_AUTHORITY\n", .{source});
        } else |err| std.debug.print("SL2A_WHOLE {s} REFUSED {s}\n", .{ source, @errorName(err) });
    }
    try testing.expectEqual(@as(usize, 39), total);
    try testing.expectEqual(@as(usize, 37), projected_count);
    try testing.expectEqual(@as(usize, 6), boolean_count);
    try testing.expectEqual(@as(usize, 5), whole_count);
}

test "SL2B Boolean payload relation binds the emitted digest and rejects the wrong result" {
    var compilation = try helpers.compileText("contract BooleanResult { pub fn below(x: u8) -> bool { return x < 10; } }");
    defer compilation.deinit();
    const hir = try compilation.db.lowerToHir(compilation.root_module_id);
    try testing.expect(hir.diagnostics.isEmpty());
    var pass = try z3.VerificationPass.init(testing.allocator);
    defer pass.deinit();
    var snapshot = try collector.collect(testing.allocator, hir.module.raw_module, &pass);
    defer snapshot.deinit();
    var projection = try collector.project(testing.allocator, hir.module.raw_module, &pass, snapshot.inventory);
    defer projection.deinit();
    const set = projection.canonical.set;
    try testing.expectEqual(@as(usize, 1), set.function_summaries.len);
    const row = set.function_summaries[0];
    const digest = try canonical_lean.functionVcSha256(set, row.id);
    try testing.expectEqualSlices(u8, &digest, &projection.bindings[0].function_vc_sha256);
    var output = std.Io.Writer.Allocating.init(testing.allocator);
    defer output.deinit();
    try emitter.writeProjection(&output.writer, testing.allocator, projection.snapshot.inventory, set, projection.bindings);
    try testing.expect(std.mem.indexOf(u8, output.written(), &std.fmt.bytesToHex(digest, .lower)) != null);
    try lifecycleDataKernelCheck(output.written());
    var context = try z3.Z3Context.init(testing.allocator);
    defer context.deinit();
    var adapter = canonical_z3.Adapter.init(&context, testing.allocator, set);
    const checked = try adapter.checkFunctionSafety(row.id, digest, 100001);
    try testing.expectEqual(@as(@TypeOf(checked.query.result.?.status), .proved), checked.query.result.?.status);
    try testing.expect(!checked.query.result.?.vacuous and !checked.query.result.?.vacuity_unknown);
    try testing.expect(checked.query.smtlib_hash != null and checked.query.smtlib_hash.? != 0);
    const vc = try adapter.encodeFunctionVC(row.id, digest);
    try testing.expectEqualSlices(u8, &digest, &vc.function_vc_sha256);
    const c = z3.z3_c;
    const payload = try adapter.functionOutput(row.id, "arm0.value0", c.Z3_mk_bool_sort(context.ctx));
    const input_id = row.parameters[0].id.?;
    const input_name = try std.fmt.allocPrintSentinel(testing.allocator, "$ora.loop.f{d}.p{d}", .{ input_id.file_id, input_id.pattern_id }, 0);
    defer testing.allocator.free(input_name);
    const parameter = c.Z3_mk_const(context.ctx, c.Z3_mk_string_symbol(context.ctx, input_name), c.Z3_mk_bv_sort(context.ctx, 8));
    for ([_]u8{ 0, 9, 10, 255 }) |input| {
        for ([_]bool{ false, true }) |answer| {
            var solver = try z3.Z3Solver.initForLogic(&context, testing.allocator, "QF_AUFBV");
            defer solver.deinit();
            try solver.assertChecked(vc.entry);
            try solver.assertChecked(vc.joint);
            try solver.assertChecked(c.Z3_mk_eq(context.ctx, parameter, c.Z3_mk_unsigned_int64(context.ctx, input, c.Z3_mk_bv_sort(context.ctx, 8))));
            try solver.assertChecked(if (answer) payload else c.Z3_mk_not(context.ctx, payload));
            try testing.expectEqual(if (answer == (input < 10)) c.Z3_L_TRUE else c.Z3_L_FALSE, try solver.checkChecked());
        }
    }
    var changed_digest = digest;
    changed_digest[0] ^= 1;
    try testing.expectError(error.FunctionVCDigestMismatch, adapter.encodeFunctionVC(row.id, changed_digest));
    try testing.expect(set.artifactDecision() == .blocked);
}

test "SL3A address results and map aliasing bind real joint queries to emitted data" {
    const sources = [_][]const u8{
        "contract AddressResult { pub fn identity(x: address) -> address { return x; } }",
        "contract AddressStore { storage var owner: address; pub fn put(x: address) -> address { owner = x; return owner; } }",
        "contract AddressMap { storage var balances: map<address,u256>; pub fn put(key: address, other: address) -> u256 { balances[other] = 9; balances[key] = 7; return balances[other]; } }",
    };
    var pass = try z3.VerificationPass.init(testing.allocator);
    defer pass.deinit();
    for (sources, 0..) |source, shape| {
        var compilation = try helpers.compileText(source);
        defer compilation.deinit();
        const hir = try compilation.db.lowerToHir(compilation.root_module_id);
        try testing.expect(hir.diagnostics.isEmpty());
        var snapshot = try collector.collect(testing.allocator, hir.module.raw_module, &pass);
        defer snapshot.deinit();
        var projection = try collector.project(testing.allocator, hir.module.raw_module, &pass, snapshot.inventory);
        defer projection.deinit();
        const set = projection.canonical.set;
        try testing.expectEqual(@as(usize, 1), set.function_summaries.len);
        const row = set.function_summaries[0];
        const digest = try canonical_lean.functionVcSha256(set, row.id);
        try testing.expectEqualSlices(u8, &digest, &projection.bindings[0].function_vc_sha256);
        var output = std.Io.Writer.Allocating.init(testing.allocator);
        defer output.deinit();
        try emitter.writeProjection(&output.writer, testing.allocator, projection.snapshot.inventory, set, projection.bindings);
        try testing.expect(std.mem.indexOf(u8, output.written(), &std.fmt.bytesToHex(digest, .lower)) != null);
        try lifecycleDataKernelCheck(output.written());
        var context = try z3.Z3Context.init(testing.allocator);
        defer context.deinit();
        var adapter = canonical_z3.Adapter.init(&context, testing.allocator, set);
        const checked = try adapter.checkFunctionSafety(row.id, digest, 100002);
        try testing.expectEqual(.proved, checked.query.result.?.status);
        try testing.expect(!checked.query.result.?.vacuous and !checked.query.result.?.vacuity_unknown);
        try testing.expect(checked.query.smtlib_hash != null and checked.query.smtlib_hash.? != 0);
        const vc = try adapter.encodeFunctionVC(row.id, digest);
        try testing.expectEqualSlices(u8, &digest, &vc.function_vc_sha256);
        const c = z3.z3_c;
        const sort = c.Z3_mk_bv_sort(context.ctx, if (shape == 2) 256 else 160);
        const payload = try adapter.functionOutput(row.id, "arm0.value0", sort);
        for ([_]bool{ false, true }) |aliases| {
            const expected: u64 = if (shape == 2) (if (aliases) 7 else 9) else 1;
            for ([_]bool{ false, true }) |correct| {
                var solver = try z3.Z3Solver.initForLogic(&context, testing.allocator, "QF_AUFBV");
                defer solver.deinit();
                try solver.assertChecked(vc.entry);
                try solver.assertChecked(vc.joint);
                for (row.parameters, 0..) |parameter, index| {
                    try testing.expectEqual(@as(u32, 14), parameter.ty.compiler_type_id);
                    const id = parameter.id.?;
                    const name = try std.fmt.allocPrintSentinel(testing.allocator, "$ora.loop.f{d}.p{d}", .{ id.file_id, id.pattern_id }, 0);
                    defer testing.allocator.free(name);
                    const address_sort = c.Z3_mk_bv_sort(context.ctx, 160);
                    const input = c.Z3_mk_const(context.ctx, c.Z3_mk_string_symbol(context.ctx, name), address_sort);
                    try solver.assertChecked(c.Z3_mk_eq(context.ctx, input, c.Z3_mk_unsigned_int64(context.ctx, if (index == 0 or aliases) 1 else 2, address_sort)));
                }
                try solver.assertChecked(c.Z3_mk_eq(context.ctx, payload, c.Z3_mk_unsigned_int64(context.ctx, if (correct) expected else expected + 1, sort)));
                try testing.expectEqual(if (correct) c.Z3_L_TRUE else c.Z3_L_FALSE, try solver.checkChecked());
            }
        }
        var bad_digest = digest;
        bad_digest[0] ^= 1;
        try testing.expectError(error.FunctionVCDigestMismatch, adapter.encodeFunctionVC(row.id, bad_digest));
        try testing.expect(set.artifactDecision() == .blocked);
    }
}

test "SL3A unchanged static address roots retain guards and real canonical queries" {
    const cases = [_]struct { source: []const u8, symbol: []const u8 }{
        .{ .source = "ora-example/refinements/dispatcher_refinement_e2e.ora", .symbol = "init" },
        .{ .source = "ora-example/refinements/dispatcher_refinement_e2e.ora", .symbol = "setOwner" },
        .{ .source = "ora-example/refinements/patterns/supply_cap.ora", .symbol = "mint" },
    };
    var pass = try z3.VerificationPass.init(testing.allocator);
    defer pass.deinit();
    for (cases) |case| {
        var compilation = try helpers.compilePackage(case.source);
        defer compilation.deinit();
        const hir = try compilation.db.lowerToHir(compilation.root_module_id);
        try testing.expect(hir.diagnostics.isEmpty());
        var snapshot = try collector.collect(testing.allocator, hir.module.raw_module, &pass);
        defer snapshot.deinit();
        const root = for (snapshot.inventory.functions, snapshot.function_ops) |function, op| {
            if (std.mem.eql(u8, function.symbol, case.symbol)) break op;
        } else return error.TestUnexpectedResult;
        var projection = try canonical_collector.collect(testing.allocator, hir.module.raw_module, .{ .project_calls = true, .lifecycle_roots = &.{root} });
        defer projection.deinit();
        const set = projection.set;
        try testing.expect(!set.hasBlockingDiagnostic());
        const row = set.function_summaries[0];
        const digest = try canonical_lean.functionVcSha256(set, row.id);
        var output = std.Io.Writer.Allocating.init(testing.allocator);
        defer output.deinit();
        try canonical_lean.writeCallCustodyModule(&output.writer, set, "Ora.Generated.SL3AStatic");
        try testing.expect(std.mem.indexOf(u8, output.written(), &std.fmt.bytesToHex(digest, .lower)) != null);
        try lifecycleDataKernelCheck(output.written());
        var checkpoints: usize = 0;
        for (row.paths) |path| for (path.timeline) |event| {
            if (event.payload == .safety_checkpoint) checkpoints += 1;
        };
        try testing.expect(checkpoints != 0);
        var context = try z3.Z3Context.init(testing.allocator);
        defer context.deinit();
        var adapter = canonical_z3.Adapter.init(&context, testing.allocator, set);
        const checked = try adapter.checkFunctionSafety(row.id, digest, 100003);
        try testing.expect(checked.query.result.?.status == .proved or checked.query.result.?.status == .failed);
        try testing.expect(!checked.query.result.?.vacuous and !checked.query.result.?.vacuity_unknown);
        try testing.expect(checked.query.smtlib_hash != null and checked.query.smtlib_hash.? != 0);
        try testing.expectEqualSlices(u8, &digest, &checked.function_vc_sha256);
        try testing.expect(set.artifactDecision() == .blocked);
        std.debug.print("SL3A_QUERY {s} checkpoints={d} status={s} digest={s}\n", .{ case.symbol, checkpoints, @tagName(checked.query.result.?.status), std.fmt.bytesToHex(digest, .lower) });
    }
}

test "SL3A address conversion guards remain real failing obligations without a precondition" {
    for ([_][]const u8{ "address", "NonZeroAddress" }) |parameter_type| {
        const source = try std.fmt.allocPrint(testing.allocator, "contract Guarded {{ storage var owner: NonZeroAddress; pub fn put(x: {s}) {{ owner = x; }} }}", .{parameter_type});
        defer testing.allocator.free(source);
        var compilation = try helpers.compileText(source);
        defer compilation.deinit();
        const hir = try compilation.db.lowerToHir(compilation.root_module_id);
        try testing.expect(hir.diagnostics.isEmpty());
        const root = mlir.oraModuleGetOperation(hir.module.raw_module);
        try testing.expect(firstOperation(root, "ora.refinement_guard") != null);
        const function = firstOperation(root, "func.func") orelse return error.TestUnexpectedResult;
        var projection = try canonical_collector.collect(testing.allocator, hir.module.raw_module, .{ .project_calls = true, .lifecycle_roots = &.{function} });
        defer projection.deinit();
        try testing.expect(!projection.set.hasBlockingDiagnostic());
        const row = projection.set.function_summaries[0];
        const digest = try canonical_lean.functionVcSha256(projection.set, row.id);
        var context = try z3.Z3Context.init(testing.allocator);
        defer context.deinit();
        var adapter = canonical_z3.Adapter.init(&context, testing.allocator, projection.set);
        const query = (try adapter.checkFunctionSafety(row.id, digest, 100004)).query;
        // Recovering a base carrier never assumes that a refinement guard has succeeded.
        try testing.expectEqual(.failed, query.result.?.status);
        try testing.expect(!query.result.?.vacuous and !query.result.?.vacuity_unknown);
        try testing.expect(query.smtlib_hash != null and query.smtlib_hash.? != 0);
        try testing.expect(projection.set.artifactDecision() == .blocked);
        const global = firstOperation(root, "ora.global") orelse return error.TestUnexpectedResult;
        const mlir_context = mlir.mlirOperationGetContext(global);
        mlir.oraOperationSetAttributeByName(global, mlir.oraStringRefCreate("ora.compiler_type_id", 20), mlir.mlirIntegerAttrGet(mlir.mlirIntegerTypeGet(mlir_context, 256), 14));
        // Integer-only metadata cannot impersonate the nominal carrier's identity.
        try testing.expectError(error.LifecycleStorageTypeMissing, canonical_collector.collect(testing.allocator, hir.module.raw_module, .{ .project_calls = true, .lifecycle_roots = &.{function} }));
    }
}

test "SL3A address mutations reject foreign inventory missing bindings and wrong storage carriers" {
    const Mutation = enum { owner, omitted_writer, digest, binding, key, value };
    for ([_]Mutation{ .owner, .omitted_writer, .digest, .binding, .key, .value }) |mutation| {
        errdefer std.debug.print("SL3A mutation {s}\n", .{@tagName(mutation)});
        var compilation = try helpers.compileText("contract Keys { storage var owner: address; storage var balances: map<address,u256>; pub fn put(key: address, wrong: u256) { owner = key; balances[key] = 1; } }");
        defer compilation.deinit();
        const hir = try compilation.db.lowerToHir(compilation.root_module_id);
        try testing.expect(hir.diagnostics.isEmpty());
        var pass = try z3.VerificationPass.init(testing.allocator);
        defer pass.deinit();
        var snapshot = try collector.collect(testing.allocator, hir.module.raw_module, &pass);
        defer snapshot.deinit();
        const root = mlir.oraModuleGetOperation(hir.module.raw_module);
        const function = firstOperation(root, "func.func") orelse return error.TestUnexpectedResult;
        const writer = for (snapshot.inventory.functions, 0..) |row, i| {
            if (std.mem.eql(u8, row.symbol, "put")) break i;
        } else return error.TestUnexpectedResult;
        if (mutation == .owner or mutation == .omitted_writer or mutation == .digest) {
            var changed = snapshot.inventory;
            const rows = try testing.allocator.dupe(data.Function, changed.functions);
            defer testing.allocator.free(rows);
            changed.functions = rows;
            switch (mutation) {
                .owner => rows[writer].owner = "foreign/owner",
                .omitted_writer => rows[writer].events = &.{},
                .digest => rows[writer].hir_sha256[0] ^= 1,
                else => unreachable,
            }
            try testing.expectError(error.LifecycleInventoryMismatch, collector.project(testing.allocator, hir.module.raw_module, &pass, changed));
            continue;
        }
        if (mutation == .binding) {
            const context = mlir.mlirOperationGetContext(function);
            mlir.oraOperationSetAttributeByName(function, mlir.oraStringRefCreate("ora.param_binding_ids", 21), mlir.oraArrayAttrCreate(context, 0, null));
            try testing.expectError(error.InvalidCallSignature, canonical_collector.collect(testing.allocator, hir.module.raw_module, .{ .project_calls = true, .lifecycle_roots = &.{function} }));
            continue;
        }
        const wrong = mlir.oraBlockGetArgument(mlir.oraRegionGetFirstBlock(mlir.oraOperationGetRegion(function, 0)), 1);
        const store = firstOperation(function, if (mutation == .key) "ora.map_store" else "ora.sstore") orelse return error.TestUnexpectedResult;
        var operands: [3]mlir.MlirValue = undefined;
        const count = mlir.oraOperationGetNumOperands(store);
        for (0..count) |i| operands[i] = mlir.oraOperationGetOperand(store, i);
        operands[if (mutation == .key) @as(usize, 1) else 0] = wrong;
        mlir.mlirOperationSetOperands(store, @intCast(count), &operands);
        try testing.expectError(error.LifecycleInventoryMismatch, collector.checkFresh(testing.allocator, hir.module.raw_module, &pass, snapshot.inventory));
        var projection = try canonical_collector.collect(testing.allocator, hir.module.raw_module, .{ .project_calls = true, .lifecycle_roots = &.{function} });
        defer projection.deinit();
        try testing.expect(projection.set.hasBlockingDiagnostic());
        try testing.expect(projection.set.artifactDecision() == .blocked);
    }
}
