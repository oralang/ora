const std = @import("std");
const testing = std.testing;
const compiler_test_preflight = @import("compiler.test.preflight.zig");
const mlir = @import("mlir_c_api").c;
const z3_verification = @import("ora_z3_verification");
const ora_root = @import("ora_root");
const compiler = ora_root.compiler;

const obligation = @import("formal/obligation.zig");
const obligation_crosscheck = @import("formal/obligation_crosscheck.zig");
const obligation_dump = @import("formal/obligation_dump.zig");
const obligation_from_mlir = @import("formal/obligation_from_mlir.zig");
const obligation_from_z3 = @import("formal/obligation_from_z3.zig");
const obligation_to_lean = @import("formal/obligation_to_lean.zig");
const obligation_to_z3 = @import("formal/obligation_to_z3.zig");
const proof_check = @import("formal/proof_check.zig");
const artifact_catalog = @import("formal/shared/artifact_catalog.zig");
const source_accounting = @import("formal/shared/source_accounting.zig");
const source_accounting_gate = @import("formal/kernel/source_accounting_gate.zig");
const source_accounting_from_syntax = @import("formal/source_accounting_from_syntax.zig");
const source_accounting_from_sema = @import("formal/source_accounting_from_sema.zig");
const source_accounting_from_comptime = @import("formal/source_accounting_from_comptime.zig");
const source_accounting_from_mlir = @import("formal/source_accounting_from_mlir.zig");
const source_accounting_from_z3 = @import("formal/source_accounting_from_z3.zig");
const source_accounting_from_package = @import("formal/source_accounting_from_package.zig");
const source_accounting_from_runtime = @import("formal/source_accounting_from_runtime.zig");
const source_accounting_pipeline = @import("formal/source_accounting_pipeline.zig");
const type_builtin = @import("ora_types").builtin;

const test_helpers = @import("compiler.test.helpers.zig");
const compilePackage = test_helpers.compilePackage;
const compileText = test_helpers.compileText;
const renderOraMlirForSource = test_helpers.renderOraMlirForSource;

// Formal CLI tests execute the exact artifact produced by this build graph;
// they never trust an independently installed binary by pathname.
const ORA_BINARY_REL = compiler_test_preflight.ora_binary_path;

test "v1 closure empty automatic target lists retain the actual artifact decision" {
    const allocator = testing.allocator;
    var tmp = testing.tmpDir(.{});
    defer tmp.cleanup();
    for ([_]bool{ false, true }) |valid| {
        const source = try std.fmt.allocPrint(
            allocator,
            "contract DefaultProofGateProbe {{ pub fn echo(input: u256) -> u256 ensures result == {s} {{ return input; }} }}",
            .{if (valid) "input" else "0"},
        );
        defer allocator.free(source);
        try tmp.dir.writeFile(std.testing.io, .{ .sub_path = "zero_targets.ora", .data = source });
        const path = try pathFromTmpAlloc(allocator, tmp, "zero_targets.ora");
        defer allocator.free(path);
        for ([_]bool{ false, true }) |explicit| {
            const relative = try std.fmt.allocPrint(allocator, "zero-{s}-{s}", .{ if (valid) "valid" else "false", if (explicit) "explicit" else "default" });
            defer allocator.free(relative);
            const output = try pathFromTmpAlloc(allocator, tmp, relative);
            defer allocator.free(output);
            var args: std.ArrayList([]const u8) = .empty;
            defer args.deinit(allocator);
            try args.appendSlice(allocator, &.{ ORA_BINARY_REL, "emit", "--emit=smt-report,bytecode", "--out-dir", output });
            if (explicit) try args.append(allocator, "--lean-proofs");
            try args.append(allocator, path);
            const run = try runProcessWithLeanPath(allocator, args.items);
            defer allocator.free(run.stdout);
            defer allocator.free(run.stderr);
            errdefer std.debug.print("zero targets valid={} explicit={}\n{s}\n{s}", .{ valid, explicit, run.stdout, run.stderr });
            try expectExited(run, if (valid) 0 else 1);
            const hex = try std.fmt.allocPrint(allocator, "{s}/zero_targets.hex", .{relative});
            defer allocator.free(hex);
            if (valid) {
                try tmp.dir.access(std.testing.io, hex, .{});
            } else {
                try testing.expectError(error.FileNotFound, tmp.dir.access(std.testing.io, hex, .{}));
                try testing.expect(std.mem.indexOf(u8, run.stdout, if (explicit) "failed_query" else "failed to prove ensures in echo") != null);
            }
        }
    }
}

test "v1 closure default loop free verification retains source assertions and invariants" {
    const allocator = testing.allocator;
    var tmp = testing.tmpDir(.{});
    defer tmp.cleanup();
    for ([_]bool{ true, false }) |invariant| {
        for ([_]bool{ true, false }) |valid| {
            const source = try std.fmt.allocPrint(
                allocator,
                "contract LoopFree {{ storage var x: u256 = 0; {s} pub fn set(v: u256) requires({s}) {{ assert(v <= 100, \"bound\"); x = v; }} }}",
                .{ if (invariant) "invariant bounded(x <= 100);" else "", if (valid) "v <= 100" else "true" },
            );
            defer allocator.free(source);
            try tmp.dir.writeFile(std.testing.io, .{ .sub_path = "loop_free.ora", .data = source });
            const path = try pathFromTmpAlloc(allocator, tmp, "loop_free.ora");
            defer allocator.free(path);
            const relative = try std.fmt.allocPrint(allocator, "{s}-{s}", .{ if (invariant) "invariant" else "assert", if (valid) "valid" else "invalid" });
            defer allocator.free(relative);
            const output = try pathFromTmpAlloc(allocator, tmp, relative);
            defer allocator.free(output);
            const run = try runProcessWithLeanPath(allocator, &.{ ORA_BINARY_REL, "emit", "--emit=smt-report,bytecode", "--out-dir", output, path });
            defer allocator.free(run.stdout);
            defer allocator.free(run.stderr);
            errdefer std.debug.print("loop-free valid={}\n{s}\n{s}", .{ valid, run.stdout, run.stderr });
            try expectExited(run, if (valid and !invariant) 0 else 1);
            const hex = try std.fmt.allocPrint(allocator, "{s}/loop_free.hex", .{relative});
            defer allocator.free(hex);
            if (valid and !invariant) {
                try tmp.dir.access(std.testing.io, hex, .{});
            } else {
                try testing.expectError(error.FileNotFound, tmp.dir.access(std.testing.io, hex, .{}));
                try testing.expect(std.mem.indexOf(u8, run.stdout, if (invariant) "UnbootstrappedContractInvariant" else "failed to prove contract invariant") != null);
                try testing.expect(std.mem.indexOf(u8, run.stdout, "Z3: rejected") != null);
            }
        }
    }
}

test "S1 Tier1 CLI global refusal cannot be bypassed by proof mode or empty targets" {
    const allocator = testing.allocator;
    var tmp = testing.tmpDir(.{});
    defer tmp.cleanup();
    try tmp.dir.writeFile(std.testing.io, .{ .sub_path = "empty.json", .data = "{\"schema_version\":1,\"proofs\":[]}" });
    const manifest = try pathFromTmpAlloc(allocator, tmp, "empty.json");
    defer allocator.free(manifest);
    const supplied = try std.fmt.allocPrint(allocator, "--lean-proofs={s}", .{manifest});
    defer allocator.free(supplied);
    const sources = [_][]const u8{
        "contract S1 { storage var value:u256; invariant positive(value>=1); pub fn get()->u256 ensures(result>=1) { return value; } }",
        "contract S1 { storage var value:u256=1; invariant positive(value>=1); pub fn get()->u256 ensures(result>=1) { return value; } }",
        "contract S1 { invariant true; pub fn get()->u256 { return 1; } }",
        "contract Unused { invariant false; } contract Live { pub fn get()->u256 { return 1; } }",
        "contract S1 { invariant true; }",
    };
    for (sources, 0..) |source, index| {
        try tmp.dir.writeFile(std.testing.io, .{ .sub_path = "s1.ora", .data = source });
        const path = try pathFromTmpAlloc(allocator, tmp, "s1.ora");
        defer allocator.free(path);
        for ([_][]const u8{ "", "--verify=basic", "--verify=full", "--lean-proofs", supplied }, 0..) |mode, mode_index| {
            const relative = try std.fmt.allocPrint(allocator, "refused-{d}-{d}", .{ index, mode_index });
            defer allocator.free(relative);
            const out = try pathFromTmpAlloc(allocator, tmp, relative);
            defer allocator.free(out);
            var argv: std.ArrayList([]const u8) = .empty;
            defer argv.deinit(allocator);
            try argv.appendSlice(allocator, &.{ ORA_BINARY_REL, "emit", "--emit=smt-report,bytecode", "--out-dir", out });
            if (mode.len != 0) try argv.append(allocator, mode);
            try argv.append(allocator, path);
            const run = try runProcessWithLeanPath(allocator, argv.items);
            defer allocator.free(run.stdout);
            defer allocator.free(run.stderr);
            errdefer std.debug.print("S1 source={d} mode={s}\n{s}\n{s}", .{ index, mode, run.stdout, run.stderr });
            try expectExited(run, 1);
            try testing.expect(std.mem.indexOf(u8, run.stdout, "UnbootstrappedContractInvariant") != null);
            const hex = try std.fmt.allocPrint(allocator, "{s}/s1.hex", .{relative});
            defer allocator.free(hex);
            try testing.expectError(error.FileNotFound, tmp.dir.access(std.testing.io, hex, .{}));
            const authority_path = try std.fmt.allocPrint(allocator, "{s}/s1.formal.artifacts.json", .{relative});
            defer allocator.free(authority_path);
            const authority_text = try tmp.dir.readFileAlloc(std.testing.io, authority_path, allocator, .limited(1024 * 1024));
            defer allocator.free(authority_text);
            const authority = try std.json.parseFromSlice(std.json.Value, allocator, authority_text, .{});
            defer authority.deinit();
            try testing.expectEqualStrings("blocked", authority.value.object.get("final_artifact_authorization").?.string);
        }
    }
}

test "v1 closure false supported loops block default emission beside call exclusions" {
    const allocator = testing.allocator;
    var tmp = testing.tmpDir(.{});
    defer tmp.cleanup();
    for ([_]bool{ false, true }) |mixed| {
        const source = try std.fmt.allocPrint(
            allocator,
            "contract MixedLoopFailure {{ fn leaf(input: u256) -> u256 {{ return input; }} pub fn run(input: u256) -> u256 {{ {s} var j: u256 = 1; while (j < 2) invariant j == 0 {{ j = j +% 1; }} return j; }} }}",
            .{if (mixed) "var i: u256 = 0; while (i < 1) invariant i <= 1 { let value: u256 = leaf(input); i = i +% 1; }" else ""},
        );
        defer allocator.free(source);
        try tmp.dir.writeFile(std.testing.io, .{ .sub_path = "false_invariant.ora", .data = source });
        const path = try pathFromTmpAlloc(allocator, tmp, "false_invariant.ora");
        defer allocator.free(path);
        const relative = if (mixed) "mixed" else "supported";
        const output = try pathFromTmpAlloc(allocator, tmp, relative);
        defer allocator.free(output);
        const run = try runProcessWithLeanPath(allocator, &.{ ORA_BINARY_REL, "emit", "--emit=smt-report,bytecode", "--out-dir", output, path });
        defer allocator.free(run.stdout);
        defer allocator.free(run.stderr);
        errdefer std.debug.print("false invariant mixed={}\n{s}\n{s}", .{ mixed, run.stdout, run.stderr });
        try expectExited(run, 1);
        try testing.expect(std.mem.indexOf(u8, run.stdout, "failed to prove invariant in run") != null);
        try testing.expect(std.mem.indexOf(u8, run.stdout, "Z3: rejected") != null);
        try testing.expectEqual(mixed, std.mem.indexOf(u8, run.stdout, "unsupported_call_in_loop:") != null);
        const hex = try std.fmt.allocPrint(allocator, "{s}/false_invariant.hex", .{relative});
        defer allocator.free(hex);
        try testing.expectError(error.FileNotFound, tmp.dir.access(std.testing.io, hex, .{}));
    }
}

test "v1 closure detects calls in complete loop regions and blocks default emission" {
    const allocator = testing.allocator;
    const cases = [_]struct { body: []const u8, loops: usize }{
        .{ .body = "while (i < 2) invariant i <= 2 { if (i == input) { let value: u256 = leaf(input); } i = i +% 1; }", .loops = 1 },
        .{ .body = "while (i < 2) invariant i <= 2 { match (input) { 0 -> { if (i == 0) { let value: u256 = leaf(input); } } else -> {} } i = i +% 1; }", .loops = 1 },
        .{ .body = "while (i < 2) invariant i <= 2 { var j: u256 = 0; while (j < 2) invariant j <= 2 { let value: u256 = leaf(input); j = j +% 1; } i = i +% 1; }", .loops = 2 },
        .{ .body = "while (i < leaf(input)) invariant i <= input { i = i +% 1; }", .loops = 1 },
    };
    var tmp = testing.tmpDir(.{});
    defer tmp.cleanup();
    for (cases, 0..) |case, index| {
        const source = try std.fmt.allocPrint(
            allocator,
            "contract CallDetector {{ fn leaf(input: u256) -> u256 {{ return input; }} pub fn run(input: u256) -> u256 {{ var i: u256 = 0; {s} return i; }} }}",
            .{case.body},
        );
        defer allocator.free(source);
        const stem = try std.fmt.allocPrint(allocator, "detector_{d}", .{index});
        defer allocator.free(stem);
        const name = try std.fmt.allocPrint(allocator, "{s}.ora", .{stem});
        defer allocator.free(name);
        try tmp.dir.writeFile(std.testing.io, .{ .sub_path = name, .data = source });
        const path = try pathFromTmpAlloc(allocator, tmp, name);
        defer allocator.free(path);
        var projected = try collectPackageObligations(allocator, path);
        defer projected.deinit();
        try testing.expectEqual(case.loops, projected.set.loop_summaries.len);
        for (projected.set.loop_summaries) |summary| try testing.expect(loopSummaryHasReason(summary, .loop_has_call));
        const output = try pathFromTmpAlloc(allocator, tmp, stem);
        defer allocator.free(output);
        const run = try runProcessWithLeanPath(allocator, &.{ ORA_BINARY_REL, "emit", "--emit=smt-report,bytecode", "--out-dir", output, path });
        defer allocator.free(run.stdout);
        defer allocator.free(run.stderr);
        errdefer std.debug.print("detector {d}\n{s}\n{s}", .{ index, run.stdout, run.stderr });
        try expectExited(run, 1);
        try testing.expectEqual(case.loops, std.mem.count(u8, run.stdout, "unsupported_call_in_loop:"));
        const hex = try std.fmt.allocPrint(allocator, "{s}/{s}.hex", .{ stem, stem });
        defer allocator.free(hex);
        try testing.expectError(error.FileNotFound, tmp.dir.access(std.testing.io, hex, .{}));
        const report_name = try std.fmt.allocPrint(allocator, "{s}/{s}.formal.loops.json", .{ stem, stem });
        defer allocator.free(report_name);
        const report = try tmp.dir.readFileAlloc(std.testing.io, report_name, allocator, .limited(1024 * 1024));
        defer allocator.free(report);
        try testing.expectEqual(case.loops, std.mem.count(u8, report, "\"verification_reason\":\"unsupported_call_in_loop\""));
        try testing.expectEqual(case.loops, std.mem.count(u8, report, "\"verification_status\":\"not_verified\""));
    }
}

test "v1 closure MLIR call inventory distinguishes indirect external and pure builtins" {
    const h = createContext();
    defer destroyContext(h);
    const module = try parseModule(h.ctx,
        \\module {
        \\  func.func private @leaf()
        \\  func.func @direct() { func.call @leaf() : () -> ()
        \\    func.return }
        \\  func.func @indirect(%callee: () -> ()) { func.call_indirect %callee() : () -> ()
        \\    func.return }
        \\  func.func @external(%target: i256, %gas: i256, %data: i256) {
        \\    %ok, %ret = "ora.external_call"(%target, %gas, %data) {call_kind = "call", trait_name = "T", method_name = "m"} : (i256, i256, i256) -> (i1, i256)
        \\    func.return }
        \\  func.func @caller() { %v = "ora.evm.caller"() : () -> i256
        \\    func.return }
        \\  func.func @callvalue() { %v = "ora.evm.callvalue"() : () -> i256
        \\    func.return }
        \\}
    );
    defer mlir.oraModuleDestroy(module);
    var function = mlir.oraOperationGetNextInBlock(mlir.oraBlockGetFirstOperation(mlir.oraModuleGetBody(module)));
    for ([_]bool{ true, true, true, false, false }) |expected| {
        try testing.expectEqual(expected, mlir.oraOperationContainsCall(function));
        function = mlir.oraOperationGetNextInBlock(function);
    }
}

test "v1 closure default proves a loop and opt out suppresses loop verification claims" {
    const allocator = testing.allocator;
    var tmp = testing.tmpDir(.{});
    defer tmp.cleanup();
    const verified_output = try pathFromTmpAlloc(allocator, tmp, "verified");
    defer allocator.free(verified_output);
    const verified = try runProcessWithLeanPath(allocator, &.{ ORA_BINARY_REL, "emit", "--emit=bytecode", "--out-dir", verified_output, "tests/formal/scalar_loop_induction.ora" });
    defer allocator.free(verified.stdout);
    defer allocator.free(verified.stderr);
    try expectExited(verified, 0);
    try testing.expect(std.mem.indexOf(u8, verified.stdout, "kernel-checked certificates: 1") != null);
    const unchecked_output = try pathFromTmpAlloc(allocator, tmp, "unchecked");
    defer allocator.free(unchecked_output);
    const unchecked = try runProcessWithLeanPath(allocator, &.{ ORA_BINARY_REL, "emit", "--no-verify", "--emit=bytecode", "--out-dir", unchecked_output, "tests/formal/call_adapter_leaf.ora" });
    defer allocator.free(unchecked.stdout);
    defer allocator.free(unchecked.stderr);
    try expectExited(unchecked, 0);
    try tmp.dir.access(std.testing.io, "unchecked/call_adapter_leaf.hex", .{});
    try testing.expect(std.mem.indexOf(u8, unchecked.stdout, "Lean scalar-loop induction") == null);
    inline for (.{ ".formal.loops.json", ".lean.proof.json", ".lean.dispatcher.proof.json" }) |suffix| {
        try testing.expectError(error.FileNotFound, tmp.dir.access(std.testing.io, "unchecked/call_adapter_leaf" ++ suffix, .{}));
    }
}

const MlirContextHandle = struct {
    ctx: mlir.MlirContext,
};

fn createContext() MlirContextHandle {
    const ctx = mlir.oraContextCreate();
    const registry = mlir.oraDialectRegistryCreate();
    mlir.oraRegisterAllDialects(registry);
    mlir.oraContextAppendDialectRegistry(ctx, registry);
    mlir.oraDialectRegistryDestroy(registry);
    mlir.oraContextLoadAllAvailableDialects(ctx);
    mlir.oraContextLoadSIRDialect(ctx);
    _ = mlir.oraDialectRegister(ctx);
    return .{ .ctx = ctx };
}

fn destroyContext(handle: MlirContextHandle) void {
    mlir.oraContextDestroy(handle.ctx);
}

fn parseModule(ctx: mlir.MlirContext, text: []const u8) !mlir.MlirModule {
    const module = mlir.oraModuleCreateParse(ctx, mlir.oraStringRefCreate(text.ptr, text.len));
    if (mlir.oraModuleIsNull(module)) return error.MlirParseFailed;
    return module;
}

fn findFirstOpByName(module: mlir.MlirModule, name: []const u8) ?mlir.MlirOperation {
    if (mlir.oraModuleIsNull(module)) return null;
    return findFirstOpByNameInOp(mlir.oraModuleGetOperation(module), name);
}

fn findFirstOpByNameInOp(op: mlir.MlirOperation, expected: []const u8) ?mlir.MlirOperation {
    if (mlir.oraOperationIsNull(op)) return null;

    const name_ref = mlir.oraOperationGetName(op);
    if (name_ref.data != null and std.mem.eql(u8, name_ref.data[0..name_ref.length], expected)) return op;

    const num_regions = mlir.oraOperationGetNumRegions(op);
    var region_index: usize = 0;
    while (region_index < num_regions) : (region_index += 1) {
        const region = mlir.oraOperationGetRegion(op, region_index);
        if (mlir.oraRegionIsNull(region)) continue;

        var block = mlir.oraRegionGetFirstBlock(region);
        while (!mlir.oraBlockIsNull(block)) : (block = mlir.oraBlockGetNextInRegion(block)) {
            var child = mlir.oraBlockGetFirstOperation(block);
            while (!mlir.oraOperationIsNull(child)) : (child = mlir.oraOperationGetNextInBlock(child)) {
                if (findFirstOpByNameInOp(child, expected)) |found| return found;
            }
        }
    }

    return null;
}

fn findFunctionByName(module: mlir.MlirModule, function_name: []const u8) ?mlir.MlirOperation {
    if (mlir.oraModuleIsNull(module)) return null;
    return findFunctionByNameInOp(mlir.oraModuleGetOperation(module), function_name);
}

fn findFunctionByNameInOp(op: mlir.MlirOperation, function_name: []const u8) ?mlir.MlirOperation {
    if (mlir.oraOperationIsNull(op)) return null;

    const name_ref = mlir.oraOperationGetName(op);
    if (name_ref.data != null and
        std.mem.eql(u8, name_ref.data[0..name_ref.length], "func.func"))
    {
        const attr = mlir.oraOperationGetAttributeByName(op, strRef("sym_name"));
        if (!mlir.oraAttributeIsNull(attr)) {
            const value = mlir.oraStringAttrGetValue(attr);
            if (value.data != null and
                std.mem.eql(u8, value.data[0..value.length], function_name)) return op;
        }
    }

    const num_regions = mlir.oraOperationGetNumRegions(op);
    var region_index: usize = 0;
    while (region_index < num_regions) : (region_index += 1) {
        const region = mlir.oraOperationGetRegion(op, region_index);
        if (mlir.oraRegionIsNull(region)) continue;

        var block = mlir.oraRegionGetFirstBlock(region);
        while (!mlir.oraBlockIsNull(block)) : (block = mlir.oraBlockGetNextInRegion(block)) {
            var child = mlir.oraBlockGetFirstOperation(block);
            while (!mlir.oraOperationIsNull(child)) : (child = mlir.oraOperationGetNextInBlock(child)) {
                if (findFunctionByNameInOp(child, function_name)) |found| return found;
            }
        }
    }

    return null;
}

fn strRef(text: []const u8) mlir.MlirStringRef {
    return .{ .data = text.ptr, .length = text.len };
}

fn countLogical(set: obligation.ObligationSet, role: obligation.LogicalRole) usize {
    var count: usize = 0;
    for (set.obligations) |item| {
        if (item.kind == .logical and item.kind.logical.role == role) count += 1;
    }
    return count;
}

fn countRuntimeGuards(set: obligation.ObligationSet) usize {
    var count: usize = 0;
    for (set.obligations) |item| {
        if (item.kind == .runtime_guard) count += 1;
    }
    return count;
}

fn loopSummaryHasReason(
    row: obligation.LoopSummaryRow,
    expected: obligation.LoopUnsupportedReason,
) bool {
    for (row.unsupported_reasons) |reason| if (reason == expected) return true;
    return false;
}

fn loopEdgePathFormulaCount(edge: obligation.LoopEdgeRow) usize {
    var count: usize = 0;
    for (edge.timeline) |event| switch (event.payload) {
        .path_assumption => |path| count += path.formulas.len,
        else => {},
    };
    return count;
}

fn loopEdgeDependencyCount(edge: obligation.LoopEdgeRow) usize {
    var count: usize = 0;
    for (edge.timeline) |event| {
        if (event.payload == .summary_dependency) count += 1;
    }
    return count;
}

fn firstLoopEdgeDependency(edge: obligation.LoopEdgeRow) ?obligation.LoopSummaryDependency {
    for (edge.timeline) |event| switch (event.payload) {
        .summary_dependency => |dependency| return dependency,
        else => {},
    };
    return null;
}

fn loopEdgeSafetyCheckpointCount(edge: obligation.LoopEdgeRow) usize {
    var count: usize = 0;
    for (edge.timeline) |event| {
        if (event.payload == .safety_checkpoint) count += 1;
    }
    return count;
}

fn expectLoopEdgeSafetyStages(edge: obligation.LoopEdgeRow, expected: []const u32) !void {
    var dependency_count: u32 = 0;
    var checkpoint_index: usize = 0;
    for (edge.timeline) |event| switch (event.payload) {
        .summary_dependency => dependency_count += 1,
        .safety_checkpoint => {
            if (checkpoint_index >= expected.len) return error.TestUnexpectedResult;
            try testing.expectEqual(expected[checkpoint_index], dependency_count);
            checkpoint_index += 1;
        },
        else => {},
    };
    try testing.expectEqual(expected.len, checkpoint_index);
}

fn loopSummaryByFunction(
    set: obligation.ObligationSet,
    function_name: []const u8,
) ?obligation.LoopSummaryRow {
    for (set.loop_summaries) |row| {
        if (row.owner == .statement and
            std.mem.eql(u8, row.owner.statement.function_name, function_name)) return row;
    }
    return null;
}

fn expectOneLoopProjectionFailure(
    set: obligation.ObligationSet,
    status: obligation.LoopProjectionCompleteness,
    reason: obligation.LoopUnsupportedReason,
) !void {
    var matches: usize = 0;
    for (set.loop_summaries) |row| {
        if (row.projection_completeness != status) continue;
        try testing.expect(loopSummaryHasReason(row, reason));
        try testing.expect(!row.projectionSupported());
        matches += 1;
    }
    try testing.expectEqual(@as(usize, 1), matches);
}

fn expectLoopSummaryRejected(
    set: obligation.ObligationSet,
    query: obligation.VerificationQuery,
    row: obligation.LoopSummaryRow,
    expected: obligation.LoopUnsupportedReason,
) !void {
    var mutated_set = set;
    const rows = [_]obligation.LoopSummaryRow{row};
    mutated_set.loop_summaries = &rows;
    switch (obligation_to_lean.querySemanticSupport(mutated_set, query)) {
        .supported => return error.TestUnexpectedResult,
        .unsupported => |reason| try testing.expectEqual(
            obligation_to_lean.SemanticUnsupportedReason{ .unsupported_loop_summary = expected },
            reason,
        ),
    }
}

fn countAssumption(set: obligation.ObligationSet, kind: obligation.AssumptionKind) usize {
    var count: usize = 0;
    for (set.assumptions) |item| {
        if (item.kind == kind) count += 1;
    }
    return count;
}

fn countResource(set: obligation.ObligationSet, op: obligation.ResourceOperation, property: obligation.ResourceProperty) usize {
    var count: usize = 0;
    for (set.obligations) |item| {
        if (item.kind == .resource and item.kind.resource.op == op and item.kind.resource.property == property) count += 1;
    }
    return count;
}

fn findResourceQuery(
    set: obligation.ObligationSet,
    op: obligation.ResourceOperation,
    property: obligation.ResourceProperty,
) !obligation.VerificationQuery {
    var resource_id: ?obligation.Id = null;
    for (set.obligations) |item| {
        if (item.kind == .resource and item.kind.resource.op == op and item.kind.resource.property == property) {
            resource_id = item.id;
            break;
        }
    }
    const id = resource_id orelse return error.TestUnexpectedResult;
    for (set.queries) |query| {
        if (query.obligation_ids.len == 1 and query.obligation_ids[0] == id) return query;
    }
    return error.TestUnexpectedResult;
}

fn expectSingleResourceSemanticSupport(
    terms: []const obligation.Term,
    goal: obligation.ResourceGoal,
    expected_supported: bool,
) !void {
    const obligation_ids = [_]obligation.Id{1};
    const obligations = [_]obligation.Obligation{.{
        .id = 1,
        .owner = .{ .function = .{ .name = "resource_test" } },
        .source = .generated(),
        .phase = .ora_mlir,
        .origin = .{ .resource_op = .{ .op = goal.op, .domain = goal.domain, .ordinal = 0 } },
        .kind = .{ .resource = goal },
    }};
    const queries = [_]obligation.VerificationQuery{.{
        .id = 2,
        .owner = .{ .function = .{ .name = "resource_test" } },
        .source = .generated(),
        .phase = .ora_mlir,
        .origin = .source,
        .kind = .obligation,
        .obligation_ids = &obligation_ids,
    }};
    const set: obligation.ObligationSet = .{
        .obligations = &obligations,
        .queries = &queries,
        .terms = terms,
    };
    const supported = switch (obligation_to_lean.querySemanticSupport(set, queries[0])) {
        .supported => true,
        .unsupported => false,
    };
    try testing.expectEqual(expected_supported, supported);
}

fn countQuantifier(set: obligation.ObligationSet) usize {
    var count: usize = 0;
    for (set.obligations) |item| {
        if (item.kind == .quantifier) count += 1;
    }
    return count;
}

fn countEffectFrame(set: obligation.ObligationSet, relation: obligation.EffectFrameRelation) usize {
    var count: usize = 0;
    for (set.obligations) |item| {
        if (item.kind == .effect_frame and item.kind.effect_frame.relation == relation) count += 1;
    }
    return count;
}

fn countArithmeticSafety(set: obligation.ObligationSet, safety: obligation.ArithmeticSafetyKind) usize {
    var count: usize = 0;
    for (set.obligations) |item| {
        if (item.kind == .logical and
            item.kind.logical.role == .arithmetic_safety and
            item.kind.logical.arithmetic_safety != null and
            item.kind.logical.arithmetic_safety.? == safety)
        {
            count += 1;
        }
    }
    return count;
}

fn countQuery(set: obligation.ObligationSet, kind: obligation.VerificationQueryKind) usize {
    var count: usize = 0;
    for (set.queries) |item| {
        if (item.kind == kind) count += 1;
    }
    return count;
}

fn countQueryRole(set: obligation.ObligationSet, role: obligation.LogicalRole) usize {
    var count: usize = 0;
    for (set.queries) |item| {
        if (item.logical_role != null and item.logical_role.? == role) count += 1;
    }
    return count;
}

fn expectParameterKey(key: obligation.PlaceKey, expected: obligation.FreeVarId) !void {
    try testing.expect(key == .parameter);
    try testing.expect(obligation.freeVarIdEql(expected, key.parameter));
}

fn expectAnyParameterKey(key: obligation.PlaceKey) !obligation.FreeVarId {
    try testing.expect(key == .parameter);
    return key.parameter;
}

fn expectConstantKey(key: obligation.PlaceKey, expected: []const u8) !void {
    try testing.expect(key == .constant);
    try testing.expectEqualStrings(expected, key.constant);
}

fn expectPlaceRoot(place: obligation.PlaceRef, expected_root: []const u8, expected_region: obligation.RegionRef) !void {
    try testing.expectEqualStrings(expected_root, place.root);
    try testing.expectEqual(expected_region, place.region);
}

fn expectPlaceReadTerm(
    set: obligation.ObligationSet,
    term_id: obligation.TermId,
    expected_root: []const u8,
    expected_region: obligation.RegionRef,
) !void {
    try testing.expect(term_id < set.terms.len);
    try testing.expect(set.terms[term_id] == .place_read);
    try expectPlaceRoot(set.terms[term_id].place_read, expected_root, expected_region);
}

fn expectPlaceReadTermWithParameterKeys(
    set: obligation.ObligationSet,
    term_id: obligation.TermId,
    expected_root: []const u8,
    expected_keys: []const obligation.FreeVarId,
) !void {
    try expectPlaceReadTerm(set, term_id, expected_root, .storage);
    const place = set.terms[term_id].place_read;
    try testing.expectEqual(expected_keys.len, place.keys.len);
    for (expected_keys, 0..) |expected, index| {
        try expectParameterKey(place.keys[index], expected);
    }
}

fn expectPlaceReadTermWithParameterKeyCount(
    set: obligation.ObligationSet,
    term_id: obligation.TermId,
    expected_root: []const u8,
    expected_count: usize,
) !void {
    try expectPlaceReadTerm(set, term_id, expected_root, .storage);
    const place = set.terms[term_id].place_read;
    try testing.expectEqual(expected_count, place.keys.len);
    for (place.keys) |key| {
        _ = try expectAnyParameterKey(key);
    }
}

fn expectPlaceReadTermWithConstantKeys(
    set: obligation.ObligationSet,
    term_id: obligation.TermId,
    expected_root: []const u8,
    expected_keys: []const []const u8,
) !void {
    try expectPlaceReadTerm(set, term_id, expected_root, .storage);
    const place = set.terms[term_id].place_read;
    try testing.expectEqual(expected_keys.len, place.keys.len);
    for (expected_keys, 0..) |expected, index| {
        try expectConstantKey(place.keys[index], expected);
    }
}

fn expectOldPlaceReadTerm(
    set: obligation.ObligationSet,
    term_id: obligation.TermId,
    expected_root: []const u8,
    expected_region: obligation.RegionRef,
) !void {
    try testing.expect(term_id < set.terms.len);
    try testing.expect(set.terms[term_id] == .old);
    try expectPlaceReadTerm(set, set.terms[term_id].old, expected_root, expected_region);
}

fn countOldTerms(set: obligation.ObligationSet) usize {
    var count: usize = 0;
    for (set.terms) |term| {
        if (term == .old) count += 1;
    }
    return count;
}

fn expectLogicalTerm(set: obligation.ObligationSet, role: obligation.LogicalRole) !obligation.TermId {
    for (set.obligations) |item| {
        if (item.kind != .logical or item.kind.logical.role != role) continue;
        if (item.kind.logical.formula != .term) return error.TestUnexpectedResult;
        return item.kind.logical.formula.term;
    }
    return error.TestUnexpectedResult;
}

fn expectQuantifiedTerm(
    set: obligation.ObligationSet,
    term_id: obligation.TermId,
    expected_quantifier: obligation.Quantifier,
    expected_name: []const u8,
) !obligation.QuantifiedTerm {
    try testing.expect(term_id < set.terms.len);
    try testing.expect(set.terms[term_id] == .quantified);
    const quantified = set.terms[term_id].quantified;
    try testing.expectEqual(expected_quantifier, quantified.quantifier);
    try testing.expectEqualStrings(expected_name, quantified.binder.name);
    return quantified;
}

fn termHasNestedQuantifier(
    set: obligation.ObligationSet,
    term_id: obligation.TermId,
    depth: usize,
) bool {
    if (term_id >= set.terms.len) return false;
    return switch (set.terms[term_id]) {
        .quantified => |quantified| depth > 0 or
            termHasNestedQuantifier(set, quantified.body, depth + 1) or
            if (quantified.condition) |condition|
                termHasNestedQuantifier(set, condition, depth + 1)
            else
                false,
        .unary => |unary| termHasNestedQuantifier(set, unary.operand, depth),
        .binary => |binary| termHasNestedQuantifier(set, binary.lhs, depth) or
            termHasNestedQuantifier(set, binary.rhs, depth),
        .old => |operand| termHasNestedQuantifier(set, operand, depth),
        .refinement_predicate => |predicate| blk: {
            if (termHasNestedQuantifier(set, predicate.value, depth)) break :blk true;
            for (predicate.args) |arg| {
                if (termHasNestedQuantifier(set, arg, depth)) break :blk true;
            }
            break :blk false;
        },
        else => false,
    };
}

fn expectBinaryTerm(
    set: obligation.ObligationSet,
    term_id: obligation.TermId,
    expected_op: obligation.BinaryOp,
) !obligation.BinaryTerm {
    try testing.expect(term_id < set.terms.len);
    try testing.expect(set.terms[term_id] == .binary);
    const binary = set.terms[term_id].binary;
    try testing.expectEqual(expected_op, binary.op);
    return binary;
}

fn expectFreeVarTerm(
    set: obligation.ObligationSet,
    term_id: obligation.TermId,
    expected_file_id: u32,
    expected_pattern_id: u32,
    expected_name: []const u8,
) !void {
    _ = try expectFreeVarTermRef(set, term_id, expected_file_id, expected_pattern_id, expected_name);
}

fn expectFreeVarTermRef(
    set: obligation.ObligationSet,
    term_id: obligation.TermId,
    expected_file_id: u32,
    expected_pattern_id: u32,
    expected_name: []const u8,
) !obligation.FreeVarRef {
    try testing.expect(term_id < set.terms.len);
    try testing.expect(set.terms[term_id] == .variable);
    const variable = set.terms[term_id].variable;
    try testing.expect(variable == .free);
    try testing.expectEqual(expected_file_id, variable.free.id.file_id);
    try testing.expectEqual(expected_pattern_id, variable.free.id.pattern_id);
    try testing.expectEqualStrings(expected_name, variable.free.name);
    return variable.free;
}

fn expectTypeRefSpelling(ty: ?obligation.TypeRef, expected: []const u8) !void {
    const actual = ty orelse return error.TestUnexpectedResult;
    try testing.expect(actual == .spelling);
    try testing.expectEqualStrings(expected, actual.spelling);
}

fn expectTypeRefBuiltin(ty: ?obligation.TypeRef, expected: type_builtin.BuiltinTypeId) !void {
    const actual = ty orelse return error.TestUnexpectedResult;
    try testing.expect(actual == .compiler_type_id);
    try testing.expectEqual(type_builtin.lookupBuiltinById(expected).comptime_type_id, actual.compiler_type_id);
}

fn builtinTypeRef(expected: type_builtin.BuiltinTypeId) obligation.TypeRef {
    return .{ .compiler_type_id = type_builtin.lookupBuiltinById(expected).comptime_type_id };
}

fn checkFormalZ3TermObligation(
    terms: []const obligation.Term,
    formula: obligation.TermId,
) !obligation_to_z3.CheckStatus {
    var z3_ctx = try z3_verification.Z3Context.init(testing.allocator);
    defer z3_ctx.deinit();

    const obligations = [_]obligation.Obligation{
        .{
            .id = 1,
            .owner = .{ .function = .{ .name = "adapter_fixture" } },
            .source = .generated(),
            .phase = .sema,
            .origin = .{ .sema_fact = .{ .kind = "adapter_fixture", .ordinal = 0 } },
            .kind = .{ .logical = .{
                .role = .ensures,
                .formula = .{ .term = formula },
            } },
        },
    };
    const obligation_ids = [_]obligation.Id{1};
    const queries = [_]obligation.VerificationQuery{
        .{
            .id = 2,
            .owner = .{ .function = .{ .name = "adapter_fixture" } },
            .source = .generated(),
            .phase = .report,
            .origin = .source,
            .backend = .z3,
            .kind = .obligation,
            .obligation_ids = &obligation_ids,
        },
    };
    const set: obligation.ObligationSet = .{
        .obligations = &obligations,
        .queries = &queries,
        .terms = terms,
    };

    var adapter = obligation_to_z3.Adapter.init(&z3_ctx, testing.allocator, set);
    return adapter.checkObligation(1);
}

fn expectBoundVarTerm(
    set: obligation.ObligationSet,
    term_id: obligation.TermId,
    expected_index: u32,
    expected_name: []const u8,
) !void {
    try testing.expect(term_id < set.terms.len);
    try testing.expect(set.terms[term_id] == .variable);
    const variable = set.terms[term_id].variable;
    try testing.expect(variable == .bound);
    try testing.expectEqual(expected_index, variable.bound.index);
    try testing.expectEqualStrings(expected_name, variable.bound.name);
}

fn expectSummaryMatchesZ3PreparedQueries(
    formal_summary: obligation.VerificationQuerySummary,
    z3_summary: z3_verification.PreparedQuerySummary,
) !void {
    const formal_total =
        @as(u64, formal_summary.base) +
        @as(u64, formal_summary.obligation) +
        @as(u64, formal_summary.loop_invariant_step) +
        @as(u64, formal_summary.loop_body_safety) +
        @as(u64, formal_summary.loop_invariant_post) +
        @as(u64, formal_summary.guard_satisfy) +
        @as(u64, formal_summary.guard_violate);
    try testing.expectEqual(formal_total, z3_summary.total);
    try testing.expectEqual(@as(u64, formal_summary.base), z3_summary.base);
    try testing.expectEqual(@as(u64, formal_summary.obligation), z3_summary.obligation);
    try testing.expectEqual(@as(u64, formal_summary.loop_invariant_step), z3_summary.loop_invariant_step);
    try testing.expectEqual(@as(u64, formal_summary.loop_body_safety), z3_summary.loop_body_safety);
    try testing.expectEqual(@as(u64, formal_summary.loop_invariant_post), z3_summary.loop_invariant_post);
    try testing.expectEqual(@as(u64, formal_summary.guard_satisfy), z3_summary.guard_satisfy);
    try testing.expectEqual(@as(u64, formal_summary.guard_violate), z3_summary.guard_violate);
}

fn emitLeanToOwnedString(allocator: std.mem.Allocator, set: obligation.ObligationSet) ![]u8 {
    var buffer = std.Io.Writer.Allocating.init(allocator);
    errdefer buffer.deinit();
    try obligation_to_lean.writeModule(&buffer.writer, set, .{});
    return try buffer.toOwnedSlice();
}

fn dumpManifestToOwnedString(allocator: std.mem.Allocator, set: obligation.ObligationSet) ![]u8 {
    var buffer = std.Io.Writer.Allocating.init(allocator);
    errdefer buffer.deinit();
    try obligation_dump.writeJsonLines(&buffer.writer, set);
    return try buffer.toOwnedSlice();
}

fn collectPackageObligations(allocator: std.mem.Allocator, path: []const u8) !obligation_from_mlir.CollectResult {
    var compilation = try compilePackage(path);
    defer compilation.deinit();

    const hir_result = try compilation.db.lowerToHir(compilation.root_module_id);
    try testing.expect(hir_result.diagnostics.isEmpty());
    return try obligation_from_mlir.collect(allocator, hir_result.module.raw_module, .{});
}

fn optionalStringEqualForTest(lhs: ?[]const u8, rhs: ?[]const u8) bool {
    if (lhs) |left| {
        if (rhs) |right| return std.mem.eql(u8, left, right);
        return false;
    }
    return rhs == null;
}

fn ownerEqualForTest(lhs: obligation.Owner, rhs: obligation.Owner) bool {
    return switch (lhs) {
        .module => |left| switch (rhs) {
            .module => |right| std.mem.eql(u8, left, right),
            else => false,
        },
        .function => |left| switch (rhs) {
            .function => |right| std.mem.eql(u8, left.name, right.name) and
                optionalStringEqualForTest(left.module, right.module) and
                optionalStringEqualForTest(left.contract, right.contract),
            else => false,
        },
        .contract => |left| switch (rhs) {
            .contract => |right| std.mem.eql(u8, left, right),
            else => false,
        },
        .trait_method => |left| switch (rhs) {
            .trait_method => |right| std.mem.eql(u8, left.trait_name, right.trait_name) and
                std.mem.eql(u8, left.method_name, right.method_name) and
                optionalStringEqualForTest(left.impl_name, right.impl_name),
            else => false,
        },
        .statement => |left| switch (rhs) {
            .statement => |right| left.ordinal == right.ordinal and
                std.mem.eql(u8, left.function_name, right.function_name),
            else => false,
        },
        .backend => |left| switch (rhs) {
            .backend => |right| left.component == right.component and std.mem.eql(u8, left.name, right.name),
            else => false,
        },
    };
}

fn obligationOwnerByIdForTest(set: obligation.ObligationSet, id: obligation.Id) ?obligation.Owner {
    for (set.obligations) |item| {
        if (item.id == id) return item.owner;
    }
    return null;
}

fn assumptionOwnerByIdForTest(set: obligation.ObligationSet, id: obligation.Id) ?obligation.Owner {
    for (set.assumptions) |item| {
        if (item.id == id) return item.owner;
    }
    return null;
}

fn expectQueriesOwnerScoped(set: obligation.ObligationSet) !void {
    for (set.queries) |query| {
        for (query.obligation_ids) |id| {
            const owner = obligationOwnerByIdForTest(set, id) orelse return error.TestUnexpectedResult;
            try testing.expect(ownerEqualForTest(query.owner, owner));
        }
        for (query.assumption_ids) |id| {
            const owner = assumptionOwnerByIdForTest(set, id) orelse return error.TestUnexpectedResult;
            try testing.expect(ownerEqualForTest(query.owner, owner));
        }
    }
}

fn pathFromTmpAlloc(allocator: std.mem.Allocator, tmp: std.testing.TmpDir, rel_path: []const u8) ![]u8 {
    return std.fmt.allocPrint(allocator, ".zig-cache/tmp/{s}/{s}", .{ tmp.sub_path, rel_path });
}

fn runProcess(allocator: std.mem.Allocator, argv: []const []const u8) !std.process.RunResult {
    var io_thread = std.Io.Threaded.init(allocator, .{
        .async_limit = .nothing,
        .concurrent_limit = .nothing,
    });
    defer io_thread.deinit();

    return std.process.run(allocator, io_thread.io(), .{
        .argv = argv,
        .stdout_limit = std.Io.Limit.limited(8 * 1024 * 1024),
        .stderr_limit = std.Io.Limit.limited(8 * 1024 * 1024),
    });
}

fn leanPathEnvArgForTest(allocator: std.mem.Allocator) ![]const u8 {
    const inherited_path = if (std.c.getenv("PATH")) |path| std.mem.span(path) else "/usr/bin:/bin:/opt/homebrew/bin";
    if (std.c.getenv("HOME")) |home| {
        const home_slice = std.mem.span(home);
        return try std.fmt.allocPrint(allocator, "PATH={s}/.elan/bin:{s}", .{ home_slice, inherited_path });
    }
    return try std.fmt.allocPrint(allocator, "PATH={s}", .{inherited_path});
}

fn runOraWithForcedUnknown(
    allocator: std.mem.Allocator,
    forced_query: []const u8,
    args: []const []const u8,
) !std.process.RunResult {
    const forced_unknown_arg = try std.fmt.allocPrint(allocator, "ORA_Z3_TEST_FORCE_UNKNOWN_QUERY={s}", .{forced_query});
    defer allocator.free(forced_unknown_arg);
    const path_arg = try leanPathEnvArgForTest(allocator);
    defer allocator.free(path_arg);

    var argv = try allocator.alloc([]const u8, args.len + 3);
    defer allocator.free(argv);
    argv[0] = "/usr/bin/env";
    argv[1] = forced_unknown_arg;
    argv[2] = path_arg;
    @memcpy(argv[3..], args);
    return runProcess(allocator, argv);
}

fn runProcessWithLeanPath(allocator: std.mem.Allocator, args: []const []const u8) !std.process.RunResult {
    const path_arg = try leanPathEnvArgForTest(allocator);
    defer allocator.free(path_arg);

    var argv = try allocator.alloc([]const u8, args.len + 2);
    defer allocator.free(argv);
    argv[0] = "/usr/bin/env";
    argv[1] = path_arg;
    @memcpy(argv[2..], args);
    return runProcess(allocator, argv);
}

fn runLeanFileForTest(allocator: std.mem.Allocator, path_from_repo_root: []const u8) !std.process.RunResult {
    const path_arg = try leanPathEnvArgForTest(allocator);
    defer allocator.free(path_arg);
    const lean_path = if (std.fs.path.isAbsolute(path_from_repo_root))
        try allocator.dupe(u8, path_from_repo_root)
    else
        try std.fs.path.join(allocator, &.{ "..", path_from_repo_root });
    defer allocator.free(lean_path);

    var io_thread = std.Io.Threaded.init(allocator, .{
        .async_limit = .nothing,
        .concurrent_limit = .nothing,
    });
    defer io_thread.deinit();

    const argv = [_][]const u8{ "/usr/bin/env", path_arg, "lake", "env", "lean", lean_path };
    return std.process.run(allocator, io_thread.io(), .{
        .argv = &argv,
        .cwd = .{ .path = "formal" },
        .stdout_limit = std.Io.Limit.limited(1024 * 1024),
        .stderr_limit = std.Io.Limit.limited(1024 * 1024),
    });
}

fn expectExited(result: std.process.RunResult, expected: u8) !void {
    switch (result.term) {
        .exited => |code| {
            if (code != expected) {
                std.debug.print("process stdout:\n{s}\nprocess stderr:\n{s}\n", .{ result.stdout, result.stderr });
            }
            try testing.expectEqual(expected, code);
        },
        else => {
            std.debug.print("process term: {any}\nprocess stdout:\n{s}\nprocess stderr:\n{s}\n", .{ result.term, result.stdout, result.stderr });
            return error.TestUnexpectedResult;
        },
    }
}

fn readFileAllocForTest(allocator: std.mem.Allocator, path: []const u8) ![]u8 {
    return std.Io.Dir.cwd().readFileAlloc(
        std.testing.io,
        path,
        allocator,
        std.Io.Limit.limited(1024 * 1024),
    );
}

fn expectNoFormalArtifactsInRepoRoot(
    allocator: std.mem.Allocator,
    stem: []const u8,
) !void {
    for (artifact_catalog.suffixes) |suffix| {
        const name = try std.fmt.allocPrint(allocator, "{s}{s}", .{ stem, suffix });
        defer allocator.free(name);
        try testing.expectError(
            error.FileNotFound,
            std.Io.Dir.cwd().access(std.testing.io, name, .{}),
        );
    }
}

fn leanProofGeneratedNamespaceForTest(allocator: std.mem.Allocator, file_path: []const u8) ![]const u8 {
    const stem = std.fs.path.stem(file_path);
    var component_out = std.Io.Writer.Allocating.init(allocator);
    defer component_out.deinit();
    const writer = &component_out.writer;

    try writer.writeAll("Source_");
    for (stem) |byte| {
        if (std.ascii.isAlphanumeric(byte) or byte == '_') {
            try writer.writeByte(byte);
        } else {
            try writer.writeByte('_');
        }
    }

    const hash = std.hash.Wyhash.hash(0, file_path);
    return try std.fmt.allocPrint(
        allocator,
        "Ora.Generated.Obligations.{s}_{x}",
        .{ component_out.written(), hash },
    );
}

fn moduleSuffixFromTmp(allocator: std.mem.Allocator, tmp: std.testing.TmpDir) ![]const u8 {
    var out = std.Io.Writer.Allocating.init(allocator);
    errdefer out.deinit();
    try out.writer.writeAll("Run_");
    for (tmp.sub_path) |byte| {
        if (std.ascii.isAlphanumeric(byte) or byte == '_') {
            try out.writer.writeByte(byte);
        } else {
            try out.writer.writeByte('_');
        }
    }
    return try out.toOwnedSlice();
}

fn findEnsuresQuery(set: obligation.ObligationSet) !obligation.VerificationQuery {
    for (set.queries) |query| {
        if (query.kind == .obligation and query.logical_role != null and query.logical_role.? == .ensures) return query;
    }
    return error.TestUnexpectedResult;
}

fn findLoopInductionQuery(set: obligation.ObligationSet) !obligation.VerificationQuery {
    for (set.queries) |query| {
        if (query.kind == .loop_induction and query.loop_summary_id != null) return query;
    }
    return error.TestUnexpectedResult;
}

fn findLogicalQuery(set: obligation.ObligationSet, role: obligation.LogicalRole) !obligation.VerificationQuery {
    for (set.queries) |query| {
        if (query.kind == .obligation and query.logical_role != null and query.logical_role.? == role) return query;
    }
    return error.TestUnexpectedResult;
}

fn expectLogicalQueryUnsupported(
    set: obligation.ObligationSet,
    role: obligation.LogicalRole,
    expected: obligation_to_lean.SemanticUnsupportedReason,
) !void {
    switch (obligation_to_lean.querySemanticSupport(set, try findLogicalQuery(set, role))) {
        .supported => return error.TestUnexpectedResult,
        .unsupported => |reason| try testing.expectEqual(expected, reason),
    }
}

fn expectLogicalQuerySupported(set: obligation.ObligationSet, role: obligation.LogicalRole) !void {
    switch (obligation_to_lean.querySemanticSupport(set, try findLogicalQuery(set, role))) {
        .supported => {},
        .unsupported => return error.TestUnexpectedResult,
    }
}

fn expectEnsuresQueryUnsupported(
    set: obligation.ObligationSet,
    expected: obligation_to_lean.SemanticUnsupportedReason,
) !void {
    switch (obligation_to_lean.querySemanticSupport(set, try findEnsuresQuery(set))) {
        .supported => return error.TestUnexpectedResult,
        .unsupported => |reason| try testing.expectEqual(expected, reason),
    }
}

fn expectEnsuresQuerySupported(set: obligation.ObligationSet) !void {
    switch (obligation_to_lean.querySemanticSupport(set, try findEnsuresQuery(set))) {
        .supported => {},
        .unsupported => return error.TestUnexpectedResult,
    }
}

fn findEffectFrameQuery(set: obligation.ObligationSet, relation: obligation.EffectFrameRelation) !obligation.VerificationQuery {
    for (set.obligations) |item| {
        if (item.kind != .effect_frame or item.kind.effect_frame.relation != relation) continue;
        for (set.queries) |query| {
            for (query.obligation_ids) |id| {
                if (id == item.id) return query;
            }
        }
    }
    return error.TestUnexpectedResult;
}

fn expectEffectFrameQuerySupported(set: obligation.ObligationSet, relation: obligation.EffectFrameRelation) !void {
    switch (obligation_to_lean.querySemanticSupport(set, try findEffectFrameQuery(set, relation))) {
        .supported => {},
        .unsupported => return error.TestUnexpectedResult,
    }
}

fn expectEffectFrameQueryUnsupported(
    set: obligation.ObligationSet,
    relation: obligation.EffectFrameRelation,
    expected: obligation_to_lean.SemanticUnsupportedReason,
) !void {
    switch (obligation_to_lean.querySemanticSupport(set, try findEffectFrameQuery(set, relation))) {
        .supported => return error.TestUnexpectedResult,
        .unsupported => |reason| try testing.expectEqual(expected, reason),
    }
}

fn expectSyntheticKeyEvidenceUnsupported(
    terms: []const obligation.Term,
    formula_term: obligation.TermId,
    assumption_kind: obligation.AssumptionKind,
    query_assumption_ids: []const obligation.Id,
    assumption_owner: obligation.Owner,
    query_owner: obligation.Owner,
    read: obligation.PlaceRef,
    write: obligation.PlaceRef,
    lhs: obligation.FreeVarId,
    rhs: obligation.FreeVarId,
    expected: obligation_to_lean.SemanticUnsupportedReason,
) !void {
    const assumption_id: obligation.Id = 10;
    const obligation_id: obligation.Id = 20;
    const obligation_ids = [_]obligation.Id{obligation_id};
    const assumptions = [_]obligation.Assumption{.{
        .id = assumption_id,
        .owner = assumption_owner,
        .source = .generated(),
        .phase = .ora_mlir,
        .origin = .source,
        .kind = assumption_kind,
        .formula = .{ .term = formula_term },
    }};
    const evidence = [_]obligation.KeyDisjointEvidence{.{
        .kind = .free_var_disequality,
        .assumption_id = assumption_id,
        .lhs = lhs,
        .rhs = rhs,
        .read = read,
        .write = write,
        .key_index = 0,
    }};
    const declared = [_]obligation.PlaceRef{write};
    const actual = [_]obligation.PlaceRef{read};
    const obligations = [_]obligation.Obligation{.{
        .id = obligation_id,
        .owner = query_owner,
        .source = .generated(),
        .phase = .ora_mlir,
        .origin = .source,
        .kind = .{ .effect_frame = .{
            .relation = .read_preserved_by_key_evidence,
            .declared = &declared,
            .actual = &actual,
            .evidence = &evidence,
        } },
    }};
    const queries = [_]obligation.VerificationQuery{.{
        .id = 30,
        .owner = query_owner,
        .source = .generated(),
        .phase = .ora_mlir,
        .origin = .source,
        .kind = .obligation,
        .obligation_ids = &obligation_ids,
        .assumption_ids = query_assumption_ids,
    }};
    const set: obligation.ObligationSet = .{
        .obligations = &obligations,
        .assumptions = &assumptions,
        .queries = &queries,
        .terms = terms,
    };

    switch (obligation_to_lean.querySemanticSupport(set, queries[0])) {
        .supported => return error.TestUnexpectedResult,
        .unsupported => |reason| try testing.expectEqual(expected, reason),
    }
}

fn writeJsonStringForTest(writer: anytype, value: []const u8) !void {
    try writer.writeByte('"');
    for (value) |byte| {
        switch (byte) {
            '\\' => try writer.writeAll("\\\\"),
            '"' => try writer.writeAll("\\\""),
            '\n' => try writer.writeAll("\\n"),
            '\r' => try writer.writeAll("\\r"),
            '\t' => try writer.writeAll("\\t"),
            else => try writer.writeByte(byte),
        }
    }
    try writer.writeByte('"');
}

fn writeProofManifestForTest(
    allocator: std.mem.Allocator,
    path: []const u8,
    module_name: []const u8,
    theorem_name: []const u8,
    proof_path: []const u8,
    query: obligation.VerificationQuery,
) !void {
    try writeProofManifestIdsForTest(
        allocator,
        path,
        module_name,
        theorem_name,
        proof_path,
        query.id,
        query.obligation_ids,
        query.assumption_ids,
    );
}

fn writeProofManifestIdsForTest(
    allocator: std.mem.Allocator,
    path: []const u8,
    module_name: []const u8,
    theorem_name: []const u8,
    proof_path: []const u8,
    query_id: obligation.Id,
    obligation_ids: []const obligation.Id,
    assumption_ids: []const obligation.Id,
) !void {
    var out = std.Io.Writer.Allocating.init(allocator);
    defer out.deinit();
    const writer = &out.writer;

    try writer.writeAll("{\n  \"schema_version\": 1,\n  \"proofs\": [\n    {\n      \"query_id\": ");
    try writer.print("{d}", .{query_id});
    try writer.writeAll(",\n      \"obligation_ids\": ");
    try obligation.writeNatList(writer, obligation_ids);
    try writer.writeAll(",\n      \"assumption_ids\": ");
    try obligation.writeNatList(writer, assumption_ids);
    try writer.writeAll(",\n      \"module_name\": ");
    try writeJsonStringForTest(writer, module_name);
    try writer.writeAll(",\n      \"theorem_name\": ");
    try writeJsonStringForTest(writer, theorem_name);
    try writer.writeAll(",\n      \"path\": ");
    try writeJsonStringForTest(writer, proof_path);
    try writer.writeAll(",\n      \"content_sha256\": null\n    }\n  ]\n}\n");

    try std.Io.Dir.cwd().writeFile(std.testing.io, .{ .sub_path = path, .data = out.written() });
}

fn writeTwoProofManifestForTest(
    allocator: std.mem.Allocator,
    path: []const u8,
    module_name: []const u8,
    first_theorem: []const u8,
    first_proof_path: []const u8,
    first_query: obligation.VerificationQuery,
    second_theorem: []const u8,
    second_proof_path: []const u8,
    second_query: obligation.VerificationQuery,
) !void {
    var out = std.Io.Writer.Allocating.init(allocator);
    defer out.deinit();
    const writer = &out.writer;

    try writer.writeAll("{\n  \"schema_version\": 1,\n  \"proofs\": [\n");
    inline for (.{ .{ first_theorem, first_proof_path, first_query }, .{ second_theorem, second_proof_path, second_query } }, 0..) |row, index| {
        if (index != 0) try writer.writeAll(",\n");
        try writer.writeAll("    {\n      \"query_id\": ");
        try writer.print("{d}", .{row[2].id});
        try writer.writeAll(",\n      \"obligation_ids\": ");
        try obligation.writeNatList(writer, row[2].obligation_ids);
        try writer.writeAll(",\n      \"assumption_ids\": ");
        try obligation.writeNatList(writer, row[2].assumption_ids);
        try writer.writeAll(",\n      \"module_name\": ");
        try writeJsonStringForTest(writer, module_name);
        try writer.writeAll(",\n      \"theorem_name\": ");
        try writeJsonStringForTest(writer, row[0]);
        try writer.writeAll(",\n      \"path\": ");
        try writeJsonStringForTest(writer, row[1]);
        try writer.writeAll(",\n      \"content_sha256\": null\n    }");
    }
    try writer.writeAll("\n  ]\n}\n");

    try std.Io.Dir.cwd().writeFile(std.testing.io, .{ .sub_path = path, .data = out.written() });
}

const ProofRecipeRowForTest = struct {
    query_id: obligation.Id,
    obligation_ids: []obligation.Id,
    assumption_ids: []obligation.Id,

    fn deinit(self: *ProofRecipeRowForTest, allocator: std.mem.Allocator) void {
        allocator.free(self.obligation_ids);
        allocator.free(self.assumption_ids);
    }
};

fn parseIdListForTest(allocator: std.mem.Allocator, text: []const u8) ![]obligation.Id {
    const trimmed = std.mem.trim(u8, text, " \t\r\n");
    if (trimmed.len == 0) return try allocator.alloc(obligation.Id, 0);

    var ids = std.ArrayList(obligation.Id).empty;
    errdefer ids.deinit(allocator);

    var it = std.mem.splitScalar(u8, trimmed, ',');
    while (it.next()) |part| {
        const item = std.mem.trim(u8, part, " \t\r\n");
        if (item.len == 0) return error.TestUnexpectedResult;
        try ids.append(allocator, try std.fmt.parseInt(obligation.Id, item, 10));
    }
    return try ids.toOwnedSlice(allocator);
}

fn parseBracketedIdListForTest(
    allocator: std.mem.Allocator,
    text: []const u8,
    marker: []const u8,
    start: usize,
) ![]obligation.Id {
    const marker_start = std.mem.indexOfPos(u8, text, start, marker) orelse return error.TestUnexpectedResult;
    const list_start = marker_start + marker.len;
    const list_end = std.mem.indexOfScalarPos(u8, text, list_start, ']') orelse return error.TestUnexpectedResult;
    return try parseIdListForTest(allocator, text[list_start..list_end]);
}

fn parseProofRecipeRowForTest(allocator: std.mem.Allocator, text: []const u8) !ProofRecipeRowForTest {
    const marker = "proof row: query_id=";
    const row_start = std.mem.indexOf(u8, text, marker) orelse return error.TestUnexpectedResult;
    const query_start = row_start + marker.len;
    const query_end = std.mem.indexOfScalarPos(u8, text, query_start, ',') orelse return error.TestUnexpectedResult;
    const query_id = try std.fmt.parseInt(obligation.Id, text[query_start..query_end], 10);

    const obligation_ids = try parseBracketedIdListForTest(allocator, text, "obligation_ids=[", query_end);
    errdefer allocator.free(obligation_ids);
    const assumption_ids = try parseBracketedIdListForTest(allocator, text, "assumption_ids=[", query_end);
    errdefer allocator.free(assumption_ids);

    return .{
        .query_id = query_id,
        .obligation_ids = obligation_ids,
        .assumption_ids = assumption_ids,
    };
}

fn proofModuleFromGeneratedObligations(
    allocator: std.mem.Allocator,
    obligations_source: []const u8,
    module_namespace: []const u8,
    query_id: obligation.Id,
    use_sorry: bool,
) ![]const u8 {
    const namespace_start = std.mem.indexOf(u8, obligations_source, "namespace ") orelse return error.TestUnexpectedResult;
    const body_start = (std.mem.indexOfPos(u8, obligations_source, namespace_start, "\n") orelse return error.TestUnexpectedResult) + 1;
    const end_start = std.mem.lastIndexOf(u8, obligations_source, "\nend ") orelse return error.TestUnexpectedResult;

    var out = std.Io.Writer.Allocating.init(allocator);
    errdefer out.deinit();
    const writer = &out.writer;

    try writer.writeAll(obligations_source[0..namespace_start]);
    try writer.writeAll("namespace ");
    try writer.writeAll(module_namespace);
    try writer.writeByte('\n');
    try writer.writeAll(obligations_source[body_start..end_start]);
    try writer.writeByte('\n');

    try writer.print("theorem discharge : emittedQuery_{d} := by\n", .{query_id});
    if (use_sorry) {
        try writer.writeAll("  sorry\n\n");
    } else {
        try writer.writeAll("  unfold ");
        try writer.print("emittedQuery_{d}", .{query_id});
        try writer.writeAll(
            \\ obligationFollowsFromAssumptions
            \\  constructor
            \\  · refine ⟨Env.empty.setFree { file_id := 0, pattern_id := 0 } (.u256 (BitVec.ofNat 256 0)), ?_⟩
            \\    have hId :
            \\        (({ file_id := 0, pattern_id := 0 } : FreeVarId) ==
            \\          { file_id := 0, pattern_id := 0 }) = true := by
            \\      decide
            \\    simp [
            \\      assumptionsDenoteInEnv,
            \\      assumptionsDenoteInEnv?,
            \\      assumptionAnd?,
            \\      assumptionDenotesInEnv?,
            \\      formulaDenotes?,
            \\      denoteFormula?,
            \\      denoteValue?,
            \\      emittedManifest,
            \\      emittedTerms,
            \\      Env.setFree,
            \\      Env.lookupVar,
            \\      Env.lookupFree,
            \\      lookupFreeBinding,
            \\      Value.eqProp?,
            \\      hId
            \\    ]
            \\  · intro env hAssumptions
            \\    intro x
            \\    simp [
            \\      obligationDenotesInEnv,
            \\      obligationDenotesInEnv?,
            \\      formulaDenotes?,
            \\      denoteFormula?,
            \\      denoteValue?,
            \\      emittedManifest,
            \\      emittedTerms,
            \\      Env.pushBound,
            \\      Env.lookupVar,
            \\      Env.lookupBound,
            \\      Value.eqProp?,
            \\      BinderRef.isU256,
            \\      BoundVarRef.isU256,
            \\      TyRef.isU256,
            \\      TyRef.isU256Carrier,
            \\      compilerTypeIdU256,
            \\      compilerTypeIdI256,
            \\      compilerTypeIdBool,
            \\      Ora.Spec.expectedCompilerTypeIdU256,
            \\      Ora.Spec.expectedCompilerTypeIdI256,
            \\      Ora.Spec.expectedCompilerTypeIdBool
            \\    ]
            \\    repeat intro
            \\    exact U256.ult_implies_ule _ _ (by assumption)
            \\
        );
    }
    try writer.writeAll("end ");
    try writer.writeAll(module_namespace);
    try writer.writeByte('\n');
    return try out.toOwnedSlice();
}

fn storageProofModuleFromGeneratedObligations(
    allocator: std.mem.Allocator,
    obligations_source: []const u8,
    module_namespace: []const u8,
    query_id: obligation.Id,
    use_sorry: bool,
) ![]const u8 {
    const namespace_start = std.mem.indexOf(u8, obligations_source, "namespace ") orelse return error.TestUnexpectedResult;
    const body_start = (std.mem.indexOfPos(u8, obligations_source, namespace_start, "\n") orelse return error.TestUnexpectedResult) + 1;
    const end_start = std.mem.lastIndexOf(u8, obligations_source, "\nend ") orelse return error.TestUnexpectedResult;

    var out = std.Io.Writer.Allocating.init(allocator);
    errdefer out.deinit();
    const writer = &out.writer;

    try writer.writeAll(obligations_source[0..namespace_start]);
    try writer.writeAll("namespace ");
    try writer.writeAll(module_namespace);
    try writer.writeByte('\n');
    try writer.writeAll(obligations_source[body_start..end_start]);
    try writer.writeByte('\n');

    try writer.print("theorem discharge : emittedQuery_{d} := by\n", .{query_id});
    if (use_sorry) {
        try writer.writeAll("  sorry\n\n");
    } else {
        try writer.writeAll("  unfold ");
        try writer.print("emittedQuery_{d}", .{query_id});
        try writer.writeAll(
            \\ obligationFollowsFromAssumptions
            \\  constructor
            \\  · refine ⟨Env.empty, ?_⟩
            \\    simp [
            \\      assumptionsDenoteInEnv,
            \\      assumptionsDenoteInEnv?,
            \\      assumptionAnd?,
            \\      assumptionDenotesInEnv?,
            \\      formulaDenotes?,
            \\      denoteFormula?,
            \\      emittedManifest,
            \\      emittedAssumptions
            \\    ]
            \\  · intro env hAssumptions
            \\    simp [
            \\      obligationDenotesInEnv,
            \\      obligationDenotesInEnv?,
            \\      formulaDenotes?,
            \\      denoteFormula?,
            \\      denoteValue?,
            \\      emittedManifest,
            \\      emittedTerms,
            \\      Env.lookupPlace
            \\    ]
            \\    exact stable_place_read_self_eq_denotes env
            \\      { root := "balance", region := .storage, fields := [], keys := [] }
            \\
            \\
        );
    }
    try writer.writeAll("end ");
    try writer.writeAll(module_namespace);
    try writer.writeByte('\n');
    return try out.toOwnedSlice();
}

fn storagePlaceProofModuleFromGeneratedObligations(
    allocator: std.mem.Allocator,
    obligations_source: []const u8,
    module_namespace: []const u8,
    query_id: obligation.Id,
    place_expr: []const u8,
) ![]const u8 {
    const namespace_start = std.mem.indexOf(u8, obligations_source, "namespace ") orelse return error.TestUnexpectedResult;
    const body_start = (std.mem.indexOfPos(u8, obligations_source, namespace_start, "\n") orelse return error.TestUnexpectedResult) + 1;
    const end_start = std.mem.lastIndexOf(u8, obligations_source, "\nend ") orelse return error.TestUnexpectedResult;

    var out = std.Io.Writer.Allocating.init(allocator);
    errdefer out.deinit();
    const writer = &out.writer;

    try writer.writeAll(obligations_source[0..namespace_start]);
    try writer.writeAll("namespace ");
    try writer.writeAll(module_namespace);
    try writer.writeByte('\n');
    try writer.writeAll(obligations_source[body_start..end_start]);
    try writer.writeByte('\n');

    try writer.print("theorem discharge : emittedQuery_{d} := by\n", .{query_id});
    try writer.writeAll("  unfold ");
    try writer.print("emittedQuery_{d}", .{query_id});
    try writer.writeAll(
        \\ obligationFollowsFromAssumptions
        \\  constructor
        \\  · refine ⟨Env.empty, ?_⟩
        \\    simp [
        \\      assumptionsDenoteInEnv,
        \\      assumptionsDenoteInEnv?,
        \\      assumptionAnd?,
        \\      assumptionDenotesInEnv?,
        \\      formulaDenotes?,
        \\      denoteFormula?,
        \\      emittedManifest,
        \\      emittedAssumptions
        \\    ]
        \\  · intro env hAssumptions
        \\    simp [
        \\      obligationDenotesInEnv,
        \\      obligationDenotesInEnv?,
        \\      formulaDenotes?,
        \\      denoteFormula?,
        \\      denoteValue?,
        \\      emittedManifest,
        \\      emittedTerms,
        \\      Env.lookupPlace
        \\    ]
        \\    exact stable_place_read_self_eq_denotes env
    );
    try writer.writeByte(' ');
    try writer.writeAll(place_expr);
    try writer.writeAll(
        \\
        \\
    );

    try writer.writeAll("end ");
    try writer.writeAll(module_namespace);
    try writer.writeByte('\n');
    return try out.toOwnedSlice();
}

fn writeLeanFreeVarIdForTest(writer: anytype, id: obligation.FreeVarId) !void {
    try writer.print("{{ file_id := {d}, pattern_id := {d} }}", .{ id.file_id, id.pattern_id });
}

fn writeLeanFreeVarEqFactForTest(
    writer: anytype,
    indent: []const u8,
    name: []const u8,
    lhs: obligation.FreeVarId,
    rhs: obligation.FreeVarId,
) !void {
    try writer.writeAll(indent);
    try writer.writeAll("have ");
    try writer.writeAll(name);
    try writer.writeAll(" : ((");
    try writeLeanFreeVarIdForTest(writer, lhs);
    try writer.writeAll(" : FreeVarId) == ");
    try writeLeanFreeVarIdForTest(writer, rhs);
    try writer.writeAll(") = ");
    try writer.writeAll(if (obligation.freeVarIdEql(lhs, rhs)) "true" else "false");
    try writer.writeAll(" := by\n");
    try writer.writeAll(indent);
    try writer.writeAll("  decide\n");
}

fn writeLeanStringLiteralForTest(writer: anytype, value: []const u8) !void {
    try writer.writeByte('"');
    for (value) |byte| {
        switch (byte) {
            '\\' => try writer.writeAll("\\\\"),
            '"' => try writer.writeAll("\\\""),
            '\n' => try writer.writeAll("\\n"),
            '\r' => try writer.writeAll("\\r"),
            '\t' => try writer.writeAll("\\t"),
            else => try writer.writeByte(byte),
        }
    }
    try writer.writeByte('"');
}

fn writeLeanRegionRefForTest(writer: anytype, region: obligation.RegionRef) !void {
    try writer.writeAll(switch (region) {
        .none => ".none",
        .storage => ".storage",
        .memory => ".memory",
        .transient => ".transient",
        .calldata => ".calldata",
    });
}

fn writeLeanPlaceKeyForTest(writer: anytype, key: obligation.PlaceKey) !void {
    switch (key) {
        .parameter => |id| {
            try writer.writeAll(".parameter ");
            try writeLeanFreeVarIdForTest(writer, id);
        },
        .comptime_parameter => |index| try writer.print(".comptimeParameter {d}", .{index}),
        .comptime_range_parameter => |index| try writer.print(".comptimeRangeParameter {d}", .{index}),
        .constant => |value| {
            try writer.writeAll(".constant ");
            try writeLeanStringLiteralForTest(writer, value);
        },
        .msg_sender => try writer.writeAll(".msgSender"),
        .tx_origin => try writer.writeAll(".txOrigin"),
        .unknown => try writer.writeAll(".unknown"),
    }
}

fn writeLeanPlaceRefForTest(writer: anytype, place: obligation.PlaceRef) !void {
    try writer.writeAll("{ root := ");
    try writeLeanStringLiteralForTest(writer, place.root);
    try writer.writeAll(", region := ");
    try writeLeanRegionRefForTest(writer, place.region);
    try writer.writeAll(", fields := [");
    for (place.fields, 0..) |field, index| {
        if (index != 0) try writer.writeAll(", ");
        try writeLeanStringLiteralForTest(writer, field);
    }
    try writer.writeAll("], keys := [");
    for (place.keys, 0..) |key, index| {
        if (index != 0) try writer.writeAll(", ");
        try writeLeanPlaceKeyForTest(writer, key);
    }
    try writer.writeAll("] }");
}

fn writeLeanPlaceEqFactForTest(
    writer: anytype,
    indent: []const u8,
    name: []const u8,
    place: obligation.PlaceRef,
) !void {
    try writer.writeAll(indent);
    try writer.writeAll("have ");
    try writer.writeAll(name);
    try writer.writeAll(" : ((");
    try writeLeanPlaceRefForTest(writer, place);
    try writer.writeAll(" : PlaceRef) == ");
    try writeLeanPlaceRefForTest(writer, place);
    try writer.writeAll(") = true := by\n");
    try writer.writeAll(indent);
    try writer.writeAll("  decide\n");
}

fn keyEvidenceFrameProofModuleFromGeneratedObligations(
    allocator: std.mem.Allocator,
    obligations_source: []const u8,
    module_namespace: []const u8,
    query_id: obligation.Id,
    read: obligation.PlaceRef,
    write: obligation.PlaceRef,
    lhs: obligation.FreeVarId,
    rhs: obligation.FreeVarId,
    write_query_id: ?obligation.Id,
    use_sorry: bool,
) ![]const u8 {
    const namespace_start = std.mem.indexOf(u8, obligations_source, "namespace ") orelse return error.TestUnexpectedResult;
    const body_start = (std.mem.indexOfPos(u8, obligations_source, namespace_start, "\n") orelse return error.TestUnexpectedResult) + 1;
    const end_start = std.mem.lastIndexOf(u8, obligations_source, "\nend ") orelse return error.TestUnexpectedResult;

    var out = std.Io.Writer.Allocating.init(allocator);
    errdefer out.deinit();
    const writer = &out.writer;

    try writer.writeAll(obligations_source[0..namespace_start]);
    try writer.writeAll("namespace ");
    try writer.writeAll(module_namespace);
    try writer.writeByte('\n');
    try writer.writeAll(obligations_source[body_start..end_start]);
    try writer.writeByte('\n');

    try writer.print("theorem discharge : emittedQuery_{d} := by\n", .{query_id});
    if (use_sorry) {
        try writer.writeAll("  sorry\n\n");
    } else {
        try writer.writeAll("  unfold ");
        try writer.print("emittedQuery_{d}", .{query_id});
        try writer.writeAll(" obligationFollowsFromAssumptions\n");
        try writeLeanFreeVarEqFactForTest(writer, "  ", "hLhsLhs", lhs, lhs);
        try writeLeanFreeVarEqFactForTest(writer, "  ", "hLhsRhs", lhs, rhs);
        try writeLeanFreeVarEqFactForTest(writer, "  ", "hRhsLhs", rhs, lhs);
        try writeLeanFreeVarEqFactForTest(writer, "  ", "hRhsRhs", rhs, rhs);
        try writer.writeAll("  have hRhsNeLhs : ((");
        try writeLeanFreeVarIdForTest(writer, rhs);
        try writer.writeAll(" : FreeVarId) != ");
        try writeLeanFreeVarIdForTest(writer, lhs);
        try writer.writeAll(") = true := by\n    decide\n");
        try writer.writeAll("  have hStorageSelfNe : (RegionRef.storage != RegionRef.storage) = false := by\n    decide\n");
        try writer.writeAll("  have hRequiresSelfNe : (AssumptionKind.requires != AssumptionKind.requires) = false := by\n    decide\n");
        try writer.writeAll("  have hBinaryNeSelfNe : (BinaryOp.ne != BinaryOp.ne) = false := by\n    decide\n");
        try writeLeanPlaceEqFactForTest(writer, "  ", "hReadRead", read);
        try writeLeanPlaceEqFactForTest(writer, "  ", "hWriteWrite", write);
        try writer.writeAll(
            \\  constructor
            \\  · refine ⟨(Env.empty.setFree
        );
        try writer.writeAll(" ");
        try writeLeanFreeVarIdForTest(writer, lhs);
        try writer.writeAll(" (.u256 (BitVec.ofNat 256 0))).setFree ");
        try writeLeanFreeVarIdForTest(writer, rhs);
        try writer.writeAll(
            \\ (.u256 (BitVec.ofNat 256 1)), ?_⟩
            \\    have hNe : ¬(BitVec.ofNat 256 0 : U256) = BitVec.ofNat 256 1 := by
            \\      decide
            \\    simp [
            \\      assumptionsDenoteInEnv,
            \\      assumptionsDenoteInEnv?,
            \\      assumptionAnd?,
            \\      assumptionDenotesInEnv?,
            \\      formulaDenotes?,
            \\      denoteFormula?,
            \\      denoteValue?,
            \\      emittedManifest,
            \\      emittedTerms,
            \\      emittedAssumptions,
            \\      Env.setFree,
            \\      Env.lookupVar,
            \\      Env.lookupFree,
            \\      lookupFreeBinding,
            \\      Value.eqProp?,
            \\      hLhsLhs,
            \\      hLhsRhs,
            \\      hRhsLhs,
            \\      hRhsRhs,
            \\      hRhsNeLhs,
            \\      hStorageSelfNe,
            \\      hRequiresSelfNe,
            \\      hBinaryNeSelfNe,
            \\      hReadRead,
            \\      hWriteWrite,
            \\      hNe
            \\    ]
            \\  · intro env hAssumptions
            \\    simp [
            \\      assumptionsDenoteInEnv,
            \\      assumptionsDenoteInEnv?,
            \\      assumptionAnd?,
            \\      assumptionDenotesInEnv?,
            \\      formulaDenotes?,
            \\      denoteFormula?,
            \\      denoteValue?,
            \\      emittedManifest,
            \\      emittedTerms,
            \\      emittedAssumptions,
            \\      Env.lookupVar,
            \\      Env.lookupFree,
            \\      lookupFreeBinding,
            \\      Value.eqProp?,
            \\      hLhsLhs,
            \\      hLhsRhs,
            \\      hRhsLhs,
            \\      hRhsRhs,
            \\      hRhsNeLhs,
            \\      hStorageSelfNe,
            \\      hRequiresSelfNe,
            \\      hBinaryNeSelfNe,
            \\      hReadRead,
            \\      hWriteWrite
            \\    ] at hAssumptions
            \\    simp [
            \\      obligationDenotesInEnv,
            \\      obligationDenotesInEnv?,
            \\      effectFrameGoalDenotes?,
            \\      placeListDisjointWithEvidence?,
            \\      placePairDisjointWithEvidence?,
            \\      placeDefinitelyDisjoint,
            \\      placeKeyListsDefinitelyDisjoint,
            \\      placeKeysDefinitelyDistinct,
            \\      RegionRef.isConcrete,
            \\      computedStorageRoot,
            \\      pairCoveredByEvidence,
            \\      evidenceMatchesPair,
            \\      evidenceListDenotes?,
            \\      keyDisjointEvidenceDenotes?,
            \\      keyEvidencePathMatches,
            \\      keyDisjointEvidenceFormulaDenotes?,
            \\      Manifest.assumptionById,
            \\      termFreeVarId?,
            \\      freeVarPairMatches,
            \\      placeKeysEqualBefore,
            \\      optionPropAnd?,
            \\      assumptionsDenoteInEnv,
            \\      assumptionsDenoteInEnv?,
            \\      assumptionAnd?,
            \\      assumptionDenotesInEnv?,
            \\      formulaDenotes?,
            \\      denoteFormula?,
            \\      denoteValue?,
            \\      emittedManifest,
            \\      emittedTerms,
            \\      emittedAssumptions,
            \\      emittedObligations,
            \\      Env.lookupVar,
            \\      Env.lookupFree,
            \\      lookupFreeBinding,
            \\      Value.eqProp?,
            \\      TyRef.isU256,
            \\      TyRef.isI256,
            \\      TyRef.isU256Carrier,
            \\      compilerTypeIdU256,
            \\      compilerTypeIdI256,
            \\      Ora.Spec.expectedCompilerTypeIdU256,
            \\      Ora.Spec.expectedCompilerTypeIdI256,
            \\      hLhsLhs,
            \\      hLhsRhs,
            \\      hRhsLhs,
            \\      hRhsRhs,
            \\      hRhsNeLhs,
            \\      hStorageSelfNe,
            \\      hRequiresSelfNe,
            \\      hBinaryNeSelfNe,
            \\      hReadRead,
            \\      hWriteWrite,
            \\      hAssumptions
            \\    ]
            \\    exact option_prop_and_true_left_intro _ hAssumptions
            \\
            \\
        );
    }

    if (write_query_id) |id| {
        try writer.print("theorem discharge_write : emittedQuery_{d} := by\n", .{id});
        try writer.writeAll("  unfold ");
        try writer.print("emittedQuery_{d}", .{id});
        try writer.writeAll(" obligationFollowsFromAssumptions\n");
        try writeLeanPlaceEqFactForTest(writer, "  ", "hWriteWrite", write);
        try writer.writeAll(
            \\  constructor
            \\  · refine ⟨Env.empty, ?_⟩
            \\    simp [
            \\      assumptionsDenoteInEnv,
            \\      assumptionsDenoteInEnv?
            \\    ]
            \\  · intro env hAssumptions
            \\    simp [
            \\      obligationDenotesInEnv,
            \\      obligationDenotesInEnv?,
            \\      effectFrameGoalDenotes?,
            \\      placeListCovers,
            \\      emittedManifest,
            \\      emittedObligations,
            \\      hWriteWrite
            \\    ]
            \\
            \\
        );
    }
    try writer.writeAll("end ");
    try writer.writeAll(module_namespace);
    try writer.writeByte('\n');
    return try out.toOwnedSlice();
}

fn signedComparisonProofModuleFromGeneratedObligations(
    allocator: std.mem.Allocator,
    obligations_source: []const u8,
    module_namespace: []const u8,
    query_id: obligation.Id,
) ![]const u8 {
    const namespace_start = std.mem.indexOf(u8, obligations_source, "namespace ") orelse return error.TestUnexpectedResult;
    const body_start = (std.mem.indexOfPos(u8, obligations_source, namespace_start, "\n") orelse return error.TestUnexpectedResult) + 1;
    const end_start = std.mem.lastIndexOf(u8, obligations_source, "\nend ") orelse return error.TestUnexpectedResult;

    var out = std.Io.Writer.Allocating.init(allocator);
    errdefer out.deinit();
    const writer = &out.writer;

    try writer.writeAll(obligations_source[0..namespace_start]);
    try writer.writeAll("namespace ");
    try writer.writeAll(module_namespace);
    try writer.writeByte('\n');
    try writer.writeAll(obligations_source[body_start..end_start]);
    try writer.writeByte('\n');

    try writer.print("theorem discharge : emittedQuery_{d} := by\n", .{query_id});
    try writer.writeAll("  unfold ");
    try writer.print("emittedQuery_{d}", .{query_id});
    try writer.writeAll(
        \\ obligationFollowsFromAssumptions
        \\  constructor
        \\  · refine ⟨Env.empty, ?_⟩
        \\    simp [
        \\      assumptionsDenoteInEnv,
        \\      assumptionsDenoteInEnv?,
        \\      assumptionAnd?,
        \\      assumptionDenotesInEnv?,
        \\      formulaDenotes?,
        \\      denoteFormula?,
        \\      emittedManifest,
        \\      emittedAssumptions
        \\    ]
        \\  · intro env hAssumptions
        \\    intro x
        \\    simp [
        \\      obligationDenotesInEnv,
        \\      obligationDenotesInEnv?,
        \\      formulaDenotes?,
        \\      denoteFormula?,
        \\      denoteValue?,
        \\      emittedManifest,
        \\      emittedTerms,
        \\      Env.pushBound,
        \\      Env.lookupVar,
        \\      Env.lookupBound,
        \\      Value.eqProp?,
        \\      BinderRef.isU256,
        \\      BoundVarRef.isU256,
        \\      TyRef.isU256,
        \\      TyRef.isI256,
        \\      TyRef.isU256Carrier,
        \\      compilerTypeIdU256,
        \\      compilerTypeIdI256,
        \\      compilerTypeIdBool,
        \\      Ora.Spec.expectedCompilerTypeIdU256,
        \\      Ora.Spec.expectedCompilerTypeIdI256,
        \\      Ora.Spec.expectedCompilerTypeIdBool,
        \\      U256.sle
        \\    ]
        \\
        \\
    );
    try writer.writeAll("end ");
    try writer.writeAll(module_namespace);
    try writer.writeByte('\n');
    return try out.toOwnedSlice();
}

fn divRemProofModuleFromGeneratedObligations(
    allocator: std.mem.Allocator,
    obligations_source: []const u8,
    module_namespace: []const u8,
    query_id: obligation.Id,
    requires_nonzero_y: bool,
) ![]const u8 {
    const namespace_start = std.mem.indexOf(u8, obligations_source, "namespace ") orelse return error.TestUnexpectedResult;
    const body_start = (std.mem.indexOfPos(u8, obligations_source, namespace_start, "\n") orelse return error.TestUnexpectedResult) + 1;
    const end_start = std.mem.lastIndexOf(u8, obligations_source, "\nend ") orelse return error.TestUnexpectedResult;

    var out = std.Io.Writer.Allocating.init(allocator);
    errdefer out.deinit();
    const writer = &out.writer;

    try writer.writeAll(obligations_source[0..namespace_start]);
    try writer.writeAll("namespace ");
    try writer.writeAll(module_namespace);
    try writer.writeByte('\n');
    try writer.writeAll(obligations_source[body_start..end_start]);
    try writer.writeByte('\n');

    try writer.print("theorem discharge : emittedQuery_{d} := by\n", .{query_id});
    try writer.writeAll("  unfold ");
    try writer.print("emittedQuery_{d}", .{query_id});
    try writer.writeAll(
        \\ obligationFollowsFromAssumptions
        \\  constructor
        \\
    );
    if (requires_nonzero_y) {
        try writer.writeAll(
            \\  · refine ⟨Env.empty.setFree { file_id := 0, pattern_id := 1 } (.u256 (BitVec.ofNat 256 1)), ?_⟩
            \\    have hY :
            \\        (({ file_id := 0, pattern_id := 1 } : FreeVarId) ==
            \\          { file_id := 0, pattern_id := 1 }) = true := by
            \\      decide
            \\    simp [
            \\      assumptionsDenoteInEnv,
            \\      assumptionsDenoteInEnv?,
            \\      assumptionAnd?,
            \\      assumptionDenotesInEnv?,
            \\      formulaDenotes?,
            \\      denoteFormula?,
            \\      denoteValue?,
            \\      emittedManifest,
            \\      emittedTerms,
            \\      emittedAssumptions,
            \\      Env.setFree,
            \\      Env.lookupVar,
            \\      Env.lookupFree,
            \\      lookupFreeBinding,
            \\      Value.eqProp?,
            \\      IntegerLiteralTerm.asU256?,
            \\      compilerTypeIdU256,
            \\      Ora.Spec.expectedCompilerTypeIdU256,
            \\      TyRef.isU256,
            \\      TyRef.isU256Carrier,
            \\      hY
            \\    ]
            \\
        );
    } else {
        try writer.writeAll(
            \\  · refine ⟨Env.empty, ?_⟩
            \\    simp [
            \\      assumptionsDenoteInEnv,
            \\      assumptionsDenoteInEnv?
            \\    ]
            \\
        );
    }
    try writer.writeAll(
        \\  · intro env hAssumptions
        \\    intro x
        \\    simp [
        \\      obligationDenotesInEnv,
        \\      obligationDenotesInEnv?,
        \\      formulaDenotes?,
        \\      denoteFormula?,
        \\      denoteValue?,
        \\      emittedManifest,
        \\      emittedTerms,
        \\      Env.pushBound,
        \\      Env.lookupVar,
        \\      Env.lookupBound,
        \\      Value.eqProp?,
        \\      Value.binaryU256?,
        \\      IntegerLiteralTerm.asU256?,
        \\      BinderRef.isU256,
        \\      TyRef.isU256,
        \\      TyRef.isI256,
        \\      TyRef.isU256Carrier,
        \\      compilerTypeIdU256,
        \\      compilerTypeIdI256,
        \\      compilerTypeIdBool,
        \\      Ora.Spec.expectedCompilerTypeIdU256,
        \\      Ora.Spec.expectedCompilerTypeIdI256,
        \\      Ora.Spec.expectedCompilerTypeIdBool,
        \\      Ora.Integer.divResult,
        \\      Ora.Integer.unsignedDivTotal
        \\    ]
        \\
        \\
    );
    try writer.writeAll("end ");
    try writer.writeAll(module_namespace);
    try writer.writeByte('\n');
    return try out.toOwnedSlice();
}

test "loop support classifier reserves induction for owner-scoped summaries" {
    const obligation_ids = [_]obligation.Id{1};
    const terms = [_]obligation.Term{.{ .bool_lit = true }};
    const obligations = [_]obligation.Obligation{.{
        .id = 1,
        .owner = .{ .function = .{ .name = "loop" } },
        .source = .generated(),
        .phase = .ora_mlir,
        .origin = .source,
        .kind = .{ .logical = .{
            .role = .loop_invariant,
            .formula = .{ .term = 0 },
        } },
    }};
    const set: obligation.ObligationSet = .{
        .obligations = &obligations,
        .terms = &terms,
    };

    inline for (.{
        obligation.VerificationQueryKind.loop_invariant_step,
        obligation.VerificationQueryKind.loop_body_safety,
    }) |kind| {
        const query: obligation.VerificationQuery = .{
            .id = 2,
            .owner = .{ .function = .{ .name = "loop" } },
            .source = .generated(),
            .phase = .report,
            .origin = .source,
            .kind = kind,
            .obligation_ids = &obligation_ids,
        };
        switch (obligation_to_lean.querySemanticSupport(set, query)) {
            .supported => return error.TestUnexpectedResult,
            .unsupported => |reason| try testing.expectEqual(
                obligation_to_lean.SemanticUnsupportedReason.unsupported_obligation_kind,
                reason,
            ),
        }
    }

    const post_query: obligation.VerificationQuery = .{
        .id = 3,
        .owner = .{ .function = .{ .name = "loop" } },
        .source = .generated(),
        .phase = .report,
        .origin = .source,
        .kind = .loop_invariant_post,
        .obligation_ids = &obligation_ids,
    };
    switch (obligation_to_lean.querySemanticSupport(set, post_query)) {
        .supported => return error.TestUnexpectedResult,
        .unsupported => |reason| try testing.expectEqual(
            obligation_to_lean.SemanticUnsupportedReason.unsupported_obligation_kind,
            reason,
        ),
    }

    var induction_query = post_query;
    induction_query.kind = .loop_induction;
    induction_query.obligation_ids = &.{};
    switch (obligation_to_lean.querySemanticSupport(set, induction_query)) {
        .supported => return error.TestUnexpectedResult,
        .unsupported => |reason| try testing.expectEqual(
            obligation_to_lean.SemanticUnsupportedReason.loop_summary_missing,
            reason,
        ),
    }
}

test "wrapping scalar loop steps do not invent arithmetic safety requirements" {
    const source_text =
        \\pub fn count_wrapping() -> u256 {
        \\    var i: u256 = 0;
        \\    while (i < 3)
        \\        invariant i <= 3
        \\    {
        \\        i = i +% 1;
        \\    }
        \\    return i;
        \\}
    ;

    var compilation = try compileText(source_text);
    defer compilation.deinit();
    const hir_result = try compilation.db.lowerToHir(compilation.root_module_id);
    try testing.expect(hir_result.diagnostics.isEmpty());

    var result = try obligation_from_mlir.collect(
        testing.allocator,
        hir_result.module.raw_module,
        .{},
    );
    defer result.deinit();

    try testing.expectEqual(@as(usize, 1), result.set.loop_summaries.len);
    const summary = result.set.loop_summaries[0];
    try testing.expectEqual(@as(usize, 0), summary.edges[0].safetyFormulaCount());
    try testing.expect(!loopSummaryHasReason(summary, .loop_body_safety_missing));
    try testing.expect(summary.projectionSupported());

    const query = try findLoopInductionQuery(result.set);
    try testing.expectEqual(obligation.ProofRequirement.lean_certificate, query.proof_requirement);
}

test "narrow scalar loop comparison retypes compiler generated literal truncation" {
    const source_text =
        \\pub fn count_wrapping_sub() -> u8 {
        \\    var i: u8 = 3;
        \\    while (i > 0) invariant i <= 3 {
        \\        i = i -% 1;
        \\    }
        \\    return i;
        \\}
    ;

    var compilation = try compileText(source_text);
    defer compilation.deinit();
    const hir_result = try compilation.db.lowerToHir(compilation.root_module_id);
    try testing.expect(hir_result.diagnostics.isEmpty());

    var result = try obligation_from_mlir.collect(testing.allocator, hir_result.module.raw_module, .{});
    defer result.deinit();
    try testing.expectEqual(@as(usize, 1), result.set.loop_summaries.len);
    const summary = result.set.loop_summaries[0];
    try testing.expect(summary.projectionSupported());

    const step_id = summary.edges[0].next_state[0].value.term;
    const step = result.set.terms[step_id].binary;
    try testing.expectEqual(obligation.BinaryOp.sub, step.op);
    try testing.expectEqual(@as(u32, 1), step.ty.?.compiler_type_id);

    const guard_formula = summary.edges[0].timeline[0].payload.path_assumption.formulas[0].formula;
    const guard = result.set.terms[guard_formula.term].binary;
    const guard_rhs = result.set.terms[guard.rhs].int_lit;
    try testing.expectEqual(obligation.BinaryOp.gt, guard.op);
    try testing.expectEqual(@as(u32, 1), guard.ty.?.compiler_type_id);
    try testing.expectEqual(@as(u32, 1), guard_rhs.ty.?.compiler_type_id);

    const query = try findLoopInductionQuery(result.set);
    try testing.expectEqual(obligation.ProofRequirement.lean_certificate, query.proof_requirement);
}

test "narrow checked scalar arithmetic safety formulas activate certification" {
    const source_text =
        \\pub fn checked_signed_sub() -> i8 {
        \\    var signedSubValue: i8 = 3;
        \\    while (signedSubValue > 0) invariant signedSubValue <= 3 {
        \\        signedSubValue = signedSubValue - 1;
        \\    }
        \\    return signedSubValue;
        \\}
        \\
        \\pub fn checked_signed_mul() -> i8 {
        \\    var signedMulValue: i8 = 1;
        \\    while (signedMulValue < 3) invariant signedMulValue <= 4 {
        \\        signedMulValue = signedMulValue * 2;
        \\    }
        \\    return signedMulValue;
        \\}
        \\
        \\pub fn checked_unsigned_mul() -> u8 {
        \\    var unsignedMulValue: u8 = 1;
        \\    while (unsignedMulValue < 3) invariant unsignedMulValue <= 4 {
        \\        unsignedMulValue = unsignedMulValue * 2;
        \\    }
        \\    return unsignedMulValue;
        \\}
    ;

    var compilation = try compileText(source_text);
    defer compilation.deinit();
    const hir_result = try compilation.db.lowerToHir(compilation.root_module_id);
    try testing.expect(hir_result.diagnostics.isEmpty());

    var result = try obligation_from_mlir.collect(testing.allocator, hir_result.module.raw_module, .{});
    defer result.deinit();
    try testing.expectEqual(@as(usize, 3), result.set.loop_summaries.len);

    for (result.set.loop_summaries) |summary| {
        try testing.expect(summary.projectionSupported());
        try testing.expect(summary.edges[0].safetyFormulaCount() > 0);
        try testing.expect(!loopSummaryHasReason(summary, .loop_formula_unsupported));
    }
    var induction_count: usize = 0;
    for (result.set.queries) |query| {
        if (query.kind != .loop_induction) continue;
        induction_count += 1;
        try testing.expectEqual(obligation.ProofRequirement.lean_certificate, query.proof_requirement);
        switch (obligation_to_lean.querySemanticSupport(result.set, query)) {
            .supported => {},
            .unsupported => return error.TestUnexpectedResult,
        }
    }
    try testing.expectEqual(@as(usize, 3), induction_count);
}

test "div rem loop safety comes from canonical source assertions" {
    const source_text =
        \\pub fn unsigned_div_loop(start: u8, divisor: u8) -> u8
        \\    requires divisor != 0
        \\{
        \\    var value: u8 = start;
        \\    while (value > 0) invariant value <= start {
        \\        value = value / divisor;
        \\    }
        \\    return value;
        \\}
        \\
        \\pub fn signed_div_loop(start: i8, divisor: i8) -> i8
        \\    requires divisor != 0
        \\{
        \\    var value: i8 = start;
        \\    while (value > 0) invariant value <= start {
        \\        value = value / divisor;
        \\    }
        \\    return value;
        \\}
        \\
        \\pub fn signed_rem_loop(start: i8, divisor: i8) -> i8
        \\    requires divisor != 0
        \\{
        \\    var value: i8 = start;
        \\    while (value > 0) invariant value <= start {
        \\        value = value % divisor;
        \\    }
        \\    return value;
        \\}
    ;

    var compilation = try compileText(source_text);
    defer compilation.deinit();
    const hir_result = try compilation.db.lowerToHir(compilation.root_module_id);
    try testing.expect(hir_result.diagnostics.isEmpty());

    var result = try obligation_from_mlir.collect(testing.allocator, hir_result.module.raw_module, .{});
    defer result.deinit();
    try testing.expectEqual(@as(usize, 3), result.set.loop_summaries.len);
    try testing.expectEqual(@as(usize, 3), countArithmeticSafety(result.set, .division_by_zero));
    try testing.expectEqual(@as(usize, 1), countArithmeticSafety(result.set, .signed_division_overflow));

    var z3_context = try z3_verification.Z3Context.init(testing.allocator);
    defer z3_context.deinit();
    var adapter = obligation_to_z3.Adapter.init(&z3_context, testing.allocator, result.set);

    for (result.set.loop_summaries) |summary| {
        try testing.expect(summary.projectionSupported());
        try testing.expect(!loopSummaryHasReason(summary, .loop_body_safety_missing));
        try testing.expectEqual(@as(usize, 2), summary.edges.len);
        const expected_safety_count: usize = if (std.mem.eql(
            u8,
            summary.owner.statement.function_name,
            "signed_div_loop",
        )) 2 else 1;
        try testing.expectEqual(expected_safety_count, summary.edges[0].safetyFormulaCount());
        try testing.expectEqual(@as(usize, 0), summary.edges[1].safetyFormulaCount());
        for (summary.query_ids.body_safety) |query_id| {
            const query = obligation.findById(result.set.queries, query_id) orelse
                return error.TestUnexpectedResult;
            const checked = try adapter.checkLoopQueryForRow(query);
            try testing.expectEqual(obligation_to_z3.CheckStatus.proved, checked.status);
        }
    }

    const unsafe_source =
        \\pub fn unsafe_div_loop(start: u8, divisor: u8) -> u8 {
        \\    var value: u8 = start;
        \\    while (value > 0) invariant value <= start {
        \\        value = value / divisor;
        \\    }
        \\    return value;
        \\}
    ;
    var unsafe_compilation = try compileText(unsafe_source);
    defer unsafe_compilation.deinit();
    const unsafe_hir = try unsafe_compilation.db.lowerToHir(unsafe_compilation.root_module_id);
    try testing.expect(unsafe_hir.diagnostics.isEmpty());
    var unsafe_result = try obligation_from_mlir.collect(
        testing.allocator,
        unsafe_hir.module.raw_module,
        .{},
    );
    defer unsafe_result.deinit();
    try testing.expectEqual(@as(usize, 1), unsafe_result.set.loop_summaries.len);
    const unsafe_summary = unsafe_result.set.loop_summaries[0];
    try testing.expectEqual(@as(usize, 2), unsafe_summary.query_ids.body_safety.len);
    const unsafe_query = obligation.findById(
        unsafe_result.set.queries,
        unsafe_summary.query_ids.body_safety[0],
    ) orelse return error.TestUnexpectedResult;
    var unsafe_z3_context = try z3_verification.Z3Context.init(testing.allocator);
    defer unsafe_z3_context.deinit();
    var unsafe_adapter = obligation_to_z3.Adapter.init(
        &unsafe_z3_context,
        testing.allocator,
        unsafe_result.set,
    );
    const unsafe_checked = try unsafe_adapter.checkLoopQueryForRow(unsafe_query);
    try testing.expectEqual(obligation_to_z3.CheckStatus.disproved, unsafe_checked.status);
}

test "every registered integer width activates scalar loop certification" {
    const expected = [_]struct {
        owner: []const u8,
        compiler_type_id: u32,
    }{
        .{ .owner = "countU8", .compiler_type_id = 1 },
        .{ .owner = "countU16", .compiler_type_id = 2 },
        .{ .owner = "countU32", .compiler_type_id = 3 },
        .{ .owner = "countU64", .compiler_type_id = 4 },
        .{ .owner = "countU128", .compiler_type_id = 5 },
        .{ .owner = "countU160", .compiler_type_id = 18 },
        .{ .owner = "countU256", .compiler_type_id = 6 },
        .{ .owner = "countI8", .compiler_type_id = 7 },
        .{ .owner = "countI16", .compiler_type_id = 8 },
        .{ .owner = "countI32", .compiler_type_id = 9 },
        .{ .owner = "countI64", .compiler_type_id = 10 },
        .{ .owner = "countI128", .compiler_type_id = 11 },
        .{ .owner = "countI256", .compiler_type_id = 12 },
    };

    var result = try collectPackageObligations(
        testing.allocator,
        "tests/formal/scalar_loop_integer_widths.ora",
    );
    defer result.deinit();

    try testing.expectEqual(expected.len, result.set.loop_summaries.len);
    var induction_query_count: usize = 0;
    for (result.set.queries) |query| {
        if (query.kind != .loop_induction) continue;
        induction_query_count += 1;
        try testing.expectEqual(obligation.ProofRequirement.lean_certificate, query.proof_requirement);
        switch (obligation_to_lean.querySemanticSupport(result.set, query)) {
            .supported => {},
            .unsupported => return error.TestUnexpectedResult,
        }
    }
    try testing.expectEqual(expected.len, induction_query_count);

    for (expected) |wanted| {
        var matched: ?obligation.LoopSummaryRow = null;
        for (result.set.loop_summaries) |summary| {
            if (summary.owner != .statement) continue;
            if (std.mem.eql(u8, summary.owner.statement.function_name, wanted.owner)) {
                matched = summary;
                break;
            }
        }
        const summary = matched orelse return error.TestUnexpectedResult;
        try testing.expect(summary.projectionSupported());
        try testing.expectEqual(@as(usize, 1), summary.variables.len);
        try testing.expectEqual(@as(usize, 1), summary.context_variables.len);
        try testing.expectEqual(
            obligation.TypeRef{ .compiler_type_id = wanted.compiler_type_id },
            summary.variables[0].ty,
        );
        try testing.expectEqual(
            obligation.TypeRef{ .compiler_type_id = wanted.compiler_type_id },
            summary.context_variables[0].ty,
        );

        // Signedness-neutral operations such as equality must retain the
        // compiler-owned parameter type instead of re-inferring unsigned from
        // a signless MLIR carrier.
        try testing.expect(summary.invariant_formulas.len >= 2);
        const context_formula_id = switch (summary.invariant_formulas[1]) {
            .term => |id| id,
            .origin_value => return error.TestUnexpectedResult,
        };
        const context_equality = switch (result.set.terms[context_formula_id]) {
            .binary => |binary| binary,
            else => return error.TestUnexpectedResult,
        };
        try testing.expectEqual(obligation.BinaryOp.eq, context_equality.op);
        try testing.expectEqual(
            obligation.TypeRef{ .compiler_type_id = wanted.compiler_type_id },
            context_equality.ty.?,
        );
        for ([_]obligation.TermId{ context_equality.lhs, context_equality.rhs }) |term_id| {
            const parameter = switch (result.set.terms[term_id]) {
                .variable => |variable| switch (variable) {
                    .free => |free| free,
                    .bound => return error.TestUnexpectedResult,
                },
                else => return error.TestUnexpectedResult,
            };
            try testing.expectEqual(
                obligation.TypeRef{ .compiler_type_id = wanted.compiler_type_id },
                parameter.ty.?,
            );
        }
    }
}

test "canonical invariant-only loop activates Lean induction in the real CLI" {
    std.Io.Dir.cwd().access(std.testing.io, ORA_BINARY_REL, .{}) catch |err| switch (err) {
        error.FileNotFound => return error.SkipZigTest,
        else => return err,
    };

    const allocator = testing.allocator;
    const source_path = "tests/formal/scalar_loop_induction.ora";
    var tmp = testing.tmpDir(.{});
    defer tmp.cleanup();

    var formal_result = try collectPackageObligations(allocator, source_path);
    defer formal_result.deinit();
    const induction_query = try findLoopInductionQuery(formal_result.set);
    const summary = obligation.findById(
        formal_result.set.loop_summaries,
        induction_query.loop_summary_id.?,
    ) orelse return error.TestUnexpectedResult;
    try testing.expect(summary.projectionSupported());
    try testing.expectEqual(@as(usize, 0), summary.post_formulas.len);
    try testing.expectEqual(obligation.ProofRequirement.lean_certificate, induction_query.proof_requirement);
    try testing.expect(formal_result.set.hasLeanCertificateRequirements());

    const blocked_out = try pathFromTmpAlloc(allocator, tmp, "blocked");
    defer allocator.free(blocked_out);
    const accepted_out = try pathFromTmpAlloc(allocator, tmp, "accepted");
    defer allocator.free(accepted_out);
    const accepted_repeat_out = try pathFromTmpAlloc(allocator, tmp, "accepted-repeat");
    defer allocator.free(accepted_repeat_out);
    const empty_manifest = try pathFromTmpAlloc(allocator, tmp, "empty-proofs.json");
    defer allocator.free(empty_manifest);
    try std.Io.Dir.cwd().writeFile(std.testing.io, .{
        .sub_path = empty_manifest,
        .data = "{\"schema_version\":1,\"proofs\":[]}",
    });

    {
        const result = try runProcessWithLeanPath(allocator, &.{
            ORA_BINARY_REL,
            "emit",
            "--emit=smt-report,sir-text,bytecode",
            "--out-dir",
            blocked_out,
            "--lean-proofs",
            empty_manifest,
            source_path,
        });
        defer allocator.free(result.stdout);
        defer allocator.free(result.stderr);
        try expectExited(result, 1);
        try testing.expect(std.mem.containsAtLeast(u8, result.stdout, 1, "certificate-required=1"));
        try testing.expect(std.mem.containsAtLeast(u8, result.stdout, 1, "lean_required_failure"));
    }
    try testing.expectError(
        error.FileNotFound,
        tmp.dir.access(std.testing.io, "blocked/scalar_loop_induction.hex", .{}),
    );

    {
        const result = try runProcessWithLeanPath(allocator, &.{
            ORA_BINARY_REL,
            "emit",
            "--emit=smt-report,sir-text,bytecode",
            "--out-dir",
            accepted_out,
            "--lean-proofs",
            source_path,
        });
        defer allocator.free(result.stdout);
        defer allocator.free(result.stderr);
        try expectExited(result, 0);
        try testing.expect(std.mem.containsAtLeast(u8, result.stdout, 1, "certificate-required=1"));
        try testing.expect(std.mem.containsAtLeast(u8, result.stdout, 1, "Lean loop proof gate accepted:\n  kernel-checked certificates: 1"));
    }

    const report_path = try pathFromTmpAlloc(allocator, tmp, "accepted/scalar_loop_induction.formal.loops.json");
    defer allocator.free(report_path);
    const report = try readFileAllocForTest(allocator, report_path);
    defer allocator.free(report);
    try testing.expect(std.mem.containsAtLeast(
        u8,
        report,
        1,
        "\"verification_boundary\":\"verifier_input_ora_mlir\"",
    ));
    try testing.expect(std.mem.containsAtLeast(u8, report, 1, "\"certificate_required\":1"));
    try testing.expect(std.mem.containsAtLeast(u8, report, 1, "\"status\":\"lean_certificate_required\""));
    try testing.expect(std.mem.containsAtLeast(u8, report, 1, "\"loop_source_op_id\":"));
    try testing.expect(std.mem.containsAtLeast(u8, report, 1, "\"excluded_by\":[]"));

    const certificate_path = try pathFromTmpAlloc(allocator, tmp, "accepted/scalar_loop_induction.lean.proof.json");
    defer allocator.free(certificate_path);
    const certificate = try readFileAllocForTest(allocator, certificate_path);
    defer allocator.free(certificate);
    try testing.expect(std.mem.containsAtLeast(u8, certificate, 1, "\"proof_count\": 1"));
    try testing.expect(std.mem.containsAtLeast(u8, certificate, 1, "\"loop_summary_id\":"));
    try testing.expect(std.mem.containsAtLeast(u8, certificate, 1, "\"loop_vc_sha256\": \""));
    try testing.expect(!std.mem.containsAtLeast(u8, certificate, 1, "sorryAx"));

    {
        const result = try runProcessWithLeanPath(allocator, &.{
            ORA_BINARY_REL,
            "emit",
            "--emit=smt-report,sir-text,bytecode",
            "--out-dir",
            accepted_repeat_out,
            "--lean-proofs",
            source_path,
        });
        defer allocator.free(result.stdout);
        defer allocator.free(result.stderr);
        try expectExited(result, 0);
    }
    const repeat_certificate_path = try pathFromTmpAlloc(
        allocator,
        tmp,
        "accepted-repeat/scalar_loop_induction.lean.proof.json",
    );
    defer allocator.free(repeat_certificate_path);
    const repeat_certificate = try readFileAllocForTest(allocator, repeat_certificate_path);
    defer allocator.free(repeat_certificate);
    try testing.expectEqualStrings(certificate, repeat_certificate);
}

test "automatic scalar while and for LoopVC induction parses and proves bounds" {
    std.Io.Dir.cwd().access(std.testing.io, ORA_BINARY_REL, .{}) catch |err| switch (err) {
        error.FileNotFound => return error.SkipZigTest,
        else => return err,
    };

    const cases = [_]struct {
        source_path: []const u8,
        stem: []const u8,
        proof_count: usize,
    }{
        .{
            .source_path = "tests/formal/scalar_loop_induction.ora",
            .stem = "scalar_loop_induction",
            .proof_count = 1,
        },
        .{
            .source_path = "tests/formal/scalar_loop_post.ora",
            .stem = "scalar_loop_post",
            .proof_count = 1,
        },
        .{
            .source_path = "tests/formal/loop_for_types.ora",
            .stem = "loop_for_types",
            .proof_count = 1,
        },
        .{
            .source_path = "tests/formal/loop_control_targets.ora",
            .stem = "loop_control_targets",
            .proof_count = 4,
        },
        .{
            .source_path = "tests/formal/loop_return_targets.ora",
            .stem = "loop_return_targets",
            .proof_count = 2,
        },
    };

    const allocator = testing.allocator;
    var tmp = testing.tmpDir(.{});
    defer tmp.cleanup();

    for (cases, 0..) |case, index| {
        const output_sub_path = try std.fmt.allocPrint(allocator, "automatic-{d}", .{index});
        defer allocator.free(output_sub_path);
        const output_path = try pathFromTmpAlloc(allocator, tmp, output_sub_path);
        defer allocator.free(output_path);

        const result = try runProcessWithLeanPath(allocator, &.{
            ORA_BINARY_REL,
            "emit",
            "--emit=smt-report,sir-text,bytecode",
            "--out-dir",
            output_path,
            "--lean-proofs",
            case.source_path,
        });
        defer allocator.free(result.stdout);
        defer allocator.free(result.stderr);
        try expectExited(result, 0);
        const certificate_required = try std.fmt.allocPrint(
            allocator,
            "certificate-required={d}",
            .{case.proof_count},
        );
        defer allocator.free(certificate_required);
        try testing.expect(std.mem.containsAtLeast(u8, result.stdout, 1, certificate_required));
        try testing.expect(!std.mem.containsAtLeast(u8, result.stdout, 1, "Automatic Lean proof gate failed"));
        try testing.expect(!std.mem.containsAtLeast(u8, result.stderr, 1, "unexpected token"));

        const certificate_sub_path = try std.fmt.allocPrint(
            allocator,
            "{s}/{s}.lean.proof.json",
            .{ output_sub_path, case.stem },
        );
        defer allocator.free(certificate_sub_path);
        const certificate_path = try pathFromTmpAlloc(allocator, tmp, certificate_sub_path);
        defer allocator.free(certificate_path);
        const certificate = try readFileAllocForTest(allocator, certificate_path);
        defer allocator.free(certificate);
        const proof_count = try std.fmt.allocPrint(
            allocator,
            "\"proof_count\": {d}",
            .{case.proof_count},
        );
        defer allocator.free(proof_count);
        try testing.expect(std.mem.containsAtLeast(u8, certificate, 1, proof_count));
        try testing.expect(std.mem.containsAtLeast(u8, certificate, 1, "\"loop_summary_id\":"));
        try testing.expect(std.mem.containsAtLeast(u8, certificate, 1, "\"loop_vc_sha256\": \""));
        try testing.expect(!std.mem.containsAtLeast(u8, certificate, 1, "sorryAx"));
    }
}

test "automatic loop proofs certify all integer widths and multiple carried variables through the real CLI" {
    std.Io.Dir.cwd().access(std.testing.io, ORA_BINARY_REL, .{}) catch |err| switch (err) {
        error.FileNotFound => return error.SkipZigTest,
        else => return err,
    };

    const cases = [_]struct {
        source_path: []const u8,
        stem: []const u8,
        proof_count: usize,
    }{
        .{
            .source_path = "tests/formal/scalar_loop_integer_widths.ora",
            .stem = "scalar_loop_integer_widths",
            .proof_count = 13,
        },
        .{
            .source_path = "tests/formal/scalar_loop_two_variables.ora",
            .stem = "scalar_loop_two_variables",
            .proof_count = 1,
        },
    };

    const allocator = testing.allocator;
    var tmp = testing.tmpDir(.{});
    defer tmp.cleanup();

    for (cases, 0..) |case, index| {
        const output_sub_path = try std.fmt.allocPrint(allocator, "automatic-generic-{d}", .{index});
        defer allocator.free(output_sub_path);
        const output_path = try pathFromTmpAlloc(allocator, tmp, output_sub_path);
        defer allocator.free(output_path);

        const result = try runProcessWithLeanPath(allocator, &.{
            ORA_BINARY_REL,
            "emit",
            "--emit=smt-report,bytecode",
            "--out-dir",
            output_path,
            "--lean-proofs",
            case.source_path,
        });
        defer allocator.free(result.stdout);
        defer allocator.free(result.stderr);
        try expectExited(result, 0);

        const required = try std.fmt.allocPrint(
            allocator,
            "certificate-required={d}",
            .{case.proof_count},
        );
        defer allocator.free(required);
        try testing.expect(std.mem.containsAtLeast(u8, result.stdout, 1, required));
        const accepted = try std.fmt.allocPrint(
            allocator,
            "Lean loop proof gate accepted:\n  kernel-checked certificates: {d}",
            .{case.proof_count},
        );
        defer allocator.free(accepted);
        try testing.expect(std.mem.containsAtLeast(u8, result.stdout, 1, accepted));
        try testing.expect(!std.mem.containsAtLeast(u8, result.stdout, 1, "Automatic Lean proof gate failed"));
        try testing.expect(!std.mem.containsAtLeast(u8, result.stdout, 1, "AutoProofScratch"));

        const certificate_sub_path = try std.fmt.allocPrint(
            allocator,
            "{s}/{s}.lean.proof.json",
            .{ output_sub_path, case.stem },
        );
        defer allocator.free(certificate_sub_path);
        const certificate_path = try pathFromTmpAlloc(allocator, tmp, certificate_sub_path);
        defer allocator.free(certificate_path);
        const certificate = try readFileAllocForTest(allocator, certificate_path);
        defer allocator.free(certificate);
        const proof_count = try std.fmt.allocPrint(
            allocator,
            "\"proof_count\": {d}",
            .{case.proof_count},
        );
        defer allocator.free(proof_count);
        try testing.expect(std.mem.containsAtLeast(u8, certificate, 1, proof_count));
        try testing.expect(std.mem.containsAtLeast(u8, certificate, case.proof_count, "\"loop_vc_sha256\": \""));
        try testing.expect(!std.mem.containsAtLeast(u8, certificate, 1, "sorryAx"));

        const bytecode_sub_path = try std.fmt.allocPrint(
            allocator,
            "{s}/{s}.hex",
            .{ output_sub_path, case.stem },
        );
        defer allocator.free(bytecode_sub_path);
        const bytecode_path = try pathFromTmpAlloc(allocator, tmp, bytecode_sub_path);
        defer allocator.free(bytecode_path);
        const bytecode = try readFileAllocForTest(allocator, bytecode_path);
        defer allocator.free(bytecode);
        try testing.expect(bytecode.len > 0);
    }

    var two_variable_result = try collectPackageObligations(
        allocator,
        "tests/formal/scalar_loop_two_variables.ora",
    );
    defer two_variable_result.deinit();
    try testing.expectEqual(@as(usize, 1), two_variable_result.set.loop_summaries.len);
    const summary = two_variable_result.set.loop_summaries[0];
    try testing.expectEqual(@as(usize, 2), summary.variables.len);
    for (summary.edges) |edge| {
        try testing.expectEqual(summary.variables.len, edge.next_state.len);
    }
}

test "loop production gate checks zero iterations and blocks invalid invariant evidence" {
    std.Io.Dir.cwd().access(std.testing.io, ORA_BINARY_REL, .{}) catch |err| switch (err) {
        error.FileNotFound => return error.SkipZigTest,
        else => return err,
    };

    const allocator = testing.allocator;
    var tmp = testing.tmpDir(.{});
    defer tmp.cleanup();

    const zero_out = try pathFromTmpAlloc(allocator, tmp, "zero");
    defer allocator.free(zero_out);
    const zero_result = try runProcessWithLeanPath(allocator, &.{
        ORA_BINARY_REL,
        "emit",
        "--emit=smt-report,bytecode",
        "--out-dir",
        zero_out,
        "--lean-proofs",
        "tests/formal/scalar_loop_zero_iteration.ora",
    });
    defer allocator.free(zero_result.stdout);
    defer allocator.free(zero_result.stderr);
    try expectExited(zero_result, 0);
    try testing.expect(std.mem.containsAtLeast(u8, zero_result.stdout, 1, "certificate-required=1"));
    try testing.expect(std.mem.containsAtLeast(u8, zero_result.stdout, 1, "Lean loop proof gate accepted:\n  kernel-checked certificates: 1"));

    const zero_report_path = try pathFromTmpAlloc(
        allocator,
        tmp,
        "zero/scalar_loop_zero_iteration.smt.report.json",
    );
    defer allocator.free(zero_report_path);
    const zero_report = try readFileAllocForTest(allocator, zero_report_path);
    defer allocator.free(zero_report);
    try testing.expect(std.mem.containsAtLeast(u8, zero_report, 1, "\"kind\": \"loop_invariant_step\""));
    try testing.expect(!std.mem.containsAtLeast(u8, zero_report, 1, "\"vacuous\": true"));

    const zero_certificate_path = try pathFromTmpAlloc(
        allocator,
        tmp,
        "zero/scalar_loop_zero_iteration.lean.proof.json",
    );
    defer allocator.free(zero_certificate_path);
    const zero_certificate = try readFileAllocForTest(allocator, zero_certificate_path);
    defer allocator.free(zero_certificate);
    try testing.expect(std.mem.containsAtLeast(u8, zero_certificate, 1, "\"proof_count\": 1"));
    try testing.expect(std.mem.containsAtLeast(u8, zero_certificate, 1, "\"loop_vc_sha256\": \""));
    try testing.expect(!std.mem.containsAtLeast(u8, zero_certificate, 1, "sorryAx"));

    const zero_index_path = try pathFromTmpAlloc(
        allocator,
        tmp,
        "zero/scalar_loop_zero_iteration.formal.artifacts.json",
    );
    defer allocator.free(zero_index_path);
    const zero_index = try readFileAllocForTest(allocator, zero_index_path);
    defer allocator.free(zero_index);
    try testing.expect(std.mem.containsAtLeast(
        u8,
        zero_index,
        1,
        "\"final_artifact_authorization\": \"allowed\"",
    ));
    try testing.expect(std.mem.containsAtLeast(u8, zero_index, 1, "\"id\": \"lean_proof_certificate\""));
    try testing.expect(std.mem.containsAtLeast(u8, zero_index, 1, "\"sha256\": \""));
    try tmp.dir.access(std.testing.io, "zero/scalar_loop_zero_iteration.hex", .{});

    const negative_cases = [_]struct {
        stem: []const u8,
        source: []const u8,
        diagnostic: []const u8,
    }{
        .{
            .stem = "false_loop_invariant",
            .source =
            \\contract FalseLoopInvariant {
            \\    pub fn run(limit: u256) -> u256 {
            \\        var value: u256 = 0;
            \\        while (value < limit)
            \\            invariant value > limit
            \\        {
            \\            value = value + 1;
            \\        }
            \\        return value;
            \\    }
            \\}
            ,
            .diagnostic = "[invariant]",
        },
        .{
            .stem = "missing_loop_invariant",
            .source =
            \\contract MissingLoopInvariant {
            \\    pub fn run(limit: u256) -> u256 {
            \\        var value: u256 = 0;
            \\        while (value < limit) {
            \\            value = value + 1;
            \\        }
            \\        return value;
            \\    }
            \\}
            ,
            .diagnostic = "loop_invariant_missing",
        },
        .{
            .stem = "insufficient_loop_safety",
            .source =
            \\contract InsufficientLoopSafety {
            \\    pub fn run(limit: u256) -> u256 {
            \\        var value: u256 = 0;
            \\        while (value <= limit)
            \\            invariant value <= limit
            \\        {
            \\            value = value + 1;
            \\        }
            \\        return value;
            \\    }
            \\}
            ,
            .diagnostic = "checked addition overflow",
        },
    };

    for (negative_cases, 0..) |case, index| {
        const source_name = try std.fmt.allocPrint(allocator, "{s}.ora", .{case.stem});
        defer allocator.free(source_name);
        try tmp.dir.writeFile(std.testing.io, .{ .sub_path = source_name, .data = case.source });
        const source_path = try pathFromTmpAlloc(allocator, tmp, source_name);
        defer allocator.free(source_path);
        const output_sub_path = try std.fmt.allocPrint(allocator, "negative-{d}", .{index});
        defer allocator.free(output_sub_path);
        const output_path = try pathFromTmpAlloc(allocator, tmp, output_sub_path);
        defer allocator.free(output_path);

        const result = try runProcessWithLeanPath(allocator, &.{
            ORA_BINARY_REL,
            "emit",
            "--emit=smt-report,bytecode",
            "--out-dir",
            output_path,
            "--lean-proofs",
            source_path,
        });
        defer allocator.free(result.stdout);
        defer allocator.free(result.stderr);
        try expectExited(result, 1);
        try testing.expect(
            std.mem.containsAtLeast(u8, result.stdout, 1, case.diagnostic) or
                std.mem.containsAtLeast(u8, result.stderr, 1, case.diagnostic),
        );

        const bytecode_sub_path = try std.fmt.allocPrint(
            allocator,
            "{s}/{s}.hex",
            .{ output_sub_path, case.stem },
        );
        defer allocator.free(bytecode_sub_path);
        try testing.expectError(
            error.FileNotFound,
            tmp.dir.access(std.testing.io, bytecode_sub_path, .{}),
        );

        if (std.mem.eql(u8, case.stem, "missing_loop_invariant")) {
            const manifest_output_path = try pathFromTmpAlloc(allocator, tmp, "missing-invariant-manifest");
            defer allocator.free(manifest_output_path);
            const nonexistent_manifest = try pathFromTmpAlloc(allocator, tmp, "does-not-exist.json");
            defer allocator.free(nonexistent_manifest);
            const manifest_result = try runProcessWithLeanPath(allocator, &.{
                ORA_BINARY_REL,
                "emit",
                "--emit=smt-report,bytecode",
                "--out-dir",
                manifest_output_path,
                "--lean-proofs",
                nonexistent_manifest,
                source_path,
            });
            defer allocator.free(manifest_result.stdout);
            defer allocator.free(manifest_result.stderr);
            try expectExited(manifest_result, 1);
            try testing.expect(
                std.mem.containsAtLeast(u8, manifest_result.stdout, 1, "loop_invariant_missing") or
                    std.mem.containsAtLeast(u8, manifest_result.stderr, 1, "loop_invariant_missing"),
            );
            try testing.expect(!std.mem.containsAtLeast(u8, manifest_result.stdout, 1, "FileNotFound"));
            try testing.expect(!std.mem.containsAtLeast(u8, manifest_result.stderr, 1, "FileNotFound"));
            try testing.expectError(
                error.FileNotFound,
                tmp.dir.access(std.testing.io, "missing-invariant-manifest/missing_loop_invariant.hex", .{}),
            );
        }
    }
}

test "unsupported automatic loop fragment is named and points to proof manifest" {
    std.Io.Dir.cwd().access(std.testing.io, ORA_BINARY_REL, .{}) catch |err| switch (err) {
        error.FileNotFound => return error.SkipZigTest,
        else => return err,
    };

    const allocator = testing.allocator;
    var tmp = testing.tmpDir(.{});
    defer tmp.cleanup();
    const output_path = try pathFromTmpAlloc(allocator, tmp, "automatic-manual");
    defer allocator.free(output_path);

    const result = try runProcessWithLeanPath(allocator, &.{
        ORA_BINARY_REL,
        "emit",
        "--emit=smt-report,bytecode",
        "--out-dir",
        output_path,
        "--lean-proofs",
        "tests/formal/scalar_loop_manual_proof_required.ora",
    });
    defer allocator.free(result.stdout);
    defer allocator.free(result.stderr);
    try expectExited(result, 1);
    try testing.expect(std.mem.containsAtLeast(
        u8,
        result.stdout,
        1,
        "Automatic Lean loop proof unavailable: automatic_nonlinear_arithmetic",
    ));
    try testing.expect(std.mem.containsAtLeast(
        u8,
        result.stdout,
        1,
        "pass the proof manifest with `--lean-proofs <proofs.json>`",
    ));
    try testing.expect(std.mem.containsAtLeast(u8, result.stdout, 1, "inline_source=true"));
    try testing.expect(!std.mem.containsAtLeast(u8, result.stdout, 1, "unexpected token"));
    try testing.expect(!std.mem.containsAtLeast(u8, result.stderr, 1, "unexpected token"));
}

test "external user proof certifies nonlinear loop through the real CLI" {
    std.Io.Dir.cwd().access(std.testing.io, ORA_BINARY_REL, .{}) catch |err| switch (err) {
        error.FileNotFound => return error.SkipZigTest,
        else => return err,
    };

    const allocator = testing.allocator;
    const source_path = "tests/formal/scalar_loop_manual_proof_required.ora";
    const manifest_path = "tests/formal/scalar_loop_manual_proof_required.proofs.json";
    const theorem_name = "Ora.UserProof.ScalarLoopManual.Query_12.discharge_q12";
    var tmp = testing.tmpDir(.{});
    defer tmp.cleanup();

    const valid_out = try pathFromTmpAlloc(allocator, tmp, "external-valid");
    defer allocator.free(valid_out);
    const valid = try runProcessWithLeanPath(allocator, &.{
        ORA_BINARY_REL,
        "emit",
        "--emit=smt-report,bytecode",
        "--out-dir",
        valid_out,
        "--lean-proofs",
        manifest_path,
        source_path,
    });
    defer allocator.free(valid.stdout);
    defer allocator.free(valid.stderr);
    try expectExited(valid, 0);
    try testing.expect(!std.mem.containsAtLeast(u8, valid.stdout, 1, "Automatic Lean loop proof unavailable"));
    try testing.expect(!std.mem.containsAtLeast(u8, valid.stderr, 1, "AutoProofScratch"));

    const certificate_path = try pathFromTmpAlloc(
        allocator,
        tmp,
        "external-valid/scalar_loop_manual_proof_required.lean.proof.json",
    );
    defer allocator.free(certificate_path);
    const certificate = try readFileAllocForTest(allocator, certificate_path);
    defer allocator.free(certificate);
    try testing.expect(std.mem.containsAtLeast(u8, certificate, 1, "\"proof_count\": 1"));
    try testing.expect(std.mem.containsAtLeast(u8, certificate, 1, theorem_name));
    try testing.expect(std.mem.containsAtLeast(u8, certificate, 1, "\"inline_source\": true"));
    try testing.expect(std.mem.containsAtLeast(u8, certificate, 1, "\"content_sha256\": \""));
    try testing.expect(std.mem.containsAtLeast(u8, certificate, 1, "\"loop_vc_sha256\": \""));
    try testing.expect(!std.mem.containsAtLeast(u8, certificate, 1, "sorryAx"));
    try testing.expect(!std.mem.containsAtLeast(u8, certificate, 1, "AutoProofScratch"));
    try testing.expect(!std.mem.containsAtLeast(u8, certificate, 1, "CompilerAutomaticProof"));

    const bytecode_path = try pathFromTmpAlloc(
        allocator,
        tmp,
        "external-valid/scalar_loop_manual_proof_required.hex",
    );
    defer allocator.free(bytecode_path);
    const bytecode = try readFileAllocForTest(allocator, bytecode_path);
    defer allocator.free(bytecode);
    try testing.expect(bytecode.len != 0);

    const stale_manifest_path = try pathFromTmpAlloc(allocator, tmp, "stale.proofs.json");
    defer allocator.free(stale_manifest_path);
    var stale_manifest = std.Io.Writer.Allocating.init(allocator);
    defer stale_manifest.deinit();
    try stale_manifest.writer.writeAll(
        "{\n  \"schema_version\": 1,\n  \"proofs\": [\n    {\n" ++
            "      \"query_id\": 12,\n      \"obligation_ids\": [],\n" ++
            "      \"assumption_ids\": [],\n      \"module_name\": \"Ora.UserProof.ScalarLoopManual\",\n" ++
            "      \"theorem_name\": \"Ora.UserProof.ScalarLoopManual.Query_12.discharge_q12\",\n" ++
            "      \"path\": \"tests/formal/scalar_loop_manual_proof_required.lean\",\n" ++
            "      \"content_sha256\": \"0000000000000000000000000000000000000000000000000000000000000000\",\n" ++
            "      \"inline_source\": true\n    }\n  ]\n}\n",
    );
    try std.Io.Dir.cwd().writeFile(std.testing.io, .{
        .sub_path = stale_manifest_path,
        .data = stale_manifest.written(),
    });

    const stale_out = try pathFromTmpAlloc(allocator, tmp, "external-stale");
    defer allocator.free(stale_out);
    const stale = try runProcessWithLeanPath(allocator, &.{
        ORA_BINARY_REL,
        "emit",
        "--emit=smt-report,bytecode",
        "--out-dir",
        stale_out,
        "--lean-proofs",
        stale_manifest_path,
        source_path,
    });
    defer allocator.free(stale.stdout);
    defer allocator.free(stale.stderr);
    try expectExited(stale, 1);
    try testing.expect(
        std.mem.containsAtLeast(u8, stale.stdout, 1, "ProofRowDigestMismatch") or
            std.mem.containsAtLeast(u8, stale.stderr, 1, "ProofRowDigestMismatch"),
    );
    try testing.expectError(
        error.FileNotFound,
        tmp.dir.access(
            std.testing.io,
            "external-stale/scalar_loop_manual_proof_required.hex",
            .{},
        ),
    );
}

test "L4 scalar loop induction proof unblocks only artifacts" {
    const allocator = testing.allocator;
    const source_path = "tests/formal/scalar_loop_post.ora";
    var compilation = try compilePackage(source_path);
    defer compilation.deinit();
    const hir_result = try compilation.db.lowerToHir(compilation.root_module_id);
    try testing.expect(hir_result.diagnostics.isEmpty());
    var formal_result = try obligation_from_mlir.collect(allocator, hir_result.module.raw_module, .{});
    defer formal_result.deinit();
    const induction_query = try findLoopInductionQuery(formal_result.set);
    const summary_id = induction_query.loop_summary_id orelse return error.TestUnexpectedResult;
    const summary = obligation.findById(formal_result.set.loop_summaries, summary_id) orelse
        return error.TestUnexpectedResult;
    try testing.expectEqual(@as(usize, 1), summary.query_ids.step.len);
    var pass = try z3_verification.VerificationPass.init(allocator);
    defer pass.deinit();
    const z3_bindings = try test_helpers.z3FormalQueryBindingsFromFormalForTest(
        allocator,
        formal_result.query_bindings,
    );
    defer allocator.free(z3_bindings);
    pass.setFormalQueryBindings(z3_bindings);
    var prepared_manifest = try pass.collectPreparedQueryManifest(hir_result.module.raw_module);
    defer prepared_manifest.deinit();
    var prepared_step_index: ?usize = null;
    for (prepared_manifest.rows, 0..) |row, index| {
        if (row.kind != .LoopInvariantStep) continue;
        if (prepared_step_index != null) return error.TestUnexpectedResult;
        prepared_step_index = index;
    }
    const forced_step_query = try std.fmt.allocPrint(
        allocator,
        "loop_invariant_step:{d}",
        .{(prepared_step_index orelse return error.TestUnexpectedResult) + 1},
    );
    defer allocator.free(forced_step_query);
    var tmp = testing.tmpDir(.{});
    defer tmp.cleanup();

    const fail_out = try pathFromTmpAlloc(allocator, tmp, "fail");
    defer allocator.free(fail_out);
    const valid_out = try pathFromTmpAlloc(allocator, tmp, "valid");
    defer allocator.free(valid_out);
    const reference_out = try pathFromTmpAlloc(allocator, tmp, "reference");
    defer allocator.free(reference_out);
    const vacuous_out = try pathFromTmpAlloc(allocator, tmp, "vacuous");
    defer allocator.free(vacuous_out);

    {
        const result = try runOraWithForcedUnknown(allocator, forced_step_query, &.{
            ORA_BINARY_REL,
            "emit",
            "--emit=smt-report,sir-text,bytecode",
            "--out-dir",
            fail_out,
            source_path,
        });
        defer allocator.free(result.stdout);
        defer allocator.free(result.stderr);
        try expectExited(result, 1);
        try testing.expect(std.mem.containsAtLeast(u8, result.stdout, 1, "Lean proof recipe for required userland obligations") or
            std.mem.containsAtLeast(u8, result.stderr, 1, "Lean proof recipe for required userland obligations"));
    }
    try testing.expectError(error.FileNotFound, tmp.dir.access(std.testing.io, "fail/scalar_loop_post.hex", .{}));

    {
        const result = try runProcessWithLeanPath(allocator, &.{
            ORA_BINARY_REL,
            "emit",
            "--emit=smt-report,sir-text,bytecode",
            "--out-dir",
            valid_out,
            "--lean-proofs",
            source_path,
        });
        defer allocator.free(result.stdout);
        defer allocator.free(result.stderr);
        try expectExited(result, 0);
    }

    const valid_cert_path = try pathFromTmpAlloc(allocator, tmp, "valid/scalar_loop_post.lean.proof.json");
    defer allocator.free(valid_cert_path);
    const valid_sir_path = try pathFromTmpAlloc(allocator, tmp, "valid/scalar_loop_post.sir");
    defer allocator.free(valid_sir_path);
    const valid_hex_path = try pathFromTmpAlloc(allocator, tmp, "valid/scalar_loop_post.hex");
    defer allocator.free(valid_hex_path);
    const valid_cert = try readFileAllocForTest(allocator, valid_cert_path);
    defer allocator.free(valid_cert);
    try testing.expect(std.mem.containsAtLeast(u8, valid_cert, 1, "\"proof_count\": 1"));
    try testing.expect(!std.mem.containsAtLeast(u8, valid_cert, 1, "sorryAx"));
    const valid_sir = try readFileAllocForTest(allocator, valid_sir_path);
    defer allocator.free(valid_sir);
    const valid_hex = try readFileAllocForTest(allocator, valid_hex_path);
    defer allocator.free(valid_hex);
    try testing.expect(std.mem.containsAtLeast(u8, valid_sir, 1, "fn guarded:"));
    try testing.expect(std.mem.containsAtLeast(u8, valid_sir, 1, "revert 0x0 0x0"));

    {
        const result = try runProcessWithLeanPath(allocator, &.{
            ORA_BINARY_REL,
            "emit",
            "--emit=sir-text,bytecode",
            "--out-dir",
            reference_out,
            "--lean-proofs",
            source_path,
        });
        defer allocator.free(result.stdout);
        defer allocator.free(result.stderr);
        try expectExited(result, 0);
    }
    const reference_sir_path = try pathFromTmpAlloc(allocator, tmp, "reference/scalar_loop_post.sir");
    defer allocator.free(reference_sir_path);
    const reference_hex_path = try pathFromTmpAlloc(allocator, tmp, "reference/scalar_loop_post.hex");
    defer allocator.free(reference_hex_path);
    const reference_sir = try readFileAllocForTest(allocator, reference_sir_path);
    defer allocator.free(reference_sir);
    const reference_hex = try readFileAllocForTest(allocator, reference_hex_path);
    defer allocator.free(reference_hex);
    try testing.expectEqualStrings(reference_sir, valid_sir);
    try testing.expectEqualStrings(reference_hex, valid_hex);

    // Contradictory source assumptions are rejected before any Lean proof can
    // authorize artifacts. The loop proposition independently carries a
    // concrete initial-state witness, so its induction proof surface cannot
    // be implication-vacuous either.
    {
        const result = try runProcess(allocator, &.{
            ORA_BINARY_REL,
            "emit",
            "--emit=smt-report,sir-text,bytecode",
            "--out-dir",
            vacuous_out,
            "tests/formal/scalar_loop_post_vacuous.ora",
        });
        defer allocator.free(result.stdout);
        defer allocator.free(result.stderr);
        try expectExited(result, 1);
        try testing.expect(std.mem.containsAtLeast(u8, result.stdout, 1, "vacuous") or
            std.mem.containsAtLeast(u8, result.stderr, 1, "assumptions inconsistent"));
    }
    try testing.expectError(
        error.FileNotFound,
        tmp.dir.access(std.testing.io, "vacuous/scalar_loop_post_vacuous.hex", .{}),
    );
}

test "B3 lean proofs unblock source-level unknown without erasing runtime guard" {
    std.Io.Dir.cwd().access(std.testing.io, ORA_BINARY_REL, .{}) catch |err| switch (err) {
        error.FileNotFound => return error.SkipZigTest,
        else => return err,
    };

    const allocator = testing.allocator;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();

    const source =
        \\contract B3LeanProofGate {
        \\    pub fn bounded(x: u256) -> bool
        \\        requires x == x
        \\        ensures (forall i: u256 where i < x => i <= x)
        \\    {
        \\        return true;
        \\    }
        \\
        \\    pub fn guarded(y: u256) -> bool
        \\        guard y > 0
        \\    {
        \\        return true;
        \\    }
        \\}
    ;
    try tmp.dir.writeFile(std.testing.io, .{ .sub_path = "b3_lean_gate.ora", .data = source });

    const source_path = try pathFromTmpAlloc(allocator, tmp, "b3_lean_gate.ora");
    defer allocator.free(source_path);
    const fail_out = try pathFromTmpAlloc(allocator, tmp, "fail");
    defer allocator.free(fail_out);
    const valid_out = try pathFromTmpAlloc(allocator, tmp, "valid");
    defer allocator.free(valid_out);
    const sorry_out = try pathFromTmpAlloc(allocator, tmp, "sorry");
    defer allocator.free(sorry_out);
    const reference_out = try pathFromTmpAlloc(allocator, tmp, "reference");
    defer allocator.free(reference_out);

    var formal_result = try collectPackageObligations(allocator, source_path);
    defer formal_result.deinit();
    const ensures_query = try findEnsuresQuery(formal_result.set);
    const generated_namespace = try leanProofGeneratedNamespaceForTest(allocator, source_path);
    defer allocator.free(generated_namespace);
    var obligations_source_out = std.Io.Writer.Allocating.init(allocator);
    defer obligations_source_out.deinit();
    try obligation_to_lean.writeModule(&obligations_source_out.writer, formal_result.set, .{
        .namespace = generated_namespace,
        .proof_surface = true,
    });
    const obligations_source = obligations_source_out.written();

    const expected_query_text = try std.fmt.allocPrint(allocator, "query: emittedQuery_{d}", .{ensures_query.id});
    defer allocator.free(expected_query_text);
    const expected_theorem_text = try std.fmt.allocPrint(allocator, "theorem discharge_q{d}", .{ensures_query.id});
    defer allocator.free(expected_theorem_text);
    const expected_query_def = try std.fmt.allocPrint(allocator, "def emittedQuery_{d} : Prop :=", .{ensures_query.id});
    defer allocator.free(expected_query_def);

    {
        const result = try runOraWithForcedUnknown(allocator, "obligation:2", &.{
            ORA_BINARY_REL,
            "emit",
            "--emit=smt-report,sir-text,bytecode",
            "--out-dir",
            fail_out,
            source_path,
        });
        defer allocator.free(result.stdout);
        defer allocator.free(result.stderr);
        try expectExited(result, 1);
        try testing.expect(std.mem.containsAtLeast(u8, result.stdout, 1, "could not prove ensures") or
            std.mem.containsAtLeast(u8, result.stderr, 1, "could not prove ensures"));
        try testing.expect(std.mem.containsAtLeast(u8, result.stdout, 1, "Lean proof recipe for required userland obligations"));
        try testing.expect(std.mem.containsAtLeast(u8, result.stdout, 1, expected_query_text));
        try testing.expect(std.mem.containsAtLeast(u8, result.stdout, 1, expected_theorem_text));
        try testing.expect(std.mem.containsAtLeast(u8, result.stdout, 1, generated_namespace));
        try testing.expect(std.mem.containsAtLeast(u8, result.stdout, 1, "--lean-proofs <proofs.json>"));
    }
    const fail_report_path = try pathFromTmpAlloc(allocator, tmp, "fail/b3_lean_gate.smt.report.json");
    defer allocator.free(fail_report_path);
    const fail_report = try readFileAllocForTest(allocator, fail_report_path);
    defer allocator.free(fail_report);
    try testing.expect(std.mem.containsAtLeast(u8, fail_report, 1, "\"status\": \"UNKNOWN\""));
    try testing.expect(std.mem.containsAtLeast(u8, fail_report, 1, "\"vacuity_unknown\": false"));
    const fail_lean_obligations_path = try pathFromTmpAlloc(allocator, tmp, "fail/b3_lean_gate.lean.obligations.lean");
    defer allocator.free(fail_lean_obligations_path);
    const fail_lean_obligations = try readFileAllocForTest(allocator, fail_lean_obligations_path);
    defer allocator.free(fail_lean_obligations);
    try testing.expect(std.mem.containsAtLeast(u8, fail_lean_obligations, 1, expected_query_def));
    try testing.expect(std.mem.containsAtLeast(u8, fail_lean_obligations, 1, generated_namespace));
    {
        const result = try runLeanFileForTest(allocator, fail_lean_obligations_path);
        defer allocator.free(result.stdout);
        defer allocator.free(result.stderr);
        try expectExited(result, 0);
    }
    try testing.expectError(error.FileNotFound, tmp.dir.access(std.testing.io, "fail/b3_lean_gate.hex", .{}));

    const module_suffix = try moduleSuffixFromTmp(allocator, tmp);
    defer allocator.free(module_suffix);
    const fixture_dir = try std.fmt.allocPrint(allocator, "formal/Ora/B3Fixture/{s}", .{module_suffix});
    defer allocator.free(fixture_dir);
    try std.Io.Dir.cwd().createDirPath(std.testing.io, fixture_dir);
    defer std.Io.Dir.cwd().deleteTree(std.testing.io, fixture_dir) catch {};
    defer std.Io.Dir.cwd().deleteDir(std.testing.io, "formal/Ora/B3Fixture") catch {};

    const valid_module = try std.fmt.allocPrint(allocator, "Ora.B3Fixture.{s}.Valid", .{module_suffix});
    defer allocator.free(valid_module);
    const sorry_module = try std.fmt.allocPrint(allocator, "Ora.B3Fixture.{s}.Sorry", .{module_suffix});
    defer allocator.free(sorry_module);
    const valid_theorem = try std.fmt.allocPrint(allocator, "{s}.discharge", .{valid_module});
    defer allocator.free(valid_theorem);
    const sorry_theorem = try std.fmt.allocPrint(allocator, "{s}.discharge", .{sorry_module});
    defer allocator.free(sorry_theorem);
    const valid_proof_path = try std.fmt.allocPrint(allocator, "{s}/Valid.lean", .{fixture_dir});
    defer allocator.free(valid_proof_path);
    const sorry_proof_path = try std.fmt.allocPrint(allocator, "{s}/Sorry.lean", .{fixture_dir});
    defer allocator.free(sorry_proof_path);

    const valid_proof = try proofModuleFromGeneratedObligations(allocator, obligations_source, valid_module, ensures_query.id, false);
    defer allocator.free(valid_proof);
    const sorry_proof = try proofModuleFromGeneratedObligations(allocator, obligations_source, sorry_module, ensures_query.id, true);
    defer allocator.free(sorry_proof);
    try std.Io.Dir.cwd().writeFile(std.testing.io, .{ .sub_path = valid_proof_path, .data = valid_proof });
    try std.Io.Dir.cwd().writeFile(std.testing.io, .{ .sub_path = sorry_proof_path, .data = sorry_proof });

    const valid_manifest = try pathFromTmpAlloc(allocator, tmp, "valid-proofs.json");
    defer allocator.free(valid_manifest);
    const sorry_manifest = try pathFromTmpAlloc(allocator, tmp, "sorry-proofs.json");
    defer allocator.free(sorry_manifest);
    try writeProofManifestForTest(allocator, valid_manifest, valid_module, valid_theorem, valid_proof_path, ensures_query);
    try writeProofManifestForTest(allocator, sorry_manifest, sorry_module, sorry_theorem, sorry_proof_path, ensures_query);

    {
        const result = try runOraWithForcedUnknown(allocator, "obligation:2", &.{
            ORA_BINARY_REL,
            "emit",
            "--emit=smt-report,sir-text,bytecode",
            "--out-dir",
            valid_out,
            "--lean-proofs",
            valid_manifest,
            source_path,
        });
        defer allocator.free(result.stdout);
        defer allocator.free(result.stderr);
        try expectExited(result, 0);
    }

    const valid_cert_path = try pathFromTmpAlloc(allocator, tmp, "valid/b3_lean_gate.lean.proof.json");
    defer allocator.free(valid_cert_path);
    const valid_sir_path = try pathFromTmpAlloc(allocator, tmp, "valid/b3_lean_gate.sir");
    defer allocator.free(valid_sir_path);
    const valid_hex_path = try pathFromTmpAlloc(allocator, tmp, "valid/b3_lean_gate.hex");
    defer allocator.free(valid_hex_path);
    const valid_cert = try readFileAllocForTest(allocator, valid_cert_path);
    defer allocator.free(valid_cert);
    try testing.expect(std.mem.containsAtLeast(u8, valid_cert, 1, "\"schema_version\": 3"));
    try testing.expect(std.mem.containsAtLeast(u8, valid_cert, 1, "\"proof_count\": 1"));
    try testing.expect(std.mem.containsAtLeast(u8, valid_cert, 1, "\"axioms\""));
    try testing.expect(!std.mem.containsAtLeast(u8, valid_cert, 1, "sorryAx"));

    const valid_sir = try readFileAllocForTest(allocator, valid_sir_path);
    defer allocator.free(valid_sir);
    try testing.expect(std.mem.containsAtLeast(u8, valid_sir, 1, "fn guarded:"));
    try testing.expect(std.mem.containsAtLeast(u8, valid_sir, 1, "c0 = const 0x0"));
    try testing.expect(std.mem.containsAtLeast(u8, valid_sir, 1, "gt v0 c0"));
    try testing.expect(std.mem.containsAtLeast(u8, valid_sir, 1, "revert 0x0 0x0"));
    const valid_hex = try readFileAllocForTest(allocator, valid_hex_path);
    defer allocator.free(valid_hex);
    try testing.expect(valid_hex.len > 0);

    {
        const result = try runProcess(allocator, &.{
            ORA_BINARY_REL,
            "emit",
            "--emit=sir-text,bytecode",
            "--out-dir",
            reference_out,
            source_path,
        });
        defer allocator.free(result.stdout);
        defer allocator.free(result.stderr);
        try expectExited(result, 0);
    }
    const reference_sir_path = try pathFromTmpAlloc(allocator, tmp, "reference/b3_lean_gate.sir");
    defer allocator.free(reference_sir_path);
    const reference_hex_path = try pathFromTmpAlloc(allocator, tmp, "reference/b3_lean_gate.hex");
    defer allocator.free(reference_hex_path);
    const reference_sir = try readFileAllocForTest(allocator, reference_sir_path);
    defer allocator.free(reference_sir);
    const reference_hex = try readFileAllocForTest(allocator, reference_hex_path);
    defer allocator.free(reference_hex);
    try testing.expectEqualStrings(reference_sir, valid_sir);
    try testing.expectEqualStrings(reference_hex, valid_hex);

    {
        const result = try runOraWithForcedUnknown(allocator, "obligation:2", &.{
            ORA_BINARY_REL,
            "emit",
            "--emit=smt-report,sir-text,bytecode",
            "--out-dir",
            sorry_out,
            "--lean-proofs",
            sorry_manifest,
            source_path,
        });
        defer allocator.free(result.stdout);
        defer allocator.free(result.stderr);
        try expectExited(result, 1);
        try testing.expect(std.mem.containsAtLeast(u8, result.stdout, 1, "sorryAx") or
            std.mem.containsAtLeast(u8, result.stderr, 1, "sorryAx") or
            std.mem.containsAtLeast(u8, result.stdout, 1, "Lean proof gate failed") or
            std.mem.containsAtLeast(u8, result.stderr, 1, "Lean proof gate failed"));
    }
    try testing.expectError(error.FileNotFound, tmp.dir.access(std.testing.io, "sorry/b3_lean_gate.hex", .{}));
    try testing.expectError(error.FileNotFound, tmp.dir.access(std.testing.io, "sorry/b3_lean_gate.lean.proof.json", .{}));
}

test "automatic Lean proofs unblock unknown without runtime-check authority" {
    std.Io.Dir.cwd().access(std.testing.io, ORA_BINARY_REL, .{}) catch |err| switch (err) {
        error.FileNotFound => return error.SkipZigTest,
        else => return err,
    };

    const allocator = testing.allocator;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const source =
        \\contract AutomaticLeanBoundary {
        \\    pub fn bounded(x: u256) -> bool
        \\        requires x == x
        \\        ensures (forall i: u256 where i < x => i <= x)
        \\    {
        \\        return true;
        \\    }
        \\
        \\    pub fn guarded(y: u256) -> bool
        \\        guard y > 0
        \\    {
        \\        return true;
        \\    }
        \\}
    ;
    try tmp.dir.writeFile(std.testing.io, .{ .sub_path = "automatic_lean_boundary.ora", .data = source });
    const source_path = try pathFromTmpAlloc(allocator, tmp, "automatic_lean_boundary.ora");
    defer allocator.free(source_path);
    const out_dir = try pathFromTmpAlloc(allocator, tmp, "out");
    defer allocator.free(out_dir);

    const result = try runOraWithForcedUnknown(allocator, "obligation:2", &.{
        ORA_BINARY_REL,
        "emit",
        "--emit=smt-report,sir-text,bytecode",
        "--out-dir",
        out_dir,
        "--lean-proofs",
        source_path,
    });
    defer allocator.free(result.stdout);
    defer allocator.free(result.stderr);
    try expectExited(result, 0);

    const certificate_path = try pathFromTmpAlloc(allocator, tmp, "out/automatic_lean_boundary.lean.proof.json");
    defer allocator.free(certificate_path);
    const certificate = try readFileAllocForTest(allocator, certificate_path);
    defer allocator.free(certificate);
    try testing.expect(std.mem.containsAtLeast(u8, certificate, 1, "\"schema_version\": 3"));
    var certificate_digest: [std.crypto.hash.sha2.Sha256.digest_length]u8 = undefined;
    std.crypto.hash.sha2.Sha256.hash(certificate, &certificate_digest, .{});
    const certificate_hash = try std.fmt.allocPrint(allocator, "{x}", .{certificate_digest});
    defer allocator.free(certificate_hash);
    const index_path = try pathFromTmpAlloc(allocator, tmp, "out/automatic_lean_boundary.formal.artifacts.json");
    defer allocator.free(index_path);
    const artifact_index = try readFileAllocForTest(allocator, index_path);
    defer allocator.free(artifact_index);
    try testing.expect(std.mem.containsAtLeast(u8, artifact_index, 1, "\"final_artifact_authorization\": \"allowed\""));
    try testing.expect(std.mem.containsAtLeast(u8, artifact_index, 1, "\"status\": \"rejected\""));
    try testing.expect(std.mem.containsAtLeast(u8, artifact_index, 2, "\"status\": \"accepted\""));
    try testing.expect(std.mem.containsAtLeast(u8, artifact_index, 1, "\"path\": \"automatic_lean_boundary.lean.proof.json\""));
    try testing.expect(std.mem.containsAtLeast(u8, artifact_index, 1, certificate_hash));

    const sir_path = try pathFromTmpAlloc(allocator, tmp, "out/automatic_lean_boundary.sir");
    defer allocator.free(sir_path);
    const sir = try readFileAllocForTest(allocator, sir_path);
    defer allocator.free(sir);
    try testing.expect(std.mem.containsAtLeast(u8, sir, 1, "fn guarded:"));
    try testing.expect(std.mem.containsAtLeast(u8, sir, 1, "gt v0 c0"));
    try testing.expect(std.mem.containsAtLeast(u8, sir, 1, "revert 0x0 0x0"));
}

test "B6 storage Lean proof fixture proves read-only old collapse without erasing guard" {
    std.Io.Dir.cwd().access(std.testing.io, ORA_BINARY_REL, .{}) catch |err| switch (err) {
        error.FileNotFound => return error.SkipZigTest,
        else => return err,
    };

    const allocator = testing.allocator;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();

    const source =
        \\contract B6StorageLeanProofGate {
        \\    storage var balance: u256;
        \\
        \\    pub fn same_balance() -> bool
        \\        ensures balance == old(balance)
        \\    {
        \\        return true;
        \\    }
        \\
        \\    pub fn guarded(y: u256) -> bool
        \\        guard y > 0
        \\    {
        \\        return true;
        \\    }
        \\}
    ;
    try tmp.dir.writeFile(std.testing.io, .{ .sub_path = "b6_storage_lean_gate.ora", .data = source });

    const source_path = try pathFromTmpAlloc(allocator, tmp, "b6_storage_lean_gate.ora");
    defer allocator.free(source_path);
    const fail_out = try pathFromTmpAlloc(allocator, tmp, "fail");
    defer allocator.free(fail_out);
    const valid_out = try pathFromTmpAlloc(allocator, tmp, "valid");
    defer allocator.free(valid_out);
    const sorry_out = try pathFromTmpAlloc(allocator, tmp, "sorry");
    defer allocator.free(sorry_out);
    const reference_out = try pathFromTmpAlloc(allocator, tmp, "reference");
    defer allocator.free(reference_out);

    var formal_result = try collectPackageObligations(allocator, source_path);
    defer formal_result.deinit();
    const ensures_query = try findEnsuresQuery(formal_result.set);
    try expectEnsuresQuerySupported(formal_result.set);

    const forced_query = "obligation:4";
    const generated_namespace = try leanProofGeneratedNamespaceForTest(allocator, source_path);
    defer allocator.free(generated_namespace);
    var obligations_source_out = std.Io.Writer.Allocating.init(allocator);
    defer obligations_source_out.deinit();
    try obligation_to_lean.writeModule(&obligations_source_out.writer, formal_result.set, .{
        .namespace = generated_namespace,
        .proof_surface = true,
    });
    const obligations_source = obligations_source_out.written();
    try testing.expect(std.mem.containsAtLeast(u8, obligations_source, 2, ".placeRead { root := \"balance\", region := .storage"));
    try testing.expect(std.mem.indexOf(u8, obligations_source, ".old ") == null);

    const expected_query_def = try std.fmt.allocPrint(allocator, "def emittedQuery_{d} : Prop :=", .{ensures_query.id});
    defer allocator.free(expected_query_def);

    {
        const result = try runOraWithForcedUnknown(allocator, forced_query, &.{
            ORA_BINARY_REL,
            "emit",
            "--emit=smt-report,sir-text,bytecode",
            "--out-dir",
            fail_out,
            source_path,
        });
        defer allocator.free(result.stdout);
        defer allocator.free(result.stderr);
        try expectExited(result, 1);
        try testing.expect(std.mem.containsAtLeast(u8, result.stdout, 1, "Lean proof recipe for required userland obligations") or
            std.mem.containsAtLeast(u8, result.stderr, 1, "Lean proof recipe for required userland obligations"));
        try testing.expect(std.mem.containsAtLeast(u8, result.stdout, 1, "query: emittedQuery_") or
            std.mem.containsAtLeast(u8, result.stderr, 1, "query: emittedQuery_"));
        try testing.expect(std.mem.containsAtLeast(u8, result.stdout, 1, "theorem discharge_q") or
            std.mem.containsAtLeast(u8, result.stderr, 1, "theorem discharge_q"));
        try testing.expect(std.mem.containsAtLeast(u8, result.stdout, 1, generated_namespace) or
            std.mem.containsAtLeast(u8, result.stderr, 1, generated_namespace));
    }
    const fail_report_path = try pathFromTmpAlloc(allocator, tmp, "fail/b6_storage_lean_gate.smt.report.json");
    defer allocator.free(fail_report_path);
    const fail_report = try readFileAllocForTest(allocator, fail_report_path);
    defer allocator.free(fail_report);
    try testing.expect(std.mem.containsAtLeast(u8, fail_report, 1, "\"status\": \"UNKNOWN\""));
    try testing.expect(std.mem.containsAtLeast(u8, fail_report, 1, "\"vacuity_unknown\": false"));
    const fail_lean_obligations_path = try pathFromTmpAlloc(allocator, tmp, "fail/b6_storage_lean_gate.lean.obligations.lean");
    defer allocator.free(fail_lean_obligations_path);
    const fail_lean_obligations = try readFileAllocForTest(allocator, fail_lean_obligations_path);
    defer allocator.free(fail_lean_obligations);
    try testing.expect(std.mem.containsAtLeast(u8, fail_lean_obligations, 1, expected_query_def));
    try testing.expect(std.mem.containsAtLeast(u8, fail_lean_obligations, 2, ".placeRead { root := \"balance\", region := .storage"));
    {
        const result = try runLeanFileForTest(allocator, fail_lean_obligations_path);
        defer allocator.free(result.stdout);
        defer allocator.free(result.stderr);
        try expectExited(result, 0);
    }
    try testing.expectError(error.FileNotFound, tmp.dir.access(std.testing.io, "fail/b6_storage_lean_gate.hex", .{}));

    const module_suffix = try moduleSuffixFromTmp(allocator, tmp);
    defer allocator.free(module_suffix);
    const fixture_dir = try std.fmt.allocPrint(allocator, "formal/Ora/B6StorageFixture/{s}", .{module_suffix});
    defer allocator.free(fixture_dir);
    try std.Io.Dir.cwd().createDirPath(std.testing.io, fixture_dir);
    defer std.Io.Dir.cwd().deleteTree(std.testing.io, fixture_dir) catch {};
    defer std.Io.Dir.cwd().deleteDir(std.testing.io, "formal/Ora/B6StorageFixture") catch {};

    const valid_module = try std.fmt.allocPrint(allocator, "Ora.B6StorageFixture.{s}.Valid", .{module_suffix});
    defer allocator.free(valid_module);
    const sorry_module = try std.fmt.allocPrint(allocator, "Ora.B6StorageFixture.{s}.Sorry", .{module_suffix});
    defer allocator.free(sorry_module);
    const valid_theorem = try std.fmt.allocPrint(allocator, "{s}.discharge", .{valid_module});
    defer allocator.free(valid_theorem);
    const sorry_theorem = try std.fmt.allocPrint(allocator, "{s}.discharge", .{sorry_module});
    defer allocator.free(sorry_theorem);
    const valid_proof_path = try std.fmt.allocPrint(allocator, "{s}/Valid.lean", .{fixture_dir});
    defer allocator.free(valid_proof_path);
    const sorry_proof_path = try std.fmt.allocPrint(allocator, "{s}/Sorry.lean", .{fixture_dir});
    defer allocator.free(sorry_proof_path);

    const valid_proof = try storageProofModuleFromGeneratedObligations(allocator, obligations_source, valid_module, ensures_query.id, false);
    defer allocator.free(valid_proof);
    const sorry_proof = try storageProofModuleFromGeneratedObligations(allocator, obligations_source, sorry_module, ensures_query.id, true);
    defer allocator.free(sorry_proof);
    try std.Io.Dir.cwd().writeFile(std.testing.io, .{ .sub_path = valid_proof_path, .data = valid_proof });
    try std.Io.Dir.cwd().writeFile(std.testing.io, .{ .sub_path = sorry_proof_path, .data = sorry_proof });

    const valid_manifest = try pathFromTmpAlloc(allocator, tmp, "valid-proofs.json");
    defer allocator.free(valid_manifest);
    const sorry_manifest = try pathFromTmpAlloc(allocator, tmp, "sorry-proofs.json");
    defer allocator.free(sorry_manifest);
    try writeProofManifestForTest(allocator, valid_manifest, valid_module, valid_theorem, valid_proof_path, ensures_query);
    try writeProofManifestForTest(allocator, sorry_manifest, sorry_module, sorry_theorem, sorry_proof_path, ensures_query);

    {
        const result = try runOraWithForcedUnknown(allocator, forced_query, &.{
            ORA_BINARY_REL,
            "emit",
            "--emit=smt-report,sir-text,bytecode",
            "--out-dir",
            valid_out,
            "--lean-proofs",
            valid_manifest,
            source_path,
        });
        defer allocator.free(result.stdout);
        defer allocator.free(result.stderr);
        try expectExited(result, 0);
    }

    const valid_cert_path = try pathFromTmpAlloc(allocator, tmp, "valid/b6_storage_lean_gate.lean.proof.json");
    defer allocator.free(valid_cert_path);
    const valid_sir_path = try pathFromTmpAlloc(allocator, tmp, "valid/b6_storage_lean_gate.sir");
    defer allocator.free(valid_sir_path);
    const valid_hex_path = try pathFromTmpAlloc(allocator, tmp, "valid/b6_storage_lean_gate.hex");
    defer allocator.free(valid_hex_path);
    const valid_cert = try readFileAllocForTest(allocator, valid_cert_path);
    defer allocator.free(valid_cert);
    try testing.expect(std.mem.containsAtLeast(u8, valid_cert, 1, "\"schema_version\": 3"));
    try testing.expect(std.mem.containsAtLeast(u8, valid_cert, 1, "\"proof_count\": 1"));
    try testing.expect(std.mem.containsAtLeast(u8, valid_cert, 1, "\"axioms\""));
    try testing.expect(!std.mem.containsAtLeast(u8, valid_cert, 1, "sorryAx"));

    const valid_sir = try readFileAllocForTest(allocator, valid_sir_path);
    defer allocator.free(valid_sir);
    try testing.expect(std.mem.containsAtLeast(u8, valid_sir, 1, "fn guarded:"));
    try testing.expect(std.mem.containsAtLeast(u8, valid_sir, 1, "gt v0 c0"));
    try testing.expect(std.mem.containsAtLeast(u8, valid_sir, 1, "revert 0x0 0x0"));
    const valid_hex = try readFileAllocForTest(allocator, valid_hex_path);
    defer allocator.free(valid_hex);
    try testing.expect(valid_hex.len > 0);

    {
        const result = try runProcess(allocator, &.{
            ORA_BINARY_REL,
            "emit",
            "--emit=sir-text,bytecode",
            "--out-dir",
            reference_out,
            source_path,
        });
        defer allocator.free(result.stdout);
        defer allocator.free(result.stderr);
        try expectExited(result, 0);
    }
    const reference_sir_path = try pathFromTmpAlloc(allocator, tmp, "reference/b6_storage_lean_gate.sir");
    defer allocator.free(reference_sir_path);
    const reference_hex_path = try pathFromTmpAlloc(allocator, tmp, "reference/b6_storage_lean_gate.hex");
    defer allocator.free(reference_hex_path);
    const reference_sir = try readFileAllocForTest(allocator, reference_sir_path);
    defer allocator.free(reference_sir);
    const reference_hex = try readFileAllocForTest(allocator, reference_hex_path);
    defer allocator.free(reference_hex);
    try testing.expectEqualStrings(reference_sir, valid_sir);
    try testing.expectEqualStrings(reference_hex, valid_hex);

    {
        const result = try runOraWithForcedUnknown(allocator, forced_query, &.{
            ORA_BINARY_REL,
            "emit",
            "--emit=smt-report,sir-text,bytecode",
            "--out-dir",
            sorry_out,
            "--lean-proofs",
            sorry_manifest,
            source_path,
        });
        defer allocator.free(result.stdout);
        defer allocator.free(result.stderr);
        try expectExited(result, 1);
        try testing.expect(std.mem.containsAtLeast(u8, result.stdout, 1, "sorryAx") or
            std.mem.containsAtLeast(u8, result.stderr, 1, "sorryAx") or
            std.mem.containsAtLeast(u8, result.stdout, 1, "Lean proof gate failed") or
            std.mem.containsAtLeast(u8, result.stderr, 1, "Lean proof gate failed"));
    }
    try testing.expectError(error.FileNotFound, tmp.dir.access(std.testing.io, "sorry/b6_storage_lean_gate.hex", .{}));
    try testing.expectError(error.FileNotFound, tmp.dir.access(std.testing.io, "sorry/b6_storage_lean_gate.lean.proof.json", .{}));
}

test "B6 storage path Lean proof unblocks constant map read old collapse" {
    std.Io.Dir.cwd().access(std.testing.io, ORA_BINARY_REL, .{}) catch |err| switch (err) {
        error.FileNotFound => return error.SkipZigTest,
        else => return err,
    };

    const allocator = testing.allocator;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();

    const source =
        \\contract B6StoragePathLeanProofGate {
        \\    storage buckets: map<u256, u256>;
        \\
        \\    pub fn framed() -> bool
        \\        ensures buckets[2] == old(buckets[2])
        \\    {
        \\        return true;
        \\    }
        \\
        \\    pub fn guarded(y: u256) -> bool
        \\        guard y > 0
        \\    {
        \\        return true;
        \\    }
        \\}
    ;
    try tmp.dir.writeFile(std.testing.io, .{ .sub_path = "b6_storage_path_lean_gate.ora", .data = source });

    const source_path = try pathFromTmpAlloc(allocator, tmp, "b6_storage_path_lean_gate.ora");
    defer allocator.free(source_path);
    const fail_out = try pathFromTmpAlloc(allocator, tmp, "fail");
    defer allocator.free(fail_out);
    const valid_out = try pathFromTmpAlloc(allocator, tmp, "valid");
    defer allocator.free(valid_out);
    const reference_out = try pathFromTmpAlloc(allocator, tmp, "reference");
    defer allocator.free(reference_out);

    var formal_result = try collectPackageObligations(allocator, source_path);
    defer formal_result.deinit();
    _ = try findEnsuresQuery(formal_result.set);
    try expectEnsuresQuerySupported(formal_result.set);
    const force_target = "obligation:2";

    const generated_namespace = try leanProofGeneratedNamespaceForTest(allocator, source_path);
    defer allocator.free(generated_namespace);
    var obligations_source_out = std.Io.Writer.Allocating.init(allocator);
    defer obligations_source_out.deinit();
    try obligation_to_lean.writeModule(&obligations_source_out.writer, formal_result.set, .{
        .namespace = generated_namespace,
        .proof_surface = true,
    });
    const obligations_source = obligations_source_out.written();
    try testing.expect(std.mem.containsAtLeast(u8, obligations_source, 2, "root := \"buckets\""));
    try testing.expect(std.mem.containsAtLeast(u8, obligations_source, 2, "keys := [.constant \"2\"]"));
    try testing.expect(std.mem.indexOf(u8, obligations_source, ".old ") == null);

    var proof_row: ProofRecipeRowForTest = undefined;
    var proof_row_set = false;
    defer if (proof_row_set) proof_row.deinit(allocator);

    {
        const result = try runOraWithForcedUnknown(allocator, force_target, &.{
            ORA_BINARY_REL,
            "emit",
            "--emit=smt-report,sir-text,bytecode",
            "--out-dir",
            fail_out,
            source_path,
        });
        defer allocator.free(result.stdout);
        defer allocator.free(result.stderr);
        try expectExited(result, 1);
        try testing.expect(std.mem.containsAtLeast(u8, result.stdout, 1, "Lean proof recipe for required userland obligations") or
            std.mem.containsAtLeast(u8, result.stderr, 1, "Lean proof recipe for required userland obligations"));
        proof_row = parseProofRecipeRowForTest(allocator, result.stdout) catch try parseProofRecipeRowForTest(allocator, result.stderr);
        proof_row_set = true;
    }
    try testing.expectError(error.FileNotFound, tmp.dir.access(std.testing.io, "fail/b6_storage_path_lean_gate.hex", .{}));

    const fail_lean_obligations_path = try pathFromTmpAlloc(allocator, tmp, "fail/b6_storage_path_lean_gate.lean.obligations.lean");
    defer allocator.free(fail_lean_obligations_path);
    const fail_lean_obligations = try readFileAllocForTest(allocator, fail_lean_obligations_path);
    defer allocator.free(fail_lean_obligations);
    const expected_query_def = try std.fmt.allocPrint(allocator, "def emittedQuery_{d} : Prop :=", .{proof_row.query_id});
    defer allocator.free(expected_query_def);
    try testing.expect(std.mem.containsAtLeast(u8, fail_lean_obligations, 1, expected_query_def));
    try testing.expect(std.mem.containsAtLeast(u8, fail_lean_obligations, 2, "root := \"buckets\""));
    try testing.expect(std.mem.containsAtLeast(u8, fail_lean_obligations, 2, "keys := [.constant \"2\"]"));

    const module_suffix = try moduleSuffixFromTmp(allocator, tmp);
    defer allocator.free(module_suffix);
    const fixture_dir = try std.fmt.allocPrint(allocator, "formal/Ora/B6StoragePathFixture/{s}", .{module_suffix});
    defer allocator.free(fixture_dir);
    try std.Io.Dir.cwd().createDirPath(std.testing.io, fixture_dir);
    defer std.Io.Dir.cwd().deleteTree(std.testing.io, fixture_dir) catch {};
    defer std.Io.Dir.cwd().deleteDir(std.testing.io, "formal/Ora/B6StoragePathFixture") catch {};

    const valid_module = try std.fmt.allocPrint(allocator, "Ora.B6StoragePathFixture.{s}.Valid", .{module_suffix});
    defer allocator.free(valid_module);
    const valid_theorem = try std.fmt.allocPrint(allocator, "{s}.discharge", .{valid_module});
    defer allocator.free(valid_theorem);
    const valid_proof_path = try std.fmt.allocPrint(allocator, "{s}/Valid.lean", .{fixture_dir});
    defer allocator.free(valid_proof_path);
    const valid_proof = try storagePlaceProofModuleFromGeneratedObligations(
        allocator,
        fail_lean_obligations,
        valid_module,
        proof_row.query_id,
        "{ root := \"buckets\", region := .storage, fields := [], keys := [.constant \"2\"] }",
    );
    defer allocator.free(valid_proof);
    try std.Io.Dir.cwd().writeFile(std.testing.io, .{ .sub_path = valid_proof_path, .data = valid_proof });

    const valid_manifest = try pathFromTmpAlloc(allocator, tmp, "valid-proofs.json");
    defer allocator.free(valid_manifest);
    try writeProofManifestIdsForTest(
        allocator,
        valid_manifest,
        valid_module,
        valid_theorem,
        valid_proof_path,
        proof_row.query_id,
        proof_row.obligation_ids,
        proof_row.assumption_ids,
    );

    {
        const result = try runOraWithForcedUnknown(allocator, force_target, &.{
            ORA_BINARY_REL,
            "emit",
            "--emit=smt-report,sir-text,bytecode",
            "--out-dir",
            valid_out,
            "--lean-proofs",
            valid_manifest,
            source_path,
        });
        defer allocator.free(result.stdout);
        defer allocator.free(result.stderr);
        try expectExited(result, 0);
    }

    const valid_cert_path = try pathFromTmpAlloc(allocator, tmp, "valid/b6_storage_path_lean_gate.lean.proof.json");
    defer allocator.free(valid_cert_path);
    const valid_sir_path = try pathFromTmpAlloc(allocator, tmp, "valid/b6_storage_path_lean_gate.sir");
    defer allocator.free(valid_sir_path);
    const valid_hex_path = try pathFromTmpAlloc(allocator, tmp, "valid/b6_storage_path_lean_gate.hex");
    defer allocator.free(valid_hex_path);
    const valid_cert = try readFileAllocForTest(allocator, valid_cert_path);
    defer allocator.free(valid_cert);
    try testing.expect(std.mem.containsAtLeast(u8, valid_cert, 1, "\"schema_version\": 3"));
    try testing.expect(!std.mem.containsAtLeast(u8, valid_cert, 1, "sorryAx"));
    const valid_sir = try readFileAllocForTest(allocator, valid_sir_path);
    defer allocator.free(valid_sir);
    const valid_hex = try readFileAllocForTest(allocator, valid_hex_path);
    defer allocator.free(valid_hex);
    try testing.expect(valid_hex.len > 0);

    {
        const result = try runProcess(allocator, &.{
            ORA_BINARY_REL,
            "emit",
            "--emit=sir-text,bytecode",
            "--out-dir",
            reference_out,
            source_path,
        });
        defer allocator.free(result.stdout);
        defer allocator.free(result.stderr);
        try expectExited(result, 0);
    }
    const reference_sir_path = try pathFromTmpAlloc(allocator, tmp, "reference/b6_storage_path_lean_gate.sir");
    defer allocator.free(reference_sir_path);
    const reference_hex_path = try pathFromTmpAlloc(allocator, tmp, "reference/b6_storage_path_lean_gate.hex");
    defer allocator.free(reference_hex_path);
    const reference_sir = try readFileAllocForTest(allocator, reference_sir_path);
    defer allocator.free(reference_sir);
    const reference_hex = try readFileAllocForTest(allocator, reference_hex_path);
    defer allocator.free(reference_hex);
    try testing.expectEqualStrings(reference_sir, valid_sir);
    try testing.expectEqualStrings(reference_hex, valid_hex);
}

test "B6 storage key evidence Lean proof unblocks structural frame gate" {
    std.Io.Dir.cwd().access(std.testing.io, ORA_BINARY_REL, .{}) catch |err| switch (err) {
        error.FileNotFound => return error.SkipZigTest,
        else => return err,
    };

    const allocator = testing.allocator;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();

    const source =
        \\contract B6KeyEvidenceGate {
        \\    storage var balances: map<u256, u256>;
        \\
        \\    pub fn copy_other(user: u256, other: u256)
        \\        modifies balances[user]
        \\        requires user != other
        \\    {
        \\        let observed: u256 = balances[other];
        \\        balances[user] = observed;
        \\    }
        \\}
    ;
    try tmp.dir.writeFile(std.testing.io, .{ .sub_path = "b6_key_evidence_gate.ora", .data = source });

    const source_path = try pathFromTmpAlloc(allocator, tmp, "b6_key_evidence_gate.ora");
    defer allocator.free(source_path);
    const empty_out = try pathFromTmpAlloc(allocator, tmp, "empty");
    defer allocator.free(empty_out);
    const valid_out = try pathFromTmpAlloc(allocator, tmp, "valid");
    defer allocator.free(valid_out);
    const sorry_out = try pathFromTmpAlloc(allocator, tmp, "sorry");
    defer allocator.free(sorry_out);
    const reference_out = try pathFromTmpAlloc(allocator, tmp, "reference");
    defer allocator.free(reference_out);

    var formal_result = try collectPackageObligations(allocator, source_path);
    defer formal_result.deinit();
    try testing.expectEqual(@as(usize, 1), countAssumption(formal_result.set, .requires));
    try testing.expectEqual(@as(usize, 1), countEffectFrame(formal_result.set, .read_preserved_by_key_evidence));
    const frame_query = try findEffectFrameQuery(formal_result.set, .read_preserved_by_key_evidence);
    try testing.expectEqual(obligation.VerificationBackend.unspecified, frame_query.backend);
    try testing.expect(frame_query.result == null);
    try testing.expectEqual(@as(usize, 1), frame_query.assumption_ids.len);
    try expectEffectFrameQuerySupported(formal_result.set, .read_preserved_by_key_evidence);
    const write_query = try findEffectFrameQuery(formal_result.set, .write_covered_by_modifies);
    try testing.expectEqual(@as(usize, 0), write_query.assumption_ids.len);

    var frame_evidence: ?obligation.KeyDisjointEvidence = null;
    for (formal_result.set.obligations) |item| {
        if (item.kind != .effect_frame or item.kind.effect_frame.relation != .read_preserved_by_key_evidence) continue;
        try testing.expectEqual(@as(usize, 1), item.kind.effect_frame.evidence.len);
        frame_evidence = item.kind.effect_frame.evidence[0];
    }
    const evidence = frame_evidence orelse return error.TestUnexpectedResult;

    const generated_namespace = try leanProofGeneratedNamespaceForTest(allocator, source_path);
    defer allocator.free(generated_namespace);
    var obligations_source_out = std.Io.Writer.Allocating.init(allocator);
    defer obligations_source_out.deinit();
    try obligation_to_lean.writeModule(&obligations_source_out.writer, formal_result.set, .{
        .namespace = generated_namespace,
        .proof_surface = true,
    });
    const obligations_source = obligations_source_out.written();
    const expected_query_def = try std.fmt.allocPrint(allocator, "def emittedQuery_{d} : Prop :=", .{frame_query.id});
    defer allocator.free(expected_query_def);
    try testing.expect(std.mem.containsAtLeast(u8, obligations_source, 1, expected_query_def));
    try testing.expect(std.mem.containsAtLeast(u8, obligations_source, 1, ".readPreservedByKeyEvidence"));
    try testing.expect(std.mem.containsAtLeast(u8, obligations_source, 1, "keyIndex := 0"));

    {
        const result = try runProcessWithLeanPath(allocator, &.{
            ORA_BINARY_REL,
            "emit",
            "--emit=smt-report,sir-text,bytecode",
            "--out-dir",
            reference_out,
            source_path,
        });
        defer allocator.free(result.stdout);
        defer allocator.free(result.stderr);
        try expectExited(result, 0);
    }
    const reference_sir_path = try pathFromTmpAlloc(allocator, tmp, "reference/b6_key_evidence_gate.sir");
    defer allocator.free(reference_sir_path);
    const reference_hex_path = try pathFromTmpAlloc(allocator, tmp, "reference/b6_key_evidence_gate.hex");
    defer allocator.free(reference_hex_path);
    const reference_sir = try readFileAllocForTest(allocator, reference_sir_path);
    defer allocator.free(reference_sir);
    const reference_hex = try readFileAllocForTest(allocator, reference_hex_path);
    defer allocator.free(reference_hex);
    try testing.expect(reference_hex.len > 0);

    const module_suffix = try moduleSuffixFromTmp(allocator, tmp);
    defer allocator.free(module_suffix);
    const fixture_dir = try std.fmt.allocPrint(allocator, "formal/Ora/B6KeyEvidenceFixture/{s}", .{module_suffix});
    defer allocator.free(fixture_dir);
    try std.Io.Dir.cwd().createDirPath(std.testing.io, fixture_dir);
    defer std.Io.Dir.cwd().deleteTree(std.testing.io, fixture_dir) catch {};
    defer std.Io.Dir.cwd().deleteDir(std.testing.io, "formal/Ora/B6KeyEvidenceFixture") catch {};

    const valid_module = try std.fmt.allocPrint(allocator, "Ora.B6KeyEvidenceFixture.{s}.Valid", .{module_suffix});
    defer allocator.free(valid_module);
    const sorry_module = try std.fmt.allocPrint(allocator, "Ora.B6KeyEvidenceFixture.{s}.Sorry", .{module_suffix});
    defer allocator.free(sorry_module);
    const valid_theorem = try std.fmt.allocPrint(allocator, "{s}.discharge", .{valid_module});
    defer allocator.free(valid_theorem);
    const sorry_theorem = try std.fmt.allocPrint(allocator, "{s}.discharge", .{sorry_module});
    defer allocator.free(sorry_theorem);
    const valid_proof_path = try std.fmt.allocPrint(allocator, "{s}/Valid.lean", .{fixture_dir});
    defer allocator.free(valid_proof_path);
    const sorry_proof_path = try std.fmt.allocPrint(allocator, "{s}/Sorry.lean", .{fixture_dir});
    defer allocator.free(sorry_proof_path);

    const valid_proof = try keyEvidenceFrameProofModuleFromGeneratedObligations(
        allocator,
        obligations_source,
        valid_module,
        frame_query.id,
        evidence.read,
        evidence.write,
        evidence.lhs,
        evidence.rhs,
        write_query.id,
        false,
    );
    defer allocator.free(valid_proof);
    const sorry_proof = try keyEvidenceFrameProofModuleFromGeneratedObligations(
        allocator,
        obligations_source,
        sorry_module,
        frame_query.id,
        evidence.read,
        evidence.write,
        evidence.lhs,
        evidence.rhs,
        write_query.id,
        true,
    );
    defer allocator.free(sorry_proof);
    try std.Io.Dir.cwd().writeFile(std.testing.io, .{ .sub_path = valid_proof_path, .data = valid_proof });
    try std.Io.Dir.cwd().writeFile(std.testing.io, .{ .sub_path = sorry_proof_path, .data = sorry_proof });

    const valid_manifest = try pathFromTmpAlloc(allocator, tmp, "valid-proofs.json");
    defer allocator.free(valid_manifest);
    const sorry_manifest = try pathFromTmpAlloc(allocator, tmp, "sorry-proofs.json");
    defer allocator.free(sorry_manifest);
    const empty_manifest = try pathFromTmpAlloc(allocator, tmp, "empty-proofs.json");
    defer allocator.free(empty_manifest);
    const valid_write_theorem = try std.fmt.allocPrint(allocator, "{s}.discharge_write", .{valid_module});
    defer allocator.free(valid_write_theorem);
    const sorry_write_theorem = try std.fmt.allocPrint(allocator, "{s}.discharge_write", .{sorry_module});
    defer allocator.free(sorry_write_theorem);
    try writeTwoProofManifestForTest(
        allocator,
        valid_manifest,
        valid_module,
        valid_theorem,
        valid_proof_path,
        frame_query,
        valid_write_theorem,
        valid_proof_path,
        write_query,
    );
    try writeTwoProofManifestForTest(
        allocator,
        sorry_manifest,
        sorry_module,
        sorry_theorem,
        sorry_proof_path,
        frame_query,
        sorry_write_theorem,
        sorry_proof_path,
        write_query,
    );
    try std.Io.Dir.cwd().writeFile(std.testing.io, .{
        .sub_path = empty_manifest,
        .data = "{\n  \"schema_version\": 1,\n  \"proofs\": []\n}\n",
    });

    {
        const result = try runProcessWithLeanPath(allocator, &.{
            ORA_BINARY_REL,
            "emit",
            "--emit=smt-report,sir-text,bytecode",
            "--out-dir",
            empty_out,
            "--lean-proofs",
            empty_manifest,
            source_path,
        });
        defer allocator.free(result.stdout);
        defer allocator.free(result.stderr);
        try expectExited(result, 1);
        try testing.expect(std.mem.containsAtLeast(u8, result.stdout, 1, "missing_proof") or
            std.mem.containsAtLeast(u8, result.stderr, 1, "missing_proof"));
    }
    try testing.expectError(error.FileNotFound, tmp.dir.access(std.testing.io, "empty/b6_key_evidence_gate.hex", .{}));

    {
        const result = try runProcessWithLeanPath(allocator, &.{
            ORA_BINARY_REL,
            "emit",
            "--emit=smt-report,sir-text,bytecode",
            "--out-dir",
            valid_out,
            "--lean-proofs",
            valid_manifest,
            source_path,
        });
        defer allocator.free(result.stdout);
        defer allocator.free(result.stderr);
        try expectExited(result, 0);
    }

    const valid_cert_path = try pathFromTmpAlloc(allocator, tmp, "valid/b6_key_evidence_gate.lean.proof.json");
    defer allocator.free(valid_cert_path);
    const valid_sir_path = try pathFromTmpAlloc(allocator, tmp, "valid/b6_key_evidence_gate.sir");
    defer allocator.free(valid_sir_path);
    const valid_hex_path = try pathFromTmpAlloc(allocator, tmp, "valid/b6_key_evidence_gate.hex");
    defer allocator.free(valid_hex_path);
    const valid_cert = try readFileAllocForTest(allocator, valid_cert_path);
    defer allocator.free(valid_cert);
    try testing.expect(std.mem.containsAtLeast(u8, valid_cert, 1, "\"schema_version\": 3"));
    try testing.expect(std.mem.containsAtLeast(u8, valid_cert, 1, "\"proof_count\": 2"));
    try testing.expect(!std.mem.containsAtLeast(u8, valid_cert, 1, "sorryAx"));
    const valid_sir = try readFileAllocForTest(allocator, valid_sir_path);
    defer allocator.free(valid_sir);
    const valid_hex = try readFileAllocForTest(allocator, valid_hex_path);
    defer allocator.free(valid_hex);
    try testing.expectEqualStrings(reference_sir, valid_sir);
    try testing.expectEqualStrings(reference_hex, valid_hex);

    {
        const result = try runProcess(allocator, &.{
            ORA_BINARY_REL,
            "emit",
            "--emit=smt-report,sir-text,bytecode",
            "--out-dir",
            sorry_out,
            "--lean-proofs",
            sorry_manifest,
            source_path,
        });
        defer allocator.free(result.stdout);
        defer allocator.free(result.stderr);
        try expectExited(result, 1);
        try testing.expect(std.mem.containsAtLeast(u8, result.stdout, 1, "sorryAx") or
            std.mem.containsAtLeast(u8, result.stderr, 1, "sorryAx") or
            std.mem.containsAtLeast(u8, result.stdout, 1, "Lean proof gate failed") or
            std.mem.containsAtLeast(u8, result.stderr, 1, "Lean proof gate failed"));
    }
    try testing.expectError(error.FileNotFound, tmp.dir.access(std.testing.io, "sorry/b6_key_evidence_gate.hex", .{}));
}

test "B6 storage key evidence proof rejects contradictory assumptions" {
    const allocator = testing.allocator;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();

    const source =
        \\contract B6KeyEvidenceContradiction {
        \\    storage var balances: map<u256, u256>;
        \\
        \\    pub fn copy_other(user: u256, other: u256)
        \\        modifies balances[user]
        \\        requires user != other
        \\        requires user == other
        \\    {
        \\        let observed: u256 = balances[other];
        \\        balances[user] = observed;
        \\    }
        \\}
    ;
    try tmp.dir.writeFile(std.testing.io, .{ .sub_path = "b6_key_evidence_contradiction.ora", .data = source });

    const source_path = try pathFromTmpAlloc(allocator, tmp, "b6_key_evidence_contradiction.ora");
    defer allocator.free(source_path);

    var formal_result = try collectPackageObligations(allocator, source_path);
    defer formal_result.deinit();
    try testing.expectEqual(@as(usize, 2), countAssumption(formal_result.set, .requires));
    try testing.expectEqual(@as(usize, 1), countEffectFrame(formal_result.set, .read_preserved_by_key_evidence));
    const frame_query = try findEffectFrameQuery(formal_result.set, .read_preserved_by_key_evidence);
    try testing.expectEqual(@as(usize, 2), frame_query.assumption_ids.len);
    try expectEffectFrameQuerySupported(formal_result.set, .read_preserved_by_key_evidence);

    var frame_evidence: ?obligation.KeyDisjointEvidence = null;
    for (formal_result.set.obligations) |item| {
        if (item.kind != .effect_frame or item.kind.effect_frame.relation != .read_preserved_by_key_evidence) continue;
        frame_evidence = item.kind.effect_frame.evidence[0];
    }
    const evidence = frame_evidence orelse return error.TestUnexpectedResult;

    const generated_namespace = try leanProofGeneratedNamespaceForTest(allocator, source_path);
    defer allocator.free(generated_namespace);
    var obligations_source_out = std.Io.Writer.Allocating.init(allocator);
    defer obligations_source_out.deinit();
    try obligation_to_lean.writeModule(&obligations_source_out.writer, formal_result.set, .{
        .namespace = generated_namespace,
        .proof_surface = true,
    });
    const obligations_source = obligations_source_out.written();

    const module_suffix = try moduleSuffixFromTmp(allocator, tmp);
    defer allocator.free(module_suffix);
    const fixture_dir = try std.fmt.allocPrint(allocator, "formal/Ora/B6KeyEvidenceContradictionFixture/{s}", .{module_suffix});
    defer allocator.free(fixture_dir);
    try std.Io.Dir.cwd().createDirPath(std.testing.io, fixture_dir);
    defer std.Io.Dir.cwd().deleteTree(std.testing.io, fixture_dir) catch {};
    defer std.Io.Dir.cwd().deleteDir(std.testing.io, "formal/Ora/B6KeyEvidenceContradictionFixture") catch {};

    const proof_module = try std.fmt.allocPrint(allocator, "Ora.B6KeyEvidenceContradictionFixture.{s}.Invalid", .{module_suffix});
    defer allocator.free(proof_module);
    const proof_path = try std.fmt.allocPrint(allocator, "{s}/Invalid.lean", .{fixture_dir});
    defer allocator.free(proof_path);
    const proof = try keyEvidenceFrameProofModuleFromGeneratedObligations(
        allocator,
        obligations_source,
        proof_module,
        frame_query.id,
        evidence.read,
        evidence.write,
        evidence.lhs,
        evidence.rhs,
        null,
        false,
    );
    defer allocator.free(proof);
    try std.Io.Dir.cwd().writeFile(std.testing.io, .{ .sub_path = proof_path, .data = proof });

    const result = try runLeanFileForTest(allocator, proof_path);
    defer allocator.free(result.stdout);
    defer allocator.free(result.stderr);
    try expectExited(result, 1);
}

test "B6 signed comparison Lean proof unblocks source-level unknown" {
    std.Io.Dir.cwd().access(std.testing.io, ORA_BINARY_REL, .{}) catch |err| switch (err) {
        error.FileNotFound => return error.SkipZigTest,
        else => return err,
    };

    const allocator = testing.allocator;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();

    const source =
        \\contract B6SignedLeanProofGate {
        \\    pub fn signed_reflexive(x: i256) -> bool
        \\        ensures x <= x
        \\    {
        \\        return true;
        \\    }
        \\}
    ;
    try tmp.dir.writeFile(std.testing.io, .{ .sub_path = "b6_signed_lean_gate.ora", .data = source });

    const source_path = try pathFromTmpAlloc(allocator, tmp, "b6_signed_lean_gate.ora");
    defer allocator.free(source_path);
    const fail_out = try pathFromTmpAlloc(allocator, tmp, "fail");
    defer allocator.free(fail_out);
    const valid_out = try pathFromTmpAlloc(allocator, tmp, "valid");
    defer allocator.free(valid_out);

    var formal_result = try collectPackageObligations(allocator, source_path);
    defer formal_result.deinit();
    const ensures_query = try findEnsuresQuery(formal_result.set);
    try expectEnsuresQuerySupported(formal_result.set);
    const force_target = try std.fmt.allocPrint(allocator, "obligation:{d}", .{ensures_query.id});
    defer allocator.free(force_target);

    const generated_namespace = try leanProofGeneratedNamespaceForTest(allocator, source_path);
    defer allocator.free(generated_namespace);
    var obligations_source_out = std.Io.Writer.Allocating.init(allocator);
    defer obligations_source_out.deinit();
    try obligation_to_lean.writeModule(&obligations_source_out.writer, formal_result.set, .{
        .namespace = generated_namespace,
    });
    const obligations_source = obligations_source_out.written();
    try testing.expect(std.mem.containsAtLeast(u8, obligations_source, 1, ".sle"));
    try testing.expect(std.mem.containsAtLeast(u8, obligations_source, 1, ".compilerTypeId 12"));

    {
        const result = try runOraWithForcedUnknown(allocator, force_target, &.{
            ORA_BINARY_REL,
            "emit",
            "--emit=smt-report,bytecode",
            "--out-dir",
            fail_out,
            source_path,
        });
        defer allocator.free(result.stdout);
        defer allocator.free(result.stderr);
        try expectExited(result, 1);
        try testing.expect(std.mem.containsAtLeast(u8, result.stdout, 1, "Lean proof recipe for required userland obligations"));
    }
    try testing.expectError(error.FileNotFound, tmp.dir.access(std.testing.io, "fail/b6_signed_lean_gate.hex", .{}));

    const module_suffix = try moduleSuffixFromTmp(allocator, tmp);
    defer allocator.free(module_suffix);
    const fixture_dir = try std.fmt.allocPrint(allocator, "formal/Ora/B6SignedFixture/{s}", .{module_suffix});
    defer allocator.free(fixture_dir);
    try std.Io.Dir.cwd().createDirPath(std.testing.io, fixture_dir);
    defer std.Io.Dir.cwd().deleteTree(std.testing.io, fixture_dir) catch {};
    defer std.Io.Dir.cwd().deleteDir(std.testing.io, "formal/Ora/B6SignedFixture") catch {};

    const valid_module = try std.fmt.allocPrint(allocator, "Ora.B6SignedFixture.{s}.Valid", .{module_suffix});
    defer allocator.free(valid_module);
    const valid_theorem = try std.fmt.allocPrint(allocator, "{s}.discharge", .{valid_module});
    defer allocator.free(valid_theorem);
    const valid_proof_path = try std.fmt.allocPrint(allocator, "{s}/Valid.lean", .{fixture_dir});
    defer allocator.free(valid_proof_path);

    const valid_proof = try signedComparisonProofModuleFromGeneratedObligations(allocator, obligations_source, valid_module, ensures_query.id);
    defer allocator.free(valid_proof);
    try std.Io.Dir.cwd().writeFile(std.testing.io, .{ .sub_path = valid_proof_path, .data = valid_proof });

    const valid_manifest = try pathFromTmpAlloc(allocator, tmp, "valid-proofs.json");
    defer allocator.free(valid_manifest);
    try writeProofManifestForTest(allocator, valid_manifest, valid_module, valid_theorem, valid_proof_path, ensures_query);

    {
        const result = try runOraWithForcedUnknown(allocator, force_target, &.{
            ORA_BINARY_REL,
            "emit",
            "--emit=smt-report,bytecode",
            "--out-dir",
            valid_out,
            "--lean-proofs",
            valid_manifest,
            source_path,
        });
        defer allocator.free(result.stdout);
        defer allocator.free(result.stderr);
        try expectExited(result, 0);
    }

    const valid_cert_path = try pathFromTmpAlloc(allocator, tmp, "valid/b6_signed_lean_gate.lean.proof.json");
    defer allocator.free(valid_cert_path);
    const valid_hex_path = try pathFromTmpAlloc(allocator, tmp, "valid/b6_signed_lean_gate.hex");
    defer allocator.free(valid_hex_path);
    const valid_cert = try readFileAllocForTest(allocator, valid_cert_path);
    defer allocator.free(valid_cert);
    try testing.expect(std.mem.containsAtLeast(u8, valid_cert, 1, "\"schema_version\": 3"));
    try testing.expect(!std.mem.containsAtLeast(u8, valid_cert, 1, "sorryAx"));
    const valid_hex = try readFileAllocForTest(allocator, valid_hex_path);
    defer allocator.free(valid_hex);
    try testing.expect(valid_hex.len > 0);
}

test "B6 div rem Lean proof unblocks source-level unknown" {
    std.Io.Dir.cwd().access(std.testing.io, ORA_BINARY_REL, .{}) catch |err| switch (err) {
        error.FileNotFound => return error.SkipZigTest,
        else => return err,
    };

    const allocator = testing.allocator;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();

    const source =
        \\contract B6DivRemLeanProofGate {
        \\    pub fn div_reflexive(x: u256) -> bool
        \\        ensures (x / 1) == (x / 1)
        \\    {
        \\        return true;
        \\    }
        \\}
    ;
    try tmp.dir.writeFile(std.testing.io, .{ .sub_path = "b6_div_rem_lean_gate.ora", .data = source });

    const source_path = try pathFromTmpAlloc(allocator, tmp, "b6_div_rem_lean_gate.ora");
    defer allocator.free(source_path);
    const fail_out = try pathFromTmpAlloc(allocator, tmp, "fail");
    defer allocator.free(fail_out);
    const valid_out = try pathFromTmpAlloc(allocator, tmp, "valid");
    defer allocator.free(valid_out);

    var formal_result = try collectPackageObligations(allocator, source_path);
    defer formal_result.deinit();
    const ensures_query = try findEnsuresQuery(formal_result.set);
    try expectEnsuresQuerySupported(formal_result.set);

    const generated_namespace = try leanProofGeneratedNamespaceForTest(allocator, source_path);
    defer allocator.free(generated_namespace);
    var obligations_source_out = std.Io.Writer.Allocating.init(allocator);
    defer obligations_source_out.deinit();
    try obligation_to_lean.writeModule(&obligations_source_out.writer, formal_result.set, .{
        .namespace = generated_namespace,
        .proof_surface = true,
    });
    const obligations_source = obligations_source_out.written();
    try testing.expect(std.mem.containsAtLeast(u8, obligations_source, 1, ".div"));
    try testing.expect(std.mem.containsAtLeast(u8, obligations_source, 1, ".compilerTypeId 6"));

    {
        // The base row and two source-owned divisor-safety rows precede
        // `ensures`; each checked division has its own safety obligation.
        const result = try runOraWithForcedUnknown(allocator, "obligation:4", &.{
            ORA_BINARY_REL,
            "emit",
            "--emit=smt-report,bytecode",
            "--out-dir",
            fail_out,
            source_path,
        });
        defer allocator.free(result.stdout);
        defer allocator.free(result.stderr);
        try expectExited(result, 1);
        try testing.expect(std.mem.containsAtLeast(u8, result.stdout, 1, "Lean proof recipe for required userland obligations"));
    }
    try testing.expectError(error.FileNotFound, tmp.dir.access(std.testing.io, "fail/b6_div_rem_lean_gate.hex", .{}));

    const module_suffix = try moduleSuffixFromTmp(allocator, tmp);
    defer allocator.free(module_suffix);
    const fixture_dir = try std.fmt.allocPrint(allocator, "formal/Ora/B6DivRemFixture/{s}", .{module_suffix});
    defer allocator.free(fixture_dir);
    try std.Io.Dir.cwd().createDirPath(std.testing.io, fixture_dir);
    defer std.Io.Dir.cwd().deleteTree(std.testing.io, fixture_dir) catch {};
    defer std.Io.Dir.cwd().deleteDir(std.testing.io, "formal/Ora/B6DivRemFixture") catch {};

    const valid_module = try std.fmt.allocPrint(allocator, "Ora.B6DivRemFixture.{s}.Valid", .{module_suffix});
    defer allocator.free(valid_module);
    const valid_theorem = try std.fmt.allocPrint(allocator, "{s}.discharge", .{valid_module});
    defer allocator.free(valid_theorem);
    const valid_proof_path = try std.fmt.allocPrint(allocator, "{s}/Valid.lean", .{fixture_dir});
    defer allocator.free(valid_proof_path);

    const valid_proof = try divRemProofModuleFromGeneratedObligations(allocator, obligations_source, valid_module, ensures_query.id, false);
    defer allocator.free(valid_proof);
    try std.Io.Dir.cwd().writeFile(std.testing.io, .{ .sub_path = valid_proof_path, .data = valid_proof });

    const valid_manifest = try pathFromTmpAlloc(allocator, tmp, "valid-proofs.json");
    defer allocator.free(valid_manifest);
    try writeProofManifestForTest(allocator, valid_manifest, valid_module, valid_theorem, valid_proof_path, ensures_query);

    {
        const result = try runOraWithForcedUnknown(allocator, "obligation:4", &.{
            ORA_BINARY_REL,
            "emit",
            "--emit=smt-report,bytecode",
            "--out-dir",
            valid_out,
            "--lean-proofs",
            valid_manifest,
            source_path,
        });
        defer allocator.free(result.stdout);
        defer allocator.free(result.stderr);
        try expectExited(result, 0);
    }

    const valid_cert_path = try pathFromTmpAlloc(allocator, tmp, "valid/b6_div_rem_lean_gate.lean.proof.json");
    defer allocator.free(valid_cert_path);
    const valid_hex_path = try pathFromTmpAlloc(allocator, tmp, "valid/b6_div_rem_lean_gate.hex");
    defer allocator.free(valid_hex_path);
    const valid_cert = try readFileAllocForTest(allocator, valid_cert_path);
    defer allocator.free(valid_cert);
    try testing.expect(std.mem.containsAtLeast(u8, valid_cert, 1, "\"schema_version\": 3"));
    try testing.expect(!std.mem.containsAtLeast(u8, valid_cert, 1, "sorryAx"));
    const valid_hex = try readFileAllocForTest(allocator, valid_hex_path);
    defer allocator.free(valid_hex);
    try testing.expect(valid_hex.len > 0);
}

test "B6 div rem Lean proof does not discharge missing divisor guard" {
    std.Io.Dir.cwd().access(std.testing.io, ORA_BINARY_REL, .{}) catch |err| switch (err) {
        error.FileNotFound => return error.SkipZigTest,
        else => return err,
    };

    const allocator = testing.allocator;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();

    const source =
        \\contract B6DivRemGuardBoundary {
        \\    pub fn div_reflexive(x: u256, y: u256) -> bool
        \\        ensures (x / y) == (x / y)
        \\    {
        \\        return true;
        \\    }
        \\}
    ;
    try tmp.dir.writeFile(std.testing.io, .{ .sub_path = "b6_div_rem_guard_boundary.ora", .data = source });

    const source_path = try pathFromTmpAlloc(allocator, tmp, "b6_div_rem_guard_boundary.ora");
    defer allocator.free(source_path);
    const blocked_out = try pathFromTmpAlloc(allocator, tmp, "blocked");
    defer allocator.free(blocked_out);

    var formal_result = try collectPackageObligations(allocator, source_path);
    defer formal_result.deinit();
    const ensures_query = try findEnsuresQuery(formal_result.set);
    try expectEnsuresQuerySupported(formal_result.set);

    const generated_namespace = try leanProofGeneratedNamespaceForTest(allocator, source_path);
    defer allocator.free(generated_namespace);
    var obligations_source_out = std.Io.Writer.Allocating.init(allocator);
    defer obligations_source_out.deinit();
    try obligation_to_lean.writeModule(&obligations_source_out.writer, formal_result.set, .{
        .namespace = generated_namespace,
        .proof_surface = true,
    });
    const obligations_source = obligations_source_out.written();
    try testing.expect(std.mem.containsAtLeast(u8, obligations_source, 1, ".div"));

    const module_suffix = try moduleSuffixFromTmp(allocator, tmp);
    defer allocator.free(module_suffix);
    const fixture_dir = try std.fmt.allocPrint(allocator, "formal/Ora/B6DivRemBoundaryFixture/{s}", .{module_suffix});
    defer allocator.free(fixture_dir);
    try std.Io.Dir.cwd().createDirPath(std.testing.io, fixture_dir);
    defer std.Io.Dir.cwd().deleteTree(std.testing.io, fixture_dir) catch {};
    defer std.Io.Dir.cwd().deleteDir(std.testing.io, "formal/Ora/B6DivRemBoundaryFixture") catch {};

    const valid_module = try std.fmt.allocPrint(allocator, "Ora.B6DivRemBoundaryFixture.{s}.Valid", .{module_suffix});
    defer allocator.free(valid_module);
    const valid_theorem = try std.fmt.allocPrint(allocator, "{s}.discharge", .{valid_module});
    defer allocator.free(valid_theorem);
    const valid_proof_path = try std.fmt.allocPrint(allocator, "{s}/Valid.lean", .{fixture_dir});
    defer allocator.free(valid_proof_path);

    const valid_proof = try divRemProofModuleFromGeneratedObligations(allocator, obligations_source, valid_module, ensures_query.id, false);
    defer allocator.free(valid_proof);
    try std.Io.Dir.cwd().writeFile(std.testing.io, .{ .sub_path = valid_proof_path, .data = valid_proof });

    const valid_manifest = try pathFromTmpAlloc(allocator, tmp, "valid-proofs.json");
    defer allocator.free(valid_manifest);
    try writeProofManifestForTest(allocator, valid_manifest, valid_module, valid_theorem, valid_proof_path, ensures_query);

    {
        const result = try runOraWithForcedUnknown(allocator, "obligation:2", &.{
            ORA_BINARY_REL,
            "emit",
            "--emit=smt-report,bytecode",
            "--out-dir",
            blocked_out,
            "--lean-proofs",
            valid_manifest,
            source_path,
        });
        defer allocator.free(result.stdout);
        defer allocator.free(result.stderr);
        try expectExited(result, 1);
        try testing.expect(std.mem.containsAtLeast(u8, result.stdout, 1, "failed_query") or
            std.mem.containsAtLeast(u8, result.stderr, 1, "failed_query") or
            std.mem.containsAtLeast(u8, result.stdout, 1, "failed to prove contract invariant") or
            std.mem.containsAtLeast(u8, result.stderr, 1, "failed to prove contract invariant"));
    }
    try testing.expectError(error.FileNotFound, tmp.dir.access(std.testing.io, "blocked/b6_div_rem_guard_boundary.hex", .{}));
}

test "lean proofs run dispatcher table proof before bytecode emission" {
    std.Io.Dir.cwd().access(std.testing.io, ORA_BINARY_REL, .{}) catch |err| switch (err) {
        error.FileNotFound => return error.SkipZigTest,
        else => return err,
    };

    const allocator = testing.allocator;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();

    const source =
        \\contract DispatcherLeanGate {
        \\    pub fn init() {}
        \\
        \\    pub fn den0_() -> u256 {
        \\        return 1;
        \\    }
        \\
        \\    pub fn den1_() -> u256 {
        \\        return 2;
        \\    }
        \\
        \\    pub fn den2_() -> u256 { return 3; }
        \\    pub fn den3_() -> u256 { return 4; }
        \\    pub fn den4_() -> u256 { return 5; }
        \\    pub fn den5_() -> u256 { return 6; }
        \\    pub fn den6_() -> u256 { return 7; }
        \\    pub fn den7_() -> u256 { return 8; }
        \\    pub fn den8_() -> u256 { return 9; }
        \\    pub fn den9_() -> u256 { return 10; }
        \\    pub fn den10_() -> u256 { return 11; }
        \\    pub fn den11_() -> u256 { return 12; }
        \\}
    ;
    try tmp.dir.writeFile(std.testing.io, .{ .sub_path = "dispatcher_lean_gate.ora", .data = source });
    const source_path = try pathFromTmpAlloc(allocator, tmp, "dispatcher_lean_gate.ora");
    defer allocator.free(source_path);
    const out_dir = try pathFromTmpAlloc(allocator, tmp, "out");
    defer allocator.free(out_dir);

    const result = try runProcessWithLeanPath(allocator, &.{
        ORA_BINARY_REL,
        "emit",
        "--emit=bytecode",
        "--out-dir",
        out_dir,
        "--lean-proofs",
        source_path,
    });
    defer allocator.free(result.stdout);
    defer allocator.free(result.stderr);
    try expectExited(result, 0);
    try testing.expect(std.mem.containsAtLeast(u8, result.stdout, 1, "Lean dispatcher userland proof accepted") or
        std.mem.containsAtLeast(u8, result.stderr, 1, "Lean dispatcher userland proof accepted"));
    try testing.expect(std.mem.containsAtLeast(u8, result.stdout, 1, "Lean dispatcher checker output:"));
    try testing.expect(std.mem.containsAtLeast(u8, result.stdout, 1, "exit status: 0"));
    try testing.expect(std.mem.containsAtLeast(
        u8,
        result.stdout,
        1,
        "input: 1 switch(es), 12 case(s)",
    ));
    try testing.expect(std.mem.containsAtLeast(
        u8,
        result.stdout,
        1,
        "current_dispatcher_planner_reference_matches",
    ));
    try testing.expect(std.mem.containsAtLeast(
        u8,
        result.stdout,
        1,
        "bytecode binding (compiler-side checks, not an additional Lean theorem)",
    ));
    try tmp.dir.access(std.testing.io, "out/dispatcher_lean_gate.hex", .{});
    const dispatcher_cert_path = try pathFromTmpAlloc(allocator, tmp, "out/dispatcher_lean_gate.lean.dispatcher.proof.json");
    defer allocator.free(dispatcher_cert_path);
    const dispatcher_cert = try readFileAllocForTest(allocator, dispatcher_cert_path);
    defer allocator.free(dispatcher_cert);
    try testing.expect(std.mem.containsAtLeast(u8, dispatcher_cert, 1, "\"schema_version\": 1"));
    try testing.expect(std.mem.containsAtLeast(u8, dispatcher_cert, 1, "\"proof_surface\": \"dispatcher_userland\""));
    try testing.expect(std.mem.containsAtLeast(u8, dispatcher_cert, 1, "\"dispatcher_manifest_schema\": \"dispatcher_v1\""));
    try testing.expect(std.mem.containsAtLeast(u8, dispatcher_cert, 1, "\"switch_count\": 1"));
    try testing.expect(std.mem.containsAtLeast(u8, dispatcher_cert, 1, "\"case_count\": 12"));
    try testing.expect(std.mem.containsAtLeast(
        u8,
        dispatcher_cert,
        1,
        "current_dispatcher_planner_reference_matches",
    ));
    try testing.expect(std.mem.containsAtLeast(u8, dispatcher_cert, 1, "current_dispatcher_manifest_rows_match"));
    try testing.expect(std.mem.containsAtLeast(u8, dispatcher_cert, 1, "current_dispatcher_builder_correct"));
    try testing.expect(std.mem.containsAtLeast(u8, dispatcher_cert, 1, "current_dispatcher_network_matches"));
    try testing.expect(std.mem.containsAtLeast(u8, dispatcher_cert, 1, "current_dispatcher_unknown_selectors_revert"));
    try testing.expect(std.mem.containsAtLeast(u8, dispatcher_cert, 1, "\"bytecode_sha256\""));
    try testing.expect(std.mem.containsAtLeast(
        u8,
        dispatcher_cert,
        1,
        "\"dispatcher_bytecode_report_sha256\"",
    ));
    try testing.expect(std.mem.containsAtLeast(u8, dispatcher_cert, 1, "\"bytecode_templates_valid\": true"));
    try testing.expect(!std.mem.containsAtLeast(u8, dispatcher_cert, 1, "current_dispatcher_table_rows_guarded"));
    try testing.expect(!std.mem.containsAtLeast(u8, dispatcher_cert, 1, "current_dispatcher_table_rows_covered"));
    try testing.expect(!std.mem.containsAtLeast(u8, dispatcher_cert, 1, "current_dispatcher_planner_searches_valid"));
    try testing.expect(!std.mem.containsAtLeast(u8, dispatcher_cert, 1, "current_dispatcher_manifest_base_rows_match"));
}

test "lean proofs verify a two-switch dispatcher network through bytecode" {
    std.Io.Dir.cwd().access(std.testing.io, ORA_BINARY_REL, .{}) catch |err| switch (err) {
        error.FileNotFound => return error.SkipZigTest,
        else => return err,
    };

    const allocator = testing.allocator;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    var source = std.Io.Writer.Allocating.init(allocator);
    defer source.deinit();
    try source.writer.writeAll(
        \\comptime const std = @import("std");
        \\
        \\contract DispatcherNetworkGate {
        \\    storage var counter: u256;
        \\
        \\    pub fn bump() requires counter < std.constants.U256_MAX {
        \\        counter = counter + 1;
        \\    }
        \\
        \\    pub fn reset() {
        \\        counter = 0;
        \\    }
        \\
    );
    for (0..14) |index| {
        try source.writer.print(
            \\    pub fn v{d}() -> u256 {{
            \\        return {d};
            \\    }}
            \\
        , .{ index, 200 + index });
    }
    try source.writer.writeAll("}\n");
    try tmp.dir.writeFile(std.testing.io, .{
        .sub_path = "dispatcher_network_gate.ora",
        .data = source.written(),
    });
    const source_path = try pathFromTmpAlloc(
        allocator,
        tmp,
        "dispatcher_network_gate.ora",
    );
    defer allocator.free(source_path);
    const out_dir = try pathFromTmpAlloc(allocator, tmp, "out");
    defer allocator.free(out_dir);

    const result = try runProcessWithLeanPath(allocator, &.{
        ORA_BINARY_REL,
        "emit",
        "--emit=bytecode",
        "--out-dir",
        out_dir,
        "--lean-proofs",
        source_path,
    });
    defer allocator.free(result.stdout);
    defer allocator.free(result.stderr);
    try expectExited(result, 0);

    const certificate_path = try pathFromTmpAlloc(
        allocator,
        tmp,
        "out/dispatcher_network_gate.lean.dispatcher.proof.json",
    );
    defer allocator.free(certificate_path);
    const certificate = try readFileAllocForTest(allocator, certificate_path);
    defer allocator.free(certificate);
    try testing.expect(std.mem.containsAtLeast(u8, certificate, 1, "\"switch_count\": 2"));
    try testing.expect(std.mem.containsAtLeast(u8, certificate, 1, "current_dispatcher_network_matches"));
    try testing.expect(std.mem.containsAtLeast(u8, certificate, 1, "current_dispatcher_unknown_selectors_revert"));
    try testing.expect(std.mem.containsAtLeast(u8, certificate, 1, "\"bytecode_templates_valid\": true"));

    // Retain the unbounded increment as an overflow rejection under default verification.
    const unbounded = try std.mem.replaceOwned(u8, allocator, source.written(), " requires counter < std.constants.U256_MAX", "");
    defer allocator.free(unbounded);
    try tmp.dir.writeFile(std.testing.io, .{ .sub_path = "unbounded.ora", .data = unbounded });
    const unbounded_path = try pathFromTmpAlloc(allocator, tmp, "unbounded.ora");
    defer allocator.free(unbounded_path);
    const rejected_path = try pathFromTmpAlloc(allocator, tmp, "rejected");
    defer allocator.free(rejected_path);
    const rejected = try runProcessWithLeanPath(allocator, &.{ ORA_BINARY_REL, "emit", "--emit=bytecode", "--out-dir", rejected_path, unbounded_path });
    defer allocator.free(rejected.stdout);
    defer allocator.free(rejected.stderr);
    try expectExited(rejected, 1);
    try testing.expect(std.mem.indexOf(u8, rejected.stderr, "checked addition overflow") != null);
    try testing.expect(std.mem.indexOf(u8, rejected.stdout, "counterexample:") != null);
    try testing.expectError(error.FileNotFound, tmp.dir.access(std.testing.io, "rejected/unbounded.hex", .{}));
}

test "lean proofs discharge multiplicative dispatcher planner trace end to end" {
    std.Io.Dir.cwd().access(std.testing.io, ORA_BINARY_REL, .{}) catch |err| switch (err) {
        error.FileNotFound => return error.SkipZigTest,
        else => return err,
    };

    const allocator = testing.allocator;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const out_dir = try pathFromTmpAlloc(allocator, tmp, "out");
    defer allocator.free(out_dir);

    const result = try runProcessWithLeanPath(allocator, &.{
        ORA_BINARY_REL,
        "emit",
        "--emit=bytecode",
        "--out-dir",
        out_dir,
        "--lean-proofs",
        "tests/conformance/dispatcher_dense_routing.ora",
    });
    defer allocator.free(result.stdout);
    defer allocator.free(result.stderr);
    try expectExited(result, 0);
    try testing.expect(std.mem.containsAtLeast(u8, result.stdout, 1, "Lean dispatcher userland proof accepted") or
        std.mem.containsAtLeast(u8, result.stderr, 1, "Lean dispatcher userland proof accepted"));

    const certificate_path = try pathFromTmpAlloc(
        allocator,
        tmp,
        "out/dispatcher_dense_routing.lean.dispatcher.proof.json",
    );
    defer allocator.free(certificate_path);
    const certificate = try readFileAllocForTest(allocator, certificate_path);
    defer allocator.free(certificate);
    try testing.expect(std.mem.containsAtLeast(
        u8,
        certificate,
        1,
        "current_dispatcher_planner_reference_matches",
    ));
    try testing.expect(std.mem.containsAtLeast(
        u8,
        certificate,
        1,
        "current_dispatcher_manifest_rows_match",
    ));
    try testing.expect(std.mem.containsAtLeast(
        u8,
        certificate,
        1,
        "current_dispatcher_builder_correct",
    ));
    try testing.expect(!std.mem.containsAtLeast(
        u8,
        certificate,
        1,
        "current_dispatcher_planner_searches_valid",
    ));
    try testing.expect(!std.mem.containsAtLeast(
        u8,
        certificate,
        1,
        "current_dispatcher_planner_core_matches",
    ));
}

test "dispatcher switch fact extraction reports missing named default" {
    const source =
        \\contract DispatcherMissingNamedDefault {
        \\    pub fn a0() -> u256 { return 0; }
        \\    pub fn a1() -> u256 { return 1; }
        \\    pub fn a2() -> u256 { return 2; }
        \\    pub fn a3() -> u256 { return 3; }
        \\}
    ;

    var compilation = try compileText(source);
    defer compilation.deinit();

    const hir_result = try compilation.db.lowerToHir(compilation.root_module_id);
    try testing.expect(hir_result.diagnostics.isEmpty());
    try testing.expect(mlir.oraConvertToSIR(hir_result.context, hir_result.module.raw_module, false));
    try testing.expect(mlir.oraBuildSIRDispatcher(hir_result.context, hir_result.module.raw_module));
    try testing.expect(mlir.oraTestStripFirstSIRSelectorSwitchDefaultBlockName(hir_result.module.raw_module));

    const facts_ref = mlir.oraExtractSIRDispatcherSwitchFacts(hir_result.context, hir_result.module.raw_module);
    defer if (facts_ref.data != null) mlir.oraStringRefFree(facts_ref);
    try testing.expect(facts_ref.data != null);

    const facts = facts_ref.data[0..facts_ref.length];
    try testing.expect(std.mem.containsAtLeast(u8, facts, 1, "\"guarded\":false"));
}

test "dispatcher table Lean checker rejects false generated row" {
    const allocator = testing.allocator;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();

    const module_suffix = try moduleSuffixFromTmp(allocator, tmp);
    defer allocator.free(module_suffix);
    const fixture_dir = try std.fmt.allocPrint(allocator, "formal/Ora/DispatcherFailureFixture/{s}", .{module_suffix});
    defer allocator.free(fixture_dir);
    try std.Io.Dir.cwd().createDirPath(std.testing.io, fixture_dir);
    defer std.Io.Dir.cwd().deleteTree(std.testing.io, fixture_dir) catch {};
    defer std.Io.Dir.cwd().deleteDir(std.testing.io, "formal/Ora/DispatcherFailureFixture") catch {};

    const bad_path = try std.fmt.allocPrint(allocator, "{s}/Bad.lean", .{fixture_dir});
    defer allocator.free(bad_path);
    var source = std.Io.Writer.Allocating.init(allocator);
    defer source.deinit();
    try source.writer.writeAll(
        \\import Ora.DispatcherTableSync
        \\
        \\namespace Ora.DispatcherFailureFixture
        \\
        \\def currentDispatcherTableRows :
        \\    List Ora.DispatcherTableSync.RawRow :=
        \\  [
        \\  { name := "bad_dense",
        \\    plan := { strategy := "dense", denseKind := "bit_window", tableSlots := 1,
        \\      indexBits := 0, indexShift := 0, mulConstant := 0 },
        \\    trace := { policy := "balanced", preconditionsMet := false, linearScore := 0,
        \\      multiplicativeSearches := [], denseCandidateCount := 0, bestDense := none,
        \\      sparseCandidateCount := 0, bestSparse := none },
        \\    cases := [{ selector := 1, label := "entry", index := 0, hasNamedDefault := false }] }
        \\  ]
        \\
        \\def currentDispatcherRowsHaveNamedDefault : Bool :=
        \\  Ora.DispatcherTableSync.Gate.rowsHaveNamedDefault currentDispatcherTableRows
        \\
        \\theorem current_dispatcher_table_rows_have_named_default :
        \\    currentDispatcherRowsHaveNamedDefault = true := by decide
        \\
        \\end Ora.DispatcherFailureFixture
        \\
    );
    try std.Io.Dir.cwd().writeFile(std.testing.io, .{ .sub_path = bad_path, .data = source.written() });

    const result = try runLeanFileForTest(allocator, bad_path);
    defer allocator.free(result.stdout);
    defer allocator.free(result.stderr);
    try expectExited(result, 1);
    try testing.expect(std.mem.containsAtLeast(u8, result.stderr, 1, "decide") or
        std.mem.containsAtLeast(u8, result.stdout, 1, "decide"));
    try testing.expect(!std.mem.containsAtLeast(u8, result.stderr, 1, "unknown identifier"));
    try testing.expect(!std.mem.containsAtLeast(u8, result.stderr, 1, "Unknown constant"));

    const bad_coverage_path = try std.fmt.allocPrint(allocator, "{s}/BadCoverage.lean", .{fixture_dir});
    defer allocator.free(bad_coverage_path);
    var bad_coverage = std.Io.Writer.Allocating.init(allocator);
    defer bad_coverage.deinit();
    try bad_coverage.writer.writeAll(
        \\import Ora.DispatcherTableSync
        \\
        \\namespace Ora.DispatcherFailureFixture.BadCoverage
        \\
        \\def currentDispatcherTableRows :
        \\    List Ora.DispatcherTableSync.RawRow :=
        \\  [
        \\  { name := "bad_coverage",
        \\    plan := { strategy := "dense", denseKind := "bit_window", tableSlots := 2,
        \\      indexBits := 1, indexShift := 0, mulConstant := 0 },
        \\    trace := { policy := "balanced", preconditionsMet := true, linearScore := 0,
        \\      multiplicativeSearches := [], denseCandidateCount := 0, bestDense := none,
        \\      sparseCandidateCount := 0, bestSparse := none },
        \\    cases := [
        \\      { selector := 1, label := "first", index := 0, hasNamedDefault := true },
        \\      { selector := 2, label := "second", index := 0, hasNamedDefault := true }] }
        \\  ]
        \\
        \\def currentDispatcherRowsCovered : Bool :=
        \\  Ora.DispatcherTableSync.Gate.rowsCovered currentDispatcherTableRows
        \\
        \\theorem current_dispatcher_table_rows_covered :
        \\    currentDispatcherRowsCovered = true := by decide
        \\
        \\end Ora.DispatcherFailureFixture.BadCoverage
        \\
    );
    try std.Io.Dir.cwd().writeFile(std.testing.io, .{ .sub_path = bad_coverage_path, .data = bad_coverage.written() });

    const bad_coverage_result = try runLeanFileForTest(allocator, bad_coverage_path);
    defer allocator.free(bad_coverage_result.stdout);
    defer allocator.free(bad_coverage_result.stderr);
    try expectExited(bad_coverage_result, 1);
    try testing.expect(std.mem.containsAtLeast(u8, bad_coverage_result.stderr, 1, "decide") or
        std.mem.containsAtLeast(u8, bad_coverage_result.stdout, 1, "decide"));
    try testing.expect(!std.mem.containsAtLeast(u8, bad_coverage_result.stderr, 1, "unknown identifier"));
    try testing.expect(!std.mem.containsAtLeast(u8, bad_coverage_result.stderr, 1, "Unknown constant"));
}

test "dispatcher userland Lean checker rejects invalid plan facts" {
    const allocator = testing.allocator;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();

    const module_suffix = try moduleSuffixFromTmp(allocator, tmp);
    defer allocator.free(module_suffix);
    const fixture_dir = try std.fmt.allocPrint(allocator, "formal/Ora/DispatcherUserlandFailureFixture/{s}", .{module_suffix});
    defer allocator.free(fixture_dir);
    try std.Io.Dir.cwd().createDirPath(std.testing.io, fixture_dir);
    defer std.Io.Dir.cwd().deleteTree(std.testing.io, fixture_dir) catch {};
    defer std.Io.Dir.cwd().deleteDir(std.testing.io, "formal/Ora/DispatcherUserlandFailureFixture") catch {};

    const bad_kind_path = try std.fmt.allocPrint(allocator, "{s}/BadKind.lean", .{fixture_dir});
    defer allocator.free(bad_kind_path);
    var bad_kind = std.Io.Writer.Allocating.init(allocator);
    defer bad_kind.deinit();
    try bad_kind.writer.writeAll(
        \\import Ora.DispatcherTableSync
        \\
        \\namespace Ora.DispatcherUserlandFailureFixture.BadKind
        \\
        \\def currentDispatcherTableRows :
        \\    List Ora.DispatcherTableSync.RawRow :=
        \\  [
        \\  { name := "bad_kind",
        \\    plan := { strategy := "dense", denseKind := "xor", tableSlots := 32,
        \\      indexBits := 5, indexShift := 0, mulConstant := 0 },
        \\    trace := { policy := "balanced", preconditionsMet := false, linearScore := 0,
        \\      multiplicativeSearches := [], denseCandidateCount := 0, bestDense := none,
        \\      sparseCandidateCount := 0, bestSparse := none },
        \\    cases := [{ selector := 1, label := "entry", index := 1, hasNamedDefault := true }] }
        \\  ]
        \\
        \\def currentDispatcherPlansAdmissible : Bool :=
        \\  Ora.DispatcherTableSync.Gate.plansAdmissible currentDispatcherTableRows
        \\
        \\theorem current_dispatcher_plans_admissible :
        \\    currentDispatcherPlansAdmissible = true := by decide
        \\
        \\end Ora.DispatcherUserlandFailureFixture.BadKind
        \\
    );
    try std.Io.Dir.cwd().writeFile(std.testing.io, .{ .sub_path = bad_kind_path, .data = bad_kind.written() });

    const bad_kind_result = try runLeanFileForTest(allocator, bad_kind_path);
    defer allocator.free(bad_kind_result.stdout);
    defer allocator.free(bad_kind_result.stderr);
    try expectExited(bad_kind_result, 1);
    try testing.expect(std.mem.containsAtLeast(u8, bad_kind_result.stderr, 1, "decide") or
        std.mem.containsAtLeast(u8, bad_kind_result.stdout, 1, "decide"));

    const bad_shape_path = try std.fmt.allocPrint(allocator, "{s}/BadShape.lean", .{fixture_dir});
    defer allocator.free(bad_shape_path);
    var bad_shape = std.Io.Writer.Allocating.init(allocator);
    defer bad_shape.deinit();
    try bad_shape.writer.writeAll(
        \\import Ora.DispatcherTableSync
        \\
        \\namespace Ora.DispatcherUserlandFailureFixture.BadShape
        \\
        \\def currentDispatcherTableRows :
        \\    List Ora.DispatcherTableSync.RawRow :=
        \\  [
        \\  { name := "bad_shape",
        \\    plan := { strategy := "dense", denseKind := "bit_window", tableSlots := 64,
        \\      indexBits := 5, indexShift := 0, mulConstant := 0 },
        \\    trace := { policy := "balanced", preconditionsMet := false, linearScore := 0,
        \\      multiplicativeSearches := [], denseCandidateCount := 0, bestDense := none,
        \\      sparseCandidateCount := 0, bestSparse := none },
        \\    cases := [
        \\      { selector := 1, label := "entry", index := 1, hasNamedDefault := true },
        \\      { selector := 2, label := "other", index := 2, hasNamedDefault := true }] }
        \\  ]
        \\
        \\def currentDispatcherPlanShapesValid : Bool :=
        \\  Ora.DispatcherTableSync.Gate.planShapesValid currentDispatcherTableRows
        \\
        \\theorem current_dispatcher_plan_shapes_valid :
        \\    currentDispatcherPlanShapesValid = true := by decide
        \\
        \\end Ora.DispatcherUserlandFailureFixture.BadShape
        \\
    );
    try std.Io.Dir.cwd().writeFile(std.testing.io, .{ .sub_path = bad_shape_path, .data = bad_shape.written() });

    const bad_shape_result = try runLeanFileForTest(allocator, bad_shape_path);
    defer allocator.free(bad_shape_result.stdout);
    defer allocator.free(bad_shape_result.stderr);
    try expectExited(bad_shape_result, 1);
    try testing.expect(std.mem.containsAtLeast(u8, bad_shape_result.stderr, 1, "decide") or
        std.mem.containsAtLeast(u8, bad_shape_result.stdout, 1, "decide"));

    const bad_planner_path = try std.fmt.allocPrint(allocator, "{s}/BadPlanner.lean", .{fixture_dir});
    defer allocator.free(bad_planner_path);
    var bad_planner = std.Io.Writer.Allocating.init(allocator);
    defer bad_planner.deinit();
    try bad_planner.writer.writeAll(
        \\import Ora.DispatcherTableSync
        \\
        \\namespace Ora.DispatcherUserlandFailureFixture.BadPlanner
        \\
        \\def currentDispatcherTableRows :
        \\    List Ora.DispatcherTableSync.RawRow :=
        \\  [
        \\  { name := "bad_planner",
        \\    plan := { strategy := "dense", denseKind := "bit_window", tableSlots := 4,
        \\      indexBits := 2, indexShift := 0, mulConstant := 0 },
        \\    trace := { policy := "balanced", preconditionsMet := false, linearScore := 2760,
        \\      multiplicativeSearches := [], denseCandidateCount := 0, bestDense := none,
        \\      sparseCandidateCount := 0, bestSparse := none },
        \\    cases := [
        \\      { selector := 0, label := "a", index := 0, hasNamedDefault := false },
        \\      { selector := 1, label := "b", index := 1, hasNamedDefault := false },
        \\      { selector := 2, label := "c", index := 2, hasNamedDefault := false },
        \\      { selector := 3, label := "d", index := 3, hasNamedDefault := false }] }
        \\  ]
        \\
        \\def currentDispatcherPlannerMatches : Bool :=
        \\  Ora.DispatcherTableSync.Gate.plannerMatches currentDispatcherTableRows
        \\
        \\theorem current_dispatcher_planner_matches :
        \\    currentDispatcherPlannerMatches = true := by decide
        \\
        \\end Ora.DispatcherUserlandFailureFixture.BadPlanner
        \\
    );
    try std.Io.Dir.cwd().writeFile(std.testing.io, .{ .sub_path = bad_planner_path, .data = bad_planner.written() });

    const bad_planner_result = try runLeanFileForTest(allocator, bad_planner_path);
    defer allocator.free(bad_planner_result.stdout);
    defer allocator.free(bad_planner_result.stderr);
    try expectExited(bad_planner_result, 1);
    try testing.expect(std.mem.containsAtLeast(u8, bad_planner_result.stderr, 1, "decide") or
        std.mem.containsAtLeast(u8, bad_planner_result.stdout, 1, "decide"));

    const bad_reference_path = try std.fmt.allocPrint(
        allocator,
        "{s}/BadReference.lean",
        .{fixture_dir},
    );
    defer allocator.free(bad_reference_path);
    var bad_reference = std.Io.Writer.Allocating.init(allocator);
    defer bad_reference.deinit();
    try bad_reference.writer.writeAll(
        \\import Ora.DispatcherTableSync
        \\
        \\namespace Ora.DispatcherUserlandFailureFixture.BadReference
        \\
        \\def currentDispatcherTableRows :
        \\    List Ora.DispatcherTableSync.RawRow :=
        \\  [
        \\  { name := "bad_reference",
        \\    plan := { strategy := "dense", denseKind := "bit_window", tableSlots := 4,
        \\      indexBits := 2, indexShift := 0, mulConstant := 0 },
        \\    trace := { policy := "balanced", preconditionsMet := true, linearScore := 0,
        \\      multiplicativeSearches := [], denseCandidateCount := 0, bestDense := none,
        \\      sparseCandidateCount := 0, bestSparse := none },
        \\    cases := [
        \\      { selector := 1447852734, label := "a", index := 0, hasNamedDefault := true },
        \\      { selector := 832491607, label := "b", index := 1, hasNamedDefault := true },
        \\      { selector := 3309386683, label := "c", index := 2, hasNamedDefault := true },
        \\      { selector := 2561671559, label := "d", index := 3, hasNamedDefault := true }] }
        \\  ]
        \\
        \\def currentDispatcherPlannerReferenceMatches : Bool :=
        \\  Ora.DispatcherTableSync.Gate.plannerReferenceMatches currentDispatcherTableRows
        \\
        \\theorem current_dispatcher_planner_reference_matches :
        \\    currentDispatcherPlannerReferenceMatches = true := by decide
        \\
        \\end Ora.DispatcherUserlandFailureFixture.BadReference
        \\
    );
    try std.Io.Dir.cwd().writeFile(std.testing.io, .{
        .sub_path = bad_reference_path,
        .data = bad_reference.written(),
    });

    const bad_reference_result = try runLeanFileForTest(allocator, bad_reference_path);
    defer allocator.free(bad_reference_result.stdout);
    defer allocator.free(bad_reference_result.stderr);
    try expectExited(bad_reference_result, 1);
    try testing.expect(std.mem.containsAtLeast(u8, bad_reference_result.stderr, 1, "decide") or
        std.mem.containsAtLeast(u8, bad_reference_result.stdout, 1, "decide"));
    try testing.expect(!std.mem.containsAtLeast(u8, bad_reference_result.stderr, 1, "unknown identifier"));

    const bad_search_path = try std.fmt.allocPrint(allocator, "{s}/BadSearch.lean", .{fixture_dir});
    defer allocator.free(bad_search_path);
    var bad_search = std.Io.Writer.Allocating.init(allocator);
    defer bad_search.deinit();
    try bad_search.writer.writeAll(
        \\import Ora.DispatcherTableSync
        \\
        \\namespace Ora.DispatcherUserlandFailureFixture.BadSearch
        \\
        \\def currentDispatcherTableRows :
        \\    List Ora.DispatcherTableSync.RawRow :=
        \\  [
        \\  { name := "bad_search",
        \\    plan := { strategy := "linear", denseKind := "", tableSlots := 4,
        \\      indexBits := 0, indexShift := 0, mulConstant := 0 },
        \\    trace := { policy := "balanced", preconditionsMet := true, linearScore := 2760,
        \\      multiplicativeSearches :=
        \\        [{ tableSlots := 4, selected := some 1,
        \\           rejected := [{ constant := 2462723855, firstCase := 0, secondCase := 1 }] },
        \\         { tableSlots := 8, selected := some 1,
        \\           rejected := [{ constant := 2462723855, firstCase := 0, secondCase := 2 }] },
        \\         { tableSlots := 16, selected := some 0, rejected := [] },
        \\         { tableSlots := 32, selected := some 0, rejected := [] },
        \\         { tableSlots := 64, selected := some 0, rejected := [] },
        \\         { tableSlots := 128, selected := some 0, rejected := [] },
        \\         { tableSlots := 256, selected := some 0, rejected := [] }],
        \\      denseCandidateCount := 44,
        \\      bestDense := some ({ strategy := "dense", denseKind := "bit_window",
        \\        tableSlots := 4, indexBits := 2, indexShift := 8, mulConstant := 0 }, 4350),
        \\      sparseCandidateCount := 56,
        \\      bestSparse := some ({ strategy := "sparse", denseKind := "",
        \\        tableSlots := 4, indexBits := 2, indexShift := 8, mulConstant := 0 }, 4430) },
        \\    cases := [
        \\      { selector := 1447852734, label := "lin0_", index := 0, hasNamedDefault := true },
        \\      { selector := 832491607, label := "lin1_", index := 1, hasNamedDefault := true },
        \\      { selector := 3309386683, label := "lin2_", index := 2, hasNamedDefault := true },
        \\      { selector := 2561671559, label := "lin3_", index := 3, hasNamedDefault := true }] }
        \\  ]
        \\
        \\def currentDispatcherPlannerSearchesValid : Bool :=
        \\  Ora.DispatcherTableSync.Gate.plannerSearchesValid currentDispatcherTableRows
        \\
        \\theorem current_dispatcher_planner_searches_valid :
        \\    currentDispatcherPlannerSearchesValid = true := by decide
        \\
        \\end Ora.DispatcherUserlandFailureFixture.BadSearch
        \\
    );
    try std.Io.Dir.cwd().writeFile(std.testing.io, .{
        .sub_path = bad_search_path,
        .data = bad_search.written(),
    });

    const bad_search_result = try runLeanFileForTest(allocator, bad_search_path);
    defer allocator.free(bad_search_result.stdout);
    defer allocator.free(bad_search_result.stderr);
    try expectExited(bad_search_result, 1);
    try testing.expect(std.mem.containsAtLeast(u8, bad_search_result.stderr, 1, "decide") or
        std.mem.containsAtLeast(u8, bad_search_result.stdout, 1, "decide"));
    try testing.expect(!std.mem.containsAtLeast(u8, bad_search_result.stderr, 1, "unknown identifier"));
    try testing.expect(!std.mem.containsAtLeast(u8, bad_search_result.stderr, 1, "Unknown constant"));
}

test "B4 unknown diagnostic rejects Lean recipe for unsupported semantic type" {
    std.Io.Dir.cwd().access(std.testing.io, ORA_BINARY_REL, .{}) catch |err| switch (err) {
        error.FileNotFound => return error.SkipZigTest,
        else => return err,
    };

    const allocator = testing.allocator;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();

    const source =
        \\contract B4UnsupportedLeanType {
        \\    pub fn same(x: address) -> bool
        \\        ensures x == x
        \\    {
        \\        return true;
        \\    }
        \\}
    ;
    try tmp.dir.writeFile(std.testing.io, .{ .sub_path = "b4_unsupported_lean_type.ora", .data = source });

    const source_path = try pathFromTmpAlloc(allocator, tmp, "b4_unsupported_lean_type.ora");
    defer allocator.free(source_path);
    const out_dir = try pathFromTmpAlloc(allocator, tmp, "b4-out");
    defer allocator.free(out_dir);

    const result = try runOraWithForcedUnknown(allocator, "obligation:2", &.{
        ORA_BINARY_REL,
        "emit",
        "--emit=smt-report",
        "--out-dir",
        out_dir,
        source_path,
    });
    defer allocator.free(result.stdout);
    defer allocator.free(result.stderr);

    try expectExited(result, 1);
    try testing.expect(std.mem.containsAtLeast(u8, result.stdout, 1, "Lean proof recipe unavailable for some required userland obligations"));
    try testing.expect(std.mem.containsAtLeast(u8, result.stdout, 1, "unsupported Lean semantic type `address`"));
    try testing.expect(!std.mem.containsAtLeast(u8, result.stdout, 1, "Lean proof recipe for required userland obligations"));
    try testing.expectError(error.FileNotFound, tmp.dir.access(std.testing.io, "b4-out/b4_unsupported_lean_type.lean.obligations.lean", .{}));
}

test "B5 unknown diagnostic reports Lean projection coverage when no recipe is available" {
    std.Io.Dir.cwd().access(std.testing.io, ORA_BINARY_REL, .{}) catch |err| switch (err) {
        error.FileNotFound => return error.SkipZigTest,
        else => return err,
    };

    const allocator = testing.allocator;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();

    const out_dir = try pathFromTmpAlloc(allocator, tmp, "b5-out");
    defer allocator.free(out_dir);

    // The first obligation is the source-level assert. It intentionally has
    // no Lean-dischargeable query; later arithmetic-safety rows are now
    // identity-bound and therefore are not valid coverage-gap probes.
    const result = try runOraWithForcedUnknown(allocator, "obligation:2", &.{
        ORA_BINARY_REL,
        "emit",
        "--emit=smt-report",
        "--out-dir",
        out_dir,
        "ora-example/formal/obligation_report_logical.ora",
    });
    defer allocator.free(result.stdout);
    defer allocator.free(result.stderr);

    try expectExited(result, 1);
    try testing.expect(std.mem.containsAtLeast(u8, result.stdout, 1, "Lean proof recipe unavailable for some required userland obligations"));
    try testing.expect(std.mem.containsAtLeast(u8, result.stdout, 1, "unmatched UNKNOWN prepared rows: 1"));
    try testing.expect(std.mem.containsAtLeast(u8, result.stdout, 1, "formula projection: term="));
    try testing.expect(std.mem.containsAtLeast(u8, result.stdout, 1, "origin_value="));
    try testing.expect(std.mem.containsAtLeast(u8, result.stdout, 1, "term_ratio_basis_points="));
    try testing.expect(std.mem.containsAtLeast(u8, result.stdout, 1, "no matching Lean-dischargeable obligation query"));
    try testing.expect(!std.mem.containsAtLeast(u8, result.stdout, 1, "Lean proof recipe for required userland obligations"));
}

test "formal query vocabulary matches Z3 prepared query vocabulary" {
    inline for (@typeInfo(z3_verification.QueryKind).@"enum".field_names) |field| {
        const z3_kind: z3_verification.QueryKind = @fromBackingInt(@intCast(@backingInt(@field(z3_verification.QueryKind, field))));
        const label = z3_verification.formalQueryKindLabel(z3_kind);
        const formal_kind = std.meta.stringToEnum(obligation.VerificationQueryKind, label) orelse return error.TestUnexpectedResult;
        try testing.expectEqualStrings(label, @tagName(formal_kind));
    }

    inline for (@typeInfo(obligation.VerificationQueryKind).@"enum".field_names) |field| {
        const formal_kind: obligation.VerificationQueryKind = @fromBackingInt(@intCast(@backingInt(@field(obligation.VerificationQueryKind, field))));
        // These are constructed from the canonical LoopVC after the legacy Z3
        // verifier has returned; neither is part of its prepared-query enum.
        if (comptime formal_kind == .loop_base or formal_kind == .loop_induction) continue;
        var found = false;
        inline for (@typeInfo(z3_verification.QueryKind).@"enum".field_names) |z3_field| {
            const z3_kind: z3_verification.QueryKind = @fromBackingInt(@intCast(@backingInt(@field(z3_verification.QueryKind, z3_field))));
            found = found or std.mem.eql(u8, field, z3_verification.formalQueryKindLabel(z3_kind));
        }
        try testing.expect(found);
    }
}

test "formal query fragments and solver logic match Z3 vocabulary" {
    inline for (@typeInfo(z3_verification.QueryFragment).@"enum".field_names) |field| {
        const z3_fragment: z3_verification.QueryFragment = @fromBackingInt(@intCast(@backingInt(@field(z3_verification.QueryFragment, field))));
        const label = z3_verification.formalQueryFragmentLabel(z3_fragment);
        const formal_fragment = std.meta.stringToEnum(obligation.VerificationQueryFragment, label) orelse return error.TestUnexpectedResult;
        try testing.expectEqualStrings(label, @tagName(formal_fragment));
    }

    inline for (@typeInfo(z3_verification.QuerySolverLogic).@"enum".field_names) |field| {
        const z3_logic: z3_verification.QuerySolverLogic = @fromBackingInt(@intCast(@backingInt(@field(z3_verification.QuerySolverLogic, field))));
        const label = z3_verification.formalQuerySolverLogicLabel(z3_logic);
        const formal_logic = std.meta.stringToEnum(obligation.VerificationSolverLogic, label) orelse return error.TestUnexpectedResult;
        try testing.expectEqualStrings(label, @tagName(formal_logic));
    }
}

test "formal logical and assumption roles match Z3 annotation vocabulary" {
    inline for (@typeInfo(z3_verification.AnnotationKind).@"enum".field_names) |field| {
        const annotation: z3_verification.AnnotationKind = @fromBackingInt(@intCast(@backingInt(@field(z3_verification.AnnotationKind, field))));
        if (z3_verification.formalLogicalRoleLabel(annotation)) |label| {
            const role = std.meta.stringToEnum(obligation.LogicalRole, label) orelse return error.TestUnexpectedResult;
            try testing.expectEqualStrings(label, @tagName(role));
        }
        if (z3_verification.formalAssumptionKindLabel(annotation)) |label| {
            const kind = std.meta.stringToEnum(obligation.AssumptionKind, label) orelse return error.TestUnexpectedResult;
            try testing.expectEqualStrings(label, @tagName(kind));
        }
    }
}

test "formal imported pending obligation roles match Z3 source vocabulary" {
    inline for (@typeInfo(z3_verification.PendingObligationSourceKind).@"enum".field_names) |field| {
        const source: z3_verification.PendingObligationSourceKind = @fromBackingInt(@intCast(@backingInt(@field(z3_verification.PendingObligationSourceKind, field))));
        if (z3_verification.formalPendingObligationRoleLabel(source)) |label| {
            const role = std.meta.stringToEnum(obligation.LogicalRole, label) orelse return error.TestUnexpectedResult;
            try testing.expectEqualStrings(label, @tagName(role));
        }
    }
}

test "formal pending constraint assumptions match Z3 source vocabulary" {
    inline for (@typeInfo(z3_verification.PendingConstraintSourceKind).@"enum".field_names) |field| {
        const source: z3_verification.PendingConstraintSourceKind = @fromBackingInt(@intCast(@backingInt(@field(z3_verification.PendingConstraintSourceKind, field))));
        if (z3_verification.formalPendingConstraintAssumptionKindLabel(source)) |label| {
            const kind = std.meta.stringToEnum(obligation.AssumptionKind, label) orelse return error.TestUnexpectedResult;
            try testing.expectEqualStrings(label, @tagName(kind));
        }
    }
}

test "formal MLIR manifest query summary matches Z3 prepared query summary" {
    const h = createContext();
    defer destroyContext(h);

    const text =
        \\module {
        \\  func.func @checked(%flag: i1) {
        \\    "ora.requires"(%flag) : (i1) -> ()
        \\    "ora.ensures"(%flag) : (i1) -> ()
        \\    "ora.assert"(%flag) <{message = "must hold"}> : (i1) -> ()
        \\    "ora.assume"(%flag) : (i1) -> ()
        \\    "ora.refinement_guard"(%flag) <{message = "guard"}> : (i1) -> ()
        \\    func.return
        \\  }
        \\}
    ;

    const module = try parseModule(h.ctx, text);
    defer mlir.oraModuleDestroy(module);

    const guard_op = findFirstOpByName(module, "ora.refinement_guard") orelse return error.TestUnexpectedResult;
    mlir.oraOperationSetAttributeByName(
        guard_op,
        strRef("ora.guard_id"),
        mlir.oraStringAttrCreate(h.ctx, strRef("guard:checked:flag")),
    );

    var formal_result = try obligation_from_mlir.collect(testing.allocator, module, .{});
    defer formal_result.deinit();

    var pass = try z3_verification.VerificationPass.init(testing.allocator);
    defer pass.deinit();
    const z3_summary = try pass.collectPreparedQuerySummary(module);

    try expectSummaryMatchesZ3PreparedQueries(
        obligation.VerificationQuerySummary.fromQueries(formal_result.set.queries),
        z3_summary,
    );
}

test "formal Z3 prepared query manifest maps into canonical queries" {
    const h = createContext();
    defer destroyContext(h);

    const text =
        \\module {
        \\  func.func @checked(%flag: i1) {
        \\    "ora.requires"(%flag) : (i1) -> ()
        \\    "ora.ensures"(%flag) : (i1) -> ()
        \\    "ora.assert"(%flag) <{message = "must hold"}> : (i1) -> ()
        \\    "ora.assume"(%flag) : (i1) -> ()
        \\    "ora.refinement_guard"(%flag) <{message = "guard"}> : (i1) -> ()
        \\    func.return
        \\  }
        \\}
    ;

    const module = try parseModule(h.ctx, text);
    defer mlir.oraModuleDestroy(module);

    const guard_op = findFirstOpByName(module, "ora.refinement_guard") orelse return error.TestUnexpectedResult;
    mlir.oraOperationSetAttributeByName(
        guard_op,
        strRef("ora.guard_id"),
        mlir.oraStringAttrCreate(h.ctx, strRef("guard:checked:flag")),
    );

    var pass = try z3_verification.VerificationPass.init(testing.allocator);
    defer pass.deinit();
    var z3_manifest = try pass.collectPreparedQueryManifest(module);
    defer z3_manifest.deinit();

    var formal_result = try obligation_from_z3.collectPreparedQueries(testing.allocator, z3_manifest.rows);
    defer formal_result.deinit();

    const summary = obligation.VerificationQuerySummary.fromQueries(formal_result.set.queries);
    try testing.expectEqual(@as(u32, 1), summary.base);
    try testing.expectEqual(@as(u32, 2), summary.obligation);
    try testing.expectEqual(@as(u32, 1), summary.guard_satisfy);
    try testing.expectEqual(@as(u32, 1), summary.guard_violate);
    try testing.expectEqual(@as(usize, 1), countQueryRole(formal_result.set, .ensures));
    try testing.expectEqual(@as(usize, 1), countQueryRole(formal_result.set, .contract_invariant));

    var guarded_queries: usize = 0;
    for (formal_result.set.queries) |query| {
        try testing.expectEqual(obligation.VerificationBackend.z3, query.backend);
        try testing.expect(query.smtlib_hash != null);
        try testing.expect(query.constraint_count > 0 or query.kind == .base);
        if (query.kind == .guard_satisfy or query.kind == .guard_violate) {
            guarded_queries += 1;
            try testing.expectEqualStrings("guard:checked:flag", query.guard_id orelse return error.TestUnexpectedResult);
        }
    }
    try testing.expectEqual(@as(usize, 2), guarded_queries);
}

test "formal Z3 prepared query results overlay MLIR manifest queries" {
    const h = createContext();
    defer destroyContext(h);

    const text =
        \\module {
        \\  func.func @checked(%flag: i1) {
        \\    "ora.requires"(%flag) : (i1) -> ()
        \\    "ora.ensures"(%flag) : (i1) -> ()
        \\    "ora.assert"(%flag) <{message = "must hold"}> : (i1) -> ()
        \\    "ora.assume"(%flag) : (i1) -> ()
        \\    "ora.refinement_guard"(%flag) <{message = "guard"}> : (i1) -> ()
        \\    func.return
        \\  }
        \\}
    ;

    const module = try parseModule(h.ctx, text);
    defer mlir.oraModuleDestroy(module);

    const guard_op = findFirstOpByName(module, "ora.refinement_guard") orelse return error.TestUnexpectedResult;
    mlir.oraOperationSetAttributeByName(
        guard_op,
        strRef("ora.guard_id"),
        mlir.oraStringAttrCreate(h.ctx, strRef("guard:checked:flag")),
    );

    var formal_result = try obligation_from_mlir.collect(testing.allocator, module, .{});
    defer formal_result.deinit();

    var pass = try z3_verification.VerificationPass.init(testing.allocator);
    defer pass.deinit();
    var verification_result = try pass.runVerificationPass(module);
    defer verification_result.deinit();
    var report = try pass.buildSmtReport(module, "/tmp/formal_overlay.ora", &verification_result);
    defer report.deinit(testing.allocator);

    const query_manifest = report.query_manifest orelse return error.TestUnexpectedResult;
    var overlay = try obligation_from_z3.overlayPreparedQueryResults(testing.allocator, formal_result.set, query_manifest.rows);
    defer overlay.deinit();

    try testing.expectEqual(formal_result.set.queries.len, overlay.set.queries.len);
    try testing.expect(!overlay.set.hasBlockingDiagnostic());
    for (overlay.set.queries) |query| {
        try testing.expectEqual(obligation.VerificationBackend.z3, query.backend);
        try testing.expect(query.smtlib_hash != null);
        try testing.expect(query.result != null);
    }
}

test "refinement provenance keeps exact predicates and branch checks bound to emitted guards" {
    var tmp = testing.tmpDir(.{});
    defer tmp.cleanup();
    try tmp.dir.writeFile(testing.io, .{ .sub_path = "provenance.ora", .data =
        \\contract Provenance {
        \\    storage slot: MinValue<u256, 100>;
        \\    pub fn param(x: MinValue<u256, 100>) -> MinValue<u256, 100> {
        \\        let y: MinValue<u256, 100> = x; return y;
        \\    }
        \\    pub fn castReturn(x: u256) -> MinValue<u256, 100> {
        \\        return @cast(MinValue<u256, 100>, x);
        \\    }
        \\    pub fn roundTrip(x: MinValue<u256, 100>) -> MinValue<u256, 100> {
        \\        let y: u256 = x; return @cast(MinValue<u256, 100>, y);
        \\    }
        \\    pub fn stronger(x: MinValue<u256, 100>) -> MinValue<u256, 200> { return x; }
        \\    pub fn carrier(x: NonZero<u256>) -> BasisPoints<u256> { return x; }
        \\    pub fn unchecked(x: u256) -> MinValue<u256, 100> { return x; }
        \\    pub fn narrow(x: MinValue<u256, 100>) -> u8 { return @truncate(MinValue<u8, 100>, x); }
        \\    pub fn signed(x: MinValue<u256, 100>) -> i256 { return @bitCast(MinValue<i256, 100>, x); }
        \\    pub fn wide(x: MinValue<u8, 100>) -> u256 { return @cast(MinValue<u256, 100>, x); }
        \\    pub fn arms(x: u256, flag: bool) -> u256 {
        \\        if (flag) { let y: MinValue<u256, 100> = x; return y; }
        \\        else { let z: MinValue<u256, 100> = x; return z; }
        \\    }
        \\    pub fn join(x: u256, flag: bool) -> u256 {
        \\        if (flag) { let y: MinValue<u256, 100> = x; }
        \\        let z: MinValue<u256, 100> = x; return z;
        \\    }
        \\    pub fn fresh() -> MinValue<u256, 100> {
        \\        let a: MinValue<u256, 100> = slot;
        \\        return slot;
        \\    }
        \\    pub fn rebind(x: u256, y: u256) -> u256 {
        \\        var a: MinValue<u256, 100> = x; a = y; return a;
        \\    }
        \\}
    });
    const path = try pathFromTmpAlloc(testing.allocator, tmp, "provenance.ora");
    defer testing.allocator.free(path);
    var compilation = try compilePackage(path);
    defer compilation.deinit();
    const hir = try compilation.db.lowerToHir(compilation.root_module_id);
    try testing.expect(hir.diagnostics.isEmpty());
    var formal = try obligation_from_mlir.collect(testing.allocator, hir.module.raw_module, .{});
    defer formal.deinit();
    try testing.expect(!formal.set.hasBlockingDiagnostic());
    const Case = struct { name: []const u8, parameters: usize, conversions: usize };
    for ([_]Case{
        .{ .name = "param", .parameters = 1, .conversions = 0 },
        .{ .name = "castReturn", .parameters = 0, .conversions = 1 },
        .{ .name = "roundTrip", .parameters = 1, .conversions = 0 },
        .{ .name = "stronger", .parameters = 1, .conversions = 1 },
        .{ .name = "carrier", .parameters = 1, .conversions = 1 },
        .{ .name = "unchecked", .parameters = 0, .conversions = 1 },
        .{ .name = "narrow", .parameters = 1, .conversions = 1 },
        .{ .name = "signed", .parameters = 1, .conversions = 1 },
        .{ .name = "wide", .parameters = 1, .conversions = 1 },
        .{ .name = "arms", .parameters = 0, .conversions = 2 },
        .{ .name = "join", .parameters = 0, .conversions = 2 },
        .{ .name = "fresh", .parameters = 0, .conversions = 2 },
        .{ .name = "rebind", .parameters = 0, .conversions = 2 },
    }) |case| {
        errdefer std.debug.print("provenance owner={s}\n", .{case.name});
        var parameters: usize = 0;
        var conversions: usize = 0;
        for (formal.set.queries) |query| {
            if (query.kind != .guard_violate or !std.mem.eql(u8, query.owner.function.name, case.name)) continue;
            switch (query.artifact_policy) {
                .runtime_checked_input => parameters += 1,
                .blocks_verified_artifacts => conversions += 1,
                else => return error.TestUnexpectedResult,
            }
        }
        try testing.expectEqual(case.parameters, parameters);
        try testing.expectEqual(case.conversions, conversions);
    }
}

test "refinement conversions bind distinct source guards to real queries and reject corruption" {
    var tmp = testing.tmpDir(.{});
    defer tmp.cleanup();
    try tmp.dir.writeFile(testing.io, .{ .sub_path = "conversions.ora", .data =
        \\contract Conversions {
        \\    pub fn array(x: u256, y: u256) -> u256
        \\        requires(x >= 100 and y >= 100)
        \\    {
        \\        let a: [MinValue<u256, 100>; 2] = [x, y];
        \\        return a[0];
        \\    }
        \\    pub fn other(x: u256) -> MinValue<u256, 100>
        \\        requires(x >= 100)
        \\    { return @cast(MinValue<u256, 100>, x); }
        \\}
    });
    const path = try pathFromTmpAlloc(testing.allocator, tmp, "conversions.ora");
    defer testing.allocator.free(path);
    var compilation = try compilePackage(path);
    defer compilation.deinit();
    const hir = try compilation.db.lowerToHir(compilation.root_module_id);
    try testing.expect(hir.diagnostics.isEmpty());
    var formal = try obligation_from_mlir.collect(testing.allocator, hir.module.raw_module, .{});
    defer formal.deinit();
    try testing.expect(!formal.set.hasBlockingDiagnostic());
    const original = try test_helpers.z3FormalQueryBindingsFromFormalForTest(testing.allocator, formal.query_bindings);
    defer testing.allocator.free(original);
    var guard_indices: std.ArrayList(usize) = .empty;
    defer guard_indices.deinit(testing.allocator);
    for (original, 0..) |binding, i| {
        if (std.mem.eql(u8, binding.kind, "guard_violate")) try guard_indices.append(testing.allocator, i);
    }
    try testing.expectEqual(@as(usize, 3), guard_indices.items.len);
    const first = original[guard_indices.items[0]];
    const second = original[guard_indices.items[1]];
    const foreign = original[guard_indices.items[2]];
    // Two elements in one declaration have separate source identity and guards.
    try testing.expect(first.source_op_id != second.source_op_id);
    try testing.expect(!std.mem.eql(u8, first.guard_id.?, second.guard_id.?));
    try testing.expectEqualStrings("array", obligation.findById(formal.set.queries, first.query_id).?.owner.function.name);
    try testing.expectEqualStrings("other", obligation.findById(formal.set.queries, foreign.query_id).?.owner.function.name);
    for (std.meta.tags(enum { intact, wrong_source, wrong_owner, wrong_role, duplicate_binding, missing_binding, duplicate_row, missing_row })) |mutation| {
        errdefer std.debug.print("refinement conversion mutation={s}\n", .{@tagName(mutation)});
        var bindings: std.ArrayList(z3_verification.FormalQueryBinding) = .empty;
        defer bindings.deinit(testing.allocator);
        try bindings.appendSlice(testing.allocator, original);
        const target_index = guard_indices.items[0];
        switch (mutation) {
            .wrong_source => bindings.items[target_index].source_op_id = second.source_op_id,
            .wrong_owner => bindings.items[target_index].source_op_id = foreign.source_op_id,
            .wrong_role => bindings.items[target_index].logical_role = "assert",
            .duplicate_binding => try bindings.append(testing.allocator, first),
            .missing_binding => _ = bindings.orderedRemove(target_index),
            else => {},
        }
        var pass = try z3_verification.VerificationPass.init(testing.allocator);
        defer pass.deinit();
        pass.setVerifyMode(.Full);
        pass.setFormalQueryBindings(bindings.items);
        var verified = try pass.runVerificationPassPreparedSequential(hir.module.raw_module);
        defer verified.deinit();
        try testing.expect(verified.success);
        var report = try pass.buildSmtReport(hir.module.raw_module, path, &verified);
        defer report.deinit(testing.allocator);
        const manifest = report.query_manifest orelse return error.TestUnexpectedResult;
        var rows: std.ArrayList(z3_verification.PreparedQueryManifestRow) = .empty;
        defer rows.deinit(testing.allocator);
        for (manifest.rows) |row| {
            if (row.formal_query_id == first.query_id and mutation == .missing_row) continue;
            try rows.append(testing.allocator, row);
            if (row.formal_query_id == first.query_id and mutation == .duplicate_row) try rows.append(testing.allocator, row);
        }
        var overlay = try obligation_from_z3.overlayPreparedQueryResults(testing.allocator, formal.set, rows.items);
        defer overlay.deinit();
        if (mutation == .intact) {
            try testing.expect(overlay.set.artifactDecision().isAllowed());
            var guards: usize = 0;
            for (overlay.set.queries) |query| {
                try testing.expectEqual(obligation.VerificationBackend.z3, query.backend);
                const digest = query.smtlib_hash orelse return error.TestUnexpectedResult;
                try testing.expect(digest != 0);
                const expected: obligation.VerificationQueryStatus = switch (query.kind) {
                    .base, .guard_satisfy => .sat,
                    else => .unsat,
                };
                try testing.expectEqual(expected, query.result.?.status);
                if (query.kind != .guard_violate) continue;
                guards += 1;
                try testing.expectEqual(obligation.ArtifactPolicy.blocks_verified_artifacts, query.artifact_policy);
                var matches: usize = 0;
                for (manifest.rows) |row| {
                    if (row.formal_query_id != query.id) continue;
                    matches += 1;
                    try testing.expectEqual(z3_verification.FormalMatchStatus.matched, row.formal_match_status);
                    try testing.expectEqual(z3_verification.QueryResultStatus.unsat, row.result_status.?);
                    try testing.expectEqual(digest, row.smtlib_hash);
                    try testing.expectEqualSlices(obligation.Id, query.assumption_ids, row.formal_assumption_ids);
                    try testing.expectEqualSlices(obligation.Id, query.obligation_ids, row.formal_obligation_ids);
                }
                try testing.expectEqual(@as(usize, 1), matches);
            }
            try testing.expectEqual(@as(usize, 3), guards);
        } else {
            try testing.expect(!overlay.set.artifactDecision().isAllowed());
            var unmatched = false;
            for (overlay.set.diagnostics) |diagnostic| {
                if (diagnostic.kind == .unmatched_report_row) unmatched = true;
            }
            try testing.expect(unmatched);
            const target = obligation.findById(overlay.set.queries, first.query_id).?;
            try testing.expect(target.result == null);
            try testing.expect(target.smtlib_hash == null);
        }
    }
}

test "C8 source assertions bind real Z3 queries and reject identity corruption" {
    const h = createContext();
    defer destroyContext(h);
    const module = try parseModule(h.ctx,
        \\module {
        \\  func.func @checked(%flag: i1) {
        \\    "ora.requires"(%flag) : (i1) -> ()
        \\    "ora.assert"(%flag) <{message = "first source assertion"}> : (i1) -> () loc("c8.ora":10:1)
        \\    "ora.assert"(%flag) <{message = "second source assertion"}> : (i1) -> () loc("c8.ora":10:1)
        \\    "ora.assert"(%flag) <{message = "checked addition overflow"}> {ora.verification_context = "arithmetic_safety"} : (i1) -> ()
        \\    cf.assert %flag, "existing invariant category"
        \\    "ora.assert"(%flag) <{message = "postcondition"}> {ora.ensures = true} : (i1) -> ()
        \\    func.return
        \\  }
        \\  func.func @other(%flag: i1) {
        \\    "ora.requires"(%flag) : (i1) -> ()
        \\    "ora.assert"(%flag) <{message = "foreign assertion"}> : (i1) -> () loc("c8.ora":10:1)
        \\    func.return
        \\  }
        \\}
    );
    defer mlir.oraModuleDestroy(module);
    var formal = try obligation_from_mlir.collect(testing.allocator, module, .{});
    defer formal.deinit();
    try testing.expect(!formal.set.hasBlockingDiagnostic());
    const original = try test_helpers.z3FormalQueryBindingsFromFormalForTest(testing.allocator, formal.query_bindings);
    defer testing.allocator.free(original);
    var assertion_indices: std.ArrayList(usize) = .empty;
    defer assertion_indices.deinit(testing.allocator);
    for (original, 0..) |binding, index| {
        if (std.mem.eql(u8, binding.logical_role orelse "", "assert")) try assertion_indices.append(testing.allocator, index);
    }
    try testing.expectEqual(@as(usize, 3), assertion_indices.items.len);
    const target_index = assertion_indices.items[0];
    const target_id = original[target_index].query_id;
    const foreign_query = obligation.findById(formal.set.queries, original[assertion_indices.items[2]].query_id).?;
    try testing.expectEqualStrings("other", foreign_query.owner.function.name);

    for (std.meta.tags(enum { intact, wrong_source, wrong_owner, wrong_role, duplicate_binding, missing_binding, duplicate_row, missing_row })) |mutation| {
        errdefer std.debug.print("C8 mutation={s}\n", .{@tagName(mutation)});
        var bindings: std.ArrayList(z3_verification.FormalQueryBinding) = .empty;
        defer bindings.deinit(testing.allocator);
        try bindings.appendSlice(testing.allocator, original);
        switch (mutation) {
            .wrong_source => bindings.items[target_index].source_op_id = original[assertion_indices.items[1]].source_op_id,
            // Same operation shape and identical source location, but a different
            // function's actual operation identity cannot supply this proof.
            .wrong_owner => bindings.items[target_index].source_op_id = original[assertion_indices.items[2]].source_op_id,
            .wrong_role => bindings.items[target_index].logical_role = "contract_invariant",
            .duplicate_binding => try bindings.append(testing.allocator, original[target_index]),
            .missing_binding => _ = bindings.orderedRemove(target_index),
            else => {},
        }
        var pass = try z3_verification.VerificationPass.init(testing.allocator);
        defer pass.deinit();
        pass.setVerifyMode(.Full);
        pass.setFormalQueryBindings(bindings.items);
        var verified = try pass.runVerificationPassPreparedSequential(module);
        defer verified.deinit();
        try testing.expect(verified.success);
        var report = try pass.buildSmtReport(module, "c8_assertions.ora", &verified);
        defer report.deinit(testing.allocator);
        const manifest = report.query_manifest orelse return error.TestUnexpectedResult;
        var rows: std.ArrayList(z3_verification.PreparedQueryManifestRow) = .empty;
        defer rows.deinit(testing.allocator);
        for (manifest.rows) |row| {
            if (row.formal_query_id == target_id and mutation == .missing_row) continue;
            try rows.append(testing.allocator, row);
            if (row.formal_query_id == target_id and mutation == .duplicate_row) try rows.append(testing.allocator, row);
        }
        var overlay = try obligation_from_z3.overlayPreparedQueryResults(testing.allocator, formal.set, rows.items);
        defer overlay.deinit();
        if (mutation == .intact) {
            try testing.expect(!overlay.set.hasBlockingDiagnostic());
            try testing.expect(overlay.set.artifactDecision().isAllowed());
            var assertions: usize = 0;
            var arithmetic: usize = 0;
            var invariants: usize = 0;
            var ensures: usize = 0;
            for (overlay.set.queries) |query| {
                try testing.expectEqual(obligation.VerificationBackend.z3, query.backend);
                const digest = query.smtlib_hash orelse return error.TestUnexpectedResult;
                try testing.expect(digest != 0);
                if (query.kind == .base) {
                    try testing.expectEqual(obligation.VerificationQueryStatus.sat, query.result.?.status);
                    continue;
                }
                try testing.expectEqual(obligation.VerificationQueryStatus.unsat, query.result.?.status);
                var matches: usize = 0;
                for (manifest.rows) |row| {
                    if (row.formal_query_id != query.id) continue;
                    matches += 1;
                    try testing.expectEqual(z3_verification.FormalMatchStatus.matched, row.formal_match_status);
                    try testing.expectEqual(z3_verification.QueryResultStatus.unsat, row.result_status.?);
                    try testing.expectEqual(digest, row.smtlib_hash);
                    try testing.expectEqualSlices(obligation.Id, query.assumption_ids, row.formal_assumption_ids);
                    try testing.expectEqualSlices(obligation.Id, query.obligation_ids, row.formal_obligation_ids);
                }
                try testing.expectEqual(@as(usize, 1), matches);
                switch (query.logical_role orelse return error.TestUnexpectedResult) {
                    .assert => assertions += 1,
                    .arithmetic_safety => arithmetic += 1,
                    .contract_invariant => invariants += 1,
                    .ensures => ensures += 1,
                    else => return error.TestUnexpectedResult,
                }
            }
            try testing.expectEqual(@as(usize, 3), assertions);
            try testing.expectEqual(@as(usize, 1), arithmetic);
            try testing.expectEqual(@as(usize, 1), invariants);
            try testing.expectEqual(@as(usize, 1), ensures);
        } else {
            try testing.expect(!overlay.set.artifactDecision().isAllowed());
            var unmatched = false;
            for (overlay.set.diagnostics) |diagnostic| {
                if (diagnostic.kind == .unmatched_report_row) unmatched = true;
            }
            try testing.expect(unmatched);
            const target = obligation.findById(overlay.set.queries, target_id).?;
            try testing.expect(target.result == null);
            try testing.expect(target.smtlib_hash == null);
        }
    }
}

test "formal Z3 overlay ignores only clean unmatched proved rows" {
    const clean_rows = [_]z3_verification.PreparedQueryManifestRow{
        .{
            .kind = .Obligation,
            .function_name = "checked",
            .file = "test.ora",
            .line = 1,
            .column = 1,
            .result_status = .unsat,
        },
    };
    var clean_overlay = try obligation_from_z3.overlayPreparedQueryResults(testing.allocator, .{}, &clean_rows);
    defer clean_overlay.deinit();
    try testing.expectEqual(@as(usize, 0), clean_overlay.set.diagnostics.len);
    try testing.expect(!clean_overlay.set.hasBlockingDiagnostic());

    const vacuous_rows = [_]z3_verification.PreparedQueryManifestRow{
        .{
            .kind = .Obligation,
            .function_name = "checked",
            .file = "test.ora",
            .line = 1,
            .column = 1,
            .result_status = .unsat,
            .vacuous = true,
        },
    };
    var vacuous_overlay = try obligation_from_z3.overlayPreparedQueryResults(testing.allocator, .{}, &vacuous_rows);
    defer vacuous_overlay.deinit();
    try testing.expectEqual(@as(usize, 1), vacuous_overlay.set.diagnostics.len);
    try testing.expect(vacuous_overlay.set.hasBlockingDiagnostic());
}

test "formal canonical Z3 quantifier row matches live prepared hash" {
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();

    const source =
        \\contract CanonicalQuantifierParity {
        \\    pub fn bounded() -> bool
        \\        ensures true
        \\    {
        \\        assume(forall i: u256 => (forall j: u256 => i <= j));
        \\        return true;
        \\    }
        \\}
    ;
    try tmp.dir.writeFile(std.testing.io, .{ .sub_path = "canonical_quantifier_parity.ora", .data = source });
    const source_path = try pathFromTmpAlloc(testing.allocator, tmp, "canonical_quantifier_parity.ora");
    defer testing.allocator.free(source_path);

    var compilation = try compilePackage(source_path);
    defer compilation.deinit();

    const hir_result = try compilation.db.lowerToHir(compilation.root_module_id);
    try testing.expect(hir_result.diagnostics.isEmpty());

    var formal_result = try obligation_from_mlir.collect(testing.allocator, hir_result.module.raw_module, .{});
    defer formal_result.deinit();

    var pass = try z3_verification.VerificationPass.init(testing.allocator);
    defer pass.deinit();
    const z3_bindings = try test_helpers.z3FormalQueryBindingsFromFormalForTest(testing.allocator, formal_result.query_bindings);
    defer testing.allocator.free(z3_bindings);
    pass.setFormalQueryBindings(z3_bindings);
    var z3_manifest = try pass.collectPreparedQueryManifest(hir_result.module.raw_module);
    defer z3_manifest.deinit();

    var overlay = try obligation_from_z3.overlayPreparedQueryResults(testing.allocator, formal_result.set, z3_manifest.rows);
    defer overlay.deinit();

    var z3_ctx = try z3_verification.Z3Context.init(testing.allocator);
    defer z3_ctx.deinit();
    var adapter = obligation_to_z3.Adapter.init(&z3_ctx, testing.allocator, overlay.set);

    var saw_quantified_row = false;
    for (overlay.set.queries) |query| {
        if (query.kind != .obligation or query.obligation_ids.len != 1) continue;
        const target_id = query.obligation_ids[0];
        var target: ?obligation.Obligation = null;
        for (overlay.set.obligations) |item| {
            if (item.id == target_id) {
                target = item;
                break;
            }
        }
        _ = target orelse continue;
        if (obligation_to_z3.queryCanonicalPromotionShape(overlay.set, query) != .quantified_formula) continue;

        var has_nested_quantified_assumption = false;
        for (query.assumption_ids) |assumption_id| {
            for (overlay.set.assumptions) |assumption| {
                if (assumption.id != assumption_id) continue;
                const term_id = switch (assumption.formula orelse continue) {
                    .term => |id| id,
                    .origin_value => continue,
                };
                if (termHasNestedQuantifier(overlay.set, term_id, 0)) {
                    has_nested_quantified_assumption = true;
                }
            }
        }
        if (!has_nested_quantified_assumption) continue;

        saw_quantified_row = true;
        try testing.expect(query.canonical_smt_annotation_pure);
        try testing.expect(query.smtlib_hash != null);
        try testing.expectEqual(
            obligation_to_z3.CanonicalPromotionShape.quantified_formula,
            obligation_to_z3.queryCanonicalPromotionShape(overlay.set, query).?,
        );
        try testing.expect(!obligation_to_z3.queryCanonicalRequiredModePromoted(overlay.set, query));

        const canonical = try adapter.queryHashForRow(query);
        try testing.expectEqual(query.constraint_count, canonical.constraint_count);
        try testing.expectEqual(query.smtlib_hash.?, canonical.smtlib_hash);
    }
    try testing.expect(saw_quantified_row);
}

test "formal canonical Z3 goal forall row matches live skolemized hash" {
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();

    const source =
        \\contract CanonicalGoalQuantifierParity {
        \\    pub fn bounded() -> bool
        \\        ensures (forall i: u256 => (forall j: u256 => i <= j))
        \\    {
        \\        return true;
        \\    }
        \\}
    ;
    try tmp.dir.writeFile(std.testing.io, .{ .sub_path = "canonical_goal_quantifier_parity.ora", .data = source });
    const source_path = try pathFromTmpAlloc(testing.allocator, tmp, "canonical_goal_quantifier_parity.ora");
    defer testing.allocator.free(source_path);

    var compilation = try compilePackage(source_path);
    defer compilation.deinit();

    const hir_result = try compilation.db.lowerToHir(compilation.root_module_id);
    try testing.expect(hir_result.diagnostics.isEmpty());

    var formal_result = try obligation_from_mlir.collect(testing.allocator, hir_result.module.raw_module, .{});
    defer formal_result.deinit();

    var pass = try z3_verification.VerificationPass.init(testing.allocator);
    defer pass.deinit();
    const z3_bindings = try test_helpers.z3FormalQueryBindingsFromFormalForTest(testing.allocator, formal_result.query_bindings);
    defer testing.allocator.free(z3_bindings);
    pass.setFormalQueryBindings(z3_bindings);
    var z3_manifest = try pass.collectPreparedQueryManifest(hir_result.module.raw_module);
    defer z3_manifest.deinit();

    var overlay = try obligation_from_z3.overlayPreparedQueryResults(testing.allocator, formal_result.set, z3_manifest.rows);
    defer overlay.deinit();

    var z3_ctx = try z3_verification.Z3Context.init(testing.allocator);
    defer z3_ctx.deinit();
    var adapter = obligation_to_z3.Adapter.init(&z3_ctx, testing.allocator, overlay.set);

    var saw_goal_forall_row = false;
    for (overlay.set.queries) |query| {
        if (query.kind != .obligation or query.obligation_ids.len != 1) continue;
        const target_id = query.obligation_ids[0];
        var target: ?obligation.Obligation = null;
        for (overlay.set.obligations) |item| {
            if (item.id == target_id) {
                target = item;
                break;
            }
        }
        const obligation_row = target orelse continue;
        if (obligation_row.kind != .logical) continue;
        const term_id = switch (obligation_row.kind.logical.formula) {
            .term => |id| id,
            .origin_value => continue,
        };
        if (!termHasNestedQuantifier(overlay.set, term_id, 0)) continue;

        saw_goal_forall_row = true;
        try testing.expect(query.canonical_smt_annotation_pure);
        try testing.expect(query.smtlib_hash != null);
        try testing.expectEqual(
            obligation_to_z3.CanonicalPromotionShape.quantified_formula,
            obligation_to_z3.queryCanonicalPromotionShape(overlay.set, query).?,
        );
        try testing.expect(!obligation_to_z3.queryCanonicalRequiredModePromoted(overlay.set, query));

        const canonical = try adapter.queryHashForRow(query);
        try testing.expectEqual(query.constraint_count, canonical.constraint_count);
        try testing.expectEqual(query.smtlib_hash.?, canonical.smtlib_hash);
    }
    try testing.expect(saw_goal_forall_row);
}

test "formal canonical Z3 treats requires runtime mirror as non-pure" {
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();

    const source =
        \\contract CanonicalRequiresMirrorBoundary {
        \\    pub fn checked_sub(x: u256, y: u256) -> u256
        \\        requires x >= y
        \\        ensures result == x - y
        \\    {
        \\        return x - y;
        \\    }
        \\}
    ;
    try tmp.dir.writeFile(std.testing.io, .{ .sub_path = "canonical_requires_mirror_boundary.ora", .data = source });
    const source_path = try pathFromTmpAlloc(testing.allocator, tmp, "canonical_requires_mirror_boundary.ora");
    defer testing.allocator.free(source_path);

    var compilation = try compilePackage(source_path);
    defer compilation.deinit();

    const hir_result = try compilation.db.lowerToHir(compilation.root_module_id);
    try testing.expect(hir_result.diagnostics.isEmpty());

    var formal_result = try obligation_from_mlir.collect(testing.allocator, hir_result.module.raw_module, .{});
    defer formal_result.deinit();

    var pass = try z3_verification.VerificationPass.init(testing.allocator);
    defer pass.deinit();
    const z3_bindings = try test_helpers.z3FormalQueryBindingsFromFormalForTest(testing.allocator, formal_result.query_bindings);
    defer testing.allocator.free(z3_bindings);
    pass.setFormalQueryBindings(z3_bindings);
    var z3_manifest = try pass.collectPreparedQueryManifest(hir_result.module.raw_module);
    defer z3_manifest.deinit();

    var overlay = try obligation_from_z3.overlayPreparedQueryResults(testing.allocator, formal_result.set, z3_manifest.rows);
    defer overlay.deinit();

    const ensures_query = try findEnsuresQuery(overlay.set);
    try testing.expect(ensures_query.smtlib_hash != null);
    try testing.expect(!ensures_query.canonical_smt_annotation_pure);
}

test "formal Z3 overlay does not require prepared rows for structural effect frames" {
    const obligation_ids = [_]obligation.Id{1};
    const obligations = [_]obligation.Obligation{.{
        .id = 1,
        .owner = .{ .function = .{ .name = "effect_surface" } },
        .source = .generated(),
        .phase = .ora_mlir,
        .origin = .source,
        .kind = .{ .effect_frame = .{
            .relation = .read_preserved_by_key_evidence,
        } },
    }};
    const queries = [_]obligation.VerificationQuery{.{
        .id = 2,
        .owner = .{ .function = .{ .name = "effect_surface" } },
        .source = .generated(),
        .phase = .ora_mlir,
        .origin = .source,
        .kind = .obligation,
        .obligation_ids = &obligation_ids,
    }};
    const set: obligation.ObligationSet = .{
        .obligations = &obligations,
        .queries = &queries,
    };

    var overlay = try obligation_from_z3.overlayPreparedQueryResults(testing.allocator, set, &.{});
    defer overlay.deinit();

    try testing.expectEqual(@as(usize, 0), overlay.set.diagnostics.len);
    try testing.expectEqual(@as(usize, 1), overlay.set.queries.len);
    try testing.expectEqual(obligation.VerificationBackend.unspecified, overlay.set.queries[0].backend);
    try testing.expect(overlay.set.queries[0].result == null);
}

test "formal Z3 adapter proves canonical term obligation from assumptions" {
    var z3_ctx = try z3_verification.Z3Context.init(testing.allocator);
    defer z3_ctx.deinit();

    const terms = [_]obligation.Term{
        .{ .variable = .{ .free = .{ .id = .{ .file_id = 0, .pattern_id = 0 }, .name = "balance", .ty = .{ .spelling = "u256" } } } },
        .{ .variable = .{ .free = .{ .id = .{ .file_id = 0, .pattern_id = 1 }, .name = "amount", .ty = .{ .spelling = "u256" } } } },
        .{ .binary = .{ .op = .ge, .lhs = 0, .rhs = 1, .ty = builtinTypeRef(.u256) } },
    };
    const assumptions = [_]obligation.Assumption{
        .{
            .id = 1,
            .owner = .{ .function = .{ .name = "transfer" } },
            .source = .generated(),
            .phase = .sema,
            .origin = .{ .sema_fact = .{ .kind = "requires", .ordinal = 0 } },
            .kind = .requires,
            .formula = .{ .term = 2 },
        },
    };
    const obligations = [_]obligation.Obligation{
        .{
            .id = 2,
            .owner = .{ .function = .{ .name = "transfer" } },
            .source = .generated(),
            .phase = .sema,
            .origin = .{ .sema_fact = .{ .kind = "resource_source_sufficient", .ordinal = 0 } },
            .kind = .{ .logical = .{
                .role = .invariant,
                .formula = .{ .term = 2 },
            } },
        },
    };
    const assumption_ids = [_]obligation.Id{1};
    const obligation_ids = [_]obligation.Id{2};
    const queries = [_]obligation.VerificationQuery{
        .{
            .id = 3,
            .owner = .{ .function = .{ .name = "transfer" } },
            .source = .generated(),
            .phase = .report,
            .origin = .source,
            .backend = .z3,
            .kind = .obligation,
            .obligation_ids = &obligation_ids,
            .assumption_ids = &assumption_ids,
        },
    };
    const set: obligation.ObligationSet = .{
        .assumptions = &assumptions,
        .obligations = &obligations,
        .queries = &queries,
        .terms = &terms,
    };

    var adapter = obligation_to_z3.Adapter.init(&z3_ctx, testing.allocator, set);
    try testing.expectEqual(obligation_to_z3.CheckStatus.proved, try adapter.checkObligation(2));
}

test "formal Z3 adapter distinguishes signed and unsigned comparison tags" {
    var z3_ctx = try z3_verification.Z3Context.init(testing.allocator);
    defer z3_ctx.deinit();

    const max_u256 = "115792089237316195423570985008687907853269984665640564039457584007913129639935";
    const terms = [_]obligation.Term{
        .{ .int_lit = .{ .value = max_u256, .ty = builtinTypeRef(.i256) } },
        .{ .int_lit = .{ .value = "0", .ty = builtinTypeRef(.i256) } },
        .{ .binary = .{ .op = .slt, .lhs = 0, .rhs = 1, .ty = builtinTypeRef(.i256) } },
        .{ .binary = .{ .op = .lt, .lhs = 0, .rhs = 1, .ty = builtinTypeRef(.u256) } },
    };
    const obligations = [_]obligation.Obligation{
        .{
            .id = 1,
            .owner = .{ .function = .{ .name = "signed" } },
            .source = .generated(),
            .phase = .sema,
            .origin = .{ .sema_fact = .{ .kind = "signed_cmp", .ordinal = 0 } },
            .kind = .{ .logical = .{
                .role = .ensures,
                .formula = .{ .term = 2 },
            } },
        },
        .{
            .id = 2,
            .owner = .{ .function = .{ .name = "unsigned" } },
            .source = .generated(),
            .phase = .sema,
            .origin = .{ .sema_fact = .{ .kind = "unsigned_cmp", .ordinal = 0 } },
            .kind = .{ .logical = .{
                .role = .ensures,
                .formula = .{ .term = 3 },
            } },
        },
    };
    const signed_ids = [_]obligation.Id{1};
    const unsigned_ids = [_]obligation.Id{2};
    const queries = [_]obligation.VerificationQuery{
        .{
            .id = 3,
            .owner = .{ .function = .{ .name = "signed" } },
            .source = .generated(),
            .phase = .report,
            .origin = .source,
            .backend = .z3,
            .kind = .obligation,
            .obligation_ids = &signed_ids,
        },
        .{
            .id = 4,
            .owner = .{ .function = .{ .name = "unsigned" } },
            .source = .generated(),
            .phase = .report,
            .origin = .source,
            .backend = .z3,
            .kind = .obligation,
            .obligation_ids = &unsigned_ids,
        },
    };
    const set: obligation.ObligationSet = .{
        .obligations = &obligations,
        .queries = &queries,
        .terms = &terms,
    };

    var adapter = obligation_to_z3.Adapter.init(&z3_ctx, testing.allocator, set);
    try testing.expectEqual(obligation_to_z3.CheckStatus.proved, try adapter.checkObligation(1));
    try testing.expectEqual(obligation_to_z3.CheckStatus.disproved, try adapter.checkObligation(2));
}

test "formal Z3 adapter totalizes unsigned div and rem by zero" {
    const terms = [_]obligation.Term{
        .{ .int_lit = .{ .value = "7", .ty = builtinTypeRef(.u256) } },
        .{ .int_lit = .{ .value = "0", .ty = builtinTypeRef(.u256) } },
        .{ .binary = .{ .op = .div, .lhs = 0, .rhs = 1, .ty = builtinTypeRef(.u256) } },
        .{ .binary = .{ .op = .mod, .lhs = 0, .rhs = 1, .ty = builtinTypeRef(.u256) } },
        .{ .int_lit = .{ .value = "0", .ty = builtinTypeRef(.u256) } },
        .{ .binary = .{ .op = .eq, .lhs = 2, .rhs = 4 } },
        .{ .binary = .{ .op = .eq, .lhs = 3, .rhs = 4 } },
    };

    try testing.expectEqual(obligation_to_z3.CheckStatus.proved, try checkFormalZ3TermObligation(&terms, 5));
    try testing.expectEqual(obligation_to_z3.CheckStatus.proved, try checkFormalZ3TermObligation(&terms, 6));
}

test "formal Z3 adapter totalizes signed min divided by negative one" {
    const max_u256 = "115792089237316195423570985008687907853269984665640564039457584007913129639935";
    const min_i256 = "57896044618658097711785492504343953926634992332820282019728792003956564819968";
    const terms = [_]obligation.Term{
        .{ .int_lit = .{ .value = min_i256, .ty = builtinTypeRef(.i256) } },
        .{ .int_lit = .{ .value = max_u256, .ty = builtinTypeRef(.i256) } },
        .{ .binary = .{ .op = .div, .lhs = 0, .rhs = 1, .ty = builtinTypeRef(.i256) } },
        .{ .binary = .{ .op = .mod, .lhs = 0, .rhs = 1, .ty = builtinTypeRef(.i256) } },
        .{ .int_lit = .{ .value = min_i256, .ty = builtinTypeRef(.i256) } },
        .{ .int_lit = .{ .value = "0", .ty = builtinTypeRef(.i256) } },
        .{ .binary = .{ .op = .eq, .lhs = 2, .rhs = 4 } },
        .{ .binary = .{ .op = .eq, .lhs = 3, .rhs = 5 } },
    };

    try testing.expectEqual(obligation_to_z3.CheckStatus.proved, try checkFormalZ3TermObligation(&terms, 6));
    try testing.expectEqual(obligation_to_z3.CheckStatus.proved, try checkFormalZ3TermObligation(&terms, 7));
}

test "formal Z3 adapter signed remainder sign follows dividend" {
    const max_u256 = "115792089237316195423570985008687907853269984665640564039457584007913129639935";
    const neg_ten = "115792089237316195423570985008687907853269984665640564039457584007913129639926";
    const neg_three = "115792089237316195423570985008687907853269984665640564039457584007913129639933";
    const terms = [_]obligation.Term{
        .{ .int_lit = .{ .value = neg_ten, .ty = builtinTypeRef(.i256) } },
        .{ .int_lit = .{ .value = "3", .ty = builtinTypeRef(.i256) } },
        .{ .binary = .{ .op = .mod, .lhs = 0, .rhs = 1, .ty = builtinTypeRef(.i256) } },
        .{ .int_lit = .{ .value = max_u256, .ty = builtinTypeRef(.i256) } },
        .{ .binary = .{ .op = .eq, .lhs = 2, .rhs = 3 } },
        .{ .int_lit = .{ .value = "10", .ty = builtinTypeRef(.i256) } },
        .{ .int_lit = .{ .value = neg_three, .ty = builtinTypeRef(.i256) } },
        .{ .binary = .{ .op = .mod, .lhs = 5, .rhs = 6, .ty = builtinTypeRef(.i256) } },
        .{ .int_lit = .{ .value = "1", .ty = builtinTypeRef(.i256) } },
        .{ .binary = .{ .op = .eq, .lhs = 7, .rhs = 8 } },
    };

    try testing.expectEqual(obligation_to_z3.CheckStatus.proved, try checkFormalZ3TermObligation(&terms, 4));
    try testing.expectEqual(obligation_to_z3.CheckStatus.proved, try checkFormalZ3TermObligation(&terms, 9));
}

test "formal Z3 adapter finds counterexample for unassumed canonical obligation" {
    var z3_ctx = try z3_verification.Z3Context.init(testing.allocator);
    defer z3_ctx.deinit();

    const terms = [_]obligation.Term{
        .{ .variable = .{ .free = .{ .id = .{ .file_id = 0, .pattern_id = 0 }, .name = "amount", .ty = .{ .spelling = "u256" } } } },
        .{ .int_lit = .{ .value = "0", .ty = .{ .spelling = "u256" } } },
        .{ .refinement_predicate = .{ .name = "NonZero", .value = 0 } },
    };
    const obligations = [_]obligation.Obligation{
        .{
            .id = 1,
            .owner = .{ .function = .{ .name = "deposit" } },
            .source = .generated(),
            .phase = .sema,
            .origin = .{ .sema_fact = .{ .kind = "refinement", .ordinal = 0 } },
            .kind = .{ .logical = .{
                .role = .refinement,
                .formula = .{ .term = 2 },
            } },
        },
    };
    const obligation_ids = [_]obligation.Id{1};
    const queries = [_]obligation.VerificationQuery{
        .{
            .id = 2,
            .owner = .{ .function = .{ .name = "deposit" } },
            .source = .generated(),
            .phase = .report,
            .origin = .source,
            .backend = .z3,
            .kind = .obligation,
            .obligation_ids = &obligation_ids,
        },
    };
    const set: obligation.ObligationSet = .{
        .obligations = &obligations,
        .queries = &queries,
        .terms = &terms,
    };

    var adapter = obligation_to_z3.Adapter.init(&z3_ctx, testing.allocator, set);
    try testing.expectEqual(obligation_to_z3.CheckStatus.disproved, try adapter.checkObligation(1));
}

test "formal Z3 adapter fails closed on MLIR-origin formulas" {
    var z3_ctx = try z3_verification.Z3Context.init(testing.allocator);
    defer z3_ctx.deinit();

    const obligations = [_]obligation.Obligation{
        .{
            .id = 1,
            .owner = .{ .function = .{ .name = "checked" } },
            .source = .generated(),
            .phase = .ora_mlir,
            .origin = .{ .mlir_op = .{ .op_name = "ora.ensures", .symbol = "checked" } },
            .kind = .{ .logical = .{
                .role = .ensures,
                .formula = .{ .origin_value = .{
                    .origin = .{ .mlir_op = .{ .op_name = "ora.ensures", .symbol = "checked" } },
                } },
            } },
        },
    };
    const obligation_ids = [_]obligation.Id{1};
    const queries = [_]obligation.VerificationQuery{
        .{
            .id = 2,
            .owner = .{ .function = .{ .name = "checked" } },
            .source = .generated(),
            .phase = .report,
            .origin = .source,
            .backend = .z3,
            .kind = .obligation,
            .obligation_ids = &obligation_ids,
        },
    };
    const set: obligation.ObligationSet = .{ .obligations = &obligations, .queries = &queries };

    var adapter = obligation_to_z3.Adapter.init(&z3_ctx, testing.allocator, set);
    try testing.expectError(error.UnsupportedOriginValue, adapter.checkObligation(1));
}

test "formal Lean emitter writes manifest rows from canonical obligations" {
    const terms = [_]obligation.Term{
        .{ .variable = .{ .free = .{ .id = .{ .file_id = 0, .pattern_id = 0 }, .name = "balance", .ty = .{ .spelling = "u256" } } } },
        .{ .variable = .{ .free = .{ .id = .{ .file_id = 0, .pattern_id = 1 }, .name = "amount", .ty = .{ .spelling = "u256" } } } },
        .{ .binary = .{ .op = .ge, .lhs = 0, .rhs = 1, .ty = builtinTypeRef(.u256) } },
    };
    const assumptions = [_]obligation.Assumption{
        .{
            .id = 1,
            .owner = .{ .function = .{ .name = "transfer" } },
            .source = .generated(),
            .phase = .sema,
            .origin = .{ .sema_fact = .{ .kind = "requires", .ordinal = 0 } },
            .kind = .requires,
            .formula = .{ .term = 2 },
        },
    };
    const obligations = [_]obligation.Obligation{
        .{
            .id = 2,
            .owner = .{ .function = .{ .name = "transfer" } },
            .source = .generated(),
            .phase = .sema,
            .origin = .{ .sema_fact = .{ .kind = "resource_source_sufficient", .ordinal = 0 } },
            .kind = .{ .logical = .{
                .role = .invariant,
                .formula = .{ .term = 2 },
            } },
        },
    };
    const assumption_ids = [_]obligation.Id{1};
    const obligation_ids = [_]obligation.Id{2};
    const queries = [_]obligation.VerificationQuery{
        .{
            .id = 3,
            .owner = .{ .function = .{ .name = "deposit" } },
            .source = .generated(),
            .phase = .report,
            .origin = .source,
            .backend = .z3,
            .kind = .obligation,
            .obligation_ids = &obligation_ids,
            .assumption_ids = &assumption_ids,
        },
    };
    const set: obligation.ObligationSet = .{
        .assumptions = &assumptions,
        .obligations = &obligations,
        .queries = &queries,
        .terms = &terms,
    };

    const actual = try emitLeanToOwnedString(testing.allocator, set);
    defer testing.allocator.free(actual);

    try testing.expect(std.mem.containsAtLeast(u8, actual, 1, "import Ora.Obligation.Theorems"));
    try testing.expect(std.mem.containsAtLeast(u8, actual, 1, ".variable (.free { id := { file_id := 0, pattern_id := 0 }, name := \"balance\""));
    try testing.expect(std.mem.containsAtLeast(u8, actual, 1, ".variable (.free { id := { file_id := 0, pattern_id := 1 }, name := \"amount\""));
    try testing.expect(std.mem.containsAtLeast(u8, actual, 1, "{ id := 1, owner := \"transfer\", kind := .requires, formula := some (.term 2) }"));
    try testing.expect(std.mem.containsAtLeast(u8, actual, 1, "{ id := 2, owner := \"transfer\", kind := .logical .invariant (.term 2) }"));
    try testing.expect(std.mem.containsAtLeast(u8, actual, 1, "theorem emitted_manifest_wf : emittedManifest.wf = true := by decide"));
    try testing.expect(std.mem.containsAtLeast(u8, actual, 1, "def emittedObligation_2 : Prop :="));
}

test "formal Lean emitter writes resource rows with places" {
    const terms = [_]obligation.Term{
        .{ .variable = .{ .free = .{ .id = .{ .file_id = 0, .pattern_id = 0 }, .name = "amount", .ty = .{ .spelling = "u256" } } } },
    };
    const source_place: obligation.PlaceRef = .{
        .root = "balances",
        .region = .storage,
        .keys = &.{.{ .parameter = .{ .file_id = 0, .pattern_id = 0 } }},
    };
    const destination_place: obligation.PlaceRef = .{
        .root = "balances",
        .region = .storage,
        .keys = &.{.{ .parameter = .{ .file_id = 0, .pattern_id = 1 } }},
    };
    const obligations = [_]obligation.Obligation{
        .{
            .id = 1,
            .owner = .{ .function = .{ .name = "transfer" } },
            .source = .generated(),
            .phase = .ora_mlir,
            .origin = .{ .resource_op = .{ .op = .move, .domain = "TokenUnit" } },
            .kind = .{ .resource = .{
                .op = .move,
                .domain = "TokenUnit",
                .source = source_place,
                .destination = destination_place,
                .amount = .{ .term = 0 },
                .property = .conservation,
            } },
        },
    };
    const obligation_ids = [_]obligation.Id{1};
    const queries = [_]obligation.VerificationQuery{
        .{
            .id = 2,
            .owner = .{ .function = .{ .name = "deposit" } },
            .source = .generated(),
            .phase = .report,
            .origin = .source,
            .backend = .z3,
            .kind = .obligation,
            .obligation_ids = &obligation_ids,
        },
    };
    const set: obligation.ObligationSet = .{
        .obligations = &obligations,
        .queries = &queries,
        .terms = &terms,
    };

    const actual = try emitLeanToOwnedString(testing.allocator, set);
    defer testing.allocator.free(actual);

    try testing.expect(std.mem.containsAtLeast(u8, actual, 1, ".resource { op := .move, domain := \"TokenUnit\""));
    try testing.expect(std.mem.containsAtLeast(u8, actual, 1, "source := some { root := \"balances\", region := .storage, fields := [], keys := [.parameter { file_id := 0, pattern_id := 0 }] }"));
    try testing.expect(std.mem.containsAtLeast(u8, actual, 1, "destination := some { root := \"balances\", region := .storage, fields := [], keys := [.parameter { file_id := 0, pattern_id := 1 }] }"));
    try testing.expect(std.mem.containsAtLeast(u8, actual, 1, "amount := some (.term 0), property := .conservation"));
    try testing.expect(std.mem.containsAtLeast(u8, actual, 1, "theorem emitted_manifest_wf"));
}

test "formal Lean support accepts u256 resource model properties" {
    const terms = [_]obligation.Term{
        .{ .variable = .{ .free = .{ .id = .{ .file_id = 0, .pattern_id = 2 }, .name = "amount", .ty = .{ .spelling = "u256" } } } },
    };
    const source_place: obligation.PlaceRef = .{
        .root = "balances",
        .region = .storage,
        .keys = &.{.{ .parameter = .{ .file_id = 0, .pattern_id = 0 } }},
    };
    const destination_place: obligation.PlaceRef = .{
        .root = "balances",
        .region = .storage,
        .keys = &.{.{ .parameter = .{ .file_id = 0, .pattern_id = 1 } }},
    };
    const properties = [_]obligation.ResourceProperty{
        .amount_non_negative,
        .source_sufficient,
        .destination_no_overflow,
        .same_place_identity,
        .conservation,
    };
    var obligations: [properties.len]obligation.Obligation = undefined;
    var queries: [properties.len]obligation.VerificationQuery = undefined;
    var obligation_ids: [properties.len][1]obligation.Id = undefined;
    for (properties, 0..) |property, index| {
        const id: obligation.Id = @intCast(index + 1);
        obligations[index] = .{
            .id = id,
            .owner = .{ .function = .{ .name = "transfer" } },
            .source = .generated(),
            .phase = .ora_mlir,
            .origin = .{ .resource_op = .{ .op = .move, .domain = "TokenUnit", .ordinal = @intCast(index) } },
            .kind = .{ .resource = .{
                .op = .move,
                .domain = "TokenUnit",
                .source = source_place,
                .destination = destination_place,
                .amount = .{ .term = 0 },
                .property = property,
            } },
        };
        obligation_ids[index] = .{id};
        queries[index] = .{
            .id = @intCast(index + 10),
            .owner = .{ .function = .{ .name = "transfer" } },
            .source = .generated(),
            .phase = .ora_mlir,
            .origin = .source,
            .kind = .obligation,
            .obligation_ids = &obligation_ids[index],
        };
    }
    const set: obligation.ObligationSet = .{
        .obligations = &obligations,
        .queries = &queries,
        .terms = &terms,
    };

    for (queries) |query| {
        switch (obligation_to_lean.querySemanticSupport(set, query)) {
            .supported => {},
            .unsupported => return error.TestUnexpectedResult,
        }
    }
}

test "formal Lean support treats move resource guards as self-move aware" {
    const terms = [_]obligation.Term{
        .{ .variable = .{ .free = .{ .id = .{ .file_id = 0, .pattern_id = 2 }, .name = "amount", .ty = .{ .spelling = "u256" } } } },
    };
    const source_place: obligation.PlaceRef = .{
        .root = "balances",
        .region = .storage,
        .keys = &.{.{ .parameter = .{ .file_id = 0, .pattern_id = 0 } }},
    };
    const destination_place: obligation.PlaceRef = .{
        .root = "balances",
        .region = .storage,
        .keys = &.{.{ .parameter = .{ .file_id = 0, .pattern_id = 1 } }},
    };

    try expectSingleResourceSemanticSupport(&terms, .{
        .op = .move,
        .domain = "TokenUnit",
        .source = source_place,
        .amount = .{ .term = 0 },
        .property = .source_sufficient,
    }, false);
    try expectSingleResourceSemanticSupport(&terms, .{
        .op = .move,
        .domain = "TokenUnit",
        .destination = destination_place,
        .amount = .{ .term = 0 },
        .property = .destination_no_overflow,
    }, false);
    try expectSingleResourceSemanticSupport(&terms, .{
        .op = .destroy,
        .domain = "TokenUnit",
        .source = source_place,
        .amount = .{ .term = 0 },
        .property = .source_sufficient,
    }, true);
    try expectSingleResourceSemanticSupport(&terms, .{
        .op = .create,
        .domain = "TokenUnit",
        .destination = destination_place,
        .amount = .{ .term = 0 },
        .property = .destination_no_overflow,
    }, true);
}

test "formal Lean emitter writes quantifier metadata rows" {
    const obligations = [_]obligation.Obligation{
        .{
            .id = 1,
            .owner = .{ .function = .{ .name = "bounded" } },
            .source = .generated(),
            .phase = .ora_mlir,
            .origin = .{ .mlir_op = .{ .op_name = "ora.quantified", .symbol = "bounded" } },
            .kind = .{ .quantifier = .{
                .quantifier = .forall,
                .variable = "i",
                .binder_type = .{ .spelling = "u256" },
                .binder_sort = .bit_vector,
                .fragment = .aufbv_quantifiers,
                .pattern_status = .absent,
            } },
            .artifact_policy = .diagnostic_only,
        },
    };
    const set: obligation.ObligationSet = .{ .obligations = &obligations };

    const actual = try emitLeanToOwnedString(testing.allocator, set);
    defer testing.allocator.free(actual);

    try testing.expect(std.mem.containsAtLeast(u8, actual, 1, ".quantifier { quantifier := .forall_, binderName := \"i\""));
    try testing.expect(std.mem.containsAtLeast(u8, actual, 1, "binderType := .spelling \"u256\", binderSort := .bitVector"));
    try testing.expect(std.mem.containsAtLeast(u8, actual, 1, "fragment := .aufbvQuantifiers, patternStatus := .absent, degradation := none"));
    try testing.expect(std.mem.containsAtLeast(u8, actual, 1, "theorem emitted_manifest_wf"));
}

test "formal Lean emitter writes backend fact rows" {
    const obligations = [_]obligation.Obligation{
        .{
            .id = 1,
            .owner = .{ .backend = .{ .component = .dispatcher, .name = "erc20" } },
            .source = .generated(),
            .phase = .sinora,
            .origin = .{ .backend_fact = .{ .component = .dispatcher, .fact = "selector_table_complete" } },
            .kind = .{ .backend_fact = .{
                .component = .dispatcher,
                .property = .preserves_selector_behavior,
            } },
            .required_backend = .lean,
        },
    };
    const set: obligation.ObligationSet = .{ .obligations = &obligations };

    const actual = try emitLeanToOwnedString(testing.allocator, set);
    defer testing.allocator.free(actual);

    try testing.expect(std.mem.containsAtLeast(u8, actual, 1, ".backendFact { component := .dispatcher, property := .preservesSelectorBehavior }"));
    try testing.expect(std.mem.containsAtLeast(u8, actual, 1, "theorem emitted_manifest_wf"));
}

test "formal Lean emitter writes userland proof artifact attachments" {
    const obligation_ids = [_]obligation.Id{1};
    const obligations = [_]obligation.Obligation{
        .{
            .id = 1,
            .owner = .{ .function = .{ .name = "transfer" } },
            .source = .generated(),
            .phase = .sema,
            .origin = .{ .sema_fact = .{ .kind = "ensures", .ordinal = 0 } },
            .kind = .{ .logical = .{
                .role = .ensures,
                .formula = .{ .term = 0 },
            } },
            .required_backend = .lean,
        },
    };
    const artifacts = [_]obligation.ProofArtifact{
        .{
            .id = 9,
            .owner = .{ .function = .{ .name = "transfer" } },
            .source = .{ .file = "proofs/ERC20/Transfer.lean" },
            .module_name = "ERC20.Transfer",
            .theorem_name = "transfer_preserves_supply",
            .path = "proofs/ERC20/Transfer.lean",
            .content_hash = 0x1234,
            .obligation_ids = &obligation_ids,
        },
    };
    const set: obligation.ObligationSet = .{
        .obligations = &obligations,
        .proof_artifacts = &artifacts,
        .terms = &.{.{ .bool_lit = true }},
    };

    const actual = try emitLeanToOwnedString(testing.allocator, set);
    defer testing.allocator.free(actual);

    try testing.expect(std.mem.containsAtLeast(u8, actual, 1, "def emittedProofArtifacts : List ProofArtifactRow := ["));
    try testing.expect(std.mem.containsAtLeast(u8, actual, 1, "kind := .userlandLean"));
    try testing.expect(std.mem.containsAtLeast(u8, actual, 1, "moduleName := \"ERC20.Transfer\", theoremName := \"transfer_preserves_supply\""));
    try testing.expect(std.mem.containsAtLeast(u8, actual, 1, "path := some \"proofs/ERC20/Transfer.lean\", contentHash := some 4660, obligationIds := [1]"));
    try testing.expect(std.mem.containsAtLeast(u8, actual, 1, "proofArtifacts := emittedProofArtifacts"));
}

test "formal Lean emitter fails closed on MLIR-origin formulas" {
    const obligations = [_]obligation.Obligation{
        .{
            .id = 1,
            .owner = .{ .function = .{ .name = "checked" } },
            .source = .generated(),
            .phase = .ora_mlir,
            .origin = .{ .mlir_op = .{ .op_name = "ora.ensures", .symbol = "checked" } },
            .kind = .{ .logical = .{
                .role = .ensures,
                .formula = .{ .origin_value = .{
                    .origin = .{ .mlir_op = .{ .op_name = "ora.ensures", .symbol = "checked" } },
                } },
            } },
        },
    };
    const set: obligation.ObligationSet = .{ .obligations = &obligations };

    var buffer = std.Io.Writer.Allocating.init(testing.allocator);
    defer buffer.deinit();
    try testing.expectError(error.UnsupportedOriginValue, obligation_to_lean.writeModule(&buffer.writer, set, .{}));
}

test "formal cross-check elides runtime guard only when Z3 proves and Lean exports" {
    const terms = [_]obligation.Term{
        .{ .variable = .{ .free = .{ .id = .{ .file_id = 0, .pattern_id = 0 }, .name = "amount", .ty = .{ .spelling = "u256" } } } },
        .{ .int_lit = .{ .value = "1", .ty = .{ .spelling = "u256" } } },
        .{ .binary = .{ .op = .ge, .lhs = 0, .rhs = 1 } },
    };
    const assumptions = [_]obligation.Assumption{
        .{
            .id = 1,
            .owner = .{ .function = .{ .name = "deposit" } },
            .source = .generated(),
            .phase = .sema,
            .origin = .{ .sema_fact = .{ .kind = "requires", .ordinal = 0 } },
            .kind = .requires,
            .formula = .{ .term = 2 },
        },
    };
    const obligations = [_]obligation.Obligation{
        .{
            .id = 2,
            .owner = .{ .function = .{ .name = "deposit" } },
            .source = .generated(),
            .phase = .sema,
            .origin = .{ .sema_fact = .{ .kind = "guard", .ordinal = 0 } },
            .kind = .{ .runtime_guard = .{
                .guard_id = "guard:deposit:amount",
                .formula = .{ .term = 2 },
                .erasure = .may_elide_if_proven,
            } },
        },
    };
    const assumption_ids = [_]obligation.Id{1};
    const obligation_ids = [_]obligation.Id{2};
    const queries = [_]obligation.VerificationQuery{
        .{
            .id = 3,
            .owner = .{ .function = .{ .name = "deposit" } },
            .source = .generated(),
            .phase = .report,
            .origin = .source,
            .backend = .z3,
            .kind = .obligation,
            .obligation_ids = &obligation_ids,
            .assumption_ids = &assumption_ids,
        },
    };
    const set: obligation.ObligationSet = .{
        .assumptions = &assumptions,
        .obligations = &obligations,
        .queries = &queries,
        .terms = &terms,
    };

    var result = try obligation_crosscheck.crossCheckRuntimeGuard(testing.allocator, set, 2);
    defer result.deinit();

    try testing.expectEqual(@as(obligation.Id, 2), result.obligation_id);
    try testing.expectEqual(obligation_to_z3.CheckStatus.proved, result.z3_status);
    try testing.expectEqual(obligation_crosscheck.RuntimeErasureDecision.elide_runtime_check, result.runtime_erasure);
    try testing.expect(std.mem.containsAtLeast(u8, result.lean_source, 1, "theorem emitted_manifest_wf"));
    try testing.expect(std.mem.containsAtLeast(u8, result.lean_source, 1, "guard:deposit:amount"));
}

test "formal cross-check keeps runtime guard when Z3 finds a counterexample" {
    const terms = [_]obligation.Term{
        .{ .variable = .{ .free = .{ .id = .{ .file_id = 0, .pattern_id = 0 }, .name = "amount", .ty = .{ .spelling = "u256" } } } },
        .{ .refinement_predicate = .{ .name = "NonZero", .value = 0 } },
    };
    const obligations = [_]obligation.Obligation{
        .{
            .id = 1,
            .owner = .{ .function = .{ .name = "deposit" } },
            .source = .generated(),
            .phase = .sema,
            .origin = .{ .sema_fact = .{ .kind = "guard", .ordinal = 0 } },
            .kind = .{ .runtime_guard = .{
                .guard_id = "guard:deposit:amount",
                .formula = .{ .term = 1 },
                .erasure = .may_elide_if_proven,
            } },
        },
    };
    const obligation_ids = [_]obligation.Id{1};
    const queries = [_]obligation.VerificationQuery{
        .{
            .id = 2,
            .owner = .{ .function = .{ .name = "deposit" } },
            .source = .generated(),
            .phase = .report,
            .origin = .source,
            .backend = .z3,
            .kind = .obligation,
            .obligation_ids = &obligation_ids,
        },
    };
    const set: obligation.ObligationSet = .{
        .obligations = &obligations,
        .queries = &queries,
        .terms = &terms,
    };

    var result = try obligation_crosscheck.crossCheckRuntimeGuard(testing.allocator, set, 1);
    defer result.deinit();

    try testing.expectEqual(obligation_to_z3.CheckStatus.disproved, result.z3_status);
    try testing.expectEqual(obligation_crosscheck.RuntimeErasureDecision.keep_runtime_check, result.runtime_erasure);
    try testing.expect(std.mem.containsAtLeast(u8, result.lean_source, 1, "emittedManifest.wf = true"));
}

test "formal cross-check keeps always-runtime guard even when proved" {
    const terms = [_]obligation.Term{
        .{ .bool_lit = true },
    };
    const obligations = [_]obligation.Obligation{
        .{
            .id = 1,
            .owner = .{ .function = .{ .name = "critical" } },
            .source = .generated(),
            .phase = .sema,
            .origin = .{ .sema_fact = .{ .kind = "assert", .ordinal = 0 } },
            .kind = .{ .runtime_guard = .{
                .guard_id = "guard:critical:true",
                .formula = .{ .term = 0 },
                .erasure = .always_runtime,
            } },
        },
    };
    const obligation_ids = [_]obligation.Id{1};
    const queries = [_]obligation.VerificationQuery{
        .{
            .id = 2,
            .owner = .{ .function = .{ .name = "critical" } },
            .source = .generated(),
            .phase = .report,
            .origin = .source,
            .backend = .z3,
            .kind = .obligation,
            .obligation_ids = &obligation_ids,
        },
    };
    const set: obligation.ObligationSet = .{
        .obligations = &obligations,
        .queries = &queries,
        .terms = &terms,
    };

    var result = try obligation_crosscheck.crossCheckRuntimeGuard(testing.allocator, set, 1);
    defer result.deinit();

    try testing.expectEqual(obligation_to_z3.CheckStatus.proved, result.z3_status);
    try testing.expectEqual(obligation_crosscheck.RuntimeErasureDecision.keep_runtime_check, result.runtime_erasure);
}

test "formal obligation MLIR adapter classifies arithmetic safety producers" {
    const h = createContext();
    defer destroyContext(h);

    const text =
        \\module {
        \\  func.func @arith(%flag: i1, %x: i256, %y: i256) {
        \\    "ora.assert"(%flag) <{message = "checked addition overflow"}> : (i1) -> ()
        \\    "ora.assert"(%flag) <{message = "checked subtraction overflow"}> : (i1) -> ()
        \\    "ora.assert"(%flag) <{message = "checked multiplication overflow"}> : (i1) -> ()
        \\    "ora.assert"(%flag) <{message = "checked power overflow"}> : (i1) -> ()
        \\    "ora.assert"(%flag) <{message = "checked negation overflow"}> : (i1) -> ()
        \\    "ora.assert"(%flag) <{message = "checked shift amount out of range"}> : (i1) -> ()
        \\    %q = arith.divui %x, %y : i256
        \\    %sl = arith.shli %x, %y : i256
        \\    %sr = arith.shrui %x, %y : i256
        \\    %ssr = arith.shrsi %x, %y : i256
        \\    func.return
        \\  }
        \\}
    ;

    const module = try parseModule(h.ctx, text);
    defer mlir.oraModuleDestroy(module);

    var result = try obligation_from_mlir.collect(testing.allocator, module, .{});
    defer result.deinit();

    try testing.expect(!result.set.hasBlockingDiagnostic());
    try testing.expectEqual(@as(usize, 7), result.set.obligations.len);
    try testing.expectEqual(@as(usize, 8), result.set.queries.len);
    try testing.expectEqual(@as(usize, 7), result.query_bindings.len);
    for (result.query_bindings) |binding| {
        try testing.expectEqual(obligation.LogicalRole.arithmetic_safety, binding.logical_role orelse return error.TestUnexpectedResult);
    }
    try testing.expectEqual(@as(usize, 1), countQuery(result.set, .base));
    try testing.expectEqual(@as(usize, 7), countQuery(result.set, .obligation));
    try testing.expectEqual(@as(usize, 1), countArithmeticSafety(result.set, .addition_overflow));
    try testing.expectEqual(@as(usize, 1), countArithmeticSafety(result.set, .subtraction_overflow));
    try testing.expectEqual(@as(usize, 1), countArithmeticSafety(result.set, .multiplication_overflow));
    try testing.expectEqual(@as(usize, 1), countArithmeticSafety(result.set, .power_overflow));
    try testing.expectEqual(@as(usize, 1), countArithmeticSafety(result.set, .negation_overflow));
    try testing.expectEqual(@as(usize, 1), countArithmeticSafety(result.set, .division_by_zero));
    try testing.expectEqual(@as(usize, 1), countArithmeticSafety(result.set, .shift_amount_bounds));
}

test "compound arithmetic gate internal quotient exemption requires the exact string marker" {
    const h = createContext();
    defer destroyContext(h);
    for ([_][]const u8{ "", " {ora.guard_internal = \"false\"}", " {ora.guard_internal = true}", " {ora.guard_internal = \"TRUE\"}", " {ora.guard_internal = \"true\"}" }, 0..) |attrs, index| {
        const text = try std.fmt.allocPrint(testing.allocator,
            \\module {{
            \\  func.func @arith(%flag: i1, %x: i256, %y: i256) {{
            \\    %q = arith.divui %x, %y{s} : i256
            \\    "ora.assert"(%flag) <{{message = "checked power overflow"}}> : (i1) -> ()
            \\    func.return
            \\  }}
            \\}}
        , .{attrs});
        defer testing.allocator.free(text);
        const module = try parseModule(h.ctx, text);
        defer mlir.oraModuleDestroy(module);
        var result = try obligation_from_mlir.collect(testing.allocator, module, .{});
        defer result.deinit();
        try testing.expect(!result.set.hasBlockingDiagnostic());
        try testing.expectEqual(@as(usize, if (index == 4) 0 else 1), countArithmeticSafety(result.set, .division_by_zero));
        try testing.expectEqual(@as(usize, 1), countArithmeticSafety(result.set, .power_overflow));
    }
}

test "formal obligation MLIR adapter collects verification markers and assumptions" {
    const h = createContext();
    defer destroyContext(h);

    const text =
        \\module {
        \\  func.func @checked(%flag: i1) {
        \\    "ora.requires"(%flag) : (i1) -> ()
        \\    "ora.ensures"(%flag) : (i1) -> ()
        \\    "ora.assert"(%flag) <{message = "must hold"}> : (i1) -> ()
        \\    "ora.assume"(%flag) : (i1) -> ()
        \\    "ora.refinement_guard"(%flag) <{message = "guard"}> : (i1) -> ()
        \\    func.return
        \\  }
        \\}
    ;

    const module = try parseModule(h.ctx, text);
    defer mlir.oraModuleDestroy(module);

    const guard_op = findFirstOpByName(module, "ora.refinement_guard") orelse return error.TestUnexpectedResult;
    mlir.oraOperationSetAttributeByName(
        guard_op,
        strRef("ora.guard_id"),
        mlir.oraStringAttrCreate(h.ctx, strRef("guard:checked:flag")),
    );

    var result = try obligation_from_mlir.collect(testing.allocator, module, .{});
    defer result.deinit();

    try testing.expect(!result.set.hasBlockingDiagnostic());
    try testing.expectEqual(@as(usize, 3), result.set.obligations.len);
    try testing.expectEqual(@as(usize, 2), result.set.assumptions.len);
    try testing.expectEqual(@as(usize, 5), result.set.queries.len);
    try testing.expectEqual(@as(usize, 1), countAssumption(result.set, .requires));
    try testing.expectEqual(@as(usize, 1), countAssumption(result.set, .assume));
    try testing.expectEqual(@as(usize, 1), countLogical(result.set, .ensures));
    try testing.expectEqual(@as(usize, 1), countLogical(result.set, .assert));
    try testing.expectEqual(@as(usize, 1), countRuntimeGuards(result.set));
    try testing.expectEqual(@as(usize, 1), countQuery(result.set, .base));
    try testing.expectEqual(@as(usize, 2), countQuery(result.set, .obligation));
    try testing.expectEqual(@as(usize, 1), countQuery(result.set, .guard_satisfy));
    try testing.expectEqual(@as(usize, 1), countQuery(result.set, .guard_violate));
    try testing.expect(result.set.assumptions[0].owner == .function);
    try testing.expectEqualStrings("checked", result.set.assumptions[0].owner.function.name);

    const guard = result.set.obligations[2];
    try testing.expect(guard.kind == .runtime_guard);
    try testing.expectEqualStrings("guard:checked:flag", guard.kind.runtime_guard.guard_id);
    try testing.expect(guard.kind.runtime_guard.formula == .term);
}

test "formal obligation MLIR adapter binds function params by compiler id not display name" {
    const h = createContext();
    defer destroyContext(h);

    const text =
        \\module {
        \\  func.func @same_display_name(%arg0: !ora.int<256, false>, %arg1: !ora.int<256, false>) attributes {
        \\    ora.param_names = ["same", "same"],
        \\    ora.param_binding_ids = ["file:7:pattern:10", "file:7:pattern:11"]
        \\  } {
        \\    %cmp = ora.cmp "ule", %arg0, %arg1 : !ora.int<256, false>, !ora.int<256, false> -> i1
        \\    "ora.ensures"(%cmp) : (i1) -> ()
        \\    func.return
        \\  }
        \\}
    ;

    const module = try parseModule(h.ctx, text);
    defer mlir.oraModuleDestroy(module);

    var result = try obligation_from_mlir.collect(testing.allocator, module, .{});
    defer result.deinit();

    try testing.expect(!result.set.hasBlockingDiagnostic());
    try testing.expectEqual(@as(usize, 1), countLogical(result.set, .ensures));

    const outer = try expectQuantifiedTerm(result.set, try expectLogicalTerm(result.set, .ensures), .forall, "same");
    try expectTypeRefBuiltin(outer.binder.ty, .u256);
    try testing.expectEqual(obligation.BinderOrigin.function_param, outer.binder.origin);
    const inner = try expectQuantifiedTerm(result.set, outer.body, .forall, "same");
    try expectTypeRefBuiltin(inner.binder.ty, .u256);
    try testing.expectEqual(obligation.BinderOrigin.function_param, inner.binder.origin);
    const body = try expectBinaryTerm(result.set, inner.body, .le);
    try expectBoundVarTerm(result.set, body.lhs, 1, "same");
    try expectBoundVarTerm(result.set, body.rhs, 0, "same");
}

test "formal obligation MLIR adapter blocks named params without compiler binding ids" {
    const h = createContext();
    defer destroyContext(h);

    const text =
        \\module {
        \\  func.func @old_metadata(%arg0: !ora.int<256, false>) attributes {
        \\    ora.param_names = ["x"]
        \\  } {
        \\    %cmp = ora.cmp "ule", %arg0, %arg0 : !ora.int<256, false>, !ora.int<256, false> -> i1
        \\    "ora.ensures"(%cmp) : (i1) -> ()
        \\    func.return
        \\  }
        \\}
    ;

    const module = try parseModule(h.ctx, text);
    defer mlir.oraModuleDestroy(module);

    var result = try obligation_from_mlir.collect(testing.allocator, module, .{});
    defer result.deinit();

    try testing.expect(result.set.hasBlockingDiagnostic());
    try testing.expectEqual(@as(usize, 1), result.set.diagnostics.len);
    try testing.expectEqual(obligation.ArtifactBlockReason.blocking_diagnostic, result.set.artifactDecision().blocked);
}

test "formal obligation MLIR adapter binds function params in inserted forall conditions" {
    const h = createContext();
    defer destroyContext(h);

    const text =
        \\module {
        \\  func.func @bounded_by_requires(%arg0: !ora.int<256, false>) attributes {
        \\    ora.param_names = ["x"],
        \\    ora.param_binding_ids = ["file:9:pattern:3"]
        \\  } {
        \\    %req = ora.cmp "ult", %arg0, %arg0 : !ora.int<256, false>, !ora.int<256, false> -> i1
        \\    "ora.requires"(%req) : (i1) -> ()
        \\    %ens = ora.cmp "ule", %arg0, %arg0 : !ora.int<256, false>, !ora.int<256, false> -> i1
        \\    "ora.ensures"(%ens) : (i1) -> ()
        \\    func.return
        \\  }
        \\}
    ;

    const module = try parseModule(h.ctx, text);
    defer mlir.oraModuleDestroy(module);

    var result = try obligation_from_mlir.collect(testing.allocator, module, .{});
    defer result.deinit();

    try testing.expect(!result.set.hasBlockingDiagnostic());
    try testing.expectEqual(@as(usize, 1), countAssumption(result.set, .requires));
    try testing.expectEqual(@as(usize, 1), countLogical(result.set, .ensures));

    const requires_formula = result.set.assumptions[0].formula orelse return error.TestUnexpectedResult;
    try testing.expect(requires_formula == .term);
    const requires_body = try expectBinaryTerm(result.set, requires_formula.term, .lt);
    const free = try expectFreeVarTermRef(result.set, requires_body.lhs, 9, 3, "x");
    try expectTypeRefBuiltin(free.ty, .u256);

    const outer = try expectQuantifiedTerm(result.set, try expectLogicalTerm(result.set, .ensures), .forall, "x");
    try expectTypeRefBuiltin(outer.binder.ty, .u256);
    const condition_id = outer.condition orelse return error.TestUnexpectedResult;
    const condition = try expectBinaryTerm(result.set, condition_id, .lt);
    try expectBoundVarTerm(result.set, condition.lhs, 0, "x");
    const body = try expectBinaryTerm(result.set, outer.body, .le);
    try expectBoundVarTerm(result.set, body.lhs, 0, "x");
}

test "formal obligation MLIR adapter derives term types from MLIR values" {
    const h = createContext();
    defer destroyContext(h);

    const text =
        \\module {
        \\  func.func @typed_terms(%arg0: !ora.int<64, true>, %arg1: !ora.int<32, false>, %arg2: !ora.address) attributes {
        \\    ora.param_names = ["sx", "uy", "who"],
        \\    ora.param_binding_ids = ["file:11:pattern:1", "file:11:pattern:2", "file:11:pattern:3"]
        \\  } {
        \\    %req = ora.cmp "ule", %arg1, %arg1 : !ora.int<32, false>, !ora.int<32, false> -> i1
        \\    "ora.requires"(%req) : (i1) -> ()
        \\    %addr = ora.cmp "eq", %arg2, %arg2 : !ora.address, !ora.address -> i1
        \\    "ora.assume"(%addr) : (i1) -> ()
        \\    %ens = ora.cmp "sle", %arg0, %arg0 : !ora.int<64, true>, !ora.int<64, true> -> i1
        \\    "ora.ensures"(%ens) : (i1) -> ()
        \\    func.return
        \\  }
        \\}
    ;

    const module = try parseModule(h.ctx, text);
    defer mlir.oraModuleDestroy(module);

    var result = try obligation_from_mlir.collect(testing.allocator, module, .{});
    defer result.deinit();

    try testing.expect(!result.set.hasBlockingDiagnostic());
    try testing.expectEqual(@as(usize, 1), countAssumption(result.set, .requires));
    try testing.expectEqual(@as(usize, 1), countAssumption(result.set, .assume));
    try testing.expectEqual(@as(usize, 1), countLogical(result.set, .ensures));

    for (result.set.assumptions) |assumption| {
        const formula = assumption.formula orelse return error.TestUnexpectedResult;
        try testing.expect(formula == .term);
        const binary = try expectBinaryTerm(result.set, formula.term, if (assumption.kind == .requires) .le else .eq);
        switch (assumption.kind) {
            .requires => {
                const free = try expectFreeVarTermRef(result.set, binary.lhs, 11, 2, "uy");
                try expectTypeRefBuiltin(free.ty, .u32);
                const rhs_free = try expectFreeVarTermRef(result.set, binary.rhs, 11, 2, "uy");
                try expectTypeRefBuiltin(rhs_free.ty, .u32);
            },
            .assume => {
                const free = try expectFreeVarTermRef(result.set, binary.lhs, 11, 3, "who");
                try expectTypeRefBuiltin(free.ty, .address);
            },
            else => return error.TestUnexpectedResult,
        }
    }

    const sx = try expectQuantifiedTerm(result.set, try expectLogicalTerm(result.set, .ensures), .forall, "sx");
    try expectTypeRefBuiltin(sx.binder.ty, .i64);
    const uy = try expectQuantifiedTerm(result.set, sx.body, .forall, "uy");
    try expectTypeRefBuiltin(uy.binder.ty, .u32);
    const who = try expectQuantifiedTerm(result.set, uy.body, .forall, "who");
    try expectTypeRefBuiltin(who.binder.ty, .address);
}

test "formal obligation MLIR adapter projects signed i256 comparison tags" {
    const h = createContext();
    defer destroyContext(h);

    const text =
        \\module {
        \\  func.func @signed_cmp(%arg0: !ora.int<256, true>, %arg1: !ora.int<256, true>) attributes {
        \\    ora.param_names = ["a", "b"],
        \\    ora.param_binding_ids = ["file:21:pattern:1", "file:21:pattern:2"]
        \\  } {
        \\    %cmp = ora.cmp "slt", %arg0, %arg1 : !ora.int<256, true>, !ora.int<256, true> -> i1
        \\    "ora.ensures"(%cmp) : (i1) -> ()
        \\    func.return
        \\  }
        \\}
    ;

    const module = try parseModule(h.ctx, text);
    defer mlir.oraModuleDestroy(module);

    var result = try obligation_from_mlir.collect(testing.allocator, module, .{});
    defer result.deinit();

    try testing.expect(!result.set.hasBlockingDiagnostic());
    try testing.expectEqual(@as(usize, 1), countLogical(result.set, .ensures));

    const a = try expectQuantifiedTerm(result.set, try expectLogicalTerm(result.set, .ensures), .forall, "a");
    try expectTypeRefBuiltin(a.binder.ty, .i256);
    const b = try expectQuantifiedTerm(result.set, a.body, .forall, "b");
    try expectTypeRefBuiltin(b.binder.ty, .i256);
    const body = try expectBinaryTerm(result.set, b.body, .slt);
    try expectTypeRefBuiltin(body.ty, .i256);
    try expectBoundVarTerm(result.set, body.lhs, 1, "a");
    try expectBoundVarTerm(result.set, body.rhs, 0, "b");
    try expectEnsuresQuerySupported(result.set);

    const lean = try emitLeanToOwnedString(testing.allocator, result.set);
    defer testing.allocator.free(lean);
    try testing.expect(std.mem.containsAtLeast(u8, lean, 1, ".slt"));
    try testing.expect(std.mem.containsAtLeast(u8, lean, 1, ".compilerTypeId 12"));
}

test "formal obligation Lean supports registered narrow signed comparison width" {
    const h = createContext();
    defer destroyContext(h);

    const text =
        \\module {
        \\  func.func @signed_narrow(%arg0: !ora.int<64, true>, %arg1: !ora.int<64, true>) attributes {
        \\    ora.param_names = ["a", "b"],
        \\    ora.param_binding_ids = ["file:22:pattern:1", "file:22:pattern:2"]
        \\  } {
        \\    %cmp = ora.cmp "slt", %arg0, %arg1 : !ora.int<64, true>, !ora.int<64, true> -> i1
        \\    "ora.ensures"(%cmp) : (i1) -> ()
        \\    func.return
        \\  }
        \\}
    ;

    const module = try parseModule(h.ctx, text);
    defer mlir.oraModuleDestroy(module);

    var result = try obligation_from_mlir.collect(testing.allocator, module, .{});
    defer result.deinit();

    try testing.expect(!result.set.hasBlockingDiagnostic());
    const a = try expectQuantifiedTerm(result.set, try expectLogicalTerm(result.set, .ensures), .forall, "a");
    try expectTypeRefBuiltin(a.binder.ty, .i64);
    const b = try expectQuantifiedTerm(result.set, a.body, .forall, "b");
    const body = try expectBinaryTerm(result.set, b.body, .slt);
    try expectTypeRefBuiltin(body.ty, .i64);
    try expectEnsuresQuerySupported(result.set);

    const lean = try emitLeanToOwnedString(testing.allocator, result.set);
    defer testing.allocator.free(lean);
    try testing.expect(std.mem.containsAtLeast(u8, lean, 1, ".slt"));
    try testing.expect(std.mem.containsAtLeast(u8, lean, 1, ".compilerTypeId 10"));
}

test "formal obligation MLIR adapter blocks signed predicate on unsigned operands" {
    const h = createContext();
    defer destroyContext(h);

    const text =
        \\module {
        \\  func.func @bad_signedness(%arg0: !ora.int<256, false>, %arg1: !ora.int<256, false>) attributes {
        \\    ora.param_names = ["a", "b"],
        \\    ora.param_binding_ids = ["file:23:pattern:1", "file:23:pattern:2"]
        \\  } {
        \\    %cmp = ora.cmp "slt", %arg0, %arg1 : !ora.int<256, false>, !ora.int<256, false> -> i1
        \\    "ora.ensures"(%cmp) : (i1) -> ()
        \\    func.return
        \\  }
        \\}
    ;

    const module = try parseModule(h.ctx, text);
    defer mlir.oraModuleDestroy(module);

    var result = try obligation_from_mlir.collect(testing.allocator, module, .{});
    defer result.deinit();

    try testing.expect(result.set.hasBlockingDiagnostic());
    try testing.expectEqual(@as(usize, 1), result.set.diagnostics.len);
    try testing.expectEqual(obligation.DiagnosticKind.comparison_signedness_mismatch, result.set.diagnostics[0].kind);
    try testing.expect(std.mem.containsAtLeast(u8, result.set.diagnostics[0].message, 1, "predicate signedness"));

    const report = try dumpManifestToOwnedString(testing.allocator, result.set);
    defer testing.allocator.free(report);
    try testing.expect(!std.mem.containsAtLeast(u8, report, 1, "\"op\":\"slt\""));
}

test "formal obligation Lean supports canonical signed type spelling" {
    const terms = [_]obligation.Term{
        .{ .variable = .{ .free = .{ .id = .{ .file_id = 31, .pattern_id = 1 }, .name = "a", .ty = .{ .spelling = "i256" } } } },
        .{ .variable = .{ .free = .{ .id = .{ .file_id = 31, .pattern_id = 2 }, .name = "b", .ty = .{ .spelling = "i256" } } } },
        .{ .binary = .{ .op = .slt, .lhs = 0, .rhs = 1, .ty = .{ .spelling = "i256" } } },
    };
    const obligations = [_]obligation.Obligation{
        .{
            .id = 1,
            .owner = .{ .function = .{ .name = "signed_cmp" } },
            .source = .generated(),
            .phase = .ora_mlir,
            .origin = .{ .mlir_op = .{ .op_name = "ora.ensures", .symbol = "signed_cmp" } },
            .kind = .{ .logical = .{
                .role = .ensures,
                .formula = .{ .term = 2 },
            } },
        },
    };
    const obligation_ids = [_]obligation.Id{1};
    const queries = [_]obligation.VerificationQuery{
        .{
            .id = 2,
            .owner = .{ .function = .{ .name = "signed_cmp" } },
            .source = .generated(),
            .phase = .report,
            .origin = .source,
            .backend = .z3,
            .kind = .obligation,
            .logical_role = .ensures,
            .obligation_ids = &obligation_ids,
        },
    };
    const set: obligation.ObligationSet = .{
        .obligations = &obligations,
        .queries = &queries,
        .terms = &terms,
    };

    try expectEnsuresQuerySupported(set);
}

test "formal obligation Lean rejects unregistered signed type spelling" {
    const terms = [_]obligation.Term{
        .{ .variable = .{ .free = .{ .id = .{ .file_id = 31, .pattern_id = 1 }, .name = "a", .ty = .{ .spelling = "i167" } } } },
        .{ .variable = .{ .free = .{ .id = .{ .file_id = 31, .pattern_id = 2 }, .name = "b", .ty = .{ .spelling = "i167" } } } },
        .{ .binary = .{ .op = .slt, .lhs = 0, .rhs = 1, .ty = .{ .spelling = "i167" } } },
    };
    const obligations = [_]obligation.Obligation{
        .{
            .id = 1,
            .owner = .{ .function = .{ .name = "unregistered_signed_cmp" } },
            .source = .generated(),
            .phase = .ora_mlir,
            .origin = .{ .mlir_op = .{ .op_name = "ora.ensures", .symbol = "unregistered_signed_cmp" } },
            .kind = .{ .logical = .{
                .role = .ensures,
                .formula = .{ .term = 2 },
            } },
        },
    };
    const obligation_ids = [_]obligation.Id{1};
    const queries = [_]obligation.VerificationQuery{
        .{
            .id = 2,
            .owner = .{ .function = .{ .name = "unregistered_signed_cmp" } },
            .source = .generated(),
            .phase = .report,
            .origin = .source,
            .backend = .z3,
            .kind = .obligation,
            .logical_role = .ensures,
            .obligation_ids = &obligation_ids,
        },
    };
    const set: obligation.ObligationSet = .{
        .obligations = &obligations,
        .queries = &queries,
        .terms = &terms,
    };

    try expectEnsuresQueryUnsupported(set, .unknown_signedness);
}

test "formal obligation Lean supports signed result, storage, and old operands" {
    const terms = [_]obligation.Term{
        .result,
        .{ .place_read = .{ .root = "reserve", .region = .storage } },
        .{ .old = 1 },
        .{ .binary = .{ .op = .sle, .lhs = 1, .rhs = 2, .ty = builtinTypeRef(.i256) } },
        .{ .binary = .{ .op = .sge, .lhs = 0, .rhs = 2, .ty = builtinTypeRef(.i256) } },
    };
    const obligations = [_]obligation.Obligation{
        .{
            .id = 1,
            .owner = .{ .function = .{ .name = "signed_storage" } },
            .source = .generated(),
            .phase = .sema,
            .origin = .{ .sema_fact = .{ .kind = "signed_storage", .ordinal = 0 } },
            .kind = .{ .logical = .{
                .role = .ensures,
                .formula = .{ .term = 3 },
            } },
        },
        .{
            .id = 2,
            .owner = .{ .function = .{ .name = "signed_result" } },
            .source = .generated(),
            .phase = .sema,
            .origin = .{ .sema_fact = .{ .kind = "signed_result", .ordinal = 0 } },
            .kind = .{ .logical = .{
                .role = .invariant,
                .formula = .{ .term = 4 },
            } },
        },
    };
    const ensures_obligation_ids = [_]obligation.Id{1};
    const invariant_obligation_ids = [_]obligation.Id{2};
    const queries = [_]obligation.VerificationQuery{
        .{
            .id = 3,
            .owner = .{ .function = .{ .name = "signed_storage" } },
            .source = .generated(),
            .phase = .report,
            .origin = .source,
            .backend = .z3,
            .kind = .obligation,
            .logical_role = .ensures,
            .obligation_ids = &ensures_obligation_ids,
        },
        .{
            .id = 4,
            .owner = .{ .function = .{ .name = "signed_result" } },
            .source = .generated(),
            .phase = .report,
            .origin = .source,
            .backend = .z3,
            .kind = .obligation,
            .logical_role = .invariant,
            .obligation_ids = &invariant_obligation_ids,
        },
    };
    const set: obligation.ObligationSet = .{
        .obligations = &obligations,
        .queries = &queries,
        .terms = &terms,
    };

    try expectEnsuresQuerySupported(set);
    try expectLogicalQuerySupported(set, .invariant);
}

test "formal obligation MLIR adapter projects div and rem value terms with opcode signedness" {
    const h = createContext();
    defer destroyContext(h);

    const text =
        \\module {
        \\  func.func @div_rem_terms(%arg0: i256, %arg1: i256) attributes {
        \\    ora.param_names = ["x", "y"],
        \\    ora.param_binding_ids = ["file:31:pattern:1", "file:31:pattern:2"]
        \\  } {
        \\    %udiv = arith.divui %arg0, %arg1 : i256
        \\    %urem = arith.remui %arg0, %arg1 : i256
        \\    %sdiv = arith.divsi %arg0, %arg1 : i256
        \\    %srem = arith.remsi %arg0, %arg1 : i256
        \\    %ucmp = arith.cmpi eq, %udiv, %urem : i256
        \\    "ora.requires"(%ucmp) : (i1) -> ()
        \\    %scmp = arith.cmpi eq, %sdiv, %srem : i256
        \\    "ora.ensures"(%scmp) : (i1) -> ()
        \\    func.return
        \\  }
        \\}
    ;

    const module = try parseModule(h.ctx, text);
    defer mlir.oraModuleDestroy(module);

    var result = try obligation_from_mlir.collect(testing.allocator, module, .{});
    defer result.deinit();

    try testing.expect(!result.set.hasBlockingDiagnostic());
    try expectEnsuresQuerySupported(result.set);

    var unsigned_divs: usize = 0;
    var unsigned_mods: usize = 0;
    var signed_divs: usize = 0;
    var signed_mods: usize = 0;
    for (result.set.terms) |term| {
        if (term != .binary) continue;
        const binary = term.binary;
        switch (binary.op) {
            .div => {
                if (binary.ty) |ty| {
                    if (ty == .compiler_type_id and ty.compiler_type_id == type_builtin.lookupBuiltinById(.u256).comptime_type_id) {
                        unsigned_divs += 1;
                    } else if (ty == .compiler_type_id and ty.compiler_type_id == type_builtin.lookupBuiltinById(.i256).comptime_type_id) {
                        signed_divs += 1;
                    }
                }
            },
            .mod => {
                if (binary.ty) |ty| {
                    if (ty == .compiler_type_id and ty.compiler_type_id == type_builtin.lookupBuiltinById(.u256).comptime_type_id) {
                        unsigned_mods += 1;
                    } else if (ty == .compiler_type_id and ty.compiler_type_id == type_builtin.lookupBuiltinById(.i256).comptime_type_id) {
                        signed_mods += 1;
                    }
                }
            },
            else => {},
        }
    }
    try testing.expect(unsigned_divs >= 1);
    try testing.expect(unsigned_mods >= 1);
    try testing.expect(signed_divs >= 1);
    try testing.expect(signed_mods >= 1);

    const rendered = try emitLeanToOwnedString(testing.allocator, result.set);
    defer testing.allocator.free(rendered);
    try testing.expect(std.mem.containsAtLeast(u8, rendered, 1, ".div"));
    try testing.expect(std.mem.containsAtLeast(u8, rendered, 1, ".mod_"));
    try testing.expect(std.mem.containsAtLeast(u8, rendered, 1, ".compilerTypeId 6"));
    try testing.expect(std.mem.containsAtLeast(u8, rendered, 1, ".compilerTypeId 12"));
}

test "formal obligation MLIR adapter projects wrapping power and shifts with compiler signedness" {
    const h = createContext();
    defer destroyContext(h);

    const text =
        \\module {
        \\  func.func @integer_terms(%arg0: i256, %arg1: i256, %small0: i8, %small1: i8) attributes {
        \\    ora.param_names = ["x", "y", "small_x", "small_y"],
        \\    ora.param_binding_ids = ["file:41:pattern:1", "file:41:pattern:2", "file:41:pattern:3", "file:41:pattern:4"]
        \\  } {
        \\    %c3 = arith.constant 3 : i256
        \\    %add = ora.add_wrapping %arg0, %arg1 {ora.integer_signed = false} : i256, i256 -> i256
        \\    %pow = ora.power %arg0, %c3 {ora.integer_checked = false, ora.integer_signed = false} : i256, i256 -> i256
        \\    %shl = ora.shl_wrapping %arg0, %arg1 {ora.integer_signed = false} : i256, i256 -> i256
        \\    %shr = ora.shr_wrapping %arg0, %arg1 {ora.integer_signed = true} : i256, i256 -> i256
        \\    %dynamic_pow = ora.power %small0, %small1 {ora.integer_checked = false, ora.integer_signed = false} : i8, i8 -> i8
        \\    %add_ok = arith.cmpi eq, %add, %arg0 : i256
        \\    %pow_ok = arith.cmpi eq, %pow, %arg0 : i256
        \\    %shl_ok = arith.cmpi eq, %shl, %arg0 : i256
        \\    %shr_ok = arith.cmpi eq, %shr, %arg0 : i256
        \\    %dynamic_pow_ok = arith.cmpi eq, %dynamic_pow, %small0 : i8
        \\    "ora.ensures"(%add_ok) : (i1) -> ()
        \\    "ora.ensures"(%pow_ok) : (i1) -> ()
        \\    "ora.ensures"(%shl_ok) : (i1) -> ()
        \\    "ora.ensures"(%shr_ok) : (i1) -> ()
        \\    "ora.ensures"(%dynamic_pow_ok) : (i1) -> ()
        \\    func.return
        \\  }
        \\}
    ;

    const module = try parseModule(h.ctx, text);
    defer mlir.oraModuleDestroy(module);

    var result = try obligation_from_mlir.collect(testing.allocator, module, .{});
    defer result.deinit();

    try testing.expect(!result.set.hasBlockingDiagnostic());
    var saw_add = false;
    var saw_pow = false;
    var saw_shl = false;
    var saw_signed_shr = false;
    for (result.set.terms) |term| {
        if (term != .binary) continue;
        const binary = term.binary;
        switch (binary.op) {
            .add => saw_add = true,
            .pow => saw_pow = true,
            .shl => saw_shl = true,
            .shr => if (binary.ty) |ty| {
                saw_signed_shr = ty == .compiler_type_id and
                    ty.compiler_type_id == type_builtin.lookupBuiltinById(.i256).comptime_type_id;
            },
            else => {},
        }
    }
    try testing.expect(saw_add);
    try testing.expect(saw_pow);
    try testing.expect(saw_shl);
    try testing.expect(saw_signed_shr);

    const rendered = try emitLeanToOwnedString(testing.allocator, result.set);
    defer testing.allocator.free(rendered);
    try testing.expect(std.mem.containsAtLeast(u8, rendered, 1, ".pow"));
    try testing.expect(std.mem.containsAtLeast(u8, rendered, 1, ".shl"));
    try testing.expect(std.mem.containsAtLeast(u8, rendered, 1, ".shr"));

    var pass = try z3_verification.VerificationPass.init(testing.allocator);
    defer pass.deinit();
    const z3_bindings = try test_helpers.z3FormalQueryBindingsFromFormalForTest(
        testing.allocator,
        result.query_bindings,
    );
    defer testing.allocator.free(z3_bindings);
    pass.setFormalQueryBindings(z3_bindings);
    var z3_manifest = try pass.collectPreparedQueryManifest(module);
    defer z3_manifest.deinit();

    var overlay = try obligation_from_z3.overlayPreparedQueryResults(
        testing.allocator,
        result.set,
        z3_manifest.rows,
    );
    defer overlay.deinit();

    var z3_ctx = try z3_verification.Z3Context.init(testing.allocator);
    defer z3_ctx.deinit();
    var adapter = obligation_to_z3.Adapter.init(&z3_ctx, testing.allocator, overlay.set);

    var parity_rows: usize = 0;
    for (overlay.set.queries) |query| {
        if (query.kind != .obligation or query.smtlib_hash == null) continue;
        const canonical = try adapter.queryHashForRow(query);
        try testing.expectEqual(query.constraint_count, canonical.constraint_count);
        try testing.expectEqual(query.smtlib_hash.?, canonical.smtlib_hash);
        parity_rows += 1;
    }
    try testing.expectEqual(@as(usize, 5), parity_rows);
}

test "formal obligation MLIR adapter rejects arithmetic with missing compiler metadata" {
    const h = createContext();
    defer destroyContext(h);

    const text =
        \\module {
        \\  func.func @missing_signedness(%arg0: i256, %arg1: i256) attributes {
        \\    ora.param_names = ["x", "y"],
        \\    ora.param_binding_ids = ["file:42:pattern:1", "file:42:pattern:2"]
        \\  } {
        \\    %shr = ora.shr_wrapping %arg0, %arg1 : i256, i256 -> i256
        \\    %pow = ora.power %arg0, %arg1 {ora.integer_signed = false} : i256, i256 -> i256
        \\    %ok = arith.cmpi eq, %shr, %arg0 : i256
        \\    %pow_ok = arith.cmpi eq, %pow, %arg0 : i256
        \\    "ora.requires"(%ok) : (i1) -> ()
        \\    "ora.requires"(%pow_ok) : (i1) -> ()
        \\    func.return
        \\  }
        \\}
    ;

    const module = try parseModule(h.ctx, text);
    defer mlir.oraModuleDestroy(module);

    var result = try obligation_from_mlir.collect(testing.allocator, module, .{});
    defer result.deinit();

    try testing.expect(result.set.hasBlockingDiagnostic());
    var saw_signedness_reason = false;
    var saw_checked_mode_reason = false;
    for (result.set.diagnostics) |diagnostic| {
        if (diagnostic.kind != .missing_type) continue;
        if (std.mem.indexOf(u8, diagnostic.message, "ora.integer_signed") != null)
            saw_signedness_reason = true;
        if (std.mem.indexOf(u8, diagnostic.message, "ora.integer_checked") != null)
            saw_checked_mode_reason = true;
    }
    try testing.expect(saw_signedness_reason);
    try testing.expect(saw_checked_mode_reason);
}

test "formal obligation Lean supports registered narrow div arithmetic width" {
    const h = createContext();
    defer destroyContext(h);

    const text =
        \\module {
        \\  func.func @narrow_div(%arg0: i32, %arg1: i32) attributes {
        \\    ora.param_names = ["x", "y"],
        \\    ora.param_binding_ids = ["file:32:pattern:1", "file:32:pattern:2"]
        \\  } {
        \\    %q = arith.divui %arg0, %arg1 : i32
        \\    %cmp = arith.cmpi eq, %q, %q : i32
        \\    "ora.ensures"(%cmp) : (i1) -> ()
        \\    func.return
        \\  }
        \\}
    ;

    const module = try parseModule(h.ctx, text);
    defer mlir.oraModuleDestroy(module);

    var result = try obligation_from_mlir.collect(testing.allocator, module, .{});
    defer result.deinit();

    try testing.expect(!result.set.hasBlockingDiagnostic());
    try expectEnsuresQuerySupported(result.set);

    const lean = try emitLeanToOwnedString(testing.allocator, result.set);
    defer testing.allocator.free(lean);
    try testing.expect(std.mem.containsAtLeast(u8, lean, 1, ".div"));
    try testing.expect(std.mem.containsAtLeast(u8, lean, 1, ".compilerTypeId 3"));
}

test "formal obligation Lean rejects signed comparison over unsigned div result" {
    const h = createContext();
    defer destroyContext(h);

    const text =
        \\module {
        \\  func.func @mixed_div_signedness(%arg0: i256, %arg1: i256) attributes {
        \\    ora.param_names = ["x", "y"],
        \\    ora.param_binding_ids = ["file:33:pattern:1", "file:33:pattern:2"]
        \\  } {
        \\    %q = arith.divui %arg0, %arg1 : i256
        \\    %cmp = arith.cmpi slt, %q, %q : i256
        \\    "ora.ensures"(%cmp) : (i1) -> ()
        \\    func.return
        \\  }
        \\}
    ;

    const module = try parseModule(h.ctx, text);
    defer mlir.oraModuleDestroy(module);

    var result = try obligation_from_mlir.collect(testing.allocator, module, .{});
    defer result.deinit();

    try testing.expect(!result.set.hasBlockingDiagnostic());
    try expectEnsuresQueryUnsupported(result.set, .mixed_arithmetic_signedness);
}

test "formal obligation MLIR adapter projects u256 scalar sload as place_read term" {
    const h = createContext();
    defer destroyContext(h);

    const text =
        \\module {
        \\  ora.contract @StorageProofs {
        \\    ora.global "reserve" : !ora.int<256, false>
        \\    func.func @check() attributes {
        \\      ora.write_slots = [],
        \\      ora.write_slots_complete = true
        \\    } {
        \\      %reserve = ora.sload "reserve" : !ora.int<256, false>
        \\      %cmp = ora.cmp "ule", %reserve, %reserve : !ora.int<256, false>, !ora.int<256, false> -> i1
        \\      "ora.ensures"(%cmp) : (i1) -> ()
        \\      func.return
        \\    }
        \\  }
        \\}
    ;

    const module = try parseModule(h.ctx, text);
    defer mlir.oraModuleDestroy(module);

    var result = try obligation_from_mlir.collect(testing.allocator, module, .{});
    defer result.deinit();

    try testing.expect(!result.set.hasBlockingDiagnostic());
    try testing.expectEqual(@as(usize, 1), countLogical(result.set, .ensures));

    const body = try expectBinaryTerm(result.set, try expectLogicalTerm(result.set, .ensures), .le);
    try expectPlaceReadTerm(result.set, body.lhs, "reserve", .storage);
    try expectPlaceReadTerm(result.set, body.rhs, "reserve", .storage);

    const report = try dumpManifestToOwnedString(testing.allocator, result.set);
    defer testing.allocator.free(report);
    try testing.expect(std.mem.containsAtLeast(u8, report, 1, "\"formula_projection_by_kind\":{\"logical\":{\"term\":1,\"origin_value\":0,\"total\":1,\"term_ratio_basis_points\":10000}"));

    const rendered = try emitLeanToOwnedString(testing.allocator, result.set);
    defer testing.allocator.free(rendered);
    try testing.expect(std.mem.containsAtLeast(u8, rendered, 2, ".placeRead { root := \"reserve\", region := .storage"));
    try testing.expect(std.mem.containsAtLeast(u8, rendered, 1, "def emittedQuery_"));
}

test "formal obligation MLIR adapter projects signed i256 scalar sload for signed comparison" {
    const h = createContext();
    defer destroyContext(h);

    const text =
        \\module {
        \\  ora.contract @StorageProofs {
        \\    ora.global "reserve" : !ora.int<256, true>
        \\    func.func @check() attributes {
        \\      ora.write_slots = [],
        \\      ora.write_slots_complete = true
        \\    } {
        \\      %reserve = ora.sload "reserve" : !ora.int<256, true>
        \\      %cmp = ora.cmp "sle", %reserve, %reserve : !ora.int<256, true>, !ora.int<256, true> -> i1
        \\      "ora.ensures"(%cmp) : (i1) -> ()
        \\      func.return
        \\    }
        \\  }
        \\}
    ;

    const module = try parseModule(h.ctx, text);
    defer mlir.oraModuleDestroy(module);

    var result = try obligation_from_mlir.collect(testing.allocator, module, .{});
    defer result.deinit();

    try testing.expect(!result.set.hasBlockingDiagnostic());
    try testing.expectEqual(@as(usize, 1), countLogical(result.set, .ensures));

    const body = try expectBinaryTerm(result.set, try expectLogicalTerm(result.set, .ensures), .sle);
    try expectTypeRefBuiltin(body.ty, .i256);
    try expectPlaceReadTerm(result.set, body.lhs, "reserve", .storage);
    try expectPlaceReadTerm(result.set, body.rhs, "reserve", .storage);
    try expectEnsuresQuerySupported(result.set);

    const lean = try emitLeanToOwnedString(testing.allocator, result.set);
    defer testing.allocator.free(lean);
    try testing.expect(std.mem.containsAtLeast(u8, lean, 1, ".sle"));
    try testing.expect(std.mem.containsAtLeast(u8, lean, 1, ".compilerTypeId 12"));
}

test "formal obligation MLIR adapter requires explicit write set before scalar sload projection" {
    const h = createContext();
    defer destroyContext(h);

    const text =
        \\module {
        \\  ora.contract @StorageProofs {
        \\    ora.global "reserve" : !ora.int<256, false>
        \\    func.func @check() {
        \\      %reserve = ora.sload "reserve" : !ora.int<256, false>
        \\      %cmp = ora.cmp "ule", %reserve, %reserve : !ora.int<256, false>, !ora.int<256, false> -> i1
        \\      "ora.ensures"(%cmp) : (i1) -> ()
        \\      func.return
        \\    }
        \\  }
        \\}
    ;

    const module = try parseModule(h.ctx, text);
    defer mlir.oraModuleDestroy(module);

    var result = try obligation_from_mlir.collect(testing.allocator, module, .{});
    defer result.deinit();

    try testing.expect(!result.set.hasBlockingDiagnostic());
    try testing.expectEqual(@as(usize, 1), countLogical(result.set, .ensures));
    try expectEnsuresQueryUnsupported(result.set, .unsupported_origin_value);

    const report = try dumpManifestToOwnedString(testing.allocator, result.set);
    defer testing.allocator.free(report);
    try testing.expect(std.mem.containsAtLeast(u8, report, 1, "\"formula_projection_by_kind\":{\"logical\":{\"term\":0,\"origin_value\":1,\"total\":1,\"term_ratio_basis_points\":0}"));
}

test "formal obligation MLIR adapter does not project scalar sload when function writes root" {
    const h = createContext();
    defer destroyContext(h);

    const text =
        \\module {
        \\  ora.contract @StorageProofs {
        \\    ora.global "balance" : !ora.int<256, false>
        \\    func.func @mutates(%arg0: !ora.int<256, false>, %arg1: !ora.int<256, false>) attributes {
        \\      ora.param_names = ["k", "next"],
        \\      ora.param_binding_ids = ["file:17:pattern:1", "file:17:pattern:2"],
        \\      ora.write_slots = ["balance"],
        \\      ora.write_slots_complete = true
        \\    } {
        \\      %before = ora.sload "balance" : !ora.int<256, false>
        \\      %req = ora.cmp "eq", %before, %arg0 : !ora.int<256, false>, !ora.int<256, false> -> i1
        \\      "ora.requires"(%req) : (i1) -> ()
        \\      ora.sstore %arg1, "balance" : !ora.int<256, false>
        \\      %after = ora.sload "balance" : !ora.int<256, false>
        \\      %ens = ora.cmp "eq", %after, %arg0 : !ora.int<256, false>, !ora.int<256, false> -> i1
        \\      "ora.ensures"(%ens) : (i1) -> ()
        \\      func.return
        \\    }
        \\  }
        \\}
    ;

    const module = try parseModule(h.ctx, text);
    defer mlir.oraModuleDestroy(module);

    var result = try obligation_from_mlir.collect(testing.allocator, module, .{});
    defer result.deinit();

    try testing.expect(!result.set.hasBlockingDiagnostic());
    try testing.expectEqual(@as(usize, 1), countAssumption(result.set, .requires));
    try testing.expectEqual(@as(usize, 1), countLogical(result.set, .ensures));
    try expectEnsuresQueryUnsupported(result.set, .unsupported_origin_value);

    const report = try dumpManifestToOwnedString(testing.allocator, result.set);
    defer testing.allocator.free(report);
    try testing.expect(std.mem.containsAtLeast(u8, report, 1, "\"formula_projection_by_kind\":{\"logical\":{\"term\":0,\"origin_value\":1,\"total\":1,\"term_ratio_basis_points\":0}"));
    try testing.expect(!std.mem.containsAtLeast(u8, report, 1, "\"tag\":\"place_read\""));
}

test "formal obligation MLIR adapter does not project scalar sload in functions with external calls" {
    const h = createContext();
    defer destroyContext(h);

    const text =
        \\module {
        \\  ora.contract @StorageProofs {
        \\    ora.global "reserve" : !ora.int<256, false>
        \\    func.func @calls(%target: !ora.address, %gas: !ora.int<256, false>, %calldata: !ora.bytes) attributes {
        \\      ora.write_slots = [],
        \\      ora.write_slots_complete = true
        \\    } {
        \\      %reserve = ora.sload "reserve" : !ora.int<256, false>
        \\      %success, %returndata = ora.external_call %target, %gas, %calldata {call_kind = "call", method_name = "ping", trait_name = "Remote"} : !ora.address, !ora.int<256, false>, !ora.bytes -> i1, !ora.bytes
        \\      %cmp = ora.cmp "ule", %reserve, %reserve : !ora.int<256, false>, !ora.int<256, false> -> i1
        \\      "ora.ensures"(%cmp) : (i1) -> ()
        \\      func.return
        \\    }
        \\  }
        \\}
    ;

    const module = try parseModule(h.ctx, text);
    defer mlir.oraModuleDestroy(module);

    var result = try obligation_from_mlir.collect(testing.allocator, module, .{});
    defer result.deinit();

    try testing.expect(!result.set.hasBlockingDiagnostic());
    try testing.expectEqual(@as(usize, 1), countLogical(result.set, .ensures));
    try expectEnsuresQueryUnsupported(result.set, .unsupported_origin_value);

    const report = try dumpManifestToOwnedString(testing.allocator, result.set);
    defer testing.allocator.free(report);
    try testing.expect(!std.mem.containsAtLeast(u8, report, 1, "\"tag\":\"place_read\""));
}

test "formal obligation MLIR adapter projects registered narrow scalar sload" {
    const h = createContext();
    defer destroyContext(h);

    const text =
        \\module {
        \\  ora.contract @StorageProofs {
        \\    ora.global "small" : !ora.int<32, false>
        \\    func.func @check() attributes {
        \\      ora.write_slots = [],
        \\      ora.write_slots_complete = true
        \\    } {
        \\      %small = ora.sload "small" : !ora.int<32, false>
        \\      %cmp = ora.cmp "ule", %small, %small : !ora.int<32, false>, !ora.int<32, false> -> i1
        \\      "ora.ensures"(%cmp) : (i1) -> ()
        \\      func.return
        \\    }
        \\  }
        \\}
    ;

    const module = try parseModule(h.ctx, text);
    defer mlir.oraModuleDestroy(module);

    var result = try obligation_from_mlir.collect(testing.allocator, module, .{});
    defer result.deinit();

    try testing.expect(!result.set.hasBlockingDiagnostic());
    try testing.expectEqual(@as(usize, 1), countLogical(result.set, .ensures));
    const body = try expectBinaryTerm(result.set, try expectLogicalTerm(result.set, .ensures), .le);
    try expectTypeRefBuiltin(body.ty, .u32);
    try expectPlaceReadTerm(result.set, body.lhs, "small", .storage);
    try expectPlaceReadTerm(result.set, body.rhs, "small", .storage);

    const report = try dumpManifestToOwnedString(testing.allocator, result.set);
    defer testing.allocator.free(report);
    try testing.expect(std.mem.containsAtLeast(u8, report, 1, "\"formula_projection_by_kind\":{\"logical\":{\"term\":1,\"origin_value\":0,\"total\":1,\"term_ratio_basis_points\":10000}"));
    try testing.expect(std.mem.containsAtLeast(u8, report, 1, "\"tag\":\"place_read\""));

    try expectEnsuresQuerySupported(result.set);
}

test "formal obligation MLIR adapter projects old written scalar sload as entry place read" {
    const h = createContext();
    defer destroyContext(h);

    const text =
        \\module {
        \\  ora.contract @StorageProofs {
        \\    ora.global "counter" : !ora.int<256, false>
        \\    func.func @snapshot() attributes {
        \\      ora.write_slots = ["counter"],
        \\      ora.write_slots_complete = true
        \\    } {
        \\      %counter = ora.sload "counter" : !ora.int<256, false>
        \\      %old = ora.old %counter : !ora.int<256, false> -> !ora.int<256, false>
        \\      %cmp = ora.cmp "ule", %old, %old : !ora.int<256, false>, !ora.int<256, false> -> i1
        \\      "ora.ensures"(%cmp) : (i1) -> ()
        \\      func.return
        \\    }
        \\  }
        \\}
    ;

    const module = try parseModule(h.ctx, text);
    defer mlir.oraModuleDestroy(module);

    var result = try obligation_from_mlir.collect(testing.allocator, module, .{});
    defer result.deinit();

    try testing.expect(!result.set.hasBlockingDiagnostic());
    try testing.expectEqual(@as(usize, 1), countLogical(result.set, .ensures));
    try expectEnsuresQuerySupported(result.set);

    const body = try expectBinaryTerm(result.set, try expectLogicalTerm(result.set, .ensures), .le);
    try expectOldPlaceReadTerm(result.set, body.lhs, "counter", .storage);
    try expectOldPlaceReadTerm(result.set, body.rhs, "counter", .storage);
    try testing.expectEqual(@as(usize, 2), countOldTerms(result.set));

    const rendered = try emitLeanToOwnedString(testing.allocator, result.set);
    defer testing.allocator.free(rendered);
    try testing.expect(std.mem.containsAtLeast(u8, rendered, 2, ".old "));
    try testing.expect(std.mem.containsAtLeast(u8, rendered, 2, ".placeRead { root := \"counter\", region := .storage"));
}

test "formal obligation MLIR adapter collapses old unwritten scalar sload to stable place read" {
    const h = createContext();
    defer destroyContext(h);

    const text =
        \\module {
        \\  ora.contract @StorageProofs {
        \\    ora.global "balance" : !ora.int<256, false>
        \\    func.func @readonly() attributes {
        \\      ora.write_slots = [],
        \\      ora.write_slots_complete = true
        \\    } {
        \\      %balance = ora.sload "balance" : !ora.int<256, false>
        \\      %old = ora.old %balance : !ora.int<256, false> -> !ora.int<256, false>
        \\      %cmp = ora.cmp "eq", %balance, %old : !ora.int<256, false>, !ora.int<256, false> -> i1
        \\      "ora.ensures"(%cmp) : (i1) -> ()
        \\      func.return
        \\    }
        \\  }
        \\}
    ;

    const module = try parseModule(h.ctx, text);
    defer mlir.oraModuleDestroy(module);

    var result = try obligation_from_mlir.collect(testing.allocator, module, .{});
    defer result.deinit();

    try testing.expect(!result.set.hasBlockingDiagnostic());
    try testing.expectEqual(@as(usize, 1), countLogical(result.set, .ensures));
    try expectEnsuresQuerySupported(result.set);

    const body = try expectBinaryTerm(result.set, try expectLogicalTerm(result.set, .ensures), .eq);
    try expectPlaceReadTerm(result.set, body.lhs, "balance", .storage);
    try expectPlaceReadTerm(result.set, body.rhs, "balance", .storage);
    try testing.expectEqual(@as(usize, 0), countOldTerms(result.set));

    const rendered = try emitLeanToOwnedString(testing.allocator, result.set);
    defer testing.allocator.free(rendered);
    try testing.expect(std.mem.indexOf(u8, rendered, ".old ") == null);
    try testing.expect(std.mem.containsAtLeast(u8, rendered, 2, ".placeRead { root := \"balance\", region := .storage"));
}

test "formal obligation MLIR adapter does not collapse old with bare written-root read" {
    const h = createContext();
    defer destroyContext(h);

    const text =
        \\module {
        \\  ora.contract @StorageProofs {
        \\    ora.global "balance" : !ora.int<256, false>
        \\    func.func @mutates() attributes {
        \\      ora.write_slots = ["balance"],
        \\      ora.write_slots_complete = true
        \\    } {
        \\      %balance = ora.sload "balance" : !ora.int<256, false>
        \\      %old = ora.old %balance : !ora.int<256, false> -> !ora.int<256, false>
        \\      %cmp = ora.cmp "eq", %balance, %old : !ora.int<256, false>, !ora.int<256, false> -> i1
        \\      "ora.ensures"(%cmp) : (i1) -> ()
        \\      func.return
        \\    }
        \\  }
        \\}
    ;

    const module = try parseModule(h.ctx, text);
    defer mlir.oraModuleDestroy(module);

    var result = try obligation_from_mlir.collect(testing.allocator, module, .{});
    defer result.deinit();

    try testing.expect(!result.set.hasBlockingDiagnostic());
    try testing.expectEqual(@as(usize, 1), countLogical(result.set, .ensures));
    try expectEnsuresQueryUnsupported(result.set, .unsupported_origin_value);
    try testing.expectEqual(@as(usize, 0), countOldTerms(result.set));
}

test "formal obligation MLIR adapter requires write metadata before old scalar sload projection" {
    const h = createContext();
    defer destroyContext(h);

    const text =
        \\module {
        \\  ora.contract @StorageProofs {
        \\    ora.global "counter" : !ora.int<256, false>
        \\    func.func @snapshot() {
        \\      %counter = ora.sload "counter" : !ora.int<256, false>
        \\      %old = ora.old %counter : !ora.int<256, false> -> !ora.int<256, false>
        \\      %cmp = ora.cmp "ule", %old, %old : !ora.int<256, false>, !ora.int<256, false> -> i1
        \\      "ora.ensures"(%cmp) : (i1) -> ()
        \\      func.return
        \\    }
        \\  }
        \\}
    ;

    const module = try parseModule(h.ctx, text);
    defer mlir.oraModuleDestroy(module);

    var result = try obligation_from_mlir.collect(testing.allocator, module, .{});
    defer result.deinit();

    try testing.expect(!result.set.hasBlockingDiagnostic());
    try testing.expectEqual(@as(usize, 1), countLogical(result.set, .ensures));
    try expectEnsuresQueryUnsupported(result.set, .unsupported_origin_value);
    try testing.expectEqual(@as(usize, 0), countOldTerms(result.set));
}

test "formal obligation MLIR adapter requires complete write metadata before scalar sload projection" {
    const h = createContext();
    defer destroyContext(h);

    const text =
        \\module {
        \\  ora.contract @StorageProofs {
        \\    ora.global "reserve" : !ora.int<256, false>
        \\    func.func @check() attributes {
        \\      ora.write_slots = []
        \\    } {
        \\      %reserve = ora.sload "reserve" : !ora.int<256, false>
        \\      %cmp = ora.cmp "ule", %reserve, %reserve : !ora.int<256, false>, !ora.int<256, false> -> i1
        \\      "ora.ensures"(%cmp) : (i1) -> ()
        \\      func.return
        \\    }
        \\  }
        \\}
    ;

    const module = try parseModule(h.ctx, text);
    defer mlir.oraModuleDestroy(module);

    var result = try obligation_from_mlir.collect(testing.allocator, module, .{});
    defer result.deinit();

    try testing.expect(!result.set.hasBlockingDiagnostic());
    try testing.expectEqual(@as(usize, 1), countLogical(result.set, .ensures));
    try expectEnsuresQueryUnsupported(result.set, .unsupported_origin_value);

    const report = try dumpManifestToOwnedString(testing.allocator, result.set);
    defer testing.allocator.free(report);
    try testing.expect(!std.mem.containsAtLeast(u8, report, 1, "\"tag\":\"place_read\""));
}

test "formal obligation MLIR adapter does not project old scalar sload in functions with external calls" {
    const h = createContext();
    defer destroyContext(h);

    const text =
        \\module {
        \\  ora.contract @StorageProofs {
        \\    ora.global "counter" : !ora.int<256, false>
        \\    func.func @snapshot(%target: !ora.address, %gas: !ora.int<256, false>, %calldata: !ora.bytes) attributes {
        \\      ora.write_slots = ["counter"],
        \\      ora.write_slots_complete = true
        \\    } {
        \\      %counter = ora.sload "counter" : !ora.int<256, false>
        \\      %old = ora.old %counter : !ora.int<256, false> -> !ora.int<256, false>
        \\      %success, %returndata = ora.external_call %target, %gas, %calldata {call_kind = "call", method_name = "ping", trait_name = "Remote"} : !ora.address, !ora.int<256, false>, !ora.bytes -> i1, !ora.bytes
        \\      %cmp = ora.cmp "ule", %old, %old : !ora.int<256, false>, !ora.int<256, false> -> i1
        \\      "ora.ensures"(%cmp) : (i1) -> ()
        \\      func.return
        \\    }
        \\  }
        \\}
    ;

    const module = try parseModule(h.ctx, text);
    defer mlir.oraModuleDestroy(module);

    var result = try obligation_from_mlir.collect(testing.allocator, module, .{});
    defer result.deinit();

    try testing.expect(!result.set.hasBlockingDiagnostic());
    try testing.expectEqual(@as(usize, 1), countLogical(result.set, .ensures));
    try expectEnsuresQueryUnsupported(result.set, .unsupported_origin_value);
    try testing.expectEqual(@as(usize, 0), countOldTerms(result.set));
}

test "formal obligation MLIR adapter keeps arbitrary old expression outside Lean term fragment" {
    const h = createContext();
    defer destroyContext(h);

    const text =
        \\module {
        \\  func.func @old_expr(%arg0: !ora.int<256, false>) attributes {
        \\    ora.param_names = ["x"],
        \\    ora.param_binding_ids = ["file:23:pattern:1"],
        \\    ora.write_slots = [],
        \\    ora.write_slots_complete = true
        \\  } {
        \\    %sum = ora.add_wrapping %arg0, %arg0 : !ora.int<256, false>, !ora.int<256, false> -> !ora.int<256, false>
        \\    %old = ora.old %sum : !ora.int<256, false> -> !ora.int<256, false>
        \\    %cmp = ora.cmp "ule", %old, %old : !ora.int<256, false>, !ora.int<256, false> -> i1
        \\    "ora.ensures"(%cmp) : (i1) -> ()
        \\    func.return
        \\  }
        \\}
    ;

    const module = try parseModule(h.ctx, text);
    defer mlir.oraModuleDestroy(module);

    var result = try obligation_from_mlir.collect(testing.allocator, module, .{});
    defer result.deinit();

    try testing.expect(!result.set.hasBlockingDiagnostic());
    try testing.expectEqual(@as(usize, 1), countLogical(result.set, .ensures));
    try expectEnsuresQueryUnsupported(result.set, .unsupported_origin_value);
    try testing.expectEqual(@as(usize, 0), countOldTerms(result.set));
}

test "formal obligation MLIR adapter supports old written scalar sload in invariant formulas" {
    const h = createContext();
    defer destroyContext(h);

    const text =
        \\module {
        \\  ora.contract @StorageProofs {
        \\    ora.global "total" : !ora.int<256, false>
        \\    func.func @loop_snapshot() attributes {
        \\      ora.write_slots = ["total"],
        \\      ora.write_slots_complete = true
        \\    } {
        \\      %total = ora.sload "total" : !ora.int<256, false>
        \\      %old = ora.old %total : !ora.int<256, false> -> !ora.int<256, false>
        \\      %cmp = ora.cmp "ule", %old, %old : !ora.int<256, false>, !ora.int<256, false> -> i1
        \\      "ora.invariant"(%cmp) : (i1) -> ()
        \\      func.return
        \\    }
        \\  }
        \\}
    ;

    const module = try parseModule(h.ctx, text);
    defer mlir.oraModuleDestroy(module);

    var result = try obligation_from_mlir.collect(testing.allocator, module, .{});
    defer result.deinit();

    try testing.expect(!result.set.hasBlockingDiagnostic());
    try testing.expectEqual(@as(usize, 1), countLogical(result.set, .invariant));
    try expectLogicalQuerySupported(result.set, .invariant);

    const body = try expectBinaryTerm(result.set, try expectLogicalTerm(result.set, .invariant), .le);
    try expectOldPlaceReadTerm(result.set, body.lhs, "total", .storage);
    try expectOldPlaceReadTerm(result.set, body.rhs, "total", .storage);
    try testing.expectEqual(@as(usize, 2), countOldTerms(result.set));
}

test "formal obligation MLIR adapter rejects invariant with old and bare written-root read" {
    const h = createContext();
    defer destroyContext(h);

    const text =
        \\module {
        \\  ora.contract @StorageProofs {
        \\    ora.global "total" : !ora.int<256, false>
        \\    func.func @loop_snapshot() attributes {
        \\      ora.write_slots = ["total"],
        \\      ora.write_slots_complete = true
        \\    } {
        \\      %total = ora.sload "total" : !ora.int<256, false>
        \\      %old = ora.old %total : !ora.int<256, false> -> !ora.int<256, false>
        \\      %cmp = ora.cmp "eq", %old, %total : !ora.int<256, false>, !ora.int<256, false> -> i1
        \\      "ora.invariant"(%cmp) : (i1) -> ()
        \\      func.return
        \\    }
        \\  }
        \\}
    ;

    const module = try parseModule(h.ctx, text);
    defer mlir.oraModuleDestroy(module);

    var result = try obligation_from_mlir.collect(testing.allocator, module, .{});
    defer result.deinit();

    try testing.expect(!result.set.hasBlockingDiagnostic());
    try testing.expectEqual(@as(usize, 1), countLogical(result.set, .invariant));
    try expectLogicalQueryUnsupported(result.set, .invariant, .unsupported_origin_value);
    try testing.expectEqual(@as(usize, 1), countOldTerms(result.set));
}

test "formal obligation MLIR adapter projects read-only keyed map sload as place_read term" {
    const h = createContext();
    defer destroyContext(h);

    const text =
        \\module {
        \\  ora.contract @StorageProofs {
        \\    ora.global "balances" : !ora.map<!ora.int<256, false>, !ora.int<256, false>>
        \\    func.func @check(%owner: !ora.int<256, false>) attributes {
        \\      ora.param_names = ["owner"],
        \\      ora.param_binding_ids = ["file:41:pattern:1"],
        \\      ora.write_slots = [],
        \\      ora.write_slots_complete = true
        \\    } {
        \\      %balances = ora.sload "balances" : !ora.map<!ora.int<256, false>, !ora.int<256, false>>
        \\      %balance = "ora.map_get"(%balances, %owner) : (!ora.map<!ora.int<256, false>, !ora.int<256, false>>, !ora.int<256, false>) -> !ora.int<256, false>
        \\      %cmp = ora.cmp "eq", %balance, %balance : !ora.int<256, false>, !ora.int<256, false> -> i1
        \\      "ora.ensures"(%cmp) : (i1) -> ()
        \\      func.return
        \\    }
        \\  }
        \\}
    ;

    const module = try parseModule(h.ctx, text);
    defer mlir.oraModuleDestroy(module);

    var result = try obligation_from_mlir.collect(testing.allocator, module, .{});
    defer result.deinit();

    try testing.expect(!result.set.hasBlockingDiagnostic());
    try testing.expectEqual(@as(usize, 1), countLogical(result.set, .ensures));
    try expectEnsuresQuerySupported(result.set);

    const quantified = try expectQuantifiedTerm(result.set, try expectLogicalTerm(result.set, .ensures), .forall, "owner");
    const body = try expectBinaryTerm(result.set, quantified.body, .eq);
    try expectPlaceReadTermWithParameterKeyCount(result.set, body.lhs, "balances", 1);
    try expectPlaceReadTermWithParameterKeyCount(result.set, body.rhs, "balances", 1);

    const rendered = try emitLeanToOwnedString(testing.allocator, result.set);
    defer testing.allocator.free(rendered);
    try testing.expect(std.mem.containsAtLeast(u8, rendered, 2, "root := \"balances\""));
    try testing.expect(std.mem.containsAtLeast(u8, rendered, 2, "keys := [.parameter { file_id := "));
}

test "formal obligation MLIR adapter ties parameter place keys to free variable ids" {
    const h = createContext();
    defer destroyContext(h);

    const text =
        \\module {
        \\  ora.contract @StorageProofs {
        \\    ora.global "balances" : !ora.map<!ora.int<256, false>, !ora.int<256, false>>
        \\    func.func @check(%user: !ora.int<256, false>, %other: !ora.int<256, false>) attributes {
        \\      ora.param_names = ["user", "other"],
        \\      ora.param_binding_ids = ["file:77:pattern:1", "file:77:pattern:2"],
        \\      ora.write_slots = [],
        \\      ora.write_slots_complete = true
        \\    } {
        \\      %req = ora.cmp "ule", %user, %other : !ora.int<256, false>, !ora.int<256, false> -> i1
        \\      "ora.requires"(%req) : (i1) -> ()
        \\      %balances = ora.sload "balances" : !ora.map<!ora.int<256, false>, !ora.int<256, false>>
        \\      %balance = "ora.map_get"(%balances, %user) : (!ora.map<!ora.int<256, false>, !ora.int<256, false>>, !ora.int<256, false>) -> !ora.int<256, false>
        \\      %cmp = ora.cmp "eq", %balance, %balance : !ora.int<256, false>, !ora.int<256, false> -> i1
        \\      "ora.ensures"(%cmp) : (i1) -> ()
        \\      func.return
        \\    }
        \\  }
        \\}
    ;

    const module = try parseModule(h.ctx, text);
    defer mlir.oraModuleDestroy(module);

    var result = try obligation_from_mlir.collect(testing.allocator, module, .{});
    defer result.deinit();

    try testing.expect(!result.set.hasBlockingDiagnostic());
    try testing.expectEqual(@as(usize, 1), countAssumption(result.set, .requires));
    try testing.expectEqual(@as(usize, 1), countLogical(result.set, .ensures));
    try expectEnsuresQuerySupported(result.set);

    var user_var_id: ?obligation.FreeVarId = null;
    var other_var_id: ?obligation.FreeVarId = null;
    var place_key_id: ?obligation.FreeVarId = null;
    for (result.set.terms) |term| {
        switch (term) {
            .variable => |variable| switch (variable) {
                .free => |free| {
                    if (std.mem.eql(u8, free.name, "user")) user_var_id = free.id;
                    if (std.mem.eql(u8, free.name, "other")) other_var_id = free.id;
                },
                .bound => {},
            },
            .place_read => |place| {
                if (std.mem.eql(u8, place.root, "balances")) {
                    try testing.expectEqual(@as(usize, 1), place.keys.len);
                    try testing.expect(place.keys[0] == .parameter);
                    place_key_id = place.keys[0].parameter;
                }
            },
            else => {},
        }
    }

    const user_id = user_var_id orelse return error.TestUnexpectedResult;
    const other_id = other_var_id orelse return error.TestUnexpectedResult;
    const key_id = place_key_id orelse return error.TestUnexpectedResult;
    try testing.expect(obligation.freeVarIdEql(user_id, key_id));
    try testing.expect(!obligation.freeVarIdEql(other_id, key_id));

    const lean = try emitLeanToOwnedString(testing.allocator, result.set);
    defer testing.allocator.free(lean);
    try testing.expect(std.mem.containsAtLeast(u8, lean, 1, ".free { id := { file_id := 77, pattern_id := 1 }, name := \"user\""));
    try testing.expect(std.mem.containsAtLeast(u8, lean, 1, "keys := [.parameter { file_id := 77, pattern_id := 1 }]"));
}

test "formal obligation source collector rejects loop block argument map keys" {
    const source_text =
        \\contract StorageProjection {
        \\    storage balances: map<u256, u256>;
        \\
        \\    pub fn check(x: u256, cap: u256)
        \\        ensures balances[x] <= cap
        \\    {
        \\        var i: u256 = 0;
        \\        while (i < x)
        \\            invariant balances[i] <= cap
        \\        {
        \\            i = i + 1;
        \\        }
        \\    }
        \\}
    ;

    const rendered = try renderOraMlirForSource(source_text);
    defer testing.allocator.free(rendered);

    try testing.expect(std.mem.containsAtLeast(u8, rendered, 1, "func.func @check"));
    try testing.expect(std.mem.containsAtLeast(u8, rendered, 1, "ora.invariant"));
    try testing.expect(std.mem.containsAtLeast(u8, rendered, 1, "ora.write_slots_complete = true"));

    const h = createContext();
    defer destroyContext(h);

    const module = try parseModule(h.ctx, rendered);
    defer mlir.oraModuleDestroy(module);

    var result = try obligation_from_mlir.collect(testing.allocator, module, .{});
    defer result.deinit();

    try testing.expect(!result.set.hasBlockingDiagnostic());
    try testing.expectEqual(@as(usize, 1), countLogical(result.set, .ensures));
    try testing.expectEqual(@as(usize, 1), countLogical(result.set, .invariant));
    try expectEnsuresQuerySupported(result.set);
    try expectLogicalQueryUnsupported(result.set, .invariant, .unsupported_origin_value);
}

test "formal obligation source collector projects environment map keys" {
    const source_text =
        \\contract EnvStorageProjection {
        \\    storage balances: map<address, u256>;
        \\    storage origins: map<address, u256>;
        \\
        \\    pub fn check()
        \\        ensures balances[std.msg.sender()] == balances[std.msg.sender()]
        \\        ensures origins[std.tx.origin()] == origins[std.tx.origin()]
        \\    {
        \\    }
        \\}
    ;

    const rendered = try renderOraMlirForSource(source_text);
    defer testing.allocator.free(rendered);

    try testing.expect(std.mem.containsAtLeast(u8, rendered, 1, "ora.evm.caller"));
    try testing.expect(std.mem.containsAtLeast(u8, rendered, 1, "ora.evm.origin"));
    try testing.expect(std.mem.containsAtLeast(u8, rendered, 1, "ora.write_slots_complete = true"));

    const h = createContext();
    defer destroyContext(h);

    const module = try parseModule(h.ctx, rendered);
    defer mlir.oraModuleDestroy(module);

    var result = try obligation_from_mlir.collect(testing.allocator, module, .{});
    defer result.deinit();

    try testing.expect(!result.set.hasBlockingDiagnostic());
    try testing.expectEqual(@as(usize, 2), countLogical(result.set, .ensures));

    var sender_reads: usize = 0;
    var origin_reads: usize = 0;
    for (result.set.terms) |term| {
        if (term != .place_read) continue;
        if (std.mem.eql(u8, term.place_read.root, "balances")) {
            try testing.expectEqual(@as(usize, 1), term.place_read.keys.len);
            try testing.expect(term.place_read.keys[0] == .msg_sender);
            sender_reads += 1;
        } else if (std.mem.eql(u8, term.place_read.root, "origins")) {
            try testing.expectEqual(@as(usize, 1), term.place_read.keys.len);
            try testing.expect(term.place_read.keys[0] == .tx_origin);
            origin_reads += 1;
        }
    }
    try testing.expectEqual(@as(usize, 2), sender_reads);
    try testing.expectEqual(@as(usize, 2), origin_reads);

    const lean = try emitLeanToOwnedString(testing.allocator, result.set);
    defer testing.allocator.free(lean);
    try testing.expect(std.mem.containsAtLeast(u8, lean, 1, ".msgSender"));
    try testing.expect(std.mem.containsAtLeast(u8, lean, 1, ".txOrigin"));
}

test "formal obligation source collector projects statically disjoint constant map paths" {
    const source_text =
        \\contract StorageProjection {
        \\    storage buckets: map<u256, u256>;
        \\
        \\    pub fn check()
        \\        ensures buckets[2] == old(buckets[2])
        \\    {
        \\        buckets[1] = 7;
        \\    }
        \\}
    ;

    const rendered = try renderOraMlirForSource(source_text);
    defer testing.allocator.free(rendered);

    try testing.expect(std.mem.containsAtLeast(u8, rendered, 1, "ora.write_slots = [\"buckets[1]\"]"));
    try testing.expect(std.mem.containsAtLeast(u8, rendered, 1, "ora.write_slots_complete = true"));

    const h = createContext();
    defer destroyContext(h);

    const module = try parseModule(h.ctx, rendered);
    defer mlir.oraModuleDestroy(module);

    var result = try obligation_from_mlir.collect(testing.allocator, module, .{});
    defer result.deinit();

    try testing.expect(!result.set.hasBlockingDiagnostic());
    try testing.expectEqual(@as(usize, 1), countLogical(result.set, .ensures));
    try expectEnsuresQuerySupported(result.set);

    const body = try expectBinaryTerm(result.set, try expectLogicalTerm(result.set, .ensures), .eq);
    try expectPlaceReadTermWithConstantKeys(result.set, body.lhs, "buckets", &.{"2"});
    try expectPlaceReadTermWithConstantKeys(result.set, body.rhs, "buckets", &.{"2"});
    try testing.expectEqual(@as(usize, 0), countOldTerms(result.set));

    const report = try dumpManifestToOwnedString(testing.allocator, result.set);
    defer testing.allocator.free(report);
    try testing.expect(std.mem.containsAtLeast(u8, report, 1, "\"formula_projection_by_kind\":{\"logical\":{\"term\":1,\"origin_value\":0,\"total\":1,\"term_ratio_basis_points\":10000}"));

    const lean = try emitLeanToOwnedString(testing.allocator, result.set);
    defer testing.allocator.free(lean);
    try testing.expect(std.mem.containsAtLeast(u8, lean, 2, "root := \"buckets\""));
    try testing.expect(std.mem.containsAtLeast(u8, lean, 2, "keys := [.constant \"2\"]"));
    try testing.expect(std.mem.indexOf(u8, lean, ".old ") == null);
}

test "formal obligation source collector keeps conditional map post-state equality outside Lean projection" {
    const source_text =
        \\contract StorageProjection {
        \\    storage balances: map<u256, u256>;
        \\
        \\    pub fn check(user: u256, other: u256, value: u256)
        \\        requires user != other
        \\        ensures balances[other] == old(balances[other])
        \\    {
        \\        balances[user] = value;
        \\    }
        \\}
    ;

    const rendered = try renderOraMlirForSource(source_text);
    defer testing.allocator.free(rendered);

    try testing.expect(std.mem.containsAtLeast(u8, rendered, 1, "ora.write_slots = [\"balances[param#0]\"]"));
    try testing.expect(std.mem.containsAtLeast(u8, rendered, 1, "ora.write_slots_complete = true"));

    const h = createContext();
    defer destroyContext(h);

    const module = try parseModule(h.ctx, rendered);
    defer mlir.oraModuleDestroy(module);

    var result = try obligation_from_mlir.collect(testing.allocator, module, .{});
    defer result.deinit();

    try testing.expect(!result.set.hasBlockingDiagnostic());
    try testing.expectEqual(@as(usize, 1), countAssumption(result.set, .requires));
    try testing.expectEqual(@as(usize, 1), countLogical(result.set, .ensures));
    try testing.expectEqual(@as(usize, 1), countEffectFrame(result.set, .read_preserved_by_key_evidence));
    try expectEffectFrameQuerySupported(result.set, .read_preserved_by_key_evidence);
    try expectEnsuresQueryUnsupported(result.set, .unsupported_origin_value);
}

test "formal obligation source queries do not mix same-ordinal parameter owners" {
    const source_text =
        \\contract StorageProjection {
        \\    storage balances: map<u256, u256>;
        \\
        \\    pub fn first(x: u256, cap: u256)
        \\        requires x <= cap
        \\        ensures balances[x] == old(balances[x])
        \\    {
        \\    }
        \\
        \\    pub fn second(x: u256, cap: u256)
        \\        requires x <= cap
        \\        ensures balances[x] == old(balances[x])
        \\    {
        \\    }
        \\}
    ;

    const rendered = try renderOraMlirForSource(source_text);
    defer testing.allocator.free(rendered);

    try testing.expect(std.mem.containsAtLeast(u8, rendered, 1, "func.func @first"));
    try testing.expect(std.mem.containsAtLeast(u8, rendered, 1, "func.func @second"));
    try testing.expect(std.mem.containsAtLeast(u8, rendered, 2, "ora.param_binding_ids"));

    const h = createContext();
    defer destroyContext(h);

    const module = try parseModule(h.ctx, rendered);
    defer mlir.oraModuleDestroy(module);

    var result = try obligation_from_mlir.collect(testing.allocator, module, .{});
    defer result.deinit();

    try testing.expect(!result.set.hasBlockingDiagnostic());
    try testing.expectEqual(@as(usize, 2), countAssumption(result.set, .requires));
    try testing.expectEqual(@as(usize, 2), countLogical(result.set, .ensures));
    try expectQueriesOwnerScoped(result.set);
}

test "formal obligation source collector does not split msg.sender from tx.origin" {
    const source_text =
        \\contract StorageProjection {
        \\    storage balances: map<address, u256>;
        \\
        \\    pub fn check(value: u256)
        \\        ensures balances[std.tx.origin()] == old(balances[std.tx.origin()])
        \\    {
        \\        balances[std.msg.sender()] = value;
        \\    }
        \\}
    ;

    const rendered = try renderOraMlirForSource(source_text);
    defer testing.allocator.free(rendered);

    try testing.expect(std.mem.containsAtLeast(u8, rendered, 1, "ora.write_slots = [\"balances[msg.sender]\"]"));
    try testing.expect(std.mem.containsAtLeast(u8, rendered, 1, "ora.write_slots_complete = true"));

    const h = createContext();
    defer destroyContext(h);

    const module = try parseModule(h.ctx, rendered);
    defer mlir.oraModuleDestroy(module);

    var result = try obligation_from_mlir.collect(testing.allocator, module, .{});
    defer result.deinit();

    try testing.expect(!result.set.hasBlockingDiagnostic());
    try expectEnsuresQueryUnsupported(result.set, .unsupported_origin_value);
}

test "formal obligation MLIR adapter requires complete write metadata before keyed map projection" {
    const h = createContext();
    defer destroyContext(h);

    const text =
        \\module {
        \\  ora.contract @StorageProofs {
        \\    ora.global "balances" : !ora.map<!ora.int<256, false>, !ora.int<256, false>>
        \\    func.func @check(%owner: !ora.int<256, false>) attributes {
        \\      ora.param_names = ["owner"],
        \\      ora.param_binding_ids = ["file:42:pattern:1"],
        \\      ora.write_slots = []
        \\    } {
        \\      %balances = ora.sload "balances" : !ora.map<!ora.int<256, false>, !ora.int<256, false>>
        \\      %balance = "ora.map_get"(%balances, %owner) : (!ora.map<!ora.int<256, false>, !ora.int<256, false>>, !ora.int<256, false>) -> !ora.int<256, false>
        \\      %cmp = ora.cmp "eq", %balance, %balance : !ora.int<256, false>, !ora.int<256, false> -> i1
        \\      "ora.ensures"(%cmp) : (i1) -> ()
        \\      func.return
        \\    }
        \\  }
        \\}
    ;

    const module = try parseModule(h.ctx, text);
    defer mlir.oraModuleDestroy(module);

    var result = try obligation_from_mlir.collect(testing.allocator, module, .{});
    defer result.deinit();

    try testing.expect(!result.set.hasBlockingDiagnostic());
    try expectEnsuresQueryUnsupported(result.set, .unsupported_origin_value);

    const report = try dumpManifestToOwnedString(testing.allocator, result.set);
    defer testing.allocator.free(report);
    try testing.expect(!std.mem.containsAtLeast(u8, report, 1, "\"tag\":\"place_read\""));
}

test "formal obligation MLIR adapter blocks parameter effect paths without binding ids" {
    const h = createContext();
    defer destroyContext(h);

    const text =
        \\module {
        \\  ora.contract @StorageProofs {
        \\    ora.global "balances" : !ora.map<!ora.int<256, false>, !ora.int<256, false>>
        \\    func.func @check(%owner: !ora.int<256, false>) attributes {
        \\      ora.write_slots = ["balances[param#0]"],
        \\      ora.write_slots_complete = true
        \\    } {
        \\      %balances = ora.sload "balances" : !ora.map<!ora.int<256, false>, !ora.int<256, false>>
        \\      %balance = "ora.map_get"(%balances, %owner) : (!ora.map<!ora.int<256, false>, !ora.int<256, false>>, !ora.int<256, false>) -> !ora.int<256, false>
        \\      %cmp = ora.cmp "eq", %balance, %balance : !ora.int<256, false>, !ora.int<256, false> -> i1
        \\      "ora.ensures"(%cmp) : (i1) -> ()
        \\      func.return
        \\    }
        \\  }
        \\}
    ;

    const module = try parseModule(h.ctx, text);
    defer mlir.oraModuleDestroy(module);

    var result = try obligation_from_mlir.collect(testing.allocator, module, .{});
    defer result.deinit();

    try testing.expect(result.set.hasBlockingDiagnostic());
    try testing.expectEqual(obligation.ArtifactBlockReason.blocking_diagnostic, result.set.artifactDecision().blocked);
    var saw_missing_binding = false;
    for (result.set.diagnostics) |diagnostic| {
        if (diagnostic.kind != .missing_effect_path) continue;
        if (std.mem.containsAtLeast(
            u8,
            diagnostic.message,
            1,
            "references parameter 0 but ora.param_binding_ids is missing or too short",
        )) {
            saw_missing_binding = true;
        }
    }
    try testing.expect(saw_missing_binding);
}

test "formal obligation MLIR adapter does not project keyed map read when function writes root" {
    const h = createContext();
    defer destroyContext(h);

    const text =
        \\module {
        \\  ora.contract @StorageProofs {
        \\    ora.global "balances" : !ora.map<!ora.int<256, false>, !ora.int<256, false>>
        \\    func.func @mutates(%owner: !ora.int<256, false>) attributes {
        \\      ora.param_names = ["owner"],
        \\      ora.param_binding_ids = ["file:43:pattern:1"],
        \\      ora.write_slots = ["balances[param#0]"],
        \\      ora.write_slots_complete = true
        \\    } {
        \\      %balances = ora.sload "balances" : !ora.map<!ora.int<256, false>, !ora.int<256, false>>
        \\      %balance = "ora.map_get"(%balances, %owner) : (!ora.map<!ora.int<256, false>, !ora.int<256, false>>, !ora.int<256, false>) -> !ora.int<256, false>
        \\      %cmp = ora.cmp "eq", %balance, %balance : !ora.int<256, false>, !ora.int<256, false> -> i1
        \\      "ora.ensures"(%cmp) : (i1) -> ()
        \\      func.return
        \\    }
        \\  }
        \\}
    ;

    const module = try parseModule(h.ctx, text);
    defer mlir.oraModuleDestroy(module);

    var result = try obligation_from_mlir.collect(testing.allocator, module, .{});
    defer result.deinit();

    try testing.expect(!result.set.hasBlockingDiagnostic());
    try expectEnsuresQueryUnsupported(result.set, .unsupported_origin_value);

    const report = try dumpManifestToOwnedString(testing.allocator, result.set);
    defer testing.allocator.free(report);
    try testing.expect(!std.mem.containsAtLeast(u8, report, 1, "\"tag\":\"place_read\""));
}

test "formal obligation MLIR adapter blocks prefix map path projection" {
    const h = createContext();
    defer destroyContext(h);

    const text =
        \\module {
        \\  ora.contract @StorageProofs {
        \\    ora.global "allowances" : !ora.map<!ora.int<256, false>, !ora.map<!ora.int<256, false>, !ora.int<256, false>>>
        \\    func.func @mutates(%owner: !ora.int<256, false>, %spender: !ora.int<256, false>) attributes {
        \\      ora.param_names = ["owner", "spender"],
        \\      ora.param_binding_ids = ["file:143:pattern:1", "file:143:pattern:2"],
        \\      ora.write_slots = ["allowances[param#0]"],
        \\      ora.write_slots_complete = true
        \\    } {
        \\      %allowances = ora.sload "allowances" : !ora.map<!ora.int<256, false>, !ora.map<!ora.int<256, false>, !ora.int<256, false>>>
        \\      %owner_allowances = "ora.map_get"(%allowances, %owner) : (!ora.map<!ora.int<256, false>, !ora.map<!ora.int<256, false>, !ora.int<256, false>>>, !ora.int<256, false>) -> !ora.map<!ora.int<256, false>, !ora.int<256, false>>
        \\      %allowance = "ora.map_get"(%owner_allowances, %spender) : (!ora.map<!ora.int<256, false>, !ora.int<256, false>>, !ora.int<256, false>) -> !ora.int<256, false>
        \\      %old = ora.old %allowance : !ora.int<256, false> -> !ora.int<256, false>
        \\      %cmp = ora.cmp "eq", %allowance, %old : !ora.int<256, false>, !ora.int<256, false> -> i1
        \\      "ora.ensures"(%cmp) : (i1) -> ()
        \\      func.return
        \\    }
        \\  }
        \\}
    ;

    const module = try parseModule(h.ctx, text);
    defer mlir.oraModuleDestroy(module);

    var result = try obligation_from_mlir.collect(testing.allocator, module, .{});
    defer result.deinit();

    try testing.expect(!result.set.hasBlockingDiagnostic());
    try expectEnsuresQueryUnsupported(result.set, .unsupported_origin_value);
    try testing.expectEqual(@as(usize, 0), countOldTerms(result.set));
}

test "formal obligation source collector does not project map read when helper writes root" {
    const source_text =
        \\contract StorageProjection {
        \\    storage balances: map<u256, u256>;
        \\
        \\    fn overwrite(owner: u256, value: u256) {
        \\        balances[owner] = value;
        \\    }
        \\
        \\    pub fn check(owner: u256)
        \\        ensures balances[owner] == balances[owner]
        \\    {
        \\        overwrite(owner, 1);
        \\    }
        \\}
    ;

    const rendered = try renderOraMlirForSource(source_text);
    defer testing.allocator.free(rendered);

    try testing.expect(std.mem.containsAtLeast(u8, rendered, 1, "func.func @check"));
    try testing.expect(std.mem.containsAtLeast(u8, rendered, 1, "ora.write_slots"));
    try testing.expect(std.mem.containsAtLeast(u8, rendered, 1, "\"balances"));
    try testing.expect(std.mem.containsAtLeast(u8, rendered, 1, "ora.write_slots_complete = true"));

    const h = createContext();
    defer destroyContext(h);

    const module = try parseModule(h.ctx, rendered);
    defer mlir.oraModuleDestroy(module);

    var result = try obligation_from_mlir.collect(testing.allocator, module, .{});
    defer result.deinit();

    try testing.expect(!result.set.hasBlockingDiagnostic());
    try expectEnsuresQueryUnsupported(result.set, .unsupported_origin_value);

    const report = try dumpManifestToOwnedString(testing.allocator, result.set);
    defer testing.allocator.free(report);
    try testing.expect(!std.mem.containsAtLeast(u8, report, 1, "\"tag\":\"place_read\""));
}

test "formal obligation MLIR adapter emits read preserved frame for static constant disjoint paths" {
    const h = createContext();
    defer destroyContext(h);

    const text =
        \\module {
        \\  func.func @effect_surface() attributes {
        \\    ora.effect = "readwrites",
        \\    ora.modifies_slots = ["buckets[1]"],
        \\    ora.read_slots = ["buckets[2]"],
        \\    ora.write_slots = ["buckets[1]"],
        \\    ora.write_slots_complete = true
        \\  } {
        \\    func.return
        \\  }
        \\}
    ;

    const module = try parseModule(h.ctx, text);
    defer mlir.oraModuleDestroy(module);

    var result = try obligation_from_mlir.collect(testing.allocator, module, .{});
    defer result.deinit();

    try testing.expect(!result.set.hasBlockingDiagnostic());
    try testing.expectEqual(@as(usize, 1), countEffectFrame(result.set, .write_covered_by_modifies));
    try testing.expectEqual(@as(usize, 1), countEffectFrame(result.set, .read_preserved_by_frame));

    var saw_read_preserved = false;
    for (result.set.obligations) |item| {
        if (item.kind != .effect_frame or item.kind.effect_frame.relation != .read_preserved_by_frame) continue;
        saw_read_preserved = true;
        const frame = item.kind.effect_frame;
        try testing.expectEqual(@as(usize, 1), frame.declared.len);
        try testing.expectEqual(@as(usize, 1), frame.actual.len);
        try testing.expectEqualStrings("buckets", frame.declared[0].root);
        try expectConstantKey(frame.declared[0].keys[0], "1");
        try testing.expectEqualStrings("buckets", frame.actual[0].root);
        try expectConstantKey(frame.actual[0].keys[0], "2");
    }
    try testing.expect(saw_read_preserved);
}

test "formal obligation MLIR adapter emits evidence-backed frame for parameter disequality" {
    const h = createContext();
    defer destroyContext(h);

    const text =
        \\module {
        \\  func.func @effect_surface(%user: !ora.int<256, false>, %other: !ora.int<256, false>) attributes {
        \\    ora.effect = "readwrites",
        \\    ora.param_names = ["user", "other"],
        \\    ora.param_binding_ids = ["file:501:pattern:1", "file:501:pattern:2"],
        \\    ora.modifies_slots = ["balances[param#0]"],
        \\    ora.read_slots = ["balances[param#1]"],
        \\    ora.write_slots = ["balances[param#0]"],
        \\    ora.write_slots_complete = true
        \\  } {
        \\    %neq = ora.cmp "ne", %user, %other : !ora.int<256, false>, !ora.int<256, false> -> i1
        \\    "ora.requires"(%neq) : (i1) -> ()
        \\    func.return
        \\  }
        \\}
    ;

    const module = try parseModule(h.ctx, text);
    defer mlir.oraModuleDestroy(module);

    var result = try obligation_from_mlir.collect(testing.allocator, module, .{});
    defer result.deinit();

    try testing.expect(!result.set.hasBlockingDiagnostic());
    try testing.expectEqual(@as(usize, 1), countAssumption(result.set, .requires));
    try testing.expectEqual(@as(usize, 1), countEffectFrame(result.set, .write_covered_by_modifies));
    try testing.expectEqual(@as(usize, 0), countEffectFrame(result.set, .read_preserved_by_frame));
    try testing.expectEqual(@as(usize, 1), countEffectFrame(result.set, .read_preserved_by_key_evidence));

    var evidence_obligation_id: ?obligation.Id = null;
    var saw_evidence = false;
    for (result.set.obligations) |item| {
        if (item.kind != .effect_frame or item.kind.effect_frame.relation != .read_preserved_by_key_evidence) continue;
        saw_evidence = true;
        evidence_obligation_id = item.id;
        const frame = item.kind.effect_frame;
        try testing.expectEqual(@as(usize, 1), frame.declared.len);
        try testing.expectEqual(@as(usize, 1), frame.actual.len);
        try testing.expectEqual(@as(usize, 1), frame.evidence.len);
        try testing.expectEqual(result.set.assumptions[0].id, frame.evidence[0].assumption_id);
        try testing.expectEqual(@as(u32, 0), frame.evidence[0].key_index);
        try testing.expect(obligation.freeVarIdEql(.{ .file_id = 501, .pattern_id = 1 }, frame.evidence[0].lhs));
        try testing.expect(obligation.freeVarIdEql(.{ .file_id = 501, .pattern_id = 2 }, frame.evidence[0].rhs));
        try expectParameterKey(frame.evidence[0].write.keys[0], .{ .file_id = 501, .pattern_id = 1 });
        try expectParameterKey(frame.evidence[0].read.keys[0], .{ .file_id = 501, .pattern_id = 2 });
    }
    try testing.expect(saw_evidence);

    var saw_query_with_assumption = false;
    for (result.set.queries) |query| {
        if (query.obligation_ids.len != 1 or query.obligation_ids[0] != evidence_obligation_id.?) continue;
        saw_query_with_assumption = true;
        try testing.expectEqual(@as(usize, 1), query.assumption_ids.len);
        try testing.expectEqual(result.set.assumptions[0].id, query.assumption_ids[0]);
        switch (obligation_to_lean.querySemanticSupport(result.set, query)) {
            .supported => {},
            .unsupported => return error.TestUnexpectedResult,
        }
    }
    try testing.expect(saw_query_with_assumption);
    try expectEffectFrameQuerySupported(result.set, .read_preserved_by_key_evidence);

    const dump = try dumpManifestToOwnedString(testing.allocator, result.set);
    defer testing.allocator.free(dump);
    try testing.expect(std.mem.containsAtLeast(u8, dump, 1, "\"schema_version\":12"));
    try testing.expect(std.mem.containsAtLeast(u8, dump, 1, "\"relation\":\"read_preserved_by_key_evidence\""));
    try testing.expect(std.mem.containsAtLeast(u8, dump, 1, "\"evidence\":[{\"kind\":\"free_var_disequality\""));
    try testing.expect(std.mem.containsAtLeast(u8, dump, 1, "\"key_index\":0"));
}

test "formal obligation MLIR adapter accepts symmetric parameter disequality evidence" {
    const h = createContext();
    defer destroyContext(h);

    const text =
        \\module {
        \\  func.func @effect_surface(%user: !ora.int<256, false>, %other: !ora.int<256, false>) attributes {
        \\    ora.effect = "readwrites",
        \\    ora.param_names = ["user", "other"],
        \\    ora.param_binding_ids = ["file:502:pattern:1", "file:502:pattern:2"],
        \\    ora.modifies_slots = ["balances[param#0]"],
        \\    ora.read_slots = ["balances[param#1]"],
        \\    ora.write_slots = ["balances[param#0]"],
        \\    ora.write_slots_complete = true
        \\  } {
        \\    %neq = ora.cmp "ne", %other, %user : !ora.int<256, false>, !ora.int<256, false> -> i1
        \\    "ora.requires"(%neq) : (i1) -> ()
        \\    func.return
        \\  }
        \\}
    ;

    const module = try parseModule(h.ctx, text);
    defer mlir.oraModuleDestroy(module);

    var result = try obligation_from_mlir.collect(testing.allocator, module, .{});
    defer result.deinit();

    try testing.expect(!result.set.hasBlockingDiagnostic());
    try testing.expectEqual(@as(usize, 1), countEffectFrame(result.set, .read_preserved_by_key_evidence));
}

test "formal obligation MLIR adapter requires complete write slots for key evidence frame" {
    const h = createContext();
    defer destroyContext(h);

    const text =
        \\module {
        \\  func.func @effect_surface(%user: !ora.int<256, false>, %other: !ora.int<256, false>) attributes {
        \\    ora.effect = "readwrites",
        \\    ora.param_names = ["user", "other"],
        \\    ora.param_binding_ids = ["file:504:pattern:1", "file:504:pattern:2"],
        \\    ora.modifies_slots = ["balances[param#0]"],
        \\    ora.read_slots = ["balances[param#1]"],
        \\    ora.write_slots = ["balances[param#0]"]
        \\  } {
        \\    %neq = ora.cmp "ne", %user, %other : !ora.int<256, false>, !ora.int<256, false> -> i1
        \\    "ora.requires"(%neq) : (i1) -> ()
        \\    func.return
        \\  }
        \\}
    ;

    const module = try parseModule(h.ctx, text);
    defer mlir.oraModuleDestroy(module);

    var result = try obligation_from_mlir.collect(testing.allocator, module, .{});
    defer result.deinit();

    try testing.expect(!result.set.hasBlockingDiagnostic());
    try testing.expectEqual(@as(usize, 1), countEffectFrame(result.set, .write_covered_by_modifies));
    try testing.expectEqual(@as(usize, 0), countEffectFrame(result.set, .read_preserved_by_frame));
    try testing.expectEqual(@as(usize, 0), countEffectFrame(result.set, .read_preserved_by_key_evidence));
}

test "formal obligation MLIR adapter requires evidence for every non-static read write pair" {
    const h = createContext();
    defer destroyContext(h);

    const text =
        \\module {
        \\  func.func @effect_surface(%user: !ora.int<256, false>, %other: !ora.int<256, false>, %admin: !ora.int<256, false>) attributes {
        \\    ora.effect = "readwrites",
        \\    ora.param_names = ["user", "other", "admin"],
        \\    ora.param_binding_ids = ["file:503:pattern:1", "file:503:pattern:2", "file:503:pattern:3"],
        \\    ora.modifies_slots = ["balances[param#0]", "balances[param#2]"],
        \\    ora.read_slots = ["balances[param#1]"],
        \\    ora.write_slots = ["balances[param#0]", "balances[param#2]"],
        \\    ora.write_slots_complete = true
        \\  } {
        \\    %neq = ora.cmp "ne", %user, %other : !ora.int<256, false>, !ora.int<256, false> -> i1
        \\    "ora.requires"(%neq) : (i1) -> ()
        \\    func.return
        \\  }
        \\}
    ;

    const module = try parseModule(h.ctx, text);
    defer mlir.oraModuleDestroy(module);

    var result = try obligation_from_mlir.collect(testing.allocator, module, .{});
    defer result.deinit();

    try testing.expect(!result.set.hasBlockingDiagnostic());
    try testing.expectEqual(@as(usize, 0), countEffectFrame(result.set, .read_preserved_by_frame));
    try testing.expectEqual(@as(usize, 0), countEffectFrame(result.set, .read_preserved_by_key_evidence));
}

test "formal obligation MLIR adapter rejects same-key evidence-backed frame" {
    const h = createContext();
    defer destroyContext(h);

    const text =
        \\module {
        \\  func.func @effect_surface(%user: !ora.int<256, false>, %other: !ora.int<256, false>) attributes {
        \\    ora.effect = "readwrites",
        \\    ora.param_names = ["user", "other"],
        \\    ora.param_binding_ids = ["file:505:pattern:1", "file:505:pattern:2"],
        \\    ora.modifies_slots = ["balances[param#0]"],
        \\    ora.read_slots = ["balances[param#0]"],
        \\    ora.write_slots = ["balances[param#0]"],
        \\    ora.write_slots_complete = true
        \\  } {
        \\    %neq = ora.cmp "ne", %user, %other : !ora.int<256, false>, !ora.int<256, false> -> i1
        \\    "ora.requires"(%neq) : (i1) -> ()
        \\    func.return
        \\  }
        \\}
    ;

    const module = try parseModule(h.ctx, text);
    defer mlir.oraModuleDestroy(module);

    var result = try obligation_from_mlir.collect(testing.allocator, module, .{});
    defer result.deinit();

    try testing.expect(!result.set.hasBlockingDiagnostic());
    try testing.expectEqual(@as(usize, 0), countEffectFrame(result.set, .read_preserved_by_frame));
    try testing.expectEqual(@as(usize, 0), countEffectFrame(result.set, .read_preserved_by_key_evidence));
}

test "formal obligation MLIR adapter rejects prefix-path evidence-backed frame" {
    const h = createContext();
    defer destroyContext(h);

    const text =
        \\module {
        \\  func.func @effect_surface(%owner: !ora.int<256, false>, %spender: !ora.int<256, false>) attributes {
        \\    ora.effect = "readwrites",
        \\    ora.param_names = ["owner", "spender"],
        \\    ora.param_binding_ids = ["file:506:pattern:1", "file:506:pattern:2"],
        \\    ora.modifies_slots = ["allowances[param#0]"],
        \\    ora.read_slots = ["allowances[param#0][param#1]"],
        \\    ora.write_slots = ["allowances[param#0]"],
        \\    ora.write_slots_complete = true
        \\  } {
        \\    %neq = ora.cmp "ne", %owner, %spender : !ora.int<256, false>, !ora.int<256, false> -> i1
        \\    "ora.requires"(%neq) : (i1) -> ()
        \\    func.return
        \\  }
        \\}
    ;

    const module = try parseModule(h.ctx, text);
    defer mlir.oraModuleDestroy(module);

    var result = try obligation_from_mlir.collect(testing.allocator, module, .{});
    defer result.deinit();

    try testing.expect(!result.set.hasBlockingDiagnostic());
    try testing.expectEqual(@as(usize, 0), countEffectFrame(result.set, .read_preserved_by_frame));
    try testing.expectEqual(@as(usize, 0), countEffectFrame(result.set, .read_preserved_by_key_evidence));
}

test "formal obligation MLIR adapter does not borrow key evidence across owners" {
    const h = createContext();
    defer destroyContext(h);

    const text =
        \\module {
        \\  func.func @provider(%user: !ora.int<256, false>, %other: !ora.int<256, false>) attributes {
        \\    ora.param_names = ["user", "other"],
        \\    ora.param_binding_ids = ["file:507:pattern:1", "file:507:pattern:2"]
        \\  } {
        \\    %neq = ora.cmp "ne", %user, %other : !ora.int<256, false>, !ora.int<256, false> -> i1
        \\    "ora.requires"(%neq) : (i1) -> ()
        \\    func.return
        \\  }
        \\
        \\  func.func @effect_surface(%user: !ora.int<256, false>, %other: !ora.int<256, false>) attributes {
        \\    ora.effect = "readwrites",
        \\    ora.param_names = ["user", "other"],
        \\    ora.param_binding_ids = ["file:507:pattern:1", "file:507:pattern:2"],
        \\    ora.modifies_slots = ["balances[param#0]"],
        \\    ora.read_slots = ["balances[param#1]"],
        \\    ora.write_slots = ["balances[param#0]"],
        \\    ora.write_slots_complete = true
        \\  } {
        \\    func.return
        \\  }
        \\}
    ;

    const module = try parseModule(h.ctx, text);
    defer mlir.oraModuleDestroy(module);

    var result = try obligation_from_mlir.collect(testing.allocator, module, .{});
    defer result.deinit();

    try testing.expect(!result.set.hasBlockingDiagnostic());
    try testing.expectEqual(@as(usize, 1), countAssumption(result.set, .requires));
    try testing.expectEqual(@as(usize, 0), countEffectFrame(result.set, .read_preserved_by_frame));
    try testing.expectEqual(@as(usize, 0), countEffectFrame(result.set, .read_preserved_by_key_evidence));
    try expectQueriesOwnerScoped(result.set);
}

test "formal obligation MLIR adapter emits evidence for nested differing parameter key" {
    const h = createContext();
    defer destroyContext(h);

    const text =
        \\module {
        \\  func.func @effect_surface(%owner: !ora.int<256, false>, %other: !ora.int<256, false>, %spender: !ora.int<256, false>) attributes {
        \\    ora.effect = "readwrites",
        \\    ora.param_names = ["owner", "other", "spender"],
        \\    ora.param_binding_ids = ["file:508:pattern:1", "file:508:pattern:2", "file:508:pattern:3"],
        \\    ora.modifies_slots = ["allowances[param#0][param#2]"],
        \\    ora.read_slots = ["allowances[param#1][param#2]"],
        \\    ora.write_slots = ["allowances[param#0][param#2]"],
        \\    ora.write_slots_complete = true
        \\  } {
        \\    %neq = ora.cmp "ne", %owner, %other : !ora.int<256, false>, !ora.int<256, false> -> i1
        \\    "ora.requires"(%neq) : (i1) -> ()
        \\    func.return
        \\  }
        \\}
    ;

    const module = try parseModule(h.ctx, text);
    defer mlir.oraModuleDestroy(module);

    var result = try obligation_from_mlir.collect(testing.allocator, module, .{});
    defer result.deinit();

    try testing.expect(!result.set.hasBlockingDiagnostic());
    try testing.expectEqual(@as(usize, 1), countEffectFrame(result.set, .read_preserved_by_key_evidence));

    const query = try findEffectFrameQuery(result.set, .read_preserved_by_key_evidence);
    try testing.expectEqual(@as(usize, 1), query.assumption_ids.len);
    for (result.set.obligations) |item| {
        if (item.kind != .effect_frame or item.kind.effect_frame.relation != .read_preserved_by_key_evidence) continue;
        try testing.expectEqual(@as(usize, 1), item.kind.effect_frame.evidence.len);
        try testing.expectEqual(@as(u32, 0), item.kind.effect_frame.evidence[0].key_index);
    }
}

test "formal obligation Lean support rejects unsupported key evidence formulas" {
    const owner: obligation.Owner = .{ .function = .{ .name = "effect_surface" } };
    const lhs_id: obligation.FreeVarId = .{ .file_id = 601, .pattern_id = 1 };
    const rhs_id: obligation.FreeVarId = .{ .file_id = 601, .pattern_id = 2 };
    const read_keys = [_]obligation.PlaceKey{.{ .parameter = rhs_id }};
    const write_keys = [_]obligation.PlaceKey{.{ .parameter = lhs_id }};
    const read: obligation.PlaceRef = .{ .root = "balances", .region = .storage, .keys = &read_keys };
    const write: obligation.PlaceRef = .{ .root = "balances", .region = .storage, .keys = &write_keys };
    const assumption_ids = [_]obligation.Id{10};
    const u256_ty: obligation.TypeRef = .{ .spelling = "u256" };
    const user: obligation.Term = .{ .variable = .{ .free = .{ .id = lhs_id, .name = "user", .ty = u256_ty } } };
    const other: obligation.Term = .{ .variable = .{ .free = .{ .id = rhs_id, .name = "other", .ty = u256_ty } } };

    {
        const terms = [_]obligation.Term{
            user,
            other,
            .{ .binary = .{ .op = .eq, .lhs = 0, .rhs = 1 } },
            .{ .unary = .{ .op = .not, .operand = 2 } },
        };
        try expectSyntheticKeyEvidenceUnsupported(
            &terms,
            3,
            .requires,
            &assumption_ids,
            owner,
            owner,
            read,
            write,
            lhs_id,
            rhs_id,
            .unsupported_key_disjoint_evidence_formula,
        );
    }

    {
        const terms = [_]obligation.Term{
            user,
            .{ .int_lit = .{ .value = "42", .ty = u256_ty } },
            .{ .binary = .{ .op = .ne, .lhs = 0, .rhs = 1 } },
        };
        try expectSyntheticKeyEvidenceUnsupported(
            &terms,
            2,
            .requires,
            &assumption_ids,
            owner,
            owner,
            read,
            write,
            lhs_id,
            rhs_id,
            .unsupported_key_disjoint_evidence_formula,
        );
    }

    {
        const terms = [_]obligation.Term{
            user,
            other,
            .{ .binary = .{ .op = .ne, .lhs = 0, .rhs = 1 } },
        };
        try expectSyntheticKeyEvidenceUnsupported(
            &terms,
            2,
            .assume,
            &assumption_ids,
            owner,
            owner,
            read,
            write,
            lhs_id,
            rhs_id,
            .unsupported_key_disjoint_evidence_formula,
        );
    }
}

test "formal obligation Lean support rejects non-u256 key evidence variables" {
    const owner: obligation.Owner = .{ .function = .{ .name = "effect_surface" } };
    const lhs_id: obligation.FreeVarId = .{ .file_id = 602, .pattern_id = 1 };
    const rhs_id: obligation.FreeVarId = .{ .file_id = 602, .pattern_id = 2 };
    const read_keys = [_]obligation.PlaceKey{.{ .parameter = rhs_id }};
    const write_keys = [_]obligation.PlaceKey{.{ .parameter = lhs_id }};
    const read: obligation.PlaceRef = .{ .root = "balances", .region = .storage, .keys = &read_keys };
    const write: obligation.PlaceRef = .{ .root = "balances", .region = .storage, .keys = &write_keys };
    const address_ty: obligation.TypeRef = .{ .spelling = "address" };
    const terms = [_]obligation.Term{
        .{ .variable = .{ .free = .{ .id = lhs_id, .name = "user", .ty = address_ty } } },
        .{ .variable = .{ .free = .{ .id = rhs_id, .name = "other", .ty = address_ty } } },
        .{ .binary = .{ .op = .ne, .lhs = 0, .rhs = 1 } },
    };

    try expectSyntheticKeyEvidenceUnsupported(
        &terms,
        2,
        .requires,
        &.{},
        owner,
        owner,
        read,
        write,
        lhs_id,
        rhs_id,
        .key_disjoint_evidence_type_unsupported,
    );
}

test "formal obligation MLIR adapter does not project keyed map read with unknown key" {
    const h = createContext();
    defer destroyContext(h);

    const text =
        \\module {
        \\  ora.contract @StorageProofs {
        \\    ora.global "balances" : !ora.map<!ora.int<256, false>, !ora.int<256, false>>
        \\    func.func @unknown_key(%owner: !ora.int<256, false>) attributes {
        \\      ora.param_names = ["owner"],
        \\      ora.param_binding_ids = ["file:44:pattern:1"],
        \\      ora.write_slots = [],
        \\      ora.write_slots_complete = true
        \\    } {
        \\      %balances = ora.sload "balances" : !ora.map<!ora.int<256, false>, !ora.int<256, false>>
        \\      %key = ora.add_wrapping %owner, %owner : !ora.int<256, false>, !ora.int<256, false> -> !ora.int<256, false>
        \\      %balance = "ora.map_get"(%balances, %key) : (!ora.map<!ora.int<256, false>, !ora.int<256, false>>, !ora.int<256, false>) -> !ora.int<256, false>
        \\      %cmp = ora.cmp "eq", %balance, %balance : !ora.int<256, false>, !ora.int<256, false> -> i1
        \\      "ora.ensures"(%cmp) : (i1) -> ()
        \\      func.return
        \\    }
        \\  }
        \\}
    ;

    const module = try parseModule(h.ctx, text);
    defer mlir.oraModuleDestroy(module);

    var result = try obligation_from_mlir.collect(testing.allocator, module, .{});
    defer result.deinit();

    try testing.expect(!result.set.hasBlockingDiagnostic());
    try expectEnsuresQueryUnsupported(result.set, .unsupported_origin_value);

    const report = try dumpManifestToOwnedString(testing.allocator, result.set);
    defer testing.allocator.free(report);
    try testing.expect(!std.mem.containsAtLeast(u8, report, 1, "\"tag\":\"place_read\""));
}

test "formal obligation MLIR adapter projects nested read-only map paths as distinct roots" {
    const h = createContext();
    defer destroyContext(h);

    const text =
        \\module {
        \\  ora.contract @StorageProofs {
        \\    ora.global "balances" : !ora.map<!ora.int<256, false>, !ora.int<256, false>>
        \\    ora.global "allowances" : !ora.map<!ora.int<256, false>, !ora.map<!ora.int<256, false>, !ora.int<256, false>>>
        \\    func.func @check(%owner: !ora.int<256, false>, %spender: !ora.int<256, false>) attributes {
        \\      ora.param_names = ["owner", "spender"],
        \\      ora.param_binding_ids = ["file:45:pattern:1", "file:45:pattern:2"],
        \\      ora.write_slots = [],
        \\      ora.write_slots_complete = true
        \\    } {
        \\      %balances = ora.sload "balances" : !ora.map<!ora.int<256, false>, !ora.int<256, false>>
        \\      %balance = "ora.map_get"(%balances, %owner) : (!ora.map<!ora.int<256, false>, !ora.int<256, false>>, !ora.int<256, false>) -> !ora.int<256, false>
        \\      %allowances = ora.sload "allowances" : !ora.map<!ora.int<256, false>, !ora.map<!ora.int<256, false>, !ora.int<256, false>>>
        \\      %owner_allowances = "ora.map_get"(%allowances, %owner) : (!ora.map<!ora.int<256, false>, !ora.map<!ora.int<256, false>, !ora.int<256, false>>>, !ora.int<256, false>) -> !ora.map<!ora.int<256, false>, !ora.int<256, false>>
        \\      %allowance = "ora.map_get"(%owner_allowances, %spender) : (!ora.map<!ora.int<256, false>, !ora.int<256, false>>, !ora.int<256, false>) -> !ora.int<256, false>
        \\      %balance_cmp = ora.cmp "eq", %balance, %balance : !ora.int<256, false>, !ora.int<256, false> -> i1
        \\      %allowance_cmp = ora.cmp "eq", %allowance, %allowance : !ora.int<256, false>, !ora.int<256, false> -> i1
        \\      "ora.ensures"(%balance_cmp) : (i1) -> ()
        \\      "ora.ensures"(%allowance_cmp) : (i1) -> ()
        \\      func.return
        \\    }
        \\  }
        \\}
    ;

    const module = try parseModule(h.ctx, text);
    defer mlir.oraModuleDestroy(module);

    var result = try obligation_from_mlir.collect(testing.allocator, module, .{});
    defer result.deinit();

    try testing.expect(!result.set.hasBlockingDiagnostic());
    try testing.expectEqual(@as(usize, 2), countLogical(result.set, .ensures));

    var saw_balances = false;
    var saw_allowances = false;
    for (result.set.terms) |term| {
        if (term != .place_read) continue;
        if (std.mem.eql(u8, term.place_read.root, "balances")) {
            saw_balances = true;
            try testing.expectEqual(@as(usize, 1), term.place_read.keys.len);
            try expectParameterKey(term.place_read.keys[0], .{ .file_id = 45, .pattern_id = 1 });
        } else if (std.mem.eql(u8, term.place_read.root, "allowances")) {
            saw_allowances = true;
            try testing.expectEqual(@as(usize, 2), term.place_read.keys.len);
            try expectParameterKey(term.place_read.keys[0], .{ .file_id = 45, .pattern_id = 1 });
            try expectParameterKey(term.place_read.keys[1], .{ .file_id = 45, .pattern_id = 2 });
        }
    }
    try testing.expect(saw_balances);
    try testing.expect(saw_allowances);

    const lean = try emitLeanToOwnedString(testing.allocator, result.set);
    defer testing.allocator.free(lean);
    try testing.expect(std.mem.containsAtLeast(u8, lean, 1, "root := \"balances\""));
    try testing.expect(std.mem.containsAtLeast(u8, lean, 1, "root := \"allowances\""));
    try testing.expect(std.mem.containsAtLeast(u8, lean, 1, "keys := [.parameter { file_id := 45, pattern_id := 1 }, .parameter { file_id := 45, pattern_id := 2 }]"));
}

test "formal obligation MLIR adapter tracks nested forall binders with De Bruijn indexes" {
    const h = createContext();
    defer destroyContext(h);

    const text =
        \\module {
        \\  func.func @nested_forall() {
        \\    %i = arith.constant {ora.bound_variable = "i"} 0 : i256
        \\    %j = arith.constant {ora.bound_variable = "j"} 0 : i256
        \\    %cmp = arith.cmpi ule, %i, %j : i256
        \\    %inner = "ora.quantified"(%cmp) <{quantifier = "forall", variable = "j", variable_type = "u256"}> : (i1) -> i1
        \\    %outer = "ora.quantified"(%inner) <{quantifier = "forall", variable = "i", variable_type = "u256"}> : (i1) -> i1
        \\    "ora.assert"(%outer) <{message = "nested"}> : (i1) -> ()
        \\    func.return
        \\  }
        \\}
    ;

    const module = try parseModule(h.ctx, text);
    defer mlir.oraModuleDestroy(module);

    var result = try obligation_from_mlir.collect(testing.allocator, module, .{});
    defer result.deinit();

    try testing.expect(!result.set.hasBlockingDiagnostic());
    try testing.expectEqual(@as(usize, 1), countLogical(result.set, .assert));

    const outer = try expectQuantifiedTerm(result.set, try expectLogicalTerm(result.set, .assert), .forall, "i");
    const inner = try expectQuantifiedTerm(result.set, outer.body, .forall, "j");
    const body = try expectBinaryTerm(result.set, inner.body, .le);
    try expectBoundVarTerm(result.set, body.lhs, 1, "i");
    try expectBoundVarTerm(result.set, body.rhs, 0, "j");
}

test "formal obligation MLIR adapter resolves shadowed forall binders to nearest binder" {
    const h = createContext();
    defer destroyContext(h);

    const text =
        \\module {
        \\  func.func @shadowed_forall() {
        \\    %i = arith.constant {ora.bound_variable = "i"} 0 : i256
        \\    %cmp = arith.cmpi ule, %i, %i : i256
        \\    %inner = "ora.quantified"(%cmp) <{quantifier = "forall", variable = "i", variable_type = "u256"}> : (i1) -> i1
        \\    %outer = "ora.quantified"(%inner) <{quantifier = "forall", variable = "i", variable_type = "u256"}> : (i1) -> i1
        \\    "ora.assert"(%outer) <{message = "shadowed"}> : (i1) -> ()
        \\    func.return
        \\  }
        \\}
    ;

    const module = try parseModule(h.ctx, text);
    defer mlir.oraModuleDestroy(module);

    var result = try obligation_from_mlir.collect(testing.allocator, module, .{});
    defer result.deinit();

    try testing.expect(!result.set.hasBlockingDiagnostic());
    try testing.expectEqual(@as(usize, 1), countLogical(result.set, .assert));

    const outer = try expectQuantifiedTerm(result.set, try expectLogicalTerm(result.set, .assert), .forall, "i");
    const inner = try expectQuantifiedTerm(result.set, outer.body, .forall, "i");
    const body = try expectBinaryTerm(result.set, inner.body, .le);
    try expectBoundVarTerm(result.set, body.lhs, 0, "i");
    try expectBoundVarTerm(result.set, body.rhs, 0, "i");
}

test "formal obligation MLIR adapter records effect frame summaries" {
    const h = createContext();
    defer destroyContext(h);

    const text =
        \\module {
        \\  func.func @effect_surface(%account: !ora.int<256, false>) attributes {
        \\    ora.effect = "readwrites",
        \\    ora.param_names = ["account"],
        \\    ora.param_binding_ids = ["file:88:pattern:1"],
        \\    ora.modifies_slots = ["balances[param#0]", "config.owner"],
        \\    ora.read_slots = ["balances[param#0]", "transient:scratch"],
        \\    ora.write_slots = ["balances[param#0]"],
        \\    ora.write_slots_complete = true
        \\  } {
        \\    func.return
        \\  }
        \\}
    ;

    const module = try parseModule(h.ctx, text);
    defer mlir.oraModuleDestroy(module);

    var result = try obligation_from_mlir.collect(testing.allocator, module, .{});
    defer result.deinit();

    try testing.expect(!result.set.hasBlockingDiagnostic());
    try testing.expectEqual(@as(usize, 2), result.set.obligations.len);
    try testing.expectEqual(@as(usize, 3), result.set.queries.len);
    try testing.expectEqual(@as(usize, 1), countQuery(result.set, .base));
    try testing.expectEqual(@as(usize, 2), countQuery(result.set, .obligation));
    try testing.expectEqual(@as(usize, 1), countEffectFrame(result.set, .write_covered_by_modifies));
    try testing.expectEqual(@as(usize, 1), countEffectFrame(result.set, .read_preserved_by_frame));

    const writes = result.set.obligations[0].kind.effect_frame;
    try testing.expectEqual(obligation.EffectFrameRelation.write_covered_by_modifies, writes.relation);
    try testing.expectEqual(@as(usize, 2), writes.declared.len);
    try testing.expectEqual(@as(usize, 1), writes.actual.len);
    try testing.expectEqualStrings("balances", writes.declared[0].root);
    try testing.expectEqual(obligation.RegionRef.storage, writes.declared[0].region);
    try testing.expectEqual(@as(usize, 1), writes.declared[0].keys.len);
    try expectParameterKey(writes.declared[0].keys[0], .{ .file_id = 88, .pattern_id = 1 });
    try testing.expectEqualStrings("config", writes.declared[1].root);
    try testing.expectEqual(@as(usize, 1), writes.declared[1].fields.len);
    try testing.expectEqualStrings("owner", writes.declared[1].fields[0]);

    const reads = result.set.obligations[1].kind.effect_frame;
    try testing.expectEqual(obligation.EffectFrameRelation.read_preserved_by_frame, reads.relation);
    try testing.expectEqual(@as(usize, 1), reads.declared.len);
    try testing.expectEqual(@as(usize, 1), reads.actual.len);
    try testing.expectEqualStrings("scratch", reads.actual[0].root);
    try testing.expectEqual(obligation.RegionRef.transient, reads.actual[0].region);
}

test "formal obligation MLIR adapter distinguishes omitted and explicitly empty modifies" {
    const h = createContext();
    defer destroyContext(h);

    const text =
        \\module {
        \\  func.func @omitted() attributes {
        \\    ora.write_slots = ["counter"],
        \\    ora.write_slots_complete = true
        \\  } { func.return }
        \\  func.func @explicit_empty() attributes {
        \\    ora.modifies_slots = [],
        \\    ora.write_slots = ["counter"],
        \\    ora.write_slots_complete = true
        \\  } { func.return }
        \\}
    ;

    const module = try parseModule(h.ctx, text);
    defer mlir.oraModuleDestroy(module);

    var result = try obligation_from_mlir.collect(testing.allocator, module, .{});
    defer result.deinit();

    try testing.expectEqual(@as(usize, 1), countEffectFrame(result.set, .write_covered_by_modifies));
    for (result.set.obligations) |item| {
        if (item.kind != .effect_frame or
            item.kind.effect_frame.relation != .write_covered_by_modifies) continue;
        try testing.expectEqualStrings("explicit_empty", item.owner.function.name);
        try testing.expectEqual(@as(usize, 0), item.kind.effect_frame.declared.len);
        try testing.expectEqual(@as(usize, 1), item.kind.effect_frame.actual.len);
    }
}

test "formal obligation MLIR adapter records lock and external frame metadata" {
    const h = createContext();
    defer destroyContext(h);

    const text =
        \\module {
        \\  func.func private @callee()
        \\  func.func @lock_and_call(%resource: i256) {
        \\    ora.lock %resource {key = "balances[msg.sender]"} : i256
        \\    func.call @callee() {ora.trusted_extern_frame = "caller_storage"} : () -> ()
        \\    func.return
        \\  }
        \\}
    ;

    const module = try parseModule(h.ctx, text);
    defer mlir.oraModuleDestroy(module);

    var result = try obligation_from_mlir.collect(testing.allocator, module, .{});
    defer result.deinit();

    try testing.expect(!result.set.hasBlockingDiagnostic());
    try testing.expectEqual(@as(usize, 2), result.set.obligations.len);
    try testing.expectEqual(@as(usize, 3), result.set.queries.len);
    try testing.expectEqual(@as(usize, 1), countQuery(result.set, .base));
    try testing.expectEqual(@as(usize, 1), countEffectFrame(result.set, .lock_covers_write));
    try testing.expectEqual(@as(usize, 1), countEffectFrame(result.set, .external_call_frame));

    const lock = result.set.obligations[0].kind.effect_frame;
    try testing.expectEqual(obligation.EffectFrameRelation.lock_covers_write, lock.relation);
    try testing.expectEqual(@as(usize, 1), lock.declared.len);
    try testing.expectEqualStrings("balances", lock.declared[0].root);
    try testing.expectEqual(@as(usize, 1), lock.declared[0].keys.len);
    try testing.expect(lock.declared[0].keys[0] == .msg_sender);

    const external = result.set.obligations[1].kind.effect_frame;
    try testing.expectEqual(obligation.EffectFrameRelation.external_call_frame, external.relation);
    try testing.expectEqual(@as(usize, 1), external.declared.len);
    try testing.expectEqual(obligation.RegionRef.none, external.declared[0].region);
    try testing.expectEqualStrings("caller_storage", external.declared[0].root);
}

test "formal obligation MLIR adapter covers Z3 assertion tags and implicit division safety" {
    const h = createContext();
    defer destroyContext(h);

    const text =
        \\module {
        \\  func.func @z3_surface(%flag: i1, %x: i256, %y: i256) {
        \\    "cf.assert"(%flag) <{msg = "requires"}> : (i1) -> ()
        \\    "ora.assert"(%flag) : (i1) -> ()
        \\    "ora.assume"(%flag) : (i1) -> ()
        \\    %q = arith.divui %x, %y : i256
        \\    %s = arith.shli %x, %y : i256
        \\    func.return
        \\  }
        \\}
    ;

    const module = try parseModule(h.ctx, text);
    defer mlir.oraModuleDestroy(module);

    const cf_assert = findFirstOpByName(module, "cf.assert") orelse return error.TestUnexpectedResult;
    mlir.oraOperationSetAttributeByName(
        cf_assert,
        strRef("ora.requires"),
        mlir.oraBoolAttrCreate(h.ctx, true),
    );

    const ora_assert = findFirstOpByName(module, "ora.assert") orelse return error.TestUnexpectedResult;
    mlir.oraOperationSetAttributeByName(
        ora_assert,
        strRef("ora.verification_type"),
        mlir.oraStringAttrCreate(h.ctx, strRef("guard")),
    );
    mlir.oraOperationSetAttributeByName(
        ora_assert,
        strRef("ora.guard_id"),
        mlir.oraStringAttrCreate(h.ctx, strRef("guard:z3_surface:flag")),
    );

    const assume_op = findFirstOpByName(module, "ora.assume") orelse return error.TestUnexpectedResult;
    mlir.oraOperationSetAttributeByName(
        assume_op,
        strRef("ora.assume_origin"),
        mlir.oraStringAttrCreate(h.ctx, strRef("path")),
    );

    var result = try obligation_from_mlir.collect(testing.allocator, module, .{});
    defer result.deinit();

    try testing.expect(!result.set.hasBlockingDiagnostic());
    try testing.expectEqual(@as(usize, 2), result.set.obligations.len);
    try testing.expectEqual(@as(usize, 1), result.set.assumptions.len);
    try testing.expectEqual(@as(usize, 4), result.set.queries.len);
    try testing.expectEqual(@as(usize, 0), countAssumption(result.set, .requires));
    try testing.expectEqual(@as(usize, 1), countAssumption(result.set, .path_assume));
    try testing.expectEqual(@as(usize, 1), countRuntimeGuards(result.set));
    try testing.expectEqual(@as(usize, 1), countLogical(result.set, .arithmetic_safety));
    try testing.expectEqual(@as(usize, 1), countQuery(result.set, .base));
    try testing.expectEqual(@as(usize, 1), countQuery(result.set, .obligation));
    try testing.expectEqual(@as(usize, 1), countQuery(result.set, .guard_satisfy));
    try testing.expectEqual(@as(usize, 1), countQuery(result.set, .guard_violate));

    for (result.set.obligations) |item| {
        if (item.kind == .logical and item.kind.logical.role == .arithmetic_safety) {
            try testing.expectEqual(obligation.ValueRefKind.derived, item.kind.logical.formula.origin_value.kind);
        }
    }
}

test "formal obligation MLIR adapter expands resource op properties" {
    const h = createContext();
    defer destroyContext(h);

    const text =
        \\module {
        \\  func.func @transfer_resource(%balances: !ora.map<!ora.address, !ora.int<256, false>>, %from: !ora.address, %to: !ora.address, %amount: !ora.int<256, false>) attributes {
        \\    ora.param_names = ["balances", "from", "to", "amount"],
        \\    ora.param_binding_ids = ["file:201:pattern:0", "file:201:pattern:1", "file:201:pattern:2", "file:201:pattern:3"]
        \\  } {
        \\    "ora.move"(%balances, %from, %balances, %to, %amount) <{operandSegmentSizes = array<i32: 2, 2, 1>, domain = "TokenUnit", carrier_type = !ora.int<256, false>, carrier_signed = false}> : (!ora.map<!ora.address, !ora.int<256, false>>, !ora.address, !ora.map<!ora.address, !ora.int<256, false>>, !ora.address, !ora.int<256, false>) -> ()
        \\    func.return
        \\  }
        \\}
    ;

    const module = try parseModule(h.ctx, text);
    defer mlir.oraModuleDestroy(module);

    var result = try obligation_from_mlir.collect(testing.allocator, module, .{});
    defer result.deinit();

    try testing.expect(!result.set.hasBlockingDiagnostic());
    try testing.expectEqual(@as(usize, 5), result.set.obligations.len);
    try testing.expectEqual(@as(usize, 6), result.set.queries.len);
    try testing.expectEqual(@as(usize, 1), countQuery(result.set, .base));
    try testing.expectEqual(@as(usize, 5), countQuery(result.set, .obligation));
    try testing.expectEqual(@as(usize, 1), countResource(result.set, .move, .amount_non_negative));
    try testing.expectEqual(@as(usize, 1), countResource(result.set, .move, .source_sufficient));
    try testing.expectEqual(@as(usize, 1), countResource(result.set, .move, .destination_no_overflow));
    try testing.expectEqual(@as(usize, 1), countResource(result.set, .move, .same_place_identity));
    try testing.expectEqual(@as(usize, 1), countResource(result.set, .move, .conservation));
    inline for (.{
        obligation.ResourceProperty.amount_non_negative,
        obligation.ResourceProperty.source_sufficient,
        obligation.ResourceProperty.destination_no_overflow,
        obligation.ResourceProperty.same_place_identity,
        obligation.ResourceProperty.conservation,
    }) |property| {
        switch (obligation_to_lean.querySemanticSupport(result.set, try findResourceQuery(result.set, .move, property))) {
            .supported => {},
            .unsupported => return error.TestUnexpectedResult,
        }
    }

    for (result.set.obligations) |item| {
        try testing.expect(item.kind == .resource);
        try testing.expectEqualStrings("TokenUnit", item.kind.resource.domain);
        try testing.expect(item.kind.resource.amount.? == .term);
        const source = item.kind.resource.source orelse return error.TestUnexpectedResult;
        const destination = item.kind.resource.destination orelse return error.TestUnexpectedResult;
        try expectPlaceRoot(source, "arg#0", .storage);
        try expectPlaceRoot(destination, "arg#0", .storage);
        try testing.expectEqual(@as(usize, 1), source.keys.len);
        try testing.expectEqual(@as(usize, 1), destination.keys.len);
        try expectParameterKey(source.keys[0], .{ .file_id = 201, .pattern_id = 1 });
        try expectParameterKey(destination.keys[0], .{ .file_id = 201, .pattern_id = 2 });
    }
}

test "formal obligation MLIR adapter records direct resource create and destroy places" {
    const h = createContext();
    defer destroyContext(h);

    const text =
        \\module {
        \\  ora.contract @R {
        \\    ora.global "reserve" : !ora.int<256, false>
        \\    func.func @direct(%amount: !ora.int<256, false>) {
        \\      %reserve = ora.sload "reserve" : !ora.int<256, false>
        \\      "ora.create"(%reserve, %amount) <{domain = "TokenUnit", carrier_type = !ora.int<256, false>, carrier_signed = false}> : (!ora.int<256, false>, !ora.int<256, false>) -> ()
        \\      %scratch = ora.tload "scratch" : !ora.int<256, false>
        \\      "ora.destroy"(%scratch, %amount) <{domain = "TokenUnit", carrier_type = !ora.int<256, false>, carrier_signed = false}> : (!ora.int<256, false>, !ora.int<256, false>) -> ()
        \\      func.return
        \\    }
        \\  }
        \\}
    ;

    const module = try parseModule(h.ctx, text);
    defer mlir.oraModuleDestroy(module);

    var result = try obligation_from_mlir.collect(testing.allocator, module, .{});
    defer result.deinit();

    try testing.expect(!result.set.hasBlockingDiagnostic());
    try testing.expectEqual(@as(usize, 4), result.set.obligations.len);
    try testing.expectEqual(@as(usize, 1), countResource(result.set, .create, .destination_no_overflow));
    try testing.expectEqual(@as(usize, 1), countResource(result.set, .destroy, .source_sufficient));
    switch (obligation_to_lean.querySemanticSupport(result.set, try findResourceQuery(result.set, .create, .amount_non_negative))) {
        .supported => {},
        .unsupported => return error.TestUnexpectedResult,
    }
    switch (obligation_to_lean.querySemanticSupport(result.set, try findResourceQuery(result.set, .create, .destination_no_overflow))) {
        .supported => {},
        .unsupported => return error.TestUnexpectedResult,
    }
    switch (obligation_to_lean.querySemanticSupport(result.set, try findResourceQuery(result.set, .destroy, .amount_non_negative))) {
        .supported => {},
        .unsupported => return error.TestUnexpectedResult,
    }
    switch (obligation_to_lean.querySemanticSupport(result.set, try findResourceQuery(result.set, .destroy, .source_sufficient))) {
        .supported => {},
        .unsupported => return error.TestUnexpectedResult,
    }

    var saw_create = false;
    var saw_destroy = false;
    for (result.set.obligations) |item| {
        if (item.kind != .resource) continue;
        switch (item.kind.resource.op) {
            .create => if (!saw_create) {
                saw_create = true;
                const destination = item.kind.resource.destination orelse return error.TestUnexpectedResult;
                try expectPlaceRoot(destination, "reserve", .storage);
                try testing.expectEqual(@as(usize, 0), destination.keys.len);
            },
            .destroy => if (!saw_destroy) {
                saw_destroy = true;
                const source = item.kind.resource.source orelse return error.TestUnexpectedResult;
                try expectPlaceRoot(source, "scratch", .transient);
                try testing.expectEqual(@as(usize, 0), source.keys.len);
            },
            .move => {},
        }
    }
    try testing.expect(saw_create);
    try testing.expect(saw_destroy);
}

test "formal obligation MLIR adapter records quantifier metadata" {
    const h = createContext();
    defer destroyContext(h);

    const text =
        \\module {
        \\  func.func @bounded(%flag: i1) {
        \\    %q = "ora.quantified"(%flag, %flag) <{quantifier = "forall", variable = "i", variable_type = "u256"}> : (i1, i1) -> i1
        \\    "ora.assert"(%q) <{message = "bounded"}> : (i1) -> ()
        \\    func.return
        \\  }
        \\}
    ;

    const module = try parseModule(h.ctx, text);
    defer mlir.oraModuleDestroy(module);

    var result = try obligation_from_mlir.collect(testing.allocator, module, .{});
    defer result.deinit();

    try testing.expect(!result.set.hasBlockingDiagnostic());
    try testing.expectEqual(@as(usize, 2), result.set.obligations.len);
    try testing.expectEqual(@as(usize, 1), countQuantifier(result.set));

    var saw_quantifier = false;
    for (result.set.obligations) |item| {
        if (item.kind != .quantifier) continue;
        saw_quantifier = true;
        const quantifier = item.kind.quantifier;
        try testing.expectEqual(obligation.ArtifactPolicy.diagnostic_only, item.artifact_policy);
        try testing.expectEqual(obligation.Quantifier.forall, quantifier.quantifier);
        try testing.expectEqualStrings("i", quantifier.variable);
        try testing.expectEqualStrings("u256", quantifier.binder_type.spelling);
        try testing.expectEqual(obligation.QuantifierBinderSort.bit_vector, quantifier.binder_sort);
        try testing.expectEqual(obligation.VerificationQueryFragment.aufbv_quantifiers, quantifier.fragment);
        try testing.expectEqual(obligation.QuantifierPatternStatus.absent, quantifier.pattern_status);
        try testing.expectEqual(@as(?obligation.QuantifierDegradation, null), quantifier.degradation);
    }
    try testing.expect(saw_quantifier);
}

test "formal report coverage summarizes representative MLIR obligation classes" {
    const h = createContext();
    defer destroyContext(h);

    const text =
        \\module {
        \\  func.func @coverage(%flag: i1, %balances: !ora.map<!ora.address, !ora.int<256, false>>, %from: !ora.address, %to: !ora.address, %amount: !ora.int<256, false>, %x: i256, %y: i256) attributes {
        \\    ora.param_names = ["flag", "balances", "from", "to", "amount", "x", "y"],
        \\    ora.param_binding_ids = ["file:202:pattern:0", "file:202:pattern:1", "file:202:pattern:2", "file:202:pattern:3", "file:202:pattern:4", "file:202:pattern:5", "file:202:pattern:6"],
        \\    ora.modifies_slots = ["balances[param#0]"],
        \\    ora.read_slots = ["balances[param#0]"],
        \\    ora.write_slots = ["balances[param#0]"]
        \\  } {
        \\    "ora.requires"(%flag) : (i1) -> ()
        \\    "ora.ensures"(%flag) : (i1) -> ()
        \\    "ora.assert"(%flag) <{message = "checked addition overflow"}> : (i1) -> ()
        \\    "ora.refinement_guard"(%flag) <{message = "guard"}> : (i1) -> ()
        \\    "ora.move"(%balances, %from, %balances, %to, %amount) <{operandSegmentSizes = array<i32: 2, 2, 1>, domain = "TokenUnit", carrier_type = !ora.int<256, false>, carrier_signed = false}> : (!ora.map<!ora.address, !ora.int<256, false>>, !ora.address, !ora.map<!ora.address, !ora.int<256, false>>, !ora.address, !ora.int<256, false>) -> ()
        \\    %q = "ora.quantified"(%flag, %flag) <{quantifier = "forall", variable = "i", variable_type = "u256"}> : (i1, i1) -> i1
        \\    "ora.assert"(%q) <{message = "bounded"}> : (i1) -> ()
        \\    %d = arith.divui %x, %y : i256
        \\    func.return
        \\  }
        \\}
    ;

    const module = try parseModule(h.ctx, text);
    defer mlir.oraModuleDestroy(module);

    const guard_op = findFirstOpByName(module, "ora.refinement_guard") orelse return error.TestUnexpectedResult;
    mlir.oraOperationSetAttributeByName(
        guard_op,
        strRef("ora.guard_id"),
        mlir.oraStringAttrCreate(h.ctx, strRef("guard:coverage:flag")),
    );

    var result = try obligation_from_mlir.collect(testing.allocator, module, .{});
    defer result.deinit();

    try testing.expect(!result.set.hasBlockingDiagnostic());
    try testing.expectEqual(@as(usize, 12), result.set.obligations.len);
    try testing.expectEqual(@as(usize, 1), result.set.assumptions.len);
    try testing.expectEqual(@as(usize, 13), result.set.queries.len);
    try testing.expectEqual(@as(usize, 1), countAssumption(result.set, .requires));
    try testing.expectEqual(@as(usize, 1), countLogical(result.set, .ensures));
    try testing.expectEqual(@as(usize, 1), countLogical(result.set, .assert));
    try testing.expectEqual(@as(usize, 2), countLogical(result.set, .arithmetic_safety));
    try testing.expectEqual(@as(usize, 1), countRuntimeGuards(result.set));
    try testing.expectEqual(@as(usize, 1), countEffectFrame(result.set, .write_covered_by_modifies));
    try testing.expectEqual(@as(usize, 0), countEffectFrame(result.set, .read_preserved_by_frame));
    try testing.expectEqual(@as(usize, 1), countResource(result.set, .move, .conservation));
    try testing.expectEqual(@as(usize, 1), countQuantifier(result.set));
    try testing.expectEqual(@as(usize, 1), countQuery(result.set, .base));
    try testing.expectEqual(@as(usize, 10), countQuery(result.set, .obligation));
    try testing.expectEqual(@as(usize, 1), countQuery(result.set, .guard_satisfy));
    try testing.expectEqual(@as(usize, 1), countQuery(result.set, .guard_violate));

    const report = try dumpManifestToOwnedString(testing.allocator, result.set);
    defer testing.allocator.free(report);

    try testing.expect(std.mem.containsAtLeast(u8, report, 1, "{\"record\":\"artifact_decision\",\"schema_version\":12,\"status\":\"blocked\",\"reason\":\"missing_proof\"}"));
    try testing.expect(std.mem.containsAtLeast(u8, report, 1, "{\"record\":\"coverage_summary\",\"schema_version\":12,\"assumptions\":1,\"obligations\":12,\"queries\":13"));
    try testing.expect(std.mem.containsAtLeast(u8, report, 1, "\"query_obligation_links\":12"));
    try testing.expect(std.mem.containsAtLeast(u8, report, 1, "\"obligation_kinds\":{\"logical\":4,\"runtime_guard\":1,\"type_wf\":0,\"type_relation\":0,\"region_relation\":0,\"effect_frame\":1,\"resource\":5,\"quantifier\":1,\"filtered_input\":0,\"backend_fact\":0}"));
    try testing.expect(std.mem.containsAtLeast(u8, report, 1, "\"query_backends\":{\"unspecified\":13,\"z3\":0,\"lean\":0}"));
    try testing.expect(std.mem.containsAtLeast(u8, report, 1, "\"query_results\":{\"missing\":13,\"sat\":0,\"unsat\":0,\"unknown\":0,\"proved\":0,\"failed\":0}"));
    try testing.expect(std.mem.containsAtLeast(u8, report, 1, "\"record\":\"assumption\",\"schema_version\":12"));
    try testing.expect(std.mem.containsAtLeast(u8, report, 1, "\"kind\":{\"tag\":\"logical\",\"role\":\"ensures\""));
    try testing.expect(std.mem.containsAtLeast(u8, report, 1, "\"arithmetic_safety\":\"addition_overflow\""));
    try testing.expect(std.mem.containsAtLeast(u8, report, 1, "\"arithmetic_safety\":\"division_by_zero\""));
    try testing.expect(std.mem.containsAtLeast(u8, report, 1, "\"kind\":{\"tag\":\"runtime_guard\",\"guard_id\":\"guard:coverage:flag\""));
    try testing.expect(std.mem.containsAtLeast(u8, report, 1, "\"kind\":{\"tag\":\"effect_frame\",\"relation\":\"write_covered_by_modifies\""));
    try testing.expect(std.mem.containsAtLeast(u8, report, 1, "\"kind\":{\"tag\":\"resource\",\"op\":\"move\",\"domain\":\"TokenUnit\""));
    try testing.expect(std.mem.containsAtLeast(u8, report, 1, "\"property\":\"conservation\""));
    try testing.expect(std.mem.containsAtLeast(u8, report, 1, "\"kind\":{\"tag\":\"quantifier\",\"quantifier\":\"forall\""));
    try testing.expect(std.mem.containsAtLeast(u8, report, 1, "\"record\":\"query\",\"schema_version\":12"));
}

test "formal loop summary rows are source-owner scoped scalar data" {
    var result = try collectPackageObligations(testing.allocator, "ora-example/smt/verification/loop_invariants.ora");
    defer result.deinit();

    var matched: ?obligation.LoopSummaryRow = null;
    for (result.set.loop_summaries) |row| {
        if (row.owner != .statement) continue;
        if (!std.mem.eql(u8, row.owner.statement.function_name, "sumToN")) continue;
        if (row.owner.statement.ordinal != 4) continue;
        matched = row;
        break;
    }
    const row = matched orelse return error.TestUnexpectedResult;
    try testing.expectEqual(obligation.LoopKind.scf_while, row.loop_kind);
    try testing.expectEqual(@as(?u32, 4), row.loop_source_op_id);
    try testing.expectEqual(@as(u32, 16), row.source.line);
    try testing.expectEqual(@as(usize, 2), row.variables.len);
    try testing.expectEqual(@as(usize, 2), row.init_formulas.len);
    try testing.expectEqual(@as(usize, 3), row.invariant_formulas.len);
    try testing.expectEqual(@as(usize, 2), row.edges.len);
    try testing.expectEqual(@as(u32, 0), row.edges[0].id);
    try testing.expectEqual(obligation.LoopControlTarget.backedge, row.edges[0].target.?);
    try testing.expectEqual(@as(usize, 1), loopEdgePathFormulaCount(row.edges[0]));
    try testing.expectEqual(@as(usize, 2), row.edges[0].next_state.len);
    try testing.expect(row.edges[0].safetyFormulaCount() > 0);
    try testing.expectEqual(@as(u32, 1), row.edges[1].id);
    try testing.expectEqual(obligation.LoopControlTarget.successor, row.edges[1].target.?);
    try testing.expectEqual(@as(usize, 1), loopEdgePathFormulaCount(row.edges[1]));
    try testing.expectEqual(@as(usize, 2), row.edges[1].next_state.len);
    try testing.expectEqual(@as(usize, 0), row.edges[1].safetyFormulaCount());
    for (row.edges) |edge| {
        for (edge.next_state, row.variables, 0..) |binding, variable, index| {
            try testing.expectEqual(@as(u32, @intCast(index)), binding.variable_index);
            try testing.expect(obligation.freeVarIdEql(variable.id.?, binding.target.?));
        }
    }
    try testing.expect(!loopSummaryHasReason(row, .loop_formula_unsupported));
    try testing.expect(row.projectionSupported());
    try testing.expectEqual(@as(usize, 1), row.context_variables.len);
    try testing.expectEqual(@as(usize, 0), row.post_formulas.len);
    try testing.expectEqual(@as(usize, 1), row.query_ids.base.len);
    try testing.expectEqual(@as(usize, 1), row.query_ids.step.len);
    try testing.expectEqual(@as(usize, 2), row.query_ids.body_safety.len);
    try testing.expectEqual(@as(usize, 1), row.query_ids.post.len);
    try testing.expectEqual(@as(usize, 1), row.query_ids.induction.len);
    const induction_query = obligation.findById(result.set.queries, row.query_ids.induction[0]) orelse
        return error.TestUnexpectedResult;
    try testing.expect(induction_query.loop_vc_sha256 != null);

    const report = try dumpManifestToOwnedString(testing.allocator, result.set);
    defer testing.allocator.free(report);
    const repeated_report = try dumpManifestToOwnedString(testing.allocator, result.set);
    defer testing.allocator.free(repeated_report);
    try testing.expectEqualStrings(report, repeated_report);
    try testing.expect(std.mem.containsAtLeast(u8, report, 1, "\"record\":\"loop_summary_coverage\",\"schema_version\":12"));
    try testing.expect(std.mem.containsAtLeast(u8, report, 1, "\"record\":\"loop_summary\",\"schema_version\":12"));
    try testing.expect(std.mem.containsAtLeast(u8, report, 1, "\"edges\":[{\"id\":0"));
    try testing.expect(std.mem.containsAtLeast(u8, report, 1, "\"target\":\"backedge\""));
    try testing.expect(std.mem.containsAtLeast(u8, report, 1, "\"target\":\"successor\""));

    var loop_report = std.Io.Writer.Allocating.init(testing.allocator);
    defer loop_report.deinit();
    try obligation_dump.writeLoopInductionReport(&loop_report.writer, result.set);
    try testing.expect(std.mem.containsAtLeast(u8, loop_report.written(), 1, "\"schema_version\":1"));
    try testing.expect(std.mem.containsAtLeast(
        u8,
        loop_report.written(),
        1,
        "\"verification_boundary\":\"verifier_input_ora_mlir\"",
    ));
    try testing.expect(std.mem.containsAtLeast(u8, loop_report.written(), 1, "\"loop_source_op_id\":4"));
    try testing.expect(std.mem.containsAtLeast(u8, loop_report.written(), 1, "\"status\":\"lean_certificate_required\""));

    var storage_loop: ?obligation.LoopSummaryRow = null;
    var nested_loop: ?obligation.LoopSummaryRow = null;
    var nested_inner_loop: ?obligation.LoopSummaryRow = null;
    for (result.set.loop_summaries) |candidate| {
        if (candidate.owner != .statement) continue;
        if (std.mem.eql(u8, candidate.owner.statement.function_name, "accumulateToStorage")) {
            storage_loop = candidate;
        }
        if (std.mem.eql(u8, candidate.owner.statement.function_name, "nestedSum")) {
            var has_dependency = false;
            for (candidate.edges) |edge| {
                if (loopEdgeDependencyCount(edge) != 0) has_dependency = true;
            }
            if (has_dependency) nested_loop = candidate else nested_inner_loop = candidate;
        }
    }
    const storage = storage_loop orelse return error.TestUnexpectedResult;
    try testing.expect(!loopSummaryHasReason(storage, .loop_has_storage_write));
    try testing.expect(!loopSummaryHasReason(storage, .loop_has_storage_read));
    // The storage-derived requirement is now inspected by the header-fact
    // projector and rejected at its precise stable-value boundary.
    try testing.expect(loopSummaryHasReason(storage, .loop_header_fact_type_unsupported));
    try testing.expectEqual(obligation.LoopProjectionCompleteness.complete, storage.projection_completeness);
    var storage_reads: usize = 0;
    var storage_writes: usize = 0;
    for (storage.edges) |edge| for (edge.timeline) |event| switch (event.payload) {
        .storage_read => storage_reads += 1,
        .storage_write => storage_writes += 1,
        else => {},
    };
    try testing.expectEqual(@as(usize, 3), storage_reads);
    try testing.expectEqual(@as(usize, 1), storage_writes);
    try testing.expect(!storage.projectionSupported());
    const nested = nested_loop orelse return error.TestUnexpectedResult;
    try testing.expect(nested.projectionSupported());
    try testing.expectEqual(obligation.LoopProjectionCompleteness.complete, nested.projection_completeness);
    try testing.expect(!loopSummaryHasReason(nested, .loop_has_nested_loop));
    try testing.expectEqual(@as(usize, 1), loopEdgeDependencyCount(nested.edges[0]));
    const dependency = firstLoopEdgeDependency(nested.edges[0]) orelse return error.TestUnexpectedResult;
    const inner = nested_inner_loop orelse return error.TestUnexpectedResult;
    try testing.expectEqual(inner.id, dependency.summary_id);
    try testing.expectEqual(obligation.LoopControlTarget.successor, dependency.target);
    try testing.expect(dependency.structural_key.len != 0);
    var inner_query: ?obligation.VerificationQuery = null;
    for (result.set.queries) |query| {
        if (query.kind == .loop_induction and query.loop_summary_id == inner.id) inner_query = query;
    }
    const certified_inner_query = inner_query orelse return error.TestUnexpectedResult;
    try testing.expect(certified_inner_query.loop_vc_sha256 != null);
    try testing.expectEqualSlices(u8, &certified_inner_query.loop_vc_sha256.?, &dependency.loop_vc_sha256);

    const outer_induction = obligation.findById(result.set.queries, nested.query_ids.induction[0]) orelse
        return error.TestUnexpectedResult;
    try testing.expectEqual(obligation.ProofRequirement.lean_certificate, outer_induction.proof_requirement);
    try testing.expect(outer_induction.loop_vc_sha256 != null);
    switch (obligation_to_lean.querySemanticSupport(result.set, outer_induction)) {
        .supported => {},
        .unsupported => return error.TestUnexpectedResult,
    }

    var nested_z3_context = try z3_verification.Z3Context.init(testing.allocator);
    defer nested_z3_context.deinit();
    var nested_adapter = obligation_to_z3.Adapter.init(
        &nested_z3_context,
        testing.allocator,
        result.set,
    );
    var nested_canonical_query_count: usize = 0;
    for (result.set.queries) |query| {
        if (query.loop_summary_id != nested.id or query.backend != .z3 or
            query.loop_vc_sha256 == null) continue;
        const checked = try nested_adapter.checkLoopQueryForRow(query);
        try testing.expectEqual(obligation_to_z3.CheckStatus.proved, checked.status);
        nested_canonical_query_count += 1;
    }
    try testing.expectEqual(@as(usize, 5), nested_canonical_query_count);

    var nested_lean = std.Io.Writer.Allocating.init(testing.allocator);
    defer nested_lean.deinit();
    try obligation_to_lean.writeModule(&nested_lean.writer, result.set, .{
        .namespace = "Ora.Generated.NestedLoopProjection",
        .proof_surface = true,
    });
    try testing.expect(std.mem.containsAtLeast(
        u8,
        nested_lean.written(),
        1,
        "Ora.Loop.loopProofAuthorizationFromAssumptionsIn",
    ));
    try testing.expect(std.mem.containsAtLeast(
        u8,
        nested_lean.written(),
        1,
        "emittedLoopSummaries",
    ));
}

test "formal direct-root storage loops preserve verifier-input event order" {
    var result = try collectPackageObligations(testing.allocator, "tests/formal/storage_loop_projection.ora");
    defer result.deinit();

    const Expected = struct {
        function_name: []const u8,
        paths: []const []const obligation.LoopTimelineEventTag,
    };
    const expected = [_]Expected{
        .{ .function_name = "readThenWrite", .paths = &.{&.{ .storage_read, .safety_checkpoint, .storage_write, .safety_checkpoint }} },
        .{ .function_name = "divideFromStorage", .paths = &.{&.{ .storage_read, .safety_checkpoint, .storage_write, .safety_checkpoint }} },
        .{ .function_name = "writeThenRead", .paths = &.{&.{ .storage_write, .storage_read, .storage_write, .safety_checkpoint }} },
        .{ .function_name = "repeatedWrite", .paths = &.{&.{ .storage_write, .safety_checkpoint, .storage_write }} },
        .{ .function_name = "branchWrite", .paths = &.{ &.{ .path_assumption, .storage_write, .safety_checkpoint }, &.{ .path_assumption, .storage_write, .safety_checkpoint } } },
        .{ .function_name = "zeroIteration", .paths = &.{&.{ .storage_write, .safety_checkpoint }} },
        .{ .function_name = "signedWrite", .paths = &.{&.{ .storage_write, .safety_checkpoint }} },
    };

    for (expected) |want| {
        var matched: ?obligation.LoopSummaryRow = null;
        for (result.set.loop_summaries) |summary| {
            if (summary.owner == .statement and
                std.mem.eql(u8, summary.owner.statement.function_name, want.function_name)) matched = summary;
        }
        const summary = matched orelse return error.TestUnexpectedResult;
        try testing.expect(summary.projectionSupported());
        try testing.expectEqual(obligation.LoopProjectionCompleteness.complete, summary.projection_completeness);
        var path_index: usize = 0;
        for (summary.edges) |edge| {
            if (edge.target != .backedge) continue;
            if (path_index >= want.paths.len) return error.TestUnexpectedResult;
            var observed: std.ArrayList(obligation.LoopTimelineEventTag) = .empty;
            defer observed.deinit(testing.allocator);
            var first_read: ?obligation.FreeVarId = null;
            for (edge.timeline, 0..) |event, event_index| {
                try testing.expectEqual(@as(u32, @intCast(event_index)), event.id);
                const tag = std.meta.activeTag(event.payload);
                switch (event.payload) {
                    .storage_read => |read| first_read = read.result,
                    .storage_write => |write| {
                        if (std.mem.eql(u8, want.function_name, "signedWrite")) try testing.expectEqual(
                            type_builtin.lookupBuiltinById(.i8).comptime_type_id,
                            write.location.value_ty.compiler_type_id,
                        );
                        if (std.mem.eql(u8, want.function_name, "branchWrite")) try testing.expectEqualStrings(
                            if (path_index == 0) "left" else "right",
                            write.location.root,
                        );
                    },
                    .safety_checkpoint => |checkpoint| {
                        if ((std.mem.eql(u8, want.function_name, "readThenWrite") or
                            std.mem.eql(u8, want.function_name, "divideFromStorage")) and first_read != null)
                        {
                            try testing.expectEqual(@as(usize, 1), checkpoint.formulas.len);
                            try testing.expectEqual(@as(usize, 1), checkpoint.formulas[0].read_results.len);
                            try testing.expect(obligation.freeVarIdEql(
                                first_read.?,
                                checkpoint.formulas[0].read_results[0],
                            ));
                            first_read = null;
                        }
                    },
                    else => {},
                }
                if (event_index == 0 and tag == .path_assumption) continue;
                try observed.append(testing.allocator, tag);
            }
            try testing.expectEqualSlices(obligation.LoopTimelineEventTag, want.paths[path_index], observed.items);
            path_index += 1;
        }
        try testing.expectEqual(want.paths.len, path_index);
    }

    var lean = std.Io.Writer.Allocating.init(testing.allocator);
    defer lean.deinit();
    try obligation_to_lean.writeModule(&lean.writer, result.set, .{
        .namespace = "Ora.Generated.StorageLoopProjection",
        .proof_surface = true,
    });
    var tmp = testing.tmpDir(.{});
    defer tmp.cleanup();
    const path = try pathFromTmpAlloc(testing.allocator, tmp, "StorageLoopProjection.lean");
    defer testing.allocator.free(path);
    try std.Io.Dir.cwd().writeFile(std.testing.io, .{ .sub_path = path, .data = lean.written() });
    const parsed = try runLeanFileForTest(testing.allocator, path);
    defer testing.allocator.free(parsed.stdout);
    defer testing.allocator.free(parsed.stderr);
    try expectExited(parsed, 0);
}

test "canonical Z3 executes direct-root storage timelines deterministically" {
    var result = try collectPackageObligations(testing.allocator, "tests/formal/storage_loop_projection.ora");
    defer result.deinit();

    var first_context = try z3_verification.Z3Context.init(testing.allocator);
    defer first_context.deinit();
    var second_context = try z3_verification.Z3Context.init(testing.allocator);
    defer second_context.deinit();
    var first = obligation_to_z3.Adapter.init(&first_context, testing.allocator, result.set);
    var second = obligation_to_z3.Adapter.init(&second_context, testing.allocator, result.set);
    var checked_count: usize = 0;
    var saw_read_then_write_failure = false;
    var saw_divide_from_read_failure = false;
    var saw_write_then_read_success = false;
    var saw_signed_success = false;

    for (result.set.queries) |query| {
        if (query.backend != .z3 or query.loop_summary_id == null or
            query.loop_vc_sha256 == null) continue;
        const summary = obligation.findById(result.set.loop_summaries, query.loop_summary_id.?) orelse
            return error.TestUnexpectedResult;
        if (summary.owner != .statement) return error.TestUnexpectedResult;
        const lhs = try first.checkLoopQueryForRow(query);
        const rhs = try second.checkLoopQueryForRow(query);
        try testing.expectEqual(lhs.status, rhs.status);
        try testing.expectEqual(lhs.query_hash.constraint_count, rhs.query_hash.constraint_count);
        try testing.expectEqual(lhs.query_hash.smtlib_hash, rhs.query_hash.smtlib_hash);
        try testing.expect(lhs.status != .unknown);
        try testing.expect(lhs.query_hash.constraint_count != 0 or query.kind == .loop_base);
        checked_count += 1;

        const name = summary.owner.statement.function_name;
        if (query.kind == .loop_body_safety and lhs.status == .disproved) {
            if (std.mem.eql(u8, name, "readThenWrite")) saw_read_then_write_failure = true;
            if (std.mem.eql(u8, name, "divideFromStorage")) saw_divide_from_read_failure = true;
        }
        if (query.kind == .loop_invariant_step and lhs.status == .proved) {
            if (std.mem.eql(u8, name, "writeThenRead")) saw_write_then_read_success = true;
            if (std.mem.eql(u8, name, "signedWrite")) saw_signed_success = true;
        }
    }

    try testing.expect(checked_count != 0);
    try testing.expect(saw_read_then_write_failure);
    try testing.expect(saw_divide_from_read_failure);
    try testing.expect(saw_write_then_read_success);
    try testing.expect(saw_signed_success);
}

test "external direct-root storage certificate binds canonical Z3 and Lean through the real CLI" {
    std.Io.Dir.cwd().access(std.testing.io, ORA_BINARY_REL, .{}) catch |err| switch (err) {
        error.FileNotFound => return error.SkipZigTest,
        else => return err,
    };

    const allocator = testing.allocator;
    const source = "tests/formal/direct_storage_loop_certificate.ora";
    const proofs = "tests/formal/direct_storage_loop_certificate.proofs.json";
    const stem = "direct_storage_loop_certificate";
    var tmp = testing.tmpDir(.{});
    defer tmp.cleanup();
    const accepted_paths = [_][]const u8{ "accepted-first", "accepted-second" };

    const empty_manifest = try pathFromTmpAlloc(allocator, tmp, "empty-proofs.json");
    defer allocator.free(empty_manifest);
    try std.Io.Dir.cwd().writeFile(std.testing.io, .{
        .sub_path = empty_manifest,
        .data = "{\"schema_version\":1,\"proofs\":[]}",
    });
    const blocked_output = try pathFromTmpAlloc(allocator, tmp, "missing-proof");
    defer allocator.free(blocked_output);
    const blocked = try runProcessWithLeanPath(allocator, &.{
        ORA_BINARY_REL,
        "emit",
        "--emit=smt-report,bytecode",
        "--out-dir",
        blocked_output,
        "--lean-proofs",
        empty_manifest,
        source,
    });
    defer allocator.free(blocked.stdout);
    defer allocator.free(blocked.stderr);
    try expectExited(blocked, 1);
    try testing.expect(std.mem.containsAtLeast(u8, blocked.stdout, 1, "lean_required_failure"));
    try testing.expectError(
        error.FileNotFound,
        tmp.dir.access(std.testing.io, "missing-proof/direct_storage_loop_certificate.hex", .{}),
    );
    const blocked_index_path = try pathFromTmpAlloc(
        allocator,
        tmp,
        "missing-proof/direct_storage_loop_certificate.formal.artifacts.json",
    );
    defer allocator.free(blocked_index_path);
    const blocked_index = try readFileAllocForTest(allocator, blocked_index_path);
    defer allocator.free(blocked_index);
    try testing.expect(std.mem.containsAtLeast(
        u8,
        blocked_index,
        1,
        "\"final_artifact_authorization\": \"blocked\"",
    ));

    for (accepted_paths) |relative| {
        const output = try pathFromTmpAlloc(allocator, tmp, relative);
        defer allocator.free(output);
        const run = try runProcessWithLeanPath(allocator, &.{
            ORA_BINARY_REL,
            "emit",
            "--emit=smt-report,bytecode",
            "--out-dir",
            output,
            "--lean-proofs",
            proofs,
            source,
        });
        defer allocator.free(run.stdout);
        defer allocator.free(run.stderr);
        try expectExited(run, 0);
        try testing.expect(std.mem.containsAtLeast(
            u8,
            run.stdout,
            1,
            "Lean loop proof gate accepted:\n  kernel-checked certificates: 1",
        ));
    }

    const first_hex_path = try pathFromTmpAlloc(
        allocator,
        tmp,
        "accepted-first/direct_storage_loop_certificate.hex",
    );
    defer allocator.free(first_hex_path);
    const first_hex = try readFileAllocForTest(allocator, first_hex_path);
    defer allocator.free(first_hex);
    const second_hex_path = try pathFromTmpAlloc(
        allocator,
        tmp,
        "accepted-second/direct_storage_loop_certificate.hex",
    );
    defer allocator.free(second_hex_path);
    const second_hex = try readFileAllocForTest(allocator, second_hex_path);
    defer allocator.free(second_hex);
    try testing.expectEqualStrings(first_hex, second_hex);

    const first_certificate_path = try pathFromTmpAlloc(
        allocator,
        tmp,
        "accepted-first/direct_storage_loop_certificate.lean.proof.json",
    );
    defer allocator.free(first_certificate_path);
    const second_certificate_path = try pathFromTmpAlloc(
        allocator,
        tmp,
        "accepted-second/direct_storage_loop_certificate.lean.proof.json",
    );
    defer allocator.free(second_certificate_path);
    const first_certificate = try readFileAllocForTest(allocator, first_certificate_path);
    defer allocator.free(first_certificate);
    const second_certificate = try readFileAllocForTest(allocator, second_certificate_path);
    defer allocator.free(second_certificate);
    try testing.expectEqualStrings(first_certificate, second_certificate);

    const index_path = try pathFromTmpAlloc(
        allocator,
        tmp,
        "accepted-first/direct_storage_loop_certificate.formal.artifacts.json",
    );
    defer allocator.free(index_path);
    const index = try readFileAllocForTest(allocator, index_path);
    defer allocator.free(index);
    try testing.expect(std.mem.containsAtLeast(u8, index, 1, "\"final_artifact_authorization\": \"allowed\""));
    try testing.expect(std.mem.containsAtLeast(u8, index, 1, "\"id\": \"z3_userland\""));
    try testing.expect(std.mem.containsAtLeast(u8, index, 1, "\"id\": \"lean_userland\""));
    try testing.expect(std.mem.containsAtLeast(u8, index, 1, "\"id\": \"lean_proof_certificate\""));

    try expectNoFormalArtifactsInRepoRoot(allocator, stem);
}

test "external mapping preservation certificate binds canonical Z3 and Lean through the real CLI" {
    std.Io.Dir.cwd().access(std.testing.io, ORA_BINARY_REL, .{}) catch |err| switch (err) {
        error.FileNotFound => return error.SkipZigTest,
        else => return err,
    };

    const allocator = testing.allocator;
    const stem = "mapping_storage_preservation_certificate";
    var tmp = testing.tmpDir(.{});
    defer tmp.cleanup();
    const output = try pathFromTmpAlloc(allocator, tmp, "accepted");
    defer allocator.free(output);
    const run = try runProcessWithLeanPath(allocator, &.{
        ORA_BINARY_REL,
        "emit",
        "--emit=smt-report,bytecode",
        "--out-dir",
        output,
        "--lean-proofs",
        "tests/formal/mapping_storage_preservation_certificate.proofs.json",
        "tests/formal/mapping_storage_preservation_certificate.ora",
    });
    defer allocator.free(run.stdout);
    defer allocator.free(run.stderr);
    try expectExited(run, 0);
    try testing.expect(std.mem.containsAtLeast(
        u8,
        run.stdout,
        1,
        "Lean loop proof gate accepted:\n  kernel-checked certificates: 1",
    ));
    try tmp.dir.access(std.testing.io, "accepted/mapping_storage_preservation_certificate.hex", .{});
    const certificate = try tmp.dir.readFileAlloc(
        std.testing.io,
        "accepted/mapping_storage_preservation_certificate.lean.proof.json",
        allocator,
        .limited(1024 * 1024),
    );
    defer allocator.free(certificate);
    try testing.expect(std.mem.containsAtLeast(u8, certificate, 1, "\"proof_count\": 2"));
    try testing.expect(std.mem.containsAtLeast(u8, certificate, 2, "\"lean_status\": \"checked\""));
    try expectNoFormalArtifactsInRepoRoot(allocator, stem);
}

test "external fixed-array preservation certificate binds canonical Z3 and Lean deterministically" {
    try expectExternalPreheaderCertificateDeterministic(
        "tests/formal/fixed_storage_preservation_certificate.ora",
        "tests/formal/fixed_storage_preservation_certificate.proofs.json",
        "fixed_storage_preservation_certificate",
        2,
    );
}

test "projected storage preservation without external certificates remains blocked" {
    std.Io.Dir.cwd().access(std.testing.io, ORA_BINARY_REL, .{}) catch |err| switch (err) {
        error.FileNotFound => return error.SkipZigTest,
        else => return err,
    };

    const allocator = testing.allocator;
    var tmp = testing.tmpDir(.{});
    defer tmp.cleanup();
    const cases = [_]struct {
        source: []const u8,
        stem: []const u8,
        diagnostic: []const u8,
        z3_diagnostic: ?[]const u8 = null,
    }{
        .{
            .source = "tests/formal/fail_mapping_storage_preheader_same_key.ora",
            .stem = "fail_mapping_storage_preheader_same_key",
            .diagnostic = "Automatic Lean loop proof unavailable",
            .z3_diagnostic = "failed to prove invariant in writeOther",
        },
        .{
            .source = "tests/formal/fail_mapping_storage_preheader_other_key.ora",
            .stem = "fail_mapping_storage_preheader_other_key",
            .diagnostic = "Automatic Lean loop proof unavailable",
        },
        .{
            .source = "tests/formal/fail_mapping_storage_preservation_no_inequality.ora",
            .stem = "fail_mapping_storage_preservation_no_inequality",
            .diagnostic = "Automatic Lean loop proof unavailable",
        },
        .{
            .source = "tests/formal/fail_fixed_storage_preservation_no_inequality.ora",
            .stem = "fail_fixed_storage_preservation_no_inequality",
            .diagnostic = "Automatic Lean loop proof unavailable",
        },
    };
    for (cases, 0..) |case, index| {
        const relative = try std.fmt.allocPrint(allocator, "blocked-{d}", .{index});
        defer allocator.free(relative);
        const output = try pathFromTmpAlloc(allocator, tmp, relative);
        defer allocator.free(output);
        const run = try runProcessWithLeanPath(allocator, &.{
            ORA_BINARY_REL,
            "emit",
            "--emit=bytecode",
            "--out-dir",
            output,
            "--lean-proofs",
            case.source,
        });
        defer allocator.free(run.stdout);
        defer allocator.free(run.stderr);
        try expectExited(run, 1);
        try testing.expect(std.mem.containsAtLeast(u8, run.stdout, 1, case.diagnostic));
        if (case.z3_diagnostic) |diagnostic| {
            try testing.expect(std.mem.containsAtLeast(u8, run.stdout, 1, diagnostic));
            try testing.expect(std.mem.containsAtLeast(
                u8,
                run.stdout,
                1,
                "userland verification failed (Z3: rejected, Lean obligation gate: rejected)",
            ));
        }
        const hex_path = try std.fmt.allocPrint(allocator, "{s}/{s}.hex", .{ relative, case.stem });
        defer allocator.free(hex_path);
        try testing.expectError(error.FileNotFound, tmp.dir.access(std.testing.io, hex_path, .{}));
        try expectNoFormalArtifactsInRepoRoot(allocator, case.stem);
    }
}

fn expectExternalPreheaderCertificateDeterministic(
    source: []const u8,
    proofs: []const u8,
    stem: []const u8,
    proof_count: usize,
) !void {
    std.Io.Dir.cwd().access(std.testing.io, ORA_BINARY_REL, .{}) catch |err| switch (err) {
        error.FileNotFound => return error.SkipZigTest,
        else => return err,
    };

    const allocator = testing.allocator;
    var obligations = try collectPackageObligations(allocator, source);
    defer obligations.deinit();
    var first_module = std.Io.Writer.Allocating.init(allocator);
    defer first_module.deinit();
    try obligation_to_lean.writeModule(&first_module.writer, obligations.set, .{});
    var second_module = std.Io.Writer.Allocating.init(allocator);
    defer second_module.deinit();
    try obligation_to_lean.writeModule(&second_module.writer, obligations.set, .{});
    try testing.expectEqualStrings(first_module.written(), second_module.written());

    var tmp = testing.tmpDir(.{});
    defer tmp.cleanup();

    for ([_][]const u8{ "first", "second" }) |relative| {
        const output = try pathFromTmpAlloc(allocator, tmp, relative);
        defer allocator.free(output);
        const run = try runProcessWithLeanPath(allocator, &.{
            ORA_BINARY_REL,
            "emit",
            "--emit=smt-report,bytecode",
            "--out-dir",
            output,
            "--lean-proofs",
            proofs,
            source,
        });
        defer allocator.free(run.stdout);
        defer allocator.free(run.stderr);
        try expectExited(run, 0);
        try testing.expect(std.mem.containsAtLeast(
            u8,
            run.stdout,
            1,
            "Lean loop proof gate accepted:\n  kernel-checked certificates: 1",
        ));

        const certificate_path = try std.fmt.allocPrint(
            allocator,
            "{s}/{s}.lean.proof.json",
            .{ relative, stem },
        );
        defer allocator.free(certificate_path);
        const certificate = try tmp.dir.readFileAlloc(
            std.testing.io,
            certificate_path,
            allocator,
            .limited(1024 * 1024),
        );
        defer allocator.free(certificate);
        const proof_count_text = try std.fmt.allocPrint(
            allocator,
            "\"proof_count\": {d}",
            .{proof_count},
        );
        defer allocator.free(proof_count_text);
        try testing.expect(std.mem.containsAtLeast(u8, certificate, 1, proof_count_text));
        const hex_path = try std.fmt.allocPrint(allocator, "{s}/{s}.hex", .{ relative, stem });
        defer allocator.free(hex_path);
        try tmp.dir.access(std.testing.io, hex_path, .{});
    }

    inline for (.{
        ".hex",
        ".lean.proof.json",
        ".formal.artifacts.json",
        ".smt.report.json",
        ".smt.report.md",
    }) |suffix| {
        const first_path = try std.fmt.allocPrint(allocator, "first/{s}{s}", .{ stem, suffix });
        defer allocator.free(first_path);
        const second_path = try std.fmt.allocPrint(allocator, "second/{s}{s}", .{ stem, suffix });
        defer allocator.free(second_path);
        const first = try tmp.dir.readFileAlloc(
            std.testing.io,
            first_path,
            allocator,
            .limited(16 * 1024 * 1024),
        );
        defer allocator.free(first);
        const second = try tmp.dir.readFileAlloc(
            std.testing.io,
            second_path,
            allocator,
            .limited(16 * 1024 * 1024),
        );
        defer allocator.free(second);
        try testing.expectEqualStrings(first, second);
    }
    try expectNoFormalArtifactsInRepoRoot(allocator, stem);
}

test "external certificate binds a nonempty direct-root preheader deterministically" {
    try expectExternalPreheaderCertificateDeterministic(
        "tests/formal/direct_storage_preheader_certificate.ora",
        "tests/formal/direct_storage_preheader_certificate.proofs.json",
        "direct_storage_preheader_certificate",
        2,
    );
}

test "external certificate binds a nonempty reviewed mapping preheader deterministically" {
    try expectExternalPreheaderCertificateDeterministic(
        "tests/formal/mapping_storage_preheader_certificate.ora",
        "tests/formal/mapping_storage_preheader_certificate.proofs.json",
        "mapping_storage_preheader_certificate",
        3,
    );
}

test "an earlier loop blocks a proof-relevant nonempty preheader by name" {
    std.Io.Dir.cwd().access(std.testing.io, ORA_BINARY_REL, .{}) catch |err| switch (err) {
        error.FileNotFound => return error.SkipZigTest,
        else => return err,
    };

    const allocator = testing.allocator;
    var tmp = testing.tmpDir(.{});
    defer tmp.cleanup();
    const output = try pathFromTmpAlloc(allocator, tmp, "blocked");
    defer allocator.free(output);
    const run = try runProcessWithLeanPath(allocator, &.{
        ORA_BINARY_REL,
        "emit",
        "--emit=bytecode",
        "--out-dir",
        output,
        "--lean-proofs",
        "tests/formal/fail_preheader_control_projection.ora",
    });
    defer allocator.free(run.stdout);
    defer allocator.free(run.stderr);
    try expectExited(run, 1);
    try testing.expect(std.mem.containsAtLeast(
        u8,
        run.stdout,
        1,
        "excluded_by=loop_preheader_control_unsupported",
    ));
    try testing.expectError(
        error.FileNotFound,
        tmp.dir.access(std.testing.io, "blocked/fail_preheader_control_projection.hex", .{}),
    );
    try expectNoFormalArtifactsInRepoRoot(allocator, "fail_preheader_control_projection");
}

test "a nonempty preheader cannot certify a false storage invariant" {
    std.Io.Dir.cwd().access(std.testing.io, ORA_BINARY_REL, .{}) catch |err| switch (err) {
        error.FileNotFound => return error.SkipZigTest,
        else => return err,
    };

    const allocator = testing.allocator;
    var tmp = testing.tmpDir(.{});
    defer tmp.cleanup();
    const output = try pathFromTmpAlloc(allocator, tmp, "blocked");
    defer allocator.free(output);
    const run = try runProcessWithLeanPath(allocator, &.{
        ORA_BINARY_REL,
        "emit",
        "--emit=bytecode",
        "--out-dir",
        output,
        "--lean-proofs",
        "tests/formal/fail_direct_storage_preheader_false_invariant.ora",
    });
    defer allocator.free(run.stdout);
    defer allocator.free(run.stderr);
    try expectExited(run, 1);
    try testing.expect(std.mem.containsAtLeast(
        u8,
        run.stdout,
        1,
        "userland verification failed (Z3: rejected, Lean obligation gate: rejected)",
    ));
    try testing.expectError(
        error.FileNotFound,
        tmp.dir.access(
            std.testing.io,
            "blocked/fail_direct_storage_preheader_false_invariant.hex",
            .{},
        ),
    );
    try expectNoFormalArtifactsInRepoRoot(
        allocator,
        "fail_direct_storage_preheader_false_invariant",
    );
}

test "canonical preheader projection inventories every operation before the loop" {
    var result = try collectPackageObligations(
        testing.allocator,
        "tests/formal/fail_mapping_storage_preheader_other_key.ora",
    );
    defer result.deinit();

    try testing.expectEqual(@as(usize, 1), result.set.loop_summaries.len);
    const summary = result.set.loop_summaries[0];
    try testing.expect(summary.preheader_projection_complete);
    try testing.expectEqual(@as(usize, 3), summary.preheader_timeline.len);
    try testing.expectEqual(obligation.LoopTimelineEventTag.path_assumption, std.meta.activeTag(
        summary.preheader_timeline[0].payload,
    ));
    try testing.expectEqual(obligation.LoopTimelineEventTag.safety_checkpoint, std.meta.activeTag(
        summary.preheader_timeline[1].payload,
    ));
    try testing.expectEqual(obligation.LoopTimelineEventTag.storage_write, std.meta.activeTag(
        summary.preheader_timeline[2].payload,
    ));
    const write = summary.preheader_timeline[2].payload.storage_write;
    try testing.expectEqualStrings("values", write.location.root);
    try testing.expectEqual(@as(usize, 1), write.location.projections.len);
    try testing.expect(write.location.projections[0] == .mapping_key);
    try testing.expect(!loopSummaryHasReason(summary, .loop_preheader_storage_relation_unsupported));
}

test "preheader projection failures are named at the canonical boundary" {
    const cases = [_]struct {
        source: []const u8,
        function: []const u8,
        reason: obligation.LoopUnsupportedReason,
    }{
        .{
            .source = "tests/formal/fail_preheader_control_projection.ora",
            .function = "run",
            .reason = .loop_preheader_control_unsupported,
        },
        .{
            .source = "tests/formal/fail_preheader_effect_projection.ora",
            .function = "run",
            .reason = .loop_preheader_effect_unsupported,
        },
        .{
            .source = "tests/formal/fail_preheader_location_projection.ora",
            .function = "run",
            .reason = .loop_preheader_location_unsupported,
        },
    };
    for (cases) |case| {
        var result = try collectPackageObligations(testing.allocator, case.source);
        defer result.deinit();
        var found = false;
        for (result.set.loop_summaries) |summary| {
            if (summary.owner != .statement or
                !std.mem.eql(u8, summary.owner.statement.function_name, case.function) or
                !loopSummaryHasReason(summary, case.reason)) continue;
            try testing.expect(!summary.preheader_projection_complete);
            found = true;
        }
        try testing.expect(found);
    }
}

test "canonical mapping preservation step rejects absent key inequality" {
    var result = try collectPackageObligations(
        testing.allocator,
        "tests/formal/fail_mapping_storage_preservation_no_inequality.ora",
    );
    defer result.deinit();
    var context = try z3_verification.Z3Context.init(testing.allocator);
    defer context.deinit();
    var adapter = obligation_to_z3.Adapter.init(&context, testing.allocator, result.set);
    var checked = false;
    for (result.set.queries) |query| {
        if (query.kind != .loop_invariant_step or query.loop_summary_id == null) continue;
        const verdict = try adapter.checkLoopQueryForRow(query);
        try testing.expectEqual(obligation_to_z3.CheckStatus.disproved, verdict.status);
        checked = true;
    }
    try testing.expect(checked);
}

test "certified call specifications resolve constant array places with exact old storage" {
    var result = try collectPackageObligations(testing.allocator, "tests/formal/loop_constant_storage_place.ora");
    defer result.deinit();
    try testing.expectEqual(@as(usize, 2), result.set.loop_summaries.len);
    var context = try z3_verification.Z3Context.init(testing.allocator);
    defer context.deinit();
    var adapter = obligation_to_z3.Adapter.init(&context, testing.allocator, result.set);
    var checked: usize = 0;
    for (result.set.queries) |query| {
        if (query.kind != .loop_invariant_step or query.loop_summary_id == null) continue;
        const summary = for (result.set.loop_summaries) |row| {
            if (row.id == query.loop_summary_id.?) break row;
        } else return error.TestUnexpectedResult;
        try testing.expect(summary.projectionSupported());
        const verdict = try adapter.checkLoopQueryForRow(query);
        const preserves = std.mem.eql(u8, summary.owner.statement.function_name, "preserve");
        try testing.expectEqual(if (preserves) obligation_to_z3.CheckStatus.proved else .disproved, verdict.status);
        checked += 1;
    }
    try testing.expectEqual(@as(usize, 2), checked);
}

test "canonical fixed-array preservation step rejects absent index inequality" {
    var result = try collectPackageObligations(
        testing.allocator,
        "tests/formal/fail_fixed_storage_preservation_no_inequality.ora",
    );
    defer result.deinit();
    var context = try z3_verification.Z3Context.init(testing.allocator);
    defer context.deinit();
    var adapter = obligation_to_z3.Adapter.init(&context, testing.allocator, result.set);
    var checked = false;
    for (result.set.queries) |query| {
        if (query.kind != .loop_invariant_step or query.loop_summary_id == null) continue;
        const verdict = try adapter.checkLoopQueryForRow(query);
        try testing.expectEqual(obligation_to_z3.CheckStatus.disproved, verdict.status);
        checked = true;
    }
    try testing.expect(checked);
}

fn expectRepositoryStorageTacticDeterministic(
    source: []const u8,
    stem: []const u8,
    expected_certificate_count: usize,
) !void {
    std.Io.Dir.cwd().access(std.testing.io, ORA_BINARY_REL, .{}) catch |err| switch (err) {
        error.FileNotFound => return error.SkipZigTest,
        else => return err,
    };

    const allocator = testing.allocator;
    var tmp = testing.tmpDir(.{});
    defer tmp.cleanup();
    const accepted_text = try std.fmt.allocPrint(
        allocator,
        "Lean loop proof gate accepted:\n  kernel-checked certificates: {d}",
        .{expected_certificate_count},
    );
    defer allocator.free(accepted_text);
    for ([_][]const u8{ "first", "second" }) |relative| {
        const output = try pathFromTmpAlloc(allocator, tmp, relative);
        defer allocator.free(output);
        const run = try runProcessWithLeanPath(allocator, &.{
            ORA_BINARY_REL,
            "emit",
            "--emit=smt-report,bytecode",
            "--out-dir",
            output,
            "--lean-proofs",
            source,
        });
        defer allocator.free(run.stdout);
        defer allocator.free(run.stderr);
        try expectExited(run, 0);
        try testing.expect(std.mem.containsAtLeast(
            u8,
            run.stdout,
            1,
            accepted_text,
        ));
        try testing.expect(!std.mem.containsAtLeast(
            u8,
            run.stdout,
            1,
            "automatic_storage_timeline_requires_external_certificate",
        ));
    }

    inline for (.{ ".hex", ".lean.proof.json", ".smt.report.json", ".smt.report.md" }) |suffix| {
        const first_path = try std.fmt.allocPrint(
            allocator,
            "first/{s}{s}",
            .{ stem, suffix },
        );
        defer allocator.free(first_path);
        const second_path = try std.fmt.allocPrint(
            allocator,
            "second/{s}{s}",
            .{ stem, suffix },
        );
        defer allocator.free(second_path);
        const first = try tmp.dir.readFileAlloc(std.testing.io, first_path, allocator, .limited(16 * 1024 * 1024));
        defer allocator.free(first);
        const second = try tmp.dir.readFileAlloc(std.testing.io, second_path, allocator, .limited(16 * 1024 * 1024));
        defer allocator.free(second);
        try testing.expectEqualStrings(first, second);
    }
    try expectNoFormalArtifactsInRepoRoot(allocator, stem);
}

fn expectRepositoryLoopAuthorityBlocked(
    source: []const u8,
    stem: []const u8,
    reason: obligation.ArtifactBlockReason,
) !void {
    std.Io.Dir.cwd().access(std.testing.io, ORA_BINARY_REL, .{}) catch |err| switch (err) {
        error.FileNotFound => return error.SkipZigTest,
        else => return err,
    };

    const allocator = testing.allocator;
    var tmp = testing.tmpDir(.{});
    defer tmp.cleanup();
    const decision_text = try std.fmt.allocPrint(
        allocator,
        "Automatic Lean proof gate did not authorize artifact emission: {s}",
        .{@tagName(reason)},
    );
    defer allocator.free(decision_text);

    for ([_][]const u8{ "first", "second" }) |relative| {
        const output = try pathFromTmpAlloc(allocator, tmp, relative);
        defer allocator.free(output);
        const run = try runProcessWithLeanPath(allocator, &.{
            ORA_BINARY_REL,
            "emit",
            "--emit=smt-report,bytecode",
            "--out-dir",
            output,
            "--lean-proofs",
            source,
        });
        defer allocator.free(run.stdout);
        defer allocator.free(run.stderr);
        try expectExited(run, 1);
        try testing.expect(std.mem.containsAtLeast(u8, run.stdout, 1, decision_text));
        try testing.expect(std.mem.containsAtLeast(
            u8,
            run.stdout,
            1,
            "userland verification failed (Z3: accepted, Lean obligation gate: rejected)",
        ));
        try testing.expect(!std.mem.containsAtLeast(
            u8,
            run.stdout,
            1,
            "Automatic Lean proof gate failed",
        ));

        const hex_path = try std.fmt.allocPrint(allocator, "{s}/{s}.hex", .{ relative, stem });
        defer allocator.free(hex_path);
        try testing.expectError(
            error.FileNotFound,
            tmp.dir.access(std.testing.io, hex_path, .{}),
        );

        const index_path = try std.fmt.allocPrint(
            allocator,
            "{s}/{s}.formal.artifacts.json",
            .{ relative, stem },
        );
        defer allocator.free(index_path);
        const index = try tmp.dir.readFileAlloc(
            std.testing.io,
            index_path,
            allocator,
            .limited(1024 * 1024),
        );
        defer allocator.free(index);
        try testing.expect(std.mem.containsAtLeast(
            u8,
            index,
            1,
            "\"final_artifact_authorization\": \"blocked\"",
        ));

        var output_dir = try tmp.dir.openDir(std.testing.io, relative, .{ .iterate = true });
        defer output_dir.close(std.testing.io);
        var entries = output_dir.iterate();
        while (try entries.next(std.testing.io)) |entry| {
            try testing.expect(!std.mem.startsWith(u8, entry.name, ".ora-staging-"));
        }
    }

    inline for (.{
        ".formal.accounting.json",
        ".formal.artifacts.json",
        ".formal.loops.json",
        ".lean.obligations.lean",
        ".smt.report.json",
        ".smt.report.md",
    }) |suffix| {
        const first_path = try std.fmt.allocPrint(allocator, "first/{s}{s}", .{ stem, suffix });
        defer allocator.free(first_path);
        const second_path = try std.fmt.allocPrint(allocator, "second/{s}{s}", .{ stem, suffix });
        defer allocator.free(second_path);
        const first = try tmp.dir.readFileAlloc(
            std.testing.io,
            first_path,
            allocator,
            .limited(16 * 1024 * 1024),
        );
        defer allocator.free(first);
        const second = try tmp.dir.readFileAlloc(
            std.testing.io,
            second_path,
            allocator,
            .limited(16 * 1024 * 1024),
        );
        defer allocator.free(second);
        try testing.expectEqualStrings(first, second);
    }
    try expectNoFormalArtifactsInRepoRoot(allocator, stem);
}

fn expectRepositoryLoopProofConstructionBlocked(
    source: []const u8,
    stem: []const u8,
    expected_certificate_count: usize,
    expected_reason: []const u8,
) !void {
    std.Io.Dir.cwd().access(std.testing.io, ORA_BINARY_REL, .{}) catch |err| switch (err) {
        error.FileNotFound => return error.SkipZigTest,
        else => return err,
    };

    const allocator = testing.allocator;
    var tmp = testing.tmpDir(.{});
    defer tmp.cleanup();
    const output = try pathFromTmpAlloc(allocator, tmp, "blocked");
    defer allocator.free(output);
    const expected_summary = try std.fmt.allocPrint(
        allocator,
        "total={d}, certificate-required={d}, excluded=0",
        .{ expected_certificate_count, expected_certificate_count },
    );
    defer allocator.free(expected_summary);
    const run = try runProcessWithLeanPath(allocator, &.{
        ORA_BINARY_REL,
        "emit",
        "--emit=smt-report,bytecode",
        "--out-dir",
        output,
        "--lean-proofs",
        source,
    });
    defer allocator.free(run.stdout);
    defer allocator.free(run.stderr);
    try expectExited(run, 1);
    try testing.expect(std.mem.containsAtLeast(u8, run.stdout, 1, expected_summary));
    try testing.expect(std.mem.containsAtLeast(
        u8,
        run.stdout,
        1,
        expected_reason,
    ));
    try testing.expect(std.mem.containsAtLeast(
        u8,
        run.stdout,
        1,
        "userland verification failed (Z3: accepted, Lean obligation gate: rejected)",
    ));
    const hex_path = try std.fmt.allocPrint(allocator, "blocked/{s}.hex", .{stem});
    defer allocator.free(hex_path);
    try testing.expectError(
        error.FileNotFound,
        tmp.dir.access(std.testing.io, hex_path, .{}),
    );
    try expectNoFormalArtifactsInRepoRoot(allocator, stem);
}

test "repository tactic certifies a direct-root storage loop through the real CLI" {
    try expectRepositoryStorageTacticDeterministic(
        "tests/formal/direct_storage_loop_auto.ora",
        "direct_storage_loop_auto",
        1,
    );
}

test "repository tactic certifies an ordered storage read-write timeline" {
    try expectRepositoryStorageTacticDeterministic(
        "tests/formal/direct_storage_loop_read_auto.ora",
        "direct_storage_loop_read_auto",
        1,
    );
}

test "repository tactic certifies storage loop closure corpus through the real CLI" {
    try expectRepositoryStorageTacticDeterministic(
        "tests/formal/storage_loop_closure_auto.ora",
        "storage_loop_closure_auto",
        4,
    );
}

test "repository tactic certifies checked arithmetic in a storage timeline" {
    try expectRepositoryStorageTacticDeterministic(
        "tests/formal/direct_storage_loop_checked_auto.ora",
        "direct_storage_loop_checked_auto",
        1,
    );
}

test "repository tactic certifies checked increments after fixed-array storage writes" {
    try expectRepositoryStorageTacticDeterministic(
        "tests/formal/checked_storage_loop_auto.ora",
        "checked_storage_loop_auto",
        2,
    );
}

test "repository tactic certifies a typed-constant loop beside an unused aggregate parameter" {
    try expectRepositoryStorageTacticDeterministic(
        "tests/formal/const_initialized_storage_loop_auto.ora",
        "const_initialized_storage_loop_auto",
        1,
    );
}

test "storage read before checked increment requires an external certificate by name" {
    std.Io.Dir.cwd().access(std.testing.io, ORA_BINARY_REL, .{}) catch |err| switch (err) {
        error.FileNotFound => return error.SkipZigTest,
        else => return err,
    };

    const allocator = testing.allocator;
    var tmp = testing.tmpDir(.{});
    defer tmp.cleanup();
    const output = try pathFromTmpAlloc(allocator, tmp, "blocked");
    defer allocator.free(output);
    const run = try runProcessWithLeanPath(allocator, &.{
        ORA_BINARY_REL,
        "emit",
        "--emit=bytecode",
        "--out-dir",
        output,
        "--lean-proofs",
        "tests/formal/fail_storage_read_before_checked_loop_auto.ora",
    });
    defer allocator.free(run.stdout);
    defer allocator.free(run.stderr);
    try expectExited(run, 1);
    try testing.expect(std.mem.containsAtLeast(
        u8,
        run.stdout,
        1,
        "automatic_storage_timeline_requires_external_certificate",
    ));
    try testing.expectError(
        error.FileNotFound,
        tmp.dir.access(std.testing.io, "blocked/fail_storage_read_before_checked_loop_auto.hex", .{}),
    );
}

test "weak loop invariant cannot authorize a checked storage-loop increment" {
    std.Io.Dir.cwd().access(std.testing.io, ORA_BINARY_REL, .{}) catch |err| switch (err) {
        error.FileNotFound => return error.SkipZigTest,
        else => return err,
    };

    const allocator = testing.allocator;
    var tmp = testing.tmpDir(.{});
    defer tmp.cleanup();
    const output = try pathFromTmpAlloc(allocator, tmp, "blocked");
    defer allocator.free(output);
    const run = try runProcessWithLeanPath(allocator, &.{
        ORA_BINARY_REL,
        "emit",
        "--emit=bytecode",
        "--out-dir",
        output,
        "--lean-proofs",
        "tests/formal/fail_storage_loop_checked_add_unproved.ora",
    });
    defer allocator.free(run.stdout);
    defer allocator.free(run.stderr);
    try expectExited(run, 1);
    try testing.expect(std.mem.containsAtLeast(
        u8,
        run.stdout,
        1,
        "failed to prove loop body safety",
    ));
    try testing.expectError(
        error.FileNotFound,
        tmp.dir.access(std.testing.io, "blocked/fail_storage_loop_checked_add_unproved.hex", .{}),
    );
}

test "repository tactic certifies a bounded fixed-array storage loop through the real CLI" {
    try expectRepositoryStorageTacticDeterministic(
        "tests/formal/fixed_storage_loop_auto.ora",
        "fixed_storage_loop_auto",
        1,
    );
}

test "transient and opaque loop storage effects fail closed with named exclusions" {
    std.Io.Dir.cwd().access(std.testing.io, ORA_BINARY_REL, .{}) catch |err| switch (err) {
        error.FileNotFound => return error.SkipZigTest,
        else => return err,
    };

    const allocator = testing.allocator;
    var tmp = testing.tmpDir(.{});
    defer tmp.cleanup();
    const cases = [_]struct { source: []const u8, stem: []const u8 }{
        .{
            .source = "tests/formal/fail_transient_storage_loop_authority.ora",
            .stem = "fail_transient_storage_loop_authority",
        },
        .{
            .source = "tests/formal/fail_opaque_storage_loop_authority.ora",
            .stem = "fail_opaque_storage_loop_authority",
        },
    };
    for (cases, 0..) |case, index| {
        const relative = try std.fmt.allocPrint(allocator, "blocked-{d}", .{index});
        defer allocator.free(relative);
        const output = try pathFromTmpAlloc(allocator, tmp, relative);
        defer allocator.free(output);
        const run = try runProcessWithLeanPath(allocator, &.{
            ORA_BINARY_REL,
            "emit",
            "--emit=bytecode",
            "--out-dir",
            output,
            "--lean-proofs",
            case.source,
        });
        defer allocator.free(run.stdout);
        defer allocator.free(run.stderr);
        try expectExited(run, 1);
        try testing.expect(std.mem.containsAtLeast(
            u8,
            run.stdout,
            1,
            "excluded_by=loop_has_storage_write,loop_projection_effect_unsupported",
        ));
        const hex_path = try std.fmt.allocPrint(
            allocator,
            "{s}/{s}.hex",
            .{ relative, case.stem },
        );
        defer allocator.free(hex_path);
        try testing.expectError(error.FileNotFound, tmp.dir.access(std.testing.io, hex_path, .{}));
        try expectNoFormalArtifactsInRepoRoot(allocator, case.stem);
    }
}

test "repository tactic checks runtime-key mapping storage but artifact authority remains unreviewed" {
    try expectRepositoryLoopAuthorityBlocked(
        "tests/formal/mapping_storage_loop_auto.ora",
        "mapping_storage_loop_auto",
        .storage_projection_authority_unreviewed,
    );
}

test "repository tactic certifies branch-specific storage writes" {
    try expectRepositoryStorageTacticDeterministic(
        "tests/formal/branch_storage_loop_auto.ora",
        "branch_storage_loop_auto",
        1,
    );
}

test "repository tactic checks nested storage mutation but child frame authority remains unreviewed" {
    try expectRepositoryLoopAuthorityBlocked(
        "tests/formal/nested_storage_loop_auto.ora",
        "nested_storage_loop_auto",
        .storage_child_frame_authority_unreviewed,
    );
}

test "repository tactic certifies read-independent dynamic sequence timelines" {
    try expectRepositoryStorageTacticDeterministic(
        "tests/formal/dynamic_sequence_loop_bounds.ora",
        "dynamic_sequence_loop_bounds",
        3,
    );
}

test "indexed region reads project with canonical owners and independent exclusions" {
    var result = try collectPackageObligations(
        testing.allocator,
        "tests/formal/indexed_region_read_projection.ora",
    );
    defer result.deinit();

    const affine = loopSummaryByFunction(result.set, "affine") orelse return error.TestUnexpectedResult;
    try testing.expect(affine.projectionSupported());
    try testing.expect(!loopSummaryHasReason(affine, .loop_formula_unsupported));
    try testing.expectEqual(@as(usize, 1), affine.query_ids.induction.len);
    const affine_query = obligation.findById(result.set.queries, affine.query_ids.induction[0]) orelse
        return error.TestUnexpectedResult;
    try testing.expectEqual(obligation.ProofRequirement.lean_certificate, affine_query.proof_requirement);
    try testing.expect(affine_query.loop_vc_sha256 != null);
    try testing.expectEqual(
        @as(?obligation.ArtifactBlockReason, null),
        result.set.loopSummaryArtifactAuthorityBlockReason(affine),
    );
    var affine_read: ?obligation.LoopIndexedRegionReadEvent = null;
    var affine_read_position: ?usize = null;
    var affine_checkpoint_position: ?usize = null;
    var affine_value_consumed = false;
    for (affine.edges) |edge| {
        if (edge.target != .backedge) continue;
        for (edge.timeline, 0..) |event, position| switch (event.payload) {
            .indexed_region_read => |read| {
                affine_read = read;
                affine_read_position = position;
                for (edge.timeline[0..position], 0..) |prior, prior_position| {
                    if (std.mem.eql(u8, prior.structural_key, read.checkpoint_structural_key)) {
                        affine_checkpoint_position = prior_position;
                    }
                }
            },
            .storage_write => |write| if (affine_read) |read| {
                for (write.value.read_results) |read_result| {
                    affine_value_consumed = affine_value_consumed or
                        obligation.freeVarIdEql(read.result, read_result);
                }
            },
            else => {},
        };
    }
    const projected_read = affine_read orelse return error.TestUnexpectedResult;
    try testing.expectEqual(obligation.LoopIndexedRegionKind.calldata, projected_read.region);
    try testing.expectEqual(
        type_builtin.lookupBuiltinById(.u256).comptime_type_id,
        projected_read.element_ty.compiler_type_id,
    );
    try testing.expect(affine_checkpoint_position.? < affine_read_position.?);
    try testing.expect(affine_value_consumed);

    const alias = loopSummaryByFunction(result.set, "sameBackingAlias") orelse
        return error.TestUnexpectedResult;
    try testing.expect(alias.projectionSupported());
    var alias_reads: [2]obligation.LoopIndexedRegionReadEvent = undefined;
    var alias_read_count: usize = 0;
    for (alias.edges) |edge| for (edge.timeline) |event| switch (event.payload) {
        .indexed_region_read => |read| {
            if (alias_read_count >= alias_reads.len) return error.TestUnexpectedResult;
            alias_reads[alias_read_count] = read;
            alias_read_count += 1;
        },
        else => {},
    };
    try testing.expectEqual(alias_reads.len, alias_read_count);
    try testing.expect(obligation.freeVarIdEql(alias_reads[0].owner, alias_reads[1].owner));
    try testing.expectEqual(
        obligation.ArtifactBlockReason.loop_indexed_region_read_formula_unreviewed,
        result.set.loopSummaryArtifactAuthorityBlockReason(alias).?,
    );
    const first_index = switch (alias_reads[0].index.formula) {
        .term => |id| switch (result.set.terms[id]) {
            .variable => |variable| switch (variable) {
                .free => |free| free.id,
                .bound => return error.TestUnexpectedResult,
            },
            else => return error.TestUnexpectedResult,
        },
        .origin_value => return error.TestUnexpectedResult,
    };
    const second_index = switch (alias_reads[1].index.formula) {
        .term => |id| switch (result.set.terms[id]) {
            .variable => |variable| switch (variable) {
                .free => |free| free.id,
                .bound => return error.TestUnexpectedResult,
            },
            else => return error.TestUnexpectedResult,
        },
        .origin_value => return error.TestUnexpectedResult,
    };
    try testing.expect(obligation.freeVarIdEql(first_index, second_index));
    try testing.expect(!obligation.freeVarIdEql(alias_reads[0].result, alias_reads[1].result));

    const two_owners = loopSummaryByFunction(result.set, "twoOwners") orelse
        return error.TestUnexpectedResult;
    try testing.expect(loopSummaryHasReason(two_owners, .loop_indexed_region_owner_count_unsupported));
    const merged = loopSummaryByFunction(result.set, "mergedOwner") orelse
        return error.TestUnexpectedResult;
    try testing.expect(loopSummaryHasReason(merged, .loop_indexed_region_alias_unsupported));
    const write = loopSummaryByFunction(result.set, "writeThroughAlias") orelse
        return error.TestUnexpectedResult;
    try testing.expect(loopSummaryHasReason(write, .loop_indexed_region_write_unreviewed));
    const call_owner = loopSummaryByFunction(result.set, "callProducedOwner") orelse
        return error.TestUnexpectedResult;
    try testing.expect(loopSummaryHasReason(call_owner, .loop_indexed_region_owner_unsupported));
    const storage_owner = loopSummaryByFunction(result.set, "storageLoadedOwner") orelse
        return error.TestUnexpectedResult;
    try testing.expect(loopSummaryHasReason(storage_owner, .loop_indexed_region_owner_unsupported));
    const call_body = loopSummaryByFunction(result.set, "callInBody") orelse
        return error.TestUnexpectedResult;
    try testing.expect(loopSummaryHasReason(call_body, .loop_has_call));
}

test "indexed region affine loop certifies automatically through the real CLI" {
    try expectRepositoryStorageTacticDeterministic(
        "tests/formal/indexed_region_read_affine.ora",
        "indexed_region_read_affine",
        1,
    );
}

test "indexed region read identities in loop formulas remain blocked by name" {
    try expectRepositoryLoopProofConstructionBlocked(
        "tests/formal/indexed_region_read_formula_unreviewed.ora",
        "indexed_region_read_formula_unreviewed",
        1,
        "loop_indexed_region_read_formula_unreviewed",
    );
}

test "formal direct-root storage loop requires compiler-owned storage type metadata" {
    var compilation = try compilePackage("tests/formal/storage_loop_projection.ora");
    defer compilation.deinit();
    const hir_result = try compilation.db.lowerToHir(compilation.root_module_id);
    try testing.expect(hir_result.diagnostics.isEmpty());
    const global = findFirstOpByName(hir_result.module.raw_module, "ora.global") orelse
        return error.TestUnexpectedResult;
    try testing.expect(mlir.mlirOperationRemoveAttributeByName(
        global,
        strRef("ora.compiler_type_id"),
    ));

    var result = try obligation_from_mlir.collect(testing.allocator, hir_result.module.raw_module, .{});
    defer result.deinit();
    for (result.set.loop_summaries) |summary| {
        if (summary.owner != .statement or
            !std.mem.eql(u8, summary.owner.statement.function_name, "readThenWrite")) continue;
        try testing.expect(loopSummaryHasReason(summary, .loop_storage_type_invalid));
        try testing.expect(!summary.projectionSupported());
        return;
    }
    return error.TestUnexpectedResult;
}

test "formal direct-root storage loop requires a complete compiler write inventory" {
    var compilation = try compilePackage("tests/formal/storage_loop_projection.ora");
    defer compilation.deinit();
    const hir_result = try compilation.db.lowerToHir(compilation.root_module_id);
    try testing.expect(hir_result.diagnostics.isEmpty());
    const function = findFunctionByName(hir_result.module.raw_module, "readThenWrite") orelse
        return error.TestUnexpectedResult;
    try testing.expect(mlir.mlirOperationRemoveAttributeByName(
        function,
        strRef("ora.write_slots_complete"),
    ));

    var result = try obligation_from_mlir.collect(testing.allocator, hir_result.module.raw_module, .{});
    defer result.deinit();
    for (result.set.loop_summaries) |summary| {
        if (summary.owner != .statement or
            !std.mem.eql(u8, summary.owner.statement.function_name, "readThenWrite")) continue;
        try testing.expect(loopSummaryHasReason(summary, .loop_has_storage_write));
        try testing.expect(!summary.projectionSupported());
        return;
    }
    return error.TestUnexpectedResult;
}

test "formal direct-root projection retains uncovered modifies as an effect-frame obligation" {
    var compilation = try compilePackage("tests/formal/storage_loop_projection.ora");
    defer compilation.deinit();
    const hir_result = try compilation.db.lowerToHir(compilation.root_module_id);
    try testing.expect(hir_result.diagnostics.isEmpty());
    const function = findFunctionByName(hir_result.module.raw_module, "readThenWrite") orelse
        return error.TestUnexpectedResult;
    mlir.oraOperationSetAttributeByName(
        function,
        strRef("ora.modifies_slots"),
        mlir.oraArrayAttrCreate(mlir.mlirOperationGetContext(function), 0, null),
    );

    var result = try obligation_from_mlir.collect(testing.allocator, hir_result.module.raw_module, .{});
    defer result.deinit();
    var projected = false;
    for (result.set.loop_summaries) |summary| {
        if (summary.owner == .statement and
            std.mem.eql(u8, summary.owner.statement.function_name, "readThenWrite"))
        {
            projected = summary.projectionSupported();
        }
    }
    try testing.expect(projected);
    for (result.set.obligations) |item| {
        if (item.owner != .function or
            !std.mem.eql(u8, item.owner.function.name, "readThenWrite") or
            item.kind != .effect_frame or
            item.kind.effect_frame.relation != .write_covered_by_modifies) continue;
        try testing.expectEqual(@as(usize, 0), item.kind.effect_frame.declared.len);
        try testing.expectEqual(@as(usize, 1), item.kind.effect_frame.actual.len);
        try testing.expectEqualStrings("counter", item.kind.effect_frame.actual[0].root);
        return;
    }
    return error.TestUnexpectedResult;
}

test "nested loop child results are checked in post-dependency safety state" {
    var result = try collectPackageObligations(
        testing.allocator,
        "tests/formal/nested_loop_dependency_safety.ora",
    );
    defer result.deinit();

    var parent: ?obligation.LoopSummaryRow = null;
    for (result.set.loop_summaries) |row| {
        for (row.edges) |edge| {
            if (loopEdgeDependencyCount(edge) != 0) parent = row;
        }
    }
    const summary = parent orelse return error.TestUnexpectedResult;
    try testing.expect(summary.projectionSupported());
    var staged_edge: ?obligation.LoopEdgeRow = null;
    for (summary.edges) |edge| {
        if (loopEdgeDependencyCount(edge) != 0) staged_edge = edge;
    }
    const edge = staged_edge orelse return error.TestUnexpectedResult;
    try testing.expectEqual(@as(usize, 1), loopEdgeDependencyCount(edge));
    try testing.expectEqual(@as(usize, 2), loopEdgeSafetyCheckpointCount(edge));
    try expectLoopEdgeSafetyStages(edge, &.{ 1, 1 });
    for (edge.timeline) |event| switch (event.payload) {
        .safety_checkpoint => |checkpoint| try testing.expectEqual(@as(usize, 1), checkpoint.formulas.len),
        else => {},
    };
    try testing.expectEqual(@as(usize, 1), summary.query_ids.induction.len);
    const query = obligation.findById(result.set.queries, summary.query_ids.induction[0]) orelse
        return error.TestUnexpectedResult;
    switch (obligation_to_lean.querySemanticSupport(result.set, query)) {
        .supported => {},
        .unsupported => return error.TestUnexpectedResult,
    }

    std.Io.Dir.cwd().access(std.testing.io, ORA_BINARY_REL, .{}) catch |err| switch (err) {
        error.FileNotFound => return error.SkipZigTest,
        else => return err,
    };
    var tmp = testing.tmpDir(.{});
    defer tmp.cleanup();
    const output_path = try pathFromTmpAlloc(testing.allocator, tmp, "dependency-safety");
    defer testing.allocator.free(output_path);
    const cli = try runProcessWithLeanPath(testing.allocator, &.{
        ORA_BINARY_REL,
        "emit",
        "--emit=smt-report,bytecode",
        "--out-dir",
        output_path,
        "--lean-proofs",
        "tests/formal/nested_loop_dependency_safety.ora",
    });
    defer testing.allocator.free(cli.stdout);
    defer testing.allocator.free(cli.stderr);
    try expectExited(cli, 0);
    try testing.expect(std.mem.containsAtLeast(
        u8,
        cli.stdout,
        1,
        "total=2, certificate-required=2, excluded=0",
    ));
    try testing.expect(std.mem.containsAtLeast(
        u8,
        cli.stdout,
        1,
        "Lean loop proof gate accepted:\n  kernel-checked certificates: 2",
    ));
    try tmp.dir.access(
        std.testing.io,
        "dependency-safety/nested_loop_dependency_safety.hex",
        .{},
    );
}

test "unsafe post-dependency arithmetic is disproved before artifact emission" {
    var result = try collectPackageObligations(
        testing.allocator,
        "tests/formal/nested_loop_dependency_safety_unsafe.ora",
    );
    defer result.deinit();

    var parent: ?obligation.LoopSummaryRow = null;
    var staged_edge: ?obligation.LoopEdgeRow = null;
    for (result.set.loop_summaries) |row| {
        for (row.edges) |edge| {
            if (loopEdgeDependencyCount(edge) == 0) continue;
            parent = row;
            staged_edge = edge;
        }
    }
    const summary = parent orelse return error.TestUnexpectedResult;
    const edge = staged_edge orelse return error.TestUnexpectedResult;
    try testing.expect(summary.projectionSupported());
    try testing.expect(loopEdgeSafetyCheckpointCount(edge) != 0);
    try expectLoopEdgeSafetyStages(edge, &.{ 1, 1 });

    var context = try z3_verification.Z3Context.init(testing.allocator);
    defer context.deinit();
    var adapter = obligation_to_z3.Adapter.init(&context, testing.allocator, result.set);
    var safety_query_found = false;
    for (result.set.queries) |query| {
        if (query.loop_summary_id != summary.id or
            query.loop_edge_id != edge.id or
            query.kind != .loop_body_safety or
            query.backend != .z3 or
            query.loop_vc_sha256 == null) continue;
        const checked = try adapter.checkLoopQueryForRow(query);
        try testing.expectEqual(obligation_to_z3.CheckStatus.disproved, checked.status);
        safety_query_found = true;
    }
    try testing.expect(safety_query_found);
}

test "safety checkpoints preserve before between and after dependency order" {
    var result = try collectPackageObligations(
        testing.allocator,
        "tests/formal/nested_loop_safety_checkpoints.ora",
    );
    defer result.deinit();

    var parent: ?obligation.LoopSummaryRow = null;
    var staged_edge: ?obligation.LoopEdgeRow = null;
    for (result.set.loop_summaries) |row| {
        for (row.edges) |edge| {
            if (loopEdgeDependencyCount(edge) != 2) continue;
            parent = row;
            staged_edge = edge;
        }
    }
    const summary = parent orelse return error.TestUnexpectedResult;
    const edge = staged_edge orelse return error.TestUnexpectedResult;
    try testing.expect(summary.projectionSupported());
    try testing.expectEqual(@as(usize, 2), loopEdgeDependencyCount(edge));
    try testing.expectEqual(@as(usize, 4), loopEdgeSafetyCheckpointCount(edge));
    const expected_stages = [_]u32{ 0, 1, 2, 2 };
    try expectLoopEdgeSafetyStages(edge, &expected_stages);

    var context = try z3_verification.Z3Context.init(testing.allocator);
    defer context.deinit();
    var adapter = obligation_to_z3.Adapter.init(&context, testing.allocator, result.set);
    var safety_query_found = false;
    for (result.set.queries) |query| {
        if (query.loop_summary_id != summary.id or
            query.loop_edge_id != edge.id or
            query.kind != .loop_body_safety or
            query.backend != .z3 or
            query.loop_vc_sha256 == null) continue;
        const checked = try adapter.checkLoopQueryForRow(query);
        try testing.expectEqual(obligation_to_z3.CheckStatus.proved, checked.status);
        safety_query_found = true;
    }
    try testing.expect(safety_query_found);

    var source = std.Io.Writer.Allocating.init(testing.allocator);
    defer source.deinit();
    try obligation_to_lean.writeModule(&source.writer, result.set, .{
        .namespace = "Ora.Generated.NestedLoopSafetyCheckpoints",
        .proof_surface = true,
    });
    var tmp = testing.tmpDir(.{});
    defer tmp.cleanup();
    const path = try pathFromTmpAlloc(testing.allocator, tmp, "NestedLoopSafetyCheckpoints.lean");
    defer testing.allocator.free(path);
    try std.Io.Dir.cwd().writeFile(std.testing.io, .{
        .sub_path = path,
        .data = source.written(),
    });
    const parsed = try runLeanFileForTest(testing.allocator, path);
    defer testing.allocator.free(parsed.stdout);
    defer testing.allocator.free(parsed.stderr);
    try expectExited(parsed, 0);
}

test "generated nested loop relational obligation module parses" {
    var result = try collectPackageObligations(
        testing.allocator,
        "tests/formal/nested_loop_relational.ora",
    );
    defer result.deinit();

    var source = std.Io.Writer.Allocating.init(testing.allocator);
    defer source.deinit();
    try obligation_to_lean.writeModule(&source.writer, result.set, .{
        .namespace = "Ora.Generated.NestedLoopRelationalParse",
        .proof_surface = true,
    });

    var tmp = testing.tmpDir(.{});
    defer tmp.cleanup();
    const path = try pathFromTmpAlloc(testing.allocator, tmp, "NestedLoopRelational.lean");
    defer testing.allocator.free(path);
    try std.Io.Dir.cwd().writeFile(std.testing.io, .{ .sub_path = path, .data = source.written() });

    const lean_result = try runLeanFileForTest(testing.allocator, path);
    defer testing.allocator.free(lean_result.stdout);
    defer testing.allocator.free(lean_result.stderr);
    try expectExited(lean_result, 0);
}

test "nested loop relation explicitly preserves an unchanged child scalar" {
    const allocator = testing.allocator;
    const source_path = "tests/formal/nested_loop_preserves_scalar.ora";
    var formal_result = try collectPackageObligations(allocator, source_path);
    defer formal_result.deinit();

    var child: ?obligation.LoopSummaryRow = null;
    for (formal_result.set.loop_summaries) |summary| {
        var has_dependency = false;
        for (summary.edges) |edge| {
            if (loopEdgeDependencyCount(edge) != 0) has_dependency = true;
        }
        if (!has_dependency and summary.variables.len == 2) child = summary;
    }
    const child_summary = child orelse return error.TestUnexpectedResult;
    try testing.expect(child_summary.projectionSupported());

    const backedge = for (child_summary.edges) |edge| {
        if (edge.target == .backedge) break edge;
    } else return error.TestUnexpectedResult;
    try testing.expectEqual(child_summary.variables.len, backedge.next_state.len);
    const preserved = backedge.next_state[1];
    try testing.expectEqual(@as(u32, 1), preserved.variable_index);
    try testing.expect(obligation.freeVarIdEql(
        child_summary.variables[1].id orelse return error.TestUnexpectedResult,
        preserved.target orelse return error.TestUnexpectedResult,
    ));
    const preserved_term_id = switch (preserved.value) {
        .term => |id| id,
        .origin_value => return error.TestUnexpectedResult,
    };
    if (preserved_term_id >= formal_result.set.terms.len) return error.TestUnexpectedResult;
    const preserved_value_id = switch (formal_result.set.terms[preserved_term_id]) {
        .variable => |variable| switch (variable) {
            .free => |free| free.id,
            .bound => return error.TestUnexpectedResult,
        },
        else => return error.TestUnexpectedResult,
    };
    try testing.expect(obligation.freeVarIdEql(
        preserved.target.?,
        preserved_value_id,
    ));

    std.Io.Dir.cwd().access(std.testing.io, ORA_BINARY_REL, .{}) catch |err| switch (err) {
        error.FileNotFound => return error.SkipZigTest,
        else => return err,
    };
    var tmp = testing.tmpDir(.{});
    defer tmp.cleanup();
    const output_path = try pathFromTmpAlloc(allocator, tmp, "nested-preserved");
    defer allocator.free(output_path);
    const result = try runProcessWithLeanPath(allocator, &.{
        ORA_BINARY_REL,
        "emit",
        "--emit=smt-report,bytecode",
        "--out-dir",
        output_path,
        "--lean-proofs",
        source_path,
    });
    defer allocator.free(result.stdout);
    defer allocator.free(result.stderr);
    try expectExited(result, 0);
    try testing.expect(std.mem.containsAtLeast(
        u8,
        result.stdout,
        1,
        "total=2, certificate-required=2, excluded=0",
    ));
    try testing.expect(std.mem.containsAtLeast(
        u8,
        result.stdout,
        1,
        "Lean loop proof gate accepted:\n  kernel-checked certificates: 2",
    ));
    try tmp.dir.access(
        std.testing.io,
        "nested-preserved/nested_loop_preserves_scalar.hex",
        .{},
    );
}

test "nested loop relational certificates authorize the real CLI" {
    std.Io.Dir.cwd().access(std.testing.io, ORA_BINARY_REL, .{}) catch |err| switch (err) {
        error.FileNotFound => return error.SkipZigTest,
        else => return err,
    };

    const allocator = testing.allocator;
    const source_path = "tests/formal/nested_loop_relational.ora";
    var tmp = testing.tmpDir(.{});
    defer tmp.cleanup();
    const output_path = try pathFromTmpAlloc(allocator, tmp, "nested-accepted");
    defer allocator.free(output_path);

    const result = try runProcessWithLeanPath(allocator, &.{
        ORA_BINARY_REL,
        "emit",
        "--emit=smt-report,sir-text,bytecode",
        "--out-dir",
        output_path,
        "--lean-proofs",
        source_path,
    });
    defer allocator.free(result.stdout);
    defer allocator.free(result.stderr);
    try expectExited(result, 0);
    try testing.expect(std.mem.containsAtLeast(
        u8,
        result.stdout,
        1,
        "total=2, certificate-required=2, excluded=0",
    ));
    try testing.expect(std.mem.containsAtLeast(
        u8,
        result.stdout,
        1,
        "Lean loop proof gate accepted:\n  kernel-checked certificates: 2",
    ));

    const certificate_path = try pathFromTmpAlloc(
        allocator,
        tmp,
        "nested-accepted/nested_loop_relational.lean.proof.json",
    );
    defer allocator.free(certificate_path);
    const certificate = try readFileAllocForTest(allocator, certificate_path);
    defer allocator.free(certificate);
    try testing.expect(std.mem.containsAtLeast(u8, certificate, 1, "\"proof_count\": 2"));
    try testing.expect(std.mem.containsAtLeast(u8, certificate, 2, "\"loop_vc_sha256\": \""));
    try testing.expect(!std.mem.containsAtLeast(u8, certificate, 1, "sorryAx"));

    const loop_report_path = try pathFromTmpAlloc(
        allocator,
        tmp,
        "nested-accepted/nested_loop_relational.formal.loops.json",
    );
    defer allocator.free(loop_report_path);
    const loop_report_source = try readFileAllocForTest(allocator, loop_report_path);
    defer allocator.free(loop_report_source);
    var loop_report = try std.json.parseFromSlice(
        std.json.Value,
        allocator,
        loop_report_source,
        .{},
    );
    defer loop_report.deinit();
    const loops = (loop_report.value.object.get("loops") orelse
        return error.TestUnexpectedResult).array.items;
    try testing.expectEqual(@as(usize, 2), loops.len);

    var required_queries = std.AutoHashMap(i64, i64).init(allocator);
    defer required_queries.deinit();
    for (loops) |loop| {
        const object = loop.object;
        try testing.expectEqualStrings(
            "complete",
            (object.get("projection_completeness") orelse
                return error.TestUnexpectedResult).string,
        );
        try testing.expect((object.get("certificate_required") orelse
            return error.TestUnexpectedResult).bool);
        const summary_id = (object.get("id") orelse
            return error.TestUnexpectedResult).integer;
        const query_id = (object.get("induction_query_id") orelse
            return error.TestUnexpectedResult).integer;
        try required_queries.put(summary_id, query_id);
    }

    var proof_certificate = try std.json.parseFromSlice(
        std.json.Value,
        allocator,
        certificate,
        .{},
    );
    defer proof_certificate.deinit();
    try testing.expectEqualStrings(
        "checked",
        (proof_certificate.value.object.get("status") orelse
            return error.TestUnexpectedResult).string,
    );
    const proofs = (proof_certificate.value.object.get("proofs") orelse
        return error.TestUnexpectedResult).array.items;
    try testing.expectEqual(loops.len, proofs.len);
    for (proofs) |proof| {
        const object = proof.object;
        try testing.expectEqualStrings(
            "loop_induction_and_nonvacuity",
            (object.get("authorization_claim") orelse
                return error.TestUnexpectedResult).string,
        );
        try testing.expectEqualStrings(
            "checked",
            (object.get("lean_status") orelse
                return error.TestUnexpectedResult).string,
        );
        try testing.expectEqualStrings(
            "proved",
            (object.get("anti_vacuity_status") orelse
                return error.TestUnexpectedResult).string,
        );
        const summary_id = (object.get("loop_summary_id") orelse
            return error.TestUnexpectedResult).integer;
        const expected_query_id = required_queries.get(summary_id) orelse
            return error.TestUnexpectedResult;
        try testing.expectEqual(
            expected_query_id,
            (object.get("target_query_id") orelse
                return error.TestUnexpectedResult).integer,
        );
    }

    const smt_report_path = try pathFromTmpAlloc(
        allocator,
        tmp,
        "nested-accepted/nested_loop_relational.smt.report.json",
    );
    defer allocator.free(smt_report_path);
    const smt_report_source = try readFileAllocForTest(allocator, smt_report_path);
    defer allocator.free(smt_report_source);
    var smt_report = try std.json.parseFromSlice(
        std.json.Value,
        allocator,
        smt_report_source,
        .{},
    );
    defer smt_report.deinit();
    const smt_root = smt_report.value.object;
    const verification = (smt_root.get("verification") orelse
        return error.TestUnexpectedResult).object;
    try testing.expect((verification.get("success") orelse
        return error.TestUnexpectedResult).bool);
    const smt_summary = (smt_root.get("summary") orelse
        return error.TestUnexpectedResult).object;
    try testing.expectEqual(
        @as(i64, 0),
        (smt_summary.get("vacuous") orelse return error.TestUnexpectedResult).integer,
    );
    try testing.expectEqual(
        @as(i64, 0),
        (smt_summary.get("vacuity_unknown") orelse return error.TestUnexpectedResult).integer,
    );

    const index_path = try pathFromTmpAlloc(
        allocator,
        tmp,
        "nested-accepted/nested_loop_relational.formal.artifacts.json",
    );
    defer allocator.free(index_path);
    const artifact_index = try readFileAllocForTest(allocator, index_path);
    defer allocator.free(artifact_index);
    var parsed_index = try std.json.parseFromSlice(
        std.json.Value,
        allocator,
        artifact_index,
        .{},
    );
    defer parsed_index.deinit();
    const index_root = parsed_index.value.object;
    try testing.expectEqualStrings(
        "allowed",
        (index_root.get("final_artifact_authorization") orelse
            return error.TestUnexpectedResult).string,
    );
    var z3_accepted = false;
    var lean_accepted = false;
    for ((index_root.get("gates") orelse return error.TestUnexpectedResult).array.items) |gate| {
        const object = gate.object;
        const id = (object.get("id") orelse return error.TestUnexpectedResult).string;
        const accepted = std.mem.eql(
            u8,
            (object.get("status") orelse return error.TestUnexpectedResult).string,
            "accepted",
        );
        if (std.mem.eql(u8, id, "z3_userland")) z3_accepted = accepted;
        if (std.mem.eql(u8, id, "lean_userland")) lean_accepted = accepted;
    }
    try testing.expect(z3_accepted);
    try testing.expect(lean_accepted);

    var saw_smt_report = false;
    var saw_loop_report = false;
    var saw_lean_certificate = false;
    for ((index_root.get("artifacts") orelse return error.TestUnexpectedResult).array.items) |artifact| {
        const object = artifact.object;
        const id = (object.get("id") orelse return error.TestUnexpectedResult).string;
        if (std.mem.eql(u8, id, "smt_report_json")) saw_smt_report = true;
        if (std.mem.eql(u8, id, "loop_induction_report")) saw_loop_report = true;
        if (std.mem.eql(u8, id, "lean_proof_certificate")) {
            saw_lean_certificate = true;
            const digest = (object.get("sha256") orelse
                return error.TestUnexpectedResult).string;
            try testing.expectEqual(@as(usize, 64), digest.len);
        }
    }
    try testing.expect(saw_smt_report);
    try testing.expect(saw_loop_report);
    try testing.expect(saw_lean_certificate);

    try tmp.dir.access(std.testing.io, "nested-accepted/nested_loop_relational.hex", .{});
}

test "multiple nested loops on distinct outer paths authorize the real CLI" {
    std.Io.Dir.cwd().access(std.testing.io, ORA_BINARY_REL, .{}) catch |err| switch (err) {
        error.FileNotFound => return error.SkipZigTest,
        else => return err,
    };

    const allocator = testing.allocator;
    const source_path = "tests/formal/nested_loop_multiple_paths.ora";
    var tmp = testing.tmpDir(.{});
    defer tmp.cleanup();
    const output_path = try pathFromTmpAlloc(allocator, tmp, "nested-paths-accepted");
    defer allocator.free(output_path);

    const result = try runProcessWithLeanPath(allocator, &.{
        ORA_BINARY_REL,
        "emit",
        "--emit=smt-report,sir-text,bytecode",
        "--out-dir",
        output_path,
        "--lean-proofs",
        source_path,
    });
    defer allocator.free(result.stdout);
    defer allocator.free(result.stderr);
    try expectExited(result, 0);
    try testing.expect(std.mem.containsAtLeast(
        u8,
        result.stdout,
        1,
        "total=3, certificate-required=3, excluded=0",
    ));
    try testing.expect(std.mem.containsAtLeast(
        u8,
        result.stdout,
        1,
        "Lean loop proof gate accepted:\n  kernel-checked certificates: 3",
    ));

    const certificate_path = try pathFromTmpAlloc(
        allocator,
        tmp,
        "nested-paths-accepted/nested_loop_multiple_paths.lean.proof.json",
    );
    defer allocator.free(certificate_path);
    const certificate = try readFileAllocForTest(allocator, certificate_path);
    defer allocator.free(certificate);
    try testing.expect(std.mem.containsAtLeast(u8, certificate, 1, "\"proof_count\": 3"));
    try testing.expect(std.mem.containsAtLeast(u8, certificate, 3, "\"loop_vc_sha256\": \""));
    try testing.expect(!std.mem.containsAtLeast(u8, certificate, 1, "sorryAx"));

    const index_path = try pathFromTmpAlloc(
        allocator,
        tmp,
        "nested-paths-accepted/nested_loop_multiple_paths.formal.artifacts.json",
    );
    defer allocator.free(index_path);
    const artifact_index = try readFileAllocForTest(allocator, index_path);
    defer allocator.free(artifact_index);
    try testing.expect(std.mem.containsAtLeast(
        u8,
        artifact_index,
        1,
        "\"id\": \"lean_userland\", \"domain\": \"userland\", \"phase\": \"proof_artifact\", \"status\": \"accepted\"",
    ));

    try tmp.dir.access(
        std.testing.io,
        "nested-paths-accepted/nested_loop_multiple_paths.hex",
        .{},
    );
}

test "nested loop formal artifacts are byte-identical across real CLI runs" {
    std.Io.Dir.cwd().access(std.testing.io, ORA_BINARY_REL, .{}) catch |err| switch (err) {
        error.FileNotFound => return error.SkipZigTest,
        else => return err,
    };

    const allocator = testing.allocator;
    const source_path = "tests/formal/nested_loop_relational.ora";
    const stem = "nested_loop_relational";
    var tmp = testing.tmpDir(.{});
    defer tmp.cleanup();
    const output_paths = [_][]const u8{ "determinism-first", "determinism-second" };

    for (output_paths) |relative_output| {
        const output_path = try pathFromTmpAlloc(allocator, tmp, relative_output);
        defer allocator.free(output_path);
        const result = try runProcessWithLeanPath(allocator, &.{
            ORA_BINARY_REL,
            "emit",
            "--emit=smt-report,bytecode",
            "--out-dir",
            output_path,
            "--lean-proofs",
            source_path,
        });
        defer allocator.free(result.stdout);
        defer allocator.free(result.stderr);
        try expectExited(result, 0);
    }

    const compared_suffixes = [_][]const u8{
        ".formal.loops.json",
        ".lean.proof.json",
        ".formal.artifacts.json",
    };
    for (compared_suffixes) |suffix| {
        const relative_first = try std.fmt.allocPrint(
            allocator,
            "{s}/{s}{s}",
            .{ output_paths[0], stem, suffix },
        );
        defer allocator.free(relative_first);
        const first_path = try pathFromTmpAlloc(allocator, tmp, relative_first);
        defer allocator.free(first_path);
        const first = try readFileAllocForTest(allocator, first_path);
        defer allocator.free(first);

        const relative_second = try std.fmt.allocPrint(
            allocator,
            "{s}/{s}{s}",
            .{ output_paths[1], stem, suffix },
        );
        defer allocator.free(relative_second);
        const second_path = try pathFromTmpAlloc(allocator, tmp, relative_second);
        defer allocator.free(second_path);
        const second = try readFileAllocForTest(allocator, second_path);
        defer allocator.free(second);

        try testing.expectEqualSlices(u8, first, second);
    }
    try expectNoFormalArtifactsInRepoRoot(allocator, stem);
}

test "nested loop artifact authority requires the exact child certificate" {
    var result = try collectPackageObligations(
        testing.allocator,
        "tests/formal/nested_loop_relational.ora",
    );
    defer result.deinit();

    var parent: ?obligation.LoopSummaryRow = null;
    var child: ?obligation.LoopSummaryRow = null;
    for (result.set.loop_summaries) |row| {
        var has_dependency = false;
        for (row.edges) |edge| {
            if (loopEdgeDependencyCount(edge) != 0) has_dependency = true;
        }
        if (has_dependency) parent = row else child = row;
    }
    const parent_summary = parent orelse return error.TestUnexpectedResult;
    const child_summary = child orelse return error.TestUnexpectedResult;
    const parent_target = obligation.findById(
        result.set.queries,
        parent_summary.query_ids.induction[0],
    ) orelse return error.TestUnexpectedResult;
    const child_target = obligation.findById(
        result.set.queries,
        child_summary.query_ids.induction[0],
    ) orelse return error.TestUnexpectedResult;

    var decided_queries = try testing.allocator.alloc(
        obligation.VerificationQuery,
        result.set.queries.len + 2,
    );
    defer testing.allocator.free(decided_queries);
    @memcpy(decided_queries[0..result.set.queries.len], result.set.queries);
    for (decided_queries[0..result.set.queries.len]) |*query| {
        query.obligation_ids = &.{};
        if (query.backend == .z3) {
            query.result = .{ .status = if (query.kind == .base) .sat else .unsat };
        } else if (query.kind != .loop_induction) {
            query.artifact_policy = .diagnostic_only;
        }
    }

    const artifacts = [_]obligation.ProofArtifact{
        .{
            .id = 100_001,
            .owner = parent_target.owner,
            .source = parent_target.source,
            .module_name = "Ora.Tests.NestedLoop",
            .theorem_name = "parent",
        },
        .{
            .id = 100_002,
            .owner = child_target.owner,
            .source = child_target.source,
            .module_name = "Ora.Tests.NestedLoop",
            .theorem_name = "child",
        },
    };
    decided_queries[result.set.queries.len] = .{
        .id = 100_003,
        .owner = parent_target.owner,
        .source = parent_target.source,
        .phase = .report,
        .origin = parent_target.origin,
        .backend = .lean,
        .kind = .loop_induction,
        .logical_role = parent_target.logical_role,
        .assumption_ids = parent_target.assumption_ids,
        .proof_artifact_id = artifacts[0].id,
        .discharges_query_id = parent_target.id,
        .loop_summary_id = parent_summary.id,
        .loop_vc_sha256 = parent_target.loop_vc_sha256,
        .result = .{ .status = .proved },
    };
    decided_queries[result.set.queries.len + 1] = .{
        .id = 100_004,
        .owner = child_target.owner,
        .source = child_target.source,
        .phase = .report,
        .origin = child_target.origin,
        .backend = .lean,
        .kind = .loop_induction,
        .logical_role = child_target.logical_role,
        .assumption_ids = child_target.assumption_ids,
        .proof_artifact_id = artifacts[1].id,
        .discharges_query_id = child_target.id,
        .loop_summary_id = child_summary.id,
        .loop_vc_sha256 = child_target.loop_vc_sha256,
        .result = .{ .status = .proved },
    };

    var decided_set = result.set;
    decided_set.obligations = &.{};
    decided_set.queries = decided_queries;
    decided_set.proof_artifacts = &artifacts;
    try decided_set.validateTermReferences();
    try decided_set.validateIdReferences();
    try testing.expectEqual(
        obligation.ArtifactDecision.allowed,
        decided_set.artifactDecision(),
    );

    decided_queries[result.set.queries.len + 1].logical_role = null;
    try testing.expectEqual(obligation.ArtifactBlockReason.lean_required_failure, decided_set.artifactDecision().blocked);
    decided_queries[result.set.queries.len + 1].logical_role = child_target.logical_role;

    decided_set.queries = decided_queries[0 .. decided_queries.len - 1];
    try testing.expectEqual(
        obligation.ArtifactBlockReason.lean_required_failure,
        decided_set.artifactDecision().blocked,
    );

    const parent_edge_index: usize = for (parent_summary.edges, 0..) |edge, index| {
        if (loopEdgeDependencyCount(edge) != 0) break index;
    } else return error.TestUnexpectedResult;
    var changed_summaries = try testing.allocator.dupe(
        obligation.LoopSummaryRow,
        result.set.loop_summaries,
    );
    defer testing.allocator.free(changed_summaries);
    const parent_index: usize = for (changed_summaries, 0..) |row, index| {
        if (row.id == parent_summary.id) break index;
    } else return error.TestUnexpectedResult;
    var changed_edges = try testing.allocator.dupe(
        obligation.LoopEdgeRow,
        parent_summary.edges,
    );
    defer testing.allocator.free(changed_edges);
    var changed_timeline = try testing.allocator.dupe(
        obligation.LoopTimelineEvent,
        changed_edges[parent_edge_index].timeline,
    );
    defer testing.allocator.free(changed_timeline);
    const dependency_event_index: usize = for (changed_timeline, 0..) |event, index| {
        if (event.payload == .summary_dependency) break index;
    } else return error.TestUnexpectedResult;
    const original_dependency = changed_timeline[dependency_event_index].payload.summary_dependency;
    changed_timeline[dependency_event_index].payload.summary_dependency.loop_vc_sha256[0] ^= 1;
    changed_edges[parent_edge_index].timeline = changed_timeline;
    changed_summaries[parent_index].edges = changed_edges;
    decided_set.loop_summaries = changed_summaries;
    decided_set.queries = decided_queries;
    try testing.expectEqual(
        obligation.ArtifactBlockReason.invalid_dependency,
        decided_set.artifactDecision().blocked,
    );

    changed_timeline[dependency_event_index].payload.summary_dependency = original_dependency;
    changed_timeline[dependency_event_index].payload.summary_dependency.summary_id = parent_summary.id;
    try testing.expectEqual(
        obligation.ArtifactBlockReason.invalid_dependency,
        decided_set.artifactDecision().blocked,
    );

    changed_timeline[dependency_event_index].payload.summary_dependency = original_dependency;
    changed_timeline[dependency_event_index].payload.summary_dependency.summary_id = std.math.maxInt(obligation.Id);
    try testing.expectEqual(
        obligation.ArtifactBlockReason.invalid_dependency,
        decided_set.artifactDecision().blocked,
    );

    changed_timeline[dependency_event_index].payload.summary_dependency = original_dependency;
    changed_timeline[dependency_event_index].payload.summary_dependency.target = .backedge;
    try testing.expectEqual(
        obligation.ArtifactBlockReason.invalid_dependency,
        decided_set.artifactDecision().blocked,
    );
}

test "unsupported loop effects fail closed through the real CLI" {
    std.Io.Dir.cwd().access(std.testing.io, ORA_BINARY_REL, .{}) catch |err| switch (err) {
        error.FileNotFound => return error.SkipZigTest,
        else => return err,
    };

    const allocator = testing.allocator;
    var tmp = testing.tmpDir(.{});
    defer tmp.cleanup();
    const source =
        \\resource TokenUnit = u256;
        \\
        \\contract LoopUnsupportedEffects {
        \\    storage var total: u256 = 0;
        \\    storage var left: Resource<TokenUnit>;
        \\    storage var right: Resource<TokenUnit>;
        \\
        \\    fn advance(value: u256) -> u256 {
        \\        return value + 1;
        \\    }
        \\
        \\    pub fn storageLoop() {
        \\        var i: u256 = 0;
        \\        while (i < 1)
        \\            invariant i <= 1
        \\        {
        \\            total = total + 1;
        \\            i = i + 1;
        \\        }
        \\    }
        \\
        \\    pub fn callLoop() -> u256 {
        \\        var i: u256 = 0;
        \\        while (i < 1)
        \\            invariant i <= 1
        \\        {
        \\            i = advance(i);
        \\        }
        \\        return i;
        \\    }
        \\
        \\    pub fn resourceLoop(amount: TokenUnit) {
        \\        var i: u256 = 0;
        \\        while (i < 1)
        \\            invariant i <= 1
        \\        {
        \\            @move(left, right, amount);
        \\            i = i + 1;
        \\        }
        \\    }
        \\}
    ;
    try tmp.dir.writeFile(std.testing.io, .{
        .sub_path = "loop_unsupported_effects.ora",
        .data = source,
    });
    const source_path = try pathFromTmpAlloc(
        allocator,
        tmp,
        "loop_unsupported_effects.ora",
    );
    defer allocator.free(source_path);
    const output_path = try pathFromTmpAlloc(allocator, tmp, "rejected");
    defer allocator.free(output_path);

    const result = try runProcessWithLeanPath(allocator, &.{
        ORA_BINARY_REL,
        "emit",
        "--emit=smt-report,bytecode",
        "--out-dir",
        output_path,
        "--lean-proofs",
        source_path,
    });
    defer allocator.free(result.stdout);
    defer allocator.free(result.stderr);
    try expectExited(result, 1);
    try testing.expect(std.mem.containsAtLeast(
        u8,
        result.stdout,
        1,
        "(storageLoop) lean_certificate query=",
    ));
    try testing.expect(std.mem.containsAtLeast(
        u8,
        result.stdout,
        1,
        "(callLoop) excluded_by=loop_has_call",
    ));
    const resource_prefix = "(resourceLoop) excluded_by=";
    const resource_start = std.mem.indexOf(u8, result.stdout, resource_prefix) orelse
        return error.TestUnexpectedResult;
    const resource_tail = result.stdout[resource_start..];
    const resource_end = std.mem.indexOfScalar(u8, resource_tail, '\n') orelse resource_tail.len;
    try testing.expect(std.mem.containsAtLeast(
        u8,
        resource_tail[0..resource_end],
        1,
        "loop_has_resource_operation",
    ));
    try testing.expect(std.mem.containsAtLeast(
        u8,
        result.stdout,
        1,
        "Lean proof gate cannot cover every runtime loop:",
    ));
    try testing.expectError(
        error.FileNotFound,
        tmp.dir.access(std.testing.io, "rejected/loop_unsupported_effects.hex", .{}),
    );
    const index_path = try pathFromTmpAlloc(
        allocator,
        tmp,
        "rejected/loop_unsupported_effects.formal.artifacts.json",
    );
    defer allocator.free(index_path);
    const index = try readFileAllocForTest(allocator, index_path);
    defer allocator.free(index);
    try testing.expect(std.mem.containsAtLeast(
        u8,
        index,
        1,
        "\"final_artifact_authorization\": \"blocked\"",
    ));
    try testing.expect(std.mem.containsAtLeast(
        u8,
        index,
        1,
        "\"id\": \"lean_userland\", \"domain\": \"userland\", \"phase\": \"proof_artifact\", \"status\": \"rejected\"",
    ));
    try testing.expect(std.mem.containsAtLeast(
        u8,
        index,
        1,
        "\"id\": \"loop_induction_report\"",
    ));
    try testing.expect(!std.mem.containsAtLeast(u8, index, 1, "\"id\": \"lean_proof_certificate\""));
    try testing.expect(!std.mem.containsAtLeast(u8, index, 1, "\"id\": \"dispatcher_proof_certificate\""));
    try tmp.dir.access(
        std.testing.io,
        "rejected/loop_unsupported_effects.formal.loops.json",
        .{},
    );
    try expectNoFormalArtifactsInRepoRoot(allocator, "loop_unsupported_effects");
}

test "structured loop paths preserve branch-local safety and state sources" {
    var result = try collectPackageObligations(testing.allocator, "tests/formal/loop_structured_paths.ora");
    defer result.deinit();

    var branch_loop: ?obligation.LoopSummaryRow = null;
    var switch_loop: ?obligation.LoopSummaryRow = null;
    var nested_branch_loop: ?obligation.LoopSummaryRow = null;
    var no_default_switch_loop: ?obligation.LoopSummaryRow = null;
    for (result.set.loop_summaries) |row| {
        if (row.owner != .statement) continue;
        if (std.mem.eql(u8, row.owner.statement.function_name, "branchUpdate")) branch_loop = row;
        if (std.mem.eql(u8, row.owner.statement.function_name, "switchUpdate")) switch_loop = row;
        if (std.mem.eql(u8, row.owner.statement.function_name, "nestedBranch")) nested_branch_loop = row;
        if (std.mem.eql(u8, row.owner.statement.function_name, "switchNoDefault")) no_default_switch_loop = row;
    }

    const branch = branch_loop orelse return error.TestUnexpectedResult;
    try testing.expect(branch.projectionSupported());
    try testing.expectEqual(obligation.LoopProjectionCompleteness.complete, branch.projection_completeness);
    try testing.expectEqual(@as(usize, 3), branch.edges.len);
    try testing.expectEqual(obligation.LoopControlTarget.backedge, branch.edges[0].target.?);
    try testing.expectEqual(obligation.LoopControlTarget.backedge, branch.edges[1].target.?);
    try testing.expectEqual(obligation.LoopControlTarget.successor, branch.edges[2].target.?);
    try testing.expect(std.mem.indexOf(u8, branch.edges[0].structural_key, "/then/") != null);
    try testing.expect(std.mem.indexOf(u8, branch.edges[1].structural_key, "/else/") != null);
    try testing.expectEqual(@as(usize, 2), loopEdgePathFormulaCount(branch.edges[0]));
    try testing.expectEqual(@as(usize, 2), loopEdgePathFormulaCount(branch.edges[1]));
    try testing.expectEqual(@as(usize, 2), branch.edges[0].safetyFormulaCount());
    try testing.expectEqual(@as(usize, 2), branch.edges[1].safetyFormulaCount());
    try testing.expect(!std.mem.eql(
        u8,
        branch.edges[0].next_state[0].source_key,
        branch.edges[1].next_state[0].source_key,
    ));
    try testing.expectEqualStrings(
        branch.edges[0].next_state[1].source_key,
        branch.edges[1].next_state[1].source_key,
    );

    const switched = switch_loop orelse return error.TestUnexpectedResult;
    try testing.expect(switched.projectionSupported());
    try testing.expectEqual(obligation.LoopProjectionCompleteness.complete, switched.projection_completeness);
    try testing.expectEqual(@as(usize, 4), switched.edges.len);
    try testing.expectEqual(@as(usize, 2), loopEdgePathFormulaCount(switched.edges[0]));
    try testing.expectEqual(@as(usize, 3), loopEdgePathFormulaCount(switched.edges[1]));
    try testing.expectEqual(@as(usize, 3), loopEdgePathFormulaCount(switched.edges[2]));
    for (switched.edges[0..3], 0..) |edge, index| {
        try testing.expectEqual(obligation.LoopControlTarget.backedge, edge.target.?);
        try testing.expect(std.mem.indexOf(u8, edge.structural_key, "/case:") != null);
        try testing.expectEqual(@as(usize, 2), edge.safetyFormulaCount());
        for (switched.edges[0..index]) |previous| {
            try testing.expect(!std.mem.eql(u8, previous.next_state[0].source_key, edge.next_state[0].source_key));
        }
    }
    try testing.expectEqual(obligation.LoopControlTarget.successor, switched.edges[3].target.?);

    const nested = nested_branch_loop orelse return error.TestUnexpectedResult;
    try testing.expect(nested.projectionSupported());
    try testing.expectEqual(@as(usize, 4), nested.edges.len);
    try testing.expect(std.mem.count(u8, nested.edges[0].structural_key, "/op:") >= 2);
    try testing.expect(std.mem.count(u8, nested.edges[1].structural_key, "/op:") >= 2);
    try testing.expect(std.mem.count(u8, nested.edges[2].structural_key, "/op:") >= 1);
    for (nested.edges[0..3], 0..) |edge, index| {
        try testing.expectEqual(obligation.LoopControlTarget.backedge, edge.target.?);
        for (nested.edges[0..index]) |previous| {
            try testing.expect(!std.mem.eql(u8, previous.structural_key, edge.structural_key));
        }
    }

    var bounded_result = try collectPackageObligations(testing.allocator, "tests/formal/loop_path_limit.ora");
    defer bounded_result.deinit();
    var bounded_path_loop: ?obligation.LoopSummaryRow = null;
    for (bounded_result.set.loop_summaries) |row| {
        if (row.owner == .statement and
            std.mem.eql(u8, row.owner.statement.function_name, "boundedPathLimit"))
        {
            bounded_path_loop = row;
        }
    }
    const bounded = bounded_path_loop orelse return error.TestUnexpectedResult;
    try testing.expectEqual(
        obligation.LoopProjectionCompleteness.path_limit_exceeded,
        bounded.projection_completeness,
    );
    try testing.expect(loopSummaryHasReason(bounded, .loop_projection_path_limit_exceeded));
    try testing.expect(!bounded.projectionSupported());
    try testing.expectEqual(@as(usize, 0), bounded.edges.len);

    const no_default = no_default_switch_loop orelse return error.TestUnexpectedResult;
    try testing.expect(no_default.projectionSupported());
    try testing.expectEqual(@as(usize, 3), no_default.edges.len);
    try testing.expect(std.mem.indexOf(u8, no_default.edges[0].structural_key, "/case:0/") != null);
    try testing.expect(std.mem.indexOf(u8, no_default.edges[1].structural_key, "/no-match/") != null);
    try testing.expectEqual(@as(usize, 2), no_default.edges[0].safetyFormulaCount());
    try testing.expectEqual(@as(usize, 1), no_default.edges[1].safetyFormulaCount());
    try testing.expectEqual(@as(usize, 0), no_default.edges[2].safetyFormulaCount());

    const report = try dumpManifestToOwnedString(testing.allocator, result.set);
    defer testing.allocator.free(report);
    const repeated = try dumpManifestToOwnedString(testing.allocator, result.set);
    defer testing.allocator.free(repeated);
    try testing.expectEqualStrings(report, repeated);
    try testing.expect(std.mem.containsAtLeast(u8, report, 1, "\"source_key\":\"module-op:"));
}

test "break and continue project as complete source-owned LoopVC edges" {
    var result = try collectPackageObligations(testing.allocator, "tests/formal/loop_control_targets.ora");
    defer result.deinit();

    const cases = [_]struct {
        function_name: []const u8,
        kind: obligation.LoopKind,
        backedges: usize,
        successors: usize,
        control_kind: []const u8,
    }{
        .{ .function_name = "whileBreak", .kind = .scf_while, .backedges = 1, .successors = 2, .control_kind = "break" },
        .{ .function_name = "whileContinue", .kind = .scf_while, .backedges = 2, .successors = 1, .control_kind = "continue" },
        .{ .function_name = "forBreak", .kind = .scf_for, .backedges = 1, .successors = 2, .control_kind = "break" },
        .{ .function_name = "forContinue", .kind = .scf_for, .backedges = 2, .successors = 1, .control_kind = "continue" },
    };

    try testing.expectEqual(cases.len, result.set.loop_summaries.len);
    for (cases) |case| {
        const summary = loopSummaryByFunction(result.set, case.function_name) orelse
            return error.TestUnexpectedResult;
        try testing.expectEqual(case.kind, summary.loop_kind);
        try testing.expectEqual(obligation.LoopProjectionCompleteness.complete, summary.projection_completeness);
        try testing.expect(summary.projectionSupported());
        try testing.expectEqual(@as(usize, 3), summary.edges.len);

        var backedges: usize = 0;
        var successors: usize = 0;
        var control_edges: usize = 0;
        for (summary.edges) |edge| {
            try testing.expectEqual(summary.variables.len, edge.next_state.len);
            switch (edge.target orelse return error.TestUnexpectedResult) {
                .backedge => backedges += 1,
                .successor => successors += 1,
                else => return error.TestUnexpectedResult,
            }
            const control_fragment = try std.fmt.allocPrint(
                testing.allocator,
                "/control:{s}:",
                .{case.control_kind},
            );
            defer testing.allocator.free(control_fragment);
            if (std.mem.indexOf(u8, edge.structural_key, control_fragment) != null) {
                control_edges += 1;
                if (case.kind == .scf_for and std.mem.eql(u8, case.control_kind, "continue")) {
                    try testing.expect(std.mem.startsWith(
                        u8,
                        edge.next_state[0].source_key,
                        "for/control-induction-step:",
                    ));
                }
            }
        }
        try testing.expectEqual(case.backedges, backedges);
        try testing.expectEqual(case.successors, successors);
        try testing.expectEqual(@as(usize, 1), control_edges);

        try testing.expectEqual(@as(usize, 1), summary.query_ids.induction.len);
        const query = obligation.findById(result.set.queries, summary.query_ids.induction[0]) orelse
            return error.TestUnexpectedResult;
        try testing.expectEqual(obligation.ProofRequirement.lean_certificate, query.proof_requirement);
    }
}

test "loop control provenance mutations fail closed by name" {
    {
        var compilation = try compilePackage("tests/formal/loop_control_targets.ora");
        defer compilation.deinit();
        const hir_result = try compilation.db.lowerToHir(compilation.root_module_id);
        try testing.expect(hir_result.diagnostics.isEmpty());
        const loop_op = findFirstOpByName(hir_result.module.raw_module, "scf.while") orelse
            return error.TestUnexpectedResult;
        try testing.expect(mlir.mlirOperationRemoveAttributeByName(
            loop_op,
            strRef("ora.loop_control_manifest"),
        ));
        var mutated = try obligation_from_mlir.collect(testing.allocator, hir_result.module.raw_module, .{});
        defer mutated.deinit();
        try expectOneLoopProjectionFailure(
            mutated.set,
            .control_manifest_missing,
            .loop_control_manifest_missing,
        );
    }

    {
        var compilation = try compilePackage("tests/formal/loop_control_targets.ora");
        defer compilation.deinit();
        const hir_result = try compilation.db.lowerToHir(compilation.root_module_id);
        try testing.expect(hir_result.diagnostics.isEmpty());
        const marker = findFirstOpByName(hir_result.module.raw_module, "ora.loop_control") orelse
            return error.TestUnexpectedResult;
        mlir.mlirOperationRemoveFromParent(marker);
        defer mlir.mlirOperationDestroy(marker);
        var mutated = try obligation_from_mlir.collect(testing.allocator, hir_result.module.raw_module, .{});
        defer mutated.deinit();
        try expectOneLoopProjectionFailure(
            mutated.set,
            .control_marker_missing,
            .loop_control_marker_missing,
        );
    }

    {
        var compilation = try compilePackage("tests/formal/loop_control_targets.ora");
        defer compilation.deinit();
        const hir_result = try compilation.db.lowerToHir(compilation.root_module_id);
        try testing.expect(hir_result.diagnostics.isEmpty());
        const marker = findFirstOpByName(hir_result.module.raw_module, "ora.loop_control") orelse
            return error.TestUnexpectedResult;
        const context = mlir.mlirOperationGetContext(marker);
        mlir.oraOperationSetAttributeByName(
            marker,
            strRef("target_loop_source_id"),
            mlir.oraIntegerAttrCreateI64FromType(mlir.oraIntegerTypeCreate(context, 64), 999_999),
        );
        var mutated = try obligation_from_mlir.collect(testing.allocator, hir_result.module.raw_module, .{});
        defer mutated.deinit();
        try expectOneLoopProjectionFailure(
            mutated.set,
            .control_marker_target_mismatch,
            .loop_control_marker_target_mismatch,
        );
    }

    {
        var compilation = try compilePackage("tests/formal/loop_control_targets.ora");
        defer compilation.deinit();
        const hir_result = try compilation.db.lowerToHir(compilation.root_module_id);
        try testing.expect(hir_result.diagnostics.isEmpty());
        const marker = findFirstOpByName(hir_result.module.raw_module, "ora.loop_control") orelse
            return error.TestUnexpectedResult;
        mlir.mlirOperationSetOperands(marker, 0, null);
        var mutated = try obligation_from_mlir.collect(testing.allocator, hir_result.module.raw_module, .{});
        defer mutated.deinit();
        try expectOneLoopProjectionFailure(
            mutated.set,
            .control_marker_malformed,
            .loop_control_marker_malformed,
        );
    }

    {
        var compilation = try compilePackage("tests/formal/loop_control_targets.ora");
        defer compilation.deinit();
        const hir_result = try compilation.db.lowerToHir(compilation.root_module_id);
        try testing.expect(hir_result.diagnostics.isEmpty());
        const loop_op = findFirstOpByName(hir_result.module.raw_module, "scf.while") orelse
            return error.TestUnexpectedResult;
        const manifest_name = strRef("ora.loop_control_manifest");
        const manifest = mlir.oraOperationGetAttributeByName(loop_op, manifest_name);
        try testing.expect(!mlir.oraAttributeIsNull(manifest));
        try testing.expectEqual(@as(usize, 1), mlir.oraArrayAttrGetNumElements(manifest));
        const item = mlir.oraArrayAttrGetElement(manifest, 0);
        const duplicated = [_]mlir.MlirAttribute{ item, item };
        mlir.oraOperationSetAttributeByName(
            loop_op,
            manifest_name,
            mlir.oraArrayAttrCreate(mlir.mlirOperationGetContext(loop_op), duplicated.len, &duplicated),
        );
        var mutated = try obligation_from_mlir.collect(testing.allocator, hir_result.module.raw_module, .{});
        defer mutated.deinit();
        try expectOneLoopProjectionFailure(
            mutated.set,
            .control_marker_duplicate,
            .loop_control_marker_duplicate,
        );
    }
}

test "successful and error returns project edge-local LoopVC target assertions" {
    var result = try collectPackageObligations(testing.allocator, "tests/formal/loop_return_targets.ora");
    defer result.deinit();
    var z3_context = try z3_verification.Z3Context.init(testing.allocator);
    defer z3_context.deinit();
    var z3_adapter = obligation_to_z3.Adapter.init(&z3_context, testing.allocator, result.set);

    const cases = [_]struct {
        function_name: []const u8,
        target: obligation.LoopControlTarget,
        control_kind: []const u8,
    }{
        .{ .function_name = "success", .target = .function_return, .control_kind = "function_return" },
        .{ .function_name = "failure", .target = .error_return, .control_kind = "error_return" },
    };

    try testing.expectEqual(cases.len, result.set.loop_summaries.len);
    for (cases) |case| {
        const summary = loopSummaryByFunction(result.set, case.function_name) orelse
            return error.TestUnexpectedResult;
        try testing.expectEqual(obligation.LoopProjectionCompleteness.complete, summary.projection_completeness);
        try testing.expect(summary.projectionSupported());

        var matched_return = false;
        for (summary.edges) |edge| {
            if (edge.target != case.target) continue;
            try testing.expect(!matched_return);
            matched_return = true;
            try testing.expect(std.mem.indexOf(u8, edge.structural_key, case.control_kind) != null);
            try testing.expect(edge.target_formulas.len > 0);
            try testing.expectEqual(@as(usize, 0), edge.safetyFormulaCount());
        }
        try testing.expect(matched_return);
        try testing.expectEqual(@as(usize, 2), summary.query_ids.post.len);
        var bound_post_count: usize = 0;
        for (summary.query_ids.post) |query_id| {
            const post_query = obligation.findById(result.set.queries, query_id) orelse
                return error.TestUnexpectedResult;
            try testing.expectEqual(obligation.ArtifactPolicy.blocks_verified_artifacts, post_query.artifact_policy);
            try testing.expectEqual(@as(usize, 1), post_query.obligation_ids.len);
            const checked = try z3_adapter.checkLoopQueryForRow(post_query);
            try testing.expectEqual(obligation_to_z3.CheckStatus.proved, checked.status);
            for (result.query_bindings) |binding| {
                if (binding.query_id != query_id or binding.kind != .loop_invariant_post) continue;
                try testing.expectEqual(@as(?obligation.LogicalRole, .ensures), binding.logical_role);
                try testing.expectEqual(@as(?usize, summary.loop_source_op_id.?), binding.loop_source_op_id);
                try testing.expectEqual(@as(usize, 1), binding.obligation_ids.len);
                bound_post_count += 1;
            }
        }
        try testing.expectEqual(@as(usize, 2), bound_post_count);
        const induction_query = obligation.findById(result.set.queries, summary.query_ids.induction[0]) orelse
            return error.TestUnexpectedResult;
        switch (obligation_to_lean.querySemanticSupport(result.set, induction_query)) {
            .supported => {},
            .unsupported => return error.TestUnexpectedResult,
        }
    }

    var postcondition_source_binding_count: usize = 0;
    for (result.source_fact_bindings) |binding| {
        if (!std.mem.eql(u8, binding.kind, "ensures") and
            !std.mem.eql(u8, binding.kind, "ensures_ok") and
            !std.mem.eql(u8, binding.kind, "ensures_err")) continue;
        postcondition_source_binding_count += 1;
        try testing.expectEqual(@as(usize, 1), binding.query_ids.len);
        const query = obligation.findById(result.set.queries, binding.query_ids[0]) orelse
            return error.TestUnexpectedResult;
        try testing.expectEqual(obligation.VerificationQueryKind.loop_invariant_post, query.kind);
    }
    try testing.expectEqual(@as(usize, 4), postcondition_source_binding_count);

    const report = try dumpManifestToOwnedString(testing.allocator, result.set);
    defer testing.allocator.free(report);
    try testing.expect(std.mem.containsAtLeast(u8, report, 1, "\"target\":\"function_return\""));
    try testing.expect(std.mem.containsAtLeast(u8, report, 1, "\"target\":\"error_return\""));
    try testing.expect(std.mem.containsAtLeast(u8, report, 2, "\"target_formulas\":["));
}

test "loop return target provenance and formula inventory mutations fail closed" {
    {
        var compilation = try compilePackage("tests/formal/loop_return_targets.ora");
        defer compilation.deinit();
        const hir_result = try compilation.db.lowerToHir(compilation.root_module_id);
        try testing.expect(hir_result.diagnostics.isEmpty());
        const marker = findFirstOpByName(hir_result.module.raw_module, "ora.loop_control") orelse
            return error.TestUnexpectedResult;
        mlir.oraOperationSetAttributeByName(
            marker,
            strRef("kind"),
            mlir.oraStringAttrCreate(mlir.mlirOperationGetContext(marker), strRef("error_return")),
        );
        var mutated = try obligation_from_mlir.collect(testing.allocator, hir_result.module.raw_module, .{});
        defer mutated.deinit();
        try expectOneLoopProjectionFailure(
            mutated.set,
            .control_marker_malformed,
            .loop_control_marker_malformed,
        );
    }

    {
        var compilation = try compilePackage("tests/formal/loop_return_targets.ora");
        defer compilation.deinit();
        const hir_result = try compilation.db.lowerToHir(compilation.root_module_id);
        try testing.expect(hir_result.diagnostics.isEmpty());
        const target = findFirstOpByName(hir_result.module.raw_module, "ora.ensures") orelse
            return error.TestUnexpectedResult;
        mlir.mlirOperationSetOperands(target, 0, null);
        var mutated = try obligation_from_mlir.collect(testing.allocator, hir_result.module.raw_module, .{});
        defer mutated.deinit();
        try expectOneLoopProjectionFailure(
            mutated.set,
            .target_mismatch,
            .loop_projection_target_mismatch,
        );
    }
}

test "for LoopVC retains and validates the source u256 induction type" {
    var result = try collectPackageObligations(testing.allocator, "tests/formal/loop_for_types.ora");
    defer result.deinit();

    try testing.expectEqual(@as(usize, 1), result.set.loop_summaries.len);
    const summary = result.set.loop_summaries[0];
    try testing.expectEqual(obligation.LoopKind.scf_for, summary.loop_kind);
    try testing.expect(summary.projectionSupported());
    try testing.expectEqual(obligation.LoopProjectionCompleteness.complete, summary.projection_completeness);
    try testing.expectEqual(@as(usize, 2), summary.variables.len);
    try testing.expectEqual(
        type_builtin.lookupBuiltinById(.u256).comptime_type_id,
        summary.variables[0].ty.compiler_type_id,
    );
    try testing.expectEqual(@as(usize, 2), summary.edges.len);
    try testing.expectEqual(obligation.LoopControlTarget.backedge, summary.edges[0].target.?);
    try testing.expectEqual(obligation.LoopControlTarget.successor, summary.edges[1].target.?);

    const query = try findLoopInductionQuery(result.set);
    try testing.expectEqual(obligation.ProofRequirement.lean_certificate, query.proof_requirement);
}

test "for LoopVC rejects missing and non-u256 induction metadata" {
    var compilation = try compilePackage("tests/formal/loop_for_types.ora");
    defer compilation.deinit();
    const hir_result = try compilation.db.lowerToHir(compilation.root_module_id);
    try testing.expect(hir_result.diagnostics.isEmpty());

    const for_op = findFirstOpByName(hir_result.module.raw_module, "scf.for") orelse
        return error.TestUnexpectedResult;
    var unsigned_cmp = false;
    try testing.expect(mlir.oraScfForOpGetUnsignedCmp(for_op, &unsigned_cmp));
    try testing.expect(unsigned_cmp);

    const attr_name = strRef("ora.induction_compiler_type_id");
    try testing.expect(mlir.mlirOperationRemoveAttributeByName(for_op, attr_name));

    var missing = try obligation_from_mlir.collect(testing.allocator, hir_result.module.raw_module, .{});
    defer missing.deinit();
    try testing.expectEqual(@as(usize, 1), missing.set.loop_summaries.len);
    try testing.expect(loopSummaryHasReason(missing.set.loop_summaries[0], .loop_value_type_unresolved));
    try testing.expect(!missing.set.loop_summaries[0].projectionSupported());

    const context = mlir.mlirOperationGetContext(for_op);
    mlir.oraOperationSetAttributeByName(
        for_op,
        attr_name,
        mlir.oraIntegerAttrCreateI64FromType(
            mlir.oraIntegerTypeCreate(context, 256),
            type_builtin.lookupBuiltinById(.u8).comptime_type_id,
        ),
    );
    var wrong_width = try obligation_from_mlir.collect(testing.allocator, hir_result.module.raw_module, .{});
    defer wrong_width.deinit();
    try testing.expectEqual(@as(usize, 1), wrong_width.set.loop_summaries.len);
    try testing.expect(loopSummaryHasReason(wrong_width.set.loop_summaries[0], .loop_value_type_unresolved));
    try testing.expect(!wrong_width.set.loop_summaries[0].projectionSupported());
}

test "structured loop certificates arm the real CLI and missing proofs block artifacts" {
    std.Io.Dir.cwd().access(std.testing.io, ORA_BINARY_REL, .{}) catch |err| switch (err) {
        error.FileNotFound => return error.SkipZigTest,
        else => return err,
    };

    const allocator = testing.allocator;
    var tmp = testing.tmpDir(.{});
    defer tmp.cleanup();
    const output_path = try pathFromTmpAlloc(allocator, tmp, "structured-loop-blocked");
    defer allocator.free(output_path);
    const empty_manifest = try pathFromTmpAlloc(allocator, tmp, "empty-proofs.json");
    defer allocator.free(empty_manifest);
    try std.Io.Dir.cwd().writeFile(std.testing.io, .{
        .sub_path = empty_manifest,
        .data = "{\"schema_version\":1,\"proofs\":[]}",
    });

    const result = try runProcessWithLeanPath(allocator, &.{
        ORA_BINARY_REL,
        "emit",
        "--emit=smt-report,sir-text,bytecode",
        "--out-dir",
        output_path,
        "--lean-proofs",
        empty_manifest,
        "tests/formal/loop_structured_cli.ora",
    });
    defer allocator.free(result.stdout);
    defer allocator.free(result.stderr);
    try expectExited(result, 1);
    try testing.expect(std.mem.containsAtLeast(u8, result.stdout, 1, "certificate-required=1"));
    try testing.expect(
        std.mem.containsAtLeast(u8, result.stdout, 1, "lean_required_failure") or
            std.mem.containsAtLeast(u8, result.stderr, 1, "lean_required_failure"),
    );
    const artifact_path = try std.fs.path.join(allocator, &.{ output_path, "loop_structured_cli.hex" });
    defer allocator.free(artifact_path);
    try testing.expectError(error.FileNotFound, std.Io.Dir.cwd().access(std.testing.io, artifact_path, .{}));
}

test "loop header facts project canonical assumptions through an independent stable prefix inventory" {
    var result = try collectPackageObligations(testing.allocator, "tests/formal/loop_header_fact_projection.ora");
    defer result.deinit();

    var requires_seen = false;
    var fallthrough_seen = false;
    var earlier_loop_count: usize = 0;
    var first_target_position: ?u32 = null;
    var dynamic_length_seen = false;
    var storage_rejected = false;
    var call_rejected = false;
    for (result.set.loop_summaries) |summary| {
        if (summary.owner != .statement) continue;
        const name = summary.owner.statement.function_name;
        if (std.mem.eql(u8, name, "fromRequires")) {
            requires_seen = true;
            try testing.expectEqual(@as(usize, 1), summary.header_facts.len);
            const fact = summary.header_facts[0];
            try testing.expectEqual(obligation.LoopHeaderFactSource.function_requires, fact.source_kind);
            try testing.expect(fact.source.position < fact.target.position);
            try testing.expectEqual(@as(usize, 1), fact.stable_values.len);
            const assumption_id = fact.assumption_id orelse return error.TestUnexpectedResult;
            try testing.expect(result.set.loopHeaderFactsMatchAssumptions(summary, &.{assumption_id}));
        } else if (std.mem.eql(u8, name, "fromFallthrough")) {
            fallthrough_seen = true;
            try testing.expectEqual(@as(usize, 1), summary.header_facts.len);
            try testing.expectEqual(obligation.LoopHeaderFactSource.path_assume, summary.header_facts[0].source_kind);
            try testing.expectEqual(@as(?obligation.Id, null), summary.header_facts[0].assumption_id);
        } else if (std.mem.eql(u8, name, "acrossEarlierLoop")) {
            earlier_loop_count += 1;
            try testing.expectEqual(@as(usize, 1), summary.header_facts.len);
            try testing.expectEqual(obligation.LoopHeaderFactSource.function_requires, summary.header_facts[0].source_kind);
            if (first_target_position) |first| try testing.expect(summary.header_facts[0].target.position > first) else first_target_position = summary.header_facts[0].target.position;
        } else if (std.mem.eql(u8, name, "fromDynamicLength")) {
            dynamic_length_seen = true;
            try testing.expectEqual(@as(usize, 1), summary.header_facts.len);
            try testing.expectEqual(@as(usize, 1), summary.header_facts[0].stable_values.len);
            try testing.expectEqual(@as(usize, 1), summary.context_variables.len);
            try testing.expect(obligation.freeVarIdEql(
                summary.header_facts[0].stable_values[0].id,
                summary.context_variables[0].id.?,
            ));
        } else if (std.mem.eql(u8, name, "storageDerivedFact")) {
            storage_rejected = loopSummaryHasReason(summary, .loop_header_fact_type_unsupported);
        } else if (std.mem.eql(u8, name, "callBetweenFactAndLoop")) {
            call_rejected = loopSummaryHasReason(summary, .loop_header_fact_effect_unsupported);
        }
    }
    try testing.expect(requires_seen and fallthrough_seen and dynamic_length_seen);
    try testing.expectEqual(@as(usize, 2), earlier_loop_count);
    try testing.expect(storage_rejected and call_rejected);
    try testing.expectEqual(
        obligation.ArtifactDecision{ .blocked = .missing_proof },
        result.set.artifactDecision(),
    );
}

test "repository tactic certifies dominating loop header facts deterministically" {
    try expectRepositoryStorageTacticDeterministic(
        "tests/formal/loop_header_fact_auto.ora",
        "loop_header_fact_auto",
        5,
    );
}

test "loop blocker batch header control slots retain requires and certify the real CLI" {
    var result = try collectPackageObligations(testing.allocator, "tests/formal/loop_header_control_slots.ora");
    defer result.deinit();
    try testing.expectEqual(@as(usize, 1), result.set.loop_summaries.len);
    const summary = result.set.loop_summaries[0];
    try testing.expect(summary.projectionSupported());
    try testing.expectEqual(@as(usize, 1), summary.header_facts.len);
    try testing.expectEqual(obligation.LoopHeaderFactSource.function_requires, summary.header_facts[0].source_kind);
    try expectRepositoryStorageTacticDeterministic(
        "tests/formal/loop_header_control_slots.ora",
        "loop_header_control_slots",
        1,
    );
}

test "loop blocker batch compound guards retain ordered conditional checkpoints" {
    var result = try collectPackageObligations(testing.allocator, "tests/formal/loop_compound_guards.ora");
    defer result.deinit();
    try testing.expectEqual(@as(usize, 3), result.set.loop_summaries.len);
    for (result.set.loop_summaries) |summary| try testing.expect(summary.projectionSupported());
    var staged = try collectPackageObligations(testing.allocator, "tests/formal/loop_compound_guard_staged_division.ora");
    defer staged.deinit();
    try testing.expectEqual(@as(usize, 1), staged.set.loop_summaries.len);
    const summary = staged.set.loop_summaries[0];
    try testing.expect(summary.projectionSupported());
    var checked_exit = false;
    var unchecked_exit = false;
    for (summary.edges) |edge| {
        if (edge.target != .successor) continue;
        var checkpoint_count: usize = 0;
        for (edge.timeline, 0..) |event, index| {
            if (event.payload != .safety_checkpoint) continue;
            checkpoint_count += 1;
            // The branch premise precedes each RHS check; the final condition
            // follows them, including when that condition selects the exit.
            try testing.expect(index > 0 and index + 1 < edge.timeline.len);
            try testing.expect(edge.timeline[0].payload == .path_assumption);
        }
        if (checkpoint_count == 0) unchecked_exit = true else {
            checked_exit = true;
            try testing.expectEqual(@as(usize, 2), checkpoint_count);
        }
    }
    try testing.expect(checked_exit and unchecked_exit);
    try expectRepositoryStorageTacticDeterministic("tests/formal/loop_compound_guards.ora", "loop_compound_guards", 3);
    try expectRepositoryStorageTacticDeterministic("tests/formal/loop_compound_guard_staged_division.ora", "loop_compound_guard_staged_division", 1);
}

test "loop blocker batch checked compound guards certify storage writes" {
    const source = "tests/formal/loop_compound_guard_storage.ora";
    var result = try collectPackageObligations(testing.allocator, source);
    defer result.deinit();
    try testing.expectEqual(@as(usize, 1), result.set.loop_summaries.len);
    const summary = result.set.loop_summaries[0];
    try testing.expect(summary.projectionSupported());
    try testing.expect(obligation_to_lean.loopSummaryUsesDirectStorageCertificate(summary));
    var found_ordered_store = false;
    for (summary.edges) |edge| {
        var checked = false;
        var branched_after_check = false;
        for (edge.timeline) |event| switch (event.payload) {
            .safety_checkpoint => checked = true,
            .path_assumption => if (checked) {
                branched_after_check = true;
            },
            .storage_write => if (branched_after_check) {
                found_ordered_store = true;
            },
            else => {},
        };
    }
    // An all-leading-paths transfer cannot justify this exact event order.
    try testing.expect(found_ordered_store);
    try expectRepositoryStorageTacticDeterministic(source, "loop_compound_guard_storage", 1);
}

test "loop blocker batch guard effects and late contradictions cannot authorize bytecode" {
    var projected = try collectPackageObligations(testing.allocator, "tests/formal/fail_loop_compound_guard_effects.ora");
    defer projected.deinit();
    try testing.expectEqual(@as(usize, 2), projected.set.loop_summaries.len);
    for (projected.set.loop_summaries) |summary| {
        try testing.expect(!summary.projectionSupported());
        try testing.expectEqual(obligation.LoopProjectionCompleteness.unsupported_effect, summary.projection_completeness);
        try testing.expect(loopSummaryHasReason(summary, .loop_formula_unsupported));
    }
    for ([_][]const u8{ "fail_loop_compound_guard_effects", "fail_loop_compound_guard_late_false" }) |stem| {
        var tmp = testing.tmpDir(.{});
        defer tmp.cleanup();
        const output = try pathFromTmpAlloc(testing.allocator, tmp, "blocked");
        defer testing.allocator.free(output);
        const source = try std.fmt.allocPrint(testing.allocator, "tests/formal/{s}.ora", .{stem});
        defer testing.allocator.free(source);
        const run = try runProcessWithLeanPath(testing.allocator, &.{ ORA_BINARY_REL, "emit", "--emit=bytecode", "--out-dir", output, "--lean-proofs", source });
        defer testing.allocator.free(run.stdout);
        defer testing.allocator.free(run.stderr);
        try expectExited(run, 1);
        if (std.mem.eql(u8, stem, "fail_loop_compound_guard_late_false")) {
            try testing.expect(std.mem.indexOf(u8, run.stdout, "Lean proof row rejected: lean_proof_rejected") != null);
        }
        const hex = try std.fmt.allocPrint(testing.allocator, "blocked/{s}.hex", .{stem});
        defer testing.allocator.free(hex);
        try testing.expectError(error.FileNotFound, tmp.dir.access(std.testing.io, hex, .{}));
    }
}

test "loop blocker batch long conjunctions retain prefix checks without impossible combinations" {
    var result = try collectPackageObligations(testing.allocator, "tests/formal/loop_long_compound_guard.ora");
    defer result.deinit();
    const summary = result.set.loop_summaries[0];
    try testing.expect(summary.projectionSupported());
    // Each short-circuit prefix retains both final condition outcomes. This
    // preserves the body/check inventory even when a guard is always false.
    try testing.expectEqual(@as(usize, 30), summary.edges.len);
    var checked_exit = false;
    for (summary.edges) |edge| {
        if (edge.target == .successor and edge.safetyFormulaCount() >= 2) checked_exit = true;
    }
    try testing.expect(checked_exit);
    var storage = try collectPackageObligations(testing.allocator, "tests/formal/loop_compound_storage_invariant.ora");
    defer storage.deinit();
    const preserved = storage.set.loop_summaries[0];
    try testing.expect(preserved.projectionSupported());
    try testing.expectEqual(@as(usize, 2), preserved.invariant_formulas.len);
    for (preserved.invariant_formulas) |formula| try testing.expect(formula == .term);
    var reads: usize = 0;
    for (preserved.edges) |edge| for (edge.timeline) |event| {
        if (event.payload == .storage_read) reads += 1;
    };
    try testing.expect(reads >= 4);
    try expectRepositoryStorageTacticDeterministic("tests/formal/loop_long_compound_guard.ora", "loop_long_compound_guard", 1);
}

test "certified call metadata retains private result and error payload signedness" {
    var compilation = try compileText(
        \\error Mixed(small: i8, wide: i256, flag: bool);
        \\fn signedByte(x: i8) -> i8 { return x; }
        \\fn unsignedByte(x: u8) -> u8 { return x; }
        \\fn signedWord(x: i256) -> i256 { return x; }
        \\fn boundedWord(x: InRange<u256, 0, 16>) -> InRange<u256, 0, 16> { return x; }
        \\fn boolean(x: bool) -> bool { return x; }
        \\fn empty() { }
        \\fn fallible(x: i8) -> !i8 | Mixed {
        \\    if (x < 0) { return Mixed(x, 0, false); }
        \\    return x;
        \\}
    );
    defer compilation.deinit();
    const hir = try compilation.db.lowerToHir(compilation.root_module_id);
    try testing.expect(hir.diagnostics.isEmpty());
    const cases = .{
        .{ "signedByte", type_builtin.lookupBuiltinById(.i8).comptime_type_id },
        .{ "unsignedByte", type_builtin.lookupBuiltinById(.u8).comptime_type_id },
        .{ "signedWord", type_builtin.lookupBuiltinById(.i256).comptime_type_id },
        .{ "boundedWord", type_builtin.lookupBuiltinById(.u256).comptime_type_id },
        .{ "boolean", 0 },
        .{ "empty", 0 },
        .{ "fallible", type_builtin.lookupBuiltinById(.i8).comptime_type_id },
    };
    inline for (cases) |case| {
        const function = findFunctionByName(hir.module.raw_module, case[0]) orelse return error.TestUnexpectedResult;
        const attr = mlir.oraOperationGetAttributeByName(function, strRef("ora.return_compiler_type_id"));
        try testing.expect(!mlir.oraAttributeIsNull(attr));
        try testing.expectEqual(@as(i64, case[1]), mlir.oraIntegerAttrGetValueSInt(attr));
    }
    const declaration = findFirstOpByName(hir.module.raw_module, "ora.error.decl") orelse return error.TestUnexpectedResult;
    const payload = mlir.oraOperationGetAttributeByName(declaration, strRef("ora.param_compiler_type_ids"));
    try testing.expectEqual(@as(usize, 3), mlir.oraArrayAttrGetNumElements(payload));
    const expected = [_]i64{ type_builtin.lookupBuiltinById(.i8).comptime_type_id, type_builtin.lookupBuiltinById(.i256).comptime_type_id, 0 };
    for (expected, 0..) |id, index| try testing.expectEqual(id, mlir.oraIntegerAttrGetValueSInt(mlir.oraArrayAttrGetElement(payload, index)));
}

test "refinement header parameter metadata still rejects a mismatched base width" {
    const h = createContext();
    defer destroyContext(h);
    for ([_]u16{ 32, 128, 256 }) |width| {
        const text = try std.fmt.allocPrint(testing.allocator,
            \\module {{
            \\  func.func @parameter(%arg0: !ora.in_range<i{d}, 0, 0, 0, 0, 0, 0, 0, 4>) attributes {{
            \\    ora.param_names = ["limit"],
            \\    ora.param_binding_ids = ["file:0:pattern:0"],
            \\    ora.param_compiler_type_ids = [{d} : i256]
            \\  }} {{
            \\    func.return
            \\  }}
            \\}}
        , .{ width, type_builtin.lookupBuiltinById(.u256).comptime_type_id });
        defer testing.allocator.free(text);
        const module = try parseModule(h.ctx, text);
        defer mlir.oraModuleDestroy(module);
        var result = try obligation_from_mlir.collect(testing.allocator, module, .{});
        defer result.deinit();
        try testing.expectEqual(width != 256, result.set.hasBlockingDiagnostic());
        if (width != 256) {
            var saw_mismatch = false;
            for (result.set.diagnostics) |diagnostic| {
                if (diagnostic.kind == .missing_type and
                    std.mem.indexOf(u8, diagnostic.message, "disagrees with its MLIR width") != null)
                    saw_mismatch = true;
            }
            try testing.expect(saw_mismatch);
        }
    }
}

test "refinement header facts preserve both positive-minimum assumptions" {
    var result = try collectPackageObligations(testing.allocator, "tests/formal/loop_header_refinement_positive_minimum.ora");
    defer result.deinit();
    try testing.expect(!result.set.hasBlockingDiagnostic());
    try testing.expectEqual(@as(usize, 1), result.set.loop_summaries.len);
    const summary = result.set.loop_summaries[0];
    try testing.expect(summary.projectionSupported());
    try testing.expectEqual(@as(usize, 2), summary.header_facts.len);
    for (summary.header_facts) |fact| {
        try testing.expectEqual(obligation.LoopHeaderFactSource.function_requires, fact.source_kind);
        try testing.expect(fact.assumption_id != null);
        try testing.expectEqual(@as(usize, 1), fact.stable_values.len);
        try testing.expectEqualDeep(
            obligation.TypeRef{ .compiler_type_id = type_builtin.lookupBuiltinById(.u256).comptime_type_id },
            fact.stable_values[0].ty,
        );
    }
}

test "refinement header loops certify wide and narrow carriers deterministically" {
    try expectRepositoryStorageTacticDeterministic(
        "tests/formal/loop_header_refinement_auto.ora",
        "loop_header_refinement_auto",
        2,
    );
}

test "refinement header satisfied requires do not erase input checks" {
    var compilation = try compilePackage("tests/formal/loop_header_refinement_auto.ora");
    defer compilation.deinit();
    const hir = try compilation.db.lowerToHir(compilation.root_module_id);
    try testing.expect(hir.diagnostics.isEmpty());
    var bindings = try test_helpers.collectFormalQueryBindingsForVerifier(testing.allocator, hir.module.raw_module);
    defer {
        bindings.formal_result.deinit();
        testing.allocator.free(bindings.z3_bindings);
    }
    var verifier = try z3_verification.VerificationPass.init(testing.allocator);
    defer verifier.deinit();
    verifier.setFormalQueryBindings(bindings.z3_bindings);
    var result = try verifier.runVerificationPass(hir.module.raw_module);
    defer result.deinit();
    try testing.expect(result.success);
    try testing.expectEqual(@as(usize, 0), result.proven_guard_ids.count());
    compiler.refinement_guards.cleanupRefinementGuardsWithOptions(
        testing.allocator,
        hir.context,
        hir.module.raw_module,
        &result.proven_guard_ids,
        .{},
    );
    const rendered = mlir.oraOperationPrintToString(mlir.oraModuleGetOperation(hir.module.raw_module));
    defer mlir.oraStringRefFree(rendered);
    try testing.expectEqual(@as(usize, 2), std.mem.count(
        u8,
        rendered.data[0..rendered.length],
        "ora.verification_context = \"refinement_guard\"",
    ));
}

test "refinement header input policy is assigned only to parameter guard violation queries" {
    const h = createContext();
    defer destroyContext(h);
    for ([_][]const u8{ "parameter_refinement", "refinement_conversion", "" }) |context| {
        const text = try std.fmt.allocPrint(testing.allocator,
            \\module {{
            \\  func.func @checked(%flag: i1) {{
            \\    "ora.refinement_guard"(%flag) <{{message = "guard"}}> {{ora.guard_id = "input", ora.verification_context = "{s}"}} : (i1) -> ()
            \\    func.return
            \\  }}
            \\}}
        , .{context});
        defer testing.allocator.free(text);
        const module = try parseModule(h.ctx, text);
        defer mlir.oraModuleDestroy(module);
        var result = try obligation_from_mlir.collect(testing.allocator, module, .{});
        defer result.deinit();
        try testing.expectEqual(@as(usize, 1), countQuery(result.set, .guard_violate));
        for (result.set.queries) |query| {
            const expected: obligation.ArtifactPolicy = if (query.kind == .guard_violate and
                std.mem.eql(u8, context, "parameter_refinement")) .runtime_checked_input else .blocks_verified_artifacts;
            try testing.expectEqual(expected, query.artifact_policy);
        }
        const dump = try dumpManifestToOwnedString(testing.allocator, result.set);
        defer testing.allocator.free(dump);
        try testing.expectEqual(
            @as(usize, if (std.mem.eql(u8, context, "parameter_refinement")) 1 else 0),
            std.mem.count(u8, dump, "\"artifact_policy\":\"runtime_checked_input\""),
        );
    }
}

test "loop witness certifies the unchanged refinement positive minimum deterministically" {
    try expectRepositoryStorageTacticDeterministic(
        "tests/formal/loop_header_refinement_positive_minimum.ora",
        "loop_header_refinement_positive_minimum",
        1,
    );
}

test "loop witness certifies narrow mixed and fallthrough environments deterministically" {
    try expectRepositoryStorageTacticDeterministic(
        "tests/formal/loop_witness_auto.ora",
        "loop_witness_auto",
        3,
    );
}

test "loop witness does not discard an unhandled mixed environment assumption" {
    try expectRepositoryLoopProofConstructionBlocked(
        "tests/formal/fail_loop_witness_mixed.ora",
        "fail_loop_witness_mixed",
        1,
        "automatic_loop_witness_rejected",
    );
}

test "loop witness retains the default nested certificate path" {
    try expectRepositoryStorageTacticDeterministic(
        "tests/formal/nested_loop_relational.ora",
        "nested_loop_relational",
        2,
    );
}

test "loop witness certifies nested inherited assumptions deterministically" {
    var result = try collectPackageObligations(
        testing.allocator,
        "tests/formal/loop_witness_nested_requires_auto.ora",
    );
    defer result.deinit();
    var inherited: usize = 0;
    for (result.set.loop_summaries) |summary| {
        if (summary.header_facts.len != 0) continue;
        inherited += 1;
        try testing.expectEqual(@as(usize, 1), summary.query_ids.induction.len);
        const query = obligation.findById(result.set.queries, summary.query_ids.induction[0]) orelse
            return error.TestUnexpectedResult;
        try testing.expectEqual(@as(usize, 2), query.assumption_ids.len);
        // Eligibility must not turn inherited conditions into local fact rows
        // or silently change the existing exact matcher.
        try testing.expect(!result.set.loopHeaderFactsMatchAssumptions(summary, query.assumption_ids));
    }
    try testing.expectEqual(@as(usize, 2), inherited);
    try expectRepositoryStorageTacticDeterministic(
        "tests/formal/loop_witness_nested_requires_auto.ora",
        "loop_witness_nested_requires_auto",
        4,
    );
}

test "loop witness retains all nested inherited assumptions" {
    try expectRepositoryLoopProofConstructionBlocked(
        "tests/formal/fail_loop_witness_nested_extra_assumption.ora",
        "fail_loop_witness_nested_extra_assumption",
        2,
        "automatic_loop_witness_rejected",
    );
}

test "loop witness contradictory bounds cannot authorize bytecode" {
    for ([_]struct { stem: []const u8, count: usize }{
        .{ .stem = "fail_loop_witness_contradiction", .count = 1 },
        .{ .stem = "fail_loop_witness_nested_contradiction", .count = 2 },
    }) |case| {
        var tmp = testing.tmpDir(.{});
        defer tmp.cleanup();
        const output = try pathFromTmpAlloc(testing.allocator, tmp, "blocked");
        defer testing.allocator.free(output);
        const source = try std.fmt.allocPrint(testing.allocator, "tests/formal/{s}.ora", .{case.stem});
        defer testing.allocator.free(source);
        const summary = try std.fmt.allocPrint(testing.allocator, "certificate-required={d}, excluded=0", .{case.count});
        defer testing.allocator.free(summary);
        const hex = try std.fmt.allocPrint(testing.allocator, "blocked/{s}.hex", .{case.stem});
        defer testing.allocator.free(hex);
        const run = try runProcessWithLeanPath(testing.allocator, &.{
            ORA_BINARY_REL,
            "emit",
            "--emit=bytecode",
            "--out-dir",
            output,
            "--lean-proofs",
            source,
        });
        defer testing.allocator.free(run.stdout);
        defer testing.allocator.free(run.stderr);
        try expectExited(run, 1);
        try testing.expect(std.mem.indexOf(u8, run.stdout, summary) != null);
        try testing.expect(std.mem.indexOf(u8, run.stdout, "automatic_loop_witness_unavailable") != null);
        try testing.expectError(error.FileNotFound, tmp.dir.access(std.testing.io, hex, .{}));
    }
}

test "unsupported loop header facts block bytecode with named exclusions" {
    std.Io.Dir.cwd().access(std.testing.io, ORA_BINARY_REL, .{}) catch |err| switch (err) {
        error.FileNotFound => return error.SkipZigTest,
        else => return err,
    };

    const allocator = testing.allocator;
    var tmp = testing.tmpDir(.{});
    defer tmp.cleanup();
    const output = try pathFromTmpAlloc(allocator, tmp, "blocked");
    defer allocator.free(output);
    const run = try runProcessWithLeanPath(allocator, &.{
        ORA_BINARY_REL,
        "emit",
        "--emit=bytecode",
        "--out-dir",
        output,
        "--lean-proofs",
        "tests/formal/loop_header_fact_projection.ora",
    });
    defer allocator.free(run.stdout);
    defer allocator.free(run.stderr);
    try expectExited(run, 1);
    try testing.expect(std.mem.containsAtLeast(
        u8,
        run.stdout,
        1,
        "excluded_by=loop_header_fact_type_unsupported",
    ));
    try testing.expect(std.mem.containsAtLeast(
        u8,
        run.stdout,
        1,
        "excluded_by=loop_header_fact_effect_unsupported",
    ));
    try testing.expectError(
        error.FileNotFound,
        tmp.dir.access(std.testing.io, "blocked/loop_header_fact_projection.hex", .{}),
    );
}

test "loop digest binding preserves unsupported assumptions as a named exclusion" {
    var result = try collectPackageObligations(testing.allocator, "ora-example/smt/verification/loop_invariants.ora");
    defer result.deinit();

    var found = false;
    for (result.set.loop_summaries) |summary| {
        if (summary.owner != .statement or
            !std.mem.eql(u8, summary.owner.statement.function_name, "accumulateToStorage")) continue;
        found = true;
        try testing.expect(loopSummaryHasReason(summary, .loop_header_fact_type_unsupported));
        try testing.expect(!summary.projectionSupported());
        try testing.expectEqual(@as(usize, 1), summary.query_ids.induction.len);
        const query = obligation.findById(
            result.set.queries,
            summary.query_ids.induction[0],
        ) orelse return error.TestUnexpectedResult;
        try testing.expectEqual(@as(?[32]u8, null), query.loop_vc_sha256);
        try testing.expectEqual(obligation.ProofRequirement.backend_result, query.proof_requirement);
        try testing.expectEqual(obligation.ArtifactPolicy.diagnostic_only, query.artifact_policy);
    }
    try testing.expect(found);
}

test "canonical LoopVC Z3 checks consume the shared edge rows" {
    var result = try collectPackageObligations(testing.allocator, "ora-example/smt/verification/loop_invariants.ora");
    defer result.deinit();

    var summary: ?obligation.LoopSummaryRow = null;
    for (result.set.loop_summaries) |row| {
        if (row.owner == .statement and
            std.mem.eql(u8, row.owner.statement.function_name, "sumToN") and
            row.owner.statement.ordinal == 4)
        {
            summary = row;
            break;
        }
    }
    const loop = summary orelse return error.TestUnexpectedResult;
    var z3_context = try z3_verification.Z3Context.init(testing.allocator);
    defer z3_context.deinit();
    var adapter = obligation_to_z3.Adapter.init(&z3_context, testing.allocator, result.set);

    var checked_count: usize = 0;
    for (result.set.queries) |query| {
        if (query.loop_summary_id != loop.id or query.loop_vc_sha256 == null or query.backend != .z3) continue;
        const checked = try adapter.checkLoopQueryForRow(query);
        try testing.expectEqual(obligation_to_z3.CheckStatus.proved, checked.status);
        try testing.expect(checked.query_hash.constraint_count > 0);
        checked_count += 1;
    }
    try testing.expectEqual(@as(usize, 5), checked_count);

    var summary_index: usize = 0;
    while (summary_index < result.set.loop_summaries.len and result.set.loop_summaries[summary_index].id != loop.id) {
        summary_index += 1;
    }
    try testing.expect(summary_index < result.set.loop_summaries.len);
    var changed_summaries = try testing.allocator.dupe(obligation.LoopSummaryRow, result.set.loop_summaries);
    defer testing.allocator.free(changed_summaries);
    changed_summaries[summary_index].query_ids.base = &.{std.math.maxInt(obligation.Id)};
    var changed_set = result.set;
    changed_set.loop_summaries = changed_summaries;
    try testing.expectError(error.InvalidDependency, changed_set.validateIdReferences());

    const safety_ids = loop.query_ids.body_safety;
    try testing.expectEqual(@as(usize, 2), safety_ids.len);
    const reversed_safety_ids = [_]obligation.Id{ safety_ids[1], safety_ids[0] };
    changed_summaries[summary_index] = loop;
    changed_summaries[summary_index].query_ids.body_safety = &reversed_safety_ids;
    try testing.expectError(error.InvalidDependency, changed_set.validateIdReferences());

    const changed_queries = try testing.allocator.dupe(obligation.VerificationQuery, result.set.queries);
    defer testing.allocator.free(changed_queries);
    for (changed_queries) |*query| {
        if (query.loop_summary_id != loop.id or query.loop_vc_sha256 == null) continue;
        query.loop_vc_sha256.?[0] ^= 1;
        break;
    }
    changed_summaries[summary_index] = loop;
    changed_set.queries = changed_queries;
    try testing.expectError(error.InvalidDependency, changed_set.validateIdReferences());

    @memcpy(changed_queries, result.set.queries);
    var changed_assumptions = false;
    for (changed_queries) |*query| {
        if (query.id != loop.query_ids.base[0]) continue;
        try testing.expect(query.assumption_ids.len > 0);
        query.assumption_ids = &.{};
        changed_assumptions = true;
        break;
    }
    try testing.expect(changed_assumptions);
    try testing.expectError(error.InvalidDependency, changed_set.validateIdReferences());

    @memcpy(changed_queries, result.set.queries);
    var changed_backend = false;
    for (changed_queries) |*query| {
        if (query.id != loop.query_ids.base[0]) continue;
        query.backend = .unspecified;
        changed_backend = true;
        break;
    }
    try testing.expect(changed_backend);
    try testing.expectError(error.InvalidDependency, changed_set.validateIdReferences());

    changed_set = result.set;
    changed_summaries[summary_index] = loop;
    changed_summaries[summary_index].invariant_formulas = &.{
        loop.edges[0].timeline[0].payload.path_assumption.formulas[0].formula,
    };
    changed_set.loop_summaries = changed_summaries;
    var stale_overlay = try obligation_from_z3.overlayPreparedQueryResults(testing.allocator, changed_set, &.{});
    defer stale_overlay.deinit();
    var saw_stale_digest = false;
    for (stale_overlay.set.diagnostics) |diagnostic| {
        if (std.mem.indexOf(u8, diagnostic.message, "loop_vc_sha256_mismatch") != null) {
            saw_stale_digest = true;
            break;
        }
    }
    try testing.expect(saw_stale_digest);
    try testing.expect(!stale_overlay.set.artifactDecision().isAllowed());
}

test "canonical LoopVC reports untouched storage roots by name" {
    const source =
        \\contract UntouchedStorageRoot {
        \\    storage var counter: u256 = 0;
        \\    storage var limit: u256 = 0;
        \\
        \\    pub fn assign()
        \\        modifies counter
        \\        ensures limit <= 10
        \\    {
        \\        var i: u256 = 0;
        \\        while (i < 1)
        \\            invariant i <= 1
        \\        {
        \\            counter = i;
        \\            i = i +% 1;
        \\        }
        \\    }
        \\}
    ;
    var compilation = try compileText(source);
    defer compilation.deinit();
    const hir_result = try compilation.db.lowerToHir(compilation.root_module_id);
    try testing.expect(hir_result.diagnostics.isEmpty());
    var result = try obligation_from_mlir.collect(
        testing.allocator,
        hir_result.module.raw_module,
        .{},
    );
    defer result.deinit();
    try testing.expectEqual(@as(usize, 1), result.set.loop_summaries.len);
    try testing.expect(result.set.loop_summaries[0].projectionSupported());

    var overlay = try obligation_from_z3.overlayPreparedQueryResults(
        testing.allocator,
        result.set,
        &.{},
    );
    defer overlay.deinit();
    var named = false;
    for (overlay.set.diagnostics) |diagnostic| {
        if (diagnostic.kind != .canonical_z3_unavailable) continue;
        if (std.mem.indexOf(u8, diagnostic.message, "loop_storage_root_untouched") == null) continue;
        try testing.expect(diagnostic.blocks_artifacts);
        named = true;
    }
    try testing.expect(named);
    try testing.expect(!overlay.set.artifactDecision().isAllowed());
}

test "scalar loop induction query projects into Lean with first-class identity" {
    var result = try collectPackageObligations(testing.allocator, "tests/formal/scalar_loop_post.ora");
    defer result.deinit();

    const query = try findLoopInductionQuery(result.set);
    const summary_id = query.loop_summary_id orelse return error.TestUnexpectedResult;
    const row = obligation.findById(result.set.loop_summaries, summary_id) orelse
        return error.TestUnexpectedResult;

    try testing.expect(row.projectionSupported());
    try testing.expectEqual(@as(usize, 1), row.context_variables.len);
    try testing.expectEqual(@as(usize, 1), row.variables.len);
    try testing.expect(row.edges[0].safetyFormulaCount() > 0);
    try testing.expectEqual(@as(usize, 1), row.post_formulas.len);
    try testing.expectEqual(obligation.ProofRequirement.lean_certificate, query.proof_requirement);
    try testing.expectEqualSlices(obligation.Id, &.{query.id}, row.query_ids.induction);
    try testing.expectEqual(@as(usize, 1), row.query_ids.post.len);
    try testing.expect(!obligation.freeVarIdEql(row.context_variables[0].id.?, row.variables[0].id.?));
    switch (obligation_to_lean.querySemanticSupport(result.set, query)) {
        .supported => {},
        .unsupported => return error.TestUnexpectedResult,
    }

    var lean = std.Io.Writer.Allocating.init(testing.allocator);
    defer lean.deinit();
    try obligation_to_lean.writeModule(&lean.writer, result.set, .{
        .namespace = "Ora.Generated.ScalarLoopProjection",
        .proof_surface = true,
    });
    const query_name = try std.fmt.allocPrint(testing.allocator, "def emittedQuery_{d} : Prop :=", .{query.id});
    defer testing.allocator.free(query_name);
    try testing.expect(std.mem.containsAtLeast(u8, lean.written(), 1, query_name));
    try testing.expect(std.mem.containsAtLeast(u8, lean.written(), 1, "Ora.Loop.loopProofAuthorizationFromAssumptions"));
    try testing.expect(std.mem.containsAtLeast(u8, lean.written(), 1, "Ora.Loop.loopVerifiedFromAssumptions"));
    try testing.expect(std.mem.containsAtLeast(u8, lean.written(), 1, "Ora.Loop.loop_authorization_sound"));
}

test "loop induction query cannot bind a different source owner" {
    var result = try collectPackageObligations(testing.allocator, "tests/formal/scalar_loop_post.ora");
    defer result.deinit();

    var mismatched = try findLoopInductionQuery(result.set);
    try testing.expect(mismatched.owner == .function);
    mismatched.owner.function.name = "different_owner";

    switch (obligation_to_lean.querySemanticSupport(result.set, mismatched)) {
        .supported => return error.TestUnexpectedResult,
        .unsupported => |reason| try testing.expectEqual(
            obligation_to_lean.SemanticUnsupportedReason.loop_summary_query_mismatch,
            reason,
        ),
    }
}

test "canonical loop edges reject missing duplicate unresolved and foreign state data" {
    var result = try collectPackageObligations(testing.allocator, "tests/formal/scalar_loop_post.ora");
    defer result.deinit();

    const query = try findLoopInductionQuery(result.set);
    const summary_id = query.loop_summary_id orelse return error.TestUnexpectedResult;
    const original = obligation.findById(result.set.loop_summaries, summary_id) orelse
        return error.TestUnexpectedResult;
    try testing.expectEqual(@as(usize, 2), original.edges.len);
    try testing.expectEqual(@as(usize, 1), original.variables.len);

    var missing_edge = original;
    missing_edge.edges = &.{};
    try expectLoopSummaryRejected(result.set, query, missing_edge, .loop_edge_missing);

    const incomplete_projections = [_]struct {
        status: obligation.LoopProjectionCompleteness,
        reason: obligation.LoopUnsupportedReason,
    }{
        .{ .status = .path_mismatch, .reason = .loop_projection_path_mismatch },
        .{ .status = .effect_mismatch, .reason = .loop_projection_effect_mismatch },
        .{ .status = .binding_mismatch, .reason = .loop_projection_binding_mismatch },
        .{ .status = .target_mismatch, .reason = .loop_projection_target_mismatch },
        .{ .status = .ambiguous_identity, .reason = .loop_projection_identity_ambiguous },
        .{ .status = .unsupported_control_flow, .reason = .loop_projection_control_flow_unsupported },
        .{ .status = .unsupported_effect, .reason = .loop_projection_effect_unsupported },
        .{ .status = .path_limit_exceeded, .reason = .loop_projection_path_limit_exceeded },
        .{ .status = .control_manifest_missing, .reason = .loop_control_manifest_missing },
        .{ .status = .control_marker_missing, .reason = .loop_control_marker_missing },
        .{ .status = .control_marker_malformed, .reason = .loop_control_marker_malformed },
        .{ .status = .control_marker_duplicate, .reason = .loop_control_marker_duplicate },
        .{ .status = .control_marker_target_mismatch, .reason = .loop_control_marker_target_mismatch },
    };
    for (incomplete_projections) |failure| {
        var incomplete = original;
        incomplete.projection_completeness = failure.status;
        try expectLoopSummaryRejected(result.set, query, incomplete, failure.reason);
    }

    var duplicate_edges = [_]obligation.LoopEdgeRow{ original.edges[0], original.edges[1] };
    duplicate_edges[1].id = 0;
    var duplicate_edge = original;
    duplicate_edge.edges = &duplicate_edges;
    try expectLoopSummaryRejected(result.set, query, duplicate_edge, .loop_duplicate_edge_identity);

    var unresolved_edges = [_]obligation.LoopEdgeRow{ original.edges[0], original.edges[1] };
    unresolved_edges[0].target = null;
    var unresolved_target = original;
    unresolved_target.edges = &unresolved_edges;
    try expectLoopSummaryRejected(result.set, query, unresolved_target, .loop_edge_target_unresolved);

    var explicit_revert_edges = [_]obligation.LoopEdgeRow{
        original.edges[0],
        original.edges[1],
        original.edges[1],
    };
    explicit_revert_edges[2].id = 2;
    explicit_revert_edges[2].structural_key = "body/explicit-revert";
    explicit_revert_edges[2].target = .explicit_revert;
    var explicit_revert = original;
    explicit_revert.edges = &explicit_revert_edges;
    try expectLoopSummaryRejected(
        result.set,
        query,
        explicit_revert,
        .loop_control_target_unsupported,
    );

    var missing_binding_edges = [_]obligation.LoopEdgeRow{ original.edges[0], original.edges[1] };
    missing_binding_edges[0].next_state = &.{};
    var missing_binding = original;
    missing_binding.edges = &missing_binding_edges;
    try expectLoopSummaryRejected(result.set, query, missing_binding, .loop_state_binding_missing);

    var missing_source_bindings = [_]obligation.LoopStepAssignment{original.edges[0].next_state[0]};
    missing_source_bindings[0].source_key = "";
    var missing_source_edges = [_]obligation.LoopEdgeRow{ original.edges[0], original.edges[1] };
    missing_source_edges[0].next_state = &missing_source_bindings;
    var missing_source = original;
    missing_source.edges = &missing_source_edges;
    try expectLoopSummaryRejected(result.set, query, missing_source, .loop_state_binding_source_missing);

    var extra_bindings = [_]obligation.LoopStepAssignment{
        original.edges[0].next_state[0],
        original.edges[0].next_state[0],
    };
    extra_bindings[1].variable_index = 1;
    var extra_binding_edges = [_]obligation.LoopEdgeRow{ original.edges[0], original.edges[1] };
    extra_binding_edges[0].next_state = &extra_bindings;
    var extra_binding = original;
    extra_binding.edges = &extra_binding_edges;
    try expectLoopSummaryRejected(result.set, query, extra_binding, .loop_state_binding_extra);

    var foreign_bindings = [_]obligation.LoopStepAssignment{original.edges[0].next_state[0]};
    foreign_bindings[0].target.?.pattern_id +%= 1;
    var foreign_binding_edges = [_]obligation.LoopEdgeRow{ original.edges[0], original.edges[1] };
    foreign_binding_edges[0].next_state = &foreign_bindings;
    var foreign_binding = original;
    foreign_binding.edges = &foreign_binding_edges;
    try expectLoopSummaryRejected(result.set, query, foreign_binding, .loop_state_binding_foreign);

    var empty_state = original;
    empty_state.variables = &.{};
    try expectLoopSummaryRejected(result.set, query, empty_state, .loop_state_empty);

    var typed_terms = try testing.allocator.alloc(obligation.Term, result.set.terms.len + 3);
    defer testing.allocator.free(typed_terms);
    @memcpy(typed_terms[0..result.set.terms.len], result.set.terms);
    typed_terms[result.set.terms.len] = .{ .int_lit = .{ .value = "0", .ty = .{ .spelling = "not-an-integer" } } };
    typed_terms[result.set.terms.len + 1] = .{ .int_lit = .{ .value = "0", .ty = .{ .compiler_type_id = 1 } } };
    typed_terms[result.set.terms.len + 2] = .{ .int_lit = .{ .value = "0", .ty = .{ .compiler_type_id = 12 } } };
    var typed_set = result.set;
    typed_set.terms = typed_terms;
    const typed_reasons = [_]obligation.LoopUnsupportedReason{
        .loop_value_type_unresolved,
        .loop_value_width_mismatch,
        .loop_value_signedness_mismatch,
    };
    for (typed_reasons, 0..) |reason, index| {
        var typed_bindings = [_]obligation.LoopStepAssignment{original.edges[0].next_state[0]};
        typed_bindings[0].value = .{ .term = @intCast(result.set.terms.len + index) };
        var typed_edges = [_]obligation.LoopEdgeRow{ original.edges[0], original.edges[1] };
        typed_edges[0].next_state = &typed_bindings;
        var typed_row = original;
        typed_row.edges = &typed_edges;
        try expectLoopSummaryRejected(typed_set, query, typed_row, reason);
    }
}

test "canonical loop edges reject duplicate state bindings" {
    var result = try collectPackageObligations(testing.allocator, "ora-example/smt/verification/loop_invariants.ora");
    defer result.deinit();

    var query: ?obligation.VerificationQuery = null;
    var original: ?obligation.LoopSummaryRow = null;
    for (result.set.loop_summaries) |row| {
        if (row.variables.len != 2 or row.query_ids.induction.len != 1) continue;
        original = row;
        query = obligation.findById(result.set.queries, row.query_ids.induction[0]);
        break;
    }
    const row = original orelse return error.TestUnexpectedResult;
    const induction = query orelse return error.TestUnexpectedResult;
    var duplicate_bindings = [_]obligation.LoopStepAssignment{
        row.edges[0].next_state[0],
        row.edges[0].next_state[1],
    };
    duplicate_bindings[1].target = duplicate_bindings[0].target;
    var edges = [_]obligation.LoopEdgeRow{ row.edges[0], row.edges[1] };
    edges[0].next_state = &duplicate_bindings;
    var mutated = row;
    mutated.edges = &edges;
    try expectLoopSummaryRejected(result.set, induction, mutated, .loop_state_binding_duplicate);
}

test "canonical LoopVC reports every remaining structural schema failure by name" {
    var scalar_result = try collectPackageObligations(testing.allocator, "tests/formal/scalar_loop_post.ora");
    defer scalar_result.deinit();
    const scalar_query = try findLoopInductionQuery(scalar_result.set);
    const scalar = obligation.findById(
        scalar_result.set.loop_summaries,
        scalar_query.loop_summary_id.?,
    ) orelse return error.TestUnexpectedResult;

    var edges = [_]obligation.LoopEdgeRow{ scalar.edges[0], scalar.edges[1] };
    edges[0].id = 7;
    var mutated = scalar;
    mutated.edges = &edges;
    try expectLoopSummaryRejected(scalar_result.set, scalar_query, mutated, .loop_edge_order_invalid);

    const no_path_timeline = try testing.allocator.dupe(
        obligation.LoopTimelineEvent,
        scalar.edges[0].timeline[1..],
    );
    defer testing.allocator.free(no_path_timeline);
    for (no_path_timeline, 0..) |*event, index| event.id = @intCast(index);
    edges = .{ scalar.edges[0], scalar.edges[1] };
    edges[0].timeline = no_path_timeline;
    mutated = scalar;
    mutated.edges = &edges;
    try expectLoopSummaryRejected(scalar_result.set, scalar_query, mutated, .loop_timeline_path_missing);

    edges = .{ scalar.edges[0], scalar.edges[1] };
    edges[0].target = .successor;
    mutated = scalar;
    mutated.edges = &edges;
    try expectLoopSummaryRejected(scalar_result.set, scalar_query, mutated, .loop_backedge_missing);

    edges = .{ scalar.edges[0], scalar.edges[1] };
    edges[1].target = .backedge;
    mutated = scalar;
    mutated.edges = &edges;
    try expectLoopSummaryRejected(scalar_result.set, scalar_query, mutated, .loop_exit_edge_missing);

    var bindings = [_]obligation.LoopStepAssignment{scalar.edges[0].next_state[0]};
    bindings[0].variable_index = 1;
    edges = .{ scalar.edges[0], scalar.edges[1] };
    edges[0].next_state = &bindings;
    mutated = scalar;
    mutated.edges = &edges;
    try expectLoopSummaryRejected(scalar_result.set, scalar_query, mutated, .loop_state_binding_invalid);

    var multi_result = try collectPackageObligations(
        testing.allocator,
        "ora-example/smt/verification/loop_invariants.ora",
    );
    defer multi_result.deinit();
    var multi: ?obligation.LoopSummaryRow = null;
    var multi_query: ?obligation.VerificationQuery = null;
    for (multi_result.set.loop_summaries) |summary| {
        if (summary.variables.len != 2 or summary.query_ids.induction.len != 1) continue;
        multi = summary;
        multi_query = obligation.findById(
            multi_result.set.queries,
            summary.query_ids.induction[0],
        );
        break;
    }
    const two_variable = multi orelse return error.TestUnexpectedResult;
    const two_variable_query = multi_query orelse return error.TestUnexpectedResult;
    var variables = [_]obligation.LoopVariable{
        two_variable.variables[0],
        two_variable.variables[1],
    };
    variables[1].id = variables[0].id;
    var duplicate_state = two_variable;
    duplicate_state.variables = &variables;
    try expectLoopSummaryRejected(
        multi_result.set,
        two_variable_query,
        duplicate_state,
        .loop_duplicate_state_identity,
    );
}

test "source accounting accepts concretely checked folded countThree invariants" {
    const path = "ora-example/corpus/declarations/inline_functions.ora";
    var compilation = try compilePackage(path);
    defer compilation.deinit();

    const const_eval = try compilation.db.constEval(compilation.root_module_id);

    var package_inventory = try source_accounting_from_package.collect(
        testing.allocator,
        &compilation.db,
        compilation.package_id,
        compilation.root_module_id,
    );
    defer package_inventory.deinit();
    const module_inventory = package_inventory.module(compilation.root_module_id) orelse
        return error.TestUnexpectedResult;
    const comptime_template_id = module_inventory.templateForOwner(
        "module/contract:InlineFunctions/function:countThree",
        .comptime_body,
    ) orelse return error.TestUnexpectedResult;
    const comptime_template = package_inventory.template(comptime_template_id) orelse
        return error.TestUnexpectedResult;
    var comptime_inventory = try source_accounting_from_comptime.collectPackageFoldExpansions(
        testing.allocator,
        &package_inventory,
        const_eval,
        .{},
    );
    defer comptime_inventory.deinit();

    try testing.expectEqual(@as(usize, 1), package_inventory.inventory.declared_sites.len);
    try testing.expectEqual(source_accounting.SourceFactKind.loop_invariant, package_inventory.inventory.declared_sites[0].key.kind);
    try testing.expect(std.mem.indexOf(u8, package_inventory.inventory.declared_sites[0].key.owner, "function:countThree") != null);
    try testing.expectEqual(@as(usize, 1), package_inventory.inventory.typed_sites.len);
    try testing.expect(comptime_template.control_nodes.len > 3);
    try testing.expect(comptime_template.control_edges.len > 1);
    try testing.expectEqual(@as(?u32, 0), comptime_template.entry_slot);
    var saw_invariant_checkpoint = false;
    for (comptime_template.control_nodes) |node| {
        if (node.kind == .loop_head and node.attached_use_ordinals.len == 2) saw_invariant_checkpoint = true;
    }
    try testing.expect(saw_invariant_checkpoint);
    try testing.expectEqual(@as(usize, 1), comptime_inventory.expansions.len);
    try testing.expectEqual(source_accounting.ExpansionDisposition.fold_committed, comptime_inventory.expansions[0].disposition);
    try testing.expectEqual(@as(usize, 2), comptime_inventory.uses.len);
    try testing.expectEqual(comptime_template.control_nodes.len, comptime_inventory.control_nodes.len);
    try testing.expectEqual(comptime_template.control_edges.len, comptime_inventory.control_edges.len);
    try testing.expectEqual(@as(usize, 1), comptime_inventory.evidence.folds.len);
    try testing.expectEqual(@as(usize, 8), comptime_inventory.evidence.predicate_events.len);
    try testing.expectEqual(@as(usize, 2), comptime_inventory.evidence.handlings.len);

    // Consume the real downstream obligation/query surface. It is empty for
    // this invariant because the current fold erased it before formal MLIR.
    const hir_result = try compilation.db.lowerToHir(compilation.root_module_id);
    try testing.expect(hir_result.diagnostics.isEmpty());
    var formal = try obligation_from_mlir.collect(testing.allocator, hir_result.module.raw_module, .{});
    defer formal.deinit();
    const no_bindings: []const source_accounting_from_mlir.Binding = &.{};
    var mlir_evidence = try source_accounting_from_mlir.collect(
        testing.allocator,
        formal.set,
        no_bindings,
        .{},
    );
    defer mlir_evidence.deinit();
    var prepared_evidence = try source_accounting_from_z3.bindPreparedQueries(
        testing.allocator,
        formal.set,
        no_bindings,
        mlir_evidence.evidence,
        &.{},
    );
    defer prepared_evidence.deinit();

    const manifest: source_accounting.Manifest = .{
        .inventory = .{
            .declared_sites = package_inventory.inventory.declared_sites,
            .typed_sites = package_inventory.inventory.typed_sites,
            .owner_templates = package_inventory.inventory.owner_templates,
            .expansions = comptime_inventory.expansions,
            .uses = comptime_inventory.uses,
            .control_nodes = comptime_inventory.control_nodes,
            .control_edges = comptime_inventory.control_edges,
        },
        .comptime_evidence = comptime_inventory.evidence,
        .symbolic = prepared_evidence.evidence,
    };
    var decision = try source_accounting_gate.decide(testing.allocator, .verified_full, manifest);
    defer decision.deinit();
    try testing.expectEqual(source_accounting_gate.Decision.accepted, decision.decision);
    try testing.expectEqual(@as(?source_accounting_gate.FailureCode, null), decision.primary_failure);

    // The folded loop is discharged concretely, so no symbolic loop row or
    // query should survive into the downstream obligation manifest.
    try testing.expectEqual(@as(usize, 0), formal.set.loop_summaries.len);
    for (formal.set.queries) |query| switch (query.kind) {
        .loop_induction, .loop_invariant_step, .loop_body_safety, .loop_invariant_post => return error.TestUnexpectedResult,
        else => {},
    };
}

test "source accounting globalizes imported module sites deterministically" {
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();

    try tmp.dir.writeFile(std.testing.io, .{
        .sub_path = "dep.ora",
        .data =
        \\pub fn helper(value: u256) -> u256
        \\    requires value < 10
        \\    ensures result == value
        \\{
        \\    return value;
        \\}
        ,
    });
    try tmp.dir.writeFile(std.testing.io, .{
        .sub_path = "main.ora",
        .data =
        \\comptime const dep = @import("./dep.ora");
        \\
        \\contract Main {
        \\    invariant true;
        \\
        \\    pub fn run(value: u256) -> u256 {
        \\        return dep.helper(value);
        \\    }
        \\}
        ,
    });

    const root_path = try std.fmt.allocPrint(testing.allocator, ".zig-cache/tmp/{s}/main.ora", .{tmp.sub_path});
    defer testing.allocator.free(root_path);
    var compilation = try compilePackage(root_path);
    defer compilation.deinit();

    var first = try source_accounting_from_package.collect(
        testing.allocator,
        &compilation.db,
        compilation.package_id,
        compilation.root_module_id,
    );
    defer first.deinit();
    var second = try source_accounting_from_package.collect(
        testing.allocator,
        &compilation.db,
        compilation.package_id,
        compilation.root_module_id,
    );
    defer second.deinit();

    try testing.expectEqual(@as(usize, 2), first.modules.len);
    try testing.expectEqual(@as(usize, 3), first.inventory.declared_sites.len);
    try testing.expectEqual(first.inventory.declared_sites.len, first.inventory.typed_sites.len);
    try testing.expectEqual(first.inventory.owner_templates.len, second.inventory.owner_templates.len);
    for (first.inventory.declared_sites, second.inventory.declared_sites) |lhs, rhs| {
        try testing.expectEqual(lhs.id, rhs.id);
        try testing.expectEqualStrings(lhs.key.path, rhs.key.path);
        try testing.expectEqualStrings(lhs.key.owner, rhs.key.owner);
        try testing.expectEqual(lhs.key.range_start, rhs.key.range_start);
        try testing.expectEqual(lhs.key.range_end, rhs.key.range_end);
        try testing.expectEqual(lhs.key.kind, rhs.key.kind);
    }

    try testing.expectEqualStrings("dep.ora", first.modules[0].canonical_path);
    try testing.expectEqualStrings("main.ora", first.modules[1].canonical_path);
    try testing.expectEqual(source_accounting.SourceFactKind.requires, first.inventory.declared_sites[0].key.kind);
    try testing.expectEqual(source_accounting.SourceFactKind.ensures, first.inventory.declared_sites[1].key.kind);
    try testing.expectEqual(source_accounting.SourceFactKind.contract_invariant, first.inventory.declared_sites[2].key.kind);
    for (first.modules) |module_inventory| for (module_inventory.site_bindings) |binding| {
        const source_fact_id = binding.source_fact_id orelse return error.TestUnexpectedResult;
        try testing.expectEqual(binding.site_id, module_inventory.siteForSourceFact(source_fact_id, binding.kind).?);
    };
    for (first.inventory.owner_templates) |template| {
        try testing.expect(std.mem.indexOf(u8, template.owner_key, ".ora::module/") != null);
    }
}

test "source accounting gives imported comptime folds deterministic package identities" {
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();

    try tmp.dir.writeFile(std.testing.io, .{
        .sub_path = "dep.ora",
        .data =
        \\pub inline fn count(limit: u256) -> u256 {
        \\    var i: u256 = 0;
        \\    while (i < limit)
        \\        invariant i <= limit
        \\    {
        \\        i = i + 1;
        \\    }
        \\    return i;
        \\}
        ,
    });
    try tmp.dir.writeFile(std.testing.io, .{
        .sub_path = "main.ora",
        .data =
        \\comptime const dep = @import("./dep.ora");
        \\
        \\contract Main {
        \\    pub fn run() -> u256 { return dep.count(2); }
        \\}
        ,
    });

    const root_path = try std.fmt.allocPrint(testing.allocator, ".zig-cache/tmp/{s}/main.ora", .{tmp.sub_path});
    defer testing.allocator.free(root_path);
    var compilation = try compilePackage(root_path);
    defer compilation.deinit();
    const const_eval = try compilation.db.constEval(compilation.root_module_id);
    try testing.expect(const_eval.diagnostics.isEmpty());

    var package_inventory = try source_accounting_from_package.collect(
        testing.allocator,
        &compilation.db,
        compilation.package_id,
        compilation.root_module_id,
    );
    defer package_inventory.deinit();
    var first = try source_accounting_from_comptime.collectPackageFoldExpansions(
        testing.allocator,
        &package_inventory,
        const_eval,
        .{},
    );
    defer first.deinit();
    var second = try source_accounting_from_comptime.collectPackageFoldExpansions(
        testing.allocator,
        &package_inventory,
        const_eval,
        .{},
    );
    defer second.deinit();

    try testing.expectEqual(@as(usize, 1), first.expansions.len);
    const expansion = first.expansions[0];
    try testing.expectEqual(source_accounting.ExpansionDisposition.fold_committed, expansion.disposition);
    try testing.expectEqualStrings("dep.ora", expansion.imported_module orelse return error.TestUnexpectedResult);
    try testing.expectEqualStrings("main.ora::module/contract:Main/function:run", expansion.root_runtime_owner);
    try testing.expectEqual(@as(usize, 1), expansion.folded_call_site_chain.len);
    try testing.expectEqualStrings("main.ora", expansion.folded_call_site_chain[0].file);
    try testing.expectEqual(@as(usize, 1), expansion.generic_bindings.len);
    try testing.expect(std.mem.indexOf(u8, expansion.generic_bindings[0], "limit") != null);
    try testing.expect(std.mem.indexOf(u8, expansion.generic_bindings[0], "integer:2") != null);
    try testing.expectEqualStrings(expansion.identity, second.expansions[0].identity);
    try testing.expectEqualStrings(expansion.root_runtime_owner, second.expansions[0].root_runtime_owner);
}

test "source accounting transfers an abandoned fold only to its symbolic boundary" {
    const source_text =
        \\contract SpeculativeHavoc {
        \\    inline fn unknown() -> u256 {
        \\        var value: u256 = 0;
        \\        havoc value;
        \\        return value;
        \\    }
        \\    pub fn run() -> u256 { return unknown(); }
        \\}
    ;
    var compilation = try compileText(source_text);
    defer compilation.deinit();
    const const_eval = try compilation.db.constEval(compilation.root_module_id);
    try testing.expect(const_eval.diagnostics.isEmpty());
    try testing.expectEqual(@as(usize, 1), const_eval.formal_folds.len);
    try testing.expectEqual(compiler.sema.ComptimeFoldDisposition.abandoned, const_eval.formal_folds[0].disposition);

    var package_inventory = try source_accounting_from_package.collect(
        testing.allocator,
        &compilation.db,
        compilation.package_id,
        compilation.root_module_id,
    );
    defer package_inventory.deinit();
    var comptime_inventory = try source_accounting_from_comptime.collectPackageFoldExpansions(
        testing.allocator,
        &package_inventory,
        const_eval,
        .{},
    );
    defer comptime_inventory.deinit();

    try testing.expectEqual(@as(usize, 2), comptime_inventory.expansions.len);
    const abandoned = comptime_inventory.expansions[0];
    const replacement = comptime_inventory.expansions[1];
    try testing.expectEqual(source_accounting.ExpansionDisposition.fold_abandoned_to_symbolic, abandoned.disposition);
    try testing.expectEqual(@as(?source_accounting.ExpansionId, replacement.id), abandoned.replacement_expansion_id);
    try testing.expectEqual(source_accounting.ActivationReason.symbolic_call, replacement.activation);
    try testing.expectEqual(source_accounting.ExpansionDisposition.symbolic, replacement.disposition);
    try testing.expectEqualStrings(abandoned.identity, replacement.identity);
    try testing.expectEqualStrings(abandoned.root_runtime_owner, replacement.root_runtime_owner);
    try testing.expectEqual(@as(usize, 0), comptime_inventory.uses.len);
    try testing.expectEqual(@as(usize, 0), comptime_inventory.control_nodes.len);
    try testing.expectEqual(@as(usize, 0), comptime_inventory.control_edges.len);
    try testing.expectEqual(@as(usize, 0), comptime_inventory.evidence.folds.len);
    try testing.expectEqual(@as(usize, 0), comptime_inventory.evidence.predicate_events.len);
    try testing.expectEqual(@as(usize, 0), comptime_inventory.evidence.handlings.len);
}

test "source-accounting syntax and spec-clause vocabularies are totality pins" {
    try testing.expectEqual(@as(usize, 93), @typeInfo(compiler.syntax.SyntaxKind).@"enum".field_names.len);
    try testing.expectEqual(@as(usize, 7), @typeInfo(compiler.ast.SpecClauseKind).@"enum".field_names.len);
    inline for (@typeInfo(compiler.ast.SpecClauseKind).@"enum".field_names) |field| {
        const clause: compiler.ast.SpecClauseKind = @fromBackingInt(@intCast(@backingInt(@field(compiler.ast.SpecClauseKind, field))));
        const classification: ?source_accounting.SourceFactKind = switch (clause) {
            .requires => .requires,
            .guard => .guard,
            .ensures => .ensures,
            .ensures_ok => .ensures_ok,
            .ensures_err => .ensures_err,
            .modifies => .modifies,
            // The AST tag is also used for owner invariants, whose concrete
            // source node is inventoried separately as ContractInvariantItem
            // or InvariantClause.
            .invariant => null,
        };
        _ = classification;
    }
}

test "source-accounting templates distinguish contract and error exit uses" {
    const source =
        \\error Rejected;
        \\
        \\contract ExitAccounting {
        \\    invariant stable(true);
        \\
        \\    pub fn choose(flag: bool, value: u256) -> !u256 | Rejected
        \\        ensures_err(value == value)
        \\    {
        \\        if (!flag) {
        \\            return Rejected;
        \\        }
        \\        return value;
        \\    }
        \\}
    ;
    var compilation = try compileText(source);
    defer compilation.deinit();
    const module = compilation.db.sources.module(compilation.root_module_id);
    const tree = try compilation.db.syntaxTree(module.file_id);
    const ast_file = try compilation.db.astFile(module.file_id);
    var syntax_inventory = try source_accounting_from_syntax.collect(
        testing.allocator,
        compiler.syntax.rootNode(tree),
        "test.ora",
    );
    defer syntax_inventory.deinit();
    var sema_inventory = try source_accounting_from_sema.collect(
        testing.allocator,
        ast_file,
        "test.ora",
        syntax_inventory.declared_sites,
    );
    defer sema_inventory.deinit();

    var contract_site_id: ?source_accounting.SiteId = null;
    var ensures_err_site_id: ?source_accounting.SiteId = null;
    for (sema_inventory.typed_sites) |site| switch (site.kind) {
        .contract_invariant => contract_site_id = site.id,
        .ensures_err => ensures_err_site_id = site.id,
        else => {},
    };
    const contract_id = contract_site_id orelse return error.TestUnexpectedResult;
    const ensures_err_id = ensures_err_site_id orelse return error.TestUnexpectedResult;
    var runtime_template: ?source_accounting.OwnerTemplate = null;
    var call_template: ?source_accounting.OwnerTemplate = null;
    var comptime_template: ?source_accounting.OwnerTemplate = null;
    for (sema_inventory.owner_templates) |template| {
        if (!std.mem.endsWith(u8, template.owner_key, "/function:choose")) continue;
        switch (template.activation) {
            .runtime_body => runtime_template = template,
            .symbolic_call_boundary => call_template = template,
            .comptime_body => comptime_template = template,
        }
    }
    const template = runtime_template orelse return error.TestUnexpectedResult;
    var contract_success_targets: usize = 0;
    var contract_error_targets: usize = 0;
    var contract_entry_contexts: usize = 0;
    var ensures_err_error_uses: usize = 0;
    for (template.uses) |use| {
        if (use.site_id == contract_id and use.role == .proof_target and use.control_node_slot == 1) contract_success_targets += 1;
        if (use.site_id == contract_id and use.role == .proof_target and use.control_node_slot == 2) contract_error_targets += 1;
        if (use.site_id == contract_id and use.role == .assumption_context and use.control_node_slot == 0) contract_entry_contexts += 1;
        if (use.site_id == ensures_err_id and use.control_node_slot == 2) ensures_err_error_uses += 1;
    }
    try testing.expectEqual(@as(usize, 1), contract_success_targets);
    try testing.expectEqual(@as(usize, 1), contract_error_targets);
    try testing.expectEqual(@as(usize, 1), contract_entry_contexts);
    try testing.expectEqual(@as(usize, 1), ensures_err_error_uses);

    const boundary = call_template orelse return error.TestUnexpectedResult;
    try testing.expectEqual(@as(usize, 1), countTemplateUses(boundary, ensures_err_id, .assumption_context));
    try testing.expectEqual(@as(usize, 0), countTemplateUses(boundary, contract_id, .proof_target));

    const concrete = comptime_template orelse return error.TestUnexpectedResult;
    try testing.expectEqual(@as(usize, 1), countTemplateUses(concrete, ensures_err_id, .proof_target));
    try testing.expectEqual(@as(usize, 1), countTemplateUses(concrete, ensures_err_id, .assumption_context));
}

fn countTemplateUses(
    template: source_accounting.OwnerTemplate,
    site_id: source_accounting.SiteId,
    role: source_accounting.UseRole,
) usize {
    var count: usize = 0;
    for (template.uses) |use| count += @intFromBool(use.site_id == site_id and use.role == role);
    return count;
}

test "HIR formal operations retain structural source-fact and runtime-owner identities" {
    const source =
        \\pub fn checked(limit: u256) -> u256
        \\    requires limit > 0
        \\    ensures result > 0
        \\{
        \\    var i: u256 = 0;
        \\    while (i < limit)
        \\        invariant i <= limit
        \\    {
        \\        i = i + 1;
        \\    }
        \\    return limit;
        \\}
    ;
    var compilation = try compileText(source);
    defer compilation.deinit();
    const hir_result = try compilation.db.lowerToHir(compilation.root_module_id);
    try testing.expect(hir_result.diagnostics.isEmpty());
    const hir_text = try hir_result.renderText(testing.allocator);
    defer testing.allocator.free(hir_text);

    try testing.expect(std.mem.containsAtLeast(u8, hir_text, 1, "ora.source_module_path"));
    try testing.expect(std.mem.containsAtLeast(u8, hir_text, 1, "ora.source_owner_key = \"module/function:checked\""));
    try testing.expect(std.mem.containsAtLeast(u8, hir_text, 1, "ora.source_template_activation = \"runtime_body\""));
    try testing.expect(std.mem.containsAtLeast(u8, hir_text, 3, "ora.source_fact_id"));
    try testing.expect(std.mem.containsAtLeast(u8, hir_text, 1, "ora.source_fact_kind = \"requires\""));
    try testing.expect(std.mem.containsAtLeast(u8, hir_text, 1, "ora.source_fact_kind = \"ensures\""));
    try testing.expect(std.mem.containsAtLeast(u8, hir_text, 1, "ora.source_fact_kind = \"loop_invariant\""));
    try testing.expect(std.mem.containsAtLeast(u8, hir_text, 1, "\"proof_target\""));
    try testing.expect(std.mem.containsAtLeast(u8, hir_text, 1, "\"assumption_context\""));
    try testing.expect(std.mem.containsAtLeast(u8, hir_text, 1, "\"runtime_condition\""));

    var formal = try obligation_from_mlir.collect(testing.allocator, hir_result.module.raw_module, .{});
    defer formal.deinit();
    try testing.expectEqual(@as(usize, 1), formal.runtime_owner_bindings.len);
    try testing.expectEqual(@as(usize, 1), formal.runtime_check_producers.len);
    const runtime_owner = formal.runtime_owner_bindings[0];
    try testing.expectEqualStrings("test.ora", runtime_owner.module_path);
    try testing.expectEqualStrings("module/function:checked", runtime_owner.owner_key);
    try testing.expectEqualStrings("runtime_body", runtime_owner.template_activation);
    try testing.expectEqual(@as(usize, 0), runtime_owner.specialization_bindings.len);
    try testing.expect(runtime_owner.trait_implementation == null);
    try testing.expect(runtime_owner.trait_method == null);
    try testing.expectEqual(@as(usize, 4), formal.source_fact_bindings.len);
    var requires_fact_id: ?u32 = null;
    var requires_binding_count: usize = 0;
    var saw_runtime_requires = false;
    var runtime_requires_id: ?u32 = null;
    for (formal.source_fact_bindings) |binding| {
        try testing.expectEqualStrings("module/function:checked", binding.owner_key orelse return error.TestUnexpectedResult);
        try testing.expectEqualStrings("runtime_body", binding.template_activation orelse return error.TestUnexpectedResult);
        if (std.mem.eql(u8, binding.kind, "requires")) {
            requires_binding_count += 1;
            if (requires_fact_id) |expected| {
                try testing.expectEqual(expected, binding.source_fact_id);
            } else requires_fact_id = binding.source_fact_id;
            saw_runtime_requires = saw_runtime_requires or binding.runtime_check_present;
            if (binding.runtime_check_ids.len != 0) {
                try testing.expectEqual(@as(usize, 1), binding.runtime_check_ids.len);
                runtime_requires_id = binding.runtime_check_ids[0];
            }
        }
    }
    try testing.expectEqual(@as(usize, 2), requires_binding_count);
    try testing.expect(saw_runtime_requires);
    try testing.expectEqual(formal.runtime_check_producers[0].id, runtime_requires_id.?);

    var package_inventory = try source_accounting_from_package.collect(
        testing.allocator,
        &compilation.db,
        compilation.package_id,
        compilation.root_module_id,
    );
    defer package_inventory.deinit();
    var runtime_inventory = try source_accounting_from_runtime.collect(
        testing.allocator,
        &package_inventory,
        formal.runtime_owner_bindings,
        .{},
    );
    defer runtime_inventory.deinit();
    try testing.expectEqual(@as(usize, 1), runtime_inventory.expansions.len);
    try testing.expectEqual(@as(usize, 5), runtime_inventory.uses.len);
    try testing.expectEqual(@as(usize, 1), runtime_inventory.bindings.len);
    try testing.expectEqual(
        runtime_inventory.expansions[0].id,
        runtime_inventory.expansionForOwner(
            "test.ora",
            "module/function:checked",
            .runtime_body,
            "checked",
            &.{},
            null,
            null,
        ).?,
    );
    try testing.expectEqual(
        package_inventory.inventory.owner_templates[0].control_nodes.len,
        runtime_inventory.control_nodes.len,
    );
    try testing.expectEqual(
        package_inventory.inventory.owner_templates[0].control_edges.len,
        runtime_inventory.control_edges.len,
    );

    var symbolic_bindings = try source_accounting_from_runtime.collectSymbolicBindings(
        testing.allocator,
        &package_inventory,
        &runtime_inventory,
        &formal,
    );
    defer symbolic_bindings.deinit();
    try testing.expectEqual(@as(usize, 5), symbolic_bindings.bindings.len);
    var mlir_evidence = try source_accounting_from_mlir.collect(
        testing.allocator,
        formal.set,
        symbolic_bindings.bindings,
        symbolic_bindings.producers,
    );
    defer mlir_evidence.deinit();
    const prepared_queries = try testing.allocator.alloc(
        source_accounting_from_z3.PreparedQueryIdentity,
        formal.set.queries.len,
    );
    defer testing.allocator.free(prepared_queries);
    for (formal.set.queries, prepared_queries, 0..) |query, *prepared, index| {
        prepared.* = .{
            .producer_id = @intCast(index + 1),
            .formal_query_id = query.id,
            .kind = switch (query.kind) {
                .base, .obligation => .obligation,
                .loop_base, .loop_induction => .other,
                .loop_invariant_step => .loop_invariant_step,
                .loop_body_safety => .loop_body_safety,
                .loop_invariant_post => .loop_invariant_post,
                .guard_satisfy => .guard_satisfy,
                .guard_violate => .guard_violate,
            },
        };
    }
    var prepared_evidence = try source_accounting_from_z3.bindPreparedQueries(
        testing.allocator,
        formal.set,
        symbolic_bindings.bindings,
        mlir_evidence.evidence,
        prepared_queries,
    );
    defer prepared_evidence.deinit();
    const manifest: source_accounting.Manifest = .{
        .inventory = .{
            .declared_sites = package_inventory.inventory.declared_sites,
            .typed_sites = package_inventory.inventory.typed_sites,
            .owner_templates = package_inventory.inventory.owner_templates,
            .expansions = runtime_inventory.expansions,
            .uses = runtime_inventory.uses,
            .control_nodes = runtime_inventory.control_nodes,
            .control_edges = runtime_inventory.control_edges,
        },
        .symbolic = prepared_evidence.evidence,
    };
    var decision = try source_accounting_gate.decide(testing.allocator, .verified_full, manifest);
    defer decision.deinit();
    try testing.expectEqual(source_accounting_gate.Decision.rejected, decision.decision);
    try testing.expectEqual(
        @as(?source_accounting_gate.FailureCode, .missing_symbolic_query),
        decision.primary_failure,
    );
}

fn c1Identity(manifest: z3_verification.PreparedQueryManifest) !source_accounting_from_z3.PreparedIdentity {
    var identity = try source_accounting_from_z3.collectPreparedIdentity(testing.allocator, manifest.source_accounting_rows);
    errdefer identity.deinit(testing.allocator);
    if (manifest.inactive_runtime_owners) |rows| {
        const owners = try testing.allocator.alloc(source_accounting_from_z3.InactiveRuntimeOwner, rows.len);
        identity.inactive_runtime_owners = owners;
        for (rows, owners) |row, *owner| owner.* = .{ .source_op_id = row.source_op_id, .module_path = row.module_path, .owner_key = row.owner_key };
    }
    return identity;
}

fn c1Account(compilation: *compiler.driver.Compilation, formal: *const obligation_from_mlir.CollectResult, identity: source_accounting_from_z3.PreparedIdentityView) !source_accounting_pipeline.Result {
    return source_accounting_pipeline.run(testing.allocator, &compilation.db, compilation.package_id, compilation.root_module_id, try compilation.db.constEval(compilation.root_module_id), formal, .verified_full, identity);
}

test "C1 inactive folds require exact reachability evidence and retain missing live queries" {
    for ([_]bool{ false, true }) |live| {
        const source = try std.fmt.allocPrint(testing.allocator,
            \\contract C {{
            \\ storage var n: u256;
            \\ pub fn callee() -> u256 requires n > 0 {{ return n; }}
            \\ {s} fn caller() -> u256 {s} {{ return callee(); }}
            \\ pub fn run() -> u256 {{ return 1; }}
            \\}}
        , .{ if (live) "pub" else "", if (live) "requires n > 0" else "" });
        defer testing.allocator.free(source);
        var tmp = testing.tmpDir(.{});
        defer tmp.cleanup();
        try tmp.dir.writeFile(std.testing.io, .{ .sub_path = "main.ora", .data = source });
        const root_path = try std.fmt.allocPrint(testing.allocator, ".zig-cache/tmp/{s}/main.ora", .{tmp.sub_path});
        defer testing.allocator.free(root_path);
        var compilation = try compilePackage(root_path);
        defer compilation.deinit();
        const hir = try compilation.db.lowerToHir(compilation.root_module_id);
        try testing.expect(hir.diagnostics.isEmpty());
        var formal = try obligation_from_mlir.collect(testing.allocator, hir.module.raw_module, .{});
        defer formal.deinit();
        var pass = try z3_verification.VerificationPass.init(testing.allocator);
        defer pass.deinit();
        const bindings = try test_helpers.z3FormalQueryBindingsFromFormalForTest(testing.allocator, formal.query_bindings);
        defer testing.allocator.free(bindings);
        pass.setFormalQueryBindings(bindings);
        var manifest = try pass.collectPreparedQueryManifest(hir.module.raw_module);
        defer manifest.deinit();
        try testing.expectEqual(@as(usize, if (live) 0 else 1), manifest.inactive_runtime_owners.?.len);
        // The reachable query-free entrypoint must never be classified inactive.
        try testing.expect(pass.runtime_reachable_function_names.contains("run"));
        for (manifest.rows) |query| try testing.expect(!std.mem.eql(u8, query.function_name, "run"));
        var prepared = try c1Identity(manifest);
        defer prepared.deinit(testing.allocator);
        const evaluated = try compilation.db.constEval(compilation.root_module_id);
        try testing.expectEqual(@as(usize, 1), evaluated.formal_folds.len);
        try testing.expectEqual(compiler.sema.ComptimeFoldDisposition.abandoned, evaluated.formal_folds[0].disposition);
        const original_fold = evaluated.formal_folds[0];
        var result = try pass.runVerificationPass(hir.module.raw_module);
        defer result.deinit();
        try testing.expect(result.success);
        var accepted = try c1Account(&compilation, &formal, prepared.view());
        defer accepted.deinit();
        try testing.expectEqual(source_accounting_gate.Decision.accepted, accepted.finished.decision);
        try testing.expectEqualDeep(original_fold, evaluated.formal_folds[0]);
        for ([_]?[]const source_accounting_from_z3.InactiveRuntimeOwner{ null, &.{} }) |absent| {
            var view = prepared.view();
            view.inactive_runtime_owners = absent;
            var unchanged = try c1Account(&compilation, &formal, view);
            defer unchanged.deinit();
            try testing.expectEqual(if (live) source_accounting_gate.Decision.accepted else .rejected, unchanged.finished.decision);
            if (!live) try testing.expectEqual(@as(?source_accounting_gate.FailureCode, .missing_symbolic_obligation), unchanged.finished.primary_failure);
        }
        if (live) {
            try testing.expect(prepared.boundary_queries.len > 0);
            var view = prepared.view();
            view.boundary_queries = &.{};
            var missing = try c1Account(&compilation, &formal, view);
            defer missing.deinit();
            try testing.expectEqual(source_accounting_gate.Decision.rejected, missing.finished.decision);
            try testing.expectEqual(@as(?source_accounting_gate.FailureCode, .missing_symbolic_obligation), missing.finished.primary_failure);
            continue;
        }
        const original = prepared.inactive_runtime_owners.?[0];
        for (0..5) |mutation| {
            var rows = [_]source_accounting_from_z3.InactiveRuntimeOwner{ original, original };
            var view = prepared.view();
            view.inactive_runtime_owners = rows[0..1];
            const failure: anyerror = switch (mutation) {
                0 => blk: {
                    rows[0].source_op_id = 0;
                    break :blk error.InvalidInactiveRuntimeOwnerInventory;
                },
                1 => blk: {
                    rows[0].source_op_id = 1;
                    break :blk error.UnknownInactiveRuntimeOwner;
                },
                2 => blk: {
                    rows[0].module_path = "foreign.ora";
                    break :blk error.InactiveRuntimeOwnerIdentityMismatch;
                },
                3 => blk: {
                    rows[0].owner_key = "module/contract:C/function:run";
                    break :blk error.InactiveRuntimeOwnerIdentityMismatch;
                },
                else => blk: {
                    view.inactive_runtime_owners = &rows;
                    break :blk error.InvalidInactiveRuntimeOwnerInventory;
                },
            };
            try testing.expectError(failure, c1Account(&compilation, &formal, view));
        }
        // An additional runtime instance of this source owner is not covered by
        // the inactive inventory. It must keep the entire root active.
        const owners = try testing.allocator.alloc(obligation_from_mlir.RuntimeOwnerBinding, formal.runtime_owner_bindings.len + 1);
        defer testing.allocator.free(owners);
        @memcpy(owners[0..formal.runtime_owner_bindings.len], formal.runtime_owner_bindings);
        for (formal.runtime_owner_bindings) |owner| if (owner.source_op_id == original.source_op_id) {
            owners[owners.len - 1] = owner;
        };
        owners[owners.len - 1].source_op_id = 1;
        var mixed = formal;
        mixed.runtime_owner_bindings = owners;
        var retained = try c1Account(&compilation, &mixed, prepared.view());
        defer retained.deinit();
        try testing.expectEqual(source_accounting_gate.Decision.rejected, retained.finished.decision);
        try testing.expectEqual(@as(?source_accounting_gate.FailureCode, .missing_symbolic_obligation), retained.finished.primary_failure);
        owners[owners.len - 1].source_op_id = original.source_op_id;
        try testing.expectError(error.DuplicateInactiveRuntimeOwnerBinding, c1Account(&compilation, &mixed, prepared.view()));
        mixed.runtime_owner_bindings = &.{};
        try testing.expectError(error.UnknownInactiveRuntimeOwner, c1Account(&compilation, &mixed, prepared.view()));
    }
}

test "C1 inactive folds preserve validation rejected roots and nested ancestry" {
    var compilation = try compileText(
        \\contract C {
        \\ storage var n: u256;
        \\ pub fn callee() -> u256 requires n > 0 { return n; }
        \\ fn inactive() -> u256 { return callee(); }
        \\ pub fn run() -> u256 { return 1; }
        \\}
    );
    defer compilation.deinit();
    const evaluated = try compilation.db.constEval(compilation.root_module_id);
    try testing.expectEqual(@as(usize, 1), evaluated.formal_folds.len);
    var package = try source_accounting_from_package.collect(testing.allocator, &compilation.db, compilation.package_id, compilation.root_module_id);
    defer package.deinit();
    const original = evaluated.formal_folds[0];
    const module = package.module(original.root_runtime_module_id).?;
    const root = try std.fmt.allocPrint(testing.allocator, "{s}::{s}", .{ module.canonical_path, original.root_runtime_owner_key });
    defer testing.allocator.free(root);
    const inactive = [_][]const u8{root};
    var plain = try source_accounting_from_comptime.collectPackageFoldExpansions(testing.allocator, &package, evaluated, .{});
    defer plain.deinit();
    try testing.expectEqual(@as(usize, 2), plain.expansions.len);
    var scoped = try source_accounting_from_comptime.collectPackageFoldExpansionsForRoots(testing.allocator, &package, evaluated, .{}, &inactive);
    defer scoped.deinit();
    try testing.expectEqual(@as(usize, 0), scoped.expansions.len);
    try testing.expectEqual(@as(usize, 0), scoped.uses.len);
    for (0..7) |mutation| {
        var rows = [_]compiler.sema.ComptimeFoldEvidence{ original, original };
        var ledger = .{ .formal_folds = @as([]const compiler.sema.ComptimeFoldEvidence, rows[0..1]) };
        const failure: anyerror = switch (mutation) {
            0 => blk: {
                rows[0].activation = .required;
                break :blk error.RequiredComptimeFoldWasAbandoned;
            },
            1 => blk: {
                rows[0].parent_invocation_id = 9999;
                break :blk error.UnknownComptimeFoldParentInvocation;
            },
            2 => blk: {
                ledger.formal_folds = &rows;
                break :blk error.DuplicateComptimeFoldInvocationIdentity;
            },
            3 => blk: {
                rows[0].callee_file_id = @fromBackingInt(@intCast(9999));
                break :blk error.ComptimeFoldCalleeFileMismatch;
            },
            4 => blk: {
                rows[0].root_runtime_module_id = @fromBackingInt(@intCast(9999));
                break :blk error.UnknownComptimeFoldRootModule;
            },
            5 => blk: {
                rows[0].generic_bindings = &.{ "z", "a" };
                break :blk error.InvalidComptimeFoldGenericBindings;
            },
            else => blk: {
                rows[0].trait_method = "m";
                break :blk error.IncompleteComptimeFoldTraitIdentity;
            },
        };
        try testing.expectError(failure, source_accounting_from_comptime.collectPackageFoldExpansionsForRoots(testing.allocator, &package, ledger, .{}, &inactive));
    }
    var rejected = original;
    rejected.disposition = .rejected;
    var rejection = try source_accounting_from_comptime.collectPackageFoldExpansionsForRoots(testing.allocator, &package, .{ .formal_folds = @as([]const compiler.sema.ComptimeFoldEvidence, &.{rejected}) }, .{}, &inactive);
    defer rejection.deinit();
    try testing.expectEqual(@as(usize, 1), rejection.expansions.len);
    try testing.expectEqual(source_accounting.ExpansionDisposition.rejected, rejection.expansions[0].disposition);
    var rows = [_]compiler.sema.ComptimeFoldEvidence{ original, original };
    rows[1].invocation_id = original.invocation_id + 1;
    rows[1].parent_invocation_id = original.invocation_id;
    const chain = [_]compiler.sema.ComptimeFoldCallSite{ original.call_site_chain[0], original.call_site_chain[0] };
    rows[1].call_site_chain = &chain;
    var nested = try source_accounting_from_comptime.collectPackageFoldExpansionsForRoots(testing.allocator, &package, .{ .formal_folds = @as([]const compiler.sema.ComptimeFoldEvidence, &rows) }, .{}, &inactive);
    defer nested.deinit();
    try testing.expectEqual(@as(usize, 4), nested.expansions.len);
    try testing.expectEqual(@as(?source_accounting.ExpansionId, nested.expansions[0].id), nested.expansions[2].parent_expansion_id);
    var wrong_root = try source_accounting_from_comptime.collectPackageFoldExpansionsForRoots(testing.allocator, &package, evaluated, .{}, &.{"other.ora::module/contract:C/function:inactive"});
    defer wrong_root.deinit();
    try testing.expectEqual(@as(usize, 2), wrong_root.expansions.len);
}

test "C1 inactive folds distinguish same-name module and contract owners" {
    var tmp = testing.tmpDir(.{});
    defer tmp.cleanup();
    for ([_][]const u8{ "left.ora", "right.ora" }) |name| try tmp.dir.writeFile(std.testing.io, .{
        .sub_path = name,
        .data =
        \\contract A {
        \\ storage var n: u256;
        \\ pub fn callee() -> u256 requires n > 0 { return n; }
        \\ fn inactive() -> u256 { return callee(); }
        \\}
        \\contract B {
        \\ storage var n: u256;
        \\ pub fn callee() -> u256 requires n > 0 { return n; }
        \\ fn inactive() -> u256 { return callee(); }
        \\}
        \\pub fn touch() -> u256 { return 1; }
        ,
    });
    try tmp.dir.writeFile(std.testing.io, .{
        .sub_path = "main.ora",
        .data =
        \\comptime const left = @import("./left.ora");
        \\comptime const right = @import("./right.ora");
        \\contract Main { pub fn run() -> u256 { return left.touch() + right.touch(); } }
        ,
    });
    const path = try std.fmt.allocPrint(testing.allocator, ".zig-cache/tmp/{s}/main.ora", .{tmp.sub_path});
    defer testing.allocator.free(path);
    var compilation = try compilePackage(path);
    defer compilation.deinit();
    var package = try source_accounting_from_package.collect(testing.allocator, &compilation.db, compilation.package_id, compilation.root_module_id);
    defer package.deinit();
    try testing.expectEqual(@as(usize, 3), package.modules.len);
    var checked: usize = 0;
    for (package.modules) |module| {
        if (module.module_id == compilation.root_module_id) continue;
        const evaluated = try compilation.db.constEval(module.module_id);
        try testing.expectEqual(@as(usize, 2), evaluated.formal_folds.len);
        const inactive = "left.ora::module/contract:A/function:inactive";
        var scoped = try source_accounting_from_comptime.collectPackageFoldExpansionsForRoots(testing.allocator, &package, evaluated, .{}, &.{inactive});
        defer scoped.deinit();
        const left = std.mem.eql(u8, module.canonical_path, "left.ora");
        try testing.expectEqual(@as(usize, if (left) 2 else 4), scoped.expansions.len);
        for (scoped.expansions) |expansion| try testing.expect(!std.mem.eql(u8, expansion.root_runtime_owner, inactive));
        checked += 1;
    }
    try testing.expectEqual(@as(usize, 2), checked);
}

test "C1 inactive folds retain committed proof events" {
    var compilation = try compileText(
        \\contract C {
        \\ fn callee() -> u256 requires true ensures result == 1 { return 1; }
        \\ fn inactive() -> u256 { return callee(); }
        \\ pub fn run() -> u256 { return 1; }
        \\}
    );
    defer compilation.deinit();
    const evaluated = try compilation.db.constEval(compilation.root_module_id);
    try testing.expectEqual(@as(usize, 1), evaluated.formal_folds.len);
    try testing.expectEqual(compiler.sema.ComptimeFoldDisposition.committed, evaluated.formal_folds[0].disposition);
    var package = try source_accounting_from_package.collect(testing.allocator, &compilation.db, compilation.package_id, compilation.root_module_id);
    defer package.deinit();
    const module = package.module(evaluated.formal_folds[0].root_runtime_module_id).?;
    const root = try std.fmt.allocPrint(testing.allocator, "{s}::{s}", .{ module.canonical_path, evaluated.formal_folds[0].root_runtime_owner_key });
    defer testing.allocator.free(root);
    var scoped = try source_accounting_from_comptime.collectPackageFoldExpansionsForRoots(testing.allocator, &package, evaluated, .{}, &.{root});
    defer scoped.deinit();
    var unscoped = try source_accounting_from_comptime.collectPackageFoldExpansions(testing.allocator, &package, evaluated, .{});
    defer unscoped.deinit();
    try testing.expectEqual(@as(usize, 1), scoped.expansions.len);
    try testing.expectEqual(@as(usize, 1), scoped.evidence.folds.len);
    try testing.expectEqual(@as(usize, 5), scoped.evidence.predicate_events.len);
    try testing.expectEqualDeep(unscoped.evidence, scoped.evidence);
}

test "result alias C3 preserves actual postcondition queries and missing-query rejection" {
    for ([_]bool{ false, true }) |false_postcondition| {
        const source = try std.fmt.allocPrint(
            testing.allocator,
            "contract C {{ storage var n: u256; pub fn get() -> u256 ensures get() {s} n {{ return n; }} }}",
            .{if (false_postcondition) "!=" else "=="},
        );
        defer testing.allocator.free(source);
        var compilation = try compileText(source);
        defer compilation.deinit();
        const hir = try compilation.db.lowerToHir(compilation.root_module_id);
        try testing.expect(hir.diagnostics.isEmpty());
        const evaluated = try compilation.db.constEval(compilation.root_module_id);
        try testing.expect(evaluated.diagnostics.isEmpty());
        try testing.expectEqual(@as(usize, 0), evaluated.formal_folds.len);
        var formal = try obligation_from_mlir.collect(testing.allocator, hir.module.raw_module, .{});
        defer formal.deinit();
        var pass = try z3_verification.VerificationPass.init(testing.allocator);
        defer pass.deinit();
        const bindings = try test_helpers.z3FormalQueryBindingsFromFormalForTest(testing.allocator, formal.query_bindings);
        defer testing.allocator.free(bindings);
        pass.setFormalQueryBindings(bindings);
        var manifest = try pass.collectPreparedQueryManifest(hir.module.raw_module);
        defer manifest.deinit();
        var prepared = try source_accounting_from_z3.collectPreparedIdentity(testing.allocator, manifest.source_accounting_rows);
        defer prepared.deinit(testing.allocator);
        try testing.expectEqual(@as(usize, 1), prepared.queries.len);
        try testing.expectEqual(@as(usize, 0), prepared.boundary_queries.len);
        var result = try pass.runVerificationPass(hir.module.raw_module);
        defer result.deinit();
        try testing.expectEqual(!false_postcondition, result.success);
        var report = try pass.buildSmtReport(hir.module.raw_module, "test.ora", &result);
        defer report.deinit(testing.allocator);
        var overlay = try obligation_from_z3.overlayPreparedQueryResults(testing.allocator, formal.set, report.query_manifest.?.rows);
        defer overlay.deinit();
        const query = obligation.findById(overlay.set.queries, prepared.queries[0].formal_query_id).?;
        try testing.expect(query.smtlib_hash != null and query.smtlib_hash.? != 0);
        try testing.expectEqual(
            if (false_postcondition) obligation.VerificationQueryStatus.sat else obligation.VerificationQueryStatus.unsat,
            query.result.?.status,
        );
        var accounted = try source_accounting_pipeline.run(
            testing.allocator,
            &compilation.db,
            compilation.package_id,
            compilation.root_module_id,
            evaluated,
            &formal,
            .verified_full,
            prepared.view(),
        );
        defer accounted.deinit();
        try testing.expectEqual(source_accounting_gate.Decision.accepted, accounted.finished.decision);
        var missing = try source_accounting_pipeline.run(
            testing.allocator,
            &compilation.db,
            compilation.package_id,
            compilation.root_module_id,
            evaluated,
            &formal,
            .verified_full,
            .{
                .queries = &.{},
                .runtime_function_names = prepared.runtime_function_names,
            },
        );
        defer missing.deinit();
        try testing.expectEqual(source_accounting_gate.Decision.rejected, missing.finished.decision);
        try testing.expectEqual(@as(?source_accounting_gate.FailureCode, .missing_symbolic_query), missing.finished.primary_failure);
    }
}

test "production source-accounting pipeline keeps loop accounting queries diagnostic-only" {
    const source =
        \\pub fn slice(limit: u256) -> u256 {
        \\    var i: u256 = 0;
        \\    while (i < limit)
        \\        invariant i <= limit
        \\    {
        \\        i = i + 1;
        \\    }
        \\    return i;
        \\}
    ;
    var compilation = try compileText(source);
    defer compilation.deinit();
    const hir_result = try compilation.db.lowerToHir(compilation.root_module_id);
    try testing.expect(hir_result.diagnostics.isEmpty());

    var formal = try obligation_from_mlir.collect(testing.allocator, hir_result.module.raw_module, .{});
    defer formal.deinit();
    var pass = try z3_verification.VerificationPass.init(testing.allocator);
    defer pass.deinit();
    const z3_bindings = try test_helpers.z3FormalQueryBindingsFromFormalForTest(
        testing.allocator,
        formal.query_bindings,
    );
    defer testing.allocator.free(z3_bindings);
    pass.setFormalQueryBindings(z3_bindings);
    var manifest = try pass.collectPreparedQueryManifest(hir_result.module.raw_module);
    defer manifest.deinit();
    var prepared = try source_accounting_from_z3.collectPreparedIdentity(testing.allocator, manifest.source_accounting_rows);
    defer prepared.deinit(testing.allocator);

    var saw_obligation = false;
    var saw_step = false;
    for (prepared.queries) |query| {
        if (query.kind == .obligation) saw_obligation = true;
        if (query.kind == .loop_invariant_step) saw_step = true;
    }
    // The generic row is an accounting identity only. First-class loop rows
    // own proof and artifact authorization for invariant/body-safety premises.
    try testing.expect(saw_obligation);
    try testing.expect(saw_step);
    var saw_diagnostic_accounting_query = false;
    for (formal.set.queries) |query| {
        if (query.kind == .obligation and
            query.logical_role == .invariant and
            query.artifact_policy == .diagnostic_only)
        {
            saw_diagnostic_accounting_query = true;
        }
    }
    try testing.expect(saw_diagnostic_accounting_query);

    const const_eval = try compilation.db.constEval(compilation.root_module_id);
    var result = try source_accounting_pipeline.run(
        testing.allocator,
        &compilation.db,
        compilation.package_id,
        compilation.root_module_id,
        const_eval,
        &formal,
        .verified_full,
        prepared.view(),
    );
    defer result.deinit();
    try testing.expectEqual(source_accounting_gate.Decision.accepted, result.finished.decision);
    try testing.expectEqual(@as(usize, 0), result.finished.failures.len);
    try testing.expect(std.mem.containsAtLeast(u8, result.finished.report, 1, "\"kind\":\"loop_invariant_step\""));
}

test "production source-accounting pipeline binds abandoned-fold call-boundary queries" {
    const source =
        \\contract BoundaryAccounting {
        \\    storage var saved: u256;
        \\
        \\    fn checked() -> u256
        \\        requires saved < 100
        \\        requires saved != 50
        \\        ensures result == 7
        \\        ensures result > 0
        \\    {
        \\        saved = saved + 1;
        \\        return 7;
        \\    }
        \\
        \\    pub fn run() -> u256 {
        \\        return checked();
        \\    }
        \\}
    ;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    try tmp.dir.writeFile(std.testing.io, .{ .sub_path = "main.ora", .data = source });
    const root_path = try std.fmt.allocPrint(testing.allocator, ".zig-cache/tmp/{s}/main.ora", .{tmp.sub_path});
    defer testing.allocator.free(root_path);
    var compilation = try compilePackage(root_path);
    defer compilation.deinit();
    const hir_result = try compilation.db.lowerToHir(compilation.root_module_id);
    try testing.expect(hir_result.diagnostics.isEmpty());

    var formal = try obligation_from_mlir.collect(testing.allocator, hir_result.module.raw_module, .{});
    defer formal.deinit();
    var pass = try z3_verification.VerificationPass.init(testing.allocator);
    defer pass.deinit();
    const z3_bindings = try test_helpers.z3FormalQueryBindingsFromFormalForTest(
        testing.allocator,
        formal.query_bindings,
    );
    defer testing.allocator.free(z3_bindings);
    pass.setFormalQueryBindings(z3_bindings);
    var manifest = try pass.collectPreparedQueryManifest(hir_result.module.raw_module);
    defer manifest.deinit();
    var prepared = try source_accounting_from_z3.collectPreparedIdentity(testing.allocator, manifest.source_accounting_rows);
    defer prepared.deinit(testing.allocator);

    var saw_target = false;
    var saw_context = false;
    var boundary_source_fact_ids = std.AutoHashMap(u32, void).init(testing.allocator);
    defer boundary_source_fact_ids.deinit();
    for (prepared.boundary_queries) |query| {
        try testing.expectEqualStrings("checked", query.callee_name);
        try testing.expect(!boundary_source_fact_ids.contains(query.source_fact_id));
        try boundary_source_fact_ids.put(query.source_fact_id, {});
        switch (query.role) {
            .proof_target => saw_target = true,
            .assumption_context => saw_context = true,
            else => return error.TestUnexpectedResult,
        }
    }
    try testing.expect(saw_target);
    try testing.expect(saw_context);
    try testing.expectEqual(@as(usize, 4), prepared.boundary_queries.len);

    const const_eval = try compilation.db.constEval(compilation.root_module_id);
    var missing = try source_accounting_pipeline.run(
        testing.allocator,
        &compilation.db,
        compilation.package_id,
        compilation.root_module_id,
        const_eval,
        &formal,
        .verified_full,
        .{
            .queries = prepared.queries,
            .boundary_queries = &.{},
            .runtime_function_names = prepared.runtime_function_names,
        },
    );
    defer missing.deinit();
    try testing.expectEqual(source_accounting_gate.Decision.rejected, missing.finished.decision);
    try testing.expectEqual(
        @as(?source_accounting_gate.FailureCode, .missing_symbolic_obligation),
        missing.finished.primary_failure,
    );

    var verified = try source_accounting_pipeline.run(
        testing.allocator,
        &compilation.db,
        compilation.package_id,
        compilation.root_module_id,
        const_eval,
        &formal,
        .verified_full,
        prepared.view(),
    );
    defer verified.deinit();
    try testing.expectEqual(source_accounting_gate.Decision.accepted, verified.finished.decision);
    try testing.expect(std.mem.containsAtLeast(u8, verified.finished.report, 1, "\"activation\":\"symbolic_call\""));
    try testing.expect(std.mem.containsAtLeast(u8, verified.finished.report, 1, "\"kind\":\"assumption_incorporated\""));
    var report = try std.json.parseFromSlice(std.json.Value, testing.allocator, verified.finished.report, .{});
    defer report.deinit();
    const report_object = report.value.object;
    const expansions = (report_object.get("expansions") orelse return error.TestUnexpectedResult).array.items;
    var boundary_expansion_ids = std.AutoHashMap(i64, void).init(testing.allocator);
    defer boundary_expansion_ids.deinit();
    for (expansions) |expansion| {
        const object = expansion.object;
        const activation = (object.get("activation") orelse return error.TestUnexpectedResult).string;
        if (!std.mem.eql(u8, activation, "symbolic_call")) continue;
        const id = (object.get("id") orelse return error.TestUnexpectedResult).integer;
        try boundary_expansion_ids.put(id, {});
    }
    const uses = (report_object.get("uses") orelse return error.TestUnexpectedResult).array.items;
    var boundary_use_ids = std.AutoHashMap(i64, void).init(testing.allocator);
    defer boundary_use_ids.deinit();
    for (uses) |use| {
        const object = use.object;
        const expansion_id = (object.get("expansion_id") orelse return error.TestUnexpectedResult).integer;
        if (!boundary_expansion_ids.contains(expansion_id)) continue;
        const id = (object.get("id") orelse return error.TestUnexpectedResult).integer;
        try boundary_use_ids.put(id, {});
    }
    var accounted_boundary_uses: usize = 0;
    const handlings = (report_object.get("handlings") orelse return error.TestUnexpectedResult).array.items;
    for (handlings) |handling| {
        const object = handling.object;
        const use_id = (object.get("use_id") orelse return error.TestUnexpectedResult).integer;
        if (!boundary_use_ids.contains(use_id)) continue;
        const query_ids = (object.get("query_ids") orelse return error.TestUnexpectedResult).array.items;
        try testing.expectEqual(@as(usize, 1), query_ids.len);
        accounted_boundary_uses += 1;
    }
    try testing.expectEqual(@as(usize, 4), accounted_boundary_uses);

    var unverified = try source_accounting_pipeline.run(
        testing.allocator,
        &compilation.db,
        compilation.package_id,
        compilation.root_module_id,
        const_eval,
        &formal,
        .unverified_emit,
        null,
    );
    defer unverified.deinit();
    try testing.expectEqual(source_accounting_gate.Decision.accepted, unverified.finished.decision);
    try testing.expect(std.mem.containsAtLeast(u8, unverified.finished.report, 1, "\"kind\":\"verification_disabled\""));
}

test "production source-accounting pipeline preserves guard and modifies evidence" {
    const source =
        \\contract AccountingSurface {
        \\    storage var balances: map<u256, u256>;
        \\
        \\    pub fn guarded(y: u256) -> bool
        \\        guard y > 0
        \\    {
        \\        return true;
        \\    }
        \\
        \\    pub fn copy(user: u256, other: u256)
        \\        modifies balances[user]
        \\        modifies balances[other]
        \\        requires user != other
        \\    {
        \\        balances[user] = balances[other];
        \\    }
        \\}
    ;
    var compilation = try compileText(source);
    defer compilation.deinit();
    const hir_result = try compilation.db.lowerToHir(compilation.root_module_id);
    try testing.expect(hir_result.diagnostics.isEmpty());

    var formal = try obligation_from_mlir.collect(testing.allocator, hir_result.module.raw_module, .{});
    defer formal.deinit();
    var modifies_binding_count: usize = 0;
    for (formal.source_fact_bindings) |binding| {
        if (!std.mem.eql(u8, binding.kind, "modifies")) continue;
        modifies_binding_count += 1;
        try testing.expect(binding.obligation_ids.len > 0);
    }
    try testing.expectEqual(@as(usize, 2), modifies_binding_count);

    var pass = try z3_verification.VerificationPass.init(testing.allocator);
    defer pass.deinit();
    const z3_bindings = try test_helpers.z3FormalQueryBindingsFromFormalForTest(
        testing.allocator,
        formal.query_bindings,
    );
    defer testing.allocator.free(z3_bindings);
    pass.setFormalQueryBindings(z3_bindings);
    var manifest = try pass.collectPreparedQueryManifest(hir_result.module.raw_module);
    defer manifest.deinit();
    var saw_guard_accounting_query = false;
    for (manifest.rows) |row| {
        if (row.obligation_kind != .Guard) continue;
        saw_guard_accounting_query = true;
        try testing.expectEqual(z3_verification.FormalMatchStatus.not_applicable, row.formal_match_status);
        try testing.expect(row.formal_query_id == null);
        try testing.expectEqual(z3_verification.FormalMatchStatus.matched, row.accounting_match_status);
        try testing.expect(row.accounting_query_id != null);
    }
    try testing.expect(saw_guard_accounting_query);

    var prepared = try source_accounting_from_z3.collectPreparedIdentity(testing.allocator, manifest.source_accounting_rows);
    defer prepared.deinit(testing.allocator);
    const const_eval = try compilation.db.constEval(compilation.root_module_id);
    var result = try source_accounting_pipeline.run(
        testing.allocator,
        &compilation.db,
        compilation.package_id,
        compilation.root_module_id,
        const_eval,
        &formal,
        .verified_full,
        prepared.view(),
    );
    defer result.deinit();
    try testing.expectEqual(source_accounting_gate.Decision.accepted, result.finished.decision);
    try testing.expect(std.mem.containsAtLeast(u8, result.finished.report, 1, "\"frame_directive\":2"));
    try testing.expect(std.mem.containsAtLeast(u8, result.finished.report, 1, "\"frame_validated\":2"));
}

test "production source-accounting pipeline preserves impl contract and collapsed switch identities" {
    const source =
        \\trait Priced { fn price(self) -> u256; }
        \\struct Quote { amount: u256, }
        \\impl Priced for Quote {
        \\    fn price(self) -> u256
        \\        requires(self.amount > 0)
        \\    { return self.amount; }
        \\}
        \\contract AccountingIdentity {
        \\    storage var state: u256 = 0;
        \\    invariant bounded(state <= 100);
        \\    pub fn run(value: u256)
        \\        modifies()
        \\    {
        \\        var scratch: u256 = 0;
        \\        havoc scratch;
        \\        switch (value) {
        \\            0 => { },
        \\            1 => { },
        \\            2 => { },
        \\            else => { }
        \\        }
        \\    }
        \\}
    ;

    var compilation = try compileText(source);
    defer compilation.deinit();
    const hir_result = try compilation.db.lowerToHir(compilation.root_module_id);
    try testing.expect(hir_result.diagnostics.isEmpty());

    var formal = try obligation_from_mlir.collect(testing.allocator, hir_result.module.raw_module, .{});
    defer formal.deinit();
    var saw_contract_invariant_scope = false;
    var saw_empty_modifies_result = false;
    for (formal.source_fact_bindings) |binding| {
        if (std.mem.eql(u8, binding.kind, "contract_invariant")) {
            saw_contract_invariant_scope = binding.runtime_symbol == null and
                std.mem.eql(u8, binding.owner_key orelse "", "module/contract:AccountingIdentity");
        }
        if (std.mem.eql(u8, binding.kind, "modifies")) {
            saw_empty_modifies_result = binding.obligation_ids.len == 1;
        }
    }
    try testing.expect(saw_contract_invariant_scope);
    try testing.expect(saw_empty_modifies_result);
    try testing.expectEqual(@as(usize, 1), formal.state_effect_producers.len);
    const const_eval = try compilation.db.constEval(compilation.root_module_id);
    try testing.expect(const_eval.diagnostics.isEmpty());
    var result = try source_accounting_pipeline.run(
        testing.allocator,
        &compilation.db,
        compilation.package_id,
        compilation.root_module_id,
        const_eval,
        &formal,
        .unverified_emit,
        null,
    );
    defer result.deinit();
    try testing.expectEqual(source_accounting_gate.Decision.accepted, result.finished.decision);
}

test "formal report coverage includes representative source contracts" {
    var logical = try collectPackageObligations(testing.allocator, "ora-example/formal/obligation_report_logical.ora");
    defer logical.deinit();
    try testing.expect(countAssumption(logical.set, .requires) > 0);
    try testing.expect(countLogical(logical.set, .ensures) > 0);
    try testing.expect(countLogical(logical.set, .assert) > 0);
    try testing.expect(countArithmeticSafety(logical.set, .subtraction_overflow) > 0);
    {
        const report = try dumpManifestToOwnedString(testing.allocator, logical.set);
        defer testing.allocator.free(report);
        try testing.expect(std.mem.containsAtLeast(u8, report, 1, "{\"record\":\"coverage_summary\",\"schema_version\":12"));
        try testing.expect(std.mem.containsAtLeast(u8, report, 1, "\"record\":\"assumption\",\"schema_version\":12"));
        try testing.expect(std.mem.containsAtLeast(u8, report, 1, "\"role\":\"ensures\""));
        try testing.expect(std.mem.containsAtLeast(u8, report, 1, "\"role\":\"assert\""));
        try testing.expect(std.mem.containsAtLeast(u8, report, 1, "\"arithmetic_safety\":\"subtraction_overflow\""));
    }

    var resource = try collectPackageObligations(testing.allocator, "ora-example/formal/obligation_report_resource.ora");
    defer resource.deinit();
    try testing.expect(countResource(resource.set, .move, .source_sufficient) > 0);
    try testing.expect(countResource(resource.set, .move, .destination_no_overflow) > 0);
    try testing.expect(countResource(resource.set, .move, .conservation) > 0);
    try testing.expect(countEffectFrame(resource.set, .write_covered_by_modifies) > 0);
    {
        const report = try dumpManifestToOwnedString(testing.allocator, resource.set);
        defer testing.allocator.free(report);
        try testing.expect(std.mem.containsAtLeast(u8, report, 1, "{\"record\":\"coverage_summary\",\"schema_version\":12"));
        try testing.expect(std.mem.containsAtLeast(u8, report, 1, "\"kind\":{\"tag\":\"effect_frame\""));
        try testing.expect(std.mem.containsAtLeast(u8, report, 1, "\"kind\":{\"tag\":\"resource\",\"op\":\"move\",\"domain\":\"TokenUnit\""));
        try testing.expect(std.mem.containsAtLeast(u8, report, 1, "\"property\":\"conservation\""));
    }

    var quantifier = try collectPackageObligations(testing.allocator, "ora-example/formal/obligation_report_quantifier.ora");
    defer quantifier.deinit();
    try testing.expect(countQuantifier(quantifier.set) > 0);
    {
        const report = try dumpManifestToOwnedString(testing.allocator, quantifier.set);
        defer testing.allocator.free(report);
        try testing.expect(std.mem.containsAtLeast(u8, report, 1, "{\"record\":\"coverage_summary\",\"schema_version\":12"));
        try testing.expect(std.mem.containsAtLeast(u8, report, 1, "\"kind\":{\"tag\":\"quantifier\",\"quantifier\":\"forall\""));
        try testing.expect(std.mem.containsAtLeast(u8, report, 1, "\"binder_type\":{\"tag\":\"spelling\",\"value\":\"u256\"}"));
    }
}

test "formal obligation MLIR adapter blocks malformed quantifier binder width" {
    const h = createContext();
    defer destroyContext(h);

    const text =
        \\module {
        \\  func.func @bad(%flag: i1) {
        \\    %q = "ora.quantified"(%flag) <{quantifier = "exists", variable = "i", variable_type = "uabc"}> : (i1) -> i1
        \\    "ora.assert"(%q) <{message = "bad"}> : (i1) -> ()
        \\    func.return
        \\  }
        \\}
    ;

    const module = try parseModule(h.ctx, text);
    defer mlir.oraModuleDestroy(module);

    var result = try obligation_from_mlir.collect(testing.allocator, module, .{});
    defer result.deinit();

    try testing.expect(result.set.hasBlockingDiagnostic());
    try testing.expectEqual(@as(usize, 1), countQuantifier(result.set));
    try testing.expectEqual(@as(usize, 1), result.set.diagnostics.len);
    try testing.expectEqual(obligation.QuantifierDegradation.malformed_binder_width, result.set.obligations[0].kind.quantifier.degradation.?);
    try testing.expectEqual(obligation.ArtifactBlockReason.blocking_diagnostic, result.set.artifactDecision().blocked);
}

test "formal obligation MLIR adapter blocks unsupported quantifier binder type" {
    const h = createContext();
    defer destroyContext(h);

    const text =
        \\module {
        \\  func.func @bad(%flag: i1) {
        \\    %q = "ora.quantified"(%flag) <{quantifier = "forall", variable = "item", variable_type = "FutureType"}> : (i1) -> i1
        \\    "ora.assert"(%q) <{message = "bad"}> : (i1) -> ()
        \\    func.return
        \\  }
        \\}
    ;

    const module = try parseModule(h.ctx, text);
    defer mlir.oraModuleDestroy(module);

    var result = try obligation_from_mlir.collect(testing.allocator, module, .{});
    defer result.deinit();

    try testing.expect(result.set.hasBlockingDiagnostic());
    try testing.expectEqual(@as(usize, 1), countQuantifier(result.set));
    try testing.expectEqual(@as(usize, 1), result.set.diagnostics.len);
    try testing.expectEqual(obligation.QuantifierBinderSort.opaque_unknown, result.set.obligations[0].kind.quantifier.binder_sort);
    try testing.expectEqual(obligation.QuantifierDegradation.unsupported_binder_type, result.set.obligations[0].kind.quantifier.degradation.?);
}

test "formal resource move without segments is rejected before obligation collection" {
    const h = createContext();
    defer destroyContext(h);

    const text =
        \\module {
        \\  func.func @bad(%balances: !ora.map<!ora.address, !ora.int<256, false>>, %from: !ora.address, %to: !ora.address, %amount: !ora.int<256, false>) {
        \\    "ora.move"(%balances, %from, %balances, %to, %amount) <{domain = "TokenUnit", carrier_type = !ora.int<256, false>, carrier_signed = false}> : (!ora.map<!ora.address, !ora.int<256, false>>, !ora.address, !ora.map<!ora.address, !ora.int<256, false>>, !ora.address, !ora.int<256, false>) -> ()
        \\    func.return
        \\  }
        \\}
    ;

    try testing.expectError(error.MlirParseFailed, parseModule(h.ctx, text));
}

test "formal obligation MLIR adapter fails closed on malformed guard marker" {
    const h = createContext();
    defer destroyContext(h);

    const text =
        \\module {
        \\  func.func @bad(%flag: i1) {
        \\    "ora.refinement_guard"(%flag) <{message = "guard"}> : (i1) -> ()
        \\    func.return
        \\  }
        \\}
    ;

    const module = try parseModule(h.ctx, text);
    defer mlir.oraModuleDestroy(module);

    var result = try obligation_from_mlir.collect(testing.allocator, module, .{});
    defer result.deinit();

    try testing.expect(result.set.hasBlockingDiagnostic());
    try testing.expectEqual(@as(usize, 0), result.set.obligations.len);
    try testing.expectEqual(@as(usize, 0), result.set.queries.len);
    try testing.expectEqual(@as(usize, 1), result.set.diagnostics.len);
}

test "certified call rows generated module parses and checks shape" {
    const ty: obligation.TypeRef = .{ .spelling = "bool" };
    const result_id: obligation.FreeVarId = .{ .file_id = 81, .pattern_id = 1 };
    const terms = [_]obligation.Term{
        .{ .bool_lit = true },
        .{ .variable = .{ .free = .{ .id = result_id, .name = "result", .ty = ty } } },
    };
    const literal: obligation.LoopTypedFormula = .{ .ty = ty, .value = .{ .formula = .{ .term = 0 } } };
    const leaf: obligation.FunctionSummaryRow = .{
        .id = 1,
        .owner = .{ .function = .{ .name = "leaf", .contract = "Calls" } },
        .source = .generated(),
        .origin = .source,
        .structural_key = "Calls/leaf",
        .outcomes = &.{.{ .outcome = .{ .error_return = 7 }, .types = &.{ty} }},
        .paths = &.{.{ .id = 0, .structural_key = "return", .returns = .{ .direct = .{
            .outcome = .{ .error_return = 7 },
            .values = &.{literal},
        } } }},
        .write_footprint_complete = true,
        .projection_completeness = .complete,
    };
    const digest = try obligation_to_lean.functionVcSha256(.{ .terms = &terms, .function_summaries = &.{leaf} }, 1);
    const events = [_]obligation.LoopTimelineEvent{.{ .id = 0, .structural_key = "caller/call", .payload = .{ .call = .{
        .callee_summary_id = 1,
        .callee_vc_sha256 = digest,
        .outcome = .{ .error_return = 7 },
        .results = &.{.{ .index = 0, .id = result_id, .ty = ty }},
    } } }};
    const caller: obligation.FunctionSummaryRow = .{
        .id = 2,
        .owner = .{ .function = .{ .name = "caller" } },
        .source = .generated(),
        .origin = .source,
        .structural_key = "caller",
        .outcomes = leaf.outcomes,
        .paths = &.{.{ .id = 0, .structural_key = "caller/return", .timeline = &events, .returns = .{ .direct = .{
            .outcome = .{ .error_return = 7 },
            .values = &.{.{ .ty = ty, .value = .{
                .formula = .{ .term = 1 },
                .read_results = &.{result_id},
            } }},
        } } }},
        .write_footprint_complete = true,
        .projection_completeness = .complete,
    };
    var source = std.Io.Writer.Allocating.init(testing.allocator);
    defer source.deinit();
    try obligation_to_lean.writeModule(&source.writer, .{ .terms = &terms, .function_summaries = &.{ leaf, caller } }, .{
        .namespace = "Ora.Generated.CallRows",
    });
    try source.writer.writeAll(
        "example : Ora.Generated.CallRows.emittedFunctionSummaries.all Ora.Loop.FunctionSummaryRow.shapeSupported = true := by decide\n",
    );
    var tmp = testing.tmpDir(.{});
    defer tmp.cleanup();
    const path = try pathFromTmpAlloc(testing.allocator, tmp, "CallRows.lean");
    defer testing.allocator.free(path);
    try std.Io.Dir.cwd().writeFile(std.testing.io, .{ .sub_path = path, .data = source.written() });
    const parsed = try runLeanFileForTest(testing.allocator, path);
    defer testing.allocator.free(parsed.stdout);
    defer testing.allocator.free(parsed.stderr);
    try expectExited(parsed, 0);
}

fn collectCallProjection(allocator: std.mem.Allocator, path: []const u8) !obligation_from_mlir.CollectResult {
    var compilation = try compilePackage(path);
    defer compilation.deinit();
    const hir = try compilation.db.lowerToHir(compilation.root_module_id);
    try testing.expect(hir.diagnostics.isEmpty());
    return obligation_from_mlir.collect(allocator, hir.module.raw_module, .{ .project_calls = true });
}

test "G9c external proof recipe names every function and adapter loop target without authority" {
    const allocator = testing.allocator;
    var tmp = testing.tmpDir(.{});
    defer tmp.cleanup();
    const manifest = try pathFromTmpAlloc(allocator, tmp, "empty.proofs.json");
    defer allocator.free(manifest);
    try tmp.dir.writeFile(std.testing.io, .{ .sub_path = "empty.proofs.json", .data = "{\"schema_version\":1,\"proofs\":[]}" });
    const option = try std.fmt.allocPrint(allocator, "--lean-proofs={s}", .{manifest});
    defer allocator.free(option);
    const out = try pathFromTmpAlloc(allocator, tmp, "out");
    defer allocator.free(out);
    const result = try runProcess(allocator, &.{ ORA_BINARY_REL, "emit", option, "--emit=smt-report,bytecode", "--out-dir", out, "tests/formal/call_projection_callee_loop_value.ora" });
    defer allocator.free(result.stdout);
    defer allocator.free(result.stderr);
    errdefer std.debug.print("G9c recipe: {s}\n{s}\n", .{ result.stdout, result.stderr });
    try expectExited(result, 1);
    try testing.expect(std.mem.indexOf(u8, result.stdout, "EmptyProofRows") != null);
    for ([_][]const u8{ ".emittedCallCustody_1", ".emittedCallCustody_2", ".emittedCallLoop_1" }) |target|
        try testing.expect(std.mem.indexOf(u8, result.stdout, target) != null);
    try testing.expectEqual(@as(usize, 3), std.mem.count(u8, result.stdout, "proof row: query_id="));
    try testing.expectError(error.FileNotFound, tmp.dir.access(std.testing.io, "out/call_projection_callee_loop_value.hex", .{}));
    const source = try pathFromTmpAlloc(allocator, tmp, "out/call_projection_callee_loop_value.lean.obligations.lean");
    defer allocator.free(source);
    const checked = try runLeanFileForTest(allocator, source);
    defer allocator.free(checked.stdout);
    defer allocator.free(checked.stderr);
    try expectExited(checked, 0);
}

test "G9c native certified calls prepare both prover targets without authority" {
    const allocator = testing.allocator;
    for ([_][]const u8{ "callee_loop_value", "loop_return", "storage", "error_storage", "prefix" }) |name| {
        errdefer std.debug.print("G9c native case {s}\n", .{name});
        const path = try std.fmt.allocPrint(allocator, "tests/formal/call_projection_{s}.ora", .{name});
        defer allocator.free(path);
        var compilation = try compilePackage(path);
        defer compilation.deinit();
        const hir = try compilation.db.lowerToHir(compilation.root_module_id);
        try testing.expect(hir.diagnostics.isEmpty());
        var fresh = try proof_check.prepareCallCustody(allocator, hir.module.raw_module, "Ora.Generated.Obligations.G9a");
        defer fresh.deinit();
        if (std.mem.eql(u8, name, "loop_return")) {
            try testing.expect(fresh.collected.set.loop_summaries[0].preheader_projection_complete);
            const source = try std.mem.concat(allocator, u8, &.{
                fresh.source,
                "\nopen Ora.Generated.Obligations.G9a\nexample : (emittedCallManifest.loop {} emittedLoopVcSha256_1 emittedLoopSummary_1).map (fun _ => ()) = .ok () := by rfl\n",
            });
            defer allocator.free(source);
            var tmp = testing.tmpDir(.{});
            defer tmp.cleanup();
            const lean_path = try pathFromTmpAlloc(allocator, tmp, "LoopReturnBinding.lean");
            defer allocator.free(lean_path);
            try std.Io.Dir.cwd().writeFile(std.testing.io, .{ .sub_path = lean_path, .data = source });
            const checked = try runLeanFileForTest(allocator, lean_path);
            defer allocator.free(checked.stdout);
            defer allocator.free(checked.stderr);
            try expectExited(checked, 0);
        }
        var native: obligation_to_z3.NativeCalls = .{ .set = fresh.collected.set, .function_ops = fresh.collected.function_ops, .source_fact_bindings = fresh.collected.source_fact_bindings };
        var pass = try z3_verification.VerificationPass.init(allocator);
        defer pass.deinit();
        const bindings = try test_helpers.z3FormalQueryBindingsFromFormalForTest(allocator, fresh.collected.query_bindings);
        defer allocator.free(bindings);
        pass.setFormalQueryBindings(bindings);
        pass.setCertifiedCalls(native.provider());
        var manifest = try pass.collectPreparedQueryManifest(hir.module.raw_module);
        defer manifest.deinit();
        try testing.expect(manifest.certified_calls.len > 0);
        var identity = try source_accounting_from_z3.collectPreparedIdentity(allocator, manifest.source_accounting_rows);
        defer identity.deinit(allocator);
        var result = try pass.runVerificationPass(hir.module.raw_module);
        defer result.deinit();
        try testing.expect(result.success);
        var report = try pass.buildSmtReport(hir.module.raw_module, path, &result);
        defer report.deinit(allocator);
        var overlay = try obligation_from_z3.overlayPreparedQueryResults(allocator, fresh.collected.set, report.query_manifest.?.rows);
        defer overlay.deinit();
        try overlay.set.validateIdReferences();
        try testing.expectEqual(obligation.ArtifactDecision{ .blocked = .call_lean_custody_missing }, overlay.set.artifactDecision());
        for (overlay.set.function_summaries) |row| {
            const query = obligation.findById(overlay.set.queries, row.query_ids.safety[0]).?;
            try testing.expectEqual(@as(?obligation.Id, row.id), query.function_summary_id);
            try testing.expectEqualSlices(u8, &try obligation_to_lean.functionVcSha256(fresh.collected.set, row.id), &query.function_vc_sha256.?);
            try testing.expect(query.smtlib_hash != null);
            try testing.expectEqual(obligation.VerificationQueryStatus.proved, query.result.?.status);
            try testing.expectEqual(obligation.ProofRequirement.lean_certificate, obligation.findById(overlay.set.queries, row.query_ids.certificate.?).?.proof_requirement);
        }
    }
}

test "G9c native call results storage and preconditions remain joint and prefix staged" {
    const cases = [_]struct { source: []const u8, succeeds: bool }{
        .{ .source = "contract C { fn next(x: u256) -> u256 { return x +% 1; } pub fn run(x: u256) { var i: u256 = 0; while (i < 1) invariant i <= 1 { let a: u256 = next(x); let b: u256 = next(a); assert(b == x +% 2); i = i +% 1; } } }", .succeeds = true },
        .{ .source = "contract C { fn next(x: u256) -> u256 { return x +% 1; } pub fn run(x: u256) { var i: u256 = 0; while (i < 1) invariant i <= 1 { let a: u256 = next(x); let b: u256 = next(a); assert(b == a); i = i +% 1; } } }", .succeeds = false },
        .{ .source = "contract C { storage var n: u256; fn next() -> u256 { let before: u256 = n; n = before +% 1; return before; } pub fn run() { var i: u256 = 0; while (i < 1) invariant i <= 1 { n = 0; let a: u256 = next(); let b: u256 = next(); assert(a == 0); assert(b == 1); assert(n == 2); i = i +% 1; } } }", .succeeds = true },
        .{ .source = "contract C { storage var n: u256; fn next() -> u256 { let before: u256 = n; n = before +% 1; return before; } pub fn run() { var i: u256 = 0; while (i < 1) invariant i <= 1 { n = 0; let a: u256 = next(); let b: u256 = next(); assert(b == 0); i = i +% 1; } } }", .succeeds = false },
        .{ .source = "contract C { fn leaf(x: u256) -> u256 requires x > 0 { return x; } pub fn run() { var i: u256 = 0; while (i < 1) invariant i <= 1 { let value: u256 = leaf(0); if (value == 1) { assert(true); } i = i +% 1; } } }", .succeeds = false },
        .{ .source = "contract C { fn leaf(x: u256) -> u256 requires x > 0 requires x < 3 { return x + 1; } pub fn run(x: u256) requires x > 0 requires x < 3 { var i: u256 = 0; while (i < 1) invariant i <= 1 { let value: u256 = leaf(x); assert(value == x + 1); i = i +% 1; } } }", .succeeds = true },
    };
    for (cases, 0..) |case, index| {
        errdefer std.debug.print("G9c native matrix case {d}\n", .{index});
        var compilation = try compileText(case.source);
        defer compilation.deinit();
        const hir = try compilation.db.lowerToHir(compilation.root_module_id);
        try testing.expect(hir.diagnostics.isEmpty());
        var projected = try obligation_from_mlir.collect(testing.allocator, hir.module.raw_module, .{ .project_calls = true });
        defer projected.deinit();
        try testing.expect(!projected.set.hasBlockingDiagnostic());
        var native: obligation_to_z3.NativeCalls = .{ .set = projected.set, .function_ops = projected.function_ops, .source_fact_bindings = projected.source_fact_bindings };
        var pass = try z3_verification.VerificationPass.init(testing.allocator);
        defer pass.deinit();
        pass.setCertifiedCalls(native.provider());
        var prepared = try pass.collectPreparedQueryManifest(hir.module.raw_module);
        defer prepared.deinit();
        try testing.expect(prepared.certified_calls.len > 0);
        var identity = try source_accounting_from_z3.collectPreparedIdentity(testing.allocator, prepared.source_accounting_rows);
        defer identity.deinit(testing.allocator);
        var result = try pass.runVerificationPass(hir.module.raw_module);
        defer result.deinit();
        try testing.expectEqual(case.succeeds, result.success);
        try testing.expectEqual(@as(usize, 0), result.proven_guard_ids.count());
    }
}

test "G9c nested call cannot erase an earlier native loop safety checkpoint" {
    const allocator = testing.allocator;
    for ([_]bool{ false, true }) |unsafe| {
        const source = try std.fmt.allocPrint(allocator, "contract C {{ fn spin(limit: u256) -> u256 requires limit == 0 {{ var j: u256 = 0; while (j <= limit) invariant j <= limit {{ j = j; }} return j; }} " ++
            "pub fn run(x: u256, limit: u256) requires limit == 0 {{ var i: u256 = 0; while (i < 1) invariant i <= 1 {{ if (x > 0) {{ assert(x {s} 0); let value: u256 = spin(limit); }} i = i +% 1; }} }} }}", .{if (unsafe) "==" else ">"});
        defer allocator.free(source);
        var compilation = try compileText(source);
        defer compilation.deinit();
        const hir = try compilation.db.lowerToHir(compilation.root_module_id);
        var projected = try obligation_from_mlir.collect(allocator, hir.module.raw_module, .{ .project_calls = true });
        defer projected.deinit();
        try testing.expect(!projected.set.hasBlockingDiagnostic());
        var native: obligation_to_z3.NativeCalls = .{ .set = projected.set, .function_ops = projected.function_ops, .source_fact_bindings = projected.source_fact_bindings };
        var pass = try z3_verification.VerificationPass.init(allocator);
        defer pass.deinit();
        pass.setCertifiedCalls(native.provider());
        var result = try pass.runVerificationPass(hir.module.raw_module);
        defer result.deinit();
        var report = try pass.buildSmtReport(hir.module.raw_module, "test.ora", &result);
        defer report.deinit(allocator);
        var checked: usize = 0;
        var failed = false;
        for (report.query_manifest.?.rows) |row| {
            if (!std.mem.eql(u8, row.function_name, "run") or row.kind != .LoopBodySafety) continue;
            checked += 1;
            try testing.expect(row.result_status == .sat or row.result_status == .unsat);
            failed = failed or row.result_status == .sat;
        }
        try testing.expect(checked > 0);
        try testing.expectEqual(unsafe, failed);
    }
}

test "G9c canonical loop step remains required independently of native call summaries" {
    const allocator = testing.allocator;
    for ([_]u8{ 1, 2 }) |increment| {
        const source = try std.fmt.allocPrint(allocator, "contract C {{ fn repeat(limit: u256) -> u256 requires limit <= 2 ensures result <= limit {{ var i: u256 = 0; while (i < limit) invariant i <= limit {{ i = i +% {d}; }} return i; }} pub fn run(limit: u256) -> u256 requires limit <= 2 ensures result <= limit {{ return repeat(limit); }} }}", .{increment});
        defer allocator.free(source);
        var compilation = try compileText(source);
        defer compilation.deinit();
        const hir = try compilation.db.lowerToHir(compilation.root_module_id);
        var projected = try obligation_from_mlir.collect(allocator, hir.module.raw_module, .{ .project_calls = true });
        defer projected.deinit();
        try testing.expect(!projected.set.hasBlockingDiagnostic());
        var native: obligation_to_z3.NativeCalls = .{ .set = projected.set, .function_ops = projected.function_ops, .source_fact_bindings = projected.source_fact_bindings };
        var pass = try z3_verification.VerificationPass.init(allocator);
        defer pass.deinit();
        const bindings = try test_helpers.z3FormalQueryBindingsFromFormalForTest(allocator, projected.query_bindings);
        defer allocator.free(bindings);
        pass.setFormalQueryBindings(bindings);
        pass.setCertifiedCalls(native.provider());
        var result = try pass.runVerificationPass(hir.module.raw_module);
        defer result.deinit();
        var report = try pass.buildSmtReport(hir.module.raw_module, "test.ora", &result);
        defer report.deinit(allocator);
        var overlay = try obligation_from_z3.overlayPreparedQueryResults(allocator, projected.set, report.query_manifest.?.rows);
        defer overlay.deinit();
        var steps: usize = 0;
        for (overlay.set.queries) |query| {
            if (query.kind != .loop_invariant_step or query.loop_summary_id == null) continue;
            steps += 1;
            try testing.expect(query.smtlib_hash != null);
            try testing.expectEqual(if (increment == 1) obligation.VerificationQueryStatus.proved else .failed, query.result.?.status);
            try testing.expect(!query.result.?.vacuous and !query.result.?.vacuity_unknown);
        }
        try testing.expectEqual(@as(usize, 1), steps);
    }
}

test "G9c supplied function proof checks against fresh HIR and exact FunctionVC targets" {
    const allocator = testing.allocator;
    var compilation = try compilePackage("tests/formal/call_adapter_leaf.ora");
    defer compilation.deinit();
    const hir = try compilation.db.lowerToHir(compilation.root_module_id);
    var fresh = try proof_check.prepareCallCustody(allocator, hir.module.raw_module, "Ora.Generated.Obligations.G9a");
    defer fresh.deinit();
    var overlay = try obligation_from_z3.overlayPreparedQueryResults(allocator, fresh.collected.set, &.{});
    defer overlay.deinit();
    const leaf = try callCustodyLeaf(overlay.set);
    const source = try callCustodyLeafProof(allocator, leaf);
    defer allocator.free(source);
    var tmp = testing.tmpDir(.{});
    defer tmp.cleanup();
    const path = try pathFromTmpAlloc(allocator, tmp, "Supplied.lean");
    defer allocator.free(path);
    try std.Io.Dir.cwd().writeFile(std.testing.io, .{ .sub_path = path, .data = source });
    const row: proof_check.ProofRow = .{ .query_id = leaf.query_ids.certificate.?, .obligation_ids = &.{}, .assumption_ids = leaf.assumption_ids, .module_name = "CustodyProof", .theorem_name = "CustodyProof.accepted", .path = path, .inline_source = true };
    const env = try callCustodyTestEnviron(allocator);
    defer env.block.deinit(allocator);
    var output = std.Io.Writer.Allocating.init(allocator);
    defer output.deinit();
    errdefer std.debug.print("G9c supplied checker: {s}\n", .{output.written()});
    var checked = try proof_check.applyCallProofRows(allocator, hir.module.raw_module, overlay.set, &.{row}, "Ora.Generated.Obligations.G9a", env, &output.writer);
    defer checked.deinit();
    const proof = checked.set.queries[checked.set.queries.len - 1];
    try testing.expectEqual(@as(?obligation.Id, leaf.id), proof.function_summary_id);
    try testing.expectEqualSlices(u8, &try obligation_to_lean.functionVcSha256(fresh.collected.set, leaf.id), &proof.function_vc_sha256.?);
    try testing.expect(checked.set.artifactDecision() == .blocked);
    try testing.expect(std.mem.indexOf(u8, checked.certificate_json, "certified_call_installation") != null);
    const targets = try allocator.dupe(obligation.VerificationQuery, overlay.set.queries);
    defer allocator.free(targets);
    var stale = overlay.set;
    stale.queries = targets;
    for (targets) |*target| {
        if (target.function_summary_id == leaf.id) target.function_vc_sha256.?[0] ^= 1;
    }
    try testing.expectError(error.CallCustodyFunctionDigestMismatch, proof_check.applyCallProofRows(allocator, hir.module.raw_module, stale, &.{row}, "Ora.Generated.Obligations.G9a", env, &output.writer));
}

test "G9c complete supplied certificates bind every loop and function target to fresh HIR" {
    const allocator = testing.allocator;
    const fixtures = [_]struct { stem: []const u8, namespace: []const u8 }{
        .{ .stem = "call_adapter_leaf", .namespace = "Ora.Generated.Obligations.Source_call_adapter_leaf_a1f737dc97a8081" },
        .{ .stem = "call_projection_callee_loop_value", .namespace = "Ora.Generated.Obligations.Source_call_projection_callee_loop_value_9b67d620b2daaec5" },
        .{ .stem = "call_projection_loop_return", .namespace = "Ora.Generated.Obligations.Source_call_projection_loop_return_6670e546cbd63471" },
    };
    for (fixtures) |fixture| {
        errdefer std.debug.print("G9c complete fixture {s}\n", .{fixture.stem});
        const path = try std.fmt.allocPrint(allocator, "tests/formal/{s}.ora", .{fixture.stem});
        defer allocator.free(path);
        var compilation = try compilePackage(path);
        defer compilation.deinit();
        const hir = try compilation.db.lowerToHir(compilation.root_module_id);
        var fresh = try proof_check.prepareCallCustody(allocator, hir.module.raw_module, fixture.namespace);
        defer fresh.deinit();
        var native: obligation_to_z3.NativeCalls = .{ .set = fresh.collected.set, .function_ops = fresh.collected.function_ops, .source_fact_bindings = fresh.collected.source_fact_bindings };
        var pass = try z3_verification.VerificationPass.init(allocator);
        defer pass.deinit();
        const bindings = try test_helpers.z3FormalQueryBindingsFromFormalForTest(allocator, fresh.collected.query_bindings);
        defer allocator.free(bindings);
        pass.setFormalQueryBindings(bindings);
        pass.setCertifiedCalls(native.provider());
        var prepared = try pass.collectPreparedQueryManifest(hir.module.raw_module);
        defer prepared.deinit();
        try testing.expect(prepared.certified_calls.len > 0);
        var result = try pass.runVerificationPass(hir.module.raw_module);
        defer result.deinit();
        try testing.expect(result.success);
        try testing.expectEqual(@as(usize, 0), result.proven_guard_ids.count());
        var report = try pass.buildSmtReport(hir.module.raw_module, path, &result);
        defer report.deinit(allocator);
        var overlay = try obligation_from_z3.overlayPreparedQueryResults(allocator, fresh.collected.set, report.query_manifest.?.rows);
        defer overlay.deinit();
        const manifest_path = try std.fmt.allocPrint(allocator, "tests/formal/{s}.proofs.json", .{fixture.stem});
        defer allocator.free(manifest_path);
        const bytes = try readFileAllocForTest(allocator, manifest_path);
        defer allocator.free(bytes);
        var manifest = try @import("formal/proof_manifest.zig").parseProofManifestJson(allocator, bytes);
        defer manifest.deinit();
        const env = try callCustodyTestEnviron(allocator);
        defer env.block.deinit(allocator);
        var output = std.Io.Writer.Allocating.init(allocator);
        defer output.deinit();
        errdefer std.debug.print("{s}\n", .{output.written()});
        // Mutate only the native receipt/join, keeping real prepared Z3 rows.
        // Neither an absent binding nor an empty canonical registry can hide a use.
        const receipts = try allocator.dupe(@TypeOf(report.query_manifest.?.certified_calls[0]), report.query_manifest.?.certified_calls);
        defer allocator.free(receipts);
        try testing.expect(receipts.len > 0);
        for (0..5) |mutation| {
            @memcpy(receipts, report.query_manifest.?.certified_calls);
            var mismatched = report;
            mismatched.query_manifest.?.certified_calls = receipts;
            var collected = fresh.collected;
            switch (mutation) {
                0 => receipts[0].source_op_id = 0,
                1 => receipts[0].function_id = 0,
                2 => receipts[0].digest[0] ^= 1,
                3 => collected.call_bindings = &.{},
                4 => collected.set.function_summaries = &.{},
                else => unreachable,
            }
            const expected = switch (mutation) {
                2 => error.CallInvocationDigestMismatch,
                4 => error.InvalidDependency,
                else => error.CallInvocationBindingMissing,
            };
            try testing.expectError(expected, @import("formal/userland/coordinator.zig").coordinate(
                allocator,
                path,
                &collected,
                mismatched,
                .{ .manifest = manifest_path },
                .{ .process_environ = env, .module = hir.module.raw_module },
                &output.writer,
            ));
        }
        var checked = try proof_check.applyCallProofRows(allocator, hir.module.raw_module, overlay.set, manifest.rows, fixture.namespace, env, &output.writer);
        defer checked.deinit();
        try testing.expectEqual(@as(usize, 3), checked.applied_count);
        try testing.expectEqual(obligation.ArtifactDecision.allowed, checked.set.artifactDecision());
        try g9cProofFailureMatrix(allocator, checked.set);
        try testing.expect(std.mem.indexOf(u8, checked.certificate_json, "sorryAx") == null);
        try testing.expect(std.mem.indexOf(u8, checked.certificate_json, "Classical.choice") == null);
        const queries = try allocator.dupe(obligation.VerificationQuery, overlay.set.queries);
        defer allocator.free(queries);
        var stale = overlay.set;
        stale.queries = queries;
        for (queries) |*query| if (query.loop_summary_id == 1) {
            query.loop_vc_sha256.?[0] ^= 1;
        };
        // All loop queries agree with one another; only fresh HIR can reject this stale identity.
        try stale.validateIdReferences();
        try testing.expectError(error.CallCustodyLoopDigestMismatch, proof_check.applyCallProofRows(allocator, hir.module.raw_module, stale, manifest.rows, fixture.namespace, env, &output.writer));
        for ([_]u8{ 0, 1, 2 }) |mutation| {
            const extra = [_]obligation.Id{if (overlay.set.assumptions.len > 0) overlay.set.assumptions[0].id else 0};
            @memcpy(queries, overlay.set.queries);
            for (queries) |*query| if (query.loop_summary_id == 1) {
                switch (mutation) {
                    0 => query.owner = .{ .function = .{ .name = "foreign" } },
                    1 => query.logical_role = .requires,
                    2 => query.assumption_ids = if (query.assumption_ids.len > 0) &.{} else &extra,
                    else => unreachable,
                }
            };
            if (mutation == 2 and overlay.set.assumptions.len == 0) continue;
            try stale.validateIdReferences();
            try testing.expectError(if (mutation == 2) error.CallCustodyManifestMismatch else error.CallCustodyLoopTargetMismatch, proof_check.applyCallProofRows(allocator, hir.module.raw_module, stale, manifest.rows, fixture.namespace, env, &output.writer));
        }
    }
}

test "G9c external CLI rejects mixed cycles and external calls before native encoding" {
    const allocator = testing.allocator;
    const cases = [_]struct { source: []const u8, reason: []const u8 }{
        .{ .source = "contract C { fn repeat() { var i: u256 = 0; while (i < 1) invariant i <= 1 { repeat(); i = i +% 1; } } pub fn run() { repeat(); } }", .reason = "call_projection_CyclicCallDependency" },
        .{ .source = "extern trait Hook { call fn ping(self) -> bool; } error ExternalCallFailed; contract C { storage var hook: address; fn invoke() -> !u256 | ExternalCallFailed { let ok: bool = try external<Hook>(hook, gas: 50000).ping(); return 1; } pub fn run(x: u256) -> !u256 | ExternalCallFailed { var i: u256 = 0; while (i < 1) invariant i <= 1 { let value: u256 = try invoke(); i = i +% 1; } return i; } }", .reason = "call_projection_CallExternalUnsupported" },
    };
    for (cases) |case| {
        var tmp = testing.tmpDir(.{});
        defer tmp.cleanup();
        try tmp.dir.writeFile(std.testing.io, .{ .sub_path = "rejected.ora", .data = case.source });
        const path = try pathFromTmpAlloc(allocator, tmp, "rejected.ora");
        defer allocator.free(path);
        const out = try pathFromTmpAlloc(allocator, tmp, "out");
        defer allocator.free(out);
        const result = try runProcess(allocator, &.{ ORA_BINARY_REL, "emit", "--lean-proofs=tests/formal/call_adapter_leaf.proofs.json", "--emit=smt-report,bytecode", "--out-dir", out, path });
        defer allocator.free(result.stdout);
        defer allocator.free(result.stderr);
        errdefer std.debug.print("G9c named rejection {s}\n{s}\n{s}\n", .{ case.reason, result.stdout, result.stderr });
        try expectExited(result, 1);
        try testing.expect(std.mem.indexOf(u8, result.stdout, case.reason) != null);
        try testing.expectError(error.FileNotFound, tmp.dir.access(std.testing.io, "out/rejected.hex", .{}));
    }
}

test "G9c external CLI requires every supplied proof before emitting bytecode" {
    const allocator = testing.allocator;
    for ([_][]const u8{ "call_adapter_leaf", "call_projection_callee_loop_value", "call_projection_loop_return" }) |stem| {
        const source_path = try std.fmt.allocPrint(allocator, "tests/formal/{s}.ora", .{stem});
        defer allocator.free(source_path);
        const original_path = try std.fmt.allocPrint(allocator, "tests/formal/{s}.proofs.json", .{stem});
        defer allocator.free(original_path);
        const bytes = try readFileAllocForTest(allocator, original_path);
        defer allocator.free(bytes);
        var manifest = try @import("formal/proof_manifest.zig").parseProofManifestJson(allocator, bytes);
        defer manifest.deinit();
        var tmp = testing.tmpDir(.{});
        defer tmp.cleanup();
        // Positive; independently omit each LoopVC or FunctionVC row; stale proof content.
        for (0..5) |case| {
            var rows: std.ArrayList(proof_check.ProofRow) = .empty;
            defer rows.deinit(allocator);
            for (manifest.rows, 0..) |row, index| {
                if (case > 0 and case < 4 and index == case - 1) continue;
                var selected = row;
                if (case == 4) selected.content_sha256 = "0000000000000000000000000000000000000000000000000000000000000000";
                try rows.append(allocator, selected);
            }
            const json = try std.json.Stringify.valueAlloc(allocator, .{ .schema_version = 1, .proofs = rows.items }, .{});
            defer allocator.free(json);
            const manifest_name = try std.fmt.allocPrint(allocator, "proofs-{d}.json", .{case});
            defer allocator.free(manifest_name);
            try tmp.dir.writeFile(std.testing.io, .{ .sub_path = manifest_name, .data = json });
            const manifest_path = try pathFromTmpAlloc(allocator, tmp, manifest_name);
            defer allocator.free(manifest_path);
            const option = try std.fmt.allocPrint(allocator, "--lean-proofs={s}", .{manifest_path});
            defer allocator.free(option);
            const out_name = try std.fmt.allocPrint(allocator, "out-{d}", .{case});
            defer allocator.free(out_name);
            const out = try pathFromTmpAlloc(allocator, tmp, out_name);
            defer allocator.free(out);
            const result = try runProcessWithLeanPath(allocator, &.{ ORA_BINARY_REL, "emit", option, "--emit=smt-report,sir-text,bytecode", "--out-dir", out, source_path });
            defer allocator.free(result.stdout);
            defer allocator.free(result.stderr);
            errdefer std.debug.print("G9c CLI {s} case {d}\n{s}\n{s}\n", .{ stem, case, result.stdout, result.stderr });
            try expectExited(result, if (case == 0) 0 else 1);
            const hex_path = try std.fmt.allocPrint(allocator, "{s}/{s}.hex", .{ out_name, stem });
            defer allocator.free(hex_path);
            if (case == 0) {
                try tmp.dir.access(std.testing.io, hex_path, .{});
                try testing.expect(std.mem.indexOf(u8, result.stdout, "Lean dispatcher userland proof accepted") != null);
                if (std.mem.eql(u8, stem, "call_projection_callee_loop_value")) {
                    const reference = try pathFromTmpAlloc(allocator, tmp, "runtime-reference");
                    defer allocator.free(reference);
                    const unchecked = try runProcess(allocator, &.{ ORA_BINARY_REL, "emit", "--no-verify", "--emit=sir-text,bytecode", "--out-dir", reference, source_path });
                    defer allocator.free(unchecked.stdout);
                    defer allocator.free(unchecked.stderr);
                    try expectExited(unchecked, 0);
                    for ([_][]const u8{ "sir", "hex" }) |extension| {
                        const verified_path = try std.fmt.allocPrint(allocator, "{s}/{s}.{s}", .{ out, stem, extension });
                        defer allocator.free(verified_path);
                        const reference_path = try std.fmt.allocPrint(allocator, "{s}/{s}.{s}", .{ reference, stem, extension });
                        defer allocator.free(reference_path);
                        const verified = try readFileAllocForTest(allocator, verified_path);
                        defer allocator.free(verified);
                        const retained = try readFileAllocForTest(allocator, reference_path);
                        defer allocator.free(retained);
                        try testing.expectEqualStrings(retained, verified);
                        if (std.mem.eql(u8, extension, "sir")) try testing.expect(std.mem.count(u8, verified, "revert 0x0 0x0") >= 2);
                    }
                }
            } else {
                try testing.expectError(error.FileNotFound, tmp.dir.access(std.testing.io, hex_path, .{}));
                const reason = if (case == 4) "ProofRowDigestMismatch" else if (case == 1) "lean_required_failure" else "call_lean_custody_missing";
                try testing.expect(std.mem.indexOf(u8, result.stdout, reason) != null or std.mem.indexOf(u8, result.stderr, reason) != null);
            }
        }
    }
}

// Constructed result mutations, not solver-produced UNKNOWN/timeouts. Each starts
// from actual Z3 results and a complete kernel-checked supplied certificate set.
fn g9cProofFailureMatrix(allocator: std.mem.Allocator, accepted: obligation.ObligationSet) !void {
    const queries = try allocator.dupe(obligation.VerificationQuery, accepted.queries);
    defer allocator.free(queries);
    var changed = accepted;
    changed.queries = queries;
    for (accepted.queries, 0..) |query, index| {
        if (query.loop_summary_id == null and query.function_summary_id == null) continue;
        if (query.backend == .z3) {
            const missing: obligation.ArtifactBlockReason = if (query.function_summary_id != null) .call_z3_relation_missing else .call_loop_obligation_missing;
            const cases = [_]struct { result: ?obligation.VerificationQueryResult, expected: obligation.ArtifactBlockReason }{
                .{ .result = null, .expected = missing },
                .{ .result = .{ .status = .proved, .degraded = true }, .expected = .degraded_query },
                .{ .result = .{ .status = .proved, .vacuous = true }, .expected = .vacuous_query },
                .{ .result = .{ .status = .proved, .vacuity_unknown = true }, .expected = .unknown_query },
                .{ .result = .{ .status = .unknown }, .expected = .unknown_query },
                .{ .result = .{ .status = .failed }, .expected = .failed_query },
            };
            for (cases) |case| {
                @memcpy(queries, accepted.queries);
                queries[index].result = case.result;
                errdefer std.debug.print("G9c mutated query {d} expected {s}\n", .{ query.id, @tagName(case.expected) });
                try testing.expectEqual(obligation.ArtifactDecision{ .blocked = case.expected }, changed.artifactDecision());
            }
            @memcpy(queries, accepted.queries);
            queries[index].smtlib_hash = null;
            try testing.expectEqual(obligation.ArtifactDecision{ .blocked = missing }, changed.artifactDecision());
            if (query.function_summary_id != null) {
                @memcpy(queries, accepted.queries);
                queries[index].function_vc_sha256.?[0] ^= 1;
                try testing.expectEqual(obligation.ArtifactDecision{ .blocked = .call_digest_mismatch }, changed.artifactDecision());
            }
        } else if (query.backend == .lean) {
            const missing: obligation.ArtifactBlockReason = if (query.function_summary_id != null) .call_lean_custody_missing else .lean_required_failure;
            for ([_]u8{ 0, 1, 2, 3 }) |mutation| {
                @memcpy(queries, accepted.queries);
                switch (mutation) {
                    0 => queries[index].result = null,
                    1 => queries[index].result = .{ .status = .failed },
                    2 => queries[index].proof_artifact_id = null,
                    3 => queries[index].artifact_policy = .diagnostic_only,
                    else => unreachable,
                }
                try testing.expectEqual(obligation.ArtifactDecision{ .blocked = missing }, changed.artifactDecision());
            }
        }
    }
    @memcpy(queries, accepted.queries);
    changed.proof_artifacts = &.{};
    try testing.expectEqual(obligation.ArtifactDecision{ .blocked = .invalid_dependency }, changed.artifactDecision());
}

fn callCustodyLeafProof(allocator: std.mem.Allocator, row: obligation.FunctionSummaryRow) ![]const u8 {
    var source = std.Io.Writer.Allocating.init(allocator);
    defer source.deinit();
    try source.writer.print(
        "namespace CustodyProof\nopen Ora.Loop Ora.Obligation Ora.Generated.Obligations.G9a\n" ++
            "def adapter := emittedCallManifest\ndef row := emittedFunctionSummary_{d}\n" ++
            "def digest := emittedFunctionVcSha256_{d}\ndef returnedTerm : Nat := {d}\n",
        .{ row.id, row.id, row.paths[0].returns.direct.values[0].value.formula.term },
    );
    try source.writer.writeAll(
        \\def body : AdapterBody := {
        \\  signature := row.callSignature
        \\  requires := row.entryFactsHold emittedManifest
        \\  paths := [{ events := [], returns := .direct .success [.term returnedTerm] }]
        \\  semantics := fun _ => (adapterSemantics {}
        \\    (directStorageContext emittedManifest Env.empty [])).withCallChecks }
        \\theorem bound : adapter.function {} digest row = .ok body := by rfl
        \\theorem certificate : body.Certificate where
        \\  safety := by
        \\    intro input typed requires initial entry path member before formula after events
        \\    simp only [body, List.mem_singleton] at member
        \\    subst path
        \\    simp at events
        \\  calls := by
        \\    intro input typed requires initial entry path member before invocation after events
        \\    simp only [body, List.mem_singleton] at member
        \\    subst path
        \\    simp at events
        \\  loopResults := by
        \\    intro input typed requires initial entry path member before child bindings after events
        \\    simp only [body, List.mem_singleton] at member
        \\    subst path
        \\    simp at events
        \\  returns := by
        \\    intro input initial path result typed requires entry member trace
        \\    simp only [body, List.mem_singleton] at member
        \\    subst path
        \\    cases trace
        \\    change row.callSignature.inputTyped input = true at typed
        \\    simp only [body, CallSignature.entry?, typed, if_true, Option.some.injEq] at entry
        \\    subst initial
        \\    refine ⟨{ outcome := .success, payload := [.integer (Ora.Integer.Value.ofNat (.unsigned .w256) 1)] }, ?_, ?_⟩
        \\    · rfl
        \\    · rfl
        \\def installed := (CallAdapterCertificates.empty adapter).addFunction digest row body bound certificate
    );
    try source.writer.print("\ntheorem accepted : emittedCallCustody_{d} := ⟨installed, rfl⟩\nend CustodyProof\n", .{row.id});
    return allocator.dupe(u8, source.written());
}

fn callCustodyTestEnviron(allocator: std.mem.Allocator) !std.process.Environ {
    var env = std.process.Environ.Map.init(allocator);
    defer env.deinit();
    const home_path = std.c.getenv("HOME") orelse return error.MissingTestHome;
    try env.put("HOME", std.mem.span(home_path));
    const path_arg = try leanPathEnvArgForTest(allocator);
    defer allocator.free(path_arg);
    try env.put("PATH", path_arg["PATH=".len..]);
    return .{ .block = try env.createPosixBlock(allocator, .{}) };
}

fn callCustodyLeaf(set: obligation.ObligationSet) !obligation.FunctionSummaryRow {
    for (set.function_summaries) |row| if (std.mem.eql(u8, row.owner.function.name, "yes")) return row;
    return error.MissingAdapterLeaf;
}

fn callCustodyInput(manifest: proof_check.CallCustodyManifest, proof: []const u8) !proof_check.CallCustodyInput {
    const row = try callCustodyLeaf(manifest.collected.set);
    return .{
        .function_id = row.id,
        .function_vc_sha256 = try obligation_to_lean.functionVcSha256(manifest.collected.set, row.id),
        .manifest_sha256 = manifest.sha256,
        .theorem_name = "CustodyProof.accepted",
        .proof_source = proof,
    };
}

test "G9a custody kernel checks a live compiler manifest deterministically without authority" {
    const allocator = testing.allocator;
    var compilation = try compilePackage("tests/formal/call_adapter_leaf.ora");
    defer compilation.deinit();
    const hir = try compilation.db.lowerToHir(compilation.root_module_id);
    try testing.expect(hir.diagnostics.isEmpty());
    const module = hir.module.raw_module;
    var manifest = try proof_check.prepareCallCustody(allocator, module, "Ora.Generated.Obligations.G9a");
    defer manifest.deinit();
    var repeated = try proof_check.prepareCallCustody(allocator, module, "Ora.Generated.Obligations.G9a");
    defer repeated.deinit();
    try testing.expectEqualStrings(manifest.source, repeated.source);
    try testing.expectEqualSlices(u8, &manifest.sha256, &repeated.sha256);
    const proof = try callCustodyLeafProof(allocator, try callCustodyLeaf(manifest.collected.set));
    defer allocator.free(proof);
    const input = try callCustodyInput(manifest, proof);
    const env = try callCustodyTestEnviron(allocator);
    defer env.block.deinit(allocator);
    var output = std.Io.Writer.Allocating.init(allocator);
    defer output.deinit();
    errdefer std.debug.print("G9a checker: {s}\n", .{output.written()});
    var first = try proof_check.checkCallCustody(allocator, module, "Ora.Generated.Obligations.G9a", input, env, &output.writer);
    defer first.deinit();
    var second = try proof_check.checkCallCustody(allocator, module, "Ora.Generated.Obligations.G9a", input, env, &output.writer);
    defer second.deinit();
    try testing.expectEqualSlices(u8, &first.manifest_sha256, &input.manifest_sha256);
    try testing.expectEqualSlices(u8, &first.function_vc_sha256, &input.function_vc_sha256);
    try testing.expectEqualSlices(u8, &first.proof_sha256, &second.proof_sha256);
    try testing.expectEqualSlices(u8, &first.checker_sha256, &second.checker_sha256);
    for (first.axioms) |axiom| try testing.expect(std.mem.eql(u8, axiom, "propext") or std.mem.eql(u8, axiom, "Quot.sound"));
    try testing.expect(std.mem.indexOf(u8, first.checker_source, ".emittedCallCustody_") != null);
    try testing.expect(std.mem.indexOf(u8, first.checker_source, "emittedVerifiedQuery_") == null);
    try testing.expectEqual(obligation.ArtifactDecision{ .blocked = .call_z3_relation_missing }, manifest.collected.set.artifactDecision());
}

test "G9a custody rejects stale foreign and forged manifests before kernel ingestion" {
    const allocator = testing.allocator;
    var compilation = try compilePackage("tests/formal/call_adapter_leaf.ora");
    defer compilation.deinit();
    const hir = try compilation.db.lowerToHir(compilation.root_module_id);
    const module = hir.module.raw_module;
    var manifest = try proof_check.prepareCallCustody(allocator, module, "Ora.Generated.Obligations.G9a");
    defer manifest.deinit();
    const original = try callCustodyInput(manifest, "theorem irrelevant : True := True.intro");
    const env = try callCustodyTestEnviron(allocator);
    defer env.block.deinit(allocator);
    var output = std.Io.Writer.Allocating.init(allocator);
    defer output.deinit();
    var changed = original;
    changed.function_id = std.math.maxInt(obligation.Id);
    try testing.expectError(error.CallCustodyFunctionMissing, proof_check.checkCallCustody(allocator, module, "Ora.Generated.Obligations.G9a", changed, env, &output.writer));
    changed = original;
    changed.function_vc_sha256[0] ^= 1;
    try testing.expectError(error.CallCustodyFunctionDigestMismatch, proof_check.checkCallCustody(allocator, module, "Ora.Generated.Obligations.G9a", changed, env, &output.writer));
    changed = original;
    changed.manifest_sha256[0] ^= 1;
    try testing.expectError(error.CallCustodyManifestMismatch, proof_check.checkCallCustody(allocator, module, "Ora.Generated.Obligations.G9a", changed, env, &output.writer));
    changed = original;
    changed.proof_source = "";
    try testing.expectError(error.CallCustodyProofMissing, proof_check.checkCallCustody(allocator, module, "Ora.Generated.Obligations.G9a", changed, env, &output.writer));
    try testing.expectError(error.CallCustodyManifestMismatch, proof_check.checkCallCustody(allocator, module, "Ora.Generated.Obligations.Foreign", original, env, &output.writer));
    // Omitting the caller body leaves the leaf's canonical digest unchanged.
    // Rehashing this forged manifest still cannot authenticate it to live HIR.
    var forged = manifest.collected.set;
    const leaf = try callCustodyLeaf(forged);
    forged.function_summaries = &.{leaf};
    var source = std.Io.Writer.Allocating.init(allocator);
    defer source.deinit();
    try obligation_to_lean.writeCallCustodyModule(&source.writer, forged, "Ora.Generated.Obligations.G9a");
    changed = original;
    std.crypto.hash.sha2.Sha256.hash(source.written(), &changed.manifest_sha256, .{});
    try testing.expectError(error.CallCustodyManifestMismatch, proof_check.checkCallCustody(allocator, module, "Ora.Generated.Obligations.G9a", changed, env, &output.writer));
    try testing.expectEqual(@as(usize, 0), output.written().len);
}

test "G9a custody rejects missing installation foreign proofs and forbidden axioms" {
    const allocator = testing.allocator;
    var compilation = try compilePackage("tests/formal/call_adapter_leaf.ora");
    defer compilation.deinit();
    const hir = try compilation.db.lowerToHir(compilation.root_module_id);
    var manifest = try proof_check.prepareCallCustody(allocator, hir.module.raw_module, "Ora.Generated.Obligations.G9a");
    defer manifest.deinit();
    const id = (try callCustodyLeaf(manifest.collected.set)).id;
    const proofs = [_][]const u8{
        // Valid theorem, wrong certificate proposition.
        "theorem CustodyProof.accepted : True := True.intro\n",
        // A certified empty registry proves no requested installation.
        "theorem CustodyProof.accepted : Ora.Generated.Obligations.G9a.emittedCallCustody_{d} := ⟨Ora.Loop.CallAdapterCertificates.empty _, rfl⟩\n",
        "theorem CustodyProof.accepted : Ora.Generated.Obligations.G9a.emittedCallCustody_{d} := by sorry\n",
    };
    const env = try callCustodyTestEnviron(allocator);
    defer env.block.deinit(allocator);
    inline for (proofs, 0..) |template, index| {
        const proof = if (index == 0) try allocator.dupe(u8, template) else try std.fmt.allocPrint(allocator, template, .{id});
        defer allocator.free(proof);
        var output = std.Io.Writer.Allocating.init(allocator);
        defer output.deinit();
        const input = try callCustodyInput(manifest, proof);
        try testing.expectError(error.LeanProofCheckFailed, proof_check.checkCallCustody(allocator, hir.module.raw_module, "Ora.Generated.Obligations.G9a", input, env, &output.writer));
        try testing.expect(std.mem.indexOf(u8, output.written(), "lean_proof_rejected") != null);
        if (index == 2) try testing.expect(std.mem.indexOf(u8, output.written(), "sorryAx") != null);
        try testing.expectEqual(obligation.ArtifactDecision{ .blocked = .call_z3_relation_missing }, manifest.collected.set.artifactDecision());
    }
}

test "G9a custody recollects changed HIR and rejects a transplanted certificate" {
    const allocator = testing.allocator;
    const source = try readFileAllocForTest(allocator, "tests/formal/call_adapter_leaf.ora");
    defer allocator.free(source);
    const changed_source = try std.mem.replaceOwned(u8, allocator, source, "return 1;", "return 2;");
    defer allocator.free(changed_source);
    var original = try compileText(source);
    defer original.deinit();
    var changed = try compileText(changed_source);
    defer changed.deinit();
    const before = try original.db.lowerToHir(original.root_module_id);
    const after = try changed.db.lowerToHir(changed.root_module_id);
    try testing.expect(before.diagnostics.isEmpty() and after.diagnostics.isEmpty());
    var old_manifest = try proof_check.prepareCallCustody(allocator, before.module.raw_module, "Ora.Generated.Obligations.G9a");
    defer old_manifest.deinit();
    var new_manifest = try proof_check.prepareCallCustody(allocator, after.module.raw_module, "Ora.Generated.Obligations.G9a");
    defer new_manifest.deinit();
    const proof = try callCustodyLeafProof(allocator, try callCustodyLeaf(old_manifest.collected.set));
    defer allocator.free(proof);
    const env = try callCustodyTestEnviron(allocator);
    defer env.block.deinit(allocator);
    var output = std.Io.Writer.Allocating.init(allocator);
    defer output.deinit();
    try testing.expectError(error.CallCustodyFunctionDigestMismatch, proof_check.checkCallCustody(
        allocator,
        after.module.raw_module,
        "Ora.Generated.Obligations.G9a",
        try callCustodyInput(old_manifest, proof),
        env,
        &output.writer,
    ));
    // Refreshing both hashes cannot make the old semantic proof prove return 2.
    try testing.expectError(error.LeanProofCheckFailed, proof_check.checkCallCustody(
        allocator,
        after.module.raw_module,
        "Ora.Generated.Obligations.G9a",
        try callCustodyInput(new_manifest, proof),
        env,
        &output.writer,
    ));
}

test "G9a custody requires installation of the requested root rather than any certificate" {
    const allocator = testing.allocator;
    var compilation = try compilePackage("tests/formal/call_adapter_leaf.ora");
    defer compilation.deinit();
    const hir = try compilation.db.lowerToHir(compilation.root_module_id);
    var manifest = try proof_check.prepareCallCustody(allocator, hir.module.raw_module, "Ora.Generated.Obligations.G9a");
    defer manifest.deinit();
    const proof = try callCustodyLeafProof(allocator, try callCustodyLeaf(manifest.collected.set));
    defer allocator.free(proof);
    var input = try callCustodyInput(manifest, proof);
    const caller = for (manifest.collected.set.function_summaries) |row| {
        if (row.id != input.function_id) break row;
    } else return error.MissingCallCustodyCaller;
    input.function_id = caller.id;
    input.function_vc_sha256 = try obligation_to_lean.functionVcSha256(manifest.collected.set, caller.id);
    const env = try callCustodyTestEnviron(allocator);
    defer env.block.deinit(allocator);
    var output = std.Io.Writer.Allocating.init(allocator);
    defer output.deinit();
    try testing.expectError(error.LeanProofCheckFailed, proof_check.checkCallCustody(allocator, hir.module.raw_module, "Ora.Generated.Obligations.G9a", input, env, &output.writer));
}

test "G9a custody retains live projection and source entry scope rejections" {
    for ([_][]const u8{ "tests/formal/call_adapter_bool_return_blocked.ora", "tests/formal/call_entry_prefix_storage_fact_blocked.ora" }) |path| {
        var compilation = try compilePackage(path);
        defer compilation.deinit();
        const hir = try compilation.db.lowerToHir(compilation.root_module_id);
        try testing.expectError(error.CallCustodyProjectionRejected, proof_check.prepareCallCustody(testing.allocator, hir.module.raw_module, "Ora.Generated.Obligations.G9a"));
    }
}

test "G9a custody requires Z3 evidence and requires checked custody even for forged successes" {
    var result = try collectCallProjection(testing.allocator, "tests/formal/call_adapter_leaf.ora");
    defer result.deinit();
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const allocator = arena.allocator();
    var set = result.set;
    const rows = try allocator.dupe(obligation.FunctionSummaryRow, set.function_summaries);
    set.function_summaries = rows;
    const queries = try allocator.alloc(obligation.VerificationQuery, set.queries.len + rows.len);
    @memcpy(queries[0..set.queries.len], set.queries);
    const extra = queries[set.queries.len..];
    for (rows, extra, 0..) |*row, *query, index| {
        const id: obligation.Id = @intCast(100000 + index);
        row.query_ids.safety = try allocator.dupe(obligation.Id, &.{id});
        query.* = .{
            .id = id,
            .owner = row.owner,
            .source = row.source,
            .phase = .report,
            .origin = row.origin,
            .kind = .obligation,
            .backend = .z3,
            .smtlib_hash = 1,
            .result = .{ .status = .proved },
        };
    }
    set.queries = queries;
    const cases = [_]struct { value: ?obligation.VerificationQueryResult, reason: obligation.ArtifactBlockReason }{
        .{ .value = null, .reason = .call_z3_relation_missing },
        .{ .value = .{ .status = .unknown }, .reason = .unknown_query },
        .{ .value = .{ .status = .proved, .degraded = true }, .reason = .degraded_query },
        .{ .value = .{ .status = .proved, .vacuous = true }, .reason = .vacuous_query },
        .{ .value = .{ .status = .proved, .vacuity_unknown = true }, .reason = .unknown_query },
        .{ .value = .{ .status = .failed }, .reason = .failed_query },
        // Arbitrary successes and an uninstalled custody receipt confer no authority.
        .{ .value = .{ .status = .proved }, .reason = .call_lean_custody_missing },
    };
    for (cases) |case| {
        extra[0].result = case.value;
        try testing.expectEqual(obligation.ArtifactDecision{ .blocked = case.reason }, set.artifactDecision());
    }
    const original = extra[0];
    extra[0].backend = .lean;
    try testing.expectEqual(obligation.ArtifactDecision{ .blocked = .call_z3_relation_missing }, set.artifactDecision());
    extra[0] = original;
    extra[0].smtlib_hash = null;
    try testing.expectEqual(obligation.ArtifactDecision{ .blocked = .call_z3_relation_missing }, set.artifactDecision());
    extra[0] = original;
    extra[0].owner = .{ .function = .{ .name = "foreign" } };
    try testing.expectEqual(obligation.ArtifactDecision{ .blocked = .call_identity_mismatch }, set.artifactDecision());
    extra[0] = original;
    extra[0].artifact_policy = .diagnostic_only;
    try testing.expectEqual(obligation.ArtifactDecision{ .blocked = .call_z3_relation_missing }, set.artifactDecision());
}

test "G8 call adapter binds compiler emitted rows without artifact authority" {
    var first = try collectCallProjection(testing.allocator, "tests/formal/call_adapter_leaf.ora");
    defer first.deinit();
    var second = try collectCallProjection(testing.allocator, "tests/formal/call_adapter_leaf.ora");
    defer second.deinit();
    errdefer for (first.set.diagnostics) |diagnostic| std.debug.print("G8: {s}\n", .{diagnostic.message});
    try testing.expect(!first.set.hasBlockingDiagnostic());
    try first.set.validateIdReferences();
    try first.set.validateTermReferences();
    try obligation_to_lean.validateCallDependencyDigests(first.set);
    try testing.expectEqual(obligation.ArtifactDecision{ .blocked = .call_z3_relation_missing }, first.set.artifactDecision());
    var emitted = std.Io.Writer.Allocating.init(testing.allocator);
    defer emitted.deinit();
    var repeated = std.Io.Writer.Allocating.init(testing.allocator);
    defer repeated.deinit();
    try obligation_to_lean.writeModule(&emitted.writer, first.set, .{ .namespace = "Ora.Generated.G8" });
    try obligation_to_lean.writeModule(&repeated.writer, second.set, .{ .namespace = "Ora.Generated.G8" });
    try testing.expectEqualStrings(emitted.written(), repeated.written());
    var leaf: ?obligation.FunctionSummaryRow = null;
    for (first.set.function_summaries) |row| if (std.mem.eql(u8, row.owner.function.name, "yes")) {
        leaf = row;
    };
    const row = leaf orelse return error.MissingAdapterLeaf;
    try testing.expectEqual(@as(usize, 1), row.parameters.len);
    try testing.expectEqual(@as(usize, 1), row.paths.len);
    try testing.expectEqual(@as(usize, 0), row.paths[0].timeline.len);
    const formula = row.paths[0].returns.direct.values[0].value.formula.term;
    var source = std.Io.Writer.Allocating.init(testing.allocator);
    defer source.deinit();
    try source.writer.writeAll("import Ora.Loop.CallAdapter\n");
    try source.writer.writeAll(emitted.written());
    try source.writer.writeAll("\nnamespace Ora.Generated.G8\nopen Ora.Loop Ora.Obligation\ndef adapter : CallAdapterManifest := { formulas := emittedManifest, loops := [");
    for (first.set.loop_summaries, 0..) |loop, index| {
        if (index != 0) try source.writer.writeAll(", ");
        try source.writer.print("(emittedLoopVcSha256_{d}, emittedLoopSummary_{d})", .{ loop.id, loop.id });
    }
    try source.writer.writeAll("], functions := [");
    for (first.set.function_summaries, 0..) |function, index| {
        if (index != 0) try source.writer.writeAll(", ");
        try source.writer.print("(emittedFunctionVcSha256_{d}, emittedFunctionSummary_{d})", .{ function.id, function.id });
    }
    try source.writer.print("] }}\ndef row := emittedFunctionSummary_{d}\ndef digest := emittedFunctionVcSha256_{d}\ndef returnedTerm : Nat := {d}\n", .{ row.id, row.id, formula });
    try source.writer.writeAll(
        \\def body : AdapterBody := {
        \\  signature := row.callSignature
        \\  requires := row.entryFactsHold emittedManifest
        \\  paths := [{ events := [], returns := .direct .success [.term returnedTerm] }]
        \\  semantics := fun _ => (adapterSemantics {}
        \\    (directStorageContext emittedManifest Env.empty [])).withCallChecks }
        \\theorem bound : adapter.function {} digest row = .ok body := by rfl
        \\theorem certificate : body.Certificate where
        \\  safety := by
        \\    intro input typed requires initial entry path member before formula after events
        \\    simp only [body, List.mem_singleton] at member
        \\    subst path
        \\    simp at events
        \\  calls := by
        \\    intro input typed requires initial entry path member before invocation after events
        \\    simp only [body, List.mem_singleton] at member
        \\    subst path
        \\    simp at events
        \\  loopResults := by
        \\    intro input typed requires initial entry path member before child bindings after events
        \\    simp only [body, List.mem_singleton] at member
        \\    subst path
        \\    simp at events
        \\  returns := by
        \\    intro input initial path result typed requires entry member trace
        \\    simp only [body, List.mem_singleton] at member
        \\    subst path
        \\    cases trace
        \\    change row.callSignature.inputTyped input = true at typed
        \\    simp only [body, CallSignature.entry?, typed, if_true, Option.some.injEq] at entry
        \\    subst initial
        \\    refine ⟨{ outcome := .success, payload := [.integer (Ora.Integer.Value.ofNat (.unsigned .w256) 1)] }, ?_, ?_⟩
        \\    · rfl
        \\    · rfl
        \\def installed := (CallAdapterCertificates.empty adapter).addFunction digest row body bound certificate
        \\example : (adapter.function {} (String.mk (List.replicate 64 '0')) row).map (fun _ => ()) =
        \\    .error "call_adapter_function_digest_mismatch" := by rfl
        \\example : (adapter.function {} digest { row with owner := "foreign" }).map (fun _ => ()) =
        \\    .error "call_adapter_function_foreign" := by rfl
        \\example : ({ adapter with functions := [("malformed", row)] }.function {} "malformed" row).map (fun _ => ()) =
        \\    .error "call_adapter_manifest_malformed" := by rfl
        \\#print axioms installed
        \\end Ora.Generated.G8
        \\
    );
    var tmp = testing.tmpDir(.{});
    defer tmp.cleanup();
    const path = try pathFromTmpAlloc(testing.allocator, tmp, "G8ProjectedAdapter.lean");
    defer testing.allocator.free(path);
    try std.Io.Dir.cwd().writeFile(std.testing.io, .{ .sub_path = path, .data = source.written() });
    const checked = try runLeanFileForTest(testing.allocator, path);
    defer testing.allocator.free(checked.stdout);
    defer testing.allocator.free(checked.stderr);
    try expectExited(checked, 0);
}

test "G8 call adapter retains captured bool projection rejection" {
    try testing.expectError(error.InvalidCallSignature, collectCallProjection(testing.allocator, "tests/formal/call_adapter_bool_parameter_blocked.ora"));
}

test "G8 call adapter retains bool return projection rejection" {
    var result = try collectCallProjection(testing.allocator, "tests/formal/call_adapter_bool_return_blocked.ora");
    defer result.deinit();
    try testing.expect(result.set.hasBlockingDiagnostic());
    var named = false;
    for (result.set.diagnostics) |diagnostic| if (std.mem.eql(u8, diagnostic.message, "call_projection_InvalidCallReturn")) {
        named = true;
    };
    try testing.expect(named);
}

test "G7 source entry facts retain exact contract declarations and prefix calls" {
    for ([_][]const u8{ "tests/formal/call_projection_prefix.ora", "tests/formal/call_entry_scopes.ora" }) |path| {
        var result = try collectCallProjection(testing.allocator, path);
        defer result.deinit();
        errdefer {
            std.debug.print("G7 fixture {s}\n", .{path});
            for (result.set.diagnostics) |diagnostic| std.debug.print("  {s}\n", .{diagnostic.message});
        }
        try testing.expect(!result.set.hasBlockingDiagnostic());
        try testing.expectEqual(@as(usize, 2), result.set.function_summaries.len);
        for (result.set.loop_summaries) |loop| {
            try testing.expect(!loopSummaryHasReason(loop, .loop_header_fact_effect_unsupported));
            try testing.expect(loop.preheader_projection_complete);
            try testing.expect(loop.projectionSupported());
        }
        const scoped = std.mem.endsWith(u8, path, "call_entry_scopes.ora");
        for (result.set.function_summaries) |row| {
            try testing.expect(row.entry_scope != null);
            try testing.expectEqual(@as(usize, if (scoped) 1 else 0), row.entry_invariant_ids.len);
            if (scoped) {
                const fact = obligation.findById(result.set.obligations, row.entry_invariant_ids[0]).?;
                try testing.expectEqual(obligation.LogicalRole.contract_invariant, fact.kind.logical.role);
                try testing.expectEqualStrings(row.entry_scope.?, fact.owner.contract);
                try testing.expectEqual(@as(usize, 1), row.assumption_ids.len);
                var source_bound = false;
                for (result.source_fact_bindings) |binding| {
                    if (!std.mem.eql(u8, binding.kind, "contract_invariant")) continue;
                    if (std.mem.eql(u8, binding.owner_key.?, "module/contract:EntryScope")) {
                        try testing.expectEqualSlices(obligation.Id, binding.obligation_ids, row.entry_invariant_ids);
                        source_bound = true;
                    } else for (binding.obligation_ids) |id| try testing.expect(id != fact.id);
                }
                try testing.expect(source_bound);
            }
        }
        try result.set.validateIdReferences();
        try result.set.validateTermReferences();
        try obligation_to_lean.validateCallDependencyDigests(result.set);
        try testing.expectEqual(obligation.ArtifactDecision{ .blocked = .call_z3_relation_missing }, result.set.artifactDecision());
        var source = std.Io.Writer.Allocating.init(testing.allocator);
        defer source.deinit();
        try obligation_to_lean.writeModule(&source.writer, result.set, .{ .namespace = "Ora.Generated.EntryScope", .proof_surface = true });
        try source.writer.writeAll("example : Ora.Generated.EntryScope.emittedFunctionSummaries.all (fun row => (row.entryFormulas? Ora.Generated.EntryScope.emittedManifest).isSome) = true := by decide\n");
        var tmp = testing.tmpDir(.{});
        defer tmp.cleanup();
        const lean_path = try pathFromTmpAlloc(testing.allocator, tmp, "EntryScope.lean");
        defer testing.allocator.free(lean_path);
        try std.Io.Dir.cwd().writeFile(std.testing.io, .{ .sub_path = lean_path, .data = source.written() });
        const parsed = try runLeanFileForTest(testing.allocator, lean_path);
        defer testing.allocator.free(parsed.stdout);
        defer testing.allocator.free(parsed.stderr);
        try expectExited(parsed, 0);
    }
}

test "G6 call projection covers the diagnosed function bodies" {
    const names = [_][]const u8{ "leaf", "checked", "storage", "branch", "error", "callee_loop", "prefix", "loop_return", "callee_loop_value", "chain", "checked_branch", "nested", "error_storage" };
    var complete = true;
    for (names) |name| {
        const path = try std.fmt.allocPrint(testing.allocator, "tests/formal/call_projection_{s}.ora", .{name});
        defer testing.allocator.free(path);
        var result = try collectCallProjection(testing.allocator, path);
        defer result.deinit();
        var valid = !result.set.hasBlockingDiagnostic() and result.set.function_summaries.len >= 2;
        for (result.set.function_summaries) |row| valid = valid and row.projection_completeness == .complete and row.write_footprint_complete and row.paths.len != 0;
        result.set.validateIdReferences() catch |err| {
            std.debug.print("G6 {s}: validation {s}\n", .{ name, @errorName(err) });
            valid = false;
        };
        result.set.validateTermReferences() catch |err| {
            std.debug.print("G6 {s}: term validation {s}\n", .{ name, @errorName(err) });
            valid = false;
        };
        if (std.mem.eql(u8, name, "prefix")) {
            // G7 transports immutable header leaves past the fully inventoried call.
            const loop = result.set.loop_summaries[0];
            try testing.expectEqualSlices(obligation.LoopUnsupportedReason, &.{}, loop.unsupported_reasons);
            try testing.expect(loop.preheader_projection_complete);
            var prefix_calls: usize = 0;
            for (loop.preheader_timeline) |event| if (event.payload == .call) {
                prefix_calls += 1;
                try testing.expectEqual(@as(usize, 1), event.payload.call.arguments.len);
                try testing.expectEqual(@as(usize, 1), event.payload.call.results.len);
            };
            try testing.expectEqual(@as(usize, 1), prefix_calls);
            var requires: usize = 0;
            for (result.set.function_summaries) |row| requires += row.assumption_ids.len;
            try testing.expectEqual(@as(usize, 1), requires);
            try obligation_to_lean.validateCallDependencyDigests(result.set);
        } else try obligation_to_lean.validateCallDependencyDigests(result.set);
        if (!valid) {
            std.debug.print("G6 {s}: functions={d}\n", .{ name, result.set.function_summaries.len });
            for (result.set.diagnostics) |diagnostic| std.debug.print("  {s}\n", .{diagnostic.message});
            for (result.set.loop_summaries) |loop| std.debug.print("  loop {d}: {any}, {any}\n", .{ loop.id, loop.projection_completeness, loop.unsupported_reasons });
        }
        complete = complete and valid;
        if (valid) try testing.expectEqual(obligation.ArtifactDecision{ .blocked = .call_z3_relation_missing }, result.set.artifactDecision());
    }
    try testing.expect(complete);
}

test "G6 call projection mutations invalidate the enclosing call digest" {
    const Mutation = enum { call, write, checkpoint, arm, order };
    const cases = [_]struct { name: []const u8, mutation: Mutation }{
        .{ .name = "chain", .mutation = .call },
        .{ .name = "storage", .mutation = .write },
        .{ .name = "checked", .mutation = .checkpoint },
        .{ .name = "branch", .mutation = .arm },
        .{ .name = "checked", .mutation = .order },
    };
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const allocator = arena.allocator();
    for (cases) |case| {
        errdefer std.debug.print("G6 mutation {s} {any}\n", .{ case.name, case.mutation });
        const file = try std.fmt.allocPrint(allocator, "tests/formal/call_projection_{s}.ora", .{case.name});
        var result = try collectCallProjection(testing.allocator, file);
        defer result.deinit();
        try obligation_to_lean.validateCallDependencyDigests(result.set);
        var changed = result.set;
        const rows = try allocator.dupe(obligation.FunctionSummaryRow, result.set.function_summaries);
        changed.function_summaries = rows;
        var mutated = false;
        outer: for (rows) |*row| {
            if (case.mutation == .arm) {
                if (row.paths.len < 2) continue;
                row.paths = row.paths[0 .. row.paths.len - 1];
                mutated = true;
                break;
            }
            const paths = try allocator.dupe(obligation.FunctionPathRow, row.paths);
            row.paths = paths;
            for (paths) |*path| {
                var matching: std.ArrayList(usize) = .empty;
                for (path.timeline, 0..) |event, index| {
                    const matches = switch (case.mutation) {
                        .call => event.payload == .call,
                        .write => event.payload == .storage_write,
                        .checkpoint, .order => event.payload == .safety_checkpoint,
                        .arm => unreachable,
                    };
                    if (matches) try matching.append(allocator, index);
                }
                if (matching.items.len < @as(usize, if (case.mutation == .order) 2 else 1)) continue;
                var events = try allocator.dupe(obligation.LoopTimelineEvent, path.timeline);
                if (case.mutation == .order) {
                    std.mem.swap(obligation.LoopTimelineEvent, &events[matching.items[0]], &events[matching.items[1]]);
                } else {
                    const removed = matching.items[0];
                    std.mem.copyForwards(obligation.LoopTimelineEvent, events[removed..], events[removed + 1 ..]);
                    events = events[0 .. events.len - 1];
                }
                for (events, 0..) |*event, index| event.id = @intCast(index);
                path.timeline = events;
                mutated = true;
                break :outer;
            }
        }
        try testing.expect(mutated);
        try testing.expectError(error.StaleCallDependency, obligation_to_lean.validateCallDependencyDigests(changed));
    }
}

test "G6 call projection missing loop exports cannot authorize an artifact" {
    var result = try collectCallProjection(testing.allocator, "tests/formal/call_projection_callee_loop_value.ora");
    defer result.deinit();
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const allocator = arena.allocator();
    var changed = result.set;
    const rows = try allocator.dupe(obligation.FunctionSummaryRow, result.set.function_summaries);
    changed.function_summaries = rows;
    var mutated = false;
    outer: for (rows) |*row| {
        const paths = try allocator.dupe(obligation.FunctionPathRow, row.paths);
        row.paths = paths;
        for (paths) |*path| {
            const events = try allocator.dupe(obligation.LoopTimelineEvent, path.timeline);
            path.timeline = events;
            for (events) |*event| if (event.payload == .summary_dependency and event.payload.summary_dependency.result_bindings.?.len != 0) {
                event.payload.summary_dependency.result_bindings = &.{};
                mutated = true;
                break :outer;
            };
        }
    }
    try testing.expect(mutated);
    try testing.expectEqual(obligation.ArtifactDecision{ .blocked = .invalid_dependency }, changed.artifactDecision());
}

test "G6 call projection serializes deterministic Lean rows" {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const allocator = arena.allocator();
    var combined = std.Io.Writer.Allocating.init(allocator);
    try combined.writer.writeAll("import Ora.Loop.CallAdapter\n");
    const cases = [_][]const u8{ "branch", "checked_branch", "error_storage", "loop_return", "callee_loop_value", "chain" };
    for (cases, 0..) |name, index| {
        const file = try std.fmt.allocPrint(allocator, "tests/formal/call_projection_{s}.ora", .{name});
        var first = try collectCallProjection(testing.allocator, file);
        defer first.deinit();
        var second = try collectCallProjection(testing.allocator, file);
        defer second.deinit();
        for (first.set.function_summaries) |row| try testing.expectEqualSlices(
            u8,
            &(try obligation_to_lean.functionVcSha256(first.set, row.id)),
            &(try obligation_to_lean.functionVcSha256(second.set, row.id)),
        );
        for (first.set.loop_summaries) |row| try testing.expectEqualSlices(
            u8,
            &(try obligation_to_lean.loopVcSha256(first.set, row.id)),
            &(try obligation_to_lean.loopVcSha256(second.set, row.id)),
        );
        var source = std.Io.Writer.Allocating.init(allocator);
        const namespace = try std.fmt.allocPrint(allocator, "Ora.Generated.G6.case_{d}", .{index});
        // Armed post queries reference source ensures rows. Use the complete
        // custody input instead of dropping obligations from a query subset.
        try obligation_to_lean.writeCallCustodyModule(&source.writer, first.set, namespace);
        var lines = std.mem.splitScalar(u8, source.written(), '\n');
        while (lines.next()) |line| if (!std.mem.startsWith(u8, line, "import ")) {
            try combined.writer.print("{s}\n", .{line});
        };
        try combined.writer.print("example : {s}.emittedFunctionSummaries.all Ora.Loop.FunctionSummaryRow.shapeSupported = true := by decide\n", .{namespace});
    }
    var tmp = testing.tmpDir(.{});
    defer tmp.cleanup();
    const path = try pathFromTmpAlloc(allocator, tmp, "G6ProjectedRows.lean");
    try std.Io.Dir.cwd().writeFile(std.testing.io, .{ .sub_path = path, .data = combined.written() });
    const checked = try runLeanFileForTest(testing.allocator, path);
    defer testing.allocator.free(checked.stdout);
    defer testing.allocator.free(checked.stderr);
    try expectExited(checked, 0);
}

test "G6 call projection rejects uninitialized private loads inside a loop" {
    const h = createContext();
    defer destroyContext(h);
    const body =
        \\module {
        \\  func.func @repeat() attributes {ora.return_compiler_type_id = 0 : i64} {
        \\    %zero = arith.constant 0 : i256
        \\    %one = arith.constant 1 : i256
        \\    %result = scf.while (%i = %zero) : (i256) -> i256 {
        \\      %guard = arith.cmpi ult, %i, %one : i256
        \\      scf.condition(%guard) %i : i256
        \\    } do {
        \\    ^bb0(%i: i256):
        \\      %inv = arith.cmpi ule, %i, %one : i256
        \\      ora.invariant %inv : i1
        \\      %slot = memref.alloca() : memref<i256>
        \\      memref.store %i, %slot[] : memref<i256>
        \\      %read = memref.load %slot[] : memref<i256>
        \\      %next = ora.add_wrapping %i, %one {ora.integer_signed = false} : i256, i256 -> i256
        \\      scf.yield %next : i256
        \\    } attributes {ora.loop_control_manifest = [], ora.result_integer_signed_0 = false, ora.result_name_0 = "i"} loc("ora.stmt.3"("ora.origin_stmt.3"("g6-slot.ora":3:1)))
        \\    func.return
        \\  }
        \\  func.func @caller() attributes {ora.return_compiler_type_id = 0 : i64} { func.call @repeat() : () -> ()
        \\    func.return }
        \\}
    ;
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    var good = true;
    for ([_]bool{ false, true }) |boolean| {
        const typed = if (boolean) try std.mem.replaceOwned(u8, arena.allocator(), body, "memref<i256>", "memref<i1>") else body;
        const initialized = if (boolean) try std.mem.replaceOwned(u8, arena.allocator(), typed, "memref.store %i,", "memref.store %inv,") else typed;
        const store = if (boolean) "memref.store %inv, %slot[] : memref<i1>" else "memref.store %i, %slot[] : memref<i256>";
        const uninitialized = try std.mem.replaceOwned(u8, arena.allocator(), initialized, store, "");
        const baseline_module = try parseModule(h.ctx, initialized);
        defer mlir.oraModuleDestroy(baseline_module);
        var baseline = try obligation_from_mlir.collect(testing.allocator, baseline_module, .{ .project_calls = true });
        defer baseline.deinit();
        try testing.expect(!baseline.set.hasBlockingDiagnostic());
        try baseline.set.validateIdReferences();
        try testing.expect(baseline.set.loop_summaries[0].projectionSupported());
        const module = try parseModule(h.ctx, uninitialized);
        defer mlir.oraModuleDestroy(module);
        if (obligation_from_mlir.collect(testing.allocator, module, .{ .project_calls = true })) |collected| {
            var result = collected;
            defer result.deinit();
            const rejected = result.set.hasBlockingDiagnostic() or !result.set.loop_summaries[0].projectionSupported();
            if (!rejected) std.debug.print("G6 uninitialized loop slot remains accepted: {s}\n", .{if (boolean) "bool" else "u256"});
            good = good and rejected;
        } else |err| try testing.expectEqual(error.CallPrivateSlotUninitialized, err);
    }
    try testing.expect(good);
}

test "G7 source entry rows agree with Lean on scoped rejection and bind every fact" {
    const owner: obligation.Owner = .{ .function = .{ .name = "leaf", .module = "m", .contract = "C" } };
    const base: obligation.FunctionSummaryRow = .{
        .id = 1,
        .owner = owner,
        .source = .generated(),
        .origin = .source,
        .structural_key = "leaf",
        .assumption_ids = &.{ 1, 2 },
        .entry_scope = "contract:1",
        .entry_invariant_ids = &.{ 3, 4 },
        .outcomes = &.{.{ .outcome = .success, .types = &.{} }},
        .paths = &.{.{ .id = 0, .structural_key = "return", .returns = .{ .direct = .{ .outcome = .success } } }},
        .projection_completeness = .complete,
        .write_footprint_complete = true,
    };
    const required = [_]obligation.Assumption{
        .{ .id = 1, .owner = owner, .source = .generated(), .phase = .ora_mlir, .origin = .source, .kind = .requires, .formula = .{ .term = 0 } },
        .{ .id = 2, .owner = owner, .source = .generated(), .phase = .ora_mlir, .origin = .source, .kind = .requires, .formula = .{ .term = 0 } },
        .{ .id = 6, .owner = .{ .function = .{ .name = "leaf", .module = "m", .contract = "D" } }, .source = .generated(), .phase = .ora_mlir, .origin = .source, .kind = .requires, .formula = .{ .term = 1 } },
    };
    const facts = [_]obligation.Obligation{
        .{ .id = 3, .owner = .{ .contract = "contract:1" }, .source = .generated(), .phase = .ora_mlir, .origin = .source, .kind = .{ .logical = .{ .role = .contract_invariant, .formula = .{ .term = 0 } } } },
        .{ .id = 4, .owner = .{ .contract = "contract:1" }, .source = .generated(), .phase = .ora_mlir, .origin = .source, .kind = .{ .logical = .{ .role = .contract_invariant, .formula = .{ .term = 0 } } } },
        .{ .id = 5, .owner = .{ .contract = "contract:2" }, .source = .generated(), .phase = .ora_mlir, .origin = .source, .kind = .{ .logical = .{ .role = .contract_invariant, .formula = .{ .term = 1 } } } },
    };
    var row = base;
    var assumptions = required;
    var obligations = facts;
    const set: obligation.ObligationSet = .{ .terms = &.{ .{ .bool_lit = true }, .{ .bool_lit = false } }, .assumptions = &assumptions, .obligations = &obligations, .function_summaries = (&row)[0..1] };
    const digest = try obligation_to_lean.functionVcSha256(set, 1);
    var source = std.Io.Writer.Allocating.init(testing.allocator);
    defer source.deinit();
    try obligation_to_lean.writeModule(&source.writer, set, .{ .namespace = "Ora.Generated.G7Entry" });
    try source.writer.writeAll("namespace Ora.Generated.G7Entry\n");
    const Mutation = enum { exact, requires_missing, requires_duplicate, requires_reordered, requires_foreign, module_foreign, requires_owner_tag, requires_kind, requires_no_formula, invariant_missing, invariant_duplicate, invariant_reordered, invariant_foreign, invariant_unknown, invariant_kind, invariant_owner_tag, scope_null, scope_empty, scope_foreign, requires_false, invariant_false };
    const cases = [_]struct { mutation: Mutation, row: []const u8 = "emittedFunctionSummary_1", manifest: []const u8 = "emittedManifest", valid: bool = false }{
        .{ .mutation = .exact, .valid = true },
        .{ .mutation = .requires_missing, .row = "{ emittedFunctionSummary_1 with assumptionIds := [1] }" },
        .{ .mutation = .requires_duplicate, .row = "{ emittedFunctionSummary_1 with assumptionIds := [1, 1] }" },
        .{ .mutation = .requires_reordered, .row = "{ emittedFunctionSummary_1 with assumptionIds := [2, 1] }" },
        .{ .mutation = .requires_foreign, .row = "{ emittedFunctionSummary_1 with assumptionIds := [1, 6] }" },
        .{ .mutation = .module_foreign, .row = "{ emittedFunctionSummary_1 with moduleName := some \"foreign\" }" },
        .{ .mutation = .requires_owner_tag, .manifest = "{ emittedManifest with assumptions := [{ emittedAssumptionRow_1 with functionOwner := false }, emittedAssumptionRow_2, emittedAssumptionRow_6] }" },
        .{ .mutation = .requires_kind, .manifest = "{ emittedManifest with assumptions := [{ emittedAssumptionRow_1 with kind := .assume }, emittedAssumptionRow_2, emittedAssumptionRow_6] }" },
        .{ .mutation = .requires_no_formula, .manifest = "{ emittedManifest with assumptions := [{ emittedAssumptionRow_1 with formula := none }, emittedAssumptionRow_2, emittedAssumptionRow_6] }" },
        .{ .mutation = .invariant_missing, .row = "{ emittedFunctionSummary_1 with entryInvariantIds := [3] }" },
        .{ .mutation = .invariant_duplicate, .row = "{ emittedFunctionSummary_1 with entryInvariantIds := [3, 3] }" },
        .{ .mutation = .invariant_reordered, .row = "{ emittedFunctionSummary_1 with entryInvariantIds := [4, 3] }" },
        .{ .mutation = .invariant_foreign, .row = "{ emittedFunctionSummary_1 with entryInvariantIds := [3, 5] }" },
        .{ .mutation = .invariant_unknown, .row = "{ emittedFunctionSummary_1 with entryInvariantIds := [3, 99] }" },
        .{ .mutation = .invariant_kind, .manifest = "{ emittedManifest with obligations := [{ emittedObligationRow_3 with kind := .logical .assert_ (.term 0) }, emittedObligationRow_4, emittedObligationRow_5] }" },
        .{ .mutation = .invariant_owner_tag, .manifest = "{ emittedManifest with obligations := [{ emittedObligationRow_3 with contractOwner := false }, emittedObligationRow_4, emittedObligationRow_5] }" },
        .{ .mutation = .scope_null, .row = "{ emittedFunctionSummary_1 with entryScope := none }" },
        .{ .mutation = .scope_empty, .row = "{ emittedFunctionSummary_1 with entryScope := some \"\" }" },
        .{ .mutation = .scope_foreign, .row = "{ emittedFunctionSummary_1 with entryScope := some \"contract:2\" }" },
        .{ .mutation = .requires_false, .valid = true, .manifest = "{ emittedManifest with assumptions := [{ emittedAssumptionRow_1 with formula := some (.term 1) }, emittedAssumptionRow_2, emittedAssumptionRow_6] }" },
        .{ .mutation = .invariant_false, .valid = true, .manifest = "{ emittedManifest with obligations := [{ emittedObligationRow_3 with kind := .logical .contractInvariant (.term 1) }, emittedObligationRow_4, emittedObligationRow_5] }" },
    };
    for (cases) |case| {
        row = base;
        assumptions = required;
        obligations = facts;
        switch (case.mutation) {
            .exact => {},
            .requires_missing => row.assumption_ids = &.{1},
            .requires_duplicate => row.assumption_ids = &.{ 1, 1 },
            .requires_reordered => row.assumption_ids = &.{ 2, 1 },
            .requires_foreign => row.assumption_ids = &.{ 1, 6 },
            .module_foreign => row.owner.function.module = "foreign",
            .requires_owner_tag => assumptions[0].owner = .{ .module = "leaf" },
            .requires_kind => assumptions[0].kind = .assume,
            .requires_no_formula => assumptions[0].formula = null,
            .invariant_missing => row.entry_invariant_ids = &.{3},
            .invariant_duplicate => row.entry_invariant_ids = &.{ 3, 3 },
            .invariant_reordered => row.entry_invariant_ids = &.{ 4, 3 },
            .invariant_foreign => row.entry_invariant_ids = &.{ 3, 5 },
            .invariant_unknown => row.entry_invariant_ids = &.{ 3, 99 },
            .invariant_kind => obligations[0].kind.logical.role = .assert,
            .invariant_owner_tag => obligations[0].owner = .{ .function = .{ .name = "contract:1" } },
            .scope_null => row.entry_scope = null,
            .scope_empty => row.entry_scope = "",
            .scope_foreign => row.entry_scope = "contract:2",
            .requires_false => assumptions[0].formula = .{ .term = 1 },
            .invariant_false => obligations[0].kind.logical.formula = .{ .term = 1 },
        }
        errdefer std.debug.print("G7 entry mutation {s}\n", .{@tagName(case.mutation)});
        if (case.valid) try set.validateIdReferences() else try testing.expectError(error.InvalidDependency, set.validateIdReferences());
        if (case.mutation != .exact and case.mutation != .invariant_unknown) {
            const changed = try obligation_to_lean.functionVcSha256(set, 1);
            try testing.expect(!std.mem.eql(u8, &digest, &changed));
        }
        try source.writer.print("example : (({s} : Ora.Loop.FunctionSummaryRow).entryFormulas? ({s} : Ora.Obligation.Manifest)).isSome = {s} := by decide\n", .{ case.row, case.manifest, if (case.valid) "true" else "false" });
    }
    try source.writer.writeAll("end Ora.Generated.G7Entry\n");
    var tmp = testing.tmpDir(.{});
    defer tmp.cleanup();
    const path = try pathFromTmpAlloc(testing.allocator, tmp, "G7Rows.lean");
    defer testing.allocator.free(path);
    try std.Io.Dir.cwd().writeFile(std.testing.io, .{ .sub_path = path, .data = source.written() });
    const checked = try runLeanFileForTest(testing.allocator, path);
    defer testing.allocator.free(checked.stdout);
    defer testing.allocator.free(checked.stderr);
    try expectExited(checked, 0);
}

test "G7 source entry prefix calls preserve only immutable header facts" {
    var positive = try collectCallProjection(testing.allocator, "tests/formal/call_entry_prefix_storage.ora");
    defer positive.deinit();
    try testing.expect(!positive.set.hasBlockingDiagnostic());
    const loop = positive.set.loop_summaries[0];
    try testing.expect(loop.projectionSupported());
    try testing.expectEqual(@as(usize, 1), loop.header_facts.len);
    var call_count: usize = 0;
    for (loop.preheader_timeline) |event| if (event.payload == .call) {
        call_count += 1;
        const callee = obligation.findById(positive.set.function_summaries, event.payload.call.callee_summary_id).?;
        try testing.expectEqual(@as(usize, 1), callee.write_footprint.len);
        try testing.expectEqualStrings("saved", callee.write_footprint[0].root);
    };
    try testing.expectEqual(@as(usize, 1), call_count);
    try positive.set.validateIdReferences();
    try obligation_to_lean.validateCallDependencyDigests(positive.set);
    var negative = try collectCallProjection(testing.allocator, "tests/formal/call_entry_prefix_storage_fact_blocked.ora");
    defer negative.deinit();
    try testing.expect(loopSummaryHasReason(negative.set.loop_summaries[0], .loop_header_fact_type_unsupported));
    try testing.expectEqual(@as(usize, 0), negative.set.loop_summaries[0].header_facts.len);
    try testing.expect(countAssumption(negative.set, .requires) > 0);
    try testing.expectError(error.InvalidDependency, obligation_to_lean.validateCallDependencyDigests(negative.set));
}

test "G7 source entry storage mode is scoped and retains invocation old" {
    const h = createContext();
    defer destroyContext(h);
    const text =
        \\module {
        \\  ora.contract @EntryStorage {
        \\    ora.global "counter" : !ora.int<256, false>
        \\    %current = ora.sload "counter" : !ora.int<256, false>
        \\    %old = ora.old %current : !ora.int<256, false> -> !ora.int<256, false>
        \\    %entry = ora.cmp "eq", %current, %old : !ora.int<256, false>, !ora.int<256, false> -> i1
        \\    "ora.invariant"(%entry) {ora.source_fact_id = 1 : i64, ora.source_fact_kind = "contract_invariant"} : (i1) -> ()
        \\    func.func @mutating() attributes {ora.write_slots = ["counter"], ora.write_slots_complete = true} {
        \\      %value = ora.sload "counter" : !ora.int<256, false>
        \\      %condition = ora.cmp "eq", %value, %value : !ora.int<256, false>, !ora.int<256, false> -> i1
        \\      "ora.requires"(%condition) : (i1) -> ()
        \\      "ora.assert"(%condition) : (i1) -> ()
        \\      func.return
        \\    }
        \\  }
        \\}
    ;
    const module = try parseModule(h.ctx, text);
    defer mlir.oraModuleDestroy(module);
    var projected = try obligation_from_mlir.collect(testing.allocator, module, .{ .project_calls = true });
    defer projected.deinit();
    const invariant = projected.set.obligations[0];
    try testing.expectEqual(obligation.LogicalRole.contract_invariant, invariant.kind.logical.role);
    const formula = projected.set.terms[invariant.kind.logical.formula.term].binary;
    try testing.expect(projected.set.terms[formula.lhs] == .place_read);
    try testing.expect(projected.set.terms[formula.rhs] == .old);
    try testing.expect(projected.set.terms[projected.set.terms[formula.rhs].old] == .place_read);
    try testing.expect(projected.set.assumptions[0].formula.? == .term);
    var body_assert: usize = 0;
    for (projected.set.obligations) |fact| if (fact.kind == .logical and fact.kind.logical.role == .assert) {
        body_assert += 1;
        try testing.expect(fact.kind.logical.formula == .origin_value);
    };
    try testing.expectEqual(@as(usize, 1), body_assert);
    var baseline = try obligation_from_mlir.collect(testing.allocator, module, .{});
    defer baseline.deinit();
    try testing.expectEqual(obligation.LogicalRole.invariant, baseline.set.obligations[0].kind.logical.role);
    try testing.expect(baseline.set.obligations[0].kind.logical.formula == .origin_value);
    try testing.expect(baseline.set.assumptions[0].formula.? == .origin_value);
}

test "G9b canonical leaf safety binds the independently kernel checked custody digest" {
    const allocator = testing.allocator;
    var compilation = try compilePackage("tests/formal/call_adapter_leaf.ora");
    defer compilation.deinit();
    const hir = try compilation.db.lowerToHir(compilation.root_module_id);
    var manifest = try proof_check.prepareCallCustody(allocator, hir.module.raw_module, "Ora.Generated.Obligations.G9a");
    defer manifest.deinit();
    const leaf = try callCustodyLeaf(manifest.collected.set);
    const proof = try callCustodyLeafProof(allocator, leaf);
    defer allocator.free(proof);
    const input = try callCustodyInput(manifest, proof);
    const env = try callCustodyTestEnviron(allocator);
    defer env.block.deinit(allocator);
    var output = std.Io.Writer.Allocating.init(allocator);
    defer output.deinit();
    var receipt = try proof_check.checkCallCustody(allocator, hir.module.raw_module, "Ora.Generated.Obligations.G9a", input, env, &output.writer);
    defer receipt.deinit();
    var projected = try collectCallProjection(allocator, "tests/formal/call_adapter_leaf.ora");
    defer projected.deinit();
    const independent_leaf = try callCustodyLeaf(projected.set);
    const digest = try obligation_to_lean.functionVcSha256(projected.set, independent_leaf.id);
    var context = try z3_verification.Z3Context.init(allocator);
    defer context.deinit();
    var adapter = obligation_to_z3.Adapter.init(&context, allocator, projected.set);
    const checked = try adapter.checkFunctionSafety(independent_leaf.id, digest, 100001);
    try testing.expectEqualSlices(u8, &receipt.function_vc_sha256, &checked.function_vc_sha256);
    try testing.expectEqual(@as(@TypeOf(checked.query.result.?.status), .proved), checked.query.result.?.status);
    var isolated_row = independent_leaf;
    isolated_row.query_ids.safety = &.{100001};
    var isolated = projected.set;
    isolated.function_summaries = (&isolated_row)[0..1];
    isolated.loop_summaries = &.{};
    isolated.queries = &.{checked.query};
    try testing.expectEqual(obligation.ArtifactDecision{ .blocked = .call_lean_custody_missing }, isolated.artifactDecision());
    var stale = digest;
    stale[0] ^= 1;
    try testing.expectError(error.FunctionVCDigestMismatch, adapter.checkFunctionSafety(independent_leaf.id, stale, 100001));
}

test "G9b projected call and loop dependencies prepare real queries without checked custody" {
    const cases = [_][]const u8{ "storage", "error_storage", "chain", "callee_loop_value", "loop_return", "nested", "prefix" };
    for (cases) |name| {
        var arena = std.heap.ArenaAllocator.init(testing.allocator);
        defer arena.deinit();
        const allocator = arena.allocator();
        const file = try std.fmt.allocPrint(allocator, "tests/formal/call_projection_{s}.ora", .{name});
        var projected = try collectCallProjection(testing.allocator, file);
        defer projected.deinit();
        try testing.expect(!projected.set.hasBlockingDiagnostic());
        var context = try z3_verification.Z3Context.init(allocator);
        defer context.deinit();
        var adapter = obligation_to_z3.Adapter.init(&context, allocator, projected.set);
        var queries: std.ArrayList(obligation.VerificationQuery) = .empty;
        try queries.appendSlice(allocator, projected.set.queries);
        const rows = try allocator.dupe(obligation.FunctionSummaryRow, projected.set.function_summaries);
        for (rows, 0..) |*row, index| {
            errdefer std.debug.print("G9b {s}/{s}\n", .{ name, row.owner.function.name });
            const digest = try obligation_to_lean.functionVcSha256(projected.set, row.id);
            const id: obligation.Id = @intCast(100001 + index);
            const checked = try adapter.checkFunctionSafety(row.id, digest, id);
            try testing.expectEqualSlices(u8, &digest, &checked.function_vc_sha256);
            try testing.expectEqual(@as(@TypeOf(checked.query.result.?.status), .proved), checked.query.result.?.status);
            try testing.expect(!checked.query.result.?.vacuous and !checked.query.result.?.vacuity_unknown);
            try testing.expect(checked.query.smtlib_hash != null);
            try queries.append(allocator, checked.query);
            row.query_ids.safety = try allocator.dupe(obligation.Id, &.{id});
        }
        for (projected.set.queries) |query| {
            if (query.loop_summary_id == null or query.backend != .z3) continue;
            errdefer std.debug.print("G9b {s} loop query {d} {s}\n", .{ name, query.id, @tagName(query.kind) });
            const checked = try adapter.checkLoopQueryForRow(query);
            try testing.expectEqual(@as(@TypeOf(checked.status), .proved), checked.status);
        }
        var prepared = projected.set;
        prepared.function_summaries = rows;
        prepared.queries = queries.items;
        try testing.expectEqual(obligation.ArtifactDecision{ .blocked = .call_z3_relation_missing }, projected.set.artifactDecision());
        try testing.expectEqual(obligation.ArtifactDecision{ .blocked = .call_lean_custody_missing }, prepared.artifactDecision());
    }
}

test "G9b composed calls retain invocation storage reads and returning loop frames" {
    var projected = try collectCallProjection(testing.allocator, "tests/formal/call_canonical_state.ora");
    defer projected.deinit();
    try testing.expect(!projected.set.hasBlockingDiagnostic());
    var context = try z3_verification.Z3Context.init(testing.allocator);
    defer context.deinit();
    var adapter = obligation_to_z3.Adapter.init(&context, testing.allocator, projected.set);
    try testing.expectEqual(@as(usize, 4), projected.set.function_summaries.len);
    const z3 = z3_verification.z3_c;
    for (projected.set.function_summaries) |row| {
        errdefer std.debug.print("G9b state {s}\n", .{row.owner.function.name});
        const checked = try adapter.checkFunctionSafety(row.id, try obligation_to_lean.functionVcSha256(projected.set, row.id), 100001);
        try testing.expectEqual(@as(@TypeOf(checked.query.result.?.status), .proved), checked.query.result.?.status);
        try testing.expect(!checked.query.result.?.vacuous);
        const vc = try adapter.encodeFunctionVC(row.id, checked.function_vc_sha256);
        var solver = try z3_verification.Z3Solver.initForLogic(&context, testing.allocator, "QF_AUFBV");
        defer solver.deinit();
        try solver.assertChecked(vc.entry);
        try solver.assertChecked(vc.joint);
        try testing.expectEqual(z3.Z3_L_TRUE, try solver.checkChecked());
        if (std.mem.eql(u8, row.owner.function.name, "sequential")) {
            const sort = z3.Z3_mk_bv_sort(context.ctx, 256);
            const payload = try adapter.functionOutput(row.id, "arm0.value0", sort);
            try solver.assertChecked(z3.Z3_mk_not(context.ctx, z3.Z3_mk_eq(context.ctx, payload, z3.Z3_mk_unsigned_int64(context.ctx, 1, sort))));
            try testing.expectEqual(z3.Z3_L_FALSE, try solver.checkChecked());
        }
    }
    for (projected.set.queries) |query| {
        if (query.loop_summary_id == null or query.backend != .z3) continue;
        const checked = try adapter.checkLoopQueryForRow(query);
        try testing.expectEqual(@as(@TypeOf(checked.status), .proved), checked.status);
    }
}

test "G9b callee preconditions cannot disappear behind a missing success return" {
    var projected = try collectCallProjection(testing.allocator, "tests/formal/call_canonical_precondition.ora");
    defer projected.deinit();
    try testing.expect(!projected.set.hasBlockingDiagnostic());
    var context = try z3_verification.Z3Context.init(testing.allocator);
    defer context.deinit();
    var adapter = obligation_to_z3.Adapter.init(&context, testing.allocator, projected.set);
    var rejected: usize = 0;
    for (projected.set.function_summaries) |row| {
        const checked = try adapter.checkFunctionSafety(row.id, try obligation_to_lean.functionVcSha256(projected.set, row.id), 100001);
        if (std.mem.eql(u8, row.owner.function.name, "run")) {
            try testing.expectEqual(@as(@TypeOf(checked.query.result.?.status), .failed), checked.query.result.?.status);
            rejected += 1;
        } else try testing.expectEqual(@as(@TypeOf(checked.query.result.?.status), .proved), checked.query.result.?.status);
        try testing.expect(!checked.query.result.?.vacuous);
    }
    try testing.expectEqual(@as(usize, 1), rejected);
}

test "G9b refreshed loop digests cannot hide a malformed exported return" {
    for ([_]bool{ false, true }) |wrong_type| {
        var projected = try collectCallProjection(testing.allocator, "tests/formal/call_projection_loop_return.ora");
        defer projected.deinit();
        var arena = std.heap.ArenaAllocator.init(testing.allocator);
        defer arena.deinit();
        const allocator = arena.allocator();
        var changed = false;
        outer: for (@constCast(projected.set.loop_summaries)) |*loop| {
            for (@constCast(loop.edges)) |*edge| {
                const returned = edge.return_value orelse continue;
                if (returned != .direct or returned.direct.values.len == 0) continue;
                const values = try allocator.dupe(obligation.LoopTypedFormula, returned.direct.values);
                if (wrong_type) values[0].ty = .{ .spelling = "bool" };
                edge.return_value = .{ .direct = .{ .outcome = returned.direct.outcome, .values = if (wrong_type) values else &.{} } };
                changed = true;
                break :outer;
            }
        }
        try testing.expect(changed);
        // Refresh every dependent selector: rejection must be semantic, not a
        // stale metadata failure masking an unchecked loop-return signature.
        try obligation_to_lean.bindCollectedCallDependencies(allocator, projected.set);
        const row = for (projected.set.function_summaries) |candidate| {
            if (std.mem.eql(u8, candidate.owner.function.name, "select")) break candidate;
        } else return error.MissingG9bFunction;
        var context = try z3_verification.Z3Context.init(allocator);
        defer context.deinit();
        var adapter = obligation_to_z3.Adapter.init(&context, allocator, projected.set);
        try testing.expectError(error.InvalidCallSignature, adapter.encodeFunctionVC(row.id, try obligation_to_lean.functionVcSha256(projected.set, row.id)));
    }
}

test "G9c caller boundary retains ordinary slice loops without scalarizing the caller" {
    var compilation = try compilePackage("tests/formal/call_projection_slice_caller.ora");
    defer compilation.deinit();
    const hir = try compilation.db.lowerToHir(compilation.root_module_id);
    var ordinary = try obligation_from_mlir.collect(testing.allocator, hir.module.raw_module, .{});
    defer ordinary.deinit();
    var projected = try obligation_from_mlir.collect(testing.allocator, hir.module.raw_module, .{ .project_calls = true });
    defer projected.deinit();
    try testing.expect(!ordinary.set.hasBlockingDiagnostic());
    try testing.expect(!projected.set.hasBlockingDiagnostic());
    try testing.expectEqual(@as(usize, 0), projected.set.function_summaries.len);
    try testing.expectEqual(@as(usize, 1), projected.set.loop_summaries.len);
    try testing.expect(projected.set.loop_summaries[0].projectionSupported());
    try testing.expectEqualSlices(
        u8,
        &(try obligation_to_lean.loopVcSha256(ordinary.set, ordinary.set.loop_summaries[0].id)),
        &(try obligation_to_lean.loopVcSha256(projected.set, projected.set.loop_summaries[0].id)),
    );
}

test "G9c caller boundary permits scalar actuals from a slice caller and rejects slice actuals" {
    var projected = try collectCallProjection(testing.allocator, "tests/formal/call_projection_slice_scalar_argument.ora");
    defer projected.deinit();
    try testing.expect(!projected.set.hasBlockingDiagnostic());
    try testing.expectEqual(@as(usize, 1), projected.set.function_summaries.len);
    const callee = projected.set.function_summaries[0];
    try testing.expectEqualStrings("repeat", callee.owner.function.name);
    try testing.expectEqual(@as(usize, 1), callee.parameters.len);
    try projected.set.validateIdReferences();
    try testing.expectError(error.InvalidCallSignature, collectCallProjection(
        testing.allocator,
        "tests/formal/call_projection_slice_argument_blocked.ora",
    ));
}

// Source-operation ordinals include non-obligation ops (such as a repeated
// index constant). Rename references bijectively, retaining the op name,
// symbol, all formulas, links and the distinction between different origins.
fn compoundRenameOrigins(value: *std.json.Value, names: *std.AutoHashMap(i64, i64)) !void {
    switch (value.*) {
        .object => |*object| {
            if (object.get("tag")) |tag| {
                if (tag == .string and std.mem.eql(u8, tag.string, "mlir_op")) {
                    const ordinal = object.getPtr("ordinal") orelse return error.TestUnexpectedResult;
                    const entry = try names.getOrPut(ordinal.integer);
                    if (!entry.found_existing) entry.value_ptr.* = @intCast(names.count());
                    ordinal.integer = entry.value_ptr.*;
                }
            }
            for (object.values()) |*child| try compoundRenameOrigins(child, names);
        },
        .array => |*array| for (array.items) |*child| try compoundRenameOrigins(child, names),
        else => {},
    }
}

fn compoundComparableDump(set: obligation.ObligationSet) ![]u8 {
    const a = testing.allocator;
    const raw = try dumpManifestToOwnedString(a, set);
    defer a.free(raw);
    var names = std.AutoHashMap(i64, i64).init(a);
    defer names.deinit();
    var output = std.Io.Writer.Allocating.init(a);
    errdefer output.deinit();
    var lines = std.mem.tokenizeScalar(u8, raw, '\n');
    while (lines.next()) |line| {
        var row = try std.json.parseFromSlice(std.json.Value, a, line, .{});
        defer row.deinit();
        try compoundRenameOrigins(&row.value, &names);
        try std.json.Stringify.value(row.value, .{}, &output.writer);
        try output.writer.writeByte('\n');
    }
    return output.toOwnedSlice();
}

fn compoundArithmeticProbe(op: []const u8, ty: []const u8, target: usize, compound: bool) ![]u8 {
    const a = testing.allocator;
    const declarations = [_][]const u8{ "", "", "", "storage var cell: T;", "storage var cells: [T; 2];", "storage var balances: map<address, T>;" };
    const initializers = [_][]const u8{ "var x: T = a;", "var values: [T; 2] = [a, a];", "var box: Box = Box { value: a };", "cell = a;", "cells[0] = a;", "balances[key] = a;" };
    const targets = [_][]const u8{ "x", "values[0]", "box.value", "cell", "cells[0]", "balances[key]" };
    const declaration = try std.mem.replaceOwned(u8, a, declarations[target], "T", ty);
    defer a.free(declaration);
    const initializer = try std.mem.replaceOwned(u8, a, initializers[target], "T", ty);
    defer a.free(initializer);
    const expression = if (compound)
        try std.fmt.allocPrint(a, "{s} {s}= b;", .{ targets[target], op })
    else
        try std.fmt.allocPrint(a, "{s} = {s} {s} b;", .{ targets[target], targets[target], op });
    defer a.free(expression);
    return std.fmt.allocPrint(a, "struct Box {{ value: {s}; }}\ncontract Probe {{ {s}\npub fn run(a: {s}, b: {s}, key: address) -> {s}\nrequires a >= 2\nrequires a <= 5\nrequires b >= 1\nrequires b <= 2\n{{\n{s}\n{s}\nreturn {s};\n}}\n}}", .{ ty, declaration, ty, ty, ty, initializer, expression, targets[target] });
}

test "compound arithmetic gate binds every source check to a real prepared query" {
    const a = testing.allocator;
    const operators = [_][]const u8{ "+", "-", "*", "/", "%", "**", "+%", "-%", "*%", "&", "|", "^", "<<", ">>" };
    for (operators) |op| {
        for ([_][]const u8{ "u8", "i8" }) |ty| {
            for (0..6) |target| {
                var ordinary_dump: ?[]u8 = null;
                defer if (ordinary_dump) |dump| a.free(dump);
                for ([_]bool{ false, true }) |compound| {
                    const source = try compoundArithmeticProbe(op, ty, target, compound);
                    defer a.free(source);
                    errdefer std.debug.print("compound arithmetic op={s} type={s} target={d} compound={}\n{s}\n", .{ op, ty, target, compound, source });
                    var compilation = try compileText(source);
                    defer compilation.deinit();
                    const hir = try compilation.db.lowerToHir(compilation.root_module_id);
                    try testing.expect(hir.diagnostics.isEmpty());
                    var collected = try test_helpers.collectFormalQueryBindingsForVerifier(a, hir.module.raw_module);
                    defer collected.formal_result.deinit();
                    defer a.free(collected.z3_bindings);
                    var pass = try z3_verification.VerificationPass.init(a);
                    defer pass.deinit();
                    pass.setFormalQueryBindings(collected.z3_bindings);
                    var checked = try pass.runVerificationPass(hir.module.raw_module);
                    defer checked.deinit();
                    var report = try pass.buildSmtReport(hir.module.raw_module, "test.ora", &checked);
                    defer report.deinit(a);
                    const prepared = report.query_manifest orelse return error.TestUnexpectedResult;
                    var overlay = try obligation_from_z3.overlayPreparedQueryResults(a, collected.formal_result.set, prepared.rows);
                    defer overlay.deinit();
                    if (overlay.set.hasBlockingDiagnostic()) {
                        for (overlay.set.diagnostics) |diagnostic| std.debug.print("compound binding diagnostic: {any}\n", .{diagnostic});
                    }
                    try testing.expect(!overlay.set.hasBlockingDiagnostic());
                    for (overlay.set.queries) |query| {
                        try testing.expectEqual(obligation.VerificationBackend.z3, query.backend);
                        try testing.expect(query.smtlib_hash != null);
                        try testing.expect(query.result != null);
                    }
                    // Missing evidence must still reject: a lowering repair cannot
                    // be implemented by silencing the unmatched-row diagnostic.
                    if (collected.formal_result.set.obligations.len != 0) {
                        var missing = try obligation_from_z3.overlayPreparedQueryResults(a, collected.formal_result.set, &.{});
                        defer missing.deinit();
                        try testing.expect(missing.set.hasBlockingDiagnostic());
                    }
                    // Compare the full pre-solver manifest under a source-location
                    // correspondence. No identity/digest used by a prover is edited.
                    const set = collected.formal_result.set;
                    for (@constCast(set.obligations)) |*row| row.source = .generated();
                    for (@constCast(set.assumptions)) |*row| row.source = .generated();
                    for (@constCast(set.queries)) |*row| row.source = .generated();
                    const dump = try compoundComparableDump(set);
                    if (ordinary_dump) |expected| {
                        defer a.free(dump);
                        try testing.expectEqualStrings(expected, dump);
                    } else ordinary_dump = dump;
                }
            }
        }
    }
}

test "compound arithmetic gate recovers exact signed and unsigned bitfield member types" {
    for ([_][]const u8{ "u8", "i8" }) |ty| {
        for ([_]bool{ false, true }) |generic| {
            const source = try std.fmt.allocPrint(testing.allocator, "bitfield Bits{s}: u256 {{ value: {s}; sibling: u8; }}\n" ++
                "contract Probe {{ storage var bits: Bits{s}; pub fn run(a: {s}, b: {s}) {{ bits.value = a; bits.value += b; }} }}", .{ if (generic) "(comptime T: type)" else "", if (generic) "T" else ty, if (generic) (if (std.mem.eql(u8, ty, "u8")) "<u8>" else "<i8>") else "", ty, ty });
            defer testing.allocator.free(source);
            var compilation = try compileText(source);
            defer compilation.deinit();
            const hir = try compilation.db.lowerToHir(compilation.root_module_id);
            try testing.expect(hir.diagnostics.isEmpty());
            const add = findFirstOpByName(hir.module.raw_module, "arith.addi") orelse return error.TestUnexpectedResult;
            const signed = mlir.oraOperationGetAttributeByName(add, strRef("ora.integer_signed"));
            try testing.expect(!mlir.oraAttributeIsNull(signed));
            try testing.expectEqual(std.mem.eql(u8, ty, "i8"), mlir.oraBoolAttrGetValue(signed));
            // Recover the member's i8 carrier, never the bitfield's u256 container.
            try testing.expectEqual(@as(i64, 8), mlir.oraIntegerTypeGetWidth(mlir.oraValueGetType(mlir.oraOperationGetResult(add, 0))));
        }
    }
}

test "compound arithmetic gate unresolved bitfield members still reject without bytecode" {
    const a = testing.allocator;
    var tmp = testing.tmpDir(.{});
    defer tmp.cleanup();
    for ([_][]const u8{ "missing", "flag" }) |member| {
        const source = try std.fmt.allocPrint(a, "bitfield Bits: u256 {{ flag: bool; }} contract Bad {{ storage var bits: Bits; pub fn run() {{ bits.{s} += 1; }} }}", .{member});
        defer a.free(source);
        try tmp.dir.writeFile(std.testing.io, .{ .sub_path = "bad.ora", .data = source });
        const path = try pathFromTmpAlloc(a, tmp, "bad.ora");
        defer a.free(path);
        const output = try pathFromTmpAlloc(a, tmp, member);
        defer a.free(output);
        const run = try runProcessWithLeanPath(a, &.{ ORA_BINARY_REL, "emit", "--emit=bytecode", "--out-dir", output, path });
        defer a.free(run.stdout);
        defer a.free(run.stderr);
        try expectExited(run, 1);
        const hex = try std.fmt.allocPrint(a, "{s}/bad.hex", .{member});
        defer a.free(hex);
        try testing.expectError(error.FileNotFound, tmp.dir.access(std.testing.io, hex, .{}));
        try testing.expect(std.mem.indexOf(u8, run.stderr, "error:") != null or std.mem.indexOf(u8, run.stdout, "error:") != null);
    }
}

test "C10 multiplication guard prefers literal divisors without rewriting the product" {
    const Case = struct { body: []const u8, swap: bool, signed: bool = false };
    for ([_]Case{
        .{ .body = "return 100 * a;", .swap = true },
        .{ .body = "return a * 100;", .swap = false },
        .{ .body = "return a * b;", .swap = false },
        .{ .body = "return 0 * a;", .swap = true },
        .{ .body = "return a * 0;", .swap = false },
        .{ .body = "var x: u256 = 100; x *= a; return x;", .swap = true },
        .{ .body = "var x: u256 = a; x *= 100; return x;", .swap = false },
        .{ .body = "return 100 * a;", .swap = false, .signed = true },
    }) |case| {
        const source = try std.fmt.allocPrint(testing.allocator, "pub fn run(a: {s}, b: {s}) -> {s} {{ {s} }}", .{
            if (case.signed) "i256" else "u256", if (case.signed) "i256" else "u256",
            if (case.signed) "i256" else "u256", case.body,
        });
        defer testing.allocator.free(source);
        errdefer std.debug.print("C10 body: {s}\n", .{case.body});
        var compilation = try compileText(source);
        defer compilation.deinit();
        const hir = try compilation.db.lowerToHir(compilation.root_module_id);
        try testing.expect(hir.diagnostics.isEmpty());
        const module = hir.module.raw_module;
        const product = findFirstOpByName(module, "arith.muli") orelse return error.TestUnexpectedResult;
        const quotient = findFirstOpByName(module, if (case.signed) "arith.divsi" else "arith.divui") orelse return error.TestUnexpectedResult;
        const divisor = mlir.oraOperationGetOperand(quotient, 1);
        try testing.expect(mlir.mlirValueEqual(divisor, mlir.oraOperationGetOperand(product, if (case.swap) 0 else 1)));
        const marker = mlir.oraOperationGetAttributeByName(quotient, strRef("ora.guard_internal"));
        try testing.expect(!mlir.oraAttributeIsNull(marker));
        const text = mlir.oraStringAttrGetValue(marker);
        try testing.expectEqualStrings("true", text.data[0..text.length]);
        try testing.expect(findFirstOpByName(module, "ora.assert") != null);
        if (!case.signed) {
            const nonzero = findFirstOpByName(module, "arith.cmpi") orelse return error.TestUnexpectedResult;
            try testing.expect(mlir.mlirValueEqual(divisor, mlir.oraOperationGetOperand(nonzero, 0)));
            const conjunction = findFirstOpByName(module, "arith.andi") orelse return error.TestUnexpectedResult;
            try testing.expect(mlir.mlirValueEqual(mlir.oraOperationGetResult(nonzero, 0), mlir.oraOperationGetOperand(conjunction, 0)));
            const comparison = mlir.oraOpResultGetOwner(mlir.oraOperationGetOperand(conjunction, 1));
            try testing.expect(mlir.mlirValueEqual(mlir.oraOperationGetOperand(comparison, 0), mlir.oraOperationGetOperand(product, if (case.swap) 1 else 0)));
            try testing.expect(mlir.mlirValueEqual(mlir.oraOperationGetOperand(comparison, 1), mlir.oraOperationGetResult(quotient, 0)));
        }
        var collected = try obligation_from_mlir.collect(testing.allocator, module, .{});
        defer collected.deinit();
        try testing.expectEqual(@as(usize, 1), countArithmeticSafety(collected.set, .multiplication_overflow));
        try testing.expectEqual(@as(usize, 0), countArithmeticSafety(collected.set, .division_by_zero));
    }
}

test "C10 multiplication canonical loop queries prove with real solver hashes" {
    for ([_][]const u8{ "tests/formal/mul_guard_loop.ora", "ora-example/verification/fv_test.ora" }) |path| {
        var result = try collectPackageObligations(testing.allocator, path);
        defer result.deinit();
        try testing.expectEqual(@as(usize, 1), result.set.loop_summaries.len);
        try testing.expect(result.set.loop_summaries[0].projectionSupported());
        var context = try z3_verification.Z3Context.init(testing.allocator);
        defer context.deinit();
        var adapter = obligation_to_z3.Adapter.init(&context, testing.allocator, result.set);
        var checked_count: usize = 0;
        for (result.set.queries) |query| {
            if (query.loop_summary_id == null or query.loop_vc_sha256 == null or query.backend != .z3) continue;
            const checked = try adapter.checkLoopQueryForRow(query);
            try testing.expectEqual(obligation_to_z3.CheckStatus.proved, checked.status);
            try testing.expect(checked.query_hash.constraint_count > 0);
            try testing.expect(checked.query_hash.smtlib_hash != 0);
            checked_count += 1;
        }
        try testing.expectEqual(@as(usize, 5), checked_count);
    }
}

test "Group 3 total unsigned bound retains independent producer safety" {
    const cases = [_]struct { body: []const u8, diagnostic: ?[]const u8 }{
        .{ .body = "pub fn f(x:u256) -> MinValue<u256,0> { return x; }", .diagnostic = null },
        .{ .body = "pub fn f(x:u256) -> MinValue<u256,0> { return x + 1; }", .diagnostic = "checked addition overflow" },
        .{ .body = "pub fn f(x:u256) -> MinValue<u256,0> { let a:[u256;2] = [1,2]; return a[x]; }", .diagnostic = "fixed-array index out of bounds" },
        .{ .body = "fn leaf(x:u256) -> u256 requires x > 0 { return x; } pub fn f(x:u256) -> MinValue<u256,0> { return leaf(x); }", .diagnostic = "callee" },
        .{ .body = "pub fn f(x:i256) -> MinValue<i256,0> { return x; }", .diagnostic = "refinement guard" },
        .{ .body = "pub fn f(x:u256) -> MinValue<u256,1> { return x; }", .diagnostic = "refinement guard" },
    };
    var tmp = testing.tmpDir(.{});
    defer tmp.cleanup();
    for (cases, 0..) |case, i| {
        const source = try std.fmt.allocPrint(testing.allocator, "contract BoundProducer {{ {s} }}", .{case.body});
        defer testing.allocator.free(source);
        const filename = try std.fmt.allocPrint(testing.allocator, "producer-{d}.ora", .{i});
        defer testing.allocator.free(filename);
        try tmp.dir.writeFile(std.testing.io, .{ .sub_path = filename, .data = source });
        const path = try pathFromTmpAlloc(testing.allocator, tmp, filename);
        defer testing.allocator.free(path);
        const output_name = try std.fmt.allocPrint(testing.allocator, "output-{d}", .{i});
        defer testing.allocator.free(output_name);
        const output = try pathFromTmpAlloc(testing.allocator, tmp, output_name);
        defer testing.allocator.free(output);
        const run = try runProcessWithLeanPath(testing.allocator, &.{ ORA_BINARY_REL, "emit", "--emit=smt-report,bytecode", "--out-dir", output, path });
        defer testing.allocator.free(run.stdout);
        defer testing.allocator.free(run.stderr);
        errdefer std.debug.print("producer case {d}\n{s}\n{s}", .{ i, run.stdout, run.stderr });
        try expectExited(run, if (case.diagnostic == null) 0 else 1);
        const hex = try std.fmt.allocPrint(testing.allocator, "{s}/producer-{d}.hex", .{ output_name, i });
        defer testing.allocator.free(hex);
        if (case.diagnostic) |diagnostic| {
            try testing.expectError(error.FileNotFound, tmp.dir.access(std.testing.io, hex, .{}));
            try testing.expect(std.mem.indexOf(u8, run.stdout, diagnostic) != null or std.mem.indexOf(u8, run.stderr, diagnostic) != null);
            const report_path = try std.fmt.allocPrint(testing.allocator, "{s}/producer-{d}.smt.report.json", .{ output_name, i });
            defer testing.allocator.free(report_path);
            const report = try tmp.dir.readFileAlloc(std.testing.io, report_path, testing.allocator, .limited(16 * 1024 * 1024));
            defer testing.allocator.free(report);
            const parsed = try std.json.parseFromSlice(std.json.Value, testing.allocator, report, .{ .allocate = .alloc_always });
            defer parsed.deinit();
            var saw_sat = false;
            for (parsed.value.object.get("queries").?.array.items) |row| {
                if (std.mem.eql(u8, row.object.get("status").?.string, "SAT")) saw_sat = true;
            }
            try testing.expect(saw_sat);
            try testing.expect(std.mem.indexOf(u8, report, "unmatched_report_row") == null);
        } else try tmp.dir.access(std.testing.io, hex, .{});
    }
}
