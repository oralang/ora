//! Emits `formal/Ora/Generated/DispatcherTableSnapshot.lean` — data-only
//! dispatcher table facts from actual Ora source compiled through
//! `oraBuildSIRDispatcher`, then classified by Sinora's real switch
//! planner/index functions.

const std = @import("std");
const ora_root = @import("ora_root");
const mlir = @import("mlir_c_api").c;
const sinora = @import("sinora");
const dispatcher_rows = @import("dispatcher_table_rows.zig");

const compiler = ora_root.compiler;
const ir = sinora.ir;
const switch_routing = sinora.switch_routing;

const header =
    \\/-
    \\GENERATED — DATA ONLY. Do NOT edit by hand and do NOT add any `theorem`,
    \\`lemma`, `example`, `axiom`, `sorry`, `instance`, `macro`, attribute, or extra
    \\`import` to this file. It contains only `def … := <literal>` dispatcher
    \\table rows emitted from actual Ora source fixtures compiled through
    \\`oraBuildSIRDispatcher`, with route indices computed by Sinora's switch
    \\planner. The TRUSTED checks live in `Ora/DispatcherTableSync.lean`.
    \\
    \\Regenerate with `scripts/check-formal-sync.sh`. Sources:
    \\src/formal/emit_dispatcher_table_snapshot.zig,
    \\src/mlir/ora/lowering/OraToSIR/SIRDispatcher.cpp,
    \\sinora/src/switch_routing.zig.
    \\-/
    \\
    \\namespace Ora.Generated
    \\
    \\
;

const DesiredPlan = enum {
    linear,
    dense_bit_window,
    dense_multiplicative,
    sparse,
};

const FixtureSpec = struct {
    name: []const u8,
    contract_name: []const u8,
    prefix: []const u8,
    desired: DesiredPlan,
    min_cases: usize,
    max_cases: usize,
    policy: switch_routing.DispatchPolicy = .balanced,
};

const fixtures = [_]FixtureSpec{
    .{
        .name = "compiled_linear_small",
        .contract_name = "LinearDispatcher",
        .prefix = "lin",
        .desired = .linear,
        .min_cases = 4,
        .max_cases = 4,
    },
    .{
        .name = "compiled_dense",
        .contract_name = "DenseDispatcher",
        .prefix = "den",
        .desired = .dense_bit_window,
        .min_cases = 12,
        .max_cases = 64,
    },
    .{
        .name = "compiled_dense_multiplicative",
        .contract_name = "MultiplicativeDispatcher",
        .prefix = "d",
        .desired = .dense_multiplicative,
        .min_cases = 16,
        .max_cases = 16,
    },
    .{
        .name = "compiled_sparse",
        .contract_name = "SparseDispatcher",
        .prefix = "spr",
        .desired = .sparse,
        .min_cases = 48,
        .max_cases = 180,
    },
};

const ExtractedSwitch = dispatcher_rows.ExtractedSwitch;

fn planMatches(plan: switch_routing.Plan, desired: DesiredPlan) bool {
    return switch (desired) {
        .linear => plan == .linear,
        .dense_bit_window => switch (plan) {
            .dense => |dense| dense.kind == .bit_window,
            else => false,
        },
        .dense_multiplicative => switch (plan) {
            .dense => |dense| dense.kind == .multiplicative,
            else => false,
        },
        .sparse => plan == .sparse,
    };
}

fn makeCandidateCases(
    allocator: std.mem.Allocator,
    prefix: []const u8,
    count: usize,
) ![]ir.SwitchCase {
    const cases = try allocator.alloc(ir.SwitchCase, count);
    errdefer allocator.free(cases);
    for (cases, 0..) |*case, i| {
        const name = try std.fmt.allocPrint(allocator, "{s}{d}", .{ prefix, i });
        errdefer allocator.free(name);
        const selector = try dispatcher_rows.selectorForName(allocator, name);
        const value = try dispatcher_rows.selectorCaseValue(allocator, selector);
        errdefer allocator.free(value);
        case.* = .{
            .value = value,
            .target = name,
            .line = @intCast(i + 1),
        };
    }
    return cases;
}

fn freeCandidateCases(allocator: std.mem.Allocator, cases: []ir.SwitchCase) void {
    for (cases) |case| {
        allocator.free(case.value);
        allocator.free(case.target);
    }
    allocator.free(cases);
}

fn findCandidateCases(allocator: std.mem.Allocator, spec: FixtureSpec) ![]ir.SwitchCase {
    var count = spec.min_cases;
    while (count <= spec.max_cases) : (count += 1) {
        const cases = try makeCandidateCases(allocator, spec.prefix, count);
        const switch_term: ir.SwitchTerminator = .{
            .selector = "selector",
            .cases = cases,
            .default_target = "fallback",
        };
        const plan = switch_routing.choosePlan(switch_term);
        if (planMatches(plan, spec.desired)) return cases;
        freeCandidateCases(allocator, cases);
    }
    return error.NoDispatcherFixtureForPlan;
}

fn buildSourceFromCases(
    allocator: std.mem.Allocator,
    contract_name: []const u8,
    cases: []const ir.SwitchCase,
) ![]u8 {
    var source: std.Io.Writer.Allocating = .init(allocator);
    errdefer source.deinit();
    try source.writer.print("contract {s} {{\n", .{contract_name});
    for (cases, 0..) |case, i| {
        try source.writer.print(
            \\    pub fn {s}() -> u256 {{
            \\        return {d};
            \\    }}
            \\
        , .{ case.target, i });
    }
    try source.writer.writeAll("}\n");
    return source.toOwnedSlice();
}

fn emitCompiledFixtureRows(
    out: anytype,
    allocator: std.mem.Allocator,
    spec: FixtureSpec,
    first_row: *bool,
) !void {
    const previous_policy = switch_routing.dispatch_policy;
    switch_routing.dispatch_policy = spec.policy;
    defer switch_routing.dispatch_policy = previous_policy;

    const candidate_cases = try findCandidateCases(allocator, spec);
    defer freeCandidateCases(allocator, candidate_cases);

    const source = try buildSourceFromCases(allocator, spec.contract_name, candidate_cases);
    defer allocator.free(source);

    var compilation = try compiler.compileSource(allocator, spec.name, source);
    defer compilation.deinit();

    const hir_result = try compilation.db.lowerToHir(compilation.root_module_id);
    if (!hir_result.diagnostics.isEmpty()) return error.DispatcherFixtureDiagnostics;
    if (!mlir.oraConvertToSIR(hir_result.context, hir_result.module.raw_module, false)) {
        return error.DispatcherFixtureOraToSirFailed;
    }
    if (!mlir.oraBuildSIRDispatcher(hir_result.context, hir_result.module.raw_module)) {
        return error.DispatcherFixtureBuildFailed;
    }

    const facts_ref = mlir.oraExtractSIRDispatcherSwitchFacts(hir_result.context, hir_result.module.raw_module);
    defer if (facts_ref.data != null) mlir.oraStringRefFree(facts_ref);
    if (facts_ref.data == null or facts_ref.length == 0) return error.DispatcherFixtureMissingFacts;
    const facts_json = facts_ref.data[0..facts_ref.length];

    const parsed = try std.json.parseFromSlice(
        dispatcher_rows.ExtractedSwitchManifest,
        allocator,
        facts_json,
        .{ .ignore_unknown_fields = true },
    );
    defer parsed.deinit();
    const switches = parsed.value.switches;
    if (switches.len != 1) return error.DispatcherFixtureExpectedOneSwitch;

    const cases = try dispatcher_rows.extractedToSwitchCases(allocator, switches[0].cases);
    defer dispatcher_rows.freeExtractedSwitchCases(allocator, cases);
    const switch_term: ir.SwitchTerminator = .{
        .selector = "selector",
        .cases = cases,
        .default_target = "fallback",
    };
    if (!planMatches(switch_routing.choosePlan(switch_term), spec.desired)) {
        return error.DispatcherFixturePlanMismatch;
    }

    if (!first_row.*) try out.writeAll(",\n");
    first_row.* = false;
    try dispatcher_rows.emitLeanRow(out, allocator, spec.name, switches[0].cases);
}

fn emitRows(out: anytype, allocator: std.mem.Allocator) !void {
    try out.writeAll(
        \\def compilerDispatcherTableRows :
        \\    List (String × (String × String × Nat × Nat × Nat × Nat) ×
        \\      (String × Bool × Nat ×
        \\        List (Nat × Option Nat × List (Nat × Nat × Nat)) ×
        \\        Nat ×
        \\        Option ((String × String × Nat × Nat × Nat × Nat) × Nat) ×
        \\        Nat ×
        \\        Option ((String × String × Nat × Nat × Nat × Nat) × Nat)) ×
        \\      List (Nat × String × Nat × Bool)) :=
        \\  [
    );

    var first = true;
    for (fixtures) |fixture| {
        try emitCompiledFixtureRows(out, allocator, fixture, &first);
    }
    try out.writeAll("\n  ]\n\n");

    try out.writeAll("def compilerDispatcherMultiplicativeCandidates : List Nat :=\n  [");
    for (0..switch_routing.multiplicative_search_budget) |candidate_index| {
        if (candidate_index != 0) {
            if (candidate_index % 12 == 0) {
                try out.writeAll(",\n   ");
            } else {
                try out.writeAll(", ");
            }
        }
        try out.print("{d}", .{switch_routing.multiplicativeCandidate(@intCast(candidate_index))});
    }
    try out.writeAll("]\n\n");
}

pub fn main(init: std.process.Init) !void {
    const io = init.io;
    var gpa = std.heap.DebugAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    var args = try std.process.Args.Iterator.initAllocator(init.minimal.args, allocator);
    defer args.deinit();
    _ = args.skip();
    const output_path = args.next();
    if (args.next() != null) return error.InvalidArguments;

    var out_buffer: [32768]u8 = undefined;
    if (output_path) |path| {
        var file = try std.Io.Dir.cwd().createFile(io, path, .{});
        defer file.close(io);
        var file_writer = file.writer(io, &out_buffer);
        const out = &file_writer.interface;

        try out.writeAll(header);
        try emitRows(out, allocator);
        try out.writeAll("end Ora.Generated\n");
        try out.flush();
    } else {
        var stdout_writer = std.Io.File.stdout().writer(io, &out_buffer);
        const out = &stdout_writer.interface;

        try out.writeAll(header);
        try emitRows(out, allocator);
        try out.writeAll("end Ora.Generated\n");
        try out.flush();
    }
}
