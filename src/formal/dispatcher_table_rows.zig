//! Shared dispatcher-table row encoding for Lean checks.
//!
//! Both the repository snapshot emitter and the user-build proof gate use this
//! module. The C++ dispatcher builder supplies selector-switch facts; this file
//! computes the Sinora route metadata and emits the data-only Lean row literal.

const std = @import("std");
const sinora = @import("sinora");

const abi = @import("ora_root").compiler.hir.abi;
const ir = sinora.ir;
const switch_routing = sinora.switch_routing;

pub const DispatcherIntentManifest = struct {
    schema_version: u32,
    intents: []const DispatcherIntent,
};

pub const DispatcherIntent = struct {
    selector: u32,
    function: []const u8,
    label: []const u8,
};

pub const ExtractedSwitchManifest = struct {
    schema_version: u32,
    switches: []const ExtractedSwitch,
};

pub const ExtractedSwitch = struct {
    ordinal: u64,
    block: []const u8,
    default_label: []const u8,
    cases: []const ExtractedCase,
};

pub const ExtractedCase = struct {
    selector: u32,
    label: []const u8,
    guarded: bool,
};

pub const ManifestValidationError = error{
    UnsupportedSchema,
    MissingMain,
    MissingSwitchBlock,
    SwitchCountMismatch,
    CaseCountMismatch,
    SelectorMismatch,
    TargetMismatch,
    DefaultMismatch,
    IntentMismatch,
    DuplicateIntent,
    DefaultDoesNotRevert,
};

pub fn validateManifestAgainstSir(
    allocator: std.mem.Allocator,
    intent_manifest: DispatcherIntentManifest,
    switch_manifest: ExtractedSwitchManifest,
    sir_text: []const u8,
) !void {
    if (intent_manifest.schema_version != 1 or switch_manifest.schema_version != 1) {
        return ManifestValidationError.UnsupportedSchema;
    }

    var diagnostics = sinora.DiagnosticBag.init(allocator);
    defer diagnostics.deinit();
    var program = try sinora.parse(allocator, sir_text, &diagnostics);
    defer program.deinit();
    try sinora.validate(allocator, program, &diagnostics);

    const main = findFunction(program, "main") orelse return ManifestValidationError.MissingMain;
    var switch_count: usize = 0;
    for (main.blocks) |block| {
        switch (block.terminator) {
            .switch_ => switch_count += 1,
            else => {},
        }
    }
    if (switch_count != switch_manifest.switches.len) {
        return ManifestValidationError.SwitchCountMismatch;
    }

    for (switch_manifest.switches) |expected| {
        const block = findBlock(main, expected.block) orelse
            return ManifestValidationError.MissingSwitchBlock;
        const actual = switch (block.terminator) {
            .switch_ => |value| value,
            else => return ManifestValidationError.MissingSwitchBlock,
        };
        if (!std.mem.eql(u8, actual.default_target, expected.default_label)) {
            return ManifestValidationError.DefaultMismatch;
        }
        if (actual.cases.len != expected.cases.len) {
            return ManifestValidationError.CaseCountMismatch;
        }
        for (actual.cases, expected.cases) |actual_case, expected_case| {
            const selector = switch_routing.parseU32Selector(actual_case.value) orelse
                return ManifestValidationError.SelectorMismatch;
            if (selector != expected_case.selector) {
                return ManifestValidationError.SelectorMismatch;
            }
            if (!std.mem.eql(u8, actual_case.target, expected_case.label)) {
                return ManifestValidationError.TargetMismatch;
            }
        }
    }

    for (intent_manifest.intents, 0..) |intent, index| {
        for (intent_manifest.intents[0..index]) |previous| {
            if (previous.selector == intent.selector) return ManifestValidationError.DuplicateIntent;
        }
        var occurrences: usize = 0;
        for (switch_manifest.switches) |sw| {
            for (sw.cases) |case| {
                if (case.selector != intent.selector) continue;
                if (!std.mem.eql(u8, case.label, intent.label)) {
                    return ManifestValidationError.IntentMismatch;
                }
                occurrences += 1;
            }
        }
        if (occurrences != 1) return ManifestValidationError.IntentMismatch;
    }
    for (switch_manifest.switches) |sw| {
        for (sw.cases) |case| {
            var authorized = false;
            for (intent_manifest.intents) |intent| {
                if (case.selector == intent.selector and std.mem.eql(u8, case.label, intent.label)) {
                    authorized = true;
                    break;
                }
            }
            if (!authorized) return ManifestValidationError.IntentMismatch;
        }
    }

    for (switch_manifest.switches) |sw| {
        if (findSwitch(switch_manifest.switches, sw.default_label) != null) continue;
        if (!std.mem.eql(u8, sw.default_label, "revert_error")) {
            return ManifestValidationError.DefaultMismatch;
        }
        const fallback = findBlock(main, sw.default_label) orelse
            return ManifestValidationError.DefaultDoesNotRevert;
        switch (fallback.terminator) {
            .revert => {},
            else => return ManifestValidationError.DefaultDoesNotRevert,
        }
    }
}

fn findFunction(program: ir.Program, name: []const u8) ?ir.Function {
    for (program.functions) |function| {
        if (std.mem.eql(u8, function.name, name)) return function;
    }
    return null;
}

fn findBlock(function: ir.Function, name: []const u8) ?ir.Block {
    for (function.blocks) |block| {
        if (std.mem.eql(u8, block.name, name)) return block;
    }
    return null;
}

fn findSwitch(switches: []const ExtractedSwitch, block: []const u8) ?ExtractedSwitch {
    for (switches) |sw| {
        if (std.mem.eql(u8, sw.block, block)) return sw;
    }
    return null;
}

pub fn writeLeanString(out: anytype, text: []const u8) !void {
    try out.writeByte('"');
    for (text) |ch| {
        switch (ch) {
            '\\' => try out.writeAll("\\\\"),
            '"' => try out.writeAll("\\\""),
            '\n' => try out.writeAll("\\n"),
            '\r' => try out.writeAll("\\r"),
            '\t' => try out.writeAll("\\t"),
            else => try out.writeByte(ch),
        }
    }
    try out.writeByte('"');
}

pub fn planStrategyName(plan: switch_routing.Plan) []const u8 {
    return switch (plan) {
        .linear => "linear",
        .sparse => "sparse",
        .dense => "dense",
    };
}

pub fn denseKindName(plan: switch_routing.Plan) []const u8 {
    return switch (plan) {
        .dense => |dense| dense.kind.jsonName(),
        else => "",
    };
}

pub fn tableSlots(plan: switch_routing.Plan, cases_len: usize) usize {
    return switch (plan) {
        .linear => cases_len,
        .sparse => |sparse| sparse.bucket_count,
        .dense => |dense| dense.table_slots,
    };
}

pub fn indexBits(plan: switch_routing.Plan) u8 {
    return switch (plan) {
        .linear => 0,
        .sparse => |sparse| sparse.bucket_bits,
        .dense => |dense| dense.index_bits orelse 0,
    };
}

pub fn indexShift(plan: switch_routing.Plan) u8 {
    return switch (plan) {
        .linear => 0,
        .sparse => |sparse| sparse.bucket_shift,
        .dense => |dense| dense.index_shift orelse 0,
    };
}

pub fn mulConstant(plan: switch_routing.Plan) u32 {
    return switch (plan) {
        .dense => |dense| dense.mul_constant orelse 0,
        else => 0,
    };
}

pub fn routeIndex(plan: switch_routing.Plan, selector: u32, ordinal: usize) usize {
    return switch (plan) {
        .linear => ordinal,
        .sparse => |sparse| switch_routing.bucketIndex(selector, sparse.bucket_bits, sparse.bucket_shift),
        .dense => |dense| switch_routing.denseIndex(selector, dense),
    };
}

fn emitLeanPlan(
    out: anytype,
    plan: switch_routing.Plan,
    cases_len: usize,
) !void {
    try out.writeAll("(");
    try writeLeanString(out, planStrategyName(plan));
    try out.writeAll(", ");
    try writeLeanString(out, denseKindName(plan));
    try out.print(", {d}, {d}, {d}, {d})", .{
        tableSlots(plan, cases_len),
        indexBits(plan),
        indexShift(plan),
        mulConstant(plan),
    });
}

fn emitLeanScoredPlan(
    out: anytype,
    scored: switch_routing.ScoredPlan,
    cases_len: usize,
) !void {
    try out.writeByte('(');
    try emitLeanPlan(out, scored.plan, cases_len);
    try out.print(", {d})", .{scored.score_x1000});
}

fn emitLeanOptionalScoredPlan(
    out: anytype,
    plan: ?switch_routing.ScoredPlan,
    cases_len: usize,
) !void {
    if (plan) |scored| {
        try out.writeAll("some ");
        try emitLeanScoredPlan(out, scored, cases_len);
    } else {
        try out.writeAll("none");
    }
}

fn emitLeanMultiplicativeSearch(
    out: anytype,
    search: switch_routing.MultiplicativeSearchTrace,
) !void {
    try out.print("({d}, ", .{search.table_slots});
    if (search.selected_candidate_index) |candidate_index| {
        try out.print("some {d}", .{candidate_index});
    } else {
        try out.writeAll("none");
    }
    try out.writeAll(", [");
    for (search.rejected, 0..) |witness, i| {
        if (i != 0) try out.writeAll(", ");
        try out.print("({d}, {d}, {d})", .{
            witness.constant,
            witness.first_case,
            witness.second_case,
        });
    }
    try out.writeAll("])");
}

test "dispatcher manifest validator ties source intents to exact SIR routes" {
    const allocator = std.testing.allocator;
    const sir =
        \\fn main:
        \\    load_selector {
        \\        selector = const 1
        \\        switch selector {
        \\        1 => @known
        \\        default => @revert_error
        \\        }
        \\    }
        \\    known {
        \\        stop
        \\    }
        \\    revert_error {
        \\        zero = const 0
        \\        revert zero zero
        \\    }
    ;
    const intents = [_]DispatcherIntent{.{
        .selector = 1,
        .function = "known",
        .label = "known",
    }};
    const cases = [_]ExtractedCase{.{
        .selector = 1,
        .label = "known",
        .guarded = true,
    }};
    const switches = [_]ExtractedSwitch{.{
        .ordinal = 0,
        .block = "load_selector",
        .default_label = "revert_error",
        .cases = &cases,
    }};
    try validateManifestAgainstSir(
        allocator,
        .{ .schema_version = 1, .intents = &intents },
        .{ .schema_version = 1, .switches = &switches },
        sir,
    );

    var wrong_cases = cases;
    wrong_cases[0].label = "revert_error";
    var wrong_switches = switches;
    wrong_switches[0].cases = &wrong_cases;
    try std.testing.expectError(
        ManifestValidationError.TargetMismatch,
        validateManifestAgainstSir(
            allocator,
            .{ .schema_version = 1, .intents = &intents },
            .{ .schema_version = 1, .switches = &wrong_switches },
            sir,
        ),
    );

    const missing_intents = [_]DispatcherIntent{.{
        .selector = 2,
        .function = "missing",
        .label = "missing",
    }};
    try std.testing.expectError(
        ManifestValidationError.IntentMismatch,
        validateManifestAgainstSir(
            allocator,
            .{ .schema_version = 1, .intents = &missing_intents },
            .{ .schema_version = 1, .switches = &switches },
            sir,
        ),
    );
}

test "dispatcher manifest validator rejects a non-reverting terminal default" {
    const allocator = std.testing.allocator;
    const sir =
        \\fn main:
        \\    load_selector {
        \\        selector = const 1
        \\        switch selector {
        \\        1 => @known
        \\        default => @revert_error
        \\        }
        \\    }
        \\    known {
        \\        stop
        \\    }
        \\    revert_error {
        \\        stop
        \\    }
    ;
    const intents = [_]DispatcherIntent{.{
        .selector = 1,
        .function = "known",
        .label = "known",
    }};
    const cases = [_]ExtractedCase{.{
        .selector = 1,
        .label = "known",
        .guarded = true,
    }};
    const switches = [_]ExtractedSwitch{.{
        .ordinal = 0,
        .block = "load_selector",
        .default_label = "revert_error",
        .cases = &cases,
    }};
    try std.testing.expectError(
        ManifestValidationError.DefaultDoesNotRevert,
        validateManifestAgainstSir(
            allocator,
            .{ .schema_version = 1, .intents = &intents },
            .{ .schema_version = 1, .switches = &switches },
            sir,
        ),
    );
}

fn emitLeanMultiplicativeSearches(
    out: anytype,
    searches: []const switch_routing.MultiplicativeSearchTrace,
) !void {
    try out.writeByte('[');
    for (searches, 0..) |search, i| {
        if (i != 0) try out.writeAll(", ");
        try emitLeanMultiplicativeSearch(out, search);
    }
    try out.writeByte(']');
}

pub fn selectorForName(allocator: std.mem.Allocator, name: []const u8) !u32 {
    const signature = try std.fmt.allocPrint(allocator, "{s}()", .{name});
    defer allocator.free(signature);
    return abi.keccakSelectorValue(signature);
}

pub fn selectorCaseValue(allocator: std.mem.Allocator, selector: u32) ![]u8 {
    return std.fmt.allocPrint(allocator, "0x{x:0>8}", .{selector});
}

pub fn extractedToSwitchCases(
    allocator: std.mem.Allocator,
    extracted: []const ExtractedCase,
) ![]ir.SwitchCase {
    const cases = try allocator.alloc(ir.SwitchCase, extracted.len);
    errdefer allocator.free(cases);
    for (extracted, cases, 0..) |case, *out, i| {
        const value = try selectorCaseValue(allocator, case.selector);
        errdefer allocator.free(value);
        out.* = .{
            .value = value,
            .target = case.label,
            .line = @intCast(i + 1),
        };
    }
    return cases;
}

pub fn freeExtractedSwitchCases(allocator: std.mem.Allocator, cases: []ir.SwitchCase) void {
    for (cases) |case| allocator.free(case.value);
    allocator.free(cases);
}

pub fn emitLeanRow(
    out: anytype,
    allocator: std.mem.Allocator,
    name: []const u8,
    extracted: []const ExtractedCase,
) !void {
    const cases = try extractedToSwitchCases(allocator, extracted);
    defer freeExtractedSwitchCases(allocator, cases);

    const switch_term: ir.SwitchTerminator = .{
        .selector = "selector",
        .cases = cases,
        .default_target = "fallback",
    };
    var trace = try switch_routing.detailedPlanTrace(allocator, switch_term);
    defer trace.deinit(allocator);
    const plan = trace.chosen;

    try out.writeAll("  (");
    try writeLeanString(out, name);
    try out.writeAll(", ");
    try emitLeanPlan(out, plan, cases.len);
    try out.writeAll(", (");
    try writeLeanString(out, trace.policy.jsonName());
    try out.print(", {s}, {d}, ", .{
        if (trace.preconditions_met) "true" else "false",
        trace.linear_score_x1000,
    });
    try emitLeanMultiplicativeSearches(out, trace.multiplicative_searches);
    try out.print(", {d}, ", .{trace.dense_candidates.len});
    try emitLeanOptionalScoredPlan(out, trace.best_dense, cases.len);
    try out.print(", {d}, ", .{trace.sparse_candidates.len});
    try emitLeanOptionalScoredPlan(out, trace.best_sparse, cases.len);
    try out.writeAll("),\n    [");

    for (extracted, 0..) |case, i| {
        if (i != 0) try out.writeAll(",\n     ");
        try out.print("({d}, ", .{case.selector});
        try writeLeanString(out, case.label);
        try out.print(", {d}, {s})", .{
            routeIndex(plan, case.selector, i),
            if (case.guarded) "true" else "false",
        });
    }
    try out.writeAll("])");
}
