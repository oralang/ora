//! Per-contract dispatcher table Lean gate.
//!
//! `--lean-proofs` is a proof gate for the compiled artifact, not only for
//! user-supplied proof rows. After SIR dispatcher construction this module
//! extracts the current contract's selector-switch facts, emits a scratch Lean
//! checker over those rows, and requires Lean to prove the same dispatcher table
//! predicates used by the repository sync snapshot.

const std = @import("std");
const mlir = @import("mlir_c_api").c;
const sinora = @import("sinora");
const dispatcher_rows = @import("dispatcher_table_rows.zig");

const Sha256 = std.crypto.hash.sha2.Sha256;
const proof_scratch_root = "/tmp/ora-dispatcher-proof";

const ProofClaim = struct {
    theorem_name: []const u8,
    summary: []const u8,
};

const proof_claims = [_]ProofClaim{
    .{
        .theorem_name = "current_dispatcher_network_matches",
        .summary = "the dispatcher network implements the intended selector topology",
    },
    .{
        .theorem_name = "current_dispatcher_unknown_selectors_revert",
        .summary = "every selector absent from the public intents reaches revert_error",
    },
    .{
        .theorem_name = "current_dispatcher_planner_reference_matches",
        .summary = "each emitted plan equals the universally admissible Lean reference plan",
    },
    .{
        .theorem_name = "current_dispatcher_manifest_rows_match",
        .summary = "row shape, coverage, collision, route-index, and planner-evidence checks hold",
    },
    .{
        .theorem_name = "current_dispatcher_builder_correct",
        .summary = "each emitted plan builds a strategy-well-formed dispatcher",
    },
};

pub const CheckResult = struct {
    arena: std.heap.ArenaAllocator,
    certificate_json: []const u8,
    switch_manifest_json: []const u8,
    switch_count: usize,
    case_count: usize,

    pub fn deinit(self: *CheckResult) void {
        self.arena.deinit();
        self.* = undefined;
    }

    pub fn writeVerificationSummary(
        self: *const CheckResult,
        writer: anytype,
        bytecode_bound: bool,
    ) !void {
        try writer.print(
            "Lean dispatcher verification summary:\n  proof surface: dispatcher_userland\n  input: {d} switch(es), {d} case(s)\n  kernel-checked conclusions:\n",
            .{ self.switch_count, self.case_count },
        );
        for (proof_claims) |claim| {
            try writer.print("    - {s}: {s}\n", .{ claim.theorem_name, claim.summary });
        }
        if (bytecode_bound) {
            try writer.writeAll(
                "  bytecode binding (compiler-side checks, not an additional Lean theorem):\n" ++
                    "    - the backend report matches the proven SIR switches, strategies, labels, and route indices\n" ++
                    "    - the backend templates_valid attestation is present\n" ++
                    "    - the certificate binds the backend report and emitted bytecode by SHA-256\n",
            );
        }
    }

    pub fn validateAndBindBytecode(
        self: *CheckResult,
        allocator: std.mem.Allocator,
        report_json: []const u8,
        bytecode_hex: []const u8,
    ) !void {
        const report_parsed = try std.json.parseFromSlice(
            BackendDispatcherReport,
            allocator,
            report_json,
            .{ .ignore_unknown_fields = true },
        );
        defer report_parsed.deinit();
        const manifest_parsed = try std.json.parseFromSlice(
            dispatcher_rows.ExtractedSwitchManifest,
            allocator,
            self.switch_manifest_json,
            .{ .ignore_unknown_fields = true },
        );
        defer manifest_parsed.deinit();
        try validateBackendReport(
            allocator,
            manifest_parsed.value.switches,
            report_parsed.value,
        );

        const bytecode_sha = try sha256HexBytesFromHex(allocator, bytecode_hex);
        defer allocator.free(bytecode_sha);
        if (!std.mem.eql(u8, bytecode_sha, report_parsed.value.bytecode_sha256)) {
            return error.DispatcherBytecodeHashMismatch;
        }
        const report_sha = try sha256HexAlloc(allocator, report_json);
        defer allocator.free(report_sha);

        const closing = std.mem.lastIndexOf(u8, self.certificate_json, "\n}") orelse
            return error.InvalidDispatcherCertificate;
        self.certificate_json = try std.fmt.allocPrint(
            self.arena.allocator(),
            "{s},\n  \"bytecode_sha256\": \"{s}\",\n  \"dispatcher_bytecode_report_sha256\": \"{s}\",\n  \"bytecode_templates_valid\": true\n}}\n",
            .{ self.certificate_json[0..closing], bytecode_sha, report_sha },
        );
    }
};

const BackendDispatcherReport = struct {
    schema_version: u32,
    bytecode_sha256: []const u8,
    switches: []const BackendDispatcherSwitch,
};

const BackendDispatcherSwitch = struct {
    block: []const u8,
    strategy: []const u8,
    default_label: []const u8,
    templates_valid: bool,
    cases: []const BackendDispatcherCase,
};

const BackendDispatcherCase = struct {
    selector: u32,
    target: []const u8,
    route_index: usize,
};

pub fn checkCurrentModule(
    allocator: std.mem.Allocator,
    ctx: mlir.MlirContext,
    module: mlir.MlirModule,
    intent_json: []const u8,
    sir_text: []const u8,
    file_path: []const u8,
    process_environ: std.process.Environ,
    stdout: anytype,
) !CheckResult {
    const facts_ref = mlir.oraExtractSIRDispatcherSwitchFacts(ctx, module);
    defer if (facts_ref.data != null) mlir.oraStringRefFree(facts_ref);
    if (facts_ref.data == null) return error.DispatcherTableFactsUnavailable;

    const facts_json = facts_ref.data[0..facts_ref.length];
    const parsed = try std.json.parseFromSlice(
        dispatcher_rows.ExtractedSwitchManifest,
        allocator,
        facts_json,
        .{ .ignore_unknown_fields = true },
    );
    defer parsed.deinit();
    const intent_parsed = try std.json.parseFromSlice(
        dispatcher_rows.DispatcherIntentManifest,
        allocator,
        intent_json,
        .{ .ignore_unknown_fields = true },
    );
    defer intent_parsed.deinit();
    try dispatcher_rows.validateManifestAgainstSir(
        allocator,
        intent_parsed.value,
        parsed.value,
        sir_text,
    );
    const switches = parsed.value.switches;
    const case_count = countCases(switches);

    const scratch_segment = try scratchSegment(allocator, file_path);
    defer allocator.free(scratch_segment);

    const io = std.Io.Threaded.global_single_threaded.io();
    const scratch_path = try std.fmt.allocPrint(
        allocator,
        "{s}/{s}",
        .{ proof_scratch_root, scratch_segment },
    );
    defer allocator.free(scratch_path);
    try std.Io.Dir.cwd().createDirPath(io, scratch_path);
    defer std.Io.Dir.cwd().deleteTree(io, scratch_path) catch {};

    const checker_path = try std.fmt.allocPrint(allocator, "{s}/DispatcherTable.lean", .{scratch_path});
    defer allocator.free(checker_path);

    const checker_source = try buildCheckerSource(
        allocator,
        intent_parsed.value.intents,
        switches,
    );
    defer allocator.free(checker_source);
    try std.Io.Dir.cwd().writeFile(io, .{ .sub_path = checker_path, .data = checker_source });

    const lean_output = try runLeanCommandCapture(
        allocator,
        &.{ "lake", "env", "lean", checker_path },
        process_environ,
        stdout,
    );
    defer allocator.free(lean_output.stdout);
    defer allocator.free(lean_output.stderr);

    var arena = std.heap.ArenaAllocator.init(allocator);
    errdefer arena.deinit();
    const arena_allocator = arena.allocator();
    const certificate_json = try buildCertificateJson(
        arena_allocator,
        checker_source,
        switches.len,
        case_count,
        process_environ,
    );
    const switch_manifest_json = try arena_allocator.dupe(u8, facts_json);

    return .{
        .arena = arena,
        .certificate_json = certificate_json,
        .switch_manifest_json = switch_manifest_json,
        .switch_count = switches.len,
        .case_count = case_count,
    };
}

fn buildCheckerSource(
    allocator: std.mem.Allocator,
    intents: []const dispatcher_rows.DispatcherIntent,
    switches: []const dispatcher_rows.ExtractedSwitch,
) ![]const u8 {
    var out = std.Io.Writer.Allocating.init(allocator);
    defer out.deinit();
    const writer = &out.writer;

    try writer.writeAll(
        \\-- Generated per-contract dispatcher proof. This file is emitted
        \\-- by `ora --lean-proofs` from the current SIR dispatcher and is never
        \\-- user input.
        \\
        \\import Ora.DispatcherTableSync
        \\
        \\namespace Ora.DispatcherRuntimeCheck
        \\
        \\def currentDispatcherIntents :
        \\    List Ora.DispatcherTableSync.RawIntent :=
        \\  [
    );
    for (intents, 0..) |intent, index| {
        if (index != 0) try writer.writeAll(",\n");
        try writer.print("    ({d}, ", .{intent.selector});
        try dispatcher_rows.writeLeanString(writer, intent.label);
        try writer.writeByte(')');
    }
    try writer.writeAll(
        \\]
        \\
        \\def currentDispatcherNetwork :
        \\    List Ora.DispatcherTableSync.RawNetworkSwitch :=
        \\  [
    );
    for (switches, 0..) |sw, switch_index| {
        if (switch_index != 0) try writer.writeAll(",\n");
        try writer.writeAll("    (");
        try dispatcher_rows.writeLeanString(writer, sw.block);
        try writer.writeAll(", ");
        try dispatcher_rows.writeLeanString(writer, sw.default_label);
        try writer.writeAll(", [");
        for (sw.cases, 0..) |case, case_index| {
            if (case_index != 0) try writer.writeAll(", ");
            try writer.print("({d}, ", .{case.selector});
            try dispatcher_rows.writeLeanString(writer, case.label);
            try writer.writeByte(')');
        }
        try writer.writeAll("])");
    }
    try writer.writeAll(
        \\]
        \\
        \\def currentDispatcherEntry : String :=
        \\
    );
    if (switches.len == 0) {
        try dispatcher_rows.writeLeanString(writer, "");
    } else {
        try dispatcher_rows.writeLeanString(writer, switches[0].block);
    }
    try writer.writeAll(
        \\
        \\def currentDispatcherNetworkMatches : Bool :=
        \\  Ora.DispatcherTableSync.Gate.networkMatches
        \\    currentDispatcherIntents currentDispatcherNetwork currentDispatcherEntry
        \\
        \\def currentDispatcherTableRows :
        \\    List Ora.DispatcherTableSync.RawRow :=
        \\  [
    );
    try writer.writeByte('\n');

    var first = true;
    for (switches) |sw| {
        if (!first) try writer.writeAll(",\n");
        first = false;
        var name_buf: [64]u8 = undefined;
        const name = try std.fmt.bufPrint(&name_buf, "user_dispatcher_switch_{d}", .{sw.ordinal});
        try dispatcher_rows.emitLeanRow(writer, allocator, name, sw.cases, .named);
    }

    try writer.writeAll(
        \\
        \\  ]
        \\
        \\theorem current_dispatcher_network_matches :
        \\    currentDispatcherNetworkMatches = true := by decide
        \\
        \\theorem current_dispatcher_unknown_selectors_revert
        \\    (selector : Nat)
        \\    (hunknown : ∀ intent, intent ∈ currentDispatcherIntents →
        \\      intent.1 ≠ selector) :
        \\    Ora.DispatcherTableSync.dispatchNetwork
        \\      (currentDispatcherNetwork.length + 1)
        \\      currentDispatcherNetwork currentDispatcherEntry selector = "revert_error" := by
    );
    try writer.writeByte('\n');
    for (intents, 0..) |intent, index| {
        if (index == 0) {
            try writer.print("  by_cases h{d} : selector = {d}\n", .{ index, intent.selector });
        } else {
            for (0..index * 2) |_| try writer.writeByte(' ');
            try writer.print("· by_cases h{d} : selector = {d}\n", .{ index, intent.selector });
        }
        for (0..(index + 1) * 2) |_| try writer.writeByte(' ');
        try writer.writeAll("· exact False.elim ((hunknown (");
        try writer.print("{d}, ", .{intent.selector});
        try dispatcher_rows.writeLeanString(writer, intent.label);
        try writer.print(") (by decide)) h{d}.symm)\n", .{index});
    }
    const final_indent = if (intents.len == 0) 2 else intents.len * 2;
    for (0..final_indent) |_| try writer.writeByte(' ');
    try writer.writeAll("· simp [currentDispatcherNetwork, currentDispatcherEntry, Ora.DispatcherTableSync.dispatchNetwork, Ora.DispatcherTableSync.networkSwitchBlock, Ora.DispatcherTableSync.networkSwitchDefault, Ora.DispatcherTableSync.networkSwitchCases, List.find?");
    for (intents, 0..) |_, index| {
        try writer.print(", beq_false_of_ne (Ne.symm h{d})", .{index});
    }
    try writer.writeAll(
        \\]
        \\
        \\def currentDispatcherRowsHaveNamedDefault : Bool :=
        \\  Ora.DispatcherTableSync.Gate.rowsHaveNamedDefault currentDispatcherTableRows
        \\
        \\def currentDispatcherRowsCovered : Bool :=
        \\  Ora.DispatcherTableSync.Gate.rowsCovered currentDispatcherTableRows
        \\
        \\def currentDenseRowsInjective : Bool :=
        \\  Ora.DispatcherTableSync.Gate.denseRowsInjective currentDispatcherTableRows
        \\
        \\def currentDispatcherTableRowsMatchModel : Bool :=
        \\  Ora.DispatcherTableSync.Gate.rowsMatchModel currentDispatcherTableRows
        \\
        \\def currentDispatcherPlanShapesValid : Bool :=
        \\  Ora.DispatcherTableSync.Gate.planShapesValid currentDispatcherTableRows
        \\
        \\def currentDispatcherPlannerMatches : Bool :=
        \\  Ora.DispatcherTableSync.Gate.plannerMatches currentDispatcherTableRows
        \\
        \\def currentDispatcherPlannerSearchesValid : Bool :=
        \\  Ora.DispatcherTableSync.Gate.plannerSearchesValid currentDispatcherTableRows
        \\
        \\def currentDispatcherPlannerCoreMatches : Bool :=
        \\  Ora.DispatcherTableSync.Gate.plannerCoreMatches currentDispatcherTableRows
        \\
        \\def currentDispatcherPlannerReferenceMatches : Bool :=
        \\  Ora.DispatcherTableSync.Gate.plannerReferenceMatches currentDispatcherTableRows
        \\
        \\def currentDispatcherPlanIndicesMatch : Bool :=
        \\  Ora.DispatcherTableSync.Gate.planIndicesMatch currentDispatcherTableRows
        \\
        \\def currentDispatcherManifestRowsMatch : Bool :=
        \\  Ora.DispatcherTableSync.Gate.manifestRowsMatch currentDispatcherTableRows
        \\
        \\def currentDispatcherManifestBaseRowsMatch : Bool :=
        \\  Ora.DispatcherTableSync.Gate.manifestBaseRowsMatch currentDispatcherTableRows
        \\
        \\theorem current_dispatcher_table_rows_have_named_default :
        \\    currentDispatcherRowsHaveNamedDefault = true := by decide
        \\
        \\theorem current_dispatcher_table_rows_covered :
        \\    currentDispatcherRowsCovered = true := by decide
        \\
        \\theorem current_dense_rows_injective :
        \\    currentDenseRowsInjective = true := by decide
        \\
        \\theorem current_dispatcher_table_rows_match_model :
        \\    currentDispatcherTableRowsMatchModel = true := by decide
        \\
        \\theorem current_dispatcher_plan_shapes_valid :
        \\    currentDispatcherPlanShapesValid = true := by decide
        \\
        \\set_option maxRecDepth 1000000 in
        \\set_option maxHeartbeats 2000000 in
        \\theorem current_dispatcher_planner_searches_valid :
        \\    currentDispatcherPlannerSearchesValid = true := by decide
        \\
        \\set_option maxRecDepth 1000000 in
        \\set_option maxHeartbeats 2000000 in
        \\theorem current_dispatcher_planner_core_matches :
        \\    currentDispatcherPlannerCoreMatches = true := by decide
        \\
        \\set_option maxRecDepth 1000000 in
        \\set_option maxHeartbeats 2000000 in
        \\theorem current_dispatcher_planner_reference_matches :
        \\    currentDispatcherPlannerReferenceMatches = true := by decide
        \\
        \\theorem current_dispatcher_planner_matches :
        \\    currentDispatcherPlannerMatches = true := by
        \\  exact Ora.DispatcherTableSync.Gate.plannerMatchesOfParts
        \\    currentDispatcherTableRows
        \\    current_dispatcher_planner_searches_valid
        \\    current_dispatcher_planner_core_matches
        \\
        \\theorem current_dispatcher_plan_indices_match :
        \\    currentDispatcherPlanIndicesMatch = true := by decide
        \\
        \\theorem current_dispatcher_manifest_base_rows_match :
        \\    currentDispatcherManifestBaseRowsMatch = true := by decide
        \\
        \\theorem current_dispatcher_manifest_rows_match :
        \\    currentDispatcherManifestRowsMatch = true := by
        \\  exact Ora.DispatcherTableSync.Gate.manifestRowsMatchOfParts
        \\    currentDispatcherTableRows
        \\    current_dispatcher_manifest_base_rows_match
        \\    current_dispatcher_planner_matches
        \\
        \\set_option maxRecDepth 1000000 in
        \\set_option maxHeartbeats 2000000 in
        \\theorem current_dispatcher_builder_correct :
        \\    ∀ row, row ∈ currentDispatcherTableRows → Ora.DispatcherTableSync.Gate.rowStrategyWF row :=
        \\  Ora.DispatcherTableSync.Gate.builderCorrect
        \\    currentDispatcherTableRows current_dispatcher_planner_reference_matches
        \\
        \\end Ora.DispatcherRuntimeCheck
        \\
    );

    return try allocator.dupe(u8, out.written());
}

fn countCases(switches: []const dispatcher_rows.ExtractedSwitch) usize {
    var total: usize = 0;
    for (switches) |sw| total += sw.cases.len;
    return total;
}

fn validateBackendReport(
    allocator: std.mem.Allocator,
    switches: []const dispatcher_rows.ExtractedSwitch,
    report: BackendDispatcherReport,
) !void {
    if (report.schema_version != 1) return error.UnsupportedDispatcherBytecodeReport;
    if (switches.len != report.switches.len) return error.DispatcherBytecodeSwitchMismatch;
    for (switches, report.switches) |expected, actual| {
        if (!actual.templates_valid) return error.DispatcherBytecodeTemplateMismatch;
        if (!std.mem.eql(u8, expected.block, actual.block) or
            !std.mem.eql(u8, expected.default_label, actual.default_label) or
            expected.cases.len != actual.cases.len)
        {
            return error.DispatcherBytecodeSwitchMismatch;
        }
        const switch_cases = try dispatcher_rows.extractedToSwitchCases(allocator, expected.cases);
        defer dispatcher_rows.freeExtractedSwitchCases(allocator, switch_cases);
        const term: sinora.ir.SwitchTerminator = .{
            .selector = "selector",
            .cases = switch_cases,
            .default_target = expected.default_label,
        };
        const plan = sinora.switch_routing.choosePlan(term);
        if (!std.mem.eql(u8, dispatcher_rows.planStrategyName(plan), actual.strategy)) {
            return error.DispatcherBytecodePlanMismatch;
        }
        for (expected.cases, 0..) |expected_case, index| {
            const actual_case = findBackendCase(actual.cases, expected_case.selector) orelse
                return error.DispatcherBytecodeCaseMismatch;
            if (!std.mem.eql(u8, expected_case.label, actual_case.target) or
                dispatcher_rows.routeIndex(plan, expected_case.selector, index) != actual_case.route_index)
            {
                return error.DispatcherBytecodeCaseMismatch;
            }
        }
    }
}

fn findBackendCase(cases: []const BackendDispatcherCase, selector: u32) ?BackendDispatcherCase {
    for (cases) |case| {
        if (case.selector == selector) return case;
    }
    return null;
}

fn sha256HexBytesFromHex(allocator: std.mem.Allocator, hex_text: []const u8) ![]const u8 {
    const payload = if (std.mem.startsWith(u8, hex_text, "0x")) hex_text[2..] else hex_text;
    if (payload.len % 2 != 0) return error.InvalidBytecodeHex;
    const bytes = try allocator.alloc(u8, payload.len / 2);
    defer allocator.free(bytes);
    for (bytes, 0..) |*byte, index| {
        byte.* = (try hexNibble(payload[index * 2])) << 4 |
            try hexNibble(payload[index * 2 + 1]);
    }
    return sha256HexAlloc(allocator, bytes);
}

fn hexNibble(byte: u8) !u8 {
    return switch (byte) {
        '0'...'9' => byte - '0',
        'a'...'f' => byte - 'a' + 10,
        'A'...'F' => byte - 'A' + 10,
        else => error.InvalidBytecodeHex,
    };
}

fn buildCertificateJson(
    allocator: std.mem.Allocator,
    checker_source: []const u8,
    switch_count: usize,
    case_count: usize,
    process_environ: std.process.Environ,
) ![]const u8 {
    const lean_version = try queryLeanVersion(allocator, process_environ);
    defer allocator.free(lean_version);
    const checker_sha256 = try sha256HexAlloc(allocator, checker_source);
    defer allocator.free(checker_sha256);

    var out = std.Io.Writer.Allocating.init(allocator);
    defer out.deinit();
    const writer = &out.writer;

    try writer.writeAll("{\n  \"schema_version\": 1,\n  \"status\": \"checked\",\n  \"checker\": \"lean\",\n  \"proof_surface\": \"dispatcher_userland\",\n  \"dispatcher_manifest_schema\": \"dispatcher_v1\",\n  \"legacy_proof_surface\": \"dispatcher_table\",\n  \"hash_algorithm\": \"sha256\",\n  \"lean_version\": ");
    try writeJsonString(writer, lean_version);
    try writer.writeAll(",\n  \"dispatcher_table_check_sha256\": ");
    try writeJsonString(writer, checker_sha256);
    try writer.print(",\n  \"switch_count\": {d},\n  \"case_count\": {d}", .{ switch_count, case_count });
    try writer.writeAll(",\n  \"theorems\": [");
    for (proof_claims, 0..) |claim, index| {
        if (index != 0) try writer.writeByte(',');
        try writer.writeAll("\n    ");
        try writeJsonString(writer, claim.theorem_name);
    }
    try writer.writeAll("\n  ]\n}\n");
    return try allocator.dupe(u8, out.written());
}

fn queryLeanVersion(allocator: std.mem.Allocator, process_environ: std.process.Environ) ![]const u8 {
    var process_io = std.Io.Threaded.init(allocator, .{
        .async_limit = .nothing,
        .concurrent_limit = .nothing,
        .environ = process_environ,
    });
    defer process_io.deinit();

    const argv = [_][]const u8{ "/usr/bin/env", "lake", "env", "lean", "--version" };
    const result = try std.process.run(allocator, process_io.io(), .{
        .argv = &argv,
        .cwd = .{ .path = "formal" },
        .stdout_limit = std.Io.Limit.limited(16 * 1024),
        .stderr_limit = std.Io.Limit.limited(16 * 1024),
    });
    defer allocator.free(result.stdout);
    defer allocator.free(result.stderr);

    switch (result.term) {
        .exited => |code| if (code == 0) {
            const version = std.mem.trim(u8, result.stdout, " \t\r\n");
            if (version.len != 0) return try allocator.dupe(u8, version);
            const stderr = std.mem.trim(u8, result.stderr, " \t\r\n");
            if (stderr.len != 0) return try allocator.dupe(u8, stderr);
        },
        else => {},
    }
    return error.LeanVersionCheckFailed;
}

fn sha256HexAlloc(allocator: std.mem.Allocator, bytes: []const u8) ![]const u8 {
    var digest: [Sha256.digest_length]u8 = undefined;
    Sha256.hash(bytes, &digest, .{});
    const digest_hex = std.fmt.bytesToHex(digest, .lower);
    return try allocator.dupe(u8, &digest_hex);
}

fn writeJsonString(writer: anytype, value: []const u8) !void {
    const hex = "0123456789abcdef";
    try writer.writeByte('"');
    for (value) |byte| {
        switch (byte) {
            '"' => try writer.writeAll("\\\""),
            '\\' => try writer.writeAll("\\\\"),
            '\n' => try writer.writeAll("\\n"),
            '\r' => try writer.writeAll("\\r"),
            '\t' => try writer.writeAll("\\t"),
            else => if (byte < 0x20) {
                try writer.writeAll("\\u00");
                try writer.writeByte(hex[byte >> 4]);
                try writer.writeByte(hex[byte & 0x0f]);
            } else {
                try writer.writeByte(byte);
            },
        }
    }
    try writer.writeByte('"');
}

fn scratchSegment(allocator: std.mem.Allocator, file_path: []const u8) ![]const u8 {
    const stem = std.fs.path.stem(file_path);
    var component = std.Io.Writer.Allocating.init(allocator);
    defer component.deinit();
    try component.writer.writeAll("Dispatcher_");
    for (stem) |byte| {
        if (std.ascii.isAlphanumeric(byte) or byte == '_') {
            try component.writer.writeByte(byte);
        } else {
            try component.writer.writeByte('_');
        }
    }
    const hash = std.hash.Wyhash.hash(0, file_path);
    const pid = std.posix.system.getpid();
    return try std.fmt.allocPrint(
        allocator,
        "{s}_{x}_{d}",
        .{ component.written(), hash, pid },
    );
}

const LeanCommandOutput = struct {
    stdout: []u8,
    stderr: []u8,
};

fn runLeanCommandCapture(
    allocator: std.mem.Allocator,
    argv: []const []const u8,
    process_environ: std.process.Environ,
    stdout: anytype,
) !LeanCommandOutput {
    var process_io = std.Io.Threaded.init(allocator, .{
        .async_limit = .nothing,
        .concurrent_limit = .nothing,
        .environ = process_environ,
    });
    defer process_io.deinit();

    var env_argv: [8][]const u8 = undefined;
    if (argv.len + 1 > env_argv.len) return error.LeanCommandTooLong;
    env_argv[0] = "/usr/bin/env";
    @memcpy(env_argv[1 .. argv.len + 1], argv);

    const result = try std.process.run(allocator, process_io.io(), .{
        .argv = env_argv[0 .. argv.len + 1],
        .cwd = .{ .path = "formal" },
        .stdout_limit = std.Io.Limit.limited(1024 * 1024),
        .stderr_limit = std.Io.Limit.limited(1024 * 1024),
    });

    switch (result.term) {
        .exited => |code| if (code == 0) {
            const lean_stdout = std.mem.trim(u8, result.stdout, " \t\r\n");
            const lean_stderr = std.mem.trim(u8, result.stderr, " \t\r\n");
            try stdout.writeAll("Lean checker output:\n  exit status: 0\n");
            if (lean_stdout.len == 0 and lean_stderr.len == 0) {
                try stdout.writeAll("  diagnostics: none\n");
            } else {
                if (lean_stdout.len != 0) try stdout.print("  stdout:\n{s}\n", .{lean_stdout});
                if (lean_stderr.len != 0) try stdout.print("  stderr:\n{s}\n", .{lean_stderr});
            }
            return .{ .stdout = result.stdout, .stderr = result.stderr };
        },
        else => {},
    }
    try stdout.writeAll("Lean dispatcher userland proof rejected.\n");
    if (result.stdout.len != 0) try stdout.print("{s}\n", .{result.stdout});
    if (result.stderr.len != 0) try stdout.print("{s}\n", .{result.stderr});
    try stdout.flush();
    allocator.free(result.stdout);
    allocator.free(result.stderr);
    return error.LeanDispatcherTableProofFailed;
}

test "dispatcher backend report rejects template and route disagreement" {
    const expected_cases = [_]dispatcher_rows.ExtractedCase{.{
        .selector = 1,
        .label = "known",
        .guarded = true,
    }};
    const expected_switches = [_]dispatcher_rows.ExtractedSwitch{.{
        .ordinal = 0,
        .block = "entry",
        .default_label = "revert_error",
        .cases = &expected_cases,
    }};
    var actual_cases = [_]BackendDispatcherCase{.{
        .selector = 1,
        .target = "known",
        .route_index = 0,
    }};
    var actual_switches = [_]BackendDispatcherSwitch{.{
        .block = "entry",
        .strategy = "linear",
        .default_label = "revert_error",
        .templates_valid = true,
        .cases = &actual_cases,
    }};
    const report: BackendDispatcherReport = .{
        .schema_version = 1,
        .bytecode_sha256 = "",
        .switches = &actual_switches,
    };
    try validateBackendReport(std.testing.allocator, &expected_switches, report);

    actual_switches[0].templates_valid = false;
    try std.testing.expectError(
        error.DispatcherBytecodeTemplateMismatch,
        validateBackendReport(std.testing.allocator, &expected_switches, report),
    );
    actual_switches[0].templates_valid = true;
    actual_cases[0].target = "revert_error";
    try std.testing.expectError(
        error.DispatcherBytecodeCaseMismatch,
        validateBackendReport(std.testing.allocator, &expected_switches, report),
    );
}

test "dispatcher proof scratch root stays outside the audited formal tree" {
    try std.testing.expect(std.fs.path.isAbsolute(proof_scratch_root));
    try std.testing.expect(!std.mem.containsAtLeast(u8, proof_scratch_root, 1, "formal"));
}

test "dispatcher certificate refuses a bytecode hash not bound by the backend report" {
    var check: CheckResult = .{
        .arena = std.heap.ArenaAllocator.init(std.testing.allocator),
        .certificate_json = "{\n  \"schema_version\": 1\n}\n",
        .switch_manifest_json =
        \\{"schema_version":1,"switches":[{"ordinal":0,"block":"entry","default_label":"revert_error","cases":[{"selector":1,"label":"known","guarded":true}]}]}
        ,
        .switch_count = 1,
        .case_count = 1,
    };
    defer check.deinit();
    const report_json =
        \\{"schema_version":1,"bytecode_sha256":"0000000000000000000000000000000000000000000000000000000000000000","switches":[{"block":"entry","strategy":"linear","default_label":"revert_error","templates_valid":true,"cases":[{"selector":1,"target":"known","route_index":0}]}]}
    ;
    try std.testing.expectError(
        error.DispatcherBytecodeHashMismatch,
        check.validateAndBindBytecode(std.testing.allocator, report_json, "00"),
    );
}
