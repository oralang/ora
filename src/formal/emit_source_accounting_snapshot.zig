//! Emits the data-only Zig/Lean source-accounting policy and decision snapshot.

const std = @import("std");
const formal = @import("ora_formal");

const accounting = formal.source_accounting;
const gate = formal.source_accounting_gate;

const header =
    "/-\n" ++
    "GENERATED — DATA ONLY. Do NOT edit by hand and do NOT add any `theorem`,\n" ++
    "`lemma`, `example`, `axiom`, `sorry`, `instance`, `macro`, attribute, or extra\n" ++
    "`import` to this file. It contains primitive enum, policy, and decision rows\n" ++
    "emitted by the Zig compiler-kernel source-accounting gate. Trusted checks\n" ++
    "live in `Ora/SourceAccountingSync.lean`.\n\n" ++
    "Regenerate with `scripts/check-formal-sync.sh`. Source:\n" ++
    "src/formal/emit_source_accounting_snapshot.zig.\n" ++
    "-/\n\n" ++
    "namespace Ora.Generated\n\n";

const invariant_key: accounting.SiteKey = .{
    .path = "fixture.ora",
    .owner = "function:run",
    .range_start = 10,
    .range_end = 20,
    .kind = .loop_invariant,
    .ordinal = 0,
};

fn missingHandlingFixture() accounting.Manifest {
    const declared = struct {
        const rows = [_]accounting.DeclaredSite{.{ .id = 1, .key = invariant_key }};
    }.rows;
    const typed = struct {
        const rows = [_]accounting.TypedSite{.{ .id = 1, .origin = .source_syntax, .kind = .loop_invariant, .key = invariant_key, .source_fact_id = invariant_key.range_start, .declared_site_id = 1 }};
    }.rows;
    const template_uses = struct {
        const rows = [_]accounting.UseTemplate{
            .{ .site_id = 1, .role = .proof_target },
            .{ .site_id = 1, .role = .assumption_context },
        };
    }.rows;
    const templates = struct {
        const rows = [_]accounting.OwnerTemplate{.{ .id = 1, .owner_key = "function:run", .activation = .comptime_body, .uses = &template_uses }};
    }.rows;
    const expansions = struct {
        const rows = [_]accounting.Expansion{.{ .id = 1, .template_id = 1, .activation = .speculative_fold, .disposition = .fold_committed, .root_runtime_owner = "function:run", .identity = "call:run:0" }};
    }.rows;
    const uses = struct {
        const rows = [_]accounting.SourceFactUse{
            .{ .id = 1, .site_id = 1, .expansion_id = 1, .template_ordinal = 0, .role = .proof_target },
            .{ .id = 2, .site_id = 1, .expansion_id = 1, .template_ordinal = 1, .role = .assumption_context },
        };
    }.rows;
    return .{ .inventory = .{
        .declared_sites = &declared,
        .typed_sites = &typed,
        .owner_templates = &templates,
        .expansions = &expansions,
        .uses = &uses,
    } };
}

fn writeLeanString(out: anytype, value: []const u8) !void {
    try out.writeByte('"');
    for (value) |byte| switch (byte) {
        '\\' => try out.writeAll("\\\\"),
        '"' => try out.writeAll("\\\""),
        '\n' => try out.writeAll("\\n"),
        '\r' => try out.writeAll("\\r"),
        '\t' => try out.writeAll("\\t"),
        else => try out.writeByte(byte),
    };
    try out.writeByte('"');
}

fn writeEnumTags(out: anytype, comptime T: type) !void {
    try out.writeByte('[');
    inline for (std.meta.fields(T), 0..) |field, index| {
        if (index != 0) try out.writeAll(", ");
        try writeLeanString(out, field.name);
    }
    try out.writeByte(']');
}

fn writeDecisionRow(out: anytype, name: []const u8, mode: accounting.CompilationMode, manifest: accounting.Manifest) !void {
    var result = try gate.decide(std.heap.page_allocator, mode, manifest);
    defer result.deinit();
    try out.writeAll("  (");
    try writeLeanString(out, name);
    try out.writeAll(", ");
    try writeLeanString(out, @tagName(mode));
    try out.writeAll(", ");
    try out.writeAll(if (result.decision == .accepted) "true" else "false");
    try out.writeAll(", ");
    if (result.primary_failure) |failure| {
        try out.writeAll("some ");
        try writeLeanString(out, @tagName(failure));
    } else try out.writeAll("none");
    try out.writeAll(", [");
    for (result.failures, 0..) |failure, index| {
        if (index != 0) try out.writeAll(", ");
        try writeLeanString(out, @tagName(failure.code));
    }
    try out.writeAll("])");
}

pub fn main(init: std.process.Init) !void {
    @setEvalBranchQuota(100_000);
    var buffer: [1 << 16]u8 = undefined;
    var stdout_writer = std.Io.File.stdout().writer(init.io, &buffer);
    const out = &stdout_writer.interface;
    try out.writeAll(header);

    try out.writeAll("def sourceAccountingCompilationModeTags : List String := ");
    try writeEnumTags(out, accounting.CompilationMode);
    try out.writeAll("\n\ndef sourceAccountingFactKindTags : List String := ");
    try writeEnumTags(out, accounting.SourceFactKind);
    try out.writeAll("\n\ndef sourceAccountingFactOriginTags : List String := ");
    try writeEnumTags(out, accounting.FactOrigin);
    try out.writeAll("\n\ndef sourceAccountingUseRoleTags : List String := ");
    try writeEnumTags(out, accounting.UseRole);
    try out.writeAll("\n\ndef sourceAccountingTemplateActivationTags : List String := ");
    try writeEnumTags(out, accounting.TemplateActivation);
    try out.writeAll("\n\ndef sourceAccountingHandlingKindTags : List String := ");
    try writeEnumTags(out, accounting.HandlingKind);
    try out.writeAll("\n\ndef sourceAccountingExpansionDispositionTags : List String := ");
    try writeEnumTags(out, accounting.ExpansionDisposition);
    try out.writeAll("\n\ndef sourceAccountingFailureCodeTags : List String := ");
    try writeEnumTags(out, gate.FailureCode);
    const failure_witnesses = try gate.observeFailureWitnesses(init.gpa);
    try out.writeAll("\n\ndef sourceAccountingFailureWitnessTags : List String := [");
    var first_witness = true;
    inline for (std.meta.fields(gate.FailureCode)) |field| {
        const code: gate.FailureCode = @enumFromInt(field.value);
        if (failure_witnesses.rows[@intFromEnum(code)].observed) {
            if (!first_witness) try out.writeAll(", ");
            first_witness = false;
            try writeLeanString(out, @tagName(code));
        }
    }
    try out.writeByte(']');
    try out.writeAll(
        "\n\ndef sourceAccountingFailureWitnessRows :\n" ++
            "    List (String × Bool × Option String × List String) :=\n[\n",
    );
    inline for (std.meta.fields(gate.FailureCode), 0..) |field, witness_index| {
        if (witness_index != 0) try out.writeAll(",\n");
        const code: gate.FailureCode = @enumFromInt(field.value);
        const observation = &failure_witnesses.rows[@intFromEnum(code)];
        try out.writeAll("  (");
        try writeLeanString(out, @tagName(code));
        try out.writeAll(", ");
        try out.writeAll(if (observation.decision == .accepted) "true" else "false");
        try out.writeAll(", ");
        if (observation.primary_failure) |primary| {
            try out.writeAll("some ");
            try writeLeanString(out, @tagName(primary));
        } else try out.writeAll("none");
        try out.writeAll(", [");
        for (observation.failures(), 0..) |failure, failure_index| {
            if (failure_index != 0) try out.writeAll(", ");
            try writeLeanString(out, @tagName(failure));
        }
        try out.writeAll("])");
    }
    try out.writeAll("\n]");

    try out.writeAll("\n\ndef sourceAccountingPolicyRows : List (String × String × String × String) :=\n[\n");
    var first_policy_row = true;
    inline for (std.meta.fields(accounting.CompilationMode)) |mode_field| {
        const mode: accounting.CompilationMode = @enumFromInt(mode_field.value);
        inline for (std.meta.fields(accounting.FactOrigin)) |origin_field| {
            const origin: accounting.FactOrigin = @enumFromInt(origin_field.value);
            inline for (std.meta.fields(accounting.SourceFactKind)) |kind_field| {
                const kind: accounting.SourceFactKind = @enumFromInt(kind_field.value);
                if (!first_policy_row) try out.writeAll(",\n");
                first_policy_row = false;
                try out.writeAll("  (");
                try writeLeanString(out, @tagName(mode));
                try out.writeAll(", ");
                try writeLeanString(out, @tagName(origin));
                try out.writeAll(", ");
                try writeLeanString(out, @tagName(kind));
                try out.writeAll(", \"");
                inline for (std.meta.fields(accounting.UseRole)) |role_field| {
                    const role: accounting.UseRole = @enumFromInt(role_field.value);
                    inline for (std.meta.fields(accounting.HandlingKind)) |handling_field| {
                        const handling: accounting.HandlingKind = @enumFromInt(handling_field.value);
                        inline for (std.meta.fields(accounting.ExpansionDisposition)) |disposition_field| {
                            const disposition: accounting.ExpansionDisposition = @enumFromInt(disposition_field.value);
                            try out.writeByte(if (gate.handlingPermitted(mode, origin, kind, role, handling, disposition)) '1' else '0');
                        }
                    }
                }
                try out.writeAll("\")");
            }
        }
    }
    try out.writeAll("\n]\n\ndef sourceAccountingDecisionRows :\n    List (String × String × Bool × Option String × List String) :=\n[");
    try writeDecisionRow(out, "empty_verified_full", .verified_full, .{});
    try out.writeAll(",\n");
    try writeDecisionRow(out, "missing_invariant_handling", .verified_full, missingHandlingFixture());
    try out.writeAll("\n]\n\nend Ora.Generated\n");
    try out.flush();
}
