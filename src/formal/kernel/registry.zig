//! Compiler-kernel formal gate registry.
//!
//! Kernel gates are listed here and exposed through phase-specific sessions.
//! The underlying gates retain ownership of proof checking and certificate
//! serialization; the registry only coordinates when each phase may run.

const std = @import("std");
const mlir = @import("mlir_c_api").c;
const dispatcher_gate = @import("../dispatcher_table_gate.zig");
const source_accounting = @import("../shared/source_accounting.zig");
const source_accounting_gate = @import("source_accounting_gate.zig");
const artifact_catalog = @import("../shared/artifact_catalog.zig");

pub const GateId = enum {
    dispatcher,
    source_accounting,
};

pub const GatePhase = enum {
    prepare_from_sir,
    bind_backend_output,
    finish_certificate,
    prepare_from_source,
    bind_comptime_evidence,
    bind_symbolic_evidence,
    finish_accounting_decision,
};

pub const GateDefinition = struct {
    id: GateId,
    name: []const u8,
    blocking: bool,
    phases: []const GatePhase,
};

const dispatcher_phases = [_]GatePhase{
    .prepare_from_sir,
    .bind_backend_output,
    .finish_certificate,
};

const source_accounting_phases = [_]GatePhase{
    .prepare_from_source,
    .bind_comptime_evidence,
    .bind_symbolic_evidence,
    .finish_accounting_decision,
};

pub const gates = [_]GateDefinition{
    .{
        .id = .dispatcher,
        .name = "dispatcher_userland",
        .blocking = true,
        .phases = &dispatcher_phases,
    },
    .{
        .id = .source_accounting,
        .name = "source_accounting_kernel",
        .blocking = true,
        .phases = &source_accounting_phases,
    },
};

pub fn definition(id: GateId) *const GateDefinition {
    return switch (id) {
        .dispatcher => &gates[0],
        .source_accounting => &gates[1],
    };
}

/// Compile-time explicit bridge between executable kernel gates and audit
/// catalog gates. String equality is never used as gate identity.
pub fn auditGateId(id: GateId) artifact_catalog.GateId {
    return switch (id) {
        .dispatcher => .dispatcher_kernel,
        .source_accounting => .source_accounting_kernel,
    };
}

/// Returns the only kernel-permitted handling for a symbolic proof-target use
/// in the compiler-derived mode. Adapters therefore do not duplicate the
/// Basic/unverified policy table.
pub fn sourceProofHandlingForMode(
    mode: source_accounting.CompilationMode,
    origin: source_accounting.FactOrigin,
    kind: source_accounting.SourceFactKind,
) !source_accounting.HandlingKind {
    inline for (.{
        source_accounting.HandlingKind.symbolic,
        source_accounting.HandlingKind.reduced_scope_excluded,
        source_accounting.HandlingKind.verification_disabled,
    }) |candidate| {
        if (source_accounting_gate.handlingPermitted(
            mode,
            origin,
            kind,
            .proof_target,
            candidate,
            .symbolic,
        )) return candidate;
    }
    return error.SourceAccountingProofTargetModeHasNoPermittedHandling;
}

pub fn renderSourceAccountingDiagnostic(
    allocator: std.mem.Allocator,
    failure: source_accounting_gate.FailureRow,
) ![]u8 {
    return source_accounting_gate.renderDiagnostic(allocator, failure);
}

pub const SourceAccountingSession = struct {
    parent_allocator: std.mem.Allocator,
    arena: *std.heap.ArenaAllocator,
    mode: source_accounting.CompilationMode,
    inventory: source_accounting.SourceInventory,
    comptime_evidence: ?source_accounting.ComptimeEvidence = null,
    symbolic_evidence: ?source_accounting.SymbolicEvidence = null,
    state: State = .prepared,
    result: ?source_accounting_gate.Result = null,
    report: ?[]const u8 = null,

    pub const FinishedView = struct {
        decision: source_accounting_gate.Decision,
        primary_failure: ?source_accounting_gate.FailureCode,
        failures: []const source_accounting_gate.FailureRow,
        report: []const u8,
    };

    const State = enum {
        prepared,
        comptime_bound,
        symbolic_bound,
        finished,
        poisoned,
    };

    pub fn prepareFromSource(
        allocator: std.mem.Allocator,
        mode: source_accounting.CompilationMode,
        inventory: source_accounting.SourceInventory,
    ) !SourceAccountingSession {
        const arena = try allocator.create(std.heap.ArenaAllocator);
        errdefer allocator.destroy(arena);
        arena.* = std.heap.ArenaAllocator.init(allocator);
        errdefer arena.deinit();
        return .{
            .parent_allocator = allocator,
            .arena = arena,
            .mode = mode,
            .inventory = try cloneInventory(arena.allocator(), inventory),
        };
    }

    pub fn deinit(self: *SourceAccountingSession) void {
        self.arena.deinit();
        self.parent_allocator.destroy(self.arena);
        self.* = undefined;
    }

    pub fn bindComptimeEvidence(self: *SourceAccountingSession, evidence: source_accounting.ComptimeEvidence) !void {
        if (self.state != .prepared) return self.poison(switch (self.state) {
            .comptime_bound, .symbolic_bound => error.SourceAccountingComptimeAlreadyBound,
            .finished => error.SourceAccountingBindAfterFinish,
            .poisoned => error.SourceAccountingSessionPoisoned,
            .prepared => unreachable,
        });
        self.comptime_evidence = cloneComptimeEvidence(self.arena.allocator(), evidence) catch |err| {
            self.state = .poisoned;
            return err;
        };
        self.state = .comptime_bound;
    }

    pub fn bindSymbolicEvidence(self: *SourceAccountingSession, evidence: source_accounting.SymbolicEvidence) !void {
        if (self.mode == .unverified_emit) return self.poison(error.SourceAccountingExplicitVerificationDisabledBindingRequired);
        return self.bindSymbolicEvidenceInternal(evidence);
    }

    fn bindSymbolicEvidenceInternal(self: *SourceAccountingSession, evidence: source_accounting.SymbolicEvidence) !void {
        if (self.state != .comptime_bound) return self.poison(switch (self.state) {
            .prepared => error.SourceAccountingComptimeNotBound,
            .symbolic_bound => error.SourceAccountingSymbolicAlreadyBound,
            .finished => error.SourceAccountingBindAfterFinish,
            .poisoned => error.SourceAccountingSessionPoisoned,
            .comptime_bound => unreachable,
        });
        self.symbolic_evidence = cloneSymbolicEvidence(self.arena.allocator(), evidence) catch |err| {
            self.state = .poisoned;
            return err;
        };
        self.state = .symbolic_bound;
    }

    pub fn bindVerificationDisabled(self: *SourceAccountingSession, evidence: source_accounting.SymbolicEvidence) !void {
        if (self.mode != .unverified_emit) return self.poison(error.SourceAccountingVerificationDisableNotPermitted);
        return self.bindSymbolicEvidenceInternal(evidence);
    }

    pub fn finishAccountingDecision(self: *SourceAccountingSession) !FinishedView {
        if (self.state != .symbolic_bound) return self.poison(switch (self.state) {
            .prepared, .comptime_bound => error.SourceAccountingFinishBeforeBind,
            .finished => error.SourceAccountingAlreadyFinished,
            .poisoned => error.SourceAccountingSessionPoisoned,
            .symbolic_bound => unreachable,
        });
        const manifest: source_accounting.Manifest = .{
            .inventory = self.inventory,
            .comptime_evidence = self.comptime_evidence.?,
            .symbolic = self.symbolic_evidence.?,
        };
        self.result = source_accounting_gate.decide(self.arena.allocator(), self.mode, manifest) catch |err| {
            self.state = .poisoned;
            return err;
        };
        self.report = source_accounting_gate.renderReport(self.arena.allocator(), self.mode, manifest, self.result.?) catch |err| {
            self.state = .poisoned;
            return err;
        };
        self.state = .finished;
        return .{
            .decision = self.result.?.decision,
            .primary_failure = self.result.?.primary_failure,
            .failures = self.result.?.failures,
            .report = self.report.?,
        };
    }

    fn poison(self: *SourceAccountingSession, err: anyerror) anyerror {
        self.state = .poisoned;
        return err;
    }

    fn cloneInventory(allocator: std.mem.Allocator, input: source_accounting.SourceInventory) !source_accounting.SourceInventory {
        const declared = try allocator.dupe(source_accounting.DeclaredSite, input.declared_sites);
        for (declared) |*row| {
            row.key = try cloneSiteKey(allocator, row.key);
            if (row.label) |value| row.label = try allocator.dupe(u8, value);
        }
        const typed = try allocator.dupe(source_accounting.TypedSite, input.typed_sites);
        for (typed) |*row| row.key = try cloneSiteKey(allocator, row.key);
        const derivations = try allocator.dupe(source_accounting.GeneratedFactDerivation, input.generated_fact_derivations);
        for (derivations) |*row| {
            row.semantic_rule = try allocator.dupe(u8, row.semantic_rule);
            row.anchor.file = try allocator.dupe(u8, row.anchor.file);
            row.parent_identity = try allocator.dupe(u8, row.parent_identity);
        }
        const templates = try allocator.dupe(source_accounting.OwnerTemplate, input.owner_templates);
        for (templates) |*row| {
            row.owner_key = try allocator.dupe(u8, row.owner_key);
            row.uses = try allocator.dupe(source_accounting.UseTemplate, row.uses);
            const source_nodes = row.control_nodes;
            const cloned_nodes = try allocator.alloc(source_accounting.ControlNodeTemplate, source_nodes.len);
            for (source_nodes, cloned_nodes) |node, *cloned| {
                cloned.* = node;
                cloned.range.file = try allocator.dupe(u8, node.range.file);
                cloned.attached_use_ordinals = try allocator.dupe(u32, node.attached_use_ordinals);
            }
            row.control_nodes = cloned_nodes;
            row.control_edges = try allocator.dupe(source_accounting.ControlEdgeTemplate, row.control_edges);
            row.terminal_slots = try allocator.dupe(u32, row.terminal_slots);
        }
        const expansions = try allocator.dupe(source_accounting.Expansion, input.expansions);
        for (expansions) |*row| {
            row.root_runtime_owner = try allocator.dupe(u8, row.root_runtime_owner);
            const chain = try allocator.dupe(source_accounting.SourceRange, row.folded_call_site_chain);
            for (chain) |*site| site.file = try allocator.dupe(u8, site.file);
            row.folded_call_site_chain = chain;
            if (row.imported_module) |value| row.imported_module = try allocator.dupe(u8, value);
            const bindings = try allocator.alloc([]const u8, row.generic_bindings.len);
            for (row.generic_bindings, bindings) |value, *binding| binding.* = try allocator.dupe(u8, value);
            row.generic_bindings = bindings;
            if (row.trait_implementation) |value| row.trait_implementation = try allocator.dupe(u8, value);
            if (row.trait_method) |value| row.trait_method = try allocator.dupe(u8, value);
            row.identity = try allocator.dupe(u8, row.identity);
        }
        const uses = try allocator.dupe(source_accounting.SourceFactUse, input.uses);
        const nodes = try allocator.dupe(source_accounting.ControlNode, input.control_nodes);
        for (nodes) |*row| {
            row.range.file = try allocator.dupe(u8, row.range.file);
            row.attached_use_ids = try allocator.dupe(source_accounting.UseId, row.attached_use_ids);
        }
        return .{
            .declared_sites = declared,
            .typed_sites = typed,
            .generated_fact_derivations = derivations,
            .owner_templates = templates,
            .expansions = expansions,
            .uses = uses,
            .control_nodes = nodes,
            .control_edges = try allocator.dupe(source_accounting.ControlEdge, input.control_edges),
        };
    }

    fn cloneSiteKey(allocator: std.mem.Allocator, input: source_accounting.SiteKey) !source_accounting.SiteKey {
        var result = input;
        result.path = try allocator.dupe(u8, input.path);
        result.owner = try allocator.dupe(u8, input.owner);
        return result;
    }

    fn cloneComptimeEvidence(allocator: std.mem.Allocator, input: source_accounting.ComptimeEvidence) !source_accounting.ComptimeEvidence {
        const folds = try allocator.dupe(source_accounting.FoldRecord, input.folds);
        for (folds) |*row| row.events = try allocator.dupe(source_accounting.TraceEvent, row.events);
        return .{
            .folds = folds,
            .predicate_events = try allocator.dupe(source_accounting.PredicateEvent, input.predicate_events),
            .handlings = try cloneHandlings(allocator, input.handlings),
        };
    }

    fn cloneSymbolicEvidence(allocator: std.mem.Allocator, input: source_accounting.SymbolicEvidence) !source_accounting.SymbolicEvidence {
        return .{
            .obligations = try cloneCoveredRows(allocator, input.obligations),
            .assumptions = try cloneCoveredRows(allocator, input.assumptions),
            .queries = try cloneQueryRows(allocator, input.queries),
            .runtime_checks = try cloneCoveredRows(allocator, input.runtime_checks),
            .frame_results = try cloneValidationRows(allocator, input.frame_results),
            .state_effects = try cloneValidationRows(allocator, input.state_effects),
            .handlings = try cloneHandlings(allocator, input.handlings),
        };
    }

    fn cloneCoveredRows(allocator: std.mem.Allocator, input: []const source_accounting.CoveredEvidence) ![]const source_accounting.CoveredEvidence {
        const rows = try allocator.dupe(source_accounting.CoveredEvidence, input);
        for (rows) |*row| row.covered_use_ids = try allocator.dupe(source_accounting.UseId, row.covered_use_ids);
        return rows;
    }

    fn cloneQueryRows(allocator: std.mem.Allocator, input: []const source_accounting.QueryEvidence) ![]const source_accounting.QueryEvidence {
        const rows = try allocator.dupe(source_accounting.QueryEvidence, input);
        for (rows) |*row| row.covered_use_ids = try allocator.dupe(source_accounting.UseId, row.covered_use_ids);
        return rows;
    }

    fn cloneValidationRows(allocator: std.mem.Allocator, input: []const source_accounting.ValidationEvidence) ![]const source_accounting.ValidationEvidence {
        const rows = try allocator.dupe(source_accounting.ValidationEvidence, input);
        for (rows) |*row| row.covered_use_ids = try allocator.dupe(source_accounting.UseId, row.covered_use_ids);
        return rows;
    }

    fn cloneHandlings(allocator: std.mem.Allocator, input: []const source_accounting.HandlingRecord) ![]const source_accounting.HandlingRecord {
        const rows = try allocator.dupe(source_accounting.HandlingRecord, input);
        for (rows) |*row| {
            row.obligation_ids = try allocator.dupe(source_accounting.EvidenceId, row.obligation_ids);
            row.assumption_ids = try allocator.dupe(source_accounting.EvidenceId, row.assumption_ids);
            row.query_ids = try allocator.dupe(source_accounting.EvidenceId, row.query_ids);
            row.runtime_check_ids = try allocator.dupe(source_accounting.EvidenceId, row.runtime_check_ids);
            row.frame_result_ids = try allocator.dupe(source_accounting.EvidenceId, row.frame_result_ids);
            row.state_effect_ids = try allocator.dupe(source_accounting.EvidenceId, row.state_effect_ids);
            row.predicate_event_ids = try allocator.dupe(source_accounting.EvidenceId, row.predicate_event_ids);
            if (row.rejection_reason) |value| row.rejection_reason = try allocator.dupe(u8, value);
        }
        return rows;
    }
};

pub const DispatcherSession = struct {
    check: dispatcher_gate.CheckResult,
    lifecycle: Lifecycle = .{},

    pub const CertificateKind = enum {
        sir,
        bytecode,
    };

    pub fn prepareFromSir(
        allocator: std.mem.Allocator,
        ctx: mlir.MlirContext,
        module: mlir.MlirModule,
        intent_json: []const u8,
        sir_text: []const u8,
        file_path: []const u8,
        process_environ: std.process.Environ,
        stdout: anytype,
    ) !DispatcherSession {
        return .{
            .check = try dispatcher_gate.checkCurrentModule(
                allocator,
                ctx,
                module,
                intent_json,
                sir_text,
                file_path,
                process_environ,
                stdout,
            ),
        };
    }

    pub fn deinit(self: *DispatcherSession) void {
        self.check.deinit();
        self.* = undefined;
    }

    pub fn bindBackendOutput(
        self: *DispatcherSession,
        allocator: std.mem.Allocator,
        report_json: []const u8,
        bytecode_hex: []const u8,
    ) !void {
        try self.lifecycle.requireBackendBinding();
        try self.check.validateAndBindBytecode(allocator, report_json, bytecode_hex);
        self.lifecycle.completeBackendBinding();
    }

    /// Completes the session and returns the certificate bytes produced by the
    /// underlying gate. No parsing or reserialization occurs in the registry.
    pub fn finishCertificate(
        self: *DispatcherSession,
        kind: CertificateKind,
    ) ![]const u8 {
        try self.lifecycle.finish(kind);
        return self.check.certificate_json;
    }

    pub fn writeVerificationSummary(
        self: *const DispatcherSession,
        writer: anytype,
    ) !void {
        const bytecode_bound = try self.lifecycle.finishedWithBytecode();
        try self.check.writeVerificationSummary(writer, bytecode_bound);
    }

    const State = enum {
        prepared,
        backend_bound,
        finished_sir,
        finished_bytecode,
    };

    const Lifecycle = struct {
        state: State = .prepared,

        fn requireBackendBinding(self: *const Lifecycle) !void {
            return switch (self.state) {
                .prepared => {},
                .backend_bound => error.DispatcherBackendAlreadyBound,
                .finished_sir, .finished_bytecode => error.DispatcherCertificateAlreadyFinished,
            };
        }

        fn completeBackendBinding(self: *Lifecycle) void {
            std.debug.assert(self.state == .prepared);
            self.state = .backend_bound;
        }

        fn finish(self: *Lifecycle, kind: CertificateKind) !void {
            switch (kind) {
                .sir => switch (self.state) {
                    .prepared => self.state = .finished_sir,
                    .backend_bound => return error.DispatcherBackendOutputRequiresBytecodeCertificate,
                    .finished_sir, .finished_bytecode => return error.DispatcherCertificateAlreadyFinished,
                },
                .bytecode => switch (self.state) {
                    .prepared => return error.DispatcherBackendOutputNotBound,
                    .backend_bound => self.state = .finished_bytecode,
                    .finished_sir, .finished_bytecode => return error.DispatcherCertificateAlreadyFinished,
                },
            }
        }

        fn finishedWithBytecode(self: *const Lifecycle) !bool {
            return switch (self.state) {
                .finished_sir => false,
                .finished_bytecode => true,
                .prepared, .backend_bound => error.DispatcherCertificateNotFinished,
            };
        }
    };
};

test "kernel registry describes the blocking dispatcher phases" {
    const dispatcher = definition(.dispatcher);
    try std.testing.expectEqual(GateId.dispatcher, dispatcher.id);
    try std.testing.expectEqualStrings("dispatcher_userland", dispatcher.name);
    try std.testing.expect(dispatcher.blocking);
    try std.testing.expectEqualSlices(GatePhase, &dispatcher_phases, dispatcher.phases);
}

test "kernel registry describes the blocking source-accounting phases" {
    const accounting_gate = definition(.source_accounting);
    try std.testing.expectEqual(GateId.source_accounting, accounting_gate.id);
    try std.testing.expectEqualStrings("source_accounting_kernel", accounting_gate.name);
    try std.testing.expect(accounting_gate.blocking);
    try std.testing.expectEqualSlices(GatePhase, &source_accounting_phases, accounting_gate.phases);
    try std.testing.expectEqual(artifact_catalog.GateId.source_accounting_kernel, auditGateId(.source_accounting));
}

test "every executable kernel gate has an explicit audit-catalog identity" {
    inline for (std.meta.fields(GateId)) |field| {
        const id: GateId = @enumFromInt(field.value);
        _ = auditGateId(id);
    }
}

test "source-accounting lifecycle accepts explicit empty binds" {
    var session = try SourceAccountingSession.prepareFromSource(
        std.testing.allocator,
        .verified_full,
        .{},
    );
    defer session.deinit();
    try session.bindComptimeEvidence(.{});
    try session.bindSymbolicEvidence(.{});
    const finished = try session.finishAccountingDecision();
    try std.testing.expectEqual(source_accounting_gate.Decision.accepted, finished.decision);
    try std.testing.expectEqual(@as(?source_accounting_gate.FailureCode, null), finished.primary_failure);
    try std.testing.expect(std.mem.indexOf(u8, finished.report, "\"decision\": \"accepted\"") != null);
    try std.testing.expectError(error.SourceAccountingAlreadyFinished, session.finishAccountingDecision());
    try std.testing.expectError(error.SourceAccountingSessionPoisoned, session.bindComptimeEvidence(.{}));
}

test "source-accounting lifecycle poisons every out-of-order transition" {
    {
        var session = try SourceAccountingSession.prepareFromSource(std.testing.allocator, .verified_full, .{});
        defer session.deinit();
        try std.testing.expectError(error.SourceAccountingFinishBeforeBind, session.finishAccountingDecision());
        try std.testing.expectError(error.SourceAccountingSessionPoisoned, session.bindComptimeEvidence(.{}));
    }
    {
        var session = try SourceAccountingSession.prepareFromSource(std.testing.allocator, .verified_full, .{});
        defer session.deinit();
        try std.testing.expectError(error.SourceAccountingComptimeNotBound, session.bindSymbolicEvidence(.{}));
        try std.testing.expectError(error.SourceAccountingSessionPoisoned, session.bindComptimeEvidence(.{}));
    }
    {
        var session = try SourceAccountingSession.prepareFromSource(std.testing.allocator, .verified_full, .{});
        defer session.deinit();
        try session.bindComptimeEvidence(.{});
        try std.testing.expectError(error.SourceAccountingComptimeAlreadyBound, session.bindComptimeEvidence(.{}));
        try std.testing.expectError(error.SourceAccountingSessionPoisoned, session.bindSymbolicEvidence(.{}));
    }
}

test "verification-disabled binding is exclusive to unverified emit" {
    {
        var session = try SourceAccountingSession.prepareFromSource(std.testing.allocator, .verified_full, .{});
        defer session.deinit();
        try session.bindComptimeEvidence(.{});
        try std.testing.expectError(
            error.SourceAccountingVerificationDisableNotPermitted,
            session.bindVerificationDisabled(.{}),
        );
    }
    {
        var session = try SourceAccountingSession.prepareFromSource(std.testing.allocator, .unverified_emit, .{});
        defer session.deinit();
        try session.bindComptimeEvidence(.{});
        try std.testing.expectError(
            error.SourceAccountingExplicitVerificationDisabledBindingRequired,
            session.bindSymbolicEvidence(.{}),
        );
        try std.testing.expectError(error.SourceAccountingSessionPoisoned, session.bindVerificationDisabled(.{}));
    }
    {
        var session = try SourceAccountingSession.prepareFromSource(std.testing.allocator, .unverified_emit, .{});
        defer session.deinit();
        try session.bindComptimeEvidence(.{});
        try session.bindVerificationDisabled(.{});
        const finished = try session.finishAccountingDecision();
        try std.testing.expectEqual(source_accounting_gate.Decision.accepted, finished.decision);
    }
}

test "dispatcher lifecycle requires backend binding before bytecode finish" {
    var lifecycle: DispatcherSession.Lifecycle = .{};
    try std.testing.expectError(
        error.DispatcherBackendOutputNotBound,
        lifecycle.finish(.bytecode),
    );
    try lifecycle.requireBackendBinding();
    lifecycle.completeBackendBinding();
    try std.testing.expectError(
        error.DispatcherBackendAlreadyBound,
        lifecycle.requireBackendBinding(),
    );
    try lifecycle.finish(.bytecode);
    try std.testing.expect(try lifecycle.finishedWithBytecode());
    try std.testing.expectError(
        error.DispatcherCertificateAlreadyFinished,
        lifecycle.finish(.bytecode),
    );
}

test "dispatcher lifecycle permits a SIR-only certificate" {
    var lifecycle: DispatcherSession.Lifecycle = .{};
    try lifecycle.finish(.sir);
    try std.testing.expect(!try lifecycle.finishedWithBytecode());
    try std.testing.expectError(
        error.DispatcherCertificateAlreadyFinished,
        lifecycle.requireBackendBinding(),
    );
}

test "dispatcher session returns the gate certificate without reserialization" {
    const certificate_json =
        \\{
        \\  "schema_version": 1,
        \\  "proof_surface": "dispatcher_userland"
        \\}
    ;
    var session: DispatcherSession = .{
        .check = .{
            .arena = std.heap.ArenaAllocator.init(std.testing.allocator),
            .certificate_json = certificate_json,
            .switch_manifest_json = "{}",
            .switch_count = 0,
            .case_count = 0,
        },
    };
    defer session.deinit();

    const finished = try session.finishCertificate(.sir);
    try std.testing.expectEqualStrings(certificate_json, finished);
    try std.testing.expectEqual(@intFromPtr(certificate_json.ptr), @intFromPtr(finished.ptr));
}
