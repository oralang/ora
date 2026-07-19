//! Compiler-kernel source-formal conservation gate.
//!
//! Producers are untrusted adapters. This module owns structural validation,
//! the closed handling policy, trace validation, reconciliation, deterministic
//! failure ordering, and the accounting report.

const std = @import("std");
const accounting = @import("../shared/source_accounting.zig");

pub const FailureCode = enum(u8) {
    duplicate_identity,
    identity_hash_collision,
    unknown_site,
    unknown_expansion,
    unknown_use,
    unknown_evidence_reference,
    orphan_evidence,
    evidence_coverage_mismatch,
    missing_semantic_site,
    unknown_semantic_site,
    missing_semantic_use,
    unexpected_semantic_use,
    invalid_site_role,
    invalid_owner_template,
    invalid_expansion_activation,
    invalid_expansion_parent,
    invalid_expansion_disposition,
    invalid_control_graph,
    invalid_fold_trace,
    abandoned_fold_evidence,
    missing_symbolic_replacement,
    invalid_generated_fact_derivation,
    missing_handling,
    duplicate_handling,
    rejected_handling,
    missing_symbolic_obligation,
    missing_symbolic_query,
    invalid_symbolic_query_kind,
    missing_assumption_incorporation,
    missing_runtime_enforcement,
    missing_frame_validation,
    missing_state_effect,
    zero_concrete_checks,
    invalid_concrete_checkpoint,
    false_concrete_target,
    false_concrete_assumption,
    invalid_control_elimination,
    invalid_frame_validation,
    invalid_state_effect,
    handling_not_permitted,
    reduced_scope_excluded_not_permitted,
    verification_disabled_not_permitted,
};

pub const Decision = enum(u8) { accepted, rejected };

pub const FailureRow = struct {
    code: FailureCode,
    source: ?accounting.SourceRange = null,
    site_id: ?accounting.SiteId = null,
    expansion_id: ?accounting.ExpansionId = null,
    use_id: ?accounting.UseId = null,
    handling_id: ?accounting.HandlingId = null,
    evidence_id: ?accounting.EvidenceId = null,
};

pub const Result = struct {
    allocator: std.mem.Allocator,
    decision: Decision,
    primary_failure: ?FailureCode,
    failures: []const FailureRow,

    pub fn deinit(self: *Result) void {
        self.allocator.free(self.failures);
        self.* = undefined;
    }
};

const UseKey = struct {
    expansion_id: accounting.ExpansionId,
    template_ordinal: u32,
};

const NodeSlotKey = struct {
    expansion_id: accounting.ExpansionId,
    slot: u32,
};

const EdgeShapeKey = struct {
    expansion_id: accounting.ExpansionId,
    from: accounting.ControlNodeId,
    to: accounting.ControlNodeId,
    kind: accounting.ControlEdgeKind,
};

const PredicateKey = struct {
    fold_id: accounting.FoldId,
    use_id: accounting.UseId,
    node_id: accounting.ControlNodeId,
    value: bool,
};

const TemplateSiteRoleKey = struct {
    template_id: accounting.TemplateId,
    site_id: accounting.SiteId,
    role: accounting.UseRole,
};

const TemplateNodeSlotKey = struct {
    template_id: accounting.TemplateId,
    slot: u32,
};

const TemplateEdgeShapeKey = struct {
    template_id: accounting.TemplateId,
    from_slot: u32,
    to_slot: u32,
    kind: accounting.ControlEdgeKind,
};

const TemplateAttachmentKey = struct {
    template_id: accounting.TemplateId,
    node_slot: u32,
    use_ordinal: u32,
};

const NodeUseKey = struct {
    node_id: accounting.ControlNodeId,
    use_id: accounting.UseId,
};

const EvidenceUseKey = struct {
    evidence_id: accounting.EvidenceId,
    use_id: accounting.UseId,
};

const FoldNodeKey = struct {
    fold_id: accounting.FoldId,
    node_id: accounting.ControlNodeId,
};

const Context = struct {
    allocator: std.mem.Allocator,
    mode: accounting.CompilationMode,
    manifest: accounting.Manifest,
    failures: std.ArrayList(FailureRow) = .empty,

    declared: std.AutoHashMap(accounting.SiteId, usize),
    typed: std.AutoHashMap(accounting.SiteId, usize),
    derivations: std.AutoHashMap(accounting.Id, usize),
    templates: std.AutoHashMap(accounting.TemplateId, usize),
    template_site_roles: std.AutoHashMap(TemplateSiteRoleKey, void),
    template_node_slots: std.AutoHashMap(TemplateNodeSlotKey, void),
    template_edge_shapes: std.AutoHashMap(TemplateEdgeShapeKey, void),
    template_attachments: std.AutoHashMap(TemplateAttachmentKey, void),
    expansions: std.AutoHashMap(accounting.ExpansionId, usize),
    uses: std.AutoHashMap(accounting.UseId, usize),
    use_keys: std.AutoHashMap(UseKey, accounting.UseId),
    nodes: std.AutoHashMap(accounting.ControlNodeId, usize),
    node_slots: std.AutoHashMap(NodeSlotKey, accounting.ControlNodeId),
    node_attached_uses: std.AutoHashMap(NodeUseKey, void),
    edges: std.AutoHashMap(accounting.ControlEdgeId, usize),
    edge_shapes: std.AutoHashMap(EdgeShapeKey, accounting.ControlEdgeId),
    folds: std.AutoHashMap(accounting.FoldId, usize),
    handlings: std.AutoHashMap(accounting.HandlingId, HandlingLocation),
    handling_by_use: std.AutoHashMap(accounting.UseId, accounting.HandlingId),
    evidence: std.AutoHashMap(accounting.EvidenceId, EvidenceLocation),
    evidence_refs: std.AutoHashMap(accounting.EvidenceId, u32),
    evidence_coverage: std.AutoHashMap(EvidenceUseKey, void),
    handling_evidence_refs: std.AutoHashMap(EvidenceUseKey, void),
    predicate_rows: std.AutoHashMap(PredicateKey, u32),
    predicate_first_ids: std.AutoHashMap(PredicateKey, accounting.EvidenceId),
    predicate_traces: std.AutoHashMap(PredicateKey, u32),
    fold_visited_nodes: std.AutoHashMap(FoldNodeKey, void),

    const HandlingLocation = union(enum) {
        concrete: usize,
        symbolic: usize,
    };

    const EvidenceLocation = union(enum) {
        predicate: usize,
        obligation: usize,
        assumption: usize,
        query: usize,
        runtime_check: usize,
        frame_result: usize,
        state_effect: usize,
    };

    fn init(allocator: std.mem.Allocator, mode: accounting.CompilationMode, manifest: accounting.Manifest) Context {
        return .{
            .allocator = allocator,
            .mode = mode,
            .manifest = manifest,
            .declared = std.AutoHashMap(accounting.SiteId, usize).init(allocator),
            .typed = std.AutoHashMap(accounting.SiteId, usize).init(allocator),
            .derivations = std.AutoHashMap(accounting.Id, usize).init(allocator),
            .templates = std.AutoHashMap(accounting.TemplateId, usize).init(allocator),
            .template_site_roles = std.AutoHashMap(TemplateSiteRoleKey, void).init(allocator),
            .template_node_slots = std.AutoHashMap(TemplateNodeSlotKey, void).init(allocator),
            .template_edge_shapes = std.AutoHashMap(TemplateEdgeShapeKey, void).init(allocator),
            .template_attachments = std.AutoHashMap(TemplateAttachmentKey, void).init(allocator),
            .expansions = std.AutoHashMap(accounting.ExpansionId, usize).init(allocator),
            .uses = std.AutoHashMap(accounting.UseId, usize).init(allocator),
            .use_keys = std.AutoHashMap(UseKey, accounting.UseId).init(allocator),
            .nodes = std.AutoHashMap(accounting.ControlNodeId, usize).init(allocator),
            .node_slots = std.AutoHashMap(NodeSlotKey, accounting.ControlNodeId).init(allocator),
            .node_attached_uses = std.AutoHashMap(NodeUseKey, void).init(allocator),
            .edges = std.AutoHashMap(accounting.ControlEdgeId, usize).init(allocator),
            .edge_shapes = std.AutoHashMap(EdgeShapeKey, accounting.ControlEdgeId).init(allocator),
            .folds = std.AutoHashMap(accounting.FoldId, usize).init(allocator),
            .handlings = std.AutoHashMap(accounting.HandlingId, HandlingLocation).init(allocator),
            .handling_by_use = std.AutoHashMap(accounting.UseId, accounting.HandlingId).init(allocator),
            .evidence = std.AutoHashMap(accounting.EvidenceId, EvidenceLocation).init(allocator),
            .evidence_refs = std.AutoHashMap(accounting.EvidenceId, u32).init(allocator),
            .evidence_coverage = std.AutoHashMap(EvidenceUseKey, void).init(allocator),
            .handling_evidence_refs = std.AutoHashMap(EvidenceUseKey, void).init(allocator),
            .predicate_rows = std.AutoHashMap(PredicateKey, u32).init(allocator),
            .predicate_first_ids = std.AutoHashMap(PredicateKey, accounting.EvidenceId).init(allocator),
            .predicate_traces = std.AutoHashMap(PredicateKey, u32).init(allocator),
            .fold_visited_nodes = std.AutoHashMap(FoldNodeKey, void).init(allocator),
        };
    }

    fn deinit(self: *Context) void {
        self.failures.deinit(self.allocator);
        self.declared.deinit();
        self.typed.deinit();
        self.derivations.deinit();
        self.templates.deinit();
        self.template_site_roles.deinit();
        self.template_node_slots.deinit();
        self.template_edge_shapes.deinit();
        self.template_attachments.deinit();
        self.expansions.deinit();
        self.uses.deinit();
        self.use_keys.deinit();
        self.nodes.deinit();
        self.node_slots.deinit();
        self.node_attached_uses.deinit();
        self.edges.deinit();
        self.edge_shapes.deinit();
        self.folds.deinit();
        self.handlings.deinit();
        self.handling_by_use.deinit();
        self.evidence.deinit();
        self.evidence_refs.deinit();
        self.evidence_coverage.deinit();
        self.handling_evidence_refs.deinit();
        self.predicate_rows.deinit();
        self.predicate_first_ids.deinit();
        self.predicate_traces.deinit();
        self.fold_visited_nodes.deinit();
    }

    fn fail(self: *Context, row: FailureRow) !void {
        try self.failures.append(self.allocator, row);
    }

    fn typedSite(self: *const Context, id: accounting.SiteId) ?accounting.TypedSite {
        const index = self.typed.get(id) orelse return null;
        return self.manifest.inventory.typed_sites[index];
    }

    fn expansion(self: *const Context, id: accounting.ExpansionId) ?accounting.Expansion {
        const index = self.expansions.get(id) orelse return null;
        return self.manifest.inventory.expansions[index];
    }

    fn use(self: *const Context, id: accounting.UseId) ?accounting.SourceFactUse {
        const index = self.uses.get(id) orelse return null;
        return self.manifest.inventory.uses[index];
    }

    fn fold(self: *const Context, id: accounting.FoldId) ?accounting.FoldRecord {
        const index = self.folds.get(id) orelse return null;
        return self.manifest.comptime_evidence.folds[index];
    }

    fn handling(self: *const Context, location: HandlingLocation) accounting.HandlingRecord {
        return switch (location) {
            .concrete => |index| self.manifest.comptime_evidence.handlings[index],
            .symbolic => |index| self.manifest.symbolic.handlings[index],
        };
    }
};

pub fn decide(
    allocator: std.mem.Allocator,
    mode: accounting.CompilationMode,
    manifest: accounting.Manifest,
) !Result {
    var ctx = Context.init(allocator, mode, manifest);
    defer ctx.deinit();

    if (manifest.version != accounting.schema_version) {
        try ctx.fail(.{ .code = .invalid_owner_template });
    }
    try indexRows(&ctx);
    try validateSites(&ctx);
    try validateTemplatesExpansionsAndUses(&ctx);
    try validateControlGraphAndFolds(&ctx);
    try validateHandlingsAndEvidence(&ctx);
    attachFailureSources(&ctx);

    std.mem.sort(FailureRow, ctx.failures.items, {}, lessFailure);
    const owned = try allocator.dupe(FailureRow, ctx.failures.items);
    return .{
        .allocator = allocator,
        .decision = if (owned.len == 0) .accepted else .rejected,
        .primary_failure = if (owned.len == 0) null else owned[0].code,
        .failures = owned,
    };
}

fn attachFailureSources(ctx: *Context) void {
    for (ctx.failures.items) |*failure| {
        const site_id = failure.site_id orelse if (failure.use_id) |use_id|
            if (ctx.use(use_id)) |use| use.site_id else null
        else
            null;
        const site = if (site_id) |id| ctx.typedSite(id) else null;
        if (site) |row| failure.source = .{
            .file = row.key.path,
            .start = row.key.range_start,
            .end = row.key.range_end,
        };
    }
}

pub fn renderReport(
    allocator: std.mem.Allocator,
    mode: accounting.CompilationMode,
    manifest: accounting.Manifest,
    result: Result,
) ![]u8 {
    const inventory_bytes = try renderInventoryCanonical(allocator, manifest.inventory);
    defer allocator.free(inventory_bytes);
    const evidence_bytes = try renderEvidenceCanonical(allocator, manifest.comptime_evidence, manifest.symbolic);
    defer allocator.free(evidence_bytes);
    const inventory_hash = sha256Hex(inventory_bytes);
    const evidence_hash = sha256Hex(evidence_bytes);

    var out = std.Io.Writer.Allocating.init(allocator);
    errdefer out.deinit();
    const writer = &out.writer;
    try writer.print("{{\n  \"schema_version\": {d},\n  \"mode\": ", .{accounting.schema_version});
    try writeJsonString(writer, @tagName(mode));
    try writer.writeAll(",\n  \"decision\": ");
    try writeJsonString(writer, @tagName(result.decision));
    try writer.writeAll(",\n  \"primary_failure\": ");
    if (result.primary_failure) |failure| try writeJsonString(writer, @tagName(failure)) else try writer.writeAll("null");
    try writer.writeAll(",\n  \"inventory_sha256\": ");
    try writeJsonString(writer, &inventory_hash);
    try writer.writeAll(",\n  \"evidence_sha256\": ");
    try writeJsonString(writer, &evidence_hash);
    try writer.print(
        ",\n  \"counts\": {{\"declared_sites\":{d},\"typed_sites\":{d},\"generated_facts\":{d},\"owner_templates\":{d},\"expansions\":{d},\"uses\":{d},\"control_nodes\":{d},\"control_edges\":{d},\"folds\":{d},\"predicate_events\":{d},\"obligations\":{d},\"assumptions\":{d},\"queries\":{d},\"runtime_checks\":{d},\"frame_results\":{d},\"state_effects\":{d},\"handlings\":{d},\"failures\":{d}",
        .{
            manifest.inventory.declared_sites.len,
            manifest.inventory.typed_sites.len,
            manifest.inventory.generated_fact_derivations.len,
            manifest.inventory.owner_templates.len,
            manifest.inventory.expansions.len,
            manifest.inventory.uses.len,
            manifest.inventory.control_nodes.len,
            manifest.inventory.control_edges.len,
            manifest.comptime_evidence.folds.len,
            manifest.comptime_evidence.predicate_events.len,
            manifest.symbolic.obligations.len,
            manifest.symbolic.assumptions.len,
            manifest.symbolic.queries.len,
            manifest.symbolic.runtime_checks.len,
            manifest.symbolic.frame_results.len,
            manifest.symbolic.state_effects.len,
            manifest.comptime_evidence.handlings.len + manifest.symbolic.handlings.len,
            result.failures.len,
        },
    );
    try writeCountBreakdowns(allocator, writer, manifest, result);
    try writer.writeByte('}');
    try writer.writeAll(",\n  \"declared_sites\": ");
    try writeDeclaredSites(writer, allocator, manifest.inventory.declared_sites);
    try writer.writeAll(",\n  \"typed_sites\": ");
    try writeTypedSites(writer, allocator, manifest.inventory.typed_sites);
    try writer.writeAll(",\n  \"generated_fact_derivations\": ");
    try writeGeneratedDerivations(writer, allocator, manifest.inventory.generated_fact_derivations);
    try writer.writeAll(",\n  \"owner_templates\": ");
    try writeTemplates(writer, allocator, manifest.inventory.owner_templates);
    try writer.writeAll(",\n  \"expansions\": ");
    try writeExpansions(writer, allocator, manifest.inventory.expansions);
    try writer.writeAll(",\n  \"uses\": ");
    try writeUses(writer, allocator, manifest.inventory.uses);
    try writer.writeAll(",\n  \"control_nodes\": ");
    try writeControlNodes(writer, allocator, manifest.inventory.control_nodes);
    try writer.writeAll(",\n  \"control_edges\": ");
    try writeControlEdges(writer, allocator, manifest.inventory.control_edges);
    try writer.writeAll(",\n  \"folds\": ");
    try writeFolds(writer, allocator, manifest.comptime_evidence.folds);
    try writer.writeAll(",\n  \"predicate_events\": ");
    try writePredicateEvents(writer, allocator, manifest.comptime_evidence.predicate_events);
    try writer.writeAll(",\n  \"obligations\": ");
    try writeCoveredEvidenceRows(writer, allocator, manifest.symbolic.obligations);
    try writer.writeAll(",\n  \"assumptions\": ");
    try writeCoveredEvidenceRows(writer, allocator, manifest.symbolic.assumptions);
    try writer.writeAll(",\n  \"queries\": ");
    try writeQueryEvidenceRows(writer, allocator, manifest.symbolic.queries);
    try writer.writeAll(",\n  \"runtime_checks\": ");
    try writeCoveredEvidenceRows(writer, allocator, manifest.symbolic.runtime_checks);
    try writer.writeAll(",\n  \"frame_results\": ");
    try writeValidationEvidenceRows(writer, allocator, manifest.symbolic.frame_results);
    try writer.writeAll(",\n  \"state_effects\": ");
    try writeValidationEvidenceRows(writer, allocator, manifest.symbolic.state_effects);
    try writer.writeAll(",\n  \"handlings\": ");
    try writeAllHandlings(writer, allocator, manifest.comptime_evidence.handlings, manifest.symbolic.handlings);
    try writer.writeAll(",\n  \"failures\": ");
    try writeFailures(writer, result.failures);
    try writer.writeAll("\n}\n");
    return try out.toOwnedSlice();
}

/// A stored report is valid only when it is the exact canonical rendering of
/// the bound manifest and decision. This recomputes every count and both
/// digests through the same canonical encoders used by `renderReport`.
pub fn validateReport(
    allocator: std.mem.Allocator,
    mode: accounting.CompilationMode,
    manifest: accounting.Manifest,
    result: Result,
    report: []const u8,
) !void {
    var parsed = try std.json.parseFromSlice(std.json.Value, allocator, report, .{});
    defer parsed.deinit();
    if (parsed.value != .object) return error.InvalidSourceAccountingReport;
    const expected = try renderReport(allocator, mode, manifest, result);
    defer allocator.free(expected);
    if (!std.mem.eql(u8, expected, report)) return error.SourceAccountingReportMismatch;
}

/// Render one stable human diagnostic from the structured failure row.  The
/// row, not this presentation string, remains the policy input and hashable
/// identity.
pub fn renderDiagnostic(allocator: std.mem.Allocator, failure: FailureRow) ![]u8 {
    var out = std.Io.Writer.Allocating.init(allocator);
    errdefer out.deinit();
    const writer = &out.writer;
    try writer.writeAll("formal source accounting rejected: ");
    try writer.writeAll(@tagName(failure.code));
    if (failure.source) |source| {
        try writer.writeAll(" at ");
        try writer.writeAll(source.file);
        try writer.print(":{d}-{d}", .{ source.start, source.end });
    }
    inline for (.{ "site_id", "expansion_id", "use_id", "handling_id", "evidence_id" }) |field| {
        if (@field(failure, field)) |value| {
            try writer.writeByte(' ');
            try writer.writeAll(field);
            try writer.print("={d}", .{value});
        }
    }
    return try out.toOwnedSlice();
}

fn writeCountBreakdowns(allocator: std.mem.Allocator, writer: anytype, manifest: accounting.Manifest, result: Result) !void {
    try writer.writeAll(",\"expansions_by_disposition\":{");
    inline for (std.meta.fields(accounting.ExpansionDisposition), 0..) |field, index| {
        if (index != 0) try writer.writeByte(',');
        const disposition: accounting.ExpansionDisposition = @enumFromInt(field.value);
        var count: usize = 0;
        for (manifest.inventory.expansions) |row| if (row.disposition == disposition) {
            count += 1;
        };
        try writeJsonString(writer, field.name);
        try writer.print(":{d}", .{count});
    }
    try writer.writeAll("},\"uses_by_role\":{");
    inline for (std.meta.fields(accounting.UseRole), 0..) |field, index| {
        if (index != 0) try writer.writeByte(',');
        const role: accounting.UseRole = @enumFromInt(field.value);
        var count: usize = 0;
        for (manifest.inventory.uses) |row| if (row.role == role) {
            count += 1;
        };
        try writeJsonString(writer, field.name);
        try writer.print(":{d}", .{count});
    }
    try writer.writeAll("},\"uses_by_fact_kind\":{");
    var site_kinds = std.AutoHashMap(accounting.SiteId, accounting.SourceFactKind).init(allocator);
    defer site_kinds.deinit();
    for (manifest.inventory.typed_sites) |site| if (!site_kinds.contains(site.id)) {
        try site_kinds.put(site.id, site.kind);
    };
    inline for (std.meta.fields(accounting.SourceFactKind), 0..) |field, index| {
        if (index != 0) try writer.writeByte(',');
        const kind: accounting.SourceFactKind = @enumFromInt(field.value);
        var count: usize = 0;
        for (manifest.inventory.uses) |use| {
            const use_kind = site_kinds.get(use.site_id) orelse continue;
            if (use_kind == kind) count += 1;
        }
        try writeJsonString(writer, field.name);
        try writer.print(":{d}", .{count});
    }
    try writer.writeAll("},\"handlings_by_kind\":{");
    inline for (std.meta.fields(accounting.HandlingKind), 0..) |field, index| {
        if (index != 0) try writer.writeByte(',');
        const kind: accounting.HandlingKind = @enumFromInt(field.value);
        var count: usize = 0;
        for (manifest.comptime_evidence.handlings) |row| if (row.kind == kind) {
            count += 1;
        };
        for (manifest.symbolic.handlings) |row| if (row.kind == kind) {
            count += 1;
        };
        try writeJsonString(writer, field.name);
        try writer.print(":{d}", .{count});
    }
    try writer.writeAll("},\"failures_by_code\":{");
    inline for (std.meta.fields(FailureCode), 0..) |field, index| {
        if (index != 0) try writer.writeByte(',');
        const code: FailureCode = @enumFromInt(field.value);
        var count: usize = 0;
        for (result.failures) |row| if (row.code == code) {
            count += 1;
        };
        try writeJsonString(writer, field.name);
        try writer.print(":{d}", .{count});
    }
    try writer.writeByte('}');
}

fn renderInventoryCanonical(allocator: std.mem.Allocator, inventory: accounting.SourceInventory) ![]u8 {
    var out = std.Io.Writer.Allocating.init(allocator);
    errdefer out.deinit();
    const writer = &out.writer;
    try writeDeclaredSites(writer, allocator, inventory.declared_sites);
    try writeTypedSites(writer, allocator, inventory.typed_sites);
    try writeGeneratedDerivations(writer, allocator, inventory.generated_fact_derivations);
    try writeTemplates(writer, allocator, inventory.owner_templates);
    try writeExpansions(writer, allocator, inventory.expansions);
    try writeUses(writer, allocator, inventory.uses);
    try writeControlNodes(writer, allocator, inventory.control_nodes);
    try writeControlEdges(writer, allocator, inventory.control_edges);
    return try out.toOwnedSlice();
}

fn renderEvidenceCanonical(allocator: std.mem.Allocator, concrete: accounting.ComptimeEvidence, symbolic: accounting.SymbolicEvidence) ![]u8 {
    var out = std.Io.Writer.Allocating.init(allocator);
    errdefer out.deinit();
    const writer = &out.writer;
    try writeFolds(writer, allocator, concrete.folds);
    try writePredicateEvents(writer, allocator, concrete.predicate_events);
    try writeCoveredEvidenceRows(writer, allocator, symbolic.obligations);
    try writeCoveredEvidenceRows(writer, allocator, symbolic.assumptions);
    try writeQueryEvidenceRows(writer, allocator, symbolic.queries);
    try writeCoveredEvidenceRows(writer, allocator, symbolic.runtime_checks);
    try writeValidationEvidenceRows(writer, allocator, symbolic.frame_results);
    try writeValidationEvidenceRows(writer, allocator, symbolic.state_effects);
    try writeAllHandlings(writer, allocator, concrete.handlings, symbolic.handlings);
    return try out.toOwnedSlice();
}

fn sha256Hex(bytes: []const u8) [std.crypto.hash.sha2.Sha256.digest_length * 2]u8 {
    var digest: [std.crypto.hash.sha2.Sha256.digest_length]u8 = undefined;
    std.crypto.hash.sha2.Sha256.hash(bytes, &digest, .{});
    return std.fmt.bytesToHex(digest, .lower);
}

fn sortedIndices(allocator: std.mem.Allocator, rows: anytype) ![]usize {
    const indices = try allocator.alloc(usize, rows.len);
    for (indices, 0..) |*index, value| index.* = value;
    const Rows = @TypeOf(rows);
    const Sort = struct {
        fn less(items: Rows, lhs: usize, rhs: usize) bool {
            const left = items[lhs].id;
            const right = items[rhs].id;
            return if (left != right) left < right else lhs < rhs;
        }
    };
    std.mem.sort(usize, indices, rows, Sort.less);
    return indices;
}

fn writeDeclaredSites(writer: anytype, allocator: std.mem.Allocator, rows: []const accounting.DeclaredSite) !void {
    const indices = try sortedIndices(allocator, rows);
    defer allocator.free(indices);
    try writer.writeByte('[');
    for (indices, 0..) |index, emitted| {
        if (emitted != 0) try writer.writeByte(',');
        const row = rows[index];
        try writer.print("{{\"id\":{d},\"path\":", .{row.id});
        try writeJsonString(writer, row.key.path);
        try writer.writeAll(",\"owner\":");
        try writeJsonString(writer, row.key.owner);
        try writer.print(",\"start\":{d},\"end\":{d},\"kind\":", .{ row.key.range_start, row.key.range_end });
        try writeJsonString(writer, @tagName(row.key.kind));
        try writer.print(",\"ordinal\":{d},\"label\":", .{row.key.ordinal});
        if (row.label) |label| try writeJsonString(writer, label) else try writer.writeAll("null");
        try writer.writeByte('}');
    }
    try writer.writeByte(']');
}

fn writeTypedSites(writer: anytype, allocator: std.mem.Allocator, rows: []const accounting.TypedSite) !void {
    const indices = try sortedIndices(allocator, rows);
    defer allocator.free(indices);
    try writer.writeByte('[');
    for (indices, 0..) |index, emitted| {
        if (emitted != 0) try writer.writeByte(',');
        const row = rows[index];
        try writer.print("{{\"id\":{d},\"origin\":", .{row.id});
        try writeJsonString(writer, @tagName(row.origin));
        try writer.writeAll(",\"kind\":");
        try writeJsonString(writer, @tagName(row.kind));
        try writer.writeAll(",\"source_fact_id\":");
        try writeOptionalId(writer, row.source_fact_id);
        try writer.writeAll(",\"declared_site_id\":");
        try writeOptionalId(writer, row.declared_site_id);
        try writer.writeAll(",\"derivation_id\":");
        try writeOptionalId(writer, row.derivation_id);
        try writer.writeAll(",\"path\":");
        try writeJsonString(writer, row.key.path);
        try writer.writeAll(",\"owner\":");
        try writeJsonString(writer, row.key.owner);
        try writer.print(",\"start\":{d},\"end\":{d},\"ordinal\":{d}", .{ row.key.range_start, row.key.range_end, row.key.ordinal });
        try writer.writeByte('}');
    }
    try writer.writeByte(']');
}

fn writeGeneratedDerivations(writer: anytype, allocator: std.mem.Allocator, rows: []const accounting.GeneratedFactDerivation) !void {
    const indices = try sortedIndices(allocator, rows);
    defer allocator.free(indices);
    try writer.writeByte('[');
    for (indices, 0..) |index, emitted| {
        if (emitted != 0) try writer.writeByte(',');
        const row = rows[index];
        try writer.print("{{\"id\":{d},\"site_id\":{d},\"semantic_rule\":", .{ row.id, row.site_id });
        try writeJsonString(writer, row.semantic_rule);
        try writer.writeAll(",\"parent_identity\":");
        try writeJsonString(writer, row.parent_identity);
        try writer.writeAll(",\"anchor\":{\"file\":");
        try writeJsonString(writer, row.anchor.file);
        try writer.print(",\"start\":{d},\"end\":{d}}}", .{ row.anchor.start, row.anchor.end });
        try writer.print(",\"ordinal\":{d}}}", .{row.ordinal});
    }
    try writer.writeByte(']');
}

fn writeTemplates(writer: anytype, allocator: std.mem.Allocator, rows: []const accounting.OwnerTemplate) !void {
    const indices = try sortedIndices(allocator, rows);
    defer allocator.free(indices);
    try writer.writeByte('[');
    for (indices, 0..) |index, emitted| {
        if (emitted != 0) try writer.writeByte(',');
        const row = rows[index];
        try writer.print("{{\"id\":{d},\"owner_key\":", .{row.id});
        try writeJsonString(writer, row.owner_key);
        try writer.writeAll(",\"activation\":");
        try writeJsonString(writer, @tagName(row.activation));
        try writer.writeAll(",\"uses\":[");
        for (row.uses, 0..) |use, use_index| {
            if (use_index != 0) try writer.writeByte(',');
            try writer.print("{{\"site_id\":{d},\"role\":", .{use.site_id});
            try writeJsonString(writer, @tagName(use.role));
            try writer.writeAll(",\"control_node_slot\":");
            try writeOptionalId(writer, use.control_node_slot);
            try writer.writeByte('}');
        }
        try writer.writeAll("],\"control_nodes\":[");
        for (row.control_nodes, 0..) |node, node_index| {
            if (node_index != 0) try writer.writeByte(',');
            try writer.print("{{\"slot\":{d},\"kind\":", .{node.slot});
            try writeJsonString(writer, @tagName(node.kind));
            try writer.writeAll(",\"range\":{\"file\":");
            try writeJsonString(writer, node.range.file);
            try writer.print(",\"start\":{d},\"end\":{d}}},\"attached_use_ordinals\":", .{ node.range.start, node.range.end });
            try writeIdList(writer, allocator, node.attached_use_ordinals);
            try writer.writeByte('}');
        }
        try writer.writeAll("],\"control_edges\":[");
        for (row.control_edges, 0..) |edge, edge_index| {
            if (edge_index != 0) try writer.writeByte(',');
            try writer.print("{{\"slot\":{d},\"from_slot\":{d},\"to_slot\":{d},\"kind\":", .{ edge.slot, edge.from_slot, edge.to_slot });
            try writeJsonString(writer, @tagName(edge.kind));
            try writer.writeByte('}');
        }
        try writer.writeAll("],\"entry_slot\":");
        try writeOptionalId(writer, row.entry_slot);
        try writer.writeAll(",\"terminal_slots\":");
        try writeIdList(writer, allocator, row.terminal_slots);
        try writer.writeByte('}');
    }
    try writer.writeByte(']');
}

fn writeExpansions(writer: anytype, allocator: std.mem.Allocator, rows: []const accounting.Expansion) !void {
    const indices = try sortedIndices(allocator, rows);
    defer allocator.free(indices);
    try writer.writeByte('[');
    for (indices, 0..) |index, emitted| {
        if (emitted != 0) try writer.writeByte(',');
        const row = rows[index];
        try writer.print("{{\"id\":{d},\"template_id\":{d},\"parent_expansion_id\":", .{ row.id, row.template_id });
        try writeOptionalId(writer, row.parent_expansion_id);
        try writer.writeAll(",\"replacement_expansion_id\":");
        try writeOptionalId(writer, row.replacement_expansion_id);
        try writer.writeAll(",\"activation\":");
        try writeJsonString(writer, @tagName(row.activation));
        try writer.writeAll(",\"disposition\":");
        try writeJsonString(writer, @tagName(row.disposition));
        try writer.writeAll(",\"root_runtime_owner\":");
        try writeJsonString(writer, row.root_runtime_owner);
        try writer.writeAll(",\"folded_call_site_chain\":[");
        for (row.folded_call_site_chain, 0..) |site, site_index| {
            if (site_index != 0) try writer.writeByte(',');
            try writer.writeAll("{\"file\":");
            try writeJsonString(writer, site.file);
            try writer.print(",\"start\":{d},\"end\":{d}}}", .{ site.start, site.end });
        }
        try writer.writeAll("],\"imported_module\":");
        try writeOptionalString(writer, row.imported_module);
        try writer.writeAll(",\"generic_bindings\":[");
        for (row.generic_bindings, 0..) |binding, binding_index| {
            if (binding_index != 0) try writer.writeByte(',');
            try writeJsonString(writer, binding);
        }
        try writer.writeAll("],\"trait_implementation\":");
        try writeOptionalString(writer, row.trait_implementation);
        try writer.writeAll(",\"trait_method\":");
        try writeOptionalString(writer, row.trait_method);
        try writer.writeAll(",\"identity\":");
        try writeJsonString(writer, row.identity);
        try writer.writeByte('}');
    }
    try writer.writeByte(']');
}

fn writeUses(writer: anytype, allocator: std.mem.Allocator, rows: []const accounting.SourceFactUse) !void {
    const indices = try sortedIndices(allocator, rows);
    defer allocator.free(indices);
    try writer.writeByte('[');
    for (indices, 0..) |index, emitted| {
        if (emitted != 0) try writer.writeByte(',');
        const row = rows[index];
        try writer.print("{{\"id\":{d},\"site_id\":{d},\"expansion_id\":{d},\"template_ordinal\":{d},\"role\":", .{ row.id, row.site_id, row.expansion_id, row.template_ordinal });
        try writeJsonString(writer, @tagName(row.role));
        try writer.writeAll(",\"control_node_id\":");
        try writeOptionalId(writer, row.control_node_id);
        try writer.writeByte('}');
    }
    try writer.writeByte(']');
}

fn writeControlNodes(writer: anytype, allocator: std.mem.Allocator, rows: []const accounting.ControlNode) !void {
    const indices = try sortedIndices(allocator, rows);
    defer allocator.free(indices);
    try writer.writeByte('[');
    for (indices, 0..) |index, emitted| {
        if (emitted != 0) try writer.writeByte(',');
        const row = rows[index];
        try writer.print("{{\"id\":{d},\"expansion_id\":{d},\"slot\":{d},\"kind\":", .{ row.id, row.expansion_id, row.slot });
        try writeJsonString(writer, @tagName(row.kind));
        try writer.writeAll(",\"attached_use_ids\":");
        try writeIdList(writer, allocator, row.attached_use_ids);
        try writer.writeAll(",\"range\":{\"file\":");
        try writeJsonString(writer, row.range.file);
        try writer.print(",\"start\":{d},\"end\":{d}}}", .{ row.range.start, row.range.end });
        try writer.writeByte('}');
    }
    try writer.writeByte(']');
}

fn writeControlEdges(writer: anytype, allocator: std.mem.Allocator, rows: []const accounting.ControlEdge) !void {
    const indices = try sortedIndices(allocator, rows);
    defer allocator.free(indices);
    try writer.writeByte('[');
    for (indices, 0..) |index, emitted| {
        if (emitted != 0) try writer.writeByte(',');
        const row = rows[index];
        try writer.print("{{\"id\":{d},\"expansion_id\":{d},\"from\":{d},\"to\":{d},\"kind\":", .{ row.id, row.expansion_id, row.from, row.to });
        try writeJsonString(writer, @tagName(row.kind));
        try writer.writeByte('}');
    }
    try writer.writeByte(']');
}

fn writeFolds(writer: anytype, allocator: std.mem.Allocator, rows: []const accounting.FoldRecord) !void {
    const indices = try sortedIndices(allocator, rows);
    defer allocator.free(indices);
    try writer.writeByte('[');
    for (indices, 0..) |index, emitted| {
        if (emitted != 0) try writer.writeByte(',');
        const row = rows[index];
        try writer.print("{{\"id\":{d},\"expansion_id\":{d},\"entry_node_id\":{d},\"terminal_node_id\":{d},\"disposition\":", .{ row.id, row.expansion_id, row.entry_node_id, row.terminal_node_id });
        try writeJsonString(writer, @tagName(row.disposition));
        try writer.writeAll(",\"events\":[");
        for (row.events, 0..) |event, event_index| {
            if (event_index != 0) try writer.writeByte(',');
            try writer.writeAll("{\"kind\":");
            try writeJsonString(writer, @tagName(event.kind));
            try writer.writeAll(",\"node_id\":");
            try writeOptionalId(writer, event.node_id);
            try writer.writeAll(",\"edge_id\":");
            try writeOptionalId(writer, event.edge_id);
            try writer.writeAll(",\"use_id\":");
            try writeOptionalId(writer, event.use_id);
            try writer.writeAll(",\"predicate_value\":");
            if (event.predicate_value) |value| try writer.writeAll(if (value) "true" else "false") else try writer.writeAll("null");
            try writer.writeByte('}');
        }
        try writer.writeAll("]}");
    }
    try writer.writeByte(']');
}

fn writePredicateEvents(writer: anytype, allocator: std.mem.Allocator, rows: []const accounting.PredicateEvent) !void {
    const indices = try sortedIndices(allocator, rows);
    defer allocator.free(indices);
    try writer.writeByte('[');
    for (indices, 0..) |index, emitted| {
        if (emitted != 0) try writer.writeByte(',');
        const row = rows[index];
        try writer.print("{{\"id\":{d},\"fold_id\":{d},\"use_id\":{d},\"node_id\":{d},\"value\":{s}}}", .{ row.id, row.fold_id, row.use_id, row.node_id, if (row.value) "true" else "false" });
    }
    try writer.writeByte(']');
}

fn writeCoveredEvidenceRows(writer: anytype, allocator: std.mem.Allocator, rows: []const accounting.CoveredEvidence) !void {
    const indices = try sortedIndices(allocator, rows);
    defer allocator.free(indices);
    try writer.writeByte('[');
    for (indices, 0..) |index, emitted| {
        if (emitted != 0) try writer.writeByte(',');
        const row = rows[index];
        try writer.print("{{\"id\":{d},\"producer_id\":{d},\"covered_use_ids\":", .{ row.id, row.producer_id });
        try writeIdList(writer, allocator, row.covered_use_ids);
        try writer.writeByte('}');
    }
    try writer.writeByte(']');
}

fn writeQueryEvidenceRows(writer: anytype, allocator: std.mem.Allocator, rows: []const accounting.QueryEvidence) !void {
    const indices = try sortedIndices(allocator, rows);
    defer allocator.free(indices);
    try writer.writeByte('[');
    for (indices, 0..) |index, emitted| {
        if (emitted != 0) try writer.writeByte(',');
        const row = rows[index];
        try writer.print("{{\"id\":{d},\"producer_id\":{d},\"kind\":", .{ row.id, row.producer_id });
        try writeJsonString(writer, @tagName(row.kind));
        try writer.writeAll(",\"covered_use_ids\":");
        try writeIdList(writer, allocator, row.covered_use_ids);
        try writer.writeByte('}');
    }
    try writer.writeByte(']');
}

fn writeValidationEvidenceRows(writer: anytype, allocator: std.mem.Allocator, rows: []const accounting.ValidationEvidence) !void {
    const indices = try sortedIndices(allocator, rows);
    defer allocator.free(indices);
    try writer.writeByte('[');
    for (indices, 0..) |index, emitted| {
        if (emitted != 0) try writer.writeByte(',');
        const row = rows[index];
        try writer.print("{{\"id\":{d},\"producer_id\":{d},\"covered_use_ids\":", .{ row.id, row.producer_id });
        try writeIdList(writer, allocator, row.covered_use_ids);
        try writer.print(",\"valid\":{s}}}", .{if (row.valid) "true" else "false"});
    }
    try writer.writeByte(']');
}

const HandlingRef = struct { source: u1, index: usize, id: accounting.HandlingId };

fn writeAllHandlings(writer: anytype, allocator: std.mem.Allocator, concrete: []const accounting.HandlingRecord, symbolic: []const accounting.HandlingRecord) !void {
    const refs = try allocator.alloc(HandlingRef, concrete.len + symbolic.len);
    defer allocator.free(refs);
    var cursor: usize = 0;
    for (concrete, 0..) |row, index| {
        refs[cursor] = .{ .source = 0, .index = index, .id = row.id };
        cursor += 1;
    }
    for (symbolic, 0..) |row, index| {
        refs[cursor] = .{ .source = 1, .index = index, .id = row.id };
        cursor += 1;
    }
    std.mem.sort(HandlingRef, refs, {}, struct {
        fn less(_: void, lhs: HandlingRef, rhs: HandlingRef) bool {
            if (lhs.id != rhs.id) return lhs.id < rhs.id;
            if (lhs.source != rhs.source) return lhs.source < rhs.source;
            return lhs.index < rhs.index;
        }
    }.less);
    try writer.writeByte('[');
    for (refs, 0..) |ref, emitted| {
        if (emitted != 0) try writer.writeByte(',');
        const row = if (ref.source == 0) concrete[ref.index] else symbolic[ref.index];
        try writer.print("{{\"id\":{d},\"use_id\":{d},\"kind\":", .{ row.id, row.use_id });
        try writeJsonString(writer, @tagName(row.kind));
        inline for (.{
            .{ "obligation_ids", "obligation_ids" },
            .{ "assumption_ids", "assumption_ids" },
            .{ "query_ids", "query_ids" },
            .{ "runtime_check_ids", "runtime_check_ids" },
            .{ "frame_result_ids", "frame_result_ids" },
            .{ "state_effect_ids", "state_effect_ids" },
            .{ "predicate_event_ids", "predicate_event_ids" },
        }) |fields| {
            try writer.writeAll(",\"");
            try writer.writeAll(fields[0]);
            try writer.writeAll("\":");
            try writeIdList(writer, allocator, @field(row, fields[1]));
        }
        try writer.writeAll(",\"fold_id\":");
        try writeOptionalId(writer, row.fold_id);
        try writer.writeAll(",\"control_event_index\":");
        try writeOptionalId(writer, row.control_event_index);
        try writer.writeAll(",\"rejection_reason\":");
        if (row.rejection_reason) |reason| try writeJsonString(writer, reason) else try writer.writeAll("null");
        try writer.writeByte('}');
    }
    try writer.writeByte(']');
}

fn writeFailures(writer: anytype, rows: []const FailureRow) !void {
    try writer.writeByte('[');
    for (rows, 0..) |row, index| {
        if (index != 0) try writer.writeByte(',');
        try writer.writeAll("{\"code\":");
        try writeJsonString(writer, @tagName(row.code));
        try writer.writeAll(",\"source\":");
        if (row.source) |source| {
            try writer.writeAll("{\"file\":");
            try writeJsonString(writer, source.file);
            try writer.print(",\"start\":{d},\"end\":{d}}}", .{ source.start, source.end });
        } else try writer.writeAll("null");
        inline for (.{ "site_id", "expansion_id", "use_id", "handling_id", "evidence_id" }) |field| {
            try writer.writeAll(",\"");
            try writer.writeAll(field);
            try writer.writeAll("\":");
            try writeOptionalId(writer, @field(row, field));
        }
        try writer.writeByte('}');
    }
    try writer.writeByte(']');
}

fn writeIdList(writer: anytype, allocator: std.mem.Allocator, ids: []const u32) !void {
    const canonical = try allocator.dupe(u32, ids);
    defer allocator.free(canonical);
    std.mem.sort(u32, canonical, {}, std.sort.asc(u32));
    try writer.writeByte('[');
    for (canonical, 0..) |id, index| {
        if (index != 0) try writer.writeByte(',');
        try writer.print("{d}", .{id});
    }
    try writer.writeByte(']');
}

fn writeOptionalId(writer: anytype, id: anytype) !void {
    if (id) |value| try writer.print("{d}", .{value}) else try writer.writeAll("null");
}

fn writeOptionalString(writer: anytype, value: ?[]const u8) !void {
    if (value) |text| try writeJsonString(writer, text) else try writer.writeAll("null");
}

fn writeJsonString(writer: anytype, value: []const u8) !void {
    try writer.writeByte('"');
    for (value) |byte| switch (byte) {
        '"' => try writer.writeAll("\\\""),
        '\\' => try writer.writeAll("\\\\"),
        '\n' => try writer.writeAll("\\n"),
        '\r' => try writer.writeAll("\\r"),
        '\t' => try writer.writeAll("\\t"),
        0...8, 11...12, 14...0x1f => try writer.print("\\u00{x:0>2}", .{byte}),
        else => try writer.writeByte(byte),
    };
    try writer.writeByte('"');
}

fn lessFailure(_: void, lhs: FailureRow, rhs: FailureRow) bool {
    const lp = failurePrecedence(lhs.code);
    const rp = failurePrecedence(rhs.code);
    if (lp != rp) return lp < rp;
    if (@intFromEnum(lhs.code) != @intFromEnum(rhs.code)) return @intFromEnum(lhs.code) < @intFromEnum(rhs.code);
    if (lhs.source == null and rhs.source != null) return true;
    if (lhs.source != null and rhs.source == null) return false;
    if (lhs.source != null and rhs.source != null) {
        const path_order = std.mem.order(u8, lhs.source.?.file, rhs.source.?.file);
        if (path_order != .eq) return path_order == .lt;
        if (lhs.source.?.start != rhs.source.?.start) return lhs.source.?.start < rhs.source.?.start;
        if (lhs.source.?.end != rhs.source.?.end) return lhs.source.?.end < rhs.source.?.end;
    }
    inline for (.{ "site_id", "expansion_id", "use_id", "handling_id", "evidence_id" }) |field| {
        const l = @field(lhs, field);
        const r = @field(rhs, field);
        if (l == null and r != null) return true;
        if (l != null and r == null) return false;
        if (l != null and r != null and l.? != r.?) return l.? < r.?;
    }
    return false;
}

fn failurePrecedence(code: FailureCode) u8 {
    return switch (code) {
        .invalid_control_graph,
        .invalid_fold_trace,
        .abandoned_fold_evidence,
        .invalid_control_elimination,
        .invalid_concrete_checkpoint,
        => 1,
        .missing_semantic_site,
        .missing_semantic_use,
        .missing_symbolic_replacement,
        .missing_handling,
        .missing_symbolic_obligation,
        .missing_symbolic_query,
        .missing_assumption_incorporation,
        .missing_runtime_enforcement,
        .missing_frame_validation,
        .missing_state_effect,
        .zero_concrete_checks,
        => 2,
        .handling_not_permitted,
        .reduced_scope_excluded_not_permitted,
        .verification_disabled_not_permitted,
        .rejected_handling,
        .false_concrete_target,
        .false_concrete_assumption,
        .invalid_frame_validation,
        .invalid_state_effect,
        => 3,
        else => 0,
    };
}

fn putUnique(
    ctx: *Context,
    map: anytype,
    id: anytype,
    value: anytype,
    row: FailureRow,
) !void {
    const entry = try map.getOrPut(id);
    if (entry.found_existing) {
        try ctx.fail(row);
    } else {
        entry.value_ptr.* = value;
    }
}

fn indexRows(ctx: *Context) !void {
    const inv = ctx.manifest.inventory;
    for (inv.declared_sites, 0..) |row, index| try putUnique(ctx, &ctx.declared, row.id, index, .{ .code = .duplicate_identity, .site_id = row.id });
    for (inv.typed_sites, 0..) |row, index| try putUnique(ctx, &ctx.typed, row.id, index, .{ .code = .duplicate_identity, .site_id = row.id });
    for (inv.generated_fact_derivations, 0..) |row, index| try putUnique(ctx, &ctx.derivations, row.id, index, .{ .code = .duplicate_identity, .site_id = row.site_id });
    for (inv.owner_templates, 0..) |row, index| {
        try putUnique(ctx, &ctx.templates, row.id, index, .{ .code = .duplicate_identity });
        for (row.uses) |use| {
            try ctx.template_site_roles.put(.{
                .template_id = row.id,
                .site_id = use.site_id,
                .role = use.role,
            }, {});
        }
        for (row.control_nodes) |node| {
            const node_entry = try ctx.template_node_slots.getOrPut(.{ .template_id = row.id, .slot = node.slot });
            if (node_entry.found_existing) try ctx.fail(.{ .code = .invalid_owner_template });
            for (node.attached_use_ordinals) |ordinal| {
                const attachment_entry = try ctx.template_attachments.getOrPut(.{
                    .template_id = row.id,
                    .node_slot = node.slot,
                    .use_ordinal = ordinal,
                });
                if (attachment_entry.found_existing) try ctx.fail(.{ .code = .invalid_owner_template });
            }
        }
        for (row.control_edges) |edge| {
            const edge_entry = try ctx.template_edge_shapes.getOrPut(.{
                .template_id = row.id,
                .from_slot = edge.from_slot,
                .to_slot = edge.to_slot,
                .kind = edge.kind,
            });
            if (edge_entry.found_existing) try ctx.fail(.{ .code = .invalid_owner_template });
        }
    }
    for (inv.expansions, 0..) |row, index| try putUnique(ctx, &ctx.expansions, row.id, index, .{ .code = .duplicate_identity, .expansion_id = row.id });
    for (inv.uses, 0..) |row, index| {
        try putUnique(ctx, &ctx.uses, row.id, index, .{ .code = .duplicate_identity, .use_id = row.id });
        try putUnique(ctx, &ctx.use_keys, UseKey{ .expansion_id = row.expansion_id, .template_ordinal = row.template_ordinal }, row.id, .{ .code = .unexpected_semantic_use, .expansion_id = row.expansion_id, .use_id = row.id });
    }
    for (inv.control_nodes, 0..) |row, index| {
        try putUnique(ctx, &ctx.nodes, row.id, index, .{ .code = .duplicate_identity, .expansion_id = row.expansion_id });
        try putUnique(ctx, &ctx.node_slots, NodeSlotKey{ .expansion_id = row.expansion_id, .slot = row.slot }, row.id, .{ .code = .invalid_control_graph, .expansion_id = row.expansion_id });
        for (row.attached_use_ids) |use_id| {
            const attached_entry = try ctx.node_attached_uses.getOrPut(.{ .node_id = row.id, .use_id = use_id });
            if (attached_entry.found_existing) try ctx.fail(.{ .code = .invalid_control_graph, .expansion_id = row.expansion_id, .use_id = use_id });
        }
    }
    for (inv.control_edges, 0..) |row, index| {
        try putUnique(ctx, &ctx.edges, row.id, index, .{ .code = .duplicate_identity, .expansion_id = row.expansion_id });
        try putUnique(ctx, &ctx.edge_shapes, EdgeShapeKey{
            .expansion_id = row.expansion_id,
            .from = row.from,
            .to = row.to,
            .kind = row.kind,
        }, row.id, .{ .code = .invalid_control_graph, .expansion_id = row.expansion_id });
    }
    for (ctx.manifest.comptime_evidence.folds, 0..) |row, index| try putUnique(ctx, &ctx.folds, row.id, index, .{ .code = .duplicate_identity, .expansion_id = row.expansion_id });

    for (ctx.manifest.comptime_evidence.predicate_events, 0..) |row, index| {
        try indexEvidence(ctx, row.id, .{ .predicate = index });
        const key: PredicateKey = .{
            .fold_id = row.fold_id,
            .use_id = row.use_id,
            .node_id = row.node_id,
            .value = row.value,
        };
        try incrementCount(&ctx.predicate_rows, key);
        if (!ctx.predicate_first_ids.contains(key)) try ctx.predicate_first_ids.put(key, row.id);
        try indexEvidenceCoverage(ctx, row.id, &.{row.use_id});
    }
    for (ctx.manifest.symbolic.obligations, 0..) |row, index| {
        try indexEvidence(ctx, row.id, .{ .obligation = index });
        try indexEvidenceCoverage(ctx, row.id, row.covered_use_ids);
    }
    for (ctx.manifest.symbolic.assumptions, 0..) |row, index| {
        try indexEvidence(ctx, row.id, .{ .assumption = index });
        try indexEvidenceCoverage(ctx, row.id, row.covered_use_ids);
    }
    for (ctx.manifest.symbolic.queries, 0..) |row, index| {
        try indexEvidence(ctx, row.id, .{ .query = index });
        try indexEvidenceCoverage(ctx, row.id, row.covered_use_ids);
    }
    for (ctx.manifest.symbolic.runtime_checks, 0..) |row, index| {
        try indexEvidence(ctx, row.id, .{ .runtime_check = index });
        try indexEvidenceCoverage(ctx, row.id, row.covered_use_ids);
    }
    for (ctx.manifest.symbolic.frame_results, 0..) |row, index| {
        try indexEvidence(ctx, row.id, .{ .frame_result = index });
        try indexEvidenceCoverage(ctx, row.id, row.covered_use_ids);
    }
    for (ctx.manifest.symbolic.state_effects, 0..) |row, index| {
        try indexEvidence(ctx, row.id, .{ .state_effect = index });
        try indexEvidenceCoverage(ctx, row.id, row.covered_use_ids);
    }

    for (ctx.manifest.comptime_evidence.handlings, 0..) |row, index| try indexHandling(ctx, row, .{ .concrete = index });
    for (ctx.manifest.symbolic.handlings, 0..) |row, index| try indexHandling(ctx, row, .{ .symbolic = index });
}

fn incrementCount(map: anytype, key: anytype) !void {
    const entry = try map.getOrPut(key);
    if (!entry.found_existing) entry.value_ptr.* = 0;
    entry.value_ptr.* +|= 1;
}

fn indexEvidence(ctx: *Context, id: accounting.EvidenceId, location: Context.EvidenceLocation) !void {
    try putUnique(ctx, &ctx.evidence, id, location, .{ .code = .duplicate_identity, .evidence_id = id });
    if (!ctx.evidence_refs.contains(id)) try ctx.evidence_refs.put(id, 0);
}

fn indexEvidenceCoverage(ctx: *Context, evidence_id: accounting.EvidenceId, use_ids: []const accounting.UseId) !void {
    for (use_ids) |use_id| {
        const entry = try ctx.evidence_coverage.getOrPut(.{ .evidence_id = evidence_id, .use_id = use_id });
        if (entry.found_existing) try ctx.fail(.{ .code = .evidence_coverage_mismatch, .use_id = use_id, .evidence_id = evidence_id });
    }
}

fn indexHandling(ctx: *Context, row: accounting.HandlingRecord, location: Context.HandlingLocation) !void {
    try putUnique(ctx, &ctx.handlings, row.id, location, .{ .code = .duplicate_identity, .use_id = row.use_id, .handling_id = row.id });
    const use_entry = try ctx.handling_by_use.getOrPut(row.use_id);
    if (use_entry.found_existing) {
        try ctx.fail(.{ .code = .duplicate_handling, .use_id = row.use_id, .handling_id = row.id });
    } else {
        use_entry.value_ptr.* = row.id;
    }
    inline for (.{
        "obligation_ids",
        "assumption_ids",
        "query_ids",
        "runtime_check_ids",
        "frame_result_ids",
        "state_effect_ids",
        "predicate_event_ids",
    }) |field| for (@field(row, field)) |evidence_id| {
        const reference_entry = try ctx.handling_evidence_refs.getOrPut(.{
            .evidence_id = evidence_id,
            .use_id = row.use_id,
        });
        if (reference_entry.found_existing) try ctx.fail(.{
            .code = .evidence_coverage_mismatch,
            .use_id = row.use_id,
            .handling_id = row.id,
            .evidence_id = evidence_id,
        });
    };
}

fn validateSites(ctx: *Context) !void {
    const declared_indices = try sortedSiteIndices(ctx.allocator, ctx.manifest.inventory.declared_sites);
    defer ctx.allocator.free(declared_indices);
    for (declared_indices, 0..) |index, canonical_index| {
        const row = ctx.manifest.inventory.declared_sites[index];
        if (row.id != @as(accounting.SiteId, @intCast(canonical_index + 1))) {
            try ctx.fail(.{ .code = .identity_hash_collision, .site_id = row.id });
        }
    }
    if (declared_indices.len > 1) for (declared_indices[1..], 1..) |index, offset| {
        const previous = ctx.manifest.inventory.declared_sites[declared_indices[offset - 1]];
        const current = ctx.manifest.inventory.declared_sites[index];
        if (siteKeyEqual(previous.key, current.key) and previous.id != current.id) {
            try ctx.fail(.{ .code = .identity_hash_collision, .site_id = current.id });
        }
    };
    for (ctx.manifest.inventory.declared_sites) |declared| {
        const typed_index = ctx.typed.get(declared.id) orelse {
            try ctx.fail(.{ .code = .missing_semantic_site, .site_id = declared.id });
            continue;
        };
        const typed = ctx.manifest.inventory.typed_sites[typed_index];
        if (typed.origin != .source_syntax or typed.declared_site_id != declared.id or
            typed.source_fact_id != declared.key.range_start or
            typed.kind != declared.key.kind or !siteKeyEqual(typed.key, declared.key))
        {
            try ctx.fail(.{ .code = .unknown_semantic_site, .site_id = typed.id });
        }
    }
    for (ctx.manifest.inventory.typed_sites) |typed| switch (typed.origin) {
        .source_syntax => {
            if (typed.source_fact_id == null or typed.source_fact_id.? != typed.key.range_start or
                typed.declared_site_id == null or ctx.declared.get(typed.declared_site_id.?) == null)
            {
                try ctx.fail(.{ .code = .unknown_semantic_site, .site_id = typed.id });
            }
            if (typed.derivation_id != null) try ctx.fail(.{ .code = .invalid_generated_fact_derivation, .site_id = typed.id });
        },
        .semantic_generated => {
            if (typed.source_fact_id != null or typed.declared_site_id != null or typed.derivation_id == null) {
                try ctx.fail(.{ .code = .invalid_generated_fact_derivation, .site_id = typed.id });
                continue;
            }
            const derivation_index = ctx.derivations.get(typed.derivation_id.?) orelse {
                try ctx.fail(.{ .code = .invalid_generated_fact_derivation, .site_id = typed.id });
                continue;
            };
            if (ctx.manifest.inventory.generated_fact_derivations[derivation_index].site_id != typed.id) {
                try ctx.fail(.{ .code = .invalid_generated_fact_derivation, .site_id = typed.id });
            }
        },
    };
    for (ctx.manifest.inventory.generated_fact_derivations) |row| {
        const site = ctx.typedSite(row.site_id) orelse {
            try ctx.fail(.{ .code = .unknown_site, .site_id = row.site_id });
            continue;
        };
        if (site.origin != .semantic_generated or site.derivation_id != row.id or
            row.semantic_rule.len == 0 or row.parent_identity.len == 0)
        {
            try ctx.fail(.{ .code = .invalid_generated_fact_derivation, .site_id = row.site_id });
        }
    }
}

fn sortedSiteIndices(allocator: std.mem.Allocator, rows: []const accounting.DeclaredSite) ![]usize {
    const indices = try allocator.alloc(usize, rows.len);
    for (indices, 0..) |*index, value| index.* = value;
    const SortContext = struct {
        rows: []const accounting.DeclaredSite,
        fn less(self: @This(), lhs: usize, rhs: usize) bool {
            const left = self.rows[lhs].key;
            const right = self.rows[rhs].key;
            const path_order = std.mem.order(u8, left.path, right.path);
            if (path_order != .eq) return path_order == .lt;
            const owner_order = std.mem.order(u8, left.owner, right.owner);
            if (owner_order != .eq) return owner_order == .lt;
            if (left.range_start != right.range_start) return left.range_start < right.range_start;
            if (left.range_end != right.range_end) return left.range_end < right.range_end;
            if (left.kind != right.kind) return @intFromEnum(left.kind) < @intFromEnum(right.kind);
            if (left.ordinal != right.ordinal) return left.ordinal < right.ordinal;
            return self.rows[lhs].id < self.rows[rhs].id;
        }
    };
    std.mem.sort(usize, indices, SortContext{ .rows = rows }, SortContext.less);
    return indices;
}

fn siteKeyEqual(lhs: accounting.SiteKey, rhs: accounting.SiteKey) bool {
    return std.mem.eql(u8, lhs.path, rhs.path) and
        std.mem.eql(u8, lhs.owner, rhs.owner) and
        lhs.range_start == rhs.range_start and lhs.range_end == rhs.range_end and
        lhs.kind == rhs.kind and lhs.ordinal == rhs.ordinal;
}

fn validateTemplatesExpansionsAndUses(ctx: *Context) !void {
    for (ctx.manifest.inventory.owner_templates) |template| {
        try validateOwnerControlTemplate(ctx, template);
        var seen_sites = std.AutoHashMap(accounting.SiteId, void).init(ctx.allocator);
        defer seen_sites.deinit();
        for (template.uses) |use_template| {
            const site = ctx.typedSite(use_template.site_id) orelse {
                try ctx.fail(.{ .code = .unknown_site, .site_id = use_template.site_id });
                continue;
            };
            if (!accounting.roleRequired(site.kind, use_template.role)) {
                try ctx.fail(.{ .code = .invalid_site_role, .site_id = site.id });
            }
            try seen_sites.put(site.id, {});
        }
        var site_it = seen_sites.keyIterator();
        while (site_it.next()) |site_id| {
            const site = ctx.typedSite(site_id.*).?;
            inline for (std.meta.fields(accounting.UseRole)) |role_field| {
                const role: accounting.UseRole = @enumFromInt(role_field.value);
                var expected_count: usize = 0;
                for (accounting.templateRoles(site.kind, template.activation)) |expected| {
                    expected_count += @intFromBool(expected == role);
                }
                var actual_count: usize = 0;
                for (template.uses) |actual| {
                    actual_count += @intFromBool(actual.site_id == site.id and actual.role == role);
                }
                if (actual_count != expected_count) {
                    try ctx.fail(.{ .code = .invalid_owner_template, .site_id = site.id });
                }
            }
        }
    }

    for (ctx.manifest.inventory.typed_sites) |site| {
        var represented = false;
        for (ctx.manifest.inventory.owner_templates) |template| {
            for (template.uses) |use| if (use.site_id == site.id) {
                represented = true;
                break;
            };
            if (represented) break;
        }
        if (!represented) try ctx.fail(.{ .code = .invalid_owner_template, .site_id = site.id });
    }

    for (ctx.manifest.inventory.expansions) |expansion| {
        const template_index = ctx.templates.get(expansion.template_id) orelse {
            try ctx.fail(.{ .code = .invalid_owner_template, .expansion_id = expansion.id });
            continue;
        };
        const template = ctx.manifest.inventory.owner_templates[template_index];
        if (expansion.identity.len == 0 or expansion.root_runtime_owner.len == 0 or
            (expansion.imported_module != null and expansion.imported_module.?.len == 0) or
            (expansion.trait_implementation == null) != (expansion.trait_method == null) or
            template.activation != accounting.templateActivation(expansion.activation) or
            !activationDispositionValid(expansion.activation, expansion.disposition))
        {
            try ctx.fail(.{ .code = .invalid_expansion_activation, .expansion_id = expansion.id });
        }
        var previous_binding: ?[]const u8 = null;
        for (expansion.generic_bindings) |binding| {
            if (binding.len == 0 or (previous_binding != null and std.mem.order(u8, previous_binding.?, binding) != .lt)) {
                try ctx.fail(.{ .code = .invalid_expansion_activation, .expansion_id = expansion.id });
            }
            previous_binding = binding;
        }
        for (expansion.folded_call_site_chain) |call_site| {
            if (call_site.file.len == 0 or call_site.end < call_site.start) {
                try ctx.fail(.{ .code = .invalid_expansion_activation, .expansion_id = expansion.id });
            }
        }
        if (expansion.activation == .runtime_owner and expansion.folded_call_site_chain.len != 0) {
            try ctx.fail(.{ .code = .invalid_expansion_activation, .expansion_id = expansion.id });
        }
        if (expansion.parent_expansion_id) |parent| {
            const parent_row = ctx.expansion(parent);
            if (parent == expansion.id or parent_row == null or
                (parent_row != null and !std.mem.eql(u8, parent_row.?.root_runtime_owner, expansion.root_runtime_owner)) or
                (parent_row != null and !callChainHasPrefix(expansion.folded_call_site_chain, parent_row.?.folded_call_site_chain)))
            {
                try ctx.fail(.{ .code = .invalid_expansion_parent, .expansion_id = expansion.id });
            }
        }
        if (expansion.disposition == .fold_abandoned_to_symbolic) {
            const replacement_id = expansion.replacement_expansion_id orelse {
                try ctx.fail(.{ .code = .missing_symbolic_replacement, .expansion_id = expansion.id });
                continue;
            };
            const replacement = ctx.expansion(replacement_id) orelse {
                try ctx.fail(.{ .code = .missing_symbolic_replacement, .expansion_id = expansion.id });
                continue;
            };
            if (replacement.disposition != .symbolic or !sameExpansionInstance(expansion, replacement)) {
                try ctx.fail(.{ .code = .missing_symbolic_replacement, .expansion_id = expansion.id });
            }
        } else if (expansion.replacement_expansion_id != null) {
            try ctx.fail(.{ .code = .invalid_expansion_disposition, .expansion_id = expansion.id });
        }
        if (expansion.activation == .required_comptime and expansion.disposition == .fold_abandoned_to_symbolic) {
            try ctx.fail(.{ .code = .invalid_expansion_disposition, .expansion_id = expansion.id });
        }
        if (expansion.disposition == .rejected) {
            try ctx.fail(.{ .code = .invalid_expansion_disposition, .expansion_id = expansion.id });
        }

        if (expansion.disposition == .fold_abandoned_to_symbolic or expansion.disposition == .rejected) {
            for (ctx.manifest.inventory.uses) |use| if (use.expansion_id == expansion.id) {
                try ctx.fail(.{ .code = .unexpected_semantic_use, .site_id = use.site_id, .expansion_id = expansion.id, .use_id = use.id });
            };
            for (ctx.manifest.inventory.control_nodes) |node| if (node.expansion_id == expansion.id) {
                try ctx.fail(.{ .code = .invalid_control_graph, .expansion_id = expansion.id });
            };
            for (ctx.manifest.inventory.control_edges) |edge| if (edge.expansion_id == expansion.id) {
                try ctx.fail(.{ .code = .invalid_control_graph, .expansion_id = expansion.id });
            };
            continue;
        }

        for (template.uses, 0..) |expected, ordinal| {
            const use_id = ctx.use_keys.get(.{ .expansion_id = expansion.id, .template_ordinal = @intCast(ordinal) }) orelse {
                try ctx.fail(.{ .code = .missing_semantic_use, .site_id = expected.site_id, .expansion_id = expansion.id });
                continue;
            };
            const use = ctx.use(use_id).?;
            if (use.site_id != expected.site_id or use.role != expected.role) {
                try ctx.fail(.{ .code = .unexpected_semantic_use, .site_id = use.site_id, .expansion_id = expansion.id, .use_id = use.id });
            }
            if (expected.control_node_slot) |slot| {
                const expected_node = ctx.node_slots.get(.{ .expansion_id = expansion.id, .slot = slot });
                if (expected_node == null or use.control_node_id != expected_node.?) {
                    try ctx.fail(.{ .code = .invalid_control_graph, .site_id = use.site_id, .expansion_id = expansion.id, .use_id = use.id });
                }
            } else if (use.control_node_id != null) {
                try ctx.fail(.{ .code = .invalid_control_graph, .site_id = use.site_id, .expansion_id = expansion.id, .use_id = use.id });
            }
        }
        for (template.control_nodes) |expected_node| {
            const node_id = ctx.node_slots.get(.{ .expansion_id = expansion.id, .slot = expected_node.slot }) orelse {
                try ctx.fail(.{ .code = .invalid_control_graph, .expansion_id = expansion.id });
                continue;
            };
            const actual = ctx.manifest.inventory.control_nodes[ctx.nodes.get(node_id).?];
            if (actual.kind != expected_node.kind or !rangeEqual(actual.range, expected_node.range)) {
                try ctx.fail(.{ .code = .invalid_control_graph, .expansion_id = expansion.id });
            }
            if (actual.attached_use_ids.len != expected_node.attached_use_ordinals.len) {
                try ctx.fail(.{ .code = .invalid_control_graph, .expansion_id = expansion.id });
            } else for (expected_node.attached_use_ordinals) |ordinal| {
                const expected_use_id = ctx.use_keys.get(.{
                    .expansion_id = expansion.id,
                    .template_ordinal = ordinal,
                }) orelse continue;
                if (!ctx.node_attached_uses.contains(.{
                    .node_id = actual.id,
                    .use_id = expected_use_id,
                })) try ctx.fail(.{ .code = .invalid_control_graph, .expansion_id = expansion.id, .use_id = expected_use_id });
            }
        }
        for (template.control_edges) |expected_edge| {
            const from = ctx.node_slots.get(.{ .expansion_id = expansion.id, .slot = expected_edge.from_slot }) orelse continue;
            const to = ctx.node_slots.get(.{ .expansion_id = expansion.id, .slot = expected_edge.to_slot }) orelse continue;
            if (!ctx.edge_shapes.contains(EdgeShapeKey{
                .expansion_id = expansion.id,
                .from = from,
                .to = to,
                .kind = expected_edge.kind,
            })) try ctx.fail(.{ .code = .invalid_control_graph, .expansion_id = expansion.id });
        }
    }

    for (ctx.manifest.inventory.uses) |use| {
        const expansion = ctx.expansion(use.expansion_id) orelse {
            try ctx.fail(.{ .code = .unknown_expansion, .site_id = use.site_id, .expansion_id = use.expansion_id, .use_id = use.id });
            continue;
        };
        const template_index = ctx.templates.get(expansion.template_id) orelse continue;
        const template = ctx.manifest.inventory.owner_templates[template_index];
        if (expansion.disposition == .fold_abandoned_to_symbolic or
            expansion.disposition == .rejected or use.template_ordinal >= template.uses.len)
        {
            try ctx.fail(.{ .code = .unexpected_semantic_use, .site_id = use.site_id, .expansion_id = use.expansion_id, .use_id = use.id });
        }
    }
    for (ctx.manifest.inventory.control_nodes) |node| {
        const expansion = ctx.expansion(node.expansion_id) orelse continue;
        if (!ctx.template_node_slots.contains(.{
            .template_id = expansion.template_id,
            .slot = node.slot,
        })) {
            try ctx.fail(.{ .code = .invalid_control_graph, .expansion_id = node.expansion_id });
        }
    }
    for (ctx.manifest.inventory.control_edges) |edge| {
        const expansion = ctx.expansion(edge.expansion_id) orelse continue;
        const from = ctx.manifest.inventory.control_nodes[ctx.nodes.get(edge.from) orelse continue];
        const to = ctx.manifest.inventory.control_nodes[ctx.nodes.get(edge.to) orelse continue];
        if (!ctx.template_edge_shapes.contains(.{
            .template_id = expansion.template_id,
            .from_slot = from.slot,
            .to_slot = to.slot,
            .kind = edge.kind,
        })) {
            try ctx.fail(.{ .code = .invalid_control_graph, .expansion_id = edge.expansion_id });
        }
    }
}

fn callChainHasPrefix(chain: []const accounting.SourceRange, prefix: []const accounting.SourceRange) bool {
    if (chain.len < prefix.len) return false;
    for (chain[0..prefix.len], prefix) |actual, expected| if (!rangeEqual(actual, expected)) return false;
    return true;
}

fn activationDispositionValid(
    activation: accounting.ActivationReason,
    disposition: accounting.ExpansionDisposition,
) bool {
    return switch (activation) {
        .runtime_owner, .symbolic_call => disposition == .symbolic or disposition == .rejected,
        .speculative_fold => disposition == .fold_committed or
            disposition == .fold_abandoned_to_symbolic or disposition == .rejected,
        .required_comptime => disposition == .fold_committed or disposition == .rejected,
    };
}

fn sameExpansionInstance(lhs: accounting.Expansion, rhs: accounting.Expansion) bool {
    if (lhs.parent_expansion_id != rhs.parent_expansion_id or
        !std.mem.eql(u8, lhs.root_runtime_owner, rhs.root_runtime_owner) or
        !std.mem.eql(u8, lhs.identity, rhs.identity) or
        lhs.folded_call_site_chain.len != rhs.folded_call_site_chain.len or
        lhs.generic_bindings.len != rhs.generic_bindings.len or
        !optionalStringEqual(lhs.imported_module, rhs.imported_module) or
        !optionalStringEqual(lhs.trait_implementation, rhs.trait_implementation) or
        !optionalStringEqual(lhs.trait_method, rhs.trait_method)) return false;
    for (lhs.folded_call_site_chain, rhs.folded_call_site_chain) |left, right| if (!rangeEqual(left, right)) return false;
    for (lhs.generic_bindings, rhs.generic_bindings) |left, right| if (!std.mem.eql(u8, left, right)) return false;
    return true;
}

fn optionalStringEqual(lhs: ?[]const u8, rhs: ?[]const u8) bool {
    if ((lhs == null) != (rhs == null)) return false;
    return lhs == null or std.mem.eql(u8, lhs.?, rhs.?);
}

fn validateOwnerControlTemplate(ctx: *Context, template: accounting.OwnerTemplate) !void {
    if (template.control_nodes.len == 0) {
        if (template.control_edges.len != 0 or template.entry_slot != null or template.terminal_slots.len != 0) {
            try ctx.fail(.{ .code = .invalid_owner_template });
        }
        for (template.uses) |use| if (use.control_node_slot != null) {
            try ctx.fail(.{ .code = .invalid_owner_template, .site_id = use.site_id });
        };
        return;
    }

    var slots = std.AutoHashMap(u32, usize).init(ctx.allocator);
    defer slots.deinit();
    var edge_slots = std.AutoHashMap(u32, void).init(ctx.allocator);
    defer edge_slots.deinit();
    var terminal_slots = std.AutoHashMap(u32, void).init(ctx.allocator);
    defer terminal_slots.deinit();
    for (template.control_nodes, 0..) |node, index| {
        const entry = try slots.getOrPut(node.slot);
        if (entry.found_existing or node.range.end < node.range.start) {
            try ctx.fail(.{ .code = .invalid_owner_template });
        } else entry.value_ptr.* = index;
        for (node.attached_use_ordinals) |ordinal| {
            if (ordinal >= template.uses.len or template.uses[ordinal].control_node_slot != node.slot) {
                try ctx.fail(.{ .code = .invalid_owner_template });
            }
        }
    }
    const entry_slot = template.entry_slot orelse {
        try ctx.fail(.{ .code = .invalid_owner_template });
        return;
    };
    const entry_index = slots.get(entry_slot) orelse {
        try ctx.fail(.{ .code = .invalid_owner_template });
        return;
    };
    if (template.control_nodes[entry_index].kind != .entry or template.terminal_slots.len == 0) {
        try ctx.fail(.{ .code = .invalid_owner_template });
    }
    for (template.terminal_slots) |slot| {
        const terminal_entry = try terminal_slots.getOrPut(slot);
        if (terminal_entry.found_existing) try ctx.fail(.{ .code = .invalid_owner_template });
        const index = slots.get(slot) orelse {
            try ctx.fail(.{ .code = .invalid_owner_template });
            continue;
        };
        const kind = template.control_nodes[index].kind;
        if (kind != .success_exit and kind != .error_exit and kind != .return_exit) {
            try ctx.fail(.{ .code = .invalid_owner_template });
        }
    }
    for (template.control_edges) |edge| {
        const slot_entry = try edge_slots.getOrPut(edge.slot);
        if (slot_entry.found_existing or !slots.contains(edge.from_slot) or !slots.contains(edge.to_slot)) {
            try ctx.fail(.{ .code = .invalid_owner_template });
        }
    }
    for (template.uses, 0..) |use, ordinal| if (use.control_node_slot) |slot| {
        _ = slots.get(slot) orelse {
            try ctx.fail(.{ .code = .invalid_owner_template, .site_id = use.site_id });
            continue;
        };
        if (!ctx.template_attachments.contains(.{
            .template_id = template.id,
            .node_slot = slot,
            .use_ordinal = @intCast(ordinal),
        })) try ctx.fail(.{ .code = .invalid_owner_template, .site_id = use.site_id });
    };
}

fn rangeEqual(lhs: accounting.SourceRange, rhs: accounting.SourceRange) bool {
    return std.mem.eql(u8, lhs.file, rhs.file) and lhs.start == rhs.start and lhs.end == rhs.end;
}

fn validateControlGraphAndFolds(ctx: *Context) !void {
    for (ctx.manifest.inventory.control_nodes) |node| {
        if (ctx.expansions.get(node.expansion_id) == null or node.range.end < node.range.start) {
            try ctx.fail(.{ .code = .invalid_control_graph, .expansion_id = node.expansion_id });
        }
        for (node.attached_use_ids) |use_id| {
            const use = ctx.use(use_id) orelse {
                try ctx.fail(.{ .code = .unknown_use, .expansion_id = node.expansion_id, .use_id = use_id });
                continue;
            };
            if (use.expansion_id != node.expansion_id or use.control_node_id != node.id) {
                try ctx.fail(.{ .code = .invalid_control_graph, .expansion_id = node.expansion_id, .use_id = use.id });
            }
        }
    }
    for (ctx.manifest.inventory.control_edges) |edge| {
        const from_index = ctx.nodes.get(edge.from) orelse {
            try ctx.fail(.{ .code = .invalid_control_graph, .expansion_id = edge.expansion_id });
            continue;
        };
        const to_index = ctx.nodes.get(edge.to) orelse {
            try ctx.fail(.{ .code = .invalid_control_graph, .expansion_id = edge.expansion_id });
            continue;
        };
        if (ctx.manifest.inventory.control_nodes[from_index].expansion_id != edge.expansion_id or
            ctx.manifest.inventory.control_nodes[to_index].expansion_id != edge.expansion_id)
        {
            try ctx.fail(.{ .code = .invalid_control_graph, .expansion_id = edge.expansion_id });
        }
    }
    for (ctx.manifest.comptime_evidence.folds) |fold| try validateFoldTrace(ctx, fold);
    for (ctx.manifest.comptime_evidence.predicate_events) |predicate| {
        const fold = ctx.fold(predicate.fold_id) orelse {
            try ctx.fail(.{ .code = .unknown_evidence_reference, .use_id = predicate.use_id, .evidence_id = predicate.id });
            continue;
        };
        const use = ctx.use(predicate.use_id) orelse {
            try ctx.fail(.{ .code = .unknown_use, .expansion_id = fold.expansion_id, .use_id = predicate.use_id, .evidence_id = predicate.id });
            continue;
        };
        if (fold.disposition != .committed or use.expansion_id != fold.expansion_id or
            (use.control_node_id != null and use.control_node_id != predicate.node_id))
        {
            try ctx.fail(.{ .code = .invalid_concrete_checkpoint, .site_id = use.site_id, .expansion_id = use.expansion_id, .use_id = use.id, .evidence_id = predicate.id });
        }
    }
    var produced = ctx.predicate_rows.iterator();
    while (produced.next()) |entry| {
        if ((ctx.predicate_traces.get(entry.key_ptr.*) orelse 0) != entry.value_ptr.*) {
            try ctx.fail(.{
                .code = .invalid_concrete_checkpoint,
                .use_id = entry.key_ptr.use_id,
                .evidence_id = ctx.predicate_first_ids.get(entry.key_ptr.*),
            });
        }
    }
    var traced = ctx.predicate_traces.iterator();
    while (traced.next()) |entry| {
        if ((ctx.predicate_rows.get(entry.key_ptr.*) orelse 0) != entry.value_ptr.*) {
            try ctx.fail(.{ .code = .invalid_concrete_checkpoint, .use_id = entry.key_ptr.use_id });
        }
    }
}

fn validateFoldTrace(ctx: *Context, fold: accounting.FoldRecord) !void {
    const expansion = ctx.expansion(fold.expansion_id) orelse {
        try ctx.fail(.{ .code = .unknown_expansion, .expansion_id = fold.expansion_id });
        return;
    };
    if (fold.disposition == .committed and expansion.disposition != .fold_committed) {
        try ctx.fail(.{ .code = .invalid_expansion_disposition, .expansion_id = fold.expansion_id });
    }
    const entry_index = ctx.nodes.get(fold.entry_node_id) orelse {
        try ctx.fail(.{ .code = .invalid_fold_trace, .expansion_id = fold.expansion_id });
        return;
    };
    const terminal_index = ctx.nodes.get(fold.terminal_node_id) orelse {
        try ctx.fail(.{ .code = .invalid_fold_trace, .expansion_id = fold.expansion_id });
        return;
    };
    if (ctx.manifest.inventory.control_nodes[entry_index].expansion_id != fold.expansion_id or
        ctx.manifest.inventory.control_nodes[terminal_index].expansion_id != fold.expansion_id)
    {
        try ctx.fail(.{ .code = .invalid_fold_trace, .expansion_id = fold.expansion_id });
        return;
    }
    if (fold.disposition == .abandoned) {
        if (fold.events.len != 0) try ctx.fail(.{ .code = .abandoned_fold_evidence, .expansion_id = fold.expansion_id });
        return;
    }
    if (fold.events.len == 0 or fold.events[0].kind != .enter_node or fold.events[0].node_id != fold.entry_node_id) {
        try ctx.fail(.{ .code = .invalid_fold_trace, .expansion_id = fold.expansion_id });
        return;
    }
    var current = fold.entry_node_id;
    try ctx.fold_visited_nodes.put(.{ .fold_id = fold.id, .node_id = current }, {});
    var pending_node: ?accounting.ControlNodeId = null;
    var required_edge_kind: ?accounting.ControlEdgeKind = null;
    var checks_this_visit = std.AutoHashMap(accounting.UseId, bool).init(ctx.allocator);
    defer checks_this_visit.deinit();
    for (fold.events[1..]) |event| switch (event.kind) {
        .enter_node => {
            if (required_edge_kind != null or event.node_id == null or pending_node == null or event.node_id.? != pending_node.?) {
                try ctx.fail(.{ .code = .invalid_fold_trace, .expansion_id = fold.expansion_id });
            } else {
                current = event.node_id.?;
                try ctx.fold_visited_nodes.put(.{ .fold_id = fold.id, .node_id = current }, {});
                pending_node = null;
                checks_this_visit.clearRetainingCapacity();
            }
        },
        .take_edge => {
            try validateVisitedCheckpoint(ctx, fold, current, &checks_this_visit);
            if (pending_node != null or event.edge_id == null) {
                try ctx.fail(.{ .code = .invalid_fold_trace, .expansion_id = fold.expansion_id });
                continue;
            }
            const edge_index = ctx.edges.get(event.edge_id.?) orelse {
                try ctx.fail(.{ .code = .invalid_fold_trace, .expansion_id = fold.expansion_id });
                continue;
            };
            const edge = ctx.manifest.inventory.control_edges[edge_index];
            if (edge.expansion_id != fold.expansion_id or edge.from != current) {
                try ctx.fail(.{ .code = .invalid_fold_trace, .expansion_id = fold.expansion_id });
            } else if (required_edge_kind != null and edge.kind != required_edge_kind.?) {
                try ctx.fail(.{ .code = .invalid_fold_trace, .expansion_id = fold.expansion_id });
            } else {
                pending_node = edge.to;
            }
            required_edge_kind = null;
        },
        .predicate_check => {
            if (required_edge_kind != null or event.node_id != current or event.use_id == null or event.predicate_value == null) {
                try ctx.fail(.{ .code = .invalid_concrete_checkpoint, .expansion_id = fold.expansion_id, .use_id = event.use_id });
                continue;
            }
            const use = ctx.use(event.use_id.?) orelse {
                try ctx.fail(.{ .code = .unknown_use, .expansion_id = fold.expansion_id, .use_id = event.use_id });
                continue;
            };
            if (use.expansion_id != fold.expansion_id or use.control_node_id != current) {
                try ctx.fail(.{ .code = .invalid_concrete_checkpoint, .site_id = use.site_id, .expansion_id = fold.expansion_id, .use_id = use.id });
            }
            const seen = try checks_this_visit.getOrPut(use.id);
            if (seen.found_existing) {
                try ctx.fail(.{ .code = .invalid_concrete_checkpoint, .site_id = use.site_id, .expansion_id = fold.expansion_id, .use_id = use.id });
            } else seen.value_ptr.* = event.predicate_value.?;
            try incrementCount(&ctx.predicate_traces, PredicateKey{
                .fold_id = fold.id,
                .use_id = use.id,
                .node_id = current,
                .value = event.predicate_value.?,
            });
        },
        .return_exit, .break_exit, .continue_backedge, .success_exit, .error_exit => {
            if (event.node_id != current) try ctx.fail(.{ .code = .invalid_fold_trace, .expansion_id = fold.expansion_id });
            if (required_edge_kind != null) try ctx.fail(.{ .code = .invalid_fold_trace, .expansion_id = fold.expansion_id });
            try validateVisitedCheckpoint(ctx, fold, current, &checks_this_visit);
            required_edge_kind = switch (event.kind) {
                .return_exit => if (current == fold.terminal_node_id) null else .return_exit,
                .break_exit => .break_exit,
                .continue_backedge => .continue_backedge,
                else => null,
            };
        },
    };
    if (pending_node != null or required_edge_kind != null or current != fold.terminal_node_id) {
        try ctx.fail(.{ .code = .invalid_fold_trace, .expansion_id = fold.expansion_id });
    }
    const terminal = ctx.manifest.inventory.control_nodes[terminal_index];
    const final_event = fold.events[fold.events.len - 1];
    const legal_terminal = switch (final_event.kind) {
        .success_exit => terminal.kind == .success_exit,
        .error_exit => terminal.kind == .error_exit,
        .return_exit => terminal.kind == .return_exit or terminal.kind == .success_exit or terminal.kind == .error_exit,
        else => false,
    };
    if (!legal_terminal) try ctx.fail(.{ .code = .invalid_fold_trace, .expansion_id = fold.expansion_id });
}

fn validateVisitedCheckpoint(
    ctx: *Context,
    fold: accounting.FoldRecord,
    node_id: accounting.ControlNodeId,
    checks: *const std.AutoHashMap(accounting.UseId, bool),
) !void {
    const node_index = ctx.nodes.get(node_id) orelse return;
    const node = ctx.manifest.inventory.control_nodes[node_index];
    for (node.attached_use_ids) |use_id| {
        const use = ctx.use(use_id) orelse continue;
        const site = ctx.typedSite(use.site_id) orelse continue;
        if (site.kind == .havoc or site.kind == .modifies) continue;
        const value = checks.get(use_id) orelse {
            try ctx.fail(.{ .code = .invalid_concrete_checkpoint, .site_id = site.id, .expansion_id = fold.expansion_id, .use_id = use.id });
            continue;
        };
        if (!value) try ctx.fail(.{
            .code = if (use.role == .assumption_context) .false_concrete_assumption else .false_concrete_target,
            .site_id = site.id,
            .expansion_id = fold.expansion_id,
            .use_id = use.id,
        });
    }
}

fn validateHandlingsAndEvidence(ctx: *Context) !void {
    for (ctx.manifest.inventory.uses) |use| {
        const expansion = ctx.expansion(use.expansion_id);
        if (expansion != null and expansion.?.disposition == .fold_abandoned_to_symbolic) {
            if (ctx.handling_by_use.get(use.id)) |handling_id| {
                try ctx.fail(.{
                    .code = .abandoned_fold_evidence,
                    .site_id = use.site_id,
                    .expansion_id = use.expansion_id,
                    .use_id = use.id,
                    .handling_id = handling_id,
                });
            }
            continue;
        }
        const handling_id = ctx.handling_by_use.get(use.id) orelse {
            try ctx.fail(.{ .code = .missing_handling, .site_id = use.site_id, .expansion_id = use.expansion_id, .use_id = use.id });
            continue;
        };
        const location = ctx.handlings.get(handling_id).?;
        try validateHandling(ctx, use, ctx.handling(location));
    }
    var handling_it = ctx.handlings.iterator();
    while (handling_it.next()) |entry| {
        const row = ctx.handling(entry.value_ptr.*);
        if (ctx.uses.get(row.use_id) == null) {
            try ctx.fail(.{ .code = .unknown_use, .use_id = row.use_id, .handling_id = row.id });
        }
    }
    var evidence_it = ctx.evidence_refs.iterator();
    while (evidence_it.next()) |entry| {
        if (entry.value_ptr.* == 0) try ctx.fail(.{ .code = .orphan_evidence, .evidence_id = entry.key_ptr.* });
        const location = ctx.evidence.get(entry.key_ptr.*).?;
        try validateCompleteEvidenceCoverage(ctx, entry.key_ptr.*, location);
    }
}

fn validateCompleteEvidenceCoverage(ctx: *Context, evidence_id: accounting.EvidenceId, location: Context.EvidenceLocation) !void {
    const covered: []const accounting.UseId = switch (location) {
        .predicate => |index| &.{ctx.manifest.comptime_evidence.predicate_events[index].use_id},
        .obligation => |index| ctx.manifest.symbolic.obligations[index].covered_use_ids,
        .assumption => |index| ctx.manifest.symbolic.assumptions[index].covered_use_ids,
        .query => |index| ctx.manifest.symbolic.queries[index].covered_use_ids,
        .runtime_check => |index| ctx.manifest.symbolic.runtime_checks[index].covered_use_ids,
        .frame_result => |index| ctx.manifest.symbolic.frame_results[index].covered_use_ids,
        .state_effect => |index| ctx.manifest.symbolic.state_effects[index].covered_use_ids,
    };
    for (covered) |use_id| {
        const handling_id = ctx.handling_by_use.get(use_id) orelse {
            try ctx.fail(.{ .code = .evidence_coverage_mismatch, .use_id = use_id, .evidence_id = evidence_id });
            continue;
        };
        const handling = ctx.handling(ctx.handlings.get(handling_id).?);
        if (!ctx.handling_evidence_refs.contains(.{ .evidence_id = evidence_id, .use_id = use_id })) {
            try ctx.fail(.{ .code = .evidence_coverage_mismatch, .use_id = use_id, .handling_id = handling.id, .evidence_id = evidence_id });
        }
    }
}

fn validateHandling(ctx: *Context, use: accounting.SourceFactUse, row: accounting.HandlingRecord) !void {
    const site = ctx.typedSite(use.site_id) orelse {
        try ctx.fail(.{ .code = .unknown_site, .site_id = use.site_id, .expansion_id = use.expansion_id, .use_id = use.id, .handling_id = row.id });
        return;
    };
    const expansion = ctx.expansion(use.expansion_id) orelse return;
    if (row.kind == .rejected) {
        try ctx.fail(.{ .code = .rejected_handling, .site_id = site.id, .expansion_id = use.expansion_id, .use_id = use.id, .handling_id = row.id });
        return;
    }

    if (!handlingPermitted(ctx.mode, site.origin, site.kind, use.role, row.kind, expansion.disposition)) {
        const code: FailureCode = switch (row.kind) {
            .reduced_scope_excluded => .reduced_scope_excluded_not_permitted,
            .verification_disabled => .verification_disabled_not_permitted,
            else => .handling_not_permitted,
        };
        try ctx.fail(.{ .code = code, .site_id = site.id, .expansion_id = use.expansion_id, .use_id = use.id, .handling_id = row.id });
    }

    if (!handlingReferencesCompatible(row)) {
        try ctx.fail(.{ .code = .handling_not_permitted, .site_id = site.id, .expansion_id = use.expansion_id, .use_id = use.id, .handling_id = row.id });
    }

    try validateEvidenceReferences(ctx, use, row);

    switch (row.kind) {
        .symbolic => {
            if (row.obligation_ids.len == 0) try ctx.fail(.{ .code = .missing_symbolic_obligation, .site_id = site.id, .expansion_id = use.expansion_id, .use_id = use.id, .handling_id = row.id });
            if (row.query_ids.len == 0) try ctx.fail(.{ .code = .missing_symbolic_query, .site_id = site.id, .expansion_id = use.expansion_id, .use_id = use.id, .handling_id = row.id });
            try validateQueryKinds(ctx, site.kind, row);
        },
        .concrete_true => {
            if (row.fold_id == null) try handlingNotPermitted(ctx, use, row);
            if (row.predicate_event_ids.len == 0) {
                try ctx.fail(.{ .code = .zero_concrete_checks, .site_id = site.id, .expansion_id = use.expansion_id, .use_id = use.id, .handling_id = row.id });
            }
            for (row.predicate_event_ids) |event_id| {
                const location = ctx.evidence.get(event_id) orelse continue;
                if (location != .predicate) continue;
                const event = ctx.manifest.comptime_evidence.predicate_events[location.predicate];
                if (!event.value) {
                    try ctx.fail(.{ .code = if (use.role == .assumption_context) .false_concrete_assumption else .false_concrete_target, .site_id = site.id, .expansion_id = use.expansion_id, .use_id = use.id, .handling_id = row.id, .evidence_id = event.id });
                }
            }
        },
        .runtime_enforced => {
            if (row.runtime_check_ids.len == 0) try ctx.fail(.{ .code = .missing_runtime_enforcement, .site_id = site.id, .expansion_id = use.expansion_id, .use_id = use.id, .handling_id = row.id });
        },
        .control_eliminated => try validateControlElimination(ctx, use, row),
        .assumption_incorporated => {
            if (row.assumption_ids.len == 0 and row.query_ids.len == 0) try ctx.fail(.{ .code = .missing_assumption_incorporation, .site_id = site.id, .expansion_id = use.expansion_id, .use_id = use.id, .handling_id = row.id });
        },
        .frame_validated => {
            if (row.frame_result_ids.len == 0) try ctx.fail(.{ .code = .missing_frame_validation, .site_id = site.id, .expansion_id = use.expansion_id, .use_id = use.id, .handling_id = row.id });
            for (row.frame_result_ids) |id| if (!validationEvidenceValid(ctx, id, .frame_result)) try ctx.fail(.{ .code = .invalid_frame_validation, .site_id = site.id, .expansion_id = use.expansion_id, .use_id = use.id, .handling_id = row.id, .evidence_id = id });
        },
        .state_effect_incorporated => {
            if (row.state_effect_ids.len == 0) try ctx.fail(.{ .code = .missing_state_effect, .site_id = site.id, .expansion_id = use.expansion_id, .use_id = use.id, .handling_id = row.id });
            for (row.state_effect_ids) |id| if (!validationEvidenceValid(ctx, id, .state_effect)) try ctx.fail(.{ .code = .invalid_state_effect, .site_id = site.id, .expansion_id = use.expansion_id, .use_id = use.id, .handling_id = row.id, .evidence_id = id });
        },
        .reduced_scope_excluded, .verification_disabled => {},
        .rejected => unreachable,
    }
}

pub fn handlingPermitted(
    mode: accounting.CompilationMode,
    origin: accounting.FactOrigin,
    kind: accounting.SourceFactKind,
    role: accounting.UseRole,
    handling: accounting.HandlingKind,
    disposition: accounting.ExpansionDisposition,
) bool {
    _ = origin; // Kept explicit in the closed policy domain; both origins obey the same table in schema 1.
    if (disposition == .rejected or disposition == .fold_abandoned_to_symbolic) return false;
    if (role == .proof_target and disposition == .symbolic and mode == .verified_basic and basicModeExcludes(kind)) {
        return handling == .reduced_scope_excluded;
    }
    if (role == .proof_target and disposition == .symbolic and mode == .unverified_emit) {
        return handling == .verification_disabled;
    }
    if (role == .assumption_context and disposition == .symbolic and mode == .unverified_emit) {
        // Runtime-owner preconditions retain their compiler-derived assumption
        // row; proof-only symbolic-call summaries are explicitly disabled by
        // the production adapter because no verifier query exists in this mode.
        return handling == .assumption_incorporated or handling == .verification_disabled;
    }
    return switch (role) {
        .proof_target => switch (disposition) {
            .symbolic => handling == .symbolic,
            .fold_committed => handling == .concrete_true or handling == .control_eliminated,
            .fold_abandoned_to_symbolic, .rejected => false,
        },
        .assumption_context => switch (disposition) {
            .symbolic => handling == .assumption_incorporated,
            .fold_committed => handling == .concrete_true or handling == .control_eliminated,
            .fold_abandoned_to_symbolic, .rejected => false,
        },
        .runtime_condition => switch (disposition) {
            .symbolic => handling == .runtime_enforced,
            .fold_committed => handling == .concrete_true or handling == .control_eliminated,
            .fold_abandoned_to_symbolic, .rejected => false,
        },
        .frame_directive => handling == .frame_validated or
            (disposition == .fold_committed and handling == .control_eliminated),
        .state_directive => switch (disposition) {
            .symbolic => handling == .state_effect_incorporated,
            .fold_committed => handling == .control_eliminated,
            .fold_abandoned_to_symbolic, .rejected => false,
        },
    };
}

fn handlingReferencesCompatible(row: accounting.HandlingRecord) bool {
    const has_obligations = row.obligation_ids.len != 0;
    const has_assumptions = row.assumption_ids.len != 0;
    const has_queries = row.query_ids.len != 0;
    const has_runtime_checks = row.runtime_check_ids.len != 0;
    const has_frame_results = row.frame_result_ids.len != 0;
    const has_state_effects = row.state_effect_ids.len != 0;
    const has_predicates = row.predicate_event_ids.len != 0;
    const has_rejection = row.rejection_reason != null;
    const has_control_event = row.control_event_index != null;
    return switch (row.kind) {
        .symbolic => !has_assumptions and !has_runtime_checks and !has_frame_results and
            !has_state_effects and !has_predicates and row.fold_id == null and
            !has_control_event and !has_rejection,
        .concrete_true => !has_obligations and !has_assumptions and !has_queries and
            !has_runtime_checks and !has_frame_results and !has_state_effects and
            row.fold_id != null and !has_control_event and !has_rejection,
        .runtime_enforced => !has_obligations and !has_assumptions and !has_queries and
            !has_frame_results and !has_state_effects and !has_predicates and
            row.fold_id == null and !has_control_event and !has_rejection,
        .control_eliminated => !has_obligations and !has_assumptions and !has_queries and
            !has_runtime_checks and !has_frame_results and !has_state_effects and
            !has_predicates and row.fold_id != null and !has_rejection,
        .assumption_incorporated => !has_obligations and
            !has_runtime_checks and !has_frame_results and !has_state_effects and
            !has_predicates and row.fold_id == null and !has_control_event and !has_rejection,
        .frame_validated => !has_obligations and !has_assumptions and !has_queries and
            !has_runtime_checks and !has_state_effects and !has_predicates and
            row.fold_id == null and !has_control_event and !has_rejection,
        .state_effect_incorporated => !has_obligations and !has_assumptions and !has_queries and
            !has_runtime_checks and !has_frame_results and !has_predicates and
            row.fold_id == null and !has_control_event and !has_rejection,
        .reduced_scope_excluded, .verification_disabled => !has_obligations and
            !has_assumptions and !has_queries and !has_runtime_checks and
            !has_frame_results and !has_state_effects and !has_predicates and
            row.fold_id == null and !has_control_event and !has_rejection,
        .rejected => has_rejection,
    };
}

fn basicModeExcludes(kind: accounting.SourceFactKind) bool {
    return switch (kind) {
        .guard, .loop_invariant, .contract_invariant, .assert, .refinement_guard => true,
        .requires, .ensures, .ensures_ok, .ensures_err, .assume, .havoc, .modifies, .runtime_guard => false,
    };
}

fn handlingNotPermitted(ctx: *Context, use: accounting.SourceFactUse, row: accounting.HandlingRecord) !void {
    try ctx.fail(.{ .code = .handling_not_permitted, .site_id = use.site_id, .expansion_id = use.expansion_id, .use_id = use.id, .handling_id = row.id });
}

fn validateQueryKinds(ctx: *Context, fact_kind: accounting.SourceFactKind, row: accounting.HandlingRecord) !void {
    var saw_obligation = false;
    var saw_step = false;
    for (row.query_ids) |id| {
        const location = ctx.evidence.get(id) orelse continue;
        if (location != .query) continue;
        const kind = ctx.manifest.symbolic.queries[location.query].kind;
        if (kind == .obligation) saw_obligation = true;
        if (kind == .loop_invariant_step) saw_step = true;
        const valid = switch (fact_kind) {
            .loop_invariant => kind == .obligation or kind == .loop_invariant_step or kind == .loop_body_safety or kind == .loop_invariant_post,
            .guard, .refinement_guard => kind == .guard_satisfy or kind == .guard_violate or kind == .obligation,
            .ensures => kind == .obligation or kind == .loop_invariant_post or kind == .other,
            else => kind == .obligation or kind == .other,
        };
        if (!valid) try ctx.fail(.{ .code = .invalid_symbolic_query_kind, .site_id = ctx.use(row.use_id).?.site_id, .use_id = row.use_id, .handling_id = row.id, .evidence_id = id });
    }
    if (fact_kind == .loop_invariant and (!saw_obligation or !saw_step)) {
        try ctx.fail(.{ .code = if (!saw_obligation) .missing_symbolic_obligation else .missing_symbolic_query, .site_id = ctx.use(row.use_id).?.site_id, .use_id = row.use_id, .handling_id = row.id });
    }
}

fn validationEvidenceValid(ctx: *Context, id: accounting.EvidenceId, expected: std.meta.Tag(Context.EvidenceLocation)) bool {
    const location = ctx.evidence.get(id) orelse return false;
    if (std.meta.activeTag(location) != expected) return false;
    return switch (location) {
        .frame_result => |index| ctx.manifest.symbolic.frame_results[index].valid,
        .state_effect => |index| ctx.manifest.symbolic.state_effects[index].valid,
        else => false,
    };
}

fn validateControlElimination(ctx: *Context, use: accounting.SourceFactUse, row: accounting.HandlingRecord) !void {
    const fold_id = row.fold_id orelse {
        try ctx.fail(.{ .code = .invalid_control_elimination, .site_id = use.site_id, .expansion_id = use.expansion_id, .use_id = use.id, .handling_id = row.id });
        return;
    };
    const fold = ctx.fold(fold_id) orelse {
        try ctx.fail(.{ .code = .invalid_control_elimination, .site_id = use.site_id, .expansion_id = use.expansion_id, .use_id = use.id, .handling_id = row.id });
        return;
    };
    if (fold.disposition != .committed or fold.expansion_id != use.expansion_id or use.control_node_id == null) {
        try ctx.fail(.{ .code = .invalid_control_elimination, .site_id = use.site_id, .expansion_id = use.expansion_id, .use_id = use.id, .handling_id = row.id });
        return;
    }
    if (row.control_event_index) |event_index| {
        if (event_index >= fold.events.len or fold.events[event_index].kind != .take_edge) {
            try ctx.fail(.{ .code = .invalid_control_elimination, .site_id = use.site_id, .expansion_id = use.expansion_id, .use_id = use.id, .handling_id = row.id });
            return;
        }
    }
    if (ctx.fold_visited_nodes.contains(.{ .fold_id = fold.id, .node_id = use.control_node_id.? })) {
        try ctx.fail(.{ .code = .invalid_control_elimination, .site_id = use.site_id, .expansion_id = use.expansion_id, .use_id = use.id, .handling_id = row.id });
    }
}

fn validateEvidenceReferences(ctx: *Context, use: accounting.SourceFactUse, row: accounting.HandlingRecord) !void {
    try validateEvidenceList(ctx, use, row, row.obligation_ids, .obligation);
    try validateEvidenceList(ctx, use, row, row.assumption_ids, .assumption);
    try validateEvidenceList(ctx, use, row, row.query_ids, .query);
    try validateEvidenceList(ctx, use, row, row.runtime_check_ids, .runtime_check);
    try validateEvidenceList(ctx, use, row, row.frame_result_ids, .frame_result);
    try validateEvidenceList(ctx, use, row, row.state_effect_ids, .state_effect);
    try validateEvidenceList(ctx, use, row, row.predicate_event_ids, .predicate);
    if (row.fold_id) |fold_id| {
        const fold = ctx.fold(fold_id) orelse {
            try ctx.fail(.{ .code = .unknown_evidence_reference, .site_id = use.site_id, .expansion_id = use.expansion_id, .use_id = use.id, .handling_id = row.id, .evidence_id = fold_id });
            return;
        };
        if (fold.disposition == .abandoned) try ctx.fail(.{ .code = .abandoned_fold_evidence, .site_id = use.site_id, .expansion_id = use.expansion_id, .use_id = use.id, .handling_id = row.id });
    }
}

fn validateEvidenceList(
    ctx: *Context,
    use: accounting.SourceFactUse,
    row: accounting.HandlingRecord,
    ids: []const accounting.EvidenceId,
    expected: std.meta.Tag(Context.EvidenceLocation),
) !void {
    for (ids) |id| {
        const location = ctx.evidence.get(id) orelse {
            try ctx.fail(.{ .code = .unknown_evidence_reference, .site_id = use.site_id, .expansion_id = use.expansion_id, .use_id = use.id, .handling_id = row.id, .evidence_id = id });
            continue;
        };
        if (std.meta.activeTag(location) != expected) {
            try ctx.fail(.{ .code = .unknown_evidence_reference, .site_id = use.site_id, .expansion_id = use.expansion_id, .use_id = use.id, .handling_id = row.id, .evidence_id = id });
            continue;
        }
        if (!ctx.evidence_coverage.contains(.{ .evidence_id = id, .use_id = use.id })) {
            try ctx.fail(.{ .code = .evidence_coverage_mismatch, .site_id = use.site_id, .expansion_id = use.expansion_id, .use_id = use.id, .handling_id = row.id, .evidence_id = id });
        }
        const ref_count = ctx.evidence_refs.getPtr(id).?;
        ref_count.* +|= 1;
    }
}

test "empty manifest is accepted only after pure reconciliation" {
    var result = try decide(std.testing.allocator, .verified_full, .{});
    defer result.deinit();
    try std.testing.expectEqual(Decision.accepted, result.decision);
    try std.testing.expectEqual(@as(?FailureCode, null), result.primary_failure);
}

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

fn acceptedSymbolicLoopFixture() accounting.Manifest {
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
        const rows = [_]accounting.OwnerTemplate{.{ .id = 1, .owner_key = "function:run", .activation = .runtime_body, .uses = &template_uses }};
    }.rows;
    const expansions = struct {
        const rows = [_]accounting.Expansion{.{ .id = 1, .template_id = 1, .activation = .runtime_owner, .disposition = .symbolic, .root_runtime_owner = "function:run", .identity = "runtime:run" }};
    }.rows;
    const uses = struct {
        const rows = [_]accounting.SourceFactUse{
            .{ .id = 1, .site_id = 1, .expansion_id = 1, .template_ordinal = 0, .role = .proof_target },
            .{ .id = 2, .site_id = 1, .expansion_id = 1, .template_ordinal = 1, .role = .assumption_context },
        };
    }.rows;
    const covered_target = struct {
        const rows = [_]accounting.UseId{1};
    }.rows;
    const covered_context = struct {
        const rows = [_]accounting.UseId{2};
    }.rows;
    const obligations = struct {
        const rows = [_]accounting.CoveredEvidence{.{ .id = 11, .producer_id = 101, .covered_use_ids = &covered_target }};
    }.rows;
    const assumptions = struct {
        const rows = [_]accounting.CoveredEvidence{.{ .id = 14, .producer_id = 104, .covered_use_ids = &covered_context }};
    }.rows;
    const queries = struct {
        const rows = [_]accounting.QueryEvidence{
            .{ .id = 12, .producer_id = 102, .kind = .obligation, .covered_use_ids = &covered_target },
            .{ .id = 13, .producer_id = 103, .kind = .loop_invariant_step, .covered_use_ids = &covered_target },
        };
    }.rows;
    const obligation_ids = struct {
        const rows = [_]accounting.EvidenceId{11};
    }.rows;
    const query_ids = struct {
        const rows = [_]accounting.EvidenceId{ 12, 13 };
    }.rows;
    const assumption_ids = struct {
        const rows = [_]accounting.EvidenceId{14};
    }.rows;
    const handlings = struct {
        const rows = [_]accounting.HandlingRecord{
            .{ .id = 1, .use_id = 1, .kind = .symbolic, .obligation_ids = &obligation_ids, .query_ids = &query_ids },
            .{ .id = 2, .use_id = 2, .kind = .assumption_incorporated, .assumption_ids = &assumption_ids },
        };
    }.rows;
    return .{
        .inventory = .{
            .declared_sites = &declared,
            .typed_sites = &typed,
            .owner_templates = &templates,
            .expansions = &expansions,
            .uses = &uses,
        },
        .symbolic = .{
            .obligations = &obligations,
            .assumptions = &assumptions,
            .queries = &queries,
            .handlings = &handlings,
        },
    };
}

fn acceptedConcreteLoopFixture() accounting.Manifest {
    var base = missingHandlingFixture();
    const template_uses = struct {
        const rows = [_]accounting.UseTemplate{
            .{ .site_id = 1, .role = .proof_target, .control_node_slot = 1 },
            .{ .site_id = 1, .role = .assumption_context, .control_node_slot = 1 },
        };
    }.rows;
    const attached_ordinals = struct {
        const rows = [_]u32{ 0, 1 };
    }.rows;
    const template_nodes = struct {
        const rows = [_]accounting.ControlNodeTemplate{
            .{ .slot = 0, .kind = .entry, .range = .{ .file = "fixture.ora", .start = 10, .end = 21 } },
            .{ .slot = 1, .kind = .loop_head, .range = .{ .file = "fixture.ora", .start = 10, .end = 20 }, .attached_use_ordinals = &attached_ordinals },
            .{ .slot = 2, .kind = .success_exit, .range = .{ .file = "fixture.ora", .start = 21, .end = 21 } },
        };
    }.rows;
    const template_edges = struct {
        const rows = [_]accounting.ControlEdgeTemplate{
            .{ .slot = 0, .from_slot = 0, .to_slot = 1, .kind = .next },
            .{ .slot = 1, .from_slot = 1, .to_slot = 2, .kind = .loop_exit },
            .{ .slot = 2, .from_slot = 1, .to_slot = 1, .kind = .backedge },
        };
    }.rows;
    const terminal_slots = struct {
        const rows = [_]u32{2};
    }.rows;
    const templates = struct {
        const rows = [_]accounting.OwnerTemplate{.{
            .id = 1,
            .owner_key = "function:run",
            .activation = .comptime_body,
            .uses = &template_uses,
            .control_nodes = &template_nodes,
            .control_edges = &template_edges,
            .entry_slot = 0,
            .terminal_slots = &terminal_slots,
        }};
    }.rows;
    const uses = struct {
        const rows = [_]accounting.SourceFactUse{
            .{ .id = 1, .site_id = 1, .expansion_id = 1, .template_ordinal = 0, .role = .proof_target, .control_node_id = 2 },
            .{ .id = 2, .site_id = 1, .expansion_id = 1, .template_ordinal = 1, .role = .assumption_context, .control_node_id = 2 },
        };
    }.rows;
    const attached = struct {
        const rows = [_]accounting.UseId{ 1, 2 };
    }.rows;
    const nodes = struct {
        const rows = [_]accounting.ControlNode{
            .{ .id = 1, .expansion_id = 1, .slot = 0, .kind = .entry, .range = .{ .file = "fixture.ora", .start = 10, .end = 21 } },
            .{ .id = 2, .expansion_id = 1, .slot = 1, .kind = .loop_head, .range = .{ .file = "fixture.ora", .start = 10, .end = 20 }, .attached_use_ids = &attached },
            .{ .id = 3, .expansion_id = 1, .slot = 2, .kind = .success_exit, .range = .{ .file = "fixture.ora", .start = 21, .end = 21 } },
        };
    }.rows;
    const edges = struct {
        const rows = [_]accounting.ControlEdge{
            .{ .id = 1, .expansion_id = 1, .from = 1, .to = 2, .kind = .next },
            .{ .id = 2, .expansion_id = 1, .from = 2, .to = 3, .kind = .loop_exit },
            .{ .id = 3, .expansion_id = 1, .from = 2, .to = 2, .kind = .backedge },
        };
    }.rows;
    const trace = struct {
        const rows = [_]accounting.TraceEvent{
            .{ .kind = .enter_node, .node_id = 1 },
            .{ .kind = .take_edge, .edge_id = 1 },
            .{ .kind = .enter_node, .node_id = 2 },
            .{ .kind = .predicate_check, .node_id = 2, .use_id = 1, .predicate_value = true },
            .{ .kind = .predicate_check, .node_id = 2, .use_id = 2, .predicate_value = true },
            .{ .kind = .take_edge, .edge_id = 2 },
            .{ .kind = .enter_node, .node_id = 3 },
            .{ .kind = .success_exit, .node_id = 3 },
        };
    }.rows;
    const folds = struct {
        const rows = [_]accounting.FoldRecord{.{ .id = 1, .expansion_id = 1, .entry_node_id = 1, .terminal_node_id = 3, .disposition = .committed, .events = &trace }};
    }.rows;
    const predicates = struct {
        const rows = [_]accounting.PredicateEvent{
            .{ .id = 21, .fold_id = 1, .use_id = 1, .node_id = 2, .value = true },
            .{ .id = 22, .fold_id = 1, .use_id = 2, .node_id = 2, .value = true },
        };
    }.rows;
    const predicate_one = struct {
        const rows = [_]accounting.EvidenceId{21};
    }.rows;
    const predicate_two = struct {
        const rows = [_]accounting.EvidenceId{22};
    }.rows;
    const handlings = struct {
        const rows = [_]accounting.HandlingRecord{
            .{ .id = 1, .use_id = 1, .kind = .concrete_true, .predicate_event_ids = &predicate_one, .fold_id = 1 },
            .{ .id = 2, .use_id = 2, .kind = .concrete_true, .predicate_event_ids = &predicate_two, .fold_id = 1 },
        };
    }.rows;
    base.inventory.owner_templates = &templates;
    base.inventory.uses = &uses;
    base.inventory.control_nodes = &nodes;
    base.inventory.control_edges = &edges;
    base.comptime_evidence = .{ .folds = &folds, .predicate_events = &predicates, .handlings = &handlings };
    return base;
}

fn resultHasFailure(result: Result, code: FailureCode) bool {
    for (result.failures) |row| if (row.code == code) return true;
    return false;
}

pub const max_failure_witness_failures = 256;

pub const FailureWitnessObservation = struct {
    observed: bool = false,
    decision: Decision = .accepted,
    primary_failure: ?FailureCode = null,
    failure_codes: [max_failure_witness_failures]FailureCode = undefined,
    failure_count: usize = 0,

    pub fn failures(self: *const FailureWitnessObservation) []const FailureCode {
        return self.failure_codes[0..self.failure_count];
    }
};

pub const FailureWitnesses = struct {
    rows: [std.meta.fields(FailureCode).len]FailureWitnessObservation,
};

fn noteFailures(witnesses: *FailureWitnesses, result: Result) !void {
    if (result.failures.len > max_failure_witness_failures) return error.SourceAccountingFailureWitnessOverflow;
    for (result.failures) |row| {
        const observation = &witnesses.rows[@intFromEnum(row.code)];
        if (observation.observed) continue;
        observation.observed = true;
        observation.decision = result.decision;
        observation.primary_failure = result.primary_failure;
        observation.failure_count = result.failures.len;
        for (result.failures, observation.failure_codes[0..result.failures.len]) |failure, *code| {
            code.* = failure.code;
        }
    }
}

const DirectiveEvidenceKind = enum { none, assumption, runtime_check, frame_result, state_effect };

fn decideDirectiveFixture(
    allocator: std.mem.Allocator,
    mode: accounting.CompilationMode,
    kind: accounting.SourceFactKind,
    role: accounting.UseRole,
    handling_kind: accounting.HandlingKind,
    evidence_kind: DirectiveEvidenceKind,
    evidence_valid: bool,
) !Result {
    const key: accounting.SiteKey = .{
        .path = "directive.ora",
        .owner = "function:run",
        .range_start = 1,
        .range_end = 2,
        .kind = kind,
        .ordinal = 0,
    };
    const declared = [_]accounting.DeclaredSite{.{ .id = 1, .key = key }};
    const typed = [_]accounting.TypedSite{.{ .id = 1, .origin = .source_syntax, .kind = kind, .key = key, .source_fact_id = key.range_start, .declared_site_id = 1 }};
    const template_uses = [_]accounting.UseTemplate{.{ .site_id = 1, .role = role }};
    const templates = [_]accounting.OwnerTemplate{.{ .id = 1, .owner_key = "function:run", .activation = .runtime_body, .uses = &template_uses }};
    const expansions = [_]accounting.Expansion{.{ .id = 1, .template_id = 1, .activation = .runtime_owner, .disposition = .symbolic, .root_runtime_owner = "function:run", .identity = "runtime:run" }};
    const uses = [_]accounting.SourceFactUse{.{ .id = 1, .site_id = 1, .expansion_id = 1, .template_ordinal = 0, .role = role }};
    const covered = [_]accounting.UseId{1};
    const evidence_ids = [_]accounting.EvidenceId{31};
    const covered_row = [_]accounting.CoveredEvidence{.{ .id = 31, .producer_id = 1, .covered_use_ids = &covered }};
    const validation_row = [_]accounting.ValidationEvidence{.{ .id = 31, .producer_id = 1, .covered_use_ids = &covered, .valid = evidence_valid }};
    var handling: accounting.HandlingRecord = .{ .id = 1, .use_id = 1, .kind = handling_kind };
    var symbolic: accounting.SymbolicEvidence = .{};
    switch (evidence_kind) {
        .none => {},
        .assumption => {
            handling.assumption_ids = &evidence_ids;
            symbolic.assumptions = &covered_row;
        },
        .runtime_check => {
            handling.runtime_check_ids = &evidence_ids;
            symbolic.runtime_checks = &covered_row;
        },
        .frame_result => {
            handling.frame_result_ids = &evidence_ids;
            symbolic.frame_results = &validation_row;
        },
        .state_effect => {
            handling.state_effect_ids = &evidence_ids;
            symbolic.state_effects = &validation_row;
        },
    }
    const handlings = [_]accounting.HandlingRecord{handling};
    symbolic.handlings = &handlings;
    return decide(allocator, mode, .{
        .inventory = .{
            .declared_sites = &declared,
            .typed_sites = &typed,
            .owner_templates = &templates,
            .expansions = &expansions,
            .uses = &uses,
        },
        .symbolic = symbolic,
    });
}

test "inventoried invariant with no terminal handling is rejected" {
    var result = try decide(std.testing.allocator, .verified_full, missingHandlingFixture());
    defer result.deinit();
    try std.testing.expectEqual(Decision.rejected, result.decision);
    try std.testing.expectEqual(@as(?FailureCode, .missing_handling), result.primary_failure);
    try std.testing.expectEqual(@as(usize, 2), result.failures.len);
}

test "symbolic and concrete invariant fixtures satisfy the accounting law" {
    var symbolic = try decide(std.testing.allocator, .verified_full, acceptedSymbolicLoopFixture());
    defer symbolic.deinit();
    try std.testing.expectEqual(Decision.accepted, symbolic.decision);
    var concrete = try decide(std.testing.allocator, .verified_full, acceptedConcreteLoopFixture());
    defer concrete.deinit();
    try std.testing.expectEqual(Decision.accepted, concrete.decision);
}

test "abandoned fold carries no authority and transfers accounting to its symbolic replacement" {
    const base = acceptedSymbolicLoopFixture();
    var comptime_template = base.inventory.owner_templates[0];
    comptime_template.id = 2;
    comptime_template.activation = .comptime_body;
    const boundary_template: accounting.OwnerTemplate = .{
        .id = 3,
        .owner_key = "function:run",
        .activation = .symbolic_call_boundary,
        .uses = &.{},
    };
    const templates = [_]accounting.OwnerTemplate{
        base.inventory.owner_templates[0],
        comptime_template,
        boundary_template,
    };
    const expansions = [_]accounting.Expansion{
        .{
            .id = 1,
            .template_id = 2,
            .replacement_expansion_id = 2,
            .activation = .speculative_fold,
            .disposition = .fold_abandoned_to_symbolic,
            .root_runtime_owner = "function:run",
            .identity = "fold:run",
        },
        .{
            .id = 2,
            .template_id = 3,
            .activation = .symbolic_call,
            .disposition = .symbolic,
            .root_runtime_owner = "function:run",
            .identity = "fold:run",
        },
        .{
            .id = 3,
            .template_id = 1,
            .activation = .runtime_owner,
            .disposition = .symbolic,
            .root_runtime_owner = "function:run",
            .identity = "runtime:run",
        },
    };
    const uses = [_]accounting.SourceFactUse{
        .{ .id = 1, .site_id = 1, .expansion_id = 3, .template_ordinal = 0, .role = .proof_target },
        .{ .id = 2, .site_id = 1, .expansion_id = 3, .template_ordinal = 1, .role = .assumption_context },
    };
    var manifest = base;
    manifest.inventory.owner_templates = &templates;
    manifest.inventory.expansions = &expansions;
    manifest.inventory.uses = &uses;
    var accepted = try decide(std.testing.allocator, .verified_full, manifest);
    defer accepted.deinit();
    try std.testing.expectEqual(Decision.accepted, accepted.decision);

    const forbidden_use: accounting.SourceFactUse = .{
        .id = 3,
        .site_id = 1,
        .expansion_id = 1,
        .template_ordinal = 0,
        .role = .proof_target,
    };
    const forbidden_uses = [_]accounting.SourceFactUse{ uses[0], uses[1], forbidden_use };
    const forbidden_handlings = [_]accounting.HandlingRecord{
        base.symbolic.handlings[0],
        base.symbolic.handlings[1],
        .{ .id = 3, .use_id = 3, .kind = .verification_disabled },
    };
    manifest.inventory.uses = &forbidden_uses;
    manifest.symbolic.handlings = &forbidden_handlings;
    var rejected = try decide(std.testing.allocator, .verified_full, manifest);
    defer rejected.deinit();
    try std.testing.expect(resultHasFailure(rejected, .abandoned_fold_evidence));
}

test "conservation mutations cannot silently remove source intent" {
    const base = acceptedSymbolicLoopFixture();
    {
        var mutated = base;
        mutated.inventory.declared_sites = &.{};
        var result = try decide(std.testing.allocator, .verified_full, mutated);
        defer result.deinit();
        try std.testing.expectEqual(Decision.rejected, result.decision);
        try std.testing.expect(resultHasFailure(result, .unknown_semantic_site));
    }
    {
        var mutated = base;
        mutated.inventory.typed_sites = &.{};
        var result = try decide(std.testing.allocator, .verified_full, mutated);
        defer result.deinit();
        try std.testing.expect(resultHasFailure(result, .missing_semantic_site));
    }
    {
        var mutated = base;
        mutated.inventory.uses = mutated.inventory.uses[0..1];
        var result = try decide(std.testing.allocator, .verified_full, mutated);
        defer result.deinit();
        try std.testing.expect(resultHasFailure(result, .missing_semantic_use));
        try std.testing.expect(resultHasFailure(result, .unknown_use));
    }
    {
        var mutated = base;
        mutated.symbolic.queries = mutated.symbolic.queries[0..1];
        var result = try decide(std.testing.allocator, .verified_full, mutated);
        defer result.deinit();
        try std.testing.expect(resultHasFailure(result, .unknown_evidence_reference));
    }
    {
        const query_ids = [_]accounting.EvidenceId{12};
        var handlings = [_]accounting.HandlingRecord{ base.symbolic.handlings[0], base.symbolic.handlings[1] };
        handlings[0].query_ids = &query_ids;
        var mutated = base;
        mutated.symbolic.handlings = &handlings;
        var result = try decide(std.testing.allocator, .verified_full, mutated);
        defer result.deinit();
        try std.testing.expect(resultHasFailure(result, .missing_symbolic_query));
        try std.testing.expect(resultHasFailure(result, .orphan_evidence));
    }
    {
        var mutated = base;
        mutated.symbolic.handlings = mutated.symbolic.handlings[0..1];
        var result = try decide(std.testing.allocator, .verified_full, mutated);
        defer result.deinit();
        try std.testing.expect(resultHasFailure(result, .missing_handling));
    }
}

test "trace and terminal-evidence mutations fail closed" {
    const base = acceptedConcreteLoopFixture();
    {
        var events = [_]accounting.TraceEvent{
            base.comptime_evidence.folds[0].events[0],
            base.comptime_evidence.folds[0].events[1],
            base.comptime_evidence.folds[0].events[2],
            base.comptime_evidence.folds[0].events[4],
            base.comptime_evidence.folds[0].events[5],
            base.comptime_evidence.folds[0].events[6],
            base.comptime_evidence.folds[0].events[7],
        };
        var folds = [_]accounting.FoldRecord{base.comptime_evidence.folds[0]};
        folds[0].events = &events;
        var mutated = base;
        mutated.comptime_evidence.folds = &folds;
        var result = try decide(std.testing.allocator, .verified_full, mutated);
        defer result.deinit();
        try std.testing.expect(resultHasFailure(result, .invalid_concrete_checkpoint));
    }
    {
        const original = base.comptime_evidence.folds[0].events;
        var events = [_]accounting.TraceEvent{
            original[0],
            original[1],
            original[2],
            original[3],
            original[4],
            .{ .kind = .take_edge, .edge_id = 3 },
            .{ .kind = .enter_node, .node_id = 2 },
            original[5],
            original[6],
            original[7],
        };
        var folds = [_]accounting.FoldRecord{base.comptime_evidence.folds[0]};
        folds[0].events = &events;
        var mutated = base;
        mutated.comptime_evidence.folds = &folds;
        var result = try decide(std.testing.allocator, .verified_full, mutated);
        defer result.deinit();
        try std.testing.expect(resultHasFailure(result, .invalid_concrete_checkpoint));
    }
    {
        var predicates = [_]accounting.PredicateEvent{ base.comptime_evidence.predicate_events[0], base.comptime_evidence.predicate_events[1] };
        predicates[0].value = false;
        var mutated = base;
        mutated.comptime_evidence.predicate_events = &predicates;
        var result = try decide(std.testing.allocator, .verified_full, mutated);
        defer result.deinit();
        try std.testing.expect(resultHasFailure(result, .invalid_concrete_checkpoint));
        try std.testing.expect(resultHasFailure(result, .false_concrete_target));
    }
    {
        var handlings = [_]accounting.HandlingRecord{ base.comptime_evidence.handlings[0], base.comptime_evidence.handlings[1] };
        handlings[0].kind = .control_eliminated;
        handlings[0].predicate_event_ids = &.{};
        var mutated = base;
        mutated.comptime_evidence.handlings = &handlings;
        var result = try decide(std.testing.allocator, .verified_full, mutated);
        defer result.deinit();
        try std.testing.expect(resultHasFailure(result, .invalid_control_elimination));
    }
    {
        var folds = [_]accounting.FoldRecord{base.comptime_evidence.folds[0]};
        folds[0].disposition = .abandoned;
        var mutated = base;
        mutated.comptime_evidence.folds = &folds;
        var result = try decide(std.testing.allocator, .verified_full, mutated);
        defer result.deinit();
        try std.testing.expect(resultHasFailure(result, .abandoned_fold_evidence));
    }
}

test "duplicate and orphan terminal rows are rejected" {
    const base = acceptedSymbolicLoopFixture();
    {
        const handlings = [_]accounting.HandlingRecord{ base.symbolic.handlings[0], base.symbolic.handlings[0], base.symbolic.handlings[1] };
        var mutated = base;
        mutated.symbolic.handlings = &handlings;
        var result = try decide(std.testing.allocator, .verified_full, mutated);
        defer result.deinit();
        try std.testing.expect(resultHasFailure(result, .duplicate_identity));
        try std.testing.expect(resultHasFailure(result, .duplicate_handling));
    }
    {
        const covered = [_]accounting.UseId{1};
        const obligations = [_]accounting.CoveredEvidence{ base.symbolic.obligations[0], .{ .id = 99, .producer_id = 999, .covered_use_ids = &covered } };
        var mutated = base;
        mutated.symbolic.obligations = &obligations;
        var result = try decide(std.testing.allocator, .verified_full, mutated);
        defer result.deinit();
        try std.testing.expect(resultHasFailure(result, .orphan_evidence));
        try std.testing.expect(resultHasFailure(result, .evidence_coverage_mismatch));
    }
}

pub fn observeFailureWitnesses(allocator: std.mem.Allocator) !FailureWitnesses {
    var witnesses: FailureWitnesses = .{
        .rows = [_]FailureWitnessObservation{.{}} ** std.meta.fields(FailureCode).len,
    };
    const symbolic_base = acceptedSymbolicLoopFixture();
    const concrete_base = acceptedConcreteLoopFixture();

    // Structural identities and source/semantic conservation.
    {
        const duplicate = [_]accounting.DeclaredSite{ symbolic_base.inventory.declared_sites[0], symbolic_base.inventory.declared_sites[0] };
        var manifest = symbolic_base;
        manifest.inventory.declared_sites = &duplicate;
        var result = try decide(allocator, .verified_full, manifest);
        defer result.deinit();
        try noteFailures(&witnesses, result);
    }
    {
        var second = symbolic_base.inventory.declared_sites[0];
        second.id = 2;
        const colliding = [_]accounting.DeclaredSite{ symbolic_base.inventory.declared_sites[0], second };
        var manifest = symbolic_base;
        manifest.inventory.declared_sites = &colliding;
        var result = try decide(allocator, .verified_full, manifest);
        defer result.deinit();
        try noteFailures(&witnesses, result);
    }
    {
        var template_uses = [_]accounting.UseTemplate{ symbolic_base.inventory.owner_templates[0].uses[0], symbolic_base.inventory.owner_templates[0].uses[1] };
        template_uses[0].site_id = 99;
        var templates = [_]accounting.OwnerTemplate{symbolic_base.inventory.owner_templates[0]};
        templates[0].uses = &template_uses;
        var manifest = symbolic_base;
        manifest.inventory.owner_templates = &templates;
        var result = try decide(allocator, .verified_full, manifest);
        defer result.deinit();
        try noteFailures(&witnesses, result);
    }
    {
        var uses = [_]accounting.SourceFactUse{ symbolic_base.inventory.uses[0], symbolic_base.inventory.uses[1] };
        uses[0].expansion_id = 99;
        var manifest = symbolic_base;
        manifest.inventory.uses = &uses;
        var result = try decide(allocator, .verified_full, manifest);
        defer result.deinit();
        try noteFailures(&witnesses, result);
    }
    {
        var handlings = [_]accounting.HandlingRecord{ symbolic_base.symbolic.handlings[0], symbolic_base.symbolic.handlings[1] };
        handlings[0].use_id = 99;
        var manifest = symbolic_base;
        manifest.symbolic.handlings = &handlings;
        var result = try decide(allocator, .verified_full, manifest);
        defer result.deinit();
        try noteFailures(&witnesses, result);
    }
    {
        const bad_ids = [_]accounting.EvidenceId{99};
        var handlings = [_]accounting.HandlingRecord{ symbolic_base.symbolic.handlings[0], symbolic_base.symbolic.handlings[1] };
        handlings[0].query_ids = &bad_ids;
        var manifest = symbolic_base;
        manifest.symbolic.handlings = &handlings;
        var result = try decide(allocator, .verified_full, manifest);
        defer result.deinit();
        try noteFailures(&witnesses, result);
    }
    {
        const wrong_coverage = [_]accounting.UseId{2};
        var queries = [_]accounting.QueryEvidence{ symbolic_base.symbolic.queries[0], symbolic_base.symbolic.queries[1] };
        queries[0].covered_use_ids = &wrong_coverage;
        var manifest = symbolic_base;
        manifest.symbolic.queries = &queries;
        var result = try decide(allocator, .verified_full, manifest);
        defer result.deinit();
        try noteFailures(&witnesses, result);
    }
    {
        const orphan = [_]accounting.CoveredEvidence{.{ .id = 99, .producer_id = 99, .covered_use_ids = &.{} }};
        var manifest = symbolic_base;
        manifest.symbolic.runtime_checks = &orphan;
        var result = try decide(allocator, .verified_full, manifest);
        defer result.deinit();
        try noteFailures(&witnesses, result);
    }
    {
        var manifest = symbolic_base;
        manifest.inventory.typed_sites = &.{};
        var result = try decide(allocator, .verified_full, manifest);
        defer result.deinit();
        try noteFailures(&witnesses, result);
    }
    {
        var manifest = symbolic_base;
        manifest.inventory.declared_sites = &.{};
        var result = try decide(allocator, .verified_full, manifest);
        defer result.deinit();
        try noteFailures(&witnesses, result);
    }
    {
        var manifest = symbolic_base;
        manifest.inventory.uses = manifest.inventory.uses[0..1];
        var result = try decide(allocator, .verified_full, manifest);
        defer result.deinit();
        try noteFailures(&witnesses, result);
    }
    {
        var uses = [_]accounting.SourceFactUse{ symbolic_base.inventory.uses[0], symbolic_base.inventory.uses[1] };
        uses[0].role = .runtime_condition;
        var manifest = symbolic_base;
        manifest.inventory.uses = &uses;
        var result = try decide(allocator, .verified_full, manifest);
        defer result.deinit();
        try noteFailures(&witnesses, result);
    }
    {
        var template_uses = [_]accounting.UseTemplate{ symbolic_base.inventory.owner_templates[0].uses[0], symbolic_base.inventory.owner_templates[0].uses[1] };
        template_uses[0].role = .frame_directive;
        var templates = [_]accounting.OwnerTemplate{symbolic_base.inventory.owner_templates[0]};
        templates[0].uses = &template_uses;
        var manifest = symbolic_base;
        manifest.inventory.owner_templates = &templates;
        var result = try decide(allocator, .verified_full, manifest);
        defer result.deinit();
        try noteFailures(&witnesses, result);
    }
    {
        var templates = [_]accounting.OwnerTemplate{symbolic_base.inventory.owner_templates[0]};
        templates[0].uses = templates[0].uses[0..1];
        var manifest = symbolic_base;
        manifest.inventory.owner_templates = &templates;
        var result = try decide(allocator, .verified_full, manifest);
        defer result.deinit();
        try noteFailures(&witnesses, result);
    }

    // Expansion, source-control, and fold-trace validity.
    {
        var expansions = [_]accounting.Expansion{symbolic_base.inventory.expansions[0]};
        expansions[0].identity = "";
        var manifest = symbolic_base;
        manifest.inventory.expansions = &expansions;
        var result = try decide(allocator, .verified_full, manifest);
        defer result.deinit();
        try noteFailures(&witnesses, result);
    }
    {
        var expansions = [_]accounting.Expansion{symbolic_base.inventory.expansions[0]};
        expansions[0].parent_expansion_id = 99;
        var manifest = symbolic_base;
        manifest.inventory.expansions = &expansions;
        var result = try decide(allocator, .verified_full, manifest);
        defer result.deinit();
        try noteFailures(&witnesses, result);
    }
    {
        var expansions = [_]accounting.Expansion{symbolic_base.inventory.expansions[0]};
        expansions[0].replacement_expansion_id = 99;
        var manifest = symbolic_base;
        manifest.inventory.expansions = &expansions;
        var result = try decide(allocator, .verified_full, manifest);
        defer result.deinit();
        try noteFailures(&witnesses, result);
    }
    {
        var nodes = [_]accounting.ControlNode{ concrete_base.inventory.control_nodes[0], concrete_base.inventory.control_nodes[1], concrete_base.inventory.control_nodes[2] };
        nodes[1].kind = .branch;
        var manifest = concrete_base;
        manifest.inventory.control_nodes = &nodes;
        var result = try decide(allocator, .verified_full, manifest);
        defer result.deinit();
        try noteFailures(&witnesses, result);
    }
    {
        var folds = [_]accounting.FoldRecord{concrete_base.comptime_evidence.folds[0]};
        folds[0].events = folds[0].events[0 .. folds[0].events.len - 1];
        var manifest = concrete_base;
        manifest.comptime_evidence.folds = &folds;
        var result = try decide(allocator, .verified_full, manifest);
        defer result.deinit();
        try noteFailures(&witnesses, result);
    }
    {
        var expansions = [_]accounting.Expansion{symbolic_base.inventory.expansions[0]};
        expansions[0].disposition = .fold_abandoned_to_symbolic;
        expansions[0].replacement_expansion_id = null;
        var manifest = symbolic_base;
        manifest.inventory.expansions = &expansions;
        var result = try decide(allocator, .verified_full, manifest);
        defer result.deinit();
        try noteFailures(&witnesses, result);
    }
    {
        var folds = [_]accounting.FoldRecord{concrete_base.comptime_evidence.folds[0]};
        folds[0].disposition = .abandoned;
        var manifest = concrete_base;
        manifest.comptime_evidence.folds = &folds;
        var result = try decide(allocator, .verified_full, manifest);
        defer result.deinit();
        try noteFailures(&witnesses, result);
    }
    {
        var generated_key = invariant_key;
        generated_key.range_start = 30;
        generated_key.range_end = 31;
        generated_key.kind = .refinement_guard;
        const typed = [_]accounting.TypedSite{
            symbolic_base.inventory.typed_sites[0],
            .{ .id = 2, .origin = .semantic_generated, .kind = .refinement_guard, .key = generated_key },
        };
        var manifest = symbolic_base;
        manifest.inventory.typed_sites = &typed;
        var result = try decide(allocator, .verified_full, manifest);
        defer result.deinit();
        try noteFailures(&witnesses, result);
    }

    // Terminal handling and evidence-specific failures.
    {
        var manifest = symbolic_base;
        manifest.symbolic.handlings = manifest.symbolic.handlings[0..1];
        var result = try decide(allocator, .verified_full, manifest);
        defer result.deinit();
        try noteFailures(&witnesses, result);
    }
    {
        const handlings = [_]accounting.HandlingRecord{ symbolic_base.symbolic.handlings[0], symbolic_base.symbolic.handlings[0], symbolic_base.symbolic.handlings[1] };
        var manifest = symbolic_base;
        manifest.symbolic.handlings = &handlings;
        var result = try decide(allocator, .verified_full, manifest);
        defer result.deinit();
        try noteFailures(&witnesses, result);
    }
    {
        var handlings = [_]accounting.HandlingRecord{ symbolic_base.symbolic.handlings[0], symbolic_base.symbolic.handlings[1] };
        handlings[0].kind = .rejected;
        var manifest = symbolic_base;
        manifest.symbolic.handlings = &handlings;
        var result = try decide(allocator, .verified_full, manifest);
        defer result.deinit();
        try noteFailures(&witnesses, result);
    }
    {
        var handlings = [_]accounting.HandlingRecord{ symbolic_base.symbolic.handlings[0], symbolic_base.symbolic.handlings[1] };
        handlings[0].obligation_ids = &.{};
        var manifest = symbolic_base;
        manifest.symbolic.obligations = &.{};
        manifest.symbolic.handlings = &handlings;
        var result = try decide(allocator, .verified_full, manifest);
        defer result.deinit();
        try noteFailures(&witnesses, result);
    }
    {
        var handlings = [_]accounting.HandlingRecord{ symbolic_base.symbolic.handlings[0], symbolic_base.symbolic.handlings[1] };
        handlings[0].query_ids = &.{};
        var manifest = symbolic_base;
        manifest.symbolic.queries = &.{};
        manifest.symbolic.handlings = &handlings;
        var result = try decide(allocator, .verified_full, manifest);
        defer result.deinit();
        try noteFailures(&witnesses, result);
    }
    {
        var queries = [_]accounting.QueryEvidence{ symbolic_base.symbolic.queries[0], symbolic_base.symbolic.queries[1] };
        queries[1].kind = .other;
        var manifest = symbolic_base;
        manifest.symbolic.queries = &queries;
        var result = try decide(allocator, .verified_full, manifest);
        defer result.deinit();
        try noteFailures(&witnesses, result);
    }
    inline for (.{
        .{ accounting.SourceFactKind.assume, accounting.UseRole.assumption_context, accounting.HandlingKind.assumption_incorporated, DirectiveEvidenceKind.none, true },
        .{ accounting.SourceFactKind.runtime_guard, accounting.UseRole.runtime_condition, accounting.HandlingKind.runtime_enforced, DirectiveEvidenceKind.none, true },
        .{ accounting.SourceFactKind.modifies, accounting.UseRole.frame_directive, accounting.HandlingKind.frame_validated, DirectiveEvidenceKind.none, true },
        .{ accounting.SourceFactKind.havoc, accounting.UseRole.state_directive, accounting.HandlingKind.state_effect_incorporated, DirectiveEvidenceKind.none, true },
        .{ accounting.SourceFactKind.modifies, accounting.UseRole.frame_directive, accounting.HandlingKind.frame_validated, DirectiveEvidenceKind.frame_result, false },
        .{ accounting.SourceFactKind.havoc, accounting.UseRole.state_directive, accounting.HandlingKind.state_effect_incorporated, DirectiveEvidenceKind.state_effect, false },
    }) |fixture| {
        var result = try decideDirectiveFixture(allocator, .verified_full, fixture[0], fixture[1], fixture[2], fixture[3], fixture[4]);
        defer result.deinit();
        try noteFailures(&witnesses, result);
    }
    {
        var handlings = [_]accounting.HandlingRecord{ concrete_base.comptime_evidence.handlings[0], concrete_base.comptime_evidence.handlings[1] };
        handlings[0].predicate_event_ids = &.{};
        var manifest = concrete_base;
        manifest.comptime_evidence.handlings = &handlings;
        var result = try decide(allocator, .verified_full, manifest);
        defer result.deinit();
        try noteFailures(&witnesses, result);
    }
    {
        var events = [_]accounting.TraceEvent{ concrete_base.comptime_evidence.folds[0].events[0], concrete_base.comptime_evidence.folds[0].events[1], concrete_base.comptime_evidence.folds[0].events[2], concrete_base.comptime_evidence.folds[0].events[3], concrete_base.comptime_evidence.folds[0].events[4], concrete_base.comptime_evidence.folds[0].events[5], concrete_base.comptime_evidence.folds[0].events[6], concrete_base.comptime_evidence.folds[0].events[7] };
        events[3].predicate_value = false;
        var predicates = [_]accounting.PredicateEvent{ concrete_base.comptime_evidence.predicate_events[0], concrete_base.comptime_evidence.predicate_events[1] };
        predicates[0].value = false;
        var folds = [_]accounting.FoldRecord{concrete_base.comptime_evidence.folds[0]};
        folds[0].events = &events;
        var manifest = concrete_base;
        manifest.comptime_evidence.folds = &folds;
        manifest.comptime_evidence.predicate_events = &predicates;
        var result = try decide(allocator, .verified_full, manifest);
        defer result.deinit();
        try noteFailures(&witnesses, result);
    }
    {
        var events = [_]accounting.TraceEvent{ concrete_base.comptime_evidence.folds[0].events[0], concrete_base.comptime_evidence.folds[0].events[1], concrete_base.comptime_evidence.folds[0].events[2], concrete_base.comptime_evidence.folds[0].events[3], concrete_base.comptime_evidence.folds[0].events[4], concrete_base.comptime_evidence.folds[0].events[5], concrete_base.comptime_evidence.folds[0].events[6], concrete_base.comptime_evidence.folds[0].events[7] };
        events[4].predicate_value = false;
        var predicates = [_]accounting.PredicateEvent{ concrete_base.comptime_evidence.predicate_events[0], concrete_base.comptime_evidence.predicate_events[1] };
        predicates[1].value = false;
        var folds = [_]accounting.FoldRecord{concrete_base.comptime_evidence.folds[0]};
        folds[0].events = &events;
        var manifest = concrete_base;
        manifest.comptime_evidence.folds = &folds;
        manifest.comptime_evidence.predicate_events = &predicates;
        var result = try decide(allocator, .verified_full, manifest);
        defer result.deinit();
        try noteFailures(&witnesses, result);
    }
    {
        var handlings = [_]accounting.HandlingRecord{ concrete_base.comptime_evidence.handlings[0], concrete_base.comptime_evidence.handlings[1] };
        handlings[0].kind = .control_eliminated;
        handlings[0].predicate_event_ids = &.{};
        var manifest = concrete_base;
        manifest.comptime_evidence.handlings = &handlings;
        var result = try decide(allocator, .verified_full, manifest);
        defer result.deinit();
        try noteFailures(&witnesses, result);
    }
    {
        var handlings = [_]accounting.HandlingRecord{ symbolic_base.symbolic.handlings[0], symbolic_base.symbolic.handlings[1] };
        handlings[0].kind = .concrete_true;
        handlings[0].obligation_ids = &.{};
        handlings[0].query_ids = &.{};
        var manifest = symbolic_base;
        manifest.symbolic.handlings = &handlings;
        var result = try decide(allocator, .verified_full, manifest);
        defer result.deinit();
        try noteFailures(&witnesses, result);
    }
    {
        var handlings = [_]accounting.HandlingRecord{ symbolic_base.symbolic.handlings[0], symbolic_base.symbolic.handlings[1] };
        handlings[0].kind = .reduced_scope_excluded;
        handlings[0].obligation_ids = &.{};
        handlings[0].query_ids = &.{};
        var manifest = symbolic_base;
        manifest.symbolic.handlings = &handlings;
        var result = try decide(allocator, .verified_full, manifest);
        defer result.deinit();
        try noteFailures(&witnesses, result);
    }
    {
        var handlings = [_]accounting.HandlingRecord{ symbolic_base.symbolic.handlings[0], symbolic_base.symbolic.handlings[1] };
        handlings[0].kind = .verification_disabled;
        handlings[0].obligation_ids = &.{};
        handlings[0].query_ids = &.{};
        var manifest = symbolic_base;
        manifest.symbolic.handlings = &handlings;
        var result = try decide(allocator, .verified_full, manifest);
        defer result.deinit();
        try noteFailures(&witnesses, result);
    }

    return witnesses;
}

test "every lifecycle-independent failure code has an executable witness" {
    const witnesses = try observeFailureWitnesses(std.testing.allocator);
    inline for (std.meta.fields(FailureCode)) |field| {
        const code: FailureCode = @enumFromInt(field.value);
        const observation = witnesses.rows[@intFromEnum(code)];
        if (!observation.observed) std.debug.print("missing source-accounting failure witness: {s}\n", .{@tagName(code)});
        try std.testing.expect(observation.observed);
        try std.testing.expectEqual(Decision.rejected, observation.decision);
        var contains_target = false;
        for (observation.failures()) |failure| if (failure == code) {
            contains_target = true;
            break;
        };
        try std.testing.expect(contains_target);
    }
}

test "decision is invariant under source row order" {
    const base = missingHandlingFixture();
    const reversed_uses = [_]accounting.SourceFactUse{ base.inventory.uses[1], base.inventory.uses[0] };
    var reversed = base;
    reversed.inventory.uses = &reversed_uses;
    var lhs = try decide(std.testing.allocator, .verified_full, base);
    defer lhs.deinit();
    var rhs = try decide(std.testing.allocator, .verified_full, reversed);
    defer rhs.deinit();
    try std.testing.expectEqualSlices(FailureRow, lhs.failures, rhs.failures);
}

test "accounting report schema and bytes are deterministic" {
    const manifest = missingHandlingFixture();
    var result = try decide(std.testing.allocator, .verified_full, manifest);
    defer result.deinit();
    const first = try renderReport(std.testing.allocator, .verified_full, manifest, result);
    defer std.testing.allocator.free(first);
    const second = try renderReport(std.testing.allocator, .verified_full, manifest, result);
    defer std.testing.allocator.free(second);
    try std.testing.expectEqualStrings(first, second);
    try std.testing.expect(std.mem.containsAtLeast(u8, first, 1, "\"schema_version\": 3"));
    try std.testing.expect(std.mem.containsAtLeast(u8, first, 1, "\"primary_failure\": \"missing_handling\""));
    try std.testing.expect(std.mem.containsAtLeast(u8, first, 1, "\"inventory_sha256\": \""));
    try std.testing.expect(std.mem.containsAtLeast(u8, first, 1, "\"evidence_sha256\": \""));
    try std.testing.expect(std.mem.containsAtLeast(u8, first, 1, "\"failures_by_code\":"));
    try validateReport(std.testing.allocator, .verified_full, manifest, result, first);

    const corrupted = try std.testing.allocator.dupe(u8, first);
    defer std.testing.allocator.free(corrupted);
    const decision_at = std.mem.indexOf(u8, corrupted, "\"decision\": \"rejected\"").?;
    corrupted[decision_at + 13] = 'a';
    try std.testing.expectError(
        error.SourceAccountingReportMismatch,
        validateReport(std.testing.allocator, .verified_full, manifest, result, corrupted),
    );
}

test "source-accounting diagnostic rendering is pinned" {
    const rendered = try renderDiagnostic(std.testing.allocator, .{
        .code = .missing_handling,
        .source = .{ .file = "fixture.ora", .start = 10, .end = 20 },
        .site_id = 1,
        .expansion_id = 2,
        .use_id = 3,
    });
    defer std.testing.allocator.free(rendered);
    try std.testing.expectEqualStrings(
        "formal source accounting rejected: missing_handling at fixture.ora:10-20 site_id=1 expansion_id=2 use_id=3",
        rendered,
    );
}

test "handling policy is total over every closed vocabulary row" {
    @setEvalBranchQuota(100_000);
    inline for (std.meta.fields(accounting.CompilationMode)) |mode_field| {
        const mode: accounting.CompilationMode = @enumFromInt(mode_field.value);
        inline for (std.meta.fields(accounting.FactOrigin)) |origin_field| {
            const origin: accounting.FactOrigin = @enumFromInt(origin_field.value);
            inline for (std.meta.fields(accounting.SourceFactKind)) |kind_field| {
                const kind: accounting.SourceFactKind = @enumFromInt(kind_field.value);
                inline for (std.meta.fields(accounting.UseRole)) |role_field| {
                    const role: accounting.UseRole = @enumFromInt(role_field.value);
                    inline for (std.meta.fields(accounting.HandlingKind)) |handling_field| {
                        const handling: accounting.HandlingKind = @enumFromInt(handling_field.value);
                        inline for (std.meta.fields(accounting.ExpansionDisposition)) |disposition_field| {
                            const disposition: accounting.ExpansionDisposition = @enumFromInt(disposition_field.value);
                            _ = handlingPermitted(mode, origin, kind, role, handling, disposition);
                        }
                    }
                }
            }
        }
    }
    try std.testing.expect(handlingPermitted(.verified_basic, .source_syntax, .loop_invariant, .proof_target, .reduced_scope_excluded, .symbolic));
    try std.testing.expect(!handlingPermitted(.verified_basic, .source_syntax, .requires, .proof_target, .reduced_scope_excluded, .symbolic));
    try std.testing.expect(!handlingPermitted(.verified_full, .source_syntax, .loop_invariant, .proof_target, .reduced_scope_excluded, .symbolic));
    try std.testing.expect(handlingPermitted(.unverified_emit, .source_syntax, .assert, .proof_target, .verification_disabled, .symbolic));
    try std.testing.expect(handlingPermitted(.unverified_emit, .source_syntax, .ensures, .assumption_context, .verification_disabled, .symbolic));
    try std.testing.expect(!handlingPermitted(.unverified_emit, .semantic_generated, .assert, .runtime_condition, .verification_disabled, .symbolic));
}
