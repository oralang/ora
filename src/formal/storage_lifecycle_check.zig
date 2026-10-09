//! Fresh proof production and compilation-owned custody for the lifecycle consumer.
const std = @import("std");
const mlir = @import("mlir_c_api").c;
const z3 = @import("ora_z3_verification");
const data = @import("storage_lifecycle.zig");
const hir = @import("storage_lifecycle_from_mlir.zig");
const lean = @import("storage_lifecycle_to_lean.zig");
const canonical = @import("canonical_encode.zig");
const custody = @import("proof_check.zig");
const prepared = @import("ora_prepared_query_row");

pub const Receipt = struct {
    allocator: std.mem.Allocator,
    native_inputs_sha256: data.Digest,
    queries: []const data.NativeQuery,
    lean: custody.LifecycleCustodyReceipt,

    pub fn deinit(self: *Receipt) void {
        self.allocator.free(self.queries);
        self.lean.deinit();
        self.* = undefined;
    }
};

pub fn produce(allocator: std.mem.Allocator, module: mlir.MlirModule, pass: *z3.VerificationPass, environ: std.process.Environ, stdout: anytype) !Receipt {
    var snapshot = try hir.collect(allocator, module, pass);
    defer snapshot.deinit();
    var projection = try hir.project(allocator, module, pass, snapshot.inventory);
    defer projection.deinit();
    const inputs = try hir.proofInputs(&projection);
    const digest = try inputs.sha256(allocator);
    var targets = std.Io.Writer.Allocating.init(allocator);
    defer targets.deinit();
    var proof = std.Io.Writer.Allocating.init(allocator);
    defer proof.deinit();
    try lean.writeProducerTargets(&targets.writer, allocator, inputs, projection.canonical.set);
    try lean.writeProducerProofs(&proof.writer, allocator, inputs, projection.canonical.set);
    var context = try z3.Z3Context.init(allocator);
    defer context.deinit();
    var adapter = canonical.Adapter.init(&context, allocator, projection.canonical.set);
    const queries = try adapter.checkLifecycle(inputs, pass.timeout_ms);
    errdefer allocator.free(queries);
    for (queries) |query| try query.requireSuccess();
    var expected: usize = 1; // One bootstrap per claim, then every root.
    for (projection.canonical.set.function_summaries) |row| {
        var writes: usize = 0;
        for (row.paths[0].timeline) |event| if (event.payload == .storage_write) {
            writes += 1;
        };
        expected += @max(writes, 1) + 1; // Prefix/frame plus nonvacuity.
    }
    if (queries.len != expected * inputs.slots.len) return error.MissingLifecycleQuery;
    var checked = try custody.checkLifecycleCustody(allocator, module, pass, .{
        .inputs_sha256 = digest,
        .targets_sha256 = data.hash(targets.written()),
        .proof_source = proof.written(),
        .theorem_name = "Ora.Generated.Obligations.Lifecycle.lifecycleProof",
    }, environ, stdout);
    errdefer checked.deinit();
    if (!std.mem.eql(u8, &checked.inputs_sha256, &digest)) return error.LifecycleInputsMismatch;
    try hir.checkFresh(allocator, module, pass, inputs.inventory);
    return .{ .allocator = allocator, .native_inputs_sha256 = digest, .queries = queries, .lean = checked };
}

/// Compilation-owned evidence. No deserializer or external receipt constructor.
/// The session is inert until the separate consumer validates its actual uses.
pub const Session = struct {
    allocator: std.mem.Allocator,
    module: mlir.MlirModule,
    pass: *z3.VerificationPass,
    module_identity: usize,
    pass_identity: usize,
    custody_identity: usize,
    projection: hir.Projection,
    inputs: data.ProofInputs,
    receipt: Receipt,
    expected_queries: []const data.NativeQueryKey,
    native_sha256: data.Digest,
    proof_sha256: data.Digest,
    checker_sha256: data.Digest,
    axiom_count: usize,
    uses: ?[]const prepared.LifecycleQueryUse = null,
    uses_sha256: ?data.Digest = null,

    pub fn init(allocator: std.mem.Allocator, module: mlir.MlirModule, pass: *z3.VerificationPass, environ: std.process.Environ, stdout: anytype) !Session {
        var snapshot = try hir.collect(allocator, module, pass);
        defer snapshot.deinit();
        var projection = try hir.project(allocator, module, pass, snapshot.inventory);
        errdefer projection.deinit();
        const inputs = try hir.proofInputs(&projection);
        var receipt = try produce(allocator, module, pass, environ, stdout);
        errdefer receipt.deinit();
        const arena = projection.snapshot.arena.allocator();
        var expected: std.ArrayList(data.NativeQueryKey) = .empty;
        for (inputs.inventory.claims, 0..) |claim, ci| {
            try expected.append(arena, .{ .role = .bootstrap, .claim_index = @intCast(ci) });
            for (projection.canonical.set.function_summaries, inputs.functions) |row, binding| {
                var writes: usize = 0;
                for (row.paths[0].timeline) |event| if (event.payload == .storage_write) {
                    try expected.append(arena, .{ .role = if (std.mem.eql(u8, claim.root, event.payload.storage_write.location.root)) .prefix else .frame, .claim_index = @intCast(ci), .function_id = row.id, .event_id = event.id, .function_vc_sha256 = binding.function_vc_sha256 });
                    writes += 1;
                };
                if (writes == 0) try expected.append(arena, .{ .role = .frame, .claim_index = @intCast(ci), .function_id = row.id, .function_vc_sha256 = binding.function_vc_sha256 });
                try expected.append(arena, .{ .role = .nonvacuity, .claim_index = @intCast(ci), .function_id = row.id, .function_vc_sha256 = binding.function_vc_sha256 });
            }
        }
        var session: Session = .{
            .allocator = allocator,
            .module = module,
            .pass = pass,
            .module_identity = @intFromPtr(module.ptr),
            .pass_identity = @intFromPtr(pass),
            .custody_identity = @intFromPtr(receipt.lean.checker_source.ptr),
            .projection = projection,
            .inputs = inputs,
            .receipt = receipt,
            .expected_queries = expected.items,
            .native_sha256 = try nativeDigest(allocator, receipt.queries),
            .proof_sha256 = receipt.lean.proof_sha256,
            .checker_sha256 = receipt.lean.checker_sha256,
            .axiom_count = receipt.lean.axioms.len,
        };
        try session.validate();
        return session;
    }

    pub fn deinit(self: *Session) void {
        self.receipt.deinit();
        self.projection.deinit();
        self.* = undefined;
    }

    pub fn validate(self: *Session) !void {
        if (@intFromPtr(self.module.ptr) != self.module_identity or @intFromPtr(self.pass) != self.pass_identity) return error.ForeignLifecycleCompilation;
        if (@intFromPtr(self.receipt.lean.checker_source.ptr) != self.custody_identity) return error.ForeignLifecycleCustody;
        try hir.checkFresh(self.allocator, self.module, self.pass, self.inputs.inventory);
        const digest = try self.inputs.sha256(self.allocator);
        if (!std.mem.eql(u8, &digest, &self.receipt.native_inputs_sha256)) return error.LifecycleNativeInputsMismatch;
        if (!std.mem.eql(u8, &digest, &self.receipt.lean.inputs_sha256)) return error.LifecycleInputsMismatch;
        try data.validateNativeQueries(self.expected_queries, self.receipt.queries);
        if (!std.mem.eql(u8, &self.native_sha256, &(try nativeDigest(self.allocator, self.receipt.queries)))) return error.LifecycleNativeCustodyMismatch;
        var targets = std.Io.Writer.Allocating.init(self.allocator);
        defer targets.deinit();
        try lean.writeProducerTargets(&targets.writer, self.allocator, self.inputs, self.projection.canonical.set);
        if (!std.mem.eql(u8, &self.receipt.lean.targets_sha256, &data.hash(targets.written()))) return error.LifecycleTargetsMismatch;
        if (std.mem.allEqual(u8, &self.proof_sha256, 0) or !std.mem.eql(u8, &self.proof_sha256, &self.receipt.lean.proof_sha256)) return error.LifecycleProofMismatch;
        if (!std.mem.eql(u8, &self.checker_sha256, &self.receipt.lean.checker_sha256) or
            !std.mem.eql(u8, &self.checker_sha256, &data.hash(self.receipt.lean.checker_source))) return error.LifecycleCheckerMismatch;
        if (self.receipt.lean.axioms.len != self.axiom_count) return error.LifecycleAxiomMismatch;
        for (self.receipt.lean.axioms, 0..) |axiom, i| {
            if (!std.mem.eql(u8, axiom, "propext") and !std.mem.eql(u8, axiom, "Quot.sound")) return error.LifecycleAxiomMismatch;
            for (self.receipt.lean.axioms[0..i]) |prior| if (std.mem.eql(u8, prior, axiom)) return error.LifecycleAxiomMismatch;
        }
        if (self.uses) |uses| {
            const original = self.uses_sha256 orelse return error.MissingLifecycleUse;
            if (!std.mem.eql(u8, &original, &(try evidenceDigest(self.allocator, uses)))) return error.LifecycleUseMismatch;
        } else if (self.uses_sha256 != null) return error.MissingLifecycleUse;
    }

    pub fn dependency(self: *const Session, binding: data.FunctionBinding) prepared.LifecycleDependency {
        return .{ .inputs_sha256 = self.receipt.lean.inputs_sha256, .module_sha256 = self.inputs.inventory.module_sha256, .function_vc_sha256 = binding.function_vc_sha256, .function_id = binding.function_id, .function_ordinal = binding.function_ordinal, .owner_ordinal = self.inputs.slots[0].owner_ordinal, .claim_count = @intCast(self.inputs.slots.len) };
    }

    pub fn validateDependency(self: *const Session, used: prepared.LifecycleDependency) !void {
        for (self.inputs.functions) |binding| {
            if (std.meta.eql(self.dependency(binding), used)) return;
        }
        return error.LifecycleEntryIdentityMismatch;
    }

    pub fn recordUses(self: *Session, uses: []const prepared.LifecycleQueryUse) !void {
        if (uses.len == 0) return error.MissingLifecycleUse;
        for (uses, 0..) |use, i| {
            try self.validateDependency(use.dependency);
            if (use.smtlib_hash == 0 or use.constraint_count < use.dependency.claim_count) return error.LifecycleQueryMissing;
            for (uses[0..i]) |prior| if (prior.query_index == use.query_index) return error.DuplicateLifecycleUse;
        }
        if (self.uses) |expected| {
            if (uses.len != expected.len) return error.MissingLifecycleUse;
            for (uses, expected) |actual, original| if (!std.meta.eql(actual, original)) return error.LifecycleUseMismatch;
        } else {
            self.uses = try self.projection.snapshot.arena.allocator().dupe(prepared.LifecycleQueryUse, uses);
            self.uses_sha256 = try evidenceDigest(self.allocator, uses);
        }
    }
};

fn nativeDigest(allocator: std.mem.Allocator, queries: []const data.NativeQuery) !data.Digest {
    return evidenceDigest(allocator, queries);
}

fn evidenceDigest(allocator: std.mem.Allocator, evidence: anytype) !data.Digest {
    var out = std.Io.Writer.Allocating.init(allocator);
    defer out.deinit();
    try std.json.Stringify.value(evidence, .{}, &out.writer);
    return data.hash(out.written());
}
