//! Diagnostic lifecycle inputs. No verifier consumes this inventory as a proof.
const std = @import("std");
const builtin = @import("ora_types").builtin;

pub const Digest = [32]u8;
pub const Bounds = struct {
    kind: enum { MinValue, MaxValue, InRange },
    minimum: u256,
    maximum: u256,
};
pub const Refusal = enum { unsupported_type, missing_type, width_mismatch, unsupported_initialization };
pub const Claim = struct {
    owner: []const u8,
    root: []const u8,
    ordinal: u32,
    type_id: ?u32,
    bounds: ?Bounds,
    zero_base: bool,
    refusal: ?Refusal,
};
pub const Event = struct {
    ordinal: u32,
    operation: []const u8,
    kind: enum { write, call, unknown },
    target: ?[]const u8,
    refusal: ?enum { unresolved_call, external_interference, unknown_effect } = null,
    hir_sha256: Digest,
};
pub const Function = struct {
    owner: []const u8,
    module_path: []const u8,
    symbol: []const u8,
    ordinal: u32,
    runtime: bool,
    constructor: bool,
    hir_sha256: Digest,
    events: []const Event,
};

pub const FunctionBinding = struct {
    function_ordinal: u32,
    function_id: u32,
    function_vc_sha256: Digest,
};

pub const NativeQuery = struct {
    role: enum { bootstrap, prefix, frame, nonvacuity },
    claim_index: u32,
    function_id: ?u32 = null,
    event_id: ?u32 = null,
    function_vc_sha256: ?Digest = null,
    status: enum { sat, unsat, unknown },
    base_status: enum { sat, unsat, unknown },
    smtlib_hash: u64,
    constraint_count: u32,

    pub fn key(self: NativeQuery) NativeQueryKey {
        return .{ .role = self.role, .claim_index = self.claim_index, .function_id = self.function_id, .event_id = self.event_id, .function_vc_sha256 = self.function_vc_sha256 };
    }

    pub fn requireSuccess(self: NativeQuery) !void {
        if (self.smtlib_hash == 0 or self.constraint_count == 0) return error.LifecycleQueryMissing;
        if (self.status == .unknown or self.base_status == .unknown) return error.LifecycleQueryUnknown;
        if (self.base_status != .sat) return error.LifecycleQueryVacuous;
        if ((self.role == .nonvacuity and self.status != .sat) or
            (self.role != .nonvacuity and self.status != .unsat)) return error.LifecycleQueryFailed;
    }
};

pub const NativeQueryKey = struct {
    role: @FieldType(NativeQuery, "role"),
    claim_index: u32,
    function_id: ?u32 = null,
    event_id: ?u32 = null,
    function_vc_sha256: ?Digest = null,
};

/// Evidence owned by one live consumer session, never decoded from a report.
/// Final authority additionally checks the ordinary query and source joins.
pub const Evidence = struct {
    inputs_sha256: Digest,
    native_inputs_sha256: Digest,
    lean_inputs_sha256: Digest,
    module_sha256: Digest,
    targets_sha256: Digest,
    proof_sha256: Digest,
    checker_sha256: Digest,
    axioms: []const []const u8,
    native_keys: []const NativeQueryKey,
    native_queries: []const NativeQuery,
    functions: []const FunctionBinding,
    slots: []const SlotBinding,
    expected_uses: []const @import("ora_prepared_query_row").LifecycleQueryUse,
    checked_uses: []const CheckedUse,
    retained_checks: bool,

    pub const CheckedUse = struct {
        use: @import("ora_prepared_query_row").LifecycleQueryUse,
        status: @FieldType(NativeQuery, "status"),
        vacuous: bool,
        vacuity_unknown: bool,
        degraded: bool,
        runtime_input: bool,
    };

    pub fn validate(self: Evidence) !void {
        if (!self.retained_checks) return error.LifecycleChecksNotRetained;
        if (emptyDigest(self.inputs_sha256) or emptyDigest(self.module_sha256) or
            !std.mem.eql(u8, &self.inputs_sha256, &self.native_inputs_sha256) or
            !std.mem.eql(u8, &self.inputs_sha256, &self.lean_inputs_sha256)) return error.LifecycleInputsMismatch;
        if (emptyDigest(self.targets_sha256) or emptyDigest(self.proof_sha256) or emptyDigest(self.checker_sha256)) return error.MissingLifecycleCustody;
        for (self.axioms, 0..) |axiom, i| {
            if (!std.mem.eql(u8, axiom, "propext") and !std.mem.eql(u8, axiom, "Quot.sound")) return error.LifecycleAxiomMismatch;
            for (self.axioms[0..i]) |prior| if (std.mem.eql(u8, prior, axiom)) return error.LifecycleAxiomMismatch;
        }
        try validateNativeQueries(self.native_keys, self.native_queries);
        if (self.functions.len == 0 or self.slots.len == 0) return error.IncompleteLifecycleProofInputs;
        if (self.expected_uses.len == 0 or self.expected_uses.len != self.checked_uses.len) return error.MissingLifecycleUse;
        for (self.expected_uses, self.checked_uses, 0..) |expected, checked, i| {
            if (!std.meta.eql(expected, checked.use)) return error.LifecycleUseMismatch;
            for (self.expected_uses[0..i]) |prior| if (prior.query_index == expected.query_index) return error.DuplicateLifecycleUse;
            const dependency = expected.dependency;
            if (!std.mem.eql(u8, &dependency.inputs_sha256, &self.inputs_sha256) or
                !std.mem.eql(u8, &dependency.module_sha256, &self.module_sha256) or
                emptyDigest(dependency.function_vc_sha256) or dependency.claim_count != self.slots.len or
                dependency.owner_ordinal != self.slots[0].owner_ordinal) return error.LifecycleEntryIdentityMismatch;
            for (self.functions) |function| {
                if (function.function_id == dependency.function_id and function.function_ordinal == dependency.function_ordinal and
                    std.mem.eql(u8, &function.function_vc_sha256, &dependency.function_vc_sha256)) break;
            } else return error.LifecycleEntryIdentityMismatch;
            if (expected.smtlib_hash == 0 or expected.constraint_count < dependency.claim_count) return error.LifecycleQueryMissing;
            if (checked.degraded) return error.LifecycleQueryDegraded;
            if (checked.vacuous) return error.LifecycleQueryVacuous;
            if (checked.vacuity_unknown or checked.status == .unknown) return error.LifecycleQueryUnknown;
            switch (expected.kind) {
                .base, .guard_satisfy => if (checked.status != .sat) return error.LifecycleQueryVacuous,
                .obligation => if (checked.status != .unsat) return error.LifecycleQueryFailed,
                .guard_violate => if (checked.status != .unsat and !(checked.status == .sat and checked.runtime_input)) return error.LifecycleQueryFailed,
            }
        }
    }
};

pub fn validateNativeQueries(expected: []const NativeQueryKey, queries: []const NativeQuery) !void {
    if (expected.len == 0 or queries.len != expected.len) return error.MissingLifecycleQuery;
    for (expected, 0..) |key, i| {
        for (expected[0..i]) |prior| if (std.meta.eql(prior, key)) return error.DuplicateLifecycleQuery;
        var found: ?NativeQuery = null;
        for (queries) |query| if (std.meta.eql(query.key(), key)) {
            if (found != null) return error.DuplicateLifecycleQuery;
            found = query;
        };
        try (found orelse return error.ForeignLifecycleQuery).requireSuccess();
    }
}

/// Process-independent selectors, rebuilt from actual HIR operations before
/// checking proofs. These rows grant no authority to ordinary FunctionVCs.
pub const SlotBinding = struct {
    claim_index: u32,
    global_ordinal: u32,
    owner_ordinal: u32,
};
pub const GuardBinding = struct {
    function_id: u32,
    event_id: u32,
    operation_ordinal: u32,
    hir_sha256: Digest,
};
pub const ProofInputs = struct {
    inventory: Inventory,
    functions: []const FunctionBinding,
    slots: []const SlotBinding,
    guards: []const GuardBinding,

    pub fn canonical(self: ProofInputs, allocator: std.mem.Allocator) ![]u8 {
        try validateBindings(self.inventory, self.functions);
        if (self.slots.len == 0 or self.slots.len != self.inventory.claims.len or self.functions.len == 0)
            return error.IncompleteLifecycleProofInputs;
        for (self.slots, self.inventory.claims, 0..) |slot, claim, index| {
            if (slot.claim_index != index or slot.global_ordinal != claim.ordinal or
                slot.owner_ordinal != self.slots[0].owner_ordinal or claim.refusal != null)
                return error.InvalidLifecycleSlotBinding;
            const info = builtin.integerInfoByComptimeTypeId(claim.type_id.?) orelse return error.InvalidLifecycleType;
            if (info.signed) return error.UnsupportedLifecycleClaim;
            if (claim.bounds.?.minimum != 0) return error.InvalidLifecycleBootstrap;
        }
        for (self.guards, 0..) |guard, index| {
            if (emptyDigest(guard.hir_sha256)) return error.InvalidLifecycleGuard;
            for (self.guards[0..index]) |prior| if (prior.operation_ordinal == guard.operation_ordinal or
                (prior.function_id == guard.function_id and prior.event_id == guard.event_id))
                return error.DuplicateLifecycleGuard;
            for (self.functions) |function| {
                if (function.function_id == guard.function_id) break;
            } else return error.ForeignLifecycleGuard;
        }
        var output = std.Io.Writer.Allocating.init(allocator);
        errdefer output.deinit();
        try output.writer.writeAll("StorageLifecycleProofInputs-v1\n");
        try std.json.Stringify.value(self, .{}, &output.writer);
        return output.toOwnedSlice();
    }

    pub fn sha256(self: ProofInputs, allocator: std.mem.Allocator) !Digest {
        const bytes = try self.canonical(allocator);
        defer allocator.free(bytes);
        return hash(bytes);
    }
};

pub fn validateBindings(inventory: Inventory, bindings: []const FunctionBinding) !void {
    try inventory.validate();
    for (bindings, 0..) |binding, index| {
        if (binding.function_id == 0 or emptyDigest(binding.function_vc_sha256)) return error.InvalidLifecycleBinding;
        for (bindings[0..index]) |previous| if (previous.function_id == binding.function_id or previous.function_ordinal == binding.function_ordinal)
            return error.DuplicateLifecycleBinding;
        const found = for (inventory.functions) |function| {
            if (function.ordinal == binding.function_ordinal) break true;
        } else false;
        if (!found) return error.ForeignLifecycleBinding;
    }
    for (inventory.functions) |function| {
        if (!function.runtime and !function.constructor) continue;
        for (bindings) |binding| {
            if (binding.function_ordinal == function.ordinal) break;
        } else return error.MissingLifecycleBinding;
    }
}

pub const Inventory = struct {
    module_sha256: Digest,
    claims: []const Claim,
    functions: []const Function,

    pub fn validate(self: Inventory) !void {
        if (emptyDigest(self.module_sha256)) return error.InvalidLifecycleDigest;
        for (self.claims, 0..) |claim, index| {
            if (claim.owner.len == 0 or claim.root.len == 0) return error.InvalidLifecycleOwner;
            for (self.claims[0..index]) |previous| {
                if (previous.ordinal == claim.ordinal or
                    (std.mem.eql(u8, previous.owner, claim.owner) and std.mem.eql(u8, previous.root, claim.root))) return error.DuplicateLifecycleClaim;
            }
            if (claim.refusal != null) continue;
            const info = builtin.integerInfoByComptimeTypeId(claim.type_id orelse return error.InvalidLifecycleType) orelse return error.InvalidLifecycleType;
            const bounds = claim.bounds orelse return error.InvalidLifecycleBounds;
            if (!claim.zero_base) return error.InvalidLifecycleInitialization;
            if (info.width < 256 and (bounds.minimum >> @as(u8, @intCast(info.width)) != 0 or bounds.maximum >> @as(u8, @intCast(info.width)) != 0)) return error.InvalidLifecycleBounds;
        }
        for (self.functions, 0..) |function, index| {
            if (function.owner.len == 0 or function.module_path.len == 0 or function.symbol.len == 0) return error.InvalidLifecycleOwner;
            if (emptyDigest(function.hir_sha256)) return error.InvalidLifecycleDigest;
            for (self.functions[0..index]) |previous| {
                if (previous.ordinal == function.ordinal or std.mem.eql(u8, previous.symbol, function.symbol)) return error.AmbiguousLifecycleFunction;
            }
            for (function.events, 0..) |event, event_index| {
                if (event.operation.len == 0 or emptyDigest(event.hir_sha256)) return error.InvalidLifecycleEvent;
                if (event_index != 0 and function.events[event_index - 1].ordinal >= event.ordinal) return error.DuplicateLifecycleEvent;
            }
        }
    }

    pub fn canonical(self: Inventory, allocator: std.mem.Allocator) ![]u8 {
        try self.validate();
        var output = std.Io.Writer.Allocating.init(allocator);
        errdefer output.deinit();
        try output.writer.writeAll("StorageLifecycleInventory-v1\n");
        try std.json.Stringify.value(self, .{}, &output.writer);
        return output.toOwnedSlice();
    }

    pub fn sha256(self: Inventory, allocator: std.mem.Allocator) !Digest {
        const bytes = try self.canonical(allocator);
        defer allocator.free(bytes);
        return hash(bytes);
    }
};

pub fn hash(bytes: []const u8) Digest {
    var result: Digest = undefined;
    std.crypto.hash.sha2.Sha256.hash(bytes, &result, .{});
    return result;
}

fn emptyDigest(digest: Digest) bool {
    return std.mem.allEqual(u8, &digest, 0);
}
