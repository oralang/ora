//! Compilation-owned entry binding and freshly checked lifecycle consumption.
const std = @import("std");
const mlir = @import("mlir_c_api").c;
const z3 = @import("ora_z3_verification");
const builtin = @import("ora_types").builtin;
const prepared = @import("ora_prepared_query_row");
const checked = @import("storage_lifecycle_check.zig");
const obligation = @import("obligation.zig");
const data = @import("storage_lifecycle.zig");
const hir = @import("storage_lifecycle_from_mlir.zig");
const lean = @import("storage_lifecycle_to_lean.zig");

/// Structural refusal gives zero facts. Once production starts, every native
/// or kernel failure propagates; it cannot retry as an unassisted success.
pub fn prepare(allocator: std.mem.Allocator, module: mlir.MlirModule, pass: *z3.VerificationPass, environ: std.process.Environ, stdout: anytype) !?checked.Session {
    structuralPreflight(allocator, module, pass) catch |err| switch (err) {
        error.MissingLifecycleClaim,
        error.UnsupportedLifecycleClaim,
        error.InvalidLifecycleBootstrap,
        error.LifecycleProjectionUnsupported,
        error.UnsupportedLifecycleRoot,
        error.AmbiguousLifecycleOwner,
        error.AmbiguousLifecycleFunction,
        error.UnsupportedLifecycleEffect,
        error.UnsupportedLifecycleLocation,
        error.UnsupportedLifecycleGuard,
        error.UnsupportedLifecycleEvent,
        error.UnsupportedLifecycleWriter,
        error.UnsupportedLifecycleParameter,
        error.UnsupportedLifecycleFormula,
        error.MissingLifecycleRetainedGuard,
        error.InvalidCallSignature,
        error.UnauthenticatedLifecycleContractInvariant,
        error.UnauthenticatedLifecycleRequires,
        error.LifecycleEntryStoreChanged,
        => return null,
        else => return err,
    };
    return try checked.Session.init(allocator, module, pass, environ, stdout);
}

fn structuralPreflight(allocator: std.mem.Allocator, module: mlir.MlirModule, pass: *z3.VerificationPass) !void {
    var snapshot = try hir.collect(allocator, module, pass);
    defer snapshot.deinit();
    if (snapshot.inventory.claims.len == 0) return error.MissingLifecycleClaim;
    for (snapshot.inventory.functions) |function| {
        if (function.runtime or function.constructor) break;
    } else return error.UnsupportedLifecycleRoot;
    for (snapshot.inventory.claims) |claim| {
        if (claim.refusal != null) return error.UnsupportedLifecycleClaim;
        const info = builtin.integerInfoByComptimeTypeId(claim.type_id.?) orelse return error.InvalidLifecycleType;
        if (info.signed) return error.UnsupportedLifecycleClaim;
        if (claim.bounds.?.minimum != 0) return error.InvalidLifecycleBootstrap;
    }
    var projection = try hir.project(allocator, module, pass, snapshot.inventory);
    defer projection.deinit();
    const inputs = try hir.proofInputs(&projection);
    var source = std.Io.Writer.Allocating.init(allocator);
    defer source.deinit();
    try lean.writeProducerProofs(&source.writer, allocator, inputs, projection.canonical.set);
}

pub const Consumer = struct {
    session: *checked.Session,

    pub fn provider(self: *Consumer) z3.LifecycleProvider {
        return .{ .data = self, .begin = begin, .entry = entry, .record = record };
    }

    /// Build audit data only after fresh native, kernel and exact-use checks.
    pub fn evidence(self: *Consumer, rows: []const z3.PreparedQueryManifestRow, set: obligation.ObligationSet, retained_checks: bool) !data.Evidence {
        try self.validateRows(rows, set, retained_checks);
        const session = self.session;
        const uses = session.uses.?;
        const checked_uses = try session.projection.snapshot.arena.allocator().alloc(data.Evidence.CheckedUse, uses.len);
        for (uses, checked_uses) |use, *checked_use| {
            const row = rows[use.query_index];
            var runtime_input = false;
            for (set.queries) |query| if (query.lifecycle_use) |query_use| {
                if (std.meta.eql(query_use, use)) runtime_input = query.artifact_policy == .runtime_checked_input;
            };
            checked_use.* = .{ .use = use, .status = switch (row.result_status.?) {
                .sat => .sat,
                .unsat => .unsat,
                .unknown => .unknown,
            }, .vacuous = row.vacuous, .vacuity_unknown = row.vacuity_unknown, .degraded = row.verified_with_caveats, .runtime_input = runtime_input };
        }
        const result: data.Evidence = .{
            .inputs_sha256 = try session.inputs.sha256(session.allocator),
            .native_inputs_sha256 = session.receipt.native_inputs_sha256,
            .lean_inputs_sha256 = session.receipt.lean.inputs_sha256,
            .module_sha256 = session.inputs.inventory.module_sha256,
            .targets_sha256 = session.receipt.lean.targets_sha256,
            .proof_sha256 = session.receipt.lean.proof_sha256,
            .checker_sha256 = session.receipt.lean.checker_sha256,
            .axioms = session.receipt.lean.axioms,
            .native_keys = session.expected_queries,
            .native_queries = session.receipt.queries,
            .functions = session.inputs.functions,
            .slots = session.inputs.slots,
            .expected_uses = uses,
            .checked_uses = checked_uses,
            .retained_checks = retained_checks,
        };
        try result.validate();
        return result;
    }

    /// Compare independently recorded uses with the actual solver rows and the
    /// ordinary exact-join output. An unmatched UNSAT row is never sufficient.
    pub fn validateRows(self: *Consumer, rows: []const z3.PreparedQueryManifestRow, set: obligation.ObligationSet, retained_checks: bool) !void {
        try self.session.validate();
        if (!retained_checks) return error.LifecycleChecksNotRetained;
        const expected = self.session.uses orelse return error.MissingLifecycleUse;
        if (set.lifecycle_uses.len != expected.len) return error.MissingLifecycleUse;
        for (set.lifecycle_uses, expected) |used, original| {
            if (!std.meta.eql(used, original)) return error.LifecycleUseMismatch;
        }
        var found: usize = 0;
        for (rows, 0..) |row, index| {
            const use = row.lifecycle_use orelse {
                if (row.lifecycle_dependency != null) return error.MissingLifecycleUse;
                continue;
            };
            if (found >= expected.len or !std.meta.eql(use, expected[found])) return error.LifecycleUseMismatch;
            found += 1;
            if (use.query_index != index or !std.meta.eql(row.lifecycle_dependency, @as(?prepared.LifecycleDependency, use.dependency)) or
                row.smtlib_hash != use.smtlib_hash or row.constraint_count != use.constraint_count or
                row.formal_query_id != use.formal_query_id or row.accounting_query_id != use.accounting_query_id) return error.LifecycleQueryIdentityMismatch;
            const kind: @FieldType(prepared.LifecycleQueryUse, "kind") = switch (row.kind) {
                .Base => .base,
                .Obligation => .obligation,
                .GuardSatisfy => .guard_satisfy,
                .GuardViolate => .guard_violate,
                else => return error.UnsupportedLifecycleQuery,
            };
            if (kind != use.kind) return error.LifecycleQueryIdentityMismatch;
            const owner = for (self.session.inputs.inventory.functions) |function| {
                if (function.ordinal == use.dependency.function_ordinal) break function;
            } else return error.ForeignLifecycleFunction;
            if (!std.mem.eql(u8, row.function_name, owner.symbol)) return error.ForeignLifecycleFunction;
            if (row.verified_with_caveats) return error.LifecycleQueryDegraded;
            if (row.vacuous) return error.LifecycleQueryVacuous;
            if (row.vacuity_unknown or row.result_status == .unknown) return error.LifecycleQueryUnknown;
            const status = row.result_status orelse return error.LifecycleQueryMissing;
            if (kind == .base) {
                if (status != .sat) return error.LifecycleQueryVacuous;
                continue;
            }
            if (kind != .guard_satisfy and (row.formal_match_status != .matched or row.formal_query_id == null)) return error.LifecycleSourceBindingMissing;
            var matched: ?obligation.VerificationQuery = null;
            for (set.queries) |query| if (query.lifecycle_use) |query_use| {
                if (!std.meta.eql(query_use, use)) continue;
                if (matched != null) return error.DuplicateLifecycleQuery;
                matched = query;
            };
            const query = matched orelse return error.LifecycleSourceBindingMissing;
            if (query.backend != .z3 or query.loop_summary_id != null or query.function_summary_id != null or
                query.proof_requirement != .backend_result or query.artifact_policy == .diagnostic_only or
                query.smtlib_hash != use.smtlib_hash or query.constraint_count != use.constraint_count or
                query.owner != .function or !std.mem.eql(u8, query.owner.function.name, row.function_name) or
                !std.mem.eql(u8, query.source.file orelse "", row.file) or query.source.line != row.line or query.source.column != row.column)
                return error.LifecycleQueryIdentityMismatch;
            if (kind == .obligation and (row.annotation_pure or query.canonical_smt_annotation_pure)) return error.LifecycleCanonicalQueryUnsupported;
            const result = query.result orelse return error.LifecycleQueryMissing;
            if (result.degraded) return error.LifecycleQueryDegraded;
            if (result.vacuous) return error.LifecycleQueryVacuous;
            if (result.vacuity_unknown or result.status == .unknown) return error.LifecycleQueryUnknown;
            const expected_status: obligation.VerificationQueryStatus = switch (status) {
                .sat => .sat,
                .unsat => .unsat,
                .unknown => unreachable,
            };
            if (result.status != expected_status) return error.LifecycleQueryIdentityMismatch;
            if (kind == .guard_satisfy) {
                if (status != .sat) return error.LifecycleQueryVacuous;
            } else if (status != .unsat) {
                if (!(kind == .guard_violate and status == .sat and query.artifact_policy == .runtime_checked_input)) return error.LifecycleQueryFailed;
            }
        }
        if (found != expected.len) return error.MissingLifecycleUse;
        for (set.queries) |query| if (query.lifecycle_use) |use| {
            for (expected) |original| {
                if (std.meta.eql(original, use)) break;
            } else return error.ForeignLifecycleUse;
        };
        if (set.hasBlockingDiagnostic()) return error.LifecycleSourceBindingMissing;
    }

    pub fn writeReport(self: *Consumer, file_path: []const u8, output_dir: ?[]const u8, evidence_value: data.Evidence) !void {
        try self.session.validate();
        try evidence_value.validate();
        const allocator = self.session.allocator;
        var out = std.Io.Writer.Allocating.init(allocator);
        defer out.deinit();
        try std.json.Stringify.value(.{
            .schema_version = @as(u32, 1),
            .source = file_path,
            .inputs = self.session.inputs,
            .evidence = evidence_value,
        }, .{}, &out.writer);
        const filename = try @import("shared/artifact_catalog.zig").filename(allocator, std.fs.path.stem(file_path), .storage_lifecycle_report);
        defer allocator.free(filename);
        const directory = output_dir orelse ".";
        const io = std.Io.Threaded.global_single_threaded.io();
        try std.Io.Dir.cwd().createDirPath(io, directory);
        const path = try std.fs.path.join(allocator, &.{ directory, filename });
        defer allocator.free(path);
        try std.Io.Dir.cwd().writeFile(io, .{ .sub_path = path, .data = out.written() });
    }

    fn get(raw: *anyopaque) *Consumer {
        return @ptrCast(@alignCast(raw));
    }

    fn begin(raw: *anyopaque) !void {
        try get(raw).session.validate();
    }

    fn entry(raw: *anyopaque, encoder: *z3.CertifiedCallEncoder, function: mlir.MlirOperation) !z3.LifecycleEntry {
        const session = get(raw).session;
        const operations = session.projection.snapshot.operations;
        const binding = for (session.inputs.functions) |candidate| {
            if (operations[candidate.function_ordinal].ptr == function.ptr) break candidate;
        } else return error.ForeignLifecycleFunction;
        const arena = session.projection.snapshot.arena.allocator();
        const values = try arena.alloc(z3.z3_c.Z3_ast, session.inputs.slots.len);
        const facts = try arena.alloc(z3.z3_c.Z3_ast, session.inputs.slots.len);
        for (session.inputs.slots, values, facts) |slot, *value, *fact| {
            const global = operations[slot.global_ordinal];
            const owner = operations[slot.owner_ordinal];
            if (mlir.mlirOperationGetParentOperation(global).ptr != owner.ptr or
                mlir.mlirOperationGetParentOperation(function).ptr != owner.ptr) return error.InvalidLifecycleOwner;
            const claim = session.inputs.inventory.claims[slot.claim_index];
            const info = builtin.integerInfoByComptimeTypeId(claim.type_id orelse return error.InvalidLifecycleType) orelse return error.InvalidLifecycleType;
            if (info.signed) return error.InvalidLifecycleType;
            const attribute = mlir.oraOperationGetAttributeByName(global, .{ .data = "type", .length = 4 });
            if (mlir.oraAttributeIsNull(attribute) or !mlir.mlirAttributeIsAType(attribute)) return error.InvalidLifecycleType;
            const bounds = claim.bounds orelse return error.InvalidLifecycleBounds;
            const native = try encoder.lifecycleEntry(claim.root, mlir.oraTypeAttrGetValue(attribute), info.width, bounds.minimum, bounds.maximum);
            value.* = native.value;
            fact.* = native.fact;
        }
        return .{ .dependency = session.dependency(binding), .values = values, .facts = facts };
    }

    fn record(raw: *anyopaque, uses: []const prepared.LifecycleQueryUse) !void {
        try get(raw).session.recordUses(uses);
    }
};
