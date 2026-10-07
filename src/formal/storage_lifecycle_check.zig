//! Proof production only. No CLI or artifact-authority consumer is installed.
const std = @import("std");
const mlir = @import("mlir_c_api").c;
const z3 = @import("ora_z3_verification");
const data = @import("storage_lifecycle.zig");
const hir = @import("storage_lifecycle_from_mlir.zig");
const lean = @import("storage_lifecycle_to_lean.zig");
const canonical = @import("canonical_encode.zig");
const custody = @import("proof_check.zig");

pub const Receipt = struct {
    allocator: std.mem.Allocator,
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
    return .{ .allocator = allocator, .queries = queries, .lean = checked };
}
