// Test root: discovers all compiler tests via per-subsystem files.
// Helpers live in compiler.test.helpers.zig.

comptime {
    _ = @import("compiler.test.abi.zig");
    _ = @import("compiler.test.comptime.zig");
    _ = @import("compiler.test.debug_artifacts.zig");
    _ = @import("compiler.test.diagnostics.zig");
    _ = @import("compiler.test.formal.zig");
    _ = @import("compiler.test.hir.zig");
    _ = @import("compiler.test.match.zig");
    _ = @import("compiler.test.misc.zig");
    _ = @import("compiler.test.oratosir.zig");
    _ = @import("compiler.test.sema_infra.zig");
    _ = @import("compiler.test.sema_regions.zig");
    _ = @import("compiler.test.sema_verify.zig");
    _ = @import("compiler.test.syntax.zig");
    _ = @import("compiler.test.traits.zig");
    _ = @import("compiler.test.verification.zig");

    // Formal-lane source files carry in-file tests (pins, matrices, fixtures).
    // Zig only collects tests from files referenced in a test/comptime block —
    // top-level imports with call-site usage do NOT collect them. Several of
    // these files' tests were orphaned (never built, never run) until this
    // block was added; keep every test-bearing formal file listed here.
    _ = @import("formal/obligation.zig");
    _ = @import("formal/obligation_dump.zig");
    _ = @import("formal/obligation_from_mlir.zig");
    _ = @import("formal/obligation_from_z3.zig");
    _ = @import("formal/obligation_to_lean.zig");
    _ = @import("formal/obligation_to_z3.zig");
    _ = @import("formal/canonical_types.zig");
    _ = @import("formal/canonical_support.zig");
    _ = @import("formal/canonical_encode.zig");
    _ = @import("formal/canonical_z3_measure.zig");
    _ = @import("formal/emit_obligation_totality_snapshot.zig");
    _ = @import("formal/dispatcher_table_gate.zig");
    _ = @import("formal/dispatcher_table_rows.zig");
    _ = @import("formal/kernel/registry.zig");
    _ = @import("formal/loop_census.zig");
    _ = @import("formal/shared/artifact_catalog.zig");
    _ = @import("formal/userland/coordinator.zig");
    _ = @import("formal/formal_test_fixture.zig");
    _ = @import("formal/proof_manifest.zig");
}
