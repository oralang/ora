// Test root: discovers all compiler tests via per-subsystem files.
// Helpers live in compiler.test.helpers.zig.

comptime {
    _ = @import("compiler.test.abi.zig");
    _ = @import("compiler.test.comptime.zig");
    _ = @import("compiler.test.debug_artifacts.zig");
    _ = @import("compiler.test.diagnostics.zig");
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
}
