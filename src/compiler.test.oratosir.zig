// OraToSIR tests are split by behavior so feature changes touch local test modules.
comptime {
    _ = @import("compiler.test.oratosir.source_inline.zig");
    _ = @import("compiler.test.oratosir.cfg.zig");
    _ = @import("compiler.test.oratosir.lowering.zig");
    _ = @import("compiler.test.oratosir.optimize.zig");
    _ = @import("compiler.test.oratosir.abi.zig");
    _ = @import("compiler.test.oratosir.refinement_guards.zig");
    _ = @import("compiler.test.oratosir.adt_error.zig");
    _ = @import("compiler.test.oratosir.storage_resource.zig");
    _ = @import("compiler.test.oratosir.dispatcher.zig");
}
