pub const default_chain_id: u64 = 31337;
const metrics = @import("metrics.zig");

pub const CompileOptions = struct {
    chain_id: u64 = default_chain_id,
    instrumentation: ?*metrics.Metrics = null,
    /// Enables measurement-only loop facts for the standalone census tool.
    /// Normal compilation leaves this disabled.
    measure_loop_census: bool = false,
};
