pub const default_chain_id: u64 = 31337;
const metrics = @import("metrics.zig");

pub const CompileOptions = struct {
    chain_id: u64 = default_chain_id,
    instrumentation: ?*metrics.Metrics = null,
};
