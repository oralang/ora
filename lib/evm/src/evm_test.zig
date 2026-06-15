// Tests for evm.zig
const std = @import("std");
const primitives = @import("voltaire");
const evm = @import("evm.zig");
const Evm = evm.Evm(.{}); // Default config
const custom_config = @import("evm_config.zig").EvmConfig;

test "Evm init and deinit" {
    const testing = std.testing;

    var evm_instance: Evm = undefined;
    try evm_instance.init(testing.allocator, null, null, null, primitives.ZERO_ADDRESS, 0, null);
    defer evm_instance.deinit();

    try testing.expectEqual(@as(usize, 0), evm_instance.frames.items.len);
}

test "Evm getCurrentFrame returns null when no frames" {
    const testing = std.testing;

    var evm_instance: Evm = undefined;
    try evm_instance.init(testing.allocator, null, null, null, primitives.ZERO_ADDRESS, 0, null);
    defer evm_instance.deinit();

    try testing.expect(evm_instance.getCurrentFrame() == null);
}

test "Evm reserves frame storage for configured call depth" {
    const testing = std.testing;
    const max_call_depth: u16 = 64;
    const SmallDepthEvm = evm.Evm(custom_config{ .max_call_depth = max_call_depth });

    var evm_instance: SmallDepthEvm = undefined;
    try evm_instance.init(testing.allocator, null, null, null, primitives.ZERO_ADDRESS, 0, null);
    defer evm_instance.deinit();

    try evm_instance.initTransactionState(null);
    try testing.expect(evm_instance.frames.capacity >= max_call_depth);
}
