// Tests for evm.zig
const std = @import("std");
const primitives = @import("voltaire");
const evm = @import("evm.zig");
const Evm = evm.Evm(.{}); // Default config

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
