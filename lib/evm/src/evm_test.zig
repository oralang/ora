// Tests for evm.zig
const std = @import("std");
const primitives = @import("primitives.zig");
const evm = @import("evm.zig");
const Evm = evm.Evm(.{}); // Default config
const custom_config = @import("evm_config.zig").EvmConfig;
const Address = primitives.Address;

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

test "Evm top-level call executes identity precompile through Ora facade" {
    const testing = std.testing;

    var evm_instance: Evm = undefined;
    try evm_instance.init(testing.allocator, null, .OSAKA, null, primitives.ZERO_ADDRESS, 0, null);
    defer evm_instance.deinit();

    const input = [_]u8{ 0xde, 0xad, 0xbe, 0xef };
    const result = evm_instance.call(.{ .call = .{
        .caller = Address.fromU256(0xCA11E),
        .to = Address.fromU256(0x04),
        .value = 0,
        .input = &input,
        .gas = 1_000_000,
    } });

    try testing.expect(result.success);
    try testing.expectEqualSlices(u8, &input, result.output);
    try testing.expect(result.gas_left < 1_000_000);
}

test "Evm top-level call treats 0x13 as Prague/Osaka precompile, not empty account" {
    const testing = std.testing;
    const fp2_to_g2 = Address.fromU256(0x13);

    var cancun_evm: Evm = undefined;
    try cancun_evm.init(testing.allocator, null, .CANCUN, null, primitives.ZERO_ADDRESS, 0, null);
    defer cancun_evm.deinit();

    const cancun_result = cancun_evm.call(.{ .call = .{
        .caller = Address.fromU256(0xCA11E),
        .to = fp2_to_g2,
        .value = 0,
        .input = &.{},
        .gas = 1_000_000,
    } });
    try testing.expect(cancun_result.success);
    try testing.expectEqual(@as(usize, 0), cancun_result.output.len);

    var osaka_evm: Evm = undefined;
    try osaka_evm.init(testing.allocator, null, .OSAKA, null, primitives.ZERO_ADDRESS, 0, null);
    defer osaka_evm.deinit();

    const fp2_zero = [_]u8{0} ** 128;
    const osaka_result = osaka_evm.call(.{ .call = .{
        .caller = Address.fromU256(0xCA11E),
        .to = fp2_to_g2,
        .value = 0,
        .input = &fp2_zero,
        .gas = 1_000_000,
    } });
    try testing.expect(osaka_result.success);
    try testing.expectEqual(@as(usize, 256), osaka_result.output.len);
    try testing.expect(osaka_result.gas_left < 1_000_000);
}

test "Evm pre-warms all active precompile addresses through 0x13" {
    const testing = std.testing;

    var evm_instance: Evm = undefined;
    try evm_instance.init(testing.allocator, null, .OSAKA, null, primitives.ZERO_ADDRESS, 0, null);
    defer evm_instance.deinit();

    try evm_instance.initTransactionState(null);
    try evm_instance.preWarmTransaction(Address.fromU256(0xCA11));

    try testing.expect(evm_instance.access_list_manager.is_address_warm(Address.fromU256(0x01)));
    try testing.expect(evm_instance.access_list_manager.is_address_warm(Address.fromU256(0x13)));
    try testing.expect(!evm_instance.access_list_manager.is_address_warm(Address.fromU256(0x14)));
}
