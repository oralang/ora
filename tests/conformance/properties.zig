const std = @import("std");
const testing = std.testing;
const abi = @import("abi.zig");
const runner = @import("runner.zig");
const types = @import("types.zig");

const PROPERTY_CALLER = "0x1000000000000000000000000000000000000000";
const PROPERTY_ADDRESS_A = "0x2000000000000000000000000000000000000000";

fn expectWord(actual: []const u8, expected: u256) !void {
    if (actual.len != 32) return error.PropertyMismatch;
    const decoded = std.mem.readInt(u256, actual[0..32], .big);
    if (decoded != expected) return error.PropertyMismatch;
}

fn expectSignedWord(actual: []const u8, bits: u16, expected: i256) !void {
    if (actual.len != 32) return error.PropertyMismatch;
    if (abi.decodeSignedWord(actual[0..32], bits) != expected) return error.PropertyMismatch;
}

fn expectBoolWord(actual: []const u8, expected: bool) !void {
    if (actual.len != 32) return error.PropertyMismatch;
    const value = std.mem.readInt(u256, actual[0..32], .big);
    if ((value != 0) != expected) return error.PropertyMismatch;
}

fn expectAddressWord(actual: []const u8, expected: []const u8) !void {
    if (actual.len != 32) return error.PropertyMismatch;
    const expected_bytes = try abi.parseHexBytesFixed(20, expected);
    if (!std.mem.allEqual(u8, actual[0..12], 0)) return error.PropertyMismatch;
    if (!std.mem.eql(u8, actual[12..32], &expected_bytes)) return error.PropertyMismatch;
}

fn expectFixedBytesWord(actual: []const u8, expected: []const u8) !void {
    if (actual.len != 32) return error.PropertyMismatch;
    const expected_bytes = try abi.parseHexBytesFixed(4, expected);
    if (!std.mem.eql(u8, actual[0..4], &expected_bytes)) return error.PropertyMismatch;
    if (!std.mem.allEqual(u8, actual[4..32], 0)) return error.PropertyMismatch;
}

const PropertyFault = enum {
    none,
    abi,
    arithmetic,
    storage,
    error_union,
};

const property_source =
    \\contract ConformanceProperties {
    \\    error Failed;
    \\
    \\    storage var slots: map<u256, u256>;
    \\
    \\    pub fn id_u256(x: u256) -> u256 { return x; }
    \\    pub fn id_i16(x: i16) -> i16 { return x; }
    \\    pub fn id_address(x: address) -> address { return x; }
    \\    pub fn id_bool(x: bool) -> bool { return x; }
    \\    pub fn id_bytes4(x: bytes4) -> bytes4 { return x; }
    \\
    \\    pub fn sum_slice(values: slice[u256]) -> u256 {
    \\        var total: u256 = 0;
    \\        for (values) |value| {
    \\            total = total +% value;
    \\        }
    \\        return total;
    \\    }
    \\
    \\    pub fn add_wrap(a: u256, b: u256) -> u256 { return a +% b; }
    \\    pub fn sub_wrap(a: u256, b: u256) -> u256 { return a -% b; }
    \\    pub fn mul_wrap(a: u256, b: u256) -> u256 { return a *% b; }
    \\    pub fn div_i16(a: i16, b: i16) -> i16 { return a / b; }
    \\    pub fn lt_i16(a: i16, b: i16) -> bool { return a < b; }
    \\
    \\    pub fn put(slot: u256, value: u256) { slots[slot] = value; }
    \\    pub fn read(slot: u256) -> u256 { return slots[slot]; }
    \\
    \\    fn may_fail(x: u256) -> !u256 | Failed {
    \\        if (x == 0) {
    \\            return Failed;
    \\        }
    \\        return x;
    \\    }
    \\
    \\    pub fn unwrap_ok(x: u256) -> !u256 | Failed {
    \\        let value: u256 = try may_fail(x);
    \\        return value;
    \\    }
    \\
    \\    pub fn catch_value(x: u256) -> u256 {
    \\        try {
    \\            let value: u256 = try may_fail(x);
    \\            return value;
    \\        } catch {
    \\            return 999;
    \\        }
    \\    }
    \\}
;

fn checkAbiRoundtripProperty(runtime: *runner.PropertyRuntime, fault: PropertyFault) !void {
    const u256_args = [_]types.ArgValue{.{ .literal = "340282366920938463463374607431768211455" }};
    const u256_result = try runtime.call("id_u256(uint256)", &u256_args);
    try testing.expect(u256_result.success);
    try expectWord(u256_result.output, if (fault == .abi) 1 else 340282366920938463463374607431768211455);

    const i16_args = [_]types.ArgValue{.{ .literal = "-12345" }};
    const i16_result = try runtime.call("id_i16(int16)", &i16_args);
    try testing.expect(i16_result.success);
    try expectSignedWord(i16_result.output, 16, if (fault == .abi) -12344 else -12345);

    const address_args = [_]types.ArgValue{.{ .literal = PROPERTY_ADDRESS_A }};
    const address_result = try runtime.call("id_address(address)", &address_args);
    try testing.expect(address_result.success);
    try expectAddressWord(address_result.output, if (fault == .abi) PROPERTY_CALLER else PROPERTY_ADDRESS_A);

    const bool_args = [_]types.ArgValue{.{ .boolean = true }};
    const bool_result = try runtime.call("id_bool(bool)", &bool_args);
    try testing.expect(bool_result.success);
    try expectBoolWord(bool_result.output, fault != .abi);

    const bytes_args = [_]types.ArgValue{.{ .literal = "0xabcdef12" }};
    const bytes_result = try runtime.call("id_bytes4(bytes4)", &bytes_args);
    try testing.expect(bytes_result.success);
    try expectFixedBytesWord(bytes_result.output, if (fault == .abi) "0xabcdef13" else "0xabcdef12");

    const slice_args = [_]types.ArgValue{.{ .literal = "[5,8,13]" }};
    const slice_result = try runtime.call("sum_slice(uint256[])", &slice_args);
    try testing.expect(slice_result.success);
    try expectWord(slice_result.output, if (fault == .abi) 25 else 26);
}

fn checkArithmeticProperty(runtime: *runner.PropertyRuntime, fault: PropertyFault) !void {
    const a: u256 = std.math.maxInt(u256) - 2;
    const b: u256 = 7;

    const add_args = [_]types.ArgValue{
        .{ .literal = "115792089237316195423570985008687907853269984665640564039457584007913129639933" },
        .{ .literal = "7" },
    };
    const add_result = try runtime.call("add_wrap(uint256,uint256)", &add_args);
    try testing.expect(add_result.success);
    try expectWord(add_result.output, if (fault == .arithmetic) 3 else a +% b);

    const sub_args = [_]types.ArgValue{ .{ .literal = "3" }, .{ .literal = "9" } };
    const sub_result = try runtime.call("sub_wrap(uint256,uint256)", &sub_args);
    try testing.expect(sub_result.success);
    try expectWord(sub_result.output, if (fault == .arithmetic) 0 else @as(u256, 3) -% 9);

    const mul_args = [_]types.ArgValue{ .{ .literal = "340282366920938463463374607431768211456" }, .{ .literal = "19" } };
    const mul_result = try runtime.call("mul_wrap(uint256,uint256)", &mul_args);
    try testing.expect(mul_result.success);
    try expectWord(mul_result.output, if (fault == .arithmetic) 18 else (@as(u256, 1) << 128) *% 19);

    const div_args = [_]types.ArgValue{ .{ .literal = "-12345" }, .{ .literal = "7" } };
    const div_result = try runtime.call("div_i16(int16,int16)", &div_args);
    try testing.expect(div_result.success);
    try expectSignedWord(div_result.output, 16, if (fault == .arithmetic) -1762 else @divTrunc(@as(i256, -12345), 7));

    const lt_args = [_]types.ArgValue{ .{ .literal = "-1" }, .{ .literal = "1" } };
    const lt_result = try runtime.call("lt_i16(int16,int16)", &lt_args);
    try testing.expect(lt_result.success);
    try expectBoolWord(lt_result.output, fault != .arithmetic);
}

fn checkStorageProperty(runtime: *runner.PropertyRuntime, fault: PropertyFault) !void {
    const rows = [_]struct { slot: []const u8, value: []const u8 }{
        .{ .slot = "0", .value = "1" },
        .{ .slot = "1", .value = "340282366920938463463374607431768211455" },
        .{ .slot = "999", .value = "42" },
    };

    for (rows, 0..) |row, index| {
        const put_args = [_]types.ArgValue{ .{ .literal = row.slot }, .{ .literal = row.value } };
        const put_result = try runtime.call("put(uint256,uint256)", &put_args);
        try testing.expect(put_result.success);
        try testing.expectEqual(@as(usize, 0), put_result.output.len);

        const read_args = [_]types.ArgValue{.{ .literal = row.slot }};
        const read_result = try runtime.call("read(uint256)", &read_args);
        try testing.expect(read_result.success);
        const expected = try abi.parseU256(row.value);
        try expectWord(read_result.output, if (fault == .storage and index == 1) expected + 1 else expected);
    }
}

fn checkErrorUnionProperty(runtime: *runner.PropertyRuntime, fault: PropertyFault) !void {
    const ok_args = [_]types.ArgValue{.{ .literal = "77" }};
    const ok_result = try runtime.call("unwrap_ok(uint256)", &ok_args);
    try testing.expect(ok_result.success);
    try expectWord(ok_result.output, if (fault == .error_union) 78 else 77);

    const catch_ok = try runtime.call("catch_value(uint256)", &ok_args);
    try testing.expect(catch_ok.success);
    try expectWord(catch_ok.output, if (fault == .error_union) 76 else 77);

    const fail_args = [_]types.ArgValue{.{ .literal = "0" }};
    const catch_fail = try runtime.call("catch_value(uint256)", &fail_args);
    try testing.expect(catch_fail.success);
    try expectWord(catch_fail.output, if (fault == .error_union) 998 else 999);

    const revert_result = try runtime.call("unwrap_ok(uint256)", &fail_args);
    if (fault == .error_union) {
        if (revert_result.success) return error.PropertyMismatch;
    } else {
        try testing.expect(!revert_result.success);
    }
}

fn runPropertyChecks(runtime: *runner.PropertyRuntime) !void {
    try checkAbiRoundtripProperty(runtime, .none);
    try testing.expectError(error.PropertyMismatch, checkAbiRoundtripProperty(runtime, .abi));

    try checkArithmeticProperty(runtime, .none);
    try testing.expectError(error.PropertyMismatch, checkArithmeticProperty(runtime, .arithmetic));

    try checkStorageProperty(runtime, .none);
    try testing.expectError(error.PropertyMismatch, checkStorageProperty(runtime, .storage));

    try checkErrorUnionProperty(runtime, .none);
    try testing.expectError(error.PropertyMismatch, checkErrorUnionProperty(runtime, .error_union));
}

pub fn run(allocator: std.mem.Allocator) !void {
    try runner.compileAndRunPropertySource(allocator, "conformance_properties", property_source, runPropertyChecks);
}
