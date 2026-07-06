const std = @import("std");
const testing = std.testing;
const abi = @import("abi.zig");
const runner = @import("runner.zig");
const types = @import("types.zig");

const PROPERTY_CALLER = "0x1000000000000000000000000000000000000000";
const PROPERTY_ADDRESS_A = "0x2000000000000000000000000000000000000000";
const PROPERTY_ADDRESS_B = "0x3000000000000000000000000000000000000000";
const PROPERTY_ADDRESS_C = "0x4000000000000000000000000000000000000000";
const PROPERTY_ADDRESS_D = "0x5000000000000000000000000000000000000000";
const PROPERTY_ADDRESS_E = "0x6000000000000000000000000000000000000000";
const PROPERTY_ADDRESS_F = "0x7000000000000000000000000000000000000000";
const PROPERTY_ADDRESS_G = "0x8000000000000000000000000000000000000000";
const PROPERTY_ADDRESS_H = "0x9000000000000000000000000000000000000000";
const U256_MAX_DEC = "115792089237316195423570985008687907853269984665640564039457584007913129639935";
const I256_MAX_DEC = "57896044618658097711785492504343953926634992332820282019728792003956564819967";
const I256_MIN_DEC = "-57896044618658097711785492504343953926634992332820282019728792003956564819968";

fn expectWord(actual: []const u8, expected: u256) !void {
    if (actual.len != 32) return error.PropertyMismatch;
    if (decodeWord(actual) != expected) return error.PropertyMismatch;
}

fn decodeWord(actual: []const u8) u256 {
    std.debug.assert(actual.len == 32);
    return std.mem.readInt(u256, actual[0..32], .big);
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
    source_abi_encode,
    arithmetic,
    storage,
    error_union,
    resource,
};

const property_source =
    \\resource PropertyUnit = u256;
    \\
    \\contract ConformanceProperties {
    \\    error Failed;
    \\
    \\    storage var slots: map<u256, u256>;
    \\    storage var resource_balances: map<address, Resource<PropertyUnit>>;
    \\
    \\    pub fn id_u256(x: u256) -> u256 { return x; }
    \\    pub fn id_i16(x: i16) -> i16 { return x; }
    \\    pub fn id_address(x: address) -> address { return x; }
    \\    pub fn id_bool(x: bool) -> bool { return x; }
    \\    pub fn id_bytes4(x: bytes4) -> bytes4 { return x; }
    \\
    \\    pub fn source_abi_roundtrip_u256(x: u256) -> u256 {
    \\        let decoded = @abiDecode(u256, @abiEncode(x));
    \\        return match (decoded) {
    \\            Ok(value) => value,
    \\            Err(_) => 0,
    \\        };
    \\    }
    \\
    \\    pub fn source_abi_roundtrip_i16(x: i16) -> i16 {
    \\        let decoded = @abiDecode(i16, @abiEncode(x));
    \\        return match (decoded) {
    \\            Ok(value) => value,
    \\            Err(_) => 0,
    \\        };
    \\    }
    \\
    \\    pub fn source_abi_roundtrip_address(x: address) -> address {
    \\        let decoded = @abiDecode(address, @abiEncode(x));
    \\        return match (decoded) {
    \\            Ok(value) => value,
    \\            Err(_) => 0x0000000000000000000000000000000000000000,
    \\        };
    \\    }
    \\
    \\    pub fn source_abi_roundtrip_bool(x: bool) -> bool {
    \\        let decoded = @abiDecode(bool, @abiEncode(x));
    \\        return match (decoded) {
    \\            Ok(value) => value,
    \\            Err(_) => false,
    \\        };
    \\    }
    \\
    \\    pub fn source_abi_roundtrip_bytes4(x: bytes4) -> bytes4 {
    \\        let decoded = @abiDecode(bytes4, @abiEncode(x));
    \\        return match (decoded) {
    \\            Ok(value) => value,
    \\            Err(_) => hex"00000000",
    \\        };
    \\    }
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
    \\    pub fn u256_gt_zero(x: u256) -> bool { return x > 0; }
    \\    pub fn i256_gt_zero(x: i256) -> bool { return x > 0; }
    \\    pub fn i256_lt_zero(x: i256) -> bool { return x < 0; }
    \\    pub fn u256_max_literal_gt_zero() -> bool {
    \\        let value: u256 = 0xffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff;
    \\        return value > 0;
    \\    }
    \\    pub fn i256_max_literal_gt_zero() -> bool {
    \\        let value: i256 = 0x7fffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff;
    \\        return value > 0;
    \\    }
    \\    pub fn i256_min_literal_lt_zero() -> bool {
    \\        let value: i256 = -57896044618658097711785492504343953926634992332820282019728792003956564819968;
    \\        return value < 0;
    \\    }
    \\
    \\    pub fn put(slot: u256, value: u256) { slots[slot] = value; }
    \\    pub fn read(slot: u256) -> u256 { return slots[slot]; }
    \\
    \\    pub fn issue_resource(to: address, amount: PropertyUnit)
    \\        requires @amount(resource_balances[to]) <= 1000 - amount
    \\    {
    \\        @create(resource_balances[to], amount);
    \\    }
    \\
    \\    pub fn move_resource(from: address, to: address, amount: PropertyUnit)
    \\        requires from != to
    \\        requires @amount(resource_balances[from]) >= amount
    \\        requires @amount(resource_balances[to]) <= 1000 - amount
    \\    {
    \\        @move(resource_balances[from], resource_balances[to], amount);
    \\    }
    \\
    \\    pub fn self_move_resource(owner: address, amount: PropertyUnit)
    \\        requires @amount(resource_balances[owner]) >= amount
    \\    {
    \\        @move(resource_balances[owner], resource_balances[owner], amount);
    \\    }
    \\
    \\    pub fn burn_resource(from: address, amount: PropertyUnit)
    \\        requires @amount(resource_balances[from]) >= amount
    \\    {
    \\        @destroy(resource_balances[from], amount);
    \\    }
    \\
    \\    pub fn resource_balance(owner: address) -> PropertyUnit {
    \\        return @amount(resource_balances[owner]);
    \\    }
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

fn checkSourceAbiEncodeRoundtripProperty(runtime: *runner.PropertyRuntime, fault: PropertyFault) !void {
    const u256_args = [_]types.ArgValue{.{ .literal = "340282366920938463463374607431768211455" }};
    const u256_result = try runtime.call("source_abi_roundtrip_u256(uint256)", &u256_args);
    try testing.expect(u256_result.success);
    try expectWord(u256_result.output, if (fault == .source_abi_encode) 1 else 340282366920938463463374607431768211455);

    const i16_args = [_]types.ArgValue{.{ .literal = "-12345" }};
    const i16_result = try runtime.call("source_abi_roundtrip_i16(int16)", &i16_args);
    try testing.expect(i16_result.success);
    try expectSignedWord(i16_result.output, 16, if (fault == .source_abi_encode) -12344 else -12345);

    const address_args = [_]types.ArgValue{.{ .literal = PROPERTY_ADDRESS_A }};
    const address_result = try runtime.call("source_abi_roundtrip_address(address)", &address_args);
    try testing.expect(address_result.success);
    try expectAddressWord(address_result.output, if (fault == .source_abi_encode) PROPERTY_CALLER else PROPERTY_ADDRESS_A);

    const bool_args = [_]types.ArgValue{.{ .boolean = true }};
    const bool_result = try runtime.call("source_abi_roundtrip_bool(bool)", &bool_args);
    try testing.expect(bool_result.success);
    try expectBoolWord(bool_result.output, fault != .source_abi_encode);

    const bytes_args = [_]types.ArgValue{.{ .literal = "0xabcdef12" }};
    const bytes_result = try runtime.call("source_abi_roundtrip_bytes4(bytes4)", &bytes_args);
    try testing.expect(bytes_result.success);
    try expectFixedBytesWord(bytes_result.output, if (fault == .source_abi_encode) "0xabcdef13" else "0xabcdef12");
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

    const u256_max_args = [_]types.ArgValue{.{ .literal = U256_MAX_DEC }};
    const u256_gt_result = try runtime.call("u256_gt_zero(uint256)", &u256_max_args);
    try testing.expect(u256_gt_result.success);
    try expectBoolWord(u256_gt_result.output, true);

    const i256_max_args = [_]types.ArgValue{.{ .literal = I256_MAX_DEC }};
    const i256_gt_result = try runtime.call("i256_gt_zero(int256)", &i256_max_args);
    try testing.expect(i256_gt_result.success);
    try expectBoolWord(i256_gt_result.output, true);

    const i256_neg_args = [_]types.ArgValue{.{ .literal = "-1" }};
    const i256_neg_gt_result = try runtime.call("i256_gt_zero(int256)", &i256_neg_args);
    try testing.expect(i256_neg_gt_result.success);
    try expectBoolWord(i256_neg_gt_result.output, false);

    const i256_min_args = [_]types.ArgValue{.{ .literal = I256_MIN_DEC }};
    const i256_min_lt_result = try runtime.call("i256_lt_zero(int256)", &i256_min_args);
    try testing.expect(i256_min_lt_result.success);
    try expectBoolWord(i256_min_lt_result.output, true);

    const u256_literal_result = try runtime.call("u256_max_literal_gt_zero()", &.{});
    try testing.expect(u256_literal_result.success);
    try expectBoolWord(u256_literal_result.output, true);

    const i256_max_literal_result = try runtime.call("i256_max_literal_gt_zero()", &.{});
    try testing.expect(i256_max_literal_result.success);
    try expectBoolWord(i256_max_literal_result.output, true);

    const i256_min_literal_result = try runtime.call("i256_min_literal_lt_zero()", &.{});
    try testing.expect(i256_min_literal_result.success);
    try expectBoolWord(i256_min_literal_result.output, true);
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

fn checkResourceProperty(runtime: *runner.PropertyRuntime, fault: PropertyFault) !void {
    const issue_args = [_]types.ArgValue{ .{ .literal = PROPERTY_ADDRESS_A }, .{ .literal = "10" } };
    const issue_result = try runtime.call("issue_resource(address,uint256)", &issue_args);
    try testing.expect(issue_result.success);

    const balance_a_args = [_]types.ArgValue{.{ .literal = PROPERTY_ADDRESS_A }};
    const balance_b_args = [_]types.ArgValue{.{ .literal = PROPERTY_ADDRESS_B }};

    const balance_a_after_issue = try runtime.call("resource_balance(address)", &balance_a_args);
    try testing.expect(balance_a_after_issue.success);
    try expectWord(balance_a_after_issue.output, if (fault == .resource) 9 else 10);

    const move_args = [_]types.ArgValue{ .{ .literal = PROPERTY_ADDRESS_A }, .{ .literal = PROPERTY_ADDRESS_B }, .{ .literal = "4" } };
    const move_result = try runtime.call("move_resource(address,address,uint256)", &move_args);
    try testing.expect(move_result.success);

    const balance_a_after_move = try runtime.call("resource_balance(address)", &balance_a_args);
    try testing.expect(balance_a_after_move.success);
    try expectWord(balance_a_after_move.output, if (fault == .resource) 7 else 6);

    const balance_b_after_move = try runtime.call("resource_balance(address)", &balance_b_args);
    try testing.expect(balance_b_after_move.success);
    try expectWord(balance_b_after_move.output, if (fault == .resource) 5 else 4);

    const self_move_args = [_]types.ArgValue{ .{ .literal = PROPERTY_ADDRESS_A }, .{ .literal = "3" } };
    const self_move_result = try runtime.call("self_move_resource(address,uint256)", &self_move_args);
    try testing.expect(self_move_result.success);

    const balance_a_after_self_move = try runtime.call("resource_balance(address)", &balance_a_args);
    try testing.expect(balance_a_after_self_move.success);
    try expectWord(balance_a_after_self_move.output, if (fault == .resource) 3 else 6);

    const overdraw_args = [_]types.ArgValue{ .{ .literal = PROPERTY_ADDRESS_A }, .{ .literal = PROPERTY_ADDRESS_B }, .{ .literal = "7" } };
    const overdraw_result = try runtime.call("move_resource(address,address,uint256)", &overdraw_args);
    if (overdraw_result.success) return error.PropertyMismatch;

    const balance_a_after_overdraw = try runtime.call("resource_balance(address)", &balance_a_args);
    try testing.expect(balance_a_after_overdraw.success);
    try expectWord(balance_a_after_overdraw.output, if (fault == .resource) 0 else 6);

    const burn_args = [_]types.ArgValue{ .{ .literal = PROPERTY_ADDRESS_B }, .{ .literal = "4" } };
    const burn_result = try runtime.call("burn_resource(address,uint256)", &burn_args);
    try testing.expect(burn_result.success);

    const balance_b_after_burn = try runtime.call("resource_balance(address)", &balance_b_args);
    try testing.expect(balance_b_after_burn.success);
    try expectWord(balance_b_after_burn.output, if (fault == .resource) 4 else 0);
}

fn resourceAmountLiteral(buf: *[80]u8, amount: u256) ![]const u8 {
    return std.fmt.bufPrint(buf, "{}", .{amount});
}

fn issueResource(runtime: *runner.PropertyRuntime, owner: []const u8, amount: u256) !void {
    var amount_buf: [80]u8 = undefined;
    const args = [_]types.ArgValue{
        .{ .literal = owner },
        .{ .literal = try resourceAmountLiteral(&amount_buf, amount) },
    };
    const result = try runtime.call("issue_resource(address,uint256)", &args);
    if (!result.success) return error.PropertyMismatch;
    try testing.expectEqual(@as(usize, 0), result.output.len);
}

fn moveResource(runtime: *runner.PropertyRuntime, from: []const u8, to: []const u8, amount: u256) !void {
    var amount_buf: [80]u8 = undefined;
    const args = [_]types.ArgValue{
        .{ .literal = from },
        .{ .literal = to },
        .{ .literal = try resourceAmountLiteral(&amount_buf, amount) },
    };
    const result = try runtime.call("move_resource(address,address,uint256)", &args);
    if (!result.success) return error.PropertyMismatch;
    try testing.expectEqual(@as(usize, 0), result.output.len);
}

fn selfMoveResource(runtime: *runner.PropertyRuntime, owner: []const u8, amount: u256) !void {
    var amount_buf: [80]u8 = undefined;
    const args = [_]types.ArgValue{
        .{ .literal = owner },
        .{ .literal = try resourceAmountLiteral(&amount_buf, amount) },
    };
    const result = try runtime.call("self_move_resource(address,uint256)", &args);
    if (!result.success) return error.PropertyMismatch;
    try testing.expectEqual(@as(usize, 0), result.output.len);
}

fn burnResource(runtime: *runner.PropertyRuntime, owner: []const u8, amount: u256) !void {
    var amount_buf: [80]u8 = undefined;
    const args = [_]types.ArgValue{
        .{ .literal = owner },
        .{ .literal = try resourceAmountLiteral(&amount_buf, amount) },
    };
    const result = try runtime.call("burn_resource(address,uint256)", &args);
    if (!result.success) return error.PropertyMismatch;
    try testing.expectEqual(@as(usize, 0), result.output.len);
}

fn resourceBalance(runtime: *runner.PropertyRuntime, owner: []const u8) !u256 {
    const args = [_]types.ArgValue{.{ .literal = owner }};
    const result = try runtime.call("resource_balance(address)", &args);
    if (!result.success or result.output.len != 32) return error.PropertyMismatch;
    return decodeWord(result.output);
}

fn expectResourceBalances(runtime: *runner.PropertyRuntime, owners: []const []const u8, expected: []const u256) !void {
    try testing.expectEqual(owners.len, expected.len);
    for (owners, expected) |owner, value| {
        try testing.expectEqual(value, try resourceBalance(runtime, owner));
    }
}

fn sumResourceModel(values: []const u256) u256 {
    var total: u256 = 0;
    for (values) |value| total += value;
    return total;
}

fn checkResourceMoveSequences(runtime: *runner.PropertyRuntime) !void {
    const owners = [_][]const u8{ PROPERTY_ADDRESS_C, PROPERTY_ADDRESS_D, PROPERTY_ADDRESS_E };
    var model = [_]u256{ 100, 40, 0 };
    for (owners, model) |owner, amount| try issueResource(runtime, owner, amount);
    const initial_total = sumResourceModel(&model);

    const Step = struct { from: usize, to: usize, amount: u256 };
    const steps = [_]Step{
        .{ .from = 0, .to = 2, .amount = 17 },
        .{ .from = 1, .to = 0, .amount = 9 },
        .{ .from = 2, .to = 1, .amount = 4 },
        .{ .from = 0, .to = 0, .amount = 12 },
        .{ .from = 1, .to = 2, .amount = 0 },
        .{ .from = 2, .to = 0, .amount = 13 },
        .{ .from = 1, .to = 2, .amount = 21 },
        .{ .from = 0, .to = 1, .amount = 5 },
        .{ .from = 2, .to = 2, .amount = 20 },
        .{ .from = 0, .to = 2, .amount = 100 },
        .{ .from = 2, .to = 1, .amount = 11 },
        .{ .from = 1, .to = 0, .amount = 30 },
        .{ .from = 2, .to = 0, .amount = 110 },
    };

    for (steps) |step| {
        if (step.from == step.to) {
            try selfMoveResource(runtime, owners[step.from], step.amount);
        } else {
            try moveResource(runtime, owners[step.from], owners[step.to], step.amount);
            model[step.from] -= step.amount;
            model[step.to] += step.amount;
        }

        try testing.expectEqual(initial_total, sumResourceModel(&model));
        try expectResourceBalances(runtime, &owners, &model);
    }
}

fn checkResourceDeltaSequences(runtime: *runner.PropertyRuntime) !void {
    const owners = [_][]const u8{ PROPERTY_ADDRESS_F, PROPERTY_ADDRESS_G, PROPERTY_ADDRESS_H };
    var model = [_]u256{ 0, 0, 0 };
    var domain_delta: u256 = 0;

    const Op = enum { create, move, burn };
    const Step = struct { op: Op, from: usize, to: usize = 0, amount: u256 };
    const steps = [_]Step{
        .{ .op = .create, .from = 0, .amount = 50 },
        .{ .op = .create, .from = 1, .amount = 20 },
        .{ .op = .move, .from = 0, .to = 2, .amount = 15 },
        .{ .op = .move, .from = 2, .to = 2, .amount = 5 },
        .{ .op = .burn, .from = 1, .amount = 7 },
        .{ .op = .create, .from = 2, .amount = 3 },
        .{ .op = .move, .from = 2, .to = 1, .amount = 10 },
        .{ .op = .burn, .from = 0, .amount = 35 },
        .{ .op = .create, .from = 1, .amount = 12 },
        .{ .op = .move, .from = 1, .to = 0, .amount = 30 },
        .{ .op = .burn, .from = 2, .amount = 8 },
        .{ .op = .create, .from = 0, .amount = 0 },
        .{ .op = .move, .from = 0, .to = 1, .amount = 0 },
        .{ .op = .burn, .from = 1, .amount = 5 },
    };

    for (steps) |step| {
        switch (step.op) {
            .create => {
                try issueResource(runtime, owners[step.from], step.amount);
                model[step.from] += step.amount;
                domain_delta += step.amount;
            },
            .move => {
                if (step.from == step.to) {
                    try selfMoveResource(runtime, owners[step.from], step.amount);
                } else {
                    try moveResource(runtime, owners[step.from], owners[step.to], step.amount);
                    model[step.from] -= step.amount;
                    model[step.to] += step.amount;
                }
            },
            .burn => {
                try burnResource(runtime, owners[step.from], step.amount);
                model[step.from] -= step.amount;
                domain_delta -= step.amount;
            },
        }

        try testing.expectEqual(domain_delta, sumResourceModel(&model));
        try expectResourceBalances(runtime, &owners, &model);
    }
}

fn runPropertyChecks(runtime: *runner.PropertyRuntime) !void {
    try checkAbiRoundtripProperty(runtime, .none);
    try testing.expectError(error.PropertyMismatch, checkAbiRoundtripProperty(runtime, .abi));

    try checkSourceAbiEncodeRoundtripProperty(runtime, .none);
    try testing.expectError(error.PropertyMismatch, checkSourceAbiEncodeRoundtripProperty(runtime, .source_abi_encode));

    try checkArithmeticProperty(runtime, .none);
    try testing.expectError(error.PropertyMismatch, checkArithmeticProperty(runtime, .arithmetic));

    try checkStorageProperty(runtime, .none);
    try testing.expectError(error.PropertyMismatch, checkStorageProperty(runtime, .storage));

    try checkErrorUnionProperty(runtime, .none);
    try testing.expectError(error.PropertyMismatch, checkErrorUnionProperty(runtime, .error_union));

    try checkResourceProperty(runtime, .none);
    try checkResourceMoveSequences(runtime);
    try checkResourceDeltaSequences(runtime);
    try testing.expectError(error.PropertyMismatch, checkResourceProperty(runtime, .resource));
}

pub fn run(allocator: std.mem.Allocator) !void {
    try runner.compileAndRunPropertySource(allocator, "conformance_properties", property_source, runPropertyChecks);
}
