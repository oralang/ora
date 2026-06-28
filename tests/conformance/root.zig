const std = @import("std");
const testing = std.testing;

const abi = @import("abi.zig");
const fuzz = @import("fuzz.zig");
const properties = @import("properties.zig");
const runner = @import("runner.zig");
const slots = @import("slots.zig");
const spec = @import("spec.zig");
const types = @import("types.zig");

test "conformance properties execute and catch planted failures" {
    try properties.run(testing.allocator);
}

test "structured ABI fuzzer rejects malformed calldata" {
    try fuzz.run(testing.allocator);
}

test "structured ABI fuzzer detects acceptance (teeth)" {
    fuzz.runTeeth(testing.allocator) catch |err| switch (err) {
        error.SkipZigTest => return error.SkipZigTest,
        error.FuzzAcceptedMalformedCalldata => return, // expected: teeth fired
        else => return err,
    };
    return error.FuzzTeethDidNotFire;
}

test "falsification harness compares checks-kept and checks-removed builds" {
    const source =
        \\contract FalsificationHarnessSmoke {
        \\    pub fn clamp_sum(a: u256, b: u256) -> u256
        \\        requires a <= 100
        \\        requires b <= 100
        \\        ensures result == a + b
        \\    {
        \\        return a + b;
        \\    }
        \\}
        \\
    ;

    try runner.compileAndRunPropertySourceDifferential(
        testing.allocator,
        "falsification_harness_smoke",
        source,
        &.{},
        &.{"--keep-proved-checks"},
        runFalsificationHarnessSmoke,
    );
}

test "falsification harness detects a kept-check divergence" {
    const source =
        \\contract FalsificationHarnessTeeth {
        \\    storage var values: map<u256, u256>;
        \\
        \\    pub fn narrow_mul_obligation(x: u8) -> u8
        \\        ensures result == x *% 2
        \\    {
        \\        return x;
        \\    }
        \\
        \\    pub fn div_obligation(x: u256, y: u256) -> u256
        \\        ensures result == x / y
        \\    {
        \\        return x;
        \\    }
        \\
        \\    pub fn signed_neg_obligation(x: i256) -> i256
        \\        ensures result == -x
        \\    {
        \\        return x;
        \\    }
        \\
        \\    pub fn map_obligation(k: u256, v: u256) -> u256
        \\        ensures values[k] == v
        \\    {
        \\        return values[k];
        \\    }
        \\
        \\    pub fn lie(x: u256) -> u256
        \\        ensures result == x + 1
        \\    {
        \\        return x;
        \\    }
        \\}
        \\
    ;

    try runner.compileAndRunPropertySourceDifferential(
        testing.allocator,
        "falsification_harness_teeth",
        source,
        &.{"--no-verify"},
        &.{ "--no-verify", "--keep-proved-checks" },
        runFalsificationHarnessTeeth,
    );
}

test "falsification corpus executes with checks removed and checks kept" {
    std.Io.Dir.cwd().access(std.Io.Threaded.global_single_threaded.io(), types.ORA_BINARY_REL, .{}) catch |err| switch (err) {
        error.FileNotFound => return error.SkipZigTest,
        else => return err,
    };

    const stems = [_][]const u8{
        "falsification_arith",
        "falsification_error_union_ok",
        "falsification_loop_invariant",
        "falsification_map_effect",
        "falsification_map_frame",
        "falsification_storage_branch",
    };

    const allocator = testing.allocator;
    for (stems) |stem| {
        const source_name = try std.fmt.allocPrint(allocator, "{s}.ora", .{stem});
        defer allocator.free(source_name);
        const spec_name = try std.fmt.allocPrint(allocator, "{s}.spec.toml", .{stem});
        defer allocator.free(spec_name);
        const source_path = try std.fs.path.join(allocator, &.{ types.CONFORMANCE_DIR_REL, source_name });
        defer allocator.free(source_path);
        const spec_path = try std.fs.path.join(allocator, &.{ types.CONFORMANCE_DIR_REL, spec_name });
        defer allocator.free(spec_path);

        runner.runConformanceSpecDifferential(allocator, source_path, spec_path) catch |err| {
            std.debug.print("falsification differential spec failed: {s}\n", .{spec_path});
            return err;
        };
    }
}

fn runFalsificationHarnessSmoke(
    checks_removed: *runner.PropertyRuntime,
    checks_kept: *runner.PropertyRuntime,
) !void {
    const zero_zero = [_]types.ArgValue{ .{ .literal = "0" }, .{ .literal = "0" } };
    try expectSameCall(checks_removed, checks_kept, "clamp_sum(uint256,uint256)", zero_zero[0..]);

    const max_envelope = [_]types.ArgValue{ .{ .literal = "100" }, .{ .literal = "100" } };
    try expectSameCall(checks_removed, checks_kept, "clamp_sum(uint256,uint256)", max_envelope[0..]);

    const requires_violation = [_]types.ArgValue{ .{ .literal = "101" }, .{ .literal = "0" } };
    try expectBothRevert(checks_removed, checks_kept, "clamp_sum(uint256,uint256)", requires_violation[0..]);
}

fn runFalsificationHarnessTeeth(
    checks_removed: *runner.PropertyRuntime,
    checks_kept: *runner.PropertyRuntime,
) !void {
    const narrow_args = [_]types.ArgValue{.{ .literal = "7" }};
    try expectRemovedSucceedsKeptReverts(checks_removed, checks_kept, "narrow_mul_obligation(uint8)", narrow_args[0..]);

    const div_args = [_]types.ArgValue{ .{ .literal = "7" }, .{ .literal = "2" } };
    try expectRemovedSucceedsKeptReverts(checks_removed, checks_kept, "div_obligation(uint256,uint256)", div_args[0..]);

    const signed_args = [_]types.ArgValue{.{ .literal = "5" }};
    try expectRemovedSucceedsKeptReverts(checks_removed, checks_kept, "signed_neg_obligation(int256)", signed_args[0..]);

    const map_args = [_]types.ArgValue{ .{ .literal = "1" }, .{ .literal = "9" } };
    try expectRemovedSucceedsKeptReverts(checks_removed, checks_kept, "map_obligation(uint256,uint256)", map_args[0..]);

    const generic_args = [_]types.ArgValue{.{ .literal = "7" }};
    try expectRemovedSucceedsKeptReverts(checks_removed, checks_kept, "lie(uint256)", generic_args[0..]);
}

fn expectRemovedSucceedsKeptReverts(
    checks_removed: *runner.PropertyRuntime,
    checks_kept: *runner.PropertyRuntime,
    signature: []const u8,
    args: []const types.ArgValue,
) !void {
    const removed = try checks_removed.call(signature, args);
    const kept = try checks_kept.call(signature, args);
    try testing.expect(removed.success);
    try testing.expect(!kept.success);
}

fn expectBothRevert(
    checks_removed: *runner.PropertyRuntime,
    checks_kept: *runner.PropertyRuntime,
    signature: []const u8,
    args: []const types.ArgValue,
) !void {
    const removed = try checks_removed.call(signature, args);
    const kept = try checks_kept.call(signature, args);
    try testing.expect(!removed.success);
    try testing.expect(!kept.success);
}

fn expectSameCall(
    checks_removed: *runner.PropertyRuntime,
    checks_kept: *runner.PropertyRuntime,
    signature: []const u8,
    args: []const types.ArgValue,
) !void {
    const removed = try checks_removed.call(signature, args);
    const kept = try checks_kept.call(signature, args);
    try testing.expectEqual(removed.success, kept.success);
    try testing.expectEqualSlices(u8, removed.output, kept.output);
}

// Suite self-test: prove the conformance layer has teeth — a correct
// spec passes, and the SAME spec with one wrong expected value is caught. If a
// corrupted expectation still "passed", the whole layer would be vacuously green.
test "conformance runner detects a wrong expected return (suite teeth)" {
    std.Io.Dir.cwd().access(std.Io.Threaded.global_single_threaded.io(), types.ORA_BINARY_REL, .{}) catch |err| switch (err) {
        error.FileNotFound => return error.SkipZigTest,
        else => return err,
    };

    const allocator = testing.allocator;
    var tmp = testing.tmpDir(.{});
    defer tmp.cleanup();

    const source =
        \\contract Teeth {
        \\    pub fn answer() -> u256 {
        \\        return 42;
        \\    }
        \\}
        \\
    ;
    try tmp.dir.writeFile(std.testing.io, .{ .sub_path = "teeth.ora", .data = source });

    const good_spec =
        \\[deploy]
        \\caller = "0x1000000000000000000000000000000000000000"
        \\value = 0
        \\args = []
        \\
        \\[[call]]
        \\fn = "answer()"
        \\caller = "0x1000000000000000000000000000000000000000"
        \\value = 0
        \\args = []
        \\returns = { u256 = 42 }
        \\
    ;
    try tmp.dir.writeFile(std.testing.io, .{ .sub_path = "teeth_good.spec.toml", .data = good_spec });

    // Same call, deliberately wrong expected value.
    const bad_spec =
        \\[deploy]
        \\caller = "0x1000000000000000000000000000000000000000"
        \\value = 0
        \\args = []
        \\
        \\[[call]]
        \\fn = "answer()"
        \\caller = "0x1000000000000000000000000000000000000000"
        \\value = 0
        \\args = []
        \\returns = { u256 = 43 }
        \\
    ;
    try tmp.dir.writeFile(std.testing.io, .{ .sub_path = "teeth_bad.spec.toml", .data = bad_spec });

    const source_path = try runner.pathFromTmpAlloc(allocator, tmp, "teeth.ora");
    defer allocator.free(source_path);
    const good_path = try runner.pathFromTmpAlloc(allocator, tmp, "teeth_good.spec.toml");
    defer allocator.free(good_path);
    const bad_path = try runner.pathFromTmpAlloc(allocator, tmp, "teeth_bad.spec.toml");
    defer allocator.free(bad_path);

    // The correct spec must pass...
    try runner.runConformanceSpec(allocator, source_path, good_path);
    // ...and the corrupted spec must be caught. Any error means the layer has
    // teeth; returning normally would mean a wrong expectation passes silently.
    if (runner.runConformanceSpec(allocator, source_path, bad_path)) |_| {
        return error.ConformanceSuiteHasNoTeeth;
    } else |_| {}
}

test "conformance runner detects gas ceilings (suite teeth)" {
    std.Io.Dir.cwd().access(std.Io.Threaded.global_single_threaded.io(), types.ORA_BINARY_REL, .{}) catch |err| switch (err) {
        error.FileNotFound => return error.SkipZigTest,
        else => return err,
    };

    const allocator = testing.allocator;
    var tmp = testing.tmpDir(.{});
    defer tmp.cleanup();

    const source =
        \\contract GasTeeth {
        \\    pub fn answer() -> u256 {
        \\        return 42;
        \\    }
        \\}
        \\
    ;
    try tmp.dir.writeFile(std.testing.io, .{ .sub_path = "gas_teeth.ora", .data = source });

    const good_spec =
        \\[deploy]
        \\caller = "0x1000000000000000000000000000000000000000"
        \\value = 0
        \\args = []
        \\
        \\[[call]]
        \\fn = "answer()"
        \\caller = "0x1000000000000000000000000000000000000000"
        \\value = 0
        \\args = []
        \\gas_max = 1000000
        \\returns = { u256 = 42 }
        \\
    ;
    try tmp.dir.writeFile(std.testing.io, .{ .sub_path = "gas_teeth_good.spec.toml", .data = good_spec });

    const bad_spec =
        \\[deploy]
        \\caller = "0x1000000000000000000000000000000000000000"
        \\value = 0
        \\args = []
        \\
        \\[[call]]
        \\fn = "answer()"
        \\caller = "0x1000000000000000000000000000000000000000"
        \\value = 0
        \\args = []
        \\gas_max = 1
        \\returns = { u256 = 42 }
        \\
    ;
    try tmp.dir.writeFile(std.testing.io, .{ .sub_path = "gas_teeth_bad.spec.toml", .data = bad_spec });

    const source_path = try runner.pathFromTmpAlloc(allocator, tmp, "gas_teeth.ora");
    defer allocator.free(source_path);
    const good_path = try runner.pathFromTmpAlloc(allocator, tmp, "gas_teeth_good.spec.toml");
    defer allocator.free(good_path);
    const bad_path = try runner.pathFromTmpAlloc(allocator, tmp, "gas_teeth_bad.spec.toml");
    defer allocator.free(bad_path);

    try runner.runConformanceSpec(allocator, source_path, good_path);
    if (runner.runConformanceSpec(allocator, source_path, bad_path)) |_| {
        return error.ConformanceGasCeilingHasNoTeeth;
    } else |_| {}
}

test "conformance corpus files have sidecars or explicit skips" {
    try runner.checkCorpusSidecars(testing.allocator, types.CONFORMANCE_DIR_REL);
}

test "conformance corpus specs execute" {
    std.Io.Dir.cwd().access(std.Io.Threaded.global_single_threaded.io(), types.ORA_BINARY_REL, .{}) catch |err| switch (err) {
        error.FileNotFound => return error.SkipZigTest,
        else => return err,
    };

    const allocator = testing.allocator;
    try runner.checkCorpusSidecars(allocator, types.CONFORMANCE_DIR_REL);

    const specs = try runner.collectSpecNames(allocator, types.CONFORMANCE_DIR_REL);
    defer runner.freeStringList(allocator, specs);
    try testing.expect(specs.len > 0);

    for (specs) |spec_name| {
        const stem = spec_name[0 .. spec_name.len - ".spec.toml".len];
        const source_name = try std.fmt.allocPrint(allocator, "{s}.ora", .{stem});
        defer allocator.free(source_name);
        const source_path = try std.fs.path.join(allocator, &.{ types.CONFORMANCE_DIR_REL, source_name });
        defer allocator.free(source_path);
        const spec_path = try std.fs.path.join(allocator, &.{ types.CONFORMANCE_DIR_REL, spec_name });
        defer allocator.free(spec_path);

        runner.runConformanceSpec(allocator, source_path, spec_path) catch |err| {
            std.debug.print("conformance spec failed: {s}\n", .{spec_path});
            return err;
        };
    }
}

test "conformance corpus completeness rejects orphan ora files" {
    var tmp = testing.tmpDir(.{});
    defer tmp.cleanup();

    try tmp.dir.writeFile(std.testing.io, .{
        .sub_path = "orphan.ora",
        .data = "contract Orphan {}",
    });

    const corpus_dir = try runner.pathFromTmpAlloc(testing.allocator, tmp, "");
    defer testing.allocator.free(corpus_dir);
    try testing.expectError(error.MissingSidecar, runner.checkCorpusSidecars(testing.allocator, corpus_dir));
}

test "conformance corpus completeness accepts explicit skips" {
    var tmp = testing.tmpDir(.{});
    defer tmp.cleanup();

    try tmp.dir.writeFile(std.testing.io, .{
        .sub_path = "orphan.ora",
        .data = "contract Orphan {}",
    });
    try tmp.dir.writeFile(std.testing.io, .{
        .sub_path = "SKIP",
        .data = "# known intentionally unchecked case\norphan.ora\n",
    });

    const corpus_dir = try runner.pathFromTmpAlloc(testing.allocator, tmp, "");
    defer testing.allocator.free(corpus_dir);
    try runner.checkCorpusSidecars(testing.allocator, corpus_dir);
}

test "conformance arg encoder writes static ABI words" {
    const wires = [_][]const u8{ "address", "uint8", "bool", "bytes4" };
    const args = [_]types.ArgValue{
        .{ .literal = "0x2000000000000000000000000000000000000000" },
        .{ .literal = "255" },
        .{ .boolean = true },
        .{ .literal = "0xabcdef12" },
    };

    const encoded = try abi.encodeArgs(testing.allocator, &wires, &args);
    defer testing.allocator.free(encoded);
    try testing.expectEqual(@as(usize, 128), encoded.len);

    try testing.expect(std.mem.allEqual(u8, encoded[0..12], 0));
    try testing.expectEqual(@as(u8, 0x20), encoded[12]);
    try testing.expect(std.mem.allEqual(u8, encoded[13..32], 0));
    try testing.expectEqual(@as(u256, 255), std.mem.readInt(u256, encoded[32..64], .big));
    try testing.expect(std.mem.allEqual(u8, encoded[64..95], 0));
    try testing.expectEqual(@as(u8, 1), encoded[95]);
    try testing.expectEqualSlices(u8, &[_]u8{ 0xab, 0xcd, 0xef, 0x12 }, encoded[96..100]);
    try testing.expect(std.mem.allEqual(u8, encoded[100..128], 0));
}

test "conformance arg encoder writes dynamic arrays and tuple aggregates" {
    const fixed_array_wires = [_][]const u8{"uint256[4]"};
    const fixed_array_args = [_]types.ArgValue{.{ .literal = "[3,5,8,13]" }};
    const fixed_array = try abi.encodeArgs(testing.allocator, &fixed_array_wires, &fixed_array_args);
    defer testing.allocator.free(fixed_array);
    try testing.expectEqual(@as(usize, 128), fixed_array.len);
    try testing.expectEqual(@as(u256, 3), std.mem.readInt(u256, fixed_array[0..32], .big));
    try testing.expectEqual(@as(u256, 5), std.mem.readInt(u256, fixed_array[32..64], .big));
    try testing.expectEqual(@as(u256, 8), std.mem.readInt(u256, fixed_array[64..96], .big));
    try testing.expectEqual(@as(u256, 13), std.mem.readInt(u256, fixed_array[96..128], .big));

    const dynamic_array_wires = [_][]const u8{"uint256[]"};
    const dynamic_array_args = [_]types.ArgValue{.{ .literal = "[5,8,13]" }};
    const dynamic_array = try abi.encodeArgs(testing.allocator, &dynamic_array_wires, &dynamic_array_args);
    defer testing.allocator.free(dynamic_array);
    try testing.expectEqual(@as(usize, 160), dynamic_array.len);
    try testing.expectEqual(@as(u256, 32), std.mem.readInt(u256, dynamic_array[0..32], .big));
    try testing.expectEqual(@as(u256, 3), std.mem.readInt(u256, dynamic_array[32..64], .big));
    try testing.expectEqual(@as(u256, 5), std.mem.readInt(u256, dynamic_array[64..96], .big));
    try testing.expectEqual(@as(u256, 8), std.mem.readInt(u256, dynamic_array[96..128], .big));
    try testing.expectEqual(@as(u256, 13), std.mem.readInt(u256, dynamic_array[128..160], .big));

    const tuple_array_wires = [_][]const u8{"(uint256,uint256,uint256)[]"};
    const tuple_array_args = [_]types.ArgValue{.{ .literal = "[(11,12,13),(21,22,23)]" }};
    const tuple_array = try abi.encodeArgs(testing.allocator, &tuple_array_wires, &tuple_array_args);
    defer testing.allocator.free(tuple_array);
    try testing.expectEqual(@as(u256, 32), std.mem.readInt(u256, tuple_array[0..32], .big));
    try testing.expectEqual(@as(u256, 2), std.mem.readInt(u256, tuple_array[32..64], .big));
    try testing.expectEqual(@as(u256, 11), std.mem.readInt(u256, tuple_array[64..96], .big));
    try testing.expectEqual(@as(u256, 23), std.mem.readInt(u256, tuple_array[224..256], .big));

    const dynamic_tuple_array_wires = [_][]const u8{"(uint256,uint256[],uint256)[]"};
    const dynamic_tuple_array_args = [_]types.ArgValue{.{ .literal = "[(11,[12,13],14),(21,[22,23,24],25)]" }};
    const dynamic_tuple_array = try abi.encodeArgs(testing.allocator, &dynamic_tuple_array_wires, &dynamic_tuple_array_args);
    defer testing.allocator.free(dynamic_tuple_array);
    try testing.expectEqual(@as(u256, 32), std.mem.readInt(u256, dynamic_tuple_array[0..32], .big));
    try testing.expectEqual(@as(u256, 2), std.mem.readInt(u256, dynamic_tuple_array[32..64], .big));
    try testing.expectEqual(@as(u256, 64), std.mem.readInt(u256, dynamic_tuple_array[64..96], .big));
    try testing.expectEqual(@as(u256, 256), std.mem.readInt(u256, dynamic_tuple_array[96..128], .big));
}

test "conformance arg encoder rejects malformed static args" {
    const uint8_wires = [_][]const u8{"uint8"};
    const too_wide_uint = [_]types.ArgValue{.{ .literal = "300" }};
    try testing.expectError(error.ValueOutOfRange, abi.encodeArgs(testing.allocator, &uint8_wires, &too_wide_uint));

    const address_wires = [_][]const u8{"address"};
    const short_address = [_]types.ArgValue{.{ .literal = "0x1234" }};
    try testing.expectError(error.InvalidHex, abi.encodeArgs(testing.allocator, &address_wires, &short_address));

    const bool_wires = [_][]const u8{"bool"};
    const invalid_bool = [_]types.ArgValue{.{ .literal = "2" }};
    try testing.expectError(error.InvalidBoolLiteral, abi.encodeArgs(testing.allocator, &bool_wires, &invalid_bool));
}

test "conformance arg encoder rejects wrong argument counts" {
    const wires = [_][]const u8{ "uint256", "uint256" };
    const args = [_]types.ArgValue{.{ .literal = "1" }};
    try testing.expectError(error.ArgumentCountMismatch, abi.encodeArgs(testing.allocator, &wires, &args));
}

test "conformance slot expressions compute mapping and nested mapping slots" {
    try testing.expectEqual(slots.mappingSlot("address", "0x2000000000000000000000000000000000000000", 0), try slots.parseSlotExpression("map(address,0x2000000000000000000000000000000000000000,0)"));

    var alpha_hash: [32]u8 = undefined;
    std.crypto.hash.sha3.Keccak256.hash("alpha", &alpha_hash, .{});
    const alpha_word = std.mem.readInt(u256, &alpha_hash, .big);
    var alpha_slot_input: [64]u8 = undefined;
    std.mem.writeInt(u256, alpha_slot_input[0..32], alpha_word, .big);
    std.mem.writeInt(u256, alpha_slot_input[32..64], 0, .big);
    var alpha_slot_hash: [32]u8 = undefined;
    std.crypto.hash.sha3.Keccak256.hash(&alpha_slot_input, &alpha_slot_hash, .{});
    try testing.expectEqual(
        std.mem.readInt(u256, &alpha_slot_hash, .big),
        try slots.parseSlotExpression("map(string,alpha,0)"),
    );

    const nested = try slots.parseSlotExpression("map(address,0x3000000000000000000000000000000000000000,map(address,0x2000000000000000000000000000000000000000,0))");
    const manual = try slots.mappingSlot("address", "0x3000000000000000000000000000000000000000", try slots.mappingSlot("address", "0x2000000000000000000000000000000000000000", 0));
    try testing.expectEqual(manual, nested);
    try testing.expectEqual(manual +% 2, try slots.parseSlotExpression("add(map(address,0x3000000000000000000000000000000000000000,map(address,0x2000000000000000000000000000000000000000,0)),2)"));
}

test "conformance slot expressions compute domain-separated computed-storage slots" {
    const root = try slots.parseSlotExpression("computed(ora.test.computed.expr)");
    const same_root = try slots.parseSlotExpression("computed(ora.test.computed.expr)");
    const one_key = try slots.parseSlotExpression("computed(ora.test.computed.expr,address,0x1000000000000000000000000000000000000001)");
    const two_keys_zero = try slots.parseSlotExpression("computed(ora.test.computed.expr,address,0x1000000000000000000000000000000000000001,uint256,0)");
    const other_namespace = try slots.parseSlotExpression("computed(ora.test.computed.other,address,0x1000000000000000000000000000000000000001)");

    try testing.expectEqual(root, same_root);
    try testing.expect(root != one_key);
    try testing.expect(one_key != two_keys_zero);
    try testing.expect(one_key != other_namespace);
    try testing.expectEqual(two_keys_zero +% 1, try slots.parseSlotExpression("add(computed(ora.test.computed.expr,address,0x1000000000000000000000000000000000000001,uint256,0),1)"));
}

test "conformance static return comparison requires canonical narrow signed words" {
    var zero_extended: [32]u8 = [_]u8{0} ** 32;
    zero_extended[30] = 0xff;
    zero_extended[31] = 0xf9;
    try testing.expectError(error.NonCanonicalAbiReturn, abi.expectStaticReturn("int16", .{ .literal = "-7" }, &zero_extended));

    var sign_extended: [32]u8 = [_]u8{0xff} ** 32;
    sign_extended[30] = 0xff;
    sign_extended[31] = 0xf9;
    try abi.expectStaticReturn("int16", .{ .literal = "-7" }, &sign_extended);

    try testing.expectEqual(@as(i256, -7), abi.decodeSignedWord(&zero_extended, 16));
}

test "conformance spec parser rejects unknown keys" {
    const source =
        \\[deploy]
        \\caller = "0x1000000000000000000000000000000000000000"
        \\value = 0
        \\args = []
        \\surprise = 1
        \\
        \\[[call]]
        \\fn = "get()"
        \\caller = "0x1000000000000000000000000000000000000000"
        \\value = 0
        \\args = []
        \\returns = { u256 = 0 }
    ;
    try testing.expectError(error.UnknownKey, spec.parse(testing.allocator, source));
}

test "conformance spec parser rejects calls without outcomes" {
    const source =
        \\[deploy]
        \\caller = "0x1000000000000000000000000000000000000000"
        \\value = 0
        \\args = []
        \\
        \\[[call]]
        \\fn = "get()"
        \\caller = "0x1000000000000000000000000000000000000000"
        \\value = 0
        \\args = []
    ;
    try testing.expectError(error.MissingOutcome, spec.parse(testing.allocator, source));
}

test "conformance spec parser rejects calls with multiple outcomes" {
    const source =
        \\[deploy]
        \\caller = "0x1000000000000000000000000000000000000000"
        \\value = 0
        \\args = []
        \\
        \\[[call]]
        \\fn = "get()"
        \\caller = "0x1000000000000000000000000000000000000000"
        \\value = 0
        \\args = []
        \\returns = { u256 = 0 }
        \\reverts = {}
    ;
    try testing.expectError(error.MultipleOutcomes, spec.parse(testing.allocator, source));
}

test "conformance spec parser accepts raw calldata and rejects typed returns on it" {
    const ok_source =
        \\[deploy]
        \\caller = "0x1000000000000000000000000000000000000000"
        \\value = 0
        \\args = []
        \\
        \\[[call]]
        \\calldata = "0xdeadbeef"
        \\caller = "0x1000000000000000000000000000000000000000"
        \\value = 0
        \\reverts = {}
    ;
    var parsed = try spec.parse(testing.allocator, ok_source);
    defer parsed.deinit();
    try testing.expect(parsed.value.calls[0].calldata != null);
    try testing.expect(parsed.value.calls[0].@"fn" == null);

    const typed_return =
        \\[deploy]
        \\caller = "0x1000000000000000000000000000000000000000"
        \\value = 0
        \\args = []
        \\
        \\[[call]]
        \\calldata = "0xdeadbeef"
        \\caller = "0x1000000000000000000000000000000000000000"
        \\value = 0
        \\returns = { u256 = 0 }
    ;
    try testing.expectError(error.RawCalldataNeedsSucceedsOrReverts, spec.parse(testing.allocator, typed_return));

    const both =
        \\[deploy]
        \\caller = "0x1000000000000000000000000000000000000000"
        \\value = 0
        \\args = []
        \\
        \\[[call]]
        \\fn = "go()"
        \\calldata = "0xdeadbeef"
        \\caller = "0x1000000000000000000000000000000000000000"
        \\value = 0
        \\reverts = {}
    ;
    try testing.expectError(error.InvalidInvocationForm, spec.parse(testing.allocator, both));
}

test "conformance spec parser parses exact revert data and rejects unknown revert keys" {
    const source =
        \\[deploy]
        \\caller = "0x1000000000000000000000000000000000000000"
        \\value = 0
        \\args = []
        \\
        \\[[call]]
        \\fn = "get()"
        \\caller = "0x1000000000000000000000000000000000000000"
        \\value = 0
        \\args = []
        \\reverts = { data = "0x000000000000000000000000000000000000000000000000000000000000000a" }
    ;
    var parsed = try spec.parse(testing.allocator, source);
    defer parsed.deinit();
    const outcome = parsed.value.calls[0].outcome;
    try testing.expect(outcome == .reverts_data);
    try testing.expectEqual(@as(usize, 32), outcome.reverts_data.len);
    try testing.expectEqual(@as(u8, 0x0a), outcome.reverts_data[31]);

    const bad_source =
        \\[deploy]
        \\caller = "0x1000000000000000000000000000000000000000"
        \\value = 0
        \\args = []
        \\
        \\[[call]]
        \\fn = "get()"
        \\caller = "0x1000000000000000000000000000000000000000"
        \\value = 0
        \\args = []
        \\reverts = { payload = "0x0a" }
    ;
    try testing.expectError(error.UnsupportedRevertExpectation, spec.parse(testing.allocator, bad_source));
}

test "conformance spec parser accepts succeeds and rejects multiple outcomes with it" {
    const ok_source =
        \\[deploy]
        \\caller = "0x1000000000000000000000000000000000000000"
        \\value = 0
        \\args = []
        \\
        \\[[call]]
        \\fn = "go()"
        \\caller = "0x1000000000000000000000000000000000000000"
        \\value = 0
        \\args = []
        \\succeeds = {}
    ;
    var parsed = try spec.parse(testing.allocator, ok_source);
    defer parsed.deinit();
    try testing.expect(parsed.value.calls[0].outcome == .succeeds_any);

    const bad_source =
        \\[deploy]
        \\caller = "0x1000000000000000000000000000000000000000"
        \\value = 0
        \\args = []
        \\
        \\[[call]]
        \\fn = "go()"
        \\caller = "0x1000000000000000000000000000000000000000"
        \\value = 0
        \\args = []
        \\succeeds = {}
        \\returns = { u256 = 0 }
    ;
    try testing.expectError(error.MultipleOutcomes, spec.parse(testing.allocator, bad_source));
}
