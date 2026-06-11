const std = @import("std");
const testing = std.testing;

const abi = @import("abi.zig");
const properties = @import("properties.zig");
const runner = @import("runner.zig");
const slots = @import("slots.zig");
const spec = @import("spec.zig");
const types = @import("types.zig");

test "conformance properties execute and catch planted failures" {
    try properties.run(testing.allocator);
}

test "conformance corpus files have sidecars or explicit skips" {
    try runner.checkCorpusSidecars(testing.allocator, types.CONFORMANCE_DIR_REL);
}

test "conformance corpus specs execute" {
    std.fs.cwd().access(types.ORA_BINARY_REL, .{}) catch |err| switch (err) {
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

    try tmp.dir.writeFile(.{
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

    try tmp.dir.writeFile(.{
        .sub_path = "orphan.ora",
        .data = "contract Orphan {}",
    });
    try tmp.dir.writeFile(.{
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

test "conformance static return comparison decodes narrow signed words" {
    var zero_extended: [32]u8 = [_]u8{0} ** 32;
    zero_extended[30] = 0xff;
    zero_extended[31] = 0xf9;
    try abi.expectStaticReturn("int16", .{ .literal = "-7" }, &zero_extended);
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
