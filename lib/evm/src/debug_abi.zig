//! ABI document loader + decoders used by the EVM debugger.
//!
//! Holds the parsed `<contract>.abi.json` and the read-only lookups
//! the TUI needs at runtime: function selector → callable, error
//! selector → custom-error spec, event topic[0] → event spec, plus
//! the wire-type resolution used to pretty-print decoded values.
//!
//! Decoding helpers (`writeAbiWord`, `readU256BE`, `writeU256BE`) sit
//! alongside the loader so callers don't need to reach into the
//! debugger crate just to format a 32-byte word.

const std = @import("std");
const debug_session = @import("debug_session.zig");

pub const AbiDoc = struct {
    allocator: std.mem.Allocator,
    json_bytes: []u8,
    parsed: std.json.Parsed(std.json.Value),

    pub fn loadFromPath(allocator: std.mem.Allocator, path: []const u8) !AbiDoc {
        const json_bytes = try debug_session.loadArtifact(allocator, path);
        errdefer allocator.free(json_bytes);

        const parsed = try std.json.parseFromSlice(std.json.Value, allocator, json_bytes, .{});
        errdefer parsed.deinit();

        return .{
            .allocator = allocator,
            .json_bytes = json_bytes,
            .parsed = parsed,
        };
    }

    pub fn deinit(self: *AbiDoc) void {
        self.parsed.deinit();
        self.allocator.free(self.json_bytes);
    }

    pub fn findCallableBySelector(self: *const AbiDoc, selector: [4]u8) ?std.json.Value {
        return self.findCallableBySelectorOfKind(selector, "function");
    }

    pub fn findErrorBySelector(self: *const AbiDoc, selector: [4]u8) ?std.json.Value {
        return self.findCallableBySelectorOfKind(selector, "error");
    }

    fn findCallableBySelectorOfKind(self: *const AbiDoc, selector: [4]u8, kind_filter: []const u8) ?std.json.Value {
        const callables = self.parsed.value.object.get("callables") orelse return null;
        if (callables != .array) return null;

        const selector_int = std.mem.readInt(u32, &selector, .big);
        var selector_buf: [10]u8 = undefined;
        const selector_text = std.fmt.bufPrint(&selector_buf, "0x{x:0>8}", .{selector_int}) catch return null;

        for (callables.array.items) |callable| {
            if (callable != .object) continue;
            const kind = callable.object.get("kind") orelse continue;
            if (kind != .string or !std.mem.eql(u8, kind.string, kind_filter)) continue;

            const wire = callable.object.get("wire") orelse continue;
            if (wire != .object) continue;
            const evm_default = wire.object.get("evm-default") orelse continue;
            if (evm_default != .object) continue;
            const sel = evm_default.object.get("selector") orelse continue;
            if (sel != .string) continue;
            if (std.mem.eql(u8, sel.string, selector_text)) return callable;
        }

        return null;
    }

    pub fn findEventByTopic0(self: *const AbiDoc, topic0: u256) ?std.json.Value {
        const callables = self.parsed.value.object.get("callables") orelse return null;
        if (callables != .array) return null;

        for (callables.array.items) |callable| {
            if (callable != .object) continue;
            const kind = callable.object.get("kind") orelse continue;
            if (kind != .string or !std.mem.eql(u8, kind.string, "event")) continue;

            const sig = callable.object.get("signature") orelse continue;
            if (sig != .string) continue;

            var hash: [32]u8 = undefined;
            std.crypto.hash.sha3.Keccak256.hash(sig.string, &hash, .{});
            var event_topic0: u256 = 0;
            for (hash) |byte| event_topic0 = (event_topic0 << 8) | byte;
            if (event_topic0 == topic0) return callable;
        }

        return null;
    }

    pub fn wireTypeForTypeId(self: *const AbiDoc, type_id: []const u8) ?[]const u8 {
        const types = self.parsed.value.object.get("types") orelse return null;
        if (types != .object) return null;
        const type_value = types.object.get(type_id) orelse return null;
        if (type_value != .object) return null;

        if (type_value.object.get("wire")) |wire| {
            if (wire == .object) {
                if (wire.object.get("evm-default")) |evm_default| {
                    if (evm_default == .object) {
                        if (evm_default.object.get("type")) |wire_type| {
                            if (wire_type == .string) return wire_type.string;
                        }
                    }
                }
            }
        }

        if (type_value.object.get("name")) |name| {
            if (name == .string) return name.string;
        }

        return null;
    }

    pub fn findInputWireType(self: *const AbiDoc, callable: std.json.Value, input_name: []const u8) ?[]const u8 {
        if (callable != .object) return null;
        const inputs = callable.object.get("inputs") orelse return null;
        if (inputs != .array) return null;

        for (inputs.array.items) |input| {
            if (input != .object) continue;
            const name = input.object.get("name") orelse continue;
            if (name != .string or !std.mem.eql(u8, name.string, input_name)) continue;

            const type_id = input.object.get("typeId") orelse return null;
            if (type_id != .string) return null;
            return self.wireTypeForTypeId(type_id.string);
        }

        return null;
    }

    pub fn findInputIndex(self: *const AbiDoc, callable: std.json.Value, input_name: []const u8) ?usize {
        _ = self;
        if (callable != .object) return null;
        const inputs = callable.object.get("inputs") orelse return null;
        if (inputs != .array) return null;

        for (inputs.array.items, 0..) |input, i| {
            if (input != .object) continue;
            const name = input.object.get("name") orelse continue;
            if (name == .string and std.mem.eql(u8, name.string, input_name)) return i;
        }

        return null;
    }

    pub fn formatInputType(self: *const AbiDoc, input: std.json.Value) ?[]const u8 {
        if (input != .object) return null;
        const type_id = input.object.get("typeId") orelse return null;
        if (type_id != .string) return null;
        return self.wireTypeForTypeId(type_id.string);
    }

    pub fn isInputIndexed(input: std.json.Value) bool {
        if (input != .object) return false;
        const indexed = input.object.get("indexed") orelse return false;
        return indexed == .bool and indexed.bool;
    }
};

/// Format a single 32-byte ABI-encoded word given its wire type.
/// Falls back to `0x` + hex for non-32-byte input or unknown wire
/// types.
pub fn writeAbiWord(writer: anytype, wire_type: []const u8, word: []const u8) !void {
    if (word.len != 32) {
        try writer.writeAll("0x");
        for (word) |b| try writer.print("{x:0>2}", .{b});
        return;
    }
    if (std.mem.eql(u8, wire_type, "address")) {
        try writer.print("0x{x}", .{word[12..32]});
        return;
    }
    if (std.mem.eql(u8, wire_type, "bool")) {
        try writer.writeAll(if (word[31] == 0) "false" else "true");
        return;
    }
    if (std.mem.startsWith(u8, wire_type, "bytes")) {
        try writer.writeAll("0x");
        for (word) |b| try writer.print("{x:0>2}", .{b});
        return;
    }
    var value: u256 = 0;
    for (word) |b| value = (value << 8) | b;
    try writer.print("{d}", .{value});
}

pub fn readU256BE(word: *const [32]u8) u256 {
    var value: u256 = 0;
    for (word) |b| value = (value << 8) | b;
    return value;
}

pub fn writeU256BE(out: *[32]u8, value: u256) void {
    var v = value;
    var i: usize = 32;
    while (i > 0) {
        i -= 1;
        out[i] = @intCast(v & 0xff);
        v >>= 8;
    }
}

// =============================================================================
// Tests
// =============================================================================

const testing = std.testing;

test "writeAbiWord: address renders as 0x prefix" {
    var buf: [128]u8 = undefined;
    var stream = std.io.fixedBufferStream(&buf);
    var word: [32]u8 = .{0} ** 32;
    word[12] = 0xab;
    word[31] = 0xcd;
    try writeAbiWord(stream.writer(), "address", &word);
    try testing.expect(std.mem.startsWith(u8, stream.getWritten(), "0x"));
}

test "writeAbiWord: bool" {
    var buf: [16]u8 = undefined;
    var stream = std.io.fixedBufferStream(&buf);
    const false_word: [32]u8 = .{0} ** 32;
    try writeAbiWord(stream.writer(), "bool", &false_word);
    try testing.expectEqualStrings("false", stream.getWritten());

    stream.reset();
    var true_word: [32]u8 = .{0} ** 32;
    true_word[31] = 1;
    try writeAbiWord(stream.writer(), "bool", &true_word);
    try testing.expectEqualStrings("true", stream.getWritten());
}

test "writeAbiWord: uint256 decimal" {
    var buf: [128]u8 = undefined;
    var stream = std.io.fixedBufferStream(&buf);
    var word: [32]u8 = .{0} ** 32;
    word[31] = 42;
    try writeAbiWord(stream.writer(), "uint256", &word);
    try testing.expectEqualStrings("42", stream.getWritten());
}

test "writeU256BE / readU256BE round-trip" {
    var word: [32]u8 = undefined;
    writeU256BE(&word, 0x1234567890abcdef);
    try testing.expectEqual(@as(u256, 0x1234567890abcdef), readU256BE(&word));

    writeU256BE(&word, 0);
    try testing.expectEqual(@as(u256, 0), readU256BE(&word));

    writeU256BE(&word, std.math.maxInt(u256));
    try testing.expectEqual(@as(u256, std.math.maxInt(u256)), readU256BE(&word));
}
