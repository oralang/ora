const std = @import("std");

// SIR operation metadata used by parser legality, debug codegen, and release
// scheduling.
//
// This is deliberately not the EVM opcode table. `asm.zig` owns raw byte values;
// this file owns the SIR-level shape of each instruction:
//
// - how many SSA value operands it consumes,
// - how many SSA values it produces,
// - whether it carries one non-value immediate operand, and
// - whether the mnemonic is part of a parameterized family such as `mload256`.
//
// Keeping those concepts separate matters. Some SIR operations lower to one EVM
// opcode (`add`), some are compiler/runtime intrinsics (`salloc`,
// `runtime_length`), and some are not normal instructions at all in Sinora's IR
// because they live as block terminators (`return`, `revert`, `stop`,
// `selfdestruct`).

pub const Extra = enum {
    // No non-value operand after the SSA inputs.
    none,
    // One numeric immediate after the SSA inputs, e.g. `const 42` or
    // `salloc 64`. The immediate is syntax metadata, not a value use.
    numeric,
    // One data segment reference after the SSA inputs, e.g.
    // `data_offset .payload`.
    data_ref,
};

pub const Fixed = struct {
    // The exact SIR mnemonic is NOT stored here: it is already the key in
    // `fixed_spec_map` (`std.StaticStringMap`), so duplicating the slice in the
    // value would make every table entry larger and every lookup copy more.
    // SSA value operands consumed by the instruction. Non-value immediates are
    // counted separately through `extra`.
    inputs: usize,
    // SSA values defined by the instruction.
    outputs: usize,
    extra: Extra = .none,
    // Plank stack scheduling may flip the first two operands for this operation
    // while preserving semantics. This is scheduler metadata, not EVM metadata:
    // `lt`/`gt` and `slt`/`sgt` are marked flippable because the scheduler knows
    // how to swap the comparison opcode with the operand order.
    flippable: bool = false,

    pub fn operandCount(self: Fixed) usize {
        return self.inputs + @intFromBool(self.extra != .none);
    }
};

pub const Memory = struct {
    // Width in bits from the mnemonic suffix. Plank text spells memory ops as
    // `mload8` ... `mload256` / `mstore8` ... `mstore256`; internally the EVM
    // still uses `MLOAD`/`MSTORE` plus masking/extension in the codegen layer.
    bits: u16,
    inputs: usize,
    outputs: usize,
};

pub const Spec = union(enum) {
    // Concrete mnemonic with fixed arity.
    fixed: Fixed,
    // Parameterized memory op families.
    memory_load: Memory,
    memory_store: Memory,
    // `icall` arity depends on the referenced function signature, so legality
    // validates it outside this fixed table.
    internal_call,
};

pub fn lookup(mnemonic: []const u8) ?Spec {
    if (parseMemory(mnemonic, "mload")) |bits| {
        return .{ .memory_load = .{
            .bits = bits,
            .inputs = 1,
            .outputs = 1,
        } };
    }
    if (parseMemory(mnemonic, "mstore")) |bits| {
        return .{ .memory_store = .{
            .bits = bits,
            .inputs = 2,
            .outputs = 0,
        } };
    }
    if (std.mem.eql(u8, mnemonic, "icall")) return .internal_call;

    if (fixed_spec_map.get(mnemonic)) |spec| return .{ .fixed = spec };
    return null;
}

pub fn isFlippable(mnemonic: []const u8) bool {
    return if (fixed_spec_map.get(mnemonic)) |spec| spec.flippable else false;
}

fn parseMemory(mnemonic: []const u8, prefix: []const u8) ?u16 {
    if (!std.mem.startsWith(u8, mnemonic, prefix)) return null;
    const suffix = mnemonic[prefix.len..];
    if (suffix.len == 0) return null;
    const bits = parseDecimalU16(suffix) orelse return null;
    if (bits < 8 or bits > 256 or bits % 8 != 0) return null;
    return bits;
}

fn parseDecimalU16(text: []const u8) ?u16 {
    var value: u16 = 0;
    for (text) |ch| {
        if (ch < '0' or ch > '9') return null;
        const digit: u16 = ch - '0';
        value = std.math.mul(u16, value, 10) catch return null;
        value = std.math.add(u16, value, digit) catch return null;
    }
    return value;
}

fn fixed(comptime inputs: usize, comptime outputs: usize) Fixed {
    return .{
        .inputs = inputs,
        .outputs = outputs,
    };
}

fn fixedFlip(comptime inputs: usize, comptime outputs: usize) Fixed {
    return .{
        .inputs = inputs,
        .outputs = outputs,
        .flippable = true,
    };
}

fn fixedExtra(comptime inputs: usize, comptime outputs: usize, comptime extra: Extra) Fixed {
    return .{
        .inputs = inputs,
        .outputs = outputs,
        .extra = extra,
    };
}

const fixed_spec_map = std.StaticStringMap(Fixed).initComptime(.{
    .{ "add", fixedFlip(2, 1) },
    .{ "mul", fixedFlip(2, 1) },
    .{ "sub", fixed(2, 1) },
    .{ "div", fixed(2, 1) },
    .{ "sdiv", fixed(2, 1) },
    .{ "mod", fixed(2, 1) },
    .{ "smod", fixed(2, 1) },
    .{ "addmod", fixedFlip(3, 1) },
    .{ "mulmod", fixedFlip(3, 1) },
    .{ "exp", fixed(2, 1) },
    .{ "signextend", fixed(2, 1) },

    .{ "lt", fixedFlip(2, 1) },
    .{ "gt", fixedFlip(2, 1) },
    .{ "slt", fixedFlip(2, 1) },
    .{ "sgt", fixedFlip(2, 1) },
    .{ "eq", fixedFlip(2, 1) },
    .{ "iszero", fixed(1, 1) },
    .{ "and", fixedFlip(2, 1) },
    .{ "or", fixedFlip(2, 1) },
    .{ "xor", fixedFlip(2, 1) },
    .{ "not", fixed(1, 1) },
    .{ "byte", fixed(2, 1) },
    .{ "shl", fixed(2, 1) },
    .{ "shr", fixed(2, 1) },
    .{ "sar", fixed(2, 1) },

    .{ "keccak256", fixed(2, 1) },

    .{ "address", fixed(0, 1) },
    .{ "balance", fixed(1, 1) },
    .{ "origin", fixed(0, 1) },
    .{ "caller", fixed(0, 1) },
    .{ "callvalue", fixed(0, 1) },
    .{ "calldataload", fixed(1, 1) },
    .{ "calldatasize", fixed(0, 1) },
    .{ "calldatacopy", fixed(3, 0) },
    .{ "codesize", fixed(0, 1) },
    .{ "codecopy", fixed(3, 0) },
    .{ "gasprice", fixed(0, 1) },
    .{ "extcodesize", fixed(1, 1) },
    .{ "extcodecopy", fixed(4, 0) },
    .{ "returndatasize", fixed(0, 1) },
    .{ "returndatacopy", fixed(3, 0) },
    .{ "extcodehash", fixed(1, 1) },
    .{ "gas", fixed(0, 1) },

    .{ "blockhash", fixed(1, 1) },
    .{ "coinbase", fixed(0, 1) },
    .{ "timestamp", fixed(0, 1) },
    .{ "number", fixed(0, 1) },
    .{ "difficulty", fixed(0, 1) },
    .{ "gaslimit", fixed(0, 1) },
    .{ "chainid", fixed(0, 1) },
    .{ "selfbalance", fixed(0, 1) },
    .{ "basefee", fixed(0, 1) },
    .{ "blobhash", fixed(1, 1) },
    .{ "blobbasefee", fixed(0, 1) },

    .{ "sload", fixed(1, 1) },
    .{ "sstore", fixed(2, 0) },
    .{ "tload", fixed(1, 1) },
    .{ "tstore", fixed(2, 0) },

    .{ "log0", fixed(2, 0) },
    .{ "log1", fixed(3, 0) },
    .{ "log2", fixed(4, 0) },
    .{ "log3", fixed(5, 0) },
    .{ "log4", fixed(6, 0) },

    .{ "create", fixed(3, 1) },
    .{ "create2", fixed(4, 1) },
    .{ "call", fixed(7, 1) },
    .{ "callcode", fixed(7, 1) },
    .{ "delegatecall", fixed(6, 1) },
    .{ "staticcall", fixed(6, 1) },

    .{ "malloc", fixed(1, 1) },
    .{ "mallocany", fixed(1, 1) },
    .{ "freeptr", fixed(0, 1) },
    .{ "salloc", fixedExtra(0, 1, .numeric) },
    .{ "sallocany", fixedExtra(0, 1, .numeric) },

    .{ "mcopy", fixed(3, 0) },
    .{ "copy", fixed(1, 1) },
    .{ "const", fixedExtra(0, 1, .numeric) },
    .{ "large_const", fixedExtra(0, 1, .numeric) },
    .{ "data_offset", fixedExtra(0, 1, .data_ref) },
    .{ "noop", fixed(0, 0) },

    .{ "runtime_start_offset", fixed(0, 1) },
    .{ "init_end_offset", fixed(0, 1) },
    .{ "runtime_length", fixed(0, 1) },
});

test "lookup resolves fixed and memory operations" {
    const add = lookup("add").?.fixed;
    try std.testing.expect(add.flippable);
    try std.testing.expectEqual(@as(usize, 2), add.inputs);
    try std.testing.expectEqual(@as(usize, 1), add.outputs);
    try std.testing.expectEqual(@as(u16, 256), lookup("mload256").?.memory_load.bits);
    try std.testing.expectEqual(@as(u16, 8), lookup("mstore8").?.memory_store.bits);
    try std.testing.expectEqual(@as(?Spec, null), lookup("mload7"));
}

test "lookup classifies non-value operands" {
    try std.testing.expectEqual(Extra.numeric, lookup("const").?.fixed.extra);
    try std.testing.expectEqual(Extra.numeric, lookup("sallocany").?.fixed.extra);
    try std.testing.expectEqual(Extra.data_ref, lookup("data_offset").?.fixed.extra);
    try std.testing.expectEqual(Spec.internal_call, lookup("icall").?);
}

test "memory operation suffix validation matches Plank text rules" {
    try std.testing.expectEqual(@as(?Spec, null), lookup("mload"));
    try std.testing.expectEqual(@as(?Spec, null), lookup("mload0"));
    try std.testing.expectEqual(@as(?Spec, null), lookup("mload257"));
    try std.testing.expectEqual(@as(?Spec, null), lookup("mload264"));
    try std.testing.expectEqual(@as(?Spec, null), lookup("mload8x"));
    try std.testing.expectEqual(@as(?Spec, null), lookup("mstore-8"));
    try std.testing.expectEqual(@as(u16, 32), lookup("mload32").?.memory_load.bits);
    try std.testing.expectEqual(@as(u16, 240), lookup("mstore240").?.memory_store.bits);
}

test "flippable classification follows Plank scheduler model" {
    try std.testing.expect(isFlippable("add"));
    try std.testing.expect(isFlippable("lt"));
    try std.testing.expect(isFlippable("sgt"));
    try std.testing.expect(isFlippable("xor"));
    try std.testing.expect(!isFlippable("sub"));
    try std.testing.expect(!isFlippable("mload256"));
    try std.testing.expect(!isFlippable("icall"));
}
