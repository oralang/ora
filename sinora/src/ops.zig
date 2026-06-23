const std = @import("std");

pub const Extra = enum {
    none,
    numeric,
    data_ref,
};

pub const Fixed = struct {
    mnemonic: []const u8,
    inputs: usize,
    outputs: usize,
    extra: Extra = .none,

    pub fn operandCount(self: Fixed) usize {
        return self.inputs + @intFromBool(self.extra != .none);
    }
};

pub const Memory = struct {
    bits: u16,
    inputs: usize,
    outputs: usize,
};

pub const Spec = union(enum) {
    fixed: Fixed,
    memory_load: Memory,
    memory_store: Memory,
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

    for (fixed_specs) |spec| {
        if (std.mem.eql(u8, mnemonic, spec.mnemonic)) return .{ .fixed = spec };
    }
    return null;
}

fn parseMemory(mnemonic: []const u8, prefix: []const u8) ?u16 {
    if (!std.mem.startsWith(u8, mnemonic, prefix)) return null;
    const suffix = mnemonic[prefix.len..];
    if (suffix.len == 0) return null;
    const bits = std.fmt.parseUnsigned(u16, suffix, 10) catch return null;
    if (bits < 8 or bits > 256 or bits % 8 != 0) return null;
    return bits;
}

const fixed_specs = [_]Fixed{
    .{ .mnemonic = "add", .inputs = 2, .outputs = 1 },
    .{ .mnemonic = "mul", .inputs = 2, .outputs = 1 },
    .{ .mnemonic = "sub", .inputs = 2, .outputs = 1 },
    .{ .mnemonic = "div", .inputs = 2, .outputs = 1 },
    .{ .mnemonic = "sdiv", .inputs = 2, .outputs = 1 },
    .{ .mnemonic = "mod", .inputs = 2, .outputs = 1 },
    .{ .mnemonic = "smod", .inputs = 2, .outputs = 1 },
    .{ .mnemonic = "addmod", .inputs = 3, .outputs = 1 },
    .{ .mnemonic = "mulmod", .inputs = 3, .outputs = 1 },
    .{ .mnemonic = "exp", .inputs = 2, .outputs = 1 },
    .{ .mnemonic = "signextend", .inputs = 2, .outputs = 1 },

    .{ .mnemonic = "lt", .inputs = 2, .outputs = 1 },
    .{ .mnemonic = "gt", .inputs = 2, .outputs = 1 },
    .{ .mnemonic = "slt", .inputs = 2, .outputs = 1 },
    .{ .mnemonic = "sgt", .inputs = 2, .outputs = 1 },
    .{ .mnemonic = "eq", .inputs = 2, .outputs = 1 },
    .{ .mnemonic = "iszero", .inputs = 1, .outputs = 1 },
    .{ .mnemonic = "and", .inputs = 2, .outputs = 1 },
    .{ .mnemonic = "or", .inputs = 2, .outputs = 1 },
    .{ .mnemonic = "xor", .inputs = 2, .outputs = 1 },
    .{ .mnemonic = "not", .inputs = 1, .outputs = 1 },
    .{ .mnemonic = "byte", .inputs = 2, .outputs = 1 },
    .{ .mnemonic = "shl", .inputs = 2, .outputs = 1 },
    .{ .mnemonic = "shr", .inputs = 2, .outputs = 1 },
    .{ .mnemonic = "sar", .inputs = 2, .outputs = 1 },

    .{ .mnemonic = "keccak256", .inputs = 2, .outputs = 1 },

    .{ .mnemonic = "address", .inputs = 0, .outputs = 1 },
    .{ .mnemonic = "balance", .inputs = 1, .outputs = 1 },
    .{ .mnemonic = "origin", .inputs = 0, .outputs = 1 },
    .{ .mnemonic = "caller", .inputs = 0, .outputs = 1 },
    .{ .mnemonic = "callvalue", .inputs = 0, .outputs = 1 },
    .{ .mnemonic = "calldataload", .inputs = 1, .outputs = 1 },
    .{ .mnemonic = "calldatasize", .inputs = 0, .outputs = 1 },
    .{ .mnemonic = "calldatacopy", .inputs = 3, .outputs = 0 },
    .{ .mnemonic = "codesize", .inputs = 0, .outputs = 1 },
    .{ .mnemonic = "codecopy", .inputs = 3, .outputs = 0 },
    .{ .mnemonic = "gasprice", .inputs = 0, .outputs = 1 },
    .{ .mnemonic = "extcodesize", .inputs = 1, .outputs = 1 },
    .{ .mnemonic = "extcodecopy", .inputs = 4, .outputs = 0 },
    .{ .mnemonic = "returndatasize", .inputs = 0, .outputs = 1 },
    .{ .mnemonic = "returndatacopy", .inputs = 3, .outputs = 0 },
    .{ .mnemonic = "extcodehash", .inputs = 1, .outputs = 1 },
    .{ .mnemonic = "gas", .inputs = 0, .outputs = 1 },

    .{ .mnemonic = "blockhash", .inputs = 1, .outputs = 1 },
    .{ .mnemonic = "coinbase", .inputs = 0, .outputs = 1 },
    .{ .mnemonic = "timestamp", .inputs = 0, .outputs = 1 },
    .{ .mnemonic = "number", .inputs = 0, .outputs = 1 },
    .{ .mnemonic = "difficulty", .inputs = 0, .outputs = 1 },
    .{ .mnemonic = "gaslimit", .inputs = 0, .outputs = 1 },
    .{ .mnemonic = "chainid", .inputs = 0, .outputs = 1 },
    .{ .mnemonic = "selfbalance", .inputs = 0, .outputs = 1 },
    .{ .mnemonic = "basefee", .inputs = 0, .outputs = 1 },
    .{ .mnemonic = "blobhash", .inputs = 1, .outputs = 1 },
    .{ .mnemonic = "blobbasefee", .inputs = 0, .outputs = 1 },

    .{ .mnemonic = "sload", .inputs = 1, .outputs = 1 },
    .{ .mnemonic = "sstore", .inputs = 2, .outputs = 0 },
    .{ .mnemonic = "tload", .inputs = 1, .outputs = 1 },
    .{ .mnemonic = "tstore", .inputs = 2, .outputs = 0 },

    .{ .mnemonic = "log0", .inputs = 2, .outputs = 0 },
    .{ .mnemonic = "log1", .inputs = 3, .outputs = 0 },
    .{ .mnemonic = "log2", .inputs = 4, .outputs = 0 },
    .{ .mnemonic = "log3", .inputs = 5, .outputs = 0 },
    .{ .mnemonic = "log4", .inputs = 6, .outputs = 0 },

    .{ .mnemonic = "create", .inputs = 3, .outputs = 1 },
    .{ .mnemonic = "create2", .inputs = 4, .outputs = 1 },
    .{ .mnemonic = "call", .inputs = 7, .outputs = 1 },
    .{ .mnemonic = "callcode", .inputs = 7, .outputs = 1 },
    .{ .mnemonic = "delegatecall", .inputs = 6, .outputs = 1 },
    .{ .mnemonic = "staticcall", .inputs = 6, .outputs = 1 },

    .{ .mnemonic = "malloc", .inputs = 1, .outputs = 1 },
    .{ .mnemonic = "mallocany", .inputs = 1, .outputs = 1 },
    .{ .mnemonic = "freeptr", .inputs = 0, .outputs = 1 },
    .{ .mnemonic = "salloc", .inputs = 0, .outputs = 1, .extra = .numeric },
    .{ .mnemonic = "sallocany", .inputs = 0, .outputs = 1, .extra = .numeric },

    .{ .mnemonic = "mcopy", .inputs = 3, .outputs = 0 },
    .{ .mnemonic = "copy", .inputs = 1, .outputs = 1 },
    .{ .mnemonic = "const", .inputs = 0, .outputs = 1, .extra = .numeric },
    .{ .mnemonic = "large_const", .inputs = 0, .outputs = 1, .extra = .numeric },
    .{ .mnemonic = "data_offset", .inputs = 0, .outputs = 1, .extra = .data_ref },
    .{ .mnemonic = "noop", .inputs = 0, .outputs = 0 },

    .{ .mnemonic = "runtime_start_offset", .inputs = 0, .outputs = 1 },
    .{ .mnemonic = "init_end_offset", .inputs = 0, .outputs = 1 },
    .{ .mnemonic = "runtime_length", .inputs = 0, .outputs = 1 },
};

test "lookup resolves fixed and memory operations" {
    try std.testing.expectEqual(@as(?Spec, .{ .fixed = .{ .mnemonic = "add", .inputs = 2, .outputs = 1 } }), lookup("add"));
    try std.testing.expectEqual(@as(u16, 256), lookup("mload256").?.memory_load.bits);
    try std.testing.expectEqual(@as(u16, 8), lookup("mstore8").?.memory_store.bits);
    try std.testing.expectEqual(@as(?Spec, null), lookup("mload7"));
}
