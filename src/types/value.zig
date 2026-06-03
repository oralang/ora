const std = @import("std");

const BigInt = std.math.big.int.Managed;

/// Cross-stage semantic constant value.
///
/// This is the value model exposed by const-eval to sema, HIR, ABI helpers, and
/// debug output. It intentionally preserves arbitrary-precision integer state so
/// sema can perform fit checks before choosing a runtime integer width.
pub const ConstValue = union(enum) {
    integer: BigInt,
    boolean: bool,
    address: u160,
    fixed_bytes: []const u8,
    string: []const u8,
    tuple: []const ConstValue,
};
