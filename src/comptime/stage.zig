//! Stage Validity
//!
//! 3-way classification for compile-time evaluation:
//! - comptime_only: Must evaluate at comptime, cannot reach runtime
//! - comptime_ok: Can evaluate at comptime or runtime
//! - runtime_only: Can only evaluate at runtime

/// Stage validity tag for IR operations/intrinsics
pub const Stage = enum {
    /// Must be evaluated at comptime; cannot reach runtime lowering.
    /// Examples: @TypeOf, @sizeOf, type introspection, meta operators
    comptime_only,

    /// Can be evaluated at comptime or runtime.
    /// Examples: arithmetic, comparisons, structural ops, bounded loops
    comptime_ok,

    /// Can only be evaluated at runtime.
    /// Examples: storage access, external calls, chain state (msg.sender, block.*)
    runtime_only,

    /// Check if this stage is valid in must_eval mode
    pub fn validInMustEval(self: Stage) bool {
        return self != .runtime_only;
    }

    /// Check if this stage can be lowered to runtime
    pub fn canLowerToRuntime(self: Stage) bool {
        return self != .comptime_only;
    }

    /// Get error message for stage violation
    pub fn violationMessage(self: Stage) []const u8 {
        return switch (self) {
            .comptime_only => "comptime-only operation cannot be used at runtime",
            .runtime_only => "runtime-only operation cannot be used at comptime",
            .comptime_ok => unreachable,
        };
    }
};

// ============================================================================
// Stage Classification Helpers
// ============================================================================

/// Classify a binary operation's stage
pub fn classifyBinaryOp(op: anytype) Stage {
    _ = op;
    // All arithmetic/comparison/bitwise ops are comptime_ok
    return .comptime_ok;
}

/// Classify a unary operation's stage
pub fn classifyUnaryOp(op: anytype) Stage {
    _ = op;
    return .comptime_ok;
}

/// Check if an intrinsic is runtime-only based on name
pub fn isRuntimeOnlyIntrinsic(name: []const u8) bool {
    const runtime_intrinsics = [_][]const u8{
        "msg.sender",
        "msg.value",
        "block.timestamp",
        "block.number",
        "block.coinbase",
        "block.difficulty",
        "block.gaslimit",
        "tx.origin",
        "tx.gasprice",
        "gasleft",
        "selfdestruct",
        "call",
        "delegatecall",
        "staticcall",
        "create",
        "create2",
        "log0",
        "log1",
        "log2",
        "log3",
        "log4",
        "sload",
        "sstore",
        "tload",
        "tstore",
    };

    for (runtime_intrinsics) |ri| {
        if (std.mem.eql(u8, name, ri)) return true;
    }
    return false;
}

/// Check if an intrinsic is comptime-only based on name
pub fn isComptimeOnlyIntrinsic(name: []const u8) bool {
    const comptime_intrinsics = [_][]const u8{
        "@TypeOf",
        "@typeInfo",
        "@sizeOf",
        "@alignOf",
        "@bitSizeOf",
        "@hasField",
        "@hasDecl",
        "@typeName",
        "@errorName",
    };

    for (comptime_intrinsics) |ci| {
        if (std.mem.eql(u8, name, ci)) return true;
    }
    return false;
}

const std = @import("std");

// ============================================================================
// Tests
// ============================================================================

test "Stage validity checks" {
    try std.testing.expect(Stage.comptime_only.validInMustEval());
    try std.testing.expect(Stage.comptime_ok.validInMustEval());
    try std.testing.expect(!Stage.runtime_only.validInMustEval());

    try std.testing.expect(!Stage.comptime_only.canLowerToRuntime());
    try std.testing.expect(Stage.comptime_ok.canLowerToRuntime());
    try std.testing.expect(Stage.runtime_only.canLowerToRuntime());
}

test "runtime-only intrinsic detection" {
    try std.testing.expect(isRuntimeOnlyIntrinsic("msg.sender"));
    try std.testing.expect(isRuntimeOnlyIntrinsic("sload"));
    try std.testing.expect(!isRuntimeOnlyIntrinsic("add"));
}

test "comptime-only intrinsic detection" {
    try std.testing.expect(isComptimeOnlyIntrinsic("@TypeOf"));
    try std.testing.expect(isComptimeOnlyIntrinsic("@sizeOf"));
    try std.testing.expect(!isComptimeOnlyIntrinsic("msg.sender"));
}
