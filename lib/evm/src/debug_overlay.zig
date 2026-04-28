//! Source-gutter overlay mode for the debugger TUI.
//!
//! Pure data type with simple parse / cycle / name behavior. Lives in
//! its own module so the parser is testable and so other front-ends
//! (probe, future DAP) can share the same enum without pulling in the
//! TUI machinery.

const std = @import("std");

pub const OverlayMode = enum {
    none,
    coverage,
    gas,
    folded,
    hoist,

    pub fn next(self: OverlayMode) OverlayMode {
        return switch (self) {
            .none => .coverage,
            .coverage => .gas,
            .gas => .folded,
            .folded => .hoist,
            .hoist => .none,
        };
    }

    pub fn name(self: OverlayMode) []const u8 {
        return switch (self) {
            .none => "none",
            .coverage => "coverage",
            .gas => "gas",
            .folded => "folded",
            .hoist => "hoist",
        };
    }

    /// Parse a user-typed mode name. Accepts a few short aliases
    /// (`cov`, `fold`, `hoisted`) so the in-app help line can stay
    /// short.
    pub fn parse(text: []const u8) ?OverlayMode {
        if (std.mem.eql(u8, text, "none")) return .none;
        if (std.mem.eql(u8, text, "coverage") or std.mem.eql(u8, text, "cov")) return .coverage;
        if (std.mem.eql(u8, text, "gas")) return .gas;
        if (std.mem.eql(u8, text, "folded") or std.mem.eql(u8, text, "fold")) return .folded;
        if (std.mem.eql(u8, text, "hoist") or std.mem.eql(u8, text, "hoisted")) return .hoist;
        return null;
    }
};

const testing = std.testing;

test "OverlayMode: cycle covers all modes and returns to none" {
    var mode: OverlayMode = .none;
    var seen = [_]bool{false} ** @typeInfo(OverlayMode).@"enum".fields.len;
    var iters: usize = 0;
    while (iters < seen.len) : (iters += 1) {
        seen[@intFromEnum(mode)] = true;
        mode = mode.next();
    }
    try testing.expectEqual(OverlayMode.none, mode);
    for (seen) |b| try testing.expect(b);
}

test "OverlayMode: parse round-trips canonical names" {
    inline for (&[_]OverlayMode{ .none, .coverage, .gas, .folded, .hoist }) |m| {
        try testing.expectEqual(m, OverlayMode.parse(m.name()).?);
    }
}

test "OverlayMode: parse accepts aliases" {
    try testing.expectEqual(OverlayMode.coverage, OverlayMode.parse("cov").?);
    try testing.expectEqual(OverlayMode.folded, OverlayMode.parse("fold").?);
    try testing.expectEqual(OverlayMode.hoist, OverlayMode.parse("hoisted").?);
}

test "OverlayMode: parse rejects unknown" {
    try testing.expectEqual(@as(?OverlayMode, null), OverlayMode.parse("nope"));
    try testing.expectEqual(@as(?OverlayMode, null), OverlayMode.parse(""));
}
