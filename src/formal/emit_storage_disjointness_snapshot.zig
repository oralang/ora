//! Emits `formal/Ora/Generated/StorageDisjointnessSnapshot.lean` — data-only
//! rows for the storage `PlaceRef` path-disjointness fixture. The trusted Lean
//! check lives in `formal/Ora/StorageDisjointnessSync.lean`.

const std = @import("std");
const formal = @import("ora_formal");

const obligation = formal.obligation;

const header =
    \\/-
    \\GENERATED — DATA ONLY. Do NOT edit by hand and do NOT add any `theorem`,
    \\`lemma`, `example`, `axiom`, `sorry`, `instance`, `macro`, attribute, or extra
    \\`import` to this file. It contains only primitive fixture rows emitted from
    \\the compiler. The TRUSTED checks live in `Ora/StorageDisjointnessSync.lean`.
    \\
    \\Regenerate with `scripts/check-formal-sync.sh`. Source:
    \\src/formal/emit_storage_disjointness_snapshot.zig,
    \\src/formal/obligation.zig.
    \\-/
    \\
    \\namespace Ora.Generated
    \\
    \\
;

fn boolText(value: bool) []const u8 {
    return if (value) "true" else "false";
}

fn regionName(region: obligation.RegionRef) []const u8 {
    return switch (region) {
        .none => "none",
        .storage => "storage",
        .memory => "memory",
        .transient => "transient",
        .calldata => "calldata",
    };
}

fn writeLeanString(out: anytype, value: []const u8) !void {
    try out.writeByte('"');
    for (value) |byte| {
        switch (byte) {
            '\\' => try out.writeAll("\\\\"),
            '"' => try out.writeAll("\\\""),
            '\n' => try out.writeAll("\\n"),
            '\r' => try out.writeAll("\\r"),
            '\t' => try out.writeAll("\\t"),
            else => try out.writeByte(byte),
        }
    }
    try out.writeByte('"');
}

fn writeStringList(out: anytype, values: []const []const u8) !void {
    if (values.len == 0) return out.writeAll("[]");
    try out.writeByte('[');
    for (values, 0..) |value, index| {
        if (index != 0) try out.writeAll(", ");
        try writeLeanString(out, value);
    }
    try out.writeByte(']');
}

fn writePlaceKey(out: anytype, key: obligation.PlaceKey) !void {
    try out.writeByte('(');
    switch (key) {
        .parameter => |index| {
            try writeLeanString(out, "parameter");
            try out.print(", \"{d}\"", .{index});
        },
        .comptime_parameter => |index| {
            try writeLeanString(out, "comptime_parameter");
            try out.print(", \"{d}\"", .{index});
        },
        .comptime_range_parameter => |index| {
            try writeLeanString(out, "comptime_range_parameter");
            try out.print(", \"{d}\"", .{index});
        },
        .constant => |value| {
            try writeLeanString(out, "constant");
            try out.writeAll(", ");
            try writeLeanString(out, value);
        },
        .msg_sender => {
            try writeLeanString(out, "msg_sender");
            try out.writeAll(", \"\"");
        },
        .tx_origin => {
            try writeLeanString(out, "tx_origin");
            try out.writeAll(", \"\"");
        },
        .unknown => {
            try writeLeanString(out, "unknown");
            try out.writeAll(", \"\"");
        },
    }
    try out.writeByte(')');
}

fn writePlaceKeyList(out: anytype, keys: []const obligation.PlaceKey) !void {
    if (keys.len == 0) return out.writeAll("[]");
    try out.writeByte('[');
    for (keys, 0..) |key, index| {
        if (index != 0) try out.writeAll(", ");
        try writePlaceKey(out, key);
    }
    try out.writeByte(']');
}

fn writePlace(out: anytype, place: obligation.PlaceRef) !void {
    try out.writeByte('(');
    try writeLeanString(out, place.root);
    try out.writeAll(", ");
    try writeLeanString(out, regionName(place.region));
    try out.writeAll(", ");
    try writeStringList(out, place.fields);
    try out.writeAll(", ");
    try writePlaceKeyList(out, place.keys);
    try out.writeByte(')');
}

pub fn main(init: std.process.Init) !void {
    const io = init.io;
    var out_buffer: [1 << 16]u8 = undefined;
    var stdout_writer = std.Io.File.stdout().writer(io, &out_buffer);
    const out = &stdout_writer.interface;

    try out.writeAll(header);
    try out.writeAll(
        \\def storageDisjointnessRows :
        \\    List (String × (String × String × List String × List (String × String)) ×
        \\      (String × String × List String × List (String × String)) × Bool) :=
        \\  [
    );
    for (obligation.storage_disjointness_fixtures, 0..) |fixture, index| {
        if (index != 0) try out.writeAll(",\n   ");
        try out.writeByte('(');
        try writeLeanString(out, fixture.label);
        try out.writeAll(", ");
        try writePlace(out, fixture.lhs);
        try out.writeAll(", ");
        try writePlace(out, fixture.rhs);
        try out.writeAll(", ");
        try out.writeAll(boolText(fixture.expected));
        try out.writeByte(')');
    }
    try out.writeAll("]\n\nend Ora.Generated\n");

    try out.flush();
}
