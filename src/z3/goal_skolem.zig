const std = @import("std");

pub const name_buffer_len = 96;
pub const binder_buffer_len = 48;

pub fn sanitizeBinderName(buffer: []u8, raw: []const u8) []const u8 {
    if (buffer.len == 0) return "q";
    var len: usize = 0;
    for (raw) |ch| {
        if (len == buffer.len) break;
        buffer[len] = if (std.ascii.isAlphanumeric(ch) or ch == '_') ch else '_';
        len += 1;
    }
    if (len == 0) return "q";
    return buffer[0..len];
}

pub fn nameZ(
    out: *[name_buffer_len]u8,
    binder_buffer: *[binder_buffer_len]u8,
    raw_binder_name: []const u8,
    depth: u32,
) [:0]const u8 {
    comptime {
        const max_u32_digits = 10;
        const required = "$ora.goal.skolem.".len + binder_buffer_len + ".".len + max_u32_digits + 1;
        if (required > name_buffer_len) @compileError("goal skolem name buffer cannot hold the maximum generated name");
    }
    const binder_name = sanitizeBinderName(binder_buffer[0..], raw_binder_name);
    return std.fmt.bufPrintZ(
        out[0..],
        "$ora.goal.skolem.{s}.{d}",
        .{ binder_name, depth },
    ) catch unreachable;
}

test "goal skolem names are sanitized and depth-stable" {
    var out: [name_buffer_len]u8 = undefined;
    var binder: [binder_buffer_len]u8 = undefined;

    try std.testing.expectEqualStrings(
        "$ora.goal.skolem.i.0",
        nameZ(&out, &binder, "i", 0),
    );
    try std.testing.expectEqualStrings(
        "$ora.goal.skolem.a_b.7",
        nameZ(&out, &binder, "a-b", 7),
    );
}
