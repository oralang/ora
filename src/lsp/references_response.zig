const std = @import("std");
const lsp = @import("lsp");
const ora_root = @import("ora_root");

const frontend = ora_root.lsp.frontend;

const Allocator = std.mem.Allocator;
const types = lsp.types;

pub fn appendLocations(
    arena: Allocator,
    locations: *std.ArrayList(types.Location),
    uri: []const u8,
    ranges: []const frontend.Range,
    range_converter: anytype,
) !void {
    if (ranges.len == 0) return;

    try locations.ensureUnusedCapacity(arena, ranges.len);
    for (ranges) |range| {
        locations.appendAssumeCapacity(.{
            .uri = uri,
            .range = try range_converter.byteRangeToLsp(uri, range),
        });
    }
}
