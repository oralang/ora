const lsp = @import("lsp");
const ora_root = @import("ora_root");

const definition = ora_root.lsp.definition;

const Allocator = @import("std").mem.Allocator;
const types = lsp.types;

pub fn build(
    arena: Allocator,
    default_uri: []const u8,
    resolved: definition.Definition,
    range_converter: anytype,
) !types.Location {
    const uri = resolved.uri orelse default_uri;
    return .{
        .uri = try arena.dupe(u8, uri),
        .range = try range_converter.byteRangeToLsp(uri, resolved.range),
    };
}
