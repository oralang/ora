const std = @import("std");
const lsp = @import("lsp");
const ora_root = @import("ora_root");

const frontend = ora_root.lsp.frontend;
const line_index_api = ora_root.lsp.line_index;
const protocol_ranges = @import("protocol_ranges.zig");
const references = ora_root.lsp.references;
const text_edits = ora_root.lsp.text_edits;

const Allocator = std.mem.Allocator;
const types = lsp.types;

pub fn build(
    arena: Allocator,
    source: []const u8,
    line_index: *const line_index_api.LineIndex,
    encoding: text_edits.PositionEncoding,
    index: *const references.OccurrenceIndex,
    position: frontend.Position,
) !?[]types.DocumentHighlight {
    const target = index.occurrenceAt(position) orelse return null;
    const refs = try references.referencesAtOccurrenceIndex(
        arena,
        index,
        target.name,
        target.definition_range,
        true,
    );
    if (refs.len == 0) return null;

    const result = try arena.alloc(types.DocumentHighlight, refs.len);
    for (refs, 0..) |ref, i| {
        result[i] = .{
            .range = protocol_ranges.byteRangeToLspOrRaw(source, line_index, encoding, ref),
            .kind = .Text,
        };
    }
    return result;
}
