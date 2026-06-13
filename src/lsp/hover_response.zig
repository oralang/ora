const lsp = @import("lsp");
const ora_root = @import("ora_root");
const protocol_ranges = @import("protocol_ranges.zig");

const hover = ora_root.lsp.hover;
const line_index = ora_root.lsp.line_index;
const text_edits = ora_root.lsp.text_edits;

const types = lsp.types;

pub fn build(
    source: []const u8,
    lines: *const line_index.LineIndex,
    encoding: text_edits.PositionEncoding,
    item: hover.Hover,
) ?types.Hover {
    const range = protocol_ranges.byteRangeToLsp(source, lines, encoding, item.range) orelse return null;
    return .{
        .contents = .{ .MarkupContent = .{
            .kind = .markdown,
            .value = item.contents,
        } },
        .range = range,
    };
}
