const std = @import("std");
const lsp = @import("lsp");
const ora_root = @import("ora_root");

const frontend = ora_root.lsp.frontend;
const line_index_api = ora_root.lsp.line_index;
const protocol_ranges = @import("protocol_ranges.zig");
const text_edits = ora_root.lsp.text_edits;
const workspace = ora_root.lsp.workspace;

const Allocator = std.mem.Allocator;
const types = lsp.types;

pub fn build(
    arena: Allocator,
    source: []const u8,
    line_index: *const line_index_api.LineIndex,
    encoding: text_edits.PositionEncoding,
    imports: []const workspace.ResolvedImport,
) !?[]types.DocumentLink {
    if (imports.len == 0) return null;

    var links = std.ArrayList(types.DocumentLink){};
    errdefer links.deinit(arena);

    for (imports) |import_item| {
        const path_range = if (import_item.specifier_range) |range|
            protocol_ranges.textRangeToLsp(source, line_index, encoding, range)
        else if (findImportPathRange(source, import_item.specifier)) |range|
            protocol_ranges.byteRangeToLspOrRaw(source, line_index, encoding, range)
        else
            continue;
        try links.append(arena, .{
            .range = path_range,
            .target = try workspace.pathToFileUri(arena, import_item.resolved_path),
            .tooltip = try std.fmt.allocPrint(arena, "Open {s}", .{import_item.specifier}),
        });
    }

    if (links.items.len == 0) return null;
    return try links.toOwnedSlice(arena);
}

fn findImportPathRange(source: []const u8, specifier: []const u8) ?frontend.Range {
    var line: u32 = 0;
    var col: u32 = 0;
    var i: usize = 0;
    while (i < source.len) : (i += 1) {
        if (source[i] == '\n') {
            line += 1;
            col = 0;
            continue;
        }
        if (i + specifier.len <= source.len and std.mem.eql(u8, source[i .. i + specifier.len], specifier)) {
            if (i > 0 and source[i - 1] == '"') {
                return .{
                    .start = .{ .line = line, .character = col },
                    .end = .{ .line = line, .character = col + @as(u32, @intCast(specifier.len)) },
                };
            }
        }
        col += 1;
    }
    return null;
}
