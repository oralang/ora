const std = @import("std");
const lsp = @import("lsp");
const ora_root = @import("ora_root");
const protocol_helpers = @import("protocol_helpers.zig");

const diagnostics = ora_root.lsp.diagnostics;
const line_index_api = ora_root.lsp.line_index;
const text_edits = ora_root.lsp.text_edits;

const Allocator = std.mem.Allocator;
const types = lsp.types;

pub fn buildPublishParams(
    arena: Allocator,
    uri: types.DocumentUri,
    source: []const u8,
    line_index: ?*const line_index_api.LineIndex,
    encoding: text_edits.PositionEncoding,
    cache_entry: ?*const diagnostics.CacheEntry,
) !types.PublishDiagnosticsParams {
    const cached = cache_entry orelse return .{
        .uri = uri,
        .diagnostics = try arena.alloc(types.Diagnostic, 0),
    };

    const result = try arena.alloc(types.Diagnostic, cached.diagnostics.len);
    for (cached.diagnostics, 0..) |diag, i| {
        result[i] = .{
            .range = protocol_helpers.diagnosticRangeToLsp(source, line_index, encoding, diag.range),
            .severity = protocol_helpers.frontendSeverityToLsp(diag.severity),
            .source = diagnostics.sourceName(diag.source),
            .message = try arena.dupe(u8, diag.message),
        };
    }

    return .{ .uri = uri, .diagnostics = result };
}
