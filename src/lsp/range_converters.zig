const lsp = @import("lsp");
const std = @import("std");
const ora_root = @import("ora_root");

const frontend = ora_root.lsp.frontend;
const line_index_api = ora_root.lsp.line_index;
const text_edits = ora_root.lsp.text_edits;

const protocol_ranges = @import("protocol_ranges.zig");

const Allocator = std.mem.Allocator;
const types = lsp.types;

pub fn OpenDocument(comptime Handler: type) type {
    return struct {
        arena: Allocator,
        handler: Handler,
        encoding: text_edits.PositionEncoding,
        cached_uri: ?[]const u8 = null,
        cached_source: []const u8 = &.{},
        cached_line_index: ?*const line_index_api.LineIndex = null,

        pub fn byteRangeToLsp(self: *@This(), uri: []const u8, range: frontend.Range) !types.Range {
            if (self.cached_uri) |cached_uri| {
                if (std.mem.eql(u8, cached_uri, uri)) {
                    return protocol_ranges.byteRangeToLspOrRaw(self.cached_source, self.cached_line_index.?, self.encoding, range);
                }
            }

            const source = self.handler.docs.sourceForUri(uri) orelse return protocol_ranges.rawRange(range);
            const line_index = (try self.handler.docs.lineIndexForUri(uri)) orelse return protocol_ranges.rawRange(range);
            self.cached_uri = uri;
            self.cached_source = source;
            self.cached_line_index = line_index;
            return protocol_ranges.byteRangeToLspOrRaw(source, line_index, self.encoding, range);
        }

        pub fn rangesToLsp(self: *@This(), uri: []const u8, ranges: []const frontend.Range) ![]types.Range {
            const result = try self.arena.alloc(types.Range, ranges.len);
            for (ranges, 0..) |range, index| {
                result[index] = try self.byteRangeToLsp(uri, range);
            }
            return result;
        }
    };
}
