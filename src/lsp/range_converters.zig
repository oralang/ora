const lsp = @import("lsp");
const ora_root = @import("ora_root");

const frontend = ora_root.lsp.frontend;
const text_edits = ora_root.lsp.text_edits;

const protocol_ranges = @import("protocol_ranges.zig");

const Allocator = @import("std").mem.Allocator;
const types = lsp.types;

pub fn OpenDocument(comptime Handler: type) type {
    return struct {
        arena: Allocator,
        handler: Handler,
        encoding: text_edits.PositionEncoding,

        pub fn byteRangeToLsp(self: *@This(), uri: []const u8, range: frontend.Range) !types.Range {
            const source = self.handler.docs.sourceForUri(uri) orelse return protocol_ranges.rawRange(range);
            const line_index = (try self.handler.docs.lineIndexForUri(uri)) orelse return protocol_ranges.rawRange(range);
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
