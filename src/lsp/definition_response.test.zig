const std = @import("std");
const lsp = @import("lsp");
const ora_root = @import("ora_root");

const definition = ora_root.lsp.definition;
const definition_response = @import("definition_response.zig");
const frontend = ora_root.lsp.frontend;

const testing = std.testing;
const types = lsp.types;

const RecordingConverter = struct {
    expected_uri: []const u8,
    expected_range: frontend.Range,
    result_range: types.Range,
    calls: *usize,

    pub fn byteRangeToLsp(self: *RecordingConverter, uri: []const u8, range: frontend.Range) !types.Range {
        try testing.expectEqualStrings(self.expected_uri, uri);
        try testing.expectEqual(self.expected_range, range);
        self.calls.* += 1;
        return self.result_range;
    }
};

test "lsp definition response: uses default uri when definition is same-file" {
    var calls: usize = 0;
    const byte_range: frontend.Range = .{
        .start = .{ .line = 2, .character = 4 },
        .end = .{ .line = 2, .character = 10 },
    };
    const lsp_range: types.Range = .{
        .start = .{ .line = 2, .character = 4 },
        .end = .{ .line = 2, .character = 10 },
    };
    var converter = RecordingConverter{
        .expected_uri = "file:///main.ora",
        .expected_range = byte_range,
        .result_range = lsp_range,
        .calls = &calls,
    };

    const location = try definition_response.build(testing.allocator, "file:///main.ora", .{ .range = byte_range }, &converter);
    defer testing.allocator.free(location.uri);

    try testing.expectEqual(@as(usize, 1), calls);
    try testing.expectEqualStrings("file:///main.ora", location.uri);
    try testing.expectEqual(lsp_range, location.range);
}

test "lsp definition response: uses resolved uri for cross-file definitions" {
    var calls: usize = 0;
    const byte_range: frontend.Range = .{
        .start = .{ .line = 0, .character = 1 },
        .end = .{ .line = 0, .character = 7 },
    };
    const lsp_range: types.Range = .{
        .start = .{ .line = 0, .character = 1 },
        .end = .{ .line = 0, .character = 7 },
    };
    var converter = RecordingConverter{
        .expected_uri = "file:///lib.ora",
        .expected_range = byte_range,
        .result_range = lsp_range,
        .calls = &calls,
    };

    const location = try definition_response.build(
        testing.allocator,
        "file:///main.ora",
        .{ .uri = "file:///lib.ora", .range = byte_range },
        &converter,
    );
    defer testing.allocator.free(location.uri);

    try testing.expectEqual(@as(usize, 1), calls);
    try testing.expectEqualStrings("file:///lib.ora", location.uri);
    try testing.expectEqual(lsp_range, location.range);
}
