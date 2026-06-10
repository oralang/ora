const std = @import("std");
const lsp = @import("lsp");
const ora_root = @import("ora_root");

const frontend = ora_root.lsp.frontend;
const references_response = @import("references_response.zig");

const testing = std.testing;
const types = lsp.types;

const RecordingConverter = struct {
    expected_uri: []const u8,
    calls: *usize,

    pub fn byteRangeToLsp(self: *RecordingConverter, uri: []const u8, range: frontend.Range) !types.Range {
        try testing.expectEqualStrings(self.expected_uri, uri);
        self.calls.* += 1;
        return .{
            .start = .{ .line = range.start.line, .character = range.start.character },
            .end = .{ .line = range.end.line, .character = range.end.character },
        };
    }
};

test "lsp references response: appends locations for uri ranges" {
    var locations = std.ArrayList(types.Location){};
    defer locations.deinit(testing.allocator);

    var calls: usize = 0;
    var converter = RecordingConverter{
        .expected_uri = "file:///main.ora",
        .calls = &calls,
    };
    const ranges = [_]frontend.Range{
        .{
            .start = .{ .line = 1, .character = 2 },
            .end = .{ .line = 1, .character = 8 },
        },
        .{
            .start = .{ .line = 3, .character = 4 },
            .end = .{ .line = 3, .character = 10 },
        },
    };

    try references_response.appendLocations(testing.allocator, &locations, "file:///main.ora", &ranges, &converter);

    try testing.expectEqual(@as(usize, 2), calls);
    try testing.expectEqual(@as(usize, 2), locations.items.len);
    try testing.expectEqualStrings("file:///main.ora", locations.items[0].uri);
    try testing.expectEqual(@as(u32, 1), locations.items[0].range.start.line);
    try testing.expectEqual(@as(u32, 8), locations.items[0].range.end.character);
    try testing.expectEqualStrings("file:///main.ora", locations.items[1].uri);
    try testing.expectEqual(@as(u32, 3), locations.items[1].range.start.line);
    try testing.expectEqual(@as(u32, 10), locations.items[1].range.end.character);
}

test "lsp references response: empty ranges do not call converter" {
    var locations = std.ArrayList(types.Location){};
    defer locations.deinit(testing.allocator);

    var calls: usize = 0;
    var converter = RecordingConverter{
        .expected_uri = "file:///main.ora",
        .calls = &calls,
    };

    try references_response.appendLocations(testing.allocator, &locations, "file:///main.ora", &.{}, &converter);

    try testing.expectEqual(@as(usize, 0), calls);
    try testing.expectEqual(@as(usize, 0), locations.items.len);
}
