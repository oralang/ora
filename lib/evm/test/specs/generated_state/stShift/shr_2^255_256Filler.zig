const std = @import("std");
const testing = std.testing;
const root = @import("../../root.zig");
const runner = root.runner;

test "shr_2_255_256" {
    const allocator = testing.allocator;

    // Read and parse the JSON test file
    const json_path = "execution-specs/tests/eest/static/state_tests/stShift/shr_2^255_256Filler.json";
    const json_content = try std.fs.cwd().readFileAlloc(allocator, json_path, 100 * 1024 * 1024);
    defer allocator.free(json_content);

    const parsed = try std.json.parseFromSlice(std.json.Value, allocator, json_content, .{});
    defer parsed.deinit();

    // Get the specific test case
    const test_name = "shr_2^255_256";
    const test_case = parsed.value.object.get(test_name) orelse return error.TestNotFound;

    // Run the test with path and name for trace generation
    try runner.runJsonTestWithPathAndName(allocator, test_case, json_path, test_name);
}
