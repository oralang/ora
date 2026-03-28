const std = @import("std");
const testing = std.testing;
const root = @import("../../../root.zig");
const runner = root.runner;

test "BlockchainTests_ValidBlocks_bcStateTests_testOpcode_60_json__testOpcode_60_Cancun" {
    const allocator = testing.allocator;

    // Read and parse the JSON test file
    const json_path = "execution-spec-tests/fixtures/blockchain_tests/ValidBlocks/bcStateTests/testOpcode_60.json";
    const json_content = try std.fs.cwd().readFileAlloc(allocator, json_path, 100 * 1024 * 1024);
    defer allocator.free(json_content);

    const parsed = try std.json.parseFromSlice(std.json.Value, allocator, json_content, .{});
    defer parsed.deinit();

    // Get the specific test case
    const test_name = "BlockchainTests/ValidBlocks/bcStateTests/testOpcode_60.json::testOpcode_60_Cancun";
    const test_case = parsed.value.object.get(test_name) orelse return error.TestNotFound;

    // Run the test with path and name for trace generation
    try runner.runJsonTestWithPathAndName(allocator, test_case, json_path, test_name);
}

test "BlockchainTests_ValidBlocks_bcStateTests_testOpcode_60_json__testOpcode_60_Prague" {
    const allocator = testing.allocator;

    // Read and parse the JSON test file
    const json_path = "execution-spec-tests/fixtures/blockchain_tests/ValidBlocks/bcStateTests/testOpcode_60.json";
    const json_content = try std.fs.cwd().readFileAlloc(allocator, json_path, 100 * 1024 * 1024);
    defer allocator.free(json_content);

    const parsed = try std.json.parseFromSlice(std.json.Value, allocator, json_content, .{});
    defer parsed.deinit();

    // Get the specific test case
    const test_name = "BlockchainTests/ValidBlocks/bcStateTests/testOpcode_60.json::testOpcode_60_Prague";
    const test_case = parsed.value.object.get(test_name) orelse return error.TestNotFound;

    // Run the test with path and name for trace generation
    try runner.runJsonTestWithPathAndName(allocator, test_case, json_path, test_name);
}

test "BlockchainTests_ValidBlocks_bcStateTests_testOpcode_60_json__testOpcode_61_Cancun" {
    const allocator = testing.allocator;

    // Read and parse the JSON test file
    const json_path = "execution-spec-tests/fixtures/blockchain_tests/ValidBlocks/bcStateTests/testOpcode_60.json";
    const json_content = try std.fs.cwd().readFileAlloc(allocator, json_path, 100 * 1024 * 1024);
    defer allocator.free(json_content);

    const parsed = try std.json.parseFromSlice(std.json.Value, allocator, json_content, .{});
    defer parsed.deinit();

    // Get the specific test case
    const test_name = "BlockchainTests/ValidBlocks/bcStateTests/testOpcode_60.json::testOpcode_61_Cancun";
    const test_case = parsed.value.object.get(test_name) orelse return error.TestNotFound;

    // Run the test with path and name for trace generation
    try runner.runJsonTestWithPathAndName(allocator, test_case, json_path, test_name);
}

test "BlockchainTests_ValidBlocks_bcStateTests_testOpcode_60_json__testOpcode_61_Prague" {
    const allocator = testing.allocator;

    // Read and parse the JSON test file
    const json_path = "execution-spec-tests/fixtures/blockchain_tests/ValidBlocks/bcStateTests/testOpcode_60.json";
    const json_content = try std.fs.cwd().readFileAlloc(allocator, json_path, 100 * 1024 * 1024);
    defer allocator.free(json_content);

    const parsed = try std.json.parseFromSlice(std.json.Value, allocator, json_content, .{});
    defer parsed.deinit();

    // Get the specific test case
    const test_name = "BlockchainTests/ValidBlocks/bcStateTests/testOpcode_60.json::testOpcode_61_Prague";
    const test_case = parsed.value.object.get(test_name) orelse return error.TestNotFound;

    // Run the test with path and name for trace generation
    try runner.runJsonTestWithPathAndName(allocator, test_case, json_path, test_name);
}

test "BlockchainTests_ValidBlocks_bcStateTests_testOpcode_60_json__testOpcode_62_Cancun" {
    const allocator = testing.allocator;

    // Read and parse the JSON test file
    const json_path = "execution-spec-tests/fixtures/blockchain_tests/ValidBlocks/bcStateTests/testOpcode_60.json";
    const json_content = try std.fs.cwd().readFileAlloc(allocator, json_path, 100 * 1024 * 1024);
    defer allocator.free(json_content);

    const parsed = try std.json.parseFromSlice(std.json.Value, allocator, json_content, .{});
    defer parsed.deinit();

    // Get the specific test case
    const test_name = "BlockchainTests/ValidBlocks/bcStateTests/testOpcode_60.json::testOpcode_62_Cancun";
    const test_case = parsed.value.object.get(test_name) orelse return error.TestNotFound;

    // Run the test with path and name for trace generation
    try runner.runJsonTestWithPathAndName(allocator, test_case, json_path, test_name);
}

test "BlockchainTests_ValidBlocks_bcStateTests_testOpcode_60_json__testOpcode_62_Prague" {
    const allocator = testing.allocator;

    // Read and parse the JSON test file
    const json_path = "execution-spec-tests/fixtures/blockchain_tests/ValidBlocks/bcStateTests/testOpcode_60.json";
    const json_content = try std.fs.cwd().readFileAlloc(allocator, json_path, 100 * 1024 * 1024);
    defer allocator.free(json_content);

    const parsed = try std.json.parseFromSlice(std.json.Value, allocator, json_content, .{});
    defer parsed.deinit();

    // Get the specific test case
    const test_name = "BlockchainTests/ValidBlocks/bcStateTests/testOpcode_60.json::testOpcode_62_Prague";
    const test_case = parsed.value.object.get(test_name) orelse return error.TestNotFound;

    // Run the test with path and name for trace generation
    try runner.runJsonTestWithPathAndName(allocator, test_case, json_path, test_name);
}

test "BlockchainTests_ValidBlocks_bcStateTests_testOpcode_60_json__testOpcode_63_Cancun" {
    const allocator = testing.allocator;

    // Read and parse the JSON test file
    const json_path = "execution-spec-tests/fixtures/blockchain_tests/ValidBlocks/bcStateTests/testOpcode_60.json";
    const json_content = try std.fs.cwd().readFileAlloc(allocator, json_path, 100 * 1024 * 1024);
    defer allocator.free(json_content);

    const parsed = try std.json.parseFromSlice(std.json.Value, allocator, json_content, .{});
    defer parsed.deinit();

    // Get the specific test case
    const test_name = "BlockchainTests/ValidBlocks/bcStateTests/testOpcode_60.json::testOpcode_63_Cancun";
    const test_case = parsed.value.object.get(test_name) orelse return error.TestNotFound;

    // Run the test with path and name for trace generation
    try runner.runJsonTestWithPathAndName(allocator, test_case, json_path, test_name);
}

test "BlockchainTests_ValidBlocks_bcStateTests_testOpcode_60_json__testOpcode_63_Prague" {
    const allocator = testing.allocator;

    // Read and parse the JSON test file
    const json_path = "execution-spec-tests/fixtures/blockchain_tests/ValidBlocks/bcStateTests/testOpcode_60.json";
    const json_content = try std.fs.cwd().readFileAlloc(allocator, json_path, 100 * 1024 * 1024);
    defer allocator.free(json_content);

    const parsed = try std.json.parseFromSlice(std.json.Value, allocator, json_content, .{});
    defer parsed.deinit();

    // Get the specific test case
    const test_name = "BlockchainTests/ValidBlocks/bcStateTests/testOpcode_60.json::testOpcode_63_Prague";
    const test_case = parsed.value.object.get(test_name) orelse return error.TestNotFound;

    // Run the test with path and name for trace generation
    try runner.runJsonTestWithPathAndName(allocator, test_case, json_path, test_name);
}

test "BlockchainTests_ValidBlocks_bcStateTests_testOpcode_60_json__testOpcode_64_Cancun" {
    const allocator = testing.allocator;

    // Read and parse the JSON test file
    const json_path = "execution-spec-tests/fixtures/blockchain_tests/ValidBlocks/bcStateTests/testOpcode_60.json";
    const json_content = try std.fs.cwd().readFileAlloc(allocator, json_path, 100 * 1024 * 1024);
    defer allocator.free(json_content);

    const parsed = try std.json.parseFromSlice(std.json.Value, allocator, json_content, .{});
    defer parsed.deinit();

    // Get the specific test case
    const test_name = "BlockchainTests/ValidBlocks/bcStateTests/testOpcode_60.json::testOpcode_64_Cancun";
    const test_case = parsed.value.object.get(test_name) orelse return error.TestNotFound;

    // Run the test with path and name for trace generation
    try runner.runJsonTestWithPathAndName(allocator, test_case, json_path, test_name);
}

test "BlockchainTests_ValidBlocks_bcStateTests_testOpcode_60_json__testOpcode_64_Prague" {
    const allocator = testing.allocator;

    // Read and parse the JSON test file
    const json_path = "execution-spec-tests/fixtures/blockchain_tests/ValidBlocks/bcStateTests/testOpcode_60.json";
    const json_content = try std.fs.cwd().readFileAlloc(allocator, json_path, 100 * 1024 * 1024);
    defer allocator.free(json_content);

    const parsed = try std.json.parseFromSlice(std.json.Value, allocator, json_content, .{});
    defer parsed.deinit();

    // Get the specific test case
    const test_name = "BlockchainTests/ValidBlocks/bcStateTests/testOpcode_60.json::testOpcode_64_Prague";
    const test_case = parsed.value.object.get(test_name) orelse return error.TestNotFound;

    // Run the test with path and name for trace generation
    try runner.runJsonTestWithPathAndName(allocator, test_case, json_path, test_name);
}

test "BlockchainTests_ValidBlocks_bcStateTests_testOpcode_60_json__testOpcode_65_Cancun" {
    const allocator = testing.allocator;

    // Read and parse the JSON test file
    const json_path = "execution-spec-tests/fixtures/blockchain_tests/ValidBlocks/bcStateTests/testOpcode_60.json";
    const json_content = try std.fs.cwd().readFileAlloc(allocator, json_path, 100 * 1024 * 1024);
    defer allocator.free(json_content);

    const parsed = try std.json.parseFromSlice(std.json.Value, allocator, json_content, .{});
    defer parsed.deinit();

    // Get the specific test case
    const test_name = "BlockchainTests/ValidBlocks/bcStateTests/testOpcode_60.json::testOpcode_65_Cancun";
    const test_case = parsed.value.object.get(test_name) orelse return error.TestNotFound;

    // Run the test with path and name for trace generation
    try runner.runJsonTestWithPathAndName(allocator, test_case, json_path, test_name);
}

test "BlockchainTests_ValidBlocks_bcStateTests_testOpcode_60_json__testOpcode_65_Prague" {
    const allocator = testing.allocator;

    // Read and parse the JSON test file
    const json_path = "execution-spec-tests/fixtures/blockchain_tests/ValidBlocks/bcStateTests/testOpcode_60.json";
    const json_content = try std.fs.cwd().readFileAlloc(allocator, json_path, 100 * 1024 * 1024);
    defer allocator.free(json_content);

    const parsed = try std.json.parseFromSlice(std.json.Value, allocator, json_content, .{});
    defer parsed.deinit();

    // Get the specific test case
    const test_name = "BlockchainTests/ValidBlocks/bcStateTests/testOpcode_60.json::testOpcode_65_Prague";
    const test_case = parsed.value.object.get(test_name) orelse return error.TestNotFound;

    // Run the test with path and name for trace generation
    try runner.runJsonTestWithPathAndName(allocator, test_case, json_path, test_name);
}

test "BlockchainTests_ValidBlocks_bcStateTests_testOpcode_60_json__testOpcode_66_Cancun" {
    const allocator = testing.allocator;

    // Read and parse the JSON test file
    const json_path = "execution-spec-tests/fixtures/blockchain_tests/ValidBlocks/bcStateTests/testOpcode_60.json";
    const json_content = try std.fs.cwd().readFileAlloc(allocator, json_path, 100 * 1024 * 1024);
    defer allocator.free(json_content);

    const parsed = try std.json.parseFromSlice(std.json.Value, allocator, json_content, .{});
    defer parsed.deinit();

    // Get the specific test case
    const test_name = "BlockchainTests/ValidBlocks/bcStateTests/testOpcode_60.json::testOpcode_66_Cancun";
    const test_case = parsed.value.object.get(test_name) orelse return error.TestNotFound;

    // Run the test with path and name for trace generation
    try runner.runJsonTestWithPathAndName(allocator, test_case, json_path, test_name);
}

test "BlockchainTests_ValidBlocks_bcStateTests_testOpcode_60_json__testOpcode_66_Prague" {
    const allocator = testing.allocator;

    // Read and parse the JSON test file
    const json_path = "execution-spec-tests/fixtures/blockchain_tests/ValidBlocks/bcStateTests/testOpcode_60.json";
    const json_content = try std.fs.cwd().readFileAlloc(allocator, json_path, 100 * 1024 * 1024);
    defer allocator.free(json_content);

    const parsed = try std.json.parseFromSlice(std.json.Value, allocator, json_content, .{});
    defer parsed.deinit();

    // Get the specific test case
    const test_name = "BlockchainTests/ValidBlocks/bcStateTests/testOpcode_60.json::testOpcode_66_Prague";
    const test_case = parsed.value.object.get(test_name) orelse return error.TestNotFound;

    // Run the test with path and name for trace generation
    try runner.runJsonTestWithPathAndName(allocator, test_case, json_path, test_name);
}

test "BlockchainTests_ValidBlocks_bcStateTests_testOpcode_60_json__testOpcode_67_Cancun" {
    const allocator = testing.allocator;

    // Read and parse the JSON test file
    const json_path = "execution-spec-tests/fixtures/blockchain_tests/ValidBlocks/bcStateTests/testOpcode_60.json";
    const json_content = try std.fs.cwd().readFileAlloc(allocator, json_path, 100 * 1024 * 1024);
    defer allocator.free(json_content);

    const parsed = try std.json.parseFromSlice(std.json.Value, allocator, json_content, .{});
    defer parsed.deinit();

    // Get the specific test case
    const test_name = "BlockchainTests/ValidBlocks/bcStateTests/testOpcode_60.json::testOpcode_67_Cancun";
    const test_case = parsed.value.object.get(test_name) orelse return error.TestNotFound;

    // Run the test with path and name for trace generation
    try runner.runJsonTestWithPathAndName(allocator, test_case, json_path, test_name);
}

test "BlockchainTests_ValidBlocks_bcStateTests_testOpcode_60_json__testOpcode_67_Prague" {
    const allocator = testing.allocator;

    // Read and parse the JSON test file
    const json_path = "execution-spec-tests/fixtures/blockchain_tests/ValidBlocks/bcStateTests/testOpcode_60.json";
    const json_content = try std.fs.cwd().readFileAlloc(allocator, json_path, 100 * 1024 * 1024);
    defer allocator.free(json_content);

    const parsed = try std.json.parseFromSlice(std.json.Value, allocator, json_content, .{});
    defer parsed.deinit();

    // Get the specific test case
    const test_name = "BlockchainTests/ValidBlocks/bcStateTests/testOpcode_60.json::testOpcode_67_Prague";
    const test_case = parsed.value.object.get(test_name) orelse return error.TestNotFound;

    // Run the test with path and name for trace generation
    try runner.runJsonTestWithPathAndName(allocator, test_case, json_path, test_name);
}

test "BlockchainTests_ValidBlocks_bcStateTests_testOpcode_60_json__testOpcode_68_Cancun" {
    const allocator = testing.allocator;

    // Read and parse the JSON test file
    const json_path = "execution-spec-tests/fixtures/blockchain_tests/ValidBlocks/bcStateTests/testOpcode_60.json";
    const json_content = try std.fs.cwd().readFileAlloc(allocator, json_path, 100 * 1024 * 1024);
    defer allocator.free(json_content);

    const parsed = try std.json.parseFromSlice(std.json.Value, allocator, json_content, .{});
    defer parsed.deinit();

    // Get the specific test case
    const test_name = "BlockchainTests/ValidBlocks/bcStateTests/testOpcode_60.json::testOpcode_68_Cancun";
    const test_case = parsed.value.object.get(test_name) orelse return error.TestNotFound;

    // Run the test with path and name for trace generation
    try runner.runJsonTestWithPathAndName(allocator, test_case, json_path, test_name);
}

test "BlockchainTests_ValidBlocks_bcStateTests_testOpcode_60_json__testOpcode_68_Prague" {
    const allocator = testing.allocator;

    // Read and parse the JSON test file
    const json_path = "execution-spec-tests/fixtures/blockchain_tests/ValidBlocks/bcStateTests/testOpcode_60.json";
    const json_content = try std.fs.cwd().readFileAlloc(allocator, json_path, 100 * 1024 * 1024);
    defer allocator.free(json_content);

    const parsed = try std.json.parseFromSlice(std.json.Value, allocator, json_content, .{});
    defer parsed.deinit();

    // Get the specific test case
    const test_name = "BlockchainTests/ValidBlocks/bcStateTests/testOpcode_60.json::testOpcode_68_Prague";
    const test_case = parsed.value.object.get(test_name) orelse return error.TestNotFound;

    // Run the test with path and name for trace generation
    try runner.runJsonTestWithPathAndName(allocator, test_case, json_path, test_name);
}

test "BlockchainTests_ValidBlocks_bcStateTests_testOpcode_60_json__testOpcode_69_Cancun" {
    const allocator = testing.allocator;

    // Read and parse the JSON test file
    const json_path = "execution-spec-tests/fixtures/blockchain_tests/ValidBlocks/bcStateTests/testOpcode_60.json";
    const json_content = try std.fs.cwd().readFileAlloc(allocator, json_path, 100 * 1024 * 1024);
    defer allocator.free(json_content);

    const parsed = try std.json.parseFromSlice(std.json.Value, allocator, json_content, .{});
    defer parsed.deinit();

    // Get the specific test case
    const test_name = "BlockchainTests/ValidBlocks/bcStateTests/testOpcode_60.json::testOpcode_69_Cancun";
    const test_case = parsed.value.object.get(test_name) orelse return error.TestNotFound;

    // Run the test with path and name for trace generation
    try runner.runJsonTestWithPathAndName(allocator, test_case, json_path, test_name);
}

test "BlockchainTests_ValidBlocks_bcStateTests_testOpcode_60_json__testOpcode_69_Prague" {
    const allocator = testing.allocator;

    // Read and parse the JSON test file
    const json_path = "execution-spec-tests/fixtures/blockchain_tests/ValidBlocks/bcStateTests/testOpcode_60.json";
    const json_content = try std.fs.cwd().readFileAlloc(allocator, json_path, 100 * 1024 * 1024);
    defer allocator.free(json_content);

    const parsed = try std.json.parseFromSlice(std.json.Value, allocator, json_content, .{});
    defer parsed.deinit();

    // Get the specific test case
    const test_name = "BlockchainTests/ValidBlocks/bcStateTests/testOpcode_60.json::testOpcode_69_Prague";
    const test_case = parsed.value.object.get(test_name) orelse return error.TestNotFound;

    // Run the test with path and name for trace generation
    try runner.runJsonTestWithPathAndName(allocator, test_case, json_path, test_name);
}

test "BlockchainTests_ValidBlocks_bcStateTests_testOpcode_60_json__testOpcode_6a_Cancun" {
    const allocator = testing.allocator;

    // Read and parse the JSON test file
    const json_path = "execution-spec-tests/fixtures/blockchain_tests/ValidBlocks/bcStateTests/testOpcode_60.json";
    const json_content = try std.fs.cwd().readFileAlloc(allocator, json_path, 100 * 1024 * 1024);
    defer allocator.free(json_content);

    const parsed = try std.json.parseFromSlice(std.json.Value, allocator, json_content, .{});
    defer parsed.deinit();

    // Get the specific test case
    const test_name = "BlockchainTests/ValidBlocks/bcStateTests/testOpcode_60.json::testOpcode_6a_Cancun";
    const test_case = parsed.value.object.get(test_name) orelse return error.TestNotFound;

    // Run the test with path and name for trace generation
    try runner.runJsonTestWithPathAndName(allocator, test_case, json_path, test_name);
}

test "BlockchainTests_ValidBlocks_bcStateTests_testOpcode_60_json__testOpcode_6a_Prague" {
    const allocator = testing.allocator;

    // Read and parse the JSON test file
    const json_path = "execution-spec-tests/fixtures/blockchain_tests/ValidBlocks/bcStateTests/testOpcode_60.json";
    const json_content = try std.fs.cwd().readFileAlloc(allocator, json_path, 100 * 1024 * 1024);
    defer allocator.free(json_content);

    const parsed = try std.json.parseFromSlice(std.json.Value, allocator, json_content, .{});
    defer parsed.deinit();

    // Get the specific test case
    const test_name = "BlockchainTests/ValidBlocks/bcStateTests/testOpcode_60.json::testOpcode_6a_Prague";
    const test_case = parsed.value.object.get(test_name) orelse return error.TestNotFound;

    // Run the test with path and name for trace generation
    try runner.runJsonTestWithPathAndName(allocator, test_case, json_path, test_name);
}

test "BlockchainTests_ValidBlocks_bcStateTests_testOpcode_60_json__testOpcode_6b_Cancun" {
    const allocator = testing.allocator;

    // Read and parse the JSON test file
    const json_path = "execution-spec-tests/fixtures/blockchain_tests/ValidBlocks/bcStateTests/testOpcode_60.json";
    const json_content = try std.fs.cwd().readFileAlloc(allocator, json_path, 100 * 1024 * 1024);
    defer allocator.free(json_content);

    const parsed = try std.json.parseFromSlice(std.json.Value, allocator, json_content, .{});
    defer parsed.deinit();

    // Get the specific test case
    const test_name = "BlockchainTests/ValidBlocks/bcStateTests/testOpcode_60.json::testOpcode_6b_Cancun";
    const test_case = parsed.value.object.get(test_name) orelse return error.TestNotFound;

    // Run the test with path and name for trace generation
    try runner.runJsonTestWithPathAndName(allocator, test_case, json_path, test_name);
}

test "BlockchainTests_ValidBlocks_bcStateTests_testOpcode_60_json__testOpcode_6b_Prague" {
    const allocator = testing.allocator;

    // Read and parse the JSON test file
    const json_path = "execution-spec-tests/fixtures/blockchain_tests/ValidBlocks/bcStateTests/testOpcode_60.json";
    const json_content = try std.fs.cwd().readFileAlloc(allocator, json_path, 100 * 1024 * 1024);
    defer allocator.free(json_content);

    const parsed = try std.json.parseFromSlice(std.json.Value, allocator, json_content, .{});
    defer parsed.deinit();

    // Get the specific test case
    const test_name = "BlockchainTests/ValidBlocks/bcStateTests/testOpcode_60.json::testOpcode_6b_Prague";
    const test_case = parsed.value.object.get(test_name) orelse return error.TestNotFound;

    // Run the test with path and name for trace generation
    try runner.runJsonTestWithPathAndName(allocator, test_case, json_path, test_name);
}

test "BlockchainTests_ValidBlocks_bcStateTests_testOpcode_60_json__testOpcode_6c_Cancun" {
    const allocator = testing.allocator;

    // Read and parse the JSON test file
    const json_path = "execution-spec-tests/fixtures/blockchain_tests/ValidBlocks/bcStateTests/testOpcode_60.json";
    const json_content = try std.fs.cwd().readFileAlloc(allocator, json_path, 100 * 1024 * 1024);
    defer allocator.free(json_content);

    const parsed = try std.json.parseFromSlice(std.json.Value, allocator, json_content, .{});
    defer parsed.deinit();

    // Get the specific test case
    const test_name = "BlockchainTests/ValidBlocks/bcStateTests/testOpcode_60.json::testOpcode_6c_Cancun";
    const test_case = parsed.value.object.get(test_name) orelse return error.TestNotFound;

    // Run the test with path and name for trace generation
    try runner.runJsonTestWithPathAndName(allocator, test_case, json_path, test_name);
}

test "BlockchainTests_ValidBlocks_bcStateTests_testOpcode_60_json__testOpcode_6c_Prague" {
    const allocator = testing.allocator;

    // Read and parse the JSON test file
    const json_path = "execution-spec-tests/fixtures/blockchain_tests/ValidBlocks/bcStateTests/testOpcode_60.json";
    const json_content = try std.fs.cwd().readFileAlloc(allocator, json_path, 100 * 1024 * 1024);
    defer allocator.free(json_content);

    const parsed = try std.json.parseFromSlice(std.json.Value, allocator, json_content, .{});
    defer parsed.deinit();

    // Get the specific test case
    const test_name = "BlockchainTests/ValidBlocks/bcStateTests/testOpcode_60.json::testOpcode_6c_Prague";
    const test_case = parsed.value.object.get(test_name) orelse return error.TestNotFound;

    // Run the test with path and name for trace generation
    try runner.runJsonTestWithPathAndName(allocator, test_case, json_path, test_name);
}

test "BlockchainTests_ValidBlocks_bcStateTests_testOpcode_60_json__testOpcode_6d_Cancun" {
    const allocator = testing.allocator;

    // Read and parse the JSON test file
    const json_path = "execution-spec-tests/fixtures/blockchain_tests/ValidBlocks/bcStateTests/testOpcode_60.json";
    const json_content = try std.fs.cwd().readFileAlloc(allocator, json_path, 100 * 1024 * 1024);
    defer allocator.free(json_content);

    const parsed = try std.json.parseFromSlice(std.json.Value, allocator, json_content, .{});
    defer parsed.deinit();

    // Get the specific test case
    const test_name = "BlockchainTests/ValidBlocks/bcStateTests/testOpcode_60.json::testOpcode_6d_Cancun";
    const test_case = parsed.value.object.get(test_name) orelse return error.TestNotFound;

    // Run the test with path and name for trace generation
    try runner.runJsonTestWithPathAndName(allocator, test_case, json_path, test_name);
}

test "BlockchainTests_ValidBlocks_bcStateTests_testOpcode_60_json__testOpcode_6d_Prague" {
    const allocator = testing.allocator;

    // Read and parse the JSON test file
    const json_path = "execution-spec-tests/fixtures/blockchain_tests/ValidBlocks/bcStateTests/testOpcode_60.json";
    const json_content = try std.fs.cwd().readFileAlloc(allocator, json_path, 100 * 1024 * 1024);
    defer allocator.free(json_content);

    const parsed = try std.json.parseFromSlice(std.json.Value, allocator, json_content, .{});
    defer parsed.deinit();

    // Get the specific test case
    const test_name = "BlockchainTests/ValidBlocks/bcStateTests/testOpcode_60.json::testOpcode_6d_Prague";
    const test_case = parsed.value.object.get(test_name) orelse return error.TestNotFound;

    // Run the test with path and name for trace generation
    try runner.runJsonTestWithPathAndName(allocator, test_case, json_path, test_name);
}

test "BlockchainTests_ValidBlocks_bcStateTests_testOpcode_60_json__testOpcode_6e_Cancun" {
    const allocator = testing.allocator;

    // Read and parse the JSON test file
    const json_path = "execution-spec-tests/fixtures/blockchain_tests/ValidBlocks/bcStateTests/testOpcode_60.json";
    const json_content = try std.fs.cwd().readFileAlloc(allocator, json_path, 100 * 1024 * 1024);
    defer allocator.free(json_content);

    const parsed = try std.json.parseFromSlice(std.json.Value, allocator, json_content, .{});
    defer parsed.deinit();

    // Get the specific test case
    const test_name = "BlockchainTests/ValidBlocks/bcStateTests/testOpcode_60.json::testOpcode_6e_Cancun";
    const test_case = parsed.value.object.get(test_name) orelse return error.TestNotFound;

    // Run the test with path and name for trace generation
    try runner.runJsonTestWithPathAndName(allocator, test_case, json_path, test_name);
}

test "BlockchainTests_ValidBlocks_bcStateTests_testOpcode_60_json__testOpcode_6e_Prague" {
    const allocator = testing.allocator;

    // Read and parse the JSON test file
    const json_path = "execution-spec-tests/fixtures/blockchain_tests/ValidBlocks/bcStateTests/testOpcode_60.json";
    const json_content = try std.fs.cwd().readFileAlloc(allocator, json_path, 100 * 1024 * 1024);
    defer allocator.free(json_content);

    const parsed = try std.json.parseFromSlice(std.json.Value, allocator, json_content, .{});
    defer parsed.deinit();

    // Get the specific test case
    const test_name = "BlockchainTests/ValidBlocks/bcStateTests/testOpcode_60.json::testOpcode_6e_Prague";
    const test_case = parsed.value.object.get(test_name) orelse return error.TestNotFound;

    // Run the test with path and name for trace generation
    try runner.runJsonTestWithPathAndName(allocator, test_case, json_path, test_name);
}

test "BlockchainTests_ValidBlocks_bcStateTests_testOpcode_60_json__testOpcode_6f_Cancun" {
    const allocator = testing.allocator;

    // Read and parse the JSON test file
    const json_path = "execution-spec-tests/fixtures/blockchain_tests/ValidBlocks/bcStateTests/testOpcode_60.json";
    const json_content = try std.fs.cwd().readFileAlloc(allocator, json_path, 100 * 1024 * 1024);
    defer allocator.free(json_content);

    const parsed = try std.json.parseFromSlice(std.json.Value, allocator, json_content, .{});
    defer parsed.deinit();

    // Get the specific test case
    const test_name = "BlockchainTests/ValidBlocks/bcStateTests/testOpcode_60.json::testOpcode_6f_Cancun";
    const test_case = parsed.value.object.get(test_name) orelse return error.TestNotFound;

    // Run the test with path and name for trace generation
    try runner.runJsonTestWithPathAndName(allocator, test_case, json_path, test_name);
}

test "BlockchainTests_ValidBlocks_bcStateTests_testOpcode_60_json__testOpcode_6f_Prague" {
    const allocator = testing.allocator;

    // Read and parse the JSON test file
    const json_path = "execution-spec-tests/fixtures/blockchain_tests/ValidBlocks/bcStateTests/testOpcode_60.json";
    const json_content = try std.fs.cwd().readFileAlloc(allocator, json_path, 100 * 1024 * 1024);
    defer allocator.free(json_content);

    const parsed = try std.json.parseFromSlice(std.json.Value, allocator, json_content, .{});
    defer parsed.deinit();

    // Get the specific test case
    const test_name = "BlockchainTests/ValidBlocks/bcStateTests/testOpcode_60.json::testOpcode_6f_Prague";
    const test_case = parsed.value.object.get(test_name) orelse return error.TestNotFound;

    // Run the test with path and name for trace generation
    try runner.runJsonTestWithPathAndName(allocator, test_case, json_path, test_name);
}
