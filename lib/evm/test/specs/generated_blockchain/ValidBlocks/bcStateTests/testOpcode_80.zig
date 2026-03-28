const std = @import("std");
const testing = std.testing;
const root = @import("../../../root.zig");
const runner = root.runner;

test "BlockchainTests_ValidBlocks_bcStateTests_testOpcode_80_json__testOpcode_80_Cancun" {
    const allocator = testing.allocator;

    // Read and parse the JSON test file
    const json_path = "execution-spec-tests/fixtures/blockchain_tests/ValidBlocks/bcStateTests/testOpcode_80.json";
    const json_content = try std.fs.cwd().readFileAlloc(allocator, json_path, 100 * 1024 * 1024);
    defer allocator.free(json_content);

    const parsed = try std.json.parseFromSlice(std.json.Value, allocator, json_content, .{});
    defer parsed.deinit();

    // Get the specific test case
    const test_name = "BlockchainTests/ValidBlocks/bcStateTests/testOpcode_80.json::testOpcode_80_Cancun";
    const test_case = parsed.value.object.get(test_name) orelse return error.TestNotFound;

    // Run the test with path and name for trace generation
    try runner.runJsonTestWithPathAndName(allocator, test_case, json_path, test_name);
}

test "BlockchainTests_ValidBlocks_bcStateTests_testOpcode_80_json__testOpcode_80_Prague" {
    const allocator = testing.allocator;

    // Read and parse the JSON test file
    const json_path = "execution-spec-tests/fixtures/blockchain_tests/ValidBlocks/bcStateTests/testOpcode_80.json";
    const json_content = try std.fs.cwd().readFileAlloc(allocator, json_path, 100 * 1024 * 1024);
    defer allocator.free(json_content);

    const parsed = try std.json.parseFromSlice(std.json.Value, allocator, json_content, .{});
    defer parsed.deinit();

    // Get the specific test case
    const test_name = "BlockchainTests/ValidBlocks/bcStateTests/testOpcode_80.json::testOpcode_80_Prague";
    const test_case = parsed.value.object.get(test_name) orelse return error.TestNotFound;

    // Run the test with path and name for trace generation
    try runner.runJsonTestWithPathAndName(allocator, test_case, json_path, test_name);
}

test "BlockchainTests_ValidBlocks_bcStateTests_testOpcode_80_json__testOpcode_81_Cancun" {
    const allocator = testing.allocator;

    // Read and parse the JSON test file
    const json_path = "execution-spec-tests/fixtures/blockchain_tests/ValidBlocks/bcStateTests/testOpcode_80.json";
    const json_content = try std.fs.cwd().readFileAlloc(allocator, json_path, 100 * 1024 * 1024);
    defer allocator.free(json_content);

    const parsed = try std.json.parseFromSlice(std.json.Value, allocator, json_content, .{});
    defer parsed.deinit();

    // Get the specific test case
    const test_name = "BlockchainTests/ValidBlocks/bcStateTests/testOpcode_80.json::testOpcode_81_Cancun";
    const test_case = parsed.value.object.get(test_name) orelse return error.TestNotFound;

    // Run the test with path and name for trace generation
    try runner.runJsonTestWithPathAndName(allocator, test_case, json_path, test_name);
}

test "BlockchainTests_ValidBlocks_bcStateTests_testOpcode_80_json__testOpcode_81_Prague" {
    const allocator = testing.allocator;

    // Read and parse the JSON test file
    const json_path = "execution-spec-tests/fixtures/blockchain_tests/ValidBlocks/bcStateTests/testOpcode_80.json";
    const json_content = try std.fs.cwd().readFileAlloc(allocator, json_path, 100 * 1024 * 1024);
    defer allocator.free(json_content);

    const parsed = try std.json.parseFromSlice(std.json.Value, allocator, json_content, .{});
    defer parsed.deinit();

    // Get the specific test case
    const test_name = "BlockchainTests/ValidBlocks/bcStateTests/testOpcode_80.json::testOpcode_81_Prague";
    const test_case = parsed.value.object.get(test_name) orelse return error.TestNotFound;

    // Run the test with path and name for trace generation
    try runner.runJsonTestWithPathAndName(allocator, test_case, json_path, test_name);
}

test "BlockchainTests_ValidBlocks_bcStateTests_testOpcode_80_json__testOpcode_82_Cancun" {
    const allocator = testing.allocator;

    // Read and parse the JSON test file
    const json_path = "execution-spec-tests/fixtures/blockchain_tests/ValidBlocks/bcStateTests/testOpcode_80.json";
    const json_content = try std.fs.cwd().readFileAlloc(allocator, json_path, 100 * 1024 * 1024);
    defer allocator.free(json_content);

    const parsed = try std.json.parseFromSlice(std.json.Value, allocator, json_content, .{});
    defer parsed.deinit();

    // Get the specific test case
    const test_name = "BlockchainTests/ValidBlocks/bcStateTests/testOpcode_80.json::testOpcode_82_Cancun";
    const test_case = parsed.value.object.get(test_name) orelse return error.TestNotFound;

    // Run the test with path and name for trace generation
    try runner.runJsonTestWithPathAndName(allocator, test_case, json_path, test_name);
}

test "BlockchainTests_ValidBlocks_bcStateTests_testOpcode_80_json__testOpcode_82_Prague" {
    const allocator = testing.allocator;

    // Read and parse the JSON test file
    const json_path = "execution-spec-tests/fixtures/blockchain_tests/ValidBlocks/bcStateTests/testOpcode_80.json";
    const json_content = try std.fs.cwd().readFileAlloc(allocator, json_path, 100 * 1024 * 1024);
    defer allocator.free(json_content);

    const parsed = try std.json.parseFromSlice(std.json.Value, allocator, json_content, .{});
    defer parsed.deinit();

    // Get the specific test case
    const test_name = "BlockchainTests/ValidBlocks/bcStateTests/testOpcode_80.json::testOpcode_82_Prague";
    const test_case = parsed.value.object.get(test_name) orelse return error.TestNotFound;

    // Run the test with path and name for trace generation
    try runner.runJsonTestWithPathAndName(allocator, test_case, json_path, test_name);
}

test "BlockchainTests_ValidBlocks_bcStateTests_testOpcode_80_json__testOpcode_83_Cancun" {
    const allocator = testing.allocator;

    // Read and parse the JSON test file
    const json_path = "execution-spec-tests/fixtures/blockchain_tests/ValidBlocks/bcStateTests/testOpcode_80.json";
    const json_content = try std.fs.cwd().readFileAlloc(allocator, json_path, 100 * 1024 * 1024);
    defer allocator.free(json_content);

    const parsed = try std.json.parseFromSlice(std.json.Value, allocator, json_content, .{});
    defer parsed.deinit();

    // Get the specific test case
    const test_name = "BlockchainTests/ValidBlocks/bcStateTests/testOpcode_80.json::testOpcode_83_Cancun";
    const test_case = parsed.value.object.get(test_name) orelse return error.TestNotFound;

    // Run the test with path and name for trace generation
    try runner.runJsonTestWithPathAndName(allocator, test_case, json_path, test_name);
}

test "BlockchainTests_ValidBlocks_bcStateTests_testOpcode_80_json__testOpcode_83_Prague" {
    const allocator = testing.allocator;

    // Read and parse the JSON test file
    const json_path = "execution-spec-tests/fixtures/blockchain_tests/ValidBlocks/bcStateTests/testOpcode_80.json";
    const json_content = try std.fs.cwd().readFileAlloc(allocator, json_path, 100 * 1024 * 1024);
    defer allocator.free(json_content);

    const parsed = try std.json.parseFromSlice(std.json.Value, allocator, json_content, .{});
    defer parsed.deinit();

    // Get the specific test case
    const test_name = "BlockchainTests/ValidBlocks/bcStateTests/testOpcode_80.json::testOpcode_83_Prague";
    const test_case = parsed.value.object.get(test_name) orelse return error.TestNotFound;

    // Run the test with path and name for trace generation
    try runner.runJsonTestWithPathAndName(allocator, test_case, json_path, test_name);
}

test "BlockchainTests_ValidBlocks_bcStateTests_testOpcode_80_json__testOpcode_84_Cancun" {
    const allocator = testing.allocator;

    // Read and parse the JSON test file
    const json_path = "execution-spec-tests/fixtures/blockchain_tests/ValidBlocks/bcStateTests/testOpcode_80.json";
    const json_content = try std.fs.cwd().readFileAlloc(allocator, json_path, 100 * 1024 * 1024);
    defer allocator.free(json_content);

    const parsed = try std.json.parseFromSlice(std.json.Value, allocator, json_content, .{});
    defer parsed.deinit();

    // Get the specific test case
    const test_name = "BlockchainTests/ValidBlocks/bcStateTests/testOpcode_80.json::testOpcode_84_Cancun";
    const test_case = parsed.value.object.get(test_name) orelse return error.TestNotFound;

    // Run the test with path and name for trace generation
    try runner.runJsonTestWithPathAndName(allocator, test_case, json_path, test_name);
}

test "BlockchainTests_ValidBlocks_bcStateTests_testOpcode_80_json__testOpcode_84_Prague" {
    const allocator = testing.allocator;

    // Read and parse the JSON test file
    const json_path = "execution-spec-tests/fixtures/blockchain_tests/ValidBlocks/bcStateTests/testOpcode_80.json";
    const json_content = try std.fs.cwd().readFileAlloc(allocator, json_path, 100 * 1024 * 1024);
    defer allocator.free(json_content);

    const parsed = try std.json.parseFromSlice(std.json.Value, allocator, json_content, .{});
    defer parsed.deinit();

    // Get the specific test case
    const test_name = "BlockchainTests/ValidBlocks/bcStateTests/testOpcode_80.json::testOpcode_84_Prague";
    const test_case = parsed.value.object.get(test_name) orelse return error.TestNotFound;

    // Run the test with path and name for trace generation
    try runner.runJsonTestWithPathAndName(allocator, test_case, json_path, test_name);
}

test "BlockchainTests_ValidBlocks_bcStateTests_testOpcode_80_json__testOpcode_85_Cancun" {
    const allocator = testing.allocator;

    // Read and parse the JSON test file
    const json_path = "execution-spec-tests/fixtures/blockchain_tests/ValidBlocks/bcStateTests/testOpcode_80.json";
    const json_content = try std.fs.cwd().readFileAlloc(allocator, json_path, 100 * 1024 * 1024);
    defer allocator.free(json_content);

    const parsed = try std.json.parseFromSlice(std.json.Value, allocator, json_content, .{});
    defer parsed.deinit();

    // Get the specific test case
    const test_name = "BlockchainTests/ValidBlocks/bcStateTests/testOpcode_80.json::testOpcode_85_Cancun";
    const test_case = parsed.value.object.get(test_name) orelse return error.TestNotFound;

    // Run the test with path and name for trace generation
    try runner.runJsonTestWithPathAndName(allocator, test_case, json_path, test_name);
}

test "BlockchainTests_ValidBlocks_bcStateTests_testOpcode_80_json__testOpcode_85_Prague" {
    const allocator = testing.allocator;

    // Read and parse the JSON test file
    const json_path = "execution-spec-tests/fixtures/blockchain_tests/ValidBlocks/bcStateTests/testOpcode_80.json";
    const json_content = try std.fs.cwd().readFileAlloc(allocator, json_path, 100 * 1024 * 1024);
    defer allocator.free(json_content);

    const parsed = try std.json.parseFromSlice(std.json.Value, allocator, json_content, .{});
    defer parsed.deinit();

    // Get the specific test case
    const test_name = "BlockchainTests/ValidBlocks/bcStateTests/testOpcode_80.json::testOpcode_85_Prague";
    const test_case = parsed.value.object.get(test_name) orelse return error.TestNotFound;

    // Run the test with path and name for trace generation
    try runner.runJsonTestWithPathAndName(allocator, test_case, json_path, test_name);
}

test "BlockchainTests_ValidBlocks_bcStateTests_testOpcode_80_json__testOpcode_86_Cancun" {
    const allocator = testing.allocator;

    // Read and parse the JSON test file
    const json_path = "execution-spec-tests/fixtures/blockchain_tests/ValidBlocks/bcStateTests/testOpcode_80.json";
    const json_content = try std.fs.cwd().readFileAlloc(allocator, json_path, 100 * 1024 * 1024);
    defer allocator.free(json_content);

    const parsed = try std.json.parseFromSlice(std.json.Value, allocator, json_content, .{});
    defer parsed.deinit();

    // Get the specific test case
    const test_name = "BlockchainTests/ValidBlocks/bcStateTests/testOpcode_80.json::testOpcode_86_Cancun";
    const test_case = parsed.value.object.get(test_name) orelse return error.TestNotFound;

    // Run the test with path and name for trace generation
    try runner.runJsonTestWithPathAndName(allocator, test_case, json_path, test_name);
}

test "BlockchainTests_ValidBlocks_bcStateTests_testOpcode_80_json__testOpcode_86_Prague" {
    const allocator = testing.allocator;

    // Read and parse the JSON test file
    const json_path = "execution-spec-tests/fixtures/blockchain_tests/ValidBlocks/bcStateTests/testOpcode_80.json";
    const json_content = try std.fs.cwd().readFileAlloc(allocator, json_path, 100 * 1024 * 1024);
    defer allocator.free(json_content);

    const parsed = try std.json.parseFromSlice(std.json.Value, allocator, json_content, .{});
    defer parsed.deinit();

    // Get the specific test case
    const test_name = "BlockchainTests/ValidBlocks/bcStateTests/testOpcode_80.json::testOpcode_86_Prague";
    const test_case = parsed.value.object.get(test_name) orelse return error.TestNotFound;

    // Run the test with path and name for trace generation
    try runner.runJsonTestWithPathAndName(allocator, test_case, json_path, test_name);
}

test "BlockchainTests_ValidBlocks_bcStateTests_testOpcode_80_json__testOpcode_87_Cancun" {
    const allocator = testing.allocator;

    // Read and parse the JSON test file
    const json_path = "execution-spec-tests/fixtures/blockchain_tests/ValidBlocks/bcStateTests/testOpcode_80.json";
    const json_content = try std.fs.cwd().readFileAlloc(allocator, json_path, 100 * 1024 * 1024);
    defer allocator.free(json_content);

    const parsed = try std.json.parseFromSlice(std.json.Value, allocator, json_content, .{});
    defer parsed.deinit();

    // Get the specific test case
    const test_name = "BlockchainTests/ValidBlocks/bcStateTests/testOpcode_80.json::testOpcode_87_Cancun";
    const test_case = parsed.value.object.get(test_name) orelse return error.TestNotFound;

    // Run the test with path and name for trace generation
    try runner.runJsonTestWithPathAndName(allocator, test_case, json_path, test_name);
}

test "BlockchainTests_ValidBlocks_bcStateTests_testOpcode_80_json__testOpcode_87_Prague" {
    const allocator = testing.allocator;

    // Read and parse the JSON test file
    const json_path = "execution-spec-tests/fixtures/blockchain_tests/ValidBlocks/bcStateTests/testOpcode_80.json";
    const json_content = try std.fs.cwd().readFileAlloc(allocator, json_path, 100 * 1024 * 1024);
    defer allocator.free(json_content);

    const parsed = try std.json.parseFromSlice(std.json.Value, allocator, json_content, .{});
    defer parsed.deinit();

    // Get the specific test case
    const test_name = "BlockchainTests/ValidBlocks/bcStateTests/testOpcode_80.json::testOpcode_87_Prague";
    const test_case = parsed.value.object.get(test_name) orelse return error.TestNotFound;

    // Run the test with path and name for trace generation
    try runner.runJsonTestWithPathAndName(allocator, test_case, json_path, test_name);
}

test "BlockchainTests_ValidBlocks_bcStateTests_testOpcode_80_json__testOpcode_88_Cancun" {
    const allocator = testing.allocator;

    // Read and parse the JSON test file
    const json_path = "execution-spec-tests/fixtures/blockchain_tests/ValidBlocks/bcStateTests/testOpcode_80.json";
    const json_content = try std.fs.cwd().readFileAlloc(allocator, json_path, 100 * 1024 * 1024);
    defer allocator.free(json_content);

    const parsed = try std.json.parseFromSlice(std.json.Value, allocator, json_content, .{});
    defer parsed.deinit();

    // Get the specific test case
    const test_name = "BlockchainTests/ValidBlocks/bcStateTests/testOpcode_80.json::testOpcode_88_Cancun";
    const test_case = parsed.value.object.get(test_name) orelse return error.TestNotFound;

    // Run the test with path and name for trace generation
    try runner.runJsonTestWithPathAndName(allocator, test_case, json_path, test_name);
}

test "BlockchainTests_ValidBlocks_bcStateTests_testOpcode_80_json__testOpcode_88_Prague" {
    const allocator = testing.allocator;

    // Read and parse the JSON test file
    const json_path = "execution-spec-tests/fixtures/blockchain_tests/ValidBlocks/bcStateTests/testOpcode_80.json";
    const json_content = try std.fs.cwd().readFileAlloc(allocator, json_path, 100 * 1024 * 1024);
    defer allocator.free(json_content);

    const parsed = try std.json.parseFromSlice(std.json.Value, allocator, json_content, .{});
    defer parsed.deinit();

    // Get the specific test case
    const test_name = "BlockchainTests/ValidBlocks/bcStateTests/testOpcode_80.json::testOpcode_88_Prague";
    const test_case = parsed.value.object.get(test_name) orelse return error.TestNotFound;

    // Run the test with path and name for trace generation
    try runner.runJsonTestWithPathAndName(allocator, test_case, json_path, test_name);
}

test "BlockchainTests_ValidBlocks_bcStateTests_testOpcode_80_json__testOpcode_89_Cancun" {
    const allocator = testing.allocator;

    // Read and parse the JSON test file
    const json_path = "execution-spec-tests/fixtures/blockchain_tests/ValidBlocks/bcStateTests/testOpcode_80.json";
    const json_content = try std.fs.cwd().readFileAlloc(allocator, json_path, 100 * 1024 * 1024);
    defer allocator.free(json_content);

    const parsed = try std.json.parseFromSlice(std.json.Value, allocator, json_content, .{});
    defer parsed.deinit();

    // Get the specific test case
    const test_name = "BlockchainTests/ValidBlocks/bcStateTests/testOpcode_80.json::testOpcode_89_Cancun";
    const test_case = parsed.value.object.get(test_name) orelse return error.TestNotFound;

    // Run the test with path and name for trace generation
    try runner.runJsonTestWithPathAndName(allocator, test_case, json_path, test_name);
}

test "BlockchainTests_ValidBlocks_bcStateTests_testOpcode_80_json__testOpcode_89_Prague" {
    const allocator = testing.allocator;

    // Read and parse the JSON test file
    const json_path = "execution-spec-tests/fixtures/blockchain_tests/ValidBlocks/bcStateTests/testOpcode_80.json";
    const json_content = try std.fs.cwd().readFileAlloc(allocator, json_path, 100 * 1024 * 1024);
    defer allocator.free(json_content);

    const parsed = try std.json.parseFromSlice(std.json.Value, allocator, json_content, .{});
    defer parsed.deinit();

    // Get the specific test case
    const test_name = "BlockchainTests/ValidBlocks/bcStateTests/testOpcode_80.json::testOpcode_89_Prague";
    const test_case = parsed.value.object.get(test_name) orelse return error.TestNotFound;

    // Run the test with path and name for trace generation
    try runner.runJsonTestWithPathAndName(allocator, test_case, json_path, test_name);
}

test "BlockchainTests_ValidBlocks_bcStateTests_testOpcode_80_json__testOpcode_8a_Cancun" {
    const allocator = testing.allocator;

    // Read and parse the JSON test file
    const json_path = "execution-spec-tests/fixtures/blockchain_tests/ValidBlocks/bcStateTests/testOpcode_80.json";
    const json_content = try std.fs.cwd().readFileAlloc(allocator, json_path, 100 * 1024 * 1024);
    defer allocator.free(json_content);

    const parsed = try std.json.parseFromSlice(std.json.Value, allocator, json_content, .{});
    defer parsed.deinit();

    // Get the specific test case
    const test_name = "BlockchainTests/ValidBlocks/bcStateTests/testOpcode_80.json::testOpcode_8a_Cancun";
    const test_case = parsed.value.object.get(test_name) orelse return error.TestNotFound;

    // Run the test with path and name for trace generation
    try runner.runJsonTestWithPathAndName(allocator, test_case, json_path, test_name);
}

test "BlockchainTests_ValidBlocks_bcStateTests_testOpcode_80_json__testOpcode_8a_Prague" {
    const allocator = testing.allocator;

    // Read and parse the JSON test file
    const json_path = "execution-spec-tests/fixtures/blockchain_tests/ValidBlocks/bcStateTests/testOpcode_80.json";
    const json_content = try std.fs.cwd().readFileAlloc(allocator, json_path, 100 * 1024 * 1024);
    defer allocator.free(json_content);

    const parsed = try std.json.parseFromSlice(std.json.Value, allocator, json_content, .{});
    defer parsed.deinit();

    // Get the specific test case
    const test_name = "BlockchainTests/ValidBlocks/bcStateTests/testOpcode_80.json::testOpcode_8a_Prague";
    const test_case = parsed.value.object.get(test_name) orelse return error.TestNotFound;

    // Run the test with path and name for trace generation
    try runner.runJsonTestWithPathAndName(allocator, test_case, json_path, test_name);
}

test "BlockchainTests_ValidBlocks_bcStateTests_testOpcode_80_json__testOpcode_8b_Cancun" {
    const allocator = testing.allocator;

    // Read and parse the JSON test file
    const json_path = "execution-spec-tests/fixtures/blockchain_tests/ValidBlocks/bcStateTests/testOpcode_80.json";
    const json_content = try std.fs.cwd().readFileAlloc(allocator, json_path, 100 * 1024 * 1024);
    defer allocator.free(json_content);

    const parsed = try std.json.parseFromSlice(std.json.Value, allocator, json_content, .{});
    defer parsed.deinit();

    // Get the specific test case
    const test_name = "BlockchainTests/ValidBlocks/bcStateTests/testOpcode_80.json::testOpcode_8b_Cancun";
    const test_case = parsed.value.object.get(test_name) orelse return error.TestNotFound;

    // Run the test with path and name for trace generation
    try runner.runJsonTestWithPathAndName(allocator, test_case, json_path, test_name);
}

test "BlockchainTests_ValidBlocks_bcStateTests_testOpcode_80_json__testOpcode_8b_Prague" {
    const allocator = testing.allocator;

    // Read and parse the JSON test file
    const json_path = "execution-spec-tests/fixtures/blockchain_tests/ValidBlocks/bcStateTests/testOpcode_80.json";
    const json_content = try std.fs.cwd().readFileAlloc(allocator, json_path, 100 * 1024 * 1024);
    defer allocator.free(json_content);

    const parsed = try std.json.parseFromSlice(std.json.Value, allocator, json_content, .{});
    defer parsed.deinit();

    // Get the specific test case
    const test_name = "BlockchainTests/ValidBlocks/bcStateTests/testOpcode_80.json::testOpcode_8b_Prague";
    const test_case = parsed.value.object.get(test_name) orelse return error.TestNotFound;

    // Run the test with path and name for trace generation
    try runner.runJsonTestWithPathAndName(allocator, test_case, json_path, test_name);
}

test "BlockchainTests_ValidBlocks_bcStateTests_testOpcode_80_json__testOpcode_8c_Cancun" {
    const allocator = testing.allocator;

    // Read and parse the JSON test file
    const json_path = "execution-spec-tests/fixtures/blockchain_tests/ValidBlocks/bcStateTests/testOpcode_80.json";
    const json_content = try std.fs.cwd().readFileAlloc(allocator, json_path, 100 * 1024 * 1024);
    defer allocator.free(json_content);

    const parsed = try std.json.parseFromSlice(std.json.Value, allocator, json_content, .{});
    defer parsed.deinit();

    // Get the specific test case
    const test_name = "BlockchainTests/ValidBlocks/bcStateTests/testOpcode_80.json::testOpcode_8c_Cancun";
    const test_case = parsed.value.object.get(test_name) orelse return error.TestNotFound;

    // Run the test with path and name for trace generation
    try runner.runJsonTestWithPathAndName(allocator, test_case, json_path, test_name);
}

test "BlockchainTests_ValidBlocks_bcStateTests_testOpcode_80_json__testOpcode_8c_Prague" {
    const allocator = testing.allocator;

    // Read and parse the JSON test file
    const json_path = "execution-spec-tests/fixtures/blockchain_tests/ValidBlocks/bcStateTests/testOpcode_80.json";
    const json_content = try std.fs.cwd().readFileAlloc(allocator, json_path, 100 * 1024 * 1024);
    defer allocator.free(json_content);

    const parsed = try std.json.parseFromSlice(std.json.Value, allocator, json_content, .{});
    defer parsed.deinit();

    // Get the specific test case
    const test_name = "BlockchainTests/ValidBlocks/bcStateTests/testOpcode_80.json::testOpcode_8c_Prague";
    const test_case = parsed.value.object.get(test_name) orelse return error.TestNotFound;

    // Run the test with path and name for trace generation
    try runner.runJsonTestWithPathAndName(allocator, test_case, json_path, test_name);
}

test "BlockchainTests_ValidBlocks_bcStateTests_testOpcode_80_json__testOpcode_8d_Cancun" {
    const allocator = testing.allocator;

    // Read and parse the JSON test file
    const json_path = "execution-spec-tests/fixtures/blockchain_tests/ValidBlocks/bcStateTests/testOpcode_80.json";
    const json_content = try std.fs.cwd().readFileAlloc(allocator, json_path, 100 * 1024 * 1024);
    defer allocator.free(json_content);

    const parsed = try std.json.parseFromSlice(std.json.Value, allocator, json_content, .{});
    defer parsed.deinit();

    // Get the specific test case
    const test_name = "BlockchainTests/ValidBlocks/bcStateTests/testOpcode_80.json::testOpcode_8d_Cancun";
    const test_case = parsed.value.object.get(test_name) orelse return error.TestNotFound;

    // Run the test with path and name for trace generation
    try runner.runJsonTestWithPathAndName(allocator, test_case, json_path, test_name);
}

test "BlockchainTests_ValidBlocks_bcStateTests_testOpcode_80_json__testOpcode_8d_Prague" {
    const allocator = testing.allocator;

    // Read and parse the JSON test file
    const json_path = "execution-spec-tests/fixtures/blockchain_tests/ValidBlocks/bcStateTests/testOpcode_80.json";
    const json_content = try std.fs.cwd().readFileAlloc(allocator, json_path, 100 * 1024 * 1024);
    defer allocator.free(json_content);

    const parsed = try std.json.parseFromSlice(std.json.Value, allocator, json_content, .{});
    defer parsed.deinit();

    // Get the specific test case
    const test_name = "BlockchainTests/ValidBlocks/bcStateTests/testOpcode_80.json::testOpcode_8d_Prague";
    const test_case = parsed.value.object.get(test_name) orelse return error.TestNotFound;

    // Run the test with path and name for trace generation
    try runner.runJsonTestWithPathAndName(allocator, test_case, json_path, test_name);
}

test "BlockchainTests_ValidBlocks_bcStateTests_testOpcode_80_json__testOpcode_8e_Cancun" {
    const allocator = testing.allocator;

    // Read and parse the JSON test file
    const json_path = "execution-spec-tests/fixtures/blockchain_tests/ValidBlocks/bcStateTests/testOpcode_80.json";
    const json_content = try std.fs.cwd().readFileAlloc(allocator, json_path, 100 * 1024 * 1024);
    defer allocator.free(json_content);

    const parsed = try std.json.parseFromSlice(std.json.Value, allocator, json_content, .{});
    defer parsed.deinit();

    // Get the specific test case
    const test_name = "BlockchainTests/ValidBlocks/bcStateTests/testOpcode_80.json::testOpcode_8e_Cancun";
    const test_case = parsed.value.object.get(test_name) orelse return error.TestNotFound;

    // Run the test with path and name for trace generation
    try runner.runJsonTestWithPathAndName(allocator, test_case, json_path, test_name);
}

test "BlockchainTests_ValidBlocks_bcStateTests_testOpcode_80_json__testOpcode_8e_Prague" {
    const allocator = testing.allocator;

    // Read and parse the JSON test file
    const json_path = "execution-spec-tests/fixtures/blockchain_tests/ValidBlocks/bcStateTests/testOpcode_80.json";
    const json_content = try std.fs.cwd().readFileAlloc(allocator, json_path, 100 * 1024 * 1024);
    defer allocator.free(json_content);

    const parsed = try std.json.parseFromSlice(std.json.Value, allocator, json_content, .{});
    defer parsed.deinit();

    // Get the specific test case
    const test_name = "BlockchainTests/ValidBlocks/bcStateTests/testOpcode_80.json::testOpcode_8e_Prague";
    const test_case = parsed.value.object.get(test_name) orelse return error.TestNotFound;

    // Run the test with path and name for trace generation
    try runner.runJsonTestWithPathAndName(allocator, test_case, json_path, test_name);
}

test "BlockchainTests_ValidBlocks_bcStateTests_testOpcode_80_json__testOpcode_8f_Cancun" {
    const allocator = testing.allocator;

    // Read and parse the JSON test file
    const json_path = "execution-spec-tests/fixtures/blockchain_tests/ValidBlocks/bcStateTests/testOpcode_80.json";
    const json_content = try std.fs.cwd().readFileAlloc(allocator, json_path, 100 * 1024 * 1024);
    defer allocator.free(json_content);

    const parsed = try std.json.parseFromSlice(std.json.Value, allocator, json_content, .{});
    defer parsed.deinit();

    // Get the specific test case
    const test_name = "BlockchainTests/ValidBlocks/bcStateTests/testOpcode_80.json::testOpcode_8f_Cancun";
    const test_case = parsed.value.object.get(test_name) orelse return error.TestNotFound;

    // Run the test with path and name for trace generation
    try runner.runJsonTestWithPathAndName(allocator, test_case, json_path, test_name);
}

test "BlockchainTests_ValidBlocks_bcStateTests_testOpcode_80_json__testOpcode_8f_Prague" {
    const allocator = testing.allocator;

    // Read and parse the JSON test file
    const json_path = "execution-spec-tests/fixtures/blockchain_tests/ValidBlocks/bcStateTests/testOpcode_80.json";
    const json_content = try std.fs.cwd().readFileAlloc(allocator, json_path, 100 * 1024 * 1024);
    defer allocator.free(json_content);

    const parsed = try std.json.parseFromSlice(std.json.Value, allocator, json_content, .{});
    defer parsed.deinit();

    // Get the specific test case
    const test_name = "BlockchainTests/ValidBlocks/bcStateTests/testOpcode_80.json::testOpcode_8f_Prague";
    const test_case = parsed.value.object.get(test_name) orelse return error.TestNotFound;

    // Run the test with path and name for trace generation
    try runner.runJsonTestWithPathAndName(allocator, test_case, json_path, test_name);
}
