const std = @import("std");
const testing = std.testing;
const root = @import("../../../root.zig");
const runner = root.runner;

test "BlockchainTests_ValidBlocks_bcStateTests_testOpcode_b0_json__testOpcode_b0_Cancun" {
    const allocator = testing.allocator;

    // Read and parse the JSON test file
    const json_path = "execution-spec-tests/fixtures/blockchain_tests/ValidBlocks/bcStateTests/testOpcode_b0.json";
    const json_content = try std.fs.cwd().readFileAlloc(allocator, json_path, 100 * 1024 * 1024);
    defer allocator.free(json_content);

    const parsed = try std.json.parseFromSlice(std.json.Value, allocator, json_content, .{});
    defer parsed.deinit();

    // Get the specific test case
    const test_name = "BlockchainTests/ValidBlocks/bcStateTests/testOpcode_b0.json::testOpcode_b0_Cancun";
    const test_case = parsed.value.object.get(test_name) orelse return error.TestNotFound;

    // Run the test with path and name for trace generation
    try runner.runJsonTestWithPathAndName(allocator, test_case, json_path, test_name);
}

test "BlockchainTests_ValidBlocks_bcStateTests_testOpcode_b0_json__testOpcode_b0_Prague" {
    const allocator = testing.allocator;

    // Read and parse the JSON test file
    const json_path = "execution-spec-tests/fixtures/blockchain_tests/ValidBlocks/bcStateTests/testOpcode_b0.json";
    const json_content = try std.fs.cwd().readFileAlloc(allocator, json_path, 100 * 1024 * 1024);
    defer allocator.free(json_content);

    const parsed = try std.json.parseFromSlice(std.json.Value, allocator, json_content, .{});
    defer parsed.deinit();

    // Get the specific test case
    const test_name = "BlockchainTests/ValidBlocks/bcStateTests/testOpcode_b0.json::testOpcode_b0_Prague";
    const test_case = parsed.value.object.get(test_name) orelse return error.TestNotFound;

    // Run the test with path and name for trace generation
    try runner.runJsonTestWithPathAndName(allocator, test_case, json_path, test_name);
}

test "BlockchainTests_ValidBlocks_bcStateTests_testOpcode_b0_json__testOpcode_b1_Cancun" {
    const allocator = testing.allocator;

    // Read and parse the JSON test file
    const json_path = "execution-spec-tests/fixtures/blockchain_tests/ValidBlocks/bcStateTests/testOpcode_b0.json";
    const json_content = try std.fs.cwd().readFileAlloc(allocator, json_path, 100 * 1024 * 1024);
    defer allocator.free(json_content);

    const parsed = try std.json.parseFromSlice(std.json.Value, allocator, json_content, .{});
    defer parsed.deinit();

    // Get the specific test case
    const test_name = "BlockchainTests/ValidBlocks/bcStateTests/testOpcode_b0.json::testOpcode_b1_Cancun";
    const test_case = parsed.value.object.get(test_name) orelse return error.TestNotFound;

    // Run the test with path and name for trace generation
    try runner.runJsonTestWithPathAndName(allocator, test_case, json_path, test_name);
}

test "BlockchainTests_ValidBlocks_bcStateTests_testOpcode_b0_json__testOpcode_b1_Prague" {
    const allocator = testing.allocator;

    // Read and parse the JSON test file
    const json_path = "execution-spec-tests/fixtures/blockchain_tests/ValidBlocks/bcStateTests/testOpcode_b0.json";
    const json_content = try std.fs.cwd().readFileAlloc(allocator, json_path, 100 * 1024 * 1024);
    defer allocator.free(json_content);

    const parsed = try std.json.parseFromSlice(std.json.Value, allocator, json_content, .{});
    defer parsed.deinit();

    // Get the specific test case
    const test_name = "BlockchainTests/ValidBlocks/bcStateTests/testOpcode_b0.json::testOpcode_b1_Prague";
    const test_case = parsed.value.object.get(test_name) orelse return error.TestNotFound;

    // Run the test with path and name for trace generation
    try runner.runJsonTestWithPathAndName(allocator, test_case, json_path, test_name);
}

test "BlockchainTests_ValidBlocks_bcStateTests_testOpcode_b0_json__testOpcode_b2_Cancun" {
    const allocator = testing.allocator;

    // Read and parse the JSON test file
    const json_path = "execution-spec-tests/fixtures/blockchain_tests/ValidBlocks/bcStateTests/testOpcode_b0.json";
    const json_content = try std.fs.cwd().readFileAlloc(allocator, json_path, 100 * 1024 * 1024);
    defer allocator.free(json_content);

    const parsed = try std.json.parseFromSlice(std.json.Value, allocator, json_content, .{});
    defer parsed.deinit();

    // Get the specific test case
    const test_name = "BlockchainTests/ValidBlocks/bcStateTests/testOpcode_b0.json::testOpcode_b2_Cancun";
    const test_case = parsed.value.object.get(test_name) orelse return error.TestNotFound;

    // Run the test with path and name for trace generation
    try runner.runJsonTestWithPathAndName(allocator, test_case, json_path, test_name);
}

test "BlockchainTests_ValidBlocks_bcStateTests_testOpcode_b0_json__testOpcode_b2_Prague" {
    const allocator = testing.allocator;

    // Read and parse the JSON test file
    const json_path = "execution-spec-tests/fixtures/blockchain_tests/ValidBlocks/bcStateTests/testOpcode_b0.json";
    const json_content = try std.fs.cwd().readFileAlloc(allocator, json_path, 100 * 1024 * 1024);
    defer allocator.free(json_content);

    const parsed = try std.json.parseFromSlice(std.json.Value, allocator, json_content, .{});
    defer parsed.deinit();

    // Get the specific test case
    const test_name = "BlockchainTests/ValidBlocks/bcStateTests/testOpcode_b0.json::testOpcode_b2_Prague";
    const test_case = parsed.value.object.get(test_name) orelse return error.TestNotFound;

    // Run the test with path and name for trace generation
    try runner.runJsonTestWithPathAndName(allocator, test_case, json_path, test_name);
}

test "BlockchainTests_ValidBlocks_bcStateTests_testOpcode_b0_json__testOpcode_b3_Cancun" {
    const allocator = testing.allocator;

    // Read and parse the JSON test file
    const json_path = "execution-spec-tests/fixtures/blockchain_tests/ValidBlocks/bcStateTests/testOpcode_b0.json";
    const json_content = try std.fs.cwd().readFileAlloc(allocator, json_path, 100 * 1024 * 1024);
    defer allocator.free(json_content);

    const parsed = try std.json.parseFromSlice(std.json.Value, allocator, json_content, .{});
    defer parsed.deinit();

    // Get the specific test case
    const test_name = "BlockchainTests/ValidBlocks/bcStateTests/testOpcode_b0.json::testOpcode_b3_Cancun";
    const test_case = parsed.value.object.get(test_name) orelse return error.TestNotFound;

    // Run the test with path and name for trace generation
    try runner.runJsonTestWithPathAndName(allocator, test_case, json_path, test_name);
}

test "BlockchainTests_ValidBlocks_bcStateTests_testOpcode_b0_json__testOpcode_b3_Prague" {
    const allocator = testing.allocator;

    // Read and parse the JSON test file
    const json_path = "execution-spec-tests/fixtures/blockchain_tests/ValidBlocks/bcStateTests/testOpcode_b0.json";
    const json_content = try std.fs.cwd().readFileAlloc(allocator, json_path, 100 * 1024 * 1024);
    defer allocator.free(json_content);

    const parsed = try std.json.parseFromSlice(std.json.Value, allocator, json_content, .{});
    defer parsed.deinit();

    // Get the specific test case
    const test_name = "BlockchainTests/ValidBlocks/bcStateTests/testOpcode_b0.json::testOpcode_b3_Prague";
    const test_case = parsed.value.object.get(test_name) orelse return error.TestNotFound;

    // Run the test with path and name for trace generation
    try runner.runJsonTestWithPathAndName(allocator, test_case, json_path, test_name);
}

test "BlockchainTests_ValidBlocks_bcStateTests_testOpcode_b0_json__testOpcode_b4_Cancun" {
    const allocator = testing.allocator;

    // Read and parse the JSON test file
    const json_path = "execution-spec-tests/fixtures/blockchain_tests/ValidBlocks/bcStateTests/testOpcode_b0.json";
    const json_content = try std.fs.cwd().readFileAlloc(allocator, json_path, 100 * 1024 * 1024);
    defer allocator.free(json_content);

    const parsed = try std.json.parseFromSlice(std.json.Value, allocator, json_content, .{});
    defer parsed.deinit();

    // Get the specific test case
    const test_name = "BlockchainTests/ValidBlocks/bcStateTests/testOpcode_b0.json::testOpcode_b4_Cancun";
    const test_case = parsed.value.object.get(test_name) orelse return error.TestNotFound;

    // Run the test with path and name for trace generation
    try runner.runJsonTestWithPathAndName(allocator, test_case, json_path, test_name);
}

test "BlockchainTests_ValidBlocks_bcStateTests_testOpcode_b0_json__testOpcode_b4_Prague" {
    const allocator = testing.allocator;

    // Read and parse the JSON test file
    const json_path = "execution-spec-tests/fixtures/blockchain_tests/ValidBlocks/bcStateTests/testOpcode_b0.json";
    const json_content = try std.fs.cwd().readFileAlloc(allocator, json_path, 100 * 1024 * 1024);
    defer allocator.free(json_content);

    const parsed = try std.json.parseFromSlice(std.json.Value, allocator, json_content, .{});
    defer parsed.deinit();

    // Get the specific test case
    const test_name = "BlockchainTests/ValidBlocks/bcStateTests/testOpcode_b0.json::testOpcode_b4_Prague";
    const test_case = parsed.value.object.get(test_name) orelse return error.TestNotFound;

    // Run the test with path and name for trace generation
    try runner.runJsonTestWithPathAndName(allocator, test_case, json_path, test_name);
}

test "BlockchainTests_ValidBlocks_bcStateTests_testOpcode_b0_json__testOpcode_b5_Cancun" {
    const allocator = testing.allocator;

    // Read and parse the JSON test file
    const json_path = "execution-spec-tests/fixtures/blockchain_tests/ValidBlocks/bcStateTests/testOpcode_b0.json";
    const json_content = try std.fs.cwd().readFileAlloc(allocator, json_path, 100 * 1024 * 1024);
    defer allocator.free(json_content);

    const parsed = try std.json.parseFromSlice(std.json.Value, allocator, json_content, .{});
    defer parsed.deinit();

    // Get the specific test case
    const test_name = "BlockchainTests/ValidBlocks/bcStateTests/testOpcode_b0.json::testOpcode_b5_Cancun";
    const test_case = parsed.value.object.get(test_name) orelse return error.TestNotFound;

    // Run the test with path and name for trace generation
    try runner.runJsonTestWithPathAndName(allocator, test_case, json_path, test_name);
}

test "BlockchainTests_ValidBlocks_bcStateTests_testOpcode_b0_json__testOpcode_b5_Prague" {
    const allocator = testing.allocator;

    // Read and parse the JSON test file
    const json_path = "execution-spec-tests/fixtures/blockchain_tests/ValidBlocks/bcStateTests/testOpcode_b0.json";
    const json_content = try std.fs.cwd().readFileAlloc(allocator, json_path, 100 * 1024 * 1024);
    defer allocator.free(json_content);

    const parsed = try std.json.parseFromSlice(std.json.Value, allocator, json_content, .{});
    defer parsed.deinit();

    // Get the specific test case
    const test_name = "BlockchainTests/ValidBlocks/bcStateTests/testOpcode_b0.json::testOpcode_b5_Prague";
    const test_case = parsed.value.object.get(test_name) orelse return error.TestNotFound;

    // Run the test with path and name for trace generation
    try runner.runJsonTestWithPathAndName(allocator, test_case, json_path, test_name);
}

test "BlockchainTests_ValidBlocks_bcStateTests_testOpcode_b0_json__testOpcode_b6_Cancun" {
    const allocator = testing.allocator;

    // Read and parse the JSON test file
    const json_path = "execution-spec-tests/fixtures/blockchain_tests/ValidBlocks/bcStateTests/testOpcode_b0.json";
    const json_content = try std.fs.cwd().readFileAlloc(allocator, json_path, 100 * 1024 * 1024);
    defer allocator.free(json_content);

    const parsed = try std.json.parseFromSlice(std.json.Value, allocator, json_content, .{});
    defer parsed.deinit();

    // Get the specific test case
    const test_name = "BlockchainTests/ValidBlocks/bcStateTests/testOpcode_b0.json::testOpcode_b6_Cancun";
    const test_case = parsed.value.object.get(test_name) orelse return error.TestNotFound;

    // Run the test with path and name for trace generation
    try runner.runJsonTestWithPathAndName(allocator, test_case, json_path, test_name);
}

test "BlockchainTests_ValidBlocks_bcStateTests_testOpcode_b0_json__testOpcode_b6_Prague" {
    const allocator = testing.allocator;

    // Read and parse the JSON test file
    const json_path = "execution-spec-tests/fixtures/blockchain_tests/ValidBlocks/bcStateTests/testOpcode_b0.json";
    const json_content = try std.fs.cwd().readFileAlloc(allocator, json_path, 100 * 1024 * 1024);
    defer allocator.free(json_content);

    const parsed = try std.json.parseFromSlice(std.json.Value, allocator, json_content, .{});
    defer parsed.deinit();

    // Get the specific test case
    const test_name = "BlockchainTests/ValidBlocks/bcStateTests/testOpcode_b0.json::testOpcode_b6_Prague";
    const test_case = parsed.value.object.get(test_name) orelse return error.TestNotFound;

    // Run the test with path and name for trace generation
    try runner.runJsonTestWithPathAndName(allocator, test_case, json_path, test_name);
}

test "BlockchainTests_ValidBlocks_bcStateTests_testOpcode_b0_json__testOpcode_b7_Cancun" {
    const allocator = testing.allocator;

    // Read and parse the JSON test file
    const json_path = "execution-spec-tests/fixtures/blockchain_tests/ValidBlocks/bcStateTests/testOpcode_b0.json";
    const json_content = try std.fs.cwd().readFileAlloc(allocator, json_path, 100 * 1024 * 1024);
    defer allocator.free(json_content);

    const parsed = try std.json.parseFromSlice(std.json.Value, allocator, json_content, .{});
    defer parsed.deinit();

    // Get the specific test case
    const test_name = "BlockchainTests/ValidBlocks/bcStateTests/testOpcode_b0.json::testOpcode_b7_Cancun";
    const test_case = parsed.value.object.get(test_name) orelse return error.TestNotFound;

    // Run the test with path and name for trace generation
    try runner.runJsonTestWithPathAndName(allocator, test_case, json_path, test_name);
}

test "BlockchainTests_ValidBlocks_bcStateTests_testOpcode_b0_json__testOpcode_b7_Prague" {
    const allocator = testing.allocator;

    // Read and parse the JSON test file
    const json_path = "execution-spec-tests/fixtures/blockchain_tests/ValidBlocks/bcStateTests/testOpcode_b0.json";
    const json_content = try std.fs.cwd().readFileAlloc(allocator, json_path, 100 * 1024 * 1024);
    defer allocator.free(json_content);

    const parsed = try std.json.parseFromSlice(std.json.Value, allocator, json_content, .{});
    defer parsed.deinit();

    // Get the specific test case
    const test_name = "BlockchainTests/ValidBlocks/bcStateTests/testOpcode_b0.json::testOpcode_b7_Prague";
    const test_case = parsed.value.object.get(test_name) orelse return error.TestNotFound;

    // Run the test with path and name for trace generation
    try runner.runJsonTestWithPathAndName(allocator, test_case, json_path, test_name);
}

test "BlockchainTests_ValidBlocks_bcStateTests_testOpcode_b0_json__testOpcode_b8_Cancun" {
    const allocator = testing.allocator;

    // Read and parse the JSON test file
    const json_path = "execution-spec-tests/fixtures/blockchain_tests/ValidBlocks/bcStateTests/testOpcode_b0.json";
    const json_content = try std.fs.cwd().readFileAlloc(allocator, json_path, 100 * 1024 * 1024);
    defer allocator.free(json_content);

    const parsed = try std.json.parseFromSlice(std.json.Value, allocator, json_content, .{});
    defer parsed.deinit();

    // Get the specific test case
    const test_name = "BlockchainTests/ValidBlocks/bcStateTests/testOpcode_b0.json::testOpcode_b8_Cancun";
    const test_case = parsed.value.object.get(test_name) orelse return error.TestNotFound;

    // Run the test with path and name for trace generation
    try runner.runJsonTestWithPathAndName(allocator, test_case, json_path, test_name);
}

test "BlockchainTests_ValidBlocks_bcStateTests_testOpcode_b0_json__testOpcode_b8_Prague" {
    const allocator = testing.allocator;

    // Read and parse the JSON test file
    const json_path = "execution-spec-tests/fixtures/blockchain_tests/ValidBlocks/bcStateTests/testOpcode_b0.json";
    const json_content = try std.fs.cwd().readFileAlloc(allocator, json_path, 100 * 1024 * 1024);
    defer allocator.free(json_content);

    const parsed = try std.json.parseFromSlice(std.json.Value, allocator, json_content, .{});
    defer parsed.deinit();

    // Get the specific test case
    const test_name = "BlockchainTests/ValidBlocks/bcStateTests/testOpcode_b0.json::testOpcode_b8_Prague";
    const test_case = parsed.value.object.get(test_name) orelse return error.TestNotFound;

    // Run the test with path and name for trace generation
    try runner.runJsonTestWithPathAndName(allocator, test_case, json_path, test_name);
}

test "BlockchainTests_ValidBlocks_bcStateTests_testOpcode_b0_json__testOpcode_b9_Cancun" {
    const allocator = testing.allocator;

    // Read and parse the JSON test file
    const json_path = "execution-spec-tests/fixtures/blockchain_tests/ValidBlocks/bcStateTests/testOpcode_b0.json";
    const json_content = try std.fs.cwd().readFileAlloc(allocator, json_path, 100 * 1024 * 1024);
    defer allocator.free(json_content);

    const parsed = try std.json.parseFromSlice(std.json.Value, allocator, json_content, .{});
    defer parsed.deinit();

    // Get the specific test case
    const test_name = "BlockchainTests/ValidBlocks/bcStateTests/testOpcode_b0.json::testOpcode_b9_Cancun";
    const test_case = parsed.value.object.get(test_name) orelse return error.TestNotFound;

    // Run the test with path and name for trace generation
    try runner.runJsonTestWithPathAndName(allocator, test_case, json_path, test_name);
}

test "BlockchainTests_ValidBlocks_bcStateTests_testOpcode_b0_json__testOpcode_b9_Prague" {
    const allocator = testing.allocator;

    // Read and parse the JSON test file
    const json_path = "execution-spec-tests/fixtures/blockchain_tests/ValidBlocks/bcStateTests/testOpcode_b0.json";
    const json_content = try std.fs.cwd().readFileAlloc(allocator, json_path, 100 * 1024 * 1024);
    defer allocator.free(json_content);

    const parsed = try std.json.parseFromSlice(std.json.Value, allocator, json_content, .{});
    defer parsed.deinit();

    // Get the specific test case
    const test_name = "BlockchainTests/ValidBlocks/bcStateTests/testOpcode_b0.json::testOpcode_b9_Prague";
    const test_case = parsed.value.object.get(test_name) orelse return error.TestNotFound;

    // Run the test with path and name for trace generation
    try runner.runJsonTestWithPathAndName(allocator, test_case, json_path, test_name);
}

test "BlockchainTests_ValidBlocks_bcStateTests_testOpcode_b0_json__testOpcode_ba_Cancun" {
    const allocator = testing.allocator;

    // Read and parse the JSON test file
    const json_path = "execution-spec-tests/fixtures/blockchain_tests/ValidBlocks/bcStateTests/testOpcode_b0.json";
    const json_content = try std.fs.cwd().readFileAlloc(allocator, json_path, 100 * 1024 * 1024);
    defer allocator.free(json_content);

    const parsed = try std.json.parseFromSlice(std.json.Value, allocator, json_content, .{});
    defer parsed.deinit();

    // Get the specific test case
    const test_name = "BlockchainTests/ValidBlocks/bcStateTests/testOpcode_b0.json::testOpcode_ba_Cancun";
    const test_case = parsed.value.object.get(test_name) orelse return error.TestNotFound;

    // Run the test with path and name for trace generation
    try runner.runJsonTestWithPathAndName(allocator, test_case, json_path, test_name);
}

test "BlockchainTests_ValidBlocks_bcStateTests_testOpcode_b0_json__testOpcode_ba_Prague" {
    const allocator = testing.allocator;

    // Read and parse the JSON test file
    const json_path = "execution-spec-tests/fixtures/blockchain_tests/ValidBlocks/bcStateTests/testOpcode_b0.json";
    const json_content = try std.fs.cwd().readFileAlloc(allocator, json_path, 100 * 1024 * 1024);
    defer allocator.free(json_content);

    const parsed = try std.json.parseFromSlice(std.json.Value, allocator, json_content, .{});
    defer parsed.deinit();

    // Get the specific test case
    const test_name = "BlockchainTests/ValidBlocks/bcStateTests/testOpcode_b0.json::testOpcode_ba_Prague";
    const test_case = parsed.value.object.get(test_name) orelse return error.TestNotFound;

    // Run the test with path and name for trace generation
    try runner.runJsonTestWithPathAndName(allocator, test_case, json_path, test_name);
}

test "BlockchainTests_ValidBlocks_bcStateTests_testOpcode_b0_json__testOpcode_bb_Cancun" {
    const allocator = testing.allocator;

    // Read and parse the JSON test file
    const json_path = "execution-spec-tests/fixtures/blockchain_tests/ValidBlocks/bcStateTests/testOpcode_b0.json";
    const json_content = try std.fs.cwd().readFileAlloc(allocator, json_path, 100 * 1024 * 1024);
    defer allocator.free(json_content);

    const parsed = try std.json.parseFromSlice(std.json.Value, allocator, json_content, .{});
    defer parsed.deinit();

    // Get the specific test case
    const test_name = "BlockchainTests/ValidBlocks/bcStateTests/testOpcode_b0.json::testOpcode_bb_Cancun";
    const test_case = parsed.value.object.get(test_name) orelse return error.TestNotFound;

    // Run the test with path and name for trace generation
    try runner.runJsonTestWithPathAndName(allocator, test_case, json_path, test_name);
}

test "BlockchainTests_ValidBlocks_bcStateTests_testOpcode_b0_json__testOpcode_bb_Prague" {
    const allocator = testing.allocator;

    // Read and parse the JSON test file
    const json_path = "execution-spec-tests/fixtures/blockchain_tests/ValidBlocks/bcStateTests/testOpcode_b0.json";
    const json_content = try std.fs.cwd().readFileAlloc(allocator, json_path, 100 * 1024 * 1024);
    defer allocator.free(json_content);

    const parsed = try std.json.parseFromSlice(std.json.Value, allocator, json_content, .{});
    defer parsed.deinit();

    // Get the specific test case
    const test_name = "BlockchainTests/ValidBlocks/bcStateTests/testOpcode_b0.json::testOpcode_bb_Prague";
    const test_case = parsed.value.object.get(test_name) orelse return error.TestNotFound;

    // Run the test with path and name for trace generation
    try runner.runJsonTestWithPathAndName(allocator, test_case, json_path, test_name);
}

test "BlockchainTests_ValidBlocks_bcStateTests_testOpcode_b0_json__testOpcode_bc_Cancun" {
    const allocator = testing.allocator;

    // Read and parse the JSON test file
    const json_path = "execution-spec-tests/fixtures/blockchain_tests/ValidBlocks/bcStateTests/testOpcode_b0.json";
    const json_content = try std.fs.cwd().readFileAlloc(allocator, json_path, 100 * 1024 * 1024);
    defer allocator.free(json_content);

    const parsed = try std.json.parseFromSlice(std.json.Value, allocator, json_content, .{});
    defer parsed.deinit();

    // Get the specific test case
    const test_name = "BlockchainTests/ValidBlocks/bcStateTests/testOpcode_b0.json::testOpcode_bc_Cancun";
    const test_case = parsed.value.object.get(test_name) orelse return error.TestNotFound;

    // Run the test with path and name for trace generation
    try runner.runJsonTestWithPathAndName(allocator, test_case, json_path, test_name);
}

test "BlockchainTests_ValidBlocks_bcStateTests_testOpcode_b0_json__testOpcode_bc_Prague" {
    const allocator = testing.allocator;

    // Read and parse the JSON test file
    const json_path = "execution-spec-tests/fixtures/blockchain_tests/ValidBlocks/bcStateTests/testOpcode_b0.json";
    const json_content = try std.fs.cwd().readFileAlloc(allocator, json_path, 100 * 1024 * 1024);
    defer allocator.free(json_content);

    const parsed = try std.json.parseFromSlice(std.json.Value, allocator, json_content, .{});
    defer parsed.deinit();

    // Get the specific test case
    const test_name = "BlockchainTests/ValidBlocks/bcStateTests/testOpcode_b0.json::testOpcode_bc_Prague";
    const test_case = parsed.value.object.get(test_name) orelse return error.TestNotFound;

    // Run the test with path and name for trace generation
    try runner.runJsonTestWithPathAndName(allocator, test_case, json_path, test_name);
}

test "BlockchainTests_ValidBlocks_bcStateTests_testOpcode_b0_json__testOpcode_bd_Cancun" {
    const allocator = testing.allocator;

    // Read and parse the JSON test file
    const json_path = "execution-spec-tests/fixtures/blockchain_tests/ValidBlocks/bcStateTests/testOpcode_b0.json";
    const json_content = try std.fs.cwd().readFileAlloc(allocator, json_path, 100 * 1024 * 1024);
    defer allocator.free(json_content);

    const parsed = try std.json.parseFromSlice(std.json.Value, allocator, json_content, .{});
    defer parsed.deinit();

    // Get the specific test case
    const test_name = "BlockchainTests/ValidBlocks/bcStateTests/testOpcode_b0.json::testOpcode_bd_Cancun";
    const test_case = parsed.value.object.get(test_name) orelse return error.TestNotFound;

    // Run the test with path and name for trace generation
    try runner.runJsonTestWithPathAndName(allocator, test_case, json_path, test_name);
}

test "BlockchainTests_ValidBlocks_bcStateTests_testOpcode_b0_json__testOpcode_bd_Prague" {
    const allocator = testing.allocator;

    // Read and parse the JSON test file
    const json_path = "execution-spec-tests/fixtures/blockchain_tests/ValidBlocks/bcStateTests/testOpcode_b0.json";
    const json_content = try std.fs.cwd().readFileAlloc(allocator, json_path, 100 * 1024 * 1024);
    defer allocator.free(json_content);

    const parsed = try std.json.parseFromSlice(std.json.Value, allocator, json_content, .{});
    defer parsed.deinit();

    // Get the specific test case
    const test_name = "BlockchainTests/ValidBlocks/bcStateTests/testOpcode_b0.json::testOpcode_bd_Prague";
    const test_case = parsed.value.object.get(test_name) orelse return error.TestNotFound;

    // Run the test with path and name for trace generation
    try runner.runJsonTestWithPathAndName(allocator, test_case, json_path, test_name);
}

test "BlockchainTests_ValidBlocks_bcStateTests_testOpcode_b0_json__testOpcode_be_Cancun" {
    const allocator = testing.allocator;

    // Read and parse the JSON test file
    const json_path = "execution-spec-tests/fixtures/blockchain_tests/ValidBlocks/bcStateTests/testOpcode_b0.json";
    const json_content = try std.fs.cwd().readFileAlloc(allocator, json_path, 100 * 1024 * 1024);
    defer allocator.free(json_content);

    const parsed = try std.json.parseFromSlice(std.json.Value, allocator, json_content, .{});
    defer parsed.deinit();

    // Get the specific test case
    const test_name = "BlockchainTests/ValidBlocks/bcStateTests/testOpcode_b0.json::testOpcode_be_Cancun";
    const test_case = parsed.value.object.get(test_name) orelse return error.TestNotFound;

    // Run the test with path and name for trace generation
    try runner.runJsonTestWithPathAndName(allocator, test_case, json_path, test_name);
}

test "BlockchainTests_ValidBlocks_bcStateTests_testOpcode_b0_json__testOpcode_be_Prague" {
    const allocator = testing.allocator;

    // Read and parse the JSON test file
    const json_path = "execution-spec-tests/fixtures/blockchain_tests/ValidBlocks/bcStateTests/testOpcode_b0.json";
    const json_content = try std.fs.cwd().readFileAlloc(allocator, json_path, 100 * 1024 * 1024);
    defer allocator.free(json_content);

    const parsed = try std.json.parseFromSlice(std.json.Value, allocator, json_content, .{});
    defer parsed.deinit();

    // Get the specific test case
    const test_name = "BlockchainTests/ValidBlocks/bcStateTests/testOpcode_b0.json::testOpcode_be_Prague";
    const test_case = parsed.value.object.get(test_name) orelse return error.TestNotFound;

    // Run the test with path and name for trace generation
    try runner.runJsonTestWithPathAndName(allocator, test_case, json_path, test_name);
}

test "BlockchainTests_ValidBlocks_bcStateTests_testOpcode_b0_json__testOpcode_bf_Cancun" {
    const allocator = testing.allocator;

    // Read and parse the JSON test file
    const json_path = "execution-spec-tests/fixtures/blockchain_tests/ValidBlocks/bcStateTests/testOpcode_b0.json";
    const json_content = try std.fs.cwd().readFileAlloc(allocator, json_path, 100 * 1024 * 1024);
    defer allocator.free(json_content);

    const parsed = try std.json.parseFromSlice(std.json.Value, allocator, json_content, .{});
    defer parsed.deinit();

    // Get the specific test case
    const test_name = "BlockchainTests/ValidBlocks/bcStateTests/testOpcode_b0.json::testOpcode_bf_Cancun";
    const test_case = parsed.value.object.get(test_name) orelse return error.TestNotFound;

    // Run the test with path and name for trace generation
    try runner.runJsonTestWithPathAndName(allocator, test_case, json_path, test_name);
}

test "BlockchainTests_ValidBlocks_bcStateTests_testOpcode_b0_json__testOpcode_bf_Prague" {
    const allocator = testing.allocator;

    // Read and parse the JSON test file
    const json_path = "execution-spec-tests/fixtures/blockchain_tests/ValidBlocks/bcStateTests/testOpcode_b0.json";
    const json_content = try std.fs.cwd().readFileAlloc(allocator, json_path, 100 * 1024 * 1024);
    defer allocator.free(json_content);

    const parsed = try std.json.parseFromSlice(std.json.Value, allocator, json_content, .{});
    defer parsed.deinit();

    // Get the specific test case
    const test_name = "BlockchainTests/ValidBlocks/bcStateTests/testOpcode_b0.json::testOpcode_bf_Prague";
    const test_case = parsed.value.object.get(test_name) orelse return error.TestNotFound;

    // Run the test with path and name for trace generation
    try runner.runJsonTestWithPathAndName(allocator, test_case, json_path, test_name);
}
