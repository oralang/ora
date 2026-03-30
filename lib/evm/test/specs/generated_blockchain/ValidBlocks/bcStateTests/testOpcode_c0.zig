const std = @import("std");
const testing = std.testing;
const root = @import("../../../root.zig");
const runner = root.runner;

test "BlockchainTests_ValidBlocks_bcStateTests_testOpcode_c0_json__testOpcode_c0_Cancun" {
    const allocator = testing.allocator;

    // Read and parse the JSON test file
    const json_path = "execution-spec-tests/fixtures/blockchain_tests/ValidBlocks/bcStateTests/testOpcode_c0.json";
    const json_content = try std.fs.cwd().readFileAlloc(allocator, json_path, 100 * 1024 * 1024);
    defer allocator.free(json_content);

    const parsed = try std.json.parseFromSlice(std.json.Value, allocator, json_content, .{});
    defer parsed.deinit();

    // Get the specific test case
    const test_name = "BlockchainTests/ValidBlocks/bcStateTests/testOpcode_c0.json::testOpcode_c0_Cancun";
    const test_case = parsed.value.object.get(test_name) orelse return error.TestNotFound;

    // Run the test with path and name for trace generation
    try runner.runJsonTestWithPathAndName(allocator, test_case, json_path, test_name);
}

test "BlockchainTests_ValidBlocks_bcStateTests_testOpcode_c0_json__testOpcode_c0_Prague" {
    const allocator = testing.allocator;

    // Read and parse the JSON test file
    const json_path = "execution-spec-tests/fixtures/blockchain_tests/ValidBlocks/bcStateTests/testOpcode_c0.json";
    const json_content = try std.fs.cwd().readFileAlloc(allocator, json_path, 100 * 1024 * 1024);
    defer allocator.free(json_content);

    const parsed = try std.json.parseFromSlice(std.json.Value, allocator, json_content, .{});
    defer parsed.deinit();

    // Get the specific test case
    const test_name = "BlockchainTests/ValidBlocks/bcStateTests/testOpcode_c0.json::testOpcode_c0_Prague";
    const test_case = parsed.value.object.get(test_name) orelse return error.TestNotFound;

    // Run the test with path and name for trace generation
    try runner.runJsonTestWithPathAndName(allocator, test_case, json_path, test_name);
}

test "BlockchainTests_ValidBlocks_bcStateTests_testOpcode_c0_json__testOpcode_c1_Cancun" {
    const allocator = testing.allocator;

    // Read and parse the JSON test file
    const json_path = "execution-spec-tests/fixtures/blockchain_tests/ValidBlocks/bcStateTests/testOpcode_c0.json";
    const json_content = try std.fs.cwd().readFileAlloc(allocator, json_path, 100 * 1024 * 1024);
    defer allocator.free(json_content);

    const parsed = try std.json.parseFromSlice(std.json.Value, allocator, json_content, .{});
    defer parsed.deinit();

    // Get the specific test case
    const test_name = "BlockchainTests/ValidBlocks/bcStateTests/testOpcode_c0.json::testOpcode_c1_Cancun";
    const test_case = parsed.value.object.get(test_name) orelse return error.TestNotFound;

    // Run the test with path and name for trace generation
    try runner.runJsonTestWithPathAndName(allocator, test_case, json_path, test_name);
}

test "BlockchainTests_ValidBlocks_bcStateTests_testOpcode_c0_json__testOpcode_c1_Prague" {
    const allocator = testing.allocator;

    // Read and parse the JSON test file
    const json_path = "execution-spec-tests/fixtures/blockchain_tests/ValidBlocks/bcStateTests/testOpcode_c0.json";
    const json_content = try std.fs.cwd().readFileAlloc(allocator, json_path, 100 * 1024 * 1024);
    defer allocator.free(json_content);

    const parsed = try std.json.parseFromSlice(std.json.Value, allocator, json_content, .{});
    defer parsed.deinit();

    // Get the specific test case
    const test_name = "BlockchainTests/ValidBlocks/bcStateTests/testOpcode_c0.json::testOpcode_c1_Prague";
    const test_case = parsed.value.object.get(test_name) orelse return error.TestNotFound;

    // Run the test with path and name for trace generation
    try runner.runJsonTestWithPathAndName(allocator, test_case, json_path, test_name);
}

test "BlockchainTests_ValidBlocks_bcStateTests_testOpcode_c0_json__testOpcode_c2_Cancun" {
    const allocator = testing.allocator;

    // Read and parse the JSON test file
    const json_path = "execution-spec-tests/fixtures/blockchain_tests/ValidBlocks/bcStateTests/testOpcode_c0.json";
    const json_content = try std.fs.cwd().readFileAlloc(allocator, json_path, 100 * 1024 * 1024);
    defer allocator.free(json_content);

    const parsed = try std.json.parseFromSlice(std.json.Value, allocator, json_content, .{});
    defer parsed.deinit();

    // Get the specific test case
    const test_name = "BlockchainTests/ValidBlocks/bcStateTests/testOpcode_c0.json::testOpcode_c2_Cancun";
    const test_case = parsed.value.object.get(test_name) orelse return error.TestNotFound;

    // Run the test with path and name for trace generation
    try runner.runJsonTestWithPathAndName(allocator, test_case, json_path, test_name);
}

test "BlockchainTests_ValidBlocks_bcStateTests_testOpcode_c0_json__testOpcode_c2_Prague" {
    const allocator = testing.allocator;

    // Read and parse the JSON test file
    const json_path = "execution-spec-tests/fixtures/blockchain_tests/ValidBlocks/bcStateTests/testOpcode_c0.json";
    const json_content = try std.fs.cwd().readFileAlloc(allocator, json_path, 100 * 1024 * 1024);
    defer allocator.free(json_content);

    const parsed = try std.json.parseFromSlice(std.json.Value, allocator, json_content, .{});
    defer parsed.deinit();

    // Get the specific test case
    const test_name = "BlockchainTests/ValidBlocks/bcStateTests/testOpcode_c0.json::testOpcode_c2_Prague";
    const test_case = parsed.value.object.get(test_name) orelse return error.TestNotFound;

    // Run the test with path and name for trace generation
    try runner.runJsonTestWithPathAndName(allocator, test_case, json_path, test_name);
}

test "BlockchainTests_ValidBlocks_bcStateTests_testOpcode_c0_json__testOpcode_c3_Cancun" {
    const allocator = testing.allocator;

    // Read and parse the JSON test file
    const json_path = "execution-spec-tests/fixtures/blockchain_tests/ValidBlocks/bcStateTests/testOpcode_c0.json";
    const json_content = try std.fs.cwd().readFileAlloc(allocator, json_path, 100 * 1024 * 1024);
    defer allocator.free(json_content);

    const parsed = try std.json.parseFromSlice(std.json.Value, allocator, json_content, .{});
    defer parsed.deinit();

    // Get the specific test case
    const test_name = "BlockchainTests/ValidBlocks/bcStateTests/testOpcode_c0.json::testOpcode_c3_Cancun";
    const test_case = parsed.value.object.get(test_name) orelse return error.TestNotFound;

    // Run the test with path and name for trace generation
    try runner.runJsonTestWithPathAndName(allocator, test_case, json_path, test_name);
}

test "BlockchainTests_ValidBlocks_bcStateTests_testOpcode_c0_json__testOpcode_c3_Prague" {
    const allocator = testing.allocator;

    // Read and parse the JSON test file
    const json_path = "execution-spec-tests/fixtures/blockchain_tests/ValidBlocks/bcStateTests/testOpcode_c0.json";
    const json_content = try std.fs.cwd().readFileAlloc(allocator, json_path, 100 * 1024 * 1024);
    defer allocator.free(json_content);

    const parsed = try std.json.parseFromSlice(std.json.Value, allocator, json_content, .{});
    defer parsed.deinit();

    // Get the specific test case
    const test_name = "BlockchainTests/ValidBlocks/bcStateTests/testOpcode_c0.json::testOpcode_c3_Prague";
    const test_case = parsed.value.object.get(test_name) orelse return error.TestNotFound;

    // Run the test with path and name for trace generation
    try runner.runJsonTestWithPathAndName(allocator, test_case, json_path, test_name);
}

test "BlockchainTests_ValidBlocks_bcStateTests_testOpcode_c0_json__testOpcode_c4_Cancun" {
    const allocator = testing.allocator;

    // Read and parse the JSON test file
    const json_path = "execution-spec-tests/fixtures/blockchain_tests/ValidBlocks/bcStateTests/testOpcode_c0.json";
    const json_content = try std.fs.cwd().readFileAlloc(allocator, json_path, 100 * 1024 * 1024);
    defer allocator.free(json_content);

    const parsed = try std.json.parseFromSlice(std.json.Value, allocator, json_content, .{});
    defer parsed.deinit();

    // Get the specific test case
    const test_name = "BlockchainTests/ValidBlocks/bcStateTests/testOpcode_c0.json::testOpcode_c4_Cancun";
    const test_case = parsed.value.object.get(test_name) orelse return error.TestNotFound;

    // Run the test with path and name for trace generation
    try runner.runJsonTestWithPathAndName(allocator, test_case, json_path, test_name);
}

test "BlockchainTests_ValidBlocks_bcStateTests_testOpcode_c0_json__testOpcode_c4_Prague" {
    const allocator = testing.allocator;

    // Read and parse the JSON test file
    const json_path = "execution-spec-tests/fixtures/blockchain_tests/ValidBlocks/bcStateTests/testOpcode_c0.json";
    const json_content = try std.fs.cwd().readFileAlloc(allocator, json_path, 100 * 1024 * 1024);
    defer allocator.free(json_content);

    const parsed = try std.json.parseFromSlice(std.json.Value, allocator, json_content, .{});
    defer parsed.deinit();

    // Get the specific test case
    const test_name = "BlockchainTests/ValidBlocks/bcStateTests/testOpcode_c0.json::testOpcode_c4_Prague";
    const test_case = parsed.value.object.get(test_name) orelse return error.TestNotFound;

    // Run the test with path and name for trace generation
    try runner.runJsonTestWithPathAndName(allocator, test_case, json_path, test_name);
}

test "BlockchainTests_ValidBlocks_bcStateTests_testOpcode_c0_json__testOpcode_c5_Cancun" {
    const allocator = testing.allocator;

    // Read and parse the JSON test file
    const json_path = "execution-spec-tests/fixtures/blockchain_tests/ValidBlocks/bcStateTests/testOpcode_c0.json";
    const json_content = try std.fs.cwd().readFileAlloc(allocator, json_path, 100 * 1024 * 1024);
    defer allocator.free(json_content);

    const parsed = try std.json.parseFromSlice(std.json.Value, allocator, json_content, .{});
    defer parsed.deinit();

    // Get the specific test case
    const test_name = "BlockchainTests/ValidBlocks/bcStateTests/testOpcode_c0.json::testOpcode_c5_Cancun";
    const test_case = parsed.value.object.get(test_name) orelse return error.TestNotFound;

    // Run the test with path and name for trace generation
    try runner.runJsonTestWithPathAndName(allocator, test_case, json_path, test_name);
}

test "BlockchainTests_ValidBlocks_bcStateTests_testOpcode_c0_json__testOpcode_c5_Prague" {
    const allocator = testing.allocator;

    // Read and parse the JSON test file
    const json_path = "execution-spec-tests/fixtures/blockchain_tests/ValidBlocks/bcStateTests/testOpcode_c0.json";
    const json_content = try std.fs.cwd().readFileAlloc(allocator, json_path, 100 * 1024 * 1024);
    defer allocator.free(json_content);

    const parsed = try std.json.parseFromSlice(std.json.Value, allocator, json_content, .{});
    defer parsed.deinit();

    // Get the specific test case
    const test_name = "BlockchainTests/ValidBlocks/bcStateTests/testOpcode_c0.json::testOpcode_c5_Prague";
    const test_case = parsed.value.object.get(test_name) orelse return error.TestNotFound;

    // Run the test with path and name for trace generation
    try runner.runJsonTestWithPathAndName(allocator, test_case, json_path, test_name);
}

test "BlockchainTests_ValidBlocks_bcStateTests_testOpcode_c0_json__testOpcode_c6_Cancun" {
    const allocator = testing.allocator;

    // Read and parse the JSON test file
    const json_path = "execution-spec-tests/fixtures/blockchain_tests/ValidBlocks/bcStateTests/testOpcode_c0.json";
    const json_content = try std.fs.cwd().readFileAlloc(allocator, json_path, 100 * 1024 * 1024);
    defer allocator.free(json_content);

    const parsed = try std.json.parseFromSlice(std.json.Value, allocator, json_content, .{});
    defer parsed.deinit();

    // Get the specific test case
    const test_name = "BlockchainTests/ValidBlocks/bcStateTests/testOpcode_c0.json::testOpcode_c6_Cancun";
    const test_case = parsed.value.object.get(test_name) orelse return error.TestNotFound;

    // Run the test with path and name for trace generation
    try runner.runJsonTestWithPathAndName(allocator, test_case, json_path, test_name);
}

test "BlockchainTests_ValidBlocks_bcStateTests_testOpcode_c0_json__testOpcode_c6_Prague" {
    const allocator = testing.allocator;

    // Read and parse the JSON test file
    const json_path = "execution-spec-tests/fixtures/blockchain_tests/ValidBlocks/bcStateTests/testOpcode_c0.json";
    const json_content = try std.fs.cwd().readFileAlloc(allocator, json_path, 100 * 1024 * 1024);
    defer allocator.free(json_content);

    const parsed = try std.json.parseFromSlice(std.json.Value, allocator, json_content, .{});
    defer parsed.deinit();

    // Get the specific test case
    const test_name = "BlockchainTests/ValidBlocks/bcStateTests/testOpcode_c0.json::testOpcode_c6_Prague";
    const test_case = parsed.value.object.get(test_name) orelse return error.TestNotFound;

    // Run the test with path and name for trace generation
    try runner.runJsonTestWithPathAndName(allocator, test_case, json_path, test_name);
}

test "BlockchainTests_ValidBlocks_bcStateTests_testOpcode_c0_json__testOpcode_c7_Cancun" {
    const allocator = testing.allocator;

    // Read and parse the JSON test file
    const json_path = "execution-spec-tests/fixtures/blockchain_tests/ValidBlocks/bcStateTests/testOpcode_c0.json";
    const json_content = try std.fs.cwd().readFileAlloc(allocator, json_path, 100 * 1024 * 1024);
    defer allocator.free(json_content);

    const parsed = try std.json.parseFromSlice(std.json.Value, allocator, json_content, .{});
    defer parsed.deinit();

    // Get the specific test case
    const test_name = "BlockchainTests/ValidBlocks/bcStateTests/testOpcode_c0.json::testOpcode_c7_Cancun";
    const test_case = parsed.value.object.get(test_name) orelse return error.TestNotFound;

    // Run the test with path and name for trace generation
    try runner.runJsonTestWithPathAndName(allocator, test_case, json_path, test_name);
}

test "BlockchainTests_ValidBlocks_bcStateTests_testOpcode_c0_json__testOpcode_c7_Prague" {
    const allocator = testing.allocator;

    // Read and parse the JSON test file
    const json_path = "execution-spec-tests/fixtures/blockchain_tests/ValidBlocks/bcStateTests/testOpcode_c0.json";
    const json_content = try std.fs.cwd().readFileAlloc(allocator, json_path, 100 * 1024 * 1024);
    defer allocator.free(json_content);

    const parsed = try std.json.parseFromSlice(std.json.Value, allocator, json_content, .{});
    defer parsed.deinit();

    // Get the specific test case
    const test_name = "BlockchainTests/ValidBlocks/bcStateTests/testOpcode_c0.json::testOpcode_c7_Prague";
    const test_case = parsed.value.object.get(test_name) orelse return error.TestNotFound;

    // Run the test with path and name for trace generation
    try runner.runJsonTestWithPathAndName(allocator, test_case, json_path, test_name);
}

test "BlockchainTests_ValidBlocks_bcStateTests_testOpcode_c0_json__testOpcode_c8_Cancun" {
    const allocator = testing.allocator;

    // Read and parse the JSON test file
    const json_path = "execution-spec-tests/fixtures/blockchain_tests/ValidBlocks/bcStateTests/testOpcode_c0.json";
    const json_content = try std.fs.cwd().readFileAlloc(allocator, json_path, 100 * 1024 * 1024);
    defer allocator.free(json_content);

    const parsed = try std.json.parseFromSlice(std.json.Value, allocator, json_content, .{});
    defer parsed.deinit();

    // Get the specific test case
    const test_name = "BlockchainTests/ValidBlocks/bcStateTests/testOpcode_c0.json::testOpcode_c8_Cancun";
    const test_case = parsed.value.object.get(test_name) orelse return error.TestNotFound;

    // Run the test with path and name for trace generation
    try runner.runJsonTestWithPathAndName(allocator, test_case, json_path, test_name);
}

test "BlockchainTests_ValidBlocks_bcStateTests_testOpcode_c0_json__testOpcode_c8_Prague" {
    const allocator = testing.allocator;

    // Read and parse the JSON test file
    const json_path = "execution-spec-tests/fixtures/blockchain_tests/ValidBlocks/bcStateTests/testOpcode_c0.json";
    const json_content = try std.fs.cwd().readFileAlloc(allocator, json_path, 100 * 1024 * 1024);
    defer allocator.free(json_content);

    const parsed = try std.json.parseFromSlice(std.json.Value, allocator, json_content, .{});
    defer parsed.deinit();

    // Get the specific test case
    const test_name = "BlockchainTests/ValidBlocks/bcStateTests/testOpcode_c0.json::testOpcode_c8_Prague";
    const test_case = parsed.value.object.get(test_name) orelse return error.TestNotFound;

    // Run the test with path and name for trace generation
    try runner.runJsonTestWithPathAndName(allocator, test_case, json_path, test_name);
}

test "BlockchainTests_ValidBlocks_bcStateTests_testOpcode_c0_json__testOpcode_c9_Cancun" {
    const allocator = testing.allocator;

    // Read and parse the JSON test file
    const json_path = "execution-spec-tests/fixtures/blockchain_tests/ValidBlocks/bcStateTests/testOpcode_c0.json";
    const json_content = try std.fs.cwd().readFileAlloc(allocator, json_path, 100 * 1024 * 1024);
    defer allocator.free(json_content);

    const parsed = try std.json.parseFromSlice(std.json.Value, allocator, json_content, .{});
    defer parsed.deinit();

    // Get the specific test case
    const test_name = "BlockchainTests/ValidBlocks/bcStateTests/testOpcode_c0.json::testOpcode_c9_Cancun";
    const test_case = parsed.value.object.get(test_name) orelse return error.TestNotFound;

    // Run the test with path and name for trace generation
    try runner.runJsonTestWithPathAndName(allocator, test_case, json_path, test_name);
}

test "BlockchainTests_ValidBlocks_bcStateTests_testOpcode_c0_json__testOpcode_c9_Prague" {
    const allocator = testing.allocator;

    // Read and parse the JSON test file
    const json_path = "execution-spec-tests/fixtures/blockchain_tests/ValidBlocks/bcStateTests/testOpcode_c0.json";
    const json_content = try std.fs.cwd().readFileAlloc(allocator, json_path, 100 * 1024 * 1024);
    defer allocator.free(json_content);

    const parsed = try std.json.parseFromSlice(std.json.Value, allocator, json_content, .{});
    defer parsed.deinit();

    // Get the specific test case
    const test_name = "BlockchainTests/ValidBlocks/bcStateTests/testOpcode_c0.json::testOpcode_c9_Prague";
    const test_case = parsed.value.object.get(test_name) orelse return error.TestNotFound;

    // Run the test with path and name for trace generation
    try runner.runJsonTestWithPathAndName(allocator, test_case, json_path, test_name);
}

test "BlockchainTests_ValidBlocks_bcStateTests_testOpcode_c0_json__testOpcode_ca_Cancun" {
    const allocator = testing.allocator;

    // Read and parse the JSON test file
    const json_path = "execution-spec-tests/fixtures/blockchain_tests/ValidBlocks/bcStateTests/testOpcode_c0.json";
    const json_content = try std.fs.cwd().readFileAlloc(allocator, json_path, 100 * 1024 * 1024);
    defer allocator.free(json_content);

    const parsed = try std.json.parseFromSlice(std.json.Value, allocator, json_content, .{});
    defer parsed.deinit();

    // Get the specific test case
    const test_name = "BlockchainTests/ValidBlocks/bcStateTests/testOpcode_c0.json::testOpcode_ca_Cancun";
    const test_case = parsed.value.object.get(test_name) orelse return error.TestNotFound;

    // Run the test with path and name for trace generation
    try runner.runJsonTestWithPathAndName(allocator, test_case, json_path, test_name);
}

test "BlockchainTests_ValidBlocks_bcStateTests_testOpcode_c0_json__testOpcode_ca_Prague" {
    const allocator = testing.allocator;

    // Read and parse the JSON test file
    const json_path = "execution-spec-tests/fixtures/blockchain_tests/ValidBlocks/bcStateTests/testOpcode_c0.json";
    const json_content = try std.fs.cwd().readFileAlloc(allocator, json_path, 100 * 1024 * 1024);
    defer allocator.free(json_content);

    const parsed = try std.json.parseFromSlice(std.json.Value, allocator, json_content, .{});
    defer parsed.deinit();

    // Get the specific test case
    const test_name = "BlockchainTests/ValidBlocks/bcStateTests/testOpcode_c0.json::testOpcode_ca_Prague";
    const test_case = parsed.value.object.get(test_name) orelse return error.TestNotFound;

    // Run the test with path and name for trace generation
    try runner.runJsonTestWithPathAndName(allocator, test_case, json_path, test_name);
}

test "BlockchainTests_ValidBlocks_bcStateTests_testOpcode_c0_json__testOpcode_cb_Cancun" {
    const allocator = testing.allocator;

    // Read and parse the JSON test file
    const json_path = "execution-spec-tests/fixtures/blockchain_tests/ValidBlocks/bcStateTests/testOpcode_c0.json";
    const json_content = try std.fs.cwd().readFileAlloc(allocator, json_path, 100 * 1024 * 1024);
    defer allocator.free(json_content);

    const parsed = try std.json.parseFromSlice(std.json.Value, allocator, json_content, .{});
    defer parsed.deinit();

    // Get the specific test case
    const test_name = "BlockchainTests/ValidBlocks/bcStateTests/testOpcode_c0.json::testOpcode_cb_Cancun";
    const test_case = parsed.value.object.get(test_name) orelse return error.TestNotFound;

    // Run the test with path and name for trace generation
    try runner.runJsonTestWithPathAndName(allocator, test_case, json_path, test_name);
}

test "BlockchainTests_ValidBlocks_bcStateTests_testOpcode_c0_json__testOpcode_cb_Prague" {
    const allocator = testing.allocator;

    // Read and parse the JSON test file
    const json_path = "execution-spec-tests/fixtures/blockchain_tests/ValidBlocks/bcStateTests/testOpcode_c0.json";
    const json_content = try std.fs.cwd().readFileAlloc(allocator, json_path, 100 * 1024 * 1024);
    defer allocator.free(json_content);

    const parsed = try std.json.parseFromSlice(std.json.Value, allocator, json_content, .{});
    defer parsed.deinit();

    // Get the specific test case
    const test_name = "BlockchainTests/ValidBlocks/bcStateTests/testOpcode_c0.json::testOpcode_cb_Prague";
    const test_case = parsed.value.object.get(test_name) orelse return error.TestNotFound;

    // Run the test with path and name for trace generation
    try runner.runJsonTestWithPathAndName(allocator, test_case, json_path, test_name);
}

test "BlockchainTests_ValidBlocks_bcStateTests_testOpcode_c0_json__testOpcode_cc_Cancun" {
    const allocator = testing.allocator;

    // Read and parse the JSON test file
    const json_path = "execution-spec-tests/fixtures/blockchain_tests/ValidBlocks/bcStateTests/testOpcode_c0.json";
    const json_content = try std.fs.cwd().readFileAlloc(allocator, json_path, 100 * 1024 * 1024);
    defer allocator.free(json_content);

    const parsed = try std.json.parseFromSlice(std.json.Value, allocator, json_content, .{});
    defer parsed.deinit();

    // Get the specific test case
    const test_name = "BlockchainTests/ValidBlocks/bcStateTests/testOpcode_c0.json::testOpcode_cc_Cancun";
    const test_case = parsed.value.object.get(test_name) orelse return error.TestNotFound;

    // Run the test with path and name for trace generation
    try runner.runJsonTestWithPathAndName(allocator, test_case, json_path, test_name);
}

test "BlockchainTests_ValidBlocks_bcStateTests_testOpcode_c0_json__testOpcode_cc_Prague" {
    const allocator = testing.allocator;

    // Read and parse the JSON test file
    const json_path = "execution-spec-tests/fixtures/blockchain_tests/ValidBlocks/bcStateTests/testOpcode_c0.json";
    const json_content = try std.fs.cwd().readFileAlloc(allocator, json_path, 100 * 1024 * 1024);
    defer allocator.free(json_content);

    const parsed = try std.json.parseFromSlice(std.json.Value, allocator, json_content, .{});
    defer parsed.deinit();

    // Get the specific test case
    const test_name = "BlockchainTests/ValidBlocks/bcStateTests/testOpcode_c0.json::testOpcode_cc_Prague";
    const test_case = parsed.value.object.get(test_name) orelse return error.TestNotFound;

    // Run the test with path and name for trace generation
    try runner.runJsonTestWithPathAndName(allocator, test_case, json_path, test_name);
}

test "BlockchainTests_ValidBlocks_bcStateTests_testOpcode_c0_json__testOpcode_cd_Cancun" {
    const allocator = testing.allocator;

    // Read and parse the JSON test file
    const json_path = "execution-spec-tests/fixtures/blockchain_tests/ValidBlocks/bcStateTests/testOpcode_c0.json";
    const json_content = try std.fs.cwd().readFileAlloc(allocator, json_path, 100 * 1024 * 1024);
    defer allocator.free(json_content);

    const parsed = try std.json.parseFromSlice(std.json.Value, allocator, json_content, .{});
    defer parsed.deinit();

    // Get the specific test case
    const test_name = "BlockchainTests/ValidBlocks/bcStateTests/testOpcode_c0.json::testOpcode_cd_Cancun";
    const test_case = parsed.value.object.get(test_name) orelse return error.TestNotFound;

    // Run the test with path and name for trace generation
    try runner.runJsonTestWithPathAndName(allocator, test_case, json_path, test_name);
}

test "BlockchainTests_ValidBlocks_bcStateTests_testOpcode_c0_json__testOpcode_cd_Prague" {
    const allocator = testing.allocator;

    // Read and parse the JSON test file
    const json_path = "execution-spec-tests/fixtures/blockchain_tests/ValidBlocks/bcStateTests/testOpcode_c0.json";
    const json_content = try std.fs.cwd().readFileAlloc(allocator, json_path, 100 * 1024 * 1024);
    defer allocator.free(json_content);

    const parsed = try std.json.parseFromSlice(std.json.Value, allocator, json_content, .{});
    defer parsed.deinit();

    // Get the specific test case
    const test_name = "BlockchainTests/ValidBlocks/bcStateTests/testOpcode_c0.json::testOpcode_cd_Prague";
    const test_case = parsed.value.object.get(test_name) orelse return error.TestNotFound;

    // Run the test with path and name for trace generation
    try runner.runJsonTestWithPathAndName(allocator, test_case, json_path, test_name);
}

test "BlockchainTests_ValidBlocks_bcStateTests_testOpcode_c0_json__testOpcode_ce_Cancun" {
    const allocator = testing.allocator;

    // Read and parse the JSON test file
    const json_path = "execution-spec-tests/fixtures/blockchain_tests/ValidBlocks/bcStateTests/testOpcode_c0.json";
    const json_content = try std.fs.cwd().readFileAlloc(allocator, json_path, 100 * 1024 * 1024);
    defer allocator.free(json_content);

    const parsed = try std.json.parseFromSlice(std.json.Value, allocator, json_content, .{});
    defer parsed.deinit();

    // Get the specific test case
    const test_name = "BlockchainTests/ValidBlocks/bcStateTests/testOpcode_c0.json::testOpcode_ce_Cancun";
    const test_case = parsed.value.object.get(test_name) orelse return error.TestNotFound;

    // Run the test with path and name for trace generation
    try runner.runJsonTestWithPathAndName(allocator, test_case, json_path, test_name);
}

test "BlockchainTests_ValidBlocks_bcStateTests_testOpcode_c0_json__testOpcode_ce_Prague" {
    const allocator = testing.allocator;

    // Read and parse the JSON test file
    const json_path = "execution-spec-tests/fixtures/blockchain_tests/ValidBlocks/bcStateTests/testOpcode_c0.json";
    const json_content = try std.fs.cwd().readFileAlloc(allocator, json_path, 100 * 1024 * 1024);
    defer allocator.free(json_content);

    const parsed = try std.json.parseFromSlice(std.json.Value, allocator, json_content, .{});
    defer parsed.deinit();

    // Get the specific test case
    const test_name = "BlockchainTests/ValidBlocks/bcStateTests/testOpcode_c0.json::testOpcode_ce_Prague";
    const test_case = parsed.value.object.get(test_name) orelse return error.TestNotFound;

    // Run the test with path and name for trace generation
    try runner.runJsonTestWithPathAndName(allocator, test_case, json_path, test_name);
}

test "BlockchainTests_ValidBlocks_bcStateTests_testOpcode_c0_json__testOpcode_cf_Cancun" {
    const allocator = testing.allocator;

    // Read and parse the JSON test file
    const json_path = "execution-spec-tests/fixtures/blockchain_tests/ValidBlocks/bcStateTests/testOpcode_c0.json";
    const json_content = try std.fs.cwd().readFileAlloc(allocator, json_path, 100 * 1024 * 1024);
    defer allocator.free(json_content);

    const parsed = try std.json.parseFromSlice(std.json.Value, allocator, json_content, .{});
    defer parsed.deinit();

    // Get the specific test case
    const test_name = "BlockchainTests/ValidBlocks/bcStateTests/testOpcode_c0.json::testOpcode_cf_Cancun";
    const test_case = parsed.value.object.get(test_name) orelse return error.TestNotFound;

    // Run the test with path and name for trace generation
    try runner.runJsonTestWithPathAndName(allocator, test_case, json_path, test_name);
}

test "BlockchainTests_ValidBlocks_bcStateTests_testOpcode_c0_json__testOpcode_cf_Prague" {
    const allocator = testing.allocator;

    // Read and parse the JSON test file
    const json_path = "execution-spec-tests/fixtures/blockchain_tests/ValidBlocks/bcStateTests/testOpcode_c0.json";
    const json_content = try std.fs.cwd().readFileAlloc(allocator, json_path, 100 * 1024 * 1024);
    defer allocator.free(json_content);

    const parsed = try std.json.parseFromSlice(std.json.Value, allocator, json_content, .{});
    defer parsed.deinit();

    // Get the specific test case
    const test_name = "BlockchainTests/ValidBlocks/bcStateTests/testOpcode_c0.json::testOpcode_cf_Prague";
    const test_case = parsed.value.object.get(test_name) orelse return error.TestNotFound;

    // Run the test with path and name for trace generation
    try runner.runJsonTestWithPathAndName(allocator, test_case, json_path, test_name);
}
