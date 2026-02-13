// ============================================================================
// CLI Argument Parsing Tests
// ============================================================================

const std = @import("std");
const testing = std.testing;
const cli = @import("args.zig");

test "emit-cfg consumes one arg and preserves following input file" {
    const args = [_][]const u8{ "--emit-cfg", "contract.ora" };
    const parsed = try cli.parseArgs(&args);

    try testing.expect(parsed.emit_cfg);
    try testing.expectEqualStrings("contract.ora", parsed.input_file.?);
}

test "emit-cfg does not skip subsequent flags" {
    const args = [_][]const u8{ "--emit-cfg", "--emit-ast", "contract.ora" };
    const parsed = try cli.parseArgs(&args);

    try testing.expect(parsed.emit_cfg);
    try testing.expect(parsed.emit_ast);
    try testing.expectEqualStrings("contract.ora", parsed.input_file.?);
}

test "emit-sir sets emit_mlir_sir" {
    const args = [_][]const u8{ "--emit-sir", "contract.ora" };
    const parsed = try cli.parseArgs(&args);

    try testing.expect(parsed.emit_mlir_sir);
    try testing.expectEqualStrings("contract.ora", parsed.input_file.?);
}

test "emit-sir-text sets emit_sir_text" {
    const args = [_][]const u8{ "--emit-sir-text", "contract.ora" };
    const parsed = try cli.parseArgs(&args);

    try testing.expect(parsed.emit_sir_text);
    try testing.expectEqualStrings("contract.ora", parsed.input_file.?);
}

test "emit-bytecode sets emit_bytecode" {
    const args = [_][]const u8{ "--emit-bytecode", "contract.ora" };
    const parsed = try cli.parseArgs(&args);

    try testing.expect(parsed.emit_bytecode);
    try testing.expectEqualStrings("contract.ora", parsed.input_file.?);
}

test "emit-abi-extras sets emit_abi_extras" {
    const args = [_][]const u8{ "--emit-abi-extras", "contract.ora" };
    const parsed = try cli.parseArgs(&args);

    try testing.expect(parsed.emit_abi_extras);
    try testing.expectEqualStrings("contract.ora", parsed.input_file.?);
}

test "emit-smt-report sets emit_smt_report" {
    const args = [_][]const u8{ "--emit-smt-report", "contract.ora" };
    const parsed = try cli.parseArgs(&args);

    try testing.expect(parsed.emit_smt_report);
    try testing.expectEqualStrings("contract.ora", parsed.input_file.?);
}
