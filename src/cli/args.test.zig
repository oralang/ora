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
