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

test "emit-cfg=ora sets cfg mode" {
    const args = [_][]const u8{ "--emit-cfg=ora", "contract.ora" };
    const parsed = try cli.parseArgs(&args);

    try testing.expect(parsed.emit_cfg);
    try testing.expect(parsed.emit_cfg_mode != null);
    try testing.expectEqualStrings("ora", parsed.emit_cfg_mode.?);
}

test "emit-mlir=sir sets emit_mlir_sir" {
    const args = [_][]const u8{ "--emit-mlir=sir", "contract.ora" };
    const parsed = try cli.parseArgs(&args);

    try testing.expect(parsed.emit_mlir_sir);
    try testing.expectEqualStrings("contract.ora", parsed.input_file.?);
}

test "emit-mlir=both sets ora and sir MLIR flags" {
    const args = [_][]const u8{ "--emit-mlir=both", "contract.ora" };
    const parsed = try cli.parseArgs(&args);

    try testing.expect(parsed.emit_mlir);
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

test "mlir advanced flags parse together" {
    const args = [_][]const u8{
        "--mlir-pass-pipeline",
        "builtin.module(canonicalize,cse)",
        "--mlir-verify-each-pass",
        "--mlir-pass-timing",
        "--mlir-pass-statistics",
        "--mlir-print-ir=before-after",
        "--mlir-print-ir-pass",
        "canonicalize",
        "--mlir-crash-reproducer",
        "tmp/reproducer.mlir",
        "--mlir-print-op-on-diagnostic",
        "contract.ora",
    };
    const parsed = try cli.parseArgs(&args);

    try testing.expect(parsed.mlir_pass_pipeline != null);
    try testing.expectEqualStrings("builtin.module(canonicalize,cse)", parsed.mlir_pass_pipeline.?);
    try testing.expect(parsed.mlir_verify_each_pass);
    try testing.expect(parsed.mlir_pass_timing);
    try testing.expect(parsed.mlir_pass_statistics);
    try testing.expect(parsed.mlir_print_ir != null);
    try testing.expectEqualStrings("before-after", parsed.mlir_print_ir.?);
    try testing.expect(parsed.mlir_print_ir_pass != null);
    try testing.expectEqualStrings("canonicalize", parsed.mlir_print_ir_pass.?);
    try testing.expect(parsed.mlir_crash_reproducer != null);
    try testing.expectEqualStrings("tmp/reproducer.mlir", parsed.mlir_crash_reproducer.?);
    try testing.expect(parsed.mlir_print_op_on_diagnostic);
    try testing.expectEqualStrings("contract.ora", parsed.input_file.?);
}
