// ============================================================================
// CLI Argument Parsing Tests
// ============================================================================

const std = @import("std");
const testing = std.testing;
const cli = @import("args.zig");

test "emit list sets cfg and preserves following input file" {
    const args = [_][]const u8{ "--emit=cfg", "contract.ora" };
    const parsed = try cli.parseArgs(&args);

    try testing.expect(parsed.emit_cfg);
    try testing.expectEqualStrings("contract.ora", parsed.input_file.?);
}

test "emit list composes multiple outputs" {
    const args = [_][]const u8{ "--emit=cfg,ast", "contract.ora" };
    const parsed = try cli.parseArgs(&args);

    try testing.expect(parsed.emit_cfg);
    try testing.expect(parsed.emit_ast);
    try testing.expectEqualStrings("contract.ora", parsed.input_file.?);
}

test "emit list cfg:ora sets cfg mode" {
    const args = [_][]const u8{ "--emit=cfg:ora", "contract.ora" };
    const parsed = try cli.parseArgs(&args);

    try testing.expect(parsed.emit_cfg);
    try testing.expect(parsed.emit_cfg_mode != null);
    try testing.expectEqualStrings("ora", parsed.emit_cfg_mode.?);
}

test "emit list mlir:sir sets emit_mlir_sir" {
    const args = [_][]const u8{ "--emit=mlir:sir", "contract.ora" };
    const parsed = try cli.parseArgs(&args);

    try testing.expect(parsed.emit_mlir_sir);
    try testing.expectEqualStrings("contract.ora", parsed.input_file.?);
}

test "emit list mlir:both sets ora and sir MLIR flags" {
    const args = [_][]const u8{ "--emit=mlir:both", "contract.ora" };
    const parsed = try cli.parseArgs(&args);

    try testing.expect(parsed.emit_mlir);
    try testing.expect(parsed.emit_mlir_sir);
    try testing.expectEqualStrings("contract.ora", parsed.input_file.?);
}

test "emit list sir-text sets emit_sir_text" {
    const args = [_][]const u8{ "--emit=sir-text", "contract.ora" };
    const parsed = try cli.parseArgs(&args);

    try testing.expect(parsed.emit_sir_text);
    try testing.expectEqualStrings("contract.ora", parsed.input_file.?);
}

test "emit list bytecode sets emit_bytecode" {
    const args = [_][]const u8{ "--emit=bytecode", "contract.ora" };
    const parsed = try cli.parseArgs(&args);

    try testing.expect(parsed.emit_bytecode);
    try testing.expectEqualStrings("contract.ora", parsed.input_file.?);
}

test "emit list smt-report sets emit_smt_report" {
    const args = [_][]const u8{ "--emit=smt-report", "contract.ora" };
    const parsed = try cli.parseArgs(&args);

    try testing.expect(parsed.emit_smt_report);
    try testing.expectEqualStrings("contract.ora", parsed.input_file.?);
}

test "emit list abi outputs" {
    const args = [_][]const u8{ "--emit=abi,abi:solidity,abi:extras", "contract.ora" };
    const parsed = try cli.parseArgs(&args);

    try testing.expect(parsed.emit_abi);
    try testing.expect(parsed.emit_abi_solidity);
    try testing.expect(parsed.emit_abi_extras);
    try testing.expectEqualStrings("contract.ora", parsed.input_file.?);
}

test "mlir debug list parses together" {
    const args = [_][]const u8{
        "--mlir-pass-pipeline",
        "builtin.module(canonicalize,cse)",
        "--mlir-debug=verify-each,timing,statistics,print-ir:before-after,print-ir-pass:canonicalize,crash-reproducer:tmp/reproducer.mlir,print-op-on-diagnostic",
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

test "chain id parses as separate value and equals form" {
    const separate_args = [_][]const u8{ "--chain-id", "11155111", "contract.ora" };
    const separate = try cli.parseArgs(&separate_args);
    try testing.expectEqual(@as(?u64, 11155111), separate.chain_id);
    try testing.expectEqualStrings("contract.ora", separate.input_file.?);

    const equals_args = [_][]const u8{ "--chain-id=31337", "contract.ora" };
    const equals = try cli.parseArgs(&equals_args);
    try testing.expectEqual(@as(?u64, 31337), equals.chain_id);
    try testing.expectEqualStrings("contract.ora", equals.input_file.?);
}
