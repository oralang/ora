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
    try testing.expectEqualStrings("ora", parsed.emit_cfg_mode.?);
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

test "help flags parse explicitly" {
    const short_args = [_][]const u8{"-h"};
    const short = try cli.parseArgs(&short_args);
    try testing.expect(short.show_help);

    const long_args = [_][]const u8{"--help"};
    const long = try cli.parseArgs(&long_args);
    try testing.expect(long.show_help);
}

test "separator allows dash-prefixed input file" {
    const args = [_][]const u8{ "--", "-contract.ora" };
    const parsed = try cli.parseArgs(&args);

    try testing.expectEqualStrings("-contract.ora", parsed.input_file.?);
}

test "formatter width must be nonzero" {
    const args = [_][]const u8{ "fmt", "--width", "0", "contract.ora" };
    try testing.expectError(error.UnknownArgument, cli.parseArgs(&args));
}

test "bare fmt token is not positional formatter mode" {
    const args = [_][]const u8{ "contract.ora", "fmt" };
    try testing.expectError(error.UnknownArgument, cli.parseArgs(&args));
}

test "emit ast formats validate at parse time" {
    const bad_ast = [_][]const u8{"--emit=ast:xml"};
    try testing.expectError(error.UnknownArgument, cli.parseArgs(&bad_ast));

    const bad_typed_ast = [_][]const u8{"--emit=typed-ast:yaml"};
    try testing.expectError(error.UnknownArgument, cli.parseArgs(&bad_typed_ast));

    const good_ast = [_][]const u8{ "--emit=ast:json", "contract.ora" };
    const parsed = try cli.parseArgs(&good_ast);
    try testing.expect(parsed.emit_ast);
    try testing.expectEqualStrings("json", parsed.emit_ast_format.?);
}

test "output target flags are explicit and mutually exclusive" {
    const out_dir_args = [_][]const u8{ "--out-dir", "artifacts", "contract.ora" };
    const out_dir = try cli.parseArgs(&out_dir_args);
    try testing.expectEqualStrings("artifacts", out_dir.output_dir.?);

    const out_file_args = [_][]const u8{ "--out-file", "contract.hex", "contract.ora" };
    const out_file = try cli.parseArgs(&out_file_args);
    try testing.expectEqualStrings("contract.hex", out_file.output_file.?);

    const duplicate = [_][]const u8{ "--out-dir", "artifacts", "--out-file", "contract.hex", "contract.ora" };
    try testing.expectError(error.DuplicateArgument, cli.parseArgs(&duplicate));
}

test "duplicate single-owner flags are rejected" {
    const width_args = [_][]const u8{ "fmt", "--width", "80", "--width", "100", "contract.ora" };
    try testing.expectError(error.DuplicateArgument, cli.parseArgs(&width_args));

    const emit_args = [_][]const u8{ "--emit=mlir", "--emit=bytecode", "contract.ora" };
    try testing.expectError(error.DuplicateArgument, cli.parseArgs(&emit_args));

    const verify_args = [_][]const u8{ "--verify", "--no-verify", "contract.ora" };
    try testing.expectError(error.DuplicateArgument, cli.parseArgs(&verify_args));

    const chain_args = [_][]const u8{ "--chain-id=1", "--chain-id=2", "contract.ora" };
    try testing.expectError(error.DuplicateArgument, cli.parseArgs(&chain_args));
}
