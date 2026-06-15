// ============================================================================
// CLI Argument Parsing
// ============================================================================

const std = @import("std");

pub const CliOptions = struct {
    output_dir: ?[]const u8 = null,
    output_file: ?[]const u8 = null,
    input_file: ?[]const u8 = null,
    emit_tokens: bool = false,
    emit_ast: bool = false,
    emit_ast_format: ?[]const u8 = null,
    emit_typed_ast: bool = false,
    emit_typed_ast_format: ?[]const u8 = null,
    emit_mlir: bool = false,
    emit_mlir_sir: bool = false,
    emit_sir_text: bool = false,
    emit_bytecode: bool = false,
    emit_cfg: bool = false,
    emit_cfg_mode: ?[]const u8 = null,
    emit_abi: bool = false,
    emit_abi_solidity: bool = false,
    emit_abi_extras: bool = false,
    cpp_lowering_stub: bool = false,
    canonicalize_mlir: bool = true,
    validate_mlir: bool = true,
    verify_z3: bool = true,
    verify_requested: bool = false,
    verify_mode: ?[]const u8 = null,
    verify_stats: bool = false,
    explain_cores: bool = false,
    z3_proofs: bool = false,
    minimize_cores: bool = false,
    keep_proved_checks: bool = false,
    emit_smt_report: bool = false,
    mlir_pass_pipeline: ?[]const u8 = null,
    mlir_verify_each_pass: bool = false,
    mlir_pass_timing: bool = false,
    mlir_pass_statistics: bool = false,
    mlir_print_ir: ?[]const u8 = null,
    mlir_print_ir_pass: ?[]const u8 = null,
    mlir_crash_reproducer: ?[]const u8 = null,
    mlir_print_op_on_diagnostic: bool = false,
    debug: bool = false,
    debug_info: bool = false,
    mlir_opt_level: ?[]const u8 = null,
    // fmt options
    fmt: bool = false,
    fmt_check: bool = false,
    fmt_diff: bool = false,
    fmt_stdout: bool = false,
    fmt_width: ?u32 = null,
    show_help: bool = false,
    show_version: bool = false,
    metrics: bool = false,
    chain_id: ?u64 = null,
};

pub const ParseError = error{
    MissingArgument,
    UnknownArgument,
    DuplicateArgument,
};

fn claim(seen: *bool) ParseError!void {
    if (seen.*) return error.DuplicateArgument;
    seen.* = true;
}

fn validAstFormat(format: []const u8) bool {
    return std.mem.eql(u8, format, "json") or std.mem.eql(u8, format, "tree");
}

fn parseEmitList(opts: *CliOptions, spec: []const u8) ParseError!void {
    var iter = std.mem.splitScalar(u8, spec, ',');
    while (iter.next()) |raw_item| {
        const item = std.mem.trim(u8, raw_item, " \t\r\n");
        if (item.len == 0) return error.UnknownArgument;

        if (std.mem.eql(u8, item, "tokens")) {
            opts.emit_tokens = true;
        } else if (std.mem.eql(u8, item, "ast")) {
            opts.emit_ast = true;
        } else if (std.mem.startsWith(u8, item, "ast:")) {
            const format = item["ast:".len..];
            if (!validAstFormat(format)) return error.UnknownArgument;
            opts.emit_ast = true;
            opts.emit_ast_format = format;
        } else if (std.mem.eql(u8, item, "typed-ast")) {
            opts.emit_typed_ast = true;
        } else if (std.mem.startsWith(u8, item, "typed-ast:")) {
            const format = item["typed-ast:".len..];
            if (!validAstFormat(format)) return error.UnknownArgument;
            opts.emit_typed_ast = true;
            opts.emit_typed_ast_format = format;
        } else if (std.mem.eql(u8, item, "mlir") or std.mem.eql(u8, item, "mlir:ora")) {
            opts.emit_mlir = true;
        } else if (std.mem.eql(u8, item, "mlir:sir")) {
            opts.emit_mlir_sir = true;
        } else if (std.mem.eql(u8, item, "mlir:both")) {
            opts.emit_mlir = true;
            opts.emit_mlir_sir = true;
        } else if (std.mem.eql(u8, item, "sir-text")) {
            opts.emit_sir_text = true;
        } else if (std.mem.eql(u8, item, "bytecode")) {
            opts.emit_bytecode = true;
        } else if (std.mem.eql(u8, item, "cfg") or std.mem.eql(u8, item, "cfg:ora")) {
            opts.emit_cfg = true;
            opts.emit_cfg_mode = if (std.mem.eql(u8, item, "cfg:ora")) "ora" else opts.emit_cfg_mode;
        } else if (std.mem.eql(u8, item, "cfg:sir")) {
            opts.emit_cfg = true;
            opts.emit_cfg_mode = "sir";
        } else if (std.mem.eql(u8, item, "smt-report")) {
            opts.emit_smt_report = true;
        } else if (std.mem.eql(u8, item, "abi")) {
            opts.emit_abi = true;
        } else if (std.mem.eql(u8, item, "abi:solidity") or std.mem.eql(u8, item, "abi-solidity")) {
            opts.emit_abi_solidity = true;
        } else if (std.mem.eql(u8, item, "abi:extras") or std.mem.eql(u8, item, "abi-extras")) {
            opts.emit_abi_extras = true;
        } else {
            return error.UnknownArgument;
        }
    }
}

fn parseMlirDebugList(opts: *CliOptions, spec: []const u8) ParseError!void {
    var iter = std.mem.splitScalar(u8, spec, ',');
    while (iter.next()) |raw_item| {
        const item = std.mem.trim(u8, raw_item, " \t\r\n");
        if (item.len == 0) return error.UnknownArgument;

        if (std.mem.eql(u8, item, "verify-each")) {
            opts.mlir_verify_each_pass = true;
        } else if (std.mem.eql(u8, item, "timing")) {
            opts.mlir_pass_timing = true;
        } else if (std.mem.eql(u8, item, "statistics")) {
            opts.mlir_pass_statistics = true;
        } else if (std.mem.eql(u8, item, "print-op-on-diagnostic")) {
            opts.mlir_print_op_on_diagnostic = true;
        } else if (std.mem.startsWith(u8, item, "print-ir:")) {
            const mode = item["print-ir:".len..];
            if (!std.ascii.eqlIgnoreCase(mode, "before") and
                !std.ascii.eqlIgnoreCase(mode, "after") and
                !std.ascii.eqlIgnoreCase(mode, "before-after") and
                !std.ascii.eqlIgnoreCase(mode, "all"))
            {
                return error.UnknownArgument;
            }
            opts.mlir_print_ir = mode;
        } else if (std.mem.startsWith(u8, item, "print-ir-pass:")) {
            opts.mlir_print_ir_pass = item["print-ir-pass:".len..];
        } else if (std.mem.startsWith(u8, item, "crash-reproducer:")) {
            opts.mlir_crash_reproducer = item["crash-reproducer:".len..];
        } else {
            return error.UnknownArgument;
        }
    }
}

pub fn parseArgs(args: []const []const u8) ParseError!CliOptions {
    var opts = CliOptions{};
    var seen_output_target = false;
    var seen_input_file = false;
    var seen_emit = false;
    var seen_cpp_lowering_stub = false;
    var seen_verify = false;
    var seen_verify_stats = false;
    var seen_explain = false;
    var seen_z3_proofs = false;
    var seen_minimize_cores = false;
    var seen_keep_proved_checks = false;
    var seen_mlir_debug = false;
    var seen_mlir_pass_pipeline = false;
    var seen_debug = false;
    var seen_debug_info = false;
    var seen_opt_level = false;
    var seen_validate_mlir = false;
    var seen_canonicalize = false;
    var seen_chain_id = false;
    var seen_fmt_check = false;
    var seen_fmt_diff = false;
    var seen_fmt_stdout = false;
    var seen_fmt_width = false;
    var seen_metrics = false;
    var seen_version = false;
    var seen_help = false;
    var i: usize = 0;

    while (i < args.len) {
        const arg = args[i];
        if (std.mem.eql(u8, arg, "--")) {
            try claim(&seen_input_file);
            if (i + 1 >= args.len) return error.MissingArgument;
            opts.input_file = args[i + 1];
            if (i + 2 != args.len) return error.UnknownArgument;
            i = args.len;
        } else if (std.mem.eql(u8, arg, "-h") or std.mem.eql(u8, arg, "--help")) {
            try claim(&seen_help);
            opts.show_help = true;
            i += 1;
        } else if (std.mem.eql(u8, arg, "-o") or std.mem.eql(u8, arg, "--output-dir") or std.mem.eql(u8, arg, "--out-dir")) {
            try claim(&seen_output_target);
            if (i + 1 >= args.len) return error.MissingArgument;
            opts.output_dir = args[i + 1];
            i += 2;
        } else if (std.mem.eql(u8, arg, "--out-file")) {
            try claim(&seen_output_target);
            if (i + 1 >= args.len) return error.MissingArgument;
            opts.output_file = args[i + 1];
            i += 2;
        } else if (std.mem.eql(u8, arg, "--emit")) {
            try claim(&seen_emit);
            if (i + 1 >= args.len) return error.MissingArgument;
            try parseEmitList(&opts, args[i + 1]);
            i += 2;
        } else if (std.mem.startsWith(u8, arg, "--emit=")) {
            try claim(&seen_emit);
            try parseEmitList(&opts, arg["--emit=".len..]);
            i += 1;
        } else if (std.mem.eql(u8, arg, "--cpp-lowering-stub")) {
            try claim(&seen_cpp_lowering_stub);
            opts.cpp_lowering_stub = true;
            i += 1;
        } else if (std.mem.eql(u8, arg, "--verify")) {
            try claim(&seen_verify);
            opts.verify_z3 = true;
            opts.verify_requested = true;
            i += 1;
        } else if (std.mem.startsWith(u8, arg, "--verify=")) {
            try claim(&seen_verify);
            const mode = arg["--verify=".len..];
            if (!std.ascii.eqlIgnoreCase(mode, "basic") and !std.ascii.eqlIgnoreCase(mode, "full")) {
                return error.UnknownArgument;
            }
            opts.verify_z3 = true;
            opts.verify_requested = true;
            opts.verify_mode = mode;
            i += 1;
        } else if (std.mem.eql(u8, arg, "--no-verify")) {
            try claim(&seen_verify);
            opts.verify_z3 = false;
            opts.verify_requested = true;
            i += 1;
        } else if (std.mem.eql(u8, arg, "--verify-stats")) {
            try claim(&seen_verify_stats);
            opts.verify_stats = true;
            i += 1;
        } else if (std.mem.eql(u8, arg, "--explain")) {
            try claim(&seen_explain);
            opts.explain_cores = true;
            i += 1;
        } else if (std.mem.eql(u8, arg, "--z3-proofs")) {
            try claim(&seen_z3_proofs);
            opts.z3_proofs = true;
            i += 1;
        } else if (std.mem.eql(u8, arg, "--minimize-cores")) {
            try claim(&seen_minimize_cores);
            opts.minimize_cores = true;
            i += 1;
        } else if (std.mem.eql(u8, arg, "--keep-proved-checks")) {
            try claim(&seen_keep_proved_checks);
            opts.keep_proved_checks = true;
            i += 1;
        } else if (std.mem.eql(u8, arg, "--mlir-debug")) {
            try claim(&seen_mlir_debug);
            if (i + 1 >= args.len) return error.MissingArgument;
            try parseMlirDebugList(&opts, args[i + 1]);
            i += 2;
        } else if (std.mem.startsWith(u8, arg, "--mlir-debug=")) {
            try claim(&seen_mlir_debug);
            try parseMlirDebugList(&opts, arg["--mlir-debug=".len..]);
            i += 1;
        } else if (std.mem.eql(u8, arg, "--mlir-pass-pipeline")) {
            try claim(&seen_mlir_pass_pipeline);
            if (i + 1 >= args.len) return error.MissingArgument;
            opts.mlir_pass_pipeline = args[i + 1];
            i += 2;
        } else if (std.mem.startsWith(u8, arg, "--mlir-pass-pipeline=")) {
            try claim(&seen_mlir_pass_pipeline);
            opts.mlir_pass_pipeline = arg["--mlir-pass-pipeline=".len..];
            i += 1;
        } else if (std.mem.eql(u8, arg, "--debug")) {
            try claim(&seen_debug);
            opts.debug = true;
            i += 1;
        } else if (std.mem.eql(u8, arg, "--debug-info")) {
            try claim(&seen_debug_info);
            opts.debug_info = true;
            i += 1;
        } else if (std.mem.eql(u8, arg, "-O0") or std.mem.eql(u8, arg, "-Onone")) {
            try claim(&seen_opt_level);
            opts.mlir_opt_level = "none";
            i += 1;
        } else if (std.mem.eql(u8, arg, "-O1") or std.mem.eql(u8, arg, "-Obasic")) {
            try claim(&seen_opt_level);
            opts.mlir_opt_level = "basic";
            i += 1;
        } else if (std.mem.eql(u8, arg, "-O2") or std.mem.eql(u8, arg, "-Oaggressive")) {
            try claim(&seen_opt_level);
            opts.mlir_opt_level = "aggressive";
            i += 1;
        } else if (std.mem.eql(u8, arg, "--no-validate-mlir")) {
            try claim(&seen_validate_mlir);
            opts.validate_mlir = false;
            i += 1;
        } else if (std.mem.eql(u8, arg, "--no-canonicalize")) {
            try claim(&seen_canonicalize);
            opts.canonicalize_mlir = false;
            i += 1;
        } else if (std.mem.eql(u8, arg, "--chain-id")) {
            try claim(&seen_chain_id);
            if (i + 1 >= args.len) return error.MissingArgument;
            opts.chain_id = std.fmt.parseInt(u64, args[i + 1], 0) catch {
                return error.UnknownArgument;
            };
            i += 2;
        } else if (std.mem.startsWith(u8, arg, "--chain-id=")) {
            try claim(&seen_chain_id);
            opts.chain_id = std.fmt.parseInt(u64, arg["--chain-id=".len..], 0) catch {
                return error.UnknownArgument;
            };
            i += 1;
        } else if (std.mem.eql(u8, arg, "--check")) {
            try claim(&seen_fmt_check);
            opts.fmt_check = true;
            i += 1;
        } else if (std.mem.eql(u8, arg, "--diff")) {
            try claim(&seen_fmt_diff);
            opts.fmt_diff = true;
            i += 1;
        } else if (std.mem.eql(u8, arg, "--stdout")) {
            try claim(&seen_fmt_stdout);
            opts.fmt_stdout = true;
            i += 1;
        } else if (std.mem.eql(u8, arg, "--width")) {
            try claim(&seen_fmt_width);
            if (i + 1 >= args.len) return error.MissingArgument;
            opts.fmt_width = std.fmt.parseInt(u32, args[i + 1], 10) catch {
                return error.UnknownArgument;
            };
            if (opts.fmt_width.? == 0) return error.UnknownArgument;
            i += 2;
        } else if (std.mem.eql(u8, arg, "--metrics") or std.mem.eql(u8, arg, "--time-report")) {
            try claim(&seen_metrics);
            opts.metrics = true;
            i += 1;
        } else if (std.mem.eql(u8, arg, "-v") or std.mem.eql(u8, arg, "--version")) {
            try claim(&seen_version);
            opts.show_version = true;
            i += 1;
        } else if (opts.input_file == null and !std.mem.startsWith(u8, arg, "-")) {
            try claim(&seen_input_file);
            opts.input_file = arg;
            i += 1;
        } else {
            return error.UnknownArgument;
        }
    }

    return opts;
}

test "parse verify mode and toggles" {
    const args = [_][]const u8{
        "--verify=full",
        "--verify-stats",
        "--explain",
        "--z3-proofs",
        "--minimize-cores",
        "--keep-proved-checks",
        "input.ora",
    };
    const parsed = try parseArgs(args[0..]);
    try std.testing.expect(parsed.verify_z3);
    try std.testing.expect(parsed.verify_mode != null);
    try std.testing.expect(std.mem.eql(u8, parsed.verify_mode.?, "full"));
    try std.testing.expect(parsed.verify_stats);
    try std.testing.expect(parsed.explain_cores);
    try std.testing.expect(parsed.z3_proofs);
    try std.testing.expect(parsed.minimize_cores);
    try std.testing.expect(parsed.keep_proved_checks);
}

test "parse invalid verify mode fails" {
    const args = [_][]const u8{"--verify=turbo"};
    try std.testing.expectError(error.UnknownArgument, parseArgs(args[0..]));
}

test "parse debug-info flag" {
    const args = [_][]const u8{ "--debug-info", "input.ora" };
    const parsed = try parseArgs(args[0..]);
    try std.testing.expect(parsed.debug_info);
    try std.testing.expectEqualStrings("input.ora", parsed.input_file.?);
}
