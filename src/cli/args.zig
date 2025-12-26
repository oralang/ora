// ============================================================================
// CLI Argument Parsing
// ============================================================================

const std = @import("std");

pub const CliOptions = struct {
    output_dir: ?[]const u8 = null,
    input_file: ?[]const u8 = null,
    emit_tokens: bool = false,
    emit_ast: bool = false,
    emit_mlir: bool = false,
    emit_mlir_sir: bool = false,
    emit_cfg: bool = false,
    emit_abi: bool = false,
    emit_abi_solidity: bool = false,
    canonicalize_mlir: bool = true,
    analyze_state: bool = false,
    verify_z3: bool = false,
    debug: bool = false,
    mlir_opt_level: ?[]const u8 = null,
};

pub const ParseError = error{
    MissingArgument,
    UnknownArgument,
};

pub fn parseArgs(args: []const []const u8) ParseError!CliOptions {
    var opts = CliOptions{};
    var i: usize = 0;

    while (i < args.len) {
        const arg = args[i];
        if (std.mem.eql(u8, arg, "-o") or std.mem.eql(u8, arg, "--output-dir")) {
            if (i + 1 >= args.len) return error.MissingArgument;
            opts.output_dir = args[i + 1];
            i += 2;
        } else if (std.mem.eql(u8, arg, "--emit-tokens")) {
            opts.emit_tokens = true;
            i += 1;
        } else if (std.mem.eql(u8, arg, "--emit-ast")) {
            opts.emit_ast = true;
            i += 1;
        } else if (std.mem.eql(u8, arg, "--emit-mlir")) {
            opts.emit_mlir = true;
            i += 1;
        } else if (std.mem.eql(u8, arg, "--emit-mlir-sir")) {
            opts.emit_mlir_sir = true;
            i += 1;
        } else if (std.mem.eql(u8, arg, "--emit-cfg")) {
            opts.emit_cfg = true;
            i += 1;
        } else if (std.mem.eql(u8, arg, "--emit-abi")) {
            opts.emit_abi = true;
            i += 1;
        } else if (std.mem.eql(u8, arg, "--emit-abi-solidity")) {
            opts.emit_abi_solidity = true;
            i += 1;
        } else if (std.mem.eql(u8, arg, "--verify")) {
            opts.verify_z3 = true;
            i += 1;
        } else if (std.mem.eql(u8, arg, "--debug")) {
            opts.debug = true;
            i += 1;
        } else if (std.mem.eql(u8, arg, "--analyze-state")) {
            opts.analyze_state = true;
            i += 1;
        } else if (std.mem.eql(u8, arg, "-O0") or std.mem.eql(u8, arg, "-Onone")) {
            opts.mlir_opt_level = "none";
            i += 1;
        } else if (std.mem.eql(u8, arg, "-O1") or std.mem.eql(u8, arg, "-Obasic")) {
            opts.mlir_opt_level = "basic";
            i += 1;
        } else if (std.mem.eql(u8, arg, "-O2") or std.mem.eql(u8, arg, "-Oaggressive")) {
            opts.mlir_opt_level = "aggressive";
            i += 1;
        } else if (std.mem.eql(u8, arg, "--no-validate-mlir")) {
            i += 1;
        } else if (std.mem.eql(u8, arg, "--no-canonicalize")) {
            opts.canonicalize_mlir = false;
            i += 1;
        } else if (opts.input_file == null and !std.mem.startsWith(u8, arg, "-")) {
            opts.input_file = arg;
            i += 1;
        } else {
            return error.UnknownArgument;
        }
    }

    return opts;
}
