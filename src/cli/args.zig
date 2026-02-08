// ============================================================================
// CLI Argument Parsing
// ============================================================================

const std = @import("std");

pub const CliOptions = struct {
    output_dir: ?[]const u8 = null,
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
    emit_abi: bool = false,
    emit_abi_solidity: bool = false,
    emit_abi_extras: bool = false,
    cpp_lowering_stub: bool = false,
    canonicalize_mlir: bool = true,
    analyze_state: bool = false,
    verify_z3: bool = true,
    debug: bool = false,
    mlir_opt_level: ?[]const u8 = null,
    // fmt options
    fmt: bool = false,
    fmt_check: bool = false,
    fmt_diff: bool = false,
    fmt_stdout: bool = false,
    fmt_width: ?u32 = null,
    show_version: bool = false,
    metrics: bool = false,
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
        } else if (std.mem.startsWith(u8, arg, "--emit-ast=")) {
            opts.emit_ast = true;
            opts.emit_ast_format = arg["--emit-ast=".len..];
            i += 1;
        } else if (std.mem.eql(u8, arg, "--emit-typed-ast")) {
            opts.emit_typed_ast = true;
            i += 1;
        } else if (std.mem.startsWith(u8, arg, "--emit-typed-ast=")) {
            opts.emit_typed_ast = true;
            opts.emit_typed_ast_format = arg["--emit-typed-ast=".len..];
            i += 1;
        } else if (std.mem.eql(u8, arg, "--emit-mlir")) {
            opts.emit_mlir = true;
            i += 1;
        } else if (std.mem.eql(u8, arg, "--emit-mlir-sir") or std.mem.eql(u8, arg, "--emit-sir")) {
            opts.emit_mlir_sir = true;
            i += 1;
        } else if (std.mem.eql(u8, arg, "--emit-sir-text")) {
            opts.emit_sir_text = true;
            i += 1;
        } else if (std.mem.eql(u8, arg, "--emit-bytecode")) {
            opts.emit_bytecode = true;
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
        } else if (std.mem.eql(u8, arg, "--emit-abi-extras")) {
            opts.emit_abi_extras = true;
            i += 1;
        } else if (std.mem.eql(u8, arg, "--cpp-lowering-stub")) {
            opts.cpp_lowering_stub = true;
            i += 1;
        } else if (std.mem.eql(u8, arg, "--verify")) {
            opts.verify_z3 = true;
            i += 1;
        } else if (std.mem.eql(u8, arg, "--no-verify")) {
            opts.verify_z3 = false;
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
        } else if (std.mem.eql(u8, arg, "fmt")) {
            opts.fmt = true;
            i += 1;
        } else if (std.mem.eql(u8, arg, "--check")) {
            opts.fmt_check = true;
            i += 1;
        } else if (std.mem.eql(u8, arg, "--diff")) {
            opts.fmt_diff = true;
            i += 1;
        } else if (std.mem.eql(u8, arg, "--stdout")) {
            opts.fmt_stdout = true;
            i += 1;
        } else if (std.mem.eql(u8, arg, "--width")) {
            if (i + 1 >= args.len) return error.MissingArgument;
            opts.fmt_width = std.fmt.parseInt(u32, args[i + 1], 10) catch {
                return error.UnknownArgument;
            };
            i += 2;
        } else if (std.mem.eql(u8, arg, "--metrics") or std.mem.eql(u8, arg, "--time-report")) {
            opts.metrics = true;
            i += 1;
        } else if (std.mem.eql(u8, arg, "-v") or std.mem.eql(u8, arg, "--version")) {
            opts.show_version = true;
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
