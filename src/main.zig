//! Ora Language Compiler CLI
//!
//! Command-line interface for the Ora compiler. Supports lexing, parsing,
//! AST generation, MLIR emission, and full compilation to Yul/bytecode.
//!
//! SECTIONS:
//!   • Main entry & argument parsing
//!   • Usage & help text
//!   • Command handlers (lex, parse, ast, compile)
//!   • Artifact saving functions
//!   • Parser & compilation workflows
//!   • MLIR integration & code generation

const std = @import("std");
const lib = @import("ora_lib");
const build_options = @import("build_options");
const cli_args = @import("cli/args.zig");
const compiler = lib.compiler;
const project_config = @import("config/mod.zig");
const import_graph = @import("ora_imports");
const log = @import("log");
const Metrics = @import("metrics.zig").Metrics;
const ManagedArrayList = std.array_list.Managed;

/// MLIR-related command line options
const MlirOptions = struct {
    emit_mlir: bool,
    emit_mlir_sir: bool,
    emit_sir_text: bool,
    emit_bytecode: bool,
    emit_cfg_mode: ?[]const u8 = null,
    opt_level: ?[]const u8,
    output_dir: ?[]const u8,
    debug_enabled: bool = false,
    debug_info: bool = false,
    canonicalize: bool = true,
    validate_mlir: bool = true,
    verify_z3: bool = true,
    verify_mode: ?[]const u8 = null,
    verify_calls: ?bool = null,
    verify_state: ?bool = null,
    verify_stats: bool = false,
    explain_cores: bool = false,
    z3_proofs: bool = false,
    minimize_cores: bool = false,
    emit_smt_report: bool = false,
    mlir_pass_pipeline: ?[]const u8 = null,
    mlir_verify_each_pass: bool = false,
    mlir_pass_timing: bool = false,
    mlir_pass_statistics: bool = false,
    mlir_print_ir: ?[]const u8 = null,
    mlir_print_ir_pass: ?[]const u8 = null,
    mlir_crash_reproducer: ?[]const u8 = null,
    mlir_print_op_on_diagnostic: bool = false,
    cpp_lowering_stub: bool = false,
    persist_ora_mlir: bool = false,
    persist_sir_mlir: bool = false,
    metrics: *Metrics = undefined,

    fn getOptimizationLevel(self: MlirOptions) OptimizationLevel {
        if (self.opt_level) |level| {
            if (std.mem.eql(u8, level, "none")) return .None;
            if (std.mem.eql(u8, level, "basic")) return .Basic;
            if (std.mem.eql(u8, level, "aggressive")) return .Aggressive;
        }

        // use build-time default if no command-line option provided
        const build_default = build_options.mlir_opt_level;
        if (std.mem.eql(u8, build_default, "none")) return .None;
        if (std.mem.eql(u8, build_default, "basic")) return .Basic;
        if (std.mem.eql(u8, build_default, "aggressive")) return .Aggressive;

        return .Basic; // Final fallback
    }
};

const OptimizationLevel = enum {
    None,
    Basic,
    Aggressive,
};

const CommandKind = enum {
    Build,
    Emit,
    Debug,
    Fmt,
};

const Subcommand = enum {
    None,
    Build,
    Emit,
    Debug,
    Fmt,
};

const DebugCliOptions = struct {
    filtered_args: std.ArrayList([]const u8),
    init_signature: ?[]const u8 = null,
    init_arg_values: std.ArrayList([]const u8),
    init_calldata_hex: ?[]const u8 = null,
    signature: ?[]const u8 = null,
    arg_values: std.ArrayList([]const u8),
    calldata_hex: ?[]const u8 = null,
    verify_requested: bool = false,
    /// Headless mode: emit debug artifacts but skip launching the TUI.
    /// Used by debug-artifact regression tests and for offline artifact
    /// generation that ships traces to another engineer.
    no_tui: bool = false,
    /// Limit overrides forwarded verbatim to ora-evm-debug-tui. Stored as
    /// the raw text so we can re-emit them onto the spawned argv without
    /// any (re)formatting drift.
    gas_limit: ?[]const u8 = null,
    max_steps: ?[]const u8 = null,
    deploy_step_cap: ?[]const u8 = null,
    artifact_max_bytes: ?[]const u8 = null,

    fn init() DebugCliOptions {
        return .{
            .filtered_args = .{},
            .init_arg_values = .{},
            .arg_values = .{},
        };
    }

    fn deinit(self: *DebugCliOptions, allocator: std.mem.Allocator) void {
        self.filtered_args.deinit(allocator);
        self.init_arg_values.deinit(allocator);
        self.arg_values.deinit(allocator);
    }
};

const InitTemplates = struct {
    const ora_toml =
        \\schema_version = "0.1"
        \\
        \\[compiler]
        \\output_dir = "./artifacts"
        \\init_args = ["initial_counter=0"]
        \\
        \\[[targets]]
        \\name = "Main"
        \\kind = "contract"
        \\root = "contracts/main.ora"
    ;

    const main_contract =
        \\contract Main {
        \\    storage var counter: u256 = 0;
        \\
        \\    pub fn init(initial_counter: u256) {
        \\        counter = initial_counter;
        \\    }
        \\
        \\    pub fn set(next: u256) {
        \\        counter = next;
        \\    }
        \\
        \\    pub fn run() -> u256 {
        \\        return counter;
        \\    }
        \\}
    ;

    const readme_md =
        \\# Ora Project
        \\
        \\This project was created with `ora init`.
        \\
        \\## Commands
        \\- `ora build`
        \\- `ora emit --emit-typed-ast contracts/main.ora`
    ;
};

fn hasEmitFlags(parsed: cli_args.CliOptions) bool {
    return parsed.emit_tokens or
        parsed.emit_ast or
        parsed.emit_typed_ast or
        parsed.emit_mlir or
        parsed.emit_mlir_sir or
        parsed.emit_sir_text or
        parsed.emit_bytecode or
        parsed.emit_cfg;
}

fn parseDebugCliOptions(allocator: std.mem.Allocator, args: []const []const u8) !DebugCliOptions {
    var opts = DebugCliOptions.init();
    errdefer opts.deinit(allocator);

    var i: usize = 0;
    while (i < args.len) {
        const arg = args[i];
        if (std.mem.eql(u8, arg, "--init-signature")) {
            if (i + 1 >= args.len) return error.MissingArgument;
            opts.init_signature = args[i + 1];
            i += 2;
            continue;
        }
        if (std.mem.eql(u8, arg, "--init-arg")) {
            if (i + 1 >= args.len) return error.MissingArgument;
            try opts.init_arg_values.append(allocator, args[i + 1]);
            i += 2;
            continue;
        }
        if (std.mem.eql(u8, arg, "--init-calldata-hex")) {
            if (i + 1 >= args.len) return error.MissingArgument;
            opts.init_calldata_hex = args[i + 1];
            i += 2;
            continue;
        }
        if (std.mem.eql(u8, arg, "--signature")) {
            if (i + 1 >= args.len) return error.MissingArgument;
            opts.signature = args[i + 1];
            i += 2;
            continue;
        }
        if (std.mem.eql(u8, arg, "--arg")) {
            if (i + 1 >= args.len) return error.MissingArgument;
            try opts.arg_values.append(allocator, args[i + 1]);
            i += 2;
            continue;
        }
        if (std.mem.eql(u8, arg, "--calldata-hex")) {
            if (i + 1 >= args.len) return error.MissingArgument;
            opts.calldata_hex = args[i + 1];
            i += 2;
            continue;
        }

        if (std.mem.eql(u8, arg, "--verify") or std.mem.startsWith(u8, arg, "--verify=")) {
            opts.verify_requested = true;
        }

        if (std.mem.eql(u8, arg, "--no-tui")) {
            opts.no_tui = true;
            i += 1;
            continue;
        }
        if (std.mem.eql(u8, arg, "--gas-limit")) {
            if (i + 1 >= args.len) return error.MissingArgument;
            opts.gas_limit = args[i + 1];
            i += 2;
            continue;
        }
        if (std.mem.eql(u8, arg, "--max-steps")) {
            if (i + 1 >= args.len) return error.MissingArgument;
            opts.max_steps = args[i + 1];
            i += 2;
            continue;
        }
        if (std.mem.eql(u8, arg, "--deploy-step-cap")) {
            if (i + 1 >= args.len) return error.MissingArgument;
            opts.deploy_step_cap = args[i + 1];
            i += 2;
            continue;
        }
        if (std.mem.eql(u8, arg, "--artifact-max-bytes")) {
            if (i + 1 >= args.len) return error.MissingArgument;
            opts.artifact_max_bytes = args[i + 1];
            i += 2;
            continue;
        }

        try opts.filtered_args.append(allocator, arg);
        i += 1;
    }

    return opts;
}

fn initProjectLayout(target_dir: []const u8) !void {
    var exists = true;
    std.fs.cwd().access(target_dir, .{}) catch |err| switch (err) {
        error.FileNotFound => exists = false,
        else => return err,
    };

    if (!exists) {
        try std.fs.cwd().makePath(target_dir);
    }

    var root_dir = std.fs.cwd().openDir(target_dir, .{ .iterate = true }) catch |err| switch (err) {
        error.NotDir => return error.InitTargetIsFile,
        else => return err,
    };
    defer root_dir.close();

    var iter = root_dir.iterate();
    if ((try iter.next()) != null) {
        return error.InitTargetNotEmpty;
    }

    try root_dir.makePath("contracts");
    try root_dir.writeFile(.{
        .sub_path = "ora.toml",
        .data = InitTemplates.ora_toml,
    });
    try root_dir.writeFile(.{
        .sub_path = "contracts/main.ora",
        .data = InitTemplates.main_contract,
    });
    try root_dir.writeFile(.{
        .sub_path = "README.md",
        .data = InitTemplates.readme_md,
    });
}

fn runInitCommand(target_dir: []const u8) !void {
    var stdout_buffer: [1024]u8 = undefined;
    var stdout_writer = std.fs.File.stdout().writer(&stdout_buffer);
    const stdout = &stdout_writer.interface;

    try initProjectLayout(target_dir);

    try stdout.print("Initialized Ora project in {s}\n", .{target_dir});
    try stdout.print("Next step: cd {s} && ora build\n", .{target_dir});
    try stdout.flush();
}

// ============================================================================
// SECTION 1: Main Entry Point & Argument Parsing
// ============================================================================

/// Ora CLI application
pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    const args = try std.process.argsAlloc(allocator);
    defer std.process.argsFree(allocator, args);

    if (args.len < 2) {
        try printUsage();
        return;
    }

    if (args.len >= 2 and std.mem.eql(u8, args[1], "init")) {
        if (args.len > 3) {
            std.debug.print("error: init accepts at most one optional <path>\n", .{});
            try printUsage();
            std.process.exit(2);
        }
        const target_dir = if (args.len == 3) args[2] else ".";
        runInitCommand(target_dir) catch |err| switch (err) {
            error.InitTargetNotEmpty => {
                std.debug.print("error: init target '{s}' is not empty\n", .{target_dir});
                std.process.exit(2);
            },
            error.InitTargetIsFile => {
                std.debug.print("error: init target '{s}' is a file, expected directory\n", .{target_dir});
                std.process.exit(2);
            },
            else => return err,
        };
        return;
    }

    const subcommand: Subcommand = if (args.len >= 2 and std.mem.eql(u8, args[1], "fmt"))
        .Fmt
    else if (args.len >= 2 and std.mem.eql(u8, args[1], "build"))
        .Build
    else if (args.len >= 2 and std.mem.eql(u8, args[1], "emit"))
        .Emit
    else if (args.len >= 2 and std.mem.eql(u8, args[1], "debug"))
        .Debug
    else
        .None;

    const args_to_parse = switch (subcommand) {
        .Fmt, .Build, .Emit, .Debug => args[2..],
        .None => args[1..],
    };

    var debug_cli: ?DebugCliOptions = null;
    defer if (debug_cli) |*opts| opts.deinit(allocator);

    var parse_args: std.ArrayList([]const u8) = .{};
    defer parse_args.deinit(allocator);
    if (subcommand == .Debug) {
        debug_cli = parseDebugCliOptions(allocator, args_to_parse) catch {
            try printUsage();
            return;
        };
        try parse_args.appendSlice(allocator, debug_cli.?.filtered_args.items);
    } else {
        for (args_to_parse) |arg| try parse_args.append(allocator, arg);
    }

    var parsed = cli_args.parseArgs(parse_args.items) catch {
        try printUsage();
        return;
    };

    if (subcommand == .Fmt) {
        parsed.fmt = true;
    }
    if (parsed.show_version) {
        try printVersion();
        return;
    }

    const output_dir: ?[]const u8 = parsed.output_dir;
    const input_file: ?[]const u8 = parsed.input_file;
    const emit_tokens: bool = parsed.emit_tokens;
    const emit_ast: bool = parsed.emit_ast;
    const emit_ast_format: ?[]const u8 = parsed.emit_ast_format;
    const emit_typed_ast: bool = parsed.emit_typed_ast;
    const emit_typed_ast_format: ?[]const u8 = parsed.emit_typed_ast_format;
    var emit_mlir: bool = parsed.emit_mlir;
    var emit_mlir_sir: bool = parsed.emit_mlir_sir;
    const emit_sir_text: bool = parsed.emit_sir_text;
    const emit_bytecode: bool = parsed.emit_bytecode;
    const emit_cfg: bool = parsed.emit_cfg;
    const emit_cfg_mode: ?[]const u8 = parsed.emit_cfg_mode;
    const canonicalize_mlir: bool = parsed.canonicalize_mlir;
    const validate_mlir: bool = parsed.validate_mlir;
    const verify_z3: bool = parsed.verify_z3;
    const verify_mode: ?[]const u8 = parsed.verify_mode;
    const verify_calls: ?bool = parsed.verify_calls;
    const verify_state: ?bool = parsed.verify_state;
    const verify_stats: bool = parsed.verify_stats;
    const explain_cores: bool = parsed.explain_cores or parsed.minimize_cores;
    const z3_proofs: bool = parsed.z3_proofs;
    const minimize_cores: bool = parsed.minimize_cores;
    const emit_smt_report: bool = parsed.emit_smt_report;
    const mlir_pass_pipeline: ?[]const u8 = parsed.mlir_pass_pipeline;
    const mlir_verify_each_pass: bool = parsed.mlir_verify_each_pass;
    const mlir_pass_timing: bool = parsed.mlir_pass_timing;
    const mlir_pass_statistics: bool = parsed.mlir_pass_statistics;
    const mlir_print_ir: ?[]const u8 = parsed.mlir_print_ir;
    const mlir_print_ir_pass: ?[]const u8 = parsed.mlir_print_ir_pass;
    const mlir_crash_reproducer: ?[]const u8 = parsed.mlir_crash_reproducer;
    const mlir_print_op_on_diagnostic: bool = parsed.mlir_print_op_on_diagnostic;
    const cpp_lowering_stub: bool = parsed.cpp_lowering_stub;
    var debug_enabled: bool = parsed.debug;
    const debug_info: bool = parsed.debug_info;
    if (!debug_enabled) {
        if (std.posix.getenv("ORA_DEBUG")) |env_value| {
            if (env_value[0] != 0 and env_value[0] != '0') {
                debug_enabled = true;
            }
        }
    }
    const mlir_opt_level: ?[]const u8 = parsed.mlir_opt_level;
    const fmt: bool = parsed.fmt;
    const fmt_check: bool = parsed.fmt_check;
    const fmt_diff: bool = parsed.fmt_diff;
    const fmt_stdout: bool = parsed.fmt_stdout;
    const fmt_width: ?u32 = parsed.fmt_width;
    const metrics_enabled: bool = parsed.metrics;

    var metrics = Metrics.init(metrics_enabled);

    log.setDebugEnabled(debug_enabled);

    // handle fmt command
    if (fmt) {
        if (input_file == null) {
            std.debug.print("error: fmt requires an input file or directory\n", .{});
            try printUsage();
            std.process.exit(2);
        }
        try runFmt(allocator, input_file.?, fmt_check, fmt_diff, fmt_stdout, fmt_width);
        return;
    }

    const emit_flags_requested = hasEmitFlags(parsed);
    const command_kind: CommandKind = switch (subcommand) {
        .Fmt => .Fmt,
        .Build => .Build,
        .Emit => .Emit,
        .Debug => .Debug,
        .None => if (emit_flags_requested) .Emit else .Build,
    };
    if (command_kind == .Build and emit_flags_requested) {
        std.debug.print("error: build mode does not accept --emit-* flags. Use 'ora emit ...' for debug outputs.\n", .{});
        std.process.exit(2);
    }

    if (command_kind == .Build and !verify_z3) {
        std.debug.print("error: build mode requires SMT verification; '--no-verify' is not supported.\n", .{});
        std.process.exit(2);
    }

    if (command_kind == .Debug and parsed.input_file == null) {
        std.debug.print("error: debug requires an input file.\n", .{});
        std.process.exit(2);
    }

    if ((mlir_verify_each_pass or mlir_pass_timing) and mlir_pass_pipeline == null) {
        std.debug.print("error: --mlir-verify-each-pass and --mlir-pass-timing require --mlir-pass-pipeline.\n", .{});
        std.process.exit(2);
    }

    if (mlir_print_ir_pass != null and mlir_print_ir == null) {
        std.debug.print("error: --mlir-print-ir-pass requires --mlir-print-ir.\n", .{});
        std.process.exit(2);
    }

    if (command_kind == .Emit) {
        // if no --emit-X flag is set, default to MLIR generation
        if (!emit_tokens and !emit_ast and !emit_typed_ast and !emit_mlir and !emit_mlir_sir and !emit_sir_text and !emit_bytecode and !emit_cfg) {
            emit_mlir = true; // Default: emit MLIR
        }
        if ((emit_sir_text or emit_bytecode or (emit_cfg and emit_cfg_mode != null and std.ascii.eqlIgnoreCase(emit_cfg_mode.?, "sir"))) and !emit_mlir_sir) {
            emit_mlir_sir = true;
        }
    }

    // create MLIR options structure
    const mlir_options = MlirOptions{
        .emit_mlir = emit_mlir,
        .emit_mlir_sir = emit_mlir_sir,
        .emit_sir_text = emit_sir_text,
        .emit_bytecode = emit_bytecode,
        .emit_cfg_mode = emit_cfg_mode,
        .opt_level = mlir_opt_level,
        .output_dir = output_dir,
        .debug_enabled = debug_enabled,
        .debug_info = debug_info,
        .canonicalize = canonicalize_mlir,
        .validate_mlir = validate_mlir,
        .verify_z3 = verify_z3,
        .verify_mode = verify_mode,
        .verify_calls = verify_calls,
        .verify_state = verify_state,
        .verify_stats = verify_stats,
        .explain_cores = explain_cores,
        .z3_proofs = z3_proofs,
        .minimize_cores = minimize_cores,
        .emit_smt_report = emit_smt_report,
        .mlir_pass_pipeline = mlir_pass_pipeline,
        .mlir_verify_each_pass = mlir_verify_each_pass,
        .mlir_pass_timing = mlir_pass_timing,
        .mlir_pass_statistics = mlir_pass_statistics,
        .mlir_print_ir = mlir_print_ir,
        .mlir_print_ir_pass = mlir_print_ir_pass,
        .mlir_crash_reproducer = mlir_crash_reproducer,
        .mlir_print_op_on_diagnostic = mlir_print_op_on_diagnostic,
        .cpp_lowering_stub = cpp_lowering_stub,
        .persist_ora_mlir = false,
        .persist_sir_mlir = false,
        .metrics = &metrics,
    };

    if (command_kind == .Debug) {
        const debug_options = debug_cli.?;
        const file_path = parsed.input_file.?;
        const resolver = try discoverResolverOptionsForFile(allocator, file_path);
        defer if (resolver.include_roots) |include_roots| {
            freeResolvedIncludeRoots(allocator, include_roots);
        };

        var debug_mlir_options = mlir_options;
        debug_mlir_options.emit_mlir = false;
        debug_mlir_options.emit_mlir_sir = true;
        debug_mlir_options.emit_sir_text = true;
        debug_mlir_options.emit_bytecode = true;
        debug_mlir_options.debug_info = true;
        if (!debug_options.verify_requested) {
            debug_mlir_options.verify_z3 = false;
            debug_mlir_options.emit_smt_report = false;
        }

        const artifact_root = try runDebugArtifacts(
            allocator,
            file_path,
            parsed.output_dir,
            debug_mlir_options,
            resolver.options,
        );
        defer allocator.free(artifact_root);

        if (debug_options.no_tui) return;

        try launchDebuggerTui(
            allocator,
            file_path,
            artifact_root,
            debug_options.init_signature,
            debug_options.init_arg_values.items,
            debug_options.init_calldata_hex,
            debug_options.signature,
            debug_options.arg_values.items,
            debug_options.calldata_hex,
            debug_options.gas_limit,
            debug_options.max_steps,
            debug_options.deploy_step_cap,
            debug_options.artifact_max_bytes,
        );
        return;
    }

    if (command_kind == .Build) {
        var matched_include_roots: ?[]const []const u8 = null;
        defer if (matched_include_roots) |include_roots| {
            freeResolvedIncludeRoots(allocator, include_roots);
        };
        var matched_init_args: ?[]project_config.InitArg = null;
        defer if (matched_init_args) |init_args| {
            freeCombinedInitArgs(allocator, init_args);
        };
        var matched_output_dir: ?[]u8 = null;
        defer if (matched_output_dir) |out_dir| {
            allocator.free(out_dir);
        };

        var build_resolver_options: import_graph.ResolverOptions = .{};
        var build_output_dir: ?[]const u8 = output_dir;
        if (input_file) |build_file_path| {
            const start_dir = std.fs.path.dirname(build_file_path) orelse ".";
            const loaded_opt = project_config.loadDiscoveredFromStartDir(allocator, start_dir) catch |err| {
                std.debug.print("error: failed to load ora.toml: {s}\n", .{@errorName(err)});
                std.process.exit(2);
            };

            if (loaded_opt) |loaded_value| {
                var loaded = loaded_value;
                defer loaded.deinit(allocator);

                matched_init_args = try combineInitArgSlices(allocator, loaded.config.compiler_init_args, &.{});

                const target_idx_opt = project_config.findMatchingTargetIndex(allocator, &loaded, build_file_path) catch |err| {
                    std.debug.print("error: failed to match target in ora.toml: {s}\n", .{@errorName(err)});
                    std.process.exit(2);
                };

                if (target_idx_opt) |target_idx| {
                    const target = loaded.config.targets[target_idx];
                    if (matched_init_args) |init_args| {
                        freeCombinedInitArgs(allocator, init_args);
                        matched_init_args = null;
                    }
                    if (target.kind == .contract) {
                        matched_init_args = try combineInitArgSlices(allocator, loaded.config.compiler_init_args, target.init_args);
                    } else {
                        matched_init_args = try allocator.alloc(project_config.InitArg, 0);
                    }
                    matched_include_roots = resolveIncludeRootsForTarget(allocator, loaded.config_dir, target.include_paths) catch |err| {
                        std.debug.print("error: failed to resolve include_paths for target '{s}': {s}\n", .{ target.name, @errorName(err) });
                        std.process.exit(2);
                    };
                    build_resolver_options.include_roots = matched_include_roots.?;

                    if (build_output_dir == null) {
                        if (target.output_dir) |target_out| {
                            matched_output_dir = project_config.resolvePathFromConfigDir(allocator, loaded.config_dir, target_out) catch |err| {
                                std.debug.print("error: failed to resolve target output_dir for '{s}': {s}\n", .{ target.name, @errorName(err) });
                                std.process.exit(2);
                            };
                        } else {
                            const base_output = if (loaded.config.compiler_output_dir) |compiler_out|
                                project_config.resolvePathFromConfigDir(allocator, loaded.config_dir, compiler_out) catch |err| {
                                    std.debug.print("error: failed to resolve compiler.output_dir: {s}\n", .{@errorName(err)});
                                    std.process.exit(2);
                                }
                            else
                                project_config.resolvePathFromConfigDir(allocator, loaded.config_dir, "artifacts") catch |err| {
                                    std.debug.print("error: failed to resolve default project output_dir: {s}\n", .{@errorName(err)});
                                    std.process.exit(2);
                                };
                            defer allocator.free(base_output);

                            if (loaded.config.targets.len > 1) {
                                matched_output_dir = std.fs.path.join(allocator, &.{ base_output, target.name }) catch {
                                    std.debug.print("error: failed to allocate target output path\n", .{});
                                    std.process.exit(2);
                                };
                            } else {
                                matched_output_dir = allocator.dupe(u8, base_output) catch {
                                    std.debug.print("error: failed to allocate target output path\n", .{});
                                    std.process.exit(2);
                                };
                            }
                        }
                        build_output_dir = matched_output_dir.?;
                    }
                } else if (build_output_dir == null) {
                    const base_output = if (loaded.config.compiler_output_dir) |compiler_out|
                        project_config.resolvePathFromConfigDir(allocator, loaded.config_dir, compiler_out) catch |err| {
                            std.debug.print("error: failed to resolve compiler.output_dir: {s}\n", .{@errorName(err)});
                            std.process.exit(2);
                        }
                    else
                        project_config.resolvePathFromConfigDir(allocator, loaded.config_dir, "artifacts") catch |err| {
                            std.debug.print("error: failed to resolve default project output_dir: {s}\n", .{@errorName(err)});
                            std.process.exit(2);
                        };
                    defer allocator.free(base_output);

                    matched_output_dir = std.fs.path.join(allocator, &.{ base_output, std.fs.path.stem(build_file_path) }) catch {
                        std.debug.print("error: failed to allocate default project output path\n", .{});
                        std.process.exit(2);
                    };
                    build_output_dir = matched_output_dir.?;
                }
            }
        }

        const build_result = if (input_file) |build_file_path|
            runBuildArtifacts(
                allocator,
                build_file_path,
                build_output_dir,
                mlir_options,
                build_resolver_options,
                if (matched_init_args) |init_args| init_args else @as([]const project_config.InitArg, &.{}),
            )
        else
            runBuildFromDiscoveredConfig(allocator, output_dir, mlir_options);

        var stderr_buffer_build: [1024]u8 = undefined;
        var stderr_writer_build = std.fs.File.stderr().writer(&stderr_buffer_build);
        const stderr_build = &stderr_writer_build.interface;
        try metrics.report(stderr_build);
        try stderr_build.flush();

        build_result catch |err| switch (err) {
            error.VerificationFailed => std.process.exit(1),
            else => return err,
        };
        return;
    }

    if (input_file == null) {
        try printUsage();
        return;
    }

    const file_path = input_file.?;

    const resolver = try discoverResolverOptionsForFile(allocator, file_path);
    defer if (resolver.include_roots) |include_roots| {
        freeResolvedIncludeRoots(allocator, include_roots);
    };

    if (!emit_tokens and !emit_ast and !emit_typed_ast and !emit_mlir and !emit_mlir_sir and !emit_sir_text and !emit_bytecode and !emit_cfg) {
        std.debug.print("error: emit requires an explicit --emit-* mode.\n", .{});
        std.process.exit(2);
    }

    if (emit_tokens) {
        try runCompilerTokenEmit(allocator, file_path);
    } else if (emit_ast or emit_typed_ast) {
        const format = if (emit_typed_ast)
            (emit_typed_ast_format orelse "tree")
        else
            (emit_ast_format orelse "tree");
        try runCompilerAstEmit(allocator, file_path, format, emit_typed_ast, resolver.options, &metrics, debug_enabled);
    } else if (emit_cfg or emit_sir_text or emit_bytecode) {
        try runMlirEmitAdvanced(allocator, file_path, mlir_options, resolver.options, debug_enabled);
    } else {
        try runCompilerMlirEmit(allocator, file_path, mlir_options, resolver.options, debug_enabled);
    }

    // print metrics report (no-op when --metrics is not passed)
    var stderr_buffer: [1024]u8 = undefined;
    var stderr_writer = std.fs.File.stderr().writer(&stderr_buffer);
    const stderr = &stderr_writer.interface;
    try metrics.report(stderr);
    try stderr.flush();
}

fn moveArtifactFile(
    allocator: std.mem.Allocator,
    root_dir: []const u8,
    file_name: []const u8,
    target_dir: []const u8,
) !void {
    const src_path = try std.fs.path.join(allocator, &[_][]const u8{ root_dir, file_name });
    defer allocator.free(src_path);
    const dst_path = try std.fs.path.join(allocator, &[_][]const u8{ target_dir, file_name });
    defer allocator.free(dst_path);
    try std.fs.cwd().makePath(target_dir);
    std.fs.cwd().rename(src_path, dst_path) catch |err| switch (err) {
        error.FileNotFound => {
            std.fs.cwd().access(dst_path, .{}) catch |dst_err| switch (dst_err) {
                error.FileNotFound => {
                    std.fs.cwd().access(src_path, .{}) catch |src_err| switch (src_err) {
                        error.FileNotFound => return error.FileNotFound,
                        else => return src_err,
                    };
                    try std.fs.cwd().copyFile(src_path, std.fs.cwd(), dst_path, .{});
                    try std.fs.cwd().deleteFile(src_path);
                },
                else => return dst_err,
            };
        },
        else => return err,
    };
}

fn moveArtifactFileIfExists(
    allocator: std.mem.Allocator,
    root_dir: []const u8,
    file_name: []const u8,
    target_dir: []const u8,
) !void {
    const src_path = try std.fs.path.join(allocator, &[_][]const u8{ root_dir, file_name });
    defer allocator.free(src_path);
    std.fs.cwd().access(src_path, .{}) catch |err| switch (err) {
        error.FileNotFound => return,
        else => return err,
    };
    try moveArtifactFile(allocator, root_dir, file_name, target_dir);
}

fn runBuildArtifacts(
    allocator: std.mem.Allocator,
    file_path: []const u8,
    output_dir: ?[]const u8,
    base_options: MlirOptions,
    resolver_options: import_graph.ResolverOptions,
    configured_init_args: []const project_config.InitArg,
) !void {
    var stdout_buffer: [1024]u8 = undefined;
    var stdout_writer = std.fs.File.stdout().writer(&stdout_buffer);
    const stdout = &stdout_writer.interface;

    const stem = std.fs.path.stem(file_path);
    const default_root = try std.fmt.allocPrint(allocator, "artifacts/{s}", .{stem});
    defer allocator.free(default_root);
    const artifact_root = output_dir orelse default_root;

    if (std.mem.endsWith(u8, artifact_root, ".hex") or
        std.mem.endsWith(u8, artifact_root, ".bytecode") or
        std.mem.endsWith(u8, artifact_root, ".bin"))
    {
        try stdout.print("error: build mode expects '-o' to be a directory, got '{s}'\n", .{artifact_root});
        std.process.exit(2);
    }

    // reset output tree so each build produces a coherent artifact bundle
    var artifact_root_exists = true;
    std.fs.cwd().access(artifact_root, .{}) catch |err| switch (err) {
        error.FileNotFound => artifact_root_exists = false,
        else => return err,
    };
    if (artifact_root_exists) {
        try std.fs.cwd().deleteTree(artifact_root);
    }
    try std.fs.cwd().makePath(artifact_root);
    var build_succeeded = false;
    errdefer if (!build_succeeded) std.fs.cwd().deleteTree(artifact_root) catch {};

    const abi_dir = try std.fs.path.join(allocator, &[_][]const u8{ artifact_root, "abi" });
    defer allocator.free(abi_dir);
    const bin_dir = try std.fs.path.join(allocator, &[_][]const u8{ artifact_root, "bin" });
    defer allocator.free(bin_dir);
    const sir_dir = try std.fs.path.join(allocator, &[_][]const u8{ artifact_root, "sir" });
    defer allocator.free(sir_dir);
    const verify_dir = try std.fs.path.join(allocator, &[_][]const u8{ artifact_root, "verify" });
    defer allocator.free(verify_dir);
    const mlir_dir = try std.fs.path.join(allocator, &[_][]const u8{ artifact_root, "mlir" });
    defer allocator.free(mlir_dir);

    try std.fs.cwd().makePath(abi_dir);
    try std.fs.cwd().makePath(bin_dir);
    try std.fs.cwd().makePath(sir_dir);
    try std.fs.cwd().makePath(verify_dir);
    try std.fs.cwd().makePath(mlir_dir);

    try validateConfiguredInitArgs(allocator, file_path, resolver_options, configured_init_args);

    // ABI bundle
    try runAbiEmit(allocator, file_path, abi_dir, true, true, true, resolver_options, base_options.debug_enabled);

    // SIR + bytecode + SMT report (verification is mandatory for build mode).
    var build_mlir_options = base_options;
    build_mlir_options.emit_mlir = false;
    build_mlir_options.emit_mlir_sir = true;
    build_mlir_options.emit_sir_text = true;
    build_mlir_options.emit_bytecode = true;
    build_mlir_options.output_dir = artifact_root;
    build_mlir_options.verify_z3 = true;
    build_mlir_options.emit_smt_report = true;
    build_mlir_options.persist_ora_mlir = true;
    build_mlir_options.persist_sir_mlir = true;
    var verification_failed = false;
    const build_emit_result = runMlirEmitAdvanced(allocator, file_path, build_mlir_options, resolver_options, build_mlir_options.debug_enabled);
    build_emit_result catch |err| switch (err) {
        error.VerificationFailed => verification_failed = true,
        else => return err,
    };

    const smt_md_file = try std.fmt.allocPrint(allocator, "{s}.smt.report.md", .{stem});
    defer allocator.free(smt_md_file);
    try moveArtifactFileIfExists(allocator, artifact_root, smt_md_file, verify_dir);

    const smt_json_file = try std.fmt.allocPrint(allocator, "{s}.smt.report.json", .{stem});
    defer allocator.free(smt_json_file);
    try moveArtifactFileIfExists(allocator, artifact_root, smt_json_file, verify_dir);

    if (verification_failed) {
        return error.VerificationFailed;
    }

    const ora_mlir_file = try std.fmt.allocPrint(allocator, "{s}.ora.mlir", .{stem});
    defer allocator.free(ora_mlir_file);
    try moveArtifactFile(allocator, artifact_root, ora_mlir_file, mlir_dir);

    try stdout.print("Artifacts saved to {s}\n", .{artifact_root});
    try stdout.flush();

    // Reorganize generated outputs under stable subfolders only after a successful
    // verification/build run. Verification failures intentionally stop before
    // OraToSIR and bytecode emission, so these artifacts do not exist yet.
    const sir_file = try std.fmt.allocPrint(allocator, "{s}.sir", .{stem});
    defer allocator.free(sir_file);
    try moveArtifactFile(allocator, artifact_root, sir_file, sir_dir);

    const hex_file = try std.fmt.allocPrint(allocator, "{s}.hex", .{stem});
    defer allocator.free(hex_file);
    try moveArtifactFile(allocator, artifact_root, hex_file, bin_dir);

    const sir_mlir_file = try std.fmt.allocPrint(allocator, "{s}.sir.mlir", .{stem});
    defer allocator.free(sir_mlir_file);
    try moveArtifactFile(allocator, artifact_root, sir_mlir_file, mlir_dir);
    build_succeeded = true;
}

fn runDebugArtifacts(
    allocator: std.mem.Allocator,
    file_path: []const u8,
    output_dir: ?[]const u8,
    base_options: MlirOptions,
    resolver_options: import_graph.ResolverOptions,
) ![]u8 {
    var stdout_buffer: [1024]u8 = undefined;
    var stdout_writer = std.fs.File.stdout().writer(&stdout_buffer);
    const stdout = &stdout_writer.interface;

    const stem = std.fs.path.stem(file_path);
    const default_root = try std.fmt.allocPrint(allocator, "artifacts/{s}", .{stem});
    defer allocator.free(default_root);
    const artifact_root = output_dir orelse default_root;

    var artifact_root_exists = true;
    std.fs.cwd().access(artifact_root, .{}) catch |err| switch (err) {
        error.FileNotFound => artifact_root_exists = false,
        else => return err,
    };
    if (artifact_root_exists) {
        try std.fs.cwd().deleteTree(artifact_root);
    }
    try std.fs.cwd().makePath(artifact_root);

    const abi_dir = try std.fs.path.join(allocator, &[_][]const u8{ artifact_root, "abi" });
    defer allocator.free(abi_dir);
    try std.fs.cwd().makePath(abi_dir);

    try runAbiEmit(allocator, file_path, abi_dir, true, true, true, resolver_options, base_options.debug_enabled);

    var debug_mlir_options = base_options;
    debug_mlir_options.output_dir = artifact_root;
    try runMlirEmitAdvanced(allocator, file_path, debug_mlir_options, resolver_options, debug_mlir_options.debug_enabled);

    try stdout.print("Debugger artifacts saved to {s}\n", .{artifact_root});
    try stdout.flush();

    return try allocator.dupe(u8, artifact_root);
}

fn pathExists(path: []const u8) bool {
    std.fs.cwd().access(path, .{}) catch return false;
    return true;
}

fn absolutePathAlloc(allocator: std.mem.Allocator, path: []const u8) ![]u8 {
    if (std.fs.path.isAbsolute(path)) return try allocator.dupe(u8, path);
    return try std.fs.cwd().realpathAlloc(allocator, path);
}

fn findOraRepoRoot(allocator: std.mem.Allocator) ![]u8 {
    const exe_dir = try std.fs.selfExeDirPathAlloc(allocator);
    defer allocator.free(exe_dir);

    const from_exe = try std.fs.path.resolve(allocator, &[_][]const u8{ exe_dir, "..", ".." });
    errdefer allocator.free(from_exe);
    const probe = try std.fs.path.join(allocator, &[_][]const u8{ from_exe, "lib", "evm", "build.zig" });
    defer allocator.free(probe);
    if (pathExists(probe)) return from_exe;

    allocator.free(from_exe);
    const cwd = try std.fs.cwd().realpathAlloc(allocator, ".");
    errdefer allocator.free(cwd);
    const cwd_probe = try std.fs.path.join(allocator, &[_][]const u8{ cwd, "lib", "evm", "build.zig" });
    defer allocator.free(cwd_probe);
    if (pathExists(cwd_probe)) return cwd;

    return error.OraRepoRootNotFound;
}

fn launchDebuggerTui(
    allocator: std.mem.Allocator,
    file_path: []const u8,
    artifact_root: []const u8,
    init_signature: ?[]const u8,
    init_arg_values: []const []const u8,
    init_calldata_hex: ?[]const u8,
    signature: ?[]const u8,
    arg_values: []const []const u8,
    calldata_hex: ?[]const u8,
    gas_limit: ?[]const u8,
    max_steps: ?[]const u8,
    deploy_step_cap: ?[]const u8,
    artifact_max_bytes: ?[]const u8,
) !void {
    const repo_root = try findOraRepoRoot(allocator);
    defer allocator.free(repo_root);

    const evm_dir = try std.fs.path.join(allocator, &[_][]const u8{ repo_root, "lib", "evm" });
    defer allocator.free(evm_dir);

    const debugger_bin = try std.fs.path.join(allocator, &[_][]const u8{ evm_dir, "zig-out", "bin", "ora-evm-debug-tui" });
    defer allocator.free(debugger_bin);

    if (!pathExists(debugger_bin)) {
        var build_child = std.process.Child.init(&.{ "zig", "build", "install" }, allocator);
        build_child.cwd = evm_dir;
        build_child.stdin_behavior = .Inherit;
        build_child.stdout_behavior = .Inherit;
        build_child.stderr_behavior = .Inherit;
        try build_child.spawn();
        const build_term = try build_child.wait();
        switch (build_term) {
            .Exited => |code| if (code != 0) return error.DebuggerBuildFailed,
            else => return error.DebuggerBuildFailed,
        }
    }

    const stem = std.fs.path.stem(file_path);
    const hex_path_rel = try std.fmt.allocPrint(allocator, "{s}/{s}.hex", .{ artifact_root, stem });
    defer allocator.free(hex_path_rel);
    const srcmap_path_rel = try std.fmt.allocPrint(allocator, "{s}/{s}.sourcemap.json", .{ artifact_root, stem });
    defer allocator.free(srcmap_path_rel);
    const debug_path_rel = try std.fmt.allocPrint(allocator, "{s}/{s}.debug.json", .{ artifact_root, stem });
    defer allocator.free(debug_path_rel);
    const abi_path_rel = try std.fmt.allocPrint(allocator, "{s}/abi/{s}.abi.json", .{ artifact_root, stem });
    defer allocator.free(abi_path_rel);

    const source_abs = try absolutePathAlloc(allocator, file_path);
    defer allocator.free(source_abs);
    const hex_abs = try absolutePathAlloc(allocator, hex_path_rel);
    defer allocator.free(hex_abs);
    const srcmap_abs = try absolutePathAlloc(allocator, srcmap_path_rel);
    defer allocator.free(srcmap_abs);
    const debug_abs = try absolutePathAlloc(allocator, debug_path_rel);
    defer allocator.free(debug_abs);
    const abi_abs = try absolutePathAlloc(allocator, abi_path_rel);
    defer allocator.free(abi_abs);

    const deploy_hex_abs = try prepareDebuggerDeployHex(
        allocator,
        artifact_root,
        stem,
        hex_abs,
        if (pathExists(abi_abs)) abi_abs else null,
        init_signature,
        init_arg_values,
        init_calldata_hex,
    );
    defer allocator.free(deploy_hex_abs);

    var argv: std.ArrayList([]const u8) = .{};
    defer argv.deinit(allocator);
    try argv.appendSlice(allocator, &.{ debugger_bin, deploy_hex_abs, srcmap_abs, source_abs });
    try argv.appendSlice(allocator, &.{ "--debug-info", debug_abs });
    if (pathExists(abi_abs)) {
        try argv.appendSlice(allocator, &.{ "--abi", abi_abs });
    }
    if (signature) |sig| {
        try argv.appendSlice(allocator, &.{ "--signature", sig });
        for (arg_values) |arg| {
            try argv.appendSlice(allocator, &.{ "--arg", arg });
        }
    } else if (calldata_hex) |hex| {
        try argv.appendSlice(allocator, &.{ "--calldata-hex", hex });
    }

    if (gas_limit) |v| try argv.appendSlice(allocator, &.{ "--gas-limit", v });
    if (max_steps) |v| try argv.appendSlice(allocator, &.{ "--max-steps", v });
    if (deploy_step_cap) |v| try argv.appendSlice(allocator, &.{ "--deploy-step-cap", v });
    if (artifact_max_bytes) |v| try argv.appendSlice(allocator, &.{ "--artifact-max-bytes", v });

    var child = std.process.Child.init(argv.items, allocator);
    child.stdin_behavior = .Inherit;
    child.stdout_behavior = .Inherit;
    child.stderr_behavior = .Inherit;
    try child.spawn();
    const term = try child.wait();
    switch (term) {
        .Exited => |code| if (code != 0) std.process.exit(@intCast(code)),
        else => std.process.exit(1),
    }
}

fn prepareDebuggerDeployHex(
    allocator: std.mem.Allocator,
    artifact_root: []const u8,
    stem: []const u8,
    base_hex_abs: []const u8,
    abi_abs_opt: ?[]const u8,
    init_signature: ?[]const u8,
    init_arg_values: []const []const u8,
    init_calldata_hex: ?[]const u8,
) ![]u8 {
    if (init_signature == null and init_calldata_hex == null) {
        return try allocator.dupe(u8, base_hex_abs);
    }

    const base_hex_text = try std.fs.cwd().readFileAlloc(allocator, base_hex_abs, 16 * 1024 * 1024);
    defer allocator.free(base_hex_text);
    const base_bytes = try decodeHexAllocLocal(allocator, base_hex_text);
    defer allocator.free(base_bytes);

    const init_bytes = if (init_signature) |sig| blk: {
        if (abi_abs_opt) |abi_abs| {
            const abi_json = try std.fs.cwd().readFileAlloc(allocator, abi_abs, 16 * 1024 * 1024);
            defer allocator.free(abi_json);
            break :blk try encodeConstructorArgsAllocLocal(allocator, abi_json, sig, init_arg_values);
        }
        break :blk try encodeConstructorArgsAllocLocal(allocator, null, sig, init_arg_values);
    } else blk: {
        break :blk try decodeHexAllocLocal(allocator, init_calldata_hex.?);
    };
    defer allocator.free(init_bytes);

    const combined = try allocator.alloc(u8, base_bytes.len + init_bytes.len);
    defer allocator.free(combined);
    @memcpy(combined[0..base_bytes.len], base_bytes);
    @memcpy(combined[base_bytes.len..], init_bytes);

    const deploy_hex_name = try std.fmt.allocPrint(allocator, "{s}.deploy.hex", .{stem});
    defer allocator.free(deploy_hex_name);
    const deploy_hex_path_rel = try std.fs.path.join(allocator, &[_][]const u8{ artifact_root, deploy_hex_name });
    defer allocator.free(deploy_hex_path_rel);

    const deploy_hex_file = if (std.fs.path.isAbsolute(deploy_hex_path_rel))
        try std.fs.createFileAbsolute(deploy_hex_path_rel, .{})
    else
        try std.fs.cwd().createFile(deploy_hex_path_rel, .{});
    defer deploy_hex_file.close();

    var writer_buffer: [4096]u8 = undefined;
    var writer = deploy_hex_file.writer(&writer_buffer);
    const out = &writer.interface;
    for (combined) |byte| {
        try out.print("{X:0>2}", .{byte});
    }
    try out.flush();

    return try absolutePathAlloc(allocator, deploy_hex_path_rel);
}

fn decodeHexAllocLocal(allocator: std.mem.Allocator, text: []const u8) ![]u8 {
    const trimmed = std.mem.trim(u8, text, " \t\r\n");
    const hex = if (std.mem.startsWith(u8, trimmed, "0x")) trimmed[2..] else trimmed;
    if (hex.len % 2 != 0) return error.InvalidHex;

    const out = try allocator.alloc(u8, hex.len / 2);
    errdefer allocator.free(out);
    _ = try std.fmt.hexToBytes(out, hex);
    return out;
}

fn encodeConstructorArgsAllocLocal(
    allocator: std.mem.Allocator,
    abi_json_opt: ?[]const u8,
    signature: []const u8,
    arg_values: []const []const u8,
) ![]u8 {
    _ = abi_json_opt;
    const open = std.mem.indexOfScalar(u8, signature, '(') orelse return error.InvalidSignature;
    const close = std.mem.lastIndexOfScalar(u8, signature, ')') orelse return error.InvalidSignature;
    if (close < open) return error.InvalidSignature;

    const type_list = signature[open + 1 .. close];
    var count: usize = 0;
    if (std.mem.trim(u8, type_list, " \t").len > 0) {
        var split_count = std.mem.splitScalar(u8, type_list, ',');
        while (split_count.next()) |_| count += 1;
    }
    if (count != arg_values.len) return error.ArgumentCountMismatch;

    const out = try allocator.alloc(u8, count * 32);
    errdefer allocator.free(out);

    var arg_index: usize = 0;
    if (count > 0) {
        var split = std.mem.splitScalar(u8, type_list, ',');
        while (split.next()) |raw_type| : (arg_index += 1) {
            const type_name = std.mem.trim(u8, raw_type, " \t");
            try encodeAbiWordLocal(switchAbiTypeLocal(type_name), arg_values[arg_index], out[arg_index * 32 ..][0..32]);
        }
    }

    return out;
}

fn switchAbiTypeLocal(type_name: []const u8) []const u8 {
    if (std.mem.eql(u8, type_name, "u256")) return "uint256";
    if (std.mem.eql(u8, type_name, "bool")) return "bool";
    if (std.mem.eql(u8, type_name, "address")) return "address";
    return type_name;
}

fn encodeAbiWordLocal(type_name: []const u8, value_text: []const u8, out_word: []u8) !void {
    if (out_word.len != 32) return error.InvalidAbiType;
    @memset(out_word, 0);

    if (std.mem.eql(u8, type_name, "bool")) {
        const is_true = std.mem.eql(u8, value_text, "true") or std.mem.eql(u8, value_text, "1");
        const is_false = std.mem.eql(u8, value_text, "false") or std.mem.eql(u8, value_text, "0");
        if (!is_true and !is_false) return error.InvalidBoolean;
        out_word[31] = if (is_true) 1 else 0;
        return;
    }

    if (std.mem.eql(u8, type_name, "address")) {
        const text = if (std.mem.startsWith(u8, value_text, "0x")) value_text[2..] else value_text;
        if (text.len > 40) return error.InvalidAddress;
        if (text.len % 2 != 0) return error.InvalidAddress;
        var buf: [20]u8 = [_]u8{0} ** 20;
        const start = 20 - text.len / 2;
        _ = try std.fmt.hexToBytes(buf[start..], text);
        @memcpy(out_word[12..32], &buf);
        return;
    }

    const value = try std.fmt.parseUnsigned(u256, value_text, 0);
    var tmp = value;
    var i: usize = 0;
    while (i < 32) : (i += 1) {
        out_word[31 - i] = @truncate(tmp & 0xff);
        tmp >>= 8;
    }
}

fn freeResolvedIncludeRoots(allocator: std.mem.Allocator, include_roots: []const []const u8) void {
    for (include_roots) |include_root| {
        allocator.free(include_root);
    }
    allocator.free(include_roots);
}

fn resolveIncludeRootsForTarget(
    allocator: std.mem.Allocator,
    config_dir: []const u8,
    include_paths: []const []const u8,
) ![]const []const u8 {
    const include_roots = try allocator.alloc([]const u8, include_paths.len);
    errdefer freeResolvedIncludeRoots(allocator, include_roots);

    for (include_paths, 0..) |include_path, idx| {
        include_roots[idx] = try project_config.resolvePathFromConfigDir(allocator, config_dir, include_path);
    }

    return include_roots;
}

fn combineInitArgSlices(
    allocator: std.mem.Allocator,
    compiler_init_args: []const project_config.InitArg,
    target_init_args: []const project_config.InitArg,
) ![]project_config.InitArg {
    var combined = std.ArrayList(project_config.InitArg){};
    defer {
        for (combined.items) |arg| {
            allocator.free(arg.name);
            allocator.free(arg.value);
        }
        combined.deinit(allocator);
    }

    for (compiler_init_args) |arg| {
        try combined.append(allocator, .{
            .name = try allocator.dupe(u8, arg.name),
            .value = try allocator.dupe(u8, arg.value),
        });
    }

    for (target_init_args) |arg| {
        var replaced = false;
        for (combined.items) |*existing| {
            if (!std.mem.eql(u8, existing.name, arg.name)) continue;
            allocator.free(existing.value);
            existing.value = try allocator.dupe(u8, arg.value);
            replaced = true;
            break;
        }
        if (!replaced) {
            try combined.append(allocator, .{
                .name = try allocator.dupe(u8, arg.name),
                .value = try allocator.dupe(u8, arg.value),
            });
        }
    }

    return try combined.toOwnedSlice(allocator);
}

fn freeCombinedInitArgs(allocator: std.mem.Allocator, init_args: []project_config.InitArg) void {
    for (init_args) |arg| {
        allocator.free(arg.name);
        allocator.free(arg.value);
    }
    allocator.free(init_args);
}

fn isHexDigit(ch: u8) bool {
    return (ch >= '0' and ch <= '9') or
        (ch >= 'a' and ch <= 'f') or
        (ch >= 'A' and ch <= 'F');
}

fn isValidAddressValue(value: []const u8) bool {
    if (value.len != 42) return false;
    if (!(value[0] == '0' and value[1] == 'x')) return false;
    for (value[2..]) |ch| {
        if (!isHexDigit(ch)) return false;
    }
    return true;
}

fn validateInitArgValue(ty: compiler.sema.Type, raw_value: []const u8) !void {
    const value = std.mem.trim(u8, raw_value, " \t\r\n");
    if (value.len == 0) return error.InvalidInitArgValue;

    switch (ty) {
        .bool => {
            if (!std.mem.eql(u8, value, "true") and !std.mem.eql(u8, value, "false")) {
                return error.InvalidInitArgValue;
            }
        },
        .integer => |integer| {
            if (integer.signed == true) {
                _ = std.fmt.parseInt(i256, value, 0) catch return error.InvalidInitArgValue;
            } else {
                _ = std.fmt.parseInt(u256, value, 0) catch return error.InvalidInitArgValue;
            }
        },
        .address => {
            if (!isValidAddressValue(value)) {
                return error.InvalidInitArgValue;
            }
        },
        .string => {},
        .refinement => |refinement| try validateInitArgValue(refinement.base_type.*, value),
        else => return error.UnsupportedInitArgType,
    }
}

fn initParameterName(ast_file: *const compiler.AstFile, parameter: compiler.ast.Parameter) ?[]const u8 {
    return switch (ast_file.pattern(parameter.pattern).*) {
        .Name => |pattern| pattern.name,
        else => null,
    };
}

fn validateConfiguredInitArgs(
    allocator: std.mem.Allocator,
    file_path: []const u8,
    resolver_options: import_graph.ResolverOptions,
    configured_init_args: []const project_config.InitArg,
) !void {
    if (configured_init_args.len == 0) return;

    var compilation = try compiler.driver.compilePackageWithResolverOptions(allocator, file_path, resolver_options);
    defer compilation.deinit();

    const module = compilation.db.sources.module(compilation.root_module_id);
    const ast_file = try compilation.db.astFile(module.file_id);
    const typecheck = try compilation.db.moduleTypeCheck(compilation.root_module_id);

    var init_fn: ?compiler.ast.FunctionItem = null;
    var init_contract_name: []const u8 = "<unknown>";

    for (ast_file.root_items) |item_id| {
        const item = ast_file.item(item_id).*;
        if (item != .Contract) continue;
        const contract = item.Contract;
        for (contract.members) |member_id| {
            const member = ast_file.item(member_id).*;
            if (member != .Function) continue;
            const func = member.Function;
            if (!std.mem.eql(u8, func.name, "init")) continue;
            if (init_fn != null) {
                std.log.warn("Configured init_args for '{s}' but multiple contracts define init().", .{file_path});
                return error.AmbiguousInitArgsTarget;
            }
            init_fn = func;
            init_contract_name = contract.name;
        }
    }

    if (init_fn == null) {
        std.log.warn("Configured init_args for '{s}' but no init() function exists in the entry contract.", .{file_path});
        return error.InitArgsRequireInitFunction;
    }

    if (init_fn.?.visibility != .public) {
        std.log.warn("Configured init_args for '{s}' but {s}.init() is not public.", .{ file_path, init_contract_name });
        return error.InvalidInitFunction;
    }
    if (functionHasBareSelf(ast_file, init_fn.?)) {
        std.log.warn("Configured init_args for '{s}' but {s}.init() declares self, which constructors do not support.", .{ file_path, init_contract_name });
        return error.InvalidInitFunction;
    }
    if (init_fn.?.return_type != null) {
        std.log.warn("Configured init_args for '{s}' but {s}.init() returns values, which constructors do not support.", .{ file_path, init_contract_name });
        return error.InvalidInitFunction;
    }

    var seen_names = std.StringHashMap(void).init(allocator);
    defer seen_names.deinit();

    for (configured_init_args) |arg| {
        if (seen_names.contains(arg.name)) {
            std.log.warn("Duplicate init arg '{s}' in build configuration.", .{arg.name});
            return error.DuplicateInitArg;
        }
        try seen_names.put(arg.name, {});

        var param_opt: ?compiler.ast.Parameter = null;
        for (init_fn.?.parameters) |parameter| {
            const parameter_name = initParameterName(ast_file, parameter) orelse continue;
            if (!std.mem.eql(u8, parameter_name, arg.name)) continue;
            param_opt = parameter;
            break;
        }

        const param = param_opt orelse {
            std.log.warn("Configured init arg '{s}' is not a parameter of {s}.init().", .{ arg.name, init_contract_name });
            return error.UnknownInitArg;
        };
        if (param.is_comptime) {
            std.log.warn("Configured init arg '{s}' targets comptime parameter, which is not supported.", .{arg.name});
            return error.UnsupportedInitArgParameter;
        }
        validateInitArgValue(typecheck.pattern_types[param.pattern.index()].type, arg.value) catch |err| {
            std.log.warn("Invalid value for init arg '{s}' in {s}.init(): {s}", .{
                arg.name,
                init_contract_name,
                @errorName(err),
            });
            return err;
        };
    }
}

fn functionHasBareSelf(file: *const compiler.AstFile, function: compiler.ast.FunctionItem) bool {
    for (function.parameters) |parameter| {
        if (parameter.is_comptime) continue;
        return switch (file.pattern(parameter.pattern).*) {
            .Name => |name| std.mem.eql(u8, name.name, "self"),
            else => false,
        };
    }
    return false;
}

fn runBuildFromDiscoveredConfig(
    allocator: std.mem.Allocator,
    cli_output_dir: ?[]const u8,
    base_options: MlirOptions,
) !void {
    var stdout_buffer: [1024]u8 = undefined;
    var stdout_writer = std.fs.File.stdout().writer(&stdout_buffer);
    const stdout = &stdout_writer.interface;

    const loaded_opt = project_config.loadDiscovered(allocator) catch |err| {
        try stdout.print("error: failed to load ora.toml: {s}\n", .{@errorName(err)});
        try stdout.flush();
        std.process.exit(2);
    };
    if (loaded_opt == null) {
        try stdout.print("error: build mode without <file.ora> requires ora.toml with [[targets]]\n", .{});
        try stdout.flush();
        std.process.exit(2);
    }

    var loaded = loaded_opt.?;
    defer loaded.deinit(allocator);

    const base_output = blk: {
        if (cli_output_dir) |out| {
            break :blk try allocator.dupe(u8, out);
        }
        if (loaded.config.compiler_output_dir) |out| {
            break :blk try project_config.resolvePathFromConfigDir(allocator, loaded.config_dir, out);
        }
        break :blk try project_config.resolvePathFromConfigDir(allocator, loaded.config_dir, "artifacts");
    };
    defer allocator.free(base_output);

    const use_target_subdirs = loaded.config.targets.len > 1;

    for (loaded.config.targets) |target| {
        const root_path = try project_config.resolvePathFromConfigDir(allocator, loaded.config_dir, target.root);
        defer allocator.free(root_path);

        const include_roots = try resolveIncludeRootsForTarget(allocator, loaded.config_dir, target.include_paths);
        defer freeResolvedIncludeRoots(allocator, include_roots);
        const target_init_args = if (target.kind == .contract)
            try combineInitArgSlices(allocator, loaded.config.compiler_init_args, target.init_args)
        else
            try allocator.alloc(project_config.InitArg, 0);
        defer freeCombinedInitArgs(allocator, target_init_args);

        const target_output = blk: {
            if (target.output_dir) |out| {
                break :blk try project_config.resolvePathFromConfigDir(allocator, loaded.config_dir, out);
            }
            if (use_target_subdirs) {
                break :blk try std.fs.path.join(allocator, &.{ base_output, target.name });
            }
            break :blk try allocator.dupe(u8, base_output);
        };
        defer allocator.free(target_output);

        try stdout.print("Building target '{s}' from {s}\n", .{ target.name, root_path });
        try stdout.flush();

        try runBuildArtifacts(
            allocator,
            root_path,
            target_output,
            base_options,
            .{
                .include_roots = include_roots,
            },
            target_init_args,
        );
    }
}

fn discoverResolverOptionsForFile(
    allocator: std.mem.Allocator,
    file_path: []const u8,
) !struct {
    include_roots: ?[]const []const u8,
    options: import_graph.ResolverOptions,
} {
    const start_dir = std.fs.path.dirname(file_path) orelse ".";
    const loaded_opt = project_config.loadDiscoveredFromStartDir(allocator, start_dir) catch |err| {
        std.debug.print("error: failed to load ora.toml: {s}\n", .{@errorName(err)});
        std.process.exit(2);
    };

    if (loaded_opt == null) {
        return .{
            .include_roots = null,
            .options = .{},
        };
    }

    var loaded = loaded_opt.?;
    defer loaded.deinit(allocator);

    const target_idx_opt = project_config.findMatchingTargetIndex(allocator, &loaded, file_path) catch |err| {
        std.debug.print("error: failed to match target in ora.toml: {s}\n", .{@errorName(err)});
        std.process.exit(2);
    };

    if (target_idx_opt) |target_idx| {
        const target = loaded.config.targets[target_idx];
        const include_roots = resolveIncludeRootsForTarget(allocator, loaded.config_dir, target.include_paths) catch |err| {
            std.debug.print("error: failed to resolve include_paths for target '{s}': {s}\n", .{ target.name, @errorName(err) });
            std.process.exit(2);
        };
        return .{
            .include_roots = include_roots,
            .options = .{
                .include_roots = include_roots,
            },
        };
    }

    return .{
        .include_roots = null,
        .options = .{},
    };
}

// ============================================================================
// SECTION 2: Usage & Help Text
// ============================================================================

fn printUsage() !void {
    var stdout_buffer: [1024]u8 = undefined;
    var stdout_writer = std.fs.File.stdout().writer(&stdout_buffer);
    const stdout = &stdout_writer.interface;
    try stdout.print("Ora Compiler v0.1 - Asuka\n", .{});
    try stdout.print("Usage: ora <file.ora>\n", .{});
    try stdout.print("       ora build [options]\n", .{});
    try stdout.print("       ora build [options] <file.ora>\n", .{});
    try stdout.print("       ora emit [emit-options] <file.ora>\n", .{});
    try stdout.print("       ora debug [debug-options] <file.ora>\n", .{});
    try stdout.print("       ora fmt [fmt-options] <file.ora|dir>\n", .{});
    try stdout.print("       ora init [path]\n", .{});
    try stdout.print("       ora -v | --version\n", .{});
    try stdout.print("\nCompilation Control:\n", .{});
    try stdout.print("  (default), build       - Full compile + artifact bundle + SMT gate\n", .{});
    try stdout.print("                           If <file.ora> is omitted, builds all [[targets]] from ora.toml\n", .{});
    try stdout.print("                           Outputs: bytecode, ABI, Ora/SIR MLIR, SIR text, SMT report\n", .{});
    try stdout.print("                           Reads ora.toml [compiler].init_args and [[targets]].init_args\n", .{});
    try stdout.print("  emit                   - Debug emission mode (use --emit-*)\n", .{});
    try stdout.print("  debug                  - Compile debugger artifacts and launch the Ora EVM debugger\n", .{});
    try stdout.print("  init [path]            - Scaffold a new Ora project directory\n", .{});
    try stdout.print("  --emit-tokens          - Stop after lexical analysis (emit tokens)\n", .{});
    try stdout.print("  --emit-ast             - Stop after parsing (emit AST)\n", .{});
    try stdout.print("  --emit-ast=json|tree   - Emit AST in JSON or tree format\n", .{});
    try stdout.print("  --emit-typed-ast       - Stop after parsing (emit typed AST)\n", .{});
    try stdout.print("  --emit-typed-ast=json|tree - Emit typed AST in JSON or tree format\n", .{});
    try stdout.print("  --emit-mlir[=ora|sir|both] - Emit MLIR (default mode: ora)\n", .{});
    try stdout.print("  --emit-sir-text        - Emit Sensei SIR text IR (after conversion)\n", .{});
    try stdout.print("  --emit-bytecode        - Emit EVM bytecode from Sensei SIR text\n", .{});
    try stdout.print("  --emit-cfg[=ora|sir]   - Generate control flow graph (default: ora)\n", .{});
    try stdout.print("  -v, --version          - Show version and logo\n", .{});
    try stdout.print("\nOutput Options:\n", .{});
    try stdout.print("  -o <dir>               - Build mode: artifact root directory (default: artifacts/<name>)\n", .{});
    try stdout.print("  -o <file|dir>          - Emit mode: output file/dir\n", .{});
    try stdout.print("\nOptimization Options:\n", .{});
    try stdout.print("  -O0, -Onone            - No optimization (default)\n", .{});
    try stdout.print("  -O1, -Obasic           - Basic optimizations\n", .{});
    try stdout.print("  -O2, -Oaggressive      - Aggressive optimizations\n", .{});
    try stdout.print("\nMLIR Options:\n", .{});
    try stdout.print("  --no-validate-mlir     - Disable automatic MLIR validation (not recommended)\n", .{});
    try stdout.print("  --no-canonicalize      - Skip Ora MLIR canonicalization pass\n", .{});
    try stdout.print("  --cpp-lowering-stub    - Use experimental C++ lowering stub (contract+func)\n", .{});
    try stdout.print("  --mlir-pass-pipeline <pipeline> - Run custom MLIR pass pipeline on Ora MLIR\n", .{});
    try stdout.print("  --mlir-verify-each-pass - Enable verifier while running custom pass pipeline\n", .{});
    try stdout.print("  --mlir-pass-timing     - Enable MLIR pass timing for custom pass pipeline\n", .{});
    try stdout.print("  --mlir-pass-statistics - Print operation-count stats across MLIR stages\n", .{});
    try stdout.print("  --mlir-print-ir=before|after|before-after|all - Print stage MLIR snapshots\n", .{});
    try stdout.print("  --mlir-print-ir-pass <stage-filter> - Filter snapshots by stage name\n", .{});
    try stdout.print("  --mlir-crash-reproducer <path> - Save current MLIR when a stage fails\n", .{});
    try stdout.print("  --mlir-print-op-on-diagnostic - Print module snapshot with MLIR diagnostics\n", .{});
    try stdout.print("\nAnalysis Options:\n", .{});
    try stdout.print("  --verify               - Run Z3 verification on MLIR annotations (default)\n", .{});
    try stdout.print("  --verify=basic|full    - Verification mode (default: full)\n", .{});
    try stdout.print("  --no-verify            - Disable Z3 verification (emit mode only)\n", .{});
    try stdout.print("  --verify-calls         - Enable call reasoning in Z3 encoder (default)\n", .{});
    try stdout.print("  --no-verify-calls      - Disable call reasoning (treat calls as unknown)\n", .{});
    try stdout.print("  --verify-state         - Enable storage/map state threading (default)\n", .{});
    try stdout.print("  --no-verify-state      - Disable state threading (treat loads as unknown)\n", .{});
    try stdout.print("  --verify-stats         - Print Z3 query stats summary\n", .{});
    try stdout.print("  --explain              - Enable unsat-core explain mode for SMT verification\n", .{});
    try stdout.print("  --z3-proofs            - Emit raw Z3 proof objects in SMT reports/debug output (slower)\n", .{});
    try stdout.print("  --minimize-cores       - Greedily minimize explain-mode unsat cores; implies --explain\n", .{});
    try stdout.print("  --emit-smt-report      - Emit SMT encoding audit report (.md + .json)\n", .{});
    try stdout.print("  --debug                - Enable compiler debug output\n", .{});
    try stdout.print("  --debug-info           - Preserve source-stable lowering for debugger artifacts\n", .{});
    try stdout.print("  --metrics              - Print compilation phase timing report\n", .{});
    try stdout.print("\nDebugger Launch Options:\n", .{});
    try stdout.print("  --signature <sig>      - Contract function signature, e.g. 'add(u256,u256)'\n", .{});
    try stdout.print("  --arg <value>          - Repeated function arguments for --signature\n", .{});
    try stdout.print("  --calldata-hex <hex>   - Raw calldata (alternative to --signature/--arg)\n", .{});
    try stdout.print("  --init-signature <sig> - Constructor/init signature, e.g. 'init(u256)'\n", .{});
    try stdout.print("  --init-arg <value>     - Repeated constructor/init arguments\n", .{});
    try stdout.print("  --init-calldata-hex <hex> - Raw constructor/init calldata\n", .{});
    try stdout.print("  --no-tui               - Emit debug artifacts and exit (no TUI launch)\n", .{});
    try stdout.print("  --gas-limit <i64>      - Frame gas budget (default 5000000)\n", .{});
    try stdout.print("  --max-steps <u64>      - Per-command opcode safety cap (default 10000000)\n", .{});
    try stdout.print("  --deploy-step-cap <usize> - Deployment opcode cap (default 200000)\n", .{});
    try stdout.print("  --artifact-max-bytes <usize> - Per-file artifact load cap (default 16777216)\n", .{});
    try stdout.flush();
}

fn compilerDiagnosticSeverityName(severity: compiler.diagnostics.Severity) []const u8 {
    return switch (severity) {
        .Error => "error",
        .Warning => "warning",
        .Note => "note",
        .Help => "help",
    };
}

fn compilerDiagnosticsHasErrors(diagnostics_list: *const compiler.diagnostics.DiagnosticList) bool {
    for (diagnostics_list.items.items) |diag| {
        if (diag.severity == .Error) return true;
    }
    return false;
}

fn sourceLineBounds(text: []const u8, line_starts: []const u32, line_index: usize) struct { start: usize, end: usize } {
    const start: usize = line_starts[line_index];
    const next_start: usize = if (line_index + 1 < line_starts.len) line_starts[line_index + 1] else text.len;
    var end = next_start;
    while (end > start and (text[end - 1] == '\n' or text[end - 1] == '\r')) : (end -= 1) {}
    return .{ .start = start, .end = end };
}

fn writeCompilerDiagnosticSnippet(
    writer: anytype,
    sources: ?*const compiler.source.SourceStore,
    labels: []const compiler.diagnostics.Label,
) !void {
    if (sources == null or labels.len == 0) return;

    const primary = labels[0];
    const file = sources.?.file(primary.location.file_id);
    const line_column = sources.?.lineColumn(primary.location);
    const line_index: usize = @intCast(line_column.line - 1);
    const bounds = sourceLineBounds(file.text, file.line_starts, line_index);
    const line_text = file.text[bounds.start..bounds.end];
    const line_no_width = std.fmt.count("{d}", .{line_column.line});
    const caret_column: usize = @intCast(line_column.column);
    const span_len = @max(primary.location.range.len(), 1);
    const caret_len: usize = @intCast(span_len);

    try writer.print("  --> {s}:{d}:{d}\n", .{ file.path, line_column.line, line_column.column });
    for (0..line_no_width) |_| try writer.writeByte(' ');
    try writer.writeAll(" |\n");
    const line_number_text = try std.fmt.allocPrint(std.heap.page_allocator, "{d}", .{line_column.line});
    defer std.heap.page_allocator.free(line_number_text);
    for (0..line_no_width - line_number_text.len) |_| try writer.writeByte(' ');
    try writer.print("{s} | {s}\n", .{ line_number_text, line_text });
    for (0..line_no_width) |_| try writer.writeByte(' ');
    try writer.writeAll(" | ");
    for (1..caret_column) |_| try writer.writeByte(' ');
    for (0..caret_len) |_| try writer.writeByte('^');
    if (primary.message.len != 0) {
        try writer.print(" {s}", .{primary.message});
    }
    try writer.writeByte('\n');
}

// ---------------------------------------------------------------------------
// Z3 verification output formatting
// ---------------------------------------------------------------------------

const z3_errors = @import("z3/errors.zig");

/// Parse guard_id format: "guard:{path}:{line}:{col}:{len}:{refinement_kind}:{var_name}"
const ParsedGuard = struct {
    file_path: []const u8,
    line: []const u8,
    column: []const u8,
    refinement_kind: []const u8,
    variable_name: []const u8,
};

fn parseGuardId(guard_id: []const u8) ?ParsedGuard {
    // Strip "guard:" prefix.
    const rest = if (std.mem.startsWith(u8, guard_id, "guard:")) guard_id[6..] else return null;

    // Split from the end — last field is var_name, second-last is refinement_kind, etc.
    // Format: {path}:{line}:{col}:{len}:{kind}:{name}
    // Path may contain colons (unlikely on unix but be safe) — parse from the right.
    const name_sep = std.mem.lastIndexOfScalar(u8, rest, ':') orelse return null;
    const variable_name = rest[name_sep + 1 ..];
    const before_name = rest[0..name_sep];

    const kind_sep = std.mem.lastIndexOfScalar(u8, before_name, ':') orelse return null;
    const refinement_kind = before_name[kind_sep + 1 ..];
    const before_kind = before_name[0..kind_sep];

    const len_sep = std.mem.lastIndexOfScalar(u8, before_kind, ':') orelse return null;
    const before_len = before_kind[0..len_sep];

    const col_sep = std.mem.lastIndexOfScalar(u8, before_len, ':') orelse return null;
    const column = before_len[col_sep + 1 ..];
    const before_col = before_len[0..col_sep];

    const line_sep = std.mem.lastIndexOfScalar(u8, before_col, ':') orelse return null;
    const line = before_col[line_sep + 1 ..];
    const file_path = before_col[0..line_sep];

    return .{
        .file_path = file_path,
        .line = line,
        .column = column,
        .refinement_kind = refinement_kind,
        .variable_name = variable_name,
    };
}

fn shortPath(path: []const u8) []const u8 {
    // Show just the filename, or last two components if nested.
    if (std.mem.lastIndexOfScalar(u8, path, '/')) |last_slash| {
        const parent = path[0..last_slash];
        if (std.mem.lastIndexOfScalar(u8, parent, '/')) |prev_slash| {
            return path[prev_slash + 1 ..];
        }
        return path[last_slash + 1 ..];
    }
    return path;
}

fn formatHexValue(raw: []const u8) []const u8 {
    // Z3 outputs #x0000...0000 — show as 0x0, 0x1, or abbreviated.
    if (!std.mem.startsWith(u8, raw, "#x")) return raw;
    const hex = raw[2..];
    // Strip leading zeros.
    var i: usize = 0;
    while (i < hex.len and hex[i] == '0') : (i += 1) {}
    if (i == hex.len) return "0x0";
    // If short enough, show full value.
    const significant = hex[i..];
    if (significant.len <= 10) return raw; // keep original
    return raw; // keep full for now
}

fn printVerificationDiagnostics(stdout: anytype, diagnostics: []const z3_errors.Diagnostic) !void {
    // Group diagnostics by function name.
    try stdout.print("\n", .{});
    try stdout.print("Verification Report: {d} refinement guard{s} can be violated\n", .{
        diagnostics.len,
        if (diagnostics.len != 1) @as([]const u8, "s") else "",
    });
    try stdout.print("{s}\n", .{"=" ** 70});

    for (diagnostics, 0..) |diag, idx| {
        const parsed = parseGuardId(diag.guard_id);

        if (parsed) |g| {
            try stdout.print("\n  {d}. {s}({s}) in {s}  [{s}:{s}]\n", .{
                idx + 1,
                g.refinement_kind,
                g.variable_name,
                diag.function_name,
                shortPath(g.file_path),
                g.line,
            });
        } else {
            try stdout.print("\n  {d}. {s} in {s}\n", .{ idx + 1, diag.guard_id, diag.function_name });
        }

        // Print counterexample values.
        var it = diag.counterexample.variables.iterator();
        var has_vars = false;
        while (it.next()) |entry| {
            if (!has_vars) {
                try stdout.print("     counterexample:\n", .{});
                has_vars = true;
            }
            try stdout.print("       {s} = {s}\n", .{ entry.key_ptr.*, formatHexValue(entry.value_ptr.*) });
        }

        // Print user-friendly explanation.
        if (parsed) |g| {
            if (std.mem.eql(u8, g.refinement_kind, "NonZeroAddress")) {
                try stdout.print("     -> `{s}` can be the zero address\n", .{g.variable_name});
            } else if (std.mem.eql(u8, g.refinement_kind, "MinValue")) {
                try stdout.print("     -> `{s}` can be zero (below minimum)\n", .{g.variable_name});
            } else if (std.mem.eql(u8, g.refinement_kind, "MaxValue")) {
                try stdout.print("     -> `{s}` can exceed maximum value\n", .{g.variable_name});
            } else if (std.mem.eql(u8, g.refinement_kind, "InRange")) {
                try stdout.print("     -> `{s}` can be out of range\n", .{g.variable_name});
            }
        }
    }

    try stdout.print("\n{s}\n", .{"-" ** 70});
    // Print raw guard IDs for debugging.
    try stdout.print("debug: raw guard IDs:\n", .{});
    for (diagnostics) |diag| {
        try stdout.print("  {s}\n", .{diag.guard_id});
    }
}

fn printRuntimeGuardSummary(stdout: anytype, diagnostics: []const z3_errors.Diagnostic) !void {
    try stdout.print("\nRefinement guards: {d} kept as runtime check{s}\n", .{
        diagnostics.len,
        if (diagnostics.len != 1) @as([]const u8, "s") else "",
    });
    for (diagnostics) |diag| {
        const parsed = parseGuardId(diag.guard_id);
        if (parsed) |g| {
            try stdout.print("  {s}({s}) in {s}  [{s}:{s}] — not statically provable, runtime check emitted\n", .{
                g.refinement_kind,
                g.variable_name,
                diag.function_name,
                shortPath(g.file_path),
                g.line,
            });
        } else {
            try stdout.print("  {s} in {s} — runtime check emitted\n", .{ diag.guard_id, diag.function_name });
        }
    }
}

fn printVerificationErrors(stdout: anytype, errors: []const z3_errors.VerificationError) !void {
    var degraded_count: usize = 0;
    for (errors) |err| {
        if (err.error_type == .EncodingDegraded) degraded_count += 1;
    }

    if (degraded_count == errors.len and errors.len > 0) {
        try stdout.print("\n❌ Verification aborted with {d} SMT encoding degradation error{s}:\n", .{
            errors.len,
            if (errors.len != 1) @as([]const u8, "s") else "",
        });
    } else {
        try stdout.print("\n❌ Verification failed with {d} error{s}:\n", .{
            errors.len,
            if (errors.len != 1) @as([]const u8, "s") else "",
        });
    }
    try stdout.print("{s}\n", .{"=" ** 70});

    for (errors, 0..) |err, idx| {
        try stdout.print("\n  {d}. {s}\n", .{ idx + 1, err.message });

        // Show source location and snippet if available
        if (err.file.len > 0 and err.line > 0) {
            try stdout.print("     --> {s}:{d}:{d}\n", .{ shortPath(err.file), err.line, err.column });
            if (readSourceLineFromFile(err.file, err.line)) |source_line| {
                const line_no_str_len = std.fmt.count("{d}", .{err.line});
                try stdout.print("      ", .{});
                for (0..line_no_str_len) |_| try stdout.print(" ", .{});
                try stdout.print("|\n", .{});
                try stdout.print("   {d} | {s}\n", .{ err.line, source_line });
                try stdout.print("      ", .{});
                for (0..line_no_str_len) |_| try stdout.print(" ", .{});
                try stdout.print("|\n", .{});
            }
        }

        if (err.counterexample) |ce| {
            var it = ce.variables.iterator();
            var has_vars = false;
            while (it.next()) |entry| {
                if (!has_vars) {
                    try stdout.print("     counterexample:\n", .{});
                    has_vars = true;
                }
                try stdout.print("       {s} = {s}\n", .{ prettifyVariableName(entry.key_ptr.*), formatHexValue(entry.value_ptr.*) });
            }
        }
    }
    try stdout.print("\n", .{});
}

/// Read a single line from a source file by line number (1-based).
/// Returns null if the file can't be read or the line doesn't exist.
fn readSourceLineFromFile(file_path: []const u8, line_number: u32) ?[]const u8 {
    const file = std.fs.cwd().openFile(file_path, .{}) catch return null;
    defer file.close();
    const content = file.readToEndAlloc(std.heap.page_allocator, 1024 * 1024) catch return null;
    // Note: we leak this allocation — it's diagnostic output, called once.
    var current_line: u32 = 1;
    var line_start: usize = 0;
    for (content, 0..) |ch, i| {
        if (ch == '\n') {
            if (current_line == line_number) {
                const line = content[line_start..i];
                // Trim trailing whitespace
                return std.mem.trimRight(u8, line, " \t\r");
            }
            current_line += 1;
            line_start = i + 1;
        }
    }
    // Last line (no trailing newline)
    if (current_line == line_number and line_start < content.len) {
        return std.mem.trimRight(u8, content[line_start..], " \t\r\n");
    }
    return null;
}

/// Make counterexample variable names more readable.
/// - "g_balance" stays as "balance" (strip g_ prefix for globals)
/// - "v_12345678" stays as-is (we don't have source names for these yet)
/// - "env_evm_caller" becomes "msg.sender"
/// - "struct_instantiate_123" becomes "struct"
fn prettifyVariableName(name: []const u8) []const u8 {
    if (std.mem.startsWith(u8, name, "env_evm_caller")) return "msg.sender";
    if (std.mem.startsWith(u8, name, "env_evm_origin")) return "tx.origin";
    if (std.mem.startsWith(u8, name, "env_evm_callvalue")) return "msg.value";
    if (std.mem.startsWith(u8, name, "env_evm_timestamp")) return "block.timestamp";
    if (std.mem.startsWith(u8, name, "g_")) return name[2..];
    if (std.mem.startsWith(u8, name, "struct_instantiate_")) return "struct";
    if (std.mem.startsWith(u8, name, "struct_field_update_")) return "struct_update";
    return name;
}

fn verifyMlirModule(stdout: anytype, module: @import("mlir_c_api").c.MlirModule, stage: []const u8) !void {
    const mlir_c = @import("mlir_c_api").c;
    const module_op = mlir_c.oraModuleGetOperation(module);
    if (mlir_c.mlirOperationVerify(module_op)) return;
    try stdout.print("error: internal compiler error: generated {s} is invalid\n", .{stage});
    try stdout.flush();
    std.process.exit(2);
}

fn writeCompilerDiagnosticSecondaryLabels(
    writer: anytype,
    sources: ?*const compiler.source.SourceStore,
    labels: []const compiler.diagnostics.Label,
) !void {
    if (sources == null or labels.len <= 1) return;

    for (labels[1..]) |label| {
        const line_column = sources.?.lineColumn(label.location);
        const file = sources.?.file(label.location.file_id);
        try writer.print("  = note: {s}:{d}:{d}", .{ file.path, line_column.line, line_column.column });
        if (label.message.len != 0) {
            try writer.print(": {s}", .{label.message});
        }
        try writer.writeByte('\n');
    }
}

fn writeCompilerDiagnosticsText(
    writer: anytype,
    sources: ?*const compiler.source.SourceStore,
    diagnostics_list: *const compiler.diagnostics.DiagnosticList,
    debug_enabled: bool,
) !void {
    if (diagnostics_list.items.items.len == 0) return;
    try writer.print("Diagnostics: {d}\n", .{diagnostics_list.items.items.len});
    for (diagnostics_list.items.items) |diag| {
        try writer.print("{s}: {s}\n", .{ compilerDiagnosticSeverityName(diag.severity), diag.message });
        if (debug_enabled) {
            if (diag.debug_detail) |detail| {
                try writer.print("  = debug: {s}\n", .{detail});
            }
        }
        try writeCompilerDiagnosticSnippet(writer, sources, diag.labels);
        try writeCompilerDiagnosticSecondaryLabels(writer, sources, diag.labels);
    }
}

fn exitOnCompilerErrors(
    writer: anytype,
    sources: ?*const compiler.source.SourceStore,
    diagnostics_list: *const compiler.diagnostics.DiagnosticList,
    debug_enabled: bool,
) !void {
    if (!compilerDiagnosticsHasErrors(diagnostics_list)) return;
    try writeCompilerDiagnosticsText(writer, sources, diagnostics_list, debug_enabled);
    try writer.flush();
    std.process.exit(1);
}

fn exitOnCompilationErrors(
    writer: anytype,
    db: *compiler.db.CompilerDb,
    module_id: compiler.source.ModuleId,
    debug_enabled: bool,
) !*const compiler.sema.TypeCheckResult {
    const module = db.sources.module(module_id);

    const syntax_diags = try db.syntaxDiagnostics(module.file_id);
    try exitOnCompilerErrors(writer, &db.sources, syntax_diags, debug_enabled);

    const ast_diags = try db.astDiagnostics(module.file_id);
    try exitOnCompilerErrors(writer, &db.sources, ast_diags, debug_enabled);

    const resolution_diags = try db.resolutionDiagnostics(module_id);
    try exitOnCompilerErrors(writer, &db.sources, resolution_diags, debug_enabled);

    const module_typecheck = try db.moduleTypeCheck(module_id);
    try exitOnCompilerErrors(writer, &db.sources, &module_typecheck.diagnostics, debug_enabled);
    return module_typecheck;
}

fn writeJsonString(writer: anytype, text: []const u8) !void {
    try writer.writeByte('"');
    for (text) |c| switch (c) {
        '"' => try writer.writeAll("\\\""),
        '\\' => try writer.writeAll("\\\\"),
        '\n' => try writer.writeAll("\\n"),
        '\r' => try writer.writeAll("\\r"),
        '\t' => try writer.writeAll("\\t"),
        else => {
            if (c < 32) {
                try writer.print("\\u{X:0>4}", .{@as(u32, c)});
            } else {
                try writer.writeByte(c);
            }
        },
    };
    try writer.writeByte('"');
}

const DebugLocalInfo = struct {
    id: u32,
    scope_id: u32,
    file_id: compiler.FileId,
    name: []const u8,
    kind: []const u8,
    binding_kind: ?[]const u8,
    storage_class: ?[]const u8,
    decl_range: compiler.TextRange,
    live_range: compiler.TextRange,
    runtime_kind: []const u8,
    runtime_name: ?[]const u8,
    runtime_location_kind: ?[]const u8,
    runtime_location_root: ?[]const u8,
    runtime_location_slot: ?u64,
    editable: bool,
    folded_value: ?[]const u8,
    is_folded: bool,
};

const DebugScopeInfo = struct {
    id: u32,
    parent: ?u32,
    file_id: compiler.FileId,
    file_path: []const u8,
    function_name: []const u8,
    contract_name: ?[]const u8,
    kind: []const u8,
    label: ?[]const u8,
    range: compiler.TextRange,
    local_start: usize,
    local_count: usize,
};

const DebugSourceScopeBundle = struct {
    scopes: []const DebugScopeInfo,
    locals: []const DebugLocalInfo,
};

fn deinitDebugSourceScopeBundle(allocator: std.mem.Allocator, bundle: *const DebugSourceScopeBundle) void {
    for (bundle.locals) |local| {
        if (local.folded_value) |folded_value| allocator.free(folded_value);
    }
    allocator.free(bundle.scopes);
    allocator.free(bundle.locals);
}

const ScopeBuildState = struct {
    next_scope_id: u32 = 0,
    next_local_id: u32 = 0,
};

const DebugGlobalSlots = std.StringHashMap(u64);

fn deinitDebugGlobalSlots(allocator: std.mem.Allocator, slots: *DebugGlobalSlots) void {
    var it = slots.iterator();
    while (it.next()) |entry| {
        allocator.free(entry.key_ptr.*);
    }
    slots.deinit();
}

const ExtraScopeBinding = struct {
    pattern_id: ?compiler.PatternId = null,
    name: ?[]const u8 = null,
    kind: []const u8,
    binding_kind: ?compiler.ast.BindingKind = null,
    storage_class: ?compiler.ast.StorageClass = null,
    decl_range: compiler.TextRange,
    live_range: compiler.TextRange,
    runtime_kind: []const u8 = "ssa",
    runtime_name: ?[]const u8 = null,
    runtime_location_kind: ?[]const u8 = null,
    runtime_location_root: ?[]const u8 = null,
    runtime_location_slot: ?u64 = null,
    editable: bool = false,
    folded_value: ?[]const u8 = null,
    is_folded: bool = false,
};

fn formatConstDebugValue(allocator: std.mem.Allocator, value: compiler.sema.ConstValue) ![]const u8 {
    return switch (value) {
        .integer => |integer| try integer.toString(allocator, 10, .lower),
        .boolean => |boolean| try allocator.dupe(u8, if (boolean) "true" else "false"),
        .address => |address| try std.fmt.allocPrint(allocator, "0x{x:0>40}", .{address}),
        .string => |text| try std.fmt.allocPrint(allocator, "\"{s}\"", .{text}),
        .tuple => |elements| blk: {
            var parts: std.ArrayList([]const u8) = .{};
            defer {
                for (parts.items) |part| allocator.free(part);
                parts.deinit(allocator);
            }
            for (elements) |element| {
                try parts.append(allocator, try formatConstDebugValue(allocator, element));
            }
            var out = std.ArrayList(u8){};
            errdefer out.deinit(allocator);
            try out.append(allocator, '(');
            for (parts.items, 0..) |part, index| {
                if (index != 0) try out.appendSlice(allocator, ", ");
                try out.appendSlice(allocator, part);
            }
            try out.append(allocator, ')');
            break :blk try out.toOwnedSlice(allocator);
        },
    };
}

fn debugBindingKindName(kind: compiler.ast.BindingKind) []const u8 {
    return switch (kind) {
        .let_ => "let",
        .var_ => "var",
        .constant => "const",
        .immutable => "immutable",
    };
}

fn debugStorageClassName(storage_class: compiler.ast.StorageClass) ?[]const u8 {
    return switch (storage_class) {
        .none => null,
        .storage => "storage",
        .memory => "memory",
        .tstore => "tstore",
    };
}

fn debugBindingEditable(
    storage_class: compiler.ast.StorageClass,
    runtime_location_slot: ?u64,
) bool {
    return switch (storage_class) {
        .storage, .memory, .tstore => runtime_location_slot != null,
        .none => false,
    };
}

fn debugRuntimeKindForStorageClass(
    storage_class: compiler.ast.StorageClass,
    runtime_location_slot: ?u64,
) []const u8 {
    return switch (storage_class) {
        .storage => "storage_field",
        .memory => if (runtime_location_slot != null) "memory_field" else "opaque_memory_field",
        .tstore => "tstore_field",
        .none => "ssa",
    };
}

fn debugRuntimeLocationKindForStorageClass(
    storage_class: compiler.ast.StorageClass,
    runtime_location_slot: ?u64,
) ?[]const u8 {
    if (runtime_location_slot == null) return null;
    return switch (storage_class) {
        .storage => "storage_root",
        .memory => "memory_root",
        .tstore => "tstore_root",
        .none => null,
    };
}

fn debugRuntimeLocationRootForStorageClass(
    storage_class: compiler.ast.StorageClass,
    runtime_location_slot: ?u64,
    root_name: ?[]const u8,
) ?[]const u8 {
    if (runtime_location_slot == null) return null;
    return switch (storage_class) {
        .storage, .memory, .tstore => root_name,
        .none => null,
    };
}

fn lineColumnForOffset(sources: *const compiler.source.SourceStore, file_id: compiler.FileId, offset: u32) compiler.source.LineColumn {
    return sources.lineColumn(.{ .file_id = file_id, .range = .empty(offset) });
}

fn writeDebugRangeJson(
    writer: anytype,
    sources: *const compiler.source.SourceStore,
    file_id: compiler.FileId,
    range: compiler.TextRange,
) !void {
    const start = lineColumnForOffset(sources, file_id, range.start);
    const end = lineColumnForOffset(sources, file_id, range.end);
    try writer.print(
        "{{\"start\":{{\"line\":{d},\"col\":{d},\"offset\":{d}}},\"end\":{{\"line\":{d},\"col\":{d},\"offset\":{d}}}}}",
        .{ start.line, start.column, range.start, end.line, end.column, range.end },
    );
}

fn appendPatternDebugLocals(
    allocator: std.mem.Allocator,
    ast_file: *const compiler.ast.AstFile,
    file_id: compiler.FileId,
    scope_id: u32,
    pattern_id: compiler.PatternId,
    kind: []const u8,
    binding_kind: ?compiler.ast.BindingKind,
    storage_class: ?compiler.ast.StorageClass,
    decl_range: compiler.TextRange,
    live_range: compiler.TextRange,
    runtime_kind: []const u8,
    runtime_name: ?[]const u8,
    runtime_location_kind: ?[]const u8,
    runtime_location_root: ?[]const u8,
    runtime_location_slot: ?u64,
    editable: bool,
    folded_value: ?[]const u8,
    is_folded: bool,
    next_local_id: *u32,
    locals: *std.ArrayList(DebugLocalInfo),
) !void {
    switch (ast_file.pattern(pattern_id).*) {
        .Name => |name| {
            try locals.append(allocator, .{
                .id = next_local_id.*,
                .scope_id = scope_id,
                .file_id = file_id,
                .name = name.name,
                .kind = kind,
                .binding_kind = if (binding_kind) |value| debugBindingKindName(value) else null,
                .storage_class = if (storage_class) |value| debugStorageClassName(value) else null,
                .decl_range = decl_range,
                .live_range = live_range,
                .runtime_kind = runtime_kind,
                .runtime_name = runtime_name,
                .runtime_location_kind = runtime_location_kind,
                .runtime_location_root = runtime_location_root,
                .runtime_location_slot = runtime_location_slot,
                .editable = editable,
                .folded_value = folded_value,
                .is_folded = is_folded,
            });
            next_local_id.* += 1;
        },
        .StructDestructure => |destructure| {
            for (destructure.fields) |field| {
                try appendPatternDebugLocals(
                    allocator,
                    ast_file,
                    file_id,
                    scope_id,
                    field.binding,
                    kind,
                    binding_kind,
                    storage_class,
                    field.range,
                    live_range,
                    runtime_kind,
                    runtime_name,
                    runtime_location_kind,
                    runtime_location_root,
                    runtime_location_slot,
                    editable,
                    folded_value,
                    is_folded,
                    next_local_id,
                    locals,
                );
            }
        },
        .Field => |field| try appendPatternDebugLocals(
            allocator,
            ast_file,
            file_id,
            scope_id,
            field.base,
            kind,
            binding_kind,
            storage_class,
            decl_range,
            live_range,
            runtime_kind,
            runtime_name,
            runtime_location_kind,
            runtime_location_root,
            runtime_location_slot,
            editable,
            folded_value,
            is_folded,
            next_local_id,
            locals,
        ),
        .Index => |index| try appendPatternDebugLocals(
            allocator,
            ast_file,
            file_id,
            scope_id,
            index.base,
            kind,
            binding_kind,
            storage_class,
            decl_range,
            live_range,
            runtime_kind,
            runtime_name,
            runtime_location_kind,
            runtime_location_root,
            runtime_location_slot,
            editable,
            folded_value,
            is_folded,
            next_local_id,
            locals,
        ),
        .Error => {},
    }
}

fn collectBodyScopeDebugInfo(
    allocator: std.mem.Allocator,
    db: *compiler.CompilerDb,
    const_eval: *const compiler.sema.ConstEvalResult,
    global_slots: *const DebugGlobalSlots,
    ast_file: *const compiler.ast.AstFile,
    file_id: compiler.FileId,
    function_name: []const u8,
    contract_name: ?[]const u8,
    body_id: compiler.BodyId,
    parent_scope_id: ?u32,
    kind: []const u8,
    label: ?[]const u8,
    extra_bindings: []const ExtraScopeBinding,
    state: *ScopeBuildState,
    scopes: *std.ArrayList(DebugScopeInfo),
    locals: *std.ArrayList(DebugLocalInfo),
) !void {
    const body = ast_file.body(body_id).*;
    const scope_id = state.next_scope_id;
    state.next_scope_id += 1;
    const local_start = locals.items.len;

    for (extra_bindings) |binding| {
        if (binding.pattern_id) |pattern_id| {
            try appendPatternDebugLocals(
                allocator,
                ast_file,
                file_id,
                scope_id,
                pattern_id,
                binding.kind,
                binding.binding_kind,
                binding.storage_class,
                binding.decl_range,
                binding.live_range,
                binding.runtime_kind,
                binding.runtime_name,
                binding.runtime_location_kind,
                binding.runtime_location_root,
                binding.runtime_location_slot,
                binding.editable,
                binding.folded_value,
                binding.is_folded,
                &state.next_local_id,
                locals,
            );
        } else if (binding.name) |name| {
            try locals.append(allocator, .{
                .id = state.next_local_id,
                .scope_id = scope_id,
                .file_id = file_id,
                .name = name,
                .kind = binding.kind,
                .binding_kind = if (binding.binding_kind) |value| debugBindingKindName(value) else null,
                .storage_class = if (binding.storage_class) |value| debugStorageClassName(value) else null,
                .decl_range = binding.decl_range,
                .live_range = binding.live_range,
                .runtime_kind = binding.runtime_kind,
                .runtime_name = binding.runtime_name,
                .runtime_location_kind = binding.runtime_location_kind,
                .runtime_location_root = binding.runtime_location_root,
                .runtime_location_slot = binding.runtime_location_slot,
                .editable = binding.editable,
                .folded_value = binding.folded_value,
                .is_folded = binding.is_folded,
            });
            state.next_local_id += 1;
        }
    }

    for (body.statements) |statement_id| {
        switch (ast_file.statement(statement_id).*) {
            .VariableDecl => |decl| {
                const folded_value = if (decl.value) |expr_id|
                    if (const_eval.values[expr_id.index()]) |value|
                        try formatConstDebugValue(allocator, value)
                    else
                        null
                else
                    null;
                // B3 (statement-level liveness): the live range below
                // is intentionally coarse — every `let` is treated as
                // live for the rest of its enclosing body, regardless
                // of where its actual last use is. The TUI's
                // `bindingPastLiveRange` check therefore only fires
                // at the body's closing brace. A future pass should
                // run a backward dataflow walk over `body.statements`
                // collecting AST name references, then narrow the
                // end position to `statements[last_use_index].range.end`.
                // No name-ref walker exists in the AST today; adding
                // one is the prerequisite work. See
                // tests/debug_artifacts/liveness_dead_after_use as
                // the regression target.
                try appendPatternDebugLocals(
                    allocator,
                    ast_file,
                    file_id,
                    scope_id,
                    decl.pattern,
                    "local",
                    decl.binding_kind,
                    decl.storage_class,
                    decl.range,
                    .{ .start = decl.range.start, .end = body.range.end },
                    debugRuntimeKindForStorageClass(
                        decl.storage_class,
                        switch (ast_file.pattern(decl.pattern).*) {
                            .Name => |name| global_slots.get(name.name),
                            else => null,
                        },
                    ),
                    switch (decl.storage_class) {
                        .none => null,
                        else => switch (ast_file.pattern(decl.pattern).*) {
                            .Name => |name| name.name,
                            else => null,
                        },
                    },
                    debugRuntimeLocationKindForStorageClass(
                        decl.storage_class,
                        switch (ast_file.pattern(decl.pattern).*) {
                            .Name => |name| global_slots.get(name.name),
                            else => null,
                        },
                    ),
                    debugRuntimeLocationRootForStorageClass(
                        decl.storage_class,
                        switch (ast_file.pattern(decl.pattern).*) {
                            .Name => |name| global_slots.get(name.name),
                            else => null,
                        },
                        switch (decl.storage_class) {
                            .none => null,
                            else => switch (ast_file.pattern(decl.pattern).*) {
                                .Name => |name| name.name,
                                else => null,
                            },
                        },
                    ),
                    switch (ast_file.pattern(decl.pattern).*) {
                        .Name => |name| global_slots.get(name.name),
                        else => null,
                    },
                    debugBindingEditable(
                        decl.storage_class,
                        switch (ast_file.pattern(decl.pattern).*) {
                            .Name => |name| global_slots.get(name.name),
                            else => null,
                        },
                    ),
                    folded_value,
                    folded_value != null,
                    &state.next_local_id,
                    locals,
                );
            },
            else => {},
        }
    }

    try scopes.append(allocator, .{
        .id = scope_id,
        .parent = parent_scope_id,
        .file_id = file_id,
        .file_path = db.sources.file(file_id).path,
        .function_name = function_name,
        .contract_name = contract_name,
        .kind = kind,
        .label = label,
        .range = body.range,
        .local_start = local_start,
        .local_count = locals.items.len - local_start,
    });

    for (body.statements) |statement_id| {
        switch (ast_file.statement(statement_id).*) {
            .If => |if_stmt| {
                try collectBodyScopeDebugInfo(allocator, db, const_eval, global_slots, ast_file, file_id, function_name, contract_name, if_stmt.then_body, scope_id, "if_then", null, &.{}, state, scopes, locals);
                if (if_stmt.else_body) |else_body| {
                    try collectBodyScopeDebugInfo(allocator, db, const_eval, global_slots, ast_file, file_id, function_name, contract_name, else_body, scope_id, "if_else", null, &.{}, state, scopes, locals);
                }
            },
            .While => |while_stmt| {
                try collectBodyScopeDebugInfo(allocator, db, const_eval, global_slots, ast_file, file_id, function_name, contract_name, while_stmt.body, scope_id, "while", while_stmt.label, &.{}, state, scopes, locals);
            },
            .For => |for_stmt| {
                var bindings = std.ArrayList(ExtraScopeBinding){};
                defer bindings.deinit(allocator);
                try bindings.append(allocator, .{
                    .pattern_id = for_stmt.item_pattern,
                    .kind = "for_item",
                    .decl_range = compiler.source.rangeOf(ast_file.pattern(for_stmt.item_pattern).*),
                    .live_range = ast_file.body(for_stmt.body).*.range,
                    .runtime_kind = "ssa",
                    .runtime_name = null,
                    .runtime_location_kind = null,
                    .runtime_location_root = null,
                    .editable = false,
                });
                if (for_stmt.index_pattern) |index_pattern| {
                    try bindings.append(allocator, .{
                        .pattern_id = index_pattern,
                        .kind = "for_index",
                        .decl_range = compiler.source.rangeOf(ast_file.pattern(index_pattern).*),
                        .live_range = ast_file.body(for_stmt.body).*.range,
                        .runtime_kind = "ssa",
                        .runtime_name = null,
                        .runtime_location_kind = null,
                        .runtime_location_root = null,
                        .editable = false,
                    });
                }
                try collectBodyScopeDebugInfo(allocator, db, const_eval, global_slots, ast_file, file_id, function_name, contract_name, for_stmt.body, scope_id, "for", for_stmt.label, bindings.items, state, scopes, locals);
            },
            .Switch => |switch_stmt| {
                for (switch_stmt.arms) |arm| {
                    try collectBodyScopeDebugInfo(allocator, db, const_eval, global_slots, ast_file, file_id, function_name, contract_name, arm.body, scope_id, "switch_arm", null, &.{}, state, scopes, locals);
                }
                if (switch_stmt.else_body) |else_body| {
                    try collectBodyScopeDebugInfo(allocator, db, const_eval, global_slots, ast_file, file_id, function_name, contract_name, else_body, scope_id, "switch_else", null, &.{}, state, scopes, locals);
                }
            },
            .Try => |try_stmt| {
                try collectBodyScopeDebugInfo(allocator, db, const_eval, global_slots, ast_file, file_id, function_name, contract_name, try_stmt.try_body, scope_id, "try", null, &.{}, state, scopes, locals);
                if (try_stmt.catch_clause) |catch_clause| {
                    var catch_bindings: [1]ExtraScopeBinding = undefined;
                    const catch_items = if (catch_clause.error_pattern) |pattern_id| blk: {
                        catch_bindings[0] = .{
                            .pattern_id = pattern_id,
                            .kind = "catch_error",
                            .decl_range = catch_clause.range,
                            .live_range = ast_file.body(catch_clause.body).*.range,
                            .runtime_kind = "ssa",
                            .runtime_name = null,
                            .runtime_location_kind = null,
                            .runtime_location_root = null,
                            .editable = false,
                        };
                        break :blk catch_bindings[0..1];
                    } else &.{};
                    try collectBodyScopeDebugInfo(allocator, db, const_eval, global_slots, ast_file, file_id, function_name, contract_name, catch_clause.body, scope_id, "catch", null, catch_items, state, scopes, locals);
                }
            },
            .Block => |block_stmt| {
                try collectBodyScopeDebugInfo(allocator, db, const_eval, global_slots, ast_file, file_id, function_name, contract_name, block_stmt.body, scope_id, "block", null, &.{}, state, scopes, locals);
            },
            .LabeledBlock => |block_stmt| {
                try collectBodyScopeDebugInfo(allocator, db, const_eval, global_slots, ast_file, file_id, function_name, contract_name, block_stmt.body, scope_id, "labeled_block", block_stmt.label, &.{}, state, scopes, locals);
            },
            else => {},
        }
    }
}

fn collectItemDebugScopes(
    allocator: std.mem.Allocator,
    db: *compiler.CompilerDb,
    const_eval: *const compiler.sema.ConstEvalResult,
    global_slots: *const DebugGlobalSlots,
    ast_file: *const compiler.ast.AstFile,
    file_id: compiler.FileId,
    item_id: compiler.ItemId,
    inherited_contract_name: ?[]const u8,
    state: *ScopeBuildState,
    scopes: *std.ArrayList(DebugScopeInfo),
    locals: *std.ArrayList(DebugLocalInfo),
) !void {
    switch (ast_file.item(item_id).*) {
        .Contract => |contract_item| {
            for (contract_item.members) |member_id| {
                try collectItemDebugScopes(allocator, db, const_eval, global_slots, ast_file, file_id, member_id, contract_item.name, state, scopes, locals);
            }
        },
        .Impl => |impl_item| {
            for (impl_item.methods) |method_id| {
                try collectItemDebugScopes(allocator, db, const_eval, global_slots, ast_file, file_id, method_id, inherited_contract_name, state, scopes, locals);
            }
        },
        .Function => |function_item| {
            if (function_item.is_comptime or function_item.is_ghost) return;
            var param_bindings: std.ArrayList(ExtraScopeBinding) = .{};
            defer param_bindings.deinit(allocator);
            for (function_item.parameters) |parameter| {
                if (parameter.is_comptime) continue;
                try param_bindings.append(allocator, .{
                    .pattern_id = parameter.pattern,
                    .kind = "param",
                    .decl_range = parameter.range,
                    .live_range = ast_file.body(function_item.body).*.range,
                    .runtime_kind = "ssa",
                    .runtime_name = null,
                    .runtime_location_kind = null,
                    .runtime_location_root = null,
                    .runtime_location_slot = null,
                    .editable = false,
                });
            }
            if (function_item.parent_contract) |contract_item_id| {
                const contract_item = ast_file.item(contract_item_id).*;
                if (contract_item == .Contract) {
                    for (contract_item.Contract.members) |member_id| {
                        const member = ast_file.item(member_id).*;
                        if (member != .Field) continue;
                        const field = member.Field;
                        try param_bindings.append(allocator, .{
                            .name = field.name,
                            .kind = "field",
                            .binding_kind = field.binding_kind,
                            .storage_class = field.storage_class,
                            .decl_range = field.range,
                            .live_range = ast_file.body(function_item.body).*.range,
                            .runtime_kind = switch (field.storage_class) {
                                .none => "optimized_out",
                                else => debugRuntimeKindForStorageClass(field.storage_class, global_slots.get(field.name)),
                            },
                            .runtime_name = field.name,
                            .runtime_location_kind = debugRuntimeLocationKindForStorageClass(field.storage_class, global_slots.get(field.name)),
                            .runtime_location_root = debugRuntimeLocationRootForStorageClass(field.storage_class, global_slots.get(field.name), field.name),
                            .runtime_location_slot = global_slots.get(field.name),
                            .editable = debugBindingEditable(field.storage_class, global_slots.get(field.name)),
                            .folded_value = null,
                            .is_folded = false,
                        });
                    }
                }
            }
            try collectBodyScopeDebugInfo(
                allocator,
                db,
                const_eval,
                global_slots,
                ast_file,
                file_id,
                function_item.name,
                inherited_contract_name,
                function_item.body,
                null,
                "function",
                null,
                param_bindings.items,
                state,
                scopes,
                locals,
            );
        },
        else => {},
    }
}

fn buildSourceScopeDebugInfo(
    allocator: std.mem.Allocator,
    db: *compiler.CompilerDb,
    root_module_id: compiler.ModuleId,
    global_slots: *const DebugGlobalSlots,
) !DebugSourceScopeBundle {
    var scopes: std.ArrayList(DebugScopeInfo) = .{};
    var locals: std.ArrayList(DebugLocalInfo) = .{};
    errdefer scopes.deinit(allocator);
    errdefer locals.deinit(allocator);

    var state = ScopeBuildState{};
    const root_module = db.sources.module(root_module_id);
    const package = db.sources.package(root_module.package_id);
    for (package.modules.items) |module_id| {
        const module = db.sources.module(module_id);
        const ast_file = try db.astFile(module.file_id);
        const const_eval = try db.constEval(module_id);
        for (ast_file.root_items) |item_id| {
            try collectItemDebugScopes(allocator, db, const_eval, global_slots, ast_file, module.file_id, item_id, null, &state, &scopes, &locals);
        }
    }

    return .{
        .scopes = try scopes.toOwnedSlice(allocator),
        .locals = try locals.toOwnedSlice(allocator),
    };
}

fn parseDebugGlobalSlots(allocator: std.mem.Allocator, json_bytes: ?[]const u8) !DebugGlobalSlots {
    var slots = DebugGlobalSlots.init(allocator);
    errdefer slots.deinit();

    const input = json_bytes orelse return slots;
    const trimmed = std.mem.trim(u8, input, " \n\r\t");
    if (trimmed.len == 0) return slots;

    var parsed = try std.json.parseFromSlice(std.json.Value, allocator, trimmed, .{});
    defer parsed.deinit();
    if (parsed.value != .object) return slots;

    var it = parsed.value.object.iterator();
    while (it.next()) |entry| {
        switch (entry.value_ptr.*) {
            .integer => |value| try slots.put(try allocator.dupe(u8, entry.key_ptr.*), @intCast(value)),
            else => {},
        }
    }
    return slots;
}

fn parseDebugGlobalSlotsFromSirText(allocator: std.mem.Allocator, sir_text: []const u8) !DebugGlobalSlots {
    var slots = DebugGlobalSlots.init(allocator);
    errdefer deinitDebugGlobalSlots(allocator, &slots);

    const marker = "ora.global_slots = {";
    const start = std.mem.indexOf(u8, sir_text, marker) orelse return slots;
    const body_start = start + marker.len;
    const body_end = std.mem.indexOfScalarPos(u8, sir_text, body_start, '}') orelse return slots;
    const body = sir_text[body_start..body_end];

    var it = std.mem.splitScalar(u8, body, ',');
    while (it.next()) |entry_raw| {
        const entry = std.mem.trim(u8, entry_raw, " \n\r\t");
        if (entry.len == 0) continue;
        const eq = std.mem.indexOfScalar(u8, entry, '=') orelse continue;
        const name = std.mem.trim(u8, entry[0..eq], " \n\r\t");
        const rest = std.mem.trim(u8, entry[eq + 1 ..], " \n\r\t");
        const colon = std.mem.indexOfScalar(u8, rest, ':') orelse continue;
        const slot_text = std.mem.trim(u8, rest[0..colon], " \n\r\t");
        const slot = std.fmt.parseInt(u64, slot_text, 10) catch continue;
        try slots.put(try allocator.dupe(u8, name), slot);
    }

    return slots;
}

fn posBeforeOrEqual(line_a: u32, col_a: u32, line_b: u32, col_b: u32) bool {
    return line_a < line_b or (line_a == line_b and col_a <= col_b);
}

fn posStrictlyBefore(line_a: u32, col_a: u32, line_b: u32, col_b: u32) bool {
    return line_a < line_b or (line_a == line_b and col_a < col_b);
}

fn posInRange(
    sources: *const compiler.source.SourceStore,
    file_id: compiler.FileId,
    range: compiler.TextRange,
    line: u32,
    col: u32,
) bool {
    const start = lineColumnForOffset(sources, file_id, range.start);
    const end = lineColumnForOffset(sources, file_id, range.end);
    return posBeforeOrEqual(start.line, start.column, line, col) and posStrictlyBefore(line, col, end.line, end.column);
}

const ExecutableStatementKind = enum {
    runtime,
    runtime_guard,
};

const ExecutableStatementStart = struct {
    line: u32,
    col: u32,
    stmt_id: ?u32,
    kind: ExecutableStatementKind,
};

const ExecutableStartList = std.ArrayList(ExecutableStatementStart);
const ExecutableLineMap = std.StringHashMap(ExecutableStartList);

fn executableStatementKindName(kind: ExecutableStatementKind) []const u8 {
    return switch (kind) {
        .runtime => "runtime",
        .runtime_guard => "runtime_guard",
    };
}

fn deinitExecutableLineMap(allocator: std.mem.Allocator, line_map: *ExecutableLineMap) void {
    var it = line_map.iterator();
    while (it.next()) |entry| {
        allocator.free(entry.key_ptr.*);
        entry.value_ptr.deinit(allocator);
    }
    line_map.deinit();
}

fn putExecutableStatementStart(
    allocator: std.mem.Allocator,
    line_map: *ExecutableLineMap,
    file_path: []const u8,
    line: u32,
    col: u32,
    stmt_id: ?u32,
    kind: ExecutableStatementKind,
) !void {
    const gop = try line_map.getOrPut(file_path);
    if (!gop.found_existing) {
        gop.key_ptr.* = try allocator.dupe(u8, file_path);
        gop.value_ptr.* = .{};
    }

    for (gop.value_ptr.items) |*entry| {
        if (entry.line == line and entry.col == col and entry.stmt_id == stmt_id) {
            if (entry.kind == .runtime_guard or kind == .runtime_guard) {
                entry.kind = .runtime_guard;
            } else {
                entry.kind = .runtime;
            }
            return;
        }
    }

    try gop.value_ptr.append(allocator, .{
        .line = line,
        .col = col,
        .stmt_id = stmt_id,
        .kind = kind,
    });
}

const ExecutableStatementMeta = struct {
    stmt_id: ?u32,
    kind: ExecutableStatementKind,
};

fn executableStatementMetaForPosition(
    line_map: *const ExecutableLineMap,
    file_path: []const u8,
    line: u32,
    col: u32,
) ?ExecutableStatementMeta {
    const file_entries = line_map.get(file_path) orelse return null;
    var best: ?ExecutableStatementMeta = null;
    var best_col: u32 = 0;
    for (file_entries.items) |entry| {
        if (entry.line != line) continue;
        if (entry.col > col) continue;
        if (best == null or entry.col >= best_col) {
            best = .{
                .stmt_id = entry.stmt_id,
                .kind = entry.kind,
            };
            best_col = entry.col;
        }
    }
    return best;
}

fn suppressHoistedDuplicateStmtMarkers(entries: anytype) void {
    var i: usize = 0;
    while (i < entries.len) : (i += 1) {
        if (!entries[i].stmt) continue;

        const src = entries[i].src;
        const line = entries[i].line;
        var saw_lower_line = false;
        var suppress = false;

        var j: usize = i + 1;
        while (j < entries.len) : (j += 1) {
            if (!entries[j].stmt) continue;
            if (entries[j].src != src) continue;

            if (entries[j].line < line) {
                saw_lower_line = true;
                continue;
            }

            if (entries[j].line == line and saw_lower_line) {
                suppress = true;
                break;
            }
        }

        if (suppress) entries[i].stmt = false;
    }
}

fn addExecutableRangeStart(
    allocator: std.mem.Allocator,
    sources: *const compiler.source.SourceStore,
    line_map: *ExecutableLineMap,
    file_id: compiler.FileId,
    statement_id: ?compiler.ast.StmtId,
    range: compiler.TextRange,
    kind: ExecutableStatementKind,
) !void {
    const pos = lineColumnForOffset(sources, file_id, range.start);
    try putExecutableStatementStart(
        allocator,
        line_map,
        sources.file(file_id).path,
        pos.line,
        pos.column,
        if (statement_id) |id| @intCast(id.index()) else null,
        kind,
    );
}

fn collectExecutableStmtLines(
    allocator: std.mem.Allocator,
    sources: *const compiler.source.SourceStore,
    ast_file: *const compiler.ast.AstFile,
    file_id: compiler.FileId,
    statement_id: compiler.ast.StmtId,
    line_map: *ExecutableLineMap,
) anyerror!void {
    const stmt = ast_file.statement(statement_id).*;
    switch (stmt) {
        .VariableDecl => |node| try addExecutableRangeStart(allocator, sources, line_map, file_id, statement_id, node.range, .runtime),
        .Return => |node| try addExecutableRangeStart(allocator, sources, line_map, file_id, statement_id, node.range, .runtime),
        .Expr => |node| try addExecutableRangeStart(allocator, sources, line_map, file_id, statement_id, node.range, .runtime),
        .Assign => |node| try addExecutableRangeStart(allocator, sources, line_map, file_id, statement_id, node.range, .runtime),
        .If => |node| {
            try addExecutableRangeStart(allocator, sources, line_map, file_id, statement_id, node.range, .runtime);
            try collectExecutableBodyLines(allocator, sources, ast_file, file_id, node.then_body, line_map);
            if (node.else_body) |else_body| {
                try collectExecutableBodyLines(allocator, sources, ast_file, file_id, else_body, line_map);
            }
        },
        .While => |node| {
            try addExecutableRangeStart(allocator, sources, line_map, file_id, statement_id, node.range, .runtime);
            try collectExecutableBodyLines(allocator, sources, ast_file, file_id, node.body, line_map);
        },
        .For => |node| {
            try addExecutableRangeStart(allocator, sources, line_map, file_id, statement_id, node.range, .runtime);
            try collectExecutableBodyLines(allocator, sources, ast_file, file_id, node.body, line_map);
        },
        .Switch => |node| {
            try addExecutableRangeStart(allocator, sources, line_map, file_id, statement_id, node.range, .runtime);
            for (node.arms) |arm| {
                try collectExecutableBodyLines(allocator, sources, ast_file, file_id, arm.body, line_map);
            }
            if (node.else_body) |else_body| {
                try collectExecutableBodyLines(allocator, sources, ast_file, file_id, else_body, line_map);
            }
        },
        .Try => |node| {
            try addExecutableRangeStart(allocator, sources, line_map, file_id, statement_id, node.range, .runtime);
            try collectExecutableBodyLines(allocator, sources, ast_file, file_id, node.try_body, line_map);
            if (node.catch_clause) |catch_clause| {
                try collectExecutableBodyLines(allocator, sources, ast_file, file_id, catch_clause.body, line_map);
            }
        },
        .Break => |node| try addExecutableRangeStart(allocator, sources, line_map, file_id, statement_id, node.range, .runtime),
        .Continue => |node| try addExecutableRangeStart(allocator, sources, line_map, file_id, statement_id, node.range, .runtime),
        .Assert => |node| try addExecutableRangeStart(allocator, sources, line_map, file_id, statement_id, node.range, .runtime_guard),
        .Log => |node| try addExecutableRangeStart(allocator, sources, line_map, file_id, statement_id, node.range, .runtime),
        .Lock => |node| try addExecutableRangeStart(allocator, sources, line_map, file_id, statement_id, node.range, .runtime),
        .Unlock => |node| try addExecutableRangeStart(allocator, sources, line_map, file_id, statement_id, node.range, .runtime),
        .Block => |node| try collectExecutableBodyLines(allocator, sources, ast_file, file_id, node.body, line_map),
        .LabeledBlock => |node| try collectExecutableBodyLines(allocator, sources, ast_file, file_id, node.body, line_map),
        .Assume, .Havoc, .Error => {},
    }
}

fn collectExecutableBodyLines(
    allocator: std.mem.Allocator,
    sources: *const compiler.source.SourceStore,
    ast_file: *const compiler.ast.AstFile,
    file_id: compiler.FileId,
    body_id: compiler.ast.BodyId,
    line_map: *ExecutableLineMap,
) anyerror!void {
    const body = ast_file.body(body_id).*;
    for (body.statements) |statement_id| {
        try collectExecutableStmtLines(allocator, sources, ast_file, file_id, statement_id, line_map);
    }
}

fn collectExecutableItemLines(
    allocator: std.mem.Allocator,
    sources: *const compiler.source.SourceStore,
    ast_file: *const compiler.ast.AstFile,
    file_id: compiler.FileId,
    item_id: compiler.ItemId,
    line_map: *ExecutableLineMap,
) anyerror!void {
    switch (ast_file.item(item_id).*) {
        .Contract => |contract_item| {
            for (contract_item.members) |member_id| {
                try collectExecutableItemLines(allocator, sources, ast_file, file_id, member_id, line_map);
            }
        },
        .Impl => |impl_item| {
            for (impl_item.methods) |method_id| {
                try collectExecutableItemLines(allocator, sources, ast_file, file_id, method_id, line_map);
            }
        },
        .Function => |function_item| {
            if (function_item.is_comptime or function_item.is_ghost) return;
            for (function_item.clauses) |clause| {
                if (clause.kind != .guard) continue;
                try addExecutableRangeStart(allocator, sources, line_map, file_id, null, clause.range, .runtime_guard);
            }
            try collectExecutableBodyLines(allocator, sources, ast_file, file_id, function_item.body, line_map);
        },
        else => {},
    }
}

fn buildExecutableLineMap(
    allocator: std.mem.Allocator,
    db: *compiler.CompilerDb,
    root_module_id: compiler.ModuleId,
) !ExecutableLineMap {
    var line_map = ExecutableLineMap.init(allocator);
    errdefer deinitExecutableLineMap(allocator, &line_map);

    const root_module = db.sources.module(root_module_id);
    const package = db.sources.package(root_module.package_id);
    for (package.modules.items) |module_id| {
        const module = db.sources.module(module_id);
        const ast_file = try db.astFile(module.file_id);
        for (ast_file.root_items) |item_id| {
            try collectExecutableItemLines(allocator, &db.sources, ast_file, module.file_id, item_id, &line_map);
        }
    }

    return line_map;
}

fn writeCompilerType(writer: anytype, ty: compiler.sema.Type) !void {
    switch (ty) {
        .unknown => try writer.writeAll("unknown"),
        .void => try writer.writeAll("void"),
        .bool => try writer.writeAll("bool"),
        .string => try writer.writeAll("string"),
        .address => try writer.writeAll("address"),
        .bytes => try writer.writeAll("bytes"),
        .external_proxy => |proxy| try writer.print("external<{s}>", .{proxy.trait_name}),
        .integer => |integer| try writer.writeAll(integer.spelling orelse "int"),
        .named => |named| try writer.writeAll(named.name),
        .contract => |named| try writer.writeAll(named.name),
        .struct_ => |named| try writer.writeAll(named.name),
        .bitfield => |named| try writer.writeAll(named.name),
        .enum_ => |named| try writer.writeAll(named.name),
        .function => |function| {
            try writer.writeAll("fn(");
            for (function.param_types, 0..) |param_type, index| {
                if (index != 0) try writer.writeAll(", ");
                try writeCompilerType(writer, param_type);
            }
            try writer.writeAll(")");
            if (function.return_types.len == 1) {
                try writer.writeAll(" -> ");
                try writeCompilerType(writer, function.return_types[0]);
            } else if (function.return_types.len > 1) {
                try writer.writeAll(" -> (");
                for (function.return_types, 0..) |return_type, index| {
                    if (index != 0) try writer.writeAll(", ");
                    try writeCompilerType(writer, return_type);
                }
                try writer.writeAll(")");
            }
        },
        .tuple => |elements| {
            try writer.writeAll("(");
            for (elements, 0..) |element, index| {
                if (index != 0) try writer.writeAll(", ");
                try writeCompilerType(writer, element);
            }
            try writer.writeAll(")");
        },
        .anonymous_struct => |struct_type| {
            try writer.writeAll("struct { ");
            for (struct_type.fields, 0..) |field, index| {
                if (index != 0) try writer.writeAll(", ");
                try writer.print("{s}: ", .{field.name});
                try writeCompilerType(writer, field.ty);
            }
            try writer.writeAll(" }");
        },
        .array => |array| {
            try writer.writeByte('[');
            try writeCompilerType(writer, array.element_type.*);
            try writer.writeAll("; ");
            if (array.len) |len| {
                try writer.print("{d}", .{len});
            } else {
                try writer.writeAll("?");
            }
            try writer.writeByte(']');
        },
        .slice => |slice| {
            try writer.writeAll("[]");
            try writeCompilerType(writer, slice.element_type.*);
        },
        .map => |map| {
            try writer.writeAll("map<");
            if (map.key_type) |key_type| {
                try writeCompilerType(writer, key_type.*);
            } else {
                try writer.writeAll("?");
            }
            try writer.writeAll(", ");
            if (map.value_type) |value_type| {
                try writeCompilerType(writer, value_type.*);
            } else {
                try writer.writeAll("?");
            }
            try writer.writeByte('>');
        },
        .error_union => |error_union| {
            try writer.writeByte('!');
            try writeCompilerType(writer, error_union.payload_type.*);
            if (error_union.error_types.len != 0) {
                try writer.writeAll(" | ");
                for (error_union.error_types, 0..) |error_type, index| {
                    if (index != 0) try writer.writeAll(", ");
                    try writeCompilerType(writer, error_type);
                }
            }
        },
        .refinement => |refinement| {
            try writer.writeAll(refinement.name);
            if (refinement.args.len != 0) {
                try writer.writeByte('<');
                for (refinement.args, 0..) |arg, index| {
                    if (index != 0) try writer.writeAll(", ");
                    switch (arg) {
                        .Type => try writer.writeAll("type"),
                        .Integer => |integer| try writer.writeAll(integer.text),
                    }
                }
                try writer.writeByte('>');
            }
        },
    }
}

fn compilerItemKindName(item: compiler.ast.Item) []const u8 {
    return switch (item) {
        .Import => "Import",
        .Contract => "Contract",
        .Function => "Function",
        .Struct => "Struct",
        .Bitfield => "Bitfield",
        .Enum => "Enum",
        .Trait => "Trait",
        .Impl => "Impl",
        .TypeAlias => "TypeAlias",
        .LogDecl => "LogDecl",
        .ErrorDecl => "ErrorDecl",
        .GhostBlock => "GhostBlock",
        .Field => "Field",
        .Constant => "Constant",
        .Error => "Error",
    };
}

fn compilerItemName(item: compiler.ast.Item) ?[]const u8 {
    return switch (item) {
        .Contract => |contract_item| contract_item.name,
        .Function => |function| function.name,
        .Struct => |struct_item| struct_item.name,
        .Bitfield => |bitfield_item| bitfield_item.name,
        .Enum => |enum_item| enum_item.name,
        .Trait => |trait_item| trait_item.name,
        .Impl => |impl_item| impl_item.target_name,
        .TypeAlias => |type_alias| type_alias.name,
        .LogDecl => |log_decl| log_decl.name,
        .ErrorDecl => |error_decl| error_decl.name,
        .Field => |field| field.name,
        .Constant => |constant| constant.name,
        else => null,
    };
}

fn writeCompilerAst(
    allocator: std.mem.Allocator,
    writer: anytype,
    ast_file: *const compiler.ast.AstFile,
    typecheck: ?*const compiler.sema.TypeCheckResult,
    diagnostics_list: *const compiler.diagnostics.DiagnosticList,
    format: []const u8,
    debug_enabled: bool,
) !void {
    if (!std.mem.eql(u8, format, "tree") and !std.mem.eql(u8, format, "json")) {
        return error.InvalidArgument;
    }

    if (std.mem.eql(u8, format, "json")) {
        try writer.writeAll("{\n  \"root_items\": [\n");
        for (ast_file.root_items, 0..) |item_id, index| {
            const item = ast_file.item(item_id).*;
            if (index != 0) try writer.writeAll(",\n");
            try writer.writeAll("    {\"kind\": ");
            try writeJsonString(writer, compilerItemKindName(item));
            if (compilerItemName(item)) |name| {
                try writer.writeAll(", \"name\": ");
                try writeJsonString(writer, name);
            }
            if (typecheck) |typed| {
                try writer.writeAll(", \"type\": ");
                var type_buffer: std.ArrayList(u8) = .{};
                defer type_buffer.deinit(allocator);
                try writeCompilerType(type_buffer.writer(allocator), typed.item_types[item_id.index()]);
                try writeJsonString(writer, type_buffer.items);
            }
            try writer.writeByte('}');
        }
        try writer.writeAll("\n  ],\n  \"diagnostics\": [\n");
        for (diagnostics_list.items.items, 0..) |diag, index| {
            if (index != 0) try writer.writeAll(",\n");
            try writer.writeAll("    {\"severity\": ");
            try writeJsonString(writer, compilerDiagnosticSeverityName(diag.severity));
            try writer.writeAll(", \"message\": ");
            try writeJsonString(writer, diag.message);
            try writer.writeAll("}");
        }
        try writer.writeAll("\n  ]\n}\n");
        return;
    }

    try writer.print("Compiler {s}AST\n", .{if (typecheck != null) "typed " else ""});
    try writer.print("Root items: {d}\n", .{ast_file.root_items.len});
    try writeCompilerDiagnosticsText(writer, null, diagnostics_list, debug_enabled);
    for (ast_file.root_items, 0..) |item_id, index| {
        const item = ast_file.item(item_id).*;
        try writer.print("[{d}] {s}", .{ index, compilerItemKindName(item) });
        if (compilerItemName(item)) |name| {
            try writer.print(" {s}", .{name});
        }
        if (typecheck) |typed| {
            try writer.writeAll(" : ");
            try writeCompilerType(writer, typed.item_types[item_id.index()]);
        }
        try writer.writeByte('\n');
    }
}

fn runCompilerAstEmit(
    allocator: std.mem.Allocator,
    file_path: []const u8,
    format: []const u8,
    include_types: bool,
    resolver_options: import_graph.ResolverOptions,
    m: *Metrics,
    debug_enabled: bool,
) !void {
    var stdout_buffer: [1024]u8 = undefined;
    var stdout_writer = std.fs.File.stdout().writer(&stdout_buffer);
    const stdout = &stdout_writer.interface;

    m.begin("compiler");
    var compilation = compiler.driver.compilePackageWithResolverOptions(allocator, file_path, resolver_options) catch |err| {
        m.end();
        try stdout.print("Compiler error: {s}\n", .{@errorName(err)});
        try stdout.flush();
        std.process.exit(1);
    };
    m.end();
    defer compilation.deinit();

    const module = compilation.db.sources.module(compilation.root_module_id);
    const ast_file = try compilation.db.astFile(module.file_id);
    const module_typecheck = try exitOnCompilationErrors(stdout, &compilation.db, compilation.root_module_id, debug_enabled);
    const typecheck = if (include_types) module_typecheck else null;
    const diagnostics_list = &module_typecheck.diagnostics;

    writeCompilerAst(allocator, stdout, ast_file, typecheck, diagnostics_list, format, debug_enabled) catch |err| switch (err) {
        error.InvalidArgument => {
            try stdout.print("error: unsupported AST format '{s}' (use 'tree' or 'json')\n", .{format});
            try stdout.flush();
            std.process.exit(2);
        },
        else => return err,
    };
    try stdout.flush();
}

fn runCompilerMlirEmit(
    allocator: std.mem.Allocator,
    file_path: []const u8,
    mlir_options: MlirOptions,
    resolver_options: import_graph.ResolverOptions,
    debug_enabled: bool,
) !void {
    const c = @import("mlir_c_api").c;

    var stdout_buffer: [1024]u8 = undefined;
    var stdout_writer = std.fs.File.stdout().writer(&stdout_buffer);
    const stdout = &stdout_writer.interface;
    const m = mlir_options.metrics;

    m.begin("compiler");
    var compilation = compiler.driver.compilePackageWithResolverOptions(allocator, file_path, resolver_options) catch |err| {
        m.end();
        try stdout.print("Compiler error: {s}\n", .{@errorName(err)});
        try stdout.flush();
        std.process.exit(1);
    };
    m.end();
    defer compilation.deinit();

    _ = try exitOnCompilationErrors(stdout, &compilation.db, compilation.root_module_id, debug_enabled);

    const lowering = try compilation.db.lowerToHir(compilation.root_module_id);
    if (mlir_options.validate_mlir) {
        try verifyMlirModule(stdout, lowering.module.raw_module, "Ora MLIR");
    }
    if (mlir_options.emit_mlir_sir) {
        if (!c.oraConvertToSIR(lowering.context, lowering.module.raw_module, mlir_options.debug_info)) {
            try stdout.print("Compiler error: Ora to SIR conversion failed\n", .{});
            try stdout.flush();
            std.process.exit(1);
        }
    }

    const module_op = c.oraModuleGetOperation(lowering.module.raw_module);
    const text_ref = c.oraOperationPrintToString(module_op);
    defer if (text_ref.data != null) {
        const mlir_c = @import("mlir_c_api");
        mlir_c.freeStringRef(text_ref);
    };

    if (text_ref.data == null or text_ref.length == 0) {
        try stdout.print("Compiler error: failed to print MLIR module\n", .{});
        try stdout.flush();
        std.process.exit(1);
    }
    try stdout.print("{s}", .{text_ref.data[0..text_ref.length]});
    try stdout.flush();
}

fn runCompilerTokenEmit(
    allocator: std.mem.Allocator,
    file_path: []const u8,
) !void {
    const source_text = try std.fs.cwd().readFileAlloc(allocator, file_path, std.math.maxInt(usize));
    defer allocator.free(source_text);

    var lexer = lib.Lexer.init(allocator, source_text);
    defer lexer.deinit();

    const tokens = try lexer.scanTokens();
    defer allocator.free(tokens);

    var stdout_buffer: [1024]u8 = undefined;
    var stdout_writer = std.fs.File.stdout().writer(&stdout_buffer);
    const stdout = &stdout_writer.interface;

    for (tokens) |token| {
        try stdout.print("{any}\n", .{token});
    }
    try stdout.flush();
}

fn runMlirEmitAdvanced(
    allocator: std.mem.Allocator,
    file_path: []const u8,
    mlir_options: MlirOptions,
    resolver_options: import_graph.ResolverOptions,
    debug_enabled: bool,
) !void {
    const c = @import("mlir_c_api").c;

    var stdout_buffer: [1024]u8 = undefined;
    var stdout_writer = std.fs.File.stdout().writer(&stdout_buffer);
    const stdout = &stdout_writer.interface;
    const m = mlir_options.metrics;

    var mlir_arena = std.heap.ArenaAllocator.init(allocator);
    defer mlir_arena.deinit();
    const mlir_allocator = mlir_arena.allocator();

    m.begin("compiler");
    var compilation = compiler.driver.compilePackageWithResolverOptions(allocator, file_path, resolver_options) catch |err| {
        m.end();
        try stdout.print("Compiler error: {s}\n", .{@errorName(err)});
        try stdout.flush();
        std.process.exit(1);
    };
    m.end();
    defer compilation.deinit();

    _ = try exitOnCompilationErrors(stdout, &compilation.db, compilation.root_module_id, debug_enabled);

    const lowering = try compilation.db.lowerToHir(compilation.root_module_id);
    const final_module = lowering.module.raw_module;
    const ctx = lowering.context;

    if (mlir_options.validate_mlir) {
        try verifyMlirModule(stdout, final_module, "Ora MLIR");
    }

    var verification_result_opt: ?@import("z3/errors.zig").VerificationResult = null;
    var verification_failed = false;
    var pending_smt_report: ?@import("z3/mod.zig").SmtReportArtifacts = null;
    defer {
        if (verification_result_opt) |*vr| vr.deinit();
        if (pending_smt_report) |*report| report.deinit(mlir_allocator);
    }

    if (mlir_options.verify_z3) {
        m.begin("z3 verification");
        const z3_verification = @import("z3/verification.zig");
        var verifier = try z3_verification.VerificationPass.initWithProofs(mlir_allocator, mlir_options.z3_proofs);
        defer verifier.deinit();

        if (mlir_options.verify_mode) |mode| {
            if (std.ascii.eqlIgnoreCase(mode, "full")) verifier.setVerifyMode(.Full) else verifier.setVerifyMode(.Basic);
        }
        if (mlir_options.verify_calls) |enabled| verifier.setVerifyCalls(enabled);
        if (mlir_options.verify_state) |enabled| verifier.setVerifyState(enabled);
        verifier.setVerifyStats(mlir_options.verify_stats);
        verifier.setExplainCores(mlir_options.explain_cores);
        verifier.setMinimizeCores(mlir_options.minimize_cores);

        const verification_result = try verifier.runVerificationPass(final_module);

        if (verification_result.diagnostics.items.len > 0) {
            // Refinement guards that SMT can't prove are lowered to runtime checks.
            // These are not failures — the guard is kept. Show detail with --debug.
            if (mlir_options.debug_enabled) {
                try printVerificationDiagnostics(stdout, verification_result.diagnostics.items);
            } else {
                try printRuntimeGuardSummary(stdout, verification_result.diagnostics.items);
            }
        }

        if (mlir_options.emit_smt_report) {
            pending_smt_report = try verifier.buildSmtReport(final_module, file_path, &verification_result);
        }

        if (!verification_result.success) {
            try printVerificationErrors(stdout, verification_result.errors.items);
            try stdout.flush();
            verification_failed = true;
        }

        verification_result_opt = verification_result;
        m.end();
    }

    if (mlir_options.emit_smt_report and !mlir_options.verify_z3) {
        m.begin("smt report");
        const z3_verification = @import("z3/verification.zig");
        var verifier = try z3_verification.VerificationPass.initWithProofs(mlir_allocator, mlir_options.z3_proofs);
        defer verifier.deinit();
        if (mlir_options.verify_mode) |mode| {
            if (std.ascii.eqlIgnoreCase(mode, "full")) verifier.setVerifyMode(.Full) else verifier.setVerifyMode(.Basic);
        }
        if (mlir_options.verify_calls) |enabled| verifier.setVerifyCalls(enabled);
        if (mlir_options.verify_state) |enabled| verifier.setVerifyState(enabled);
        verifier.setExplainCores(mlir_options.explain_cores);
        verifier.setMinimizeCores(mlir_options.minimize_cores);
        pending_smt_report = try verifier.buildSmtReport(final_module, file_path, null);
        m.end();
    }

    if (verification_failed) {
        return error.VerificationFailed;
    }

    if (mlir_options.canonicalize and (mlir_options.emit_mlir or mlir_options.emit_mlir_sir)) {
        m.begin("canonicalization");
        if (!c.oraCanonicalizeOraMLIR(ctx, final_module)) {
            try stdout.print("❌ Ora MLIR canonicalization failed\n", .{});
            try stdout.flush();
            std.process.exit(1);
        }
        m.end();
    }

    if (mlir_options.emit_mlir_sir) {
        const refinement_guards = @import("mlir/refinement_guards.zig");
        if (verification_result_opt) |*vr| {
            refinement_guards.cleanupRefinementGuards(ctx, final_module, &vr.proven_guard_ids);
        } else {
            var empty_guards = std.StringHashMap(void).init(mlir_allocator);
            defer empty_guards.deinit();
            refinement_guards.cleanupRefinementGuards(ctx, final_module, &empty_guards);
        }
    }

    if (mlir_options.emit_mlir or mlir_options.persist_ora_mlir) {
        const module_op_ora = c.oraModuleGetOperation(final_module);
        const mlir_str_ora = c.oraOperationPrintToString(module_op_ora);
        defer if (mlir_str_ora.data != null) {
            const mlir_c = @import("mlir_c_api");
            mlir_c.freeStringRef(mlir_str_ora);
        };

        if (mlir_str_ora.data != null and mlir_str_ora.length > 0) {
            const mlir_content_ora = mlir_str_ora.data[0..mlir_str_ora.length];
            if (mlir_options.persist_ora_mlir) {
                if (mlir_options.output_dir) |output_dir| {
                    std.fs.cwd().makeDir(output_dir) catch |err| switch (err) {
                        error.PathAlreadyExists => {},
                        else => return err,
                    };
                    const basename = std.fs.path.stem(file_path);
                    const filename = try std.mem.concat(allocator, u8, &[_][]const u8{ basename, ".ora.mlir" });
                    defer allocator.free(filename);
                    const output_file = try std.fs.path.join(allocator, &[_][]const u8{ output_dir, filename });
                    defer allocator.free(output_file);
                    var mlir_file = try std.fs.cwd().createFile(output_file, .{});
                    defer mlir_file.close();
                    try mlir_file.writeAll(mlir_content_ora);
                }
            }
            if (mlir_options.emit_mlir) {
                try stdout.print("//===----------------------------------------------------------------------===//\n", .{});
                try stdout.print("// Ora MLIR (before conversion)\n", .{});
                try stdout.print("//===----------------------------------------------------------------------===//\n\n", .{});
                try stdout.print("{s}\n", .{mlir_content_ora});
                try stdout.flush();
            }
        }
    }

    if (mlir_options.emit_mlir_sir) {
        if (!c.oraConvertToSIR(ctx, final_module, mlir_options.debug_info)) {
            try stdout.print("Error: Ora to SIR conversion failed\n", .{});
            try stdout.flush();
            std.process.exit(1);
        }
    }

    const explicit_sir_mlir_output = mlir_options.emit_mlir_sir and !mlir_options.emit_sir_text and !mlir_options.emit_bytecode;
    const emit_sir_mlir_output = explicit_sir_mlir_output or mlir_options.persist_sir_mlir;

    if (emit_sir_mlir_output) {
        const module_op_sir = c.oraModuleGetOperation(final_module);
        const mlir_str_sir = c.oraOperationPrintToString(module_op_sir);
        defer if (mlir_str_sir.data != null) {
            const mlir_c = @import("mlir_c_api");
            mlir_c.freeStringRef(mlir_str_sir);
        };

        if (mlir_str_sir.data == null or mlir_str_sir.length == 0) {
            try stdout.print("Failed to print SIR MLIR\n", .{});
            return;
        }

        const mlir_content_sir = mlir_str_sir.data[0..mlir_str_sir.length];
        if (mlir_options.persist_sir_mlir and mlir_options.output_dir != null) {
            const output_dir = mlir_options.output_dir.?;
            std.fs.cwd().makeDir(output_dir) catch |err| switch (err) {
                error.PathAlreadyExists => {},
                else => return err,
            };
            const basename = std.fs.path.stem(file_path);
            const filename = try std.mem.concat(allocator, u8, &[_][]const u8{ basename, ".sir.mlir" });
            defer allocator.free(filename);
            const output_file = try std.fs.path.join(allocator, &[_][]const u8{ output_dir, filename });
            defer allocator.free(output_file);
            var mlir_file = try std.fs.cwd().createFile(output_file, .{});
            defer mlir_file.close();
            try mlir_file.writeAll(mlir_content_sir);
        }
        if (explicit_sir_mlir_output) {
            if (mlir_options.output_dir) |output_dir| {
                std.fs.cwd().makeDir(output_dir) catch |err| switch (err) {
                    error.PathAlreadyExists => {},
                    else => return err,
                };
                const basename = std.fs.path.stem(file_path);
                const filename = try std.mem.concat(allocator, u8, &[_][]const u8{ basename, ".mlir" });
                defer allocator.free(filename);
                const output_file = try std.fs.path.join(allocator, &[_][]const u8{ output_dir, filename });
                defer allocator.free(output_file);
                var mlir_file = try std.fs.cwd().createFile(output_file, .{});
                defer mlir_file.close();
                try mlir_file.writeAll(mlir_content_sir);
                try stdout.print("SIR MLIR saved to {s}\n", .{output_file});
            } else {
                try stdout.print("{s}", .{mlir_content_sir});
            }
        }
    }

    if (mlir_options.emit_cfg_mode) |cfg_mode| {
        const mlir_cfg = @import("mlir/cfg.zig");
        const dot = try mlir_cfg.generateCFG(ctx, final_module, allocator);
        defer allocator.free(dot);

        if (mlir_options.output_dir) |output_dir| {
            const basename = std.fs.path.stem(file_path);
            const suffix = if (std.ascii.eqlIgnoreCase(cfg_mode, "sir")) ".sir.dot" else ".ora.dot";

            std.fs.cwd().makeDir(output_dir) catch |err| switch (err) {
                error.PathAlreadyExists => {},
                else => return err,
            };

            const filename = try std.mem.concat(allocator, u8, &[_][]const u8{ basename, suffix });
            defer allocator.free(filename);
            const output_file = try std.fs.path.join(allocator, &[_][]const u8{ output_dir, filename });
            defer allocator.free(output_file);
            var dot_file = try std.fs.cwd().createFile(output_file, .{});
            defer dot_file.close();
            try dot_file.writeAll(dot);
        } else {
            try stdout.print("{s}", .{dot});
        }
    }

    if (mlir_options.emit_sir_text or mlir_options.emit_bytecode) {
        if (!c.oraBuildSIRDispatcher(ctx, final_module)) {
            try stdout.print("Error: SIR dispatcher build failed\n", .{});
            try stdout.flush();
            std.process.exit(1);
        }
        if (!c.oraLegalizeSIRText(ctx, final_module)) {
            try stdout.print("Error: SIR text legalizer failed\n", .{});
            try stdout.flush();
            std.process.exit(1);
        }

        m.begin("sir text emission");
        const sir_text_ref = c.oraEmitSIRText(ctx, final_module);
        defer if (sir_text_ref.data != null) {
            const mlir_c = @import("mlir_c_api");
            mlir_c.freeStringRef(sir_text_ref);
        };
        if (sir_text_ref.data == null or sir_text_ref.length == 0) {
            m.end();
            try stdout.print("Failed to emit SIR text\n", .{});
            return;
        }
        m.end();

        // Extract source locations from SIR MLIR (sidecar to SIR text).
        // Op indices match the SIR text op ordering for later correlation
        // with Sensei's bytecode offsets.
        const mlir_c_api = @import("mlir_c_api");
        const sir_locations_ref = c.oraExtractSIRLocations(ctx, final_module);
        defer if (sir_locations_ref.data != null) {
            mlir_c_api.freeStringRef(sir_locations_ref);
        };
        const sir_locations: ?[]const u8 = if (sir_locations_ref.data != null and sir_locations_ref.length > 0)
            sir_locations_ref.data[0..sir_locations_ref.length]
        else
            null;

        const sir_debug_info_ref: c.MlirStringRef = if (mlir_options.debug_info)
            c.oraExtractSIRDebugInfo(ctx, final_module)
        else
            .{ .data = null, .length = 0 };
        defer if (sir_debug_info_ref.data != null) {
            mlir_c_api.freeStringRef(sir_debug_info_ref);
        };
        const sir_debug_info: ?[]const u8 = if (sir_debug_info_ref.data != null and sir_debug_info_ref.length > 0)
            sir_debug_info_ref.data[0..sir_debug_info_ref.length]
        else
            null;

        const sir_line_map_ref = c.oraExtractSIRLineMap(ctx, final_module);
        defer if (sir_line_map_ref.data != null) {
            mlir_c_api.freeStringRef(sir_line_map_ref);
        };
        const sir_line_map: ?[]const u8 = if (sir_line_map_ref.data != null and sir_line_map_ref.length > 0)
            sir_line_map_ref.data[0..sir_line_map_ref.length]
        else
            null;

        const sir_text = sir_text_ref.data[0..sir_text_ref.length];
        const sir_global_slots_ref: c.MlirStringRef = if (mlir_options.debug_info)
            c.oraExtractSIRGlobalSlots(ctx, final_module)
        else
            .{ .data = null, .length = 0 };
        defer if (sir_global_slots_ref.data != null) {
            mlir_c_api.freeStringRef(sir_global_slots_ref);
        };
        const sir_global_slots_json: ?[]const u8 = if (sir_global_slots_ref.data != null and sir_global_slots_ref.length > 0)
            sir_global_slots_ref.data[0..sir_global_slots_ref.length]
        else
            null;
        var debug_global_slots = try parseDebugGlobalSlots(allocator, sir_global_slots_json);
        if (debug_global_slots.count() == 0 and mlir_options.debug_info) {
            deinitDebugGlobalSlots(allocator, &debug_global_slots);
            debug_global_slots = try parseDebugGlobalSlotsFromSirText(allocator, sir_text);
        }
        defer deinitDebugGlobalSlots(allocator, &debug_global_slots);
        const source_scopes = if (mlir_options.debug_info)
            try buildSourceScopeDebugInfo(allocator, &compilation.db, compilation.root_module_id, &debug_global_slots)
        else
            null;
        defer if (source_scopes) |bundle| {
            deinitDebugSourceScopeBundle(allocator, &bundle);
        };

        if (mlir_options.emit_sir_text) {
            if (mlir_options.output_dir) |output_dir| {
                std.fs.cwd().makeDir(output_dir) catch |err| switch (err) {
                    error.PathAlreadyExists => {},
                    else => return err,
                };
                const basename = std.fs.path.stem(file_path);
                const filename = try std.mem.concat(allocator, u8, &[_][]const u8{ basename, ".sir" });
                defer allocator.free(filename);
                const output_file = try std.fs.path.join(allocator, &[_][]const u8{ output_dir, filename });
                defer allocator.free(output_file);
                var sir_file = try std.fs.cwd().createFile(output_file, .{});
                defer sir_file.close();
                try sir_file.writeAll(sir_text);
            } else {
                try stdout.print("{s}", .{sir_text});
            }
        }
        if (mlir_options.emit_bytecode) {
            if (mlir_options.emit_sir_text and mlir_options.output_dir == null) try stdout.print("\n", .{});
            m.begin("bytecode generation");
            try emitBytecodeFromSirText(allocator, &compilation.db, compilation.root_module_id, &compilation.db.sources, sir_text, file_path, mlir_options.output_dir, sir_locations, sir_line_map, sir_debug_info, source_scopes, stdout);
            m.end();
        }
    }

    if (pending_smt_report) |*report| {
        try writeSmtReportArtifacts(allocator, file_path, mlir_options.output_dir, report.*, stdout);
        report.deinit(mlir_allocator);
        pending_smt_report = null;
    }

    try stdout.flush();
    if (verification_failed) return error.VerificationFailed;
}

fn printVersion() !void {
    var stdout_buffer: [8192]u8 = undefined;
    var stdout_writer = std.fs.File.stdout().writer(&stdout_buffer);
    const stdout = &stdout_writer.interface;
    const logo =
        \\=================
        \\                                       +====-----------====
        \\                                      +++++==---------==+=+=+
        \\                                    ++++++++==-------=+++++++++
        \\                                  ++++++++++===-----==+++++++++++
        \\                    +++          *++++++++++=====-====++++++++++*#         ++++
        \\                  +++++++        ##*++++++++==========++++++++*###        +++++++
        \\                 ++++++++++      ####*++++++===========+++++**####      ++++++++++
        \\                   ++++++++++    ######*+++======+======+++*######    ++++++++++
        \\                    =++++++++    ########+=====+++++=====+########    ++++++++
        \\                    ++=++++++      ########+=++******+=+########      +++++++++
        \\                   =++++++***        ########*********########        +**+++++++
        \\                   ++++++++**      ############*****############      **++++++++
        \\             ======++++++++++    #######*########*#########*######    +++++++++++=====
        \\           =====+++++++++++++    ######   ###############   ######    +++++++++++++=====
        \\         ======++++++++++++++    ######     ###########     ######    +++++++++++++++=====
        \\         ++==++++++   +++++++    #######    ###########    *######    +++++++   +++++++=++
        \\         ++++++++   ++++++++*      #############################      +++++++++   ++++++++
        \\         +++++++    =++++++          #########################          +++++++    +++++++
        \\         ++++++++     -++         ###############################         +++    +++++++++
        \\         ++++++++++               ##%#########################%%%               ++++++++++
        \\         +++++++++*##      +###########################################       ##**++++++++
        \\           +++++*######    =######%############################%%######     ######*+++++
        \\             +*##########  =######%%%#########################%%%######   *#########*+
        \\               *###################%%%#######################%%%####################
        \\                **##################%%%%###################%%%%%%*################
        \\           ##     +*############  ##%%%%%%################%%%%%%%  **############     ##
        \\         #*####     +*########    #############################%%    **#######      %#####
        \\        #*######                  #%%#########################%%%                  ########
        \\      ##########%%%%%%%%   =#%%%%%%%%%%##*##################%%%%%%%%%%%    #%%%%%%%%#########
        \\    ############%%%%%%%%   =#%%%%%%%%%%% +################ %%%%%%%%%%%%%  =#%%%%%%%############
        \\   ########**###%%%%%%%%   =#%%%%%%%%%   +################   %%%%%%%%%%%  =#%%%%%%%####*#########
        \\ ##########  *#%%%%%%%%%%%%#%%%%%%%%%   *#%%%%%##*##%%%%%%    ##%%%%%%%%####%%%%%%%%##  *#########
        \\ *#######     ##%%%%%%%##%%%%%%%%%%   *##%%%%%%#* =*%%%%%%%%    ###%%%%%%%%%%%%%%%%%      *#######
        \\ ######         ##############%##    ###%%%%%##     +#%%%%%###    **#############%%        *######
        \\ ######## ##%                      ##########%        *#########                      ### *#######
        \\  *##########%                   *#########%            ##########                  %############
        \\   *########%%%%               ##########%    *%   %@    ##########                %%%##########
        \\     #######%%%     %%%%%%%%# ##########    %%%%   %%%@    ########## #%%%%%%%%     ##########
        \\      ######%      %%%%%%%%%% #########   %%%%%%   %%%%%    *######## #%%%%%%%%%      ######
        \\        ###      %%%%%%%%%%%@ #######    %%%%%%%   %%%%%%     *###### #%%%%%%%%%%%     *###
        \\               %%%%%%%%%@%@@@ ##%%%%%%    #%%%%%   %%%%%%    %%%%%%%% #%%%%%%%%%%%%%
        \\                %%%%%%%%%%%   %%%%%%%%% %%%%%%%%   %%%%%%%  %%%%%%%%%   %%%%%%%%%%%
        \\                              *#%%%%%%%%%%%%%%%%   %%%%%%%%%%%%%%%%%%
        \\                                #%%%%%%%%%%%%%%%   %%%%%%%%%%%%%%%%
        \\                                  %%%%%@@@@%%%%     #%%%@@@@%%%%%
        \\                                   %%@@@@@@@@@       #%@@@@@@@%
        \\                                     %@@@@@@           %@@@@@@
        \\                                       %@@@             %@@@
    ;
    try stdout.print("{s}\n", .{logo});
    try stdout.print("Ora Compiler v0.1 - Asuka\n", .{});
    try stdout.flush();
}

// ============================================================================
// SECTION 3: Command Handlers (lex, parse, ast, compile)
// ============================================================================

/// Print unified diff between original and formatted code
fn printUnifiedDiff(_: std.mem.Allocator, original: []const u8, formatted: []const u8, file_path: []const u8) !void {
    var stdout_buffer: [4096]u8 = undefined;
    var stdout_writer = std.fs.File.stdout().writer(&stdout_buffer);
    const stdout = &stdout_writer.interface;

    try stdout.print("--- {s}\n", .{file_path});
    try stdout.print("+++ {s}\n", .{file_path});

    // Simple line-by-line diff
    var orig_lines = std.mem.splitScalar(u8, original, '\n');
    var fmt_lines = std.mem.splitScalar(u8, formatted, '\n');

    var line_num: u32 = 1;
    var orig_line = orig_lines.next();
    var fmt_line = fmt_lines.next();

    while (orig_line != null or fmt_line != null) {
        const orig = orig_line orelse "";
        const fmt = fmt_line orelse "";

        if (!std.mem.eql(u8, orig, fmt)) {
            if (orig_line != null) {
                try stdout.print("-{d}: {s}\n", .{ line_num, orig });
            }
            if (fmt_line != null) {
                try stdout.print("+{d}: {s}\n", .{ line_num, fmt });
            }
        } else {
            try stdout.print(" {d}: {s}\n", .{ line_num, orig });
        }

        if (orig_line != null) orig_line = orig_lines.next();
        if (fmt_line != null) fmt_line = fmt_lines.next();
        line_num += 1;
    }
}

/// Run formatter on file(s)
fn formatSingleFile(
    allocator: std.mem.Allocator,
    file_path: []const u8,
    check: bool,
    diff: bool,
    stdout: bool,
    options: anytype,
) !bool {
    const fmt_mod = @import("fmt/mod.zig");

    // Read source file
    const source = std.fs.cwd().readFileAlloc(allocator, file_path, std.math.maxInt(usize)) catch |err| {
        std.debug.print("error: cannot read file '{s}': {s}\n", .{ file_path, @errorName(err) });
        std.process.exit(1);
    };
    defer allocator.free(source);

    // Format
    var formatter = fmt_mod.Formatter.init(allocator, source, options);
    defer formatter.deinit();

    const formatted = formatter.format() catch |err| {
        std.debug.print("error: failed to format {s}: {}\n", .{ file_path, err });
        std.process.exit(2);
    };
    defer allocator.free(formatted);

    // Check if already formatted
    const already_formatted = std.mem.eql(u8, source, formatted);

    if (check) {
        if (!already_formatted) {
            std.debug.print("{s} needs formatting\n", .{file_path});
            return true;
        }
        return false;
    }

    if (diff) {
        if (!already_formatted) {
            // Generate unified diff
            try printUnifiedDiff(allocator, source, formatted, file_path);
            return true;
        }
        return false;
    }

    if (stdout) {
        try std.fs.File.stdout().writeAll(formatted);
        return false;
    }

    // Write formatted output
    if (!already_formatted) {
        try std.fs.cwd().writeFile(.{ .sub_path = file_path, .data = formatted });
    }
    return false;
}

fn formatDirectoryRecursive(
    allocator: std.mem.Allocator,
    dir_path: []const u8,
    check: bool,
    diff: bool,
    options: anytype,
    found_mismatch: *bool,
) !void {
    var dir = try std.fs.cwd().openDir(dir_path, .{ .iterate = true });
    defer dir.close();

    var iter = dir.iterate();
    while (try iter.next()) |entry| {
        const child_path = try std.fs.path.join(allocator, &.{ dir_path, entry.name });
        defer allocator.free(child_path);

        switch (entry.kind) {
            .directory => try formatDirectoryRecursive(allocator, child_path, check, diff, options, found_mismatch),
            .file => {
                if (!std.mem.endsWith(u8, entry.name, ".ora")) continue;
                if (try formatSingleFile(allocator, child_path, check, diff, false, options)) {
                    found_mismatch.* = true;
                }
            },
            else => {},
        }
    }
}

fn runFmt(allocator: std.mem.Allocator, file_path: []const u8, check: bool, diff: bool, stdout: bool, width: ?u32) !void {
    const fmt_mod = @import("fmt/mod.zig");
    const FormatOptions = fmt_mod.FormatOptions;

    const options = FormatOptions{
        .line_width = width orelse 100,
        .indent_size = 4,
    };

    var dir_probe = std.fs.cwd().openDir(file_path, .{}) catch |err| switch (err) {
        error.NotDir => null,
        else => {
            std.debug.print("error: cannot access '{s}': {s}\n", .{ file_path, @errorName(err) });
            std.process.exit(1);
        },
    };
    if (dir_probe) |*dir| {
        dir.close();
        if (stdout) {
            std.debug.print("error: --stdout does not support directory inputs\n", .{});
            std.process.exit(2);
        }

        var found_mismatch = false;
        try formatDirectoryRecursive(allocator, file_path, check, diff, options, &found_mismatch);
        if (found_mismatch) {
            std.process.exit(1);
        }
        return;
    }

    if (try formatSingleFile(allocator, file_path, check, diff, stdout, options)) {
        std.process.exit(1);
    }
}

// ============================================================================
// SECTION 5: Parser & Compilation Workflows
// ============================================================================

/// Generate Ora ABI outputs
fn runAbiEmit(
    allocator: std.mem.Allocator,
    file_path: []const u8,
    output_dir: ?[]const u8,
    emit_abi: bool,
    emit_abi_solidity: bool,
    emit_abi_extras: bool,
    resolver_options: import_graph.ResolverOptions,
    debug_enabled: bool,
) !void {
    var stdout_buffer: [1024]u8 = undefined;
    var stdout_writer = std.fs.File.stdout().writer(&stdout_buffer);
    const stdout = &stdout_writer.interface;

    var compilation = compiler.driver.compilePackageWithResolverOptions(allocator, file_path, resolver_options) catch |err| {
        try stdout.print("Compiler error: {s}\n", .{@errorName(err)});
        return;
    };
    defer compilation.deinit();

    _ = try exitOnCompilationErrors(stdout, &compilation.db, compilation.root_module_id, debug_enabled);

    var contract_abi = try lib.abi.generateCompilerAbi(allocator, &compilation);
    defer contract_abi.deinit();

    const base_name = std.fs.path.stem(file_path);
    const emit_sidecar = emit_abi_extras or (emit_abi and output_dir != null);

    if (output_dir) |out_dir| {
        std.fs.cwd().makeDir(out_dir) catch |err| switch (err) {
            error.PathAlreadyExists => {},
            else => return err,
        };
        if (emit_abi) {
            const abi_json = try contract_abi.toJson(allocator);
            defer allocator.free(abi_json);
            const pretty_json = try formatJsonPretty(allocator, abi_json);
            defer allocator.free(pretty_json);
            const abi_path = try std.fmt.allocPrint(allocator, "{s}/{s}.abi.json", .{ out_dir, base_name });
            defer allocator.free(abi_path);
            var abi_file = try std.fs.cwd().createFile(abi_path, .{});
            defer abi_file.close();
            try abi_file.writeAll(pretty_json);
        }
        if (emit_abi_solidity) {
            const abi_json = try contract_abi.toSolidityJson(allocator);
            defer allocator.free(abi_json);
            const pretty_json = try formatJsonPretty(allocator, abi_json);
            defer allocator.free(pretty_json);
            const abi_path = try std.fmt.allocPrint(allocator, "{s}/{s}.abi.sol.json", .{ out_dir, base_name });
            defer allocator.free(abi_path);
            var abi_file = try std.fs.cwd().createFile(abi_path, .{});
            defer abi_file.close();
            try abi_file.writeAll(pretty_json);
        }
        if (emit_sidecar) {
            const extras_json = try contract_abi.toExtrasJson(allocator);
            defer allocator.free(extras_json);
            const pretty_json = try formatJsonPretty(allocator, extras_json);
            defer allocator.free(pretty_json);
            const extras_path = try std.fmt.allocPrint(allocator, "{s}/{s}.abi.extras.json", .{ out_dir, base_name });
            defer allocator.free(extras_path);
            var extras_file = try std.fs.cwd().createFile(extras_path, .{});
            defer extras_file.close();
            try extras_file.writeAll(pretty_json);
        }
    } else {
        if (emit_abi) {
            const abi_json = try contract_abi.toJson(allocator);
            defer allocator.free(abi_json);
            const pretty_json = try formatJsonPretty(allocator, abi_json);
            defer allocator.free(pretty_json);
            try stdout.print("{s}\n", .{pretty_json});
        }
        if (emit_abi_solidity) {
            const abi_json = try contract_abi.toSolidityJson(allocator);
            defer allocator.free(abi_json);
            const pretty_json = try formatJsonPretty(allocator, abi_json);
            defer allocator.free(pretty_json);
            try stdout.print("{s}\n", .{pretty_json});
        }
        if (emit_abi_extras) {
            const extras_json = try contract_abi.toExtrasJson(allocator);
            defer allocator.free(extras_json);
            const pretty_json = try formatJsonPretty(allocator, extras_json);
            defer allocator.free(pretty_json);
            try stdout.print("{s}\n", .{pretty_json});
        }
        try stdout.flush();
    }
}

fn formatJsonPretty(allocator: std.mem.Allocator, json_text: []const u8) ![]u8 {
    var parsed = try std.json.parseFromSlice(std.json.Value, allocator, json_text, .{});
    defer parsed.deinit();

    return try std.json.Stringify.valueAlloc(allocator, parsed.value, .{ .whitespace = .indent_2 });
}

fn writeSmtReportArtifacts(
    allocator: std.mem.Allocator,
    file_path: []const u8,
    output_dir: ?[]const u8,
    report: @import("z3/mod.zig").SmtReportArtifacts,
    stdout: anytype,
) !void {
    const base_name = std.fs.path.stem(file_path);
    const md_name = try std.fmt.allocPrint(allocator, "{s}.smt.report.md", .{base_name});
    defer allocator.free(md_name);
    const json_name = try std.fmt.allocPrint(allocator, "{s}.smt.report.json", .{base_name});
    defer allocator.free(json_name);

    var md_path_buf: ?[]u8 = null;
    defer if (md_path_buf) |buf| allocator.free(buf);
    var json_path_buf: ?[]u8 = null;
    defer if (json_path_buf) |buf| allocator.free(buf);

    const md_path = if (output_dir) |dir| blk: {
        std.fs.cwd().makeDir(dir) catch |err| switch (err) {
            error.PathAlreadyExists => {},
            else => return err,
        };
        md_path_buf = try std.fs.path.join(allocator, &[_][]const u8{ dir, md_name });
        break :blk md_path_buf.?;
    } else md_name;

    const json_path = if (output_dir) |dir| blk: {
        json_path_buf = try std.fs.path.join(allocator, &[_][]const u8{ dir, json_name });
        break :blk json_path_buf.?;
    } else json_name;

    var md_file = try std.fs.cwd().createFile(md_path, .{});
    defer md_file.close();
    try md_file.writeAll(report.markdown);

    const pretty_json = try formatJsonPretty(allocator, report.json);
    defer allocator.free(pretty_json);

    var json_file = try std.fs.cwd().createFile(json_path, .{});
    defer json_file.close();
    try json_file.writeAll(pretty_json);

    try stdout.print("SMT report saved to {s}\n", .{md_path});
    try stdout.print("SMT report JSON saved to {s}\n", .{json_path});
}

// ============================================================================
// SECTION 6: MLIR Integration & Code Generation
// ============================================================================

fn resolveSenseiSirPath(allocator: std.mem.Allocator) ![]const u8 {
    if (std.posix.getenv("ORA_SENSEI_SIR")) |path| {
        return allocator.dupe(u8, path);
    }

    const default_path = "vendor/sensei/senseic/target/release/sir";
    if (std.fs.cwd().access(default_path, .{}) catch null) |_| {
        return allocator.dupe(u8, default_path);
    }

    return error.SenseiSirNotFound;
}

fn emitBytecodeFromSirText(
    allocator: std.mem.Allocator,
    db: *compiler.CompilerDb,
    root_module_id: compiler.ModuleId,
    sources: *const compiler.source.SourceStore,
    sir_text: []const u8,
    file_path: []const u8,
    output_dir: ?[]const u8,
    sir_locations_json: ?[]const u8,
    sir_line_map_json: ?[]const u8,
    sir_debug_info_json: ?[]const u8,
    source_scopes: ?DebugSourceScopeBundle,
    stdout: anytype,
) !void {
    const sir_path = try resolveSenseiSirPath(allocator);
    defer allocator.free(sir_path);

    const basename = std.fs.path.stem(file_path);
    const sir_extension = ".sir";
    const sir_filename = try std.mem.concat(allocator, u8, &[_][]const u8{ basename, sir_extension });
    defer allocator.free(sir_filename);

    const temp_root = "/tmp/ora_sir";
    std.fs.makeDirAbsolute(temp_root) catch |err| switch (err) {
        error.PathAlreadyExists => {},
        else => return err,
    };
    const temp_dir = try std.fmt.allocPrint(allocator, "{s}/{x}", .{ temp_root, std.crypto.random.int(u64) });
    defer allocator.free(temp_dir);
    std.fs.makeDirAbsolute(temp_dir) catch |err| switch (err) {
        error.PathAlreadyExists => {},
        else => return err,
    };
    const sir_file_path = try std.fs.path.join(allocator, &[_][]const u8{ temp_dir, sir_filename });
    defer allocator.free(sir_file_path);

    var sir_file = try std.fs.createFileAbsolute(sir_file_path, .{});
    defer sir_file.close();
    try sir_file.writeAll(sir_text);

    // If we have source locations, tell Sensei to produce a source map
    const sensei_srcmap_path = if (sir_locations_json != null)
        try std.fs.path.join(allocator, &[_][]const u8{ temp_dir, "sensei_srcmap.json" })
    else
        null;
    defer if (sensei_srcmap_path) |p| allocator.free(p);

    // Build argv: sir <input> [--source-map <path>]
    var argv_buf: [4][]const u8 = undefined;
    var argc: usize = 0;
    argv_buf[argc] = sir_path;
    argc += 1;
    argv_buf[argc] = sir_file_path;
    argc += 1;
    if (sensei_srcmap_path) |srcmap_path| {
        argv_buf[argc] = "--source-map";
        argc += 1;
        argv_buf[argc] = srcmap_path;
        argc += 1;
    }
    const argv = argv_buf[0..argc];
    var child = std.process.Child.init(argv, allocator);
    child.stdin_behavior = .Ignore;
    child.stdout_behavior = .Pipe;
    child.stderr_behavior = .Pipe;

    try child.spawn();

    const max_output = 16 * 1024 * 1024;
    const stdout_bytes = try child.stdout.?.readToEndAlloc(allocator, max_output);
    defer allocator.free(stdout_bytes);
    const stderr_bytes = try child.stderr.?.readToEndAlloc(allocator, max_output);
    defer allocator.free(stderr_bytes);

    const term = try child.wait();
    switch (term) {
        .Exited => |code| {
            if (code != 0) {
                try stdout.print("\nError: sensei bytecode compilation failed (exit code {d})\n", .{code});
                // Show sensei's error output (the actual error reason)
                const trimmed_stderr = std.mem.trim(u8, stderr_bytes, " \n\r\t");
                if (trimmed_stderr.len > 0) {
                    // Extract the meaningful error line (skip Rust backtrace noise)
                    var line_iter = std.mem.splitScalar(u8, trimmed_stderr, '\n');
                    while (line_iter.next()) |line| {
                        const trimmed = std.mem.trim(u8, line, " \r\t");
                        if (trimmed.len == 0) continue;
                        // Skip Rust backtrace/thread info noise
                        if (std.mem.startsWith(u8, trimmed, "note: run with")) continue;
                        if (std.mem.startsWith(u8, trimmed, "stack backtrace:")) continue;
                        if (trimmed.len > 3 and trimmed[0] >= '0' and trimmed[0] <= '9' and trimmed[1] == ':') continue;
                        try stdout.print("  {s}\n", .{trimmed});
                    }
                } else {
                    const trimmed_stdout_err = std.mem.trim(u8, stdout_bytes, " \n\r\t");
                    if (trimmed_stdout_err.len > 0) {
                        try stdout.print("  {s}\n", .{trimmed_stdout_err});
                    }
                }
                try stdout.print("SIR input saved to: {s}\n", .{sir_file_path});
                try stdout.flush();
                std.process.exit(1);
            }
        },
        else => {
            try stdout.print("\nError: sensei bytecode compiler terminated unexpectedly\n", .{});
            try stdout.print("SIR input saved to: {s}\n", .{sir_file_path});
            try stdout.flush();
            std.process.exit(1);
        },
    }

    const bytecode = std.mem.trim(u8, stdout_bytes, " \n\r\t");
    if (bytecode.len == 0) {
        try stdout.print("\nError: sensei produced empty bytecode\n", .{});
        try stdout.print("SIR input saved to: {s}\n", .{sir_file_path});
        try stdout.flush();
        std.process.exit(1);
    }

    var bytecode_output_path: ?[]const u8 = null;
    defer if (bytecode_output_path) |p| allocator.free(p);

    if (output_dir) |out_dir| {
        const out_is_file = std.mem.endsWith(u8, out_dir, ".hex") or
            std.mem.endsWith(u8, out_dir, ".bytecode") or
            std.mem.endsWith(u8, out_dir, ".bin");

        if (out_is_file) {
            var out_file = if (std.fs.path.isAbsolute(out_dir))
                try std.fs.createFileAbsolute(out_dir, .{})
            else
                try std.fs.cwd().createFile(out_dir, .{});
            defer out_file.close();
            try out_file.writeAll(bytecode);
            bytecode_output_path = try allocator.dupe(u8, out_dir);
            try stdout.print("Bytecode saved to {s}\n", .{out_dir});
        } else {
            std.fs.cwd().makeDir(out_dir) catch |err| switch (err) {
                error.PathAlreadyExists => {},
                else => return err,
            };

            const extension = ".hex";
            const filename = try std.mem.concat(allocator, u8, &[_][]const u8{ basename, extension });
            defer allocator.free(filename);
            const output_file = try std.fs.path.join(allocator, &[_][]const u8{ out_dir, filename });
            defer allocator.free(output_file);

            var out_file = try std.fs.cwd().createFile(output_file, .{});
            defer out_file.close();
            try out_file.writeAll(bytecode);
            bytecode_output_path = try allocator.dupe(u8, output_file);
            try stdout.print("Bytecode saved to {s}\n", .{output_file});
        }
    } else {
        try stdout.print("{s}\n", .{bytecode});
    }

    // Merge source maps: sir_locations (op_idx → file:line:col) + sensei (op_idx → pc)
    // into final .sourcemap.json
    if (sir_locations_json != null and sensei_srcmap_path != null and bytecode_output_path != null) {
        try mergeSourceMaps(allocator, db, root_module_id, sir_locations_json.?, sir_line_map_json, sir_debug_info_json, sensei_srcmap_path.?, bytecode_output_path.?, stdout);
    }
    if (sir_debug_info_json != null and bytecode_output_path != null) {
        try writeDebugInfoSidecar(allocator, db, root_module_id, sources, sir_debug_info_json.?, source_scopes, bytecode_output_path.?, stdout);
    }
}

fn mergeSourceMaps(
    allocator: std.mem.Allocator,
    db: *compiler.CompilerDb,
    root_module_id: compiler.ModuleId,
    sir_locations_json: []const u8,
    sir_line_map_json: ?[]const u8,
    sir_debug_info_json: ?[]const u8,
    sensei_srcmap_path: []const u8,
    bytecode_output_path: []const u8,
    stdout: anytype,
) !void {
    // Read Sensei's source map: {"runtime_start_pc":123,"ops":[{"idx":0,"pc":0},{"idx":1,"pc":5},...]}
    const sensei_json = std.fs.cwd().readFileAlloc(allocator, sensei_srcmap_path, 16 * 1024 * 1024) catch |err| {
        try stdout.print("Warning: could not read Sensei source map: {}\n", .{err});
        return;
    };
    defer allocator.free(sensei_json);

    // Parse Sensei ops into a map: op_idx → pc
    const SenseiMap = struct {
        runtime_start_pc: ?u32 = null,
        ops: []const struct { idx: u32, pc: u32 } = &.{},
    };
    const sensei_parsed = std.json.parseFromSlice(SenseiMap, allocator, sensei_json, .{
        .ignore_unknown_fields = true,
    }) catch |err| {
        try stdout.print("Warning: could not parse Sensei source map: {}\n", .{err});
        return;
    };
    defer sensei_parsed.deinit();

    // Build lookup: op_idx → pc
    var idx_to_pc = std.AutoHashMap(u32, u32).init(allocator);
    defer idx_to_pc.deinit();
    var max_mapped_idx: u32 = 0;
    var have_mapped_idx = false;
    for (sensei_parsed.value.ops) |op| {
        try idx_to_pc.put(op.idx, op.pc);
        if (!have_mapped_idx or op.idx > max_mapped_idx) {
            max_mapped_idx = op.idx;
            have_mapped_idx = true;
        }
    }

    // Parse SIR locations: [{"idx":0,"file":"main.ora","line":3,"col":5},...]
    const LocEntry = struct { idx: u32, file: []const u8, line: u32, col: u32 };
    const sir_locs = std.json.parseFromSlice([]const LocEntry, allocator, sir_locations_json, .{
        .ignore_unknown_fields = true,
    }) catch |err| {
        try stdout.print("Warning: could not parse SIR locations: {}\n", .{err});
        return;
    };
    defer sir_locs.deinit();

    var idx_to_sir_line = std.AutoHashMap(u32, u32).init(allocator);
    defer idx_to_sir_line.deinit();
    if (sir_line_map_json) |json| {
        const SirLineEntry = struct { idx: u32, line: u32 };
        const sir_lines = std.json.parseFromSlice([]const SirLineEntry, allocator, json, .{
            .ignore_unknown_fields = true,
        }) catch |err| {
            try stdout.print("Warning: could not parse SIR line map: {}\n", .{err});
            return;
        };
        defer sir_lines.deinit();
        for (sir_lines.value) |line_entry| {
            try idx_to_sir_line.put(line_entry.idx, line_entry.line);
        }
    }

    const OpProvenance = struct {
        op: ?[]const u8 = null,
        function: ?[]const u8 = null,
        statement_id: ?u32 = null,
        origin_statement_id: ?u32 = null,
        execution_region_id: ?u32 = null,
        statement_run_index: ?u32 = null,
        is_hoisted: ?bool = null,
        is_duplicated: ?bool = null,
        is_synthetic: bool = false,
        synthetic_index: ?u32 = null,
        synthetic_count: ?u32 = null,
        synthetic_path: ?[]const u8 = null,
    };
    var idx_to_provenance = std.AutoHashMap(u32, OpProvenance).init(allocator);
    defer idx_to_provenance.deinit();
    if (sir_debug_info_json) |json| {
        const DebugOp = struct {
            idx: u32,
            op: ?[]const u8 = null,
            function: ?[]const u8 = null,
            statement_id: ?u32 = null,
            origin_statement_id: ?u32 = null,
            execution_region_id: ?u32 = null,
            statement_run_index: ?u32 = null,
            is_hoisted: ?bool = null,
            is_duplicated: ?bool = null,
            is_synthetic: bool = false,
            synthetic_index: ?u32 = null,
            synthetic_count: ?u32 = null,
            synthetic_path: ?[]const u8 = null,
        };
        const DebugInfoParse = struct {
            ops: []const DebugOp = &.{},
        };
        const parsed_debug = std.json.parseFromSlice(DebugInfoParse, allocator, json, .{
            .ignore_unknown_fields = true,
        }) catch |err| {
            try stdout.print("Warning: could not parse SIR debug info for source-map provenance: {}\n", .{err});
            return;
        };
        defer parsed_debug.deinit();
        for (parsed_debug.value.ops) |op| {
            try idx_to_provenance.put(op.idx, .{
                .op = op.op,
                .function = op.function,
                .statement_id = op.statement_id,
                .origin_statement_id = op.origin_statement_id,
                .execution_region_id = op.execution_region_id,
                .statement_run_index = op.statement_run_index,
                .is_hoisted = op.is_hoisted,
                .is_duplicated = op.is_duplicated,
                .is_synthetic = op.is_synthetic,
                .synthetic_index = op.synthetic_index,
                .synthetic_count = op.synthetic_count,
                .synthetic_path = op.synthetic_path,
            });
        }
    }

    var executable_lines = try buildExecutableLineMap(allocator, db, root_module_id);
    defer deinitExecutableLineMap(allocator, &executable_lines);

    // Merge: for each SIR location entry, look up its PC from Sensei
    // Collect unique source files
    var sources: std.ArrayList([]const u8) = .{};
    defer sources.deinit(allocator);
    var source_indices = std.StringHashMap(u32).init(allocator);
    defer source_indices.deinit();

    // Build merged entries
    var out: std.ArrayList(u8) = .{};
    defer out.deinit(allocator);
    const writer = out.writer(allocator);

    try writer.print("{{\"version\":1,\"runtime_start_pc\":{},\"sources\":[", .{sensei_parsed.value.runtime_start_pc orelse 0});

    // First pass: collect sources and build entries
    const MergedEntry = struct {
        idx: u32,
        pc: u32,
        src: u32,
        line: u32,
        col: u32,
        stmt_id: ?u32,
        origin_stmt_id: ?u32,
        execution_region_id: ?u32,
        statement_run_index: ?u32,
        function_name: ?[]const u8,
        sir_line: ?u32,
        is_synthetic: bool = false,
        synthetic_index: ?u32 = null,
        synthetic_count: ?u32 = null,
        synthetic_path: ?[]const u8 = null,
        is_hoisted: bool = false,
        is_duplicated: bool = false,
        stmt: bool,
        kind: ?ExecutableStatementKind = null,
    };
    var entries: std.ArrayList(MergedEntry) = .{};
    defer entries.deinit(allocator);

    var missing_idx_count: usize = 0;
    var missing_idx_examples: [8]u32 = undefined;
    var missing_idx_examples_len: usize = 0;
    var untranslated_tail_count: usize = 0;

    for (sir_locs.value) |loc| {
        const pc = idx_to_pc.get(loc.idx) orelse {
            if (have_mapped_idx and loc.idx > max_mapped_idx) {
                untranslated_tail_count += 1;
            } else {
                missing_idx_count += 1;
                if (missing_idx_examples_len < missing_idx_examples.len) {
                    missing_idx_examples[missing_idx_examples_len] = loc.idx;
                    missing_idx_examples_len += 1;
                }
            }
            continue;
        };

        // Get or create source index
        const src_idx = source_indices.get(loc.file) orelse blk: {
            const idx: u32 = @intCast(sources.items.len);
            try sources.append(allocator, loc.file);
            try source_indices.put(loc.file, idx);
            break :blk idx;
        };

        try entries.append(allocator, .{
            .idx = loc.idx,
            .pc = pc,
            .src = src_idx,
            .line = loc.line,
            .col = loc.col,
            .stmt_id = null,
            .origin_stmt_id = null,
            .execution_region_id = if (idx_to_provenance.get(loc.idx)) |p| p.execution_region_id else null,
            .statement_run_index = if (idx_to_provenance.get(loc.idx)) |p| p.statement_run_index else null,
            .function_name = if (idx_to_provenance.get(loc.idx)) |p| p.function else null,
            .sir_line = idx_to_sir_line.get(loc.idx),
            .is_synthetic = if (idx_to_provenance.get(loc.idx)) |p| p.is_synthetic else false,
            .synthetic_index = if (idx_to_provenance.get(loc.idx)) |p| p.synthetic_index else null,
            .synthetic_count = if (idx_to_provenance.get(loc.idx)) |p| p.synthetic_count else null,
            .synthetic_path = if (idx_to_provenance.get(loc.idx)) |p| p.synthetic_path else null,
            .is_hoisted = false,
            .is_duplicated = false,
            .stmt = false,
            .kind = null,
        });
    }

    if (missing_idx_count != 0) {
        try stdout.print("Error: source map index mismatch ({d} MLIR location entries had no bytecode PC)\n", .{missing_idx_count});
        if (missing_idx_examples_len != 0) {
            try stdout.print("  missing op indices:", .{});
            for (missing_idx_examples[0..missing_idx_examples_len]) |idx| {
                try stdout.print(" {d}", .{idx});
            }
            try stdout.print("\n", .{});
        }
        try stdout.flush();
        return error.SourceMapMismatch;
    }

    if (untranslated_tail_count != 0) {
        try stdout.print("Note: dropped {d} untranslated tail source locations beyond backend op index {d}\n", .{
            untranslated_tail_count,
            max_mapped_idx,
        });
    }

    // Sort entries by PC
    std.mem.sort(MergedEntry, entries.items, {}, struct {
        fn lessThan(_: void, a: MergedEntry, b: MergedEntry) bool {
            if (a.pc != b.pc) return a.pc < b.pc;
            if (a.idx != b.idx) return a.idx < b.idx;
            if (a.src != b.src) return a.src < b.src;
            if (a.line != b.line) return a.line < b.line;
            return a.col < b.col;
        }
    }.lessThan);

    var prev_stmt_id: ?u32 = null;
    var prev_line: ?u32 = null;
    var prev_col: ?u32 = null;
    var prev_src: ?u32 = null;
    for (entries.items) |*entry| {
        const file_path = sources.items[entry.src];
        const stmt_meta = executableStatementMetaForPosition(&executable_lines, file_path, entry.line, entry.col);
        const provenance = idx_to_provenance.get(entry.idx);
        entry.stmt_id = if (provenance) |p| p.statement_id else if (stmt_meta) |meta| meta.stmt_id else null;
        entry.origin_stmt_id = if (provenance) |p| p.origin_statement_id orelse entry.stmt_id else entry.stmt_id;
        entry.kind = if (stmt_meta) |meta| meta.kind else null;
        entry.is_hoisted = if (provenance) |p| p.is_hoisted orelse false else false;
        entry.is_duplicated = if (provenance) |p| p.is_duplicated orelse false else false;
        entry.stmt = if (stmt_meta) |meta|
            if (meta.stmt_id) |stmt_id|
                (prev_stmt_id == null) or (stmt_id != prev_stmt_id.?) or (entry.src != prev_src.?)
            else
                (prev_line == null) or (entry.line != prev_line.?) or (entry.col != prev_col.?) or (entry.src != prev_src.?)
        else
            false;
        prev_stmt_id = entry.stmt_id;
        prev_line = entry.line;
        prev_col = entry.col;
        prev_src = entry.src;
    }

    var stmt_has_non_invalid = std.AutoHashMap(u32, bool).init(allocator);
    defer stmt_has_non_invalid.deinit();
    for (entries.items) |entry| {
        const stmt_id = entry.stmt_id orelse continue;
        const provenance = idx_to_provenance.get(entry.idx) orelse continue;
        const op_name = provenance.op orelse continue;
        if (std.mem.eql(u8, op_name, "invalid") or std.mem.eql(u8, op_name, "sir.invalid")) continue;
        try stmt_has_non_invalid.put(stmt_id, true);
    }

    for (entries.items) |*entry| {
        const stmt_id = entry.stmt_id orelse continue;
        const provenance = idx_to_provenance.get(entry.idx) orelse continue;
        const op_name = provenance.op orelse continue;
        if (!(std.mem.eql(u8, op_name, "invalid") or std.mem.eql(u8, op_name, "sir.invalid"))) continue;
        if (stmt_has_non_invalid.get(stmt_id) orelse false) {
            entry.stmt = false;
        }
    }

    const StmtAnchor = struct {
        stmt_entry_index: usize,
        stmt_is_invalid: bool,
        first_non_invalid_index: ?usize,
    };
    suppressHoistedDuplicateStmtMarkers(entries.items);

    var stmt_anchors = std.AutoHashMap(u32, StmtAnchor).init(allocator);
    defer stmt_anchors.deinit();
    for (entries.items, 0..) |entry, entry_index| {
        const stmt_id = entry.stmt_id orelse continue;
        const provenance = idx_to_provenance.get(entry.idx);
        const is_invalid_op = if (provenance) |p|
            if (p.op) |op_name|
                std.mem.eql(u8, op_name, "invalid") or std.mem.eql(u8, op_name, "sir.invalid")
            else
                false
        else
            false;
        var anchor = stmt_anchors.get(stmt_id) orelse StmtAnchor{
            .stmt_entry_index = std.math.maxInt(usize),
            .stmt_is_invalid = false,
            .first_non_invalid_index = null,
        };
        if (entry.stmt and anchor.stmt_entry_index == std.math.maxInt(usize)) {
            anchor.stmt_entry_index = entry_index;
            anchor.stmt_is_invalid = is_invalid_op;
        }
        if (!is_invalid_op and anchor.first_non_invalid_index == null) {
            anchor.first_non_invalid_index = entry_index;
        }
        try stmt_anchors.put(stmt_id, anchor);
    }
    var stmt_anchor_it = stmt_anchors.iterator();
    while (stmt_anchor_it.next()) |it| {
        const anchor = it.value_ptr.*;
        if (anchor.stmt_entry_index == std.math.maxInt(usize)) continue;
        if (!anchor.stmt_is_invalid) continue;
        const replacement_index = anchor.first_non_invalid_index orelse continue;
        entries.items[anchor.stmt_entry_index].stmt = false;
        entries.items[replacement_index].stmt = true;
    }

    var stmt_marker_counts = std.AutoHashMap(u32, u32).init(allocator);
    defer stmt_marker_counts.deinit();
    for (entries.items) |entry| {
        if (!entry.stmt) continue;
        const stmt_id = entry.stmt_id orelse continue;
        const existing = stmt_marker_counts.get(stmt_id) orelse 0;
        try stmt_marker_counts.put(stmt_id, existing + 1);
    }
    for (entries.items) |*entry| {
        if (entry.stmt_id) |stmt_id| {
            const count = stmt_marker_counts.get(stmt_id) orelse 0;
            if (!entry.is_duplicated) entry.is_duplicated = count > 1;
        }
    }

    // Emit sources array
    for (sources.items, 0..) |src, i| {
        if (i > 0) try writer.writeAll(",");
        try writeJsonString(writer, src);
    }
    try writer.writeAll("],\"entries\":[");

    // Emit entries
    for (entries.items, 0..) |entry, i| {
        if (i > 0) try writer.writeAll(",");
        try writer.print("{{\"idx\":{d},\"pc\":{d},\"src\":{d},\"line\":{d},\"col\":{d},\"stmt\":{s}", .{
            entry.idx, entry.pc, entry.src, entry.line, entry.col, if (entry.stmt) "true" else "false",
        });
        if (entry.stmt_id) |stmt_id| {
            try writer.print(",\"statement_id\":{d}", .{stmt_id});
        }
        if (entry.origin_stmt_id) |origin_stmt_id| {
            try writer.print(",\"origin_statement_id\":{d}", .{origin_stmt_id});
        }
        if (entry.execution_region_id) |execution_region_id| {
            try writer.print(",\"execution_region_id\":{d}", .{execution_region_id});
        }
        if (entry.statement_run_index) |statement_run_index| {
            try writer.print(",\"statement_run_index\":{d}", .{statement_run_index});
        }
        if (entry.function_name) |function_name| {
            try writer.writeAll(",\"function\":");
            try writeJsonString(writer, function_name);
        }
        if (entry.sir_line) |sir_line| {
            try writer.print(",\"sir_line\":{d}", .{sir_line});
        }
        if (entry.is_synthetic) {
            try writer.writeAll(",\"is_synthetic\":true");
            if (entry.synthetic_index) |synthetic_index| {
                try writer.print(",\"synthetic_index\":{d}", .{synthetic_index});
            }
            if (entry.synthetic_count) |synthetic_count| {
                try writer.print(",\"synthetic_count\":{d}", .{synthetic_count});
            }
            if (entry.synthetic_path) |synthetic_path| {
                try writer.writeAll(",\"synthetic_path\":");
                try writeJsonString(writer, synthetic_path);
            }
        }
        if (entry.is_hoisted) {
            try writer.writeAll(",\"is_hoisted\":true");
        }
        if (entry.is_duplicated) {
            try writer.writeAll(",\"is_duplicated\":true");
        }
        if (entry.kind) |kind| {
            try writer.writeAll(",\"kind\":");
            try writeJsonString(writer, executableStatementKindName(kind));
        }
        try writer.writeAll("}");
    }
    try writer.writeAll("]}");

    // Write the merged source map next to the bytecode artifact.
    const basename = std.fs.path.stem(bytecode_output_path);
    const srcmap_name = try std.mem.concat(allocator, u8, &[_][]const u8{ basename, ".sourcemap.json" });
    defer allocator.free(srcmap_name);
    const srcmap_path = if (std.fs.path.dirname(bytecode_output_path)) |dir|
        try std.fs.path.join(allocator, &[_][]const u8{ dir, srcmap_name })
    else
        try allocator.dupe(u8, srcmap_name);
    defer allocator.free(srcmap_path);

    var srcmap_file = try std.fs.cwd().createFile(srcmap_path, .{});
    defer srcmap_file.close();
    try srcmap_file.writeAll(out.items);

    try stdout.print("Source map saved to {s} ({d} entries)\n", .{ srcmap_path, entries.items.len });
}

fn writeDebugInfoSidecar(
    allocator: std.mem.Allocator,
    db: *compiler.CompilerDb,
    root_module_id: compiler.ModuleId,
    sources: *const compiler.source.SourceStore,
    sir_debug_info_json: []const u8,
    source_scopes: ?DebugSourceScopeBundle,
    bytecode_output_path: []const u8,
    stdout: anytype,
) !void {
    const DebugOp = struct {
        idx: u32,
        op: []const u8,
        function: []const u8,
        block: []const u8,
        file: ?[]const u8 = null,
        line: ?u32 = null,
        col: ?u32 = null,
        result_names: []const []const u8 = &.{},
        is_terminator: bool = false,
        is_synthetic: bool = false,
        synthetic_index: ?u32 = null,
        synthetic_count: ?u32 = null,
        synthetic_path: ?[]const u8 = null,
        statement_id: ?u32 = null,
        origin_statement_id: ?u32 = null,
        execution_region_id: ?u32 = null,
        statement_run_index: ?u32 = null,
        is_hoisted: ?bool = null,
        is_duplicated: ?bool = null,
    };
    const DebugInfoParse = struct {
        version: u32 = 1,
        ops: []const DebugOp = &.{},
    };

    const basename = std.fs.path.stem(bytecode_output_path);
    const debug_name = try std.mem.concat(allocator, u8, &[_][]const u8{ basename, ".debug.json" });
    defer allocator.free(debug_name);
    const debug_path = if (std.fs.path.dirname(bytecode_output_path)) |dir|
        try std.fs.path.join(allocator, &[_][]const u8{ dir, debug_name })
    else
        try allocator.dupe(u8, debug_name);
    defer allocator.free(debug_path);

    var debug_file = try std.fs.cwd().createFile(debug_path, .{});
    defer debug_file.close();
    var out: std.ArrayList(u8) = .{};
    defer out.deinit(allocator);
    const writer = out.writer(allocator);
    const parsed_debug = std.json.parseFromSlice(DebugInfoParse, allocator, sir_debug_info_json, .{
        .ignore_unknown_fields = true,
    }) catch |err| {
        try stdout.print("Warning: could not parse debug sidecar for visibility enrichment: {}\n", .{err});
        return;
    };
    defer parsed_debug.deinit();

    var executable_lines = try buildExecutableLineMap(allocator, db, root_module_id);
    defer deinitExecutableLineMap(allocator, &executable_lines);

    const EnrichedOp = struct {
        idx: u32,
        op: []const u8,
        function: []const u8,
        block: []const u8,
        file: ?[]const u8 = null,
        line: ?u32 = null,
        col: ?u32 = null,
        result_names: []const []const u8 = &.{},
        is_terminator: bool = false,
        is_synthetic: bool = false,
        synthetic_index: ?u32 = null,
        synthetic_count: ?u32 = null,
        synthetic_path: ?[]const u8 = null,
        statement_id: ?u32 = null,
        origin_statement_id: ?u32 = null,
        execution_region_id: ?u32 = null,
        statement_run_index: ?u32 = null,
        kind: ?ExecutableStatementKind = null,
        is_hoisted: bool = false,
        is_duplicated: bool = false,
    };
    var enriched_ops = try allocator.alloc(EnrichedOp, parsed_debug.value.ops.len);
    defer allocator.free(enriched_ops);

    for (parsed_debug.value.ops, 0..) |op, i| {
        const stmt_meta = if (op.file != null and op.line != null and op.col != null)
            executableStatementMetaForPosition(&executable_lines, op.file.?, op.line.?, op.col.?)
        else
            null;
        enriched_ops[i] = .{
            .idx = op.idx,
            .op = op.op,
            .function = op.function,
            .block = op.block,
            .file = op.file,
            .line = op.line,
            .col = op.col,
            .result_names = op.result_names,
            .is_terminator = op.is_terminator,
            .is_synthetic = op.is_synthetic,
            .synthetic_index = op.synthetic_index,
            .synthetic_count = op.synthetic_count,
            .synthetic_path = op.synthetic_path,
            .statement_id = op.statement_id orelse if (stmt_meta) |meta| meta.stmt_id else null,
            .origin_statement_id = op.origin_statement_id orelse op.statement_id orelse if (stmt_meta) |meta| meta.stmt_id else null,
            .execution_region_id = op.execution_region_id,
            .statement_run_index = op.statement_run_index,
            .kind = if (stmt_meta) |meta| meta.kind else null,
            .is_hoisted = op.is_hoisted orelse false,
            .is_duplicated = op.is_duplicated orelse false,
        };
    }

    try writer.print("{{\"version\":{d},\"ops\":[", .{if (parsed_debug.value.version < 2) @as(u32, 2) else parsed_debug.value.version});
    for (enriched_ops, 0..) |op, i| {
        if (i != 0) try writer.writeAll(",");
        try writer.print("{{\"idx\":{d},\"op\":", .{op.idx});
        try writeJsonString(writer, op.op);
        try writer.writeAll(",\"function\":");
        try writeJsonString(writer, op.function);
        try writer.writeAll(",\"block\":");
        try writeJsonString(writer, op.block);
        try writer.writeAll(",\"file\":");
        if (op.file) |file| {
            try writeJsonString(writer, file);
        } else {
            try writer.writeAll("null");
        }
        try writer.writeAll(",\"line\":");
        if (op.line) |line| try writer.print("{d}", .{line}) else try writer.writeAll("null");
        try writer.writeAll(",\"col\":");
        if (op.col) |col| try writer.print("{d}", .{col}) else try writer.writeAll("null");
        try writer.writeAll(",\"result_names\":[");
        for (op.result_names, 0..) |name, name_index| {
            if (name_index != 0) try writer.writeAll(",");
            try writeJsonString(writer, name);
        }
        try writer.print("],\"is_terminator\":{s},\"is_synthetic\":{s}", .{
            if (op.is_terminator) "true" else "false",
            if (op.is_synthetic) "true" else "false",
        });
        try writer.writeAll(",\"statement_id\":");
        if (op.statement_id) |statement_id| try writer.print("{d}", .{statement_id}) else try writer.writeAll("null");
        try writer.writeAll(",\"origin_statement_id\":");
        if (op.origin_statement_id) |origin_statement_id| try writer.print("{d}", .{origin_statement_id}) else try writer.writeAll("null");
        try writer.writeAll(",\"execution_region_id\":");
        if (op.execution_region_id) |execution_region_id| try writer.print("{d}", .{execution_region_id}) else try writer.writeAll("null");
        try writer.writeAll(",\"statement_run_index\":");
        if (op.statement_run_index) |statement_run_index| try writer.print("{d}", .{statement_run_index}) else try writer.writeAll("null");
        if (op.synthetic_index) |synthetic_index| {
            try writer.print(",\"synthetic_index\":{d}", .{synthetic_index});
        }
        if (op.synthetic_count) |synthetic_count| {
            try writer.print(",\"synthetic_count\":{d}", .{synthetic_count});
        }
        if (op.synthetic_path) |synthetic_path| {
            try writer.writeAll(",\"synthetic_path\":");
            try writeJsonString(writer, synthetic_path);
        }
        if (op.kind) |kind| {
            try writer.writeAll(",\"kind\":");
            try writeJsonString(writer, executableStatementKindName(kind));
        }
        if (op.is_hoisted) try writer.writeAll(",\"is_hoisted\":true");
        if (op.is_duplicated) try writer.writeAll(",\"is_duplicated\":true");
        try writer.writeAll("}");
    }
    try writer.writeAll("]");

    try writer.writeAll(",\"source_scopes\":[");
    if (source_scopes) |scope_info| {
        for (scope_info.scopes, 0..) |scope, scope_index| {
            if (scope_index != 0) try writer.writeAll(",");
            try writer.print("{{\"id\":{d},\"parent\":", .{scope.id});
            if (scope.parent) |parent| {
                try writer.print("{d}", .{parent});
            } else {
                try writer.writeAll("null");
            }
            try writer.writeAll(",\"file\":");
            try writeJsonString(writer, scope.file_path);
            try writer.writeAll(",\"function\":");
            try writeJsonString(writer, scope.function_name);
            try writer.writeAll(",\"contract\":");
            if (scope.contract_name) |contract_name| {
                try writeJsonString(writer, contract_name);
            } else {
                try writer.writeAll("null");
            }
            try writer.writeAll(",\"kind\":");
            try writeJsonString(writer, scope.kind);
            try writer.writeAll(",\"label\":");
            if (scope.label) |label| {
                try writeJsonString(writer, label);
            } else {
                try writer.writeAll("null");
            }
            try writer.writeAll(",\"range\":");
            try writeDebugRangeJson(writer, sources, scope.file_id, scope.range);
            // Reserved schema slot for the future inliner — see
            // `lib/evm/src/debug_info.zig:InlinedFrame`. The compiler
            // doesn't perform MLIR-level function inlining yet so
            // this is always empty; emit it explicitly so artifacts
            // written today round-trip cleanly through the
            // schema-aware loader.
            try writer.writeAll(",\"inlined_from\":[]");
            try writer.writeAll(",\"locals\":[");
            var first_local = true;
            for (scope_info.locals[scope.local_start .. scope.local_start + scope.local_count]) |local| {
                if (!first_local) try writer.writeAll(",");
                first_local = false;
                try writer.print("{{\"id\":{d},\"name\":", .{local.id});
                try writeJsonString(writer, local.name);
                try writer.writeAll(",\"kind\":");
                try writeJsonString(writer, local.kind);
                try writer.writeAll(",\"binding_kind\":");
                if (local.binding_kind) |binding_kind| {
                    try writeJsonString(writer, binding_kind);
                } else {
                    try writer.writeAll("null");
                }
                try writer.writeAll(",\"storage_class\":");
                if (local.storage_class) |storage_class| {
                    try writeJsonString(writer, storage_class);
                } else {
                    try writer.writeAll("null");
                }
                try writer.writeAll(",\"runtime\":{");
                try writer.writeAll("\"kind\":");
                try writeJsonString(writer, local.runtime_kind);
                try writer.writeAll(",\"name\":");
                if (local.runtime_name) |runtime_name| {
                    try writeJsonString(writer, runtime_name);
                } else {
                    try writer.writeAll("null");
                }
                try writer.writeAll(",\"location\":");
                if (local.runtime_location_kind) |runtime_location_kind| {
                    try writer.writeAll("{\"kind\":");
                    try writeJsonString(writer, runtime_location_kind);
                    try writer.writeAll(",\"root\":");
                    if (local.runtime_location_root) |runtime_location_root| {
                        try writeJsonString(writer, runtime_location_root);
                    } else {
                        try writer.writeAll("null");
                    }
                    try writer.writeAll(",\"slot\":");
                    if (local.runtime_location_slot) |runtime_location_slot| {
                        try writer.print("{d}", .{runtime_location_slot});
                    } else {
                        try writer.writeAll("null");
                    }
                    try writer.writeAll("}");
                } else {
                    try writer.writeAll("null");
                }
                try writer.print(",\"editable\":{s}", .{if (local.editable) "true" else "false"});
                try writer.writeAll("}");
                try writer.writeAll(",\"folded_value\":");
                if (local.folded_value) |folded_value| {
                    try writeJsonString(writer, folded_value);
                } else {
                    try writer.writeAll("null");
                }
                try writer.print(",\"is_folded\":{s}", .{if (local.is_folded) "true" else "false"});
                try writer.writeAll(",\"decl\":");
                try writeDebugRangeJson(writer, sources, local.file_id, local.decl_range);
                try writer.writeAll(",\"live\":");
                try writeDebugRangeJson(writer, sources, local.file_id, local.live_range);
                try writer.writeAll("}");
            }
            try writer.writeAll("]}");
        }
    }
    try writer.writeAll("],\"op_visibility\":[");
    if (source_scopes) |scope_info| {
        var first_visibility = true;
        for (parsed_debug.value.ops) |op| {
            const file = op.file orelse continue;
            const line = op.line orelse continue;
            const col = op.col orelse continue;

            var visible_scope_count: usize = 0;
            var visible_local_count: usize = 0;
            for (scope_info.scopes) |scope| {
                if (!std.mem.eql(u8, scope.file_path, file)) continue;
                if (!posInRange(sources, scope.file_id, scope.range, line, col)) continue;
                visible_scope_count += 1;
                for (scope_info.locals[scope.local_start .. scope.local_start + scope.local_count]) |local| {
                    if (posInRange(sources, local.file_id, local.live_range, line, col)) {
                        visible_local_count += 1;
                    }
                }
            }
            if (visible_scope_count == 0 and visible_local_count == 0) continue;

            if (!first_visibility) try writer.writeAll(",");
            first_visibility = false;
            try writer.print("{{\"idx\":{d},\"scope_ids\":[", .{op.idx});

            var first_scope = true;
            for (scope_info.scopes) |scope| {
                if (!std.mem.eql(u8, scope.file_path, file)) continue;
                if (!posInRange(sources, scope.file_id, scope.range, line, col)) continue;
                if (!first_scope) try writer.writeAll(",");
                first_scope = false;
                try writer.print("{d}", .{scope.id});
            }

            try writer.writeAll("],\"visible_local_ids\":[");
            var first_local_id = true;
            for (scope_info.scopes) |scope| {
                if (!std.mem.eql(u8, scope.file_path, file)) continue;
                if (!posInRange(sources, scope.file_id, scope.range, line, col)) continue;
                for (scope_info.locals[scope.local_start .. scope.local_start + scope.local_count]) |local| {
                    if (!posInRange(sources, local.file_id, local.live_range, line, col)) continue;
                    if (!first_local_id) try writer.writeAll(",");
                    first_local_id = false;
                    try writer.print("{d}", .{local.id});
                }
            }
            try writer.writeAll("]}");
        }
    }
    try writer.writeAll("]}");
    try debug_file.writeAll(out.items);

    try stdout.print("Debug info saved to {s}\n", .{debug_path});
}

test "build config init_args: validates init parameters" {
    const allocator = std.testing.allocator;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();

    try tmp.dir.writeFile(.{
        .sub_path = "Main.ora",
        .data =
        \\contract Main {
        \\    pub fn init(seed: u256, enabled: bool) {
        \\        let x: u256 = seed;
        \\        if (enabled) {
        \\            let y: u256 = x;
        \\        }
        \\    }
        \\
        \\    pub fn run() -> u256 {
        \\        return 1;
        \\    }
        \\}
        ,
    });

    const entry_path = try std.fmt.allocPrint(allocator, ".zig-cache/tmp/{s}/Main.ora", .{tmp.sub_path});
    defer allocator.free(entry_path);

    const init_args = [_]project_config.InitArg{
        .{ .name = "seed", .value = "10" },
        .{ .name = "enabled", .value = "true" },
    };

    try validateConfiguredInitArgs(allocator, entry_path, .{}, init_args[0..]);
}

test "build config init_args: unknown init arg errors" {
    const allocator = std.testing.allocator;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();

    try tmp.dir.writeFile(.{
        .sub_path = "Main.ora",
        .data =
        \\contract Main {
        \\    pub fn init(seed: u256) {
        \\        let x: u256 = seed;
        \\        if (x > 0) {
        \\            let y: u256 = x;
        \\        }
        \\    }
        \\}
        ,
    });

    const entry_path = try std.fmt.allocPrint(allocator, ".zig-cache/tmp/{s}/Main.ora", .{tmp.sub_path});
    defer allocator.free(entry_path);

    const init_args = [_]project_config.InitArg{
        .{ .name = "missing", .value = "10" },
    };

    try std.testing.expectError(error.UnknownInitArg, validateConfiguredInitArgs(allocator, entry_path, .{}, init_args[0..]));
}

test "build config init_args: missing init function errors" {
    const allocator = std.testing.allocator;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();

    try tmp.dir.writeFile(.{
        .sub_path = "Main.ora",
        .data =
        \\contract Main {
        \\    pub fn run() -> u256 {
        \\        return 1;
        \\    }
        \\}
        ,
    });

    const entry_path = try std.fmt.allocPrint(allocator, ".zig-cache/tmp/{s}/Main.ora", .{tmp.sub_path});
    defer allocator.free(entry_path);

    const init_args = [_]project_config.InitArg{
        .{ .name = "seed", .value = "10" },
    };

    try std.testing.expectError(error.InitArgsRequireInitFunction, validateConfiguredInitArgs(allocator, entry_path, .{}, init_args[0..]));
}

test "build config init_args: invalid constructor shape errors" {
    const allocator = std.testing.allocator;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();

    try tmp.dir.writeFile(.{
        .sub_path = "Main.ora",
        .data =
        \\contract Main {
        \\    fn init(seed: u256) -> bool {
        \\        return true;
        \\    }
        \\}
        ,
    });

    const entry_path = try std.fmt.allocPrint(allocator, ".zig-cache/tmp/{s}/Main.ora", .{tmp.sub_path});
    defer allocator.free(entry_path);

    const init_args = [_]project_config.InitArg{
        .{ .name = "seed", .value = "10" },
    };

    try std.testing.expectError(error.InvalidInitFunction, validateConfiguredInitArgs(allocator, entry_path, .{}, init_args[0..]));
}

test "init command: scaffolds new project layout" {
    const allocator = std.testing.allocator;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();

    const target_path = try std.fmt.allocPrint(allocator, ".zig-cache/tmp/{s}/demo", .{tmp.sub_path});
    defer allocator.free(target_path);

    try initProjectLayout(target_path);

    const toml_path = try std.fmt.allocPrint(allocator, "{s}/ora.toml", .{target_path});
    defer allocator.free(toml_path);
    const contract_path = try std.fmt.allocPrint(allocator, "{s}/contracts/main.ora", .{target_path});
    defer allocator.free(contract_path);
    const readme_path = try std.fmt.allocPrint(allocator, "{s}/README.md", .{target_path});
    defer allocator.free(readme_path);

    const toml = try std.fs.cwd().readFileAlloc(allocator, toml_path, 64 * 1024);
    defer allocator.free(toml);
    const contract = try std.fs.cwd().readFileAlloc(allocator, contract_path, 64 * 1024);
    defer allocator.free(contract);
    const readme = try std.fs.cwd().readFileAlloc(allocator, readme_path, 64 * 1024);
    defer allocator.free(readme);

    try std.testing.expect(std.mem.indexOf(u8, toml, "[[targets]]") != null);
    try std.testing.expect(std.mem.indexOf(u8, toml, "init_args") != null);
    try std.testing.expect(std.mem.indexOf(u8, contract, "contract Main") != null);
    try std.testing.expect(std.mem.indexOf(u8, readme, "ora build") != null);
}

test "init command: rejects non-empty target directory" {
    const allocator = std.testing.allocator;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();

    try tmp.dir.makePath("occupied");
    try tmp.dir.writeFile(.{
        .sub_path = "occupied/existing.txt",
        .data = "keep",
    });

    const target_path = try std.fmt.allocPrint(allocator, ".zig-cache/tmp/{s}/occupied", .{tmp.sub_path});
    defer allocator.free(target_path);

    try std.testing.expectError(error.InitTargetNotEmpty, initProjectLayout(target_path));
}

test "compiler typed AST writer prints root item types" {
    const allocator = std.testing.allocator;
    const source_text =
        \\fn id(x: u256) -> u256 {
        \\    return x;
        \\}
    ;

    var compilation = try compiler.driver.compileSource(allocator, "test.ora", source_text);
    defer compilation.deinit();

    const module = compilation.db.sources.module(compilation.root_module_id);
    const ast_file = try compilation.db.astFile(module.file_id);
    const typecheck = try compilation.db.moduleTypeCheck(compilation.root_module_id);

    var buffer: std.ArrayList(u8) = .{};
    defer buffer.deinit(allocator);
    try writeCompilerAst(allocator, buffer.writer(allocator), ast_file, typecheck, &typecheck.diagnostics, "tree", false);

    try std.testing.expect(std.mem.indexOf(u8, buffer.items, "Compiler typed AST") != null);
    try std.testing.expect(std.mem.indexOf(u8, buffer.items, "Function id : fn(u256) -> u256") != null);
}

test "compiler JSON AST writer includes diagnostics array" {
    const allocator = std.testing.allocator;
    const source_text =
        \\fn bad() -> u256 {
        \\    return true;
        \\}
    ;

    var compilation = try compiler.driver.compileSource(allocator, "bad.ora", source_text);
    defer compilation.deinit();

    const module = compilation.db.sources.module(compilation.root_module_id);
    const ast_file = try compilation.db.astFile(module.file_id);
    const typecheck = try compilation.db.moduleTypeCheck(compilation.root_module_id);

    var buffer: std.ArrayList(u8) = .{};
    defer buffer.deinit(allocator);
    try writeCompilerAst(allocator, buffer.writer(allocator), ast_file, typecheck, &typecheck.diagnostics, "json", false);

    try std.testing.expect(std.mem.indexOf(u8, buffer.items, "\"root_items\"") != null);
    try std.testing.expect(std.mem.indexOf(u8, buffer.items, "\"diagnostics\"") != null);
}

test "simple test" {
    var list = ManagedArrayList(i32).init(std.testing.allocator);
    defer list.deinit(); // Try commenting this out and see if zig detects the memory leak!
    try list.append(42);
    try std.testing.expectEqual(@as(i32, 42), list.pop());
}

test "use lexer module" {
    var lexer = lib.Lexer.init(std.testing.allocator, "contract Test {}");
    defer lexer.deinit();

    const tokens = try lexer.scanTokens();
    defer std.testing.allocator.free(tokens);

    // should have at least: contract, Test, {, }, EOF = 5 tokens
    try std.testing.expect(tokens.len >= 5);
    try std.testing.expect(tokens[0].type == lib.TokenType.Contract);
    try std.testing.expect(tokens[1].type == lib.TokenType.Identifier);
    try std.testing.expect(tokens[tokens.len - 1].type == lib.TokenType.Eof);
}

test "fuzz example" {
    const Context = struct {
        fn testOne(context: @This(), input: []const u8) anyerror!void {
            _ = context;
            // try passing `--fuzz` to `zig build test` and see if it manages to fail this test case!
            try std.testing.expect(!std.mem.eql(u8, "canyoufindme", input));
        }
    };
    try std.testing.fuzz(Context{}, Context.testOne, .{});
}
