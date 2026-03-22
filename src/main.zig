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
    canonicalize: bool = true,
    validate_mlir: bool = true,
    verify_z3: bool = true,
    verify_mode: ?[]const u8 = null,
    verify_calls: ?bool = null,
    verify_state: ?bool = null,
    verify_stats: bool = false,
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
    Fmt,
};

const Subcommand = enum {
    None,
    Build,
    Emit,
    Fmt,
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
    else
        .None;

    const args_to_parse = switch (subcommand) {
        .Fmt, .Build, .Emit => args[2..],
        .None => args[1..],
    };

    var parsed = cli_args.parseArgs(args_to_parse) catch {
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
        .canonicalize = canonicalize_mlir,
        .validate_mlir = validate_mlir,
        .verify_z3 = verify_z3,
        .verify_mode = verify_mode,
        .verify_calls = verify_calls,
        .verify_state = verify_state,
        .verify_stats = verify_stats,
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
    try std.fs.cwd().rename(src_path, dst_path);
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
    try stdout.print("       ora fmt [fmt-options] <file.ora|dir>\n", .{});
    try stdout.print("       ora init [path]\n", .{});
    try stdout.print("       ora -v | --version\n", .{});
    try stdout.print("\nCompilation Control:\n", .{});
    try stdout.print("  (default), build       - Full compile + artifact bundle + SMT gate\n", .{});
    try stdout.print("                           If <file.ora> is omitted, builds all [[targets]] from ora.toml\n", .{});
    try stdout.print("                           Outputs: bytecode, ABI, Ora/SIR MLIR, SIR text, SMT report\n", .{});
    try stdout.print("                           Reads ora.toml [compiler].init_args and [[targets]].init_args\n", .{});
    try stdout.print("  emit                   - Debug emission mode (use --emit-*)\n", .{});
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
    try stdout.print("  --emit-smt-report      - Emit SMT encoding audit report (.md + .json)\n", .{});
    try stdout.print("  --debug                - Enable compiler debug output\n", .{});
    try stdout.print("  --metrics              - Print compilation phase timing report\n", .{});
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
    try stdout.print("\n❌ Verification failed with {d} error{s}:\n", .{
        errors.len,
        if (errors.len != 1) @as([]const u8, "s") else "",
    });
    try stdout.print("{s}\n", .{"=" ** 70});

    for (errors, 0..) |err, idx| {
        try stdout.print("\n  {d}. {s}\n", .{ idx + 1, err.message });
        if (err.counterexample) |ce| {
            var it = ce.variables.iterator();
            var has_vars = false;
            while (it.next()) |entry| {
                if (!has_vars) {
                    try stdout.print("     counterexample:\n", .{});
                    has_vars = true;
                }
                try stdout.print("       {s} = {s}\n", .{ entry.key_ptr.*, formatHexValue(entry.value_ptr.*) });
            }
        }
    }
    try stdout.print("\n", .{});
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
    try std.json.Stringify.value(text, .{}, writer);
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
        if (!c.oraConvertToSIR(lowering.context, lowering.module.raw_module)) {
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
        var verifier = try z3_verification.VerificationPass.init(mlir_allocator);
        defer verifier.deinit();

        if (mlir_options.verify_mode) |mode| {
            if (std.ascii.eqlIgnoreCase(mode, "full")) verifier.setVerifyMode(.Full) else verifier.setVerifyMode(.Basic);
        }
        if (mlir_options.verify_calls) |enabled| verifier.setVerifyCalls(enabled);
        if (mlir_options.verify_state) |enabled| verifier.setVerifyState(enabled);
        verifier.setVerifyStats(mlir_options.verify_stats);

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
        var verifier = try z3_verification.VerificationPass.init(mlir_allocator);
        defer verifier.deinit();
        if (mlir_options.verify_mode) |mode| {
            if (std.ascii.eqlIgnoreCase(mode, "full")) verifier.setVerifyMode(.Full) else verifier.setVerifyMode(.Basic);
        }
        if (mlir_options.verify_calls) |enabled| verifier.setVerifyCalls(enabled);
        if (mlir_options.verify_state) |enabled| verifier.setVerifyState(enabled);
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
        if (!c.oraConvertToSIR(ctx, final_module)) {
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

        const sir_text = sir_text_ref.data[0..sir_text_ref.length];
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
            try emitBytecodeFromSirText(allocator, sir_text, file_path, mlir_options.output_dir, stdout);
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
    sir_text: []const u8,
    file_path: []const u8,
    output_dir: ?[]const u8,
    stdout: anytype,
) !void {
    const sir_path = try resolveSenseiSirPath(allocator);
    defer allocator.free(sir_path);

    const basename = std.fs.path.stem(file_path);
    const sir_extension = ".sir";
    const sir_filename = try std.mem.concat(allocator, u8, &[_][]const u8{ basename, sir_extension });
    defer allocator.free(sir_filename);

    const temp_dir = "/tmp/ora_sir";
    std.fs.makeDirAbsolute(temp_dir) catch |err| switch (err) {
        error.PathAlreadyExists => {},
        else => return err,
    };
    const sir_file_path = try std.fs.path.join(allocator, &[_][]const u8{ temp_dir, sir_filename });
    defer allocator.free(sir_file_path);

    var sir_file = try std.fs.createFileAbsolute(sir_file_path, .{});
    defer sir_file.close();
    try sir_file.writeAll(sir_text);

    var argv = [_][]const u8{ sir_path, sir_file_path };
    var child = std.process.Child.init(&argv, allocator);
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
            try stdout.print("Bytecode saved to {s}\n", .{output_file});
        }
    } else {
        try stdout.print("{s}\n", .{bytecode});
    }
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
