// ============================================================================
// Build Script
// ============================================================================
//
// build configuration for the Ora compiler, tests, and toolchain integration.
//
// ============================================================================

const std = @import("std");

// Although this function looks imperative, note that its job is to
// declaratively construct a build graph that will be executed by an external
// runner.
pub fn build(b: *std.Build) void {
    // standard target options allows the person running `zig build` to choose
    // what target to build for. Here we do not override the defaults, which
    // means any target is allowed, and the default is native. Other options
    // for restricting supported target set are available.
    const target = b.standardTargetOptions(.{});

    // standard optimization options allow the person running `zig build` to select
    // between Debug, ReleaseSafe, ReleaseFast, and ReleaseSmall. Here we do not
    // set a preferred release mode, allowing the user to decide how to optimize.
    const optimize = b.standardOptimizeOption(.{});

    // mlir-specific build options
    const enable_mlir_debug = b.option(bool, "mlir-debug", "Enable MLIR debug features and verification passes") orelse false;
    const enable_mlir_timing = b.option(bool, "mlir-timing", "Enable MLIR pass timing by default") orelse false;
    const mlir_opt_level = b.option([]const u8, "mlir-opt", "Default MLIR optimization level (none, basic, aggressive)") orelse "basic";
    const enable_mlir_passes = b.option([]const u8, "mlir-passes", "Default MLIR pass pipeline") orelse null;
    const skip_mlir_build = b.option(bool, "skip-mlir", "Skip MLIR/SIR/Ora dialect CMake builds (use existing libs)") orelse false;

    // this creates a "module", which represents a collection of source files alongside
    // some compilation options, such as optimization mode and linked system libraries.
    // every executable or library we compile will be based on one or more modules.
    const lib_mod = b.createModule(.{
        // `root_source_file` is the Zig "entry point" of the module. If a module
        // only contains e.g. external object files, you can make this `null`.
        // in this case the main source file is merely a path, however, in more
        // complicated build scripts, this could be a generated file.
        .root_source_file = b.path("src/root.zig"),
        .target = target,
        .optimize = optimize,
    });

    // we will also create a module for our other entry point, 'main.zig'.
    const exe_mod = b.createModule(.{
        // `root_source_file` is the Zig "entry point" of the module. If a module
        // only contains e.g. external object files, you can make this `null`.
        // in this case the main source file is merely a path, however, in more
        // complicated build scripts, this could be a generated file.
        .root_source_file = b.path("src/main.zig"),
        .target = target,
        .optimize = optimize,
    });
    const log_mod = b.createModule(.{
        .root_source_file = b.path("src/logging.zig"),
        .target = target,
        .optimize = optimize,
    });

    const mlir_c_mod = b.createModule(.{
        .root_source_file = b.path("src/mlir/c.zig"),
        .target = target,
        .optimize = optimize,
    });
    mlir_c_mod.addIncludePath(b.path("vendor/mlir/include"));
    mlir_c_mod.addIncludePath(b.path("src/mlir/ora/include"));
    mlir_c_mod.addIncludePath(b.path("src/mlir/IR/include"));

    // modules can depend on one another using the `std.Build.Module.addImport` function.
    // this is what allows Zig source code to use `@import("foo")` where 'foo' is not a
    // file path. In this case, we set up `exe_mod` to import `lib_mod`.
    exe_mod.addImport("ora_lib", lib_mod);
    exe_mod.addImport("mlir_c_api", mlir_c_mod);
    exe_mod.addImport("log", log_mod);
    lib_mod.addImport("mlir_c_api", mlir_c_mod);
    lib_mod.addImport("log", log_mod);

    // now, we will create a static library based on the module we created above.
    // this creates a `std.Build.Step.Compile`, which is the build step responsible
    // for actually invoking the compiler.
    const lib = b.addLibrary(.{
        .linkage = .static,
        .name = "ora",
        .root_module = lib_mod,
    });

    // this declares intent for the library to be installed into the standard
    // location when the user invokes the "install" step (the default step when
    // running `zig build`).
    b.installArtifact(lib);

    // this creates another `std.Build.Step.Compile`, but this one builds an executable
    // rather than a static library.
    const exe = b.addExecutable(.{
        .name = "ora",
        .root_module = exe_mod,
    });

    // add MLIR build options as compile-time constants
    const mlir_options = b.addOptions();
    mlir_options.addOption(bool, "mlir_debug", enable_mlir_debug);
    mlir_options.addOption(bool, "mlir_timing", enable_mlir_timing);
    mlir_options.addOption([]const u8, "mlir_opt_level", mlir_opt_level);
    if (enable_mlir_passes) |passes| {
        mlir_options.addOption(?[]const u8, "mlir_passes", passes);
    } else {
        mlir_options.addOption(?[]const u8, "mlir_passes", null);
    }

    exe.root_module.addOptions("build_options", mlir_options);
    lib_mod.addOptions("build_options", mlir_options);

    // add include paths
    exe.addIncludePath(b.path("src"));
    lib.addIncludePath(b.path("src"));

    // add Ora dialect include path (for OraCAPI.h)
    const ora_dialect_include_path = b.path("src/mlir/ora/include");
    exe.addIncludePath(ora_dialect_include_path);

    // add SIR dialect include path
    const sir_dialect_include_path = b.path("src/mlir/IR/include");
    exe.addIncludePath(sir_dialect_include_path);

    // build and link MLIR (required) - only for executable, not library
    const mlir_step = if (skip_mlir_build) null else buildMlirLibraries(b, target, optimize);
    // build SIR dialect first (Ora dialect depends on it)
    const sir_dialect_step = if (skip_mlir_build) null else buildSIRDialectLibrary(b, mlir_step.?, target, optimize);
    const ora_dialect_step = if (skip_mlir_build) null else buildOraDialectLibrary(b, mlir_step.?, sir_dialect_step.?, target, optimize);
    linkMlirLibraries(b, exe, mlir_step, ora_dialect_step, sir_dialect_step, target);

    // build and link Z3 (for formal verification) - only for executable
    const z3_step = buildZ3Libraries(b, target, optimize);
    linkZ3Libraries(b, exe, z3_step, target);

    // this declares intent for the executable to be installed into the
    // standard location when the user invokes the "install" step (the default
    // step when running `zig build`).
    b.installArtifact(exe);

    // this *creates* a Run step in the build graph, to be executed when another
    // step is evaluated that depends on it. The next line below will establish
    // such a dependency.
    const run_cmd = b.addRunArtifact(exe);

    // by making the run step depend on the install step, it will be run from the
    // installation directory rather than directly from within the cache directory.
    // this is not necessary, however, if the application depends on other installed
    // files, this ensures they will be present and in the expected location.
    run_cmd.step.dependOn(b.getInstallStep());

    // this allows the user to pass arguments to the application in the build
    // command itself, like this: `zig build run -- arg1 arg2 etc`
    if (b.args) |args| {
        run_cmd.addArgs(args);
    }

    // this creates a build step. It will be visible in the `zig build --help` menu,
    // and can be selected like this: `zig build run`
    // this will evaluate the `run` step rather than the default, which is "install".
    const run_step = b.step("run", "Run the app");
    run_step.dependOn(&run_cmd.step);

    // Sensei SIR CLI integration (vendored Rust tool)
    const sensei_root = "vendor/sensei/senseic";
    const sensei_cargo = b.fmt("{s}/Cargo.toml", .{sensei_root});
    if (std.fs.cwd().access(sensei_cargo, .{}) catch null) |_| {
        const sensei_build_cmd = b.addSystemCommand(&[_][]const u8{
            "cargo",
            "build",
            "-p",
            "sir-cli",
            "--release",
        });
        sensei_build_cmd.setCwd(b.path(sensei_root));

        const sensei_build_step = b.step("sensei-sir", "Build Sensei SIR CLI");
        sensei_build_step.dependOn(&sensei_build_cmd.step);

        if (@import("builtin").os.tag != .windows) {
            const sir_bin = b.fmt("{s}/target/release/sir", .{sensei_root});
            const sample_input = "tests/sensei/sample.sir";
            const e2e_script = b.fmt(
                "out=$({s} {s}); test \"${{out#0x}}\" != \"$out\" -a ${{#out}} -gt 2",
                .{ sir_bin, sample_input },
            );

            const sensei_e2e_cmd = b.addSystemCommand(&[_][]const u8{
                "bash",
                "-lc",
                e2e_script,
            });
            sensei_e2e_cmd.step.dependOn(&sensei_build_cmd.step);

            const sensei_e2e_step = b.step("sensei-e2e", "Run Sensei SIR -> bytecode smoke test");
            sensei_e2e_step.dependOn(&sensei_e2e_cmd.step);
        }
    }

    // create optimization demo executable
    const optimization_demo_mod = b.createModule(.{
        .root_source_file = b.path("examples/demos/optimization_demo.zig"),
        .target = target,
        .optimize = optimize,
    });
    optimization_demo_mod.addImport("ora_lib", lib_mod);

    const optimization_demo = b.addExecutable(.{
        .name = "optimization_demo",
        .root_module = optimization_demo_mod,
    });

    const run_optimization_demo = b.addRunArtifact(optimization_demo);
    run_optimization_demo.step.dependOn(b.getInstallStep());

    const optimization_demo_step = b.step("optimization-demo", "Run the optimization demo");
    optimization_demo_step.dependOn(&run_optimization_demo.step);

    // create formal verification demo executable
    const formal_verification_demo_mod = b.createModule(.{
        .root_source_file = b.path("examples/demos/formal_verification_demo.zig"),
        .target = target,
        .optimize = optimize,
    });
    formal_verification_demo_mod.addImport("ora_lib", lib_mod);

    const formal_verification_demo = b.addExecutable(.{
        .name = "formal_verification_demo",
        .root_module = formal_verification_demo_mod,
    });

    const run_formal_verification_demo = b.addRunArtifact(formal_verification_demo);
    run_formal_verification_demo.step.dependOn(b.getInstallStep());

    const formal_verification_demo_step = b.step("formal-verification-demo", "Run the formal verification demo");
    formal_verification_demo_step.dependOn(&run_formal_verification_demo.step);

    // mlir-specific build steps
    const mlir_debug_step = b.step("mlir-debug", "Build with MLIR debug features enabled");
    mlir_debug_step.dependOn(b.getInstallStep());

    const mlir_release_step = b.step("mlir-release", "Build with aggressive MLIR optimizations");
    mlir_release_step.dependOn(b.getInstallStep());

    // add step to test MLIR functionality
    const test_mlir_step = b.step("test-mlir", "Run MLIR-specific tests");
    test_mlir_step.dependOn(b.getInstallStep());

    const fast_step = b.step("fast", "Fast build (use -Dskip-mlir=true)");
    fast_step.dependOn(b.getInstallStep());

    // test suite - Unit tests are co-located with source files
    // tests are added to build.zig as they are created (e.g., src/lexer.test.zig)
    const test_step = b.step("test", "Run all tests");

    // lexer tests
    const lexer_test_mod = b.createModule(.{
        .root_source_file = b.path("src/lexer.test.zig"),
        .target = target,
        .optimize = optimize,
    });
    lexer_test_mod.addImport("ora_root", lib_mod);
    const lexer_tests = b.addTest(.{ .root_module = lexer_test_mod });
    test_step.dependOn(&b.addRunArtifact(lexer_tests).step);

    // lexer error recovery tests
    const error_recovery_test_mod = b.createModule(.{
        .root_source_file = b.path("src/lexer/error_recovery.test.zig"),
        .target = target,
        .optimize = optimize,
    });
    error_recovery_test_mod.addImport("ora_root", lib_mod);
    const error_recovery_tests = b.addTest(.{ .root_module = error_recovery_test_mod });
    test_step.dependOn(&b.addRunArtifact(error_recovery_tests).step);

    // semantics locals binder tests
    const locals_binder_test_mod = b.createModule(.{
        .root_source_file = b.path("src/semantics/locals_binder.test.zig"),
        .target = target,
        .optimize = optimize,
    });
    locals_binder_test_mod.addImport("ora_root", lib_mod);
    const locals_binder_tests = b.addTest(.{ .root_module = locals_binder_test_mod });
    test_step.dependOn(&b.addRunArtifact(locals_binder_tests).step);

    // cli argument parsing tests
    const cli_args_test_mod = b.createModule(.{
        .root_source_file = b.path("src/cli/args.test.zig"),
        .target = target,
        .optimize = optimize,
    });
    const cli_args_tests = b.addTest(.{ .root_module = cli_args_test_mod });
    test_step.dependOn(&b.addRunArtifact(cli_args_tests).step);

    // ABI tests
    const abi_test_mod = b.createModule(.{
        .root_source_file = b.path("src/abi.test.zig"),
        .target = target,
        .optimize = optimize,
    });
    abi_test_mod.addImport("ora_root", lib_mod);
    const abi_tests = b.addTest(.{ .root_module = abi_test_mod });
    test_step.dependOn(&b.addRunArtifact(abi_tests).step);

    // scanner tests - Numbers
    const numbers_test_mod = b.createModule(.{
        .root_source_file = b.path("src/lexer/scanners/numbers.test.zig"),
        .target = target,
        .optimize = optimize,
    });
    numbers_test_mod.addImport("ora_root", lib_mod);
    const numbers_tests = b.addTest(.{ .root_module = numbers_test_mod });
    test_step.dependOn(&b.addRunArtifact(numbers_tests).step);

    // scanner tests - Strings
    const strings_test_mod = b.createModule(.{
        .root_source_file = b.path("src/lexer/scanners/strings.test.zig"),
        .target = target,
        .optimize = optimize,
    });
    strings_test_mod.addImport("ora_root", lib_mod);
    const strings_tests = b.addTest(.{ .root_module = strings_test_mod });
    test_step.dependOn(&b.addRunArtifact(strings_tests).step);

    // scanner tests - Identifiers
    const identifiers_test_mod = b.createModule(.{
        .root_source_file = b.path("src/lexer/scanners/identifiers.test.zig"),
        .target = target,
        .optimize = optimize,
    });
    identifiers_test_mod.addImport("ora_root", lib_mod);
    const identifiers_tests = b.addTest(.{ .root_module = identifiers_test_mod });
    test_step.dependOn(&b.addRunArtifact(identifiers_tests).step);

    // parser tests - Expression Parser
    const expression_parser_test_mod = b.createModule(.{
        .root_source_file = b.path("src/parser/expression_parser.test.zig"),
        .target = target,
        .optimize = optimize,
    });
    expression_parser_test_mod.addImport("ora_root", lib_mod);
    const expression_parser_tests = b.addTest(.{ .root_module = expression_parser_test_mod });
    test_step.dependOn(&b.addRunArtifact(expression_parser_tests).step);

    // ast tests - AST Builder
    const ast_builder_test_mod = b.createModule(.{
        .root_source_file = b.path("src/ast/ast_builder.test.zig"),
        .target = target,
        .optimize = optimize,
    });
    ast_builder_test_mod.addImport("ora_root", lib_mod);
    const ast_builder_tests = b.addTest(.{ .root_module = ast_builder_test_mod });
    test_step.dependOn(&b.addRunArtifact(ast_builder_tests).step);

    // parser tests - Statement Parser
    const statement_parser_test_mod = b.createModule(.{
        .root_source_file = b.path("src/parser/statement_parser.test.zig"),
        .target = target,
        .optimize = optimize,
    });
    statement_parser_test_mod.addImport("ora_root", lib_mod);
    const statement_parser_tests = b.addTest(.{ .root_module = statement_parser_test_mod });
    test_step.dependOn(&b.addRunArtifact(statement_parser_tests).step);

    // parser tests - Parser Core
    const parser_core_test_mod = b.createModule(.{
        .root_source_file = b.path("src/parser/parser_core.test.zig"),
        .target = target,
        .optimize = optimize,
    });
    parser_core_test_mod.addImport("ora_root", lib_mod);
    const parser_core_tests = b.addTest(.{ .root_module = parser_core_test_mod });
    test_step.dependOn(&b.addRunArtifact(parser_core_tests).step);

    // parser tests - Declaration Parser
    const declaration_parser_test_mod = b.createModule(.{
        .root_source_file = b.path("src/parser/declaration_parser.test.zig"),
        .target = target,
        .optimize = optimize,
    });
    declaration_parser_test_mod.addImport("ora_root", lib_mod);
    const declaration_parser_tests = b.addTest(.{ .root_module = declaration_parser_test_mod });
    test_step.dependOn(&b.addRunArtifact(declaration_parser_tests).step);

    // parser tests - Type Parser
    const type_parser_test_mod = b.createModule(.{
        .root_source_file = b.path("src/parser/type_parser.test.zig"),
        .target = target,
        .optimize = optimize,
    });
    type_parser_test_mod.addImport("ora_root", lib_mod);
    const type_parser_tests = b.addTest(.{ .root_module = type_parser_test_mod });
    test_step.dependOn(&b.addRunArtifact(type_parser_tests).step);

    // z3 encoder tests (requires MLIR + Z3)
    const z3_encoder_test_mod = b.createModule(.{
        .root_source_file = b.path("src/z3/encoder.test.zig"),
        .target = target,
        .optimize = optimize,
    });
    z3_encoder_test_mod.addImport("mlir_c_api", mlir_c_mod);
    const z3_encoder_tests = b.addTest(.{ .root_module = z3_encoder_test_mod });
    linkMlirLibraries(b, z3_encoder_tests, mlir_step, ora_dialect_step, sir_dialect_step, target);
    linkZ3Libraries(b, z3_encoder_tests, z3_step, target);
    test_step.dependOn(&b.addRunArtifact(z3_encoder_tests).step);

    // mlir type mapper tests
    const mlir_types_test_mod = b.createModule(.{
        .root_source_file = b.path("src/mlir/types.test.zig"),
        .target = target,
        .optimize = optimize,
    });
    mlir_types_test_mod.addImport("ora_lib", lib_mod);
    mlir_types_test_mod.addImport("mlir_c_api", mlir_c_mod);
    mlir_types_test_mod.addImport("log", log_mod);
    const mlir_types_tests = b.addTest(.{ .root_module = mlir_types_test_mod });
    linkMlirLibraries(b, mlir_types_tests, mlir_step, ora_dialect_step, sir_dialect_step, target);
    test_step.dependOn(&b.addRunArtifact(mlir_types_tests).step);
    test_mlir_step.dependOn(&b.addRunArtifact(mlir_types_tests).step);

    // refinement guard tests (MLIR)
    const refinement_guard_test_mod = b.createModule(.{
        .root_source_file = b.path("src/mlir/refinements.test.zig"),
        .target = target,
        .optimize = optimize,
    });
    refinement_guard_test_mod.addImport("ora_lib", lib_mod);
    refinement_guard_test_mod.addImport("mlir_c_api", mlir_c_mod);
    refinement_guard_test_mod.addImport("log", log_mod);
    const refinement_guard_tests = b.addTest(.{ .root_module = refinement_guard_test_mod });
    linkMlirLibraries(b, refinement_guard_tests, mlir_step, ora_dialect_step, sir_dialect_step, target);
    test_step.dependOn(&b.addRunArtifact(refinement_guard_tests).step);
    test_mlir_step.dependOn(&b.addRunArtifact(refinement_guard_tests).step);

    // MLIR effect metadata tests
    const mlir_effects_test_mod = b.createModule(.{
        .root_source_file = b.path("src/mlir/effects.test.zig"),
        .target = target,
        .optimize = optimize,
    });
    mlir_effects_test_mod.addImport("ora_lib", lib_mod);
    mlir_effects_test_mod.addImport("mlir_c_api", mlir_c_mod);
    mlir_effects_test_mod.addImport("log", log_mod);
    const mlir_effects_tests = b.addTest(.{ .root_module = mlir_effects_test_mod });
    linkMlirLibraries(b, mlir_effects_tests, mlir_step, ora_dialect_step, sir_dialect_step, target);
    test_step.dependOn(&b.addRunArtifact(mlir_effects_tests).step);
    test_mlir_step.dependOn(&b.addRunArtifact(mlir_effects_tests).step);

    // ast tests - Expressions
    const ast_expressions_test_mod = b.createModule(.{
        .root_source_file = b.path("src/ast/expressions.test.zig"),
        .target = target,
        .optimize = optimize,
    });
    ast_expressions_test_mod.addImport("ora_root", lib_mod);
    const ast_expressions_tests = b.addTest(.{ .root_module = ast_expressions_test_mod });
    test_step.dependOn(&b.addRunArtifact(ast_expressions_tests).step);

    // ast tests - Statements
    const ast_statements_test_mod = b.createModule(.{
        .root_source_file = b.path("src/ast/statements.test.zig"),
        .target = target,
        .optimize = optimize,
    });
    ast_statements_test_mod.addImport("ora_root", lib_mod);
    const ast_statements_tests = b.addTest(.{ .root_module = ast_statements_test_mod });
    test_step.dependOn(&b.addRunArtifact(ast_statements_tests).step);

    // ast tests - Type Resolver (logs)
    const type_resolver_logs_test_mod = b.createModule(.{
        .root_source_file = b.path("src/ast/type_resolver_logs.test.zig"),
        .target = target,
        .optimize = optimize,
    });
    type_resolver_logs_test_mod.addImport("ora_root", lib_mod);
    const type_resolver_logs_tests = b.addTest(.{ .root_module = type_resolver_logs_test_mod });
    test_step.dependOn(&b.addRunArtifact(type_resolver_logs_tests).step);

    // unit tests will be added here as they are created.
    // example pattern:
    // const lexer_test_mod = b.createModule(.{
    //     .root_source_file = b.path("src/lexer.test.zig"),
    //     .target = target,
    //     .optimize = optimize,
    // });
    // lexer_test_mod.addImport("ora_root", lib_mod);
    // const lexer_tests = b.addTest(.{ .root_module = lexer_test_mod });
    // test_step.dependOn(&b.addRunArtifact(lexer_tests).step);
}

/// Create a step that runs the installed lexer test suite with --verbose
fn createRunLexerVerboseStep(b: *std.Build) *std.Build.Step {
    const step = b.allocator.create(std.Build.Step) catch @panic("OOM");
    step.* = std.Build.Step.init(.{
        .id = .custom,
        .name = "lexer-suite-verbose",
        .owner = b,
        .makeFn = runLexerVerbose,
    });
    return step;
}

fn runLexerVerbose(step: *std.Build.Step, options: std.Build.Step.MakeOptions) anyerror!void {
    _ = options;
    const b = step.owner;
    const allocator = b.allocator;
    const res = std.process.Child.run(.{
        .allocator = allocator,
        .argv = &[_][]const u8{ "./zig-out/bin/lexer_test_suite", "--verbose" },
        .cwd = ".",
    }) catch |err| {
        std.log.err("Failed to run lexer_test_suite: {}", .{err});
        return err;
    };
    switch (res.term) {
        .Exited => |code| {
            if (code != 0) {
                std.log.err("lexer_test_suite failed with exit code {}", .{code});
                std.log.err("stderr: {s}", .{res.stderr});
                return error.LexerSuiteFailed;
            }
        },
        .Signal => |sig| {
            std.log.err("lexer_test_suite terminated by signal {}", .{sig});
            std.log.err("stderr: {s}", .{res.stderr});
            return error.LexerSuiteFailed;
        },
        else => {},
    }
}

/// Build MLIR from vendored llvm-project and install into vendor/mlir
fn buildMlirLibraries(b: *std.Build, target: std.Build.ResolvedTarget, optimize: std.builtin.OptimizeMode) *std.Build.Step {
    _ = target;
    _ = optimize;

    const step = b.allocator.create(std.Build.Step) catch @panic("OOM");
    step.* = std.Build.Step.init(.{
        .id = .custom,
        .name = "cmake-build-mlir",
        .owner = b,
        .makeFn = buildMlirLibrariesImpl,
    });
    return step;
}

/// Implementation of CMake build for MLIR libraries
fn buildMlirLibrariesImpl(step: *std.Build.Step, options: std.Build.Step.MakeOptions) anyerror!void {
    _ = options;

    const b = step.owner;
    const allocator = b.allocator;

    // ensure submodule exists
    const cwd = std.fs.cwd();
    _ = cwd.openDir("vendor/llvm-project", .{ .iterate = false }) catch {
        std.log.err("Missing submodule: vendor/llvm-project. Add it and pin a commit.", .{});
        std.log.err("Example: git submodule add https://github.com/llvm/llvm-project.git vendor/llvm-project", .{});
        return error.SubmoduleMissing;
    };

    // create build and install directories
    const build_dir = "vendor/llvm-project/build-mlir";
    cwd.makeDir(build_dir) catch |err| switch (err) {
        error.PathAlreadyExists => {},
        else => return err,
    };

    // clear CMake cache if it exists to avoid stale SDK paths after system updates
    const cache_file = try std.fmt.allocPrint(allocator, "{s}/CMakeCache.txt", .{build_dir});
    defer allocator.free(cache_file);
    if (cwd.access(cache_file, .{}) catch null) |_| {
        std.log.info("Clearing stale MLIR CMake cache after macOS/Xcode update", .{});
        cwd.deleteFile(cache_file) catch |err| {
            std.log.warn("Could not delete MLIR CMakeCache.txt: {}", .{err});
        };
    }

    // also clear CMakeFiles directory which may contain cached package configs
    const cmake_files_dir = try std.fmt.allocPrint(allocator, "{s}/CMakeFiles", .{build_dir});
    defer allocator.free(cmake_files_dir);
    if (cwd.access(cmake_files_dir, .{}) catch null) |_| {
        std.log.info("Clearing MLIR CMakeFiles directory to remove stale package configs", .{});
        cwd.deleteTree(cmake_files_dir) catch |err| {
            std.log.warn("Could not delete MLIR CMakeFiles directory: {}", .{err});
        };
    }

    const install_prefix = "vendor/mlir";
    cwd.makeDir(install_prefix) catch |err| switch (err) {
        error.PathAlreadyExists => {},
        else => return err,
    };

    // platform-specific flags
    const builtin = @import("builtin");
    var cmake_args = std.array_list.Managed([]const u8).init(allocator);
    defer cmake_args.deinit();

    // prefer Ninja generator when available for faster, more parallel builds
    var use_ninja: bool = false;
    {
        const probe = std.process.Child.run(.{ .allocator = allocator, .argv = &[_][]const u8{ "ninja", "--version" }, .cwd = "." }) catch null;
        if (probe) |res| {
            switch (res.term) {
                .Exited => |code| {
                    if (code == 0) use_ninja = true;
                },
                else => {},
            }
        }
        if (!use_ninja) {
            const probe_alt = std.process.Child.run(.{ .allocator = allocator, .argv = &[_][]const u8{ "ninja-build", "--version" }, .cwd = "." }) catch null;
            if (probe_alt) |res2| {
                switch (res2.term) {
                    .Exited => |code| {
                        if (code == 0) use_ninja = true;
                    },
                    else => {},
                }
            }
        }
    }

    try cmake_args.append("cmake");
    if (use_ninja) {
        try cmake_args.append("-G");
        try cmake_args.append("Ninja");
    }
    try cmake_args.appendSlice(&[_][]const u8{
        "-S",
        "vendor/llvm-project/llvm",
        "-B",
        build_dir,
        "-DCMAKE_BUILD_TYPE=Release",
        "-DLLVM_ENABLE_PROJECTS=mlir",
        "-DLLVM_TARGETS_TO_BUILD=Native",
        "-DLLVM_INCLUDE_TESTS=OFF",
        "-DMLIR_INCLUDE_TESTS=OFF",
        "-DLLVM_INCLUDE_BENCHMARKS=OFF",
        "-DLLVM_INCLUDE_EXAMPLES=OFF",
        "-DLLVM_INCLUDE_DOCS=OFF",
        "-DMLIR_INCLUDE_DOCS=OFF",
        "-DMLIR_ENABLE_BINDINGS_PYTHON=OFF",
        "-DMLIR_ENABLE_EXECUTION_ENGINE=OFF",
        "-DMLIR_ENABLE_CUDA=OFF",
        "-DMLIR_ENABLE_ROCM=OFF",
        "-DMLIR_ENABLE_SPIRV_CPU_RUNNER=OFF",
        "-DLLVM_ENABLE_ZLIB=OFF",
        "-DLLVM_ENABLE_TERMINFO=OFF",
        "-DLLVM_ENABLE_RTTI=ON",
        "-DLLVM_ENABLE_EH=ON",
        "-DLLVM_ENABLE_PIC=ON",
        "-DLLVM_BUILD_LLVM_DYLIB=OFF",
        "-DLLVM_LINK_LLVM_DYLIB=OFF",
        "-DLLVM_BUILD_TOOLS=ON", // needed for tblgen
        "-DMLIR_BUILD_MLIR_C_DYLIB=ON",
        b.fmt("-DCMAKE_INSTALL_PREFIX={s}", .{install_prefix}),
    });

    if (builtin.os.tag == .linux) {
        try cmake_args.append("-DCMAKE_CXX_FLAGS=-stdlib=libc++ -lc++abi");
        try cmake_args.append("-DCMAKE_EXE_LINKER_FLAGS=-stdlib=libc++ -lc++abi");
        try cmake_args.append("-DCMAKE_SHARED_LINKER_FLAGS=-stdlib=libc++ -lc++abi");
        try cmake_args.append("-DCMAKE_MODULE_LINKER_FLAGS=-stdlib=libc++ -lc++abi");
        try cmake_args.append("-DCMAKE_CXX_COMPILER=clang++");
        try cmake_args.append("-DCMAKE_C_COMPILER=clang");
    } else if (builtin.os.tag == .macos) {
        try cmake_args.append("-DCMAKE_CXX_FLAGS=-stdlib=libc++");

        // fix SDK path issue after macOS/Xcode update
        // use xcrun to get the actual SDK path and set it explicitly
        const sdk_path_result = std.process.Child.run(.{
            .allocator = allocator,
            .argv = &[_][]const u8{ "xcrun", "--show-sdk-path" },
            .cwd = ".",
        }) catch null;
        if (sdk_path_result) |result| {
            if (result.term.Exited == 0) {
                const sdk_path = std.mem.trim(u8, result.stdout, " \n\r\t");
                if (sdk_path.len > 0) {
                    const sysroot_flag = try std.fmt.allocPrint(allocator, "-DCMAKE_OSX_SYSROOT={s}", .{sdk_path});
                    defer allocator.free(sysroot_flag);
                    try cmake_args.append(sysroot_flag);
                    std.log.info("Setting MLIR CMAKE_OSX_SYSROOT={s}", .{sdk_path});
                }
            }
        }

        if (std.process.getEnvVarOwned(allocator, "ORA_CMAKE_OSX_ARCH") catch null) |arch| {
            defer allocator.free(arch);
            const flag = b.fmt("-DCMAKE_OSX_ARCHITECTURES={s}", .{arch});
            try cmake_args.append(flag);
            std.log.info("Using CMAKE_OSX_ARCHITECTURES={s}", .{arch});
        }
    } else if (builtin.os.tag == .windows) {
        try cmake_args.append("-DCMAKE_CXX_FLAGS=/std:c++20");
    }

    var cfg_child = std.process.Child.init(cmake_args.items, allocator);
    cfg_child.cwd = ".";
    cfg_child.stdin_behavior = .Inherit;
    cfg_child.stdout_behavior = .Inherit;
    cfg_child.stderr_behavior = .Inherit;
    const cfg_term = cfg_child.spawnAndWait() catch |err| {
        std.log.err("Failed to configure MLIR CMake: {}", .{err});
        return err;
    };
    switch (cfg_term) {
        .Exited => |code| if (code != 0) {
            std.log.err("MLIR CMake configure failed with exit code: {}", .{code});
            return error.CMakeConfigureFailed;
        },
        else => {
            std.log.err("MLIR CMake configure did not exit cleanly", .{});
            return error.CMakeConfigureFailed;
        },
    }

    // build and install MLIR (with sparse checkout and minimal flags above this is lightweight)
    var build_args = [_][]const u8{ "cmake", "--build", build_dir, "--parallel", "--target", "install" };
    var build_child = std.process.Child.init(&build_args, allocator);
    build_child.cwd = ".";
    build_child.stdin_behavior = .Inherit;
    build_child.stdout_behavior = .Inherit;
    build_child.stderr_behavior = .Inherit;
    const build_term = build_child.spawnAndWait() catch |err| {
        std.log.err("Failed to build MLIR with CMake: {}", .{err});
        return err;
    };
    switch (build_term) {
        .Exited => |code| if (code != 0) {
            std.log.err("MLIR CMake build failed with exit code: {}", .{code});
            return error.CMakeBuildFailed;
        },
        else => {
            std.log.err("MLIR CMake build did not exit cleanly", .{});
            return error.CMakeBuildFailed;
        },
    }

    std.log.info("Successfully built MLIR libraries", .{});
}

/// Build Ora dialect library using CMake
fn buildOraDialectLibrary(b: *std.Build, mlir_step: *std.Build.Step, sir_dialect_step: *std.Build.Step, target: std.Build.ResolvedTarget, optimize: std.builtin.OptimizeMode) *std.Build.Step {
    _ = target;
    _ = optimize;

    const step = b.allocator.create(std.Build.Step) catch @panic("OOM");
    step.* = std.Build.Step.init(.{
        .id = .custom,
        .name = "cmake-build-ora-dialect",
        .owner = b,
        .makeFn = buildOraDialectLibraryImpl,
    });
    step.dependOn(mlir_step);
    step.dependOn(sir_dialect_step); // Ora dialect needs SIR headers
    return step;
}

/// Implementation of CMake build for Ora dialect library
fn buildOraDialectLibraryImpl(step: *std.Build.Step, options: std.Build.Step.MakeOptions) anyerror!void {
    _ = options;

    const b = step.owner;
    const allocator = b.allocator;
    const cwd = std.fs.cwd();

    // create build directory (clean if it exists to avoid CMake cache conflicts)
    const build_dir = "vendor/ora-dialect-build";
    // remove existing build directory to avoid CMake cache conflicts
    cwd.deleteTree(build_dir) catch {};
    cwd.makeDir(build_dir) catch |err| switch (err) {
        error.PathAlreadyExists => {},
        else => return err,
    };

    const install_prefix = "vendor/mlir";
    const mlir_dir = b.fmt("{s}/lib/cmake/mlir", .{install_prefix});

    // platform-specific flags
    const builtin = @import("builtin");
    var cmake_args = std.array_list.Managed([]const u8).init(allocator);
    defer cmake_args.deinit();

    // prefer Ninja generator when available
    var use_ninja: bool = false;
    {
        const probe = std.process.Child.run(.{ .allocator = allocator, .argv = &[_][]const u8{ "ninja", "--version" }, .cwd = "." }) catch null;
        if (probe) |res| {
            switch (res.term) {
                .Exited => |code| {
                    if (code == 0) use_ninja = true;
                },
                else => {},
            }
        }
    }

    try cmake_args.append("cmake");
    if (use_ninja) {
        try cmake_args.append("-G");
        try cmake_args.append("Ninja");
    }
    try cmake_args.appendSlice(&[_][]const u8{
        "-S",
        "src/mlir/ora",
        "-B",
        build_dir,
        "-DCMAKE_BUILD_TYPE=Release",
        b.fmt("-DMLIR_DIR={s}", .{mlir_dir}),
        b.fmt("-DCMAKE_INSTALL_PREFIX={s}", .{install_prefix}),
    });

    if (builtin.os.tag == .linux) {
        try cmake_args.append("-DCMAKE_CXX_FLAGS=-stdlib=libc++ -lc++abi");
        try cmake_args.append("-DCMAKE_CXX_COMPILER=clang++");
        try cmake_args.append("-DCMAKE_C_COMPILER=clang");
    } else if (builtin.os.tag == .macos) {
        try cmake_args.append("-DCMAKE_CXX_FLAGS=-stdlib=libc++");
    }

    var cfg_child = std.process.Child.init(cmake_args.items, allocator);
    cfg_child.cwd = ".";
    cfg_child.stdin_behavior = .Inherit;
    cfg_child.stdout_behavior = .Inherit;
    cfg_child.stderr_behavior = .Inherit;
    const cfg_term = cfg_child.spawnAndWait() catch |err| {
        std.log.err("Failed to configure Ora dialect CMake: {}", .{err});
        return err;
    };
    switch (cfg_term) {
        .Exited => |code| if (code != 0) {
            std.log.err("Ora dialect CMake configure failed with exit code: {}", .{code});
            return error.CMakeConfigureFailed;
        },
        else => {
            std.log.err("Ora dialect CMake configure did not exit cleanly", .{});
            return error.CMakeConfigureFailed;
        },
    }

    // build and install
    var build_args = [_][]const u8{ "cmake", "--build", build_dir, "--parallel", "--target", "install" };
    var build_child = std.process.Child.init(&build_args, allocator);
    build_child.cwd = ".";
    build_child.stdin_behavior = .Inherit;
    build_child.stdout_behavior = .Inherit;
    build_child.stderr_behavior = .Inherit;
    const build_term = build_child.spawnAndWait() catch |err| {
        std.log.err("Failed to build Ora dialect with CMake: {}", .{err});
        return err;
    };
    switch (build_term) {
        .Exited => |code| if (code != 0) {
            std.log.err("Ora dialect CMake build failed with exit code: {}", .{code});
            return error.CMakeBuildFailed;
        },
        else => {
            std.log.err("Ora dialect CMake build did not exit cleanly", .{});
            return error.CMakeBuildFailed;
        },
    }

    std.log.info("Successfully built Ora dialect library", .{});
}

/// Build SIR dialect library using CMake
fn buildSIRDialectLibrary(b: *std.Build, mlir_step: *std.Build.Step, target: std.Build.ResolvedTarget, optimize: std.builtin.OptimizeMode) *std.Build.Step {
    _ = target;
    _ = optimize;

    const step = b.allocator.create(std.Build.Step) catch @panic("OOM");
    step.* = std.Build.Step.init(.{
        .id = .custom,
        .name = "cmake-build-sir-dialect",
        .owner = b,
        .makeFn = buildSIRDialectLibraryImpl,
    });
    step.dependOn(mlir_step);
    return step;
}

/// Implementation of CMake build for SIR dialect library
fn buildSIRDialectLibraryImpl(step: *std.Build.Step, options: std.Build.Step.MakeOptions) anyerror!void {
    _ = options;

    const b = step.owner;
    const allocator = b.allocator;
    const cwd = std.fs.cwd();

    // create build directory (clean if it exists to avoid CMake cache conflicts)
    const build_dir = "vendor/sir-dialect-build";
    // remove existing build directory to avoid CMake cache conflicts
    cwd.deleteTree(build_dir) catch {};
    cwd.makeDir(build_dir) catch |err| switch (err) {
        error.PathAlreadyExists => {},
        else => return err,
    };

    const install_prefix = "vendor/mlir";
    const mlir_dir = b.fmt("{s}/lib/cmake/mlir", .{install_prefix});

    // platform-specific flags
    const builtin = @import("builtin");
    var cmake_args = std.array_list.Managed([]const u8).init(allocator);
    defer cmake_args.deinit();

    // prefer Ninja generator when available
    var use_ninja: bool = false;
    {
        const probe = std.process.Child.run(.{ .allocator = allocator, .argv = &[_][]const u8{ "ninja", "--version" }, .cwd = "." }) catch null;
        if (probe) |res| {
            switch (res.term) {
                .Exited => |code| {
                    if (code == 0) use_ninja = true;
                },
                else => {},
            }
        }
    }

    try cmake_args.append("cmake");
    if (use_ninja) {
        try cmake_args.append("-G");
        try cmake_args.append("Ninja");
    }
    try cmake_args.appendSlice(&[_][]const u8{
        "-S",
        "src/mlir/IR",
        "-B",
        build_dir,
        "-DCMAKE_BUILD_TYPE=Release",
        b.fmt("-DMLIR_DIR={s}", .{mlir_dir}),
        b.fmt("-DCMAKE_INSTALL_PREFIX={s}", .{install_prefix}),
    });

    if (builtin.os.tag == .linux) {
        try cmake_args.append("-DCMAKE_CXX_FLAGS=-stdlib=libc++ -lc++abi");
        try cmake_args.append("-DCMAKE_CXX_COMPILER=clang++");
        try cmake_args.append("-DCMAKE_C_COMPILER=clang");
    } else if (builtin.os.tag == .macos) {
        try cmake_args.append("-DCMAKE_CXX_FLAGS=-stdlib=libc++");
    }

    var cfg_child = std.process.Child.init(cmake_args.items, allocator);
    cfg_child.cwd = ".";
    cfg_child.stdin_behavior = .Inherit;
    cfg_child.stdout_behavior = .Inherit;
    cfg_child.stderr_behavior = .Inherit;
    const cfg_term = cfg_child.spawnAndWait() catch |err| {
        std.log.err("Failed to configure SIR dialect CMake: {}", .{err});
        return err;
    };
    switch (cfg_term) {
        .Exited => |code| if (code != 0) {
            std.log.err("SIR dialect CMake configure failed with exit code: {}", .{code});
            return error.CMakeConfigureFailed;
        },
        else => {
            std.log.err("SIR dialect CMake configure did not exit cleanly", .{});
            return error.CMakeConfigureFailed;
        },
    }

    // build and install
    var build_args = [_][]const u8{ "cmake", "--build", build_dir, "--parallel", "--target", "install" };
    var build_child = std.process.Child.init(&build_args, allocator);
    build_child.cwd = ".";
    build_child.stdin_behavior = .Inherit;
    build_child.stdout_behavior = .Inherit;
    build_child.stderr_behavior = .Inherit;
    const build_term = build_child.spawnAndWait() catch |err| {
        std.log.err("Failed to build SIR dialect with CMake: {}", .{err});
        return err;
    };
    switch (build_term) {
        .Exited => |code| if (code != 0) {
            std.log.err("SIR dialect CMake build failed with exit code: {}", .{code});
            return error.CMakeBuildFailed;
        },
        else => {
            std.log.err("SIR dialect CMake build did not exit cleanly", .{});
            return error.CMakeBuildFailed;
        },
    }

    std.log.info("Successfully built SIR dialect library", .{});
}

/// Link MLIR to the given executable using the installed prefix
fn linkMlirLibraries(
    b: *std.Build,
    exe: *std.Build.Step.Compile,
    mlir_step: ?*std.Build.Step,
    ora_dialect_step: ?*std.Build.Step,
    sir_dialect_step: ?*std.Build.Step,
    target: std.Build.ResolvedTarget,
) void {
    // depend on MLIR build and dialect builds when requested
    if (mlir_step) |step| exe.step.dependOn(step);
    if (ora_dialect_step) |step| exe.step.dependOn(step);
    if (sir_dialect_step) |step| exe.step.dependOn(step);

    const include_path = b.path("vendor/mlir/include");
    const lib_path = b.path("vendor/mlir/lib");
    const ora_dialect_include_path = b.path("src/mlir/ora/include");
    const sir_dialect_include_path = b.path("src/mlir/IR/include");

    exe.addIncludePath(include_path);
    exe.addIncludePath(ora_dialect_include_path);
    exe.addIncludePath(sir_dialect_include_path);
    exe.addLibraryPath(lib_path);

    exe.linkSystemLibrary("MLIR-C");
    exe.linkSystemLibrary("MLIROraDialectC");
    exe.linkSystemLibrary("MLIRSIRDialect");

    switch (target.result.os.tag) {
        .linux => {
            exe.linkLibCpp();
            exe.linkSystemLibrary("c++abi");
        },
        .macos => {
            exe.linkLibCpp();
        },
        else => {
            exe.linkLibCpp();
        },
    }
}

/// Create example testing step that runs the compiler on all .ora files
fn createExampleTestStep(b: *std.Build, exe: *std.Build.Step.Compile) *std.Build.Step {
    const test_step = b.allocator.create(std.Build.Step) catch @panic("OOM");
    test_step.* = std.Build.Step.init(.{
        .id = .custom,
        .name = "test-examples",
        .owner = b,
        .makeFn = runExampleTests,
    });

    // depend on the main executable being built
    test_step.dependOn(&exe.step);

    return test_step;
}

/// Implementation of example testing
fn runExampleTests(step: *std.Build.Step, options: std.Build.Step.MakeOptions) anyerror!void {
    _ = options;

    const b = step.owner;
    const allocator = b.allocator;

    std.log.info("Testing all .ora example files...", .{});

    // get examples directory
    var examples_dir = std.fs.cwd().openDir("ora-example", .{ .iterate = true }) catch |err| {
        std.log.err("Failed to open examples directory: {}", .{err});
        return err;
    };
    defer examples_dir.close();

    // iterate through all .ora files
    var walker = examples_dir.walk(allocator) catch |err| {
        std.log.err("Failed to walk examples directory: {}", .{err});
        return err;
    };
    defer walker.deinit();

    var tested_count: u32 = 0;
    var failed_count: u32 = 0;

    while (walker.next() catch |err| {
        std.log.err("Failed to get next file: {}", .{err});
        return err;
    }) |entry| {
        // skip non-.ora files
        if (entry.kind != .file) continue;
        if (!std.mem.endsWith(u8, entry.basename, ".ora")) continue;

        const file_path = b.fmt("ora-example/{s}", .{entry.path});
        std.log.info("Testing: {s}", .{file_path});

        // test each compilation phase
        const phases = [_][]const u8{ "lex", "parse", "analyze", "compile" };

        for (phases) |phase| {
            const result = std.process.Child.run(.{
                .allocator = allocator,
                .argv = &[_][]const u8{
                    "./zig-out/bin/ora",
                    phase,
                    file_path,
                },
                .cwd = ".",
            }) catch |err| {
                std.log.err("Failed to run test for {s} with phase {s}: {}", .{ file_path, phase, err });
                failed_count += 1;
                continue;
            };

            switch (result.term) {
                .Exited => |code| {
                    if (code != 0) {
                        std.log.err("FAILED: {s} (phase: {s}) with exit code {}", .{ file_path, phase, code });
                        std.log.err("Error output: {s}", .{result.stderr});
                        failed_count += 1;
                        break; // Don't test further phases if one fails
                    }
                },
                .Signal => |sig| {
                    std.log.err("FAILED: {s} (phase: {s}) terminated by signal {}", .{ file_path, phase, sig });
                    std.log.err("Error output: {s}", .{result.stderr});
                    failed_count += 1;
                    break; // Don't test further phases if one fails
                },
                else => {
                    std.log.err("FAILED: {s} (phase: {s}) with unexpected termination", .{ file_path, phase });
                    std.log.err("Error output: {s}", .{result.stderr});
                    failed_count += 1;
                    break; // Don't test further phases if one fails
                },
            }
        }

        tested_count += 1;
    }

    std.log.info("Example testing complete: {}/{} files passed", .{ tested_count - failed_count, tested_count });

    if (failed_count > 0) {
        std.log.err("{} example files failed compilation", .{failed_count});
        return error.ExampleTestsFailed;
    }
}

/// Build Z3 from vendored z3 repository and install into vendor/z3-install
fn buildZ3Libraries(b: *std.Build, target: std.Build.ResolvedTarget, optimize: std.builtin.OptimizeMode) *std.Build.Step {
    _ = target;
    _ = optimize;

    const step = b.allocator.create(std.Build.Step) catch @panic("OOM");
    step.* = std.Build.Step.init(.{
        .id = .custom,
        .name = "cmake-build-z3",
        .owner = b,
        .makeFn = buildZ3LibrariesImpl,
    });
    return step;
}

fn pathExists(path: []const u8) bool {
    std.fs.cwd().access(path, .{}) catch return false;
    return true;
}

fn anyPathExists(paths: []const []const u8) bool {
    for (paths) |path| {
        if (pathExists(path)) return true;
    }
    return false;
}

fn hostHasSystemZ3() bool {
    const builtin = @import("builtin");

    const header_paths = switch (builtin.os.tag) {
        .macos => [_][]const u8{
            "/opt/homebrew/include/z3.h",
            "/usr/local/include/z3.h",
            "/usr/include/z3.h",
        },
        .linux => [_][]const u8{
            "/usr/include/z3.h",
            "/usr/local/include/z3.h",
        },
        .windows => [_][]const u8{
            "C:/Program Files/Z3/include/z3.h",
            "C:/tools/z3/include/z3.h",
        },
        else => [_][]const u8{
            "/usr/include/z3.h",
        },
    };

    const lib_paths = switch (builtin.os.tag) {
        .macos => [_][]const u8{
            "/opt/homebrew/lib/libz3.dylib",
            "/opt/homebrew/lib/libz3.a",
            "/usr/local/lib/libz3.dylib",
            "/usr/local/lib/libz3.a",
        },
        .linux => [_][]const u8{
            "/usr/lib/libz3.so",
            "/usr/lib/libz3.a",
            "/usr/lib64/libz3.so",
            "/usr/local/lib/libz3.so",
            "/usr/local/lib/libz3.a",
            "/usr/lib/x86_64-linux-gnu/libz3.so",
            "/usr/lib/x86_64-linux-gnu/libz3.a",
            "/usr/lib/aarch64-linux-gnu/libz3.so",
            "/usr/lib/aarch64-linux-gnu/libz3.a",
        },
        .windows => [_][]const u8{
            "C:/Program Files/Z3/bin/libz3.dll",
            "C:/Program Files/Z3/lib/libz3.lib",
            "C:/tools/z3/bin/libz3.dll",
            "C:/tools/z3/lib/libz3.lib",
        },
        else => [_][]const u8{
            "/usr/lib/libz3.so",
            "/usr/lib/libz3.a",
        },
    };

    return anyPathExists(&header_paths) and anyPathExists(&lib_paths);
}

/// Implementation of CMake build for Z3 libraries
fn buildZ3LibrariesImpl(step: *std.Build.Step, options: std.Build.Step.MakeOptions) anyerror!void {
    _ = options;

    const b = step.owner;
    const allocator = b.allocator;

    if (hostHasSystemZ3()) {
        std.log.info("System Z3 headers and library detected; skipping vendored Z3 build", .{});
        return;
    }

    // system Z3 headers/libraries not found, try to build from vendor
    std.log.info("System Z3 not fully available, checking vendor/z3...", .{});

    const cwd = std.fs.cwd();
    _ = cwd.openDir("vendor/z3", .{ .iterate = false }) catch {
        std.log.warn("Z3 not found! Please install Z3 or add it as a submodule:", .{});
        std.log.warn("  System install (recommended): brew install z3  (macOS)", .{});
        std.log.warn("  System install (recommended): sudo apt install z3  (Linux)", .{});
        std.log.warn("  Or add as submodule: git submodule add https://github.com/Z3Prover/z3.git vendor/z3", .{});
        std.log.warn("Z3 is optional for now - continuing without formal verification support", .{});
        return;
    };

    // create build and install directories
    const build_dir = "vendor/z3/build-release";
    cwd.makeDir(build_dir) catch |err| switch (err) {
        error.PathAlreadyExists => {},
        else => return err,
    };

    // clear CMake cache if it exists to avoid stale SDK paths after system updates
    const cache_file = try std.fmt.allocPrint(allocator, "{s}/CMakeCache.txt", .{build_dir});
    defer allocator.free(cache_file);
    if (cwd.access(cache_file, .{}) catch null) |_| {
        std.log.info("Clearing stale Z3 CMake cache after macOS/Xcode update", .{});
        cwd.deleteFile(cache_file) catch |err| {
            std.log.warn("Could not delete Z3 CMakeCache.txt: {}", .{err});
        };
    }

    // also clear CMakeFiles directory which may contain cached package configs
    const cmake_files_dir = try std.fmt.allocPrint(allocator, "{s}/CMakeFiles", .{build_dir});
    defer allocator.free(cmake_files_dir);
    if (cwd.access(cmake_files_dir, .{}) catch null) |_| {
        std.log.info("Clearing Z3 CMakeFiles directory to remove stale package configs", .{});
        cwd.deleteTree(cmake_files_dir) catch |err| {
            std.log.warn("Could not delete Z3 CMakeFiles directory: {}", .{err});
        };
    }

    const install_prefix = "vendor/z3-install";
    cwd.makeDir(install_prefix) catch |err| switch (err) {
        error.PathAlreadyExists => {},
        else => return err,
    };

    // platform-specific flags
    const builtin = @import("builtin");
    var cmake_args = std.array_list.Managed([]const u8).init(allocator);
    defer cmake_args.deinit();

    // prefer Ninja generator when available
    var use_ninja: bool = false;
    {
        const probe = std.process.Child.run(.{ .allocator = allocator, .argv = &[_][]const u8{ "ninja", "--version" }, .cwd = "." }) catch null;
        if (probe) |res| {
            switch (res.term) {
                .Exited => |code| {
                    if (code == 0) use_ninja = true;
                },
                else => {},
            }
        }
    }

    try cmake_args.append("cmake");
    if (use_ninja) {
        try cmake_args.append("-G");
        try cmake_args.append("Ninja");
    }
    try cmake_args.appendSlice(&[_][]const u8{
        "-S",
        "vendor/z3",
        "-B",
        build_dir,
        "-DCMAKE_BUILD_TYPE=Release",
        "-DZ3_BUILD_LIBZ3_SHARED=OFF", // Static library
        "-DZ3_BUILD_PYTHON_BINDINGS=OFF",
        "-DZ3_BUILD_DOTNET_BINDINGS=OFF",
        "-DZ3_BUILD_JAVA_BINDINGS=OFF",
        "-DZ3_BUILD_EXECUTABLE=OFF", // Don't need the z3 binary
        "-DZ3_ENABLE_EXAMPLE_TARGETS=OFF",
        "-DZ3_BUILD_TEST_EXECUTABLES=OFF",
        b.fmt("-DCMAKE_INSTALL_PREFIX={s}", .{install_prefix}),
    });

    if (builtin.os.tag == .linux) {
        try cmake_args.append("-DCMAKE_CXX_FLAGS=-stdlib=libc++ -lc++abi");
        try cmake_args.append("-DCMAKE_CXX_COMPILER=clang++");
        try cmake_args.append("-DCMAKE_C_COMPILER=clang");
    } else if (builtin.os.tag == .macos) {
        try cmake_args.append("-DCMAKE_CXX_FLAGS=-stdlib=libc++");

        // fix SDK path issue after macOS/Xcode update
        // use xcrun to get the actual SDK path and set it explicitly
        const sdk_path_result = std.process.Child.run(.{
            .allocator = allocator,
            .argv = &[_][]const u8{ "xcrun", "--show-sdk-path" },
            .cwd = ".",
        }) catch null;
        if (sdk_path_result) |result| {
            if (result.term.Exited == 0) {
                const sdk_path = std.mem.trim(u8, result.stdout, " \n\r\t");
                if (sdk_path.len > 0) {
                    const sysroot_flag = try std.fmt.allocPrint(allocator, "-DCMAKE_OSX_SYSROOT={s}", .{sdk_path});
                    defer allocator.free(sysroot_flag);
                    try cmake_args.append(sysroot_flag);
                    std.log.info("Setting Z3 CMAKE_OSX_SYSROOT={s}", .{sdk_path});
                }
            }
        }
    }

    var cfg_child = std.process.Child.init(cmake_args.items, allocator);
    cfg_child.cwd = ".";
    cfg_child.stdin_behavior = .Inherit;
    cfg_child.stdout_behavior = .Inherit;
    cfg_child.stderr_behavior = .Inherit;
    const cfg_term = cfg_child.spawnAndWait() catch |err| {
        std.log.err("Failed to configure Z3 CMake: {}", .{err});
        return err;
    };
    switch (cfg_term) {
        .Exited => |code| if (code != 0) {
            std.log.err("Z3 CMake configure failed with exit code: {}", .{code});
            return error.CMakeConfigureFailed;
        },
        else => {
            std.log.err("Z3 CMake configure did not exit cleanly", .{});
            return error.CMakeConfigureFailed;
        },
    }

    // build and install Z3
    var build_args = [_][]const u8{ "cmake", "--build", build_dir, "--parallel", "--target", "install" };
    var build_child = std.process.Child.init(&build_args, allocator);
    build_child.cwd = ".";
    build_child.stdin_behavior = .Inherit;
    build_child.stdout_behavior = .Inherit;
    build_child.stderr_behavior = .Inherit;
    const build_term = build_child.spawnAndWait() catch |err| {
        std.log.err("Failed to build Z3 with CMake: {}", .{err});
        return err;
    };
    switch (build_term) {
        .Exited => |code| if (code != 0) {
            std.log.err("Z3 CMake build failed with exit code: {}", .{code});
            return error.CMakeBuildFailed;
        },
        else => {
            std.log.err("Z3 CMake build did not exit cleanly", .{});
            return error.CMakeBuildFailed;
        },
    }

    std.log.info("Successfully built Z3 libraries", .{});
}

/// Link Z3 to the given executable
fn linkZ3Libraries(b: *std.Build, exe: *std.Build.Step.Compile, z3_step: *std.Build.Step, target: std.Build.ResolvedTarget) void {
    // depend on Z3 build
    exe.step.dependOn(z3_step);
    const using_system_z3 = hostHasSystemZ3();

    // Ensure vendored search paths exist to avoid hard failures when Zig checks
    // linker search directories before the z3 custom step populates them.
    std.fs.cwd().makePath("vendor/z3-install/include") catch {};
    std.fs.cwd().makePath("vendor/z3-install/lib") catch {};

    // Always expose vendored include/lib paths. The custom z3_step may populate
    // these directories later in the build graph.
    exe.addIncludePath(b.path("vendor/z3-install/include"));
    exe.addLibraryPath(b.path("vendor/z3-install/lib"));

    if (using_system_z3) {
        std.log.info("Linking against system Z3", .{});
        // add system Z3 paths based on platform
        switch (target.result.os.tag) {
            .macos => {
                // homebrew paths
                if (target.result.cpu.arch == .aarch64) {
                    exe.addIncludePath(.{ .cwd_relative = "/opt/homebrew/include" });
                    exe.addLibraryPath(.{ .cwd_relative = "/opt/homebrew/lib" });
                } else {
                    exe.addIncludePath(.{ .cwd_relative = "/usr/local/include" });
                    exe.addLibraryPath(.{ .cwd_relative = "/usr/local/lib" });
                }
            },
            .linux => {
                exe.addIncludePath(.{ .cwd_relative = "/usr/include" });
                exe.addLibraryPath(.{ .cwd_relative = "/usr/lib" });
                exe.addLibraryPath(.{ .cwd_relative = "/usr/local/lib" });
                if (target.result.cpu.arch == .x86_64) {
                    exe.addLibraryPath(.{ .cwd_relative = "/usr/lib/x86_64-linux-gnu" });
                }
            },
            else => {
                exe.addIncludePath(.{ .cwd_relative = "/usr/include" });
                exe.addLibraryPath(.{ .cwd_relative = "/usr/lib" });
            },
        }
    } else {
        std.log.info("Linking against vendored Z3 (or vendor build output)", .{});
    }

    // link Z3 library
    exe.linkSystemLibrary("z3");

    // link C++ standard library (Z3 is C++)
    switch (target.result.os.tag) {
        .linux => {
            exe.linkLibCpp();
            exe.linkSystemLibrary("c++abi");
        },
        .macos => {
            exe.linkLibCpp();
        },
        else => {
            exe.linkLibCpp();
        },
    }
}

/// Add platform-specific Boost include and library paths
/// For cross-compilation, we use the host system paths where dependencies are actually installed
fn addBoostPaths(b: *std.Build, compile_step: *std.Build.Step.Compile, target: std.Build.ResolvedTarget) void {
    const target_info = target.result;
    const host_info = @import("builtin").target;

    // determine if we're cross-compiling
    const is_cross_compiling = target_info.os.tag != host_info.os.tag or
        target_info.cpu.arch != host_info.cpu.arch;

    // for cross-compilation, use host system paths where dependencies are installed
    // for native compilation, use target system paths
    const os_to_use = if (is_cross_compiling) host_info.os.tag else target_info.os.tag;
    const arch_to_use = if (is_cross_compiling) host_info.cpu.arch else target_info.cpu.arch;

    if (is_cross_compiling) {
        std.log.info("Cross-compiling from {s}-{s} to {s}-{s} - using host paths for Boost", .{ @tagName(host_info.os.tag), @tagName(host_info.cpu.arch), @tagName(target_info.os.tag), @tagName(target_info.cpu.arch) });
    }

    switch (os_to_use) {
        .macos => {
            // check if Apple Silicon or Intel Mac
            if (arch_to_use == .aarch64) {
                // apple Silicon - Homebrew installs to /opt/homebrew
                std.log.info("Adding Boost paths for Apple Silicon Mac", .{});
                compile_step.addSystemIncludePath(.{ .cwd_relative = "/opt/homebrew/include" });
                compile_step.addLibraryPath(.{ .cwd_relative = "/opt/homebrew/lib" });
            } else {
                // intel Mac - Homebrew installs to /usr/local
                std.log.info("Adding Boost paths for Intel Mac", .{});
                compile_step.addSystemIncludePath(.{ .cwd_relative = "/usr/local/include" });
                compile_step.addLibraryPath(.{ .cwd_relative = "/usr/local/lib" });
            }
        },
        .linux => {
            // linux - check common package manager locations
            std.log.info("Adding Boost paths for Linux", .{});
            compile_step.addSystemIncludePath(.{ .cwd_relative = "/usr/include" });
            compile_step.addSystemIncludePath(.{ .cwd_relative = "/usr/local/include" });
            compile_step.addLibraryPath(.{ .cwd_relative = "/usr/lib" });
            compile_step.addLibraryPath(.{ .cwd_relative = "/usr/local/lib" });

            // also check for x86_64-linux-gnu paths (common on Ubuntu)
            if (arch_to_use == .x86_64) {
                compile_step.addLibraryPath(.{ .cwd_relative = "/usr/lib/x86_64-linux-gnu" });
            }
        },
        .windows => {
            // windows - typically vcpkg or manual installation
            std.log.info("Adding Boost paths for Windows", .{});
            // check for vcpkg installation
            if (std.process.hasEnvVarConstant("VCPKG_ROOT")) {
                compile_step.addSystemIncludePath(.{ .cwd_relative = "C:/vcpkg/installed/x64-windows/include" });
                compile_step.addLibraryPath(.{ .cwd_relative = "C:/vcpkg/installed/x64-windows/lib" });
            } else {
                // default paths for manual installation
                compile_step.addSystemIncludePath(.{ .cwd_relative = "C:/boost/include" });
                compile_step.addLibraryPath(.{ .cwd_relative = "C:/boost/lib" });
            }
        },
        else => {
            // fallback for other platforms
            std.log.warn("Unknown platform for Boost paths - using default system paths", .{});
            compile_step.addSystemIncludePath(.{ .cwd_relative = "/usr/include" });
            compile_step.addSystemIncludePath(.{ .cwd_relative = "/usr/local/include" });
            compile_step.addLibraryPath(.{ .cwd_relative = "/usr/lib" });
            compile_step.addLibraryPath(.{ .cwd_relative = "/usr/local/lib" });
        },
    }

    // try to check if Boost is available via environment variable
    const boost_root = std.process.getEnvVarOwned(b.allocator, "BOOST_ROOT") catch null;
    if (boost_root) |root| {
        defer b.allocator.free(root);
        std.log.info("Using BOOST_ROOT environment variable: {s}", .{root});
        const include_path = std.fmt.allocPrint(b.allocator, "{s}/include", .{root}) catch @panic("OOM");
        const lib_path = std.fmt.allocPrint(b.allocator, "{s}/lib", .{root}) catch @panic("OOM");
        defer b.allocator.free(include_path);
        defer b.allocator.free(lib_path);
        compile_step.addSystemIncludePath(.{ .cwd_relative = include_path });
        compile_step.addLibraryPath(.{ .cwd_relative = lib_path });
    }
}
