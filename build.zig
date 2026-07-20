// ============================================================================
// Build Script
// ============================================================================
//
// build configuration for the Ora compiler, tests, and toolchain integration.
//
// ============================================================================

const std = @import("std");

const LLVM_REPO_URL = "https://github.com/llvm/llvm-project.git";
const LLVM_COMMIT = "ee8c14be14deabace692ab51f5d5d432b0a83d58";

const NativeSanitizer = enum {
    none,
    address,
    undefined,
    address_undefined,

    fn enabled(self: NativeSanitizer) bool {
        return self != .none;
    }

    fn suffix(self: NativeSanitizer) []const u8 {
        return switch (self) {
            .none => "release",
            .address => "asan",
            .undefined => "ubsan",
            .address_undefined => "asan-ubsan",
        };
    }

    fn cmakeFlags(self: NativeSanitizer) []const u8 {
        return switch (self) {
            .none => "",
            .address => "-fsanitize=address -fno-omit-frame-pointer -g",
            .undefined => "-fsanitize=undefined -fno-omit-frame-pointer -g",
            .address_undefined => "-fsanitize=address,undefined -fno-omit-frame-pointer -g",
        };
    }
};

const NativeCMakeStep = struct {
    step: std.Build.Step,
    sanitizer: NativeSanitizer,
};

fn parseNativeSanitizer(value: []const u8) NativeSanitizer {
    if (std.mem.eql(u8, value, "none")) return .none;
    if (std.mem.eql(u8, value, "address") or std.mem.eql(u8, value, "asan")) return .address;
    if (std.mem.eql(u8, value, "undefined") or std.mem.eql(u8, value, "ubsan")) return .undefined;
    if (std.mem.eql(u8, value, "address,undefined") or
        std.mem.eql(u8, value, "undefined,address") or
        std.mem.eql(u8, value, "asan,ubsan") or
        std.mem.eql(u8, value, "asan-ubsan"))
    {
        return .address_undefined;
    }
    std.debug.panic("invalid -Dnative-sanitize value '{s}' (expected none, address, undefined, or address,undefined)", .{value});
}

fn nativeMlirInstallPrefix(b: *std.Build, sanitizer: NativeSanitizer) []const u8 {
    return if (sanitizer.enabled())
        b.fmt("vendor/mlir-{s}", .{sanitizer.suffix()})
    else
        "vendor/mlir";
}

fn nativeDialectBuildDir(b: *std.Build, base: []const u8, sanitizer: NativeSanitizer) []const u8 {
    return if (sanitizer.enabled())
        b.fmt("{s}-{s}", .{ base, sanitizer.suffix() })
    else
        base;
}

fn joinCmakeFlags(b: *std.Build, base: []const u8, extra: []const u8) []const u8 {
    if (base.len == 0) return extra;
    if (extra.len == 0) return base;
    return b.fmt("{s} {s}", .{ base, extra });
}

fn appendCmakeDefine(cmake_args: *std.array_list.Managed([]const u8), b: *std.Build, name: []const u8, value: []const u8) !void {
    try cmake_args.append(b.fmt("-D{s}={s}", .{ name, value }));
}

fn appendNativeCmakeToolchainFlags(cmake_args: *std.array_list.Managed([]const u8), b: *std.Build, sanitizer: NativeSanitizer) !void {
    const builtin = @import("builtin");
    const sanitizer_flags = sanitizer.cmakeFlags();

    if (sanitizer.enabled() and builtin.os.tag == .windows) {
        std.debug.panic("-Dnative-sanitize is currently supported on macOS/Linux CMake builds only", .{});
    }

    if (builtin.os.tag == .linux) {
        try appendCmakeDefine(cmake_args, b, "CMAKE_CXX_FLAGS", joinCmakeFlags(b, "-stdlib=libc++", sanitizer_flags));
        try appendCmakeDefine(cmake_args, b, "CMAKE_EXE_LINKER_FLAGS", joinCmakeFlags(b, "-stdlib=libc++ -lc++abi", sanitizer_flags));
        try appendCmakeDefine(cmake_args, b, "CMAKE_SHARED_LINKER_FLAGS", joinCmakeFlags(b, "-stdlib=libc++ -lc++abi", sanitizer_flags));
        try appendCmakeDefine(cmake_args, b, "CMAKE_MODULE_LINKER_FLAGS", joinCmakeFlags(b, "-stdlib=libc++ -lc++abi", sanitizer_flags));
        try cmake_args.append("-DCMAKE_CXX_COMPILER=clang++");
        try cmake_args.append("-DCMAKE_C_COMPILER=clang");
    } else if (builtin.os.tag == .macos) {
        try appendCmakeDefine(cmake_args, b, "CMAKE_CXX_FLAGS", joinCmakeFlags(b, "-stdlib=libc++", sanitizer_flags));
        if (sanitizer.enabled()) {
            try appendCmakeDefine(cmake_args, b, "CMAKE_EXE_LINKER_FLAGS", sanitizer_flags);
            try appendCmakeDefine(cmake_args, b, "CMAKE_SHARED_LINKER_FLAGS", sanitizer_flags);
            try appendCmakeDefine(cmake_args, b, "CMAKE_MODULE_LINKER_FLAGS", sanitizer_flags);
        }
    } else if (builtin.os.tag == .windows) {
        try cmake_args.append("-DCMAKE_CXX_FLAGS=/std:c++20");
    }
}

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
    const native_sanitize = parseNativeSanitizer(b.option([]const u8, "native-sanitize", "Build native MLIR dialect libraries with sanitizer (none, address, undefined, address,undefined)") orelse "none");

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
    const ora_refinements_mod = b.createModule(.{
        .root_source_file = b.path("src/refinements/root.zig"),
        .target = target,
        .optimize = optimize,
    });

    // Standalone pipeline modules with explicit dependency boundaries.
    // These enable fast builds/tests for frontend-only work without MLIR/Z3.
    const ora_types_mod = b.createModule(.{
        .root_source_file = b.path("src/types/root.zig"),
        .target = target,
        .optimize = optimize,
    });
    ora_types_mod.addImport("ora_refinements", ora_refinements_mod);

    const ora_lexer_mod = b.createModule(.{
        .root_source_file = b.path("src/lexer.zig"),
        .target = target,
        .optimize = optimize,
    });
    ora_lexer_mod.addImport("ora_types", ora_types_mod);

    const ora_imports_mod = b.createModule(.{
        .root_source_file = b.path("src/imports/mod.zig"),
        .target = target,
        .optimize = optimize,
    });
    const stdlib_embedded_mod = b.createModule(.{
        .root_source_file = b.path("src/stdlib_embedded.zig"),
        .target = target,
        .optimize = optimize,
    });
    ora_imports_mod.addImport("ora_lexer", ora_lexer_mod);
    ora_imports_mod.addImport("stdlib_embedded", stdlib_embedded_mod);

    const ora_fmt_mod = b.createModule(.{
        .root_source_file = b.path("src/fmt/mod.zig"),
        .target = target,
        .optimize = optimize,
    });
    ora_fmt_mod.addImport("ora_lib", lib_mod);

    const mlir_c_mod = b.createModule(.{
        .root_source_file = b.path("src/mlir/c.zig"),
        .target = target,
        .optimize = optimize,
    });
    mlir_c_mod.addIncludePath(b.path("vendor/mlir/include"));
    if (native_sanitize.enabled()) {
        mlir_c_mod.addIncludePath(b.path(b.fmt("{s}/include", .{nativeMlirInstallPrefix(b, native_sanitize)})));
    }
    mlir_c_mod.addIncludePath(b.path("src/mlir/ora/include"));
    mlir_c_mod.addIncludePath(b.path("src/mlir/IR/include"));
    const mlir_helpers_mod = b.createModule(.{
        .root_source_file = b.path("src/mlir/helpers.zig"),
        .target = target,
        .optimize = optimize,
    });
    mlir_helpers_mod.addImport("mlir_c_api", mlir_c_mod);
    mlir_helpers_mod.addImport("ora_types", ora_types_mod);
    const prepared_query_row_mod = b.createModule(.{
        .root_source_file = b.path("src/formal/shared/prepared_query_row.zig"),
        .target = target,
        .optimize = optimize,
    });
    const z3_verification_mod = b.createModule(.{
        .root_source_file = b.path("src/z3/verification.zig"),
        .target = target,
        .optimize = optimize,
    });
    z3_verification_mod.addImport("mlir_c_api", mlir_c_mod);
    z3_verification_mod.addImport("ora_lib", lib_mod);
    z3_verification_mod.addImport("ora_types", ora_types_mod);
    z3_verification_mod.addImport("ora_prepared_query_row", prepared_query_row_mod);

    const evm_blst_lib = createEvmBlstLibrary(b, target, optimize);
    const evm_c_kzg_lib = createEvmCKzgLibrary(b, target, optimize, evm_blst_lib);
    const evm_c_kzg_mod = b.addModule("c_kzg", .{
        .root_source_file = b.path("lib/evm/vendor/c-kzg-4844/bindings/zig/root.zig"),
        .target = target,
        .optimize = optimize,
    });
    evm_c_kzg_mod.linkLibrary(evm_c_kzg_lib);
    evm_c_kzg_mod.linkLibrary(evm_blst_lib);
    evm_c_kzg_mod.addIncludePath(b.path("lib/evm/vendor/c-kzg-4844/src"));
    evm_c_kzg_mod.addIncludePath(b.path("lib/evm/vendor/c-kzg-4844/blst/bindings"));
    const evm_rust_crypto_lib_path = b.path("lib/evm/target/release/libora_evm_crypto_wrappers.a");

    const bootstrap_ora_evm_crypto = b.addSystemCommand(&.{
        "cargo",
        "build",
        "--manifest-path",
        b.path("lib/evm/Cargo.toml").getPath(b),
        "--release",
    });
    bootstrap_ora_evm_crypto.setName("bootstrap-ora-evm-crypto");

    const ora_evm_mod = b.createModule(.{
        .root_source_file = b.path("lib/evm/src/root.zig"),
        .target = target,
        .optimize = optimize,
        .imports = &.{
            .{ .name = "c_kzg", .module = evm_c_kzg_mod },
        },
    });
    ora_evm_mod.addObjectFile(evm_rust_crypto_lib_path);
    ora_evm_mod.link_libc = true;

    // modules can depend on one another using the `std.Build.Module.addImport` function.
    // this is what allows Zig source code to use `@import("foo")` where 'foo' is not a
    // file path. In this case, we set up `exe_mod` to import `lib_mod`.
    exe_mod.addImport("ora_lib", lib_mod);
    exe_mod.addImport("ora_imports", ora_imports_mod);
    exe_mod.addImport("mlir_c_api", mlir_c_mod);
    exe_mod.addImport("mlir_helpers", mlir_helpers_mod);
    exe_mod.addImport("log", log_mod);
    exe_mod.addImport("ora_lexer", ora_lexer_mod);
    exe_mod.addImport("ora_types", ora_types_mod);
    exe_mod.addImport("ora_refinements", ora_refinements_mod);
    exe_mod.addImport("ora_z3_verification", z3_verification_mod);
    exe_mod.addImport("ora_prepared_query_row", prepared_query_row_mod);
    exe_mod.addImport("ora_root", lib_mod);
    lib_mod.addImport("mlir_c_api", mlir_c_mod);
    lib_mod.addImport("mlir_helpers", mlir_helpers_mod);
    lib_mod.addImport("log", log_mod);
    lib_mod.addImport("ora_types", ora_types_mod);
    lib_mod.addImport("ora_refinements", ora_refinements_mod);
    lib_mod.addImport("ora_lexer", ora_lexer_mod);
    lib_mod.addImport("ora_imports", ora_imports_mod);

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

    // standalone LSP server executable (frontend-only path)
    const lsp_exe_mod = b.createModule(.{
        .root_source_file = b.path("src/lsp/main.zig"),
        .target = target,
        .optimize = optimize,
    });
    lsp_exe_mod.addImport("ora_root", lib_mod);
    lsp_exe_mod.addImport("ora_lib", lib_mod);
    lsp_exe_mod.addImport("ora_fmt", ora_fmt_mod);
    lsp_exe_mod.addImport("lsp", b.dependency("lsp_kit", .{}).module("lsp"));

    const lsp_exe = b.addExecutable(.{
        .name = "ora-lsp",
        .root_module = lsp_exe_mod,
    });

    const lsp_release_optimize: std.builtin.OptimizeMode = .ReleaseFast;
    const lsp_release_refinements_mod = b.createModule(.{
        .root_source_file = b.path("src/refinements/root.zig"),
        .target = target,
        .optimize = lsp_release_optimize,
    });
    const lsp_release_types_mod = b.createModule(.{
        .root_source_file = b.path("src/types/root.zig"),
        .target = target,
        .optimize = lsp_release_optimize,
    });
    lsp_release_types_mod.addImport("ora_refinements", lsp_release_refinements_mod);

    const lsp_release_lexer_mod = b.createModule(.{
        .root_source_file = b.path("src/lexer.zig"),
        .target = target,
        .optimize = lsp_release_optimize,
    });
    lsp_release_lexer_mod.addImport("ora_types", lsp_release_types_mod);

    const lsp_release_stdlib_embedded_mod = b.createModule(.{
        .root_source_file = b.path("src/stdlib_embedded.zig"),
        .target = target,
        .optimize = lsp_release_optimize,
    });
    const lsp_release_imports_mod = b.createModule(.{
        .root_source_file = b.path("src/imports/mod.zig"),
        .target = target,
        .optimize = lsp_release_optimize,
    });
    lsp_release_imports_mod.addImport("ora_lexer", lsp_release_lexer_mod);
    lsp_release_imports_mod.addImport("stdlib_embedded", lsp_release_stdlib_embedded_mod);

    const lsp_release_mlir_c_mod = b.createModule(.{
        .root_source_file = b.path("src/mlir/c.zig"),
        .target = target,
        .optimize = lsp_release_optimize,
    });
    lsp_release_mlir_c_mod.addIncludePath(b.path("vendor/mlir/include"));
    if (native_sanitize.enabled()) {
        lsp_release_mlir_c_mod.addIncludePath(b.path(b.fmt("{s}/include", .{nativeMlirInstallPrefix(b, native_sanitize)})));
    }
    lsp_release_mlir_c_mod.addIncludePath(b.path("src/mlir/ora/include"));
    lsp_release_mlir_c_mod.addIncludePath(b.path("src/mlir/IR/include"));

    const lsp_release_mlir_helpers_mod = b.createModule(.{
        .root_source_file = b.path("src/mlir/helpers.zig"),
        .target = target,
        .optimize = lsp_release_optimize,
    });
    lsp_release_mlir_helpers_mod.addImport("mlir_c_api", lsp_release_mlir_c_mod);
    lsp_release_mlir_helpers_mod.addImport("ora_types", lsp_release_types_mod);

    const lsp_release_log_mod = b.createModule(.{
        .root_source_file = b.path("src/logging.zig"),
        .target = target,
        .optimize = lsp_release_optimize,
    });
    const lsp_release_lib_mod = b.createModule(.{
        .root_source_file = b.path("src/root.zig"),
        .target = target,
        .optimize = lsp_release_optimize,
    });
    lsp_release_lib_mod.addImport("mlir_c_api", lsp_release_mlir_c_mod);
    lsp_release_lib_mod.addImport("mlir_helpers", lsp_release_mlir_helpers_mod);
    lsp_release_lib_mod.addImport("log", lsp_release_log_mod);
    lsp_release_lib_mod.addImport("ora_types", lsp_release_types_mod);
    lsp_release_lib_mod.addImport("ora_refinements", lsp_release_refinements_mod);
    lsp_release_lib_mod.addImport("ora_lexer", lsp_release_lexer_mod);
    lsp_release_lib_mod.addImport("ora_imports", lsp_release_imports_mod);

    const lsp_release_fmt_mod = b.createModule(.{
        .root_source_file = b.path("src/fmt/mod.zig"),
        .target = target,
        .optimize = lsp_release_optimize,
    });
    lsp_release_fmt_mod.addImport("ora_lib", lsp_release_lib_mod);

    const lsp_release_exe_mod = b.createModule(.{
        .root_source_file = b.path("src/lsp/main.zig"),
        .target = target,
        .optimize = lsp_release_optimize,
    });
    lsp_release_exe_mod.addImport("ora_root", lsp_release_lib_mod);
    lsp_release_exe_mod.addImport("ora_lib", lsp_release_lib_mod);
    lsp_release_exe_mod.addImport("ora_fmt", lsp_release_fmt_mod);
    lsp_release_exe_mod.addImport("lsp", b.dependency("lsp_kit", .{}).module("lsp"));

    const lsp_release_exe = b.addExecutable(.{
        .name = "ora-lsp-release",
        .root_module = lsp_release_exe_mod,
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
    lsp_release_lib_mod.addOptions("build_options", mlir_options);

    // add include paths
    exe.root_module.addIncludePath(b.path("src"));
    lib.root_module.addIncludePath(b.path("src"));

    // add Ora dialect include path (for OraCAPI.h)
    const ora_dialect_include_path = b.path("src/mlir/ora/include");
    exe.root_module.addIncludePath(ora_dialect_include_path);

    // add SIR dialect include path
    const sir_dialect_include_path = b.path("src/mlir/IR/include");
    exe.root_module.addIncludePath(sir_dialect_include_path);

    // build and link MLIR (required) - only for executable, not library
    const mlir_step = if (skip_mlir_build) null else buildMlirLibraries(b, target, optimize);
    // build SIR dialect first (Ora dialect depends on it)
    const sir_dialect_step = if (skip_mlir_build) null else buildSIRDialectLibrary(b, mlir_step.?, target, optimize, native_sanitize);
    const ora_dialect_step = if (skip_mlir_build) null else buildOraDialectLibrary(b, mlir_step.?, sir_dialect_step.?, target, optimize, native_sanitize);
    linkMlirLibraries(b, exe, mlir_step, ora_dialect_step, sir_dialect_step, target, native_sanitize);

    // The LSP executables cImport mlir-c headers from vendor/mlir/include but do
    // not link the MLIR libraries, so nothing orders their compile after the MLIR
    // build. In from-source builds (e.g. the CI mlir-build job on a cache miss)
    // that lets ora-lsp compile before the headers are installed -> "mlir-c/IR.h
    // not found". Order them after the MLIR build step. skip-mlir builds use the
    // prebuilt artifact (mlir_step == null) and need no ordering.
    if (mlir_step) |ms| {
        lsp_exe.step.dependOn(ms);
        lsp_release_exe.step.dependOn(ms);
    }

    // build and link Z3 (for formal verification) - only for executable
    const z3_step = buildZ3Libraries(b, target, optimize);
    linkZ3Libraries(b, exe, z3_step, target);

    const loop_census_mod = b.createModule(.{
        .root_source_file = b.path("src/loop_census_main.zig"),
        .target = target,
        .optimize = optimize,
    });
    loop_census_mod.addImport("ora_root", lib_mod);
    loop_census_mod.addImport("ora_lib", lib_mod);
    loop_census_mod.addImport("ora_types", ora_types_mod);
    loop_census_mod.addImport("mlir_c_api", mlir_c_mod);
    loop_census_mod.addImport("ora_z3_verification", z3_verification_mod);
    const loop_census_exe = b.addExecutable(.{
        .name = "ora-loop-census",
        .root_module = loop_census_mod,
    });
    linkMlirLibraries(b, loop_census_exe, mlir_step, ora_dialect_step, sir_dialect_step, target, native_sanitize);
    linkZ3Libraries(b, loop_census_exe, z3_step, target);
    const loop_census_install = b.addInstallArtifact(loop_census_exe, .{});
    const loop_census_tool_step = b.step("loop-census-tool", "Build the measurement-only loop census emitter");
    loop_census_tool_step.dependOn(&loop_census_install.step);

    // this declares intent for the executable to be installed into the
    // standard location when the user invokes the "install" step (the default
    // step when running `zig build`).
    b.installArtifact(exe);
    b.installArtifact(lsp_exe);

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

    const cfg_cmd = b.addRunArtifact(exe);
    cfg_cmd.step.dependOn(b.getInstallStep());
    cfg_cmd.addArgs(&[_][]const u8{ "emit", "--emit=cfg:sir" });
    if (b.args) |args| {
        cfg_cmd.addArgs(args);
    }
    const cfg_step = b.step("cfg", "Emit SIR CFG DOT for an Ora source file");
    cfg_step.dependOn(&cfg_cmd.step);

    const lsp_build_step = b.step("ora-lsp", "Build the Ora LSP server");
    lsp_build_step.dependOn(&lsp_exe.step);

    const evm_debug_tui_cmd = b.addSystemCommand(&[_][]const u8{
        "zig",
        "build",
        "-Ddebug-ui=true",
        "debug-tui",
        "--",
    });
    evm_debug_tui_cmd.setCwd(b.path("lib/evm"));
    if (b.args) |args| {
        evm_debug_tui_cmd.addArgs(args);
    }
    const evm_debug_tui_step = b.step("debug-tui", "Run the Ora EVM debugger TUI");
    evm_debug_tui_step.dependOn(&evm_debug_tui_cmd.step);

    // DAP server. Same pattern as debug-tui: shell out to the
    // lib/evm sub-build so the top-level `zig build debug-dap`
    // works without duplicating the DAP build configuration.
    const evm_debug_dap_cmd = b.addSystemCommand(&[_][]const u8{
        "zig",
        "build",
        "-Ddebug-ui=true",
        "debug-dap",
        "--",
    });
    evm_debug_dap_cmd.setCwd(b.path("lib/evm"));
    if (b.args) |args| {
        evm_debug_dap_cmd.addArgs(args);
    }
    const evm_debug_dap_step = b.step("debug-dap", "Run the Ora EVM debugger DAP server (Content-Length-framed JSON-RPC over stdio)");
    evm_debug_dap_step.dependOn(&evm_debug_dap_cmd.step);

    // Sinora: Ora's owned Zig SIR->EVM backend. Build it by default so Ora can
    // emit bytecode with no external SIR backend.
    const sinora_root = "sinora";
    const sinora_mod = b.createModule(.{
        .root_source_file = b.path("sinora/src/sinora.zig"),
        .target = target,
        .optimize = optimize,
    });
    var sinora_build_dependency: ?*std.Build.Step = null;
    if (std.Io.Dir.cwd().access(b.graph.io, b.fmt("{s}/build.zig", .{sinora_root}), .{}) catch null) |_| {
        const sinora_build_cmd = b.addSystemCommand(&[_][]const u8{ "zig", "build" });
        sinora_build_cmd.setCwd(b.path(sinora_root));
        sinora_build_dependency = &sinora_build_cmd.step;

        const sinora_build_step = b.step("sinora", "Build the Sinora SIR -> EVM backend binary");
        sinora_build_step.dependOn(&sinora_build_cmd.step);
    }
    if (sinora_build_dependency) |dep| b.getInstallStep().dependOn(dep);
    exe_mod.addImport("sinora", sinora_mod);

    const dispatcher_table_snapshot_mod = b.createModule(.{
        .root_source_file = b.path("src/formal/emit_dispatcher_table_snapshot.zig"),
        .target = target,
        .optimize = optimize,
    });
    dispatcher_table_snapshot_mod.addImport("ora_root", lib_mod);
    dispatcher_table_snapshot_mod.addImport("mlir_c_api", mlir_c_mod);
    dispatcher_table_snapshot_mod.addImport("sinora", sinora_mod);
    const dispatcher_table_snapshot_exe = b.addExecutable(.{
        .name = "ora-dispatcher-table-snapshot",
        .root_module = dispatcher_table_snapshot_mod,
    });
    linkMlirLibraries(b, dispatcher_table_snapshot_exe, mlir_step, ora_dialect_step, sir_dialect_step, target, native_sanitize);
    const dispatcher_table_snapshot_run = b.addRunArtifact(dispatcher_table_snapshot_exe);
    if (b.args) |args| {
        dispatcher_table_snapshot_run.addArgs(args);
    }
    const dispatcher_table_snapshot_step = b.step("emit-dispatcher-table-snapshot", "Emit formal dispatcher table snapshot");
    dispatcher_table_snapshot_step.dependOn(&dispatcher_table_snapshot_run.step);

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

    // Runtime conformance harness: compile Ora through the installed CLI,
    // deploy the emitted bytecode into lib/evm, and assert sidecar outcomes.
    const conformance_test_mod = b.createModule(.{
        .root_source_file = b.path("tests/conformance/root.zig"),
        .target = target,
        .optimize = optimize,
    });
    conformance_test_mod.addImport("ora_evm", ora_evm_mod);
    const conformance_tests = b.addTest(.{
        .name = "ora-conformance-tests",
        .root_module = conformance_test_mod,
    });
    conformance_tests.step.dependOn(&bootstrap_ora_evm_crypto.step);
    const conformance_tests_run = b.addRunArtifact(conformance_tests);
    conformance_tests_run.step.dependOn(b.getInstallStep());

    // Pinned verified-build lane: classify the full corpus by verifier outcome
    // and bytecode equality, then execute every verifier-rewritten contract.
    const verified_conformance_test_mod = b.createModule(.{
        .root_source_file = b.path("tests/conformance/verified_build.zig"),
        .target = target,
        .optimize = optimize,
    });
    verified_conformance_test_mod.addImport("ora_evm", ora_evm_mod);
    const verified_conformance_tests = b.addTest(.{
        .name = "ora-verified-conformance-tests",
        .root_module = verified_conformance_test_mod,
    });
    verified_conformance_tests.step.dependOn(&bootstrap_ora_evm_crypto.step);
    const verified_conformance_tests_run = b.addRunArtifact(verified_conformance_tests);
    verified_conformance_tests_run.step.dependOn(b.getInstallStep());

    const test_conformance_verified_step = b.step("test-conformance-verified", "Run the verified-build Ora bytecode conformance lane");
    test_conformance_verified_step.dependOn(&verified_conformance_tests_run.step);

    const verified_conformance_rail_test_mod = b.createModule(.{
        .root_source_file = b.path("tests/conformance/verified_build.zig"),
        .target = target,
        .optimize = optimize,
    });
    verified_conformance_rail_test_mod.addImport("ora_evm", ora_evm_mod);
    const verified_conformance_rail_tests = b.addTest(.{
        .name = "ora-verified-conformance-rail-tests",
        .root_module = verified_conformance_rail_test_mod,
        .filters = &.{
            "verified emit classification rejects abnormal termination",
            "verified emit classification names normal verifier rejections",
            "verified conformance manifest parser rejects unordered membership",
            "verified conformance manifest requires class-C reasons",
        },
    });
    verified_conformance_rail_tests.step.dependOn(&bootstrap_ora_evm_crypto.step);
    const verified_conformance_rail_tests_run = b.addRunArtifact(verified_conformance_rail_tests);
    const test_conformance_verified_rail_step = b.step("test-conformance-verified-rail", "Run verified-build classification rail tests");
    test_conformance_verified_rail_step.dependOn(&verified_conformance_rail_tests_run.step);

    const test_conformance_step = b.step("test-conformance", "Run Ora bytecode conformance tests on lib/evm");
    test_conformance_step.dependOn(&conformance_tests_run.step);
    test_conformance_step.dependOn(&verified_conformance_tests_run.step);

    // Single-spec lib/evm runner — runs ONE .ora+.spec.toml outside the harness
    // (used by the Anvil differential proof to observe lib/evm on one call).
    const conformance_one_mod = b.createModule(.{
        .root_source_file = b.path("tests/conformance/single_spec.zig"),
        .target = target,
        .optimize = optimize,
    });
    conformance_one_mod.addImport("ora_evm", ora_evm_mod);
    const conformance_one_exe = b.addExecutable(.{
        .name = "conformance-one",
        .root_module = conformance_one_mod,
    });
    conformance_one_exe.step.dependOn(&bootstrap_ora_evm_crypto.step);
    const conformance_one_install = b.addInstallArtifact(conformance_one_exe, .{});
    const conformance_one_step = b.step("conformance-one", "Build the single-spec lib/evm runner");
    conformance_one_step.dependOn(&conformance_one_install.step);

    // Local Anvil/revm differential over the conformance corpus. This is
    // intentionally not part of the default gate because it drives a live RPC
    // process, but it is the local entrypoint and mirrors the scheduled CI
    // differential job.
    const conformance_anvil_cmd = b.addSystemCommand(&[_][]const u8{
        "bash",
        "scripts/anvil-diff-corpus.sh",
    });
    conformance_anvil_cmd.step.dependOn(b.getInstallStep());
    const test_conformance_anvil_step = b.step("test-conformance-anvil", "Run Ora conformance differential tests on Anvil/revm");
    test_conformance_anvil_step.dependOn(&conformance_anvil_cmd.step);

    const conformance_anvil_parser_check = b.addSystemCommand(&.{
        "python3",
        "scripts/conformance-anvil-diff.py",
        "--self-test",
    });
    const check_conformance_anvil_parser_step = b.step("check-conformance-anvil-parser", "Check Anvil RPC revert-data parsing");
    check_conformance_anvil_parser_step.dependOn(&conformance_anvil_parser_check.step);
    test_step.dependOn(&conformance_anvil_parser_check.step);

    // Metrics snapshot harness — prints gas + bytecode-size metrics per corpus
    // entry for the change-quality benchmark.
    const metrics_snapshot_mod = b.createModule(.{
        .root_source_file = b.path("tests/conformance/metrics_snapshot.zig"),
        .target = target,
        .optimize = optimize,
    });
    metrics_snapshot_mod.addImport("ora_evm", ora_evm_mod);
    const metrics_snapshot_exe = b.addExecutable(.{
        .name = "metrics-snapshot",
        .root_module = metrics_snapshot_mod,
    });
    metrics_snapshot_exe.step.dependOn(&bootstrap_ora_evm_crypto.step);
    const metrics_snapshot_install = b.addInstallArtifact(metrics_snapshot_exe, .{});
    const metrics_snapshot_step = b.step("metrics-snapshot", "Build the gas + bytecode-size metrics harness");
    metrics_snapshot_step.dependOn(&metrics_snapshot_install.step);
    const check_conformance_bytecode_size_cmd = b.addSystemCommand(&[_][]const u8{
        "python3",
        "scripts/metrics-check.py",
        "--check-size",
        "--report-dir",
        "zig-out/metrics/conformance-size",
    });
    check_conformance_bytecode_size_cmd.step.dependOn(b.getInstallStep());
    check_conformance_bytecode_size_cmd.step.dependOn(&metrics_snapshot_install.step);
    // Serialize behind the test bar: the harness shells the compiler per
    // corpus spec, and under full gate parallelism resource contention can
    // fail a heavy spec transiently. Running after tests costs seconds and
    // removes the race window.
    check_conformance_bytecode_size_cmd.step.dependOn(test_step);
    const check_conformance_bytecode_size_step = b.step("check-conformance-bytecode-size", "Check conformance bytecode-size metrics against the deterministic baseline");
    check_conformance_bytecode_size_step.dependOn(&check_conformance_bytecode_size_cmd.step);
    const check_metrics_report_cmd = b.addSystemCommand(&[_][]const u8{
        "sh",
        "scripts/check-metrics-report.sh",
    });
    const check_metrics_report_step = b.step("check-metrics-report", "Check metrics report and size-gate script behavior");
    check_metrics_report_step.dependOn(&check_metrics_report_cmd.step);
    test_step.dependOn(&check_metrics_report_cmd.step);
    const sir_framework_spike_cmd = b.addSystemCommand(&[_][]const u8{
        "sh",
        "scripts/run-sir-framework-canonicalizer-spike.sh",
    });
    sir_framework_spike_cmd.step.dependOn(b.getInstallStep());
    const sir_framework_spike_step = b.step("sir-framework-canonicalizer-spike", "Run the default SIR framework canonicalizer over the Ora example corpus");
    sir_framework_spike_step.dependOn(&sir_framework_spike_cmd.step);
    const sir_framework_metrics_cmd = b.addSystemCommand(&[_][]const u8{
        "python3",
        "scripts/metrics-check.py",
        "--diff",
        "--report-dir",
        "zig-out/metrics/sir-framework-canonicalizer-size",
    });
    sir_framework_metrics_cmd.step.dependOn(b.getInstallStep());
    sir_framework_metrics_cmd.step.dependOn(&metrics_snapshot_install.step);
    const sir_framework_metrics_step = b.step("sir-framework-canonicalizer-metrics", "Write size/gas report for the default SIR framework canonicalizer pipeline");
    sir_framework_metrics_step.dependOn(&sir_framework_metrics_cmd.step);

    // Compiler frontend metrics harness — prints deterministic compile-time
    // allocation/work-count metrics for package-mode Ora examples.
    const compile_metrics_mod = b.createModule(.{
        .root_source_file = b.path("tests/compile_metrics_snapshot.zig"),
        .target = target,
        .optimize = optimize,
    });
    compile_metrics_mod.addImport("ora_root", lib_mod);
    const compile_metrics_exe = b.addExecutable(.{
        .name = "compile-metrics",
        .root_module = compile_metrics_mod,
    });
    linkMlirLibraries(b, compile_metrics_exe, mlir_step, ora_dialect_step, sir_dialect_step, target, native_sanitize);
    const compile_metrics_install = b.addInstallArtifact(compile_metrics_exe, .{});
    const compile_metrics_step = b.step("compile-metrics", "Build the compiler frontend metrics harness");
    compile_metrics_step.dependOn(&compile_metrics_install.step);
    const check_compile_metrics_cmd = b.addSystemCommand(&[_][]const u8{
        "python3",
        "scripts/compile-metrics-check.py",
        "--check",
    });
    check_compile_metrics_cmd.step.dependOn(&compile_metrics_install.step);
    const check_compile_metrics_step = b.step("check-compile-metrics", "Check compiler frontend metrics against the deterministic baseline");
    check_compile_metrics_step.dependOn(&check_compile_metrics_cmd.step);

    const evm_tests_cmd = b.addSystemCommand(&[_][]const u8{
        "zig",
        "build",
        "unit",
        "--summary",
        "all",
    });
    evm_tests_cmd.setCwd(b.path("lib/evm"));
    const test_evm_step = b.step("test-evm", "Run lib/evm unit tests");
    test_evm_step.dependOn(&evm_tests_cmd.step);

    // lexer tests
    const lexer_test_mod = b.createModule(.{
        .root_source_file = b.path("src/lexer.test.zig"),
        .target = target,
        .optimize = optimize,
    });
    lexer_test_mod.addImport("ora_root", lib_mod);
    const lexer_tests = b.addTest(.{ .root_module = lexer_test_mod });
    test_step.dependOn(&b.addRunArtifact(lexer_tests).step);

    // cli argument parsing tests
    const cli_args_test_mod = b.createModule(.{
        .root_source_file = b.path("src/cli/args.test.zig"),
        .target = target,
        .optimize = optimize,
    });
    const cli_args_tests = b.addTest(.{ .root_module = cli_args_test_mod });
    test_step.dependOn(&b.addRunArtifact(cli_args_tests).step);

    // project config tests
    const project_config_test_mod = b.createModule(.{
        .root_source_file = b.path("src/config/mod.test.zig"),
        .target = target,
        .optimize = optimize,
    });
    const project_config_tests = b.addTest(.{ .root_module = project_config_test_mod });
    test_step.dependOn(&b.addRunArtifact(project_config_tests).step);

    // ABI tests
    const abi_test_mod = b.createModule(.{
        .root_source_file = b.path("src/abi.test.zig"),
        .target = target,
        .optimize = optimize,
    });
    abi_test_mod.addImport("ora_root", lib_mod);
    abi_test_mod.addImport("mlir_c_api", mlir_c_mod);
    abi_test_mod.addImport("ora_types", ora_types_mod);
    const abi_tests = b.addTest(.{ .root_module = abi_test_mod });
    linkMlirLibraries(b, abi_tests, mlir_step, ora_dialect_step, sir_dialect_step, target, native_sanitize);
    linkZ3Libraries(b, abi_tests, z3_step, target);
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

    // import resolver tests
    const import_resolver_test_mod = b.createModule(.{
        .root_source_file = b.path("src/imports/mod.test.zig"),
        .target = target,
        .optimize = optimize,
    });
    import_resolver_test_mod.addImport("ora_lexer", ora_lexer_mod);
    import_resolver_test_mod.addImport("stdlib_embedded", stdlib_embedded_mod);
    import_resolver_test_mod.addImport("ora_types", ora_types_mod);
    const import_resolver_tests = b.addTest(.{ .root_module = import_resolver_test_mod });
    test_step.dependOn(&b.addRunArtifact(import_resolver_tests).step);

    // z3 encoder tests (requires MLIR + Z3). Split by category from the former
    // 22K-line encoder.test.zig; all share encoder_test_prelude.zig.
    const z3_encoder_test_files = [_][]const u8{
        "src/z3/encoder.test.types.zig",
        "src/z3/encoder.test.arith.zig",
        "src/z3/encoder.test.controlflow.zig",
        "src/z3/encoder.test.summaries.zig",
        "src/z3/encoder.test.storage.zig",
        "src/z3/encoder.test.misc.zig",
    };
    for (z3_encoder_test_files) |test_file| {
        const enc_mod = b.createModule(.{
            .root_source_file = b.path(test_file),
            .target = target,
            .optimize = optimize,
        });
        enc_mod.addImport("mlir_c_api", mlir_c_mod);
        const enc_tests = b.addTest(.{ .root_module = enc_mod });
        linkMlirLibraries(b, enc_tests, mlir_step, ora_dialect_step, sir_dialect_step, target, native_sanitize);
        linkZ3Libraries(b, enc_tests, z3_step, target);
        test_step.dependOn(&b.addRunArtifact(enc_tests).step);
    }

    // z3 verification tests (requires MLIR + Z3)
    const z3_verification_test_mod = b.createModule(.{
        .root_source_file = b.path("src/z3/verification.zig"),
        .target = target,
        .optimize = optimize,
    });
    z3_verification_test_mod.addImport("mlir_c_api", mlir_c_mod);
    z3_verification_test_mod.addImport("ora_lib", lib_mod);
    z3_verification_test_mod.addImport("ora_types", ora_types_mod);
    z3_verification_test_mod.addImport("ora_prepared_query_row", prepared_query_row_mod);
    const z3_verification_tests = b.addTest(.{ .root_module = z3_verification_test_mod });
    linkMlirLibraries(b, z3_verification_tests, mlir_step, ora_dialect_step, sir_dialect_step, target, native_sanitize);
    linkZ3Libraries(b, z3_verification_tests, z3_step, target);
    const z3_verification_tests_run = b.addRunArtifact(z3_verification_tests);
    const test_z3_verification_step = b.step("test-z3-verification", "Run Z3 verification tests");
    test_z3_verification_step.dependOn(&z3_verification_tests_run.step);
    test_step.dependOn(&z3_verification_tests_run.step);

    // Formal proof-checker tests. This target keeps the userland Lean proof
    // acceptance code in the normal build graph even before B3 wires it to CLI
    // artifact emission.
    const proof_check_test_mod = b.createModule(.{
        .root_source_file = b.path("src/formal/proof_check.zig"),
        .target = target,
        .optimize = optimize,
    });
    proof_check_test_mod.addImport("ora_types", ora_types_mod);
    const proof_check_tests = b.addTest(.{
        .name = "formal-proof-check-tests",
        .root_module = proof_check_test_mod,
    });
    const proof_check_tests_run = b.addRunArtifact(proof_check_tests);
    test_step.dependOn(&proof_check_tests_run.step);

    const test_proof_check_step = b.step("test-proof-check", "Run formal proof-checker tests");
    test_proof_check_step.dependOn(&proof_check_tests_run.step);

    // MLIR verifier-negative tests
    const mlir_verifiers_test_mod = b.createModule(.{
        .root_source_file = b.path("src/mlir/verifiers.test.zig"),
        .target = target,
        .optimize = optimize,
    });
    mlir_verifiers_test_mod.addImport("ora_lib", lib_mod);
    mlir_verifiers_test_mod.addImport("mlir_c_api", mlir_c_mod);
    mlir_verifiers_test_mod.addImport("log", log_mod);
    const mlir_verifiers_tests = b.addTest(.{ .root_module = mlir_verifiers_test_mod });
    linkMlirLibraries(b, mlir_verifiers_tests, mlir_step, ora_dialect_step, sir_dialect_step, target, native_sanitize);
    test_step.dependOn(&b.addRunArtifact(mlir_verifiers_tests).step);
    test_mlir_step.dependOn(&b.addRunArtifact(mlir_verifiers_tests).step);

    // lsp tests - Frontend diagnostics
    const lsp_frontend_test_mod = b.createModule(.{
        .root_source_file = b.path("src/lsp/frontend.test.zig"),
        .target = target,
        .optimize = optimize,
    });
    lsp_frontend_test_mod.addImport("ora_root", lib_mod);
    const lsp_frontend_tests = b.addTest(.{ .root_module = lsp_frontend_test_mod });
    test_step.dependOn(&b.addRunArtifact(lsp_frontend_tests).step);

    const lsp_workspace_test_mod = b.createModule(.{
        .root_source_file = b.path("src/lsp/workspace.test.zig"),
        .target = target,
        .optimize = optimize,
    });
    lsp_workspace_test_mod.addImport("ora_root", lib_mod);
    const lsp_workspace_tests = b.addTest(.{ .root_module = lsp_workspace_test_mod });
    test_step.dependOn(&b.addRunArtifact(lsp_workspace_tests).step);

    const lsp_dependency_graph_test_mod = b.createModule(.{
        .root_source_file = b.path("src/lsp/dependency_graph.test.zig"),
        .target = target,
        .optimize = optimize,
    });
    lsp_dependency_graph_test_mod.addImport("ora_root", lib_mod);
    const lsp_dependency_graph_tests = b.addTest(.{ .root_module = lsp_dependency_graph_test_mod });
    test_step.dependOn(&b.addRunArtifact(lsp_dependency_graph_tests).step);

    const lsp_semantic_index_test_mod = b.createModule(.{
        .root_source_file = b.path("src/lsp/semantic_index.test.zig"),
        .target = target,
        .optimize = optimize,
    });
    lsp_semantic_index_test_mod.addImport("ora_root", lib_mod);
    const lsp_semantic_index_tests = b.addTest(.{ .root_module = lsp_semantic_index_test_mod });
    test_step.dependOn(&b.addRunArtifact(lsp_semantic_index_tests).step);

    const lsp_text_edits_test_mod = b.createModule(.{
        .root_source_file = b.path("src/lsp/text_edits.test.zig"),
        .target = target,
        .optimize = optimize,
    });
    lsp_text_edits_test_mod.addImport("ora_root", lib_mod);
    const lsp_text_edits_tests = b.addTest(.{ .root_module = lsp_text_edits_test_mod });
    test_step.dependOn(&b.addRunArtifact(lsp_text_edits_tests).step);

    const lsp_hover_test_mod = b.createModule(.{
        .root_source_file = b.path("src/lsp/hover.test.zig"),
        .target = target,
        .optimize = optimize,
    });
    lsp_hover_test_mod.addImport("ora_root", lib_mod);
    const lsp_hover_tests = b.addTest(.{ .root_module = lsp_hover_test_mod });
    test_step.dependOn(&b.addRunArtifact(lsp_hover_tests).step);

    const lsp_definition_test_mod = b.createModule(.{
        .root_source_file = b.path("src/lsp/definition.test.zig"),
        .target = target,
        .optimize = optimize,
    });
    lsp_definition_test_mod.addImport("ora_root", lib_mod);
    const lsp_definition_tests = b.addTest(.{ .root_module = lsp_definition_test_mod });
    test_step.dependOn(&b.addRunArtifact(lsp_definition_tests).step);

    const lsp_references_test_mod = b.createModule(.{
        .root_source_file = b.path("src/lsp/references.test.zig"),
        .target = target,
        .optimize = optimize,
    });
    lsp_references_test_mod.addImport("ora_root", lib_mod);
    const lsp_references_tests = b.addTest(.{ .root_module = lsp_references_test_mod });
    test_step.dependOn(&b.addRunArtifact(lsp_references_tests).step);

    const lsp_document_highlight_test_mod = b.createModule(.{
        .root_source_file = b.path("src/lsp/document_highlight.test.zig"),
        .target = target,
        .optimize = optimize,
    });
    lsp_document_highlight_test_mod.addImport("ora_root", lib_mod);
    lsp_document_highlight_test_mod.addImport("lsp", b.dependency("lsp_kit", .{}).module("lsp"));
    const lsp_document_highlight_tests = b.addTest(.{ .root_module = lsp_document_highlight_test_mod });
    test_step.dependOn(&b.addRunArtifact(lsp_document_highlight_tests).step);

    const lsp_rename_test_mod = b.createModule(.{
        .root_source_file = b.path("src/lsp/rename.test.zig"),
        .target = target,
        .optimize = optimize,
    });
    lsp_rename_test_mod.addImport("ora_root", lib_mod);
    const lsp_rename_tests = b.addTest(.{ .root_module = lsp_rename_test_mod });
    test_step.dependOn(&b.addRunArtifact(lsp_rename_tests).step);

    const lsp_completion_test_mod = b.createModule(.{
        .root_source_file = b.path("src/lsp/completion.test.zig"),
        .target = target,
        .optimize = optimize,
    });
    lsp_completion_test_mod.addImport("ora_root", lib_mod);
    lsp_completion_test_mod.addImport("ora_types", ora_types_mod);
    const lsp_completion_tests = b.addTest(.{ .root_module = lsp_completion_test_mod });
    test_step.dependOn(&b.addRunArtifact(lsp_completion_tests).step);

    const lsp_signature_help_test_mod = b.createModule(.{
        .root_source_file = b.path("src/lsp/signature_help.test.zig"),
        .target = target,
        .optimize = optimize,
    });
    lsp_signature_help_test_mod.addImport("ora_root", lib_mod);
    const lsp_signature_help_tests = b.addTest(.{ .root_module = lsp_signature_help_test_mod });
    test_step.dependOn(&b.addRunArtifact(lsp_signature_help_tests).step);

    const lsp_semantic_tokens_test_mod = b.createModule(.{
        .root_source_file = b.path("src/lsp/semantic_tokens.test.zig"),
        .target = target,
        .optimize = optimize,
    });
    lsp_semantic_tokens_test_mod.addImport("ora_root", lib_mod);
    const lsp_semantic_tokens_tests = b.addTest(.{ .root_module = lsp_semantic_tokens_test_mod });
    test_step.dependOn(&b.addRunArtifact(lsp_semantic_tokens_tests).step);

    const lsp_token_cache_test_mod = b.createModule(.{
        .root_source_file = b.path("src/lsp/token_cache.test.zig"),
        .target = target,
        .optimize = optimize,
    });
    lsp_token_cache_test_mod.addImport("ora_root", lib_mod);
    const lsp_token_cache_tests = b.addTest(.{ .root_module = lsp_token_cache_test_mod });
    test_step.dependOn(&b.addRunArtifact(lsp_token_cache_tests).step);

    const lsp_code_lens_test_mod = b.createModule(.{
        .root_source_file = b.path("src/lsp/code_lens.test.zig"),
        .target = target,
        .optimize = optimize,
    });
    lsp_code_lens_test_mod.addImport("ora_root", lib_mod);
    const lsp_code_lens_tests = b.addTest(.{ .root_module = lsp_code_lens_test_mod });
    test_step.dependOn(&b.addRunArtifact(lsp_code_lens_tests).step);

    const lsp_folding_test_mod = b.createModule(.{
        .root_source_file = b.path("src/lsp/folding.test.zig"),
        .target = target,
        .optimize = optimize,
    });
    lsp_folding_test_mod.addImport("ora_root", lib_mod);
    const lsp_folding_tests = b.addTest(.{ .root_module = lsp_folding_test_mod });
    test_step.dependOn(&b.addRunArtifact(lsp_folding_tests).step);

    const lsp_cache_stats_response_test_mod = b.createModule(.{
        .root_source_file = b.path("src/lsp/cache_stats_response.test.zig"),
        .target = target,
        .optimize = optimize,
    });
    const lsp_cache_stats_response_tests = b.addTest(.{ .root_module = lsp_cache_stats_response_test_mod });
    test_step.dependOn(&b.addRunArtifact(lsp_cache_stats_response_tests).step);

    const lsp_allocation_stats_test_mod = b.createModule(.{
        .root_source_file = b.path("src/lsp/allocation_stats.test.zig"),
        .target = target,
        .optimize = optimize,
    });
    lsp_allocation_stats_test_mod.addImport("ora_root", lib_mod);
    const lsp_allocation_stats_tests = b.addTest(.{ .root_module = lsp_allocation_stats_test_mod });
    test_step.dependOn(&b.addRunArtifact(lsp_allocation_stats_tests).step);

    const lsp_signature_help_response_test_mod = b.createModule(.{
        .root_source_file = b.path("src/lsp/signature_help_response.test.zig"),
        .target = target,
        .optimize = optimize,
    });
    lsp_signature_help_response_test_mod.addImport("ora_root", lib_mod);
    lsp_signature_help_response_test_mod.addImport("lsp", b.dependency("lsp_kit", .{}).module("lsp"));
    const lsp_signature_help_response_tests = b.addTest(.{ .root_module = lsp_signature_help_response_test_mod });
    test_step.dependOn(&b.addRunArtifact(lsp_signature_help_response_tests).step);

    const lsp_builtin_docs_test_mod = b.createModule(.{
        .root_source_file = b.path("src/lsp/builtin_docs.test.zig"),
        .target = target,
        .optimize = optimize,
    });
    lsp_builtin_docs_test_mod.addImport("ora_root", lib_mod);
    const lsp_builtin_docs_tests = b.addTest(.{ .root_module = lsp_builtin_docs_test_mod });
    test_step.dependOn(&b.addRunArtifact(lsp_builtin_docs_tests).step);

    const lsp_keyword_docs_test_mod = b.createModule(.{
        .root_source_file = b.path("src/lsp/keyword_docs.test.zig"),
        .target = target,
        .optimize = optimize,
    });
    lsp_keyword_docs_test_mod.addImport("ora_root", lib_mod);
    const lsp_keyword_docs_tests = b.addTest(.{ .root_module = lsp_keyword_docs_test_mod });
    test_step.dependOn(&b.addRunArtifact(lsp_keyword_docs_tests).step);

    const lsp_std_docs_test_mod = b.createModule(.{
        .root_source_file = b.path("src/lsp/std_docs.test.zig"),
        .target = target,
        .optimize = optimize,
    });
    lsp_std_docs_test_mod.addImport("ora_root", lib_mod);
    const lsp_std_docs_tests = b.addTest(.{ .root_module = lsp_std_docs_test_mod });
    test_step.dependOn(&b.addRunArtifact(lsp_std_docs_tests).step);

    const lsp_call_hierarchy_test_mod = b.createModule(.{
        .root_source_file = b.path("src/lsp/call_hierarchy.test.zig"),
        .target = target,
        .optimize = optimize,
    });
    lsp_call_hierarchy_test_mod.addImport("ora_root", lib_mod);
    const lsp_call_hierarchy_tests = b.addTest(.{ .root_module = lsp_call_hierarchy_test_mod });
    test_step.dependOn(&b.addRunArtifact(lsp_call_hierarchy_tests).step);

    const lsp_call_hierarchy_prepare_test_mod = b.createModule(.{
        .root_source_file = b.path("src/lsp/call_hierarchy_prepare.test.zig"),
        .target = target,
        .optimize = optimize,
    });
    lsp_call_hierarchy_prepare_test_mod.addImport("ora_root", lib_mod);
    lsp_call_hierarchy_prepare_test_mod.addImport("lsp", b.dependency("lsp_kit", .{}).module("lsp"));
    const lsp_call_hierarchy_prepare_tests = b.addTest(.{ .root_module = lsp_call_hierarchy_prepare_test_mod });
    test_step.dependOn(&b.addRunArtifact(lsp_call_hierarchy_prepare_tests).step);

    const lsp_call_hierarchy_calls_test_mod = b.createModule(.{
        .root_source_file = b.path("src/lsp/call_hierarchy_calls.test.zig"),
        .target = target,
        .optimize = optimize,
    });
    lsp_call_hierarchy_calls_test_mod.addImport("ora_root", lib_mod);
    lsp_call_hierarchy_calls_test_mod.addImport("lsp", b.dependency("lsp_kit", .{}).module("lsp"));
    const lsp_call_hierarchy_calls_tests = b.addTest(.{ .root_module = lsp_call_hierarchy_calls_test_mod });
    test_step.dependOn(&b.addRunArtifact(lsp_call_hierarchy_calls_tests).step);

    const lsp_code_action_test_mod = b.createModule(.{
        .root_source_file = b.path("src/lsp/code_action.test.zig"),
        .target = target,
        .optimize = optimize,
    });
    lsp_code_action_test_mod.addImport("lsp", b.dependency("lsp_kit", .{}).module("lsp"));
    const lsp_code_action_tests = b.addTest(.{ .root_module = lsp_code_action_test_mod });
    test_step.dependOn(&b.addRunArtifact(lsp_code_action_tests).step);

    const lsp_code_lens_response_test_mod = b.createModule(.{
        .root_source_file = b.path("src/lsp/code_lens_response.test.zig"),
        .target = target,
        .optimize = optimize,
    });
    lsp_code_lens_response_test_mod.addImport("ora_root", lib_mod);
    lsp_code_lens_response_test_mod.addImport("lsp", b.dependency("lsp_kit", .{}).module("lsp"));
    const lsp_code_lens_response_tests = b.addTest(.{ .root_module = lsp_code_lens_response_test_mod });
    test_step.dependOn(&b.addRunArtifact(lsp_code_lens_response_tests).step);

    const lsp_completion_items_test_mod = b.createModule(.{
        .root_source_file = b.path("src/lsp/completion_items.test.zig"),
        .target = target,
        .optimize = optimize,
    });
    lsp_completion_items_test_mod.addImport("ora_root", lib_mod);
    lsp_completion_items_test_mod.addImport("lsp", b.dependency("lsp_kit", .{}).module("lsp"));
    const lsp_completion_items_tests = b.addTest(.{ .root_module = lsp_completion_items_test_mod });
    test_step.dependOn(&b.addRunArtifact(lsp_completion_items_tests).step);

    const lsp_definition_response_test_mod = b.createModule(.{
        .root_source_file = b.path("src/lsp/definition_response.test.zig"),
        .target = target,
        .optimize = optimize,
    });
    lsp_definition_response_test_mod.addImport("ora_root", lib_mod);
    lsp_definition_response_test_mod.addImport("lsp", b.dependency("lsp_kit", .{}).module("lsp"));
    const lsp_definition_response_tests = b.addTest(.{ .root_module = lsp_definition_response_test_mod });
    test_step.dependOn(&b.addRunArtifact(lsp_definition_response_tests).step);

    const lsp_diagnostics_response_test_mod = b.createModule(.{
        .root_source_file = b.path("src/lsp/diagnostics_response.test.zig"),
        .target = target,
        .optimize = optimize,
    });
    lsp_diagnostics_response_test_mod.addImport("ora_root", lib_mod);
    lsp_diagnostics_response_test_mod.addImport("lsp", b.dependency("lsp_kit", .{}).module("lsp"));
    const lsp_diagnostics_response_tests = b.addTest(.{ .root_module = lsp_diagnostics_response_test_mod });
    test_step.dependOn(&b.addRunArtifact(lsp_diagnostics_response_tests).step);

    const lsp_diagnostic_debounce_test_mod = b.createModule(.{
        .root_source_file = b.path("src/lsp/diagnostic_debounce.zig"),
        .target = target,
        .optimize = optimize,
    });
    const lsp_diagnostic_debounce_tests = b.addTest(.{ .root_module = lsp_diagnostic_debounce_test_mod });
    test_step.dependOn(&b.addRunArtifact(lsp_diagnostic_debounce_tests).step);

    const lsp_document_link_test_mod = b.createModule(.{
        .root_source_file = b.path("src/lsp/document_link.test.zig"),
        .target = target,
        .optimize = optimize,
    });
    lsp_document_link_test_mod.addImport("ora_root", lib_mod);
    lsp_document_link_test_mod.addImport("lsp", b.dependency("lsp_kit", .{}).module("lsp"));
    const lsp_document_link_tests = b.addTest(.{ .root_module = lsp_document_link_test_mod });
    test_step.dependOn(&b.addRunArtifact(lsp_document_link_tests).step);

    const lsp_document_symbol_test_mod = b.createModule(.{
        .root_source_file = b.path("src/lsp/document_symbol.test.zig"),
        .target = target,
        .optimize = optimize,
    });
    lsp_document_symbol_test_mod.addImport("ora_root", lib_mod);
    lsp_document_symbol_test_mod.addImport("lsp", b.dependency("lsp_kit", .{}).module("lsp"));
    const lsp_document_symbol_tests = b.addTest(.{ .root_module = lsp_document_symbol_test_mod });
    test_step.dependOn(&b.addRunArtifact(lsp_document_symbol_tests).step);

    const lsp_folding_ranges_response_test_mod = b.createModule(.{
        .root_source_file = b.path("src/lsp/folding_ranges_response.test.zig"),
        .target = target,
        .optimize = optimize,
    });
    lsp_folding_ranges_response_test_mod.addImport("ora_root", lib_mod);
    lsp_folding_ranges_response_test_mod.addImport("lsp", b.dependency("lsp_kit", .{}).module("lsp"));
    const lsp_folding_ranges_response_tests = b.addTest(.{ .root_module = lsp_folding_ranges_response_test_mod });
    test_step.dependOn(&b.addRunArtifact(lsp_folding_ranges_response_tests).step);

    const lsp_formatting_edits_test_mod = b.createModule(.{
        .root_source_file = b.path("src/lsp/formatting_edits.test.zig"),
        .target = target,
        .optimize = optimize,
    });
    lsp_formatting_edits_test_mod.addImport("ora_root", lib_mod);
    lsp_formatting_edits_test_mod.addImport("lsp", b.dependency("lsp_kit", .{}).module("lsp"));
    const lsp_formatting_edits_tests = b.addTest(.{ .root_module = lsp_formatting_edits_test_mod });
    test_step.dependOn(&b.addRunArtifact(lsp_formatting_edits_tests).step);

    const lsp_hover_response_test_mod = b.createModule(.{
        .root_source_file = b.path("src/lsp/hover_response.test.zig"),
        .target = target,
        .optimize = optimize,
    });
    lsp_hover_response_test_mod.addImport("ora_root", lib_mod);
    lsp_hover_response_test_mod.addImport("lsp", b.dependency("lsp_kit", .{}).module("lsp"));
    const lsp_hover_response_tests = b.addTest(.{ .root_module = lsp_hover_response_test_mod });
    test_step.dependOn(&b.addRunArtifact(lsp_hover_response_tests).step);

    const lsp_inlay_hint_response_test_mod = b.createModule(.{
        .root_source_file = b.path("src/lsp/inlay_hint_response.test.zig"),
        .target = target,
        .optimize = optimize,
    });
    lsp_inlay_hint_response_test_mod.addImport("ora_root", lib_mod);
    lsp_inlay_hint_response_test_mod.addImport("lsp", b.dependency("lsp_kit", .{}).module("lsp"));
    const lsp_inlay_hint_response_tests = b.addTest(.{ .root_module = lsp_inlay_hint_response_test_mod });
    test_step.dependOn(&b.addRunArtifact(lsp_inlay_hint_response_tests).step);

    const lsp_inlay_hints_test_mod = b.createModule(.{
        .root_source_file = b.path("src/lsp/inlay_hints.test.zig"),
        .target = target,
        .optimize = optimize,
    });
    lsp_inlay_hints_test_mod.addImport("ora_root", lib_mod);
    const lsp_inlay_hints_tests = b.addTest(.{ .root_module = lsp_inlay_hints_test_mod });
    test_step.dependOn(&b.addRunArtifact(lsp_inlay_hints_tests).step);

    const lsp_line_index_test_mod = b.createModule(.{
        .root_source_file = b.path("src/lsp/line_index.test.zig"),
        .target = target,
        .optimize = optimize,
    });
    lsp_line_index_test_mod.addImport("ora_root", lib_mod);
    const lsp_line_index_tests = b.addTest(.{ .root_module = lsp_line_index_test_mod });
    test_step.dependOn(&b.addRunArtifact(lsp_line_index_tests).step);

    const lsp_protocol_ranges_test_mod = b.createModule(.{
        .root_source_file = b.path("src/lsp/protocol_ranges.test.zig"),
        .target = target,
        .optimize = optimize,
    });
    lsp_protocol_ranges_test_mod.addImport("ora_root", lib_mod);
    lsp_protocol_ranges_test_mod.addImport("lsp", b.dependency("lsp_kit", .{}).module("lsp"));
    const lsp_protocol_ranges_tests = b.addTest(.{ .root_module = lsp_protocol_ranges_test_mod });
    test_step.dependOn(&b.addRunArtifact(lsp_protocol_ranges_tests).step);

    const lsp_references_response_test_mod = b.createModule(.{
        .root_source_file = b.path("src/lsp/references_response.test.zig"),
        .target = target,
        .optimize = optimize,
    });
    lsp_references_response_test_mod.addImport("ora_root", lib_mod);
    lsp_references_response_test_mod.addImport("lsp", b.dependency("lsp_kit", .{}).module("lsp"));
    const lsp_references_response_tests = b.addTest(.{ .root_module = lsp_references_response_test_mod });
    test_step.dependOn(&b.addRunArtifact(lsp_references_response_tests).step);

    const lsp_rename_response_test_mod = b.createModule(.{
        .root_source_file = b.path("src/lsp/rename_response.test.zig"),
        .target = target,
        .optimize = optimize,
    });
    lsp_rename_response_test_mod.addImport("ora_root", lib_mod);
    lsp_rename_response_test_mod.addImport("lsp", b.dependency("lsp_kit", .{}).module("lsp"));
    const lsp_rename_response_tests = b.addTest(.{ .root_module = lsp_rename_response_test_mod });
    test_step.dependOn(&b.addRunArtifact(lsp_rename_response_tests).step);

    const lsp_selection_range_test_mod = b.createModule(.{
        .root_source_file = b.path("src/lsp/selection_range.test.zig"),
        .target = target,
        .optimize = optimize,
    });
    lsp_selection_range_test_mod.addImport("ora_root", lib_mod);
    lsp_selection_range_test_mod.addImport("lsp", b.dependency("lsp_kit", .{}).module("lsp"));
    const lsp_selection_range_tests = b.addTest(.{ .root_module = lsp_selection_range_test_mod });
    test_step.dependOn(&b.addRunArtifact(lsp_selection_range_tests).step);

    const lsp_semantic_tokens_response_test_mod = b.createModule(.{
        .root_source_file = b.path("src/lsp/semantic_tokens_response.test.zig"),
        .target = target,
        .optimize = optimize,
    });
    lsp_semantic_tokens_response_test_mod.addImport("ora_root", lib_mod);
    lsp_semantic_tokens_response_test_mod.addImport("lsp", b.dependency("lsp_kit", .{}).module("lsp"));
    const lsp_semantic_tokens_response_tests = b.addTest(.{ .root_module = lsp_semantic_tokens_response_test_mod });
    test_step.dependOn(&b.addRunArtifact(lsp_semantic_tokens_response_tests).step);

    const lsp_uri_ranges_test_mod = b.createModule(.{
        .root_source_file = b.path("src/lsp/uri_ranges.test.zig"),
        .target = target,
        .optimize = optimize,
    });
    lsp_uri_ranges_test_mod.addImport("ora_root", lib_mod);
    lsp_uri_ranges_test_mod.addImport("lsp", b.dependency("lsp_kit", .{}).module("lsp"));
    const lsp_uri_ranges_tests = b.addTest(.{ .root_module = lsp_uri_ranges_test_mod });
    test_step.dependOn(&b.addRunArtifact(lsp_uri_ranges_tests).step);

    const lsp_workspace_discovery_test_mod = b.createModule(.{
        .root_source_file = b.path("src/lsp/workspace_discovery.test.zig"),
        .target = target,
        .optimize = optimize,
    });
    lsp_workspace_discovery_test_mod.addImport("ora_root", lib_mod);
    const lsp_workspace_discovery_tests = b.addTest(.{ .root_module = lsp_workspace_discovery_test_mod });
    test_step.dependOn(&b.addRunArtifact(lsp_workspace_discovery_tests).step);

    const lsp_workspace_index_test_mod = b.createModule(.{
        .root_source_file = b.path("src/lsp/workspace_index.test.zig"),
        .target = target,
        .optimize = optimize,
    });
    lsp_workspace_index_test_mod.addImport("ora_root", lib_mod);
    const lsp_workspace_index_tests = b.addTest(.{ .root_module = lsp_workspace_index_test_mod });
    test_step.dependOn(&b.addRunArtifact(lsp_workspace_index_tests).step);

    const lsp_workspace_symbol_response_test_mod = b.createModule(.{
        .root_source_file = b.path("src/lsp/workspace_symbol_response.test.zig"),
        .target = target,
        .optimize = optimize,
    });
    lsp_workspace_symbol_response_test_mod.addImport("ora_root", lib_mod);
    lsp_workspace_symbol_response_test_mod.addImport("lsp", b.dependency("lsp_kit", .{}).module("lsp"));
    const lsp_workspace_symbol_response_tests = b.addTest(.{ .root_module = lsp_workspace_symbol_response_test_mod });
    test_step.dependOn(&b.addRunArtifact(lsp_workspace_symbol_response_tests).step);

    const lsp_formatting_test_mod = b.createModule(.{
        .root_source_file = b.path("src/lsp/formatting.test.zig"),
        .target = target,
        .optimize = optimize,
    });
    lsp_formatting_test_mod.addImport("ora_lib", lib_mod);
    lsp_formatting_test_mod.addImport("ora_fmt", ora_fmt_mod);
    const lsp_formatting_tests = b.addTest(.{ .root_module = lsp_formatting_test_mod });
    test_step.dependOn(&b.addRunArtifact(lsp_formatting_tests).step);

    const compiler_test_mod = b.createModule(.{
        .root_source_file = b.path("src/compiler.test.zig"),
        .target = target,
        .optimize = optimize,
    });
    compiler_test_mod.addImport("ora_root", lib_mod);
    compiler_test_mod.addImport("ora_lib", lib_mod);
    compiler_test_mod.addImport("mlir_c_api", mlir_c_mod);
    compiler_test_mod.addImport("ora_z3_verification", z3_verification_mod);
    compiler_test_mod.addImport("ora_prepared_query_row", prepared_query_row_mod);
    compiler_test_mod.addImport("ora_types", ora_types_mod);
    compiler_test_mod.addImport("ora_lexer", ora_lexer_mod);
    compiler_test_mod.addImport("sinora", sinora_mod);
    const compiler_tests = b.addTest(.{
        .root_module = compiler_test_mod,
        .filters = compilerTestFilters(
            b,
            b.option([]const u8, "compiler-test-filter", "Filter compiler tests by test name substring"),
        ),
    });
    linkMlirLibraries(b, compiler_tests, mlir_step, ora_dialect_step, sir_dialect_step, target, native_sanitize);
    linkZ3Libraries(b, compiler_tests, z3_step, target);

    const compiler_tests_run = b.addRunArtifact(compiler_tests);
    compiler_tests_run.step.dependOn(b.getInstallStep());
    const evm_debug_probe_install_cmd = b.addSystemCommand(&[_][]const u8{
        "zig",
        "build",
        "install",
    });
    evm_debug_probe_install_cmd.setCwd(b.path("lib/evm"));
    compiler_tests_run.step.dependOn(&evm_debug_probe_install_cmd.step);
    test_step.dependOn(&compiler_tests_run.step);

    const test_compiler_step = b.step("test-compiler", "Run compiler core tests");
    test_compiler_step.dependOn(&compiler_tests_run.step);

    // zig build test-source-accounting
    // Pure kernel/adapters plus the real compiler concrete-discharge case.
    const source_accounting_test_mod = b.createModule(.{
        .root_source_file = b.path("src/source_accounting_tests.zig"),
        .target = target,
        .optimize = optimize,
    });
    source_accounting_test_mod.addImport("ora_lexer", ora_lexer_mod);
    source_accounting_test_mod.addImport("ora_types", ora_types_mod);
    source_accounting_test_mod.addImport("ora_refinements", ora_refinements_mod);
    source_accounting_test_mod.addImport("ora_prepared_query_row", prepared_query_row_mod);
    const source_accounting_tests = b.addTest(.{ .root_module = source_accounting_test_mod });
    const source_accounting_tests_run = b.addRunArtifact(source_accounting_tests);

    const source_accounting_repro_tests = b.addTest(.{
        .root_module = compiler_test_mod,
        .filters = &.{
            "source accounting accepts concretely checked folded countThree invariants",
            "production source-accounting pipeline accepts a runtime loop with actual verifier queries",
            "production source-accounting pipeline binds abandoned-fold call-boundary queries",
            "production source-accounting pipeline preserves guard and modifies evidence",
            "production source-accounting pipeline preserves impl contract and collapsed switch identities",
            "source-accounting syntax and spec-clause vocabularies are totality pins",
            "source-accounting templates distinguish contract and error exit uses",
            "source-accounting lifecycle",
            "verification-disabled binding is exclusive",
            "kernel registry describes the blocking source-accounting phases",
            "every executable kernel gate has an explicit audit-catalog identity",
        },
    });
    linkMlirLibraries(b, source_accounting_repro_tests, mlir_step, ora_dialect_step, sir_dialect_step, target, native_sanitize);
    linkZ3Libraries(b, source_accounting_repro_tests, z3_step, target);
    const source_accounting_repro_run = b.addRunArtifact(source_accounting_repro_tests);
    source_accounting_repro_run.step.dependOn(b.getInstallStep());
    source_accounting_repro_run.step.dependOn(&evm_debug_probe_install_cmd.step);

    const test_source_accounting_step = b.step("test-source-accounting", "Run source-formal accounting kernel, adapter, and reproduction tests");
    test_source_accounting_step.dependOn(&source_accounting_tests_run.step);
    test_source_accounting_step.dependOn(&source_accounting_repro_run.step);
    test_compiler_step.dependOn(&source_accounting_tests_run.step);
    test_compiler_step.dependOn(&source_accounting_repro_run.step);
    test_step.dependOn(&source_accounting_tests_run.step);
    test_step.dependOn(&source_accounting_repro_run.step);

    // ========================================================================
    // Per-module test targets (no MLIR/Z3 required)
    // ========================================================================

    // zig build test-types
    const test_types_step = b.step("test-types", "Run ora_types unit tests");
    const types_test_mod = b.addTest(.{ .root_module = ora_types_mod });
    test_types_step.dependOn(&b.addRunArtifact(types_test_mod).step);

    // zig build test-lexer
    const test_lexer_step = b.step("test-lexer", "Run lexer unit tests (no MLIR/Z3)");
    const lexer_standalone_tests = b.addTest(.{ .root_module = ora_lexer_mod });
    test_lexer_step.dependOn(&b.addRunArtifact(lexer_standalone_tests).step);
    test_lexer_step.dependOn(&b.addRunArtifact(lexer_tests).step);
    test_lexer_step.dependOn(&b.addRunArtifact(numbers_tests).step);
    test_lexer_step.dependOn(&b.addRunArtifact(strings_tests).step);
    test_lexer_step.dependOn(&b.addRunArtifact(identifiers_tests).step);

    // zig build test-lsp
    const test_lsp_step = b.step("test-lsp", "Run LSP frontend tests (no MLIR/Z3)");
    test_lsp_step.dependOn(&b.addRunArtifact(lsp_frontend_tests).step);
    test_lsp_step.dependOn(&b.addRunArtifact(lsp_workspace_tests).step);
    test_lsp_step.dependOn(&b.addRunArtifact(lsp_dependency_graph_tests).step);
    test_lsp_step.dependOn(&b.addRunArtifact(lsp_semantic_index_tests).step);
    test_lsp_step.dependOn(&b.addRunArtifact(lsp_text_edits_tests).step);
    test_lsp_step.dependOn(&b.addRunArtifact(lsp_hover_tests).step);
    test_lsp_step.dependOn(&b.addRunArtifact(lsp_definition_tests).step);
    test_lsp_step.dependOn(&b.addRunArtifact(lsp_references_tests).step);
    test_lsp_step.dependOn(&b.addRunArtifact(lsp_document_highlight_tests).step);
    test_lsp_step.dependOn(&b.addRunArtifact(lsp_rename_tests).step);
    test_lsp_step.dependOn(&b.addRunArtifact(lsp_completion_tests).step);
    test_lsp_step.dependOn(&b.addRunArtifact(lsp_signature_help_tests).step);
    test_lsp_step.dependOn(&b.addRunArtifact(lsp_semantic_tokens_tests).step);
    test_lsp_step.dependOn(&b.addRunArtifact(lsp_token_cache_tests).step);
    test_lsp_step.dependOn(&b.addRunArtifact(lsp_code_lens_tests).step);
    test_lsp_step.dependOn(&b.addRunArtifact(lsp_folding_tests).step);
    test_lsp_step.dependOn(&b.addRunArtifact(lsp_cache_stats_response_tests).step);
    test_lsp_step.dependOn(&b.addRunArtifact(lsp_allocation_stats_tests).step);
    test_lsp_step.dependOn(&b.addRunArtifact(lsp_signature_help_response_tests).step);
    test_lsp_step.dependOn(&b.addRunArtifact(lsp_builtin_docs_tests).step);
    test_lsp_step.dependOn(&b.addRunArtifact(lsp_keyword_docs_tests).step);
    test_lsp_step.dependOn(&b.addRunArtifact(lsp_std_docs_tests).step);
    test_lsp_step.dependOn(&b.addRunArtifact(lsp_call_hierarchy_tests).step);
    test_lsp_step.dependOn(&b.addRunArtifact(lsp_call_hierarchy_prepare_tests).step);
    test_lsp_step.dependOn(&b.addRunArtifact(lsp_call_hierarchy_calls_tests).step);
    test_lsp_step.dependOn(&b.addRunArtifact(lsp_code_action_tests).step);
    test_lsp_step.dependOn(&b.addRunArtifact(lsp_code_lens_response_tests).step);
    test_lsp_step.dependOn(&b.addRunArtifact(lsp_completion_items_tests).step);
    test_lsp_step.dependOn(&b.addRunArtifact(lsp_definition_response_tests).step);
    test_lsp_step.dependOn(&b.addRunArtifact(lsp_diagnostics_response_tests).step);
    test_lsp_step.dependOn(&b.addRunArtifact(lsp_diagnostic_debounce_tests).step);
    test_lsp_step.dependOn(&b.addRunArtifact(lsp_document_link_tests).step);
    test_lsp_step.dependOn(&b.addRunArtifact(lsp_document_symbol_tests).step);
    test_lsp_step.dependOn(&b.addRunArtifact(lsp_folding_ranges_response_tests).step);
    test_lsp_step.dependOn(&b.addRunArtifact(lsp_formatting_edits_tests).step);
    test_lsp_step.dependOn(&b.addRunArtifact(lsp_hover_response_tests).step);
    test_lsp_step.dependOn(&b.addRunArtifact(lsp_inlay_hint_response_tests).step);
    test_lsp_step.dependOn(&b.addRunArtifact(lsp_inlay_hints_tests).step);
    test_lsp_step.dependOn(&b.addRunArtifact(lsp_line_index_tests).step);
    test_lsp_step.dependOn(&b.addRunArtifact(lsp_protocol_ranges_tests).step);
    test_lsp_step.dependOn(&b.addRunArtifact(lsp_references_response_tests).step);
    test_lsp_step.dependOn(&b.addRunArtifact(lsp_rename_response_tests).step);
    test_lsp_step.dependOn(&b.addRunArtifact(lsp_selection_range_tests).step);
    test_lsp_step.dependOn(&b.addRunArtifact(lsp_semantic_tokens_response_tests).step);
    test_lsp_step.dependOn(&b.addRunArtifact(lsp_uri_ranges_tests).step);
    test_lsp_step.dependOn(&b.addRunArtifact(lsp_workspace_discovery_tests).step);
    test_lsp_step.dependOn(&b.addRunArtifact(lsp_workspace_index_tests).step);
    test_lsp_step.dependOn(&b.addRunArtifact(lsp_workspace_symbol_response_tests).step);
    test_lsp_step.dependOn(&b.addRunArtifact(lsp_formatting_tests).step);

    const lsp_smoke_cmd = b.addSystemCommand(&[_][]const u8{
        "python3",
        "scripts/lsp-jsonrpc-smoke.py",
    });
    lsp_smoke_cmd.addArtifactArg(lsp_exe);
    const lsp_smoke_step = b.step("lsp-smoke", "Run LSP JSON-RPC smoke test");
    lsp_smoke_step.dependOn(&lsp_smoke_cmd.step);

    const lsp_build_mode = optimizeModeName(optimize);

    const lsp_bench_cmd = b.addSystemCommand(&[_][]const u8{
        "python3",
        "scripts/lsp-jsonrpc-benchmark.py",
        "--profile",
        "quick",
        "--build-mode",
        lsp_build_mode,
        "--strict-future-gates",
    });
    lsp_bench_cmd.addArtifactArg(lsp_exe);
    const lsp_bench_step = b.step("lsp-bench", "Run LSP JSON-RPC benchmark");
    lsp_bench_step.dependOn(&lsp_bench_cmd.step);

    const lsp_bench_release_cmd = b.addSystemCommand(&[_][]const u8{
        "python3",
        "scripts/lsp-jsonrpc-benchmark.py",
        "--profile",
        "release",
        "--build-mode",
        optimizeModeName(lsp_release_optimize),
        "--require-build-mode",
        "ReleaseFast",
        "--strict-future-gates",
    });
    lsp_bench_release_cmd.addArtifactArg(lsp_release_exe);
    const lsp_bench_release_step = b.step("lsp-bench-release", "Run LSP JSON-RPC benchmark in ReleaseFast");
    lsp_bench_release_step.dependOn(&lsp_bench_release_cmd.step);

    // zig build check-verifier-introspection
    const verifier_introspection_cmd = b.addSystemCommand(&[_][]const u8{
        "sh",
        "scripts/check-verifier-introspection.sh",
    });
    const check_verifier_introspection_step = b.step("check-verifier-introspection", "Run verifier self-introspection static checks");
    check_verifier_introspection_step.dependOn(&verifier_introspection_cmd.step);

    // zig build check-refinement-registry-sync
    const refinement_registry_sync_cmd = b.addSystemCommand(&[_][]const u8{
        "sh",
        "scripts/check-refinement-registry-sync.sh",
    });
    const check_refinement_registry_sync_step = b.step("check-refinement-registry-sync", "Run refinement registry/docs sync checks");
    check_refinement_registry_sync_step.dependOn(&refinement_registry_sync_cmd.step);

    // zig build check-formal-sync
    // Forward -Dskip-mlir so the script's child `zig build` emitter steps link
    // the prebuilt vendor/mlir instead of re-running the vendored-LLVM CMake
    // build (which fails on runners without the LLVM source tree).
    const formal_sync_cmd = if (skip_mlir_build)
        b.addSystemCommand(&[_][]const u8{
            "bash", "scripts/check-formal-sync.sh", "--skip-mlir",
        })
    else
        b.addSystemCommand(&[_][]const u8{
            "bash", "scripts/check-formal-sync.sh",
        });
    const check_formal_sync_step = b.step("check-formal-sync", "Regenerate formal snapshots and run Lean verification checks");
    check_formal_sync_step.dependOn(&formal_sync_cmd.step);

    // zig build measure-loop-census
    const loop_census_report_path = "zig-out/loop-census/report.json";
    const loop_census_cmd = b.addSystemCommand(&[_][]const u8{
        "python3",
        "scripts/measure-loop-census.py",
        "--tool",
    });
    loop_census_cmd.addArtifactArg(loop_census_exe);
    loop_census_cmd.addArg("--compiler");
    loop_census_cmd.addArtifactArg(exe);
    loop_census_cmd.addArgs(&.{
        "--corpus-root",
        "ora-example",
        "--json-out",
        loop_census_report_path,
        "--activation-out-dir",
        "zig-out/loop-census/formal-activation",
        "ora-example",
    });
    const loop_census_step = b.step("measure-loop-census", "Measure all source loops and prepared loop queries");
    loop_census_step.dependOn(&loop_census_cmd.step);

    // zig build check-canonical-z3-required
    const canonical_z3_required_out_dir = b.fmt("/tmp/ora-canonical-z3-required-gate-{d}", .{std.posix.system.getpid()});
    const canonical_z3_required_json = b.fmt("{s}/report.json", .{canonical_z3_required_out_dir});
    const canonical_z3_required_cmd = b.addSystemCommand(&[_][]const u8{
        "python3",
        "scripts/measure-canonical-z3-corpus.py",
        "--out-dir",
        canonical_z3_required_out_dir,
        "--json-out",
        canonical_z3_required_json,
        "--fail-required",
        "--min-required",
        "10",
        "--ora",
    });
    canonical_z3_required_cmd.addArtifactArg(exe);
    canonical_z3_required_cmd.addArg("ora-example/formal");
    const check_canonical_z3_required_step = b.step("check-canonical-z3-required", "Gate required canonical SMT hash rows on the formal corpus");
    check_canonical_z3_required_step.dependOn(&canonical_z3_required_cmd.step);

    // zig build check-lock-guarding
    const lock_guarding_cmd = b.addSystemCommand(&[_][]const u8{
        "sh",
        "scripts/check-lock-guarding.sh",
    });
    const check_lock_guarding_step = b.step("check-lock-guarding", "Run lock guard insertion static checks");
    check_lock_guarding_step.dependOn(&lock_guarding_cmd.step);

    // zig build check-abi-layout-ownership
    const abi_layout_ownership_cmd = b.addSystemCommand(&[_][]const u8{
        "sh",
        "scripts/check-abi-layout-ownership.sh",
    });
    const check_abi_layout_ownership_step = b.step("check-abi-layout-ownership", "Run ABI layout source-of-truth static checks");
    check_abi_layout_ownership_step.dependOn(&abi_layout_ownership_cmd.step);
    test_step.dependOn(&abi_layout_ownership_cmd.step);

    // zig build check-no-duplicate-fixed-bytes-parsers
    const no_duplicate_fixed_bytes_parsers_cmd = b.addSystemCommand(&[_][]const u8{
        "sh",
        "scripts/check-no-duplicate-fixed-bytes-parsers.sh",
    });
    const check_no_duplicate_fixed_bytes_parsers_step = b.step("check-no-duplicate-fixed-bytes-parsers", "Run fixed-bytes parser ownership static checks");
    check_no_duplicate_fixed_bytes_parsers_step.dependOn(&no_duplicate_fixed_bytes_parsers_cmd.step);
    test_step.dependOn(&no_duplicate_fixed_bytes_parsers_cmd.step);

    // zig build check-no-width-defaults
    const no_width_defaults_cmd = b.addSystemCommand(&[_][]const u8{
        "sh",
        "scripts/check-no-width-defaults.sh",
    });
    const check_no_width_defaults_step = b.step("check-no-width-defaults", "Run integer width/signedness default static checks");
    check_no_width_defaults_step.dependOn(&no_width_defaults_cmd.step);
    test_step.dependOn(&no_width_defaults_cmd.step);

    // zig build check-no-hir-op-null-fallbacks
    const no_hir_op_null_fallbacks_cmd = b.addSystemCommand(&[_][]const u8{
        "sh",
        "scripts/check-no-hir-op-null-fallbacks.sh",
    });
    const check_no_hir_op_null_fallbacks_step = b.step("check-no-hir-op-null-fallbacks", "Run HIR op-creation fail-closed static checks");
    check_no_hir_op_null_fallbacks_step.dependOn(&no_hir_op_null_fallbacks_cmd.step);
    test_step.dependOn(&no_hir_op_null_fallbacks_cmd.step);

    // zig build check-no-scattered-process-exit
    const no_scattered_process_exit_cmd = b.addSystemCommand(&[_][]const u8{
        "sh",
        "scripts/check-no-scattered-process-exit.sh",
    });
    const check_no_scattered_process_exit_step = b.step("check-no-scattered-process-exit", "Run process-exit boundary static checks");
    check_no_scattered_process_exit_step.dependOn(&no_scattered_process_exit_cmd.step);
    test_step.dependOn(&no_scattered_process_exit_cmd.step);

    // zig build check-query-view-ownership
    const query_view_ownership_cmd = b.addSystemCommand(&[_][]const u8{
        "sh",
        "scripts/check-query-view-ownership.sh",
    });
    const check_query_view_ownership_step = b.step("check-query-view-ownership", "Run compiler query-view ownership static checks");
    check_query_view_ownership_step.dependOn(&query_view_ownership_cmd.step);
    test_step.dependOn(&query_view_ownership_cmd.step);

    // zig build check-oratosir-coverage
    const oratosir_coverage_cmd = b.addSystemCommand(&[_][]const u8{
        "python3",
        "scripts/check-oratosir-coverage.py",
        "tests/oratosir_debloat_coverage.json",
    });
    const check_oratosir_coverage_step = b.step("check-oratosir-coverage", "Validate the OraToSIR coverage manifest");
    check_oratosir_coverage_step.dependOn(&oratosir_coverage_cmd.step);
    test_step.dependOn(&oratosir_coverage_cmd.step);

    // zig build check-resource-mutation-tripwires
    const resource_mutation_tripwires_cmd = b.addSystemCommand(&[_][]const u8{
        "python3",
        "scripts/check-resource-mutation-tripwires.py",
    });
    const check_resource_mutation_tripwires_step = b.step("check-resource-mutation-tripwires", "Validate resource lowering mutation tripwires");
    check_resource_mutation_tripwires_step.dependOn(&resource_mutation_tripwires_cmd.step);
    test_step.dependOn(&resource_mutation_tripwires_cmd.step);

    // zig build check-feature-execution-coverage
    const feature_coverage_cmd = b.addSystemCommand(&[_][]const u8{
        "python3",
        "scripts/check-feature-execution-coverage.py",
        "tests/conformance/feature_coverage.json",
    });
    const check_feature_coverage_step = b.step("check-feature-execution-coverage", "Validate the feature-execution coverage manifest");
    check_feature_coverage_step.dependOn(&feature_coverage_cmd.step);
    test_step.dependOn(&feature_coverage_cmd.step);

    // zig build check-negative-corpus
    const negative_corpus_cmd = b.addSystemCommand(&[_][]const u8{
        "python3",
        "scripts/check-negative-corpus.py",
    });
    negative_corpus_cmd.step.dependOn(b.getInstallStep());
    const check_negative_corpus_step = b.step("check-negative-corpus", "Run the negative expected-diagnostic corpus");
    check_negative_corpus_step.dependOn(&negative_corpus_cmd.step);

    // zig build check-findings-ledger — surfaces known defects the green gate masks
    const findings_ledger_cmd = b.addSystemCommand(&[_][]const u8{
        "python3",
        "scripts/check-findings-ledger.py",
    });
    const check_findings_ledger_step = b.step("check-findings-ledger", "Validate the findings ledger and report masked known defects");
    check_findings_ledger_step.dependOn(&findings_ledger_cmd.step);

    // zig build check-verifier-mutations — bounded deterministic verifier soundness
    const verifier_mutations_cmd = b.addSystemCommand(&[_][]const u8{
        "python3",
        "scripts/verify_mutations.py",
        "--compiler",
        "./zig-out/bin/ora",
        "--timeout",
        "60",
    });
    verifier_mutations_cmd.step.dependOn(b.getInstallStep());
    const check_verifier_mutations_step = b.step("check-verifier-mutations", "Run the bounded verifier soundness mutation set");
    check_verifier_mutations_step.dependOn(&verifier_mutations_cmd.step);

    // zig build check-mlir-ora
    const mlir_ora_checks_cmd = b.addSystemCommand(&[_][]const u8{
        "bash",
        "scripts/run-mlir-checks.sh",
    });
    mlir_ora_checks_cmd.step.dependOn(b.getInstallStep());
    const check_mlir_ora_step = b.step("check-mlir-ora", "Run Ora MLIR FileCheck snapshots");
    check_mlir_ora_step.dependOn(&mlir_ora_checks_cmd.step);

    // zig build check-mlir-sir
    const mlir_sir_checks_cmd = b.addSystemCommand(&[_][]const u8{
        "bash",
        "scripts/run-mlir-checks-sir.sh",
    });
    mlir_sir_checks_cmd.step.dependOn(b.getInstallStep());
    const check_mlir_sir_step = b.step("check-mlir-sir", "Run SIR MLIR FileCheck snapshots");
    check_mlir_sir_step.dependOn(&mlir_sir_checks_cmd.step);

    // zig build check-sir-text
    const sir_text_checks_cmd = b.addSystemCommand(&[_][]const u8{
        "bash",
        "scripts/run-sir-text-checks.sh",
    });
    sir_text_checks_cmd.step.dependOn(b.getInstallStep());
    const check_sir_text_step = b.step("check-sir-text", "Run SIR text FileCheck snapshots");
    check_sir_text_step.dependOn(&sir_text_checks_cmd.step);

    // zig build gate-oratosir-debloat
    const oratosir_debloat_gate_step = b.step("gate-oratosir-debloat", "Run the OraToSIR regression gate");
    oratosir_debloat_gate_step.dependOn(check_oratosir_coverage_step);
    oratosir_debloat_gate_step.dependOn(check_mlir_ora_step);
    oratosir_debloat_gate_step.dependOn(check_mlir_sir_step);
    oratosir_debloat_gate_step.dependOn(check_sir_text_step);
    oratosir_debloat_gate_step.dependOn(test_conformance_step);
    oratosir_debloat_gate_step.dependOn(test_evm_step);

    // zig build check-sir-shift-operand-order
    const sir_shift_operand_order_cmd = b.addSystemCommand(&[_][]const u8{
        "sh",
        "scripts/check-sir-shift-operand-order.sh",
    });
    const check_sir_shift_operand_order_step = b.step("check-sir-shift-operand-order", "Run SIR shift operand-order static checks");
    check_sir_shift_operand_order_step.dependOn(&sir_shift_operand_order_cmd.step);
    test_step.dependOn(&sir_shift_operand_order_cmd.step);

    // zig build check-smt-modifies-corpus
    const smt_modifies_corpus_cmd = b.addSystemCommand(&[_][]const u8{
        "sh",
        "scripts/check-smt-modifies-corpus.sh",
    });
    const check_smt_modifies_corpus_step = b.step("check-smt-modifies-corpus", "Run SMT modifies corpus checks");
    smt_modifies_corpus_cmd.step.dependOn(b.getInstallStep());
    check_smt_modifies_corpus_step.dependOn(&smt_modifies_corpus_cmd.step);

    // zig build check-smt-resource-corpus
    const smt_resource_corpus_cmd = b.addSystemCommand(&[_][]const u8{
        "sh",
        "scripts/check-smt-resource-corpus.sh",
    });
    const check_smt_resource_corpus_step = b.step("check-smt-resource-corpus", "Run SMT resource corpus checks");
    smt_resource_corpus_cmd.step.dependOn(b.getInstallStep());
    check_smt_resource_corpus_step.dependOn(&smt_resource_corpus_cmd.step);

    // zig build gate — the full pre-push bar; every commit must pass this on the committed state.
    const gate_step = b.step("gate", "Run the full pre-push bar (test + OraToSIR gate + Ora MLIR checks + SMT corpus + LSP smoke)");
    gate_step.dependOn(test_step);
    gate_step.dependOn(check_formal_sync_step);
    gate_step.dependOn(check_canonical_z3_required_step);
    gate_step.dependOn(oratosir_debloat_gate_step);
    gate_step.dependOn(check_resource_mutation_tripwires_step);
    gate_step.dependOn(check_mlir_ora_step);
    // Bytecode-size baseline: without this in the bar the committed baseline
    // rots silently (it once drifted for 78 commits unseen).
    gate_step.dependOn(check_conformance_bytecode_size_step);
    gate_step.dependOn(check_negative_corpus_step);
    gate_step.dependOn(check_no_width_defaults_step);
    gate_step.dependOn(check_no_hir_op_null_fallbacks_step);
    gate_step.dependOn(check_no_scattered_process_exit_step);
    gate_step.dependOn(check_findings_ledger_step);
    gate_step.dependOn(check_verifier_mutations_step);
    gate_step.dependOn(check_smt_modifies_corpus_step);
    gate_step.dependOn(check_smt_resource_corpus_step);
    gate_step.dependOn(lsp_smoke_step);
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
    const res = std.process.run(allocator, b.graph.io, .{
        .argv = &[_][]const u8{ "./zig-out/bin/lexer_test_suite", "--verbose" },
        .cwd = .{ .path = "." },
    }) catch |err| {
        std.log.err("Failed to run lexer_test_suite: {}", .{err});
        return err;
    };
    switch (res.term) {
        .exited => |code| {
            if (code != 0) {
                std.log.err("lexer_test_suite failed with exit code {}", .{code});
                std.log.err("stderr: {s}", .{res.stderr});
                return error.LexerSuiteFailed;
            }
        },
        .signal => |sig| {
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

fn runInheritedProcess(b: *std.Build, argv: []const []const u8, cwd: []const u8) !std.process.Child.Term {
    var child = try std.process.spawn(b.graph.io, .{
        .argv = argv,
        .cwd = .{ .path = cwd },
        .environ_map = &b.graph.environ_map,
        .stdin = .inherit,
        .stdout = .inherit,
        .stderr = .inherit,
    });
    return try child.wait(b.graph.io);
}

/// Implementation of CMake build for MLIR libraries
fn buildMlirLibrariesImpl(step: *std.Build.Step, options: std.Build.Step.MakeOptions) anyerror!void {
    _ = options;

    const b = step.owner;
    const allocator = b.allocator;

    const root = b.build_root.handle;
    const build_root = b.build_root.path orelse ".";

    // skip if MLIR C API library is already installed
    const mlir_c_lib = if (@import("builtin").os.tag == .macos) "vendor/mlir/lib/libMLIR-C.dylib" else "vendor/mlir/lib/libMLIR-C.so";
    if (root.access(b.graph.io, mlir_c_lib, .{}) catch null) |_| {
        std.log.info("MLIR libraries already installed, skipping build (delete vendor/mlir to force rebuild)", .{});
        return;
    }

    // CMake is configured against llvm-project/llvm, not just the checkout root.
    _ = root.openDir(b.graph.io, "vendor/llvm-project/llvm", .{ .iterate = false }) catch {
        std.log.err("Missing LLVM source tree: vendor/llvm-project/llvm", .{});
        std.log.err("Build root: {s}", .{build_root});
        std.log.err("Run: cd {s} && ./setup.sh --skip-deps --skip-build", .{build_root});
        std.log.err("Or manually fetch the pinned source:", .{});
        std.log.err("  cd {s}", .{build_root});
        std.log.err("  git init vendor/llvm-project", .{});
        std.log.err("  git -C vendor/llvm-project remote add origin {s}", .{LLVM_REPO_URL});
        std.log.err("  git -C vendor/llvm-project fetch --depth=1 origin {s}", .{LLVM_COMMIT});
        std.log.err("  git -C vendor/llvm-project checkout --detach FETCH_HEAD", .{});
        return error.SubmoduleMissing;
    };

    // create build and install directories
    const build_dir = "vendor/llvm-project/build-mlir";
    try root.createDirPath(b.graph.io, build_dir);

    const install_prefix = "vendor/mlir";
    try root.createDirPath(b.graph.io, install_prefix);

    const builtin = @import("builtin");
    var cmake_args = std.array_list.Managed([]const u8).init(allocator);
    defer cmake_args.deinit();

    // prefer Ninja generator when available for faster, more parallel builds
    var use_ninja: bool = false;
    {
        const probe = std.process.run(allocator, b.graph.io, .{ .argv = &[_][]const u8{ "ninja", "--version" }, .cwd = .{ .path = "." } }) catch null;
        if (probe) |res| {
            switch (res.term) {
                .exited => |code| {
                    if (code == 0) use_ninja = true;
                },
                else => {},
            }
        }
        if (!use_ninja) {
            const probe_alt = std.process.run(allocator, b.graph.io, .{ .argv = &[_][]const u8{ "ninja-build", "--version" }, .cwd = .{ .path = "." } }) catch null;
            if (probe_alt) |res2| {
                switch (res2.term) {
                    .exited => |code| {
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
        "-DCMAKE_POSITION_INDEPENDENT_CODE=ON",
        "-DBUILD_SHARED_LIBS=ON",
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
        // Keep tablegen utilities but avoid linking the full LLVM/MLIR tool suite.
        // This significantly lowers peak memory usage in constrained builders.
        "-DLLVM_BUILD_TOOLS=OFF",
        "-DMLIR_BUILD_TOOLS=OFF",
        "-DLLVM_TOOL_LTO_BUILD=OFF",
        "-DLLVM_TOOL_LLVM_LTO_BUILD=OFF",
        "-DLLVM_TOOL_LLVM_LTO2_BUILD=OFF",
        "-DMLIR_BUILD_MLIR_C_DYLIB=ON",
        b.fmt("-DCMAKE_INSTALL_PREFIX={s}", .{install_prefix}),
    });

    if (builtin.os.tag == .linux) {
        try cmake_args.append("-DCMAKE_CXX_FLAGS=-stdlib=libc++");
        try cmake_args.append("-DCMAKE_EXE_LINKER_FLAGS=-stdlib=libc++ -lc++abi");
        try cmake_args.append("-DCMAKE_SHARED_LINKER_FLAGS=-stdlib=libc++ -lc++abi");
        try cmake_args.append("-DCMAKE_MODULE_LINKER_FLAGS=-stdlib=libc++ -lc++abi");
        try cmake_args.append("-DCMAKE_CXX_COMPILER=clang++");
        try cmake_args.append("-DCMAKE_C_COMPILER=clang");
    } else if (builtin.os.tag == .macos) {
        try cmake_args.append("-DCMAKE_CXX_FLAGS=-stdlib=libc++");

        // fix SDK path issue after macOS/Xcode update
        // use xcrun to get the actual SDK path and set it explicitly
        const sdk_path_result = std.process.run(allocator, b.graph.io, .{
            .argv = &[_][]const u8{ "xcrun", "--show-sdk-path" },
            .cwd = .{ .path = "." },
        }) catch null;
        if (sdk_path_result) |result| {
            if (result.term == .exited and result.term.exited == 0) {
                const sdk_path = std.mem.trim(u8, result.stdout, " \n\r\t");
                if (sdk_path.len > 0) {
                    const sysroot_flag = try std.fmt.allocPrint(allocator, "-DCMAKE_OSX_SYSROOT={s}", .{sdk_path});
                    defer allocator.free(sysroot_flag);
                    try cmake_args.append(sysroot_flag);
                    std.log.info("Setting MLIR CMAKE_OSX_SYSROOT={s}", .{sdk_path});
                }
            }
        }

        if (b.graph.environ_map.get("ORA_CMAKE_OSX_ARCH")) |arch| {
            const flag = b.fmt("-DCMAKE_OSX_ARCHITECTURES={s}", .{arch});
            try cmake_args.append(flag);
            std.log.info("Using CMAKE_OSX_ARCHITECTURES={s}", .{arch});
        }
    } else if (builtin.os.tag == .windows) {
        try cmake_args.append("-DCMAKE_CXX_FLAGS=/std:c++20");
    }

    const cfg_term = runInheritedProcess(b, cmake_args.items, build_root) catch |err| {
        std.log.err("Failed to configure MLIR CMake: {}", .{err});
        return err;
    };
    switch (cfg_term) {
        .exited => |code| if (code != 0) {
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
    const build_term = runInheritedProcess(b, &build_args, build_root) catch |err| {
        std.log.err("Failed to build MLIR with CMake: {}", .{err});
        return err;
    };
    switch (build_term) {
        .exited => |code| if (code != 0) {
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
fn buildOraDialectLibrary(
    b: *std.Build,
    mlir_step: *std.Build.Step,
    sir_dialect_step: *std.Build.Step,
    target: std.Build.ResolvedTarget,
    optimize: std.builtin.OptimizeMode,
    sanitizer: NativeSanitizer,
) *std.Build.Step {
    _ = target;
    _ = optimize;

    const native_step = b.allocator.create(NativeCMakeStep) catch @panic("OOM");
    native_step.* = .{
        .step = std.Build.Step.init(.{
            .id = .custom,
            .name = if (sanitizer.enabled()) b.fmt("cmake-build-ora-dialect-{s}", .{sanitizer.suffix()}) else "cmake-build-ora-dialect",
            .owner = b,
            .makeFn = buildOraDialectLibraryImpl,
        }),
        .sanitizer = sanitizer,
    };
    const step = &native_step.step;
    step.dependOn(mlir_step);
    step.dependOn(sir_dialect_step); // Ora dialect needs SIR headers
    return step;
}

/// Implementation of CMake build for Ora dialect library
fn buildOraDialectLibraryImpl(step: *std.Build.Step, options: std.Build.Step.MakeOptions) anyerror!void {
    _ = options;

    const b = step.owner;
    const allocator = b.allocator;
    const root = b.build_root.handle;
    const build_root = b.build_root.path orelse ".";
    const native_step: *NativeCMakeStep = @fieldParentPtr("step", step);
    const sanitizer = native_step.sanitizer;

    const build_dir = nativeDialectBuildDir(b, "vendor/ora-dialect-build", sanitizer);
    try root.createDirPath(b.graph.io, build_dir);

    const install_prefix = nativeMlirInstallPrefix(b, sanitizer);
    const install_prefix_abs = b.fmt("{s}/{s}", .{ build_root, install_prefix });
    const mlir_dir = b.fmt("{s}/vendor/mlir/lib/cmake/mlir", .{build_root});

    var cmake_args = std.array_list.Managed([]const u8).init(allocator);
    defer cmake_args.deinit();

    // prefer Ninja generator when available
    var use_ninja: bool = false;
    {
        const probe = std.process.run(allocator, b.graph.io, .{ .argv = &[_][]const u8{ "ninja", "--version" }, .cwd = .{ .path = "." } }) catch null;
        if (probe) |res| {
            switch (res.term) {
                .exited => |code| {
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
        "-DBUILD_SHARED_LIBS=ON",
        b.fmt("-DMLIR_DIR={s}", .{mlir_dir}),
        b.fmt("-DCMAKE_INSTALL_PREFIX={s}", .{install_prefix_abs}),
    });

    try appendNativeCmakeToolchainFlags(&cmake_args, b, sanitizer);

    const cfg_term = runInheritedProcess(b, cmake_args.items, build_root) catch |err| {
        std.log.err("Failed to configure Ora dialect CMake: {}", .{err});
        return err;
    };
    switch (cfg_term) {
        .exited => |code| if (code != 0) {
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
    const build_term = runInheritedProcess(b, &build_args, build_root) catch |err| {
        std.log.err("Failed to build Ora dialect with CMake: {}", .{err});
        return err;
    };
    switch (build_term) {
        .exited => |code| if (code != 0) {
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
fn buildSIRDialectLibrary(
    b: *std.Build,
    mlir_step: *std.Build.Step,
    target: std.Build.ResolvedTarget,
    optimize: std.builtin.OptimizeMode,
    sanitizer: NativeSanitizer,
) *std.Build.Step {
    _ = target;
    _ = optimize;

    const native_step = b.allocator.create(NativeCMakeStep) catch @panic("OOM");
    native_step.* = .{
        .step = std.Build.Step.init(.{
            .id = .custom,
            .name = if (sanitizer.enabled()) b.fmt("cmake-build-sir-dialect-{s}", .{sanitizer.suffix()}) else "cmake-build-sir-dialect",
            .owner = b,
            .makeFn = buildSIRDialectLibraryImpl,
        }),
        .sanitizer = sanitizer,
    };
    const step = &native_step.step;
    step.dependOn(mlir_step);
    return step;
}

/// Implementation of CMake build for SIR dialect library
fn buildSIRDialectLibraryImpl(step: *std.Build.Step, options: std.Build.Step.MakeOptions) anyerror!void {
    _ = options;

    const b = step.owner;
    const allocator = b.allocator;
    const root = b.build_root.handle;
    const build_root = b.build_root.path orelse ".";
    const native_step: *NativeCMakeStep = @fieldParentPtr("step", step);
    const sanitizer = native_step.sanitizer;

    const build_dir = nativeDialectBuildDir(b, "vendor/sir-dialect-build", sanitizer);
    try root.createDirPath(b.graph.io, build_dir);

    const install_prefix = nativeMlirInstallPrefix(b, sanitizer);
    const install_prefix_abs = b.fmt("{s}/{s}", .{ build_root, install_prefix });
    const mlir_dir = b.fmt("{s}/vendor/mlir/lib/cmake/mlir", .{build_root});

    var cmake_args = std.array_list.Managed([]const u8).init(allocator);
    defer cmake_args.deinit();

    // prefer Ninja generator when available
    var use_ninja: bool = false;
    {
        const probe = std.process.run(allocator, b.graph.io, .{ .argv = &[_][]const u8{ "ninja", "--version" }, .cwd = .{ .path = "." } }) catch null;
        if (probe) |res| {
            switch (res.term) {
                .exited => |code| {
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
        "-DBUILD_SHARED_LIBS=ON",
        b.fmt("-DMLIR_DIR={s}", .{mlir_dir}),
        b.fmt("-DCMAKE_INSTALL_PREFIX={s}", .{install_prefix_abs}),
    });

    try appendNativeCmakeToolchainFlags(&cmake_args, b, sanitizer);

    const cfg_term = runInheritedProcess(b, cmake_args.items, build_root) catch |err| {
        std.log.err("Failed to configure SIR dialect CMake: {}", .{err});
        return err;
    };
    switch (cfg_term) {
        .exited => |code| if (code != 0) {
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
    const build_term = runInheritedProcess(b, &build_args, build_root) catch |err| {
        std.log.err("Failed to build SIR dialect with CMake: {}", .{err});
        return err;
    };
    switch (build_term) {
        .exited => |code| if (code != 0) {
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
    sanitizer: NativeSanitizer,
) void {
    // depend on MLIR build and dialect builds when requested
    if (mlir_step) |step| exe.step.dependOn(step);
    if (ora_dialect_step) |step| exe.step.dependOn(step);
    if (sir_dialect_step) |step| exe.step.dependOn(step);

    const include_path = b.path("vendor/mlir/include");
    const lib_path = b.path("vendor/mlir/lib");
    const native_prefix = nativeMlirInstallPrefix(b, sanitizer);
    const native_include_path = b.path(b.fmt("{s}/include", .{native_prefix}));
    const native_lib_path = b.path(b.fmt("{s}/lib", .{native_prefix}));
    const ora_dialect_include_path = b.path("src/mlir/ora/include");
    const sir_dialect_include_path = b.path("src/mlir/IR/include");

    if (sanitizer.enabled()) {
        exe.root_module.addIncludePath(native_include_path);
        exe.root_module.addLibraryPath(native_lib_path);
        exe.root_module.addRPath(native_lib_path);
    }
    exe.root_module.addIncludePath(include_path);
    exe.root_module.addIncludePath(ora_dialect_include_path);
    exe.root_module.addIncludePath(sir_dialect_include_path);
    exe.root_module.addLibraryPath(lib_path);
    exe.root_module.addRPath(lib_path);

    // Link only top-level C API/dialect libraries.
    // Their transitive MLIR/LLVM dependencies are resolved by CMake shared-library linkage.
    exe.root_module.linkSystemLibrary("MLIR-C", .{});
    exe.root_module.linkSystemLibrary("MLIROraDialectC", .{});
    exe.root_module.linkSystemLibrary("MLIRSIRDialect", .{});

    switch (target.result.os.tag) {
        .linux => {
            exe.root_module.link_libcpp = true;
            exe.root_module.linkSystemLibrary("c++abi", .{});
        },
        .macos => {
            exe.root_module.link_libcpp = true;
        },
        else => {
            exe.root_module.link_libcpp = true;
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
    var examples_dir = std.Io.Dir.cwd().openDir(b.graph.io, "ora-example", .{ .iterate = true }) catch |err| {
        std.log.err("Failed to open examples directory: {}", .{err});
        return err;
    };
    defer examples_dir.close(b.graph.io);

    // iterate through all .ora files
    var walker = examples_dir.walk(allocator) catch |err| {
        std.log.err("Failed to walk examples directory: {}", .{err});
        return err;
    };
    defer walker.deinit();

    var tested_count: u32 = 0;
    var failed_count: u32 = 0;

    while (walker.next(b.graph.io) catch |err| {
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
            const result = std.process.run(allocator, b.graph.io, .{
                .argv = &[_][]const u8{
                    "./zig-out/bin/ora",
                    phase,
                    file_path,
                },
                .cwd = .{ .path = "." },
            }) catch |err| {
                std.log.err("Failed to run test for {s} with phase {s}: {}", .{ file_path, phase, err });
                failed_count += 1;
                continue;
            };

            switch (result.term) {
                .exited => |code| {
                    if (code != 0) {
                        std.log.err("FAILED: {s} (phase: {s}) with exit code {}", .{ file_path, phase, code });
                        std.log.err("Error output: {s}", .{result.stderr});
                        failed_count += 1;
                        break; // Don't test further phases if one fails
                    }
                },
                .signal => |sig| {
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

fn pathExists(b: *std.Build, path: []const u8) bool {
    std.Io.Dir.cwd().access(b.graph.io, path, .{}) catch return false;
    return true;
}

fn anyPathExists(b: *std.Build, paths: []const []const u8) bool {
    for (paths) |path| {
        if (pathExists(b, path)) return true;
    }
    return false;
}

fn hostHasSystemZ3(b: *std.Build) bool {
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

    return anyPathExists(b, &header_paths) and anyPathExists(b, &lib_paths);
}

/// Implementation of CMake build for Z3 libraries
fn buildZ3LibrariesImpl(step: *std.Build.Step, options: std.Build.Step.MakeOptions) anyerror!void {
    _ = options;

    const b = step.owner;
    const allocator = b.allocator;

    if (hostHasSystemZ3(b)) {
        std.log.info("System Z3 headers and library detected; skipping vendored Z3 build", .{});
        return;
    }

    // system Z3 headers/libraries not found, try to build from vendor
    std.log.info("System Z3 not fully available, checking vendor/z3...", .{});

    const cwd = std.Io.Dir.cwd();
    _ = cwd.openDir(b.graph.io, "vendor/z3", .{ .iterate = false }) catch {
        std.log.err("Z3 is required for compiler verification tests and no usable Z3 installation was found.", .{});
        std.log.err("Install Z3 or provide the vendored submodule:", .{});
        std.log.err("  macOS: brew install z3", .{});
        std.log.err("  Linux: sudo apt install z3 libz3-dev", .{});
        std.log.err("  vendor: git submodule update --init --recursive vendor/z3", .{});
        return error.Z3Unavailable;
    };

    // create build and install directories
    const build_dir = "vendor/z3/build-release";
    cwd.createDirPath(b.graph.io, build_dir) catch |err| switch (err) {
        error.PathAlreadyExists => {},
        else => return err,
    };

    const install_prefix = "vendor/z3-install";
    cwd.createDirPath(b.graph.io, install_prefix) catch |err| switch (err) {
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
        const probe = std.process.run(allocator, b.graph.io, .{ .argv = &[_][]const u8{ "ninja", "--version" }, .cwd = .{ .path = "." } }) catch null;
        if (probe) |res| {
            switch (res.term) {
                .exited => |code| {
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
        try cmake_args.append("-DCMAKE_CXX_FLAGS=-stdlib=libc++");
        try cmake_args.append("-DCMAKE_EXE_LINKER_FLAGS=-stdlib=libc++ -lc++abi");
        try cmake_args.append("-DCMAKE_SHARED_LINKER_FLAGS=-stdlib=libc++ -lc++abi");
        try cmake_args.append("-DCMAKE_MODULE_LINKER_FLAGS=-stdlib=libc++ -lc++abi");
        try cmake_args.append("-DCMAKE_CXX_COMPILER=clang++");
        try cmake_args.append("-DCMAKE_C_COMPILER=clang");
    } else if (builtin.os.tag == .macos) {
        try cmake_args.append("-DCMAKE_CXX_FLAGS=-stdlib=libc++");

        // fix SDK path issue after macOS/Xcode update
        // use xcrun to get the actual SDK path and set it explicitly
        const sdk_path_result = std.process.run(allocator, b.graph.io, .{
            .argv = &[_][]const u8{ "xcrun", "--show-sdk-path" },
            .cwd = .{ .path = "." },
        }) catch null;
        if (sdk_path_result) |result| {
            if (result.term == .exited and result.term.exited == 0) {
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

    const cfg_term = runInheritedProcess(b, cmake_args.items, ".") catch |err| {
        std.log.err("Failed to configure Z3 CMake: {}", .{err});
        return err;
    };
    switch (cfg_term) {
        .exited => |code| if (code != 0) {
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
    const build_term = runInheritedProcess(b, &build_args, ".") catch |err| {
        std.log.err("Failed to build Z3 with CMake: {}", .{err});
        return err;
    };
    switch (build_term) {
        .exited => |code| if (code != 0) {
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
    const using_system_z3 = hostHasSystemZ3(b);

    // Ensure vendored search paths exist to avoid hard failures when Zig checks
    // linker search directories before the z3 custom step populates them.
    std.Io.Dir.cwd().createDirPath(b.graph.io, "vendor/z3-install/include") catch {};
    std.Io.Dir.cwd().createDirPath(b.graph.io, "vendor/z3-install/lib") catch {};

    // Always expose vendored include/lib paths. The custom z3_step may populate
    // these directories later in the build graph.
    exe.root_module.addIncludePath(b.path("vendor/z3-install/include"));
    exe.root_module.addLibraryPath(b.path("vendor/z3-install/lib"));

    if (using_system_z3) {
        // add system Z3 paths based on platform
        switch (target.result.os.tag) {
            .macos => {
                // homebrew paths
                if (target.result.cpu.arch == .aarch64) {
                    exe.root_module.addIncludePath(.{ .cwd_relative = "/opt/homebrew/include" });
                    exe.root_module.addLibraryPath(.{ .cwd_relative = "/opt/homebrew/lib" });
                } else {
                    exe.root_module.addIncludePath(.{ .cwd_relative = "/usr/local/include" });
                    exe.root_module.addLibraryPath(.{ .cwd_relative = "/usr/local/lib" });
                }
            },
            .linux => {
                exe.root_module.addIncludePath(.{ .cwd_relative = "/usr/include" });
                exe.root_module.addLibraryPath(.{ .cwd_relative = "/usr/lib" });
                exe.root_module.addLibraryPath(.{ .cwd_relative = "/usr/local/lib" });
                if (target.result.cpu.arch == .x86_64) {
                    exe.root_module.addLibraryPath(.{ .cwd_relative = "/usr/lib/x86_64-linux-gnu" });
                }
            },
            else => {
                exe.root_module.addIncludePath(.{ .cwd_relative = "/usr/include" });
                exe.root_module.addLibraryPath(.{ .cwd_relative = "/usr/lib" });
            },
        }
    } else {
        // vendored Z3 — paths already set above
    }

    // Link Z3 without pkg-config. We add the stable system/vendored include
    // and library paths above; letting pkg-config inject Homebrew Cellar paths
    // makes CI sensitive to stale Zig caches and old absolute Z3 versions.
    exe.root_module.linkSystemLibrary("z3", .{ .use_pkg_config = .no });

    // link C++ standard library (Z3 is C++)
    switch (target.result.os.tag) {
        .linux => {
            exe.root_module.link_libcpp = true;
            exe.root_module.linkSystemLibrary("c++abi", .{});
        },
        .macos => {
            exe.root_module.link_libcpp = true;
        },
        else => {
            exe.root_module.link_libcpp = true;
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
                compile_step.root_module.addSystemIncludePath(.{ .cwd_relative = "/opt/homebrew/include" });
                compile_step.root_module.addLibraryPath(.{ .cwd_relative = "/opt/homebrew/lib" });
            } else {
                // intel Mac - Homebrew installs to /usr/local
                std.log.info("Adding Boost paths for Intel Mac", .{});
                compile_step.root_module.addSystemIncludePath(.{ .cwd_relative = "/usr/local/include" });
                compile_step.root_module.addLibraryPath(.{ .cwd_relative = "/usr/local/lib" });
            }
        },
        .linux => {
            // linux - check common package manager locations
            std.log.info("Adding Boost paths for Linux", .{});
            compile_step.root_module.addSystemIncludePath(.{ .cwd_relative = "/usr/include" });
            compile_step.root_module.addSystemIncludePath(.{ .cwd_relative = "/usr/local/include" });
            compile_step.root_module.addLibraryPath(.{ .cwd_relative = "/usr/lib" });
            compile_step.root_module.addLibraryPath(.{ .cwd_relative = "/usr/local/lib" });

            // also check for x86_64-linux-gnu paths (common on Ubuntu)
            if (arch_to_use == .x86_64) {
                compile_step.root_module.addLibraryPath(.{ .cwd_relative = "/usr/lib/x86_64-linux-gnu" });
            }
        },
        .windows => {
            // windows - typically vcpkg or manual installation
            std.log.info("Adding Boost paths for Windows", .{});
            // check for vcpkg installation
            if (std.process.hasEnvVarConstant("VCPKG_ROOT")) {
                compile_step.root_module.addSystemIncludePath(.{ .cwd_relative = "C:/vcpkg/installed/x64-windows/include" });
                compile_step.root_module.addLibraryPath(.{ .cwd_relative = "C:/vcpkg/installed/x64-windows/lib" });
            } else {
                // default paths for manual installation
                compile_step.root_module.addSystemIncludePath(.{ .cwd_relative = "C:/boost/include" });
                compile_step.root_module.addLibraryPath(.{ .cwd_relative = "C:/boost/lib" });
            }
        },
        else => {
            // fallback for other platforms
            std.log.warn("Unknown platform for Boost paths - using default system paths", .{});
            compile_step.root_module.addSystemIncludePath(.{ .cwd_relative = "/usr/include" });
            compile_step.root_module.addSystemIncludePath(.{ .cwd_relative = "/usr/local/include" });
            compile_step.root_module.addLibraryPath(.{ .cwd_relative = "/usr/lib" });
            compile_step.root_module.addLibraryPath(.{ .cwd_relative = "/usr/local/lib" });
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
        compile_step.root_module.addSystemIncludePath(.{ .cwd_relative = include_path });
        compile_step.root_module.addLibraryPath(.{ .cwd_relative = lib_path });
    }
}

fn createEvmBlstLibrary(
    b: *std.Build,
    target: std.Build.ResolvedTarget,
    optimize: std.builtin.OptimizeMode,
) *std.Build.Step.Compile {
    const is_wasm = target.result.cpu.arch == .wasm32 or target.result.cpu.arch == .wasm64;
    const lib = b.addLibrary(.{
        .name = "blst",
        .linkage = .static,
        .use_llvm = true,
        .root_module = b.createModule(.{
            .target = target,
            .optimize = optimize,
        }),
    });
    lib.root_module.link_libc = true;

    if (!is_wasm) {
        const blst_build_cmd = if (target.result.os.tag == .windows)
            b.addSystemCommand(&.{ "bash", "./build.sh" })
        else
            b.addSystemCommand(&.{"./build.sh"});
        blst_build_cmd.setCwd(b.path("lib/evm/vendor/c-kzg-4844/blst"));
        lib.step.dependOn(&blst_build_cmd.step);
        lib.root_module.addAssemblyFile(b.path("lib/evm/vendor/c-kzg-4844/blst/build/assembly.S"));
    }

    const c_flags = if (is_wasm)
        &[_][]const u8{ "-std=c99", "-D__BLST_PORTABLE__", "-D__BLST_NO_ASM__", "-fno-sanitize=undefined" }
    else
        &[_][]const u8{ "-std=c99", "-fPIC", "-D__BLST_PORTABLE__", "-fno-sanitize=undefined" };

    lib.root_module.addCSourceFiles(.{
        .files = &.{
            "lib/evm/vendor/c-kzg-4844/blst/src/server.c",
        },
        .flags = c_flags,
    });
    lib.root_module.addIncludePath(b.path("lib/evm/vendor/c-kzg-4844/blst/bindings"));
    return lib;
}

fn createEvmCKzgLibrary(
    b: *std.Build,
    target: std.Build.ResolvedTarget,
    optimize: std.builtin.OptimizeMode,
    blst_lib: *std.Build.Step.Compile,
) *std.Build.Step.Compile {
    const lib = b.addLibrary(.{
        .name = "c-kzg-4844",
        .linkage = .static,
        .use_llvm = true,
        .root_module = b.createModule(.{
            .target = target,
            .optimize = optimize,
        }),
    });
    lib.root_module.link_libc = true;
    lib.root_module.linkLibrary(blst_lib);
    lib.root_module.addCSourceFiles(.{
        .files = &.{
            "lib/evm/vendor/c-kzg-4844/src/ckzg.c",
        },
        .flags = &.{ "-std=c99", "-fPIC", "-fno-sanitize=undefined" },
    });
    lib.root_module.addIncludePath(b.path("lib/evm/vendor/c-kzg-4844/src"));
    lib.root_module.addIncludePath(b.path("lib/evm/vendor/c-kzg-4844/blst/bindings"));
    return lib;
}

fn optimizeModeName(optimize: std.builtin.OptimizeMode) []const u8 {
    return switch (optimize) {
        .Debug => "Debug",
        .ReleaseSafe => "ReleaseSafe",
        .ReleaseFast => "ReleaseFast",
        .ReleaseSmall => "ReleaseSmall",
    };
}

fn compilerTestFilters(b: *std.Build, option_filter: ?[]const u8) []const []const u8 {
    var filters: std.ArrayList([]const u8) = .empty;
    if (option_filter) |filter| {
        filters.append(b.allocator, filter) catch @panic("OOM");
    }

    const args = b.args orelse {
        return filters.toOwnedSlice(b.allocator) catch @panic("OOM");
    };

    var index: usize = 0;
    while (index < args.len) : (index += 1) {
        const arg = args[index];
        if (std.mem.eql(u8, arg, "--test-filter")) {
            index += 1;
            if (index >= args.len) @panic("--test-filter requires a value");
            filters.append(b.allocator, args[index]) catch @panic("OOM");
        } else if (std.mem.startsWith(u8, arg, "--test-filter=")) {
            filters.append(b.allocator, arg["--test-filter=".len..]) catch @panic("OOM");
        }
    }

    return filters.toOwnedSlice(b.allocator) catch @panic("OOM");
}
