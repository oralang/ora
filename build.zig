const std = @import("std");

// Although this function looks imperative, note that its job is to
// declaratively construct a build graph that will be executed by an external
// runner.
pub fn build(b: *std.Build) void {
    // Standard target options allows the person running `zig build` to choose
    // what target to build for. Here we do not override the defaults, which
    // means any target is allowed, and the default is native. Other options
    // for restricting supported target set are available.
    const target = b.standardTargetOptions(.{});

    // Standard optimization options allow the person running `zig build` to select
    // between Debug, ReleaseSafe, ReleaseFast, and ReleaseSmall. Here we do not
    // set a preferred release mode, allowing the user to decide how to optimize.
    const optimize = b.standardOptimizeOption(.{});

    // Build Solidity libraries using CMake
    const cmake_step = buildSolidityLibraries(b, target, optimize);

    // This creates a "module", which represents a collection of source files alongside
    // some compilation options, such as optimization mode and linked system libraries.
    // Every executable or library we compile will be based on one or more modules.
    const lib_mod = b.createModule(.{
        // `root_source_file` is the Zig "entry point" of the module. If a module
        // only contains e.g. external object files, you can make this `null`.
        // In this case the main source file is merely a path, however, in more
        // complicated build scripts, this could be a generated file.
        .root_source_file = b.path("src/root.zig"),
        .target = target,
        .optimize = optimize,
    });

    // We will also create a module for our other entry point, 'main.zig'.
    const exe_mod = b.createModule(.{
        // `root_source_file` is the Zig "entry point" of the module. If a module
        // only contains e.g. external object files, you can make this `null`.
        // In this case the main source file is merely a path, however, in more
        // complicated build scripts, this could be a generated file.
        .root_source_file = b.path("src/main.zig"),
        .target = target,
        .optimize = optimize,
    });

    // Modules can depend on one another using the `std.Build.Module.addImport` function.
    // This is what allows Zig source code to use `@import("foo")` where 'foo' is not a
    // file path. In this case, we set up `exe_mod` to import `lib_mod`.
    exe_mod.addImport("ora_lib", lib_mod);

    // Now, we will create a static library based on the module we created above.
    // This creates a `std.Build.Step.Compile`, which is the build step responsible
    // for actually invoking the compiler.
    const lib = b.addLibrary(.{
        .linkage = .static,
        .name = "ora",
        .root_module = lib_mod,
    });

    // This declares intent for the library to be installed into the standard
    // location when the user invokes the "install" step (the default step when
    // running `zig build`).
    b.installArtifact(lib);

    // This creates another `std.Build.Step.Compile`, but this one builds an executable
    // rather than a static library.
    const exe = b.addExecutable(.{
        .name = "ora",
        .root_module = exe_mod,
    });

    // Build and link Yul wrapper
    const yul_wrapper = buildYulWrapper(b, target, optimize, cmake_step);
    exe.addObject(yul_wrapper);

    // Add include path for yul_wrapper.h
    exe.addIncludePath(b.path("src"));
    lib.addIncludePath(b.path("src"));

    // Link Solidity libraries to the executable
    linkSolidityLibraries(b, exe, cmake_step, target);

    // This declares intent for the executable to be installed into the
    // standard location when the user invokes the "install" step (the default
    // step when running `zig build`).
    b.installArtifact(exe);

    // This *creates* a Run step in the build graph, to be executed when another
    // step is evaluated that depends on it. The next line below will establish
    // such a dependency.
    const run_cmd = b.addRunArtifact(exe);

    // By making the run step depend on the install step, it will be run from the
    // installation directory rather than directly from within the cache directory.
    // This is not necessary, however, if the application depends on other installed
    // files, this ensures they will be present and in the expected location.
    run_cmd.step.dependOn(b.getInstallStep());

    // This allows the user to pass arguments to the application in the build
    // command itself, like this: `zig build run -- arg1 arg2 etc`
    if (b.args) |args| {
        run_cmd.addArgs(args);
    }

    // This creates a build step. It will be visible in the `zig build --help` menu,
    // and can be selected like this: `zig build run`
    // This will evaluate the `run` step rather than the default, which is "install".
    const run_step = b.step("run", "Run the app");
    run_step.dependOn(&run_cmd.step);

    // Create yul test executable
    const yul_test_mod = b.createModule(.{
        .root_source_file = b.path("examples/demos/yul_test.zig"),
        .target = target,
        .optimize = optimize,
    });
    yul_test_mod.addImport("ora_lib", lib_mod);

    const yul_test = b.addExecutable(.{
        .name = "yul_test",
        .root_module = yul_test_mod,
    });

    // Link Yul wrapper and Solidity libraries
    const yul_test_wrapper = buildYulWrapper(b, target, optimize, cmake_step);
    yul_test.addObject(yul_test_wrapper);

    // Add include path for yul_wrapper.h
    yul_test.addIncludePath(b.path("src"));

    linkSolidityLibraries(b, yul_test, cmake_step, target);

    const run_yul_test = b.addRunArtifact(yul_test);
    run_yul_test.step.dependOn(b.getInstallStep());

    const yul_test_step = b.step("yul-test", "Run the Yul integration test");
    yul_test_step.dependOn(&run_yul_test.step);

    // Create optimization demo executable
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

    // Create formal verification demo executable
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

    // Add comprehensive compiler testing framework
    addCompilerTestFramework(b, lib_mod, target, optimize);

    // Add new lexer testing framework
    addLexerTestFramework(b, lib_mod, target, optimize);

    // Add individual test suites - always included (files are present in repo)
    const ast_tests = b.addTest(.{
        .root_source_file = b.path("tests/ast_test.zig"),
        .target = target,
        .optimize = optimize,
    });
    ast_tests.root_module.addImport("ora", lib_mod);
    const run_ast_tests = b.addRunArtifact(ast_tests);

    const ast_clean_tests = b.addTest(.{
        .root_source_file = b.path("tests/ast_test_clean.zig"),
        .target = target,
        .optimize = optimize,
    });
    ast_clean_tests.root_module.addImport("ora", lib_mod);
    const run_ast_clean_tests = b.addRunArtifact(ast_clean_tests);

    // Legacy lexer test (keep for compatibility)
    const simple_lexer_test = b.addTest(.{
        .root_source_file = b.path("tests/lexer_test.zig"),
        .target = target,
        .optimize = optimize,
    });
    simple_lexer_test.root_module.addImport("ora", lib_mod);
    const run_simple_lexer_test = b.addRunArtifact(simple_lexer_test);

    const test_lexer_legacy_step = b.step("test-lexer-legacy", "Run legacy lexer tests");
    test_lexer_legacy_step.dependOn(&run_simple_lexer_test.step);

    // Add expression parser tests
    const expression_parser_tests = b.addTest(.{
        .root_source_file = b.path("tests/expression_parser_test.zig"),
        .target = target,
        .optimize = optimize,
    });
    expression_parser_tests.root_module.addImport("ora", lib_mod);
    const run_expression_parser_tests = b.addRunArtifact(expression_parser_tests);

    const test_expression_parser_step = b.step("test-expression-parser", "Run expression parser tests");
    test_expression_parser_step.dependOn(&run_expression_parser_tests.step);

    // Create AST test step
    const test_ast_step = b.step("test-ast", "Run AST tests");
    test_ast_step.dependOn(&run_ast_tests.step);
    test_ast_step.dependOn(&run_ast_clean_tests.step);

    // Create comprehensive test step that runs all tests
    const test_all_step = b.step("test", "Run all tests");
    test_all_step.dependOn(b.getInstallStep()); // Ensure everything is built first

    // Add all test categories
    const test_framework_internal = b.addTest(.{
        .root_source_file = b.path("tests/test_framework.zig"),
        .target = target,
        .optimize = optimize,
    });
    test_framework_internal.root_module.addImport("ora", lib_mod);

    test_all_step.dependOn(&b.addRunArtifact(test_framework_internal).step);
    test_all_step.dependOn(test_ast_step);
    test_all_step.dependOn(&run_expression_parser_tests.step);
    // Optionally: expose lexer-suite-verbose as a separate step (not part of default test)
    // Run invalid parser fixtures
    const invalid_parser_tests = b.addTest(.{
        .root_source_file = b.path("tests/parser/parser_invalid_fixture_suite.zig"),
        .target = target,
        .optimize = optimize,
    });
    invalid_parser_tests.root_module.addImport("ora", lib_mod);
    test_all_step.dependOn(&b.addRunArtifact(invalid_parser_tests).step);

    // Lossless round-trip tests
    const lossless_tests = b.addTest(.{
        .root_source_file = b.path("tests/lossless_roundtrip_test.zig"),
        .target = target,
        .optimize = optimize,
    });
    lossless_tests.root_module.addImport("ora", lib_mod);
    test_all_step.dependOn(&b.addRunArtifact(lossless_tests).step);

    // Doc comment extraction tests
    const doc_tests = b.addTest(.{
        .root_source_file = b.path("tests/doc_comment_extraction_test.zig"),
        .target = target,
        .optimize = optimize,
    });
    doc_tests.root_module.addImport("ora", lib_mod);
    test_all_step.dependOn(&b.addRunArtifact(doc_tests).step);

    // CST token stream tests
    const cst_tests = b.addTest(.{
        .root_source_file = b.path("tests/cst_token_stream_test.zig"),
        .target = target,
        .optimize = optimize,
    });
    cst_tests.root_module.addImport("ora", lib_mod);
    test_all_step.dependOn(&b.addRunArtifact(cst_tests).step);

    // CST parser integration test
    const cst_parser_tests = b.addTest(.{
        .root_source_file = b.path("tests/cst_parser_top_level_test.zig"),
        .target = target,
        .optimize = optimize,
    });
    cst_parser_tests.root_module.addImport("ora", lib_mod);
    test_all_step.dependOn(&b.addRunArtifact(cst_parser_tests).step);

    // Semantics fixtures suite
    const semantics_fixture_tests = b.addTest(.{
        .root_source_file = b.path("tests/semantics/semantics_fixture_suite.zig"),
        .target = target,
        .optimize = optimize,
    });
    semantics_fixture_tests.root_module.addImport("ora", lib_mod);
    test_all_step.dependOn(&b.addRunArtifact(semantics_fixture_tests).step);

    // TypeInfo render and equality tests
    const type_info_tests = b.addTest(.{
        .root_source_file = b.path("tests/type_info_render_and_eq_test.zig"),
        .target = target,
        .optimize = optimize,
    });
    type_info_tests.root_module.addImport("ora", lib_mod);
    test_all_step.dependOn(&b.addRunArtifact(type_info_tests).step);

    // Byte offset span tests
    const span_tests = b.addTest(.{
        .root_source_file = b.path("tests/byte_offset_span_test.zig"),
        .target = target,
        .optimize = optimize,
    });
    span_tests.root_module.addImport("ora", lib_mod);
    test_all_step.dependOn(&b.addRunArtifact(span_tests).step);

    // Documentation generation
    const install_docs = b.addInstallDirectory(.{
        .source_dir = lib.getEmittedDocs(),
        .install_dir = .prefix,
        .install_subdir = "docs",
    });

    const docs_step = b.step("docs", "Generate and install documentation");
    docs_step.dependOn(&install_docs.step);

    // Add formal verification test
    const formal_test_mod = b.createModule(.{
        .root_source_file = b.path("examples/demos/formal_test.zig"),
        .target = target,
        .optimize = optimize,
    });
    formal_test_mod.addImport("ora_lib", lib_mod);

    const formal_test = b.addExecutable(.{
        .name = "formal_test",
        .root_module = formal_test_mod,
    });
    // Don't install by default - only run when requested

    const formal_test_step = b.step("formal-test", "Run formal verification test");
    const formal_test_run = b.addRunArtifact(formal_test);
    formal_test_step.dependOn(&formal_test_run.step);

    // Add example testing step
    const test_examples_step = b.step("test-examples", "Test all example .ora files");
    const test_examples_run = createExampleTestStep(b, exe);
    test_examples_step.dependOn(test_examples_run);
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

/// Build Solidity libraries using CMake
fn buildSolidityLibraries(b: *std.Build, target: std.Build.ResolvedTarget, optimize: std.builtin.OptimizeMode) *std.Build.Step {
    _ = target;
    _ = optimize;

    const cmake_step = b.allocator.create(std.Build.Step) catch @panic("OOM");
    cmake_step.* = std.Build.Step.init(.{
        .id = .custom,
        .name = "cmake-build-solidity",
        .owner = b,
        .makeFn = buildSolidityLibrariesImpl,
    });

    return cmake_step;
}

/// Implementation of CMake build for Solidity libraries
fn buildSolidityLibrariesImpl(step: *std.Build.Step, options: std.Build.Step.MakeOptions) anyerror!void {
    _ = options;

    const b = step.owner;
    const allocator = b.allocator;

    // Detect target platform at runtime for CMake configuration
    const builtin = @import("builtin");

    // Create build directory
    const build_dir = "vendor/solidity/build";
    std.fs.cwd().makeDir(build_dir) catch |err| switch (err) {
        error.PathAlreadyExists => {},
        else => return err,
    };

    // Determine platform-specific CMake flags for C++ ABI compatibility
    var cmake_args = std.ArrayList([]const u8).init(allocator);
    defer cmake_args.deinit();

    try cmake_args.appendSlice(&[_][]const u8{
        "cmake",
        "-S",
        "vendor/solidity",
        "-B",
        build_dir,
        "-DCMAKE_BUILD_TYPE=Release",
        "-DONLY_BUILD_SOLIDITY_LIBRARIES=ON",
        "-DTESTS=OFF",
    });

    // Add platform-specific flags to ensure consistent C++ ABI
    if (builtin.os.tag == .linux) {
        std.log.info("Adding Boost paths for Linux", .{});
        // Force libc++ usage on Linux to match macOS ABI
        try cmake_args.append("-DCMAKE_CXX_FLAGS=-stdlib=libc++ -lc++abi");
        try cmake_args.append("-DCMAKE_CXX_COMPILER=clang++");
        try cmake_args.append("-DCMAKE_C_COMPILER=clang");
    } else if (builtin.os.tag == .macos) {
        std.log.info("Adding Boost paths for Apple Silicon Mac", .{});
        // macOS already uses libc++ by default, but be explicit
        try cmake_args.append("-DCMAKE_CXX_FLAGS=-stdlib=libc++");
    } else if (builtin.os.tag == .windows) {
        std.log.info("Adding Boost paths for Windows", .{});
        // Windows: Use MSVC and configure Boost paths
        try cmake_args.append("-DCMAKE_CXX_FLAGS=/std:c++20");

        // Try multiple possible Boost locations for Windows
        try cmake_args.append("-DCMAKE_PREFIX_PATH=C:/vcpkg/installed/x64-windows;C:/ProgramData/chocolatey/lib/boost-msvc-14.3/lib/native;C:/tools/boost");

        // Set Boost-specific variables for older CMake compatibility
        try cmake_args.append("-DBoost_USE_STATIC_LIBS=ON");
        try cmake_args.append("-DBoost_USE_MULTITHREADED=ON");
        try cmake_args.append("-DBoost_USE_STATIC_RUNTIME=OFF");

        // Suppress CMake developer warnings
        try cmake_args.append("-Wno-dev");
    }

    // Configure CMake
    const cmake_configure = std.process.Child.run(.{
        .allocator = allocator,
        .argv = cmake_args.items,
        .cwd = ".",
    }) catch |err| {
        std.log.err("Failed to configure CMake: {}", .{err});
        return err;
    };

    if (cmake_configure.term.Exited != 0) {
        std.log.err("CMake configure failed with exit code: {}", .{cmake_configure.term.Exited});
        std.log.err("CMake stderr: {s}", .{cmake_configure.stderr});
        return error.CMakeConfigureFailed;
    }

    // Build libraries
    const cmake_build = std.process.Child.run(.{
        .allocator = allocator,
        .argv = &[_][]const u8{
            "cmake",
            "--build",
            build_dir,
            "--target",
            "solutil",
            "--target",
            "langutil",
            "--target",
            "smtutil",
            "--target",
            "evmasm",
            "--target",
            "yul",
        },
    }) catch |err| {
        std.log.err("Failed to build with CMake: {}", .{err});
        return err;
    };

    if (cmake_build.term.Exited != 0) {
        std.log.err("CMake build failed with exit code: {}", .{cmake_build.term.Exited});
        std.log.err("CMake stderr: {s}", .{cmake_build.stderr});
        return error.CMakeBuildFailed;
    }

    std.log.info("Successfully built Solidity libraries", .{});
}

/// Build the Yul wrapper C++ library
fn buildYulWrapper(b: *std.Build, target: std.Build.ResolvedTarget, optimize: std.builtin.OptimizeMode, cmake_step: *std.Build.Step) *std.Build.Step.Compile {
    const yul_wrapper_mod = b.createModule(.{
        .target = target,
        .optimize = optimize,
    });

    const yul_wrapper = b.addObject(.{
        .name = "yul_wrapper",
        .root_module = yul_wrapper_mod,
    });

    // Depend on CMake build
    yul_wrapper.step.dependOn(cmake_step);

    // Add the C++ source file
    yul_wrapper.addCSourceFile(.{
        .file = b.path("src/yul_wrapper.cpp"),
        .flags = &[_][]const u8{
            "-std=c++20",
            "-fPIC",
            "-Wno-deprecated",
        },
    });

    // Add include directories
    yul_wrapper.addIncludePath(b.path("src"));
    yul_wrapper.addIncludePath(b.path("vendor/solidity"));
    yul_wrapper.addIncludePath(b.path("vendor/solidity/libsolutil"));
    yul_wrapper.addIncludePath(b.path("vendor/solidity/liblangutil"));
    yul_wrapper.addIncludePath(b.path("vendor/solidity/libsmtutil"));
    yul_wrapper.addIncludePath(b.path("vendor/solidity/libevmasm"));
    yul_wrapper.addIncludePath(b.path("vendor/solidity/libyul"));
    yul_wrapper.addIncludePath(b.path("vendor/solidity/deps/fmtlib/include"));
    yul_wrapper.addIncludePath(b.path("vendor/solidity/deps/range-v3/include"));
    yul_wrapper.addIncludePath(b.path("vendor/solidity/deps/nlohmann-json/include"));

    // Add platform-specific Boost paths
    addBoostPaths(b, yul_wrapper, target);

    // Link C++ standard library
    // Use the appropriate C++ stdlib based on the target
    switch (target.result.os.tag) {
        .linux => {
            // On Linux, use libstdc++ for better compatibility
            yul_wrapper.linkSystemLibrary("stdc++");
        },
        .macos => {
            // On macOS, use libc++
            yul_wrapper.linkLibCpp();
        },
        else => {
            // Default to libc++
            yul_wrapper.linkLibCpp();
        },
    }

    return yul_wrapper;
}

/// Link Solidity libraries to the executable
fn linkSolidityLibraries(b: *std.Build, exe: *std.Build.Step.Compile, cmake_step: *std.Build.Step, target: std.Build.ResolvedTarget) void {
    // Make executable depend on CMake build
    exe.step.dependOn(cmake_step);

    // Add library paths
    exe.addLibraryPath(b.path("vendor/solidity/build/libsolutil"));
    exe.addLibraryPath(b.path("vendor/solidity/build/liblangutil"));
    exe.addLibraryPath(b.path("vendor/solidity/build/libsmtutil"));
    exe.addLibraryPath(b.path("vendor/solidity/build/libevmasm"));
    exe.addLibraryPath(b.path("vendor/solidity/build/libyul"));

    // Link libraries (order matters - dependencies first)
    exe.linkSystemLibrary("solutil");
    exe.linkSystemLibrary("langutil");
    exe.linkSystemLibrary("smtutil");
    exe.linkSystemLibrary("evmasm");
    exe.linkSystemLibrary("yul");

    // Link C++ standard library
    // Use the appropriate C++ stdlib based on the target
    switch (target.result.os.tag) {
        .linux => {
            // On Linux, use libstdc++ for better compatibility
            exe.linkSystemLibrary("stdc++");
        },
        .macos => {
            // On macOS, use libc++
            exe.linkLibCpp();
        },
        else => {
            // Default to libc++
            exe.linkLibCpp();
        },
    }

    // Add include directories for headers
    exe.addIncludePath(b.path("vendor/solidity"));
    exe.addIncludePath(b.path("vendor/solidity/libsolutil"));
    exe.addIncludePath(b.path("vendor/solidity/liblangutil"));
    exe.addIncludePath(b.path("vendor/solidity/libsmtutil"));
    exe.addIncludePath(b.path("vendor/solidity/libevmasm"));
    exe.addIncludePath(b.path("vendor/solidity/libyul"));
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

    // Depend on the main executable being built
    test_step.dependOn(&exe.step);

    return test_step;
}

/// Implementation of example testing
fn runExampleTests(step: *std.Build.Step, options: std.Build.Step.MakeOptions) anyerror!void {
    _ = options;

    const b = step.owner;
    const allocator = b.allocator;

    std.log.info("Testing all .ora example files...", .{});

    // Get examples directory
    var examples_dir = std.fs.cwd().openDir("examples", .{ .iterate = true }) catch |err| {
        std.log.err("Failed to open examples directory: {}", .{err});
        return err;
    };
    defer examples_dir.close();

    // Iterate through all .ora files
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
        // Skip non-.ora files
        if (entry.kind != .file) continue;
        if (!std.mem.endsWith(u8, entry.basename, ".ora")) continue;

        const file_path = b.fmt("examples/{s}", .{entry.path});
        std.log.info("Testing: {s}", .{file_path});

        // Test each compilation phase
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

            if (result.term.Exited != 0) {
                std.log.err("FAILED: {s} (phase: {s})", .{ file_path, phase });
                std.log.err("Error output: {s}", .{result.stderr});
                failed_count += 1;
                break; // Don't test further phases if one fails
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

/// Add platform-specific Boost include and library paths
/// For cross-compilation, we use the host system paths where dependencies are actually installed
fn addBoostPaths(b: *std.Build, compile_step: *std.Build.Step.Compile, target: std.Build.ResolvedTarget) void {
    const target_info = target.result;
    const host_info = @import("builtin").target;

    // Determine if we're cross-compiling
    const is_cross_compiling = target_info.os.tag != host_info.os.tag or
        target_info.cpu.arch != host_info.cpu.arch;

    // For cross-compilation, use host system paths where dependencies are installed
    // For native compilation, use target system paths
    const os_to_use = if (is_cross_compiling) host_info.os.tag else target_info.os.tag;
    const arch_to_use = if (is_cross_compiling) host_info.cpu.arch else target_info.cpu.arch;

    if (is_cross_compiling) {
        std.log.info("Cross-compiling from {s}-{s} to {s}-{s} - using host paths for Boost", .{ @tagName(host_info.os.tag), @tagName(host_info.cpu.arch), @tagName(target_info.os.tag), @tagName(target_info.cpu.arch) });
    }

    switch (os_to_use) {
        .macos => {
            // Check if Apple Silicon or Intel Mac
            if (arch_to_use == .aarch64) {
                // Apple Silicon - Homebrew installs to /opt/homebrew
                std.log.info("Adding Boost paths for Apple Silicon Mac", .{});
                compile_step.addSystemIncludePath(.{ .cwd_relative = "/opt/homebrew/include" });
                compile_step.addLibraryPath(.{ .cwd_relative = "/opt/homebrew/lib" });
            } else {
                // Intel Mac - Homebrew installs to /usr/local
                std.log.info("Adding Boost paths for Intel Mac", .{});
                compile_step.addSystemIncludePath(.{ .cwd_relative = "/usr/local/include" });
                compile_step.addLibraryPath(.{ .cwd_relative = "/usr/local/lib" });
            }
        },
        .linux => {
            // Linux - check common package manager locations
            std.log.info("Adding Boost paths for Linux", .{});
            compile_step.addSystemIncludePath(.{ .cwd_relative = "/usr/include" });
            compile_step.addSystemIncludePath(.{ .cwd_relative = "/usr/local/include" });
            compile_step.addLibraryPath(.{ .cwd_relative = "/usr/lib" });
            compile_step.addLibraryPath(.{ .cwd_relative = "/usr/local/lib" });

            // Also check for x86_64-linux-gnu paths (common on Ubuntu)
            if (arch_to_use == .x86_64) {
                compile_step.addLibraryPath(.{ .cwd_relative = "/usr/lib/x86_64-linux-gnu" });
            }
        },
        .windows => {
            // Windows - typically vcpkg or manual installation
            std.log.info("Adding Boost paths for Windows", .{});
            // Check for vcpkg installation
            if (std.process.hasEnvVarConstant("VCPKG_ROOT")) {
                compile_step.addSystemIncludePath(.{ .cwd_relative = "C:/vcpkg/installed/x64-windows/include" });
                compile_step.addLibraryPath(.{ .cwd_relative = "C:/vcpkg/installed/x64-windows/lib" });
            } else {
                // Default paths for manual installation
                compile_step.addSystemIncludePath(.{ .cwd_relative = "C:/boost/include" });
                compile_step.addLibraryPath(.{ .cwd_relative = "C:/boost/lib" });
            }
        },
        else => {
            // Fallback for other platforms
            std.log.warn("Unknown platform for Boost paths - using default system paths", .{});
            compile_step.addSystemIncludePath(.{ .cwd_relative = "/usr/include" });
            compile_step.addSystemIncludePath(.{ .cwd_relative = "/usr/local/include" });
            compile_step.addLibraryPath(.{ .cwd_relative = "/usr/lib" });
            compile_step.addLibraryPath(.{ .cwd_relative = "/usr/local/lib" });
        },
    }

    // Try to check if Boost is available via environment variable
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

/// Add comprehensive compiler testing framework to build system
fn addCompilerTestFramework(b: *std.Build, lib_mod: *std.Build.Module, target: std.Build.ResolvedTarget, optimize: std.builtin.OptimizeMode) void {
    // Test framework infrastructure tests
    const test_framework_tests = b.addTest(.{
        .root_source_file = b.path("tests/test_framework.zig"),
        .target = target,
        .optimize = optimize,
    });
    test_framework_tests.root_module.addImport("ora", lib_mod);

    // Test arena and memory management tests
    const test_arena_tests = b.addTest(.{
        .root_source_file = b.path("tests/common/test_arena.zig"),
        .target = target,
        .optimize = optimize,
    });

    // Test result types tests
    const test_result_tests = b.addTest(.{
        .root_source_file = b.path("tests/common/test_result.zig"),
        .target = target,
        .optimize = optimize,
    });

    // Test runner tests
    const test_runner_tests = b.addTest(.{
        .root_source_file = b.path("tests/common/test_runner.zig"),
        .target = target,
        .optimize = optimize,
    });

    // Assertions tests
    const assertions_tests = b.addTest(.{
        .root_source_file = b.path("tests/common/assertions.zig"),
        .target = target,
        .optimize = optimize,
    });
    assertions_tests.root_module.addImport("ora", lib_mod);

    // Test helpers tests
    const test_helpers_tests = b.addTest(.{
        .root_source_file = b.path("tests/common/test_helpers.zig"),
        .target = target,
        .optimize = optimize,
    });
    test_helpers_tests.root_module.addImport("ora", lib_mod);

    // Fixtures tests
    const fixtures_tests = b.addTest(.{
        .root_source_file = b.path("tests/common/fixtures.zig"),
        .target = target,
        .optimize = optimize,
    });

    // Fixture cache tests
    const fixture_cache_tests = b.addTest(.{
        .root_source_file = b.path("tests/common/fixture_cache.zig"),
        .target = target,
        .optimize = optimize,
    });

    // Coverage tests
    const coverage_tests = b.addTest(.{
        .root_source_file = b.path("tests/common/coverage.zig"),
        .target = target,
        .optimize = optimize,
    });

    // CI integration tests
    const ci_integration_tests = b.addTest(.{
        .root_source_file = b.path("tests/common/ci_integration.zig"),
        .target = target,
        .optimize = optimize,
    });

    // Create test steps for the framework components
    const test_framework_step = b.step("test-framework", "Run test framework infrastructure tests");
    test_framework_step.dependOn(&b.addRunArtifact(test_framework_tests).step);
    test_framework_step.dependOn(&b.addRunArtifact(test_arena_tests).step);
    test_framework_step.dependOn(&b.addRunArtifact(test_result_tests).step);
    test_framework_step.dependOn(&b.addRunArtifact(test_runner_tests).step);
    test_framework_step.dependOn(&b.addRunArtifact(assertions_tests).step);
    test_framework_step.dependOn(&b.addRunArtifact(test_helpers_tests).step);
    test_framework_step.dependOn(&b.addRunArtifact(fixtures_tests).step);
    test_framework_step.dependOn(&b.addRunArtifact(fixture_cache_tests).step);
    test_framework_step.dependOn(&b.addRunArtifact(coverage_tests).step);
    test_framework_step.dependOn(&b.addRunArtifact(ci_integration_tests).step);
}

/// Add lexer testing framework to build system
fn addLexerTestFramework(b: *std.Build, lib_mod: *std.Build.Module, target: std.Build.ResolvedTarget, optimize: std.builtin.OptimizeMode) void {
    // Lexer test suite
    const lexer_test_suite = b.addTest(.{
        .root_source_file = b.path("tests/lexer/lexer_test_suite.zig"),
        .target = target,
        .optimize = optimize,
    });
    lexer_test_suite.root_module.addImport("ora", lib_mod);

    // Lexer test fixtures
    const lexer_test_fixtures = b.addTest(.{
        .root_source_file = b.path("tests/lexer/lexer_test_fixtures.zig"),
        .target = target,
        .optimize = optimize,
    });
    lexer_test_fixtures.root_module.addImport("ora", lib_mod);

    // Create lexer test executable for standalone running
    const lexer_test_exe = b.addExecutable(.{
        .name = "lexer_test_suite",
        .root_source_file = b.path("tests/lexer/lexer_test_suite.zig"),
        .target = target,
        .optimize = optimize,
    });
    lexer_test_exe.root_module.addImport("ora", lib_mod);

    // Create test steps
    const test_lexer_step = b.step("test-lexer", "Run comprehensive lexer tests");
    test_lexer_step.dependOn(&b.addRunArtifact(lexer_test_suite).step);
    test_lexer_step.dependOn(&b.addRunArtifact(lexer_test_fixtures).step);

    // Create lexer benchmark step
    const benchmark_lexer_step = b.step("benchmark-lexer", "Run lexer performance benchmarks");
    const lexer_benchmark_run = b.addRunArtifact(lexer_test_exe);
    lexer_benchmark_run.addArg("--benchmark");
    benchmark_lexer_step.dependOn(&lexer_benchmark_run.step);

    // Create lexer test with verbose output
    const test_lexer_verbose_step = b.step("test-lexer-verbose", "Run lexer tests with verbose output");
    const lexer_verbose_run = b.addRunArtifact(lexer_test_exe);
    lexer_verbose_run.addArg("--verbose");
    test_lexer_verbose_step.dependOn(&lexer_verbose_run.step);

    // Install lexer test executable
    b.installArtifact(lexer_test_exe);
}
