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
    linkSolidityLibraries(b, exe, cmake_step);

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

    // Create parser demo executable
    const parser_demo_mod = b.createModule(.{
        .root_source_file = b.path("examples/demos/parser_demo.zig"),
        .target = target,
        .optimize = optimize,
    });
    parser_demo_mod.addImport("ora_lib", lib_mod);

    const parser_demo = b.addExecutable(.{
        .name = "parser_demo",
        .root_module = parser_demo_mod,
    });

    const run_parser_demo = b.addRunArtifact(parser_demo);
    run_parser_demo.step.dependOn(b.getInstallStep());

    const parser_demo_step = b.step("parser-demo", "Run the parser demo");
    parser_demo_step.dependOn(&run_parser_demo.step);

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

    linkSolidityLibraries(b, yul_test, cmake_step);

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

    // Creates a step for unit testing. This only builds the test executable
    // but does not run it.
    const lib_unit_tests = b.addTest(.{
        .root_module = lib_mod,
    });

    const run_lib_unit_tests = b.addRunArtifact(lib_unit_tests);

    const exe_unit_tests = b.addTest(.{
        .root_module = exe_mod,
    });

    const run_exe_unit_tests = b.addRunArtifact(exe_unit_tests);

    // Similar to creating the run step earlier, this exposes a `test` step to
    // the `zig build --help` menu, providing a way for the user to request
    // running the unit tests.
    const test_step = b.step("test", "Run unit tests");
    test_step.dependOn(&run_lib_unit_tests.step);
    test_step.dependOn(&run_exe_unit_tests.step);

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

    // Create build directory
    const build_dir = "vendor/solidity/build";
    std.fs.cwd().makeDir(build_dir) catch |err| switch (err) {
        error.PathAlreadyExists => {},
        else => return err,
    };

    // Configure CMake
    const cmake_configure = std.process.Child.run(.{
        .allocator = allocator,
        .argv = &[_][]const u8{
            "cmake",
            "-S",
            "vendor/solidity",
            "-B",
            build_dir,
            "-DCMAKE_BUILD_TYPE=Release",
            "-DONLY_BUILD_SOLIDITY_LIBRARIES=ON",
            "-DTESTS=OFF",
        },
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
    yul_wrapper.linkLibCpp();

    return yul_wrapper;
}

/// Link Solidity libraries to the executable
fn linkSolidityLibraries(b: *std.Build, exe: *std.Build.Step.Compile, cmake_step: *std.Build.Step) void {
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
    exe.linkLibCpp();

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
fn addBoostPaths(b: *std.Build, compile_step: *std.Build.Step.Compile, target: std.Build.ResolvedTarget) void {
    const target_info = target.result;

    switch (target_info.os.tag) {
        .macos => {
            // Check if Apple Silicon or Intel Mac
            if (target_info.cpu.arch == .aarch64) {
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
            if (target_info.cpu.arch == .x86_64) {
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
        std.log.info("Using BOOST_ROOT environment variable: {s}", .{root});
        const include_path = std.fmt.allocPrint(b.allocator, "{s}/include", .{root}) catch @panic("OOM");
        const lib_path = std.fmt.allocPrint(b.allocator, "{s}/lib", .{root}) catch @panic("OOM");
        compile_step.addSystemIncludePath(.{ .cwd_relative = include_path });
        compile_step.addLibraryPath(.{ .cwd_relative = lib_path });
    }
}
