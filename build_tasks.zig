// Native build actions run as ordinary processes: Zig 0.17 has no custom make callbacks.
const std = @import("std");
const LLVM_REPO_URL = "https://github.com/llvm/llvm-project.git";
const LLVM_COMMIT = "ee8c14be14deabace692ab51f5d5d432b0a83d58";

pub const NativeSanitizer = enum {
    none,
    address,
    undefined,
    address_undefined,

    pub fn enabled(self: NativeSanitizer) bool {
        return self != .none;
    }

    pub fn suffix(self: NativeSanitizer) []const u8 {
        return switch (self) {
            .none => "release",
            .address => "asan",
            .undefined => "ubsan",
            .address_undefined => "asan-ubsan",
        };
    }

    pub fn cmakeFlags(self: NativeSanitizer) []const u8 {
        return switch (self) {
            .none => "",
            .address => "-fsanitize=address -fno-omit-frame-pointer -g",
            .undefined => "-fsanitize=undefined -fno-omit-frame-pointer -g",
            .address_undefined => "-fsanitize=address,undefined -fno-omit-frame-pointer -g",
        };
    }
};

const Context = struct {
    allocator: std.mem.Allocator,
    io: std.Io,
    environ_map: *const std.process.Environ.Map,
    root_path: []const u8,
    fn fmt(b: *const Context, comptime format: []const u8, args: anytype) []const u8 {
        return std.fmt.allocPrint(b.allocator, format, args) catch @panic("OOM");
    }
};

pub fn main(init: std.process.Init) !void {
    const allocator = init.arena.allocator();
    const args = try init.minimal.args.toSlice(allocator);
    if (args.len != 4) return error.InvalidBuildTaskArguments;
    const sanitizer = std.meta.stringToEnum(NativeSanitizer, args[3]) orelse return error.InvalidSanitizer;
    var context: Context = .{ .allocator = allocator, .io = init.io, .environ_map = init.environ_map, .root_path = args[2] };
    const Task = enum { mlir, sir, ora, z3, examples };
    switch (std.meta.stringToEnum(Task, args[1]) orelse return error.UnknownBuildTask) {
        .mlir => try buildMlirLibrariesImpl(&context),
        .sir => try buildSIRDialectLibraryImpl(&context, sanitizer),
        .ora => try buildOraDialectLibraryImpl(&context, sanitizer),
        .z3 => try buildZ3LibrariesImpl(&context),
        .examples => try runExampleTests(&context),
    }
}

pub fn parseNativeSanitizer(value: []const u8) NativeSanitizer {
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

pub fn nativeMlirInstallPrefix(b: anytype, sanitizer: NativeSanitizer) []const u8 {
    return if (sanitizer.enabled())
        b.fmt("vendor/mlir-{s}", .{sanitizer.suffix()})
    else
        "vendor/mlir";
}

pub fn nativeDialectBuildDir(b: anytype, base: []const u8, sanitizer: NativeSanitizer) []const u8 {
    return if (sanitizer.enabled())
        b.fmt("{s}-{s}", .{ base, sanitizer.suffix() })
    else
        base;
}

pub fn joinCmakeFlags(b: anytype, base: []const u8, extra: []const u8) []const u8 {
    if (base.len == 0) return extra;
    if (extra.len == 0) return base;
    return b.fmt("{s} {s}", .{ base, extra });
}

pub fn appendCmakeDefine(cmake_args: *std.array_list.Managed([]const u8), b: anytype, name: []const u8, value: []const u8) !void {
    try cmake_args.append(b.fmt("-D{s}={s}", .{ name, value }));
}

pub fn appendNativeCmakeToolchainFlags(cmake_args: *std.array_list.Managed([]const u8), b: anytype, sanitizer: NativeSanitizer) !void {
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

pub fn pathExists(io: std.Io, path: []const u8) bool {
    std.Io.Dir.cwd().access(io, path, .{}) catch return false;
    return true;
}

pub fn anyPathExists(io: std.Io, paths: []const []const u8) bool {
    for (paths) |path| {
        if (pathExists(io, path)) return true;
    }
    return false;
}

pub fn hostHasSystemZ3(io: std.Io) bool {
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

    return anyPathExists(io, &header_paths) and anyPathExists(io, &lib_paths);
}

fn runInheritedProcess(b: *Context, argv: []const []const u8, cwd: []const u8) !std.process.Child.Term {
    var child = try std.process.spawn(b.io, .{
        .argv = argv,
        .cwd = .{ .path = cwd },
        .environ_map = b.environ_map,
        .stdin = .inherit,
        .stdout = .inherit,
        .stderr = .inherit,
    });
    return try child.wait(b.io);
}

fn buildMlirLibrariesImpl(b: *Context) anyerror!void {
    const allocator = b.allocator;

    const root = std.Io.Dir.cwd();
    const build_root = b.root_path;

    // skip if MLIR C API library is already installed
    const mlir_c_lib = if (@import("builtin").os.tag == .macos) "vendor/mlir/lib/libMLIR-C.dylib" else "vendor/mlir/lib/libMLIR-C.so";
    if (root.access(b.io, mlir_c_lib, .{}) catch null) |_| {
        std.log.info("MLIR libraries already installed, skipping build (delete vendor/mlir to force rebuild)", .{});
        return;
    }

    // CMake is configured against llvm-project/llvm, not just the checkout root.
    _ = root.openDir(b.io, "vendor/llvm-project/llvm", .{ .iterate = false }) catch {
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
    try root.createDirPath(b.io, build_dir);

    const install_prefix = "vendor/mlir";
    try root.createDirPath(b.io, install_prefix);

    const builtin = @import("builtin");
    var cmake_args = std.array_list.Managed([]const u8).init(allocator);
    defer cmake_args.deinit();

    // prefer Ninja generator when available for faster, more parallel builds
    var use_ninja: bool = false;
    {
        const probe = std.process.run(allocator, b.io, .{ .argv = &[_][]const u8{ "ninja", "--version" }, .cwd = .{ .path = "." } }) catch null;
        if (probe) |res| {
            switch (res.term) {
                .exited => |code| {
                    if (code == 0) use_ninja = true;
                },
                else => {},
            }
        }
        if (!use_ninja) {
            const probe_alt = std.process.run(allocator, b.io, .{ .argv = &[_][]const u8{ "ninja-build", "--version" }, .cwd = .{ .path = "." } }) catch null;
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
        const sdk_path_result = std.process.run(allocator, b.io, .{
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

        if (b.environ_map.get("ORA_CMAKE_OSX_ARCH")) |arch| {
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

fn buildOraDialectLibraryImpl(b: *Context, sanitizer: NativeSanitizer) anyerror!void {
    const allocator = b.allocator;
    const root = std.Io.Dir.cwd();
    const build_root = b.root_path;

    const build_dir = nativeDialectBuildDir(b, "vendor/ora-dialect-build", sanitizer);
    try root.createDirPath(b.io, build_dir);

    const install_prefix = nativeMlirInstallPrefix(b, sanitizer);
    const install_prefix_abs = b.fmt("{s}/{s}", .{ build_root, install_prefix });
    const mlir_dir = b.fmt("{s}/vendor/mlir/lib/cmake/mlir", .{build_root});

    var cmake_args = std.array_list.Managed([]const u8).init(allocator);
    defer cmake_args.deinit();

    // prefer Ninja generator when available
    var use_ninja: bool = false;
    {
        const probe = std.process.run(allocator, b.io, .{ .argv = &[_][]const u8{ "ninja", "--version" }, .cwd = .{ .path = "." } }) catch null;
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

fn buildSIRDialectLibraryImpl(b: *Context, sanitizer: NativeSanitizer) anyerror!void {
    const allocator = b.allocator;
    const root = std.Io.Dir.cwd();
    const build_root = b.root_path;

    const build_dir = nativeDialectBuildDir(b, "vendor/sir-dialect-build", sanitizer);
    try root.createDirPath(b.io, build_dir);

    const install_prefix = nativeMlirInstallPrefix(b, sanitizer);
    const install_prefix_abs = b.fmt("{s}/{s}", .{ build_root, install_prefix });
    const mlir_dir = b.fmt("{s}/vendor/mlir/lib/cmake/mlir", .{build_root});

    var cmake_args = std.array_list.Managed([]const u8).init(allocator);
    defer cmake_args.deinit();

    // prefer Ninja generator when available
    var use_ninja: bool = false;
    {
        const probe = std.process.run(allocator, b.io, .{ .argv = &[_][]const u8{ "ninja", "--version" }, .cwd = .{ .path = "." } }) catch null;
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

fn runExampleTests(b: *Context) anyerror!void {
    const allocator = b.allocator;

    std.log.info("Testing all .ora example files...", .{});

    // get examples directory
    var examples_dir = std.Io.Dir.cwd().openDir(b.io, "ora-example", .{ .iterate = true }) catch |err| {
        std.log.err("Failed to open examples directory: {}", .{err});
        return err;
    };
    defer examples_dir.close(b.io);

    // iterate through all .ora files
    var walker = examples_dir.walk(allocator) catch |err| {
        std.log.err("Failed to walk examples directory: {}", .{err});
        return err;
    };
    defer walker.deinit();

    var tested_count: u32 = 0;
    var failed_count: u32 = 0;

    while (walker.next(b.io) catch |err| {
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
            const result = std.process.run(allocator, b.io, .{
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

fn buildZ3LibrariesImpl(b: *Context) anyerror!void {
    const allocator = b.allocator;

    if (hostHasSystemZ3(b.io)) {
        std.log.info("System Z3 headers and library detected; skipping vendored Z3 build", .{});
        return;
    }

    // system Z3 headers/libraries not found, try to build from vendor
    std.log.info("System Z3 not fully available, checking vendor/z3...", .{});

    const cwd = std.Io.Dir.cwd();
    _ = cwd.openDir(b.io, "vendor/z3", .{ .iterate = false }) catch {
        std.log.err("Z3 is required for compiler verification tests and no usable Z3 installation was found.", .{});
        std.log.err("Install Z3 or provide the vendored submodule:", .{});
        std.log.err("  macOS: brew install z3", .{});
        std.log.err("  Linux: sudo apt install z3 libz3-dev", .{});
        std.log.err("  vendor: git submodule update --init --recursive vendor/z3", .{});
        return error.Z3Unavailable;
    };

    // create build and install directories
    const build_dir = "vendor/z3/build-release";
    cwd.createDirPath(b.io, build_dir) catch |err| switch (err) {
        error.PathAlreadyExists => {},
        else => return err,
    };

    const install_prefix = "vendor/z3-install";
    cwd.createDirPath(b.io, install_prefix) catch |err| switch (err) {
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
        const probe = std.process.run(allocator, b.io, .{ .argv = &[_][]const u8{ "ninja", "--version" }, .cwd = .{ .path = "." } }) catch null;
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
        const sdk_path_result = std.process.run(allocator, b.io, .{
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
