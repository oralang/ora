const std = @import("std");

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});
    const debug_ui = b.option(bool, "debug-ui", "Build debugger TUI/DAP binaries that depend on vaxis") orelse false;
    const blst_lib = createBlstLibrary(b, target, optimize);
    const c_kzg_lib = createCKzgLibrary(b, target, optimize, blst_lib);
    const c_kzg_mod = b.addModule("c_kzg", .{
        .root_source_file = b.path("vendor/c-kzg-4844/bindings/zig/root.zig"),
        .target = target,
        .optimize = optimize,
    });
    c_kzg_mod.linkLibrary(c_kzg_lib);
    c_kzg_mod.linkLibrary(blst_lib);
    c_kzg_mod.addIncludePath(b.path("vendor/c-kzg-4844/src"));
    c_kzg_mod.addIncludePath(b.path("vendor/c-kzg-4844/blst/bindings"));
    const rust_crypto_lib_path = b.path("target/release/libora_evm_crypto_wrappers.a");

    const bootstrap_crypto = b.addSystemCommand(&.{
        "cargo",
        "build",
        "--manifest-path",
        b.path("Cargo.toml").getPath(b),
        "--release",
    });
    bootstrap_crypto.setName("bootstrap-ora-evm-crypto");

    const evm_mod = b.addModule("ora_evm", .{
        .root_source_file = b.path("src/root.zig"),
        .target = target,
        .optimize = optimize,
        .imports = &.{
            .{ .name = "c_kzg", .module = c_kzg_mod },
        },
    });
    evm_mod.addObjectFile(rust_crypto_lib_path);
    evm_mod.link_libc = true;

    const debug_probe_exe = b.addExecutable(.{
        .name = "ora-evm-debug-probe",
        .root_module = b.createModule(.{
            .root_source_file = b.path("src/debug_probe.zig"),
            .target = target,
            .optimize = optimize,
            .imports = &.{
                .{ .name = "ora_evm", .module = evm_mod },
            },
        }),
    });
    debug_probe_exe.step.dependOn(&bootstrap_crypto.step);
    b.installArtifact(debug_probe_exe);
    const run_debug_probe = b.addRunArtifact(debug_probe_exe);
    if (b.args) |args| run_debug_probe.addArgs(args);

    if (debug_ui) {
        const vaxis_dep = b.dependency("vaxis", .{
            .target = target,
            .optimize = optimize,
        });
        const vaxis_mod = vaxis_dep.module("vaxis");

        const debug_tui_exe = b.addExecutable(.{
            .name = "ora-evm-debug-tui",
            .root_module = b.createModule(.{
                .root_source_file = b.path("src/debug_tui.zig"),
                .target = target,
                .optimize = optimize,
                .imports = &.{
                    .{ .name = "ora_evm", .module = evm_mod },
                    .{ .name = "vaxis", .module = vaxis_mod },
                },
            }),
        });
        debug_tui_exe.step.dependOn(&bootstrap_crypto.step);
        b.installArtifact(debug_tui_exe);
        const run_debug_tui = b.addRunArtifact(debug_tui_exe);
        if (b.args) |args| run_debug_tui.addArgs(args);

        // DAP server (see src/debug_dap.zig). Imports debug_tui.zig
        // directly to reuse Session / SessionSeed / loadSeedFromConfig,
        // which transitively pulls in vaxis at compile time even though
        // the DAP binary doesn't render anything — cheaper than carving
        // a shared loader module while the typed boundary is still
        // moving.
        const debug_dap_exe = b.addExecutable(.{
            .name = "ora-evm-debug-dap",
            .root_module = b.createModule(.{
                .root_source_file = b.path("src/debug_dap.zig"),
                .target = target,
                .optimize = optimize,
                .imports = &.{
                    .{ .name = "ora_evm", .module = evm_mod },
                    .{ .name = "vaxis", .module = vaxis_mod },
                },
            }),
        });
        debug_dap_exe.step.dependOn(&bootstrap_crypto.step);
        b.installArtifact(debug_dap_exe);
        const run_debug_dap = b.addRunArtifact(debug_dap_exe);
        if (b.args) |args| run_debug_dap.addArgs(args);

        const debug_tui_step = b.step("debug-tui", "Run the Ora EVM debugger TUI against emitted bytecode");
        debug_tui_step.dependOn(&run_debug_tui.step);

        const debug_dap_step = b.step("debug-dap", "Run the Ora EVM DAP server (Content-Length-framed JSON-RPC over stdio)");
        debug_dap_step.dependOn(&run_debug_dap.step);
    } else {
        const debug_tui_disabled = b.addFail("debug-tui requires -Ddebug-ui=true");
        const debug_tui_step = b.step("debug-tui", "Run the Ora EVM debugger TUI against emitted bytecode");
        debug_tui_step.dependOn(&debug_tui_disabled.step);

        const debug_dap_disabled = b.addFail("debug-dap requires -Ddebug-ui=true");
        const debug_dap_step = b.step("debug-dap", "Run the Ora EVM DAP server (Content-Length-framed JSON-RPC over stdio)");
        debug_dap_step.dependOn(&debug_dap_disabled.step);
    }

    const unit_tests = b.addTest(.{
        .name = "ora-evm-unit-tests",
        .root_module = evm_mod,
    });
    unit_tests.step.dependOn(&bootstrap_crypto.step);
    const run_unit_tests = b.addRunArtifact(unit_tests);

    const spec_mod = b.createModule(.{
        .root_source_file = b.path("test/specs/root.zig"),
        .target = target,
        .optimize = optimize,
    });
    spec_mod.addImport("evm", evm_mod);

    const spec_tests = b.addTest(.{
        .name = "ora-evm-spec-tests",
        .root_module = spec_mod,
    });
    spec_tests.step.dependOn(&bootstrap_crypto.step);
    const run_spec_tests = b.addRunArtifact(spec_tests);

    const test_step = b.step("test", "Run Ora EVM unit and spec tests");
    test_step.dependOn(&run_unit_tests.step);
    test_step.dependOn(&run_spec_tests.step);

    const unit_step = b.step("unit", "Run Ora EVM unit tests");
    unit_step.dependOn(&run_unit_tests.step);

    const spec_step = b.step("spec", "Run Ora EVM execution spec tests");
    spec_step.dependOn(&run_spec_tests.step);

    const debug_probe_step = b.step("debug-probe", "Run the Ora EVM debugger probe against emitted bytecode");
    debug_probe_step.dependOn(&run_debug_probe.step);

    // Per-step debugger micro-benchmark. Tracks the per-step wall-clock cost
    // of stepOpcode + statement-boundary check; fails if it exceeds the
    // budget defined inside the bench (see test/bench/step_bench.zig). Run
    // with `zig build bench`.
    const step_bench_exe = b.addExecutable(.{
        .name = "ora-evm-step-bench",
        .root_module = b.createModule(.{
            .root_source_file = b.path("test/bench/step_bench.zig"),
            .target = target,
            .optimize = .ReleaseFast,
            .imports = &.{
                .{ .name = "ora_evm", .module = evm_mod },
            },
        }),
    });
    step_bench_exe.step.dependOn(&bootstrap_crypto.step);
    const run_step_bench = b.addRunArtifact(step_bench_exe);

    const bench_step = b.step("bench", "Run Ora EVM debugger per-step benchmark");
    bench_step.dependOn(&run_step_bench.step);
}

fn createBlstLibrary(
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
        blst_build_cmd.setCwd(b.path("vendor/c-kzg-4844/blst"));
        lib.step.dependOn(&blst_build_cmd.step);
        lib.root_module.addAssemblyFile(b.path("vendor/c-kzg-4844/blst/build/assembly.S"));
    }

    const c_flags = if (is_wasm)
        &[_][]const u8{ "-std=c99", "-D__BLST_PORTABLE__", "-D__BLST_NO_ASM__", "-fno-sanitize=undefined" }
    else
        &[_][]const u8{ "-std=c99", "-fPIC", "-D__BLST_PORTABLE__", "-fno-sanitize=undefined" };

    lib.root_module.addCSourceFiles(.{
        .files = &.{
            "vendor/c-kzg-4844/blst/src/server.c",
        },
        .flags = c_flags,
    });
    lib.root_module.addIncludePath(b.path("vendor/c-kzg-4844/blst/bindings"));
    return lib;
}

fn createCKzgLibrary(
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
            "vendor/c-kzg-4844/src/ckzg.c",
        },
        .flags = &.{ "-std=c99", "-fPIC", "-fno-sanitize=undefined" },
    });
    lib.root_module.addIncludePath(b.path("vendor/c-kzg-4844/src"));
    lib.root_module.addIncludePath(b.path("vendor/c-kzg-4844/blst/bindings"));
    return lib;
}
