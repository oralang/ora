const std = @import("std");

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});
    const voltaire_root = "../../../voltaire";

    const voltaire_dep = b.dependency("voltaire", .{
        .target = target,
        .optimize = optimize,
    });
    const vaxis_dep = b.dependency("vaxis", .{
        .target = target,
        .optimize = optimize,
    });
    const primitives_mod = voltaire_dep.module("primitives");
    const crypto_mod = voltaire_dep.module("crypto");
    const precompiles_mod = voltaire_dep.module("precompiles");
    const blockchain_mod = voltaire_dep.module("blockchain");
    const vaxis_mod = vaxis_dep.module("vaxis");

    const bootstrap_crypto = b.addSystemCommand(&.{
        "cargo",
        "build",
        "--manifest-path",
        b.pathJoin(&.{ voltaire_root, "Cargo.toml" }),
        "--release",
    });
    bootstrap_crypto.setName("bootstrap-voltaire-crypto");

    const evm_mod = b.addModule("ora_evm", .{
        .root_source_file = b.path("src/root.zig"),
        .target = target,
        .optimize = optimize,
        .imports = &.{
            .{ .name = "voltaire", .module = primitives_mod },
            .{ .name = "crypto", .module = crypto_mod },
            .{ .name = "precompiles", .module = precompiles_mod },
        },
    });

    const debug_probe_exe = b.addExecutable(.{
        .name = "ora-evm-debug-probe",
        .root_module = b.createModule(.{
            .root_source_file = b.path("src/debug_probe.zig"),
            .target = target,
            .optimize = optimize,
            .imports = &.{
                .{ .name = "ora_evm", .module = evm_mod },
                .{ .name = "voltaire", .module = primitives_mod },
                .{ .name = "crypto", .module = crypto_mod },
                .{ .name = "precompiles", .module = precompiles_mod },
            },
        }),
    });
    debug_probe_exe.step.dependOn(&bootstrap_crypto.step);
    b.installArtifact(debug_probe_exe);
    const run_debug_probe = b.addRunArtifact(debug_probe_exe);
    if (b.args) |args| run_debug_probe.addArgs(args);

    const debug_tui_exe = b.addExecutable(.{
        .name = "ora-evm-debug-tui",
        .root_module = b.createModule(.{
            .root_source_file = b.path("src/debug_tui.zig"),
            .target = target,
            .optimize = optimize,
            .imports = &.{
                .{ .name = "ora_evm", .module = evm_mod },
                .{ .name = "voltaire", .module = primitives_mod },
                .{ .name = "crypto", .module = crypto_mod },
                .{ .name = "precompiles", .module = precompiles_mod },
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
                .{ .name = "voltaire", .module = primitives_mod },
                .{ .name = "crypto", .module = crypto_mod },
                .{ .name = "precompiles", .module = precompiles_mod },
                .{ .name = "vaxis", .module = vaxis_mod },
            },
        }),
    });
    debug_dap_exe.step.dependOn(&bootstrap_crypto.step);
    b.installArtifact(debug_dap_exe);
    const run_debug_dap = b.addRunArtifact(debug_dap_exe);
    if (b.args) |args| run_debug_dap.addArgs(args);

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
    spec_mod.addImport("voltaire", primitives_mod);
    spec_mod.addImport("client_blockchain", blockchain_mod);

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

    const debug_tui_step = b.step("debug-tui", "Run the Ora EVM debugger TUI against emitted bytecode");
    debug_tui_step.dependOn(&run_debug_tui.step);

    const debug_dap_step = b.step("debug-dap", "Run the Ora EVM DAP server (Content-Length-framed JSON-RPC over stdio)");
    debug_dap_step.dependOn(&run_debug_dap.step);

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
                .{ .name = "voltaire", .module = primitives_mod },
                .{ .name = "crypto", .module = crypto_mod },
                .{ .name = "precompiles", .module = precompiles_mod },
            },
        }),
    });
    step_bench_exe.step.dependOn(&bootstrap_crypto.step);
    const run_step_bench = b.addRunArtifact(step_bench_exe);

    const bench_step = b.step("bench", "Run Ora EVM debugger per-step benchmark");
    bench_step.dependOn(&run_step_bench.step);
}
