const std = @import("std");

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});
    const voltaire_root = "../../../voltaire";

    const voltaire_dep = b.dependency("voltaire", .{
        .target = target,
        .optimize = optimize,
    });
    const primitives_mod = voltaire_dep.module("primitives");
    const crypto_mod = voltaire_dep.module("crypto");
    const precompiles_mod = voltaire_dep.module("precompiles");
    const blockchain_mod = voltaire_dep.module("blockchain");

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
}
