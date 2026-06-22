const std = @import("std");

pub fn build(b: *std.Build) void {
    // ============================================================
    // 1. Standard Build Options
    // ============================================================
    // Target configuration (native by default, can be cross-compiled)
    const target = b.standardTargetOptions(.{});

    // Optimization level (Debug, ReleaseSafe, ReleaseFast, ReleaseSmall)
    const optimize = b.standardOptimizeOption(.{});

    // ============================================================
    // 2. Main Module Definition
    // ============================================================
    // Create the main library module that can be imported by other projects
    const mod = b.addModule("z_ens_normalize", .{
        .root_source_file = b.path("src/root.zig"),
        .target = target,
        .optimize = optimize,
    });

    // ============================================================
    // 3. Static Library Build (for distribution)
    // ============================================================
    // Build a static library artifact that can be linked into other projects
    const lib = b.addLibrary(.{
        .name = "z_ens_normalize",
        .root_module = mod,
        .linkage = .static,
    });

    // Install the library to zig-out/lib/
    b.installArtifact(lib);

    // ============================================================
    // 4. Test Configuration
    // ============================================================
    // Create test executable that runs all unit tests in the project
    const mod_tests = b.addTest(.{
        .root_module = mod,
    });

    // Create a run step for the tests
    const run_mod_tests = b.addRunArtifact(mod_tests);

    // Define the test step that users can run with "zig build test"
    const test_step = b.step("test", "Run all unit tests");
    test_step.dependOn(&run_mod_tests.step);

    // ============================================================
    // 5. External Test Files (tests/ directory)
    // ============================================================
    // Add external test files that import the main module
    // These tests can access the module via @import("z_ens_normalize")

    // Create test modules
    const ensip15_test_mod = b.createModule(.{
        .root_source_file = b.path("tests/ensip15_test.zig"),
        .target = target,
        .optimize = optimize,
    });
    ensip15_test_mod.addImport("z_ens_normalize", mod);

    const nf_test_mod = b.createModule(.{
        .root_source_file = b.path("tests/nf_test.zig"),
        .target = target,
        .optimize = optimize,
    });
    nf_test_mod.addImport("z_ens_normalize", mod);

    const init_test_mod = b.createModule(.{
        .root_source_file = b.path("tests/init_test.zig"),
        .target = target,
        .optimize = optimize,
    });
    init_test_mod.addImport("z_ens_normalize", mod);

    // ENSIP15 normalization tests
    const ensip15_tests = b.addTest(.{
        .root_module = ensip15_test_mod,
    });
    const run_ensip15_tests = b.addRunArtifact(ensip15_tests);
    test_step.dependOn(&run_ensip15_tests.step);

    // NF normalization tests
    const nf_tests = b.addTest(.{
        .root_module = nf_test_mod,
    });
    const run_nf_tests = b.addRunArtifact(nf_tests);
    test_step.dependOn(&run_nf_tests.step);

    // Init data loading tests
    const init_tests = b.addTest(.{
        .root_module = init_test_mod,
    });
    const run_init_tests = b.addRunArtifact(init_tests);
    test_step.dependOn(&run_init_tests.step);

    // ============================================================
    // 5. Debug Executables
    // ============================================================
    const debug_mod = b.createModule(.{
        .root_source_file = b.path("test_decomp.zig"),
        .target = target,
        .optimize = optimize,
    });
    debug_mod.addImport("z_ens_normalize", mod);

    const debug_exe = b.addExecutable(.{
        .name = "test_decomp",
        .root_module = debug_mod,
    });
    b.installArtifact(debug_exe);

    const excl_mod = b.createModule(.{
        .root_source_file = b.path("test_excl.zig"),
        .target = target,
        .optimize = optimize,
    });
    excl_mod.addImport("z_ens_normalize", mod);

    const excl_exe = b.addExecutable(.{
        .name = "test_excl",
        .root_module = excl_mod,
    });
    b.installArtifact(excl_exe);

    // ============================================================
    // 6. C FFI Library
    // ============================================================
    // Build C-compatible library with exported C functions
    const c_mod = b.addModule("z_ens_normalize_c", .{
        .root_source_file = b.path("src/root_c.zig"),
        .target = target,
        .optimize = optimize,
    });
    c_mod.addImport("z_ens_normalize", mod);

    const c_lib = b.addLibrary(.{
        .name = "z_ens_normalize_c",
        .root_module = c_mod,
        .linkage = .static,
    });
    c_lib.root_module.link_libc = true;

    // Add build step for C library
    const c_lib_step = b.step("c-lib", "Build C FFI library");
    c_lib_step.dependOn(&b.addInstallArtifact(c_lib, .{}).step);

    // Generate C header file
    const c_header = b.addInstallFile(
        b.path("include/z_ens_normalize.h"),
        "include/z_ens_normalize.h",
    );
    c_lib_step.dependOn(&c_header.step);

    // ============================================================
    // 7. WebAssembly Builds
    // ============================================================
    // WASM freestanding build (for web browsers, Node.js)
    const wasm_target = b.resolveTargetQuery(.{
        .cpu_arch = .wasm32,
        .os_tag = .freestanding,
    });

    const wasm_mod = b.addModule("z_ens_normalize_wasm", .{
        .root_source_file = b.path("src/root_c.zig"),
        .target = wasm_target,
        .optimize = optimize,
    });
    wasm_mod.addImport("z_ens_normalize", mod);

    const wasm_lib = b.addExecutable(.{
        .name = "z_ens_normalize",
        .root_module = wasm_mod,
    });
    wasm_lib.entry = .disabled;
    wasm_lib.rdynamic = true;

    // Add build step for WASM
    const wasm_step = b.step("wasm", "Build WebAssembly module (freestanding)");
    wasm_step.dependOn(&b.addInstallArtifact(wasm_lib, .{}).step);

    // WASI build (WebAssembly System Interface)
    const wasi_target = b.resolveTargetQuery(.{
        .cpu_arch = .wasm32,
        .os_tag = .wasi,
    });

    const wasi_mod = b.addModule("z_ens_normalize_wasi", .{
        .root_source_file = b.path("src/root_c.zig"),
        .target = wasi_target,
        .optimize = optimize,
    });
    wasi_mod.addImport("z_ens_normalize", mod);

    const wasi_lib = b.addExecutable(.{
        .name = "z_ens_normalize_wasi",
        .root_module = wasi_mod,
    });
    wasi_lib.entry = .disabled;
    wasi_lib.rdynamic = true;

    // Add build step for WASI
    const wasi_step = b.step("wasi", "Build WebAssembly module (WASI)");
    wasi_step.dependOn(&b.addInstallArtifact(wasi_lib, .{}).step);

    // Build both WASM variants
    const wasm_all_step = b.step("wasm-all", "Build all WebAssembly variants");
    wasm_all_step.dependOn(wasm_step);
    wasm_all_step.dependOn(wasi_step);

    // ============================================================
    // 8. Test Data Copy Step
    // ============================================================
    // Create a step to copy test data files (JSON files) to zig-out/test-data/
    // This is useful for tests that need to load external data files
    const copy_test_data = b.step("copy-test-data", "Copy test data files to zig-out/test-data/");

    // Check if test-data directory exists before attempting to copy
    const test_data_dir = "test-data";
    const install_subdir = "test-data";

    // Use installDirectory to copy all files from test-data/ to zig-out/test-data/
    // This will only execute if the source directory exists
    const test_data_exists = checkDirExists(b, test_data_dir);
    if (test_data_exists) {
        const install_test_data = b.addInstallDirectory(.{
            .source_dir = b.path(test_data_dir),
            .install_dir = .prefix,
            .install_subdir = install_subdir,
        });
        copy_test_data.dependOn(&install_test_data.step);

        // Optionally make tests depend on test data being copied
        // Uncomment the line below if tests require the data files
        // test_step.dependOn(copy_test_data);
    }
}

// ============================================================
// Helper Functions
// ============================================================

/// Check if a directory exists at the given path
fn checkDirExists(b: *std.Build, path: []const u8) bool {
    var dir = std.Io.Dir.cwd().openDir(b.graph.io, path, .{}) catch return false;
    defer dir.close(b.graph.io);
    return true;
}
