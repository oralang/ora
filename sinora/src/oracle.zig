const std = @import("std");
const builtin = @import("builtin");

const debug_codegen = @import("debug_codegen.zig");
const diagnostics = @import("diagnostics.zig");
const legality = @import("legality.zig");
const parser = @import("parser.zig");
const release_generic_backend = @import("release_generic_backend.zig");

pub const BackendMode = enum {
    debug,
    /// Legacy shape-specific release backend (`release_codegen.zig`).
    release,
    /// Generic Plank-port release backend (`release_generic_backend.zig`).
    /// Compared against the same Rust Plank `--release` bytecode oracle.
    release_generic,

    pub fn label(self: BackendMode) []const u8 {
        return switch (self) {
            .debug => "debug",
            .release => "release",
            .release_generic => "release-generic",
        };
    }
};

pub const Classification = enum {
    both_accept_codegen_pending,
    both_accept_bytecode_equal,
    bytecode_mismatch,
    zig_codegen_rejected,
    zig_rejected,
    rust_rejected,
    both_rejected,

    pub fn label(self: Classification) []const u8 {
        return switch (self) {
            .both_accept_codegen_pending => "both-accept-codegen-pending",
            .both_accept_bytecode_equal => "both-accept-bytecode-equal",
            .bytecode_mismatch => "bytecode-mismatch",
            .zig_codegen_rejected => "zig-codegen-rejected",
            .zig_rejected => "zig-rejected",
            .rust_rejected => "rust-rejected",
            .both_rejected => "both-rejected",
        };
    }

    pub fn parse(text: []const u8) ?Classification {
        const values = [_]Classification{
            .both_accept_codegen_pending,
            .both_accept_bytecode_equal,
            .bytecode_mismatch,
            .zig_codegen_rejected,
            .zig_rejected,
            .rust_rejected,
            .both_rejected,
        };
        for (values) |value| {
            if (std.mem.eql(u8, text, value.label())) return value;
        }
        return null;
    }
};

pub const ZigCheck = struct {
    accepted: bool,
    diagnostics: []const diagnostics.Diagnostic,
};

pub const DebugBytecodeComparison = struct {
    zig_accepted: bool,
    equal: ?bool = null,
    zig_stdout_bytes: usize = 0,
    rust_stdout_bytes: usize = 0,
};

pub const RustPlankResult = struct {
    accepted: bool,
    term: std.process.Child.Term,
    stdout: []const u8,
    stderr: []const u8,

    pub fn deinit(self: RustPlankResult, allocator: std.mem.Allocator) void {
        allocator.free(self.stdout);
        allocator.free(self.stderr);
    }
};

pub const Comparison = struct {
    classification: Classification,
    zig: ZigCheck,
    rust: ?RustPlankResult,
    debug_bytecode: ?DebugBytecodeComparison = null,
    release_bytecode: ?DebugBytecodeComparison = null,

    pub fn deinit(self: Comparison, allocator: std.mem.Allocator) void {
        if (self.rust) |rust| rust.deinit(allocator);
    }
};

pub fn compareSirFile(
    allocator: std.mem.Allocator,
    io: std.Io,
    sir_path: []const u8,
    rust_plank_path: []const u8,
    mode: BackendMode,
) !Comparison {
    const source = try std.Io.Dir.cwd().readFileAlloc(io, sir_path, allocator, std.Io.Limit.limited(64 * 1024 * 1024));
    defer allocator.free(source);

    var bag = diagnostics.Bag.init(allocator);
    defer bag.deinit();

    var generated_debug_hex: ?[]const u8 = null;
    defer if (generated_debug_hex) |hex| allocator.free(hex);
    var generated_release_hex: ?[]const u8 = null;
    defer if (generated_release_hex) |hex| allocator.free(hex);

    var debug_bytecode: ?DebugBytecodeComparison = null;
    var release_bytecode: ?DebugBytecodeComparison = null;
    const zig_accepted = blk: {
        var program = parser.parse(allocator, source, &bag) catch break :blk false;
        defer program.deinit();
        legality.validate(allocator, program, &bag) catch break :blk false;
        switch (mode) {
            .debug => {
                debug_bytecode = .{ .zig_accepted = false };
                const bytes = debug_codegen.emitFunction(allocator, program, "init", &bag) catch |err| {
                    if (err == error.FunctionNotFound) {
                        try bag.errorAt(0, 1, "function 'init' not found", .{});
                    }
                    break :blk true;
                };
                defer allocator.free(bytes);
                generated_debug_hex = try allocHexOutput(allocator, bytes);
                debug_bytecode = .{
                    .zig_accepted = true,
                    .zig_stdout_bytes = generated_debug_hex.?.len + 1,
                };
            },
            // Both labels now run the generic Plank-port backend (the legacy
            // `release_codegen.zig` shape scaffold was deleted once the generic
            // path reached full corpus byte parity). `.release_generic` is kept
            // as an alias for the migration gate.
            .release, .release_generic => {
                release_bytecode = .{ .zig_accepted = false };
                // Any non-OOM emission failure is an honest codegen gap and is
                // classified as pending (release_bytecode = null), never as a
                // hard mismatch. Only emitted-but-wrong bytecode is a mismatch.
                const bytes = release_generic_backend.emitRelease(allocator, program) catch |err| switch (err) {
                    error.OutOfMemory => return err,
                    else => {
                        release_bytecode = null;
                        break :blk true;
                    },
                };
                defer allocator.free(bytes);
                generated_release_hex = try allocHexOutput(allocator, bytes);
                release_bytecode = .{
                    .zig_accepted = true,
                    .zig_stdout_bytes = generated_release_hex.?.len + 1,
                };
            },
        }
        break :blk true;
    };

    const diagnostic_snapshot = try cloneDiagnostics(allocator, bag.items.items);
    errdefer freeDiagnostics(allocator, diagnostic_snapshot);

    const rust = try runRustPlank(allocator, io, rust_plank_path, sir_path, mode);
    errdefer rust.deinit(allocator);

    if (debug_bytecode) |*debug| {
        debug.rust_stdout_bytes = rust.stdout.len;
        if (debug.zig_accepted and rust.accepted) {
            const rust_hex = std.mem.trim(u8, rust.stdout, " \t\r\n");
            debug.equal = std.mem.eql(u8, generated_debug_hex.?, rust_hex);
        }
    }
    if (release_bytecode) |*release| {
        release.rust_stdout_bytes = rust.stdout.len;
        if (release.zig_accepted and rust.accepted) {
            const rust_hex = std.mem.trim(u8, rust.stdout, " \t\r\n");
            release.equal = std.mem.eql(u8, generated_release_hex.?, rust_hex);
        }
    }

    const bytecode_comparison = switch (mode) {
        .debug => debug_bytecode,
        .release, .release_generic => release_bytecode,
    };
    const classification = classify(zig_accepted, rust.accepted, bytecode_comparison);
    return .{
        .classification = classification,
        .zig = .{
            .accepted = zig_accepted,
            .diagnostics = diagnostic_snapshot,
        },
        .rust = rust,
        .debug_bytecode = debug_bytecode,
        .release_bytecode = release_bytecode,
    };
}

pub fn classify(zig_accepted: bool, rust_accepted: bool, debug_bytecode: ?DebugBytecodeComparison) Classification {
    if (zig_accepted and rust_accepted) {
        if (debug_bytecode) |debug| {
            if (!debug.zig_accepted) return .zig_codegen_rejected;
            if (debug.equal) |equal| {
                return if (equal) .both_accept_bytecode_equal else .bytecode_mismatch;
            }
        }
        return .both_accept_codegen_pending;
    }
    if (!zig_accepted and !rust_accepted) return .both_rejected;
    if (!zig_accepted) return .zig_rejected;
    return .rust_rejected;
}

pub fn resolveRustPlankPath(allocator: std.mem.Allocator, io: std.Io) ![]const u8 {
    if (libcEnv("SINORA_PLANK_SIR")) |path| {
        if (canAccess(io, path)) return allocator.dupe(u8, path);
        return error.PlankSirNotFound;
    }
    if (libcEnv("ORA_PLANK_SIR")) |path| {
        if (canAccess(io, path)) return allocator.dupe(u8, path);
        return error.PlankSirNotFound;
    }

    const candidates = [_][]const u8{
        "vendor/plank/plankc/target/release/sir",
        "../vendor/plank/plankc/target/release/sir",
    };
    for (candidates) |candidate| {
        if (canAccess(io, candidate)) return allocator.dupe(u8, candidate);
    }
    return error.PlankSirNotFound;
}

pub fn runRustPlank(
    allocator: std.mem.Allocator,
    io: std.Io,
    rust_plank_path: []const u8,
    sir_path: []const u8,
    mode: BackendMode,
) !RustPlankResult {
    var argv_buf: [3][]const u8 = undefined;
    var argc: usize = 0;
    argv_buf[argc] = rust_plank_path;
    argc += 1;
    argv_buf[argc] = sir_path;
    argc += 1;
    // Both release backends are scored against the same Rust Plank `--release`
    // oracle; only debug mode omits the flag.
    if (mode != .debug) {
        argv_buf[argc] = "--release";
        argc += 1;
    }

    const max_output = 16 * 1024 * 1024;
    var process_io = std.Io.Threaded.init(allocator, .{
        .async_limit = .nothing,
        .concurrent_limit = .nothing,
    });
    defer process_io.deinit();

    _ = io;
    const run_result = try std.process.run(allocator, process_io.io(), .{
        .argv = argv_buf[0..argc],
        .stdout_limit = std.Io.Limit.limited(max_output),
        .stderr_limit = std.Io.Limit.limited(max_output),
    });
    errdefer allocator.free(run_result.stdout);
    errdefer allocator.free(run_result.stderr);

    return .{
        .accepted = switch (run_result.term) {
            .exited => |code| code == 0,
            else => false,
        },
        .term = run_result.term,
        .stdout = run_result.stdout,
        .stderr = run_result.stderr,
    };
}

pub fn freeDiagnostics(allocator: std.mem.Allocator, items: []const diagnostics.Diagnostic) void {
    for (items) |item| {
        allocator.free(item.message);
    }
    allocator.free(items);
}

fn cloneDiagnostics(allocator: std.mem.Allocator, items: []const diagnostics.Diagnostic) ![]const diagnostics.Diagnostic {
    const cloned = try allocator.alloc(diagnostics.Diagnostic, items.len);
    errdefer allocator.free(cloned);
    for (items, 0..) |item, index| {
        cloned[index] = item;
        cloned[index].message = try allocator.dupe(u8, item.message);
    }
    return cloned;
}

fn allocHexOutput(allocator: std.mem.Allocator, bytes: []const u8) ![]const u8 {
    const hex = "0123456789abcdef";
    const output = try allocator.alloc(u8, 2 + bytes.len * 2);
    output[0] = '0';
    output[1] = 'x';
    for (bytes, 0..) |byte, index| {
        output[2 + index * 2] = hex[byte >> 4];
        output[2 + index * 2 + 1] = hex[byte & 0x0f];
    }
    return output;
}

fn canAccess(io: std.Io, path: []const u8) bool {
    return (std.Io.Dir.cwd().access(io, path, .{}) catch null) != null;
}

fn libcEnv(comptime name: [:0]const u8) ?[]const u8 {
    if (!builtin.link_libc) return null;
    return if (std.c.getenv(name)) |value| std.mem.span(value) else null;
}

test "classifies current migration states honestly" {
    try std.testing.expectEqual(Classification.both_accept_codegen_pending, classify(true, true, null));
    try std.testing.expectEqual(Classification.both_accept_bytecode_equal, classify(true, true, .{
        .zig_accepted = true,
        .equal = true,
    }));
    try std.testing.expectEqual(Classification.bytecode_mismatch, classify(true, true, .{
        .zig_accepted = true,
        .equal = false,
    }));
    try std.testing.expectEqual(Classification.zig_codegen_rejected, classify(true, true, .{
        .zig_accepted = false,
    }));
    try std.testing.expectEqual(Classification.zig_rejected, classify(false, true, null));
    try std.testing.expectEqual(Classification.rust_rejected, classify(true, false, null));
    try std.testing.expectEqual(Classification.both_rejected, classify(false, false, null));
}

test "classification parses stable labels" {
    try std.testing.expectEqual(Classification.both_accept_bytecode_equal, Classification.parse("both-accept-bytecode-equal").?);
    try std.testing.expectEqual(@as(?Classification, null), Classification.parse("unknown"));
}

test "allocates normalized hex output" {
    const output = try allocHexOutput(std.testing.allocator, &.{ 0x00, 0xab, 0xff });
    defer std.testing.allocator.free(output);
    try std.testing.expectEqualStrings("0x00abff", output);
}
