const std = @import("std");
const c_kzg = @import("c_kzg");

var initialized: std.atomic.Value(bool) = std.atomic.Value(bool).init(false);
var init_mutex: std.Io.Mutex = .init;
var verify_mutex: std.Io.Mutex = .init;

pub fn init() !void {
    if (initialized.load(.acquire)) return;

    const io = std.Io.Threaded.global_single_threaded.io();
    init_mutex.lockUncancelable(io);
    defer init_mutex.unlock(io);

    if (initialized.load(.acquire)) return;

    c_kzg.loadTrustedSetupFromText(c_kzg.embedded_trusted_setup, 0) catch |err| {
        if (err != error.TrustedSetupAlreadyLoaded) return error.TrustedSetupLoadFailed;
    };
    initialized.store(true, .release);
}

pub fn deinit() void {
    const io = std.Io.Threaded.global_single_threaded.io();
    init_mutex.lockUncancelable(io);
    defer init_mutex.unlock(io);

    if (!initialized.load(.acquire)) return;
    c_kzg.freeTrustedSetup() catch |err| {
        if (err != error.TrustedSetupNotLoaded) @panic("unexpected KZG trusted setup cleanup failure");
    };
    initialized.store(false, .release);
}

pub fn isInitialized() bool {
    return initialized.load(.acquire);
}

pub fn verifyProof(
    commitment: *const c_kzg.KZGCommitment,
    z: *const c_kzg.Bytes32,
    y: *const c_kzg.Bytes32,
    proof: *const c_kzg.KZGProof,
) !bool {
    const io = std.Io.Threaded.global_single_threaded.io();
    verify_mutex.lockUncancelable(io);
    defer verify_mutex.unlock(io);

    return try c_kzg.verifyKZGProof(commitment, z, y, proof);
}

test "KZG setup initializes idempotently" {
    try init();
    try std.testing.expect(isInitialized());
    try init();
    try std.testing.expect(isInitialized());
    deinit();
}
