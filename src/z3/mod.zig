//===----------------------------------------------------------------------===//
//
// Z3 Formal Verification Module
//
//===----------------------------------------------------------------------===//
//
// This module provides Z3 SMT solver integration for formal verification
// of Ora smart contracts. It includes:
//
// - Context and solver management
// - MLIR to SMT encoding
// - Verification condition generation
// - Counterexample extraction and reporting
//
//===----------------------------------------------------------------------===//

/// Z3 C API bindings
pub const c = @import("c.zig");

/// Context management for Z3 solver
pub const context = @import("context.zig");

/// SMT encoding utilities
pub const encoder = @import("encoder.zig");

/// Solver interface and query management
pub const solver = @import("solver.zig");

/// Verification pass coordinator
pub const verification = @import("verification.zig");

/// Error handling and counterexample reporting
pub const errors = @import("errors.zig");

/// Re-export commonly used types
pub const Context = context.Context;
pub const Solver = solver.Solver;
pub const Encoder = encoder.Encoder;
pub const VerificationPass = verification.VerificationPass;
pub const VerificationResult = errors.VerificationResult;
pub const VerificationError = errors.VerificationError;
pub const SmtReportArtifacts = verification.SmtReportArtifacts;

//===----------------------------------------------------------------------===//
// Version and Feature Detection
//===----------------------------------------------------------------------===//

/// Check if Z3 is available at runtime
pub fn isZ3Available() bool {
    // try to create a Z3 context
    const cfg = c.Z3_mk_config();
    if (cfg == null) return false;
    defer c.Z3_del_config(cfg);

    const ctx = c.Z3_mk_context(cfg);
    if (ctx == null) return false;
    defer c.Z3_del_context(ctx);

    return true;
}

/// Get Z3 version information
pub fn getZ3Version() ?struct { major: u32, minor: u32, build: u32, revision: u32 } {
    var major: c_uint = 0;
    var minor: c_uint = 0;
    var build: c_uint = 0;
    var revision: c_uint = 0;

    c.c.Z3_get_version(&major, &minor, &build, &revision);

    return .{
        .major = major,
        .minor = minor,
        .build = build,
        .revision = revision,
    };
}

//===----------------------------------------------------------------------===//
// Module Tests
//===----------------------------------------------------------------------===//

const std = @import("std");
const testing = std.testing;

test "Z3 availability" {
    if (isZ3Available()) {
        std.debug.print("✅ Z3 is available\n", .{});
        if (getZ3Version()) |version| {
            std.debug.print("   Version: {}.{}.{}.{}\n", .{ version.major, version.minor, version.build, version.revision });
        }
    } else {
        std.debug.print("⚠️  Z3 not available - formal verification disabled\n", .{});
    }
}
