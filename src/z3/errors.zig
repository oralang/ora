//===----------------------------------------------------------------------===//
//
// Verification Error Handling
//
//===----------------------------------------------------------------------===//
//
// Defines error types, counterexample extraction, and user-facing
// error reporting for verification failures.
//
//===----------------------------------------------------------------------===//

const std = @import("std");
const c = @import("c.zig").c;

/// Type of verification error
pub const VerificationErrorType = enum {
    ArithmeticOverflow,
    ArithmeticUnderflow,
    DivisionByZero,
    ArrayOutOfBounds,
    StorageInconsistency,
    InvariantViolation,
    PreconditionViolation,
    PostconditionViolation,
    UnreachableCode,
    Unknown,
};

/// Verification error with source location and counterexample
pub const VerificationError = struct {
    error_type: VerificationErrorType,
    message: []const u8,
    file: []const u8,
    line: u32,
    column: u32,
    counterexample: ?Counterexample,

    allocator: std.mem.Allocator,

    pub fn deinit(self: *VerificationError) void {
        self.allocator.free(self.message);
        if (self.counterexample) |*ce| {
            ce.deinit();
        }
    }
};

/// Counterexample showing values that violate the property
pub const Counterexample = struct {
    variables: std.StringHashMap([]const u8),
    allocator: std.mem.Allocator,

    pub fn init(allocator: std.mem.Allocator) Counterexample {
        return .{
            .variables = std.StringHashMap([]const u8).init(allocator),
            .allocator = allocator,
        };
    }

    pub fn deinit(self: *Counterexample) void {
        var iter = self.variables.iterator();
        while (iter.next()) |entry| {
            self.allocator.free(entry.key_ptr.*);
            self.allocator.free(entry.value_ptr.*);
        }
        self.variables.deinit();
    }

    pub fn addVariable(self: *Counterexample, name: []const u8, value: []const u8) !void {
        const name_copy = try self.allocator.dupe(u8, name);
        errdefer self.allocator.free(name_copy);

        const value_copy = try self.allocator.dupe(u8, value);
        errdefer self.allocator.free(value_copy);

        try self.variables.put(name_copy, value_copy);
    }
};

/// Result of verification pass
pub const VerificationResult = struct {
    success: bool,
    errors: std.ArrayList(VerificationError),
    allocator: std.mem.Allocator,

    pub fn init(allocator: std.mem.Allocator) VerificationResult {
        return .{
            .success = true,
            .errors = std.ArrayList(VerificationError).init(allocator),
            .allocator = allocator,
        };
    }

    pub fn deinit(self: *VerificationResult) void {
        for (self.errors.items) |*err| {
            err.deinit();
        }
        self.errors.deinit();
    }

    pub fn addError(self: *VerificationResult, err: VerificationError) !void {
        self.success = false;
        try self.errors.append(err);
    }
};
