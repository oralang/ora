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
const c = @import("c.zig");
const ManagedArrayList = std.array_list.Managed;

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
    RefinementViolation,
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

/// Diagnostic showing how a guard can be violated (not an error, but useful info)
pub const Diagnostic = struct {
    guard_id: []const u8,
    function_name: []const u8,
    counterexample: Counterexample,
    allocator: std.mem.Allocator,

    pub fn deinit(self: *Diagnostic) void {
        self.allocator.free(self.guard_id);
        self.allocator.free(self.function_name);
        var ce = self.counterexample;
        ce.deinit();
    }
};

/// Result of verification pass
pub const VerificationResult = struct {
    success: bool,
    errors: ManagedArrayList(VerificationError),
    diagnostics: ManagedArrayList(Diagnostic),
    seen_keys: std.StringHashMap(void),
    proven_guard_ids: std.StringHashMap(void),
    allocator: std.mem.Allocator,

    pub fn init(allocator: std.mem.Allocator) VerificationResult {
        return .{
            .success = true,
            .errors = ManagedArrayList(VerificationError).init(allocator),
            .diagnostics = ManagedArrayList(Diagnostic).init(allocator),
            .seen_keys = std.StringHashMap(void).init(allocator),
            .proven_guard_ids = std.StringHashMap(void).init(allocator),
            .allocator = allocator,
        };
    }

    pub fn deinit(self: *VerificationResult) void {
        for (self.errors.items) |*err| {
            err.deinit();
        }
        self.errors.deinit();
        for (self.diagnostics.items) |*diag| {
            diag.deinit();
        }
        self.diagnostics.deinit();
        var iter = self.seen_keys.iterator();
        while (iter.next()) |entry| {
            self.allocator.free(entry.key_ptr.*);
        }
        self.seen_keys.deinit();
        var guard_iter = self.proven_guard_ids.iterator();
        while (guard_iter.next()) |entry| {
            self.allocator.free(entry.key_ptr.*);
        }
        self.proven_guard_ids.deinit();
    }

    pub fn addError(self: *VerificationResult, err: VerificationError) !void {
        const key = try std.fmt.allocPrint(
            self.allocator,
            "{s}|{s}|{d}|{d}|{s}",
            .{ @tagName(err.error_type), err.file, err.line, err.column, err.message },
        );
        errdefer self.allocator.free(key);

        if (self.seen_keys.contains(key)) {
            self.allocator.free(key);
            var dup = err;
            dup.deinit();
            return;
        }

        try self.seen_keys.put(key, {});
        self.success = false;
        try self.errors.append(err);
    }
};
