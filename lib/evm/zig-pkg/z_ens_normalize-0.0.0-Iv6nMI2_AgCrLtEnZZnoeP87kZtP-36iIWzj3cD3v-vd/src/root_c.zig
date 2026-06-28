// C FFI bindings for z-ens-normalize
const std = @import("std");
const root = @import("root.zig");

// Error codes for C API
pub const ZensError = enum(c_int) {
    Success = 0,
    OutOfMemory = -1,
    InvalidUtf8 = -2,
    InvalidLabelExtension = -3,
    IllegalMixture = -4,
    WholeConfusable = -5,
    LeadingUnderscore = -6,
    FencedLeading = -7,
    FencedAdjacent = -8,
    FencedTrailing = -9,
    DisallowedCharacter = -10,
    EmptyLabel = -11,
    CMLeading = -12,
    CMAfterEmoji = -13,
    NSMDuplicate = -14,
    NSMExcessive = -15,
    UnknownError = -99,

    pub fn fromError(err: anyerror) ZensError {
        return switch (err) {
            error.OutOfMemory => .OutOfMemory,
            error.InvalidUtf8 => .InvalidUtf8,
            error.InvalidLabelExtension => .InvalidLabelExtension,
            error.IllegalMixture => .IllegalMixture,
            error.WholeConfusable => .WholeConfusable,
            error.LeadingUnderscore => .LeadingUnderscore,
            error.FencedLeading => .FencedLeading,
            error.FencedAdjacent => .FencedAdjacent,
            error.FencedTrailing => .FencedTrailing,
            error.DisallowedCharacter => .DisallowedCharacter,
            error.EmptyLabel => .EmptyLabel,
            error.CMLeading => .CMLeading,
            error.CMAfterEmoji => .CMAfterEmoji,
            error.NSMDuplicate => .NSMDuplicate,
            error.NSMExcessive => .NSMExcessive,
            else => .UnknownError,
        };
    }
};

// Opaque allocator handle for C API
const ZensAllocator = opaque {};

// Result struct for C API
pub const ZensResult = extern struct {
    data: ?[*]u8,
    len: usize,
    error_code: c_int,
};

// Global allocator - using GeneralPurposeAllocator for C API
var gpa: ?std.heap.GeneralPurposeAllocator(.{}) = null;
var gpa_mutex: std.Thread.Mutex = .{};

fn getGlobalAllocator() std.mem.Allocator {
    gpa_mutex.lock();
    defer gpa_mutex.unlock();

    if (gpa == null) {
        gpa = std.heap.GeneralPurposeAllocator(.{}){};
    }
    return gpa.?.allocator();
}

/// Initialize the library (optional, but recommended to call once)
/// Returns 0 on success, non-zero on failure
export fn zens_init() c_int {
    _ = getGlobalAllocator();
    return 0;
}

/// Cleanup the library (call at program exit)
export fn zens_deinit() void {
    gpa_mutex.lock();
    defer gpa_mutex.unlock();

    if (gpa) |*g| {
        _ = g.deinit();
        gpa = null;
    }
}

/// Normalize an ENS name
///
/// @param input: Input name as UTF-8 bytes (null-terminated)
/// @param input_len: Length of input (or 0 to use strlen)
/// @return ZensResult with normalized name or error
export fn zens_normalize(input: [*c]const u8, input_len: usize) ZensResult {
    if (input == null) {
        return ZensResult{
            .data = null,
            .len = 0,
            .error_code = @intFromEnum(ZensError.InvalidUtf8),
        };
    }

    const allocator = getGlobalAllocator();

    // Determine input length
    const len = if (input_len == 0) std.mem.len(input) else input_len;
    const input_slice = input[0..len];

    const result = root.normalize(allocator, input_slice) catch |err| {
        return ZensResult{
            .data = null,
            .len = 0,
            .error_code = @intFromEnum(ZensError.fromError(err)),
        };
    };

    return ZensResult{
        .data = result.ptr,
        .len = result.len,
        .error_code = @intFromEnum(ZensError.Success),
    };
}

/// Beautify an ENS name
///
/// @param input: Input name as UTF-8 bytes (null-terminated)
/// @param input_len: Length of input (or 0 to use strlen)
/// @return ZensResult with beautified name or error
export fn zens_beautify(input: [*c]const u8, input_len: usize) ZensResult {
    if (input == null) {
        return ZensResult{
            .data = null,
            .len = 0,
            .error_code = @intFromEnum(ZensError.InvalidUtf8),
        };
    }

    const allocator = getGlobalAllocator();

    // Determine input length
    const len = if (input_len == 0) std.mem.len(input) else input_len;
    const input_slice = input[0..len];

    const result = root.beautify(allocator, input_slice) catch |err| {
        return ZensResult{
            .data = null,
            .len = 0,
            .error_code = @intFromEnum(ZensError.fromError(err)),
        };
    };

    return ZensResult{
        .data = result.ptr,
        .len = result.len,
        .error_code = @intFromEnum(ZensError.Success),
    };
}

/// Free memory allocated by zens_normalize or zens_beautify
///
/// @param result: Result struct from zens_normalize or zens_beautify
export fn zens_free(result: ZensResult) void {
    if (result.data) |ptr| {
        const allocator = getGlobalAllocator();
        const slice = ptr[0..result.len];
        allocator.free(slice);
    }
}

/// Get error message for error code
///
/// @param error_code: Error code from ZensResult
/// @return Null-terminated error message string (do not free)
export fn zens_error_message(error_code: c_int) [*c]const u8 {
    const err: ZensError = @enumFromInt(error_code);
    return switch (err) {
        .Success => "Success",
        .OutOfMemory => "Out of memory",
        .InvalidUtf8 => "Invalid UTF-8 encoding",
        .InvalidLabelExtension => "Invalid label extension (-- at positions 2-3)",
        .IllegalMixture => "Illegal script mixture",
        .WholeConfusable => "Whole confusable",
        .LeadingUnderscore => "Leading underscore",
        .FencedLeading => "Fenced leading",
        .FencedAdjacent => "Fenced adjacent",
        .FencedTrailing => "Fenced trailing",
        .DisallowedCharacter => "Disallowed character",
        .EmptyLabel => "Empty label",
        .CMLeading => "Combining mark leading",
        .CMAfterEmoji => "Combining mark after emoji",
        .NSMDuplicate => "Non-spacing mark duplicate",
        .NSMExcessive => "Non-spacing mark excessive",
        .UnknownError => "Unknown error",
    };
}

// WASM-specific exports with simpler signatures
comptime {
    if (@import("builtin").target.isWasm()) {
        // Export memory for WASM
        @export(&std.heap.page_allocator, .{ .name = "memory" });
    }
}
