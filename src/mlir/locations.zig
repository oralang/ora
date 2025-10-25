// ============================================================================
// Location Tracking
// ============================================================================
//
// Preserves source code location information in MLIR for debugging and errors.
//
// FEATURES:
//   • Creates MLIR locations from AST source spans
//   • Tracks file, line, column, and byte offsets
//   • Supports unknown/fallback locations
//
// ============================================================================

const std = @import("std");
const c = @import("c.zig").c;
const lib = @import("ora_lib");
const h = @import("helpers.zig"); // Import helpers

/// Location tracking system for preserving source information in MLIR
pub const LocationTracker = struct {
    ctx: c.MlirContext,
    filename: []const u8,

    pub fn init(ctx: c.MlirContext) LocationTracker {
        return .{ .ctx = ctx, .filename = "input.ora" };
    }

    pub fn initWithFilename(ctx: c.MlirContext, filename: []const u8) LocationTracker {
        return .{ .ctx = ctx, .filename = filename };
    }

    /// Create a location from SourceSpan information
    pub fn createLocation(self: *const LocationTracker, span: ?lib.ast.SourceSpan) c.MlirLocation {
        if (span) |s| {
            const fname_ref = c.mlirStringRefCreate(self.filename.ptr, self.filename.len);
            return c.mlirLocationFileLineColGet(self.ctx, fname_ref, s.line, s.column);
        }
        return h.unknownLoc(self.ctx);
    }

    /// Create a file location with custom filename, line, and column
    pub fn createFileLocation(self: *const LocationTracker, filename: []const u8, line: u32, column: u32) c.MlirLocation {
        const fname_ref = c.mlirStringRefCreate(filename.ptr, filename.len);
        return c.mlirLocationFileLineColGet(self.ctx, fname_ref, line, column);
    }

    /// Get location from an operation
    pub fn getLocationFromOp(_: *const LocationTracker, op: c.MlirOperation) c.MlirLocation {
        return c.mlirOperationGetLocation(op);
    }

    /// Create unknown location when span is not available
    pub fn getUnknownLocation(self: *const LocationTracker) c.MlirLocation {
        return h.unknownLoc(self.ctx);
    }

    /// Validate that a location is properly formed
    pub fn validateLocation(_: *const LocationTracker, loc: c.MlirLocation) bool {
        return !c.mlirLocationIsNull(loc);
    }
};
