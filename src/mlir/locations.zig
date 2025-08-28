const std = @import("std");
const c = @import("c.zig").c;
const lib = @import("ora_lib");

/// Location tracking system for preserving source information in MLIR
pub const LocationTracker = struct {
    ctx: c.MlirContext,

    pub fn init(ctx: c.MlirContext) LocationTracker {
        return .{ .ctx = ctx };
    }

    /// Create a location from source span information
    pub fn createLocation(self: *const LocationTracker, span: ?lib.ast.SourceSpan) c.MlirLocation {
        if (span) |s| {
            // Use the existing location creation logic from lower.zig
            const fname = c.mlirStringRefCreateFromCString("input.ora");
            return c.mlirLocationFileLineColGet(self.ctx, fname, s.line, s.column);
        } else {
            return c.mlirLocationUnknownGet(self.ctx);
        }
    }

    /// Attach location to an operation
    pub fn attachLocationToOp(self: *const LocationTracker, op: c.MlirOperation, span: ?lib.ast.SourceSpan) void {
        if (span) |_| {
            const location = self.createLocation(span);
            // Note: MLIR operations are immutable after creation, so we can't modify
            // the location of an existing operation. This function serves as a reminder
            // that locations should be set during operation creation.
            _ = location;
            _ = op;
        }
    }

    /// Create a file location with line and column information
    pub fn createFileLocation(self: *const LocationTracker, filename: []const u8, line: u32, column: u32) c.MlirLocation {
        const fname_ref = c.mlirStringRefCreate(filename.ptr, filename.len);
        return c.mlirLocationFileLineColGet(self.ctx, fname_ref, line, column);
    }

    /// Create a file location from a source span (working function from lower.zig)
    pub fn createFileLocationFromSpan(self: *const LocationTracker, span: lib.ast.SourceSpan) c.MlirLocation {
        const fname = c.mlirStringRefCreateFromCString("input.ora");
        return c.mlirLocationFileLineColGet(self.ctx, fname, span.line, span.column);
    }

    /// Create a fused location combining multiple locations
    pub fn createFusedLocation(self: *const LocationTracker, locations: []const c.MlirLocation, _: ?c.MlirAttribute) c.MlirLocation {
        if (locations.len == 0) {
            return c.mlirLocationUnknownGet(self.ctx);
        }

        if (locations.len == 1) {
            return locations[0];
        }

        // For now, return the first location as a simple fallback
        // In the future, this could use mlirLocationFusedGet when available
        return locations[0];
    }

    /// Get location from an operation
    pub fn getLocationFromOp(self: *const LocationTracker, op: c.MlirOperation) c.MlirLocation {
        _ = self;
        return c.mlirOperationGetLocation(op);
    }
};
