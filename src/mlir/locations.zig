const std = @import("std");
const c = @import("c.zig").c;
const lib = @import("ora_lib");

/// Location tracking system for preserving source information in MLIR
pub const LocationTracker = struct {
    ctx: c.MlirContext,

    pub fn init(ctx: c.MlirContext) LocationTracker {
        return .{ .ctx = ctx };
    }

    /// Create a location from SourceSpan information with byte offset and length preservation
    pub fn createLocation(self: *const LocationTracker, span: ?lib.ast.SourceSpan) c.MlirLocation {
        if (span) |s| {
            // Create file location with line and column information
            const fname = c.mlirStringRefCreateFromCString("input.ora");
            const file_loc = c.mlirLocationFileLineColGet(self.ctx, fname, s.line, s.column);

            // TODO: In the future, we could create a fused location that includes
            // byte offset and length information as metadata attributes
            // For now, return the basic file location
            return file_loc;
        } else {
            return c.mlirLocationUnknownGet(self.ctx);
        }
    }

    /// Attach location to an operation (Note: MLIR operations are immutable after creation)
    /// This function serves as documentation that locations should be set during operation creation
    pub fn attachLocationToOp(self: *const LocationTracker, op: c.MlirOperation, span: ?lib.ast.SourceSpan) void {
        if (span) |_| {
            const location = self.createLocation(span);
            // Note: MLIR operations are immutable after creation, so we can't modify
            // the location of an existing operation. This function serves as a reminder
            // that locations should be set during operation creation using createLocationForOp.
            _ = location;
            _ = op;
        }
    }

    /// Create location for operation creation - use this when creating operations
    pub fn createLocationForOp(self: *const LocationTracker, span: ?lib.ast.SourceSpan) c.MlirLocation {
        return self.createLocation(span);
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

    /// Create location with byte offset and length information preserved as attributes
    pub fn createLocationWithSpanInfo(self: *const LocationTracker, span: lib.ast.SourceSpan, filename: ?[]const u8) c.MlirLocation {
        const fname = if (filename) |f|
            c.mlirStringRefCreate(f.ptr, f.len)
        else
            c.mlirStringRefCreateFromCString("input.ora");

        const file_loc = c.mlirLocationFileLineColGet(self.ctx, fname, span.line, span.column);

        // TODO: In a full implementation, we could create a fused location with metadata
        // that includes byte offset (span.start) and length (span.end - span.start)
        // For now, return the basic file location
        return file_loc;
    }

    /// Create location from lexeme information (preserving original source text)
    pub fn createLocationFromLexeme(self: *const LocationTracker, span: lib.ast.SourceSpan, lexeme: ?[]const u8) c.MlirLocation {
        // TODO: In a full implementation, we could preserve the original lexeme text
        // as metadata in the location for better debugging
        _ = lexeme; // For now, ignore the lexeme text
        return self.createLocation(span);
    }

    /// Helper function for consistent location attachment across all operations
    pub fn getLocationForSpan(self: *const LocationTracker, span: lib.ast.SourceSpan) c.MlirLocation {
        return self.createLocationWithSpanInfo(span, null);
    }

    /// Helper function to create unknown location when span is not available
    pub fn getUnknownLocation(self: *const LocationTracker) c.MlirLocation {
        return c.mlirLocationUnknownGet(self.ctx);
    }

    /// Create name location for debugging purposes
    pub fn createNameLocation(self: *const LocationTracker, name: []const u8, child_loc: ?c.MlirLocation) c.MlirLocation {
        const name_ref = c.mlirStringRefCreate(name.ptr, name.len);
        const name_attr = c.mlirStringAttrGet(self.ctx, name_ref);

        const base_loc = child_loc orelse c.mlirLocationUnknownGet(self.ctx);

        // TODO: Use mlirLocationNameGet when available in MLIR C API
        // For now, return the base location
        _ = name_attr; // Suppress unused variable warning
        return base_loc;
    }

    /// Create call site location for function calls
    pub fn createCallSiteLocation(self: *const LocationTracker, callee_loc: c.MlirLocation, caller_loc: c.MlirLocation) c.MlirLocation {
        _ = self;
        // TODO: Use mlirLocationCallSiteGet when available in MLIR C API
        // For now, return the caller location
        _ = callee_loc; // Suppress unused variable warning
        return caller_loc;
    }

    /// Validate that a location is properly formed
    pub fn validateLocation(self: *const LocationTracker, loc: c.MlirLocation) bool {
        _ = self;
        // Check if the location is null (invalid)
        return !c.mlirLocationIsNull(loc);
    }

    /// Extract line number from a file location
    pub fn getLineFromLocation(self: *const LocationTracker, loc: c.MlirLocation) ?u32 {
        _ = self;
        _ = loc;
        // TODO: Extract line number from MLIR location when API is available
        // For now, return null to indicate unavailable
        return null;
    }

    /// Extract column number from a file location
    pub fn getColumnFromLocation(self: *const LocationTracker, loc: c.MlirLocation) ?u32 {
        _ = self;
        _ = loc;
        // TODO: Extract column number from MLIR location when API is available
        // For now, return null to indicate unavailable
        return null;
    }

    /// Extract filename from a file location
    pub fn getFilenameFromLocation(self: *const LocationTracker, loc: c.MlirLocation) ?[]const u8 {
        _ = self;
        _ = loc;
        // TODO: Extract filename from MLIR location when API is available
        // For now, return null to indicate unavailable
        return null;
    }
};
