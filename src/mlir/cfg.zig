// ============================================================================
// MLIR Control Flow Graph Generation
// ============================================================================
//
// Uses MLIR's built-in view-op-graph pass to generate CFG visualizations.
//
// ============================================================================

const std = @import("std");
const c = @import("c.zig").c;

/// Generate CFG from MLIR module using MLIR's built-in pass
/// This uses the view-op-graph pass which generates Graphviz DOT format
/// The dialect will be registered automatically via the C++ API
pub fn generateCFG(ctx: c.MlirContext, module: c.MlirModule, allocator: std.mem.Allocator) ![]const u8 {
    // Use the C++ API function that registers the dialect and generates CFG
    // Include control flow edges to show dominance relationships
    const dot_string_ref = c.oraGenerateCFG(ctx, module, true);

    if (dot_string_ref.data == null) {
        std.log.err("CFG generation returned null (length: {d})", .{dot_string_ref.length});
        return error.CFGGenerationFailed;
    }

    if (dot_string_ref.length == 0) {
        std.log.err("CFG generation returned empty string", .{});
        return error.CFGGenerationFailed;
    }

    // Copy the string to Zig-managed memory
    const dot_content = try allocator.dupe(u8, dot_string_ref.data[0..dot_string_ref.length]);

    // Free the C-allocated string
    // Note: The C API should provide mlirStringRefFree, but for now we'll use free
    const c_allocator = std.heap.c_allocator;
    c_allocator.free(dot_string_ref.data[0..dot_string_ref.length]);

    return dot_content;
}
