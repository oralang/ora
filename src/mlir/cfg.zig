// ============================================================================
// MLIR Control Flow Graph Generation
// ============================================================================
//
// Uses Ora's MLIR C API to generate mode-specific Graphviz DOT CFGs.
//
// ============================================================================

const std = @import("std");
const mlir_c_api = @import("mlir_c_api");
const c = mlir_c_api.c;

pub const Mode = enum {
    ora,
    sir,

    fn string(self: Mode) []const u8 {
        return switch (self) {
            .ora => "ora",
            .sir => "sir",
        };
    }
};

pub const Options = struct {
    mode: Mode,
    proven_guard_ids: ?*const std.StringHashMap(void) = null,
};

/// Generate a Graphviz DOT CFG from an MLIR module.
/// `sir` mode emits a true SIR basic-block CFG. `ora` mode emits a structured
/// region/control view with FV guard overlay data when proven IDs are supplied.
pub fn generateCFG(
    ctx: c.MlirContext,
    module: c.MlirModule,
    allocator: std.mem.Allocator,
    options: Options,
) ![]const u8 {
    const mode_text = options.mode.string();
    var guard_refs: std.ArrayList(c.MlirStringRef) = .empty;
    defer guard_refs.deinit(allocator);

    if (options.proven_guard_ids) |ids| {
        var iter = ids.iterator();
        while (iter.next()) |entry| {
            try guard_refs.append(allocator, c.oraStringRefCreate(entry.key_ptr.*.ptr, entry.key_ptr.*.len));
        }
    }

    const dot_string_ref = c.oraGenerateCFGWithOptions(
        ctx,
        module,
        c.oraStringRefCreate(mode_text.ptr, mode_text.len),
        if (guard_refs.items.len > 0) guard_refs.items.ptr else null,
        guard_refs.items.len,
    );

    if (dot_string_ref.data == null) {
        std.log.err("CFG generation returned null (length: {d})", .{dot_string_ref.length});
        return error.CFGGenerationFailed;
    }

    if (dot_string_ref.length == 0) {
        std.log.err("CFG generation returned empty string", .{});
        return error.CFGGenerationFailed;
    }

    // copy the string to Zig-managed memory
    const dot_content = try allocator.dupe(u8, dot_string_ref.data[0..dot_string_ref.length]);

    mlir_c_api.freeStringRef(dot_string_ref);

    return dot_content;
}
