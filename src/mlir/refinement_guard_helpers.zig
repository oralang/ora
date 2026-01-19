// ============================================================================
// Refinement Guard Helpers
// ============================================================================
//
// Shared guard emission logic used by statement and expression lowerers.
//
// ============================================================================

const std = @import("std");
const c = @import("mlir_c_api").c;
const h = @import("helpers.zig");
const lib = @import("ora_lib");
const OraDialect = @import("dialect.zig").OraDialect;
const LocationTracker = @import("locations.zig").LocationTracker;

pub fn emitRefinementGuard(
    ctx: c.MlirContext,
    block: c.MlirBlock,
    ora_dialect: *OraDialect,
    locations: LocationTracker,
    guard_cache: ?*std.AutoHashMap(u128, void),
    span: lib.ast.SourceSpan,
    condition: c.MlirValue,
    message: []const u8,
    refinement_kind: []const u8,
    var_name: ?[]const u8,
    allocator: std.mem.Allocator,
) void {
    const name_slice = var_name orelse "";
    var hasher_hi = std.hash.Wyhash.init(0);
    var hasher_lo = std.hash.Wyhash.init(1);
    hasher_hi.update(locations.filename);
    hasher_hi.update(std.mem.asBytes(&span.line));
    hasher_hi.update(std.mem.asBytes(&span.column));
    hasher_hi.update(std.mem.asBytes(&span.length));
    hasher_hi.update(refinement_kind);
    hasher_hi.update(name_slice);
    hasher_lo.update(locations.filename);
    hasher_lo.update(std.mem.asBytes(&span.line));
    hasher_lo.update(std.mem.asBytes(&span.column));
    hasher_lo.update(std.mem.asBytes(&span.length));
    hasher_lo.update(refinement_kind);
    hasher_lo.update(name_slice);
    const key: u128 = (@as(u128, hasher_hi.final()) << 64) | hasher_lo.final();
    if (guard_cache) |cache| {
        if (cache.contains(key)) {
            return;
        }
        cache.put(key, {}) catch {};
    }

    const loc = c.oraLocationFileLineColGet(
        ctx,
        h.strRef(locations.filename),
        span.line,
        span.column,
    );
    const guard_op = ora_dialect.createRefinementGuard(condition, loc, message);
    h.appendOp(block, guard_op);

    const guard_id = if (var_name) |name|
        std.fmt.allocPrint(
            allocator,
            "guard:{s}:{d}:{d}:{d}:{s}:{s}",
            .{ locations.filename, span.line, span.column, span.length, refinement_kind, name },
        ) catch return
    else
        std.fmt.allocPrint(
            allocator,
            "guard:{s}:{d}:{d}:{d}:{s}",
            .{ locations.filename, span.line, span.column, span.length, refinement_kind },
        ) catch return;
    defer allocator.free(guard_id);

    c.oraOperationSetAttributeByName(guard_op, h.strRef("ora.guard_id"), h.stringAttr(ctx, guard_id));
    c.oraOperationSetAttributeByName(guard_op, h.strRef("ora.refinement_kind"), h.stringAttr(ctx, refinement_kind));
}
