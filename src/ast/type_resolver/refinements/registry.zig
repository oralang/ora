// ============================================================================
// Refinement Registry
// ============================================================================
// Phase 0: Registry interface and configuration
// ============================================================================

const std = @import("std");
const OraType = @import("../../type_info.zig").OraType;
const SourceSpan = @import("../../source_span.zig").SourceSpan;
const BinaryOp = @import("../../expressions.zig").BinaryOp;
const TypeResolutionError = @import("../mod.zig").TypeResolutionError;
const Obligation = @import("../core/mod.zig").Obligation;

/// Configuration for refinement system (no hardcoded constants)
pub const RefinementConfig = struct {
    scaled_max_decimals: u32 = 77, // Maximum decimals for Scaled types

    pub fn default() RefinementConfig {
        return RefinementConfig{};
    }
};

/// Context for obligation generation
pub const ObligationContext = struct {
    source_span: SourceSpan,
    // Phase 0: Minimal, will expand in Phase 1
};

/// Refinement handler interface
pub const RefinementHandler = struct {
    name: []const u8,

    validate: *const fn (cfg: *const RefinementConfig, ty: *const OraType) TypeResolutionError!void,

    // Optional: returns refined result; null means "fall back to base arithmetic"
    inferArithmetic: ?*const fn (
        cfg: *const RefinementConfig,
        op: BinaryOp,
        lhs: OraType,
        rhs: OraType,
    ) ?OraType,

    checkSubtype: *const fn (
        cfg: *const RefinementConfig,
        src: OraType,
        dst: OraType,
    ) bool,

    extractBase: *const fn (ty: OraType) ?OraType,

    // Optional hook to generate obligations
    obligationsForUse: ?*const fn (
        cfg: *const RefinementConfig,
        context: ObligationContext,
        refined: OraType,
    ) []Obligation,
};

/// Registry for pluggable refinement handlers
pub const RefinementRegistry = struct {
    by_name: std.StringHashMap(u16),
    handlers: std.ArrayList(RefinementHandler),
    allocator: std.mem.Allocator,

    pub fn init(allocator: std.mem.Allocator) RefinementRegistry {
        return RefinementRegistry{
            .by_name = std.StringHashMap(u16).init(allocator),
            .handlers = std.ArrayList(RefinementHandler){},
            .allocator = allocator,
        };
    }

    pub fn deinit(self: *RefinementRegistry) void {
        self.by_name.deinit();
        self.handlers.deinit(self.allocator);
    }

    /// Register a refinement handler, returns handler_id
    pub fn register(self: *RefinementRegistry, h: RefinementHandler) !u16 {
        const id: u16 = @intCast(self.handlers.items.len);
        try self.handlers.append(self.allocator, h);
        try self.by_name.put(h.name, id);
        return id;
    }

    /// Get handler ID by name
    pub fn getId(self: *const RefinementRegistry, name: []const u8) ?u16 {
        return self.by_name.get(name);
    }

    /// Get handler by ID (O(1) lookup)
    pub fn handler(self: *const RefinementRegistry, id: u16) ?*const RefinementHandler {
        if (id >= self.handlers.items.len) return null;
        return &self.handlers.items[id];
    }
};
