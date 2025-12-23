// ============================================================================
// Refinement System
// ============================================================================
// Phase 1: Implement refinement system with registry
// ============================================================================

const std = @import("std");
const ast = @import("../../../ast.zig");
const OraType = @import("../../type_info.zig").OraType;
const TypeInfo = @import("../../type_info.zig").TypeInfo;
const TypeCategory = @import("../../type_info.zig").TypeCategory;
const SourceSpan = @import("../../source_span.zig").SourceSpan;
const BinaryOp = @import("../../expressions.zig").BinaryOp;
const TypeResolutionError = @import("../mod.zig").TypeResolutionError;
const utils = @import("../utils/mod.zig");
const extract = @import("../utils/extraction.zig");

// Sub-modules
pub const registry = @import("registry.zig");
pub const validator = @import("validator.zig");
pub const arithmetic = @import("arithmetic.zig");
pub const subtyping = @import("subtyping.zig");

// Re-export
pub const RefinementHandler = registry.RefinementHandler;
pub const RefinementRegistry = registry.RefinementRegistry;
pub const RefinementConfig = registry.RefinementConfig;
pub const ObligationContext = registry.ObligationContext;

pub const RefinementSystem = struct {
    allocator: std.mem.Allocator,
    registry: RefinementRegistry,
    config: RefinementConfig,
    utils: *utils.Utils,

    pub fn init(allocator: std.mem.Allocator, utils_sys: *utils.Utils) RefinementSystem {
        var system = RefinementSystem{
            .allocator = allocator,
            .registry = RefinementRegistry.init(allocator),
            .config = RefinementConfig.default(),
            .utils = utils_sys,
        };

        // Phase 1: Register built-in refinement handlers
        registerBuiltinHandlers(&system) catch |err| {
            std.debug.panic("Failed to register refinement handlers: {s}\n", .{@errorName(err)});
        };

        return system;
    }

    pub fn deinit(self: *RefinementSystem) void {
        self.registry.deinit();
    }

    /// Validate a refinement type
    pub fn validate(self: *RefinementSystem, ora_type: OraType) TypeResolutionError!void {
        try validator.validateRefinementType(&self.config, &self.registry, ora_type);
    }

    /// Infer arithmetic result type
    pub fn inferArithmetic(
        self: *RefinementSystem,
        operator: BinaryOp,
        lhs_type: ?TypeInfo,
        rhs_type: ?TypeInfo,
    ) ?TypeInfo {
        _ = self;
        return arithmetic.inferArithmeticResultType(operator, lhs_type, rhs_type);
    }

    /// Check if source is a subtype of target
    pub fn checkSubtype(
        self: *RefinementSystem,
        source: OraType,
        target: OraType,
        comptime isBaseTypeCompatible: fn (OraType, OraType) bool,
    ) bool {
        _ = self;
        return subtyping.checkRefinementSubtyping(source, target, isBaseTypeCompatible);
    }
};

/// Register all built-in refinement type handlers
fn registerBuiltinHandlers(system: *RefinementSystem) !void {
    // MinValue handler
    const min_value_id = try system.registry.register(RefinementHandler{
        .name = "MinValue",
        .validate = validateMinValue,
        .inferArithmetic = inferMinValueArithmetic,
        .checkSubtype = checkMinValueSubtype,
        .extractBase = extractMinValueBase,
        .obligationsForUse = null,
    });
    _ = min_value_id;

    // MaxValue handler
    const max_value_id = try system.registry.register(RefinementHandler{
        .name = "MaxValue",
        .validate = validateMaxValue,
        .inferArithmetic = inferMaxValueArithmetic,
        .checkSubtype = checkMaxValueSubtype,
        .extractBase = extractMaxValueBase,
        .obligationsForUse = null,
    });
    _ = max_value_id;

    // InRange handler
    const in_range_id = try system.registry.register(RefinementHandler{
        .name = "InRange",
        .validate = validateInRange,
        .inferArithmetic = inferInRangeArithmetic,
        .checkSubtype = checkInRangeSubtype,
        .extractBase = extractInRangeBase,
        .obligationsForUse = null,
    });
    _ = in_range_id;

    // Scaled handler
    const scaled_id = try system.registry.register(RefinementHandler{
        .name = "Scaled",
        .validate = validateScaled,
        .inferArithmetic = inferScaledArithmetic,
        .checkSubtype = checkScaledSubtype,
        .extractBase = extractScaledBase,
        .obligationsForUse = null,
    });
    _ = scaled_id;

    // Exact handler
    const exact_id = try system.registry.register(RefinementHandler{
        .name = "Exact",
        .validate = validateExact,
        .inferArithmetic = null, // Exact doesn't preserve through arithmetic
        .checkSubtype = checkExactSubtype,
        .extractBase = extractExactBase,
        .obligationsForUse = null,
    });
    _ = exact_id;

    // NonZeroAddress handler
    const non_zero_address_id = try system.registry.register(RefinementHandler{
        .name = "NonZeroAddress",
        .validate = validateNonZeroAddress,
        .inferArithmetic = null,
        .checkSubtype = checkNonZeroAddressSubtype,
        .extractBase = extractNonZeroAddressBase,
        .obligationsForUse = null,
    });
    _ = non_zero_address_id;
}

// ============================================================================
// MinValue Handlers
// ============================================================================

fn validateMinValue(cfg: *const RefinementConfig, ty: *const OraType) TypeResolutionError!void {
    _ = cfg;
    switch (ty.*) {
        .min_value => |mv| {
            if (!mv.base.isInteger()) {
                return TypeResolutionError.TypeMismatch;
            }
        },
        else => return TypeResolutionError.TypeMismatch,
    }
}

fn inferMinValueArithmetic(
    cfg: *const RefinementConfig,
    op: BinaryOp,
    lhs: OraType,
    rhs: OraType,
) ?OraType {
    _ = cfg;
    const result = switch (op) {
        .Plus => arithmetic.inferAdditionResultType(lhs, rhs, null),
        .Minus => arithmetic.inferSubtractionResultType(lhs, rhs, null),
        .Star => arithmetic.inferMultiplicationResultType(lhs, rhs, null),
        else => null,
    };
    if (result) |r| {
        return r.ora_type;
    }
    return null;
}

fn checkMinValueSubtype(
    cfg: *const RefinementConfig,
    src: OraType,
    dst: OraType,
) bool {
    _ = cfg;
    return subtyping.checkRefinementSubtyping(src, dst, defaultBaseCompatible);
}

fn extractMinValueBase(ty: OraType) ?OraType {
    return extract.extractBaseType(ty);
}

// ============================================================================
// MaxValue Handlers
// ============================================================================

fn validateMaxValue(cfg: *const RefinementConfig, ty: *const OraType) TypeResolutionError!void {
    _ = cfg;
    switch (ty.*) {
        .max_value => |mv| {
            if (!mv.base.isInteger()) {
                return TypeResolutionError.TypeMismatch;
            }
        },
        else => return TypeResolutionError.TypeMismatch,
    }
}

fn inferMaxValueArithmetic(
    cfg: *const RefinementConfig,
    op: BinaryOp,
    lhs: OraType,
    rhs: OraType,
) ?OraType {
    _ = cfg;
    const result = switch (op) {
        .Plus => arithmetic.inferAdditionResultType(lhs, rhs, null),
        .Minus => arithmetic.inferSubtractionResultType(lhs, rhs, null),
        .Star => arithmetic.inferMultiplicationResultType(lhs, rhs, null),
        else => null,
    };
    return if (result) |r| r.ora_type else null;
}

fn checkMaxValueSubtype(
    cfg: *const RefinementConfig,
    src: OraType,
    dst: OraType,
) bool {
    _ = cfg;
    return subtyping.checkRefinementSubtyping(src, dst, defaultBaseCompatible);
}

fn extractMaxValueBase(ty: OraType) ?OraType {
    return extract.extractBaseType(ty);
}

// ============================================================================
// InRange Handlers
// ============================================================================

fn validateInRange(cfg: *const RefinementConfig, ty: *const OraType) TypeResolutionError!void {
    _ = cfg;
    switch (ty.*) {
        .in_range => |ir| {
            if (!ir.base.isInteger()) {
                return TypeResolutionError.TypeMismatch;
            }
            if (ir.min > ir.max) {
                return TypeResolutionError.TypeMismatch;
            }
        },
        else => return TypeResolutionError.TypeMismatch,
    }
}

fn inferInRangeArithmetic(
    cfg: *const RefinementConfig,
    op: BinaryOp,
    lhs: OraType,
    rhs: OraType,
) ?OraType {
    _ = cfg;
    const result = switch (op) {
        .Plus => arithmetic.inferAdditionResultType(lhs, rhs, null),
        .Minus => arithmetic.inferSubtractionResultType(lhs, rhs, null),
        .Star => arithmetic.inferMultiplicationResultType(lhs, rhs, null),
        else => null,
    };
    return if (result) |r| r.ora_type else null;
}

fn checkInRangeSubtype(
    cfg: *const RefinementConfig,
    src: OraType,
    dst: OraType,
) bool {
    _ = cfg;
    return subtyping.checkRefinementSubtyping(src, dst, defaultBaseCompatible);
}

fn extractInRangeBase(ty: OraType) ?OraType {
    return extract.extractBaseType(ty);
}

// ============================================================================
// Scaled Handlers
// ============================================================================

fn validateScaled(cfg: *const RefinementConfig, ty: *const OraType) TypeResolutionError!void {
    _ = cfg;
    switch (ty.*) {
        .scaled => |s| {
            if (!s.base.isInteger()) {
                return TypeResolutionError.TypeMismatch;
            }
            // Note: No hardcoded decimals > 77 constraint
        },
        else => return TypeResolutionError.TypeMismatch,
    }
}

fn inferScaledArithmetic(
    cfg: *const RefinementConfig,
    op: BinaryOp,
    lhs: OraType,
    rhs: OraType,
) ?OraType {
    _ = cfg;
    const result = switch (op) {
        .Plus => arithmetic.inferAdditionResultType(lhs, rhs, null),
        .Minus => arithmetic.inferSubtractionResultType(lhs, rhs, null),
        .Star => arithmetic.inferMultiplicationResultType(lhs, rhs, null),
        else => null,
    };
    return if (result) |r| r.ora_type else null;
}

fn checkScaledSubtype(
    cfg: *const RefinementConfig,
    src: OraType,
    dst: OraType,
) bool {
    _ = cfg;
    return subtyping.checkRefinementSubtyping(src, dst, defaultBaseCompatible);
}

fn extractScaledBase(ty: OraType) ?OraType {
    return extract.extractBaseType(ty);
}

// ============================================================================
// Exact Handlers
// ============================================================================

fn validateExact(cfg: *const RefinementConfig, ty: *const OraType) TypeResolutionError!void {
    _ = cfg;
    switch (ty.*) {
        .exact => |e| {
            if (!e.isInteger()) {
                return TypeResolutionError.TypeMismatch;
            }
        },
        else => return TypeResolutionError.TypeMismatch,
    }
}

fn checkExactSubtype(
    cfg: *const RefinementConfig,
    src: OraType,
    dst: OraType,
) bool {
    _ = cfg;
    return subtyping.checkRefinementSubtyping(src, dst, defaultBaseCompatible);
}

fn extractExactBase(ty: OraType) ?OraType {
    return extract.extractBaseType(ty);
}

// ============================================================================
// NonZeroAddress Handlers
// ============================================================================

fn validateNonZeroAddress(cfg: *const RefinementConfig, ty: *const OraType) TypeResolutionError!void {
    _ = cfg;
    switch (ty.*) {
        .non_zero_address => {},
        else => return TypeResolutionError.TypeMismatch,
    }
}

fn checkNonZeroAddressSubtype(
    cfg: *const RefinementConfig,
    src: OraType,
    dst: OraType,
) bool {
    _ = cfg;
    return subtyping.checkRefinementSubtyping(src, dst, defaultBaseCompatible);
}

fn extractNonZeroAddressBase(ty: OraType) ?OraType {
    return extract.extractBaseType(ty);
}

// ============================================================================
// Helper Functions
// ============================================================================

fn defaultBaseCompatible(source: OraType, target: OraType) bool {
    return OraType.equals(source, target);
}
