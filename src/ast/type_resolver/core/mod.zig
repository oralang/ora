// ============================================================================
// Core Type Resolver
// ============================================================================
// Core types (Typed, Effect, LockDelta, Obligation) and CoreResolver interface
// ============================================================================

const std = @import("std");
const ast = @import("../../../ast.zig");
const TypeInfo = @import("../../type_info.zig").TypeInfo;
const OraType = @import("../../type_info.zig").OraType;
const SourceSpan = @import("../../source_span.zig").SourceSpan;
const state = @import("../../../semantics/state.zig");
const semantics = @import("../../../semantics.zig");
const SymbolTable = state.SymbolTable;
const Scope = state.Scope;
const MemoryRegion = @import("../../statements.zig").MemoryRegion;

const validation = @import("../validation/mod.zig");
const utils = @import("../utils/mod.zig");
const refinements = @import("../refinements/mod.zig");

// ============================================================================
// Core Types
// ============================================================================

/// Complete typing result: type + effects + locks + obligations
pub const Typed = struct {
    ty: TypeInfo, // Base type (includes region via located type)
    eff: Effect, // Pure | Writes(S)
    lock_delta: LockDelta, // New locks produced
    obligations: []Obligation, // Refinement proof goals / asserts

    pub fn init(ty: TypeInfo, eff: Effect, allocator: std.mem.Allocator) Typed {
        return Typed{
            .ty = ty,
            .eff = eff,
            .lock_delta = LockDelta.emptyWithAllocator(allocator),
            .obligations = &.{},
        };
    }

    pub fn deinit(self: *Typed, allocator: std.mem.Allocator) void {
        // obligations are currently empty arrays, no cleanup needed
        // todo: When obligations are implemented, add cleanup:
        // for (self.obligations) |*ob| {
        //     ob.deinit(allocator);
        // }
        // allocator.free(self.obligations);
        self.lock_delta.deinit(allocator);
        self.eff.deinit(allocator);
    }
};

/// Effect of an expression/statement
pub const Effect = union(enum) {
    Pure,
    Writes: SlotSet,

    pub fn pure() Effect {
        return .Pure;
    }

    pub fn writes(slots: SlotSet) Effect {
        return .{ .Writes = slots };
    }

    pub fn deinit(self: *Effect, allocator: std.mem.Allocator) void {
        switch (self.*) {
            .Pure => {},
            .Writes => |*slots| slots.deinit(allocator),
        }
    }
};

/// Set of storage slots (for effect tracking)
pub const SlotSet = struct {
    slots: std.ArrayList([]const u8), // Slot identifiers

    pub fn init(_: std.mem.Allocator) SlotSet {
        return SlotSet{
            .slots = std.ArrayList([]const u8){},
        };
    }

    pub fn deinit(self: *SlotSet, allocator: std.mem.Allocator) void {
        self.slots.deinit(allocator);
    }

    pub fn add(self: *SlotSet, allocator: std.mem.Allocator, slot: []const u8) !void {
        try self.slots.append(allocator, slot);
    }

    pub fn contains(self: *const SlotSet, slot: []const u8) bool {
        for (self.slots.items) |s| {
            if (std.mem.eql(u8, s, slot)) return true;
        }
        return false;
    }
};

/// Lock delta: new locks produced by this expression/statement
pub const LockDelta = struct {
    locked_slots: SlotSet,

    pub fn init(allocator: std.mem.Allocator) LockDelta {
        return LockDelta{
            .locked_slots = SlotSet.init(allocator),
        };
    }

    pub fn emptyWithAllocator(allocator: std.mem.Allocator) LockDelta {
        // returns empty delta (no locks)
        return LockDelta{
            .locked_slots = SlotSet.init(allocator),
        };
    }

    pub fn deinit(self: *LockDelta, allocator: std.mem.Allocator) void {
        self.locked_slots.deinit(allocator);
    }
};

/// Obligation: proof goal or runtime assert candidate
pub const Obligation = struct {
    predicate: Predicate,
    origin: SourceSpan,
    severity: ObligationSeverity,

    pub fn deinit(self: *Obligation, allocator: std.mem.Allocator) void {
        self.predicate.deinit(allocator);
    }
};

/// Predicate for obligations (SMT-friendly representation)
pub const Predicate = struct {
    kind: PredicateKind,
    data: []const u8, // Serialized predicate

    pub fn deinit(self: *Predicate, allocator: std.mem.Allocator) void {
        allocator.free(self.data);
    }
};

pub const PredicateKind = enum {
    Comparison, // x <= MAX, x >= MIN, etc.
    Overflow, // (lhs + rhs) does not overflow
    Refinement, // Refinement-specific constraint
    Custom, // Other
};

pub const ObligationSeverity = enum {
    Error, // Must be proven or compilation fails
    Assert, // Emit runtime assert
    Assume, // Assume for verification (policy)
};

/// Type context for resolution
pub const TypeContext = struct {
    expected_type: ?TypeInfo = null,
    enum_underlying_type: ?TypeInfo = null,
    function_return_type: ?TypeInfo = null,
    expected_region: ?MemoryRegion = null,

    pub fn withExpectedType(expected: TypeInfo) TypeContext {
        return TypeContext{ .expected_type = expected };
    }

    pub fn withEnumType(enum_type: TypeInfo) TypeContext {
        return TypeContext{ .enum_underlying_type = enum_type };
    }

    pub fn withRegion(region: MemoryRegion) TypeContext {
        return TypeContext{ .expected_region = region };
    }
};

// ============================================================================
// Core Resolver
// ============================================================================

pub const CoreResolver = struct {
    allocator: std.mem.Allocator,
    symbol_table: *SymbolTable,
    current_scope: ?*Scope,
    validation: *validation.ValidationSystem,
    utils: *utils.Utils,
    refinement_system: *refinements.RefinementSystem,
    // optional: function registry for argument validation (set by TypeResolver)
    // type-erased to avoid circular dependency with ast.zig
    function_registry: ?*std.StringHashMap(*anyopaque) = null,
    // builtin registry for resolving std.* constants and functions
    builtin_registry: ?*const semantics.builtins.BuiltinRegistry = null,

    pub fn init(
        allocator: std.mem.Allocator,
        symbol_table: *SymbolTable,
        validation_sys: *validation.ValidationSystem,
        utils_sys: *utils.Utils,
        refinement_sys: *refinements.RefinementSystem,
    ) CoreResolver {
        return CoreResolver{
            .allocator = allocator,
            .symbol_table = symbol_table,
            .current_scope = &symbol_table.root,
            .validation = validation_sys,
            .utils = utils_sys,
            .refinement_system = refinement_sys,
            .builtin_registry = &symbol_table.builtin_registry,
        };
    }

    pub fn deinit(self: *CoreResolver) void {
        _ = self;
        // no cleanup needed - sub-systems handle their own cleanup
    }

    /// Synthesize (infer) type for an expression
    pub fn synthExpr(self: *CoreResolver, expr: *ast.Expressions.ExprNode) !Typed {
        return @import("expression.zig").synthExpr(self, expr);
    }

    /// Check expression against expected type
    pub fn checkExpr(self: *CoreResolver, expr: *ast.Expressions.ExprNode, expected: TypeInfo) !Typed {
        return @import("expression.zig").checkExpr(self, expr, expected);
    }

    /// Resolve types for a statement
    pub fn resolveStatement(self: *CoreResolver, stmt: *ast.Statements.StmtNode, context: TypeContext) !Typed {
        return @import("statement.zig").resolveStatement(self, stmt, context);
    }

    /// Look up identifier in symbol table
    pub fn lookupIdentifier(self: *CoreResolver, name: []const u8) ?TypeInfo {
        return @import("identifier.zig").lookupIdentifier(self, name);
    }
};
