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
const comptime_eval = @import("../../../comptime/mod.zig");
const log = @import("log");
const TypeResolutionError = @import("../mod.zig").TypeResolutionError;

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

pub fn takeEffect(typed: *Typed) Effect {
    const eff = typed.eff;
    typed.eff = Effect.pure();
    return eff;
}

/// Effect of an expression/statement
pub const Effect = union(enum) {
    Pure,
    Reads: SlotSet,
    Writes: SlotSet,
    ReadsWrites: struct {
        reads: SlotSet,
        writes: SlotSet,
    },

    pub fn pure() Effect {
        return .Pure;
    }

    pub fn reads(slots: SlotSet) Effect {
        return .{ .Reads = slots };
    }

    pub fn writes(slots: SlotSet) Effect {
        return .{ .Writes = slots };
    }

    pub fn readsWrites(read_slots: SlotSet, write_slots: SlotSet) Effect {
        return .{ .ReadsWrites = .{ .reads = read_slots, .writes = write_slots } };
    }

    pub fn deinit(self: *Effect, allocator: std.mem.Allocator) void {
        switch (self.*) {
            .Pure => {},
            .Reads => |*slots| {
                log.debug("Effect.deinit Reads slots=0x{x}", .{@intFromPtr(slots)});
                slots.deinit(allocator);
            },
            .Writes => |*slots| {
                log.debug("Effect.deinit Writes slots=0x{x}", .{@intFromPtr(slots)});
                slots.deinit(allocator);
            },
            .ReadsWrites => |*rw| {
                log.debug("Effect.deinit ReadsWrites reads=0x{x} writes=0x{x}", .{
                    @intFromPtr(&rw.reads),
                    @intFromPtr(&rw.writes),
                });
                rw.reads.deinit(allocator);
                rw.writes.deinit(allocator);
            },
        }
    }
};

pub fn combineEffects(
    eff1: Effect,
    eff2: Effect,
    allocator: std.mem.Allocator,
) Effect {
    const SlotSetOps = struct {
        fn copy(alloc: std.mem.Allocator, set: SlotSet) SlotSet {
            var out = SlotSet.init(alloc);
            for (set.slots.items) |slot| {
                out.addWithSrc(alloc, slot, @src()) catch {};
            }
            return out;
        }

        fn merge(alloc: std.mem.Allocator, a: SlotSet, b: SlotSet) SlotSet {
            var out = @This().copy(alloc, a);
            for (b.slots.items) |slot| {
                if (!out.contains(slot)) {
                    out.addWithSrc(alloc, slot, @src()) catch {};
                }
            }
            return out;
        }
    };

    return switch (eff1) {
        .Pure => switch (eff2) {
            .Pure => Effect.pure(),
            .Reads => |r| Effect.reads(SlotSetOps.copy(allocator, r)),
            .Writes => |w| Effect.writes(SlotSetOps.copy(allocator, w)),
            .ReadsWrites => |rw| Effect.readsWrites(
                SlotSetOps.copy(allocator, rw.reads),
                SlotSetOps.copy(allocator, rw.writes),
            ),
        },
        .Reads => |r1| switch (eff2) {
            .Pure => Effect.reads(SlotSetOps.copy(allocator, r1)),
            .Reads => Effect.reads(SlotSetOps.merge(allocator, r1, eff2.Reads)),
            .Writes => Effect.readsWrites(
                SlotSetOps.copy(allocator, r1),
                SlotSetOps.copy(allocator, eff2.Writes),
            ),
            .ReadsWrites => |rw2| Effect.readsWrites(
                SlotSetOps.merge(allocator, r1, rw2.reads),
                SlotSetOps.copy(allocator, rw2.writes),
            ),
        },
        .Writes => |w1| switch (eff2) {
            .Pure => Effect.writes(SlotSetOps.copy(allocator, w1)),
            .Writes => Effect.writes(SlotSetOps.merge(allocator, w1, eff2.Writes)),
            .Reads => Effect.readsWrites(
                SlotSetOps.copy(allocator, eff2.Reads),
                SlotSetOps.copy(allocator, w1),
            ),
            .ReadsWrites => |rw2| Effect.readsWrites(
                SlotSetOps.copy(allocator, rw2.reads),
                SlotSetOps.merge(allocator, w1, rw2.writes),
            ),
        },
        .ReadsWrites => |rw1| switch (eff2) {
            .Pure => Effect.readsWrites(
                SlotSetOps.copy(allocator, rw1.reads),
                SlotSetOps.copy(allocator, rw1.writes),
            ),
            .Reads => |r2| Effect.readsWrites(
                SlotSetOps.merge(allocator, rw1.reads, r2),
                SlotSetOps.copy(allocator, rw1.writes),
            ),
            .Writes => |w2| Effect.readsWrites(
                SlotSetOps.copy(allocator, rw1.reads),
                SlotSetOps.merge(allocator, rw1.writes, w2),
            ),
            .ReadsWrites => |rw2| Effect.readsWrites(
                SlotSetOps.merge(allocator, rw1.reads, rw2.reads),
                SlotSetOps.merge(allocator, rw1.writes, rw2.writes),
            ),
        },
    };
}

pub fn consumeEffects(
    allocator: std.mem.Allocator,
    eff1: Effect,
    eff2: Effect,
) Effect {
    const combined = combineEffects(eff1, eff2, allocator);
    var tmp1 = eff1;
    var tmp2 = eff2;
    tmp1.deinit(allocator);
    tmp2.deinit(allocator);
    return combined;
}

pub fn mergeEffects(
    allocator: std.mem.Allocator,
    acc: *Effect,
    next: Effect,
) void {
    acc.* = consumeEffects(allocator, acc.*, next);
}

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

    pub inline fn add(self: *SlotSet, allocator: std.mem.Allocator, slot: []const u8) !void {
        return self.addWithSrc(allocator, slot, @src());
    }

    pub fn addWithSrc(
        self: *SlotSet,
        allocator: std.mem.Allocator,
        slot: []const u8,
        src: std.builtin.SourceLocation,
    ) !void {
        try self.slots.append(allocator, slot);
        _ = src;
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
    type_storage_allocator: std.mem.Allocator,
    symbol_table: *SymbolTable,
    current_scope: ?*Scope,
    current_contract_name: ?[]const u8 = null,
    in_try_block: bool = false,
    last_try_error_union: ?TypeInfo = null,
    current_function_return_type: ?TypeInfo = null,
    validation: *validation.ValidationSystem,
    utils: *utils.Utils,
    refinement_system: *refinements.RefinementSystem,
    // optional: function registry for argument validation (set by TypeResolver)
    // type-erased to avoid circular dependency with ast.zig
    function_registry: ?*std.StringHashMap(*anyopaque) = null,
    // builtin registry for resolving std.* constants and functions
    builtin_registry: ?*const semantics.builtins.BuiltinRegistry = null,
    // Comptime environment (Zig-style comptime system)
    comptime_env: comptime_eval.CtEnv,
    // Comptime constant pool (persists across evaluations)
    comptime_pool: comptime_eval.ConstPool,

    pub fn init(
        allocator: std.mem.Allocator,
        type_storage_allocator: std.mem.Allocator,
        symbol_table: *SymbolTable,
        validation_sys: *validation.ValidationSystem,
        utils_sys: *utils.Utils,
        refinement_sys: *refinements.RefinementSystem,
    ) CoreResolver {
        return CoreResolver{
            .allocator = allocator,
            .type_storage_allocator = type_storage_allocator,
            .symbol_table = symbol_table,
            .current_scope = symbol_table.root,
            .current_contract_name = null,
            .in_try_block = false,
            .current_function_return_type = null,
            .validation = validation_sys,
            .utils = utils_sys,
            .refinement_system = refinement_sys,
            .builtin_registry = &symbol_table.builtin_registry,
            .comptime_env = comptime_eval.CtEnv.init(allocator, comptime_eval.EvalConfig.default),
            .comptime_pool = comptime_eval.ConstPool.init(allocator),
        };
    }

    pub fn deinit(self: *CoreResolver) void {
        self.comptime_env.deinit();
        self.comptime_pool.deinit();
    }

    /// Synthesize (infer) type for an expression
    pub fn synthExpr(self: *CoreResolver, expr: *ast.Expressions.ExprNode) !Typed {
        return @import("expression.zig").synthExpr(self, expr);
    }

    /// Check expression against expected type
    pub fn checkExpr(self: *CoreResolver, expr: *ast.Expressions.ExprNode, expected: TypeInfo) !Typed {
        return @import("expression.zig").checkExpr(self, expr, expected);
    }

    pub fn validateErrorUnionType(self: *CoreResolver, type_info: TypeInfo) TypeResolutionError!void {
        if (type_info.ora_type) |ora_type| {
            switch (ora_type) {
                .error_union => {},
                ._union => |union_types| {
                    if (union_types.len == 0) return;
                    for (union_types, 0..) |union_type, i| {
                        if (i == 0) {
                            if (union_type != .error_union) {
                                return TypeResolutionError.InvalidErrorUsage;
                            }
                            continue;
                        }

                        switch (union_type) {
                            .struct_type => |error_name| {
                                if (self.symbol_table.error_signatures.get(error_name)) |_| {
                                    continue;
                                }
                                const root_scope: ?*const Scope = @as(?*const Scope, @ptrCast(&self.symbol_table.root));
                                if (SymbolTable.findUp(root_scope, error_name)) |symbol| {
                                    if (symbol.kind == .Error) continue;
                                }
                                return TypeResolutionError.InvalidErrorUsage;
                            },
                            else => return TypeResolutionError.InvalidErrorUsage,
                        }
                    }
                },
                else => {},
            }
        }
    }

    /// Fix a struct_type OraType that is actually an enum or bitfield.
    /// The parser initially assumes all named types are struct_type;
    /// this corrects based on the symbol table.
    pub fn resolveNamedOraType(self: *CoreResolver, ot: *OraType) void {
        if (ot.* == .struct_type) {
            const name = ot.struct_type;
            const root_scope: ?*const Scope = @as(?*const Scope, @ptrCast(self.symbol_table.root));
            if (SymbolTable.findUp(root_scope, name)) |sym| {
                if (sym.kind == .Enum) {
                    ot.* = OraType{ .enum_type = name };
                } else if (sym.kind == .Bitfield) {
                    ot.* = OraType{ .bitfield_type = name };
                }
            }
        }
    }

    /// Resolve types for a statement
    pub fn resolveStatement(self: *CoreResolver, stmt: *ast.Statements.StmtNode, context: TypeContext) !Typed {
        return @import("statement.zig").resolveStatement(self, stmt, context);
    }

    /// Look up identifier in symbol table
    pub fn lookupIdentifier(self: *CoreResolver, name: []const u8) ?TypeInfo {
        return @import("identifier.zig").lookupIdentifier(self, name);
    }

    /// Set a compile-time value by name (scoped to current comptime env scope)
    pub fn setComptimeValue(self: *CoreResolver, scope: *Scope, name: []const u8, value: comptime_eval.CtValue) !void {
        _ = scope;
        // Use scoped bind so function-local consts don't leak across functions.
        // Top-level consts (outside functions) use the root scope and remain visible.
        _ = try self.comptime_env.bind(name, value);
    }

    /// Set a compile-time integer constant
    pub fn setComptimeInteger(self: *CoreResolver, scope: *Scope, name: []const u8, value: u256) !void {
        try self.setComptimeValue(scope, name, .{ .integer = value });
    }

    /// Set a compile-time boolean constant
    pub fn setComptimeBool(self: *CoreResolver, scope: *Scope, name: []const u8, value: bool) !void {
        try self.setComptimeValue(scope, name, .{ .boolean = value });
    }

    /// Lookup a compile-time value by name
    pub fn lookupComptimeValue(self: *CoreResolver, name: []const u8) ?comptime_eval.CtValue {
        return self.comptime_env.lookupValue(name);
    }

    /// Lookup a compile-time integer by name
    pub fn lookupComptimeInteger(self: *CoreResolver, name: []const u8) ?u256 {
        if (self.comptime_env.lookupValue(name)) |ct_value| {
            if (ct_value == .integer) return ct_value.integer;
        }
        return null;
    }

    /// Get the ConstPool
    pub fn getConstPool(self: *CoreResolver) *comptime_eval.ConstPool {
        return &self.comptime_pool;
    }

    /// Get the CtEnv
    pub fn getComptimeEnv(self: *CoreResolver) *comptime_eval.CtEnv {
        return &self.comptime_env;
    }

    /// Evaluate an AST expression at compile time
    pub fn evaluateConstantExpression(self: *CoreResolver, expr: *ast.Expressions.ExprNode) comptime_eval.AstEvalResult {
        const lookup = comptime_eval.IdentifierLookup{
            .ctx = self,
            .lookupFn = lookupComptimeValueThunk,
            .enumLookupFn = lookupEnumValueThunk,
        };
        const fn_lookup = comptime_eval.FunctionLookup{
            .ctx = self,
            .lookupFn = lookupComptimeFnThunk,
        };
        var eval = comptime_eval.AstEvaluator.initWithFnLookup(&self.comptime_env, .try_eval, .forgiving, lookup, fn_lookup);
        return eval.evalExpr(expr);
    }

    /// Evaluate an AST expression and intern result into ConstPool
    /// Returns a ConstId that can be stored in symbol tables
    pub fn evaluateAndIntern(self: *CoreResolver, expr: *ast.Expressions.ExprNode) comptime_eval.InternResult {
        const lookup = comptime_eval.IdentifierLookup{
            .ctx = self,
            .lookupFn = lookupComptimeValueThunk,
            .enumLookupFn = lookupEnumValueThunk,
        };
        var eval = comptime_eval.AstEvaluator.initWithPool(&self.comptime_env, .try_eval, .forgiving, lookup, &self.comptime_pool);
        return eval.evalAndIntern(expr);
    }

    /// Intern a CtValue into the ConstPool
    pub fn internValue(self: *CoreResolver, value: comptime_eval.CtValue) !comptime_eval.ConstId {
        return self.comptime_pool.intern(&self.comptime_env, value);
    }
};

pub fn lookupComptimeValueThunk(ctx: *anyopaque, name: []const u8) ?comptime_eval.CtValue {
    const self: *CoreResolver = @ptrCast(@alignCast(ctx));
    return self.lookupComptimeValue(name);
}

pub fn lookupEnumValueThunk(ctx: *anyopaque, enum_name: []const u8, variant_name: []const u8) ?comptime_eval.CtValue {
    const self: *CoreResolver = @ptrCast(@alignCast(ctx));
    if (self.symbol_table.enum_variants.get(enum_name)) |variants| {
        for (variants, 0..) |variant, i| {
            if (!std.mem.eql(u8, variant, variant_name)) continue;
            return comptime_eval.CtValue{ .integer = @as(u256, @intCast(i)) };
        }
    }
    return null;
}

fn lookupComptimeFnThunk(ctx: *anyopaque, name: []const u8) ?comptime_eval.ComptimeFnInfo {
    const self: *CoreResolver = @ptrCast(@alignCast(ctx));
    // Check function purity — only pure functions can be evaluated at comptime
    if (self.symbol_table.function_effects.get(name)) |effect| {
        if (effect != .Pure) return null; // has side effects
    }
    // else: effect not yet computed (first pass) — optimistically allow

    // Look up the function body in the registry
    const registry = self.function_registry orelse return null;
    const fn_node_ptr = registry.get(name) orelse return null;
    const fn_node: *const ast.FunctionNode = @ptrCast(@alignCast(fn_node_ptr));

    // Transitive purity check: walk the body for calls to non-pure functions
    // and runtime-only builtins
    if (!isBodyComptimeSafe(&fn_node.body, self.symbol_table, registry)) return null;

    // Build param names slice (use type_storage_allocator — arena, freed at end of compilation)
    const param_names = self.type_storage_allocator.alloc([]const u8, fn_node.parameters.len) catch return null;
    const is_comptime_param = self.type_storage_allocator.alloc(bool, fn_node.parameters.len) catch return null;
    for (fn_node.parameters, 0..) |param, i| {
        param_names[i] = param.name;
        is_comptime_param[i] = param.is_comptime;
    }

    return comptime_eval.ComptimeFnInfo{
        .body = &fn_node.body,
        .param_names = param_names,
        .is_comptime_param = is_comptime_param,
    };
}

/// Runtime-only builtins that cannot be used in comptime context
const runtime_only_builtins = [_][]const u8{
    "msg.sender",   "msg.value",  "msg.data",
    "block.number", "block.timestamp", "block.coinbase",
    "tx.origin",    "tx.gasprice",
    "address(this)",
};

/// Check if a function body is safe for comptime evaluation.
/// Returns false if it calls non-pure functions or uses runtime-only builtins.
fn isBodyComptimeSafe(
    body: *const ast.Statements.BlockNode,
    symbol_table: *const SymbolTable,
    registry: *const std.StringHashMap(*anyopaque),
) bool {
    for (body.statements) |*stmt| {
        if (!isStmtComptimeSafe(stmt, symbol_table, registry)) return false;
    }
    return true;
}

fn isStmtComptimeSafe(
    stmt: *const ast.Statements.StmtNode,
    symbol_table: *const SymbolTable,
    registry: *const std.StringHashMap(*anyopaque),
) bool {
    return switch (stmt.*) {
        .Expr => |expr| isExprComptimeSafe(&expr, symbol_table, registry),
        .VariableDecl => |vd| if (vd.value) |v| isExprComptimeSafe(v, symbol_table, registry) else true,
        .Return => |ret| if (ret.value) |v| isExprComptimeSafe(&v, symbol_table, registry) else true,
        .If => |if_stmt| blk: {
            if (!isExprComptimeSafe(&if_stmt.condition, symbol_table, registry)) break :blk false;
            if (!isBodyComptimeSafe(&if_stmt.then_branch, symbol_table, registry)) break :blk false;
            if (if_stmt.else_branch) |*else_b| {
                if (!isBodyComptimeSafe(else_b, symbol_table, registry)) break :blk false;
            }
            break :blk true;
        },
        .While => |while_stmt| blk: {
            if (!isExprComptimeSafe(&while_stmt.condition, symbol_table, registry)) break :blk false;
            if (!isBodyComptimeSafe(&while_stmt.body, symbol_table, registry)) break :blk false;
            break :blk true;
        },
        .ForLoop => |for_stmt| blk: {
            if (!isExprComptimeSafe(&for_stmt.iterable, symbol_table, registry)) break :blk false;
            if (!isBodyComptimeSafe(&for_stmt.body, symbol_table, registry)) break :blk false;
            break :blk true;
        },
        .CompoundAssignment => |ca| isExprComptimeSafe(ca.value, symbol_table, registry),
        .Break, .Continue => true,
        .Assert, .Invariant, .Requires, .Ensures, .Assume, .Havoc => true,
        // Log, Lock, Unlock, TryBlock are runtime-only
        .Log, .Lock, .Unlock, .TryBlock => false,
        else => true,
    };
}

fn isExprComptimeSafe(
    expr: *const ast.Expressions.ExprNode,
    symbol_table: *const SymbolTable,
    registry: *const std.StringHashMap(*anyopaque),
) bool {
    return switch (expr.*) {
        .Call => |call| blk: {
            // Check callee name
            if (call.callee.* == .Identifier) {
                const callee_name = call.callee.Identifier.name;
                // Block runtime-only builtins
                for (&runtime_only_builtins) |builtin| {
                    if (std.mem.eql(u8, callee_name, builtin)) break :blk false;
                }
                // Check if called function is pure (transitively)
                if (callee_name.len > 0 and callee_name[0] != '@') {
                    if (symbol_table.function_effects.get(callee_name)) |effect| {
                        if (effect != .Pure) break :blk false;
                    }
                }
            } else if (call.callee.* == .FieldAccess) {
                // Check for runtime-only member access (msg.sender, block.number, etc.)
                const fa = &call.callee.FieldAccess;
                if (fa.target.* == .Identifier) {
                    const base = fa.target.Identifier.name;
                    if (std.mem.eql(u8, base, "msg") or
                        std.mem.eql(u8, base, "block") or
                        std.mem.eql(u8, base, "tx"))
                    {
                        break :blk false;
                    }
                }
            }
            // Check arguments recursively
            for (call.arguments) |arg| {
                if (!isExprComptimeSafe(arg, symbol_table, registry)) break :blk false;
            }
            break :blk true;
        },
        .Binary => |bin| isExprComptimeSafe(bin.lhs, symbol_table, registry) and
            isExprComptimeSafe(bin.rhs, symbol_table, registry),
        .Unary => |un| isExprComptimeSafe(un.operand, symbol_table, registry),
        .FieldAccess => |fa| blk: {
            // Block runtime-only field access patterns
            if (fa.target.* == .Identifier) {
                const base = fa.target.Identifier.name;
                if (std.mem.eql(u8, base, "msg") or
                    std.mem.eql(u8, base, "block") or
                    std.mem.eql(u8, base, "tx"))
                {
                    break :blk false;
                }
            }
            break :blk isExprComptimeSafe(fa.target, symbol_table, registry);
        },
        .Index => |idx| isExprComptimeSafe(idx.target, symbol_table, registry) and
            isExprComptimeSafe(idx.index, symbol_table, registry),
        .Cast => |cast| isExprComptimeSafe(cast.operand, symbol_table, registry),
        .Literal, .Identifier, .EnumLiteral => true,
        .Tuple => |tup| blk: {
            for (tup.elements) |el| {
                if (!isExprComptimeSafe(el, symbol_table, registry)) break :blk false;
            }
            break :blk true;
        },
        else => true, // Conservatively allow others
    };
}
