//! Comptime Evaluator
//!
//! Interprets typed IR (HIR) at compile time with:
//! - Stage validity checking
//! - Fuel-based limits
//! - Copy-on-write mutation semantics

const std = @import("std");
const value = @import("value.zig");
const error_mod = @import("error.zig");
const limits = @import("limits.zig");
const env_mod = @import("env.zig");
const stage_mod = @import("stage.zig");

const CtValue = value.CtValue;
const SlotId = value.SlotId;
const HeapId = value.HeapId;
const ConstId = value.ConstId;
const TypeId = value.TypeId;

const CtError = error_mod.CtError;
const CtErrorKind = error_mod.CtErrorKind;
const TryEvalPolicy = error_mod.TryEvalPolicy;
const SourceSpan = error_mod.SourceSpan;

const EvalConfig = limits.EvalConfig;
const EvalStats = limits.EvalStats;
const LimitCheck = limits.LimitCheck;

const CtEnv = env_mod.CtEnv;
const Stage = stage_mod.Stage;

/// Evaluation mode
pub const EvalMode = enum {
    /// Best-effort folding: if it can't compute, return .runtime
    try_eval,

    /// Required evaluation: if it can't compute, return .err
    must_eval,

    /// Check if errors should become runtime in this mode
    pub fn convertsErrorsToRuntime(self: EvalMode, policy: TryEvalPolicy) bool {
        return self == .try_eval and policy == .forgiving;
    }
};

/// Control flow during evaluation
pub const ControlFlow = union(enum) {
    return_val: ?CtValue,
    break_val: ?CtValue,
    continue_val,
};

/// Result of evaluation
pub const EvalResult = union(enum) {
    /// Successfully evaluated to a value
    value: CtValue,

    /// Cannot evaluate at comptime (try_eval only)
    runtime: void,

    /// Evaluation error
    err: CtError,

    /// Control flow
    control: ControlFlow,

    pub fn isSuccess(self: EvalResult) bool {
        return self == .value;
    }

    pub fn isRuntime(self: EvalResult) bool {
        return self == .runtime;
    }

    pub fn isError(self: EvalResult) bool {
        return self == .err;
    }

    pub fn getValue(self: EvalResult) ?CtValue {
        return if (self == .value) self.value else null;
    }

    pub fn getError(self: EvalResult) ?CtError {
        return if (self == .err) self.err else null;
    }

    /// Create a success result
    pub fn ok(v: CtValue) EvalResult {
        return .{ .value = v };
    }

    /// Create a runtime result
    pub fn asRuntime() EvalResult {
        return .{ .runtime = {} };
    }

    /// Create an error result
    pub fn fail(e: CtError) EvalResult {
        return .{ .err = e };
    }
};

/// Comptime evaluator
pub const Evaluator = struct {
    env: *CtEnv,
    mode: EvalMode,
    policy: TryEvalPolicy,
    current_span: SourceSpan,

    pub fn init(env: *CtEnv, mode: EvalMode, policy: TryEvalPolicy) Evaluator {
        return .{
            .env = env,
            .mode = mode,
            .policy = policy,
            .current_span = .{ .line = 0, .column = 0, .length = 0 },
        };
    }

    // ========================================================================
    // Core Evaluation
    // ========================================================================

    /// Evaluate a binary operation
    pub fn evalBinaryOp(
        self: *Evaluator,
        op: BinaryOp,
        lhs: CtValue,
        rhs: CtValue,
        span: SourceSpan,
    ) EvalResult {
        // Check fuel
        if (self.step(span)) |result| return result;

        // Handle logical operators (require boolean operands)
        switch (op) {
            .land => {
                const l = switch (lhs) {
                    .boolean => |v| v,
                    else => return self.typeError(span, "logical and requires boolean operands"),
                };
                const r = switch (rhs) {
                    .boolean => |v| v,
                    else => return self.typeError(span, "logical and requires boolean operands"),
                };
                return EvalResult.ok(.{ .boolean = l and r });
            },
            .lor => {
                const l = switch (lhs) {
                    .boolean => |v| v,
                    else => return self.typeError(span, "logical or requires boolean operands"),
                };
                const r = switch (rhs) {
                    .boolean => |v| v,
                    else => return self.typeError(span, "logical or requires boolean operands"),
                };
                return EvalResult.ok(.{ .boolean = l or r });
            },
            else => {},
        }

        // Integer operations
        const l = switch (lhs) {
            .integer => |v| v,
            else => return self.typeError(span, "binary operation requires integer operands"),
        };
        const r = switch (rhs) {
            .integer => |v| v,
            else => return self.typeError(span, "binary operation requires integer operands"),
        };

        return switch (op) {
            .add => self.checkedAdd(l, r, span),
            .sub => self.checkedSub(l, r, span),
            .mul => self.checkedMul(l, r, span),
            .div => self.checkedDiv(l, r, span),
            .mod => self.checkedMod(l, r, span),
            .wadd => EvalResult.ok(.{ .integer = l +% r }),
            .wsub => EvalResult.ok(.{ .integer = l -% r }),
            .wmul => EvalResult.ok(.{ .integer = l *% r }),
            .eq => EvalResult.ok(.{ .boolean = l == r }),
            .neq => EvalResult.ok(.{ .boolean = l != r }),
            .lt => EvalResult.ok(.{ .boolean = l < r }),
            .lte => EvalResult.ok(.{ .boolean = l <= r }),
            .gt => EvalResult.ok(.{ .boolean = l > r }),
            .gte => EvalResult.ok(.{ .boolean = l >= r }),
            .band => EvalResult.ok(.{ .integer = l & r }),
            .bor => EvalResult.ok(.{ .integer = l | r }),
            .bxor => EvalResult.ok(.{ .integer = l ^ r }),
            .shl => self.checkedShl(l, r, span),
            .shr => if (r >= 256) EvalResult.ok(.{ .integer = 0 }) else EvalResult.ok(.{ .integer = l >> @intCast(r) }),
            .wshl => if (r >= 256) EvalResult.ok(.{ .integer = 0 }) else EvalResult.ok(.{ .integer = l << @intCast(r) }),
            .wshr => if (r >= 256) EvalResult.ok(.{ .integer = 0 }) else EvalResult.ok(.{ .integer = l >> @intCast(r) }),
            .land, .lor => unreachable, // handled above
        };
    }

    /// Evaluate a unary operation
    pub fn evalUnaryOp(
        self: *Evaluator,
        op: UnaryOp,
        operand: CtValue,
        span: SourceSpan,
    ) EvalResult {
        if (self.step(span)) |result| return result;

        return switch (op) {
            .neg => switch (operand) {
                .integer => |v| {
                    // For u256, negation is wrapping (two's complement)
                    const result = 0 -% v;
                    return EvalResult.ok(.{ .integer = result });
                },
                else => self.typeError(span, "negation requires integer operand"),
            },
            .not => switch (operand) {
                .boolean => |v| EvalResult.ok(.{ .boolean = !v }),
                else => self.typeError(span, "logical not requires boolean operand"),
            },
            .bnot => switch (operand) {
                .integer => |v| EvalResult.ok(.{ .integer = ~v }),
                else => self.typeError(span, "bitwise not requires integer operand"),
            },
        };
    }

    /// Evaluate array index access
    pub fn evalIndex(
        self: *Evaluator,
        array: CtValue,
        index: CtValue,
        span: SourceSpan,
    ) EvalResult {
        if (self.step(span)) |result| return result;

        const heap_id = switch (array) {
            .array_ref => |id| id,
            .tuple_ref => |id| id,
            .bytes_ref => |id| id,
            .string_ref => |id| id,
            else => return self.typeError(span, "indexing requires array, tuple, bytes, or string"),
        };

        const idx = switch (index) {
            .integer => |v| v,
            else => return self.typeError(span, "index must be an integer"),
        };

        // Check bounds and get element
        return switch (array) {
            .array_ref => {
                const arr = self.env.heap.getArray(heap_id);
                if (idx >= arr.elems.len) {
                    return self.indexError(span, idx, arr.elems.len);
                }
                return EvalResult.ok(arr.elems[@intCast(idx)]);
            },
            .tuple_ref => {
                const tup = self.env.heap.getTuple(heap_id);
                if (idx >= tup.elems.len) {
                    return self.indexError(span, idx, tup.elems.len);
                }
                return EvalResult.ok(tup.elems[@intCast(idx)]);
            },
            .bytes_ref => {
                const bytes = self.env.heap.getBytes(heap_id);
                if (idx >= bytes.len) {
                    return self.indexError(span, idx, bytes.len);
                }
                return EvalResult.ok(.{ .integer = bytes[@intCast(idx)] });
            },
            .string_ref => {
                const str = self.env.heap.getString(heap_id);
                if (idx >= str.len) {
                    return self.indexError(span, idx, str.len);
                }
                return EvalResult.ok(.{ .integer = str[@intCast(idx)] });
            },
            else => unreachable,
        };
    }

    /// Evaluate field access
    pub fn evalFieldAccess(
        self: *Evaluator,
        struct_val: CtValue,
        field_id: value.FieldId,
        span: SourceSpan,
    ) EvalResult {
        if (self.step(span)) |result| return result;

        const heap_id = switch (struct_val) {
            .struct_ref => |id| id,
            else => return self.typeError(span, "field access requires struct"),
        };

        const s = self.env.heap.getStruct(heap_id);
        for (s.fields) |f| {
            if (f.field_id == field_id) {
                return EvalResult.ok(f.value);
            }
        }

        return EvalResult.fail(CtError.init(.field_not_found, span, "field not found"));
    }

    // ========================================================================
    // Stage Checking
    // ========================================================================

    /// Check if an operation's stage is valid for current mode
    pub fn checkStage(self: *Evaluator, op_stage: Stage, span: SourceSpan) ?EvalResult {
        return switch (op_stage) {
            .comptime_only, .comptime_ok => null, // OK in both modes
            .runtime_only => switch (self.mode) {
                .must_eval => EvalResult.fail(CtError.stageViolation(span, "runtime-only operation")),
                .try_eval => EvalResult.asRuntime(),
            },
        };
    }

    /// Check knownness - returns error if value depends on runtime
    pub fn requireKnown(self: *Evaluator, known: bool, span: SourceSpan, reason: []const u8) ?EvalResult {
        if (known) return null;

        return switch (self.mode) {
            .must_eval => EvalResult.fail(CtError.notComptime(span, reason)),
            .try_eval => EvalResult.asRuntime(),
        };
    }

    // ========================================================================
    // Fuel / Limits
    // ========================================================================

    /// Consume one step of fuel, returns error if limit exceeded
    pub fn step(self: *Evaluator, span: SourceSpan) ?EvalResult {
        self.current_span = span;
        self.env.stats.recordStep();

        const check = LimitCheck.init(self.env.config, &self.env.stats);
        if (check.checkSteps()) |kind| {
            return EvalResult.fail(CtError.init(kind, span, "evaluation step limit exceeded"));
        }
        return null;
    }

    // ========================================================================
    // Arithmetic with Overflow Checking
    // ========================================================================

    fn checkedAdd(self: *Evaluator, a: u256, b: u256, span: SourceSpan) EvalResult {
        const result, const overflow = @addWithOverflow(a, b);
        if (overflow != 0) {
            return self.arithmeticError(span, .overflow);
        }
        return EvalResult.ok(.{ .integer = result });
    }

    fn checkedSub(self: *Evaluator, a: u256, b: u256, span: SourceSpan) EvalResult {
        const result, const overflow = @subWithOverflow(a, b);
        if (overflow != 0) {
            return self.arithmeticError(span, .underflow);
        }
        return EvalResult.ok(.{ .integer = result });
    }

    fn checkedMul(self: *Evaluator, a: u256, b: u256, span: SourceSpan) EvalResult {
        const result, const overflow = @mulWithOverflow(a, b);
        if (overflow != 0) {
            return self.arithmeticError(span, .overflow);
        }
        return EvalResult.ok(.{ .integer = result });
    }

    fn checkedDiv(self: *Evaluator, a: u256, b: u256, span: SourceSpan) EvalResult {
        if (b == 0) {
            return self.arithmeticError(span, .division_by_zero);
        }
        return EvalResult.ok(.{ .integer = a / b });
    }

    fn checkedMod(self: *Evaluator, a: u256, b: u256, span: SourceSpan) EvalResult {
        if (b == 0) {
            return self.arithmeticError(span, .division_by_zero);
        }
        return EvalResult.ok(.{ .integer = a % b });
    }

    fn checkedShl(_: *Evaluator, a: u256, b: u256, _: SourceSpan) EvalResult {
        if (b >= 256) {
            return EvalResult.ok(.{ .integer = 0 });
        }
        return EvalResult.ok(.{ .integer = a << @intCast(b) });
    }

    // ========================================================================
    // Error Helpers
    // ========================================================================

    fn arithmeticError(self: *Evaluator, span: SourceSpan, kind: CtErrorKind) EvalResult {
        if (self.mode.convertsErrorsToRuntime(self.policy)) {
            return EvalResult.asRuntime();
        }
        return EvalResult.fail(CtError.init(kind, span, kind.description()));
    }

    fn typeError(self: *Evaluator, span: SourceSpan, message: []const u8) EvalResult {
        if (self.mode.convertsErrorsToRuntime(self.policy)) {
            return EvalResult.asRuntime();
        }
        return EvalResult.fail(CtError.init(.type_mismatch, span, message));
    }

    fn indexError(self: *Evaluator, span: SourceSpan, index: u256, len: usize) EvalResult {
        _ = index;
        _ = len;
        if (self.mode.convertsErrorsToRuntime(self.policy)) {
            return EvalResult.asRuntime();
        }
        return EvalResult.fail(CtError.init(.index_out_of_bounds, span, "index out of bounds"));
    }
};

/// Binary operation types
pub const BinaryOp = enum {
    // Arithmetic
    add,
    sub,
    mul,
    div,
    mod,
    wadd,
    wsub,
    wmul,

    // Comparison
    eq,
    neq,
    lt,
    lte,
    gt,
    gte,

    // Bitwise
    band,
    bor,
    bxor,
    shl,
    shr,
    wshl,
    wshr,

    // Logical
    land,
    lor,
};

/// Unary operation types
pub const UnaryOp = enum {
    neg,
    not,
    bnot,
};

// ============================================================================
// Tests
// ============================================================================

test "Evaluator basic arithmetic" {
    var env = CtEnv.init(std.testing.allocator, .{});
    defer env.deinit();

    var eval = Evaluator.init(&env, .must_eval, .strict);
    const span = SourceSpan{ .line = 1, .column = 1, .length = 1 };

    // Addition
    const add_result = eval.evalBinaryOp(.add, .{ .integer = 10 }, .{ .integer = 5 }, span);
    try std.testing.expectEqual(@as(u256, 15), add_result.getValue().?.integer);

    // Subtraction
    const sub_result = eval.evalBinaryOp(.sub, .{ .integer = 10 }, .{ .integer = 5 }, span);
    try std.testing.expectEqual(@as(u256, 5), sub_result.getValue().?.integer);

    // Multiplication
    const mul_result = eval.evalBinaryOp(.mul, .{ .integer = 10 }, .{ .integer = 5 }, span);
    try std.testing.expectEqual(@as(u256, 50), mul_result.getValue().?.integer);

    // Division
    const div_result = eval.evalBinaryOp(.div, .{ .integer = 10 }, .{ .integer = 5 }, span);
    try std.testing.expectEqual(@as(u256, 2), div_result.getValue().?.integer);
}

test "Evaluator comparison ops" {
    var env = CtEnv.init(std.testing.allocator, .{});
    defer env.deinit();

    var eval = Evaluator.init(&env, .must_eval, .strict);
    const span = SourceSpan{ .line = 1, .column = 1, .length = 1 };

    const eq_result = eval.evalBinaryOp(.eq, .{ .integer = 5 }, .{ .integer = 5 }, span);
    try std.testing.expect(eq_result.getValue().?.boolean);

    const lt_result = eval.evalBinaryOp(.lt, .{ .integer = 3 }, .{ .integer = 5 }, span);
    try std.testing.expect(lt_result.getValue().?.boolean);

    const gt_result = eval.evalBinaryOp(.gt, .{ .integer = 3 }, .{ .integer = 5 }, span);
    try std.testing.expect(!gt_result.getValue().?.boolean);
}

test "Evaluator division by zero" {
    var env = CtEnv.init(std.testing.allocator, .{});
    defer env.deinit();

    var eval = Evaluator.init(&env, .must_eval, .strict);
    const span = SourceSpan{ .line = 1, .column = 1, .length = 1 };

    const result = eval.evalBinaryOp(.div, .{ .integer = 10 }, .{ .integer = 0 }, span);
    try std.testing.expect(result.isError());
    try std.testing.expectEqual(CtErrorKind.division_by_zero, result.getError().?.kind);
}

test "Evaluator overflow detection" {
    var env = CtEnv.init(std.testing.allocator, .{});
    defer env.deinit();

    var eval = Evaluator.init(&env, .must_eval, .strict);
    const span = SourceSpan{ .line = 1, .column = 1, .length = 1 };

    // Max u256 + 1 should overflow
    const max_u256: u256 = std.math.maxInt(u256);
    const result = eval.evalBinaryOp(.add, .{ .integer = max_u256 }, .{ .integer = 1 }, span);
    try std.testing.expect(result.isError());
    try std.testing.expectEqual(CtErrorKind.overflow, result.getError().?.kind);
}

test "Evaluator try_eval forgiving mode" {
    var env = CtEnv.init(std.testing.allocator, .{});
    defer env.deinit();

    var eval = Evaluator.init(&env, .try_eval, .forgiving);
    const span = SourceSpan{ .line = 1, .column = 1, .length = 1 };

    // Division by zero in forgiving mode returns runtime
    const result = eval.evalBinaryOp(.div, .{ .integer = 10 }, .{ .integer = 0 }, span);
    try std.testing.expect(result.isRuntime());
}

test "Evaluator unary operations" {
    var env = CtEnv.init(std.testing.allocator, .{});
    defer env.deinit();

    var eval = Evaluator.init(&env, .must_eval, .strict);
    const span = SourceSpan{ .line = 1, .column = 1, .length = 1 };

    // Logical not
    const not_result = eval.evalUnaryOp(.not, .{ .boolean = true }, span);
    try std.testing.expect(!not_result.getValue().?.boolean);

    // Bitwise not
    const bnot_result = eval.evalUnaryOp(.bnot, .{ .integer = 0 }, span);
    try std.testing.expectEqual(std.math.maxInt(u256), bnot_result.getValue().?.integer);
}

test "Evaluator array indexing" {
    var env = CtEnv.init(std.testing.allocator, .{});
    defer env.deinit();

    // Create array on heap
    const arr_id = try env.heap.allocArray(&[_]CtValue{
        .{ .integer = 10 },
        .{ .integer = 20 },
        .{ .integer = 30 },
    });

    var eval = Evaluator.init(&env, .must_eval, .strict);
    const span = SourceSpan{ .line = 1, .column = 1, .length = 1 };

    const result = eval.evalIndex(.{ .array_ref = arr_id }, .{ .integer = 1 }, span);
    try std.testing.expectEqual(@as(u256, 20), result.getValue().?.integer);

    // Out of bounds
    const oob_result = eval.evalIndex(.{ .array_ref = arr_id }, .{ .integer = 10 }, span);
    try std.testing.expect(oob_result.isError());
}

test "Evaluator stage checking" {
    var env = CtEnv.init(std.testing.allocator, .{});
    defer env.deinit();

    const span = SourceSpan{ .line = 1, .column = 1, .length = 1 };

    // must_eval rejects runtime_only
    {
        var eval = Evaluator.init(&env, .must_eval, .strict);
        const result = eval.checkStage(.runtime_only, span);
        try std.testing.expect(result != null);
        try std.testing.expect(result.?.isError());
    }

    // try_eval returns runtime for runtime_only
    {
        var eval = Evaluator.init(&env, .try_eval, .strict);
        const result = eval.checkStage(.runtime_only, span);
        try std.testing.expect(result != null);
        try std.testing.expect(result.?.isRuntime());
    }

    // comptime_ok is always allowed
    {
        var eval = Evaluator.init(&env, .must_eval, .strict);
        const result = eval.checkStage(.comptime_ok, span);
        try std.testing.expect(result == null);
    }
}

test "Evaluator wrapping arithmetic and shift ops" {
    var env = CtEnv.init(std.testing.allocator, .{});
    defer env.deinit();

    var eval = Evaluator.init(&env, .must_eval, .strict);
    const span = SourceSpan{ .line = 1, .column = 1, .length = 1 };
    const max_u256: u256 = std.math.maxInt(u256);

    const add_wrap = eval.evalBinaryOp(.wadd, .{ .integer = max_u256 }, .{ .integer = 1 }, span);
    try std.testing.expectEqual(@as(u256, 0), add_wrap.getValue().?.integer);

    const sub_wrap = eval.evalBinaryOp(.wsub, .{ .integer = 0 }, .{ .integer = 1 }, span);
    try std.testing.expectEqual(max_u256, sub_wrap.getValue().?.integer);

    const mul_wrap = eval.evalBinaryOp(.wmul, .{ .integer = max_u256 }, .{ .integer = 2 }, span);
    try std.testing.expectEqual(max_u256 - 1, mul_wrap.getValue().?.integer);

    const shl_wrap = eval.evalBinaryOp(.wshl, .{ .integer = 1 }, .{ .integer = 255 }, span);
    try std.testing.expectEqual((@as(u256, 1) << 255), shl_wrap.getValue().?.integer);

    const shr_wrap = eval.evalBinaryOp(.wshr, .{ .integer = 1 }, .{ .integer = 300 }, span);
    try std.testing.expectEqual(@as(u256, 0), shr_wrap.getValue().?.integer);
}
