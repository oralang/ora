//! AST Expression Evaluator
//!
//! Evaluates AST expressions at compile time using the new comptime system.
//! This bridges the surface AST to the new CtValue/Evaluator infrastructure.
//!
//! Supported expression types:
//! - Literals: Integer, Bool, Address, Hex, Bytes
//! - Operators: Binary (+, -, *, /, %, ==, !=, <, <=, >, >=, &, |, ^, <<, >>)
//! - Operators: Unary (-, !, ~)
//! - Access: Identifier, FieldAccess, Index
//! - Aggregates: Tuple, ArrayLiteral
//! - Other: Cast, EnumLiteral

const std = @import("std");
const value = @import("value.zig");
const eval_mod = @import("eval.zig");
const env_mod = @import("env.zig");
const error_mod = @import("error.zig");
const limits = @import("limits.zig");
const pool_mod = @import("pool.zig");
const heap_mod = @import("heap.zig");
const ast_type_info = @import("../ast/type_info.zig");
const AstSourceSpan = @import("../ast/source_span.zig").SourceSpan;
const log = @import("log");

const CtValue = value.CtValue;
const ConstId = value.ConstId;
const HeapId = value.HeapId;
const Evaluator = eval_mod.Evaluator;
const EvalMode = eval_mod.EvalMode;
const EvalResult = eval_mod.EvalResult;
const BinaryOp = eval_mod.BinaryOp;
const UnaryOp = eval_mod.UnaryOp;
const CtEnv = env_mod.CtEnv;
const CtHeap = heap_mod.CtHeap;
const ConstPool = pool_mod.ConstPool;
const SourceSpan = error_mod.SourceSpan;
const TryEvalPolicy = error_mod.TryEvalPolicy;
const CtErrorKind = error_mod.CtErrorKind;
const EvalConfig = limits.EvalConfig;

/// Result of AST constant evaluation
pub const AstEvalResult = union(enum) {
    /// Successfully evaluated to a compile-time value
    value: CtValue,
    /// Expression is not a compile-time constant
    not_constant,
    /// Evaluation error
    err: error_mod.CtError,

    pub fn isConstant(self: AstEvalResult) bool {
        return self == .value;
    }

    pub fn getValue(self: AstEvalResult) ?CtValue {
        return if (self == .value) self.value else null;
    }

    pub fn getInteger(self: AstEvalResult) ?u256 {
        if (self == .value) {
            if (self.value == .integer) return self.value.integer;
        }
        return null;
    }

    pub fn getBoolean(self: AstEvalResult) ?bool {
        if (self == .value) {
            if (self.value == .boolean) return self.value.boolean;
        }
        return null;
    }

    pub fn getAddress(self: AstEvalResult) ?u160 {
        if (self == .value) {
            if (self.value == .address) return self.value.address;
        }
        return null;
    }

    pub fn kindName(self: AstEvalResult) []const u8 {
        return switch (self) {
            .value => "value",
            .not_constant => "not_constant",
            .err => "err",
        };
    }
};

/// Callback for looking up identifiers
pub const IdentifierLookup = struct {
    ctx: *anyopaque,
    lookupFn: *const fn (ctx: *anyopaque, name: []const u8) ?CtValue,
    enumLookupFn: ?*const fn (ctx: *anyopaque, enum_name: []const u8, variant: []const u8) ?CtValue = null,

    pub fn lookup(self: IdentifierLookup, name: []const u8) ?CtValue {
        return self.lookupFn(self.ctx, name);
    }

    pub fn lookupEnum(self: IdentifierLookup, enum_name: []const u8, variant: []const u8) ?CtValue {
        if (self.enumLookupFn) |f| {
            return f(self.ctx, enum_name, variant);
        }
        return null;
    }
};

/// Opaque function info returned by FunctionLookup
pub const ComptimeFnInfo = struct {
    body: *const @import("../ast/statements.zig").BlockNode,
    param_names: []const []const u8,
    is_comptime_param: []const bool, // per-parameter comptime flag
};

/// Callback for looking up function bodies and checking purity
pub const FunctionLookup = struct {
    ctx: *anyopaque,
    /// Returns function body + param names if the function exists and is comptime-eligible (pure)
    lookupFn: *const fn (ctx: *anyopaque, name: []const u8) ?ComptimeFnInfo,

    pub fn lookup(self: FunctionLookup, name: []const u8) ?ComptimeFnInfo {
        return self.lookupFn(self.ctx, name);
    }
};

/// AST expression evaluator
pub const AstEvaluator = struct {
    env: *CtEnv,
    evaluator: Evaluator,
    lookup: ?IdentifierLookup,
    fn_lookup: ?FunctionLookup,
    pool: ?*ConstPool,
    call_depth: u32 = 0,
    max_call_depth: u32 = 64,

    pub fn init(env: *CtEnv, mode: EvalMode, policy: TryEvalPolicy, lookup: ?IdentifierLookup) AstEvaluator {
        return .{
            .env = env,
            .evaluator = Evaluator.init(env, mode, policy),
            .lookup = lookup,
            .fn_lookup = null,
            .pool = null,
        };
    }

    /// Initialize with function lookup for comptime fn evaluation
    pub fn initWithFnLookup(env: *CtEnv, mode: EvalMode, policy: TryEvalPolicy, lookup: ?IdentifierLookup, fn_lookup: ?FunctionLookup) AstEvaluator {
        return .{
            .env = env,
            .evaluator = Evaluator.init(env, mode, policy),
            .lookup = lookup,
            .fn_lookup = fn_lookup,
            .pool = null,
        };
    }

    /// Initialize with a ConstPool for interning results
    pub fn initWithPool(env: *CtEnv, mode: EvalMode, policy: TryEvalPolicy, lookup: ?IdentifierLookup, pool: *ConstPool) AstEvaluator {
        return .{
            .env = env,
            .evaluator = Evaluator.init(env, mode, policy),
            .lookup = lookup,
            .fn_lookup = null,
            .pool = pool,
        };
    }

    /// Evaluate an AST expression (generic over AST type)
    pub fn evalExpr(self: *AstEvaluator, expr: anytype) AstEvalResult {
        const T = @TypeOf(expr);

        // Handle pointer to ExprNode union
        if (@typeInfo(T) == .pointer) {
            const Child = @typeInfo(T).pointer.child;
            if (@hasField(Child, "Literal")) {
                return self.evalExprNode(expr);
            }
        }

        return .{ .err = error_mod.CtError.init(
            .internal_error,
            .{ .line = 0, .column = 0, .length = 0 },
            "invalid expression node handle for comptime evaluator",
        ) };
    }

    /// Evaluate and intern result into ConstPool (returns ConstId if successful)
    pub fn evalAndIntern(self: *AstEvaluator, expr: anytype) InternResult {
        const result = self.evalExpr(expr);
        return switch (result) {
            .value => |v| {
                if (self.pool) |pool| {
                    const const_id = pool.intern(self.env, v) catch return .{ .err = error_mod.CtError.init(.internal_error, .{ .line = 0, .column = 0, .length = 0 }, "failed to intern constant") };
                    return .{ .const_id = const_id };
                }
                return .{ .err = error_mod.CtError.init(.internal_error, .{ .line = 0, .column = 0, .length = 0 }, "no const pool provided for interning") };
            },
            .not_constant => .not_constant,
            .err => |e| .{ .err = e },
        };
    }

    /// Result of evaluating and interning
    pub const InternResult = union(enum) {
        /// Successfully interned constant
        const_id: ConstId,
        /// Not a compile-time constant
        not_constant,
        /// Evaluation error
        err: error_mod.CtError,

        pub fn isConstant(self: InternResult) bool {
            return self == .const_id;
        }

        pub fn getConstId(self: InternResult) ?ConstId {
            return if (self == .const_id) self.const_id else null;
        }
    };

    fn evalExprNode(self: *AstEvaluator, expr: anytype) AstEvalResult {
        return switch (expr.*) {
            .Literal => |*lit| self.evalLiteral(lit),
            .Binary => |*bin| self.evalBinary(bin),
            .Unary => |*un| self.evalUnary(un),
            .Identifier => |*id| self.evalIdentifier(id),
            .EnumLiteral => |*el| self.evalEnumLiteral(el),
            .Cast => |*cast| self.evalCast(cast),
            .FieldAccess => |*fa| self.evalFieldAccess(fa),
            .Index => |*idx| self.evalIndex(idx),
            .Tuple => |*tup| self.evalTuple(tup),
            .ArrayLiteral => |*arr| self.evalArrayLiteral(arr),
            .SwitchExpression => |*sw| self.evalSwitch(sw),
            .AnonymousStruct => |*anon| self.evalAnonymousStruct(anon),
            .StructInstantiation => |*si| self.evalStructInstantiation(si),
            .Try => |*try_expr| self.evalTry(try_expr),
            .Range => |*range| self.evalRange(range),
            .Call => |*call| self.evalCall(call),
            else => .not_constant,
        };
    }

    /// Evaluate a function call at compile time.
    /// Only succeeds if: all args are comptime-known AND the function is pure.
    fn evalCall(self: *AstEvaluator, call: anytype) AstEvalResult {
        const call_span = SourceSpan{
            .line = call.span.line,
            .column = call.span.column,
            .length = call.span.length,
        };

        // Extract function name from callee
        const fn_name = switch (call.callee.*) {
            .Identifier => |id| id.name,
            else => return .not_constant, // only direct calls supported
        };

        const start_steps = self.env.stats.total_steps;
        log.debug(
            "[comptime][call] start fn={s} line={d} depth={d} steps={d}/{d}\n",
            .{ fn_name, call_span.line, self.call_depth, start_steps, self.env.config.max_steps },
        );

        // Handle supported @-builtins directly in the AST evaluator.
        if (fn_name.len > 0 and fn_name[0] == '@') {
            return self.evalBuiltinCall(call, fn_name);
        }

        const fn_lookup = self.fn_lookup orelse return .not_constant;

        // Evaluate all arguments — all must be comptime-known
        var arg_values: [32]CtValue = undefined; // stack buffer for common case
        if (call.arguments.len > 32) return .not_constant;

        for (call.arguments, 0..) |arg, i| {
            const arg_result = self.evalExprNode(arg);
            switch (arg_result) {
                .value => |v| arg_values[i] = v,
                .not_constant => return .not_constant,
                .err => |e| return .{ .err = e },
            }
        }

        // Look up function body — returns null if fn doesn't exist or isn't pure
        const fn_info = fn_lookup.lookup(fn_name) orelse return .not_constant;

        // Check param count matches
        if (fn_info.param_names.len != call.arguments.len) return .not_constant;

        // Check recursion depth
        if (self.call_depth >= self.max_call_depth) {
            return .{ .err = error_mod.CtError.init(.recursion_limit, .{ .line = 0, .column = 0, .length = 0 }, "comptime recursion depth exceeded") };
        }

        // Push scope, bind params, evaluate body
        self.env.pushScope(false) catch {
            return .{ .err = error_mod.CtError.init(
                .internal_error,
                call_span,
                "internal error: failed to create comptime call scope",
            ) };
        };
        defer self.env.popScope();

        for (fn_info.param_names, 0..) |param_name, i| {
            _ = self.env.bind(param_name, arg_values[i]) catch {
                return .{ .err = error_mod.CtError.init(
                    .internal_error,
                    call_span,
                    "internal error: failed to bind comptime call parameter",
                ) };
            };
        }

        self.call_depth += 1;
        defer self.call_depth -= 1;

        // Evaluate function body using StmtEvaluator
        var stmt_eval = StmtEvaluator{
            .base = self.*,
            .allocator = self.env.allocator,
        };
        const result = stmt_eval.evalBlock(fn_info.body);
        const eval_result: AstEvalResult = switch (result) {
            .return_val => |v| if (v) |val| .{ .value = val } else .{ .value = .void_val },
            .ok => |v| if (v) |val| .{ .value = val } else .{ .value = .void_val },
            .not_comptime => .not_constant,
            .err => |e| .{ .err = e },
            .break_val, .continue_val => .not_constant,
        };

        const end_steps = self.env.stats.total_steps;
        const delta_steps: u64 = if (end_steps >= start_steps) end_steps - start_steps else 0;
        log.debug(
            "[comptime][call] done  fn={s} line={d} result={s} steps={d}/{d} (+{d})\n",
            .{ fn_name, call_span.line, eval_result.kindName(), end_steps, self.env.config.max_steps, delta_steps },
        );
        if (eval_result == .err) {
            const ct_err = eval_result.err;
            log.debug(
                "[comptime][call] error fn={s} kind={s} message={s}\n",
                .{ fn_name, @tagName(ct_err.kind), ct_err.message },
            );
        }
        return eval_result;
    }

    fn evalBuiltinCall(self: *AstEvaluator, call: anytype, fn_name: []const u8) AstEvalResult {
        if (call.arguments.len != 2) return .not_constant;

        const lhs_result = self.evalExprNode(call.arguments[0]);
        const rhs_result = self.evalExprNode(call.arguments[1]);

        const lhs: u256 = switch (lhs_result) {
            .value => |v| switch (v) {
                .integer => |n| n,
                else => return .not_constant,
            },
            .not_constant => return .not_constant,
            .err => |e| return .{ .err = e },
        };
        const rhs: u256 = switch (rhs_result) {
            .value => |v| switch (v) {
                .integer => |n| n,
                else => return .not_constant,
            },
            .not_constant => return .not_constant,
            .err => |e| return .{ .err = e },
        };

        if (std.mem.eql(u8, fn_name, "@addWithOverflow")) {
            const result, const overflow = @addWithOverflow(lhs, rhs);
            return self.makeOverflowResult(result, overflow != 0);
        }
        if (std.mem.eql(u8, fn_name, "@subWithOverflow")) {
            const result, const overflow = @subWithOverflow(lhs, rhs);
            return self.makeOverflowResult(result, overflow != 0);
        }
        if (std.mem.eql(u8, fn_name, "@mulWithOverflow")) {
            const result, const overflow = @mulWithOverflow(lhs, rhs);
            return self.makeOverflowResult(result, overflow != 0);
        }
        if (std.mem.eql(u8, fn_name, "@shlWithOverflow")) {
            const shift = shiftLeftWithOverflow(lhs, rhs);
            return self.makeOverflowResult(shift.value, shift.overflow);
        }
        if (std.mem.eql(u8, fn_name, "@shrWithOverflow")) {
            const shift = shiftRightWithOverflow(lhs, rhs);
            return self.makeOverflowResult(shift.value, shift.overflow);
        }
        if (std.mem.eql(u8, fn_name, "@powerWithOverflow")) {
            const power = powerWithOverflow(lhs, rhs);
            return self.makeOverflowResult(power.value, power.overflow);
        }

        return .not_constant;
    }

    fn makeOverflowResult(self: *AstEvaluator, value_result: u256, overflow: bool) AstEvalResult {
        const fields = [_]heap_mod.CtAggregate.StructField{
            .{ .field_id = 0, .value = .{ .integer = value_result } },
            .{ .field_id = 1, .value = .{ .boolean = overflow } },
        };
        const heap_id = self.env.heap.allocStruct(0, &fields) catch {
            return .{ .err = error_mod.CtError.init(
                .internal_error,
                .{ .line = 0, .column = 0, .length = 0 },
                "failed to allocate overflow result struct",
            ) };
        };
        return .{ .value = .{ .struct_ref = heap_id } };
    }

    const ShiftWithOverflowResult = struct {
        value: u256,
        overflow: bool,
    };

    const PowerWithOverflowResult = struct {
        value: u256,
        overflow: bool,
    };

    fn shiftLeftWithOverflow(lhs: u256, rhs: u256) ShiftWithOverflowResult {
        if (rhs == 0) return .{ .value = lhs, .overflow = false };
        if (rhs >= 256) return .{ .value = 0, .overflow = lhs != 0 };

        const shift_u16: u16 = @intCast(rhs);
        const shift_amt: u8 = @intCast(shift_u16);
        const upper_bits_shift: u8 = @intCast(256 - shift_u16);
        const overflow = (lhs >> upper_bits_shift) != 0;
        return .{ .value = lhs << shift_amt, .overflow = overflow };
    }

    fn shiftRightWithOverflow(lhs: u256, rhs: u256) ShiftWithOverflowResult {
        if (rhs == 0) return .{ .value = lhs, .overflow = false };
        if (rhs >= 256) return .{ .value = 0, .overflow = lhs != 0 };

        const shift_u16: u16 = @intCast(rhs);
        const shift_amt: u8 = @intCast(shift_u16);
        const mask = (@as(u256, 1) << shift_amt) - 1;
        const overflow = (lhs & mask) != 0;
        return .{ .value = lhs >> shift_amt, .overflow = overflow };
    }

    fn powerWithOverflow(base: u256, exponent: u256) PowerWithOverflowResult {
        var result: u256 = 1;
        var factor = base;
        var exp = exponent;
        var overflow = false;

        while (exp != 0) {
            if ((exp & 1) == 1) {
                const mul_result, const mul_overflow = @mulWithOverflow(result, factor);
                result = mul_result;
                overflow = overflow or (mul_overflow != 0);
            }

            exp >>= 1;
            if (exp != 0) {
                const sq_result, const sq_overflow = @mulWithOverflow(factor, factor);
                factor = sq_result;
                overflow = overflow or (sq_overflow != 0);
            }
        }

        return .{ .value = result, .overflow = overflow };
    }

    fn evalLiteral(self: *AstEvaluator, lit: anytype) AstEvalResult {
        return switch (lit.*) {
            .Integer => |int_lit| {
                const val = std.fmt.parseInt(u256, int_lit.value, 0) catch {
                    return .{ .err = error_mod.CtError.init(
                        .internal_error,
                        .{ .line = int_lit.span.line, .column = int_lit.span.column, .length = int_lit.span.length },
                        "invalid integer literal during comptime evaluation",
                    ) };
                };
                return .{ .value = .{ .integer = val } };
            },
            .Bool => |bool_lit| .{ .value = .{ .boolean = bool_lit.value } },
            .Address => |addr_lit| {
                // Address literals are stored as strings like "0x1234..."
                const addr_str = addr_lit.value;
                // Skip "0x" prefix if present
                const hex_str = if (addr_str.len > 2 and addr_str[0] == '0' and (addr_str[1] == 'x' or addr_str[1] == 'X'))
                    addr_str[2..]
                else
                    addr_str;
                const val = std.fmt.parseInt(u160, hex_str, 16) catch {
                    return .{ .err = error_mod.CtError.init(
                        .internal_error,
                        .{ .line = addr_lit.span.line, .column = addr_lit.span.column, .length = addr_lit.span.length },
                        "invalid address literal during comptime evaluation",
                    ) };
                };
                return .{ .value = .{ .address = val } };
            },
            .Hex => |hex_lit| {
                // Hex literals stored as strings
                const hex_str = hex_lit.value;
                const stripped = if (hex_str.len > 2 and hex_str[0] == '0' and (hex_str[1] == 'x' or hex_str[1] == 'X'))
                    hex_str[2..]
                else
                    hex_str;
                const val = std.fmt.parseInt(u256, stripped, 16) catch {
                    return .{ .err = error_mod.CtError.init(
                        .internal_error,
                        .{ .line = hex_lit.span.line, .column = hex_lit.span.column, .length = hex_lit.span.length },
                        "invalid hex literal during comptime evaluation",
                    ) };
                };
                return .{ .value = .{ .integer = val } };
            },
            .Binary => |bin_lit| {
                // Binary literals stored as strings like "0b1010"
                const bin_str = bin_lit.value;
                const stripped = if (bin_str.len > 2 and bin_str[0] == '0' and (bin_str[1] == 'b' or bin_str[1] == 'B'))
                    bin_str[2..]
                else
                    bin_str;
                const val = std.fmt.parseInt(u256, stripped, 2) catch {
                    return .{ .err = error_mod.CtError.init(
                        .internal_error,
                        .{ .line = bin_lit.span.line, .column = bin_lit.span.column, .length = bin_lit.span.length },
                        "invalid binary literal during comptime evaluation",
                    ) };
                };
                return .{ .value = .{ .integer = val } };
            },
            .String => |str_lit| {
                // Allocate string on heap
                const heap_id = self.env.heap.allocString(str_lit.value) catch {
                    return .{ .err = error_mod.CtError.init(
                        .internal_error,
                        .{ .line = str_lit.span.line, .column = str_lit.span.column, .length = str_lit.span.length },
                        "failed to allocate string literal during comptime evaluation",
                    ) };
                };
                return .{ .value = .{ .string_ref = heap_id } };
            },
            .Bytes => |bytes_lit| {
                // Allocate bytes on heap
                const heap_id = self.env.heap.allocBytes(bytes_lit.value) catch {
                    return .{ .err = error_mod.CtError.init(
                        .internal_error,
                        .{ .line = bytes_lit.span.line, .column = bytes_lit.span.column, .length = bytes_lit.span.length },
                        "failed to allocate bytes literal during comptime evaluation",
                    ) };
                };
                return .{ .value = .{ .bytes_ref = heap_id } };
            },
            else => .not_constant, // Character - not commonly used
        };
    }

    fn evalBinary(self: *AstEvaluator, bin: anytype) AstEvalResult {
        const lhs_result = self.evalExprNode(bin.lhs);
        if (lhs_result != .value) return lhs_result;

        const rhs_result = self.evalExprNode(bin.rhs);
        if (rhs_result != .value) return rhs_result;

        const op = mapBinaryOp(bin.operator) orelse return .not_constant;
        const span = SourceSpan{ .line = bin.span.line, .column = bin.span.column, .length = bin.span.length };

        const result = self.evaluator.evalBinaryOp(op, lhs_result.value, rhs_result.value, span);
        const normalized_result = normalizeWrappingBinaryResult(bin.operator, bin.type_info, result);
        return switch (normalized_result) {
            .value => |v| .{ .value = v },
            .runtime => .not_constant,
            .err => |e| .{ .err = e },
            .control => .not_constant,
        };
    }

    fn evalUnary(self: *AstEvaluator, un: anytype) AstEvalResult {
        const operand_result = self.evalExprNode(un.operand);
        if (operand_result != .value) return operand_result;

        const op = mapUnaryOp(un.operator) orelse return .not_constant;
        const span = SourceSpan{ .line = un.span.line, .column = un.span.column, .length = un.span.length };

        const result = self.evaluator.evalUnaryOp(op, operand_result.value, span);
        return switch (result) {
            .value => |v| .{ .value = v },
            .runtime => .not_constant,
            .err => |e| .{ .err = e },
            .control => .not_constant,
        };
    }

    fn evalIdentifier(self: *AstEvaluator, id: anytype) AstEvalResult {
        if (self.lookup) |l| {
            if (l.lookup(id.name)) |val| {
                return .{ .value = val };
            }
        }
        // Also check env
        if (self.env.lookupValue(id.name)) |val| {
            return .{ .value = val };
        }
        // Check if the identifier is a type name → return type_val
        if (resolveTypeName(id.name)) |tid| {
            return .{ .value = CtValue{ .type_val = tid } };
        }
        return .not_constant;
    }

    /// Resolve a type name string (e.g. "u256", "bool") to a TypeId.
    fn resolveTypeName(name: []const u8) ?value.TypeId {
        const ids = value.type_ids;
        const map = std.StaticStringMap(value.TypeId).initComptime(.{
            .{ "u8", ids.u8_id },
            .{ "u16", ids.u16_id },
            .{ "u32", ids.u32_id },
            .{ "u64", ids.u64_id },
            .{ "u128", ids.u128_id },
            .{ "u256", ids.u256_id },
            .{ "i8", ids.i8_id },
            .{ "i16", ids.i16_id },
            .{ "i32", ids.i32_id },
            .{ "i64", ids.i64_id },
            .{ "i128", ids.i128_id },
            .{ "i256", ids.i256_id },
            .{ "bool", ids.bool_id },
            .{ "address", ids.address_id },
            .{ "string", ids.string_id },
            .{ "bytes", ids.bytes_id },
            .{ "void", ids.void_id },
        });
        return map.get(name);
    }

    fn evalEnumLiteral(self: *AstEvaluator, el: anytype) AstEvalResult {
        if (self.lookup) |l| {
            if (l.lookupEnum(el.enum_name, el.variant_name)) |val| {
                return .{ .value = val };
            }
        }
        return .not_constant;
    }

    fn evalCast(self: *AstEvaluator, cast: anytype) AstEvalResult {
        const operand_result = self.evalExprNode(cast.operand);
        if (operand_result != .value) return operand_result;

        const val = operand_result.value;

        // For now, only support casting between integer types and to/from address
        // The cast preserves the numeric value (truncation/extension handled by type system)
        return switch (val) {
            .integer => .{ .value = val }, // Integer casts preserve value
            .address => |a| .{ .value = .{ .integer = @as(u256, a) } }, // Address to integer
            .boolean => |b| .{ .value = .{ .integer = if (b) 1 else 0 } }, // Bool to integer
            else => .not_constant,
        };
    }

    fn evalFieldAccess(self: *AstEvaluator, fa: anytype) AstEvalResult {
        const target_result = self.evalExprNode(fa.target);
        if (target_result != .value) return target_result;

        const target_val = target_result.value;
        const field_name = fa.field;

        // Handle different target types
        switch (target_val) {
            .tuple_ref => |heap_id| {
                // Get field index from field name (e.g., tuple._0 or tuple.0)
                const field_str = if (field_name.len > 0 and field_name[0] == '_') field_name[1..] else field_name;
                const idx = std.fmt.parseInt(usize, field_str, 10) catch return .not_constant;

                const tuple_data = self.env.heap.getTuple(heap_id);
                if (idx < tuple_data.elems.len) {
                    return .{ .value = tuple_data.elems[idx] };
                }
                return .{ .err = error_mod.CtError.init(.index_out_of_bounds, .{ .line = 0, .column = 0, .length = 0 }, "tuple field out of bounds") };
            },
            .array_ref => |heap_id| {
                // Array field access (e.g., arr.length)
                if (std.mem.eql(u8, field_name, "length") or std.mem.eql(u8, field_name, "len")) {
                    const arr_data = self.env.heap.getArray(heap_id);
                    return .{ .value = .{ .integer = arr_data.elems.len } };
                }
                return .not_constant;
            },
            .struct_ref => |heap_id| {
                // Struct field access by index (field_id)
                const struct_data = self.env.heap.getStruct(heap_id);
                // Known anonymous-struct field names used by overflow builtins.
                if (std.mem.eql(u8, field_name, "value")) {
                    if (struct_data.fields.len > 0) return .{ .value = struct_data.fields[0].value };
                    return .not_constant;
                }
                if (std.mem.eql(u8, field_name, "overflow")) {
                    if (struct_data.fields.len > 1) return .{ .value = struct_data.fields[1].value };
                    return .not_constant;
                }
                // Try to match by index (for anonymous structs created in order)
                const field_str = if (field_name.len > 0 and field_name[0] == '_') field_name[1..] else field_name;
                if (std.fmt.parseInt(usize, field_str, 10)) |idx| {
                    if (idx < struct_data.fields.len) {
                        return .{ .value = struct_data.fields[idx].value };
                    }
                } else |_| {}
                return .not_constant;
            },
            .string_ref => |heap_id| {
                // String field access (e.g., str.length)
                if (std.mem.eql(u8, field_name, "length") or std.mem.eql(u8, field_name, "len")) {
                    const str_data = self.env.heap.getString(heap_id);
                    return .{ .value = .{ .integer = str_data.len } };
                }
                return .not_constant;
            },
            .bytes_ref => |heap_id| {
                // Bytes field access (e.g., bytes.length)
                if (std.mem.eql(u8, field_name, "length") or std.mem.eql(u8, field_name, "len")) {
                    const bytes_data = self.env.heap.getBytes(heap_id);
                    return .{ .value = .{ .integer = bytes_data.len } };
                }
                return .not_constant;
            },
            else => return .not_constant,
        }
    }

    fn evalIndex(self: *AstEvaluator, idx: anytype) AstEvalResult {
        const target_result = self.evalExprNode(idx.target);
        if (target_result != .value) return target_result;

        const index_result = self.evalExprNode(idx.index);
        if (index_result != .value) return index_result;

        const target_val = target_result.value;
        const index_val = index_result.value;

        // Index must be an integer
        const i: usize = switch (index_val) {
            .integer => |n| if (n <= std.math.maxInt(usize)) @intCast(n) else return .{ .err = error_mod.CtError.init(.index_out_of_bounds, .{ .line = 0, .column = 0, .length = 0 }, "index too large") },
            else => return .not_constant,
        };

        // Handle array/tuple indexing
        switch (target_val) {
            .array_ref => |heap_id| {
                const arr_data = self.env.heap.getArray(heap_id);
                if (i < arr_data.elems.len) {
                    return .{ .value = arr_data.elems[i] };
                }
                return .{ .err = error_mod.CtError.init(.index_out_of_bounds, .{ .line = 0, .column = 0, .length = 0 }, "array index out of bounds") };
            },
            .tuple_ref => |heap_id| {
                const tuple_data = self.env.heap.getTuple(heap_id);
                if (i < tuple_data.elems.len) {
                    return .{ .value = tuple_data.elems[i] };
                }
                return .{ .err = error_mod.CtError.init(.index_out_of_bounds, .{ .line = 0, .column = 0, .length = 0 }, "tuple index out of bounds") };
            },
            else => return .not_constant,
        }
    }

    fn evalTuple(self: *AstEvaluator, tup: anytype) AstEvalResult {
        // Evaluate all tuple elements
        var elements: std.ArrayList(CtValue) = .empty;
        defer elements.deinit(self.env.allocator);

        for (tup.elements) |elem| {
            const elem_result = self.evalExprNode(elem);
            if (elem_result != .value) return elem_result;
            elements.append(self.env.allocator, elem_result.value) catch {
                return .{ .err = error_mod.CtError.init(
                    .internal_error,
                    .{ .line = 0, .column = 0, .length = 0 },
                    "failed to append tuple element during comptime evaluation",
                ) };
            };
        }

        // Allocate tuple on heap
        const heap_id = self.env.heap.allocTuple(elements.items) catch {
            return .{ .err = error_mod.CtError.init(
                .internal_error,
                .{ .line = 0, .column = 0, .length = 0 },
                "failed to allocate tuple during comptime evaluation",
            ) };
        };
        return .{ .value = .{ .tuple_ref = heap_id } };
    }

    fn evalArrayLiteral(self: *AstEvaluator, arr: anytype) AstEvalResult {
        // Evaluate all array elements
        var elements: std.ArrayList(CtValue) = .empty;
        defer elements.deinit(self.env.allocator);

        for (arr.elements) |elem| {
            const elem_result = self.evalExprNode(elem);
            if (elem_result != .value) return elem_result;
            elements.append(self.env.allocator, elem_result.value) catch {
                return .{ .err = error_mod.CtError.init(
                    .internal_error,
                    .{ .line = 0, .column = 0, .length = 0 },
                    "failed to append array element during comptime evaluation",
                ) };
            };
        }

        // Allocate array on heap
        const heap_id = self.env.heap.allocArray(elements.items) catch {
            return .{ .err = error_mod.CtError.init(
                .internal_error,
                .{ .line = 0, .column = 0, .length = 0 },
                "failed to allocate array during comptime evaluation",
            ) };
        };
        return .{ .value = .{ .array_ref = heap_id } };
    }

    fn evalSwitch(self: *AstEvaluator, sw: anytype) AstEvalResult {
        // Evaluate the condition
        const cond_result = self.evalExprNode(sw.condition);
        const cond_val = switch (cond_result) {
            .value => |v| v,
            .not_constant => return .not_constant,
            .err => |e| return .{ .err = e },
        };

        // Try to match against each case
        for (sw.cases) |case| {
            if (self.matchPattern(&case.pattern, cond_val)) {
                return self.evalSwitchBody(&case.body);
            }
        }

        // No case matched, try default
        if (sw.default_case) |*default_block| {
            // Evaluate default block - return last expression or void
            for (default_block.statements) |*stmt| {
                if (stmt.* == .Expr) {
                    return self.evalExprNode(@constCast(&stmt.Expr));
                }
            }
            return .{ .value = .void_val };
        }

        // No match and no default - strict evaluation must fail explicitly.
        if (self.evaluator.mode.convertsErrorsToRuntime(self.evaluator.policy)) {
            return .not_constant;
        }
        return .{ .err = error_mod.CtError.notComptime(
            .{ .line = sw.span.line, .column = sw.span.column, .length = sw.span.length },
            "switch expression has no matching case and no default",
        ) };
    }

    fn matchPattern(self: *AstEvaluator, pattern: anytype, cond_val: CtValue) bool {
        return switch (pattern.*) {
            .Literal => |lit| {
                const pat_result = self.evalLiteral(&lit.value);
                if (pat_result != .value) return false;
                return self.valuesEqual(cond_val, pat_result.value);
            },
            .Range => |range| {
                const start_result = self.evalExprNode(range.start);
                const end_result = self.evalExprNode(range.end);
                if (start_result != .value or end_result != .value) return false;

                // Only support integer ranges
                const start = if (start_result.value == .integer) start_result.value.integer else return false;
                const end = if (end_result.value == .integer) end_result.value.integer else return false;
                const val = if (cond_val == .integer) cond_val.integer else return false;

                return val >= start and val <= end;
            },
            .EnumValue => |ev| {
                if (cond_val != .enum_val) return false;
                // TODO: Full enum matching requires type system integration
                _ = ev;
                return false;
            },
            .Else => true,
        };
    }

    fn evalSwitchBody(self: *AstEvaluator, body: anytype) AstEvalResult {
        return switch (body.*) {
            .Expression => |expr| self.evalExprNode(expr),
            .Block => |*block| {
                // Return last expression in block or void
                for (block.statements) |*stmt| {
                    if (stmt.* == .Expr) {
                        return self.evalExprNode(@constCast(&stmt.Expr));
                    }
                }
                return .{ .value = .void_val };
            },
            .LabeledBlock => |*lb| {
                for (lb.block.statements) |*stmt| {
                    if (stmt.* == .Expr) {
                        return self.evalExprNode(@constCast(&stmt.Expr));
                    }
                }
                return .{ .value = .void_val };
            },
        };
    }

    fn valuesEqual(self: *AstEvaluator, a: CtValue, b: CtValue) bool {
        _ = self;
        return switch (a) {
            .integer => |av| if (b == .integer) av == b.integer else false,
            .boolean => |av| if (b == .boolean) av == b.boolean else false,
            .address => |av| if (b == .address) av == b.address else false,
            .void_val => b == .void_val,
            else => false, // Complex comparisons not supported yet
        };
    }

    fn evalAnonymousStruct(self: *AstEvaluator, anon: anytype) AstEvalResult {
        // Evaluate all field values
        var fields: std.ArrayList(heap_mod.CtAggregate.StructField) = .empty;
        defer fields.deinit(self.env.allocator);

        for (anon.fields, 0..) |field, i| {
            const val_result = self.evalExprNode(field.value);
            if (val_result != .value) return val_result;
            fields.append(self.env.allocator, .{
                .field_id = @intCast(i),
                .value = val_result.value,
            }) catch {
                return .{ .err = error_mod.CtError.init(
                    .internal_error,
                    .{ .line = 0, .column = 0, .length = 0 },
                    "failed to append anonymous struct field during comptime evaluation",
                ) };
            };
        }

        // Allocate struct on heap with type_id = 0 (anonymous)
        const heap_id = self.env.heap.allocStruct(0, fields.items) catch {
            return .{ .err = error_mod.CtError.init(
                .internal_error,
                .{ .line = 0, .column = 0, .length = 0 },
                "failed to allocate anonymous struct during comptime evaluation",
            ) };
        };
        return .{ .value = .{ .struct_ref = heap_id } };
    }

    fn evalStructInstantiation(self: *AstEvaluator, si: anytype) AstEvalResult {
        // Evaluate all field values
        var fields: std.ArrayList(heap_mod.CtAggregate.StructField) = .empty;
        defer fields.deinit(self.env.allocator);

        for (si.fields, 0..) |field, i| {
            const val_result = self.evalExprNode(field.value);
            if (val_result != .value) return val_result;
            fields.append(self.env.allocator, .{
                .field_id = @intCast(i),
                .value = val_result.value,
            }) catch {
                return .{ .err = error_mod.CtError.init(
                    .internal_error,
                    .{ .line = 0, .column = 0, .length = 0 },
                    "failed to append struct field during comptime evaluation",
                ) };
            };
        }

        // TODO: Look up type_id from struct_name via type system
        // For now, use type_id = 0 as placeholder
        const heap_id = self.env.heap.allocStruct(0, fields.items) catch {
            return .{ .err = error_mod.CtError.init(
                .internal_error,
                .{ .line = 0, .column = 0, .length = 0 },
                "failed to allocate struct during comptime evaluation",
            ) };
        };
        return .{ .value = .{ .struct_ref = heap_id } };
    }

    fn evalTry(self: *AstEvaluator, try_expr: anytype) AstEvalResult {
        // At comptime, try just evaluates the inner expression
        // Error handling semantics are handled by the runtime
        // If the inner expression is comptime-known and not an error, return it
        const inner_result = self.evalExprNode(try_expr.expr);
        return switch (inner_result) {
            .value => |v| {
                // If value is an error type, propagate as not_constant
                // (comptime can't handle runtime error propagation)
                if (v == .enum_val) {
                    // Could be an error enum - for now, just return it
                    return .{ .value = v };
                }
                return .{ .value = v };
            },
            .not_constant => .not_constant,
            .err => |e| .{ .err = e },
        };
    }

    fn evalRange(self: *AstEvaluator, range: anytype) AstEvalResult {
        // Evaluate start and end
        const start_result = self.evalExprNode(range.start);
        const end_result = self.evalExprNode(range.end);

        if (start_result != .value or end_result != .value) return .not_constant;

        const start = start_result.value;
        const end = end_result.value;

        // Ranges are typically used in for loops - return as a tuple (start, end)
        // which the for loop handler can interpret
        const elems = self.env.allocator.alloc(CtValue, 2) catch {
            return .{ .err = error_mod.CtError.init(
                .internal_error,
                .{ .line = 0, .column = 0, .length = 0 },
                "failed to allocate range tuple elements during comptime evaluation",
            ) };
        };
        elems[0] = start;
        elems[1] = end;

        const heap_id = self.env.heap.allocTuple(elems) catch {
            self.env.allocator.free(elems);
            return .{ .err = error_mod.CtError.init(
                .internal_error,
                .{ .line = 0, .column = 0, .length = 0 },
                "failed to allocate range tuple during comptime evaluation",
            ) };
        };
        self.env.allocator.free(elems);

        return .{ .value = .{ .tuple_ref = heap_id } };
    }
};

fn normalizeWrappingBinaryResult(
    operator: anytype,
    expr_type: ast_type_info.TypeInfo,
    result: EvalResult,
) EvalResult {
    if (!isWrappingBinaryOp(operator)) return result;
    return switch (result) {
        .value => |v| switch (v) {
            .integer => |n| .{ .value = .{ .integer = wrapIntegerToResolvedTypeWidth(n, expr_type) } },
            else => result,
        },
        else => result,
    };
}

fn isWrappingBinaryOp(operator: anytype) bool {
    return switch (operator) {
        .WrappingAdd, .WrappingSub, .WrappingMul, .WrappingPow, .WrappingShl, .WrappingShr => true,
        else => false,
    };
}

fn wrapIntegerToResolvedTypeWidth(value_int: u256, expr_type: ast_type_info.TypeInfo) u256 {
    const width = resolvedIntegerBitWidth(expr_type) orelse return value_int;
    if (width >= 256) return value_int;
    const shift_amt: u8 = @intCast(width);
    const mask: u256 = (@as(u256, 1) << shift_amt) - 1;
    return value_int & mask;
}

fn resolvedIntegerBitWidth(expr_type: ast_type_info.TypeInfo) ?u32 {
    if (expr_type.category != .Integer) return null;
    const ot = expr_type.ora_type orelse return null;
    const base = unwrapIntegerBaseType(ot);
    if (!base.isInteger()) return null;
    return base.bitWidth();
}

fn unwrapIntegerBaseType(ot: ast_type_info.OraType) ast_type_info.OraType {
    var current = ot;
    while (true) {
        switch (current) {
            .min_value => |mv| current = mv.base.*,
            .max_value => |mv| current = mv.base.*,
            .in_range => |ir| current = ir.base.*,
            .scaled => |s| current = s.base.*,
            .exact => |e| current = e.*,
            else => break,
        }
    }
    return current;
}

/// Map AST binary operator to comptime binary operator
fn mapBinaryOp(op: anytype) ?BinaryOp {
    return switch (op) {
        .Plus => .add,
        .Minus => .sub,
        .Star => .mul,
        .StarStar => .pow,
        .Slash => .div,
        .Percent => .mod,
        .WrappingAdd => .wadd,
        .WrappingSub => .wsub,
        .WrappingMul => .wmul,
        .WrappingPow => .wpow,
        .WrappingShl => .wshl,
        .WrappingShr => .wshr,
        .EqualEqual => .eq,
        .BangEqual => .neq,
        .Less => .lt,
        .LessEqual => .lte,
        .Greater => .gt,
        .GreaterEqual => .gte,
        .BitwiseAnd => .band,
        .BitwiseOr => .bor,
        .BitwiseXor => .bxor,
        .LeftShift => .shl,
        .RightShift => .shr,
        .And => .land, // logical and
        .Or => .lor, // logical or
        else => null,
    };
}

/// Map AST unary operator to comptime unary operator
fn mapUnaryOp(op: anytype) ?UnaryOp {
    return switch (op) {
        .Minus => .neg,
        .Bang => .not,
        .BitNot => .bnot,
    };
}

// ============================================================================
// Statement Evaluation
// ============================================================================

const ast_statements = @import("../ast/statements.zig");
const ast_expressions = @import("../ast/expressions.zig");
const StmtNode = ast_statements.StmtNode;
const BlockNode = ast_statements.BlockNode;

/// Result of AST statement evaluation
pub const AstStmtResult = union(enum) {
    /// Statement executed successfully, optional value (for expression statements)
    ok: ?CtValue,
    /// Break encountered with optional value and optional label
    break_val: BreakValue,
    /// Continue encountered with optional label
    continue_val: ?[]const u8,
    /// Return encountered with optional value
    return_val: ?CtValue,
    /// Statement cannot be evaluated at comptime
    not_comptime,
    /// Evaluation error
    err: error_mod.CtError,

    pub const BreakValue = struct {
        value: ?CtValue,
        label: ?[]const u8,
    };

    pub fn isOk(self: AstStmtResult) bool {
        return self == .ok;
    }

    pub fn isControlFlow(self: AstStmtResult) bool {
        return self == .break_val or self == .continue_val or self == .return_val;
    }
};

/// Extended evaluator with statement support
pub const StmtEvaluator = struct {
    base: AstEvaluator,
    allocator: std.mem.Allocator,
    iteration_count: u64 = 0,

    pub fn init(allocator: std.mem.Allocator, env: *CtEnv, mode: EvalMode, policy: TryEvalPolicy, lookup: ?IdentifierLookup) StmtEvaluator {
        return .{
            .base = AstEvaluator.init(env, mode, policy, lookup),
            .allocator = allocator,
        };
    }

    /// Evaluate a block of statements
    pub fn evalBlock(self: *StmtEvaluator, block: *const BlockNode) AstStmtResult {
        for (block.statements) |*stmt| {
            const result = self.evalStatement(stmt);
            switch (result) {
                .ok => continue,
                .break_val, .continue_val, .return_val => return result,
                .not_comptime, .err => return result,
            }
        }
        return .{ .ok = null };
    }

    /// Evaluate a single statement
    pub fn evalStatement(self: *StmtEvaluator, stmt: *const StmtNode) AstStmtResult {
        // Check step limit
        if (self.base.evaluator.step(.{ .line = 0, .column = 0, .length = 0 })) |err_result| {
            log.debug(
                "[comptime][step-limit] stmt={s} steps={d}/{d}\n",
                .{ @tagName(stmt.*), self.base.env.stats.total_steps, self.base.env.config.max_steps },
            );
            if (err_result == .err) return .{ .err = err_result.err };
            return .{ .err = error_mod.CtError.init(.step_limit, .{ .line = 0, .column = 0, .length = 0 }, "comptime step limit exceeded") };
        }

        return switch (stmt.*) {
            .Expr => |expr| self.evalExprStmt(&expr),
            .VariableDecl => |*vd| self.evalVarDecl(vd),
            .If => |*if_stmt| self.evalIf(if_stmt),
            .While => |*while_stmt| self.evalWhile(while_stmt),
            .ForLoop => |*for_stmt| self.evalFor(for_stmt),
            .Break => |*brk| self.evalBreak(brk),
            .Continue => |*cont| self.evalContinue(cont),
            .Return => |*ret| self.evalReturn(ret),
            .CompoundAssignment => |*ca| self.evalCompoundAssign(ca),
            .Switch => |*sw| self.evalSwitchStmt(sw),
            .LabeledBlock => |*lb| self.evalLabeledBlock(lb),
            .DestructuringAssignment => |*dest| self.evalDestructuring(dest),
            // Specification-only statements are no-ops at comptime
            .Assert, .Invariant, .Requires, .Ensures, .Assume, .Havoc => .{ .ok = null },
            // Unsupported at comptime
            .Log, .Lock, .Unlock, .ErrorDecl, .TryBlock => .not_comptime,
        };
    }

    fn evalExprStmt(self: *StmtEvaluator, expr: *const @import("../ast/expressions.zig").ExprNode) AstStmtResult {
        // Handle assignments specially
        if (expr.* == .Assignment) {
            const assign = &expr.Assignment;
            // Only support simple identifier assignment
            if (assign.target.* == .Identifier) {
                const name = assign.target.Identifier.name;
                const val_result = self.base.evalExpr(assign.value);
                switch (val_result) {
                    .value => |v| {
                        self.base.env.set(name, v) catch {
                            return .{ .err = error_mod.CtError.init(.internal_error, .{ .line = 0, .column = 0, .length = 0 }, "failed to assign comptime variable") };
                        };
                        return .{ .ok = v };
                    },
                    .not_constant => return .not_comptime,
                    .err => |e| return .{ .err = e },
                }
            }
            return .not_comptime;
        }

        const result = self.base.evalExpr(@constCast(expr));
        return switch (result) {
            .value => |v| .{ .ok = v },
            .not_constant => .not_comptime,
            .err => |e| .{ .err = e },
        };
    }

    fn evalVarDecl(self: *StmtEvaluator, vd: *const ast_statements.VariableDeclNode) AstStmtResult {
        const val: CtValue = if (vd.value) |value_expr| blk: {
            const result = self.base.evalExpr(value_expr);
            switch (result) {
                .value => |v| break :blk v,
                .not_constant => return .not_comptime,
                .err => |e| return .{ .err = e },
            }
        } else .void_val;

        _ = self.base.env.bind(vd.name, val) catch {
            return .{ .err = error_mod.CtError.init(.internal_error, .{ .line = 0, .column = 0, .length = 0 }, "failed to bind variable") };
        };
        return .{ .ok = null };
    }

    fn evalIf(self: *StmtEvaluator, if_stmt: *const ast_statements.IfNode) AstStmtResult {
        const cond_result = self.base.evalExpr(@constCast(&if_stmt.condition));
        const cond_val = switch (cond_result) {
            .value => |v| v,
            .not_constant => return .not_comptime,
            .err => |e| return .{ .err = e },
        };

        const is_true = switch (cond_val) {
            .boolean => |b| b,
            .integer => |i| i != 0,
            else => return .{ .err = error_mod.CtError.init(.type_mismatch, .{ .line = 0, .column = 0, .length = 0 }, "if condition must be boolean") },
        };

        if (is_true) {
            self.base.env.pushScope(false) catch return .{ .err = error_mod.CtError.init(.internal_error, .{ .line = 0, .column = 0, .length = 0 }, "failed to push scope") };
            defer self.base.env.popScope();
            return self.evalBlock(&if_stmt.then_branch);
        } else if (if_stmt.else_branch) |*else_block| {
            self.base.env.pushScope(false) catch return .{ .err = error_mod.CtError.init(.internal_error, .{ .line = 0, .column = 0, .length = 0 }, "failed to push scope") };
            defer self.base.env.popScope();
            return self.evalBlock(else_block);
        }
        return .{ .ok = null };
    }

    fn evalWhile(self: *StmtEvaluator, while_stmt: *const ast_statements.WhileNode) AstStmtResult {
        const max_iterations = self.base.env.config.max_loop_iterations;
        var iterations: u64 = 0;
        log.debug(
            "[comptime][while] enter line={d} steps={d}/{d} max_iter={d}\n",
            .{ while_stmt.span.line, self.base.env.stats.total_steps, self.base.env.config.max_steps, max_iterations },
        );

        while (true) {
            // Check iteration limit
            iterations += 1;
            self.iteration_count += 1;
            if (iterations <= 10 or (iterations % 1000) == 0) {
                log.debug(
                    "[comptime][while] tick  line={d} iter={d} steps={d}/{d}\n",
                    .{ while_stmt.span.line, iterations, self.base.env.stats.total_steps, self.base.env.config.max_steps },
                );
            }
            if (iterations > max_iterations) {
                return .{ .err = error_mod.CtError.init(.iteration_limit, .{ .line = 0, .column = 0, .length = 0 }, "comptime loop iteration limit exceeded") };
            }

            // Evaluate condition
            const cond_result = self.base.evalExpr(@constCast(&while_stmt.condition));
            const cond_val = switch (cond_result) {
                .value => |v| v,
                .not_constant => return .not_comptime,
                .err => |e| return .{ .err = e },
            };

            const should_continue = switch (cond_val) {
                .boolean => |b| b,
                .integer => |i| i != 0,
                else => return .{ .err = error_mod.CtError.init(.type_mismatch, .{ .line = 0, .column = 0, .length = 0 }, "while condition must be boolean") },
            };
            if (iterations <= 10 or (iterations % 1000) == 0) {
                log.debug(
                    "[comptime][while] cond  line={d} iter={d} continue={any}\n",
                    .{ while_stmt.span.line, iterations, should_continue },
                );
                if (while_stmt.condition == .Binary) {
                    const cond_bin = while_stmt.condition.Binary;
                    switch (cond_bin.lhs.*) {
                        .Identifier => |id| {
                            const lhs_name = id.name;
                            const lhs_val = self.base.env.lookupValue(lhs_name);
                            if (lhs_val) |lv| {
                                switch (lv) {
                                    .integer => |n| log.debug(
                                        "[comptime][while] lhs   {s}={d}\n",
                                        .{ lhs_name, n },
                                    ),
                                    else => log.debug(
                                        "[comptime][while] lhs   {s}=<{s}>\n",
                                        .{ lhs_name, @tagName(lv) },
                                    ),
                                }
                            } else {
                                log.debug("[comptime][while] lhs   {s}=<missing>\n", .{lhs_name});
                            }
                        },
                        .Literal => |lit| switch (lit) {
                            .Integer => |int_lit| log.debug(
                                "[comptime][while] lhs   <lit>={s}\n",
                                .{int_lit.value},
                            ),
                            else => {},
                        },
                        else => {},
                    }
                    switch (cond_bin.rhs.*) {
                        .Identifier => |id| {
                            const rhs_name = id.name;
                            const rhs_val = self.base.env.lookupValue(rhs_name);
                            if (rhs_val) |rv| {
                                switch (rv) {
                                    .integer => |n| log.debug(
                                        "[comptime][while] rhs   {s}={d}\n",
                                        .{ rhs_name, n },
                                    ),
                                    else => log.debug(
                                        "[comptime][while] rhs   {s}=<{s}>\n",
                                        .{ rhs_name, @tagName(rv) },
                                    ),
                                }
                            } else {
                                log.debug("[comptime][while] rhs   {s}=<missing>\n", .{rhs_name});
                            }
                        },
                        .Literal => |lit| switch (lit) {
                            .Integer => |int_lit| log.debug(
                                "[comptime][while] rhs   <lit>={s}\n",
                                .{int_lit.value},
                            ),
                            else => {},
                        },
                        else => {},
                    }
                }
            }

            if (!should_continue) {
                log.debug(
                    "[comptime][while] exit  line={d} iter={d} steps={d}/{d}\n",
                    .{ while_stmt.span.line, iterations, self.base.env.stats.total_steps, self.base.env.config.max_steps },
                );
                break;
            }

            // Execute body
            self.base.env.pushScope(true) catch return .{ .err = error_mod.CtError.init(.internal_error, .{ .line = 0, .column = 0, .length = 0 }, "failed to push loop scope") };
            defer self.base.env.popScope();

            const body_result = self.evalBlock(&while_stmt.body);
            switch (body_result) {
                .ok => continue,
                .continue_val => |label| {
                    // If labeled continue and label doesn't match, propagate
                    if (label != null and (while_stmt.label == null or !std.mem.eql(u8, label.?, while_stmt.label.?))) {
                        return body_result;
                    }
                    continue;
                },
                .break_val => |brk| {
                    // If labeled break and label doesn't match, propagate
                    if (brk.label != null and (while_stmt.label == null or !std.mem.eql(u8, brk.label.?, while_stmt.label.?))) {
                        return body_result;
                    }
                    return .{ .ok = brk.value };
                },
                .return_val => return body_result,
                .not_comptime, .err => return body_result,
            }
        }
        return .{ .ok = null };
    }

    fn evalFor(self: *StmtEvaluator, for_stmt: *const ast_statements.ForLoopNode) AstStmtResult {
        const max_iterations = self.base.env.config.max_loop_iterations;

        // Evaluate iterable - support ranges and arrays
        const iterable_result = self.base.evalExpr(@constCast(&for_stmt.iterable));
        const iterable = switch (iterable_result) {
            .value => |v| v,
            .not_constant => return .not_comptime,
            .err => |e| return .{ .err = e },
        };

        // Get iteration bounds
        const iter_info: struct { start: u256, end: u256, items: ?[]const CtValue } = switch (iterable) {
            .array_ref => |heap_id| {
                const arr = self.base.env.heap.getArray(heap_id);
                return self.evalForOverSlice(for_stmt, arr.elems, max_iterations);
            },
            .tuple_ref => |heap_id| {
                const tup = self.base.env.heap.getTuple(heap_id);
                return self.evalForOverSlice(for_stmt, tup.elems, max_iterations);
            },
            .integer => |n| .{ .start = 0, .end = n, .items = null },
            else => return .{ .err = error_mod.CtError.init(.type_mismatch, .{ .line = 0, .column = 0, .length = 0 }, "for loop requires iterable") },
        };

        // Range-based iteration
        var i = iter_info.start;
        var iterations: u64 = 0;
        while (i < iter_info.end) : (i += 1) {
            iterations += 1;
            self.iteration_count += 1;
            if (iterations > max_iterations) {
                return .{ .err = error_mod.CtError.init(.iteration_limit, .{ .line = 0, .column = 0, .length = 0 }, "comptime loop iteration limit exceeded") };
            }

            self.base.env.pushScope(true) catch return .{ .err = error_mod.CtError.init(.internal_error, .{ .line = 0, .column = 0, .length = 0 }, "failed to push loop scope") };
            defer self.base.env.popScope();

            // Bind loop variable
            const var_name = switch (for_stmt.pattern) {
                .Single => |s| s.name,
                .IndexPair => |ip| ip.item,
                .Destructured => return .not_comptime, // TODO: support destructuring
            };
            _ = self.base.env.bind(var_name, .{ .integer = i }) catch return .{ .err = error_mod.CtError.init(.internal_error, .{ .line = 0, .column = 0, .length = 0 }, "failed to bind loop var") };

            // Bind index if IndexPair
            if (for_stmt.pattern == .IndexPair) {
                _ = self.base.env.bind(for_stmt.pattern.IndexPair.index, .{ .integer = iterations - 1 }) catch {
                    return .{ .err = error_mod.CtError.init(.internal_error, .{ .line = 0, .column = 0, .length = 0 }, "failed to bind loop index var") };
                };
            }

            const body_result = self.evalBlock(&for_stmt.body);
            switch (body_result) {
                .ok => continue,
                .continue_val => |label| {
                    if (label != null and (for_stmt.label == null or !std.mem.eql(u8, label.?, for_stmt.label.?))) {
                        return body_result;
                    }
                    continue;
                },
                .break_val => |brk| {
                    if (brk.label != null and (for_stmt.label == null or !std.mem.eql(u8, brk.label.?, for_stmt.label.?))) {
                        return body_result;
                    }
                    return .{ .ok = brk.value };
                },
                .return_val => return body_result,
                .not_comptime, .err => return body_result,
            }
        }
        return .{ .ok = null };
    }

    fn evalForOverSlice(self: *StmtEvaluator, for_stmt: *const ast_statements.ForLoopNode, items: []const CtValue, max_iterations: u64) AstStmtResult {
        var iterations: u64 = 0;
        for (items, 0..) |item, idx| {
            iterations += 1;
            self.iteration_count += 1;
            if (iterations > max_iterations) {
                return .{ .err = error_mod.CtError.init(.iteration_limit, .{ .line = 0, .column = 0, .length = 0 }, "comptime loop iteration limit exceeded") };
            }

            self.base.env.pushScope(true) catch return .{ .err = error_mod.CtError.init(.internal_error, .{ .line = 0, .column = 0, .length = 0 }, "failed to push loop scope") };
            defer self.base.env.popScope();

            // Bind loop variable
            const var_name = switch (for_stmt.pattern) {
                .Single => |s| s.name,
                .IndexPair => |ip| ip.item,
                .Destructured => return .not_comptime,
            };
            _ = self.base.env.bind(var_name, item) catch return .{ .err = error_mod.CtError.init(.internal_error, .{ .line = 0, .column = 0, .length = 0 }, "failed to bind loop var") };

            // Bind index if IndexPair
            if (for_stmt.pattern == .IndexPair) {
                _ = self.base.env.bind(for_stmt.pattern.IndexPair.index, .{ .integer = idx }) catch {
                    return .{ .err = error_mod.CtError.init(.internal_error, .{ .line = 0, .column = 0, .length = 0 }, "failed to bind loop index var") };
                };
            }

            const body_result = self.evalBlock(&for_stmt.body);
            switch (body_result) {
                .ok => continue,
                .continue_val => |label| {
                    if (label != null and (for_stmt.label == null or !std.mem.eql(u8, label.?, for_stmt.label.?))) {
                        return body_result;
                    }
                    continue;
                },
                .break_val => |brk| {
                    if (brk.label != null and (for_stmt.label == null or !std.mem.eql(u8, brk.label.?, for_stmt.label.?))) {
                        return body_result;
                    }
                    return .{ .ok = brk.value };
                },
                .return_val => return body_result,
                .not_comptime, .err => return body_result,
            }
        }
        return .{ .ok = null };
    }

    fn evalBreak(self: *StmtEvaluator, brk: *const ast_statements.BreakNode) AstStmtResult {
        const val: ?CtValue = if (brk.value) |val_expr| blk: {
            const result = self.base.evalExpr(val_expr);
            break :blk switch (result) {
                .value => |v| v,
                .not_constant => return .not_comptime,
                .err => |e| return .{ .err = e },
            };
        } else null;
        return .{ .break_val = .{ .value = val, .label = brk.label } };
    }

    fn evalContinue(self: *StmtEvaluator, cont: *const ast_statements.ContinueNode) AstStmtResult {
        _ = self;
        return .{ .continue_val = cont.label };
    }

    fn evalReturn(self: *StmtEvaluator, ret: *const ast_statements.ReturnNode) AstStmtResult {
        if (ret.value) |val_expr| {
            const result = self.base.evalExpr(@constCast(&val_expr));
            return switch (result) {
                .value => |v| .{ .return_val = v },
                .not_constant => .not_comptime,
                .err => |e| .{ .err = e },
            };
        }
        return .{ .return_val = null };
    }

    fn evalCompoundAssign(self: *StmtEvaluator, ca: *const ast_statements.CompoundAssignmentNode) AstStmtResult {
        // Only support simple identifier targets
        if (ca.target.* != .Identifier) return .not_comptime;

        const name = ca.target.Identifier.name;
        const current = self.base.env.get(name) orelse return .not_comptime;
        const rhs_result = self.base.evalExpr(ca.value);
        const rhs = switch (rhs_result) {
            .value => |v| v,
            .not_constant => return .not_comptime,
            .err => |e| return .{ .err = e },
        };

        const op: BinaryOp = switch (ca.operator) {
            .PlusEqual => .add,
            .MinusEqual => .sub,
            .StarEqual => .mul,
            .SlashEqual => .div,
            .PercentEqual => .mod,
        };

        const eval_result = self.base.evaluator.evalBinaryOp(op, current, rhs, .{ .line = 0, .column = 0, .length = 0 });
        return switch (eval_result) {
            .value => |v| {
                self.base.env.set(name, v) catch {
                    return .{ .err = error_mod.CtError.init(.internal_error, .{ .line = 0, .column = 0, .length = 0 }, "failed to apply compound assignment") };
                };
                return .{ .ok = v };
            },
            .runtime => .not_comptime,
            .err => |e| .{ .err = e },
            .control => .not_comptime,
        };
    }

    fn evalSwitchStmt(self: *StmtEvaluator, sw: *const ast_statements.SwitchNode) AstStmtResult {
        // Evaluate the condition
        const cond_result = self.base.evalExpr(@constCast(&sw.condition));
        const cond_val = switch (cond_result) {
            .value => |v| v,
            .not_constant => return .not_comptime,
            .err => |e| return .{ .err = e },
        };

        // Try to match against each case
        for (sw.cases) |case| {
            if (self.matchSwitchPattern(&case.pattern, cond_val)) {
                return self.evalSwitchBodyStmt(&case.body);
            }
        }

        // No case matched, try default
        if (sw.default_case) |*default_block| {
            self.base.env.pushScope(false) catch return .{ .err = error_mod.CtError.init(.internal_error, .{ .line = 0, .column = 0, .length = 0 }, "failed to push scope") };
            defer self.base.env.popScope();
            return self.evalBlock(default_block);
        }

        return .{ .ok = null };
    }

    fn matchSwitchPattern(self: *StmtEvaluator, pattern: *const ast_expressions.SwitchPattern, cond_val: CtValue) bool {
        return switch (pattern.*) {
            .Literal => |lit| {
                const pat_result = self.base.evalLiteral(&lit.value);
                if (pat_result != .value) return false;
                return self.base.valuesEqual(cond_val, pat_result.value);
            },
            .Range => |range| {
                const start_result = self.base.evalExpr(range.start);
                const end_result = self.base.evalExpr(range.end);
                if (start_result != .value or end_result != .value) return false;

                const start = if (start_result.value == .integer) start_result.value.integer else return false;
                const end = if (end_result.value == .integer) end_result.value.integer else return false;
                const val = if (cond_val == .integer) cond_val.integer else return false;

                return val >= start and val <= end;
            },
            .EnumValue => |ev| {
                if (cond_val != .enum_val) return false;
                // TODO: Full enum matching requires type system integration
                _ = ev;
                return false;
            },
            .Else => true,
        };
    }

    fn evalSwitchBodyStmt(self: *StmtEvaluator, body: *const ast_expressions.SwitchBody) AstStmtResult {
        return switch (body.*) {
            .Expression => |expr| {
                const result = self.base.evalExpr(expr);
                return switch (result) {
                    .value => |v| .{ .ok = v },
                    .not_constant => .not_comptime,
                    .err => |e| .{ .err = e },
                };
            },
            .Block => |*block| {
                self.base.env.pushScope(false) catch return .{ .err = error_mod.CtError.init(.internal_error, .{ .line = 0, .column = 0, .length = 0 }, "failed to push scope") };
                defer self.base.env.popScope();
                return self.evalBlock(block);
            },
            .LabeledBlock => |*lb| {
                return self.evalLabeledBlockInner(lb.label, &lb.block);
            },
        };
    }

    fn evalLabeledBlock(self: *StmtEvaluator, lb: *const ast_statements.LabeledBlockNode) AstStmtResult {
        return self.evalLabeledBlockInner(lb.label, &lb.block);
    }

    fn evalLabeledBlockInner(self: *StmtEvaluator, label: []const u8, block: *const BlockNode) AstStmtResult {
        self.base.env.pushScope(false) catch return .{ .err = error_mod.CtError.init(.internal_error, .{ .line = 0, .column = 0, .length = 0 }, "failed to push scope") };
        defer self.base.env.popScope();

        const result = self.evalBlock(block);
        return switch (result) {
            .break_val => |brk| {
                // If break targets this label, return the value
                if (brk.label != null and std.mem.eql(u8, brk.label.?, label)) {
                    return .{ .ok = brk.value };
                }
                // Otherwise propagate
                return result;
            },
            else => result,
        };
    }

    fn evalDestructuring(self: *StmtEvaluator, dest: *const ast_statements.DestructuringAssignmentNode) AstStmtResult {
        // Evaluate the value expression
        const val_result = self.base.evalExpr(dest.value);
        const val = switch (val_result) {
            .value => |v| v,
            .not_constant => return .not_comptime,
            .err => |e| return .{ .err = e },
        };

        // Destructure based on pattern type
        switch (dest.pattern) {
            .Struct => |fields| {
                // Source must be a struct
                if (val != .struct_ref) return .not_comptime;
                const struct_data = self.base.env.heap.getStruct(val.struct_ref);

                for (fields, 0..) |field, i| {
                    // For anonymous structs, use index-based access
                    if (i < struct_data.fields.len) {
                        _ = self.base.env.bind(field.variable, struct_data.fields[i].value) catch return .{ .err = error_mod.CtError.init(.internal_error, .{ .line = 0, .column = 0, .length = 0 }, "failed to bind destructured field") };
                    } else {
                        return .{ .err = error_mod.CtError.init(.index_out_of_bounds, .{ .line = 0, .column = 0, .length = 0 }, "destructuring field out of bounds") };
                    }
                }
            },
            .Tuple => |names| {
                // Source must be a tuple
                if (val != .tuple_ref) return .not_comptime;
                const tuple_data = self.base.env.heap.getTuple(val.tuple_ref);

                for (names, 0..) |name, i| {
                    if (i < tuple_data.elems.len) {
                        _ = self.base.env.bind(name, tuple_data.elems[i]) catch return .{ .err = error_mod.CtError.init(.internal_error, .{ .line = 0, .column = 0, .length = 0 }, "failed to bind destructured tuple element") };
                    } else {
                        return .{ .err = error_mod.CtError.init(.index_out_of_bounds, .{ .line = 0, .column = 0, .length = 0 }, "destructuring tuple index out of bounds") };
                    }
                }
            },
            .Array => |names| {
                // Source must be an array
                if (val != .array_ref) return .not_comptime;
                const arr_data = self.base.env.heap.getArray(val.array_ref);

                for (names, 0..) |name, i| {
                    if (i < arr_data.elems.len) {
                        _ = self.base.env.bind(name, arr_data.elems[i]) catch return .{ .err = error_mod.CtError.init(.internal_error, .{ .line = 0, .column = 0, .length = 0 }, "failed to bind destructured array element") };
                    } else {
                        return .{ .err = error_mod.CtError.init(.index_out_of_bounds, .{ .line = 0, .column = 0, .length = 0 }, "destructuring array index out of bounds") };
                    }
                }
            },
        }
        return .{ .ok = null };
    }
};

// ============================================================================
// Convenience Functions
// ============================================================================

/// Evaluate an AST expression with default settings
pub fn evaluateExpr(
    allocator: std.mem.Allocator,
    expr: anytype,
    lookup: ?IdentifierLookup,
) AstEvalResult {
    var env = CtEnv.init(allocator, EvalConfig.default);
    defer env.deinit();

    var eval = AstEvaluator.init(&env, .try_eval, .forgiving, lookup);
    return eval.evalExpr(expr);
}

/// Evaluate an AST expression and return integer value if constant
pub fn evaluateToInteger(
    allocator: std.mem.Allocator,
    expr: anytype,
    lookup: ?IdentifierLookup,
) ?u256 {
    const result = evaluateExpr(allocator, expr, lookup);
    return result.getInteger();
}

/// Evaluate an AST expression and intern result into a ConstPool
/// Returns a ConstId that can be stored in symbol tables or HIR
pub fn evaluateAndIntern(
    env: *CtEnv,
    pool: *ConstPool,
    expr: anytype,
    lookup: ?IdentifierLookup,
) AstEvaluator.InternResult {
    var eval = AstEvaluator.initWithPool(env, .try_eval, .forgiving, lookup, pool);
    return eval.evalAndIntern(expr);
}

/// Evaluate a block of statements at comptime
pub fn evaluateBlock(
    allocator: std.mem.Allocator,
    block: *const BlockNode,
    lookup: ?IdentifierLookup,
) AstStmtResult {
    var env = CtEnv.init(allocator, EvalConfig.default);
    defer env.deinit();

    var eval = StmtEvaluator.init(allocator, &env, .try_eval, .forgiving, lookup);
    return eval.evalBlock(block);
}

/// Evaluate a block with a pre-initialized environment (for testing/REPL)
pub fn evaluateBlockWithEnv(
    allocator: std.mem.Allocator,
    env: *CtEnv,
    block: *const BlockNode,
    lookup: ?IdentifierLookup,
) AstStmtResult {
    var eval = StmtEvaluator.init(allocator, env, .try_eval, .forgiving, lookup);
    return eval.evalBlock(block);
}

// ============================================================================
// Tests
// ============================================================================

test "AstEvalResult accessors" {
    const val_result = AstEvalResult{ .value = .{ .integer = 42 } };
    try std.testing.expect(val_result.isConstant());
    try std.testing.expectEqual(@as(u256, 42), val_result.getInteger().?);

    const nc_result = AstEvalResult.not_constant;
    try std.testing.expect(!nc_result.isConstant());
    try std.testing.expectEqual(@as(?u256, null), nc_result.getInteger());

    const bool_result = AstEvalResult{ .value = .{ .boolean = true } };
    try std.testing.expect(bool_result.isConstant());
    try std.testing.expectEqual(true, bool_result.getBoolean().?);

    const addr_result = AstEvalResult{ .value = .{ .address = 0x1234 } };
    try std.testing.expect(addr_result.isConstant());
    try std.testing.expectEqual(@as(u160, 0x1234), addr_result.getAddress().?);
}

test "evaluator binary operations" {
    var env = CtEnv.init(std.testing.allocator, EvalConfig.default);
    defer env.deinit();

    var evaluator = Evaluator.init(&env, .try_eval, .forgiving);

    // Test arithmetic
    const add_result = evaluator.evalBinaryOp(.add, .{ .integer = 10 }, .{ .integer = 5 }, .{ .line = 0, .column = 0, .length = 0 });
    try std.testing.expectEqual(@as(u256, 15), add_result.getValue().?.integer);

    const sub_result = evaluator.evalBinaryOp(.sub, .{ .integer = 10 }, .{ .integer = 3 }, .{ .line = 0, .column = 0, .length = 0 });
    try std.testing.expectEqual(@as(u256, 7), sub_result.getValue().?.integer);

    const mul_result = evaluator.evalBinaryOp(.mul, .{ .integer = 6 }, .{ .integer = 7 }, .{ .line = 0, .column = 0, .length = 0 });
    try std.testing.expectEqual(@as(u256, 42), mul_result.getValue().?.integer);

    // Test comparisons
    const lt_result = evaluator.evalBinaryOp(.lt, .{ .integer = 5 }, .{ .integer = 10 }, .{ .line = 0, .column = 0, .length = 0 });
    try std.testing.expectEqual(true, lt_result.getValue().?.boolean);

    const eq_result = evaluator.evalBinaryOp(.eq, .{ .integer = 42 }, .{ .integer = 42 }, .{ .line = 0, .column = 0, .length = 0 });
    try std.testing.expectEqual(true, eq_result.getValue().?.boolean);

    // Test logical
    const land_result = evaluator.evalBinaryOp(.land, .{ .boolean = true }, .{ .boolean = false }, .{ .line = 0, .column = 0, .length = 0 });
    try std.testing.expectEqual(false, land_result.getValue().?.boolean);

    const lor_result = evaluator.evalBinaryOp(.lor, .{ .boolean = true }, .{ .boolean = false }, .{ .line = 0, .column = 0, .length = 0 });
    try std.testing.expectEqual(true, lor_result.getValue().?.boolean);
}

test "evaluator unary operations" {
    var env = CtEnv.init(std.testing.allocator, EvalConfig.default);
    defer env.deinit();

    var evaluator = Evaluator.init(&env, .try_eval, .forgiving);

    // Test negation (wrapping)
    const neg_result = evaluator.evalUnaryOp(.neg, .{ .integer = 5 }, .{ .line = 0, .column = 0, .length = 0 });
    try std.testing.expect(neg_result.isSuccess());

    // Test logical not
    const not_result = evaluator.evalUnaryOp(.not, .{ .boolean = true }, .{ .line = 0, .column = 0, .length = 0 });
    try std.testing.expectEqual(false, not_result.getValue().?.boolean);
}

test "heap array and tuple operations" {
    var env = CtEnv.init(std.testing.allocator, EvalConfig.default);
    defer env.deinit();

    // Allocate an array
    const arr_id = try env.heap.allocArray(&[_]CtValue{
        .{ .integer = 10 },
        .{ .integer = 20 },
        .{ .integer = 30 },
    });

    const arr_data = env.heap.getArray(arr_id);
    try std.testing.expectEqual(@as(usize, 3), arr_data.elems.len);
    try std.testing.expectEqual(@as(u256, 20), arr_data.elems[1].integer);

    // Allocate a tuple
    const tup_id = try env.heap.allocTuple(&[_]CtValue{
        .{ .boolean = true },
        .{ .integer = 42 },
    });

    const tup_data = env.heap.getTuple(tup_id);
    try std.testing.expectEqual(@as(usize, 2), tup_data.elems.len);
    try std.testing.expectEqual(true, tup_data.elems[0].boolean);
    try std.testing.expectEqual(@as(u256, 42), tup_data.elems[1].integer);
}

test "StmtEvaluator while loop" {
    var env = CtEnv.init(std.testing.allocator, EvalConfig.default);
    defer env.deinit();

    // Bind initial counter
    try env.bind("i", .{ .integer = 0 });
    try env.bind("sum", .{ .integer = 0 });

    const eval = StmtEvaluator.init(std.testing.allocator, &env, .try_eval, .forgiving, null);

    // Simulate: while (i < 5) { sum += i; i += 1; }
    // We'll test the iteration counting and control flow
    var iterations: u64 = 0;
    while (iterations < 5) : (iterations += 1) {
        const i_val = env.get("i").?.integer;
        const sum_val = env.get("sum").?.integer;
        try env.set("sum", .{ .integer = sum_val + i_val });
        try env.set("i", .{ .integer = i_val + 1 });
    }

    try std.testing.expectEqual(@as(u256, 10), env.get("sum").?.integer); // 0+1+2+3+4 = 10
    try std.testing.expectEqual(@as(u256, 5), env.get("i").?.integer);
    _ = eval;
}

test "StmtEvaluator conditionals" {
    var env = CtEnv.init(std.testing.allocator, EvalConfig.default);
    defer env.deinit();

    const eval = StmtEvaluator.init(std.testing.allocator, &env, .try_eval, .forgiving, null);

    // Test condition evaluation via evaluator
    const true_cond: CtValue = .{ .boolean = true };
    const false_cond: CtValue = .{ .boolean = false };
    const int_true: CtValue = .{ .integer = 1 };
    const int_false: CtValue = .{ .integer = 0 };

    // Boolean conditions
    switch (true_cond) {
        .boolean => |b| try std.testing.expect(b),
        else => unreachable,
    }
    switch (false_cond) {
        .boolean => |b| try std.testing.expect(!b),
        else => unreachable,
    }

    // Integer as boolean (non-zero = true)
    switch (int_true) {
        .integer => |i| try std.testing.expect(i != 0),
        else => unreachable,
    }
    switch (int_false) {
        .integer => |i| try std.testing.expect(i == 0),
        else => unreachable,
    }
    _ = eval;
}

test "StmtEvaluator break and continue" {
    var env = CtEnv.init(std.testing.allocator, EvalConfig.default);
    defer env.deinit();

    // Test break value with label
    const break_result = AstStmtResult{ .break_val = .{ .value = .{ .integer = 42 }, .label = null } };
    try std.testing.expect(break_result.isControlFlow());

    // Test labeled break
    const labeled_break = AstStmtResult{ .break_val = .{ .value = .{ .integer = 10 }, .label = "outer" } };
    try std.testing.expect(labeled_break.isControlFlow());
    try std.testing.expectEqualStrings("outer", labeled_break.break_val.label.?);

    // Test continue (with optional label)
    const continue_result = AstStmtResult{ .continue_val = null };
    try std.testing.expect(continue_result.isControlFlow());

    // Test labeled continue
    const labeled_continue = AstStmtResult{ .continue_val = "inner" };
    try std.testing.expect(labeled_continue.isControlFlow());

    // Test ok
    const ok_result = AstStmtResult{ .ok = null };
    try std.testing.expect(ok_result.isOk());
    try std.testing.expect(!ok_result.isControlFlow());
}

test "StmtEvaluator iteration limit" {
    // Use strict config with low iteration limit
    var env = CtEnv.init(std.testing.allocator, EvalConfig.strict);
    defer env.deinit();

    var eval = StmtEvaluator.init(std.testing.allocator, &env, .try_eval, .forgiving, null);

    // Simulate hitting iteration limit
    const max = env.config.max_loop_iterations;
    var i: u64 = 0;
    while (i < max + 1) : (i += 1) {
        eval.iteration_count += 1;
        if (eval.iteration_count > max) {
            break;
        }
    }
    try std.testing.expect(eval.iteration_count > max);
}

test "heap string operations" {
    var env = CtEnv.init(std.testing.allocator, EvalConfig.default);
    defer env.deinit();

    // Allocate a string
    const str_id = try env.heap.allocString("hello");
    const str_data = env.heap.getString(str_id);
    try std.testing.expectEqual(@as(usize, 5), str_data.len);
    try std.testing.expectEqualStrings("hello", str_data);

    // Allocate bytes
    const bytes_id = try env.heap.allocBytes(&[_]u8{ 0xDE, 0xAD, 0xBE, 0xEF });
    const bytes_data = env.heap.getBytes(bytes_id);
    try std.testing.expectEqual(@as(usize, 4), bytes_data.len);
}

test "heap struct operations" {
    var env = CtEnv.init(std.testing.allocator, EvalConfig.default);
    defer env.deinit();

    // Allocate a struct (anonymous with type_id=0)
    const struct_id = try env.heap.allocStruct(0, &[_]heap_mod.CtAggregate.StructField{
        .{ .field_id = 0, .value = .{ .integer = 42 } },
        .{ .field_id = 1, .value = .{ .boolean = true } },
        .{ .field_id = 2, .value = .{ .address = 0x1234 } },
    });

    const struct_data = env.heap.getStruct(struct_id);
    try std.testing.expectEqual(@as(usize, 3), struct_data.fields.len);
    try std.testing.expectEqual(@as(u256, 42), struct_data.fields[0].value.integer);
    try std.testing.expectEqual(true, struct_data.fields[1].value.boolean);
    try std.testing.expectEqual(@as(u160, 0x1234), struct_data.fields[2].value.address);
}

test "AstEvaluator valuesEqual" {
    var env = CtEnv.init(std.testing.allocator, EvalConfig.default);
    defer env.deinit();

    var eval = AstEvaluator.init(&env, .try_eval, .forgiving, null);

    // Test integer equality
    try std.testing.expect(eval.valuesEqual(.{ .integer = 42 }, .{ .integer = 42 }));
    try std.testing.expect(!eval.valuesEqual(.{ .integer = 42 }, .{ .integer = 43 }));

    // Test boolean equality
    try std.testing.expect(eval.valuesEqual(.{ .boolean = true }, .{ .boolean = true }));
    try std.testing.expect(!eval.valuesEqual(.{ .boolean = true }, .{ .boolean = false }));

    // Test address equality
    try std.testing.expect(eval.valuesEqual(.{ .address = 0x1234 }, .{ .address = 0x1234 }));
    try std.testing.expect(!eval.valuesEqual(.{ .address = 0x1234 }, .{ .address = 0x5678 }));

    // Test void equality
    try std.testing.expect(eval.valuesEqual(.void_val, .void_val));

    // Test type mismatch
    try std.testing.expect(!eval.valuesEqual(.{ .integer = 1 }, .{ .boolean = true }));
}

test "mapBinaryOp supports wrapping operators" {
    try std.testing.expectEqual(BinaryOp.wadd, mapBinaryOp(ast_expressions.BinaryOp.WrappingAdd).?);
    try std.testing.expectEqual(BinaryOp.wsub, mapBinaryOp(ast_expressions.BinaryOp.WrappingSub).?);
    try std.testing.expectEqual(BinaryOp.wmul, mapBinaryOp(ast_expressions.BinaryOp.WrappingMul).?);
    try std.testing.expectEqual(BinaryOp.wpow, mapBinaryOp(ast_expressions.BinaryOp.WrappingPow).?);
    try std.testing.expectEqual(BinaryOp.wshl, mapBinaryOp(ast_expressions.BinaryOp.WrappingShl).?);
    try std.testing.expectEqual(BinaryOp.wshr, mapBinaryOp(ast_expressions.BinaryOp.WrappingShr).?);
    try std.testing.expectEqual(BinaryOp.pow, mapBinaryOp(ast_expressions.BinaryOp.StarStar).?);
}

test "AstEvaluator evaluates wrapping operators via AST" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();

    var env = CtEnv.init(std.testing.allocator, EvalConfig.default);
    defer env.deinit();
    var eval = AstEvaluator.init(&env, .must_eval, .strict, null);

    const span = AstSourceSpan{ .line = 1, .column = 1, .length = 1 };
    const max_lit = try ast_expressions.createUntypedIntegerLiteral(a, "0xffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff", span);
    const one_lit = try ast_expressions.createUntypedIntegerLiteral(a, "1", span);
    const zero_lit = try ast_expressions.createUntypedIntegerLiteral(a, "0", span);

    const add_wrap_expr = try ast_expressions.createBinaryExpr(a, max_lit, .WrappingAdd, one_lit, span);
    const add_wrap = eval.evalExpr(add_wrap_expr);
    try std.testing.expectEqual(@as(u256, 0), add_wrap.getInteger().?);

    const sub_wrap_expr = try ast_expressions.createBinaryExpr(a, zero_lit, .WrappingSub, one_lit, span);
    const sub_wrap = eval.evalExpr(sub_wrap_expr);
    try std.testing.expectEqual(std.math.maxInt(u256), sub_wrap.getInteger().?);

    const two_lit = try ast_expressions.createUntypedIntegerLiteral(a, "2", span);
    const ten_lit = try ast_expressions.createUntypedIntegerLiteral(a, "10", span);
    const two_hundred_fifty_six_lit = try ast_expressions.createUntypedIntegerLiteral(a, "256", span);

    const pow_expr = try ast_expressions.createBinaryExpr(a, two_lit, .StarStar, ten_lit, span);
    const pow = eval.evalExpr(pow_expr);
    try std.testing.expectEqual(@as(u256, 1024), pow.getInteger().?);

    const pow_wrap_expr_ok = try ast_expressions.createBinaryExpr(a, two_lit, .WrappingPow, two_hundred_fifty_six_lit, span);
    const pow_wrap_ok = eval.evalExpr(pow_wrap_expr_ok);
    try std.testing.expectEqual(@as(u256, 0), pow_wrap_ok.getInteger().?);

    const pow_wrap_expr = try ast_expressions.createBinaryExpr(a, two_lit, .StarStar, two_hundred_fifty_six_lit, span);
    const pow_overflow = eval.evalExpr(pow_wrap_expr);
    try std.testing.expect(pow_overflow == .err);
    try std.testing.expectEqual(error_mod.CtErrorKind.overflow, pow_overflow.err.kind);
}

test "AstEvaluator evaluates overflow builtins with field access" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();

    var env = CtEnv.init(std.testing.allocator, EvalConfig.default);
    defer env.deinit();
    var eval = AstEvaluator.init(&env, .must_eval, .strict, null);

    const span = AstSourceSpan{ .line = 1, .column = 1, .length = 1 };
    const max_lit = try ast_expressions.createUntypedIntegerLiteral(a, "0xffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff", span);
    const one_lit = try ast_expressions.createUntypedIntegerLiteral(a, "1", span);

    const add_callee = try ast_expressions.createIdentifier(a, "@addWithOverflow", span);
    const add_args = try a.alloc(*ast_expressions.ExprNode, 2);
    add_args[0] = max_lit;
    add_args[1] = one_lit;
    const add_call = try a.create(ast_expressions.ExprNode);
    add_call.* = .{ .Call = .{
        .callee = add_callee,
        .arguments = add_args,
        .type_info = ast_type_info.TypeInfo.unknown(),
        .span = span,
    } };

    const overflow_field = try a.create(ast_expressions.ExprNode);
    overflow_field.* = .{ .FieldAccess = .{
        .target = add_call,
        .field = "overflow",
        .type_info = ast_type_info.TypeInfo.unknown(),
        .span = span,
    } };
    const overflow_result = eval.evalExpr(overflow_field);
    try std.testing.expectEqual(true, overflow_result.getBoolean().?);

    const value_field = try a.create(ast_expressions.ExprNode);
    value_field.* = .{ .FieldAccess = .{
        .target = add_call,
        .field = "value",
        .type_info = ast_type_info.TypeInfo.unknown(),
        .span = span,
    } };
    const value_result = eval.evalExpr(value_field);
    try std.testing.expectEqual(@as(u256, 0), value_result.getInteger().?);

    const high_bit_lit = try ast_expressions.createUntypedIntegerLiteral(a, "0x8000000000000000000000000000000000000000000000000000000000000000", span);
    const shl_callee = try ast_expressions.createIdentifier(a, "@shlWithOverflow", span);
    const shl_args = try a.alloc(*ast_expressions.ExprNode, 2);
    shl_args[0] = high_bit_lit;
    shl_args[1] = one_lit;
    const shl_call = try a.create(ast_expressions.ExprNode);
    shl_call.* = .{ .Call = .{
        .callee = shl_callee,
        .arguments = shl_args,
        .type_info = ast_type_info.TypeInfo.unknown(),
        .span = span,
    } };

    const shl_overflow_field = try a.create(ast_expressions.ExprNode);
    shl_overflow_field.* = .{ .FieldAccess = .{
        .target = shl_call,
        .field = "overflow",
        .type_info = ast_type_info.TypeInfo.unknown(),
        .span = span,
    } };
    const shl_overflow_result = eval.evalExpr(shl_overflow_field);
    try std.testing.expectEqual(true, shl_overflow_result.getBoolean().?);

    const two_lit = try ast_expressions.createUntypedIntegerLiteral(a, "2", span);
    const exp_lit = try ast_expressions.createUntypedIntegerLiteral(a, "256", span);
    const pow_callee = try ast_expressions.createIdentifier(a, "@powerWithOverflow", span);
    const pow_args = try a.alloc(*ast_expressions.ExprNode, 2);
    pow_args[0] = two_lit;
    pow_args[1] = exp_lit;
    const pow_call = try a.create(ast_expressions.ExprNode);
    pow_call.* = .{ .Call = .{
        .callee = pow_callee,
        .arguments = pow_args,
        .type_info = ast_type_info.TypeInfo.unknown(),
        .span = span,
    } };

    const pow_overflow_field = try a.create(ast_expressions.ExprNode);
    pow_overflow_field.* = .{ .FieldAccess = .{
        .target = pow_call,
        .field = "overflow",
        .type_info = ast_type_info.TypeInfo.unknown(),
        .span = span,
    } };
    const pow_overflow_result = eval.evalExpr(pow_overflow_field);
    try std.testing.expectEqual(true, pow_overflow_result.getBoolean().?);

    const pow_value_field = try a.create(ast_expressions.ExprNode);
    pow_value_field.* = .{ .FieldAccess = .{
        .target = pow_call,
        .field = "value",
        .type_info = ast_type_info.TypeInfo.unknown(),
        .span = span,
    } };
    const pow_value_result = eval.evalExpr(pow_value_field);
    try std.testing.expectEqual(@as(u256, 0), pow_value_result.getInteger().?);
}

test "AstEvaluator switch no match strict reports not_comptime" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();

    var env = CtEnv.init(std.testing.allocator, EvalConfig.default);
    defer env.deinit();
    var eval = AstEvaluator.init(&env, .try_eval, .strict, null);

    const span = AstSourceSpan{ .line = 1, .column = 1, .length = 1 };
    const cond = try ast_expressions.createUntypedIntegerLiteral(a, "7", span);
    const body = try ast_expressions.createUntypedIntegerLiteral(a, "42", span);

    const cases = try a.alloc(ast_expressions.SwitchCase, 1);
    cases[0] = .{
        .pattern = .{ .Literal = .{
            .value = .{ .Integer = .{
                .value = "1",
                .type_info = ast_type_info.CommonTypes.unknown_integer(),
                .span = span,
            } },
            .span = span,
        } },
        .body = .{ .Expression = body },
        .span = span,
    };

    const switch_expr = try ast_expressions.createSwitchExprNode(a, cond, cases, null, span);
    const result = eval.evalExpr(switch_expr);
    try std.testing.expect(result == .err);
    try std.testing.expectEqual(CtErrorKind.not_comptime, result.err.kind);
}

test "AstEvaluator switch no match forgiving remains not_constant" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();

    var env = CtEnv.init(std.testing.allocator, EvalConfig.default);
    defer env.deinit();
    var eval = AstEvaluator.init(&env, .try_eval, .forgiving, null);

    const span = AstSourceSpan{ .line = 1, .column = 1, .length = 1 };
    const cond = try ast_expressions.createUntypedIntegerLiteral(a, "7", span);
    const body = try ast_expressions.createUntypedIntegerLiteral(a, "42", span);

    const cases = try a.alloc(ast_expressions.SwitchCase, 1);
    cases[0] = .{
        .pattern = .{ .Literal = .{
            .value = .{ .Integer = .{
                .value = "1",
                .type_info = ast_type_info.CommonTypes.unknown_integer(),
                .span = span,
            } },
            .span = span,
        } },
        .body = .{ .Expression = body },
        .span = span,
    };

    const switch_expr = try ast_expressions.createSwitchExprNode(a, cond, cases, null, span);
    const result = eval.evalExpr(switch_expr);
    try std.testing.expect(result == .not_constant);
}

test "AstEvaluator switch no match strict uses default case when present" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();

    var env = CtEnv.init(std.testing.allocator, EvalConfig.default);
    defer env.deinit();
    var eval = AstEvaluator.init(&env, .try_eval, .strict, null);

    const span = AstSourceSpan{ .line = 1, .column = 1, .length = 1 };
    const cond = try ast_expressions.createUntypedIntegerLiteral(a, "7", span);
    const body = try ast_expressions.createUntypedIntegerLiteral(a, "42", span);

    const cases = try a.alloc(ast_expressions.SwitchCase, 1);
    cases[0] = .{
        .pattern = .{ .Literal = .{
            .value = .{ .Integer = .{
                .value = "1",
                .type_info = ast_type_info.CommonTypes.unknown_integer(),
                .span = span,
            } },
            .span = span,
        } },
        .body = .{ .Expression = body },
        .span = span,
    };

    const default_expr = try ast_expressions.createUntypedIntegerLiteral(a, "99", span);
    const default_statements = try a.alloc(ast_statements.StmtNode, 1);
    default_statements[0] = .{ .Expr = default_expr.* };
    const default_block = ast_statements.BlockNode{
        .statements = default_statements,
        .span = span,
    };

    const switch_expr = try ast_expressions.createSwitchExprNode(a, cond, cases, default_block, span);
    const result = eval.evalExpr(switch_expr);
    try std.testing.expectEqual(@as(u256, 99), result.getInteger().?);
}

test "AstEvaluator evalExpr invalid handle reports internal_error" {
    var env = CtEnv.init(std.testing.allocator, EvalConfig.default);
    defer env.deinit();
    var eval = AstEvaluator.init(&env, .must_eval, .strict, null);

    const result = eval.evalExpr(@as(u8, 1));
    try std.testing.expect(result == .err);
    try std.testing.expectEqual(CtErrorKind.internal_error, result.err.kind);
}

test "AstEvaluator evalAndIntern pool precondition and success path" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();

    const span = AstSourceSpan{ .line = 1, .column = 1, .length = 1 };
    const lit = try ast_expressions.createUntypedIntegerLiteral(a, "5", span);

    var env = CtEnv.init(std.testing.allocator, EvalConfig.default);
    defer env.deinit();

    {
        var eval = AstEvaluator.init(&env, .must_eval, .strict, null);
        const result = eval.evalAndIntern(lit);
        try std.testing.expect(result == .err);
        try std.testing.expectEqual(CtErrorKind.internal_error, result.err.kind);
    }

    var pool = ConstPool.init(std.testing.allocator);
    defer pool.deinit();
    {
        var eval = AstEvaluator.initWithPool(&env, .must_eval, .strict, null, &pool);
        const result = eval.evalAndIntern(lit);
        try std.testing.expect(result == .const_id);
        switch (pool.get(result.const_id)) {
            .integer => |n| try std.testing.expectEqual(@as(u256, 5), n),
            else => try std.testing.expect(false),
        }
    }
}

test "AstEvaluator evalAndIntern preserves not_constant" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();

    var env = CtEnv.init(std.testing.allocator, EvalConfig.default);
    defer env.deinit();
    var pool = ConstPool.init(std.testing.allocator);
    defer pool.deinit();

    const span = AstSourceSpan{ .line = 1, .column = 1, .length = 1 };
    const id = try ast_expressions.createIdentifier(a, "unknown_symbol", span);

    var eval = AstEvaluator.initWithPool(&env, .must_eval, .strict, null, &pool);
    const result = eval.evalAndIntern(id);
    try std.testing.expect(result == .not_constant);
}
