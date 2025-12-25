// ============================================================================
// Expression Type Resolution
// ============================================================================
// Bidirectional typing: synthExpr (infer) and checkExpr (verify)
// ============================================================================

const std = @import("std");
const ast = @import("../../../ast.zig");
const TypeInfo = @import("../../type_info.zig").TypeInfo;
const TypeCategory = @import("../../type_info.zig").TypeCategory;
const CommonTypes = @import("../../type_info.zig").CommonTypes;
const OraType = @import("../../type_info.zig").OraType;
const SourceSpan = @import("../../source_span.zig").SourceSpan;
const BinaryOp = @import("../../expressions.zig").BinaryOp;
const state = @import("../../../semantics/state.zig");
const semantics = @import("../../../semantics.zig");
const builtins = semantics.builtins;
const SymbolTable = state.SymbolTable;
const Scope = state.Scope;
const TypeResolutionError = @import("../mod.zig").TypeResolutionError;
const Typed = @import("mod.zig").Typed;
const Effect = @import("mod.zig").Effect;
const LockDelta = @import("mod.zig").LockDelta;
const TypeContext = @import("mod.zig").TypeContext;
const validation = @import("../validation/mod.zig");
const refinements = @import("../refinements/mod.zig");
const utils = @import("../utils/mod.zig");
const FunctionNode = ast.FunctionNode;

const CoreResolver = @import("mod.zig").CoreResolver;

/// Synthesize (infer) the type of an expression.
/// This is the "inference" direction of bidirectional typing.
/// Returns the inferred type along with effects, locks, and obligations.
pub fn synthExpr(
    self: *CoreResolver,
    expr: *ast.Expressions.ExprNode,
) TypeResolutionError!Typed {
    return switch (expr.*) {
        .Literal => |*lit| synthLiteral(self, lit),
        .Identifier => |*id| synthIdentifier(self, id),
        .Binary => |*bin| synthBinary(self, bin),
        .Unary => |*unary| synthUnary(self, unary),
        .Call => |*call| synthCall(self, call),
        .FieldAccess => |*fa| synthFieldAccess(self, fa),
        .Index => |*idx| synthIndex(self, idx),
        .EnumLiteral => |*el| synthEnumLiteral(self, el),
        .Old => |*old_expr| synthOld(self, old_expr),
        .ErrorReturn => |*err_ret| synthErrorReturn(self, err_ret),
        .Try => |*try_expr| synthTry(self, try_expr),
        .Cast => |*cast_expr| synthCast(self, cast_expr),
        .Range => |*range_expr| synthRange(self, range_expr),
        .Assignment => |*assign| synthAssignment(self, assign),
        else => {
            // unhandled expression types - return unknown
            // todo: Implement remaining expression types (Range, etc.)
            return Typed.init(
                TypeInfo.unknown(),
                Effect.pure(),
                self.allocator,
            );
        },
    };
}

/// Check an expression against an expected type.
/// This is the "verification" direction of bidirectional typing.
/// Returns the typed expression if compatible, otherwise returns an error.
pub fn checkExpr(
    self: *CoreResolver,
    expr: *ast.Expressions.ExprNode,
    expected: TypeInfo,
) TypeResolutionError!Typed {
    std.debug.print("[checkExpr] Checking expression type={any}, expected category={s}\n", .{ @tagName(expr.*), @tagName(expected.category) });

    // special case: if expression is ErrorReturn and expected is ErrorUnion, allow it
    if (expr.* == .ErrorReturn and expected.category == .ErrorUnion) {
        const err_ret = &expr.ErrorReturn;
        std.debug.print("[checkExpr] Special handling: ErrorReturn '{s}' with ErrorUnion expected\n", .{err_ret.error_name});
        const typed = try synthErrorReturn(self, err_ret);
        // error category is compatible with ErrorUnion
        return typed;
    }

    // synthesize type first, then validate compatibility.
    // future optimization: use expected type during synthesis for better error messages.
    var typed = try synthExpr(self, expr);
    std.debug.print("[checkExpr] Synthesized type: category={s}\n", .{@tagName(typed.ty.category)});

    // special case: if synthesized type is ErrorUnion (or Union with error_union) and expected is the success type, unwrap it
    // this handles cases like `var x: bool = transfer(...)` inside a try block
    if ((typed.ty.category == .ErrorUnion or typed.ty.category == .Union) and expected.category != .ErrorUnion and expected.category != .Unknown) {
        if (typed.ty.ora_type) |ora_ty| {
            if (ora_ty == .error_union) {
                const success_ora_type = ora_ty.error_union.*;
                const success_category = success_ora_type.getCategory();
                // check if expected type matches the success type
                if (expected.ora_type) |expected_ora_type| {
                    if (OraType.equals(success_ora_type, expected_ora_type) and success_category == expected.category) {
                        // unwrap ErrorUnion to success type
                        typed.ty = expected;
                        return typed;
                    }
                } else if (success_category == expected.category) {
                    // categories match, create success type
                    const success_type = TypeInfo{
                        .category = success_category,
                        .ora_type = success_ora_type,
                        .source = .inferred,
                        .span = expected.span,
                    };
                    typed.ty = success_type;
                    return typed;
                }
            } else if (ora_ty == ._union) {
                // error union with explicit errors: !T | Error1 | Error2
                // check if first element is error_union
                if (ora_ty._union.len > 0) {
                    const first_type = ora_ty._union[0];
                    if (first_type == .error_union) {
                        const success_ora_type = first_type.error_union.*;
                        const success_category = success_ora_type.getCategory();
                        // check if expected type matches the success type
                        if (expected.ora_type) |expected_ora_type| {
                            if (OraType.equals(success_ora_type, expected_ora_type) and success_category == expected.category) {
                                // unwrap ErrorUnion to success type
                                typed.ty = expected;
                                return typed;
                            }
                        } else if (success_category == expected.category) {
                            // categories match, create success type
                            const success_type = TypeInfo{
                                .category = success_category,
                                .ora_type = success_ora_type,
                                .source = .inferred,
                                .span = expected.span,
                            };
                            typed.ty = success_type;
                            return typed;
                        }
                    }
                }
            }
        }
    }

    // note: Success type compatibility with ErrorUnion is now handled in areTypesCompatible
    // no need to wrap here - the compatibility check will handle it

    // special case: if synthesized type is Enum (from error.X field access) and expected is ErrorUnion, allow it
    if (typed.ty.category == .Enum and expected.category == .ErrorUnion) {
        std.debug.print("[checkExpr] Found Enum with ErrorUnion expected, expr type: {any}\n", .{@tagName(expr.*)});
        // check if this is actually an error (error.X field access)
        if (expr.* == .FieldAccess) {
            const fa = &expr.FieldAccess;
            std.debug.print("[checkExpr] FieldAccess: target type={any}, field={s}\n", .{ @tagName(fa.target.*), fa.field });
            if (fa.target.* == .Identifier) {
                std.debug.print("[checkExpr] Target is Identifier: {s}\n", .{fa.target.Identifier.name});
                if (std.mem.eql(u8, fa.target.Identifier.name, "error")) {
                    std.debug.print("[checkExpr] Special handling: FieldAccess error.{s} (Enum) with ErrorUnion expected - treating as Error\n", .{fa.field});
                    // treat Enum as Error for compatibility with ErrorUnion
                    var error_typed = typed;
                    error_typed.ty.category = .Error;
                    return error_typed;
                }
            }
        }
    }

    const has_expected_hint = expected.isResolved() or expected.category != .Unknown;

    // if we couldn't infer a type but the caller supplied any usable hint,
    // trust the expected type and avoid spurious mismatches on untyped literals.
    if (!typed.ty.isResolved() and has_expected_hint) {
        var result = typed;
        result.ty = expected;
        return result;
    }

    // validate compatibility
    if (!self.validation.isAssignable(typed.ty, expected)) {
        const got_str = formatTypeInfo(typed.ty, self.allocator) catch "unknown";
        const expected_str = formatTypeInfo(expected, self.allocator) catch "unknown";
        std.debug.print(
            "[type_resolver] TypeMismatch: got {s}, expected {s}\n",
            .{ got_str, expected_str },
        );
        return TypeResolutionError.TypeMismatch;
    }

    return typed;
}

// ============================================================================
// Synthesis Helpers
// ============================================================================

fn synthLiteral(
    self: *CoreResolver,
    lit: *ast.Expressions.LiteralExpr,
) TypeResolutionError!Typed {
    // literal types are set during parsing/initialization
    // extract type_info from the union variant
    const lit_type_info = switch (lit.*) {
        .Integer => |int_lit| int_lit.type_info,
        .String => |str_lit| str_lit.type_info,
        .Bool => |bool_lit| bool_lit.type_info,
        .Address => |addr_lit| addr_lit.type_info,
        .Hex => |hex_lit| hex_lit.type_info,
        .Binary => |bin_lit| bin_lit.type_info,
        .Character => |char_lit| char_lit.type_info,
        .Bytes => |bytes_lit| bytes_lit.type_info,
    };

    // if not resolved, infer from literal value
    if (!lit_type_info.isResolved()) {
        // preserve whatever partial information the parser provided (e.g. unknown_integer)
        // so downstream compatibility checks still treat this as numeric.
        return Typed.init(lit_type_info, Effect.pure(), self.allocator);
    }
    return Typed.init(lit_type_info, Effect.pure(), self.allocator);
}

fn synthOld(
    self: *CoreResolver,
    old_expr: *ast.Expressions.OldExpr,
) TypeResolutionError!Typed {
    // old() expressions have the same type as their inner expression
    // they're used in postconditions to refer to pre-state values
    return synthExpr(self, old_expr.expr);
}

fn synthEnumLiteral(
    self: *CoreResolver,
    el: *ast.Expressions.EnumLiteralExpr,
) TypeResolutionError!Typed {
    const enum_ty = TypeInfo.fromOraType(OraType{ .enum_type = el.enum_name });
    return Typed.init(enum_ty, Effect.pure(), self.allocator);
}

fn synthAssignment(
    self: *CoreResolver,
    assign: *ast.Expressions.AssignmentExpr,
) TypeResolutionError!Typed {
    // get target type from L-value (Identifier, FieldAccess, or Index)
    const target_type = blk: {
        switch (assign.target.*) {
            .Identifier => |*id| {
                // ensure identifier type is resolved
                try @import("identifier.zig").resolveIdentifierType(self, id);
                break :blk id.type_info;
            },
            .FieldAccess => |*fa| {
                // ensure field access type is resolved
                _ = try synthFieldAccess(self, fa);
                break :blk fa.type_info;
            },
            .Index => |*idx| {
                // ensure index type is resolved and get element type
                var index_typed = try synthIndex(self, idx);
                defer index_typed.deinit(self.allocator);
                // index expressions return the element type
                break :blk index_typed.ty;
            },
            else => {
                // unsupported L-value type
                return TypeResolutionError.TypeMismatch;
            },
        }
    };

    // ensure target type is resolved
    if (!target_type.isResolved()) {
        return TypeResolutionError.UnresolvedType;
    }

    // check that value is assignable to target
    const value_typed = try synthExpr(self, assign.value);

    if (!self.validation.isAssignable(target_type, value_typed.ty)) {
        const got_str = formatTypeInfo(value_typed.ty, self.allocator) catch "unknown";
        const expected_str = formatTypeInfo(target_type, self.allocator) catch "unknown";
        std.debug.print(
            "[type_resolver] Assignment type mismatch: got {s}, expected {s}\n",
            .{ got_str, expected_str },
        );
        return TypeResolutionError.TypeMismatch;
    }

    // validate refinement types if present and set skip_guard
    if (target_type.ora_type) |target_ora_type| {
        // validate refinement type structure
        try self.refinement_system.validate(target_ora_type);

        // check guard optimizations: skip guard if optimization applies
        // import shouldSkipGuard from statement.zig
        const statement_mod = @import("statement.zig");
        assign.skip_guard = statement_mod.shouldSkipGuard(self, assign.value, target_ora_type);
    }

    // assignment expressions return the target type (the assigned value's type)
    return Typed.init(target_type, Effect.pure(), self.allocator);
}

fn synthCast(
    self: *CoreResolver,
    cast: *ast.Expressions.CastExpr,
) TypeResolutionError!Typed {
    // ensure target type is resolved
    if (!cast.target_type.isResolved()) {
        return TypeResolutionError.UnresolvedType;
    }

    // infer operand type
    const operand_typed = try synthExpr(self, cast.operand);

    // conservative rule: cast is allowed only if operand is assignable to target
    if (!self.validation.isAssignable(cast.target_type, operand_typed.ty)) {
        const got_str = formatTypeInfo(operand_typed.ty, self.allocator) catch "unknown";
        const expected_str = formatTypeInfo(cast.target_type, self.allocator) catch "unknown";
        std.debug.print(
            "[type_resolver] Cast type mismatch: got {s}, expected {s}\n",
            .{ got_str, expected_str },
        );
        return TypeResolutionError.TypeMismatch;
    }

    // cast expression yields the target type
    return Typed.init(cast.target_type, Effect.pure(), self.allocator);
}

fn synthRange(
    self: *CoreResolver,
    range: *ast.Expressions.RangeExpr,
) TypeResolutionError!Typed {
    const compat = @import("../validation/compatibility.zig");

    // infer start/end types
    const start_typed = try synthExpr(self, range.start);
    const end_typed = try synthExpr(self, range.end);

    // require integer categories
    if (start_typed.ty.category != .Integer or end_typed.ty.category != .Integer) {
        return TypeResolutionError.TypeMismatch;
    }

    // choose a common integer type (prefer a resolved OraType; otherwise keep category)
    var chosen = start_typed.ty;
    if (start_typed.ty.ora_type != null and end_typed.ty.ora_type != null) {
        const s = start_typed.ty.ora_type.?;
        const e = end_typed.ty.ora_type.?;
        if (compat.isBaseTypeCompatible(s, e)) {
            chosen = TypeInfo.fromOraType(e);
        } else if (compat.isBaseTypeCompatible(e, s)) {
            chosen = TypeInfo.fromOraType(s);
        } else {
            return TypeResolutionError.TypeMismatch;
        }
    } else if (!chosen.isResolved()) {
        // fallback: unresolved integer literal -> keep Integer category
        chosen = TypeInfo{
            .category = .Integer,
            .ora_type = null,
            .source = .inferred,
            .span = range.span,
        };
    }

    // store type on AST node for downstream passes
    range.type_info = chosen;

    return Typed.init(chosen, Effect.pure(), self.allocator);
}

fn synthErrorReturn(
    self: *CoreResolver,
    err_ret: *ast.Expressions.ErrorReturnExpr,
) TypeResolutionError!Typed {
    // look up the error in the symbol table to verify it exists
    const symbol = SymbolTable.findUp(self.current_scope, err_ret.error_name);
    if (symbol == null) {
        return TypeResolutionError.UndefinedIdentifier;
    }

    const found_symbol = symbol.?;
    if (found_symbol.kind != .Error) {
        return TypeResolutionError.TypeMismatch;
    }

    // error returns are compatible with ErrorUnion types
    // return a type with Error category so it can be checked against ErrorUnion
    // the actual ErrorUnion type will be determined by the function's return type
    const error_ty = TypeInfo{
        .category = .Error,
        .ora_type = null, // ErrorUnion type will be determined by context
        .source = .inferred,
        .span = err_ret.span,
    };

    return Typed.init(error_ty, Effect.pure(), self.allocator);
}

fn synthTry(
    self: *CoreResolver,
    try_expr: *ast.Expressions.TryExpr,
) TypeResolutionError!Typed {
    // synthesize the inner expression (should return an ErrorUnion)
    var inner_typed = try synthExpr(self, try_expr.expr);
    defer inner_typed.deinit(self.allocator);

    // extract success type from ErrorUnion
    if (inner_typed.ty.category == .ErrorUnion) {
        if (inner_typed.ty.ora_type) |ora_ty| {
            if (ora_ty == .error_union) {
                // extract success type from error union
                const success_ora_type = ora_ty.error_union.*;
                const success_category = success_ora_type.getCategory();
                const success_type = TypeInfo{
                    .category = success_category,
                    .ora_type = success_ora_type,
                    .source = .inferred,
                    .span = try_expr.span,
                };
                return Typed.init(success_type, Effect.pure(), self.allocator);
            } else if (ora_ty == ._union) {
                // error union with explicit errors: !T | Error1 | Error2
                // first type should be error_union, extract its success type
                if (ora_ty._union.len > 0) {
                    const first_type = ora_ty._union[0];
                    if (first_type == .error_union) {
                        const success_ora_type = first_type.error_union.*;
                        const success_category = success_ora_type.getCategory();
                        const success_type = TypeInfo{
                            .category = success_category,
                            .ora_type = success_ora_type,
                            .source = .inferred,
                            .span = try_expr.span,
                        };
                        return Typed.init(success_type, Effect.pure(), self.allocator);
                    }
                }
            }
        }
    }

    // if inner expression is not an ErrorUnion, return unknown (should be caught by validation)
    return Typed.init(TypeInfo.unknown(), Effect.pure(), self.allocator);
}

fn synthIdentifier(
    self: *CoreResolver,
    id: *ast.Expressions.IdentifierExpr,
) TypeResolutionError!Typed {
    // use identifier resolution module
    try @import("identifier.zig").resolveIdentifierType(self, id);
    return Typed.init(id.type_info, Effect.pure(), self.allocator);
}

fn synthBinary(
    self: *CoreResolver,
    bin: *ast.Expressions.BinaryExpr,
) TypeResolutionError!Typed {
    // synthesize operand types
    var lhs_typed = try synthExpr(self, bin.lhs);
    defer lhs_typed.deinit(self.allocator);
    var rhs_typed = try synthExpr(self, bin.rhs);
    defer rhs_typed.deinit(self.allocator);

    // validate operator types
    try validateBinaryOperator(
        self,
        bin.operator,
        lhs_typed.ty,
        rhs_typed.ty,
    );

    // infer result type
    const result_type: TypeInfo = switch (bin.operator) {
        // comparison operators return bool
        .EqualEqual, .BangEqual, .Less, .LessEqual, .Greater, .GreaterEqual => CommonTypes.bool_type(),
        // logical operators return bool
        .And, .Or => CommonTypes.bool_type(),
        // arithmetic and bitwise - try refinement inference
        else => blk: {
            if (self.refinement_system.inferArithmetic(
                bin.operator,
                lhs_typed.ty,
                rhs_typed.ty,
            )) |inferred| {
                break :blk inferred;
            } else {
                // fallback to lhs type, but ensure it's resolved
                var fallback_type = lhs_typed.ty;
                if (fallback_type.ora_type) |ot| {
                    // ensure category matches ora_type
                    fallback_type.category = ot.getCategory();
                }
                break :blk fallback_type;
            }
        },
    };

    // ensure result type is resolved (fix category if needed)
    var final_result_type = result_type;
    if (result_type.ora_type) |ot| {
        final_result_type.category = ot.getCategory();
    }
    bin.type_info = final_result_type;

    // combine effects (both operands evaluated)
    const combined_eff = combineEffects(lhs_typed.eff, rhs_typed.eff, self.allocator);

    const empty_delta = LockDelta.emptyWithAllocator(self.allocator);
    return Typed{
        .ty = result_type,
        .eff = combined_eff,
        .lock_delta = empty_delta,
        .obligations = &.{}, // Phase 3: No obligations yet
    };
}

fn synthUnary(
    self: *CoreResolver,
    unary: *ast.Expressions.UnaryExpr,
) TypeResolutionError!Typed {
    var operand_typed = try synthExpr(self, unary.operand);
    defer operand_typed.deinit(self.allocator);

    // unary operators preserve operand type (except ! which returns bool)
    const result_type = switch (unary.operator) {
        .Bang => CommonTypes.bool_type(),
        else => operand_typed.ty,
    };

    unary.type_info = result_type;
    const empty_delta = LockDelta.emptyWithAllocator(self.allocator);
    return Typed{
        .ty = result_type,
        .eff = operand_typed.eff,
        .lock_delta = empty_delta,
        .obligations = &.{},
    };
}

fn synthCall(
    self: *CoreResolver,
    call: *ast.Expressions.CallExpr,
) TypeResolutionError!Typed {
    // resolve callee expression (to ensure it's typed)
    _ = try synthExpr(self, call.callee);

    // resolve all argument types
    for (call.arguments) |arg| {
        _ = try synthExpr(self, arg);
    }

    // if callee is an identifier, look up the function
    if (call.callee.* == .Identifier) {
        const func_name = call.callee.Identifier.name;
        std.debug.print("[synthCall] Looking up function '{s}'\n", .{func_name});

        // first try symbol table
        const symbol = SymbolTable.findUp(self.current_scope, func_name);

        // if not in symbol table, check function registry
        if (symbol == null) {
            if (self.function_registry) |registry| {
                const registry_map = @as(*std.StringHashMap(*FunctionNode), @ptrCast(@alignCast(registry)));
                if (registry_map.get(func_name)) |function| {
                    // found in function registry - use return type
                    if (function.return_type_info) |ret_info| {
                        call.type_info = ret_info;
                        return Typed.init(ret_info, Effect.pure(), self.allocator);
                    } else {
                        // no return type - return unknown
                        call.type_info = TypeInfo.unknown();
                        return Typed.init(TypeInfo.unknown(), Effect.pure(), self.allocator);
                    }
                }
            }
            return TypeResolutionError.UndefinedIdentifier;
        }

        // look up function in symbol table
        const found_symbol = symbol.?;

        if (found_symbol.kind == .Function) {
            // function's type_info is a function type, extract return type
            if (found_symbol.typ) |typ| {
                if (typ.isResolved()) {
                    // extract return type from function type
                    var return_type: ?TypeInfo = null;
                    if (typ.ora_type) |ora_ty| {
                        if (ora_ty == .function) {
                            if (ora_ty.function.return_type) |ret_ora_type_ptr| {
                                const ret_ora_type = ret_ora_type_ptr.*;
                                var ret_category = ret_ora_type.getCategory();

                                std.debug.print("[synthCall] Function '{s}' return type: category={s}, ora_type={any}\n", .{ func_name, @tagName(ret_category), ret_ora_type });

                                // special handling: if it's a _union with error_union as first element,
                                // treat it as ErrorUnion category
                                if (ret_category == .Union and ret_ora_type == ._union) {
                                    if (ret_ora_type._union.len > 0) {
                                        const first_type = ret_ora_type._union[0];
                                        std.debug.print("[synthCall] First union element: {any}\n", .{first_type});
                                        if (first_type == .error_union) {
                                            ret_category = .ErrorUnion;
                                            std.debug.print("[synthCall] Detected error union, changing category to ErrorUnion\n", .{});
                                        }
                                    }
                                }

                                return_type = TypeInfo{
                                    .category = ret_category,
                                    .ora_type = ret_ora_type,
                                    .source = .inferred,
                                    .span = call.span,
                                };
                            } else {
                                // no return type - void function
                                return_type = TypeInfo{
                                    .category = .Void,
                                    .ora_type = OraType.void,
                                    .source = .inferred,
                                    .span = call.span,
                                };
                            }
                        } else {
                            std.debug.print("[synthCall] Function '{s}' ora_type is not .function: {any}\n", .{ func_name, ora_ty });
                        }
                    } else {
                        std.debug.print("[synthCall] Function '{s}' typ has no ora_type\n", .{func_name});
                    }

                    // if we couldn't extract return type, try function registry
                    if (return_type == null) {
                        if (self.function_registry) |registry| {
                            const registry_map = @as(*std.StringHashMap(*FunctionNode), @ptrCast(@alignCast(registry)));
                            if (registry_map.get(func_name)) |function| {
                                if (function.return_type_info) |ret_info| {
                                    return_type = ret_info;
                                }
                            }
                        }
                    }

                    if (return_type) |ret_ty| {
                        call.type_info = ret_ty;
                        return Typed.init(ret_ty, Effect.pure(), self.allocator);
                    } else {
                        // fallback: use the function type itself (shouldn't happen)
                        call.type_info = typ;
                        return Typed.init(typ, Effect.pure(), self.allocator);
                    }
                } else {
                    return TypeResolutionError.UnresolvedType;
                }
            } else {
                return TypeResolutionError.UnresolvedType;
            }
        } else if (found_symbol.kind == .Error) {
            // error call (e.g., InsufficientBalance(amount, balance))
            // return Error type compatible with ErrorUnion
            const error_ty = TypeInfo{
                .category = .Error,
                .ora_type = null,
                .source = .inferred,
                .span = call.span,
            };
            call.type_info = error_ty;
            return Typed.init(error_ty, Effect.pure(), self.allocator);
        } else {
            return TypeResolutionError.TypeMismatch; // Not a function or error
        }
    }

    // callee is not an identifier - can't resolve
    return TypeResolutionError.TypeMismatch;
}

fn synthFieldAccess(
    self: *CoreResolver,
    fa: *ast.Expressions.FieldAccessExpr,
) TypeResolutionError!Typed {
    // first check if this is a builtin (e.g., std.constants.ZERO_ADDRESS)
    if (self.builtin_registry) |registry| {
        // check for single-level access (std.something)
        if (fa.target.* == .Identifier) {
            const base = fa.target.Identifier.name;
            if (std.mem.eql(u8, base, "std")) {
                // build path like "std.constants"
                const path = std.fmt.allocPrint(self.allocator, "{s}.{s}", .{ base, fa.field }) catch return TypeResolutionError.OutOfMemory;
                defer self.allocator.free(path);

                if (registry.lookup(path)) |builtin_info| {
                    if (!builtin_info.is_call) {
                        // it's a constant - return its type
                        fa.type_info = TypeInfo.fromOraType(builtin_info.return_type);
                        return Typed.init(fa.type_info, Effect.pure(), self.allocator);
                    }
                    // it's a function namespace - continue to check for deeper access
                }
            }
        } else if (fa.target.* == .FieldAccess) {
            // multi-level access like std.constants.ZERO_ADDRESS
            // build the full path using builtins helper
            const base_path = builtins.getMemberAccessPath(self.allocator, fa.target) catch return TypeResolutionError.OutOfMemory;
            defer self.allocator.free(base_path);

            const full_path = std.fmt.allocPrint(self.allocator, "{s}.{s}", .{ base_path, fa.field }) catch return TypeResolutionError.OutOfMemory;
            defer self.allocator.free(full_path);

            if (registry.lookup(full_path)) |builtin_info| {
                if (!builtin_info.is_call) {
                    // it's a constant
                    fa.type_info = TypeInfo.fromOraType(builtin_info.return_type);
                    return Typed.init(fa.type_info, Effect.pure(), self.allocator);
                }
                // it's a function - return its return type (functions can be used as values)
                // this allows std.transaction.sender to be used without parentheses
                fa.type_info = TypeInfo.fromOraType(builtin_info.return_type);
                return Typed.init(fa.type_info, Effect.pure(), self.allocator);
            }
        }
    }

    // resolve base expression
    var base_typed = try synthExpr(self, fa.target);
    defer base_typed.deinit(self.allocator);

    // get the type of the base
    const target_type = base_typed.ty;

    // check if the type is a struct
    if (target_type.category == .Struct) {
        // extract struct name from ora_type
        if (target_type.ora_type) |ora_ty| {
            if (ora_ty == .struct_type) {
                const struct_name = ora_ty.struct_type;
                // look up struct fields from symbol table
                if (self.symbol_table.struct_fields.get(struct_name)) |fields| {
                    // find the field by name
                    for (fields) |field| {
                        if (std.mem.eql(u8, field.name, fa.field)) {
                            // found the field! Use its type
                            fa.type_info = field.type_info;
                            return Typed.init(field.type_info, Effect.pure(), self.allocator);
                        }
                    }
                    // field not found in struct
                    return TypeResolutionError.TypeMismatch;
                }
            }
        }
    }

    // could not resolve - mark as unresolved
    fa.type_info = TypeInfo.unknown();
    return TypeResolutionError.UnresolvedType;
}

fn synthIndex(
    self: *CoreResolver,
    idx: *ast.Expressions.IndexExpr,
) TypeResolutionError!Typed {
    // resolve target and index expressions
    var target_typed = try synthExpr(self, idx.target);
    defer target_typed.deinit(self.allocator);
    var index_typed = try synthExpr(self, idx.index);
    defer index_typed.deinit(self.allocator);

    // get the type of the target
    const target_type = target_typed.ty;

    // for array types, extract the element type
    if (target_type.category == .Array) {
        // extract element type from ora_type
        if (target_type.ora_type) |ora_ty| {
            if (ora_ty == .array) {
                const elem_ora_type = ora_ty.array.elem.*;
                const elem_type_info = TypeInfo.inferred(elem_ora_type.getCategory(), elem_ora_type, null);
                return Typed.init(elem_type_info, Effect.pure(), self.allocator);
            }
        }
    }
    // for map types, extract the value type
    else if (target_type.category == .Map) {
        // extract value type from ora_type
        if (target_type.ora_type) |ora_ty| {
            if (ora_ty == .map) {
                const value_ora_type = ora_ty.map.value.*;
                const value_type_info = TypeInfo.inferred(value_ora_type.getCategory(), value_ora_type, null);
                return Typed.init(value_type_info, Effect.pure(), self.allocator);
            }
        }
    }

    // fallback: unknown type
    return Typed.init(TypeInfo.unknown(), Effect.pure(), self.allocator);
}

// ============================================================================
// Validation Helpers
// ============================================================================

/// Validate that binary operator operands have compatible types.
fn validateBinaryOperator(
    self: *CoreResolver,
    op: BinaryOp,
    lhs_type: TypeInfo,
    rhs_type: TypeInfo,
) TypeResolutionError!void {

    // errorUnion types should not be used directly in binary operations
    // if an operand is an ErrorUnion or Union with error_union, extract its success type for the operation
    var lhs_check = lhs_type;
    var rhs_check = rhs_type;

    // check if lhs is ErrorUnion or Union with error_union
    if (lhs_type.category == .ErrorUnion or lhs_type.category == .Union) {
        if (lhs_type.ora_type) |lhs_ora| {
            if (lhs_ora == .error_union) {
                const success_ora_type = lhs_ora.error_union.*;
                const success_category = success_ora_type.getCategory();
                lhs_check = TypeInfo{
                    .category = success_category,
                    .ora_type = success_ora_type,
                    .source = .inferred,
                    .span = lhs_type.span,
                };
            } else if (lhs_ora == ._union and lhs_ora._union.len > 0) {
                const first_type = lhs_ora._union[0];
                if (first_type == .error_union) {
                    const success_ora_type = first_type.error_union.*;
                    const success_category = success_ora_type.getCategory();
                    lhs_check = TypeInfo{
                        .category = success_category,
                        .ora_type = success_ora_type,
                        .source = .inferred,
                        .span = lhs_type.span,
                    };
                }
            }
        }
    }

    // check if rhs is ErrorUnion or Union with error_union
    if (rhs_type.category == .ErrorUnion or rhs_type.category == .Union) {
        if (rhs_type.ora_type) |rhs_ora| {
            if (rhs_ora == .error_union) {
                const success_ora_type = rhs_ora.error_union.*;
                const success_category = success_ora_type.getCategory();
                rhs_check = TypeInfo{
                    .category = success_category,
                    .ora_type = success_ora_type,
                    .source = .inferred,
                    .span = rhs_type.span,
                };
            } else if (rhs_ora == ._union and rhs_ora._union.len > 0) {
                const first_type = rhs_ora._union[0];
                if (first_type == .error_union) {
                    const success_ora_type = first_type.error_union.*;
                    const success_category = success_ora_type.getCategory();
                    rhs_check = TypeInfo{
                        .category = success_category,
                        .ora_type = success_ora_type,
                        .source = .inferred,
                        .span = rhs_type.span,
                    };
                }
            }
        }
    }

    // allow Unknown types in binary operations - they should be resolved by the time we get here,
    // but if not, we'll allow them to pass (they'll be caught later in validation)
    if (lhs_check.category == .Unknown or rhs_check.category == .Unknown) {
        return;
    }

    // error types should not be used directly in binary operations (except comparisons)
    // they should be part of an ErrorUnion or used in error handling contexts
    // however, allow Error types in comparison operations (==, !=) for error handling
    const is_comparison = op == .EqualEqual or op == .BangEqual;
    if (lhs_check.category == .Error or rhs_check.category == .Error) {
        if (!is_comparison) {
            // error types in non-comparison operations - this is invalid
            return TypeResolutionError.IncompatibleTypes;
        }
        // for comparison operations, allow Error types (they'll be handled by areCompatible)
        // this allows Error == Error comparisons and Error == Enum comparisons (for error.X)
    }

    // comma operator doesn't require type compatibility - it evaluates both operands and returns the right-hand side
    if (op == .Comma) {
        return; // Comma operator allows any types
    }

    // check types are compatible for the operator
    if (!self.validation.areCompatible(lhs_check, rhs_check)) {
        return TypeResolutionError.IncompatibleTypes;
    }
}

// Function argument validation moved to mod.zig to avoid circular dependency

// ============================================================================
// Effect Helpers
// ============================================================================

fn combineEffects(
    eff1: Effect,
    eff2: Effect,
    allocator: std.mem.Allocator,
) Effect {
    const SlotSet = @import("mod.zig").SlotSet;
    return switch (eff1) {
        .Pure => eff2,
        .Writes => |s1| switch (eff2) {
            .Pure => eff1,
            .Writes => |s2| {
                // combine slot sets - create new combined set
                // note: s1 and s2 will be cleaned up by their Typed.deinit calls
                var combined = SlotSet.init(allocator);
                for (s1.slots.items) |slot| {
                    combined.add(allocator, slot) catch {};
                }
                for (s2.slots.items) |slot| {
                    if (!combined.contains(slot)) {
                        combined.add(allocator, slot) catch {};
                    }
                }
                return Effect.writes(combined);
            },
        },
    };
}

/// Format TypeInfo for error messages, including refinement details
fn formatTypeInfo(type_info: TypeInfo, allocator: std.mem.Allocator) ![]const u8 {
    if (type_info.ora_type) |ora_type| {
        return formatOraType(ora_type, allocator);
    }
    // fallback to category if no ora_type
    return try std.fmt.allocPrint(allocator, "{s}", .{@tagName(type_info.category)});
}

/// Format OraType with refinement details
fn formatOraType(ora_type: OraType, allocator: std.mem.Allocator) ![]const u8 {
    return switch (ora_type) {
        .min_value => |mv| {
            const base_str = try formatOraType(mv.base.*, allocator);
            defer allocator.free(base_str);
            return try std.fmt.allocPrint(allocator, "MinValue<{s}, {d}>", .{ base_str, mv.min });
        },
        .max_value => |mv| {
            const base_str = try formatOraType(mv.base.*, allocator);
            defer allocator.free(base_str);
            return try std.fmt.allocPrint(allocator, "MaxValue<{s}, {d}>", .{ base_str, mv.max });
        },
        .in_range => |ir| {
            const base_str = try formatOraType(ir.base.*, allocator);
            defer allocator.free(base_str);
            return try std.fmt.allocPrint(allocator, "InRange<{s}, {d}, {d}>", .{ base_str, ir.min, ir.max });
        },
        .scaled => |s| {
            const base_str = try formatOraType(s.base.*, allocator);
            defer allocator.free(base_str);
            return try std.fmt.allocPrint(allocator, "Scaled<{s}, {d}>", .{ base_str, s.decimals });
        },
        .exact => |e| {
            const base_str = try formatOraType(e.*, allocator);
            defer allocator.free(base_str);
            return try std.fmt.allocPrint(allocator, "Exact<{s}>", .{base_str});
        },
        .non_zero_address => try std.fmt.allocPrint(allocator, "NonZeroAddress", .{}),
        .u8 => try std.fmt.allocPrint(allocator, "u8", .{}),
        .u16 => try std.fmt.allocPrint(allocator, "u16", .{}),
        .u32 => try std.fmt.allocPrint(allocator, "u32", .{}),
        .u64 => try std.fmt.allocPrint(allocator, "u64", .{}),
        .u128 => try std.fmt.allocPrint(allocator, "u128", .{}),
        .u256 => try std.fmt.allocPrint(allocator, "u256", .{}),
        .i8 => try std.fmt.allocPrint(allocator, "i8", .{}),
        .i16 => try std.fmt.allocPrint(allocator, "i16", .{}),
        .i32 => try std.fmt.allocPrint(allocator, "i32", .{}),
        .i64 => try std.fmt.allocPrint(allocator, "i64", .{}),
        .i128 => try std.fmt.allocPrint(allocator, "i128", .{}),
        .i256 => try std.fmt.allocPrint(allocator, "i256", .{}),
        .bool => try std.fmt.allocPrint(allocator, "bool", .{}),
        .string => try std.fmt.allocPrint(allocator, "string", .{}),
        .address => try std.fmt.allocPrint(allocator, "address", .{}),
        .bytes => try std.fmt.allocPrint(allocator, "bytes", .{}),
        .void => try std.fmt.allocPrint(allocator, "void", .{}),
        .struct_type => |name| try std.fmt.allocPrint(allocator, "struct {s}", .{name}),
        .enum_type => |name| try std.fmt.allocPrint(allocator, "enum {s}", .{name}),
        .contract_type => |name| try std.fmt.allocPrint(allocator, "contract {s}", .{name}),
        else => try std.fmt.allocPrint(allocator, "{s}", .{@tagName(ora_type)}),
    };
}
