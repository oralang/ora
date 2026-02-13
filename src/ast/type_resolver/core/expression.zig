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
const SlotSet = @import("mod.zig").SlotSet;
const combineEffects = @import("mod.zig").combineEffects;
const mergeEffects = @import("mod.zig").mergeEffects;
const takeEffect = @import("mod.zig").takeEffect;
const MemoryRegion = @import("../../region.zig").MemoryRegion;
const TypeContext = @import("mod.zig").TypeContext;
const validation = @import("../validation/mod.zig");
const refinements = @import("../refinements/mod.zig");
const utils = @import("../utils/mod.zig");
const extract = utils.extract;
const FunctionNode = ast.FunctionNode;
const log = @import("log");

const CoreResolver = @import("mod.zig").CoreResolver;

fn isRefinementOraType(ora_type: OraType) bool {
    return switch (ora_type) {
        .min_value, .max_value, .in_range, .scaled, .exact, .non_zero_address => true,
        else => false,
    };
}

fn isErrorUnionTypeInfo(type_info: TypeInfo) bool {
    if (type_info.category == .ErrorUnion) return true;
    if (type_info.ora_type) |ora_ty| {
        return switch (ora_ty) {
            .error_union => true,
            ._union => |members| members.len > 0 and members[0] == .error_union,
            else => false,
        };
    }
    return false;
}

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
        .ArrayLiteral => |*arr| synthArrayLiteral(self, arr),
        .Tuple => |*tuple_expr| synthTuple(self, tuple_expr),
        .Old => |*old_expr| synthOld(self, old_expr),
        .ErrorReturn => |*err_ret| synthErrorReturn(self, err_ret),
        .Try => |*try_expr| synthTry(self, try_expr),
        .Cast => |*cast_expr| synthCast(self, cast_expr),
        .Range => |*range_expr| synthRange(self, range_expr),
        .Assignment => |*assign| synthAssignment(self, assign),
        .StructInstantiation => |*si| synthStructInstantiation(self, si),
        .Comptime => |*comptime_expr| synthComptime(self, comptime_expr),
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

fn synthComptime(
    self: *CoreResolver,
    comptime_expr: *ast.Expressions.ComptimeExpr,
) TypeResolutionError!Typed {
    // Comptime blocks should contain a return or expression statement.
    for (comptime_expr.block.statements) |*stmt| {
        switch (stmt.*) {
            .Return => |ret| {
                if (ret.value) |*expr| {
                    const typed = try synthExpr(self, @constCast(expr));
                    comptime_expr.type_info = typed.ty;
                    return typed;
                }
            },
            .Expr => |expr_node| {
                const typed = try synthExpr(self, @constCast(&expr_node));
                comptime_expr.type_info = typed.ty;
                return typed;
            },
            else => {
                // For now, only support pure expression/return in comptime blocks.
                return TypeResolutionError.TypeMismatch;
            },
        }
    }

    return Typed.init(
        TypeInfo.unknown(),
        Effect.pure(),
        self.allocator,
    );
}

fn synthStructInstantiation(
    self: *CoreResolver,
    si: *ast.Expressions.StructInstantiationExpr,
) TypeResolutionError!Typed {
    const struct_name = switch (si.struct_name.*) {
        .Identifier => |id| id.name,
        else => return TypeResolutionError.TypeMismatch,
    };

    const fields = self.symbol_table.struct_fields.get(struct_name) orelse {
        return TypeResolutionError.TypeMismatch;
    };

    var combined_eff = Effect.pure();
    for (si.fields) |*field_init| {
        var found: ?ast.StructField = null;
        for (fields) |field| {
            if (std.mem.eql(u8, field.name, field_init.name)) {
                found = field;
                break;
            }
        }
        const field_info = found orelse return TypeResolutionError.TypeMismatch;

        var checked = try checkExpr(self, field_init.value, field_info.type_info);
        defer checked.deinit(self.allocator);
        const eff = takeEffect(&checked);
        mergeEffects(self.allocator, &combined_eff, eff);
    }

    const struct_ty = TypeInfo.fromOraType(OraType{ .struct_type = struct_name });
    return Typed.init(struct_ty, combined_eff, self.allocator);
}

/// Check an expression against an expected type.
/// This is the "verification" direction of bidirectional typing.
/// Returns the typed expression if compatible, otherwise returns an error.
pub fn checkExpr(
    self: *CoreResolver,
    expr: *ast.Expressions.ExprNode,
    expected: TypeInfo,
) TypeResolutionError!Typed {
    log.debug("[checkExpr] Checking expression type={any}, expected category={s}\n", .{ @tagName(expr.*), @tagName(expected.category) });

    if (expr.* == .Comptime and expected.category != .Unknown) {
        return checkComptime(self, &expr.Comptime, expected);
    }

    // special case: if expression is ErrorReturn and expected is ErrorUnion, allow it
    if (expr.* == .ErrorReturn and expected.category == .ErrorUnion) {
        const err_ret = &expr.ErrorReturn;
        log.debug("[checkExpr] Special handling: ErrorReturn '{s}' with ErrorUnion expected\n", .{err_ret.error_name});
        const typed = try synthErrorReturn(self, err_ret);
        // error category is compatible with ErrorUnion
        return typed;
    }

    // synthesize type first, then validate compatibility.
    // future optimization: use expected type during synthesis for better error messages.
    if (expr.* == .ArrayLiteral and (expected.category == .Array or expected.category == .Slice)) {
        return checkArrayLiteral(self, &expr.ArrayLiteral, expected);
    }
    var typed = try synthExpr(self, expr);
    errdefer typed.deinit(self.allocator);
    log.debug("[checkExpr] Synthesized type: category={s}\n", .{@tagName(typed.ty.category)});

    // special case: allow hex literals to satisfy bytes expectations
    if (expected.category == .Bytes and expr.* == .Literal) {
        if (expr.Literal == .Hex) {
            expr.Literal.Hex.type_info = expected;
            var result = typed;
            result.ty = expected;
            return result;
        }
    }

    // special case: unwrap ErrorUnion to success type inside try blocks only
    // this handles cases like `var x: bool = transfer(...)` inside a try block
    if (self.in_try_block and (typed.ty.category == .ErrorUnion or typed.ty.category == .Union) and expected.category != .ErrorUnion and expected.category != .Unknown) {
        if (typed.ty.ora_type) |ora_ty| {
            if (ora_ty == .error_union) {
                const success_ora_type = ora_ty.error_union.*;
                const success_category = success_ora_type.getCategory();
                // check if expected type matches the success type
                if (expected.ora_type) |expected_ora_type| {
                    if (OraType.equals(success_ora_type, expected_ora_type) and success_category == expected.category) {
                        // Record the error union type for catch block before unwrapping
                        self.last_try_error_union = typed.ty;
                        // unwrap ErrorUnion to success type
                        typed.ty = expected;
                        return typed;
                    }
                } else if (success_category == expected.category) {
                    // Record the error union type for catch block before unwrapping
                    self.last_try_error_union = typed.ty;
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
                                // Record the error union type for catch block before unwrapping
                                self.last_try_error_union = typed.ty;
                                // unwrap ErrorUnion to success type
                                typed.ty = expected;
                                return typed;
                            }
                        } else if (success_category == expected.category) {
                            // Record the error union type for catch block before unwrapping
                            self.last_try_error_union = typed.ty;
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
        log.debug("[checkExpr] Found Enum with ErrorUnion expected, expr type: {any}\n", .{@tagName(expr.*)});
        // check if this is actually an error (error.X field access)
        if (expr.* == .FieldAccess) {
            const fa = &expr.FieldAccess;
            log.debug("[checkExpr] FieldAccess: target type={any}, field={s}\n", .{ @tagName(fa.target.*), fa.field });
            if (fa.target.* == .Identifier) {
                log.debug("[checkExpr] Target is Identifier: {s}\n", .{fa.target.Identifier.name});
                if (std.mem.eql(u8, fa.target.Identifier.name, "error")) {
                    log.debug("[checkExpr] Special handling: FieldAccess error.{s} (Enum) with ErrorUnion expected - treating as Error\n", .{fa.field});
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
        log.err(
            "[type_resolver] TypeMismatch: got {s}, expected {s}\n",
            .{ got_str, expected_str },
        );
        log.err(
            "[type_resolver] TypeMismatch details: got category={s} ora_type={any} expected category={s} ora_type={any}\n",
            .{ @tagName(typed.ty.category), typed.ty.ora_type, @tagName(expected.category), expected.ora_type },
        );
        return TypeResolutionError.TypeMismatch;
    }

    return typed;
}

fn checkComptime(
    self: *CoreResolver,
    comptime_expr: *ast.Expressions.ComptimeExpr,
    expected: TypeInfo,
) TypeResolutionError!Typed {
    for (comptime_expr.block.statements) |*stmt| {
        switch (stmt.*) {
            .Return => |ret| {
                if (ret.value) |*expr| {
                    const typed = try checkExpr(self, @constCast(expr), expected);
                    comptime_expr.type_info = typed.ty;
                    return typed;
                }
            },
            .Expr => |expr_node| {
                const typed = try checkExpr(self, @constCast(&expr_node), expected);
                comptime_expr.type_info = typed.ty;
                return typed;
            },
            else => {
                return TypeResolutionError.TypeMismatch;
            },
        }
    }

    return TypeResolutionError.TypeMismatch;
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
    // if the enum type exists, treat as proper enum literal
    const root_scope: ?*const Scope = @as(?*const Scope, @ptrCast(self.symbol_table.root));
    const type_symbol = SymbolTable.findUp(root_scope, el.enum_name);
    if (type_symbol) |tsym| {
        if (tsym.kind == .Enum) {
            const enum_ty = TypeInfo.fromOraType(OraType{ .enum_type = el.enum_name });
            return Typed.init(enum_ty, Effect.pure(), self.allocator);
        }
    }

    // fallback: reinterpret as field access (e.g., history.length)
    var ident_expr = ast.Expressions.ExprNode{ .Identifier = ast.Expressions.IdentifierExpr{
        .name = el.enum_name,
        .type_info = TypeInfo.unknown(),
        .span = el.span,
    } };
    var field_expr = ast.Expressions.FieldAccessExpr{
        .target = &ident_expr,
        .field = el.variant_name,
        .type_info = TypeInfo.unknown(),
        .span = el.span,
    };
    return synthFieldAccess(self, &field_expr);
}

fn synthTuple(
    self: *CoreResolver,
    tuple_expr: *ast.Expressions.TupleExpr,
) TypeResolutionError!Typed {
    var types = std.ArrayList(TypeInfo){};
    defer types.deinit(self.allocator);

    var combined_eff = Effect.pure();

    for (tuple_expr.elements) |elem| {
        var elem_typed = try synthExpr(self, elem);
        defer elem_typed.deinit(self.allocator);
        try types.append(self.allocator, elem_typed.ty);
        mergeEffects(self.allocator, &combined_eff, takeEffect(&elem_typed));
    }

    const ora_types = try self.type_storage_allocator.alloc(OraType, types.items.len);
    for (types.items, 0..) |t, i| {
        var resolved = t;
        if (resolved.ora_type == null and resolved.category == .Integer) {
            resolved.ora_type = .u256;
            resolved.source = .default;
            types.items[i] = resolved;
        }
        if (resolved.ora_type == null) return TypeResolutionError.UnresolvedType;
        ora_types[i] = resolved.ora_type.?;
    }

    const tuple_type = TypeInfo{
        .category = .Tuple,
        .ora_type = OraType{ .tuple = ora_types },
        .source = .inferred,
        .span = tuple_expr.span,
    };

    return Typed.init(tuple_type, combined_eff, self.allocator);
}

fn synthArrayLiteral(
    self: *CoreResolver,
    arr: *ast.Expressions.ArrayLiteralExpr,
) TypeResolutionError!Typed {
    var combined_eff = Effect.pure();

    if (arr.elements.len == 0) {
        return Typed.init(TypeInfo.unknown(), combined_eff, self.allocator);
    }

    var first_typed = try synthExpr(self, arr.elements[0]);
    defer first_typed.deinit(self.allocator);
    mergeEffects(self.allocator, &combined_eff, takeEffect(&first_typed));

    if (first_typed.ty.ora_type == null) return TypeResolutionError.UnresolvedType;
    const elem_ora = first_typed.ty.ora_type.?;

    for (arr.elements[1..]) |elem| {
        var elem_typed = try synthExpr(self, elem);
        defer elem_typed.deinit(self.allocator);
        mergeEffects(self.allocator, &combined_eff, takeEffect(&elem_typed));

        if (elem_typed.ty.ora_type == null) return TypeResolutionError.UnresolvedType;
        if (!OraType.equals(elem_typed.ty.ora_type.?, elem_ora)) {
            return TypeResolutionError.TypeMismatch;
        }
    }

    const elem_ptr = try self.type_storage_allocator.create(OraType);
    elem_ptr.* = elem_ora;
    const array_type = TypeInfo{
        .category = .Array,
        .ora_type = OraType{ .array = .{ .elem = elem_ptr, .len = @intCast(arr.elements.len) } },
        .source = .inferred,
        .span = arr.span,
    };
    return Typed.init(array_type, combined_eff, self.allocator);
}

fn checkArrayLiteral(
    self: *CoreResolver,
    arr: *ast.Expressions.ArrayLiteralExpr,
    expected: TypeInfo,
) TypeResolutionError!Typed {
    const ora_ty = expected.ora_type orelse return TypeResolutionError.TypeMismatch;
    const elem_ora = switch (ora_ty) {
        .array => |arr_ty| arr_ty.elem.*,
        .slice => |elem_ptr| elem_ptr.*,
        else => return TypeResolutionError.TypeMismatch,
    };
    const elem_ty = TypeInfo.fromOraType(elem_ora);
    arr.element_type = elem_ty;

    var combined_eff = Effect.pure();
    for (arr.elements) |elem| {
        var checked = try checkExpr(self, elem, elem_ty);
        defer checked.deinit(self.allocator);
        mergeEffects(self.allocator, &combined_eff, takeEffect(&checked));
    }

    return Typed.init(expected, combined_eff, self.allocator);
}

fn synthAssignment(
    self: *CoreResolver,
    assign: *ast.Expressions.AssignmentExpr,
) TypeResolutionError!Typed {
    // get target type from L-value (Identifier, FieldAccess, or Index)
    // For identifiers, use the DECLARED type (not flow-refined) for assignment validation
    const target_type = blk: {
        switch (assign.target.*) {
            .Identifier => |*id| {
                // For assignment targets, use the DECLARED type, not flow-refined type.
                // This allows `if (x == 0) { x = 1; }` to work correctly.
                // Flow refinements constrain reads, not writes.
                const declared_sym = SymbolTable.findDeclaredUp(
                    @as(?*const Scope, @ptrCast(self.current_scope)),
                    id.name,
                );
                if (declared_sym) |sym| {
                    if (sym.typ) |typ| {
                        // Update the identifier's type_info to the declared type
                        id.type_info = typ;
                        break :blk typ;
                    }
                }
                // Fall back to normal resolution if no declared symbol found
                try @import("identifier.zig").resolveIdentifierType(self, id);
                break :blk id.type_info;
            },
            .FieldAccess => |*fa| {
                // ensure field access type is resolved
                var field_typed = try synthFieldAccess(self, fa);
                defer field_typed.deinit(self.allocator);
                break :blk field_typed.ty;
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
    var value_typed = try synthExpr(self, assign.value);
    defer value_typed.deinit(self.allocator);

    // special case: allow hex literals to satisfy bytes assignments
    if (target_type.category == .Bytes and assign.value.* == .Literal and assign.value.Literal == .Hex) {
        assign.value.Literal.Hex.type_info = target_type;
        value_typed.ty = target_type;
    }

    if (!self.validation.isAssignable(target_type, value_typed.ty)) {
        const got_str = formatTypeInfo(value_typed.ty, self.allocator) catch "unknown";
        const expected_str = formatTypeInfo(target_type, self.allocator) catch "unknown";
        log.debug(
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

    const target_region = inferExprRegion(self, assign.target);
    const source_region = inferExprRegion(self, assign.value);
    if (!isRegionAssignmentAllowed(target_region, source_region, assign.target)) {
        const target_tag = @tagName(target_region);
        const source_tag = @tagName(source_region);
        const target_kind = @tagName(assign.target.*);
        const value_kind = @tagName(assign.value.*);
        log.err(
            "[type_resolver] RegionMismatch: target={s}({s}) source={s}({s})\n",
            .{ target_tag, target_kind, source_tag, value_kind },
        );
        return TypeResolutionError.RegionMismatch;
    }

    // compute effects for implicit region transitions
    const region_eff = try regionEffectForAssignment(
        self,
        target_region,
        source_region,
        assign.target,
        assign.value,
    );

    var combined_eff = takeEffect(&value_typed);
    mergeEffects(self.allocator, &combined_eff, region_eff);
    return Typed.init(target_type, combined_eff, self.allocator);
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
    var operand_typed = try synthExpr(self, cast.operand);
    defer operand_typed.deinit(self.allocator);

    // conservative rule: cast is allowed only if operand is assignable to target
    if (!self.validation.isAssignable(cast.target_type, operand_typed.ty)) {
        const got_str = formatTypeInfo(operand_typed.ty, self.allocator) catch "unknown";
        const expected_str = formatTypeInfo(cast.target_type, self.allocator) catch "unknown";
        log.debug(
            "[type_resolver] Cast type mismatch: got {s}, expected {s}\n",
            .{ got_str, expected_str },
        );
        return TypeResolutionError.TypeMismatch;
    }

    // cast expression yields the target type
    const eff = takeEffect(&operand_typed);
    return Typed.init(cast.target_type, eff, self.allocator);
}

fn synthRange(
    self: *CoreResolver,
    range: *ast.Expressions.RangeExpr,
) TypeResolutionError!Typed {
    const compat = @import("../validation/compatibility.zig");

    // infer start/end types
    var start_typed = try synthExpr(self, range.start);
    defer start_typed.deinit(self.allocator);
    var end_typed = try synthExpr(self, range.end);
    defer end_typed.deinit(self.allocator);

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

    var combined_eff = takeEffect(&start_typed);
    mergeEffects(self.allocator, &combined_eff, takeEffect(&end_typed));
    return Typed.init(chosen, combined_eff, self.allocator);
}

fn synthErrorReturn(
    self: *CoreResolver,
    err_ret: *ast.Expressions.ErrorReturnExpr,
) TypeResolutionError!Typed {
    // look up the error in the symbol table to verify it exists
    const scope = if (self.current_scope) |s| s else self.symbol_table.root;
    var symbol = self.symbol_table.safeFindUpOpt(scope, err_ret.error_name);
    if (symbol == null and scope != self.symbol_table.root) {
        symbol = self.symbol_table.safeFindUpOpt(self.symbol_table.root, err_ret.error_name);
    }
    if (symbol == null) {
        return TypeResolutionError.UndefinedIdentifier;
    }

    const found_symbol = symbol.?;
    if (found_symbol.kind != .Error) {
        return TypeResolutionError.TypeMismatch;
    }

    if (self.symbol_table.error_signatures.get(err_ret.error_name)) |params_opt| {
        if (params_opt != null and params_opt.?.len > 0) {
            return TypeResolutionError.InvalidErrorUsage;
        }
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

    if (!isErrorUnionTypeInfo(inner_typed.ty)) {
        return TypeResolutionError.TypeMismatch;
    }

    // track the error union type for catch binding
    self.last_try_error_union = inner_typed.ty;

    const return_ty = self.current_function_return_type orelse TypeInfo.unknown();
    if (!self.in_try_block and !isErrorUnionTypeInfo(return_ty)) {
        return TypeResolutionError.ErrorUnionOutsideTry;
    }

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
                const eff = takeEffect(&inner_typed);
                return Typed.init(success_type, eff, self.allocator);
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
                        const eff = takeEffect(&inner_typed);
                        return Typed.init(success_type, eff, self.allocator);
                    }
                }
            }
        }
    }

    // if inner expression is not an ErrorUnion, return unknown (should be caught by validation)
    const eff = takeEffect(&inner_typed);
    return Typed.init(TypeInfo.unknown(), eff, self.allocator);
}

fn synthIdentifier(
    self: *CoreResolver,
    id: *ast.Expressions.IdentifierExpr,
) TypeResolutionError!Typed {
    // use identifier resolution module
    try @import("identifier.zig").resolveIdentifierType(self, id);
    var eff = Effect.pure();
    if (self.current_scope) |scope| {
        var sym_opt = self.symbol_table.safeFindUpOpt(scope, id.name);
        if (sym_opt == null and scope != self.symbol_table.root) {
            sym_opt = self.symbol_table.safeFindUpOpt(self.symbol_table.root, id.name);
        }
        if (sym_opt) |sym| {
            if (sym.region == .Storage or sym.region == .TStore) {
                var slots = SlotSet.init(self.allocator);
                try slots.addWithSrc(self.allocator, sym.name, @src());
                eff = Effect.reads(slots);
            }
        }
    }
    return Typed.init(id.type_info, eff, self.allocator);
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

    // validate refinement arithmetic constraints (e.g., scale mismatch)
    try self.refinement_system.validateArithmetic(
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
                // If we couldn't infer a refinement result, drop refinement to base type.
                // This avoids incorrectly narrowing types (e.g. MaxValue * MinValue).
                if ((lhs_typed.ty.ora_type != null and isRefinementOraType(lhs_typed.ty.ora_type.?)) or
                    (rhs_typed.ty.ora_type != null and isRefinementOraType(rhs_typed.ty.ora_type.?)))
                {
                    if (lhs_typed.ty.ora_type) |lhs_ot| {
                        if (extract.extractBaseType(lhs_ot)) |base| {
                            break :blk TypeInfo.inferred(TypeCategory.Integer, base, lhs_typed.ty.span);
                        }
                    }
                }
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
    var combined_eff = takeEffect(&lhs_typed);
    mergeEffects(self.allocator, &combined_eff, takeEffect(&rhs_typed));

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

    if (!self.in_try_block and isErrorUnionTypeInfo(operand_typed.ty)) {
        return TypeResolutionError.ErrorUnionOutsideTry;
    }

    // unary operators preserve operand type (except ! which returns bool)
    const result_type = switch (unary.operator) {
        .Bang => CommonTypes.bool_type(),
        .Minus => blk: {
            // Unary negation is only valid on signed integer types.
            if (operand_typed.ty.ora_type) |ot| {
                if (ot.isUnsignedInteger()) {
                    return TypeResolutionError.IncompatibleTypes;
                }
            }
            break :blk operand_typed.ty;
        },
        else => operand_typed.ty,
    };

    unary.type_info = result_type;
    const empty_delta = LockDelta.emptyWithAllocator(self.allocator);
    const eff = takeEffect(&operand_typed);
    return Typed{
        .ty = result_type,
        .eff = eff,
        .lock_delta = empty_delta,
        .obligations = &.{},
    };
}

fn synthCall(
    self: *CoreResolver,
    call: *ast.Expressions.CallExpr,
) TypeResolutionError!Typed {
    // resolve callee expression (to ensure it's typed)
    var callee_typed = try synthExpr(self, call.callee);
    defer callee_typed.deinit(self.allocator);

    // resolve all argument types
    var combined_eff = takeEffect(&callee_typed);
    for (call.arguments) |arg| {
        var arg_typed = try synthExpr(self, arg);
        defer arg_typed.deinit(self.allocator);
        const eff = takeEffect(&arg_typed);
        mergeEffects(self.allocator, &combined_eff, eff);
    }

    // if callee is a builtin field access, resolve via builtin registry
    if (call.callee.* == .FieldAccess) {
        if (self.builtin_registry) |registry| {
            const base_path = builtins.getMemberAccessPath(self.allocator, call.callee) catch return TypeResolutionError.OutOfMemory;
            defer self.allocator.free(base_path);
            if (registry.lookup(base_path)) |builtin_info| {
                if (builtin_info.is_call) {
                    const ret_info = TypeInfo.fromOraType(builtin_info.return_type);
                    call.type_info = ret_info;
                    return Typed.init(ret_info, combined_eff, self.allocator);
                }
            }
        }
    }

    // if callee is an identifier, look up the function
    if (call.callee.* == .Identifier) {
        const func_name = call.callee.Identifier.name;
        log.debug("[synthCall] Looking up function '{s}'\n", .{func_name});

        // first try symbol table
        const scope = if (self.current_scope) |s| s else self.symbol_table.root;
        var symbol = self.symbol_table.safeFindUpOpt(scope, func_name);
        if (symbol == null and scope != self.symbol_table.root) {
            symbol = self.symbol_table.safeFindUpOpt(self.symbol_table.root, func_name);
        }

        // if not in symbol table, check function registry
        if (symbol == null) {
            if (self.function_registry) |registry| {
                const registry_map = @as(*std.StringHashMap(*FunctionNode), @ptrCast(@alignCast(registry)));
                if (registry_map.get(func_name)) |function| {
                    // found in function registry - use return type
                    if (function.return_type_info) |ret_info| {
                        call.type_info = ret_info;
                        return Typed.init(ret_info, combined_eff, self.allocator);
                    } else {
                        // no return type - return unknown
                        call.type_info = TypeInfo.unknown();
                        return Typed.init(TypeInfo.unknown(), combined_eff, self.allocator);
                    }
                }
            }
            // handle @-prefixed builtins (overflow reporters, divTrunc, etc.)
            if (func_name.len > 1 and func_name[0] == '@') {
                const builtin_suffix = func_name[1..];
                if (resolveOverflowBuiltinType(builtin_suffix, call, self.allocator)) |ret_info| {
                    call.type_info = ret_info;
                    return Typed.init(ret_info, combined_eff, self.allocator);
                }
                // @divTrunc etc. return same type as arguments
                if (std.mem.eql(u8, builtin_suffix, "divTrunc") or
                    std.mem.eql(u8, builtin_suffix, "divFloor") or
                    std.mem.eql(u8, builtin_suffix, "divCeil") or
                    std.mem.eql(u8, builtin_suffix, "divExact") or
                    std.mem.eql(u8, builtin_suffix, "truncate"))
                {
                    if (call.arguments.len > 0) {
                        const first_arg_info = extractExprTypeInfo(call.arguments[0]);
                        call.type_info = first_arg_info;
                        return Typed.init(first_arg_info, combined_eff, self.allocator);
                    }
                    call.type_info = TypeInfo.unknown();
                    return Typed.init(TypeInfo.unknown(), combined_eff, self.allocator);
                }
                if (std.mem.eql(u8, builtin_suffix, "divmod")) {
                    call.type_info = TypeInfo.unknown();
                    return Typed.init(TypeInfo.unknown(), combined_eff, self.allocator);
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

                                log.debug("[synthCall] Function '{s}' return type: category={s}, ora_type={any}\n", .{ func_name, @tagName(ret_category), ret_ora_type });

                                // special handling: if it's a _union with error_union as first element,
                                // treat it as ErrorUnion category
                                if (ret_category == .Union and ret_ora_type == ._union) {
                                    if (ret_ora_type._union.len > 0) {
                                        const first_type = ret_ora_type._union[0];
                                        log.debug("[synthCall] First union element: {any}\n", .{first_type});
                                        if (first_type == .error_union) {
                                            ret_category = .ErrorUnion;
                                            log.debug("[synthCall] Detected error union, changing category to ErrorUnion\n", .{});
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
                            log.debug("[synthCall] Function '{s}' ora_type is not .function: {any}\n", .{ func_name, ora_ty });
                        }
                    } else {
                        log.debug("[synthCall] Function '{s}' typ has no ora_type\n", .{func_name});
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
                        return Typed.init(ret_ty, combined_eff, self.allocator);
                    } else {
                        // fallback: use the function type itself (shouldn't happen)
                        call.type_info = typ;
                        return Typed.init(typ, combined_eff, self.allocator);
                    }
                } else {
                    return TypeResolutionError.UnresolvedType;
                }
            } else {
                return TypeResolutionError.UnresolvedType;
            }
        } else if (found_symbol.kind == .Error) {
            // error call (e.g., InsufficientBalance(amount, balance))
            if (self.symbol_table.error_signatures.get(func_name)) |params_opt| {
                if (params_opt == null) {
                    if (call.arguments.len > 0) {
                        return TypeResolutionError.InvalidErrorUsage;
                    }
                } else {
                    const params = params_opt.?;
                    if (call.arguments.len != params.len) {
                        return TypeResolutionError.InvalidErrorUsage;
                    }
                }
            } else {
                return TypeResolutionError.InvalidErrorUsage;
            }
            // return Error type compatible with ErrorUnion
            const error_ty = TypeInfo{
                .category = .Error,
                .ora_type = null,
                .source = .inferred,
                .span = call.span,
            };
            call.type_info = error_ty;
            return Typed.init(error_ty, combined_eff, self.allocator);
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

    // handle anonymous struct literal field access: (.{ .a = 1 }).a
    if (fa.target.* == .AnonymousStruct) {
        const anon = &fa.target.AnonymousStruct;
        for (anon.fields) |*field| {
            if (!std.mem.eql(u8, field.name, fa.field)) continue;
            var field_typed = try synthExpr(self, field.value);
            defer field_typed.deinit(self.allocator);
            fa.type_info = field_typed.ty;
            return Typed.init(field_typed.ty, takeEffect(&field_typed), self.allocator);
        }
        return TypeResolutionError.TypeMismatch;
    }

    // resolve base expression
    var base_typed = try synthExpr(self, fa.target);
    defer base_typed.deinit(self.allocator);

    // get the type of the base
    const target_type = base_typed.ty;

    // allow pseudo-field access on arrays/slices (e.g., .length)
    if ((target_type.category == .Array or target_type.category == .Slice) and
        std.mem.eql(u8, fa.field, "length"))
    {
        var len_info = TypeInfo.fromOraType(.u256);
        len_info.region = base_typed.ty.region;
        fa.type_info = len_info;
        const eff = takeEffect(&base_typed);
        return Typed.init(len_info, eff, self.allocator);
    }

    // tuple field access (t.0, t.1, ...)
    if (target_type.category == .Tuple) {
        if (target_type.ora_type) |ora_ty| {
            if (ora_ty == .tuple) {
                const field_str = if (fa.field.len > 0 and fa.field[0] == '_') fa.field[1..] else fa.field;
                const index = std.fmt.parseInt(usize, field_str, 10) catch return TypeResolutionError.TypeMismatch;
                if (index >= ora_ty.tuple.len) return TypeResolutionError.TypeMismatch;
                const elem_ora_type = ora_ty.tuple[index];
                const elem_type_info = TypeInfo.inferred(elem_ora_type.getCategory(), elem_ora_type, fa.span);
                var updated = elem_type_info;
                updated.region = base_typed.ty.region;
                fa.type_info = updated;

                const eff = takeEffect(&base_typed);
                return Typed.init(updated, eff, self.allocator);
            }
        }
    }

    // check if the type is a struct
    if (target_type.category == .Struct) {
        if (target_type.ora_type) |ora_ty| {
            if (ora_ty == .struct_type) {
                const struct_name = ora_ty.struct_type;
                if (self.symbol_table.struct_fields.get(struct_name)) |fields| {
                    for (fields) |field| {
                        if (std.mem.eql(u8, field.name, fa.field)) {
                            fa.type_info = field.type_info;
                            fa.type_info.region = base_typed.ty.region;
                            const eff = takeEffect(&base_typed);
                            return Typed.init(field.type_info, eff, self.allocator);
                        }
                    }
                    return TypeResolutionError.TypeMismatch;
                }
            }
        }
    }

    // check if the type is a bitfield
    if (target_type.category == .Bitfield) {
        if (target_type.ora_type) |ora_ty| {
            switch (ora_ty) {
                .bitfield_type => |bf_name| {
                    if (self.symbol_table.bitfield_fields.get(bf_name)) |fields| {
                        for (fields) |field| {
                            if (std.mem.eql(u8, field.name, fa.field)) {
                                fa.type_info = field.type_info;
                                fa.type_info.region = base_typed.ty.region;
                                const eff = takeEffect(&base_typed);
                                return Typed.init(field.type_info, eff, self.allocator);
                            }
                        }
                        return TypeResolutionError.TypeMismatch;
                    }
                },
                else => {},
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
    var combined_eff = takeEffect(&target_typed);
    mergeEffects(self.allocator, &combined_eff, takeEffect(&index_typed));
    if (target_type.category == .Array or target_type.category == .Slice) {
        // extract element type from ora_type
        if (target_type.ora_type) |ora_ty| {
            const elem_ora_type = switch (ora_ty) {
                .array => |arr_ty| arr_ty.elem.*,
                .slice => |elem_ptr| elem_ptr.*,
                else => null,
            };
            if (elem_ora_type) |elem_ora| {
                const elem_type_info = TypeInfo.inferred(elem_ora.getCategory(), elem_ora, null);
                var updated = elem_type_info;
                updated.region = target_typed.ty.region;
                return Typed.init(updated, combined_eff, self.allocator);
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
                var updated = value_type_info;
                updated.region = target_typed.ty.region;
                return Typed.init(updated, combined_eff, self.allocator);
            }
        }
    }

    // fallback: unknown type
    return Typed.init(TypeInfo.unknown(), combined_eff, self.allocator);
}

pub fn inferExprRegion(self: *CoreResolver, expr: *ast.Expressions.ExprNode) MemoryRegion {
    return switch (expr.*) {
        .Identifier => |id| id.type_info.region orelse blk: {
            if (std.mem.eql(u8, id.name, "std")) {
                break :blk MemoryRegion.Calldata;
            }
            if (self.current_scope) |scope| {
                if (self.symbol_table.safeFindUpOpt(scope, id.name)) |sym| {
                    break :blk sym.region orelse MemoryRegion.Stack;
                }
            }
            break :blk MemoryRegion.Stack;
        },
        .FieldAccess => |fa| inferExprRegion(self, fa.target),
        .Index => |ix| inferExprRegion(self, ix.target),
        .Call => MemoryRegion.Stack,
        else => MemoryRegion.Stack,
    };
}

fn isStorageLike(r: MemoryRegion) bool {
    return r == .Storage or r == .TStore;
}

fn isStackOrMemory(r: MemoryRegion) bool {
    return r == .Stack or r == .Memory;
}

pub fn isRegionAssignmentAllowed(target_region: MemoryRegion, source_region: MemoryRegion, target_node: *ast.Expressions.ExprNode) bool {
    _ = target_node;
    if (target_region == .Calldata) return false;
    if (target_region == source_region) return true;
    if (source_region == .Calldata) {
        return isStackOrMemory(target_region) or target_region == .Storage or target_region == .TStore;
    }

    if (isStackOrMemory(target_region)) {
        return isStackOrMemory(source_region) or isStorageLike(source_region) or source_region == .Calldata;
    }

    if (target_region == .Storage) {
        return isStackOrMemory(source_region);
    }

    if (target_region == .TStore) {
        return isStackOrMemory(source_region);
    }

    return false;
}

fn resolveRegionSlot(
    self: *CoreResolver,
    target: *ast.Expressions.ExprNode,
    region: MemoryRegion,
) ?[]const u8 {
    const base_name = findBaseIdentifier(target) orelse return null;
    if (self.current_scope) |scope| {
        if (self.symbol_table.safeFindUpOpt(scope, base_name)) |sym| {
            if (sym.region == region) return sym.name;
        }
    }
    if (self.symbol_table.root.findInCurrent(base_name)) |idx| {
        const sym = self.symbol_table.root.symbols.items[idx];
        if (sym.region == region) return sym.name;
    }
    return null;
}

fn findBaseIdentifier(expr: *const ast.Expressions.ExprNode) ?[]const u8 {
    return switch (expr.*) {
        .Identifier => |id| id.name,
        .FieldAccess => |fa| findBaseIdentifier(fa.target),
        .Index => |ix| findBaseIdentifier(ix.target),
        else => null,
    };
}

fn regionReadEffect(self: *CoreResolver, expr: *ast.Expressions.ExprNode, region: MemoryRegion) TypeResolutionError!Effect {
    if (region == .Storage or region == .TStore) {
        if (resolveRegionSlot(self, expr, region)) |slot_name| {
            var slots = SlotSet.init(self.allocator);
            try slots.addWithSrc(self.allocator, slot_name, @src());
            return Effect.reads(slots);
        }
    }
    return Effect.pure();
}

fn regionWriteEffect(self: *CoreResolver, expr: *ast.Expressions.ExprNode, region: MemoryRegion) TypeResolutionError!Effect {
    if (region == .Storage or region == .TStore) {
        if (resolveRegionSlot(self, expr, region)) |slot_name| {
            var slots = SlotSet.init(self.allocator);
            try slots.addWithSrc(self.allocator, slot_name, @src());
            return Effect.writes(slots);
        }
    }
    return Effect.pure();
}

pub fn regionEffectForAssignment(
    self: *CoreResolver,
    target_region: MemoryRegion,
    source_region: MemoryRegion,
    target_expr: *ast.Expressions.ExprNode,
    source_expr: *ast.Expressions.ExprNode,
) TypeResolutionError!Effect {
    if (target_region == source_region) return Effect.pure();

    if (isStackOrMemory(target_region)) {
        if (source_region == .Storage or source_region == .TStore) {
            return regionReadEffect(self, source_expr, source_region);
        }
        return Effect.pure();
    }

    if ((target_region == .Storage or target_region == .TStore) and isStackOrMemory(source_region)) {
        return regionWriteEffect(self, target_expr, target_region);
    }
    if ((target_region == .Storage or target_region == .TStore) and source_region == .Calldata) {
        return regionWriteEffect(self, target_expr, target_region);
    }

    return Effect.pure();
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
    if (!self.in_try_block and (isErrorUnionTypeInfo(lhs_type) or isErrorUnionTypeInfo(rhs_type))) {
        return TypeResolutionError.ErrorUnionOutsideTry;
    }

    // errorUnion types should not be used directly in binary operations
    // only unwrap inside try blocks
    var lhs_check = lhs_type;
    var rhs_check = rhs_type;

    if (self.in_try_block) {
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

/// Format TypeInfo for error messages, including refinement details
/// Extract type_info from any expression node (for builtin type resolution).
fn extractExprTypeInfo(expr: *const ast.Expressions.ExprNode) TypeInfo {
    return switch (expr.*) {
        .Binary => |b| b.type_info,
        .Unary => |u| u.type_info,
        .Identifier => |i| i.type_info,
        .Literal => |l| switch (l) {
            .Integer => |int_lit| int_lit.type_info,
            .String => |str_lit| str_lit.type_info,
            .Bool => |bool_lit| bool_lit.type_info,
            .Address => |addr_lit| addr_lit.type_info,
            .Hex => |hex_lit| hex_lit.type_info,
            .Binary => |bin_lit| bin_lit.type_info,
            .Character => |char_lit| char_lit.type_info,
            .Bytes => |bytes_lit| bytes_lit.type_info,
        },
        .Call => |call_expr| call_expr.type_info,
        .FieldAccess => |f| f.type_info,
        .Cast => |cast_expr| cast_expr.target_type,
        else => TypeInfo.unknown(),
    };
}

/// Resolve the return type for @opWithOverflow builtins.
/// Returns a tuple type (value: T, overflow: bool) represented as anonymous_struct.
fn resolveOverflowBuiltinType(
    builtin_suffix: []const u8,
    call: *ast.Expressions.CallExpr,
    allocator: std.mem.Allocator,
) ?TypeInfo {
    const overflow_builtins = [_][]const u8{
        "addWithOverflow", "subWithOverflow", "mulWithOverflow",
        "divWithOverflow", "modWithOverflow", "negWithOverflow",
        "shlWithOverflow", "shrWithOverflow",
    };
    var is_overflow = false;
    for (overflow_builtins) |name| {
        if (std.mem.eql(u8, builtin_suffix, name)) {
            is_overflow = true;
            break;
        }
    }
    if (!is_overflow) return null;

    // Determine the value type from the first argument
    var value_type: OraType = .u256; // default
    if (call.arguments.len > 0) {
        const arg_info = extractExprTypeInfo(call.arguments[0]);
        if (arg_info.ora_type) |ot| {
            if (ot.isInteger()) value_type = ot;
        }
    }

    // Build tuple type: (value: T, overflow: bool)
    const value_type_ptr = allocator.create(OraType) catch return null;
    value_type_ptr.* = value_type;
    const bool_type_ptr = allocator.create(OraType) catch return null;
    bool_type_ptr.* = .bool;
    const fields = allocator.alloc(ast.type_info.AnonymousStructFieldType, 2) catch return null;
    fields[0] = .{ .name = "0", .typ = value_type_ptr };
    fields[1] = .{ .name = "1", .typ = bool_type_ptr };
    return TypeInfo{
        .category = .Tuple,
        .ora_type = .{ .anonymous_struct = fields },
        .source = .inferred,
        .span = null,
    };
}

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
