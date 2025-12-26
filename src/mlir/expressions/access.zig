// ============================================================================
// Access Expression Lowering
// ============================================================================
// Lowering for identifiers, field access, and indexing operations

const std = @import("std");
const c = @import("mlir_c_api").c;
const lib = @import("ora_lib");
const constants = @import("../lower.zig");
const h = @import("../helpers.zig");
const TypeMapper = @import("../types.zig").TypeMapper;
const LocationTracker = @import("../locations.zig").LocationTracker;
const OraDialect = @import("../dialect.zig").OraDialect;
const expr_helpers = @import("helpers.zig");
const log = @import("log");

/// ExpressionLowerer type (forward declaration)
const ExpressionLowerer = @import("mod.zig").ExpressionLowerer;

/// Lower identifier expressions
pub fn lowerIdentifier(
    self: *const ExpressionLowerer,
    identifier: *const lib.ast.Expressions.IdentifierExpr,
) c.MlirValue {
    if (std.mem.eql(u8, identifier.name, "std")) {
        log.debug("WARNING: 'std' namespace accessed directly without member - this is a bug\n", .{});
        return self.createConstant(0, identifier.span);
    }

    if (self.param_map) |pm| {
            if (pm.getParamIndex(identifier.name)) |param_index| {
                if (pm.getBlockArgument(identifier.name)) |block_arg| {
                    return block_arg;
                } else {
                    log.debug("FATAL ERROR: Function parameter '{s}' at index {d} not found - compilation aborted\n", .{ identifier.name, param_index });
                    return self.reportLoweringError(
                        identifier.span,
                        "missing function parameter during MLIR lowering",
                        "check parameter mapping and verify function signature lowering",
                    );
                }
            }
        }

    var is_storage_variable = false;
    if (self.storage_map) |sm| {
        if (sm.hasStorageVariable(identifier.name)) {
            is_storage_variable = true;
        }
    }

    if (is_storage_variable) {
        const memory_manager = @import("../memory.zig").MemoryManager.init(self.ctx, self.ora_dialect);
        const var_type = if (identifier.type_info.ora_type) |_| blk: {
            break :blk self.type_mapper.toMlirType(identifier.type_info);
        } else if (self.symbol_table) |st| blk: {
            if (st.lookupSymbol(identifier.name)) |symbol| {
                break :blk symbol.type;
            }
            break :blk c.mlirIntegerTypeGet(self.ctx, constants.DEFAULT_INTEGER_BITS);
        } else blk: {
            break :blk c.mlirIntegerTypeGet(self.ctx, constants.DEFAULT_INTEGER_BITS);
        };

        const result_name = identifier.name;
        log.debug("[lowerIdentifier] Loading storage variable '{s}' with type {any}\n", .{ identifier.name, var_type });
        const load_op = memory_manager.createStorageLoadWithName(identifier.name, var_type, self.fileLoc(identifier.span), result_name);
        h.appendOp(self.block, load_op);
        const result = h.getResult(load_op, 0);
        const actual_type = c.mlirValueGetType(result);
        log.debug("[lowerIdentifier] Storage load for '{s}' returned type {any}\n", .{ identifier.name, actual_type });
        return result;
    }

    if (self.local_var_map) |lvm| {
        if (lvm.getLocalVar(identifier.name)) |local_var_ref| {
            const var_type = c.mlirValueGetType(local_var_ref);

            if (c.mlirTypeIsAMemRef(var_type)) {
                const element_type = c.mlirShapedTypeGetElementType(var_type);
                const bool_ty = h.boolType(self.ctx);
                const rank = c.mlirShapedTypeGetRank(var_type);
                var is_scalar: bool = false;
                if (rank == 0) {
                    if (c.mlirTypeIsAInteger(element_type)) {
                        is_scalar = true;
                    } else if (c.mlirTypeEqual(element_type, bool_ty)) {
                        is_scalar = true;
                    }
                }

                if (is_scalar) {
                    const indices = [_]c.MlirValue{};
                    const load_op = self.ora_dialect.createMemrefLoad(local_var_ref, &indices, element_type, self.fileLoc(identifier.span));
                    h.appendOp(self.block, load_op);
                    return h.getResult(load_op, 0);
                }

                return local_var_ref;
            }

            log.debug("[lowerIdentifier] Returning scalar SSA value: {any}\n", .{local_var_ref});
            return local_var_ref;
        }
    }

    if (self.symbol_table) |st| {
        if (st.lookupSymbol(identifier.name)) |symbol| {
            // check if this is an error identifier (errors are symbols with .Error kind)
            if (symbol.symbol_kind == .Error) {
                // lower as an error return expression
                const ty = c.mlirIntegerTypeGet(self.ctx, constants.DEFAULT_INTEGER_BITS);
                var state = h.opState("arith.constant", self.fileLoc(identifier.span));
                c.mlirOperationStateAddResults(&state, 1, @ptrCast(&ty));

                const attr = c.mlirIntegerAttrGet(ty, 1);
                const value_id = h.identifier(self.ctx, "value");
                const error_id = h.identifier(self.ctx, "ora.error");
                const error_name_attr = h.stringAttr(self.ctx, identifier.name);

                var attrs = [_]c.MlirNamedAttribute{
                    c.mlirNamedAttributeGet(value_id, attr),
                    c.mlirNamedAttributeGet(error_id, error_name_attr),
                };
                c.mlirOperationStateAddAttributes(&state, attrs.len, &attrs);

                const op = c.mlirOperationCreate(&state);
                h.appendOp(self.block, op);
                return h.getResult(op, 0);
            }

            if (symbol.symbol_kind == .Constant) {
                log.debug("[lowerIdentifier] Found constant: {s}, creating value in current block\n", .{identifier.name});

                if (st.lookupConstantDecl(identifier.name)) |const_decl| {
                    const const_value = self.lowerExpression(const_decl.value);
                    log.debug("[lowerIdentifier] Created constant value for: {s}\n", .{identifier.name});
                    return const_value;
                } else {
                    log.debug("WARNING: Constant '{s}' declaration not found\n", .{identifier.name});
                }
            }
        }

        if (st.lookupType(identifier.name)) |type_symbol| {
            log.debug("[lowerIdentifier] Found type: {s}, type_kind: {any}\n", .{ identifier.name, type_symbol.type_kind });
            const type_val = c.mlirIntegerTypeGet(self.ctx, constants.DEFAULT_INTEGER_BITS);
            const zero_const = c.mlirIntegerTypeGet(self.ctx, constants.DEFAULT_INTEGER_BITS);
            var state = h.opState("arith.constant", self.fileLoc(identifier.span));
            const zero_attr = c.mlirIntegerAttrGet(zero_const, 0);
            const attr_id = h.identifier(self.ctx, "value");
            const named_attr = c.mlirNamedAttributeGet(attr_id, zero_attr);
            c.mlirOperationStateAddAttributes(&state, 1, &named_attr);
            c.mlirOperationStateAddResults(&state, 1, @ptrCast(&type_val));
            const const_op = c.mlirOperationCreate(&state);
            h.appendOp(self.block, const_op);
            return h.getResult(const_op, 0);
        }
    }

    // check builtin registry as last resort (for constants like std.constants.ZERO_ADDRESS)
    // note: Builtin functions should be accessed via function calls, not identifiers
    if (self.builtin_registry) |registry| {
        // check if this identifier matches a builtin constant path
        // this handles cases where a builtin constant is accessed directly
        if (registry.lookup(identifier.name)) |builtin_info| {
            if (!builtin_info.is_call) {
                // it's a constant - create a constant value
                const result_type = self.type_mapper.toMlirType(.{
                    .ora_type = builtin_info.return_type,
                });
                // for now, return a zero value - proper builtin constant lowering should be implemented
                const zero_const = c.mlirIntegerTypeGet(self.ctx, constants.DEFAULT_INTEGER_BITS);
                var state = h.opState("arith.constant", self.fileLoc(identifier.span));
                const zero_attr = c.mlirIntegerAttrGet(zero_const, 0);
                const attr_id = h.identifier(self.ctx, "value");
                const named_attr = c.mlirNamedAttributeGet(attr_id, zero_attr);
                c.mlirOperationStateAddAttributes(&state, 1, &named_attr);
                c.mlirOperationStateAddResults(&state, 1, @ptrCast(&result_type));
                const const_op = c.mlirOperationCreate(&state);
                h.appendOp(self.block, const_op);
                return h.getResult(const_op, 0);
            }
        }
    }

    // if identifier is still undefined, return error placeholder instead of panicking
    // this allows compilation to continue and report errors more gracefully
    log.debug("ERROR: Undefined identifier '{s}' at {d}:{d}\n", .{ identifier.name, identifier.span.line, identifier.span.column });
    return self.createErrorPlaceholder(identifier.span, "Undefined identifier");
}

/// Lower index expressions
pub fn lowerIndex(
    self: *const ExpressionLowerer,
    index: *const lib.ast.Expressions.IndexExpr,
) c.MlirValue {
    const target = self.lowerExpression(index.target);
    const index_val = self.lowerExpression(index.index);
    const target_type = c.mlirValueGetType(target);

    // handle memrefs and tensors/shaped types with tensor.extract/memref.load
    if (c.mlirTypeIsAMemRef(target_type) or c.mlirTypeIsAShaped(target_type)) {
        return createArrayIndexLoad(self, target, index_val, index.span);
    } else {
        var result_type: ?c.MlirType = null;

        // try to get the result type from the target's AST type info if it's an identifier
        if (index.target.* == .Identifier) {
            const ident = index.target.Identifier;
            log.debug("[lowerIndex] Target is identifier: {s}\n", .{ident.name});
            if (ident.type_info.ora_type) |ora_type| {
                log.debug("[lowerIndex] Identifier has ora_type\n", .{});
                if (ora_type == .map) {
                    log.debug("[lowerIndex] Type is map, extracting value type\n", .{});
                    // extract the value type from the map type
                    const value_ora_type = ora_type.map.value.*;
                    const value_type_info = lib.ast.Types.TypeInfo{
                        .category = switch (value_ora_type) {
                            .struct_type => .Struct,
                            .u256, .u128, .u64, .u32, .u16, .u8 => .Integer,
                            .address => .Address,
                            .bool => .Bool,
                            else => .Unknown,
                        },
                        .ora_type = value_ora_type,
                        .source = .inferred,
                        .span = null,
                    };
                    result_type = self.type_mapper.toMlirType(value_type_info);
                    log.debug("[lowerIndex] Extracted result_type from AST\n", .{});
                }
            }
        }

        // fallback to extracting from MLIR map type
        if (result_type == null) {
            log.debug("[lowerIndex] Falling back to MLIR map type extraction\n", .{});
            log.debug("[lowerIndex] Target MLIR type: {any}\n", .{target_type});
            const extracted_value_type = c.oraMapTypeGetValueType(target_type);
            if (extracted_value_type.ptr != null) {
                log.debug("[lowerIndex] Extracted value type from MLIR map type: {any}\n", .{extracted_value_type});
                result_type = extracted_value_type;
            } else {
                log.debug("[lowerIndex] WARNING: Could not extract value type from MLIR map type\n", .{});
            }
        }

        // debug: Print what result_type we're using
        if (result_type) |rt| {
            log.debug("[lowerIndex] Final result_type for map_get: {any}\n", .{rt});
        } else {
            log.debug("[lowerIndex] WARNING: result_type is null, will use fallback\n", .{});
        }

        // verify the map type's value type matches our expected result_type
        const map_value_type = c.oraMapTypeGetValueType(target_type);
        if (map_value_type.ptr != null) {
            log.debug("[lowerIndex] Map type's value type: {any}\n", .{map_value_type});
            if (result_type) |rt| {
                if (!c.mlirTypeEqual(map_value_type, rt)) {
                    log.debug("[lowerIndex] WARNING: Map value type {any} doesn't match expected result_type {any}\n", .{ map_value_type, rt });
                }
            }
        }

        return createMapIndexLoad(self, target, index_val, result_type, index.span);
    }
}

/// Lower field access expressions
pub fn lowerFieldAccess(
    self: *const ExpressionLowerer,
    field: *const lib.ast.Expressions.FieldAccessExpr,
) c.MlirValue {
    if (self.builtin_registry) |registry| {
        const field_expr_node = lib.ast.Expressions.ExprNode{ .FieldAccess = field.* };
        const builtins = lib.semantics.builtins;
        const path = builtins.getMemberAccessPath(registry.allocator, &field_expr_node) catch {
            const target = self.lowerExpression(field.target);
            _ = c.mlirValueGetType(target);
            return createStructFieldExtract(self, target, field.field, field.span);
        };
        defer registry.allocator.free(path);

        if (registry.lookup(path)) |builtin_info| {
            if (!builtin_info.is_call) {
                return lowerBuiltinConstant(self, &builtin_info, field.span);
            }
        }
    }

    const target = self.lowerExpression(field.target);
    _ = c.mlirValueGetType(target);
    return createStructFieldExtract(self, target, field.field, field.span);
}

/// Lower builtin constant (e.g., std.constants.ZERO_ADDRESS)
fn lowerBuiltinConstant(
    self: *const ExpressionLowerer,
    builtin_info: *const lib.semantics.builtins.BuiltinInfo,
    span: lib.ast.SourceSpan,
) c.MlirValue {
    const ty = self.type_mapper.toMlirType(.{
        .ora_type = builtin_info.return_type,
    });

    if (std.mem.eql(u8, builtin_info.full_path, "std.constants.ZERO_ADDRESS")) {
        const addr_ty = self.type_mapper.toMlirType(.{
            .ora_type = builtin_info.return_type,
        });

        const i160_ty = c.mlirIntegerTypeGet(self.ctx, 160);
        var state = h.opState("arith.constant", self.fileLoc(span));
        c.mlirOperationStateAddResults(&state, 1, @ptrCast(&i160_ty));

        const value_attr = c.mlirIntegerAttrGet(i160_ty, 0);
        const value_id = h.identifier(self.ctx, "value");
        const gas_cost_attr = c.mlirIntegerAttrGet(c.mlirIntegerTypeGet(self.ctx, 64), 0);
        const gas_cost_id = h.identifier(self.ctx, "gas_cost");

        var attrs = [_]c.MlirNamedAttribute{
            c.mlirNamedAttributeGet(value_id, value_attr),
            c.mlirNamedAttributeGet(gas_cost_id, gas_cost_attr),
        };
        c.mlirOperationStateAddAttributes(&state, attrs.len, &attrs);

        const const_op = c.mlirOperationCreate(&state);
        h.appendOp(self.block, const_op);
        const i160_value = h.getResult(const_op, 0);

        var bitcast_state = h.opState("arith.bitcast", self.fileLoc(span));
        c.mlirOperationStateAddOperands(&bitcast_state, 1, @ptrCast(&i160_value));
        c.mlirOperationStateAddResults(&bitcast_state, 1, @ptrCast(&addr_ty));
        const bitcast_op = c.mlirOperationCreate(&bitcast_state);
        h.appendOp(self.block, bitcast_op);
        return h.getResult(bitcast_op, 0);
    }

    if (std.mem.eql(u8, builtin_info.full_path, "std.constants.U256_MAX") or
        std.mem.eql(u8, builtin_info.full_path, "std.constants.U128_MAX") or
        std.mem.eql(u8, builtin_info.full_path, "std.constants.U64_MAX") or
        std.mem.eql(u8, builtin_info.full_path, "std.constants.U32_MAX"))
    {
        const op = self.ora_dialect.createArithConstant(-1, ty, self.fileLoc(span));
        h.appendOp(self.block, op);
        return h.getResult(op, 0);
    }

    const op = self.ora_dialect.createArithConstant(0, ty, self.fileLoc(span));
    h.appendOp(self.block, op);
    return h.getResult(op, 0);
}

/// Create struct field extract operation
pub fn createStructFieldExtract(
    self: *const ExpressionLowerer,
    struct_val: c.MlirValue,
    field_name: []const u8,
    span: lib.ast.SourceSpan,
) c.MlirValue {
    var result_ty = c.mlirIntegerTypeGet(self.ctx, constants.DEFAULT_INTEGER_BITS);

    if (self.symbol_table) |st| {
        var type_iter = st.types.iterator();
        while (type_iter.next()) |entry| {
            const type_symbols = entry.value_ptr.*;
            for (type_symbols) |type_sym| {
                if (type_sym.type_kind == .Struct) {
                    if (type_sym.fields) |fields| {
                        for (fields) |field| {
                            if (std.mem.eql(u8, field.name, field_name)) {
                                result_ty = field.field_type;
                                break;
                            }
                        }
                    }
                }
            }
        }
    }

    const op = self.ora_dialect.createStructFieldExtract(struct_val, field_name, result_ty, self.fileLoc(span));
    h.appendOp(self.block, op);
    return h.getResult(op, 0);
}

/// Create pseudo-field access for built-in types (e.g., array.length)
pub fn createPseudoFieldAccess(
    self: *const ExpressionLowerer,
    target: c.MlirValue,
    field_name: []const u8,
    span: lib.ast.SourceSpan,
) c.MlirValue {
    if (std.mem.eql(u8, field_name, "length")) {
        return createLengthAccess(self, target, span);
    } else {
        log.debug("WARNING: Unknown pseudo-field '{s}'\n", .{field_name});
        return self.createErrorPlaceholder(span, "Unknown pseudo-field");
    }
}

/// Create length access for arrays and slices
pub fn createLengthAccess(
    self: *const ExpressionLowerer,
    target: c.MlirValue,
    span: lib.ast.SourceSpan,
) c.MlirValue {
    const target_type = c.mlirValueGetType(target);
    const index_ty = c.mlirIndexTypeGet(self.ctx);

    if (c.mlirTypeIsAMemRef(target_type) or c.mlirTypeIsAShaped(target_type)) {
        if (c.mlirTypeIsAShaped(target_type) and c.mlirShapedTypeHasStaticShape(target_type)) {
            const num_dims = c.mlirShapedTypeGetRank(target_type);
            if (num_dims > 0) {
                const dim_size = c.mlirShapedTypeGetDimSize(target_type, 0);

                if (dim_size >= 0) {
                    const len_const = self.createConstant(@intCast(dim_size), span);
                    return convertIndexToIndexType(self, len_const, span);
                }
            }
        }

        if (c.mlirTypeIsAMemRef(target_type)) {
            var state = h.opState("memref.dim", self.fileLoc(span));
            c.mlirOperationStateAddOperands(&state, 1, @ptrCast(&target));

            var dim_const_state = h.opState("arith.constant", self.fileLoc(span));
            c.mlirOperationStateAddResults(&dim_const_state, 1, @ptrCast(&index_ty));
            const dim_attr = c.mlirIntegerAttrGet(index_ty, 0);
            const dim_value_id = h.identifier(self.ctx, "value");
            var dim_attrs = [_]c.MlirNamedAttribute{c.mlirNamedAttributeGet(dim_value_id, dim_attr)};
            c.mlirOperationStateAddAttributes(&dim_const_state, dim_attrs.len, &dim_attrs);
            const dim_const_op = c.mlirOperationCreate(&dim_const_state);
            h.appendOp(self.block, dim_const_op);
            const dim_index = h.getResult(dim_const_op, 0);

            c.mlirOperationStateAddOperands(&state, 1, @ptrCast(&dim_index));
            c.mlirOperationStateAddResults(&state, 1, @ptrCast(&index_ty));

            const op = c.mlirOperationCreate(&state);
            h.appendOp(self.block, op);
            return h.getResult(op, 0);
        }

        var state = h.opState("tensor.dim", self.fileLoc(span));
        c.mlirOperationStateAddOperands(&state, 1, @ptrCast(&target));

        var dim_const_state = h.opState("arith.constant", self.fileLoc(span));
        c.mlirOperationStateAddResults(&dim_const_state, 1, @ptrCast(&index_ty));
        const dim_attr = c.mlirIntegerAttrGet(index_ty, 0);
        const dim_value_id = h.identifier(self.ctx, "value");
        var dim_attrs = [_]c.MlirNamedAttribute{c.mlirNamedAttributeGet(dim_value_id, dim_attr)};
        c.mlirOperationStateAddAttributes(&dim_const_state, dim_attrs.len, &dim_attrs);
        const dim_const_op = c.mlirOperationCreate(&dim_const_state);
        h.appendOp(self.block, dim_const_op);
        const dim_index = h.getResult(dim_const_op, 0);

        c.mlirOperationStateAddOperands(&state, 1, @ptrCast(&dim_index));
        c.mlirOperationStateAddResults(&state, 1, @ptrCast(&index_ty));

        const op = c.mlirOperationCreate(&state);
        h.appendOp(self.block, op);
        return h.getResult(op, 0);
    } else {
        var state = h.opState("ora.length", self.fileLoc(span));
        c.mlirOperationStateAddOperands(&state, 1, @ptrCast(&target));
        c.mlirOperationStateAddResults(&state, 1, @ptrCast(&index_ty));
        const op = c.mlirOperationCreate(&state);
        h.appendOp(self.block, op);
        return h.getResult(op, 0);
    }
}

/// Convert index value to MLIR index type
pub fn convertIndexToIndexType(
    self: *const ExpressionLowerer,
    index: c.MlirValue,
    span: lib.ast.SourceSpan,
) c.MlirValue {
    const index_type = c.mlirValueGetType(index);
    const mlir_index_type = c.mlirIndexTypeGet(self.ctx);

    if (c.mlirTypeEqual(index_type, mlir_index_type)) {
        return index;
    }

    var current_value = index;
    const current_type = c.mlirValueGetType(current_value);

    if (c.mlirTypeIsAInteger(current_type)) {
        const width = c.mlirIntegerTypeGetWidth(current_type);
        const signless_type = c.mlirIntegerTypeGet(self.ctx, width);
        if (!c.mlirTypeEqual(current_type, signless_type)) {
            var signless_cast_state = h.opState("arith.bitcast", self.fileLoc(span));
            c.mlirOperationStateAddOperands(&signless_cast_state, 1, @ptrCast(&current_value));
            c.mlirOperationStateAddResults(&signless_cast_state, 1, @ptrCast(&signless_type));
            const signless_cast_op = c.mlirOperationCreate(&signless_cast_state);
            h.appendOp(self.block, signless_cast_op);
            current_value = h.getResult(signless_cast_op, 0);
        }
    }

    var cast_state = h.opState("arith.index_castui", self.fileLoc(span));
    c.mlirOperationStateAddOperands(&cast_state, 1, @ptrCast(&current_value));
    c.mlirOperationStateAddResults(&cast_state, 1, @ptrCast(&mlir_index_type));
    const index_cast_op = c.mlirOperationCreate(&cast_state);
    h.appendOp(self.block, index_cast_op);
    return h.getResult(index_cast_op, 0);
}

/// Create array index load with bounds checking
pub fn createArrayIndexLoad(
    self: *const ExpressionLowerer,
    array: c.MlirValue,
    index: c.MlirValue,
    span: lib.ast.SourceSpan,
) c.MlirValue {
    const array_type = c.mlirValueGetType(array);
    const index_index = convertIndexToIndexType(self, index, span);
    const enable_bounds_check = true;

    if (enable_bounds_check and (c.mlirTypeIsAMemRef(array_type) or c.mlirTypeIsAShaped(array_type))) {
        const array_length = createLengthAccess(self, array, span);

        var cmp_state = h.opState("arith.cmpi", self.fileLoc(span));
        c.mlirOperationStateAddOperands(&cmp_state, 2, @ptrCast(&[_]c.MlirValue{ index_index, array_length }));
        const bool_ty = h.boolType(self.ctx);
        c.mlirOperationStateAddResults(&cmp_state, 1, @ptrCast(&bool_ty));

        const predicate_value = expr_helpers.predicateStringToInt("ult");
        const predicate_attr = c.mlirIntegerAttrGet(c.mlirIntegerTypeGet(self.ctx, 64), predicate_value);
        const predicate_id = h.identifier(self.ctx, "predicate");
        var cmp_attrs = [_]c.MlirNamedAttribute{
            c.mlirNamedAttributeGet(predicate_id, predicate_attr),
        };
        c.mlirOperationStateAddAttributes(&cmp_state, cmp_attrs.len, &cmp_attrs);

        const cmp_op = c.mlirOperationCreate(&cmp_state);
        h.appendOp(self.block, cmp_op);
        const in_bounds = h.getResult(cmp_op, 0);

        var assert_state = h.opState("cf.assert", self.fileLoc(span));
        c.mlirOperationStateAddOperands(&assert_state, 1, @ptrCast(&in_bounds));
        const msg_id = h.identifier(self.ctx, "msg");
        const msg_attr = h.stringAttr(self.ctx, "array index out of bounds");
        c.mlirOperationStateAddAttributes(&assert_state, 1, @ptrCast(&c.mlirNamedAttributeGet(msg_id, msg_attr)));
        const assert_op = c.mlirOperationCreate(&assert_state);
        h.appendOp(self.block, assert_op);
    }

    const element_type = if (c.mlirTypeIsAMemRef(array_type) or c.mlirTypeIsAShaped(array_type))
        c.mlirShapedTypeGetElementType(array_type)
    else
        c.mlirIntegerTypeGet(self.ctx, constants.DEFAULT_INTEGER_BITS);

    if (c.mlirTypeIsAMemRef(array_type)) {
        var load_state = h.opState("memref.load", self.fileLoc(span));
        c.mlirOperationStateAddOperands(&load_state, 2, @ptrCast(&[_]c.MlirValue{ array, index_index }));
        c.mlirOperationStateAddResults(&load_state, 1, @ptrCast(&element_type));
        const load_op = c.mlirOperationCreate(&load_state);
        h.appendOp(self.block, load_op);
        return h.getResult(load_op, 0);
    } else if (c.mlirTypeIsAShaped(array_type)) {
        var extract_state = h.opState("tensor.extract", self.fileLoc(span));
        c.mlirOperationStateAddOperands(&extract_state, 1, @ptrCast(&array));
        c.mlirOperationStateAddOperands(&extract_state, 1, @ptrCast(&index_index));
        c.mlirOperationStateAddResults(&extract_state, 1, @ptrCast(&element_type));
        const extract_op = c.mlirOperationCreate(&extract_state);
        h.appendOp(self.block, extract_op);
        return h.getResult(extract_op, 0);
    } else {
        return array;
    }
}

/// Create map index load operation
pub fn createMapIndexLoad(
    self: *const ExpressionLowerer,
    map: c.MlirValue,
    key: c.MlirValue,
    result_type: ?c.MlirType,
    span: lib.ast.SourceSpan,
) c.MlirValue {
    // ensure we're not using ora.map_get on tensors - they should use tensor.extract
    // check if the type is a shaped type but not a map (tensors are shaped but not maps)
    const map_type_check = c.mlirValueGetType(map);
    if (c.mlirTypeIsAShaped(map_type_check)) {
        // try to extract value type - if it fails, it's likely a tensor, not a map
        const test_value_type = c.oraMapTypeGetValueType(map_type_check);
        if (test_value_type.ptr == null) {
            log.debug("ERROR: createMapIndexLoad called on tensor/shaped type - should use tensor.extract instead\n", .{});
            return self.reportLoweringError(
                span,
                "cannot use ora.map_get on tensor - use tensor.extract",
                "use array indexing or tensor.extract for shaped types",
            );
        }
    }

    var result_ty: c.MlirType = undefined;
    if (result_type) |ty| {
        log.debug("[createMapIndexLoad] Using provided result_type\n", .{});
        result_ty = ty;
    } else {
        log.debug("[createMapIndexLoad] Extracting value type from map type\n", .{});
        const map_type = c.mlirValueGetType(map);
        const extracted_value_type = c.oraMapTypeGetValueType(map_type);
        if (extracted_value_type.ptr != null) {
            log.debug("[createMapIndexLoad] Extracted value type from map\n", .{});
            result_ty = extracted_value_type;
        } else {
            log.debug("WARNING: createMapIndexLoad: Could not extract value type from map, defaulting to i256.\n", .{});
            result_ty = c.mlirIntegerTypeGet(self.ctx, constants.DEFAULT_INTEGER_BITS);
        }
    }
    log.debug("[createMapIndexLoad] Creating ora.map_get with result_type: {any}\n", .{result_ty});

    // verify map type's value type matches our expected result_type
    const map_type_verify = c.mlirValueGetType(map);
    const map_value_type = c.oraMapTypeGetValueType(map_type_verify);
    if (map_value_type.ptr != null) {
        log.debug("[createMapIndexLoad] Map type's value type: {any}\n", .{map_value_type});
        if (!c.mlirTypeEqual(map_value_type, result_ty)) {
            log.debug("[createMapIndexLoad] ERROR: Map value type {any} doesn't match expected result_type {any}\n", .{ map_value_type, result_ty });
            log.debug("[createMapIndexLoad] This will cause ora.map_get to return the wrong type!\n", .{});
            log.debug("[createMapIndexLoad] Using map_value_type instead of result_type to avoid type mismatch\n", .{});
            // use the map's value type instead of the expected result_type to avoid type mismatch
            result_ty = map_value_type;
        }
    }

    const op = self.ora_dialect.createMapGet(map, key, result_ty, self.fileLoc(span));
    h.appendOp(self.block, op);
    const result = h.getResult(op, 0);
    const actual_result_type = c.mlirValueGetType(result);
    log.debug("[createMapIndexLoad] ora.map_get created, actual result type: {any}\n", .{actual_result_type});

    // check if map_get returned the wrong type
    if (!c.mlirTypeEqual(actual_result_type, result_ty)) {
        log.debug("[createMapIndexLoad] ERROR: map_get returned {any} but we requested {any}\n", .{ actual_result_type, result_ty });
        log.debug("[createMapIndexLoad] This is a bug - ora.map_get should respect the result_type parameter\n", .{});
        // for now, return the result as-is - the type system will catch this error later
    }

    return result;
}
