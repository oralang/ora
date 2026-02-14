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
                if (self.prefer_refinement_base_cache) {
                    if (pm.getBaseArgument(identifier.name)) |base_arg| {
                        return base_arg;
                    }
                    if (self.refinement_base_cache) |cache| {
                        const key = if (c.mlirValueIsABlockArgument(block_arg)) blk: {
                            const owner = c.mlirBlockArgumentGetOwner(block_arg);
                            const arg_no = c.mlirBlockArgumentGetArgNumber(block_arg);
                            const block_key = @intFromPtr(owner.ptr);
                            const arg_key: usize = @intCast(arg_no);
                            break :blk block_key ^ (arg_key *% 0x9e3779b97f4a7c15);
                        } else @intFromPtr(block_arg.ptr);
                        if (cache.get(key)) |cached| {
                            return cached;
                        }
                    }
                }
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
            break :blk c.oraIntegerTypeCreate(self.ctx, constants.DEFAULT_INTEGER_BITS);
        } else blk: {
            break :blk c.oraIntegerTypeCreate(self.ctx, constants.DEFAULT_INTEGER_BITS);
        };

        const result_name = identifier.name;
        log.debug("[lowerIdentifier] Loading storage variable '{s}' with type {any}\n", .{ identifier.name, var_type });
        const load_op = memory_manager.createStorageLoadWithName(identifier.name, var_type, self.fileLoc(identifier.span), result_name);
        h.appendOp(self.block, load_op);
        const result = h.getResult(load_op, 0);
        const actual_type = c.oraValueGetType(result);
        log.debug("[lowerIdentifier] Storage load for '{s}' returned type {any}\n", .{ identifier.name, actual_type });
        return result;
    }

    if (self.local_var_map) |lvm| {
        if (lvm.getLocalVar(identifier.name)) |local_var_ref| {
            const var_type = c.oraValueGetType(local_var_ref);
            log.debug("[lowerIdentifier] local var '{s}' type={any} memref={} shaped={}\n", .{
                identifier.name,
                var_type,
                c.oraTypeIsAMemRef(var_type),
                c.oraTypeIsAShaped(var_type),
            });

            if (c.oraTypeIsAMemRef(var_type)) {
                const element_type = c.oraShapedTypeGetElementType(var_type);
                const bool_ty = h.boolType(self.ctx);
                const rank = c.oraShapedTypeGetRank(var_type);
                var is_scalar: bool = false;
                if (rank == 0) {
                    if (c.oraTypeIsAInteger(element_type)) {
                        is_scalar = true;
                    } else if (c.oraTypeEqual(element_type, bool_ty)) {
                        is_scalar = true;
                    } else if (c.oraRefinementTypeGetBaseType(element_type).ptr != null) {
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
            if (std.mem.eql(u8, symbol.region, "tstore")) {
                const memory_manager = @import("../memory.zig").MemoryManager.init(self.ctx, self.ora_dialect);
                const load_op = memory_manager.createTStoreLoad(identifier.name, self.fileLoc(identifier.span));
                h.appendOp(self.block, load_op);
                return h.getResult(load_op, 0);
            }
            // check if this is an error identifier (errors are symbols with .Error kind)
            if (symbol.symbol_kind == .Error) {
                // lower as an error return expression
                const ty = c.oraIntegerTypeCreate(self.ctx, constants.DEFAULT_INTEGER_BITS);
                const error_id = h.identifier(self.ctx, "ora.error");
                const error_name_attr = h.stringAttr(self.ctx, identifier.name);

                const attrs = [_]c.MlirNamedAttribute{
                    c.oraNamedAttributeGet(error_id, error_name_attr),
                };
                const op = self.ora_dialect.createArithConstantWithAttrs(1, ty, &attrs, self.fileLoc(identifier.span));
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
            const type_val = c.oraIntegerTypeCreate(self.ctx, constants.DEFAULT_INTEGER_BITS);
            const const_op = self.ora_dialect.createArithConstant(0, type_val, self.fileLoc(identifier.span));
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
                const const_op = self.ora_dialect.createArithConstant(0, result_type, self.fileLoc(identifier.span));
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
    const target_type = c.oraValueGetType(target);

    // handle memrefs and tensors/shaped types with tensor.extract/memref.load
    if (c.oraTypeIsAMemRef(target_type) or c.oraTypeIsAShaped(target_type)) {
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
                if (!c.oraTypeEqual(map_value_type, rt)) {
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
            _ = c.oraValueGetType(target);
            return createStructFieldExtract(self, target, field.field, field.span);
        };
        defer registry.allocator.free(path);

        if (registry.lookup(path)) |builtin_info| {
            if (builtin_info.is_call) {
                return lowerBuiltinFieldCall(self, &builtin_info, field.span);
            }
            return lowerBuiltinConstant(self, &builtin_info, field.span);
        }
    }

    const target = self.lowerExpression(field.target);
    const target_type = c.oraValueGetType(target);

    if ((c.oraTypeIsAMemRef(target_type) or c.oraTypeIsAShaped(target_type)) and
        std.mem.eql(u8, field.field, "length"))
    {
        return createPseudoFieldAccess(self, target, field.field, field.span);
    }

    // Check if target is a bitfield type → emit shift+mask extraction
    if (self.symbol_table) |st| {
        const target_type_info = @import("operators.zig").extractTypeInfo(field.target);
        if (target_type_info.ora_type) |ora_type| {
            if (ora_type == .bitfield_type) {
                if (st.lookupType(ora_type.bitfield_type)) |type_sym| {
                    if (type_sym.type_kind == .Bitfield) {
                        return createBitfieldFieldExtract(self, target, field.field, type_sym, field.span);
                    }
                }
            }
        }
        // Fallback: if the MLIR type is an integer (bitfields lower to i256), scan all
        // bitfield types for a matching field. Handles cases where the AST type_info is
        // unknown (e.g., enum-literal fallback path).
        if (c.oraTypeIsAInteger(target_type) or c.oraTypeIsAOraInteger(target_type)) {
            var type_iter = st.types.iterator();
            while (type_iter.next()) |entry| {
                for (entry.value_ptr.*) |type_sym| {
                    if (type_sym.type_kind == .Bitfield) {
                        if (type_sym.fields) |fields| {
                            for (fields) |f| {
                                if (std.mem.eql(u8, f.name, field.field)) {
                                    return createBitfieldFieldExtract(self, target, field.field, &type_sym, field.span);
                                }
                            }
                        }
                    }
                }
            }
        }
    }

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
        const i160_ty = c.oraIntegerTypeCreate(self.ctx, 160);
        const const_op = self.ora_dialect.createArithConstant(0, i160_ty, self.fileLoc(span));
        h.appendOp(self.block, const_op);
        const i160_value = h.getResult(const_op, 0);

        const addr_op = c.oraI160ToAddrOpCreate(self.ctx, self.fileLoc(span), i160_value);
        h.appendOp(self.block, addr_op);
        return h.getResult(addr_op, 0);
    }

    if (builtin_info.return_type.isInteger()) {
        if (std.mem.endsWith(u8, builtin_info.full_path, "_MIN")) {
            return lowerIntegerBoundaryConstant(self, builtin_info.return_type, .min, span);
        }
        if (std.mem.endsWith(u8, builtin_info.full_path, "_MAX")) {
            return lowerIntegerBoundaryConstant(self, builtin_info.return_type, .max, span);
        }
    }

    const op = self.ora_dialect.createArithConstant(0, ty, self.fileLoc(span));
    h.appendOp(self.block, op);
    return h.getResult(op, 0);
}

const IntegerBoundary = enum { min, max };

/// Lower integer boundary constants (e.g., U256_MAX, I128_MIN).
fn lowerIntegerBoundaryConstant(
    self: *const ExpressionLowerer,
    ora_type: lib.OraType,
    boundary: IntegerBoundary,
    span: lib.ast.SourceSpan,
) c.MlirValue {
    const loc = self.fileLoc(span);
    const ty = self.type_mapper.toMlirType(.{ .ora_type = ora_type });

    // Unsigned boundaries are cheap: MIN = 0, MAX = all ones.
    if (!ora_type.isSignedInteger()) {
        const value: i64 = switch (boundary) {
            .min => 0,
            .max => -1,
        };
        const op = self.ora_dialect.createArithConstant(value, ty, loc);
        h.appendOp(self.block, op);
        return h.getResult(op, 0);
    }

    const one_op = self.ora_dialect.createArithConstant(1, ty, loc);
    h.appendOp(self.block, one_op);
    const one = h.getResult(one_op, 0);

    // Signed MIN bit pattern: 1 << (bits - 1)
    if (boundary == .min) {
        const width = ora_type.bitWidth() orelse constants.DEFAULT_INTEGER_BITS;
        const shift = self.ora_dialect.createArithConstant(@intCast(width - 1), ty, loc);
        h.appendOp(self.block, shift);
        const min_op = c.oraArithShlIOpCreate(self.ctx, loc, one, h.getResult(shift, 0));
        h.appendOp(self.block, min_op);
        return h.getResult(min_op, 0);
    }

    // Signed MAX bit pattern: logical_shift_right(all_ones, 1)
    const all_ones_op = self.ora_dialect.createArithConstant(-1, ty, loc);
    h.appendOp(self.block, all_ones_op);
    const max_op = c.oraArithShrUIOpCreate(self.ctx, loc, h.getResult(all_ones_op, 0), one);
    h.appendOp(self.block, max_op);
    return h.getResult(max_op, 0);
}

/// Lower builtin "field access" that represents a call (e.g., std.transaction.sender).
fn lowerBuiltinFieldCall(
    self: *const ExpressionLowerer,
    builtin_info: *const lib.semantics.builtins.BuiltinInfo,
    span: lib.ast.SourceSpan,
) c.MlirValue {
    const op_name = std.fmt.allocPrint(std.heap.page_allocator, "ora.evm.{s}", .{builtin_info.evm_opcode}) catch {
        log.err("Failed to allocate opcode name for builtin field call\n", .{});
        return self.createErrorPlaceholder(span, "Failed to create builtin field call");
    };
    defer std.heap.page_allocator.free(op_name);

    const location = self.fileLoc(span);
    const result_type = self.type_mapper.toMlirType(.{
        .ora_type = builtin_info.return_type,
    });

    const op = c.oraEvmOpCreate(
        self.ctx,
        location,
        h.strRef(op_name),
        null,
        0,
        result_type,
    );
    h.appendOp(self.block, op);
    return h.getResult(op, 0);
}

/// Extract a bitfield field via shift+mask: (word >> offset) & mask
/// For signed fields: sign-extend via SHL + SAR
fn createBitfieldFieldExtract(
    self: *const ExpressionLowerer,
    word: c.MlirValue,
    field_name: []const u8,
    type_sym: *const constants.TypeSymbol,
    span: lib.ast.SourceSpan,
) c.MlirValue {
    const loc = self.fileLoc(span);
    const int_ty = c.oraIntegerTypeCreate(self.ctx, constants.DEFAULT_INTEGER_BITS);

    // Find the field in the type symbol
    if (type_sym.fields) |fields| {
        for (fields) |field| {
            if (!std.mem.eql(u8, field.name, field_name)) continue;

            const offset: i64 = if (field.offset) |o| @intCast(o) else 0;
            const width: u32 = field.bit_width orelse 256;

            // Step 1: SHR by offset → shifted = word >> offset
            const offset_const_op = self.ora_dialect.createArithConstant(offset, int_ty, loc);
            h.appendOp(self.block, offset_const_op);
            const offset_val = h.getResult(offset_const_op, 0);

            const shr_op = c.oraArithShrUIOpCreate(self.ctx, loc, word, offset_val);
            h.appendOp(self.block, shr_op);
            const shifted = h.getResult(shr_op, 0);

            // Step 2: AND with mask → raw = shifted & ((1 << width) - 1)
            const mask: i64 = if (width >= 64) -1 else (@as(i64, 1) << @intCast(width)) - 1;
            const mask_const_op = self.ora_dialect.createArithConstant(mask, int_ty, loc);
            h.appendOp(self.block, mask_const_op);
            const mask_val = h.getResult(mask_const_op, 0);

            const and_op = c.oraArithAndIOpCreate(self.ctx, loc, shifted, mask_val);
            h.appendOp(self.block, and_op);
            const raw = h.getResult(and_op, 0);

            // Step 3: For signed fields, sign-extend via SHL+SAR
            const is_signed = if (field.ora_type_info.ora_type) |ot| switch (ot) {
                .i8, .i16, .i32, .i64, .i128, .i256 => true,
                else => false,
            } else false;

            if (is_signed and width < 256) {
                const shift_amt: i64 = 256 - @as(i64, width);
                const shift_const_op = self.ora_dialect.createArithConstant(shift_amt, int_ty, loc);
                h.appendOp(self.block, shift_const_op);
                const shift_val = h.getResult(shift_const_op, 0);

                // SHL to put sign bit at bit 255
                const shl_op = c.oraArithShlIOpCreate(self.ctx, loc, raw, shift_val);
                h.appendOp(self.block, shl_op);
                const shl_result = h.getResult(shl_op, 0);

                // SAR to sign-extend back
                const sar_op = c.oraArithShrSIOpCreate(self.ctx, loc, shl_result, shift_val);
                h.appendOp(self.block, sar_op);
                const result = h.getResult(sar_op, 0);

                // Emit signed bound assumption: -(2^(width-1)) <= result < 2^(width-1)
                emitBitfieldBoundAssume(self, result, width, true, int_ty, loc);
                return result;
            }

            // Emit unsigned bound assumption: 0 <= result <= mask
            emitBitfieldBoundAssume(self, raw, width, false, int_ty, loc);
            return raw;
        }
    }

    // Field not found in bitfield, fall back to error placeholder
    log.debug("ERROR: Bitfield field '{s}' not found in type '{s}'\n", .{ field_name, type_sym.name });
    return self.createErrorPlaceholder(span, "bitfield field not found");
}

/// Create struct field extract operation
pub fn createStructFieldExtract(
    self: *const ExpressionLowerer,
    struct_val: c.MlirValue,
    field_name: []const u8,
    span: lib.ast.SourceSpan,
) c.MlirValue {
    var result_ty = c.oraIntegerTypeCreate(self.ctx, constants.DEFAULT_INTEGER_BITS);

    if (self.symbol_table) |st| {
        // Check named struct types
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
        // Check anonymous structs from type mapper (e.g. overflow builtins, tuples)
        // Match by struct type to avoid ambiguity when multiple anon structs share field names.
        const default_ty = c.oraIntegerTypeCreate(self.ctx, constants.DEFAULT_INTEGER_BITS);
        if (c.oraTypeEqual(result_ty, default_ty)) {
            const struct_val_type = c.oraValueGetType(struct_val);
            var anon_iter = self.type_mapper.iterAnonymousStructs();
            while (anon_iter.next()) |anon_entry| {
                const anon_ty = c.oraStructTypeGet(self.ctx, h.strRef(anon_entry.key_ptr.*));
                if (!c.oraTypeEqual(struct_val_type, anon_ty)) continue;
                for (anon_entry.value_ptr.*) |field| {
                    if (std.mem.eql(u8, field.name, field_name)) {
                        const field_type_info = lib.ast.Types.TypeInfo{
                            .category = field.typ.*.getCategory(),
                            .ora_type = field.typ.*,
                            .source = .inferred,
                            .span = null,
                        };
                        result_ty = self.type_mapper.toMlirType(field_type_info);
                        break;
                    }
                }
                break; // found matching struct, stop
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
    const target_type = c.oraValueGetType(target);
    const index_ty = c.oraIndexTypeCreate(self.ctx);

    if (c.oraTypeIsAMemRef(target_type) or c.oraTypeIsAShaped(target_type)) {
        if (c.oraTypeIsAShaped(target_type) and c.oraShapedTypeHasStaticShape(target_type)) {
            const num_dims = c.oraShapedTypeGetRank(target_type);
            if (num_dims > 0) {
                const dim_size = c.oraShapedTypeGetDimSize(target_type, 0);

                if (dim_size >= 0) {
                    const len_const = self.createConstant(@intCast(dim_size), span);
                    return convertIndexToIndexType(self, len_const, span);
                }
            }
        }

        if (c.oraTypeIsAMemRef(target_type)) {
            const dim_const_op = self.ora_dialect.createArithConstant(0, index_ty, self.fileLoc(span));
            h.appendOp(self.block, dim_const_op);
            const dim_index = h.getResult(dim_const_op, 0);

            const op = c.oraMemrefDimOpCreate(self.ctx, self.fileLoc(span), target, dim_index);
            h.appendOp(self.block, op);
            return h.getResult(op, 0);
        }

        const dim_const_op = self.ora_dialect.createArithConstant(0, index_ty, self.fileLoc(span));
        h.appendOp(self.block, dim_const_op);
        const dim_index = h.getResult(dim_const_op, 0);

        const op = c.oraTensorDimOpCreate(self.ctx, self.fileLoc(span), target, dim_index);
        h.appendOp(self.block, op);
        return h.getResult(op, 0);
    } else {
        const op = c.oraLengthOpCreate(self.ctx, self.fileLoc(span), target, index_ty);
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
    const index_type = c.oraValueGetType(index);
    const mlir_index_type = c.oraIndexTypeCreate(self.ctx);

    if (c.oraTypeEqual(index_type, mlir_index_type)) {
        return index;
    }

    var current_value = expr_helpers.unwrapRefinementValue(
        self.ctx,
        self.block,
        self.locations,
        self.refinement_base_cache,
        index,
        span,
    );
    const current_type = c.oraValueGetType(current_value);

    if (c.oraTypeIsAInteger(current_type)) {
        const width = c.oraIntegerTypeGetWidth(current_type);
        const signless_type = c.oraIntegerTypeCreate(self.ctx, width);
        if (!c.oraTypeEqual(current_type, signless_type)) {
            const signless_cast_op = c.oraArithBitcastOpCreate(self.ctx, self.fileLoc(span), current_value, signless_type);
            h.appendOp(self.block, signless_cast_op);
            current_value = h.getResult(signless_cast_op, 0);
        }
    }

    const index_cast_op = c.oraArithIndexCastUIOpCreate(self.ctx, self.fileLoc(span), current_value, mlir_index_type);
    h.appendOp(self.block, index_cast_op);
    return h.getResult(index_cast_op, 0);
}

/// Convert index value to default integer type (u256)
pub fn convertIndexToIntegerType(
    self: *const ExpressionLowerer,
    index: c.MlirValue,
    span: lib.ast.SourceSpan,
) c.MlirValue {
    const int_type = c.oraIntegerTypeCreate(self.ctx, constants.DEFAULT_INTEGER_BITS);
    const index_type = c.oraValueGetType(index);

    if (c.oraTypeEqual(index_type, int_type)) {
        return index;
    }

    var current_value = expr_helpers.unwrapRefinementValue(
        self.ctx,
        self.block,
        self.locations,
        self.refinement_base_cache,
        index,
        span,
    );
    const current_type = c.oraValueGetType(current_value);

    if (c.oraTypeIsAInteger(current_type)) {
        const width = c.oraIntegerTypeGetWidth(current_type);
        const signless_type = c.oraIntegerTypeCreate(self.ctx, width);
        if (!c.oraTypeEqual(current_type, signless_type)) {
            const signless_cast_op = c.oraArithBitcastOpCreate(self.ctx, self.fileLoc(span), current_value, signless_type);
            h.appendOp(self.block, signless_cast_op);
            current_value = h.getResult(signless_cast_op, 0);
        }
    }

    const cast_op = c.oraArithIndexCastUIOpCreate(self.ctx, self.fileLoc(span), current_value, int_type);
    h.appendOp(self.block, cast_op);
    return h.getResult(cast_op, 0);
}

/// Create array index load with bounds checking
pub fn createArrayIndexLoad(
    self: *const ExpressionLowerer,
    array: c.MlirValue,
    index: c.MlirValue,
    span: lib.ast.SourceSpan,
) c.MlirValue {
    const array_type = c.oraValueGetType(array);
    const index_index = convertIndexToIndexType(self, index, span);
    const enable_bounds_check = true;

    if (enable_bounds_check and (c.oraTypeIsAMemRef(array_type) or c.oraTypeIsAShaped(array_type))) {
        const array_length = createLengthAccess(self, array, span);

        const predicate_value = expr_helpers.predicateStringToInt("ult");
        const cmp_op = c.oraArithCmpIOpCreate(self.ctx, self.fileLoc(span), predicate_value, index_index, array_length);
        h.appendOp(self.block, cmp_op);
        const in_bounds = h.getResult(cmp_op, 0);

        const assert_op = self.ora_dialect.createCfAssert(in_bounds, "array index out of bounds", self.fileLoc(span));
        h.appendOp(self.block, assert_op);
    }

    const element_type = if (c.oraTypeIsAMemRef(array_type) or c.oraTypeIsAShaped(array_type))
        c.oraShapedTypeGetElementType(array_type)
    else
        c.oraIntegerTypeCreate(self.ctx, constants.DEFAULT_INTEGER_BITS);

    if (c.oraTypeIsAMemRef(array_type)) {
        const load_op = c.oraMemrefLoadOpCreate(
            self.ctx,
            self.fileLoc(span),
            array,
            &[_]c.MlirValue{index_index},
            1,
            element_type,
        );
        h.appendOp(self.block, load_op);
        return h.getResult(load_op, 0);
    } else if (c.oraTypeIsAShaped(array_type)) {
        const extract_op = c.oraTensorExtractOpCreate(
            self.ctx,
            self.fileLoc(span),
            array,
            &[_]c.MlirValue{index_index},
            1,
            element_type,
        );
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
    const map_type_check = c.oraValueGetType(map);
    if (c.oraTypeIsAShaped(map_type_check)) {
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
        const map_type = c.oraValueGetType(map);
        const extracted_value_type = c.oraMapTypeGetValueType(map_type);
        if (extracted_value_type.ptr != null) {
            log.debug("[createMapIndexLoad] Extracted value type from map\n", .{});
            result_ty = extracted_value_type;
        } else {
            log.debug("WARNING: createMapIndexLoad: Could not extract value type from map, defaulting to i256.\n", .{});
            result_ty = c.oraIntegerTypeCreate(self.ctx, constants.DEFAULT_INTEGER_BITS);
        }
    }
    log.debug("[createMapIndexLoad] Creating ora.map_get with result_type: {any}\n", .{result_ty});

    // verify map type's value type matches our expected result_type
    const map_type_verify = c.oraValueGetType(map);
    const map_value_type = c.oraMapTypeGetValueType(map_type_verify);
    if (map_value_type.ptr != null) {
        log.debug("[createMapIndexLoad] Map type's value type: {any}\n", .{map_value_type});
        if (!c.oraTypeEqual(map_value_type, result_ty)) {
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
    const actual_result_type = c.oraValueGetType(result);
    log.debug("[createMapIndexLoad] ora.map_get created, actual result type: {any}\n", .{actual_result_type});

    // check if map_get returned the wrong type
    if (!c.oraTypeEqual(actual_result_type, result_ty)) {
        log.debug("[createMapIndexLoad] ERROR: map_get returned {any} but we requested {any}\n", .{ actual_result_type, result_ty });
        log.debug("[createMapIndexLoad] This is a bug - ora.map_get should respect the result_type parameter\n", .{});
        // for now, return the result as-is - the type system will catch this error later
    }

    return result;
}

/// Emit an `ora.assume` with bitfield field bounds for SMT verification.
/// Unsigned: result <= (1 << width) - 1
/// Signed: -(1 << (width-1)) <= result <= (1 << (width-1)) - 1
fn emitBitfieldBoundAssume(
    self: *const ExpressionLowerer,
    result: c.MlirValue,
    width: u32,
    is_signed: bool,
    int_ty: c.MlirType,
    loc: c.MlirLocation,
) void {
    if (width >= 256) return; // full-width, no useful bound

    if (!is_signed) {
        // result <= mask (unsigned upper bound)
        const mask: i64 = (@as(i64, 1) << @intCast(width)) - 1;
        const mask_op = self.ora_dialect.createArithConstant(mask, int_ty, loc);
        h.appendOp(self.block, mask_op);
        const cmp = c.oraArithCmpIOpCreate(self.ctx, loc, 7, result, h.getResult(mask_op, 0)); // 7 = ule
        h.appendOp(self.block, cmp);
        const assume_op = self.ora_dialect.createAssume(h.getResult(cmp, 0), loc);
        h.appendOp(self.block, assume_op);
    } else {
        // result >= -(1 << (width-1))  AND  result <= (1 << (width-1)) - 1
        const half: i64 = @as(i64, 1) << @intCast(width - 1);
        // lower bound: result >= -half (signed)
        const lo_op = self.ora_dialect.createArithConstant(-half, int_ty, loc);
        h.appendOp(self.block, lo_op);
        const cmp_lo = c.oraArithCmpIOpCreate(self.ctx, loc, 5, result, h.getResult(lo_op, 0)); // 5 = sge
        h.appendOp(self.block, cmp_lo);
        const assume_lo = self.ora_dialect.createAssume(h.getResult(cmp_lo, 0), loc);
        h.appendOp(self.block, assume_lo);

        // upper bound: result <= half - 1 (signed)
        const hi_op = self.ora_dialect.createArithConstant(half - 1, int_ty, loc);
        h.appendOp(self.block, hi_op);
        const cmp_hi = c.oraArithCmpIOpCreate(self.ctx, loc, 3, result, h.getResult(hi_op, 0)); // 3 = sle
        h.appendOp(self.block, cmp_hi);
        const assume_hi = self.ora_dialect.createAssume(h.getResult(cmp_hi, 0), loc);
        h.appendOp(self.block, assume_hi);
    }
}
