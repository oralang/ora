// ============================================================================
// Declaration Lowering - Functions
// ============================================================================

const std = @import("std");
const c = @import("mlir_c_api").c;
const lib = @import("ora_lib");
const h = @import("../helpers.zig");
const TypeMapper = @import("../types.zig").TypeMapper;
const LocalVarMap = @import("../symbols.zig").LocalVarMap;
const local_var_analysis = @import("../analysis/local_vars.zig");
const ParamMap = @import("../symbols.zig").ParamMap;
const StorageMap = @import("../memory.zig").StorageMap;
const ExpressionLowerer = @import("../expressions.zig").ExpressionLowerer;
const StatementLowerer = @import("../statements.zig").StatementLowerer;
const LoweringError = StatementLowerer.LoweringError;
const DeclarationLowerer = @import("mod.zig").DeclarationLowerer;
const helpers = @import("helpers.zig");
const refinements = @import("refinements.zig");
const expr_helpers = @import("../expressions/helpers.zig");
const log = @import("log");
const crypto = std.crypto;

fn canonicalAbiType(allocator: std.mem.Allocator, ora_type: lib.ast.Types.TypeInfo) ![]const u8 {
    const ot = ora_type.ora_type orelse return error.InvalidType;
    return switch (ot) {
        .u8 => allocator.dupe(u8, "uint8"),
        .u16 => allocator.dupe(u8, "uint16"),
        .u32 => allocator.dupe(u8, "uint32"),
        .u64 => allocator.dupe(u8, "uint64"),
        .u128 => allocator.dupe(u8, "uint128"),
        .u256 => allocator.dupe(u8, "uint256"),
        .i8 => allocator.dupe(u8, "int8"),
        .i16 => allocator.dupe(u8, "int16"),
        .i32 => allocator.dupe(u8, "int32"),
        .i64 => allocator.dupe(u8, "int64"),
        .i128 => allocator.dupe(u8, "int128"),
        .i256 => allocator.dupe(u8, "int256"),
        .bool => allocator.dupe(u8, "bool"),
        .address => allocator.dupe(u8, "address"),
        .bytes => allocator.dupe(u8, "bytes"),
        .string => allocator.dupe(u8, "string"),
        .array => |arr| {
            const elem_info = lib.ast.Types.TypeInfo.fromOraType(arr.elem.*);
            const elem = try canonicalAbiType(allocator, elem_info);
            defer allocator.free(elem);
            return std.fmt.allocPrint(allocator, "{s}[{d}]", .{ elem, arr.len });
        },
        .slice => |elem| {
            const elem_info = lib.ast.Types.TypeInfo.fromOraType(elem.*);
            const elem_str = try canonicalAbiType(allocator, elem_info);
            defer allocator.free(elem_str);
            return std.fmt.allocPrint(allocator, "{s}[]", .{elem_str});
        },
        .tuple => |members| {
            var parts = std.ArrayList([]const u8){};
            defer parts.deinit(allocator);
            for (members) |m| {
                const mi = lib.ast.Types.TypeInfo.fromOraType(m);
                const ms = try canonicalAbiType(allocator, mi);
                try parts.append(allocator, ms);
            }
            defer {
                for (parts.items) |p| allocator.free(p);
            }
            const joined = try std.mem.join(allocator, ",", parts.items);
            defer allocator.free(joined);
            return std.fmt.allocPrint(allocator, "({s})", .{joined});
        },
        .error_union => |succ| {
            const si = lib.ast.Types.TypeInfo.fromOraType(succ.*);
            return canonicalAbiType(allocator, si);
        },
        ._union => |members| {
            if (members.len > 0 and members[0] == .error_union) {
                const si = lib.ast.Types.TypeInfo.fromOraType(members[0].error_union.*);
                return canonicalAbiType(allocator, si);
            }
            return error.InvalidType;
        },
        .min_value => |mv| {
            const base_info = lib.ast.Types.TypeInfo.fromOraType(mv.base.*);
            return canonicalAbiType(allocator, base_info);
        },
        .max_value => |mv| {
            const base_info = lib.ast.Types.TypeInfo.fromOraType(mv.base.*);
            return canonicalAbiType(allocator, base_info);
        },
        .in_range => |ir| {
            const base_info = lib.ast.Types.TypeInfo.fromOraType(ir.base.*);
            return canonicalAbiType(allocator, base_info);
        },
        .scaled => |s| {
            const base_info = lib.ast.Types.TypeInfo.fromOraType(s.base.*);
            return canonicalAbiType(allocator, base_info);
        },
        .exact => |e| {
            const base_info = lib.ast.Types.TypeInfo.fromOraType(e.*);
            return canonicalAbiType(allocator, base_info);
        },
        .non_zero_address => allocator.dupe(u8, "address"),
        else => error.InvalidType,
    };
}

fn keccakSelectorHex(allocator: std.mem.Allocator, signature: []const u8) ![]const u8 {
    var hash: [32]u8 = undefined;
    crypto.hash.sha3.Keccak256.hash(signature, &hash, .{});
    const selector = hash[0..4];
    var buf: [8]u8 = undefined;
    var i: usize = 0;
    while (i < selector.len) : (i += 1) {
        const byte = selector[i];
        const hi = std.fmt.hex_charset[byte >> 4];
        const lo = std.fmt.hex_charset[byte & 0x0f];
        buf[i * 2] = hi;
        buf[i * 2 + 1] = lo;
    }
    return std.fmt.allocPrint(allocator, "0x{s}", .{buf[0..]});
}

fn refinementCacheKey(value: c.MlirValue) usize {
    if (c.mlirValueIsABlockArgument(value)) {
        const owner = c.mlirBlockArgumentGetOwner(value);
        const arg_no = c.mlirBlockArgumentGetArgNumber(value);
        const block_key = @intFromPtr(owner.ptr);
        const arg_key: usize = @intCast(arg_no);
        return block_key ^ (arg_key *% 0x9e3779b97f4a7c15);
    }
    return @intFromPtr(value.ptr);
}

/// Lower function declarations with enhanced features
pub fn lowerFunction(self: *const DeclarationLowerer, func: *const lib.FunctionNode, contract_storage_map: ?*StorageMap, local_var_map: ?*LocalVarMap) c.MlirOperation {
    // create a local variable map for this function
    var local_vars = LocalVarMap.init(std.heap.page_allocator);
    defer local_vars.deinit();

    // create parameter mapping for calldata parameters
    var param_map = ParamMap.init(std.heap.page_allocator);
    defer param_map.deinit();

    // collect parameter types for MLIR block arguments
    var param_types_buf: [16]c.MlirType = undefined; // Support up to 16 parameters
    const param_types = if (func.parameters.len <= 16) param_types_buf[0..func.parameters.len] else blk: {
        break :blk std.heap.page_allocator.alloc(c.MlirType, func.parameters.len) catch {
            log.err("Failed to allocate parameter types\n", .{});
            @panic("Allocation failure");
        };
    };
    defer if (func.parameters.len > 16) std.heap.page_allocator.free(param_types);

    for (func.parameters, 0..) |param, i| {
        // function parameters are calldata by default in Ora
        param_map.addParam(param.name, i) catch {};

        // get MLIR type for parameter
        const param_type = self.type_mapper.toMlirType(param.type_info);
        param_types[i] = param_type;
    }

    const func_loc = helpers.createFileLocation(self, func.span);

    // add function name
    const name_ref = c.oraStringRefCreate(func.name.ptr, func.name.len);
    const name_attr = c.oraStringAttrCreate(self.ctx, name_ref);
    const sym_name_id = h.identifier(self.ctx, "sym_name");

    // collect all function attributes
    var attributes = std.ArrayList(c.MlirNamedAttribute){};
    defer attributes.deinit(std.heap.page_allocator);

    // add function name attribute
    attributes.append(std.heap.page_allocator, c.oraNamedAttributeGet(sym_name_id, name_attr)) catch {};

    // add visibility modifier attribute
    const visibility_attr = switch (func.visibility) {
        .Public => c.oraStringAttrCreate(self.ctx, h.strRef("pub")),
        .Private => c.oraStringAttrCreate(self.ctx, h.strRef("private")),
    };
    const visibility_id = h.identifier(self.ctx, "ora.visibility");
    attributes.append(std.heap.page_allocator, c.oraNamedAttributeGet(visibility_id, visibility_attr)) catch {};

    // Compute and attach selector for public functions (Solidity-style ABI)
    if (func.visibility == .Public) {
        var parts = std.ArrayList([]const u8){};
        defer parts.deinit(std.heap.page_allocator);
        var abi_param_attrs = std.ArrayList(c.MlirAttribute){};
        defer abi_param_attrs.deinit(std.heap.page_allocator);
        const selector_ok = true;
        for (func.parameters) |param| {
            const s = canonicalAbiType(std.heap.page_allocator, param.type_info) catch |err| {
                log.err("ABI type unsupported for selector generation in {s}: {s}\n", .{ func.name, @errorName(err) });
                if (self.error_handler) |eh| {
                    eh.reportError(.MlirOperationFailed, func.span, "ABI type unsupported for selector generation", @errorName(err)) catch {};
                }
                return c.MlirOperation{ .ptr = null };
            };
            parts.append(std.heap.page_allocator, s) catch {};
            const s_attr = c.oraStringAttrCreate(self.ctx, h.strRef(s));
            abi_param_attrs.append(std.heap.page_allocator, s_attr) catch {};
        }
        defer {
            for (parts.items) |p| std.heap.page_allocator.free(p);
        }
        if (selector_ok) {
                const joined = std.mem.join(std.heap.page_allocator, ",", parts.items) catch |err| {
                    log.err("Failed to build ABI signature for {s}: {s}\n", .{ func.name, @errorName(err) });
                    if (self.error_handler) |eh| {
                        eh.reportError(.MlirOperationFailed, func.span, "Failed to build ABI signature", @errorName(err)) catch {};
                    }
                    return c.MlirOperation{ .ptr = null };
                };
                defer std.heap.page_allocator.free(joined);

                const sig = std.fmt.allocPrint(std.heap.page_allocator, "{s}({s})", .{ func.name, joined }) catch |err| {
                    log.err("Failed to build ABI signature for {s}: {s}\n", .{ func.name, @errorName(err) });
                    if (self.error_handler) |eh| {
                        eh.reportError(.MlirOperationFailed, func.span, "Failed to build ABI signature", @errorName(err)) catch {};
                    }
                    return c.MlirOperation{ .ptr = null };
                };
                defer std.heap.page_allocator.free(sig);

                const selector = keccakSelectorHex(std.heap.page_allocator, sig) catch |err| {
                    log.err("Failed to compute selector for {s}: {s}\n", .{ func.name, @errorName(err) });
                    if (self.error_handler) |eh| {
                        eh.reportError(.MlirOperationFailed, func.span, "Failed to compute selector", @errorName(err)) catch {};
                    }
                    return c.MlirOperation{ .ptr = null };
                };
                defer std.heap.page_allocator.free(selector);

                const sel_attr = c.oraStringAttrCreate(self.ctx, h.strRef(selector));
                const sel_id = h.identifier(self.ctx, "ora.selector");
                attributes.append(std.heap.page_allocator, c.oraNamedAttributeGet(sel_id, sel_attr)) catch {};
        }
        if (!selector_ok) {
            if (self.error_handler) |eh| {
                eh.reportError(.MlirOperationFailed, func.span, "Failed to compute ABI selector", null) catch {};
            }
            return c.MlirOperation{ .ptr = null };
        }

        // Attach ABI param type list for dispatcher decoding.
        const abi_params_id = h.identifier(self.ctx, "ora.abi_params");
        const abi_params_attr = c.oraArrayAttrCreate(self.ctx, @intCast(abi_param_attrs.items.len), abi_param_attrs.items.ptr);
        attributes.append(std.heap.page_allocator, c.oraNamedAttributeGet(abi_params_id, abi_params_attr)) catch {};

        // Attach ABI return type if present.
        if (func.return_type_info) |ret_info| {
            const ret_str = canonicalAbiType(std.heap.page_allocator, ret_info) catch |err| {
                log.err("ABI return type unsupported in {s}: {s}\n", .{ func.name, @errorName(err) });
                if (self.error_handler) |eh| {
                    eh.reportError(.MlirOperationFailed, func.span, "ABI return type unsupported", @errorName(err)) catch {};
                }
                return c.MlirOperation{ .ptr = null };
            };
            defer std.heap.page_allocator.free(ret_str);
            if (ret_str.len > 0) {
                const ret_attr = c.oraStringAttrCreate(self.ctx, h.strRef(ret_str));
                const ret_id = h.identifier(self.ctx, "ora.abi_return");
                attributes.append(std.heap.page_allocator, c.oraNamedAttributeGet(ret_id, ret_attr)) catch {};
            }
        }
    }

    // add function effect metadata (pure/writes + slot list)
    if (self.symbol_table) |table| {
        if (table.function_effects.get(func.name)) |eff| {
            switch (eff) {
                .Pure => {
                    const effect_attr = c.oraStringAttrCreate(self.ctx, h.strRef("pure"));
                    const effect_id = h.identifier(self.ctx, "ora.effect");
                    attributes.append(std.heap.page_allocator, c.oraNamedAttributeGet(effect_id, effect_attr)) catch {};
                },
                .Reads => |slots| {
                    const effect_attr = c.oraStringAttrCreate(self.ctx, h.strRef("reads"));
                    const effect_id = h.identifier(self.ctx, "ora.effect");
                    attributes.append(std.heap.page_allocator, c.oraNamedAttributeGet(effect_id, effect_attr)) catch {};

                    var slot_attrs = std.ArrayList(c.MlirAttribute){};
                    defer slot_attrs.deinit(std.heap.page_allocator);
                    for (slots.items) |slot| {
                        const slot_attr = c.oraStringAttrCreate(self.ctx, h.strRef(slot));
                        slot_attrs.append(std.heap.page_allocator, slot_attr) catch {};
                    }
                    const slots_attr = c.oraArrayAttrCreate(self.ctx, @intCast(slot_attrs.items.len), slot_attrs.items.ptr);
                    const slots_id = h.identifier(self.ctx, "ora.read_slots");
                    attributes.append(std.heap.page_allocator, c.oraNamedAttributeGet(slots_id, slots_attr)) catch {};
                },
                .Writes => |slots| {
                    const effect_attr = c.oraStringAttrCreate(self.ctx, h.strRef("writes"));
                    const effect_id = h.identifier(self.ctx, "ora.effect");
                    attributes.append(std.heap.page_allocator, c.oraNamedAttributeGet(effect_id, effect_attr)) catch {};

                    var slot_attrs = std.ArrayList(c.MlirAttribute){};
                    defer slot_attrs.deinit(std.heap.page_allocator);
                    for (slots.items) |slot| {
                        const slot_attr = c.oraStringAttrCreate(self.ctx, h.strRef(slot));
                        slot_attrs.append(std.heap.page_allocator, slot_attr) catch {};
                    }
                    const slots_attr = c.oraArrayAttrCreate(self.ctx, @intCast(slot_attrs.items.len), slot_attrs.items.ptr);
                    const slots_id = h.identifier(self.ctx, "ora.write_slots");
                    attributes.append(std.heap.page_allocator, c.oraNamedAttributeGet(slots_id, slots_attr)) catch {};
                },
                .ReadsWrites => |rw| {
                    const effect_attr = c.oraStringAttrCreate(self.ctx, h.strRef("readwrites"));
                    const effect_id = h.identifier(self.ctx, "ora.effect");
                    attributes.append(std.heap.page_allocator, c.oraNamedAttributeGet(effect_id, effect_attr)) catch {};

                    var read_attrs = std.ArrayList(c.MlirAttribute){};
                    defer read_attrs.deinit(std.heap.page_allocator);
                    for (rw.reads.items) |slot| {
                        const slot_attr = c.oraStringAttrCreate(self.ctx, h.strRef(slot));
                        read_attrs.append(std.heap.page_allocator, slot_attr) catch {};
                    }
                    const read_slots_attr = c.oraArrayAttrCreate(self.ctx, @intCast(read_attrs.items.len), read_attrs.items.ptr);
                    const read_slots_id = h.identifier(self.ctx, "ora.read_slots");
                    attributes.append(std.heap.page_allocator, c.oraNamedAttributeGet(read_slots_id, read_slots_attr)) catch {};

                    var write_attrs = std.ArrayList(c.MlirAttribute){};
                    defer write_attrs.deinit(std.heap.page_allocator);
                    for (rw.writes.items) |slot| {
                        const slot_attr = c.oraStringAttrCreate(self.ctx, h.strRef(slot));
                        write_attrs.append(std.heap.page_allocator, slot_attr) catch {};
                    }
                    const write_slots_attr = c.oraArrayAttrCreate(self.ctx, @intCast(write_attrs.items.len), write_attrs.items.ptr);
                    const write_slots_id = h.identifier(self.ctx, "ora.write_slots");
                    attributes.append(std.heap.page_allocator, c.oraNamedAttributeGet(write_slots_id, write_slots_attr)) catch {};
                },
            }
        }
    }

    // add special function name attributes
    if (std.mem.eql(u8, func.name, "init")) {
        const init_attr = h.boolAttr(self.ctx, 1);
        const init_id = h.identifier(self.ctx, "ora.init");
        attributes.append(std.heap.page_allocator, c.oraNamedAttributeGet(init_id, init_attr)) catch {};
    }

    // add comprehensive verification metadata for function contracts
    if (func.requires_clauses.len > 0 or func.ensures_clauses.len > 0) {
        // add verification marker for formal verification tools
        const verification_attr = h.boolAttr(self.ctx, 1);
        const verification_id = h.identifier(self.ctx, "ora.verification");
        attributes.append(std.heap.page_allocator, c.oraNamedAttributeGet(verification_id, verification_attr)) catch {};

        // add formal verification marker
        const formal_attr = h.boolAttr(self.ctx, 1);
        const formal_id = h.identifier(self.ctx, "ora.formal");
        attributes.append(std.heap.page_allocator, c.oraNamedAttributeGet(formal_id, formal_attr)) catch {};

        // add verification context attribute
        const context_attr = c.oraStringAttrCreate(self.ctx, h.strRef("function_contract"));
        const context_id = h.identifier(self.ctx, "ora.verification_context");
        attributes.append(std.heap.page_allocator, c.oraNamedAttributeGet(context_id, context_attr)) catch {};

        // add requires clauses count
        if (func.requires_clauses.len > 0) {
            const requires_count_attr = c.oraIntegerAttrCreateI64FromType(c.oraIntegerTypeCreate(self.ctx, 32), @intCast(func.requires_clauses.len));
            const requires_count_id = h.identifier(self.ctx, "ora.requires_count");
            attributes.append(std.heap.page_allocator, c.oraNamedAttributeGet(requires_count_id, requires_count_attr)) catch {};
        }

        // add ensures clauses count
        if (func.ensures_clauses.len > 0) {
            const ensures_count_attr = c.oraIntegerAttrCreateI64FromType(c.oraIntegerTypeCreate(self.ctx, 32), @intCast(func.ensures_clauses.len));
            const ensures_count_id = h.identifier(self.ctx, "ora.ensures_count");
            attributes.append(std.heap.page_allocator, c.oraNamedAttributeGet(ensures_count_id, ensures_count_attr)) catch {};
        }

        // add contract verification level
        const contract_level_attr = c.oraStringAttrCreate(self.ctx, h.strRef("full"));
        const contract_level_id = h.identifier(self.ctx, "ora.contract_level");
        attributes.append(std.heap.page_allocator, c.oraNamedAttributeGet(contract_level_id, contract_level_attr)) catch {};
    }

    // add function type
    const fn_type = helpers.createFunctionType(self, func);
    const fn_type_attr = c.oraTypeAttrCreateFromType(fn_type);
    const fn_type_id = h.identifier(self.ctx, "function_type");
    attributes.append(std.heap.page_allocator, c.oraNamedAttributeGet(fn_type_id, fn_type_attr)) catch {};

    // create locations for block arguments
    var param_locs_buf: [16]c.MlirLocation = undefined;
    const param_locs = if (func.parameters.len <= 16) param_locs_buf[0..func.parameters.len] else blk: {
        break :blk std.heap.page_allocator.alloc(c.MlirLocation, func.parameters.len) catch {
            log.err("Failed to allocate parameter locations\n", .{});
            @panic("Allocation failure");
        };
    };
    defer if (func.parameters.len > 16) std.heap.page_allocator.free(param_locs);

    for (func.parameters, 0..) |param, i| {
        param_locs[i] = helpers.createFileLocation(self, param.span);
    }

    const func_op = c.oraFuncFuncOpCreate(
        self.ctx,
        func_loc,
        attributes.items.ptr,
        attributes.items.len,
        param_types.ptr,
        param_locs.ptr,
        param_types.len,
    );
    if (c.oraOperationIsNull(func_op)) {
        @panic("Failed to create func.func operation");
    }

    // get the function body block
    const block = c.oraFuncOpGetBodyBlock(func_op);
    if (c.oraBlockIsNull(block)) {
        @panic("func.func missing body block");
    }

    // add function to symbol table BEFORE lowering body (so calls within body can look it up)
    if (self.symbol_table) |sym_table| {
        const return_type = if (func.return_type_info) |ret_info|
            self.type_mapper.toMlirType(ret_info)
        else
            c.oraNoneTypeCreate(self.ctx);

        // allocate param_types array for symbol table (it will be owned by FunctionSymbol)
        if (sym_table.allocator.alloc(c.MlirType, param_types.len)) |allocated| {
            @memcpy(allocated, param_types);
            sym_table.addFunction(func.name, func_op, allocated, return_type) catch {
                sym_table.allocator.free(allocated);
                log.warn("Failed to add function {s} to symbol table\n", .{func.name});
            };
        } else |_| {
            log.warn("Failed to allocate parameter types for function {s}\n", .{func.name});
            // continue without adding to symbol table if allocation fails
        }
    }

    // share refinement caches across parameters, preconditions, and body lowering
    var refinement_base_cache = std.AutoHashMap(usize, c.MlirValue).init(std.heap.page_allocator);
    defer refinement_base_cache.deinit();
    var refinement_guard_cache = std.AutoHashMap(u128, void).init(std.heap.page_allocator);
    defer refinement_guard_cache.deinit();

    // map parameter names to block arguments
    for (func.parameters, 0..) |param, i| {
        const block_arg = c.oraBlockGetArgument(block, @intCast(i));
        param_map.setBlockArgument(param.name, block_arg) catch {};

        // insert refinement type guards for parameters
        if (param.type_info.ora_type) |ora_type| {
            refinements.insertRefinementGuard(self, block, block_arg, ora_type, param.span, param.name, &refinement_base_cache, &refinement_guard_cache) catch |err| {
                log.err("inserting refinement guard for parameter {s}: {s}\n", .{ param.name, @errorName(err) });
            };
            const base_value = expr_helpers.unwrapRefinementValue(self.ctx, block, self.locations, &refinement_base_cache, block_arg, param.span);
            param_map.setBaseArgument(param.name, base_value) catch {};
        }
    }

    // analyze locals and mark SSA vs memref choices before lowering body
    if (local_var_map) |lvm| {
        var reprs = local_var_analysis.analyzeLocalVarReprs(std.heap.page_allocator, func);
        defer reprs.deinit();
        var it = reprs.iterator();
        while (it.next()) |entry| {
            lvm.setLocalVarKind(entry.key_ptr.*, entry.value_ptr.*) catch {};
        }
    }

    // add precondition assertions for requires clauses
    if (func.requires_clauses.len > 0) {
        lowerRequiresClauses(self, func.requires_clauses, block, &param_map, contract_storage_map, local_var_map orelse &local_vars, &refinement_base_cache) catch |err| {
            log.err("lowering requires clauses: {s}\n", .{@errorName(err)});
        };
    }

    // lower the function body
    lowerFunctionBody(self, func, func_op, block, &param_map, contract_storage_map, local_var_map orelse &local_vars, &refinement_base_cache, &refinement_guard_cache) catch |err| {
        // format error message based on error type
        const error_message = switch (err) {
            error.InvalidLValue => "Cannot assign to immutable variable",
            error.TypeMismatch => "Type mismatch in function body",
            error.UndefinedSymbol => "Undefined symbol in function body",
            error.InvalidMemoryRegion => "Invalid memory region access",
            error.MalformedExpression => "Malformed expression in function body",
            error.MlirOperationFailed => "MLIR operation failed",
            error.OutOfMemory => "Out of memory during lowering",
            error.InvalidControlFlow => "Invalid control flow",
            error.InvalidSwitch => "Invalid switch statement",
            error.UnsupportedStatement => "Unsupported statement type",
        };

        const suggestion = switch (err) {
            error.InvalidLValue => "Use 'var' instead of 'let' or 'const' for mutable variables",
            error.TypeMismatch => "Check that all types match expected signatures",
            error.UndefinedSymbol => "Ensure all symbols are declared before use",
            error.InvalidMemoryRegion => "Check memory region annotations",
            error.MalformedExpression => "Verify expression syntax",
            error.MlirOperationFailed => "Check function implementation for type errors and unsupported operations",
            error.OutOfMemory => "Reduce memory usage or increase available memory",
            error.InvalidControlFlow => "Check control flow structure",
            error.InvalidSwitch => "Verify switch statement syntax",
            error.UnsupportedStatement => "This statement type is not yet supported",
        };

        // report to error handler if available
        if (self.error_handler) |eh| {
            eh.reportError(.MlirOperationFailed, func.span, error_message, suggestion) catch {};
        }

        // ensure a terminator exists for void functions even on error.
        if (func.return_type_info == null) {
            const return_op = self.ora_dialect.createFuncReturn(helpers.createFileLocation(self, func.span));
            h.appendOp(block, return_op);
        }

        return func_op;
    };

    // ensures clauses are now handled before each return statement (see return.zig)
    // this ensures postconditions are checked at every return point, not just at the end

    // ensure a terminator exists (void return)
    if (func.return_type_info == null) {
        const return_op = self.ora_dialect.createFuncReturn(helpers.createFileLocation(self, func.span));
        h.appendOp(block, return_op);
    }

    // function operation was created earlier (before body lowering) for symbol table registration

    // set ora.type attributes on function arguments and return values
    const ora_type_attr_name = h.strRef("ora.type");

    // set attributes on function arguments
    for (func.parameters, 0..) |param, i| {
        const param_mlir_type = self.type_mapper.toMlirType(param.type_info);
        const type_attr = c.oraTypeAttrCreateFromType(param_mlir_type);
        const success = c.oraFuncSetArgAttr(func_op, @intCast(i), ora_type_attr_name, type_attr);
        if (!success) {
            log.warn("Failed to set ora.type attribute on parameter {s}\n", .{param.name});
        }
    }

    // set attribute on function return value (if present)
    if (func.return_type_info) |ret_info| {
        const ret_mlir_type = self.type_mapper.toMlirType(ret_info);
        const type_attr = c.oraTypeAttrCreateFromType(ret_mlir_type);
        const success = c.oraFuncSetResultAttr(func_op, 0, ora_type_attr_name, type_attr);
        if (!success) {
            log.warn("Failed to set ora.type attribute on return value\n", .{});
        }
    }

    return func_op;
}

/// Lower function body statements
fn lowerFunctionBody(self: *const DeclarationLowerer, func: *const lib.FunctionNode, func_op: c.MlirOperation, block: c.MlirBlock, param_map: *const ParamMap, storage_map: ?*const StorageMap, local_var_map: ?*LocalVarMap, refinement_base_cache: *std.AutoHashMap(usize, c.MlirValue), refinement_guard_cache: *std.AutoHashMap(u128, void)) LoweringError!void {
    // create a statement lowerer for this function
    const const_local_var_map = if (local_var_map) |lvm| @as(*const LocalVarMap, lvm) else null;
    var expr_lowerer = ExpressionLowerer.init(self.ctx, block, self.type_mapper, param_map, storage_map, const_local_var_map, self.symbol_table, self.builtin_registry, self.error_handler, self.locations, self.ora_dialect);
    expr_lowerer.refinement_base_cache = refinement_base_cache;
    expr_lowerer.prefer_refinement_base_cache = true;
    expr_lowerer.prefer_refinement_base_cache = true;
    expr_lowerer.refinement_guard_cache = refinement_guard_cache;

    // get the function's return type
    const function_return_type = if (func.return_type_info) |ret_info|
        self.type_mapper.toMlirType(ret_info)
    else
        null;

    const function_return_type_info = if (func.return_type_info) |ret_info| ret_info else null;
    expr_lowerer.current_function_return_type = function_return_type;
    expr_lowerer.current_function_return_type_info = function_return_type_info;
    var stmt_lowerer = StatementLowerer.init(self.ctx, block, self.type_mapper, &expr_lowerer, param_map, storage_map, local_var_map, self.locations, self.symbol_table, self.builtin_registry, std.heap.page_allocator, refinement_guard_cache, function_return_type, function_return_type_info, self.ora_dialect, func.ensures_clauses);
    stmt_lowerer.current_func_op = func_op;

    // lower the function body
    _ = try stmt_lowerer.lowerBlockBody(func.body, block);
    stmt_lowerer.fixEmptyBlocks();
}

/// Lower requires clauses as precondition assertions with enhanced verification metadata (Requirements 6.4)
fn lowerRequiresClauses(self: *const DeclarationLowerer, requires_clauses: []*lib.ast.Expressions.ExprNode, block: c.MlirBlock, param_map: *const ParamMap, storage_map: ?*const StorageMap, local_var_map: ?*LocalVarMap, refinement_base_cache: *std.AutoHashMap(usize, c.MlirValue)) LoweringError!void {
    const const_local_var_map = if (local_var_map) |lvm| @as(*const LocalVarMap, lvm) else null;
    var expr_lowerer = ExpressionLowerer.init(self.ctx, block, self.type_mapper, param_map, storage_map, const_local_var_map, self.symbol_table, self.builtin_registry, self.error_handler, self.locations, self.ora_dialect);
    expr_lowerer.refinement_base_cache = refinement_base_cache;
    expr_lowerer.prefer_refinement_base_cache = true;
    if (param_map.base_args.count() > 0) {
        var it = param_map.base_args.iterator();
        while (it.next()) |entry| {
            const name = entry.key_ptr.*;
            if (param_map.getBlockArgument(name)) |block_arg| {
                refinement_base_cache.put(refinementCacheKey(block_arg), entry.value_ptr.*) catch {};
            }
        }
    }

    for (requires_clauses, 0..) |clause, i| {
        // lower the requires expression
        const condition_value = expr_lowerer.lowerExpression(clause);

        const clause_loc = helpers.createFileLocation(self, helpers.getExpressionSpan(self, clause));

        // collect verification attributes
        var attributes = std.ArrayList(c.MlirNamedAttribute){};
        defer attributes.deinit(std.heap.page_allocator);

        // add required 'msg' attribute first (cf.assert requires this)
        const msg_text = try std.fmt.allocPrint(std.heap.page_allocator, "Precondition {d} failed", .{i});
        defer std.heap.page_allocator.free(msg_text);
        const msg_attr = h.stringAttr(self.ctx, msg_text);
        const msg_id = h.identifier(self.ctx, "msg");
        attributes.append(std.heap.page_allocator, c.oraNamedAttributeGet(msg_id, msg_attr)) catch {};

        // add ora.requires attribute to mark this as a precondition
        const requires_attr = h.boolAttr(self.ctx, 1);
        const requires_id = h.identifier(self.ctx, "ora.requires");
        attributes.append(std.heap.page_allocator, c.oraNamedAttributeGet(requires_id, requires_attr)) catch {};

        // add verification context attribute
        const context_attr = c.oraStringAttrCreate(self.ctx, h.strRef("function_precondition"));
        const context_id = h.identifier(self.ctx, "ora.verification_context");
        attributes.append(std.heap.page_allocator, c.oraNamedAttributeGet(context_id, context_attr)) catch {};

        // add verification marker for formal verification tools
        const verification_attr = h.boolAttr(self.ctx, 1);
        const verification_id = h.identifier(self.ctx, "ora.verification");
        attributes.append(std.heap.page_allocator, c.oraNamedAttributeGet(verification_id, verification_attr)) catch {};

        // add precondition index for multiple requires clauses
        const index_attr = c.oraIntegerAttrCreateI64FromType(c.oraIntegerTypeCreate(self.ctx, 32), @intCast(i));
        const index_id = h.identifier(self.ctx, "ora.precondition_index");
        attributes.append(std.heap.page_allocator, c.oraNamedAttributeGet(index_id, index_attr)) catch {};

        // add formal verification marker
        const formal_attr = h.boolAttr(self.ctx, 1);
        const formal_id = h.identifier(self.ctx, "ora.formal");
        attributes.append(std.heap.page_allocator, c.oraNamedAttributeGet(formal_id, formal_attr)) catch {};

        const assert_op = self.ora_dialect.createCfAssertWithAttrs(condition_value, attributes.items, clause_loc);
        h.appendOp(block, assert_op);
    }
}

/// Lower ensures clauses as postcondition assertions with enhanced verification metadata (Requirements 6.5)
fn lowerEnsuresClauses(self: *const DeclarationLowerer, ensures_clauses: []*lib.ast.Expressions.ExprNode, block: c.MlirBlock, param_map: *const ParamMap, storage_map: ?*const StorageMap, local_var_map: ?*LocalVarMap) LoweringError!void {
    const const_local_var_map = if (local_var_map) |lvm| @as(*const LocalVarMap, lvm) else null;
    var expr_lowerer = ExpressionLowerer.init(self.ctx, block, self.type_mapper, param_map, storage_map, const_local_var_map, self.symbol_table, self.builtin_registry, self.error_handler, self.locations, self.ora_dialect);
    expr_lowerer.prefer_refinement_base_cache = true;

    for (ensures_clauses, 0..) |clause, i| {
        // lower the ensures expression
        const condition_value = expr_lowerer.lowerExpression(clause);

        const clause_loc = helpers.createFileLocation(self, helpers.getExpressionSpan(self, clause));

        // collect verification attributes
        var attributes = std.ArrayList(c.MlirNamedAttribute){};
        defer attributes.deinit(std.heap.page_allocator);

        // add ora.ensures attribute to mark this as a postcondition
        const ensures_attr = h.boolAttr(self.ctx, 1);
        const ensures_id = h.identifier(self.ctx, "ora.ensures");
        attributes.append(std.heap.page_allocator, c.oraNamedAttributeGet(ensures_id, ensures_attr)) catch {};

        // add verification context attribute
        const context_attr = c.oraStringAttrCreate(self.ctx, h.strRef("function_postcondition"));
        const context_id = h.identifier(self.ctx, "ora.verification_context");
        attributes.append(std.heap.page_allocator, c.oraNamedAttributeGet(context_id, context_attr)) catch {};

        // add verification marker for formal verification tools
        const verification_attr = h.boolAttr(self.ctx, 1);
        const verification_id = h.identifier(self.ctx, "ora.verification");
        attributes.append(std.heap.page_allocator, c.oraNamedAttributeGet(verification_id, verification_attr)) catch {};

        // add postcondition index for multiple ensures clauses
        const index_attr = c.oraIntegerAttrCreateI64FromType(c.oraIntegerTypeCreate(self.ctx, 32), @intCast(i));
        const index_id = h.identifier(self.ctx, "ora.postcondition_index");
        attributes.append(std.heap.page_allocator, c.oraNamedAttributeGet(index_id, index_attr)) catch {};

        // add formal verification marker
        const formal_attr = h.boolAttr(self.ctx, 1);
        const formal_id = h.identifier(self.ctx, "ora.formal");
        attributes.append(std.heap.page_allocator, c.oraNamedAttributeGet(formal_id, formal_attr)) catch {};

        // add return value reference for postconditions
        const return_ref_attr = c.oraStringAttrCreate(self.ctx, h.strRef("return_value"));
        const return_ref_id = h.identifier(self.ctx, "ora.return_reference");
        attributes.append(std.heap.page_allocator, c.oraNamedAttributeGet(return_ref_id, return_ref_attr)) catch {};

        const assert_op = self.ora_dialect.createCfAssertWithAttrs(condition_value, attributes.items, clause_loc);
        h.appendOp(block, assert_op);
    }
}
