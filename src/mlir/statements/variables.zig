// ============================================================================
// Variable Declaration Statement Lowering
// ============================================================================
// Variable declarations: var, let, const for different memory regions

const std = @import("std");
const c = @import("mlir_c_api").c;
const lib = @import("ora_lib");
const h = @import("../helpers.zig");
const StatementLowerer = @import("statement_lowerer.zig").StatementLowerer;
const LoweringError = StatementLowerer.LoweringError;
const helpers = @import("helpers.zig");
const log = @import("log");

/// Lower variable declarations with region-specific handling
pub fn lowerVariableDecl(self: *const StatementLowerer, var_decl: *const lib.ast.Statements.VariableDeclNode) LoweringError!void {
    const loc = self.fileLoc(var_decl.span);

    if (var_decl.tuple_names) |names| {
        if (var_decl.region != .Stack) {
            return LoweringError.InvalidMemoryRegion;
        }
        const tuple_ora = var_decl.type_info.ora_type orelse return LoweringError.TypeMismatch;
        if (tuple_ora != .tuple) {
            return LoweringError.TypeMismatch;
        }
        if (var_decl.value == null) {
            return LoweringError.MalformedExpression;
        }
        if (tuple_ora.tuple.len != names.len) {
            return LoweringError.TypeMismatch;
        }

        const init_value = helpers.lowerValueWithImplicitTry(@constCast(self), &var_decl.value.?.*, var_decl.type_info);

        for (names, 0..) |name, idx| {
            const elem_ora = tuple_ora.tuple[idx];
            var elem_type_info = lib.ast.type_info.TypeInfo.fromOraType(elem_ora);
            elem_type_info.region = var_decl.region;
            const elem_mlir_type = self.type_mapper.toMlirType(elem_type_info);

            var field_name_buf: [16]u8 = undefined;
            const field_name = std.fmt.bufPrint(&field_name_buf, "{d}", .{idx}) catch "0";
            var element_value = self.expr_lowerer.createStructFieldExtract(init_value, field_name, var_decl.span);
            element_value = self.expr_lowerer.convertToType(element_value, elem_mlir_type, var_decl.span);

            const is_aggregate = isAggregateType(elem_ora);
            const is_scalar = isScalarValueType(elem_ora);
            const prefer_ssa = if (self.local_var_map) |lvm|
                if (lvm.getLocalVarKind(name)) |kind| kind == .SSA else false
            else
                false;
            const needs_memref = self.force_stack_memref or ((var_decl.kind == .Var) and (is_aggregate or (is_scalar and !prefer_ssa)));

            if (needs_memref) {
                const rank: u32 = if (elem_ora == .array) 1 else 0;
                var shape_buf: [1]i64 = .{0};
                const shape: ?*const i64 = if (elem_ora == .array) blk: {
                    shape_buf[0] = @intCast(elem_ora.array.len);
                    break :blk &shape_buf[0];
                } else null;
                const memref_type = h.memRefType(self.ctx, elem_mlir_type, rank, shape, h.nullAttr(), h.nullAttr());
                const alloca_op = self.ora_dialect.createMemrefAlloca(memref_type, loc);
                h.appendOp(self.block, alloca_op);
                const memref = h.getResult(alloca_op, 0);

                var store_value = element_value;
                store_value = try helpers.insertRefinementGuard(self, store_value, elem_ora, var_decl.span, name, false);
                const element_type = c.oraShapedTypeGetElementType(memref_type);
                store_value = self.expr_lowerer.convertToType(store_value, element_type, var_decl.span);
                const store_op = self.ora_dialect.createMemrefStore(store_value, memref, &[_]c.MlirValue{}, loc);
                h.appendOp(self.block, store_op);

                if (self.local_var_map) |lvm| {
                    lvm.addLocalVar(name, memref) catch return LoweringError.OutOfMemory;
                }
                if (self.symbol_table) |st| {
                    st.addSymbol(name, memref_type, var_decl.region, null, var_decl.kind) catch {
                        return LoweringError.OutOfMemory;
                    };
                    st.updateSymbolValue(name, memref) catch {};
                }
            } else {
                element_value = try helpers.insertRefinementGuard(self, element_value, elem_ora, var_decl.span, name, false);
                if (self.local_var_map) |lvm| {
                    lvm.addLocalVar(name, element_value) catch return LoweringError.OutOfMemory;
                }
                if (self.symbol_table) |st| {
                    st.addSymbol(name, elem_mlir_type, var_decl.region, null, var_decl.kind) catch {
                        return LoweringError.OutOfMemory;
                    };
                    st.updateSymbolValue(name, element_value) catch {};
                }
            }
        }

        return;
    }

    // print variable type info before lowering
    log.debug("[BEFORE MLIR] Variable: {s}\n", .{var_decl.name});
    log.debug("  isResolved: {any}\n", .{var_decl.type_info.isResolved()});
    log.debug("  category: {s}\n", .{@tagName(var_decl.type_info.category)});
    log.debug("  source: {s}\n", .{@tagName(var_decl.type_info.source)});
    if (var_decl.type_info.ora_type) |ora_type| {
        log.debug("  ora_type: present ({s})\n", .{@tagName(ora_type)});
    } else {
        log.debug("  ora_type: null\n", .{});
    }
    if (var_decl.value) |value| {
        log.debug("  has initializer: true\n", .{});
        switch (value.*) {
            .Binary => |*b| {
                log.debug("  initializer type: Binary\n", .{});
                log.debug("    binary.isResolved: {any}\n", .{b.type_info.isResolved()});
                if (b.type_info.ora_type) |bt| {
                    log.debug("    binary.ora_type: present ({s})\n", .{@tagName(bt)});
                } else {
                    log.debug("    binary.ora_type: null\n", .{});
                }
                // check if operands are in symbol table
                if (b.lhs.* == .Identifier) {
                    const lhs_name = b.lhs.Identifier.name;
                    const lhs_found = if (self.symbol_table) |st| st.lookupSymbol(lhs_name) != null else false;
                    log.debug("    binary.lhs ({s}) in symbol table: {any}\n", .{ lhs_name, lhs_found });
                }
                if (b.rhs.* == .Identifier) {
                    const rhs_name = b.rhs.Identifier.name;
                    const rhs_found = if (self.symbol_table) |st| st.lookupSymbol(rhs_name) != null else false;
                    log.debug("    binary.rhs ({s}) in symbol table: {any}\n", .{ rhs_name, rhs_found });
                }
            },
            else => {
                log.debug("  initializer type: {s}\n", .{@tagName(value.*)});
            },
        }
    } else {
        log.debug("  has initializer: false\n", .{});
    }

    // use type from AST - types should have been copied during type resolution
    // if base pointers are invalid, it means the arena was freed too early
    const mlir_type = self.type_mapper.toMlirType(var_decl.type_info);

    log.debug("[lowerVariableDecl] {s}: region={any}, kind={any}\n", .{ var_decl.name, var_decl.region, var_decl.kind });

    // add symbol to symbol table if available
    if (self.symbol_table) |st| {
        st.addSymbol(var_decl.name, mlir_type, var_decl.region, null, var_decl.kind) catch {
            log.debug("ERROR: Failed to add symbol to table: {s}\n", .{var_decl.name});
            return LoweringError.OutOfMemory;
        };
    }

    // handle variable declarations based on memory region
    switch (var_decl.region) {
        .Stack => {
            log.debug("[lowerVariableDecl] Calling lowerStackVariableDecl for: {s}\n", .{var_decl.name});
            try lowerStackVariableDecl(self, var_decl, mlir_type, loc);
        },
        .Storage => {
            try lowerStorageVariableDecl(self, var_decl, mlir_type, loc);
        },
        .Memory => {
            try lowerMemoryVariableDecl(self, var_decl, mlir_type, loc);
        },
        .TStore => {
            try lowerTStoreVariableDecl(self, var_decl, mlir_type, loc);
        },
        .Calldata => {
            log.debug("ERROR: Calldata variables are not allowed: {s}\n", .{var_decl.name});
            return LoweringError.InvalidMemoryRegion;
        },
    }
}

/// Check if an Ora type is an aggregate that needs memref (arrays, slices, maps)
/// Structs are handled as SSA values, not memrefs (canonical MLIR approach)
fn isAggregateType(ora_type: ?lib.ast.type_info.OraType) bool {
    if (ora_type) |ty| {
        const result = switch (ty) {
            .array, .slice, .map => true, // These need memrefs for indexing
            .struct_type, .tuple, .anonymous_struct => false, // These are SSA values
            else => false,
        };
        log.debug("[isAggregateType] type={any}, result={}\n", .{ ty, result });
        return result;
    }
    log.debug("[isAggregateType] type=null, result=false\n", .{});
    return false;
}

/// Check if an Ora type is a scalar value type (integers, bools, addresses, etc.)
/// These are normally lowered as SSA values, but mutable scalars (var) may be
/// represented as memrefs when we need accumulator-style semantics across loops.
fn isScalarValueType(ora_type: ?lib.ast.type_info.OraType) bool {
    if (ora_type) |ty| {
        // treat anything that is not an aggregate container as a scalar value type.
        return switch (ty) {
            .array, .slice, .map, .struct_type, .tuple, .anonymous_struct => false,
            else => true,
        };
    }
    return false;
}

/// Lower stack variable declarations (local variables)
pub fn lowerStackVariableDecl(self: *const StatementLowerer, var_decl: *const lib.ast.Statements.VariableDeclNode, mlir_type: c.MlirType, loc: c.MlirLocation) LoweringError!void {
    // for mutable variables (var): use memref for aggregates (arrays, maps, slices)
    // and for scalar accumulators that may be mutated inside loops and read after.
    // immutable locals (let/const) remain SSA values.
    const is_aggregate = isAggregateType(var_decl.type_info.ora_type);
    const is_scalar = isScalarValueType(var_decl.type_info.ora_type);
    const prefer_ssa = if (self.local_var_map) |lvm|
        if (lvm.getLocalVarKind(var_decl.name)) |kind| kind == .SSA else false
    else
        false;
    const force_array_memref = if (var_decl.type_info.ora_type) |ora_type| ora_type == .array else false;
    const needs_memref = self.force_stack_memref or force_array_memref or ((var_decl.kind == .Var) and (is_aggregate or (is_scalar and !prefer_ssa)));

    log.debug(
        "[lowerStackVariableDecl] {s}: kind={any}, is_aggregate={}, is_scalar={}, force_memref={}, needs_memref={}\n",
        .{ var_decl.name, var_decl.kind, is_aggregate, is_scalar, self.force_stack_memref, needs_memref },
    );

    if (needs_memref) {
        log.debug("[lowerStackVariableDecl] Creating memref-backed local: {s}\n", .{var_decl.name});

        const ora_type = var_decl.type_info.ora_type orelse {
            log.debug("[lowerStackVariableDecl] ERROR: ora_type is null for aggregate variable {s}\n", .{var_decl.name});
            @panic("lowerStackVariableDecl: ora_type is null for aggregate - this indicates a type system bug");
        };

        // If we already have a memref/shaped initializer for an array, reuse it directly.
        if (ora_type == .array) {
            if (var_decl.value) |init_expr| {
                if (init_expr.* == .ArrayLiteral) {
                    const init_value = helpers.lowerValueWithImplicitTry(@constCast(self), &init_expr.*, var_decl.type_info);
                    if (self.local_var_map) |lvm| {
                        lvm.addLocalVar(var_decl.name, init_value) catch {
                            log.debug("ERROR: Failed to add local variable memref to map: {s}\n", .{var_decl.name});
                            return LoweringError.OutOfMemory;
                        };
                    }
                    if (self.symbol_table) |st| {
                        st.updateSymbolValue(var_decl.name, init_value) catch {
                            log.debug("WARNING: Failed to update symbol value: {s}\n", .{var_decl.name});
                        };
                    }
                    return;
                }
                const init_value = helpers.lowerValueWithImplicitTry(@constCast(self), &init_expr.*, var_decl.type_info);
                const init_type = c.oraValueGetType(init_value);
                if (c.oraTypeIsAMemRef(init_type) or c.oraTypeIsAShaped(init_type)) {
                    if (self.local_var_map) |lvm| {
                        lvm.addLocalVar(var_decl.name, init_value) catch {
                            log.debug("ERROR: Failed to add local variable memref to map: {s}\n", .{var_decl.name});
                            return LoweringError.OutOfMemory;
                        };
                    }
                    if (self.symbol_table) |st| {
                        st.updateSymbolValue(var_decl.name, init_value) catch {
                            log.debug("WARNING: Failed to update symbol value: {s}\n", .{var_decl.name});
                        };
                    }
                    return;
                }
            }
        }

        // arrays need rank 1 with shape, other aggregates and scalar accumulators use rank 0
        const rank: u32 = if (ora_type == .array) 1 else 0;
        var shape_buf: [1]i64 = .{0};
        const shape: ?*const i64 = if (ora_type == .array) blk: {
            shape_buf[0] = @intCast(ora_type.array.len);
            break :blk &shape_buf[0];
        } else null;

        const memref_elem_type = if (ora_type == .array)
            self.type_mapper.toMlirType(.{ .ora_type = ora_type.array.elem.* })
        else
            mlir_type;
        const memref_type = h.memRefType(self.ctx, memref_elem_type, rank, shape, h.nullAttr(), h.nullAttr());

        // allocate memory on the stack
        const alloca_op = self.ora_dialect.createMemrefAlloca(memref_type, loc);
        h.appendOp(self.block, alloca_op);
        const memref = h.getResult(alloca_op, 0);

        // initialize the variable if there's an initializer
        if (var_decl.value) |init_expr| {
            var init_value = helpers.lowerValueWithImplicitTry(@constCast(self), &init_expr.*, var_decl.type_info);

            // insert refinement guard for variable initialization (skip if optimized)
            init_value = try helpers.insertRefinementGuard(self, init_value, ora_type, var_decl.span, var_decl.name, var_decl.skip_guard);

            // ensure value type matches memref element type (both should be Ora types now)
            const element_type = c.oraShapedTypeGetElementType(memref_type);
            init_value = self.expr_lowerer.convertToType(init_value, element_type, var_decl.span);

            const store_op = self.ora_dialect.createMemrefStore(init_value, memref, &[_]c.MlirValue{}, loc);
            h.appendOp(self.block, store_op);
        } else {
            // store default value
            var default_value = try createDefaultValue(self, mlir_type, var_decl.kind, loc);

            // insert refinement guard for default value
            default_value = try helpers.insertRefinementGuard(self, default_value, ora_type, var_decl.span, var_decl.name, var_decl.skip_guard);

            // ensure value type matches memref element type (both should be Ora types now)
            const element_type = c.oraShapedTypeGetElementType(memref_type);
            default_value = self.expr_lowerer.convertToType(default_value, element_type, var_decl.span);

            const store_op = self.ora_dialect.createMemrefStore(default_value, memref, &[_]c.MlirValue{}, loc);
            h.appendOp(self.block, store_op);
        }

        // store the memref in the local variable map
        if (self.local_var_map) |lvm| {
            lvm.addLocalVar(var_decl.name, memref) catch {
                log.debug("ERROR: Failed to add local variable memref to map: {s}\n", .{var_decl.name});
                return LoweringError.OutOfMemory;
            };
        }

        // update symbol table with the memref
        if (self.symbol_table) |st| {
            st.updateSymbolValue(var_decl.name, memref) catch {
                log.debug("WARNING: Failed to update symbol value: {s}\n", .{var_decl.name});
            };
        }
    } else {
        // scalar mutable (var) or immutable (let/const) - use SSA values
        log.debug("[lowerStackVariableDecl] Creating SSA value for: {s}\n", .{var_decl.name});
        var init_value: c.MlirValue = undefined;

        if (var_decl.value) |init_expr| {
            // lower the initializer expression
            init_value = helpers.lowerValueWithImplicitTry(@constCast(self), &init_expr.*, var_decl.type_info);

            // ensure the initializer value matches the declared type
            // this is critical for structs - if map load returns i256, convert it to struct type
            if (var_decl.type_info.ora_type) |ora_type| {
                // insert refinement guard BEFORE type conversion to avoid base→refinement→base round-trip
                init_value = try helpers.insertRefinementGuard(self, init_value, ora_type, var_decl.span, var_decl.name, var_decl.skip_guard);

                // convert to the declared type if needed (e.g., map load might return wrong type)
                const expected_type = self.expr_lowerer.type_mapper.toMlirType(var_decl.type_info);
                init_value = self.expr_lowerer.convertToType(init_value, expected_type, var_decl.span);
            }
        } else {
            // create default value based on variable kind
            init_value = try createDefaultValue(self, mlir_type, var_decl.kind, loc);

            // insert refinement guard for default value (skip if optimized)
            if (var_decl.type_info.ora_type) |ora_type| {
                init_value = try helpers.insertRefinementGuard(self, init_value, ora_type, var_decl.span, var_decl.name, var_decl.skip_guard);
            }
        }

        // store the local variable in our map for later reference
        if (self.local_var_map) |lvm| {
            lvm.addLocalVar(var_decl.name, init_value) catch {
                log.debug("ERROR: Failed to add local variable to map: {s}\n", .{var_decl.name});
                return LoweringError.OutOfMemory;
            };
        }

        // update symbol table with the value
        if (self.symbol_table) |st| {
            st.updateSymbolValue(var_decl.name, init_value) catch {
                log.debug("WARNING: Failed to update symbol value: {s}\n", .{var_decl.name});
            };
        }
    }
}

/// Lower storage variable declarations
pub fn lowerStorageVariableDecl(self: *const StatementLowerer, var_decl: *const lib.ast.Statements.VariableDeclNode, _: c.MlirType, loc: c.MlirLocation) LoweringError!void {
    // storage variables are typically handled at the contract level
    // if there's an initializer, we need to generate a store operation
    if (var_decl.value) |init_expr| {
        const init_value = helpers.lowerValueWithImplicitTry(@constCast(self), &init_expr.*, var_decl.type_info);

        // generate storage store operation
        const store_op = self.memory_manager.createStorageStore(init_value, var_decl.name, loc);
        h.appendOp(self.block, store_op);
    }

    // ensure storage variable is registered
    if (self.storage_map) |sm| {
        _ = @constCast(sm).addStorageVariable(var_decl.name, var_decl.span) catch {
            log.debug("WARNING: Failed to register storage variable: {s}\n", .{var_decl.name});
        };
    }
}

/// Lower memory variable declarations
pub fn lowerMemoryVariableDecl(self: *const StatementLowerer, var_decl: *const lib.ast.Statements.VariableDeclNode, mlir_type: c.MlirType, loc: c.MlirLocation) LoweringError!void {
    var alloca_result: c.MlirValue = undefined;

    // Get ora_type to determine if this is an array
    const ora_type = var_decl.type_info.ora_type;

    if (var_decl.value) |init_expr| {
        // lower initializer first; array literals already produce a memref
        const init_value = helpers.lowerValueWithImplicitTry(@constCast(self), &init_expr.*, var_decl.type_info);
        const init_type = c.oraValueGetType(init_value);

        if (c.oraTypeIsAMemRef(init_type)) {
            // reuse the initializer memref as the storage for this variable
            alloca_result = init_value;
        } else {
            // create memory allocation and store initializer
            const memref_type = createMemrefTypeForVariable(self, mlir_type, ora_type, var_decl.region);
            const alloca_op = self.memory_manager.createAllocaOp(memref_type, var_decl.region, var_decl.name, loc);
            h.appendOp(self.block, alloca_op);
            alloca_result = h.getResult(alloca_op, 0);

            // Convert init_value to match memref element type (e.g., !ora.non_zero_address -> !ora.address)
            const element_type = c.oraShapedTypeGetElementType(memref_type);
            const store_value = if (!c.oraTypeEqual(init_type, element_type))
                self.type_mapper.createConversionOp(self.block, init_value, element_type, var_decl.span)
            else
                init_value;

            const store_op = self.memory_manager.createStoreOp(store_value, alloca_result, var_decl.region, loc);
            h.appendOp(self.block, store_op);
        }
    } else {
        // create memory allocation without initializer
        const memref_type = createMemrefTypeForVariable(self, mlir_type, ora_type, var_decl.region);
        const alloca_op = self.memory_manager.createAllocaOp(memref_type, var_decl.region, var_decl.name, loc);
        h.appendOp(self.block, alloca_op);
        alloca_result = h.getResult(alloca_op, 0);
    }

    // store the memory reference in local variable map
    if (self.local_var_map) |lvm| {
        lvm.addLocalVar(var_decl.name, alloca_result) catch {
            log.debug("ERROR: Failed to add memory variable to map: {s}\n", .{var_decl.name});
            return LoweringError.OutOfMemory;
        };
    }
}

/// Create the appropriate memref type for a variable, handling arrays specially
fn createMemrefTypeForVariable(
    self: *const StatementLowerer,
    mlir_type: c.MlirType,
    ora_type: ?lib.ast.type_info.OraType,
    region: lib.ast.Statements.MemoryRegion,
) c.MlirType {
    // If already a memref, use it directly
    if (c.oraTypeIsAMemRef(mlir_type)) {
        return mlir_type;
    }

    // For arrays, create memref<NxT> with proper shape
    if (ora_type) |ot| {
        if (ot == .array) {
            // Get element type from the array's nested ora_type
            const elem_ora_type = ot.array.elem.*;
            const elem_mlir_type = self.type_mapper.toMlirType(.{ .ora_type = elem_ora_type });
            var shape_buf: [1]i64 = .{@intCast(ot.array.len)};
            return h.memRefType(
                self.ctx,
                elem_mlir_type,
                1,
                &shape_buf[0],
                h.nullAttr(),
                self.memory_manager.getMemorySpaceAttribute(region),
            );
        }
    }

    // For scalars, create memref<T> (rank 0)
    return h.memRefType(
        self.ctx,
        mlir_type,
        0,
        null,
        h.nullAttr(),
        self.memory_manager.getMemorySpaceAttribute(region),
    );
}

/// Lower transient storage variable declarations
pub fn lowerTStoreVariableDecl(self: *const StatementLowerer, var_decl: *const lib.ast.Statements.VariableDeclNode, mlir_type: c.MlirType, loc: c.MlirLocation) LoweringError!void {
    // transient storage variables are similar to storage but temporary
    var init_value: c.MlirValue = undefined;

    if (var_decl.value) |init_expr| {
        init_value = self.expr_lowerer.lowerExpression(&init_expr.*);

        // generate transient storage store operation
        const store_op = self.memory_manager.createTStoreStore(init_value, var_decl.name, loc);
        h.appendOp(self.block, store_op);
    } else {
        // create default value for uninitialized transient storage variables
        init_value = try createDefaultValue(self, mlir_type, var_decl.kind, loc);
    }

    // store the transient storage variable in local variable map
    if (self.local_var_map) |lvm| {
        lvm.addLocalVar(var_decl.name, init_value) catch {
            log.debug("ERROR: Failed to add transient storage variable to map: {s}\n", .{var_decl.name});
            return LoweringError.OutOfMemory;
        };
    }

    // update symbol table with the value
    if (self.symbol_table) |st| {
        st.updateSymbolValue(var_decl.name, init_value) catch {
            log.debug("WARNING: Failed to update transient storage symbol value: {s}\n", .{var_decl.name});
        };
    }
}

/// Create default value for uninitialized variables
pub fn createDefaultValue(self: *const StatementLowerer, mlir_type: c.MlirType, kind: lib.ast.Statements.VariableKind, loc: c.MlirLocation) LoweringError!c.MlirValue {
    _ = kind; // Variable kind might affect default value in the future

    // for now, create zero value for integer types
    const const_op = self.ora_dialect.createArithConstant(0, mlir_type, loc);
    h.appendOp(self.block, const_op);
    return h.getResult(const_op, 0);
}
