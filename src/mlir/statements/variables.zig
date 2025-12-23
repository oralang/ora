// ============================================================================
// Variable Declaration Statement Lowering
// ============================================================================
// Variable declarations: var, let, const for different memory regions

const std = @import("std");
const c = @import("../c.zig").c;
const lib = @import("ora_lib");
const h = @import("../helpers.zig");
const StatementLowerer = @import("statement_lowerer.zig").StatementLowerer;
const LoweringError = StatementLowerer.LoweringError;
const helpers = @import("helpers.zig");

/// Lower variable declarations with region-specific handling
pub fn lowerVariableDecl(self: *const StatementLowerer, var_decl: *const lib.ast.Statements.VariableDeclNode) LoweringError!void {
    const loc = self.fileLoc(var_decl.span);

    // Print variable type info before lowering
    std.debug.print("[BEFORE MLIR] Variable: {s}\n", .{var_decl.name});
    std.debug.print("  isResolved: {any}\n", .{var_decl.type_info.isResolved()});
    std.debug.print("  category: {s}\n", .{@tagName(var_decl.type_info.category)});
    std.debug.print("  source: {s}\n", .{@tagName(var_decl.type_info.source)});
    if (var_decl.type_info.ora_type) |ora_type| {
        std.debug.print("  ora_type: present ({s})\n", .{@tagName(ora_type)});
    } else {
        std.debug.print("  ora_type: null\n", .{});
    }
    if (var_decl.value) |value| {
        std.debug.print("  has initializer: true\n", .{});
        switch (value.*) {
            .Binary => |*b| {
                std.debug.print("  initializer type: Binary\n", .{});
                std.debug.print("    binary.isResolved: {any}\n", .{b.type_info.isResolved()});
                if (b.type_info.ora_type) |bt| {
                    std.debug.print("    binary.ora_type: present ({s})\n", .{@tagName(bt)});
                } else {
                    std.debug.print("    binary.ora_type: null\n", .{});
                }
                // Check if operands are in symbol table
                if (b.lhs.* == .Identifier) {
                    const lhs_name = b.lhs.Identifier.name;
                    const lhs_found = if (self.symbol_table) |st| st.lookupSymbol(lhs_name) != null else false;
                    std.debug.print("    binary.lhs ({s}) in symbol table: {any}\n", .{ lhs_name, lhs_found });
                }
                if (b.rhs.* == .Identifier) {
                    const rhs_name = b.rhs.Identifier.name;
                    const rhs_found = if (self.symbol_table) |st| st.lookupSymbol(rhs_name) != null else false;
                    std.debug.print("    binary.rhs ({s}) in symbol table: {any}\n", .{ rhs_name, rhs_found });
                }
            },
            else => {
                std.debug.print("  initializer type: {s}\n", .{@tagName(value.*)});
            },
        }
    } else {
        std.debug.print("  has initializer: false\n", .{});
    }

    // Use type from AST - types should have been copied during type resolution
    // If base pointers are invalid, it means the arena was freed too early
    const mlir_type = self.type_mapper.toMlirType(var_decl.type_info);

    std.debug.print("[lowerVariableDecl] {s}: region={any}, kind={any}\n", .{ var_decl.name, var_decl.region, var_decl.kind });

    // Add symbol to symbol table if available
    if (self.symbol_table) |st| {
        st.addSymbol(var_decl.name, mlir_type, var_decl.region, null, var_decl.kind) catch {
            std.debug.print("ERROR: Failed to add symbol to table: {s}\n", .{var_decl.name});
            return LoweringError.OutOfMemory;
        };
    }

    // Handle variable declarations based on memory region
    switch (var_decl.region) {
        .Stack => {
            std.debug.print("[lowerVariableDecl] Calling lowerStackVariableDecl for: {s}\n", .{var_decl.name});
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
        std.debug.print("[isAggregateType] type={any}, result={}\n", .{ ty, result });
        return result;
    }
    std.debug.print("[isAggregateType] type=null, result=false\n", .{});
    return false;
}

/// Check if an Ora type is a scalar value type (integers, bools, addresses, etc.)
/// These are normally lowered as SSA values, but mutable scalars (var) may be
/// represented as memrefs when we need accumulator-style semantics across loops.
fn isScalarValueType(ora_type: ?lib.ast.type_info.OraType) bool {
    if (ora_type) |ty| {
        // Treat anything that is not an aggregate container as a scalar value type.
        return switch (ty) {
            .array, .slice, .map, .struct_type, .tuple, .anonymous_struct => false,
            else => true,
        };
    }
    return false;
}

/// Lower stack variable declarations (local variables)
pub fn lowerStackVariableDecl(self: *const StatementLowerer, var_decl: *const lib.ast.Statements.VariableDeclNode, mlir_type: c.MlirType, loc: c.MlirLocation) LoweringError!void {
    // For mutable variables (var): use memref for aggregates (arrays, maps, slices)
    // and for scalar accumulators that may be mutated inside loops and read after.
    // Immutable locals (let/const) remain SSA values.
    const is_aggregate = isAggregateType(var_decl.type_info.ora_type);
    const is_scalar = isScalarValueType(var_decl.type_info.ora_type);
    const needs_memref = (var_decl.kind == .Var) and (is_aggregate or is_scalar);

    std.debug.print(
        "[lowerStackVariableDecl] {s}: kind={any}, is_aggregate={}, is_scalar={}, needs_memref={}\n",
        .{ var_decl.name, var_decl.kind, is_aggregate, is_scalar, needs_memref },
    );

    if (needs_memref) {
        std.debug.print("[lowerStackVariableDecl] Creating memref-backed local: {s}\n", .{var_decl.name});

        const ora_type = var_decl.type_info.ora_type orelse {
            std.debug.print("[lowerStackVariableDecl] ERROR: ora_type is null for aggregate variable {s}\n", .{var_decl.name});
            @panic("lowerStackVariableDecl: ora_type is null for aggregate - this indicates a type system bug");
        };

        // Arrays need rank 1 with shape, other aggregates and scalar accumulators use rank 0
        const rank: u32 = if (ora_type == .array) 1 else 0;
        const shape: ?[*]const i64 = if (ora_type == .array) blk: {
            const array_len: i64 = @intCast(ora_type.array.len);
            break :blk &[_]i64{array_len};
        } else null;

        const memref_type = c.mlirMemRefTypeGet(mlir_type, rank, shape, c.mlirAttributeGetNull(), c.mlirAttributeGetNull());

        // Allocate memory on the stack
        const alloca_op = self.ora_dialect.createMemrefAlloca(memref_type, loc);
        h.appendOp(self.block, alloca_op);
        const memref = h.getResult(alloca_op, 0);

        // Initialize the variable if there's an initializer
        if (var_decl.value) |init_expr| {
            var init_value = self.expr_lowerer.lowerExpression(&init_expr.*);

            // Insert refinement guard for variable initialization (skip if optimized)
            init_value = try helpers.insertRefinementGuard(self, init_value, ora_type, var_decl.span, var_decl.skip_guard);

            // Ensure value type matches memref element type (both should be Ora types now)
            const element_type = c.mlirShapedTypeGetElementType(memref_type);
            init_value = self.expr_lowerer.convertToType(init_value, element_type, var_decl.span);

            const store_op = self.ora_dialect.createMemrefStore(init_value, memref, &[_]c.MlirValue{}, loc);
            h.appendOp(self.block, store_op);
        } else {
            // Store default value
            var default_value = try createDefaultValue(self, mlir_type, var_decl.kind, loc);

            // Insert refinement guard for default value
            default_value = try helpers.insertRefinementGuard(self, default_value, ora_type, var_decl.span, var_decl.skip_guard);

            // Ensure value type matches memref element type (both should be Ora types now)
            const element_type = c.mlirShapedTypeGetElementType(memref_type);
            default_value = self.expr_lowerer.convertToType(default_value, element_type, var_decl.span);

            const store_op = self.ora_dialect.createMemrefStore(default_value, memref, &[_]c.MlirValue{}, loc);
            h.appendOp(self.block, store_op);
        }

        // Store the memref in the local variable map
        if (self.local_var_map) |lvm| {
            lvm.addLocalVar(var_decl.name, memref) catch {
                std.debug.print("ERROR: Failed to add local variable memref to map: {s}\n", .{var_decl.name});
                return LoweringError.OutOfMemory;
            };
        }

        // Update symbol table with the memref
        if (self.symbol_table) |st| {
            st.updateSymbolValue(var_decl.name, memref) catch {
                std.debug.print("WARNING: Failed to update symbol value: {s}\n", .{var_decl.name});
            };
        }
    } else {
        // Scalar mutable (var) or immutable (let/const) - use SSA values
        std.debug.print("[lowerStackVariableDecl] Creating SSA value for: {s}\n", .{var_decl.name});
        var init_value: c.MlirValue = undefined;

        if (var_decl.value) |init_expr| {
            // Lower the initializer expression
            init_value = self.expr_lowerer.lowerExpression(&init_expr.*);

            // Ensure the initializer value matches the declared type
            // This is critical for structs - if map load returns i256, convert it to struct type
            if (var_decl.type_info.ora_type) |ora_type| {
                // Convert to the declared type if needed (e.g., map load might return wrong type)
                const expected_type = self.expr_lowerer.type_mapper.toMlirType(var_decl.type_info);
                init_value = self.expr_lowerer.convertToType(init_value, expected_type, var_decl.span);

                // Insert refinement guard for variable initialization (skip if optimized)
                init_value = try helpers.insertRefinementGuard(self, init_value, ora_type, var_decl.span, var_decl.skip_guard);
            }
        } else {
            // Create default value based on variable kind
            init_value = try createDefaultValue(self, mlir_type, var_decl.kind, loc);

            // Insert refinement guard for default value (skip if optimized)
            if (var_decl.type_info.ora_type) |ora_type| {
                init_value = try helpers.insertRefinementGuard(self, init_value, ora_type, var_decl.span, var_decl.skip_guard);
            }
        }

        // Store the local variable in our map for later reference
        if (self.local_var_map) |lvm| {
            lvm.addLocalVar(var_decl.name, init_value) catch {
                std.debug.print("ERROR: Failed to add local variable to map: {s}\n", .{var_decl.name});
                return LoweringError.OutOfMemory;
            };
        }

        // Update symbol table with the value
        if (self.symbol_table) |st| {
            st.updateSymbolValue(var_decl.name, init_value) catch {
                std.debug.print("WARNING: Failed to update symbol value: {s}\n", .{var_decl.name});
            };
        }
    }
}

/// Lower storage variable declarations
pub fn lowerStorageVariableDecl(self: *const StatementLowerer, var_decl: *const lib.ast.Statements.VariableDeclNode, _: c.MlirType, loc: c.MlirLocation) LoweringError!void {
    // Storage variables are typically handled at the contract level
    // If there's an initializer, we need to generate a store operation
    if (var_decl.value) |init_expr| {
        const init_value = self.expr_lowerer.lowerExpression(&init_expr.*);

        // Generate storage store operation
        const store_op = self.memory_manager.createStorageStore(init_value, var_decl.name, loc);
        h.appendOp(self.block, store_op);
    }

    // Ensure storage variable is registered
    if (self.storage_map) |sm| {
        _ = @constCast(sm).addStorageVariable(var_decl.name, var_decl.span) catch {
            std.debug.print("WARNING: Failed to register storage variable: {s}\n", .{var_decl.name});
        };
    }
}

/// Lower memory variable declarations
pub fn lowerMemoryVariableDecl(self: *const StatementLowerer, var_decl: *const lib.ast.Statements.VariableDeclNode, mlir_type: c.MlirType, loc: c.MlirLocation) LoweringError!void {
    // Create memory allocation
    const alloca_op = self.memory_manager.createAllocaOp(mlir_type, var_decl.region, var_decl.name, loc);
    h.appendOp(self.block, alloca_op);
    const alloca_result = h.getResult(alloca_op, 0);

    if (var_decl.value) |init_expr| {
        // Lower initializer and store to memory
        const init_value = self.expr_lowerer.lowerExpression(&init_expr.*);

        const store_op = self.memory_manager.createStoreOp(init_value, alloca_result, var_decl.region, loc);
        h.appendOp(self.block, store_op);
    }

    // Store the memory reference in local variable map
    if (self.local_var_map) |lvm| {
        lvm.addLocalVar(var_decl.name, alloca_result) catch {
            std.debug.print("ERROR: Failed to add memory variable to map: {s}\n", .{var_decl.name});
            return LoweringError.OutOfMemory;
        };
    }
}

/// Lower transient storage variable declarations
pub fn lowerTStoreVariableDecl(self: *const StatementLowerer, var_decl: *const lib.ast.Statements.VariableDeclNode, mlir_type: c.MlirType, loc: c.MlirLocation) LoweringError!void {
    // Transient storage variables are similar to storage but temporary
    var init_value: c.MlirValue = undefined;

    if (var_decl.value) |init_expr| {
        init_value = self.expr_lowerer.lowerExpression(&init_expr.*);

        // Generate transient storage store operation
        const store_op = self.memory_manager.createTStoreStore(init_value, var_decl.name, loc);
        h.appendOp(self.block, store_op);
    } else {
        // Create default value for uninitialized transient storage variables
        init_value = try createDefaultValue(self, mlir_type, var_decl.kind, loc);
    }

    // Store the transient storage variable in local variable map
    if (self.local_var_map) |lvm| {
        lvm.addLocalVar(var_decl.name, init_value) catch {
            std.debug.print("ERROR: Failed to add transient storage variable to map: {s}\n", .{var_decl.name});
            return LoweringError.OutOfMemory;
        };
    }

    // Update symbol table with the value
    if (self.symbol_table) |st| {
        st.updateSymbolValue(var_decl.name, init_value) catch {
            std.debug.print("WARNING: Failed to update transient storage symbol value: {s}\n", .{var_decl.name});
        };
    }
}

/// Create default value for uninitialized variables
pub fn createDefaultValue(self: *const StatementLowerer, mlir_type: c.MlirType, kind: lib.ast.Statements.VariableKind, loc: c.MlirLocation) LoweringError!c.MlirValue {
    _ = kind; // Variable kind might affect default value in the future

    // For now, create zero value for integer types
    var const_state = h.opState("arith.constant", loc);
    c.mlirOperationStateAddResults(&const_state, 1, @ptrCast(&mlir_type));

    const attr = c.mlirIntegerAttrGet(mlir_type, 0);
    const value_id = h.identifier(self.ctx, "value");
    var attrs = [_]c.MlirNamedAttribute{c.mlirNamedAttributeGet(value_id, attr)};
    c.mlirOperationStateAddAttributes(&const_state, attrs.len, &attrs);

    const const_op = c.mlirOperationCreate(&const_state);
    h.appendOp(self.block, const_op);

    return h.getResult(const_op, 0);
}
