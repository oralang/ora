// ============================================================================
// Assignment Statement Lowering
// ============================================================================
// Assignment operations: simple assignments, compound assignments, destructuring

const std = @import("std");
const c = @import("mlir_c_api").c;
const lib = @import("ora_lib");
const h = @import("../helpers.zig");
const constants = @import("../lower.zig");
const StatementLowerer = @import("statement_lowerer.zig").StatementLowerer;
const LoweringError = StatementLowerer.LoweringError;
const MemoryManager = @import("../memory.zig").MemoryManager;
const helpers = @import("helpers.zig");
const error_handling = @import("../error_handling.zig");
const log = @import("log");

fn reportAssignmentError(
    self: *const StatementLowerer,
    span: lib.ast.SourceSpan,
    error_type: error_handling.ErrorType,
    message: []const u8,
    suggestion: ?[]const u8,
) void {
    if (self.expr_lowerer.error_handler) |handler| {
        handler.reportError(error_type, span, message, suggestion) catch {};
    } else {
        log.err("{s}\n", .{message});
    }
}

/// Lower expression statements (including assignments)
pub fn lowerExpressionStatement(self: *const StatementLowerer, expr: *const lib.ast.Statements.ExprNode) LoweringError!void {
    switch (expr.*) {
        .Assignment => |assign| {
            try lowerAssignmentExpression(self, &assign);
        },
        .CompoundAssignment => |compound| {
            try lowerCompoundAssignmentExpr(self, &compound);
        },
        else => {
            // lower other expression statements
            _ = helpers.lowerValueWithImplicitTry(self, expr, null);
        },
    }
}

/// Lower assignment expressions with comprehensive lvalue resolution
pub fn lowerAssignmentExpression(self: *const StatementLowerer, assign: *const lib.ast.Expressions.AssignmentExpr) LoweringError!void {
    // lower the value expression first
    var expected_type: ?lib.ast.Types.TypeInfo = null;
    switch (assign.target.*) {
        .Identifier => |ident| expected_type = ident.type_info,
        .FieldAccess => |fa| expected_type = fa.type_info,
        else => {},
    }
    var value = helpers.lowerValueWithImplicitTry(self, assign.value, expected_type);

    // insert refinement guard if target type is a refinement type
    // get type from target identifier if it's an identifier
    if (assign.target.* == .Identifier) {
        const ident = &assign.target.Identifier;
        if (ident.type_info.ora_type) |target_ora_type| {
            // use skip_guard flag (set during type resolution if optimization applies)
            // skip_guard is set in synthAssignment when: constant satisfies constraint,
            // subtyping applies, or value comes from trusted builtin
            value = try helpers.insertRefinementGuard(self, value, target_ora_type, assign.span, assign.skip_guard);
        }
    }

    // resolve the lvalue and generate appropriate store operation
    try lowerLValueAssignment(self, assign.target, value, getExpressionSpan(assign.target));
}

/// Lower lvalue assignments (handles identifiers, field access, array indexing)
pub fn lowerLValueAssignment(self: *const StatementLowerer, target: *const lib.ast.Expressions.ExprNode, value: c.MlirValue, span: lib.ast.SourceSpan) LoweringError!void {
    const loc = self.fileLoc(span);

    switch (target.*) {
        .Identifier => |ident| {
            try lowerIdentifierAssignment(self, &ident, value, loc);
        },
        .FieldAccess => |field_access| {
            try lowerFieldAccessAssignment(self, &field_access, value, loc);
        },
        .Index => |index_expr| {
            try lowerIndexAssignment(self, &index_expr, value, loc);
        },
        else => {
            log.err("Unsupported lvalue type for assignment: {s}\n", .{@tagName(target.*)});
            reportAssignmentError(
                self,
                span,
                .UnsupportedFeature,
                "Unsupported assignment target; expected identifier, field access, or index",
                null,
            );
            return LoweringError.InvalidLValue;
        },
    }
}

/// Lower identifier assignments
pub fn lowerIdentifierAssignment(self: *const StatementLowerer, ident: *const lib.ast.Expressions.IdentifierExpr, value: c.MlirValue, loc: c.MlirLocation) LoweringError!void {
    // check symbol table first for memory region information
    if (self.symbol_table) |st| {
        log.debug("[lowerIdentifierAssignment] Looking up symbol: {s}\n", .{ident.name});
        if (st.lookupSymbol(ident.name)) |symbol| {
            // store variable_kind early to avoid shadowing issues in nested scopes
            const var_kind = symbol.variable_kind;
            log.debug("[lowerIdentifierAssignment] Symbol found: {s}, variable_kind: {any}, symbol_kind: {any}\n", .{ ident.name, var_kind, symbol.symbol_kind });

            // only variables have variable_kind - parameters and constants don't
            if (symbol.symbol_kind != .Variable) {
                log.debug("[lowerIdentifierAssignment] Symbol is not a variable (kind: {any}), skipping variable_kind check\n", .{symbol.symbol_kind});
            }
            const region = blk: {
                if (std.mem.eql(u8, symbol.region, "storage")) break :blk lib.ast.Statements.MemoryRegion.Storage;
                if (std.mem.eql(u8, symbol.region, "memory")) break :blk lib.ast.Statements.MemoryRegion.Memory;
                if (std.mem.eql(u8, symbol.region, "tstore")) break :blk lib.ast.Statements.MemoryRegion.TStore;
                if (std.mem.eql(u8, symbol.region, "calldata")) break :blk lib.ast.Statements.MemoryRegion.Calldata;
                if (std.mem.eql(u8, symbol.region, "stack")) break :blk lib.ast.Statements.MemoryRegion.Stack;
                // if parsing fails, check storage_map as fallback for storage variables
                if (self.storage_map) |sm| {
                    if (sm.hasStorageVariable(ident.name)) {
                        // this is a storage variable - handle it directly
                        const store_op = self.memory_manager.createStorageStore(value, ident.name, loc);
                        h.appendOp(self.block, store_op);
                        return;
                    }
                }
                // default to Stack if parsing fails and not in storage_map
                break :blk lib.ast.Statements.MemoryRegion.Stack;
            };

            switch (region) {
                .Storage => {
                    // storage always holds i256 values in EVM
                    // if value is i1 (boolean), extend it to i256
                    const value_type = c.oraValueGetType(value);
                    const actual_value = if (c.oraTypeIsAInteger(value_type) and c.oraIntegerTypeGetWidth(value_type) == 1) blk: {
                        // this is i1, need to extend to i256
                        const i256_type = c.oraIntegerTypeCreate(self.ctx, constants.DEFAULT_INTEGER_BITS);
                        const ext_op = c.oraArithExtUIOpCreate(self.ctx, loc, value, i256_type);
                        h.appendOp(self.block, ext_op);
                        break :blk h.getResult(ext_op, 0);
                    } else blk: {
                        break :blk value;
                    };

                    const store_op = self.memory_manager.createStorageStore(actual_value, ident.name, loc);
                    h.appendOp(self.block, store_op);
                    return;
                },
                .Memory => {
                    // for memory variables, we need the memref address
                    if (self.local_var_map) |lvm| {
                        if (lvm.getLocalVar(ident.name)) |memref| {
                            const store_op = self.memory_manager.createStoreOp(value, memref, region, loc);
                            h.appendOp(self.block, store_op);
                            return;
                        }
                    }
                },
                .TStore => {
                    const store_op = self.memory_manager.createTStoreStore(value, ident.name, loc);
                    h.appendOp(self.block, store_op);
                    return;
                },
                .Calldata => {
                    reportAssignmentError(
                        self,
                        ident.span,
                        .InvalidMemoryRegion,
                        "Cannot assign to calldata parameter",
                        "Copy into a local variable if mutation is required.",
                    );
                    return LoweringError.InvalidLValue;
                },
                .Stack => {
                    // for stack variables, check if we have a memref (mutable aggregate var) or direct SSA value (scalar var or let/const)
                    if (self.local_var_map) |lvm| {
                        if (lvm.getLocalVar(ident.name)) |var_value| {
                            // check if this is a memref type (mutable aggregate variable)
                            const var_type = c.oraValueGetType(var_value);
                            if (c.oraTypeIsAMemRef(var_type)) {
                                // it's a memref (aggregate) - ensure value type matches element type
                                const element_type = c.oraShapedTypeGetElementType(var_type);
                                const value_type = c.oraValueGetType(value);
                                log.debug("[ASSIGN Stack] Variable: {s}, value_type != element_type: {}, value_is_ora: {}, element_is_ora: {}\n", .{ ident.name, !c.oraTypeEqual(value_type, element_type), c.oraTypeIsIntegerType(value_type), c.oraTypeIsIntegerType(element_type) });
                                // always convert to ensure type compatibility
                                const store_value = self.expr_lowerer.convertToType(value, element_type, ident.span);
                                const store_value_type = c.oraValueGetType(store_value);
                                log.debug("[ASSIGN Stack] After conversion: types_equal: {}, store_value_is_ora: {}, element_is_ora: {}\n", .{ c.oraTypeEqual(store_value_type, element_type), c.oraTypeIsIntegerType(store_value_type), c.oraTypeIsIntegerType(element_type) });
                                const store_op = self.ora_dialect.createMemrefStore(store_value, var_value, &[_]c.MlirValue{}, loc);
                                h.appendOp(self.block, store_op);
                                return;
                            } else {
                                // it's an SSA value - check if it's mutable (var) or immutable (let/const)
                                // use the variable_kind we stored earlier
                                log.debug("[ASSIGN Stack] SSA value found, checking variable_kind for: {s}\n", .{ident.name});
                                log.debug("[ASSIGN Stack] Symbol variable_kind: {any}, symbol_kind: {any}\n", .{ var_kind, symbol.symbol_kind });

                                // only check variable_kind for variables - parameters and constants are handled differently
                                if (symbol.symbol_kind != .Variable) {
                                    log.debug("[ASSIGN Stack] Symbol is not a variable (kind: {any}), cannot assign\n", .{symbol.symbol_kind});
                                    reportAssignmentError(
                                        self,
                                        ident.span,
                                        .TypeMismatch,
                                        "Cannot assign to non-variable symbol",
                                        "Ensure the assignment target is a mutable variable.",
                                    );
                                    return LoweringError.InvalidLValue;
                                }

                                if (var_kind) |kind| {
                                    log.debug("[ASSIGN Stack] Kind is: {any}\n", .{kind});
                                    if (kind == .Var) {
                                        // scalar mutable variable (Var) should be memref-backed
                                        // if it's not a memref, this is a bug - scalar Var variables should always be memrefs
                                        log.err("Scalar Var variable '{s}' is not memref-backed - this should not happen\n", .{ident.name});
                                        log.debug("  Variable should have been created as memref in lowerStackVariableDecl\n", .{});
                                        reportAssignmentError(
                                            self,
                                            ident.span,
                                            .InternalError,
                                            "Mutable variable is not memref-backed; assignment would break SSA dominance",
                                            "This is a compiler bug. Please report with a minimal repro.",
                                        );
                                        return LoweringError.InvalidLValue;
                                    } else {
                                        // it's an immutable variable (let/const) - can't reassign
                                        log.err("Cannot assign to immutable variable: {s} (kind: {any})\n", .{ ident.name, kind });
                                        reportAssignmentError(
                                            self,
                                            ident.span,
                                            .TypeMismatch,
                                            "Cannot assign to immutable variable",
                                            "Declare the variable with 'var' if it needs to be reassigned.",
                                        );
                                        return LoweringError.InvalidLValue;
                                    }
                                } else {
                                    log.debug("[ASSIGN Stack] variable_kind is null for: {s}\n", .{ident.name});
                                    // no variable_kind - assume immutable for safety
                                    log.err("Cannot assign to immutable variable: {s} (no mutability info)\n", .{ident.name});
                                    reportAssignmentError(
                                        self,
                                        ident.span,
                                        .TypeMismatch,
                                        "Cannot assign to immutable variable",
                                        "Declare the variable with 'var' if it needs to be reassigned.",
                                    );
                                    return LoweringError.InvalidLValue;
                                }
                            }
                        }
                    }
                    // variable not found in map
                    log.err("Variable not found for assignment: {s}\n", .{ident.name});
                    reportAssignmentError(
                        self,
                        ident.span,
                        .UndefinedSymbol,
                        "Assignment target is not defined in the current scope",
                        "Declare the variable before assigning to it.",
                    );
                    return LoweringError.InvalidLValue;
                },
            }
        }
    }

    // fallback: check storage map
    if (self.storage_map) |sm| {
        if (sm.hasStorageVariable(ident.name)) {
            const store_op = self.memory_manager.createStorageStore(value, ident.name, loc);
            h.appendOp(self.block, store_op);
            return;
        }
    }

    // fallback: check local variable map
    if (self.local_var_map) |lvm| {
        if (lvm.getLocalVar(ident.name)) |var_value| {
            const var_type = c.oraValueGetType(var_value);
            if (c.oraTypeIsAMemRef(var_type)) {
                const element_type = c.oraShapedTypeGetElementType(var_type);
                const store_value = self.expr_lowerer.convertToType(value, element_type, ident.span);
                const store_op = self.ora_dialect.createMemrefStore(store_value, var_value, &[_]c.MlirValue{}, loc);
                h.appendOp(self.block, store_op);
                return;
            }
            if (self.force_stack_memref) {
                reportAssignmentError(
                    self,
                    ident.span,
                    .InternalError,
                    "Mutable variable is not memref-backed; assignment would break SSA dominance",
                    "This is a compiler bug. Please report with a minimal repro.",
                );
                return LoweringError.InvalidLValue;
            }
            lvm.addLocalVar(ident.name, value) catch {
                return LoweringError.OutOfMemory;
            };
            return;
        }
    }

    return LoweringError.UndefinedSymbol;
}

/// Lower field access assignments (struct.field = value)
pub fn lowerFieldAccessAssignment(self: *const StatementLowerer, field_access: *const lib.ast.Expressions.FieldAccessExpr, value: c.MlirValue, loc: c.MlirLocation) LoweringError!void {
    // lower the target expression to get the struct
    var target = self.expr_lowerer.lowerExpression(field_access.target);
    var target_type = c.oraValueGetType(target);

    // if the target is a local variable, ensure we have the correct struct type
    // the target should be a struct SSA value, not i256 or address
    if (field_access.target.* == .Identifier) {
        const ident = field_access.target.Identifier;

        // check if we have type info that says this should be a struct
        if (ident.type_info.ora_type) |ora_type| {
            if (ora_type == .struct_type) {
                const expected_struct_type = self.expr_lowerer.type_mapper.toMlirType(ident.type_info);
                const actual_type = c.oraValueGetType(target);

                // check if the actual type is an address (this is wrong - we need a struct)
                if (c.oraTypeIsAddressType(actual_type)) {
                    log.err("Variable '{s}' is address type but should be struct type for field access\n", .{ident.name});
                    log.debug("  This likely means a map load returned !ora.address instead of the struct type\n", .{});
                    log.debug("  Expected struct type: {any}\n", .{expected_struct_type});
                    reportAssignmentError(
                        self,
                        field_access.span,
                        .TypeMismatch,
                        "Cannot assign to field on address-typed value; expected struct value",
                        "Ensure the target expression resolves to a struct type before assigning to a field.",
                    );
                    return LoweringError.TypeMismatch;
                }

                // if the actual type doesn't match the expected struct type, we have a problem
                // this can happen if map loads return i256 instead of struct types
                if (!c.oraTypeEqual(actual_type, expected_struct_type)) {
                    log.err("Variable '{s}' should be struct type but got: {any}\n", .{ ident.name, actual_type });
                    log.debug("  Expected struct type: {any}\n", .{expected_struct_type});
                    log.debug("  This likely means a map load returned wrong type instead of the struct type\n", .{});

                    // try to convert the value to the correct struct type
                    // but this won't work if it's actually i256 - we need to fix the map load
                    // for now, error out with a clear message
                    reportAssignmentError(
                        self,
                        field_access.span,
                        .TypeMismatch,
                        "Cannot assign to field on non-struct value; expected struct type",
                        "Check map/field types and ensure loads return the struct type, not a scalar.",
                    );
                    return LoweringError.TypeMismatch;
                }
            }
        }

        // if the target is stored as memref (shouldn't happen for structs now, but check anyway)
        if (self.local_var_map) |var_map| {
            if (var_map.getLocalVar(ident.name)) |var_value| {
                const var_type = c.oraValueGetType(var_value);
                if (c.oraTypeIsAMemRef(var_type)) {
                    // load the struct from memref
                    const element_type = c.oraShapedTypeGetElementType(var_type);
                    const load_op = c.oraMemrefLoadOpCreate(self.ctx, loc, var_value, null, 0, element_type);
                    h.appendOp(self.block, load_op);
                    target = h.getResult(load_op, 0);
                    target_type = element_type;
                }
            }
        }
    } else {
        // for non-identifier targets (e.g., nested field access or map loads),
        // check if the target is an address type and error if so
        const actual_type = c.oraValueGetType(target);
        if (c.oraTypeIsAddressType(actual_type)) {
            log.err("Field access target is address type but should be struct type\n", .{});
            log.debug("  This likely means a map load returned !ora.address instead of the struct type\n", .{});
            reportAssignmentError(
                self,
                field_access.span,
                .TypeMismatch,
                "Cannot assign to field on address-typed value; expected struct value",
                "Ensure the target expression resolves to a struct type before assigning to a field.",
            );
            return LoweringError.TypeMismatch;
        }
    }

    // look up actual field index from struct definition in symbol table
    var field_idx: i64 = 0;
    var struct_name: ?[]const u8 = null;
    if (self.symbol_table) |st| {
        // iterate through all struct types to find matching field
        var type_iter = st.types.iterator();
        while (type_iter.next()) |entry| {
            const type_symbols = entry.value_ptr.*;
            for (type_symbols) |type_sym| {
                if (type_sym.type_kind == .Struct) {
                    if (type_sym.fields) |fields| {
                        for (fields, 0..) |field, i| {
                            if (std.mem.eql(u8, field.name, field_access.field)) {
                                field_idx = @intCast(i);
                                struct_name = entry.key_ptr.*;
                                break;
                            }
                        }
                    }
                }
            }
        }
    }

    // use ora.struct_field_update (pure operation, works with !ora.struct types)
    // result type is automatically inferred from input struct type (SameOperandsAndResultType trait)
    const update_op = self.ora_dialect.createStructFieldUpdate(target, field_access.field, value, loc);
    h.appendOp(self.block, update_op);
    const updated_struct = h.getResult(update_op, 0);

    if (field_access.target.* == .Identifier) {
        const ident = field_access.target.Identifier;
        if (self.local_var_map) |var_map| {
            if (var_map.getLocalVar(ident.name)) |var_value| {
                const var_type = c.oraValueGetType(var_value);
                if (c.oraTypeIsAMemRef(var_type)) {
                    const element_type = c.oraShapedTypeGetElementType(var_type);
                    const store_value = self.expr_lowerer.convertToType(updated_struct, element_type, field_access.span);
                    const store_op = self.ora_dialect.createMemrefStore(store_value, var_value, &[_]c.MlirValue{}, loc);
                    h.appendOp(self.block, store_op);
                    return;
                }
                if (self.force_stack_memref) {
                    reportAssignmentError(
                        self,
                        field_access.span,
                        .InternalError,
                        "Mutable variable is not memref-backed; assignment would break SSA dominance",
                        "This is a compiler bug. Please report with a minimal repro.",
                    );
                    return LoweringError.InvalidLValue;
                }
            }
            var_map.addLocalVar(ident.name, updated_struct) catch {
                log.warn("Failed to update local variable after field assignment: {s}\n", .{ident.name});
            };
        }
        // also update symbol table
        if (self.symbol_table) |st| {
            st.updateSymbolValue(ident.name, updated_struct) catch {
                log.warn("Failed to update symbol table after field assignment: {s}\n", .{ident.name});
            };
        }
    } else {
        // for complex field access (e.g., nested structs: user.address.street = value)
        // we need to handle this by updating the nested struct, then updating the parent
        // for now, this is a limitation - complex nested field updates need more work
        // but we should NOT use struct_field_store for locals (canonical SSA style)
        log.warn("Complex field access assignment not fully supported for SSA structs: {s}\n", .{field_access.field});
        // todo: Handle nested field updates properly:
        // 1. Extract nested struct from parent
        // 2. Update nested struct field
        // 3. Update parent struct with new nested struct
        // 4. Rebind if parent is a local variable
    }
}

/// Lower array/map index assignments (arr[index] = value)
pub fn lowerIndexAssignment(self: *const StatementLowerer, index_expr: *const lib.ast.Expressions.IndexExpr, value: c.MlirValue, loc: c.MlirLocation) LoweringError!void {
    // lower the target expression to get the array/map
    const target = self.expr_lowerer.lowerExpression(index_expr.target);
    const index_val = self.expr_lowerer.lowerExpression(index_expr.index);
    const target_type = c.oraValueGetType(target);

    // determine the type of indexing operation
    if (c.oraTypeIsAMemRef(target_type)) {
        // array indexing using memref.store
        // convert index to index type for memref operations
        const index_index = self.expr_lowerer.convertIndexToIndexType(index_val, index_expr.span);
        // convert value to match element type (handles refinement types, etc.)
        const element_type = c.oraShapedTypeGetElementType(target_type);
        const store_value = self.expr_lowerer.convertToType(value, element_type, index_expr.span);
        const store_op = c.oraMemrefStoreOpCreate(
            self.ctx,
            loc,
            store_value,
            target,
            &[_]c.MlirValue{index_index},
            1,
        );
        h.appendOp(self.block, store_op);
    } else {
        // map indexing or other complex indexing operations
        // use ora.map_store directly (registered operation)
        const map_store_op = self.ora_dialect.createMapStore(target, index_val, value, loc);
        h.appendOp(self.block, map_store_op);
    }
}

/// Lower destructuring assignment statements with field extraction operations
pub fn lowerDestructuringAssignment(self: *const StatementLowerer, assignment: *const lib.ast.Statements.DestructuringAssignmentNode) LoweringError!void {
    const loc = self.fileLoc(assignment.span);

    // lower the value expression to destructure
    const value = self.expr_lowerer.lowerExpression(assignment.value);

    // handle different destructuring patterns
    try lowerDestructuringPattern(self, assignment.pattern, value, loc);
}

/// Lower destructuring patterns
pub fn lowerDestructuringPattern(self: *const StatementLowerer, pattern: lib.ast.Expressions.DestructuringPattern, value: c.MlirValue, loc: c.MlirLocation) LoweringError!void {
    const dummy_span = lib.ast.SourceSpan{ .line = 0, .column = 0, .length = 0 };
    switch (pattern) {
        .Struct => |fields| {
            // extract each field from the struct value
            for (fields, 0..) |field, i| {
                // create llvm.extractvalue operation for each field
                const result_ty = c.oraIntegerTypeCreate(self.ctx, constants.DEFAULT_INTEGER_BITS);
                const indices = [_]u32{@intCast(i)};
                const extract_op = self.ora_dialect.createLlvmExtractvalue(value, &indices, result_ty, loc);
                h.appendOp(self.block, extract_op);
                const field_value = h.getResult(extract_op, 0);

                if (self.local_var_map) |lvm| {
                    if (lvm.getLocalVar(field.name)) |var_value| {
                        const var_type = c.oraValueGetType(var_value);
                        if (c.oraTypeIsAMemRef(var_type)) {
                            const element_type = c.oraShapedTypeGetElementType(var_type);
                            const store_value = self.expr_lowerer.convertToType(field_value, element_type, dummy_span);
                            const store_op = self.ora_dialect.createMemrefStore(store_value, var_value, &[_]c.MlirValue{}, loc);
                            h.appendOp(self.block, store_op);
                            continue;
                        }
                        if (self.force_stack_memref) {
                            reportAssignmentError(
                                self,
                                dummy_span,
                                .InternalError,
                                "Mutable variable is not memref-backed; assignment would break SSA dominance",
                                "This is a compiler bug. Please report with a minimal repro.",
                            );
                            return LoweringError.InvalidLValue;
                        }
                    }
                    lvm.addLocalVar(field.name, field_value) catch {
                        log.err("Failed to add destructured field to map: {s}\n", .{field.name});
                        return LoweringError.OutOfMemory;
                    };
                }

                if (self.symbol_table) |st| {
                    st.updateSymbolValue(field.name, field_value) catch {
                        log.warn("Failed to update symbol for destructured field: {s}\n", .{field.name});
                    };
                }
            }
        },
        .Tuple => |elements| {
            // similar to struct destructuring but for tuple elements
            for (elements, 0..) |element_name, i| {
                // create llvm.extractvalue operation for each tuple element
                const result_ty = c.oraIntegerTypeCreate(self.ctx, constants.DEFAULT_INTEGER_BITS);
                const indices = [_]u32{@intCast(i)};
                const extract_op = self.ora_dialect.createLlvmExtractvalue(value, &indices, result_ty, loc);
                h.appendOp(self.block, extract_op);
                const element_value = h.getResult(extract_op, 0);

                if (self.local_var_map) |lvm| {
                    if (lvm.getLocalVar(element_name)) |var_value| {
                        const var_type = c.oraValueGetType(var_value);
                        if (c.oraTypeIsAMemRef(var_type)) {
                            const element_type = c.oraShapedTypeGetElementType(var_type);
                            const store_value = self.expr_lowerer.convertToType(element_value, element_type, dummy_span);
                            const store_op = self.ora_dialect.createMemrefStore(store_value, var_value, &[_]c.MlirValue{}, loc);
                            h.appendOp(self.block, store_op);
                            continue;
                        }
                        if (self.force_stack_memref) {
                            reportAssignmentError(
                                self,
                                dummy_span,
                                .InternalError,
                                "Mutable variable is not memref-backed; assignment would break SSA dominance",
                                "This is a compiler bug. Please report with a minimal repro.",
                            );
                            return LoweringError.InvalidLValue;
                        }
                    }
                    lvm.addLocalVar(element_name, element_value) catch {
                        log.err("Failed to add destructured element to map: {s}\n", .{element_name});
                        return LoweringError.OutOfMemory;
                    };
                }

                if (self.symbol_table) |st| {
                    st.updateSymbolValue(element_name, element_value) catch {
                        log.warn("Failed to update symbol for destructured element: {s}\n", .{element_name});
                    };
                }
            }
        },
        .Array => |elements| {
            // extract each element from the array value
            for (elements, 0..) |element_name, i| {
                // create memref.load operation for each array element
                // first, create index constant
                const index_ty = c.oraIntegerTypeCreate(self.ctx, constants.DEFAULT_INTEGER_BITS);
                const index_op = self.ora_dialect.createArithConstant(@intCast(i), index_ty, loc);
                h.appendOp(self.block, index_op);
                const index_value = h.getResult(index_op, 0);

                // create memref.load operation
                const result_ty = c.oraIntegerTypeCreate(self.ctx, constants.DEFAULT_INTEGER_BITS);
                const load_op = c.oraMemrefLoadOpCreate(
                    self.ctx,
                    loc,
                    value,
                    &[_]c.MlirValue{index_value},
                    1,
                    result_ty,
                );
                h.appendOp(self.block, load_op);
                const element_value = h.getResult(load_op, 0);

                if (self.local_var_map) |lvm| {
                    if (lvm.getLocalVar(element_name)) |var_value| {
                        const var_type = c.oraValueGetType(var_value);
                        if (c.oraTypeIsAMemRef(var_type)) {
                            const element_type = c.oraShapedTypeGetElementType(var_type);
                            const store_value = self.expr_lowerer.convertToType(element_value, element_type, dummy_span);
                            const store_op = self.ora_dialect.createMemrefStore(store_value, var_value, &[_]c.MlirValue{}, loc);
                            h.appendOp(self.block, store_op);
                            continue;
                        }
                        if (self.force_stack_memref) {
                            reportAssignmentError(
                                self,
                                dummy_span,
                                .InternalError,
                                "Mutable variable is not memref-backed; assignment would break SSA dominance",
                                "This is a compiler bug. Please report with a minimal repro.",
                            );
                            return LoweringError.InvalidLValue;
                        }
                    }
                    lvm.addLocalVar(element_name, element_value) catch {
                        log.err("Failed to add destructured element to map: {s}\n", .{element_name});
                        return LoweringError.OutOfMemory;
                    };
                }

                if (self.symbol_table) |st| {
                    st.updateSymbolValue(element_name, element_value) catch {
                        log.warn("Failed to update symbol for destructured element: {s}\n", .{element_name});
                    };
                }
            }
        },
    }
}

/// Lower expression-level compound assignment expressions
pub fn lowerCompoundAssignmentExpr(self: *const StatementLowerer, assignment: *const lib.ast.Expressions.CompoundAssignmentExpr) LoweringError!void {
    // compound assignments delegated to expression lowering
    // expression-level handling managed by binary operation lowering
    _ = self;
    _ = assignment;
}

/// Lower compound assignment statements
pub fn lowerCompoundAssignment(self: *const StatementLowerer, assignment: *const lib.ast.Statements.CompoundAssignmentNode) LoweringError!void {
    // compound assignment to storage variables
    // complex target expressions (field access, indices) handled by lowerAssignableExpression
    if (assignment.target.* == .Identifier) {
        const ident = assignment.target.Identifier;

        if (self.storage_map) |sm| {
            _ = sm; // Use the variable to avoid warning

            // define result type for arithmetic operations
            const result_ty = c.oraIntegerTypeCreate(self.ctx, constants.DEFAULT_INTEGER_BITS);

            // load current value from storage using ora.sload
            const memory_manager = MemoryManager.init(self.ctx, self.ora_dialect);
            const load_op = memory_manager.createStorageLoad(ident.name, result_ty, self.fileLoc(ident.span));
            h.appendOp(self.block, load_op);
            const current_value = h.getResult(load_op, 0);

            // lower the right-hand side expression
            const rhs_value = self.expr_lowerer.lowerExpression(assignment.value);

            // perform the compound operation
            var new_value: c.MlirValue = undefined;
            switch (assignment.operator) {
                .PlusEqual => {
                    // current_value + rhs_value
                    const add_op = self.ora_dialect.createArithAddi(current_value, rhs_value, result_ty, self.fileLoc(ident.span));
                    h.appendOp(self.block, add_op);
                    new_value = h.getResult(add_op, 0);
                },
                .MinusEqual => {
                    // current_value - rhs_value
                    const sub_op = self.ora_dialect.createArithSubi(current_value, rhs_value, result_ty, self.fileLoc(ident.span));
                    h.appendOp(self.block, sub_op);
                    new_value = h.getResult(sub_op, 0);
                },
                .StarEqual => {
                    // current_value * rhs_value
                    const mul_op = self.ora_dialect.createArithMuli(current_value, rhs_value, result_ty, self.fileLoc(ident.span));
                    h.appendOp(self.block, mul_op);
                    new_value = h.getResult(mul_op, 0);
                },
                .SlashEqual => {
                    // current_value / rhs_value
                    const div_op = self.ora_dialect.createArithDivsi(current_value, rhs_value, result_ty, self.fileLoc(ident.span));
                    h.appendOp(self.block, div_op);
                    new_value = h.getResult(div_op, 0);
                },
                .PercentEqual => {
                    // current_value % rhs_value
                    const rem_op = self.ora_dialect.createArithRemsi(current_value, rhs_value, result_ty, self.fileLoc(ident.span));
                    h.appendOp(self.block, rem_op);
                    new_value = h.getResult(rem_op, 0);
                },
            }

            // store the result back to storage using ora.sstore
            const store_op = memory_manager.createStorageStore(new_value, ident.name, self.fileLoc(ident.span));
            h.appendOp(self.block, store_op);
        } else {
            log.err("Compound assignment lowering for storage maps is unimplemented\n", .{});
            @panic("compound assignment lowering not implemented");
        }
    } else {
        // for now, skip non-identifier compound assignments
    }
}

/// Helper function to get expression span
fn getExpressionSpan(expr: *const lib.ast.Expressions.ExprNode) lib.ast.SourceSpan {
    switch (expr.*) {
        .Identifier => |ident| return ident.span,
        .FieldAccess => |field| return field.span,
        .Index => |index| return index.span,
        .Binary => |bin| return bin.span,
        .Unary => |unary| return unary.span,
        .Call => |call| return call.span,
        .Literal => |lit| {
            switch (lit) {
                .Integer => |int_lit| return int_lit.span,
                .String => |str_lit| return str_lit.span,
                .Bool => |bool_lit| return bool_lit.span,
                .Address => |addr_lit| return addr_lit.span,
                .Hex => |hex_lit| return hex_lit.span,
                .Binary => |bin_lit| return bin_lit.span,
                .Character => |char_lit| return char_lit.span,
                .Bytes => |bytes_lit| return bytes_lit.span,
            }
        },
        .Cast => |cast| return cast.span,
        .Assignment => |assign| return assign.span,
        .CompoundAssignment => |compound| return compound.span,
        .Comptime => |comptime_expr| return comptime_expr.span,
        .Old => |old| return old.span,
        .Tuple => |tuple| return tuple.span,
        else => {
            // default span for unknown expressions
            return lib.ast.SourceSpan{ .line = 0, .column = 0, .length = 0 };
        },
    }
}
