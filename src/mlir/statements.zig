const std = @import("std");
const c = @import("c.zig").c;
const lib = @import("ora_lib");
const constants = @import("constants.zig");
const TypeMapper = @import("types.zig").TypeMapper;
const ExpressionLowerer = @import("expressions.zig").ExpressionLowerer;
const ParamMap = @import("symbols.zig").ParamMap;
const StorageMap = @import("memory.zig").StorageMap;
const LocalVarMap = @import("symbols.zig").LocalVarMap;
const LocationTracker = @import("locations.zig").LocationTracker;
const MemoryManager = @import("memory.zig").MemoryManager;
const SymbolTable = @import("symbols.zig").SymbolTable;

/// Statement lowering system for converting Ora statements to MLIR operations
pub const StatementLowerer = struct {
    ctx: c.MlirContext,
    block: c.MlirBlock,
    type_mapper: *const TypeMapper,
    expr_lowerer: *const ExpressionLowerer,
    param_map: ?*const ParamMap,
    storage_map: ?*const StorageMap,
    local_var_map: ?*LocalVarMap,
    locations: LocationTracker,
    symbol_table: ?*SymbolTable,
    memory_manager: MemoryManager,
    allocator: std.mem.Allocator,

    pub fn init(ctx: c.MlirContext, block: c.MlirBlock, type_mapper: *const TypeMapper, expr_lowerer: *const ExpressionLowerer, param_map: ?*const ParamMap, storage_map: ?*const StorageMap, local_var_map: ?*LocalVarMap, locations: LocationTracker, symbol_table: ?*SymbolTable, allocator: std.mem.Allocator) StatementLowerer {
        return .{
            .ctx = ctx,
            .block = block,
            .type_mapper = type_mapper,
            .expr_lowerer = expr_lowerer,
            .param_map = param_map,
            .storage_map = storage_map,
            .local_var_map = local_var_map,
            .locations = locations,
            .symbol_table = symbol_table,
            .memory_manager = MemoryManager.init(ctx),
            .allocator = allocator,
        };
    }

    /// Main dispatch function for lowering statements
    /// This is the central entry point for all statement lowering
    pub fn lowerStatement(self: *const StatementLowerer, stmt: *const lib.ast.Statements.StmtNode) LoweringError!void {
        // Attach location information to all operations
        _ = self.fileLoc(self.getStatementSpan(stmt));

        switch (stmt.*) {
            .Return => |ret| {
                try self.lowerReturn(&ret);
            },
            .VariableDecl => |var_decl| {
                try self.lowerVariableDecl(&var_decl);
            },
            .DestructuringAssignment => |assignment| {
                try self.lowerDestructuringAssignment(&assignment);
            },
            .CompoundAssignment => |assignment| {
                try self.lowerCompoundAssignment(&assignment);
            },
            .If => |if_stmt| {
                try self.lowerIf(&if_stmt);
            },
            .While => |while_stmt| {
                try self.lowerWhile(&while_stmt);
            },
            .ForLoop => |for_stmt| {
                try self.lowerFor(&for_stmt);
            },
            .Switch => |switch_stmt| {
                try self.lowerSwitch(&switch_stmt);
            },
            .Break => |break_stmt| {
                try self.lowerBreak(&break_stmt);
            },
            .Continue => |continue_stmt| {
                try self.lowerContinue(&continue_stmt);
            },
            .Log => |log_stmt| {
                try self.lowerLog(&log_stmt);
            },
            .Lock => |lock_stmt| {
                try self.lowerLock(&lock_stmt);
            },
            .Unlock => |unlock_stmt| {
                try self.lowerUnlock(&unlock_stmt);
            },
            .Move => |move_stmt| {
                try self.lowerMove(&move_stmt);
            },
            .TryBlock => |try_stmt| {
                try self.lowerTryBlock(&try_stmt);
            },
            .ErrorDecl => |error_decl| {
                try self.lowerErrorDecl(&error_decl);
            },
            .Invariant => |invariant| {
                try self.lowerInvariant(&invariant);
            },
            .Requires => |requires| {
                try self.lowerRequires(&requires);
            },
            .Ensures => |ensures| {
                try self.lowerEnsures(&ensures);
            },
            .Expr => |expr| {
                try self.lowerExpressionStatement(&expr);
            },
            .LabeledBlock => |labeled_block| {
                try self.lowerLabeledBlock(&labeled_block);
            },
        }
    }

    /// Error types for statement lowering
    pub const LoweringError = error{
        UnsupportedStatement,
        TypeMismatch,
        UndefinedSymbol,
        InvalidMemoryRegion,
        MalformedExpression,
        MlirOperationFailed,
        OutOfMemory,
        InvalidControlFlow,
        InvalidLValue,
    };

    /// Get the source span for any statement type
    fn getStatementSpan(_: *const StatementLowerer, stmt: *const lib.ast.Statements.StmtNode) lib.ast.SourceSpan {
        return switch (stmt.*) {
            .Return => |ret| ret.span,
            .VariableDecl => |var_decl| var_decl.span,
            .DestructuringAssignment => |assignment| assignment.span,
            .CompoundAssignment => |assignment| assignment.span,
            .If => |if_stmt| if_stmt.span,
            .While => |while_stmt| while_stmt.span,
            .ForLoop => |for_stmt| for_stmt.span,
            .Switch => |switch_stmt| switch_stmt.span,
            .Break => |break_stmt| break_stmt.span,
            .Continue => |continue_stmt| continue_stmt.span,
            .Log => |log_stmt| log_stmt.span,
            .Lock => |lock_stmt| lock_stmt.span,
            .Unlock => |unlock_stmt| unlock_stmt.span,
            .Move => |move_stmt| move_stmt.span,
            .TryBlock => |try_stmt| try_stmt.span,
            .ErrorDecl => |error_decl| error_decl.span,
            .Invariant => |invariant| invariant.span,
            .Requires => |requires| requires.span,
            .Ensures => |ensures| ensures.span,
            .Expr => |expr| getExpressionSpan(&expr),
            .LabeledBlock => |labeled_block| labeled_block.span,
        };
    }

    /// Get the source span for any expression type (helper for expression statements)
    fn getExpressionSpan(_: *const lib.ast.Statements.ExprNode) lib.ast.SourceSpan {
        // This would need to be implemented based on the expression AST structure
        // For now, return a default span
        return lib.ast.SourceSpan{ .line = 1, .column = 1, .length = 0 };
    }

    /// Lower return statements using func.return with proper value handling
    pub fn lowerReturn(self: *const StatementLowerer, ret: *const lib.ast.Statements.ReturnNode) LoweringError!void {
        const loc = self.fileLoc(ret.span);
        var state = c.mlirOperationStateGet(c.mlirStringRefCreateFromCString("func.return"), loc);

        if (ret.value) |e| {
            const v = self.expr_lowerer.lowerExpression(&e);
            c.mlirOperationStateAddOperands(&state, 1, @ptrCast(&v));
        }

        const op = c.mlirOperationCreate(&state);
        c.mlirBlockAppendOwnedOperation(self.block, op);
    }

    /// Lower break statements with label support using appropriate control flow transfers
    pub fn lowerBreak(self: *const StatementLowerer, break_stmt: *const lib.ast.Statements.BreakNode) LoweringError!void {
        const loc = self.fileLoc(break_stmt.span);

        if (break_stmt.label) |label| {
            // Labeled break - use cf.br with label reference
            var state = c.mlirOperationStateGet(c.mlirStringRefCreateFromCString("cf.br"), loc);

            // Add label as attribute
            const label_ref = c.mlirStringRefCreate(label.ptr, label.len);
            const label_attr = c.mlirStringAttrGet(self.ctx, label_ref);
            const label_id = c.mlirIdentifierGet(self.ctx, c.mlirStringRefCreateFromCString("label"));
            var attrs = [_]c.MlirNamedAttribute{c.mlirNamedAttributeGet(label_id, label_attr)};
            c.mlirOperationStateAddAttributes(&state, attrs.len, &attrs);

            const op = c.mlirOperationCreate(&state);
            c.mlirBlockAppendOwnedOperation(self.block, op);
        } else {
            // Unlabeled break - use scf.break or cf.br depending on context
            var state = c.mlirOperationStateGet(c.mlirStringRefCreateFromCString("scf.break"), loc);

            // Add break value if present
            if (break_stmt.value) |value_expr| {
                const value = self.expr_lowerer.lowerExpression(value_expr);
                c.mlirOperationStateAddOperands(&state, 1, @ptrCast(&value));
            }

            const op = c.mlirOperationCreate(&state);
            c.mlirBlockAppendOwnedOperation(self.block, op);
        }
    }

    /// Lower continue statements with label support using loop continuation operations
    pub fn lowerContinue(self: *const StatementLowerer, continue_stmt: *const lib.ast.Statements.ContinueNode) LoweringError!void {
        const loc = self.fileLoc(continue_stmt.span);

        if (continue_stmt.label) |label| {
            // Labeled continue - use cf.br with label reference
            var state = c.mlirOperationStateGet(c.mlirStringRefCreateFromCString("cf.br"), loc);

            // Add label as attribute
            const label_ref = c.mlirStringRefCreate(label.ptr, label.len);
            const label_attr = c.mlirStringAttrGet(self.ctx, label_ref);
            const label_id = c.mlirIdentifierGet(self.ctx, c.mlirStringRefCreateFromCString("label"));
            var attrs = [_]c.MlirNamedAttribute{c.mlirNamedAttributeGet(label_id, label_attr)};
            c.mlirOperationStateAddAttributes(&state, attrs.len, &attrs);

            const op = c.mlirOperationCreate(&state);
            c.mlirBlockAppendOwnedOperation(self.block, op);
        } else {
            // Unlabeled continue - use scf.continue
            var state = c.mlirOperationStateGet(c.mlirStringRefCreateFromCString("scf.continue"), loc);

            // Add continue value if present (for labeled switch continue)
            if (continue_stmt.value) |value_expr| {
                const value = self.expr_lowerer.lowerExpression(value_expr);
                c.mlirOperationStateAddOperands(&state, 1, @ptrCast(&value));
            }

            const op = c.mlirOperationCreate(&state);
            c.mlirBlockAppendOwnedOperation(self.block, op);
        }
    }

    /// Lower variable declaration statements with proper memory region handling
    pub fn lowerVariableDecl(self: *const StatementLowerer, var_decl: *const lib.ast.Statements.VariableDeclNode) LoweringError!void {
        const loc = self.fileLoc(var_decl.span);

        // Map Ora type to MLIR type
        const mlir_type = self.type_mapper.toMlirType(var_decl.type_info);

        // Add symbol to symbol table if available
        if (self.symbol_table) |st| {
            st.addSymbol(var_decl.name, mlir_type, var_decl.region, null) catch {
                std.debug.print("ERROR: Failed to add symbol to table: {s}\n", .{var_decl.name});
                return LoweringError.OutOfMemory;
            };
        }

        // Handle variable declarations based on memory region
        switch (var_decl.region) {
            .Stack => {
                try self.lowerStackVariableDecl(var_decl, mlir_type, loc);
            },
            .Storage => {
                try self.lowerStorageVariableDecl(var_decl, mlir_type, loc);
            },
            .Memory => {
                try self.lowerMemoryVariableDecl(var_decl, mlir_type, loc);
            },
            .TStore => {
                try self.lowerTStoreVariableDecl(var_decl, mlir_type, loc);
            },
        }
    }

    /// Lower stack variable declarations (local variables)
    fn lowerStackVariableDecl(self: *const StatementLowerer, var_decl: *const lib.ast.Statements.VariableDeclNode, mlir_type: c.MlirType, loc: c.MlirLocation) LoweringError!void {
        var init_value: c.MlirValue = undefined;

        if (var_decl.value) |init_expr| {
            // Lower the initializer expression
            init_value = self.expr_lowerer.lowerExpression(&init_expr.*);
        } else {
            // Create default value based on variable kind
            init_value = try self.createDefaultValue(mlir_type, var_decl.kind, loc);
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

    /// Lower storage variable declarations
    fn lowerStorageVariableDecl(self: *const StatementLowerer, var_decl: *const lib.ast.Statements.VariableDeclNode, _: c.MlirType, loc: c.MlirLocation) LoweringError!void {
        // Storage variables are typically handled at the contract level
        // If there's an initializer, we need to generate a store operation
        if (var_decl.value) |init_expr| {
            const init_value = self.expr_lowerer.lowerExpression(&init_expr.*);

            // Generate storage store operation
            const store_op = self.memory_manager.createStorageStore(init_value, var_decl.name, loc);
            c.mlirBlockAppendOwnedOperation(self.block, store_op);
        }

        // Ensure storage variable is registered
        if (self.storage_map) |sm| {
            _ = @constCast(sm).addStorageVariable(var_decl.name, var_decl.span) catch {
                std.debug.print("WARNING: Failed to register storage variable: {s}\n", .{var_decl.name});
            };
        }
    }

    /// Lower memory variable declarations
    fn lowerMemoryVariableDecl(self: *const StatementLowerer, var_decl: *const lib.ast.Statements.VariableDeclNode, mlir_type: c.MlirType, loc: c.MlirLocation) LoweringError!void {
        // Create memory allocation
        const alloca_op = self.memory_manager.createAllocaOp(mlir_type, var_decl.region, var_decl.name, loc);
        c.mlirBlockAppendOwnedOperation(self.block, alloca_op);
        const alloca_result = c.mlirOperationGetResult(alloca_op, 0);

        if (var_decl.value) |init_expr| {
            // Lower initializer and store to memory
            const init_value = self.expr_lowerer.lowerExpression(&init_expr.*);

            const store_op = self.memory_manager.createStoreOp(init_value, alloca_result, var_decl.region, loc);
            c.mlirBlockAppendOwnedOperation(self.block, store_op);
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
    fn lowerTStoreVariableDecl(self: *const StatementLowerer, var_decl: *const lib.ast.Statements.VariableDeclNode, _: c.MlirType, loc: c.MlirLocation) LoweringError!void {
        // Transient storage variables are similar to storage but temporary
        if (var_decl.value) |init_expr| {
            const init_value = self.expr_lowerer.lowerExpression(&init_expr.*);

            // Generate transient storage store operation
            const store_op = self.memory_manager.createTStoreStore(init_value, var_decl.name, loc);
            c.mlirBlockAppendOwnedOperation(self.block, store_op);
        }
    }

    /// Create default value for uninitialized variables
    fn createDefaultValue(self: *const StatementLowerer, mlir_type: c.MlirType, kind: lib.ast.Statements.VariableKind, loc: c.MlirLocation) LoweringError!c.MlirValue {
        _ = kind; // Variable kind might affect default value in the future

        // For now, create zero value for integer types
        var const_state = c.mlirOperationStateGet(c.mlirStringRefCreateFromCString("arith.constant"), loc);
        c.mlirOperationStateAddResults(&const_state, 1, @ptrCast(&mlir_type));

        const attr = c.mlirIntegerAttrGet(mlir_type, 0);
        const value_id = c.mlirIdentifierGet(self.ctx, c.mlirStringRefCreateFromCString("value"));
        var attrs = [_]c.MlirNamedAttribute{c.mlirNamedAttributeGet(value_id, attr)};
        c.mlirOperationStateAddAttributes(&const_state, attrs.len, &attrs);

        const const_op = c.mlirOperationCreate(&const_state);
        c.mlirBlockAppendOwnedOperation(self.block, const_op);

        return c.mlirOperationGetResult(const_op, 0);
    }

    /// Lower destructuring assignment statements with field extraction operations
    pub fn lowerDestructuringAssignment(self: *const StatementLowerer, assignment: *const lib.ast.Statements.DestructuringAssignmentNode) LoweringError!void {
        const loc = self.fileLoc(assignment.span);

        // Lower the value expression to destructure
        const value = self.expr_lowerer.lowerExpression(assignment.value);

        // Handle different destructuring patterns
        try self.lowerDestructuringPattern(assignment.pattern, value, loc);
    }

    /// Lower destructuring patterns
    fn lowerDestructuringPattern(self: *const StatementLowerer, pattern: lib.ast.Expressions.DestructuringPattern, value: c.MlirValue, loc: c.MlirLocation) LoweringError!void {
        switch (pattern) {
            .Struct => |fields| {
                // Extract each field from the struct value
                for (fields, 0..) |field, i| {
                    // Create llvm.extractvalue operation for each field
                    var extract_state = c.mlirOperationStateGet(c.mlirStringRefCreateFromCString("llvm.extractvalue"), loc);
                    c.mlirOperationStateAddOperands(&extract_state, 1, @ptrCast(&value));

                    // Add field index as attribute
                    const index_ty = c.mlirIntegerTypeGet(self.ctx, 32);
                    const index_attr = c.mlirIntegerAttrGet(index_ty, @intCast(i));
                    const index_id = c.mlirIdentifierGet(self.ctx, c.mlirStringRefCreateFromCString("position"));
                    var attrs = [_]c.MlirNamedAttribute{c.mlirNamedAttributeGet(index_id, index_attr)};
                    c.mlirOperationStateAddAttributes(&extract_state, attrs.len, &attrs);

                    // Add result type (for now, use default integer type)
                    const result_ty = c.mlirIntegerTypeGet(self.ctx, constants.DEFAULT_INTEGER_BITS);
                    c.mlirOperationStateAddResults(&extract_state, 1, @ptrCast(&result_ty));

                    const extract_op = c.mlirOperationCreate(&extract_state);
                    c.mlirBlockAppendOwnedOperation(self.block, extract_op);
                    const field_value = c.mlirOperationGetResult(extract_op, 0);

                    // Assign the extracted value to the field variable
                    if (self.local_var_map) |lvm| {
                        lvm.addLocalVar(field.name, field_value) catch {
                            std.debug.print("ERROR: Failed to add destructured field to map: {s}\n", .{field.name});
                            return LoweringError.OutOfMemory;
                        };
                    }

                    // Update symbol table
                    if (self.symbol_table) |st| {
                        st.updateSymbolValue(field.name, field_value) catch {
                            std.debug.print("WARNING: Failed to update symbol for destructured field: {s}\n", .{field.name});
                        };
                    }
                }
            },
            .Tuple => |elements| {
                // Similar to struct destructuring but for tuple elements
                for (elements, 0..) |element_name, i| {
                    // Create llvm.extractvalue operation for each tuple element
                    var extract_state = c.mlirOperationStateGet(c.mlirStringRefCreateFromCString("llvm.extractvalue"), loc);
                    c.mlirOperationStateAddOperands(&extract_state, 1, @ptrCast(&value));

                    // Add element index as attribute
                    const index_ty = c.mlirIntegerTypeGet(self.ctx, 32);
                    const index_attr = c.mlirIntegerAttrGet(index_ty, @intCast(i));
                    const index_id = c.mlirIdentifierGet(self.ctx, c.mlirStringRefCreateFromCString("position"));
                    var attrs = [_]c.MlirNamedAttribute{c.mlirNamedAttributeGet(index_id, index_attr)};
                    c.mlirOperationStateAddAttributes(&extract_state, attrs.len, &attrs);

                    // Add result type
                    const result_ty = c.mlirIntegerTypeGet(self.ctx, constants.DEFAULT_INTEGER_BITS);
                    c.mlirOperationStateAddResults(&extract_state, 1, @ptrCast(&result_ty));

                    const extract_op = c.mlirOperationCreate(&extract_state);
                    c.mlirBlockAppendOwnedOperation(self.block, extract_op);
                    const element_value = c.mlirOperationGetResult(extract_op, 0);

                    // Assign the extracted value to the element variable
                    if (self.local_var_map) |lvm| {
                        lvm.addLocalVar(element_name, element_value) catch {
                            std.debug.print("ERROR: Failed to add destructured element to map: {s}\n", .{element_name});
                            return LoweringError.OutOfMemory;
                        };
                    }

                    // Update symbol table
                    if (self.symbol_table) |st| {
                        st.updateSymbolValue(element_name, element_value) catch {
                            std.debug.print("WARNING: Failed to update symbol for destructured element: {s}\n", .{element_name});
                        };
                    }
                }
            },
            .Array => |elements| {
                // Extract each element from the array value
                for (elements, 0..) |element_name, i| {
                    // Create memref.load operation for each array element
                    // First, create index constant
                    const index_ty = c.mlirIntegerTypeGet(self.ctx, constants.DEFAULT_INTEGER_BITS);
                    var index_state = c.mlirOperationStateGet(c.mlirStringRefCreateFromCString("arith.constant"), loc);
                    c.mlirOperationStateAddResults(&index_state, 1, @ptrCast(&index_ty));
                    const index_attr = c.mlirIntegerAttrGet(index_ty, @intCast(i));
                    const value_id = c.mlirIdentifierGet(self.ctx, c.mlirStringRefCreateFromCString("value"));
                    var index_attrs = [_]c.MlirNamedAttribute{c.mlirNamedAttributeGet(value_id, index_attr)};
                    c.mlirOperationStateAddAttributes(&index_state, index_attrs.len, &index_attrs);
                    const index_op = c.mlirOperationCreate(&index_state);
                    c.mlirBlockAppendOwnedOperation(self.block, index_op);
                    const index_value = c.mlirOperationGetResult(index_op, 0);

                    // Create memref.load operation
                    var load_state = c.mlirOperationStateGet(c.mlirStringRefCreateFromCString("memref.load"), loc);
                    const operands = [_]c.MlirValue{ value, index_value };
                    c.mlirOperationStateAddOperands(&load_state, operands.len, &operands);

                    // Add result type
                    const result_ty = c.mlirIntegerTypeGet(self.ctx, constants.DEFAULT_INTEGER_BITS);
                    c.mlirOperationStateAddResults(&load_state, 1, @ptrCast(&result_ty));

                    const load_op = c.mlirOperationCreate(&load_state);
                    c.mlirBlockAppendOwnedOperation(self.block, load_op);
                    const element_value = c.mlirOperationGetResult(load_op, 0);

                    // Assign the extracted value to the element variable
                    if (self.local_var_map) |lvm| {
                        lvm.addLocalVar(element_name, element_value) catch {
                            std.debug.print("ERROR: Failed to add destructured element to map: {s}\n", .{element_name});
                            return LoweringError.OutOfMemory;
                        };
                    }

                    // Update symbol table
                    if (self.symbol_table) |st| {
                        st.updateSymbolValue(element_name, element_value) catch {
                            std.debug.print("WARNING: Failed to update symbol for destructured element: {s}\n", .{element_name});
                        };
                    }
                }
            },
        }
    }

    /// Lower expression-level compound assignment expressions
    pub fn lowerCompoundAssignmentExpr(self: *const StatementLowerer, assignment: *const lib.ast.Expressions.CompoundAssignmentExpr) LoweringError!void {
        // Debug: print what we're compound assigning to
        std.debug.print("DEBUG: Compound assignment to expression\n", .{});

        // For now, just skip expression-level compound assignments
        // TODO: Implement proper expression-level compound assignment handling
        _ = self; // Use self parameter
        _ = assignment.target; // Use the parameter to avoid warning
        _ = assignment.operator; // Use the parameter to avoid warning
        _ = assignment.value; // Use the parameter to avoid warning
        _ = assignment.span; // Use the parameter to avoid warning
    }

    /// Lower compound assignment statements
    pub fn lowerCompoundAssignment(self: *const StatementLowerer, assignment: *const lib.ast.Statements.CompoundAssignmentNode) LoweringError!void {
        // Debug: print what we're compound assigning to
        std.debug.print("DEBUG: Compound assignment to expression\n", .{});

        // Handle compound assignment to storage variables
        // For now, we'll assume the target is an identifier expression
        // TODO: Handle more complex target expressions
        if (assignment.target.* == .Identifier) {
            const ident = assignment.target.Identifier;
            std.debug.print("DEBUG: Would compound assign to storage variable: {s}\n", .{ident.name});

            if (self.storage_map) |sm| {
                // Ensure the variable exists in storage (create if needed)
                // TODO: Fix const qualifier issue - getOrCreateAddress expects mutable pointer
                // _ = sm.getOrCreateAddress(ident.name) catch 0;
                _ = sm; // Use the variable to avoid warning

                // Define result type for arithmetic operations
                const result_ty = c.mlirIntegerTypeGet(self.ctx, constants.DEFAULT_INTEGER_BITS);

                // Load current value from storage using ora.sload
                const memory_manager = MemoryManager.init(self.ctx);
                const load_op = memory_manager.createStorageLoad(ident.name, result_ty, self.fileLoc(ident.span));
                c.mlirBlockAppendOwnedOperation(self.block, load_op);
                const current_value = c.mlirOperationGetResult(load_op, 0);

                // Lower the right-hand side expression
                const rhs_value = self.expr_lowerer.lowerExpression(assignment.value);

                // Perform the compound operation
                var new_value: c.MlirValue = undefined;
                switch (assignment.operator) {
                    .PlusEqual => {
                        // current_value + rhs_value
                        var add_state = c.mlirOperationStateGet(c.mlirStringRefCreateFromCString("arith.addi"), self.fileLoc(ident.span));
                        c.mlirOperationStateAddOperands(&add_state, 2, @ptrCast(&[_]c.MlirValue{ current_value, rhs_value }));
                        c.mlirOperationStateAddResults(&add_state, 1, @ptrCast(&result_ty));
                        const add_op = c.mlirOperationCreate(&add_state);
                        c.mlirBlockAppendOwnedOperation(self.block, add_op);
                        new_value = c.mlirOperationGetResult(add_op, 0);
                    },
                    .MinusEqual => {
                        // current_value - rhs_value
                        var sub_state = c.mlirOperationStateGet(c.mlirStringRefCreateFromCString("arith.subi"), self.fileLoc(ident.span));
                        c.mlirOperationStateAddOperands(&sub_state, 2, @ptrCast(&[_]c.MlirValue{ current_value, rhs_value }));
                        c.mlirOperationStateAddResults(&sub_state, 1, @ptrCast(&result_ty));
                        const sub_op = c.mlirOperationCreate(&sub_state);
                        c.mlirBlockAppendOwnedOperation(self.block, sub_op);
                        new_value = c.mlirOperationGetResult(sub_op, 0);
                    },
                    .StarEqual => {
                        // current_value * rhs_value
                        var mul_state = c.mlirOperationStateGet(c.mlirStringRefCreateFromCString("arith.muli"), self.fileLoc(ident.span));
                        c.mlirOperationStateAddOperands(&mul_state, 2, @ptrCast(&[_]c.MlirValue{ current_value, rhs_value }));
                        c.mlirOperationStateAddResults(&mul_state, 1, @ptrCast(&result_ty));
                        const mul_op = c.mlirOperationCreate(&mul_state);
                        c.mlirBlockAppendOwnedOperation(self.block, mul_op);
                        new_value = c.mlirOperationGetResult(mul_op, 0);
                    },
                    .SlashEqual => {
                        // current_value / rhs_value
                        var div_state = c.mlirOperationStateGet(c.mlirStringRefCreateFromCString("arith.divsi"), self.fileLoc(ident.span));
                        c.mlirOperationStateAddOperands(&div_state, 2, @ptrCast(&[_]c.MlirValue{ current_value, rhs_value }));
                        c.mlirOperationStateAddResults(&div_state, 1, @ptrCast(&result_ty));
                        const div_op = c.mlirOperationCreate(&div_state);
                        c.mlirBlockAppendOwnedOperation(self.block, div_op);
                        new_value = c.mlirOperationGetResult(div_op, 0);
                    },
                    .PercentEqual => {
                        // current_value % rhs_value
                        var rem_state = c.mlirOperationStateGet(c.mlirStringRefCreateFromCString("arith.remsi"), self.fileLoc(ident.span));
                        c.mlirOperationStateAddOperands(&rem_state, 2, @ptrCast(&[_]c.MlirValue{ current_value, rhs_value }));
                        c.mlirOperationStateAddResults(&rem_state, 1, @ptrCast(&result_ty));
                        const rem_op = c.mlirOperationCreate(&rem_state);
                        c.mlirBlockAppendOwnedOperation(self.block, rem_op);
                        new_value = c.mlirOperationGetResult(rem_op, 0);
                    },
                }

                // Store the result back to storage using ora.sstore
                const store_op = memory_manager.createStorageStore(new_value, ident.name, self.fileLoc(ident.span));
                c.mlirBlockAppendOwnedOperation(self.block, store_op);
            } else {
                // No storage map - fall back to placeholder
                var state = c.mlirOperationStateGet(c.mlirStringRefCreateFromCString("ora.compound_assign"), self.fileLoc(ident.span));
                const op = c.mlirOperationCreate(&state);
                c.mlirBlockAppendOwnedOperation(self.block, op);
            }
        } else {
            std.debug.print("DEBUG: Compound assignment target is not an Identifier: {s}\n", .{@tagName(assignment.target.*)});
            // For now, skip non-identifier compound assignments
        }
    }

    /// Lower if statements using scf.if with then/else regions
    pub fn lowerIf(self: *const StatementLowerer, if_stmt: *const lib.ast.Statements.IfNode) LoweringError!void {
        const loc = self.fileLoc(if_stmt.span);

        // Lower the condition expression
        const condition = self.expr_lowerer.lowerExpression(&if_stmt.condition);

        // Create the scf.if operation with proper then/else regions
        var state = c.mlirOperationStateGet(c.mlirStringRefCreateFromCString("scf.if"), loc);

        // Add the condition operand
        c.mlirOperationStateAddOperands(&state, 1, @ptrCast(&condition));

        // Create then region
        const then_region = c.mlirRegionCreate();
        const then_block = c.mlirBlockCreate(0, null, null);
        c.mlirRegionInsertOwnedBlock(then_region, 0, then_block);
        c.mlirOperationStateAddOwnedRegions(&state, 1, @ptrCast(&then_region));

        // Create else region if present
        if (if_stmt.else_branch) |else_branch| {
            const else_region = c.mlirRegionCreate();
            const else_block = c.mlirBlockCreate(0, null, null);
            c.mlirRegionInsertOwnedBlock(else_region, 0, else_block);
            c.mlirOperationStateAddOwnedRegions(&state, 1, @ptrCast(&else_region));

            // Lower else branch
            try self.lowerBlockBody(else_branch, else_block);
        }

        const op = c.mlirOperationCreate(&state);
        c.mlirBlockAppendOwnedOperation(self.block, op);

        // Lower then branch
        try self.lowerBlockBody(if_stmt.then_branch, then_block);
    }

    /// Lower while statements using scf.while with condition and body regions
    pub fn lowerWhile(self: *const StatementLowerer, while_stmt: *const lib.ast.Statements.WhileNode) LoweringError!void {
        const loc = self.fileLoc(while_stmt.span);

        // Create scf.while operation
        var state = c.mlirOperationStateGet(c.mlirStringRefCreateFromCString("scf.while"), loc);

        // Create before region (condition)
        const before_region = c.mlirRegionCreate();
        const before_block = c.mlirBlockCreate(0, null, null);
        c.mlirRegionInsertOwnedBlock(before_region, 0, before_block);
        c.mlirOperationStateAddOwnedRegions(&state, 1, @ptrCast(&before_region));

        // Create after region (body)
        const after_region = c.mlirRegionCreate();
        const after_block = c.mlirBlockCreate(0, null, null);
        c.mlirRegionInsertOwnedBlock(after_region, 0, after_block);
        c.mlirOperationStateAddOwnedRegions(&state, 1, @ptrCast(&after_region));

        const op = c.mlirOperationCreate(&state);
        c.mlirBlockAppendOwnedOperation(self.block, op);

        // Lower condition in before region
        // Create a new statement lowerer for the condition block
        _ = StatementLowerer.init(self.ctx, before_block, self.type_mapper, self.expr_lowerer, self.param_map, self.storage_map, self.local_var_map, self.locations, self.symbol_table, self.allocator);

        const condition = self.expr_lowerer.lowerExpression(&while_stmt.condition);

        // Create scf.condition operation in before block
        var cond_state = c.mlirOperationStateGet(c.mlirStringRefCreateFromCString("scf.condition"), loc);
        c.mlirOperationStateAddOperands(&cond_state, 1, @ptrCast(&condition));
        const cond_op = c.mlirOperationCreate(&cond_state);
        c.mlirBlockAppendOwnedOperation(before_block, cond_op);

        // Lower loop invariants if present
        for (while_stmt.invariants) |*invariant| {
            const invariant_value = self.expr_lowerer.lowerExpression(invariant);

            // Create ora.invariant operation
            var inv_state = c.mlirOperationStateGet(c.mlirStringRefCreateFromCString("ora.invariant"), loc);
            c.mlirOperationStateAddOperands(&inv_state, 1, @ptrCast(&invariant_value));
            const inv_op = c.mlirOperationCreate(&inv_state);
            c.mlirBlockAppendOwnedOperation(before_block, inv_op);
        }

        // Lower body in after region
        try self.lowerBlockBody(while_stmt.body, after_block);

        // Add scf.yield at end of body to continue loop
        var yield_state = c.mlirOperationStateGet(c.mlirStringRefCreateFromCString("scf.yield"), loc);
        const yield_op = c.mlirOperationCreate(&yield_state);
        c.mlirBlockAppendOwnedOperation(after_block, yield_op);
    }

    /// Lower for loop statements using scf.for with proper iteration variables
    pub fn lowerFor(self: *const StatementLowerer, for_stmt: *const lib.ast.Statements.ForLoopNode) LoweringError!void {
        const loc = self.fileLoc(for_stmt.span);

        // Lower the iterable expression
        const iterable = self.expr_lowerer.lowerExpression(&for_stmt.iterable);

        // Handle different loop patterns
        switch (for_stmt.pattern) {
            .Single => |single| {
                try self.lowerSimpleForLoop(single.name, iterable, for_stmt.body, loc);
            },
            .IndexPair => |pair| {
                try self.lowerIndexedForLoop(pair.item, pair.index, iterable, for_stmt.body, loc);
            },
            .Destructured => |destructured| {
                try self.lowerDestructuredForLoop(destructured.pattern, iterable, for_stmt.body, loc);
            },
        }
    }

    /// Lower simple for loop (for (iterable) |item| body)
    fn lowerSimpleForLoop(self: *const StatementLowerer, item_name: []const u8, iterable: c.MlirValue, body: lib.ast.Statements.BlockNode, loc: c.MlirLocation) LoweringError!void {
        // Create scf.for operation
        var state = c.mlirOperationStateGet(c.mlirStringRefCreateFromCString("scf.for"), loc);

        // Get the iterable type to determine proper iteration strategy
        const iterable_ty = c.mlirValueGetType(iterable);
        const zero_ty = c.mlirIntegerTypeGet(self.ctx, constants.DEFAULT_INTEGER_BITS);

        // Determine iteration strategy based on type
        var lower_bound: c.MlirValue = undefined;
        var upper_bound: c.MlirValue = undefined;
        var step: c.MlirValue = undefined;

        // Check if iterable is a memref (array/map) or other type
        if (c.mlirTypeIsAMemRef(iterable_ty)) {
            // For memref types, get the dimension as upper bound
            var dim_state = c.mlirOperationStateGet(c.mlirStringRefCreateFromCString("memref.dim"), loc);
            c.mlirOperationStateAddOperands(&dim_state, 2, @ptrCast(&[_]c.MlirValue{ iterable, iterable }));
            c.mlirOperationStateAddResults(&dim_state, 1, @ptrCast(&zero_ty));
            const dim_op = c.mlirOperationCreate(&dim_state);
            c.mlirBlockAppendOwnedOperation(self.block, dim_op);
            upper_bound = c.mlirOperationGetResult(dim_op, 0);
        } else {
            // For other types, use a default range
            upper_bound = iterable;
        }

        // Create constants for loop bounds
        var zero_state = c.mlirOperationStateGet(c.mlirStringRefCreateFromCString("arith.constant"), loc);
        c.mlirOperationStateAddResults(&zero_state, 1, @ptrCast(&zero_ty));
        const zero_attr = c.mlirIntegerAttrGet(zero_ty, 0);
        const value_id = c.mlirIdentifierGet(self.ctx, c.mlirStringRefCreateFromCString("value"));
        var zero_attrs = [_]c.MlirNamedAttribute{c.mlirNamedAttributeGet(value_id, zero_attr)};
        c.mlirOperationStateAddAttributes(&zero_state, zero_attrs.len, &zero_attrs);
        const zero_op = c.mlirOperationCreate(&zero_state);
        c.mlirBlockAppendOwnedOperation(self.block, zero_op);
        lower_bound = c.mlirOperationGetResult(zero_op, 0);

        // Create step constant
        var step_state = c.mlirOperationStateGet(c.mlirStringRefCreateFromCString("arith.constant"), loc);
        c.mlirOperationStateAddResults(&step_state, 1, @ptrCast(&zero_ty));
        const step_attr = c.mlirIntegerAttrGet(zero_ty, 1);
        var step_attrs = [_]c.MlirNamedAttribute{c.mlirNamedAttributeGet(value_id, step_attr)};
        c.mlirOperationStateAddAttributes(&step_state, step_attrs.len, &step_attrs);
        const step_op = c.mlirOperationCreate(&step_state);
        c.mlirBlockAppendOwnedOperation(self.block, step_op);
        step = c.mlirOperationGetResult(step_op, 0);

        // Add operands to scf.for
        const operands = [_]c.MlirValue{ lower_bound, upper_bound, step };
        c.mlirOperationStateAddOperands(&state, operands.len, &operands);

        // Create body region
        const body_region = c.mlirRegionCreate();
        const body_block = c.mlirBlockCreate(1, @ptrCast(&zero_ty), null);
        c.mlirRegionInsertOwnedBlock(body_region, 0, body_block);
        c.mlirOperationStateAddOwnedRegions(&state, 1, @ptrCast(&body_region));

        const for_op = c.mlirOperationCreate(&state);
        c.mlirBlockAppendOwnedOperation(self.block, for_op);

        // Get the induction variable
        const induction_var = c.mlirBlockGetArgument(body_block, 0);

        // Add the loop variable to local variable map
        if (self.local_var_map) |lvm| {
            lvm.addLocalVar(item_name, induction_var) catch {
                std.debug.print("WARNING: Failed to add loop variable to map: {s}\n", .{item_name});
            };
        }

        // Lower the loop body
        try self.lowerBlockBody(body, body_block);

        // Add scf.yield at end of body
        var yield_state = c.mlirOperationStateGet(c.mlirStringRefCreateFromCString("scf.yield"), loc);
        const yield_op = c.mlirOperationCreate(&yield_state);
        c.mlirBlockAppendOwnedOperation(body_block, yield_op);
    }

    /// Lower indexed for loop (for (iterable) |item, index| body)
    fn lowerIndexedForLoop(self: *const StatementLowerer, item_name: []const u8, index_name: []const u8, iterable: c.MlirValue, body: lib.ast.Statements.BlockNode, loc: c.MlirLocation) LoweringError!void {
        // Create scf.for operation similar to simple for loop
        var state = c.mlirOperationStateGet(c.mlirStringRefCreateFromCString("scf.for"), loc);

        // Create integer type for loop bounds
        const zero_ty = c.mlirIntegerTypeGet(self.ctx, constants.DEFAULT_INTEGER_BITS);

        // Create constants for loop bounds
        var zero_state = c.mlirOperationStateGet(c.mlirStringRefCreateFromCString("arith.constant"), loc);
        c.mlirOperationStateAddResults(&zero_state, 1, @ptrCast(&zero_ty));
        const zero_attr = c.mlirIntegerAttrGet(zero_ty, 0);
        const value_id = c.mlirIdentifierGet(self.ctx, c.mlirStringRefCreateFromCString("value"));
        var zero_attrs = [_]c.MlirNamedAttribute{c.mlirNamedAttributeGet(value_id, zero_attr)};
        c.mlirOperationStateAddAttributes(&zero_state, zero_attrs.len, &zero_attrs);
        const zero_op = c.mlirOperationCreate(&zero_state);
        c.mlirBlockAppendOwnedOperation(self.block, zero_op);
        const lower_bound = c.mlirOperationGetResult(zero_op, 0);

        // Use iterable as upper bound (simplified)
        const upper_bound = iterable;

        // Create step constant
        var step_state = c.mlirOperationStateGet(c.mlirStringRefCreateFromCString("arith.constant"), loc);
        c.mlirOperationStateAddResults(&step_state, 1, @ptrCast(&zero_ty));
        const step_attr = c.mlirIntegerAttrGet(zero_ty, 1);
        var step_attrs = [_]c.MlirNamedAttribute{c.mlirNamedAttributeGet(value_id, step_attr)};
        c.mlirOperationStateAddAttributes(&step_state, step_attrs.len, &step_attrs);
        const step_op = c.mlirOperationCreate(&step_state);
        c.mlirBlockAppendOwnedOperation(self.block, step_op);
        const step = c.mlirOperationGetResult(step_op, 0);

        // Add operands to scf.for
        const operands = [_]c.MlirValue{ lower_bound, upper_bound, step };
        c.mlirOperationStateAddOperands(&state, operands.len, &operands);

        // Create body region with two arguments: index and item
        const body_region = c.mlirRegionCreate();
        const body_block = c.mlirBlockCreate(2, @ptrCast(&[_]c.MlirType{ zero_ty, zero_ty }), null);
        c.mlirRegionInsertOwnedBlock(body_region, 0, body_block);
        c.mlirOperationStateAddOwnedRegions(&state, 1, @ptrCast(&body_region));

        const for_op = c.mlirOperationCreate(&state);
        c.mlirBlockAppendOwnedOperation(self.block, for_op);

        // Get the induction variables (index and item)
        const index_var = c.mlirBlockGetArgument(body_block, 0);
        const item_var = c.mlirBlockGetArgument(body_block, 1);

        // Add both loop variables to local variable map
        if (self.local_var_map) |lvm| {
            lvm.addLocalVar(index_name, index_var) catch {
                std.debug.print("WARNING: Failed to add index variable to map: {s}\n", .{index_name});
            };
            lvm.addLocalVar(item_name, item_var) catch {
                std.debug.print("WARNING: Failed to add item variable to map: {s}\n", .{item_name});
            };
        }

        // Lower the loop body
        try self.lowerBlockBody(body, body_block);

        // Add scf.yield at end of body
        var yield_state = c.mlirOperationStateGet(c.mlirStringRefCreateFromCString("scf.yield"), loc);
        const yield_op = c.mlirOperationCreate(&yield_state);
        c.mlirBlockAppendOwnedOperation(body_block, yield_op);
    }

    /// Lower destructured for loop (for (iterable) |.{field1, field2}| body)
    fn lowerDestructuredForLoop(self: *const StatementLowerer, pattern: lib.ast.Expressions.DestructuringPattern, iterable: c.MlirValue, body: lib.ast.Statements.BlockNode, loc: c.MlirLocation) LoweringError!void {
        // Create scf.for operation similar to simple for loop
        var state = c.mlirOperationStateGet(c.mlirStringRefCreateFromCString("scf.for"), loc);

        // Create integer type for loop bounds
        const zero_ty = c.mlirIntegerTypeGet(self.ctx, constants.DEFAULT_INTEGER_BITS);

        // Create constants for loop bounds
        var zero_state = c.mlirOperationStateGet(c.mlirStringRefCreateFromCString("arith.constant"), loc);
        c.mlirOperationStateAddResults(&zero_state, 1, @ptrCast(&zero_ty));
        const zero_attr = c.mlirIntegerAttrGet(zero_ty, 0);
        const value_id = c.mlirIdentifierGet(self.ctx, c.mlirStringRefCreateFromCString("value"));
        var zero_attrs = [_]c.MlirNamedAttribute{c.mlirNamedAttributeGet(value_id, zero_attr)};
        c.mlirOperationStateAddAttributes(&zero_state, zero_attrs.len, &zero_attrs);
        const zero_op = c.mlirOperationCreate(&zero_state);
        c.mlirBlockAppendOwnedOperation(self.block, zero_op);
        const lower_bound = c.mlirOperationGetResult(zero_op, 0);

        // Use iterable as upper bound (simplified)
        const upper_bound = iterable;

        // Create step constant
        var step_state = c.mlirOperationStateGet(c.mlirStringRefCreateFromCString("arith.constant"), loc);
        c.mlirOperationStateAddResults(&step_state, 1, @ptrCast(&zero_ty));
        const step_attr = c.mlirIntegerAttrGet(zero_ty, 1);
        var step_attrs = [_]c.MlirNamedAttribute{c.mlirNamedAttributeGet(value_id, step_attr)};
        c.mlirOperationStateAddAttributes(&step_state, step_attrs.len, &step_attrs);
        const step_op = c.mlirOperationCreate(&step_state);
        c.mlirBlockAppendOwnedOperation(self.block, step_op);
        const step = c.mlirOperationGetResult(step_op, 0);

        // Add operands to scf.for
        const operands = [_]c.MlirValue{ lower_bound, upper_bound, step };
        c.mlirOperationStateAddOperands(&state, operands.len, &operands);

        // Create body region with one argument: the item to destructure
        const body_region = c.mlirRegionCreate();
        const body_block = c.mlirBlockCreate(1, @ptrCast(&zero_ty), null);
        c.mlirRegionInsertOwnedBlock(body_region, 0, body_block);
        c.mlirOperationStateAddOwnedRegions(&state, 1, @ptrCast(&body_region));

        const for_op = c.mlirOperationCreate(&state);
        c.mlirBlockAppendOwnedOperation(self.block, for_op);

        // Get the item variable
        const item_var = c.mlirBlockGetArgument(body_block, 0);

        // Add destructured fields to local variable map
        if (self.local_var_map) |lvm| {
            switch (pattern) {
                .Struct => |struct_pattern| {
                    for (struct_pattern, 0..) |field, i| {
                        // Create field access for each destructured field
                        var field_access_state = c.mlirOperationStateGet(c.mlirStringRefCreateFromCString("llvm.extractvalue"), loc);
                        c.mlirOperationStateAddOperands(&field_access_state, 1, @ptrCast(&item_var));

                        // Add field index as attribute (for now, assume sequential)
                        const field_index_id = c.mlirIdentifierGet(self.ctx, c.mlirStringRefCreateFromCString("position"));
                        const field_index_attr = c.mlirIntegerAttrGet(c.mlirIntegerTypeGet(self.ctx, 32), @intCast(i));
                        var field_attrs = [_]c.MlirNamedAttribute{c.mlirNamedAttributeGet(field_index_id, field_index_attr)};
                        c.mlirOperationStateAddAttributes(&field_access_state, field_attrs.len, &field_attrs);

                        const field_access_op = c.mlirOperationCreate(&field_access_state);
                        c.mlirBlockAppendOwnedOperation(body_block, field_access_op);
                        const field_value = c.mlirOperationGetResult(field_access_op, 0);

                        // Add to variable map
                        lvm.addLocalVar(field.variable, field_value) catch {
                            std.debug.print("WARNING: Failed to add destructured field to map: {s}\n", .{field.variable});
                        };
                    }
                },
                else => {
                    std.debug.print("WARNING: Unsupported destructuring pattern type\n", .{});
                },
            }
        }

        // Lower the loop body
        try self.lowerBlockBody(body, body_block);

        // Add scf.yield at end of body
        var yield_state = c.mlirOperationStateGet(c.mlirStringRefCreateFromCString("scf.yield"), loc);
        const yield_op = c.mlirOperationCreate(&yield_state);
        c.mlirBlockAppendOwnedOperation(body_block, yield_op);
    }

    /// Lower switch statements using cf.switch with case blocks
    pub fn lowerSwitch(self: *const StatementLowerer, switch_stmt: *const lib.ast.Statements.SwitchNode) LoweringError!void {
        const loc = self.fileLoc(switch_stmt.span);

        // Lower the condition expression
        const condition = self.expr_lowerer.lowerExpression(&switch_stmt.condition);

        // Create cf.switch operation
        var state = c.mlirOperationStateGet(c.mlirStringRefCreateFromCString("cf.switch"), loc);
        c.mlirOperationStateAddOperands(&state, 1, @ptrCast(&condition));

        // Create case values and blocks
        if (switch_stmt.cases.len > 0) {
            // Implement proper case handling with case values and blocks

            // Create blocks for each case
            var case_blocks = std.ArrayList(c.MlirBlock).init(self.allocator);
            defer case_blocks.deinit();

            // Create case values array
            var case_values = std.ArrayList(c.MlirValue).init(self.allocator);
            defer case_values.deinit();

            // Process each case
            for (switch_stmt.cases) |case| {
                // Create case block
                const case_block = c.mlirBlockCreate(0, null, null);
                case_blocks.append(case_block) catch {};

                // Lower case value if it's a literal
                switch (case.pattern) {
                    .Literal => |lit| {
                        const case_value = self.expr_lowerer.lowerLiteral(&lit.value);
                        case_values.append(case_value) catch {};
                    },
                    .Range => |range| {
                        // For range patterns, create a range check
                        const start_val = self.expr_lowerer.lowerExpression(range.start);
                        const end_val = self.expr_lowerer.lowerExpression(range.end);
                        const case_value = self.createRangeCheck(start_val, end_val, range.inclusive, case.span);
                        case_values.append(case_value) catch {};
                    },
                    .EnumValue => |enum_val| {
                        // For enum values, create an enum constant
                        const case_value = self.createEnumConstant(enum_val.enum_name, enum_val.variant_name, case.span);
                        case_values.append(case_value) catch {};
                    },
                    .Else => {
                        // Else case doesn't need a value
                        case_values.append(case_values.items[0]) catch {}; // Use first case value as placeholder
                    },
                }

                // Lower case body
                switch (case.body) {
                    .Expression => |expr| {
                        _ = self.expr_lowerer.lowerExpression(expr);
                    },
                    .Block => |block| {
                        try self.lowerBlockBody(block, case_block);
                    },
                    .LabeledBlock => |labeled| {
                        try self.lowerBlockBody(labeled.block, case_block);
                    },
                }
            }

            // Create default block
            const default_block = c.mlirBlockCreate(0, null, null);

            // Add default case if present
            if (switch_stmt.default_case) |default_case| {
                try self.lowerBlockBody(default_case, default_block);
            } else {
                // Create empty default block with unreachable
                var unreachable_state = c.mlirOperationStateGet(c.mlirStringRefCreateFromCString("cf.unreachable"), loc);
                const unreachable_op = c.mlirOperationCreate(&unreachable_state);
                c.mlirBlockAppendOwnedOperation(default_block, unreachable_op);
            }

            // Add case blocks to the switch operation
            c.mlirOperationStateAddSuccessors(&state, @intCast(case_blocks.items.len), case_blocks.items.ptr);
            c.mlirOperationStateAddSuccessors(&state, 1, @ptrCast(&default_block));

            // Add case values
            if (case_values.items.len > 0) {
                c.mlirOperationStateAddOperands(&state, @intCast(case_values.items.len), case_values.items.ptr);
            }
        }

        const op = c.mlirOperationCreate(&state);
        c.mlirBlockAppendOwnedOperation(self.block, op);
    }

    /// Lower expression statements with proper lvalue resolution
    pub fn lowerExpressionStatement(self: *const StatementLowerer, expr: *const lib.ast.Statements.ExprNode) LoweringError!void {
        switch (expr.*) {
            .Assignment => |assign| {
                try self.lowerAssignmentExpression(&assign);
            },
            .CompoundAssignment => |compound| {
                try self.lowerCompoundAssignmentExpr(&compound);
            },
            else => {
                // Lower other expression statements
                _ = self.expr_lowerer.lowerExpression(expr);
            },
        }
    }

    /// Lower assignment expressions with comprehensive lvalue resolution
    fn lowerAssignmentExpression(self: *const StatementLowerer, assign: *const lib.ast.Expressions.AssignmentExpr) LoweringError!void {
        // Lower the value expression first
        const value = self.expr_lowerer.lowerExpression(assign.value);

        // Resolve the lvalue and generate appropriate store operation
        try self.lowerLValueAssignment(assign.target, value, getExpressionSpan(assign.target));
    }

    /// Lower lvalue assignments (handles identifiers, field access, array indexing)
    fn lowerLValueAssignment(self: *const StatementLowerer, target: *const lib.ast.Expressions.ExprNode, value: c.MlirValue, span: lib.ast.SourceSpan) LoweringError!void {
        const loc = self.fileLoc(span);

        switch (target.*) {
            .Identifier => |ident| {
                try self.lowerIdentifierAssignment(&ident, value, loc);
            },
            .FieldAccess => |field_access| {
                try self.lowerFieldAccessAssignment(&field_access, value, loc);
            },
            .Index => |index_expr| {
                try self.lowerIndexAssignment(&index_expr, value, loc);
            },
            else => {
                std.debug.print("ERROR: Unsupported lvalue type for assignment: {s}\n", .{@tagName(target.*)});
                return LoweringError.InvalidLValue;
            },
        }
    }

    /// Lower identifier assignments
    fn lowerIdentifierAssignment(self: *const StatementLowerer, ident: *const lib.ast.Expressions.IdentifierExpr, value: c.MlirValue, loc: c.MlirLocation) LoweringError!void {
        // Check symbol table first for memory region information
        if (self.symbol_table) |st| {
            if (st.lookupSymbol(ident.name)) |symbol| {
                const region = std.meta.stringToEnum(lib.ast.Statements.MemoryRegion, symbol.region) orelse lib.ast.Statements.MemoryRegion.Stack;

                switch (region) {
                    .Storage => {
                        const store_op = self.memory_manager.createStorageStore(value, ident.name, loc);
                        c.mlirBlockAppendOwnedOperation(self.block, store_op);
                        return;
                    },
                    .Memory => {
                        // For memory variables, we need the memref address
                        if (self.local_var_map) |lvm| {
                            if (lvm.getLocalVar(ident.name)) |memref| {
                                const store_op = self.memory_manager.createStoreOp(value, memref, region, loc);
                                c.mlirBlockAppendOwnedOperation(self.block, store_op);
                                return;
                            }
                        }
                    },
                    .TStore => {
                        const store_op = self.memory_manager.createTStoreStore(value, ident.name, loc);
                        c.mlirBlockAppendOwnedOperation(self.block, store_op);
                        return;
                    },
                    .Stack => {
                        // Update local variable map
                        if (self.local_var_map) |lvm| {
                            lvm.addLocalVar(ident.name, value) catch {
                                return LoweringError.OutOfMemory;
                            };
                            return;
                        }
                    },
                }
            }
        }

        // Fallback: check storage map
        if (self.storage_map) |sm| {
            if (sm.hasStorageVariable(ident.name)) {
                const store_op = self.memory_manager.createStorageStore(value, ident.name, loc);
                c.mlirBlockAppendOwnedOperation(self.block, store_op);
                return;
            }
        }

        // Fallback: check local variable map
        if (self.local_var_map) |lvm| {
            if (lvm.hasLocalVar(ident.name)) {
                lvm.addLocalVar(ident.name, value) catch {
                    return LoweringError.OutOfMemory;
                };
                return;
            }
        }

        std.debug.print("ERROR: Variable not found for assignment: {s}\n", .{ident.name});
        return LoweringError.UndefinedSymbol;
    }

    /// Lower field access assignments (struct.field = value)
    fn lowerFieldAccessAssignment(self: *const StatementLowerer, field_access: *const lib.ast.Expressions.FieldAccessExpr, value: c.MlirValue, loc: c.MlirLocation) LoweringError!void {
        // Lower the target expression to get the struct
        const target = self.expr_lowerer.lowerExpression(field_access.target);
        const target_type = c.mlirValueGetType(target);

        // For struct field assignment, we need to:
        // 1. Load the current struct value
        // 2. Insert the new field value
        // 3. Store the updated struct back

        // Create llvm.insertvalue operation to update the field
        var insert_state = c.mlirOperationStateGet(c.mlirStringRefCreateFromCString("llvm.insertvalue"), loc);
        c.mlirOperationStateAddOperands(&insert_state, 2, @ptrCast(&[_]c.MlirValue{ target, value }));

        // Add field index as attribute (for now, assume field index 0)
        // TODO: Look up actual field index from struct definition
        const field_index_id = c.mlirIdentifierGet(self.ctx, c.mlirStringRefCreateFromCString("position"));
        const field_index_attr = c.mlirIntegerAttrGet(c.mlirIntegerTypeGet(self.ctx, 32), 0);
        var attrs = [_]c.MlirNamedAttribute{c.mlirNamedAttributeGet(field_index_id, field_index_attr)};
        c.mlirOperationStateAddAttributes(&insert_state, attrs.len, &attrs);

        // Set result type to be the same as the struct type
        c.mlirOperationStateAddResults(&insert_state, 1, @ptrCast(&target_type));

        const insert_op = c.mlirOperationCreate(&insert_state);
        c.mlirBlockAppendOwnedOperation(self.block, insert_op);
        const updated_struct = c.mlirOperationGetResult(insert_op, 0);

        // If the target is a variable, store the updated struct back
        if (field_access.target.* == .Identifier) {
            const ident = field_access.target.Identifier;
            if (self.local_var_map) |var_map| {
                if (var_map.getLocalVar(ident.name)) |var_value| {
                    var store_state = c.mlirOperationStateGet(c.mlirStringRefCreateFromCString("memref.store"), loc);
                    c.mlirOperationStateAddOperands(&store_state, 2, @ptrCast(&[_]c.MlirValue{ updated_struct, var_value }));
                    const store_op = c.mlirOperationCreate(&store_state);
                    c.mlirBlockAppendOwnedOperation(self.block, store_op);
                } else {
                    std.debug.print("ERROR: Variable not found for field assignment: {s}\n", .{ident.name});
                    return LoweringError.UndefinedSymbol;
                }
            } else {
                std.debug.print("ERROR: No local variable map available for field assignment\n", .{});
                return LoweringError.UndefinedSymbol;
            }
        } else {
            // For complex field access (e.g., nested structs), use ora.field_store
            var field_store_state = c.mlirOperationStateGet(c.mlirStringRefCreateFromCString("ora.field_store"), loc);
            c.mlirOperationStateAddOperands(&field_store_state, 2, @ptrCast(&[_]c.MlirValue{ updated_struct, target }));

            // Add field name as attribute
            const field_name_id = c.mlirIdentifierGet(self.ctx, c.mlirStringRefCreateFromCString("field"));
            const field_name_attr = c.mlirStringAttrGet(self.ctx, c.mlirStringRefCreateFromCString(field_access.field.ptr));
            var field_attrs = [_]c.MlirNamedAttribute{c.mlirNamedAttributeGet(field_name_id, field_name_attr)};
            c.mlirOperationStateAddAttributes(&field_store_state, field_attrs.len, &field_attrs);

            const field_store_op = c.mlirOperationCreate(&field_store_state);
            c.mlirBlockAppendOwnedOperation(self.block, field_store_op);
        }
    }

    /// Lower array/map index assignments (arr[index] = value)
    fn lowerIndexAssignment(self: *const StatementLowerer, index_expr: *const lib.ast.Expressions.IndexExpr, value: c.MlirValue, loc: c.MlirLocation) LoweringError!void {
        // Lower the target expression to get the array/map
        const target = self.expr_lowerer.lowerExpression(index_expr.target);
        const index_val = self.expr_lowerer.lowerExpression(index_expr.index);
        const target_type = c.mlirValueGetType(target);

        // Determine the type of indexing operation
        if (c.mlirTypeIsAMemRef(target_type)) {
            // Array indexing using memref.store
            var store_state = c.mlirOperationStateGet(c.mlirStringRefCreateFromCString("memref.store"), loc);
            c.mlirOperationStateAddOperands(&store_state, 3, @ptrCast(&[_]c.MlirValue{ value, target, index_val }));
            const store_op = c.mlirOperationCreate(&store_state);
            c.mlirBlockAppendOwnedOperation(self.block, store_op);
        } else {
            // Map indexing or other complex indexing operations
            // For now, use a generic store operation with ora.map_store attribute
            var store_state = c.mlirOperationStateGet(c.mlirStringRefCreateFromCString("ora.store"), loc);
            c.mlirOperationStateAddOperands(&store_state, 3, @ptrCast(&[_]c.MlirValue{ value, target, index_val }));

            // Add map store attribute
            const map_store_id = c.mlirIdentifierGet(self.ctx, c.mlirStringRefCreateFromCString("ora.map_store"));
            const map_store_attr = c.mlirBoolAttrGet(self.ctx, 1);
            var attrs = [_]c.MlirNamedAttribute{c.mlirNamedAttributeGet(map_store_id, map_store_attr)};
            c.mlirOperationStateAddAttributes(&store_state, attrs.len, &attrs);

            const store_op = c.mlirOperationCreate(&store_state);
            c.mlirBlockAppendOwnedOperation(self.block, store_op);
        }
    }

    /// Lower labeled block statements using scf.execute_region
    pub fn lowerLabeledBlock(self: *const StatementLowerer, labeled_block: *const lib.ast.Statements.LabeledBlockNode) LoweringError!void {
        const loc = self.fileLoc(labeled_block.span);

        // Create scf.execute_region operation
        var state = c.mlirOperationStateGet(c.mlirStringRefCreateFromCString("scf.execute_region"), loc);

        // Add label as attribute
        const label_ref = c.mlirStringRefCreate(labeled_block.label.ptr, labeled_block.label.len);
        const label_attr = c.mlirStringAttrGet(self.ctx, label_ref);
        const label_id = c.mlirIdentifierGet(self.ctx, c.mlirStringRefCreateFromCString("label"));
        var attrs = [_]c.MlirNamedAttribute{c.mlirNamedAttributeGet(label_id, label_attr)};
        c.mlirOperationStateAddAttributes(&state, attrs.len, &attrs);

        // Create region for the labeled block
        const region = c.mlirRegionCreate();
        const block = c.mlirBlockCreate(0, null, null);
        c.mlirRegionInsertOwnedBlock(region, 0, block);
        c.mlirOperationStateAddOwnedRegions(&state, 1, @ptrCast(&region));

        const op = c.mlirOperationCreate(&state);
        c.mlirBlockAppendOwnedOperation(self.block, op);

        // Lower the block body in the new region
        try self.lowerBlockBody(labeled_block.block, block);
    }

    /// Lower log statements using ora.log operations with indexed parameter handling
    pub fn lowerLog(self: *const StatementLowerer, log_stmt: *const lib.ast.Statements.LogNode) LoweringError!void {
        const loc = self.fileLoc(log_stmt.span);

        // Create ora.log operation
        var state = c.mlirOperationStateGet(c.mlirStringRefCreateFromCString("ora.log"), loc);

        // Add event name as attribute
        const event_ref = c.mlirStringRefCreate(log_stmt.event_name.ptr, log_stmt.event_name.len);
        const event_attr = c.mlirStringAttrGet(self.ctx, event_ref);
        const event_id = c.mlirIdentifierGet(self.ctx, c.mlirStringRefCreateFromCString("event"));
        var attrs = [_]c.MlirNamedAttribute{c.mlirNamedAttributeGet(event_id, event_attr)};
        c.mlirOperationStateAddAttributes(&state, attrs.len, &attrs);

        // Lower arguments and add as operands
        if (log_stmt.args.len > 0) {
            var operands = try self.allocator.alloc(c.MlirValue, log_stmt.args.len);
            defer self.allocator.free(operands);

            for (log_stmt.args, 0..) |*arg, i| {
                operands[i] = self.expr_lowerer.lowerExpression(arg);
            }

            c.mlirOperationStateAddOperands(&state, @intCast(operands.len), operands.ptr);
        }

        const op = c.mlirOperationCreate(&state);
        c.mlirBlockAppendOwnedOperation(self.block, op);
    }

    /// Lower lock statements using ora.lock operations
    pub fn lowerLock(self: *const StatementLowerer, lock_stmt: *const lib.ast.Statements.LockNode) LoweringError!void {
        const loc = self.fileLoc(lock_stmt.span);

        // Lower the path expression
        const path_value = self.expr_lowerer.lowerExpression(&lock_stmt.path);

        // Create ora.lock operation
        var state = c.mlirOperationStateGet(c.mlirStringRefCreateFromCString("ora.lock"), loc);
        c.mlirOperationStateAddOperands(&state, 1, @ptrCast(&path_value));

        const op = c.mlirOperationCreate(&state);
        c.mlirBlockAppendOwnedOperation(self.block, op);
    }

    /// Lower unlock statements using ora.unlock operations
    pub fn lowerUnlock(self: *const StatementLowerer, unlock_stmt: *const lib.ast.Statements.UnlockNode) LoweringError!void {
        const loc = self.fileLoc(unlock_stmt.span);

        // Lower the path expression
        const path_value = self.expr_lowerer.lowerExpression(&unlock_stmt.path);

        // Create ora.unlock operation
        var state = c.mlirOperationStateGet(c.mlirStringRefCreateFromCString("ora.unlock"), loc);
        c.mlirOperationStateAddOperands(&state, 1, @ptrCast(&path_value));

        const op = c.mlirOperationCreate(&state);
        c.mlirBlockAppendOwnedOperation(self.block, op);
    }

    /// Lower move statements with atomic transfer operations and ora.move attributes
    pub fn lowerMove(self: *const StatementLowerer, move_stmt: *const lib.ast.Statements.MoveNode) LoweringError!void {
        const loc = self.fileLoc(move_stmt.span);

        // Lower all the expressions
        const expr_value = self.expr_lowerer.lowerExpression(&move_stmt.expr);

        const source_value = self.expr_lowerer.lowerExpression(&move_stmt.source);

        const dest_value = self.expr_lowerer.lowerExpression(&move_stmt.dest);

        const amount_value = self.expr_lowerer.lowerExpression(&move_stmt.amount);

        // Create ora.move operation with atomic transfer semantics
        var state = c.mlirOperationStateGet(c.mlirStringRefCreateFromCString("ora.move"), loc);

        const operands = [_]c.MlirValue{ expr_value, source_value, dest_value, amount_value };
        c.mlirOperationStateAddOperands(&state, operands.len, &operands);

        // Add atomic attribute
        const atomic_attr = c.mlirBoolAttrGet(self.ctx, 1);
        const atomic_id = c.mlirIdentifierGet(self.ctx, c.mlirStringRefCreateFromCString("atomic"));
        var attrs = [_]c.MlirNamedAttribute{c.mlirNamedAttributeGet(atomic_id, atomic_attr)};
        c.mlirOperationStateAddAttributes(&state, attrs.len, &attrs);

        const op = c.mlirOperationCreate(&state);
        c.mlirBlockAppendOwnedOperation(self.block, op);
    }

    /// Lower try-catch statements with exception handling constructs and error propagation
    pub fn lowerTryBlock(self: *const StatementLowerer, try_stmt: *const lib.ast.Statements.TryBlockNode) LoweringError!void {
        const loc = self.fileLoc(try_stmt.span);

        // Create ora.try operation
        var state = c.mlirOperationStateGet(c.mlirStringRefCreateFromCString("ora.try"), loc);

        // Create try region
        const try_region = c.mlirRegionCreate();
        const try_block = c.mlirBlockCreate(0, null, null);
        c.mlirRegionInsertOwnedBlock(try_region, 0, try_block);
        c.mlirOperationStateAddOwnedRegions(&state, 1, @ptrCast(&try_region));

        // Create catch region if present
        if (try_stmt.catch_block) |catch_block| {
            const catch_region = c.mlirRegionCreate();
            const catch_block_mlir = c.mlirBlockCreate(0, null, null);
            c.mlirRegionInsertOwnedBlock(catch_region, 0, catch_block_mlir);
            c.mlirOperationStateAddOwnedRegions(&state, 1, @ptrCast(&catch_region));

            // Lower catch block
            try self.lowerBlockBody(catch_block.block, catch_block_mlir);
        }

        // Add error variable as attribute if present
        if (try_stmt.catch_block) |catch_block| {
            if (catch_block.error_variable) |error_var| {
                const error_ref = c.mlirStringRefCreate(error_var.ptr, error_var.len);
                const error_attr = c.mlirStringAttrGet(self.ctx, error_ref);
                const error_id = c.mlirIdentifierGet(self.ctx, c.mlirStringRefCreateFromCString("error_var"));
                var attrs = [_]c.MlirNamedAttribute{c.mlirNamedAttributeGet(error_id, error_attr)};
                c.mlirOperationStateAddAttributes(&state, attrs.len, &attrs);
            }
        }

        const op = c.mlirOperationCreate(&state);
        c.mlirBlockAppendOwnedOperation(self.block, op);

        // Lower try block
        try self.lowerBlockBody(try_stmt.try_block, try_block);
    }

    /// Lower error declarations with error type definitions
    pub fn lowerErrorDecl(self: *const StatementLowerer, error_decl: *const lib.ast.Statements.ErrorDeclNode) LoweringError!void {
        const loc = self.fileLoc(error_decl.span);

        // Create ora.error.decl operation
        var state = c.mlirOperationStateGet(c.mlirStringRefCreateFromCString("ora.error.decl"), loc);

        // Add error name as attribute
        const name_ref = c.mlirStringRefCreate(error_decl.name.ptr, error_decl.name.len);
        const name_attr = c.mlirStringAttrGet(self.ctx, name_ref);
        const name_id = c.mlirIdentifierGet(self.ctx, c.mlirStringRefCreateFromCString("name"));
        var attrs = [_]c.MlirNamedAttribute{c.mlirNamedAttributeGet(name_id, name_attr)};
        c.mlirOperationStateAddAttributes(&state, attrs.len, &attrs);

        // Handle error parameters if present
        if (error_decl.parameters) |parameters| {
            // Add parameters as attributes
            for (parameters) |param| {
                const param_ref = c.mlirStringRefCreate(param.name.ptr, param.name.len);
                const param_attr = c.mlirStringAttrGet(self.ctx, param_ref);
                const param_id = c.mlirIdentifierGet(self.ctx, c.mlirStringRefCreateFromCString("param"));
                var param_attrs = [_]c.MlirNamedAttribute{c.mlirNamedAttributeGet(param_id, param_attr)};
                c.mlirOperationStateAddAttributes(&state, param_attrs.len, &param_attrs);
            }
        }

        const op = c.mlirOperationCreate(&state);
        c.mlirBlockAppendOwnedOperation(self.block, op);
    }

    /// Lower invariant statements for loop invariants
    pub fn lowerInvariant(self: *const StatementLowerer, invariant: *const lib.ast.Statements.InvariantNode) LoweringError!void {
        const loc = self.fileLoc(invariant.span);

        // Lower the condition expression
        const condition = self.expr_lowerer.lowerExpression(&invariant.condition);

        // Create ora.invariant operation
        var state = c.mlirOperationStateGet(c.mlirStringRefCreateFromCString("ora.invariant"), loc);
        c.mlirOperationStateAddOperands(&state, 1, @ptrCast(&condition));

        const op = c.mlirOperationCreate(&state);
        c.mlirBlockAppendOwnedOperation(self.block, op);
    }

    /// Lower requires statements for function preconditions
    pub fn lowerRequires(self: *const StatementLowerer, requires: *const lib.ast.Statements.RequiresNode) LoweringError!void {
        const loc = self.fileLoc(requires.span);

        // Lower the condition expression
        const condition = self.expr_lowerer.lowerExpression(&requires.condition);

        // Create ora.requires operation
        var state = c.mlirOperationStateGet(c.mlirStringRefCreateFromCString("ora.requires"), loc);
        c.mlirOperationStateAddOperands(&state, 1, @ptrCast(&condition));

        const op = c.mlirOperationCreate(&state);
        c.mlirBlockAppendOwnedOperation(self.block, op);
    }

    /// Lower ensures statements for function postconditions
    pub fn lowerEnsures(self: *const StatementLowerer, ensures: *const lib.ast.Statements.EnsuresNode) LoweringError!void {
        const loc = self.fileLoc(ensures.span);

        // Lower the condition expression
        const condition = self.expr_lowerer.lowerExpression(&ensures.condition);

        // Create ora.ensures operation
        var state = c.mlirOperationStateGet(c.mlirStringRefCreateFromCString("ora.ensures"), loc);
        c.mlirOperationStateAddOperands(&state, 1, @ptrCast(&condition));

        const op = c.mlirOperationCreate(&state);
        c.mlirBlockAppendOwnedOperation(self.block, op);
    }

    /// Lower block body with proper error handling and location tracking
    pub fn lowerBlockBody(self: *const StatementLowerer, b: lib.ast.Statements.BlockNode, block: c.MlirBlock) LoweringError!void {
        // Push new scope for block-local variables
        if (self.symbol_table) |st| {
            st.pushScope() catch {
                std.debug.print("WARNING: Failed to push scope for block\n", .{});
            };
        }

        // Process each statement in the block
        for (b.statements) |*s| {
            // Create a new statement lowerer for this block
            var stmt_lowerer = StatementLowerer.init(self.ctx, block, self.type_mapper, self.expr_lowerer, self.param_map, self.storage_map, self.local_var_map, self.locations, self.symbol_table, self.allocator);

            // Lower the statement with error handling
            stmt_lowerer.lowerStatement(s) catch |err| {
                std.debug.print("ERROR: Failed to lower statement type {s}: {s}\n", .{ @tagName(s.*), @errorName(err) });

                // Pop scope before returning error
                if (self.symbol_table) |st| {
                    st.popScope();
                }
                return err;
            };
        }

        // Pop scope after processing all statements
        if (self.symbol_table) |st| {
            st.popScope();
        }
    }

    /// Create file location for operations
    fn fileLoc(self: *const StatementLowerer, span: lib.ast.SourceSpan) c.MlirLocation {
        return LocationTracker.createFileLocationFromSpan(&self.locations, span);
    }

    /// Create range check for switch case patterns
    fn createRangeCheck(self: *const StatementLowerer, start_val: c.MlirValue, end_val: c.MlirValue, inclusive: bool, span: lib.ast.SourceSpan) c.MlirValue {
        const loc = self.fileLoc(span);

        // Create a range check operation that returns a boolean
        const result_ty = c.mlirIntegerTypeGet(self.ctx, 1);
        var state = c.mlirOperationStateGet(c.mlirStringRefCreateFromCString("ora.range_check"), loc);
        c.mlirOperationStateAddOperands(&state, 2, @ptrCast(&[_]c.MlirValue{ start_val, end_val }));
        c.mlirOperationStateAddResults(&state, 1, @ptrCast(&result_ty));

        // Add inclusive flag as attribute
        const inclusive_attr = c.mlirBoolAttrGet(self.ctx, if (inclusive) 1 else 0);
        const inclusive_id = c.mlirIdentifierGet(self.ctx, c.mlirStringRefCreateFromCString("inclusive"));
        var attrs = [_]c.MlirNamedAttribute{
            c.mlirNamedAttributeGet(inclusive_id, inclusive_attr),
        };
        c.mlirOperationStateAddAttributes(&state, attrs.len, &attrs);

        const op = c.mlirOperationCreate(&state);
        c.mlirBlockAppendOwnedOperation(self.block, op);
        return c.mlirOperationGetResult(op, 0);
    }

    /// Create enum constant for switch case patterns
    fn createEnumConstant(self: *const StatementLowerer, enum_name: []const u8, variant_name: []const u8, span: lib.ast.SourceSpan) c.MlirValue {
        const loc = self.fileLoc(span);

        // Create an enum constant operation
        const result_ty = c.mlirIntegerTypeGet(self.ctx, constants.DEFAULT_INTEGER_BITS);
        var state = c.mlirOperationStateGet(c.mlirStringRefCreateFromCString("ora.enum_constant"), loc);
        c.mlirOperationStateAddResults(&state, 1, @ptrCast(&result_ty));

        // Add enum name and variant name as attributes
        const enum_name_attr = c.mlirStringAttrGet(self.ctx, c.mlirStringRefCreateFromCString(enum_name.ptr));
        const enum_name_id = c.mlirIdentifierGet(self.ctx, c.mlirStringRefCreateFromCString("enum_name"));

        const variant_name_attr = c.mlirStringAttrGet(self.ctx, c.mlirStringRefCreateFromCString(variant_name.ptr));
        const variant_name_id = c.mlirIdentifierGet(self.ctx, c.mlirStringRefCreateFromCString("variant_name"));

        var attrs = [_]c.MlirNamedAttribute{
            c.mlirNamedAttributeGet(enum_name_id, enum_name_attr),
            c.mlirNamedAttributeGet(variant_name_id, variant_name_attr),
        };
        c.mlirOperationStateAddAttributes(&state, attrs.len, &attrs);

        const op = c.mlirOperationCreate(&state);
        c.mlirBlockAppendOwnedOperation(self.block, op);
        return c.mlirOperationGetResult(op, 0);
    }
};
