// ============================================================================
// Statement Lowering
// ============================================================================
//
// Converts Ora AST statements to MLIR operations and control flow.
//
// SUPPORTED STATEMENTS:
//   • Variable declarations (let/const/var)
//   • Assignments (simple, compound, destructuring)
//   • Control flow: if/else, while, for loops
//   • Error handling: try/catch, error returns
//   • Advanced: switch expressions, labeled blocks, pattern matching
//   • Blockchain: log statements, storage operations
//   • Verification: requires, ensures, invariant clauses
//
// FEATURES:
//   • Scoped variable management
//   • Control flow graph construction
//   • Loop optimization and lowering
//   • Error propagation
//   • Location tracking for all operations
//
// ============================================================================

const std = @import("std");
const c = @import("c.zig").c;
const lib = @import("ora_lib");
const constants = @import("lower.zig");
const h = @import("helpers.zig");
const TypeMapper = @import("types.zig").TypeMapper;
const ExpressionLowerer = @import("expressions.zig").ExpressionLowerer;
const ParamMap = @import("lower.zig").ParamMap;
const StorageMap = @import("memory.zig").StorageMap;
const LocalVarMap = @import("lower.zig").LocalVarMap;
const LocationTracker = @import("lower.zig").LocationTracker;
const MemoryManager = @import("memory.zig").MemoryManager;
const SymbolTable = @import("lower.zig").SymbolTable;

/// Context for labeled switch continue support
pub const LabelContext = struct {
    label: []const u8,
    continue_flag_memref: c.MlirValue,
    value_memref: c.MlirValue,
};

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
    current_function_return_type: ?c.MlirType,
    ora_dialect: *@import("dialect.zig").OraDialect,
    label_context: ?*const LabelContext,

    pub fn init(ctx: c.MlirContext, block: c.MlirBlock, type_mapper: *const TypeMapper, expr_lowerer: *const ExpressionLowerer, param_map: ?*const ParamMap, storage_map: ?*const StorageMap, local_var_map: ?*LocalVarMap, locations: LocationTracker, symbol_table: ?*SymbolTable, allocator: std.mem.Allocator, function_return_type: ?c.MlirType, ora_dialect: *@import("dialect.zig").OraDialect) StatementLowerer {
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
            .memory_manager = MemoryManager.init(ctx, ora_dialect),
            .allocator = allocator,
            .current_function_return_type = function_return_type,
            .ora_dialect = ora_dialect,
            .label_context = null,
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
        InvalidSwitch,
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

        if (ret.value) |e| {
            const v = self.expr_lowerer.lowerExpression(&e);
            const op = self.ora_dialect.createFuncReturnWithValue(v, loc);
            c.mlirBlockAppendOwnedOperation(self.block, op);
        } else {
            const op = self.ora_dialect.createFuncReturn(loc);
            c.mlirBlockAppendOwnedOperation(self.block, op);
        }
    }

    /// Lower return statements in control flow context using scf.yield
    pub fn lowerReturnInControlFlow(self: *const StatementLowerer, ret: *const lib.ast.Statements.ReturnNode) LoweringError!void {
        const loc = self.fileLoc(ret.span);

        if (ret.value) |e| {
            const v = self.expr_lowerer.lowerExpression(&e);
            const op = self.ora_dialect.createScfYieldWithValues(&[_]c.MlirValue{v}, loc);
            c.mlirBlockAppendOwnedOperation(self.block, op);
        } else {
            const op = self.ora_dialect.createScfYield(loc);
            c.mlirBlockAppendOwnedOperation(self.block, op);
        }
    }

    /// Lower break statements with label support using appropriate control flow transfers
    pub fn lowerBreak(self: *const StatementLowerer, break_stmt: *const lib.ast.Statements.BreakNode) LoweringError!void {
        const loc = self.fileLoc(break_stmt.span);

        if (break_stmt.label) |_| {
            // Labeled break - for now, use scf.yield to exit the labeled scf.execute_region
            // TODO: Implement proper label resolution and cf.br to the correct block
            var operands = std.ArrayList(c.MlirValue){};
            defer operands.deinit(self.allocator);

            // Add break value if present
            if (break_stmt.value) |value_expr| {
                const value = self.expr_lowerer.lowerExpression(value_expr);
                operands.append(self.allocator, value) catch unreachable;
            }

            const op = self.ora_dialect.createScfYield(loc);
            c.mlirBlockAppendOwnedOperation(self.block, op);
        } else {
            // Unlabeled break - use scf.break or cf.br depending on context
            var operands = std.ArrayList(c.MlirValue){};
            defer operands.deinit(self.allocator);

            // Add break value if present
            if (break_stmt.value) |value_expr| {
                const value = self.expr_lowerer.lowerExpression(value_expr);
                operands.append(self.allocator, value) catch unreachable;
            }

            const op = self.ora_dialect.createScfBreak(operands.items, loc);
            c.mlirBlockAppendOwnedOperation(self.block, op);
        }
    }

    /// Lower continue statements with label support using loop continuation operations
    pub fn lowerContinue(self: *const StatementLowerer, continue_stmt: *const lib.ast.Statements.ContinueNode) LoweringError!void {
        const loc = self.fileLoc(continue_stmt.span);

        if (continue_stmt.label) |label| {
            // Check if we have a label context and if it matches
            if (self.label_context) |label_ctx| {
                if (std.mem.eql(u8, label, label_ctx.label)) {
                    // This is a continue to our labeled switch!
                    // Store the new value if provided
                    if (continue_stmt.value) |value_expr| {
                        const value = self.expr_lowerer.lowerExpression(value_expr);
                        const value_to_store = blk: {
                            const val_type = c.mlirValueGetType(value);
                            // If it's a memref, load the value
                            if (c.mlirTypeIsAMemRef(val_type)) {
                                const element_type = c.mlirShapedTypeGetElementType(val_type);
                                var load_state = c.mlirOperationStateGet(c.mlirStringRefCreateFromCString("memref.load"), loc);
                                c.mlirOperationStateAddOperands(&load_state, 1, @ptrCast(&value));
                                c.mlirOperationStateAddResults(&load_state, 1, @ptrCast(&element_type));
                                const load_op = c.mlirOperationCreate(&load_state);
                                c.mlirBlockAppendOwnedOperation(self.block, load_op);
                                break :blk c.mlirOperationGetResult(load_op, 0);
                            }
                            break :blk value;
                        };

                        // Store new value to value_memref
                        var value_store_state = c.mlirOperationStateGet(c.mlirStringRefCreateFromCString("memref.store"), loc);
                        c.mlirOperationStateAddOperands(&value_store_state, 2, @ptrCast(&[_]c.MlirValue{ value_to_store, label_ctx.value_memref }));
                        const value_store = c.mlirOperationCreate(&value_store_state);
                        c.mlirBlockAppendOwnedOperation(self.block, value_store);
                    }

                    // Set continue_flag to true
                    const i1_type = c.mlirIntegerTypeGet(self.ctx, 1);
                    var true_const_state = c.mlirOperationStateGet(c.mlirStringRefCreateFromCString("arith.constant"), loc);
                    const true_attr = c.mlirIntegerAttrGet(i1_type, 1);
                    const value_id = c.mlirIdentifierGet(self.ctx, c.mlirStringRefCreateFromCString("value"));
                    var true_attrs = [_]c.MlirNamedAttribute{c.mlirNamedAttributeGet(value_id, true_attr)};
                    c.mlirOperationStateAddAttributes(&true_const_state, true_attrs.len, &true_attrs);
                    c.mlirOperationStateAddResults(&true_const_state, 1, @ptrCast(&i1_type));
                    const true_const = c.mlirOperationCreate(&true_const_state);
                    c.mlirBlockAppendOwnedOperation(self.block, true_const);
                    const true_val = c.mlirOperationGetResult(true_const, 0);

                    var flag_store_state = c.mlirOperationStateGet(c.mlirStringRefCreateFromCString("memref.store"), loc);
                    c.mlirOperationStateAddOperands(&flag_store_state, 2, @ptrCast(&[_]c.MlirValue{ true_val, label_ctx.continue_flag_memref }));
                    const flag_store = c.mlirOperationCreate(&flag_store_state);
                    c.mlirBlockAppendOwnedOperation(self.block, flag_store);

                    // Add scf.yield to exit the current case
                    var yield_state = c.mlirOperationStateGet(c.mlirStringRefCreateFromCString("scf.yield"), loc);
                    const yield_op = c.mlirOperationCreate(&yield_state);
                    c.mlirBlockAppendOwnedOperation(self.block, yield_op);
                    return;
                }
            }

            // Labeled continue without matching context - use scf.yield
            const op = self.ora_dialect.createScfYield(loc);
            c.mlirBlockAppendOwnedOperation(self.block, op);
        } else {
            // Unlabeled continue - use scf.continue
            var operands = std.ArrayList(c.MlirValue){};
            defer operands.deinit(self.allocator);

            // Add continue value if present (for labeled switch continue)
            if (continue_stmt.value) |value_expr| {
                const value = self.expr_lowerer.lowerExpression(value_expr);
                operands.append(self.allocator, value) catch unreachable;
            }

            const op = self.ora_dialect.createScfContinue(operands.items, loc);
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
        // For mutable variables (var), use memref.alloca to enable proper SSA semantics with assignments
        if (var_decl.kind == .Var) {
            // Create memref type for the variable
            const memref_type = c.mlirMemRefTypeGet(mlir_type, 0, null, c.mlirAttributeGetNull(), c.mlirAttributeGetNull());

            // Allocate memory on the stack
            const alloca_op = self.ora_dialect.createMemrefAlloca(memref_type, loc);
            c.mlirBlockAppendOwnedOperation(self.block, alloca_op);
            const memref = c.mlirOperationGetResult(alloca_op, 0);

            // Initialize the variable if there's an initializer
            if (var_decl.value) |init_expr| {
                const init_value = self.expr_lowerer.lowerExpression(&init_expr.*);
                const store_op = self.ora_dialect.createMemrefStore(init_value, memref, &[_]c.MlirValue{}, loc);
                c.mlirBlockAppendOwnedOperation(self.block, store_op);
            } else {
                // Store default value
                const default_value = try self.createDefaultValue(mlir_type, var_decl.kind, loc);
                const store_op = self.ora_dialect.createMemrefStore(default_value, memref, &[_]c.MlirValue{}, loc);
                c.mlirBlockAppendOwnedOperation(self.block, store_op);
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
            // For immutable variables (let/const), use direct SSA values
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
    fn lowerTStoreVariableDecl(self: *const StatementLowerer, var_decl: *const lib.ast.Statements.VariableDeclNode, mlir_type: c.MlirType, loc: c.MlirLocation) LoweringError!void {
        // Transient storage variables are similar to storage but temporary
        var init_value: c.MlirValue = undefined;

        if (var_decl.value) |init_expr| {
            init_value = self.expr_lowerer.lowerExpression(&init_expr.*);

            // Generate transient storage store operation
            const store_op = self.memory_manager.createTStoreStore(init_value, var_decl.name, loc);
            c.mlirBlockAppendOwnedOperation(self.block, store_op);
        } else {
            // Create default value for uninitialized transient storage variables
            init_value = try self.createDefaultValue(mlir_type, var_decl.kind, loc);
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
                    const result_ty = c.mlirIntegerTypeGet(self.ctx, constants.DEFAULT_INTEGER_BITS);
                    const indices = [_]u32{@intCast(i)};
                    const extract_op = self.ora_dialect.createLlvmExtractvalue(value, &indices, result_ty, loc);
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
                    const result_ty = c.mlirIntegerTypeGet(self.ctx, constants.DEFAULT_INTEGER_BITS);
                    const indices = [_]u32{@intCast(i)};
                    const extract_op = self.ora_dialect.createLlvmExtractvalue(value, &indices, result_ty, loc);
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

        // Handle compound assignment to storage variables
        // For now, we'll assume the target is an identifier expression
        // TODO: Handle more complex target expressions
        if (assignment.target.* == .Identifier) {
            const ident = assignment.target.Identifier;

            if (self.storage_map) |sm| {
                // Ensure the variable exists in storage (create if needed)
                // TODO: Fix const qualifier issue - getOrCreateAddress expects mutable pointer
                // _ = sm.getOrCreateAddress(ident.name) catch 0;
                _ = sm; // Use the variable to avoid warning

                // Define result type for arithmetic operations
                const result_ty = c.mlirIntegerTypeGet(self.ctx, constants.DEFAULT_INTEGER_BITS);

                // Load current value from storage using ora.sload
                const memory_manager = MemoryManager.init(self.ctx, self.ora_dialect);
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
                        const add_op = self.ora_dialect.createArithAddi(current_value, rhs_value, result_ty, self.fileLoc(ident.span));
                        c.mlirBlockAppendOwnedOperation(self.block, add_op);
                        new_value = c.mlirOperationGetResult(add_op, 0);
                    },
                    .MinusEqual => {
                        // current_value - rhs_value
                        const sub_op = self.ora_dialect.createArithSubi(current_value, rhs_value, result_ty, self.fileLoc(ident.span));
                        c.mlirBlockAppendOwnedOperation(self.block, sub_op);
                        new_value = c.mlirOperationGetResult(sub_op, 0);
                    },
                    .StarEqual => {
                        // current_value * rhs_value
                        const mul_op = self.ora_dialect.createArithMuli(current_value, rhs_value, result_ty, self.fileLoc(ident.span));
                        c.mlirBlockAppendOwnedOperation(self.block, mul_op);
                        new_value = c.mlirOperationGetResult(mul_op, 0);
                    },
                    .SlashEqual => {
                        // current_value / rhs_value
                        const div_op = self.ora_dialect.createArithDivsi(current_value, rhs_value, result_ty, self.fileLoc(ident.span));
                        c.mlirBlockAppendOwnedOperation(self.block, div_op);
                        new_value = c.mlirOperationGetResult(div_op, 0);
                    },
                    .PercentEqual => {
                        // current_value % rhs_value
                        const rem_op = self.ora_dialect.createArithRemsi(current_value, rhs_value, result_ty, self.fileLoc(ident.span));
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

            // For now, skip non-identifier compound assignments
        }
    }

    /// Lower if statements using scf.if with then/else regions
    pub fn lowerIf(self: *const StatementLowerer, if_stmt: *const lib.ast.Statements.IfNode) LoweringError!void {
        const loc = self.fileLoc(if_stmt.span);

        // Lower the condition expression
        const condition = self.expr_lowerer.lowerExpression(&if_stmt.condition);

        // Check if this if statement contains return statements
        if (self.ifStatementHasReturns(if_stmt)) {
            // For now, we'll handle this by not using scf.if for return statements
            // This is a temporary fix - ideally we'd restructure to use scf.yield
            try self.lowerIfWithReturns(if_stmt, condition, loc);
            return;
        }

        // Create the scf.if operation with proper then/else regions
        var state = c.mlirOperationStateGet(c.mlirStringRefCreateFromCString("scf.if"), loc);

        // Add the condition operand
        c.mlirOperationStateAddOperands(&state, 1, @ptrCast(&condition));

        // Create then region
        const then_region = c.mlirRegionCreate();
        const then_block = c.mlirBlockCreate(0, null, null);
        c.mlirRegionInsertOwnedBlock(then_region, 0, then_block);

        // Create else region (scf.if always requires both regions)
        const else_region = c.mlirRegionCreate();
        const else_block = c.mlirBlockCreate(0, null, null);
        c.mlirRegionInsertOwnedBlock(else_region, 0, else_block);

        // Add both regions to the operation state
        const regions = [_]c.MlirRegion{ then_region, else_region };
        c.mlirOperationStateAddOwnedRegions(&state, 2, @ptrCast(&regions));

        const if_op = c.mlirOperationCreate(&state);

        // Lower else branch if present, otherwise add scf.yield to empty region
        if (if_stmt.else_branch) |else_branch| {
            try self.lowerBlockBody(else_branch, else_block);
        } else {
            // Add scf.yield to empty else region to satisfy MLIR requirements
            const yield_op = self.ora_dialect.createScfYield(loc);
            c.mlirBlockAppendOwnedOperation(else_block, yield_op);
        }

        // Lower then branch FIRST (before creating the scf.if operation)
        try self.lowerBlockBody(if_stmt.then_branch, then_block);

        // Add scf.yield to then region if it doesn't end with one
        if (!self.blockEndsWithYield(then_block)) {
            const yield_op = self.ora_dialect.createScfYield(loc);
            c.mlirBlockAppendOwnedOperation(then_block, yield_op);
        }

        // Add scf.yield to else region if it doesn't end with one (for non-empty else branches)
        if (if_stmt.else_branch != null and !self.blockEndsWithYield(else_block)) {
            const yield_op = self.ora_dialect.createScfYield(loc);
            c.mlirBlockAppendOwnedOperation(else_block, yield_op);
        }

        // NOW append the scf.if operation to the block
        c.mlirBlockAppendOwnedOperation(self.block, if_op);
    }

    /// Check if a block ends with scf.yield
    fn blockEndsWithYield(_: *const StatementLowerer, _: c.MlirBlock) bool {
        // For now, we'll assume blocks don't end with scf.yield
        // This is a simplified implementation - in a real implementation,
        // we'd need to iterate through the block's operations
        return false;
    }

    /// Check if an if statement contains return statements
    fn ifStatementHasReturns(_: *const StatementLowerer, if_stmt: *const lib.ast.Statements.IfNode) bool {
        // Check then branch
        for (if_stmt.then_branch.statements) |stmt| {
            if (stmt == .Return) return true;
        }

        // Check else branch if present
        if (if_stmt.else_branch) |else_branch| {
            for (else_branch.statements) |stmt| {
                if (stmt == .Return) return true;
            }
        }

        return false;
    }

    /// Get the return type from an if statement's return statements
    fn getReturnTypeFromIfStatement(self: *const StatementLowerer, _: *const lib.ast.Statements.IfNode) ?c.MlirType {
        // Use the function's return type instead of trying to infer from individual returns
        return self.current_function_return_type;
    }

    /// Get the MLIR type from an expression
    fn getExprType(self: *const StatementLowerer, expr: lib.ast.Expressions.ExprNode) c.MlirType {
        switch (expr) {
            .Literal => |lit| {
                switch (lit) {
                    .Integer => |int_lit| return self.expr_lowerer.type_mapper.toMlirType(int_lit.type_info),
                    .String => |str_lit| return self.expr_lowerer.type_mapper.toMlirType(str_lit.type_info),
                    .Bool => |bool_lit| return self.expr_lowerer.type_mapper.toMlirType(bool_lit.type_info),
                    .Address => |addr_lit| return self.expr_lowerer.type_mapper.toMlirType(addr_lit.type_info),
                    .Hex => |hex_lit| return self.expr_lowerer.type_mapper.toMlirType(hex_lit.type_info),
                    .Binary => |bin_lit| return self.expr_lowerer.type_mapper.toMlirType(bin_lit.type_info),
                    .Character => |char_lit| return self.expr_lowerer.type_mapper.toMlirType(char_lit.type_info),
                    .Bytes => |bytes_lit| return self.expr_lowerer.type_mapper.toMlirType(bytes_lit.type_info),
                }
            },
            .Binary => |bin| return self.expr_lowerer.type_mapper.toMlirType(bin.type_info),
            .Unary => |unary| return self.expr_lowerer.type_mapper.toMlirType(unary.type_info),
            .Call => |call| return self.expr_lowerer.type_mapper.toMlirType(call.type_info),
            .FieldAccess => |field| return self.expr_lowerer.type_mapper.toMlirType(field.type_info),
            .Cast => |cast| return self.expr_lowerer.type_mapper.toMlirType(cast.target_type),
            .Comptime => |_| {
                // For comptime expressions, we need to handle the block
                // For now, return a default type - this is complex to handle properly
                return c.mlirIntegerTypeGet(self.ctx, 1); // Default to i1
            },
            .Old => |_| {
                // For old() expressions, we need to get the type from the inner expression
                // For now, return a default type - this is complex to handle properly
                return c.mlirIntegerTypeGet(self.ctx, 1); // Default to i1
            },
            .Tuple => |_| {
                // For tuple expressions, we need to handle the elements
                // For now, return a default type - this is complex to handle properly
                return c.mlirIntegerTypeGet(self.ctx, 1); // Default to i1
            },
            .SwitchExpression => |switch_expr| return self.expr_lowerer.type_mapper.toMlirType(switch_expr.type_info),
            .Identifier => |ident| return self.expr_lowerer.type_mapper.toMlirType(ident.type_info),
            else => {
                // For expressions without type_info (like Assignment, CompoundAssignment, Index),
                // return a default type - this shouldn't happen in return statements
                return c.mlirIntegerTypeGet(self.ctx, 1); // Default to i1
            },
        }
    }

    /// Lower if statements with returns by using scf.if with scf.yield and single return
    fn lowerIfWithReturns(self: *const StatementLowerer, if_stmt: *const lib.ast.Statements.IfNode, condition: c.MlirValue, loc: c.MlirLocation) LoweringError!void {
        // For if statements with returns, we need to restructure the logic:
        // 1. Use scf.if with scf.yield to pass values out of regions
        // 2. Have a single func.return at the end that uses the result from scf.if

        // Create the scf.if operation with proper then/else regions
        var state = c.mlirOperationStateGet(c.mlirStringRefCreateFromCString("scf.if"), loc);

        // Add the condition operand
        c.mlirOperationStateAddOperands(&state, 1, @ptrCast(&condition));

        // Create then region
        const then_region = c.mlirRegionCreate();
        const then_block = c.mlirBlockCreate(0, null, null);
        c.mlirRegionInsertOwnedBlock(then_region, 0, then_block);
        c.mlirOperationStateAddOwnedRegions(&state, 1, @ptrCast(&then_region));

        // Create else region (always needed for scf.if)
        const else_region = c.mlirRegionCreate();
        const else_block = c.mlirBlockCreate(0, null, null);
        c.mlirRegionInsertOwnedBlock(else_region, 0, else_block);
        c.mlirOperationStateAddOwnedRegions(&state, 1, @ptrCast(&else_region));

        // Determine the result type from the return statements
        const result_type = self.getReturnTypeFromIfStatement(if_stmt);
        if (result_type) |ret_type| {
            c.mlirOperationStateAddResults(&state, 1, @ptrCast(&ret_type));
        }

        // Lower then branch - replace return statements with scf.yield
        try self.lowerBlockBodyWithYield(if_stmt.then_branch, then_block);

        // Lower else branch if present, otherwise add scf.yield to empty region
        if (if_stmt.else_branch) |else_branch| {
            try self.lowerBlockBodyWithYield(else_branch, else_block);
        } else {
            // Add scf.yield to empty else region
            const yield_op = self.ora_dialect.createScfYield(loc);
            c.mlirBlockAppendOwnedOperation(else_block, yield_op);
        }

        // Create and append the scf.if operation to the block
        const op = c.mlirOperationCreate(&state);
        c.mlirBlockAppendOwnedOperation(self.block, op);

        if (result_type) |_| {
            const result_value = c.mlirOperationGetResult(op, 0);
            // Add a func.return to return the result
            const return_op = self.ora_dialect.createFuncReturnWithValue(result_value, loc);
            c.mlirBlockAppendOwnedOperation(self.block, return_op);
        }
    }

    /// Lower block body with yield - replaces return statements with scf.yield
    fn lowerBlockBodyWithYield(self: *const StatementLowerer, block_body: lib.ast.Statements.BlockNode, target_block: c.MlirBlock) LoweringError!void {
        // Create a temporary lowerer for this block by copying the current one and changing the block
        var temp_lowerer = self.*;
        temp_lowerer.block = target_block;

        // Lower each statement, replacing returns with yields
        for (block_body.statements) |stmt| {
            switch (stmt) {
                .Return => |ret| {
                    // Replace return with scf.yield
                    const loc = temp_lowerer.fileLoc(ret.span);

                    if (ret.value) |e| {
                        const v = temp_lowerer.expr_lowerer.lowerExpression(&e);
                        const yield_op = temp_lowerer.ora_dialect.createScfYieldWithValues(&[_]c.MlirValue{v}, loc);
                        c.mlirBlockAppendOwnedOperation(target_block, yield_op);
                    } else {
                        const yield_op = temp_lowerer.ora_dialect.createScfYield(loc);
                        c.mlirBlockAppendOwnedOperation(target_block, yield_op);
                    }
                },
                else => {
                    // Lower other statements normally
                    try temp_lowerer.lowerStatement(&stmt);
                },
            }
        }
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
        _ = StatementLowerer.init(self.ctx, before_block, self.type_mapper, self.expr_lowerer, self.param_map, self.storage_map, self.local_var_map, self.locations, self.symbol_table, self.allocator, self.current_function_return_type, self.ora_dialect);

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
        const yield_op = self.ora_dialect.createScfYield(loc);
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
        const yield_op = self.ora_dialect.createScfYield(loc);
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
        const yield_op = self.ora_dialect.createScfYield(loc);
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
                        const result_ty = c.mlirIntegerTypeGet(self.ctx, constants.DEFAULT_INTEGER_BITS);
                        const indices = [_]u32{@intCast(i)};
                        const field_access_op = self.ora_dialect.createLlvmExtractvalue(item_var, &indices, result_ty, loc);
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
        const yield_op = self.ora_dialect.createScfYield(loc);
        c.mlirBlockAppendOwnedOperation(body_block, yield_op);
    }

    /// Helper to load value from memref if needed
    fn ensureValue(self: *const StatementLowerer, val: c.MlirValue, loc: c.MlirLocation) c.MlirValue {
        const val_type = c.mlirValueGetType(val);

        // If it's a memref, load the value
        if (c.mlirTypeIsAMemRef(val_type)) {
            const element_type = c.mlirShapedTypeGetElementType(val_type);
            var load_state = c.mlirOperationStateGet(c.mlirStringRefCreateFromCString("memref.load"), loc);
            c.mlirOperationStateAddOperands(&load_state, 1, @ptrCast(&val));
            c.mlirOperationStateAddResults(&load_state, 1, @ptrCast(&element_type));
            const load_op = c.mlirOperationCreate(&load_state);
            c.mlirBlockAppendOwnedOperation(self.block, load_op);
            return c.mlirOperationGetResult(load_op, 0);
        }

        return val;
    }

    /// Lower switch statements using cf.switch with case blocks
    pub fn lowerSwitch(self: *const StatementLowerer, switch_stmt: *const lib.ast.Statements.SwitchNode) LoweringError!void {
        const loc = self.fileLoc(switch_stmt.span);

        // Lower the condition expression
        const condition_raw = self.expr_lowerer.lowerExpression(&switch_stmt.condition);

        // Ensure we have a value (load from memref if needed)
        const condition = self.ensureValue(condition_raw, loc);

        // Lower switch statement as a chain of nested scf.if operations
        // Start with the first case and nest subsequent cases in the else block
        // Pass the default case to be handled in the final else block
        try self.lowerSwitchCases(switch_stmt.cases, condition, 0, self.block, loc, switch_stmt.default_case);
    }

    /// Recursively lower switch cases as nested if-else-if chain
    fn lowerSwitchCases(self: *const StatementLowerer, cases: []const lib.ast.Expressions.SwitchCase, condition: c.MlirValue, case_idx: usize, target_block: c.MlirBlock, loc: c.MlirLocation, default_case: ?lib.ast.Statements.BlockNode) LoweringError!void {
        // Base case: no more cases, handle default if present
        if (case_idx >= cases.len) {
            if (default_case) |default_block| {
                try self.lowerBlockBody(default_block, target_block);
            }
            return;
        }

        const case = cases[case_idx];

        // Create condition for this case
        const case_condition = switch (case.pattern) {
            .Literal => |lit| blk: {
                const case_value = self.expr_lowerer.lowerLiteral(&lit.value);
                // Compare condition with case value using arith.cmpi
                var cmp_state = c.mlirOperationStateGet(c.mlirStringRefCreateFromCString("arith.cmpi"), loc);

                // Add predicate attribute (0 = eq)
                const pred_id = c.mlirIdentifierGet(self.ctx, c.mlirStringRefCreateFromCString("predicate"));
                const pred_attr = c.mlirIntegerAttrGet(c.mlirIntegerTypeGet(self.ctx, 64), 0); // eq
                var attrs = [_]c.MlirNamedAttribute{c.mlirNamedAttributeGet(pred_id, pred_attr)};
                c.mlirOperationStateAddAttributes(&cmp_state, attrs.len, &attrs);

                c.mlirOperationStateAddOperands(&cmp_state, 2, @ptrCast(&[_]c.MlirValue{ condition, case_value }));
                c.mlirOperationStateAddResults(&cmp_state, 1, @ptrCast(&c.mlirIntegerTypeGet(self.ctx, 1)));

                const cmp_op = c.mlirOperationCreate(&cmp_state);
                c.mlirBlockAppendOwnedOperation(self.block, cmp_op);
                break :blk c.mlirOperationGetResult(cmp_op, 0);
            },
            .Range => |range| blk: {
                // Create range check: condition >= start && condition <= end
                const start_val = self.expr_lowerer.lowerExpression(range.start);
                const end_val = self.expr_lowerer.lowerExpression(range.end);

                // Lower bound check: condition >= start
                var lower_cmp_state = c.mlirOperationStateGet(c.mlirStringRefCreateFromCString("arith.cmpi"), loc);
                const pred_id = c.mlirIdentifierGet(self.ctx, c.mlirStringRefCreateFromCString("predicate"));
                const pred_sge_attr = c.mlirIntegerAttrGet(c.mlirIntegerTypeGet(self.ctx, 64), 5); // sge
                var lower_attrs = [_]c.MlirNamedAttribute{c.mlirNamedAttributeGet(pred_id, pred_sge_attr)};
                c.mlirOperationStateAddAttributes(&lower_cmp_state, lower_attrs.len, &lower_attrs);
                c.mlirOperationStateAddOperands(&lower_cmp_state, 2, @ptrCast(&[_]c.MlirValue{ condition, start_val }));
                c.mlirOperationStateAddResults(&lower_cmp_state, 1, @ptrCast(&c.mlirIntegerTypeGet(self.ctx, 1)));
                const lower_cmp_op = c.mlirOperationCreate(&lower_cmp_state);
                c.mlirBlockAppendOwnedOperation(self.block, lower_cmp_op);
                const lower_bound = c.mlirOperationGetResult(lower_cmp_op, 0);

                // Upper bound check: condition <= end
                var upper_cmp_state = c.mlirOperationStateGet(c.mlirStringRefCreateFromCString("arith.cmpi"), loc);
                const pred_sle_attr = c.mlirIntegerAttrGet(c.mlirIntegerTypeGet(self.ctx, 64), 3); // sle
                var upper_attrs = [_]c.MlirNamedAttribute{c.mlirNamedAttributeGet(pred_id, pred_sle_attr)};
                c.mlirOperationStateAddAttributes(&upper_cmp_state, upper_attrs.len, &upper_attrs);
                c.mlirOperationStateAddOperands(&upper_cmp_state, 2, @ptrCast(&[_]c.MlirValue{ condition, end_val }));
                c.mlirOperationStateAddResults(&upper_cmp_state, 1, @ptrCast(&c.mlirIntegerTypeGet(self.ctx, 1)));
                const upper_cmp_op = c.mlirOperationCreate(&upper_cmp_state);
                c.mlirBlockAppendOwnedOperation(self.block, upper_cmp_op);
                const upper_bound = c.mlirOperationGetResult(upper_cmp_op, 0);

                // AND the two conditions
                var and_state = c.mlirOperationStateGet(c.mlirStringRefCreateFromCString("arith.andi"), loc);
                c.mlirOperationStateAddOperands(&and_state, 2, @ptrCast(&[_]c.MlirValue{ lower_bound, upper_bound }));
                c.mlirOperationStateAddResults(&and_state, 1, @ptrCast(&c.mlirIntegerTypeGet(self.ctx, 1)));
                const and_op = c.mlirOperationCreate(&and_state);
                c.mlirBlockAppendOwnedOperation(self.block, and_op);
                break :blk c.mlirOperationGetResult(and_op, 0);
            },
            .EnumValue => |enum_val| blk: {
                // Look up enum variant index
                if (self.symbol_table) |st| {
                    if (st.lookupType(enum_val.enum_name)) |enum_type| {
                        if (enum_type.getVariantIndex(enum_val.variant_name)) |variant_idx| {
                            // Create constant for variant index
                            var const_state = c.mlirOperationStateGet(c.mlirStringRefCreateFromCString("arith.constant"), loc);
                            const value_attr = c.mlirIntegerAttrGet(c.mlirIntegerTypeGet(self.ctx, 256), @intCast(variant_idx));
                            const value_id = c.mlirIdentifierGet(self.ctx, c.mlirStringRefCreateFromCString("value"));
                            var attrs = [_]c.MlirNamedAttribute{c.mlirNamedAttributeGet(value_id, value_attr)};
                            c.mlirOperationStateAddAttributes(&const_state, attrs.len, &attrs);
                            c.mlirOperationStateAddResults(&const_state, 1, @ptrCast(&c.mlirIntegerTypeGet(self.ctx, 256)));
                            const const_op = c.mlirOperationCreate(&const_state);
                            c.mlirBlockAppendOwnedOperation(self.block, const_op);
                            const variant_const = c.mlirOperationGetResult(const_op, 0);

                            // Compare condition with variant constant
                            var cmp_state = c.mlirOperationStateGet(c.mlirStringRefCreateFromCString("arith.cmpi"), loc);
                            const pred_id = c.mlirIdentifierGet(self.ctx, c.mlirStringRefCreateFromCString("predicate"));
                            const pred_attr = c.mlirIntegerAttrGet(c.mlirIntegerTypeGet(self.ctx, 64), 0); // eq
                            var cmp_attrs = [_]c.MlirNamedAttribute{c.mlirNamedAttributeGet(pred_id, pred_attr)};
                            c.mlirOperationStateAddAttributes(&cmp_state, cmp_attrs.len, &cmp_attrs);
                            c.mlirOperationStateAddOperands(&cmp_state, 2, @ptrCast(&[_]c.MlirValue{ condition, variant_const }));
                            c.mlirOperationStateAddResults(&cmp_state, 1, @ptrCast(&c.mlirIntegerTypeGet(self.ctx, 1)));
                            const cmp_op = c.mlirOperationCreate(&cmp_state);
                            c.mlirBlockAppendOwnedOperation(self.block, cmp_op);
                            break :blk c.mlirOperationGetResult(cmp_op, 0);
                        }
                    }
                }
                // Fallback: create a false condition
                var const_state = c.mlirOperationStateGet(c.mlirStringRefCreateFromCString("arith.constant"), loc);
                const value_attr = c.mlirIntegerAttrGet(c.mlirIntegerTypeGet(self.ctx, 1), 0);
                const value_id = c.mlirIdentifierGet(self.ctx, c.mlirStringRefCreateFromCString("value"));
                var attrs = [_]c.MlirNamedAttribute{c.mlirNamedAttributeGet(value_id, value_attr)};
                c.mlirOperationStateAddAttributes(&const_state, attrs.len, &attrs);
                c.mlirOperationStateAddResults(&const_state, 1, @ptrCast(&c.mlirIntegerTypeGet(self.ctx, 1)));
                const const_op = c.mlirOperationCreate(&const_state);
                c.mlirBlockAppendOwnedOperation(self.block, const_op);
                break :blk c.mlirOperationGetResult(const_op, 0);
            },
            .Else => blk: {
                // Else pattern always matches, create a true condition
                var const_state = c.mlirOperationStateGet(c.mlirStringRefCreateFromCString("arith.constant"), loc);
                const value_attr = c.mlirIntegerAttrGet(c.mlirIntegerTypeGet(self.ctx, 1), 1);
                const value_id = c.mlirIdentifierGet(self.ctx, c.mlirStringRefCreateFromCString("value"));
                var attrs = [_]c.MlirNamedAttribute{c.mlirNamedAttributeGet(value_id, value_attr)};
                c.mlirOperationStateAddAttributes(&const_state, attrs.len, &attrs);
                c.mlirOperationStateAddResults(&const_state, 1, @ptrCast(&c.mlirIntegerTypeGet(self.ctx, 1)));
                const const_op = c.mlirOperationCreate(&const_state);
                c.mlirBlockAppendOwnedOperation(self.block, const_op);
                break :blk c.mlirOperationGetResult(const_op, 0);
            },
        };

        // Create scf.if operation for this case
        var if_state = c.mlirOperationStateGet(c.mlirStringRefCreateFromCString("scf.if"), loc);
        c.mlirOperationStateAddOperands(&if_state, 1, @ptrCast(&case_condition));

        // Create then region for the case body
        const then_region = c.mlirRegionCreate();
        const then_block = c.mlirBlockCreate(0, null, null);
        c.mlirRegionInsertOwnedBlock(then_region, 0, then_block);

        // Create a new expression lowerer for this case body with the correct block
        const case_expr_lowerer = ExpressionLowerer.init(self.ctx, then_block, self.type_mapper, self.param_map, self.storage_map, self.local_var_map, self.locations, self.ora_dialect);

        // Lower case body into the then block
        switch (case.body) {
            .Expression => |expr| {
                _ = case_expr_lowerer.lowerExpression(expr);
            },
            .Block => |block| {
                try self.lowerBlockBody(block, then_block);
            },
            .LabeledBlock => |labeled| {
                try self.lowerBlockBody(labeled.block, then_block);
            },
        }

        // Add yield to then block
        var yield_state = c.mlirOperationStateGet(c.mlirStringRefCreateFromCString("scf.yield"), loc);
        const yield_op = c.mlirOperationCreate(&yield_state);
        c.mlirBlockAppendOwnedOperation(then_block, yield_op);

        // Create else region for next cases
        const else_region = c.mlirRegionCreate();
        const else_block = c.mlirBlockCreate(0, null, null);
        c.mlirRegionInsertOwnedBlock(else_region, 0, else_block);

        // Recursively process remaining cases in the else block
        try self.lowerSwitchCases(cases, condition, case_idx + 1, else_block, loc, default_case);

        // Only add yield if we're not at the last case with a default
        // (the base case with default block will add its own yield)
        const is_last_case = (case_idx + 1 >= cases.len);
        if (!(is_last_case and default_case != null)) {
            var else_yield_state = c.mlirOperationStateGet(c.mlirStringRefCreateFromCString("scf.yield"), loc);
            const else_yield_op = c.mlirOperationCreate(&else_yield_state);
            c.mlirBlockAppendOwnedOperation(else_block, else_yield_op);
        }

        // Add regions to the if operation
        c.mlirOperationStateAddOwnedRegions(&if_state, 1, @ptrCast(&then_region));
        c.mlirOperationStateAddOwnedRegions(&if_state, 1, @ptrCast(&else_region));

        const if_op = c.mlirOperationCreate(&if_state);
        c.mlirBlockAppendOwnedOperation(target_block, if_op);
    }

    /// Lower switch cases with label context for continue support
    fn lowerSwitchCasesWithLabel(self: *const StatementLowerer, cases: []const lib.ast.Expressions.SwitchCase, condition: c.MlirValue, case_idx: usize, target_block: c.MlirBlock, loc: c.MlirLocation, default_case: ?lib.ast.Statements.BlockNode, label: []const u8, continue_flag_memref: c.MlirValue, value_memref: c.MlirValue) LoweringError!void {
        // Create label context
        const label_ctx = LabelContext{
            .label = label,
            .continue_flag_memref = continue_flag_memref,
            .value_memref = value_memref,
        };

        // Create new statement lowerer with label context
        var lowerer_with_label = StatementLowerer.init(
            self.ctx,
            target_block,
            self.type_mapper,
            self.expr_lowerer,
            self.param_map,
            self.storage_map,
            self.local_var_map,
            self.locations,
            self.symbol_table,
            self.allocator,
            self.current_function_return_type,
            self.ora_dialect,
        );
        lowerer_with_label.label_context = &label_ctx;

        // Lower the switch cases
        try lowerer_with_label.lowerSwitchCases(cases, condition, case_idx, target_block, loc, default_case);
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
                        // For stack variables, check if we have a memref (mutable var) or direct SSA value (let/const)
                        if (self.local_var_map) |lvm| {
                            if (lvm.getLocalVar(ident.name)) |var_value| {
                                // Check if this is a memref type (mutable variable)
                                const var_type = c.mlirValueGetType(var_value);
                                if (c.mlirTypeIsAMemRef(var_type)) {
                                    // It's a memref - use memref.store
                                    const store_op = self.ora_dialect.createMemrefStore(value, var_value, &[_]c.MlirValue{}, loc);
                                    c.mlirBlockAppendOwnedOperation(self.block, store_op);
                                    return;
                                } else {
                                    // It's an immutable variable - can't reassign
                                    std.debug.print("ERROR: Cannot assign to immutable variable: {s}\n", .{ident.name});
                                    return LoweringError.InvalidLValue;
                                }
                            }
                        }
                        // Variable not found in map
                        std.debug.print("ERROR: Variable not found for assignment: {s}\n", .{ident.name});
                        return LoweringError.InvalidLValue;
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

        // std.debug.print("ERROR: Variable not found for assignment: {s}\n", .{ident.name});
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

    /// Lower labeled switch using scf.while to support continue :label (value)
    fn lowerLabeledSwitch(self: *const StatementLowerer, labeled_block: *const lib.ast.Statements.LabeledBlockNode) LoweringError!void {
        const loc = self.fileLoc(labeled_block.span);

        // Find the switch statement
        const switch_stmt = blk: {
            for (labeled_block.block.statements) |stmt| {
                if (stmt == .Switch) break :blk &stmt.Switch;
            }
            return LoweringError.InvalidSwitch;
        };

        // Create memref for continue flag (i1)
        const i1_type = c.mlirIntegerTypeGet(self.ctx, 1);
        const empty_attr = c.mlirAttributeGetNull();
        const continue_flag_memref_type = c.mlirMemRefTypeGet(i1_type, 0, null, empty_attr, empty_attr);
        var continue_flag_alloca_state = c.mlirOperationStateGet(c.mlirStringRefCreateFromCString("memref.alloca"), loc);
        c.mlirOperationStateAddResults(&continue_flag_alloca_state, 1, @ptrCast(&continue_flag_memref_type));
        const continue_flag_alloca = c.mlirOperationCreate(&continue_flag_alloca_state);
        c.mlirBlockAppendOwnedOperation(self.block, continue_flag_alloca);
        const continue_flag_memref = c.mlirOperationGetResult(continue_flag_alloca, 0);

        // Create memref for switch value (same type as switch condition)
        const condition_raw = self.expr_lowerer.lowerExpression(&switch_stmt.condition);
        const initial_value = self.ensureValue(condition_raw, loc);
        const value_type = c.mlirValueGetType(initial_value);
        const value_memref_type = c.mlirMemRefTypeGet(value_type, 0, null, empty_attr, empty_attr);
        var value_alloca_state = c.mlirOperationStateGet(c.mlirStringRefCreateFromCString("memref.alloca"), loc);
        c.mlirOperationStateAddResults(&value_alloca_state, 1, @ptrCast(&value_memref_type));
        const value_alloca = c.mlirOperationCreate(&value_alloca_state);
        c.mlirBlockAppendOwnedOperation(self.block, value_alloca);
        const value_memref = c.mlirOperationGetResult(value_alloca, 0);

        // Initialize continue_flag to true
        var true_const_state = c.mlirOperationStateGet(c.mlirStringRefCreateFromCString("arith.constant"), loc);
        const true_attr = c.mlirIntegerAttrGet(i1_type, 1);
        const value_id = c.mlirIdentifierGet(self.ctx, c.mlirStringRefCreateFromCString("value"));
        var true_attrs = [_]c.MlirNamedAttribute{c.mlirNamedAttributeGet(value_id, true_attr)};
        c.mlirOperationStateAddAttributes(&true_const_state, true_attrs.len, &true_attrs);
        c.mlirOperationStateAddResults(&true_const_state, 1, @ptrCast(&i1_type));
        const true_const = c.mlirOperationCreate(&true_const_state);
        c.mlirBlockAppendOwnedOperation(self.block, true_const);
        const true_val = c.mlirOperationGetResult(true_const, 0);

        // Store true to continue_flag
        var init_flag_store_state = c.mlirOperationStateGet(c.mlirStringRefCreateFromCString("memref.store"), loc);
        c.mlirOperationStateAddOperands(&init_flag_store_state, 2, @ptrCast(&[_]c.MlirValue{ true_val, continue_flag_memref }));
        const init_flag_store = c.mlirOperationCreate(&init_flag_store_state);
        c.mlirBlockAppendOwnedOperation(self.block, init_flag_store);

        // Store initial value to value_memref
        var init_value_store_state = c.mlirOperationStateGet(c.mlirStringRefCreateFromCString("memref.store"), loc);
        c.mlirOperationStateAddOperands(&init_value_store_state, 2, @ptrCast(&[_]c.MlirValue{ initial_value, value_memref }));
        const init_value_store = c.mlirOperationCreate(&init_value_store_state);
        c.mlirBlockAppendOwnedOperation(self.block, init_value_store);

        // Create scf.while operation
        var while_state = c.mlirOperationStateGet(c.mlirStringRefCreateFromCString("scf.while"), loc);

        // Create before region (condition check)
        const before_region = c.mlirRegionCreate();
        const before_block = c.mlirBlockCreate(0, null, null);
        c.mlirRegionInsertOwnedBlock(before_region, 0, before_block);

        // In before block: load continue_flag and use scf.condition
        var load_flag_state = c.mlirOperationStateGet(c.mlirStringRefCreateFromCString("memref.load"), loc);
        c.mlirOperationStateAddOperands(&load_flag_state, 1, @ptrCast(&continue_flag_memref));
        c.mlirOperationStateAddResults(&load_flag_state, 1, @ptrCast(&i1_type));
        const load_flag = c.mlirOperationCreate(&load_flag_state);
        c.mlirBlockAppendOwnedOperation(before_block, load_flag);
        const should_continue = c.mlirOperationGetResult(load_flag, 0);

        var condition_state = c.mlirOperationStateGet(c.mlirStringRefCreateFromCString("scf.condition"), loc);
        c.mlirOperationStateAddOperands(&condition_state, 1, @ptrCast(&should_continue));
        const condition_op = c.mlirOperationCreate(&condition_state);
        c.mlirBlockAppendOwnedOperation(before_block, condition_op);

        // Create after region (loop body with switch)
        const after_region = c.mlirRegionCreate();
        const after_block = c.mlirBlockCreate(0, null, null);
        c.mlirRegionInsertOwnedBlock(after_region, 0, after_block);

        // In after block: set continue_flag to false, then execute switch
        var false_const_state = c.mlirOperationStateGet(c.mlirStringRefCreateFromCString("arith.constant"), loc);
        const false_attr = c.mlirIntegerAttrGet(i1_type, 0);
        var false_attrs = [_]c.MlirNamedAttribute{c.mlirNamedAttributeGet(value_id, false_attr)};
        c.mlirOperationStateAddAttributes(&false_const_state, false_attrs.len, &false_attrs);
        c.mlirOperationStateAddResults(&false_const_state, 1, @ptrCast(&i1_type));
        const false_const = c.mlirOperationCreate(&false_const_state);
        c.mlirBlockAppendOwnedOperation(after_block, false_const);
        const false_val = c.mlirOperationGetResult(false_const, 0);

        var reset_flag_store_state = c.mlirOperationStateGet(c.mlirStringRefCreateFromCString("memref.store"), loc);
        c.mlirOperationStateAddOperands(&reset_flag_store_state, 2, @ptrCast(&[_]c.MlirValue{ false_val, continue_flag_memref }));
        const reset_flag_store = c.mlirOperationCreate(&reset_flag_store_state);
        c.mlirBlockAppendOwnedOperation(after_block, reset_flag_store);

        // Load the switch value
        var load_value_state = c.mlirOperationStateGet(c.mlirStringRefCreateFromCString("memref.load"), loc);
        c.mlirOperationStateAddOperands(&load_value_state, 1, @ptrCast(&value_memref));
        c.mlirOperationStateAddResults(&load_value_state, 1, @ptrCast(&value_type));
        const load_value = c.mlirOperationCreate(&load_value_state);
        c.mlirBlockAppendOwnedOperation(after_block, load_value);
        const switch_value = c.mlirOperationGetResult(load_value, 0);

        // Lower switch cases in after_block (passing label and memrefs for continue support)
        try self.lowerSwitchCasesWithLabel(switch_stmt.cases, switch_value, 0, after_block, loc, switch_stmt.default_case, labeled_block.label, continue_flag_memref, value_memref);

        // Add scf.yield at the end of after_block
        var yield_state = c.mlirOperationStateGet(c.mlirStringRefCreateFromCString("scf.yield"), loc);
        const yield_op = c.mlirOperationCreate(&yield_state);
        c.mlirBlockAppendOwnedOperation(after_block, yield_op);

        // Add regions to while operation
        c.mlirOperationStateAddOwnedRegions(&while_state, 1, @ptrCast(&before_region));
        c.mlirOperationStateAddOwnedRegions(&while_state, 1, @ptrCast(&after_region));

        const while_op = c.mlirOperationCreate(&while_state);
        c.mlirBlockAppendOwnedOperation(self.block, while_op);
    }

    /// Lower labeled block statements using scf.execute_region or scf.while for switches
    pub fn lowerLabeledBlock(self: *const StatementLowerer, labeled_block: *const lib.ast.Statements.LabeledBlockNode) LoweringError!void {
        const loc = self.fileLoc(labeled_block.span);

        // Check if this labeled block contains a switch statement
        const contains_switch = blk: {
            for (labeled_block.block.statements) |stmt| {
                if (stmt == .Switch) break :blk true;
            }
            break :blk false;
        };

        if (contains_switch) {
            // For labeled switches, use scf.while to support continue :label (value)
            try self.lowerLabeledSwitch(labeled_block);
        } else {
            // For regular labeled blocks, use scf.execute_region
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

        // Create a new expression lowerer with the correct block context
        const expr_lowerer = ExpressionLowerer.init(self.ctx, block, self.type_mapper, self.param_map, self.storage_map, self.local_var_map, self.locations, self.ora_dialect);

        // Process each statement in the block
        for (b.statements) |*s| {
            // Create a new statement lowerer for this block with the correct expression lowerer
            var stmt_lowerer = StatementLowerer.init(self.ctx, block, self.type_mapper, &expr_lowerer, self.param_map, self.storage_map, self.local_var_map, self.locations, self.symbol_table, self.allocator, self.current_function_return_type, self.ora_dialect);

            // Preserve label context if present
            stmt_lowerer.label_context = self.label_context;

            // Lower the statement with error handling
            stmt_lowerer.lowerStatement(s) catch |err| {
                // std.debug.print("ERROR: Failed to lower statement type {s}: {s}\n", .{ @tagName(s.*), @errorName(err) });

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
        return self.locations.createLocation(span);
    }

    /// Create range check for switch case patterns using standard MLIR operations
    fn createRangeCheck(self: *const StatementLowerer, _: c.MlirValue, _: c.MlirValue, _: bool, span: lib.ast.SourceSpan) c.MlirValue {
        const loc = self.fileLoc(span);
        const result_ty = c.mlirIntegerTypeGet(self.ctx, 1);

        // For range patterns like `1...10`, we need to check if the switch value is within the range
        // This requires the switch value, which we don't have here. Instead, we'll create a placeholder
        // that will be used in the switch comparison logic.

        // Create a constant true for now - the actual range check will be done in the switch comparison
        var const_state = c.mlirOperationStateGet(c.mlirStringRefCreateFromCString("arith.constant"), loc);
        const true_attr = c.mlirBoolAttrGet(self.ctx, 1);
        const true_id = c.mlirIdentifierGet(self.ctx, c.mlirStringRefCreateFromCString("value"));
        var const_attrs = [_]c.MlirNamedAttribute{
            c.mlirNamedAttributeGet(true_id, true_attr),
        };
        c.mlirOperationStateAddAttributes(&const_state, const_attrs.len, &const_attrs);
        c.mlirOperationStateAddResults(&const_state, 1, @ptrCast(&result_ty));

        const const_op = c.mlirOperationCreate(&const_state);
        c.mlirBlockAppendOwnedOperation(self.block, const_op);
        return c.mlirOperationGetResult(const_op, 0);
    }

    /// Create enum constant for switch case patterns using standard MLIR operations
    fn createEnumConstant(self: *const StatementLowerer, _: []const u8, _: []const u8, span: lib.ast.SourceSpan) c.MlirValue {
        const loc = self.fileLoc(span);
        const result_ty = c.mlirIntegerTypeGet(self.ctx, constants.DEFAULT_INTEGER_BITS);

        // For now, create a placeholder constant. In a real implementation, we would:
        // 1. Look up the enum definition to get the variant value
        // 2. Create an arith.constant with that value
        // For now, we'll use a placeholder value of 0

        var const_state = c.mlirOperationStateGet(c.mlirStringRefCreateFromCString("arith.constant"), loc);
        const value_attr = c.mlirIntegerAttrGet(c.mlirIntegerTypeGet(self.ctx, constants.DEFAULT_INTEGER_BITS), 0);
        const value_id = c.mlirIdentifierGet(self.ctx, c.mlirStringRefCreateFromCString("value"));
        var const_attrs = [_]c.MlirNamedAttribute{
            c.mlirNamedAttributeGet(value_id, value_attr),
        };
        c.mlirOperationStateAddAttributes(&const_state, const_attrs.len, &const_attrs);
        c.mlirOperationStateAddResults(&const_state, 1, @ptrCast(&result_ty));

        const const_op = c.mlirOperationCreate(&const_state);
        c.mlirBlockAppendOwnedOperation(self.block, const_op);
        return c.mlirOperationGetResult(const_op, 0);
    }
};
