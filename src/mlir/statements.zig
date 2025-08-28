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

    pub fn init(ctx: c.MlirContext, block: c.MlirBlock, type_mapper: *const TypeMapper, expr_lowerer: *const ExpressionLowerer, param_map: ?*const ParamMap, storage_map: ?*const StorageMap, local_var_map: ?*LocalVarMap, locations: LocationTracker) StatementLowerer {
        return .{
            .ctx = ctx,
            .block = block,
            .type_mapper = type_mapper,
            .expr_lowerer = expr_lowerer,
            .param_map = param_map,
            .storage_map = storage_map,
            .local_var_map = local_var_map,
            .locations = locations,
        };
    }

    /// Main dispatch function for lowering statements
    pub fn lowerStatement(self: *const StatementLowerer, stmt: *const lib.ast.Statements.StmtNode) void {
        switch (stmt.*) {
            .Return => |ret| {
                self.lowerReturn(&ret);
            },
            .VariableDecl => |var_decl| {
                self.lowerVariableDecl(&var_decl);
            },
            .DestructuringAssignment => |assignment| {
                self.lowerDestructuringAssignment(&assignment);
            },
            .CompoundAssignment => |assignment| {
                self.lowerCompoundAssignment(&assignment);
            },
            .If => |if_stmt| {
                self.lowerIf(&if_stmt);
            },
            .While => |while_stmt| {
                self.lowerWhile(&while_stmt);
            },
            .ForLoop => |for_stmt| {
                self.lowerFor(&for_stmt);
            },
            .Switch => |switch_stmt| {
                self.lowerSwitch(&switch_stmt);
            },
            .Expr => |expr| {
                self.lowerExpressionStatement(&expr);
            },
            .LabeledBlock => |labeled_block| {
                self.lowerLabeledBlock(&labeled_block);
            },
            .Continue => {
                // For now, skip continue statements
                // TODO: Add proper continue statement handling
            },
            else => @panic("Unhandled statement type"),
        }
    }

    /// Lower return statements
    pub fn lowerReturn(self: *const StatementLowerer, ret: *const lib.ast.Statements.ReturnNode) void {
        var state = c.mlirOperationStateGet(c.mlirStringRefCreateFromCString("func.return"), self.fileLoc(ret.span));
        if (ret.value) |e| {
            const v = self.expr_lowerer.lowerExpression(&e);
            c.mlirOperationStateAddOperands(&state, 1, @ptrCast(&v));
        }
        const op = c.mlirOperationCreate(&state);
        c.mlirBlockAppendOwnedOperation(self.block, op);
    }

    /// Lower variable declaration statements
    pub fn lowerVariableDecl(self: *const StatementLowerer, var_decl: *const lib.ast.Statements.VariableDeclNode) void {
        std.debug.print("DEBUG: Processing variable declaration: {s} (region: {s})\n", .{ var_decl.name, @tagName(var_decl.region) });
        // Handle variable declarations based on memory region
        switch (var_decl.region) {
            .Stack => {
                // This is a local variable - we need to handle it properly
                if (var_decl.value) |init_expr| {
                    // Lower the initializer expression
                    const init_value = self.expr_lowerer.lowerExpression(&init_expr.*);

                    // Store the local variable in our map for later reference
                    if (self.local_var_map) |lvm| {
                        lvm.addLocalVar(var_decl.name, init_value) catch {
                            std.debug.print("WARNING: Failed to add local variable to map: {s}\n", .{var_decl.name});
                        };
                    }
                } else {
                    // Local variable without initializer - create a default value and store it
                    if (self.local_var_map) |lvm| {
                        // Create a default value (0 for now)
                        const default_ty = c.mlirIntegerTypeGet(self.ctx, constants.DEFAULT_INTEGER_BITS);
                        var const_state = c.mlirOperationStateGet(c.mlirStringRefCreateFromCString("arith.constant"), self.fileLoc(var_decl.span));
                        c.mlirOperationStateAddResults(&const_state, 1, @ptrCast(&default_ty));
                        const attr = c.mlirIntegerAttrGet(default_ty, 0);
                        const value_id = c.mlirIdentifierGet(self.ctx, c.mlirStringRefCreateFromCString("value"));
                        var attrs = [_]c.MlirNamedAttribute{c.mlirNamedAttributeGet(value_id, attr)};
                        c.mlirOperationStateAddAttributes(&const_state, attrs.len, &attrs);
                        const const_op = c.mlirOperationCreate(&const_state);
                        c.mlirBlockAppendOwnedOperation(self.block, const_op);
                        const default_value = c.mlirOperationGetResult(const_op, 0);

                        lvm.addLocalVar(var_decl.name, default_value) catch {
                            std.debug.print("WARNING: Failed to add local variable to map: {s}\n", .{var_decl.name});
                        };
                        std.debug.print("DEBUG: Added local variable to map: {s}\n", .{var_decl.name});
                    }
                }
            },
            .Storage => {
                // Storage variables are handled at the contract level
                // Just lower the initializer if present
                if (var_decl.value) |init_expr| {
                    _ = self.expr_lowerer.lowerExpression(&init_expr.*);
                }
            },
            .Memory => {
                // Memory variables are temporary and should be handled like local variables
                if (var_decl.value) |init_expr| {
                    const init_value = self.expr_lowerer.lowerExpression(&init_expr.*);

                    // Store the memory variable in our local variable map for now
                    // In a full implementation, we'd allocate memory with scf.alloca
                    if (self.local_var_map) |lvm| {
                        lvm.addLocalVar(var_decl.name, init_value) catch {
                            std.debug.print("WARNING: Failed to add memory variable to map: {s}\n", .{var_decl.name});
                        };
                    }
                } else {
                    // Memory variable without initializer - create a default value and store it
                    if (self.local_var_map) |lvm| {
                        // Create a default value (0 for now)
                        const default_ty = c.mlirIntegerTypeGet(self.ctx, constants.DEFAULT_INTEGER_BITS);
                        var const_state = c.mlirOperationStateGet(c.mlirStringRefCreateFromCString("arith.constant"), self.fileLoc(var_decl.span));
                        c.mlirOperationStateAddResults(&const_state, 1, @ptrCast(&default_ty));
                        const attr = c.mlirIntegerAttrGet(default_ty, 0);
                        const value_id = c.mlirIdentifierGet(self.ctx, c.mlirStringRefCreateFromCString("value"));
                        var attrs = [_]c.MlirNamedAttribute{c.mlirNamedAttributeGet(value_id, attr)};
                        c.mlirOperationStateAddAttributes(&const_state, attrs.len, &attrs);
                        const const_op = c.mlirOperationCreate(&const_state);
                        c.mlirBlockAppendOwnedOperation(self.block, const_op);
                        const default_value = c.mlirOperationGetResult(const_op, 0);

                        lvm.addLocalVar(var_decl.name, default_value) catch {
                            std.debug.print("WARNING: Failed to add memory variable to map: {s}\n", .{var_decl.name});
                        };
                        std.debug.print("DEBUG: Added memory variable to map: {s}\n", .{var_decl.name});
                    }
                }
            },
            .TStore => {
                // Transient storage variables are persistent across calls but temporary
                // For now, treat them like storage variables
                if (var_decl.value) |init_expr| {
                    _ = self.expr_lowerer.lowerExpression(&init_expr.*);
                }
            },
        }
    }

    /// Lower destructuring assignment statements
    pub fn lowerDestructuringAssignment(self: *const StatementLowerer, assignment: *const lib.ast.Statements.DestructuringAssignmentNode) void {
        // Debug: print what we're assigning to
        std.debug.print("DEBUG: Assignment to: {s}\n", .{@tagName(assignment.pattern)});

        // For now, just skip destructuring assignments
        // TODO: Implement proper destructuring assignment handling
        // Note: assignment.value contains the expression to destructure
        _ = self; // Use self parameter
        _ = assignment.pattern; // Use the parameter to avoid warning
        _ = assignment.value; // Use the parameter to avoid warning
        _ = assignment.span; // Use the parameter to avoid warning
    }

    /// Lower expression-level compound assignment expressions
    pub fn lowerCompoundAssignmentExpr(self: *const StatementLowerer, assignment: *const lib.ast.Expressions.CompoundAssignmentExpr) void {
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
    pub fn lowerCompoundAssignment(self: *const StatementLowerer, assignment: *const lib.ast.Statements.CompoundAssignmentNode) void {
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

    /// Lower if statements
    pub fn lowerIf(self: *const StatementLowerer, if_stmt: *const lib.ast.Statements.IfNode) void {
        // Lower the condition expression
        const condition = self.expr_lowerer.lowerExpression(&if_stmt.condition);

        // Create the scf.if operation with proper then/else regions
        var state = c.mlirOperationStateGet(c.mlirStringRefCreateFromCString("scf.if"), self.fileLoc(if_stmt.span));

        // Add the condition operand
        c.mlirOperationStateAddOperands(&state, 1, @ptrCast(&condition));

        // Create then region
        const then_region = c.mlirRegionCreate();
        const then_block = c.mlirBlockCreate(0, null, null);
        c.mlirRegionInsertOwnedBlock(then_region, 0, then_block);
        c.mlirOperationStateAddOwnedRegions(&state, 1, @ptrCast(&then_region));

        // Lower then branch
        self.lowerBlockBody(if_stmt.then_branch, then_block);

        // Create else region if present
        if (if_stmt.else_branch) |else_branch| {
            const else_region = c.mlirRegionCreate();
            const else_block = c.mlirBlockCreate(0, null, null);
            c.mlirRegionInsertOwnedBlock(else_region, 0, else_block);
            c.mlirOperationStateAddOwnedRegions(&state, 1, @ptrCast(&else_region));

            // Lower else branch
            self.lowerBlockBody(else_branch, else_block);
        }

        const op = c.mlirOperationCreate(&state);
        c.mlirBlockAppendOwnedOperation(self.block, op);
    }

    /// Lower while statements
    pub fn lowerWhile(self: *const StatementLowerer, while_stmt: *const lib.ast.Statements.WhileNode) void {
        // TODO: Implement while statement lowering
        _ = self;
        _ = while_stmt;
    }

    /// Lower for loop statements
    pub fn lowerFor(self: *const StatementLowerer, for_stmt: *const lib.ast.Statements.ForLoopNode) void {
        // TODO: Implement for loop statement lowering
        _ = self;
        _ = for_stmt;
    }

    /// Lower switch statements
    pub fn lowerSwitch(self: *const StatementLowerer, switch_stmt: *const lib.ast.Statements.SwitchNode) void {
        _ = self.expr_lowerer.lowerExpression(&switch_stmt.condition);
        if (switch_stmt.default_case) |default_case| {
            self.lowerBlockBody(default_case, self.block);
        }
    }

    /// Lower expression statements
    pub fn lowerExpressionStatement(self: *const StatementLowerer, expr: *const lib.ast.Statements.ExprNode) void {
        switch (expr.*) {
            .Assignment => |assign| {
                // Handle assignment statements - these are expression-level assignments
                // Lower the value expression first
                const value = self.expr_lowerer.lowerExpression(assign.value);

                // Check if the target is an identifier that should be stored to storage
                if (assign.target.* == .Identifier) {
                    const ident = assign.target.Identifier;

                    // Check if this is a storage variable
                    if (self.storage_map) |sm| {
                        if (sm.hasStorageVariable(ident.name)) {
                            // This is a storage variable - create ora.sstore operation
                            const memory_manager = @import("memory.zig").MemoryManager.init(self.ctx);
                            const store_op = memory_manager.createStorageStore(value, ident.name, self.fileLoc(ident.span));
                            c.mlirBlockAppendOwnedOperation(self.block, store_op);
                            return;
                        }
                    }

                    // Check if this is a local variable
                    if (self.local_var_map) |lvm| {
                        if (lvm.hasLocalVar(ident.name)) {
                            // This is a local variable - store to the local variable
                            // For now, just update the map (in a real implementation, we'd create a store operation)
                            _ = lvm.addLocalVar(ident.name, value) catch {};
                            return;
                        }
                    }

                    // If we can't find the variable, this is an error
                    std.debug.print("ERROR: Variable not found for assignment: {s}\n", .{ident.name});
                }
                // TODO: Handle non-identifier targets
            },
            .CompoundAssignment => |compound| {
                // Handle compound assignment statements
                self.lowerCompoundAssignmentExpr(&compound);
            },
            else => {
                // Lower other expression statements
                _ = self.expr_lowerer.lowerExpression(expr);
            },
        }
    }

    /// Lower labeled block statements
    pub fn lowerLabeledBlock(self: *const StatementLowerer, labeled_block: *const lib.ast.Statements.LabeledBlockNode) void {
        // For now, just lower the block body
        self.lowerBlockBody(labeled_block.block, self.block);
        // TODO: Add proper labeled block handling
    }

    /// Lower block body
    pub fn lowerBlockBody(self: *const StatementLowerer, b: lib.ast.Statements.BlockNode, block: c.MlirBlock) void {
        std.debug.print("DEBUG: Processing block with {d} statements\n", .{b.statements.len});
        for (b.statements) |*s| {
            std.debug.print("DEBUG: Processing statement type: {s}\n", .{@tagName(s.*)});
            // Create a new statement lowerer for this block
            var stmt_lowerer = StatementLowerer.init(self.ctx, block, self.type_mapper, self.expr_lowerer, self.param_map, self.storage_map, self.local_var_map, self.locations);
            stmt_lowerer.lowerStatement(s);
        }
    }

    /// Create file location for operations
    fn fileLoc(self: *const StatementLowerer, span: lib.ast.SourceSpan) c.MlirLocation {
        return @import("locations.zig").LocationTracker.createFileLocationFromSpan(&self.locations, span);
    }
};
