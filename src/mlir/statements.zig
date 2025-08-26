const std = @import("std");
const c = @import("c.zig").c;
const lib = @import("ora_lib");

/// Statement lowering system for converting Ora statements to MLIR operations
pub const StatementLowerer = struct {
    ctx: c.MlirContext,
    block: c.MlirBlock,
    type_mapper: *const @import("types.zig").TypeMapper,
    expr_lowerer: *const @import("expressions.zig").ExpressionLowerer,

    pub fn init(ctx: c.MlirContext, block: c.MlirBlock, type_mapper: *const @import("types.zig").TypeMapper, expr_lowerer: *const @import("expressions.zig").ExpressionLowerer) StatementLowerer {
        return .{
            .ctx = ctx,
            .block = block,
            .type_mapper = type_mapper,
            .expr_lowerer = expr_lowerer,
        };
    }

    /// Main dispatch function for lowering statements
    pub fn lowerStatement(self: *const StatementLowerer, stmt: *const lib.ast.Statements.StmtNode) void {
        // Use the existing statement lowering logic from lower.zig
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
            else => {
                // TODO: Handle other statement types
                // For now, just skip other statement types
            },
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
        // TODO: Implement variable declaration lowering with proper memory region handling
        // For now, just skip the variable declaration
        _ = self;
        _ = var_decl;
    }

    /// Lower destructuring assignment statements
    pub fn lowerDestructuringAssignment(self: *const StatementLowerer, assignment: *const lib.ast.Statements.DestructuringAssignmentNode) void {
        // TODO: Implement destructuring assignment lowering
        // For now, just skip the assignment
        _ = self;
        _ = assignment;
    }

    /// Lower compound assignment statements
    pub fn lowerCompoundAssignment(self: *const StatementLowerer, assignment: *const lib.ast.Statements.CompoundAssignmentNode) void {
        // TODO: Implement compound assignment lowering
        // For now, just skip the assignment
        _ = self;
        _ = assignment;
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

    /// Lower while loops
    pub fn lowerWhile(self: *const StatementLowerer, while_stmt: *const lib.ast.Statements.WhileNode) void {
        // TODO: Implement while loop lowering using scf.while
        // For now, just skip the while loop
        _ = self;
        _ = while_stmt;
    }

    /// Lower for loops
    pub fn lowerFor(self: *const StatementLowerer, for_stmt: *const lib.ast.Statements.ForLoopNode) void {
        // TODO: Implement for loop lowering using scf.for
        // For now, just skip the for loop
        _ = self;
        _ = for_stmt;
    }

    /// Lower return statements with values
    pub fn lowerReturnWithValue(self: *const StatementLowerer, ret: *const lib.ast.Statements.ReturnNode) void {
        // TODO: Implement return statement lowering using func.return
        // For now, just skip the return statement
        _ = self;
        _ = ret;
    }

    /// Create scf.if operation
    pub fn createScfIf(self: *const StatementLowerer, condition: c.MlirValue, then_block: c.MlirBlock, else_block: ?c.MlirBlock, loc: c.MlirLocation) c.MlirOperation {
        _ = self; // Context not used in this simplified implementation
        var state = c.mlirOperationStateGet(c.mlirStringRefCreateFromCString("scf.if"), loc);

        // Add the condition operand
        c.mlirOperationStateAddOperands(&state, 1, @ptrCast(&condition));

        // Add the then region
        const then_region = c.mlirRegionCreate();
        c.mlirRegionInsertOwnedBlock(then_region, 0, then_block);
        c.mlirOperationStateAddOwnedRegions(&state, 1, @ptrCast(&then_region));

        // Add the else region if provided
        if (else_block) |else_blk| {
            const else_region = c.mlirRegionCreate();
            c.mlirRegionInsertOwnedBlock(else_region, 0, else_blk);
            c.mlirOperationStateAddOwnedRegions(&state, 1, @ptrCast(&else_region));
        }

        return c.mlirOperationCreate(&state);
    }

    /// Lower block body by processing all statements
    pub fn lowerBlockBody(self: *const StatementLowerer, b: lib.ast.Statements.BlockNode, block: c.MlirBlock) void {
        std.debug.print("DEBUG: Processing block with {d} statements\n", .{b.statements.len});
        for (b.statements) |*s| {
            std.debug.print("DEBUG: Processing statement type: {s}\n", .{@tagName(s.*)});
            // Create a new statement lowerer for this block
            var stmt_lowerer = StatementLowerer.init(self.ctx, block, self.type_mapper, self.expr_lowerer);
            stmt_lowerer.lowerStatement(s);
        }
    }

    /// Helper function to create file location
    fn fileLoc(self: *const StatementLowerer, span: anytype) c.MlirLocation {
        const fname = c.mlirStringRefCreateFromCString("input.ora");
        return c.mlirLocationFileLineColGet(self.ctx, fname, span.line, span.column);
    }
};
