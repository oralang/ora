// ============================================================================
// AST Visitor
// ============================================================================
//
// Generic visitor pattern for traversing AST nodes with flexible strategies.
//
// FEATURES:
//   • Comptime-parameterized visitor with custom context
//   • Optional visit callbacks for selective traversal
//   • Depth-first and breadth-first strategies
//   • Specialized visitors (symbol/type collecting)
//   • Type-erased AnyVisitor wrapper
//
// USAGE:
//   const MyCtx = struct { count: usize };
//   var ctx = MyCtx{ .count = 0 };
//   var visitor = Visitor(MyCtx, void){ .context = &ctx, .visitFunc = myFn };
//   visitor.visit(&node);
//
// ============================================================================

const std = @import("std");
const ast = @import("../ast.zig");
const ManagedArrayList = std.array_list.Managed;

/// Generic visitor interface with flexible traversal strategies
pub fn Visitor(comptime Context: type, comptime ReturnType: type) type {
    return struct {
        const Self = @This();

        context: *Context,

        // node visit functions - all optional to allow selective visiting
        visitModule: ?*const fn (*Self, *ast.ModuleNode) ReturnType = null,
        visitContract: ?*const fn (*Self, *ast.ContractNode) ReturnType = null,
        visitFunction: ?*const fn (*Self, *ast.FunctionNode) ReturnType = null,
        visitConstant: ?*const fn (*Self, *ast.ConstantNode) ReturnType = null,
        visitVariableDecl: ?*const fn (*Self, *ast.Statements.VariableDeclNode) ReturnType = null,
        visitStructDecl: ?*const fn (*Self, *ast.StructDeclNode) ReturnType = null,
        visitBitfieldDecl: ?*const fn (*Self, *ast.BitfieldDeclNode) ReturnType = null,
        visitEnumDecl: ?*const fn (*Self, *ast.EnumDeclNode) ReturnType = null,
        visitLogDecl: ?*const fn (*Self, *ast.LogDeclNode) ReturnType = null,
        visitImport: ?*const fn (*Self, *ast.ImportNode) ReturnType = null,
        visitErrorDecl: ?*const fn (*Self, *ast.Statements.ErrorDeclNode) ReturnType = null,
        visitBlock: ?*const fn (*Self, *ast.Statements.BlockNode) ReturnType = null,
        visitExpressionNode: ?*const fn (*Self, *ast.Expressions.ExprNode) ReturnType = null,
        visitStatementNode: ?*const fn (*Self, *ast.Statements.StmtNode) ReturnType = null,
        visitTryBlock: ?*const fn (*Self, *ast.Statements.TryBlockNode) ReturnType = null,

        // expression-specific visit functions
        visitIdentifier: ?*const fn (*Self, *ast.Expressions.IdentifierExpr) ReturnType = null,
        visitLiteral: ?*const fn (*Self, *ast.Expressions.LiteralExpr) ReturnType = null,
        visitBinary: ?*const fn (*Self, *ast.Expressions.BinaryExpr) ReturnType = null,
        visitUnary: ?*const fn (*Self, *ast.Expressions.UnaryExpr) ReturnType = null,
        visitAssignment: ?*const fn (*Self, *ast.Expressions.AssignmentExpr) ReturnType = null,
        visitCompoundAssignment: ?*const fn (*Self, *ast.Expressions.CompoundAssignmentExpr) ReturnType = null,
        visitCall: ?*const fn (*Self, *ast.Expressions.CallExpr) ReturnType = null,
        visitIndex: ?*const fn (*Self, *ast.Expressions.IndexExpr) ReturnType = null,
        visitFieldAccess: ?*const fn (*Self, *ast.Expressions.FieldAccessExpr) ReturnType = null,
        visitCast: ?*const fn (*Self, *ast.Expressions.CastExpr) ReturnType = null,
        visitComptime: ?*const fn (*Self, *ast.Expressions.ComptimeExpr) ReturnType = null,
        visitOld: ?*const fn (*Self, *ast.Expressions.OldExpr) ReturnType = null,
        visitTuple: ?*const fn (*Self, *ast.Expressions.TupleExpr) ReturnType = null,
        visitTry: ?*const fn (*Self, *ast.Expressions.TryExpr) ReturnType = null,
        visitErrorReturn: ?*const fn (*Self, *ast.Expressions.ErrorReturnExpr) ReturnType = null,
        visitErrorCast: ?*const fn (*Self, *ast.Expressions.ErrorCastExpr) ReturnType = null,
        visitShift: ?*const fn (*Self, *ast.Expressions.ShiftExpr) ReturnType = null,
        visitStructInstantiation: ?*const fn (*Self, *ast.Expressions.StructInstantiationExpr) ReturnType = null,
        visitEnumLiteral: ?*const fn (*Self, *ast.Expressions.EnumLiteralExpr) ReturnType = null,
        visitArrayLiteral: ?*const fn (*Self, *ast.Literals.Array) ReturnType = null,
        visitSwitchExpression: ?*const fn (*Self, *ast.Switch.ExprNode) ReturnType = null,
        visitRangeExpr: ?*const fn (*Self, *ast.Expressions.RangeExpr) ReturnType = null,
        visitQuantified: ?*const fn (*Self, *ast.Expressions.QuantifiedExpr) ReturnType = null,
        visitAnonymousStruct: ?*const fn (*Self, *ast.Expressions.AnonymousStructExpr) ReturnType = null,
        visitDestructuring: ?*const fn (*Self, *ast.Expressions.DestructuringExpr) ReturnType = null,
        visitLabeledBlockExpr: ?*const fn (*Self, *ast.Expressions.LabeledBlockExpr) ReturnType = null,

        // statement-specific visit functions
        visitReturn: ?*const fn (*Self, *ast.Statements.ReturnNode) ReturnType = null,
        visitIf: ?*const fn (*Self, *ast.Statements.IfNode) ReturnType = null,
        visitWhile: ?*const fn (*Self, *ast.Statements.WhileNode) ReturnType = null,
        visitLog: ?*const fn (*Self, *ast.Statements.LogNode) ReturnType = null,
        visitLock: ?*const fn (*Self, *ast.Statements.LockNode) ReturnType = null,
        visitInvariant: ?*const fn (*Self, *ast.Statements.InvariantNode) ReturnType = null,
        visitRequires: ?*const fn (*Self, *ast.Statements.RequiresNode) ReturnType = null,
        visitEnsures: ?*const fn (*Self, *ast.Statements.EnsuresNode) ReturnType = null,
        visitForLoop: ?*const fn (*Self, *ast.Statements.ForLoopNode) ReturnType = null,
        visitBreak: ?*const fn (*Self, *ast.Statements.BreakNode) ReturnType = null,
        visitContinue: ?*const fn (*Self, *ast.Statements.ContinueNode) ReturnType = null,
        visitUnlock: ?*const fn (*Self, *ast.Statements.UnlockNode) ReturnType = null,
        visitDestructuringAssignment: ?*const fn (*Self, *ast.Statements.DestructuringAssignmentNode) ReturnType = null,
        visitLabeledBlockStmt: ?*const fn (*Self, *ast.Statements.LabeledBlockNode) ReturnType = null,

        // traversal control hooks
        shouldVisitChildren: ?*const fn (*Self, *ast.AstNode) bool = null,
        preVisit: ?*const fn (*Self, *ast.AstNode) void = null,
        postVisit: ?*const fn (*Self, *ast.AstNode) void = null,

        /// Main visit function that dispatches to appropriate handlers
        pub fn visit(self: *Self, node: *ast.AstNode) ReturnType {
            // pre-visit hook
            if (self.preVisit) |preVisitFn| {
                preVisitFn(self, node);
            }

            const result = self.visitNode(node);

            // post-visit hook
            if (self.postVisit) |postVisitFn| {
                postVisitFn(self, node);
            }

            return result;
        }

        /// Internal node dispatch function
        fn visitNode(self: *Self, node: *ast.AstNode) ReturnType {
            switch (node.*) {
                .Module => |*module| {
                    if (self.visitModule) |visitFn| {
                        return visitFn(self, module);
                    }
                },
                .Contract => |*contract| {
                    if (self.visitContract) |visitFn| {
                        return visitFn(self, contract);
                    }
                },
                .Function => |*function| {
                    if (self.visitFunction) |visitFn| {
                        return visitFn(self, function);
                    }
                },
                .Constant => |*constant| {
                    if (self.visitConstant) |visitFn| {
                        return visitFn(self, constant);
                    }
                },
                .VariableDecl => |*var_decl| {
                    if (self.visitVariableDecl) |visitFn| {
                        return visitFn(self, var_decl);
                    }
                },
                .StructDecl => |*struct_decl| {
                    if (self.visitStructDecl) |visitFn| {
                        return visitFn(self, struct_decl);
                    }
                },
                .BitfieldDecl => |*bitfield_decl| {
                    if (self.visitBitfieldDecl) |visitFn| {
                        return visitFn(self, bitfield_decl);
                    }
                },
                .EnumDecl => |*enum_decl| {
                    if (self.visitEnumDecl) |visitFn| {
                        return visitFn(self, enum_decl);
                    }
                },
                .LogDecl => |*log_decl| {
                    if (self.visitLogDecl) |visitFn| {
                        return visitFn(self, log_decl);
                    }
                },
                .Import => |*import| {
                    if (self.visitImport) |visitFn| {
                        return visitFn(self, import);
                    }
                },
                .ErrorDecl => |*error_decl| {
                    if (self.visitErrorDecl) |visitFn| {
                        return visitFn(self, error_decl);
                    }
                },
                .Block => |*block| {
                    if (self.visitBlock) |visitFn| {
                        return visitFn(self, block);
                    }
                },
                .Expression => |expr| {
                    return self.dispatchExpression(expr);
                },
                .Statement => |stmt| {
                    return self.dispatchStatement(stmt);
                },
                .TryBlock => |*try_block| {
                    if (self.visitTryBlock) |visitFn| {
                        return visitFn(self, try_block);
                    }
                },
            }

            // default return value if no handler is set
            return if (ReturnType == void) {} else @as(ReturnType, undefined);
        }

        /// Visit expression nodes with detailed dispatch
        fn dispatchExpression(self: *Self, expr: *ast.Expressions.ExprNode) ReturnType {
            // call general expression handler first if available
            if (self.visitExpressionNode) |visitFn| {
                return visitFn(self, expr);
            }

            // then dispatch to specific expression handlers
            switch (expr.*) {
                .Identifier => |*identifier| {
                    if (self.visitIdentifier) |visitFn| {
                        return visitFn(self, identifier);
                    }
                },
                .Literal => |*literal| {
                    if (self.visitLiteral) |visitFn| {
                        return visitFn(self, literal);
                    }
                },
                .Binary => |*binary| {
                    if (self.visitBinary) |visitFn| {
                        return visitFn(self, binary);
                    }
                },
                .Unary => |*unary| {
                    if (self.visitUnary) |visitFn| {
                        return visitFn(self, unary);
                    }
                },
                .Assignment => |*assignment| {
                    if (self.visitAssignment) |visitFn| {
                        return visitFn(self, assignment);
                    }
                },
                .CompoundAssignment => |*comp_assignment| {
                    if (self.visitCompoundAssignment) |visitFn| {
                        return visitFn(self, comp_assignment);
                    }
                },
                .Call => |*call| {
                    if (self.visitCall) |visitFn| {
                        return visitFn(self, call);
                    }
                },
                .Index => |*index| {
                    if (self.visitIndex) |visitFn| {
                        return visitFn(self, index);
                    }
                },
                .FieldAccess => |*field_access| {
                    if (self.visitFieldAccess) |visitFn| {
                        return visitFn(self, field_access);
                    }
                },
                .Cast => |*cast| {
                    if (self.visitCast) |visitFn| {
                        return visitFn(self, cast);
                    }
                },
                .Comptime => |*comptime_expr| {
                    if (self.visitComptime) |visitFn| {
                        return visitFn(self, comptime_expr);
                    }
                },
                .Old => |*old| {
                    if (self.visitOld) |visitFn| {
                        return visitFn(self, old);
                    }
                },
                .Tuple => |*tuple| {
                    if (self.visitTuple) |visitFn| {
                        return visitFn(self, tuple);
                    }
                },
                .Try => |*try_expr| {
                    if (self.visitTry) |visitFn| {
                        return visitFn(self, try_expr);
                    }
                },
                .ErrorReturn => |*error_return| {
                    if (self.visitErrorReturn) |visitFn| {
                        return visitFn(self, error_return);
                    }
                },
                .ErrorCast => |*error_cast| {
                    if (self.visitErrorCast) |visitFn| {
                        return visitFn(self, error_cast);
                    }
                },
                .Shift => |*shift| {
                    if (self.visitShift) |visitFn| {
                        return visitFn(self, shift);
                    }
                },
                .StructInstantiation => |*struct_inst| {
                    if (self.visitStructInstantiation) |visitFn| {
                        return visitFn(self, struct_inst);
                    }
                },
                .EnumLiteral => |*enum_literal| {
                    if (self.visitEnumLiteral) |visitFn| {
                        return visitFn(self, enum_literal);
                    }
                },
                .ArrayLiteral => |*array_literal| {
                    if (self.visitArrayLiteral) |visitFn| {
                        return visitFn(self, array_literal);
                    }
                },
                .SwitchExpression => |*switch_expr| {
                    if (self.visitSwitchExpression) |visitFn| {
                        return visitFn(self, switch_expr);
                    }
                },
                .Range => |*range_expr| {
                    if (self.visitRangeExpr) |visitFn| {
                        return visitFn(self, range_expr);
                    }
                },
                // handle additional expression types without specific visitors
                .Quantified => |*quantified| {
                    if (self.visitQuantified) |visitFn| {
                        return visitFn(self, quantified);
                    }
                },
                .AnonymousStruct => |*anon_struct| {
                    if (self.visitAnonymousStruct) |visitFn| {
                        return visitFn(self, anon_struct);
                    }
                },
                .LabeledBlock => |*lbl_block| {
                    if (self.visitLabeledBlockExpr) |visitFn| {
                        return visitFn(self, lbl_block);
                    }
                },
                .Destructuring => |*destruct| {
                    if (self.visitDestructuring) |visitFn| {
                        return visitFn(self, destruct);
                    }
                },
            }

            return if (ReturnType == void) {} else @as(ReturnType, undefined);
        }

        /// Visit statement nodes with detailed dispatch
        fn dispatchStatement(self: *Self, stmt: *ast.Statements.StmtNode) ReturnType {
            // call general statement handler first if available
            if (self.visitStatementNode) |visitFn| {
                return visitFn(self, stmt);
            }

            // then dispatch to specific statement handlers
            switch (stmt.*) {
                .Expr => |*expr| {
                    return self.dispatchExpression(expr);
                },
                .VariableDecl => |*var_decl| {
                    if (self.visitVariableDecl) |visitFn| {
                        return visitFn(self, var_decl);
                    }
                },
                .Return => |*ret| {
                    if (self.visitReturn) |visitFn| {
                        return visitFn(self, ret);
                    }
                },
                .If => |*if_stmt| {
                    if (self.visitIf) |visitFn| {
                        return visitFn(self, if_stmt);
                    }
                },
                .While => |*while_stmt| {
                    if (self.visitWhile) |visitFn| {
                        return visitFn(self, while_stmt);
                    }
                },
                .Break => |*brk| {
                    if (self.visitBreak) |visitFn| {
                        return visitFn(self, brk);
                    }
                },
                .Continue => |*cont| {
                    if (self.visitContinue) |visitFn| {
                        return visitFn(self, cont);
                    }
                },
                .Log => |*log| {
                    if (self.visitLog) |visitFn| {
                        return visitFn(self, log);
                    }
                },
                .Lock => |*lock| {
                    if (self.visitLock) |visitFn| {
                        return visitFn(self, lock);
                    }
                },
                .Invariant => |*invariant| {
                    if (self.visitInvariant) |visitFn| {
                        return visitFn(self, invariant);
                    }
                },
                .Requires => |*requires| {
                    if (self.visitRequires) |visitFn| {
                        return visitFn(self, requires);
                    }
                },
                .Ensures => |*ensures| {
                    if (self.visitEnsures) |visitFn| {
                        return visitFn(self, ensures);
                    }
                },
                .ErrorDecl => |*error_decl| {
                    if (self.visitErrorDecl) |visitFn| {
                        return visitFn(self, error_decl);
                    }
                },
                .TryBlock => |*try_block| {
                    if (self.visitTryBlock) |visitFn| {
                        return visitFn(self, try_block);
                    }
                },
                // handle additional statement types
                .DestructuringAssignment => |*dassign| {
                    if (self.visitDestructuringAssignment) |visitFn| {
                        return visitFn(self, dassign);
                    }
                },
                .ForLoop => |*for_loop| {
                    if (self.visitForLoop) |visitFn| {
                        return visitFn(self, for_loop);
                    }
                },
                .Unlock => |*unlock| {
                    if (self.visitUnlock) |visitFn| {
                        return visitFn(self, unlock);
                    }
                },
                .Switch => |*switch_stmt| {
                    // no dedicated handler; fall through for now
                    _ = switch_stmt;
                },
                .LabeledBlock => |*lbl_stmt| {
                    if (self.visitLabeledBlockStmt) |visitFn| {
                        return visitFn(self, lbl_stmt);
                    }
                },
                .CompoundAssignment => {
                    // no specific handler for compound assignment statements yet
                },
            }

            return if (ReturnType == void) {} else @as(ReturnType, undefined);
        }

        /// Pre-order traversal (visit node before children)
        pub fn walkPreOrder(self: *Self, node: *ast.AstNode) ReturnType {
            // visit current node first
            const result = self.visit(node);

            // then visit children if allowed
            if (self.shouldVisitChildren == null or self.shouldVisitChildren.?(self, node)) {
                self.visitChildren(node);
            }

            return result;
        }

        /// Post-order traversal (visit children before node)
        pub fn walkPostOrder(self: *Self, node: *ast.AstNode) ReturnType {
            // visit children first if allowed
            if (self.shouldVisitChildren == null or self.shouldVisitChildren.?(self, node)) {
                self.visitChildren(node);
            }

            // then visit current node
            return self.visit(node);
        }

        /// Breadth-first traversal
        pub fn walkBreadthFirst(self: *Self, node: *ast.AstNode, allocator: std.mem.Allocator) !ReturnType {
            var queue = ManagedArrayList(*ast.AstNode).init(allocator);
            defer queue.deinit();

            try queue.append(node);
            var result: ReturnType = if (ReturnType == void) {} else @as(ReturnType, undefined);

            while (queue.items.len > 0) {
                const current = queue.orderedRemove(0);
                result = self.visit(current);

                // add children to queue if allowed
                if (self.shouldVisitChildren == null or self.shouldVisitChildren.?(self, current)) {
                    try self.addChildrenToQueue(current, &queue);
                }
            }

            return result;
        }

        /// Visit all children of a node
        fn visitChildren(self: *Self, node: *ast.AstNode) void {
            switch (node.*) {
                .Module => |*module| {
                    for (module.declarations) |*decl| {
                        _ = self.walkPreOrder(decl);
                    }
                },
                .Contract => |*contract| {
                    for (contract.body) |*child| {
                        _ = self.walkPreOrder(child);
                    }
                },
                .Function => |*function| {
                    for (function.requires_clauses) |*clause| {
                        var expr_node = ast.AstNode{ .Expression = clause.* };
                        _ = self.walkPreOrder(&expr_node);
                    }
                    for (function.ensures_clauses) |*clause| {
                        var expr_node = ast.AstNode{ .Expression = clause.* };
                        _ = self.walkPreOrder(&expr_node);
                    }
                    var block_node = ast.AstNode{ .Block = function.body };
                    _ = self.walkPreOrder(&block_node);
                },
                .Constant => |*constant| {
                    // visit the constant's value expression
                    var expr_node = ast.AstNode{ .Expression = constant.value };
                    _ = self.walkPreOrder(&expr_node);
                },
                .VariableDecl => |*var_decl| {
                    // visit the variable's initializer if present
                    if (var_decl.value) |value| {
                        var expr_node = ast.AstNode{ .Expression = value };
                        _ = self.walkPreOrder(&expr_node);
                    }
                },
                .Block => |*block| {
                    for (block.statements) |*stmt| {
                        var stmt_node = ast.AstNode{ .Statement = stmt };
                        _ = self.walkPreOrder(&stmt_node);
                    }
                },
                .Expression => |expr| {
                    self.visitExpressionChildren(expr);
                },
                .Statement => |stmt| {
                    self.visitStatementChildren(stmt);
                },
                .TryBlock => |*try_block| {
                    var try_block_node = ast.AstNode{ .Block = try_block.try_block };
                    _ = self.walkPreOrder(&try_block_node);
                    if (try_block.catch_block) |*catch_block| {
                        var catch_block_node = ast.AstNode{ .Block = catch_block.block };
                        _ = self.walkPreOrder(&catch_block_node);
                    }
                },
                else => {
                    // other node types don't have children or are handled elsewhere
                },
            }
        }

        /// Visit children of expression nodes
        fn visitExpressionChildren(self: *Self, expr: *ast.Expressions.ExprNode) void {
            switch (expr.*) {
                .Binary => |*binary| {
                    _ = self.dispatchExpression(binary.lhs);
                    _ = self.dispatchExpression(binary.rhs);
                },
                .Unary => |*unary| {
                    _ = self.dispatchExpression(unary.operand);
                },
                .Assignment => |*assignment| {
                    _ = self.dispatchExpression(assignment.target);
                    _ = self.dispatchExpression(assignment.value);
                },
                .CompoundAssignment => |*comp_assignment| {
                    _ = self.dispatchExpression(comp_assignment.target);
                    _ = self.dispatchExpression(comp_assignment.value);
                },
                .Call => |*call| {
                    _ = self.dispatchExpression(call.callee);
                    for (call.arguments) |arg| {
                        _ = self.dispatchExpression(arg);
                    }
                },
                .Index => |*index| {
                    _ = self.dispatchExpression(index.target);
                    _ = self.dispatchExpression(index.index);
                },
                .FieldAccess => |*field_access| {
                    _ = self.dispatchExpression(field_access.target);
                },
                .Cast => |*cast| {
                    _ = self.dispatchExpression(cast.operand);
                },
                .Comptime => |*comptime_expr| {
                    var block_node = ast.AstNode{ .Block = comptime_expr.block };
                    _ = self.walkPreOrder(&block_node);
                },
                .Old => |*old| {
                    _ = self.dispatchExpression(old.expr);
                },
                .Tuple => |*tuple| {
                    for (tuple.elements) |element| {
                        _ = self.dispatchExpression(element);
                    }
                },
                .Try => |*try_expr| {
                    _ = self.dispatchExpression(try_expr.expr);
                },
                .ErrorCast => |*error_cast| {
                    _ = self.dispatchExpression(error_cast.operand);
                },
                .Shift => |*shift| {
                    _ = self.dispatchExpression(shift.mapping);
                    _ = self.dispatchExpression(shift.source);
                    _ = self.dispatchExpression(shift.dest);
                    _ = self.dispatchExpression(shift.amount);
                },
                .StructInstantiation => |*struct_inst| {
                    _ = self.dispatchExpression(struct_inst.struct_name);
                    for (struct_inst.fields) |field| {
                        _ = self.dispatchExpression(field.value);
                    }
                },
                .ArrayLiteral => |*array_literal| {
                    for (array_literal.elements) |element| {
                        _ = self.dispatchExpression(element);
                    }
                },
                .SwitchExpression => |*switch_expr| {
                    _ = self.dispatchExpression(switch_expr.condition);
                    // skip complex switch case handling for now
                },
                .Range => |*range_expr| {
                    _ = self.dispatchExpression(range_expr.start);
                    _ = self.dispatchExpression(range_expr.end);
                },
                .Quantified => |*quantified| {
                    if (quantified.condition) |condition| {
                        _ = self.dispatchExpression(condition);
                    }
                    _ = self.dispatchExpression(quantified.body);
                },
                .AnonymousStruct => |*anon_struct| {
                    for (anon_struct.fields) |field| {
                        _ = self.dispatchExpression(field.value);
                    }
                },
                .LabeledBlock => |*lbl_block| {
                    var block_node = ast.AstNode{ .Block = lbl_block.block };
                    _ = self.walkPreOrder(&block_node);
                },
                .Destructuring => |*destruct| {
                    _ = self.dispatchExpression(destruct.value);
                },
                else => {
                    // leaf nodes (literals, identifiers, etc.) have no children
                },
            }
        }

        /// Visit children of statement nodes
        fn visitStatementChildren(self: *Self, stmt: *ast.Statements.StmtNode) void {
            switch (stmt.*) {
                .Expr => |*expr| {
                    _ = self.dispatchExpression(expr);
                },
                .VariableDecl => |_| {
                    // skip value handling for now
                },
                .Return => |_| {
                    // skip value handling for now
                },
                .If => |*if_stmt| {
                    _ = self.dispatchExpression(&if_stmt.condition);
                    var then_node = ast.AstNode{ .Block = if_stmt.then_branch };
                    _ = self.walkPreOrder(&then_node);
                    if (if_stmt.else_branch) |*else_branch| {
                        var else_node = ast.AstNode{ .Block = else_branch.* };
                        _ = self.walkPreOrder(&else_node);
                    }
                },
                .While => |*while_stmt| {
                    _ = self.dispatchExpression(&while_stmt.condition);
                    for (while_stmt.invariants) |*invariant| {
                        _ = self.dispatchExpression(invariant);
                    }
                    var body_node = ast.AstNode{ .Block = while_stmt.body };
                    _ = self.walkPreOrder(&body_node);
                },
                .Log => |*log| {
                    for (log.args) |*arg| {
                        _ = self.dispatchExpression(arg);
                    }
                },
                .Lock => |*lock| {
                    _ = self.dispatchExpression(&lock.path);
                },
                .Invariant => |*invariant| {
                    _ = self.dispatchExpression(&invariant.condition);
                },
                .Requires => |*requires| {
                    _ = self.dispatchExpression(&requires.condition);
                },
                .Ensures => |*ensures| {
                    _ = self.dispatchExpression(&ensures.condition);
                },
                .TryBlock => |*try_block| {
                    var try_node = ast.AstNode{ .Block = try_block.try_block };
                    _ = self.walkPreOrder(&try_node);
                    if (try_block.catch_block) |*catch_block| {
                        var catch_node = ast.AstNode{ .Block = catch_block.block };
                        _ = self.walkPreOrder(&catch_node);
                    }
                },
                .ForLoop => |*for_loop| {
                    _ = self.dispatchExpression(&for_loop.iterable);
                    var body_node = ast.AstNode{ .Block = for_loop.body };
                    _ = self.walkPreOrder(&body_node);
                },
                .Break => |*brk| {
                    if (brk.value) |value| {
                        _ = self.dispatchExpression(value);
                    }
                },
                .Continue => |_| {},
                .DestructuringAssignment => |*dassign| {
                    _ = self.dispatchExpression(dassign.value);
                },
                .LabeledBlock => |*lbl_stmt| {
                    var block_node = ast.AstNode{ .Block = lbl_stmt.block };
                    _ = self.walkPreOrder(&block_node);
                },
                else => {
                    // other statements have no children or are handled elsewhere
                },
            }
        }

        /// Add children to queue for breadth-first traversal
        fn addChildrenToQueue(self: *Self, node: *ast.AstNode, queue: *ManagedArrayList(*ast.AstNode)) !void {
            _ = self; // Suppress unused parameter warning

            switch (node.*) {
                .Contract => |*contract| {
                    for (contract.body) |*child| {
                        try queue.append(child);
                    }
                },
                .Function => |*function| {
                    // add requires/ensures clauses as expression nodes
                    for (function.requires_clauses) |*clause| {
                        const expr_node = try queue.allocator.create(ast.AstNode);
                        expr_node.* = ast.AstNode{ .Expression = clause.* };
                        try queue.append(expr_node);
                    }
                    for (function.ensures_clauses) |*clause| {
                        const expr_node = try queue.allocator.create(ast.AstNode);
                        expr_node.* = ast.AstNode{ .Expression = clause.* };
                        try queue.append(expr_node);
                    }
                    const block_node = try queue.allocator.create(ast.AstNode);
                    block_node.* = ast.AstNode{ .Block = function.body };
                    try queue.append(block_node);
                },
                .Block => |*block| {
                    for (block.statements) |*stmt| {
                        const stmt_node = try queue.allocator.create(ast.AstNode);
                        stmt_node.* = ast.AstNode{ .Statement = stmt };
                        try queue.append(stmt_node);
                    }
                },
                .Expression => |expr| {
                    switch (expr.*) {
                        .LabeledBlock => |*lbl_block| {
                            const block_node = try queue.allocator.create(ast.AstNode);
                            block_node.* = ast.AstNode{ .Block = lbl_block.block };
                            try queue.append(block_node);
                        },
                        else => {},
                    }
                },
                else => {
                    // other node types handled in expression/statement specific methods
                },
            }
        }
    };
}

/// Convenience type aliases for common visitor patterns
pub const VoidVisitor = Visitor(void, void);
pub const BoolVisitor = Visitor(void, bool);
pub const ErrorVisitor = Visitor(void, anyerror!void);

/// Traversal strategy enumeration
pub const TraversalStrategy = enum {
    PreOrder,
    PostOrder,
    BreadthFirst,
};

pub const TraversalError = error{
    MissingAllocator,
};

/// Generic traversal function that accepts any visitor and strategy
pub fn traverse(
    comptime VisitorType: type,
    visitor: *VisitorType,
    node: *ast.AstNode,
    strategy: TraversalStrategy,
    allocator: ?std.mem.Allocator,
) anyerror!void {
    switch (strategy) {
        .PreOrder => _ = visitor.walkPreOrder(node),
        .PostOrder => _ = visitor.walkPostOrder(node),
        .BreadthFirst => {
            if (allocator) |alloc| {
                _ = try visitor.walkBreadthFirst(node, alloc);
            } else {
                return TraversalError.MissingAllocator;
            }
        },
    }
}

test "ast_visitor: breadth-first traversal requires allocator" {
    const Ctx = struct {
        count: usize = 0,
    };
    var ctx = Ctx{};
    var visitor = Visitor(Ctx, void){ .context = &ctx };

    var node = ast.AstNode{
        .Contract = ast.ContractNode{
            .name = "Test",
            .body = &[_]ast.AstNode{},
            .span = ast.SourceSpan{ .line = 1, .column = 1, .length = 1 },
        },
    };

    try std.testing.expectError(TraversalError.MissingAllocator, traverse(@TypeOf(visitor), &visitor, &node, .BreadthFirst, null));
}

/// Specialized visitor types for common use cases
/// Mutable visitor for AST modification operations
pub const MutableVisitor = struct {
    const Self = @This();

    allocator: std.mem.Allocator,
    modification_count: u32 = 0,
    error_count: u32 = 0,

    // modification functions - all optional, covering all AST node types
    modifyContract: ?*const fn (*Self, *ast.ContractNode) anyerror!void = null,
    modifyFunction: ?*const fn (*Self, *ast.FunctionNode) anyerror!void = null,
    modifyVariableDecl: ?*const fn (*Self, *ast.Statements.VariableDeclNode) anyerror!void = null,
    modifyStructDecl: ?*const fn (*Self, *ast.StructDeclNode) anyerror!void = null,
    modifyBitfieldDecl: ?*const fn (*Self, *ast.BitfieldDeclNode) anyerror!void = null,
    modifyEnumDecl: ?*const fn (*Self, *ast.EnumDeclNode) anyerror!void = null,
    modifyLogDecl: ?*const fn (*Self, *ast.LogDeclNode) anyerror!void = null,
    modifyImport: ?*const fn (*Self, *ast.ImportNode) anyerror!void = null,
    modifyErrorDecl: ?*const fn (*Self, *ast.Statements.ErrorDeclNode) anyerror!void = null,
    modifyBlock: ?*const fn (*Self, *ast.Statements.BlockNode) anyerror!void = null,
    modifyExpression: ?*const fn (*Self, *ast.Expressions.ExprNode) anyerror!void = null,
    modifyStatement: ?*const fn (*Self, *ast.Statements.StmtNode) anyerror!void = null,
    modifyTryBlock: ?*const fn (*Self, *ast.Statements.TryBlockNode) anyerror!void = null,

    // replacement functions - return new node or null to keep original
    replaceExpression: ?*const fn (*Self, *ast.Expressions.ExprNode) anyerror!?*ast.Expressions.ExprNode = null,
    replaceStatement: ?*const fn (*Self, *ast.Statements.StmtNode) anyerror!?*ast.Statements.StmtNode = null,
    replaceContract: ?*const fn (*Self, *ast.ContractNode) anyerror!?*ast.ContractNode = null,
    replaceFunction: ?*const fn (*Self, *ast.FunctionNode) anyerror!?*ast.FunctionNode = null,

    // error handling
    onError: ?*const fn (*Self, []const u8, ?ast.SourceSpan) void = null,

    pub fn init(allocator: std.mem.Allocator) Self {
        return Self{
            .allocator = allocator,
        };
    }

    pub fn visitAndModify(self: *Self, node: *ast.AstNode) anyerror!void {
        switch (node.*) {
            .Contract => |*contract| {
                if (self.replaceContract) |replaceFn| {
                    if (try replaceFn(self, contract)) |new_contract| {
                        contract.* = new_contract.*;
                        self.modification_count += 1;
                        return;
                    }
                }
                if (self.modifyContract) |modifyFn| {
                    modifyFn(self, contract) catch |err| {
                        self.error_count += 1;
                        if (self.onError) |errorFn| {
                            errorFn(self, "Error modifying contract", contract.span);
                        }
                        return err;
                    };
                    self.modification_count += 1;
                }
                // visit children
                for (contract.body) |*child| {
                    try self.visitAndModify(child);
                }
            },
            .Function => |*function| {
                if (self.replaceFunction) |replaceFn| {
                    if (try replaceFn(self, function)) |new_function| {
                        function.* = new_function.*;
                        self.modification_count += 1;
                        return;
                    }
                }
                if (self.modifyFunction) |modifyFn| {
                    modifyFn(self, function) catch |err| {
                        self.error_count += 1;
                        if (self.onError) |errorFn| {
                            errorFn(self, "Error modifying function", function.span);
                        }
                        return err;
                    };
                    self.modification_count += 1;
                }
                // visit function body
                var block_node = ast.AstNode{ .Block = function.body };
                try self.visitAndModify(&block_node);
            },
            .VariableDecl => |*var_decl| {
                if (self.modifyVariableDecl) |modifyFn| {
                    modifyFn(self, var_decl) catch |err| {
                        self.error_count += 1;
                        if (self.onError) |errorFn| {
                            errorFn(self, "Error modifying variable declaration", var_decl.span);
                        }
                        return err;
                    };
                    self.modification_count += 1;
                }
                // visit initializer if present
                if (var_decl.value) |*value| {
                    var expr_node = ast.AstNode{ .Expression = value.* };
                    try self.visitAndModify(&expr_node);
                }
            },
            .StructDecl => |*struct_decl| {
                if (self.modifyStructDecl) |modifyFn| {
                    modifyFn(self, struct_decl) catch |err| {
                        self.error_count += 1;
                        if (self.onError) |errorFn| {
                            errorFn(self, "Error modifying struct declaration", struct_decl.span);
                        }
                        return err;
                    };
                    self.modification_count += 1;
                }
            },
            .BitfieldDecl => |*bitfield_decl| {
                if (self.modifyBitfieldDecl) |modifyFn| {
                    modifyFn(self, bitfield_decl) catch |err| {
                        self.error_count += 1;
                        if (self.onError) |errorFn| {
                            errorFn(self, "Error modifying bitfield declaration", bitfield_decl.span);
                        }
                        return err;
                    };
                    self.modification_count += 1;
                }
            },
            .EnumDecl => |*enum_decl| {
                if (self.modifyEnumDecl) |modifyFn| {
                    modifyFn(self, enum_decl) catch |err| {
                        self.error_count += 1;
                        if (self.onError) |errorFn| {
                            errorFn(self, "Error modifying enum declaration", enum_decl.span);
                        }
                        return err;
                    };
                    self.modification_count += 1;
                }
            },
            .LogDecl => |*log_decl| {
                if (self.modifyLogDecl) |modifyFn| {
                    modifyFn(self, log_decl) catch |err| {
                        self.error_count += 1;
                        if (self.onError) |errorFn| {
                            errorFn(self, "Error modifying log declaration", log_decl.span);
                        }
                        return err;
                    };
                    self.modification_count += 1;
                }
            },
            .Import => |*import| {
                if (self.modifyImport) |modifyFn| {
                    modifyFn(self, import) catch |err| {
                        self.error_count += 1;
                        if (self.onError) |errorFn| {
                            errorFn(self, "Error modifying import", import.span);
                        }
                        return err;
                    };
                    self.modification_count += 1;
                }
            },
            .ErrorDecl => |*error_decl| {
                if (self.modifyErrorDecl) |modifyFn| {
                    modifyFn(self, error_decl) catch |err| {
                        self.error_count += 1;
                        if (self.onError) |errorFn| {
                            errorFn(self, "Error modifying error declaration", error_decl.span);
                        }
                        return err;
                    };
                    self.modification_count += 1;
                }
            },
            .Block => |*block| {
                if (self.modifyBlock) |modifyFn| {
                    modifyFn(self, block) catch |err| {
                        self.error_count += 1;
                        if (self.onError) |errorFn| {
                            errorFn(self, "Error modifying block", block.span);
                        }
                        return err;
                    };
                    self.modification_count += 1;
                }
                for (block.statements) |*stmt| {
                    var stmt_node = ast.AstNode{ .Statement = stmt.* };
                    try self.visitAndModify(&stmt_node);
                }
            },
            .Expression => |expr| {
                try self.visitAndModifyExpression(expr);
            },
            .Statement => |stmt| {
                try self.visitAndModifyStatement(stmt);
            },
            .TryBlock => |*try_block| {
                if (self.modifyTryBlock) |modifyFn| {
                    modifyFn(self, try_block) catch |err| {
                        self.error_count += 1;
                        if (self.onError) |errorFn| {
                            errorFn(self, "Error modifying try block", try_block.span);
                        }
                        return err;
                    };
                    self.modification_count += 1;
                }
                var try_node = ast.AstNode{ .Block = try_block.try_block };
                try self.visitAndModify(&try_node);
                if (try_block.catch_block) |*catch_block| {
                    var catch_node = ast.AstNode{ .Block = catch_block.block };
                    try self.visitAndModify(&catch_node);
                }
            },
        }
    }

    pub fn getModificationCount(self: *const Self) u32 {
        return self.modification_count;
    }

    pub fn getErrorCount(self: *const Self) u32 {
        return self.error_count;
    }

    pub fn reset(self: *Self) void {
        self.modification_count = 0;
        self.error_count = 0;
    }

    fn visitAndModifyExpression(self: *Self, expr: *ast.Expressions.ExprNode) anyerror!void {
        // try replacement first
        if (self.replaceExpression) |replaceFn| {
            if (try replaceFn(self, expr)) |new_expr| {
                expr.* = new_expr.*;
                return;
            }
        }

        // apply modifications
        if (self.modifyExpression) |modifyFn| {
            try modifyFn(self, expr);
        }

        // visit children
        switch (expr.*) {
            .Binary => |*binary| {
                try self.visitAndModifyExpression(binary.lhs);
                try self.visitAndModifyExpression(binary.rhs);
            },
            .Unary => |*unary| {
                try self.visitAndModifyExpression(unary.operand);
            },
            .Call => |*call| {
                try self.visitAndModifyExpression(call.callee);
                for (call.arguments) |*arg| {
                    try self.visitAndModifyExpression(arg);
                }
            },
            .Assignment => |*assignment| {
                try self.visitAndModifyExpression(assignment.target);
                try self.visitAndModifyExpression(assignment.value);
            },
            else => {
                // leaf nodes or other expressions
            },
        }
    }

    fn visitAndModifyStatement(self: *Self, stmt: *ast.Statements.StmtNode) anyerror!void {
        // try replacement first
        if (self.replaceStatement) |replaceFn| {
            if (try replaceFn(self, stmt)) |new_stmt| {
                stmt.* = new_stmt.*;
                return;
            }
        }

        // apply modifications
        if (self.modifyStatement) |modifyFn| {
            try modifyFn(self, stmt);
        }

        // visit children
        switch (stmt.*) {
            .Expr => |*expr| {
                try self.visitAndModifyExpression(expr);
            },
            .If => |*if_stmt| {
                try self.visitAndModifyExpression(&if_stmt.condition);
                var then_node = ast.AstNode{ .Block = if_stmt.then_branch };
                try self.visitAndModify(&then_node);
                if (if_stmt.else_branch) |else_branch| {
                    var else_node = ast.AstNode{ .Block = else_branch };
                    try self.visitAndModify(&else_node);
                }
            },
            .While => |*while_stmt| {
                try self.visitAndModifyExpression(&while_stmt.condition);
                var body_node = ast.AstNode{ .Block = while_stmt.body };
                try self.visitAndModify(&body_node);
            },
            else => {
                // other statement types
            },
        }
    }
};

/// Query visitor for node searching and analysis
pub const QueryVisitor = struct {
    const Self = @This();

    allocator: std.mem.Allocator,
    results: ManagedArrayList(*ast.AstNode),
    search_count: u32 = 0,
    max_results: ?u32 = null,

    // query predicates
    predicate: ?*const fn (*ast.AstNode) bool = null,
    node_type_filter: ?std.meta.Tag(ast.AstNode) = null,

    // specific search functions
    findByName: ?[]const u8 = null,
    findByType: ?std.meta.Tag(ast.Expressions.ExprNode) = null,
    findBySpan: ?ast.SourceSpan = null,

    // advanced search options
    case_sensitive: bool = true,
    exact_match: bool = true,
    include_children: bool = true,

    // search statistics
    nodes_visited: u32 = 0,
    matches_found: u32 = 0,

    pub fn init(allocator: std.mem.Allocator) Self {
        return Self{
            .allocator = allocator,
            .results = ManagedArrayList(*ast.AstNode).init(allocator),
        };
    }

    pub fn deinit(self: *Self) void {
        self.results.deinit();
    }

    pub fn search(self: *Self, node: *ast.AstNode) ![]const *ast.AstNode {
        self.results.clearRetainingCapacity();
        self.nodes_visited = 0;
        self.matches_found = 0;
        self.search_count += 1;
        try self.searchRecursive(node);
        return self.results.items;
    }

    pub fn searchFirst(self: *Self, node: *ast.AstNode) !?*ast.AstNode {
        const old_max = self.max_results;
        self.max_results = 1;
        defer self.max_results = old_max;

        const results = try self.search(node);
        return if (results.len > 0) results[0] else null;
    }

    pub fn count(self: *Self, node: *ast.AstNode) !u32 {
        const results = try self.search(node);
        return @intCast(results.len);
    }

    pub fn exists(self: *Self, node: *ast.AstNode) !bool {
        const result = try self.searchFirst(node);
        return result != null;
    }

    pub fn getSearchStats(self: *const Self) struct { nodes_visited: u32, matches_found: u32, search_count: u32 } {
        return .{
            .nodes_visited = self.nodes_visited,
            .matches_found = self.matches_found,
            .search_count = self.search_count,
        };
    }

    fn searchRecursive(self: *Self, node: *ast.AstNode) std.mem.Allocator.Error!void {
        // check if this node matches our criteria
        if (self.matches(node)) {
            try self.results.append(node);
        }

        // search children
        switch (node.*) {
            .Contract => |*contract| {
                for (contract.body) |*child| {
                    try self.searchRecursive(child);
                }
            },
            .Function => |*function| {
                var block_node = ast.AstNode{ .Block = function.body };
                try self.searchRecursive(&block_node);
            },
            .Block => |*block| {
                for (block.statements) |*stmt| {
                    var stmt_node = ast.AstNode{ .Statement = stmt.* };
                    try self.searchRecursive(&stmt_node);
                }
            },
            .Expression => |expr| {
                try self.searchExpressionRecursive(expr);
            },
            .Statement => |stmt| {
                try self.searchStatementRecursive(stmt);
            },
            else => {},
        }
    }

    fn searchExpressionRecursive(self: *Self, expr: *ast.Expressions.ExprNode) std.mem.Allocator.Error!void {
        var expr_node = ast.AstNode{ .Expression = expr.* };
        if (self.matches(&expr_node)) {
            try self.results.append(&expr_node);
        }

        switch (expr.*) {
            .Binary => |*binary| {
                try self.searchExpressionRecursive(binary.lhs);
                try self.searchExpressionRecursive(binary.rhs);
            },
            .Unary => |*unary| {
                try self.searchExpressionRecursive(unary.operand);
            },
            .Call => |*call| {
                try self.searchExpressionRecursive(call.callee);
                for (call.arguments) |*arg| {
                    try self.searchExpressionRecursive(arg);
                }
            },
            else => {},
        }
    }

    fn searchStatementRecursive(self: *Self, stmt: *ast.Statements.StmtNode) std.mem.Allocator.Error!void {
        var stmt_node = ast.AstNode{ .Statement = stmt.* };
        if (self.matches(&stmt_node)) {
            try self.results.append(&stmt_node);
        }

        switch (stmt.*) {
            .Expr => |*expr| {
                try self.searchExpressionRecursive(expr);
            },
            .If => |*if_stmt| {
                try self.searchExpressionRecursive(&if_stmt.condition);
                var then_node = ast.AstNode{ .Block = if_stmt.then_branch };
                try self.searchRecursive(&then_node);
                if (if_stmt.else_branch) |else_branch| {
                    var else_node = ast.AstNode{ .Block = else_branch };
                    try self.searchRecursive(&else_node);
                }
            },
            else => {},
        }
    }

    fn matches(self: *Self, node: *ast.AstNode) bool {
        // check custom predicate first
        if (self.predicate) |pred| {
            if (!pred(node)) return false;
        }

        // check node type filter
        if (self.node_type_filter) |filter| {
            if (std.meta.activeTag(node.*) != filter) return false;
        }

        // check name-based search
        if (self.findByName) |name| {
            switch (node.*) {
                .Contract => |*contract| return std.mem.eql(u8, contract.name, name),
                .Function => |*function| return std.mem.eql(u8, function.name, name),
                .VariableDecl => |*var_decl| return std.mem.eql(u8, var_decl.name, name),
                .Expression => |expr| {
                    switch (expr.*) {
                        .Identifier => |*id| return std.mem.eql(u8, id.name, name),
                        else => return false,
                    }
                },
                else => return false,
            }
        }

        // check expression type filter
        if (self.findByType) |expr_type| {
            switch (node.*) {
                .Expression => |*expr| return std.meta.activeTag(expr.*) == expr_type,
                else => return false,
            }
        }

        return true;
    }
};

/// Transform visitor for AST transformation with error handling
pub const TransformVisitor = struct {
    const Self = @This();

    allocator: std.mem.Allocator,
    errors: ManagedArrayList(TransformError),

    // transformation functions
    transformContract: ?*const fn (*Self, *ast.ContractNode) anyerror!?*ast.ContractNode = null,
    transformFunction: ?*const fn (*Self, *ast.FunctionNode) anyerror!?*ast.FunctionNode = null,
    transformExpression: ?*const fn (*Self, *ast.Expressions.ExprNode) anyerror!?*ast.Expressions.ExprNode = null,
    transformStatement: ?*const fn (*Self, *ast.Statements.StmtNode) anyerror!?*ast.Statements.StmtNode = null,

    pub const TransformError = struct {
        message: []const u8,
        node_span: ?ast.SourceSpan = null,
        error_type: ErrorType = .General,

        pub const ErrorType = enum {
            General,
            TypeMismatch,
            InvalidTransformation,
            MissingDependency,
        };
    };

    pub fn init(allocator: std.mem.Allocator) Self {
        return Self{
            .allocator = allocator,
            .errors = ManagedArrayList(TransformError).init(allocator),
        };
    }

    pub fn deinit(self: *Self) void {
        self.errors.deinit();
    }

    pub fn transform(self: *Self, node: *ast.AstNode) !*ast.AstNode {
        self.errors.clearRetainingCapacity();
        return self.transformRecursive(node);
    }

    pub fn hasErrors(self: *const Self) bool {
        return self.errors.items.len > 0;
    }

    pub fn getErrors(self: *const Self) []const TransformError {
        return self.errors.items;
    }

    fn transformRecursive(self: *Self, node: *ast.AstNode) anyerror!*ast.AstNode {
        switch (node.*) {
            .Contract => |*contract| {
                if (self.transformContract) |transformFn| {
                    if (try transformFn(self, contract)) |new_contract| {
                        const new_node = try self.allocator.create(ast.AstNode);
                        new_node.* = ast.AstNode{ .Contract = new_contract.* };
                        return new_node;
                    }
                }
                // transform children
                var new_body = ManagedArrayList(ast.AstNode).init(self.allocator);
                for (contract.body) |*child| {
                    const transformed_child = try self.transformRecursive(child);
                    try new_body.append(transformed_child.*);
                }

                const new_contract = try self.allocator.create(ast.ContractNode);
                new_contract.* = ast.ContractNode{
                    .name = contract.name,
                    .body = try new_body.toOwnedSlice(),
                    .span = contract.span,
                };

                const new_node = try self.allocator.create(ast.AstNode);
                new_node.* = ast.AstNode{ .Contract = new_contract.* };
                return new_node;
            },
            .Function => |*function| {
                if (self.transformFunction) |transformFn| {
                    if (try transformFn(self, function)) |new_function| {
                        const new_node = try self.allocator.create(ast.AstNode);
                        new_node.* = ast.AstNode{ .Function = new_function.* };
                        return new_node;
                    }
                }
                // return copy of original function for now
                const new_node = try self.allocator.create(ast.AstNode);
                new_node.* = node.*;
                return new_node;
            },
            .Expression => |expr| {
                return try self.transformExpressionRecursive(expr);
            },
            .Statement => |stmt| {
                return try self.transformStatementRecursive(stmt);
            },
            else => {
                // return copy of original node
                const new_node = try self.allocator.create(ast.AstNode);
                new_node.* = node.*;
                return new_node;
            },
        }
    }

    fn transformExpressionRecursive(self: *Self, expr: *ast.Expressions.ExprNode) !*ast.AstNode {
        if (self.transformExpression) |transformFn| {
            if (try transformFn(self, expr)) |new_expr| {
                const new_node = try self.allocator.create(ast.AstNode);
                new_node.* = ast.AstNode{ .Expression = new_expr.* };
                return new_node;
            }
        }

        // transform children and create new expression
        switch (expr.*) {
            .Binary => |*binary| {
                const lhs_node = try self.transformExpressionRecursive(binary.lhs);
                const rhs_node = try self.transformExpressionRecursive(binary.rhs);

                const new_binary = try self.allocator.create(ast.Expressions.BinaryExpr);
                new_binary.* = ast.Expressions.BinaryExpr{
                    .lhs = &lhs_node.Expression,
                    .operator = binary.operator,
                    .rhs = &rhs_node.Expression,
                    .span = binary.span,
                };

                const new_expr = try self.allocator.create(ast.Expressions.ExprNode);
                new_expr.* = ast.Expressions.ExprNode{ .Binary = new_binary.* };

                const new_node = try self.allocator.create(ast.AstNode);
                new_node.* = ast.AstNode{ .Expression = new_expr.* };
                return new_node;
            },
            else => {
                // return copy of original expression
                const new_node = try self.allocator.create(ast.AstNode);
                new_node.* = ast.AstNode{ .Expression = expr.* };
                return new_node;
            },
        }
    }

    fn transformStatementRecursive(self: *Self, stmt: *ast.Statements.StmtNode) !*ast.AstNode {
        if (self.transformStatement) |transformFn| {
            if (try transformFn(self, stmt)) |new_stmt| {
                const new_node = try self.allocator.create(ast.AstNode);
                new_node.* = ast.AstNode{ .Statement = new_stmt.* };
                return new_node;
            }
        }

        // return copy of original statement for now
        const new_node = try self.allocator.create(ast.AstNode);
        new_node.* = ast.AstNode{ .Statement = stmt.* };
        return new_node;
    }

    fn addError(self: *Self, message: []const u8, span: ?ast.SourceSpan, error_type: TransformError.ErrorType) !void {
        try self.errors.append(TransformError{
            .message = message,
            .node_span = span,
            .error_type = error_type,
        });
    }
};

/// Validation visitor for comprehensive AST validation
pub const ValidationVisitor = struct {
    const Self = @This();

    allocator: std.mem.Allocator,
    errors: ManagedArrayList(ValidationError),
    warnings: ManagedArrayList(ValidationWarning),

    // validation functions
    validateContract: ?*const fn (*Self, *ast.ContractNode) anyerror!void = null,
    validateFunction: ?*const fn (*Self, *ast.FunctionNode) anyerror!void = null,
    validateExpression: ?*const fn (*Self, *ast.Expressions.ExprNode) anyerror!void = null,
    validateStatement: ?*const fn (*Self, *ast.Statements.StmtNode) anyerror!void = null,

    pub const ValidationError = struct {
        message: []const u8,
        span: ast.SourceSpan,
        severity: Severity = .Error,

        pub const Severity = enum {
            Error,
            Warning,
            Info,
        };
    };

    pub const ValidationWarning = struct {
        message: []const u8,
        span: ast.SourceSpan,
        suggestion: ?[]const u8 = null,
    };

    pub const ValidationResult = struct {
        is_valid: bool,
        errors: []const ValidationError,
        warnings: []const ValidationWarning,
    };

    pub fn init(allocator: std.mem.Allocator) Self {
        return Self{
            .allocator = allocator,
            .errors = ManagedArrayList(ValidationError).init(allocator),
            .warnings = ManagedArrayList(ValidationWarning).init(allocator),
        };
    }

    pub fn deinit(self: *Self) void {
        self.errors.deinit();
        self.warnings.deinit();
    }

    pub fn validate(self: *Self, node: *ast.AstNode) !ValidationResult {
        self.errors.clearRetainingCapacity();
        self.warnings.clearRetainingCapacity();

        try self.validateRecursive(node);

        return ValidationResult{
            .is_valid = self.errors.items.len == 0,
            .errors = self.errors.items,
            .warnings = self.warnings.items,
        };
    }

    fn validateRecursive(self: *Self, node: *ast.AstNode) anyerror!void {
        switch (node.*) {
            .Contract => |*contract| {
                if (self.validateContract) |validateFn| {
                    try validateFn(self, contract);
                }
                // validate children
                for (contract.body) |*child| {
                    try self.validateRecursive(child);
                }
            },
            .Function => |*function| {
                if (self.validateFunction) |validateFn| {
                    try validateFn(self, function);
                }
                // validate function body
                var block_node = ast.AstNode{ .Block = function.body };
                try self.validateRecursive(&block_node);
            },
            .Expression => |expr| {
                if (self.validateExpression) |validateFn| {
                    try validateFn(self, expr);
                }
                try self.validateExpressionRecursive(expr);
            },
            .Statement => |stmt| {
                if (self.validateStatement) |validateFn| {
                    try validateFn(self, stmt);
                }
                try self.validateStatementRecursive(stmt);
            },
            .Block => |*block| {
                for (block.statements) |*stmt| {
                    var stmt_node = ast.AstNode{ .Statement = stmt.* };
                    try self.validateRecursive(&stmt_node);
                }
            },
            else => {},
        }
    }

    fn validateExpressionRecursive(self: *Self, expr: *ast.Expressions.ExprNode) anyerror!void {
        switch (expr.*) {
            .Binary => |*binary| {
                try self.validateExpressionRecursive(binary.lhs);
                try self.validateExpressionRecursive(binary.rhs);
            },
            .Unary => |*unary| {
                try self.validateExpressionRecursive(unary.operand);
            },
            .Call => |*call| {
                try self.validateExpressionRecursive(call.callee);
                for (call.arguments) |*arg| {
                    try self.validateExpressionRecursive(arg);
                }
            },
            else => {},
        }
    }

    fn validateStatementRecursive(self: *Self, stmt: *ast.Statements.StmtNode) anyerror!void {
        switch (stmt.*) {
            .Expr => |*expr| {
                try self.validateExpressionRecursive(expr);
            },
            .If => |*if_stmt| {
                try self.validateExpressionRecursive(&if_stmt.condition);
                var then_node = ast.AstNode{ .Block = if_stmt.then_branch };
                try self.validateRecursive(&then_node);
                if (if_stmt.else_branch) |else_branch| {
                    var else_node = ast.AstNode{ .Block = else_branch };
                    try self.validateRecursive(&else_node);
                }
            },
            else => {},
        }
    }

    pub fn addError(self: *Self, message: []const u8, span: ast.SourceSpan) !void {
        try self.errors.append(ValidationError{
            .message = message,
            .span = span,
        });
    }

    pub fn addWarning(self: *Self, message: []const u8, span: ast.SourceSpan, suggestion: ?[]const u8) !void {
        try self.warnings.append(ValidationWarning{
            .message = message,
            .span = span,
            .suggestion = suggestion,
        });
    }
};

/// Visitor composition and chaining system
/// Allows combining multiple visitors into a pipeline with error propagation and recovery
/// Generic visitor wrapper for composition
pub const AnyVisitor = struct {
    ptr: *anyopaque,
    visitFn: *const fn (*anyopaque, *ast.AstNode) anyerror!bool,
    deinitFn: ?*const fn (*anyopaque) void,
    name: []const u8,

    pub fn init(comptime T: type, visitor: *T, name: []const u8) AnyVisitor {
        const Impl = struct {
            fn visit(ptr: *anyopaque, node: *ast.AstNode) anyerror!bool {
                const self: *T = @ptrCast(@alignCast(ptr));
                if (T == MutableVisitor) {
                    try self.visitAndModify(node);
                    return true;
                } else if (T == ValidationVisitor) {
                    const result = try self.validate(node);
                    return result.is_valid;
                } else if (T == TransformVisitor) {
                    const transformed = try self.transform(node);
                    return transformed != null;
                } else {
                    // for generic visitors, assume void return means success
                    _ = self.visit(node);
                    return true;
                }
            }

            fn deinit(ptr: *anyopaque) void {
                const self: *T = @ptrCast(@alignCast(ptr));
                if (@hasDecl(T, "deinit")) {
                    self.deinit();
                }
            }
        };

        return AnyVisitor{
            .ptr = visitor,
            .visitFn = Impl.visit,
            .deinitFn = if (@hasDecl(T, "deinit")) Impl.deinit else null,
            .name = name,
        };
    }

    pub fn visit(self: *AnyVisitor, node: *ast.AstNode) anyerror!bool {
        return self.visitFn(self.ptr, node);
    }

    pub fn deinit(self: *AnyVisitor) void {
        if (self.deinitFn) |deinitFn| {
            deinitFn(self.ptr);
        }
    }
};

/// Visitor chain for combining multiple visitors
pub const VisitorChain = struct {
    const Self = @This();

    allocator: std.mem.Allocator,
    visitors: ManagedArrayList(AnyVisitor),
    early_exit_on_error: bool,
    continue_on_error: bool,
    error_count: u32,
    success_count: u32,
    results: ManagedArrayList(VisitorResult),

    pub const VisitorResult = struct {
        visitor_name: []const u8,
        success: bool,
        error_message: ?[]const u8,
        node_count: u32,
    };

    pub fn init(allocator: std.mem.Allocator) Self {
        return Self{
            .allocator = allocator,
            .visitors = ManagedArrayList(AnyVisitor).init(allocator),
            .early_exit_on_error = false,
            .continue_on_error = true,
            .error_count = 0,
            .success_count = 0,
            .results = ManagedArrayList(VisitorResult).init(allocator),
        };
    }

    pub fn deinit(self: *Self) void {
        for (self.visitors.items) |*visitor| {
            visitor.deinit();
        }
        self.visitors.deinit();
        self.results.deinit();
    }

    /// Add a visitor to the chain
    pub fn add(self: *Self, visitor: AnyVisitor) !void {
        try self.visitors.append(visitor);
    }

    /// Add a typed visitor to the chain
    pub fn addTyped(self: *Self, comptime T: type, visitor: *T, name: []const u8) !void {
        const any_visitor = AnyVisitor.init(T, visitor, name);
        try self.add(any_visitor);
    }

    /// Set early exit behavior
    pub fn setEarlyExit(self: *Self, early_exit: bool) void {
        self.early_exit_on_error = early_exit;
    }

    /// Set error continuation behavior
    pub fn setContinueOnError(self: *Self, continue_on_error: bool) void {
        self.continue_on_error = continue_on_error;
    }

    /// Visit a node with all visitors in the chain
    pub fn visit(self: *Self, node: *ast.AstNode) !void {
        self.error_count = 0;
        self.success_count = 0;
        self.results.clearRetainingCapacity();

        for (self.visitors.items) |*visitor| {
            var result = VisitorResult{
                .visitor_name = visitor.name,
                .success = false,
                .error_message = null,
                .node_count = 1,
            };

            const success = visitor.visit(node) catch |err| {
                self.error_count += 1;
                result.error_message = @errorName(err);

                if (!self.continue_on_error) {
                    try self.results.append(result);
                    return err;
                }

                if (self.early_exit_on_error) {
                    try self.results.append(result);
                    break;
                }
            };

            result.success = success;
            if (success) {
                self.success_count += 1;
            } else {
                self.error_count += 1;
            }

            try self.results.append(result);

            if (self.early_exit_on_error and !success) {
                break;
            }
        }
    }

    /// Visit with early exit support - returns true if should continue
    pub fn visitWithEarlyExit(self: *Self, node: *ast.AstNode) !bool {
        try self.visit(node);
        return self.error_count == 0 or self.continue_on_error;
    }

    /// Get aggregated results
    pub fn getResults(self: *const Self) []const VisitorResult {
        return self.results.items;
    }

    /// Check if all visitors succeeded
    pub fn allSucceeded(self: *const Self) bool {
        return self.error_count == 0;
    }

    /// Get success rate
    pub fn getSuccessRate(self: *const Self) f32 {
        if (self.visitors.items.len == 0) return 1.0;
        return @as(f32, @floatFromInt(self.success_count)) / @as(f32, @floatFromInt(self.visitors.items.len));
    }

    /// Filter results by success/failure
    pub fn getSuccessfulResults(self: *const Self, allocator: std.mem.Allocator) ![]VisitorResult {
        var successful = ManagedArrayList(VisitorResult).init(allocator);
        for (self.results.items) |result| {
            if (result.success) {
                try successful.append(result);
            }
        }
        return successful.toOwnedSlice();
    }

    pub fn getFailedResults(self: *const Self, allocator: std.mem.Allocator) ![]VisitorResult {
        var failed = ManagedArrayList(VisitorResult).init(allocator);
        for (self.results.items) |result| {
            if (!result.success) {
                try failed.append(result);
            }
        }
        return failed.toOwnedSlice();
    }
};

/// Visitor pipeline with advanced error propagation and recovery
pub const VisitorPipeline = struct {
    const Self = @This();

    allocator: std.mem.Allocator,
    stages: ManagedArrayList(PipelineStage),
    recovery_strategies: std.HashMap([]const u8, RecoveryStrategy),
    global_error_handler: ?*const fn ([]const u8, ?ast.SourceSpan) void,

    pub const PipelineStage = struct {
        name: []const u8,
        chain: VisitorChain,
        required: bool,
        retry_count: u32,
        timeout_ms: ?u32,
    };

    pub const RecoveryStrategy = enum {
        Skip,
        Retry,
        Fallback,
        Abort,
    };

    pub const PipelineResult = struct {
        success: bool,
        stages_completed: u32,
        total_stages: u32,
        errors: [][]const u8,
        stage_results: []StageResult,
    };

    pub const StageResult = struct {
        stage_name: []const u8,
        success: bool,
        retry_attempts: u32,
        visitor_results: []VisitorChain.VisitorResult,
    };

    pub fn init(allocator: std.mem.Allocator) Self {
        return Self{
            .allocator = allocator,
            .stages = ManagedArrayList(PipelineStage).init(allocator),
            .recovery_strategies = std.HashMap([]const u8, RecoveryStrategy).init(allocator),
        };
    }

    pub fn deinit(self: *Self) void {
        for (self.stages.items) |*stage| {
            stage.chain.deinit();
        }
        self.stages.deinit();
        self.recovery_strategies.deinit();
    }

    /// Add a pipeline stage
    pub fn addStage(self: *Self, name: []const u8, required: bool) !*PipelineStage {
        const stage = PipelineStage{
            .name = name,
            .chain = VisitorChain.init(self.allocator),
            .required = required,
            .retry_count = 0,
            .timeout_ms = null,
        };
        try self.stages.append(stage);
        return &self.stages.items[self.stages.items.len - 1];
    }

    /// Set recovery strategy for a stage
    pub fn setRecoveryStrategy(self: *Self, stage_name: []const u8, strategy: RecoveryStrategy) !void {
        try self.recovery_strategies.put(stage_name, strategy);
    }

    /// Set global error handler
    pub fn setGlobalErrorHandler(self: *Self, handler: *const fn ([]const u8, ?ast.SourceSpan) void) void {
        self.global_error_handler = handler;
    }

    /// Execute the pipeline on a node
    pub fn execute(self: *Self, node: *ast.AstNode) !PipelineResult {
        var result = PipelineResult{
            .success = true,
            .stages_completed = 0,
            .total_stages = @intCast(self.stages.items.len),
            .errors = &[_][]const u8{},
            .stage_results = &[_]StageResult{},
        };

        var errors = ManagedArrayList([]const u8).init(self.allocator);
        defer errors.deinit();

        var stage_results = ManagedArrayList(StageResult).init(self.allocator);
        defer stage_results.deinit();

        for (self.stages.items) |*stage| {
            var stage_result = StageResult{
                .stage_name = stage.name,
                .success = false,
                .retry_attempts = 0,
                .visitor_results = &[_]VisitorChain.VisitorResult{},
            };

            const max_retries = if (stage.retry_count > 0) stage.retry_count else 1;
            var retry_attempt: u32 = 0;

            while (retry_attempt < max_retries) {
                stage.chain.visit(node) catch |err| {
                    retry_attempt += 1;
                    stage_result.retry_attempts = retry_attempt;

                    if (self.global_error_handler) |handler| {
                        handler(@errorName(err), null);
                    }

                    const strategy = self.recovery_strategies.get(stage.name) orelse .Skip;
                    switch (strategy) {
                        .Skip => break,
                        .Retry => {
                            if (retry_attempt < max_retries) continue;
                            break;
                        },
                        .Fallback => {
                            // could implement fallback logic here
                            break;
                        },
                        .Abort => {
                            result.success = false;
                            try errors.append(@errorName(err));
                            try stage_results.append(stage_result);
                            result.errors = try errors.toOwnedSlice();
                            result.stage_results = try stage_results.toOwnedSlice();
                            return result;
                        },
                    }
                };

                // if we get here, the stage succeeded
                stage_result.success = true;
                stage_result.visitor_results = stage.chain.getResults();
                break;
            }

            if (!stage_result.success and stage.required) {
                result.success = false;
                try errors.append("Required stage failed");
            }

            if (stage_result.success) {
                result.stages_completed += 1;
            }

            try stage_results.append(stage_result);
        }

        result.errors = try errors.toOwnedSlice();
        result.stage_results = try stage_results.toOwnedSlice();
        return result;
    }
};
