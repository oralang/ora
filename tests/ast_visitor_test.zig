const std = @import("std");
const testing = std.testing;
const ora = @import("ora");
const ast = ora.ast;

// Comprehensive test context for tracking visited nodes
const TestContext = struct {
    visited_nodes: std.StringHashMap(u32),
    allocator: std.mem.Allocator,

    pub fn init(allocator: std.mem.Allocator) TestContext {
        return TestContext{
            .visited_nodes = std.StringHashMap(u32).init(allocator),
            .allocator = allocator,
        };
    }

    pub fn deinit(self: *TestContext) void {
        self.visited_nodes.deinit();
    }

    pub fn addVisited(self: *TestContext, node_type: []const u8) void {
        if (self.visited_nodes.get(node_type)) |count| {
            _ = self.visited_nodes.put(node_type, count + 1) catch unreachable;
        } else {
            _ = self.visited_nodes.put(node_type, 1) catch unreachable;
        }
    }

    pub fn hasVisited(self: *TestContext, node_type: []const u8) bool {
        return self.visited_nodes.contains(node_type);
    }

    pub fn getVisitCount(self: *TestContext, node_type: []const u8) u32 {
        return self.visited_nodes.get(node_type) orelse 0;
    }
};

// Test all AST node types to identify missing visitor implementations
test "comprehensive AST visitor coverage" {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const allocator = arena.allocator();

    var context = TestContext.init(testing.allocator);
    defer context.deinit();

    // Create a comprehensive AST with all node types
    const root_node = try createComprehensiveAST(allocator);

    // Create visitor that tracks all node types
    var visitor = ast.ast_visitor.Visitor(TestContext, void){
        .context = &context,

        // Top-level node visitors
        .visitModule = visitModule,
        .visitContract = visitContract,
        .visitFunction = visitFunction,
        .visitConstant = visitConstant,
        .visitVariableDecl = visitVariableDecl,
        .visitStructDecl = visitStructDecl,
        .visitEnumDecl = visitEnumDecl,
        .visitLogDecl = visitLogDecl,
        .visitImport = visitImport,
        .visitErrorDecl = visitErrorDecl,
        .visitBlock = visitBlock,
        .visitTryBlock = visitTryBlock,

        // Expression visitors
        .visitIdentifier = visitIdentifier,
        .visitLiteral = visitLiteral,
        .visitBinary = visitBinary,
        .visitUnary = visitUnary,
        .visitAssignment = visitAssignment,
        .visitCompoundAssignment = visitCompoundAssignment,
        .visitCall = visitCall,
        .visitIndex = visitIndex,
        .visitFieldAccess = visitFieldAccess,
        .visitCast = visitCast,
        .visitComptime = visitComptime,
        .visitOld = visitOld,
        .visitTuple = visitTuple,
        .visitTry = visitTry,
        .visitErrorReturn = visitErrorReturn,
        .visitErrorCast = visitErrorCast,
        .visitShift = visitShift,
        .visitStructInstantiation = visitStructInstantiation,
        .visitEnumLiteral = visitEnumLiteral,
        .visitArrayLiteral = visitArrayLiteral,
        .visitSwitchExpression = visitSwitchExpression,
        .visitRangeExpr = visitRangeExpr,
        .visitQuantified = visitQuantified,
        .visitAnonymousStruct = struct_fn_expr_visit_anon,
        .visitDestructuring = struct_fn_expr_visit_destruct,
        .visitLabeledBlockExpr = struct_fn_expr_visit_labeled_block,

        // Statement visitors
        .visitReturn = visitReturn,
        .visitIf = visitIf,
        .visitWhile = visitWhile,
        .visitLog = visitLog,
        .visitLock = visitLock,
        .visitInvariant = visitInvariant,
        .visitRequires = visitRequires,
        .visitEnsures = visitEnsures,
        .visitForLoop = visitForLoop,
        .visitBreak = visitBreak,
        .visitContinue = visitContinue,
        .visitUnlock = visitUnlock,
        .visitMove = visitMove,
        .visitDestructuringAssignment = visitDestructuringAssignment,
        .visitLabeledBlockStmt = visitLabeledBlockStmt,
    };

    // Test pre-order traversal
    _ = visitor.walkPreOrder(root_node);

    // Verify all expected node types were visited
    const expected_node_types = [_][]const u8{
        // Top-level nodes
        "Module",          "Contract",                "Function",           "Constant",  "VariableDecl",
        "StructDecl",      "EnumDecl",                "LogDecl",            "Import",    "ErrorDecl",
        "Block",           "TryBlock",

        // Expression nodes
                       "Identifier",         "Literal",   "Binary",
        "Unary",           "Assignment",              "CompoundAssignment", "Call",      "Index",
        "FieldAccess",     "Cast",                    "Comptime",           "Old",       "Tuple",
        "Try",             "ErrorReturn",             "ErrorCast",          "Shift",     "StructInstantiation",
        "EnumLiteral",     "ArrayLiteral",            "SwitchExpression",   "Range",     "Quantified",
        "AnonymousStruct", "DestructuringExpr",       "LabeledBlockExpr",

        // Statement nodes
          "Return",    "If",
        "While",           "Log",                     "Lock",               "Invariant", "Requires",
        "Ensures",         "ForLoop",                 "Break",              "Continue",  "Unlock",
        "Move",            "DestructuringAssignment", "LabeledBlockStmt",
    };

    for (expected_node_types) |node_type| {
        if (!context.hasVisited(node_type)) {
            std.debug.print("MISSING: {s} visitor not implemented\n", .{node_type});
        } else {
            const count = context.getVisitCount(node_type);
            std.debug.print("✓ {s}: visited {d} times\n", .{ node_type, count });
        }
    }

    // Test that we have at least some coverage
    var total_visited: u32 = 0;
    var iterator = context.visited_nodes.iterator();
    while (iterator.next()) |entry| {
        total_visited += entry.value_ptr.*;
    }

    try testing.expect(total_visited > 0);
    std.debug.print("\nTotal nodes visited: {d}\n", .{total_visited});
}

// Visitor functions for all node types
fn visitModule(visitor: *ast.ast_visitor.Visitor(TestContext, void), _: *ast.ModuleNode) void {
    visitor.context.addVisited("Module");
}

fn visitContract(visitor: *ast.ast_visitor.Visitor(TestContext, void), _: *ast.ContractNode) void {
    visitor.context.addVisited("Contract");
}

fn visitFunction(visitor: *ast.ast_visitor.Visitor(TestContext, void), _: *ast.FunctionNode) void {
    visitor.context.addVisited("Function");
}

fn visitConstant(visitor: *ast.ast_visitor.Visitor(TestContext, void), _: *ast.ConstantNode) void {
    visitor.context.addVisited("Constant");
}

fn visitVariableDecl(visitor: *ast.ast_visitor.Visitor(TestContext, void), _: *ast.Statements.VariableDeclNode) void {
    visitor.context.addVisited("VariableDecl");
}

fn visitStructDecl(visitor: *ast.ast_visitor.Visitor(TestContext, void), _: *ast.StructDeclNode) void {
    visitor.context.addVisited("StructDecl");
}

fn visitEnumDecl(visitor: *ast.ast_visitor.Visitor(TestContext, void), _: *ast.EnumDeclNode) void {
    visitor.context.addVisited("EnumDecl");
}

fn visitLogDecl(visitor: *ast.ast_visitor.Visitor(TestContext, void), _: *ast.LogDeclNode) void {
    visitor.context.addVisited("LogDecl");
}

fn visitImport(visitor: *ast.ast_visitor.Visitor(TestContext, void), _: *ast.ImportNode) void {
    visitor.context.addVisited("Import");
}

fn visitErrorDecl(visitor: *ast.ast_visitor.Visitor(TestContext, void), _: *ast.Statements.ErrorDeclNode) void {
    visitor.context.addVisited("ErrorDecl");
}

fn visitBlock(visitor: *ast.ast_visitor.Visitor(TestContext, void), _: *ast.Statements.BlockNode) void {
    visitor.context.addVisited("Block");
}

fn visitTryBlock(visitor: *ast.ast_visitor.Visitor(TestContext, void), _: *ast.Statements.TryBlockNode) void {
    visitor.context.addVisited("TryBlock");
}

fn visitIdentifier(visitor: *ast.ast_visitor.Visitor(TestContext, void), _: *ast.Expressions.IdentifierExpr) void {
    visitor.context.addVisited("Identifier");
}

fn visitLiteral(visitor: *ast.ast_visitor.Visitor(TestContext, void), _: *ast.Expressions.LiteralExpr) void {
    visitor.context.addVisited("Literal");
}

fn visitBinary(visitor: *ast.ast_visitor.Visitor(TestContext, void), _: *ast.Expressions.BinaryExpr) void {
    visitor.context.addVisited("Binary");
}

fn visitUnary(visitor: *ast.ast_visitor.Visitor(TestContext, void), _: *ast.Expressions.UnaryExpr) void {
    visitor.context.addVisited("Unary");
}

fn visitAssignment(visitor: *ast.ast_visitor.Visitor(TestContext, void), _: *ast.Expressions.AssignmentExpr) void {
    visitor.context.addVisited("Assignment");
}

fn visitCompoundAssignment(visitor: *ast.ast_visitor.Visitor(TestContext, void), _: *ast.Expressions.CompoundAssignmentExpr) void {
    visitor.context.addVisited("CompoundAssignment");
}

fn visitCall(visitor: *ast.ast_visitor.Visitor(TestContext, void), _: *ast.Expressions.CallExpr) void {
    visitor.context.addVisited("Call");
}

fn visitIndex(visitor: *ast.ast_visitor.Visitor(TestContext, void), _: *ast.Expressions.IndexExpr) void {
    visitor.context.addVisited("Index");
}

fn visitFieldAccess(visitor: *ast.ast_visitor.Visitor(TestContext, void), _: *ast.Expressions.FieldAccessExpr) void {
    visitor.context.addVisited("FieldAccess");
}

fn visitCast(visitor: *ast.ast_visitor.Visitor(TestContext, void), _: *ast.Expressions.CastExpr) void {
    visitor.context.addVisited("Cast");
}

fn visitComptime(visitor: *ast.ast_visitor.Visitor(TestContext, void), _: *ast.Expressions.ComptimeExpr) void {
    visitor.context.addVisited("Comptime");
}

fn visitOld(visitor: *ast.ast_visitor.Visitor(TestContext, void), _: *ast.Expressions.OldExpr) void {
    visitor.context.addVisited("Old");
}

fn visitTuple(visitor: *ast.ast_visitor.Visitor(TestContext, void), _: *ast.Expressions.TupleExpr) void {
    visitor.context.addVisited("Tuple");
}

fn visitTry(visitor: *ast.ast_visitor.Visitor(TestContext, void), _: *ast.Expressions.TryExpr) void {
    visitor.context.addVisited("Try");
}

fn visitErrorReturn(visitor: *ast.ast_visitor.Visitor(TestContext, void), _: *ast.Expressions.ErrorReturnExpr) void {
    visitor.context.addVisited("ErrorReturn");
}

fn visitErrorCast(visitor: *ast.ast_visitor.Visitor(TestContext, void), _: *ast.Expressions.ErrorCastExpr) void {
    visitor.context.addVisited("ErrorCast");
}

fn visitShift(visitor: *ast.ast_visitor.Visitor(TestContext, void), _: *ast.Expressions.ShiftExpr) void {
    visitor.context.addVisited("Shift");
}

fn visitStructInstantiation(visitor: *ast.ast_visitor.Visitor(TestContext, void), _: *ast.Expressions.StructInstantiationExpr) void {
    visitor.context.addVisited("StructInstantiation");
}

fn visitEnumLiteral(visitor: *ast.ast_visitor.Visitor(TestContext, void), _: *ast.Expressions.EnumLiteralExpr) void {
    visitor.context.addVisited("EnumLiteral");
}

fn visitArrayLiteral(visitor: *ast.ast_visitor.Visitor(TestContext, void), _: *ast.Literals.Array) void {
    visitor.context.addVisited("ArrayLiteral");
}

fn visitSwitchExpression(visitor: *ast.ast_visitor.Visitor(TestContext, void), _: *ast.Switch.ExprNode) void {
    visitor.context.addVisited("SwitchExpression");
}

fn visitRangeExpr(visitor: *ast.ast_visitor.Visitor(TestContext, void), _: *ast.Expressions.RangeExpr) void {
    visitor.context.addVisited("Range");
}

fn visitQuantified(visitor: *ast.ast_visitor.Visitor(TestContext, void), _: *ast.Expressions.QuantifiedExpr) void {
    visitor.context.addVisited("Quantified");
}

fn struct_fn_expr_visit_anon(visitor: *ast.ast_visitor.Visitor(TestContext, void), _: *ast.Expressions.AnonymousStructExpr) void {
    visitor.context.addVisited("AnonymousStruct");
}

fn struct_fn_expr_visit_destruct(visitor: *ast.ast_visitor.Visitor(TestContext, void), _: *ast.Expressions.DestructuringExpr) void {
    visitor.context.addVisited("DestructuringExpr");
}

fn struct_fn_expr_visit_labeled_block(visitor: *ast.ast_visitor.Visitor(TestContext, void), _: *ast.Expressions.LabeledBlockExpr) void {
    visitor.context.addVisited("LabeledBlockExpr");
}

fn visitReturn(visitor: *ast.ast_visitor.Visitor(TestContext, void), _: *ast.Statements.ReturnNode) void {
    visitor.context.addVisited("Return");
}

fn visitIf(visitor: *ast.ast_visitor.Visitor(TestContext, void), _: *ast.Statements.IfNode) void {
    visitor.context.addVisited("If");
}

fn visitWhile(visitor: *ast.ast_visitor.Visitor(TestContext, void), _: *ast.Statements.WhileNode) void {
    visitor.context.addVisited("While");
}

fn visitLog(visitor: *ast.ast_visitor.Visitor(TestContext, void), _: *ast.Statements.LogNode) void {
    visitor.context.addVisited("Log");
}

fn visitLock(visitor: *ast.ast_visitor.Visitor(TestContext, void), _: *ast.Statements.LockNode) void {
    visitor.context.addVisited("Lock");
}

fn visitInvariant(visitor: *ast.ast_visitor.Visitor(TestContext, void), _: *ast.Statements.InvariantNode) void {
    visitor.context.addVisited("Invariant");
}

fn visitRequires(visitor: *ast.ast_visitor.Visitor(TestContext, void), _: *ast.Statements.RequiresNode) void {
    visitor.context.addVisited("Requires");
}

fn visitEnsures(visitor: *ast.ast_visitor.Visitor(TestContext, void), _: *ast.Statements.EnsuresNode) void {
    visitor.context.addVisited("Ensures");
}

fn visitForLoop(visitor: *ast.ast_visitor.Visitor(TestContext, void), _: *ast.Statements.ForLoopNode) void {
    visitor.context.addVisited("ForLoop");
}

fn visitBreak(visitor: *ast.ast_visitor.Visitor(TestContext, void), _: *ast.Statements.BreakNode) void {
    visitor.context.addVisited("Break");
}

fn visitContinue(visitor: *ast.ast_visitor.Visitor(TestContext, void), _: *ast.Statements.ContinueNode) void {
    visitor.context.addVisited("Continue");
}

fn visitUnlock(visitor: *ast.ast_visitor.Visitor(TestContext, void), _: *ast.Statements.UnlockNode) void {
    visitor.context.addVisited("Unlock");
}

fn visitMove(visitor: *ast.ast_visitor.Visitor(TestContext, void), _: *ast.Statements.MoveNode) void {
    visitor.context.addVisited("Move");
}

fn visitDestructuringAssignment(visitor: *ast.ast_visitor.Visitor(TestContext, void), _: *ast.Statements.DestructuringAssignmentNode) void {
    visitor.context.addVisited("DestructuringAssignment");
}

fn visitLabeledBlockStmt(visitor: *ast.ast_visitor.Visitor(TestContext, void), _: *ast.Statements.LabeledBlockNode) void {
    visitor.context.addVisited("LabeledBlockStmt");
}

// Helper function to create a comprehensive AST with all node types
fn createComprehensiveAST(allocator: std.mem.Allocator) !*ast.AstNode {
    const root_node = try allocator.create(ast.AstNode);

    // Create a module with various declarations
    const module_node = try allocator.create(ast.ModuleNode);
    module_node.* = .{
        .name = "test_module",
        .imports = &[_]ast.ImportNode{},
        .declarations = &[_]ast.AstNode{},
        .span = .{ .line = 1, .column = 1, .length = 10, .byte_offset = 0 },
    };

    // Create various declarations
    var declarations = std.ArrayList(ast.AstNode).init(allocator);

    // Add contract
    try declarations.append(try createContractNode(allocator));

    // Add function
    try declarations.append(try createFunctionNode(allocator));

    // Add struct
    try declarations.append(try createStructNode(allocator));

    // Add enum
    try declarations.append(try createEnumNode(allocator));

    // Add constant
    try declarations.append(try createConstantNode(allocator));

    // Add variable declaration
    try declarations.append(try createVariableDeclNode(allocator));

    // Add log declaration
    try declarations.append(try createLogDeclNode(allocator));

    // Add import
    try declarations.append(try createImportNode(allocator));

    // Add error declaration
    try declarations.append(try createErrorDeclNode(allocator));

    // Add try block
    try declarations.append(try createTryBlockNode(allocator));

    module_node.declarations = try declarations.toOwnedSlice();
    root_node.* = .{ .Module = module_node.* };

    return root_node;
}

fn createContractNode(allocator: std.mem.Allocator) !ast.AstNode {
    const contract = try allocator.create(ast.ContractNode);
    contract.* = .{
        .name = "TestContract",
        .extends = null,
        .implements = &[_][]const u8{},
        .attributes = &[_]u8{},
        .body = &[_]ast.AstNode{},
        .span = .{ .line = 2, .column = 1, .length = 20, .byte_offset = 0 },
    };
    return .{ .Contract = contract.* };
}

fn createFunctionNode(allocator: std.mem.Allocator) !ast.AstNode {
    const function = try allocator.create(ast.FunctionNode);

    // Create some statements for the function body
    var statements = std.ArrayList(ast.Statements.StmtNode).init(allocator);

    // Add a return statement
    const literal_expr = try createLiteralExpr(allocator);
    const return_stmt = ast.Statements.StmtNode{
        .Return = .{
            .value = literal_expr.*,
            .span = .{ .line = 4, .column = 5, .length = 10, .byte_offset = 0 },
        },
    };
    try statements.append(return_stmt);

    // Add an if statement
    const binary_expr = try createBinaryExpr(allocator);
    const if_stmt = ast.Statements.StmtNode{
        .If = .{
            .condition = binary_expr.*,
            .then_branch = .{
                .statements = &[_]ast.Statements.StmtNode{},
                .span = .{ .line = 5, .column = 5, .length = 5, .byte_offset = 0 },
            },
            .else_branch = null,
            .span = .{ .line = 5, .column = 1, .length = 20, .byte_offset = 0 },
        },
    };
    try statements.append(if_stmt);

    // Add a while statement
    const unary_expr = try createUnaryExpr(allocator);
    const while_stmt = ast.Statements.StmtNode{
        .While = .{
            .condition = unary_expr.*,
            .body = .{
                .statements = &[_]ast.Statements.StmtNode{},
                .span = .{ .line = 6, .column = 5, .length = 5, .byte_offset = 0 },
            },
            .invariants = &[_]ast.Expressions.ExprNode{},
            .span = .{ .line = 6, .column = 1, .length = 20, .byte_offset = 0 },
        },
    };
    try statements.append(while_stmt);

    // Add a for loop statement
    const for_iterable = try createIdentifierExpr(allocator, "items");
    const for_body = ast.Statements.BlockNode{
        .statements = &[_]ast.Statements.StmtNode{},
        .span = .{ .line = 6, .column = 5, .length = 5, .byte_offset = 0 },
    };
    const loop_pattern = ast.Statements.LoopPattern{ .Single = .{ .name = "item", .span = .{ .line = 6, .column = 10, .length = 4, .byte_offset = 0 } } };
    const for_stmt = ast.Statements.StmtNode{
        .ForLoop = .{
            .iterable = for_iterable.*,
            .pattern = loop_pattern,
            .body = for_body,
            .span = .{ .line = 6, .column = 1, .length = 20, .byte_offset = 0 },
        },
    };
    try statements.append(for_stmt);

    // Add a log statement
    const log_stmt = ast.Statements.StmtNode{
        .Log = .{
            .event_name = "test_log",
            .args = &[_]ast.Expressions.ExprNode{},
            .span = .{ .line = 7, .column = 1, .length = 15, .byte_offset = 0 },
        },
    };
    try statements.append(log_stmt);

    // Add a lock statement
    const lock_path = try createIdentifierExpr(allocator, "lock_target");
    const lock_stmt = ast.Statements.StmtNode{
        .Lock = .{
            .path = lock_path.*,
            .span = .{ .line = 8, .column = 1, .length = 15, .byte_offset = 0 },
        },
    };
    try statements.append(lock_stmt);

    // Add an invariant statement
    const invariant_condition = try createLiteralExpr(allocator);
    const invariant_stmt = ast.Statements.StmtNode{
        .Invariant = .{
            .condition = invariant_condition.*,
            .span = .{ .line = 9, .column = 1, .length = 15, .byte_offset = 0 },
        },
    };
    try statements.append(invariant_stmt);

    // Add a requires statement
    const requires_condition = try createLiteralExpr(allocator);
    const requires_stmt = ast.Statements.StmtNode{
        .Requires = .{
            .condition = requires_condition.*,
            .span = .{ .line = 10, .column = 1, .length = 15, .byte_offset = 0 },
        },
    };
    try statements.append(requires_stmt);

    // Add an ensures statement
    const ensures_condition = try createLiteralExpr(allocator);
    const ensures_stmt = ast.Statements.StmtNode{
        .Ensures = .{
            .condition = ensures_condition.*,
            .span = .{ .line = 11, .column = 1, .length = 15, .byte_offset = 0 },
        },
    };
    try statements.append(ensures_stmt);

    // Add an expression statement with a call
    const call_expr = try createCallExpr(allocator);
    const expr_stmt = ast.Statements.StmtNode{
        .Expr = call_expr.*,
    };
    try statements.append(expr_stmt);

    // Add an expression statement with field access
    const field_access_expr = try createFieldAccessExpr(allocator);
    const field_stmt = ast.Statements.StmtNode{
        .Expr = field_access_expr.*,
    };
    try statements.append(field_stmt);

    // Add an expression statement with index access
    const index_expr = try createIndexExpr(allocator);
    const index_stmt = ast.Statements.StmtNode{
        .Expr = index_expr.*,
    };
    try statements.append(index_stmt);

    // Add an expression statement with cast
    const cast_expr = try createCastExpr(allocator);
    const cast_stmt = ast.Statements.StmtNode{
        .Expr = cast_expr.*,
    };
    try statements.append(cast_stmt);

    // Add an expression statement with tuple
    const tuple_expr = try createTupleExpr(allocator);
    const tuple_stmt = ast.Statements.StmtNode{
        .Expr = tuple_expr.*,
    };
    try statements.append(tuple_stmt);

    // Add an expression statement with range
    const range_expr = try createRangeExpr(allocator);
    const range_stmt = ast.Statements.StmtNode{
        .Expr = range_expr.*,
    };
    try statements.append(range_stmt);

    // Add an expression statement with array literal
    const array_expr = try createArrayLiteralExpr(allocator);
    const array_stmt = ast.Statements.StmtNode{
        .Expr = array_expr.*,
    };
    try statements.append(array_stmt);

    // Add an expression statement with try
    const try_expr = try createTryExpr(allocator);
    const try_stmt = ast.Statements.StmtNode{
        .Expr = try_expr.*,
    };
    try statements.append(try_stmt);

    // Add an expression statement with error return
    const error_return_expr = try createErrorReturnExpr(allocator);
    const error_return_stmt = ast.Statements.StmtNode{
        .Expr = error_return_expr.*,
    };
    try statements.append(error_return_stmt);

    // Add an expression statement with error cast
    const error_cast_expr = try createErrorCastExpr(allocator);
    const error_cast_stmt = ast.Statements.StmtNode{
        .Expr = error_cast_expr.*,
    };
    try statements.append(error_cast_stmt);

    // Add an expression statement with shift
    const shift_expr = try createShiftExpr(allocator);
    const shift_stmt = ast.Statements.StmtNode{
        .Expr = shift_expr.*,
    };
    try statements.append(shift_stmt);

    // Add an expression statement with struct instantiation
    const struct_inst_expr = try createStructInstantiationExpr(allocator);
    const struct_inst_stmt = ast.Statements.StmtNode{
        .Expr = struct_inst_expr.*,
    };
    try statements.append(struct_inst_stmt);

    // Add an expression statement with enum literal
    const enum_literal_expr = try createEnumLiteralExpr(allocator);
    const enum_literal_stmt = ast.Statements.StmtNode{
        .Expr = enum_literal_expr.*,
    };
    try statements.append(enum_literal_stmt);

    // Add an expression statement with switch expression
    const switch_expr = try createSwitchExpressionExpr(allocator);
    const switch_stmt = ast.Statements.StmtNode{
        .Expr = switch_expr.*,
    };
    try statements.append(switch_stmt);

    // Add an expression statement with quantified expression
    const quantified_expr = try createQuantifiedExpr(allocator);
    const quantified_stmt = ast.Statements.StmtNode{
        .Expr = quantified_expr.*,
    };
    try statements.append(quantified_stmt);

    // Add an expression statement with assignment
    const assignment_expr = try createAssignmentExpr(allocator);
    const assignment_stmt = ast.Statements.StmtNode{
        .Expr = assignment_expr.*,
    };
    try statements.append(assignment_stmt);

    // Add an expression statement with compound assignment
    const compound_assignment_expr = try createCompoundAssignmentExpr(allocator);
    const compound_assignment_stmt = ast.Statements.StmtNode{
        .Expr = compound_assignment_expr.*,
    };
    try statements.append(compound_assignment_stmt);

    // Add an expression statement with comptime
    const comptime_expr = try createComptimeExpr(allocator);
    const comptime_stmt = ast.Statements.StmtNode{
        .Expr = comptime_expr.*,
    };
    try statements.append(comptime_stmt);

    // Add an expression statement with old
    const old_expr = try createOldExpr(allocator);
    const old_stmt = ast.Statements.StmtNode{
        .Expr = old_expr.*,
    };
    try statements.append(old_stmt);

    // Add a break statement
    const break_stmt = ast.Statements.StmtNode{
        .Break = .{
            .label = null,
            .value = null,
            .span = .{ .line = 31, .column = 1, .length = 5, .byte_offset = 0 },
        },
    };
    try statements.append(break_stmt);

    // Add a continue statement
    const continue_stmt = ast.Statements.StmtNode{
        .Continue = .{
            .label = null,
            .value = null,
            .span = .{ .line = 32, .column = 1, .length = 8, .byte_offset = 0 },
        },
    };
    try statements.append(continue_stmt);

    // Add an unlock statement
    const unlock_path = try createIdentifierExpr(allocator, "unlock_target");
    const unlock_stmt = ast.Statements.StmtNode{
        .Unlock = .{
            .path = unlock_path.*,
            .span = .{ .line = 33, .column = 1, .length = 7, .byte_offset = 0 },
        },
    };
    try statements.append(unlock_stmt);

    // Add a move statement
    const move_expr = try createIdentifierExpr(allocator, "amount");
    const move_source = try createIdentifierExpr(allocator, "source");
    const move_dest = try createIdentifierExpr(allocator, "dest");
    const move_amount = try createLiteralExpr(allocator);
    const move_stmt = ast.Statements.StmtNode{
        .Move = .{
            .expr = move_expr.*,
            .source = move_source.*,
            .dest = move_dest.*,
            .amount = move_amount.*,
            .span = .{ .line = 34, .column = 1, .length = 20, .byte_offset = 0 },
        },
    };
    try statements.append(move_stmt);

    // Add a labeled block statement
    const labeled_block_content = ast.Statements.BlockNode{
        .statements = &[_]ast.Statements.StmtNode{},
        .span = .{ .line = 35, .column = 5, .length = 5, .byte_offset = 0 },
    };
    const labeled_block_stmt = ast.Statements.StmtNode{
        .LabeledBlock = .{
            .label = "test_label",
            .block = labeled_block_content,
            .span = .{ .line = 35, .column = 1, .length = 25, .byte_offset = 0 },
        },
    };
    try statements.append(labeled_block_stmt);

    // Add a destructuring assignment statement
    const destructuring_fields = try allocator.alloc(ast.Expressions.StructDestructureField, 2);
    destructuring_fields[0] = .{ .name = "field1", .variable = "field1", .span = .{ .line = 36, .column = 5, .length = 6, .byte_offset = 0 } };
    destructuring_fields[1] = .{ .name = "field2", .variable = "field2", .span = .{ .line = 36, .column = 12, .length = 6, .byte_offset = 0 } };

    const destructuring_pattern = ast.Expressions.DestructuringPattern{
        .Struct = destructuring_fields,
    };
    const destructuring_value = try createIdentifierExpr(allocator, "struct_value");
    const destructuring_stmt = ast.Statements.StmtNode{
        .DestructuringAssignment = .{
            .pattern = destructuring_pattern,
            .value = destructuring_value,
            .span = .{ .line = 36, .column = 1, .length = 30, .byte_offset = 0 },
        },
    };
    try statements.append(destructuring_stmt);

    // Add a compound assignment statement
    const compound_assignment_stmt_node = ast.Statements.StmtNode{
        .CompoundAssignment = .{
            .target = try createIdentifierExpr(allocator, "counter"),
            .operator = .PlusEqual,
            .value = try createLiteralExpr(allocator),
            .span = .{ .line = 37, .column = 1, .length = 15, .byte_offset = 0 },
        },
    };
    try statements.append(compound_assignment_stmt_node);

    // Add an expression statement with anonymous struct
    const anonymous_struct_expr = try createAnonymousStructExpr(allocator);
    const anonymous_struct_stmt = ast.Statements.StmtNode{
        .Expr = anonymous_struct_expr.*,
    };
    try statements.append(anonymous_struct_stmt);

    // Add an expression statement with labeled block expression
    const labeled_block_expr = try createLabeledBlockExpr(allocator);
    const labeled_block_expr_stmt = ast.Statements.StmtNode{
        .Expr = labeled_block_expr.*,
    };
    try statements.append(labeled_block_expr_stmt);

    // Add an expression statement with destructuring expression
    const destructuring_expr = try createDestructuringExpr(allocator);
    const destructuring_expr_stmt = ast.Statements.StmtNode{
        .Expr = destructuring_expr.*,
    };
    try statements.append(destructuring_expr_stmt);

    const block = try allocator.create(ast.Statements.BlockNode);
    block.* = .{
        .statements = try statements.toOwnedSlice(),
        .span = .{ .line = 3, .column = 1, .length = 10, .byte_offset = 0 },
    };

    function.* = .{
        .name = "test_function",
        .parameters = &[_]ast.ParameterNode{},
        .return_type_info = null,
        .body = block.*,
        .visibility = .Public,
        .attributes = &[_]u8{},
        .is_inline = false,
        .requires_clauses = &[_]*ast.Expressions.ExprNode{},
        .ensures_clauses = &[_]*ast.Expressions.ExprNode{},
        .span = .{ .line = 3, .column = 1, .length = 30, .byte_offset = 0 },
    };
    return .{ .Function = function.* };
}

fn createStructNode(allocator: std.mem.Allocator) !ast.AstNode {
    const struct_decl = try allocator.create(ast.StructDeclNode);

    // Create struct fields
    const fields = try allocator.alloc(ast.StructField, 2);
    fields[0] = .{
        .name = "field1",
        .type_info = ast.Types.TypeInfo.explicit(.Integer, .u256, .{ .line = 4, .column = 5, .length = 6, .byte_offset = 0 }),
        .span = .{ .line = 4, .column = 5, .length = 6, .byte_offset = 0 },
    };
    fields[1] = .{
        .name = "field2",
        .type_info = ast.Types.TypeInfo.explicit(.Bool, .bool, .{ .line = 4, .column = 12, .length = 6, .byte_offset = 0 }),
        .span = .{ .line = 4, .column = 12, .length = 6, .byte_offset = 0 },
    };

    struct_decl.* = .{
        .name = "TestStruct",
        .fields = fields,
        .span = .{ .line = 4, .column = 1, .length = 20, .byte_offset = 0 },
    };
    return .{ .StructDecl = struct_decl.* };
}

fn createEnumNode(allocator: std.mem.Allocator) !ast.AstNode {
    const enum_decl = try allocator.create(ast.EnumDeclNode);

    // Create enum variants
    const variants = try allocator.alloc(ast.EnumVariant, 2);
    variants[0] = .{
        .name = "Variant1",
        .value = null,
        .resolved_value = null,
        .span = .{ .line = 5, .column = 5, .length = 8, .byte_offset = 0 },
    };
    variants[1] = .{
        .name = "Variant2",
        .value = null,
        .resolved_value = null,
        .span = .{ .line = 5, .column = 14, .length = 8, .byte_offset = 0 },
    };

    enum_decl.* = .{
        .name = "TestEnum",
        .variants = variants,
        .underlying_type_info = null,
        .span = .{ .line = 5, .column = 1, .length = 20, .byte_offset = 0 },
        .has_explicit_values = false,
    };
    return .{ .EnumDecl = enum_decl.* };
}

fn createConstantNode(allocator: std.mem.Allocator) !ast.AstNode {
    const constant = try allocator.create(ast.ConstantNode);
    const literal_expr = try createLiteralExpr(allocator);

    constant.* = .{
        .name = "TEST_CONSTANT",
        .typ = ast.Types.TypeInfo.explicit(.Integer, .u256, .{ .line = 6, .column = 1, .length = 20, .byte_offset = 0 }),
        .value = literal_expr,
        .visibility = .Public,
        .span = .{ .line = 6, .column = 1, .length = 30, .byte_offset = 0 },
    };
    return .{ .Constant = constant.* };
}

fn createVariableDeclNode(allocator: std.mem.Allocator) !ast.AstNode {
    const var_decl = try allocator.create(ast.Statements.VariableDeclNode);
    var_decl.* = .{
        .name = "test_var",
        .type_info = ast.Types.TypeInfo.explicit(.Integer, .u256, .{ .line = 7, .column = 1, .length = 15, .byte_offset = 0 }),
        .value = null,
        .kind = .Let,
        .region = .Memory,
        .locked = false,
        .tuple_names = null,
        .span = .{ .line = 7, .column = 1, .length = 20, .byte_offset = 0 },
    };
    return .{ .VariableDecl = var_decl.* };
}

fn createLogDeclNode(allocator: std.mem.Allocator) !ast.AstNode {
    const log_decl = try allocator.create(ast.LogDeclNode);

    // Create log fields
    const fields = try allocator.alloc(ast.LogField, 2);
    fields[0] = .{
        .name = "event_data",
        .type_info = ast.Types.TypeInfo.explicit(.Integer, .u256, .{ .line = 8, .column = 5, .length = 10, .byte_offset = 0 }),
        .indexed = false,
        .span = .{ .line = 8, .column = 5, .length = 10, .byte_offset = 0 },
    };
    fields[1] = .{
        .name = "user_address",
        .type_info = ast.Types.TypeInfo.explicit(.Address, .address, .{ .line = 8, .column = 16, .length = 12, .byte_offset = 0 }),
        .indexed = true,
        .span = .{ .line = 8, .column = 16, .length = 12, .byte_offset = 0 },
    };

    log_decl.* = .{
        .name = "TestLog",
        .fields = fields,
        .span = .{ .line = 8, .column = 1, .length = 15, .byte_offset = 0 },
    };
    return .{ .LogDecl = log_decl.* };
}

fn createImportNode(allocator: std.mem.Allocator) !ast.AstNode {
    const import = try allocator.create(ast.ImportNode);
    import.* = .{
        .path = "test_module",
        .alias = null,
        .span = .{ .line = 9, .column = 1, .length = 15, .byte_offset = 0 },
    };
    return .{ .Import = import.* };
}

fn createErrorDeclNode(allocator: std.mem.Allocator) !ast.AstNode {
    const error_decl = try allocator.create(ast.Statements.ErrorDeclNode);

    // Create error parameters
    const params = try allocator.alloc(ast.ParameterNode, 1);
    params[0] = .{
        .name = "error_code",
        .type_info = ast.Types.TypeInfo.explicit(.Integer, .u256, .{ .line = 10, .column = 5, .length = 10, .byte_offset = 0 }),
        .is_mutable = false,
        .default_value = null,
        .span = .{ .line = 10, .column = 5, .length = 10, .byte_offset = 0 },
    };

    error_decl.* = .{
        .name = "TestError",
        .parameters = params,
        .span = .{ .line = 10, .column = 1, .length = 20, .byte_offset = 0 },
    };
    return .{ .ErrorDecl = error_decl.* };
}

fn createTryBlockNode(allocator: std.mem.Allocator) !ast.AstNode {
    const try_block = try allocator.create(ast.Statements.TryBlockNode);

    // Create try block content
    const try_block_content = ast.Statements.BlockNode{
        .statements = &[_]ast.Statements.StmtNode{},
        .span = .{ .line = 11, .column = 1, .length = 10, .byte_offset = 0 },
    };

    // Create catch block content
    const catch_block = ast.Statements.CatchBlock{
        .error_variable = null,
        .block = ast.Statements.BlockNode{
            .statements = &[_]ast.Statements.StmtNode{},
            .span = .{ .line = 12, .column = 1, .length = 10, .byte_offset = 0 },
        },
        .span = .{ .line = 12, .column = 1, .length = 10, .byte_offset = 0 },
    };

    try_block.* = .{
        .try_block = try_block_content,
        .catch_block = catch_block,
        .span = .{ .line = 11, .column = 1, .length = 20, .byte_offset = 0 },
    };

    return .{ .TryBlock = try_block.* };
}

fn createLiteralExpr(allocator: std.mem.Allocator) !*ast.Expressions.ExprNode {
    const literal_node = ast.Expressions.LiteralExpr{
        .Integer = .{ .value = "42", .type_info = ast.Types.TypeInfo.explicit(.Integer, .u256, .{ .line = 11, .column = 1, .length = 2, .byte_offset = 0 }), .span = .{ .line = 11, .column = 1, .length = 2, .byte_offset = 0 } },
    };

    const expr_node = try allocator.create(ast.Expressions.ExprNode);
    expr_node.* = .{ .Literal = literal_node };
    return expr_node;
}

fn createBinaryExpr(allocator: std.mem.Allocator) !*ast.Expressions.ExprNode {
    const left = try createIdentifierExpr(allocator, "x");
    const right = try createLiteralExpr(allocator);

    const binary_node = ast.Expressions.BinaryExpr{
        .lhs = left,
        .operator = .EqualEqual,
        .rhs = right,
        .type_info = ast.Types.TypeInfo.explicit(.Bool, .bool, .{ .line = 12, .column = 1, .length = 5, .byte_offset = 0 }),
        .span = .{ .line = 12, .column = 1, .length = 5, .byte_offset = 0 },
    };

    const expr_node = try allocator.create(ast.Expressions.ExprNode);
    expr_node.* = .{ .Binary = binary_node };
    return expr_node;
}

fn createIdentifierExpr(allocator: std.mem.Allocator, name: []const u8) !*ast.Expressions.ExprNode {
    const identifier_node = ast.Expressions.IdentifierExpr{
        .name = name,
        .span = .{ .line = 13, .column = 1, .length = @intCast(name.len), .byte_offset = 0 },
    };

    const expr_node = try allocator.create(ast.Expressions.ExprNode);
    expr_node.* = .{ .Identifier = identifier_node };
    return expr_node;
}

fn createUnaryExpr(allocator: std.mem.Allocator) !*ast.Expressions.ExprNode {
    const operand = try createIdentifierExpr(allocator, "condition");

    const unary_node = ast.Expressions.UnaryExpr{
        .operator = .Bang,
        .operand = operand,
        .type_info = ast.Types.TypeInfo.explicit(.Bool, .bool, .{ .line = 14, .column = 1, .length = 5, .byte_offset = 0 }),
        .span = .{ .line = 14, .column = 1, .length = 5, .byte_offset = 0 },
    };

    const expr_node = try allocator.create(ast.Expressions.ExprNode);
    expr_node.* = .{ .Unary = unary_node };
    return expr_node;
}

fn createCallExpr(allocator: std.mem.Allocator) !*ast.Expressions.ExprNode {
    const callee = try createIdentifierExpr(allocator, "test_function");
    const arg = try createLiteralExpr(allocator);

    const arguments = try allocator.alloc(*ast.Expressions.ExprNode, 1);
    arguments[0] = arg;

    const call_node = ast.Expressions.CallExpr{
        .callee = callee,
        .arguments = arguments,
        .type_info = ast.Types.TypeInfo.explicit(.Integer, .u256, .{ .line = 15, .column = 1, .length = 5, .byte_offset = 0 }),
        .span = .{ .line = 15, .column = 1, .length = 5, .byte_offset = 0 },
    };

    const expr_node = try allocator.create(ast.Expressions.ExprNode);
    expr_node.* = .{ .Call = call_node };
    return expr_node;
}

fn createFieldAccessExpr(allocator: std.mem.Allocator) !*ast.Expressions.ExprNode {
    const target = try createIdentifierExpr(allocator, "struct_instance");

    const field_access_node = ast.Expressions.FieldAccessExpr{
        .target = target,
        .field = "field_name",
        .type_info = ast.Types.TypeInfo.explicit(.Integer, .u256, .{ .line = 16, .column = 1, .length = 5, .byte_offset = 0 }),
        .span = .{ .line = 16, .column = 1, .length = 5, .byte_offset = 0 },
    };

    const expr_node = try allocator.create(ast.Expressions.ExprNode);
    expr_node.* = .{ .FieldAccess = field_access_node };
    return expr_node;
}

fn createIndexExpr(allocator: std.mem.Allocator) !*ast.Expressions.ExprNode {
    const target = try createIdentifierExpr(allocator, "array");
    const index = try createLiteralExpr(allocator);

    const index_node = ast.Expressions.IndexExpr{
        .target = target,
        .index = index,
        .span = .{ .line = 17, .column = 1, .length = 5, .byte_offset = 0 },
    };

    const expr_node = try allocator.create(ast.Expressions.ExprNode);
    expr_node.* = .{ .Index = index_node };
    return expr_node;
}

fn createCastExpr(allocator: std.mem.Allocator) !*ast.Expressions.ExprNode {
    const operand = try createLiteralExpr(allocator);

    const cast_node = ast.Expressions.CastExpr{
        .operand = operand,
        .target_type = ast.Types.TypeInfo.explicit(.Integer, .u256, .{ .line = 18, .column = 1, .length = 5, .byte_offset = 0 }),
        .cast_type = .Unsafe,
        .span = .{ .line = 18, .column = 1, .length = 5, .byte_offset = 0 },
    };

    const expr_node = try allocator.create(ast.Expressions.ExprNode);
    expr_node.* = .{ .Cast = cast_node };
    return expr_node;
}

fn createTupleExpr(allocator: std.mem.Allocator) !*ast.Expressions.ExprNode {
    const elem1 = try createLiteralExpr(allocator);
    const elem2 = try createIdentifierExpr(allocator, "y");

    const elements = try allocator.alloc(*ast.Expressions.ExprNode, 2);
    elements[0] = elem1;
    elements[1] = elem2;

    const tuple_node = ast.Expressions.TupleExpr{
        .elements = elements,
        .span = .{ .line = 19, .column = 1, .length = 5, .byte_offset = 0 },
    };

    const expr_node = try allocator.create(ast.Expressions.ExprNode);
    expr_node.* = .{ .Tuple = tuple_node };
    return expr_node;
}

fn createRangeExpr(allocator: std.mem.Allocator) !*ast.Expressions.ExprNode {
    const start = try createLiteralExpr(allocator);
    const end = try createLiteralExpr(allocator);

    const range_node = ast.Expressions.RangeExpr{
        .start = start,
        .end = end,
        .inclusive = true,
        .type_info = ast.Types.TypeInfo.explicit(.Integer, .u256, .{ .line = 20, .column = 1, .length = 5, .byte_offset = 0 }),
        .span = .{ .line = 20, .column = 1, .length = 5, .byte_offset = 0 },
    };

    const expr_node = try allocator.create(ast.Expressions.ExprNode);
    expr_node.* = .{ .Range = range_node };
    return expr_node;
}

fn createArrayLiteralExpr(allocator: std.mem.Allocator) !*ast.Expressions.ExprNode {
    const elem1 = try createLiteralExpr(allocator);
    const elem2 = try createLiteralExpr(allocator);

    const elements = try allocator.alloc(*ast.Expressions.ExprNode, 2);
    elements[0] = elem1;
    elements[1] = elem2;

    const array_node = ast.Literals.Array{
        .elements = elements,
        .element_type = null,
        .span = .{ .line = 21, .column = 1, .length = 5, .byte_offset = 0 },
    };

    const expr_node = try allocator.create(ast.Expressions.ExprNode);
    expr_node.* = .{ .ArrayLiteral = array_node };
    return expr_node;
}

fn createTryExpr(allocator: std.mem.Allocator) !*ast.Expressions.ExprNode {
    const expr = try createLiteralExpr(allocator);

    const try_node = ast.Expressions.TryExpr{
        .expr = expr,
        .span = .{ .line = 22, .column = 1, .length = 5, .byte_offset = 0 },
    };

    const expr_node = try allocator.create(ast.Expressions.ExprNode);
    expr_node.* = .{ .Try = try_node };
    return expr_node;
}

fn createErrorReturnExpr(allocator: std.mem.Allocator) !*ast.Expressions.ExprNode {
    const error_return_node = ast.Expressions.ErrorReturnExpr{
        .error_name = "TestError",
        .span = .{ .line = 23, .column = 1, .length = 5, .byte_offset = 0 },
    };

    const expr_node = try allocator.create(ast.Expressions.ExprNode);
    expr_node.* = .{ .ErrorReturn = error_return_node };
    return expr_node;
}

fn createErrorCastExpr(allocator: std.mem.Allocator) !*ast.Expressions.ExprNode {
    const operand = try createLiteralExpr(allocator);

    const error_cast_node = ast.Expressions.ErrorCastExpr{
        .operand = operand,
        .target_type = ast.Types.TypeInfo.explicit(.Integer, .u256, .{ .line = 24, .column = 1, .length = 5, .byte_offset = 0 }),
        .span = .{ .line = 24, .column = 1, .length = 5, .byte_offset = 0 },
    };

    const expr_node = try allocator.create(ast.Expressions.ExprNode);
    expr_node.* = .{ .ErrorCast = error_cast_node };
    return expr_node;
}

fn createShiftExpr(allocator: std.mem.Allocator) !*ast.Expressions.ExprNode {
    const mapping = try createIdentifierExpr(allocator, "balances");
    const source = try createIdentifierExpr(allocator, "sender");
    const dest = try createIdentifierExpr(allocator, "recipient");
    const amount = try createLiteralExpr(allocator);

    const shift_node = ast.Expressions.ShiftExpr{
        .mapping = mapping,
        .source = source,
        .dest = dest,
        .amount = amount,
        .span = .{ .line = 25, .column = 1, .length = 5, .byte_offset = 0 },
    };

    const expr_node = try allocator.create(ast.Expressions.ExprNode);
    expr_node.* = .{ .Shift = shift_node };
    return expr_node;
}

fn createStructInstantiationExpr(allocator: std.mem.Allocator) !*ast.Expressions.ExprNode {
    const struct_name = try createIdentifierExpr(allocator, "TestStruct");

    const struct_inst_node = ast.Expressions.StructInstantiationExpr{
        .struct_name = struct_name,
        .fields = &[_]ast.Expressions.StructInstantiationField{},
        .span = .{ .line = 26, .column = 1, .length = 5, .byte_offset = 0 },
    };

    const expr_node = try allocator.create(ast.Expressions.ExprNode);
    expr_node.* = .{ .StructInstantiation = struct_inst_node };
    return expr_node;
}

fn createEnumLiteralExpr(allocator: std.mem.Allocator) !*ast.Expressions.ExprNode {
    const enum_literal_node = ast.Expressions.EnumLiteralExpr{
        .enum_name = "TestEnum",
        .variant_name = "Variant1",
        .span = .{ .line = 27, .column = 1, .length = 5, .byte_offset = 0 },
    };

    const expr_node = try allocator.create(ast.Expressions.ExprNode);
    expr_node.* = .{ .EnumLiteral = enum_literal_node };
    return expr_node;
}

fn createSwitchExpressionExpr(allocator: std.mem.Allocator) !*ast.Expressions.ExprNode {
    const condition = try createIdentifierExpr(allocator, "value");

    const switch_node = ast.Switch.ExprNode{
        .condition = condition,
        .cases = &[_]ast.Expressions.SwitchCase{},
        .default_case = null,
        .span = .{ .line = 28, .column = 1, .length = 5, .byte_offset = 0 },
    };

    const expr_node = try allocator.create(ast.Expressions.ExprNode);
    expr_node.* = .{ .SwitchExpression = switch_node };
    return expr_node;
}

fn createQuantifiedExpr(allocator: std.mem.Allocator) !*ast.Expressions.ExprNode {
    const body = try createLiteralExpr(allocator);

    const quantified_node = ast.Expressions.QuantifiedExpr{
        .quantifier = .Forall,
        .variable = "x",
        .variable_type = ast.Types.TypeInfo.explicit(.Integer, .u256, .{ .line = 29, .column = 1, .length = 5, .byte_offset = 0 }),
        .condition = null,
        .body = body,
        .span = .{ .line = 29, .column = 1, .length = 5, .byte_offset = 0 },
    };

    const expr_node = try allocator.create(ast.Expressions.ExprNode);
    expr_node.* = .{ .Quantified = quantified_node };
    return expr_node;
}

fn createAssignmentExpr(allocator: std.mem.Allocator) !*ast.Expressions.ExprNode {
    const target = try createIdentifierExpr(allocator, "x");
    const value = try createLiteralExpr(allocator);

    const assignment_node = ast.Expressions.AssignmentExpr{
        .target = target,
        .value = value,
        .span = .{ .line = 34, .column = 1, .length = 5, .byte_offset = 0 },
    };

    const expr_node = try allocator.create(ast.Expressions.ExprNode);
    expr_node.* = .{ .Assignment = assignment_node };
    return expr_node;
}

fn createCompoundAssignmentExpr(allocator: std.mem.Allocator) !*ast.Expressions.ExprNode {
    const target = try createIdentifierExpr(allocator, "x");
    const value = try createLiteralExpr(allocator);

    const compound_assignment_node = ast.Expressions.CompoundAssignmentExpr{
        .target = target,
        .operator = .PlusEqual,
        .value = value,
        .span = .{ .line = 35, .column = 1, .length = 5, .byte_offset = 0 },
    };

    const expr_node = try allocator.create(ast.Expressions.ExprNode);
    expr_node.* = .{ .CompoundAssignment = compound_assignment_node };
    return expr_node;
}

fn createComptimeExpr(allocator: std.mem.Allocator) !*ast.Expressions.ExprNode {
    const block = ast.Statements.BlockNode{
        .statements = &[_]ast.Statements.StmtNode{},
        .span = .{ .line = 36, .column = 1, .length = 5, .byte_offset = 0 },
    };

    const comptime_node = ast.Expressions.ComptimeExpr{
        .block = block,
        .span = .{ .line = 36, .column = 1, .length = 5, .byte_offset = 0 },
    };

    const expr_node = try allocator.create(ast.Expressions.ExprNode);
    expr_node.* = .{ .Comptime = comptime_node };
    return expr_node;
}

fn createOldExpr(allocator: std.mem.Allocator) !*ast.Expressions.ExprNode {
    const old_node = ast.Expressions.OldExpr{
        .expr = try createIdentifierExpr(allocator, "x"),
        .span = .{ .line = 37, .column = 1, .length = 5, .byte_offset = 0 },
    };

    const expr_node = try allocator.create(ast.Expressions.ExprNode);
    expr_node.* = .{ .Old = old_node };
    return expr_node;
}

fn createAnonymousStructExpr(allocator: std.mem.Allocator) !*ast.Expressions.ExprNode {
    const fields = try allocator.alloc(ast.Expressions.AnonymousStructField, 2);
    fields[0] = .{ .name = "field1", .value = try createLiteralExpr(allocator), .span = .{ .line = 38, .column = 5, .length = 6, .byte_offset = 0 } };
    fields[1] = .{ .name = "field2", .value = try createIdentifierExpr(allocator, "value"), .span = .{ .line = 38, .column = 12, .length = 6, .byte_offset = 0 } };

    const anonymous_struct_node = ast.Expressions.AnonymousStructExpr{
        .fields = fields,
        .span = .{ .line = 38, .column = 1, .length = 5, .byte_offset = 0 },
    };

    const expr_node = try allocator.create(ast.Expressions.ExprNode);
    expr_node.* = .{ .AnonymousStruct = anonymous_struct_node };
    return expr_node;
}

fn createLabeledBlockExpr(allocator: std.mem.Allocator) !*ast.Expressions.ExprNode {
    const block_content = ast.Statements.BlockNode{
        .statements = &[_]ast.Statements.StmtNode{},
        .span = .{ .line = 39, .column = 5, .length = 5, .byte_offset = 0 },
    };

    const labeled_block_node = ast.Expressions.LabeledBlockExpr{
        .label = "test_label",
        .block = block_content,
        .span = .{ .line = 39, .column = 1, .length = 25, .byte_offset = 0 },
    };

    const expr_node = try allocator.create(ast.Expressions.ExprNode);
    expr_node.* = .{ .LabeledBlock = labeled_block_node };
    return expr_node;
}

fn createDestructuringExpr(allocator: std.mem.Allocator) !*ast.Expressions.ExprNode {
    const destructuring_fields = try allocator.alloc(ast.Expressions.StructDestructureField, 2);
    destructuring_fields[0] = .{ .name = "field1", .variable = "field1", .span = .{ .line = 40, .column = 1, .length = 6, .byte_offset = 0 } };
    destructuring_fields[1] = .{ .name = "field2", .variable = "field2", .span = .{ .line = 40, .column = 8, .length = 6, .byte_offset = 0 } };

    const pattern = ast.Expressions.DestructuringPattern{
        .Struct = destructuring_fields,
    };

    const destructuring_node = ast.Expressions.DestructuringExpr{
        .pattern = pattern,
        .value = try createIdentifierExpr(allocator, "struct_value"),
        .span = .{ .line = 40, .column = 1, .length = 20, .byte_offset = 0 },
    };

    const expr_node = try allocator.create(ast.Expressions.ExprNode);
    expr_node.* = .{ .Destructuring = destructuring_node };
    return expr_node;
}

// Helper function to create StructDestructureField
fn createStructDestructureField(_: std.mem.Allocator, name: []const u8) ast.Expressions.StructDestructureField {
    return .{
        .name = name,
        .span = .{ .line = 40, .column = 1, .length = @intCast(name.len), .byte_offset = 0 },
    };
}

// Helper function to create AnonymousStructField
fn createAnonymousStructField(_: std.mem.Allocator, name: []const u8, value: *ast.Expressions.ExprNode) ast.Expressions.AnonymousStructField {
    return .{
        .name = name,
        .value = value,
        .span = .{ .line = 41, .column = 1, .length = @intCast(name.len), .byte_offset = 0 },
    };
}
