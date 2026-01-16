// ============================================================================
// AST Statements Tests
// ============================================================================
// Tests for statement node creation, structure, and source span preservation
// ============================================================================

const std = @import("std");
const testing = std.testing;
const ora_root = @import("ora_root");
const ast = ora_root.ast;
const ast_arena = ora_root.ast_arena;
const ast_builder = ora_root.ast_builder;

// ============================================================================
// Statement Node Creation Tests
// ============================================================================

test "statements: return statement without value" {
    const allocator = testing.allocator;
    var arena = ast_arena.AstArena.init(allocator);
    defer arena.deinit();

    var builder = ast_builder.AstBuilder.init(&arena);
    defer builder.deinit();

    const span = ast.SourceSpan{ .line = 1, .column = 1, .length = 6, .byte_offset = 0 };
    const stmt_builder = builder.stmt();
    const stmt = try stmt_builder.returnStmt(null, span);

    try testing.expect(stmt == .Return);
    try testing.expect(stmt.Return.value == null);
    try testing.expectEqual(span.line, stmt.Return.span.line);
    try testing.expectEqual(span.column, stmt.Return.span.column);
}

test "statements: return statement with value" {
    const allocator = testing.allocator;
    var arena = ast_arena.AstArena.init(allocator);
    defer arena.deinit();

    var builder = ast_builder.AstBuilder.init(&arena);
    defer builder.deinit();

    const span_expr = ast.SourceSpan{ .line = 1, .column = 8, .length = 2, .byte_offset = 7 };
    const span_return = ast.SourceSpan{ .line = 1, .column = 1, .length = 10, .byte_offset = 0 };

    const value = try builder.integerLiteral("42", span_expr);
    const stmt_builder = builder.stmt();
    const stmt = try stmt_builder.returnStmt(value, span_return);

    try testing.expect(stmt == .Return);
    try testing.expect(stmt.Return.value != null);
    if (stmt.Return.value) |val| {
        try testing.expect(val == .Literal);
    }
    try testing.expectEqual(span_return.line, stmt.Return.span.line);
}

test "statements: empty block creation" {
    const allocator = testing.allocator;
    var arena = ast_arena.AstArena.init(allocator);
    defer arena.deinit();

    var builder = ast_builder.AstBuilder.init(&arena);
    defer builder.deinit();

    const span = ast.SourceSpan{ .line = 1, .column = 1, .length = 2, .byte_offset = 0 };
    var empty_statements = [_]ast.Statements.StmtNode{};
    const block = try builder.block(empty_statements[0..], span);

    try testing.expect(block.statements.len == 0);
    try testing.expectEqual(span.line, block.span.line);
    try testing.expectEqual(span.column, block.span.column);
}

test "statements: block with statements" {
    const allocator = testing.allocator;
    var arena = ast_arena.AstArena.init(allocator);
    defer arena.deinit();

    var builder = ast_builder.AstBuilder.init(&arena);
    defer builder.deinit();

    const span1 = ast.SourceSpan{ .line = 1, .column = 1, .length = 6, .byte_offset = 0 };
    const span2 = ast.SourceSpan{ .line = 2, .column = 1, .length = 6, .byte_offset = 10 };
    const span_block = ast.SourceSpan{ .line = 1, .column = 1, .length = 20, .byte_offset = 0 };

    const stmt_builder = builder.stmt();
    const stmt1 = try stmt_builder.returnStmt(null, span1);
    const stmt2 = try stmt_builder.returnStmt(null, span2);

    var statements_array = [_]ast.Statements.StmtNode{ stmt1, stmt2 };
    const block = try builder.block(statements_array[0..], span_block);

    try testing.expect(block.statements.len == 2);
    try testing.expectEqual(span_block.line, block.span.line);
}

test "statements: if statement without else" {
    const allocator = testing.allocator;
    var arena = ast_arena.AstArena.init(allocator);
    defer arena.deinit();

    var builder = ast_builder.AstBuilder.init(&arena);
    defer builder.deinit();

    const span_cond = ast.SourceSpan{ .line = 1, .column = 5, .length = 1, .byte_offset = 4 };
    const span_stmt = ast.SourceSpan{ .line = 1, .column = 9, .length = 6, .byte_offset = 8 };
    const span_block = ast.SourceSpan{ .line = 1, .column = 9, .length = 8, .byte_offset = 8 };
    const span_if = ast.SourceSpan{ .line = 1, .column = 1, .length = 18, .byte_offset = 0 };

    const condition = try builder.identifier("x", span_cond);
    const stmt_builder = builder.stmt();
    const return_stmt = try stmt_builder.returnStmt(null, span_stmt);
    var then_statements = [_]ast.Statements.StmtNode{return_stmt};
    const then_block = try builder.block(then_statements[0..], span_block);
    const stmt = try stmt_builder.ifStmt(condition, then_block, null, span_if);

    try testing.expect(stmt == .If);
    try testing.expect(stmt.If.condition == .Identifier);
    try testing.expect(stmt.If.else_branch == null);
    try testing.expectEqual(span_if.line, stmt.If.span.line);
}

test "statements: if statement with else" {
    const allocator = testing.allocator;
    var arena = ast_arena.AstArena.init(allocator);
    defer arena.deinit();

    var builder = ast_builder.AstBuilder.init(&arena);
    defer builder.deinit();

    const span_cond = ast.SourceSpan{ .line = 1, .column = 5, .length = 1, .byte_offset = 4 };
    const span_stmt1 = ast.SourceSpan{ .line = 1, .column = 9, .length = 6, .byte_offset = 8 };
    const span_stmt2 = ast.SourceSpan{ .line = 1, .column = 16, .length = 6, .byte_offset = 15 };
    const span_then = ast.SourceSpan{ .line = 1, .column = 9, .length = 8, .byte_offset = 8 };
    const span_else = ast.SourceSpan{ .line = 1, .column = 16, .length = 8, .byte_offset = 15 };
    const span_if = ast.SourceSpan{ .line = 1, .column = 1, .length = 25, .byte_offset = 0 };

    const condition = try builder.identifier("x", span_cond);
    const stmt_builder = builder.stmt();
    const return_stmt1 = try stmt_builder.returnStmt(null, span_stmt1);
    const return_stmt2 = try stmt_builder.returnStmt(null, span_stmt2);
    var then_statements = [_]ast.Statements.StmtNode{return_stmt1};
    var else_statements = [_]ast.Statements.StmtNode{return_stmt2};
    const then_block = try builder.block(then_statements[0..], span_then);
    const else_block = try builder.block(else_statements[0..], span_else);
    const stmt = try stmt_builder.ifStmt(condition, then_block, else_block, span_if);

    try testing.expect(stmt == .If);
    try testing.expect(stmt.If.else_branch != null);
    try testing.expectEqual(span_if.line, stmt.If.span.line);
}

test "statements: while loop creation" {
    const allocator = testing.allocator;
    var arena = ast_arena.AstArena.init(allocator);
    defer arena.deinit();

    var builder = ast_builder.AstBuilder.init(&arena);
    defer builder.deinit();

    const span_cond = ast.SourceSpan{ .line = 1, .column = 8, .length = 1, .byte_offset = 7 };
    const span_body = ast.SourceSpan{ .line = 1, .column = 12, .length = 2, .byte_offset = 11 };
    const span_while = ast.SourceSpan{ .line = 1, .column = 1, .length = 15, .byte_offset = 0 };

    const condition = try builder.identifier("x", span_cond);
    const stmt_builder = builder.stmt();
    const return_stmt = try stmt_builder.returnStmt(null, span_body);
    var body_statements = [_]ast.Statements.StmtNode{return_stmt};
    const body = try builder.block(body_statements[0..], span_body);
    const stmt = try stmt_builder.whileStmt(condition, body, span_while);

    try testing.expect(stmt == .While);
    try testing.expect(stmt.While.condition == .Identifier);
    try testing.expectEqual(span_while.line, stmt.While.span.line);
}

test "statements: try/catch block creation" {
    const allocator = testing.allocator;
    var arena = ast_arena.AstArena.init(allocator);
    defer arena.deinit();

    var builder = ast_builder.AstBuilder.init(&arena);
    defer builder.deinit();

    const span_try = ast.SourceSpan{ .line = 1, .column = 1, .length = 10, .byte_offset = 0 };
    const span_block = ast.SourceSpan{ .line = 1, .column = 5, .length = 2, .byte_offset = 4 };

    var empty_statements = [_]ast.Statements.StmtNode{};
    const try_block = try builder.block(empty_statements[0..], span_block);
    const catch_block = ast.Statements.CatchBlock{
        .error_variable = "e",
        .block = try builder.block(empty_statements[0..], span_block),
        .span = span_block,
    };

    const stmt = ast.Statements.StmtNode{ .TryBlock = .{
        .try_block = try_block,
        .catch_block = catch_block,
        .span = span_try,
    } };

    try testing.expect(stmt == .TryBlock);
    try testing.expect(stmt.TryBlock.catch_block != null);
    try testing.expectEqual(span_try.line, stmt.TryBlock.span.line);
}

test "statements: switch statement creation" {
    const allocator = testing.allocator;
    var arena = ast_arena.AstArena.init(allocator);
    defer arena.deinit();

    var builder = ast_builder.AstBuilder.init(&arena);
    defer builder.deinit();

    const span_switch = ast.SourceSpan{ .line = 1, .column = 1, .length = 20, .byte_offset = 0 };
    const span_cond = ast.SourceSpan{ .line = 1, .column = 8, .length = 1, .byte_offset = 7 };
    const span_case = ast.SourceSpan{ .line = 1, .column = 12, .length = 1, .byte_offset = 11 };
    const span_block = ast.SourceSpan{ .line = 1, .column = 14, .length = 2, .byte_offset = 13 };

    const condition = try builder.identifier("x", span_cond);
    var empty_statements = [_]ast.Statements.StmtNode{};
    const body_block = try builder.block(empty_statements[0..], span_block);
    const case_value = ast.Expressions.LiteralExpr{
        .Integer = .{
            .value = "0",
            .type_info = ast.Types.CommonTypes.unknown_integer(),
            .span = span_case,
        },
    };
    const cases = try arena.allocator().alloc(ast.Expressions.SwitchCase, 1);
    cases[0] = .{
        .pattern = .{ .Literal = .{ .value = case_value, .span = span_case } },
        .body = .{ .Block = body_block },
        .span = span_case,
    };

    const stmt = ast.Statements.StmtNode{ .Switch = .{
        .condition = condition.*,
        .cases = cases,
        .default_case = null,
        .span = span_switch,
    } };

    try testing.expect(stmt == .Switch);
    try testing.expectEqual(@as(usize, 1), stmt.Switch.cases.len);
    try testing.expectEqual(span_switch.line, stmt.Switch.span.line);
}

// ============================================================================
// Source Span Preservation Tests
// ============================================================================

test "statements: source span preserved in return statement" {
    const allocator = testing.allocator;
    var arena = ast_arena.AstArena.init(allocator);
    defer arena.deinit();

    var builder = ast_builder.AstBuilder.init(&arena);
    defer builder.deinit();

    const span = ast.SourceSpan{ .line = 10, .column = 5, .length = 6, .byte_offset = 200 };
    const stmt_builder = builder.stmt();
    const stmt = try stmt_builder.returnStmt(null, span);

    try testing.expect(stmt == .Return);
    try testing.expectEqual(@as(u32, 10), stmt.Return.span.line);
    try testing.expectEqual(@as(u32, 5), stmt.Return.span.column);
    try testing.expectEqual(@as(u32, 6), stmt.Return.span.length);
    try testing.expectEqual(@as(u32, 200), stmt.Return.span.byte_offset);
}

test "statements: source span preserved in block" {
    const allocator = testing.allocator;
    var arena = ast_arena.AstArena.init(allocator);
    defer arena.deinit();

    var builder = ast_builder.AstBuilder.init(&arena);
    defer builder.deinit();

    const span = ast.SourceSpan{ .line = 5, .column = 3, .length = 10, .byte_offset = 50 };
    var empty_statements = [_]ast.Statements.StmtNode{};
    const block = try builder.block(empty_statements[0..], span);

    try testing.expectEqual(@as(u32, 5), block.span.line);
    try testing.expectEqual(@as(u32, 3), block.span.column);
    try testing.expectEqual(@as(u32, 10), block.span.length);
}

// ============================================================================
// Statement Structure Tests
// ============================================================================

test "statements: if statement structure" {
    const allocator = testing.allocator;
    var arena = ast_arena.AstArena.init(allocator);
    defer arena.deinit();

    var builder = ast_builder.AstBuilder.init(&arena);
    defer builder.deinit();

    const span_cond = ast.SourceSpan{ .line = 1, .column = 5, .length = 1, .byte_offset = 4 };
    const span_then = ast.SourceSpan{ .line = 1, .column = 9, .length = 2, .byte_offset = 8 };
    const span_if = ast.SourceSpan{ .line = 1, .column = 1, .length = 10, .byte_offset = 0 };

    const condition = try builder.boolLiteral(true, span_cond);
    const stmt_builder = builder.stmt();
    const return_stmt = try stmt_builder.returnStmt(null, span_then);
    var then_statements = [_]ast.Statements.StmtNode{return_stmt};
    const then_block = try builder.block(then_statements[0..], span_then);
    const stmt = try stmt_builder.ifStmt(condition, then_block, null, span_if);

    try testing.expect(stmt == .If);
    // check condition structure
    try testing.expect(stmt.If.condition == .Literal);
    if (stmt.If.condition == .Literal and stmt.If.condition.Literal == .Bool) {
        try testing.expect(stmt.If.condition.Literal.Bool.value == true);
    }

    // check then branch structure
    try testing.expect(stmt.If.then_branch.statements.len == 1);
}

test "statements: while loop structure" {
    const allocator = testing.allocator;
    var arena = ast_arena.AstArena.init(allocator);
    defer arena.deinit();

    var builder = ast_builder.AstBuilder.init(&arena);
    defer builder.deinit();

    const span_cond = ast.SourceSpan{ .line = 1, .column = 8, .length = 1, .byte_offset = 7 };
    const span_body = ast.SourceSpan{ .line = 1, .column = 12, .length = 2, .byte_offset = 11 };
    const span_while = ast.SourceSpan{ .line = 1, .column = 1, .length = 15, .byte_offset = 0 };

    const condition = try builder.boolLiteral(true, span_cond);
    const stmt_builder = builder.stmt();
    const return_stmt = try stmt_builder.returnStmt(null, span_body);
    var body_statements = [_]ast.Statements.StmtNode{return_stmt};
    const body = try builder.block(body_statements[0..], span_body);
    const stmt = try stmt_builder.whileStmt(condition, body, span_while);

    try testing.expect(stmt == .While);
    // check condition structure
    try testing.expect(stmt.While.condition == .Literal);

    // check body structure
    try testing.expect(stmt.While.body.statements.len == 1);
}

// ============================================================================
// Expression Statement Tests
// ============================================================================

test "statements: expression statement creation" {
    const allocator = testing.allocator;
    var arena = ast_arena.AstArena.init(allocator);
    defer arena.deinit();

    var builder = ast_builder.AstBuilder.init(&arena);
    defer builder.deinit();

    const span = ast.SourceSpan{ .line = 1, .column = 1, .length = 1, .byte_offset = 0 };
    const expr = try builder.identifier("x", span);
    const stmt_builder = builder.stmt();
    const stmt = stmt_builder.exprStmt(expr);

    try testing.expect(stmt == .Expr);
    try testing.expect(stmt.Expr == .Identifier);
    try testing.expectEqual(span.line, stmt.Expr.Identifier.span.line);
}
