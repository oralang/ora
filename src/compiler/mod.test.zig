const std = @import("std");
const testing = std.testing;
const ora_root = @import("ora_root");
const compiler = ora_root.compiler;

fn compileText(source_text: []const u8) !compiler.driver.Compilation {
    return compiler.compileSource(testing.allocator, "test.ora", source_text);
}

fn renderHirTextForSource(source_text: []const u8) ![]u8 {
    var compilation = try compileText(source_text);
    defer compilation.deinit();

    const hir_result = try compilation.db.lowerToHir(compilation.root_module_id);
    return hir_result.renderText(testing.allocator);
}

fn firstChildNodeOfKind(node: compiler.SyntaxNode, kind: compiler.syntax.SyntaxKind) ?compiler.SyntaxNode {
    var it = node.children();
    while (it.next()) |child| {
        switch (child) {
            .node => |child_node| if (child_node.kind() == kind) return child_node,
            .token => {},
        }
    }
    return null;
}

fn nthChildNodeOfKind(node: compiler.SyntaxNode, kind: compiler.syntax.SyntaxKind, ordinal: usize) ?compiler.SyntaxNode {
    var seen: usize = 0;
    var it = node.children();
    while (it.next()) |child| {
        switch (child) {
            .node => |child_node| {
                if (child_node.kind() != kind) continue;
                if (seen == ordinal) return child_node;
                seen += 1;
            },
            .token => {},
        }
    }
    return null;
}

fn containsNodeOfKind(node: compiler.SyntaxNode, kind: compiler.syntax.SyntaxKind) bool {
    if (node.kind() == kind) return true;

    var it = node.children();
    while (it.next()) |child| {
        switch (child) {
            .node => |child_node| if (containsNodeOfKind(child_node, kind)) return true,
            .token => {},
        }
    }
    return false;
}

fn findVariablePatternByName(ast_file: *const compiler.ast.AstFile, statements: []const compiler.ast.StmtId, name: []const u8) ?compiler.ast.PatternId {
    for (statements) |statement_id| {
        const statement = ast_file.statement(statement_id).*;
        if (statement != .VariableDecl) continue;
        const pattern_id = statement.VariableDecl.pattern;
        const pattern = ast_file.pattern(pattern_id).*;
        if (pattern != .Name) continue;
        if (std.mem.eql(u8, pattern.Name.name, name)) return pattern_id;
    }
    return null;
}

fn nthDescendantNodeOfKind(node: compiler.SyntaxNode, kind: compiler.syntax.SyntaxKind, ordinal: usize) ?compiler.SyntaxNode {
    var remaining = ordinal;
    return nthDescendantNodeOfKindInner(node, kind, &remaining);
}

fn nthDescendantNodeOfKindInner(node: compiler.SyntaxNode, kind: compiler.syntax.SyntaxKind, remaining: *usize) ?compiler.SyntaxNode {
    if (node.kind() == kind) {
        if (remaining.* == 0) return node;
        remaining.* -= 1;
    }

    var it = node.children();
    while (it.next()) |child| {
        switch (child) {
            .node => |child_node| {
                if (nthDescendantNodeOfKindInner(child_node, kind, remaining)) |found| return found;
            },
            .token => {},
        }
    }
    return null;
}

test "compiler syntax preserves source and syntax pointers resolve" {
    const source_text =
        \\// leading comment
        \\contract Test {
        \\    pub fn run() -> u256 {
        \\        return 42;
        \\    }
        \\}
    ;

    var compilation = try compileText(source_text);
    defer compilation.deinit();

    const module = compilation.db.sources.module(compilation.root_module_id);
    const tree = try compilation.db.syntaxTree(module.file_id);
    const rebuilt = try tree.reconstructSource(testing.allocator);
    defer testing.allocator.free(rebuilt);

    try testing.expectEqualStrings(source_text, rebuilt);

    const root = compiler.syntax.rootNode(tree);
    const ptr = root.ptr();
    try testing.expect(ptr.resolve(tree) != null);

    const contract = firstChildNodeOfKind(root, .ContractItem);
    try testing.expect(contract != null);

    const function = firstChildNodeOfKind(contract.?, .FunctionItem);
    try testing.expect(function != null);

    const body = firstChildNodeOfKind(function.?, .Body);
    try testing.expect(body != null);
}

test "compiler syntax parses statement-level bodies" {
    const source_text =
        \\pub fn run(values: u256) -> u256 {
        \\    if (values > 0) {
        \\        return values;
        \\    } else {
        \\        while (values > 1) {
        \\            break;
        \\        }
        \\    }
        \\
        \\    switch (values) {
        \\        0 => return 0;
        \\        else => {
        \\            continue;
        \\        }
        \\    }
        \\
        \\    try {
        \\        havoc values;
        \\    } catch (err) {
        \\        assert(values >= 0, "non-negative");
        \\    }
        \\}
    ;

    var compilation = try compileText(source_text);
    defer compilation.deinit();

    const module = compilation.db.sources.module(compilation.root_module_id);
    const tree = try compilation.db.syntaxTree(module.file_id);
    const root = compiler.syntax.rootNode(tree);
    const function = firstChildNodeOfKind(root, .FunctionItem);
    try testing.expect(function != null);

    const body = firstChildNodeOfKind(function.?, .Body);
    try testing.expect(body != null);
    try testing.expect(nthChildNodeOfKind(body.?, .IfStmt, 0) != null);
    try testing.expect(nthChildNodeOfKind(body.?, .SwitchStmt, 0) != null);
    try testing.expect(nthChildNodeOfKind(body.?, .TryStmt, 0) != null);

    const if_stmt = nthChildNodeOfKind(body.?, .IfStmt, 0).?;
    const then_body = nthChildNodeOfKind(if_stmt, .Body, 0);
    const else_body = nthChildNodeOfKind(if_stmt, .Body, 1);
    try testing.expect(then_body != null);
    try testing.expect(else_body != null);
    try testing.expect(firstChildNodeOfKind(then_body.?, .ReturnStmt) != null);
    try testing.expect(firstChildNodeOfKind(else_body.?, .WhileStmt) != null);

    const switch_stmt = nthChildNodeOfKind(body.?, .SwitchStmt, 0).?;
    const first_arm = nthChildNodeOfKind(switch_stmt, .SwitchArm, 0);
    const second_arm = nthChildNodeOfKind(switch_stmt, .SwitchArm, 1);
    try testing.expect(first_arm != null);
    try testing.expect(second_arm != null);
    try testing.expect(containsNodeOfKind(first_arm.?, .ExprStmt));

    const try_stmt = nthChildNodeOfKind(body.?, .TryStmt, 0).?;
    const try_body = nthChildNodeOfKind(try_stmt, .Body, 0);
    const catch_clause = nthChildNodeOfKind(try_stmt, .CatchClause, 0);
    const catch_body = if (catch_clause) |clause| nthChildNodeOfKind(clause, .Body, 0) else null;
    try testing.expect(try_body != null);
    try testing.expect(catch_clause != null);
    try testing.expect(catch_body != null);
    try testing.expect(firstChildNodeOfKind(try_body.?, .HavocStmt) != null);
    try testing.expect(firstChildNodeOfKind(catch_body.?, .AssertStmt) != null);
}

test "compiler syntax bounds spec clauses loop invariants and item members" {
    const source_text =
        \\struct Pair {
        \\    left: u256,
        \\    right: u256;
        \\}
        \\
        \\bitfield Flags: u256 {
        \\    enabled: bool @bits(0..1);
        \\}
        \\
        \\enum Mode {
        \\    Off,
        \\    On,
        \\}
        \\
        \\pub fn run(values: u256) -> u256
        \\    requires values >= 0
        \\    ensures result >= 0
        \\{
        \\    while (values > 0)
        \\        invariant values >= 0;
        \\    {
        \\        break;
        \\    }
        \\
        \\    for (values) |value|
        \\        invariant value >= 0;
        \\    {
        \\        continue;
        \\    }
        \\
        \\    return values;
        \\}
    ;

    var compilation = try compileText(source_text);
    defer compilation.deinit();

    const module = compilation.db.sources.module(compilation.root_module_id);
    const tree = try compilation.db.syntaxTree(module.file_id);
    const root = compiler.syntax.rootNode(tree);

    const struct_item = nthChildNodeOfKind(root, .StructItem, 0);
    const bitfield_item = nthChildNodeOfKind(root, .BitfieldItem, 0);
    const enum_item = nthChildNodeOfKind(root, .EnumItem, 0);
    const function = nthChildNodeOfKind(root, .FunctionItem, 0);
    try testing.expect(struct_item != null);
    try testing.expect(bitfield_item != null);
    try testing.expect(enum_item != null);
    try testing.expect(function != null);

    try testing.expect(nthChildNodeOfKind(struct_item.?, .StructField, 0) != null);
    try testing.expect(nthChildNodeOfKind(struct_item.?, .StructField, 1) != null);
    try testing.expect(nthChildNodeOfKind(bitfield_item.?, .BitfieldField, 0) != null);
    try testing.expect(nthChildNodeOfKind(enum_item.?, .EnumVariant, 0) != null);
    try testing.expect(nthChildNodeOfKind(enum_item.?, .EnumVariant, 1) != null);

    const requires_clause = nthChildNodeOfKind(function.?, .SpecClause, 0);
    const ensures_clause = nthChildNodeOfKind(function.?, .SpecClause, 1);
    const body = nthChildNodeOfKind(function.?, .Body, 0);
    try testing.expect(requires_clause != null);
    try testing.expect(ensures_clause != null);
    try testing.expect(body != null);
    try testing.expect(firstChildNodeOfKind(requires_clause.?, .Body) == null);
    try testing.expect(firstChildNodeOfKind(ensures_clause.?, .Body) == null);

    const while_stmt = nthChildNodeOfKind(body.?, .WhileStmt, 0);
    const for_stmt = nthChildNodeOfKind(body.?, .ForStmt, 0);
    try testing.expect(while_stmt != null);
    try testing.expect(for_stmt != null);
    try testing.expect(nthChildNodeOfKind(while_stmt.?, .InvariantClause, 0) != null);
    try testing.expect(nthChildNodeOfKind(for_stmt.?, .InvariantClause, 0) != null);
}

test "compiler syntax parses expression precedence and postfix chains" {
    const source_text =
        \\pub fn run() -> u256 {
        \\    foo.bar(1 + 2 * 3, xs[4]).baz;
        \\    total = a + b * c;
        \\    let value = left + right * other;
        \\    assert(value >= 0, "ok");
        \\    return (left + right) * value;
        \\}
    ;

    var compilation = try compileText(source_text);
    defer compilation.deinit();

    const module = compilation.db.sources.module(compilation.root_module_id);
    const tree = try compilation.db.syntaxTree(module.file_id);
    const root = compiler.syntax.rootNode(tree);
    const function = nthChildNodeOfKind(root, .FunctionItem, 0).?;
    const body = nthChildNodeOfKind(function, .Body, 0).?;

    const expr_stmt = nthChildNodeOfKind(body, .ExprStmt, 0).?;
    const call_chain = nthChildNodeOfKind(expr_stmt, .FieldExpr, 0);
    try testing.expect(call_chain != null);
    try testing.expect(containsNodeOfKind(expr_stmt, .CallExpr));
    try testing.expect(containsNodeOfKind(expr_stmt, .IndexExpr));

    const assign_stmt = nthChildNodeOfKind(body, .AssignStmt, 0).?;
    const assign_rhs = nthChildNodeOfKind(assign_stmt, .BinaryExpr, 0);
    try testing.expect(assign_rhs != null);
    try testing.expect(containsNodeOfKind(assign_rhs.?, .BinaryExpr));

    const var_stmt = nthChildNodeOfKind(body, .VariableDeclStmt, 0).?;
    try testing.expect(containsNodeOfKind(var_stmt, .BinaryExpr));

    const assert_stmt = nthChildNodeOfKind(body, .AssertStmt, 0).?;
    try testing.expect(containsNodeOfKind(assert_stmt, .BinaryExpr));

    const return_stmt = nthChildNodeOfKind(body, .ReturnStmt, 0).?;
    try testing.expect(containsNodeOfKind(return_stmt, .BinaryExpr));
    try testing.expect(containsNodeOfKind(return_stmt, .GroupExpr));
}

test "compiler syntax parses control-flow conditions and structured expression contexts" {
    const source_text =
        \\pub fn run(items: u256, value: u256) -> u256 {
        \\    if (a + b > c) {
        \\        while (value > 0)
        \\            invariant value + 1 > 0;
        \\        {
        \\            break;
        \\        }
        \\    }
        \\
        \\    switch (value + 1) {
        \\        0 => 1;
        \\        1...2 => Point{ x: left + 1, y: right };
        \\        else => 3;
        \\    }
        \\
        \\    return value;
        \\}
    ;

    var compilation = try compileText(source_text);
    defer compilation.deinit();

    const module = compilation.db.sources.module(compilation.root_module_id);
    const tree = try compilation.db.syntaxTree(module.file_id);
    const root = compiler.syntax.rootNode(tree);
    const function = nthChildNodeOfKind(root, .FunctionItem, 0).?;
    const body = nthChildNodeOfKind(function, .Body, 0).?;

    const if_stmt = nthChildNodeOfKind(body, .IfStmt, 0).?;
    try testing.expect(containsNodeOfKind(if_stmt, .BinaryExpr));
    try testing.expect(firstChildNodeOfKind(if_stmt, .GroupParen) == null);

    const if_body = nthChildNodeOfKind(if_stmt, .Body, 0).?;
    const while_stmt = nthChildNodeOfKind(if_body, .WhileStmt, 0).?;
    const invariant = nthChildNodeOfKind(while_stmt, .InvariantClause, 0).?;
    try testing.expect(containsNodeOfKind(while_stmt, .BinaryExpr));
    try testing.expect(containsNodeOfKind(invariant, .BinaryExpr));

    const switch_stmt = nthChildNodeOfKind(body, .SwitchStmt, 0).?;
    try testing.expect(containsNodeOfKind(switch_stmt, .BinaryExpr));
    try testing.expect(firstChildNodeOfKind(switch_stmt, .GroupParen) == null);

    const range_arm = nthChildNodeOfKind(switch_stmt, .SwitchArm, 1).?;
    try testing.expect(containsNodeOfKind(range_arm, .RangeExpr));
    try testing.expect(containsNodeOfKind(range_arm, .StructLiteral));
    try testing.expect(containsNodeOfKind(range_arm, .AnonymousStructLiteralField));
    try testing.expect(containsNodeOfKind(range_arm, .BinaryExpr));
}

test "compiler syntax parses type expressions in signatures and declarations" {
    const source_text =
        \\struct Pair {
        \\    left: map<u256, address>,
        \\    right: (u256, bool),
        \\}
        \\
        \\bitfield Flags: !Result | Error {
        \\    enabled: slice[address] @bits(0..1);
        \\}
        \\
        \\pub fn run(
        \\    values: map<u256, address>,
        \\    pair: (u256, bool),
        \\    items: [u256; 4],
        \\    addrs: slice[address],
        \\) -> !map<u256, address> | Error {
        \\    let local: struct { x: u256, y: bool } = Point{ x: 1, y: true };
        \\    return values;
        \\}
    ;

    var compilation = try compileText(source_text);
    defer compilation.deinit();

    const module = compilation.db.sources.module(compilation.root_module_id);
    const tree = try compilation.db.syntaxTree(module.file_id);
    const root = compiler.syntax.rootNode(tree);

    const struct_item = nthChildNodeOfKind(root, .StructItem, 0).?;
    try testing.expect(containsNodeOfKind(struct_item, .StructField));
    try testing.expect(containsNodeOfKind(struct_item, .GenericType));
    try testing.expect(containsNodeOfKind(struct_item, .TupleType));

    const bitfield_item = nthChildNodeOfKind(root, .BitfieldItem, 0).?;
    try testing.expect(containsNodeOfKind(bitfield_item, .ErrorUnionType));
    try testing.expect(containsNodeOfKind(bitfield_item, .SliceType));

    const function = nthChildNodeOfKind(root, .FunctionItem, 0).?;
    const params = nthChildNodeOfKind(function, .ParameterList, 0).?;
    try testing.expect(nthChildNodeOfKind(params, .Parameter, 0) != null);
    try testing.expect(nthChildNodeOfKind(params, .Parameter, 3) != null);
    try testing.expect(containsNodeOfKind(params, .GenericType));
    try testing.expect(containsNodeOfKind(params, .TupleType));
    try testing.expect(containsNodeOfKind(params, .ArrayType));
    try testing.expect(containsNodeOfKind(params, .SliceType));
    try testing.expect(containsNodeOfKind(function, .ErrorUnionType));

    const body = nthChildNodeOfKind(function, .Body, 0).?;
    const local_decl = nthChildNodeOfKind(body, .VariableDeclStmt, 0).?;
    try testing.expect(containsNodeOfKind(local_decl, .AnonymousStructType));
    try testing.expect(containsNodeOfKind(local_decl, .AnonymousStructField));
}

test "compiler lowers syntax into immutable AST items" {
    const source_text =
        \\contract Counter {
        \\    storage var value: u256;
        \\    invariant value >= 0;
        \\
        \\    pub fn get() -> u256 {
        \\        return value;
        \\    }
        \\}
    ;

    var compilation = try compileText(source_text);
    defer compilation.deinit();

    const module = compilation.db.sources.module(compilation.root_module_id);
    const file = try compilation.db.astFile(module.file_id);

    try testing.expectEqual(@as(usize, 1), file.root_items.len);
    try testing.expect(file.item(file.root_items[0]).* == .Contract);
    try testing.expect(file.item(file.root_items[0]).Contract.members.len >= 2);
}

test "compiler parses and lowers structured top-level item forms" {
    const source_text =
        \\comptime const math = @import("./math.ora");
        \\storage var total: u256 = 1;
        \\const LIMIT: u256 = 2;
        \\log Transfer(indexed from: address, to: address, amount: u256);
        \\error Failed(code: u256);
    ;

    var compilation = try compileText(source_text);
    defer compilation.deinit();

    const module = compilation.db.sources.module(compilation.root_module_id);
    const tree = try compilation.db.syntaxTree(module.file_id);
    const root = compiler.syntax.rootNode(tree);

    try testing.expect(nthChildNodeOfKind(root, .ImportItem, 0) != null);
    try testing.expect(nthChildNodeOfKind(root, .FieldItem, 0) != null);
    try testing.expect(nthChildNodeOfKind(root, .ConstantItem, 0) != null);
    try testing.expect(nthChildNodeOfKind(root, .LogDeclItem, 0) != null);
    try testing.expect(nthChildNodeOfKind(root, .ErrorDeclItem, 0) != null);

    const log_decl = nthChildNodeOfKind(root, .LogDeclItem, 0).?;
    try testing.expect(nthChildNodeOfKind(log_decl, .LogField, 0) != null);

    const error_decl = nthChildNodeOfKind(root, .ErrorDeclItem, 0).?;
    try testing.expect(nthChildNodeOfKind(error_decl, .ParameterList, 0) != null);

    const file = try compilation.db.astFile(module.file_id);
    try testing.expectEqual(@as(usize, 5), file.root_items.len);

    var found_import = false;
    var found_field = false;
    var found_constant = false;
    var found_log = false;
    var found_error = false;

    for (file.root_items) |item_id| {
        switch (file.item(item_id).*) {
            .Import => |import_item| {
                found_import = true;
                try testing.expect(import_item.is_comptime);
                try testing.expectEqualStrings("math", import_item.alias.?);
                try testing.expectEqualStrings("./math.ora", import_item.path);
            },
            .Field => |field_item| {
                found_field = true;
                try testing.expectEqual(.storage, field_item.storage_class);
                try testing.expectEqual(.var_, field_item.binding_kind);
                try testing.expect(field_item.type_expr != null);
                try testing.expect(field_item.value != null);
            },
            .Constant => |constant_item| {
                if (std.mem.eql(u8, constant_item.name, "LIMIT")) {
                    found_constant = true;
                    try testing.expect(!constant_item.is_comptime);
                    try testing.expect(constant_item.type_expr != null);
                }
            },
            .LogDecl => |lowered_log| {
                found_log = true;
                try testing.expectEqual(@as(usize, 3), lowered_log.fields.len);
                try testing.expect(lowered_log.fields[0].indexed);
                try testing.expect(!lowered_log.fields[1].indexed);
            },
            .ErrorDecl => |lowered_error| {
                found_error = true;
                try testing.expectEqual(@as(usize, 1), lowered_error.parameters.len);
            },
            else => {},
        }
    }

    try testing.expect(found_import);
    try testing.expect(found_field);
    try testing.expect(found_constant);
    try testing.expect(found_log);
    try testing.expect(found_error);
}

test "compiler semantic queries index names and infer expression types" {
    const source_text =
        \\pub fn add(x: u256, y: u256) -> u256 {
        \\    let z = x + y;
        \\    return z;
        \\}
    ;

    var compilation = try compileText(source_text);
    defer compilation.deinit();

    const item_index = try compilation.db.itemIndex(compilation.root_module_id);
    try testing.expect(item_index.lookup("add") != null);

    const typecheck = try compilation.db.typeCheck(compilation.root_module_id, .{ .item = item_index.lookup("add").? });

    var saw_integer = false;
    for (typecheck.expr_types) |expr_type| {
        if (expr_type.kind() == .integer) {
            saw_integer = true;
            break;
        }
    }
    try testing.expect(saw_integer);
}

test "compiler extracts verification facts and lowers HIR handles" {
    const source_text =
        \\contract Counter {
        \\    invariant total >= 0;
        \\    storage var total: u256;
        \\
        \\    pub fn set(next: u256) -> u256
        \\        requires next >= 0;
        \\        ensures result >= 0;
        \\    {
        \\        total = next;
        \\        return total;
        \\    }
        \\}
    ;

    var compilation = try compileText(source_text);
    defer compilation.deinit();

    const ast_file = try compilation.db.astFile(compilation.db.sources.module(compilation.root_module_id).file_id);
    const contract_id = ast_file.root_items[0];
    const facts = try compilation.db.verificationFacts(compilation.root_module_id, .{ .item = contract_id });
    try testing.expect(facts.facts.len >= 1);

    const hir_result = try compilation.db.lowerToHir(compilation.root_module_id);
    try testing.expect(hir_result.items.len >= 2);
    try testing.expect(hir_result.module.raw_module.ptr != null);

    const hir_text = try hir_result.renderText(testing.allocator);
    defer testing.allocator.free(hir_text);

    try testing.expect(std.mem.containsAtLeast(u8, hir_text, 1, "ora.contract"));
    try testing.expect(std.mem.containsAtLeast(u8, hir_text, 1, "func.func @set"));
    try testing.expect(std.mem.containsAtLeast(u8, hir_text, 1, "ora.global"));
    try testing.expect(std.mem.containsAtLeast(u8, hir_text, 1, "ora.requires"));
    try testing.expect(std.mem.containsAtLeast(u8, hir_text, 1, "ora.ensures"));
    try testing.expect(std.mem.containsAtLeast(u8, hir_text, 1, "ora.sstore"));
}

test "compiler aggregates sema and verification across multiple root items" {
    const source_text =
        \\pub fn first(x: u256) -> u256
        \\    requires x >= 0;
        \\{
        \\    return x;
        \\}
        \\
        \\pub fn second(flag: bool) -> bool
        \\    ensures result == flag;
        \\{
        \\    return flag;
        \\}
    ;

    var compilation = try compileText(source_text);
    defer compilation.deinit();

    const module = compilation.db.sources.module(compilation.root_module_id);
    const ast_file = try compilation.db.astFile(module.file_id);
    try testing.expectEqual(@as(usize, 2), ast_file.root_items.len);

    const module_typecheck = try compilation.db.moduleTypeCheck(compilation.root_module_id);
    const second_fn = ast_file.item(ast_file.root_items[1]).Function;
    const second_body = ast_file.body(second_fn.body);
    const second_return = ast_file.statement(second_body.statements[0]).Return;
    try testing.expectEqual(compiler.sema.TypeKind.bool, module_typecheck.exprType(second_return.value.?).kind());

    const module_facts = try compilation.db.moduleVerificationFacts(compilation.root_module_id);
    try testing.expectEqual(@as(usize, 2), module_facts.facts.len);
    try testing.expectEqual(compiler.ast.SpecClauseKind.requires, module_facts.facts[0].kind);
    try testing.expectEqual(compiler.ast.SpecClauseKind.ensures, module_facts.facts[1].kind);

    const hir_result = try compilation.db.lowerToHir(compilation.root_module_id);
    const hir_text = try hir_result.renderText(testing.allocator);
    defer testing.allocator.free(hir_text);
    try testing.expect(std.mem.containsAtLeast(u8, hir_text, 1, "func.func @first"));
    try testing.expect(std.mem.containsAtLeast(u8, hir_text, 1, "func.func @second"));
}

test "compiler parses control flow and verification statements" {
    const source_text =
        \\pub fn run(values: u256) -> u256 {
        \\    for (values) |value, index|
        \\        invariant value >= index;
        \\    {
        \\        assert(value >= index, "ordered");
        \\        assume(value >= 0);
        \\        havoc state;
        \\    }
        \\
        \\    switch (values) {
        \\        0 => 1;
        \\        1...2 => {
        \\            return 2;
        \\        }
        \\        else => {
        \\            return 3;
        \\        }
        \\    }
        \\
        \\    try {
        \\        return 0;
        \\    } catch (err) {
        \\        return 1;
        \\    }
        \\}
    ;

    var compilation = try compileText(source_text);
    defer compilation.deinit();

    const ast_file = try compilation.db.astFile(compilation.db.sources.module(compilation.root_module_id).file_id);
    const function = ast_file.item(ast_file.root_items[0]).Function;
    const body = ast_file.body(function.body);

    try testing.expectEqual(@as(usize, 3), body.statements.len);
    try testing.expect(ast_file.statement(body.statements[0]).* == .For);
    try testing.expect(ast_file.statement(body.statements[1]).* == .Switch);
    try testing.expect(ast_file.statement(body.statements[2]).* == .Try);

    const for_stmt = ast_file.statement(body.statements[0]).For;
    try testing.expectEqual(@as(usize, 1), for_stmt.invariants.len);
    const for_body = ast_file.body(for_stmt.body);
    try testing.expectEqual(@as(usize, 3), for_body.statements.len);
    try testing.expect(ast_file.statement(for_body.statements[0]).* == .Assert);
    try testing.expect(ast_file.statement(for_body.statements[1]).* == .Assume);
    try testing.expect(ast_file.statement(for_body.statements[2]).* == .Havoc);

    const switch_stmt = ast_file.statement(body.statements[1]).Switch;
    try testing.expectEqual(@as(usize, 2), switch_stmt.arms.len);
    try testing.expect(switch_stmt.else_body != null);

    const try_stmt = ast_file.statement(body.statements[2]).Try;
    try testing.expect(try_stmt.catch_clause != null);
    try testing.expect(try_stmt.catch_clause.?.error_pattern != null);

    const resolution = try compilation.db.resolveNames(compilation.root_module_id);
    const invariant_expr = ast_file.expression(for_stmt.invariants[0]).Binary;
    try testing.expect(resolution.expr_bindings[invariant_expr.lhs.index()] != null);
    try testing.expect(resolution.expr_bindings[invariant_expr.rhs.index()] != null);
    const assert_stmt = ast_file.statement(for_body.statements[0]).Assert;
    const assert_expr = ast_file.expression(assert_stmt.condition).Binary;
    try testing.expect(resolution.expr_bindings[assert_expr.lhs.index()] != null);
    try testing.expect(resolution.expr_bindings[assert_expr.rhs.index()] != null);
}

test "compiler memoizes semantic queries per key" {
    const source_text =
        \\contract Counter {
        \\    invariant total >= 0;
        \\    storage var total: u256;
        \\
        \\    pub fn set(next: u256) -> u256
        \\        requires next >= 0;
        \\        ensures result >= 0;
        \\    {
        \\        assert(next >= 0, "non-negative");
        \\        return next;
        \\    }
        \\}
    ;

    var compilation = try compileText(source_text);
    defer compilation.deinit();

    const ast_file = try compilation.db.astFile(compilation.db.sources.module(compilation.root_module_id).file_id);
    const contract_id = ast_file.root_items[0];
    var function_id: ?compiler.ast.ItemId = null;
    for (ast_file.item(contract_id).Contract.members) |member_id| {
        if (ast_file.item(member_id).* == .Function) {
            function_id = member_id;
            break;
        }
    }

    try testing.expect(function_id != null);
    const function = ast_file.item(function_id.?).Function;

    const item_typecheck_1 = try compilation.db.typeCheck(compilation.root_module_id, .{ .item = function_id.? });
    const item_typecheck_2 = try compilation.db.typeCheck(compilation.root_module_id, .{ .item = function_id.? });
    try testing.expectEqual(@intFromPtr(item_typecheck_1), @intFromPtr(item_typecheck_2));

    const body_typecheck_1 = try compilation.db.typeCheck(compilation.root_module_id, .{ .body = function.body });
    const body_typecheck_2 = try compilation.db.typeCheck(compilation.root_module_id, .{ .body = function.body });
    try testing.expectEqual(@intFromPtr(body_typecheck_1), @intFromPtr(body_typecheck_2));
    try testing.expect(@intFromPtr(item_typecheck_1) != @intFromPtr(body_typecheck_1));

    const contract_facts_1 = try compilation.db.verificationFacts(compilation.root_module_id, .{ .item = contract_id });
    const contract_facts_2 = try compilation.db.verificationFacts(compilation.root_module_id, .{ .item = contract_id });
    try testing.expectEqual(@intFromPtr(contract_facts_1), @intFromPtr(contract_facts_2));

    const function_facts_1 = try compilation.db.verificationFacts(compilation.root_module_id, .{ .item = function_id.? });
    const function_facts_2 = try compilation.db.verificationFacts(compilation.root_module_id, .{ .item = function_id.? });
    try testing.expectEqual(@intFromPtr(function_facts_1), @intFromPtr(function_facts_2));
    try testing.expect(@intFromPtr(contract_facts_1) != @intFromPtr(function_facts_1));
    try testing.expectEqual(@as(usize, 1), contract_facts_1.facts.len);
    try testing.expectEqual(@as(usize, 2), function_facts_1.facts.len);
}

test "compiler verification facts respect item and body keys" {
    const source_text =
        \\contract Counter {
        \\    invariant total >= 0;
        \\    storage var total: u256;
        \\
        \\    pub fn set(next: u256) -> u256
        \\        requires next >= 0;
        \\        ensures result >= 0;
        \\    {
        \\        return next;
        \\    }
        \\}
    ;

    var compilation = try compileText(source_text);
    defer compilation.deinit();

    const module = compilation.db.sources.module(compilation.root_module_id);
    const ast_file = try compilation.db.astFile(module.file_id);
    const contract_id = ast_file.root_items[0];
    const contract = ast_file.item(contract_id).Contract;

    var function_id: ?compiler.ast.ItemId = null;
    for (contract.members) |member_id| {
        if (ast_file.item(member_id).* == .Function) {
            function_id = member_id;
            break;
        }
    }
    try testing.expect(function_id != null);

    const function = ast_file.item(function_id.?).Function;
    const contract_facts = try compilation.db.verificationFacts(compilation.root_module_id, .{ .item = contract_id });
    const function_facts = try compilation.db.verificationFacts(compilation.root_module_id, .{ .item = function_id.? });
    const body_facts = try compilation.db.verificationFacts(compilation.root_module_id, .{ .body = function.body });

    try testing.expectEqual(@as(usize, 1), contract_facts.facts.len);
    try testing.expectEqual(compiler.ast.SpecClauseKind.invariant, contract_facts.facts[0].kind);

    try testing.expectEqual(@as(usize, 2), function_facts.facts.len);
    try testing.expectEqual(compiler.ast.SpecClauseKind.requires, function_facts.facts[0].kind);
    try testing.expectEqual(compiler.ast.SpecClauseKind.ensures, function_facts.facts[1].kind);

    try testing.expectEqual(@as(usize, 2), body_facts.facts.len);
    try testing.expectEqual(compiler.ast.SpecClauseKind.requires, body_facts.facts[0].kind);
    try testing.expectEqual(compiler.ast.SpecClauseKind.ensures, body_facts.facts[1].kind);
}

test "compiler invalidates cached queries after source update" {
    const original_source =
        \\comptime const dep = @import("old_dep");
        \\
        \\pub fn old_name() -> u256 {
        \\    return 1;
        \\}
    ;
    const updated_source =
        \\comptime const dep = @import("new_dep");
        \\
        \\pub fn new_name() -> u256 {
        \\    return 2;
        \\}
    ;

    var compilation = try compileText(original_source);
    defer compilation.deinit();

    const module = compilation.db.sources.module(compilation.root_module_id);
    const file_id = module.file_id;

    const tree_before = try compilation.db.syntaxTree(file_id);
    const rebuilt_before = try tree_before.reconstructSource(testing.allocator);
    defer testing.allocator.free(rebuilt_before);
    try testing.expectEqualStrings(original_source, rebuilt_before);

    const graph_before = try compilation.db.moduleGraph(compilation.package_id);
    try testing.expectEqual(@as(usize, 1), graph_before.modules.len);
    try testing.expectEqual(@as(usize, 1), graph_before.modules[0].imports.len);
    try testing.expectEqualStrings("old_dep", graph_before.modules[0].imports[0].path);

    const index_before = try compilation.db.itemIndex(compilation.root_module_id);
    try testing.expect(index_before.lookup("old_name") != null);

    try compilation.db.updateSourceFile(file_id, updated_source);

    const tree_after = try compilation.db.syntaxTree(file_id);
    const rebuilt_after = try tree_after.reconstructSource(testing.allocator);
    defer testing.allocator.free(rebuilt_after);
    try testing.expectEqualStrings(updated_source, rebuilt_after);

    const graph_after = try compilation.db.moduleGraph(compilation.package_id);
    try testing.expectEqual(@as(usize, 1), graph_after.modules.len);
    try testing.expectEqual(@as(usize, 1), graph_after.modules[0].imports.len);
    try testing.expectEqualStrings("new_dep", graph_after.modules[0].imports[0].path);

    const index_after = try compilation.db.itemIndex(compilation.root_module_id);
    try testing.expect(index_after.lookup("old_name") == null);
    const new_function_id = index_after.lookup("new_name");
    try testing.expect(new_function_id != null);

    const ast_file = try compilation.db.astFile(file_id);
    try testing.expectEqual(@as(usize, 2), ast_file.root_items.len);
    const typecheck = try compilation.db.typeCheck(compilation.root_module_id, .{ .item = new_function_id.? });

    var saw_integer = false;
    for (typecheck.expr_types) |expr_type| {
        if (expr_type.kind() == .integer) {
            saw_integer = true;
            break;
        }
    }
    try testing.expect(saw_integer);
}

test "compiler handles empty modules in module-level queries" {
    var compilation = try compileText("");
    defer compilation.deinit();

    const module = compilation.db.sources.module(compilation.root_module_id);
    const ast_file = try compilation.db.astFile(module.file_id);
    try testing.expectEqual(@as(usize, 0), ast_file.root_items.len);
    try testing.expectEqual(@as(usize, 0), ast_file.bodies.len);

    const module_typecheck = try compilation.db.moduleTypeCheck(compilation.root_module_id);
    try testing.expectEqual(@as(usize, 0), module_typecheck.item_types.len);
    try testing.expectEqual(@as(usize, 0), module_typecheck.pattern_types.len);
    try testing.expectEqual(@as(usize, 0), module_typecheck.expr_types.len);
    try testing.expectEqual(@as(usize, 0), module_typecheck.body_types.len);
    try testing.expect(module_typecheck.diagnostics.isEmpty());

    const module_facts = try compilation.db.moduleVerificationFacts(compilation.root_module_id);
    try testing.expectEqual(@as(usize, 0), module_facts.facts.len);

    const hir_result = try compilation.db.lowerToHir(compilation.root_module_id);
    try testing.expectEqual(@as(usize, 0), hir_result.items.len);
}

test "compiler module graph resolves dependencies and detects cycles" {
    var compiler_db = compiler.db.CompilerDb.init(testing.allocator);
    defer compiler_db.deinit();

    const package_id = try compiler_db.addPackage("main");

    const c_file = try compiler_db.addSourceFile("c.ora",
        \\pub fn c() -> u256 {
        \\    return 1;
        \\}
    );
    const b_file = try compiler_db.addSourceFile("b.ora",
        \\comptime const c_dep = @import("./c.ora");
        \\pub fn b() -> u256 {
        \\    return 1;
        \\}
    );
    const a_file = try compiler_db.addSourceFile("a.ora",
        \\comptime const b_dep = @import("./b.ora");
        \\pub fn a() -> u256 {
        \\    return 1;
        \\}
    );

    const c_module = try compiler_db.addModule(package_id, c_file, "c");
    const b_module = try compiler_db.addModule(package_id, b_file, "b");
    const a_module = try compiler_db.addModule(package_id, a_file, "a");

    const graph = try compiler_db.moduleGraph(package_id);
    try testing.expectEqual(@as(usize, 3), graph.modules.len);
    try testing.expect(!graph.has_cycles);
    try testing.expectEqual(@as(usize, 3), graph.topo_order.len);
    try testing.expectEqual(c_module, graph.topo_order[0]);
    try testing.expectEqual(b_module, graph.topo_order[1]);
    try testing.expectEqual(a_module, graph.topo_order[2]);

    const c_summary = for (graph.modules) |module_summary| {
        if (module_summary.module_id == c_module) break module_summary;
    } else unreachable;
    const b_summary = for (graph.modules) |module_summary| {
        if (module_summary.module_id == b_module) break module_summary;
    } else unreachable;
    const a_summary = for (graph.modules) |module_summary| {
        if (module_summary.module_id == a_module) break module_summary;
    } else unreachable;

    try testing.expectEqual(@as(usize, 0), c_summary.dependencies.len);

    try testing.expectEqual(@as(usize, 1), b_summary.dependencies.len);
    try testing.expectEqual(c_module, b_summary.dependencies[0]);
    try testing.expectEqual(c_module, b_summary.imports[0].target_module_id.?);

    try testing.expectEqual(@as(usize, 1), a_summary.dependencies.len);
    try testing.expectEqual(b_module, a_summary.dependencies[0]);
    try testing.expectEqual(b_module, a_summary.imports[0].target_module_id.?);

    try compiler_db.updateSourceFile(c_file,
        \\comptime const a_dep = @import("./a.ora");
        \\pub fn c() -> u256 {
        \\    return 1;
        \\}
    );

    const cycle_graph = try compiler_db.moduleGraph(package_id);
    try testing.expect(cycle_graph.has_cycles);
    const cycle_c_summary = for (cycle_graph.modules) |module_summary| {
        if (module_summary.module_id == c_module) break module_summary;
    } else unreachable;
    try testing.expectEqual(@as(usize, 1), cycle_c_summary.dependencies.len);
    try testing.expectEqual(a_module, cycle_c_summary.dependencies[0]);
}

test "compiler invalidates dependent module caches after source update" {
    var compiler_db = compiler.db.CompilerDb.init(testing.allocator);
    defer compiler_db.deinit();

    const package_id = try compiler_db.addPackage("main");

    const c_file = try compiler_db.addSourceFile("c.ora",
        \\pub fn c() -> u256 {
        \\    return 1;
        \\}
    );
    const b_file = try compiler_db.addSourceFile("b.ora",
        \\comptime const c_dep = @import("./c.ora");
        \\pub fn b() -> u256 {
        \\    return 2;
        \\}
    );
    const a_file = try compiler_db.addSourceFile("a.ora",
        \\comptime const b_dep = @import("./b.ora");
        \\pub fn a() -> u256 {
        \\    return 3;
        \\}
    );

    const c_module = try compiler_db.addModule(package_id, c_file, "c");
    const b_module = try compiler_db.addModule(package_id, b_file, "b");
    const a_module = try compiler_db.addModule(package_id, a_file, "a");

    _ = try compiler_db.moduleGraph(package_id);

    const b_index = try compiler_db.itemIndex(b_module);
    const a_index = try compiler_db.itemIndex(a_module);
    const b_item = b_index.lookup("b").?;
    const a_item = a_index.lookup("a").?;

    const b_typecheck_before = try compiler_db.typeCheck(b_module, .{ .item = b_item });
    const a_typecheck_before = try compiler_db.typeCheck(a_module, .{ .item = a_item });
    const a_hir_before = try compiler_db.lowerToHir(a_module);

    try compiler_db.updateSourceFile(c_file,
        \\pub fn c() -> u256 {
        \\    return 99;
        \\}
    );

    const b_typecheck_after = try compiler_db.typeCheck(b_module, .{ .item = b_item });
    const a_typecheck_after = try compiler_db.typeCheck(a_module, .{ .item = a_item });
    const a_hir_after = try compiler_db.lowerToHir(a_module);
    const graph_after = try compiler_db.moduleGraph(package_id);

    try testing.expect(b_typecheck_before != b_typecheck_after);
    try testing.expect(a_typecheck_before != a_typecheck_after);
    try testing.expect(a_hir_before != a_hir_after);
    try testing.expectEqual(@as(usize, 3), graph_after.modules.len);

    _ = c_module;
}

test "compiler parses log, error, and bitfield declarations" {
    const source_text =
        \\contract Ledger {
        \\    bitfield Flags: u256 {
        \\        enabled: bool(1);
        \\        mode: u8 @bits(1..9);
        \\    }
        \\
        \\    log Transfer(indexed from: address, to: address);
        \\    error Failure(code: u256);
        \\    storage var total: u256;
        \\
        \\    pub fn emit_transfer(to: address) -> u256 {
        \\        log Transfer(total, to);
        \\        return total;
        \\    }
        \\}
    ;

    var compilation = try compileText(source_text);
    defer compilation.deinit();

    const ast_file = try compilation.db.astFile(compilation.db.sources.module(compilation.root_module_id).file_id);
    const ast_diags = try compilation.db.astDiagnostics(compilation.db.sources.module(compilation.root_module_id).file_id);
    try testing.expect(ast_diags.isEmpty());
    const contract = ast_file.item(ast_file.root_items[0]).Contract;

    try testing.expectEqual(@as(usize, 5), contract.members.len);
    try testing.expect(ast_file.item(contract.members[0]).* == .Bitfield);
    try testing.expect(ast_file.item(contract.members[1]).* == .LogDecl);
    try testing.expect(ast_file.item(contract.members[2]).* == .ErrorDecl);
    try testing.expect(ast_file.item(contract.members[3]).* == .Field);
    try testing.expect(ast_file.item(contract.members[4]).* == .Function);

    const bitfield = ast_file.item(contract.members[0]).Bitfield;
    try testing.expectEqualStrings("Flags", bitfield.name);
    try testing.expect(bitfield.base_type != null);
    try testing.expectEqual(@as(usize, 2), bitfield.fields.len);
    try testing.expect(bitfield.fields[0].width != null);
    try testing.expectEqual(@as(u32, 1), bitfield.fields[0].width.?);
    try testing.expect(bitfield.fields[1].offset != null);
    try testing.expect(bitfield.fields[1].width != null);
    try testing.expectEqual(@as(u32, 1), bitfield.fields[1].offset.?);
    try testing.expectEqual(@as(u32, 8), bitfield.fields[1].width.?);

    const log_decl = ast_file.item(contract.members[1]).LogDecl;
    try testing.expectEqualStrings("Transfer", log_decl.name);
    try testing.expectEqual(@as(usize, 2), log_decl.fields.len);
    try testing.expect(log_decl.fields[0].indexed);
    try testing.expect(!log_decl.fields[1].indexed);

    const error_decl = ast_file.item(contract.members[2]).ErrorDecl;
    try testing.expectEqualStrings("Failure", error_decl.name);
    try testing.expectEqual(@as(usize, 1), error_decl.parameters.len);

    const function = ast_file.item(contract.members[4]).Function;
    const body = ast_file.body(function.body);
    try testing.expectEqual(@as(usize, 2), body.statements.len);
    try testing.expect(ast_file.statement(body.statements[0]).* == .Log);

    const log_stmt = ast_file.statement(body.statements[0]).Log;
    try testing.expectEqualStrings("Transfer", log_stmt.name);
    try testing.expectEqual(@as(usize, 2), log_stmt.args.len);

    const item_index = try compilation.db.itemIndex(compilation.root_module_id);
    try testing.expect(item_index.lookup("Flags") != null);
    try testing.expect(item_index.lookup("Transfer") != null);
    try testing.expect(item_index.lookup("Failure") != null);
    try testing.expect(item_index.lookup("Ledger.Transfer") != null);

    const resolution = try compilation.db.resolveNames(compilation.root_module_id);
    try testing.expect(resolution.expr_bindings[log_stmt.args[0].index()] != null);
    try testing.expect(resolution.expr_bindings[log_stmt.args[1].index()] != null);

    const hir_result = try compilation.db.lowerToHir(compilation.root_module_id);
    var saw_bitfield = false;
    var saw_log_decl = false;
    var saw_error_decl = false;
    for (hir_result.items) |item| {
        switch (item.kind) {
            .bitfield => saw_bitfield = true,
            .log_decl => saw_log_decl = true,
            .error_decl => saw_error_decl = true,
            else => {},
        }
    }
    try testing.expect(saw_bitfield);
    try testing.expect(saw_log_decl);
    try testing.expect(saw_error_decl);

    const hir_text = try hir_result.renderText(testing.allocator);
    defer testing.allocator.free(hir_text);
    try testing.expect(std.mem.containsAtLeast(u8, hir_text, 1, "ora.log_decl"));
    try testing.expect(std.mem.containsAtLeast(u8, hir_text, 1, "ora.error_decl"));
    try testing.expect(!std.mem.containsAtLeast(u8, hir_text, 1, "\"ora.log_decl\""));
    try testing.expect(!std.mem.containsAtLeast(u8, hir_text, 1, "\"ora.error_decl\""));
}

test "compiler lowers structured type expressions" {
    const source_text =
        \\struct Types {
        \\    values: [u256; 5];
        \\    view: slice[address];
        \\    table: map<address, map<address, u256>>;
        \\}
        \\
        \\error Failure(code: u256);
        \\
        \\pub fn use_types(
        \\    buffer: slice[u256],
        \\    pair: (u8, bool),
        \\    bounded: MinValue<u256, 100>,
        \\    maybe: !u256 | Failure,
        \\) -> slice[u256] {
        \\    return buffer;
        \\}
    ;

    var compilation = try compileText(source_text);
    defer compilation.deinit();

    const ast_file = try compilation.db.astFile(compilation.db.sources.module(compilation.root_module_id).file_id);
    const ast_diags = try compilation.db.astDiagnostics(compilation.db.sources.module(compilation.root_module_id).file_id);
    try testing.expect(ast_diags.isEmpty());

    const struct_item = ast_file.item(ast_file.root_items[0]).Struct;
    try testing.expect(ast_file.typeExpr(struct_item.fields[0].type_expr).* == .Array);
    try testing.expect(ast_file.typeExpr(struct_item.fields[1].type_expr).* == .Slice);
    try testing.expect(ast_file.typeExpr(struct_item.fields[2].type_expr).* == .Generic);

    const array_type = ast_file.typeExpr(struct_item.fields[0].type_expr).Array;
    try testing.expectEqualStrings("5", array_type.size.text);
    try testing.expect(ast_file.typeExpr(array_type.element).* == .Path);
    try testing.expectEqualStrings("u256", ast_file.typeExpr(array_type.element).Path.name);

    const map_type = ast_file.typeExpr(struct_item.fields[2].type_expr).Generic;
    try testing.expectEqualStrings("map", map_type.name);
    try testing.expectEqual(@as(usize, 2), map_type.args.len);
    try testing.expect(map_type.args[0] == .Type);
    try testing.expect(map_type.args[1] == .Type);
    try testing.expect(ast_file.typeExpr(map_type.args[1].Type).* == .Generic);

    const function = ast_file.item(ast_file.root_items[2]).Function;
    try testing.expectEqual(@as(usize, 4), function.parameters.len);
    try testing.expect(ast_file.typeExpr(function.parameters[0].type_expr).* == .Slice);
    try testing.expect(ast_file.typeExpr(function.parameters[1].type_expr).* == .Tuple);
    try testing.expect(ast_file.typeExpr(function.parameters[2].type_expr).* == .Generic);
    try testing.expect(ast_file.typeExpr(function.parameters[3].type_expr).* == .ErrorUnion);
    try testing.expect(ast_file.typeExpr(function.return_type.?).* == .Slice);

    const bounded_type = ast_file.typeExpr(function.parameters[2].type_expr).Generic;
    try testing.expectEqualStrings("MinValue", bounded_type.name);
    try testing.expectEqual(@as(usize, 2), bounded_type.args.len);
    try testing.expect(bounded_type.args[0] == .Type);
    try testing.expect(bounded_type.args[1] == .Integer);
    try testing.expectEqualStrings("100", bounded_type.args[1].Integer.text);

    const maybe_type = ast_file.typeExpr(function.parameters[3].type_expr).ErrorUnion;
    try testing.expect(ast_file.typeExpr(maybe_type.payload).* == .Path);
    try testing.expectEqual(@as(usize, 1), maybe_type.errors.len);
    try testing.expectEqualStrings("Failure", ast_file.typeExpr(maybe_type.errors[0]).Path.name);

    const typecheck = try compilation.db.typeCheck(compilation.root_module_id, .{ .item = ast_file.root_items[2] });
    const body = ast_file.body(function.body);
    const return_stmt = ast_file.statement(body.statements[0]).Return;
    const return_expr = return_stmt.value.?;
    try testing.expectEqual(compiler.sema.TypeKind.slice, typecheck.exprType(return_expr).kind());
    try testing.expect(typecheck.exprType(return_expr).elementType() != null);
    try testing.expectEqual(compiler.sema.TypeKind.integer, typecheck.exprType(return_expr).elementType().?.kind());

    const buffer_type = typecheck.pattern_types[function.parameters[0].pattern.index()];
    try testing.expectEqual(compiler.sema.TypeKind.slice, buffer_type.kind());
    try testing.expect(buffer_type.elementType() != null);
    try testing.expectEqual(compiler.sema.TypeKind.integer, buffer_type.elementType().?.kind());

    const maybe_typecheck = typecheck.pattern_types[function.parameters[3].pattern.index()];
    try testing.expectEqual(compiler.sema.TypeKind.error_union, maybe_typecheck.kind());
    try testing.expect(maybe_typecheck.payloadType() != null);
    try testing.expectEqual(compiler.sema.TypeKind.integer, maybe_typecheck.payloadType().?.kind());
    try testing.expectEqual(@as(usize, 1), maybe_typecheck.errorTypes().len);
    try testing.expectEqual(compiler.sema.TypeKind.named, maybe_typecheck.errorTypes()[0].kind());
}

test "compiler lowers struct and enum declarations through real decl ops" {
    const source_text =
        \\struct Pair {
        \\    left: u256;
        \\    right: bool;
        \\}
        \\
        \\enum Mode {
        \\    off,
        \\    on,
        \\}
    ;

    var compilation = try compileText(source_text);
    defer compilation.deinit();

    const hir_result = try compilation.db.lowerToHir(compilation.root_module_id);
    const hir_text = try hir_result.renderText(testing.allocator);
    defer testing.allocator.free(hir_text);

    try testing.expect(std.mem.containsAtLeast(u8, hir_text, 1, "ora.struct.decl"));
    try testing.expect(std.mem.containsAtLeast(u8, hir_text, 1, "ora.enum.decl"));
    try testing.expect(!std.mem.containsAtLeast(u8, hir_text, 1, "\"ora.struct_decl\""));
    try testing.expect(!std.mem.containsAtLeast(u8, hir_text, 1, "\"ora.enum_decl\""));
}

test "compiler lowers struct instantiate and field extract through real ops" {
    const source_text =
        \\struct Pair {
        \\    left: u256;
        \\    right: u256;
        \\}
        \\
        \\pub fn read() -> u256 {
        \\    let pair = Pair { right: 2, left: 1 };
        \\    return pair.left;
        \\}
    ;

    var compilation = try compileText(source_text);
    defer compilation.deinit();

    const hir_result = try compilation.db.lowerToHir(compilation.root_module_id);
    const hir_text = try hir_result.renderText(testing.allocator);
    defer testing.allocator.free(hir_text);

    try testing.expect(std.mem.containsAtLeast(u8, hir_text, 1, "ora.struct_instantiate"));
    try testing.expect(std.mem.containsAtLeast(u8, hir_text, 1, "ora.struct_field_extract"));
    try testing.expect(!std.mem.containsAtLeast(u8, hir_text, 1, "\"ora.struct.create\""));
    try testing.expect(!std.mem.containsAtLeast(u8, hir_text, 1, "\"ora.field_access\""));
}

test "compiler lowers enum variant access through real enum constant op" {
    const source_text =
        \\enum Mode {
        \\    off,
        \\    on,
        \\}
        \\
        \\pub fn current() -> Mode {
        \\    return Mode.on;
        \\}
    ;

    var compilation = try compileText(source_text);
    defer compilation.deinit();

    const hir_result = try compilation.db.lowerToHir(compilation.root_module_id);
    const hir_text = try hir_result.renderText(testing.allocator);
    defer testing.allocator.free(hir_text);

    try testing.expect(std.mem.containsAtLeast(u8, hir_text, 1, "ora.enum_constant"));
    try testing.expect(!std.mem.containsAtLeast(u8, hir_text, 1, "\"ora.field_access\""));
}

test "compiler wraps payload returns into real error ok op" {
    const source_text =
        \\error Failure(code: u256);
        \\
        \\pub fn lift(value: u256) -> !u256 | Failure {
        \\    return value;
        \\}
    ;

    var compilation = try compileText(source_text);
    defer compilation.deinit();

    const hir_result = try compilation.db.lowerToHir(compilation.root_module_id);
    const hir_text = try hir_result.renderText(testing.allocator);
    defer testing.allocator.free(hir_text);

    try testing.expect(std.mem.containsAtLeast(u8, hir_text, 1, "ora.error.ok"));
}

test "compiler lowers try expressions through real error helper ops" {
    const source_text =
        \\error Failure(code: u256);
        \\
        \\pub fn lift(maybe: !u256 | Failure) -> !u256 | Failure {
        \\    return try maybe;
        \\}
    ;

    var compilation = try compileText(source_text);
    defer compilation.deinit();

    const hir_result = try compilation.db.lowerToHir(compilation.root_module_id);
    const hir_text = try hir_result.renderText(testing.allocator);
    defer testing.allocator.free(hir_text);

    try testing.expect(std.mem.containsAtLeast(u8, hir_text, 1, "ora.error.is_error"));
    try testing.expect(std.mem.containsAtLeast(u8, hir_text, 1, "ora.error.get_error"));
    try testing.expect(std.mem.containsAtLeast(u8, hir_text, 1, "ora.error.err"));
    try testing.expect(std.mem.containsAtLeast(u8, hir_text, 1, "ora.error.unwrap"));
}

test "compiler lowers struct field mutation through real field update op" {
    const source_text =
        \\struct Pair {
        \\    left: u256;
        \\    right: u256;
        \\}
        \\
        \\pub fn update() -> u256 {
        \\    let pair = Pair { left: 1, right: 2 };
        \\    pair.left = 42;
        \\    return pair.left;
        \\}
    ;

    var compilation = try compileText(source_text);
    defer compilation.deinit();

    const hir_result = try compilation.db.lowerToHir(compilation.root_module_id);
    const hir_text = try hir_result.renderText(testing.allocator);
    defer testing.allocator.free(hir_text);

    try testing.expect(std.mem.containsAtLeast(u8, hir_text, 1, "ora.struct_field_update"));
    try testing.expect(std.mem.containsAtLeast(u8, hir_text, 1, "ora.struct_field_extract"));
    try testing.expect(!std.mem.containsAtLeast(u8, hir_text, 1, "\"ora.field_access\""));
}

test "compiler lowers string, bytes, and address literals through real ops" {
    const source_text =
        \\pub fn owner() -> address {
        \\    let note = "hello";
        \\    let payload = hex"deadbeef";
        \\    let who = 0x1234567890abcdef1234567890abcdef12345678;
        \\    return who;
        \\}
    ;

    var compilation = try compileText(source_text);
    defer compilation.deinit();

    const hir_result = try compilation.db.lowerToHir(compilation.root_module_id);
    const hir_text = try hir_result.renderText(testing.allocator);
    defer testing.allocator.free(hir_text);

    try testing.expect(std.mem.containsAtLeast(u8, hir_text, 1, "ora.string.constant"));
    try testing.expect(std.mem.containsAtLeast(u8, hir_text, 1, "ora.bytes.constant"));
    try testing.expect(std.mem.containsAtLeast(u8, hir_text, 1, "ora.i160.to.addr"));
    try testing.expect(!std.mem.containsAtLeast(u8, hir_text, 1, "ora.string_const"));
    try testing.expect(!std.mem.containsAtLeast(u8, hir_text, 1, "ora.bytes_const"));
    try testing.expect(!std.mem.containsAtLeast(u8, hir_text, 1, "ora.address_const"));
}

test "compiler lowers top-level const items through ora.const" {
    const source_text =
        \\const LIMIT: u256 = 2;
        \\const READY: bool = true;
        \\
        \\pub fn run() -> u256 {
        \\    if (READY) {
        \\        return LIMIT;
        \\    }
        \\    return 0;
        \\}
    ;

    var compilation = try compileText(source_text);
    defer compilation.deinit();

    const hir_result = try compilation.db.lowerToHir(compilation.root_module_id);
    const hir_text = try hir_result.renderText(testing.allocator);
    defer testing.allocator.free(hir_text);

    try testing.expect(std.mem.containsAtLeast(u8, hir_text, 2, "ora.const"));
    try testing.expect(std.mem.containsAtLeast(u8, hir_text, 1, "LIMIT"));
    try testing.expect(std.mem.containsAtLeast(u8, hir_text, 1, "READY"));
    try testing.expect(!std.mem.containsAtLeast(u8, hir_text, 1, "ora.constant_decl"));
}

test "compiler lowers literal top-level const items through ora.const" {
    const source_text =
        \\const NAME: string = "ora";
        \\const PAYLOAD: bytes = hex"deadbeef";
        \\const OWNER: address = 0x1234567890abcdef1234567890abcdef12345678;
        \\
        \\pub fn run() -> u256 {
        \\    return 1;
        \\}
    ;

    var compilation = try compileText(source_text);
    defer compilation.deinit();

    const hir_result = try compilation.db.lowerToHir(compilation.root_module_id);
    const hir_text = try hir_result.renderText(testing.allocator);
    defer testing.allocator.free(hir_text);

    try testing.expect(std.mem.containsAtLeast(u8, hir_text, 3, "ora.const"));
    try testing.expect(std.mem.containsAtLeast(u8, hir_text, 1, "NAME"));
    try testing.expect(std.mem.containsAtLeast(u8, hir_text, 1, "PAYLOAD"));
    try testing.expect(std.mem.containsAtLeast(u8, hir_text, 1, "OWNER"));
    try testing.expect(std.mem.containsAtLeast(u8, hir_text, 1, "\"ora\""));
    try testing.expect(std.mem.containsAtLeast(u8, hir_text, 1, "\"deadbeef\""));
    try testing.expect(!std.mem.containsAtLeast(u8, hir_text, 1, "ora.constant_decl"));
}

test "compiler lowers immutable contract fields through ora.immutable" {
    const source_text =
        \\contract C {
        \\    immutable OWNER: address = 0x1234567890abcdef1234567890abcdef12345678;
        \\
        \\    pub fn owner() -> address {
        \\        return OWNER;
        \\    }
        \\}
    ;

    var compilation = try compileText(source_text);
    defer compilation.deinit();

    const hir_result = try compilation.db.lowerToHir(compilation.root_module_id);
    const hir_text = try hir_result.renderText(testing.allocator);
    defer testing.allocator.free(hir_text);

    try testing.expect(std.mem.containsAtLeast(u8, hir_text, 1, "ora.immutable"));
    try testing.expect(std.mem.containsAtLeast(u8, hir_text, 1, "OWNER"));
    try testing.expect(std.mem.containsAtLeast(u8, hir_text, 2, "ora.i160.to.addr"));
    try testing.expect(!std.mem.containsAtLeast(u8, hir_text, 1, "ora.field_decl"));
    try testing.expect(!std.mem.containsAtLeast(u8, hir_text, 1, "ora.immutable_decl"));
}

test "compiler lowers computed top-level const items through ora.const" {
    const source_text =
        \\const LIMIT: u256 = 1 + 2 * 3;
        \\const READY: bool = 4 > 3;
        \\
        \\pub fn run() -> u256 {
        \\    if (READY) {
        \\        return LIMIT;
        \\    }
        \\    return 0;
        \\}
    ;

    var compilation = try compileText(source_text);
    defer compilation.deinit();

    const hir_result = try compilation.db.lowerToHir(compilation.root_module_id);
    const hir_text = try hir_result.renderText(testing.allocator);
    defer testing.allocator.free(hir_text);

    try testing.expect(std.mem.containsAtLeast(u8, hir_text, 2, "ora.const"));
    try testing.expect(std.mem.containsAtLeast(u8, hir_text, 1, "LIMIT"));
    try testing.expect(std.mem.containsAtLeast(u8, hir_text, 1, "READY"));
    try testing.expect(!std.mem.containsAtLeast(u8, hir_text, 1, "ora.constant_decl"));
}

test "compiler lowers bitfield types as wire integers with metadata attrs" {
    const source_text =
        \\contract Bits {
        \\    bitfield Flags: u256 {
        \\        enabled: u1,
        \\        signed: i8,
        \\    }
        \\
        \\    storage packed: Flags;
        \\
        \\    pub fn use_bits(flag: Flags) -> u256 {
        \\        packed = flag;
        \\        return packed;
        \\    }
        \\}
    ;

    var compilation = try compileText(source_text);
    defer compilation.deinit();

    const hir_result = try compilation.db.lowerToHir(compilation.root_module_id);
    const hir_text = try hir_result.renderText(testing.allocator);
    defer testing.allocator.free(hir_text);

    try testing.expect(std.mem.containsAtLeast(u8, hir_text, 2, "ora.bitfield = \"Flags\""));
    try testing.expect(std.mem.containsAtLeast(u8, hir_text, 1, "ora.bitfield_layout"));
    try testing.expect(std.mem.containsAtLeast(u8, hir_text, 1, "i256"));
}

test "compiler resolves named path types to declaration kinds" {
    const source_text =
        \\struct Pair {
        \\    x: u256;
        \\}
        \\
        \\enum Mode: u8 {
        \\    idle,
        \\}
        \\
        \\pub fn wrap(pair: Pair, mode: Mode) -> Pair {
        \\    return pair;
        \\}
    ;

    var compilation = try compileText(source_text);
    defer compilation.deinit();

    const module = compilation.db.sources.module(compilation.root_module_id);
    const ast_file = try compilation.db.astFile(module.file_id);
    const function = ast_file.item(ast_file.root_items[2]).Function;
    const typecheck = try compilation.db.typeCheck(compilation.root_module_id, .{ .item = ast_file.root_items[2] });

    const pair_type = typecheck.pattern_types[function.parameters[0].pattern.index()];
    try testing.expectEqual(compiler.sema.TypeKind.struct_, pair_type.kind());

    const mode_type = typecheck.pattern_types[function.parameters[1].pattern.index()];
    try testing.expectEqual(compiler.sema.TypeKind.enum_, mode_type.kind());

    const return_type = typecheck.body_types[function.body.index()];
    try testing.expectEqual(compiler.sema.TypeKind.struct_, return_type.kind());
}

test "compiler infers field and index access types" {
    const source_text =
        \\struct Pair {
        \\    first: u256;
        \\    second: bool;
        \\}
        \\
        \\contract Box {
        \\    storage var total: u256;
        \\}
        \\
        \\pub fn probe(pair: Pair, values: [u256; 2], table: map<address, bool>) -> bool {
        \\    let a = pair.first;
        \\    let b = values[0];
        \\    let c = table[0x1234567890abcdef1234567890abcdef12345678];
        \\    let d = (1, false)[1];
        \\    return c;
        \\}
    ;

    var compilation = try compileText(source_text);
    defer compilation.deinit();

    const module = compilation.db.sources.module(compilation.root_module_id);
    const ast_file = try compilation.db.astFile(module.file_id);
    const function = ast_file.item(ast_file.root_items[2]).Function;
    const body = ast_file.body(function.body);
    const typecheck = try compilation.db.typeCheck(compilation.root_module_id, .{ .item = ast_file.root_items[2] });

    const a_stmt = ast_file.statement(body.statements[0]).VariableDecl;
    const b_stmt = ast_file.statement(body.statements[1]).VariableDecl;
    const c_stmt = ast_file.statement(body.statements[2]).VariableDecl;
    const d_stmt = ast_file.statement(body.statements[3]).VariableDecl;
    const ret_stmt = ast_file.statement(body.statements[4]).Return;

    try testing.expectEqual(compiler.sema.TypeKind.integer, typecheck.pattern_types[a_stmt.pattern.index()].kind());
    try testing.expectEqual(compiler.sema.TypeKind.integer, typecheck.pattern_types[b_stmt.pattern.index()].kind());
    try testing.expectEqual(compiler.sema.TypeKind.bool, typecheck.pattern_types[c_stmt.pattern.index()].kind());
    try testing.expectEqual(compiler.sema.TypeKind.bool, typecheck.pattern_types[d_stmt.pattern.index()].kind());
    try testing.expectEqual(compiler.sema.TypeKind.bool, typecheck.exprType(ret_stmt.value.?).kind());
}

test "compiler infers direct and member call return types" {
    const source_text =
        \\contract Ledger {
        \\    pub fn amount() -> u256 {
        \\        return 1;
        \\    }
        \\}
        \\
        \\pub fn helper() -> bool {
        \\    return true;
        \\}
        \\
        \\pub fn probe() -> u256 {
        \\    let a = helper();
        \\    let b = Ledger.amount();
        \\    return b;
        \\}
    ;

    var compilation = try compileText(source_text);
    defer compilation.deinit();

    const module = compilation.db.sources.module(compilation.root_module_id);
    const ast_file = try compilation.db.astFile(module.file_id);
    const function = ast_file.item(ast_file.root_items[2]).Function;
    const body = ast_file.body(function.body);
    const typecheck = try compilation.db.typeCheck(compilation.root_module_id, .{ .item = ast_file.root_items[2] });

    const a_stmt = ast_file.statement(body.statements[0]).VariableDecl;
    const b_stmt = ast_file.statement(body.statements[1]).VariableDecl;
    const ret_stmt = ast_file.statement(body.statements[2]).Return;

    try testing.expectEqual(compiler.sema.TypeKind.bool, typecheck.pattern_types[a_stmt.pattern.index()].kind());
    try testing.expectEqual(compiler.sema.TypeKind.integer, typecheck.pattern_types[b_stmt.pattern.index()].kind());
    try testing.expectEqual(compiler.sema.TypeKind.integer, typecheck.exprType(ret_stmt.value.?).kind());
}

test "compiler infers operator result types from operand compatibility" {
    const source_text =
        \\error Failure(code: u256);
        \\
        \\pub fn probe(flag: bool, value: u256, other: u8, maybe: !u256 | Failure) -> u256 {
        \\    let sum = value + other;
        \\    let bits = value & other;
        \\    let cmp = value < other;
        \\    let logic = flag && true;
        \\    let negated = -value;
        \\    let failed = !value;
        \\    let unwrapped = try maybe;
        \\    return unwrapped;
        \\}
    ;

    var compilation = try compileText(source_text);
    defer compilation.deinit();

    const module = compilation.db.sources.module(compilation.root_module_id);
    const ast_file = try compilation.db.astFile(module.file_id);
    const function = ast_file.item(ast_file.root_items[1]).Function;
    const body = ast_file.body(function.body);
    const typecheck = try compilation.db.typeCheck(compilation.root_module_id, .{ .item = ast_file.root_items[1] });

    const sum_stmt = ast_file.statement(body.statements[0]).VariableDecl;
    const bits_stmt = ast_file.statement(body.statements[1]).VariableDecl;
    const cmp_stmt = ast_file.statement(body.statements[2]).VariableDecl;
    const logic_stmt = ast_file.statement(body.statements[3]).VariableDecl;
    const negated_stmt = ast_file.statement(body.statements[4]).VariableDecl;
    const failed_stmt = ast_file.statement(body.statements[5]).VariableDecl;
    const unwrapped_stmt = ast_file.statement(body.statements[6]).VariableDecl;
    const ret_stmt = ast_file.statement(body.statements[7]).Return;

    try testing.expectEqual(compiler.sema.TypeKind.integer, typecheck.pattern_types[sum_stmt.pattern.index()].kind());
    try testing.expectEqual(compiler.sema.TypeKind.integer, typecheck.pattern_types[bits_stmt.pattern.index()].kind());
    try testing.expectEqual(compiler.sema.TypeKind.bool, typecheck.pattern_types[cmp_stmt.pattern.index()].kind());
    try testing.expectEqual(compiler.sema.TypeKind.bool, typecheck.pattern_types[logic_stmt.pattern.index()].kind());
    try testing.expectEqual(compiler.sema.TypeKind.integer, typecheck.pattern_types[negated_stmt.pattern.index()].kind());
    try testing.expectEqual(compiler.sema.TypeKind.unknown, typecheck.pattern_types[failed_stmt.pattern.index()].kind());
    try testing.expectEqual(compiler.sema.TypeKind.integer, typecheck.pattern_types[unwrapped_stmt.pattern.index()].kind());
    try testing.expectEqual(compiler.sema.TypeKind.integer, typecheck.exprType(ret_stmt.value.?).kind());
}

test "compiler tracks HIR unknown type fallbacks" {
    const source_text =
        \\pub fn probe(value: u256) -> u256 {
        \\    return value.missing;
        \\}
    ;

    var compilation = try compileText(source_text);
    defer compilation.deinit();

    const module = compilation.db.sources.module(compilation.root_module_id);
    const ast_file = try compilation.db.astFile(module.file_id);
    const type_diags = try compilation.db.typeCheckDiagnostics(compilation.root_module_id, .{ .item = ast_file.root_items[0] });
    try testing.expect(type_diags.items.items.len > 0);

    const hir_result = try compilation.db.lowerToHir(compilation.root_module_id);
    try testing.expect(hir_result.type_fallback_count > 0);
    try testing.expectEqual(compiler.hir.TypeFallbackReason.sema_unknown, hir_result.type_fallbacks[0].reason);
    try testing.expectEqual(module.file_id, hir_result.type_fallbacks[0].location.file_id);
}

test "compiler lowers tuple locals without tuple fallback" {
    const source_text =
        \\pub fn probe() -> u256 {
        \\    let coords = (1, 2);
        \\    return 1;
        \\}
    ;

    var compilation = try compileText(source_text);
    defer compilation.deinit();

    const hir_result = try compilation.db.lowerToHir(compilation.root_module_id);
    for (hir_result.type_fallbacks) |fallback| {
        try testing.expect(fallback.reason != .unsupported_tuple_sema_type);
        try testing.expect(fallback.reason != .unsupported_syntax_type);
    }
}

test "compiler lowers tuple HIR types without tuple fallback" {
    const source_text =
        \\pub fn pair() -> (u256, bool) {
        \\    let coords = (1, true);
        \\    return coords;
        \\}
    ;

    var compilation = try compileText(source_text);
    defer compilation.deinit();

    const hir_result = try compilation.db.lowerToHir(compilation.root_module_id);
    const hir_text = try hir_result.renderText(testing.allocator);
    defer testing.allocator.free(hir_text);

    for (hir_result.type_fallbacks) |fallback| {
        try testing.expect(fallback.reason != .unsupported_tuple_sema_type);
        try testing.expect(fallback.reason != .unsupported_syntax_type);
    }

    try testing.expect(std.mem.containsAtLeast(u8, hir_text, 1, "!ora.tuple<i256, i1>"));
}

test "compiler lowers tuple expressions through real tuple ops" {
    const source_text =
        \\pub fn pair() -> u256 {
        \\    let coords = (1, true);
        \\    return coords[0];
        \\}
    ;

    var compilation = try compileText(source_text);
    defer compilation.deinit();

    const hir_result = try compilation.db.lowerToHir(compilation.root_module_id);
    const hir_text = try hir_result.renderText(testing.allocator);
    defer testing.allocator.free(hir_text);

    try testing.expect(std.mem.containsAtLeast(u8, hir_text, 1, "ora.tuple_create"));
    try testing.expect(std.mem.containsAtLeast(u8, hir_text, 1, "ora.tuple_extract"));
    try testing.expect(!std.mem.containsAtLeast(u8, hir_text, 1, "\"ora.tuple.create\""));
    try testing.expect(!std.mem.containsAtLeast(u8, hir_text, 1, "\"ora.index_access\""));
}

test "compiler lowers function-valued bindings without function fallback" {
    const source_text =
        \\fn helper(value: u256) -> u256 {
        \\    return value;
        \\}
        \\
        \\pub fn probe() -> u256 {
        \\    let f = helper;
        \\    return 1;
        \\}
    ;

    var compilation = try compileText(source_text);
    defer compilation.deinit();

    const hir_result = try compilation.db.lowerToHir(compilation.root_module_id);
    const hir_text = try hir_result.renderText(testing.allocator);
    defer testing.allocator.free(hir_text);

    for (hir_result.type_fallbacks) |fallback| {
        try testing.expect(fallback.reason != .unsupported_function_sema_type);
    }
    try testing.expect(std.mem.containsAtLeast(u8, hir_text, 1, "ora.function_ref"));
    try testing.expect(std.mem.containsAtLeast(u8, hir_text, 1, "!ora.function<"));
    try testing.expect(!std.mem.containsAtLeast(u8, hir_text, 1, "\"ora.function_ref\""));
}

test "compiler syntax splits nested generic closing tokens" {
    const source_text =
        \\struct Types {
        \\    table: map<address, map<address, u256>>;
        \\}
    ;

    var compilation = try compileText(source_text);
    defer compilation.deinit();

    const module = compilation.db.sources.module(compilation.root_module_id);
    const tree = try compilation.db.syntaxTree(module.file_id);
    const root = compiler.syntax.rootNode(tree);
    const outer_generic = nthDescendantNodeOfKind(root, .GenericType, 0).?;
    const inner_generic = nthDescendantNodeOfKind(root, .GenericType, 1).?;

    const outer_close = outer_generic.lastToken().?;
    const inner_close = inner_generic.lastToken().?;

    try testing.expectEqual(compiler.syntax.TokenKind.Greater, outer_close.kind());
    try testing.expectEqual(compiler.syntax.TokenKind.Greater, inner_close.kind());
    try testing.expect(outer_close.id != inner_close.id);
    try testing.expectEqual(@as(usize, 1), outer_close.range().end - outer_close.range().start);
    try testing.expectEqual(@as(usize, 1), inner_close.range().end - inner_close.range().start);
}

test "compiler lowers builtin, quantified, and verification expressions" {
    const source_text =
        \\pub fn verify(values: slice[u256], next: address) -> u256
        \\    ensures old(result) >= 0;
        \\{
        \\    assert(forall i: u256 where i < 4 => values[i] >= 0);
        \\    let casted = @cast(address, next);
        \\    let quotient = @divTrunc(10, 3);
        \\    let snapshot = old(quotient);
        \\    let addr = 0x1234567890abcdef1234567890abcdef12345678;
        \\    let data = hex"deadbeef";
        \\    return quotient;
        \\}
    ;

    var compilation = try compileText(source_text);
    defer compilation.deinit();

    const module = compilation.db.sources.module(compilation.root_module_id);
    const ast_file = try compilation.db.astFile(module.file_id);
    const ast_diags = try compilation.db.astDiagnostics(module.file_id);
    try testing.expect(ast_diags.isEmpty());

    const function = ast_file.item(ast_file.root_items[0]).Function;
    try testing.expectEqual(@as(usize, 1), function.clauses.len);
    const ensures_expr = function.clauses[0].expr;
    try testing.expect(ast_file.expression(ensures_expr).* == .Binary);

    const ensures_binary = ast_file.expression(ensures_expr).Binary;
    try testing.expect(ast_file.expression(ensures_binary.lhs).* == .Old);
    const old_result = ast_file.expression(ensures_binary.lhs).Old;
    try testing.expect(ast_file.expression(old_result.expr).* == .Result);

    const body = ast_file.body(function.body);
    try testing.expectEqual(@as(usize, 7), body.statements.len);

    const assert_stmt = ast_file.statement(body.statements[0]).Assert;
    try testing.expect(ast_file.expression(assert_stmt.condition).* == .Quantified);
    const quantified = ast_file.expression(assert_stmt.condition).Quantified;
    try testing.expectEqual(compiler.ast.Quantifier.forall, quantified.quantifier);
    try testing.expect(ast_file.typeExpr(quantified.type_expr).* == .Path);
    try testing.expectEqualStrings("u256", ast_file.typeExpr(quantified.type_expr).Path.name);
    try testing.expect(quantified.condition != null);

    const casted_stmt = ast_file.statement(body.statements[1]).VariableDecl;
    const quotient_stmt = ast_file.statement(body.statements[2]).VariableDecl;
    const snapshot_stmt = ast_file.statement(body.statements[3]).VariableDecl;
    const addr_stmt = ast_file.statement(body.statements[4]).VariableDecl;
    const data_stmt = ast_file.statement(body.statements[5]).VariableDecl;
    const return_stmt = ast_file.statement(body.statements[6]).Return;

    try testing.expect(ast_file.expression(casted_stmt.value.?).* == .Builtin);
    try testing.expect(ast_file.expression(quotient_stmt.value.?).* == .Builtin);
    try testing.expect(ast_file.expression(snapshot_stmt.value.?).* == .Old);
    try testing.expect(ast_file.expression(addr_stmt.value.?).* == .AddressLiteral);
    try testing.expect(ast_file.expression(data_stmt.value.?).* == .BytesLiteral);
    try testing.expect(ast_file.expression(return_stmt.value.?).* == .Name);

    const casted_builtin = ast_file.expression(casted_stmt.value.?).Builtin;
    try testing.expectEqualStrings("cast", casted_builtin.name);
    try testing.expect(casted_builtin.type_arg != null);

    const quotient_builtin = ast_file.expression(quotient_stmt.value.?).Builtin;
    try testing.expectEqualStrings("divTrunc", quotient_builtin.name);
    try testing.expectEqual(@as(usize, 2), quotient_builtin.args.len);

    const resolution = try compilation.db.resolveNames(compilation.root_module_id);
    const quantified_condition = ast_file.expression(quantified.condition.?).Binary;
    try testing.expect(resolution.expr_bindings[quantified_condition.lhs.index()] != null);
    const quantified_body = ast_file.expression(quantified.body).Binary;
    const quantified_index = ast_file.expression(quantified_body.lhs).Index;
    try testing.expect(resolution.expr_bindings[quantified_index.index.index()] != null);

    const typecheck = try compilation.db.typeCheck(compilation.root_module_id, .{ .item = ast_file.root_items[0] });
    try testing.expectEqual(compiler.sema.TypeKind.integer, typecheck.exprType(old_result.expr).kind());
    try testing.expectEqual(compiler.sema.TypeKind.integer, typecheck.exprType(ensures_binary.lhs).kind());
    try testing.expectEqual(compiler.sema.TypeKind.bool, typecheck.exprType(assert_stmt.condition).kind());
    try testing.expectEqual(compiler.sema.TypeKind.address, typecheck.exprType(casted_stmt.value.?).kind());
    try testing.expectEqual(compiler.sema.TypeKind.integer, typecheck.exprType(quotient_stmt.value.?).kind());
    try testing.expectEqual(compiler.sema.TypeKind.integer, typecheck.exprType(snapshot_stmt.value.?).kind());
    try testing.expectEqual(compiler.sema.TypeKind.address, typecheck.exprType(addr_stmt.value.?).kind());
    try testing.expectEqual(compiler.sema.TypeKind.bytes, typecheck.exprType(data_stmt.value.?).kind());

    const consteval = try compilation.db.constEval(compilation.root_module_id);
    try testing.expect(consteval.values[quotient_stmt.value.?.index()] != null);
    try testing.expectEqual(@as(i128, 3), try consteval.values[quotient_stmt.value.?.index()].?.integer.toInt(i128));
}

test "compiler render ladder step 1 struct decl struct literal field extract" {
    const source_text =
        \\struct Pair {
        \\    first: u256;
        \\    second: u256;
        \\}
        \\
        \\pub fn build() -> u256 {
        \\    let pair = Pair { first: 1, second: 2 };
        \\    return pair.first;
        \\}
    ;

    const hir_text = try renderHirTextForSource(source_text);
    defer testing.allocator.free(hir_text);

    try testing.expect(std.mem.containsAtLeast(u8, hir_text, 1, "ora.struct_instantiate"));
    try testing.expect(std.mem.containsAtLeast(u8, hir_text, 1, "ora.struct_field_extract"));
}

test "compiler render ladder step 2 add error decl" {
    const source_text =
        \\struct Pair {
        \\    first: u256;
        \\    second: u256;
        \\}
        \\
        \\error Failure(code: u256);
        \\
        \\pub fn build() -> u256 {
        \\    let pair = Pair { first: 1, second: 2 };
        \\    return pair.first;
        \\}
    ;

    const hir_text = try renderHirTextForSource(source_text);
    defer testing.allocator.free(hir_text);

    try testing.expect(std.mem.containsAtLeast(u8, hir_text, 1, "ora.struct.decl"));
    try testing.expect(std.mem.containsAtLeast(u8, hir_text, 1, "ora.error.decl"));
}

test "compiler render ladder step 3 add array literal" {
    const source_text =
        \\struct Pair {
        \\    first: u256;
        \\    second: u256;
        \\}
        \\
        \\error Failure(code: u256);
        \\
        \\pub fn build() -> u256 {
        \\    let items = [1, 2, 3];
        \\    let pair = Pair { first: 1, second: 2 };
        \\    return pair.first;
        \\}
    ;

    const hir_text = try renderHirTextForSource(source_text);
    defer testing.allocator.free(hir_text);

    try testing.expect(std.mem.containsAtLeast(u8, hir_text, 1, "memref.alloca"));
    try testing.expect(std.mem.containsAtLeast(u8, hir_text, 1, "memref.store"));
}

test "compiler render ladder step 4 add array indexing into struct literal" {
    const source_text =
        \\struct Pair {
        \\    first: u256;
        \\    second: u256;
        \\}
        \\
        \\error Failure(code: u256);
        \\
        \\pub fn build() -> u256 {
        \\    let items = [1, 2, 3];
        \\    let pair = Pair { first: items[0], second: items[1] };
        \\    return pair.first;
        \\}
    ;

    const hir_text = try renderHirTextForSource(source_text);
    defer testing.allocator.free(hir_text);

    try testing.expect(std.mem.containsAtLeast(u8, hir_text, 1, "memref.load"));
    try testing.expect(std.mem.containsAtLeast(u8, hir_text, 1, "ora.struct_instantiate"));
}

test "compiler render ladder step 5 add tuple from indexed values" {
    const source_text =
        \\struct Pair {
        \\    first: u256;
        \\    second: u256;
        \\}
        \\
        \\error Failure(code: u256);
        \\
        \\pub fn build() -> u256 {
        \\    let items = [1, 2, 3];
        \\    let coords = (items[0], items[1]);
        \\    let pair = Pair { first: items[0], second: items[1] };
        \\    return pair.first;
        \\}
    ;

    const hir_text = try renderHirTextForSource(source_text);
    defer testing.allocator.free(hir_text);

    try testing.expect(std.mem.containsAtLeast(u8, hir_text, 1, "ora.tuple_create"));
}

test "compiler render ladder switch step a single case no else" {
    const source_text =
        \\struct Pair {
        \\    first: u256;
        \\    second: u256;
        \\}
        \\
        \\error Failure(code: u256);
        \\
        \\pub fn build() -> u256 {
        \\    let items = [1, 2, 3];
        \\    let coords = (items[0], items[1]);
        \\    let pair = Pair { first: items[0], second: items[1] };
        \\    let value = switch (0) {
        \\        0 => 1,
        \\    };
        \\    return value;
        \\}
    ;

    const hir_text = try renderHirTextForSource(source_text);
    defer testing.allocator.free(hir_text);

    try testing.expect(std.mem.containsAtLeast(u8, hir_text, 1, "ora.switch_expr"));
    try testing.expect(!std.mem.containsAtLeast(u8, hir_text, 1, "<<UNKNOWN SSA VALUE>>"));
    try testing.expect(!std.mem.containsAtLeast(u8, hir_text, 1, "<<NULL TYPE>>"));
}

test "compiler render ladder switch step b single case plus else" {
    const source_text =
        \\struct Pair {
        \\    first: u256;
        \\    second: u256;
        \\}
        \\
        \\error Failure(code: u256);
        \\
        \\pub fn build() -> u256 {
        \\    let items = [1, 2, 3];
        \\    let coords = (items[0], items[1]);
        \\    let pair = Pair { first: items[0], second: items[1] };
        \\    let value = switch (0) {
        \\        0 => 1,
        \\        else => 3,
        \\    };
        \\    return value;
        \\}
    ;

    const hir_text = try renderHirTextForSource(source_text);
    defer testing.allocator.free(hir_text);

    try testing.expect(std.mem.containsAtLeast(u8, hir_text, 1, "ora.switch_expr"));
    try testing.expect(!std.mem.containsAtLeast(u8, hir_text, 1, "<<UNKNOWN SSA VALUE>>"));
    try testing.expect(!std.mem.containsAtLeast(u8, hir_text, 1, "<<NULL TYPE>>"));
}

test "compiler render ladder switch step c two cases plus else" {
    const source_text =
        \\struct Pair {
        \\    first: u256;
        \\    second: u256;
        \\}
        \\
        \\error Failure(code: u256);
        \\
        \\pub fn build() -> u256 {
        \\    let items = [1, 2, 3];
        \\    let coords = (items[0], items[1]);
        \\    let pair = Pair { first: items[0], second: items[1] };
        \\    let value = switch (0) {
        \\        0 => 1,
        \\        1 => 2,
        \\        else => 3,
        \\    };
        \\    return value;
        \\}
    ;

    const hir_text = try renderHirTextForSource(source_text);
    defer testing.allocator.free(hir_text);

    try testing.expect(std.mem.containsAtLeast(u8, hir_text, 1, "ora.switch_expr"));
    try testing.expect(!std.mem.containsAtLeast(u8, hir_text, 1, "<<UNKNOWN SSA VALUE>>"));
    try testing.expect(!std.mem.containsAtLeast(u8, hir_text, 1, "<<NULL TYPE>>"));
}

test "compiler render ladder switch step d two cases no else" {
    const source_text =
        \\struct Pair {
        \\    first: u256;
        \\    second: u256;
        \\}
        \\
        \\error Failure(code: u256);
        \\
        \\pub fn build() -> u256 {
        \\    let items = [1, 2, 3];
        \\    let coords = (items[0], items[1]);
        \\    let pair = Pair { first: items[0], second: items[1] };
        \\    let value = switch (0) {
        \\        0 => 1,
        \\        1 => 2,
        \\    };
        \\    return value;
        \\}
    ;

    const hir_text = try renderHirTextForSource(source_text);
    defer testing.allocator.free(hir_text);

    try testing.expect(std.mem.containsAtLeast(u8, hir_text, 1, "ora.switch_expr"));
    try testing.expect(!std.mem.containsAtLeast(u8, hir_text, 1, "<<UNKNOWN SSA VALUE>>"));
    try testing.expect(!std.mem.containsAtLeast(u8, hir_text, 1, "<<NULL TYPE>>"));
}

test "compiler renders minimal two-case switch expression" {
    const source_text =
        \\pub fn build() -> u256 {
        \\    let value = switch (0) {
        \\        0 => 1,
        \\        1 => 2,
        \\    };
        \\    return value;
        \\}
    ;

    const hir_text = try renderHirTextForSource(source_text);
    defer testing.allocator.free(hir_text);

    try testing.expect(std.mem.containsAtLeast(u8, hir_text, 1, "ora.switch_expr"));
    try testing.expect(!std.mem.containsAtLeast(u8, hir_text, 1, "<<UNKNOWN SSA VALUE>>"));
    try testing.expect(!std.mem.containsAtLeast(u8, hir_text, 1, "<<NULL TYPE>>"));
}

test "compiler render ladder step 6 add boolean switch expression" {
    const source_text =
        \\struct Pair {
        \\    first: u256;
        \\    second: u256;
        \\}
        \\
        \\error Failure(code: u256);
        \\
        \\pub fn build() -> u256 {
        \\    let items = [1, 2, 3];
        \\    let coords = (items[0], items[1]);
        \\    let pair = Pair { first: items[0], second: items[1] };
        \\    let value = switch (true) {
        \\        true => 1,
        \\        false => 2,
        \\        else => 3,
        \\    };
        \\    return value;
        \\}
    ;

    const hir_text = try renderHirTextForSource(source_text);
    defer testing.allocator.free(hir_text);

    try testing.expect(std.mem.containsAtLeast(u8, hir_text, 1, "ora.switch_expr"));
    try testing.expect(!std.mem.containsAtLeast(u8, hir_text, 1, "<<UNKNOWN SSA VALUE>>"));
    try testing.expect(!std.mem.containsAtLeast(u8, hir_text, 1, "<<NULL TYPE>>"));
}

test "compiler render ladder step 6b add integer switch expression" {
    const source_text =
        \\struct Pair {
        \\    first: u256;
        \\    second: u256;
        \\}
        \\
        \\error Failure(code: u256);
        \\
        \\pub fn build() -> u256 {
        \\    let items = [1, 2, 3];
        \\    let coords = (items[0], items[1]);
        \\    let pair = Pair { first: items[0], second: items[1] };
        \\    let value = switch (0) {
        \\        0 => 1,
        \\        1 => 2,
        \\        else => 3,
        \\    };
        \\    return value;
        \\}
    ;

    const hir_text = try renderHirTextForSource(source_text);
    defer testing.allocator.free(hir_text);

    try testing.expect(std.mem.containsAtLeast(u8, hir_text, 1, "ora.switch_expr"));
    try testing.expect(!std.mem.containsAtLeast(u8, hir_text, 1, "<<UNKNOWN SSA VALUE>>"));
    try testing.expect(!std.mem.containsAtLeast(u8, hir_text, 1, "<<NULL TYPE>>"));
}

test "compiler render ladder step 7 add error constructor expression" {
    const source_text =
        \\struct Pair {
        \\    first: u256;
        \\    second: u256;
        \\}
        \\
        \\error Failure(code: u256);
        \\
        \\pub fn build() -> u256 {
        \\    let items = [1, 2, 3];
        \\    let coords = (items[0], items[1]);
        \\    let pair = Pair { first: items[0], second: items[1] };
        \\    let value = switch (true) {
        \\        true => 1,
        \\        false => 2,
        \\        else => 3,
        \\    };
        \\    let problem = error.Failure(7);
        \\    return value;
        \\}
    ;

    const hir_text = try renderHirTextForSource(source_text);
    defer testing.allocator.free(hir_text);

    try testing.expect(std.mem.containsAtLeast(u8, hir_text, 1, "ora.error.return"));
}

test "compiler lowers tuple, array, struct, switch, and error return expressions" {
    const source_text =
        \\struct Pair {
        \\    first: u256;
        \\    second: u256;
        \\}
        \\
        \\error Failure(code: u256);
        \\
        \\pub fn build() -> u256 {
        \\    let items = [1, 2, 3];
        \\    let coords = (items[0], items[1]);
        \\    let pair = Pair { first: items[0], second: items[1] };
        \\    let value = switch (true) {
        \\        true => 1,
        \\        false => 2,
        \\        else => 3,
        \\    };
        \\    let problem = error.Failure(7);
        \\    return value;
        \\}
    ;

    var compilation = try compileText(source_text);
    defer compilation.deinit();

    const module = compilation.db.sources.module(compilation.root_module_id);
    const ast_file = try compilation.db.astFile(module.file_id);
    const ast_diags = try compilation.db.astDiagnostics(module.file_id);
    try testing.expect(ast_diags.isEmpty());

    const function = ast_file.item(ast_file.root_items[2]).Function;
    const body = ast_file.body(function.body);
    try testing.expectEqual(@as(usize, 6), body.statements.len);

    const items_stmt = ast_file.statement(body.statements[0]).VariableDecl;
    const coords_stmt = ast_file.statement(body.statements[1]).VariableDecl;
    const pair_stmt = ast_file.statement(body.statements[2]).VariableDecl;
    const value_stmt = ast_file.statement(body.statements[3]).VariableDecl;
    const problem_stmt = ast_file.statement(body.statements[4]).VariableDecl;
    const return_stmt = ast_file.statement(body.statements[5]).Return;

    try testing.expect(ast_file.expression(items_stmt.value.?).* == .ArrayLiteral);
    try testing.expect(ast_file.expression(coords_stmt.value.?).* == .Tuple);
    try testing.expect(ast_file.expression(pair_stmt.value.?).* == .StructLiteral);
    try testing.expect(ast_file.expression(value_stmt.value.?).* == .Switch);
    try testing.expect(ast_file.expression(problem_stmt.value.?).* == .ErrorReturn);

    const items_expr = ast_file.expression(items_stmt.value.?).ArrayLiteral;
    try testing.expectEqual(@as(usize, 3), items_expr.elements.len);

    const coords_expr = ast_file.expression(coords_stmt.value.?).Tuple;
    try testing.expectEqual(@as(usize, 2), coords_expr.elements.len);

    const pair_expr = ast_file.expression(pair_stmt.value.?).StructLiteral;
    try testing.expectEqualStrings("Pair", pair_expr.type_name);
    try testing.expectEqual(@as(usize, 2), pair_expr.fields.len);

    const value_expr = ast_file.expression(value_stmt.value.?).Switch;
    try testing.expectEqual(@as(usize, 2), value_expr.arms.len);
    try testing.expect(value_expr.else_expr != null);

    const problem_expr = ast_file.expression(problem_stmt.value.?).ErrorReturn;
    try testing.expectEqualStrings("Failure", problem_expr.name);
    try testing.expectEqual(@as(usize, 1), problem_expr.args.len);

    const typecheck = try compilation.db.typeCheck(compilation.root_module_id, .{ .item = ast_file.root_items[2] });
    try testing.expectEqual(compiler.sema.TypeKind.array, typecheck.pattern_types[items_stmt.pattern.index()].kind());
    try testing.expectEqual(compiler.sema.TypeKind.tuple, typecheck.pattern_types[coords_stmt.pattern.index()].kind());
    try testing.expectEqual(compiler.sema.TypeKind.struct_, typecheck.pattern_types[pair_stmt.pattern.index()].kind());
    try testing.expectEqual(compiler.sema.TypeKind.integer, typecheck.pattern_types[value_stmt.pattern.index()].kind());
    try testing.expectEqual(compiler.sema.TypeKind.integer, typecheck.exprType(return_stmt.value.?).kind());

    const consteval = try compilation.db.constEval(compilation.root_module_id);
    try testing.expect(consteval.values[value_stmt.value.?.index()] != null);
    try testing.expectEqual(@as(i128, 1), try consteval.values[value_stmt.value.?.index()].?.integer.toInt(i128));

    const hir_result = try compilation.db.lowerToHir(compilation.root_module_id);
    const hir_text = try hir_result.renderText(testing.allocator);
    defer testing.allocator.free(hir_text);

    try testing.expect(std.mem.containsAtLeast(u8, hir_text, 1, "ora.error.return"));
    try testing.expect(!std.mem.containsAtLeast(u8, hir_text, 1, "\"ora.error.return\""));
}

test "compiler lowers grouped struct literal bases" {
    const source_text =
        \\pub fn make() -> Pair {
        \\    let pair = (Pair){ x: 1, y: 2 };
        \\    return pair;
        \\}
    ;

    var compilation = try compileText(source_text);
    defer compilation.deinit();

    const module = compilation.db.sources.module(compilation.root_module_id);
    const ast_file = try compilation.db.astFile(module.file_id);
    const ast_diags = try compilation.db.astDiagnostics(module.file_id);
    try testing.expect(ast_diags.isEmpty());

    const function = ast_file.item(ast_file.root_items[0]).Function;
    const body = ast_file.body(function.body);
    const pair_stmt = ast_file.statement(body.statements[0]).VariableDecl;
    const pair_expr = ast_file.expression(pair_stmt.value.?).StructLiteral;
    try testing.expectEqualStrings("Pair", pair_expr.type_name);
    try testing.expectEqual(@as(usize, 2), pair_expr.fields.len);
}

test "compiler emits AST validation diagnostics for duplicate same-scope names" {
    const source_text =
        \\pub fn helper() -> u256 {
        \\    return 1;
        \\}
        \\pub fn helper() -> u256 {
        \\    return 2;
        \\}
        \\struct Pair {
        \\    left: u256,
        \\    left: bool,
        \\}
    ;

    var compilation = try compileText(source_text);
    defer compilation.deinit();

    const module = compilation.db.sources.module(compilation.root_module_id);
    const ast_diags = try compilation.db.astDiagnostics(module.file_id);

    try testing.expectEqual(@as(usize, 2), ast_diags.len());
    try testing.expectEqualStrings("duplicate item name 'helper' in root scope", ast_diags.items.items[0].message);
    try testing.expectEqualStrings("duplicate struct field name 'left'", ast_diags.items.items[1].message);
}

test "compiler lowers malformed syntax nodes into AST errors" {
    const source_text =
        \\pub fn run() {
        \\    let value: = 1;
        \\    let other = ;
        \\}
    ;

    var compilation = try compileText(source_text);
    defer compilation.deinit();

    const module = compilation.db.sources.module(compilation.root_module_id);
    const ast_file = try compilation.db.astFile(module.file_id);
    const function = ast_file.item(ast_file.root_items[0]).Function;
    const body = ast_file.body(function.body);

    const bad_type_stmt = ast_file.statement(body.statements[0]).VariableDecl;
    try testing.expect(bad_type_stmt.type_expr != null);
    try testing.expect(ast_file.typeExpr(bad_type_stmt.type_expr.?).* == .Error);

    const bad_value_stmt = ast_file.statement(body.statements[1]).VariableDecl;
    try testing.expect(bad_value_stmt.value != null);
    try testing.expect(ast_file.expression(bad_value_stmt.value.?).* == .Error);
}

test "compiler lowers malformed function items into AST errors" {
    const source_text =
        \\pub fn broken {
        \\    return 1;
        \\}
    ;

    var compilation = try compileText(source_text);
    defer compilation.deinit();

    const module = compilation.db.sources.module(compilation.root_module_id);
    const syntax_diags = try compilation.db.syntaxDiagnostics(module.file_id);
    try testing.expect(!syntax_diags.isEmpty());
    const ast_file = try compilation.db.astFile(module.file_id);
    try testing.expectEqual(@as(usize, 1), ast_file.root_items.len);
    try testing.expect(ast_file.item(ast_file.root_items[0]).* == .Error);
}

test "compiler lowers compound assignment syntax through syntax AST path" {
    const source_text =
        \\pub fn bump(x: u256) -> u256 {
        \\    let total = x;
        \\    total += 1;
        \\    return total;
        \\}
    ;

    var compilation = try compileText(source_text);
    defer compilation.deinit();

    const module = compilation.db.sources.module(compilation.root_module_id);
    const ast_file = try compilation.db.astFile(module.file_id);
    const function = ast_file.item(ast_file.root_items[0]).Function;
    const body = ast_file.body(function.body);
    const assign = ast_file.statement(body.statements[1]).Assign;

    try testing.expectEqual(compiler.ast.AssignmentOp.add_assign, assign.op);
}

test "compiler lowers labeled block statements through syntax AST path" {
    const source_text =
        \\pub fn run() -> u256 {
        \\    outer: {
        \\        let value = 1;
        \\    }
        \\    return 0;
        \\}
    ;

    var compilation = try compileText(source_text);
    defer compilation.deinit();

    const module = compilation.db.sources.module(compilation.root_module_id);
    const ast_file = try compilation.db.astFile(module.file_id);
    const function = ast_file.item(ast_file.root_items[0]).Function;
    const body = ast_file.body(function.body);
    const labeled = ast_file.statement(body.statements[0]).LabeledBlock;

    try testing.expectEqualStrings("outer", labeled.label);
    try testing.expectEqual(@as(usize, 1), ast_file.body(labeled.body).statements.len);
}

test "compiler lowers lock and unlock statements through syntax AST and HIR paths" {
    const source_text =
        \\contract Vault {
        \\    storage balances: u256;
        \\
        \\    pub fn run() {
        \\        @lock(balances);
        \\        @unlock(balances);
        \\    }
        \\}
    ;

    var compilation = try compileText(source_text);
    defer compilation.deinit();

    const module = compilation.db.sources.module(compilation.root_module_id);
    const ast_file = try compilation.db.astFile(module.file_id);
    const contract = ast_file.item(ast_file.root_items[0]).Contract;
    const function = ast_file.item(contract.members[1]).Function;
    const body = ast_file.body(function.body);
    try testing.expect(ast_file.statement(body.statements[0]).* == .Lock);
    try testing.expect(ast_file.statement(body.statements[1]).* == .Unlock);

    const hir_result = try compilation.db.lowerToHir(compilation.root_module_id);
    const hir_text = try hir_result.renderText(testing.allocator);
    defer testing.allocator.free(hir_text);

    try testing.expect(std.mem.containsAtLeast(u8, hir_text, 1, "ora.lock"));
    try testing.expect(std.mem.containsAtLeast(u8, hir_text, 1, "ora.unlock"));
    try testing.expect(!std.mem.containsAtLeast(u8, hir_text, 1, "ora.lock_placeholder"));
    try testing.expect(!std.mem.containsAtLeast(u8, hir_text, 1, "ora.unlock_placeholder"));
}

test "compiler emits tstore guard before guarded storage writes" {
    const source_text =
        \\contract GuardedWrites {
        \\    storage balances: u256;
        \\
        \\    pub fn touch() {
        \\        @lock(balances);
        \\        balances = 1;
        \\    }
        \\}
    ;

    var compilation = try compiler.compileSource(testing.allocator, "guarded-write.ora", source_text);
    defer compilation.deinit();

    const hir_result = try compilation.db.lowerToHir(compilation.root_module_id);
    const hir_text = try hir_result.renderText(testing.allocator);
    defer testing.allocator.free(hir_text);

    try testing.expect(std.mem.containsAtLeast(u8, hir_text, 1, "ora.lock"));
    try testing.expect(std.mem.containsAtLeast(u8, hir_text, 1, "ora.tstore.guard"));
    try testing.expect(std.mem.containsAtLeast(u8, hir_text, 1, "ora.sstore"));
    try testing.expect(!std.mem.containsAtLeast(u8, hir_text, 1, "ora.lock_placeholder"));
}

test "compiler lowers grouped lock paths through real lock ops" {
    const source_text =
        \\contract Vault {
        \\    storage balances: map<address, u256>;
        \\
        \\    pub fn run(user: address) {
        \\        @lock((balances[user]));
        \\        @unlock((balances[user]));
        \\    }
        \\}
    ;

    var compilation = try compileText(source_text);
    defer compilation.deinit();

    const hir_result = try compilation.db.lowerToHir(compilation.root_module_id);
    const hir_text = try hir_result.renderText(testing.allocator);
    defer testing.allocator.free(hir_text);

    try testing.expect(std.mem.containsAtLeast(u8, hir_text, 1, "ora.lock"));
    try testing.expect(std.mem.containsAtLeast(u8, hir_text, 1, "ora.unlock"));
    try testing.expect(!std.mem.containsAtLeast(u8, hir_text, 1, "ora.lock_placeholder"));
    try testing.expect(!std.mem.containsAtLeast(u8, hir_text, 1, "ora.unlock_placeholder"));
}

test "compiler lowers simple memref-backed for loops through scf.for" {
    const source_text =
        \\pub fn scan(values: slice[u256]) {
        \\    for (values) |value, index| {
        \\        assert(value >= index, "ordered");
        \\    }
        \\}
    ;

    var compilation = try compileText(source_text);
    defer compilation.deinit();

    const hir_result = try compilation.db.lowerToHir(compilation.root_module_id);
    const hir_text = try hir_result.renderText(testing.allocator);
    defer testing.allocator.free(hir_text);

    try testing.expect(std.mem.containsAtLeast(u8, hir_text, 1, "scf.for"));
    try testing.expect(std.mem.containsAtLeast(u8, hir_text, 1, "memref.load"));
    try testing.expect(std.mem.containsAtLeast(u8, hir_text, 1, "arith.index_castui"));
    try testing.expect(!std.mem.containsAtLeast(u8, hir_text, 1, "ora.for_placeholder"));
}

test "compiler threads carried locals through scf.for iter args" {
    const source_text =
        \\pub fn sum(values: slice[u256]) -> u256 {
        \\    let total = 0;
        \\    for (values) |value, index| {
        \\        total = total + value + index;
        \\    }
        \\    return total;
        \\}
    ;

    var compilation = try compileText(source_text);
    defer compilation.deinit();

    const hir_result = try compilation.db.lowerToHir(compilation.root_module_id);
    const hir_text = try hir_result.renderText(testing.allocator);
    defer testing.allocator.free(hir_text);

    try testing.expect(std.mem.containsAtLeast(u8, hir_text, 1, "scf.for"));
    try testing.expect(std.mem.containsAtLeast(u8, hir_text, 1, "memref.load"));
    try testing.expect(std.mem.containsAtLeast(u8, hir_text, 1, "arith.index_castui"));
    try testing.expect(std.mem.count(u8, hir_text, "arith.addi") >= 2);
    try testing.expect(!std.mem.containsAtLeast(u8, hir_text, 1, "ora.for_placeholder"));
}

test "compiler lowers real HIR for loops with break and continue" {
    const source_text =
        \\pub fn count(values: slice[u256]) -> u256 {
        \\    let continued = 0;
        \\    for (values) |value| {
        \\        continued = continued + value;
        \\        continue;
        \\    }
        \\    let stopped = 0;
        \\    for (values) |value| {
        \\        stopped = stopped + value;
        \\        break;
        \\    }
        \\    return continued + stopped;
        \\}
    ;

    var compilation = try compileText(source_text);
    defer compilation.deinit();

    const hir_result = try compilation.db.lowerToHir(compilation.root_module_id);
    const hir_text = try hir_result.renderText(testing.allocator);
    defer testing.allocator.free(hir_text);

    try testing.expect(std.mem.containsAtLeast(u8, hir_text, 2, "scf.for"));
    try testing.expect(std.mem.containsAtLeast(u8, hir_text, 1, "scf.if"));
    try testing.expect(std.mem.containsAtLeast(u8, hir_text, 1, "memref.alloca"));
    try testing.expect(!std.mem.containsAtLeast(u8, hir_text, 1, "ora.for_placeholder"));
}

test "compiler lowers for invariants through ora.invariant" {
    const source_text =
        \\pub fn scan(values: slice[u256]) {
        \\    for (values) |value, index|
        \\        invariant value >= index;
        \\    {
        \\        assert(value >= index, "ordered");
        \\    }
        \\}
    ;

    var compilation = try compileText(source_text);
    defer compilation.deinit();

    const hir_result = try compilation.db.lowerToHir(compilation.root_module_id);
    const hir_text = try hir_result.renderText(testing.allocator);
    defer testing.allocator.free(hir_text);

    try testing.expect(std.mem.containsAtLeast(u8, hir_text, 1, "scf.for"));
    try testing.expect(std.mem.containsAtLeast(u8, hir_text, 1, "ora.invariant"));
    try testing.expect(!std.mem.containsAtLeast(u8, hir_text, 1, "ora.for_placeholder"));
}

test "compiler lowers direct map index load and store through real map ops" {
    const source_text =
        \\contract Maps {
        \\    storage table: map<u256, u256>;
        \\
        \\    pub fn touch() -> u256 {
        \\        table[1] = 2;
        \\        return table[1];
        \\    }
        \\}
    ;

    var compilation = try compiler.compileSource(testing.allocator, "maps.ora", source_text);
    defer compilation.deinit();

    const hir_result = try compilation.db.lowerToHir(compilation.root_module_id);
    const hir_text = try hir_result.renderText(testing.allocator);
    defer testing.allocator.free(hir_text);

    try testing.expect(std.mem.containsAtLeast(u8, hir_text, 1, "ora.map_store"));
    try testing.expect(std.mem.containsAtLeast(u8, hir_text, 1, "ora.map_get"));
    try testing.expect(!std.mem.containsAtLeast(u8, hir_text, 1, "ora.index_access"));
}

test "compiler rethreads nested map assignment to outer map" {
    const source_text =
        \\contract Test {
        \\    storage allowances: map<address, map<address, u256>>;
        \\
        \\    pub fn setAllowance(owner: address, spender: address, amount: u256) {
        \\        allowances[owner][spender] = amount;
        \\    }
        \\}
    ;

    var compilation = try compiler.compileSource(testing.allocator, "nested-map-store.ora", source_text);
    defer compilation.deinit();

    const hir_result = try compilation.db.lowerToHir(compilation.root_module_id);
    const hir_text = try hir_result.renderText(testing.allocator);
    defer testing.allocator.free(hir_text);

    try testing.expect(std.mem.count(u8, hir_text, "ora.map_store") >= 2);
    try testing.expect(std.mem.containsAtLeast(u8, hir_text, 1, "ora.map_get"));
    try testing.expect(!std.mem.containsAtLeast(u8, hir_text, 1, "ora.index_access"));
}

test "compiler lowers slice index load and store through memref ops" {
    const source_text =
        \\pub fn touch(values: slice[u256]) -> u256 {
        \\    values[0] = 7;
        \\    return values[0];
        \\}
    ;

    var compilation = try compiler.compileSource(testing.allocator, "slice-index.ora", source_text);
    defer compilation.deinit();

    const hir_result = try compilation.db.lowerToHir(compilation.root_module_id);
    const hir_text = try hir_result.renderText(testing.allocator);
    defer testing.allocator.free(hir_text);

    try testing.expect(std.mem.containsAtLeast(u8, hir_text, 1, "memref.store"));
    try testing.expect(std.mem.containsAtLeast(u8, hir_text, 1, "memref.load"));
    try testing.expect(std.mem.containsAtLeast(u8, hir_text, 1, "arith.index_castui"));
    try testing.expect(std.mem.containsAtLeast(u8, hir_text, 1, "memref<?xi256>"));
    try testing.expect(!std.mem.containsAtLeast(u8, hir_text, 1, "ora.index_access"));
}

test "compiler lowers array literals through memref allocation and stores" {
    const source_text =
        \\pub fn read_first() -> u256 {
        \\    let items = [1, 2, 3];
        \\    return items[0];
        \\}
    ;

    var compilation = try compiler.compileSource(testing.allocator, "array-literal.ora", source_text);
    defer compilation.deinit();

    const hir_result = try compilation.db.lowerToHir(compilation.root_module_id);
    const hir_text = try hir_result.renderText(testing.allocator);
    defer testing.allocator.free(hir_text);

    try testing.expect(std.mem.containsAtLeast(u8, hir_text, 1, "memref.alloca"));
    try testing.expect(std.mem.count(u8, hir_text, "memref.store") >= 3);
    try testing.expect(std.mem.containsAtLeast(u8, hir_text, 1, "memref.load"));
    try testing.expect(std.mem.containsAtLeast(u8, hir_text, 1, "memref<3xi256>"));
    try testing.expect(!std.mem.containsAtLeast(u8, hir_text, 1, "ora.array.create"));
}

test "compiler lowers destructuring bindings through struct field extracts" {
    const source_text =
        \\struct Pair {
        \\    left: u256;
        \\    right: u256;
        \\}
        \\
        \\pub fn sum() -> u256 {
        \\    let pair = Pair { left: 1, right: 2 };
        \\    let .{ left: a, right: b } = pair;
        \\    return a + b;
        \\}
    ;

    var compilation = try compiler.compileSource(testing.allocator, "destructure-bind.ora", source_text);
    defer compilation.deinit();

    const hir_result = try compilation.db.lowerToHir(compilation.root_module_id);
    const hir_text = try hir_result.renderText(testing.allocator);
    defer testing.allocator.free(hir_text);

    try testing.expect(std.mem.count(u8, hir_text, "ora.struct_field_extract") >= 2);
    try testing.expect(!std.mem.containsAtLeast(u8, hir_text, 1, "\"ora.field_access\""));
}

test "compiler lowers destructuring assignment through struct field extracts" {
    const source_text =
        \\struct Pair {
        \\    left: u256;
        \\    right: u256;
        \\}
        \\
        \\pub fn sum() -> u256 {
        \\    let left = 0;
        \\    let right = 0;
        \\    .{ left, right } = Pair { left: 4, right: 5 };
        \\    return left + right;
        \\}
    ;

    var compilation = try compiler.compileSource(testing.allocator, "destructure-assign.ora", source_text);
    defer compilation.deinit();

    const hir_result = try compilation.db.lowerToHir(compilation.root_module_id);
    const hir_text = try hir_result.renderText(testing.allocator);
    defer testing.allocator.free(hir_text);

    try testing.expect(std.mem.count(u8, hir_text, "ora.struct_field_extract") >= 2);
    try testing.expect(std.mem.containsAtLeast(u8, hir_text, 1, "arith.addi"));
    try testing.expect(!std.mem.containsAtLeast(u8, hir_text, 1, "\"ora.field_access\""));
}

test "compiler lowers fallback break and continue through real ops" {
    const source_text =
        \\pub fn stop() {
        \\    break;
        \\    continue;
        \\}
    ;

    var compilation = try compiler.compileSource(testing.allocator, "fallback-jumps.ora", source_text);
    defer compilation.deinit();

    const hir_result = try compilation.db.lowerToHir(compilation.root_module_id);
    const hir_text = try hir_result.renderText(testing.allocator);
    defer testing.allocator.free(hir_text);

    try testing.expect(std.mem.containsAtLeast(u8, hir_text, 1, "ora.break"));
    try testing.expect(std.mem.containsAtLeast(u8, hir_text, 1, "ora.continue"));
    try testing.expect(!std.mem.containsAtLeast(u8, hir_text, 1, "\"ora.continue\""));
}

test "compiler lowers comptime block expressions through syntax AST path" {
    const source_text =
        \\pub fn run() -> u256 {
        \\    return comptime {
        \\        let value = 1;
        \\        value;
        \\    };
        \\}
    ;

    var compilation = try compileText(source_text);
    defer compilation.deinit();

    const module = compilation.db.sources.module(compilation.root_module_id);
    const ast_file = try compilation.db.astFile(module.file_id);
    const function = ast_file.item(ast_file.root_items[0]).Function;
    const body = ast_file.body(function.body);
    const ret = ast_file.statement(body.statements[0]).Return;
    try testing.expect(ret.value != null);
    const comptime_expr = ast_file.expression(ret.value.?).Comptime;
    try testing.expectEqual(@as(usize, 2), ast_file.body(comptime_expr.body).statements.len);
}

test "compiler keeps malformed declaration shapes on syntax lowering path" {
    const source_text =
        \\struct Pair {
        \\    left:;
        \\}
        \\
        \\pub fn broken(value) -> u256 {
        \\    return 0;
        \\}
    ;

    var compilation = try compileText(source_text);
    defer compilation.deinit();

    const module = compilation.db.sources.module(compilation.root_module_id);
    const ast_diags = try compilation.db.astDiagnostics(module.file_id);
    try testing.expect(!ast_diags.isEmpty());

    const ast_file = try compilation.db.astFile(module.file_id);
    const pair_item = ast_file.item(ast_file.root_items[0]).Struct;
    try testing.expect(ast_file.typeExpr(pair_item.fields[0].type_expr).* == .Error);

    const function = ast_file.item(ast_file.root_items[1]).Function;
    try testing.expect(ast_file.typeExpr(function.parameters[0].type_expr).* == .Error);
}

test "compiler reports sema diagnostics for unresolved names and invalid operations" {
    const source_text =
        \\pub fn broken(flag: bool, value: u256) -> u256 {
        \\    let missing = nope;
        \\    let bad_not = !value;
        \\    let bad_add = flag + value;
        \\    let bad_field = value.balance;
        \\    let bad_index = flag[0];
        \\    let bad_call = value();
        \\    return 0;
        \\}
    ;

    var compilation = try compileText(source_text);
    defer compilation.deinit();

    const module = compilation.db.sources.module(compilation.root_module_id);
    const ast_file = try compilation.db.astFile(module.file_id);
    const function = ast_file.item(ast_file.root_items[0]).Function;

    const resolution_diags = try compilation.db.resolutionDiagnostics(compilation.root_module_id);
    try testing.expectEqual(@as(usize, 1), resolution_diags.len());
    try testing.expectEqualStrings("undefined name 'nope'", resolution_diags.items.items[0].message);

    const type_diags = try compilation.db.typeCheckDiagnostics(compilation.root_module_id, .{ .item = ast_file.root_items[0] });
    try testing.expectEqual(@as(usize, 5), type_diags.len());
    try testing.expectEqualStrings("invalid unary operator '!' for type 'u256'", type_diags.items.items[0].message);
    try testing.expectEqualStrings("invalid binary operator '+' for types 'bool' and 'u256'", type_diags.items.items[1].message);
    try testing.expectEqualStrings("type 'u256' has no field 'balance'", type_diags.items.items[2].message);
    try testing.expectEqualStrings("type 'bool' is not indexable", type_diags.items.items[3].message);
    try testing.expectEqualStrings("type 'u256' is not callable", type_diags.items.items[4].message);

    const typecheck = try compilation.db.typeCheck(compilation.root_module_id, .{ .item = ast_file.root_items[0] });
    const body = ast_file.body(function.body);
    const missing_stmt = ast_file.statement(body.statements[0]).VariableDecl;
    const bad_not_stmt = ast_file.statement(body.statements[1]).VariableDecl;
    const bad_add_stmt = ast_file.statement(body.statements[2]).VariableDecl;
    const bad_field_stmt = ast_file.statement(body.statements[3]).VariableDecl;
    const bad_index_stmt = ast_file.statement(body.statements[4]).VariableDecl;
    const bad_call_stmt = ast_file.statement(body.statements[5]).VariableDecl;

    try testing.expectEqual(compiler.sema.TypeKind.unknown, typecheck.pattern_types[missing_stmt.pattern.index()].kind());
    try testing.expectEqual(compiler.sema.TypeKind.unknown, typecheck.pattern_types[bad_not_stmt.pattern.index()].kind());
    try testing.expectEqual(compiler.sema.TypeKind.unknown, typecheck.pattern_types[bad_add_stmt.pattern.index()].kind());
    try testing.expectEqual(compiler.sema.TypeKind.unknown, typecheck.pattern_types[bad_field_stmt.pattern.index()].kind());
    try testing.expectEqual(compiler.sema.TypeKind.unknown, typecheck.pattern_types[bad_index_stmt.pattern.index()].kind());
    try testing.expectEqual(compiler.sema.TypeKind.unknown, typecheck.pattern_types[bad_call_stmt.pattern.index()].kind());
}

test "compiler suppresses cascading diagnostics from unknown expressions" {
    const source_text =
        \\pub fn broken() -> u256 {
        \\    let value = missing.field + 1;
        \\    return 0;
        \\}
    ;

    var compilation = try compileText(source_text);
    defer compilation.deinit();

    const module = compilation.db.sources.module(compilation.root_module_id);
    const ast_file = try compilation.db.astFile(module.file_id);
    const function = ast_file.item(ast_file.root_items[0]).Function;

    const resolution_diags = try compilation.db.resolutionDiagnostics(compilation.root_module_id);
    try testing.expectEqual(@as(usize, 1), resolution_diags.len());
    try testing.expectEqualStrings("undefined name 'missing'", resolution_diags.items.items[0].message);

    const type_diags = try compilation.db.typeCheckDiagnostics(compilation.root_module_id, .{ .item = ast_file.root_items[0] });
    try testing.expectEqual(@as(usize, 0), type_diags.len());

    const typecheck = try compilation.db.typeCheck(compilation.root_module_id, .{ .item = ast_file.root_items[0] });
    const body = ast_file.body(function.body);
    const value_stmt = ast_file.statement(body.statements[0]).VariableDecl;
    try testing.expectEqual(compiler.sema.TypeKind.unknown, typecheck.pattern_types[value_stmt.pattern.index()].kind());
}

test "compiler reports sema diagnostics for declaration assignment and return mismatches" {
    const source_text =
        \\contract Vault {
        \\    storage var total: u256 = true;
        \\}
        \\
        \\const LIMIT: u256 = false;
        \\
        \\pub fn broken(flag: bool) -> u256 {
        \\    let a: u256 = flag;
        \\    let b = 1;
        \\    b = flag;
        \\    return flag;
        \\}
    ;

    var compilation = try compileText(source_text);
    defer compilation.deinit();

    const module = compilation.db.sources.module(compilation.root_module_id);
    const ast_file = try compilation.db.astFile(module.file_id);
    const type_diags = try compilation.db.typeCheckDiagnostics(compilation.root_module_id, .{ .item = ast_file.root_items[2] });

    try testing.expectEqual(@as(usize, 5), type_diags.len());
    try testing.expectEqualStrings("field 'total' expects type 'u256', found 'bool'", type_diags.items.items[0].message);
    try testing.expectEqualStrings("constant 'LIMIT' expects type 'u256', found 'bool'", type_diags.items.items[1].message);
    try testing.expectEqualStrings("declaration expects type 'u256', found 'bool'", type_diags.items.items[2].message);
    try testing.expectEqualStrings("assignment expects type 'integer', found 'bool'", type_diags.items.items[3].message);
    try testing.expectEqualStrings("return expects type 'u256', found 'bool'", type_diags.items.items[4].message);

    const function = ast_file.item(ast_file.root_items[2]).Function;
    const full_typecheck = try compilation.db.typeCheck(compilation.root_module_id, .{ .item = ast_file.root_items[2] });
    const body = ast_file.body(function.body);
    const a_stmt = ast_file.statement(body.statements[0]).VariableDecl;
    const ret_stmt = ast_file.statement(body.statements[3]).Return;

    try testing.expectEqual(compiler.sema.TypeKind.integer, full_typecheck.pattern_types[a_stmt.pattern.index()].kind());
    try testing.expectEqual(compiler.sema.TypeKind.bool, full_typecheck.exprType(ret_stmt.value.?).kind());
}

test "compiler reports sema diagnostics for control flow conditions and switch branch mismatches" {
    const source_text =
        \\pub fn broken(flag: u256) -> u256 {
        \\    if (1) {
        \\        let a = 1;
        \\    }
        \\    while (2) {
        \\        break;
        \\    }
        \\    assert(3);
        \\    assume(4);
        \\    let value = switch (flag) {
        \\        0 => 1,
        \\        1 => false,
        \\        else => 3,
        \\    };
        \\    return value;
        \\}
    ;

    var compilation = try compileText(source_text);
    defer compilation.deinit();

    const module = compilation.db.sources.module(compilation.root_module_id);
    const ast_file = try compilation.db.astFile(module.file_id);
    const type_diags = try compilation.db.typeCheckDiagnostics(compilation.root_module_id, .{ .item = ast_file.root_items[0] });

    try testing.expectEqual(@as(usize, 5), type_diags.len());
    try testing.expectEqualStrings("if condition must be 'bool', found 'integer'", type_diags.items.items[0].message);
    try testing.expectEqualStrings("while condition must be 'bool', found 'integer'", type_diags.items.items[1].message);
    try testing.expectEqualStrings("assert condition must be 'bool', found 'integer'", type_diags.items.items[2].message);
    try testing.expectEqualStrings("assume condition must be 'bool', found 'integer'", type_diags.items.items[3].message);
    try testing.expectEqualStrings("switch expression branches have incompatible types 'integer' and 'bool'", type_diags.items.items[4].message);

    const function = ast_file.item(ast_file.root_items[0]).Function;
    const body = ast_file.body(function.body);
    const value_stmt = ast_file.statement(body.statements[4]).VariableDecl;
    const ret_stmt = ast_file.statement(body.statements[5]).Return;
    const typecheck = try compilation.db.typeCheck(compilation.root_module_id, .{ .item = ast_file.root_items[0] });

    try testing.expectEqual(compiler.sema.TypeKind.unknown, typecheck.pattern_types[value_stmt.pattern.index()].kind());
    try testing.expectEqual(compiler.sema.TypeKind.unknown, typecheck.exprType(ret_stmt.value.?).kind());
}

test "compiler merges compatible integer branch types to the wider integer" {
    const source_text =
        \\pub fn widen(flag: bool, small: u8, big: u256) -> u256 {
        \\    let value = switch (flag) {
        \\        true => small,
        \\        else => big,
        \\    };
        \\    return value;
        \\}
    ;

    var compilation = try compileText(source_text);
    defer compilation.deinit();

    const module = compilation.db.sources.module(compilation.root_module_id);
    const ast_file = try compilation.db.astFile(module.file_id);
    const type_diags = try compilation.db.typeCheckDiagnostics(compilation.root_module_id, .{ .item = ast_file.root_items[0] });
    try testing.expect(type_diags.isEmpty());

    const function = ast_file.item(ast_file.root_items[0]).Function;
    const body = ast_file.body(function.body);
    const value_stmt = ast_file.statement(body.statements[0]).VariableDecl;
    const ret_stmt = ast_file.statement(body.statements[1]).Return;
    const typecheck = try compilation.db.typeCheck(compilation.root_module_id, .{ .item = ast_file.root_items[0] });

    try testing.expectEqual(compiler.sema.TypeKind.integer, typecheck.pattern_types[value_stmt.pattern.index()].kind());
    try testing.expectEqualStrings("u256", typecheck.pattern_types[value_stmt.pattern.index()].name().?);
    try testing.expectEqual(compiler.sema.TypeKind.integer, typecheck.exprType(ret_stmt.value.?).kind());
    try testing.expectEqualStrings("u256", typecheck.exprType(ret_stmt.value.?).name().?);
}

test "compiler types and lowers all-constant switch expressions without else" {
    const source_text =
        \\pub fn choose(tag: u256) -> u256 {
        \\    let value = switch (tag) {
        \\        0 => 1,
        \\        1 => 2,
        \\    };
        \\    return value;
        \\}
    ;

    var compilation = try compileText(source_text);
    defer compilation.deinit();

    const module = compilation.db.sources.module(compilation.root_module_id);
    const ast_file = try compilation.db.astFile(module.file_id);
    const type_diags = try compilation.db.typeCheckDiagnostics(compilation.root_module_id, .{ .item = ast_file.root_items[0] });
    try testing.expectEqual(@as(usize, 0), type_diags.len());

    const function = ast_file.item(ast_file.root_items[0]).Function;
    const body = ast_file.body(function.body);
    const value_stmt = ast_file.statement(body.statements[0]).VariableDecl;
    const ret_stmt = ast_file.statement(body.statements[1]).Return;
    const typecheck = try compilation.db.typeCheck(compilation.root_module_id, .{ .item = ast_file.root_items[0] });

    try testing.expectEqual(compiler.sema.TypeKind.integer, typecheck.pattern_types[value_stmt.pattern.index()].kind());
    try testing.expectEqual(compiler.sema.TypeKind.integer, typecheck.exprType(ret_stmt.value.?).kind());

    const hir_result = try compilation.db.lowerToHir(compilation.root_module_id);
    const hir_text = try hir_result.renderText(testing.allocator);
    defer testing.allocator.free(hir_text);

    try testing.expect(std.mem.containsAtLeast(u8, hir_text, 1, "ora.switch_expr"));
    try testing.expect(!std.mem.containsAtLeast(u8, hir_text, 1, "\"ora.switch_expr\""));
}

test "compiler reports heterogeneous array literals and keeps tuple element types" {
    const source_text =
        \\pub fn build(flag: bool, small: u8, big: u256) -> bool {
        \\    let ok = [1, 2, 3];
        \\    let widened = [small, big, 3];
        \\    let bad = [1, false];
        \\    let pair = (flag, 7);
        \\    return pair[0];
        \\}
    ;

    var compilation = try compileText(source_text);
    defer compilation.deinit();

    const module = compilation.db.sources.module(compilation.root_module_id);
    const ast_file = try compilation.db.astFile(module.file_id);
    const type_diags = try compilation.db.typeCheckDiagnostics(compilation.root_module_id, .{ .item = ast_file.root_items[0] });
    try testing.expectEqual(@as(usize, 1), type_diags.len());
    try testing.expectEqualStrings("array literal elements have incompatible types 'integer' and 'bool'", type_diags.items.items[0].message);

    const function = ast_file.item(ast_file.root_items[0]).Function;
    const body = ast_file.body(function.body);
    const typecheck = try compilation.db.typeCheck(compilation.root_module_id, .{ .item = ast_file.root_items[0] });

    const ok_pattern = findVariablePatternByName(ast_file, body.statements, "ok").?;
    const widened_pattern = findVariablePatternByName(ast_file, body.statements, "widened").?;

    const ok_type = typecheck.pattern_types[ok_pattern.index()];
    try testing.expectEqual(compiler.sema.TypeKind.array, ok_type.kind());
    try testing.expectEqual(compiler.sema.TypeKind.integer, ok_type.elementType().?.kind());

    const widened_type = typecheck.pattern_types[widened_pattern.index()];
    try testing.expectEqual(compiler.sema.TypeKind.array, widened_type.kind());
    try testing.expectEqual(compiler.sema.TypeKind.integer, widened_type.elementType().?.kind());
    try testing.expectEqualStrings("u256", widened_type.elementType().?.name().?);

    const pair_pattern = findVariablePatternByName(ast_file, body.statements, "pair").?;
    const pair_type = typecheck.pattern_types[pair_pattern.index()];
    try testing.expectEqual(compiler.sema.TypeKind.tuple, pair_type.kind());
    try testing.expectEqual(@as(usize, 2), pair_type.tupleTypes().len);
    try testing.expectEqual(compiler.sema.TypeKind.bool, pair_type.tupleTypes()[0].kind());
    try testing.expectEqual(compiler.sema.TypeKind.integer, pair_type.tupleTypes()[1].kind());
    const ret_stmt = ast_file.statement(body.statements[body.statements.len - 1]).Return;
    try testing.expectEqual(compiler.sema.TypeKind.bool, typecheck.exprType(ret_stmt.value.?).kind());
}

test "compiler uses const-evaluated tuple indices during type checking" {
    const source_text =
        \\pub fn pick(flag: bool) -> bool {
        \\    let pair = (flag, 7);
        \\    return pair[1 - 1];
        \\}
    ;

    var compilation = try compileText(source_text);
    defer compilation.deinit();

    const module = compilation.db.sources.module(compilation.root_module_id);
    const ast_file = try compilation.db.astFile(module.file_id);
    const type_diags = try compilation.db.typeCheckDiagnostics(compilation.root_module_id, .{ .item = ast_file.root_items[0] });
    try testing.expect(type_diags.isEmpty());

    const function = ast_file.item(ast_file.root_items[0]).Function;
    const body = ast_file.body(function.body);
    const ret_stmt = ast_file.statement(body.statements[1]).Return;
    const typecheck = try compilation.db.typeCheck(compilation.root_module_id, .{ .item = ast_file.root_items[0] });

    try testing.expectEqual(compiler.sema.TypeKind.bool, typecheck.exprType(ret_stmt.value.?).kind());
}

test "compiler reports invalid constant shift amounts" {
    const source_text =
        \\pub fn shift(v: u8) -> u8 {
        \\    let ok = v << 7;
        \\    let bad = v << 8;
        \\    return ok;
        \\}
    ;

    var compilation = try compileText(source_text);
    defer compilation.deinit();

    const module = compilation.db.sources.module(compilation.root_module_id);
    const ast_file = try compilation.db.astFile(module.file_id);
    const type_diags = try compilation.db.typeCheckDiagnostics(compilation.root_module_id, .{ .item = ast_file.root_items[0] });
    try testing.expectEqual(@as(usize, 1), type_diags.len());
    try testing.expectEqualStrings("shift amount 8 out of range for type 'u8'", type_diags.items.items[0].message);

    const function = ast_file.item(ast_file.root_items[0]).Function;
    const body = ast_file.body(function.body);
    const typecheck = try compilation.db.typeCheck(compilation.root_module_id, .{ .item = ast_file.root_items[0] });
    const ok_pattern = findVariablePatternByName(ast_file, body.statements, "ok").?;
    const bad_pattern = findVariablePatternByName(ast_file, body.statements, "bad").?;

    try testing.expectEqual(compiler.sema.TypeKind.integer, typecheck.pattern_types[ok_pattern.index()].kind());
    try testing.expectEqualStrings("u8", typecheck.pattern_types[ok_pattern.index()].name().?);
    try testing.expectEqual(compiler.sema.TypeKind.unknown, typecheck.pattern_types[bad_pattern.index()].kind());
}

test "compiler reports integer constant overflow against declared widths" {
    const source_text =
        \\storage var total: u8 = 256;
        \\const LIMIT: u8 = 256;
        \\pub fn narrow() -> u8 {
        \\    let a: u8 = 256;
        \\    let b: u8 = 1;
        \\    b = 256;
        \\    return 256;
        \\}
    ;

    var compilation = try compileText(source_text);
    defer compilation.deinit();

    const module = compilation.db.sources.module(compilation.root_module_id);
    const ast_file = try compilation.db.astFile(module.file_id);
    const type_diags = try compilation.db.typeCheckDiagnostics(compilation.root_module_id, .{ .item = ast_file.root_items[2] });
    try testing.expectEqual(@as(usize, 5), type_diags.len());
    try testing.expectEqualStrings("constant value 256 does not fit in type 'u8'", type_diags.items.items[0].message);
    try testing.expectEqualStrings("constant value 256 does not fit in type 'u8'", type_diags.items.items[1].message);
    try testing.expectEqualStrings("constant value 256 does not fit in type 'u8'", type_diags.items.items[2].message);
    try testing.expectEqualStrings("constant value 256 does not fit in type 'u8'", type_diags.items.items[3].message);
    try testing.expectEqualStrings("constant value 256 does not fit in type 'u8'", type_diags.items.items[4].message);
}

test "compiler reports constant cast overflow against target integer widths" {
    const source_text =
        \\pub fn casted() -> u8 {
        \\    let ok = @cast(u8, 255);
        \\    let bad = @cast(u8, 256);
        \\    return ok;
        \\}
    ;

    var compilation = try compileText(source_text);
    defer compilation.deinit();

    const module = compilation.db.sources.module(compilation.root_module_id);
    const ast_file = try compilation.db.astFile(module.file_id);
    const type_diags = try compilation.db.typeCheckDiagnostics(compilation.root_module_id, .{ .item = ast_file.root_items[0] });
    try testing.expectEqual(@as(usize, 1), type_diags.len());
    try testing.expectEqualStrings("constant value 256 does not fit in cast target type 'u8'", type_diags.items.items[0].message);

    const function = ast_file.item(ast_file.root_items[0]).Function;
    const body = ast_file.body(function.body);
    const typecheck = try compilation.db.typeCheck(compilation.root_module_id, .{ .item = ast_file.root_items[0] });
    const ok_pattern = findVariablePatternByName(ast_file, body.statements, "ok").?;
    const bad_pattern = findVariablePatternByName(ast_file, body.statements, "bad").?;

    try testing.expectEqual(compiler.sema.TypeKind.integer, typecheck.pattern_types[ok_pattern.index()].kind());
    try testing.expectEqualStrings("u8", typecheck.pattern_types[ok_pattern.index()].name().?);
    try testing.expectEqual(compiler.sema.TypeKind.unknown, typecheck.pattern_types[bad_pattern.index()].kind());
}

test "compiler const eval preserves integers wider than i128" {
    const source_text =
        \\pub fn huge() -> u256 {
        \\    return 340282366920938463463374607431768211456;
        \\}
    ;

    var compilation = try compileText(source_text);
    defer compilation.deinit();

    const module = compilation.db.sources.module(compilation.root_module_id);
    const ast_file = try compilation.db.astFile(module.file_id);
    const function = ast_file.item(ast_file.root_items[0]).Function;
    const body = ast_file.body(function.body);
    const ret_stmt = ast_file.statement(body.statements[0]).Return;

    const consteval = try compilation.db.constEval(compilation.root_module_id);
    const value = consteval.values[ret_stmt.value.?.index()].?.integer;
    const text = try value.toString(testing.allocator, 10, .lower);
    defer testing.allocator.free(text);

    try testing.expectEqualStrings("340282366920938463463374607431768211456", text);

    const type_diags = try compilation.db.typeCheckDiagnostics(compilation.root_module_id, .{ .item = ast_file.root_items[0] });
    try testing.expectEqual(@as(usize, 0), type_diags.len());
}

test "compiler lowers ghost items into ghost AST nodes" {
    const source_text =
        \\contract Spec {
        \\    ghost const LIMIT: u256 = 1;
        \\    ghost storage var hidden: u256;
        \\    ghost fn helper() -> u256 {
        \\        return LIMIT;
        \\    }
        \\    ghost {
        \\        let shadow = hidden;
        \\    }
        \\}
    ;

    var compilation = try compileText(source_text);
    defer compilation.deinit();

    const module = compilation.db.sources.module(compilation.root_module_id);
    const ast_file = try compilation.db.astFile(module.file_id);
    const contract = ast_file.item(ast_file.root_items[0]).Contract;
    try testing.expectEqual(@as(usize, 4), contract.members.len);

    const ghost_const = ast_file.item(contract.members[0]).Constant;
    try testing.expect(ghost_const.is_ghost);

    const ghost_field = ast_file.item(contract.members[1]).Field;
    try testing.expect(ghost_field.is_ghost);

    const ghost_fn = ast_file.item(contract.members[2]).Function;
    try testing.expect(ghost_fn.is_ghost);

    const ghost_block = ast_file.item(contract.members[3]).GhostBlock;
    const ghost_body = ast_file.body(ghost_block.body);
    try testing.expectEqual(@as(usize, 1), ghost_body.statements.len);
    try testing.expect(ast_file.statement(ghost_body.statements[0]).* == .VariableDecl);
}

test "compiler lowers real HIR if and try regions" {
    const source_text =
        \\log Ping(value: u256);
        \\
        \\pub fn flow(ok: bool, next: u256) -> u256 {
        \\    let value = next;
        \\    if (ok) {
        \\        log Ping(next);
        \\    } else {
        \\        assume(next >= 0);
        \\    }
        \\    try {
        \\        assert(ok);
        \\    } catch (err) {
        \\        havoc err;
        \\    }
        \\    return value;
        \\}
    ;

    var compilation = try compileText(source_text);
    defer compilation.deinit();

    const hir_result = try compilation.db.lowerToHir(compilation.root_module_id);
    const hir_text = try hir_result.renderText(testing.allocator);
    defer testing.allocator.free(hir_text);

    try testing.expect(std.mem.containsAtLeast(u8, hir_text, 1, "ora.conditional_return"));
    try testing.expect(std.mem.containsAtLeast(u8, hir_text, 1, "ora.try_stmt"));
    try testing.expect(std.mem.containsAtLeast(u8, hir_text, 1, "ora.log"));
    try testing.expect(std.mem.containsAtLeast(u8, hir_text, 1, "ora.assume"));
    try testing.expect(std.mem.containsAtLeast(u8, hir_text, 1, "ora.assert"));
    try testing.expect(std.mem.containsAtLeast(u8, hir_text, 1, "ora.havoc"));
}

test "compiler lowers real HIR if regions with carried locals" {
    const source_text =
        \\pub fn choose(ok: bool, start: u256) -> u256 {
        \\    let value = start;
        \\    if (ok) {
        \\        value = value + 1;
        \\    } else {
        \\        value = value + 2;
        \\    }
        \\    return value;
        \\}
    ;

    var compilation = try compileText(source_text);
    defer compilation.deinit();

    const hir_result = try compilation.db.lowerToHir(compilation.root_module_id);
    const hir_text = try hir_result.renderText(testing.allocator);
    defer testing.allocator.free(hir_text);

    try testing.expect(std.mem.containsAtLeast(u8, hir_text, 1, "scf.if"));
    try testing.expect(!std.mem.containsAtLeast(u8, hir_text, 1, "ora.if_placeholder"));
}

test "compiler lowers real HIR try regions with carried locals" {
    const source_text =
        \\pub fn recover(start: u256) -> u256 {
        \\    let value = start;
        \\    try {
        \\        value = value + 1;
        \\    } catch (err) {
        \\        value = value + 2;
        \\    }
        \\    return value;
        \\}
    ;

    var compilation = try compileText(source_text);
    defer compilation.deinit();

    const hir_result = try compilation.db.lowerToHir(compilation.root_module_id);
    const hir_text = try hir_result.renderText(testing.allocator);
    defer testing.allocator.free(hir_text);

    try testing.expect(std.mem.containsAtLeast(u8, hir_text, 1, "ora.try_stmt"));
    try testing.expect(!std.mem.containsAtLeast(u8, hir_text, 1, "ora.try_placeholder"));
}

test "compiler lowers real HIR try regions with shadowed catch locals" {
    const source_text =
        \\pub fn recover(start: u256) -> u256 {
        \\    let err = start;
        \\    try {
        \\        err = err + 1;
        \\    } catch (err) {
        \\        assert(err >= 0);
        \\    }
        \\    return err;
        \\}
    ;

    var compilation = try compileText(source_text);
    defer compilation.deinit();

    const hir_result = try compilation.db.lowerToHir(compilation.root_module_id);
    const hir_text = try hir_result.renderText(testing.allocator);
    defer testing.allocator.free(hir_text);

    try testing.expect(std.mem.containsAtLeast(u8, hir_text, 1, "ora.try_stmt"));
    try testing.expect(!std.mem.containsAtLeast(u8, hir_text, 1, "ora.try_placeholder"));
    try testing.expect(std.mem.containsAtLeast(u8, hir_text, 1, "ora.assert"));
}

test "compiler lowers real HIR switch regions with carried locals" {
    const source_text =
        \\pub fn choose(tag: u256, start: u256) -> u256 {
        \\    let value = start;
        \\    switch (tag) {
        \\        0 => {
        \\            value = value + 1;
        \\        },
        \\        else => {
        \\            value = value + 2;
        \\        }
        \\    }
        \\    return value;
        \\}
    ;

    var compilation = try compileText(source_text);
    defer compilation.deinit();

    const hir_result = try compilation.db.lowerToHir(compilation.root_module_id);
    const hir_text = try hir_result.renderText(testing.allocator);
    defer testing.allocator.free(hir_text);

    try testing.expect(std.mem.containsAtLeast(u8, hir_text, 1, "ora.switch"));
    try testing.expect(!std.mem.containsAtLeast(u8, hir_text, 1, "ora.switch_placeholder"));
}

test "compiler lowers real HIR switch expressions" {
    const source_text =
        \\pub fn choose(tag: u256, start: u256) -> u256 {
        \\    return switch (tag) {
        \\        0 => start + 1,
        \\        else => start + 2,
        \\    };
        \\}
    ;

    var compilation = try compileText(source_text);
    defer compilation.deinit();

    const hir_result = try compilation.db.lowerToHir(compilation.root_module_id);
    const hir_text = try hir_result.renderText(testing.allocator);
    defer testing.allocator.free(hir_text);

    try testing.expect(std.mem.containsAtLeast(u8, hir_text, 1, "ora.switch_expr"));
    try testing.expect(!std.mem.containsAtLeast(u8, hir_text, 1, "\"ora.switch_expr\""));
}

test "compiler lowers switch expressions with computed constant patterns" {
    const source_text =
        \\pub fn choose(tag: u256) -> u256 {
        \\    return switch (tag) {
        \\        1 + 2 => 7,
        \\        else => 9,
        \\    };
        \\}
    ;

    var compilation = try compileText(source_text);
    defer compilation.deinit();

    const hir_result = try compilation.db.lowerToHir(compilation.root_module_id);
    const hir_text = try hir_result.renderText(testing.allocator);
    defer testing.allocator.free(hir_text);

    try testing.expect(std.mem.containsAtLeast(u8, hir_text, 1, "case 3 =>"));
    try testing.expect(std.mem.containsAtLeast(u8, hir_text, 1, "ora.switch_expr"));
    try testing.expect(!std.mem.containsAtLeast(u8, hir_text, 1, "\"ora.switch_expr\""));
}

test "compiler lowers real HIR switch regions" {
    const source_text =
        \\pub fn classify(v: u256, fallback: u256) -> u256 {
        \\    switch (v) {
        \\        0 => {
        \\            assert(true);
        \\        },
        \\        1...2 => {
        \\            assume(v >= 1);
        \\        },
        \\        else => {
        \\            havoc fallback;
        \\        }
        \\    }
        \\    return v;
        \\}
    ;

    var compilation = try compileText(source_text);
    defer compilation.deinit();

    const hir_result = try compilation.db.lowerToHir(compilation.root_module_id);
    const hir_text = try hir_result.renderText(testing.allocator);
    defer testing.allocator.free(hir_text);

    try testing.expect(std.mem.containsAtLeast(u8, hir_text, 1, "ora.switch"));
    try testing.expect(!std.mem.containsAtLeast(u8, hir_text, 1, "ora.switch_placeholder"));
    try testing.expect(std.mem.containsAtLeast(u8, hir_text, 1, "ora.assert"));
    try testing.expect(std.mem.containsAtLeast(u8, hir_text, 1, "ora.assume"));
    try testing.expect(std.mem.containsAtLeast(u8, hir_text, 1, "ora.havoc"));
}

test "compiler lowers real HIR while loops for storage-driven loops" {
    const source_text =
        \\storage count: u256;
        \\
        \\pub fn drain(limit: u256) {
        \\    while (count < limit) {
        \\        count = count + 1;
        \\        assert(count >= 1);
        \\    }
        \\}
    ;

    var compilation = try compileText(source_text);
    defer compilation.deinit();

    const hir_result = try compilation.db.lowerToHir(compilation.root_module_id);
    const hir_text = try hir_result.renderText(testing.allocator);
    defer testing.allocator.free(hir_text);

    try testing.expect(std.mem.containsAtLeast(u8, hir_text, 1, "scf.while"));
    try testing.expect(std.mem.containsAtLeast(u8, hir_text, 1, "scf.condition"));
    try testing.expect(std.mem.containsAtLeast(u8, hir_text, 1, "ora.sload"));
    try testing.expect(std.mem.containsAtLeast(u8, hir_text, 1, "ora.sstore"));
    try testing.expect(!std.mem.containsAtLeast(u8, hir_text, 1, "ora.while_placeholder"));
}

test "compiler lowers real HIR while loops with carried locals" {
    const source_text =
        \\pub fn count(limit: u256) -> u256 {
        \\    let value = 0;
        \\    while (value < limit) {
        \\        value = value + 1;
        \\        assert(value >= 1);
        \\    }
        \\    return value;
        \\}
    ;

    var compilation = try compileText(source_text);
    defer compilation.deinit();

    const hir_result = try compilation.db.lowerToHir(compilation.root_module_id);
    const hir_text = try hir_result.renderText(testing.allocator);
    defer testing.allocator.free(hir_text);

    try testing.expect(std.mem.containsAtLeast(u8, hir_text, 1, "scf.while"));
    try testing.expect(!std.mem.containsAtLeast(u8, hir_text, 1, "ora.while_placeholder"));
}

test "compiler lowers real HIR while loops with top-level break and continue" {
    const source_text =
        \\pub fn count(limit: u256) -> u256 {
        \\    let value = 0;
        \\    while (value < limit) {
        \\        value = value + 1;
        \\        continue;
        \\    }
        \\    let seen = 0;
        \\    while (seen < limit) {
        \\        seen = seen + 1;
        \\        break;
        \\    }
        \\    return value + seen;
        \\}
    ;

    var compilation = try compileText(source_text);
    defer compilation.deinit();

    const hir_result = try compilation.db.lowerToHir(compilation.root_module_id);
    const hir_text = try hir_result.renderText(testing.allocator);
    defer testing.allocator.free(hir_text);

    try testing.expect(std.mem.containsAtLeast(u8, hir_text, 2, "scf.while"));
    try testing.expect(std.mem.containsAtLeast(u8, hir_text, 1, "memref.alloca"));
    try testing.expect(!std.mem.containsAtLeast(u8, hir_text, 1, "ora.while_placeholder"));
}

test "compiler lowers real HIR while loops with nested merged if locals" {
    const source_text =
        \\pub fn count(limit: u256, take_big_step: bool) -> u256 {
        \\    let value = 0;
        \\    while (value < limit) {
        \\        if (take_big_step) {
        \\            value = value + 2;
        \\        } else {
        \\            value = value + 1;
        \\        }
        \\        assert(value >= 1);
        \\    }
        \\    return value;
        \\}
    ;

    var compilation = try compileText(source_text);
    defer compilation.deinit();

    const hir_result = try compilation.db.lowerToHir(compilation.root_module_id);
    const hir_text = try hir_result.renderText(testing.allocator);
    defer testing.allocator.free(hir_text);

    try testing.expect(std.mem.containsAtLeast(u8, hir_text, 1, "scf.while"));
    try testing.expect(std.mem.containsAtLeast(u8, hir_text, 1, "scf.if"));
    try testing.expect(!std.mem.containsAtLeast(u8, hir_text, 1, "ora.while_placeholder"));
}

test "compiler lowers real HIR while loops with nested merged try locals" {
    const source_text =
        \\pub fn count(limit: u256) -> u256 {
        \\    let value = 0;
        \\    while (value < limit) {
        \\        try {
        \\            value = value + 2;
        \\        } catch (err) {
        \\            value = value + 1;
        \\        }
        \\        assert(value >= 1);
        \\    }
        \\    return value;
        \\}
    ;

    var compilation = try compileText(source_text);
    defer compilation.deinit();

    const hir_result = try compilation.db.lowerToHir(compilation.root_module_id);
    const hir_text = try hir_result.renderText(testing.allocator);
    defer testing.allocator.free(hir_text);

    try testing.expect(std.mem.containsAtLeast(u8, hir_text, 1, "scf.while"));
    try testing.expect(std.mem.containsAtLeast(u8, hir_text, 1, "ora.try_stmt"));
    try testing.expect(!std.mem.containsAtLeast(u8, hir_text, 1, "ora.while_placeholder"));
}

test "compiler lowers real HIR while loops with nested merged switch locals" {
    const source_text =
        \\pub fn count(limit: u256, step: u256) -> u256 {
        \\    let value = 0;
        \\    while (value < limit) {
        \\        switch (step) {
        \\            1 => {
        \\                value = value + 1;
        \\            },
        \\            else => {
        \\                value = value + 2;
        \\            }
        \\        }
        \\        assert(value >= 1);
        \\    }
        \\    return value;
        \\}
    ;

    var compilation = try compileText(source_text);
    defer compilation.deinit();

    const hir_result = try compilation.db.lowerToHir(compilation.root_module_id);
    const hir_text = try hir_result.renderText(testing.allocator);
    defer testing.allocator.free(hir_text);

    try testing.expect(std.mem.containsAtLeast(u8, hir_text, 1, "scf.while"));
    try testing.expect(std.mem.containsAtLeast(u8, hir_text, 1, "ora.switch"));
    try testing.expect(!std.mem.containsAtLeast(u8, hir_text, 1, "ora.while_placeholder"));
}

test "compiler lowers deeply nested carried locals without placeholders" {
    const source_text =
        \\pub fn count(limit: u256, flag: bool, step: u256, start: u256) -> u256 {
        \\    let value = start;
        \\    while (value < limit) {
        \\        if (flag) {
        \\            switch (step) {
        \\                0 => {
        \\                    try {
        \\                        value = value + 1;
        \\                    } catch (err) {
        \\                        value = value + 2;
        \\                    }
        \\                },
        \\                else => {
        \\                    try {
        \\                        value = value + 3;
        \\                    } catch (err) {
        \\                        value = value + 4;
        \\                    }
        \\                }
        \\            }
        \\        } else {
        \\            value = value + 5;
        \\        }
        \\        assert(value >= start);
        \\    }
        \\    return value;
        \\}
    ;

    var compilation = try compileText(source_text);
    defer compilation.deinit();

    const hir_result = try compilation.db.lowerToHir(compilation.root_module_id);
    const hir_text = try hir_result.renderText(testing.allocator);
    defer testing.allocator.free(hir_text);

    try testing.expect(std.mem.containsAtLeast(u8, hir_text, 1, "scf.while"));
    try testing.expect(std.mem.containsAtLeast(u8, hir_text, 1, "scf.if"));
    try testing.expect(std.mem.containsAtLeast(u8, hir_text, 1, "ora.switch"));
    try testing.expect(std.mem.containsAtLeast(u8, hir_text, 1, "ora.try_stmt"));
    try testing.expect(std.mem.containsAtLeast(u8, hir_text, 1, "ora.assert"));
    try testing.expect(!std.mem.containsAtLeast(u8, hir_text, 1, "ora.if_placeholder"));
    try testing.expect(!std.mem.containsAtLeast(u8, hir_text, 1, "ora.while_placeholder"));
    try testing.expect(!std.mem.containsAtLeast(u8, hir_text, 1, "ora.switch_placeholder"));
    try testing.expect(!std.mem.containsAtLeast(u8, hir_text, 1, "ora.try_placeholder"));
}

test "compiler lowers real HIR while loops with nested switch expressions" {
    const source_text =
        \\pub fn count(limit: u256, step: u256) -> u256 {
        \\    let value = 0;
        \\    while (value < limit) {
        \\        value = switch (step) {
        \\            1 => value + 1,
        \\            else => value + 2,
        \\        };
        \\        assert(value >= 1);
        \\    }
        \\    return value;
        \\}
    ;

    var compilation = try compileText(source_text);
    defer compilation.deinit();

    const hir_result = try compilation.db.lowerToHir(compilation.root_module_id);
    const hir_text = try hir_result.renderText(testing.allocator);
    defer testing.allocator.free(hir_text);

    try testing.expect(std.mem.containsAtLeast(u8, hir_text, 1, "scf.while"));
    try testing.expect(std.mem.containsAtLeast(u8, hir_text, 1, "ora.switch_expr"));
    try testing.expect(!std.mem.containsAtLeast(u8, hir_text, 1, "ora.while_placeholder"));
}
