const std = @import("std");
const testing = std.testing;
const ora_root = @import("ora_root");
const compiler = ora_root.compiler;
const mlir = @import("mlir_c_api").c;
const z3_verification = @import("ora_z3_verification");

const h = @import("compiler.test.helpers.zig");
const compileText = h.compileText;
const renderHirTextForSource = h.renderHirTextForSource;
const renderOraMlirForSource = h.renderOraMlirForSource;
const renderSirTextForModule = h.renderSirTextForModule;
const compilePackage = h.compilePackage;
const expectOraToSirConverts = h.expectOraToSirConverts;
const expectNoResidualOraRuntimeOps = h.expectNoResidualOraRuntimeOps;
const VerificationProbeSummary = h.VerificationProbeSummary;
const expectVerificationProbeEquivalent = h.expectVerificationProbeEquivalent;
const verifyExampleWithoutDegradation = h.verifyExampleWithoutDegradation;
const verifyTextWithoutDegradation = h.verifyTextWithoutDegradation;
const verifyTextWithoutDegradationWithTimeout = h.verifyTextWithoutDegradationWithTimeout;
const firstChildNodeOfKind = h.firstChildNodeOfKind;
const nthChildNodeOfKind = h.nthChildNodeOfKind;
const containsNodeOfKind = h.containsNodeOfKind;
const findVariablePatternByName = h.findVariablePatternByName;
const diagnosticMessagesContain = h.diagnosticMessagesContain;
const countDiagnosticMessages = h.countDiagnosticMessages;
const DiagnosticProbePhase = h.DiagnosticProbePhase;
const expectDiagnosticProbeContains = h.expectDiagnosticProbeContains;
const containsEffectSlot = h.containsEffectSlot;
const containsKeyedEffectSlot = h.containsKeyedEffectSlot;
const nthDescendantNodeOfKind = h.nthDescendantNodeOfKind;
const nthDescendantNodeOfKindInner = h.nthDescendantNodeOfKindInner;

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

test "compiler syntax parses guard clauses alongside requires and ensures" {
    const source_text =
        \\pub fn run(values: u256) -> u256
        \\    requires values >= 0
        \\    guard values < 100
        \\    ensures result >= 0
        \\{
        \\    return values;
        \\}
    ;

    var compilation = try compileText(source_text);
    defer compilation.deinit();

    const module = compilation.db.sources.module(compilation.root_module_id);
    const tree = try compilation.db.syntaxTree(module.file_id);
    const root = compiler.syntax.rootNode(tree);
    const function = nthChildNodeOfKind(root, .FunctionItem, 0);
    try testing.expect(function != null);

    const requires_clause = nthChildNodeOfKind(function.?, .SpecClause, 0);
    const guard_clause = nthChildNodeOfKind(function.?, .SpecClause, 1);
    const ensures_clause = nthChildNodeOfKind(function.?, .SpecClause, 2);
    try testing.expect(requires_clause != null);
    try testing.expect(guard_clause != null);
    try testing.expect(ensures_clause != null);
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

