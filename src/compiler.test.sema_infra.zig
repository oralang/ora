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

test "compiler parses tuple dot access as index expressions" {
    const source_text =
        \\contract TupleDot {
        \\    pub fn run() -> u256 {
        \\        let t: (u256, bool) = (100, true);
        \\        if (t.1) {
        \\            return t.0;
        \\        }
        \\        return 0;
        \\    }
        \\}
    ;

    var compilation = try compileText(source_text);
    defer compilation.deinit();

    const module = compilation.db.sources.module(compilation.root_module_id);
    const tree = try compilation.db.syntaxTree(module.file_id);
    const root = compiler.syntax.rootNode(tree);

    const file = try compilation.db.astFile(module.file_id);
    const contract = file.item(file.root_items[0]).Contract;
    const run_fn = file.item(contract.members[0]).Function;
    const body = file.body(run_fn.body).*;
    const contract_node = nthChildNodeOfKind(root, .ContractItem, 0).?;
    const function = nthChildNodeOfKind(contract_node, .FunctionItem, 0).?;
    const syntax_body = nthChildNodeOfKind(function, .Body, 0).?;
    try testing.expect(containsNodeOfKind(syntax_body, .IndexExpr));

    const if_stmt = file.statement(body.statements[1]).If;
    const if_condition = file.expression(if_stmt.condition).*;
    try testing.expect(if_condition == .Index);

    const then_body = file.body(if_stmt.then_body).*;
    const ret_stmt = file.statement(then_body.statements[0]).Return;
    try testing.expect(ret_stmt.value != null);
    try testing.expect(file.expression(ret_stmt.value.?).* == .Index);
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
    try testing.expectEqual(@as(usize, 5), graph_before.modules.len);
    const root_before = for (graph_before.modules) |summary| {
        if (summary.module_id == compilation.root_module_id) break summary;
    } else return error.TestUnexpectedResult;
    try testing.expectEqual(@as(usize, 1), root_before.imports.len);
    try testing.expectEqualStrings("old_dep", root_before.imports[0].path);

    const index_before = try compilation.db.itemIndex(compilation.root_module_id);
    try testing.expect(index_before.lookup("old_name") != null);

    try compilation.db.updateSourceFile(file_id, updated_source);

    const tree_after = try compilation.db.syntaxTree(file_id);
    const rebuilt_after = try tree_after.reconstructSource(testing.allocator);
    defer testing.allocator.free(rebuilt_after);
    try testing.expectEqualStrings(updated_source, rebuilt_after);

    const graph_after = try compilation.db.moduleGraph(compilation.package_id);
    try testing.expectEqual(@as(usize, 5), graph_after.modules.len);
    const root_after = for (graph_after.modules) |summary| {
        if (summary.module_id == compilation.root_module_id) break summary;
    } else return error.TestUnexpectedResult;
    try testing.expectEqual(@as(usize, 1), root_after.imports.len);
    try testing.expectEqualStrings("new_dep", root_after.imports[0].path);

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

test "compiler tracks per-expression keyed index effects" {
    const source_text =
        \\contract Effects {
        \\    storage balances: map<address, u256>;
        \\
        \\    pub fn read_balance(user: address) -> u256 {
        \\        return balances[user];
        \\    }
        \\}
    ;

    var compilation = try compileText(source_text);
    defer compilation.deinit();

    const root_file_id = compilation.db.sources.module(compilation.root_module_id).file_id;
    const ast_file = try compilation.db.astFile(root_file_id);
    const contract = ast_file.item(ast_file.root_items[0]).Contract;
    const function = ast_file.item(contract.members[1]).Function;
    const ret_stmt = ast_file.statement(ast_file.body(function.body).statements[0]).Return;
    const typecheck = try compilation.db.typeCheck(compilation.root_module_id, .{ .item = ast_file.root_items[0] });
    const user_key = [_]compiler.sema.KeySegment{.{ .parameter = 0 }};

    switch (typecheck.exprEffect(ret_stmt.value.?)) {
        .reads => |effect| try testing.expect(containsKeyedEffectSlot(effect.slots, "balances", .storage, &user_key)),
        else => return error.TestUnexpectedResult,
    }
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

test "compiler evaluates division builtins with distinct semantics" {
    const source_text =
        \\pub fn probe() -> i256 {
        \\    let trunc = @divTrunc(@cast(i256, -7), @cast(i256, 3));
        \\    let floor = @divFloor(@cast(i256, -7), @cast(i256, 3));
        \\    let ceil = @divCeil(@cast(i256, -7), @cast(i256, 3));
        \\    let exact = @divExact(12, 4);
        \\    return trunc + floor + ceil + exact;
        \\}
    ;

    var compilation = try compileText(source_text);
    defer compilation.deinit();

    const module = compilation.db.sources.module(compilation.root_module_id);
    const ast_file = try compilation.db.astFile(module.file_id);
    const consteval = try compilation.db.constEval(compilation.root_module_id);
    const body = ast_file.item(ast_file.root_items[0]).Function.body;
    const statements = ast_file.body(body).statements;

    const trunc_expr = ast_file.statement(statements[0]).VariableDecl.value.?;
    const floor_expr = ast_file.statement(statements[1]).VariableDecl.value.?;
    const ceil_expr = ast_file.statement(statements[2]).VariableDecl.value.?;
    const exact_expr = ast_file.statement(statements[3]).VariableDecl.value.?;

    try testing.expectEqual(@as(i128, -2), try consteval.values[trunc_expr.index()].?.integer.toInt(i128));
    try testing.expectEqual(@as(i128, -3), try consteval.values[floor_expr.index()].?.integer.toInt(i128));
    try testing.expectEqual(@as(i128, -2), try consteval.values[ceil_expr.index()].?.integer.toInt(i128));
    try testing.expectEqual(@as(i128, 3), try consteval.values[exact_expr.index()].?.integer.toInt(i128));
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

test "compiler supports tuple index access in storage assignment expressions" {
    const source_text =
        \\contract TupleStore {
        \\    storage var res: u256;
        \\
        \\    pub fn run() {
        \\        let t: (u256, u256) = (42, 58);
        \\        res = t.0 + t.1;
        \\    }
        \\}
    ;

    var compilation = try compileText(source_text);
    defer compilation.deinit();

    const module = compilation.db.sources.module(compilation.root_module_id);
    const ast_file = try compilation.db.astFile(module.file_id);
    const typecheck = try compilation.db.typeCheck(compilation.root_module_id, .{ .item = ast_file.root_items[0] });
    try testing.expect(typecheck.diagnostics.isEmpty());

    const hir_text = try renderHirTextForSource(source_text);
    defer testing.allocator.free(hir_text);

    try testing.expect(std.mem.containsAtLeast(u8, hir_text, 2, "ora.tuple_extract"));
    try testing.expect(!std.mem.containsAtLeast(u8, hir_text, 1, "ora.index_access"));
}

test "compiler lowers native string and bytes index access" {
    const source_text =
        \\pub fn string_at(text: string, i: u256) -> u8 {
        \\    return text[i];
        \\}
        \\
        \\pub fn bytes_at(data: bytes, i: u256) -> u8 {
        \\    return data[i];
        \\}
    ;

    const rendered = try renderOraMlirForSource(source_text);
    defer testing.allocator.free(rendered);

    try testing.expect(std.mem.containsAtLeast(u8, rendered, 2, "ora.byte_at"));
    try testing.expect(std.mem.containsAtLeast(u8, rendered, 2, "arith.trunci"));
}
