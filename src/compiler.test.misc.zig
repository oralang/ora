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

test "compiler extracts the second binding from multi-field named error arms" {
    const source_text =
        \\error Failure(code: u256, owner: address);
        \\error Denied(owner: address);
        \\pub fn run(value: !u256 | Failure | Denied) -> address {
        \\    return match (value) {
        \\        Ok(inner) => 0x0000000000000000000000000000000000000000,
        \\        Failure(code, owner) => owner,
        \\        Denied(owner) => owner,
        \\    };
        \\}
    ;

    var compilation = try compileText(source_text);
    defer compilation.deinit();

    const typecheck = try compilation.db.moduleTypeCheck(compilation.root_module_id);
    try testing.expect(typecheck.diagnostics.isEmpty());

    const rendered = try renderOraMlirForSource(source_text);
    defer testing.allocator.free(rendered);
    try testing.expect(std.mem.indexOf(u8, rendered, "ora.error.get_error") != null);
    try testing.expect(std.mem.indexOf(u8, rendered, "ora.tuple_extract") != null);
    try testing.expect(std.mem.indexOf(u8, rendered, "[1]") != null);
}

test "compiler types bare self in impl bodies as the impl target" {
    const source_text =
        \\trait Marker {
        \\    fn marked(self) -> bool;
        \\}
        \\
        \\struct Box {
        \\    value: u256,
        \\}
        \\
        \\impl Marker for Box {
        \\    fn marked(self) -> bool {
        \\        return self.value > 0;
        \\    }
        \\}
    ;

    var compilation = try compileText(source_text);
    defer compilation.deinit();

    const module = compilation.db.sources.module(compilation.root_module_id);
    const ast_file = try compilation.db.astFile(module.file_id);
    const item_index = try compilation.db.itemIndex(compilation.root_module_id);
    const typecheck = try compilation.db.moduleTypeCheck(compilation.root_module_id);
    try testing.expect(typecheck.diagnostics.isEmpty());

    const impl_id = item_index.lookupImpl("Marker", "Box").?;
    const impl_item = ast_file.item(impl_id).Impl;
    const method_id = impl_item.methods[0];
    const method = ast_file.item(method_id).Function;
    try testing.expectEqualStrings("Box", typecheck.itemLocatedType(method_id).type.function.param_types[0].name().?);
    try testing.expectEqualStrings("Box", typecheck.pattern_types[method.parameters[0].pattern.index()].type.name().?);
}

test "compiler contextualizes typed tuple literals with mixed element types" {
    const source_text =
        \\comptime const std = @import("std");
        \\
        \\pub fn sender_and_amount() -> u256 {
        \\    let t: (address, u256) = (std.msg.sender(), 500);
        \\    return t.1;
        \\}
    ;

    const hir_text = try renderHirTextForSource(source_text);
    defer testing.allocator.free(hir_text);

    try testing.expect(std.mem.containsAtLeast(u8, hir_text, 1, "!ora.tuple<!ora.address, i256>"));
    try testing.expect(std.mem.containsAtLeast(u8, hir_text, 1, "ora.tuple_create"));
}

test "compiler contextualizes typed tuple literals with narrow integer elements" {
    const source_text =
        \\pub fn amount_only() -> u256 {
        \\    let t: (u8, u256) = (255, 1000);
        \\    return t.1;
        \\}
    ;

    const hir_text = try renderHirTextForSource(source_text);
    defer testing.allocator.free(hir_text);

    try testing.expect(std.mem.containsAtLeast(u8, hir_text, 1, "!ora.tuple<i8, i256>"));
    try testing.expect(std.mem.containsAtLeast(u8, hir_text, 1, "arith.constant -1 : i8"));
    try testing.expect(std.mem.containsAtLeast(u8, hir_text, 1, "ora.tuple_create"));
}

test "compiler HIR output runs through Z3 verification" {
    const source_text =
        \\pub fn keep(next: u256) -> u256
        \\    requires next >= 0;
        \\    ensures result >= 0;
        \\{
        \\    return next;
        \\}
    ;

    var compilation = try compileText(source_text);
    defer compilation.deinit();

    const hir_result = try compilation.db.lowerToHir(compilation.root_module_id);

    var verifier = try z3_verification.VerificationPass.init(testing.allocator);
    defer verifier.deinit();
    verifier.parallel = false;

    var result = try verifier.runVerificationPass(hir_result.module.raw_module);
    defer result.deinit();

    try testing.expect(result.success);
    try testing.expectEqual(@as(usize, 0), result.errors.items.len);
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

test "compiler package loader bridges import graph into source modules" {
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();

    try tmp.dir.writeFile(.{
        .sub_path = "dep.ora",
        .data =
        \\pub fn helper() -> u256 {
        \\    return 7;
        \\}
        ,
    });
    try tmp.dir.writeFile(.{
        .sub_path = "main.ora",
        .data =
        \\comptime const dep = @import("./dep.ora");
        \\
        \\pub fn run() -> u256 {
        \\    return 1;
        \\}
        ,
    });

    const root_path = try std.fmt.allocPrint(testing.allocator, ".zig-cache/tmp/{s}/main.ora", .{tmp.sub_path});
    defer testing.allocator.free(root_path);

    var compilation = try compiler.compilePackage(testing.allocator, root_path);
    defer compilation.deinit();

    const package = compilation.db.sources.package(compilation.package_id);
    try testing.expectEqual(@as(usize, 2), package.modules.items.len);
    try testing.expectEqualStrings("main", compilation.db.sources.module(compilation.root_module_id).name);

    const graph = try compilation.db.moduleGraph(compilation.package_id);
    try testing.expectEqual(@as(usize, 2), graph.modules.len);

    const root_summary = for (graph.modules) |summary| {
        if (summary.module_id == compilation.root_module_id) break summary;
    } else unreachable;
    try testing.expectEqual(@as(usize, 1), root_summary.imports.len);
    try testing.expect(root_summary.imports[0].target_module_id != null);
}

test "compiler source loader injects embedded std modules" {
    const source_text =
        \\comptime const std = @import("std");
        \\
        \\pub fn run() -> u256 {
        \\    return std.constants.U256_MAX;
        \\}
    ;

    var compilation = try compileText(source_text);
    defer compilation.deinit();

    const package = compilation.db.sources.package(compilation.package_id);
    try testing.expectEqual(@as(usize, 5), package.modules.items.len);

    const graph = try compilation.db.moduleGraph(compilation.package_id);
    const root_summary = for (graph.modules) |summary| {
        if (summary.module_id == compilation.root_module_id) break summary;
    } else return error.TestUnexpectedResult;

    try testing.expectEqual(@as(usize, 1), root_summary.imports.len);
    try testing.expect(root_summary.imports[0].target_module_id != null);
}

test "compiler module graph distinguishes imports with the same basename in different directories" {
    var compiler_db = compiler.db.CompilerDb.init(testing.allocator);
    defer compiler_db.deinit();

    const package_id = try compiler_db.addPackage("main");

    const left_math_file = try compiler_db.addSourceFile("left/math.ora",
        \\pub fn left() -> u256 { return 1; }
    );
    const right_math_file = try compiler_db.addSourceFile("right/math.ora",
        \\pub fn right() -> u256 { return 2; }
    );
    const main_file = try compiler_db.addSourceFile("main.ora",
        \\comptime const left_math = @import("./left/math.ora");
        \\comptime const right_math = @import("./right/math.ora");
        \\pub fn run() -> u256 { return 0; }
    );

    const left_math_module = try compiler_db.addModule(package_id, left_math_file, "math");
    const right_math_module = try compiler_db.addModule(package_id, right_math_file, "math");
    const main_module = try compiler_db.addModule(package_id, main_file, "main");

    const graph = try compiler_db.moduleGraph(package_id);
    const main_summary = for (graph.modules) |summary| {
        if (summary.module_id == main_module) break summary;
    } else unreachable;

    try testing.expectEqual(@as(usize, 2), main_summary.imports.len);
    try testing.expectEqual(left_math_module, main_summary.imports[0].target_module_id.?);
    try testing.expectEqual(right_math_module, main_summary.imports[1].target_module_id.?);
    try testing.expect(std.mem.indexOfScalar(compiler.source.ModuleId, main_summary.dependencies, left_math_module) != null);
    try testing.expect(std.mem.indexOfScalar(compiler.source.ModuleId, main_summary.dependencies, right_math_module) != null);
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

test "compiler preserves typed local names in assignments" {
    const source_text =
        \\contract Wallet {
        \\    storage balances: map<address, u256>;
        \\
        \\    pub fn deposit(amount: u256) {
        \\        const sender: address = std.msg.sender;
        \\        balances[sender] = amount;
        \\    }
        \\}
    ;

    var compilation = try compileText(source_text);
    defer compilation.deinit();

    const hir_result = try compilation.db.lowerToHir(compilation.root_module_id);
    const hir_text = try hir_result.renderText(testing.allocator);
    defer testing.allocator.free(hir_text);

    try testing.expect(std.mem.containsAtLeast(u8, hir_text, 1, "ora.map_store"));
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

test "compiler does not lower type-value expression statements to placeholder ops" {
    const source_text =
        \\struct Pair(comptime T: type) {
        \\    left: T,
        \\    right: T,
        \\}
        \\
        \\pub fn run() -> u256 {
        \\    return comptime {
        \\        Pair<u256>;
        \\        1;
        \\    };
        \\}
    ;

    var compilation = try compileText(source_text);
    defer compilation.deinit();

    const hir_result = try compilation.db.lowerToHir(compilation.root_module_id);
    const hir_text = try hir_result.renderText(testing.allocator);
    defer testing.allocator.free(hir_text);

    try testing.expect(!std.mem.containsAtLeast(u8, hir_text, 1, "ora.type_value"));
}

test "compiler allows writes to a different keyed map entry" {
    const source_text =
        \\contract Locked {
        \\    storage balances: map<address, u256>;
        \\
        \\    pub fn guarded(user: address, other: address, value: u256) {
        \\        @lock(balances[user]);
        \\        balances[other] = value;
        \\    }
        \\}
    ;

    var compilation = try compileText(source_text);
    defer compilation.deinit();

    const root_file_id = compilation.db.sources.module(compilation.root_module_id).file_id;
    const ast_file = try compilation.db.astFile(root_file_id);
    const typecheck = try compilation.db.typeCheck(compilation.root_module_id, .{ .item = ast_file.root_items[0] });

    try testing.expect(!diagnosticMessagesContain(&typecheck.diagnostics, "cannot write locked storage slot 'balances'"));
}

test "compiler allows writes to a different constant keyed map entry" {
    const source_text =
        \\contract Locked {
        \\    storage counts: map<u256, u256>;
        \\
        \\    pub fn guarded(value: u256) {
        \\        @lock(counts[1]);
        \\        counts[2] = value;
        \\    }
        \\}
    ;

    var compilation = try compileText(source_text);
    defer compilation.deinit();

    const root_file_id = compilation.db.sources.module(compilation.root_module_id).file_id;
    const ast_file = try compilation.db.astFile(root_file_id);
    const typecheck = try compilation.db.typeCheck(compilation.root_module_id, .{ .item = ast_file.root_items[0] });

    try testing.expect(!diagnosticMessagesContain(&typecheck.diagnostics, "cannot write locked storage slot 'counts'"));
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

test "compiler persists divmod as a tuple consteval value" {
    const source_text =
        \\pub fn probe() -> u256 {
        \\    let pair = @divmod(17, 5);
        \\    return pair.0 * 5 + pair.1;
        \\}
    ;

    var compilation = try compileText(source_text);
    defer compilation.deinit();

    const module = compilation.db.sources.module(compilation.root_module_id);
    const ast_file = try compilation.db.astFile(module.file_id);
    const consteval = try compilation.db.constEval(compilation.root_module_id);
    const body = ast_file.item(ast_file.root_items[0]).Function.body;
    const pair_expr = ast_file.statement(ast_file.body(body).statements[0]).VariableDecl.value.?;
    const value = consteval.values[pair_expr.index()] orelse return error.TestUnexpectedResult;
    try testing.expect(value == .tuple);
    try testing.expectEqual(@as(usize, 2), value.tuple.len);
    try testing.expectEqual(@as(i128, 3), try value.tuple[0].integer.toInt(i128));
    try testing.expectEqual(@as(i128, 2), try value.tuple[1].integer.toInt(i128));
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
        \\    let problem = Failure(7);
        \\    return value;
        \\}
    ;

    const hir_text = try renderHirTextForSource(source_text);
    defer testing.allocator.free(hir_text);

    try testing.expect(std.mem.containsAtLeast(u8, hir_text, 1, "ora.error.return"));
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

test "compiler skips unknown carried locals in for lowering" {
    const source_text =
        \\pub fn scan(values: slice[u256]) -> u256 {
        \\    let total = 0;
        \\    let bad = total.missing;
        \\    for (values) |value| {
        \\        bad = value.missing;
        \\        total = total + value;
        \\    }
        \\    return total;
        \\}
    ;

    var compilation = try compileText(source_text);
    defer compilation.deinit();

    const hir_result = try compilation.db.lowerToHir(compilation.root_module_id);
    const hir_text = try hir_result.renderText(testing.allocator);
    defer testing.allocator.free(hir_text);

    try testing.expect(hir_result.type_fallback_count > 0);
    try testing.expect(std.mem.containsAtLeast(u8, hir_text, 1, "scf.for"));
    try testing.expect(!std.mem.containsAtLeast(u8, hir_text, 1, "ora.for_placeholder"));
}

test "compiler treats scaled arithmetic as scaled in sema" {
    const source_text =
        \\fn addScaled(a: Scaled<u256, 18>, b: Scaled<u256, 18>) {
        \\    let sum = a + b;
        \\}
        \\
        \\fn mulScaled(a: Scaled<u256, 18>, b: Scaled<u256, 6>) {
        \\    let product = a * b;
        \\}
    ;

    var compilation = try compileText(source_text);
    defer compilation.deinit();

    const root_module = compilation.db.sources.module(compilation.root_module_id);
    const ast_file = try compilation.db.astFile(root_module.file_id);
    const typecheck = try compilation.db.typeCheck(compilation.root_module_id, .{ .item = ast_file.root_items[0] });
    const add_fn = ast_file.items[0].Function;
    const add_body = ast_file.body(add_fn.body).*;
    const sum_decl = ast_file.statement(add_body.statements[0]).VariableDecl;
    const sum_type = typecheck.pattern_types[sum_decl.pattern.index()].type;
    try testing.expect(sum_type.kind() == .refinement);
    try testing.expectEqualStrings("Scaled", sum_type.refinement.name);
    try testing.expectEqualStrings("18", sum_type.refinement.args[1].Integer.text);

    const mul_typecheck = try compilation.db.typeCheck(compilation.root_module_id, .{ .item = ast_file.root_items[1] });
    const mul_fn = ast_file.items[1].Function;
    const mul_body = ast_file.body(mul_fn.body).*;
    const product_decl = ast_file.statement(mul_body.statements[0]).VariableDecl;
    const product_type = mul_typecheck.pattern_types[product_decl.pattern.index()].type;
    try testing.expect(product_type.kind() == .refinement);
    try testing.expectEqualStrings("Scaled", product_type.refinement.name);
    try testing.expectEqualStrings("24", product_type.refinement.args[1].Integer.text);
}

test "compiler accepts scaled returns from scaled arithmetic" {
    const source_text =
        \\fn addScaled(a: Scaled<u256, 18>, b: Scaled<u256, 18>) -> Scaled<u256, 18> {
        \\    return a + b;
        \\}
        \\
        \\fn mulScaled(a: Scaled<u256, 18>, b: Scaled<u256, 6>) -> Scaled<u256, 24> {
        \\    return a * b;
        \\}
    ;

    var compilation = try compileText(source_text);
    defer compilation.deinit();

    const root_module = compilation.db.sources.module(compilation.root_module_id);
    const ast_file = try compilation.db.astFile(root_module.file_id);
    _ = try compilation.db.typeCheck(compilation.root_module_id, .{ .item = ast_file.root_items[0] });
    _ = try compilation.db.typeCheck(compilation.root_module_id, .{ .item = ast_file.root_items[1] });

    const diagnostics_list = try compilation.db.typeCheckDiagnostics(compilation.root_module_id, .{ .item = ast_file.root_items[0] });
    try testing.expectEqual(@as(usize, 0), diagnostics_list.items.items.len);
    const diagnostics_list_mul = try compilation.db.typeCheckDiagnostics(compilation.root_module_id, .{ .item = ast_file.root_items[1] });
    try testing.expectEqual(@as(usize, 0), diagnostics_list_mul.items.items.len);
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

test "compiler assigns integer array literals to concrete integer array types" {
    const source_text =
        \\pub fn build() -> [u256; 4] {
        \\    let dest: [u256; 4] = [0, 0, 0, 0];
        \\    return dest;
        \\}
    ;

    var compilation = try compileText(source_text);
    defer compilation.deinit();

    const typecheck = try compilation.db.moduleTypeCheck(compilation.root_module_id);
    try testing.expect(typecheck.diagnostics.isEmpty());
}

test "compiler contextualizes nested array literals to their declared element types" {
    const source_text =
        \\pub fn nested() -> [[u256; 2]; 2] {
        \\    let inner1: [u256; 2] = [1, 2];
        \\    let inner2: [u256; 2] = [3, 4];
        \\    return [inner1, inner2];
        \\}
    ;

    var compilation = try compileText(source_text);
    defer compilation.deinit();

    const typecheck = try compilation.db.moduleTypeCheck(compilation.root_module_id);
    try testing.expect(typecheck.diagnostics.isEmpty());

    const hir_result = try compilation.db.lowerToHir(compilation.root_module_id);

    const hir_text = try hir_result.renderText(testing.allocator);
    defer testing.allocator.free(hir_text);

    try testing.expect(std.mem.containsAtLeast(u8, hir_text, 1, "func.func @nested() -> memref<2xmemref<2xi256>>"));
    try testing.expect(std.mem.containsAtLeast(u8, hir_text, 1, "memref.alloca() : memref<2xmemref<2xi256>>"));
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

test "compiler supports general anonymous struct types" {
    const source_text =
        \\pub fn run(amount: u256) -> u256 {
        \\    let payload: struct { amount: u256, ok: bool } = .{ .amount = amount, .ok = true };
        \\    if (payload.ok) {
        \\        return payload.amount;
        \\    }
        \\    return 0;
        \\}
    ;

    var compilation = try compileText(source_text);
    defer compilation.deinit();

    const module = compilation.db.sources.module(compilation.root_module_id);
    const ast_file = try compilation.db.astFile(module.file_id);
    const typecheck = try compilation.db.typeCheck(compilation.root_module_id, .{ .item = ast_file.root_items[0] });
    try testing.expect(typecheck.diagnostics.isEmpty());

    const function = ast_file.item(ast_file.root_items[0]).Function;
    const body = ast_file.body(function.body);
    const payload_pattern = findVariablePatternByName(ast_file, body.statements, "payload").?;
    const payload_type = typecheck.pattern_types[payload_pattern.index()].type;
    try testing.expectEqual(compiler.sema.TypeKind.anonymous_struct, payload_type.kind());
    try testing.expectEqual(@as(usize, 2), payload_type.anonymous_struct.fields.len);
    try testing.expectEqualStrings("amount", payload_type.anonymous_struct.fields[0].name);
    try testing.expectEqualStrings("ok", payload_type.anonymous_struct.fields[1].name);

    const hir_text = try renderHirTextForSource(source_text);
    defer testing.allocator.free(hir_text);
    try testing.expect(std.mem.containsAtLeast(u8, hir_text, 1, "!ora.struct_anon<"));
    try testing.expect(std.mem.containsAtLeast(u8, hir_text, 1, "ora.struct_init"));
    try testing.expect(std.mem.containsAtLeast(u8, hir_text, 2, "ora.struct_field_extract"));
}

test "compiler permits recursive runtime structs through map indirection" {
    const source_text =
        \\struct Node {
        \\    children: map<u256, Node>,
        \\}
        \\
        \\pub fn probe() -> u256 {
        \\    return 0;
        \\}
    ;

    var compilation = try compileText(source_text);
    defer compilation.deinit();

    const typecheck = try compilation.db.moduleTypeCheck(compilation.root_module_id);
    try testing.expect(typecheck.diagnostics.isEmpty());
}

test "compiler permits recursive runtime structs through slice indirection" {
    const source_text =
        \\struct Node {
        \\    children: slice[Node],
        \\}
        \\
        \\pub fn probe() -> u256 {
        \\    return 0;
        \\}
    ;

    var compilation = try compileText(source_text);
    defer compilation.deinit();

    const typecheck = try compilation.db.moduleTypeCheck(compilation.root_module_id);
    try testing.expect(typecheck.diagnostics.isEmpty());
}

test "compiler allows local name after early-return if" {
    const source_text =
        \\contract EarlyReturn {
        \\    pub fn process(x: u256) -> u256 {
        \\        if (x > 1000) {
        \\            return 1000;
        \\        }
        \\
        \\        var output: u256 = x * 2;
        \\        if (output > 500) {
        \\            return 500;
        \\        }
        \\
        \\        output += 10;
        \\        return output;
        \\    }
        \\}
    ;

    var compilation = try compileText(source_text);
    defer compilation.deinit();

    const module_typecheck = try compilation.db.moduleTypeCheck(compilation.root_module_id);
    try testing.expect(module_typecheck.diagnostics.isEmpty());

    _ = try compilation.db.lowerToHir(compilation.root_module_id);
}

test "compiler partially evaluates pure helper calls in runtime return expressions" {
    const source_text =
        \\fn helper(a: u256, b: u256) -> u256 {
        \\    return a + b;
        \\}
        \\
        \\contract Sample {
        \\    pub fn run(x: u256) -> u256 {
        \\        return x + helper(2, 3);
        \\    }
        \\}
    ;

    const rendered = try renderOraMlirForSource(source_text);
    defer testing.allocator.free(rendered);
    try testing.expectEqual(0, std.mem.count(u8, rendered, "call @helper"));
    try testing.expect(std.mem.indexOf(u8, rendered, " 5 : i256") != null);
}

test "compiler partially evaluates nested pure helper calls in runtime functions" {
    const source_text =
        \\fn helper(a: u256, b: u256) -> u256 {
        \\    return a + b;
        \\}
        \\
        \\contract Sample {
        \\    pub fn run(x: u256) -> u256 {
        \\        return x + helper(helper(1, 2), 3);
        \\    }
        \\}
    ;

    const rendered = try renderOraMlirForSource(source_text);
    defer testing.allocator.free(rendered);
    try testing.expectEqual(0, std.mem.count(u8, rendered, "call @helper"));
    try testing.expect(std.mem.indexOf(u8, rendered, " 6 : i256") != null);
}

test "compiler partially evaluates pure helper calls in runtime const declarations" {
    const source_text =
        \\fn helper(a: u256, b: u256) -> u256 {
        \\    return a + b;
        \\}
        \\
        \\contract Sample {
        \\    pub fn run(x: u256) -> u256 {
        \\        const ct: u256 = helper(2, 3);
        \\        return x + ct;
        \\    }
        \\}
    ;

    const rendered = try renderOraMlirForSource(source_text);
    defer testing.allocator.free(rendered);
    try testing.expectEqual(0, std.mem.count(u8, rendered, "call @helper"));
    try testing.expect(std.mem.indexOf(u8, rendered, " 5 : i256") != null);
}

test "compiler partially evaluates nested pure helper calls in runtime const declarations" {
    const source_text =
        \\fn helper(a: u256, b: u256) -> u256 {
        \\    return a + b;
        \\}
        \\
        \\contract Sample {
        \\    pub fn run(x: u256) -> u256 {
        \\        const nested: u256 = helper(helper(1, 2), 3);
        \\        return x + nested;
        \\    }
        \\}
    ;

    const rendered = try renderOraMlirForSource(source_text);
    defer testing.allocator.free(rendered);
    try testing.expectEqual(0, std.mem.count(u8, rendered, "call @helper"));
    try testing.expect(std.mem.indexOf(u8, rendered, " 6 : i256") != null);
}

test "compiler does not partially evaluate impure helper calls in runtime functions" {
    const source_text =
        \\contract Sample {
        \\    storage var counter: u256;
        \\
        \\    fn read_counter() -> u256 {
        \\        return counter;
        \\    }
        \\
        \\    pub fn run(x: u256) -> u256 {
        \\        return x + read_counter();
        \\    }
        \\}
    ;

    const rendered = try renderOraMlirForSource(source_text);
    defer testing.allocator.free(rendered);
    try testing.expect(std.mem.indexOf(u8, rendered, "call @read_counter") != null);
    try testing.expect(std.mem.indexOf(u8, rendered, "ora.sload") != null);
}

test "compiler does not partially evaluate impure helper calls in runtime const declarations" {
    const source_text =
        \\contract Sample {
        \\    storage var counter: u256;
        \\
        \\    fn read_counter() -> u256 {
        \\        return counter;
        \\    }
        \\
        \\    pub fn run(x: u256) -> u256 {
        \\        const live: u256 = read_counter();
        \\        return x + live;
        \\    }
        \\}
    ;

    const rendered = try renderOraMlirForSource(source_text);
    defer testing.allocator.free(rendered);
    try testing.expect(std.mem.indexOf(u8, rendered, "call @read_counter") != null);
    try testing.expect(std.mem.indexOf(u8, rendered, "ora.sload") != null);
}

test "compiler unrolls small constant runtime for-count loops" {
    const source_text =
        \\contract Sample {
        \\    pub fn run() -> u256 {
        \\        let sum = 0;
        \\        for (4) |i| {
        \\            sum += i;
        \\        }
        \\        return sum;
        \\    }
        \\}
    ;

    const rendered = try renderOraMlirForSource(source_text);
    defer testing.allocator.free(rendered);
    try testing.expect(std.mem.indexOf(u8, rendered, "scf.for") == null);
    try testing.expect(std.mem.indexOf(u8, rendered, "ora.for_placeholder") == null);
    try testing.expect(std.mem.count(u8, rendered, "ora.stmt.1") >= 4);
    try testing.expect(std.mem.count(u8, rendered, "ora.stmt.2") >= 4);
    try testing.expect(std.mem.indexOf(u8, rendered, "ora.synthetic.0.4") != null);
    try testing.expect(std.mem.indexOf(u8, rendered, "ora.synthetic.1.4") != null);
    try testing.expect(std.mem.indexOf(u8, rendered, "ora.synthetic.2.4") != null);
    try testing.expect(std.mem.indexOf(u8, rendered, "ora.synthetic.3.4") != null);
}

test "compiler unrolls small constant runtime range-for loops" {
    const source_text =
        \\contract Sample {
        \\    pub fn run() -> u256 {
        \\        let sum = 0;
        \\        for (2..5) |i, _| {
        \\            sum += i;
        \\        }
        \\        return sum;
        \\    }
        \\}
    ;

    const rendered = try renderOraMlirForSource(source_text);
    defer testing.allocator.free(rendered);
    try testing.expect(std.mem.indexOf(u8, rendered, "scf.for") == null);
    try testing.expect(std.mem.indexOf(u8, rendered, "ora.for_placeholder") == null);
    try testing.expect(std.mem.count(u8, rendered, "ora.stmt.1") >= 3);
    try testing.expect(std.mem.count(u8, rendered, "ora.stmt.2") >= 3);
    try testing.expect(std.mem.indexOf(u8, rendered, "ora.synthetic.0.3") != null);
    try testing.expect(std.mem.indexOf(u8, rendered, "ora.synthetic.1.3") != null);
    try testing.expect(std.mem.indexOf(u8, rendered, "ora.synthetic.2.3") != null);
}

test "compiler unrolls small constant runtime for loops with unlabeled break" {
    const source_text =
        \\contract Sample {
        \\    pub fn run() -> u256 {
        \\        let sum = 0;
        \\        for (4) |i| {
        \\            if (i == 2) {
        \\                break;
        \\            }
        \\            sum += i;
        \\        }
        \\        return sum;
        \\    }
        \\}
    ;

    const rendered = try renderOraMlirForSource(source_text);
    defer testing.allocator.free(rendered);
    try testing.expect(std.mem.indexOf(u8, rendered, "scf.for") == null);
    try testing.expect(std.mem.indexOf(u8, rendered, "ora.break") == null);
    try testing.expect(std.mem.indexOf(u8, rendered, "ora.synthetic.") != null);
}

test "compiler unrolls small constant runtime for loops with unlabeled continue" {
    const source_text =
        \\contract Sample {
        \\    pub fn run() -> u256 {
        \\        let sum = 0;
        \\        for (4) |i| {
        \\            if (i == 2) {
        \\                continue;
        \\            }
        \\            sum += i;
        \\        }
        \\        return sum;
        \\    }
        \\}
    ;

    const rendered = try renderOraMlirForSource(source_text);
    defer testing.allocator.free(rendered);
    try testing.expect(std.mem.indexOf(u8, rendered, "scf.for") == null);
    try testing.expect(std.mem.indexOf(u8, rendered, "ora.continue") == null);
    try testing.expect(std.mem.indexOf(u8, rendered, "ora.synthetic.0.4") != null);
    try testing.expect(std.mem.indexOf(u8, rendered, "ora.synthetic.2.4") != null);
}

test "compiler unrolls small constant labeled runtime for loops with labeled break" {
    const source_text =
        \\contract Sample {
        \\    pub fn run() -> u256 {
        \\        let sum = 0;
        \\        outer: for (4) |i| {
        \\            if (i == 2) {
        \\                break :outer;
        \\            }
        \\            sum += i;
        \\        }
        \\        return sum;
        \\    }
        \\}
    ;

    const rendered = try renderOraMlirForSource(source_text);
    defer testing.allocator.free(rendered);
    try testing.expect(std.mem.indexOf(u8, rendered, "scf.for") == null);
    try testing.expect(std.mem.indexOf(u8, rendered, "ora.break") == null);
    try testing.expect(std.mem.indexOf(u8, rendered, "ora.synthetic.") != null);
}

test "compiler unrolls small constant labeled runtime for loops with labeled continue" {
    const source_text =
        \\contract Sample {
        \\    pub fn run() -> u256 {
        \\        let sum = 0;
        \\        outer: for (4) |i| {
        \\            if (i == 2) {
        \\                continue :outer;
        \\            }
        \\            sum += i;
        \\        }
        \\        return sum;
        \\    }
        \\}
    ;

    const rendered = try renderOraMlirForSource(source_text);
    defer testing.allocator.free(rendered);
    try testing.expect(std.mem.indexOf(u8, rendered, "scf.for") == null);
    try testing.expect(std.mem.indexOf(u8, rendered, "ora.continue") == null);
    try testing.expect(std.mem.indexOf(u8, rendered, "ora.synthetic.0.4") != null);
    try testing.expect(std.mem.indexOf(u8, rendered, "ora.synthetic.2.4") != null);
}

test "compiler does not unroll runtime for-count loops above the unroll limit" {
    const source_text =
        \\contract Sample {
        \\    pub fn run() -> u256 {
        \\        let sum = 0;
        \\        for (9) |i| {
        \\            sum += i;
        \\        }
        \\        return sum;
        \\    }
        \\}
    ;

    const rendered = try renderOraMlirForSource(source_text);
    defer testing.allocator.free(rendered);
    try testing.expect(std.mem.indexOf(u8, rendered, "ora.for_placeholder") != null);
    try testing.expect(std.mem.indexOf(u8, rendered, "ora.synthetic.") == null);
}

test "compiler does not unroll runtime range-for loops above the unroll limit" {
    const source_text =
        \\contract Sample {
        \\    pub fn run() -> u256 {
        \\        let sum = 0;
        \\        for (0..9) |i, _| {
        \\            sum += i;
        \\        }
        \\        return sum;
        \\    }
        \\}
    ;

    const rendered = try renderOraMlirForSource(source_text);
    defer testing.allocator.free(rendered);
    try testing.expect(std.mem.indexOf(u8, rendered, "ora.for_placeholder") != null or
        std.mem.indexOf(u8, rendered, "scf.for") != null);
    try testing.expect(std.mem.indexOf(u8, rendered, "ora.synthetic.") == null);
}

test "compiler unrolls small constant runtime while loops" {
    const source_text =
        \\contract Sample {
        \\    pub fn run() -> u256 {
        \\        let i = 0;
        \\        let sum = 0;
        \\        while (i < 4) {
        \\            sum += i;
        \\            i += 1;
        \\        }
        \\        return sum;
        \\    }
        \\}
    ;

    const rendered = try renderOraMlirForSource(source_text);
    defer testing.allocator.free(rendered);
    try testing.expect(std.mem.indexOf(u8, rendered, "scf.while") == null);
    try testing.expect(std.mem.indexOf(u8, rendered, "ora.while_placeholder") == null);
    try testing.expect(std.mem.indexOf(u8, rendered, "ora.synthetic.0.4") != null);
    try testing.expect(std.mem.indexOf(u8, rendered, "ora.synthetic.1.4") != null);
    try testing.expect(std.mem.indexOf(u8, rendered, "ora.synthetic.2.4") != null);
    try testing.expect(std.mem.indexOf(u8, rendered, "ora.synthetic.3.4") != null);
}

test "compiler does not unroll runtime while loops above the unroll limit" {
    const source_text =
        \\contract Sample {
        \\    pub fn run() -> u256 {
        \\        let i = 0;
        \\        let sum = 0;
        \\        while (i < 9) {
        \\            sum += i;
        \\            i += 1;
        \\        }
        \\        return sum;
        \\    }
        \\}
    ;

    const rendered = try renderOraMlirForSource(source_text);
    defer testing.allocator.free(rendered);
    try testing.expect(std.mem.indexOf(u8, rendered, "scf.while") != null);
    try testing.expect(std.mem.indexOf(u8, rendered, "ora.synthetic.") == null);
}

test "compiler unrolls small constant runtime while loops with unlabeled break" {
    const source_text =
        \\contract Sample {
        \\    pub fn run() -> u256 {
        \\        let i = 0;
        \\        let sum = 0;
        \\        while (i < 4) {
        \\            if (i == 2) {
        \\                break;
        \\            }
        \\            sum += i;
        \\            i += 1;
        \\        }
        \\        return sum;
        \\    }
        \\}
    ;

    const rendered = try renderOraMlirForSource(source_text);
    defer testing.allocator.free(rendered);
    try testing.expect(std.mem.indexOf(u8, rendered, "scf.while") == null);
    try testing.expect(std.mem.indexOf(u8, rendered, "ora.break") == null);
    try testing.expect(std.mem.indexOf(u8, rendered, "ora.synthetic.") != null);
}

test "compiler unrolls small constant runtime while loops with unlabeled continue" {
    const source_text =
        \\contract Sample {
        \\    pub fn run() -> u256 {
        \\        let i = 0;
        \\        let sum = 0;
        \\        while (i < 4) {
        \\            i += 1;
        \\            if (i == 2) {
        \\                continue;
        \\            }
        \\            sum += i;
        \\        }
        \\        return sum;
        \\    }
        \\}
    ;

    const rendered = try renderOraMlirForSource(source_text);
    defer testing.allocator.free(rendered);
    try testing.expect(std.mem.indexOf(u8, rendered, "scf.while") == null);
    try testing.expect(std.mem.indexOf(u8, rendered, "ora.continue") == null);
    try testing.expect(std.mem.indexOf(u8, rendered, "ora.synthetic.0.4") != null);
    try testing.expect(std.mem.indexOf(u8, rendered, "ora.synthetic.1.4") != null);
    try testing.expect(std.mem.indexOf(u8, rendered, "ora.synthetic.2.4") != null);
    try testing.expect(std.mem.indexOf(u8, rendered, "ora.synthetic.3.4") != null);
}

test "compiler unrolls small constant labeled runtime while loops with labeled break" {
    const source_text =
        \\contract Sample {
        \\    pub fn run() -> u256 {
        \\        let i = 0;
        \\        let sum = 0;
        \\        outer: while (i < 4) {
        \\            if (i == 2) {
        \\                break :outer;
        \\            }
        \\            sum += i;
        \\            i += 1;
        \\        }
        \\        return sum;
        \\    }
        \\}
    ;

    const rendered = try renderOraMlirForSource(source_text);
    defer testing.allocator.free(rendered);
    try testing.expect(std.mem.indexOf(u8, rendered, "scf.while") == null);
    try testing.expect(std.mem.indexOf(u8, rendered, "ora.break") == null);
    try testing.expect(std.mem.indexOf(u8, rendered, "ora.synthetic.") != null);
}

test "compiler unrolls small constant labeled runtime while loops with labeled continue" {
    const source_text =
        \\contract Sample {
        \\    pub fn run() -> u256 {
        \\        let i = 0;
        \\        let sum = 0;
        \\        outer: while (i < 4) {
        \\            i += 1;
        \\            if (i == 2) {
        \\                continue :outer;
        \\            }
        \\            sum += i;
        \\        }
        \\        return sum;
        \\    }
        \\}
    ;

    const rendered = try renderOraMlirForSource(source_text);
    defer testing.allocator.free(rendered);
    try testing.expect(std.mem.indexOf(u8, rendered, "scf.while") == null);
    try testing.expect(std.mem.indexOf(u8, rendered, "ora.continue") == null);
    try testing.expect(std.mem.indexOf(u8, rendered, "ora.synthetic.0.4") != null);
    try testing.expect(std.mem.indexOf(u8, rendered, "ora.synthetic.1.4") != null);
    try testing.expect(std.mem.indexOf(u8, rendered, "ora.synthetic.2.4") != null);
    try testing.expect(std.mem.indexOf(u8, rendered, "ora.synthetic.3.4") != null);
}

test "compiler unrolls bounded runtime while loops with simple local declarations" {
    const source_text =
        \\contract Sample {
        \\    pub fn run() -> u256 {
        \\        let i = 0;
        \\        let sum = 0;
        \\        while (i < 4) {
        \\            let next = i + 1;
        \\            sum += next;
        \\            i = next;
        \\        }
        \\        return sum;
        \\    }
        \\}
    ;

    const rendered = try renderOraMlirForSource(source_text);
    defer testing.allocator.free(rendered);
    try testing.expect(std.mem.indexOf(u8, rendered, "scf.while") == null);
    try testing.expect(std.mem.indexOf(u8, rendered, "ora.while_placeholder") == null);
    try testing.expect(std.mem.indexOf(u8, rendered, "ora.synthetic.0.4") != null);
    try testing.expect(std.mem.indexOf(u8, rendered, "ora.synthetic.1.4") != null);
    try testing.expect(std.mem.indexOf(u8, rendered, "ora.synthetic.2.4") != null);
    try testing.expect(std.mem.indexOf(u8, rendered, "ora.synthetic.3.4") != null);
}

test "compiler unrolls bounded runtime while loops with boolean local declarations" {
    const source_text =
        \\contract Sample {
        \\    pub fn run() -> u256 {
        \\        let i = 0;
        \\        let sum = 0;
        \\        while (i < 4) {
        \\            let done = i == 2;
        \\            if (done) {
        \\                break;
        \\            }
        \\            sum += i;
        \\            i += 1;
        \\        }
        \\        return sum;
        \\    }
        \\}
    ;

    const rendered = try renderOraMlirForSource(source_text);
    defer testing.allocator.free(rendered);
    try testing.expect(std.mem.indexOf(u8, rendered, "scf.while") == null);
    try testing.expect(std.mem.indexOf(u8, rendered, "ora.while_placeholder") == null);
    try testing.expect(std.mem.indexOf(u8, rendered, "ora.break") == null);
    try testing.expect(std.mem.indexOf(u8, rendered, "ora.synthetic.") != null);
}

test "compiler unrolls bounded runtime while loops with declarations inside if branches" {
    const source_text =
        \\contract Sample {
        \\    pub fn run() -> u256 {
        \\        let i = 0;
        \\        let sum = 0;
        \\        while (i < 4) {
        \\            if (i == 2) {
        \\                let stop = true;
        \\                if (stop) {
        \\                    break;
        \\                }
        \\            } else {
        \\                let next = i + 1;
        \\                sum += next;
        \\                i = next;
        \\                continue;
        \\            }
        \\            i += 1;
        \\        }
        \\        return sum;
        \\    }
        \\}
    ;

    const rendered = try renderOraMlirForSource(source_text);
    defer testing.allocator.free(rendered);
    try testing.expect(std.mem.indexOf(u8, rendered, "scf.while") == null);
    try testing.expect(std.mem.indexOf(u8, rendered, "ora.while_placeholder") == null);
    try testing.expect(std.mem.indexOf(u8, rendered, "ora.break") == null);
    try testing.expect(std.mem.indexOf(u8, rendered, "ora.continue") == null);
    try testing.expect(std.mem.indexOf(u8, rendered, "ora.synthetic.") != null);
}

test "compiler preserves outer while while unrolling nested small runtime for loops" {
    const source_text =
        \\contract Sample {
        \\    pub fn run() -> u256 {
        \\        let outer = 0;
        \\        let sum = 0;
        \\        while (outer < 3) {
        \\            for (2) |i| {
        \\                sum += i;
        \\            }
        \\            outer += 1;
        \\        }
        \\        return sum;
        \\    }
        \\}
    ;

    const rendered = try renderOraMlirForSource(source_text);
    defer testing.allocator.free(rendered);
    try testing.expect(std.mem.indexOf(u8, rendered, "scf.while") != null);
    try testing.expect(std.mem.indexOf(u8, rendered, "scf.for") == null);
    try testing.expect(std.mem.indexOf(u8, rendered, "ora.synthetic.0.2") != null);
    try testing.expect(std.mem.indexOf(u8, rendered, "ora.synthetic.1.2") != null);
}

test "compiler fully unrolls bounded runtime while loops nested inside small runtime for loops" {
    const source_text =
        \\contract Sample {
        \\    pub fn run() -> u256 {
        \\        let total = 0;
        \\        for (2) |outer| {
        \\            let i = 0;
        \\            while (i < 3) {
        \\                total += outer + i;
        \\                i += 1;
        \\            }
        \\        }
        \\        return total;
        \\    }
        \\}
    ;

    const rendered = try renderOraMlirForSource(source_text);
    defer testing.allocator.free(rendered);
    try testing.expect(std.mem.indexOf(u8, rendered, "scf.for") == null);
    try testing.expect(std.mem.indexOf(u8, rendered, "scf.while") == null);
    try testing.expect(std.mem.indexOf(u8, rendered, "ora.synthetic.0.2") != null);
    try testing.expect(std.mem.indexOf(u8, rendered, "ora.synthetic.1.2") != null);
    try testing.expect(std.mem.indexOf(u8, rendered, "ora.synthetic.0.3") != null);
    try testing.expect(std.mem.indexOf(u8, rendered, "ora.synthetic.1.3") != null);
    try testing.expect(std.mem.indexOf(u8, rendered, "ora.synthetic.2.3") != null);
    try testing.expect(std.mem.indexOf(u8, rendered, "ora.synthetic.0.3\"(\"ora.synthetic.0.2\"") != null);
    try testing.expect(std.mem.indexOf(u8, rendered, "ora.synthetic.2.3\"(\"ora.synthetic.1.2\"") != null);
}

test "compiler fully unrolls bounded runtime while loops nested inside bounded runtime while loops" {
    const source_text =
        \\contract Sample {
        \\    pub fn run() -> u256 {
        \\        let outer = 0;
        \\        let total = 0;
        \\        while (outer < 2) {
        \\            let i = 0;
        \\            while (i < 3) {
        \\                total += outer + i;
        \\                i += 1;
        \\            }
        \\            outer += 1;
        \\        }
        \\        return total;
        \\    }
        \\}
    ;

    const rendered = try renderOraMlirForSource(source_text);
    defer testing.allocator.free(rendered);
    try testing.expect(std.mem.indexOf(u8, rendered, "scf.while") == null);
    try testing.expect(std.mem.indexOf(u8, rendered, "ora.synthetic.0.2") != null);
    try testing.expect(std.mem.indexOf(u8, rendered, "ora.synthetic.1.2") != null);
    try testing.expect(std.mem.indexOf(u8, rendered, "ora.synthetic.0.3") != null);
    try testing.expect(std.mem.indexOf(u8, rendered, "ora.synthetic.1.3") != null);
    try testing.expect(std.mem.indexOf(u8, rendered, "ora.synthetic.2.3") != null);
    try testing.expect(std.mem.indexOf(u8, rendered, "ora.synthetic.0.3\"(\"ora.synthetic.0.2\"") != null);
    try testing.expect(std.mem.indexOf(u8, rendered, "ora.synthetic.2.3\"(\"ora.synthetic.1.2\"") != null);
}

test "compiler fully unrolls nested small runtime for loops" {
    const source_text =
        \\contract Sample {
        \\    pub fn run() -> u256 {
        \\        let total = 0;
        \\        for (2) |outer| {
        \\            for (1..3) |inner, _| {
        \\                total += outer + inner;
        \\            }
        \\        }
        \\        return total;
        \\    }
        \\}
    ;

    const rendered = try renderOraMlirForSource(source_text);
    defer testing.allocator.free(rendered);
    try testing.expect(std.mem.indexOf(u8, rendered, "scf.for") == null);
    try testing.expect(std.mem.indexOf(u8, rendered, "ora.synthetic.0.2") != null);
    try testing.expect(std.mem.indexOf(u8, rendered, "ora.synthetic.1.2") != null);
}

test "compiler keeps nested runtime while structured when nested unroll budget would explode" {
    const source_text =
        \\contract Sample {
        \\    pub fn run() -> u256 {
        \\        let total = 0;
        \\        for (5) |outer| {
        \\            let i = 0;
        \\            while (i < 5) {
        \\                total += outer + i;
        \\                i += 1;
        \\            }
        \\        }
        \\        return total;
        \\    }
        \\}
    ;

    const rendered = try renderOraMlirForSource(source_text);
    defer testing.allocator.free(rendered);
    try testing.expect(std.mem.indexOf(u8, rendered, "ora.synthetic.0.5") != null);
    try testing.expect(std.mem.indexOf(u8, rendered, "ora.synthetic.4.5") != null);
    try testing.expect(std.mem.indexOf(u8, rendered, "scf.while") != null);
}

test "compiler leaves optional partial evaluation runtime when recursion limit is exceeded" {
    const source_text =
        \\comptime fn factorial(n: u256) -> u256 {
        \\    if (n == 0) { return 1; }
        \\    return n * factorial(n - 1);
        \\}
        \\
        \\contract Sample {
        \\    pub fn run(x: u256) -> u256 {
        \\        const ct: u256 = factorial(5);
        \\        return x + ct;
        \\    }
        \\}
    ;

    var compilation = try compileText(source_text);
    defer compilation.deinit();

    const module = compilation.db.sources.module(compilation.root_module_id);
    const ast_file = try compilation.db.astFile(module.file_id);
    const contract = ast_file.item(ast_file.root_items[1]).Contract;
    const function = ast_file.item(contract.members[0]).Function;
    const body = ast_file.body(function.body);
    const decl = ast_file.statement(body.statements[0]).VariableDecl;

    var consteval = try compiler.comptime_eval.constEval(testing.allocator, ast_file, .{
        .config = .{
            .max_recursion_depth = 2,
        },
    });
    defer consteval.deinit();

    try testing.expectEqual(@as(?compiler.sema.ConstValue, null), consteval.values[decl.value.?.index()]);
    try testing.expectEqual(@as(usize, 0), consteval.diagnostics.items.items.len);
}

test "compiler leaves optional partial evaluation runtime when loop iteration limit is exceeded" {
    const source_text =
        \\comptime fn power(base: u256, exp: u256) -> u256 {
        \\    let acc = 1;
        \\    let i = 0;
        \\    while (i < exp) {
        \\        acc *= base;
        \\        i += 1;
        \\    }
        \\    return acc;
        \\}
        \\
        \\contract Sample {
        \\    pub fn run(x: u256) -> u256 {
        \\        const ct: u256 = power(2, 10);
        \\        return x + ct;
        \\    }
        \\}
    ;

    var compilation = try compileText(source_text);
    defer compilation.deinit();

    const module = compilation.db.sources.module(compilation.root_module_id);
    const ast_file = try compilation.db.astFile(module.file_id);
    const contract = ast_file.item(ast_file.root_items[1]).Contract;
    const function = ast_file.item(contract.members[0]).Function;
    const body = ast_file.body(function.body);
    const decl = ast_file.statement(body.statements[0]).VariableDecl;

    var consteval = try compiler.comptime_eval.constEval(testing.allocator, ast_file, .{
        .config = .{
            .max_loop_iterations = 2,
        },
    });
    defer consteval.deinit();

    try testing.expectEqual(@as(?compiler.sema.ConstValue, null), consteval.values[decl.value.?.index()]);
    try testing.expectEqual(@as(usize, 0), consteval.diagnostics.items.items.len);
}

test "compiler leaves optional partial evaluation runtime when step limit is exceeded" {
    const source_text =
        \\comptime fn sum_to(n: u256) -> u256 {
        \\    let total = 0;
        \\    for (n) |i| {
        \\        total += i;
        \\    }
        \\    return total;
        \\}
        \\
        \\contract Sample {
        \\    pub fn run(x: u256) -> u256 {
        \\        const ct: u256 = sum_to(5);
        \\        return x + ct;
        \\    }
        \\}
    ;

    var compilation = try compileText(source_text);
    defer compilation.deinit();

    const module = compilation.db.sources.module(compilation.root_module_id);
    const ast_file = try compilation.db.astFile(module.file_id);
    const contract = ast_file.item(ast_file.root_items[1]).Contract;
    const function = ast_file.item(contract.members[0]).Function;
    const body = ast_file.body(function.body);
    const decl = ast_file.statement(body.statements[0]).VariableDecl;

    var consteval = try compiler.comptime_eval.constEval(testing.allocator, ast_file, .{
        .config = .{
            .max_loop_iterations = 100,
            .max_steps = 8,
        },
    });
    defer consteval.deinit();

    try testing.expectEqual(@as(?compiler.sema.ConstValue, null), consteval.values[decl.value.?.index()]);
    try testing.expectEqual(@as(usize, 0), consteval.diagnostics.items.items.len);
}

test "compiler preserves later early returns after deferred return slots are introduced" {
    const source_text =
        \\pub fn choose(a: u256, b: u256, c: u256) -> u256 {
        \\    if (a == 0) {
        \\        return 0;
        \\    }
        \\    let value = 1;
        \\    if (b > 0) {
        \\        value = 2;
        \\    } else {
        \\        return 5;
        \\    }
        \\    if (c > 0) {
        \\        return 7;
        \\    }
        \\    return value;
        \\}
    ;

    var compilation = try compileText(source_text);
    defer compilation.deinit();

    const hir_result = try compilation.db.lowerToHir(compilation.root_module_id);
    const hir_text = try hir_result.renderText(testing.allocator);
    defer testing.allocator.free(hir_text);

    try testing.expect(std.mem.containsAtLeast(u8, hir_text, 3, "ora.conditional_return"));
    try testing.expect(std.mem.containsAtLeast(u8, hir_text, 1, "memref.store %true"));
    try testing.expect(!std.mem.containsAtLeast(u8, hir_text, 1, "ora.if_placeholder"));
}

test "private call summary handles deferred return fallthrough exactly" {
    const source_text =
        \\contract Test {
        \\    fn choose(a: u256, b: u256, c: u256) -> u256 {
        \\        if (a == 0) {
        \\            return 0;
        \\        }
        \\        let value = 1;
        \\        if (b > 0) {
        \\            value = 2;
        \\        } else {
        \\            return 5;
        \\        }
        \\        if (c > 0) {
        \\            return 7;
        \\        }
        \\        return value;
        \\    }
        \\    pub fn invoke(a: u256, b: u256, c: u256) -> u256 {
        \\        let out = choose(a, b, c);
        \\        return out;
        \\    }
        \\}
    ;

    var result = try verifyTextWithoutDegradation(source_text, "invoke");
    defer result.deinit(testing.allocator);

    try testing.expect(!result.degraded);
}

test "compiler skips unknown carried locals in if lowering" {
    const source_text =
        \\pub fn choose(flag: bool, value: u256) -> u256 {
        \\    let total = 0;
        \\    let bad = value.missing;
        \\    if (flag) {
        \\        bad = value.missing;
        \\        total = 1;
        \\    } else {
        \\        total = 2;
        \\    }
        \\    return total;
        \\}
    ;

    var compilation = try compileText(source_text);
    defer compilation.deinit();

    const hir_result = try compilation.db.lowerToHir(compilation.root_module_id);
    const hir_text = try hir_result.renderText(testing.allocator);
    defer testing.allocator.free(hir_text);

    try testing.expect(hir_result.type_fallback_count > 0);
    try testing.expect(std.mem.containsAtLeast(u8, hir_text, 1, "scf.if"));
    try testing.expect(!std.mem.containsAtLeast(u8, hir_text, 1, "ora.if_placeholder"));
}

test "compiler skips unknown carried locals in switch lowering" {
    const source_text =
        \\pub fn choose(flag: bool, value: u256) -> u256 {
        \\    let total = 0;
        \\    let bad = value.missing;
        \\    switch (flag) {
        \\        true => {
        \\            bad = value.missing;
        \\            total = total + 1;
        \\        },
        \\        else => {
        \\            total = total + 2;
        \\        }
        \\    }
        \\    return total;
        \\}
    ;

    var compilation = try compileText(source_text);
    defer compilation.deinit();

    const hir_result = try compilation.db.lowerToHir(compilation.root_module_id);
    const hir_text = try hir_result.renderText(testing.allocator);
    defer testing.allocator.free(hir_text);

    try testing.expect(hir_result.type_fallback_count > 0);
    try testing.expect(std.mem.containsAtLeast(u8, hir_text, 1, "ora.switch"));
    try testing.expect(!std.mem.containsAtLeast(u8, hir_text, 1, "ora.switch_placeholder"));
}

test "compiler skips unknown carried locals in while lowering" {
    const source_text =
        \\pub fn count(limit: u256) -> u256 {
        \\    let total = 0;
        \\    let bad = total.missing;
        \\    while (total < limit) {
        \\        bad = total.missing;
        \\        total = total + 1;
        \\    }
        \\    return total;
        \\}
    ;

    var compilation = try compileText(source_text);
    defer compilation.deinit();

    const hir_result = try compilation.db.lowerToHir(compilation.root_module_id);
    const hir_text = try hir_result.renderText(testing.allocator);
    defer testing.allocator.free(hir_text);

    try testing.expect(hir_result.type_fallback_count > 0);
    try testing.expect(std.mem.containsAtLeast(u8, hir_text, 1, "scf.while"));
    try testing.expect(!std.mem.containsAtLeast(u8, hir_text, 1, "ora.while_placeholder"));
}

test "compiler allows passing calldata values to function parameters" {
    const source_text =
        \\pub fn helper(x: u256) -> u256 {
        \\    return x;
        \\}
        \\
        \\pub fn example(value: u256) -> u256 {
        \\    return helper(value);
        \\}
    ;

    var compilation = try compileText(source_text);
    defer compilation.deinit();

    const module_typecheck = try compilation.db.moduleTypeCheck(compilation.root_module_id);
    try testing.expect(module_typecheck.diagnostics.isEmpty());
}

test "compiler emits recursive ABI layout attrs for dynamic aggregate public returns" {
    const source_text =
        \\struct Snapshot {
        \\    owner: address;
        \\    note: string;
        \\}
        \\
        \\error Failure();
        \\
        \\contract Vault {
        \\    pub fn quote() -> !(u256, string) | Failure {
        \\        return (1, "ok");
        \\    }
        \\
        \\    pub fn snapshot() -> !Snapshot | Failure {
        \\        return Snapshot { owner: 0x0000000000000000000000000000000000000000, note: "hi" };
        \\    }
        \\}
    ;

    const rendered = try renderHirTextForSource(source_text);
    defer testing.allocator.free(rendered);

    try testing.expect(std.mem.containsAtLeast(u8, rendered, 1, "ora.abi_return_layout"));
    try testing.expect(std.mem.containsAtLeast(u8, rendered, 1, "\"(uint256,string)\""));
    try testing.expect(std.mem.containsAtLeast(u8, rendered, 1, "\"(address,string)\""));
}

test "compiler supports call-style payload error constructors" {
    const source_text =
        \\error Failure(code: u256);
        \\
        \\contract Probe {
        \\    pub fn run(flag: bool) -> !bool | Failure {
        \\        if (flag) {
        \\            return Failure(7);
        \\        }
        \\        return false;
        \\    }
        \\}
    ;

    var compilation = try compileText(source_text);
    defer compilation.deinit();

    const module = compilation.db.sources.module(compilation.root_module_id);
    const ast_file = try compilation.db.astFile(module.file_id);
    const typecheck = try compilation.db.typeCheck(compilation.root_module_id, .{ .item = ast_file.root_items[1] });
    try testing.expectEqual(@as(usize, 0), typecheck.diagnostics.items.items.len);

    const hir_result = try compilation.db.lowerToHir(compilation.root_module_id);
    const hir_text = try hir_result.renderText(testing.allocator);
    defer testing.allocator.free(hir_text);

    try testing.expect(std.mem.containsAtLeast(u8, hir_text, 1, "ora.error.return \"Failure\""));
}

test "compiler wraps returned payload error constructors as error branches" {
    const source_text =
        \\error Failure(code: u256);
        \\
        \\pub fn run() -> !u256 | Failure {
        \\    return Failure(7);
        \\}
    ;

    const hir_text = try renderHirTextForSource(source_text);
    defer testing.allocator.free(hir_text);

    try testing.expect(std.mem.containsAtLeast(u8, hir_text, 1, "ora.error.return \"Failure\""));
    try testing.expect(std.mem.containsAtLeast(u8, hir_text, 1, "ora.error.err"));
    try testing.expect(!std.mem.containsAtLeast(u8, hir_text, 1, "ora.error.ok"));
}

test "ora dialect exposes external call ops through C API" {
    const ctx = mlir.oraContextCreate();
    defer mlir.oraContextDestroy(ctx);
    try testing.expect(mlir.oraDialectRegister(ctx));

    const loc = mlir.oraLocationUnknownGet(ctx);
    const module = mlir.oraModuleCreateEmpty(loc);
    defer mlir.oraModuleDestroy(module);
    const body = mlir.oraModuleGetBody(module);

    const i32_ty = mlir.oraIntegerTypeCreate(ctx, 32);
    const i1_ty = mlir.oraBoolTypeGet(ctx);
    const i256_ty = mlir.oraIntegerTypeGet(ctx, 256, false);
    const addr_ty = mlir.oraAddressTypeGet(ctx);

    const selector_attr = mlir.oraIntegerAttrCreateI64FromType(i32_ty, 0xa9059cbb);
    const arg_type_attrs = [_]mlir.MlirAttribute{
        mlir.oraStringAttrCreate(ctx, mlir.oraStringRefCreateFromCString("address")),
        mlir.oraStringAttrCreate(ctx, mlir.oraStringRefCreateFromCString("uint256")),
    };
    const arg_types_attr = mlir.oraArrayAttrCreate(ctx, arg_type_attrs.len, &arg_type_attrs);
    const return_type_attrs = [_]mlir.MlirAttribute{
        mlir.oraStringAttrCreate(ctx, mlir.oraStringRefCreateFromCString("bool")),
    };
    const return_types_attr = mlir.oraArrayAttrCreate(ctx, return_type_attrs.len, &return_type_attrs);

    const zero_attr = mlir.oraIntegerAttrCreateI64FromType(i256_ty, 0);
    const gas_attr = mlir.oraIntegerAttrCreateI64FromType(i256_ty, 50000);
    const amount_attr = mlir.oraIntegerAttrCreateI64FromType(i256_ty, 1);

    const target_const = mlir.oraArithConstantOpCreate(ctx, loc, addr_ty, zero_attr);
    const gas_const = mlir.oraArithConstantOpCreate(ctx, loc, i256_ty, gas_attr);
    const amount_const = mlir.oraArithConstantOpCreate(ctx, loc, i256_ty, amount_attr);
    mlir.oraBlockAppendOwnedOperation(body, target_const);
    mlir.oraBlockAppendOwnedOperation(body, gas_const);
    mlir.oraBlockAppendOwnedOperation(body, amount_const);

    const target = mlir.oraOperationGetResult(target_const, 0);
    const gas = mlir.oraOperationGetResult(gas_const, 0);
    const amount = mlir.oraOperationGetResult(amount_const, 0);

    const encode_operands = [_]mlir.MlirValue{ target, amount };
    const encode_op = mlir.oraAbiEncodeOpCreate(ctx, loc, selector_attr, arg_types_attr, &encode_operands, encode_operands.len, i256_ty);
    mlir.oraBlockAppendOwnedOperation(body, encode_op);

    const calldata = mlir.oraOperationGetResult(encode_op, 0);
    const external_call_op = mlir.oraExternalCallOpCreate(
        ctx,
        loc,
        mlir.oraStringRefCreateFromCString("call"),
        mlir.oraStringRefCreateFromCString("ERC20"),
        mlir.oraStringRefCreateFromCString("transfer"),
        target,
        gas,
        calldata,
        i1_ty,
        i256_ty,
    );
    mlir.oraBlockAppendOwnedOperation(body, external_call_op);

    const returndata = mlir.oraOperationGetResult(external_call_op, 1);
    const decode_op = mlir.oraAbiDecodeOpCreate(ctx, loc, return_types_attr, returndata, i1_ty);
    mlir.oraBlockAppendOwnedOperation(body, decode_op);

    const module_text_ref = mlir.oraOperationPrintToString(mlir.oraModuleGetOperation(module));
    defer if (module_text_ref.data != null) mlir.oraStringRefFree(module_text_ref);
    const rendered = module_text_ref.data[0..module_text_ref.length];

    try testing.expect(std.mem.containsAtLeast(u8, rendered, 1, "ora.abi_encode"));
    try testing.expect(std.mem.containsAtLeast(u8, rendered, 1, "ora.external_call"));
    try testing.expect(std.mem.containsAtLeast(u8, rendered, 1, "ora.abi_decode"));
    try testing.expect(std.mem.containsAtLeast(u8, rendered, 1, "\"call\""));
    try testing.expect(std.mem.containsAtLeast(u8, rendered, 1, "\"ERC20\""));
    try testing.expect(std.mem.containsAtLeast(u8, rendered, 1, "\"transfer\""));
}

test "complex SMT app probes do not degrade verification encoding" {
    const probes = [_]struct { path: []const u8, function_name: []const u8 }{
        .{ .path = "ora-example/apps/defi_lending_pool.ora", .function_name = "calculate_utilization_rate" },
        .{ .path = "ora-example/apps/defi_lending_pool.ora", .function_name = "get_available_liquidity" },
        .{ .path = "ora-example/apps/erc20_bitfield_comptime_generics.ora", .function_name = "transfer" },
        .{ .path = "ora-example/smt/soundness/conditional_return_split.ora", .function_name = "withdraw" },
        .{ .path = "ora-example/smt/soundness/switch_arm_path_predicates.ora", .function_name = "categorize" },
        .{ .path = "ora-example/smt/soundness/fail_loop_invariant_post.ora", .function_name = "countTo" },
    };

    for (probes) |probe| {
        var result = try verifyExampleWithoutDegradation(probe.path, probe.function_name, false, 5_000);
        defer result.deinit(testing.allocator);
        try testing.expect(!result.degraded);
    }
}

test "stale flagship probes remain non-degraded on current branch" {
    const probes = [_]struct { path: []const u8, function_name: []const u8, timeout_ms: u32 }{
        .{ .path = "ora-example/apps/erc20_verified.ora", .function_name = "transferFrom", .timeout_ms = 5_000 },
        .{ .path = "ora-example/apps/defi_lending_pool.ora", .function_name = "borrow", .timeout_ms = 15_000 },
        .{ .path = "ora-example/apps/defi_lending_pool_fv.ora", .function_name = "borrow", .timeout_ms = 15_000 },
    };

    for (probes) |probe| {
        var result = try verifyExampleWithoutDegradation(probe.path, probe.function_name, false, probe.timeout_ms);
        defer result.deinit(testing.allocator);

        try testing.expect(!result.degraded);
        try testing.expect(result.success);
        try testing.expectEqual(@as(usize, 0), result.errors_len);
    }
}

test "SMT degradation probes fail closed in sequential and parallel verification" {
    const probes = [_]struct {
        path: []const u8,
        function_name: []const u8,
    }{
        .{
            .path = "ora-example/smt/fail-closed/fail_degraded_must_not_succeed.ora",
            .function_name = "incrementLock",
        },
        .{
            .path = "ora-example/smt/fail-closed/fail_loop_result_degraded.ora",
            .function_name = "run",
        },
        .{
            .path = "ora-example/smt/fail-closed/fail_swapped_compare_decrement.ora",
            .function_name = "run",
        },
        .{
            .path = "ora-example/smt/fail-closed/fail_signed_swapped_compare_decrement.ora",
            .function_name = "run",
        },
    };

    for (probes) |probe| {
        var seq_result = try verifyExampleWithoutDegradation(probe.path, probe.function_name, false, 5_000);
        defer seq_result.deinit(testing.allocator);

        var par_result = try verifyExampleWithoutDegradation(probe.path, probe.function_name, true, 5_000);
        defer par_result.deinit(testing.allocator);

        try testing.expect(seq_result.degraded);
        try testing.expect(par_result.degraded);
        try testing.expect(!seq_result.success);
        try testing.expect(!par_result.success);
        try testing.expectEqualStrings("EncodingDegraded", seq_result.error_kinds);
        try testing.expectEqualStrings("EncodingDegraded", par_result.error_kinds);
        try expectVerificationProbeEquivalent(&seq_result, &par_result);
    }
}
