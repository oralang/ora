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

test "compiler lowers ensures on implicit void returns" {
    const source_text =
        \\contract Counter {
        \\    storage var total: u256 = 0;
        \\
        \\    pub fn bump() ensures total >= 1 {
        \\        total = 1;
        \\    }
        \\}
    ;

    var compilation = try compileText(source_text);
    defer compilation.deinit();

    const hir_result = try compilation.db.lowerToHir(compilation.root_module_id);
    const hir_text = try hir_result.renderText(testing.allocator);
    defer testing.allocator.free(hir_text);

    const ensures_index = std.mem.indexOf(u8, hir_text, "ora.ensures") orelse return error.TestUnexpectedResult;
    const return_index = std.mem.indexOf(u8, hir_text, "ora.return") orelse return error.TestUnexpectedResult;
    try testing.expect(ensures_index < return_index);
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

test "compiler verification facts include guard clauses" {
    const source_text =
        \\pub fn keep(next: u256) -> bool
        \\    requires next >= 0
        \\    guard next < 10
        \\    ensures result
        \\{
        \\    return true;
        \\}
    ;

    var compilation = try compileText(source_text);
    defer compilation.deinit();

    const ast_file = try compilation.db.astFile(compilation.db.sources.module(compilation.root_module_id).file_id);
    const function_id = ast_file.root_items[0];

    const function_facts = try compilation.db.verificationFacts(compilation.root_module_id, .{ .item = function_id });
    const body_facts = try compilation.db.verificationFacts(compilation.root_module_id, .{ .body = ast_file.item(function_id).Function.body });

    try testing.expectEqual(@as(usize, 3), function_facts.facts.len);
    try testing.expectEqual(compiler.ast.SpecClauseKind.requires, function_facts.facts[0].kind);
    try testing.expectEqual(compiler.ast.SpecClauseKind.guard, function_facts.facts[1].kind);
    try testing.expectEqual(compiler.ast.SpecClauseKind.ensures, function_facts.facts[2].kind);

    try testing.expectEqual(@as(usize, 3), body_facts.facts.len);
    try testing.expectEqual(compiler.ast.SpecClauseKind.requires, body_facts.facts[0].kind);
    try testing.expectEqual(compiler.ast.SpecClauseKind.guard, body_facts.facts[1].kind);
    try testing.expectEqual(compiler.ast.SpecClauseKind.ensures, body_facts.facts[2].kind);
}

test "compiler lowers guard clauses through runtime assert and assume" {
    const source_text =
        \\pub fn safe_add(amount: u256) -> bool
        \\    guard amount < 10;
        \\{
        \\    return true;
        \\}
    ;

    const hir_text = try renderHirTextForSource(source_text);
    defer testing.allocator.free(hir_text);

    try testing.expect(std.mem.containsAtLeast(u8, hir_text, 1, "ora.assert"));
    try testing.expect(std.mem.containsAtLeast(u8, hir_text, 1, "ora.assume"));
    try testing.expect(std.mem.containsAtLeast(u8, hir_text, 1, "\"guard_clause\""));
}

test "compiler does not duplicate guard lowering when function already starts with same filter" {
    const source_text =
        \\pub fn safe_add(amount: u256) -> bool
        \\    guard amount < 10;
        \\{
        \\    if (!(amount < 10)) {
        \\        return false;
        \\    }
        \\    return true;
        \\}
    ;

    const hir_text = try renderHirTextForSource(source_text);
    defer testing.allocator.free(hir_text);

    try testing.expectEqual(@as(usize, 0), std.mem.count(u8, hir_text, "\"guard violation path: amount < 10\""));
    try testing.expectEqual(@as(usize, 0), std.mem.count(u8, hir_text, "\"guard_clause\""));
}

test "compiler removes proven guard clauses after verification" {
    const source_text =
        \\pub fn safe_add(amount: u256) -> bool
        \\    requires amount < 10;
        \\    guard amount < 10;
        \\{
        \\    return true;
        \\}
    ;

    var compilation = try compileText(source_text);
    defer compilation.deinit();

    const hir_result = try compilation.db.lowerToHir(compilation.root_module_id);

    var verifier = try z3_verification.VerificationPass.init(testing.allocator);
    defer verifier.deinit();
    verifier.parallel = false;

    var vr = try verifier.runVerificationPass(hir_result.module.raw_module);
    defer vr.deinit();

    try testing.expect(vr.success);
    try testing.expect(vr.proven_guard_ids.count() > 0);

    const mutable_hir_result = @constCast(hir_result);
    mutable_hir_result.cleanupRefinementGuards(&vr.proven_guard_ids);
    const hir_text = try mutable_hir_result.renderText(testing.allocator);
    defer testing.allocator.free(hir_text);

    try testing.expectEqual(@as(usize, 0), std.mem.count(u8, hir_text, "\"guard_clause\""));
    try testing.expectEqual(@as(usize, 0), std.mem.count(u8, hir_text, "cf.assert"));
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

test "compiler inserts parameter refinement guards in HIR" {
    const source_text =
        \\pub fn guarded(
        \\    bounded: MinValue<u256, 100>,
        \\    target: NonZeroAddress,
        \\) -> u256 {
        \\    return bounded;
        \\}
    ;

    const hir_text = try renderHirTextForSource(source_text);
    defer testing.allocator.free(hir_text);

    try testing.expect(std.mem.containsAtLeast(u8, hir_text, 1, "ora.refinement_guard"));
    try testing.expect(std.mem.containsAtLeast(u8, hir_text, 1, "parameter_refinement"));
    try testing.expect(std.mem.containsAtLeast(u8, hir_text, 1, "MinValue"));
    try testing.expect(std.mem.containsAtLeast(u8, hir_text, 1, "ora.refinement_to_base"));
}

test "compiler inserts refinement flow conversions in HIR" {
    const source_text =
        \\pub fn promote(value: u256) -> MinValue<u256, 100> {
        \\    return @cast(MinValue<u256, 100>, value);
        \\}
    ;

    const hir_text = try renderHirTextForSource(source_text);
    defer testing.allocator.free(hir_text);

    try testing.expect(std.mem.containsAtLeast(u8, hir_text, 1, "ora.base_to_refinement"));
}

test "compiler inserts refinement call conversions in HIR" {
    const source_text =
        \\fn take_base(value: u256) -> u256 {
        \\    return value;
        \\}
        \\
        \\fn take_refined(value: MinValue<u256, 100>) -> MinValue<u256, 100> {
        \\    return value;
        \\}
        \\
        \\pub fn bridge(
        \\    raw: u256,
        \\    bounded: MinValue<u256, 100>,
        \\) -> u256 {
        \\    let left = take_base(bounded);
        \\    let right = take_refined(raw);
        \\    return left;
        \\}
    ;

    const hir_text = try renderHirTextForSource(source_text);
    defer testing.allocator.free(hir_text);

    try testing.expect(std.mem.containsAtLeast(u8, hir_text, 2, "ora.refinement_to_base"));
    try testing.expect(std.mem.containsAtLeast(u8, hir_text, 1, "ora.base_to_refinement"));
}

test "compiler inserts refinement conversions for struct field construction" {
    const source_text =
        \\struct Box {
        \\    value: MinValue<u256, 10>;
        \\}
        \\
        \\pub fn read(raw: u256) -> u256 {
        \\    let box = Box { value: raw };
        \\    return box.value;
        \\}
    ;

    const hir_text = try renderHirTextForSource(source_text);
    defer testing.allocator.free(hir_text);

    try testing.expect(std.mem.containsAtLeast(u8, hir_text, 1, "ora.base_to_refinement"));
    try testing.expect(std.mem.containsAtLeast(u8, hir_text, 1, "ora.struct_instantiate"));
    try testing.expect(std.mem.containsAtLeast(u8, hir_text, 1, "ora.refinement_to_base"));
}

test "compiler accepts guardable refinement flows into ADT variant payloads" {
    const source_text =
        \\enum MaybeAmount {
        \\    None,
        \\    Value(MinValue<u256, 10>),
        \\}
        \\
        \\pub fn read(raw: u256) -> u256 {
        \\    let maybe = MaybeAmount.Value(raw);
        \\    return match (maybe) {
        \\        MaybeAmount.Value(value) => value,
        \\        MaybeAmount.None => 0,
        \\    };
        \\}
    ;

    var compilation = try compileText(source_text);
    defer compilation.deinit();

    const typecheck = try compilation.db.moduleTypeCheck(compilation.root_module_id);
    try testing.expect(typecheck.diagnostics.isEmpty());

    const hir_result = try compilation.db.lowerToHir(compilation.root_module_id);
    const hir_text = try hir_result.renderText(testing.allocator);
    defer testing.allocator.free(hir_text);

    try testing.expect(std.mem.containsAtLeast(u8, hir_text, 1, "ora.base_to_refinement"));
    try testing.expect(std.mem.containsAtLeast(u8, hir_text, 1, "ora.adt.construct"));
    try testing.expect(std.mem.containsAtLeast(u8, hir_text, 1, "ora.refinement_to_base"));
}

test "compiler cleans refinement guards to cf.assert" {
    const source_text =
        \\pub fn guarded(
        \\    bounded: MinValue<u256, 100>,
        \\) -> u256 {
        \\    return bounded;
        \\}
    ;

    var compilation = try compileText(source_text);
    defer compilation.deinit();

    const hir_result = try compilation.db.lowerToHir(compilation.root_module_id);
    const mutable_hir_result = @constCast(hir_result);
    var empty_guards = std.StringHashMap(void).init(testing.allocator);
    defer empty_guards.deinit();

    mutable_hir_result.cleanupRefinementGuards(&empty_guards);
    const hir_text = try mutable_hir_result.renderText(testing.allocator);
    defer testing.allocator.free(hir_text);

    try testing.expect(!std.mem.containsAtLeast(u8, hir_text, 1, "ora.refinement_guard"));
    try testing.expect(std.mem.containsAtLeast(u8, hir_text, 1, "cf.assert"));
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

test "compiler lowers named contract invariants through ora.invariant" {
    const source_text =
        \\contract Counter {
        \\    storage var value: u256;
        \\    invariant value_nonnegative(value >= 0);
        \\}
    ;

    var compilation = try compileText(source_text);
    defer compilation.deinit();

    const hir_result = try compilation.db.lowerToHir(compilation.root_module_id);
    const hir_text = try hir_result.renderText(testing.allocator);
    defer testing.allocator.free(hir_text);

    try testing.expect(std.mem.containsAtLeast(u8, hir_text, 1, "ora.invariant"));
    try testing.expect(std.mem.containsAtLeast(u8, hir_text, 1, "arith.cmpi"));
    try testing.expect(!std.mem.containsAtLeast(u8, hir_text, 1, "\"ora.call\""));
}

test "compiler lowers unsigned requires comparisons with unsigned predicates" {
    const source_text =
        \\comptime const std = @import("std");
        \\
        \\contract Counter {
        \\    storage var value: u256;
        \\
        \\    pub fn bump()
        \\        requires (value < std.constants.U256_MAX)
        \\    {
        \\        value = value + 1;
        \\    }
        \\}
    ;

    var compilation = try compileText(source_text);
    defer compilation.deinit();

    const hir_result = try compilation.db.lowerToHir(compilation.root_module_id);
    const hir_text = try hir_result.renderText(testing.allocator);
    defer testing.allocator.free(hir_text);

    try testing.expect(std.mem.containsAtLeast(u8, hir_text, 1, "arith.cmpi ult"));
    try testing.expect(!std.mem.containsAtLeast(u8, hir_text, 1, "arith.cmpi slt"));
}

test "compiler allows base integer flow into exact and scaled refinements" {
    const source_text =
        \\fn exactLiteral() {
        \\    let total: Exact<u256> = 1000;
        \\}
        \\
        \\fn scaledLiteral() {
        \\    let amount: Scaled<u256, 18> = 1_000_000_000_000_000_000;
        \\}
        \\
        \\fn exactFromBase(x: u256) {
        \\    let exact_x: Exact<u256> = x;
        \\}
    ;

    var compilation = try compileText(source_text);
    defer compilation.deinit();

    const root_module = compilation.db.sources.module(compilation.root_module_id);
    const ast_file = try compilation.db.astFile(root_module.file_id);
    _ = try compilation.db.typeCheck(compilation.root_module_id, .{ .item = ast_file.root_items[0] });
    _ = try compilation.db.typeCheck(compilation.root_module_id, .{ .item = ast_file.root_items[1] });
    _ = try compilation.db.typeCheck(compilation.root_module_id, .{ .item = ast_file.root_items[2] });
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

test "compiler lowers ghost declarations into verification HIR" {
    const source_text =
        \\contract Spec {
        \\    ghost const LIMIT: u256 = 1;
        \\    ghost storage var hidden: u256;
        \\    ghost fn helper() -> u256 {
        \\        assert(true, "fn");
        \\        return LIMIT;
        \\    }
        \\    ghost {
        \\        assert(hidden == LIMIT, "block");
        \\    }
        \\}
    ;

    var compilation = try compileText(source_text);
    defer compilation.deinit();

    const hir_result = try compilation.db.lowerToHir(compilation.root_module_id);
    const hir_text = try hir_result.renderText(testing.allocator);
    defer testing.allocator.free(hir_text);

    try testing.expect(std.mem.containsAtLeast(u8, hir_text, 1, "ora.ghost = true"));
    try testing.expect(std.mem.containsAtLeast(u8, hir_text, 1, "ora.formal = true"));
    try testing.expect(std.mem.containsAtLeast(u8, hir_text, 1, "ghost_function"));
    try testing.expect(std.mem.containsAtLeast(u8, hir_text, 1, "ghost_variable"));
    try testing.expect(std.mem.containsAtLeast(u8, hir_text, 1, "ghost_constant"));
    try testing.expect(std.mem.containsAtLeast(u8, hir_text, 1, "ghost_assertion"));
    try testing.expect(std.mem.containsAtLeast(u8, hir_text, 1, "ora.assert"));
}

test "compiler preserves nested while continue by guarding later statements" {
    const source_text =
        \\pub fn count(limit: u256) -> u256 {
        \\    let sum = 0;
        \\    let i = 0;
        \\    while (i < limit) {
        \\        i = i + 1;
        \\        if (i == 2) {
        \\            continue;
        \\        }
        \\        sum = sum + i;
        \\    }
        \\    return sum;
        \\}
    ;

    var compilation = try compileText(source_text);
    defer compilation.deinit();

    const hir_result = try compilation.db.lowerToHir(compilation.root_module_id);
    const hir_text = try hir_result.renderText(testing.allocator);
    defer testing.allocator.free(hir_text);

    try testing.expect(std.mem.containsAtLeast(u8, hir_text, 2, "memref.alloca() : memref<i1>"));
    try testing.expect(std.mem.containsAtLeast(u8, hir_text, 2, "memref.store %false"));
    try testing.expect(std.mem.containsAtLeast(u8, hir_text, 1, "scf.if %"));
    try testing.expect(std.mem.containsAtLeast(u8, hir_text, 1, "-> (i256, i256)"));
    try testing.expect(!std.mem.containsAtLeast(u8, hir_text, 1, "ora.while_placeholder"));
}

test "compiler lowers while invariants through ora.invariant" {
    const source_text =
        \\pub fn count(limit: u256) -> u256 {
        \\    let sum = 0;
        \\    let i = 0;
        \\    while (i < limit)
        \\        invariant(i <= limit)
        \\        invariant(sum <= limit)
        \\    {
        \\        i = i + 1;
        \\        sum = sum + 1;
        \\    }
        \\    return sum;
        \\}
    ;

    var compilation = try compileText(source_text);
    defer compilation.deinit();

    const hir_result = try compilation.db.lowerToHir(compilation.root_module_id);
    const hir_text = try hir_result.renderText(testing.allocator);
    defer testing.allocator.free(hir_text);

    try testing.expect(std.mem.containsAtLeast(u8, hir_text, 2, "ora.invariant"));
    try testing.expect(std.mem.containsAtLeast(u8, hir_text, 1, "scf.while"));
}

test "compiler accepts identical refinement types across calls and assignments" {
    const source_text =
        \\contract Probe {
        \\    fn same(x: MinValue<u256, 1>) -> bool {
        \\        return ok(x);
        \\    }
        \\    fn ok(x: MinValue<u256, 1>) -> bool {
        \\        return true;
        \\    }
        \\    storage var max_ltv: InRange<u256, 0, 10000> = 7500;
        \\    pub fn set(new_ltv: InRange<u256, 0, 10000>) {
        \\        max_ltv = new_ltv;
        \\    }
        \\}
    ;

    var compilation = try compileText(source_text);
    defer compilation.deinit();

    const ast_file = try compilation.db.astFile(compilation.db.sources.module(compilation.root_module_id).file_id);
    const typecheck = try compilation.db.typeCheck(compilation.root_module_id, .{ .item = ast_file.root_items[0] });
    try testing.expect(typecheck.diagnostics.isEmpty());
}

test "compiler accepts valid refinement subtyping example package" {
    var compilation = try compilePackage("ora-example/type-system/refinements/refinement_subtyping.ora");
    defer compilation.deinit();

    const typecheck = try compilation.db.moduleTypeCheck(compilation.root_module_id);
    try testing.expect(typecheck.diagnostics.isEmpty());
}

test "compiler accepts valid refinement coercion example package" {
    var compilation = try compilePackage("ora-example/type-system/refinements/refinement_coercion.ora");
    defer compilation.deinit();

    const typecheck = try compilation.db.moduleTypeCheck(compilation.root_module_id);
    try testing.expect(typecheck.diagnostics.isEmpty());
}

test "compiler accepts valid refinement in functions example package" {
    var compilation = try compilePackage("ora-example/type-system/refinements/refinement_in_functions.ora");
    defer compilation.deinit();

    const typecheck = try compilation.db.moduleTypeCheck(compilation.root_module_id);
    try testing.expect(typecheck.diagnostics.isEmpty());
}
