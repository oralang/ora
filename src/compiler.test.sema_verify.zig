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
const containsFieldEffectSlot = h.containsFieldEffectSlot;
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

test "compiler accepts v1 modifies storage paths" {
    const source_text =
        \\contract Vault {
        \\    struct Config {
        \\        owner: address,
        \\    }
        \\
        \\    storage total: u256 = 0;
        \\    storage config: Config;
        \\    storage balances: map<address, u256>;
        \\    storage buckets: map<u256, u256>;
        \\    storage allowances: map<address, map<address, u256>>;
        \\
        \\    pub fn run(owner: address, spender: address, value: u256)
        \\        modifies total
        \\        modifies config.owner
        \\        modifies balances[owner]
        \\        modifies balances[msg.sender]
        \\        modifies balances[tx.origin]
        \\        modifies buckets[42]
        \\        modifies allowances[owner][spender]
        \\        ensures total == value
        \\    {
        \\        total = value;
        \\    }
        \\
        \\    pub fn comma(owner: address)
        \\        modifies balances[owner], total
        \\    {
        \\    }
        \\}
    ;

    var compilation = try compileText(source_text);
    defer compilation.deinit();

    const typecheck = try compilation.db.moduleTypeCheck(compilation.root_module_id);
    try testing.expect(typecheck.diagnostics.isEmpty());
}

test "compiler rejects unsupported modifies map keys fail closed" {
    const source_text =
        \\contract Vault {
        \\    storage balances: map<address, u256>;
        \\    storage users: map<u256, address>;
        \\
        \\    pub fn complex(i: u256)
        \\        modifies balances[users[i]]
        \\    {
        \\    }
        \\}
    ;

    var compilation = try compileText(source_text);
    defer compilation.deinit();

    const typecheck = try compilation.db.moduleTypeCheck(compilation.root_module_id);
    try testing.expect(diagnosticMessagesContain(&typecheck.diagnostics, "`modifies` map keys must be literals, function parameters, `msg.sender`, or `tx.origin` in v1"));
}

test "compiler rejects external storage modifies paths fail closed" {
    const source_text =
        \\contract Vault {
        \\    pub fn external_name(owner: address)
        \\        modifies caller_storage[owner]
        \\    {
        \\    }
        \\
        \\    pub fn callee_name(owner: address)
        \\        modifies callee_storage[owner]
        \\    {
        \\    }
        \\
        \\    pub fn external_storage_name(owner: address)
        \\        modifies external_storage[owner]
        \\    {
        \\    }
        \\}
    ;

    var compilation = try compileText(source_text);
    defer compilation.deinit();

    const typecheck = try compilation.db.moduleTypeCheck(compilation.root_module_id);
    try testing.expectEqual(@as(usize, 3), countDiagnosticMessages(&typecheck.diagnostics, "`modifies` v1 only supports current-contract storage paths such as `total`, `config.owner`, `balances[user]`, or `allowances[owner][spender]`"));
}

test "compiler enforces modifies declarations against storage writes" {
    const source_text =
        \\contract Vault {
        \\    struct Config {
        \\        owner: address,
        \\        admin: address,
        \\    }
        \\
        \\    storage total: u256 = 0;
        \\    storage config: Config;
        \\    storage balances: map<address, u256>;
        \\
        \\    pub fn ok(owner: address, value: u256)
        \\        modifies balances[owner], total, config.owner
        \\    {
        \\        balances[owner] = value;
        \\        total = value;
        \\        config.owner = owner;
        \\    }
        \\
        \\    pub fn wrong_param(owner: address, other: address, value: u256)
        \\        modifies balances[owner]
        \\    {
        \\        balances[other] = value;
        \\    }
        \\
        \\    pub fn wrong_sender_origin(value: u256)
        \\        modifies balances[msg.sender]
        \\    {
        \\        balances[tx.origin] = value;
        \\    }
        \\
        \\    pub fn wrong_field(next_admin: address)
        \\        modifies config.owner
        \\    {
        \\        config.admin = next_admin;
        \\    }
        \\}
    ;

    var compilation = try compileText(source_text);
    defer compilation.deinit();

    const typecheck = try compilation.db.moduleTypeCheck(compilation.root_module_id);
    const ast_file = try compilation.db.astFile(compilation.db.sources.module(compilation.root_module_id).file_id);
    const contract = ast_file.item(ast_file.root_items[0]).Contract;
    var wrong_param: ?compiler.ast.ItemId = null;
    var wrong_sender_origin: ?compiler.ast.ItemId = null;
    var wrong_field: ?compiler.ast.ItemId = null;
    for (contract.members) |member_id| {
        switch (ast_file.item(member_id).*) {
            .Function => |function| {
                if (std.mem.eql(u8, function.name, "wrong_param")) wrong_param = member_id;
                if (std.mem.eql(u8, function.name, "wrong_sender_origin")) wrong_sender_origin = member_id;
                if (std.mem.eql(u8, function.name, "wrong_field")) wrong_field = member_id;
            },
            else => {},
        }
    }
    const other_key = [_]compiler.sema.KeySegment{.{ .parameter = 1 }};
    const tx_origin_key = [_]compiler.sema.KeySegment{.{ .tx_origin = {} }};
    const admin_field = [_][]const u8{"admin"};

    switch (typecheck.itemEffect(wrong_param.?)) {
        .reads_writes => |effect| try testing.expect(containsKeyedEffectSlot(effect.writes, "balances", .storage, &other_key)),
        else => return error.TestUnexpectedResult,
    }
    switch (typecheck.itemEffect(wrong_sender_origin.?)) {
        .reads_writes => |effect| try testing.expect(containsKeyedEffectSlot(effect.writes, "balances", .storage, &tx_origin_key)),
        else => return error.TestUnexpectedResult,
    }
    switch (typecheck.itemEffect(wrong_field.?)) {
        .writes => |effect| try testing.expect(containsFieldEffectSlot(effect.slots, "config", .storage, &admin_field)),
        else => return error.TestUnexpectedResult,
    }

    try testing.expect(diagnosticMessagesContain(&typecheck.diagnostics, "storage write to 'balances[other]' is not covered by this function's `modifies` clause"));
    try testing.expect(diagnosticMessagesContain(&typecheck.diagnostics, "storage write to 'balances[tx.origin]' is not covered by this function's `modifies` clause"));
    try testing.expect(diagnosticMessagesContain(&typecheck.diagnostics, "storage write to 'config.admin' is not covered by this function's `modifies` clause"));
}

test "compiler treats modifies empty form as no storage writes" {
    const source_text =
        \\contract Vault {
        \\    storage total: u256 = 0;
        \\    storage balances: map<address, u256>;
        \\
        \\    pub fn ok(user: address) -> u256
        \\        modifies()
        \\    {
        \\        return balances[user];
        \\    }
        \\
        \\    pub fn bad(value: u256)
        \\        modifies()
        \\    {
        \\        total = value;
        \\    }
        \\}
    ;

    var compilation = try compileText(source_text);
    defer compilation.deinit();

    const typecheck = try compilation.db.moduleTypeCheck(compilation.root_module_id);
    try testing.expect(diagnosticMessagesContain(&typecheck.diagnostics, "storage write to 'total' is not covered by this function's `modifies` clause"));
}

test "compiler rejects modifies empty form combined with non-empty clauses" {
    const source_text =
        \\contract Vault {
        \\    storage total: u256 = 0;
        \\
        \\    pub fn bad(value: u256)
        \\        modifies()
        \\        modifies total
        \\    {
        \\        total = value;
        \\    }
        \\
        \\    pub fn also_bad(value: u256)
        \\        modifies total
        \\        modifies()
        \\    {
        \\        total = value;
        \\    }
        \\}
    ;

    var compilation = try compileText(source_text);
    defer compilation.deinit();

    const typecheck = try compilation.db.moduleTypeCheck(compilation.root_module_id);
    try testing.expectEqual(@as(usize, 2), countDiagnosticMessages(&typecheck.diagnostics, "`modifies()` cannot be combined with non-empty `modifies` clauses"));
}

test "compiler lowers checked modifies declarations into HIR metadata" {
    const source_text =
        \\contract Vault {
        \\    struct Config {
        \\        owner: address,
        \\        admin: address,
        \\    }
        \\
        \\    storage config: Config;
        \\    storage balances: map<address, u256>;
        \\    storage allowances: map<address, map<address, u256>>;
        \\
        \\    fn touch(owner: address, spender: address, value: u256)
        \\        modifies config.owner, balances[owner], allowances[owner][spender]
        \\    {
        \\        config.owner = owner;
        \\        balances[owner] = value;
        \\        allowances[owner][spender] = value;
        \\    }
        \\}
    ;

    const hir_text = try renderHirTextForSource(source_text);
    defer testing.allocator.free(hir_text);

    try testing.expectEqual(@as(usize, 1), std.mem.count(u8, hir_text, "ora.modifies_slots = [\"config.owner\", \"balances[param#0]\", \"allowances[param#0][param#1]\"]"));
}

test "compiler corpus covers modifies sema matrix" {
    const Probe = struct {
        path: []const u8,
        expected_diagnostic: []const u8 = "",
    };

    const probes = [_]Probe{
        .{ .path = "ora-example/corpus/modifies/pass_supported_paths.ora" },
        .{ .path = "ora-example/corpus/modifies/pass_empty_no_writes.ora" },
        .{
            .path = "ora-example/corpus/modifies/fail_unsupported_map_key.ora",
            .expected_diagnostic = "`modifies` map keys must be literals, function parameters, `msg.sender`, or `tx.origin` in v1",
        },
        .{
            .path = "ora-example/corpus/modifies/fail_external_storage_path.ora",
            .expected_diagnostic = "`modifies` v1 only supports current-contract storage paths such as `total`, `config.owner`, `balances[user]`, or `allowances[owner][spender]`",
        },
        .{
            .path = "ora-example/corpus/modifies/fail_write_outside_declared.ora",
            .expected_diagnostic = "is not covered by this function's `modifies` clause",
        },
        .{
            .path = "ora-example/corpus/modifies/fail_empty_with_write.ora",
            .expected_diagnostic = "storage write to 'total' is not covered by this function's `modifies` clause",
        },
        .{
            .path = "ora-example/corpus/modifies/fail_empty_mixed.ora",
            .expected_diagnostic = "`modifies()` cannot be combined with non-empty `modifies` clauses",
        },
    };

    for (probes) |probe| {
        var compilation = try compilePackage(probe.path);
        defer compilation.deinit();

        const typecheck = try compilation.db.moduleTypeCheck(compilation.root_module_id);
        if (probe.expected_diagnostic.len == 0) {
            try testing.expect(typecheck.diagnostics.isEmpty());
        } else {
            try testing.expect(diagnosticMessagesContain(&typecheck.diagnostics, probe.expected_diagnostic));
        }
    }
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
        \\    bounded_max: MaxValue<u256, 200>,
        \\    bounded_range: InRange<u256, 50, 150>,
        \\    rate: BasisPoints<u256>,
        \\    amount: NonZero<u256>,
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
    try testing.expect(std.mem.containsAtLeast(u8, hir_text, 1, "MaxValue"));
    try testing.expect(std.mem.containsAtLeast(u8, hir_text, 1, "InRange"));
    try testing.expect(std.mem.containsAtLeast(u8, hir_text, 1, "BasisPoints"));
    try testing.expect(std.mem.containsAtLeast(u8, hir_text, 1, "NonZero"));
    // Runtime guard messages are pinned to sema.refinements.expectationText.
    try testing.expect(std.mem.containsAtLeast(u8, hir_text, 1, "expected MinValue value >= 100"));
    try testing.expect(std.mem.containsAtLeast(u8, hir_text, 1, "expected MaxValue value <= 200"));
    try testing.expect(std.mem.containsAtLeast(u8, hir_text, 1, "expected InRange value between 50 and 150"));
    try testing.expect(std.mem.containsAtLeast(u8, hir_text, 1, "expected BasisPoints value between 0 and 10000"));
    try testing.expect(std.mem.containsAtLeast(u8, hir_text, 1, "expected NonZero value != 0"));
    try testing.expect(std.mem.containsAtLeast(u8, hir_text, 1, "expected NonZeroAddress value != 0"));
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

test "compiler rejects base integer flow into exact and scaled refinements" {
    const exact_source =
        \\pub fn exactFromBase(x: u256) {
        \\    let exact_x: Exact<u256> = x;
        \\}
    ;

    var exact_compilation = try compileText(exact_source);
    defer exact_compilation.deinit();

    const exact_root = exact_compilation.db.sources.module(exact_compilation.root_module_id);
    const exact_ast = try exact_compilation.db.astFile(exact_root.file_id);
    const exact_typecheck = try exact_compilation.db.typeCheck(exact_compilation.root_module_id, .{ .item = exact_ast.root_items[0] });
    try testing.expect(diagnosticMessagesContain(&exact_typecheck.diagnostics, "declaration expects type 'Exact"));

    const scaled_source =
        \\pub fn scaledFromBase(x: u256) {
        \\    let scaled_x: Scaled<u256, 18> = x;
        \\}
    ;

    var scaled_compilation = try compileText(scaled_source);
    defer scaled_compilation.deinit();

    const scaled_root = scaled_compilation.db.sources.module(scaled_compilation.root_module_id);
    const scaled_ast = try scaled_compilation.db.astFile(scaled_root.file_id);
    const scaled_typecheck = try scaled_compilation.db.typeCheck(scaled_compilation.root_module_id, .{ .item = scaled_ast.root_items[0] });
    try testing.expect(diagnosticMessagesContain(&scaled_typecheck.diagnostics, "declaration expects type 'Scaled"));
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
